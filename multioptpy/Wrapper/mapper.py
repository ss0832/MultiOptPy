"""
mapper.py - Chemical Reaction Network Mapper
============================================

Autonomously maps a chemical reaction network using AutoTSWorkflow
and external optimization utilities.
"""

from __future__ import annotations

import bisect
import traceback
import atexit
import glob
import heapq
import json
import logging
import multiprocessing
import os
import time
import re
import copy
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment


try:
    from multioptpy.Wrapper.autots import AutoTSWorkflow
except ImportError as _autots_import_err:
    logging.getLogger(__name__).warning(
        "[mapper] AutoTSWorkflow not available: %s", _autots_import_err
    )
    AutoTSWorkflow = None  # type: ignore[assignment,misc]

try:
    from multioptpy.Wrapper.optimize_wrapper import OptimizationJob
except ImportError as _opt_import_err:
    logging.getLogger(__name__).warning(
        "[mapper] OptimizationJob not available: %s", _opt_import_err
    )
    OptimizationJob = None # type: ignore[assignment,misc]

try:
    from multioptpy.Parameters.covalent_radii import covalent_radii_lib
    from multioptpy.Parameters.unit_values import UnitValueLib
    _BOHR2ANG = UnitValueLib().bohr2angstroms
except ImportError as _covalent_import_err:
    logging.getLogger(__name__).warning(
        "[mapper] covalent_radii_lib not available: %s", _covalent_import_err
    )
    covalent_radii_lib = None
    _BOHR2ANG = 0.529177210903


# ---------------------------------------------------------------------------
# Module-level worker functions for subprocess isolation
# ---------------------------------------------------------------------------

def _autots_worker(config: dict, run_dir: str, workspace: str) -> list[str]:
    import glob as _glob
    import os as _os
    from multioptpy.Wrapper.autots import AutoTSWorkflow as _AutoTSWorkflow

    _os.chdir(run_dir)
    workflow = _AutoTSWorkflow(config=config)
    workflow.run_workflow()

    pattern = _os.path.join(workspace, "**", "*_Step4_Profile")
    return sorted(_glob.glob(pattern, recursive=True))


def _autots_worker_with_queue(
    config: dict,
    run_dir: str,
    workspace: str,
    result_queue: "multiprocessing.Queue[tuple]",
) -> None:
    try:
        profiles = _autots_worker(config, run_dir, workspace)
        result_queue.put(("ok", profiles))
    except Exception as exc:  # noqa: BLE001
        tb_str = traceback.format_exc()
        if len(tb_str) > 1500:
            tb_str = "...(truncated)...\n" + tb_str[-1500:]
        result_queue.put(("err", tb_str))


logger = logging.getLogger(__name__)

# Module-level physical constants
HARTREE_TO_KCALMOL: float = 627.509474
K_B_HARTREE: float = 3.166811563e-6


# ===========================================================================
# Section 1 : XYZ Utilities
# ===========================================================================

_XYZ_PATTERN: re.Pattern = re.compile(
    r"\s*([A-Za-z]+)\s+"
    r"([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s*"
)


def get_pattern_xyz() -> re.Pattern:
    return _XYZ_PATTERN

def parse_xyz(filepath: str) -> tuple[list[str], np.ndarray]:
    with open(filepath, "r", encoding="utf-8") as fh:

        n_atoms = None
        header_line_idx = 0
        for line_idx, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.isdigit():
                n_atoms = int(stripped)
                header_line_idx = line_idx
                break
            else:
                raise ValueError(
                    f"Invalid XYZ format at {filepath}:{line_idx}: "
                    f"Expected atom count, got '{stripped}'"
                )

        if n_atoms is None:
            raise ValueError(f"Empty or invalid XYZ file: {filepath}")

        next(fh, None)

        symbols: list[str] = []
        coords_raw: list[list[float]] = []

        atom_line_start = header_line_idx + 2
        for atom_line_idx, line in enumerate(fh, atom_line_start):
            stripped = line.strip()
            if not stripped:
                continue

            match = _XYZ_PATTERN.match(line)
            if not match:
                raise ValueError(
                    f"Invalid atom data at {filepath}:{atom_line_idx}: '{stripped}'"
                )

            sym, x, y, z = match.groups()
            symbols.append(sym)
            coords_raw.append([float(x), float(y), float(z)])

            if len(symbols) >= n_atoms:
                break

        if len(symbols) < n_atoms:
            raise ValueError(
                f"Unexpected EOF in {filepath}: "
                f"Expected {n_atoms} atoms, but only found {len(symbols)}."
            )

        return symbols, np.array(coords_raw)

def distance_matrix(coords: np.ndarray) -> np.ndarray:
    return cdist(coords, coords)


# ===========================================================================
# Section 2a : StructureChecker
# ===========================================================================


class StructureChecker:
    """Check whether two molecular structures are identical up to rotation and
    atom-index permutation.

    Parameters
    ----------
    rmsd_threshold : float
        Structures with minimum RMSD below this value are declared identical.
    bond_checker : BondTopologyChecker or None
        When provided, atom groupings are refined by bond environment to
        prevent chemically invalid permutations.
    """

    _DEGENERACY_REL_TOL: float = 0.02
    _DEGENERACY_ABS_TOL: float = 1e-4

    _SIGN_FLIP_MATS: tuple[np.ndarray, ...] = (
        np.diag([ 1.0,  1.0,  1.0]),
        np.diag([-1.0, -1.0,  1.0]),
        np.diag([-1.0,  1.0, -1.0]),
        np.diag([ 1.0, -1.0, -1.0]),
    )

    def __init__(
        self,
        rmsd_threshold: float = 0.30,
        bond_checker: BondTopologyChecker | None = None,
    ) -> None:
        self.rmsd_threshold = rmsd_threshold
        self._bond_checker = bond_checker

    def are_similar(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> bool:
        return self.compute_rmsd(sym_a, coords_a, sym_b, coords_b) < self.rmsd_threshold

    def compute_rmsd(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> float:
        if len(sym_a) != len(sym_b) or Counter(sym_a) != Counter(sym_b):
            return float("inf")

        ca = coords_a - coords_a.mean(axis=0)
        cb = coords_b - coords_b.mean(axis=0)

        ca_pca, eigvals_a = self._pca_align(ca)
        cb_pca, eigvals_b = self._pca_align(cb)

        groups_a, groups_b = self._build_groups(sym_a, ca, sym_b, cb)

        min_rmsd = self._try_candidates(
            self._sign_flip_candidates(),
            ca_pca, cb_pca, groups_a, groups_b,
        )
        if min_rmsd < self.rmsd_threshold:
            return min_rmsd

        deg_01, deg_12 = self._degeneracy_flags(eigvals_a, eigvals_b)
        if not deg_01 and not deg_12:
            return min_rmsd

        coarse = self._build_planar_candidates(deg_01, deg_12, n_plane=6, n_sphere=4)
        min_rmsd = min(min_rmsd, self._try_candidates(
            coarse, ca_pca, cb_pca, groups_a, groups_b,
        ))
        if min_rmsd < self.rmsd_threshold:
            return min_rmsd

        if deg_01 and deg_12:
            fine = self._build_planar_candidates(deg_01, deg_12, n_plane=8, n_sphere=6)
            min_rmsd = min(min_rmsd, self._try_candidates(
                fine, ca_pca, cb_pca, groups_a, groups_b,
            ))

        return min_rmsd

    def _build_groups(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> tuple[dict, dict]:
        if self._bond_checker is not None:
            env_a = self._bond_checker.atom_environments(sym_a, coords_a)
            env_b = self._bond_checker.atom_environments(sym_b, coords_b)
            keys_a: list = [(s, e) for s, e in zip(sym_a, env_a)]
            keys_b: list = [(s, e) for s, e in zip(sym_b, env_b)]
        else:
            keys_a = list(sym_a)
            keys_b = list(sym_b)

        def _groups(keys: list) -> dict:
            mapping: dict = {}
            for i, k in enumerate(keys):
                mapping.setdefault(k, []).append(i)
            return {k: np.array(v, dtype=np.intp) for k, v in mapping.items()}

        return _groups(keys_a), _groups(keys_b)

    def _try_candidates(
        self,
        candidates: list[np.ndarray],
        ca: np.ndarray,
        cb: np.ndarray,
        groups_a: dict,
        groups_b: dict,
    ) -> float:
        min_rmsd = float("inf")
        for R in candidates:
            cb_rot = cb @ R.T
            perm = self._optimal_mapping(ca, cb_rot, groups_a, groups_b)
            if perm is None:
                continue
            rmsd = self._kabsch_rmsd(ca, cb_rot[perm])
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                if min_rmsd < self.rmsd_threshold:
                    return min_rmsd
        return min_rmsd

    @staticmethod
    def _pca_align(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(coords) < 2:
            return coords, np.ones(3)

        cov = (coords.T @ coords) / max(1, len(coords) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        for i in range(2):
            dominant = np.argmax(np.abs(eigvecs[:, i]))
            if eigvecs[dominant, i] < 0:
                eigvecs[:, i] *= -1

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        return coords @ eigvecs, eigvals

    @classmethod
    def _sign_flip_candidates(cls) -> list[np.ndarray]:
        return list(cls._SIGN_FLIP_MATS)

    @classmethod
    def _build_planar_candidates(
        cls,
        deg_01: bool,
        deg_12: bool,
        n_plane: int,
        n_sphere: int,
    ) -> list[np.ndarray]:
        sign_flips = cls._sign_flip_candidates()

        if deg_01 and deg_12:
            extra = cls._so3_grid(n_sphere)
        elif deg_01:
            extra = [cls._Rz(2.0 * np.pi * k / n_plane) for k in range(n_plane)]
        else:
            extra = [cls._Rx(2.0 * np.pi * k / n_plane) for k in range(n_plane)]

        return [S @ R for S in sign_flips for R in extra]

    @classmethod
    def _degeneracy_flags(
        cls,
        eigvals_a: np.ndarray,
        eigvals_b: np.ndarray,
    ) -> tuple[bool, bool]:
        rtol = cls._DEGENERACY_REL_TOL
        atol = cls._DEGENERACY_ABS_TOL

        def _close(ev: np.ndarray, i: int, j: int) -> bool:
            return bool(np.isclose(ev[i], ev[j], rtol=rtol, atol=atol))

        deg_01 = _close(eigvals_a, 0, 1) or _close(eigvals_b, 0, 1)
        deg_12 = _close(eigvals_a, 1, 2) or _close(eigvals_b, 1, 2)
        return deg_01, deg_12

    @staticmethod
    def _so3_grid(n: int) -> list[np.ndarray]:
        k = np.arange(n)
        alphas = 2.0 * np.pi * k / n
        betas  = np.arccos(np.clip(1.0 - 2.0 * (k + 0.5) / n, -1.0, 1.0))
        gammas = 2.0 * np.pi * k / n

        ca, sa = np.cos(alphas), np.sin(alphas)
        cb, sb = np.cos(betas),  np.sin(betas)
        cg, sg = np.cos(gammas), np.sin(gammas)
        z, o = np.zeros(n), np.ones(n)

        Rza = np.stack([np.stack([ca,-sa,z],-1), np.stack([sa,ca,z],-1), np.stack([z,z,o],-1)], 1)
        Ryb = np.stack([np.stack([cb,z,sb],-1), np.stack([z,o,z],-1), np.stack([-sb,z,cb],-1)], 1)
        Rzg = np.stack([np.stack([cg,-sg,z],-1), np.stack([sg,cg,z],-1), np.stack([z,z,o],-1)], 1)

        R_all = Rza[:, None, None] @ Ryb[None, :, None] @ Rzg[None, None, :]
        return list(R_all.reshape(-1, 3, 3))

    @staticmethod
    def _Rx(t: float) -> np.ndarray:
        c, s = np.cos(t), np.sin(t)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    @staticmethod
    def _Rz(t: float) -> np.ndarray:
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _optimal_mapping(
        coords_a: np.ndarray,
        coords_b: np.ndarray,
        groups_a: dict,
        groups_b: dict,
    ) -> np.ndarray | None:
        n = len(coords_a)
        if n != len(coords_b):
            return None

        perm = np.empty(n, dtype=np.intp)
        n_assigned = 0

        for group_key, idx_a in groups_a.items():
            idx_b = groups_b.get(group_key)
            if idx_b is None or len(idx_a) != len(idx_b):
                return None
            cost = cdist(coords_a[idx_a], coords_b[idx_b], metric="sqeuclidean")
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                perm[idx_a[r]] = idx_b[c]
            n_assigned += len(idx_a)

        return perm if n_assigned == n else None

    @staticmethod
    def _kabsch_rmsd(pa: np.ndarray, pb: np.ndarray) -> float:
        U, S, Vt = np.linalg.svd(pb.T @ pa)
        d = np.linalg.det(U) * np.linalg.det(Vt)

        if d < 0:
            return float("inf")

        E0 = np.sum(pa ** 2) + np.sum(pb ** 2)
        rmsd_sq = max(0.0, E0 - 2.0 * (S[0] + S[1] + S[2])) / len(pa)
        return float(np.sqrt(rmsd_sq))


# ===========================================================================
# Section 2b : BondTopologyChecker
# ===========================================================================


class BondTopologyChecker:
    """Detect covalent bond rearrangements by comparing bond-type fingerprints.

    Parameters
    ----------
    covalent_margin : float
        Multiplicative margin applied to the sum of covalent radii.
    """

    def __init__(self, covalent_margin: float = 1.2) -> None:
        self.covalent_margin = covalent_margin

    def fingerprint(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> dict[tuple[str, str], int]:
        bonded_si, bonded_sj = self._bonded_pairs(symbols, coords)
        if len(bonded_si) == 0:
            return {}
        pairs = np.sort(np.column_stack((bonded_si, bonded_sj)), axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        return {tuple(p): int(c) for p, c in zip(unique_pairs, counts)}

    def has_rearrangement(
        self,
        ref_symbols: list[str],
        ref_coords: np.ndarray,
        new_symbols: list[str],
        new_coords: np.ndarray,
    ) -> bool:
        if sorted(ref_symbols) != sorted(new_symbols):
            return True
        return self.fingerprint(ref_symbols, ref_coords) != \
               self.fingerprint(new_symbols, new_coords)

    def atom_environments(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[tuple[str, ...]]:
        n = len(symbols)
        bonded_si, bonded_sj, ii_idx, jj_idx = self._bonded_pairs(
            symbols, coords, return_indices=True
        )

        neighbors: list[list[str]] = [[] for _ in range(n)]
        for i, j in zip(ii_idx, jj_idx):
            neighbors[i].append(symbols[j])
            neighbors[j].append(symbols[i])

        return [tuple(sorted(nb)) for nb in neighbors]

    def _bonded_pairs(
        self,
        symbols: list[str],
        coords: np.ndarray,
        *,
        return_indices: bool = False,
    ):
        n = len(symbols)
        radii_arr = np.array([self._covalent_radius(s) for s in symbols])

        dists = pdist(coords)
        ii, jj = np.triu_indices(n, k=1)
        thresholds = self.covalent_margin * (radii_arr[ii] + radii_arr[jj])
        bonded_mask = dists <= thresholds

        sym_arr = np.array(symbols)
        bonded_ii = ii[bonded_mask]
        bonded_jj = jj[bonded_mask]

        if return_indices:
            return sym_arr[bonded_ii], sym_arr[bonded_jj], bonded_ii, bonded_jj
        return sym_arr[bonded_ii], sym_arr[bonded_jj]

    @staticmethod
    def _covalent_radius(symbol: str) -> float:
        if covalent_radii_lib is not None:
            try:
                return float(covalent_radii_lib(symbol)) * _BOHR2ANG
            except (KeyError, TypeError):
                pass
        return 0.75


# ===========================================================================
# Section 3a : ExplorationQueue
# ===========================================================================

@dataclass
class ExplorationTask:
    node_id: int
    xyz_file: str
    afir_params: list[str]
    priority: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeUpdateData:
    energy: float | None = None
    free_energy: float | None = None


class ExplorationQueue(ABC):
    """Abstract base class for priority-ordered exploration queues."""

    def __init__(self, rng_seed: int = 42) -> None:
        self._tasks: dict[int, ExplorationTask] = {}
        self._task_counter: int = 0
        self._heap: list[tuple[float, int, int]] = []
        self._push_counter: int = 0
        self._submitted: set[tuple] = set()
        self._rng = np.random.default_rng(rng_seed)
        self._in_flight: set[tuple] = set()
        self._current_ref_e: float | None = None

    def push(self, task: ExplorationTask) -> bool:
        key = (task.node_id, tuple(task.afir_params))
        if key in self._submitted or key in self._in_flight:
            return False

        task.priority = self.compute_priority(task)

        self._task_counter += 1
        task_id = self._task_counter
        self._tasks[task_id] = task

        heapq.heappush(self._heap, (-task.priority, self._push_counter, task_id))

        self._push_counter += 1
        self._submitted.add(key)
        return True

    def pop(self) -> ExplorationTask | None:
        n_tasks = len(self._tasks)
        if len(self._heap) > max(100, 2 * n_tasks):
            valid = [e for e in self._heap if e[2] in self._tasks]
            heapq.heapify(valid)
            logger.debug("ExplorationQueue: heap purged — %d stale entries removed.",
                         len(self._heap) - len(valid))
            self._heap = valid

        while self._heap:
            _, _, task_id = heapq.heappop(self._heap)
            task = self._tasks.pop(task_id, None)
            if task is None:
                continue

            key = (task.node_id, tuple(task.afir_params))
            self._submitted.discard(key)
            self._in_flight.add(key)

            if self._current_ref_e is not None:
                node_g = task.metadata.get("source_node_free_energy")
                node_e = task.metadata.get("source_node_energy")
                eff_e  = node_g if node_g is not None else node_e
                if eff_e is not None:
                    task.metadata["delta_E_hartree"] = eff_e - self._current_ref_e
                task.priority = self.compute_priority(task)

            return task
        return None

    def is_submitted(self, key: tuple) -> bool:
        return key in self._submitted or key in self._in_flight

    def release(self, key: tuple) -> None:
        self._in_flight.discard(key)

    def should_add(self, node: "EQNode", reference_energy: float, **kwargs) -> bool:
        eff_e = getattr(node, "free_energy", None)
        if eff_e is None:
            eff_e = node.energy
        dummy_task = ExplorationTask(
            node_id=node.node_id,
            xyz_file=node.xyz_file,
            afir_params=[],
            metadata={
                "delta_E_hartree": (
                    eff_e - reference_energy if eff_e is not None else 0.0
                ),
                "source_node_energy":      node.energy,
                "source_node_free_energy": getattr(node, "free_energy", None),
            },
        )
        p = self.compute_priority(dummy_task)
        return bool(self._rng.random() < p)

    def refresh_priorities(self, ref_e: float | None) -> None:
        if ref_e is not None:
            self._current_ref_e = ref_e

    def export_queue_status(self) -> list[dict]:
        return [
            {
                "node_id":     t.node_id,
                "priority":    t.priority,
                "afir_params": t.afir_params,
            }
            for t in self._tasks.values()
        ]

    def __len__(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def compute_priority(self, task: ExplorationTask) -> float:
        ...


class BoltzmannQueue(ExplorationQueue):
    """Probabilistic exploration queue based on the Boltzmann distribution."""

    def __init__(self, temperature_K: float = 300.0, rng_seed: int = 42) -> None:
        super().__init__(rng_seed=rng_seed)
        self.temperature_K = temperature_K

    def compute_priority(self, task: ExplorationTask) -> float:
        delta_e: float = task.metadata.get("delta_E_hartree", 0.0)
        if delta_e <= 0.0:
            return 1.0
        return min(1.0, float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K))))


# ===========================================================================
# Section 3b : ExploredPairsLog
# ===========================================================================

class ExploredPairsLog:
    """Persistent log of explored (EQ node, atom pair, gamma sign) combinations.

    File format (one record per line)::
        EQ{node_id:06d} {atom_i_1based} {atom_j_1based} {gamma_sign}
    """

    def __init__(self, filepath: str, flush_interval: int = 1) -> None:
        if flush_interval < 1:
            logger.warning("ExploredPairsLog: flush_interval must be >= 1; resetting to 1.")
            flush_interval = 1
        self._filepath       = filepath
        self._flush_interval = flush_interval
        self._write_count    = 0
        self._explored: set[tuple[int, int, int, str]] = set()
        self._explored_node_ids: set[int] = set()
        self._load()
        self._fh = open(self._filepath, "a", encoding="utf-8")  # noqa: SIM115
        atexit.register(self.close)

    def _load(self) -> None:
        if not os.path.isfile(self._filepath):
            return
        with open(self._filepath, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    node_id    = int(parts[0][2:])
                    atom_i     = int(parts[1])
                    atom_j     = int(parts[2])
                    gamma_sign = parts[3]
                    if gamma_sign not in ("+", "-"):
                        continue
                    self._explored.add((node_id, atom_i, atom_j, gamma_sign))
                    self._explored_node_ids.add(node_id)
                except (ValueError, IndexError):
                    logger.warning(
                        "ExploredPairsLog._load: malformed record at %s:%d (%r) — skipped.",
                        self._filepath, lineno, line,
                    )
        logger.info("ExploredPairsLog: loaded %d records from %s",
                    len(self._explored), self._filepath)

    def has(self, node_id: int, atom_i: int, atom_j: int, gamma_sign: str) -> bool:
        return (node_id, atom_i, atom_j, gamma_sign) in self._explored

    def has_any_for_node(self, node_id: int) -> bool:
        return node_id in self._explored_node_ids

    def contains(self, key: tuple[int, int, int, str]) -> bool:
        return key in self._explored

    def record(self, node_id: int, atom_i: int, atom_j: int, gamma_sign: str) -> None:
        key = (node_id, atom_i, atom_j, gamma_sign)
        if key in self._explored:
            return
        self._explored.add(key)
        self._explored_node_ids.add(node_id)
        self._fh.write(f"EQ{node_id:06d} {atom_i} {atom_j} {gamma_sign}\n")
        self._write_count += 1
        if self._write_count % self._flush_interval == 0:
            self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "ExploredPairsLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._explored)


# ===========================================================================
# Section 4 : PerturbationGenerator
# ===========================================================================

class PerturbationGenerator:
    def __init__(
        self,
        afir_gamma_kJmol: float = 100.0,
        max_pairs: int = 5,
        dist_lower_ang: float = 1.5,
        dist_upper_ang: float = 5.0,
        rng_seed: int = 0,
        covalent_margin: float = 1.2,
        active_atoms: list[int] | None = None,
        include_negative_gamma: bool = False,
    ) -> None:
        self.afir_gamma_kJmol       = afir_gamma_kJmol
        self.max_pairs              = max_pairs
        self.dist_lower_ang         = dist_lower_ang
        self.dist_upper_ang         = dist_upper_ang
        self.covalent_margin        = covalent_margin
        self.active_atoms           = set(active_atoms) if active_atoms is not None else None
        self.include_negative_gamma = include_negative_gamma
        self._rng = np.random.default_rng(rng_seed)

    def _build_candidates(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[tuple[int, int]]:
        n = len(symbols)
        if n < 2:
            return []

        if self.active_atoms is not None:
            atom_indices = np.array(
                [i for i in range(n) if (i + 1) in self.active_atoms], dtype=np.intp
            )
        else:
            atom_indices = np.arange(n, dtype=np.intp)

        if len(atom_indices) < 2:
            return []

        sub_coords  = coords[atom_indices]
        sub_symbols = [symbols[i] for i in atom_indices]
        m = len(atom_indices)

        dmat  = distance_matrix(sub_coords)
        ii, jj = np.triu_indices(m, k=1)
        dists  = dmat[ii, jj]

        dist_mask = (dists >= self.dist_lower_ang) & (dists <= self.dist_upper_ang)

        if covalent_radii_lib is not None:
            radii = np.empty(m, dtype=np.float64)
            for k, sym in enumerate(sub_symbols):
                try:
                    radii[k] = covalent_radii_lib(sym) * _BOHR2ANG
                except (KeyError, TypeError):
                    radii[k] = 0.75
            cov_thresh = self.covalent_margin * (radii[ii] + radii[jj])
            cov_mask = dists > cov_thresh
        else:
            cov_mask = dists > (self.covalent_margin * 1.5)

        keep    = dist_mask & cov_mask
        orig_ii = atom_indices[ii[keep]]
        orig_jj = atom_indices[jj[keep]]
        return list(zip(orig_ii.tolist(), orig_jj.tolist()))

    def get_candidate_pairs(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[tuple[int, int]]:
        return self._build_candidates(symbols, coords)

    def generate_afir_perturbations(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[list[str]]:
        candidates = self._build_candidates(symbols, coords)
        if not candidates:
            return []

        n_sel  = min(self.max_pairs, len(candidates))
        chosen = self._rng.choice(len(candidates), size=n_sel, replace=False)

        pos_gamma_str = f"{self.afir_gamma_kJmol:.6g}"
        neg_gamma_str = f"{-self.afir_gamma_kJmol:.6g}"

        result: list[list[str]] = []
        for idx in chosen:
            i, j = candidates[int(idx)]
            i1, j1 = str(i + 1), str(j + 1)
            result.append([pos_gamma_str, i1, j1])
            if self.include_negative_gamma:
                result.append([neg_gamma_str, i1, j1])

        return result


# ===========================================================================
# Section 5 : Graph data model
# ===========================================================================

@dataclass
class EQNode:
    node_id: int
    xyz_file: str
    energy: float | None
    symbols: list[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    source_run_dir: str = ""
    extra: dict = field(default_factory=dict)
    free_energy: float | None = None

    @property
    def has_real_energy(self) -> bool:
        return self.energy is not None

    @property
    def effective_energy(self) -> float | None:
        return self.free_energy if self.free_energy is not None else self.energy

    def to_dict(self) -> dict:
        data: dict = {
            "node_id":             self.node_id,
            "xyz_file":            self.xyz_file,
            "energy_hartree":      self.energy,
            "free_energy_hartree": self.free_energy,
            "source_run_dir":      self.source_run_dir,
        }
        safe_extra = {}
        for k, v in self.extra.items():
            try:
                json.dumps(v)
                safe_extra[k] = v
            except (TypeError, ValueError):
                safe_extra[k] = str(v)
        data["extra"] = safe_extra
        return data

@dataclass
class TSEdge:
    edge_id: int
    node_id_1: int
    node_id_2: int
    ts_xyz_file: str | None
    ts_energy: float | None
    barrier_fwd: float | None = None
    barrier_rev: float | None = None
    source_run_dir: str = ""
    symbols: list[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    extra: dict = field(default_factory=dict)
    ts_free_energy: float | None = None
    barrier_fwd_free: float | None = None
    barrier_rev_free: float | None = None

    @property
    def has_coords(self) -> bool:
        return self.coords.size > 0

    def to_dict(self) -> dict:
        data: dict = {
            "edge_id":                self.edge_id,
            "node_id_1":              self.node_id_1,
            "node_id_2":              self.node_id_2,
            "ts_xyz_file":            self.ts_xyz_file,
            "ts_energy_hartree":      self.ts_energy,
            "barrier_fwd_kcal":       self.barrier_fwd,
            "barrier_rev_kcal":       self.barrier_rev,
            "ts_free_energy_hartree": self.ts_free_energy,
            "barrier_fwd_free_kcal":  self.barrier_fwd_free,
            "barrier_rev_free_kcal":  self.barrier_rev_free,
            "source_run_dir":         self.source_run_dir,
        }
        for k, v in self.extra.items():
            try:
                json.dumps(v)
                data[k] = v
            except (TypeError, ValueError):
                data[k] = str(v)
        return data

_UNSET = object()


class NetworkGraph:
    def __init__(self) -> None:
        self._nodes: dict[int, EQNode] = {}
        self._edges: dict[int, TSEdge] = {}
        self._node_counter: int = 0
        self._edge_counter: int = 0
        self._ref_energy_cache: float | None = _UNSET  # type: ignore[assignment]
        self.last_iteration: int = 0

        self._node_energy_index: list[tuple[float, int]] = []
        self._node_none_ids:     list[int]               = []
        self._edge_energy_index: list[tuple[float, int]] = []
        self._edge_none_ids:     list[int]               = []
        self._index_dirty: bool = False

    def add_node(self, node: EQNode) -> None:
        self._nodes[node.node_id] = node
        self._ref_energy_cache = _UNSET  # type: ignore[assignment]
        if node.energy is not None:
            bisect.insort(self._node_energy_index, (node.energy, node.node_id))
        else:
            self._node_none_ids.append(node.node_id)

    def get_node(self, node_id: int) -> EQNode | None:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[EQNode]:
        return list(self._nodes.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def next_node_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def add_edge(self, edge: TSEdge) -> None:
        self._edges[edge.edge_id] = edge
        if edge.ts_energy is not None:
            bisect.insort(self._edge_energy_index, (edge.ts_energy, edge.edge_id))
        else:
            self._edge_none_ids.append(edge.edge_id)

    def all_edges(self) -> list[TSEdge]:
        return list(self._edges.values())

    def next_edge_id(self) -> int:
        eid = self._edge_counter
        self._edge_counter += 1
        return eid

    def invalidate_energy_cache(self) -> None:
        self._ref_energy_cache = _UNSET  # type: ignore[assignment]
        self._index_dirty = True

    def reference_energy(self) -> float | None:
        if self._ref_energy_cache is not _UNSET:
            return self._ref_energy_cache  # type: ignore[return-value]
        free_energies = [n.free_energy for n in self._nodes.values()
                         if n.free_energy is not None]
        if free_energies:
            result: float | None = min(free_energies)
        else:
            real_energies = [n.energy for n in self._nodes.values() if n.has_real_energy]
            result = min(real_energies) if real_energies else None
        self._ref_energy_cache = result  # type: ignore[assignment]
        return result

    def save(self, filepath: str) -> None:
        data = {
            "nodes":    [n.to_dict() for n in self._nodes.values()],
            "edges":    [e.to_dict() for e in self._edges.values()],
            "metadata": {
                "n_nodes": len(self._nodes),
                "n_edges": len(self._edges),
                "last_iteration": self.last_iteration,
            },
        }
        dir_name = os.path.dirname(os.path.abspath(filepath))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        new_nodes: dict[int, EQNode] = {}
        new_node_energy_index: list[tuple[float, int]] = []
        new_node_none_ids:     list[int]               = []

        for nd in data.get("nodes", []):
            node = EQNode(
                node_id=nd["node_id"],
                xyz_file=nd["xyz_file"],
                energy=nd["energy_hartree"],
                symbols=[],
                source_run_dir=nd.get("source_run_dir", ""),
                free_energy=nd.get("free_energy_hartree"),
                extra=nd.get("extra", {}),
            )
            if os.path.isfile(node.xyz_file):
                try:
                    node.symbols, node.coords = parse_xyz(node.xyz_file)
                except Exception as exc:
                    logger.warning("Could not re-read %s: %s", node.xyz_file, exc)
            new_nodes[node.node_id] = node
            if node.energy is not None:
                bisect.insort(new_node_energy_index, (node.energy, node.node_id))
            else:
                new_node_none_ids.append(node.node_id)

        new_edges: dict[int, TSEdge] = {}
        new_edge_energy_index: list[tuple[float, int]] = []
        new_edge_none_ids:     list[int]               = []

        for ed in data.get("edges", []):
            edge = TSEdge(
                edge_id=ed["edge_id"],
                node_id_1=ed["node_id_1"],
                node_id_2=ed["node_id_2"],
                ts_xyz_file=ed.get("ts_xyz_file"),
                ts_energy=ed.get("ts_energy_hartree"),
                barrier_fwd=ed.get("barrier_fwd_kcal"),
                barrier_rev=ed.get("barrier_rev_kcal"),
                source_run_dir=ed.get("source_run_dir", ""),
                ts_free_energy=ed.get("ts_free_energy_hartree"),
                barrier_fwd_free=ed.get("barrier_fwd_free_kcal"),
                barrier_rev_free=ed.get("barrier_rev_free_kcal"),
            )
            if edge.ts_xyz_file and os.path.isfile(edge.ts_xyz_file):
                try:
                    edge.symbols, edge.coords = parse_xyz(edge.ts_xyz_file)
                except Exception as exc:
                    logger.warning("NetworkGraph.load: cannot parse TS XYZ %s: %s",
                                   edge.ts_xyz_file, exc)
            elif edge.ts_xyz_file:
                logger.warning("NetworkGraph.load: TS XYZ not found: %s", edge.ts_xyz_file)
            new_edges[edge.edge_id] = edge
            if edge.ts_energy is not None:
                bisect.insort(new_edge_energy_index, (edge.ts_energy, edge.edge_id))
            else:
                new_edge_none_ids.append(edge.edge_id)

        self._nodes               = new_nodes
        self._edges               = new_edges
        self._node_counter        = (max(new_nodes.keys()) + 1) if new_nodes else 0
        self._edge_counter        = (max(new_edges.keys()) + 1) if new_edges else 0
        self._node_energy_index   = new_node_energy_index
        self._node_none_ids       = new_node_none_ids
        self._edge_energy_index   = new_edge_energy_index
        self._edge_none_ids       = new_edge_none_ids
        self._index_dirty         = False
        self._ref_energy_cache    = _UNSET  # type: ignore[assignment]
        self.last_iteration       = data.get("metadata", {}).get("last_iteration", 0)

    def _rebuild_indexes(self) -> None:
        self._node_energy_index = sorted([
            (n.energy, n.node_id) for n in self._nodes.values() if n.energy is not None
        ])
        self._node_none_ids = [
            n.node_id for n in self._nodes.values() if n.energy is None
        ]
        self._edge_energy_index = sorted([
            (e.ts_energy, e.edge_id) for e in self._edges.values() if e.ts_energy is not None
        ])
        self._edge_none_ids = [
            e.edge_id for e in self._edges.values() if e.ts_energy is None
        ]
        self._index_dirty = False

    def nodes_in_energy_window(self, center: float, tol: float) -> list[EQNode]:
        if self._index_dirty:
            self._rebuild_indexes()
        lo = bisect.bisect_left(self._node_energy_index,  (center - tol, -1))
        hi = bisect.bisect_right(self._node_energy_index, (center + tol, float("inf")))
        result = [self._nodes[nid] for _, nid in self._node_energy_index[lo:hi]
                  if nid in self._nodes]
        for nid in self._node_none_ids:
            node = self._nodes.get(nid)
            if node is not None:
                result.append(node)
        return result

    def edges_in_energy_window(self, center: float, tol: float) -> list[TSEdge]:
        if self._index_dirty:
            self._rebuild_indexes()
        lo = bisect.bisect_left(self._edge_energy_index,  (center - tol, -1))
        hi = bisect.bisect_right(self._edge_energy_index, (center + tol, float("inf")))
        result = [self._edges[eid] for _, eid in self._edge_energy_index[lo:hi]
                  if eid in self._edges]
        for eid in self._edge_none_ids:
            edge = self._edges.get(eid)
            if edge is not None:
                result.append(edge)
        return result

    def summary(self) -> str:
        lines = [f"[NetworkGraph]  nodes={len(self._nodes)}  edges={len(self._edges)}"]
        ref = self.reference_energy()

        for node in sorted(
            self._nodes.values(),
            key=lambda n: (
                n.effective_energy is None,
                n.effective_energy if n.effective_energy is not None else 0.0,
            ),
        ):
            eff   = node.effective_energy
            label = "G" if node.free_energy is not None else "E"
            if eff is not None and ref is not None:
                rel   = (eff - ref) * HARTREE_TO_KCALMOL
                e_str = f"{eff:+.8f} Ha  ({rel:+.2f} kcal/mol) [{label}]"
            elif eff is not None:
                e_str = f"{eff:+.8f} Ha  [{label}]"
            else:
                e_str = "energy unknown"
            lines.append(f"  EQ{node.node_id:06d}: {e_str}  [{node.xyz_file}]")

        for edge in sorted(self._edges.values(), key=lambda e: e.edge_id):
            fwd      = f"{edge.barrier_fwd:.2f}"      if edge.barrier_fwd      is not None else "N/A"
            rev      = f"{edge.barrier_rev:.2f}"      if edge.barrier_rev      is not None else "N/A"
            ts_e     = f"{edge.ts_energy:.8f} Ha"     if edge.ts_energy        is not None else "N/A"
            fwd_free = f"{edge.barrier_fwd_free:.2f}" if edge.barrier_fwd_free is not None else "N/A"
            rev_free = f"{edge.barrier_rev_free:.2f}" if edge.barrier_rev_free is not None else "N/A"
            lines.append(
                f"  TS{edge.edge_id:06d}: "
                f"EQ{edge.node_id_1} -- EQ{edge.node_id_2}  "
                f"E(TS)={ts_e}  "
                f"Ea(fwd)={fwd} kcal/mol  Ea(rev)={rev} kcal/mol  "
                f"ΔG_ddag(fwd)={fwd_free} kcal/mol  ΔG_ddag(rev)={rev_free} kcal/mol"
            )
        return "\n".join(lines)


# ===========================================================================
# Section 6 : ProfileParser
# ===========================================================================

class ProfileParser:
    def parse(self, profile_dir: str) -> dict | None:
        ep1        = os.path.join(profile_dir, "endpoint_1_opt.xyz")
        ep2        = os.path.join(profile_dir, "endpoint_2_opt.xyz")
        ts_matches = sorted(glob.glob(os.path.join(profile_dir, "*_ts_final.xyz")))
        txt_path   = os.path.join(profile_dir, "energy_profile.txt")

        if not os.path.isfile(ep1) or not os.path.isfile(ep2):
            logger.info("ProfileParser: endpoint file(s) missing in %s.", profile_dir)
            return None
        if not ts_matches:
            logger.info("ProfileParser: no *_ts_final.xyz found in %s.", profile_dir)
            return None

        ts_file = ts_matches[0]
        if len(ts_matches) > 1:
            logger.warning("ProfileParser: %d TS files in %s — using first.",
                           len(ts_matches), profile_dir)

        energies      = self._parse_energy_txt(txt_path)
        free_energies = self._parse_free_energy_txt(txt_path)

        ts_e  = energies.get("TS");       ep1_e = energies.get("Endpoint_1");    ep2_e = energies.get("Endpoint_2")
        ts_g  = free_energies.get("TS");  ep1_g = free_energies.get("Endpoint_1"); ep2_g = free_energies.get("Endpoint_2")

        def _barrier(e_eq: float | None, e_ts: float | None) -> float | None:
            if e_eq is None or e_ts is None:
                return None
            return (e_ts - e_eq) * HARTREE_TO_KCALMOL

        return {
            "ts_xyz_file":            ts_file,
            "ts_energy":              ts_e,
            "endpoint_1_xyz":         ep1,
            "endpoint_2_xyz":         ep2,
            "endpoint_1_energy":      ep1_e,
            "endpoint_2_energy":      ep2_e,
            "barrier_fwd":            _barrier(ep1_e, ts_e),
            "barrier_rev":            _barrier(ep2_e, ts_e),
            "ts_free_energy":         ts_g,
            "endpoint_1_free_energy": ep1_g,
            "endpoint_2_free_energy": ep2_g,
            "barrier_fwd_free":       _barrier(ep1_g, ts_g),
            "barrier_rev_free":       _barrier(ep2_g, ts_g),
        }

    @staticmethod
    def _parse_energy_txt(txt_path: str) -> dict:
        result: dict[str, float | None] = {"TS": None, "Endpoint_1": None, "Endpoint_2": None}
        if not os.path.isfile(txt_path):
            return result
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if "FREE ENERGY SECTION" in stripped:
                    break
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) < 3 or parts[0] not in result:
                    continue
                try:
                    result[parts[0]] = float(parts[2])
                except ValueError:
                    logger.warning("_parse_energy_txt: cannot parse %r in %s", parts[2], txt_path)
        return result

    @staticmethod
    def _parse_free_energy_txt(txt_path: str) -> dict:
        result: dict[str, float | None] = {"TS": None, "Endpoint_1": None, "Endpoint_2": None}
        if not os.path.isfile(txt_path):
            return result
        in_section = False
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if "FREE ENERGY SECTION" in stripped:
                    in_section = True; continue
                if not in_section or not stripped or stripped.startswith("#"):
                    continue
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) < 2 or parts[0] not in result:
                    continue
                raw = parts[1].strip()
                if raw.upper() == "N/A" or not raw:
                    result[parts[0]] = None
                else:
                    try:
                        result[parts[0]] = float(raw)
                    except ValueError:
                        logger.warning("_parse_free_energy_txt: cannot parse %r in %s",
                                       raw, txt_path)
                        result[parts[0]] = None
        return result


# ===========================================================================
# Section 7 : ReactionNetworkMapper
# ===========================================================================

class ReactionNetworkMapper:
    # ------------------------------------------------------------------ #
    #  Config validation                                                   #
    # ------------------------------------------------------------------ #

    _KNOWN_MAPPER_SETTINGS_KEYS: frozenset[str] = frozenset({
        "temperature_K", "rmsd_threshold", "max_iterations",
        "afir_gamma_kJmol", "max_pairs", "dist_lower_ang", "dist_upper_ang",
        "output_dir", "rng_seed", "active_atoms", "include_negative_gamma",
        "excluded_node_ids", "exclude_bond_rearrangement", "use_rcmc",
        "rcmc_temperature_K", "rcmc_reaction_time_s", "rcmc_start_node_id",
        "n_parallel", "worker_timeout_s", "energy_tolerance_kcalmol",
    })

    # (key, type_or_types, condition)
    _REQUIRED_SETTINGS: list[tuple] = [
        ("temperature_K",    (int, float), lambda v: v > 0),
        ("rmsd_threshold",   (int, float), lambda v: v > 0),
        ("max_iterations",   int,          lambda v: v >= 0),
        ("afir_gamma_kJmol", (int, float), lambda v: v > 0),
        ("max_pairs",        int,          lambda v: v > 0),
        ("dist_lower_ang",   (int, float), lambda v: v >= 0),
        ("dist_upper_ang",   (int, float), lambda v: v > 0),
        ("rng_seed",         int,          lambda v: v >= 0),
    ]

    @classmethod
    def _validate_mapper_settings(cls, config: dict) -> None:
        if "mapper_settings" not in config:
            raise ValueError("Config missing top-level key 'mapper_settings'.")
        ms = config["mapper_settings"]

        unknown = set(ms.keys()) - cls._KNOWN_MAPPER_SETTINGS_KEYS
        if unknown:
            raise ValueError(
                f"Unknown mapper_settings keys: {sorted(unknown)}. "
                f"Known keys: {sorted(cls._KNOWN_MAPPER_SETTINGS_KEYS)}"
            )

        for key, typ, cond in cls._REQUIRED_SETTINGS:
            if key not in ms:
                raise ValueError(f"mapper_settings missing required key '{key}'.")
            val = ms[key]
            if not isinstance(val, typ):
                raise ValueError(
                    f"mapper_settings.{key}: expected {typ}, "
                    f"got {type(val).__name__} ({val!r})."
                )
            if not cond(val):
                raise ValueError(f"mapper_settings.{key}={val!r} failed validation.")

        if "output_dir" in ms and not isinstance(ms["output_dir"], str):
            raise ValueError("mapper_settings.output_dir must be a string.")

        if "excluded_node_ids" in ms:
            en = ms["excluded_node_ids"]
            if not isinstance(en, list) or not all(isinstance(x, int) for x in en):
                raise ValueError("mapper_settings.excluded_node_ids must be list[int].")

        lo, hi = ms.get("dist_lower_ang"), ms.get("dist_upper_ang")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(
                f"dist_lower_ang ({lo}) must be < dist_upper_ang ({hi})."
            )

        if "n_parallel" in ms:
            val = ms["n_parallel"]
            if not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"mapper_settings.n_parallel must be int >= 1, got {val!r}."
                )

        logger.info("_validate_mapper_settings: all checks passed.")

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        base_config: dict,
        queue: ExplorationQueue | None = None,
        structure_checker: StructureChecker | None = None,
        perturbation_generator: PerturbationGenerator | None = None,
        output_dir: str = "mapper_output",
        work_dir: str | None = None,
        graph_json: str = "reaction_network.json",
        max_iterations: int = 50,
        resume: bool = False,
        rng_seed: int = 42,
        energy_tolerance_kcalmol: float = 1.0,
        config_file_path: str | None = None,
        excluded_node_ids: list[int] | set[int] | None = None,
        exclude_bond_rearrangement: bool = False,
        bond_checker: BondTopologyChecker | None = None,
        n_parallel: int = 1,
    ) -> None:
        self.base_config = base_config
        self._validate_mapper_settings(base_config)

        ms = base_config.get("mapper_settings", {})
        temperature_K              = float(ms.get("temperature_K",              300.0))
        rmsd_threshold             = float(ms.get("rmsd_threshold",             0.30))
        max_iterations             = int(  ms.get("max_iterations",             max_iterations))
        afir_gamma_kJmol           = float(ms.get("afir_gamma_kJmol",           100.0))
        max_pairs                  = int(  ms.get("max_pairs",                  10))
        dist_lower_ang             = float(ms.get("dist_lower_ang",             2.0))
        dist_upper_ang             = float(ms.get("dist_upper_ang",             5.0))
        n_parallel                 = int(  ms.get("n_parallel",                 n_parallel))
        rng_seed                   = int(  ms.get("rng_seed",                   rng_seed))
        active_atoms               = ms.get("active_atoms",                None)
        excluded_node_ids          = ms.get("excluded_node_ids",           excluded_node_ids)
        exclude_bond_rearrangement = bool(ms.get("exclude_bond_rearrangement",  exclude_bond_rearrangement))
        include_negative_gamma     = bool(ms.get("include_negative_gamma",      False))
        energy_tolerance_kcalmol   = float(ms.get("energy_tolerance_kcalmol",   energy_tolerance_kcalmol))

        if queue is not None:
            self.queue = queue
        elif ms.get("use_rcmc", False):
            self.queue = RCMCQueue(
                temperature_K=float(ms.get("rcmc_temperature_K", temperature_K)),
                reaction_time_s=float(ms.get("rcmc_reaction_time_s", 1.0)),
                start_node_id=ms.get("rcmc_start_node_id", None),
            )
        else:
            self.queue = BoltzmannQueue(temperature_K=temperature_K)

        self.checker   = structure_checker or StructureChecker(rmsd_threshold=rmsd_threshold)
        self.perturber = perturbation_generator or PerturbationGenerator(
            afir_gamma_kJmol=afir_gamma_kJmol,
            max_pairs=max_pairs,
            dist_lower_ang=dist_lower_ang,
            dist_upper_ang=dist_upper_ang,
            include_negative_gamma=include_negative_gamma,
            active_atoms=active_atoms,
            rng_seed=rng_seed,
        )

        self.output_dir      = os.path.abspath(output_dir)
        self.work_dir        = os.path.abspath(work_dir or output_dir)
        self.graph_json_path = os.path.join(self.output_dir, graph_json)
        self.max_iterations  = max_iterations
        self.resume          = resume
        self._rng            = np.random.default_rng(rng_seed)
        self.graph           = NetworkGraph()
        self._iteration      = 0
        self._pending_node_updates: dict[int, NodeUpdateData] = {}

        if self.resume or not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            raise FileExistsError(
                f"Output directory '{self.output_dir}' already exists. "
                "Use --resume to continue an existing run."
            )
        os.makedirs(self.work_dir, exist_ok=True)
        for sub in ("nodes", "runs"):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

        self.energy_tolerance = energy_tolerance_kcalmol / HARTREE_TO_KCALMOL
        self.explored_log     = ExploredPairsLog(
            os.path.join(self.work_dir, "explored_pairs.txt")
        )
        self.config_file_path: str | None = (
            os.path.abspath(config_file_path) if config_file_path else None
        )

        self.excluded_node_ids: set[int] = set(excluded_node_ids) if excluded_node_ids else set()
        self._bond_rearrangement_excluded_ids: set[int] = set()
        self.exclude_bond_rearrangement = exclude_bond_rearrangement
        self.bond_checker = bond_checker or BondTopologyChecker()
        self._ref_bond_fingerprint: dict[tuple[str, str], int] | None = None
        self._ref_bond_symbols: list[str] = []
        self._ref_bond_coords: np.ndarray = np.empty((0, 3), dtype=float)

        self.n_parallel = max(1, int(n_parallel))
        self._mp_ctx    = multiprocessing.get_context("spawn")
        raw_to = ms.get("worker_timeout_s", None)
        self.worker_timeout_s: int | None = int(raw_to) if raw_to is not None else None

    # ------------------------------------------------------------------ #
    #  AFIR key parsing                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_afir_task_key(task: ExplorationTask) -> tuple[str, int, int]:
        gamma_sign = "-" if task.afir_params and task.afir_params[0].startswith("-") else "+"
        try:
            atom_i = int(task.afir_params[1]) if len(task.afir_params) >= 3 else 0
            atom_j = int(task.afir_params[2]) if len(task.afir_params) >= 3 else 0
        except (ValueError, IndexError):
            logger.warning("_parse_afir_task_key: bad afir_params=%r", task.afir_params)
            atom_i, atom_j = 0, 0
        return gamma_sign, atom_i, atom_j

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self._validate_mapper_settings(self.base_config)

        if self.resume and os.path.isfile(self.graph_json_path):
            self.graph.load(self.graph_json_path)
            self._iteration = self.graph.last_iteration
            self._requeue_all_nodes()
        else:
            self._register_seed_structure()

        self._save_config_snapshot()

        history_log  = os.path.join(self.output_dir, "exploration_history.log")
        priority_log = os.path.join(self.output_dir, "queue_priority.log")

        try:
            while True:
                if os.path.isfile(os.path.join(self.output_dir, "stop.txt")):
                    logger.info("stop.txt detected — stopping.")
                    break
                if self.max_iterations > 0 and self._iteration >= self.max_iterations:
                    logger.info("Reached max_iterations (%d).", self.max_iterations)
                    break

                # ── Parallel path ───────────────────────────────────────
                if self.n_parallel > 1:
                    self.queue.refresh_priorities(self.graph.reference_energy())
                    remaining = (self.max_iterations - self._iteration
                                 if self.max_iterations > 0 else self.n_parallel)
                    if remaining <= 0:
                        break
                    batch = self._collect_task_batch(min(self.n_parallel, remaining))
                    if not batch:
                        logger.info("All (EQ, pair) combinations exhausted.")
                        break
                    self._iteration += len(batch)
                    self.graph.last_iteration = self._iteration
                    self._run_batch_parallel(batch, history_log, priority_log)
                    continue

                # ── Sequential path ─────────────────────────────────────
                self.queue.refresh_priorities(self.graph.reference_energy())
                task = self.queue.pop()

                if task is None:
                    logger.info("Queue empty; re-scanning all nodes.")
                    for node in self.graph.all_nodes():
                        self._enqueue_perturbations(node, force_add=True)
                    if not len(self.queue):
                        logger.info("All (EQ, pair) combinations exhausted.")
                        break
                    continue

                gamma_sign, atom_i, atom_j = self._parse_afir_task_key(task)
                if self.explored_log.has(task.node_id, atom_i, atom_j, gamma_sign):
                    logger.debug("Skipping already-explored task EQ%06d %d-%d %s.",
                                 task.node_id, atom_i, atom_j, gamma_sign)
                    self.queue.release((task.node_id, tuple(task.afir_params)))
                    continue

                self._iteration += 1
                self.graph.last_iteration = self._iteration
                self._append_history(history_log, self._iteration, task)

                run_dir = self._make_run_dir(task)
                try:
                    profile_dirs = self._run_autots(task, run_dir)
                except Exception as exc:
                    logger.error("AutoTS failed for %s: %s", run_dir, exc)
                    self.queue.release((task.node_id, tuple(task.afir_params)))
                    self._finalize_iteration(run_dir, task, "FAILED", [], priority_log)
                    continue

                self.explored_log.record(task.node_id, atom_i, atom_j, gamma_sign)
                self.queue.release((task.node_id, tuple(task.afir_params)))
                logger.info("Iter %06d: %d profile(s).", self._iteration, len(profile_dirs))
                for pdir in profile_dirs:
                    self._process_profile(pdir, run_dir)
                if hasattr(self.queue, "set_graph"):
                    self.queue.set_graph(self.graph)
                self._finalize_iteration(run_dir, task, "DONE", profile_dirs, priority_log)

        finally:
            self.graph.save(self.graph_json_path)
            self.explored_log.close()

    def _append_history(self, path: str, iteration: int, task: ExplorationTask,
                        status: str = "") -> None:
        with open(path, "a", encoding="utf-8") as fh:
            line = (f"Iter: {iteration:06d} | Node: EQ{task.node_id:06d} "
                    f"| Priority: {task.priority:.6f} | AFIR: {task.afir_params}")
            if status:
                line += f" | Status: {status}"
            fh.write(line + "\n")

    def _finalize_iteration(self, run_dir: str, task: ExplorationTask, status: str,
                            profile_dirs: list[str], priority_log: str,
                            iteration: int | None = None) -> None:
        self._flush_node_energy_updates()
        self._save_run_metadata(run_dir, task, status, profile_dirs, iteration)
        self.graph.save(self.graph_json_path)
        self._write_priority_log(priority_log)

    # ------------------------------------------------------------------ #
    #  Batch collection and parallel execution                             #
    # ------------------------------------------------------------------ #

    def _collect_task_batch(
        self, n: int
    ) -> list[tuple[ExplorationTask, str, str, int, int, int]]:
        batch: list[tuple] = []
        _MAX_REFILLS, refill_count = 2, 0
        while len(batch) < n:
            task = self.queue.pop()
            if task is None:
                if refill_count >= _MAX_REFILLS:
                    break
                size_before = len(self.queue)
                for node in self.graph.all_nodes():
                    self._enqueue_perturbations(node, force_add=True)
                refill_count += 1
                if len(self.queue) == size_before:
                    break
                continue

            gamma_sign, atom_i, atom_j = self._parse_afir_task_key(task)
            if self.explored_log.has(task.node_id, atom_i, atom_j, gamma_sign):
                self.queue.release((task.node_id, tuple(task.afir_params)))
                continue

            next_iter = self._iteration + len(batch) + 1
            run_dir   = self._make_run_dir(task, iteration=next_iter)
            batch.append((task, run_dir, gamma_sign, atom_i, atom_j, next_iter))
        return batch

    def _run_batch_parallel(
        self,
        batch: list[tuple],
        history_log: str,
        priority_log: str,
    ) -> None:
        import queue as _queue_mod

        # pid -> (proc, q, task, run_dir, iteration, gamma_sign, atom_i, atom_j, t0)
        active: dict[int, tuple] = {}

        for task, run_dir, gamma_sign, atom_i, atom_j, iteration in batch:
            workspace = os.path.join(run_dir, "autots_workspace")
            config    = self._make_autots_config(task, workspace)
            try:
                with open(os.path.join(run_dir, "config_used.json"), "w") as fh:
                    json.dump(config, fh, indent=2, default=str)
            except Exception as exc:
                logger.warning("Could not write config_used.json: %s", exc)

            q    = self._mp_ctx.Queue()
            proc = self._mp_ctx.Process(target=_autots_worker_with_queue,
                                        args=(config, run_dir, workspace, q))
            try:
                proc.start()
            except OSError:
                self._close_queue(q)
                self._process_single_result(task, run_dir, [], "FAILED", iteration,
                                            history_log, gamma_sign, atom_i, atom_j)
                continue
            if proc.pid is None:
                self._process_single_result(task, run_dir, [], "FAILED", iteration,
                                            history_log, gamma_sign, atom_i, atom_j)
                self._close_queue(q); proc.join(timeout=5)
                continue
            active[proc.pid] = (proc, q, task, run_dir, iteration,
                                gamma_sign, atom_i, atom_j, time.time())

        poll_interval = 60.0
        try:
            while active:
                # Per-process timeout check
                if self.worker_timeout_s is not None:
                    now = time.time()
                    for pid in [p for p, e in active.items()
                                if now - e[8] >= self.worker_timeout_s]:
                        proc, q, task, run_dir, it, gs, ai, aj, _ = active[pid]
                        self._kill_proc(proc); self._close_queue(q)
                        logger.error("Worker timeout: %s", run_dir)
                        self._process_single_result(task, run_dir, [], "TIMEOUT",
                                                    it, history_log, gs, ai, aj)
                        del active[pid]
                    if not active:
                        break

                for pid, (proc, q, task, run_dir, it,
                          gs, ai, aj, t0) in list(active.items()):
                    try:
                        tag, payload = q.get_nowait()
                    except _queue_mod.Empty:
                        if not proc.is_alive():
                            try:
                                tag, payload = q.get(timeout=5.0)
                            except _queue_mod.Empty:
                                tag, payload = "err", None
                            finally:
                                proc.join(timeout=30); self._close_queue(q)
                        else:
                            continue
                    else:
                        proc.join(timeout=120)
                        if proc.is_alive(): self._kill_proc(proc)
                        self._close_queue(q)

                    if tag == "err":
                        logger.error("AutoTS failed for %s:\n%s", run_dir, payload)
                        payload = []
                    self._process_single_result(
                        task, run_dir, payload if tag == "ok" else [],
                        "DONE" if tag == "ok" else "FAILED",
                        it, history_log, gs, ai, aj,
                    )
                    del active[pid]

                if active:
                    sleep_t = poll_interval
                    if self.worker_timeout_s is not None:
                        now = time.time()
                        sleep_t = min(poll_interval,
                                      max(0.0, min(self.worker_timeout_s - (now - e[8])
                                                   for e in active.values())))
                    if sleep_t > 0:
                        time.sleep(sleep_t)

        finally:
            for pid, (proc, q, task, run_dir, it,
                      gs, ai, aj, _) in list(active.items()):
                if not self.queue.is_submitted((task.node_id, tuple(task.afir_params))):
                    continue
                if proc.is_alive(): self._kill_proc(proc)
                status, payload = "FAILED", []
                try:
                    tag, payload = q.get_nowait()
                    if tag == "err":
                        logger.error("AutoTS failed (finally) %s:\n%s", run_dir, payload)
                        payload = []
                    status = "DONE" if tag == "ok" else "FAILED"
                except _queue_mod.Empty:
                    pass
                try:
                    self._process_single_result(task, run_dir, payload, status,
                                                it, history_log, gs, ai, aj)
                except Exception as exc:
                    logger.error("_process_single_result failed (finally) %s: %s", run_dir, exc)
                finally:
                    self._close_queue(q)
            try:
                self.graph.save(self.graph_json_path)
                self._write_priority_log(priority_log)
            except Exception as exc:
                logger.error("Cleanup save/log failed: %s", exc)

    @staticmethod
    def _kill_proc(proc) -> None:
        proc.terminate(); proc.join(timeout=10)
        if proc.is_alive(): proc.kill(); proc.join(timeout=10)

    @staticmethod
    def _close_queue(q) -> None:
        try: q.cancel_join_thread(); q.close()
        except Exception: pass

    def _process_single_result(
        self,
        task: ExplorationTask,
        run_dir: str,
        profile_dirs: list[str],
        status: str,
        iteration: int,
        history_log: str,
        gamma_sign: str,
        atom_i: int,
        atom_j: int,
    ) -> None:
        self.graph.last_iteration = max(self.graph.last_iteration, iteration)
        self._append_history(history_log, iteration, task, status)

        if status == "DONE":
            logger.info("Iter %06d (batch): %d profile(s).", iteration, len(profile_dirs))
            for pdir in profile_dirs:
                self._process_profile(pdir, run_dir)
            self.explored_log.record(task.node_id, atom_i, atom_j, gamma_sign)

        self.queue.release((task.node_id, tuple(task.afir_params)))
        if hasattr(self.queue, "set_graph"):
            self.queue.set_graph(self.graph)
        self._finalize_iteration(
            run_dir, task, status, profile_dirs,
            os.path.join(self.output_dir, "queue_priority.log"),
            iteration,
        )

    # ------------------------------------------------------------------ #
    #  Config snapshot                                                     #
    # ------------------------------------------------------------------ #

    def _save_config_snapshot(self) -> None:
        if not self.config_file_path or not os.path.isfile(self.config_file_path):
            return
        dst = os.path.join(self.work_dir, "config_snapshot.json")
        if os.path.abspath(self.config_file_path) == os.path.abspath(dst):
            return
        try:
            shutil.copy2(self.config_file_path, dst)
        except Exception as exc:
            logger.warning("_save_config_snapshot failed: %s", exc)

    # ------------------------------------------------------------------ #
    #  Seed registration and initial optimisation                          #
    # ------------------------------------------------------------------ #

    def _register_seed_structure(self) -> None:
        seed_xyz        = os.path.abspath(self.base_config["initial_mol_file"])
        opt_xyz, energy = self._run_initial_optimization(seed_xyz)
        node_id         = self.graph.next_node_id()
        saved_xyz       = self._persist_node_xyz(opt_xyz, node_id)
        try:
            symbols, coords = parse_xyz(saved_xyz)
        except Exception as exc:
            raise RuntimeError(f"Cannot parse seed structure {saved_xyz}: {exc}") from exc

        node = EQNode(node_id=node_id, xyz_file=saved_xyz, energy=energy,
                      symbols=symbols, coords=coords,
                      source_run_dir="initial_optimization")
        self.graph.add_node(node)
        self.graph.save(self.graph_json_path)

        if self.exclude_bond_rearrangement and node.coords.size > 0:
            self._ref_bond_fingerprint = self.bond_checker.fingerprint(node.symbols, node.coords)
            self._ref_bond_symbols     = list(node.symbols)
            self._ref_bond_coords      = node.coords.copy()
            logger.info("Reference bond fingerprint stored: %s", self._ref_bond_fingerprint)

        self._enqueue_perturbations(node, force_add=True)

    def _run_initial_optimization(self, seed_xyz: str) -> tuple[str, float | None]:
        if OptimizationJob is None:
            raise RuntimeError("OptimizationJob not available.")
        try:
            _AFIR_KEYS = {"manual_AFIR", "afir_gamma_kJmol", "max_pairs"}
            opt_kwargs = {k: v for k, v in self.base_config.get("step1_settings", {}).items()
                          if k not in _AFIR_KEYS}
            if "software_path_file_source" in self.base_config:
                opt_kwargs["software_path_file"] = self.base_config["software_path_file_source"]
            opt_kwargs["run_type"] = "opt"

            opt_job = OptimizationJob(os.path.relpath(seed_xyz))
            if opt_kwargs:
                opt_job.set_options(**opt_kwargs)
            opt_job.run()
            optimizer = opt_job.get_results()
            if optimizer is None:
                raise RuntimeError("opt_job.get_results() returned None.")

            files = glob.glob(os.path.join(optimizer.BPA_FOLDER_DIRECTORY, "*_optimized.xyz"))
            if not files:
                raise FileNotFoundError("Optimized XYZ not found.")
            opt_xyz = os.path.abspath(files[0])

            energy: float | None = None
            for name in ("get_energy", "get_final_energy", "get_scf_energy"):
                m = getattr(optimizer, name, None)
                if callable(m):
                    try:
                        raw = m()
                        if raw is not None:
                            energy = float(raw); break
                    except Exception:
                        pass
            if energy is None:
                for attr in ("energy", "final_energy", "scf_energy"):
                    raw = getattr(optimizer, attr, None)
                    if raw is not None:
                        try:
                            energy = float(raw); break
                        except (TypeError, ValueError):
                            pass
            if energy is None:
                logger.warning("Could not retrieve energy from optimizer. EQ0 will have energy=None.")
            return opt_xyz, energy

        except Exception as exc:
            raise RuntimeError(f"Initial optimization failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Logging helpers                                                     #
    # ------------------------------------------------------------------ #

    def _write_priority_log(self, priority_log: str) -> None:
        ref_e   = self.graph.reference_energy()
        ref_str = f"{ref_e:.10f} Ha" if ref_e is not None else "N/A"
        node_priority:   dict[int, float] = {}
        node_pair_count: dict[int, int]   = {}
        for item in self.queue.export_queue_status():
            nid = item["node_id"]
            node_priority[nid]   = item["priority"]
            node_pair_count[nid] = node_pair_count.get(nid, 0) + 1
        with open(priority_log, "w", encoding="utf-8") as fh:
            fh.write(f"# Iter {self._iteration:06d} | queued={len(self.queue)} | "
                     f"explored_pairs={len(self.explored_log)} | ref_energy={ref_str}\n")
            for nid in sorted(node_priority, key=lambda x: node_priority[x], reverse=True):
                fh.write(f"Node: EQ{nid:06d} | Priority: {node_priority[nid]:.6f} | "
                         f"pending_pairs={node_pair_count[nid]}\n")

    # ------------------------------------------------------------------ #
    #  Resume helper                                                       #
    # ------------------------------------------------------------------ #

    def _requeue_all_nodes(self) -> None:
        # Step 1: restore reference bond fingerprint from EQ0
        if self.exclude_bond_rearrangement and self._ref_bond_fingerprint is None:
            seed = self.graph.get_node(0)
            if seed is not None and seed.coords.size > 0:
                self._ref_bond_fingerprint = self.bond_checker.fingerprint(seed.symbols, seed.coords)
                self._ref_bond_symbols     = list(seed.symbols)
                self._ref_bond_coords      = seed.coords.copy()
                logger.info("_requeue_all_nodes: reference fingerprint restored from EQ0.")
            elif seed is not None:
                logger.warning(
                    "_requeue_all_nodes: EQ0 coords empty (XYZ missing?). "
                    "bond-rearrangement filter disabled. Expected: %s", seed.xyz_file,
                )

        # Step 2: rebuild bond-rearrangement exclusion list.
        # BUG FIX: original code had broken indentation and missing arguments
        # to has_rearrangement() (node.symbols, node.coords were omitted).
        if self.exclude_bond_rearrangement and self._ref_bond_fingerprint is not None:
            for node in self.graph.all_nodes():
                if node.coords.size == 0:
                    continue
                if self.bond_checker.has_rearrangement(
                    self._ref_bond_symbols, self._ref_bond_coords,
                    node.symbols, node.coords,
                ):
                    self.excluded_node_ids.add(node.node_id)
                    self._bond_rearrangement_excluded_ids.add(node.node_id)
                    logger.info("_requeue_all_nodes: EQ%d restored to excluded_node_ids.",
                                node.node_id)

        # Step 3: rebuild queues
        for node in self.graph.all_nodes():
            self._enqueue_perturbations(node, force_add=True)

    # ------------------------------------------------------------------ #
    #  Run-directory helpers                                               #
    # ------------------------------------------------------------------ #

    def _make_run_dir(self, task: ExplorationTask, iteration: int | None = None) -> str:
        iter_num = self._iteration if iteration is None else iteration
        run_dir  = os.path.join(self.output_dir, "runs",
                                f"run_{iter_num:06d}_node{task.node_id}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "autots_workspace"), exist_ok=True)
        with open(os.path.join(run_dir, "input.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"node_id = {task.node_id}\n")
        return run_dir

    def _save_run_metadata(self, run_dir: str, task: ExplorationTask, status: str,
                           profile_dirs: list[str], iteration: int | None = None) -> None:
        with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "iteration":    iteration if iteration is not None else self._iteration,
                "node_id":      task.node_id,
                "afir_params":  task.afir_params,
                "priority":     task.priority,
                "status":       status,
                "profile_dirs": profile_dirs,
            }, fh, indent=2)

    # ------------------------------------------------------------------ #
    #  AutoTS subprocess execution                                         #
    # ------------------------------------------------------------------ #

    def _make_autots_config(self, task: ExplorationTask, workspace: str) -> dict:
        config = copy.deepcopy(self.base_config)
        config["initial_mol_file"] = task.xyz_file
        config["work_dir"]         = workspace
        config.setdefault("step1_settings", {})
        config["step1_settings"]["manual_AFIR"] = task.afir_params
        config["run_step4"] = True
        return config

    def _run_autots(self, task: ExplorationTask, run_dir: str) -> list[str]:
        if AutoTSWorkflow is None:
            raise RuntimeError("AutoTSWorkflow not available.")

        workspace = os.path.join(run_dir, "autots_workspace")
        config    = self._make_autots_config(task, workspace)
        try:
            with open(os.path.join(run_dir, "config_used.json"), "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2, default=str)
        except Exception as exc:
            logger.warning("_run_autots: could not write config_used.json: %s", exc)

        result_q: multiprocessing.Queue = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(target=_autots_worker_with_queue,
                                    args=(config, run_dir, workspace, result_q))
        try:
            proc.start()
        except OSError:
            result_q.close(); raise

        import queue as _queue_mod
        poll_interval, start_time = 60, time.time()
        tag = payload = None

        # Polling loop: detects worker crashes without imposing a wall-clock limit.
        while True:
            try:
                tag, payload = result_q.get(timeout=poll_interval)
                break
            except _queue_mod.Empty:
                pass

            if not proc.is_alive():
                # Worker exited without putting a result — it crashed.
                try:
                    tag, payload = result_q.get(timeout=5.0)
                    break
                except _queue_mod.Empty:
                    pass
                proc.join(timeout=30); self._close_queue(result_q)
                raise RuntimeError(
                    f"Worker terminated unexpectedly (exit_code={proc.exitcode}). "
                    "Possible causes: segfault, OOM kill, unhandled C-level exception."
                )

            elapsed = time.time() - start_time
            if self.worker_timeout_s is not None and elapsed >= self.worker_timeout_s:
                self._kill_proc(proc); self._close_queue(result_q)
                raise RuntimeError(
                    f"Worker exceeded hard timeout of {self.worker_timeout_s}s "
                    f"(elapsed={elapsed:.0f}s)."
                )

        # Drain residual items, then join (cancel_join_thread prevents deadlock).
        try:
            result_q.cancel_join_thread()
        except Exception:
            pass
        while True:
            try:
                result_q.get_nowait()
            except (_queue_mod.Empty, OSError):
                break

        try:
            proc.join(timeout=120)
            if proc.is_alive(): self._kill_proc(proc)
            if tag == "err":
                raise RuntimeError(f"AutoTSWorkflow subprocess failed:\n{payload}")
            return payload
        finally:
            self._close_queue(result_q)

    # ------------------------------------------------------------------ #
    #  Energy backfill                                                     #
    # ------------------------------------------------------------------ #

    def _flush_node_energy_updates(self) -> None:
        if not self._pending_node_updates:
            return
        for node_id, updates in self._pending_node_updates.items():
            node = self.graph.get_node(node_id)
            if node is None:
                continue
            if updates.energy is not None and node.energy is None:
                node.energy = updates.energy
                logger.info("_flush_node_energy_updates: EQ%d energy=%.10f Ha",
                            node_id, updates.energy)
            if updates.free_energy is not None and node.free_energy is None:
                node.free_energy = updates.free_energy
                logger.info("_flush_node_energy_updates: EQ%d free_energy=%.10f Ha",
                            node_id, updates.free_energy)
        self._pending_node_updates.clear()
        self.graph.invalidate_energy_cache()

    # ------------------------------------------------------------------ #
    #  Node identity and registration                                      #
    # ------------------------------------------------------------------ #

    def _find_or_register_node(
        self,
        xyz_file: str,
        energy: float | None,
        run_dir: str,
        free_energy: float | None = None,
    ) -> int | None:
        try:
            symbols, coords = parse_xyz(xyz_file)
        except Exception as exc:
            logger.warning("_find_or_register_node: cannot parse %s: %s", xyz_file, exc)
            return None

        candidates = (
            self.graph.nodes_in_energy_window(energy, self.energy_tolerance)
            if energy is not None else self.graph.all_nodes()
        )
        for existing in candidates:
            if (existing.node_id in self.excluded_node_ids
                    and existing.node_id not in self._bond_rearrangement_excluded_ids):
                continue
            if existing.coords.size == 0:
                continue
            if self.checker.compute_rmsd(symbols, coords,
                                         existing.symbols, existing.coords) \
                    < self.checker.rmsd_threshold:
                # Queue energy backfill for null fields
                prev = self._pending_node_updates.get(existing.node_id, NodeUpdateData())
                if energy is not None and existing.energy is None and prev.energy is None:
                    prev.energy = energy
                if free_energy is not None and existing.free_energy is None \
                        and prev.free_energy is None:
                    prev.free_energy = free_energy
                if prev.energy is not None or prev.free_energy is not None:
                    self._pending_node_updates[existing.node_id] = prev
                return existing.node_id

        # New unique node
        node_id   = self.graph.next_node_id()
        saved_xyz = self._persist_node_xyz(xyz_file, node_id)
        new_node  = EQNode(node_id=node_id, xyz_file=saved_xyz, energy=energy,
                           symbols=symbols, coords=coords, source_run_dir=run_dir,
                           free_energy=free_energy)
        self.graph.add_node(new_node)

        if (self.exclude_bond_rearrangement
                and self._ref_bond_fingerprint is not None
                and new_node.coords.size > 0
                and self.bond_checker.has_rearrangement(
                    self._ref_bond_symbols, self._ref_bond_coords,
                    new_node.symbols, new_node.coords)):
            self.excluded_node_ids.add(node_id)
            self._bond_rearrangement_excluded_ids.add(node_id)
            logger.info("EQ%d excluded: bond topology differs from seed.", node_id)
            return node_id

        ref_e = self.graph.reference_energy()
        self._enqueue_perturbations(
            new_node,
            force_add=(ref_e is None or new_node.energy is None),
        )
        return node_id

    # ------------------------------------------------------------------ #
    #  Profile processing                                                  #
    # ------------------------------------------------------------------ #

    def _process_profile(self, profile_dir: str, run_dir: str) -> None:
        result = ProfileParser().parse(profile_dir)
        if result is None:
            return

        ts_sym:    list[str]  = []
        ts_coords: np.ndarray = np.empty((0, 3), dtype=float)
        ts_xyz = result.get("ts_xyz_file", "") or ""
        if ts_xyz and os.path.isfile(ts_xyz):
            try:
                ts_sym, ts_coords = parse_xyz(ts_xyz)
            except Exception as exc:
                logger.warning("Cannot parse TS XYZ %s: %s — profile skipped.", ts_xyz, exc)
                return
        else:
            logger.warning("TS XYZ missing (%r) — profile skipped.", ts_xyz)
            return

        node_id_1 = self._find_or_register_node(
            result["endpoint_1_xyz"], result["endpoint_1_energy"], run_dir,
            result.get("endpoint_1_free_energy"))
        node_id_2 = self._find_or_register_node(
            result["endpoint_2_xyz"], result["endpoint_2_energy"], run_dir,
            result.get("endpoint_2_free_energy"))

        if node_id_1 is None or node_id_2 is None:
            return

        ts_energy  = result["ts_energy"]
        candidates = (self.graph.edges_in_energy_window(ts_energy, self.energy_tolerance)
                      if ts_energy is not None else self.graph.all_edges())
        for edge in candidates:
            if not edge.has_coords:
                logger.warning("TS%06d has no coords — duplicate check skipped.", edge.edge_id)
                continue
            if self.checker.are_similar(ts_sym, ts_coords, edge.symbols, edge.coords):
                logger.info("Duplicate TS (matches TS%06d) — skipped EQ%d--EQ%d.",
                            edge.edge_id, node_id_1, node_id_2)
                return

        edge_id  = self.graph.next_edge_id()
        saved_ts = self._persist_ts_xyz(ts_xyz, edge_id) if ts_xyz else None
        self.graph.add_edge(TSEdge(
            edge_id=edge_id, node_id_1=node_id_1, node_id_2=node_id_2,
            ts_xyz_file=saved_ts, ts_energy=ts_energy,
            barrier_fwd=result["barrier_fwd"], barrier_rev=result["barrier_rev"],
            source_run_dir=run_dir, symbols=ts_sym, coords=ts_coords,
            ts_free_energy=result.get("ts_free_energy"),
            barrier_fwd_free=result.get("barrier_fwd_free"),
            barrier_rev_free=result.get("barrier_rev_free"),
        ))
        logger.info("New TS%06d: EQ%d -- EQ%d  E(TS)=%s  Ea(fwd)=%s kcal/mol",
                    edge_id, node_id_1, node_id_2,
                    f"{ts_energy:.8f} Ha" if ts_energy is not None else "N/A",
                    f"{result['barrier_fwd']:.2f}" if result["barrier_fwd"] is not None else "N/A")

    # ------------------------------------------------------------------ #
    #  Exploration helpers                                                 #
    # ------------------------------------------------------------------ #

    def _node_has_been_explored(self, node_id: int) -> bool:
        return self.explored_log.has_any_for_node(node_id)

    def _enqueue_perturbations(self, node: EQNode, force_add: bool = False) -> None:
        if node.coords.size == 0:
            return

        # Skip excluded nodes that have already been explored at least once.
        # Exception: single-node graph cannot be skipped or the network stalls.
        if (self.graph.node_count > 1
                and node.node_id in self.excluded_node_ids
                and self._node_has_been_explored(node.node_id)):
            return

        ref_e = self.graph.reference_energy()
        if not force_add and node.energy is not None and ref_e is not None:
            if not self.queue.should_add(node, ref_e):
                return

        all_candidates = self.perturber.get_candidate_pairs(node.symbols, node.coords)
        if not all_candidates:
            return

        pos_gamma   = f"{self.perturber.afir_gamma_kJmol:.6g}"
        neg_gamma   = f"{-self.perturber.afir_gamma_kJmol:.6g}"
        gamma_signs = ["+"] + (["-"] if self.perturber.include_negative_gamma else [])
        sign_to_g   = {"+": pos_gamma, "-": neg_gamma}
        nid         = node.node_id

        unexplored = [
            (i0, j0) for i0, j0 in all_candidates
            if any(
                not self.explored_log.has(nid, i0 + 1, j0 + 1, s)
                and not self.queue.is_submitted((nid, (sign_to_g[s], str(i0+1), str(j0+1))))
                for s in gamma_signs
            )
        ]
        if not unexplored:
            return

        n_sel   = min(self.perturber.max_pairs, len(unexplored))
        chosen  = self._rng.choice(len(unexplored), size=n_sel, replace=False)
        eff_e   = node.effective_energy
        delta_e = (eff_e - ref_e) if (eff_e is not None and ref_e is not None) else 0.0

        for idx in chosen:
            i0, j0 = unexplored[int(idx)]
            ai, aj = i0 + 1, j0 + 1
            for sign in gamma_signs:
                g = sign_to_g[sign]
                if (self.explored_log.has(nid, ai, aj, sign)
                        or self.queue.is_submitted((nid, (g, str(ai), str(aj))))):
                    continue
                self.queue.push(ExplorationTask(
                    node_id=nid, xyz_file=node.xyz_file,
                    afir_params=[g, str(ai), str(aj)],
                    metadata={
                        "delta_E_hartree":         delta_e,
                        "source_node_energy":      node.energy,
                        "source_node_free_energy": getattr(node, "free_energy", None),
                    },
                ))

    # ------------------------------------------------------------------ #
    #  File persistence                                                    #
    # ------------------------------------------------------------------ #

    def _persist_node_xyz(self, src_xyz: str, node_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"EQ{node_id:06d}.xyz")
        try:
            if os.path.abspath(src_xyz) != os.path.abspath(dst):
                shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot persist EQ{node_id:06d} XYZ ({src_xyz!r} → {dst!r}): {exc}"
            ) from exc

    def _persist_ts_xyz(self, src_xyz: str, edge_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"TS{edge_id:06d}.xyz")
        try:
            if os.path.abspath(src_xyz) != os.path.abspath(dst):
                shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot persist TS{edge_id:06d} XYZ ({src_xyz!r} → {dst!r}): {exc}"
            ) from exc