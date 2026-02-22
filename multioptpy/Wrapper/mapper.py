"""
mapper.py - Chemical Reaction Network Mapper
============================================

Autonomously maps a chemical reaction network using AutoTSWorkflow
and external optimization utilities.
"""

from __future__ import annotations

import bisect
import glob
import json
import logging
import os
import copy
import shutil
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

# Internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from multioptpy.Wrapper.autots import AutoTSWorkflow
except ImportError as _autots_import_err:
    print(f"[mapper] Warning: could not import AutoTSWorkflow: {_autots_import_err}")
    AutoTSWorkflow = None  # type: ignore[assignment,misc]

try:
    from optimize_wrapper import OptimizationJob
except ImportError as _opt_import_err:
    print(f"[mapper] Warning: could not import OptimizationJob: {_opt_import_err}")
    OptimizationJob = None # type: ignore[assignment,misc]

try:
    from multioptpy.Parameters.covalent_radii import covalent_radii_lib
    from multioptpy.Parameters.unit_values import UnitValueLib
    _BOHR2ANG = UnitValueLib().bohr2angstroms
except ImportError as _covalent_import_err:
    print(f"[mapper] Warning: could not import covalent radii lib: {_covalent_import_err}")
    covalent_radii_lib = None
    _BOHR2ANG = 0.529177210903

logger = logging.getLogger(__name__)

# Module-level physical constants
HARTREE_TO_KCALMOL: float = 627.509474
K_B_HARTREE: float = 3.166811563e-6


# ===========================================================================
# Section 1 : XYZ Utilities
# ===========================================================================

def parse_xyz(filepath: str) -> tuple[list[str], np.ndarray]:
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    n_atoms: int | None = None
    data_start: int = 0

    non_blank = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
    if non_blank and non_blank[0][1].isdigit():
        n_atoms = int(non_blank[0][1])
        data_start = non_blank[0][0] + 2

    symbols: list[str] = []
    coords_raw: list[list[float]] = []

    for ln in lines[data_start:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            symbols.append(parts[0])
            coords_raw.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        if n_atoms is not None and len(symbols) >= n_atoms:
            break

    if not symbols:
        raise ValueError(f"No atomic coordinates found in: {filepath}")

    return symbols, np.array(coords_raw, dtype=float)

def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# ===========================================================================
# Section 2 : Fast StructureChecker
# ===========================================================================

class StructureChecker:
    def __init__(self, rmsd_threshold: float = 0.30) -> None:
        self.rmsd_threshold = rmsd_threshold

    def are_similar(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> bool:
        """Evaluates structural similarity based on PCA alignment and greedy matching."""
        rmsd_val = self.compute_rmsd(sym_a, coords_a, sym_b, coords_b)
        return rmsd_val < self.rmsd_threshold

    def compute_rmsd(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> float:
        if len(sym_a) != len(sym_b) or set(sym_a) != set(sym_b):
            return float("inf")

        ca = coords_a - coords_a.mean(axis=0)
        cb = coords_b - coords_b.mean(axis=0)

        ca_aligned = self._principal_axis_align(ca)
        cb_aligned = self._principal_axis_align(cb)

        min_rmsd = float("inf")
        
        # Test 8 octant reflections for principal axes signs
        reflections = [
            np.array([1, 1, 1]), np.array([-1, 1, 1]), np.array([1, -1, 1]), np.array([1, 1, -1]),
            np.array([-1, -1, 1]), np.array([-1, 1, -1]), np.array([1, -1, -1]), np.array([-1, -1, -1])
        ]

        for ref in reflections:
            cb_ref = cb_aligned * ref
            perm = self._greedy_mapping(sym_a, ca_aligned, sym_b, cb_ref)
            if perm is not None:
                rmsd_current = self._kabsch_rmsd(ca_aligned, cb_ref[perm])
                if rmsd_current < min_rmsd:
                    min_rmsd = rmsd_current

        return min_rmsd

    @staticmethod
    def _principal_axis_align(coords: np.ndarray) -> np.ndarray:
        if len(coords) < 2:
            return coords
        cov_matrix = np.cov(coords.T)
        _, eigvecs = np.linalg.eigh(cov_matrix)
        # Sort eigenvectors by descending eigenvalues for consistent orientation
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        return coords @ eigvecs

    @staticmethod
    def _greedy_mapping(
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> list[int] | None:
        """Heuristic greedy matching per element. Faster than Hungarian algorithm."""
        perm: list[int | None] = [None] * len(sym_a)
        
        unique_elements = set(sym_a)
        for elem in unique_elements:
            idx_a = [i for i, s in enumerate(sym_a) if s == elem]
            idx_b = [i for i, s in enumerate(sym_b) if s == elem]
            
            if len(idx_a) != len(idx_b):
                return None

            sub_a = coords_a[idx_a]
            sub_b = coords_b[idx_b]
            
            dists = cdist(sub_a, sub_b, metric='sqeuclidean')
            
            assigned_b = set()
            for r in range(len(idx_a)):
                sorted_c_indices = np.argsort(dists[r])
                for c in sorted_c_indices:
                    if c not in assigned_b:
                        assigned_b.add(c)
                        perm[idx_a[r]] = idx_b[c]
                        break

        if None in perm:
            return None
        return perm  # type: ignore

    @staticmethod
    def _kabsch_rmsd(pa: np.ndarray, pb: np.ndarray) -> float:
        H = pb.T @ pa
        U, _, Vt = np.linalg.svd(H)
        sign_d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, sign_d])
        R = Vt.T @ D @ U.T
        diff = pa - pb @ R.T
        return float(np.sqrt((diff ** 2).sum() / len(pa)))


# ===========================================================================
# Section 3 : ExplorationQueue
# ===========================================================================

@dataclass
class ExplorationTask:
    node_id: int
    xyz_file: str
    afir_params: list[str]
    priority: float = 0.0
    metadata: dict = field(default_factory=dict)

class ExplorationQueue(ABC):
    """Abstract base class for priority-ordered exploration queues.

    To add a new exploration strategy, only :meth:`compute_priority` needs
    to be overridden.  :meth:`should_add` has a default implementation that
    uses the return value of ``compute_priority`` directly as an acceptance
    probability, but it can also be overridden when a different probabilistic
    model is desired.

    Examples
    --------
    Priority linearly proportional to energy difference::

        class LinearQueue(ExplorationQueue):
            def compute_priority(self, task):
                delta_e = task.metadata.get("delta_E_hartree", 0.0)
                return max(0.0, 1.0 - delta_e * 100)

    Random exploration (priority ignored)::

        class RandomQueue(ExplorationQueue):
            def compute_priority(self, task):
                return float(np.random.random())
    """

    def __init__(self, rng_seed: int = 42) -> None:
        self._tasks: list[ExplorationTask] = []
        self._submitted: set[tuple] = set()
        self._rng = np.random.default_rng(rng_seed)

    def push(self, task: ExplorationTask) -> bool:
        key = (task.node_id, tuple(task.afir_params))
        if key in self._submitted:
            return False

        task.priority = self.compute_priority(task)
        # _tasks is maintained in descending priority order. Since bisect
        # assumes ascending order, negate the key to find the correct
        # insertion index in O(log n).
        keys = [-t.priority for t in self._tasks]
        idx = bisect.bisect_right(keys, -task.priority)
        self._tasks.insert(idx, task)
        self._submitted.add(key)
        return True

    def pop(self) -> ExplorationTask | None:
        return self._tasks.pop(0) if self._tasks else None

    def should_add(self, node: EQNode, reference_energy: float, **kwargs) -> bool:
        """Decide probabilistically whether to enqueue a node.

        The return value of ``compute_priority`` (in the range 0–1) is used
        directly as the acceptance probability.  Override with ``return True``
        for deterministic acceptance of all nodes, or override with a custom
        probabilistic model as needed.
        """
        dummy_task = ExplorationTask(
            node_id=node.node_id,
            xyz_file=node.xyz_file,
            afir_params=[],
            metadata={
                "delta_E_hartree": (
                    node.energy - reference_energy
                    if node.energy is not None else 0.0
                ),
                "source_node_energy": node.energy,
            },
        )
        p = self.compute_priority(dummy_task)
        return bool(self._rng.random() < p)

    def export_queue_status(self) -> list[dict]:
        return [
            {
                "node_id": t.node_id,
                "priority": t.priority,
                "afir_params": t.afir_params,
            }
            for t in self._tasks
        ]

    def __len__(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def compute_priority(self, task: ExplorationTask) -> float:
        """Return the priority of a task as a float in the range 0–1.

        This value is used both for ordering tasks in the queue and as the
        acceptance probability in :meth:`should_add`.  Subclasses only need
        to implement this method to define a new exploration strategy.

        Parameters
        ----------
        task : ExplorationTask
            ``task.metadata["delta_E_hartree"]`` holds the energy difference
            from the reference energy in Hartree.

        Returns
        -------
        float
            0.0 (lowest priority) – 1.0 (highest priority).
        """


class BoltzmannQueue(ExplorationQueue):
    """Probabilistic exploration queue based on the Boltzmann distribution (default).

    Low-energy nodes are explored with high priority, while higher-energy
    nodes are accepted probabilistically according to the temperature
    parameter ``temperature_K``.
    """

    def __init__(self, temperature_K: float = 300.0, rng_seed: int = 42) -> None:
        super().__init__(rng_seed=rng_seed)
        self.temperature_K = temperature_K

    def compute_priority(self, task: ExplorationTask) -> float:
        """Return exp(-ΔE / k_B T) directly as the priority."""
        delta_e: float = task.metadata.get("delta_E_hartree", 0.0)
        if delta_e <= 0.0:
            return 1.0
        return min(1.0, float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K))))


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
        """
        Parameters
        ----------
        afir_gamma_kJmol : float
            AFIR push/pull strength [kJ/mol].  Positive values attract the
            selected atom pair (the usual SC-AFIR usage).
        max_pairs : int
            Maximum number of atom pairs sampled per call.
        dist_lower_ang / dist_upper_ang : float
            Distance window [Å] for candidate pair selection.
        rng_seed : int
            NumPy RNG seed for reproducibility.
        covalent_margin : float
            Pairs closer than ``covalent_margin * (r_i + r_j)`` are skipped
            (already bonded).
        active_atoms : list[int] | None
            1-based atom label numbers to restrict pair search.
            ``None`` (default) means all atoms are considered.
        include_negative_gamma : bool
            If ``True``, each selected pair also generates a repulsive
            perturbation with ``-afir_gamma_kJmol`` (negative gamma).
            Default is ``False`` (attractive direction only).
        """
        self.afir_gamma_kJmol = afir_gamma_kJmol
        self.max_pairs = max_pairs
        self.dist_lower_ang = dist_lower_ang
        self.dist_upper_ang = dist_upper_ang
        self.covalent_margin = covalent_margin
        self.active_atoms = set(active_atoms) if active_atoms is not None else None
        self.include_negative_gamma = include_negative_gamma
        self._rng = np.random.default_rng(rng_seed)

    def generate_afir_perturbations(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[list[str]]:
        """Return a list of AFIR parameter lists ready for AutoTSWorkflow step1.

        Each entry has the form ``[gamma_str, atom_i_1based, atom_j_1based]``.

        When ``include_negative_gamma`` is ``True``, every chosen pair is
        duplicated with a negated gamma value, so both attractive and repulsive
        directions are explored.  The total number of entries can therefore be
        up to ``2 * max_pairs``.
        """
        n = len(symbols)
        if n < 2:
            return []

        dmat = distance_matrix(coords)
        candidates: list[tuple[int, int]] = []

        # Build the pool of atom indices subject to active_atoms restriction.
        # active_atoms stores 1-based labels; convert to 0-based for indexing.
        if self.active_atoms is not None:
            atom_indices = [i for i in range(n) if (i + 1) in self.active_atoms]
        else:
            atom_indices = list(range(n))

        for idx, i in enumerate(atom_indices):
            for j in atom_indices[idx + 1:]:
                dist = dmat[i, j]
                if self.dist_lower_ang <= dist <= self.dist_upper_ang:
                    if covalent_radii_lib is not None:
                        try:
                            r_i = covalent_radii_lib(symbols[i]) * _BOHR2ANG
                            r_j = covalent_radii_lib(symbols[j]) * _BOHR2ANG
                            if dist <= self.covalent_margin * (r_i + r_j):
                                continue
                        except KeyError:
                            pass
                    candidates.append((i, j))

        if not candidates:
            return []

        n_sel = min(self.max_pairs, len(candidates))
        chosen = self._rng.choice(len(candidates), size=n_sel, replace=False)

        pos_gamma_str = f"{self.afir_gamma_kJmol:.6g}"
        neg_gamma_str = f"{-self.afir_gamma_kJmol:.6g}"

        result: list[list[str]] = []
        for idx in chosen:
            i, j = candidates[int(idx)]
            i1, j1 = str(i + 1), str(j + 1)
            # Attractive direction (positive gamma) — always included
            result.append([pos_gamma_str, i1, j1])
            # Repulsive direction (negative gamma) — only when requested
            if self.include_negative_gamma:
                result.append([neg_gamma_str, i1, j1])

        return result

    def enumerate_all_pairs(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[list[str]]:
        """Return ALL valid AFIR parameter lists for the given structure.

        Unlike :meth:`generate_afir_perturbations`, this method returns every
        candidate pair that satisfies the distance and covalent-bond filters,
        without random sub-sampling.  The result is deterministic and is used
        for exhaustive Boltzmann-driven exploration.

        Each entry has the form ``[gamma_str, atom_i_1based, atom_j_1based]``.
        When ``include_negative_gamma`` is ``True`` each pair is also given a
        repulsive variant.
        """
        n = len(symbols)
        if n < 2:
            return []

        dmat = distance_matrix(coords)

        if self.active_atoms is not None:
            atom_indices = [i for i in range(n) if (i + 1) in self.active_atoms]
        else:
            atom_indices = list(range(n))

        pos_gamma_str = f"{self.afir_gamma_kJmol:.6g}"
        neg_gamma_str = f"{-self.afir_gamma_kJmol:.6g}"

        result: list[list[str]] = []
        for idx, i in enumerate(atom_indices):
            for j in atom_indices[idx + 1:]:
                dist = dmat[i, j]
                if not (self.dist_lower_ang <= dist <= self.dist_upper_ang):
                    continue
                if covalent_radii_lib is not None:
                    try:
                        r_i = covalent_radii_lib(symbols[i]) * _BOHR2ANG
                        r_j = covalent_radii_lib(symbols[j]) * _BOHR2ANG
                        if dist <= self.covalent_margin * (r_i + r_j):
                            continue
                    except KeyError:
                        pass
                i1, j1 = str(i + 1), str(j + 1)
                result.append([pos_gamma_str, i1, j1])
                if self.include_negative_gamma:
                    result.append([neg_gamma_str, i1, j1])

        return result


# ===========================================================================
# Section 4b : ExploredPairsLog  (persistent record of explored EQ/pair combos)
# ===========================================================================

class ExploredPairsLog:
    """Persistent text-file registry of already-explored (EQ, AFIR-pair) combinations.

    File format — one record per line::

        EQ{node_id:06d} {gamma} {atom_i} {atom_j}

    The log is append-only during a run.  On resume the entire file is read
    back into an in-memory set so that look-ups remain O(1).
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._explored: set[tuple] = set()
        if os.path.isfile(filepath):
            self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_explored(self, node_id: int, afir_params: list[str]) -> bool:
        """Return ``True`` if this (node, afir_params) combination has been logged."""
        return self._make_key(node_id, afir_params) in self._explored

    def mark_explored(self, node_id: int, afir_params: list[str]) -> None:
        """Record the combination as explored (in memory and on disk)."""
        key = self._make_key(node_id, afir_params)
        if key in self._explored:
            return
        self._explored.add(key)
        with open(self._filepath, "a", encoding="utf-8") as fh:
            params_str = " ".join(afir_params)
            fh.write(f"EQ{node_id:06d} {params_str}\n")

    def __len__(self) -> int:
        return len(self._explored)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(node_id: int, afir_params: list[str]) -> tuple:
        return (node_id, tuple(afir_params))

    def _load(self) -> None:
        """Parse an existing log file into the in-memory set."""
        with open(self._filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # parts[0] is "EQ{node_id}", rest are afir_params tokens
                try:
                    node_id = int(parts[0].lstrip("EQ"))
                    afir_params = parts[1:]
                    self._explored.add(self._make_key(node_id, afir_params))
                except (ValueError, IndexError):
                    continue


# ===========================================================================
# Section 5 : Graph data model (NetworkX Removed)
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

    @property
    def has_real_energy(self) -> bool:
        return self.energy is not None

    def to_dict(self) -> dict:
        return {
            "node_id":        self.node_id,
            "xyz_file":       self.xyz_file,
            "energy_hartree": self.energy,
            "source_run_dir": self.source_run_dir,
            **self.extra,
        }

@dataclass
class TSEdge:
    edge_id: int
    node_id_1: int
    node_id_2: int
    ts_xyz_file: str | None
    ts_energy: float
    barrier_fwd: float | None = None
    barrier_rev: float | None = None
    source_run_dir: str = ""
    duplicate_of: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "edge_id":           self.edge_id,
            "node_id_1":         self.node_id_1,
            "node_id_2":         self.node_id_2,
            "ts_xyz_file":       self.ts_xyz_file,
            "ts_energy_hartree": self.ts_energy,
            "barrier_fwd_kcal":  self.barrier_fwd,
            "barrier_rev_kcal":  self.barrier_rev,
            "source_run_dir":    self.source_run_dir,
            **self.extra,
        }
        if self.duplicate_of is not None:
            data["duplicate_of"] = self.duplicate_of
        return data

class NetworkGraph:
    def __init__(self) -> None:
        self._nodes: dict[int, EQNode] = {}
        self._edges: dict[int, TSEdge] = {}
        self._node_counter: int = 0
        self._edge_counter: int = 0

    def add_node(self, node: EQNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: int) -> EQNode | None:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[EQNode]:
        return list(self._nodes.values())

    def next_node_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def add_edge(self, edge: TSEdge) -> None:
        self._edges[edge.edge_id] = edge

    def all_edges(self) -> list[TSEdge]:
        return list(self._edges.values())

    def next_edge_id(self) -> int:
        eid = self._edge_counter
        self._edge_counter += 1
        return eid

    def reference_energy(self) -> float | None:
        real_energies = [n.energy for n in self._nodes.values() if n.has_real_energy]
        return min(real_energies) if real_energies else None

    def save(self, filepath: str) -> None:
        data = {
            "nodes":    [n.to_dict() for n in self._nodes.values()],
            "edges":    [e.to_dict() for e in self._edges.values()],
            "metadata": {
                "n_nodes": len(self._nodes),
                "n_edges": len(self._edges),
            },
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._node_counter = 0
        self._edge_counter = 0

        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for nd in data.get("nodes", []):
            node = EQNode(
                node_id=nd["node_id"],
                xyz_file=nd["xyz_file"],
                energy=nd["energy_hartree"],
                symbols=[],
                source_run_dir=nd.get("source_run_dir", ""),
            )
            if os.path.isfile(node.xyz_file):
                try:
                    node.symbols, node.coords = parse_xyz(node.xyz_file)
                except Exception as exc:
                    logger.warning("Could not re-read %s: %s", node.xyz_file, exc)
            self.add_node(node)

        for ed in data.get("edges", []):
            edge = TSEdge(
                edge_id=ed["edge_id"],
                node_id_1=ed["node_id_1"],
                node_id_2=ed["node_id_2"],
                ts_xyz_file=ed.get("ts_xyz_file"),
                ts_energy=ed["ts_energy_hartree"],
                barrier_fwd=ed.get("barrier_fwd_kcal"),
                barrier_rev=ed.get("barrier_rev_kcal"),
                source_run_dir=ed.get("source_run_dir", ""),
                duplicate_of=ed.get("duplicate_of")
            )
            self.add_edge(edge)

        if self._nodes:
            self._node_counter = max(self._nodes) + 1
        if self._edges:
            self._edge_counter = max(self._edges) + 1

    def summary(self) -> str:
        lines = [f"[NetworkGraph]  nodes={len(self._nodes)}  edges={len(self._edges)}"]
        ref = self.reference_energy()

        for node in sorted(self._nodes.values(), key=lambda n: (n.energy is None, n.energy)):
            if node.energy is not None and ref is not None:
                rel = (node.energy - ref) * HARTREE_TO_KCALMOL
                e_str = f"{node.energy:+.8f} Ha  (+{rel:.2f} kcal/mol)"
            elif node.energy is not None:
                e_str = f"{node.energy:+.8f} Ha"
            else:
                e_str = "energy unknown"
            lines.append(f"  EQ{node.node_id:06d}: {e_str}  [{node.xyz_file}]")

        for edge in self._edges.values():
            fwd = f"{edge.barrier_fwd:.2f}" if edge.barrier_fwd is not None else "N/A"
            rev = f"{edge.barrier_rev:.2f}" if edge.barrier_rev is not None else "N/A"
            dup_info = f" (Dup of TS{edge.duplicate_of:06d})" if edge.duplicate_of is not None else ""
            lines.append(
                f"  TS{edge.edge_id:06d}{dup_info}: "
                f"EQ{edge.node_id_1} -- EQ{edge.node_id_2}  "
                f"Ea(fwd)={fwd} kcal/mol  Ea(rev)={rev} kcal/mol"
            )
        return "\n".join(lines)


# ===========================================================================
# Section 6 : ProfileParser
# ===========================================================================

class ProfileParser:
    def parse(self, profile_dir: str) -> dict | None:
        ep1 = os.path.join(profile_dir, "endpoint_1_opt.xyz")
        ep2 = os.path.join(profile_dir, "endpoint_2_opt.xyz")
        ts_matches = glob.glob(os.path.join(profile_dir, "*_ts_final.xyz"))
        txt_path   = os.path.join(profile_dir, "energy_profile.txt")

        if not os.path.isfile(ep1) or not os.path.isfile(ep2):
            return None
        if not ts_matches:
            return None

        ts_file = ts_matches[0]
        energies = self._parse_energy_txt(txt_path)

        ts_e   = energies.get("TS")
        ep1_e  = energies.get("Endpoint_1")
        ep2_e  = energies.get("Endpoint_2")

        def _barrier(e_eq: float | None, e_ts: float | None) -> float | None:
            if e_eq is None or e_ts is None:
                return None
            return (e_ts - e_eq) * HARTREE_TO_KCALMOL

        return {
            "ts_xyz_file":       ts_file,
            "ts_energy":         ts_e,
            "endpoint_1_xyz":    ep1,
            "endpoint_2_xyz":    ep2,
            "endpoint_1_energy": ep1_e,
            "endpoint_2_energy": ep2_e,
            "barrier_fwd":       _barrier(ep1_e, ts_e),
            "barrier_rev":       _barrier(ep2_e, ts_e),
        }

    @staticmethod
    def _parse_energy_txt(txt_path: str) -> dict:
        result: dict[str, float | None] = {"TS": None, "Endpoint_1": None, "Endpoint_2": None}
        if not os.path.isfile(txt_path):
            return result

        with open(txt_path, "r") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) < 3:
                    continue
                key = parts[0]
                if key not in result:
                    continue
                try:
                    result[key] = float(parts[2])
                except ValueError:
                    pass
        return result


# ===========================================================================
# Section 7 : ReactionNetworkMapper
# ===========================================================================

class ReactionNetworkMapper:
    def __init__(
        self,
        base_config: dict,
        queue: ExplorationQueue | None = None,
        structure_checker: StructureChecker | None = None,
        perturbation_generator: PerturbationGenerator | None = None,
        output_dir: str = "mapper_output",
        graph_json: str = "reaction_network.json",
        max_iterations: int = 50,
        resume: bool = False,
        temperature_K: float = 300.0,
        max_resample_attempts: int = 10000,
    ) -> None:
        self.base_config = base_config
        self.queue    = queue if queue is not None else BoltzmannQueue(temperature_K=temperature_K)
        self.checker  = structure_checker if structure_checker is not None else StructureChecker()
        self.perturber = perturbation_generator if perturbation_generator is not None else PerturbationGenerator()
        self.output_dir = os.path.abspath(output_dir)
        self.graph_json_path = os.path.join(self.output_dir, graph_json)
        self.max_iterations = max_iterations
        self.resume = resume
        self.temperature_K = temperature_K
        # Maximum number of (node, pair) re-draws before giving up when all
        # combinations appear to have been explored.
        self.max_resample_attempts = max_resample_attempts

        self.graph = NetworkGraph()
        self._iteration: int = 0
        self._rng = np.random.default_rng(42)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "nodes"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "runs"), exist_ok=True)

        # Persistent log of explored (EQ-node, AFIR-pair) combinations.
        self._explored_log = ExploredPairsLog(
            os.path.join(self.output_dir, "explored_pairs.txt")
        )

        self.energy_tolerance = 1.0 / HARTREE_TO_KCALMOL

    def run(self) -> None:
        if self.resume and os.path.isfile(self.graph_json_path):
            self.graph.load(self.graph_json_path)
            self._requeue_all_nodes()
        else:
            self._register_seed_structure()

        history_log = os.path.join(self.output_dir, "exploration_history.log")
        priority_log = os.path.join(self.output_dir, "queue_priority.log")

        while True:
            if os.path.isfile(os.path.join(self.output_dir, "stop.txt")):
                logger.info("stop.txt detected in output_dir. Stopping.")
                break

            if self.max_iterations > 0 and self._iteration >= self.max_iterations:
                break

            # Primary path: consume pre-queued tasks.
            task = self.queue.pop()

            # Fallback path: queue is exhausted — resample via Boltzmann until
            # an unexplored (EQ, pair) combination is found.
            if task is None:
                task = self._boltzmann_sample_task()
                if task is None:
                    logger.info(
                        "All (EQ, atom-pair) combinations appear to have been "
                        "explored and no further task could be drawn.  Stopping."
                    )
                    break

            # Mark this (node, pair) combination in the persistent log before
            # running the expensive AutoTS step so that a crash does not cause
            # it to be re-run on resume.
            self._explored_log.mark_explored(task.node_id, task.afir_params)

            self._iteration += 1

            with open(history_log, "a", encoding="utf-8") as fh:
                fh.write(
                    f"Iter: {self._iteration:06d} | Node: EQ{task.node_id:06d} "
                    f"| Priority: {task.priority:.6f} | AFIR: {task.afir_params}\n"
                )

            run_dir = self._make_run_dir(task)
            try:
                profile_dirs = self._run_autots(task, run_dir)
            except Exception as e:
                logger.error(f"AutoTS failed for run {run_dir}: {e}")
                self._save_run_metadata(run_dir, task, status="FAILED", profile_dirs=[])
                self.graph.save(self.graph_json_path)
                self._write_priority_log(priority_log)
                continue

            for pdir in profile_dirs:
                self._process_profile(pdir, run_dir)

            self._save_run_metadata(run_dir, task, status="DONE", profile_dirs=profile_dirs)
            self.graph.save(self.graph_json_path)
            # Write after new nodes and tasks have been registered so the log
            # reflects the state produced by this iteration.
            self._write_priority_log(priority_log)

        self.graph.save(self.graph_json_path)

    def _boltzmann_sample_task(self) -> ExplorationTask | None:
        """Draw an unexplored (EQ-node, AFIR-pair) combination via Boltzmann sampling.

        Node selection probability is proportional to the Boltzmann weight
        ``exp(-ΔE / k_B T)``, where ΔE is the node energy relative to the
        lowest-energy node in the current graph.

        After a node is selected, ALL valid atom pairs for that node (as
        enumerated by :meth:`PerturbationGenerator.enumerate_all_pairs`) are
        considered.  One pair is chosen uniformly at random.  If that
        (node, pair) combination is already recorded in the persistent
        :attr:`_explored_log`, the draw is rejected and repeated from
        scratch up to :attr:`max_resample_attempts` times.

        Returns
        -------
        ExplorationTask | None
            A fresh, unexplored task, or ``None`` if no unexplored combination
            could be found within ``max_resample_attempts`` attempts.
        """
        nodes = self.graph.all_nodes()
        if not nodes:
            return None

        ref_e = self.graph.reference_energy()

        # Compute Boltzmann weights for every node.
        raw_weights: list[float] = []
        for node in nodes:
            if node.energy is not None and ref_e is not None:
                delta_e = node.energy - ref_e
                w = float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K)))
            else:
                # Nodes without an energy get weight 1 (treated as reference).
                w = 1.0
            raw_weights.append(max(w, 1e-300))  # avoid exact-zero weights

        weights_arr = np.array(raw_weights, dtype=float)
        weights_arr /= weights_arr.sum()

        for attempt in range(self.max_resample_attempts):
            # --- 1. Boltzmann-sample a node ---
            node_idx = int(self._rng.choice(len(nodes), p=weights_arr))
            node = nodes[node_idx]

            # --- 2. Enumerate all valid pairs for this node ---
            all_pairs = self.perturber.enumerate_all_pairs(node.symbols, node.coords)
            if not all_pairs:
                # This node has no valid pairs at all; avoid infinite loops by
                # reducing its weight to zero and resampling.
                weights_arr[node_idx] = 0.0
                total = weights_arr.sum()
                if total <= 0.0:
                    break
                weights_arr /= total
                continue

            # --- 3. Pick a pair uniformly at random ---
            pair_idx = int(self._rng.integers(0, len(all_pairs)))
            afir_params = all_pairs[pair_idx]

            # --- 4. Reject if already explored ---
            if self._explored_log.is_explored(node.node_id, afir_params):
                logger.debug(
                    "Boltzmann sample attempt %d: EQ%06d %s already explored — resampling.",
                    attempt + 1,
                    node.node_id,
                    afir_params,
                )
                continue

            # --- 5. Also skip pairs that are already in the pre-queued set ---
            key = (node.node_id, tuple(afir_params))
            # (The queue's _submitted set is not exposed, so we rely solely on
            #  the persistent log which is written before each AutoTS run.)

            # --- 6. Build and return the task ---
            delta_e = (
                (node.energy - ref_e)
                if (node.energy is not None and ref_e is not None)
                else 0.0
            )
            task = ExplorationTask(
                node_id=node.node_id,
                xyz_file=node.xyz_file,
                afir_params=afir_params,
                metadata={
                    "delta_E_hartree":    delta_e,
                    "source_node_energy": node.energy,
                    "sampled_by":         "boltzmann_fallback",
                },
            )
            # Assign priority using the queue's own formula for consistency.
            task.priority = self.queue.compute_priority(task)

            logger.info(
                "Boltzmann fallback (attempt %d): selected EQ%06d %s "
                "(priority=%.4f).",
                attempt + 1,
                node.node_id,
                afir_params,
                task.priority,
            )
            return task

        logger.warning(
            "Could not find an unexplored (EQ, pair) combination after %d attempts.",
            self.max_resample_attempts,
        )
        return None

    def _register_seed_structure(self) -> None:
        seed_xyz = os.path.abspath(self.base_config["initial_mol_file"])
        node_id = self.graph.next_node_id()

        optimized_xyz_path, energy = self._run_initial_optimization(seed_xyz)

        saved_xyz = self._persist_node_xyz(optimized_xyz_path, node_id)

        try:
            symbols, coords = parse_xyz(saved_xyz)
        except Exception as e:
            logger.error(f"Failed to parse initial structure: {e}")
            sys.exit(1)

        node = EQNode(
            node_id=node_id,
            xyz_file=saved_xyz,
            energy=energy,
            symbols=symbols,
            coords=coords,
            source_run_dir="initial_optimization",
        )
        self.graph.add_node(node)
        self.graph.save(self.graph_json_path)   # Persist immediately after EQ0 is registered
        self._enqueue_perturbations(node, force_add=True)

    def _run_initial_optimization(
        self, seed_xyz: str
    ) -> tuple[str, float | None]:
        """Run geometry optimisation on the seed structure.

        Returns
        -------
        optimized_xyz_path : str
            Path to the optimised XYZ file (falls back to ``seed_xyz`` on failure).
        energy : float | None
            SCF energy in Hartree, or ``None`` if unavailable.
        """
        if OptimizationJob is None:
            logger.warning("OptimizationJob not available; skipping initial optimization.")
            return seed_xyz, None

        try:
            logger.info("Running initial structure optimization via OptimizationJob...")

            # Build keyword arguments, stripping AFIR-specific keys.
            _AFIR_KEYS = {"manual_AFIR", "afir_gamma_kJmol", "max_pairs"}
            step1_config = self.base_config.get("step1_settings", {})
            opt_kwargs = {k: v for k, v in step1_config.items() if k not in _AFIR_KEYS}

            if "software_path_file_source" in self.base_config:
                opt_kwargs["software_path_file"] = self.base_config["software_path_file_source"]

            opt_kwargs["run_type"] = "opt"

            # OptimizationJob expects a path relative to the current working
            # directory, so convert the absolute seed path accordingly.
            seed_xyz_rel = os.path.relpath(seed_xyz)
            opt_job = OptimizationJob(seed_xyz_rel)
            if opt_kwargs:
                opt_job.set_options(**opt_kwargs)

            opt_job.run()

            # --- Locate optimised geometry file ---
            optimized_xyz_path = seed_xyz
            potential_opt_files = glob.glob("*_opt.xyz")
            if potential_opt_files:
                optimized_xyz_path = os.path.abspath(potential_opt_files[0])

            # --- Extract energy via get_results() -> Optimize instance ---
            # OptimizationJob itself holds no energy; it lives on the internal
            # multioptpy.optimization.Optimize object returned by get_results().
            energy: float | None = None
            optimizer = opt_job.get_results()
            if optimizer is not None:
                for attr in ("energy", "final_energy", "scf_energy", "result_energy",
                             "optimized_energy", "last_energy", "minimum_energy"):
                    raw = getattr(optimizer, attr, None)
                    if raw is not None:
                        try:
                            energy = float(raw)
                            logger.info(
                                f"Initial optimization energy (optimizer.{attr}): {energy:.10f} Ha"
                            )
                            break
                        except (TypeError, ValueError):
                            pass

            if energy is None:
                logger.warning(
                    "Could not retrieve energy from optimizer instance. "
                    "Node 0 will have energy=None. "
                    "Check multioptpy.optimization.Optimize for the correct energy attribute name."
                )
                # Log all float attributes on the optimizer to assist diagnosis.
                if optimizer is not None:
                    float_attrs = {k: v for k, v in vars(optimizer).items()
                                   if isinstance(v, float)}
                    if float_attrs:
                        logger.warning(f"Float attributes on optimizer: {float_attrs}")

            return optimized_xyz_path, energy

        except Exception as e:
            logger.error(
                f"Initial optimization failed: {e}. Falling back to unoptimized geometry."
            )
            traceback.print_exc()
            return seed_xyz, None

    def _write_priority_log(self, priority_log: str) -> None:
        """Overwrite the priority log with the current queue state.

        Called at the end of each iteration, after new nodes and tasks have
        been registered, so that tasks discovered in that iteration are
        correctly reflected with their computed priorities.
        """
        ref_e = self.graph.reference_energy()
        with open(priority_log, "w", encoding="utf-8") as fh:
            fh.write(
                f"# Iter {self._iteration:06d} | "
                f"queued={len(self.queue)} | "
                f"ref_energy={ref_e:.10f} Ha\n"
                if ref_e is not None else
                f"# Iter {self._iteration:06d} | queued={len(self.queue)} | ref_energy=N/A\n"
            )
            for item in self.queue.export_queue_status():
                fh.write(
                    f"Node: EQ{item['node_id']:06d} | "
                    f"Priority: {item['priority']:.6f} | "
                    f"AFIR: {item['afir_params']}\n"
                )

    def _requeue_all_nodes(self) -> None:
        for node in self.graph.all_nodes():
            self._enqueue_perturbations(node, force_add=True)

    def _make_run_dir(self, task: ExplorationTask) -> str:
        name = f"run_{self._iteration:06d}_node{task.node_id}"
        run_dir = os.path.join(self.output_dir, "runs", name)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "autots_workspace"), exist_ok=True)
        self._save_input_txt(run_dir, task)
        return run_dir

    def _save_input_txt(self, run_dir: str, task: ExplorationTask) -> None:
        txt_path = os.path.join(run_dir, "input.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"node_id = {task.node_id}\n")

    def _save_run_metadata(
        self,
        run_dir: str,
        task: ExplorationTask,
        status: str,
        profile_dirs: list[str],
    ) -> None:
        info = {
            "iteration":    self._iteration,
            "node_id":      task.node_id,
            "afir_params":  task.afir_params,
            "priority":     task.priority,
            "status":       status,
            "profile_dirs": profile_dirs,
        }
        info_path = os.path.join(run_dir, "run_info.json")
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

    def _run_autots(self, task: ExplorationTask, run_dir: str) -> list[str]:
        workspace = os.path.join(run_dir, "autots_workspace")

        config = copy.deepcopy(self.base_config)
        config["initial_mol_file"] = task.xyz_file
        config["work_dir"]         = workspace
        config.setdefault("step1_settings", {})
        config["step1_settings"]["manual_AFIR"] = task.afir_params
        config["run_step4"]     = True

        original_cwd = os.getcwd()
        try:
            os.chdir(run_dir)
            workflow = AutoTSWorkflow(config=config)
            workflow.run_workflow()
        finally:
            os.chdir(original_cwd)

        pattern = os.path.join(workspace, "**", "*_Step4_Profile")
        profiles = sorted(glob.glob(pattern, recursive=True))
        return profiles

    def _process_profile(self, profile_dir: str, run_dir: str) -> None:
        parser = ProfileParser()
        result = parser.parse(profile_dir)
        if result is None:
            return

        ts_energy = result["ts_energy"] if result["ts_energy"] is not None else 0.0

        node_id_1 = self._find_or_register_node(
            xyz_file=result["endpoint_1_xyz"],
            energy=result["endpoint_1_energy"],
            run_dir=run_dir,
        )
        node_id_2 = self._find_or_register_node(
            xyz_file=result["endpoint_2_xyz"],
            energy=result["endpoint_2_energy"],
            run_dir=run_dir,
        )

        if node_id_1 is None or node_id_2 is None or node_id_1 == node_id_2:
            return

        edge_id = self.graph.next_edge_id()
        
        # TS Identity Check
        ts_duplicate_of = None
        try:
            ts_sym, ts_coords = parse_xyz(result["ts_xyz_file"])
            for existing_edge in self.graph.all_edges():
                if existing_edge.ts_xyz_file and os.path.isfile(existing_edge.ts_xyz_file):
                    ex_sym, ex_coords = parse_xyz(existing_edge.ts_xyz_file)
                    # Energy filter pre-check
                    if abs(ts_energy - existing_edge.ts_energy) < self.energy_tolerance:
                        if self.checker.are_similar(ts_sym, ts_coords, ex_sym, ex_coords):
                            ts_duplicate_of = existing_edge.edge_id
                            break
        except Exception:
            pass
        
        if ts_duplicate_of is not None:
            # Duplicate TS detected: identity check succeeded but the transition
            # state already exists in the network.  Do NOT add a redundant edge
            # to the graph or the reaction_network.json.
            logger.info(
                "TS for edge %d is a duplicate of existing edge %d — skipping.",
                edge_id,
                ts_duplicate_of,
            )
            return

        saved_ts_xyz = self._persist_ts_xyz(result["ts_xyz_file"], edge_id)

        edge = TSEdge(
            edge_id=edge_id,
            node_id_1=node_id_1,
            node_id_2=node_id_2,
            ts_xyz_file=saved_ts_xyz,
            ts_energy=ts_energy,
            barrier_fwd=result["barrier_fwd"],
            barrier_rev=result["barrier_rev"],
            source_run_dir=run_dir,
            duplicate_of=None,
        )
        self.graph.add_edge(edge)

    def _find_or_register_node(
        self,
        xyz_file: str,
        energy: float | None,
        run_dir: str,
    ) -> int | None:
        try:
            symbols, coords = parse_xyz(xyz_file)
        except Exception:
            return None

        for existing in self.graph.all_nodes():
            if existing.coords.size == 0:
                continue
                
            if energy is not None and existing.energy is not None:
                if abs(energy - existing.energy) > self.energy_tolerance:
                    continue

            if self.checker.are_similar(symbols, coords, existing.symbols, existing.coords):
                return existing.node_id

        node_id   = self.graph.next_node_id()
        saved_xyz = self._persist_node_xyz(xyz_file, node_id)
        new_node  = EQNode(
            node_id=node_id,
            xyz_file=saved_xyz,
            energy=energy,
            symbols=symbols,
            coords=coords,
            source_run_dir=run_dir,
        )
        self.graph.add_node(new_node)

        ref_e = self.graph.reference_energy()
        if ref_e is None or new_node.energy is None:
            self._enqueue_perturbations(new_node, force_add=True)
        else:
            self._enqueue_perturbations(new_node, force_add=False)

        return node_id

    def _enqueue_perturbations(self, node: EQNode, force_add: bool = False) -> None:
        if node.coords.size == 0:
            return

        perturbations = self.perturber.generate_afir_perturbations(node.symbols, node.coords)
        if not perturbations:
            return

        ref_e = self.graph.reference_energy()

        if not force_add and node.energy is not None and ref_e is not None:
            accepted = self.queue.should_add(node, ref_e)
        else:
            accepted = True

        if not accepted:
            return

        for afir_params in perturbations:
            delta_e = ((node.energy - ref_e) if (node.energy is not None and ref_e is not None) else 0.0)
            task = ExplorationTask(
                node_id=node.node_id,
                xyz_file=node.xyz_file,
                afir_params=afir_params,
                metadata={
                    "delta_E_hartree":    delta_e,
                    "source_node_energy": node.energy,
                },
            )
            if self.queue.push(task):
                # Mirror every successfully queued task in the persistent log so
                # that the Boltzmann fallback sampler never re-draws the same pair.
                self._explored_log.mark_explored(node.node_id, afir_params)

    def _persist_node_xyz(self, src_xyz: str, node_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"EQ{node_id:06d}.xyz")
        try:
            if os.path.abspath(src_xyz) != os.path.abspath(dst):
                shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception:
            return os.path.abspath(src_xyz)

    def _persist_ts_xyz(self, src_xyz: str, edge_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"TS{edge_id:06d}.xyz")
        try:
            if os.path.abspath(src_xyz) != os.path.abspath(dst):
                shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception:
            return os.path.abspath(src_xyz)