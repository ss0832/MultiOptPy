"""
mapper.py - Chemical Reaction Network Mapper
============================================

Autonomously maps a chemical reaction network using AutoTSWorkflow
and external optimization utilities.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import traceback
import atexit
import glob
import heapq
import json
import logging
import multiprocessing
import os
import re
import copy
import shutil
import sys
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
    # logger is safe to use before basicConfig: the root logger buffers
    # records until a handler is attached, so no output is lost.
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
# Module-level worker function for subprocess isolation
# ---------------------------------------------------------------------------
# Must be defined at module level (not as a method or closure) so that it
# can be pickled by the ``spawn`` start method on all platforms.
#
# Design rationale
# ----------------
# AutoTSWorkflow calls ``os.chdir(work_dir)`` inside ``setup_workspace()``.
# Running it in a *spawned* child process confines that CWD change to the
# child's address space, so the mapper's main process (and any other child
# that may be running concurrently) is never affected.
#
# The worker:
#   1. Changes to ``run_dir`` so that any relative paths emitted by
#      OptimizationJob / NEBJob resolve to the correct location.
#   2. Instantiates AutoTSWorkflow with the provided config and calls
#      ``run_workflow()``.
#   3. Returns the sorted list of Step-4 profile directories found under
#      ``workspace``.  The list is pickled and sent back to the parent
#      process via the standard ``concurrent.futures`` result channel.
#
# Exceptions propagate normally: if ``run_workflow()`` raises, the
# ``Future`` will hold the exception and ``_run_autots`` will re-raise it
# in the parent process exactly as before.

def _autots_worker(config: dict, run_dir: str, workspace: str) -> list[str]:
    """Isolated AutoTSWorkflow execution — intended for spawned subprocesses.

    Parameters
    ----------
    config:
        Fully-resolved configuration dict (deepcopy of ``base_config`` with
        ``initial_mol_file``, ``work_dir``, ``manual_AFIR``, and
        ``run_step4`` already applied).
    run_dir:
        Absolute path to the per-task run directory.  The worker changes
        into this directory before starting the workflow so that any
        relative-path output from AutoTS lands inside ``run_dir``.
    workspace:
        Absolute path to the ``autots_workspace`` sub-directory; used to
        glob for ``*_Step4_Profile`` directories after the workflow exits.

    Returns
    -------
    list[str]
        Sorted absolute paths of all Step-4 profile directories produced by
        the workflow.  May be empty when the workflow produced no profiles.
    """
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
    """Thin wrapper around :func:`_autots_worker` for use with
    :class:`multiprocessing.Process`.

    The return value (or raised exception) is transmitted back to the
    parent process via *result_queue* as a ``(tag, payload)`` tuple:

    * ``("ok", profiles)`` on success.
    * ``("err", traceback_str)`` on failure (full traceback as a string).

    This avoids the per-call ``ProcessPoolExecutor`` setup/teardown
    overhead incurred by the sequential (``n_parallel=1``) code path.
    """
    try:
        profiles = _autots_worker(config, run_dir, workspace)
        result_queue.put(("ok", profiles))
    except Exception as exc:  # noqa: BLE001
        # Serialise the full traceback as a string so it crosses the process
        # boundary safely (some exception types are not picklable).
        result_queue.put(("err", traceback.format_exc()))


logger = logging.getLogger(__name__)

# Module-level physical constants
HARTREE_TO_KCALMOL: float = 627.509474
K_B_HARTREE: float = 3.166811563e-6


# ===========================================================================
# Section 1 : XYZ Utilities
# ===========================================================================


# Pre-compiled XYZ atom-line pattern.  Placed at module level so the regex is
# compiled exactly once rather than on every parse_xyz() call.
_XYZ_PATTERN: re.Pattern = re.compile(
    r"\s*([A-Za-z]+)\s+"
    r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)\s*"
)


def get_pattern_xyz() -> re.Pattern:
    """Return the pre-compiled XYZ atom-line regex.

    Kept for backward compatibility; callers should prefer ``_XYZ_PATTERN``
    directly.  The regex is no longer recompiled on each call.
    """
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

        # Skip comment line
        next(fh, None)

        symbols: list[str] = []
        coords_raw: list[list[float]] = []

        # line number of the first atom line = header + comment + 1
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
    # cdist avoids the (N,N,3) intermediate array produced by manual broadcasting.
    return cdist(coords, coords)


# ===========================================================================
# Section 2a : StructureChecker
# ===========================================================================


class StructureChecker:
    """Check whether two molecular structures are identical up to rotation and
    atom-index permutation.

    The algorithm proceeds in four stages:

    1. **Sign-flip candidates** — four proper rotations arising from PCA
       eigenvector sign ambiguity.  Sufficient for all non-degenerate molecules.
    2. **Degeneracy check** — detects whether adjacent PCA eigenvalue pairs are
       degenerate in either structure.  Skips heavier stages when unnecessary.
    3. **Coarse planar/SO(3) grid** — samples the degenerate subspace with a
       moderate number of rotation candidates.
    4. **Fine SO(3) grid** — applied only for fully degenerate (near-spherical)
       structures.

    Each candidate rotation is combined with the optimal atom mapping found by
    the Hungarian algorithm, and the residual RMSD is computed via Kabsch.

    Chirality handling
    ------------------
    ``_kabsch_rmsd`` returns ``inf`` whenever the optimal superposition requires
    an improper rotation (det = -1), i.e. a mirror reflection.  This ensures
    that enantiomers are never declared identical regardless of how flat the
    molecule is.

    Known limitation — pseudosymmetric enantiomers
    -----------------------------------------------
    When a molecule contains multiple atoms of the same element, the Hungarian
    algorithm may discover an atom-index permutation that effectively implements
    a mirror reflection without triggering the d < 0 guard: after re-indexing,
    the Kabsch step sees a proper rotation (d = +1) and may return a small RMSD.

    **Resolution via BondTopologyChecker**

    Passing a ``BondTopologyChecker`` instance to the constructor enables
    *bond-environment-aware grouping*.  Instead of grouping atoms by element
    symbol alone, atoms are grouped by ``(element, bond_environment)`` where
    the bond environment is the sorted tuple of neighbour element symbols
    returned by ``BondTopologyChecker.atom_environments()``.

    This constrains the Hungarian algorithm to only permute atoms that are
    chemically equivalent (same element **and** same bonded neighbours),
    preventing swaps that would implement a reflection.

    **When this resolves the limitation**

    For typical chiral molecules the chiral centre has four distinct
    substituents, giving each substituent-atom a unique bond environment.
    The grouping then reduces to a single-atom group per environment, forcing
    the permutation to the chemically correct one-to-one mapping.  Under this
    unique mapping the Kabsch step correctly finds d = -1 and returns ``inf``.

    **Residual limitation**

    When all atoms of a given element share identical bond environments (e.g. a
    minimal C₂-symmetric fragment such as ["C","C","O","O"] where every C is
    bonded only to one O), the grouping remains ambiguous and the atom swap is
    still possible in principle.  In practice the RMSD after such a swap is
    large enough to exceed the threshold for real molecules, but it is not
    guaranteed by the algorithm.  Complete resolution would require
    higher-order graph invariants (e.g. Morgan-algorithm fingerprints) or
    explicit CIP stereochemistry, which are outside the scope of this
    geometry-based checker.

    Parameters
    ----------
    rmsd_threshold : float
        Structures with minimum RMSD below this value are declared identical.
    bond_checker : BondTopologyChecker or None
        When provided, atom groupings are refined by bond environment to
        prevent chemically invalid permutations.  Recommended for chiral
        molecule comparison.
    """

    _DEGENERACY_REL_TOL: float = 0.02

    # Raised from 1e-10 to 1e-4 so that near-zero eigenvalue pairs in
    # extremely flat molecules are still identified as degenerate.
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

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

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
        """Return the minimum RMSD between the two structures over all proper
        rotations and atom-index permutations.

        Returns ``inf`` if the element compositions differ or if no valid
        proper-rotation superposition exists (i.e. the structures are
        enantiomers).
        """
        if len(sym_a) != len(sym_b) or Counter(sym_a) != Counter(sym_b):
            return float("inf")

        ca = coords_a - coords_a.mean(axis=0)
        cb = coords_b - coords_b.mean(axis=0)

        ca_pca, eigvals_a = self._pca_align(ca)
        cb_pca, eigvals_b = self._pca_align(cb)

        # Atom groupings built once and shared across all candidate stages.
        # When a bond_checker is present, groups are keyed by
        # (element, bond_environment) rather than element alone, preventing
        # the Hungarian algorithm from discovering chemically invalid
        # atom-index permutations that would implement a mirror reflection.
        #
        # Bond detection uses centered (pre-PCA) coordinates; distances are
        # rotation-invariant so the result is identical to using raw coords.
        groups_a, groups_b = self._build_groups(sym_a, ca, sym_b, cb)

        # ── Stage 1: four sign-flip candidates ─────────────────────────
        min_rmsd = self._try_candidates(
            self._sign_flip_candidates(),
            ca_pca, cb_pca, groups_a, groups_b,
        )
        if min_rmsd < self.rmsd_threshold:
            return min_rmsd

        # ── Stage 2: degeneracy check ───────────────────────────────────
        deg_01, deg_12 = self._degeneracy_flags(eigvals_a, eigvals_b)
        if not deg_01 and not deg_12:
            return min_rmsd

        # ── Stage 3: coarse planar / SO(3) grid ────────────────────────
        coarse = self._build_planar_candidates(deg_01, deg_12, n_plane=6, n_sphere=4)
        min_rmsd = min(min_rmsd, self._try_candidates(
            coarse, ca_pca, cb_pca, groups_a, groups_b,
        ))
        if min_rmsd < self.rmsd_threshold:
            return min_rmsd

        # ── Stage 4: fine SO(3) grid (full degeneracy only) ────────────
        if deg_01 and deg_12:
            fine = self._build_planar_candidates(deg_01, deg_12, n_plane=8, n_sphere=6)
            min_rmsd = min(min_rmsd, self._try_candidates(
                fine, ca_pca, cb_pca, groups_a, groups_b,
            ))

        return min_rmsd

    # ------------------------------------------------------------------ #
    #  Atom grouping                                                       #
    # ------------------------------------------------------------------ #

    def _build_groups(
        self,
        sym_a: list[str],
        coords_a: np.ndarray,
        sym_b: list[str],
        coords_b: np.ndarray,
    ) -> tuple[dict, dict]:
        """Build atom-index groupings for the Hungarian mapping step.

        Without bond_checker
            Groups atoms by element symbol only.  All carbon atoms form one
            group, all nitrogen atoms another, etc.

        With bond_checker
            Groups atoms by ``(element, bond_environment_fingerprint)``.  Only
            atoms that share the same element **and** the same set of bonded
            neighbour elements are considered interchangeable, preventing the
            Hungarian algorithm from discovering permutations that swap
            chemically inequivalent atoms.

        The grouping keys are arbitrary hashables, so the downstream
        ``_optimal_mapping`` method works unchanged — it simply looks up
        matching groups by key.
        """
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

    # ------------------------------------------------------------------ #
    #  Candidate evaluation                                                #
    # ------------------------------------------------------------------ #

    def _try_candidates(
        self,
        candidates: list[np.ndarray],
        ca: np.ndarray,
        cb: np.ndarray,
        groups_a: dict,
        groups_b: dict,
    ) -> float:
        """Evaluate every rotation candidate and return the minimum RMSD."""
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
                    return min_rmsd  # early exit
        return min_rmsd

    # ------------------------------------------------------------------ #
    #  PCA alignment                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pca_align(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate *coords* so its principal axes align with the Cartesian axes
        (largest variance → x, smallest → z).

        Sign normalisation (columns 0 and 1 only)
            ``np.linalg.eigh`` does not guarantee eigenvector signs.  For each
            of the first two columns, the sign is chosen so that the component
            with the largest absolute value is positive, giving a canonical and
            numerically stable orientation.

            Column 2 is intentionally excluded: it is fixed deterministically
            by the det = +1 constraint below.  Applying sign normalisation to
            all three columns and then flipping column 2 for the determinant
            would silently undo the normalisation for that column.

        Proper-rotation guarantee (det = +1)
            Without this, the PCA step can silently apply a reflection, causing
            enantiomers to be declared identical.
        """
        if len(coords) < 2:
            return coords, np.ones(3)

        cov = (coords.T @ coords) / max(1, len(coords) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Canonicalise sign for columns 0 and 1 only (column 2 set by det).
        for i in range(2):
            dominant = np.argmax(np.abs(eigvecs[:, i]))
            if eigvecs[dominant, i] < 0:
                eigvecs[:, i] *= -1

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        return coords @ eigvecs, eigvals

    # ------------------------------------------------------------------ #
    #  Rotation candidates                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def _sign_flip_candidates(cls) -> list[np.ndarray]:
        """Four proper rotations (det = +1) covering all PCA sign-flip combos."""
        return list(cls._SIGN_FLIP_MATS)

    @classmethod
    def _build_planar_candidates(
        cls,
        deg_01: bool,
        deg_12: bool,
        n_plane: int,
        n_sphere: int,
    ) -> list[np.ndarray]:
        """Build rotation candidates for degenerate PCA subspaces.

        * ``deg_01`` only  → rotations around z (the unique axis).
        * ``deg_12`` only  → rotations around x (the unique axis).
        * both             → SO(3) grid via ZYZ Euler angles.

        Each candidate is combined with all four sign-flip matrices.
        ``n_sphere > 8`` is not recommended (cost scales as n³).
        """
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
        """Return ``(deg_01, deg_12)`` — which adjacent eigenvalue pairs are
        degenerate in at least one structure.

        ``atol = _DEGENERACY_ABS_TOL`` (1e-4) catches near-zero eigenvalue
        pairs in extremely flat/linear molecules that would escape a
        purely relative tolerance test.
        """
        rtol = cls._DEGENERACY_REL_TOL
        atol = cls._DEGENERACY_ABS_TOL

        def _close(ev: np.ndarray, i: int, j: int) -> bool:
            return bool(np.isclose(ev[i], ev[j], rtol=rtol, atol=atol))

        deg_01 = _close(eigvals_a, 0, 1) or _close(eigvals_b, 0, 1)
        deg_12 = _close(eigvals_a, 1, 2) or _close(eigvals_b, 1, 2)
        return deg_01, deg_12

    @staticmethod
    def _so3_grid(n: int) -> list[np.ndarray]:
        """Uniform-ish SO(3) grid via ZYZ Euler angles (n³ matrices total).

        Alpha/gamma sampled uniformly in [0, 2π); beta cos-spaced in [0, π]
        for equal solid-angle coverage.  All products are batched via NumPy.
        """
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

    # ------------------------------------------------------------------ #
    #  Atom mapping — Hungarian algorithm                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _optimal_mapping(
        coords_a: np.ndarray,
        coords_b: np.ndarray,
        groups_a: dict,
        groups_b: dict,
    ) -> np.ndarray | None:
        """Return the optimal atom permutation as an integer ndarray, or None.

        Atoms are permuted within each group defined by the shared grouping key
        (element symbol when no bond_checker; (element, bond_env) otherwise).
        The Hungarian algorithm minimises total squared distance independently
        per group.

        Returns ``None`` if any group is unmatched or has inconsistent size.
        Uses a sentinel-free counter (``n_assigned``) rather than scanning for
        ``-1`` in the permutation array.
        """
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

    # ------------------------------------------------------------------ #
    #  Kabsch RMSD                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _kabsch_rmsd(pa: np.ndarray, pb: np.ndarray) -> float:
        """Minimum RMSD between *pa* and *pb* over all proper rotations.

        Chirality guard (d < 0 → return inf)
        -------------------------------------
        When the optimal superposition requires an improper rotation (det < 0),
        the classic Kabsch approach penalises it via a sign-corrected singular
        value but still returns a finite RMSD:

            RMSD_corrected ≈ 2 · sqrt(S[2] / N)

        For flat chiral molecules S[2] is tiny, so this can fall below any
        threshold — incorrectly declaring an enantiomer pair as identical.

        Fix: return ``inf`` immediately when d < 0.  The decision is based on
        the *state* (reflection required) rather than a numerical magnitude,
        making it independent of molecular geometry.

        Achiral molecules (H₂O, meso compounds) are unaffected because their
        mirror image superimposes via a *proper* rotation (d = +1), which is
        found by at least one candidate in ``_try_candidates``.

        Formula note
        ------------
        Execution reaches the RMSD calculation only when d = +1, so the
        original ``d * S[2]`` reduces to ``S[2]`` and is written as such.
        """
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

    A bond "fingerprint" is a dictionary mapping sorted element-pair tuples
    to the number of bonds of that type in the structure, e.g.::

        {("C", "C"): 2, ("C", "H"): 6, ("N", "H"): 1}

    Comparison is atom-index-permutation-invariant because it relies solely
    on element identities and bond counts, not on specific atom indices.  This
    is sufficient to detect formation or cleavage of covalent bonds (i.e.,
    reactions) while being insensitive to conformational changes.

    Parameters
    ----------
    covalent_margin : float
        Multiplicative margin applied to the sum of covalent radii when
        deciding whether two atoms are bonded.  A value of 1.2 (default)
        means atoms are considered bonded when their distance is within
        120 % of (r_i + r_j).
    """

    def __init__(self, covalent_margin: float = 1.2) -> None:
        self.covalent_margin = covalent_margin

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def fingerprint(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> dict[tuple[str, str], int]:
        """Return the bond-type count dictionary for *symbols* / *coords*.

        Each key is a ``(elem_a, elem_b)`` tuple with elements in sorted
        order (so C–H and H–C map to the same key).  The value is the
        number of such bonds.
        """
        bonded_si, bonded_sj = self._bonded_pairs(symbols, coords)
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
        """Return ``True`` when the bond topology of *new* differs from *ref*.

        Structures with different stoichiometry are considered to have
        undergone rearrangement (returns ``True``).
        """
        if sorted(ref_symbols) != sorted(new_symbols):
            return True
        return self.fingerprint(ref_symbols, ref_coords) != \
               self.fingerprint(new_symbols, new_coords)

    def atom_environments(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[tuple[str, ...]]:
        """Return a per-atom bond-environment fingerprint.

        For each atom *i*, the fingerprint is the **sorted tuple of element
        symbols of all atoms bonded to i**.  Two atoms are considered to have
        the same chemical environment if their fingerprints are equal.

        Examples
        --------
        Chiral carbon in CHFClBr bonded to four distinct halogens:
            C  → ("Br", "Cl", "F", "H")   # unique — no valid swap
            Br → ("C",)
            ...

        Carbon in a symmetric C₂ fragment bonded only to identical neighbours:
            C1 → ("O",)
            C2 → ("O",)   # indistinguishable — swap remains possible

        This method is used by ``StructureChecker`` to refine atom groupings
        so that the Hungarian algorithm only considers chemically valid
        atom-index permutations.
        """
        n = len(symbols)
        bonded_si, bonded_sj, ii_idx, jj_idx = self._bonded_pairs(
            symbols, coords, return_indices=True
        )

        neighbors: list[list[str]] = [[] for _ in range(n)]
        for i, j in zip(ii_idx, jj_idx):
            neighbors[i].append(symbols[j])
            neighbors[j].append(symbols[i])

        return [tuple(sorted(nb)) for nb in neighbors]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _bonded_pairs(
        self,
        symbols: list[str],
        coords: np.ndarray,
        *,
        return_indices: bool = False,
    ):
        """Compute all bonded atom pairs and return their element symbols.

        When *return_indices* is True, also return the raw index arrays
        ``ii`` and ``jj`` needed for per-atom adjacency construction.
        """
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
        """Return the covalent radius of *symbol* in Angstrom."""
        if covalent_radii_lib is not None:
            try:
                return float(covalent_radii_lib(symbol)) * _BOHR2ANG
            except (KeyError, TypeError):
                pass
        return 0.75  # generic fallback

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
    """Typed container for pending node energy backfill data.

    Both fields are nullable so that partial updates (e.g. only ``energy``
    is known at registration time) are correctly represented.  ``frozen``
    is intentionally left ``False`` so that future fields can be added
    without breaking existing serialisation/deserialisation code.
    """

    energy: float | None = None
    free_energy: float | None = None


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
        # _tasks: id(task) → ExplorationTask mapping.
        # Using a dict instead of a list gives O(1) deletion in pop() without
        # the O(N) list.remove() cost that previously negated the heap benefit.
        # Subclasses that need an ordered view should call
        # sorted(self._tasks.values(), ...) rather than operating on the dict
        # directly.
        self._tasks: dict[int, ExplorationTask] = {}
        # _task_counter: monotonically increasing integer assigned to each task
        # on push().  Using a stable counter instead of id(task) prevents the
        # id-reuse bug where a GC'd task's memory address is assigned to a new
        # object, causing _tasks.pop() to silently remove a live entry.
        self._task_counter: int = 0
        # _heap: min-heap of (raw_energy, counter, task_id) for O(log N)
        # extraction.  Stale entries (whose task_id is absent from _tasks) are
        # silently skipped in pop().
        # Storing raw_energy (absolute, not delta) means the heap order is
        # invariant to reference-energy shifts, so refresh_priorities() no
        # longer needs to rebuild the heap.
        self._heap: list[tuple[float, int, int]] = []
        self._push_counter: int = 0
        self._submitted: set[tuple] = set()
        self._rng = np.random.default_rng(rng_seed)
        # Latest reference energy supplied by refresh_priorities(); used for
        # lazy delta_E recalculation inside pop().
        self._current_ref_e: float | None = None

    def push(self, task: ExplorationTask) -> bool:
        key = (task.node_id, tuple(task.afir_params))
        if key in self._submitted:
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
        """Pop the highest-priority task using the heap (O(log N) amortised).

        ``_tasks`` deletion is O(1) because it is now a ``dict`` keyed by
        ``id(task)``.  Stale heap entries (whose ``task_id`` has already been
        removed from ``_tasks``) are silently skipped.

        When :meth:`refresh_priorities` has been called since the last push,
        ``delta_E_hartree`` is recomputed lazily for the popped task using the
        stored ``_current_ref_e`` before returning it, keeping the returned
        task's metadata current without O(N) heap reconstruction.

        **Stale-entry purge:** if ``len(_heap) > 2 * len(_tasks)`` at the
        start of a ``pop()`` call, the heap is rebuilt from ``_tasks`` to
        prevent unbounded accumulation of stale entries.  The 2× threshold
        keeps the amortised cost of each push/pop pair to O(log N) while
        bounding worst-case ``pop()`` latency to O(N log N) in the rare case
        where many tasks are discarded without being popped.
        """
        # ── Stale-entry purge ────────────────────────────────────────────
        n_tasks = len(self._tasks)
        if len(self._heap) > max(100, 2 * n_tasks):
            # Rebuild heap retaining only entries still present in _tasks.
            valid = [
                entry for entry in self._heap
                if entry[2] in self._tasks
            ]
            heapq.heapify(valid)
            n_purged = len(self._heap) - len(valid)
            self._heap = valid
            logger.debug(
                "ExplorationQueue: heap purged — %d stale entries removed, "
                "%d valid entries remain.",
                n_purged, len(valid),
            )

        while self._heap:
            _, _, task_id = heapq.heappop(self._heap)
            task = self._tasks.pop(task_id, None)
            if task is None:
                continue  # stale entry — task was already removed

            # Remove the key from _submitted so the set only tracks tasks that
            # are *currently* in the queue.  Without this removal, _submitted
            # would grow without bound over a long run because tasks are added
            # on push() but were never removed.  Duplicate-submission prevention
            # is now handled by ExploredPairsLog.has() in _enqueue_perturbations.
            key = (task.node_id, tuple(task.afir_params))
            self._submitted.discard(key)

            # Lazy delta_E / priority refresh for the single task being returned.
            if self._current_ref_e is not None:
                node_g = task.metadata.get("source_node_free_energy")
                node_e = task.metadata.get("source_node_energy")
                eff_e = node_g if node_g is not None else node_e
                if eff_e is not None:
                    task.metadata["delta_E_hartree"] = eff_e - self._current_ref_e
                task.priority = self.compute_priority(task)

            return task
        return None

    def is_submitted(self, key: tuple) -> bool:
        """Return ``True`` if *key* has already been submitted to the queue.

        Parameters
        ----------
        key : tuple
            A ``(node_id, tuple(afir_params))`` pair matching the format
            used internally by :meth:`push`.  For example::

                (0, ("100", "1", "5"))

        This method provides a public, encapsulated alternative to
        directly accessing the internal ``_submitted`` set.
        """
        return key in self._submitted

    def should_add(self, node: "EQNode", reference_energy: float, **kwargs) -> bool:
        """Decide probabilistically whether to enqueue a node.

        The return value of ``compute_priority`` (in the range 0–1) is used
        directly as the acceptance probability.  Override with ``return True``
        for deterministic acceptance of all nodes, or override with a custom
        probabilistic model as needed.

        When the node carries a Gibbs free energy (``node.free_energy``), that
        value is used in place of the electronic energy for the ΔE calculation
        so that the Boltzmann weight reflects the free-energy landscape.  The
        ``reference_energy`` argument should already be the unified reference
        returned by :meth:`NetworkGraph.reference_energy` (G-based when any
        node has thermochemistry, electronic otherwise).
        """
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
        """Record the current reference energy for lazy evaluation at pop() time.

        The previous implementation rebuilt the entire heap with
        ``heapq.heapify()`` on every iteration — O(N) cost.  This was
        unnecessary because the heap is ordered by *raw* (absolute) node
        energy, and the relative order of tasks is invariant to a uniform
        shift of the reference energy.

        The new design simply stores ``ref_e`` so that :meth:`pop` can
        recompute ``delta_E_hartree`` and ``priority`` for the single task
        it is about to return, eliminating the per-iteration O(N) rebuild
        entirely.

        For subclasses whose ``compute_priority`` is *not* monotone in
        ``delta_E`` (i.e. a lower raw energy does not always imply a higher
        priority), the heap order may be approximate.  In those cases
        subclasses should override ``pop()`` with a full-sort strategy
        (as ``RCMCQueue`` does) rather than relying on the heap.

        Parameters
        ----------
        ref_e:
            Unified reference energy from
            :meth:`NetworkGraph.reference_energy` (G-based when any node
            carries thermochemistry, electronic otherwise).  ``None`` is
            accepted and leaves the stored value unchanged, so the method
            is safe to call unconditionally at the start of each iteration.
        """
        if ref_e is not None:
            self._current_ref_e = ref_e
            
        
        if self._heap:
            new_heap = []
            for old_priority, task_id, task in self._heap:
        
                new_priority = self._calculate_priority(task)
                new_heap.append((new_priority, task_id, task))
        
            heapq.heapify(new_heap)
            self._heap = new_heap

    def export_queue_status(self) -> list[dict]:
        return [
            {
                "node_id": t.node_id,
                "priority": t.priority,
                "afir_params": t.afir_params,
            }
            for t in self._tasks.values()
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
# Section 3b : ExploredPairsLog
# ===========================================================================

class ExploredPairsLog:
    """Persistent log of explored (EQ node, atom pair, gamma sign) combinations.

    Records are stored one per line in a plain-text file located in ``work_dir``
    so that duplicate exploration is avoided across separate mapper runs.

    File format (one record per line)::
        EQ{node_id:06d} {atom_i_1based} {atom_j_1based} {gamma_sign}

    where ``gamma_sign`` is ``'+'`` for a positive (attractive) AFIR gamma and
    ``'-'`` for a negative (repulsive) AFIR gamma.

    Parameters
    ----------
    filepath : str
        Absolute path to the text file used for persistence.
    flush_interval : int
        Number of records to batch before flushing to disk. Defaults to 5.
        Higher values reduce I/O bottleneck but risk losing the latest 
        records if the process crashes unexpectedly.
    """

    def __init__(self, filepath: str, flush_interval: int = 5) -> None:
        self._filepath = filepath
        self._flush_interval = flush_interval
        self._write_count = 0

        # In-memory set for O(1) look-up: (node_id, atom_i, atom_j, gamma_sign)
        self._explored: set[tuple[int, int, int, str]] = set()
        # Fast O(1) membership check: which node_ids have been explored at all.
        self._explored_node_ids: set[int] = set()
        self._load()

        # Keep the append handle open so each record() call is a single
        # write() rather than open/write/close. 
        # The handle is flushed periodically based on flush_interval.
        self._fh = open(self._filepath, "a", encoding="utf-8")  # noqa: SIM115
        atexit.register(self.close)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

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
                    node_id = int(parts[0][2:])
                    atom_i = int(parts[1])
                    atom_j = int(parts[2])
                    gamma_sign = parts[3]
    
                    if gamma_sign not in ("+", "-"):
                        continue
    
                    self._explored.add((node_id, atom_i, atom_j, gamma_sign))
                    self._explored_node_ids.add(node_id)
                except (ValueError, IndexError):
                    logger.warning(
                        "ExploredPairsLog._load: malformed record at %s:%d "
                        "(content=%r) — skipped.",
                        self._filepath,
                        lineno,
                        line,
                    )
                    continue
                    
        logger.info(
            "ExploredPairsLog: loaded %d records from %s",
            len(self._explored),
            self._filepath,
        )

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def has(self, node_id: int, atom_i: int, atom_j: int, gamma_sign: str) -> bool:
        """Return ``True`` if this (node, pair, sign) has already been explored."""
        return (node_id, atom_i, atom_j, gamma_sign) in self._explored

    def has_any_for_node(self, node_id: int) -> bool:
        """Return ``True`` if *node_id* has at least one recorded exploration.

        Uses an internal O(1) set keyed by node_id so this method is constant-time
        regardless of the total number of explored pairs.
        """
        return node_id in self._explored_node_ids

    def contains(self, key: tuple[int, int, int, str]) -> bool:
        """Return ``True`` if the exact (node_id, atom_i, atom_j, gamma_sign) tuple exists.

        This is a convenience wrapper around :meth:`has` that accepts a pre-built
        4-tuple, which is useful when the caller has already constructed the key
        for other purposes (e.g. set-difference checks).
        """
        return key in self._explored

    def record(self, node_id: int, atom_i: int, atom_j: int, gamma_sign: str) -> None:
        """Mark the combination as explored and append it to the text file."""
        key = (node_id, atom_i, atom_j, gamma_sign)
        if key in self._explored:
            return

        self._explored.add(key)
        self._explored_node_ids.add(node_id)
        
        # Write to buffer
        self._fh.write(f"EQ{node_id:06d} {atom_i} {atom_j} {gamma_sign}\n")
        
        # Batch flushing
        self._write_count += 1
        if self._write_count % self._flush_interval == 0:
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying append file handle.

        Should be called when the log will no longer be written to (e.g. at
        the end of :meth:`ReactionNetworkMapper.run`). Safe to call multiple
        times — subsequent calls after the first are no-ops.
        """
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    # Context-manager support so callers can use ``with ExploredPairsLog(path) as log:``
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Return all atom pairs that satisfy the distance and covalency filters.

        Pairs are expressed as 0-based index tuples ``(i, j)`` with ``i < j``.

        Implementation uses numpy broadcasting to evaluate all N*(N-1)/2 pairs
        in a single vectorised pass, avoiding an O(N²) Python-level loop.
        """
        n = len(symbols)
        if n < 2:
            return []

        # Build the pool of atom indices subject to active_atoms restriction.
        # active_atoms stores 1-based labels; convert to 0-based for indexing.
        if self.active_atoms is not None:
            atom_indices = np.array(
                [i for i in range(n) if (i + 1) in self.active_atoms], dtype=np.intp
            )
        else:
            atom_indices = np.arange(n, dtype=np.intp)

        if len(atom_indices) < 2:
            return []

        # Restrict coords/symbols to the active subset.
        sub_coords  = coords[atom_indices]           # (m, 3)
        sub_symbols = [symbols[i] for i in atom_indices]
        m = len(atom_indices)

        # Pairwise distances for the active subset — vectorised.
        dmat = distance_matrix(sub_coords)                     # (m,m)

        # Upper-triangle indices (i < j).
        ii, jj = np.triu_indices(m, k=1)   # each has shape (P,) where P = m*(m-1)/2
        dists = dmat[ii, jj]               # (P,)

        # ── Distance window filter ────────────────────────────────────────
        dist_mask = (dists >= self.dist_lower_ang) & (dists <= self.dist_upper_ang)

        # ── Covalent-bond exclusion filter ────────────────────────────────
        if covalent_radii_lib is not None:
            # Build per-atom radii array for the active subset, falling back to
            # a generic value for unknown elements so the filter still works.
            radii = np.empty(m, dtype=np.float64)
            for k, sym in enumerate(sub_symbols):
                try:
                    radii[k] = covalent_radii_lib(sym) * _BOHR2ANG
                except KeyError:
                    radii[k] = 0.75   # generic fallback [Å]
            # Vectorised threshold for every upper-triangle pair.
            cov_thresh = self.covalent_margin * (radii[ii] + radii[jj])  # (P,)
            cov_mask = dists > cov_thresh
        else:
            # No radii library — apply a fixed generic threshold.
            cov_mask = dists > (self.covalent_margin * 1.5)

        keep = dist_mask & cov_mask   # (P,) boolean

        # Convert back to original (0-based) atom indices.
        orig_ii = atom_indices[ii[keep]]
        orig_jj = atom_indices[jj[keep]]
        return list(zip(orig_ii.tolist(), orig_jj.tolist()))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_candidate_pairs(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Return all valid non-covalent atom pairs without sampling.

        Unlike :meth:`generate_afir_perturbations`, this method returns the
        *complete* candidate pool (0-based index tuples) so that the caller can
        implement its own sampling strategy (e.g. Boltzmann-weighted node
        selection followed by uniform pair sampling).

        Returns
        -------
        list[tuple[int, int]]
            0-based ``(i, j)`` pairs with ``i < j``.
        """
        return self._build_candidates(symbols, coords)

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
        candidates = self._build_candidates(symbols, coords)

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
    # Gibbs free energy (G_tot) from thermochemistry analysis; None when
    # vibrational analysis was not performed for this structure.
    free_energy: float | None = None

    @property
    def has_real_energy(self) -> bool:
        return self.energy is not None

    @property
    def effective_energy(self) -> float | None:
        """Return free_energy when available, otherwise fall back to energy.

        Used throughout the mapper so that nodes with thermochemistry results
        are prioritised by Gibbs free energy while nodes without (e.g. the
        seed structure optimisation) still use electronic energy — both
        compared against the unified reference from
        :meth:`NetworkGraph.reference_energy`.
        """
        return self.free_energy if self.free_energy is not None else self.energy

    def to_dict(self) -> dict:
        return {
            "node_id":             self.node_id,
            "xyz_file":            self.xyz_file,
            "energy_hartree":      self.energy,
            "free_energy_hartree": self.free_energy,
            "source_run_dir":      self.source_run_dir,
            "extra":               self.extra,
        }

@dataclass
class TSEdge:
    edge_id: int
    node_id_1: int
    node_id_2: int
    ts_xyz_file: str | None
    ts_energy: float | None  # None when the energy profile could not be parsed
    barrier_fwd: float | None = None
    barrier_rev: float | None = None
    source_run_dir: str = ""
    # Geometry cached in memory for TS-vs-TS comparisons without disk access.
    symbols: list[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    extra: dict = field(default_factory=dict)
    # Free-energy counterparts (G_tot-based); None when thermochemistry was not
    # performed for the TS or one of its IRC endpoints.
    ts_free_energy: float | None = None
    barrier_fwd_free: float | None = None
    barrier_rev_free: float | None = None

    @property
    def has_coords(self) -> bool:
        """Return True when parsed atomic coordinates are available in memory."""
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
        # Merge extra, skipping non-JSON-serialisable values
        for k, v in self.extra.items():
            try:
                json.dumps(v)
                data[k] = v
            except (TypeError, ValueError):
                data[k] = str(v)
        return data

# Sentinel object used by NetworkGraph to distinguish "not yet computed"
# from a cached value of None (meaning no energy is available).
_UNSET = object()


class NetworkGraph:
    def __init__(self) -> None:
        self._nodes: dict[int, EQNode] = {}
        self._edges: dict[int, TSEdge] = {}
        self._node_counter: int = 0
        self._edge_counter: int = 0
        # Cached reference energy; set to _UNSET when invalidated.
        self._ref_energy_cache: float | None = _UNSET  # type: ignore[assignment]
        self.last_iteration: int = 0

        # ── Energy-sorted indexes for O(log N) window queries ──────────────
        # Each list stores (energy, id) tuples in ascending energy order,
        # maintained by bisect.insort on every add_node / add_edge call.
        # Nodes/edges whose energy is None are tracked separately in the
        # _*_none_ids lists and always included in window query results
        # (mirroring the original behaviour of skipping the energy pre-filter
        # when either energy is None).
        # The _index_dirty flag is raised by invalidate_energy_cache() and
        # causes a full rebuild on the next window-query call, covering the
        # case where backfill changes an energy from None to a real value.
        self._node_energy_index: list[tuple[float, int]] = []
        self._node_none_ids:     list[int]               = []
        self._edge_energy_index: list[tuple[float, int]] = []
        self._edge_none_ids:     list[int]               = []
        self._index_dirty: bool = False

    def add_node(self, node: EQNode) -> None:
        self._nodes[node.node_id] = node
        self._ref_energy_cache = _UNSET  # type: ignore[assignment]
        # Maintain energy index incrementally.
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
        """Return the number of nodes in O(1) without allocating a list copy."""
        return len(self._nodes)

    def next_node_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def add_edge(self, edge: TSEdge) -> None:
        self._edges[edge.edge_id] = edge
        # Maintain energy index incrementally.
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
        """Invalidate the cached reference energy.

        Should be called whenever node energies are modified externally
        (e.g. by energy backfill in ``_flush_node_energy_updates``) so that
        the next call to :meth:`reference_energy` recomputes from scratch.

        This method provides a public, encapsulated alternative to directly
        manipulating the internal ``_ref_energy_cache`` sentinel.

        Also marks the energy-sorted indexes as dirty so that the next call to
        :meth:`nodes_in_energy_window` or :meth:`edges_in_energy_window`
        triggers a full rebuild.  This covers the backfill case where a node's
        energy transitions from ``None`` to a real value, which would otherwise
        leave the node stranded in ``_node_none_ids`` without a sorted entry.
        """
        self._ref_energy_cache = _UNSET  # type: ignore[assignment]
        self._index_dirty = True

    def reference_energy(self) -> float | None:
        """Return the unified reference energy for priority/Boltzmann calculations.

        Strategy (matches ``EQNode.effective_energy``):

        * When **any** node carries a Gibbs free energy (``free_energy``),
          return the minimum of all available ``free_energy`` values.
          Nodes without thermochemistry results will compare their
          electronic energy against this G-based reference, which is the
          intended mixed-mode behaviour (see Q2 design decision).
        * When **no** node has free energy, fall back to the minimum
          electronic energy, preserving the original behaviour.

        The result is cached and automatically invalidated whenever a new
        node is added via :meth:`add_node`.
        """
        if self._ref_energy_cache is not _UNSET:
            return self._ref_energy_cache  # type: ignore[return-value]

        free_energies = [
            n.free_energy for n in self._nodes.values()
            if n.free_energy is not None
        ]
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
        # Write to a temporary file in the same directory, then atomically
        # rename it over the target path.  This ensures the JSON is never
        # left in a partially-written state if the process is interrupted
        # during the write (critical for long-running exploration jobs).
        dir_name = os.path.dirname(os.path.abspath(filepath))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filepath)   # atomic on POSIX; best-effort on Windows
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, filepath: str) -> None:
        """Load a reaction network from *filepath*, replacing the current graph.

        **Atomic-swap guarantee:** all parsing is performed into temporary
        local structures.  The live ``_nodes`` / ``_edges`` dicts and the
        energy index are only replaced once every node and edge has been
        successfully constructed.  If the JSON file is missing, malformed, or
        contains an unrecoverable parse error, the existing graph data is
        preserved unchanged and the exception propagates to the caller.

        Raises
        ------
        FileNotFoundError, json.JSONDecodeError, KeyError, …
            Any exception raised while opening or parsing the file.
        """
        # ── Phase 1: parse — all work done on local variables ─────────────
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
            # Populate in-memory geometry cache for TS-vs-TS comparisons.
            if edge.ts_xyz_file and os.path.isfile(edge.ts_xyz_file):
                try:
                    edge.symbols, edge.coords = parse_xyz(edge.ts_xyz_file)
                except Exception as exc:
                    logger.warning(
                        "NetworkGraph.load: could not parse TS geometry from %s: %s",
                        edge.ts_xyz_file, exc,
                    )
            elif edge.ts_xyz_file:
                logger.warning(
                    "NetworkGraph.load: TS XYZ file not found on disk for TS%06d "
                    "(path=%s). Duplicate detection for this edge will be skipped.",
                    edge.edge_id, edge.ts_xyz_file,
                )
            new_edges[edge.edge_id] = edge
            if edge.ts_energy is not None:
                bisect.insort(new_edge_energy_index, (edge.ts_energy, edge.edge_id))
            else:
                new_edge_none_ids.append(edge.edge_id)

        new_node_counter = (max(new_nodes.keys()) + 1) if new_nodes else 0
        new_edge_counter = (max(new_edges.keys()) + 1) if new_edges else 0
        new_last_iteration = data.get("metadata", {}).get("last_iteration", 0)

        # ── Phase 2: commit — replace live state atomically ────────────────
        # Only reached when Phase 1 completed without raising.
        self._nodes = new_nodes
        self._edges = new_edges
        self._node_counter = new_node_counter
        self._edge_counter = new_edge_counter
        self._node_energy_index = new_node_energy_index
        self._node_none_ids     = new_node_none_ids
        self._edge_energy_index = new_edge_energy_index
        self._edge_none_ids     = new_edge_none_ids
        self._index_dirty       = False
        self._ref_energy_cache  = _UNSET  # type: ignore[assignment]
        self.last_iteration     = new_last_iteration

    # ------------------------------------------------------------------
    # Energy-window index helpers
    # ------------------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        """Rebuild both energy-sorted indexes from scratch.

        Called automatically by the window-query methods when
        ``_index_dirty`` is ``True``.  The most common trigger is
        :meth:`invalidate_energy_cache` after a backfill pass in
        ``_flush_node_energy_updates``, where node energies may have
        transitioned from ``None`` to real values.

        Complexity: O(N log N) for nodes + O(E log E) for edges.
        Called at most once per mapper iteration (the next backfill
        flush) so the amortised cost per ``_process_profile`` call
        remains O(log N) in steady state.
        """
        self._node_energy_index = []
        self._node_none_ids     = []
        for node in self._nodes.values():
            if node.energy is not None:
                bisect.insort(self._node_energy_index, (node.energy, node.node_id))
            else:
                self._node_none_ids.append(node.node_id)

        self._edge_energy_index = []
        self._edge_none_ids     = []
        for edge in self._edges.values():
            if edge.ts_energy is not None:
                bisect.insort(self._edge_energy_index, (edge.ts_energy, edge.edge_id))
            else:
                self._edge_none_ids.append(edge.edge_id)

        self._index_dirty = False

    def nodes_in_energy_window(
        self,
        center: float,
        tol: float,
    ) -> list[EQNode]:
        """Return EQ nodes whose energy lies within *center* ± *tol*.

        Uses a binary-search on the internal sorted index for O(log N +
        k) complexity, where k is the number of nodes in the window.
        Nodes with ``energy = None`` are always included (the energy
        pre-filter is not applied to them in the caller either).

        Parameters
        ----------
        center:
            The query energy in Hartree (typically the incoming node's
            electronic energy).
        tol:
            Half-width of the energy window in Hartree (``self.energy_tolerance``
            in :class:`ReactionMapper`).

        Returns
        -------
        list[EQNode]
            Candidate nodes to check.  May be empty.
        """
        if self._index_dirty:
            self._rebuild_indexes()

        lo = bisect.bisect_left(self._node_energy_index,  (center - tol, -1))
        hi = bisect.bisect_right(self._node_energy_index, (center + tol, float("inf")))

        result: list[EQNode] = [
            self._nodes[nid]
            for _, nid in self._node_energy_index[lo:hi]
            if nid in self._nodes
        ]
        # Always include nodes whose energy is None.
        for nid in self._node_none_ids:
            node = self._nodes.get(nid)
            if node is not None:
                result.append(node)
        return result

    def edges_in_energy_window(
        self,
        center: float,
        tol: float,
    ) -> list[TSEdge]:
        """Return TS edges whose ``ts_energy`` lies within *center* ± *tol*.

        Analogous to :meth:`nodes_in_energy_window` but for
        :class:`TSEdge` objects.  Edges without parsed energies
        (``ts_energy = None``) are always included.

        Parameters
        ----------
        center:
            The incoming TS energy in Hartree.
        tol:
            Half-width of the energy window in Hartree.

        Returns
        -------
        list[TSEdge]
            Candidate edges to check for geometric duplication.
        """
        if self._index_dirty:
            self._rebuild_indexes()

        lo = bisect.bisect_left(self._edge_energy_index,  (center - tol, -1))
        hi = bisect.bisect_right(self._edge_energy_index, (center + tol, float("inf")))

        result: list[TSEdge] = [
            self._edges[eid]
            for _, eid in self._edge_energy_index[lo:hi]
            if eid in self._edges
        ]
        # Always include edges whose ts_energy is None.
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
            eff = node.effective_energy
            label = "G" if node.free_energy is not None else "E"
            if eff is not None and ref is not None:
                rel = (eff - ref) * HARTREE_TO_KCALMOL
                e_str = f"{eff:+.8f} Ha  ({rel:+.2f} kcal/mol) [{label}]"
            elif eff is not None:
                e_str = f"{eff:+.8f} Ha  [{label}]"
            else:
                e_str = "energy unknown"
            lines.append(f"  EQ{node.node_id:06d}: {e_str}  [{node.xyz_file}]")

        for edge in sorted(self._edges.values(), key=lambda e: e.edge_id):
            fwd      = f"{edge.barrier_fwd:.2f}"       if edge.barrier_fwd      is not None else "N/A"
            rev      = f"{edge.barrier_rev:.2f}"       if edge.barrier_rev      is not None else "N/A"
            ts_e     = f"{edge.ts_energy:.8f} Ha"      if edge.ts_energy        is not None else "N/A"
            fwd_free = f"{edge.barrier_fwd_free:.2f}"  if edge.barrier_fwd_free is not None else "N/A"
            rev_free = f"{edge.barrier_rev_free:.2f}"  if edge.barrier_rev_free is not None else "N/A"
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
        ep1 = os.path.join(profile_dir, "endpoint_1_opt.xyz")
        ep2 = os.path.join(profile_dir, "endpoint_2_opt.xyz")
        ts_matches = glob.glob(os.path.join(profile_dir, "*_ts_final.xyz"))
        txt_path   = os.path.join(profile_dir, "energy_profile.txt")

        if not os.path.isfile(ep1) or not os.path.isfile(ep2):
            logger.info(
                "ProfileParser: endpoint file(s) missing in %s "
                "(ep1_exists=%s  ep2_exists=%s) -> parse() returns None.",
                profile_dir,
                os.path.isfile(ep1), os.path.isfile(ep2),
            )
            return None
        if not ts_matches:
            logger.info(
                "ProfileParser: no *_ts_final.xyz found in %s -> parse() returns None.",
                profile_dir,
            )
            return None

        ts_file = ts_matches[0]
        if len(ts_matches) > 1:
            logger.warning(
                "ProfileParser: %d *_ts_final.xyz files found in %s — "
                "using the first one (%s). Check for unexpected duplicates.",
                len(ts_matches), profile_dir, ts_file,
            )
        energies      = self._parse_energy_txt(txt_path)
        free_energies = self._parse_free_energy_txt(txt_path)

        ts_e   = energies.get("TS")
        ep1_e  = energies.get("Endpoint_1")
        ep2_e  = energies.get("Endpoint_2")

        ts_g   = free_energies.get("TS")
        ep1_g  = free_energies.get("Endpoint_1")
        ep2_g  = free_energies.get("Endpoint_2")

        logger.info(
            "ProfileParser: parsed  ts=%s  ep1=%s  ep2=%s  "
            "ts_G=%s  ep1_G=%s  ep2_G=%s  [%s]",
            f"{ts_e:.8f} Ha"  if ts_e  is not None else "None",
            f"{ep1_e:.8f} Ha" if ep1_e is not None else "None",
            f"{ep2_e:.8f} Ha" if ep2_e is not None else "None",
            f"{ts_g:.8f} Ha"  if ts_g  is not None else "None",
            f"{ep1_g:.8f} Ha" if ep1_g is not None else "None",
            f"{ep2_g:.8f} Ha" if ep2_g is not None else "None",
            profile_dir,
        )

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
            # Free-energy counterparts (None when thermochemistry unavailable)
            "ts_free_energy":         ts_g,
            "endpoint_1_free_energy": ep1_g,
            "endpoint_2_free_energy": ep2_g,
            "barrier_fwd_free":       _barrier(ep1_g, ts_g),
            "barrier_rev_free":       _barrier(ep2_g, ts_g),
        }

    @staticmethod
    def _parse_energy_txt(txt_path: str) -> dict:
        """Parse the electronic-energy table from energy_profile.txt.
    
        Reads lines of the form ``Structure, file_path, energy, bias_energy``
        and returns a dict mapping structure key to the third column (Hartree).
        Stops reading at the free-energy section header so that G_tot values
        are never accidentally picked up by this method.
        """
        result: dict[str, float | None] = {"TS": None, "Endpoint_1": None, "Endpoint_2": None}
        if not os.path.isfile(txt_path):
            return result
    
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                # Stop at the free-energy section so G_tot is not mis-parsed
                # as an electronic energy (the section starts with "# ===...
                # FREE ENERGY SECTION").
                if "FREE ENERGY SECTION" in stripped:
                    break
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
                    logger.warning(
                        "_parse_energy_txt: could not parse energy value for '%s' "
                        "in %s: %r",
                        key, txt_path, parts[2],
                    )
        return result

    @staticmethod
    def _parse_free_energy_txt(txt_path: str) -> dict:
        """Parse the free-energy section appended by autots.py to energy_profile.txt.
    
        The section written by ``_write_energy_profile_text`` looks like::
    
            # ===================================================================
            # FREE ENERGY SECTION (G_tot from thermochemistry analysis)
            # ===================================================================
            # ...
            Structure,    G_tot [Hartree]
            TS,           -123.456789012345
            Endpoint_1,   -122.345678901234
            Endpoint_2,   N/A
    
        Lines whose second column is ``N/A`` (vibrational analysis was not
        performed) are stored as ``None``.  The section is terminated by the
        next ``# ---...`` separator or end-of-file.
    
        Returns a dict with keys ``"TS"``, ``"Endpoint_1"``, ``"Endpoint_2"``
        mapping to ``float | None``.
        """
        result: dict[str, float | None] = {"TS": None, "Endpoint_1": None, "Endpoint_2": None}
        if not os.path.isfile(txt_path):
            return result
    
        in_section = False
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                # Detect entry into free-energy section
                if "FREE ENERGY SECTION" in stripped:
                    in_section = True
                    continue
                if not in_section:
                    continue
                # Skip comment lines and the column-header line
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) < 2:
                    continue
                key = parts[0]
                if key not in result:
                    continue
                raw = parts[1].strip()
                if raw.upper() == "N/A" or not raw:
                    result[key] = None
                else:
                    try:
                        result[key] = float(raw)
                    except ValueError:
                        logger.warning(
                            "_parse_free_energy_txt: could not parse free energy "
                            "value for '%s' in %s: %r",
                            key, txt_path, raw,
                        )
                        result[key] = None
        return result

# ===========================================================================
# Section 7 : ReactionNetworkMapper
# ===========================================================================

class ReactionNetworkMapper:
    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    # Canonical set of keys accepted under the "mapper_settings" section.
    # Any key present in the config that is NOT listed here raises ValueError.
    _KNOWN_MAPPER_SETTINGS_KEYS: frozenset[str] = frozenset({
            "temperature_K",
            "rmsd_threshold",
            "max_iterations",
            "afir_gamma_kJmol",
            "max_pairs",
            "dist_lower_ang",
            "dist_upper_ang",
            "output_dir",
            "rng_seed",
            "active_atoms",
            "include_negative_gamma",
            "excluded_node_ids",
            "exclude_bond_rearrangement",
            "use_rcmc",
            "rcmc_temperature_K",
            "rcmc_reaction_time_s",
            "rcmc_start_node_id",
            "n_parallel",
            "worker_timeout_s", 
    })

    @classmethod
    def _validate_mapper_settings(cls, config: dict) -> None:
        """Validate the ``mapper_settings`` section of *config*.

        Raises
        ------
        ValueError
            - ``mapper_settings`` key is missing from *config*.
            - Any unknown key is found inside ``mapper_settings``
              (likely a typo — stops the run before wasting compute time).
            - Any required numeric key is absent.
            - Any value fails a range or type check.

        This method is intentionally strict: every detected problem raises
        ``ValueError`` immediately rather than logging a warning, because a
        misconfigured mapper on a long chemical-exploration run can silently
        corrupt the reaction network with no easy way to recover.
        """
        if "mapper_settings" not in config:
            raise ValueError(
                "Config validation failed: top-level key 'mapper_settings' is missing."
            )

        ms: dict = config["mapper_settings"]

        # ── Unknown-key check ─────────────────────────────────────────────
        unknown = set(ms.keys()) - cls._KNOWN_MAPPER_SETTINGS_KEYS
        if unknown:
            raise ValueError(
                f"Config validation failed: unknown key(s) in 'mapper_settings': "
                f"{sorted(unknown)}. "
                f"Known keys are: {sorted(cls._KNOWN_MAPPER_SETTINGS_KEYS)}."
            )

        # ── Required keys and per-field checks ───────────────────────────
        # (key, type_or_types, condition_fn, condition_description)
        _checks: list[tuple] = [
            ("temperature_K",        (int, float), lambda v: v > 0,   "must be > 0"),
            ("rmsd_threshold",       (int, float), lambda v: v > 0,   "must be > 0"),
            ("max_iterations",       int,          lambda v: v > 0,   "must be a positive integer"),
            ("afir_gamma_kJmol",     (int, float), lambda v: v > 0,   "must be > 0"),
            ("max_pairs",            int,          lambda v: v > 0,   "must be a positive integer"),
            ("dist_lower_ang",       (int, float), lambda v: v >= 0,  "must be >= 0"),
            ("dist_upper_ang",       (int, float), lambda v: v > 0,   "must be > 0"),
            ("rng_seed",             int,          lambda v: v >= 0,  "must be a non-negative integer"),
        ]

        for key, expected_type, condition, description in _checks:
            if key not in ms:
                raise ValueError(
                    f"Config validation failed: required key '{key}' is missing "
                    f"from 'mapper_settings'."
                )
            val = ms[key]
            if not isinstance(val, expected_type):
                raise ValueError(
                    f"Config validation failed: 'mapper_settings.{key}' must be "
                    f"{expected_type} but got {type(val).__name__} ({val!r})."
                )
            if not condition(val):
                raise ValueError(
                    f"Config validation failed: 'mapper_settings.{key}' {description} "
                    f"(got {val!r})."
                )

        # ── output_dir: optional str, but must be str when present ────────
        if "output_dir" in ms and not isinstance(ms["output_dir"], str):
            raise ValueError(
                f"Config validation failed: 'mapper_settings.output_dir' must be a "
                f"string (got {type(ms['output_dir']).__name__})."
            )

        # ── excluded_node_ids: optional, must be list[int] when present ──────
        if "excluded_node_ids" in ms:
            en = ms["excluded_node_ids"]
            if not isinstance(en, list):
                raise ValueError(
                    f"Config validation failed: 'mapper_settings.excluded_node_ids' must be "
                    f"a list of integers (got {type(en).__name__})."
                )
            for idx, item in enumerate(en):
                if not isinstance(item, int):
                    raise ValueError(
                        f"Config validation failed: 'mapper_settings.excluded_node_ids[{idx}]' "
                        f"must be an integer (got {type(item).__name__}: {item!r})."
                    )

        # ── Cross-field consistency ───────────────────────────────────────
        lo = ms.get("dist_lower_ang")
        hi = ms.get("dist_upper_ang")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(
                f"Config validation failed: 'mapper_settings.dist_lower_ang' ({lo}) "
                f"must be strictly less than 'dist_upper_ang' ({hi})."
            )

        # ── n_parallel: optional int >= 1 ────────────────────────────────
        if "n_parallel" in ms:
            np_val = ms["n_parallel"]
            if not isinstance(np_val, int):
                raise ValueError(
                    f"Config validation failed: 'mapper_settings.n_parallel' must be "
                    f"an integer (got {type(np_val).__name__}: {np_val!r})."
                )
            if np_val < 1:
                raise ValueError(
                    f"Config validation failed: 'mapper_settings.n_parallel' must be "
                    f">= 1 (got {np_val!r})."
                )

        logger.info(
            "_validate_mapper_settings: all checks passed for 'mapper_settings'."
        )

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
        """
        Parameters
        ----------
        base_config : dict
            Configuration dictionary forwarded to AutoTSWorkflow.
        queue : ExplorationQueue | None
            Priority queue used for initial exploration tasks generated by
            :meth:`_enqueue_perturbations`.  Defaults to :class:`BoltzmannQueue`.
        structure_checker : StructureChecker | None
            Single RMSD-based duplicate detector used for **both** EQ and TS
            structures.  The same checker instance and threshold are applied in
            each case; what differs is *which pool is searched*:

            * EQ identity  — the candidate is compared only against every
              :class:`EQNode` already stored in :attr:`graph` (via
              :meth:`_find_or_register_node`).
            * TS identity  — the candidate is compared only against every
              :class:`TSEdge` already stored in :attr:`graph` (via
              :meth:`_process_profile`).

            Defaults to ``StructureChecker(rmsd_threshold=0.30)``.
        perturbation_generator : PerturbationGenerator | None
            Generator for AFIR atom-pair perturbations.
        output_dir : str
            Root directory for all mapper outputs (nodes, runs, logs).
        work_dir : str | None
            Directory used for persistent exploration state files such as
            ``explored_pairs.txt``.  Defaults to ``output_dir`` when ``None``.
        graph_json : str
            Filename (relative to ``output_dir``) for the reaction network JSON.
        max_iterations : int
            Maximum number of AutoTS workflow executions.  The mapper loops
            indefinitely over (EQ, atom-pair) combinations sampled via the
            Boltzmann distribution and stops only when this limit is reached
            (or when a ``stop.txt`` sentinel file is detected).
        resume : bool
            If ``True`` and a previous graph JSON exists, resume from it.
        rng_seed : int
            Seed for the internal NumPy RNG used during Boltzmann sampling.
        energy_tolerance_kcalmol : float
            Energy window [kcal/mol] used as a quick pre-filter before the
            full RMSD check.  Applied to both EQ node and TS edge comparisons.
            Only active when both energies being compared are known (not None);
            the filter is skipped silently when either value is None so that
            structures with unavailable energies are still checked geometrically.
            Default: 1.0 kcal/mol.
        config_file_path : str | None
            Absolute path to the JSON configuration file used for this run.
            When provided, the file is copied to ``work_dir`` at the start of
            :meth:`run` under the name ``config_snapshot.json``.  This makes
            every mapper output directory self-contained and reproducible.
            ``None`` disables the copy (default).
        excluded_node_ids : list[int] | set[int] | None
            EQ node IDs that must never be used as AFIR exploration starting
            points.  Matching nodes are still registered in the graph (so that
            edges connecting to them are preserved) but no perturbation tasks
            are ever enqueued for them.  ``None`` (default) means no exclusions.
        exclude_bond_rearrangement : bool
            When ``True``, any newly discovered EQ node whose covalent bond
            topology differs from that of the seed structure (EQ0) is
            automatically added to ``excluded_node_ids`` and will not be
            explored further.  Default is ``False``.
        bond_checker : BondTopologyChecker | None
            Instance used for bond-topology comparisons when
            ``exclude_bond_rearrangement`` is ``True``.  Defaults to
            ``BondTopologyChecker()`` when ``None`` and the option is enabled.
        n_parallel : int
            Maximum number of AutoTSWorkflow tasks to execute concurrently.
            Each task runs in its own **spawned** subprocess so that
            ``os.chdir()`` calls inside AutoTSWorkflow are fully isolated from
            the mapper process and from each other.

            * ``1`` (default) — sequential execution; one subprocess at a time.
              Behaviour is identical to the previous single-process design
              except that the mapper's CWD is never mutated.
            * ``N > 1`` — up to *N* tasks are submitted simultaneously to a
              :class:`concurrent.futures.ProcessPoolExecutor`.  After all *N*
              subprocesses complete, their profiles are processed and the graph
              is saved once for the whole batch.

            Regardless of this value, the ``spawn`` start method is always
            used so the subprocess environment is clean even on Linux (where
            ``fork`` is the platform default).
        """
        self.base_config = base_config

        # ── Validate config at construction time ──────────────────────────
        # Raises ValueError immediately if mapper_settings is invalid so that
        # misconfigured jobs never reach the expensive calculation loop.
        self._validate_mapper_settings(base_config)

        self.queue    = queue if queue is not None else BoltzmannQueue()
        # A single StructureChecker is used for both EQ and TS comparisons.
        # The two types are kept strictly separate by the pools they search:
        #   EQ duplicates -> _find_or_register_node  searches graph.all_nodes()
        #   TS duplicates -> _process_profile         searches graph.all_edges()
        self.checker  = structure_checker if structure_checker is not None else StructureChecker(rmsd_threshold=0.30)
        self.perturber = perturbation_generator if perturbation_generator is not None else PerturbationGenerator()
        self.output_dir = os.path.abspath(output_dir)
        self.work_dir   = os.path.abspath(work_dir if work_dir is not None else output_dir)
        self.graph_json_path = os.path.join(self.output_dir, graph_json)
        self.max_iterations = max_iterations
        self.resume = resume
        self._rng = np.random.default_rng(rng_seed)

        self.graph = NetworkGraph()
        self._iteration: int = 0

        # Accumulates energy backfill requests discovered during each iteration.
        # Maps node_id -> NodeUpdateData (typed container for energy/free_energy).
        # Applied in bulk by _flush_node_energy_updates() before graph.save().
        self._pending_node_updates: dict[int, NodeUpdateData] = {}
        
        if self.resume:
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            os.makedirs(self.output_dir, exist_ok=False) # for safe
        os.makedirs(self.work_dir,   exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "nodes"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "runs"),  exist_ok=True)

        # One shared energy tolerance used for both EQ and TS pre-filtering.
        self.energy_tolerance = energy_tolerance_kcalmol / HARTREE_TO_KCALMOL

        # Persistent log of explored (EQ, atom_pair, gamma_sign) combinations.
        explored_log_path = os.path.join(self.work_dir, "explored_pairs.txt")
        self.explored_log = ExploredPairsLog(explored_log_path)

        # Path to the JSON config file; copied to work_dir at run() start when set.
        self.config_file_path: str | None = (
            os.path.abspath(config_file_path) if config_file_path else None
        )

        # ── EQ exclusion options ──────────────────────────────────────────
        # Set of node IDs that are excluded from AFIR exploration.
        self.excluded_node_ids: set[int] = set(excluded_node_ids) if excluded_node_ids else set()

        # Subset of excluded_node_ids that were added automatically by the
        # bond-rearrangement filter (exclude_bond_rearrangement=True).
        # Kept separate so that the identity-comparison pool can treat these
        # nodes differently from manually-specified exclusions: bond-rearrangement
        # excluded nodes are still used as reference structures for deduplication,
        # whereas manually-specified nodes are removed from that pool entirely.
        self._bond_rearrangement_excluded_ids: set[int] = set()

        # Bond-rearrangement filter.
        self.exclude_bond_rearrangement: bool = exclude_bond_rearrangement
        self.bond_checker: BondTopologyChecker = (
            bond_checker if bond_checker is not None else BondTopologyChecker()
        )
        # Fingerprint of the seed structure (EQ0); set in _register_seed_structure.
        self._ref_bond_fingerprint: dict[tuple[str, str], int] | None = None
        self._ref_bond_symbols: list[str] = []
        self._ref_bond_coords: np.ndarray = np.empty((0, 3), dtype=float)

        # ── Subprocess isolation settings ─────────────────────────────────
        # n_parallel=1 → sequential (one spawn at a time).
        # n_parallel>1 → concurrent batch via ProcessPoolExecutor.
        # spawn is always forced so the child env is clean on all platforms.
        self.n_parallel: int = max(1, int(n_parallel))
        self._mp_ctx = multiprocessing.get_context("spawn")
        # ── Worker subprocess timeout ─────────────────────────────────────────
        # [FIX] Companion setting for the crash-aware polling loop in _run_autots.
        #
        # None (default) — no hard wall-clock limit.  The polling loop waits
        #     indefinitely as long as the worker process is alive, so legitimately
        #     long quantum chemistry calculations are never interrupted.
        # int > 0        — hard ceiling in seconds.  Use only in HPC environments
        #     where the job scheduler enforces its own wall-clock limit and the
        #     mapper must terminate cleanly before that limit is reached.
        #     Set via mapper_settings["worker_timeout_s"] in the JSON config.
        _raw_timeout = base_config.get("mapper_settings", {}).get("worker_timeout_s", None)
        self.worker_timeout_s: int | None = int(_raw_timeout) if _raw_timeout is not None else None
        
    # ------------------------------------------------------------------
    # AFIR task key parsing (Fix 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_afir_task_key(task: ExplorationTask) -> tuple[str, int, int]:
        """Extract (gamma_sign, atom_i_1based, atom_j_1based) from a task.

        This helper eliminates the duplicated parsing logic that was
        previously inlined in two separate locations within :meth:`run`.

        Returns
        -------
        gamma_sign : str
            ``'+'`` for attractive (positive gamma), ``'-'`` for repulsive.
        atom_i : int
            1-based index of the first atom in the pair (0 if unavailable).
        atom_j : int
            1-based index of the second atom in the pair (0 if unavailable).
        """
        gamma_sign = (
            "-" if task.afir_params and task.afir_params[0].startswith("-")
            else "+"
        )
        try:
            atom_i = int(task.afir_params[1]) if len(task.afir_params) >= 3 else 0
            atom_j = int(task.afir_params[2]) if len(task.afir_params) >= 3 else 0
        except (ValueError, IndexError):
            logger.warning(
                "_parse_afir_task_key: could not parse atom indices from "
                "afir_params=%r — falling back to (0, 0).",
                task.afir_params,
            )
            atom_i, atom_j = 0, 0
        return gamma_sign, atom_i, atom_j

    # ------------------------------------------------------------------
    # Main loop (Fix 3: deduplicated AFIR key parsing)
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the reaction network mapping loop.
    
        The mapper iterates up to ``max_iterations`` times.  Each iteration
        selects the next (EQ node, atom pair) to explore using the following
        strategy:
    
        1. **Queue drain** – Tasks previously enqueued by
        :meth:`_enqueue_perturbations` are consumed first (highest-priority
        first, according to the queue's Boltzmann weights).
        2. **Queue re-scan** – When the queue is empty, all nodes are
        re-scanned via :meth:`_enqueue_perturbations` to find any
        remaining unexplored pairs.  If none are found the loop stops.
    
        The loop runs indefinitely (no limit on the number of unique EQ/pair
        combinations) and stops only when ``max_iterations`` executions have
        been completed or a ``stop.txt`` sentinel file is detected in ``output_dir``.
        """
        # ── Re-validate config at run() entry ────────────────────────────
        # base_config could have been mutated between __init__ and run(), so
        # a second check here guarantees the config is still valid at the
        # moment the expensive calculation loop actually starts.
        self._validate_mapper_settings(self.base_config)
    
        if self.resume and os.path.isfile(self.graph_json_path):
            self.graph.load(self.graph_json_path)
            self._iteration = self.graph.last_iteration
            self._requeue_all_nodes()
        else:
            self._register_seed_structure()
    
        # Copy the JSON config to work_dir so the output directory is
        # self-contained.  Done after directory creation (handled in __init__)
        # and after the resume branch so work_dir is guaranteed to exist.
        self._save_config_snapshot()
    
        history_log = os.path.join(self.output_dir, "exploration_history.log")
        priority_log = os.path.join(self.output_dir, "queue_priority.log")
    
        # The try/finally guarantees explored_log is flushed and closed
        # even when an unhandled exception propagates out of the loop.
        try:
            while True:
                # ── Stop-file sentinel ──────────────────────────────────────────
                if os.path.isfile(os.path.join(self.output_dir, "stop.txt")):
                    logger.info("stop.txt detected in output_dir. Stopping.")
                    break
    
                # ── Iteration limit ─────────────────────────────────────────────
                if self.max_iterations > 0 and self._iteration >= self.max_iterations:
                    logger.info(
                        "Reached max_iterations (%d). Stopping.", self.max_iterations
                    )
                    break
    
                # ── Parallel path (n_parallel > 1) ──────────────────────────────
                if self.n_parallel > 1:
                    self.queue.refresh_priorities(self.graph.reference_energy())
                    # Clamp collection size so we never exceed max_iterations.
                    # This must be computed *before* _collect_task_batch because
                    # that method now pre-increments _iteration for each task.
                    if self.max_iterations > 0:
                        remaining = self.max_iterations - self._iteration
                        n_collect = min(self.n_parallel, remaining)
                    else:
                        n_collect = self.n_parallel
                    batch = self._collect_task_batch(n_collect)
                    if not batch:
                        logger.info(
                            "All candidate (EQ, pair) combinations exhausted. Stopping."
                        )
                        break
                    self._run_batch_parallel(batch, history_log, priority_log)
                    continue
    
                # ── Sequential path (n_parallel == 1) ───────────────────────────
                # Refresh queue priorities against the current reference energy
                # before popping so that tasks enqueued under a higher ref_e are
                # re-weighted correctly (relevant when a new lowest-energy node
                # has been found since those tasks were pushed).
                self.queue.refresh_priorities(self.graph.reference_energy())
    
                # Phase 1: drain the pre-built queue (respects Boltzmann priority).
                task = self.queue.pop()
    
                if task is not None:
                    # Parse AFIR key once — used for both the skip check and
                    # the explored_log.record() call below.
                    gamma_sign, atom_i, atom_j = self._parse_afir_task_key(task)
    
                    # Verify that this queue task has not already been executed
                    # (relevant after a resume where the queue is rebuilt from
                    # scratch but the explored_pairs log retains prior history).
                    if self.explored_log.has(task.node_id, atom_i, atom_j, gamma_sign):
                        logger.debug(
                            "Skipping queued task (EQ%06d, %d-%d, %s): already explored.",
                            task.node_id, atom_i, atom_j, gamma_sign,
                        )
                        # Do NOT count against max_iterations; try again next cycle.
                        continue
    
                    # ── [FIX] Record immediately after has() confirms task is new ─
                    # Reason for fix:
                    # In the previous implementation, `explored_log.record()` was called after
                    # `_iteration += 1` and `_run_autots()`. Since `pop()` internally calls
                    # `_submitted.discard(key)`, there was a window before `record()` completed
                    # where `_enqueue_perturbations(force_add=True)` could `push()` the same 
                    # pair again, potentially leading to double execution.
                    # By calling `record()` immediately after confirming it is unexplored via 
                    # `has()`, we close this window. Now, even if `_enqueue_perturbations` is 
                    # called inside the subsequent `_run_autots()`, the pair will be evaluated 
                    # as already recorded, preventing double enqueueing.
                    self.explored_log.record(task.node_id, atom_i, atom_j, gamma_sign)
    
                else:
                    # Phase 2: queue exhausted — deterministically re-scan every node
                    # for unexplored pairs and re-fill the queue.  This replaces the
                    # former rejection-sampling loop (_boltzmann_sample_task), which
                    # could miss valid pairs as the explored set grew large.
                    logger.info(
                        "Queue empty; re-scanning all nodes for unexplored pairs."
                    )
                    for node in self.graph.all_nodes():
                        self._enqueue_perturbations(node, force_add=True)
                    if len(self.queue) == 0:
                        logger.info(
                            "All candidate (EQ, pair) combinations exhausted. Stopping."
                        )
                        break
                    # Restart the loop so the freshly enqueued tasks are popped
                    # in priority order on the next iteration.
                    continue
    
                # ── Execute the task ────────────────────────────────────────────
                self._iteration += 1
                self.graph.last_iteration = self._iteration
    
                with open(history_log, "a", encoding="utf-8") as fh:
                    fh.write(
                        f"Iter: {self._iteration:06d} | Node: EQ{task.node_id:06d} "
                        f"| Priority: {task.priority:.6f} | AFIR: {task.afir_params}\n"
                    )
    
                run_dir = self._make_run_dir(task)
                try:
                    profile_dirs = self._run_autots(task, run_dir)
                except Exception as e:
                    logger.error("AutoTS failed for run %s: %s", run_dir, e)
                    self._flush_node_energy_updates()
                    self._save_run_metadata(run_dir, task, status="FAILED", profile_dirs=[])
                    self.graph.save(self.graph_json_path)
                    self._write_priority_log(priority_log)
                    continue
    
                logger.info(
                    "Iter %06d: _run_autots returned %d profile director%s.",
                    self._iteration, len(profile_dirs),
                    "y" if len(profile_dirs) == 1 else "ies",
                )
                for pdir in profile_dirs:
                    self._process_profile(pdir, run_dir)
    
                # Notify the queue about the updated graph (required by RCMCQueue
                # to recompute transient-population priorities over the full network).
                if hasattr(self.queue, "set_graph"):
                    self.queue.set_graph(self.graph)
    
                self._flush_node_energy_updates()
                self._save_run_metadata(run_dir, task, status="DONE", profile_dirs=profile_dirs)
                self.graph.save(self.graph_json_path)
                # Write after new nodes and tasks have been registered so the log
                # reflects the state produced by this iteration.
                self._write_priority_log(priority_log)
    
        finally:
            self.graph.save(self.graph_json_path)
            self.explored_log.close()
        
    def _collect_task_batch(self, n: int) -> list[tuple[ExplorationTask, str, str, int, int, int]]:
        """Pop up to *n* ready tasks and create their run directories.

        Returns a list of ``(task, run_dir, gamma_sign, atom_i, atom_j, iteration)``
        6-tuples — one per task that was selected.  The explored-pairs log
        is updated immediately for each task (before execution) so that
        concurrent workers do not re-select the same pair.  ``self._iteration``
        is pre-incremented once per task here so that each task in the batch
        receives a unique run directory name even when ``n_parallel > 1``.

        An empty list is returned when no runnable tasks are available,
        which signals :meth:`_run_batch_parallel` to stop the loop.
        """
        batch: list[tuple[ExplorationTask, str, str, int, int, int]] = []
        # Safety limit: if every task remaining in the queue is already
        # explored, stop after at most (current queue size + n) skips rather
        # than draining the entire queue one pop at a time.
        max_skip = len(self.queue) + n
        skip_count = 0
        while len(batch) < n:
            task = self.queue.pop()
            if task is None:
                # Queue empty — try to refill once.
                for node in self.graph.all_nodes():
                    self._enqueue_perturbations(node, force_add=True)
                task = self.queue.pop()
                if task is None:
                    break   # truly exhausted
            gamma_sign, atom_i, atom_j = self._parse_afir_task_key(task)
            if self.explored_log.has(task.node_id, atom_i, atom_j, gamma_sign):
                skip_count += 1
                if skip_count > max_skip:
                    logger.debug(
                        "_collect_task_batch: skip_count exceeded max_skip (%d); "
                        "breaking to avoid stall.",
                        max_skip,
                    )
                    break
                logger.debug(
                    "_collect_task_batch: skipping already-explored task "
                    "(EQ%06d, %d-%d, %s).",
                    task.node_id, atom_i, atom_j, gamma_sign,
                )
                continue
            self.explored_log.record(task.node_id, atom_i, atom_j, gamma_sign)
            # Pre-increment so every task in the batch gets its own unique
            # iteration number and therefore its own run directory.
            self._iteration += 1
            run_dir = self._make_run_dir(task, iteration=self._iteration)
            batch.append((task, run_dir, gamma_sign, atom_i, atom_j, self._iteration))
        return batch

    def _run_batch_parallel(
        self,
        batch: list[tuple[ExplorationTask, str, str, int, int, int]],
        history_log: str,
        priority_log: str,
    ) -> None:
        """Submit *batch* tasks concurrently and process all results.

        Each task is executed in its own spawned subprocess via
        :class:`concurrent.futures.ProcessPoolExecutor`.  After *all*
        subprocesses have finished (success or failure) the profiles are
        processed sequentially in submission order, the graph is saved
        once, and the priority log is updated.

        Parameters
        ----------
        batch:
            Output of :meth:`_collect_task_batch`.
        history_log, priority_log:
            Absolute paths to the two log files maintained by :meth:`run`.
        """
        # ── Build per-task configs and persist config_used.json ──────────
        task_configs: list[tuple[ExplorationTask, str, str, dict, int]] = []
        for task, run_dir, *_, iteration in batch:
            workspace = os.path.join(run_dir, "autots_workspace")
            config = copy.deepcopy(self.base_config)
            config["initial_mol_file"]              = task.xyz_file
            config["work_dir"]                      = workspace
            config.setdefault("step1_settings", {})
            config["step1_settings"]["manual_AFIR"] = task.afir_params
            config["run_step4"]                     = True

            config_used_path = os.path.join(run_dir, "config_used.json")
            try:
                with open(config_used_path, "w", encoding="utf-8") as fh:
                    json.dump(config, fh, indent=2, default=str)
            except Exception as exc:
                logger.warning(
                    "_run_batch_parallel: could not write config_used.json "
                    "(%s): %s", config_used_path, exc,
                )
            task_configs.append((task, run_dir, workspace, config, iteration))

        # ── Submit all tasks to the pool ──────────────────────────────────
        futures: dict[concurrent.futures.Future, tuple[ExplorationTask, str, str, int]] = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_parallel,
            mp_context=self._mp_ctx,
        ) as executor:
            for task, run_dir, workspace, config, iteration in task_configs:
                f = executor.submit(_autots_worker, config, run_dir, workspace)
                futures[f] = (task, run_dir, workspace, iteration)

            # ── Collect results in completion order via as_completed ─────────
            # as_completed() yields each Future as soon as its subprocess
            # finishes, so shorter tasks are processed without waiting for
            # slower ones in the same batch.  Graph updates still happen
            # sequentially (graph is not thread-safe), but we no longer block
            # the fastest results behind the slowest worker.
            results: list[tuple[ExplorationTask, str, list[str], str, int]] = []
            for f in concurrent.futures.as_completed(futures):
                task, run_dir, _workspace, iteration = futures[f]
                try:
                    profile_dirs: list[str] = f.result()
                    results.append((task, run_dir, profile_dirs, "DONE", iteration))
                except Exception as exc:
                    logger.error(
                        "_run_batch_parallel: AutoTS failed for run %s: %s",
                        run_dir, exc,
                    )
                    results.append((task, run_dir, [], "FAILED", iteration))

        # ── Process results sequentially (graph is not thread-safe) ───────
        for task, run_dir, profile_dirs, status, iteration in results:
            # _iteration was already incremented in _collect_task_batch for
            # each task; update the graph's last_iteration to the highest value.
            self.graph.last_iteration = max(self.graph.last_iteration, iteration)

            with open(history_log, "a", encoding="utf-8") as fh:
                fh.write(
                    f"Iter: {iteration:06d} | Node: EQ{task.node_id:06d} "
                    f"| Priority: {task.priority:.6f} | AFIR: {task.afir_params} "
                    f"| Status: {status}\n"
                )

            if status == "DONE":
                logger.info(
                    "Iter %06d (batch): _run_autots returned %d profile director%s.",
                    iteration, len(profile_dirs),
                    "y" if len(profile_dirs) == 1 else "ies",
                )
                for pdir in profile_dirs:
                    self._process_profile(pdir, run_dir)

            if hasattr(self.queue, "set_graph"):
                self.queue.set_graph(self.graph)

            self._flush_node_energy_updates()
            self._save_run_metadata(
                run_dir, task,
                status=status,
                profile_dirs=profile_dirs,
                iteration=iteration,
            )

        self.graph.save(self.graph_json_path)
        self._write_priority_log(priority_log)

    def _save_config_snapshot(self) -> None:
        """Copy the JSON config file to ``work_dir`` as ``config_snapshot.json``.

        The destination filename is always ``config_snapshot.json`` regardless
        of the original filename, so that every mapper output directory is
        self-contained.  When resuming an interrupted run the snapshot is
        overwritten with the current config so the file always reflects the
        settings of the most-recent invocation.

        The copy is skipped silently when:

        * ``config_file_path`` was not provided (``None``).
        * The source file does not exist or is not readable.
        * Source and destination are the same path (no-op copy).
        """
        if not self.config_file_path:
            return

        src_path = self.config_file_path
        if not os.path.isfile(src_path):
            logger.warning(
                "_save_config_snapshot: source config file not found at %s; "
                "snapshot will not be saved.",
                src_path,
            )
            return

        dst_path = os.path.join(self.work_dir, "config_snapshot.json")

        # Skip when source and destination are already the same file.
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            logger.debug(
                "_save_config_snapshot: source and destination are the same "
                "file (%s); skipping copy.",
                dst_path,
            )
            return

        try:
            shutil.copy2(src_path, dst_path)
            logger.info(
                "Config snapshot saved: %s -> %s",
                src_path, dst_path,
            )
        except Exception as exc:
            logger.warning(
                "_save_config_snapshot: failed to copy config to %s: %s",
                dst_path, exc,
            )

    # ------------------------------------------------------------------
    # Node registration helpers
    # ------------------------------------------------------------------

    def _register_seed_structure(self) -> None:
        seed_xyz = os.path.abspath(self.base_config["initial_mol_file"])
        node_id = self.graph.next_node_id()

        optimized_xyz_path, energy = self._run_initial_optimization(seed_xyz)

        saved_xyz = self._persist_node_xyz(optimized_xyz_path, node_id)

        try:
            symbols, coords = parse_xyz(saved_xyz)
        except Exception as e:
            raise RuntimeError(
                f"_register_seed_structure: failed to parse initial structure "
                f"at '{saved_xyz}': {e}"
            ) from e

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

        # Store the reference bond topology for the rearrangement filter.
        if self.exclude_bond_rearrangement and node.coords.size > 0:
            self._ref_bond_fingerprint = self.bond_checker.fingerprint(
                node.symbols, node.coords
            )
            self._ref_bond_symbols = list(node.symbols)
            self._ref_bond_coords  = node.coords.copy()
            logger.info(
                "_register_seed_structure: reference bond fingerprint stored: %s",
                self._ref_bond_fingerprint,
            )

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
            seed_xyz_rel = os.path.relpath(seed_xyz)
            opt_job = OptimizationJob(seed_xyz_rel)

            if opt_kwargs:
                opt_job.set_options(**opt_kwargs)

            opt_job.run()
            optimizer = opt_job.get_results()
            # --- Locate optimised geometry file ---
            optimized_xyz_path = seed_xyz
            potential_opt_files = glob.glob(optimizer.BPA_FOLDER_DIRECTORY+"*_optimized.xyz")
            if potential_opt_files:
                optimized_xyz_path = os.path.abspath(potential_opt_files[0])

            # --- Extract energy via get_results() -> Optimize instance ---
            # OptimizationJob itself holds no energy; it lives on the internal
            # multioptpy.optimization.Optimize object returned by get_results().
            energy: float | None = None

            if optimizer is not None:
                # Priority 1: use a canonical accessor method when available.
                # This is the most forward-compatible approach — if the Optimize
                # class exposes get_energy() or get_final_energy(), use it
                # directly rather than probing instance attributes.
                for method_name in ("get_energy", "get_final_energy", "get_scf_energy"):
                    method = getattr(optimizer, method_name, None)
                    if callable(method):
                        try:
                            raw = method()
                            if raw is not None:
                                energy = float(raw)
                                logger.info(
                                    "Initial optimization energy "
                                    "(optimizer.%s()): %.10f Ha",
                                    method_name, energy,
                                )
                                break   # success — stop trying methods
                        except Exception:
                            continue   # this method failed; try the next one

                # Priority 2: fall back to well-known instance attributes only
                # when no accessor method was found.  The list is intentionally
                # kept short; if none match, the warning below guides diagnosis.
                if energy is None:
                    for attr in ("energy", "final_energy", "scf_energy"):
                        raw = getattr(optimizer, attr, None)
                        if raw is not None:
                            try:
                                energy = float(raw)
                                logger.info(
                                    "Initial optimization energy (optimizer.%s): %.10f Ha",
                                    attr, energy,
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
                        logger.warning("Float attributes on optimizer: %s", float_attrs)

            return optimized_xyz_path, energy

        except Exception as e:
            logger.error(
                "Initial optimization failed: %s. Falling back to unoptimized geometry.", e
            )
            traceback.print_exc()
            return seed_xyz, None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _write_priority_log(self, priority_log: str) -> None:
        """Overwrite the priority log with the current queue state.

        Each EQ node is written as a single line showing its priority and the
        total number of pending AFIR pairs, rather than one line per pair.
        Nodes are sorted by priority descending (highest first).

        Format
        ------
        # Iter 000002 | queued=294 | explored_pairs=2 | ref_energy=-18.1413 Ha
        Node: EQ000000 | Priority: 1.000000 | pending_pairs=71
        Node: EQ000001 | Priority: 0.500000 | pending_pairs=68
        ...
        """
        ref_e = self.graph.reference_energy()
        with open(priority_log, "w", encoding="utf-8") as fh:
            fh.write(
                f"# Iter {self._iteration:06d} | "
                f"queued={len(self.queue)} | "
                f"explored_pairs={len(self.explored_log)} | "
                f"ref_energy={ref_e:.10f} Ha\n"
                if ref_e is not None else
                f"# Iter {self._iteration:06d} | queued={len(self.queue)} | "
                f"explored_pairs={len(self.explored_log)} | ref_energy=N/A\n"
            )

            # Group tasks by node_id; keep the representative priority for each
            # node (all tasks on the same node share the same priority value).
            node_priority: dict[int, float] = {}
            node_pair_count: dict[int, int] = {}
            for item in self.queue.export_queue_status():
                nid = item["node_id"]
                node_priority[nid] = item["priority"]
                node_pair_count[nid] = node_pair_count.get(nid, 0) + 1

            # Write one line per EQ node, sorted by priority descending.
            for nid in sorted(node_priority, key=lambda x: node_priority[x], reverse=True):
                fh.write(
                    f"Node: EQ{nid:06d} | "
                    f"Priority: {node_priority[nid]:.6f} | "
                    f"pending_pairs={node_pair_count[nid]}\n"
                )

    def _requeue_all_nodes(self) -> None:
        # ── Step 1: Restore the reference fingerprint from EQ0 ─────────────────
        if self.exclude_bond_rearrangement and self._ref_bond_fingerprint is None:
            seed_node = self.graph.get_node(0)
            if seed_node is not None and seed_node.coords.size > 0:
                self._ref_bond_fingerprint = self.bond_checker.fingerprint(
                    seed_node.symbols, seed_node.coords
                )
                self._ref_bond_symbols = list(seed_node.symbols)
                self._ref_bond_coords  = seed_node.coords.copy()
                logger.info(
                    "_requeue_all_nodes: reference bond fingerprint restored "
                    "from EQ0: %s",
                    self._ref_bond_fingerprint,
                )

        # ── Step 2: Reconstruct the bond-rearrangement exclusion list across all nodes ───────
        # Since `_bond_rearrangement_excluded_ids` is reset to empty after a resume,
        # re-evaluate the topology of each node to restore the exclusion list.
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
                    logger.info(
                        "_requeue_all_nodes: EQ%d restored to excluded_node_ids "
                        "(bond topology differs from seed structure).",
                        node.node_id,
                    )

        # ── Step 3: Rebuild the queues for all nodes ───────────────────────────
        for node in self.graph.all_nodes():
            self._enqueue_perturbations(node, force_add=True)

    def _make_run_dir(self, task: ExplorationTask, iteration: int | None = None) -> str:
        iter_num = self._iteration if iteration is None else iteration
        name = f"run_{iter_num:06d}_node{task.node_id}"
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
        iteration=None,
    ) -> None:
        info = {
            "iteration": iteration if iteration is not None else self._iteration,
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
        """Execute AutoTSWorkflow in an isolated spawned subprocess.
    
        The previous implementation called ``os.chdir(run_dir)`` in the mapper
        process itself, which was not thread-safe and caused CWD corruption when
        more than one task ran concurrently.  Now the CWD change happens
        exclusively inside a spawned child process so the mapper's working
        directory is never mutated.
    
        The child process runs :func:`_autots_worker`, which:
    
        1. ``os.chdir(run_dir)``
        2. Instantiates and runs ``AutoTSWorkflow``
        3. Returns the sorted list of ``*_Step4_Profile`` directories
    
        A non-zero exit code is converted to a ``RuntimeError`` so the caller's
        existing ``except Exception`` handler triggers unchanged.
    
        Parameters
        ----------
        task : ExplorationTask
            The task whose ``xyz_file`` and ``afir_params`` configure the run.
        run_dir : str
            Absolute path to the per-task directory created by
            :meth:`_make_run_dir`.
    
        Returns
        -------
        list[str]
            Sorted absolute paths of all Step-4 profile directories produced.
        """
        if AutoTSWorkflow is None:
            raise RuntimeError(
                "AutoTSWorkflow is not available. "
                "Install multioptpy to use the reaction network mapper."
            )
        workspace = os.path.join(run_dir, "autots_workspace")
    
        config = copy.deepcopy(self.base_config)
        config["initial_mol_file"] = task.xyz_file
        config["work_dir"]         = workspace
        config.setdefault("step1_settings", {})
        config["step1_settings"]["manual_AFIR"] = task.afir_params
        config["run_step4"]     = True
    
        # ── Persist the exact config passed to AutoTS ────────────────────
        # Written before the workflow starts so the file survives a crash.
        config_used_path = os.path.join(run_dir, "config_used.json")
        try:
            with open(config_used_path, "w", encoding="utf-8") as _fh:
                json.dump(config, _fh, indent=2, default=str)
            logger.debug("_run_autots: wrote config_used.json -> %s", config_used_path)
        except Exception as _exc:
            logger.warning(
                "_run_autots: could not write config_used.json (%s): %s",
                config_used_path, _exc,
            )
    
        # ── Run workflow in a spawned subprocess ─────────────────────────
        # Use mp_context.Process directly for the sequential path so that
        # each iteration avoids the ProcessPoolExecutor setup/teardown
        # overhead (executor pool creation, queue initialisation, etc.).
        # The return value travels back via a multiprocessing.Queue that
        # is safe to use across spawn boundaries.
        #
        # IMPORTANT — deadlock prevention
        # --------------------------------
        # result_q.get() MUST be called *before* proc.join().
        # If proc.join() is called first, and the worker's single put() blocks
        # because the OS pipe buffer is full (e.g. a very large traceback
        # string), the parent will wait forever for the child to exit while
        # the child waits for the parent to consume the queue — deadlock.
        # Consuming the item first guarantees the worker can finish and exit.
        result_q: multiprocessing.Queue = self._mp_ctx.Queue()  # type: ignore[type-arg]
        proc = self._mp_ctx.Process(
            target=_autots_worker_with_queue,
            args=(config, run_dir, workspace, result_q),
        )
        proc.start()
    
        # ── Step 1: crash-aware polling loop ─────────────────────────────
        # [FIX] The original implementation used result_q.get(timeout=None),
        # which blocks indefinitely.  This caused the mapper process to hang
        # permanently when the worker crashed (segfault, OOM kill, unhandled
        # C-level exception) before executing result_q.put(), because the
        # queue would never receive an item.
        #
        # A naive fixed-duration timeout (e.g. timeout=3600) solves the hang
        # but introduces a new problem: quantum chemistry calculations can
        # legitimately take many hours depending on molecular size and the
        # chosen level of theory.  A wall-clock limit would incorrectly abort
        # valid long-running calculations.
        #
        # Solution — poll proc.is_alive() every poll_interval seconds:
        #   - Worker alive  → calculation is still running normally; keep waiting.
        #   - Worker dead   → it crashed without putting a result; raise immediately.
        # This detects crashes quickly without imposing any time limit on
        # well-behaved calculations.
        #
        # An optional hard ceiling (self.worker_timeout_s, default None = disabled)
        # is provided for environments where an absolute wall-clock limit is
        # operationally required (e.g. HPC job schedulers).  It should not be
        # set in normal usage.
        import queue as _queue_mod
    
        poll_interval = 60  # seconds: crash-detection granularity.
                            # Smaller values detect crashes sooner at the cost
                            # of slightly more CPU wakeups.
        elapsed = 0.0
    
        while True:
            try:
                tag, payload = result_q.get(timeout=poll_interval)
                break  # result received successfully — exit the polling loop
            except _queue_mod.Empty:
                pass  # no result yet — check whether the process is still alive
    
            if not proc.is_alive():
                # The worker exited without putting anything on the queue.
                # This is unambiguously a crash: a healthy worker always calls
                # result_q.put() (either "ok" or "err") before returning.
                proc.join()
                raise RuntimeError(
                    f"_run_autots: worker process terminated unexpectedly without "
                    f"returning a result (exit_code={proc.exitcode}). "
                    "Possible causes: segfault, OOM kill, or unhandled C-level "
                    "exception in AutoTSWorkflow."
                )
    
            elapsed += poll_interval
    
            # ── Optional hard timeout (disabled by default) ───────────────
            # Activated only when mapper_settings["worker_timeout_s"] is set
            # to a positive integer.  Intended for HPC environments where the
            # job scheduler enforces a wall-clock limit and the mapper must
            # exit cleanly before that limit is reached.
            # In normal usage this branch is never taken (worker_timeout_s is None).
            if self.worker_timeout_s is not None and elapsed >= self.worker_timeout_s:
                proc.terminate()
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=10)
                raise RuntimeError(
                    f"_run_autots: worker exceeded hard timeout of "
                    f"{self.worker_timeout_s}s (elapsed={elapsed:.0f}s). "
                    "Consider increasing mapper_settings['worker_timeout_s'] or "
                    "setting it to null to disable the hard limit entirely."
                )
    
            logger.debug(
                "_run_autots: worker still running (elapsed=%.0fs, pid=%d).",
                elapsed, proc.pid,
            )
    
        # ── Step 2: drain any unexpected residual items (get_nowait loop) ─
        # The worker is designed to put exactly one item, but drain defensively
        # so that no unconsumed data keeps the child's feeder thread alive and
        # prevents clean exit.
        while True:
            try:
                result_q.get_nowait()
            except _queue_mod.Empty:
                break
    
        # ── Step 3: join after the queue is empty — never deadlocks ──────
        proc.join(timeout=120)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
    
        if tag == "err":
            # payload is a formatted traceback string (see _autots_worker_with_queue).
            # Wrap in RuntimeError so it can be raised and caught normally.
            raise RuntimeError(
                f"AutoTSWorkflow subprocess failed:\n{payload}"
            )
        profiles: list[str] = payload
        return profiles

    # ------------------------------------------------------------------
    # Energy backfill (Fix 2d: uses graph.invalidate_energy_cache())
    # ------------------------------------------------------------------

    def _flush_node_energy_updates(self) -> None:
        """Apply all accumulated node energy backfills.

        Design decisions (confirmed with user):
        * Called once per iteration, just before ``graph.save()``.
        * Both ``energy`` (electronic) and ``free_energy`` are updated when
          the existing value is ``None`` and the incoming value is not.
        * Existing TSEdge barrier values are **never** modified here — they
          always reflect the actual measured values from the IRC run that
          produced them (see Q3 design decision).
        """
        if not self._pending_node_updates:
            return

        for node_id, updates in self._pending_node_updates.items():
            node = self.graph.get_node(node_id)
            if node is None:
                logger.warning(
                    "_flush_node_energy_updates: node EQ%d not found in graph; skipping.",
                    node_id,
                )
                continue

            changed: list[str] = []

            new_e = updates.energy
            if new_e is not None and node.energy is None:
                node.energy = new_e
                changed.append(f"energy={new_e:.10f} Ha")
                
                # BUGFIX: Remove the node from the un-evaluated list so it no longer
                # bypasses the O(log N) energy window check in nodes_in_energy_window()
                # before the next _rebuild_indexes() call.
                try:
                    self.graph._node_none_ids.remove(node_id)
                except ValueError:
                    pass

            new_g = updates.free_energy
            if new_g is not None and node.free_energy is None:
                node.free_energy = new_g
                changed.append(f"free_energy={new_g:.10f} Ha")

            if changed:
                logger.info(
                    "_flush_node_energy_updates: EQ%d updated — %s",
                    node_id,
                    " ".join(changed),
                )
            else:
                logger.debug(
                    "_flush_node_energy_updates: EQ%d queued but no null fields "
                    "to fill (already populated by a prior update).",
                    node_id,
                )

        self._pending_node_updates.clear()

        # Node energies may have changed; invalidate the cached reference energy
        # via the public method instead of touching the private cache directly.
        self.graph.invalidate_energy_cache()

    # ------------------------------------------------------------------
    # Node identity and registration
    # ------------------------------------------------------------------

    def _find_or_register_node(
        self,
        xyz_file: str,
        energy: float | None,
        run_dir: str,
        free_energy: float | None = None,
    ) -> int | None:
        """Return the node_id of a matching EQNode, registering a new one if needed.

        Comparison pool
        ---------------
        Only :class:`EQNode` objects stored in ``self.graph`` are searched.
        TS saddle-point geometries held in :class:`TSEdge` are never involved
        in this comparison; they are handled exclusively by
        :meth:`_process_profile`.

        Parameters
        ----------
        free_energy : float | None
            Gibbs free energy (G_tot, Hartree) from thermochemistry analysis,
            or ``None`` when vibrational analysis was not performed.  Stored
            on the newly created :class:`EQNode` and used for priority
            calculations when available.
        """
        try:
            symbols, coords = parse_xyz(xyz_file)
        except Exception as exc:
            logger.warning(
                "_find_or_register_node: failed to parse EQ XYZ at %s: %s",
                xyz_file, exc,
            )
            return None

        # Narrow the candidate pool to nodes within the energy window.
        # When the incoming energy is None we fall back to all_nodes() so
        # that structures without parsed energies still undergo the full
        # geometric check (matching the original behaviour).
        # Nodes listed in excluded_node_ids are also skipped from the
        # comparison pool so that a structure identical to an excluded node
        # is treated as a new, independent node rather than being collapsed
        # into the excluded one.
        # Exception: nodes excluded automatically by the bond-rearrangement
        # filter (_bond_rearrangement_excluded_ids) are still included in
        # the comparison pool — they serve as valid deduplication references
        # even though they are not explored further.
        candidates: list[EQNode] = (
            self.graph.nodes_in_energy_window(energy, self.energy_tolerance)
            if energy is not None
            else self.graph.all_nodes()
        )
        for existing in candidates:
            if (
                existing.node_id in self.excluded_node_ids
                and existing.node_id not in self._bond_rearrangement_excluded_ids
            ):
                logger.debug(
                    "_find_or_register_node: EQ%d is in excluded_node_ids (manual) — "
                    "skipped from identity comparison pool.",
                    existing.node_id,
                )
                continue
            if existing.coords.size == 0:
                logger.debug(
                    "_find_or_register_node: EQ%d has empty coords, skipped.", existing.node_id
                )
                continue

            # Energy pre-filter log — the window already enforces the tolerance
            # for nodes with energy, so this branch fires only for the None-energy
            # fallback path (both sides None) or cross-None cases.
            if energy is None or existing.energy is None:
                logger.debug(
                    "_find_or_register_node: energy pre-filter skipped for EQ%d "
                    "(new_E=%s  existing_E=%s).",
                    existing.node_id,
                    f"{energy:.8f} Ha" if energy is not None else "None",
                    f"{existing.energy:.8f} Ha" if existing.energy is not None else "None",
                )

            rmsd = self.checker.compute_rmsd(symbols, coords, existing.symbols, existing.coords)
            if rmsd < self.checker.rmsd_threshold:
                logger.debug(
                    "_find_or_register_node: MATCH EQ%d  RMSD=%.4f A < threshold=%.3f A.",
                    existing.node_id, rmsd, self.checker.rmsd_threshold,
                )
                # ── Energy backfill ───────────────────────────────────────
                # When the matched node was registered without energy or free-
                # energy data (e.g. seed node from a plain optimisation) and
                # the new incoming structure carries those values, queue an
                # update so that _flush_node_energy_updates() can populate the
                # null fields before the next graph.save().
                needs_update = NodeUpdateData(
                    energy=energy if existing.energy is None else None,
                    free_energy=free_energy if existing.free_energy is None else None,
                )
                if needs_update.energy is not None or needs_update.free_energy is not None:
                    prev = self._pending_node_updates.get(
                        existing.node_id, NodeUpdateData()
                    )
                    # Only overwrite fields that are still absent in prior queued updates.
                    if prev.energy is None and needs_update.energy is not None:
                        prev.energy = needs_update.energy
                    if prev.free_energy is None and needs_update.free_energy is not None:
                        prev.free_energy = needs_update.free_energy
                    self._pending_node_updates[existing.node_id] = prev
                    logger.debug(
                        "_find_or_register_node: queued backfill for EQ%d: %s",
                        existing.node_id, needs_update,
                    )
                return existing.node_id
            else:
                logger.debug(
                    "_find_or_register_node: no match EQ%d  RMSD=%.4f A >= threshold=%.3f A.",
                    existing.node_id, rmsd, self.checker.rmsd_threshold,
                )

        node_id   = self.graph.next_node_id()
        saved_xyz = self._persist_node_xyz(xyz_file, node_id)
        new_node  = EQNode(
            node_id=node_id,
            xyz_file=saved_xyz,
            energy=energy,
            symbols=symbols,
            coords=coords,
            source_run_dir=run_dir,
            free_energy=free_energy,
        )
        self.graph.add_node(new_node)

        # ── Bond-rearrangement filter ──────────────────────────────────────
        if (
            self.exclude_bond_rearrangement
            and self._ref_bond_fingerprint is not None
            and new_node.coords.size > 0
        ):
            rearranged = self.bond_checker.has_rearrangement(
                self._ref_bond_symbols, self._ref_bond_coords,
                new_node.symbols, new_node.coords,
            )
            if rearranged:
                self.excluded_node_ids.add(node_id)
                self._bond_rearrangement_excluded_ids.add(node_id)
                logger.info(
                    "_find_or_register_node: EQ%d has a different bond topology "
                    "from the seed structure — added to excluded_node_ids "
                    "(exclude_bond_rearrangement=True).",
                    node_id,
                )

        ref_e = self.graph.reference_energy()
        if ref_e is None or new_node.energy is None:
            self._enqueue_perturbations(new_node, force_add=True)
        else:
            self._enqueue_perturbations(new_node, force_add=False)

        return node_id

    # ------------------------------------------------------------------
    # Exploration state helpers (Fix 2d: uses public methods)
    # ------------------------------------------------------------------

    def _node_has_been_explored(self, node_id: int) -> bool:
        """Return ``True`` if *node_id* has at least one recorded exploration.

        Uses the public :meth:`ExploredPairsLog.has_any_for_node` method
        instead of directly accessing the private ``_explored_node_ids`` set.
        """
        return self.explored_log.has_any_for_node(node_id)

    # ------------------------------------------------------------------
    # Perturbation enqueuing (Fix 2d: uses public methods)
    # ------------------------------------------------------------------

    def _enqueue_perturbations(self, node: EQNode, force_add: bool = False) -> None:
        if node.coords.size == 0:
            return

        # ── Exclusion check ───────────────────────────────────────────────
        # Exclusion is applied only from the second exploration onwards so
        # that each excluded node is still explored at least once.
        #
        # Special case: when the graph contains only one node, all exclusion
        # rules are disabled so that the sole available node is never silently
        # skipped (the network cannot grow if the only node is excluded).
        n_total_nodes = self.graph.node_count
        if n_total_nodes == 1 and node.node_id in self.excluded_node_ids:
            logger.debug(
                "_enqueue_perturbations: EQ%d is in excluded_node_ids but "
                "graph has only 1 node — exclusion suppressed.",
                node.node_id,
            )
        elif node.node_id in self.excluded_node_ids and self._node_has_been_explored(node.node_id):
            logger.debug(
                "_enqueue_perturbations: EQ%d is in excluded_node_ids and has "
                "already been explored at least once — skipped.",
                node.node_id,
            )
            return

        ref_e = self.graph.reference_energy()

        if not force_add and node.energy is not None and ref_e is not None:
            accepted = self.queue.should_add(node, ref_e)
        else:
            accepted = True

        if not accepted:
            return

        # ── Build the complete candidate pool ───────────────────────────
        # Use get_candidate_pairs (returns all valid pairs without sampling)
        # so we can filter before sampling, avoiding the bug where
        # generate_afir_perturbations could randomly select only already-
        # explored pairs and produce zero useful tasks.
        all_candidates = self.perturber.get_candidate_pairs(node.symbols, node.coords)
        if not all_candidates:
            return

        pos_gamma_str = f"{self.perturber.afir_gamma_kJmol:.6g}"
        neg_gamma_str = f"{-self.perturber.afir_gamma_kJmol:.6g}"
        gamma_signs: list[str] = ["+"]
        if self.perturber.include_negative_gamma:
            gamma_signs.append("-")

        # ── Filter out already-explored and already-queued pairs ─────────
        # Use public methods instead of directly accessing private sets.
        nid = node.node_id
        sign_to_gamma = {"+": pos_gamma_str, "-": neg_gamma_str}

        unexplored: list[tuple[int, int, str]] = [
            (i0, j0, sign)
            for (i0, j0) in all_candidates
            for sign in gamma_signs
            if not self.explored_log.has(nid, i0 + 1, j0 + 1, sign)
            and not self.queue.is_submitted(
                (nid, (sign_to_gamma[sign], str(i0 + 1), str(j0 + 1)))
            )
        ]

        if not unexplored:
            logger.debug(
                "_enqueue_perturbations: EQ%d — all candidate pairs already "
                "explored or queued; nothing to enqueue.",
                node.node_id,
            )
            return

        # ── Sample up to max_pairs from the unexplored combinations ──────
        n_sel = min(self.perturber.max_pairs, len(unexplored))
        chosen_indices = self._rng.choice(len(unexplored), size=n_sel, replace=False)

        # Use free energy for ΔE when available; fall back to electronic energy.
        # The reference (ref_e) is already the unified reference from
        # NetworkGraph.reference_energy() (G-based when any node has G_tot).
        eff_e = node.effective_energy
        delta_e = (
            (eff_e - ref_e)
            if (eff_e is not None and ref_e is not None)
            else 0.0
        )

        for idx in chosen_indices:
            i0, j0, sign = unexplored[int(idx)]
            atom_i, atom_j = i0 + 1, j0 + 1
            gamma_str = sign_to_gamma[sign]
            afir_params = [gamma_str, str(atom_i), str(atom_j)]
            task = ExplorationTask(
                node_id=node.node_id,
                xyz_file=node.xyz_file,
                afir_params=afir_params,
                metadata={
                    "delta_E_hartree":         delta_e,
                    "source_node_energy":      node.energy,
                    "source_node_free_energy": getattr(node, "free_energy", None),
                },
            )
            self.queue.push(task)

    # ------------------------------------------------------------------
    # File persistence helpers
    # ------------------------------------------------------------------

    def _persist_node_xyz(self, src_xyz: str, node_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"EQ{node_id:06d}.xyz")
        try:
            if os.path.abspath(src_xyz) != os.path.abspath(dst):
                shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception as exc:
            error_msg = (
                f"Failed to persist node XYZ file for EQ{node_id:06d}. "
                f"Cannot copy from '{src_xyz}' to '{dst}'. "
                f"Please check disk space, directory existence, and access permissions."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

    def _persist_ts_xyz(self, src_xyz: str, edge_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"TS{edge_id:06d}.xyz")
        try:
            if os.path.abspath(src_xyz) != os.path.abspath(dst):
                shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception as exc:
            error_msg = (
                f"Failed to persist edge XYZ file for TS{edge_id:06d}. "
                f"Cannot copy from '{src_xyz}' to '{dst}'. "
                f"Please check disk space, directory existence, and access permissions."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc