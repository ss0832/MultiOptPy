"""
mapper.py - Chemical Reaction Network Mapper
============================================

Autonomously maps a chemical reaction network by running
AutoTSWorkflow (autots.py) as the search engine iteratively.

Each AutoTSWorkflow invocation is isolated in its own run directory
under output_dir/runs/run_NNNN/.

Module layout
-------------
Section 1 : Constants and XYZ utilities
Section 2 : StructureChecker  - structure identity via Kabsch + Hungarian
Section 3 : ExplorationQueue  - abstract priority queue (extension point)
            BoltzmannQueue    - concrete Boltzmann implementation
Section 4 : PerturbationGenerator - AFIR / dihedral parameter generation
Section 5 : EQNode, TSEdge, NetworkGraph - graph data model
Section 6 : ProfileParser  - parse AutoTSWorkflow Step-4 output directories
Section 7 : ReactionNetworkMapper - main control loop

Dependencies: numpy, scipy, networkx
No cheminformatics libraries (rdkit etc.) are used.
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import shutil
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from multioptpy.Wrapper.autots import AutoTSWorkflow
except ImportError as _autots_import_err:
    print(f"[mapper] Warning: could not import AutoTSWorkflow: {_autots_import_err}")
    AutoTSWorkflow = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level physical constants
# ---------------------------------------------------------------------------
HARTREE_TO_KCALMOL: float = 627.509474
K_B_HARTREE: float = 3.166811563e-6   # Boltzmann constant [Ha/K]


# ===========================================================================
# Section 1 : XYZ utilities
# ===========================================================================

def parse_xyz(filepath: str) -> tuple[list[str], np.ndarray]:
    """
    Read a standard XYZ file and return (element symbols, coordinate array).

    Supports both the standard format (first line = atom count, second =
    comment) and files without the header lines. Only the first frame is
    read from multi-frame trajectory files.

    Parameters
    ----------
    filepath : str
        Path to the XYZ file.

    Returns
    -------
    symbols : list[str]
        Element symbols, length N.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates in the file's native unit (typically Angstrom).

    Raises
    ------
    ValueError
        If no atomic coordinates can be parsed from the file.
    """
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    # Determine whether a standard XYZ header exists.
    # A valid header has an integer-only first non-blank line.
    n_atoms: int | None = None
    data_start: int = 0

    non_blank = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
    if non_blank and non_blank[0][1].isdigit():
        n_atoms = int(non_blank[0][1])
        data_start = non_blank[0][0] + 2  # skip atom-count line and comment line

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
        # Stop after reading exactly n_atoms lines when the count is known.
        if n_atoms is not None and len(symbols) >= n_atoms:
            break

    if not symbols:
        raise ValueError(f"No atomic coordinates found in: {filepath}")

    return symbols, np.array(coords_raw, dtype=float)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute the pairwise distance matrix (N x N) from Cartesian coordinates."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    return np.sqrt((diff ** 2).sum(axis=-1))                     # (N, N)


# ===========================================================================
# Section 2 : StructureChecker
# ===========================================================================

class StructureChecker:
    """
    Decide whether two molecular structures are identical by RMSD.

    Algorithm
    ---------
    1. Translate both structures to their centroid.
    2. Apply a rough pre-alignment along principal inertia axes.
    3. Build a cost matrix C_ij = ||x_i^A - x_j^B||^2 for each element
       block and solve the optimal atom mapping with the Hungarian method.
    4. Compute the minimum RMSD via the Kabsch algorithm.

    Parameters
    ----------
    rmsd_threshold : float
        Structures with RMSD < threshold are considered identical [Angstrom].
        Default: 0.30 A.
    """

    def __init__(self, rmsd_threshold: float = 0.30) -> None:
        self.rmsd_threshold = rmsd_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def are_same(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> bool:
        """Return True when the two structures are considered identical."""
        return self.compute_rmsd(sym_a, coords_a, sym_b, coords_b) < self.rmsd_threshold

    def compute_rmsd(
        self,
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> float:
        """
        Compute the minimum RMSD between two structures via Hungarian + Kabsch.

        Returns inf when the structures cannot be compared (different atom
        count or element composition).
        """
        if len(sym_a) != len(sym_b):
            return float("inf")
        if set(sym_a) != set(sym_b):
            return float("inf")

        ca = coords_a - coords_a.mean(axis=0)
        cb = coords_b - coords_b.mean(axis=0)
        cb = self._principal_axis_align(cb)

        perm = self._hungarian_mapping(sym_a, ca, sym_b, cb)
        if perm is None:
            return float("inf")

        return self._kabsch_rmsd(ca, cb[perm])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _principal_axis_align(coords: np.ndarray) -> np.ndarray:
        """Rotate coords so principal inertia axes align with x, y, z."""
        if len(coords) < 2:
            return coords
        _, eigvecs = np.linalg.eigh(np.cov(coords.T))
        return coords @ eigvecs  # (N, 3)

    @staticmethod
    def _hungarian_mapping(
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> list[int] | None:
        """
        Find the optimal atom permutation of B that minimises the sum of
        squared distances to A, element-by-element.

        Returns None when element counts disagree between the two structures.
        """
        perm: list[int | None] = [None] * len(sym_a)

        for elem in set(sym_a):
            idx_a = [i for i, s in enumerate(sym_a) if s == elem]
            idx_b = [i for i, s in enumerate(sym_b) if s == elem]
            if len(idx_a) != len(idx_b):
                return None

            # Cost matrix: C_ij = ||x_i^A - x_j^B||^2
            sub_a = coords_a[idx_a]                                 # (m, 3)
            sub_b = coords_b[idx_b]                                 # (m, 3)
            diff = sub_a[:, np.newaxis, :] - sub_b[np.newaxis, :, :]
            cost = (diff ** 2).sum(axis=-1)

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                perm[idx_a[r]] = idx_b[c]

        return perm  # type: ignore[return-value]

    @staticmethod
    def _kabsch_rmsd(pa: np.ndarray, pb: np.ndarray) -> float:
        """
        Compute the minimum RMSD between two centred point sets using
        the Kabsch algorithm.  Handles improper rotations (reflections)
        by correcting the sign of the determinant.
        """
        H = pb.T @ pa                         # (3, 3) cross-covariance
        U, _, Vt = np.linalg.svd(H)
        # Sign correction: prevent reflection when det(R) = -1
        sign_d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, sign_d])
        R = Vt.T @ D @ U.T                   # optimal rotation matrix
        diff = pa - pb @ R.T
        return float(np.sqrt((diff ** 2).sum() / len(pa)))


# ===========================================================================
# Section 3 : ExplorationQueue  (extension point)
# ===========================================================================

@dataclass
class ExplorationTask:
    """
    A single unit of work in the exploration queue.

    Attributes
    ----------
    node_id : int
        ID of the EQ node used as the starting structure.
    xyz_file : str
        Absolute path to the input XYZ file (must be absolute).
    afir_params : list[str]
        Arguments for step1_settings["manual_AFIR"]
        (see interface.py parser_for_biasforce, -ma option).
        Format: [gamma_kJmol, frag1_atoms, frag2_atoms, ...]
        Example: ["100.0", "1,2", "3,4,5"]
    priority : float
        Higher value means higher priority (set automatically on push).
    metadata : dict
        Arbitrary key-value data used by the queue strategy.
        Common keys:
            "delta_E_hartree"    : float  relative energy vs reference [Ha]
            "barrier_kcalmol"    : float  activation energy [kcal/mol]
            "source_node_energy" : float  origin node energy [Ha]
    """
    node_id: int
    xyz_file: str
    afir_params: list[str]
    priority: float = 0.0
    metadata: dict = field(default_factory=dict)


class ExplorationQueue(ABC):
    """
    Abstract base class for exploration priority queues.

    HOW TO ADD A NEW PRIORITY STRATEGY
    ------------------------------------
    Subclass ExplorationQueue and implement exactly two methods:

        compute_priority(task) -> float
            Return a score for task; higher = executed first.

        should_add(node, reference_energy, **kwargs) -> bool
            Return True to enqueue perturbations from this EQ node.

    Pass the subclass instance to ReactionNetworkMapper(queue=...).

    Example (prefer low-barrier routes):
    ::
        class LowBarrierQueue(ExplorationQueue):
            def compute_priority(self, task):
                return -task.metadata.get("barrier_kcalmol", 0.0)
            def should_add(self, node, reference_energy, **kwargs):
                return True
    """

    def __init__(self) -> None:
        self._tasks: list[ExplorationTask] = []
        # Track (node_id, frozen afir_params) pairs already enqueued to
        # avoid submitting the exact same job twice.
        self._submitted: set[tuple] = set()

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------

    def push(self, task: ExplorationTask) -> bool:
        """
        Compute the task priority and add it to the queue.

        Returns False (and does nothing) when an identical task
        (same node_id and afir_params) has already been submitted.
        """
        key = (task.node_id, tuple(task.afir_params))
        if key in self._submitted:
            logger.debug("Queue: duplicate task skipped  node=%d  afir=%s",
                         task.node_id, task.afir_params)
            return False

        task.priority = self.compute_priority(task)
        self._tasks.append(task)
        self._tasks.sort(key=lambda t: t.priority, reverse=True)
        self._submitted.add(key)
        logger.debug("Queue: push  node=%d  priority=%.4f  afir=%s",
                     task.node_id, task.priority, task.afir_params)
        return True

    def pop(self) -> ExplorationTask | None:
        """Remove and return the highest-priority task, or None if empty."""
        return self._tasks.pop(0) if self._tasks else None

    def __len__(self) -> int:
        return len(self._tasks)

    def pending_summary(self) -> list[dict]:
        """Return a JSON-serialisable snapshot of the pending tasks."""
        return [
            {
                "node_id": t.node_id,
                "priority": t.priority,
                "afir_params": t.afir_params,
                "metadata": {
                    k: (float(v) if isinstance(v, float) else v)
                    for k, v in t.metadata.items()
                },
            }
            for t in self._tasks
        ]

    # ------------------------------------------------------------------
    # Abstract methods  (implement in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_priority(self, task: ExplorationTask) -> float:
        """
        Return a priority score for the given task (higher = first).

        Useful metadata keys (may not always be present):
            "delta_E_hartree"    : float  energy difference vs reference [Ha]
            "barrier_kcalmol"    : float  forward activation energy [kcal/mol]
            "source_node_energy" : float  energy of the origin node [Ha]
        """

    @abstractmethod
    def should_add(
        self,
        node: EQNode,
        reference_energy: float,
        **kwargs,
    ) -> bool:
        """
        Decide whether to enqueue new perturbation tasks for an EQ node.

        Parameters
        ----------
        node : EQNode
            Newly registered equilibrium structure.
        reference_energy : float
            Energy of the most stable known structure [Ha].
        **kwargs :
            Strategy-specific extra data (e.g. barrier_kcalmol=...).
        """


class BoltzmannQueue(ExplorationQueue):
    """
    Priority queue based on the Boltzmann distribution (default strategy).

    Acceptance probability:
        P = min(1, exp(-dE / k_B * T))

    Priority score = P  in [0, 1], so more stable structures are explored
    first while higher-energy structures are accepted stochastically.

    Parameters
    ----------
    temperature_K : float
        System temperature in Kelvin.  Higher T -> more exploratory.
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(self, temperature_K: float = 300.0, rng_seed: int = 42) -> None:
        super().__init__()
        self.temperature_K = temperature_K
        self._rng = np.random.default_rng(rng_seed)

    def compute_priority(self, task: ExplorationTask) -> float:
        """
        Boltzmann weight from metadata["delta_E_hartree"].
        dE <= 0 always returns 1.0 (downhill = highest priority).
        """
        delta_e: float = task.metadata.get("delta_E_hartree", 0.0)
        if delta_e <= 0.0:
            return 1.0
        return min(1.0, float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K))))

    def should_add(
        self,
        node: EQNode,
        reference_energy: float,
        **kwargs,
    ) -> bool:
        """
        Monte-Carlo acceptance: accept with probability P = min(1, exp(-dE/k_BT)).
        dE = node.energy - reference_energy.
        """
        delta_e = node.energy - reference_energy
        if delta_e <= 0.0:
            return True
        p = float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K)))
        accepted = bool(self._rng.random() < p)
        logger.debug("Boltzmann: dE=%.6f Ha  T=%.1f K  P=%.4f  -> %s",
                     delta_e, self.temperature_K, p,
                     "accept" if accepted else "reject")
        return accepted


# ===========================================================================
# Section 4 : PerturbationGenerator
# ===========================================================================

class PerturbationGenerator:
    """
    Generate AFIR and dihedral-angle perturbation parameters from
    the coordinate geometry of an EQ structure.

    Perturbation types
    ------------------
    1. AFIR push/pull:
       Select atom pairs within a distance range from the distance matrix
       and format them as manual_AFIR strings for step1_settings.
       Reference: AFIRPotential.py, interface.py (-ma option).

    2. Dihedral-angle harmonic potential (optional):
       Detect rotatable dihedrals from the bonding graph and generate
       keep_dihedral_angle parameters corresponding to the spec equation:
           V(phi) = 0.5 * k * (phi - phi_target)^2
       Reference: keep_dihedral_angle_potential.py, interface.py (-kda option).

    Parameters
    ----------
    afir_gamma_kJmol : float
        AFIR gamma value [kJ/mol].  Positive = attractive, negative = repulsive.
    max_pairs : int
        Maximum number of AFIR perturbations generated per call.
    dist_lower_ang : float
        Lower bound of the inter-atomic distance selection window [Angstrom].
    dist_upper_ang : float
        Upper bound [Angstrom].
    rng_seed : int
        Random seed.
    """

    def __init__(
        self,
        afir_gamma_kJmol: float = 100.0,
        max_pairs: int = 5,
        dist_lower_ang: float = 1.5,
        dist_upper_ang: float = 5.0,
        rng_seed: int = 0,
    ) -> None:
        self.afir_gamma_kJmol = afir_gamma_kJmol
        self.max_pairs = max_pairs
        self.dist_lower_ang = dist_lower_ang
        self.dist_upper_ang = dist_upper_ang
        self._rng = np.random.default_rng(rng_seed)

    def generate_afir_perturbations(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[list[str]]:
        """
        Build AFIR perturbation parameter lists from the distance matrix.

        Each returned list can be assigned directly to
        step1_settings["manual_AFIR"] in the AutoTSWorkflow config.
        Format: ["gamma_kJmol", "frag1_1indexed", "frag2_1indexed"]
        Example: ["100.0", "1", "3"]

        Returns
        -------
        list[list[str]]
            One entry per perturbation pattern; may be empty if no atom
            pair falls within the distance window.
        """
        n = len(symbols)
        if n < 2:
            return []

        dmat = distance_matrix(coords)
        candidates: list[tuple[int, int]] = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if self.dist_lower_ang <= dmat[i, j] <= self.dist_upper_ang
        ]

        if not candidates:
            logger.debug("PerturbationGenerator: no atom pairs in distance window")
            return []

        n_sel = min(self.max_pairs, len(candidates))
        chosen = self._rng.choice(len(candidates), size=n_sel, replace=False)

        gamma_str = f"{self.afir_gamma_kJmol:.6g}"
        result: list[list[str]] = []
        for idx in chosen:
            i, j = candidates[int(idx)]
            result.append([gamma_str, str(i + 1), str(j + 1)])  # 1-indexed

        logger.debug("PerturbationGenerator: generated %d AFIR perturbations", len(result))
        return result

    def generate_dihedral_perturbations(
        self,
        symbols: list[str],
        coords: np.ndarray,
        spring_const_au: float = 0.1,
        bond_threshold_ang: float = 1.8,
        n_targets: int = 3,
        delta_phi_deg: float = 90.0,
    ) -> list[dict]:
        """
        Detect rotatable dihedral angles and produce harmonic-potential
        perturbation parameters (interface.py -kda format).

        Each returned dict has:
            "keep_dihedral_angle" : list[str]  - value for step1_settings
            "description"         : str        - human-readable summary

        The target angle is the current dihedral shifted by delta_phi_deg,
        implementing V(phi) = 0.5 * k * (phi - phi_target)^2.
        """
        n = len(symbols)
        if n < 4:
            return []

        dmat = distance_matrix(coords)
        adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
        bonds: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if dmat[i, j] < bond_threshold_ang:
                    bonds.append((i, j))
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        # Enumerate i-j-k-l chains (dihedrals)
        dihedrals: list[tuple[int, int, int, int]] = []
        for j, k in bonds:
            for i in adjacency[j]:
                if i == k:
                    continue
                for lv in adjacency[k]:
                    if lv == j or lv == i:
                        continue
                    dihedrals.append((i, j, k, lv))

        if not dihedrals:
            return []

        n_sel = min(n_targets, len(dihedrals))
        chosen = self._rng.choice(len(dihedrals), size=n_sel, replace=False)

        results: list[dict] = []
        for idx in chosen:
            i, j, k, lv = dihedrals[int(idx)]
            current_deg = float(np.degrees(
                _calc_dihedral(coords[i], coords[j], coords[k], coords[lv])
            ))
            target_deg = current_deg + delta_phi_deg
            # -kda format: [spring_const(au), target_angle(deg), atom1,atom2,atom3,atom4]
            kda = [f"{spring_const_au:.6g}", f"{target_deg:.4f}",
                   f"{i+1},{j+1},{k+1},{lv+1}"]
            results.append({
                "keep_dihedral_angle": kda,
                "description": (
                    f"dihedral ({i+1}-{j+1}-{k+1}-{lv+1}): "
                    f"current={current_deg:.1f} deg  "
                    f"target={target_deg:.1f} deg"
                ),
            })

        logger.debug("PerturbationGenerator: generated %d dihedral perturbations",
                     len(results))
        return results


def _calc_dihedral(
    p0: np.ndarray, p1: np.ndarray,
    p2: np.ndarray, p3: np.ndarray,
) -> float:
    """Compute the dihedral angle of four points in [-pi, pi] radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if nn1 < 1e-10 or nn2 < 1e-10:
        return 0.0
    n1 /= nn1
    n2 /= nn2
    b2_hat = b2 / (np.linalg.norm(b2) + 1e-12)
    return float(np.arctan2(
        np.dot(np.cross(n1, n2), b2_hat),
        np.dot(n1, n2),
    ))


# ===========================================================================
# Section 5 : Graph data model
# ===========================================================================

@dataclass
class EQNode:
    """
    An equilibrium (minimum-energy) structure node in the reaction network.

    Attributes
    ----------
    node_id : int
    xyz_file : str
        Absolute path to the representative XYZ file for this node.
    energy : float
        Electronic energy [Hartree].  Set to None when unknown
        (e.g. the initial structure before any calculation).
    symbols : list[str]
        Element symbols, length N.
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates [Angstrom].
    source_run_dir : str
        Absolute path of the run directory that produced this node.
        "initial" for the seed structure.
    extra : dict
        Reserved for user-defined metadata.
    """
    node_id: int
    xyz_file: str
    energy: float | None
    symbols: list[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    source_run_dir: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def has_real_energy(self) -> bool:
        """True when energy is a computed quantum-chemical value."""
        return self.energy is not None and self.source_run_dir != "initial"

    def to_dict(self) -> dict:
        return {
            "node_id":        self.node_id,
            "xyz_file":       self.xyz_file,
            "energy_hartree": self.energy,
            "symbols":        self.symbols,
            "source_run_dir": self.source_run_dir,
            **self.extra,
        }


@dataclass
class TSEdge:
    """
    A transition state connecting two EQ nodes in the reaction network.

    Attributes
    ----------
    edge_id : int
    node_id_1, node_id_2 : int
        IDs of the two connected EQ nodes (endpoint_1 and endpoint_2).
    ts_xyz_file : str
        Absolute path to the TS structure file.
    ts_energy : float
        TS electronic energy [Hartree].
    barrier_fwd : float | None
        Activation energy from node_1 to TS [kcal/mol].
    barrier_rev : float | None
        Activation energy from node_2 to TS [kcal/mol].
    source_run_dir : str
        Run directory that produced this edge.
    extra : dict
        Reserved for user-defined metadata.
    """
    edge_id: int
    node_id_1: int
    node_id_2: int
    ts_xyz_file: str
    ts_energy: float
    barrier_fwd: float | None = None
    barrier_rev: float | None = None
    source_run_dir: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
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


class NetworkGraph:
    """
    Manage EQNode vertices and TSEdge edges as a NetworkX MultiGraph.

    Serialisation uses JSON for both persistence and resuming.
    """

    def __init__(self) -> None:
        self._nx: nx.MultiGraph = nx.MultiGraph()
        self._nodes: dict[int, EQNode] = {}
        self._edges: dict[int, TSEdge] = {}
        self._node_counter: int = 0
        self._edge_counter: int = 0

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: EQNode) -> None:
        self._nodes[node.node_id] = node
        self._nx.add_node(node.node_id, **node.to_dict())

    def get_node(self, node_id: int) -> EQNode | None:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[EQNode]:
        return list(self._nodes.values())

    def next_node_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: TSEdge) -> None:
        self._edges[edge.edge_id] = edge
        self._nx.add_edge(
            edge.node_id_1, edge.node_id_2,
            key=edge.edge_id, **edge.to_dict(),
        )

    def all_edges(self) -> list[TSEdge]:
        return list(self._edges.values())

    def next_edge_id(self) -> int:
        eid = self._edge_counter
        self._edge_counter += 1
        return eid

    # ------------------------------------------------------------------
    # Reference energy  (uses only nodes with real computed energies)
    # ------------------------------------------------------------------

    def reference_energy(self) -> float | None:
        """
        Return the lowest energy among nodes that have a real computed energy.
        Returns None when no such node exists yet (e.g. only the seed node).
        """
        real_energies = [
            n.energy for n in self._nodes.values() if n.has_real_energy
        ]
        return min(real_energies) if real_energies else None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Write the full graph to a JSON file."""
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
        logger.info("Graph saved -> %s", filepath)

    def load(self, filepath: str) -> None:
        """
        Restore the graph from a JSON file.

        WARNING: clears any existing state before loading.
        """
        # Reset all state to avoid merging with stale data.
        self._nx = nx.MultiGraph()
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
                symbols=nd.get("symbols", []),
                source_run_dir=nd.get("source_run_dir", ""),
            )
            # Re-load coordinates from the XYZ file if it still exists.
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
                ts_xyz_file=ed["ts_xyz_file"],
                ts_energy=ed["ts_energy_hartree"],
                barrier_fwd=ed.get("barrier_fwd_kcal"),
                barrier_rev=ed.get("barrier_rev_kcal"),
                source_run_dir=ed.get("source_run_dir", ""),
            )
            self.add_edge(edge)

        # Advance counters past the loaded IDs to prevent collisions.
        if self._nodes:
            self._node_counter = max(self._nodes) + 1
        if self._edges:
            self._edge_counter = max(self._edges) + 1

        logger.info("Graph loaded: %d nodes, %d edges  <- %s",
                    len(self._nodes), len(self._edges), filepath)

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary sorted by energy."""
        lines = [
            f"[NetworkGraph]  nodes={len(self._nodes)}  edges={len(self._edges)}"
        ]
        ref = self.reference_energy()

        for node in sorted(self._nodes.values(),
                           key=lambda n: (n.energy is None, n.energy)):
            if node.energy is not None and ref is not None:
                rel = (node.energy - ref) * HARTREE_TO_KCALMOL
                e_str = f"{node.energy:+.8f} Ha  (+{rel:.2f} kcal/mol)"
            elif node.energy is not None:
                e_str = f"{node.energy:+.8f} Ha"
            else:
                e_str = "energy unknown"
            lines.append(f"  EQ{node.node_id:04d}: {e_str}  [{node.xyz_file}]")

        for edge in self._edges.values():
            fwd = f"{edge.barrier_fwd:.2f}" if edge.barrier_fwd is not None else "N/A"
            rev = f"{edge.barrier_rev:.2f}" if edge.barrier_rev is not None else "N/A"
            lines.append(
                f"  TS{edge.edge_id:04d}: "
                f"EQ{edge.node_id_1} -- EQ{edge.node_id_2}  "
                f"Ea(fwd)={fwd} kcal/mol  Ea(rev)={rev} kcal/mol"
            )
        return "\n".join(lines)


# ===========================================================================
# Section 6 : ProfileParser
# ===========================================================================

class ProfileParser:
    """
    Parse a single AutoTSWorkflow Step-4 profile directory.

    Expected directory layout (produced by autots.py _run_step4_irc_and_opt):
        <profile_dir>/
            endpoint_1_opt.xyz
            endpoint_2_opt.xyz
            *_ts_final.xyz
            energy_profile.txt     <- primary energy source

    The energy_profile.txt format written by _write_energy_profile_text:
        # header lines starting with #
        TS,          <path>, <final_E_Ha>, <bias_E_Ha>
        Endpoint_1,  <path>, <final_E_Ha>, <bias_E_Ha>
        Endpoint_2,  <path>, <final_E_Ha>, <bias_E_Ha>
    Column index 2 (0-based) is Final_Energy [Hartree].
    """

    def parse(self, profile_dir: str) -> dict | None:
        """
        Parse profile_dir and return a result dict, or None on failure.

        Result keys
        -----------
        ts_xyz_file       : str
        ts_energy         : float | None   [Ha]
        endpoint_1_xyz    : str
        endpoint_2_xyz    : str
        endpoint_1_energy : float | None   [Ha]
        endpoint_2_energy : float | None   [Ha]
        barrier_fwd       : float | None   [kcal/mol]  (endpoint_1 -> TS)
        barrier_rev       : float | None   [kcal/mol]  (endpoint_2 -> TS)
        """
        ep1 = os.path.join(profile_dir, "endpoint_1_opt.xyz")
        ep2 = os.path.join(profile_dir, "endpoint_2_opt.xyz")
        ts_matches = glob.glob(os.path.join(profile_dir, "*_ts_final.xyz"))
        txt_path   = os.path.join(profile_dir, "energy_profile.txt")

        if not os.path.isfile(ep1) or not os.path.isfile(ep2):
            logger.warning("Missing endpoint files in: %s", profile_dir)
            return None
        if not ts_matches:
            logger.warning("No TS file found in: %s", profile_dir)
            return None

        ts_file = ts_matches[0]
        energies = self._parse_energy_txt(txt_path)

        ts_e   = energies.get("TS")
        ep1_e  = energies.get("Endpoint_1")
        ep2_e  = energies.get("Endpoint_2")

        if any(v is None for v in (ts_e, ep1_e, ep2_e)):
            logger.warning(
                "Incomplete energies (TS=%s EP1=%s EP2=%s) in: %s",
                ts_e, ep1_e, ep2_e, profile_dir,
            )

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
        """Extract energies from energy_profile.txt (column index 2)."""
        result: dict[str, float | None] = {
            "TS": None, "Endpoint_1": None, "Endpoint_2": None,
        }
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
    """
    Main controller that autonomously maps a chemical reaction network.

    Workflow
    --------
    1. Register the initial structure as EQ0 and push AFIR perturbation
       tasks into the priority queue.
    2. Pop the highest-priority task and invoke AutoTSWorkflow (v1):
       Steps 1 (AFIR) -> 2 (NEB) -> 3 (TS refine) -> 4 (IRC + opt).
       All outputs for this invocation go into a dedicated run directory:
           output_dir/runs/run_{iter:04d}_node{node_id}/
    3. Parse every Step-4 profile directory inside the run directory.
    4. Compare each endpoint against existing nodes via StructureChecker:
       - Match found  -> use existing node ID.
       - No match     -> register as a new EQNode, copy XYZ to nodes/.
    5. Register the TS as a TSEdge connecting the two endpoint nodes.
    6. For each new EQNode, call ExplorationQueue.should_add and, if
       accepted, push new AFIR perturbation tasks.
    7. Repeat until the queue is empty or max_iterations is reached.
    8. Save reaction_network.json after every iteration (crash-safe).

    Parameters
    ----------
    base_config : dict
        Configuration dict following the run_autots.py / run_mapper.py
        conventions (step1_settings, step2_settings, ... keys).
    queue : ExplorationQueue | None
        Priority queue strategy.  Default: BoltzmannQueue(300 K).
    structure_checker : StructureChecker | None
        Structure identity oracle.  Default: StructureChecker(0.30 A).
    perturbation_generator : PerturbationGenerator | None
        AFIR parameter generator.  Default: PerturbationGenerator().
    output_dir : str
        Root directory for all outputs.  Will be created if absent.
    graph_json : str
        Filename of the network JSON inside output_dir.
    max_iterations : int
        Hard cap on the number of AutoTSWorkflow invocations (0 = none).
    resume : bool
        When True, load an existing graph_json and continue from there.
    """

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
    ) -> None:
        self.base_config = base_config
        self.queue    = queue               if queue               is not None else BoltzmannQueue()
        self.checker  = structure_checker  if structure_checker  is not None else StructureChecker()
        self.perturber = perturbation_generator if perturbation_generator is not None else PerturbationGenerator()
        self.output_dir = os.path.abspath(output_dir)
        self.graph_json_path = os.path.join(self.output_dir, graph_json)
        self.max_iterations = max_iterations
        self.resume = resume

        self.graph = NetworkGraph()
        self._iteration: int = 0

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "nodes"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "runs"), exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the mapping loop."""
        logger.info("=== ReactionNetworkMapper START ===")
        logger.info("  output_dir     : %s", self.output_dir)
        logger.info("  max_iterations : %s",
                    self.max_iterations if self.max_iterations > 0 else "unlimited")
        logger.info("  queue type     : %s", type(self.queue).__name__)

        if self.resume and os.path.isfile(self.graph_json_path):
            logger.info("Resuming from existing graph: %s", self.graph_json_path)
            self.graph.load(self.graph_json_path)
            # Re-enqueue all known nodes so unexplored AFIR variants are retried.
            # The ExplorationQueue._submitted set prevents exact duplicates.
            self._requeue_all_nodes()
        else:
            self._register_seed_structure()

        while True:
            if self.max_iterations > 0 and self._iteration >= self.max_iterations:
                logger.info("Reached max_iterations=%d. Stopping.", self.max_iterations)
                break

            task = self.queue.pop()
            if task is None:
                logger.info("Queue is empty. Stopping.")
                break

            self._iteration += 1
            logger.info(
                "━━━ Iteration %d/%s  (queue remaining: %d) ━━━",
                self._iteration,
                str(self.max_iterations) if self.max_iterations > 0 else "∞",
                len(self.queue),
            )
            logger.info(
                "  Task: EQ%d  priority=%.4f  afir=%s",
                task.node_id, task.priority, task.afir_params,
            )

            run_dir = self._make_run_dir(task)
            try:
                profile_dirs = self._run_autots(task, run_dir)
            except Exception:
                logger.error(
                    "AutoTSWorkflow failed in %s:\n%s",
                    run_dir, traceback.format_exc(),
                )
                self._save_run_metadata(
                    run_dir, task, status="FAILED", profile_dirs=[],
                )
                self.graph.save(self.graph_json_path)
                continue

            for pdir in profile_dirs:
                self._process_profile(pdir, run_dir)

            self._save_run_metadata(
                run_dir, task, status="DONE", profile_dirs=profile_dirs,
            )
            self.graph.save(self.graph_json_path)
            logger.info(self.graph.summary())

        logger.info("=== ReactionNetworkMapper DONE ===")
        self.graph.save(self.graph_json_path)
        logger.info("Final graph:\n%s", self.graph.summary())

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _register_seed_structure(self) -> None:
        """Register the initial XYZ as EQ0 and enqueue its perturbations."""
        seed_xyz = os.path.abspath(self.base_config["initial_mol_file"])
        if not os.path.isfile(seed_xyz):
            raise FileNotFoundError(f"Seed structure not found: {seed_xyz}")

        symbols, coords = parse_xyz(seed_xyz)
        node = EQNode(
            node_id=self.graph.next_node_id(),
            xyz_file=seed_xyz,
            energy=None,          # energy unknown before any calculation
            symbols=symbols,
            coords=coords,
            source_run_dir="initial",
        )
        self.graph.add_node(node)
        logger.info("Registered seed structure as EQ%d: %s", node.node_id, seed_xyz)

        # Force-push: skip Boltzmann acceptance for the seed node.
        self._enqueue_perturbations(node, force_add=True)

    def _requeue_all_nodes(self) -> None:
        """Re-push perturbations for all loaded nodes (resume mode)."""
        for node in self.graph.all_nodes():
            # force_add=True; the queue's _submitted set prevents real duplicates.
            self._enqueue_perturbations(node, force_add=True)

    # ------------------------------------------------------------------
    # Run directory management
    # ------------------------------------------------------------------

    def _make_run_dir(self, task: ExplorationTask) -> str:
        """
        Create and return the dedicated directory for one AutoTSWorkflow run.

        Layout:
            output_dir/runs/run_{iter:04d}_node{node_id}/
                autots_workspace/    <- AutoTSWorkflow working directory
                input.txt            <- mapper configuration snapshot (written here)
                run_info.json        <- written after completion
        """
        name = f"run_{self._iteration:04d}_node{task.node_id}"
        run_dir = os.path.join(self.output_dir, "runs", name)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "autots_workspace"), exist_ok=True)
        self._save_input_txt(run_dir, task)
        return run_dir

    def _save_input_txt(self, run_dir: str, task: ExplorationTask) -> None:
        """
        Write a human-readable configuration snapshot to run_dir/input.txt.

        The file records every parameter that controls this particular
        AutoTSWorkflow invocation so that the run can be reproduced or
        inspected without having to trace back through the code.

        Sections
        --------
        [run]           iteration index and source EQ node
        [afir]          the AFIR parameters sent to Step 1
        [mapper]        queue strategy and structure-checker settings
        [autots_config] the full AutoTSWorkflow config dict (JSON block)
        """
        lines: list[str] = []

        def section(title: str) -> None:
            lines.append(f"[{title}]")

        def kv(key: str, value) -> None:
            lines.append(f"{key} = {value}")

        # ---- [run] ----
        section("run")
        kv("iteration",        self._iteration)
        kv("node_id",          task.node_id)
        kv("priority",         f"{task.priority:.6f}")
        kv("xyz_file",         task.xyz_file)
        kv("run_dir",          run_dir)
        lines.append("")

        # ---- [afir] ----
        section("afir")
        # manual_AFIR format: [gamma, frag1, frag2, ...]
        afir = task.afir_params
        kv("gamma_kJmol", afir[0] if len(afir) > 0 else "N/A")
        kv("fragment_1",  afir[1] if len(afir) > 1 else "N/A")
        kv("fragment_2",  afir[2] if len(afir) > 2 else "N/A")
        if len(afir) > 3:
            kv("extra_afir_args", " ".join(afir[3:]))
        lines.append("")

        # ---- [mapper] ----
        section("mapper")
        kv("queue_type",           type(self.queue).__name__)
        kv("rmsd_threshold_ang",   self.checker.rmsd_threshold)
        kv("afir_gamma_kJmol",     self.perturber.afir_gamma_kJmol)
        kv("max_pairs_per_node",   self.perturber.max_pairs)
        kv("dist_lower_ang",       self.perturber.dist_lower_ang)
        kv("dist_upper_ang",       self.perturber.dist_upper_ang)
        kv("max_iterations",       self.max_iterations)
        kv("output_dir",           self.output_dir)
        # Expose temperature if the queue is BoltzmannQueue
        if isinstance(self.queue, BoltzmannQueue):
            kv("boltzmann_temperature_K", self.queue.temperature_K)
        lines.append("")

        # ---- task metadata ----
        if task.metadata:
            section("task_metadata")
            for k, v in task.metadata.items():
                kv(k, v)
            lines.append("")

        # ---- [autots_config] ----
        section("autots_config")
        lines.append("# Full AutoTSWorkflow configuration (JSON):")

        # Strip the internal _mapper key before serialising
        config_to_dump = {
            k: v for k, v in self.base_config.items() if k != "_mapper"
        }
        lines.append(json.dumps(config_to_dump, indent=2, ensure_ascii=False))

        txt_path = os.path.join(run_dir, "input.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.debug("input.txt written -> %s", txt_path)

    def _save_run_metadata(
        self,
        run_dir: str,
        task: ExplorationTask,
        status: str,
        profile_dirs: list[str],
    ) -> None:
        """Write run_info.json into the run directory for traceability."""
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

    # ------------------------------------------------------------------
    # AutoTSWorkflow invocation
    # ------------------------------------------------------------------

    def _run_autots(self, task: ExplorationTask, run_dir: str) -> list[str]:
        """
        Invoke AutoTSWorkflow (v1) inside run_dir/autots_workspace/ and
        return a sorted list of *_Step4_Profile directory paths found.

        The AutoTSWorkflow is configured to run all four steps:
            Step 1: AFIR scan   (manual_AFIR = task.afir_params)
            Step 2: NEB
            Step 3: TS refine
            Step 4: IRC + endpoint optimisation

        The CWD is saved and restored even when the workflow raises.
        task.xyz_file must be an absolute path.
        """
        if AutoTSWorkflow is None:
            raise RuntimeError("AutoTSWorkflow could not be imported.")

        workspace = os.path.join(run_dir, "autots_workspace")

        config = copy.deepcopy(self.base_config)
        config["initial_mol_file"] = task.xyz_file   # must be absolute
        config["work_dir"]         = workspace
        config.setdefault("step1_settings", {})
        config["step1_settings"]["manual_AFIR"] = task.afir_params
        config["run_step4"]     = True
        config["skip_step1"]    = False
        config["skip_to_step4"] = False

        original_cwd = os.getcwd()
        try:
            # AutoTSWorkflow.setup_workspace() will chdir into workspace.
            # We restore the CWD in the finally block regardless.
            os.chdir(run_dir)
            workflow = AutoTSWorkflow(config=config)
            workflow.run_workflow()
        finally:
            os.chdir(original_cwd)

        # Collect Step-4 profile directories (created inside workspace).
        pattern = os.path.join(workspace, "**", "*_Step4_Profile")
        profiles = sorted(glob.glob(pattern, recursive=True))
        logger.info(
            "  run_%04d: found %d profile director%s",
            self._iteration, len(profiles),
            "y" if len(profiles) == 1 else "ies",
        )
        return profiles

    # ------------------------------------------------------------------
    # Profile processing
    # ------------------------------------------------------------------

    def _process_profile(self, profile_dir: str, run_dir: str) -> None:
        """
        Process one Step-4 profile directory:
        parse -> match/register endpoints -> register TS edge.
        """
        parser = ProfileParser()
        result = parser.parse(profile_dir)
        if result is None:
            logger.warning("Skipping unparseable profile: %s", profile_dir)
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

        if node_id_1 is None or node_id_2 is None:
            logger.warning(
                "Could not resolve both endpoints for profile: %s", profile_dir,
            )
            return

        # Avoid registering a self-loop (both endpoints identical).
        if node_id_1 == node_id_2:
            logger.info(
                "  Both endpoints matched the same node EQ%d; skipping TS edge.",
                node_id_1,
            )
            return

        edge = TSEdge(
            edge_id=self.graph.next_edge_id(),
            node_id_1=node_id_1,
            node_id_2=node_id_2,
            ts_xyz_file=result["ts_xyz_file"],
            ts_energy=ts_energy,
            barrier_fwd=result["barrier_fwd"],
            barrier_rev=result["barrier_rev"],
            source_run_dir=run_dir,
        )
        self.graph.add_edge(edge)
        fwd_str = f"{edge.barrier_fwd:.2f}" if edge.barrier_fwd is not None else "N/A"
        logger.info(
            "  TS edge added: EQ%d -- EQ%d  Ea(fwd)=%s kcal/mol",
            node_id_1, node_id_2, fwd_str,
        )

    def _find_or_register_node(
        self,
        xyz_file: str,
        energy: float | None,
        run_dir: str,
    ) -> int | None:
        """
        Match the structure at xyz_file against all existing nodes.

        Returns the matching node_id on success, or the new node_id after
        registration.  Returns None only when the XYZ file cannot be read.
        """
        try:
            symbols, coords = parse_xyz(xyz_file)
        except Exception as exc:
            logger.warning("Cannot read XYZ %s: %s", xyz_file, exc)
            return None

        # Search existing nodes via Kabsch + Hungarian RMSD.
        for existing in self.graph.all_nodes():
            if existing.coords.size == 0:
                continue
            if self.checker.are_same(symbols, coords, existing.symbols, existing.coords):
                logger.info(
                    "  Matched existing EQ%d  (RMSD < %.3f A)",
                    existing.node_id, self.checker.rmsd_threshold,
                )
                return existing.node_id

        # Register as a new node.
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
        e_str = f"{energy:.8f} Ha" if energy is not None else "unknown"
        logger.info(
            "  New node EQ%d registered: E=%s  [%s]",
            node_id, e_str, saved_xyz,
        )

        # Boltzmann acceptance check before enqueuing further perturbations.
        ref_e = self.graph.reference_energy()
        if ref_e is None or new_node.energy is None:
            # No real reference energy available yet; accept unconditionally.
            self._enqueue_perturbations(new_node, force_add=True)
        else:
            self._enqueue_perturbations(new_node, force_add=False)

        return node_id

    # ------------------------------------------------------------------
    # Perturbation enqueueing
    # ------------------------------------------------------------------

    def _enqueue_perturbations(
        self,
        node: EQNode,
        force_add: bool = False,
    ) -> None:
        """
        Generate AFIR perturbations for node and push accepted tasks.

        When force_add=True the ExplorationQueue.should_add check is
        bypassed (used for the seed node and resume mode).
        The queue's internal _submitted set still prevents exact duplicates.
        """
        if node.coords.size == 0:
            logger.debug("EQ%d: no coordinates; skipping perturbation", node.node_id)
            return

        perturbations = self.perturber.generate_afir_perturbations(
            node.symbols, node.coords,
        )
        if not perturbations:
            logger.debug("EQ%d: no valid AFIR perturbations generated", node.node_id)
            return

        ref_e = self.graph.reference_energy()

        # Acceptance decision (once per node, not per perturbation):
        # if the node itself is accepted, all its perturbation variants are queued.
        if not force_add and node.energy is not None and ref_e is not None:
            accepted = self.queue.should_add(node, ref_e)
        else:
            accepted = True   # seed / no reference energy / force_add

        if not accepted:
            logger.info(
                "  EQ%d: rejected by queue strategy (E=%.8f Ha  ref=%.8f Ha)",
                node.node_id,
                node.energy if node.energy is not None else float("nan"),
                ref_e if ref_e is not None else float("nan"),
            )
            return

        pushed = 0
        for afir_params in perturbations:
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
                },
            )
            if self.queue.push(task):
                pushed += 1

        logger.info(
            "  EQ%d: %d/%d perturbation tasks pushed to queue",
            node.node_id, pushed, len(perturbations),
        )

    # ------------------------------------------------------------------
    # XYZ persistence
    # ------------------------------------------------------------------

    def _persist_node_xyz(self, src_xyz: str, node_id: int) -> str:
        """
        Copy an endpoint XYZ file to output_dir/nodes/EQ{node_id:04d}.xyz
        for permanent storage, and return the absolute destination path.
        """
        dst = os.path.join(self.output_dir, "nodes", f"EQ{node_id:04d}.xyz")
        try:
            shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception as exc:
            logger.warning(
                "Could not copy %s -> %s: %s; using original path.",
                src_xyz, dst, exc,
            )
            return os.path.abspath(src_xyz)