"""
mapper.py - Chemical Reaction Network Mapper
============================================

Autonomously maps a chemical reaction network by running
AutoTSWorkflow (autots.py) as the search engine iteratively.
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

try:
    from multioptpy.Parameters.covalent_radii import covalent_radii_lib
    from multioptpy.Parameters.unit_values import UnitValueLib
    _BOHR2ANG = UnitValueLib().bohr2angstroms
except ImportError as _covalent_import_err:
    print(f"[mapper] Warning: could not import covalent radii lib: {_covalent_import_err}")
    covalent_radii_lib = None
    _BOHR2ANG = 0.529177210903

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level physical constants
# ---------------------------------------------------------------------------
HARTREE_TO_KCALMOL: float = 627.509474
K_B_HARTREE: float = 3.166811563e-6


# ===========================================================================
# Section 1 : XYZ utilities
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
# Section 2 : StructureChecker
# ===========================================================================

class StructureChecker:
    def __init__(self, rmsd_threshold: float = 0.30) -> None:
        self.rmsd_threshold = rmsd_threshold

    def are_same(
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
        if len(sym_a) != len(sym_b) or set(sym_a) != set(sym_b):
            return float("inf")

        ca = coords_a - coords_a.mean(axis=0)
        cb = coords_b - coords_b.mean(axis=0)
        cb = self._principal_axis_align(cb)

        perm = self._hungarian_mapping(sym_a, ca, sym_b, cb)
        if perm is None:
            return float("inf")

        return self._kabsch_rmsd(ca, cb[perm])

    @staticmethod
    def _principal_axis_align(coords: np.ndarray) -> np.ndarray:
        if len(coords) < 2:
            return coords
        _, eigvecs = np.linalg.eigh(np.cov(coords.T))
        return coords @ eigvecs

    @staticmethod
    def _hungarian_mapping(
        sym_a: list[str], coords_a: np.ndarray,
        sym_b: list[str], coords_b: np.ndarray,
    ) -> list[int] | None:
        perm: list[int | None] = [None] * len(sym_a)

        for elem in set(sym_a):
            idx_a = [i for i, s in enumerate(sym_a) if s == elem]
            idx_b = [i for i, s in enumerate(sym_b) if s == elem]
            if len(idx_a) != len(idx_b):
                return None

            sub_a = coords_a[idx_a]
            sub_b = coords_b[idx_b]
            diff = sub_a[:, np.newaxis, :] - sub_b[np.newaxis, :, :]
            cost = (diff ** 2).sum(axis=-1)

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                perm[idx_a[r]] = idx_b[c]

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
    def __init__(self) -> None:
        self._tasks: list[ExplorationTask] = []
        self._submitted: set[tuple] = set()

    def push(self, task: ExplorationTask) -> bool:
        key = (task.node_id, tuple(task.afir_params))
        if key in self._submitted:
            return False

        task.priority = self.compute_priority(task)
        self._tasks.append(task)
        self._tasks.sort(key=lambda t: t.priority, reverse=True)
        self._submitted.add(key)
        return True

    def pop(self) -> ExplorationTask | None:
        return self._tasks.pop(0) if self._tasks else None

    def __len__(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def compute_priority(self, task: ExplorationTask) -> float:
        pass

    @abstractmethod
    def should_add(self, node: EQNode, reference_energy: float, **kwargs) -> bool:
        pass


class BoltzmannQueue(ExplorationQueue):
    def __init__(self, temperature_K: float = 300.0, rng_seed: int = 42) -> None:
        super().__init__()
        self.temperature_K = temperature_K
        self._rng = np.random.default_rng(rng_seed)

    def compute_priority(self, task: ExplorationTask) -> float:
        delta_e: float = task.metadata.get("delta_E_hartree", 0.0)
        if delta_e <= 0.0:
            return 1.0
        return min(1.0, float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K))))

    def should_add(self, node: EQNode, reference_energy: float, **kwargs) -> bool:
        delta_e = node.energy - reference_energy
        if delta_e <= 0.0:
            return True
        p = float(np.exp(-delta_e / (K_B_HARTREE * self.temperature_K)))
        return bool(self._rng.random() < p)


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
        covalent_margin: float = 1.2
    ) -> None:
        self.afir_gamma_kJmol = afir_gamma_kJmol
        self.max_pairs = max_pairs
        self.dist_lower_ang = dist_lower_ang
        self.dist_upper_ang = dist_upper_ang
        self.covalent_margin = covalent_margin
        self._rng = np.random.default_rng(rng_seed)

    def generate_afir_perturbations(
        self,
        symbols: list[str],
        coords: np.ndarray,
    ) -> list[list[str]]:
        n = len(symbols)
        if n < 2:
            return []

        dmat = distance_matrix(coords)
        candidates: list[tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
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

        gamma_str = f"{self.afir_gamma_kJmol:.6g}"
        result: list[list[str]] = []
        for idx in chosen:
            i, j = candidates[int(idx)]
            result.append([gamma_str, str(i + 1), str(j + 1)]) 

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

    @property
    def has_real_energy(self) -> bool:
        return self.energy is not None and self.source_run_dir != "initial"

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
    def __init__(self) -> None:
        self._nx: nx.MultiGraph = nx.MultiGraph()
        self._nodes: dict[int, EQNode] = {}
        self._edges: dict[int, TSEdge] = {}
        self._node_counter: int = 0
        self._edge_counter: int = 0

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
                ts_xyz_file=ed["ts_xyz_file"],
                ts_energy=ed["ts_energy_hartree"],
                barrier_fwd=ed.get("barrier_fwd_kcal"),
                barrier_rev=ed.get("barrier_rev_kcal"),
                source_run_dir=ed.get("source_run_dir", ""),
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

    def run(self) -> None:
        if self.resume and os.path.isfile(self.graph_json_path):
            self.graph.load(self.graph_json_path)
            self._requeue_all_nodes()
        else:
            self._register_seed_structure()

        while True:
            if os.path.isfile(os.path.join(self.output_dir, "stop.txt")):
                logger.info("stop.txt detected in output_dir. Stopping.")
                break

            if self.max_iterations > 0 and self._iteration >= self.max_iterations:
                break

            task = self.queue.pop()
            if task is None:
                break

            self._iteration += 1

            run_dir = self._make_run_dir(task)
            try:
                profile_dirs = self._run_autots(task, run_dir)
            except Exception:
                self._save_run_metadata(run_dir, task, status="FAILED", profile_dirs=[])
                self.graph.save(self.graph_json_path)
                continue

            for pdir in profile_dirs:
                self._process_profile(pdir, run_dir)

            self._save_run_metadata(run_dir, task, status="DONE", profile_dirs=profile_dirs)
            self.graph.save(self.graph_json_path)

        self.graph.save(self.graph_json_path)

    def _register_seed_structure(self) -> None:
        seed_xyz = os.path.abspath(self.base_config["initial_mol_file"])
        symbols, coords = parse_xyz(seed_xyz)
        node = EQNode(
            node_id=self.graph.next_node_id(),
            xyz_file=seed_xyz,
            energy=None,
            symbols=symbols,
            coords=coords,
            source_run_dir="initial",
        )
        self.graph.add_node(node)
        self._enqueue_perturbations(node, force_add=True)

    def _requeue_all_nodes(self) -> None:
        for node in self.graph.all_nodes():
            self._enqueue_perturbations(node, force_add=True)

    def _make_run_dir(self, task: ExplorationTask) -> str:
        name = f"run_{self._iteration:04d}_node{task.node_id}"
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
            if self.checker.are_same(symbols, coords, existing.symbols, existing.coords):
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
            self.queue.push(task)

    def _persist_node_xyz(self, src_xyz: str, node_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"EQ{node_id:04d}.xyz")
        try:
            shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception:
            return os.path.abspath(src_xyz)

    def _persist_ts_xyz(self, src_xyz: str, edge_id: int) -> str:
        dst = os.path.join(self.output_dir, "nodes", f"TS{edge_id:04d}.xyz")
        try:
            shutil.copy(src_xyz, dst)
            return os.path.abspath(dst)
        except Exception:
            return os.path.abspath(src_xyz)