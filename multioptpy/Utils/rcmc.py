"""
rcmc.py - Rate Constant Matrix Contraction (RCMC) Queue
=======================================================
Queue implementation applying the RCMC algorithm to dynamically
expanding chemical reaction networks. Determines the exploration priority
of frontier EQ nodes based on transient population dynamics.

ref.: https://doi.org/10.48550/arXiv.2312.05470
      https://doi.org/10.1002/jcc.24526
"""

import logging
import os

import numpy as np
from scipy.linalg import lu_factor, lu_solve
try:
    from multioptpy.Wrapper.mapper import ExplorationQueue, ExplorationTask
except ImportError:
    # ── Minimal stub base classes used when running as __main__ ──────────
    # These are NOT part of RCMCQueue and are never used in normal library
    # operation (multioptpy must be installed for that).
    class ExplorationQueue:  # type: ignore[no-redef]
        """Stub base class – library mode requires the real multioptpy."""
        def __init__(self, rng_seed: int = 42) -> None:
            self._tasks: list = []

        def push(self, task) -> None:
            self._tasks.append(task)

    class ExplorationTask:  # type: ignore[no-redef]
        """Stub task object – mirrors the real ExplorationTask signature."""
        def __init__(
            self,
            node_id: int,
            xyz_file: str = "",
            afir_params: dict | None = None,
            priority: float = 0.0,
            metadata: dict | None = None,
            **kwargs,
        ) -> None:
            self.node_id = node_id
            self.xyz_file = xyz_file
            self.afir_params = afir_params if afir_params is not None else {}
            self.priority = priority
            self.metadata = metadata if metadata is not None else {}

logger = logging.getLogger(__name__)

# Physical constants
K_B_J_K = 1.380649e-23
H_J_S = 6.62607015e-34
K_B_HARTREE = 3.166811563e-6

class RCMCQueue(ExplorationQueue):
    def __init__(
        self,
        temperature_K: float = 300.0,
        reaction_time_s: float = 1.0,
        rng_seed: int = 42,
        start_node_id: int = 0,
        output_dir: str | None = None,
        use_free_energy: bool = False,
    ) -> None:
        super().__init__(rng_seed=rng_seed)
        self.temperature_K = temperature_K
        self.reaction_time_s = reaction_time_s
        self.start_node_id = start_node_id
        self.output_dir = output_dir
        self.graph = None
        self._pop_count: int = 0
        # When True, use Gibbs free energy (G_tot) in place of electronic
        # energy for rate-constant and population calculations.
        # Defaults to False for backward compatibility.
        self.use_free_energy: bool = use_free_energy

    def compute_priority(self, task: ExplorationTask) -> float:
        """
        Returns a placeholder priority. Actual priorities are dynamically
        computed over the entire network within the pop() method.
        """
        return 0.0

    def set_graph(self, graph) -> None:
        """Injects the updated reaction network graph."""
        self.graph = graph

    def should_add(self, node, reference_energy: float, **kwargs) -> bool:
        """Accepts all exploration candidates for dynamic evaluation."""
        return True

    def _save_K_matrix(
        self,
        D: "np.ndarray",
        T_indices: list,
        superstate_members: dict,
        nodes: list,
        pop_count: int,
    ) -> None:
        """Update the single contracted super-state rate matrix CSV.

        The file is overwritten on every pop() call so it always reflects the
        most recent contraction result.  Each super-state row is followed by a
        comment line listing the EQ nodes that belong to it.

        File path: ``{output_dir}/rcmc_K_contracted.csv``

        Format
        ------
        # RCMC contracted super-state rate matrix ...
        # superstate_members: EQ0=[EQ0,EQ3], EQ1=[EQ1,EQ2], ...
        node,EQ0,EQ1,...
        EQ0,<D[0,0]>,<D[0,1]>,...
        EQ1,<D[1,0]>,<D[1,1]>,...

        Parameters
        ----------
        D : np.ndarray
            The contracted (|T| × |T|) effective rate matrix [s⁻¹].
        T_indices : list[int]
            Global indices into ``nodes`` for the rows/columns of D.
        superstate_members : dict[int, list[int]]
            Mapping from T global index → list of global indices of all EQ
            nodes absorbed into that super-state.
        nodes : list[EQNode]
            Full ordered node list.
        pop_count : int
            Current pop() call index (recorded in the header comment).
        """
        if self.output_dir is None:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            fpath = os.path.join(self.output_dir, "rcmc_K_contracted.csv")
            super_labels = [f"EQ{nodes[i].node_id}" for i in T_indices]
            n = len(super_labels)

            # Build member annotation: "EQ0=[EQ0,EQ3]"
            member_parts = []
            for t_global, lbl in zip(T_indices, super_labels):
                members = superstate_members.get(t_global, [t_global])
                member_str = "+".join(f"EQ{nodes[m].node_id}" for m in members)
                member_parts.append(f"{lbl}=[{member_str}]")

            with open(fpath, "w", encoding="utf-8") as fh:
                fh.write(
                    f"# RCMC contracted super-state rate matrix D [s^-1]  "
                    f"pop_step={pop_count}  "
                    f"T={self.temperature_K} K  "
                    f"n_superstates={n}\n"
                )
                fh.write(
                    f"# superstate_members: {',  '.join(member_parts)}\n"
                )
                fh.write("node," + ",".join(super_labels) + "\n")
                for i, lbl in enumerate(super_labels):
                    row = ",".join(f"{D[i, j]:.6e}" for j in range(n))
                    fh.write(f"{lbl},{row}\n")
            logger.info(
                "RCMC contracted K matrix updated: %s  "
                "(pop_step=%d  superstates=%s)",
                fpath,
                pop_count,
                super_labels,
            )
        except OSError as exc:
            logger.warning("RCMC contracted K matrix could not be saved: %s", exc)

    def _save_population(
        self,
        q: "np.ndarray",
        nodes: list,
        pop_count: int,
    ) -> None:
        """Append the per-node transient population distribution to the
        contracted K-matrix CSV file.

        Called immediately after :meth:`_save_K_matrix` so the two tables
        appear in the same file, separated by a blank line.

        File path: ``{output_dir}/rcmc_K_contracted.csv``

        Format (appended below the K-matrix block)
        -------------------------------------------
        # RCMC transient population  pop_step=N  T=300.0 K  t=1.0 s
        node,population
        EQ0,4.56000000e-01
        EQ1,3.21000000e-01
        ...
        """
        if self.output_dir is None:
            return
        try:
            fpath = os.path.join(self.output_dir, "rcmc_K_contracted.csv")
            with open(fpath, "a", encoding="utf-8") as fh:
                fh.write("\n")
                fh.write(
                    f"# RCMC transient population  "
                    f"pop_step={pop_count}  "
                    f"T={self.temperature_K} K  "
                    f"t={self.reaction_time_s} s\n"
                )
                fh.write("node,population\n")
                for i, node in enumerate(nodes):
                    fh.write(f"EQ{node.node_id},{q[i]:.8e}\n")
            logger.info(
                "RCMC population distribution appended: %s  "
                "(pop_step=%d  n_nodes=%d)",
                fpath,
                pop_count,
                len(nodes),
            )
        except OSError as exc:
            logger.warning("RCMC population CSV could not be saved: %s", exc)

    def pop(self) -> ExplorationTask | None:
        if not self._tasks:
            return None

        if self.graph is None or len(self.graph.all_edges()) == 0:
            # No graph yet — return the task with the lowest raw energy.
            selected = min(
                self._tasks.values(),
                key=lambda t: t.metadata.get("delta_E_hartree", 0.0),
            )
            target_key = next(k for k, v in self._tasks.items() if v is selected)
            del self._tasks[target_key]
            return selected

        # ── Select nodes and energy accessor based on energy mode ───────────
        # In free-energy mode only nodes that carry a G_tot value are included;
        # nodes with free_energy=None are silently excluded (they were already
        # filtered/warned at the CLI level before pop() is called).
        if self.use_free_energy:
            nodes = [
                n for n in self.graph.all_nodes()
                if getattr(n, "free_energy", None) is not None
            ]
            def _node_e(n):
                return n.free_energy
            def _edge_ts_e(edge):
                return getattr(edge, "ts_free_energy", None)
        else:
            nodes = [n for n in self.graph.all_nodes() if n.has_real_energy]
            def _node_e(n):
                return n.energy
            def _edge_ts_e(edge):
                return edge.ts_energy

        if not nodes:
            if not self._tasks:
                return None
            selected = next(iter(self._tasks.values()))
            target_key = next(k for k, v in self._tasks.items() if v is selected)
            del self._tasks[target_key]
            return selected

        n_nodes = len(nodes)
        node_to_idx = {n.node_id: i for i, n in enumerate(nodes)}

        K = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        kB_T_h = (K_B_J_K * self.temperature_K) / H_J_S
        RT_ha = K_B_HARTREE * self.temperature_K

        for edge in self.graph.all_edges():
            ts_e = _edge_ts_e(edge)
            if ts_e is None:
                continue
            if edge.node_id_1 not in node_to_idx or edge.node_id_2 not in node_to_idx:
                continue

            u = node_to_idx[edge.node_id_1]
            v = node_to_idx[edge.node_id_2]
            E_u  = _node_e(nodes[u])
            E_v  = _node_e(nodes[v])
            E_TS = ts_e

            E_TS_u = max(E_TS, E_u)
            E_TS_v = max(E_TS, E_v)

            k_uv = kB_T_h * np.exp(-(E_TS_u - E_u) / RT_ha)
            k_vu = kB_T_h * np.exp(-(E_TS_v - E_v) / RT_ha)

            K[v, u] += k_uv
            K[u, v] += k_vu

        for i in range(n_nodes):
            K[i, i] = -np.sum(K[:, i]) + K[i, i]

        # ── Log rate matrix K ─────────────────────────────────────────────
        if logger.isEnabledFor(logging.DEBUG):
            node_labels = [f"EQ{n.node_id}" for n in nodes]
            header = "  " + "  ".join(f"{lbl:>10s}" for lbl in node_labels)
            rows = [header]
            for i, lbl in enumerate(node_labels):
                row_vals = "  ".join(f"{K[i, j]:10.3e}" for j in range(n_nodes))
                rows.append(f"{lbl:>10s}  {row_vals}")
            logger.debug("RCMC rate matrix K [s^-1]:\n%s", "\n".join(rows))

        p = np.zeros(n_nodes, dtype=np.float64)
        if self.start_node_id in node_to_idx:
            p[node_to_idx[self.start_node_id]] = 1.0
        else:
            start_idx = np.argmin([_node_e(n) for n in nodes])
            p[start_idx] = 1.0

        S = []
        T = list(range(n_nodes))
        D = K.copy()
        time_current = 0.0

        # superstate_members[global_idx] = list of global indices absorbed
        # into the super-state whose representative is global_idx.
        # Initially every node is its own super-state.
        superstate_members: dict[int, list[int]] = {i: [i] for i in range(n_nodes)}

        # ── Incremental K_SS buffer ───────────────────────────────────────
        # Maintained by block-appending each newly contracted node so we
        # avoid rebuilding via fancy indexing (K[np.ix_(S,S)]) every step.
        K_SS_buf: np.ndarray = np.empty((0, 0), dtype=np.float64)

        while len(T) > 1:
            # Only the diagonal is needed for argmax — extract with np.diag
            # rather than keeping a full abs matrix.
            j_local = int(np.argmax(np.abs(np.diag(D))))
            j_global = T[j_local]
            D_jj = D[j_local, j_local]

            if abs(D_jj) < 1e-30:
                break

            # Boolean mask is faster than np.arange comparison for slicing.
            mask = np.ones(len(T), dtype=bool)
            mask[j_local] = False

            D_Tj = D[mask, j_local]          # shape (n-1,)
            D_jT = D[j_local, mask]          # shape (n-1,)

            # Schur-complement rank-1 update.
            D_new = D[np.ix_(mask, mask)] - np.outer(D_Tj, D_jT) / D_jj

            # Vectorised diagonal correction (replaces Python for-loop):
            # enforce column-sum-to-zero so numerical drift does not accumulate.
            off_diag_col_sums = D_new.sum(axis=0) - D_new.diagonal()
            np.fill_diagonal(D_new, -off_diag_col_sums)

            # Assign j to the T state most strongly coupled to it,
            # then merge j's member list into that state's members.
            # D_Tj was already computed above — reuse it.
            coupling = np.abs(D_Tj)
            absorb_local = int(np.argmax(coupling)) if coupling.max() > 0 else 0
            remaining_T = [t for k, t in enumerate(T) if k != j_local]
            absorb_global = remaining_T[absorb_local]
            superstate_members[absorb_global].extend(
                superstate_members.pop(j_global)
            )

            # ── Incremental K_SS expansion ────────────────────────────────
            # Append j_global as a new row/column instead of re-indexing K.
            if K_SS_buf.size == 0:
                K_SS_buf = np.array([[K[j_global, j_global]]])
            else:
                new_col = K[S, j_global].reshape(-1, 1)
                new_row = K[j_global, S].reshape(1, -1)
                K_SS_buf = np.block([
                    [K_SS_buf,                          new_col             ],
                    [new_row,  np.array([[K[j_global, j_global]]])]
                ])

            S.append(j_global)
            T.pop(j_local)
            D = D_new

            # ── Convergence check using a single LU factorisation ─────────
            # lu_solve(…, trans=1) solves the transposed system without a
            # second factorisation, replacing two separate np.linalg.solve
            # calls on -K_SS and -K_SS.T.
            try:
                lu, piv = lu_factor(-K_SS_buf)
                ones_S = np.ones(len(S))
                inv_1S   = lu_solve((lu, piv), ones_S)
                inv_T_1S = lu_solve((lu, piv), ones_S, trans=1)
                rho_KSS_inv = min(float(np.max(inv_1S)), float(np.max(inv_T_1S)))
                sigma_KSS = 1.0 / rho_KSS_inv if rho_KSS_inv > 0 else 1e-30
            except Exception:
                sigma_KSS = 1e-30

            # Compute abs(D) once and reuse for both axis sums.
            abs_D = np.abs(D)
            rho_D = min(
                float(np.max(abs_D.sum(axis=1))),
                float(np.max(abs_D.sum(axis=0))),
            )

            if sigma_KSS > 0 and rho_D > 0:
                time_current = np.log(2) / np.sqrt(sigma_KSS * rho_D)

            logger.debug(
                "RCMC contraction step %d: contracted_nodes=%s, "
                "remaining=%s, time_current=%.4e s (target=%.4e s)",
                len(S),
                [nodes[idx].node_id for idx in S],
                [nodes[idx].node_id for idx in T],
                time_current,
                self.reaction_time_s,
            )

            if time_current > self.reaction_time_s:
                break

        # ── Update contracted super-state K matrix (D at termination) ────
        self._save_K_matrix(D, T, superstate_members, nodes, self._pop_count)

        q = np.zeros(n_nodes, dtype=np.float64)
        if len(S) > 0 and len(T) > 0:
            # K_SS_buf is already up to date — no need to re-index K.
            K_ST = K[np.ix_(S, T)]
            K_TS = K[np.ix_(T, S)]
            p_S = p[S]
            p_T = p[T]

            try:
                # Factorise K_SS once; reuse for the three back-solves.
                lu, piv = lu_factor(K_SS_buf)
                X_ST   = lu_solve((lu, piv), K_ST)
                X_pS   = lu_solve((lu, piv), p_S)
                X_ST_2 = lu_solve((lu, piv), X_ST)

                M = np.eye(len(T)) + K_TS @ X_ST_2
                m_vec = np.sum(M, axis=0)
                V_TT_diag = 1.0 / np.where(np.abs(m_vec) > 1e-16, m_vec, 1e-16)

                q_T = V_TT_diag * (p_T - K_TS @ X_pS)
                q_S = - X_ST @ q_T

                # Guarantee non-negativity and normalize
                q_T = np.maximum(q_T, 0.0)
                q_S = np.maximum(q_S, 0.0)
                q[T] = q_T
                q[S] = q_S
                total_q = float(np.sum(q))
                if total_q > 0.0:
                    q /= total_q

            except Exception:
                q = p
        else:
            q = p

        # ── Append population distribution below the K-matrix CSV ────────
        self._save_population(q, nodes, self._pop_count)
        self._pop_count += 1

        for task in self._tasks.values():
            if task.node_id in node_to_idx:
                idx = node_to_idx[task.node_id]
                task.priority = q[idx]
            else:
                task.priority = 0.0

        selected = max(self._tasks.values(), key=lambda t: t.priority)
        target_key = next(k for k, v in self._tasks.items() if v is selected)
        del self._tasks[target_key]
        logger.debug(
            "RCMC pop(): selected EQ%d  priority(q)=%.6f  "
            "remaining_tasks=%d",
            selected.node_id,
            selected.priority,
            len(self._tasks),
        )
        if logger.isEnabledFor(logging.DEBUG):
            # self._tasks is dict[int, ExplorationTask] — must convert to list
            # before slicing; dict[:N] raises TypeError.
            remaining = sorted(
                self._tasks.values(), key=lambda t: t.priority, reverse=True
            )
            pop_lines = [
                f"  EQ{t.node_id}: q={t.priority:.6f}"
                for t in remaining[:10]
            ]
            if len(self._tasks) > 10:
                pop_lines.append(f"  ... ({len(self._tasks) - 10} more)")
            logger.debug("RCMC remaining queue (top 10):\n%s", "\n".join(pop_lines))
        return selected

# ══════════════════════════════════════════════════════════════════════════════
# Standalone CLI entry point
# Usage:
#   python rcmc.py reaction_network.json --temperature 500 --time 1e-6
#   python rcmc.py reaction_network.json -T 300 -t 1.0 --start-node 0 --output-dir ./out
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    """Standalone CLI entry point for RCMC analysis.

    Can be called directly (``python rcmc.py …``) or via the installed
    console script ``run_rcmc`` registered in pyproject.toml.

    Parameters
    ----------
    argv:
        Argument list. ``None`` means ``sys.argv[1:]`` (normal CLI behaviour).
    """
    import argparse
    import json
    import sys
    from dataclasses import dataclass

    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="rcmc.py",
        description=(
            "Standalone RCMC analysis.\n"
            "Loads a reaction-network JSON, builds the rate-constant matrix,\n"
            "applies Type-A contraction up to the specified time scale, and\n"
            "reports the transient population distribution for each EQ node."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "json_file",
        help="Path to reaction_network.json produced by MultiOptPy.",
    )
    parser.add_argument(
        "-T", "--temperature",
        type=float,
        default=300.0,
        metavar="K",
        help="Temperature in Kelvin (default: 300.0).",
    )
    parser.add_argument(
        "-t", "--time",
        type=float,
        default=1.0,
        metavar="S",
        help="Reaction time scale in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--start-node",
        type=int,
        default=0,
        metavar="ID",
        help="node_id of the initial EQ node (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write rcmc_K_contracted.csv "
            "(default: none – results printed to stdout only)."
        ),
    )
    parser.add_argument(
        "--energy-type",
        type=str,
        default="electronic",
        choices=["electronic", "free"],
        metavar="TYPE",
        help=(
            "Energy type used for rate-constant calculations. "
"            'electronic' (default) uses the SCF energy (energy_hartree). "
"            'free' uses the Gibbs free energy (free_energy_hartree). "
"            In free mode, nodes whose free_energy_hartree is null are "
"            removed along with all edges connected to them before RCMC."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    args = parser.parse_args(argv)

    # ── Logging setup ─────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # ── Lightweight graph stubs ───────────────────────────────────────────────
    @dataclass
    class _Node:
        node_id: int
        energy: float | None        # Hartree (electronic)
        xyz_file: str = ""
        has_real_energy: bool = True
        free_energy: float | None = None   # Hartree (Gibbs, None if unavailable)

    @dataclass
    class _Edge:
        node_id_1: int
        node_id_2: int
        ts_energy: float | None          # Hartree (electronic TS energy)
        ts_free_energy: float | None = None  # Hartree (free-energy TS)

    class _Graph:
        def __init__(self, nodes: list, edges: list) -> None:
            self._nodes = nodes
            self._edges = edges

        def all_nodes(self) -> list:
            return self._nodes

        def all_edges(self) -> list:
            return self._edges

    # ── Load JSON ─────────────────────────────────────────────────────────────
    try:
        with open(args.json_file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: Cannot read '{args.json_file}': {exc}")

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    if not raw_nodes:
        sys.exit("ERROR: No nodes found in the JSON file.")

    use_free_energy: bool = (args.energy_type == "free")

    nodes = [
        _Node(
            node_id=nd["node_id"],
            energy=nd["energy_hartree"],
            xyz_file=nd.get("xyz_file", ""),
            has_real_energy=nd["energy_hartree"] is not None,
            free_energy=nd.get("free_energy_hartree"),
        )
        for nd in raw_nodes
    ]

    # ── Free-energy mode: remove nodes with null G_tot and their edges ────
    if use_free_energy:
        null_ids = {nd.node_id for nd in nodes if nd.free_energy is None}
        if null_ids:
            print(
                f"WARNING: The following nodes have no free energy and will be "
                f"excluded from the RCMC calculation:\n"
                + "\n".join(f"  EQ{nid}" for nid in sorted(null_ids))
            )
            nodes = [nd for nd in nodes if nd.node_id not in null_ids]
            if not nodes:
                sys.exit(
                    "ERROR: No nodes with free_energy_hartree remain after filtering. "
                    "Run without --energy-type free or recompute with vibrational analysis."
                )
            # Check start node before filtering edges
            if args.start_node in null_ids:
                sys.exit(
                    f"ERROR: --start-node EQ{args.start_node} has no free energy "
                    f"and was excluded. Choose a different start node or use "
                    f"--energy-type electronic."
                )

    edges = []
    for ed in raw_edges:
        # In free-energy mode use ts_free_energy; fall back to ts_energy otherwise.
        if use_free_energy:
            ts_e      = None   # not used for K-matrix in free mode
            ts_free_e = ed.get("ts_free_energy_hartree")
            if ts_free_e is None:
                continue       # skip edges without a free TS energy
        else:
            ts_e      = ed.get("ts_energy_hartree")
            ts_free_e = ed.get("ts_free_energy_hartree")
            if ts_e is None:
                continue       # skip edges without a TS energy

        if ed["node_id_1"] == ed["node_id_2"]:
            continue           # skip self-loops

        # In free-energy mode skip edges connected to excluded (null G) nodes
        if use_free_energy and (
            ed["node_id_1"] in null_ids or ed["node_id_2"] in null_ids
        ):
            continue

        edges.append(
            _Edge(
                node_id_1=ed["node_id_1"],
                node_id_2=ed["node_id_2"],
                ts_energy=ts_e,
                ts_free_energy=ts_free_e,
            )
        )

    graph = _Graph(nodes, edges)
    node_ids = {nd.node_id for nd in nodes}

    # ── Build RCMCQueue and populate with stub tasks ──────────────────────────
    queue = RCMCQueue(
        temperature_K=args.temperature,
        reaction_time_s=args.time,
        start_node_id=args.start_node,
        output_dir=args.output_dir,
        use_free_energy=use_free_energy,
    )
    queue.set_graph(graph)

    # Add every EQ node as a candidate task so pop() can rank them all.
    for nd in nodes:
        eff_e = nd.free_energy if (use_free_energy and nd.free_energy is not None) else nd.energy
        task = ExplorationTask(
            node_id=nd.node_id,
            xyz_file=nd.xyz_file if hasattr(nd, "xyz_file") else "",
            afir_params={},
            priority=0.0,
            metadata={"delta_E_hartree": eff_e},
        )
        queue.push(task)

    # ── Run RCMC – one full pop() pass ────────────────────────────────────────
    energy_label = "Gibbs free energy (G_tot)" if use_free_energy else "Electronic energy (SCF)"
    print(
        f"\n{'═'*60}\n"
        f"  RCMC standalone analysis\n"
        f"  JSON      : {args.json_file}\n"
        f"  Nodes     : {len(nodes)}  |  Edges: {len(edges)}\n"
        f"  T         : {args.temperature} K\n"
        f"  Time scale: {args.time:.3e} s\n"
        f"  Start node: EQ{args.start_node}\n"
        f"  Energy    : {energy_label}\n"
        f"{'═'*60}"
    )

    top_task = queue.pop()

    # ── Print ranked population distribution ─────────────────────────────────
    # After pop() the remaining tasks carry updated priorities (populations).
    # Re-collect all tasks including the one that was popped.
    all_tasks = list(queue._tasks.values())
    if top_task is not None:
        all_tasks.insert(0, top_task)   # re-insert at front (highest priority)

    all_tasks.sort(key=lambda t: t.priority, reverse=True)

    col_w = max(len(f"EQ{t.node_id}") for t in all_tasks) + 2
    print(f"\n{'Node':<{col_w}}  {'Population':>18}  {'Rank':>6}")
    print(f"{'─'*col_w}  {'─'*18}  {'─'*6}")
    for rank, task in enumerate(all_tasks, start=1):
        marker = "  ◀ highest priority" if rank == 1 else ""
        print(
            f"EQ{task.node_id:<{col_w-2}}  {task.priority:18.8e}  {rank:6d}{marker}"
        )

    if top_task is not None:
        print(
            f"\nRCMC recommends exploring: EQ{top_task.node_id}  "
            f"(population = {top_task.priority:.6e})"
        )

    if args.output_dir:
        out_file = os.path.join(args.output_dir, "rcmc_K_contracted.csv")
        print(f"\nContracted K-matrix + populations saved to: {out_file}")

    print()


if __name__ == "__main__":
    main()