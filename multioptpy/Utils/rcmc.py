"""
rcmc.py - Rate Constant Matrix Contraction (RCMC) Queue
=======================================================
Queue implementation applying the RCMC algorithm (Type A) to dynamically
expanding chemical reaction networks. Determines the exploration priority
of frontier EQ nodes based on transient population dynamics.

ref.: https://doi.org/10.48550/arXiv.2312.05470
      https://doi.org/10.1002/jcc.24526
"""

import numpy as np
from multioptpy.Wrapper.mapper import ExplorationQueue, ExplorationTask

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
        start_node_id: int = 0
    ) -> None:
        super().__init__(rng_seed=rng_seed)
        self.temperature_K = temperature_K
        self.reaction_time_s = reaction_time_s
        self.start_node_id = start_node_id
        self.graph = None

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

    def pop(self) -> ExplorationTask | None:
        if not self._tasks:
            return None

        if self.graph is None or len(self.graph.all_edges()) == 0:
            self._tasks.sort(key=lambda t: t.metadata.get("delta_E_hartree", 0.0))
            return self._tasks.pop(0)

        nodes = [n for n in self.graph.all_nodes() if n.has_real_energy]
        if not nodes:
            return self._tasks.pop(0)

        n_nodes = len(nodes)
        node_to_idx = {n.node_id: i for i, n in enumerate(nodes)}

        K = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        kB_T_h = (K_B_J_K * self.temperature_K) / H_J_S
        RT_ha = K_B_HARTREE * self.temperature_K

        for edge in self.graph.all_edges():
            if edge.ts_energy is None:
                continue
            if edge.node_id_1 not in node_to_idx or edge.node_id_2 not in node_to_idx:
                continue
            
            u = node_to_idx[edge.node_id_1]
            v = node_to_idx[edge.node_id_2]
            E_u = nodes[u].energy
            E_v = nodes[v].energy
            E_TS = edge.ts_energy

            E_TS_u = max(E_TS, E_u)
            E_TS_v = max(E_TS, E_v)

            k_uv = kB_T_h * np.exp(-(E_TS_u - E_u) / RT_ha)
            k_vu = kB_T_h * np.exp(-(E_TS_v - E_v) / RT_ha)

            K[v, u] += k_uv
            K[u, v] += k_vu

        for i in range(n_nodes):
            K[i, i] = -np.sum(K[:, i]) + K[i, i]

        p = np.zeros(n_nodes, dtype=np.float64)
        if self.start_node_id in node_to_idx:
            p[node_to_idx[self.start_node_id]] = 1.0
        else:
            start_idx = np.argmin([n.energy for n in nodes])
            p[start_idx] = 1.0

        S = []
        T = list(range(n_nodes))
        D = K.copy()
        time_current = 0.0

        while len(T) > 1:
            diag_D = np.abs(np.diag(D))
            j_local = np.argmax(diag_D)
            j_global = T[j_local]
            D_jj = D[j_local, j_local]

            if abs(D_jj) < 1e-30:
                break

            mask = np.arange(len(T)) != j_local
            D_TT = D[np.ix_(mask, mask)]
            D_Tj = D[mask, j_local].reshape(-1, 1)
            D_jT = D[j_local, mask].reshape(1, -1)
            
            D_new = D_TT - (D_Tj @ D_jT) / D_jj
            
            # Recalculate diagonal elements to preserve numerical stability
            for i in range(D_new.shape[0]):
                D_new[i, i] = -np.sum(D_new[:, i]) + D_new[i, i]

            S.append(j_global)
            T.pop(j_local)
            D = D_new

            if len(S) > 0:
                K_SS = K[np.ix_(S, S)]
                try:
                    inv_1S = np.linalg.solve(-K_SS, np.ones(len(S)))
                    inv_T_1S = np.linalg.solve(-K_SS.T, np.ones(len(S)))
                    rho_KSS_inv = min(np.max(inv_1S), np.max(inv_T_1S))
                    sigma_KSS = 1.0 / rho_KSS_inv if rho_KSS_inv > 0 else 1e-30
                except np.linalg.LinAlgError:
                    sigma_KSS = 1e-30

                rho_D = min(np.max(np.sum(np.abs(D), axis=1)), np.max(np.sum(np.abs(D), axis=0)))

                if sigma_KSS > 0 and rho_D > 0:
                    time_current = np.log(2) / np.sqrt(sigma_KSS * rho_D)

            if time_current > self.reaction_time_s:
                break

        q = np.zeros(n_nodes, dtype=np.float64)
        if len(S) > 0 and len(T) > 0:
            K_SS = K[np.ix_(S, S)]
            K_ST = K[np.ix_(S, T)]
            K_TS = K[np.ix_(T, S)]
            p_S = p[S]
            p_T = p[T]

            try:
                X_ST = np.linalg.solve(K_SS, K_ST)
                X_pS = np.linalg.solve(K_SS, p_S)
                X_ST_2 = np.linalg.solve(K_SS, X_ST)

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
                total_q = np.sum(q)
                if total_q > 0.0:
                    q /= total_q

            except np.linalg.LinAlgError:
                q = p
        else:
            q = p

        for task in self._tasks:
            if task.node_id in node_to_idx:
                idx = node_to_idx[task.node_id]
                task.priority = q[idx]
            else:
                task.priority = 0.0

        self._tasks.sort(key=lambda t: t.priority, reverse=True)
        return self._tasks.pop(0)