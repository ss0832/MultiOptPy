import numpy as np
import copy
from scipy.signal import argrelextrema


def extremum_list_index(energy_list):
    local_max_energy_list_index = argrelextrema(energy_list, np.greater)
    inverse_energy_list = (-1)*energy_list
    local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)

    local_max_energy_list_index = local_max_energy_list_index[0].tolist()
    local_min_energy_list_index = local_min_energy_list_index[0].tolist()
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    return local_max_energy_list_index, local_min_energy_list_index

class CaluculationQSMv2:
    """
    Implementation of Tangent-based Projection using Ayala & Schlegel (1997) method.
    Replaces the B-matrix logic of QSM with geometric tangent propagation from the TS.
    """
    def __init__(self, APPLY_CI_NEB=99999):
        self.spring_constant_k = 0.01
        self.APPLY_CI_NEB = APPLY_CI_NEB
        self.force_const_for_cineb = 0.01
        self.tau_list = [] # Cache tangents for Hessian projection

    def _normalize(self, v):
        """Helper function to normalize a vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v

    def _calc_arc_tangent(self, q_cur, q_uphill, t_uphill):
        """Ayala & Schlegel Eq. 3c: Arc approximation"""
        chord = q_cur - q_uphill
        diff_norm_sq = np.dot(chord, chord)
        denom = 2 * np.dot(t_uphill, chord)
        
        if abs(denom) < 1e-10:
            return self._normalize(chord)

        r = diff_norm_sq / denom
        t_new = (chord - r * t_uphill) / r
        return self._normalize(t_new)

    def _calc_parabola_tangent(self, q_cur, q_uphill, t_uphill):
        """Ayala & Schlegel Eq. 3d: Parabola approximation"""
        chord = q_cur - q_uphill
        chord_len = np.linalg.norm(chord)
        if chord_len < 1e-10:
            return t_uphill

        cos_theta = np.dot(chord, t_uphill) / chord_len
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        proj = np.dot(chord, t_uphill)
        n_vec = chord - proj * t_uphill
        n_vec = self._normalize(n_vec)
        
        tan_val = np.tan(theta - (np.pi / 4.0))
        t_new = n_vec - tan_val * (t_uphill - n_vec)
        return self._normalize(t_new)

    def _calculate_all_tangents(self, geometry_num_list, energy_list):
        """
        Calculate tangents for all nodes using Ayala & Schlegel logic.
        Propagates from the highest energy point (TS) downwards.
        """
        geoms = np.array(geometry_num_list, dtype=np.float64)
        energies = np.array(energy_list, dtype=np.float64)
        n_images = len(geoms)
        
        # Flatten geometries to treat system as a single vector (3N dim)
        flat_geoms = geoms.reshape(n_images, -1)
        tangents = np.zeros_like(flat_geoms)

        # 1. Find TS (Highest Energy)
        # Exclude fixed endpoints (0 and -1) from being the TS source if possible
        search_energies = energies.copy()
     
        ts_idx = np.argmax(search_energies)
        
        # Safety for edge cases
        if ts_idx == 0: 
            ts_idx = 1
        if ts_idx == n_images - 1: 
            ts_idx = n_images - 2

        # 2. Tangent at TS (Eq. 3a)
        q_ts = flat_geoms[ts_idx]
        q_prev = flat_geoms[ts_idx - 1]
        q_next = flat_geoms[ts_idx + 1]
        
        v_prev = q_prev - q_ts
        v_next = q_next - q_ts
        
        dist_prev_sq = max(np.dot(v_prev, v_prev), 1e-10)
        dist_next_sq = max(np.dot(v_next, v_next), 1e-10)
        
        ts_tangent = (v_next / dist_next_sq) - (v_prev / dist_prev_sq)
        tangents[ts_idx] = self._normalize(ts_tangent)

        # 3. Propagate Downhill (Reactant side: ts-1 -> 1)
        for i in range(ts_idx - 1, 0, -1):
            q_cur = flat_geoms[i]
            q_uphill = flat_geoms[i+1]
            t_uphill = tangents[i+1]
            
            chord = q_cur - q_uphill
            chord_u = self._normalize(chord)
            angle = np.arccos(np.clip(np.dot(chord_u, t_uphill), -1.0, 1.0))
            
            if angle <= (np.pi / 4.0):
                tangents[i] = self._calc_arc_tangent(q_cur, q_uphill, t_uphill)
            else:
                tangents[i] = self._calc_parabola_tangent(q_cur, q_uphill, t_uphill)

        # 4. Propagate Downhill (Product side: ts+1 -> N-2)
        for i in range(ts_idx + 1, n_images - 1):
            q_cur = flat_geoms[i]
            q_uphill = flat_geoms[i-1]
            t_uphill = tangents[i-1]
            
            chord = q_cur - q_uphill
            chord_u = self._normalize(chord)
            angle = np.arccos(np.clip(np.dot(chord_u, t_uphill), -1.0, 1.0))
            
            if angle <= (np.pi / 4.0):
                tangents[i] = self._calc_arc_tangent(q_cur, q_uphill, t_uphill)
            else:
                tangents[i] = self._calc_parabola_tangent(q_cur, q_uphill, t_uphill)

        # Reshape tangents back to (N, Atoms, 3)
        return -1*tangents.reshape(geoms.shape)

    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        print("AyalaMethodV2_AyalaMethodV2_AyalaMethodV2_AyalaMethodV2")
        
        nnode = len(energy_list)
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        
        # Calculate Ayala Tangents for all nodes
        ayala_tangents = self._calculate_all_tangents(geometry_num_list, energy_list)
        
        # Store for calc_proj_hess use
        self.tau_list = ayala_tangents
        
        total_force_list = []

        for i in range(nnode):
            # Fixed Endpoints
            if i == 0 or i == nnode - 1:
                total_force_list.append(-1 * np.array(gradient_list[i], dtype="float64"))
                continue
            
            # Current Gradient and Tangent
            grad = np.array(gradient_list[i], dtype="float64").flatten()
            tangent = ayala_tangents[i].flatten()
            
            # Project Gradient
            # g_parallel = (g . tau) * tau
            g_parallel = np.dot(grad, tangent) * tangent
            g_perp = grad - g_parallel
            
            # Force Calculation
            # Basic NEB Force (Projected Gradient): F = -g_perp
            force = -1.0 * g_perp
            
            # CI-NEB Logic
            # If CI-NEB is active and this node is a local max (TS candidate)
            if optimize_num > self.APPLY_CI_NEB and (i in local_max_energy_list_index) and (i != 1 and i != nnode-2):
                print(f"CI-NEB was applied to # NODE {i} (Ayala Tangent)")
                # Invert the parallel component to climb up: F_ci = -g_perp + g_parallel
                # (Usually F = -Grad_perp - Grad_parallel_spring + ..., here we simulate climbing)
                # Removing spring force along path and adding potential force UP the path.
                # Since standard force here is just -g_perp, adding climbing force means adding component along gradient?
                # Standard CI-NEB: F = -Gradient + 2*(Gradient.Tangent)*Tangent
                # F = - (g_perp + g_para) + 2 * g_para = -g_perp + g_para
                force = -1.0 * g_perp + g_parallel
                
            elif optimize_num > self.APPLY_CI_NEB and (i + 1 in local_max_energy_list_index or i - 1 in local_max_energy_list_index) and (i != 1 and i != nnode-2):
                # Restrict step for neighbors of TS (as in original code)
                print(f"Restrict step of # NODE {i} for CI-NEB")
                force *= 0.001

            total_force_list.append(force.reshape(geometry_num_list[i].shape))

        total_force_list = np.array(total_force_list, dtype="float64")
        
        # Note: Original code applied a global 'projection' here. 
        # Since we explicitly projected using Ayala tangents inside the loop, 
        # applying another Gram-Schmidt based on simple neighbor differences might be redundant or conflicting.
        # However, if the original 'projection' function handles constraints other than the path tangent 
        # (like removing rotation/translation), it might still be needed.
        # Assuming 'projection' in the original snippet was purely for path orthogonality:
        # We DO NOT call the global 'projection' function here to strictly follow the Ayala tangent method.
        
        return total_force_list

    def calc_proj_hess(self, hess, node_num, geometry_num_list):
        """
        Project the Hessian using the cached Ayala tangent.
        P = I - |tau><tau|
        H_proj = P * H * P
        """
        # Ensure tangents are available (calc_force should be called before this)
        if len(self.tau_list) == 0:
            # Fallback or error handling if calc_force wasn't called. 
            # For safety, we return original hess, or one needs to calculate tangents here using stored energies.
            # Assuming usage pattern: calc_force -> opt step -> (optional) calc_hess
            print("Warning: Tangent list empty in calc_proj_hess. Returning original Hessian.")
            return hess
            
        if node_num == 0 or node_num == len(geometry_num_list)-1:
            return hess

        print(f"Applying Ayala tangent projection to Hessian at node {node_num}")
        
        tangent = self.tau_list[node_num].flatten()
        
        # Make Projector P = I - t * t.T
        dim = len(tangent)
        identity = np.eye(dim)
        projector = identity - np.outer(tangent, tangent)
        
        # H_proj = P . H . P
        # Note: hess is likely (3N, 3N)
        tmp_proj_hess = np.dot(np.dot(projector, hess), projector.T)
        
        # Ensure symmetry
        tmp_proj_hess = 0.5 * (tmp_proj_hess + tmp_proj_hess.T)
        
        return tmp_proj_hess

    def get_tau(self, node_num):
        """Returns the flattened tangent vector at the specified node."""
        if len(self.tau_list) == 0:
            raise ValueError("Tangent list is empty. Calculate forces first.")
        return self.tau_list[node_num]

    def calculate_gamma(self, q_triplet, E_triplet, g_triplet, tangent):
        """
        Calculates the curvature gamma along the path using quintic polynomial fitting.
        
        Args:
            q_triplet: List of [q_prev, q_curr, q_next] coordinates
            E_triplet: List of [E_prev, E_curr, E_next] energies
            g_triplet: List of [g_prev, g_curr, g_next] gradients
            tangent: Normalized tangent vector at the current node
            
        Returns:
            gamma: Curvature (2nd derivative) along the path at the current node
        """
        q_prev, q_curr, q_next = q_triplet
        E_prev, E_curr, E_next = E_triplet
        g_prev, g_curr, g_next = g_triplet
        
        # 1. Distances along the path
        dist_prev = np.linalg.norm(q_curr - q_prev)
        dist_next = np.linalg.norm(q_next - q_curr)
        
        if dist_prev < 1e-6 or dist_next < 1e-6:
            return 0.0

        # s coordinates: prev at -dist_prev, curr at 0, next at +dist_next
        s_p = -dist_prev
        s_c = 0.0
        s_n = dist_next
        
        # 2. Project gradients onto path
        # Tangent at i-1: Approximated by direction from i-1 to i
        t_prev = (q_curr - q_prev) / dist_prev
        gp_proj = np.dot(g_prev.flatten(), t_prev.flatten())
        
        # Tangent at i: Given tangent
        gc_proj = np.dot(g_curr.flatten(), tangent.flatten())
        
        # Tangent at i+1: Approximated by direction from i to i+1
        t_next = (q_next - q_curr) / dist_next
        gn_proj = np.dot(g_next.flatten(), t_next.flatten())
        
        # 3. Solve Quintic Polynomial Coefficients
        # E(s) = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5
        A = np.array([
            [1, s_p, s_p**2, s_p**3, s_p**4, s_p**5],
            [1, s_c, s_c**2, s_c**3, s_c**4, s_c**5],
            [1, s_n, s_n**2, s_n**3, s_n**4, s_n**5],
            [0, 1, 2*s_p, 3*s_p**2, 4*s_p**3, 5*s_p**4],
            [0, 1, 2*s_c, 3*s_c**2, 4*s_c**3, 5*s_c**4],
            [0, 1, 2*s_n, 3*s_n**2, 4*s_n**3, 5*s_n**4]
        ])
        
        b = np.array([E_prev, E_curr, E_next, gp_proj, gc_proj, gn_proj])
        
        try:
            coeffs = np.linalg.solve(A, b)
            # Curvature gamma = E''(0) = 2 * c2
            gamma = 2.0 * coeffs[2]
            return gamma
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            return 0.0