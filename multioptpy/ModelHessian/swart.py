import numpy as np
import math

# ============================================================================
# Vectorized Helper Logic
# ============================================================================

# Covalent radii in Bohr (from Pyykkö & Atsumi, Chem. Eur. J. 2009, 15, 186)
COVALENT_RADII = {
    'H': 0.59, 'He': 0.54,
    'Li': 2.43, 'Be': 1.72, 'B': 1.53, 'C': 1.40, 'N': 1.34, 'O': 1.25, 'F': 1.18, 'Ne': 1.14,
    'Na': 2.89, 'Mg': 2.53, 'Al': 2.19, 'Si': 2.10, 'P': 2.04, 'S': 1.97, 'Cl': 1.87, 'Ar': 1.82,
    'K': 3.42, 'Ca': 3.06, 'Sc': 2.85, 'Ti': 2.70, 'V': 2.55, 'Cr': 2.49, 'Mn': 2.49,
    'Fe': 2.44, 'Co': 2.38, 'Ni': 2.32, 'Cu': 2.42, 'Zn': 2.40,
    'Ga': 2.27, 'Ge': 2.19, 'As': 2.17, 'Se': 2.10, 'Br': 2.04, 'Kr': 2.06,
    'Rb': 3.70, 'Sr': 3.40, 'Y': 3.21, 'Zr': 2.98, 'Nb': 2.85, 'Mo': 2.72, 'Tc': 2.61,
    'Ru': 2.55, 'Rh': 2.51, 'Pd': 2.55, 'Ag': 2.68, 'Cd': 2.72,
    'In': 2.61, 'Sn': 2.55, 'Sb': 2.51, 'Te': 2.48, 'I': 2.44, 'Xe': 2.48,
    'Cs': 4.03, 'Ba': 3.59,
    'La': 3.34, 'Ce': 3.25, 'Pr': 3.23, 'Nd': 3.21, 'Pm': 3.19, 'Sm': 3.17, 'Eu': 3.17,
    'Gd': 3.15, 'Tb': 3.13, 'Dy': 3.13, 'Ho': 3.11, 'Er': 3.11, 'Tm': 3.09, 'Yb': 3.09, 'Lu': 3.06,
    'Hf': 2.89, 'Ta': 2.76, 'W': 2.61, 'Re': 2.49, 'Os': 2.46, 'Ir': 2.42, 'Pt': 2.42, 'Au': 2.55, 'Hg': 2.72,
    'Tl': 2.68, 'Pb': 2.68, 'Bi': 2.68, 'Po': 2.61, 'At': 2.57, 'Rn': 2.63,
}

class SwartApproxHessian:
    def __init__(self):
        # Swart's model Hessian parameters
        self.wthr = 0.3
        self.f = 0.12
        self.tolth = 0.2
        
        # Derived parameters
        self.eps1 = self.wthr**2
        self.eps2 = self.wthr**2 / math.exp(1)
        
        self.cart_hess = None

    def _get_radii_array(self, element_list):
        """Convert element list to numpy array of covalent radii."""
        return np.array([COVALENT_RADII.get(e.capitalize(), 0.0) for e in element_list])

    def _precompute_geometry(self, coord, radii):
        """
        Precompute distance matrix, difference vectors, and screening matrix.
        Returns:
            diff_vec: (N, N, 3) array where diff_vec[i, j] = xyz[i] - xyz[j]
            dists: (N, N) distance matrix
            screen: (N, N) screening matrix rho_AB
        """
        # diff_vec[i, j] = coord[i] - coord[j]
        diff_vec = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
        
        # Distance matrix
        dists = np.linalg.norm(diff_vec, axis=2)
        
        # Avoid division by zero on diagonal (though not used, good for safety)
        np.fill_diagonal(dists, 1.0) 
        
        # Cov radii sum matrix: cov_sum[i, j] = r_i + r_j
        cov_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Screening function: exp(1.0 - dist / cov_sum)
        # Note: Diagonal will be exp(1-1/2r) but we ignore diagonal anyway.
        screen = np.exp(1.0 - dists / cov_sum)
        np.fill_diagonal(screen, 0.0)
        
        return diff_vec, dists, screen

    def swart_bond_vectorized(self, n_atoms, diff_vec, dists, screen):
        """Vectorized bond calculation."""
        # Identify valid pairs (upper triangle only to avoid double counting)
        # We process all pairs where i < j.
        # Original logic calculates all pairs. 
        # Force constant is small if screen is small, but original code 
        # calculates strictly for all i, j>i. We replicate this.
        
        rows, cols = np.triu_indices(n_atoms, k=1)
        
        # Extract values for pairs
        r_ij = dists[rows, cols]      # (M,)
        vec_ij = diff_vec[rows, cols] # (M, 3) vec_ij = r_i - r_j
        s_ij = screen[rows, cols]     # (M,)
        
        # Force constant: H_int = 0.35 * screen^3
        H_int = 0.35 * (s_ij ** 3)
        
        # Wilson B matrix components
        # vec_ij is already (r_i - r_j).
        # Normalized vector n = vec / l
        norm_vec = vec_ij / r_ij[:, np.newaxis] # (M, 3)
        
        # Construct full B vector (M, 6)
        # B = [n, -n]
        B = np.hstack([norm_vec, -norm_vec])
        
        # Outer product B * B.T -> (M, 6, 6)
        # einsum is efficient for batch outer product: 'bi,bj->bij'
        BB_T = np.einsum('bi,bj->bij', B, B)
        
        # Contribution: H_int * BB_T
        hess_contrib = H_int[:, np.newaxis, np.newaxis] * BB_T
        
        # Accumulate into global Hessian
        # Since we cannot simply slice assign with duplicate indices if any (here distinct pairs),
        # direct loop over the pre-calculated bonds is efficient enough (M iterations).
        # M = N*(N-1)/2. For N=100, M=5000, which is trivial.
        
        for idx, (i, j) in enumerate(zip(rows, cols)):
            idx_i = slice(3*i, 3*i+3)
            idx_j = slice(3*j, 3*j+3)
            
            # BB_T structure is:
            # [0:3, 0:3] -> ii block
            # [0:3, 3:6] -> ij block
            # [3:6, 0:3] -> ji block
            # [3:6, 3:6] -> jj block
            
            h_local = hess_contrib[idx]
            
            self.cart_hess[idx_i, idx_i] += h_local[0:3, 0:3]
            self.cart_hess[idx_i, idx_j] += h_local[0:3, 3:6]
            self.cart_hess[idx_j, idx_i] += h_local[3:6, 0:3]
            self.cart_hess[idx_j, idx_j] += h_local[3:6, 3:6]

    def _calculate_batch_angle_B(self, vec1, vec2, l1, l2):
        """
        Batch calculation of angle B-matrix.
        vec1: (K, 3), vec2: (K, 3), l1: (K,), l2: (K,)
        Returns B (K, 9) and sin_theta_sq (K,)
        """
        nvec1 = vec1 / l1[:, np.newaxis]
        nvec2 = vec2 / l2[:, np.newaxis]
        
        dot_prod = np.sum(nvec1 * nvec2, axis=1)
        # Clip to avoid numerical errors
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        
        sin_theta_sq = np.maximum(1e-15, 1.0 - dot_prod**2)
        inv_sin = 1.0 / np.sqrt(sin_theta_sq)
        
        # dl terms (derivative of unit vectors)
        # dl is conceptually (K, 2, 6) but we compute needed parts
        
        # Derived formula for B from swart.py logic:
        # dinprod terms.
        # It's cleaner to implement the vector algebra directly for the angle gradient.
        # B_angle vectors are orthogonal to bonds and in the plane.
        
        # n1 x n2 (cross product) -> vector normal to plane
        # (n1 x n2) x n1 -> vector orthogonal to n1 in plane
        # (n2 x n1) x n2 -> vector orthogonal to n2 in plane
        
        # Standard Wilson angle formulas:
        # u = n1, v = n2. theta = angle.
        # B_i = (cos theta * u - v) / (r1 * sin theta)
        # B_k = (cos theta * v - u) / (r2 * sin theta)
        # B_j = - (B_i + B_k)
        
        cos_theta = dot_prod
        
        # Term 1 for atom i (connected by vec1): (cos * n1 - n2)
        term_i = cos_theta[:, np.newaxis] * nvec1 - nvec2
        B_i = term_i / (l1[:, np.newaxis] * np.sqrt(sin_theta_sq)[:, np.newaxis])
        
        # Term 2 for atom k (connected by vec2): (cos * n2 - n1)
        term_k = cos_theta[:, np.newaxis] * nvec2 - nvec1
        B_k = term_k / (l2[:, np.newaxis] * np.sqrt(sin_theta_sq)[:, np.newaxis])
        
        B_j = -(B_i + B_k)
        
        # Flatten to (K, 9) -> [Bi, Bj, Bk]
        B = np.hstack([B_i, B_j, B_k])
        
        return B, cos_theta, sin_theta_sq

    def _calculate_batch_linear_B(self, vec1, vec2, l1, l2):
        """
        Batch calculation of linear angle B-matrix components.
        Returns: B_lin (K, 2, 9)
        """
        nvec1 = vec1 / l1[:, np.newaxis]
        nvec2 = vec2 / l2[:, np.newaxis]
        
        vn = np.cross(vec1, vec2)
        nvn = np.linalg.norm(vn, axis=1)
        
        # Handle small norms (collinear vectors)
        mask_small = nvn < 1e-15
        
        # Prepare fallback vectors
        vn_safe = vn.copy()
        
        # Strategy: if cross product is 0, pick an arbitrary orthogonal vector.
        # Try [1,0,0], if that's collinear, try [0,1,0].
        if np.any(mask_small):
            # Indices where we need fallback
            idx = np.where(mask_small)[0]
            
            # Try ref1 = [1,0,0]
            ref1 = np.array([1.0, 0.0, 0.0])
            # Project out vec1 component: v_orth = ref - (ref.v1)*v1/l1^2
            proj1 = np.sum(ref1 * vec1[idx], axis=1)[:, np.newaxis] / (l1[idx]**2)[:, np.newaxis] * vec1[idx]
            cand1 = ref1 - proj1
            cand1_norm = np.linalg.norm(cand1, axis=1)
            
            # Check if cand1 is valid
            valid1 = cand1_norm >= 1e-15
            
            # Assign valid ones
            vn_safe[idx[valid1]] = cand1[valid1]
            nvn[idx[valid1]] = cand1_norm[valid1]
            
            # If still invalid, try ref2 = [0,1,0]
            still_bad = ~valid1
            if np.any(still_bad):
                idx2 = idx[still_bad]
                ref2 = np.array([0.0, 1.0, 0.0])
                proj2 = np.sum(ref2 * vec1[idx2], axis=1)[:, np.newaxis] / (l1[idx2]**2)[:, np.newaxis] * vec1[idx2]
                cand2 = ref2 - proj2
                cand2_norm = np.linalg.norm(cand2, axis=1)
                
                vn_safe[idx2] = cand2
                nvn[idx2] = cand2_norm
        
        # Normalize normal vector
        vn_norm = vn_safe / nvn[:, np.newaxis] # (K, 3)
        
        # Second orthogonal direction
        # vn2 = cross(vec1 - vec2, vn) / norm
        v_diff = vec1 - vec2
        vn2 = np.cross(v_diff, vn_norm)
        vn2_norm = np.linalg.norm(vn2, axis=1)
        # Avoid div zero if extremely pathological, though unlikely if vn is well defined
        vn2_n = vn2 / np.maximum(vn2_norm[:, np.newaxis], 1e-15)
        
        B = np.zeros((len(vec1), 2, 9))
        
        # Mode 2 (using vn_norm)
        # i component: vn / l1
        B[:, 1, 0:3] = vn_norm / l1[:, np.newaxis]
        # k component: vn / l2
        B[:, 1, 6:9] = vn_norm / l2[:, np.newaxis]
        # j component
        B[:, 1, 3:6] = -B[:, 1, 0:3] - B[:, 1, 6:9]
        
        # Mode 1 (using vn2_n)
        B[:, 0, 0:3] = vn2_n / l1[:, np.newaxis]
        B[:, 0, 6:9] = vn2_n / l2[:, np.newaxis]
        B[:, 0, 3:6] = -B[:, 0, 0:3] - B[:, 0, 6:9]
        
        return B

    def swart_angle_vectorized(self, n_atoms, diff_vec, dists, screen):
        """Vectorized angle calculation centering on atom j."""
        
        # To optimize, we loop over the central atom j.
        # This reduces memory vs full N^3 arrays, but utilizes vectorization for neighbors.
        
        for j in range(n_atoms):
            # Find candidate neighbors for j
            # Condition: screen[i, j] >= eps2
            # Since screen is symmetric, screen[j, :] works
            
            # s_ji = screen[j, :]
            # indices i where s_ji >= self.eps2 and i != j
            # Note: screen diagonal is 0, so i!=j is implicit if eps2 > 0
            
            neighbors = np.where(screen[j, :] >= self.eps2)[0]
            k_neighbors = len(neighbors)
            
            if k_neighbors < 2:
                continue
            
            # Create pairs (i, k) from neighbors
            # We only want unique angles, usually i < k
            # Using meshgrid to generate all pairs
            idx_i_grid, idx_k_grid = np.meshgrid(neighbors, neighbors, indexing='ij')
            
            # Select strict upper triangle to avoid duplicates (i < k) and i == k
            mask_pairs = idx_i_grid < idx_k_grid
            
            idx_i = idx_i_grid[mask_pairs]
            idx_k = idx_k_grid[mask_pairs]
            
            if len(idx_i) == 0:
                continue
                
            # Now we have K pairs of (i, k) centered at j
            # Retrieve data
            # vec1 = r_i - r_j = diff_vec[i, j]
            # vec2 = r_k - r_j = diff_vec[k, j]
            
            vec1 = diff_vec[idx_i, j]
            vec2 = diff_vec[idx_k, j]
            
            l1 = dists[idx_i, j]
            l2 = dists[idx_k, j]
            
            s_ij = screen[idx_i, j]
            s_jk = screen[j, idx_k] # symmetric
            s_ij_jk = s_ij * s_jk
            
            # Filter by combined screening eps1
            mask_screen = s_ij_jk >= self.eps1
            
            if not np.any(mask_screen):
                continue
            
            # Apply screening mask to reduce computation
            vec1 = vec1[mask_screen]
            vec2 = vec2[mask_screen]
            l1 = l1[mask_screen]
            l2 = l2[mask_screen]
            s_ij_jk = s_ij_jk[mask_screen]
            idx_i = idx_i[mask_screen]
            idx_k = idx_k[mask_screen]
            
            # Calculate Angles and B-matrices
            B_norm, cos_theta, sin_theta_sq = self._calculate_batch_angle_B(vec1, vec2, l1, l2)
            sin_theta = np.sqrt(sin_theta_sq)
            
            # Force constant
            # H_int = 0.075 * s_ij_jk^2 * (f + (1-f)*sin(theta))^2
            H_base = 0.075 * (s_ij_jk**2) * ((self.f + (1.0 - self.f) * sin_theta)**2)
            
            # Check linearity logic
            # th1 calculation
            # if cos > 1 - tol: th1 = 1 - cos
            # else: th1 = 1 + cos
            
            # Vectorized th1
            th1 = np.where(cos_theta > (1.0 - self.tolth), 1.0 - cos_theta, 1.0 + cos_theta)
            
            is_near_linear = th1 < self.tolth
            
            # Case 1: Normal angles (not near linear)
            # We can process everything as normal first, then correct the linear ones, 
            # or split indices. Splitting is cleaner for Hessian updates.
            
            mask_linear = is_near_linear
            mask_normal = ~mask_linear
            
            # --- Update Normal Angles ---
            if np.any(mask_normal):
                idx_n = np.where(mask_normal)[0]
                H_n = H_base[idx_n]
                B_n = B_norm[idx_n] # (Kn, 9)
                
                # Outer product (Kn, 9, 9)
                BB_n = np.einsum('ki,kj->kij', B_n, B_n)
                
                # Contribution
                hess_contrib_n = H_n[:, np.newaxis, np.newaxis] * BB_n
                
                # Accumulate
                self._accumulate_angle_hessian(idx_i[idx_n], j, idx_k[idx_n], hess_contrib_n)

            # --- Update Near Linear/Zero Angles ---
            if np.any(mask_linear):
                idx_l = np.where(mask_linear)[0]
                
                # Relevant scalar values
                th1_l = th1[idx_l]
                cos_l = cos_theta[idx_l]
                H_l = H_base[idx_l]
                
                scale_lin = (1.0 - (th1_l / self.tolth)**2)**2
                
                # Compute Linear B matrices
                B_lin_l = self._calculate_batch_linear_B(vec1[idx_l], vec2[idx_l], l1[idx_l], l2[idx_l])
                B_norm_l = B_norm[idx_l]
                
                # Determine if 180 (linear) or 0 (zero)
                # cond: cos > 1 - tolth -> near 180
                cond_180 = cos_l > (1.0 - self.tolth)
                
                # Prepare combined Hessians
                hess_contrib_l = np.zeros((len(idx_l), 9, 9))
                
                # Sub-case: Near 180
                if np.any(cond_180):
                    sub_180 = np.where(cond_180)[0]
                    
                    # B_combined = scale * B_lin + (1-scale) * B_norm
                    # Note: B_lin has 2 components. Swart.py uses B_lin[0] for combination 
                    # Actually swart.py uses B_lin[0] for mixing? 
                    # Checking code: "B_combined = scale_lin * B_lin[0, :] + ..."
                    # Wait, swart.py says:
                    # B_lin = bmat_linear_angle(...) -> returns (2, 9)
                    # B_combined = scale * B_lin[0] + (1-scale) * B  <-- WAIT, B_lin[0] is orthogonal 1
                    # self.cart_hess += H * outer(B_lin[1], B_lin[1]) <--- Adds pure linear mode (orthogonal 2)
                    # self.cart_hess += H * outer(B_combined, B_combined)
                    
                    s_180 = scale_lin[sub_180][:, np.newaxis]
                    B_comb_180 = s_180 * B_lin_l[sub_180, 0, :] + (1.0 - s_180) * B_norm_l[sub_180]
                    B_pure_180 = B_lin_l[sub_180, 1, :]
                    
                    H_180 = H_l[sub_180][:, np.newaxis, np.newaxis]
                    
                    # Add terms
                    hess_contrib_l[sub_180] += H_180 * np.einsum('ki,kj->kij', B_pure_180, B_pure_180)
                    hess_contrib_l[sub_180] += H_180 * np.einsum('ki,kj->kij', B_comb_180, B_comb_180)

                # Sub-case: Near 0
                cond_0 = ~cond_180
                if np.any(cond_0):
                    sub_0 = np.where(cond_0)[0]
                    s_0 = scale_lin[sub_0][:, np.newaxis]
                    
                    # B_scaled = (1 - scale) * B
                    B_scaled_0 = (1.0 - s_0) * B_norm_l[sub_0]
                    H_0 = H_l[sub_0][:, np.newaxis, np.newaxis]
                    
                    hess_contrib_l[sub_0] += H_0 * np.einsum('ki,kj->kij', B_scaled_0, B_scaled_0)

                # Accumulate
                self._accumulate_angle_hessian(idx_i[idx_l], j, idx_k[idx_l], hess_contrib_l)

    def _accumulate_angle_hessian(self, idx_i_arr, j, idx_k_arr, hess_blocks):
        """Helper to add (K, 9, 9) blocks to the global Hessian."""
        # This part is still a loop but over the batch of valid angles for center j
        # K is usually small (valence angles), so direct loop is acceptable.
        
        # Block structure in 9x9 is:
        # 0:3 -> i, 3:6 -> j, 6:9 -> k
        
        # Pre-calculate slices?
        # A simple loop over K is clear and robust.
        
        for idx, (i, k) in enumerate(zip(idx_i_arr, idx_k_arr)):
            block = hess_blocks[idx]
            
            range_i = slice(3*i, 3*i+3)
            range_j = slice(3*j, 3*j+3)
            range_k = slice(3*k, 3*k+3)
            
            # Extract 3x3 subblocks
            ii = block[0:3, 0:3]
            ij = block[0:3, 3:6]
            ik = block[0:3, 6:9]
            
            ji = block[3:6, 0:3]
            jj = block[3:6, 3:6]
            jk = block[3:6, 6:9]
            
            ki = block[6:9, 0:3]
            kj = block[6:9, 3:6]
            kk = block[6:9, 6:9]
            
            # Add to self.cart_hess
            self.cart_hess[range_i, range_i] += ii
            self.cart_hess[range_i, range_j] += ij
            self.cart_hess[range_i, range_k] += ik
            
            self.cart_hess[range_j, range_i] += ji
            self.cart_hess[range_j, range_j] += jj
            self.cart_hess[range_j, range_k] += jk
            
            self.cart_hess[range_k, range_i] += ki
            self.cart_hess[range_k, range_j] += kj
            self.cart_hess[range_k, range_k] += kk

    def swart_dihedral_angle(self, coord, element_list):
        pass
      
    def swart_out_of_plane(self, coord, element_list):
        pass

    def main(self, coord, element_list, cart_gradient=None):
        """
        Main method to calculate the approximate Hessian using Swart's model.
        """
        print("Generating Swart's approximate hessian (O1NumHess variant) ...")
        
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms * 3, n_atoms * 3), dtype="float64")
        
        # 1. Precompute geometry and screening
        radii = self._get_radii_array(element_list)
        diff_vec, dists, screen = self._precompute_geometry(coord, radii)
        
        # 2. Vectorized Bonds
        self.swart_bond_vectorized(n_atoms, diff_vec, dists, screen)
        
        # 3. Vectorized Angles
        self.swart_angle_vectorized(n_atoms, diff_vec, dists, screen)
        
        self.swart_dihedral_angle(coord, element_list)
        self.swart_out_of_plane(coord, element_list)
        
        # External dependency: Project out translation/rotation
        # Assuming Calculationtools is available as in original script
        try:
            from multioptpy.Utils.calc_tools import Calculationtools
            hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        except ImportError:
            # Fallback if library not present for standalone testing
            hess_proj = self.cart_hess

        return hess_proj