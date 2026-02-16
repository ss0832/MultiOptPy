import numpy as np
import math

from multioptpy.Utils.calc_tools import Calculationtools
# ============================================================================
# Vectorized Helper Logic with Singularity Handling & Fallback
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
    """
    Implements Swart's Approximate Hessian Model (O1NumHess variant).
    
    This model constructs a Hessian matrix based on empirical force constants derived 
    from bond lengths and angles, screened by covalent radii.

    References:
        1. Swart, M. "Addendum to 'A new family of gradient-corrected exchange-correlation functionals'", 
           Chem. Phys. Lett. 2009, 480, 295–297.
        2. Swart, M.; Bickelhaupt, F. M. "Optimization of strong and weak coordinates", 
           Int. J. Quantum Chem. 2006, 106, 2536–2544.
        3. Pyykkö, P.; Atsumi, M. "Molecular Single-Bond Covalent Radii for Elements 1-118", 
           Chem. Eur. J. 2009, 15, 186-197. (Source of Covalent Radii)
    """

    def __init__(self):
        # Swart's model Hessian parameters
        self.wthr = 0.3
        self.f = 0.12
        self.tolth = 0.2
        
        # Derived parameters
        self.eps1 = self.wthr**2
        self.eps2 = self.wthr**2 / math.exp(1)
        
        # Numerical stability constants
        self.MIN_DIST = 1e-8       # Avoid division by zero for overlapping atoms
        self.MIN_SIN_SQ = 1e-12    # Avoid division by zero for linear angles
        self.MIN_NORM = 1e-12      # Avoid division by zero in vector normalization
        
        self.cart_hess = None

    def _get_radii_array(self, element_list):
        """Convert element list to numpy array of covalent radii."""
        return np.array([COVALENT_RADII.get(e.capitalize(), 1.0) for e in element_list])

    def _precompute_geometry(self, coord, radii):
        """
        Precompute distance matrix, difference vectors, and screening matrix.
        """
        diff_vec = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
        dists = np.linalg.norm(diff_vec, axis=2)
        
        # Safety: Avoid zero distance
        dists = np.maximum(dists, self.MIN_DIST)
        np.fill_diagonal(dists, 1.0)
        
        cov_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        cov_sum = np.maximum(cov_sum, self.MIN_DIST)
        
        screen = np.exp(1.0 - dists / cov_sum)
        np.fill_diagonal(screen, 0.0)
        
        return diff_vec, dists, screen

    def swart_bond_vectorized(self, n_atoms, diff_vec, dists, screen):
        """Vectorized bond calculation."""
        rows, cols = np.triu_indices(n_atoms, k=1)
        
        r_ij = dists[rows, cols]
        vec_ij = diff_vec[rows, cols]
        s_ij = screen[rows, cols]
        
        H_int = 0.35 * (s_ij ** 3)
        
        norm_vec = vec_ij / r_ij[:, np.newaxis]
        B = np.hstack([norm_vec, -norm_vec])
        BB_T = np.einsum('bi,bj->bij', B, B)
        hess_contrib = H_int[:, np.newaxis, np.newaxis] * BB_T
        
        for idx, (i, j) in enumerate(zip(rows, cols)):
            idx_i = slice(3*i, 3*i+3)
            idx_j = slice(3*j, 3*j+3)
            
            h_local = hess_contrib[idx]
            
            self.cart_hess[idx_i, idx_i] += h_local[0:3, 0:3]
            self.cart_hess[idx_i, idx_j] += h_local[0:3, 3:6]
            self.cart_hess[idx_j, idx_i] += h_local[3:6, 0:3]
            self.cart_hess[idx_j, idx_j] += h_local[3:6, 3:6]

    def _calculate_batch_angle_B(self, vec1, vec2, l1, l2):
        """Batch calculation of angle B-matrix with singularity checks."""
        l1_safe = np.maximum(l1, self.MIN_DIST)
        l2_safe = np.maximum(l2, self.MIN_DIST)

        nvec1 = vec1 / l1_safe[:, np.newaxis]
        nvec2 = vec2 / l2_safe[:, np.newaxis]
        
        dot_prod = np.sum(nvec1 * nvec2, axis=1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        
        sin_theta_sq = np.maximum(self.MIN_SIN_SQ, 1.0 - dot_prod**2)
        sin_theta = np.sqrt(sin_theta_sq)
        denom_safe = np.maximum(sin_theta, 1e-6)

        term_i = dot_prod[:, np.newaxis] * nvec1 - nvec2
        B_i = term_i / (l1_safe[:, np.newaxis] * denom_safe[:, np.newaxis])
        
        term_k = dot_prod[:, np.newaxis] * nvec2 - nvec1
        B_k = term_k / (l2_safe[:, np.newaxis] * denom_safe[:, np.newaxis])
        
        B_j = -(B_i + B_k)
        B = np.hstack([B_i, B_j, B_k])
        
        return B, dot_prod, sin_theta_sq

    def _calculate_batch_linear_B(self, vec1, vec2, l1, l2):
        """Batch calculation of linear angle B-matrix components."""
        l1_safe = np.maximum(l1, self.MIN_DIST)
        l2_safe = np.maximum(l2, self.MIN_DIST)

        vn = np.cross(vec1, vec2)
        nvn = np.linalg.norm(vn, axis=1)
        mask_small = nvn < self.MIN_NORM
        
        vn_safe = vn.copy()
        
        if np.any(mask_small):
            idx = np.where(mask_small)[0]
            ref1 = np.array([1.0, 0.0, 0.0])
            scale1 = np.sum(ref1 * vec1[idx], axis=1) / (l1_safe[idx]**2)
            proj1 = scale1[:, np.newaxis] * vec1[idx]
            cand1 = ref1 - proj1
            cand1_norm = np.linalg.norm(cand1, axis=1)
            
            valid1 = cand1_norm >= self.MIN_NORM
            idx_valid1 = idx[valid1]
            vn_safe[idx_valid1] = cand1[valid1]
            nvn[idx_valid1] = cand1_norm[valid1]
            
            still_bad = ~valid1
            if np.any(still_bad):
                idx2 = idx[still_bad]
                ref2 = np.array([0.0, 1.0, 0.0])
                scale2 = np.sum(ref2 * vec1[idx2], axis=1) / (l1_safe[idx2]**2)
                proj2 = scale2[:, np.newaxis] * vec1[idx2]
                cand2 = ref2 - proj2
                cand2_norm = np.linalg.norm(cand2, axis=1)
                cand2_norm = np.maximum(cand2_norm, self.MIN_NORM)
                
                vn_safe[idx2] = cand2
                nvn[idx2] = cand2_norm
        
        nvn_safe = np.maximum(nvn, self.MIN_NORM)
        vn_norm = vn_safe / nvn_safe[:, np.newaxis]
        
        v_diff = vec1 - vec2
        vn2 = np.cross(v_diff, vn_norm)
        vn2_norm = np.linalg.norm(vn2, axis=1)
        vn2_norm_safe = np.maximum(vn2_norm, self.MIN_NORM)
        vn2_n = vn2 / vn2_norm_safe[:, np.newaxis]
        
        B = np.zeros((len(vec1), 2, 9))
        
        B[:, 1, 0:3] = vn_norm / l1_safe[:, np.newaxis]
        B[:, 1, 6:9] = vn_norm / l2_safe[:, np.newaxis]
        B[:, 1, 3:6] = -B[:, 1, 0:3] - B[:, 1, 6:9]
        
        B[:, 0, 0:3] = vn2_n / l1_safe[:, np.newaxis]
        B[:, 0, 6:9] = vn2_n / l2_safe[:, np.newaxis]
        B[:, 0, 3:6] = -B[:, 0, 0:3] - B[:, 0, 6:9]
        
        return B

    def swart_angle_vectorized(self, n_atoms, diff_vec, dists, screen):
        """Vectorized angle calculation centering on atom j."""
        for j in range(n_atoms):
            neighbors = np.where(screen[j, :] >= self.eps2)[0]
            if len(neighbors) < 2:
                continue
            
            idx_i_grid, idx_k_grid = np.meshgrid(neighbors, neighbors, indexing='ij')
            mask_pairs = idx_i_grid < idx_k_grid
            idx_i = idx_i_grid[mask_pairs]
            idx_k = idx_k_grid[mask_pairs]
            
            if len(idx_i) == 0:
                continue
                
            vec1 = diff_vec[idx_i, j]
            vec2 = diff_vec[idx_k, j]
            l1 = dists[idx_i, j]
            l2 = dists[idx_k, j]
            
            s_ij = screen[idx_i, j]
            s_jk = screen[j, idx_k]
            s_ij_jk = s_ij * s_jk
            
            mask_screen = s_ij_jk >= self.eps1
            if not np.any(mask_screen):
                continue
            
            vec1 = vec1[mask_screen]
            vec2 = vec2[mask_screen]
            l1 = l1[mask_screen]
            l2 = l2[mask_screen]
            s_ij_jk = s_ij_jk[mask_screen]
            idx_i = idx_i[mask_screen]
            idx_k = idx_k[mask_screen]
            
            valid_lengths = (l1 > self.MIN_DIST) & (l2 > self.MIN_DIST)
            if not np.any(valid_lengths):
                continue

            vec1 = vec1[valid_lengths]
            vec2 = vec2[valid_lengths]
            l1 = l1[valid_lengths]
            l2 = l2[valid_lengths]
            s_ij_jk = s_ij_jk[valid_lengths]
            idx_i = idx_i[valid_lengths]
            idx_k = idx_k[valid_lengths]

            B_norm, cos_theta, sin_theta_sq = self._calculate_batch_angle_B(vec1, vec2, l1, l2)
            sin_theta = np.sqrt(sin_theta_sq)
            H_base = 0.075 * (s_ij_jk**2) * ((self.f + (1.0 - self.f) * sin_theta)**2)
            
            th1 = np.where(cos_theta > (1.0 - self.tolth), 1.0 - cos_theta, 1.0 + cos_theta)
            is_near_linear = th1 < self.tolth
            
            mask_linear = is_near_linear
            mask_normal = ~mask_linear
            
            if np.any(mask_normal):
                idx_n = np.where(mask_normal)[0]
                H_n = H_base[idx_n]
                B_n = B_norm[idx_n]
                BB_n = np.einsum('ki,kj->kij', B_n, B_n)
                hess_contrib_n = H_n[:, np.newaxis, np.newaxis] * BB_n
                self._accumulate_angle_hessian(idx_i[idx_n], j, idx_k[idx_n], hess_contrib_n)

            if np.any(mask_linear):
                idx_l = np.where(mask_linear)[0]
                th1_l = th1[idx_l]
                cos_l = cos_theta[idx_l]
                H_l = H_base[idx_l]
                safe_tolth = max(self.tolth, 1e-9)
                scale_lin = (1.0 - (th1_l / safe_tolth)**2)**2
                
                B_lin_l = self._calculate_batch_linear_B(vec1[idx_l], vec2[idx_l], l1[idx_l], l2[idx_l])
                B_norm_l = B_norm[idx_l]
                cond_180 = cos_l > (1.0 - self.tolth)
                hess_contrib_l = np.zeros((len(idx_l), 9, 9))
                
                if np.any(cond_180):
                    sub_180 = np.where(cond_180)[0]
                    s_180 = scale_lin[sub_180][:, np.newaxis]
                    B_comb_180 = s_180 * B_lin_l[sub_180, 0, :] + (1.0 - s_180) * B_norm_l[sub_180]
                    B_pure_180 = B_lin_l[sub_180, 1, :]
                    H_180 = H_l[sub_180][:, np.newaxis, np.newaxis]
                    hess_contrib_l[sub_180] += H_180 * np.einsum('ki,kj->kij', B_pure_180, B_pure_180)
                    hess_contrib_l[sub_180] += H_180 * np.einsum('ki,kj->kij', B_comb_180, B_comb_180)

                cond_0 = ~cond_180
                if np.any(cond_0):
                    sub_0 = np.where(cond_0)[0]
                    s_0 = scale_lin[sub_0][:, np.newaxis]
                    B_scaled_0 = (1.0 - s_0) * B_norm_l[sub_0]
                    H_0 = H_l[sub_0][:, np.newaxis, np.newaxis]
                    hess_contrib_l[sub_0] += H_0 * np.einsum('ki,kj->kij', B_scaled_0, B_scaled_0)

                self._accumulate_angle_hessian(idx_i[idx_l], j, idx_k[idx_l], hess_contrib_l)

    def _accumulate_angle_hessian(self, idx_i_arr, j, idx_k_arr, hess_blocks):
        """Helper to add (K, 9, 9) blocks to the global Hessian."""
        for idx, (i, k) in enumerate(zip(idx_i_arr, idx_k_arr)):
            block = hess_blocks[idx]
            range_i = slice(3*i, 3*i+3)
            range_j = slice(3*j, 3*j+3)
            range_k = slice(3*k, 3*k+3)
            
            self.cart_hess[range_i, range_i] += block[0:3, 0:3]
            self.cart_hess[range_i, range_j] += block[0:3, 3:6]
            self.cart_hess[range_i, range_k] += block[0:3, 6:9]
            
            self.cart_hess[range_j, range_i] += block[3:6, 0:3]
            self.cart_hess[range_j, range_j] += block[3:6, 3:6]
            self.cart_hess[range_j, range_k] += block[3:6, 6:9]
            
            self.cart_hess[range_k, range_i] += block[6:9, 0:3]
            self.cart_hess[range_k, range_j] += block[6:9, 3:6]
            self.cart_hess[range_k, range_k] += block[6:9, 6:9]

    def swart_dihedral_angle(self, coord, element_list):
        pass
      
    def swart_out_of_plane(self, coord, element_list):
        pass

    def main(self, coord, element_list, cart_gradient=None):
        """
        Main method to calculate the approximate Hessian using Swart's model.
        Includes a fallback mechanism to distance-only Hessian if NaN/Inf occurs.
        """
        print("Generating Swart's approximate hessian (O1NumHess variant) with Safe Math ...")
        
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
        
        # --- Fallback Logic ---
        # If NaN or Inf detected in the computed Hessian, fallback to simpler model
        if not np.all(np.isfinite(self.cart_hess)):
            print("WARNING: Singularities (NaN or Inf) detected in Swart Hessian calculation.")
            print("         Falling back to Distance-Only (Bond) Hessian.")
            
            # Reset Hessian
            self.cart_hess.fill(0.0)
            
            # Recalculate only bonds (most stable component)
            self.swart_bond_vectorized(n_atoms, diff_vec, dists, screen)
        
       
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
   

        return hess_proj