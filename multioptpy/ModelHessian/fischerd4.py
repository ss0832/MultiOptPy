import numpy as np
import itertools
from scipy.spatial.distance import cdist 

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import D4Parameters
from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.ModelHessian.calc_params import torsion2, stretch2, bend2

# Optimized Helper: JIT-friendly structure
def _fischer_block_assignment(cart_hess, indices, force_const, b_vecs):
    """Helper function to assign force constant to Hessian blocks using slicing."""
    n_indices = len(indices)
    for a in range(n_indices):
        m_atom = indices[a]
        vec_a = b_vecs[a]
        start_m, end_m = 3 * m_atom, 3 * m_atom + 3
        
        for b in range(n_indices):
            n_atom = indices[b]
            vec_b = b_vecs[b]
            start_n, end_n = 3 * n_atom, 3 * n_atom + 3
            
            # Use outer product for speed
            cart_hess[start_m:end_m, start_n:end_n] += force_const * np.outer(vec_a, vec_b)


class FischerD4ApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation with D4 dispersion correction
        Optimized for performance.
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.bond_factor = 1.3
        self.d4_params = D4Parameters()
        self.cart_hess = None
        
    def calc_bond_force_const(self, r_ab, r_ab_cov):
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        val = r_ab_cov * r_ac_cov
        if abs(val) < 1.0e-10: return 0.0
        return 0.089 + 0.11 / (val) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        )
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        val = r_ab * r_ab_cov
        if abs(val) < 1.0e-10: return 0.0
        return 0.0015 + 14.0 * max(bond_sum, 0) ** 0.57 / (val) ** 4.0 * np.exp(
            -2.85 * (r_ab - r_ab_cov)
        )
    
    # --- D4 Utility Methods ---
    def get_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix (Vectorized)"""
        dist_mat = cdist(coord, coord)

        cov_radii = np.array([covalent_radii_lib(e) for e in element_list])
        pair_cov_radii_mat = cov_radii[:, None] + cov_radii[None, :]
        
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)
        
        return bond_mat, dist_mat, pair_cov_radii_mat

    def estimate_atomic_charges(self, element_list, bond_mat):
        """Estimate atomic partial charges (Fully Vectorized)"""
        n_atoms = len(element_list)
        charges = np.zeros(n_atoms)
        en_list = np.array([self.d4_params.get_electronegativity(e) for e in element_list])
        
        i_indices, j_indices = np.where(np.triu(bond_mat, k=1))
        
        if len(i_indices) > 0:
            en_i = en_list[i_indices]
            en_j = en_list[j_indices]
            
            charge_transfer = 0.2 * (en_j - en_i) / (en_i + en_j)
            
            np.add.at(charges, i_indices, charge_transfer)
            np.add.at(charges, j_indices, -charge_transfer)
            
        return charges

    def get_charge_scaling_factor(self, element, charge):
        return np.exp(-self.d4_params.ga * charge * charge)
    
    def get_c6_coefficient(self, element_i, element_j, q_i=0.0, q_j=0.0):
        alpha_i = self.d4_params.get_polarizability(element_i)
        alpha_j = self.d4_params.get_polarizability(element_j)
        scale_i = self.get_charge_scaling_factor(element_i, q_i)
        scale_j = self.get_charge_scaling_factor(element_j, q_j)
        denom = (alpha_i / scale_i) + (alpha_j / scale_j)
        return 2.0 * alpha_i * alpha_j / denom * 0.75
    
    def get_c8_coefficient(self, element_i, element_j, q_i=0.0, q_j=0.0):
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        r4r2_i = self.d4_params.get_r4r2(element_i)
        r4r2_j = self.d4_params.get_r4r2(element_j)
        return 3.0 * c6_ij * np.sqrt(r4r2_i * r4r2_j)
    
    def get_r0_value(self, element_i, element_j):
        try:
            return (covalent_radii_lib(element_i) + covalent_radii_lib(element_j)) * 4.0/3.0
        except:
            return (covalent_radii_lib(element_i) + covalent_radii_lib(element_j)) * 2.0
    
    def d4_damping_function(self, r_ij, r0, order=6):
        if order == 6:
            a1, a2 = self.d4_params.a1, self.d4_params.a2
        else:
            a1, a2 = self.d4_params.a1, self.d4_params.a2 + 2.0
        
        r_pow = r_ij**order
        denominator = r_pow + (a1 * r0 + a2)**order
        return r_pow / denominator
    
    def three_body_damping(self, r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki):
        f_ij = self.d4_damping_function(r_ij, r0_ij, order=6)
        f_jk = self.d4_damping_function(r_jk, r0_jk, order=6)
        f_ki = self.d4_damping_function(r_ki, r0_ki, order=6)
        return f_ij * f_jk * f_ki
    
    # --- Fischer Term Calculations ---
    def fischer_bond(self, coord, element_list):
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(b_c_mat)
        
        for idx in bond_indices:
            i, j = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            force_const = self.calc_bond_force_const(r_ij, r_ij_cov)
            t_xyz = np.array([coord[i], coord[j]])
            _, b_vec = stretch2(t_xyz)
            _fischer_block_assignment(self.cart_hess, (i, j), force_const, b_vec)
    
    def fischer_angle(self, coord, element_list):
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        angle_indices = BC.angle_connect_table(b_c_mat)
        
        for idx in angle_indices:
            i, j, k = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            force_const = self.calc_bend_force_const(r_ij, r_jk, r_ij_cov, r_jk_cov)
            t_xyz = np.array([coord[i], coord[j], coord[k]])
            _, b_vec = bend2(t_xyz)
            _fischer_block_assignment(self.cart_hess, (i, j, k), force_const, b_vec)
    
    def fischer_dihedral(self, coord, element_list, bond_mat):
        """Optimized dihedral calculation with linearity check."""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        bond_counts = bond_mat.sum(axis=1)

        def get_sin_sq_angle(idx1, idx2, idx3):
            v1 = coord[idx1] - coord[idx2]
            v2 = coord[idx3] - coord[idx2]
            cross_prod = np.cross(v1, v2)
            cross_sq = np.dot(cross_prod, cross_prod)
            n1_sq = np.dot(v1, v1)
            n2_sq = np.dot(v2, v2)
            if n1_sq * n2_sq < 1e-12: return 0.0
            return cross_sq / (n1_sq * n2_sq)

        for idx in dihedral_indices:
            i, j, k, l = idx
            
            if get_sin_sq_angle(i, j, k) < 1.0e-3 or get_sin_sq_angle(j, k, l) < 1.0e-3:
                continue
            
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            bond_sum = bond_counts[j] + bond_counts[k] - 2
            
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            
            try:
                _, b_vec = torsion2(t_xyz)
                _fischer_block_assignment(self.cart_hess, (i, j, k, l), force_const, b_vec)
            except Exception:
                continue

    # --- Optimized D4 Methods ---

    def _precompute_active_triplets(self, coord, cutoff=20.0):
        """Identify triplets within cutoff distance ONCE."""
        n_atoms = len(coord)
        cutoff_sq = cutoff**2
        active_triplets = []
        
        neighbors = {i: [] for i in range(n_atoms)}
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d2 = np.sum((coord[i] - coord[j])**2)
                if d2 < cutoff_sq:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        for i in range(n_atoms):
            i_neighbors = set(neighbors[i])
            for j in neighbors[i]:
                if j <= i: continue 
                
                j_neighbors = neighbors[j]
                for k in j_neighbors:
                    if k <= j: continue 
                    if k in i_neighbors:
                        active_triplets.append((i, j, k))
                        
        return active_triplets

    def d4_three_body_gradient_optimized(self, coord, element_list, charges, active_triplets):
        """Calculate gradient using pre-computed active triplets."""
        n_atoms = len(coord)
        gradients = np.zeros((n_atoms, 3))
        
        for i, j, k in active_triplets:
            r_vec_ij = coord[j] - coord[i]
            r_vec_jk = coord[k] - coord[j]
            r_vec_ki = coord[i] - coord[k]
            
            d2_ij = np.dot(r_vec_ij, r_vec_ij)
            d2_jk = np.dot(r_vec_jk, r_vec_jk)
            d2_ki = np.dot(r_vec_ki, r_vec_ki)
            
            r_ij = np.sqrt(d2_ij)
            r_jk = np.sqrt(d2_jk)
            r_ki = np.sqrt(d2_ki)
            
            if min(r_ij, r_jk, r_ki) < 0.1: continue
            
            u_ij = r_vec_ij / r_ij
            u_jk = r_vec_jk / r_jk
            u_ki = r_vec_ki / r_ki
            
            c6_ij = self.get_c6_coefficient(element_list[i], element_list[j], charges[i], charges[j])
            c6_jk = self.get_c6_coefficient(element_list[j], element_list[k], charges[j], charges[k])
            c6_ki = self.get_c6_coefficient(element_list[k], element_list[i], charges[k], charges[i])
            c9_ijk = np.sqrt(c6_ij * c6_jk * c6_ki)
            
            cos_i = np.dot(-u_ki, u_ij)
            cos_j = np.dot(-u_ij, u_jk)
            cos_k = np.dot(-u_jk, u_ki)
            
            angle_factor = 1.0 + 3.0 * cos_i * cos_j * cos_k
            
            r0_ij = self.get_r0_value(element_list[i], element_list[j])
            r0_jk = self.get_r0_value(element_list[j], element_list[k])
            r0_ki = self.get_r0_value(element_list[k], element_list[i])
            damp_factor = self.three_body_damping(r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki)
            
            pre_factor = self.d4_params.s9 * c9_ijk * damp_factor * angle_factor
            
            denom = (r_ij * r_jk * r_ki)**3
            g_base = -3.0 * pre_factor / denom
            
            gradients[i] += g_base * (u_ij / r_ij - u_ki / r_ki)
            gradients[j] += g_base * (u_jk / r_jk - u_ij / r_ij)
            gradients[k] += g_base * (u_ki / r_ki - u_jk / r_jk)
            
            g_ang_coeff = 3.0 * pre_factor / denom
            
            g_ang_i = g_ang_coeff * cos_j * cos_k * (-u_ij - u_ki)
            g_ang_j = g_ang_coeff * cos_i * cos_k * (-u_jk - u_ij)
            g_ang_k = g_ang_coeff * cos_i * cos_j * (-u_ki - u_jk)
            
            gradients[i] += g_ang_i
            gradients[j] += g_ang_j
            gradients[k] += g_ang_k
            
        return gradients

    def d4_three_body_hessian(self, coord, element_list, charges):
        """Calculate Hessian from three-body term using optimized finite difference."""
        n_atoms = len(coord)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        
        active_triplets = self._precompute_active_triplets(coord, cutoff=20.0)
        
        if not active_triplets:
            return hessian

        base_gradients = self.d4_three_body_gradient_optimized(coord, element_list, charges, active_triplets)
        
        delta = 1e-5
        
        for i in range(n_atoms):
            for j in range(3):
                coord_plus = np.copy(coord)
                coord_plus[i, j] += delta
                
                grad_plus = self.d4_three_body_gradient_optimized(coord_plus, element_list, charges, active_triplets)
                
                row_start = 3 * i + j
                row_block = (grad_plus - base_gradients).flatten() / delta
                
                hessian[row_start, :] = row_block
        
        return (hessian + hessian.T) / 2.0

    # --- Re-added Missing Method ---
    def d4_hessian_contribution(self, r_vec, r_ij, element_i, element_j, q_i, q_j):
        """Calculate D4 dispersion contribution to Hessian for a pair of atoms."""
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        r0 = self.get_r0_value(element_i, element_j)
        
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        a1, a2 = self.d4_params.a1, self.d4_params.a2
        a1_8, a2_8 = self.d4_params.a1, self.d4_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        g6 = -self.d4_params.s6 * c6_ij * ((-6.0 / r_ij**7) * f_damp6 + (1.0 / r_ij**6) * df_damp6)
        g8 = -self.d4_params.s8 * c8_ij * ((-8.0 / r_ij**9) * f_damp8 + (1.0 / r_ij**8) * df_damp8)
        
        h6_proj_approx = self.d4_params.s6 * c6_ij / r_ij**8 * (42.0 * f_damp6 - r_ij * df_damp6)
        h8_proj_approx = self.d4_params.s8 * c8_ij / r_ij**10 * (72.0 * f_damp8 - r_ij * df_damp8)
        
        h_proj = h6_proj_approx + h8_proj_approx 
        h_perp = (g6 + g8) / r_ij             

        unit_vec = r_vec / r_ij
        proj_op = np.outer(unit_vec, unit_vec)
        
        identity = np.eye(3)
        return h_proj * proj_op + h_perp * (identity - proj_op)

    def d4_dispersion_hessian(self, coord, element_list, bond_mat, dist_mat):
        """Optimized Pairwise D4 Hessian"""
        n_atoms = len(coord)
        charges = self.estimate_atomic_charges(element_list, bond_mat)
        
        for i in range(n_atoms):
            el_i = element_list[i]
            q_i = charges[i]
            
            for j in range(i):
                if bond_mat[i, j]: continue
                r_ij = dist_mat[i, j]
                if r_ij < 0.1: continue
                
                el_j = element_list[j]
                q_j = charges[j]
                r_vec = coord[i] - coord[j]
                
                hess_block = self.d4_hessian_contribution(r_vec, r_ij, el_i, el_j, q_i, q_j)
                
                si, ei = 3 * i, 3 * i + 3
                sj, ej = 3 * j, 3 * j + 3
                
                self.cart_hess[si:ei, si:ei] += hess_block
                self.cart_hess[sj:ej, sj:ej] += hess_block
                self.cart_hess[si:ei, sj:ej] -= hess_block
                self.cart_hess[sj:ej, si:ei] -= hess_block
        
        three_body_hess = self.d4_three_body_hessian(coord, element_list, charges)
        self.cart_hess += three_body_hess

    def main(self, coord, element_list, cart_gradient):
        print("Generating Hessian using Optimized Fischer + D4 model...")
        
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        self.d4_dispersion_hessian(coord, element_list, bond_mat, dist_mat)
        
        self.cart_hess = (self.cart_hess + self.cart_hess.T) / 2.0
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj