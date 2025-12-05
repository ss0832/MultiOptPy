import numpy as np
from scipy.spatial.distance import cdist # Highly recommended for vectorized distance calculation
from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import D3Parameters, D2_C6_coeff_lib, D2_VDW_radii_lib
from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.ModelHessian.calc_params import torsion2, stretch2, bend2


class FischerD3ApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation with D3 dispersion correction
        Ref: Fischer and Almlöf, J. Phys. Chem., 1992, 96, 24, 9768–9774
        Implementation Ref.: pysisyphus.optimizers.guess_hessians
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.bond_factor = 1.3  # Bond detection threshold factor
        
        # D3 dispersion correction parameters (default: PBE0)
        self.d3_params = D3Parameters()
        self.cart_hess = None
        
    def calc_bond_force_const(self, r_ab, r_ab_cov):
        """Calculate force constant for bond stretching using Fischer formula"""
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        """Calculate force constant for angle bending"""
        val = r_ab_cov * r_ac_cov
        if abs(val) < 1.0e-10:
            return 0.0
        
        return 0.089 + 0.11 / (val) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        )
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        """Calculate force constant for dihedral torsion"""
        val = r_ab * r_ab_cov
        if abs(val) < 1.0e-10:
            return 0.0
        return 0.0015 + 14.0 * max(bond_sum, 0) ** 0.57 / (val) ** 4.0 * np.exp(
            -2.85 * (r_ab - r_ab_cov)
        )
    
    def get_c6_coefficient(self, element_i, element_j):
        """Get C6 coefficient based on D3 model (simplified)"""
        c6_i = D2_C6_coeff_lib(element_i)
        c6_j = D2_C6_coeff_lib(element_j)
        c6_ij = np.sqrt(c6_i * c6_j)
        return c6_ij
    
    def get_c8_coefficient(self, element_i, element_j):
        """Calculate C8 coefficient based on D3 model using reference r4r2 values"""
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        r4r2_i = self.d3_params.get_r4r2(element_i)
        r4r2_j = self.d3_params.get_r4r2(element_j)
        c8_ij = 3.0 * c6_ij * np.sqrt(r4r2_i * r4r2_j)
        return c8_ij
    
    def get_r0_value(self, element_i, element_j):
        """Calculate R0 value for D3 model (characteristic distance for atom pair)"""
        try:
            r_i = D2_VDW_radii_lib(element_i)
            r_j = D2_VDW_radii_lib(element_j)
            return r_i + r_j
        except:
            r_i = covalent_radii_lib(element_i) * 1.5
            r_j = covalent_radii_lib(element_j) * 1.5
            return r_i + r_j
    
    def d3_damping_function(self, r_ij, r0, order=6):
        """BJ (Becke-Johnson) damping function for D3"""
        if order == 6:
            a1, a2 = self.d3_params.a1, self.d3_params.a2
        else:
            a1, a2 = self.d3_params.a1, self.d3_params.a2 + 2.0
            
        denominator = r_ij**order + (a1 * r0 + a2)**order
        return r_ij**order / denominator

    def d3_hessian_contribution(self, r_vec, r_ij, element_i, element_j):
        """Calculate D3 dispersion contribution to Hessian"""
        if r_ij < 0.1:
            return np.zeros((3, 3))
            
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        r0 = self.get_r0_value(element_i, element_j)
        
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d3_params.a1, self.d3_params.a2
        a1_8, a2_8 = self.d3_params.a1, self.d3_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        # df_damp/dr
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # dE/dr (Gradient magnitude)
        g6 = -self.d3_params.s6 * c6_ij * ((-6.0 / r_ij**7) * f_damp6 + (1.0 / r_ij**6) * df_damp6)
        g8 = -self.d3_params.s8 * c8_ij * ((-8.0 / r_ij**9) * f_damp8 + (1.0 / r_ij**8) * df_damp8)
        
        # Unit vector and projection operator
        unit_vec = r_vec / r_ij
        proj_op = np.outer(unit_vec, unit_vec) # P = r_hat * r_hat^T
        
        # Coefficients for H = (d^2E/dr^2) * P + (1/r * dE/dr) * (I - P)
        # Using the simplified structure from the original code for d^2E/dr^2 approximation:
        h6_proj_coeff = self.d3_params.s6 * c6_ij / r_ij**8 * (42.0 * f_damp6 - r_ij * df_damp6)
        h8_proj_coeff = self.d3_params.s8 * c8_ij / r_ij**10 * (72.0 * f_damp8 - r_ij * df_damp8)
        
        h_proj = h6_proj_coeff + h8_proj_coeff # d^2E/dr^2 approximation
        h_perp = (g6 + g8) / r_ij             # 1/r * dE/dr (Perpendicular coefficient)
        
        # Construct Hessian matrix
        identity = np.eye(3)
        hessian = h_proj * proj_op + h_perp * (identity - proj_op)
        
        return hessian

    # --- Optimized: Vectorized connectivity calculation ---
    def get_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix and related data (Optimized with vectorization)"""
        n_atoms = len(coord)
        
        # 1. Distance matrix (Vectorized)
        try:
            dist_mat = cdist(coord, coord)
        except NameError:
            diff = coord[:, None, :] - coord[None, :, :]
            dist_mat = np.linalg.norm(diff, axis=-1)

        # 2. Covalent radii sums (Vectorized)
        cov_radii = np.array([covalent_radii_lib(e) for e in element_list])
        pair_cov_radii_mat = cov_radii[:, None] + cov_radii[None, :]
        
        # 3. Bond connectivity matrix
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)
        
        return bond_mat, dist_mat, pair_cov_radii_mat
    
    # --- Optimized: Block assignment using slicing ---
    def fischer_bond(self, coord, element_list):
        """Calculate Hessian components for bond stretching (Optimized with slicing)"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(b_c_mat)
        
        for idx in bond_indices:
            i, j = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            
            force_const = self.calc_bond_force_const(r_ij, r_ij_cov)
            
            t_xyz = np.array([coord[i], coord[j]])
            r, b_vec = stretch2(t_xyz)
            
            # Optimized: Use NumPy slicing and outer product
            b_vec_i, b_vec_j = b_vec[0], b_vec[1] 
            
            H_ii_block = force_const * np.outer(b_vec_i, b_vec_i)
            H_jj_block = force_const * np.outer(b_vec_j, b_vec_j)
            H_ij_block = force_const * np.outer(b_vec_i, b_vec_j)
            H_ji_block = force_const * np.outer(b_vec_j, b_vec_i) 

            start_i, end_i = 3 * i, 3 * i + 3
            start_j, end_j = 3 * j, 3 * j + 3

            self.cart_hess[start_i:end_i, start_i:end_i] += H_ii_block
            self.cart_hess[start_j:end_j, start_j:end_j] += H_jj_block
            self.cart_hess[start_i:end_i, start_j:end_j] += H_ij_block
            self.cart_hess[start_j:end_j, start_i:end_i] += H_ji_block


    def fischer_angle(self, coord, element_list):
        """Calculate Hessian components for angle bending (Optimized with slicing)"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        angle_indices = BC.angle_connect_table(b_c_mat)
        
        for idx in angle_indices:
            i, j, k = idx  # i-j-k angle
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            force_const = self.calc_bend_force_const(r_ij, r_jk, r_ij_cov, r_jk_cov)
            
            t_xyz = np.array([coord[i], coord[j], coord[k]])
            theta, b_vec = bend2(t_xyz)
            
            # Optimized: Use NumPy slicing and outer product
            atoms = [i, j, k]
            
            for m_idx, m_atom in enumerate(atoms):
                for n_idx, n_atom in enumerate(atoms):
                    start_m, end_m = 3 * m_atom, 3 * m_atom + 3
                    start_n, end_n = 3 * n_atom, 3 * n_atom + 3
                    
                    H_mn_block = force_const * np.outer(b_vec[m_idx], b_vec[n_idx])
                    self.cart_hess[start_m:end_m, start_n:end_n] += H_mn_block


    def fischer_dihedral(self, coord, element_list, bond_mat):
        """Calculate Hessian components for dihedral torsion (Optimized with singularity damping)"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        # Calculate bond count for central atoms in dihedrals
        tors_atom_bonds = {}
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            bond_sum = bond_mat[j].sum() + bond_mat[k].sum() - 2
            tors_atom_bonds[(j, k)] = bond_sum

        for idx in dihedral_indices:
            i, j, k, l = idx
            
            # Vector calculations
            vec_ji = coord[i] - coord[j]
            vec_jk = coord[k] - coord[j]
            vec_kl = coord[l] - coord[k]
            
            r_jk = np.linalg.norm(vec_jk)
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            bond_sum = tors_atom_bonds.get((j, k), 0)
            
            # Calculate base force constant
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            
            # --- Singularity Handling (Damping for linear angles) ---
            # Determine linearity of angles i-j-k and j-k-l
            
            # Angle 1: i-j-k 
            n_ji = np.linalg.norm(vec_ji)
            n_jk = r_jk # already calculated
            
            # Avoid division by zero if atoms overlap
            if n_ji < 1e-8 or n_jk < 1e-8: continue
            
            cos_theta1 = np.dot(vec_ji, vec_jk) / (n_ji * n_jk)
            
            # Angle 2: j-k-l (Note: vec_kj is -vec_jk)
            vec_kj = -vec_jk
            n_kl = np.linalg.norm(vec_kl)
            
            if n_kl < 1e-8: continue
            
            cos_theta2 = np.dot(vec_kj, vec_kl) / (n_jk * n_kl)

            # --- Damping Factor Calculation ---
            # The Wilson B-matrix contains 1/sin(theta) terms which diverge at 180 deg.
            # We scale the force constant by sin^2(theta) to cancel this divergence.
            # sin^2(theta) = 1 - cos^2(theta)
            
            sin2_theta1 = 1.0 - min(cos_theta1**2, 1.0)
            sin2_theta2 = 1.0 - min(cos_theta2**2, 1.0)
            
            # Hard cutoff: If geometry is extremely linear, skip to avoid NaN
            if sin2_theta1 < 1e-4 or sin2_theta2 < 1e-4:
                continue

            # Apply scaling factor
            # This ensures the force constant goes to 0 as the angle becomes linear
            scaling_factor = sin2_theta1 * sin2_theta2
            force_const *= scaling_factor
            
            # --------------------------------------------------------

            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            
            try:
                tau, b_vec = torsion2(t_xyz)
            except (ValueError, ArithmeticError):
                # Skip if numerical errors occur in torsion calculation
                continue
            
            # Optimized: Use NumPy slicing and outer product
            atoms = [i, j, k, l]
            
            for m_idx, m_atom in enumerate(atoms):
                for n_idx, n_atom in enumerate(atoms):
                    start_m, end_m = 3 * m_atom, 3 * m_atom + 3
                    start_n, end_n = 3 * n_atom, 3 * n_atom + 3
                    
                    H_mn_block = force_const * np.outer(b_vec[m_idx], b_vec[n_idx])
                    self.cart_hess[start_m:end_m, start_n:end_n] += H_mn_block

    # --- Optimized: Block assignment using slicing (and fixed logic) ---
    def d3_dispersion_hessian(self, coord, element_list, bond_mat):
        """Calculate Hessian correction based on D3 dispersion forces (Optimized/Corrected)"""
        n_atoms = len(coord)
        
        # Calculate D3 dispersion correction for all atom pairs (i > j)
        for i in range(n_atoms):
            for j in range(i):
                # Skip bonded atom pairs
                if bond_mat[i, j]:
                    continue
                    
                r_vec = coord[i] - coord[j]
                r_ij = np.linalg.norm(r_vec)
                
                if r_ij < 0.1:
                    continue
                
                # Calculate D3 Hessian contribution (3x3 block)
                hess_block = self.d3_hessian_contribution(r_vec, r_ij, element_list[i], element_list[j])
                
                # Use slicing for efficient block assignment
                # H_ii += H_block, H_jj += H_block, H_ij -= H_block, H_ji -= H_block
                start_i, end_i = 3 * i, 3 * i + 3
                start_j, end_j = 3 * j, 3 * j + 3
                
                self.cart_hess[start_i:end_i, start_i:end_i] += hess_block
                self.cart_hess[start_j:end_j, start_j:end_j] += hess_block
                self.cart_hess[start_i:end_i, start_j:end_j] -= hess_block
                self.cart_hess[start_j:end_j, start_i:end_i] -= hess_block
    
    # --- Optimized: Main function flow ---
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian combining Fischer model and D3 dispersion correction
        """
        print("Generating Hessian using Fischer model with D3 dispersion correction...")
        
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate bond connectivity matrix ONCE (Optimized internally)
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Calculate Hessian components from Fischer model (Optimized internally with slicing)
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        # Calculate Hessian components from D3 dispersion correction (Optimized internally with slicing)
        self.d3_dispersion_hessian(coord, element_list, bond_mat)
        
        # Optimized: Symmetrize the Hessian matrix
        self.cart_hess = (self.cart_hess + self.cart_hess.T) / 2.0
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj