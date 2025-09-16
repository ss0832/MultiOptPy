import numpy as np

from multioptpy.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.calc_tools import Calculationtools
from multioptpy.parameter import D3Parameters, D2_C6_coeff_lib, D2_VDW_radii_lib
from multioptpy.bond_connectivity import BondConnectivity
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
        
    def calc_bond_force_const(self, r_ab, r_ab_cov):
        """Calculate force constant for bond stretching using Fischer formula"""
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        """Calculate force constant for angle bending"""
        val = r_ab_cov * r_ac_cov
        if abs(val) < 1.0e-10:
            return 0.0  # Avoid division by zero
        
        return 0.089 + 0.11 / (val) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        ) 
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        """Calculate force constant for dihedral torsion"""
        val = r_ab * r_ab_cov
        if abs(val) < 1.0e-10:
            return 0.0  # Avoid division by zero
        return 0.0015 + 14.0 * max(bond_sum, 0) ** 0.57 / (val) ** 4.0 * np.exp(
            -2.85 * (r_ab - r_ab_cov)
        )
    
    def get_c6_coefficient(self, element_i, element_j):
        """Get C6 coefficient based on D3 model (simplified)"""
        # Use D2's C6 coefficients as base
        c6_i = D2_C6_coeff_lib(element_i)
        c6_j = D2_C6_coeff_lib(element_j)
        
        # In D3, this becomes environment-dependent, but for simplicity, use the same calculation method as D2
        c6_ij = np.sqrt(c6_i * c6_j)
        return c6_ij
    
    def get_c8_coefficient(self, element_i, element_j):
        """Calculate C8 coefficient based on D3 model using reference r4r2 values"""
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        r4r2_i = self.d3_params.get_r4r2(element_i)
        r4r2_j = self.d3_params.get_r4r2(element_j)
        
        # C8 = 3 * C6 * sqrt(r4r2_i * r4r2_j)
        c8_ij = 3.0 * c6_ij * np.sqrt(r4r2_i * r4r2_j)
        return c8_ij
    
    def get_r0_value(self, element_i, element_j):
        """Calculate R0 value for D3 model (characteristic distance for atom pair)"""
        # Use existing D2 van der Waals radii
        try:
            r_i = D2_VDW_radii_lib(element_i)
            r_j = D2_VDW_radii_lib(element_j)
            return r_i + r_j
        except:
            # If exception occurs, estimate from covalent radii
            r_i = covalent_radii_lib(element_i) * 1.5
            r_j = covalent_radii_lib(element_j) * 1.5
            return r_i + r_j
    
    def d3_damping_function(self, r_ij, r0, order=6):
        """
        BJ (Becke-Johnson) damping function for D3
        
        Parameters:
            r_ij: Interatomic distance
            r0: Reference radius
            order: 6 for C6 term, 8 for C8 term
        """
        if order == 6:
            a1, a2 = self.d3_params.a1, self.d3_params.a2
        else:  # order == 8
            a1, a2 = self.d3_params.a1, self.d3_params.a2 + 2.0  # C8 damping is slightly different
            
        # BJ-damping (Becke-Johnson)
        denominator = r_ij**order + (a1 * r0 + a2)**order
        return r_ij**order / denominator
    
    def d3_energy_contribution(self, r_ij, element_i, element_j):
        """
        Calculate D3 dispersion energy
        
        Parameters:
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return 0.0
            
        # Get C6 and C8 coefficients
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Energy calculation
        e6 = -self.d3_params.s6 * c6_ij / r_ij**6 * f_damp6
        e8 = -self.d3_params.s8 * c8_ij / r_ij**8 * f_damp8
        
        return e6 + e8
    
    def d3_gradient_contribution(self, r_vec, r_ij, element_i, element_j):
        """
        Calculate D3 dispersion gradient
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros(3)
            
        # Get C6 and C8 coefficients
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d3_params.a1, self.d3_params.a2
        a1_8, a2_8 = self.d3_params.a1, self.d3_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Gradient calculation
        g6 = -self.d3_params.s6 * c6_ij * ((-6 / r_ij**7) * f_damp6 + (1 / r_ij**6) * df_damp6)
        g8 = -self.d3_params.s8 * c8_ij * ((-8 / r_ij**9) * f_damp8 + (1 / r_ij**8) * df_damp8)
        
        unit_vec = r_vec / r_ij
        return (g6 + g8) * unit_vec
    
    def d3_hessian_contribution(self, r_vec, r_ij, element_i, element_j):
        """
        Calculate D3 dispersion contribution to Hessian
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros((3, 3))
            
        # Get C6 and C8 coefficients
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d3_params.a1, self.d3_params.a2
        a1_8, a2_8 = self.d3_params.a1, self.d3_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Unit vector and projection operator
        unit_vec = r_vec / r_ij
        proj_op = np.outer(unit_vec, unit_vec)
        
        # C6 term contribution to Hessian
        h6_coeff = self.d3_params.s6 * c6_ij / r_ij**8 * (42.0 * f_damp6 - r_ij * df_damp6)
        
        # C8 term contribution to Hessian
        h8_coeff = self.d3_params.s8 * c8_ij / r_ij**10 * (72.0 * f_damp8 - r_ij * df_damp8)
        
        # Calculation of projection and perpendicular parts
        h_proj = h6_coeff + h8_coeff
        h_perp = (self.d3_params.s6 * c6_ij * 6.0 / r_ij**8 + self.d3_params.s8 * c8_ij * 8.0 / r_ij**10) * f_damp6
        
        # Construct Hessian matrix
        identity = np.eye(3)
        hessian = h_proj * proj_op + h_perp * (identity - proj_op)
        
        return hessian
    
    def get_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix and related data"""
        n_atoms = len(coord)
        dist_mat = np.zeros((n_atoms, n_atoms))
        pair_cov_radii_mat = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coord[i] - coord[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
                
                cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                pair_cov_radii_mat[i, j] = pair_cov_radii_mat[j, i] = cov_sum
        
        # Bond connectivity matrix (True if bond exists between atoms)
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)  # No self-bonds
        
        return bond_mat, dist_mat, pair_cov_radii_mat
    
    def count_bonds_for_dihedral(self, bond_mat, central_atoms):
        """Count number of bonds for central atoms in a dihedral"""
        a, b = central_atoms
        # Sum bonds for both central atoms and subtract 2 (the bond between them is counted twice)
        bond_sum = bond_mat[a].sum() + bond_mat[b].sum() - 2
        return bond_sum
    
    def fischer_bond(self, coord, element_list):
        """Calculate Hessian components for bond stretching"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(b_c_mat)
        
        for idx in bond_indices:
            i, j = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_bond_force_const(r_ij, r_ij_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j]])
            r, b_vec = stretch2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
    
    def fischer_angle(self, coord, element_list):
        """Calculate Hessian components for angle bending"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        angle_indices = BC.angle_connect_table(b_c_mat)
        
        for idx in angle_indices:
            i, j, k = idx  # i-j-k angle
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_bend_force_const(r_ij, r_jk, r_ij_cov, r_jk_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k]])
            theta, b_vec = bend2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
    
    def fischer_dihedral(self, coord, element_list, bond_mat):
        """Calculate Hessian components for dihedral torsion"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        # Calculate bond count for central atoms in dihedrals
        tors_atom_bonds = {}
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            bond_sum = self.count_bonds_for_dihedral(bond_mat, (j, k))
            tors_atom_bonds[(j, k)] = bond_sum
        
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            
            # Central bond
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Get bond count for central atoms
            bond_sum = tors_atom_bonds.get((j, k), 0)
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            tau, b_vec = torsion2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    self.cart_hess[3*l+n, 3*l+m] += force_const * b_vec[3][n] * b_vec[3][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*i+n, 3*l+m] += force_const * b_vec[0][n] * b_vec[3][m]
                    
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*l+m] += force_const * b_vec[1][n] * b_vec[3][m]
                    
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*l+m] += force_const * b_vec[2][n] * b_vec[3][m]
                    
                    self.cart_hess[3*l+n, 3*i+m] += force_const * b_vec[3][n] * b_vec[0][m]
                    self.cart_hess[3*l+n, 3*j+m] += force_const * b_vec[3][n] * b_vec[1][m]
                    self.cart_hess[3*l+n, 3*k+m] += force_const * b_vec[3][n] * b_vec[2][m]
    
    def d3_dispersion_hessian(self, coord, element_list, bond_mat):
        """Calculate Hessian correction based on D3 dispersion forces"""
        n_atoms = len(coord)
        
        # Calculate D3 dispersion correction for all atom pairs
        for i in range(n_atoms):
            for j in range(i):
                # Skip bonded atom pairs (already accounted for in Fischer model)
                if bond_mat[i, j]:
                    continue
                    
                # Calculate distance vector and magnitude
                r_vec = coord[i] - coord[j]
                r_ij = np.linalg.norm(r_vec)
                
                # Skip if atoms are too close
                if r_ij < 0.1:
                    continue
                
                # Calculate D3 Hessian contribution
                hess_block = self.d3_hessian_contribution(r_vec, r_ij, element_list[i], element_list[j])
                
                # Add to the Hessian matrix
                for n in range(3):
                    for m in range(3):
                        self.cart_hess[3*i+n, 3*i+m] += hess_block[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_block[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_block[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_block[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_block[n, m]
                        self.cart_hess[3*j+n, 3*i+m] -= hess_block[n, m]
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian combining Fischer model and D3 dispersion correction
        
        Parameters:
            coord: Atomic coordinates (N×3 array, Bohr)
            element_list: List of element symbols
            cart_gradient: Gradient in Cartesian coordinates
            
        Returns:
            hess_proj: Hessian with rotational and translational modes projected out
        """
        print("Generating Hessian using Fischer model with D3 dispersion correction...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate bond connectivity matrix
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Calculate Hessian components from Fischer model
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        # Calculate Hessian components from D3 dispersion correction
        self.d3_dispersion_hessian(coord, element_list, bond_mat)
        
        # Symmetrize the lower triangle of the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj
    