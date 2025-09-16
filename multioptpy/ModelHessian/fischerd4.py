import numpy as np
import itertools

from multioptpy.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.calc_tools import Calculationtools
from multioptpy.parameter import D4Parameters
from multioptpy.bond_connectivity import BondConnectivity
from multioptpy.ModelHessian.calc_params import torsion2, stretch2, bend2



class FischerD4ApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation with D4 dispersion correction
        Ref: Fischer and Almlöf, J. Phys. Chem., 1992, 96, 24, 9768–9774
        D4 Ref: Caldeweyher et al., J. Chem. Phys., 2019, 150, 154122
        Implementation Ref.:pysisyphus.optimizers.guess_hessians
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.bond_factor = 1.3  # Bond detection threshold factor
        
        # D4 dispersion correction parameters (default: PBE0)
        self.d4_params = D4Parameters()
        
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
    
    def calculate_coordination_numbers(self, coord, element_list):
        """
        Calculate atomic coordination numbers
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            
        Returns:
            cn: array of coordination numbers for each atom
        """
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                # Get covalent radii
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                
                # Calculate distance
                dist_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Calculate coordination number contribution using cutoff function
                r_cov = r_cov_i + r_cov_j
                tmp = np.exp(-16.0 * ((dist_ij / (r_cov * 1.2)) - 1.0))
                cn[i] += 1.0 / (1.0 + tmp)
                
        return cn
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic partial charges using a simple electronegativity model
        In a real implementation, this would use proper EEQ method or external charges
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            
        Returns:
            charges: array of estimated atomic charges
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Simple estimation using bond distances and electronegativity differences
        bond_mat, dist_mat, _ = self.get_bond_connectivity(coord, element_list)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bond_mat[i, j]:
                    en_i = self.d4_params.get_electronegativity(element_list[i])
                    en_j = self.d4_params.get_electronegativity(element_list[j])
                    
                    # Simple electronegativity-based charge transfer
                    # The 0.1 factor is a simplification; real EEQ would be more complex
                    charge_transfer = 0.1 * (en_j - en_i) / (en_i + en_j) * 2.0
                    charges[i] += charge_transfer
                    charges[j] -= charge_transfer
        
        return charges
    
    def get_charge_scaling_factor(self, element, charge):
        """
        Calculate charge scaling factor for D4 dispersion coefficients
        
        Parameters:
            element: element symbol
            charge: atomic charge
            
        Returns:
            charge_factor: exponential charge scaling factor
        """
        ga = self.d4_params.ga
        charge_factor = np.exp(-ga * charge * charge)
        return charge_factor
    
    def get_c6_coefficient(self, element_i, element_j, q_i=0.0, q_j=0.0):
        """
        Get C6 coefficient based on D4 model with charge scaling
        
        Parameters:
            element_i, element_j: element symbols for atoms
            q_i, q_j: atomic charges
            
        Returns:
            c6_ij: dispersion coefficient for atom pair
        """
        # Get reference polarizabilities
        alpha_i = self.d4_params.get_polarizability(element_i)
        alpha_j = self.d4_params.get_polarizability(element_j)
        
        # Apply charge scaling
        scale_i = self.get_charge_scaling_factor(element_i, q_i)
        scale_j = self.get_charge_scaling_factor(element_j, q_j)
        
        # D4 approach using dynamic polarizabilities
        # The 0.75 factor is an empirical scaling constant
        c6_ij = 2.0 * alpha_i * alpha_j / (alpha_i / scale_i + alpha_j / scale_j) * 0.75
        
        return c6_ij
    
    def get_c8_coefficient(self, element_i, element_j, q_i=0.0, q_j=0.0):
        """
        Calculate C8 coefficient based on D4 model
        
        Parameters:
            element_i, element_j: element symbols for atoms
            q_i, q_j: atomic charges
            
        Returns:
            c8_ij: higher-order dispersion coefficient
        """
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        r4r2_i = self.d4_params.get_r4r2(element_i)
        r4r2_j = self.d4_params.get_r4r2(element_j)
        
        # C8 = 3 * C6 * sqrt(r4r2_i * r4r2_j)
        c8_ij = 3.0 * c6_ij * np.sqrt(r4r2_i * r4r2_j)
        return c8_ij
    
    def get_r0_value(self, element_i, element_j):
        """
        Calculate R0 value for D4 model (characteristic distance for atom pair)
        
        Parameters:
            element_i, element_j: element symbols for atoms
            
        Returns:
            r0: reference distance for damping function
        """
        # Using covalent radii as a base for reference distance
        try:
            r_i = covalent_radii_lib(element_i) * 4.0/3.0
            r_j = covalent_radii_lib(element_j) * 4.0/3.0
            return r_i + r_j
        except:
            # If exception occurs, estimate from covalent radii
            r_i = covalent_radii_lib(element_i) * 2.0
            r_j = covalent_radii_lib(element_j) * 2.0
            return r_i + r_j
    
    def d4_damping_function(self, r_ij, r0, order=6):
        """
        BJ (Becke-Johnson) damping function for D4
        
        Parameters:
            r_ij: Interatomic distance
            r0: Reference radius
            order: 6 for C6 term, 8 for C8 term
        
        Returns:
            f_damp: value of damping function
        """
        if order == 6:
            a1, a2 = self.d4_params.a1, self.d4_params.a2
        else:  # order == 8
            a1, a2 = self.d4_params.a1, self.d4_params.a2 + 2.0  # C8 damping is slightly different
            
        # BJ-damping (Becke-Johnson)
        denominator = r_ij**order + (a1 * r0 + a2)**order
        return r_ij**order / denominator
    
    def three_body_damping(self, r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki):
        """
        Three-body damping function for the ATM term
        
        Parameters:
            r_ij, r_jk, r_ki: Interatomic distances for the triangle
            r0_ij, r0_jk, r0_ki: Reference radii for the atom pairs
            
        Returns:
            f_damp: value of three-body damping function
        """
        # The geometric average of the three damping functions
        f_ij = self.d4_damping_function(r_ij, r0_ij, order=6)
        f_jk = self.d4_damping_function(r_jk, r0_jk, order=6)
        f_ki = self.d4_damping_function(r_ki, r0_ki, order=6)
        
        return f_ij * f_jk * f_ki
    
    def d4_energy_contribution(self, r_ij, element_i, element_j, q_i, q_j):
        """
        Calculate D4 pairwise dispersion energy
        
        Parameters:
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
            q_i, q_j: Atomic charges
        
        Returns:
            energy: dispersion energy contribution for the atom pair
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return 0.0
            
        # Get C6 and C8 coefficients with charge-dependence
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        # Energy calculation
        e6 = -self.d4_params.s6 * c6_ij / r_ij**6 * f_damp6
        e8 = -self.d4_params.s8 * c8_ij / r_ij**8 * f_damp8
        
        return e6 + e8
    
    def calculate_three_body_term(self, coord, element_list, charges):
        """
        Calculate the three-body dispersion energy term (ATM)
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            charges: array of atomic charges
            
        Returns:
            energy: three-body dispersion energy
        """
        energy = 0.0
        n_atoms = len(coord)
        
        # Calculate for all atom triplets
        for i, j, k in itertools.combinations(range(n_atoms), 3):
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ki = np.linalg.norm(coord[k] - coord[i])
            
            # Skip if any atoms are too close
            if min(r_ij, r_jk, r_ki) < 0.1:
                continue
                
            # Charge-dependent three-body C9 coefficient
            c6_ij = self.get_c6_coefficient(element_list[i], element_list[j], charges[i], charges[j])
            c6_jk = self.get_c6_coefficient(element_list[j], element_list[k], charges[j], charges[k])
            c6_ki = self.get_c6_coefficient(element_list[k], element_list[i], charges[k], charges[i])
            c9_ijk = np.sqrt(c6_ij * c6_jk * c6_ki)
            
            # Calculate angular dependent factor for ATM term
            r_vec_ij = coord[j] - coord[i]
            r_vec_jk = coord[k] - coord[j]
            r_vec_ki = coord[i] - coord[k]
            
            cos_i = np.dot(-r_vec_ki, r_vec_ij) / (r_ki * r_ij)
            cos_j = np.dot(-r_vec_ij, r_vec_jk) / (r_ij * r_jk)
            cos_k = np.dot(-r_vec_jk, r_vec_ki) / (r_jk * r_ki)
            
            angle_factor = 1.0 + 3.0 * cos_i * cos_j * cos_k
            
            # Calculate damping function
            r0_ij = self.get_r0_value(element_list[i], element_list[j])
            r0_jk = self.get_r0_value(element_list[j], element_list[k])
            r0_ki = self.get_r0_value(element_list[k], element_list[i])
            damp_factor = self.three_body_damping(r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki)
            
            # Axilrod-Teller-Muto term
            e_atm = angle_factor * c9_ijk * damp_factor / (r_ij * r_jk * r_ki)**3
            energy += e_atm
            
        return energy * self.d4_params.s9
    
    def d4_gradient_contribution(self, r_vec, r_ij, element_i, element_j, q_i, q_j):
        """
        Calculate D4 pairwise dispersion gradient
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
            q_i, q_j: Atomic charges
        
        Returns:
            gradient: dispersion gradient contribution for the atom pair
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros(3)
            
        # Get C6 and C8 coefficients with charge-dependence
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d4_params.a1, self.d4_params.a2
        a1_8, a2_8 = self.d4_params.a1, self.d4_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Gradient calculation
        g6 = -self.d4_params.s6 * c6_ij * ((-6 / r_ij**7) * f_damp6 + (1 / r_ij**6) * df_damp6)
        g8 = -self.d4_params.s8 * c8_ij * ((-8 / r_ij**9) * f_damp8 + (1 / r_ij**8) * df_damp8)
        
        unit_vec = r_vec / r_ij
        return (g6 + g8) * unit_vec
    
    def d4_three_body_gradient(self, coord, element_list, charges):
        """
        Calculate gradient contributions from the three-body dispersion term
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            charges: array of atomic charges
            
        Returns:
            gradients: gradient contributions from three-body term (N×3 array)
        """
        n_atoms = len(coord)
        gradients = np.zeros((n_atoms, 3))
        
        # Calculate for all atom triplets
        for i, j, k in itertools.combinations(range(n_atoms), 3):
            # Distances between atoms
            r_vec_ij = coord[j] - coord[i]
            r_vec_jk = coord[k] - coord[j]
            r_vec_ki = coord[i] - coord[k]
            
            r_ij = np.linalg.norm(r_vec_ij)
            r_jk = np.linalg.norm(r_vec_jk)
            r_ki = np.linalg.norm(r_vec_ki)
            
            # Skip if any atoms are too close
            if min(r_ij, r_jk, r_ki) < 0.1:
                continue
                
            # Unit vectors
            u_ij = r_vec_ij / r_ij
            u_jk = r_vec_jk / r_jk
            u_ki = r_vec_ki / r_ki
            
            # Charge-dependent three-body C9 coefficient
            c6_ij = self.get_c6_coefficient(element_list[i], element_list[j], charges[i], charges[j])
            c6_jk = self.get_c6_coefficient(element_list[j], element_list[k], charges[j], charges[k])
            c6_ki = self.get_c6_coefficient(element_list[k], element_list[i], charges[k], charges[i])
            c9_ijk = np.sqrt(c6_ij * c6_jk * c6_ki)
            
            # Get cosines for the angles
            cos_i = np.dot(-u_ki, u_ij)
            cos_j = np.dot(-u_ij, u_jk)
            cos_k = np.dot(-u_jk, u_ki)
            
            # Angular factor and its derivatives
            angle_factor = 1.0 + 3.0 * cos_i * cos_j * cos_k
            
            # Calculate damping
            # Calculate damping function
            r0_ij = self.get_r0_value(element_list[i], element_list[j])
            r0_jk = self.get_r0_value(element_list[j], element_list[k])
            r0_ki = self.get_r0_value(element_list[k], element_list[i])
            damp_factor = self.three_body_damping(r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki)
            
            # Prefactor for gradient calculation
            pre_factor = self.d4_params.s9 * c9_ijk * damp_factor * angle_factor
            
            # Base gradient components (without angular derivatives)
            g_base = -3.0 * pre_factor / (r_ij * r_jk * r_ki)**3
            
            # Gradient contributions for each atom from distances
            g_i = g_base * (u_ij / r_ij - u_ki / r_ki)
            g_j = g_base * (u_jk / r_jk - u_ij / r_ij)
            g_k = g_base * (u_ki / r_ki - u_jk / r_jk)
            
            # Add angular derivative components (simplified approximation)
            # This is a simplified approach; a full analytical derivation would be more complex
            g_ang_i = 3.0 * pre_factor * (cos_j * cos_k / (r_ij * r_jk * r_ki)**3) * (-u_ij - u_ki)
            g_ang_j = 3.0 * pre_factor * (cos_i * cos_k / (r_ij * r_jk * r_ki)**3) * (-u_jk - u_ij)
            g_ang_k = 3.0 * pre_factor * (cos_i * cos_j / (r_ij * r_jk * r_ki)**3) * (-u_ki - u_jk)
            
            # Combine distance and angular contributions
            gradients[i] += g_i + g_ang_i
            gradients[j] += g_j + g_ang_j
            gradients[k] += g_k + g_ang_k
            
        return gradients
    
    def d4_hessian_contribution(self, r_vec, r_ij, element_i, element_j, q_i, q_j):
        """
        Calculate D4 dispersion contribution to Hessian for a pair of atoms
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
            q_i, q_j: Atomic charges
        
        Returns:
            hessian: 3×3 Hessian block for the atom pair
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros((3, 3))
            
        # Get C6 and C8 coefficients with charge-dependence
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d4_params.a1, self.d4_params.a2
        a1_8, a2_8 = self.d4_params.a1, self.d4_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Second derivatives of damping functions (approximate)
        d2f_damp6 = (30 * r_ij**4 / denom6 - 60 * r_ij**11 / denom6**2) - df_damp6 / r_ij
        d2f_damp8 = (56 * r_ij**6 / denom8 - 112 * r_ij**15 / denom8**2) - df_damp8 / r_ij
        
        # Unit vector and projection operator
        unit_vec = r_vec / r_ij
        proj_op = np.outer(unit_vec, unit_vec)
        
        # C6 term contribution to Hessian
        h6_coeff = self.d4_params.s6 * c6_ij * (
            (42 / r_ij**8) * f_damp6 - 
            (6 / r_ij**7) * df_damp6 + 
            (1 / r_ij**6) * d2f_damp6
        )
        
        # C8 term contribution to Hessian
        h8_coeff = self.d4_params.s8 * c8_ij * (
            (72 / r_ij**10) * f_damp8 - 
            (8 / r_ij**9) * df_damp8 + 
            (1 / r_ij**8) * d2f_damp8
        )
        
        # Calculation of projection and perpendicular parts
        h_proj = h6_coeff + h8_coeff
        h_perp = (
            self.d4_params.s6 * c6_ij * (6 / r_ij**8) * f_damp6 + 
            self.d4_params.s8 * c8_ij * (8 / r_ij**10) * f_damp8
        )
        
        # Construct Hessian matrix
        identity = np.eye(3)
        hessian = h_proj * proj_op + h_perp * (identity - proj_op)
        
        return hessian
    
    def d4_three_body_hessian(self, coord, element_list, charges):
        """
        Calculate Hessian contributions from the three-body dispersion term
        This is a simplified version using finite differences
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            charges: array of atomic charges
            
        Returns:
            hessian: Hessian matrix contribution from three-body term (3N×3N array)
        """
        n_atoms = len(coord)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        
        # Calculate base gradients
        base_gradients = self.d4_three_body_gradient(coord, element_list, charges)
        
        # Numerical Hessian calculation using finite differences
        delta = 1e-5  # Small displacement for finite difference
        
        # For each atom and coordinate
        for i in range(n_atoms):
            for j in range(3):
                # Create displaced coordinates
                coord_plus = np.copy(coord)
                coord_plus[i, j] += delta
                
                # Calculate gradients at displaced coordinates
                grad_plus = self.d4_three_body_gradient(coord_plus, element_list, charges)
                
                # Calculate Hessian elements using finite difference
                for k in range(n_atoms):
                    for l in range(3):
                        hessian[3*i+j, 3*k+l] += (grad_plus[k, l] - base_gradients[k, l]) / delta
        
        # Make Hessian symmetric
        for i in range(3*n_atoms):
            for j in range(i):
                hessian[j, i] = hessian[i, j]
        
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
    
    def d4_dispersion_hessian(self, coord, element_list, bond_mat):
        """
        Calculate Hessian correction based on D4 dispersion forces
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            bond_mat: bond connectivity matrix
        """
        n_atoms = len(coord)
        
        # Estimate atomic charges
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate pairwise D4 dispersion correction for all atom pairs
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
                
                # Calculate D4 Hessian contribution (pairwise)
                hess_block = self.d4_hessian_contribution(r_vec, r_ij, element_list[i], element_list[j], charges[i], charges[j])
                
                # Add to the Hessian matrix
                for n in range(3):
                    for m in range(3):
                        self.cart_hess[3*i+n, 3*i+m] += hess_block[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_block[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_block[n, m]
                        self.cart_hess[3*j+n, 3*i+m] -= hess_block[n, m]
        
        # Calculate three-body D4 contribution
        # Note: This can be computationally expensive for large systems
        # For production use, this could be made optional or use additional cutoffs
        three_body_hessian = self.d4_three_body_hessian(coord, element_list, charges)
        self.cart_hess += three_body_hessian
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian combining Fischer model and D4 dispersion correction
        
        Parameters:
            coord: Atomic coordinates (N×3 array, Bohr)
            element_list: List of element symbols
            cart_gradient: Gradient in Cartesian coordinates
            
        Returns:
            hess_proj: Hessian with rotational and translational modes projected out
        """
        print("Generating Hessian using Fischer model with D4 dispersion correction...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate bond connectivity matrix
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Calculate Hessian components from Fischer model
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        # Calculate Hessian components from D4 dispersion correction
        self.d4_dispersion_hessian(coord, element_list, bond_mat)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj
    