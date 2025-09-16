import numpy as np
import itertools

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib, UFF_VDW_distance_lib, D4Parameters, triple_covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UFF_VDW_distance_lib, number_element
from multioptpy.Coordinate.redundant_coordinate import RedundantInternalCoordinates
from multioptpy.Utils.bond_connectivity import BondConnectivity

class SchlegelD4ApproxHessian:
    def __init__(self):
        """
        Schlegel's approximate Hessian with D4 dispersion corrections and special handling for cyano groups
        References:
        - Schlegel: Journal of Molecular Structure: THEOCHEM Volumes 398–399, 30 June 1997, Pages 55-61
        - D4: E. Caldeweyher, C. Bannwarth, S. Grimme, J. Chem. Phys., 2017, 147, 034112
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # D4 dispersion parameters
        self.d4_params = D4Parameters()
        
        # Cyano group parameters - enhanced force constants
        self.cn_stretch_factor = 2.0   # Enhance stretch force constants for C≡N triple bond
        self.cn_angle_factor = 1.5     # Enhance angle force constants involving C≡N
        self.cn_torsion_factor = 0.5   # Reduce torsion force constants involving C≡N (more flexible)
        
    def detect_cyano_groups(self, coord, element_list):
        """Detect C≡N triple bonds in the structure"""
        cyano_atoms = []  # List of (C_idx, N_idx) tuples
        
        for i in range(len(coord)):
            if element_list[i] != 'C':
                continue
                
            for j in range(len(coord)):
                if i == j or element_list[j] != 'N':
                    continue
                    
                # Calculate distance between C and N
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Check if distance is close to a triple bond length
                cn_triple_bond = triple_covalent_radii_lib('C') + triple_covalent_radii_lib('N')
                
                if abs(r_ij - cn_triple_bond) < 0.3:  # Within 0.3 bohr of ideal length
                    # Check if C is connected to only one other atom (besides N)
                    connections_to_c = 0
                    for k in range(len(coord)):
                        if k == i or k == j:
                            continue
                            
                        r_ik = np.linalg.norm(coord[i] - coord[k])
                        cov_dist = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        if r_ik < 1.3 * cov_dist:  # Using 1.3 as a factor to account for bond length variations
                            connections_to_c += 1
                    
                    # If C has only one other connection, it's likely a terminal cyano group
                    if connections_to_c <= 1:
                        cyano_atoms.append((i, j))
        
        return cyano_atoms
    
    def calculate_coordination_numbers(self, coord, element_list):
        """Calculate atomic coordination numbers for D4 scaling"""
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                # Calculate distance
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Get covalent radii
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                
                # Coordination number contribution using counting function
                # k1 = 16.0, k2 = 4.0/3.0 (standard values from DFT-D4)
                k1 = 16.0
                k2 = 4.0/3.0
                r0 = r_cov_i + r_cov_j
                
                # Avoid overflow in exp
                if k1 * (r_ij / r0 - 1.0) > 25.0:
                    continue
                
                cn_contrib = 1.0 / (1.0 + np.exp(-k1 * (k2 * r0 / r_ij - 1.0)))
                cn[i] += cn_contrib
        
        return cn
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic charges using electronegativity equalization
        Simplified version for Hessian generation
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Calculate reference electronegativities
        en_list = [self.d4_params.get_electronegativity(elem) for elem in element_list]
        
        # Simple charge estimation based on electronegativity differences
        # This is a very simplified model
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coord[i] - coord[j])
                r_cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Only consider bonded atoms
                if r_ij < 1.3 * r_cov_sum:
                    en_diff = en_list[j] - en_list[i]
                    charge_transfer = 0.1 * en_diff  # Simple approximation
                    charges[i] += charge_transfer
                    charges[j] -= charge_transfer
        
        return charges
    
    def return_schlegel_const(self, element_1, element_2):
        """Return Schlegel's constant for a given element pair"""
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        parameter_B_matrix = [
            [0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
            [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
            [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
            [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
            [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
            [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
            [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110]
        ]  # Bohr
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        third_period_table = ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
        fourth_period_table = ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br","Kr"]
        fifth_period_table = ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc","Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te","I", "Xe"]
        sixth_period_table = ["Cs", "Ba", "La","Ce","Pr","Nd","Pm","Sm", "Eu", "Gd", "Tb", "Dy" ,"Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
        
        if element_1 in first_period_table:
            idx_1 = 0
        elif element_1 in second_period_table:
            idx_1 = 1
        elif element_1 in third_period_table:
            idx_1 = 2
        elif element_1 in fourth_period_table:
            idx_1 = 3
        elif element_1 in fifth_period_table:
            idx_1 = 4
        elif element_1 in sixth_period_table:
            idx_1 = 5
        else:
            idx_1 = 6
        
        if element_2 in first_period_table:
            idx_2 = 0
        elif element_2 in second_period_table:
            idx_2 = 1
        elif element_2 in third_period_table:
            idx_2 = 2
        elif element_2 in fourth_period_table:
            idx_2 = 3
        elif element_2 in fifth_period_table:
            idx_2 = 4
        elif element_2 in sixth_period_table:
            idx_2 = 5
        else:
            idx_2 = 6
        
        const_b = parameter_B_matrix[idx_1][idx_2]
        return const_b
    
    def d4_damping_function(self, r_ij, r0, order=6):
        """D4 rational damping function"""
        a1 = self.d4_params.a1
        a2 = self.d4_params.a2
        
        if order == 6:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        elif order == 8:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a2 * r0)) ** a1)
        return 0.0
    
    def charge_scale_factor(self, charge, element):
        """Calculate charge scaling factor for D4"""
        ga = self.d4_params.ga  # D4 charge scaling parameter (default=3.0)
        q_ref = self.d4_params.get_electronegativity(element)
        
        # Prevent numerical issues with large exponents
        exp_arg = -ga * abs(charge)
        if exp_arg < -50.0:  # Avoid underflow
            return 0.0
            
        return np.exp(exp_arg)
    
    def get_d4_parameters(self, elem1, elem2, q1=0.0, q2=0.0, cn1=None, cn2=None):
        """Get D4 parameters for a pair of elements with charge scaling"""
        # Get polarizabilities
        alpha1 = self.d4_params.get_polarizability(elem1)
        alpha2 = self.d4_params.get_polarizability(elem2)
        
        # Charge scaling
        qscale1 = self.charge_scale_factor(q1, elem1)
        qscale2 = self.charge_scale_factor(q2, elem2)
        
        # Get R4/R2 values
        r4r2_1 = self.d4_params.get_r4r2(elem1)
        r4r2_2 = self.d4_params.get_r4r2(elem2)
        
        # C6 coefficients with charge scaling
        c6_1 = alpha1 * r4r2_1 * qscale1
        c6_2 = alpha2 * r4r2_2 * qscale2
        c6_param = 2.0 * c6_1 * c6_2 / (c6_1 + c6_2)  # Effective C6 using harmonic mean
        
        # C8 coefficients
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (combined vdW radii)
        r0_param = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6_param, c8_param, r0_param
    
    def calc_d4_correction(self, r_ij, elem1, elem2, q1=0.0, q2=0.0, cn1=None, cn2=None):
        """Calculate D4 dispersion correction to the force constant"""
        # Get D4 parameters with charge scaling
        c6_param, c8_param, r0_param = self.get_d4_parameters(
            elem1, elem2, q1=q1, q2=q2, cn1=cn1, cn2=cn2
        )
        
        # Damping functions
        damp6 = self.d4_damping_function(r_ij, r0_param, order=6)
        damp8 = self.d4_damping_function(r_ij, r0_param, order=8)
        
        # D4 energy contribution 
        s6 = self.d4_params.s6
        s8 = self.d4_params.s8
        e_disp = -s6 * c6_param / r_ij**6 * damp6 - s8 * c8_param / r_ij**8 * damp8
        
        # Approximate second derivative (force constant)
        fc_disp = s6 * c6_param * (42.0 / r_ij**8) * damp6 + s8 * c8_param * (72.0 / r_ij**10) * damp8
        
        return fc_disp * 0.01  # Scale factor to match overall Hessian scale
    
    def calculate_three_body_term(self, coord, element_list, charges, cn):
        """Calculate three-body dispersion contribution"""
        n_atoms = len(coord)
        s9 = self.d4_params.s9
        
        # Skip if three-body term is turned off
        if abs(s9) < 1e-12:
            return np.zeros((3 * n_atoms, 3 * n_atoms))
        
        # Initialize three-body Hessian contribution
        three_body_hess = np.zeros((3 * n_atoms, 3 * n_atoms))
        
        # Loop over all atom triplets
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                for k in range(j+1, n_atoms):
                    # Get positions
                    r_i = coord[i]
                    r_j = coord[j]
                    r_k = coord[k]
                    
                    # Calculate interatomic distances
                    r_ij = np.linalg.norm(r_i - r_j)
                    r_jk = np.linalg.norm(r_j - r_k)
                    r_ki = np.linalg.norm(r_k - r_i)
                    
                    # Get coordination-number scaled C6 coefficients
                    c6_ij, _, r0_ij = self.get_d4_parameters(
                        element_list[i], element_list[j], 
                        q1=charges[i], q2=charges[j], 
                        cn1=cn[i], cn2=cn[j]
                    )
                    
                    c6_jk, _, r0_jk = self.get_d4_parameters(
                        element_list[j], element_list[k], 
                        q1=charges[j], q2=charges[k], 
                        cn1=cn[j], cn2=cn[k]
                    )
                    
                    c6_ki, _, r0_ki = self.get_d4_parameters(
                        element_list[k], element_list[i], 
                        q1=charges[k], q2=charges[i], 
                        cn1=cn[k], cn2=cn[i]
                    )
                    
                    # Calculate geometric mean of C6 coefficients
                    c9 = np.cbrt(c6_ij * c6_jk * c6_ki)
                    
                    # Calculate damping
                    damp_ij = self.d4_damping_function(r_ij, r0_ij)
                    damp_jk = self.d4_damping_function(r_jk, r0_jk)
                    damp_ki = self.d4_damping_function(r_ki, r0_ki)
                    damp = damp_ij * damp_jk * damp_ki
                    
                    # Skip if damping is too small
                    if damp < 1e-8:
                        continue
                    
                    # Calculate angle factor
                    r_ij_vec = r_j - r_i
                    r_jk_vec = r_k - r_j
                    r_ki_vec = r_i - r_k
                    
                    cos_ijk = np.dot(r_ij_vec, r_jk_vec) / (r_ij * r_jk)
                    cos_jki = np.dot(r_jk_vec, -r_ki_vec) / (r_jk * r_ki)
                    cos_kij = np.dot(-r_ki_vec, r_ij_vec) / (r_ki * r_ij)
                    
                    angle_factor = 1.0 + 3.0 * cos_ijk * cos_jki * cos_kij
                    
                    # Calculate three-body energy term
                    e_3 = -s9 * angle_factor * c9 * damp / (r_ij * r_jk * r_ki) ** 3
                    
                    # Approximate Hessian contribution (simplified)
                    # We use a small scaling factor to avoid dominating the Hessian
                    fc_scale = 0.002 * s9 * angle_factor * c9 * damp
                    
                    # Add approximate three-body contributions to Hessian
                    for n in range(3):
                        for m in range(3):
                            # Properly define all indices
                            idx_i_n = i * 3 + n
                            idx_j_n = j * 3 + n
                            idx_k_n = k * 3 + n
                            
                            idx_i_m = i * 3 + m
                            idx_j_m = j * 3 + m
                            idx_k_m = k * 3 + m
                            
                            # Diagonal blocks (diagonal atoms)
                            if n == m:
                                three_body_hess[idx_i_n, idx_i_n] += fc_scale / (r_ij**6) + fc_scale / (r_ki**6)
                                three_body_hess[idx_j_n, idx_j_n] += fc_scale / (r_ij**6) + fc_scale / (r_jk**6)
                                three_body_hess[idx_k_n, idx_k_n] += fc_scale / (r_jk**6) + fc_scale / (r_ki**6)
                            
                            # Off-diagonal blocks (between atoms)
                            three_body_hess[idx_i_n, idx_j_m] -= fc_scale / (r_ij**6)
                            three_body_hess[idx_j_m, idx_i_n] -= fc_scale / (r_ij**6)
                            
                            three_body_hess[idx_j_n, idx_k_m] -= fc_scale / (r_jk**6)
                            three_body_hess[idx_k_m, idx_j_n] -= fc_scale / (r_jk**6)
                            
                            three_body_hess[idx_k_n, idx_i_m] -= fc_scale / (r_ki**6)
                            three_body_hess[idx_i_m, idx_k_n] -= fc_scale / (r_ki**6)
        
        return three_body_hess
    
    def guess_schlegel_hessian(self, coord, element_list, charges, cn):
        """
        Calculate approximate Hessian using Schlegel's approach augmented with D4 dispersion
        and special handling for cyano groups
        """
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        # Setup connectivity tables using BondConnectivity utility
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        connectivity_table = [BC.bond_connect_table(b_c_mat), 
                              BC.angle_connect_table(b_c_mat), 
                              BC.dihedral_angle_connect_table(b_c_mat)]
        
        # Initialize RIC index list for all atom pairs
        RIC_idx_list = [[i[0], i[1]] for i in itertools.combinations(range(len(coord)), 2)]
        self.RIC_variable_num = len(RIC_idx_list)
        RIC_approx_diag_hessian = [0.0] * self.RIC_variable_num
        
        # Process connectivity table to build Hessian
        for idx_list in connectivity_table:
            for idx in idx_list:
                # Bond stretching terms
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    distance = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    const_b = self.return_schlegel_const(elem_1, elem_2)
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    
                    # Base Schlegel force constant
                    F = 1.734 / (distance - const_b) ** 3
                    
                    # Check if this is a cyano bond
                    is_cyano_bond = False
                    for c_idx, n_idx in cyano_atoms:
                        if (idx[0] == c_idx and idx[1] == n_idx) or (idx[0] == n_idx and idx[1] == c_idx):
                            is_cyano_bond = True
                            break
                    
                    # Add D4 dispersion contribution with charge scaling
                    d4_correction = self.calc_d4_correction(
                        distance, elem_1, elem_2, 
                        q1=charges[idx[0]], q2=charges[idx[1]],
                        cn1=cn[idx[0]], cn2=cn[idx[1]]
                    )
                    
                    if is_cyano_bond:
                        # Enhanced force constant for C≡N triple bond
                        RIC_approx_diag_hessian[tmpnum] += self.cn_stretch_factor * F + d4_correction
                    else:
                        RIC_approx_diag_hessian[tmpnum] += F + d4_correction
                
                # Angle bending terms
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    
                    # Check if angle involves cyano group
                    is_cyano_angle = (idx[0] in cyano_set or idx[1] in cyano_set or idx[2] in cyano_set)
                    
                    # Base Schlegel force constant
                    if elem_1 == "H" or elem_3 == "H":
                        F_angle = 0.160
                    else:
                        F_angle = 0.250
                    
                    # Add D4 dispersion contribution with charge scaling
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    
                    d4_correction_1 = self.calc_d4_correction(
                        d3_r1, elem_1, elem_2, 
                        q1=charges[idx[0]], q2=charges[idx[1]],
                        cn1=cn[idx[0]], cn2=cn[idx[1]]
                    ) * 0.2
                    
                    d4_correction_2 = self.calc_d4_correction(
                        d3_r2, elem_2, elem_3, 
                        q1=charges[idx[1]], q2=charges[idx[2]],
                        cn1=cn[idx[1]], cn2=cn[idx[2]]
                    ) * 0.2
                    
                    if is_cyano_angle:
                        # Enhanced angle force constants for angles involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_angle_factor * F_angle + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_angle_factor * F_angle + d4_correction_2
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_angle + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_angle + d4_correction_2
                
                # Torsion (dihedral) terms
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    elem_4 = element_list[idx[3]]
                    distance = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    bond_length = covalent_radii_lib(elem_2) + covalent_radii_lib(elem_3)
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    
                    # Base Schlegel torsion force constant
                    F_torsion = 0.0023 - 0.07 * (distance - bond_length)
                    
                    # Check if torsion involves cyano group
                    is_cyano_torsion = (idx[0] in cyano_set or idx[1] in cyano_set or 
                                        idx[2] in cyano_set or idx[3] in cyano_set)
                    
                    # Add D4 dispersion contribution with charge scaling
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    d3_r3 = np.linalg.norm(coord[idx[2]] - coord[idx[3]])
                    
                    d4_correction_1 = self.calc_d4_correction(
                        d3_r1, elem_1, elem_2, 
                        q1=charges[idx[0]], q2=charges[idx[1]],
                        cn1=cn[idx[0]], cn2=cn[idx[1]]
                    ) * 0.05
                    
                    d4_correction_2 = self.calc_d4_correction(
                        d3_r2, elem_2, elem_3, 
                        q1=charges[idx[1]], q2=charges[idx[2]],
                        cn1=cn[idx[1]], cn2=cn[idx[2]]
                    ) * 0.05
                    
                    d4_correction_3 = self.calc_d4_correction(
                        d3_r3, elem_3, elem_4, 
                        q1=charges[idx[2]], q2=charges[idx[3]],
                        cn1=cn[idx[2]], cn2=cn[idx[3]]
                    ) * 0.05
                    
                    if is_cyano_torsion:
                        # Reduced torsion force constants for torsions involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_torsion_factor * F_torsion + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_torsion_factor * F_torsion + d4_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += self.cn_torsion_factor * F_torsion + d4_correction_3
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_torsion + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_torsion + d4_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += F_torsion + d4_correction_3
        
        # Convert to numpy array
        RIC_approx_hessian = np.diag(RIC_approx_diag_hessian).astype("float64")
        return RIC_approx_hessian
    
    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian"""
        print("Generating Schlegel's approximate Hessian with D4 dispersion correction...")
        
        # Calculate coordination numbers and atomic charges for D4
        cn = self.calculate_coordination_numbers(coord, element_list)
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate B matrix for redundant internal coordinates
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        # Calculate approximate Hessian in internal coordinates
        int_approx_hess = self.guess_schlegel_hessian(coord, element_list, charges, cn)
        
        # Convert to Cartesian coordinates
        cart_hess = np.dot(b_mat.T, np.dot(int_approx_hess, b_mat))
        
        # Add three-body contribution (specific to D4)
        three_body_hess = self.calculate_three_body_term(coord, element_list, charges, cn)
        cart_hess += three_body_hess
        
        # Handle NaN values
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        
        # Ensure Hessian is symmetric
        n = len(coord) * 3
        for i in range(n):
            for j in range(i):
                avg = (cart_hess[i, j] + cart_hess[j, i]) / 2
                cart_hess[i, j] = avg
                cart_hess[j, i] = avg
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        
        print("D4 dispersion correction applied successfully")
        return hess_proj
    