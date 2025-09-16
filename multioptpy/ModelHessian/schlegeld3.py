import numpy as np
import itertools

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib, UFF_VDW_distance_lib, D3Parameters, triple_covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UFF_VDW_distance_lib, number_element
from multioptpy.Coordinate.redundant_coordinate import RedundantInternalCoordinates
from multioptpy.Utils.bond_connectivity import BondConnectivity


class SchlegelD3ApproxHessian:
    def __init__(self):
        """
        Schlegel's approximate Hessian with D3 dispersion corrections and special handling for cyano groups
        References:
        - Schlegel: Journal of Molecular Structure: THEOCHEM Volumes 398–399, 30 June 1997, Pages 55-61
        - D3: S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys., 2010, 132, 154104
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # D3 dispersion parameters
        self.d3_params = D3Parameters()
        
        # Cyano group parameters - enhanced force constants
        self.cn_stretch_factor = 2.0  # Enhance stretch force constants for C≡N triple bond
        self.cn_angle_factor = 1.5    # Enhance angle force constants involving C≡N
        self.cn_torsion_factor = 0.5  # Reduce torsion force constants involving C≡N (more flexible)
        
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
    
    def d3_damping_function(self, r_ij, r0):
        """Calculate D3 rational damping function"""
        a1 = self.d3_params.a1
        a2 = self.d3_params.a2
        
        # Rational damping function for C6 term
        damp = 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        return damp
    
    def get_d3_parameters(self, elem1, elem2):
        """Get D3 parameters for a pair of elements"""
        # Get R4/R2 values
        r4r2_1 = self.d3_params.get_r4r2(elem1)
        r4r2_2 = self.d3_params.get_r4r2(elem2)
        
        # C6 coefficients
        c6_1 = r4r2_1 ** 2
        c6_2 = r4r2_2 ** 2
        c6 = np.sqrt(c6_1 * c6_2)
        
        # C8 coefficients
        c8 = 3.0 * c6 * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (vdW radii)
        r0 = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6, c8, r0
    
    def calc_d3_correction(self, r_ij, elem1, elem2):
        """Calculate D3 dispersion correction to the force constant"""
        # Get D3 parameters
        c6, c8, r0 = self.get_d3_parameters(elem1, elem2)
        
        # Damping functions
        damp6 = self.d3_damping_function(r_ij, r0)
        
        # D3 energy contribution (simplified)
        e_disp = -self.d3_params.s6 * c6 / r_ij**6 * damp6
        
        # Approximate second derivative (force constant)
        fc_disp = self.d3_params.s6 * c6 * (42.0 / r_ij**8) * damp6
        
        return fc_disp * 0.01  # Scale factor to match overall Hessian scale
    
    def guess_schlegel_hessian(self, coord, element_list):
        """
        Calculate approximate Hessian using Schlegel's approach augmented with D3 dispersion
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
                    
                    # Add D3 dispersion contribution
                    d3_correction = self.calc_d3_correction(distance, elem_1, elem_2)
                    
                    if is_cyano_bond:
                        # Enhanced force constant for C≡N triple bond
                        RIC_approx_diag_hessian[tmpnum] += self.cn_stretch_factor * F + d3_correction
                    else:
                        RIC_approx_diag_hessian[tmpnum] += F + d3_correction
                
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
                    
                    # Add D3 dispersion contribution
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    d3_correction_1 = self.calc_d3_correction(d3_r1, elem_1, elem_2) * 0.2
                    d3_correction_2 = self.calc_d3_correction(d3_r2, elem_2, elem_3) * 0.2
                    
                    if is_cyano_angle:
                        # Enhanced angle force constants for angles involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_angle_factor * F_angle + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_angle_factor * F_angle + d3_correction_2
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_angle + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_angle + d3_correction_2
                
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
                    
                    # Add D3 dispersion contribution
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    d3_r3 = np.linalg.norm(coord[idx[2]] - coord[idx[3]])
                    d3_correction_1 = self.calc_d3_correction(d3_r1, elem_1, elem_2) * 0.05
                    d3_correction_2 = self.calc_d3_correction(d3_r2, elem_2, elem_3) * 0.05
                    d3_correction_3 = self.calc_d3_correction(d3_r3, elem_3, elem_4) * 0.05
                    
                    if is_cyano_torsion:
                        # Reduced torsion force constants for torsions involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_torsion_factor * F_torsion + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_torsion_factor * F_torsion + d3_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += self.cn_torsion_factor * F_torsion + d3_correction_3
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_torsion + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_torsion + d3_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += F_torsion + d3_correction_3
        
        # Convert to numpy array
        RIC_approx_hessian = np.diag(RIC_approx_diag_hessian).astype("float64")
        return RIC_approx_hessian
    
    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian"""
        print("Generating Schlegel's approximate Hessian with D3 dispersion correction...")
        
        # Calculate B matrix for redundant internal coordinates
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        # Calculate approximate Hessian in internal coordinates
        int_approx_hess = self.guess_schlegel_hessian(coord, element_list)
        
        # Convert to Cartesian coordinates
        cart_hess = np.dot(b_mat.T, np.dot(int_approx_hess, b_mat))
        
        # Handle NaN values
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        
        return hess_proj

