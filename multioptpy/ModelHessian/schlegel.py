import numpy as np
import itertools

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import number_element
from multioptpy.Coordinate.redundant_coordinate import RedundantInternalCoordinates
from multioptpy.Utils.bond_connectivity import BondConnectivity


class SchlegelApproxHessian:
    def __init__(self):
        #Lindh's approximate hessian
        #ref: Journal of Molecular Structure: THEOCHEM Volumes 398–399, 30 June 1997, Pages 55-61
        #ref: Theoret. Chim. Acta (Berl.) 66, 333–340 (1984)
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
    
    def return_schlegel_const(self, element_1, element_2):
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        parameter_B_matrix = [[0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
                              [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
                              [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
                              [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
                              [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
                              [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
                              [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110]]# Bohr
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        third_period_table = ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br","Kr"]
        fourth_period_table = ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc","Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te","I", "Xe"]
        fifth_period_table = ["Cs", "Ba", "La","Ce","Pr","Nd","Pm","Sm", "Eu", "Gd", "Tb", "Dy" ,"Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
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
        else:
            idx_1 = 5 
        
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
        else:
            idx_2 = 5 
        
        
        const_b = parameter_B_matrix[idx_1][idx_2]
        
        return const_b
    
    
    def guess_schlegel_hessian(self, coord, element_list):
        #coord: cartecian coord, Bohr (atom num × 3)
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        val_num = len(coord)*3
        connectivity_table = [BC.bond_connect_table(b_c_mat), BC.angle_connect_table(b_c_mat), BC.dihedral_angle_connect_table(b_c_mat)]
        #RIC_approx_diag_hessian = []
        RIC_approx_diag_hessian = [0.0 for i in range(self.RIC_variable_num)]
        RIC_idx_list = [[i[0], i[1]] for i in itertools.combinations([j for j in range(len(coord))] , 2)]
        
        for idx_list in connectivity_table:
            for idx in idx_list:
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    distance = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    const_b = self.return_schlegel_const(elem_1, elem_2)   
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    F = 1.734 / (distance - const_b) ** 3
                    RIC_approx_diag_hessian[tmpnum] += F
                    
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    if elem_1 == "H" or elem_3 == "H":
                        RIC_approx_diag_hessian[tmpnum_1] += 0.160
                        RIC_approx_diag_hessian[tmpnum_2] += 0.160
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += 0.250
                        RIC_approx_diag_hessian[tmpnum_2] += 0.250
                                 
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    distance = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    bond_length = covalent_radii_lib(elem_2) + covalent_radii_lib(elem_3)
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    RIC_approx_diag_hessian[tmpnum_1] += 0.0023 -1* 0.07 * (distance - bond_length)
                    RIC_approx_diag_hessian[tmpnum_2] += 0.0023 -1* 0.07 * (distance - bond_length)
                    RIC_approx_diag_hessian[tmpnum_3] += 0.0023 -1* 0.07 * (distance - bond_length)

        
        RIC_approx_hessian = np.array(np.diag(RIC_approx_diag_hessian), dtype="float64")        
        return RIC_approx_hessian
    
    
    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        print("generating Schlegel's approximate hessian...")
        
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        
        int_approx_hess = self.guess_schlegel_hessian(coord, element_list)
        cart_hess = np.dot(b_mat.T, np.dot(int_approx_hess, b_mat))
        
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        #eigenvalue, _ = np.linalg.eig(cart_hess)
        #print(sorted(eigenvalue))
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        return hess_proj#cart_hess
