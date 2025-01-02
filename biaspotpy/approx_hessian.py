import itertools

import numpy as np

from bond_connectivity import BondConnectivity
from parameter import UnitValueLib, number_element, element_number, covalent_radii_lib, UFF_effective_charge_lib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, atomic_mass, D2_VDW_radii_lib, D2_C6_coeff_lib, D2_S6_parameter, double_covalent_radii_lib, triple_covalent_radii_lib
from redundant_coordinations import RedundantInternalCoordinates
from calc_tools import Calculationtools


class LindhApproxHessian:
    def __init__(self):
        #Lindh's approximate hessian
        #Ref: https://doi.org/10.1016/0009-2614(95)00646-L
        #Lindh, R., Chemical Physics Letters 1995, 241 (4), 423–428.
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.force_const_list = [0.45, 0.15, 0.005]  #bond, angle, dihedral_angle
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        return
    def LJ_force_const(self, elem_1, elem_2, coord_1, coord_2):
        eps_1 = UFF_VDW_well_depth_lib(elem_1)
        eps_2 = UFF_VDW_well_depth_lib(elem_2)
        sigma_1 = UFF_VDW_distance_lib(elem_1)
        sigma_2 = UFF_VDW_distance_lib(elem_2)
        eps = np.sqrt(eps_1 * eps_2)
        sigma = np.sqrt(sigma_1 * sigma_2)
        distance = np.linalg.norm(coord_1 - coord_2)
        LJ_force_const = -12 * eps * (-7*(sigma ** 6 / distance ** 8) + 13*(sigma ** 12 / distance ** 14))
        
        return LJ_force_const
    
    def electrostatic_force_const(self, elem_1, elem_2, coord_1, coord_2):
        effective_elec_charge = UFF_effective_charge_lib(elem_1) * UFF_effective_charge_lib(elem_2)
        distance = np.linalg.norm(coord_1 - coord_2)
        
        ES_force_const = 664.12 * (effective_elec_charge / distance ** 3) * (self.bohr2angstroms ** 2 / self.hartree2kcalmol)
        
        return ES_force_const#atom unit
    
    
    
    def return_lindh_const(self, element_1, element_2):
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        const_R_list = [[1.35, 2.10, 2.53],
                        [2.10, 2.87, 3.40],
                        [2.53, 3.40, 3.40]]
        
        const_alpha_list = [[1.0000, 0.3949, 0.3949],
                            [0.3949, 0.2800, 0.2800],
                            [0.3949, 0.2800, 0.2800]]      
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        
        if element_1 in first_period_table:
            idx_1 = 0
        elif element_1 in second_period_table:
            idx_1 = 1
        else:
            idx_1 = 2 
        
        if element_2 in first_period_table:
            idx_2 = 0
        elif element_2 in second_period_table:
            idx_2 = 1
        else:
            idx_2 = 2    
        
        #const_R = const_R_list[idx_1][idx_2]
        const_R = covalent_radii_lib(element_1) + covalent_radii_lib(element_2)
        const_alpha = const_alpha_list[idx_1][idx_2]
        
        return const_R, const_alpha

    def guess_lindh_hessian(self, coord, element_list):
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
                force_const = self.force_const_list[len(idx)-2]
                for i in range(len(idx)-1):
                    elem_1 = element_list[idx[i]]
                    elem_2 = element_list[idx[i+1]]
                    const_R, const_alpha = self.return_lindh_const(elem_1, elem_2)
                    
                    R = np.linalg.norm(coord[idx[i]] - coord[idx[i+1]])
                    force_const *= np.exp(const_alpha * (const_R**2 - R**2)) 
                
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    mass_1 = atomic_mass(element_list[tmp_idx[0]]) 
                    mass_2 = atomic_mass(element_list[tmp_idx[1]])
                    
                    reduced_mass = (mass_1 * mass_2) / (mass_1 + mass_2)
                    
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    RIC_approx_diag_hessian[tmpnum] += force_const/reduced_mass
                  
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    RIC_approx_diag_hessian[tmpnum_1] += force_const
                    RIC_approx_diag_hessian[tmpnum_2] += force_const
                    
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    RIC_approx_diag_hessian[tmpnum_1] += force_const
                    RIC_approx_diag_hessian[tmpnum_2] += force_const
                    RIC_approx_diag_hessian[tmpnum_3] += force_const
                
                else:
                    print("error")
                    raise
                
        for num, pair in enumerate(RIC_idx_list):
            if pair in connectivity_table[0]:#bond connectivity
                continue#non bonding interaction
            RIC_approx_diag_hessian[num] += self.LJ_force_const(element_list[pair[0]], element_list[pair[1]], coord[pair[0]], coord[pair[1]])
            RIC_approx_diag_hessian[num] += self.electrostatic_force_const(element_list[pair[0]], element_list[pair[1]], coord[pair[0]], coord[pair[1]])
            
       
        
        RIC_approx_hessian = np.array(np.diag(RIC_approx_diag_hessian), dtype="float64")
        
        return RIC_approx_hessian

    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        
        print("generating Lindh's approximate hessian...")
        cart_gradient = cart_gradient.reshape(3*(len(cart_gradient)), 1)
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        int_grad = RedundantInternalCoordinates().cartgrad2RICgrad(cart_gradient, b_mat)
        int_approx_hess = self.guess_lindh_hessian(coord, element_list)
        BC = BondConnectivity()
        
        connnectivity = BC.connectivity_table(coord, element_list)
        #print(connnectivity, len(connnectivity[0])+len(connnectivity[1])+len(connnectivity[2]))
        cart_hess = RedundantInternalCoordinates().RIChess2carthess(coord, connnectivity, 
                                                                    int_approx_hess, b_mat, int_grad)
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        #eigenvalue, _ = np.linalg.eig(cart_hess)
        #print(sorted(eigenvalue))
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        return hess_proj#cart_hess
    

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

def calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij):
   
    x_ij = x_ij + 1.0e-12
    y_ij = y_ij + 1.0e-12
    z_ij = z_ij + 1.0e-12
    a_vdw = 20.0
    t_1 = D2_S6_parameter * C6_param_ij
    t_2 = x_ij ** 2 
    t_3 = y_ij ** 2 
    t_4 = z_ij ** 2
    t_5 = t_2 + t_3 + t_4
    t_6 = t_5 ** 2
    t_7 = t_6 ** 2 
    t_10 = t_5 ** 0.5
    t_11 = 0.1 / C6_VDW_ij
   
    t_15 = np.exp(-1 * a_vdw * (t_10 * t_11 - 0.1)) + 1.0e-12
    t_16 = 0.1 + t_15
    t_17 = 0.1 / t_16
    t_24 = t_16 ** 2
    t_25 = 0.1 / t_24
    t_29 = t_11 * t_15
    t_33 = 0.1 / t_7
    t_41 = a_vdw ** 2
    t_42 = C6_VDW_ij ** 2
    t_44 = t_41 / t_42
    t_45 = t_15 ** 2
    #print(t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_10, t_11, t_15, t_16, t_17, t_24, t_25, t_29, t_33, t_41, t_42, t_44, t_45)
    t_62 = -0.48 * t_1 / t_7 / t_5 * t_17 * t_2 + 0.13 * t_1 / t_10 / t_7 *  t_25 * t_2 * a_vdw * t_29 + 0.6 * t_1 * t_33 * t_17 - 0.2 * t_1 * t_33 / t_24 / t_16 * t_44 * t_2 * t_45 - t_1 / t_10 / t_6 / t_5 * t_25 * a_vdw * t_29 + t_1 * t_33 * t_25 * t_44 * t_2 * t_15
    
    return t_62

def calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij):
    a_vdw = 20.0
    x_ij = x_ij + 1.0e-12
    y_ij = y_ij + 1.0e-12
    z_ij = z_ij + 1.0e-12
    
    t_1 = D2_S6_parameter * C6_param_ij
    t_2 = x_ij ** 2
    t_3 = y_ij ** 2 
    t_4 = z_ij ** 2 
    t_5 = t_2 + t_3 + t_4
    t_6 = t_5 ** 2
    t_7 = t_6 ** 2
    t_11 = t_5 ** 0.5
    t_12 = 0.1 / C6_VDW_ij
   
    t_16 = np.exp(-1 * a_vdw * (t_11 * t_12 - 0.1)) + 1.0e-12
    t_17 = 0.1 + t_16 
    t_25 = t_17 ** 2 
    t_26 = 0.1 / t_25
    t_35 = 0.1 / t_7
    t_40 = a_vdw ** 2
    t_41 = C6_VDW_ij ** 2
    t_43 = t_40 / t_41
    t_44 = t_16 ** 2
    #print(t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_11, t_12, t_16, t_17, t_25, t_26, t_35, t_40, t_41, t_43, t_44)
    t_56 = -0.48 * t_1 / t_7 / t_5 / t_17 * x_ij * y_ij + 0.13 * t_1 / t_11 / t_7 * t_26 * x_ij * a_vdw * t_12 * y_ij * t_16 - 0.2 * t_1 * t_35 / t_25 / t_17 * t_43 * x_ij * t_44 * y_ij + t_1 * t_35 * t_26 * t_43 * x_ij * y_ij * t_16
    #print(t_16, t_56)
    return t_56
    
    
def outofplane2(t_xyz):
    r_1 = t_xyz[0] - t_xyz[3]
    q_41 = np.linalg.norm(r_1)
    e_41 = r_1 / q_41

    r_2 = t_xyz[1] - t_xyz[3]
    q_42 = np.linalg.norm(r_2)
    e_42 = r_2 / q_42

    r_3 = t_xyz[2] - t_xyz[3]
    q_43 = np.linalg.norm(r_3)
    e_43 = r_3 / q_43

    cosfi1 = np.dot(e_43, e_42)

    fi_1 = np.arccos(cosfi1)

    if abs(fi_1 - np.pi) < 1e-12:
        teta = 0.0
        bt = 0.0
        return teta, bt

    cosfi2 = np.dot(e_41, e_43)
    fi_2 = np.arccos(cosfi2)
    #deg_fi_2 = 180.0 * fi_2 / np.pi

    cosfi3 = np.dot(e_41, e_42)

    fi_3 = np.arccos(cosfi3)
    #deg_fi_3 = 180.0 * fi_3 / np.pi

    c14 = np.zeros((3, 3))

    c14[0] = t_xyz[0]
    c14[1] = t_xyz[3]

    r_42 = t_xyz[1] - t_xyz[3]
    r_43 = t_xyz[2] - t_xyz[3]
    c14[2][0] = r_42[1] * r_43[2] - r_42[2] * r_43[1]
    c14[2][1] = r_42[2] * r_43[0] - r_42[0] * r_43[2]
    c14[2][2] = r_42[0] * r_43[1] - r_42[1] * r_43[0]

    if ((c14[2][0] ** 2 + c14[2][1] ** 2 + c14[2][2] ** 2) < 1e-12):
        teta = 0.0
        bt = 0.0
        return teta, bt
    c14[2][0] = c14[2][0] + t_xyz[3][0]
    c14[2][1] = c14[2][1] + t_xyz[3][1]
    c14[2][2] = c14[2][2] + t_xyz[3][2]

    teta, br_14 = bend2(c14)

    teta = teta -0.5 * np.pi

    bt = np.zeros((4, 3))

    for i_x in range(1, 4):
        i_y = (i_x + 1) % 4 + (i_x + 1) // 4
        i_z = (i_y + 1) % 4 + (i_y + 1) // 4
        
        bt[0][i_x-1] = -1 * br_14[2][i_x-1]
        bt[1][i_x-1] = -1 * br_14[2][i_y-1]
        bt[2][i_x-1] = -1 * br_14[2][i_z-1]
        bt[3][i_x-1] = -1 * (bt[0][i_x-1] + bt[1][i_x-1] + bt[2][i_x-1])
        
    bt *= -1.0 

    return teta, bt# angle, move_vector

def torsion2(t_xyz):
    r_ij_1, b_r_ij = stretch2(t_xyz[0:2])
    r_ij_2, b_r_jk = stretch2(t_xyz[1:3])
    r_ij_3, b_r_kl = stretch2(t_xyz[2:4])

    fi_2, b_fi_2 = bend2(t_xyz[0:3])
    fi_3, b_fi_3 = bend2(t_xyz[1:4])
    sin_fi_2 = np.sin(fi_2)
    sin_fi_3 = np.sin(fi_3)
    cos_fi_2 = np.cos(fi_2)
    cos_fi_3 = np.cos(fi_3)
    costau = ( (b_r_ij[0][1] * b_r_jk[1][2] - b_r_jk[1][1] * b_r_ij[0][2])) * (b_r_jk[0][1] * b_r_kl[1][2] - b_r_jk[0][2] * b_r_kl[1][1]) + (b_r_ij[0][2] * b_r_jk[1][0] - b_r_ij[0][0] * b_r_jk[1][2]) * (b_r_jk[0][2] * b_r_kl[1][0] - b_r_jk[0][0] * b_r_kl[1][2]) + (b_r_ij[0][0] * b_r_jk[1][1] - b_r_ij[0][1] * b_r_jk[1][0]) * (b_r_jk[0][0] * b_r_kl[1][1] - b_r_jk[0][1] * b_r_kl[1][0]) / (sin_fi_2 * sin_fi_3)
    sintau = (b_r_ij[1][0] * (b_r_jk[0][1] * b_r_kl[1][2] - b_r_jk[0][2] * b_r_kl[1][1]) + b_r_ij[1][1] * (b_r_jk[0][2] * b_r_kl[1][0] - b_r_jk[0][0] * b_r_kl[1][2]) + b_r_ij[1][2] * (b_r_jk[0][0] * b_r_kl[1][1] - b_r_jk[0][1] * b_r_kl[1][0])) / (sin_fi_2 * sin_fi_3)

    tau = np.arctan2(sintau, costau)

    if abs(tau) == np.pi:
        tau = np.pi

    bt = np.zeros((4, 3))
    for i_x in range(1,4):
        i_y = i_x + 1
        if i_y > 3:
            i_y = i_y - 3
        i_z = i_y + 1
        if i_z > 3:
            i_z = i_z - 3
        bt[0][i_x-1] = (b_r_ij[1][i_y-1] * b_r_jk[1][i_z-1] - b_r_ij[1][i_z-1] * b_r_jk[1][i_y-1]) / (r_ij_1 * sin_fi_2 ** 2)
        bt[3][i_x-1] = (b_r_kl[0][i_y-1] * b_r_jk[0][i_z-1] - b_r_kl[0][i_z-1] * b_r_jk[0][i_y-1]) / (r_ij_3 * sin_fi_3 ** 2)
        bt[1][i_x-1] = -1 * ((r_ij_2 - r_ij_1 * cos_fi_2) * bt[0][i_x-1] + r_ij_3 * cos_fi_3 * bt[3][i_x-1]) / r_ij_2
        bt[2][i_x-1] = -1 * (bt[0][i_x-1] + bt[1][i_x-1] + bt[3][i_x-1])

    for ix in range(1, 4):
        iy = ix + 1
        if iy > 3:
            iy = iy - 3
        iz = iy + 1
        if iz > 3:
            iz = iz - 3
        bt[0][ix-1] = (b_r_ij[1][iy-1] * b_r_jk[1][iz-1] - b_r_ij[1][iz-1] * b_r_jk[1][iy-1]) / (r_ij_1 * sin_fi_2 ** 2)
        bt[3][ix-1] = (b_r_kl[0][iy-1] * b_r_jk[0][iz-1] - b_r_kl[0][iz-1] * b_r_jk[0][iy-1]) / (r_ij_3 * sin_fi_3 ** 2)
        bt[1][ix-1] = -1 * ((r_ij_2 - r_ij_1 * cos_fi_2) * bt[0][ix-1] + r_ij_3 * cos_fi_3 * bt[3][ix-1]) / r_ij_2
        bt[2][ix-1] = -1 * (bt[0][ix-1] + bt[1][ix-1] + bt[3][ix-1])

    return tau, bt # angle, move_vector

def bend2(t_xyz):
    r_ij_1, b_r_ij = stretch2(t_xyz[0:2])
    r_jk_1, b_r_jk = stretch2(t_xyz[1:3])
    c_o = 0.0
    crap = 0.0
    for i in range(3):
        c_o += b_r_ij[0][i] * b_r_jk[1][i]
        crap += b_r_ij[0][i] ** 2
        crap += b_r_jk[1][i] ** 2

    if np.sqrt(crap) < 1e-12:
        fir = np.pi - np.arcsin(np.sqrt(crap))
        si = np.sqrt(crap)
    else:
        fir = np.arccos(c_o)
        si = np.sqrt(1 - c_o ** 2)

    if np.abs(fir - np.pi) < 1e-12:
        fir = np.pi

    bf = np.zeros((3, 3))
    for i in range(3):
        bf[0][i] = (c_o * b_r_ij[0][i] - b_r_jk[1][i]) / (r_ij_1 * si)
        bf[2][i] = (c_o * b_r_jk[1][i] - b_r_ij[0][i]) / (r_jk_1 * si)
        bf[1][i] = -1 * (bf[0][i] + bf[2][i])

    return fir, bf # angle, move_vector

def stretch2(t_xyz):
    dist = t_xyz[0] - t_xyz[1]
    norm_dist = np.linalg.norm(dist)
    b = np.zeros((2,3))
    b[0] = -1 * dist / norm_dist
    b[1] = dist / norm_dist
    return norm_dist, b # distance, move_vectors (unit_vector)

    
    
    
class SwartD2ApproxHessian:
    def __init__(self):
        #Swart's Model Hessian augmented with D2
        #ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536–2544.
        #ref.: https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        
        self.kr = 0.35
        self.kf = 0.15
        self.kt = 0.005
        
        self.cutoff = 70.0
        self.eps = 1.0e-12
        
        #self.s6 = 20.0
        return
    
    def calc_force_const(self, alpha, covalent_length, distance):
        force_const = np.exp(-1 * alpha * (distance / covalent_length - 1.0))
        
        return force_const
        
    def calc_vdw_force_const(self, alpha, vdw_length, distance):
        vdw_force_const = np.exp(-1 * alpha * (vdw_length - distance) ** 2)
        return vdw_force_const
    
    def swart_bond(self, coord, element_list):
        for i in range(len(coord)):
            for j in range(i):
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                vdw_length = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                C6_param_i = D2_C6_coeff_lib(element_list[i])
                C6_param_j = D2_C6_coeff_lib(element_list[j])
                C6_param_ij = np.sqrt(C6_param_i * C6_param_j)
                C6_VDW_ij = D2_VDW_radii_lib(element_list[i]) + D2_VDW_radii_lib(element_list[j])
                VDW_xx = calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xy = calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xz = calc_vdw_anisotropic(x_ij, z_ij, y_ij,C6_param_ij, C6_VDW_ij)
                VDW_yy = calc_vdw_isotopic(y_ij, x_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_yz = calc_vdw_anisotropic(y_ij, z_ij, x_ij, C6_param_ij, C6_VDW_ij)
                VDW_zz = calc_vdw_isotopic(z_ij, x_ij, y_ij, C6_param_ij, C6_VDW_ij)
        
                g_mm = self.kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * self.calc_vdw_force_const(5.0, vdw_length, r_ij)
                
                hess_xx = g_mm * x_ij ** 2 / r_ij_2 - VDW_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - VDW_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - VDW_xz
                hess_yy = g_mm * y_ij ** 2 / r_ij_2 - VDW_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - VDW_yz
                hess_zz = g_mm * z_ij ** 2 / r_ij_2 - VDW_zz
                
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
       
        return 
    
    def swart_angle(self, coord, element_list):
        for i in range(len(coord)):
            for j in range(len(coord)):
                if i == j:
                    continue
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                
                for k in range(j):
                    if i == k:
                        continue
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                    
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    g_ij = self.calc_force_const(1.0, covalent_length_ij, r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, r_ij)
                    g_ik = self.calc_force_const(1.0, covalent_length_ik, r_ik) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ik, r_ik)
                    
                    g_jk = self.kf * g_ij * g_ik
                    
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:
                        cos_theta = (r_ij_2 + r_ik_2 - r_jk_2) / (2.0 * r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                       
                        if sin_theta > self.eps: # non-linear
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        # m = i, i = j, j = k
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                       
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        
                                     
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val

                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            for ii in range(2):
                                # m = i, i = j, j = k
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                for l in range(3):
                                    for m in range(3):#Under construction
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                for l in range(3):#Under construction
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass
                    
                
        return

    def swart_dihedral_angle(self, coord, element_list):
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
    
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_kl = coord[k] - coord[l]
                        vdw_length_kl = UFF_VDW_distance_lib(element_list[k]) + UFF_VDW_distance_lib(element_list[l])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        a35 = (35.0/180)* np.pi
                        cosfi_max=np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, norm_r_ij)
                        g_jk = self.calc_force_const(1.0, covalent_length_jk, norm_r_jk) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_jk, norm_r_jk)
                        g_kl = self.calc_force_const(1.0, covalent_length_kl, norm_r_kl) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_kl, norm_r_kl) 
                       
                        t_ij = self.kt * g_ij * g_jk * g_kl
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] = self.cart_hess[3 * i + n][3 * j + m] + t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] = self.cart_hess[3 * i + n][3 * k + m] + t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] = self.cart_hess[3 * i + n][3 * l + m] + t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] = self.cart_hess[3 * j + n][3 * k + m] + t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] = self.cart_hess[3 * j + n][3 * l + m] + t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] = self.cart_hess[3 * k + n][3 * l + m] + t_ij * s_k[n] * s_l[m]
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] = self.cart_hess[3 * i + n][3 * i + m] + t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] = self.cart_hess[3 * j + n][3 * j + m] + t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] = self.cart_hess[3 * k + n][3 * k + m] + t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] = self.cart_hess[3 * l + n][3 * l + m] + t_ij * s_l[n] * s_l[m]
                       
        return
       
    def swart_out_of_plane(self, coord, element_list):
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        vdw_length_il = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[l])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, norm_r_ij)
                        
                        g_ik = self.calc_force_const(1.0, covalent_length_ik, norm_r_ik) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ik, norm_r_ik)
                        
                        g_il = self.calc_force_const(1.0, covalent_length_il, norm_r_il) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_il, norm_r_il)
                        
                        
                        t_ij = self.kt * g_ij * g_ik * g_il #self.ko * g_ij * g_ik * g_il
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                          
        return



    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        print("generating Swart's approximate hessian...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        
        self.swart_bond(coord, element_list)
        self.swart_angle(coord, element_list)
        self.swart_dihedral_angle(coord, element_list)
        self.swart_out_of_plane(coord, element_list)
        
        for i in range(len(coord)*3):
            for j in range(len(coord)*3):
                
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj#cart_hess

class Lindh2007D2ApproxHessian:
    def __init__(self):
        #Lindh's Model Hessian (2007) augmented with D2
        #ref.: https://github.com/ss0832/xtb/blob/main/src/model_hessian.f90
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        self.bond_threshold_scale = 1.0
        self.kr = 0.45
        self.kf = 0.10
        self.kt = 0.0025
        self.ko = 0.16
        self.kd = 0.05
        self.cutoff = 50.0
        self.eps = 1.0e-12
        
        self.rAv = np.array([[1.3500,   2.1000,   2.5300],
                    [2.1000,   2.8700,   3.8000],
                    [2.5300,   3.8000,   4.5000]]) 
        
        self.aAv = np.array([[1.0000,   0.3949,   0.3949],
                    [0.3949,   0.2800,   0.1200],
                    [0.3949,   0.1200,   0.0600]]) 
        
        self.dAv = np.array([[0.0000,   3.6000,   3.6000],
                    [3.6000,   5.3000,   5.3000],
                    [3.6000,   5.3000,   5.3000]])
        
        #self.s6 = 20.0
        return
    
    def select_idx(self, elem_num):
        if type(elem_num) == str:
            elem_num = element_number(elem_num)

        if (elem_num > 0 and elem_num < 2):
            idx = 0
        elif (elem_num >= 2 and elem_num < 10):
            idx = 1
        elif (elem_num >= 10 and elem_num < 18):
            idx = 2
        elif (elem_num >= 18 and elem_num < 36):
            idx = 2
        elif (elem_num >= 36 and elem_num < 54):
            idx = 2
        elif (elem_num >= 54 and elem_num < 86):
            idx = 2
        elif (elem_num >= 86):
            idx = 2
        else:
            idx = 2

        return idx
    
    
    def calc_force_const(self, alpha, r_0, distance_2):
        force_const = np.exp(alpha * (r_0 ** 2 -1.0 * distance_2))
        
        return force_const
        
    def calc_vdw_force_const(self, alpha, vdw_length, distance):
        vdw_force_const = np.exp(-4 * alpha * (vdw_length - distance) ** 2)
        return vdw_force_const
    
    def lindh2007_bond(self, coord, element_list):
        for i in range(len(coord)):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(i):
                j_idx = self.select_idx(element_list[j])
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                r_0 = self.rAv[i_idx][j_idx]
                d_0 = self.dAv[i_idx][j_idx]
                alpha = self.aAv[i_idx][j_idx]
                
                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                
                #if self.bond_threshold_scale * triple_bond > r_ij:
                #    covalent_length = triple_bond
                #elif self.bond_threshold_scale * double_bond > r_ij:

                #    covalent_length = double_bond
                #else:
                covalent_length = single_bond
               
                vdw_length = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                C6_param_i = D2_C6_coeff_lib(element_list[i])
                C6_param_j = D2_C6_coeff_lib(element_list[j])
                C6_param_ij = np.sqrt(C6_param_i * C6_param_j)
                C6_VDW_ij = D2_VDW_radii_lib(element_list[i]) + D2_VDW_radii_lib(element_list[j])
                VDW_xx = calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xy = calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xz = calc_vdw_anisotropic(x_ij, z_ij, y_ij,C6_param_ij, C6_VDW_ij)
                VDW_yy = calc_vdw_isotopic(y_ij, x_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_yz = calc_vdw_anisotropic(y_ij, z_ij, x_ij, C6_param_ij, C6_VDW_ij)
                VDW_zz = calc_vdw_isotopic(z_ij, x_ij, y_ij, C6_param_ij, C6_VDW_ij)
        
                g_mm = self.kr * self.calc_force_const(alpha, covalent_length, r_ij_2) + self.kd * self.calc_vdw_force_const(4.0, vdw_length, r_ij)
                
                hess_xx = g_mm * x_ij ** 2 / r_ij_2 - VDW_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - VDW_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - VDW_xz
                hess_yy = g_mm * y_ij ** 2 / r_ij_2 - VDW_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - VDW_yz
                hess_zz = g_mm * z_ij ** 2 / r_ij_2 - VDW_zz
                
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
       
        return 
    
    def lindh2007_angle(self, coord, element_list):
        for i in range(len(coord)):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(len(coord)):
                if i == j:
                    continue
                j_idx = self.select_idx(element_list[j])
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)

                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                
                #if self.bond_threshold_scale * triple_bond > r_ij:
                #    covalent_length_ij = triple_bond
                #elif self.bond_threshold_scale * double_bond > r_ij:
                #    covalent_length_ij = double_bond
                #else:
                covalent_length_ij = single_bond

                vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                    
                r_ij_0 = self.rAv[i_idx][j_idx]
                d_ij_0 = self.dAv[i_idx][j_idx]
                alpha_ij = self.aAv[i_idx][j_idx]
                
                for k in range(j):
                    if i == k:
                        continue
                    k_idx = self.select_idx(element_list[k])
                    
                    r_ik_0 = self.rAv[i_idx][k_idx]
                    d_ik_0 = self.dAv[i_idx][k_idx]
                    alpha_ik = self.aAv[i_idx][k_idx]
                    
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)


                    single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[k])
                    triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[k])
                    
                    #if self.bond_threshold_scale * triple_bond > r_ik:
                    #    covalent_length_ik = triple_bond
                    #elif self.bond_threshold_scale * double_bond > r_ik:
                    #    covalent_length_ik = double_bond
                    #else:
                    covalent_length_ik = single_bond

                    #covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                    
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ij, r_ij)
                    g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ik, r_ik)
                    
                    g_jk = self.kf * (g_ij + 0.5 * self.kd / self.kr * d_ij_0) * (g_ik * 0.5 * self.kd / self.kr * d_ik_0)
                    
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:

                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                        cos_theta = dot_product_r_ij_r_ik / (r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        if sin_theta > self.eps: # non-linear
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        # m = i, i = j, j = k
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                       
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        
                                     
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val

                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            for ii in range(2):
                                # m = i, i = j, j = k
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                for l in range(3):
                                    for m in range(3):#Under construction
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                for l in range(3):#Under construction
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass
                    
                
        return

    def lindh2007_dihedral_angle(self, coord, element_list):
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
                        i_idx = self.select_idx(element_list[i])
                        j_idx = self.select_idx(element_list[j])
                        k_idx = self.select_idx(element_list[k])
                        l_idx = self.select_idx(element_list[l])
                        
                        r_ij_0 = self.rAv[i_idx][j_idx]
                        d_ij_0 = self.dAv[i_idx][j_idx]
                        alpha_ij = self.aAv[i_idx][j_idx]
                        
                        
                        r_jk_0 = self.rAv[j_idx][k_idx]
                        d_jk_0 = self.dAv[j_idx][k_idx]
                        alpha_jk = self.aAv[j_idx][k_idx]
                        
                        r_kl_0 = self.rAv[k_idx][l_idx]
                        d_kl_0 = self.dAv[k_idx][l_idx]
                        alpha_kl = self.aAv[k_idx][l_idx]
                        
                        
                        
                        
                        
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = double_bond
                        #else:
                        covalent_length_ij = single_bond

                        #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])

                        #single_bond = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[j])
                        #double_bond = double_covalent_radii_lib(element_list[k]) + double_covalent_radii_lib(element_list[j])
                        #triple_bond = triple_covalent_radii_lib(element_list[k]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_jk):
                        #    covalent_length_jk = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_jk):
                        #    covalent_length_jk = double_bond
                        #else:
                        covalent_length_jk = single_bond

                        #covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        #r_jk = coord[j] - coord[k]
                        #vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        #covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_kl = coord[k] - coord[l]

                        #single_bond = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        #double_bond = double_covalent_radii_lib(element_list[k]) + double_covalent_radii_lib(element_list[l])
                        #triple_bond = triple_covalent_radii_lib(element_list[k]) + triple_covalent_radii_lib(element_list[l])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_kl):
                        #    covalent_length_kl = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_kl):
                        #    covalent_length_kl = double_bond
                        #else:
                        covalent_length_kl = single_bond

                        vdw_length_kl = UFF_VDW_distance_lib(element_list[k]) + UFF_VDW_distance_lib(element_list[l])
                        #covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        a35 = (35.0/180)* np.pi
                        cosfi_max=np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ij, norm_r_ij)
                        g_jk = self.calc_force_const(alpha_jk, covalent_length_jk, r_jk_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_jk, norm_r_jk)
                        g_kl = self.calc_force_const(alpha_kl, covalent_length_kl, r_kl_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_kl, norm_r_kl) 
                       
                        t_ij = self.kt * (g_ij * 0.5 * self.kd / self.kr * d_ij_0) * (g_jk * 0.5 * self.kd / self.kr * d_jk_0) * (g_kl * 0.5 * self.kd / self.kr * d_kl_0)
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] = self.cart_hess[3 * i + n][3 * j + m] + t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] = self.cart_hess[3 * i + n][3 * k + m] + t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] = self.cart_hess[3 * i + n][3 * l + m] + t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] = self.cart_hess[3 * j + n][3 * k + m] + t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] = self.cart_hess[3 * j + n][3 * l + m] + t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] = self.cart_hess[3 * k + n][3 * l + m] + t_ij * s_k[n] * s_l[m]
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] = self.cart_hess[3 * i + n][3 * i + m] + t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] = self.cart_hess[3 * j + n][3 * j + m] + t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] = self.cart_hess[3 * k + n][3 * k + m] + t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] = self.cart_hess[3 * l + n][3 * l + m] + t_ij * s_l[n] * s_l[m]
                       
        return
       
    def lindh2007_out_of_plane(self, coord, element_list):
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = double_bond
                        #else:
                        covalent_length_ij = single_bond

                        #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[k])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[k])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ik):
                        #    covalent_length_ik = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ik):
                        #    covalent_length_ik = double_bond
                        #else:
                        covalent_length_ik = single_bond

                        #covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        vdw_length_il = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[l])

                        #single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        #double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[l])
                        #triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[l])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_il):
                        #    covalent_length_il = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_il):
                        #    covalent_length_il = double_bond
                        #else:
                        covalent_length_il = single_bond

                        #covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        idx_i = self.select_idx(element_list[i])
                        idx_j = self.select_idx(element_list[j])
                        idx_k = self.select_idx(element_list[k])
                        idx_l = self.select_idx(element_list[l])
                        
                        d_ij_0 = self.dAv[idx_i][idx_j]
                        r_ij_0 = self.rAv[idx_i][idx_j]
                        alpha_ij = self.aAv[idx_i][idx_j]
                        
                        d_ik_0 = self.dAv[idx_i][idx_k]
                        r_ik_0 = self.rAv[idx_i][idx_k]
                        alpha_ik = self.aAv[idx_i][idx_k]
                        
                        d_il_0 = self.dAv[idx_i][idx_l]
                        r_il_0 = self.rAv[idx_i][idx_l]
                        alpha_il = self.aAv[idx_i][idx_l]
                        
                        
                        
                        
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        kd = 0.0
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_ij, norm_r_ij)
                        
                        g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_ik, norm_r_ik)
                        
                        g_il = self.calc_force_const(alpha_il, covalent_length_il, r_il_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_il, norm_r_il)
                        
                        
                        t_ij = self.ko * g_ij * g_ik * g_il #self.ko * g_ij * g_ik * g_il
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                          
        return

    def main(self, coord, element_list, cart_gradient):
        norm_grad = np.linalg.norm(cart_gradient)
        scale = 0.1
        eigval_scale = scale * np.exp(-1 * norm_grad ** 2.0)
        #coord: Bohr
        print("generating Lindh's (2007) approximate hessian...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        self.lindh2007_bond(coord, element_list)
        self.lindh2007_angle(coord, element_list)
        self.lindh2007_dihedral_angle(coord, element_list)
        self.lindh2007_out_of_plane(coord, element_list)
        
        for i in range(len(coord)*3):
            for j in range(len(coord)*3):
                
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        hess_proj = np.dot(np.dot(eigenvectors, np.diag(np.abs(eigenvalues) * eigval_scale)), np.linalg.inv(eigenvectors))
        
        return hess_proj#cart_hess



class SimpleApproxHessianv1:
    def __init__(self):
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        return
    
    def guess_hessian(self, coord, element_list):
        RIC_approx_diag_hessian = []
        for i, j in itertools.combinations([k for k in range(len(coord))], 2):
            a = 1.2
            b = 0.15
            elem_1 = element_list[i]
            elem_2 = element_list[j]
            covalent_length = (covalent_radii_lib(elem_1) + covalent_radii_lib(elem_2)) * a
            distance = np.linalg.norm(coord[i] - coord[j])
            spring_const = -2 * b ** 2 * np.exp((covalent_length - distance) * 2 * b) * (np.exp(b * (distance - covalent_length)) - 2.0) # second derivative of Morse potential
            RIC_approx_diag_hessian.append(spring_const)
        RIC_approx_diag_hessian = np.array(RIC_approx_diag_hessian, dtype="float64")
        RIC_approx_hessian = np.array(np.diag(RIC_approx_diag_hessian), dtype="float64")
        
        return RIC_approx_hessian
    
    def guess_vdw_hessian(self, coord, element_list):
        RIC_approx_diag_hessian = []
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        connnectivity = [[[i, j] for i, j in itertools.combinations([k for k in range(len(coord))], 2)], [], []]
        connectivity_table = [BC.bond_connect_table(b_c_mat),[],[]]
        vbw_connectivity_table = [[[i, j] for i, j in itertools.combinations([k for k in range(len(coord))], 2) if not [i, j] in connectivity_table[0]], [], []]
        for pair in connnectivity[0]:
            if pair in connectivity_table[0]:
                RIC_approx_diag_hessian.append(0.0)
                
            else:
                distance = np.linalg.norm(coord[pair[0]] - coord[pair[1]])
                epsilon = np.sqrt(UFF_VDW_well_depth_lib(element_list[pair[0]]) * UFF_VDW_well_depth_lib(element_list[pair[1]]))
                sigma = UFF_VDW_distance_lib(element_list[pair[0]]) + UFF_VDW_distance_lib(element_list[pair[1]])
                vdw_fc = -1 * (84 * epsilon * sigma ** 6 * distance ** 6 - 156 * epsilon * sigma ** 12) / distance ** 14
                RIC_approx_diag_hessian.append(vdw_fc)
        RIC_approx_vdw_diag_hessian = np.array(RIC_approx_diag_hessian, dtype="float64")
        RIC_approx_vdw_hessian = np.array(np.diag(RIC_approx_vdw_diag_hessian), dtype="float64")
        return RIC_approx_vdw_hessian, vbw_connectivity_table
    
    def main(self, coord, element_list, cart_gradient):
        print("generating simple spring model hessian...")
        cart_gradient = cart_gradient.reshape(3*(len(cart_gradient)), 1)
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        int_grad = RedundantInternalCoordinates().cartgrad2RICgrad(cart_gradient, b_mat)
        int_approx_hess = self.guess_hessian(coord, element_list)
        connnectivity = [[[i, j] for i, j in itertools.combinations([k for k in range(len(coord))], 2)], [], []]
        cart_hess = RedundantInternalCoordinates().RIChess2carthess(coord, connnectivity, 
                                                                    int_approx_hess, b_mat, int_grad)
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        #int_vdw_approx_hess, vbw_connectivity_table = self.guess_vdw_hessian(coord, element_list)
        #cart_vdw_hess = RedundantInternalCoordinates().RIChess2carthess(coord, vbw_connectivity_table,
        #                                                                int_vdw_approx_hess, b_mat, int_grad)
        #cart_vdw_hess = np.nan_to_num(cart_vdw_hess, nan=0.0)
        #cart_hess = cart_hess + cart_vdw_hess
        
        eigenvalue, _ = np.linalg.eigh(cart_hess)
        print("eigenvalue:", sorted(eigenvalue))
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        hess_proj = cart_hess
        
        return hess_proj




class ApproxHessian:
    def __init__(self):
        return
    
    def main(self, coord, element_list, cart_gradient, approx_hess_type="Lindh2007"):
        #coord: Bohr
        if approx_hess_type == "Lindh":
            LAH = LindhApproxHessian(coord, element_list, cart_gradient)
            hess_proj = LAH.main(coord, element_list, cart_gradient)
             
        elif approx_hess_type == "Schlegel":
            SAH = SchlegelApproxHessian()
            hess_proj = SAH.main(coord, element_list, cart_gradient)
        elif approx_hess_type == "Swart":
            SWH = SwartD2ApproxHessian()
            hess_proj = SWH.main(coord, element_list, cart_gradient)
        elif approx_hess_type == "Lindh2007":
            LH2007 = Lindh2007D2ApproxHessian()
            hess_proj = LH2007.main(coord, element_list, cart_gradient)
        else:
            SH = SimpleApproxHessianv1()
            hess_proj = SH.main(coord, element_list, cart_gradient)
        
        return hess_proj#cart_hess


def test():
    AH = ApproxHessian()
    words = ["O        1.607230637      0.000000000     -4.017111134",
             "O        1.607230637      0.463701826     -2.637210910",
             "H        2.429229637      0.052572461     -2.324941515",
             "H        0.785231637     -0.516274287     -4.017735703"]
    
    elements = []
    coord = []
    
    for word in words:
        sw = word.split()
        elements.append(sw[0])
        coord.append(sw[1:4])
    
    coord = np.array(coord, dtype="float64")/UnitValueLib().bohr2angstroms#Bohr
    gradient = np.array([[-0.0028911  ,  -0.0015559   ,  0.0002471],
                         [ 0.0028769  ,  -0.0013954   ,  0.0007272],
                         [-0.0025737   ,  0.0013921   , -0.0007226],
                         [ 0.0025880   ,  0.0015592  ,  -0.0002518]], dtype="float64")#a. u.
    
    hess_proj = AH.main(coord, elements, gradient)
    
    return hess_proj



if __name__ == "__main__":#test
    test()
    
    
    
    