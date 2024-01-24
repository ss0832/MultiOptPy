import itertools

import numpy as np

from bond_connectivity import BondConnectivity
from param import UnitValueLib, number_element, covalent_radii_lib, UFF_effective_charge_lib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib
from redundant_coordinations import RedundantInternalCoordinates
from calc_tools import Calculationtools

class ApproxHessian:
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
    
    def guess_hessian(self, coord, element_list):
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
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    RIC_approx_diag_hessian[tmpnum] += force_const
                  
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
        int_approx_hess = self.guess_hessian(coord, element_list)
        BC = BondConnectivity()
        
        connnectivity = BC.connectivity_table(coord, element_list)
        #print(connnectivity, len(connnectivity[0])+len(connnectivity[1])+len(connnectivity[2]))
        cart_hess = RedundantInternalCoordinates().RIChess2carthess(coord, connnectivity, 
                                                                    int_approx_hess, b_mat, int_grad)
        hess_proj = Calculationtools().project_out_hess_tr_and_rot(cart_hess, element_list, coord)
        return hess_proj
        
if __name__ == "__main__":#test
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
    
    coord = np.array(coord, dtype="float64")

    coord = np.array(coord, dtype="float64")/UnitValueLib().bohr2angstroms#Bohr
    gradient = np.array([[-0.0028911  ,  -0.0015559   ,  0.0002471],
                         [ 0.0028769  ,  -0.0013954   ,  0.0007272],
                         [-0.0025737   ,  0.0013921   , -0.0007226],
                         [ 0.0025880   ,  0.0015592  ,  -0.0002518]], dtype="float64")#a. u.
    
    hess_proj = AH.main(coord, elements, gradient)
    
    
    
    