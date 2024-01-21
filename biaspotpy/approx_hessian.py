import numpy as np

from bond_connectivity import BondConnectivity
from param import UnitValueLib, number_element
from redundant_coordinations import RedundantInternalCoordinates
from calc_tools import Calculationtools

class ApproxHessian:
    def __init__(self):
        #Lindh's approximate hessian
        #Ref: https://doi.org/10.1016/0009-2614(95)00646-L
        #Lindh, R., Chemical Physics Letters 1995, 241 (4), 423–428.
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.force_const_list = [0.45, 0.15, 0.005]  #bond, angle, dihedral_angle
        
        return
    
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
        
        const_R = const_R_list[idx_1][idx_2]
        const_alpha = const_alpha_list[idx_1][idx_2]
        
        return const_R, const_alpha
    
    def guess_hessian(self, coord, element_list):
        #coord: cartecian coord, Bohr (atom num × 3)
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        val_num = len(coord)*3
        connectivity_table = [BC.bond_connect_table(b_c_mat), BC.angle_connect_table(b_c_mat), BC.dihedral_angle_connect_table(b_c_mat)]
        RIC_approx_diag_hessian = []
      
        
        for idx_list in connectivity_table:
            for idx in idx_list:
                force_const = self.force_const_list[len(idx)-2]
                for i in range(len(idx)-1):
                    elem_1 = element_list[idx[i]]
                    elem_2 = element_list[idx[i+1]]
                    const_R, const_alpha = self.return_lindh_const(elem_1, elem_2)
                    
                    R = np.linalg.norm(coord[idx[i]] - coord[idx[i+1]])
                    force_const *= np.exp(const_alpha * (const_R**2 - R**2)) 
               
                RIC_approx_diag_hessian.append(force_const)
                
            
        
       
        
        RIC_approx_hessian = np.array(np.diag(RIC_approx_diag_hessian), dtype="float64")
        
        return RIC_approx_hessian
    
    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        
        print("generating Lindh's approximate hessian...")
        cart_gradient = cart_gradient.reshape(3*(len(cart_gradient)), 1)
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        int_grad = RedundantInternalCoordinates().cartgrad2RICgrad(cart_gradient, b_mat)
        int_approx_hess = self.guess_hessian(coord, element_list)
        BC = BondConnectivity()
        
        connnectivity = BC.connectivity_table(coord, element_list)
        print(connnectivity, len(connnectivity[0])+len(connnectivity[1])+len(connnectivity[2]))
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
    
    AH.main(coord, elements, gradient)
    gradient = gradient.reshape(12, 1)
    b_mat = RedundantInternalCoordinates().B_matrix(coord)
    int_grad = RedundantInternalCoordinates().cartgrad2RICgrad(gradient, b_mat)
    
    results = AH.guess_hessian(coord, elements)
    print(results)
    BC = BondConnectivity()
    connnectivity = BC.connectivity_table(coord, elements)
    
    print(connnectivity)
    cart_hess = RedundantInternalCoordinates().RIChess2carthess(coord, connnectivity, results, b_mat, int_grad)
    print(cart_hess)
    eigenvalue, eigenvector = np.linalg.eig(cart_hess)
    print(sorted(eigenvalue))
    from calc_tools import Calculationtools
    hess_proj = Calculationtools().project_out_hess_tr_and_rot(cart_hess, elements, coord)
    eigenvalue, eigenvector = np.linalg.eig(hess_proj)
    
    
    