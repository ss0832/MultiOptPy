import numpy as np
import copy
from redundant_coordinations import calc_int_grad_from_pBmat, calc_cart_grad_from_pBmat, calc_inv_B_mat, calc_G_mat, calc_inv_G_mat

class CaluculationBNEB():# Wilson's B-matrix-constrained NEB
    def __init__(self, APPLY_CI_NEB=99999):
        return
    
    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        nnode = len(energy_list)
        total_force_list = []
        for i in range(nnode):
            if i == 0:
                total_force_list.append(-1*np.array(gradient_list[0], dtype = "float64"))
                continue
            elif i == nnode-1:
                total_force_list.append(-1*np.array(gradient_list[nnode-1], dtype = "float64"))
                continue
            tmp_grad = copy.copy(gradient_list[i]).reshape(-1, 1)
            force = self.calc_project_out_grad(geometry_num_list[i-1], geometry_num_list[i], geometry_num_list[i+1], tmp_grad)
            total_force_list.append(-1*force) 
        return np.array(total_force_list, dtype = "float64")
    
    def calc_project_out_grad(self, coord_1, coord_2, coord_3, grad_2):# grad: (3N, 1), geom_num_list: (N, 3)
        natom = len(coord_2)
        tmp_grad = copy.copy(grad_2)
   
       
        B_mat = self.calc_B_matrix_for_NEB(coord_1, coord_2, coord_3)
                
        int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)
        projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
        proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
        proj_grad = proj_grad.reshape(natom, 3)
        
        return proj_grad


    def calc_B_matrix_for_NEB(self, coord_1, coord_2, coord_3):
        natom = len(coord_2)
        B_mat = np.zeros((3*natom, 3*natom))
        
        for i in range(natom):
            norm_12 = np.linalg.norm(coord_1[i] - coord_2[i]) + 1e-15
            dr12_dx2 = (coord_1[i][0] - coord_2[i][0]) / norm_12
            dr12_dy2 = (coord_1[i][1] - coord_2[i][1]) / norm_12
            dr12_dz2 = (coord_1[i][2] - coord_2[i][2]) / norm_12
            B_mat[i][3*i] = dr12_dx2
            B_mat[i][3*i+1] = dr12_dy2
            B_mat[i][3*i+2] = dr12_dz2
            
        for i in range(natom):
            norm_23 = np.linalg.norm(coord_2[i] - coord_3[i]) + 1e-15
            dr32_dx2 = (coord_3[i][0] - coord_2[i][0]) / norm_23
            dr32_dy2 = (coord_3[i][1] - coord_2[i][1]) / norm_23
            dr32_dz2 = (coord_3[i][2] - coord_2[i][2]) / norm_23
            B_mat[natom+i][3*i] = dr32_dx2
            B_mat[natom+i][3*i+1] = dr32_dy2
            B_mat[natom+i][3*i+2] = dr32_dz2
        
        for i in range(natom):
            norm_13 = np.linalg.norm(coord_3[i] - coord_1[i]) + 1e-15
            dr13_dx2 = (coord_3[i][0] - coord_1[i][0]) / norm_13
            dr13_dy2 = (coord_3[i][1] - coord_1[i][1]) / norm_13
            dr13_dz2 = (coord_3[i][2] - coord_1[i][2]) / norm_13
            B_mat[2*natom+i][3*i] = dr13_dx2
            B_mat[2*natom+i][3*i+1] = dr13_dy2
            B_mat[2*natom+i][3*i+2] = dr13_dz2
        
            
        return B_mat

