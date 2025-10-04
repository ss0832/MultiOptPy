import numpy as np
import copy
from scipy.signal import argrelextrema
from multioptpy.Coordinate.redundant_coordinate import calc_int_grad_from_pBmat, calc_cart_grad_from_pBmat


def extremum_list_index(energy_list):
    local_max_energy_list_index = argrelextrema(energy_list, np.greater)
    inverse_energy_list = (-1)*energy_list
    local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)

    local_max_energy_list_index = local_max_energy_list_index[0].tolist()
    local_min_energy_list_index = local_min_energy_list_index[0].tolist()
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    return local_max_energy_list_index, local_min_energy_list_index


    
    
class CaluculationQSM:
    def __init__(self, APPLY_CI_NEB=99999):
        self.spring_constant_k = 0.01
        self.APPLY_CI_NEB = APPLY_CI_NEB
        self.force_const_for_cineb = 0.01

    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        print("QSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSMQSM")
        #ref. J. Chem. Phys. 124, 054109 (2006)
        nnode = len(energy_list)
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        total_force_list = []
        self.tau_list = []
        for i in range(nnode):
            if i == 0:
                total_force_list.append(-1*np.array(gradient_list[0], dtype = "float64"))
                self.tau_list.append(np.zeros_like(geometry_num_list[0], dtype = "float64"))
                continue
            elif i == nnode-1:
                total_force_list.append(-1*np.array(gradient_list[nnode-1], dtype = "float64"))
                self.tau_list.append(np.zeros_like(geometry_num_list[nnode-1], dtype = "float64"))
                continue
            tmp_grad = copy.copy(gradient_list[i]).reshape(-1, 1)
            force, tangent_grad = self.calc_project_out_grad(geometry_num_list[i-1], geometry_num_list[i], geometry_num_list[i+1], tmp_grad, energy_list[i-1:i+2])     
            if optimize_num > self.APPLY_CI_NEB and (i + 1 in local_max_energy_list_index or i - 1 in local_max_energy_list_index) and (i != 1 and i != nnode-2):
                force *= 0.001
                print("Restrect step of # NODE", i, " for CI-NEB")
            elif optimize_num > self.APPLY_CI_NEB and (i in local_max_energy_list_index) and (i != 1 or i != nnode-2):
                force = self.calc_ci_neb_force(tmp_grad, tangent_grad)
                print("CI-NEB was applied to # NODE", i)
            else:
                pass
            
            
           
            total_force_list.append(-1*force.reshape(-1, 3))
            self.tau_list.append(tangent_grad.reshape(-1, 3) / (np.linalg.norm(tangent_grad) + 1e-15))
        
        total_force_list = np.array(total_force_list, dtype = "float64")
        total_force_list = projection(total_force_list, geometry_num_list)
            
        return total_force_list
    
    def calc_project_out_grad(self, coord_1, coord_2, coord_3, grad_2, energy_list):# grad: (3N, 1), geom_num_list: (N, 3)
        natom = len(coord_2)
        tmp_grad = copy.copy(grad_2)
        if energy_list[0] < energy_list[1] and energy_list[1] < energy_list[2]:
            B_mat = self.calc_B_matrix_for_NEB_tangent(coord_2, coord_3)
            int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)
            projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
            tangent_grad = projection_grad
        elif energy_list[0] > energy_list[1] and energy_list[1] > energy_list[2]:
            B_mat = self.calc_B_matrix_for_NEB_tangent(coord_1, coord_2)
            int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)
            projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
            tangent_grad = projection_grad
        else:
            B_mat_plus = self.calc_B_matrix_for_NEB_tangent(coord_2, coord_3)
            B_mat_minus = self.calc_B_matrix_for_NEB_tangent(coord_1, coord_2)
            int_grad_plus = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat_plus)
            int_grad_minus = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat_minus)
            max_ene = max(abs(energy_list[2] - energy_list[1]), abs(energy_list[1] - energy_list[0]))
            min_ene = min(abs(energy_list[2] - energy_list[1]), abs(energy_list[1] - energy_list[0]))
            a = (max_ene) / (max_ene + min_ene + 1e-8)
            b = (min_ene) / (max_ene + min_ene + 1e-8)
            
            if energy_list[0] < energy_list[2]:
                projection_grad_plus = calc_cart_grad_from_pBmat(-a*int_grad_plus, B_mat_plus)
                projection_grad_minus = calc_cart_grad_from_pBmat(-b*int_grad_minus, B_mat_minus)
            
            else:
                projection_grad_plus = calc_cart_grad_from_pBmat(-b*int_grad_plus, B_mat_plus)
                projection_grad_minus = calc_cart_grad_from_pBmat(-a*int_grad_minus, B_mat_minus)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad_plus + projection_grad_minus
            tangent_grad = projection_grad_plus + projection_grad_minus
        return proj_grad, tangent_grad

    
    def calc_B_matrix_for_NEB_tangent(self, coord_1, coord_2):
        natom = len(coord_2)
        B_mat = np.zeros((natom, 3*natom))
        
        for i in range(natom):
            norm_12 = np.linalg.norm(coord_1[i] - coord_2[i]) + 1e-15
            dr12_dx2 = (coord_2[i][0] - coord_1[i][0]) / norm_12
            dr12_dy2 = (coord_2[i][1] - coord_1[i][1]) / norm_12
            dr12_dz2 = (coord_2[i][2] - coord_1[i][2]) / norm_12
            B_mat[i][3*i] = dr12_dx2
            B_mat[i][3*i+1] = dr12_dy2
            B_mat[i][3*i+2] = dr12_dz2

        return B_mat
    
    def calc_proj_hess(self, hess, node_num, geometry_num_list):
        if node_num == 0 or node_num == len(geometry_num_list)-1:
            return hess
        proj_hess = projection_hess(hess, geometry_num_list, node_num)
        # Make sure the Hessian is symmetric
        proj_hess = 0.5 * (proj_hess + proj_hess.T)
        return proj_hess
    
    
    
def projection(move_vector_list, geometry_list):
    print("Applying projection to move vectors")
    for i in range(1, len(geometry_list)-1):
        vec_1 = geometry_list[i] - geometry_list[i-1]
        vec_2 = geometry_list[i+1] - geometry_list[i]
        vec_1_norm = np.linalg.norm(vec_1)
        vec_2_norm = np.linalg.norm(vec_2)
        if vec_1_norm < 1e-8 or vec_2_norm < 1e-8:
            continue
        vec_1 = vec_1 / vec_1_norm
        vec_2 = vec_2 / vec_2_norm
        vec_1 = vec_1.reshape(-1, 1)
        vec_2 = vec_2.reshape(-1, 1)
        # Gram-Schmidt process
        vec_2 -= np.dot(vec_2.T, vec_1) * vec_1
        if np.linalg.norm(vec_2) < 1e-8:
            continue
        vec_2 /= np.linalg.norm(vec_2)
        
        P = np.eye(len(vec_1)) - np.outer(vec_1, vec_1) - np.outer(vec_2, vec_2)
        tmp_proj_move_vec = np.dot(P, move_vector_list[i].reshape(-1, 1))
        move_vector_list[i] = tmp_proj_move_vec.reshape(-1, 3)
    return move_vector_list


def projection_hess(hessian, geometry_list, node_num):
    print("Applying projection to Hessian")

    vec_1 = geometry_list[node_num] - geometry_list[node_num-1]
    vec_2 = geometry_list[node_num+1] - geometry_list[node_num]
    vec_1_norm = np.linalg.norm(vec_1)
    vec_2_norm = np.linalg.norm(vec_2)
    if vec_1_norm < 1e-8 or vec_2_norm < 1e-8:
        return hessian
    vec_1 = vec_1 / vec_1_norm
    vec_2 = vec_2 / vec_2_norm
    vec_1 = vec_1.reshape(-1, 1)
    vec_2 = vec_2.reshape(-1, 1)
    # Gram-Schmidt process
    vec_2 -= np.dot(vec_2.T, vec_1) * vec_1
    if np.linalg.norm(vec_2) < 1e-8:
        return hessian
    vec_2 /= np.linalg.norm(vec_2)

    P = np.eye(len(vec_1)) - np.outer(vec_1, vec_1) - np.outer(vec_2, vec_2)
    tmp_proj_hess = np.dot(np.dot(P, hessian), P.T)
    return tmp_proj_hess