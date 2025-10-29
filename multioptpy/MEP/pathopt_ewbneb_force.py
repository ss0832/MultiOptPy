import numpy as np
import copy
from multioptpy.Coordinate.redundant_coordinate import calc_int_grad_from_pBmat, calc_cart_grad_from_pBmat
from scipy.signal import argrelextrema

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



class CaluculationEWBNEB():# Wilson's B-matrix-constrained EW-NEB
    def __init__(self, APPLY_CI_NEB=99999):
        self.APPLY_CI_NEB = APPLY_CI_NEB
        # Energy-Weighted spring const: J. Chem. Theory Comput. 2021, 17, 8, 4929–4945
        self.upper_spring_constant = 0.005
        self.lower_spring_constant = 0.0001
        return
    
    def calc_ci_neb_force(self, grad, tangent_grad):
        #ref.: J. Chem. Phys. 113, 9901–9904 (2000)
        #ref.: J. Chem. Phys. 142, 024106 (2015)
        #available for optimizer using only first order differential 
        ci_force = -2.0 * tangent_grad
        
        return ci_force
    
    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        print("EWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEBEWBNEB")
        nnode = len(energy_list)
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        max_ene = max(energy_list)
        if energy_list[0] > energy_list[-1]:
            ref_ene = energy_list[-1]
        else:
            ref_ene = energy_list[0]
        
        spring_force_list = []
        for i in range(nnode):
            if i == 0:
                spring_force_list.append(0.0)
            elif i == nnode-1:
                spring_force_list.append(0.0)
            else:
                denominator = max_ene - ref_ene
                if denominator < 1e-8:
                    denominator = 1e-8
                alpha = (energy_list[i] - ref_ene) / denominator
                if energy_list[i] < ref_ene:
                    tmp_spring_force = self.lower_spring_constant
                elif energy_list[i] > ref_ene:
                    tmp_spring_force = self.upper_spring_constant * (1.0 - alpha) + alpha * self.lower_spring_constant
                else:
                    print("Error: energy_list[i] is not in the range of ref_ene")
                    tmp_spring_force = 0.0
                spring_force_list.append(tmp_spring_force)
            
        
        
        total_force_list = []
        for i in range(nnode):
            if i == 0:
                total_force_list.append(-1*np.array(gradient_list[0], dtype = "float64"))
                continue
            elif i == nnode-1:
                total_force_list.append(-1*np.array(gradient_list[nnode-1], dtype = "float64"))
                continue
            
            # project out the gradient
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
            
            # add spring force
            norm_tangent_grad = np.linalg.norm(tangent_grad)
           
            fwd_vec = geometry_num_list[i+1] - geometry_num_list[i]
            bwd_vec = geometry_num_list[i] - geometry_num_list[i-1]
            
            norm_fwd_vec = np.linalg.norm(fwd_vec)
            norm_bwd_vec = np.linalg.norm(bwd_vec)
            
            if norm_fwd_vec > 1e-8:
                norm_fwd_vec = fwd_vec / norm_fwd_vec
            else:
                norm_fwd_vec = np.zeros_like(fwd_vec)
            
            if norm_bwd_vec > 1e-8:
                norm_bwd_vec = bwd_vec / norm_bwd_vec
            else:
                norm_bwd_vec = np.zeros_like(bwd_vec)
            
            spring_force = (spring_force_list[i] * norm_fwd_vec - spring_force_list[i-1] * norm_bwd_vec).reshape(-1, 1)
            force = force + spring_force
            
            total_force_list.append(-1*force.reshape(-1, 3)) 
        
        total_force_list = np.array(total_force_list, dtype = "float64")
        
            
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
    
    def projection_hessian(self, coord_1, coord_2, coord_3, gradient_list, hessian_2, energy_list):
        #ref.: J. Chem. Theory. Comput. 2013, 9, 3498−3504
        natom = len(coord_2)
        gradient_2 = gradient_list[1].reshape(-1, 1)
        gradient_1 = gradient_list[0].reshape(-1, 1)
        gradient_3 = gradient_list[2].reshape(-1, 1)
        if energy_list[0] < energy_list[1] and energy_list[1] < energy_list[2]:
            tangent = coord_3.reshape(-1, 1) - coord_2.reshape(-1, 1)
            grad_tangent = np.dot(np.eye(3*natom, k=1) - np.eye(3*natom, k=0), np.ones((3*natom, 3*natom)))
        elif energy_list[0] > energy_list[1] and energy_list[1] > energy_list[2]:
            tangent = coord_2.reshape(-1, 1) - coord_1.reshape(-1, 1)
            
            grad_tangent = np.dot(np.eye(3*natom, k=0) - np.eye(3*natom, k=-1), np.ones((3*natom, 3*natom)))
        else:
            ene_max = max(abs(energy_list[2] - energy_list[1]), abs(energy_list[1] - energy_list[0]))
            ene_min = min(abs(energy_list[2] - energy_list[1]), abs(energy_list[1] - energy_list[0]))
            tangent_plus = coord_3.reshape(-1, 1) - coord_2.reshape(-1, 1)
            tangent_minus = coord_2.reshape(-1, 1) - coord_1.reshape(-1, 1)
            if energy_list[0] < energy_list[2]:
                tangent = tangent_plus * ene_max + tangent_minus * ene_min
            else:
                tangent = tangent_plus * ene_min + tangent_minus * ene_max
            
            a = np.linalg.norm(coord_3 - coord_2)
            b = np.linalg.norm(coord_2 - coord_1)
            grad_a = np.sign(energy_list[2] - energy_list[1]) * (np.dot(np.eye(3*natom, k=1), gradient_3) - np.dot(np.eye(3*natom, k=0), gradient_2))
            grad_b = np.sign(energy_list[0] - energy_list[1]) * (np.dot(np.eye(3*natom, k=-1), gradient_1) - np.dot(np.eye(3*natom, k=0), gradient_2))
            grad_tangent = np.dot((a * np.eye(3*natom, k=1) + (b - a) * np.eye(3*natom, k=0) - b * np.eye(3*natom, k=-1)), np.ones((3*natom, 3*natom))) + np.dot(tangent_plus, grad_a.T) + np.dot(tangent_minus, grad_b.T)
        
        
        unit_tangent = tangent / (np.linalg.norm(tangent) + 1e-15)
        A = np.sum(gradient_2 * unit_tangent) * np.ones((3*natom, 3*natom)) + np.dot(unit_tangent, gradient_2.T) * (np.ones((3*natom, 3*natom)) - np.dot(unit_tangent, unit_tangent.T)) / (np.linalg.norm(tangent) + 1e-15)
        hessian_2 = hessian_2 -1 * np.dot(unit_tangent, np.dot(hessian_2, unit_tangent).T) + np.dot(A, grad_tangent)
        return hessian_2
    

    
