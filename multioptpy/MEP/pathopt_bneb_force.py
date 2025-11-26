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



class CaluculationBNEB():# Wilson's B-matrix-constrained NEB
    def __init__(self, APPLY_CI_NEB=99999):
        self.APPLY_CI_NEB = APPLY_CI_NEB
        return
    
    def calc_ci_neb_force(self, grad, tangent_grad):
        #ref.: J. Chem. Phys. 113, 9901–9904 (2000)
        #ref.: J. Chem. Phys. 142, 024106 (2015)
        #available for optimizer using only first order differential 
        ci_force = -2.0 * tangent_grad
        return ci_force
    
    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        print("BNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEB")
        nnode = len(energy_list)
        self.tau_list = []
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        total_force_list = []
        for i in range(nnode):
            if i == 0:
                total_force_list.append(-1*np.array(gradient_list[0], dtype = "float64"))
                self.tau_list.append(np.zeros_like(gradient_list[0]).reshape(-1, 3))
                continue
            elif i == nnode-1:
                total_force_list.append(-1*np.array(gradient_list[nnode-1], dtype = "float64"))
                self.tau_list.append(np.zeros_like(gradient_list[nnode-1]).reshape(-1, 3))
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
            self.tau_list.append(tangent_grad.reshape(-1, 3))
        
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
    

    def get_tau(self, node_num):
        """Returns the flattened tangent vector at the specified node."""
        if len(self.tau_list) == 0:
            raise ValueError("Tangent list is empty. Calculate forces first.")
        return self.tau_list[node_num]

    def calculate_gamma(self, q_triplet, E_triplet, g_triplet, tangent):
        """
        Calculates the curvature gamma along the path using quintic polynomial fitting.
        
        Args:
            q_triplet: List of [q_prev, q_curr, q_next] coordinates
            E_triplet: List of [E_prev, E_curr, E_next] energies
            g_triplet: List of [g_prev, g_curr, g_next] gradients
            tangent: Normalized tangent vector at the current node
            
        Returns:
            gamma: Curvature (2nd derivative) along the path at the current node
        """
        q_prev, q_curr, q_next = q_triplet
        E_prev, E_curr, E_next = E_triplet
        g_prev, g_curr, g_next = g_triplet
        
        # 1. Distances along the path
        dist_prev = np.linalg.norm(q_curr - q_prev)
        dist_next = np.linalg.norm(q_next - q_curr)
        
        if dist_prev < 1e-6 or dist_next < 1e-6:
            return 0.0

        # s coordinates: prev at -dist_prev, curr at 0, next at +dist_next
        s_p = -dist_prev
        s_c = 0.0
        s_n = dist_next
        
        # 2. Project gradients onto path
        # Tangent at i-1: Approximated by direction from i-1 to i
        t_prev = (q_curr - q_prev) / dist_prev
        gp_proj = np.dot(g_prev.flatten(), t_prev.flatten())
        
        # Tangent at i: Given tangent
        gc_proj = np.dot(g_curr.flatten(), tangent.flatten())
        
        # Tangent at i+1: Approximated by direction from i to i+1
        t_next = (q_next - q_curr) / dist_next
        gn_proj = np.dot(g_next.flatten(), t_next.flatten())
        
        # 3. Solve Quintic Polynomial Coefficients
        # E(s) = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5
        A = np.array([
            [1, s_p, s_p**2, s_p**3, s_p**4, s_p**5],
            [1, s_c, s_c**2, s_c**3, s_c**4, s_c**5],
            [1, s_n, s_n**2, s_n**3, s_n**4, s_n**5],
            [0, 1, 2*s_p, 3*s_p**2, 4*s_p**3, 5*s_p**4],
            [0, 1, 2*s_c, 3*s_c**2, 4*s_c**3, 5*s_c**4],
            [0, 1, 2*s_n, 3*s_n**2, 4*s_n**3, 5*s_n**4]
        ])
        
        b = np.array([E_prev, E_curr, E_next, gp_proj, gc_proj, gn_proj])
        
        try:
            coeffs = np.linalg.solve(A, b)
            # Curvature gamma = E''(0) = 2 * c2
            gamma = 2.0 * coeffs[2]
            return gamma
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            return 0.0

    

class CaluculationBNEB2():# Wilson's B-matrix-constrained NEB
    def __init__(self, APPLY_CI_NEB=99999):
        return
    
    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        print("BNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEB")
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
   
        B_mat = self.calc_B_matrix_for_NEB_1st_stage(coord_1, coord_2, coord_3)
        int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)
        projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
        proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
        
        B_mat = self.calc_B_matrix_for_NEB_2nd_stage(coord_1, coord_3)
        int_grad = calc_int_grad_from_pBmat(proj_grad, B_mat)
        projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
        proj_grad = proj_grad + projection_grad
        proj_grad = proj_grad.reshape(natom, 3)
        
        return proj_grad


    def calc_B_matrix_for_NEB_1st_stage(self, coord_1, coord_2, coord_3):
        natom = len(coord_2)
        B_mat = np.zeros((2*natom, 3*natom))
        
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
        
            
        return B_mat


    def calc_B_matrix_for_NEB_2nd_stage(self, coord_1, coord_3):
        natom = len(coord_1)
        B_mat = np.zeros((natom, 3*natom))
        
        for i in range(natom):
            norm_13 = np.linalg.norm(coord_3[i] - coord_1[i]) + 1e-15
            dr13_dx2 = (coord_3[i][0] - coord_1[i][0]) / norm_13
            dr13_dy2 = (coord_3[i][1] - coord_1[i][1]) / norm_13
            dr13_dz2 = (coord_3[i][2] - coord_1[i][2]) / norm_13
            B_mat[i][3*i] = dr13_dx2
            B_mat[i][3*i+1] = dr13_dy2
            B_mat[i][3*i+2] = dr13_dz2
        
        return B_mat

class CaluculationBNEB3():# Wilson's B-matrix-constrained NEB
    def __init__(self, APPLY_CI_NEB=99999):
        self.APPLY_CI_NEB = APPLY_CI_NEB
        self.spring_force_const = 0.05
        return
    
    def calc_ci_neb_force(self, grad, tangent_grad):
        #ref.: J. Chem. Phys. 113, 9901–9904 (2000)
        #ref.: J. Chem. Phys. 142, 024106 (2015)
        #available for optimizer using only first order differential 
        ci_force = -2.0 * tangent_grad
        return ci_force
    
    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        print("BNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEBBNEB")
        nnode = len(energy_list)
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        total_force_list = []
        for i in range(nnode):
            if i == 0:
                total_force_list.append(-1*np.array(gradient_list[0], dtype = "float64"))
                continue
            elif i == nnode-1:
                total_force_list.append(-1*np.array(gradient_list[nnode-1], dtype = "float64"))
                continue
            tmp_grad = copy.copy(gradient_list[i]).reshape(-1, 1)
            force, _ = self.calc_project_out_grad(geometry_num_list[i-1], geometry_num_list[i], geometry_num_list[i+1], tmp_grad, energy_list[i-1:i+2])     
            if i > 1 and i < nnode - 2:
                spring_force = self.calc_spring_force(geometry_num_list[i-2], geometry_num_list[i-1], geometry_num_list[i], geometry_num_list[i+1], geometry_num_list[i+2], i, nnode)
            else:
                spring_force = 0.0 * force

            total_force_list.append(-1*force.reshape(-1, 3) -1* spring_force.reshape(-1, 3))

        total_force_list = np.array(total_force_list, dtype = "float64")
        
            
        return total_force_list
    
    def calc_spring_force(self, coord_0, coord_1, coord_2, coord_3, coord_4, node_num, max_node_num):
        if 1 < node_num and max_node_num - 2 > node_num:
            # coord_0, coord_1, coord_2
            force_1 = self.spring_force_const * (np.linalg.norm(coord_1 - coord_2) - np.linalg.norm(coord_0 - coord_1)) * (coord_1 - coord_2) / (np.linalg.norm(coord_1 - coord_2) + 1e-15)
            
            # coord_1, coord_2, coord_3
            force_2 = self.spring_force_const * (np.linalg.norm(coord_2 - coord_3) - np.linalg.norm(coord_1 - coord_2)) * (-1 * (coord_1 - coord_2) / (np.linalg.norm(coord_1 - coord_2) + 1e-15) - (coord_2 - coord_3) / (np.linalg.norm(coord_2 - coord_3) + 1e-15))

            # coord_2, coord_3, coord_4
            force_3 = self.spring_force_const * (np.linalg.norm(coord_3 - coord_4) - np.linalg.norm(coord_2 - coord_3)) * (coord_3 - coord_4) / (np.linalg.norm(coord_3 - coord_4) + 1e-15)

        force = force_1 + force_2 + force_3
        return force
    
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
