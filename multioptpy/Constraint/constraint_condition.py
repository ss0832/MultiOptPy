import numpy as np
import copy
import torch
from collections import deque

from multioptpy.Parameters.parameter import atomic_mass, UnitValueLib
from multioptpy.Utils.calc_tools import (calc_bond_length_from_vec, 
                        calc_angle_from_vec, 
                        calc_dihedral_angle_from_vec, change_atom_distance_both_side, 
                        change_bond_angle_both_side, 
                        change_torsion_angle_both_side,
                        change_fragm_distance_both_side,
                        Calculationtools
                        )
from multioptpy.Coordinate.redundant_coordinate import (TorchDerivatives, 
                                     partial_stretch_B_matirx,
                                     partial_bend_B_matrix,
                                     partial_torsion_B_matrix,
                                     RedundantInternalCoordinates,
                                     torch_B_matrix,
                                     torch_B_matrix_derivative,
                                     torch_calc_distance,
                                     torch_calc_fragm_distance,
                                     torch_calc_angle,
                                     torch_calc_dihedral_angle,
                                     calc_dot_B_deriv_int_grad,
                                     calc_int_hess_from_pBmat_for_non_stationary_point,
                                     calc_cart_hess_from_pBmat_for_non_stationary_point,
                                     calc_int_cart_coupling_hess_from_pBmat_for_non_stationary_point,
                                     calc_int_grad_from_pBmat,
                                     calc_cart_grad_from_pBmat,
                                     )

def isduplicated(num_list):
    numbers = [item for sublist in num_list for item in sublist]
    boolean = len(numbers) != len(set(numbers))
    return boolean



def shake_parser(constraints):
    bond_list = []
    angle_list = []
    dihedral_angle_list = []
    
    for i in range(len(constraints)):
        constraint = constraints[i].split(",")
        if len(constraint) == 3:
            bond_list.append([float(constraint[0])]+list(map(int, constraint[1:])))
        elif len(constraint) == 4:
            angle_list.append([float(constraint[0])]+list(map(int, constraint[1:])))
        elif len(constraint) == 5:
            dihedral_angle_list.append([float(constraint[0])]+list(map(int, constraint[1:])))
        else:
            print("error")
            raise "error (invaild input of constraint conditions)"
    constraints_list = [bond_list, angle_list, dihedral_angle_list]
    return constraints_list



class SHAKE:
    def __init__(self, time_scale, constraints=[]):
        #ref.: Journal of Computational Physics. 23, (3), 327–341.
        self.convergent_criterion = 1e-5
        self.maxiter = 100000
        self.time_scale = time_scale
        self.constraint_condition = constraints[0] + constraints[1] + constraints[2]

        
    def run(self, geom_num_list, prev_geom_num_list, momentum_list, element_list):
        print("applying constraint conditions...")
        new_geometry = copy.copy(geom_num_list)
        new_momentum_list = copy.copy(momentum_list)
        for iter in range(self.maxiter):
            isconverged = True
            for constraint in self.constraint_condition:
                if len(constraint) == 3: # bond
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    constraint_distance = constraint[0] / UnitValueLib().bohr2angstroms
                    r_ij = new_geometry[idx_i] - new_geometry[idx_j]
                    check_convergence = abs(constraint_distance - np.linalg.norm(r_ij))
                   
                    if check_convergence < self.convergent_criterion:
                        
                        continue
                    isconverged = False
                    prev_r_ij = prev_geom_num_list[idx_i] - prev_geom_num_list[idx_j]
                    g_ij = (np.linalg.norm(r_ij) ** 2 - constraint_distance ** 2) / (2 * (np.sum(r_ij * prev_r_ij)) * (1/atomic_mass(element_list[idx_i]) + 1/atomic_mass(element_list[idx_j])))
                    new_geometry[idx_i] -= g_ij / atomic_mass(element_list[idx_i]) * prev_r_ij
                    new_geometry[idx_j] += g_ij / atomic_mass(element_list[idx_j]) * prev_r_ij
                    new_momentum_list[idx_i] -= g_ij / self.time_scale * prev_r_ij
                    new_momentum_list[idx_j] += g_ij / self.time_scale * prev_r_ij
                    
                elif len(constraint) == 4: # angle
                    # ref.:J. Chem. Phys. 133, 034114 (2010)
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    idx_k = constraint[3] - 1
                    constraint_angle = np.deg2rad(constraint[0])
                    r_ij = new_geometry[idx_i] - new_geometry[idx_j]
                    r_kj = new_geometry[idx_k] - new_geometry[idx_j]
                    inner_product_r_ij_r_kj = np.sum(r_ij * r_kj)
                    cos = inner_product_r_ij_r_kj / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj))
                    constraint_cos = np.cos(constraint_angle)
                    check_convergence = abs(cos ** 2 - constraint_cos ** 2)
                    #print(check_convergence)
                    if check_convergence < self.convergent_criterion:
                        
                        continue
                    isconverged = False
                    h_i = -2 * cos * (-1 * cos * r_ij / np.linalg.norm(r_ij) + r_kj / np.linalg.norm(r_kj)) / np.linalg.norm(r_ij) * (self.time_scale  ** 2 / atomic_mass(element_list[idx_i]))
                    h_k = -2 * cos * (-1 * cos * r_kj / np.linalg.norm(r_kj) + r_ij / np.linalg.norm(r_ij)) / np.linalg.norm(r_kj) * (self.time_scale  ** 2 / atomic_mass(element_list[idx_k]))
                    h_j = -1 * (h_i + h_k)
                    LAMBDA = 2 * cos * (((np.sum(-1 * r_ij * (h_j - h_k)) + np.sum(-1 * r_kj * (h_j - h_i))) / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj))) -1 * ((np.sum(-1 * r_ij * (h_j - h_i)) / np.linalg.norm(r_ij) ** 2) + (np.sum(-1 * r_kj * (h_j - h_k)) / np.linalg.norm(r_kj) ** 2)) * cos)
                    
                    new_momentum_list[idx_i] = h_i * self.time_scale
                    new_momentum_list[idx_j] = h_j * self.time_scale
                    new_momentum_list[idx_k] = h_k * self.time_scale

                    new_geometry[idx_i] -= 1e+5 * LAMBDA * h_i
                    new_geometry[idx_j] -= 1e+5 * LAMBDA * h_j
                    new_geometry[idx_k] -= 1e+5 * LAMBDA * h_k
                    
            
                else: # dihedral angle
                    # ref.:J. Chem. Phys. 133, 034114 (2010)
                    idx_a = constraint[1] - 1
                    idx_b = constraint[2] - 1
                    idx_c = constraint[3] - 1
                    idx_d = constraint[4] - 1
                    constraint_dihedral_angle = np.deg2rad(constraint[0])
                    r_ba = new_geometry[idx_b] - new_geometry[idx_a]
                    r_bc = new_geometry[idx_b] - new_geometry[idx_c]
                    r_cd = new_geometry[idx_c] - new_geometry[idx_d]
                    a = r_ba -1 * (np.sum(r_ba * r_bc / np.linalg.norm(r_bc)) * r_bc / np.linalg.norm(r_bc))
                    b = r_cd -1 * (np.sum(r_cd * r_bc / np.linalg.norm(r_bc)) * r_bc / np.linalg.norm(r_bc))
                    cos = np.sum(a / np.linalg.norm(a) * b / np.linalg.norm(b))
                    constraint_cos = np.cos(constraint_dihedral_angle)
                    check_convergence = abs(cos ** 2 - constraint_cos ** 2)
                    
                    if check_convergence < self.convergent_criterion:
                        continue
                    isconverged = False
                    h_a = 2 * cos * (1 / (np.linalg.norm(a))) * (b / np.linalg.norm(b) -1 * cos * a / np.linalg.norm(a)) * (self.time_scale  ** 2 / atomic_mass(element_list[idx_a]))
                    h_d = 2 * cos * (1 / (np.linalg.norm(b))) * (a / np.linalg.norm(a) -1 * cos * b / np.linalg.norm(b)) * (self.time_scale  ** 2 / atomic_mass(element_list[idx_d]))
                    h_b = 2 * cos * (h_a / (2 * cos) * ((np.sum(r_ba * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc)) -1) + h_d / (2 * cos) * (np.sum(r_cd * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc)))  * (self.time_scale  ** 2 / atomic_mass(element_list[idx_b]))
                    h_c = 2 * cos * (-1 * h_d / (2 * cos) * ((np.sum(r_cd * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc)) -1) -1* h_a / (2 * cos) * (np.sum(r_ba * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc))) * (self.time_scale  ** 2 / atomic_mass(element_list[idx_c]))
                    cross_r_ab_r_bc = np.cross(-1*r_ba, r_bc)
                    cross_r_cd_h_bc = np.cross(r_cd, (h_b - h_c))
                    cross_h_cd_r_bc = np.cross((h_c - h_d), r_bc)
                    cross_r_bc_r_cd = np.cross(r_bc, r_cd)
                    cross_r_bc_h_ab = np.cross(r_bc, (h_a - h_b))
                    cross_h_bc_r_ab = np.cross((h_b - h_c), -1*r_ba)
                    
                    
                    LAMBDA = -2 * cos * (((np.sum(cross_r_ab_r_bc * (cross_r_cd_h_bc + cross_h_cd_r_bc)) + np.sum(cross_r_bc_r_cd * (cross_r_bc_h_ab + cross_h_bc_r_ab))) / (np.linalg.norm(cross_r_ab_r_bc) * np.linalg.norm(cross_r_bc_r_cd))) -1 * ((np.sum(cross_r_ab_r_bc * (cross_r_bc_h_ab + cross_h_bc_r_ab))/np.linalg.norm(cross_r_ab_r_bc) ** 2) + (np.sum(cross_r_bc_r_cd * (cross_r_cd_h_bc + cross_h_cd_r_bc))/np.linalg.norm(cross_r_bc_r_cd) ** 2)) * cos)
                    new_momentum_list[idx_a] = h_a * self.time_scale
                    new_momentum_list[idx_b] = h_b * self.time_scale
                    new_momentum_list[idx_c] = h_c * self.time_scale
                    new_momentum_list[idx_d] = h_d * self.time_scale

                    new_geometry[idx_a] -= 1e+7 * LAMBDA * h_a
                    new_geometry[idx_b] -= 1e+7 * LAMBDA * h_b
                    new_geometry[idx_c] -= 1e+7 * LAMBDA * h_c
                    new_geometry[idx_d] -= 1e+7 * LAMBDA * h_d
                    
            if isconverged:
                print("converged!!! (SHAKE)")
                break
        else:
            print("not converged... (SHAKE)")
        
        return new_geometry, new_momentum_list
    
class GradientSHAKE:
    def __init__(self, constraints=[]):
        #ref.:J Comput Chem 1995, 16 (11), 1351–1356.
        self.convergent_criterion = 1e-5
        self.maxiter = 100000
        self.constraint_condition = constraints[0] + constraints[1] + constraints[2]
                
    def run_grad(self, prev_geom_num_list, gradient_list):
        new_gradient = gradient_list
        #Gradient SHAKE
        for iter in range(self.maxiter):
            isconverged = True
            for constraint in self.constraint_condition:
                if len(constraint) == 3: # bond
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    constraint_distance = constraint[0] / UnitValueLib().bohr2angstroms
                    prev_r_ij = prev_geom_num_list[idx_i] - prev_geom_num_list[idx_j]
                    relative_force =  gradient_list[idx_i] - gradient_list[idx_j]
                    delta = np.sum(relative_force * prev_r_ij)
                    delta_2 = abs(delta * 0.01 / constraint_distance)
                    
                    if delta_2 < self.convergent_criterion:
                        continue
                    isconverged = False
                    eta_ij = delta / (2 * constraint_distance ** 2)

                    new_gradient[idx_i] -=  eta_ij  * prev_r_ij
                    new_gradient[idx_j] +=  eta_ij  * prev_r_ij

                    
                elif len(constraint) == 4: # angle
                    print("Gradient SHAKE for angle is not implemented...")
                    """
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    idx_k = constraint[3] - 1
                    constraint_angle = np.deg2rad(constraint[0])
                    prev_r_ij = prev_geom_num_list[idx_i] - prev_geom_num_list[idx_j]
                    prev_r_kj = prev_geom_num_list[idx_k] - prev_geom_num_list[idx_j]
                    relative_force_ij =  gradient_list[idx_i] - gradient_list[idx_j]
                    relative_force_kj =  gradient_list[idx_k] - gradient_list[idx_j]
                    inner_product_r_ij_r_kj = np.sum(prev_r_ij * prev_r_kj)
                    cos = inner_product_r_ij_r_kj / (np.linalg.norm(prev_r_ij) * np.linalg.norm(prev_r_kj))
                    constraint_cos = np.cos(constraint_angle)
                    
                    delta_ij = np.sum(relative_force_ij * prev_r_ij)
                    delta_kj = np.sum(relative_force_kj * prev_r_kj)
                    delta_2 = abs((delta_ij + delta_kj) * 1e-18/ constraint_cos)
                    #print(delta_2)
                    if delta_2 < self.convergent_criterion:
                        continue
                    isconverged = False
                    eta_ij = delta_ij / (2 * constraint_cos ** 2 + 1e+8)
                    eta_kj = delta_kj / (2 * constraint_cos ** 2 + 1e+8)
                    new_gradient[idx_i] -=  eta_ij  * prev_r_ij
                    new_gradient[idx_k] -=  eta_kj  * prev_r_kj
                    new_gradient[idx_j] +=  eta_ij  * prev_r_ij + eta_kj  * prev_r_kj
                    """
                else: # dihedral angle
                    print("Gradient SHAKE for dihedral angle is not implemented...")
                    
            if isconverged:
                print("converged!!! (Gradient SHAKE)")
                break
        else:
            print("not converged... (Gradient SHAKE)")
        return new_gradient
    
    def run_coord(self, prev_geom_num_list, geom_num_list, element_list):
        #SHAKE for energy minimalization
        new_geometry = geom_num_list
        for iter in range(self.maxiter):
            isconverged = True
            for constraint in self.constraint_condition:
                if len(constraint) == 3: # bond
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    constraint_distance = constraint[0] / UnitValueLib().bohr2angstroms
                    r_ij = new_geometry[idx_i] - new_geometry[idx_j]
                    check_convergence = abs(constraint_distance - np.linalg.norm(r_ij))
                    if check_convergence < self.convergent_criterion:
                        continue
                    isconverged = False
                    prev_r_ij = prev_geom_num_list[idx_i] - prev_geom_num_list[idx_j]
                    g_ij = (np.linalg.norm(r_ij) ** 2 - constraint_distance ** 2) / (2 * (np.sum(r_ij * prev_r_ij)) * (1/atomic_mass(element_list[idx_i]) + 1/atomic_mass(element_list[idx_j])))
                    new_geometry[idx_i] -=  g_ij / atomic_mass(element_list[idx_i]) * prev_r_ij
                    new_geometry[idx_j] +=  g_ij / atomic_mass(element_list[idx_j]) * prev_r_ij

                    
                elif len(constraint) == 4: # angle
                    # ref.:J. Chem. Phys. 133, 034114 (2010)
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    idx_k = constraint[3] - 1
                    constraint_angle = np.deg2rad(constraint[0])
                    r_ij = new_geometry[idx_i] - new_geometry[idx_j]
                    r_kj = new_geometry[idx_k] - new_geometry[idx_j]
                    inner_product_r_ij_r_kj = np.sum(r_ij * r_kj)
                    cos = inner_product_r_ij_r_kj / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj))
                    constraint_cos = np.cos(constraint_angle)
                    check_convergence = abs(cos ** 2 - constraint_cos ** 2)
                    #print(check_convergence)
                    if check_convergence < self.convergent_criterion:
                        
                        continue
                    isconverged = False
                    h_i = -2 * cos * (-1 * cos * r_ij / np.linalg.norm(r_ij) + r_kj / np.linalg.norm(r_kj)) / np.linalg.norm(r_ij) * (1.0 / atomic_mass(element_list[idx_i]))
                    h_k = -2 * cos * (-1 * cos * r_kj / np.linalg.norm(r_kj) + r_ij / np.linalg.norm(r_ij)) / np.linalg.norm(r_kj) * (1.0 / atomic_mass(element_list[idx_k]))
                    h_j = -1 * (h_i + h_k)
                    LAMBDA = 2 * cos * (((np.sum(-1 * r_ij * (h_j - h_k)) + np.sum(-1 * r_kj * (h_j - h_i))) / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj))) -1 * ((np.sum(-1 * r_ij * (h_j - h_i)) / np.linalg.norm(r_ij) ** 2) + (np.sum(-1 * r_kj * (h_j - h_k)) / np.linalg.norm(r_kj) ** 2)) * cos)
                    
                    new_geometry[idx_i] -= 1e+1 * LAMBDA * h_i
                    new_geometry[idx_j] -= 1e+1 * LAMBDA * h_j
                    new_geometry[idx_k] -= 1e+1 * LAMBDA * h_k
            
                else: # dihedral angle
                    # ref.:J. Chem. Phys. 133, 034114 (2010)
                    idx_a = constraint[1] - 1
                    idx_b = constraint[2] - 1
                    idx_c = constraint[3] - 1
                    idx_d = constraint[4] - 1
                    constraint_dihedral_angle = np.deg2rad(constraint[0])
                    r_ba = new_geometry[idx_b] - new_geometry[idx_a]
                    r_bc = new_geometry[idx_b] - new_geometry[idx_c]
                    r_cd = new_geometry[idx_c] - new_geometry[idx_d]
                    a = r_ba -1 * (np.sum(r_ba * r_bc / np.linalg.norm(r_bc)) * r_bc / np.linalg.norm(r_bc))
                    b = r_cd -1 * (np.sum(r_cd * r_bc / np.linalg.norm(r_bc)) * r_bc / np.linalg.norm(r_bc))
                    cos = np.sum(a / np.linalg.norm(a) * b / np.linalg.norm(b))
                    constraint_cos = np.cos(constraint_dihedral_angle)
                    check_convergence = abs(cos ** 2 - constraint_cos ** 2)
                    
                    if check_convergence < self.convergent_criterion:
                        continue
                    isconverged = False
                    h_a = 2 * cos * (1 / (np.linalg.norm(a))) * (b / np.linalg.norm(b) -1 * cos * a / np.linalg.norm(a)) * (1.0 / atomic_mass(element_list[idx_a]))
                    h_d = 2 * cos * (1 / (np.linalg.norm(b))) * (a / np.linalg.norm(a) -1 * cos * b / np.linalg.norm(b)) * (1.0 / atomic_mass(element_list[idx_d]))
                    h_b = 2 * cos * (h_a / (2 * cos) * ((np.sum(r_ba * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc)) -1) + h_d / (2 * cos) * (np.sum(r_cd * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc)))  * (1.0 / atomic_mass(element_list[idx_b]))
                    h_c = 2 * cos * (-1 * h_d / (2 * cos) * ((np.sum(r_cd * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc)) -1) -1* h_a / (2 * cos) * (np.sum(r_ba * r_bc / np.linalg.norm(r_bc)) / np.linalg.norm(r_bc))) * (1.0 / atomic_mass(element_list[idx_c]))
                    cross_r_ab_r_bc = np.cross(-1*r_ba, r_bc)
                    cross_r_cd_h_bc = np.cross(r_cd, (h_b - h_c))
                    cross_h_cd_r_bc = np.cross((h_c - h_d), r_bc)
                    cross_r_bc_r_cd = np.cross(r_bc, r_cd)
                    cross_r_bc_h_ab = np.cross(r_bc, (h_a - h_b))
                    cross_h_bc_r_ab = np.cross((h_b - h_c), -1*r_ba)
                    
                    
                    LAMBDA = -2 * cos * (((np.sum(cross_r_ab_r_bc * (cross_r_cd_h_bc + cross_h_cd_r_bc)) + np.sum(cross_r_bc_r_cd * (cross_r_bc_h_ab + cross_h_bc_r_ab))) / (np.linalg.norm(cross_r_ab_r_bc) * np.linalg.norm(cross_r_bc_r_cd))) -1 * ((np.sum(cross_r_ab_r_bc * (cross_r_bc_h_ab + cross_h_bc_r_ab))/np.linalg.norm(cross_r_ab_r_bc) ** 2) + (np.sum(cross_r_bc_r_cd * (cross_r_cd_h_bc + cross_h_cd_r_bc))/np.linalg.norm(cross_r_bc_r_cd) ** 2)) * cos)


                    new_geometry[idx_a] -= 1e+3 * LAMBDA * h_a
                    new_geometry[idx_b] -= 1e+3 * LAMBDA * h_b
                    new_geometry[idx_c] -= 1e+3 * LAMBDA * h_c
                    new_geometry[idx_d] -= 1e+3 * LAMBDA * h_d
            if isconverged:
                print("converged!!! (SHAKE for energy minimalization)")
                break
        else:
            print("not converged... (SHAKE for energy minimalization)")
        
        
        return new_geometry


class ProjectOutConstrain:
    def __init__(self, constraint_name, constraint_atoms_list, constraint_constant=[]):
        self.constraint_name = constraint_name
        self.constraint_atoms_list = []
        for i in range(len(constraint_atoms_list)):
            tmp_list = []
            for j in range(len(constraint_atoms_list[i])):
                tmp_list.append(int(constraint_atoms_list[i][j]))
            self.constraint_atoms_list.append(tmp_list)
            
        self.constraint_constant = constraint_constant
        self.iteration = 1
        self.init_tag = True
        self.spring_const = 0.0
        self.projection_vec = None
        self.arbitrary_proj_vec = None
        
        # --- Advanced Adaptive Stiffness Parameters ---
        self.reference_scale = None  
        self.alpha_smoothing = 0.7   
        self.current_step = 0
        
        # --- Multi-Secant Like History Storage ---
        # Store past Q vectors to approximate constraint curvature
        self.history_size = 5
        self.q_history = deque(maxlen=self.history_size)
        
        # Feedback parameter
        self.last_shake_iter = 0
        
        return

    def initialize(self, geom_num_list, **kwargs):
        # ... [Initialize logic remains the same] ...
        tmp_init_constraint = []
        tmp_projection_vec = []
        tmp_arbitrary_proj_vec = []
        
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0] - 1]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1] - 1]
                init_bond_dist = calc_bond_length_from_vec(vec_1, vec_2) 
                tmp_init_constraint.append(init_bond_dist)
            elif self.constraint_name[i] == "fbond":
                divide_index = self.constraint_atoms_list[i][-1]
                fragm_1 = np.array(self.constraint_atoms_list[i][:divide_index], dtype=np.int32) - 1
                fragm_2 = np.array(self.constraint_atoms_list[i][divide_index:], dtype=np.int32) - 1
                vec_1 = np.mean(geom_num_list[fragm_1], axis=0)
                vec_2 = np.mean(geom_num_list[fragm_2], axis=0)
                init_bond_dist = calc_bond_length_from_vec(vec_1, vec_2) 
                tmp_init_constraint.append(init_bond_dist)  
            elif self.constraint_name[i] == "angle":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0] - 1] - geom_num_list[self.constraint_atoms_list[i][1] - 1]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][2] - 1] - geom_num_list[self.constraint_atoms_list[i][1] - 1]
                init_angle = calc_angle_from_vec(vec_1, vec_2)
                tmp_init_constraint.append(init_angle)
            elif self.constraint_name[i] == "dihedral":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0] - 1] - geom_num_list[self.constraint_atoms_list[i][1] - 1]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1] - 1] - geom_num_list[self.constraint_atoms_list[i][2] - 1]
                vec_3 = geom_num_list[self.constraint_atoms_list[i][2] - 1] - geom_num_list[self.constraint_atoms_list[i][3] - 1]
                init_dihedral = calc_dihedral_angle_from_vec(vec_1, vec_2, vec_3)
                tmp_init_constraint.append(init_dihedral)
            elif self.constraint_name[i] == "x":
                tmp_init_constraint.append(geom_num_list[self.constraint_atoms_list[i][0] - 1][0])
            elif self.constraint_name[i] == "y":
                tmp_init_constraint.append(geom_num_list[self.constraint_atoms_list[i][0] - 1][1])
            elif self.constraint_name[i] == "z":
                tmp_init_constraint.append(geom_num_list[self.constraint_atoms_list[i][0] - 1][2])
            elif self.constraint_name[i] == "rot":
                tmp_init_constraint.append(geom_num_list)
            elif self.constraint_name[i] == "eigvec":
                mode_index = int(self.constraint_atoms_list[i][0])
                if "hessian" in kwargs:
                    hessian = copy.copy(kwargs["hessian"])
                    eigvals, eigvecs = np.linalg.eigh(hessian)
                    valid_indices = np.where(np.abs(eigvals) > 1.0e-10)[0]
                    sorted_indices = valid_indices[np.argsort(eigvals[valid_indices])]
                    target_mode = sorted_indices[mode_index]
                    init_eigvec = eigvecs[:, target_mode]
                    tmp_init_constraint.append(geom_num_list)
                    tmp_projection_vec.append(init_eigvec)
                else:
                    raise Exception("error (Hessian is required for eigvec constraint)")
            elif self.constraint_name[i] == "atoms_pair":
                atom_label_1 = self.constraint_atoms_list[i][0] - 1
                atom_label_2 = self.constraint_atoms_list[i][1] - 1
                vec = np.zeros_like(geom_num_list)
                vec[atom_label_1] = -geom_num_list[atom_label_1] + geom_num_list[atom_label_2]
                vec[atom_label_2] = -geom_num_list[atom_label_2] + geom_num_list[atom_label_1]
                norm_vec = np.linalg.norm(vec)
                if norm_vec < 1.0e-10:
                    raise Exception("error (the distance between the pair atoms is too small)")
                unit_vec = vec / norm_vec
                unit_vec = unit_vec.reshape(-1, 1)
                tmp_arbitrary_proj_vec.append(unit_vec)
                tmp_init_constraint.append(geom_num_list)
            else:
                raise Exception("error (invaild input of constraint conditions)")
      
        self.projection_vec = tmp_projection_vec
       
        def gram_schmidt(vectors):
            ortho = []
            for v in vectors:
                w = v.copy()
                for u in ortho:
                    w -= np.dot(u.T, w) * u
                norm = np.linalg.norm(w)
                if norm > 1e-10:
                    ortho.append(w / norm)
            return ortho

        self.arbitrary_proj_vec = gram_schmidt(tmp_arbitrary_proj_vec)
        
        if self.init_tag:
            if len(self.constraint_constant) == 0:
                self.init_constraint = tmp_init_constraint
            else:
                self.init_constraint = []
                for i in range(len(self.constraint_constant)):
                    if self.constraint_name[i] in ["bond", "fbond", "x", "y", "z"]:
                        self.init_constraint.append(self.constraint_constant[i] / UnitValueLib().bohr2angstroms)
                    elif self.constraint_name[i] in ["angle", "dihedral"]:
                        self.init_constraint.append(np.deg2rad(self.constraint_constant[i]))
                    elif self.constraint_name[i] in ["rot", "eigvec", "atoms_pair"]:
                        self.init_constraint.append(geom_num_list)
                    else:
                        raise Exception("error (invaild input of constraint conditions)")
                
            self.init_tag = False
       
        return tmp_init_constraint

    def adjust_init_coord(self, coord, hessian=None):
        if self.init_tag or not hasattr(self, 'init_constraint'):
            self.initialize(coord, hessian=hessian)
        print("Adjusting initial coordinates... (SHAKE-like method) ")
        jiter = 10000
        shake_like_method_threshold = 1.0e-10
        
        # ... [Special constraint projection] ...
        for i_constrain in range(len(self.constraint_name)):
            if self.constraint_name[i_constrain] == "rot":
                print("fix fragment rotation... (Experimental Implementation)")     
                atom_label = self.constraint_atoms_list[i_constrain]
                init_coord = self.init_constraint[i_constrain]
                coord = rotate_partial_struct(coord, init_coord, atom_label)
            elif self.constraint_name[i_constrain] == "eigvec":
                print("projecting out eigenvector... (Experimental Implementation)")     
                init_coord = self.init_constraint[i_constrain]
                coord, _ = Calculationtools().kabsch_algorithm(coord, init_coord)
            elif self.constraint_name[i_constrain] == "atoms_pair":
                print("projecting out translation along the vector between the pair atoms... (Experimental Implementation)")     
                init_coord = self.init_constraint[i_constrain]
                coord, _ = Calculationtools().kabsch_algorithm(coord, init_coord)

        final_iter_count = 0
        for jter in range(jiter): 
            final_iter_count = jter
            for i_constrain in range(len(self.constraint_name)):
                if self.constraint_name[i_constrain] == "bond":
                    atom_label_1 = self.constraint_atoms_list[i_constrain][0] - 1
                    atom_label_2 = self.constraint_atoms_list[i_constrain][1] - 1
                    coord = change_atom_distance_both_side(coord, atom_label_1, atom_label_2, self.init_constraint[i_constrain])
                elif self.constraint_name[i_constrain] == "fbond":
                    divide_index = self.constraint_atoms_list[i_constrain][-1]
                    fragm_1 = np.array(self.constraint_atoms_list[i_constrain][:divide_index], dtype=np.int32) - 1
                    fragm_2 = np.array(self.constraint_atoms_list[i_constrain][divide_index:], dtype=np.int32) - 1
                    coord = change_fragm_distance_both_side(coord, fragm_1, fragm_2, self.init_constraint[i_constrain])
                elif self.constraint_name[i_constrain] == "angle":
                    atom_label_1 = self.constraint_atoms_list[i_constrain][0] - 1
                    atom_label_2 = self.constraint_atoms_list[i_constrain][1] - 1
                    atom_label_3 = self.constraint_atoms_list[i_constrain][2] - 1
                    coord = change_bond_angle_both_side(coord, atom_label_1, atom_label_2, atom_label_3, self.init_constraint[i_constrain])
                elif self.constraint_name[i_constrain] == "dihedral":
                    atom_label_1 = self.constraint_atoms_list[i_constrain][0] - 1
                    atom_label_2 = self.constraint_atoms_list[i_constrain][1] - 1
                    atom_label_3 = self.constraint_atoms_list[i_constrain][2] - 1
                    atom_label_4 = self.constraint_atoms_list[i_constrain][3] - 1
                    coord = change_torsion_angle_both_side(coord, atom_label_1, atom_label_2, atom_label_3, atom_label_4, self.init_constraint[i_constrain])
                elif self.constraint_name[i_constrain] == "x":
                    atom_label = self.constraint_atoms_list[i_constrain][0] - 1 
                    coord[atom_label][0] = self.init_constraint[i_constrain] 
                elif self.constraint_name[i_constrain] == "y":
                    atom_label = self.constraint_atoms_list[i_constrain][0] - 1 
                    coord[atom_label][1] = self.init_constraint[i_constrain]
                elif self.constraint_name[i_constrain] == "z":
                    atom_label = self.constraint_atoms_list[i_constrain][0] - 1 
                    coord[atom_label][2] = self.init_constraint[i_constrain]
        
            tmp_current_coord = self.initialize(coord, hessian=hessian)
            current_coord = []
            tmp_init_constraint = []
            for i_constrain in range(len(self.constraint_name)):
                if self.constraint_name[i_constrain] not in ["rot", "eigvec", "atoms_pair"]:
                    current_coord.append(tmp_current_coord[i_constrain])
                    tmp_init_constraint.append(self.init_constraint[i_constrain])
            
            current_coord = np.array(current_coord)
            tmp_init_constraint = np.array(tmp_init_constraint)
            if len(current_coord) > 0:
                if np.linalg.norm(current_coord - tmp_init_constraint) < shake_like_method_threshold:
                    print(f"Adjusted!!! : ITR. {jter}")
                    break
            else:
                break
        
        self.last_shake_iter = final_iter_count
        return coord

    def _get_all_constraint_vectors(self, coord):
        natom = len(coord)
        B_vectors = []
        
        projection_vec_count = 0
        arbitrary_vec_count = 0

        for i_constrain in range(len(self.constraint_name)):
            vec = None
            if self.constraint_name[i_constrain] == "bond":
                atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1]]
                vec = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_distance).detach().numpy().reshape(-1)
                
            elif self.constraint_name[i_constrain] == "fbond":
                divide_index = self.constraint_atoms_list[i_constrain][-1]
                fragm_1 = torch.tensor(self.constraint_atoms_list[i_constrain][:divide_index], dtype=torch.int64)
                fragm_2 = torch.tensor(self.constraint_atoms_list[i_constrain][divide_index:], dtype=torch.int64)
                atom_label = [fragm_1, fragm_2]
                vec = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_fragm_distance).detach().numpy().reshape(-1)
            
            elif self.constraint_name[i_constrain] == "angle":
                atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1], self.constraint_atoms_list[i_constrain][2]]
                vec = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_angle).detach().numpy().reshape(-1)
            
            elif self.constraint_name[i_constrain] == "dihedral":
                atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1], self.constraint_atoms_list[i_constrain][2], self.constraint_atoms_list[i_constrain][3]]
                vec = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_dihedral_angle).detach().numpy().reshape(-1)
            
            elif self.constraint_name[i_constrain] == "x":
                atom_label = self.constraint_atoms_list[i_constrain][0]
                vec = np.zeros(3*natom)
                vec[3*(atom_label - 1) + 0] = 1.0
            
            elif self.constraint_name[i_constrain] == "y":
                atom_label = self.constraint_atoms_list[i_constrain][0]
                vec = np.zeros(3*natom)
                vec[3*(atom_label - 1) + 1] = 1.0
            
            elif self.constraint_name[i_constrain] == "z":
                atom_label = self.constraint_atoms_list[i_constrain][0]
                vec = np.zeros(3*natom)
                vec[3*(atom_label - 1) + 2] = 1.0
                
            elif self.constraint_name[i_constrain] == "rot":
                atom_label = self.constraint_atoms_list[i_constrain]
                rot_B = constract_partial_rot_B_mat(coord, atom_label) 
                for row in rot_B:
                    B_vectors.append(row)
                vec = None 

            elif self.constraint_name[i_constrain] == "eigvec":
                if projection_vec_count < len(self.projection_vec):
                    vec = self.projection_vec[projection_vec_count]
                    projection_vec_count += 1
                
            elif self.constraint_name[i_constrain] == "atoms_pair":
                if arbitrary_vec_count < len(self.arbitrary_proj_vec):
                    vec = self.arbitrary_proj_vec[arbitrary_vec_count].reshape(-1)
                    arbitrary_vec_count += 1

            if vec is not None:
                B_vectors.append(vec)
                
        if len(B_vectors) == 0:
            return None
            
        return np.array(B_vectors)

    def _get_orthonormal_basis(self, coord):
        """
        Uses SVD to find the orthonormal basis Q for constraints.
        """
        B_mat = self._get_all_constraint_vectors(coord)
        if B_mat is None: return None
        try:
            U, S, Vt = np.linalg.svd(B_mat.T, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        threshold = 1e-6
        rank = np.sum(S > threshold)
        if rank == 0: return None
        Q = U[:, :rank]
        return Q

    def calc_project_out_grad(self, coord, grad):
        """
        Projects gradient + Re-orthogonalization (Purification)
        """
        natom = len(coord)
        grad_vec = grad.reshape(3*natom, 1)
        
        Q = self._get_orthonormal_basis(coord)
        if Q is None: return grad
            
        # P = I - Q Q^T
        qt_g = np.dot(Q.T, grad_vec)
        grad_proj = grad_vec - np.dot(Q, qt_g)
        
        # Purification
        qt_g_residual = np.dot(Q.T, grad_proj)
        grad_proj = grad_proj - np.dot(Q, qt_g_residual)
        
        return grad_proj.reshape(natom, 3)
    
    def calc_project_out_hess(self, coord, grad, hessian):
        """
        Multi-Secant Like Subspace Projection Strategy.
        Constructs a stiffness wall using:
        1. Current Q (Hard Wall)
        2. Orthogonalized History of Qs (Curvature Guiding Wall)
        """
        natom = len(coord)
        Q = self._get_orthonormal_basis(coord)
        
        if Q is None:
            return hessian
            
        # --- 1. Update History ---
        # Add current Q to history for future steps
        self.q_history.append(Q)
        
        # --- 2. Construct Augmented Projector ---
        # Start with current Q (Basis of current constraints)
        # We need to find "Where the constraints WERE but ARE NOT ANYMORE".
        # This difference represents the curvature of the constraint surface.
        
        # List of basis vectors to exclude (Current + History Residuals)
        exclusion_vectors = []
        
        # Add current basis vectors
        for i in range(Q.shape[1]):
            exclusion_vectors.append(Q[:, i])
            
        # Add orthogonal components from history (Gram-Schmidt style)
        for Q_hist in self.q_history:
            # We treat Q_hist as a block of vectors
            for i in range(Q_hist.shape[1]):
                vec = Q_hist[:, i]
                
                # Orthogonalize against all vectors currently in exclusion list
                for basis_vec in exclusion_vectors:
                    proj = np.dot(vec, basis_vec)
                    vec = vec - proj * basis_vec
                
                # If residual is significant, it's a new direction (curvature)
                norm = np.linalg.norm(vec)
                if norm > 0.1: # Threshold to ignore noise/duplicates
                    vec = vec / norm
                    exclusion_vectors.append(vec)
        
        # Now exclusion_vectors contains [Q_current, Q_history_perp, ...]
        # We build two projectors:
        # P_hard: Defined by Q_current (Stiffness: High)
        # P_soft: Defined by Q_history_perp (Stiffness: Medium)
        
        num_hard = Q.shape[1]
        
        P_hard = np.zeros_like(hessian)
        P_soft = np.zeros_like(hessian)
        
        for idx, vec in enumerate(exclusion_vectors):
            outer_prod = np.outer(vec, vec)
            if idx < num_hard:
                P_hard += outer_prod
            else:
                P_soft += outer_prod
                
        # --- 3. Physical Projection ---
        # We strictly remove curvature along CURRENT constraints
        # P_hard acts as the Projector P = Q Q^T
        Proj_H = np.dot(P_hard, hessian)
        H_Proj = np.dot(hessian, P_hard)
        Proj_H_Proj = np.dot(P_hard, H_Proj)
        
        PHP = hessian - Proj_H - H_Proj + Proj_H_Proj
        
        # --- 4. Adaptive Stiffness Calculation ---
        self.current_step += 1
        current_max_diag = np.max(np.abs(np.diag(hessian)))
        current_scale = max(current_max_diag, 0.5)

        if self.reference_scale is None:
            self.reference_scale = current_scale
        else:
            self.reference_scale = (self.alpha_smoothing * self.reference_scale + 
                                   (1.0 - self.alpha_smoothing) * current_scale)
        
        # Multipliers

        hard_mult = 100.0
        
        # Soft wall is weaker (e.g., 20% of hard wall)
        # This guides the optimizer without locking it up
        soft_mult = hard_mult * 0.2
            
        stiffness_hard = self.reference_scale * hard_mult
        stiffness_soft = self.reference_scale * soft_mult
        
        # --- 5. Final Effective Hessian ---
        # H_eff = PHP + k_hard * P_hard + k_soft * P_soft
        H_eff = PHP + stiffness_hard * P_hard + stiffness_soft * P_soft
        
        return H_eff
    
    def reset_stiffness(self):
        self.reference_scale = None
        self.current_step = 0
        self.q_history.clear()
        self.last_shake_iter = 0



def constract_partial_rot_B_mat(geom_num_list, target_atoms_list):#1-based index
    target_atoms_list = np.array(target_atoms_list, dtype=np.int32) - 1
    center = np.mean(geom_num_list[target_atoms_list], axis=0)
    centroid_geom_num_list = geom_num_list[target_atoms_list] - center
    B_mat = np.zeros((3 * len(target_atoms_list), 3 * len(geom_num_list)))
    for j in range(len(target_atoms_list)):
        i = target_atoms_list[j]
      
        B_mat[3*j][3*i+0] = 0.0
        B_mat[3*j][3*i+1] = centroid_geom_num_list[i][2]
        B_mat[3*j][3*i+2] = -1 * centroid_geom_num_list[i][1]
        B_mat[3*j+1][3*i+0] = -1 * centroid_geom_num_list[i][2]
        B_mat[3*j+1][3*i+1] = 0.0
        B_mat[3*j+1][3*i+2] = centroid_geom_num_list[i][0]
        B_mat[3*j+2][3*i+0] = centroid_geom_num_list[i][1]
        B_mat[3*j+2][3*i+1] = -1 * centroid_geom_num_list[i][0]
        B_mat[3*j+2][3*i+2] = 0.0
    
    return B_mat
    
    
def constract_partial_rot_B_mat_1st_derivative(geom_num_list, target_atoms_list):#1-based index
    return

def rotate_partial_struct(geom_num_list, init_geom_num_list, target_atoms_list):#1-based index
    target_atoms_list = np.array(target_atoms_list, dtype=np.int32) - 1
    center = np.mean(geom_num_list[target_atoms_list], axis=0)
    
    partial_geom_num_list = geom_num_list[target_atoms_list]
    init_partial_geom_num_list = init_geom_num_list[target_atoms_list]
    rotated_partial_geom_num_list, _ = Calculationtools().kabsch_algorithm(partial_geom_num_list, init_partial_geom_num_list)
    rotated_partial_geom_num_list = rotated_partial_geom_num_list + center
    geom_num_list[target_atoms_list] = rotated_partial_geom_num_list
    return geom_num_list