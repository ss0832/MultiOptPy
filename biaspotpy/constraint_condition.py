import numpy as np
import copy
import torch
from parameter import atomic_mass, UnitValueLib
from calc_tools import (calc_bond_length_from_vec, 
                        calc_angle_from_vec, 
                        calc_dihedral_angle_from_vec, change_atom_distance_both_side, 
                        change_bond_angle_both_side, 
                        change_torsion_angle_both_side,
                        change_fragm_distance_both_side
                        )
from redundant_coordinations import (TorchDerivatives, 
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


class LagrangeConstrain:
    def __init__(self, constraint_name, constraint_atoms_list):
        self.constraint_name = constraint_name
        self.constraint_atoms_list = []
        for i in range(len(constraint_atoms_list)):
            tmp_list = []
            for j in range(len(constraint_atoms_list[i])):
                tmp_list.append(int(constraint_atoms_list[i][j])-1)
            self.constraint_atoms_list.append(tmp_list)
            
        return
    
    def initialize(self, geom_num_list):
        tmp_init_constraint = []
        tmp_init_constraint_grad = []
        tmp_init_constraint_hess = []
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1]]
                init_bond_dist = calc_bond_length_from_vec(vec_1, vec_2) 
                tmp_init_constraint.append(init_bond_dist)
                
                stack_coord = np.vstack([vec_1, vec_2])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)    
                bond_grad = torch.func.jacrev(TorchDerivatives().distance, 0)(tmp_tensor).detach().numpy()
                tmp_init_constraint_grad.append(bond_grad)
                
                bond_hess = torch.func.hessian(TorchDerivatives().distance, 0)(tmp_tensor).detach().numpy()
                bond_hess = bond_hess.reshape(6, 6)
                tmp_init_constraint_hess.append(bond_hess)
                
                 
            elif self.constraint_name[i] == "angle":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][2]] - geom_num_list[self.constraint_atoms_list[i][1]]
                init_angle = calc_angle_from_vec(vec_1, vec_2)
                tmp_init_constraint.append(init_angle)
                
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                angle_grad = torch.func.jacrev(TorchDerivatives().angle, 0)(tmp_tensor).detach().numpy()
                tmp_init_constraint_grad.append(angle_grad)
                
                angle_hess = torch.func.hessian(TorchDerivatives().angle, 0)(tmp_tensor).detach().numpy()
                angle_hess = angle_hess.reshape(9, 9)
                tmp_init_constraint_hess.append(angle_hess)
                
                
            elif self.constraint_name[i] == "dihedral":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1]] - geom_num_list[self.constraint_atoms_list[i][2]]
                vec_3 = geom_num_list[self.constraint_atoms_list[i][2]] - geom_num_list[self.constraint_atoms_list[i][3]]
                init_dihedral = calc_dihedral_angle_from_vec(vec_1, vec_2, vec_3)
                tmp_init_constraint.append(init_dihedral)
                
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]], geom_num_list[self.constraint_atoms_list[i][3]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                dihedral_grad = torch.func.jacrev(TorchDerivatives().dihedral, 0)(tmp_tensor).detach().numpy()
                tmp_init_constraint_grad.append(dihedral_grad)
                
                dihedral_hess = torch.func.hessian(TorchDerivatives().dihedral, 0)(tmp_tensor).detach().numpy()
                dihedral_hess = dihedral_hess.reshape(12, 12)
                tmp_init_constraint_hess.append(dihedral_hess)
                
                
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"
        
        self.init_constraint = tmp_init_constraint
        self.init_constraint_grad = tmp_init_constraint_grad
        self.init_constraint_hess = tmp_init_constraint_hess

        
        return 
    
    def calc_lagrange_constraint_energy(self, geom_num_list, lagrange_lambda_list):
        lagrange_constraint_energy = 0.0
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                r = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                norm_r = np.linalg.norm(r) 
                norm_r_0 = self.init_constraint[i]
                lagrange_constraint_energy += lagrange_lambda_list[i] * (norm_r - norm_r_0) ** 2
            elif self.constraint_name[i] == "angle":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][2]] - geom_num_list[self.constraint_atoms_list[i][1]]
                angle = calc_angle_from_vec(vec_1, vec_2)
                angle_0 = self.init_constraint[i]
                lagrange_constraint_energy += lagrange_lambda_list[i] * (angle - angle_0) ** 2
            elif self.constraint_name[i] == "dihedral":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1]] - geom_num_list[self.constraint_atoms_list[i][2]]
                vec_3 = geom_num_list[self.constraint_atoms_list[i][2]] - geom_num_list[self.constraint_atoms_list[i][3]]
                dihedral = calc_dihedral_angle_from_vec(vec_1, vec_2, vec_3)
                dihedral_0 = self.init_constraint[i]
                lagrange_constraint_energy += lagrange_lambda_list[i] * (dihedral - dihedral_0) ** 2
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"
        print("#####")
        print("lagrange_constraint_energy: ", lagrange_constraint_energy) 
        print("#####")
        return lagrange_constraint_energy
    
    def lagrange_constraint_grad_calc(self, geom_num_list, lagrange_lambda_list):
        tmp_lagrange_constraint_atom_addgrad = np.zeros((len(geom_num_list), 3))
        for i in range(len(self.constraint_name)):
            
            if self.constraint_name[i] == "bond":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]]])
                r_norm = np.linalg.norm(stack_coord[0] - stack_coord[1])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)    
                bond_grad = torch.func.jacrev(TorchDerivatives().distance, 0)(tmp_tensor).detach().numpy()
                
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][0]] += (bond_grad[0] - self.init_constraint_grad[i][0]) * lagrange_lambda_list[i] * 2 * (r_norm - self.init_constraint[i])
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][1]] += (bond_grad[1] - self.init_constraint_grad[i][1]) * lagrange_lambda_list[i] * 2 * (r_norm - self.init_constraint[i])
            elif self.constraint_name[i] == "angle":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                angle_grad = torch.func.jacrev(TorchDerivatives().angle, 0)(tmp_tensor).detach().numpy()
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][0]] += (angle_grad[0] - self.init_constraint_grad[i][0]) * lagrange_lambda_list[i]
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][1]] += (angle_grad[1] - self.init_constraint_grad[i][1]) * lagrange_lambda_list[i]
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][2]] += (angle_grad[2] - self.init_constraint_grad[i][2]) * lagrange_lambda_list[i]
                           
            elif self.constraint_name[i] == "dihedral":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]], geom_num_list[self.constraint_atoms_list[i][3]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                dihedral_grad = torch.func.jacrev(TorchDerivatives().dihedral, 0)(tmp_tensor).detach().numpy()
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][0]] += (dihedral_grad[0] - self.init_constraint_grad[i][0]) * lagrange_lambda_list[i]
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][1]] += (dihedral_grad[1] - self.init_constraint_grad[i][1]) * lagrange_lambda_list[i]
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][2]] += (dihedral_grad[2] - self.init_constraint_grad[i][2]) * lagrange_lambda_list[i]
                tmp_lagrange_constraint_atom_addgrad[self.constraint_atoms_list[i][3]] += (dihedral_grad[3] - self.init_constraint_grad[i][3]) * lagrange_lambda_list[i]
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"    
        
        return tmp_lagrange_constraint_atom_addgrad

    def lagrange_lambda_grad_calc(self, geom_num_list):
        lagrange_lambda_grad_list = []
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                r = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                norm_r = np.linalg.norm(r)
                norm_r_0 = self.init_constraint[i]
                lagrange_lambda_grad_list.append([(norm_r - norm_r_0) ** 2])
            elif self.constraint_name[i] == "angle":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][2]] - geom_num_list[self.constraint_atoms_list[i][1]]
                angle = calc_angle_from_vec(vec_1, vec_2)
                angle_0 = self.init_constraint[i]
                lagrange_lambda_grad_list.append([(angle - angle_0) ** 2])
            elif self.constraint_name[i] == "dihedral":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0]] - geom_num_list[self.constraint_atoms_list[i][1]]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1]] - geom_num_list[self.constraint_atoms_list[i][2]]
                vec_3 = geom_num_list[self.constraint_atoms_list[i][2]] - geom_num_list[self.constraint_atoms_list[i][3]]
                dihedral = calc_dihedral_angle_from_vec(vec_1, vec_2, vec_3)
                dihedral_0 = self.init_constraint[i]
                lagrange_lambda_grad_list.append([(dihedral - dihedral_0) ** 2])
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"
        lagrange_lambda_grad_list = np.array(lagrange_lambda_grad_list)
        print("#####")
        print("lagrange_lambda_grad_list: ", lagrange_lambda_grad_list)
        print("#####")
        return lagrange_lambda_grad_list

    def lagrange_constraint_atom_hess_calc(self, geom_num_list, lagrange_lambda_list):
        langrange_add_hessian = np.zeros((len(geom_num_list) * 3, len(geom_num_list) * 3))
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]]])
                r_norm = np.linalg.norm(stack_coord[0] - stack_coord[1])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)    
                bond_hess = torch.func.hessian(TorchDerivatives().distance, 0)(tmp_tensor).detach().numpy()
                bond_grad = torch.func.jacrev(TorchDerivatives().distance, 0)(tmp_tensor).detach().numpy()
                bond_hess = bond_hess.reshape(6, 6)
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (bond_hess[0:3, 0:3] - self.init_constraint_hess[i][0:3, 0:3]) * lagrange_lambda_list[i] * 2 * (r_norm - self.init_constraint[i]) + 2 * lagrange_lambda_list[i] * 2 * np.dot((bond_grad[0] - self.init_constraint_grad[i]).T, (bond_grad[0] - self.init_constraint_grad[i]))
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (bond_hess[3:6, 3:6] - self.init_constraint_hess[i][3:6, 3:6]) * lagrange_lambda_list[i] * 2 * (r_norm - self.init_constraint[i]) + 2 * lagrange_lambda_list[i] * 2 * np.dot((bond_grad[1] - self.init_constraint_grad[i]).T, (bond_grad[1] - self.init_constraint_grad[i]))
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (bond_hess[0:3, 3:6] - self.init_constraint_hess[i][0:3, 3:6]) * lagrange_lambda_list[i] * 2 * (r_norm - self.init_constraint[i]) + 2 * lagrange_lambda_list[i] * 2 * np.dot((bond_grad[0] - self.init_constraint_grad[i]).T, (bond_grad[1] - self.init_constraint_grad[i]))
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (bond_hess[3:6, 0:3] - self.init_constraint_hess[i][3:6, 0:3]) * lagrange_lambda_list[i] * 2 * (r_norm - self.init_constraint[i]) + 2 * lagrange_lambda_list[i] * 2 * np.dot((bond_grad[1] - self.init_constraint_grad[i]).T, (bond_grad[0] - self.init_constraint_grad[i]))
                
                
    
            elif self.constraint_name[i] == "angle":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                angle_hess = torch.func.hessian(TorchDerivatives().angle, 0)(tmp_tensor).detach().numpy()
                angle_hess = angle_hess.reshape(9, 9)
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (angle_hess[0:3, 0:3] - self.init_constraint_hess[i][0:3, 0:3]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (angle_hess[3:6, 3:6] - self.init_constraint_hess[i][3:6, 3:6]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (angle_hess[6:9, 6:9] - self.init_constraint_hess[i][6:9, 6:9]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (angle_hess[0:3, 3:6] - self.init_constraint_hess[i][0:3, 3:6]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (angle_hess[3:6, 0:3] - self.init_constraint_hess[i][3:6, 0:3]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (angle_hess[3:6, 6:9] - self.init_constraint_hess[i][3:6, 6:9]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (angle_hess[6:9, 3:6] - self.init_constraint_hess[i][6:9, 3:6]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (angle_hess[0:3, 6:9] - self.init_constraint_hess[i][0:3, 6:9]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (angle_hess[6:9, 0:3] - self.init_constraint_hess[i][6:9, 0:3]) * lagrange_lambda_list[i]
                
               
                           
            elif self.constraint_name[i] == "dihedral":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]], geom_num_list[self.constraint_atoms_list[i][3]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                dihedral_hess = torch.func.hessian(TorchDerivatives().dihedral, 0)(tmp_tensor).detach().numpy()
                dihedral_hess = dihedral_hess.reshape(12, 12)
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (dihedral_hess[0:3, 0:3]  - self.init_constraint_hess[i][0:3, 0:3])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (dihedral_hess[3:6, 3:6]  - self.init_constraint_hess[i][3:6, 3:6])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (dihedral_hess[6:9, 6:9]  - self.init_constraint_hess[i][6:9, 6:9])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3, 3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3] +=(dihedral_hess[9:12, 9:12] - self.init_constraint_hess[i][9:12, 9:12]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (dihedral_hess[0:3, 3:6]  - self.init_constraint_hess[i][0:3, 3:6])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (dihedral_hess[3:6, 0:3]  - self.init_constraint_hess[i][3:6, 0:3])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (dihedral_hess[3:6, 6:9]  - self.init_constraint_hess[i][3:6, 6:9])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (dihedral_hess[6:9, 6:9]  - self.init_constraint_hess[i][6:9, 6:9])* lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3, 3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3] += (dihedral_hess[6:9, 9:12] - self.init_constraint_hess[i][6:9, 9:12]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (dihedral_hess[9:12, 6:9] - self.init_constraint_hess[i][9:12, 6:9]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3, 3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3] += (dihedral_hess[0:3, 9:12] - self.init_constraint_hess[i][0:3, 9:12]) * lagrange_lambda_list[i]
                langrange_add_hessian[3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (dihedral_hess[9:12, 0:3] - self.init_constraint_hess[i][9:12, 0:3]) * lagrange_lambda_list[i]
                
            
        return langrange_add_hessian

    def lagrange_constraint_couple_hess_calc(self, geom_num_list):
        coupling_hess = np.zeros((len(self.constraint_name), len(geom_num_list) * 3))
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)    
                r_norm = np.linalg.norm(stack_coord[0] - stack_coord[1])
                bond_couple_hess = torch.func.jacrev(TorchDerivatives().distance, 0)(tmp_tensor).detach().numpy()
                coupling_hess[i, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (bond_couple_hess[0] - self.init_constraint_grad[i][0]) * 2 * (r_norm - self.init_constraint[i])
                coupling_hess[i, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (bond_couple_hess[1] - self.init_constraint_grad[i][1]) * 2 * (r_norm - self.init_constraint[i])
               
            elif self.constraint_name[i] == "angle":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                angle_couple_hess = torch.func.jacrev(TorchDerivatives().angle, 0)(tmp_tensor).detach().numpy()
                coupling_hess[i, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (angle_couple_hess[0] - self.init_constraint_grad[i][0])
                coupling_hess[i, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (angle_couple_hess[1] - self.init_constraint_grad[i][1])
                coupling_hess[i, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (angle_couple_hess[2] - self.init_constraint_grad[i][2])
                
               
                           
            elif self.constraint_name[i] == "dihedral":
                stack_coord = np.vstack([geom_num_list[self.constraint_atoms_list[i][0]], geom_num_list[self.constraint_atoms_list[i][1]], geom_num_list[self.constraint_atoms_list[i][2]], geom_num_list[self.constraint_atoms_list[i][3]]])
                tmp_tensor = torch.tensor(stack_coord, requires_grad=True)
                dihedral_couple_hess = torch.func.jacrev(TorchDerivatives().dihedral, 0)(tmp_tensor).detach().numpy()
                coupling_hess[i, 3*(self.constraint_atoms_list[i][0]):3*(self.constraint_atoms_list[i][0])+3] += (dihedral_couple_hess[0] - self.init_constraint_grad[i][0])
                coupling_hess[i, 3*(self.constraint_atoms_list[i][1]):3*(self.constraint_atoms_list[i][1])+3] += (dihedral_couple_hess[1] - self.init_constraint_grad[i][1])
                coupling_hess[i, 3*(self.constraint_atoms_list[i][2]):3*(self.constraint_atoms_list[i][2])+3] += (dihedral_couple_hess[2] - self.init_constraint_grad[i][2])
                coupling_hess[i, 3*(self.constraint_atoms_list[i][3]):3*(self.constraint_atoms_list[i][3])+3] += (dihedral_couple_hess[3] - self.init_constraint_grad[i][3])
                
                
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"
        return coupling_hess
    
    def make_combined_lagrangian_hess(self, hessian, lagrange_coupling_hessian):
        combine_hessian = np.vstack((hessian, lagrange_coupling_hessian))
        zero_mat = np.zeros((len(self.constraint_name), len(self.constraint_name)))
        
        tmp_coupling_hess = np.vstack((lagrange_coupling_hessian.T, zero_mat))
        
        combine_hessian = np.hstack((combine_hessian, tmp_coupling_hess))
        return  combine_hessian
    
    def make_couple_zero_hess(self, geom_num_list):
        zero_mat = np.zeros((len(self.constraint_name), len(geom_num_list) * 3))
        return zero_mat
    
    def lagrange_init_lambda_calc(self, B_g, geom_num_list):# B_g: (3N, 1)
        lambda_list = [] # (M, 1)
        
        
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                b_mat = partial_stretch_B_matirx(geom_num_list, self.constraint_atoms_list[i][0]+1, self.constraint_atoms_list[i][1]+1)
                d_E_d_r = RedundantInternalCoordinates().cartgrad2RICgrad(B_g, b_mat)[0]
                lambda_list.append(-1*d_E_d_r)
                #lambda_list.append(1.0)
                
                
            elif self.constraint_name[i] == "angle":
                b_mat = partial_bend_B_matrix(geom_num_list, self.constraint_atoms_list[i][0]+1, self.constraint_atoms_list[i][1]+1, self.constraint_atoms_list[i][2]+1)
                d_E_d_theta = RedundantInternalCoordinates().cartgrad2RICgrad(B_g, b_mat)[0]
                lambda_list.append(-1*d_E_d_theta)
                #lambda_list.append(1.0)
                
            elif self.constraint_name[i] == "dihedral":
                b_mat = partial_torsion_B_matrix(geom_num_list, self.constraint_atoms_list[i][0]+1, self.constraint_atoms_list[i][1]+1, self.constraint_atoms_list[i][2]+1, self.constraint_atoms_list[i][3]+1)
                d_E_d_phi = RedundantInternalCoordinates().cartgrad2RICgrad(B_g, b_mat)[0]
                lambda_list.append(-1*d_E_d_phi)
                #lambda_list.append(1.0)
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"
            pass
        
        return lambda_list


class ProjectOutConstrain:
    def __init__(self, constraint_name, constraint_atoms_list):
        self.constraint_name = constraint_name
        self.constraint_atoms_list = []
        for i in range(len(constraint_atoms_list)):
            tmp_list = []
            for j in range(len(constraint_atoms_list[i])):
                tmp_list.append(int(constraint_atoms_list[i][j]))
            self.constraint_atoms_list.append(tmp_list)
            
        self.iteration = 1
        self.init_tag = True
        #self.func_list = None
        self.spring_const = 0.0

        return


    def initialize(self, geom_num_list):
        tmp_init_constraint = []
        #tmp_func_list = []
        for i in range(len(self.constraint_name)):
            if self.constraint_name[i] == "bond":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0] - 1]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1] - 1]
                init_bond_dist = calc_bond_length_from_vec(vec_1, vec_2) 
                tmp_init_constraint.append(init_bond_dist)
                #tmp_func_list.append(torch_calc_distance)
                
            elif self.constraint_name[i] == "fbond":
                divide_index = self.constraint_atoms_list[i][-1]
                fragm_1 = np.array(self.constraint_atoms_list[i][:divide_index], dtype=np.int32) - 1
                fragm_2 = np.array(self.constraint_atoms_list[i][divide_index:], dtype=np.int32) - 1


                vec_1 = np.mean(geom_num_list[fragm_1], axis=0)
                vec_2 = np.mean(geom_num_list[fragm_2], axis=0)
                init_bond_dist = calc_bond_length_from_vec(vec_1, vec_2) 
                tmp_init_constraint.append(init_bond_dist)  
                #tmp_func_list.append(torch_calc_fragm_distance)
                 
            elif self.constraint_name[i] == "angle":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0] - 1] - geom_num_list[self.constraint_atoms_list[i][1] - 1]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][2] - 1] - geom_num_list[self.constraint_atoms_list[i][1] - 1]
                init_angle = calc_angle_from_vec(vec_1, vec_2)
                tmp_init_constraint.append(init_angle)
                #tmp_func_list.append(torch_calc_angle)
                
      
                
            elif self.constraint_name[i] == "dihedral":
                vec_1 = geom_num_list[self.constraint_atoms_list[i][0] - 1] - geom_num_list[self.constraint_atoms_list[i][1] - 1]
                vec_2 = geom_num_list[self.constraint_atoms_list[i][1] - 1] - geom_num_list[self.constraint_atoms_list[i][2] - 1]
                vec_3 = geom_num_list[self.constraint_atoms_list[i][2] - 1] - geom_num_list[self.constraint_atoms_list[i][3] - 1]
                init_dihedral = calc_dihedral_angle_from_vec(vec_1, vec_2, vec_3)
                tmp_init_constraint.append(init_dihedral)
              
                #tmp_func_list.append(torch_calc_dihedral_angle)
                
                   
                
            else:
                print("error")
                raise "error (invaild input of constraint conditions)"
      
        if self.init_tag:
            self.init_constraint = tmp_init_constraint
            #self.func_list = tmp_func_list
            self.init_tag = False
        return tmp_init_constraint

    def adjust_init_coord(self, coord):#coord:Bohr
        print("Adjusting initial coordinates... (SHAKE-like method) ")
        jiter = 10000
        shake_lile_method_threshold = 1.0e-10
        for jter in range(jiter): # SHAKE-like algorithm
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
                    
                else:
                    pass
        
            current_coord = np.array(self.initialize(coord))
            
            if np.linalg.norm(current_coord - np.array(self.init_constraint)) < shake_lile_method_threshold:
                print("Adjusted!!! : ITR. ", jter)
                break
            
        
                
        return coord




    def calc_project_out_grad(self, coord, grad):# B_g: (3N, 1), geom_num_list: (N, 3)
        natom = len(coord)
        prev_proj_grad = copy.copy(grad)
        tmp_grad = copy.copy(grad)



        current_geom = self.initialize(coord)

        for j in range(1):
            for i_constrain in range(len(self.constraint_name)):
                if self.constraint_name[i_constrain] == "bond":
                    atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1]]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_distance).detach().numpy().reshape(1, -1)
                
                elif self.constraint_name[i_constrain] == "fbond":

                    divide_index = self.constraint_atoms_list[i_constrain][-1]
                    fragm_1 = torch.tensor(self.constraint_atoms_list[i_constrain][:divide_index], dtype=torch.int64)
                    fragm_2 = torch.tensor(self.constraint_atoms_list[i_constrain][divide_index:], dtype=torch.int64)
                    atom_label = [fragm_1, fragm_2]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_fragm_distance).detach().numpy().reshape(1, -1)
                
                elif self.constraint_name[i_constrain] == "angle":
                    atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1], self.constraint_atoms_list[i_constrain][2]]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_angle).detach().numpy().reshape(1, -1)
                
                elif self.constraint_name[i_constrain] == "dihedral":
                    atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1], self.constraint_atoms_list[i_constrain][2], self.constraint_atoms_list[i_constrain][3]]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_dihedral_angle).detach().numpy().reshape(1, -1)
                else:
                    print("error")
                    raise "error (invaild input of constraint conditions)"
                
                
                if i_constrain == 0:
                    B_mat = tmp_b_mat        
                else:
                    B_mat = np.vstack((B_mat, tmp_b_mat))
            
            int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)

            constraint_int_grad = []
            for i_constrain in range(len(self.constraint_name)):
                grad = self.spring_const * (current_geom[i_constrain] - self.init_constraint[i_constrain])
                constraint_int_grad.append([grad])
            constraint_int_grad = np.array(constraint_int_grad)
            projection_grad = calc_cart_grad_from_pBmat(-1*int_grad + constraint_int_grad, B_mat)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
            proj_grad = proj_grad.reshape(natom, 3)
            delta_grad = proj_grad - prev_proj_grad
            prev_proj_grad = copy.copy(proj_grad)
            #print("delta_grad: ", np.linalg.norm(delta_grad))
            if np.linalg.norm(delta_grad) < 1.0e-6:
                break
            tmp_grad = copy.copy(proj_grad)
        
        return proj_grad
    
    def calc_project_out_hess(self, coord, grad, hessian):# hessian:(3N, 3N), B_g: (3N, 1), geom_num_list: (N, 3)
        natom = len(coord)


        current_geom = self.initialize(coord)
        prev_proj_hess = copy.copy(hessian)
        tmp_grad = copy.copy(grad)
        tmp_hessian = copy.copy(hessian)
        for j in range(self.iteration):
            for i_constrain in range(len(self.constraint_name)):
                if self.constraint_name[i_constrain] == "bond":
                    atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1]]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_distance).detach().numpy().reshape(1, -1)
                    tmp_b_mat_1st_derivative = torch_B_matrix_derivative(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_distance).detach().numpy()

                elif self.constraint_name[i_constrain] == "fbond":
                    divide_index = self.constraint_atoms_list[i_constrain][-1]
                    fragm_1 = torch.tensor(self.constraint_atoms_list[i_constrain][:divide_index], dtype=torch.int64) 
                    fragm_2 = torch.tensor(self.constraint_atoms_list[i_constrain][divide_index:], dtype=torch.int64) 
                    atom_label = [fragm_1, fragm_2]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_fragm_distance).detach().numpy().reshape(1, -1)
                    tmp_b_mat_1st_derivative = torch_B_matrix_derivative(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_fragm_distance).detach().numpy()
                    
                elif self.constraint_name[i_constrain] == "angle":
                    atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1], self.constraint_atoms_list[i_constrain][2]]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_angle).detach().numpy().reshape(1, -1)
                    tmp_b_mat_1st_derivative = torch_B_matrix_derivative(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_angle).detach().numpy()
                
                elif self.constraint_name[i_constrain] == "dihedral":
                    atom_label = [self.constraint_atoms_list[i_constrain][0], self.constraint_atoms_list[i_constrain][1], self.constraint_atoms_list[i_constrain][2], self.constraint_atoms_list[i_constrain][3]]
                    tmp_b_mat = torch_B_matrix(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_dihedral_angle).detach().numpy().reshape(1, -1)
                    tmp_b_mat_1st_derivative = torch_B_matrix_derivative(torch.tensor(coord, dtype=torch.float64), atom_label, torch_calc_dihedral_angle).detach().numpy()
                else:
                    print("error")
                    raise "error (invaild input of constraint conditions)"
                
                
                if i_constrain == 0:
                    B_mat = tmp_b_mat
                    B_mat_1st_derivative = tmp_b_mat_1st_derivative
                else:
                    B_mat = np.vstack((B_mat, tmp_b_mat))
                    B_mat_1st_derivative = np.concatenate((B_mat_1st_derivative, tmp_b_mat_1st_derivative), axis=2)
            
            
            int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)

            constraint_int_grad = []
            
            for i_constrain in range(len(self.constraint_name)):
                grad = self.spring_const * (current_geom[i_constrain] - self.init_constraint[i_constrain])

                constraint_int_grad.append([grad])
            constraint_int_grad = np.array(constraint_int_grad)

            constraint_int_hess = np.eye((len(self.constraint_name))) * self.spring_const

            int_hess = calc_int_hess_from_pBmat_for_non_stationary_point(tmp_hessian, B_mat, B_mat_1st_derivative, int_grad)#-1*int_hess +
            projection_hess = calc_cart_hess_from_pBmat_for_non_stationary_point( constraint_int_hess, B_mat, B_mat_1st_derivative, int_grad + constraint_int_grad)
            proj_hess = tmp_hessian + projection_hess

            couple_hess = calc_int_cart_coupling_hess_from_pBmat_for_non_stationary_point(tmp_hessian, B_mat, B_mat_1st_derivative, int_grad)
            eff_hess = np.dot(couple_hess.T, np.dot(np.linalg.pinv(int_hess + np.eye((len(int_hess))) * 1e-15), couple_hess))

            proj_hess = proj_hess - eff_hess

            delta_hess = proj_hess - prev_proj_hess
            prev_proj_hess = copy.copy(proj_hess)
            #print("delta_hess: ", np.linalg.norm(delta_hess))
            if np.linalg.norm(delta_hess) < 1.0e-6:
                break
            tmp_hessian = copy.copy(proj_hess)
            
        
        return proj_hess
    
    
    

