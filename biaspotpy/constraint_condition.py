import numpy as np
import copy
from parameter import atomic_mass, UnitValueLib

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