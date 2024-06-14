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
        self.maxiter = 10000
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
                    new_geometry[idx_j] += g_ij / atomic_mass(element_list[idx_i]) * prev_r_ij
                    new_momentum_list[idx_i] -= g_ij / self.time_scale * prev_r_ij
                    new_momentum_list[idx_j] += g_ij / self.time_scale * prev_r_ij
                    
                elif len(constraint) == 4: # angle
                    # Computer Physics Communications 180(2009)360-364
                    idx_i = constraint[1] - 1
                    idx_j = constraint[2] - 1
                    idx_k = constraint[3] - 1
                    constraint_angle = np.deg2rad(constraint[0])
                    r_ij = new_geometry[idx_i] - new_geometry[idx_j]
                    r_kj = new_geometry[idx_k] - new_geometry[idx_j]
                    inner_product_r_ij_r_kj = np.sum(r_ij * r_kj)
                    cos = inner_product_r_ij_r_kj / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj) + 1e-8)
                    constraint_cos = np.cos(constraint_angle)
                    check_convergence = abs(cos ** 2 - constraint_cos ** 2)
                    
                    if check_convergence < self.convergent_criterion:
                        print(check_convergence)
                        continue
                    isconverged = False
                    
                    
                    
                    d_sigma_d_r_i = 2 * inner_product_r_ij_r_kj * (r_kj * np.linalg.norm(r_ij) ** 2 - r_ij * inner_product_r_ij_r_kj) / (np.linalg.norm(r_ij) ** 4 * np.linalg.norm(r_kj) ** 2 + 1e-8)
                    d_sigma_d_r_k = 2 * inner_product_r_ij_r_kj * (r_ij * np.linalg.norm(r_kj) ** 2 - r_kj * inner_product_r_ij_r_kj) / (np.linalg.norm(r_kj) ** 4 * np.linalg.norm(r_ij) ** 2 + 1e-8)
                    d_sigma_d_r_j = -1 * (d_sigma_d_r_i + d_sigma_d_r_k)
                    new_momentum_list[idx_i] = d_sigma_d_r_i * self.time_scale  ** 2 
                    new_momentum_list[idx_j] = d_sigma_d_r_j * self.time_scale  ** 2
                    new_momentum_list[idx_k] = d_sigma_d_r_k * self.time_scale  ** 2 
                    LAMBDA = 2 * inner_product_r_ij_r_kj * (((np.sum((new_momentum_list[idx_i] - new_momentum_list[idx_j]) * r_kj) * np.sum((new_momentum_list[idx_k] - new_momentum_list[idx_j]) * r_ij))/(np.linalg.norm(r_ij) ** 2 * np.linalg.norm(r_kj) ** 2 + 1e-8)) - ((inner_product_r_ij_r_kj * (np.linalg.norm(r_ij) ** 2 * np.sum((new_momentum_list[idx_k] - new_momentum_list[idx_j]) * r_kj) + np.linalg.norm(r_kj) ** 2 * np.sum((new_momentum_list[idx_i] - new_momentum_list[idx_j]) * r_ij) ))/(np.linalg.norm(r_ij) ** 4 * np.linalg.norm(r_kj) ** 4 + 1e-8)))
                    new_geometry[idx_i] -= 1e+3 * LAMBDA * new_momentum_list[idx_i]
                    new_geometry[idx_j] -= 1e+3 * LAMBDA * new_momentum_list[idx_j]
                    new_geometry[idx_k] -= 1e+3 * LAMBDA * new_momentum_list[idx_k]
                    
            
                else: # dihedral angle
                    print("SHAKE for dihedral angle is not implemented...")
                    raise
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
        self.maxiter = 10000
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
                    raise
            
                else: # dihedral angle
                    print("Gradient SHAKE for dihedral angle is not implemented...")
                    raise 
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
                    new_geometry[idx_j] +=  g_ij / atomic_mass(element_list[idx_i]) * prev_r_ij

                    
                elif len(constraint) == 4: # angle
                    print("SHAKE for energy minimalization for angle is not implemented...")
                    raise
            
                else: # dihedral angle
                    print("SHAKE for energy minimalization for dihedral angle is not implemented...")
                    raise 
            if isconverged:
                print("converged!!! (SHAKE for energy minimalization)")
                break
        else:
            print("not converged... (SHAKE for energy minimalization)")
        
        
        return new_geometry