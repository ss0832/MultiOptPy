import numpy as np

class OptMESX2:
    def __init__(self):
        # ref.: Theor Chem Acc 99, 95–99 (1998)
        # This reference describes the Gradient Projection method.
        # The implementation has been corrected to follow the standard GP formulation
        # as described in J. Am. Chem. Soc. 2015, 137, 3433-3445 .
        return
    
    def calc_energy(self, energy_1, energy_2):
        tot_energy = (energy_1 + energy_2) / 2.0 
        print("energy_1:", energy_1, "hartree")
        print("energy_2:", energy_2, "hartree")
        print("|energy_1 - energy_2|:", abs(energy_1 - energy_2), "hartree")
        return tot_energy
        
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        # 1. Difference Vector (normalized)
        delta_grad = grad_1 - grad_2
        norm_delta_grad = np.linalg.norm(delta_grad)
        
        if norm_delta_grad < 1e-8:
            dgv_vec = np.zeros_like(delta_grad)
        else:
            dgv_vec = delta_grad / norm_delta_grad
            
        dgv_vec = dgv_vec.reshape(-1, 1)
        grad_1_flat = grad_1.reshape(-1, 1)
        grad_2_flat = grad_2.reshape(-1, 1)

        # 2. Projection Matrix (P = I - v v^T)
        P_matrix = np.eye(len(dgv_vec)) - np.dot(dgv_vec, dgv_vec.T)
        
        # 3. Mean Gradient
        mean_grad = 0.5 * (grad_1_flat + grad_2_flat)
        
        # 4. Recomposed Gradient 
        # Replaces the arbitrary '140' factor with the analytical gap force 2(E1-E2)
        gap_force = 2.0 * (energy_1 - energy_2) * dgv_vec
        seam_force = np.dot(P_matrix, mean_grad)
        
        gp_grad = gap_force + seam_force
        
        return gp_grad.reshape(-1, 3)
    
    def calc_hess(self, grad_1, grad_2, hess_1, hess_2):
        # Robust Hessian construction for GP
        delta_grad = grad_1 - grad_2
        norm_delta_grad = np.linalg.norm(delta_grad)
        if norm_delta_grad < 1e-8:
            dgv_vec = np.zeros_like(delta_grad)
        else:
            dgv_vec = delta_grad / norm_delta_grad
            
        dgv_vec = dgv_vec.reshape(-1, 1)
        
        P_matrix = np.eye(len(dgv_vec)) - np.dot(dgv_vec, dgv_vec.T)
        mean_hess = 0.5 * (hess_1 + hess_2)
        
        # Projected Mean Hessian + Gap Curvature
        proj_hess = np.dot(P_matrix, np.dot(mean_hess, P_matrix))
        gap_curvature = 2.0 * np.dot(dgv_vec, dgv_vec.T)
        
        gp_hess = proj_hess + gap_curvature
        return gp_hess