import numpy as np

class OptMESX:
    def __init__(self):
        # ref.: J. Am. Chem. Soc. 2015, 137, 3433-3445 
        # MESX optimization using Gradient Projection (GP) method.
        # Only the difference gradient vector (DG or f) is projected out.
        return
    
    def calc_energy(self, energy_1, energy_2):
        # The objective is to minimize the mean energy on the seam.
        tot_energy = (energy_1 + energy_2) / 2.0 
        print("energy_1:", energy_1, "hartree")
        print("energy_2:", energy_2, "hartree")
        print("|energy_1 - energy_2|:", abs(energy_1 - energy_2), "hartree")
        return tot_energy
        
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        # 1. Calculate Difference Gradient Vector (DGV) / f vector
        delta_grad = grad_1 - grad_2
        norm_delta_grad = np.linalg.norm(delta_grad)
        
        if norm_delta_grad < 1e-8:
            dgv_vec = np.zeros_like(delta_grad)
        else:
            dgv_vec = delta_grad / norm_delta_grad
        
        # Ensure correct shape for matrix operations
        dgv_vec = dgv_vec.reshape(-1, 1)
        grad_1 = grad_1.reshape(-1, 1)
        grad_2 = grad_2.reshape(-1, 1)
        
        # 2. Define Projection Matrix P for MESX
        # Projects out the component along the difference vector (degenerate lifting direction)
        # P = I - v * v.T
        P_matrix = np.eye(len(dgv_vec)) - np.dot(dgv_vec, dgv_vec.T)
        
        # 3. Calculate Mean Gradient
        mean_grad = 0.5 * (grad_1 + grad_2)
        
        # 4. Compose Gradient Projection (GP) Gradient 
        # Force to reduce energy gap: 2 * (E1 - E2) * dgv
        # Force to minimize mean energy on seam: P * mean_grad
        gap_force = 2.0 * (energy_1 - energy_2) * dgv_vec
        seam_force = np.dot(P_matrix, mean_grad)
        
        gp_grad = gap_force + seam_force
        
        return gp_grad.reshape(-1, 3)
    
    def calc_hess(self, grad_1, grad_2, hess_1, hess_2):
        # Approximate Hessian for GP method
        delta_grad = grad_1 - grad_2
        norm_delta_grad = np.linalg.norm(delta_grad)
        
        if norm_delta_grad < 1e-8:
            dgv_vec = np.zeros_like(delta_grad)
        else:
            dgv_vec = delta_grad / norm_delta_grad
            
        dgv_vec = dgv_vec.reshape(-1, 1)
        
        # Projection Matrix
        P_matrix = np.eye(len(dgv_vec)) - np.dot(dgv_vec, dgv_vec.T)
        
        # Mean Hessian
        mean_hess = 0.5 * (hess_1 + hess_2)
        
        # Projected Mean Hessian
        # This describes curvature along the seam.
        proj_hess = np.dot(P_matrix, np.dot(mean_hess, P_matrix))
        
        # Gap Curvature (Penalty term approximation)
        # Adds large curvature along the difference vector to enforce the gap constraint strongly.
        gap_curvature = 2.0 * np.dot(dgv_vec, dgv_vec.T)
        
        gp_hess = proj_hess + gap_curvature
        
        return gp_hess