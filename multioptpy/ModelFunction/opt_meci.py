import numpy as np

class OptMECI:
    def __init__(self):
        # ref.: J. Am. Chem. Soc. 2015, 137, 3433-3445
        # MECI optimization using GP method with Branching Plane Updating (BPU) 
       
        self.approx_cdv_vec = None # Represents 'y' vector (orthogonal to dgv inside BP)
        self.prev_dgv_vec = None   # Represents 'x_{k-1}'
        self.prev_y_vec = None     # Represents 'y_{k-1}'
        return
    
    def calc_energy(self, energy_1, energy_2):
        tot_energy = (energy_1 + energy_2) / 2.0 
        print("energy_1:", energy_1, "hartree")
        print("energy_2:", energy_2, "hartree")
        print("|energy_1 - energy_2|:", abs(energy_1 - energy_2), "hartree")
        return tot_energy
        
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        # Reshape inputs
        grad_1_flat = grad_1.reshape(-1, 1)
        grad_2_flat = grad_2.reshape(-1, 1)
        
        # 1. Calculate Difference Gradient Vector (x_k) 
        delta_grad = grad_1_flat - grad_2_flat
        norm_delta_grad = np.linalg.norm(delta_grad)
        if norm_delta_grad < 1e-8:
            dgv_vec = np.zeros_like(delta_grad) # Avoid division by zero
        else:
            dgv_vec = delta_grad / norm_delta_grad # x_k
        
        # 2. Determine Approximate Coupling Vector (y_k) using BPU 
        if self.prev_dgv_vec is None:
            # Initialization Step 
            # "A plane made of x0 and the mean energy gradient vector was used as an initial BP."
            mean_grad = 0.5 * (grad_1_flat + grad_2_flat)
            
            # Project mean_grad to be orthogonal to dgv_vec (Gram-Schmidt)
            overlap = np.dot(mean_grad.T, dgv_vec)
            ortho_vec = mean_grad - overlap * dgv_vec
            
            norm_ortho = np.linalg.norm(ortho_vec)
            if norm_ortho < 1e-8:
                 # Fallback if mean grad is parallel to diff grad (unlikely)
                 ortho_vec = np.random.rand(*dgv_vec.shape)
                 ortho_vec = ortho_vec - np.dot(ortho_vec.T, dgv_vec) * dgv_vec
                 norm_ortho = np.linalg.norm(ortho_vec)

            self.approx_cdv_vec = ortho_vec / norm_ortho # Initial y_0
            
        else:
            # Update Step using Eq 4 
            # y_k = [ (y_{k-1}.x_k) * x_{k-1} - (x_{k-1}.x_k) * y_{k-1} ] / normalization
            
            x_k = dgv_vec
            x_prev = self.prev_dgv_vec
            y_prev = self.prev_y_vec
            
            dot_yx = np.dot(y_prev.T, x_k)
            dot_xx = np.dot(x_prev.T, x_k)
            
            numerator = dot_yx * x_prev - dot_xx * y_prev
            norm_num = np.linalg.norm(numerator)
            
            if norm_num < 1e-8:
                 # If x_k didn't change much, keep y_prev orthogonalized to x_k
                 numerator = y_prev - np.dot(y_prev.T, x_k) * x_k
                 norm_num = np.linalg.norm(numerator)

            self.approx_cdv_vec = numerator / norm_num # y_k

        # Store vectors for next step
        self.prev_dgv_vec = dgv_vec.copy()
        self.prev_y_vec = self.approx_cdv_vec.copy()
        
        # 3. Construct Projection Matrix P for MECI 
        # Projects out BOTH dgv (x) and approx_cdv (y) directions
        P_matrix = np.eye(len(dgv_vec)) \
                   - np.dot(dgv_vec, dgv_vec.T) \
                   - np.dot(self.approx_cdv_vec, self.approx_cdv_vec.T)
        
        # 4. Compose Gradient Projection (GP) Gradient
        # Force to reduce energy gap: 2 * (E1 - E2) * dgv
        # Force to minimize mean energy on intersection space (N-2 dim): P * mean_grad
        mean_grad = 0.5 * (grad_1_flat + grad_2_flat)
        
        gap_force = 2.0 * (energy_1 - energy_2) * dgv_vec
        seam_force = np.dot(P_matrix, mean_grad)
        
        gp_grad = gap_force + seam_force
        
        return gp_grad.reshape(len(grad_1), 3)

    def calc_hess(self, hess_1, hess_2):
        # Approximate Hessian for GP method
        # Projects the mean Hessian onto the intersection space
        mean_hess = 0.5 * (hess_1 + hess_2)
        
        # Need current P_matrix. Reconstruct it from stored vectors.
        if self.approx_cdv_vec is None or self.prev_dgv_vec is None:
             # Should not happen if calc_grad is called first
             return mean_hess
             
        dgv_vec = self.prev_dgv_vec
        cdv_vec = self.approx_cdv_vec
        
        P_matrix = np.eye(len(dgv_vec)) \
                   - np.dot(dgv_vec, dgv_vec.T) \
                   - np.dot(cdv_vec, cdv_vec.T)
                   
        # Projected Mean Hessian + Gap Penalty Curvature
        proj_hess = np.dot(P_matrix, np.dot(mean_hess, P_matrix))
        gap_curvature = 2.0 * np.dot(dgv_vec, dgv_vec.T)
        
        gp_hess = proj_hess + gap_curvature
        return gp_hess