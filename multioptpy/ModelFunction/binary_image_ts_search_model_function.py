import numpy as np

# J. Chem. Phys. 157, 124107 (2022)
# https://doi.org/10.1063/5.0102145

class BITSSModelFunction:
    def __init__(self, geom_num_list_1, geom_num_list_2):
        self.f = 0.5
        self.alpha = 10.0
        # self.beta controls the strength of the distance constraint.
        # Smaller beta -> Larger kappa_d -> Stronger attractive force.
        # Default: 0.1 -> Modified: 0.02 for stronger constraint.
        self.beta = 0.02
        
        # Initial distance
        diff = geom_num_list_1 - geom_num_list_2
        self.d = np.linalg.norm(diff)
        
        # Initialize variables to avoid AttributeError if calc_hess/grad called before iter % 500 == 0
        self.kappa_e = 0.0
        self.kappa_d = 0.0
        self.E_B = 0.0
        
        # Store vector info for Hessian calculation
        self.diff_vec = diff.reshape(-1, 1) # (3N, 1)
        self.current_dist = self.d

    def calc_energy(self, energy_1, energy_2, geom_num_list_1, geom_num_list_2, gradient_1, gradient_2, iter):
        # Update distances
        diff_vec = geom_num_list_1 - geom_num_list_2
        current_distance = np.linalg.norm(diff_vec)
        
        # Update parameters periodically
        if iter % 500 == 0:
            self.E_B = abs(energy_1 - energy_2)
            # Avoid division by zero
            self.kappa_e = self.alpha / (2.0 * self.E_B + 1e-10)
        
            unit_vec = diff_vec / (current_distance + 1e-10)
            
            # Project gradients onto the distance vector direction
            # grad_1 is (N, 3), unit_vec is (N, 3). Element-wise mult then sum gives dot product.
            proj_grad_1 = np.sum(gradient_1 * (-1) * unit_vec)
            proj_grad_2 = np.sum(gradient_2 * unit_vec)

            # Eq. (5) logic
            grad_norm_term = np.sqrt(proj_grad_1**2 + proj_grad_2**2)
            a = grad_norm_term / (2 ** 1.5 * self.beta * self.d + 1e-10)
            b = self.E_B / (self.beta * self.d ** 2 + 1e-10)
            self.kappa_d = max(a, b)
            
            # Reset target distance d to current distance at update step
            self.d = current_distance
        
        # Reduce target distance
        self.d = max((1.0 - self.f) * self.d, 1e-10)
        
        # Calculate BITSS Energy
        # Formula: E1 + E2 + ke * (E1 - E2)^2 + kd * (d - d0)^2
        energy = energy_1 + energy_2 + self.kappa_e * (energy_1 - energy_2) ** 2 + self.kappa_d * (current_distance - self.d) ** 2
        
        return energy
        
    def calc_grad(self, energy_1, energy_2, geom_num_list_1, geom_num_list_2, gradient_1, gradient_2):
        # Calculate vector r = x1 - x2 and distance d
        current_vec = geom_num_list_1 - geom_num_list_2
        current_dist = np.linalg.norm(current_vec) + 1e-10

        # Store for calc_hess (flattened for matrix ops)
        self.diff_vec = current_vec.reshape(-1, 1)
        self.current_dist = current_dist

        # Common terms
        delta_E = energy_1 - energy_2
        dist_diff = current_dist - self.d
        
        # Gradient term for distance: 2 * kd * (d - d0) * (r / d)
        grad_dist_term = current_vec * 2.0 * self.kappa_d * dist_diff / current_dist

        # Gradient term for energy: 2 * ke * (E1 - E2) * (g1 - g2)
        # Total Gradient 1: g1 + 2*ke*dE*g1 + dist_term
        bitss_grad_1 = gradient_1 * (1.0 + 2.0 * self.kappa_e * delta_E) + grad_dist_term

        # Total Gradient 2: g2 + 2*ke*dE*(-g2) - dist_term (since d(r)/dx2 = -r/d)
        bitss_grad_2 = gradient_2 * (1.0 - 2.0 * self.kappa_e * delta_E) - grad_dist_term
        
        return bitss_grad_1, bitss_grad_2

    def calc_hess(self, energy_1, energy_2, grad_1, grad_2, hess_1, hess_2):
        """
        Calculate the 6N x 6N Hessian matrix for BITSS.
        H = [ H11  H12 ]
            [ H21  H22 ]
        """
        # Ensure inputs are flattened (3N, 1) or (3N, 3N)
        N3 = self.diff_vec.shape[0]
        g1 = grad_1.reshape(N3, 1)
        g2 = grad_2.reshape(N3, 1)
        
        delta_E = energy_1 - energy_2
        dist_diff = self.current_dist - self.d
        
        # --- Distance Constraint Hessian Terms ---
        # Vd = kd * (d - d0)^2
        # P = r * r.T / d^2 (Projection onto bond axis)
        r = self.diff_vec
        d = self.current_dist
        P = np.dot(r, r.T) / (d**2)
        I = np.eye(N3)
        # H_dist_block = 2*kd * [ P + (d-d0)/d * (I - P) ]
        term_d = P + (dist_diff / d) * (I - P)
        H_dist = 2.0 * self.kappa_d * term_d
        
        # --- Total Hessian Blocks ---
        
        # Block 11: d^2 E / dx1^2
        # = H1 * (1 + 2*ke*dE) + 2*ke * g1 * g1.T + H_dist
        H11 = hess_1 * (1.0 + 2.0 * self.kappa_e * delta_E) + \
              2.0 * self.kappa_e * np.dot(g1, g1.T) + \
              H_dist
              
        # Block 22: d^2 E / dx2^2
        # = H2 * (1 - 2*ke*dE) + 2*ke * g2 * g2.T + H_dist
        H22 = hess_2 * (1.0 - 2.0 * self.kappa_e * delta_E) + \
              2.0 * self.kappa_e * np.dot(g2, g2.T) + \
              H_dist
              
        # Block 12: d^2 E / dx1 dx2
        # = -2*ke * g1 * g2.T - H_dist
        H12 = -2.0 * self.kappa_e * np.dot(g1, g2.T) - H_dist
        
        # Block 21: d^2 E / dx2 dx1
        H21 = H12.T # Symmetric
        
        # Construct Full Matrix
        H_top = np.hstack((H11, H12))
        H_bot = np.hstack((H21, H22))
        H_total = np.vstack((H_top, H_bot))
        
        return H_total