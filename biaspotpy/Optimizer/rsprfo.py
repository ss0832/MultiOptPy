import numpy as np
import copy
from .hessian_update import ModelHessianUpdate

class RSPRFO:
    def __init__(self, **config):
        # Rational Step P-RFO (Rational Function Optimization) for transition state searches
        # References:
        # [1] Banerjee et al., Phys. Chem., 89, 52-57 (1985)
        # [2] Heyden et al., J. Chem. Phys., 123, 224101 (2005)
        # [3] Baker, J. Comput. Chem., 7, 385-395 (1986)
        # [4] Besalú and Bofill, Theor. Chem. Acc., 100, 265-274 (1998)
        
        # Trust radius for step restriction
        self.trust_radius = config.get("trust_radius", 0.1)
        # Initial alpha parameter for RS-RFO
        self.alpha0 = config.get("alpha0", 1.0)
        # Maximum number of micro-cycles
        self.max_micro_cycles = config.get("max_micro_cycles", 5)
        # Saddle order (0=minimum, 1=first-order saddle/TS, 2=second-order saddle, etc.)
        self.saddle_order = config.get("saddle_order", 1)
        # Hessian update method ('BFGS', 'SR1', 'FSB', 'Bofill', 'PSB', 'MSP', 'auto')
        self.hessian_update_method = config.get("method", "auto")
        
        self.display_flag = config.get("display_flag", True)
        self.config = config
        self.Initialization = True
        
        self.hessian = None
        self.bias_hessian = None
        
        # For tracking optimization
        self.prev_eigvec_max = None
        self.prev_eigvec_min = None
        self.predicted_energy_changes = []
        self.prev_geometry = None
        self.prev_gradient = None
        
        # Define which mode(s) to maximize along based on saddle order
        self.roots = list(range(self.saddle_order))
        
        # Check for valid saddle_order
        if self.saddle_order == 0:
            self.log("Warning: Using RSPRFO for minimum search (saddle_order=0). This is unusual.")
            
        # Initialize the hessian update module
        self.hessian_updater = ModelHessianUpdate()
        return
        
    def log(self, message):
        if self.display_flag:
            print(message)
            
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        if self.Initialization:
            self.prev_eigvec_max = None
            self.prev_eigvec_min = None
            self.predicted_energy_changes = []
            self.prev_geometry = None
            self.prev_gradient = None
            self.Initialization = False
            
        # If hessian isn't set, we can't perform the optimization
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        # Update Hessian if we have previous geometry and gradient information
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_B_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, B_g, pre_geom, pre_B_g)
            
        # Ensure gradient is properly shaped as a 1D array
        gradient = np.asarray(B_g).flatten()
        H = self.hessian
            
        # Compute eigenvalues and eigenvectors of the hessian
        eigvals, eigvecs = np.linalg.eigh(H)
        
       
        # Transform gradient to eigensystem of hessian and ensure it's a 1D array
        gradient_trans = eigvecs.T.dot(gradient).flatten()
        
        # Special case for saddle_order=0
        if self.saddle_order == 0:
            min_indices = list(range(len(gradient_trans)))
            max_indices = []
        else:
            # Minimize energy along all modes, except those specified by saddle_order
            min_indices = [i for i in range(gradient_trans.size) if i not in self.roots]
            # Maximize energy along the modes determined by saddle_order
            max_indices = self.roots
        
        alpha = self.alpha0
        step = np.zeros_like(gradient_trans)  # This ensures step is 1D
        
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-PRFO micro cycle {mu:02d}, alpha={alpha:.6f}")
            
            # Only process max_indices if there are any
            if len(max_indices) > 0:
                # Maximize energy along the chosen saddle direction(s)
                H_aug_max = self.get_augmented_hessian(
                    eigvals[max_indices], gradient_trans[max_indices], alpha
                )
                step_max, eigval_max, nu_max, self.prev_eigvec_max = self.solve_rfo(
                    H_aug_max, "max", prev_eigvec=self.prev_eigvec_max
                )
                step[max_indices] = step_max
            else:
                eigval_max = 0
                step_max = np.array([])
            
            # Only process min_indices if there are any
            if len(min_indices) > 0:
                # Minimize energy along all other modes
                H_aug_min = self.get_augmented_hessian(
                    eigvals[min_indices], gradient_trans[min_indices], alpha
                )
                step_min, eigval_min, nu_min, self.prev_eigvec_min = self.solve_rfo(
                    H_aug_min, "min", prev_eigvec=self.prev_eigvec_min
                )
                step[min_indices] = step_min
            else:
                eigval_min = 0
                step_min = np.array([])
            
            # Calculate step norm
            step_norm = np.linalg.norm(step)
            
            if len(max_indices) > 0:
                self.log(f"norm(step_max)={np.linalg.norm(step_max):.6f}")
            if len(min_indices) > 0:
                self.log(f"norm(step_min)={np.linalg.norm(step_min):.6f}")
                
            self.log(f"norm(step)={step_norm:.6f}")
            
            # Check if step is within trust radius
            inside_trust = step_norm <= self.trust_radius
            if inside_trust:
                self.log(f"Restricted step satisfies trust radius of {self.trust_radius:.6f}")
                self.log(f"Micro-cycles converged in cycle {mu:02d} with alpha={alpha:.6f}!")
                break
                
            # If step is too large, adjust alpha to restrict step size
            # Calculate derivatives only if the corresponding subspaces exist
            dstep2_dalpha_max = 0
            if len(max_indices) > 0:
                dstep2_dalpha_max = (
                    2
                    * eigval_max
                    / (1 + np.dot(step_max, step_max) * alpha)
                    * np.sum(
                        gradient_trans[max_indices] ** 2
                        / (eigvals[max_indices] - eigval_max * alpha) ** 3
                    )
                )
                
            dstep2_dalpha_min = 0
            if len(min_indices) > 0:
                dstep2_dalpha_min = (
                    2
                    * eigval_min
                    / (1 + np.dot(step_min, step_min) * alpha)
                    * np.sum(
                        gradient_trans[min_indices] ** 2
                        / (eigvals[min_indices] - eigval_min * alpha) ** 3
                    )
                )
                
            dstep2_dalpha = dstep2_dalpha_max + dstep2_dalpha_min
            
            # Avoid division by zero
            if abs(dstep2_dalpha) < 1e-10:
                alpha *= 1.5  # Simple heuristic if derivative is very small
            else:
                # Update alpha
                alpha_step = (
                    2 * (self.trust_radius * step_norm - step_norm**2) / dstep2_dalpha
                )
                alpha += alpha_step
            
        # Transform step back to original coordinates
        move_vector = eigvecs.dot(step)
        step_norm = np.linalg.norm(move_vector)
        
        # Scale step if necessary
        if step_norm > self.trust_radius:
            move_vector = move_vector / step_norm * self.trust_radius
            
        self.log(f"Final norm(step)={np.linalg.norm(move_vector):.6f}")
        self.log("")
        
        # Calculate predicted energy change
        predicted_energy_change = self.rfo_model(gradient, H, move_vector)
        self.predicted_energy_changes.append(predicted_energy_change)
        
        # Store current geometry and gradient for the next Hessian update
        self.prev_geometry = copy.deepcopy(geom_num_list)
        self.prev_gradient = copy.deepcopy(B_g)
        
        return move_vector.reshape(-1, 1)
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """Update the Hessian using the specified update method"""
        # Calculate displacement and gradient difference
        displacement = np.array(current_geom - previous_geom).reshape(-1, 1)
        delta_grad = np.array(current_grad - previous_grad).reshape(-1, 1)
        
        # Skip update if changes are too small
        disp_norm = np.linalg.norm(displacement)
        grad_diff_norm = np.linalg.norm(delta_grad)
        
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update due to small changes")
            return
            
        # Check if displacement and gradient difference are sufficiently aligned
        dot_product = np.dot(displacement.T, delta_grad)[0, 0]
        if dot_product <= 0:
            self.log("Skipping Hessian update due to poor alignment")
            return
        print(f"Hessian update: displacement norm={disp_norm:.6f}, gradient diff norm={grad_diff_norm:.6f}, dot product={dot_product:.6f}")
        # Apply the selected Hessian update method
        if "flowchart" in self.hessian_update_method.lower():
            print(f"Hessian update method: flowchart")            
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad, "auto"
            )
        elif "bfgs" in self.hessian_update_method.lower():
            print(f"Hessian update method: bfgs")            
            delta_hess = self.hessian_updater.BFGS_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "sr1" in self.hessian_update_method.lower():
            print(f"Hessian update method: sr1")            
            delta_hess = self.hessian_updater.SR1_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "fsb" in self.hessian_update_method.lower():
            print(f"Hessian update method: fsb")            
            delta_hess = self.hessian_updater.FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "bofill" in self.hessian_update_method.lower():
            print(f"Hessian update method: bofill")
            delta_hess = self.hessian_updater.Bofill_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "psb" in self.hessian_update_method.lower():
            print(f"Hessian update method: psb")            
            delta_hess = self.hessian_updater.PSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "msp" in self.hessian_update_method.lower():
            print(f"Hessian update method: msp")
            delta_hess = self.hessian_updater.MSP_hessian_update(
                self.hessian, displacement, delta_grad
            )
        else:
            print(f"Hessian update method: unknown")
            self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad
            )
            
        # Update the Hessian
        self.hessian += delta_hess
        
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        self.hessian = (self.hessian + self.hessian.T) / 2
    
    def get_augmented_hessian(self, eigenvalues, gradient_components, alpha):
        """Create the augmented hessian matrix for RFO calculation"""
        n = len(eigenvalues)
        H_aug = np.zeros((n + 1, n + 1))
        
        # Fill the upper-left block with eigenvalues / alpha
        np.fill_diagonal(H_aug[:n, :n], eigenvalues / alpha)
        
        # Make sure gradient_components is flattened to the right shape
        gradient_components = np.asarray(gradient_components).flatten()
        
        # Fill the upper-right and lower-left blocks with gradient components / alpha
        H_aug[:n, n] = gradient_components / alpha
        H_aug[n, :n] = gradient_components / alpha
        
        return H_aug
    
    def solve_rfo(self, H_aug, mode="min", prev_eigvec=None):
        """Solve the RFO equations to get the step"""
        eigvals, eigvecs = np.linalg.eigh(H_aug)
        
        if mode == "min":
            idx = np.argmin(eigvals)
        else:  # mode == "max"
            idx = np.argmax(eigvals)
            
        # Check if we need to flip the eigenvector to maintain consistency with the previous step
        if prev_eigvec is not None:
            overlap = np.dot(eigvecs[:, idx], prev_eigvec)
            if overlap < 0:
                eigvecs[:, idx] *= -1
                
        eigval = eigvals[idx]
        eigvec = eigvecs[:, idx]
        
        # The last component is nu
        nu = eigvec[-1]
        
        # The step is -p/nu where p are the first n components of the eigenvector
        step = -eigvec[:-1] / nu
        
        return step, eigval, nu, eigvec
    
    def rfo_model(self, gradient, hessian, step):
        """Estimate energy change based on RFO model"""
        return np.dot(gradient, step) + 0.5 * np.dot(step, np.dot(hessian, step))
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian