import numpy as np
from numpy.linalg import norm
from .hessian_update import ModelHessianUpdate
from multioptpy.calc_tools import Calculationtools

class CubicNewton:
    def __init__(self, **config):
        """
        Cubic Regularized Newton Method with Global O(1/k²) Convergence
        
        References:
        [1] Mishchenko, K., "Regularized Newton Method with Global O(1/k²) Convergence",
            https://arxiv.org/abs/2112.02089
        """
        # Configuration parameters
        self.max_micro_cycles = config.get("max_micro_cycles", 40)
        self.hessian_update_method = config.get("method", "bfgs")
        self.small_eigval_thresh = config.get("small_eigval_thresh", 1e-6)
        
        # Line search parameters
        self.initial_step_length = config.get("initial_step_length", 0.1)
        self.lipschitz_scale_factor = config.get("lipschitz_scale_factor", 2.0)
        self.min_lipschitz = config.get("min_lipschitz", 1e-6)
        self.max_lipschitz = config.get("max_lipschitz", 1e6)
        
        # Convergence criteria
        self.energy_change_threshold = config.get("energy_change_threshold", 1e-8)
        self.gradient_norm_threshold = config.get("gradient_norm_threshold", 1e-6)
        self.step_norm_tolerance = config.get("step_norm_tolerance", 1e-5)
        
        # Debug and display settings
        self.debug_mode = config.get("debug_mode", False)
        self.display_flag = config.get("display_flag", True)
        
        # Initialize state variables
        self.Initialization = True
        self.hessian = None
        self.bias_hessian = None
        
        # For tracking optimization
        self.prev_geometry = None
        self.prev_gradient = None
        self.prev_energy = None
        self.prev_lipschitz = None
        self.converged = False
        self.iteration = 0
        self.line_search_cycles = 0
        
        # Initialize the hessian update module
        self.hessian_updater = ModelHessianUpdate()
    
    def log(self, message, force=False):
        """Print message if display flag is enabled and either force is True or in debug mode"""
        if self.display_flag and (force or self.debug_mode):
            print(message)
            
    def filter_small_eigvals(self, eigvals, eigvecs, mask=False):
        """Remove small eigenvalues and corresponding eigenvectors from the Hessian"""
        small_inds = np.abs(eigvals) < self.small_eigval_thresh
        small_num = np.sum(small_inds)
        
        if small_num > 0:
            self.log(f"Found {small_num} small eigenvalues in Hessian. Removed corresponding eigenvalues and eigenvectors.")
            
        filtered_eigvals = eigvals[~small_inds]
        filtered_eigvecs = eigvecs[:, ~small_inds]
        
        if small_num > 6:
            self.log(f"Warning: Found {small_num} small eigenvalues, which is more than expected. "
                    "This may indicate numerical issues. Proceeding with caution.", force=True)
        
        if mask:
            return filtered_eigvals, filtered_eigvecs, small_inds
        else:
            return filtered_eigvals, filtered_eigvecs
            
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """Execute one step of Cubic Regularized Newton optimization"""
        # Print iteration header
        self.log(f"\n{'='*50}\nCubic Newton Iteration {self.iteration}\n{'='*50}", force=True)
        
        # Initialize on first call
        if self.Initialization:
            self.prev_geometry = None
            self.prev_gradient = None
            self.prev_energy = None
            self.prev_lipschitz = None
            self.converged = False
            self.iteration = 0
            self.line_search_cycles = 0
            self.Initialization = False
            
        # Check if hessian is set
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        # Update Hessian if we have previous geometry and gradient information
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_B_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, B_g, pre_geom, pre_B_g)
            
        # Check for convergence based on gradient
        gradient_norm = np.linalg.norm(B_g)
        self.log(f"Gradient norm: {gradient_norm:.6f}", force=True)
        
        if gradient_norm < self.gradient_norm_threshold:
            self.log(f"Converged: Gradient norm {gradient_norm:.6f} below threshold {self.gradient_norm_threshold:.6f}", force=True)
            self.converged = True
            return np.zeros_like(B_g).reshape(-1, 1)
        
        # Check for convergence based on energy change
        if self.prev_energy is not None:
            energy_change = abs(B_e - self.prev_energy)
            if energy_change < self.energy_change_threshold:
                self.log(f"Converged: Energy change {energy_change:.6f} below threshold {self.energy_change_threshold:.6f}", force=True)
                self.converged = True
                return np.zeros_like(B_g).reshape(-1, 1)
        
        # Ensure gradient is properly shaped as a 1D array
        gradient = np.asarray(B_g).ravel()
        
        # Use effective Hessian
        H = self.hessian
        if self.bias_hessian is not None:
            H = self.hessian + self.bias_hessian
            
        # Project out translations and rotations if geom_num_list is provided
        if len(geom_num_list) > 0:
            H = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                H, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), False
            )
        
        # Get the cubic Newton step using line search
        move_vector = -1 * self.get_cubic_newton_step(
            geom_num_list, gradient, H, B_e
        )
        
        # Store current geometry, gradient and energy for next iteration
        self.prev_geometry = geom_num_list
        self.prev_gradient = B_g
        self.prev_energy = B_e
        
        # Increment iteration counter
        self.iteration += 1
        
        return move_vector.reshape(-1, 1)
    
    def get_cubic_newton_step(self, geom_num_list, gradient, hessian, energy):
        """Compute the step using Cubic Regularized Newton method with line search"""
        gradient_norm = np.linalg.norm(gradient)
        
        # Initialize Lipschitz constant estimate if this is the first iteration
        if self.prev_lipschitz is None:
            # Initial Lipschitz constant estimate; line 2 in algorithm 2 in [1]
            trial_step_length = self.initial_step_length
            trial_step = trial_step_length * (-gradient / gradient_norm)
            
            # Get energy and gradient at trial point
            trial_coords = geom_num_list.ravel() + trial_step
            self.log(f"Estimating initial Lipschitz constant using step of length {trial_step_length:.4f}")
            
            # For MultiOptPy, we need to call external calculator, so we'll return this trial step
            # and continue on the next iteration with the results
            if self.prev_gradient is None:
                self.log("First iteration: returning initial step to estimate Lipschitz constant")
                return trial_step
                
            trial_gradient = self.prev_gradient.ravel()
            
            # Compute Lipschitz constant
            H = (
                np.linalg.norm(trial_gradient - gradient - hessian.dot(trial_step))
                / np.linalg.norm(trial_step) ** 2
            )
            # Ensure H is within reasonable bounds
            H = max(self.min_lipschitz, min(H, self.max_lipschitz))
        else:
            # Start with previous Lipschitz constant divided by 4 (as in the paper)
            H = self.prev_lipschitz / 4
            H = max(self.min_lipschitz, min(H, self.max_lipschitz))
            
        self.log(f"Starting Lipschitz constant in cycle {self.iteration}, H={H:.4f}")
        
        # Line search to find a good step
        best_step = None
        for i in range(self.max_micro_cycles):
            self.line_search_cycles += 1
            H *= self.lipschitz_scale_factor
            H = min(H, self.max_lipschitz)
            
            self.log(f"Adaptive Newton line search, cycle {i} using H={H:.4f}")
            
            # Regularization parameter λ = sqrt(H * ||g||)
            lambda_ = np.sqrt(H * gradient_norm)
            
            # Solve the regularized Newton system: (H + λI)p = -g
            try:
                trial_step = np.linalg.solve(
                    hessian + lambda_ * np.eye(gradient.size), -gradient
                )
            except np.linalg.LinAlgError:
                # Fallback if the system is singular
                self.log("Linear system is singular, using gradient descent step")
                trial_step = -gradient / gradient_norm * self.initial_step_length
                
            trial_step_norm = np.linalg.norm(trial_step)
            self.log(f"Trial step norm: {trial_step_norm:.6f}")
            
            # For the cubic Newton method, we need to check two conditions:
            # 1. ||∇f(x+p)|| ≤ 2λ||p|| 
            # 2. f(x+p) ≤ f(x) - (2/3)λ||p||²
            #
            # In MultiOptPy's framework, we can't check these conditions directly because
            # we don't have f(x+p) and ∇f(x+p) yet. Instead, we return the step and let
            # the caller evaluate the new point. On the next iteration, we can use that
            # information to adjust our Lipschitz constant.
            
            # For now, we'll store this step as our best guess
            best_step = trial_step
            break
            
        if best_step is None:
            # Fallback to gradient descent
            self.log("Line search failed, using gradient descent step", force=True)
            best_step = -gradient / gradient_norm * self.initial_step_length
            
        # Store current Lipschitz constant for next iteration
        self.prev_lipschitz = H
            
        return best_step
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """Update the Hessian using the specified update method"""
        # Calculate displacement and gradient difference
        displacement = np.asarray(current_geom - previous_geom).reshape(-1, 1)
        delta_grad = np.asarray(current_grad - previous_grad).reshape(-1, 1)
        
        # Skip update if changes are too small
        disp_norm = np.linalg.norm(displacement)
        grad_diff_norm = np.linalg.norm(delta_grad)
        
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update due to small changes")
            return
            
        # Check if displacement and gradient difference are sufficiently aligned
        dot_product = np.dot(displacement.T, delta_grad)
        dot_product = dot_product[0, 0]  # Extract scalar value from 1x1 matrix
        if dot_product <= 0:
            self.log("Skipping Hessian update due to poor alignment")
            return
            
        self.log(f"Hessian update: displacement norm={disp_norm:.6f}, gradient diff norm={grad_diff_norm:.6f}, dot product={dot_product:.6f}")
        
        # Apply the selected Hessian update method
        if "flowchart" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: flowchart")            
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad, "auto"
            )
        elif "bfgs" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: bfgs")            
            delta_hess = self.hessian_updater.BFGS_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "sr1" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: sr1")            
            delta_hess = self.hessian_updater.SR1_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "fsb" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: fsb")            
            delta_hess = self.hessian_updater.FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "bofill" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: bofill")
            delta_hess = self.hessian_updater.Bofill_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "psb" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: psb")            
            delta_hess = self.hessian_updater.PSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "msp" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: msp")
            delta_hess = self.hessian_updater.MSP_hessian_update(
                self.hessian, displacement, delta_grad
            )
        else:
            self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using BFGS.")
            delta_hess = self.hessian_updater.BFGS_hessian_update(
                self.hessian, displacement, delta_grad
            )
            
        # Update the Hessian (in-place addition)
        self.hessian += delta_hess
      
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        # Use in-place operation for symmetrization
        self.hessian = 0.5 * (self.hessian + self.hessian.T)
    
    def is_converged(self):
        """Check if optimization has converged"""
        return self.converged
    
    def set_hessian(self, hessian):
        """Set the Hessian matrix"""
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        """Set the bias Hessian matrix"""
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        """Get the current Hessian matrix"""
        return self.hessian
    
    def get_bias_hessian(self):
        """Get the current bias Hessian matrix"""
        return self.bias_hessian
        
    def get_line_search_cycles(self):
        """Get the number of line search cycles performed"""
        return self.line_search_cycles