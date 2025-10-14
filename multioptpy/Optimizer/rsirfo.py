import numpy as np

from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Optimizer.block_hessian_update import BlockHessianUpdate

from scipy.optimize import brentq
from multioptpy.Utils.calc_tools import Calculationtools

class RSIRFO:
    def __init__(self, **config):
        """
        Rational Step Image-RFO (Rational Function Optimization) for transition state searches
        
        References:
        [1] Banerjee et al., Phys. Chem., 89, 52-57 (1985)
        [2] Heyden et al., J. Chem. Phys., 123, 224101 (2005)
        [3] Baker, J. Comput. Chem., 7, 385-395 (1986)
        [4] Besalú and Bofill, Theor. Chem. Acc., 100, 265-274 (1998)
        """
        # Configuration parameters
        self.alpha0 = config.get("alpha0", 1.0)
        self.max_micro_cycles = config.get("max_micro_cycles", 40)
        self.saddle_order = config.get("saddle_order", 1)
        self.hessian_update_method = config.get("method", "auto")
        self.small_eigval_thresh = config.get("small_eigval_thresh", 1e-6)
        
        self.alpha_max = config.get("alpha_max", 1e6)
        self.alpha_step_max = config.get("alpha_step_max", 10.0)
        
        # Trust radius parameters
        if self.saddle_order == 0:
            self.trust_radius_initial = config.get("trust_radius", 0.5)
            self.trust_radius_max = config.get("trust_radius_max", 0.5)
        else:
            self.trust_radius_initial = config.get("trust_radius", 0.1)
            self.trust_radius_max = config.get("trust_radius_max", 0.1)
            
        self.trust_radius = self.trust_radius_initial
        self.trust_radius_min = config.get("trust_radius_min", 0.01)
        
        # Trust radius adjustment parameters
        self.good_step_threshold = config.get("good_step_threshold", 0.75)
        self.poor_step_threshold = config.get("poor_step_threshold", 0.25)
        self.trust_radius_increase_factor = config.get("trust_radius_increase_factor", 1.2)
        self.trust_radius_decrease_factor = config.get("trust_radius_decrease_factor", 0.5)
        
        # Convergence criteria
        self.energy_change_threshold = config.get("energy_change_threshold", 1e-6)
        self.gradient_norm_threshold = config.get("gradient_norm_threshold", 1e-4)
        self.step_norm_tolerance = config.get("step_norm_tolerance", 1e-3)
        
        # Debug and display settings
        self.debug_mode = config.get("debug_mode", False)
        self.display_flag = config.get("display_flag", True)
        
        # Initialize state variables
        self.Initialization = True
        self.hessian = None
        self.bias_hessian = None
        
        # For tracking optimization (using more compact storage)
        self.prev_eigvec_min = None
        self.prev_eigvec_size = None
        # Only store last few changes instead of full history for memory efficiency
        self.predicted_energy_changes = []
        self.actual_energy_changes = []
        self.prev_geometry = None  # Will be set with numpy array reference (no deepcopy)
        self.prev_gradient = None  # Will be set with numpy array reference (no deepcopy)
        self.prev_energy = None
        self.converged = False
        self.iteration = 0
  
        # Define modes to maximize based on saddle order
        self.roots = list(range(self.saddle_order))
        
        # Initialize the hessian update module
        self.hessian_updater = ModelHessianUpdate()
        self.block_hessian_updater = BlockHessianUpdate()
        
        # Initial alpha values to try - more memory efficient than np.linspace
        self.alpha_init_values = [0.001 + (10.0 - 0.001) * i / 14 for i in range(15)]
        self.NEB_mode = False
    
    def switch_NEB_mode(self):
        if self.NEB_mode:
            self.NEB_mode = False
        else:
            self.NEB_mode = True
            
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
        """Execute one step of RS-I-RFO optimization"""
        # Print iteration header
        self.log(f"\n{'='*50}\nRS-I-RFO Iteration {self.iteration}\n{'='*50}", force=True)
        
        # Initialize on first call
        if self.Initialization:
            self.prev_eigvec_min = None
            self.prev_eigvec_size = None
            self.predicted_energy_changes = []
            self.actual_energy_changes = []
            self.prev_geometry = None
            self.prev_gradient = None
            self.prev_energy = None
            self.converged = False
            self.iteration = 0
            self.Initialization = False
        else:
            # Adjust trust radius based on previous step's performance
            if self.prev_energy is not None:
                actual_energy_change = B_e - self.prev_energy
                
                # Keep limited history - only store the last few values
                if len(self.actual_energy_changes) >= 3:
                    self.actual_energy_changes.pop(0)
                self.actual_energy_changes.append(actual_energy_change)
                
                if self.predicted_energy_changes:
                    self.adjust_trust_radius(actual_energy_change, self.predicted_energy_changes[-1])
            
        # Check if hessian is set
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        # Update Hessian if we have previous geometry and gradient information
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, g, pre_geom, pre_g)
            
        # Check for convergence based on gradient
        gradient_norm = np.linalg.norm(B_g)
        self.log(f"Gradient norm: {gradient_norm:.6f}", force=True)
        
        if gradient_norm < self.gradient_norm_threshold:
            self.log(f"Converged: Gradient norm {gradient_norm:.6f} below threshold {self.gradient_norm_threshold:.6f}", force=True)
            self.converged = True
        
        # Check for convergence based on energy change
        if self.actual_energy_changes:
            last_energy_change = abs(self.actual_energy_changes[-1])
            if last_energy_change < self.energy_change_threshold:
                self.log(f"Converged: Energy change {last_energy_change:.6f} below threshold {self.energy_change_threshold:.6f}", force=True)
                self.converged = True
            
        # Store current energy
        current_energy = B_e
        
        # Ensure gradient is properly shaped as a 1D array (reuse existing array without copy)
        gradient = np.asarray(B_g).ravel()
        
        # Use effective Hessian
        tmp_hess = self.hessian
        if self.bias_hessian is not None:
            # Add bias_hessian directly to H - avoid creating intermediate matrix
            #print("Adding bias_hessian to hessian")
            H = Calculationtools().project_out_hess_tr_and_rot_for_coord(tmp_hess + self.bias_hessian, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), False)
        else:
            H = Calculationtools().project_out_hess_tr_and_rot_for_coord(tmp_hess, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), False)
        # Compute eigenvalues and eigenvectors of the hessian
        H = 0.5 * (H + H.T)  # Ensure symmetry
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Count negative eigenvalues for diagnostic purposes
        neg_eigvals = np.sum(eigvals < -1e-10)
        self.log(f"Found {neg_eigvals} negative eigenvalues (target for this saddle order: {self.saddle_order})", force=True)
        
        # Create the projection matrix for RS-I-RFO
        self.log(f"Using projection to construct image potential gradient and hessian for root(s) {self.roots}.")
        
        # More efficient projection matrix construction for multiple roots
        P = np.eye(gradient.size)
        root_num = 0
        i = 0
        while root_num < len(self.roots):
            if np.abs(eigvals[i]) > 1e-10: 
                # Extract the eigenvector once
                trans_vec = eigvecs[:, i]
                # Use inplace operation to update P (avoid new allocation)
                if self.NEB_mode:
                    P -= np.outer(trans_vec, trans_vec)
                else:
                    P -= 2 * np.outer(trans_vec, trans_vec)
                root_num += 1
            i += 1
        # Create the image Hessian H_star and image gradient grad_star
        H_star = np.dot(P, H)
        H_star = 0.5 * (H_star + H_star.T)  # Symmetrize the Hessian
        grad_star = np.dot(P, gradient)
        
        # Compute eigenvalues and eigenvectors of the image Hessian
        eigvals_star, eigvecs_star = np.linalg.eigh(H_star)
        
        # Filter out small eigenvalues
        eigvals_star, eigvecs_star = self.filter_small_eigvals(eigvals_star, eigvecs_star)
        
        # Remember the size of the eigenvalue/vector arrays after filtering
        current_eigvec_size = eigvecs_star.shape[1]
        self.log(f"Using {current_eigvec_size} eigenvalues/vectors after filtering")
        
        # Reset previous eigenvector if dimensions don't match
        if self.prev_eigvec_size is not None and self.prev_eigvec_size != current_eigvec_size:
            self.log(f"Resetting previous eigenvector due to dimension change: "
                     f"{self.prev_eigvec_size} → {current_eigvec_size}")
            self.prev_eigvec_min = None
        
        # Get the RS step using the image Hessian and gradient
        move_vector = self.get_rs_step(eigvals_star, eigvecs_star, grad_star)
        
        # Update prev_eigvec_size for next iteration
        self.prev_eigvec_size = current_eigvec_size
        
        # Calculate predicted energy change
        predicted_energy_change = -1*self.rfo_model(gradient, H, move_vector)
        
        # Keep limited history - only store the last few values
        if len(self.predicted_energy_changes) >= 3:
            self.predicted_energy_changes.pop(0)
        self.predicted_energy_changes.append(predicted_energy_change)
        
        self.log(f"Predicted energy change: {predicted_energy_change:.6f}", force=True)
        
        # Evaluate step quality if we have history
        if self.actual_energy_changes and len(self.predicted_energy_changes) > 1:
            self.evaluate_step_quality()
        
        # Store current geometry, gradient and energy for next iteration (no deep copy)
        # Just store references to avoid duplicating large arrays
        self.prev_geometry = geom_num_list
        self.prev_gradient = B_g
        self.prev_energy = current_energy
        
        # Increment iteration counter
        self.iteration += 1
        
        return move_vector.reshape(-1, 1)
    
    def adjust_trust_radius(self, actual_change, predicted_change):
        """Dynamically adjust trust radius based on the agreement between actual and predicted energy changes"""
        # Skip if either value is too small
        if abs(predicted_change) < 1e-10:
            self.log("Skipping trust radius update: predicted change too small")
            return
            
        # Calculate ratio between actual and predicted changes
        ratio = actual_change / predicted_change
        
        self.log(f"Energy change: actual={actual_change:.6f}, predicted={predicted_change:.6f}, ratio={ratio:.3f}", force=True)
        
        old_trust_radius = self.trust_radius
        
        # Adjust trust radius based on the ratio
        if ratio > self.good_step_threshold:
            # Good agreement - increase trust radius
            self.trust_radius = min(self.trust_radius * self.trust_radius_increase_factor, 
                                    self.trust_radius_max)
            if self.trust_radius != old_trust_radius:
                self.log(f"Good step quality (ratio={ratio:.3f}), increasing trust radius to {self.trust_radius:.6f}", force=True)
        elif ratio < self.poor_step_threshold:
            # Poor agreement - decrease trust radius
            self.trust_radius = max(self.trust_radius * self.trust_radius_decrease_factor, 
                                    self.trust_radius_min)
            if self.trust_radius != old_trust_radius:
                self.log(f"Poor step quality (ratio={ratio:.3f}), decreasing trust radius to {self.trust_radius:.6f}", force=True)
        else:
            # Acceptable agreement - keep trust radius
            self.log(f"Acceptable step quality (ratio={ratio:.3f}), keeping trust radius at {self.trust_radius:.6f}", force=True)
    
    def evaluate_step_quality(self):
        """Evaluate the quality of recent optimization steps"""
        if len(self.predicted_energy_changes) < 2 or len(self.actual_energy_changes) < 2:
            return "unknown"
            
        # Calculate ratios correctly considering the sign
        ratios = []
        for actual, predicted in zip(self.actual_energy_changes[-2:], self.predicted_energy_changes[-2:]):
            if abs(predicted) > 1e-10:
                # Directly use the raw ratio without taking absolute values
                ratios.append(actual / predicted)
                
        if not ratios:
            return "unknown"
            
        avg_ratio = sum(ratios) / len(ratios)
        
        # Check if energy is decreasing (energy changes have same sign and in expected direction)
        same_direction = all(
            (actual * predicted > 0) for actual, predicted in zip(
                self.actual_energy_changes[-2:], self.predicted_energy_changes[-2:]
            )
        )
        
        if 0.8 < avg_ratio < 1.2 and same_direction:
            quality = "good"
        elif 0.5 < avg_ratio < 1.5 and same_direction:
            quality = "acceptable"
        else:
            quality = "poor"
            
        self.log(f"Step quality assessment: {quality} (avg ratio: {avg_ratio:.3f})", force=True)
        return quality
    
    def get_rs_step(self, eigvals, eigvecs, gradient):
        """Compute the Rational Step using the RS-I-RFO algorithm"""
        # Transform gradient to basis of eigenvectors - use matrix multiplication for efficiency
        gradient_trans = np.dot(eigvecs.T, gradient)
        
        # Try initial alpha (alpha0) first
        try:
            # Calculate step with default alpha
            H_aug_initial = self.get_augmented_hessian(eigvals, gradient_trans, self.alpha0)
            initial_step, _, _, _ = self.solve_rfo(H_aug_initial, "min")
            initial_step_norm = np.linalg.norm(initial_step)
            
            self.log(f"Initial step with alpha={self.alpha0:.6f} has norm={initial_step_norm:.6f}", force=True)
            
            # If the step is already within trust radius, use it directly
            if initial_step_norm <= self.trust_radius:
                self.log(f"Initial step is within trust radius ({self.trust_radius:.6f}), using it directly", force=True)
                # Transform step back to original basis
                final_step = np.dot(eigvecs, initial_step)
                return final_step
                
            self.log(f"Initial step exceeds trust radius, optimizing alpha...", force=True)
        except Exception as e:
            self.log(f"Error calculating initial step: {str(e)}", force=True)
            # Continue with optimization as a fallback
            
        # Try multiple initial alpha values and select the best step
        best_overall_step = None
        best_overall_norm_diff = float('inf')
        best_alpha_value = None
        
        # Only show number of trials in debug mode or when forced
        self.log(f"Trying {len(self.alpha_init_values)} different initial alpha values:", force=True)
        
        for trial_idx, alpha_init in enumerate(self.alpha_init_values):
            # Only print detailed trial info in debug mode
            if self.debug_mode:
                self.log(f"\n--- Alpha Trial {trial_idx+1}/{len(self.alpha_init_values)}: alpha_init = {alpha_init:.6f} ---")
            
            # Try to compute a step using this initial alpha
            try:
                step_, step_norm, alpha_final = self.compute_rsprfo_step(
                    eigvals, gradient_trans, alpha_init
                )
                
                # Evaluate how close this step is to the trust radius
                norm_diff = abs(step_norm - self.trust_radius)
                
                # Only print detailed trial results in debug mode
                if self.debug_mode:
                    self.log(f"Alpha trial {trial_idx+1}: alpha_init={alpha_init:.6f} -> alpha_final={alpha_final:.6f}, "
                             f"step_norm={step_norm:.6f}, diff={norm_diff:.6f}")
                
                # Check if this is the best step so far
                # Prioritize steps within trust radius, then minimize distance to trust radius
                is_better = False
                
                if best_overall_step is None:
                    is_better = True
                elif step_norm <= self.trust_radius and best_overall_norm_diff > self.trust_radius:
                    # This step is within trust radius but current best is not
                    is_better = True
                elif (step_norm <= self.trust_radius) == (best_overall_norm_diff <= self.trust_radius):
                    # Both steps are either within or outside trust radius, choose the closest
                    if norm_diff < best_overall_norm_diff:
                        is_better = True
                        
                if is_better:
                    # Avoid unnecessary copies - use direct assignment
                    best_overall_step = step_
                    best_overall_norm_diff = norm_diff
                    best_alpha_value = alpha_init
                
            except Exception as e:
                # Only log errors in debug mode
                if self.debug_mode:
                    self.log(f"Error in alpha trial {trial_idx+1}: {str(e)}")
                
        if best_overall_step is None:
            # If all trials failed, use a steepest descent step
            self.log("All alpha trials failed, using steepest descent step as fallback", force=True)
            sd_step = -gradient_trans
            sd_norm = np.linalg.norm(sd_step)
            
            if sd_norm > self.trust_radius:
                best_overall_step = sd_step / sd_norm * self.trust_radius
            else:
                best_overall_step = sd_step
        else:
            # Only show final selected alpha value
            self.log(f"Selected alpha value: {best_alpha_value:.6f}", force=True)
            
        # Transform step back to original basis (use matrix multiplication for efficiency)
        step = np.dot(eigvecs, best_overall_step)
        
        step_norm = np.linalg.norm(step)
        self.log(f"Final norm(step)={step_norm:.6f}", force=True)
        
        return step
    
    def compute_rsprfo_step(self, eigvals, gradient_trans, alpha_init):
        """Compute an RS-P-RFO step using a specific initial alpha value"""
        # Create proxy functions for step norm calculation
        def calculate_step(alpha):
            """Calculate RFO step for a given alpha value"""
            try:
                H_aug = self.get_augmented_hessian(eigvals, gradient_trans, alpha)
                step, _, _, _ = self.solve_rfo(H_aug, "min")
                return step
            except Exception as e:
                self.log(f"Error in step calculation: {str(e)}")
                raise
                
        def step_norm_squared(alpha):
            """Calculate ||step||^2 for a given alpha value"""
            step = calculate_step(alpha)
            return np.dot(step, step)
            
        def objective_function(alpha):
            """U(a) = ||step||^2 - R^2"""
            return step_norm_squared(alpha) - self.trust_radius**2
        
        # Find alpha that gives step with norm close to trust radius
        # First, try bracketing the root
        alpha_lo = 1e-6  # Very small alpha gives large step
        alpha_hi = self.alpha_max  # Very large alpha gives small step
        
        # Check step norms at boundaries to establish bracket
        try:
            step_lo = calculate_step(alpha_lo)
            norm_lo = np.linalg.norm(step_lo)
            obj_lo = norm_lo**2 - self.trust_radius**2
            
            step_hi = calculate_step(alpha_hi) 
            norm_hi = np.linalg.norm(step_hi)
            obj_hi = norm_hi**2 - self.trust_radius**2
            
            self.log(f"Bracket search: alpha_lo={alpha_lo:.6e}, step_norm={norm_lo:.6f}, obj={obj_lo:.6e}")
            self.log(f"Bracket search: alpha_hi={alpha_hi:.6e}, step_norm={norm_hi:.6f}, obj={obj_hi:.6e}")
            
            # Check if we have a proper bracket (signs differ)
            if obj_lo * obj_hi >= 0:
                # No bracket, so use initial alpha and proceed with Newton iterations
                self.log("Could not establish bracket with opposite signs, proceeding with Newton iterations")
                alpha = alpha_init
            else:
                # We have a bracket, use Brent's method for robust root finding
                self.log("Bracket established, using Brent's method for root finding")
                try:
                    alpha = brentq(objective_function, alpha_lo, alpha_hi, 
                                   xtol=1e-6, rtol=1e-6, maxiter=50)
                    self.log(f"Brent's method converged to alpha={alpha:.6e}")
                except Exception as e:
                    self.log(f"Brent's method failed: {str(e)}, using initial alpha")
                    alpha = alpha_init
        except Exception as e:
            self.log(f"Error establishing bracket: {str(e)}, using initial alpha")
            alpha = alpha_init
            
        # Use Newton iterations to refine alpha
        alpha = alpha_init if 'alpha' not in locals() else alpha
        # Use a fixed size numpy array instead of growing list for step_norm_history
        step_norm_history = np.zeros(self.max_micro_cycles)
        history_count = 0
        best_step = None
        best_step_norm_diff = float('inf')
        
        # Variables to track bracketing
        alpha_left = None
        alpha_right = None
        objval_left = None
        objval_right = None
        
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-I-RFO micro cycle {mu:02d}, alpha={alpha:.6f}")
            
            try:
                # Calculate current step and its properties
                step = calculate_step(alpha)
                step_norm = np.linalg.norm(step)
                self.log(f"norm(step)={step_norm:.6f}")
                
                # Keep track of the best step seen so far (closest to trust radius)
                norm_diff = abs(step_norm - self.trust_radius)
                if norm_diff < best_step_norm_diff:
                    if best_step is None:
                        best_step = step.copy()
                    else:
                        # In-place update of best_step
                        best_step[:] = step
                    best_step_norm_diff = norm_diff
                
                # Calculate objective function value U(a) = ||step||^2 - R^2
                objval = step_norm**2 - self.trust_radius**2
                self.log(f"U(a)={objval:.6e}")
                
                # Update bracketing information
                if objval < 0 and (alpha_left is None or alpha > alpha_left):
                    alpha_left = alpha
                    objval_left = objval
                elif objval > 0 and (alpha_right is None or alpha < alpha_right):
                    alpha_right = alpha
                    objval_right = objval
                
                # Check if we're already very close to the target radius
                if abs(objval) < 1e-8 or norm_diff < self.step_norm_tolerance:
                    self.log(f"Step norm {step_norm:.6f} is sufficiently close to trust radius")
                    
                    # Even if we're within tolerance, do at least one or two iterations to try to improve
                    if mu >= 1:
                        break
                
                # Track step norm history for convergence detection (use fixed size array)
                if history_count < self.max_micro_cycles:
                    step_norm_history[history_count] = step_norm
                    history_count += 1
                
                # Compute derivative of squared step norm with respect to alpha
                dstep2_dalpha = self.get_step_derivative(alpha, eigvals, gradient_trans, step)
                self.log(f"d(||step||^2)/dα={dstep2_dalpha:.6e}")
                
                # Update alpha with correct Newton formula: a' = a - U(a)/U'(a)
                if abs(dstep2_dalpha) < 1e-10:
                    # Small derivative - use bisection if bracket is available
                    if alpha_left is not None and alpha_right is not None:
                        alpha_new = (alpha_left + alpha_right) / 2
                        self.log(f"Small derivative, using bisection: alpha {alpha:.6f} -> {alpha_new:.6f}")
                    else:
                        # No bracket yet, use heuristic scaling
                        if objval > 0:  # Step too small, need smaller alpha
                            alpha_new = max(alpha / 2, 1e-6)
                        else:  # Step too large, need larger alpha
                            alpha_new = min(alpha * 2, self.alpha_max)
                        self.log(f"Small derivative, no bracket, using heuristic: alpha {alpha:.6f} -> {alpha_new:.6f}")
                else:
                    # Use Newton update with proper U(a)/U'(a)
                    alpha_step_raw = -objval / dstep2_dalpha
                    
                    # Apply safeguards to Newton step
                    alpha_step = np.clip(alpha_step_raw, -self.alpha_step_max, self.alpha_step_max)
                    if abs(alpha_step) != abs(alpha_step_raw):
                        self.log(f"Limited alpha step from {alpha_step_raw:.6f} to {alpha_step:.6f}")
                    
                    alpha_new = alpha + alpha_step
                    
                    # Additional protection: if bracket available, ensure we stay within bracket
                    if alpha_left is not None and alpha_right is not None:
                        # Safeguard to keep alpha within established bracket
                        alpha_new = max(min(alpha_new, alpha_right * 0.99), alpha_left * 1.01)
                        if alpha_new != alpha + alpha_step:
                            self.log(f"Safeguarded alpha to stay within bracket: {alpha_new:.6f}")
                
                # Update alpha with bounds checking
                old_alpha = alpha
                alpha = min(max(alpha_new, 1e-6), self.alpha_max)
                self.log(f"Updated alpha: {old_alpha:.6f} -> {alpha:.6f}")
                
                # Check if alpha is hitting limits
                if alpha == self.alpha_max or alpha == 1e-6:
                    self.log(f"Alpha hit boundary at {alpha:.6e}, stopping iterations")
                    break
                
                # Check for convergence in step norm using the last 3 values
                if history_count >= 3:
                    idx = history_count - 1
                    recent_changes = [
                        abs(step_norm_history[idx] - step_norm_history[idx-1]),
                        abs(step_norm_history[idx-1] - step_norm_history[idx-2])
                    ]
                    if all(change < 1e-6 for change in recent_changes):
                        self.log("Step norm not changing significantly, stopping iterations")
                        break
                        
            except Exception as e:
                self.log(f"Error in micro-cycle {mu}: {str(e)}")
                # If we have a good step, use it and stop
                if best_step is not None:
                    self.log("Using best step found so far due to error")
                    step = best_step
                    step_norm = np.linalg.norm(step)
                    break
                else:
                    # Last resort: steepest descent
                    self.log("Falling back to steepest descent due to errors")
                    step = -gradient_trans
                    step_norm = np.linalg.norm(step)
                    if step_norm > self.trust_radius:
                        step = step / step_norm * self.trust_radius
                        step_norm = self.trust_radius
                    break
        else:
            # If we exhausted micro-cycles without converging
            self.log(f"RS-I-RFO did not converge in {self.max_micro_cycles} cycles")
            if best_step is not None:
                self.log("Using best step found during iterations")
                step = best_step
                step_norm = np.linalg.norm(step)
        
        return step, step_norm, alpha
    
    def get_step_derivative(self, alpha, eigvals, gradient_trans, step=None):
        """Compute derivative of squared step norm with respect to alpha directly"""
        # If step was not provided, compute it
        if step is None:
            try:
                H_aug = self.get_augmented_hessian(eigvals, gradient_trans, alpha)
                step, eigval_min, _, _ = self.solve_rfo(H_aug, "min")
            except Exception as e:
                self.log(f"Error in step calculation for derivative: {str(e)}")
                return 1e-8  # Return a small value as fallback
        
        try:
            # Get the minimum eigenvalue from the augmented Hessian
            H_aug = self.get_augmented_hessian(eigvals, gradient_trans, alpha)
            _, eigval_min, _, _ = self.solve_rfo(H_aug, "min")
            
            # Calculate the denominators with safety
            denominators = eigvals - eigval_min * alpha
            
            # Handle small denominators safely (vectorized operations for efficiency)
            small_denoms = np.abs(denominators) < 1e-8
            if np.any(small_denoms):
                # Create safe denominators with minimal new memory allocation
                safe_denoms = denominators.copy()
                safe_denoms[small_denoms] = np.sign(safe_denoms[small_denoms]) * np.maximum(1e-8, np.abs(safe_denoms[small_denoms]))
                # Apply sign correction for zeros
                zero_mask = safe_denoms[small_denoms] == 0
                if np.any(zero_mask):
                    safe_denoms[small_denoms][zero_mask] = 1e-8
                denominators = safe_denoms
                
            # Calculate the summation term - use vectorized operations
            numerator = gradient_trans**2
            denominator = denominators**3
            
            # Avoid division by very small values
            valid_indices = np.abs(denominator) > 1e-10
            
            if not np.any(valid_indices):
                return 1e-8  # Return a small positive value if no valid indices
                
            # Initialize sum terms as zeros to avoid allocation inside loop
            sum_terms = np.zeros_like(numerator)
            sum_terms[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
            
            # Clip extremely large values
            max_magnitude = 1e20
            large_values = np.abs(sum_terms) > max_magnitude
            if np.any(large_values):
                sum_terms[large_values] = np.sign(sum_terms[large_values]) * max_magnitude
                
            sum_term = np.sum(sum_terms)
            
            # Calculate the derivative with protection
            dstep2_dalpha = 2.0 * eigval_min * sum_term
            
            # Additional safety check
            if not np.isfinite(dstep2_dalpha) or abs(dstep2_dalpha) > max_magnitude:
                dstep2_dalpha = np.sign(dstep2_dalpha) * max_magnitude if dstep2_dalpha != 0 else 1e-8
                
            return dstep2_dalpha
            
        except Exception as e:
            self.log(f"Error in derivative calculation: {str(e)}")
            return 1e-8  # Return a small positive value as fallback
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """Update the Hessian using the specified update method"""
        # Calculate displacement and gradient difference (avoid unnecessary reshaping)
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
        elif "block_cfd_fsb" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: block_cfd_fsb")            
            delta_hess = self.block_hessian_updater.block_CFD_FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "block_cfd_bofill" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: block_cfd_bofill")            
            delta_hess = self.block_hessian_updater.block_CFD_Bofill_hessian_update(
                self.hessian, displacement, delta_grad
            )
        
        elif "block_bfgs" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: block_bfgs")            
            delta_hess = self.block_hessian_updater.block_BFGS_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "block_fsb" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: block_fsb")            
            delta_hess = self.block_hessian_updater.block_FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "block_bofill" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: block_bofill")            
            delta_hess = self.block_hessian_updater.block_Bofill_hessian_update(
                self.hessian, displacement, delta_grad
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
        elif "pcfd_bofill" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: pcfd_bofill")
            delta_hess = self.hessian_updater.pCFD_Bofill_hessian_update(
                self.hessian, displacement, delta_grad
            )    
            
        elif "cfd_fsb" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: cfd_fsb")            
            delta_hess = self.hessian_updater.CFD_FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "cfd_bofill" in self.hessian_update_method.lower():
            self.log(f"Hessian update method: cfd_bofill")
            delta_hess = self.hessian_updater.CFD_Bofill_hessian_update(
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
            self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad
            )
            
        # Update the Hessian (in-place addition)
        self.hessian += delta_hess
      
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        # Use in-place operation for symmetrization
        self.hessian = 0.5 * (self.hessian + self.hessian.T)
    
    def get_augmented_hessian(self, eigvals, gradient_components, alpha=1.0):
        """Create the augmented hessian matrix for RFO calculation"""
        n = len(eigvals)
        H_aug = np.zeros((n + 1, n + 1))
        
        # Fill the upper-left block with eigenvalues / alpha
        np.fill_diagonal(H_aug[:n, :n], eigvals / alpha)
        
        # Make sure gradient_components is flattened
        gradient_components = np.asarray(gradient_components).ravel()
        
        # Fill the upper-right and lower-left blocks with gradient components / alpha
        H_aug[:n, n] = gradient_components / alpha
        H_aug[n, :n] = gradient_components / alpha
        
        return H_aug
    
    def solve_rfo(self, H_aug, mode="min", prev_eigvec=None):
        """Solve the RFO equations to get the step"""
        # Solve the eigenvalue problem for the augmented Hessian
        eigvals, eigvecs = np.linalg.eigh(H_aug)
        
        # Select the appropriate eigenvalue/vector based on mode
        if mode == "min":
            idx = np.argmin(eigvals)
        else:  # mode == "max"
            idx = np.argmax(eigvals)
        
        # Get the selected eigenvector
        eigval = eigvals[idx]
        eigvec = eigvecs[:, idx]
        
        # Check if we need to flip the eigenvector to maintain consistency with the previous step
        if prev_eigvec is not None:
            # Check dimensions first to avoid the ValueError
            if prev_eigvec.size == eigvec.size:
                overlap = np.dot(eigvec, prev_eigvec)
                if overlap < 0:
                    eigvec *= -1
            else:
                # Dimensions don't match, can't compute overlap
                self.log(f"Warning: Eigenvector dimension mismatch. "
                         f"Current: {eigvec.size}, Previous: {prev_eigvec.size}. "
                         f"Skipping eigenvector consistency check.")
                prev_eigvec = None  # Reset for future iterations
                
        # The last component is nu
        nu = eigvec[-1]
        
        # Ensure nu is not too close to zero
        if abs(nu) < 1e-10:
            self.log(f"Warning: Very small nu value: {nu}. Using safe value.")
            nu = np.sign(nu) * max(1e-10, abs(nu))
            
        # The step is -p/nu where p are the first n components of the eigenvector
        step = -eigvec[:-1] / nu
        
        return step, eigval, nu, eigvec
    
    def rfo_model(self, gradient, hessian, step):
        """Estimate energy change based on RFO model"""
        # Use more efficient matrix operations
        return np.dot(gradient, step) + 0.5 * np.dot(np.dot(step, hessian), step)
    
    def is_converged(self):
        """Check if optimization has converged"""
        return self.converged
        
    def get_predicted_energy_changes(self):
        """Get the history of predicted energy changes"""
        return self.predicted_energy_changes
        
    def get_actual_energy_changes(self):
        """Get the history of actual energy changes"""
        return self.actual_energy_changes
    
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
    
    def reset_trust_radius(self):
        """Reset trust radius to its initial value"""
        self.trust_radius = self.trust_radius_initial
        self.log(f"Trust radius reset to initial value: {self.trust_radius:.6f}", force=True)