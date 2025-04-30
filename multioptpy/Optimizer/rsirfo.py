import numpy as np
from numpy.linalg import norm
import copy
from .hessian_update import ModelHessianUpdate

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
        # Initial alpha parameter for RS-RFO
        self.alpha0 = config.get("alpha0", 1.0)
        # Maximum number of micro-cycles
        self.max_micro_cycles = config.get("max_micro_cycles", 20)
        # Saddle order (0=minimum, 1=first-order saddle/TS, 2=second-order saddle, etc.)
        self.saddle_order = config.get("saddle_order", 1)
        # Hessian update method ('BFGS', 'SR1', 'FSB', 'Bofill', 'PSB', 'MSP', 'auto')
        self.hessian_update_method = config.get("method", "auto")
        # Threshold for filtering small eigenvalues
        self.small_eigval_thresh = config.get("small_eigval_thresh", 1e-6)
        
        # Alpha constraints to prevent numerical instability
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
        
        # Trust radius adjustment thresholds
        self.good_step_threshold = config.get("good_step_threshold", 0.75)
        self.poor_step_threshold = config.get("poor_step_threshold", 0.25)
        self.trust_radius_increase_factor = config.get("trust_radius_increase_factor", 1.2)
        self.trust_radius_decrease_factor = config.get("trust_radius_decrease_factor", 0.5)
        
        # Convergence criteria
        self.energy_change_threshold = config.get("energy_change_threshold", 1e-6)
        self.gradient_norm_threshold = config.get("gradient_norm_threshold", 1e-4)
        
        self.display_flag = config.get("display_flag", True)
        self.config = config
        self.Initialization = True
        
        self.hessian = None
        self.bias_hessian = None
        
        # For tracking optimization
        self.prev_eigvec_min = None
        self.prev_eigvec_size = None  # Store the size of the previous eigenvector
        self.predicted_energy_changes = []
        self.actual_energy_changes = []
        self.prev_geometry = None
        self.prev_gradient = None
        self.prev_energy = None
        self.converged = False
        self.iteration = 0
        
        # Define which mode(s) to maximize along based on saddle order
        self.roots = list(range(self.saddle_order))
        
        # Initialize the hessian update module
        self.hessian_updater = ModelHessianUpdate()
        return
        
    def log(self, message):
        """Print message if display flag is enabled"""
        if self.display_flag:
            print(message)
            
    def filter_small_eigvals(self, eigvals, eigvecs, mask=False):
        """
        Remove small eigenvalues and corresponding eigenvectors from the Hessian
        
        Parameters:
        eigvals: numpy.ndarray - Eigenvalues of the Hessian
        eigvecs: numpy.ndarray - Eigenvectors of the Hessian
        mask: bool - Whether to return the mask of small eigenvalues
        
        Returns:
        tuple - Filtered eigenvalues and eigenvectors (and optionally the mask)
        """
        small_inds = np.abs(eigvals) < self.small_eigval_thresh
        small_num = sum(small_inds)
        
        if small_num > 0:
            self.log(
                f"Found {small_num} small eigenvalues in Hessian. Removed "
                "corresponding eigenvalues and eigenvectors."
            )
            
        filtered_eigvals = eigvals[~small_inds]
        filtered_eigvecs = eigvecs[:, ~small_inds]
        
        # Ensure we don't have too many small eigenvalues
        if small_num > 6:
            self.log(
                f"Warning: Found {small_num} small eigenvalues, which is more than expected. "
                "This may indicate numerical issues. Proceeding with caution."
            )
        
        if mask:
            return filtered_eigvals, filtered_eigvecs, small_inds
        else:
            return filtered_eigvals, filtered_eigvecs
        
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """
        Execute one step of RS-I-RFO optimization
        
        Parameters:
        geom_num_list: numpy.ndarray - Current geometry coordinates
        B_g: numpy.ndarray - Current gradient
        pre_B_g: numpy.ndarray - Previous gradient
        pre_geom: numpy.ndarray - Previous geometry
        B_e: float - Current energy
        pre_B_e: float - Previous energy
        pre_move_vector: numpy.ndarray - Previous step vector
        initial_geom_num_list: numpy.ndarray - Initial geometry
        g: numpy.ndarray - Alternative gradient representation
        pre_g: numpy.ndarray - Previous alternative gradient representation
        
        Returns:
        numpy.ndarray - Optimization step vector
        """
        # Print iteration header
        self.log(f"\n{'='*50}\nRS-I-RFO Iteration {self.iteration}\n{'='*50}")
        
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
                self.actual_energy_changes.append(actual_energy_change)
                
                if len(self.predicted_energy_changes) > 0:
                    self.adjust_trust_radius(actual_energy_change, self.predicted_energy_changes[-1])
            
        # Check if hessian is set
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        # Update Hessian if we have previous geometry and gradient information
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_B_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, B_g, pre_geom, pre_B_g)
            
        # Check for convergence based on gradient
        gradient_norm = np.linalg.norm(B_g)
        self.log(f"Gradient norm: {gradient_norm:.6f}")
        
        if gradient_norm < self.gradient_norm_threshold:
            self.log(f"Converged: Gradient norm {gradient_norm:.6f} below threshold {self.gradient_norm_threshold:.6f}")
            self.converged = True
        
        # Check for convergence based on energy change
        if len(self.actual_energy_changes) > 0:
            last_energy_change = abs(self.actual_energy_changes[-1])
            if last_energy_change < self.energy_change_threshold:
                self.log(f"Converged: Energy change {last_energy_change:.6f} below threshold {self.energy_change_threshold:.6f}")
                self.converged = True
            
        # Store current energy
        current_energy = B_e
        
        # Ensure gradient is properly shaped as a 1D array
        gradient = np.asarray(B_g).flatten()
        H = self.hessian + self.bias_hessian if self.bias_hessian is not None else self.hessian
            
        # Compute eigenvalues and eigenvectors of the hessian
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Count negative eigenvalues for diagnostic purposes
        neg_eigvals = sum(eigvals < -1e-6)
        self.log(f"Found {neg_eigvals} negative eigenvalues (target for this saddle order: {self.saddle_order})")
        
        # Create the projection matrix for RS-I-RFO
        self.log(f"Using projection to construct image potential gradient and hessian for root(s) {self.roots}.")
        P = np.eye(gradient.size)
        for root in self.roots:
            trans_vec = eigvecs[:, root]
            P -= 2 * np.outer(trans_vec, trans_vec)
        
        # Create the image Hessian H_star and image gradient grad_star
        H_star = P.dot(H)
        grad_star = P.dot(gradient)
        
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
        predicted_energy_change = self.rfo_model(gradient, H, move_vector)
        self.predicted_energy_changes.append(predicted_energy_change)
        
        self.log(f"Predicted energy change: {predicted_energy_change:.6f}")
        
        # Evaluate step quality if we have history
        if len(self.actual_energy_changes) > 0 and len(self.predicted_energy_changes) > 1:
            self.evaluate_step_quality()
        
        # Store current geometry, gradient and energy for next iteration
        self.prev_geometry = copy.deepcopy(geom_num_list)
        self.prev_gradient = copy.deepcopy(B_g)
        self.prev_energy = current_energy
        
        # Increment iteration counter
        self.iteration += 1
        
        return move_vector.reshape(-1, 1)
    
    def adjust_trust_radius(self, actual_change, predicted_change):
        """
        Dynamically adjust trust radius based on the agreement between
        actual and predicted energy changes
        
        Parameters:
        actual_change: float - Actual energy change from the last step
        predicted_change: float - Predicted energy change from the last step
        """
        # Skip if either value is too small
        if abs(predicted_change) < 1e-10:
            self.log("Skipping trust radius update: predicted change too small")
            return
            
        # Calculate ratio between actual and predicted changes
        ratio = actual_change / predicted_change
        
        self.log(f"Energy change: actual={actual_change:.6f}, predicted={predicted_change:.6f}, ratio={ratio:.3f}")
        
        old_trust_radius = self.trust_radius
        
        # Adjust trust radius based on the ratio
        if ratio > self.good_step_threshold:
            # Good agreement - increase trust radius
            self.trust_radius = min(self.trust_radius * self.trust_radius_increase_factor, 
                                    self.trust_radius_max)
            if self.trust_radius != old_trust_radius:
                self.log(f"Good step quality (ratio={ratio:.3f}), increasing trust radius to {self.trust_radius:.6f}")
        elif ratio < self.poor_step_threshold:
            # Poor agreement - decrease trust radius
            self.trust_radius = max(self.trust_radius * self.trust_radius_decrease_factor, 
                                    self.trust_radius_min)
            if self.trust_radius != old_trust_radius:
                self.log(f"Poor step quality (ratio={ratio:.3f}), decreasing trust radius to {self.trust_radius:.6f}")
        else:
            # Acceptable agreement - keep trust radius
            self.log(f"Acceptable step quality (ratio={ratio:.3f}), keeping trust radius at {self.trust_radius:.6f}")
    
    def evaluate_step_quality(self):
        """
        Evaluate the quality of recent optimization steps by analyzing
        the trend in predicted and actual energy changes
        
        Returns:
        str - Quality assessment ('good', 'acceptable', or 'poor')
        """
        if len(self.predicted_energy_changes) < 2 or len(self.actual_energy_changes) < 2:
            return "unknown"
            
        # Calculate average prediction accuracy
        ratios = []
        for actual, predicted in zip(self.actual_energy_changes[-2:], self.predicted_energy_changes[-2:]):
            if abs(predicted) > 1e-10:
                ratios.append(actual / predicted)
                
        if not ratios:
            return "unknown"
            
        avg_ratio = sum(ratios) / len(ratios)
        
        # Check if energy is consistently decreasing
        energy_decreasing = all(change < 0 for change in self.actual_energy_changes[-2:])
        
        if 0.8 < avg_ratio < 1.2 and energy_decreasing:
            quality = "good"
        elif 0.5 < avg_ratio < 1.5 and energy_decreasing:
            quality = "acceptable"
        else:
            quality = "poor"
            
        self.log(f"Step quality assessment: {quality} (avg ratio: {avg_ratio:.3f})")
        return quality
    
    def get_rs_step(self, eigvals, eigvecs, gradient):
        """
        Compute the Rational Step using the RS-I-RFO algorithm
        
        Parameters:
        eigvals: numpy.ndarray - Eigenvalues of the image Hessian
        eigvecs: numpy.ndarray - Eigenvectors of the image Hessian
        gradient: numpy.ndarray - Image gradient
        
        Returns:
        numpy.ndarray - Step vector
        """
        # Transform gradient to basis of eigenvectors
        gradient_trans = eigvecs.T.dot(gradient).flatten()

        # Start with the initial alpha value
        alpha = self.alpha0
        
        # Track step norms for convergence detection
        step_norm_history = []
        
        # Variables for the backup strategy
        best_step = None
        best_step_norm_diff = float('inf')
        
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-I-RFO micro cycle {mu:02d}, alpha={alpha:.6f}")
            
            try:
                # Create a fresh augmented Hessian matrix for this cycle
                H_aug = self.get_augmented_hessian(eigvals, gradient_trans, alpha)
                
                # Solve RFO equations to get the step
                rfo_step, eigval_min, nu, current_eigvec_min = self.solve_rfo(
                    H_aug, "min", prev_eigvec=self.prev_eigvec_min
                )
                
                # Store current eigenvector for the next iteration
                self.prev_eigvec_min = current_eigvec_min
                
                # Calculate the norm of the current step
                rfo_norm = np.linalg.norm(rfo_step)
                self.log(f"norm(rfo step)={rfo_norm:.6f}")
                
                # Track step norms for convergence detection
                step_norm_history.append(rfo_norm)
                
                # Save this step if it's closest to the trust radius (as backup)
                norm_diff = abs(rfo_norm - self.trust_radius)
                if norm_diff < best_step_norm_diff:
                    best_step = rfo_step.copy()
                    best_step_norm_diff = norm_diff
                
                # Check if step is within trust radius
                if rfo_norm <= self.trust_radius:
                    step_ = rfo_step.copy()
                    self.log(f"Step satisfies trust radius of {self.trust_radius:.6f}")
                    break
    
                # Update alpha using the improved method
                try:
                    # Calculate the derivative of squared step norm with respect to alpha
                    dstep2_dalpha = self.get_step_derivative(alpha, eigval_min, eigvals, gradient_trans)
                    self.log(f"d(step^2)/dα={dstep2_dalpha:.6e}")
                    
                    # Check for very small derivative to avoid unstable updates
                    if abs(dstep2_dalpha) < 1e-10:
                        # Use more aggressive heuristic (double alpha)
                        old_alpha = alpha
                        alpha = min(alpha * 2.0, self.alpha_max)
                        self.log(f"Small derivative, using heuristic: alpha {old_alpha:.6f} -> {alpha:.6f}")
                    else:
                        # Calculate alpha step using the trust radius formula
                        alpha_step_raw = 2.0 * (self.trust_radius * rfo_norm - rfo_norm**2) / dstep2_dalpha
                        
                        # Limit the step size to prevent numerical instability
                        alpha_step = np.clip(alpha_step_raw, -self.alpha_step_max, self.alpha_step_max)
                        
                        if abs(alpha_step) != abs(alpha_step_raw):
                            self.log(f"Limited alpha step from {alpha_step_raw:.6f} to {alpha_step:.6f}")
                        
                        # Update alpha with bounds checking
                        old_alpha = alpha
                        alpha = min(max(old_alpha + alpha_step, 1e-6), self.alpha_max)
                        self.log(f"Updated alpha: {old_alpha:.6f} -> {alpha:.6f}")
                    
                    # Detect if alpha is hitting the upper limit
                    if alpha == self.alpha_max:
                        self.log(f"Warning: alpha reached maximum value ({self.alpha_max})")
                        if best_step is not None:
                            self.log("Using best step found since alpha reached maximum")
                            step_ = best_step.copy()
                            break
                    
                    # Check if alpha update is not making progress
                    if len(step_norm_history) >= 3:
                        # Calculate consecutive changes in step norm
                        recent_changes = [abs(step_norm_history[i] - step_norm_history[i-1]) 
                                        for i in range(len(step_norm_history)-1, max(0, len(step_norm_history)-3), -1)]
                        
                        # If step norms are not changing significantly, break the loop
                        if all(change < 1e-6 for change in recent_changes):
                            self.log(f"Step norm not changing significantly: {step_norm_history[-3:]}")
                            self.log("Breaking micro-cycle loop")
                            
                            # Use the best step found so far
                            if best_step is not None:
                                step_ = best_step.copy()
                                self.log("Using best step found so far")
                            else:
                                step_ = rfo_step.copy()
                            break
                
                except Exception as e:
                    self.log(f"Error in alpha update: {str(e)}")
                    # Use best step or scale the current step
                    if best_step is not None:
                        self.log("Using best step found so far due to error")
                        step_ = best_step.copy()
                    else:
                        # Only scale if the step exceeds trust radius
                        if rfo_norm > self.trust_radius:
                            self.log("Scaling step to trust radius due to error")
                            step_ = rfo_step / rfo_norm * self.trust_radius
                        else:
                            step_ = rfo_step.copy()
                    break
            
            except Exception as e:
                self.log(f"Error in RFO solution: {str(e)}")
                # Reset prev_eigvec_min and try again without it
                if self.prev_eigvec_min is not None:
                    self.log("Resetting previous eigenvector and retrying")
                    self.prev_eigvec_min = None
                    try:
                        H_aug = self.get_augmented_hessian(eigvals, gradient_trans, alpha)
                        rfo_step_, eigval_min, nu, _ = self.solve_rfo(H_aug, "min")
                        step_ = rfo_step_
                        break
                    except Exception as e2:
                        self.log(f"Second attempt also failed: {str(e2)}")
                
                # Use a fallback step based on steepest descent with trust radius
                self.log("Using fallback steepest descent step")
                sd_step = -gradient_trans
                sd_norm = np.linalg.norm(sd_step)
                if sd_norm > self.trust_radius:
                    step_ = sd_step / sd_norm * self.trust_radius
                else:
                    step_ = sd_step
                break
        else:
            # If micro cycles did not converge
            self.log(
                f"RS-I-RFO algorithm did not converge in {self.max_micro_cycles} cycles. "
                "Using best step found or scaling the last computed step."
            )
            
            # Use the best step found or the last computed step
            if best_step is not None and best_step_norm_diff < abs(rfo_norm - self.trust_radius):
                self.log(f"Using previously found step with norm closest to trust radius")
                step_ = best_step.copy()
            else:
                # Only scale if the step exceeds the trust radius
                if rfo_norm > self.trust_radius:
                    self.log(f"Scaling step to trust radius")
                    step_ = rfo_step / rfo_norm * self.trust_radius
                else:
                    self.log(f"Using last computed step (within trust radius)")
                    step_ = rfo_step.copy()

        # Transform step back to original basis
        step = eigvecs.dot(step_)
        
        step_norm = np.linalg.norm(step)
        self.log(f"Final norm(step)={step_norm:.6f}")
        
        return step
    
    def get_step_derivative(self, alpha, eigval_min, eigvals, gradient_trans):
        """
        Compute derivative of squared step norm with respect to alpha
        with improved numerical stability
        
        Parameters:
        alpha: float - Current alpha value
        eigval_min: float - Minimum eigenvalue of the augmented Hessian
        eigvals: numpy.ndarray - Eigenvalues of the Hessian
        gradient_trans: numpy.ndarray - Gradient in the eigenvector basis
        
        Returns:
        float - Derivative of squared step norm with respect to alpha
        """
        # Safeguard against potential numerical issues with eigval_min
        if abs(eigval_min) < 1e-12:
            # Return a small but non-zero value with appropriate sign
            self.log(f"Warning: Very small eigval_min ({eigval_min}), using safe replacement")
            return 1e-8 if eigval_min >= 0 else -1e-8
        
        try:
            # Calculate denominators with safety checks
            denominators = eigvals - eigval_min * alpha
            
            # Handle small denominators
            small_denoms = np.abs(denominators) < 1e-8
            if np.any(small_denoms):
                self.log(f"Warning: {np.sum(small_denoms)} small denominators detected in derivative calculation")
                safe_denoms = denominators.copy()
                for i in np.where(small_denoms)[0]:
                    if safe_denoms[i] != 0:  # Avoid changing zero to non-zero
                        sign = 1 if safe_denoms[i] > 0 else -1
                        safe_denoms[i] = sign * max(1e-8, abs(safe_denoms[i]))
                    else:
                        safe_denoms[i] = 1e-8  # Handle zero case
                denominators = safe_denoms
            
            # Compute numerator and denominator terms for the derivative
            numerator = gradient_trans**2
            denominator = denominators**3
            
            # Check for valid indices to avoid division by very small values
            valid_indices = np.abs(denominator) > 1e-10
            
            if not np.any(valid_indices):
                self.log("All denominators too small, using safe derivative value")
                return 1e-8  # Return a small positive value as a fallback
            
            # Calculate sum terms with protection against extreme values
            sum_terms = np.zeros_like(numerator)
            sum_terms[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
            
            # Clip extremely large values to prevent overflow
            max_magnitude = 1e20
            if np.any(np.abs(sum_terms) > max_magnitude):
                self.log(f"Warning: Extremely large terms detected (max={np.max(np.abs(sum_terms))}), clipping to {max_magnitude}")
                sum_terms = np.clip(sum_terms, -max_magnitude, max_magnitude)
            
            # Sum the terms and calculate derivative
            sum_term = np.sum(sum_terms)
            dstep2_dalpha = 2.0 * eigval_min * sum_term
            
            # Additional safety check for the final result
            if abs(dstep2_dalpha) > max_magnitude:
                self.log(f"Warning: Extreme derivative value ({dstep2_dalpha}), clipping")
                dstep2_dalpha = np.sign(dstep2_dalpha) * max_magnitude
                
            return dstep2_dalpha
            
        except Exception as e:
            self.log(f"Error in step derivative calculation: {str(e)}")
            # Return a small non-zero value as a fallback
            return 1e-8
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """
        Update the Hessian using the specified update method
        
        Parameters:
        current_geom: numpy.ndarray - Current geometry
        current_grad: numpy.ndarray - Current gradient
        previous_geom: numpy.ndarray - Previous geometry
        previous_grad: numpy.ndarray - Previous gradient
        """
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
            self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad
            )
            
        # Update the Hessian
        self.hessian += delta_hess
        
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        self.hessian = (self.hessian + self.hessian.T) / 2
    
    def get_augmented_hessian(self, eigvals, gradient_components, alpha=1.0):
        """
        Create the augmented hessian matrix for RFO calculation
        
        Parameters:
        eigvals: numpy.ndarray - Eigenvalues of the Hessian
        gradient_components: numpy.ndarray - Gradient components in eigenvector basis
        alpha: float - Alpha parameter for RS-RFO
        
        Returns:
        numpy.ndarray - Augmented Hessian matrix for RFO calculation
        """
        n = len(eigvals)
        H_aug = np.zeros((n + 1, n + 1))
        
        # Fill the upper-left block with eigenvalues / alpha
        np.fill_diagonal(H_aug[:n, :n], eigvals / alpha)
        
        # Make sure gradient_components is flattened to the right shape
        gradient_components = np.asarray(gradient_components).flatten()
        
        # Fill the upper-right and lower-left blocks with gradient components / alpha
        H_aug[:n, n] = gradient_components / alpha
        H_aug[n, :n] = gradient_components / alpha
        
        return H_aug
    
    def solve_rfo(self, H_aug, mode="min", prev_eigvec=None):
        """
        Solve the RFO equations to get the step
        
        Parameters:
        H_aug: numpy.ndarray - Augmented Hessian matrix
        mode: str - "min" for energy minimization, "max" for maximization
        prev_eigvec: numpy.ndarray - Previous eigenvector for consistent direction
        
        Returns:
        tuple - (step, eigenvalue, nu parameter, eigenvector)
        """
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
        """
        Estimate energy change based on RFO model
        
        Parameters:
        gradient: numpy.ndarray - Energy gradient
        hessian: numpy.ndarray - Hessian matrix
        step: numpy.ndarray - Step vector
        
        Returns:
        float - Predicted energy change
        """
        return np.dot(gradient, step) + 0.5 * np.dot(step, np.dot(hessian, step))
    
    def is_converged(self):
        """
        Check if optimization has converged
        
        Returns:
        bool - True if converged, False otherwise
        """
        return self.converged
        
    def get_predicted_energy_changes(self):
        """
        Get the history of predicted energy changes
        
        Returns:
        list - Predicted energy changes for each iteration
        """
        return self.predicted_energy_changes
        
    def get_actual_energy_changes(self):
        """
        Get the history of actual energy changes
        
        Returns:
        list - Actual energy changes for each iteration
        """
        return self.actual_energy_changes
    
    def set_hessian(self, hessian):
        """
        Set the Hessian matrix
        
        Parameters:
        hessian: numpy.ndarray - Hessian matrix
        """
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        """
        Set the bias Hessian matrix
        
        Parameters:
        bias_hessian: numpy.ndarray - Bias Hessian matrix
        """
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        """
        Get the current Hessian matrix
        
        Returns:
        numpy.ndarray - Hessian matrix
        """
        return self.hessian
    
    def get_bias_hessian(self):
        """
        Get the current bias Hessian matrix
        
        Returns:
        numpy.ndarray - Bias Hessian matrix
        """
        return self.bias_hessian
    
    def reset_trust_radius(self):
        """
        Reset trust radius to its initial value
        """
        self.trust_radius = self.trust_radius_initial
        self.log(f"Trust radius reset to initial value: {self.trust_radius:.6f}")