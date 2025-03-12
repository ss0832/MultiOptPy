import numpy as np


class C2DIIS:
    """
    Implementation of C2-DIIS (C-squared DIIS) optimization method.
    
    This method uses squared error vectors to improve DIIS performance,
    particularly in challenging optimization scenarios.
    
    ref.: https://doi.org/10.1002/qua.560450106
    """
    def __init__(self):
        # C2DIIS parameters
        self.c2diis_history_size = 5        # History size
        self.c2diis_min_points = 2          # Minimum points to start C2DIIS
        self.c2diis_weight_initial = 0.2    # Initial C2DIIS weight
        self.c2diis_weight_max = 0.8        # Maximum C2DIIS weight
        
        # Robust coefficient handling
        self.c2diis_coeff_min = -0.5        # Min coefficient value
        self.c2diis_coeff_max = 1.5         # Max coefficient value
        self.c2diis_regularization = 1e-7   # Regularization parameter
        
        # Error recovery
        self.c2diis_failure_count = 0       # Failure counter
        self.c2diis_max_failures = 2        # Max failures before reset
        self.c2diis_recovery_steps = 2      # Recovery mode steps
        self.c2diis_current_recovery = 0    # Current recovery counter
        
        # Step validation parameters
        self.c2diis_step_ratio_max = 2.5    # Max allowed step ratio
        
        # Weight adjustment
        self.c2diis_weight_current = self.c2diis_weight_initial  # Current weight
        self.c2diis_weight_increment = 0.05  # Increment for success
        self.c2diis_weight_decrement = 0.1   # Decrement for failure
        
        # History storage
        self.geom_history = []
        self.grad_history = []
        self.c2error_history = []           # Squared error vectors
        self.quality_history = []
        
        # Convergence monitoring
        self.prev_grad_rms = float('inf')
        self.non_improving_count = 0
        self.iter = 0
    
    def _compute_c2error(self, gradient):
        """
        Compute the C2-error vector (squared form of gradient)
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
            
        Returns:
        --------
        numpy.ndarray
            C2-error vector
        """
        # Create squared form: g * g^T * g
        g_flat = gradient.flatten()
        g_norm = np.linalg.norm(g_flat)
        
        # Avoid numerical issues with very small gradients
        if g_norm < 1e-10:
            return gradient.copy()
            
        # Normalize to avoid numerical issues with very large/small gradients
        g_norm = max(g_norm, 1e-10)
        g_normalized = g_flat / g_norm
        
        # Compute outer product
        g_outer = np.outer(g_normalized, g_normalized)
        
        # Multiply with original gradient to get C2-error
        c2error = np.dot(g_outer, g_flat).reshape(gradient.shape)
        
        return c2error
    
    def _update_history(self, geometry, gradient, step_quality=1.0):
        """
        Update the C2DIIS history
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry
        gradient : numpy.ndarray
            Current gradient
        step_quality : float
            Quality metric for this point (1.0 = good, <1.0 = lower quality)
        """
        # Compute C2-error vector
        c2error = self._compute_c2error(gradient)
        
        # Add current point to history
        self.geom_history.append(geometry.copy())
        self.grad_history.append(gradient.copy())
        self.c2error_history.append(c2error)
        self.quality_history.append(step_quality)
        
        # If in recovery mode, limit history
        if self.c2diis_current_recovery > 0:
            self.c2diis_current_recovery -= 1
            if len(self.geom_history) > 2:
                self.geom_history = self.geom_history[-2:]
                self.grad_history = self.grad_history[-2:]
                self.c2error_history = self.c2error_history[-2:]
                self.quality_history = self.quality_history[-2:]
            return
        
        # Limit history size
        if len(self.geom_history) > self.c2diis_history_size:
            # Remove lowest quality point (except most recent)
            if len(self.geom_history) > 2:
                oldest_qualities = self.quality_history[:-1]
                worst_idx = np.argmin(oldest_qualities)
                
                self.geom_history.pop(worst_idx)
                self.grad_history.pop(worst_idx)
                self.c2error_history.pop(worst_idx)
                self.quality_history.pop(worst_idx)
            else:
                # Default to removing oldest point
                self.geom_history.pop(0)
                self.grad_history.pop(0)
                self.c2error_history.pop(0)
                self.quality_history.pop(0)
    
    def _solve_c2diis_equations(self):
        """
        Solve C2DIIS equations with robustness measures
        
        Returns:
        --------
        numpy.ndarray
            C2DIIS coefficients
        """
        n_points = len(self.c2error_history)
        
        if n_points < 2:
            return np.array([1.0])
        
        # Construct the B matrix with C2-error dot products
        B = np.zeros((n_points + 1, n_points + 1))
        
        # Fill B matrix with error vector dot products
        for i in range(n_points):
            for j in range(n_points):
                weight_factor = np.sqrt(self.quality_history[i] * self.quality_history[j])
                B[i, j] = weight_factor * np.dot(
                    self.c2error_history[i].flatten(),
                    self.c2error_history[j].flatten()
                )
        
        # Add regularization to diagonal
        diag_indices = np.diag_indices(n_points)
        B[diag_indices] += self.c2diis_regularization
        
        # Add Lagrange multiplier constraints
        B[n_points, :n_points] = 1.0
        B[:n_points, n_points] = 1.0
        B[n_points, n_points] = 0.0
        
        # Right-hand side vector
        rhs = np.zeros(n_points + 1)
        rhs[n_points] = 1.0
        
        # Try to solve with multiple methods
        methods = [
            lambda: np.linalg.solve(B, rhs),
            lambda: np.linalg.lstsq(B, rhs, rcond=1e-10)[0],
            lambda: self._minimal_solution(n_points)
        ]
        
        coefficients = None
        for solver in methods:
            try:
                coefficients = solver()
                if not np.any(np.isnan(coefficients)) and abs(np.sum(coefficients[:n_points]) - 1.0) < 0.01:
                    break
            except:
                continue
        
        # If all methods failed, use most recent point
        if coefficients is None or np.any(np.isnan(coefficients)):
            coefficients = np.zeros(n_points + 1)
            coefficients[n_points-1] = 1.0  # Use most recent point
        
        # Extract actual coefficients
        return coefficients[:n_points]
    
    def _minimal_solution(self, n_points):
        """
        Fallback solution when numerical methods fail
        """
        result = np.zeros(n_points + 1)
        
        # Exponential weighting for recent points
        total_weight = 0
        for i in range(n_points):
            # Exponential weighting: more weight to recent points
            result[i] = 2.0**(i)
            total_weight += result[i]
        
        # Normalize to sum=1
        result[:n_points] /= total_weight
        return result
    
    def _filter_coefficients(self, coeffs):
        """
        Filter extreme coefficient values
        
        Parameters:
        -----------
        coeffs : numpy.ndarray
            C2DIIS coefficients
            
        Returns:
        --------
        tuple
            (filtered_coeffs, was_filtered, quality)
        """
        # Check for extreme values
        extreme_values = np.logical_or(coeffs < self.c2diis_coeff_min, 
                                       coeffs > self.c2diis_coeff_max)
        has_extreme = np.any(extreme_values)
        
        quality = 1.0
        
        if has_extreme:
            print(f"Warning: Extreme C2DIIS coefficients detected: {[f'{c:.3f}' for c in coeffs]}")
            
            # Clip and renormalize
            clipped = np.clip(coeffs, self.c2diis_coeff_min, self.c2diis_coeff_max)
            sum_clipped = np.sum(clipped)
            
            if abs(sum_clipped - 1.0) > 1e-10 and sum_clipped > 1e-10:
                normalized = clipped / sum_clipped
            else:
                # Fall back to most recent point dominance
                normalized = np.zeros_like(coeffs)
                normalized[-1] = 0.7
                
                # Distribute remaining weight
                if len(coeffs) > 1:
                    for i in range(len(coeffs)-1):
                        normalized[i] = 0.3 / (len(coeffs)-1)
            
            # Reduce quality metric
            extreme_ratio = np.sum(np.abs(coeffs[extreme_values])) / np.sum(np.abs(coeffs))
            quality = max(0.2, 1.0 - extreme_ratio)
            
            self.c2diis_failure_count += 1
            return normalized, True, quality
        
        self.c2diis_failure_count = max(0, self.c2diis_failure_count - 1)
        return coeffs, False, quality
    
    def _calculate_c2diis_geometry(self):
        """
        Calculate new geometry using C2DIIS
        
        Returns:
        --------
        tuple
            (extrapolated_geometry, coeffs, success, quality)
        """
        n_points = len(self.geom_history)
        
        if n_points < self.c2diis_min_points:
            return None, None, False, 0.0
        
        # Reset history if too many failures
        if self.c2diis_failure_count >= self.c2diis_max_failures:
            print(f"Warning: {self.c2diis_failure_count} consecutive C2DIIS failures, resetting history")
            if len(self.geom_history) > 0:
                self.geom_history = [self.geom_history[-1]]
                self.grad_history = [self.grad_history[-1]]
                self.c2error_history = [self.c2error_history[-1]]
                self.quality_history = [1.0]
            
            self.c2diis_failure_count = 0
            self.c2diis_current_recovery = self.c2diis_recovery_steps
            self.c2diis_weight_current = max(0.2, self.c2diis_weight_current / 2)
            
            return None, None, False, 0.0
        
        try:
            # Calculate C2DIIS coefficients
            coeffs = self._solve_c2diis_equations()
            coeffs, was_filtered, quality = self._filter_coefficients(coeffs)
            
            # Calculate new geometry
            extrapolated_geometry = np.zeros_like(self.geom_history[0])
            for i in range(n_points):
                extrapolated_geometry += coeffs[i] * self.geom_history[i]
            
            # Check for NaN values
            if np.any(np.isnan(extrapolated_geometry)):
                print("Warning: NaN values in extrapolated geometry, C2DIIS calculation failed")
                self.c2diis_failure_count += 1
                return None, None, False, 0.0
            
            print("C2DIIS coefficients:", ", ".join(f"{c:.4f}" for c in coeffs))
            print(f"C2DIIS quality metric: {quality:.4f}")
            
            return extrapolated_geometry, coeffs, True, quality
            
        except Exception as e:
            print(f"C2DIIS extrapolation failed: {str(e)}")
            self.c2diis_failure_count += 1
            return None, None, False, 0.0
    
    def _validate_step(self, original_step, c2diis_step, gradient, quality):
        """
        Validate the C2DIIS step
        
        Parameters:
        -----------
        original_step : numpy.ndarray
            Original step
        c2diis_step : numpy.ndarray
            C2DIIS step
        gradient : numpy.ndarray
            Current gradient
        quality : float
            Quality from coefficient calculation
            
        Returns:
        --------
        tuple
            (is_valid, validation_quality)
        """
        # Check step size ratio
        original_norm = np.linalg.norm(original_step)
        c2diis_norm = np.linalg.norm(c2diis_step)
        
        if original_norm > 1e-10:
            step_ratio = c2diis_norm / original_norm
            if step_ratio > self.c2diis_step_ratio_max:
                print(f"C2DIIS step too large: {step_ratio:.2f} times Original step")
                return False, 0.0
            
            ratio_quality = 1.0 - min(1.0, abs(np.log10(step_ratio)))
        else:
            ratio_quality = 0.5
        
        # Check gradient alignment
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-10:
            neg_grad = -gradient / grad_norm
            c2diis_alignment = np.dot(c2diis_step.flatten(), neg_grad.flatten()) / np.linalg.norm(c2diis_step)
            
            # For C2DIIS, we're more lenient about alignment since squared error
            # may have different directional properties
            alignment_quality = 0.5 + 0.5 * max(-0.5, min(1.0, c2diis_alignment))
        else:
            alignment_quality = 0.5
        
        # Overall validation quality
        validation_quality = 0.6 * ratio_quality + 0.4 * alignment_quality
        
        # Only reject if validation quality is very low
        if validation_quality < 0.2:
            return False, 0.0
        
        return True, validation_quality
    
    def run(self, geom_num_list, gradient, pre_gradient, original_move_vector):
        """
        Run C2DIIS optimization step
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry
        gradient : numpy.ndarray
            Current gradient
        pre_gradient : numpy.ndarray
            Previous gradient
        original_move_vector : numpy.ndarray
            Original optimization step
            
        Returns:
        --------
        numpy.ndarray
            Optimized step vector
        """
        print("C2DIIS method")
        grad_rms = np.sqrt(np.mean(gradient ** 2))
        n_coords = len(geom_num_list)
        
        print(f"Gradient RMS: {grad_rms:.8f}")
        
        # Check convergence progress
        improving = grad_rms < self.prev_grad_rms * 0.95
        if improving:
            self.non_improving_count = 0
        else:
            self.non_improving_count += 1
            if self.non_improving_count > 2:
                self.c2diis_weight_current = max(0.1, self.c2diis_weight_current - 0.1)
                print(f"Optimization stalling, reducing C2DIIS weight to {self.c2diis_weight_current:.2f}")
                self.non_improving_count = 0
        
        self.prev_grad_rms = grad_rms
        
        # Calculate step quality
        step_quality = 1.0
        if self.iter > 0 and np.linalg.norm(pre_gradient) > 1e-10:
            grad_change_ratio = np.linalg.norm(gradient) / np.linalg.norm(pre_gradient)
            if grad_change_ratio < 1.0:
                step_quality = 1.0
            else:
                step_quality = max(0.3, 1.0 / (1.0 + np.log(grad_change_ratio)))
        
        # Update history
        self._update_history(geom_num_list, gradient, step_quality)
        
        # Skip if in recovery mode
        if self.c2diis_current_recovery > 0:
            self.c2diis_current_recovery -= 1
            print(f"In C2DIIS recovery mode ({self.c2diis_current_recovery} steps remaining)")
            move_vector = original_move_vector
        # Apply C2DIIS if enough history points
        elif len(self.geom_history) >= self.c2diis_min_points:
            # Calculate C2DIIS geometry
            c2diis_geom, c2diis_coeffs, success, quality = self._calculate_c2diis_geometry()
            
            if success and c2diis_geom is not None:
                # Calculate C2DIIS step
                c2diis_step = (c2diis_geom - geom_num_list).reshape(n_coords, 1)
                
                # Validate step
                is_valid, validation_quality = self._validate_step(original_move_vector, c2diis_step, gradient, quality)
                
                if is_valid:
                    # Calculate adaptive weight
                    if self.c2diis_failure_count > 0:
                        c2diis_weight = max(0.1, self.c2diis_weight_current - 
                                          self.c2diis_failure_count * self.c2diis_weight_decrement)
                    elif grad_rms < 0.01:
                        c2diis_weight = min(self.c2diis_weight_max,
                                          self.c2diis_weight_current + self.c2diis_weight_increment)
                    else:
                        c2diis_weight = self.c2diis_weight_current
                    
                    # Scale by validation quality
                    c2diis_weight *= validation_quality
                    
                    original_weight = 1.0 - c2diis_weight
                    
                    # Calculate blended step
                    move_vector = original_weight * original_move_vector + c2diis_weight * c2diis_step
                    print(f"Using blended step: {original_weight:.4f}*Original + {c2diis_weight:.4f}*C2DIIS")
                    
                    # Update current weight for next iteration
                    self.c2diis_weight_current = 0.7 * self.c2diis_weight_current + 0.3 * c2diis_weight
                else:
                    print("C2DIIS step validation failed, using Original step")
                    move_vector = original_move_vector
                    self.c2diis_failure_count += 1
            else:
                move_vector = original_move_vector
                if not success:
                    self.c2diis_failure_count += 1
        else:
            print(f"Building C2DIIS history ({len(self.geom_history)}/{self.c2diis_min_points} points)")
            move_vector = original_move_vector
        
        # Final safety check
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10:
            print("Warning: Step size too small, using scaled gradient")
            move_vector = -0.1 * gradient.reshape(n_coords, 1)
        elif np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Numerical issues in step, using scaled gradient")
            move_vector = -0.1 * gradient.reshape(n_coords, 1)
            # Reset history
            self.geom_history = []
            self.grad_history = []
            self.c2error_history = []
            self.quality_history = []
            
        self.iter += 1
        return move_vector