import numpy as np


class GDIIS:
    def __init__(self):
        """
        ref.: Chemical Physics Letters, 1980, 73(2), 393-398.
              Journal of Molecular Structure, 1984, 114, 31-34.
              Physical Chemistry Chemical Physics, 2002, 4(1), 11-15.
        """
        # GDIIS parameters with enhanced defaults
        self.gdiis_history_size = 5        # Reduced history size for better stability
        self.gdiis_min_points = 3          # Require more points before starting GDIIS
        self.gdiis_error_threshold = 0.5   # More conservative error threshold
        self.gdiis_weight_initial = 0.1    # Start with lower GDIIS contribution
        self.gdiis_weight_max = 0.7        # Maximum GDIIS weight
        
        # Robust coefficient handling
        self.gdiis_coeff_min = -0.7        # Stricter minimum coefficient value
        self.gdiis_coeff_max = 0.7         # Stricter maximum coefficient value
        self.gdiis_regularization = 1e-8   # Increased regularization parameter
        
        # Enhanced error recovery
        self.gdiis_failure_count = 0       # Counter for consecutive GDIIS failures
        self.gdiis_max_failures = 2        # Reset history after fewer failures
        self.gdiis_recovery_steps = 3      # Number of steps in recovery mode
        self.gdiis_current_recovery = 0    # Current recovery step counter
        
        # Aggressive outlier detection
        self.gdiis_step_ratio_max = 2.0    # Maximum allowed ratio between GDIIS and Original steps
        self.gdiis_outlier_threshold = 3.0 # Standard deviations for outlier detection
        
        # Dynamic weight adjustment
        self.gdiis_weight_current = self.gdiis_weight_initial  # Current weight
        self.gdiis_weight_increment = 0.05  # Increment for successful iterations
        self.gdiis_weight_decrement = 0.15  # Larger decrement for failures
        
        # GDIIS history storage with quality metrics
        self.geom_history = []
        self.grad_history = []
        self.quality_history = []          # Track quality of each point
        
        # Convergence monitoring
        self.prev_grad_rms = float('inf')
        self.non_improving_count = 0
        self.iter = 0
        return
    
    def _update_gdiis_history(self, geometry, gradient, step_quality=1.0):
        """
        Update the GDIIS history with quality-based filtering
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry
        gradient : numpy.ndarray
            Current gradient
        step_quality : float
            Quality metric for this point (1.0 = good, <1.0 = lower quality)
        """
        # Add current point to history with quality metric
        self.geom_history.append(geometry.copy())
        self.grad_history.append(gradient.copy())
        self.quality_history.append(step_quality)
        
        # If in recovery mode, only keep the most recent points
        if self.gdiis_current_recovery > 0:
            self.gdiis_current_recovery -= 1
            if len(self.geom_history) > 2:
                self.geom_history = self.geom_history[-2:]
                self.grad_history = self.grad_history[-2:]
                self.quality_history = self.quality_history[-2:]
            return
        
        # Limit history size
        if len(self.geom_history) > self.gdiis_history_size:
            # Remove lowest quality point (except for the newest point)
            if len(self.geom_history) > 2:
                # Don't consider the most recent point for removal
                oldest_qualities = self.quality_history[:-1]
                worst_idx = np.argmin(oldest_qualities)
                
                # Remove the lowest quality point
                self.geom_history.pop(worst_idx)
                self.grad_history.pop(worst_idx)
                self.quality_history.pop(worst_idx)
            else:
                # Default to removing oldest point if we only have 2 points
                self.geom_history.pop(0)
                self.grad_history.pop(0)
                self.quality_history.pop(0)

    def _condition_b_matrix(self, B, n_points):
        """
        Apply advanced conditioning to improve B matrix stability
        
        Parameters:
        -----------
        B : numpy.ndarray
            The B matrix to condition
        n_points : int
            Number of actual data points
            
        Returns:
        --------
        numpy.ndarray
            Conditioned B matrix
        """
        # 1. Add regularization to diagonal for numerical stability
        np.fill_diagonal(B[:n_points, :n_points], 
                        np.diag(B[:n_points, :n_points]) + self.gdiis_regularization)
        
        # 2. Apply weighted regularization based on point quality
        if hasattr(self, 'quality_history') and len(self.quality_history) == n_points:
            for i in range(n_points):
                # Lower quality points get more regularization
                quality_factor = self.quality_history[i]
                B[i, i] += self.gdiis_regularization * (2.0 - quality_factor) / quality_factor
        
        # 3. Improve conditioning with SVD-based truncation
        try:
            # Apply SVD to the main block
            u, s, vh = np.linalg.svd(B[:n_points, :n_points])
            
            # Truncate small singular values (improves condition number)
            s_max = np.max(s)
            s_cutoff = s_max * 1e-10
            s_fixed = np.array([max(sv, s_cutoff) for sv in s])
            
            # Reconstruct with improved conditioning
            B_improved = np.dot(u * s_fixed, vh)
            
            # Put the improved block back
            B[:n_points, :n_points] = B_improved
        except:
            # If SVD fails, use simpler Tikhonov regularization
            identity = np.eye(n_points)
            B[:n_points, :n_points] += 1e-7 * identity
        
        return B

    def _solve_gdiis_equations(self, error_vectors, qualities=None):
        """
        Solve GDIIS equations with multiple robustness techniques
        """
        n_points = len(error_vectors)
        
        # Handle case of too few points
        if n_points < 2:
            return np.array([1.0])
        
        # Use quality weighting if available
        if qualities is None:
            qualities = np.ones(n_points)
        
        # Construct the B matrix with dot products of error vectors
        B = np.zeros((n_points + 1, n_points + 1))
        
        # Fill B matrix with weighted error vector dot products
        for i in range(n_points):
            for j in range(n_points):
                # Weight error dot products by quality
                weight_factor = np.sqrt(qualities[i] * qualities[j])
                B[i, j] = weight_factor * np.dot(error_vectors[i].T, error_vectors[j])
        
        # Apply advanced conditioning to the B matrix
        B = self._condition_b_matrix(B, n_points)
        
        # Add Lagrange multiplier constraints
        B[n_points, :n_points] = 1.0
        B[:n_points, n_points] = 1.0
        B[n_points, n_points] = 0.0
        
        # Right-hand side vector with constraint
        rhs = np.zeros(n_points + 1)
        rhs[n_points] = 1.0
        
        # Multi-stage solver with progressive fallbacks
        methods = [
            ("Standard solve", lambda: np.linalg.solve(B, rhs)),
            ("SVD solve", lambda: self._svd_solve(B, rhs, 1e-12)),
            ("Regularized solve", lambda: np.linalg.solve(B + np.diag([1e-6]*(n_points+1)), rhs)),
            ("Least squares", lambda: np.linalg.lstsq(B, rhs, rcond=1e-8)[0]),
            ("Minimal solution", lambda: self._minimal_solution(n_points))
        ]
        
        coefficients = None
        for method_name, solver in methods:
            try:
                coefficients = solver()
                # Check if solution is reasonable
                if not np.any(np.isnan(coefficients)) and np.abs(np.sum(coefficients[:n_points]) - 1.0) < 0.01:
                    print(f"GDIIS using {method_name}")
                    break
            except Exception as e:
                print(f"{method_name} failed: {str(e)}")
        
        # If all methods failed, default to using the most recent point
        if coefficients is None or np.any(np.isnan(coefficients)):
            print("All GDIIS solvers failed, using last point only")
            coefficients = np.zeros(n_points + 1)
            coefficients[n_points-1] = 1.0  # Use the most recent point
            coefficients[n_points] = 0.0    # Zero Lagrange multiplier
        
        # Extract actual coefficients (without Lagrange multiplier)
        return coefficients[:n_points]

    def _svd_solve(self, A, b, rcond=1e-15):
        """
        Solve linear system using SVD with improved handling of small singular values
        """
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        
        # More sophisticated singular value filtering
        s_max = np.max(s)
        mask = s > rcond * s_max
        
        # Create pseudo-inverse with smooth cutoff for small singular values
        s_inv = np.zeros_like(s)
        for i, (val, use) in enumerate(zip(s, mask)):
            if use:
                s_inv[i] = 1.0/val
            else:
                # Smooth transition to zero for small values
                ratio = val/(rcond * s_max)
                s_inv[i] = ratio/(val * (1.0 + (1.0 - ratio)**2))
        
        # Calculate solution using pseudo-inverse
        return np.dot(np.dot(np.dot(vh.T, np.diag(s_inv)), u.T), b)

    def _minimal_solution(self, n_points):
        """
        Fallback solution when all numerical methods fail
        """
        # Create a solution that gives higher weight to more recent points
        result = np.zeros(n_points + 1)
        
        # Linear ramp with highest weight to most recent point
        total_weight = 0
        for i in range(n_points):
            # Linear weighting: i+1 gives more weight to later points
            result[i] = i + 1
            total_weight += result[i]
        
        # Normalize to sum=1 and add zero Lagrange multiplier
        result[:n_points] /= total_weight
        return result

    def _filter_gdiis_coefficients(self, coeffs, strict=False):
        """
        Advanced filtering of extreme coefficient values
        
        Parameters:
        -----------
        coeffs : numpy.ndarray
            DIIS coefficients
        strict : bool
            Whether to use stricter filtering limits
        
        Returns:
        --------
        tuple
            (filtered_coeffs, was_filtered, quality_metric)
        """
        # Adjust bounds based on strictness
        coeff_min = self.gdiis_coeff_min * (1.5 if strict else 1.0)
        coeff_max = self.gdiis_coeff_max * (0.9 if strict else 1.0)
        
        # Check for extreme values
        extreme_values = np.logical_or(coeffs < coeff_min, coeffs > coeff_max)
        has_extreme_values = np.any(extreme_values)
        
        # Calculate quality metric (1.0 = perfect, lower values indicate problems)
        quality = 1.0
        if has_extreme_values:
            # Reduce quality based on how extreme the coefficients are
            extreme_ratio = np.sum(np.abs(coeffs[extreme_values])) / np.sum(np.abs(coeffs))
            quality = max(0.1, 1.0 - extreme_ratio)
            
            print(f"Warning: Extreme GDIIS coefficients detected: {[f'{c:.3f}' for c in coeffs]}")
            
            # Apply multi-stage filtering
            
            # 1. First attempt: Simple clipping and renormalization
            clipped_coeffs = np.clip(coeffs, coeff_min, coeff_max)
            sum_clipped = np.sum(clipped_coeffs)
            
            if abs(sum_clipped - 1.0) > 1e-10 and sum_clipped > 1e-10:
                normalized_coeffs = clipped_coeffs / sum_clipped
            else:
                # 2. If simple clipping failed, try redistribution approach
                print("Warning: Simple coefficient normalization failed, using redistribution")
                
                # Start with minimum values
                adjusted_coeffs = np.full_like(coeffs, coeff_min)
                
                # Distribute available weight (1.0 - sum(mins)) proportionally to valid coefficients
                valid_indices = ~extreme_values
                if np.any(valid_indices):
                    # Use only valid coefficients for distribution
                    valid_sum = np.sum(coeffs[valid_indices])
                    if abs(valid_sum) > 1e-10:
                        remaining = 1.0 - len(coeffs) * coeff_min
                        adjusted_coeffs[valid_indices] += remaining * (coeffs[valid_indices] / valid_sum)
                    else:
                        # If all valid coefficients sum to near zero, use uniform distribution
                        adjusted_coeffs = np.ones_like(coeffs) / len(coeffs)
                else:
                    # If all coefficients are extreme, use exponentially weighted recent points
                    n = len(coeffs)
                    for i in range(n):
                        adjusted_coeffs[i] = 0.5**min(n-i-1, 3)  # Exponentially weighted recent points
                    adjusted_coeffs /= np.sum(adjusted_coeffs)
                    
                normalized_coeffs = adjusted_coeffs
            
            # 3. Check if coefficients still have issues
            if np.any(np.isnan(normalized_coeffs)) or abs(np.sum(normalized_coeffs) - 1.0) > 1e-8:
                # Final fallback: use most recent point with small contributions from others
                print("Warning: Advanced filtering failed, falling back to recent-point dominated solution")
                n = len(coeffs)
                last_dominated = np.zeros_like(coeffs)
                last_dominated[-1] = 0.7  # 70% weight to most recent point
                
                # Distribute remaining 30% to other points
                remaining_weight = 0.3
                if n > 1:
                    for i in range(n-1):
                        last_dominated[i] = remaining_weight / (n-1)
                
                normalized_coeffs = last_dominated
            
            self.gdiis_failure_count += 1
            return normalized_coeffs, True, quality
        else:
            # Calculate quality based on coefficient distribution
            # Prefer solutions where coefficients are more evenly distributed
            n = len(coeffs)
            if n > 1:
                # Shannon entropy as a measure of coefficient distribution
                entropy = 0
                for c in coeffs:
                    if c > 0:
                        entropy -= c * np.log(c)
                
                # Normalize to [0,1] range
                max_entropy = np.log(n)
                if max_entropy > 0:
                    distribution_quality = min(1.0, entropy / max_entropy)
                    quality = 0.5 + 0.5 * distribution_quality
            
            self.gdiis_failure_count = max(0, self.gdiis_failure_count - 1)  # Reduce failure count on success
            return coeffs, False, quality

    def _calculate_gdiis_geometry(self):
        """
        Calculate a new geometry using GDIIS with comprehensive robustness measures
        """
        n_points = len(self.geom_history)
        
        if n_points < self.gdiis_min_points:
            return None, None, False, 0.0
        
        # Reset history if we've had too many failures
        if self.gdiis_failure_count >= self.gdiis_max_failures:
            print(f"Warning: {self.gdiis_failure_count} consecutive GDIIS failures, resetting history")
            # Keep only the most recent point
            if len(self.geom_history) > 0:
                self.geom_history = [self.geom_history[-1]]
                self.grad_history = [self.grad_history[-1]]
                self.quality_history = [1.0] if hasattr(self, 'quality_history') else []
            
            self.gdiis_failure_count = 0
            self.gdiis_current_recovery = self.gdiis_recovery_steps
            self.gdiis_weight_current = max(0.2, self.gdiis_weight_current / 2)  # Reduce weight
            
            return None, None, False, 0.0
        
        try:
            # Calculate GDIIS coefficients with comprehensive robustness measures
            if hasattr(self, 'quality_history') and len(self.quality_history) == n_points:
                qualities = self.quality_history
            else:
                qualities = np.ones(n_points)
            
            # First pass with standard filtering
            coeffs = self._solve_gdiis_equations(self.grad_history, qualities)
            coeffs, was_filtered, quality = self._filter_gdiis_coefficients(coeffs, strict=False)
            
            # If first pass needed filtering, try again with stricter limits
            if was_filtered:
                strict_coeffs = self._solve_gdiis_equations(self.grad_history, qualities)
                strict_coeffs, strict_filtered, strict_quality = self._filter_gdiis_coefficients(strict_coeffs, strict=True)
                
                # Use the better quality result
                if strict_quality > quality:
                    coeffs = strict_coeffs
                    quality = strict_quality
                    print("Using stricter coefficient filtering (better quality)")
            
            # Calculate the new geometry as a linear combination
            extrapolated_geometry = np.zeros_like(self.geom_history[0])
            for i in range(n_points):
                extrapolated_geometry += coeffs[i] * self.geom_history[i]
            
            # Check for NaN values in the result
            if np.any(np.isnan(extrapolated_geometry)):
                print("Warning: NaN values in extrapolated geometry, GDIIS calculation failed")
                self.gdiis_failure_count += 1
                return None, None, False, 0.0
            
            # Print coefficients (only if they're reasonable)
            print("GDIIS coefficients:", ", ".join(f"{c:.4f}" for c in coeffs))
            print(f"GDIIS quality metric: {quality:.4f}")
            
            return extrapolated_geometry, coeffs, True, quality
            
        except Exception as e:
            print(f"GDIIS extrapolation failed: {str(e)}")
            self.gdiis_failure_count += 1
            return None, None, False, 0.0

    def _validate_gdiis_step(self, original_step, gdiis_step, B_g, quality):
        """
        Comprehensive validation of the GDIIS step
        
        Parameters:
        -----------
        original_step : numpy.ndarray
            Step calculated by the Original method
        gdiis_step : numpy.ndarray
            Step calculated by the GDIIS method
        B_g : numpy.ndarray
            Current gradient
        quality : float
            Quality metric from coefficient calculation
            
        Returns:
        --------
        tuple
            (is_valid, validation_quality)
        """
        # 1. Check gradient alignment
        grad_norm = np.linalg.norm(B_g)
        if grad_norm > 1e-10:
            # Calculate normalized dot products with negative gradient
            neg_grad = -B_g / grad_norm
            original_alignment = np.dot(original_step.flatten(), neg_grad.flatten()) / np.linalg.norm(original_step)
            gdiis_alignment = np.dot(gdiis_step.flatten(), neg_grad.flatten()) / np.linalg.norm(gdiis_step)
            
            # GDIIS should point in a reasonable direction compared to Original
            if original_alignment > 0.3 and gdiis_alignment < 0:
                print(f"GDIIS step rejected: opposing gradient direction (Original: {original_alignment:.4f}, GDIIS: {gdiis_alignment:.4f})")
                return False, 0.0
        
        # 2. Check step size ratio
        original_norm = np.linalg.norm(original_step)
        gdiis_norm = np.linalg.norm(gdiis_step)
        
        if original_norm > 1e-10:
            step_ratio = gdiis_norm / original_norm
            if step_ratio > self.gdiis_step_ratio_max:
                print(f"GDIIS step too large: {step_ratio:.2f} times Original step")
                return False, 0.0
            
            # Calculate quality based on step ratio (closer to 1.0 is better)
            ratio_quality = 1.0 - min(1.0, abs(np.log10(step_ratio)))
        else:
            ratio_quality = 0.5  # Neutral if original step is near zero
        
        # 3. Check for outliers in the step components
        step_diff = gdiis_step - original_step
        mean_diff = np.mean(step_diff)
        std_diff = np.std(step_diff)
        
        if std_diff > 1e-10:
            # Check for components that are far from the mean difference
            outliers = np.abs(step_diff - mean_diff) > self.gdiis_outlier_threshold * std_diff
            outlier_fraction = np.sum(outliers) / len(step_diff)
            
            if outlier_fraction > 0.1:  # More than 10% of components are outliers
                print(f"GDIIS step rejected: {outlier_fraction*100:.1f}% of components are outliers")
                return False, 0.0
        
        # 4. Overall validation quality (combine multiple factors)
        validation_quality = (ratio_quality + quality) / 2.0
        
        return True, validation_quality

    def run(self, geom_num_list, B_g, pre_B_g, original_move_vector):
        print("GDIIS method")
        grad_rms = np.sqrt(np.mean(B_g ** 2))
        n_coords = len(geom_num_list)
        
        print(f"Gradient RMS: {grad_rms:.8f}")
        
        # Check convergence progress
        improving = grad_rms < self.prev_grad_rms * 0.95
        if improving:
            self.non_improving_count = 0
        else:
            self.non_improving_count += 1
            if self.non_improving_count > 2:
                # Reduce GDIIS weight if optimization is stalling
                self.gdiis_weight_current = max(0.1, self.gdiis_weight_current - 0.1)
                print(f"Optimization stalling, reducing GDIIS weight to {self.gdiis_weight_current:.2f}")
                self.non_improving_count = 0
        
        self.prev_grad_rms = grad_rms

        # Update GDIIS history with quality information
        step_quality = 1.0  # Default quality
        if self.iter > 0 and np.linalg.norm(pre_B_g) > 1e-10:
            # Estimate quality based on gradient reduction
            grad_change_ratio = np.linalg.norm(B_g) / np.linalg.norm(pre_B_g)
            if grad_change_ratio < 1.0:
                # Gradient decreased, good quality
                step_quality = 1.0
            else:
                # Gradient increased, lower quality
                step_quality = max(0.3, 1.0 / (1.0 + 2*np.log(grad_change_ratio)))
        
        self._update_gdiis_history(geom_num_list, B_g, step_quality)
        
        # Skip GDIIS if in recovery mode
        if self.gdiis_current_recovery > 0:
            self.gdiis_current_recovery -= 1
            print(f"In GDIIS recovery mode ({self.gdiis_current_recovery} steps remaining), skipping GDIIS")
            move_vector = original_move_vector
        # Apply GDIIS if enough history has been accumulated
        elif len(self.geom_history) >= self.gdiis_min_points:
            # Calculate GDIIS geometry with robust coefficient handling
            gdiis_geom, gdiis_coeffs, success, quality = self._calculate_gdiis_geometry()
            
            if success and gdiis_geom is not None:
                # Calculate GDIIS step
                gdiis_step = (gdiis_geom - geom_num_list).reshape(n_coords, 1)
                
                # Validate GDIIS step
                is_valid, validation_quality = self._validate_gdiis_step(original_move_vector, gdiis_step, B_g, quality)
                
                if is_valid:
                    # Calculate adaptive weight based on quality metrics
                    if self.gdiis_failure_count > 0:
                        # Reduce GDIIS weight if we've had failures
                        gdiis_weight = max(0.1, self.gdiis_weight_current - self.gdiis_failure_count * self.gdiis_weight_decrement)
                    elif grad_rms < 0.01:
                        # Increase GDIIS weight as we converge
                        gdiis_weight = min(self.gdiis_weight_max, self.gdiis_weight_current + self.gdiis_weight_increment)
                    elif self.non_improving_count > 0:
                        # Reduce weight if progress is stalling
                        gdiis_weight = max(0.1, self.gdiis_weight_current - 0.05 * self.non_improving_count)
                    else:
                        gdiis_weight = self.gdiis_weight_current
                    
                    # Scale weight by validation quality
                    gdiis_weight *= validation_quality
                    
                    original_weight = 1.0 - gdiis_weight
                    
                    # Calculate blended step
                    move_vector = original_weight * original_move_vector + gdiis_weight * gdiis_step
                    print(f"Using blended step: {original_weight:.4f}*Original + {gdiis_weight:.4f}*GDIIS")
                    
                    # Safety check: verify step size is reasonable
                    original_norm = np.linalg.norm(original_move_vector)
                    blended_norm = np.linalg.norm(move_vector)
                    
                    if blended_norm > 2.0 * original_norm and blended_norm > 0.3:
                        # Cap step size to avoid large jumps
                        print("Warning: GDIIS step too large, scaling down")
                        scale_factor = 2.0 * original_norm / blended_norm
                        move_vector = original_move_vector + scale_factor * (move_vector - original_move_vector)
                        print(f"Step scaled by {scale_factor:.3f}")
                    
                    # Update current weight for next iteration (with moderate memory)
                    self.gdiis_weight_current = 0.7 * self.gdiis_weight_current + 0.3 * gdiis_weight
                else:
                    print("GDIIS step validation failed, using Original step only")
                    move_vector = original_move_vector
                    self.gdiis_failure_count += 1
            else:
                # GDIIS failed
                move_vector = original_move_vector
                if not success:  # Only increment failure count for actual failures, not insufficient history
                    self.gdiis_failure_count += 1
        else:
            # Not enough history points yet, use standard Original
            print(f"Building GDIIS history ({len(self.geom_history)}/{self.gdiis_min_points} points), using Original step")
            move_vector = original_move_vector
        
        # Final safety check for step size and numerical issues
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10:
            print("Warning: Step size too small, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
        elif np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Numerical issues detected in step, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
            # Reset GDIIS history on numerical failure
            self.geom_history = []
            self.grad_history = []
            self.quality_history = []
            self.gdiis_failure_count = 0

        self.iter += 1

        return move_vector