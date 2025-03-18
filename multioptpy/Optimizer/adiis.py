import numpy as np
from scipy.optimize import minimize


class ADIIS:
    """
    Implementation of ADIIS (Augmented DIIS) optimization method.
    
    ADIIS combines aspects of DIIS and EDIIS to provide robust convergence,
    particularly in early optimization stages.
    """
    def __init__(self):
        # ADIIS parameters
        self.adiis_history_size = 5        # History size
        self.adiis_min_points = 2          # Minimum points to start
        self.adiis_weight_initial = 0.4    # Initial weight
        self.adiis_weight_max = 0.8        # Maximum weight
        
        # Optimization parameters
        self.adiis_regularization = 1e-7   # Regularization parameter
        self.adiis_step_ratio_max = 3.0    # Maximum allowed step ratio
        
        # Error recovery
        self.adiis_failure_count = 0       # Failure counter
        self.adiis_max_failures = 2        # Max failures before reset
        self.adiis_recovery_steps = 2      # Recovery mode steps
        self.adiis_current_recovery = 0    # Current recovery counter
        
        # Weight adjustment
        self.adiis_weight_current = self.adiis_weight_initial  # Current weight
        self.adiis_weight_increment = 0.05  # Increment for success
        self.adiis_weight_decrement = 0.1   # Decrement for failure
        
        # History storage
        self.geom_history = []
        self.energy_history = []
        self.grad_history = []
        self.quality_history = []
        
        # Convergence monitoring
        self.prev_energy = float('inf')
        self.prev_grad_rms = float('inf')
        self.non_improving_count = 0
        self.iter = 0
    
    def _update_history(self, geometry, energy, gradient, step_quality=1.0):
        """
        Update the ADIIS history
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry
        energy : float
            Current energy
        gradient : numpy.ndarray
            Current gradient
        step_quality : float
            Quality metric for this point
        """
        # Add current point to history
        self.geom_history.append(geometry.copy())
        self.energy_history.append(energy)
        self.grad_history.append(gradient.copy())
        self.quality_history.append(step_quality)
        
        # If in recovery mode, limit history
        if self.adiis_current_recovery > 0:
            self.adiis_current_recovery -= 1
            if len(self.geom_history) > 2:
                self.geom_history = self.geom_history[-2:]
                self.energy_history = self.energy_history[-2:]
                self.grad_history = self.grad_history[-2:]
                self.quality_history = self.quality_history[-2:]
            return
        
        # Limit history size
        if len(self.geom_history) > self.adiis_history_size:
            if len(self.geom_history) > 2:
                # Calculate combined metric (energy and quality)
                metrics = []
                e_min = min(self.energy_history)
                e_range = max(self.energy_history) - e_min
                
                if e_range > 1e-10:
                    for i in range(len(self.energy_history)-1):  # Skip most recent point
                        e_metric = (self.energy_history[i] - e_min) / e_range
                        metrics.append(e_metric - 0.5 * self.quality_history[i])
                    
                    worst_idx = np.argmax(metrics)
                else:
                    # If energies are very close, use quality only
                    oldest_qualities = self.quality_history[:-1]
                    worst_idx = np.argmin(oldest_qualities)
                
                # Remove worst point
                self.geom_history.pop(worst_idx)
                self.energy_history.pop(worst_idx)
                self.grad_history.pop(worst_idx)
                self.quality_history.pop(worst_idx)
            else:
                # Default to removing oldest point
                self.geom_history.pop(0)
                self.energy_history.pop(0)
                self.grad_history.pop(0)
                self.quality_history.pop(0)
    
    def _solve_adiis_equations(self):
        """
        Solve ADIIS equations
        
        Returns:
        --------
        numpy.ndarray
            ADIIS coefficients
        """
        n_points = len(self.geom_history)
        
        if n_points < 2:
            return np.array([1.0])
            
        # Construct energy difference and gradient matrices
        E_diff = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    E_diff[i, j] = 0.0
                else:
                    dx = self.geom_history[j] - self.geom_history[i]
                    # Energy difference
                    e_diff = self.energy_history[j] - self.energy_history[i]
                    # Approximate first-order term
                    first_order = np.dot(self.grad_history[i].flatten(), dx.flatten())
                    # Augmentation: Include gradient dot product as approximate 
                    # second-order information
                    aug = np.dot(
                        (self.grad_history[j] - self.grad_history[i]).flatten(), 
                        dx.flatten()
                    )
                    # Combined energy difference with augmentation
                    E_diff[i, j] = e_diff - first_order
                    # Scale augmentation term based on previous performance
                    if self.adiis_failure_count > 0:
                        aug_scale = 0.5  # Reduce augmentation influence if having failures
                    else:
                        aug_scale = 1.0
                    
                    # Apply quality weighting
                    quality_factor = (self.quality_history[i] + self.quality_history[j]) / 2.0
                    
                    E_diff[i, j] *= quality_factor
                    E_diff[i, j] += aug_scale * aug * quality_factor
        
        try:
            # Define the ADIIS objective function
            def objective(x):
                # Quadratic term
                quad = 0.0
                for i in range(n_points):
                    for j in range(n_points):
                        quad += x[i] * x[j] * E_diff[i, j]
                
                # Add small regularization for stability
                reg = self.adiis_regularization * np.sum((x - 1.0/n_points)**2)
                
                return quad + reg
            
            # Constraint: sum of coefficients = 1
            def constraint(x):
                return np.sum(x) - 1.0
            
            # Non-negative constraints
            bounds = [(0, 1) for _ in range(n_points)]
            constraints = {'type': 'eq', 'fun': constraint}
            
            # Initial guess: equal weights
            x0 = np.ones(n_points) / n_points
            
            # Solve the constrained minimization problem
            result = minimize(
                objective, x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints, 
                options={'ftol': 1e-6, 'maxiter': 200}
            )
            
            if result.success:
                return result.x
            else:
                # Fall back to most recent point with small contributions from others
                fallback = np.zeros(n_points)
                fallback[-1] = 0.7  # 70% latest point
                remaining = 0.3
                if n_points > 1:
                    for i in range(n_points-1):
                        fallback[i] = remaining / (n_points-1)
                else:
                    fallback[0] = 1.0
                
                print("ADIIS equation solver failed, using fallback coefficients")
                return fallback
                
        except Exception as e:
            print(f"ADIIS solver exception: {str(e)}")
            # Fallback: exponential weighting favoring recent points
            fallback = np.zeros(n_points)
            total_weight = 0.0
            for i in range(n_points):
                fallback[i] = 2.0 ** i  # Exponential weighting
                total_weight += fallback[i]
            fallback /= total_weight
            
            return fallback
    
    def _calculate_adiis_geometry(self):
        """
        Calculate new geometry using ADIIS
        
        Returns:
        --------
        tuple
            (extrapolated_geometry, coeffs, success, quality)
        """
        n_points = len(self.geom_history)
        
        if n_points < self.adiis_min_points:
            return None, None, False, 0.0
        
        # Reset history if too many failures
        if self.adiis_failure_count >= self.adiis_max_failures:
            print(f"Warning: {self.adiis_failure_count} consecutive ADIIS failures, resetting history")
            if len(self.geom_history) > 0:
                self.geom_history = [self.geom_history[-1]]
                self.energy_history = [self.energy_history[-1]]
                self.grad_history = [self.grad_history[-1]]
                self.quality_history = [1.0]
            
            self.adiis_failure_count = 0
            self.adiis_current_recovery = self.adiis_recovery_steps
            self.adiis_weight_current = max(0.2, self.adiis_weight_current / 2)
            
            return None, None, False, 0.0
        
        try:
            # Calculate ADIIS coefficients
            coeffs = self._solve_adiis_equations()
            
            # Check coefficient validity
            if np.any(np.isnan(coeffs)) or abs(np.sum(coeffs) - 1.0) > 1e-4:
                print("Warning: Invalid ADIIS coefficients, using most recent point")
                coeffs = np.zeros(n_points)
                coeffs[-1] = 1.0
                quality = 0.5
            else:
                # Calculate quality based on coefficient distribution
                # For ADIIS, prefer solutions that give more weight to lower energy points
                energy_weighted_quality = 0.0
                e_min = min(self.energy_history)
                e_range = max(self.energy_history) - e_min
                
                if e_range > 1e-10:
                    for i in range(n_points):
                        # Normalize energy (0 = lowest, 1 = highest)
                        e_norm = (self.energy_history[i] - e_min) / e_range
                        # Higher quality for more weight on lower energy points
                        energy_weighted_quality += coeffs[i] * (1.0 - e_norm)
                else:
                    energy_weighted_quality = 0.5
                
                quality = max(0.3, energy_weighted_quality)
            
            # Calculate the new geometry as a linear combination
            extrapolated_geometry = np.zeros_like(self.geom_history[0])
            for i in range(n_points):
                extrapolated_geometry += coeffs[i] * self.geom_history[i]
            
            # Check for NaN values in the result
            if np.any(np.isnan(extrapolated_geometry)):
                print("Warning: NaN values in extrapolated geometry, ADIIS calculation failed")
                self.adiis_failure_count += 1
                return None, None, False, 0.0
            
            # Print coefficients
            print("ADIIS coefficients:", ", ".join(f"{c:.4f}" for c in coeffs))
            print(f"ADIIS quality metric: {quality:.4f}")
            
            return extrapolated_geometry, coeffs, True, quality
            
        except Exception as e:
            print(f"ADIIS extrapolation failed: {str(e)}")
            self.adiis_failure_count += 1
            return None, None, False, 0.0
    
    def _validate_adiis_step(self, original_step, adiis_step, gradient, energy):
        """
        Validate the ADIIS step
        
        Parameters:
        -----------
        original_step : numpy.ndarray
            Original step
        adiis_step : numpy.ndarray
            ADIIS step
        gradient : numpy.ndarray
            Current gradient
        energy : float
            Current energy
            
        Returns:
        --------
        tuple
            (is_valid, validation_quality)
        """
        # 1. Check step size ratio
        original_norm = np.linalg.norm(original_step)
        adiis_norm = np.linalg.norm(adiis_step)
        
        if original_norm > 1e-10:
            step_ratio = adiis_norm / original_norm
            if step_ratio > self.adiis_step_ratio_max:
                print(f"ADIIS step too large: {step_ratio:.2f} times Original step")
                return False, 0.0
            
            # Calculate quality based on step ratio
            ratio_quality = 1.0 - min(1.0, abs(np.log10(step_ratio)))
        else:
            ratio_quality = 0.5
        
        # 2. Check gradient alignment (ADIIS is more lenient here)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-10:
            neg_grad = -gradient / grad_norm
            original_alignment = np.dot(original_step.flatten(), neg_grad.flatten()) / np.linalg.norm(original_step)
            adiis_alignment = np.dot(adiis_step.flatten(), neg_grad.flatten()) / np.linalg.norm(adiis_step)
            
            # Only reject if ADIIS step is strongly opposing gradient while original step is downhill
            if original_alignment > 0.5 and adiis_alignment < -0.5:
                print(f"ADIIS step rejected: opposing gradient direction")
                return False, 0.0
                
            # Calculate alignment quality
            alignment_quality = 0.5 + 0.5 * max(-0.5, min(1.0, adiis_alignment))
        else:
            alignment_quality = 0.5
        
        # 3. Check energy improvement if we have history
        energy_quality = 0.5
        if len(self.energy_history) > 0:
            prev_best = min(self.energy_history)
            if energy < prev_best:
                # New lowest energy, excellent
                energy_quality = 1.0
            elif energy > max(self.energy_history):
                # Energy increased, poor
                energy_quality = 0.3
            else:
                # In between
                e_range = max(self.energy_history) - prev_best
                if e_range > 1e-10:
                    normalized_e = (energy - prev_best) / e_range
                    energy_quality = 1.0 - 0.5 * normalized_e
        
        # 4. Overall validation quality (weighted combination)
        validation_quality = 0.4 * ratio_quality + 0.3 * alignment_quality + 0.3 * energy_quality
        
        return True, validation_quality
    
    def run(self, geom_num_list, energy, gradient, original_move_vector):
        """
        Run ADIIS optimization step
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry
        energy : float
            Current energy
        gradient : numpy.ndarray
            Current gradient
        original_move_vector : numpy.ndarray
            Original optimization step
            
        Returns:
        --------
        numpy.ndarray
            Optimized step vector
        """
        print("ADIIS method")
        n_coords = len(geom_num_list)
        grad_rms = np.sqrt(np.mean(gradient ** 2))
        
        print(f"Energy: {energy:.8f}, Gradient RMS: {grad_rms:.8f}")
        
        # Track energy and gradient improvements
        energy_improving = energy < self.prev_energy
        grad_improving = grad_rms < self.prev_grad_rms * 0.95
        
        if energy_improving or grad_improving:
            self.non_improving_count = 0
        else:
            self.non_improving_count += 1
            if self.non_improving_count > 2:
                # Reduce ADIIS weight if optimization is stalling
                self.adiis_weight_current = max(0.1, self.adiis_weight_current - 0.1)
                print(f"Optimization stalling, reducing ADIIS weight to {self.adiis_weight_current:.2f}")
                self.non_improving_count = 0
        
        self.prev_energy = energy
        self.prev_grad_rms = grad_rms
        
        # Calculate step quality based on energy and gradient improvement
        step_quality = 1.0
        if self.iter > 0 and len(self.energy_history) > 0:
            # Average the energy and gradient quality factors
            if energy < min(self.energy_history):
                energy_quality = 1.0  # Best energy so far
            elif energy > max(self.energy_history):
                energy_quality = 0.5  # Worst energy so far
            else:
                # Scale based on where in the range it falls
                e_range = max(self.energy_history) - min(self.energy_history)
                if e_range > 1e-10:
                    e_norm = (energy - min(self.energy_history)) / e_range
                    energy_quality = 1.0 - 0.5 * e_norm
                else:
                    energy_quality = 0.7
            
            grad_quality = 1.0
            if len(self.grad_history) > 0 and np.linalg.norm(self.grad_history[-1]) > 1e-10:
                grad_change_ratio = grad_rms / np.sqrt(np.mean(self.grad_history[-1] ** 2))
                if grad_change_ratio < 1.0:
                    grad_quality = 1.0
                else:
                    grad_quality = max(0.3, 1.0 / (1.0 + np.log(grad_change_ratio)))
            
            # Combined quality metric
            step_quality = 0.7 * energy_quality + 0.3 * grad_quality
        
        # Update history
        self._update_history(geom_num_list, energy, gradient, step_quality)
        
        # Skip ADIIS if in recovery mode
        if self.adiis_current_recovery > 0:
            self.adiis_current_recovery -= 1
            print(f"In ADIIS recovery mode ({self.adiis_current_recovery} steps remaining), skipping ADIIS")
            move_vector = original_move_vector
        # Apply ADIIS if enough history has been accumulated
        elif len(self.geom_history) >= self.adiis_min_points:
            # Calculate ADIIS geometry
            adiis_geom, adiis_coeffs, success, quality = self._calculate_adiis_geometry()
            
            if success and adiis_geom is not None:
                # Calculate ADIIS step
                adiis_step = (adiis_geom - geom_num_list).reshape(n_coords, 1)
                
                # Validate step
                is_valid, validation_quality = self._validate_adiis_step(
                    original_move_vector, adiis_step, gradient, energy
                )
                
                if is_valid:
                    # Calculate adaptive weight
                    if self.adiis_failure_count > 0:
                        # Reduce weight if we've had failures
                        adiis_weight = max(0.1, self.adiis_weight_current - 
                                          self.adiis_failure_count * self.adiis_weight_decrement)
                    else:
                        # Otherwise use current weight, possibly increasing it
                        adiis_weight = min(self.adiis_weight_max, 
                                          self.adiis_weight_current + self.adiis_weight_increment)
                    
                    # Scale weight by validation quality
                    adiis_weight *= validation_quality
                    
                    original_weight = 1.0 - adiis_weight
                    
                    # Calculate blended step
                    move_vector = original_weight * original_move_vector + adiis_weight * adiis_step
                    print(f"Using blended step: {original_weight:.4f}*Original + {adiis_weight:.4f}*ADIIS")
                    
                    # Safety check: verify step size is reasonable
                    original_norm = np.linalg.norm(original_move_vector)
                    blended_norm = np.linalg.norm(move_vector)
                    
                    if blended_norm > 2.5 * original_norm and blended_norm > 0.3:
                        # Cap step size to avoid large jumps
                        print("Warning: ADIIS step too large, scaling down")
                        scale_factor = 2.5 * original_norm / blended_norm
                        move_vector = original_move_vector + scale_factor * (move_vector - original_move_vector)
                        print(f"Step scaled by {scale_factor:.3f}")
                    
                    # Update current weight for next iteration
                    self.adiis_weight_current = 0.7 * self.adiis_weight_current + 0.3 * adiis_weight
                else:
                    print("ADIIS step validation failed, using Original step only")
                    move_vector = original_move_vector
                    self.adiis_failure_count += 1
            else:
                move_vector = original_move_vector
                if not success:
                    self.adiis_failure_count += 1
        else:
            print(f"Building ADIIS history ({len(self.geom_history)}/{self.adiis_min_points} points)")
            move_vector = original_move_vector
        
        # Final safety checks
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10:
            print("Warning: Step size too small, using scaled gradient instead")
            move_vector = -0.1 * gradient.reshape(n_coords, 1)
        elif np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Numerical issues detected in step, using scaled gradient instead")
            move_vector = -0.1 * gradient.reshape(n_coords, 1)
            # Reset history on numerical failure
            self.geom_history = []
            self.energy_history = []
            self.grad_history = []
            self.quality_history = []
            self.adiis_failure_count = 0
        
        self.iter += 1
        return move_vector