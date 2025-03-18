import numpy as np


class EDIIS:
    def __init__(self):
        """
        ref.: The Journal of Chemical Physics, 2002, 116(19), 8255-8261.
              International Journal of Quantum Chemistry, 2000, 79(2), 82-90.
              The Journal of Chemical Physics, 2010, 132(5), 054109.
        """
        
        # EDIIS parameters with robust defaults
        self.ediis_history_size = 5       # Moderate history size
        self.ediis_min_points = 2         # Minimum points needed (just 2 for EDIIS)
        self.ediis_weight_initial = 0.3   # Start with moderate EDIIS contribution
        self.ediis_weight_max = 0.8       # Maximum EDIIS weight
        
        # Error handling parameters
        self.ediis_failure_count = 0      # Counter for consecutive EDIIS failures
        self.ediis_max_failures = 2       # Reset history after these many failures
        self.ediis_recovery_steps = 2     # Steps in recovery mode after failure
        self.ediis_current_recovery = 0   # Current recovery step counter
        
        # Adaptive weighting
        self.ediis_weight_current = self.ediis_weight_initial  # Current weight
        self.ediis_weight_increment = 0.05  # Increment for successful iterations
        self.ediis_weight_decrement = 0.15  # Decrement for failures
        
        # Robust optimization parameters
        self.ediis_regularization = 1e-8  # Regularization for numerical stability
        self.ediis_step_ratio_max = 2.5   # Maximum allowed ratio between EDIIS and Original steps
        
        # EDIIS history storage
        self.geom_history = []
        self.energy_history = []
        self.grad_history = []  # Store gradients too for energy extrapolation
        self.quality_history = []
        
        # Energy tracking for improvement monitoring
        self.prev_energy = float('inf')
        self.non_improving_count = 0
        self.iter = 0
        return
    
    def _update_ediis_history(self, geometry, energy, gradient, step_quality=1.0):
        """
        Update the EDIIS history with quality-based filtering
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry
        energy : float
            Current energy value
        gradient : numpy.ndarray
            Current gradient (used for energy extrapolation)
        step_quality : float
            Quality metric for this point (1.0 = good, <1.0 = lower quality)
        """
        # Add current point to history with quality metric
        self.geom_history.append(geometry.copy())
        self.energy_history.append(energy)
        self.grad_history.append(gradient.copy())
        self.quality_history.append(step_quality)
        
        # If in recovery mode, only keep the most recent points
        if self.ediis_current_recovery > 0:
            self.ediis_current_recovery -= 1
            if len(self.geom_history) > 2:
                self.geom_history = self.geom_history[-2:]
                self.energy_history = self.energy_history[-2:]
                self.grad_history = self.grad_history[-2:]
                self.quality_history = self.quality_history[-2:]
            return
        
        # Limit history size
        if len(self.geom_history) > self.ediis_history_size:
            # For EDIIS, prioritize keeping points with lower energies
            if len(self.geom_history) > 2:
                # Don't consider the most recent point for removal
                # Combine energy values and quality for decision
                metrics = []
                for i in range(len(self.energy_history)-1):
                    # Lower energy and higher quality are better
                    normalized_energy = (self.energy_history[i] - min(self.energy_history)) / \
                                       (max(self.energy_history) - min(self.energy_history) + 1e-10)
                    metrics.append(normalized_energy - 0.5 * self.quality_history[i])
                
                worst_idx = np.argmax(metrics)
                
                # Remove the worst point
                self.geom_history.pop(worst_idx)
                self.energy_history.pop(worst_idx)
                self.grad_history.pop(worst_idx)
                self.quality_history.pop(worst_idx)
            else:
                # Default to removing oldest point if we only have 2 points
                self.geom_history.pop(0)
                self.energy_history.pop(0)
                self.grad_history.pop(0)
                self.quality_history.pop(0)

    def _solve_ediis_equations(self):
        """
        Solve EDIIS equations with non-negative coefficients constraint
        
        Returns:
        --------
        numpy.ndarray
            EDIIS coefficients that minimize energy in the subspace
        """
        n_points = len(self.geom_history)
        
        if n_points < 2:
            return np.array([1.0])
        
        # Construct the energy difference matrix
        E_diff = np.zeros((n_points, n_points))
        
        # Calculate energy differences using energies and gradients
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    E_diff[i, j] = 0.0
                else:
                    # Energy difference from i to j:
                    # E_j ≈ E_i + g_i·(x_j-x_i) + 0.5·(x_j-x_i)·H·(x_j-x_i)
                    # We approximate the Hessian term as zero for simplicity
                    dx = self.geom_history[j] - self.geom_history[i]
                    E_diff[i, j] = self.energy_history[j] - self.energy_history[i] - \
                                  np.dot(self.grad_history[i].flatten(), dx.flatten())
        
        try:
            # Formulate the quadratic programming problem:
            # minimize 0.5*x^T*A*x + b^T*x subject to sum(x) = 1, x >= 0
            # where A is the energy difference matrix and b is zero
            
            # We'll use a simpler constrained minimization approach
            from scipy.optimize import minimize
            
            def objective(x):
                return 0.5 * np.sum(np.outer(x, x) * E_diff)
            
            def constraint(x):
                return np.sum(x) - 1.0
            
            bounds = [(0, 1) for _ in range(n_points)]
            constraints = {'type': 'eq', 'fun': constraint}
            
            # Initial guess: equal weights
            x0 = np.ones(n_points) / n_points
            
            # Solve the constrained minimization problem
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'ftol': 1e-6})
            
            if result.success:
                return result.x
            else:
                print("EDIIS solver failed, using uniform coefficients")
                return np.ones(n_points) / n_points
                
        except Exception as e:
            print(f"EDIIS equation solver failed: {str(e)}")
            # Fallback: use most recent point
            coeffs = np.zeros(n_points)
            coeffs[-1] = 1.0
            return coeffs
    
    def _calculate_ediis_geometry(self):
        """
        Calculate a new geometry using EDIIS with robust error handling
        
        Returns:
        --------
        tuple
            (extrapolated_geometry, coefficients, success, quality)
        """
        n_points = len(self.geom_history)
        
        if n_points < self.ediis_min_points:
            return None, None, False, 0.0
        
        # Reset history if we've had too many failures
        if self.ediis_failure_count >= self.ediis_max_failures:
            print(f"Warning: {self.ediis_failure_count} consecutive EDIIS failures, resetting history")
            # Keep only the most recent point
            if len(self.geom_history) > 0:
                self.geom_history = [self.geom_history[-1]]
                self.energy_history = [self.energy_history[-1]]
                self.grad_history = [self.grad_history[-1]]
                self.quality_history = [1.0]
            
            self.ediis_failure_count = 0
            self.ediis_current_recovery = self.ediis_recovery_steps
            self.ediis_weight_current = max(0.2, self.ediis_weight_current / 2)
            
            return None, None, False, 0.0
        
        try:
            # Calculate EDIIS coefficients
            coeffs = self._solve_ediis_equations()
            
            # Check coefficient validity
            if np.any(np.isnan(coeffs)) or abs(np.sum(coeffs) - 1.0) > 1e-4:
                print("Warning: Invalid EDIIS coefficients, using most recent point")
                coeffs = np.zeros(n_points)
                coeffs[-1] = 1.0
                quality = 0.5
            else:
                # Calculate quality metric based on coefficient distribution
                # We prefer solutions with fewer dominant points (unlike GDIIS)
                nonzero_count = np.sum(coeffs > 0.01)
                total_count = len(coeffs)
                # Quality is higher when we have fewer nonzero coefficients
                quality = 0.5 + 0.5 * (1.0 - nonzero_count / total_count)
            
            # Calculate the new geometry as a linear combination
            extrapolated_geometry = np.zeros_like(self.geom_history[0])
            for i in range(n_points):
                extrapolated_geometry += coeffs[i] * self.geom_history[i]
            
            # Check for NaN values in the result
            if np.any(np.isnan(extrapolated_geometry)):
                print("Warning: NaN values in extrapolated geometry, EDIIS calculation failed")
                self.ediis_failure_count += 1
                return None, None, False, 0.0
            
            # Print coefficients
            print("EDIIS coefficients:", ", ".join(f"{c:.4f}" for c in coeffs))
            print(f"EDIIS quality metric: {quality:.4f}")
            
            return extrapolated_geometry, coeffs, True, quality
            
        except Exception as e:
            print(f"EDIIS extrapolation failed: {str(e)}")
            self.ediis_failure_count += 1
            return None, None, False, 0.0

    def _validate_ediis_step(self, original_step, ediis_step, B_g):
        """
        Validate the EDIIS step to ensure it's reasonable
        
        Parameters:
        -----------
        original_step : numpy.ndarray
            Step calculated by the original method
        ediis_step : numpy.ndarray
            Step calculated by the EDIIS method
        B_g : numpy.ndarray
            Current gradient
            
        Returns:
        --------
        tuple
            (is_valid, validation_quality)
        """
        # 1. Check step size ratio
        original_norm = np.linalg.norm(original_step)
        ediis_norm = np.linalg.norm(ediis_step)
        
        if original_norm > 1e-10:
            step_ratio = ediis_norm / original_norm
            if step_ratio > self.ediis_step_ratio_max:
                print(f"EDIIS step too large: {step_ratio:.2f} times Original step")
                return False, 0.0
            
            # Calculate quality based on step ratio
            ratio_quality = 1.0 - min(1.0, abs(np.log10(step_ratio)))
        else:
            ratio_quality = 0.5
        
        # 2. Check gradient alignment
        grad_norm = np.linalg.norm(B_g)
        if grad_norm > 1e-10:
            # For EDIIS, we don't require strict downhill movement,
            # but extreme disagreement with gradient direction is suspicious
            neg_grad = -B_g / grad_norm
            original_alignment = np.dot(original_step.flatten(), neg_grad.flatten()) / np.linalg.norm(original_step)
            ediis_alignment = np.dot(ediis_step.flatten(), neg_grad.flatten()) / np.linalg.norm(ediis_step)
            
            # Only reject if EDIIS step is strongly opposing gradient while original step is downhill
            if original_alignment > 0.5 and ediis_alignment < -0.7:
                print(f"EDIIS step rejected: strongly opposing gradient direction")
                return False, 0.0
                
            # Calculate gradient alignment quality
            alignment_quality = 0.5 + 0.5 * max(-1.0, min(1.0, ediis_alignment))
        else:
            alignment_quality = 0.5
        
        # 3. Overall validation quality
        validation_quality = 0.4 * ratio_quality + 0.6 * alignment_quality
        
        return True, validation_quality

    def run(self, geom_num_list, energy, B_g, original_move_vector):
        """
        Run EDIIS optimization step
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry
        energy : float
            Current energy value
        B_g : numpy.ndarray
            Current gradient
        original_move_vector : numpy.ndarray
            Step calculated by the original method
            
        Returns:
        --------
        numpy.ndarray
            Optimized step vector
        """
        print("EDIIS method")
        n_coords = len(geom_num_list)
        
        # Track energy improvements
        improving = energy < self.prev_energy
        if improving:
            self.non_improving_count = 0
        else:
            self.non_improving_count += 1
            if self.non_improving_count > 2:
                # Reduce EDIIS weight if optimization is stalling
                self.ediis_weight_current = max(0.1, self.ediis_weight_current - 0.1)
                print(f"Energy not improving, reducing EDIIS weight to {self.ediis_weight_current:.2f}")
                self.non_improving_count = 0
        
        self.prev_energy = energy
        
        # Update EDIIS history with quality information
        step_quality = 1.0
        if self.iter > 0 and len(self.energy_history) > 0:
            # Calculate quality based on energy change
            if energy < min(self.energy_history):
                # New lowest energy, excellent quality
                step_quality = 1.0
            elif energy > max(self.energy_history):
                # Energy increased, poor quality
                step_quality = 0.5
            else:
                # In between, scale quality
                energy_range = max(self.energy_history) - min(self.energy_history)
                if energy_range > 1e-10:
                    normalized_energy = (energy - min(self.energy_history)) / energy_range
                    step_quality = 1.0 - 0.5 * normalized_energy
                else:
                    step_quality = 0.8
        
        self._update_ediis_history(geom_num_list, energy, B_g, step_quality)
        
        # Skip EDIIS if in recovery mode
        if self.ediis_current_recovery > 0:
            self.ediis_current_recovery -= 1
            print(f"In EDIIS recovery mode ({self.ediis_current_recovery} steps remaining), skipping EDIIS")
            move_vector = original_move_vector
        # Apply EDIIS if enough history has been accumulated
        elif len(self.geom_history) >= self.ediis_min_points:
            # Calculate EDIIS geometry
            ediis_geom, ediis_coeffs, success, quality = self._calculate_ediis_geometry()
            
            if success and ediis_geom is not None:
                # Calculate EDIIS step
                ediis_step = (ediis_geom - geom_num_list).reshape(n_coords, 1)
                
                # Validate EDIIS step
                is_valid, validation_quality = self._validate_ediis_step(original_move_vector, ediis_step, B_g)
                
                if is_valid:
                    # Calculate adaptive weight
                    if self.ediis_failure_count > 0:
                        # Reduce weight if we've had failures
                        ediis_weight = max(0.1, self.ediis_weight_current - self.ediis_failure_count * self.ediis_weight_decrement)
                    else:
                        # Otherwise use current weight, possibly increasing it
                        ediis_weight = min(self.ediis_weight_max, self.ediis_weight_current + self.ediis_weight_increment)
                    
                    # Scale weight by validation quality
                    ediis_weight *= validation_quality
                    
                    original_weight = 1.0 - ediis_weight
                    
                    # Calculate blended step
                    move_vector = original_weight * original_move_vector + ediis_weight * ediis_step
                    print(f"Using blended step: {original_weight:.4f}*Original + {ediis_weight:.4f}*EDIIS")
                    
                    # Update current weight for next iteration
                    self.ediis_weight_current = 0.7 * self.ediis_weight_current + 0.3 * ediis_weight
                else:
                    print("EDIIS step validation failed, using Original step only")
                    move_vector = original_move_vector
                    self.ediis_failure_count += 1
            else:
                move_vector = original_move_vector
                if not success:
                    self.ediis_failure_count += 1
        else:
            print(f"Building EDIIS history ({len(self.geom_history)}/{self.ediis_min_points} points)")
            move_vector = original_move_vector
        
        # Final safety checks
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10 or np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Step issue detected, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
            # Reset history on numerical failure
            if np.any(np.isnan(move_vector)):
                self.geom_history = []
                self.energy_history = []
                self.grad_history = []
                self.quality_history = []
        
        self.iter += 1
        return move_vector