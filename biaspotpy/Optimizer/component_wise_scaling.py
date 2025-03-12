import numpy as np


class ComponentWiseScaling:
    """
    Component-wise scaling optimization method.
    
    This method applies individual scaling factors to each coordinate
    of the optimization step based on their characteristics and behavior.
    """
    
    def __init__(self):
        # Basic parameters
        self.scaling_strategy = 'adaptive'   # 'adaptive', 'gradient_based', 'history_based', 'curvature'
        self.scaling_mode = 'individual'     # 'individual', 'grouped', 'mixed'
        
        # Initial scaling parameters
        self.initial_scale_factor = 1.0      # Starting scale factor for all coordinates
        self.min_scale_factor = 0.01         # Minimum allowed scaling factor
        self.max_scale_factor = 5.0          # Maximum allowed scaling factor
        
        # Adaptive parameters
        self.adaptation_rate = 0.2           # How quickly scaling factors adapt (0-1)
        self.gradient_scaling_power = 0.5    # Power for gradient-based scaling
        self.history_memory = 5              # Number of previous steps to consider
        
        # Grouping parameters
        self.group_coordinates = False       # Whether to group similar coordinates
        self.group_similarity_threshold = 0.8 # Threshold for grouping coordinates
        self.coordinate_groups = []          # List of coordinate groups
        
        # Technical parameters
        self.normalize_gradients = True      # Whether to use normalized gradients
        self.use_global_damping = True       # Whether to apply global damping factor
        self.global_damping = 0.9            # Global damping factor
        self.outlier_threshold = 3.0         # Z-score threshold for outlier detection
        
        # State variables
        self.scaling_factors = None          # Current scaling factors
        self.gradient_history = []           # History of gradients
        self.step_history = []               # History of optimization steps
        self.factor_history = []             # History of scaling factors
        self.coordinate_activity = None      # Measure of each coordinate's activity
        
        # Iteration tracking
        self.iter = 0
    
    def _initialize_scaling_factors(self, n_coords):
        """
        Initialize the scaling factors based on the current strategy.
        
        Parameters:
        -----------
        n_coords : int
            Number of coordinates
        """
        if self.scaling_factors is not None and len(self.scaling_factors) == n_coords:
            return  # Already initialized
        
        # Create initial scaling factors
        self.scaling_factors = np.ones(n_coords) * self.initial_scale_factor
        self.coordinate_activity = np.ones(n_coords)
        
        print(f"Scaling factors initialized with value {self.initial_scale_factor}")
    
    def _update_coordinate_activity(self, gradient):
        """
        Update the measure of each coordinate's activity based on gradients.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
        """
        grad_flat = gradient.flatten()
        grad_abs = np.abs(grad_flat)
        
        # First iteration or reset
        if self.coordinate_activity is None or len(self.coordinate_activity) != len(grad_flat):
            self.coordinate_activity = grad_abs.copy()
            return
        
        # Update activity with exponential smoothing
        memory_factor = 0.8
        self.coordinate_activity = (memory_factor * self.coordinate_activity + 
                                    (1 - memory_factor) * grad_abs)
    
    def _apply_gradient_based_scaling(self, gradient):
        """
        Update scaling factors based on gradient magnitudes.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
        """
        grad_flat = gradient.flatten()
        grad_abs = np.abs(grad_flat)
        
        # Avoid division by zero
        grad_abs = np.maximum(grad_abs, 1e-10)
        
        if self.normalize_gradients:
            # Normalized gradient-based scaling
            mean_grad = np.mean(grad_abs)
            if mean_grad > 1e-10:
                # Calculate normalized gradient magnitudes
                normalized_grad = grad_abs / mean_grad
                
                # Apply power scaling to get factors
                scale_factors = normalized_grad ** (-self.gradient_scaling_power)
                
                # Limit to reasonable range
                scale_factors = np.clip(scale_factors, self.min_scale_factor, self.max_scale_factor)
                
                # Update with smoothing
                self.scaling_factors = ((1 - self.adaptation_rate) * self.scaling_factors + 
                                        self.adaptation_rate * scale_factors)
        else:
            # Direct gradient-based scaling (inverse relationship)
            # Coordinates with larger gradients get smaller steps
            scale_factors = 1.0 / (grad_abs ** self.gradient_scaling_power)
            
            # Normalize to keep average scaling around 1.0
            scale_factors = scale_factors / np.mean(scale_factors)
            
            # Limit to reasonable range
            scale_factors = np.clip(scale_factors, self.min_scale_factor, self.max_scale_factor)
            
            # Update with smoothing
            self.scaling_factors = ((1 - self.adaptation_rate) * self.scaling_factors + 
                                    self.adaptation_rate * scale_factors)
    
    def _apply_history_based_scaling(self):
        """
        Update scaling factors based on the history of gradients and steps.
        """
        if len(self.gradient_history) < 2 or len(self.step_history) < 2:
            return  # Not enough history
        
        # Calculate gradient changes and step sizes from recent history
        recent_grads = self.gradient_history[-min(self.history_memory, len(self.gradient_history)):]
        recent_steps = self.step_history[-min(self.history_memory, len(self.step_history)):]
        
        # Calculate coordinate-wise statistics
        grad_changes = np.zeros(len(self.scaling_factors))
        step_sizes = np.zeros(len(self.scaling_factors))
        
        for i in range(1, len(recent_grads)):
            grad_diff = recent_grads[i].flatten() - recent_grads[i-1].flatten()
            grad_changes += np.abs(grad_diff)
            step_sizes += np.abs(recent_steps[i-1].flatten())
        
        # Avoid division by zero
        step_sizes = np.maximum(step_sizes, 1e-10)
        
        # Calculate scaling adjustments based on gradient changes per step size
        # Coordinates where gradients change rapidly with small steps get smaller scaling
        effectiveness = grad_changes / step_sizes
        effectiveness = np.maximum(effectiveness, 1e-10)
        
        # Normalize effectiveness
        mean_effectiveness = np.mean(effectiveness)
        if mean_effectiveness > 1e-10:
            relative_effectiveness = effectiveness / mean_effectiveness
            
            # Calculate scale adjustments (inverse relationship)
            scale_adjustments = relative_effectiveness ** (-0.5)
            
            # Limit adjustments to reasonable range
            scale_adjustments = np.clip(scale_adjustments, 0.5, 2.0)
            
            # Apply adjustments with smoothing
            self.scaling_factors *= (1.0 + self.adaptation_rate * (scale_adjustments - 1.0))
            
            # Final bounds check
            self.scaling_factors = np.clip(self.scaling_factors, 
                                           self.min_scale_factor, 
                                           self.max_scale_factor)
    
    def _apply_curvature_based_scaling(self):
        """
        Update scaling factors based on estimated local curvature.
        """
        if len(self.gradient_history) < 3:
            return  # Not enough history for curvature estimation
        
        # Get recent gradients
        g_current = self.gradient_history[-1].flatten()
        g_prev = self.gradient_history[-2].flatten()
        g_prev2 = self.gradient_history[-3].flatten()
        
        # Finite difference approximation of curvature
        # Using central difference: f''(x) ≈ (f'(x+h) - 2f'(x) + f'(x-h)) / h²
        # Here we use gradient values as f'(x) at different points
        curvature_approx = np.abs(g_current - 2*g_prev + g_prev2)
        
        # Avoid division by zero
        curvature_approx = np.maximum(curvature_approx, 1e-10)
        
        # Scale inversely with curvature - higher curvature needs smaller steps
        curvature_scaling = 1.0 / np.sqrt(curvature_approx)
        
        # Normalize to keep average scaling around 1.0
        mean_scaling = np.mean(curvature_scaling)
        if mean_scaling > 1e-10:
            curvature_scaling /= mean_scaling
        
        # Clip to reasonable range
        curvature_scaling = np.clip(curvature_scaling, 0.2, 5.0)
        
        # Apply with smoothing
        self.scaling_factors = ((1 - self.adaptation_rate) * self.scaling_factors + 
                                self.adaptation_rate * curvature_scaling)
        
        # Final bounds check
        self.scaling_factors = np.clip(self.scaling_factors, 
                                       self.min_scale_factor, 
                                       self.max_scale_factor)
    
    def _detect_and_handle_outliers(self):
        """
        Detect and handle outliers in scaling factors.
        """
        # Calculate z-scores
        mean_scale = np.mean(self.scaling_factors)
        std_scale = np.std(self.scaling_factors)
        
        if std_scale < 1e-10:
            return  # No significant variation
        
        z_scores = np.abs((self.scaling_factors - mean_scale) / std_scale)
        
        # Identify outliers
        outliers = z_scores > self.outlier_threshold
        outlier_count = np.sum(outliers)
        
        if outlier_count > 0:
            print(f"Detected {outlier_count} outlier scaling factors")
            
            # Handle outliers by bringing them closer to the mean
            adjustment = 0.5 * (self.scaling_factors[outliers] - mean_scale)
            self.scaling_factors[outliers] -= adjustment
    
    def _apply_group_scaling(self):
        """
        Apply consistent scaling to groups of coordinates.
        """
        if not self.group_coordinates or not self.coordinate_groups:
            # Try to detect groups if enabled but not defined
            if self.group_coordinates and not self.coordinate_groups and len(self.gradient_history) >= 3:
                self._detect_coordinate_groups()
            else:
                return  # No grouping or groups not yet detected
        
        # Apply consistent scaling within each group
        for group in self.coordinate_groups:
            group_scales = self.scaling_factors[group]
            median_scale = np.median(group_scales)  # Use median for robustness
            
            # Move all scales in the group towards the median
            adjustment_rate = 0.7  # How strongly to enforce group consistency
            self.scaling_factors[group] = (adjustment_rate * median_scale + 
                                          (1 - adjustment_rate) * group_scales)
    
    def _detect_coordinate_groups(self):
        """
        Attempt to automatically detect coordinate groups based on gradient correlation.
        """
        # Need enough history to detect correlations
        if len(self.gradient_history) < 3:
            return
        
        n_coords = len(self.scaling_factors)
        
        # Calculate correlation matrix between coordinates using gradient history
        grad_history_array = np.vstack([g.flatten() for g in self.gradient_history[-5:]])
        
        # Calculate correlation coefficient matrix
        correlation = np.corrcoef(grad_history_array.T)
        
        # Replace NaN with 0
        correlation = np.nan_to_num(correlation)
        
        # Find highly correlated coordinates
        groups = []
        visited = set()
        
        for i in range(n_coords):
            if i in visited:
                continue
                
            # Find coordinates highly correlated with i
            group = [i]
            for j in range(n_coords):
                if j != i and j not in visited and abs(correlation[i, j]) > self.group_similarity_threshold:
                    group.append(j)
            
            if len(group) > 1:
                groups.append(group)
                visited.update(group)
        
        if groups:
            self.coordinate_groups = groups
            print(f"Detected {len(groups)} coordinate groups for consistent scaling")
    
    def _apply_scaling_to_step(self, move_vector):
        """
        Apply the current scaling factors to a move vector.
        
        Parameters:
        -----------
        move_vector : numpy.ndarray
            Original move vector
            
        Returns:
        --------
        numpy.ndarray
            Scaled move vector
        """
        # Create a scaling factors array in the same shape as move_vector
        scaling_expanded = self.scaling_factors.reshape(-1, 1)
        
        # Apply scaling factors
        scaled_vector = move_vector * scaling_expanded
        
        # Apply global damping if enabled
        if self.use_global_damping:
            scaled_vector *= self.global_damping
        
        return scaled_vector
    
    def run(self, geom_num_list, energy, gradient, original_move_vector):
        """
        Run Component-wise Scaling optimization step.
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry
        energy : float
            Current energy value
        gradient : numpy.ndarray
            Current gradient
        original_move_vector : numpy.ndarray
            Original optimization step
            
        Returns:
        --------
        numpy.ndarray
            Scaled optimization step
        """
        print("Component-wise Scaling method")
        n_coords = len(geom_num_list)
        
        # Store gradient history
        self.gradient_history.append(gradient.copy())
        if len(self.gradient_history) > 2 * self.history_memory:
            self.gradient_history.pop(0)
        
        # Initialize scaling factors if needed
        if self.scaling_factors is None:
            self._initialize_scaling_factors(n_coords)
        
        # Update coordinate activity measure
        self._update_coordinate_activity(gradient)
        
        # Apply scaling strategy
        if self.scaling_strategy == 'gradient_based' or self.scaling_strategy == 'adaptive':
            self._apply_gradient_based_scaling(gradient)
        
        if self.scaling_strategy == 'history_based' or self.scaling_strategy == 'adaptive':
            self._apply_history_based_scaling()
        
        if self.scaling_strategy == 'curvature':
            self._apply_curvature_based_scaling()
        
        # Detect and handle outlier scaling factors
        self._detect_and_handle_outliers()
        
        # Apply group scaling if enabled
        if self.group_coordinates:
            self._apply_group_scaling()
        
        # Apply scaling to the move vector
        scaled_move_vector = self._apply_scaling_to_step(original_move_vector)
        
        # Store step history
        self.step_history.append(scaled_move_vector.copy())
        if len(self.step_history) > 2 * self.history_memory:
            self.step_history.pop(0)
        
        # Store factor history
        self.factor_history.append(self.scaling_factors.copy())
        if len(self.factor_history) > 2 * self.history_memory:
            self.factor_history.pop(0)
        
        # Print scaling statistics
        min_scale = np.min(self.scaling_factors)
        max_scale = np.max(self.scaling_factors)
        mean_scale = np.mean(self.scaling_factors)
        print(f"Scaling factors - min: {min_scale:.4f}, max: {max_scale:.4f}, mean: {mean_scale:.4f}")
        
        self.iter += 1
        return scaled_move_vector