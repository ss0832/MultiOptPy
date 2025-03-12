import numpy as np


class CoordinateLocking:
    """
    Coordinate Locking/Unlocking optimization method.
    
    This method selectively freezes (locks) and releases (unlocks) coordinates
    during optimization to improve convergence and control structural changes.
    """
    
    def __init__(self):
        # Basic parameters
        self.lock_mode = 'adaptive'        # 'manual', 'adaptive', 'scheduled', 'threshold'
        self.lock_schedule = 'gradual'     # 'gradual', 'stepwise', 'oscillating'
        self.lock_fraction_initial = 0.5   # Initial fraction of coordinates to lock
        self.lock_fraction_final = 0.0     # Final fraction of coordinates to lock
        self.lock_transition_steps = 10    # Steps to transition from initial to final
        
        # Adaptive locking parameters
        self.threshold_type = 'gradient'   # 'gradient', 'displacement', 'both'
        self.gradient_threshold = 0.05     # Lock coordinates with gradient below this
        self.displacement_threshold = 0.03 # Lock coordinates with displacement below this
        self.gradient_threshold_factor = 2.0  # Multiplier for adaptive gradient threshold
        
        # Group locking parameters
        self.use_groups = False           # Whether to lock coordinates in groups
        self.group_definitions = []       # List of coordinate groups (indices)
        self.group_coupling_threshold = 0.7  # Threshold for detecting coupled coordinates
        
        # Technical parameters
        self.update_frequency = 2         # How often to update locked coordinates
        self.lock_inertia = 3             # Steps a coordinate stays locked before review
        self.lock_max_fraction = 0.8      # Maximum fraction that can be locked at once
        
        # State variables
        self.lock_mask = None             # Boolean mask of locked coordinates (True = locked)
        self.lock_history = {}            # History of when coordinates were locked
        self.step_history = []            # History of previous steps
        self.gradient_history = []        # History of previous gradients
        self.current_lock_fraction = None # Current fraction of locked coordinates
        
        # Iteration tracking
        self.iter = 0
    
    def _initialize_lock_mask(self, n_coords):
        """
        Initialize the lock mask based on the current mode.
        
        Parameters:
        -----------
        n_coords : int
            Number of coordinates
        """
        if self.lock_mask is not None and len(self.lock_mask) == n_coords:
            return  # Already initialized
        
        # Create initial lock mask (False = unlocked, True = locked)
        self.lock_mask = np.zeros(n_coords, dtype=bool)
        self.lock_history = {i: 0 for i in range(n_coords)}
        
        # Initialize according to mode
        if self.lock_mode == 'manual':
            # In manual mode, all coordinates start unlocked
            pass
        elif self.lock_mode == 'scheduled' or self.lock_mode == 'adaptive':
            # Lock initial fraction of coordinates based on schedule
            self.current_lock_fraction = self.lock_fraction_initial
            self._apply_lock_fraction(n_coords)
            
        print(f"Lock mask initialized: {np.sum(self.lock_mask)}/{n_coords} coordinates locked")
        
    def _apply_lock_fraction(self, n_coords):
        """
        Apply the current lock fraction by locking coordinates.
        
        Parameters:
        -----------
        n_coords : int
            Number of coordinates
        """
        if self.current_lock_fraction is None:
            self.current_lock_fraction = self.lock_fraction_initial
        
        # Calculate how many coordinates should be locked
        target_locked = int(n_coords * min(self.current_lock_fraction, self.lock_max_fraction))
        currently_locked = np.sum(self.lock_mask)
        
        # Adjust locking
        if target_locked > currently_locked:
            # Need to lock more coordinates
            unlocked_indices = np.where(~self.lock_mask)[0]
            to_lock_count = min(target_locked - currently_locked, len(unlocked_indices))
            
            if to_lock_count > 0 and len(unlocked_indices) > 0:
                # Prefer locking coordinates with lower historical activity
                if len(self.gradient_history) > 0:
                    gradient_activity = np.zeros(n_coords)
                    for grad in self.gradient_history[-5:]:
                        gradient_activity += np.abs(grad.flatten())
                    
                    # Get indices of unlocked coordinates with lowest activity
                    activity_scores = gradient_activity[unlocked_indices]
                    lowest_activity_idx = np.argsort(activity_scores)[:to_lock_count]
                    to_lock = unlocked_indices[lowest_activity_idx]
                else:
                    # No history, just choose randomly
                    to_lock = np.random.choice(unlocked_indices, to_lock_count, replace=False)
                
                self.lock_mask[to_lock] = True
                for idx in to_lock:
                    self.lock_history[idx] = self.iter
        
        elif target_locked < currently_locked:
            # Need to unlock some coordinates
            locked_indices = np.where(self.lock_mask)[0]
            to_unlock_count = min(currently_locked - target_locked, len(locked_indices))
            
            if to_unlock_count > 0 and len(locked_indices) > 0:
                # Prefer unlocking coordinates with higher historical activity
                if len(self.gradient_history) > 0:
                    gradient_activity = np.zeros(n_coords)
                    for grad in self.gradient_history[-5:]:
                        gradient_activity += np.abs(grad.flatten())
                    
                    # Get indices of locked coordinates with highest activity
                    activity_scores = gradient_activity[locked_indices]
                    highest_activity_idx = np.argsort(activity_scores)[-to_unlock_count:]
                    to_unlock = locked_indices[highest_activity_idx]
                else:
                    # No history, just choose randomly
                    to_unlock = np.random.choice(locked_indices, to_unlock_count, replace=False)
                
                self.lock_mask[to_unlock] = False
                for idx in to_unlock:
                    self.lock_history[idx] = self.iter
    
    def _update_lock_fraction(self):
        """
        Update the current lock fraction based on the schedule.
        """
        if self.lock_mode != 'scheduled':
            return
        
        # Skip if we've reached the final value
        if self.current_lock_fraction == self.lock_fraction_final:
            return
            
        # Update based on schedule type
        if self.lock_schedule == 'gradual':
            # Linear interpolation
            progress = min(1.0, self.iter / self.lock_transition_steps)
            self.current_lock_fraction = self.lock_fraction_initial + progress * (
                self.lock_fraction_final - self.lock_fraction_initial)
                
        elif self.lock_schedule == 'stepwise':
            # Step changes at specific intervals
            step_interval = self.lock_transition_steps / 5
            num_steps = int(self.iter / step_interval)
            progress = min(1.0, num_steps / 5)
            self.current_lock_fraction = self.lock_fraction_initial + progress * (
                self.lock_fraction_final - self.lock_fraction_initial)
                
        elif self.lock_schedule == 'oscillating':
            # Oscillate between initial and final, eventually settling at final
            if self.iter > self.lock_transition_steps:
                self.current_lock_fraction = self.lock_fraction_final
            else:
                progress = self.iter / self.lock_transition_steps
                oscillation = 0.5 + 0.5 * np.cos(4 * np.pi * progress)
                oscillation_factor = 1.0 - progress
                self.current_lock_fraction = self.lock_fraction_final + (
                    oscillation * oscillation_factor * 
                    (self.lock_fraction_initial - self.lock_fraction_final))
    
    def _update_adaptive_locks(self, gradient, move_vector):
        """
        Update locked coordinates based on gradient and displacement thresholds.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
        move_vector : numpy.ndarray
            Current optimization step
        """
        if self.lock_mode != 'adaptive':
            return
            
        # Skip if we're not at an update iteration
        if self.iter % self.update_frequency != 0:
            return
            
        n_coords = len(gradient)
        grad_flat = gradient.flatten()
        move_flat = move_vector.flatten()
        
        # Calculate gradient and displacement magnitudes
        grad_mag = np.abs(grad_flat)
        move_mag = np.abs(move_flat)
        
        # Calculate adaptive thresholds
        if len(self.gradient_history) > 2:
            # Use median of recent history to set thresholds
            recent_grads = np.vstack([g.flatten() for g in self.gradient_history[-3:]])
            median_grad = np.median(np.abs(recent_grads))
            grad_threshold = median_grad / self.gradient_threshold_factor
        else:
            grad_threshold = self.gradient_threshold
            
        # Create masks for coordinates meeting thresholds
        low_grad_mask = grad_mag < grad_threshold
        low_move_mask = move_mag < self.displacement_threshold
        
        # Combine criteria based on threshold type
        if self.threshold_type == 'gradient':
            criteria_mask = low_grad_mask
        elif self.threshold_type == 'displacement':
            criteria_mask = low_move_mask
        else:  # 'both'
            criteria_mask = low_grad_mask & low_move_mask
        
        # Apply inertia: only lock/unlock if criteria have been met for multiple steps
        new_locks = []
        new_unlocks = []
        
        for i in range(n_coords):
            if criteria_mask[i] and not self.lock_mask[i]:
                # Potential new lock
                if i in self.lock_history and self.iter - self.lock_history[i] > self.lock_inertia:
                    new_locks.append(i)
                    self.lock_history[i] = self.iter
            elif not criteria_mask[i] and self.lock_mask[i]:
                # Potential new unlock
                if i in self.lock_history and self.iter - self.lock_history[i] > self.lock_inertia:
                    new_unlocks.append(i)
                    self.lock_history[i] = self.iter
        
        # Apply new locks and unlocks
        if new_locks:
            self.lock_mask[new_locks] = True
            print(f"Newly locked coordinates: {len(new_locks)}")
            
        if new_unlocks:
            self.lock_mask[new_unlocks] = False
            print(f"Newly unlocked coordinates: {len(new_unlocks)}")
    
    def _apply_group_locking(self, gradient):
        """
        Apply locking to coordinate groups based on coupling.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
        """
        if not self.use_groups:
            return
            
        if not self.group_definitions:
            # Try to detect coordinate groups automatically
            self._detect_coordinate_groups(gradient)
            
        # Apply consistent locking within groups
        for group in self.group_definitions:
            # Check if majority of group is locked
            lock_count = np.sum(self.lock_mask[group])
            if lock_count > len(group) / 2:
                # Lock the entire group
                self.lock_mask[group] = True
                
            # Check if majority of group is unlocked
            unlock_count = np.sum(~self.lock_mask[group])
            if unlock_count > len(group) / 2:
                # Unlock the entire group
                self.lock_mask[group] = False
    
    def _detect_coordinate_groups(self, gradient):
        """
        Attempt to automatically detect coordinate groups based on coupling.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
        """
        n_coords = len(gradient)
        
        # Need history to detect coupling
        if len(self.gradient_history) < 5:
            return
        
        # Calculate correlation matrix between coordinates using gradient history
        grad_history_array = np.vstack([g.flatten() for g in self.gradient_history[-5:]])
        
        if grad_history_array.shape[0] < 3:
            return
            
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
            group = {i}
            for j in range(n_coords):
                if j != i and j not in visited and abs(correlation[i, j]) > self.group_coupling_threshold:
                    group.add(j)
            
            if len(group) > 1:
                groups.append(list(group))
                visited.update(group)
        
        if groups:
            self.group_definitions = groups
            print(f"Detected {len(groups)} coordinate groups")
    
    def _apply_lock_mask(self, move_vector):
        """
        Apply the current lock mask to a move vector.
        
        Parameters:
        -----------
        move_vector : numpy.ndarray
            Original move vector
            
        Returns:
        --------
        numpy.ndarray
            Modified move vector with locked coordinates zeroed out
        """
        # Create a mask in the same shape as move_vector
        mask_expanded = np.logical_not(self.lock_mask).reshape(-1, 1)
        
        # Apply mask to zero out locked coordinates
        return move_vector * mask_expanded
    
    def run(self, geom_num_list, energy, gradient, original_move_vector):
        """
        Run Coordinate Locking optimization step.
        
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
            Modified optimization step
        """
        print("Coordinate Locking method")
        n_coords = len(geom_num_list)
        
        # Store gradient history
        self.gradient_history.append(gradient.copy())
        if len(self.gradient_history) > 10:
            self.gradient_history.pop(0)
        
        # Initialize or update lock mask
        if self.lock_mask is None:
            self._initialize_lock_mask(n_coords)
        
        # Update lock fraction based on schedule
        self._update_lock_fraction()
        
        # Apply scheduled lock fraction if applicable
        if self.lock_mode == 'scheduled':
            self._apply_lock_fraction(n_coords)
            
        # Update adaptive locks based on gradient and movement
        self._update_adaptive_locks(gradient, original_move_vector)
        
        # Apply group locking logic
        self._apply_group_locking(gradient)
        
        # Apply lock mask to move vector
        modified_move_vector = self._apply_lock_mask(original_move_vector)
        
        # Store step history
        self.step_history.append(modified_move_vector.copy())
        if len(self.step_history) > 10:
            self.step_history.pop(0)
        
        # Print status
        locked_count = np.sum(self.lock_mask)
        print(f"Locked coordinates: {locked_count}/{n_coords} ({locked_count/n_coords*100:.1f}%)")
        
        self.iter += 1
        return modified_move_vector