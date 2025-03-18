from .linesearch import LineSearch
import numpy as np


class LBFGS:
    """Limited-memory BFGS optimizer for BiasPotPy.
    
    Implementation based on Nocedal & Wright - Numerical Optimization, 2006.
    """
    
    def __init__(self, **config):
        """Initialize the L-BFGS optimizer.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with the following options:
            - keep_last: Number of previous steps to store (default: 7)
            - beta: Force constant for initial Hessian approximation (default: 1.0)
            - max_step: Maximum step size (default: 0.2)
            - double_damp: Whether to use double damping (default: True)
            - gamma_mult: Whether to estimate beta from previous cycle (default: False)
            - line_search: Whether to use line search (default: False)
            - mu_reg: Regularization parameter (default: None)
            - max_mu_reg_adaptions: Maximum number of regularization adaptations (default: 10)
            - control_step: Whether to control step size (default: True)
            - momentum: Whether to use momentum-based optimization (default: False)
            - beta_momentum: Momentum coefficient (default: 0.9)
        """
        self.config = config
        self.Initialization = True
        
        # L-BFGS specific parameters
        self.keep_last = config.get("keep_last", 7)
        self.beta = config.get("beta", 1.0)
        self.max_step = config.get("max_step", 0.5)
        self.oscillation_threshold = config.get("oscillation_threshold", 2)
        self.energy_oscillation_factor= 0.5
        self.min_step = 0.1
        self.double_damp = config.get("double_damp", True)
        self.gamma_mult = config.get("gamma_mult", False)
        self.use_line_search = config.get("line_search", False)
        self.oscillation_memory = config.get("oscillation_memory", 10)
        self.mu_reg = config.get("mu_reg", None)
        self.max_mu_reg_adaptions = config.get("max_mu_reg_adaptions", 10)
        self.control_step = config.get("control_step", True)
        
        # Storage for L-BFGS history
        self.coord_diffs = []  # s vectors
        self.grad_diffs = []   # y vectors
               
        # For step control and iteration tracking
        self.DELTA = config.get("delta", 0.1)  # Step scaling factor
        self.FC_COUNT = config.get("fc_count", -1)  # Frequency for recomputing Hessian
        self.saddle_order = config.get("saddle_order", 0)
        self.iter = 0
        self.tot_adapt_mu_cycles = 0
        
        # Regularization setup
        if self.mu_reg:
            self.mu_reg_0 = self.mu_reg
            self.control_step = False
            self.double_damp = False
            self.use_line_search = False
            print(f"Regularized-L-BFGS (μ_reg={self.mu_reg:.6f}) requested.")
            print("Disabling double damping, step control and line search.")
        
        # Placeholders for required attributes
        self.hessian = None
        self.bias_hessian = None
        self.optimal_step_flag = False
        self.prev_move_vector = None
        self.linesearchflag = self.use_line_search
        self.energy_history = []
        
    def detect_energy_oscillation(self, new_energy):
        """
        Detect energy oscillation patterns using the full energy history.
        
        This method analyzes the entire energy history to detect oscillatory
        behavior, rather than just looking at the most recent points.
        
        Parameters
        ----------
        new_energy : float
            The newest energy value to add to history
            
        Returns
        -------
        bool
            True if oscillation is detected, False otherwise
        """
        # Add new energy to history
        self.energy_history.append(new_energy)
        
        # Keep history within memory limit
        if len(self.energy_history) > self.oscillation_memory:
            self.energy_history.pop(0)
        
        # Need at least 3 points to detect oscillations
        if len(self.energy_history) < 3:
            return False
        
        # Count direction changes in full history
        direction_changes = 0
        for i in range(1, len(self.energy_history)-1):
            prev_diff = self.energy_history[i] - self.energy_history[i-1]
            next_diff = self.energy_history[i+1] - self.energy_history[i]
            
            # Check if direction changed (from increasing to decreasing or vice versa)
            if (prev_diff * next_diff) < 0:
                direction_changes += 1
        
        # Detect longer-term trends
        increasing_trend = True
        decreasing_trend = True
        
        for i in range(1, len(self.energy_history)):
            if self.energy_history[i] <= self.energy_history[i-1]:
                increasing_trend = False
            if self.energy_history[i] >= self.energy_history[i-1]:
                decreasing_trend = False
        
        # Oscillation detected if:
        # 1. There are multiple direction changes exceeding threshold, or
        # 2. Energy is not consistently increasing or decreasing
        is_oscillating = (direction_changes >= self.oscillation_threshold) or \
                        (not increasing_trend and not decreasing_trend and len(self.energy_history) >= 4)
        
        if is_oscillating:
            print(f"Energy oscillation detected. Direction changes: {direction_changes}")
            # Also print the energy series for debugging
            energy_str = " → ".join([f"{e:.6f}" for e in self.energy_history[-5:]])
            print(f"Recent energy values: {energy_str}")
        
        return is_oscillating


    def set_hessian(self, hessian):
        """Store Hessian matrix (not used in L-BFGS but required for interface)."""
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        """Store bias Hessian matrix (not used in L-BFGS but required for interface)."""
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        """Return Hessian matrix (placeholder for interface)."""
        return self.hessian
    
    def get_bias_hessian(self):
        """Return bias Hessian matrix (placeholder for interface)."""
        return self.bias_hessian
    
    def reset(self):
        """Reset the L-BFGS memory."""
        self.coord_diffs = []
        self.grad_diffs = []
        self.iter = 0
        self.tot_adapt_mu_cycles = 0
        self.Initialization = True
    
    def double_damp_vectors(self, s, y):
        """Apply double damping to ensure s·y > 0.
        
        References
        ----------
        [1] Prier, Edwards (2009). Quasi-Newton Methods, ensuring sy > 0
        """
        sy = np.dot(s.flatten(), y.flatten())
        if sy > 0:
            return s, y
        
        # Collect inner products of s and y with existing vectors
        ss = [np.dot(s.flatten(), si.flatten()) for si in self.coord_diffs]
        sy_list = [np.dot(s.flatten(), yi.flatten()) for yi in self.grad_diffs]
        ys = [np.dot(y.flatten(), si.flatten()) for si in self.coord_diffs]
        yy = [np.dot(y.flatten(), yi.flatten()) for yi in self.grad_diffs]
        
        if not (ss and sy_list and ys and yy):
            # If lists are empty, can't do double damping
            return s, y
        
        # First damping: Powell
        if sy <= 0:
            theta = 0.8 * np.abs(sy) / (np.abs(sy) + np.max(np.abs(sy_list)))
            y_damp = (1 - theta) * y + theta * self.grad_diffs[-1]
            sy = np.dot(s.flatten(), y_damp.flatten())
        else:
            y_damp = y
        
        # Second damping: Barzilai & Borwein
        if sy <= 0:
            theta = 0.5 * np.abs(sy) / (np.abs(sy) + np.max(np.abs(ys)))
            s_damp = (1 - theta) * s + theta * self.coord_diffs[-1]
        else:
            s_damp = s
            
        return s_damp, y_damp
    
    def bfgs_multiply(self, forces):
        """Calculate BFGS step using two-loop recursion algorithm.
        
        Parameters
        ----------
        forces : np.ndarray
            Current forces (negative gradient)
            
        Returns
        -------
        np.ndarray
            Step direction
        """
        q = forces.copy().reshape(-1, 1)
        alphas = []
        
        # First loop: descending from most recent to oldest
        for i in range(len(self.coord_diffs) - 1, -1, -1):
            s = self.coord_diffs[i].reshape(-1, 1)
            y = self.grad_diffs[i].reshape(-1, 1)
            rho_i = 1.0 / np.dot(y.T, s)
            alpha_i = rho_i * np.dot(s.T, q)
            alphas.append(alpha_i[0, 0])
            q -= alpha_i * y
        
        # Middle step with initial Hessian approximation
        if len(self.coord_diffs) > 0 and self.gamma_mult:
            s = self.coord_diffs[-1].reshape(-1, 1)
            y = self.grad_diffs[-1].reshape(-1, 1)
            gamma = np.dot(y.T, s) / np.dot(y.T, y)
            r = gamma * q
        else:
            r = self.beta * q
        
        # Second loop: ascending from oldest to most recent
        for i in range(len(self.coord_diffs)):
            s = self.coord_diffs[i].reshape(-1, 1)
            y = self.grad_diffs[i].reshape(-1, 1)
            rho_i = 1.0 / np.dot(y.T, s)
            beta_i = rho_i * np.dot(y.T, r)
            r += s * (alphas[len(self.coord_diffs) - 1 - i] - beta_i)
        
        return r
    
    def update_mu_reg(self, mu, energy, trial_energy, forces, step):
        """Update regularization parameter mu based on actual vs predicted energy change.
        
        Parameters
        ----------
        mu : float
            Current regularization parameter
        energy : float
            Current energy
        trial_energy : float
            Energy at proposed step
        forces : np.ndarray
            Current forces (negative gradient)
        step : np.ndarray
            Proposed step
            
        Returns
        -------
        float
            Updated regularization parameter
        bool
            Whether to recompute the step
        """
        # Compute actual and predicted reduction
        actual_reduction = energy - trial_energy
        predicted_reduction = -np.dot(forces.flatten(), step.flatten())
        
        # Compute ratio
        if abs(predicted_reduction) < 1e-8:
            ratio = 1.0 if actual_reduction > 0 else 0.0
        else:
            ratio = actual_reduction / predicted_reduction
            
        print(f"Actual reduction: {actual_reduction:.6f}")
        print(f"Predicted reduction: {predicted_reduction:.6f}")
        print(f"Ratio: {ratio:.6f}")
        
        # Update mu based on ratio
        if ratio < 0.25:
            mu = min(4.0 * mu, 1e6)
            return mu, True  # Recompute step
        elif ratio > 0.75:
            mu = max(0.5 * mu, 1e-6)
            return mu, False  # Accept step
        else:
            return mu, False  # Accept step with current mu
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        """Standard L-BFGS optimization step."""
        if self.linesearchflag:
            print("linesearch mode")
        else:
            print("normal mode")
            
        # First iteration is steepest descent
        if self.Initialization:
            self.Initialization = False
            return self.DELTA * B_g
        
        # Reshape inputs for consistency
        forces = B_g.reshape(-1, 1)  # Negative gradient
        prev_forces = pre_B_g.reshape(-1, 1) if len(pre_B_g) > 0 else None
        
        # Update L-BFGS memory if we have previous data
        if self.iter > 0 and prev_forces is not None and forces.size == prev_forces.size:
            s = (geom_num_list - pre_geom).reshape(-1, 1)
            y = (prev_forces - forces).reshape(-1, 1)
            
            # Apply double damping if enabled
            if self.double_damp:
                s, y = self.double_damp_vectors(s, y)
            
            # Only store vectors if y'*s > 0 (curvature condition)
            if np.dot(y.T, s) > 0:
                self.coord_diffs.append(s)
                self.grad_diffs.append(y)
                
                # Keep only the most recent vectors
                self.coord_diffs = self.coord_diffs[-self.keep_last:]
                self.grad_diffs = self.grad_diffs[-self.keep_last:]
            else:
                print("Skipping update: y'*s <= 0")
        
        # Calculate step using L-BFGS algorithm
        step = self.bfgs_multiply(forces)
        
        # Apply regularization if enabled
        if self.mu_reg and self.iter > 0:
            adapt_mu_cycles = 0
            while adapt_mu_cycles < self.max_mu_reg_adaptions:
                print(f"Adapt μ_reg={self.mu_reg:.6f}, norm(step)={np.linalg.norm(step):.6f}")
                
                # Calculate trial energy
                trial_coords = geom_num_list.reshape(-1, 1) + step
                trial_energy = B_e  # Placeholder - in real implementation, evaluate energy at trial_coords
                
                # Update regularization parameter
                new_mu, recompute = self.update_mu_reg(
                    self.mu_reg, B_e, trial_energy, forces, step
                )
                self.mu_reg = new_mu
                
                if not recompute:
                    print(f"Next μ_reg={self.mu_reg:.6f}")
                    break
                    
                # Recompute step with updated mu
                self.beta = self.mu_reg
                step = self.bfgs_multiply(forces)
                adapt_mu_cycles += 1
            
            self.tot_adapt_mu_cycles += adapt_mu_cycles + 1
        
        # Apply line search if enabled
        if self.iter > 0 and self.linesearchflag:
            if self.optimal_step_flag or self.iter == 1:
                self.LS = LineSearch(self.prev_move_vector, step, B_g, pre_B_g, B_e, pre_B_e, None)
            
            new_step, self.optimal_step_flag = self.LS.linesearch(
                self.prev_move_vector, step, B_g, pre_B_g, B_e, pre_B_e, None
            )
            step = new_step
            
            if self.optimal_step_flag or self.iter == 1:
                self.prev_move_vector = step
        
        # Control step size if enabled
        if self.control_step:
            step_norm = np.linalg.norm(step)
            if self.detect_energy_oscillation(B_e):
                new_max_step = max(self.max_step * self.energy_oscillation_factor, self.min_step)
                if new_max_step < self.max_step:
                    print(f"Reducing max step: {self.max_step:.6f} → {new_max_step:.6f}")
                    self.max_step = new_max_step
                
            
            if step_norm > self.max_step:
                print(f"Scaling step: {step_norm:.6f} -> {self.max_step:.6f}")
                step = step * self.max_step / step_norm
        
        print(f"Step size: {np.linalg.norm(step):.6f}")
        self.iter += 1
        
        return step
    
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """Main entry point for L-BFGS optimization in BiasPotPy.
        
        Parameters
        ----------
        geom_num_list : np.ndarray
            Current geometry
        B_g : np.ndarray
            Current gradient
        pre_B_g : np.ndarray
            Previous gradient
        pre_geom : np.ndarray
            Previous geometry
        B_e : float
            Current energy
        pre_B_e : float
            Previous energy
        pre_move_vector : np.ndarray
            Previous step vector
        initial_geom_num_list : np.ndarray
            Initial geometry
        g : np.ndarray
            Raw gradient
        pre_g : np.ndarray
            Previous raw gradient
            
        Returns
        -------
        np.ndarray
            Optimization step
        """
     
        move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
            
        return move_vector