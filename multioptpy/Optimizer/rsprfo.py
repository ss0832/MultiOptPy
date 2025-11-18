import numpy as np
from numpy.linalg import norm
import copy

from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Optimizer.block_hessian_update import BlockHessianUpdate
from multioptpy.Utils.calc_tools import Calculationtools


class EnhancedRSPRFO:
    def __init__(self, **config):
        """
        Enhanced Rational Step P-RFO (Rational Function Optimization) for transition state searches
        with dynamic trust radius adjustment based on trust region methodology
        
        References:
        [1] Banerjee et al., Phys. Chem., 89, 52-57 (1985)
        [2] Heyden et al., J. Chem. Phys., 123, 224101 (2005)
        [3] Baker, J. Comput. Chem., 7, 385-395 (1986)
        [4] Besalú and Bofill, Theor. Chem. Acc., 100, 265-274 (1998)
        [5] Jensen and Jørgensen, J. Chem. Phys., 80, 1204 (1984) [Eigenvector following]
        [6] Yuan, SIAM J. Optim. 11, 325-357 (2000) [Trust region methods]
        
        This code is made based on the below codes.
        1, https://github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/RSPRFOptimizer.py

        
        """
        # Standard RSPRFO parameters
        self.alpha0 = config.get("alpha0", 1.0)
        self.max_micro_cycles = config.get("max_micro_cycles", 20)  # Increased from 1 to 20
        self.saddle_order = config.get("saddle_order", 1)
        self.hessian_update_method = config.get("method", "auto")
        self.display_flag = config.get("display_flag", True)
        self.debug = config.get("debug", False)
        
        # Alpha constraints to prevent numerical instability
        self.alpha_max = config.get("alpha_max", 1e6)
        self.alpha_step_max = config.get("alpha_step_max", 10.0)
        
        # Trust region parameters
        if self.saddle_order == 0:
            self.trust_radius_initial = config.get("trust_radius", 0.5)
            self.trust_radius_max = config.get("trust_radius_max", 0.5)  # Upper bound (delta_hat)
        else:
            self.trust_radius_initial = config.get("trust_radius", 0.1)
            self.trust_radius_max = config.get("trust_radius_max", 0.1)  # Upper bound for TS search
            
        self.trust_radius = self.trust_radius_initial  # Current trust radius (delta_tr)
        self.trust_radius_min = config.get("trust_radius_min", 0.01)  # Lower bound (delta_min)
        
        # Trust region acceptance thresholds
        self.accept_poor_threshold = config.get("accept_poor_threshold", 0.25)  # Threshold for poor steps
        self.accept_good_threshold = config.get("accept_good_threshold", 0.75)  # Threshold for very good steps
        self.shrink_factor = config.get("shrink_factor", 0.50)  # Factor to shrink trust radius
        self.expand_factor = config.get("expand_factor", 2.00)   # Factor to expand trust radius
        self.rtol_boundary = config.get("rtol_boundary", 0.10)   # Relative tolerance for boundary detection
        
        # Whether to use trust radius adaptation
        self.adapt_trust_radius = config.get("adapt_trust_radius", True)
        
        # Rest of initialization
        self.config = config
        self.Initialization = True
        self.iter = 0
        
        # Hessian-related variables
        self.hessian = None
        self.bias_hessian = None
        
        # Optimization tracking variables
        self.prev_eigvec_max = None
        self.prev_eigvec_min = None
        self.predicted_energy_changes = []
        self.actual_energy_changes = []
        self.reduction_ratios = []
        self.trust_radius_history = []
        self.prev_geometry = None
        self.prev_gradient = None
        self.prev_energy = None
        self.prev_move_vector = None
        
        # Mode Following specific parameters
        self.mode_following_enabled = config.get("mode_following", True)
        self.eigvec_history = []  # History of eigenvectors for consistent tracking
        self.ts_mode_idx = None   # Current index of transition state direction
        
        # Eigenvector Following settings
        self.eigvec_following = config.get("eigvec_following", True)
        self.overlap_threshold = config.get("overlap_threshold", 0.5)
        self.mixing_threshold = config.get("mixing_threshold", 0.3)
        
        # Define modes based on saddle order
        self.roots = list(range(self.saddle_order))
            
        # Initialize the hessian update module
        self.hessian_updater = ModelHessianUpdate()
        self.block_hessian_updater = BlockHessianUpdate()
        
        # Build Hessian updater dispatch list
        self._build_hessian_updater_list()
        
        self.log(f"Initialized EnhancedRSPRFO with trust radius={self.trust_radius:.6f}, "
                f"bounds=[{self.trust_radius_min:.6f}, {self.trust_radius_max:.6f}]")

    def _build_hessian_updater_list(self):
        """Builds the prioritized dispatch list for Hessian updaters (from RSIRFO)."""
        self.default_update_method = (
            "auto (default)",
            lambda h, d, g: self.hessian_updater.flowchart_hessian_update(h, d, g, "auto")
        )
        self.updater_dispatch_list = [
            ("flowchart", "flowchart", lambda h, d, g: self.hessian_updater.flowchart_hessian_update(h, d, g, "auto")),
            ("block_cfd_fsb_dd", "block_cfd_fsb_dd", self.block_hessian_updater.block_CFD_FSB_hessian_update_dd),
            ("block_cfd_fsb_weighted", "block_cfd_fsb_weighted", self.block_hessian_updater.block_CFD_FSB_hessian_update_weighted),
            ("block_cfd_fsb", "block_cfd_fsb", self.block_hessian_updater.block_CFD_FSB_hessian_update),
            ("block_cfd_bofill_weighted", "block_cfd_bofill_weighted", self.block_hessian_updater.block_CFD_Bofill_hessian_update_weighted),
            ("block_cfd_bofill", "block_cfd_bofill", self.block_hessian_updater.block_CFD_Bofill_hessian_update),
            ("block_bfgs_dd", "block_bfgs_dd", self.block_hessian_updater.block_BFGS_hessian_update_dd),
            ("block_bfgs", "block_bfgs", self.block_hessian_updater.block_BFGS_hessian_update),
            ("block_fsb_dd", "block_fsb_dd", self.block_hessian_updater.block_FSB_hessian_update_dd),
            ("block_fsb_weighted", "block_fsb_weighted", self.block_hessian_updater.block_FSB_hessian_update_weighted),
            ("block_fsb", "block_fsb", self.block_hessian_updater.block_FSB_hessian_update),
            ("block_bofill_weighted", "block_bofill_weighted", self.block_hessian_updater.block_Bofill_hessian_update_weighted),
            ("block_bofill", "block_bofill", self.block_hessian_updater.block_Bofill_hessian_update),
            ("bfgs_dd", "bfgs_dd", self.hessian_updater.BFGS_hessian_update_dd),
            ("bfgs", "bfgs", self.hessian_updater.BFGS_hessian_update),
            ("sr1", "sr1", self.hessian_updater.SR1_hessian_update),
            ("pcfd_bofill", "pcfd_bofill", self.hessian_updater.pCFD_Bofill_hessian_update),
            ("cfd_fsb_dd", "cfd_fsb_dd", self.hessian_updater.CFD_FSB_hessian_update_dd),
            ("cfd_fsb", "cfd_fsb", self.hessian_updater.CFD_FSB_hessian_update),
            ("cfd_bofill", "cfd_bofill", self.hessian_updater.CFD_Bofill_hessian_update),
            ("fsb_dd", "fsb_dd", self.hessian_updater.FSB_hessian_update_dd),
            ("fsb", "fsb", self.hessian_updater.FSB_hessian_update),
            ("bofill", "bofill", self.hessian_updater.Bofill_hessian_update),
            ("psb", "psb", self.hessian_updater.PSB_hessian_update),
            ("msp", "msp", self.hessian_updater.MSP_hessian_update),
        ]

    def compute_reduction_ratio(self, gradient, hessian, step, actual_reduction):
        """
        Compute ratio between actual and predicted reduction in energy
        
        Parameters:
        gradient: numpy.ndarray - Current gradient
        hessian: numpy.ndarray - Current approximate Hessian
        step: numpy.ndarray - Step vector
        actual_reduction: float - Actual energy reduction (previous_energy - current_energy)
        
        Returns:
        float: Ratio of actual to predicted reduction
        """
        # Calculate predicted reduction from quadratic model
        g_flat = gradient.flatten()
        step_flat = step.flatten()
        
        # Linear term of the model: g^T * p
        linear_term = np.dot(g_flat, step_flat)
        
        # Quadratic term of the model: 0.5 * p^T * H * p
        quadratic_term = 0.5 * np.dot(step_flat, np.dot(hessian, step_flat))
        
        # Predicted reduction: -g^T * p - 0.5 * p^T * H * p
        # Negative sign because we're predicting the reduction (energy decrease)
        predicted_reduction = -(linear_term + quadratic_term)
        
        # Avoid division by zero or very small numbers
        if abs(predicted_reduction) < 1e-10:
            self.log("Warning: Predicted reduction is near zero")
            return 0.0
            
        # Calculate ratio
        ratio = actual_reduction / predicted_reduction
        
        # Safeguard against numerical issues
        if not np.isfinite(ratio):
            self.log("Warning: Non-finite reduction ratio, using 0.0")
            return 0.0
            
        self.log(f"Actual reduction: {actual_reduction:.6e}, "
                f"Predicted reduction: {predicted_reduction:.6e}, "
                f"Ratio: {ratio:.4f}")
        
        return ratio
        
    def adjust_trust_radius(self, actual_energy_change, predicted_energy_change, step_norm):
        """
        Dynamically adjust the trust radius based on ratio between actual and predicted reductions
        using the trust region methodology
        """
        if not self.adapt_trust_radius or actual_energy_change is None or predicted_energy_change is None:
            return
            
        # Avoid division by zero or very small numbers
        if abs(predicted_energy_change) < 1e-10:
            self.log("Skipping trust radius update due to negligible predicted energy change")
            return
            
        # Calculate the ratio between actual and predicted energy changes
        # Use absolute values to focus on magnitude of agreement
        ratio = abs(actual_energy_change / predicted_energy_change)
        self.log(f"Raw reduction ratio: {actual_energy_change / predicted_energy_change:.4f}")
        self.log(f"Absolute reduction ratio: {ratio:.4f}")
        self.reduction_ratios.append(ratio)
        
        old_trust_radius = self.trust_radius
        
        # Improved boundary detection - check if step is close to current trust radius
        at_boundary = step_norm >= old_trust_radius * 0.95  # Within 5% of trust radius
        self.log(f"Step norm: {step_norm:.6f}, Trust radius: {old_trust_radius:.6f}, At boundary: {at_boundary}")
        
        # Better logic for trust radius adjustment
        if ratio < 0.25 or ratio > 4.0:  # Predicted energy change is very different from actual
            # Poor prediction - decrease the trust radius
            self.trust_radius = max(self.shrink_factor * self.trust_radius, self.trust_radius_min)
            if self.trust_radius != old_trust_radius:
                self.log(f"Poor step quality (ratio={ratio:.3f}), shrinking trust radius to {self.trust_radius:.6f}")
        elif (0.8 <= ratio <= 1.25) and at_boundary:
            # Very good prediction and step at trust radius boundary - increase the trust radius
            self.trust_radius = min(self.expand_factor * self.trust_radius, self.trust_radius_max)
            if self.trust_radius != old_trust_radius:
                self.log(f"Good step quality (ratio={ratio:.3f}) at boundary, expanding trust radius to {self.trust_radius:.6f}")
        else:
            # Acceptable prediction or step not at boundary - keep the same trust radius
            self.log(f"Acceptable step quality (ratio={ratio:.3f}), keeping trust radius at {self.trust_radius:.6f}")       
            
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """
        Execute one step of enhanced RSPRFO optimization with trust radius adjustment
        
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
        self.log(f"\n{'='*50}\nIteration {self.iter}\n{'='*50}")
        
        if self.Initialization:
            self.prev_eigvec_max = None
            self.prev_eigvec_min = None
            self.predicted_energy_changes = []
            self.actual_energy_changes = []
            self.reduction_ratios = []
            self.trust_radius_history = []
            self.prev_geometry = None
            self.prev_gradient = None
            self.prev_energy = None
            self.prev_move_vector = None
            self.eigvec_history = []
            self.ts_mode_idx = None
            self.Initialization = False
            self.log(f"First iteration - using initial trust radius {self.trust_radius:.6f}")
        else:
            # Adjust trust radius based on the previous step if we have energy data
            if self.prev_energy is not None and len(self.predicted_energy_changes) > 0:
                actual_energy_change = B_e - self.prev_energy
                predicted_energy_change = self.predicted_energy_changes[-1]
                self.actual_energy_changes.append(actual_energy_change)
                
                # Get the previous step length
                if len(pre_move_vector) > 0:
                    prev_step_norm = norm(pre_move_vector.flatten())
                elif self.prev_move_vector is not None:
                    prev_step_norm = norm(self.prev_move_vector.flatten())
                else:
                    prev_step_norm = 0.0
                
                # Log energy comparison
                self.log(f"Previous energy: {self.prev_energy:.6f}, Current energy: {B_e:.6f}")
                self.log(f"Actual energy change: {actual_energy_change:.6f}")
                self.log(f"Predicted energy change: {predicted_energy_change:.6f}")
                self.log(f"Previous step norm: {prev_step_norm:.6f}")
                
                # Complete Hessian for the reduction ratio calculation
                H = self.hessian + self.bias_hessian if self.bias_hessian is not None else self.hessian
                H = Calculationtools().project_out_hess_tr_and_rot_for_coord(H, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), display_eigval=False)
                # Compute reduction ratio
                reduction_ratio = self.compute_reduction_ratio(
                    self.prev_gradient, H, self.prev_move_vector, actual_energy_change)
                
                # Adjust trust radius based on step quality and length
                self.adjust_trust_radius(actual_energy_change, predicted_energy_change, prev_step_norm)
            
        # Check Hessian
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        # Update Hessian if we have previous geometry and gradient information
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_B_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, B_g, pre_geom, pre_B_g)
            
        # Ensure gradient is properly shaped as a 1D array
        gradient = np.asarray(B_g).flatten()
        H = self.hessian + self.bias_hessian if self.bias_hessian is not None else self.hessian
            
        # Compute eigenvalues and eigenvectors of the hessian
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Count negative eigenvalues for diagnostic purposes
        neg_eigval_count = np.sum(eigvals < -1e-6)
        self.log(f"Found {neg_eigval_count} negative eigenvalues, target for this saddle order: {self.saddle_order}")
        
        # Store previous eigenvector information
        prev_eigvecs = None
        if len(self.eigvec_history) > 0:
            prev_eigvecs = self.eigvec_history[-1]
        
        # Standard mode selection (with mode following if enabled)
        if self.mode_following_enabled and self.saddle_order > 0:
            if self.ts_mode_idx is None:
                # For first run, select mode with most negative eigenvalue
                self.ts_mode_idx = np.argmin(eigvals)
                self.log(f"Initial TS mode selected: {self.ts_mode_idx} with eigenvalue {eigvals[self.ts_mode_idx]:.6f}")
                
            # Find corresponding modes between steps
            mode_indices = self.find_corresponding_mode(eigvals, eigvecs, prev_eigvecs, self.ts_mode_idx)
            
            # Apply Eigenvector Following for cases with mode mixing
            if self.eigvec_following and len(mode_indices) > 1:
                mode_indices = self.apply_eigenvector_following(eigvals, eigvecs, gradient.dot(eigvecs), mode_indices)
                
            # Update tracked mode
            if mode_indices:
                self.ts_mode_idx = mode_indices[0]
                self.log(f"Mode following: tracking mode {self.ts_mode_idx} with eigenvalue {eigvals[self.ts_mode_idx]:.6f}")
                
                # Update max_indices (saddle point direction)
                max_indices = mode_indices
            else:
                # If no corresponding mode found, use standard approach
                self.log("No corresponding mode found, using default mode selection")
                max_indices = self.roots
        else:
            # Standard mode selection when mode following is disabled
            if self.saddle_order == 0:
                min_indices = list(range(len(gradient)))
                max_indices = []
            else:
                min_indices = [i for i in range(gradient.size) if i not in self.roots]
                max_indices = self.roots
                
        # Store eigenvectors in history
        self.eigvec_history.append(eigvecs)
        if len(self.eigvec_history) > 5:  # Keep only last 5 steps
            self.eigvec_history.pop(0)
            
        # Transform gradient to eigenvector space
        gradient_trans = eigvecs.T.dot(gradient).flatten()
        
        # Set minimization directions (all directions not in max_indices)
        min_indices = [i for i in range(gradient.size) if i not in max_indices]
            
        # Initialize alpha parameter
        alpha = self.alpha0
        
        # Tracking variables
        best_step = None
        best_step_norm_diff = float('inf')
        step_norm_history = []
        
        # NEW IMPLEMENTATION: Micro-cycle loop with improved alpha calculation
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-PRFO micro cycle {mu:02d}, alpha={alpha:.6f}, trust radius={self.trust_radius:.6f}")
            
            try:
                # Make a fresh step vector for this cycle - essential to ensure proper recalculation
                step = np.zeros_like(gradient_trans)
                
                # Maximization subspace calculation
                step_max = np.array([])
                eigval_max = 0
                if len(max_indices) > 0:
                    # Calculate augmented Hessian
                    H_aug_max = self.get_augmented_hessian(
                        eigvals[max_indices], gradient_trans[max_indices], alpha
                    )
                    
                    # Solve RFO equations
                    step_max, eigval_max, nu_max, eigvec_max = self.solve_rfo(
                        H_aug_max, "max", prev_eigvec=self.prev_eigvec_max
                    )
                    
                    # Store eigenvector for next iteration
                    self.prev_eigvec_max = eigvec_max
                    
                    # Copy step to the main step vector
                    step[max_indices] = step_max
                
                # Minimization subspace calculation
                step_min = np.array([])
                eigval_min = 0
                if len(min_indices) > 0:
                    # Calculate augmented Hessian
                    H_aug_min = self.get_augmented_hessian(
                        eigvals[min_indices], gradient_trans[min_indices], alpha
                    )
                    
                    # Solve RFO equations
                    step_min, eigval_min, nu_min, eigvec_min = self.solve_rfo(
                        H_aug_min, "min", prev_eigvec=self.prev_eigvec_min
                    )
                    
                    # Store eigenvector for next iteration
                    self.prev_eigvec_min = eigvec_min
                    
                    # Copy step to the main step vector
                    step[min_indices] = step_min
                
                # Calculate norms of the current step
                step_max_norm = np.linalg.norm(step_max) if len(max_indices) > 0 else 0.0
                step_min_norm = np.linalg.norm(step_min) if len(min_indices) > 0 else 0.0
                step_norm = np.linalg.norm(step)
                
                # Log the current norms
                if len(max_indices) > 0:
                    self.log(f"norm(step_max)={step_max_norm:.6f}")
                if len(min_indices) > 0:
                    self.log(f"norm(step_min)={step_min_norm:.6f}")
                
                self.log(f"norm(step)={step_norm:.6f}")
                
                # Keep track of step norm history for convergence detection
                step_norm_history.append(step_norm)
                
                # Save this step if it's closest to trust radius (for later use)
                norm_diff = abs(step_norm - self.trust_radius)
                if norm_diff < best_step_norm_diff:
                    best_step = step.copy()
                    best_step_norm_diff = norm_diff
                
                # Check if step is already within trust radius
                if step_norm <= self.trust_radius:
                    self.log(f"Step satisfies trust radius {self.trust_radius:.6f}")
                    break
                
                # Calculate alpha update for each subspace
                # Max subspace
                alpha_step_max = 0.0
                if len(max_indices) > 0:
                    alpha_step_max = self.get_alpha_step(
                        alpha, eigval_max, step_max_norm, eigvals[max_indices], 
                        gradient_trans[max_indices], "max"
                    )
                
                # Min subspace
                alpha_step_min = 0.0
                if len(min_indices) > 0:
                    alpha_step_min = self.get_alpha_step(
                        alpha, eigval_min, step_min_norm, eigvals[min_indices], 
                        gradient_trans[min_indices], "min"
                    )
                
                # Combine alpha steps with appropriate weighting
                alpha_step = 0.0
                if alpha_step_max != 0.0 and alpha_step_min != 0.0:
                    # Weight by squared norms
                    w_max = step_max_norm**2 if step_max_norm > 0.0 else 0.0
                    w_min = step_min_norm**2 if step_min_norm > 0.0 else 0.0
                    if w_max + w_min > 0.0:
                        alpha_step = (w_max * alpha_step_max + w_min * alpha_step_min) / (w_max + w_min)
                    else:
                        alpha_step = alpha_step_max if abs(alpha_step_max) > abs(alpha_step_min) else alpha_step_min
                else:
                    alpha_step = alpha_step_max if alpha_step_max != 0.0 else alpha_step_min
                
                # If alpha_step is still 0, use a direct calculation with the total step
                if abs(alpha_step) < 1e-10 and step_norm > 0.0:
                    try:
                        # Calculate derivative directly using analytic formula
                        dstep2_dalpha = self.calculate_step_derivative(
                            alpha, eigval_max, eigval_min, eigvals, 
                            max_indices, min_indices, gradient_trans, step_norm
                        )
                        
                        if abs(dstep2_dalpha) > 1e-10:
                            alpha_step = 2.0 * (self.trust_radius * step_norm - step_norm**2) / dstep2_dalpha
                            self.log(f"Direct alpha_step calculation: {alpha_step:.6f}")
                    except Exception as e:
                        self.log(f"Error in direct derivative calculation: {str(e)}")
                        alpha_step = 0.0
                
                # Update alpha with proper bounds
                old_alpha = alpha
                
                # If derivative-based approach fails, use heuristic
                if abs(alpha_step) < 1e-10:
                    # Apply a more aggressive heuristic - double alpha
                    alpha = min(alpha * 2.0, self.alpha_max)
                    self.log(f"Using heuristic alpha update: {old_alpha:.6f} -> {alpha:.6f}")
                else:
                    # Apply safety bounds to alpha_step
                    alpha_step_limited = np.clip(alpha_step, -self.alpha_step_max, self.alpha_step_max)
                    
                    if abs(alpha_step_limited) != abs(alpha_step):
                        self.log(f"Limited alpha_step from {alpha_step:.6f} to {alpha_step_limited:.6f}")
                    
                    # Ensure alpha remains positive and within bounds
                    alpha = min(max(old_alpha + alpha_step_limited, 1e-6), self.alpha_max)
                    self.log(f"Updated alpha: {old_alpha:.6f} -> {alpha:.6f}")
                
                # Check if alpha reached its maximum value
                if alpha == self.alpha_max:
                    self.log(f"Alpha reached maximum value ({self.alpha_max}), using best step found")
                    if best_step is not None:
                        step = best_step.copy()
                    break
                
                # Check for progress in step norm adjustments
                if len(step_norm_history) >= 3:
                    # Calculate consecutive changes in step norm
                    recent_changes = [abs(step_norm_history[-i] - step_norm_history[-(i+1)]) 
                                      for i in range(1, min(3, len(step_norm_history)))]
                    
                    # If step norms are not changing significantly, break the loop
                    if all(change < 1e-6 for change in recent_changes):
                        self.log(f"Step norms not changing significantly: {step_norm_history[-3:]}")
                        self.log("Breaking micro-cycle loop")
                        
                        # Use the best step found so far
                        if best_step is not None and best_step_norm_diff < norm_diff:
                            step = best_step.copy()
                            self.log("Using best step found so far")
                        
                        break
                
            except Exception as e:
                self.log(f"Error in micro-cycle: {str(e)}")
                # Use best step if available, otherwise scale current step
                if best_step is not None:
                    self.log("Using best step due to error")
                    step = best_step.copy()
                else:
                    # Simple scaling fallback
                    if step_norm > 0 and step_norm > self.trust_radius:
                        scale_factor = self.trust_radius / step_norm
                        step = step * scale_factor
                        self.log(f"Scaled step to trust radius due to error")
                break
        
        else:
            # If micro-cycles did not converge
            self.log(f"Micro-cycles did not converge in {self.max_micro_cycles} iterations")
            # Use the best step if available
            if best_step is not None and best_step_norm_diff < abs(step_norm - self.trust_radius):
                self.log("Using best step found during micro-cycles")
                step = best_step.copy()
        
        # Transform step back to original coordinates
        move_vector = eigvecs.dot(step)
        step_norm = norm(move_vector)
        
        # Only scale down steps that exceed the trust radius
        if step_norm > self.trust_radius:
            self.log(f"Step norm {step_norm:.6f} exceeds trust radius {self.trust_radius:.6f}, scaling down")
            move_vector = move_vector * (self.trust_radius / step_norm)
            step_norm = self.trust_radius
        else:
            self.log(f"Step norm {step_norm:.6f} is within trust radius {self.trust_radius:.6f}, no scaling needed")
        
        self.log(f"Final norm(step)={norm(move_vector):.6f}")
        
        # Apply maxstep constraint if specified in config
        if self.config.get("maxstep") is not None:
            maxstep = self.config.get("maxstep")
            
            # Calculate step lengths
            if move_vector.size % 3 == 0 and move_vector.size > 3:  # Likely atomic coordinates in 3D
                move_vector_reshaped = move_vector.reshape(-1, 3)
                steplengths = np.sqrt((move_vector_reshaped**2).sum(axis=1))
                longest_step = np.max(steplengths)
            else:
                # Generic vector - just compute total norm
                longest_step = norm(move_vector)
            
            # Scale step if necessary
            if longest_step > maxstep:
                move_vector = move_vector * (maxstep / longest_step)
                self.log(f"Step constrained by maxstep={maxstep:.6f}")
        
        # Calculate predicted energy change
        predicted_energy_change = self.rfo_model(gradient, H, move_vector)
        self.predicted_energy_changes.append(predicted_energy_change)
        self.log(f"Predicted energy change: {predicted_energy_change:.6f}")
        
        # Store current geometry, gradient, energy, and move vector for next iteration
        self.prev_geometry = copy.deepcopy(geom_num_list)
        self.prev_gradient = copy.deepcopy(B_g)
        self.prev_energy = B_e
        self.prev_move_vector = copy.deepcopy(move_vector)
        
        # Increment iteration counter
        self.iter += 1
        
        return move_vector.reshape(-1, 1)

    def get_alpha_step(self, alpha, rfo_eigval, step_norm, eigvals, gradient, mode="min"):
        """
        Calculate alpha step update for a specific subspace using the improved method
        
        Parameters:
        alpha: float - Current alpha value
        rfo_eigval: float - RFO eigenvalue for this subspace
        step_norm: float - Norm of the step in this subspace
        eigvals: numpy.ndarray - Eigenvalues for this subspace
        gradient: numpy.ndarray - Gradient components in this subspace
        mode: str - "min" or "max" for minimization or maximization subspace
        
        Returns:
        float: Calculated alpha step update
        """
        try:
            # Calculate denominators with safety checks
            denominators = eigvals - rfo_eigval * alpha
            
            # Handle small denominators
            small_denoms = np.abs(denominators) < 1e-10
            if np.any(small_denoms):
                self.log(f"Small denominators detected in {mode} subspace: {np.sum(small_denoms)}")
                safe_denoms = denominators.copy()
                for i in np.where(small_denoms)[0]:
                    safe_denoms[i] = 1e-10 * np.sign(safe_denoms[i]) if safe_denoms[i] != 0 else 1e-10
                denominators = safe_denoms
            
            # Calculate quotient term
            numerator = gradient**2
            denominator = denominators**3
            quot = np.sum(numerator / denominator)
            self.log(f"{mode} subspace quot={quot:.6e}")
            
            # Calculate step term with safety
            step_term = 1.0 + step_norm**2 * alpha
            if abs(step_term) < 1e-10:
                step_term = 1e-10 * np.sign(step_term) if step_term != 0 else 1e-10
            
            # Calculate derivative of squared step norm with respect to alpha
            dstep2_dalpha = 2.0 * rfo_eigval / step_term * quot
            self.log(f"{mode} subspace d(step^2)/dα={dstep2_dalpha:.6e}")
            
            # Return 0 if derivative is too small
            if abs(dstep2_dalpha) < 1e-10:
                return 0.0
            
            # Calculate alpha step using the trust radius formula
            alpha_step = 2.0 * (self.trust_radius * step_norm - step_norm**2) / dstep2_dalpha
            self.log(f"{mode} subspace alpha_step={alpha_step:.6f}")
            
            return alpha_step
            
        except Exception as e:
            self.log(f"Error in get_alpha_step ({mode}): {str(e)}")
            return 0.0
    
    def calculate_step_derivative(self, alpha, eigval_max, eigval_min, eigvals, max_indices, min_indices, gradient_trans, step_norm):
        """
        Calculate the derivative of the squared step norm with respect to alpha
        for the combined step from both subspaces
        
        Parameters:
        alpha: float - Current alpha value
        eigval_max, eigval_min: float - RFO eigenvalues from max and min subspaces
        eigvals: numpy.ndarray - All eigenvalues
        max_indices, min_indices: list - Indices of max and min subspaces
        gradient_trans: numpy.ndarray - Transformed gradient
        step_norm: float - Current total step norm
        
        Returns:
        float: Combined derivative of squared step norm with respect to alpha
        """
        try:
            dstep2_dalpha_max = 0.0
            if len(max_indices) > 0:
                # Calculate denominator for max subspace
                denom_max = 1.0 + np.dot(gradient_trans[max_indices], gradient_trans[max_indices]) * alpha
                if abs(denom_max) < 1e-10:
                    denom_max = 1e-10 * np.sign(denom_max) if denom_max != 0 else 1e-10
                
                # Handle small denominators in eigenvalue terms
                eigvals_max = eigvals[max_indices].copy()
                denom_terms_max = eigvals_max - eigval_max * alpha
                
                small_denoms = np.abs(denom_terms_max) < 1e-10
                if np.any(small_denoms):
                    for i in np.where(small_denoms)[0]:
                        denom_terms_max[i] = 1e-10 * np.sign(denom_terms_max[i]) if denom_terms_max[i] != 0 else 1e-10
                
                # Calculate derivative component for max subspace
                dstep2_dalpha_max = (
                    2.0 * eigval_max / denom_max * np.sum(gradient_trans[max_indices]**2 / denom_terms_max**3)
                )
            
            dstep2_dalpha_min = 0.0
            if len(min_indices) > 0:
                # Calculate denominator for min subspace
                denom_min = 1.0 + np.dot(gradient_trans[min_indices], gradient_trans[min_indices]) * alpha
                if abs(denom_min) < 1e-10:
                    denom_min = 1e-10 * np.sign(denom_min) if denom_min != 0 else 1e-10
                
                # Handle small denominators in eigenvalue terms
                eigvals_min = eigvals[min_indices].copy()
                denom_terms_min = eigvals_min - eigval_min * alpha
                
                small_denoms = np.abs(denom_terms_min) < 1e-10
                if np.any(small_denoms):
                    for i in np.where(small_denoms)[0]:
                        denom_terms_min[i] = 1e-10 * np.sign(denom_terms_min[i]) if denom_terms_min[i] != 0 else 1e-10
                
                # Calculate derivative component for min subspace
                dstep2_dalpha_min = (
                    2.0 * eigval_min / denom_min * np.sum(gradient_trans[min_indices]**2 / denom_terms_min**3)
                )
            
            # Combine derivatives from both subspaces
            dstep2_dalpha = dstep2_dalpha_max + dstep2_dalpha_min
            self.log(f"Combined dstep2_dalpha={dstep2_dalpha:.6e}")
            
            return dstep2_dalpha
            
        except Exception as e:
            self.log(f"Error in calculate_step_derivative: {str(e)}")
            return 0.0

    def find_corresponding_mode(self, eigvals, eigvecs, prev_eigvecs, target_mode_idx):
        """
        Find corresponding mode in current step based on eigenvector overlap
        
        Parameters:
        eigvals: numpy.ndarray - Current eigenvalues
        eigvecs: numpy.ndarray - Current eigenvectors as column vectors
        prev_eigvecs: numpy.ndarray - Previous eigenvectors
        target_mode_idx: int - Index of target mode from previous step
        
        Returns:
        list - List of indices of corresponding modes in current step
        """
        if prev_eigvecs is None or target_mode_idx is None:
            # For first step or reset, simply select by eigenvalue
            if self.saddle_order > 0:
                # For TS search, choose modes with most negative eigenvalues
                sorted_idx = np.argsort(eigvals)
                return sorted_idx[:self.saddle_order].tolist()
            else:
                # For minimization, no special mode
                return []
                
        # Calculate overlap between target mode from previous step and all current modes
        target_vec = prev_eigvecs[:, target_mode_idx].reshape(-1, 1)
        overlaps = np.abs(np.dot(eigvecs.T, target_vec)).flatten()
        
        # Sort by overlap magnitude (descending)
        sorted_idx = np.argsort(-overlaps)
        
        if self.display_flag:
            self.log(f"Mode overlaps with previous TS mode: {overlaps[sorted_idx[0]]:.4f}, {overlaps[sorted_idx[1]]:.4f}, {overlaps[sorted_idx[2]]:.4f}")
        
        # Return mode with overlap above threshold
        if overlaps[sorted_idx[0]] > self.overlap_threshold:
            return [sorted_idx[0]]
        
        # Consider mode mixing if no single mode has sufficient overlap
        mixed_modes = []
        cumulative_overlap = 0.0
        
        for idx in sorted_idx:
            mixed_modes.append(idx)
            cumulative_overlap += overlaps[idx]**2  # Sum of squares
            
            if cumulative_overlap > 0.8:  # 80% coverage
                break
                
        return mixed_modes
    
    def apply_eigenvector_following(self, eigvals, eigvecs, gradient_trans, mode_indices):
        """
        Apply Eigenvector Following method to handle mixed modes
        
        Parameters:
        eigvals: numpy.ndarray - Current eigenvalues
        eigvecs: numpy.ndarray - Current eigenvectors
        gradient_trans: numpy.ndarray - Gradient in eigenvector basis
        mode_indices: list - Indices of candidate modes
        
        Returns:
        list - Selected mode indices after eigenvector following
        """
        if not mode_indices or len(mode_indices) <= 1:
            # No mode mixing, apply standard RSPRFO processing
            return mode_indices
            
        # For mixed modes, build a weighted mode
        weights = np.zeros(len(eigvals))
        total_weight = 0.0
        
        for idx in mode_indices:
            # Use inverse of eigenvalue as weight (keep negative values as is)
            if eigvals[idx] < 0:
                weights[idx] = abs(1.0 / eigvals[idx])
            else:
                # Small weight for positive eigenvalues
                weights[idx] = 0.01
                
            total_weight += weights[idx]
            
        # Normalize weights
        if total_weight > 0:
            weights /= total_weight
            
        # Calculate centroid of mixed modes
        mixed_mode_idx = np.argmax(weights)
        
        self.log(f"Eigenvector following: selected mixed mode {mixed_mode_idx} from candidates {mode_indices}")
        self.log(f"Selected mode eigenvalue: {eigvals[mixed_mode_idx]:.6f}")
        
        return [mixed_mode_idx]
    
    def get_augmented_hessian(self, eigenvalues, gradient_components, alpha):
        """
        Create the augmented hessian matrix for RFO calculation
        
        Parameters:
        eigenvalues: numpy.ndarray - Eigenvalues for the selected subspace
        gradient_components: numpy.ndarray - Gradient components in the selected subspace
        alpha: float - Alpha parameter for RS-RFO
        
        Returns:
        numpy.ndarray - Augmented Hessian matrix for RFO calculation
        """
        n = len(eigenvalues)
        H_aug = np.zeros((n + 1, n + 1))
        
        # Fill the upper-left block with eigenvalues / alpha
        np.fill_diagonal(H_aug[:n, :n], eigenvalues / alpha)
        
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
        eigvals, eigvecs = np.linalg.eigh(H_aug)
        
        if mode == "min":
            idx = np.argmin(eigvals)
        else:  # mode == "max"
            idx = np.argmax(eigvals)
            
        # Check if we need to flip the eigenvector to maintain consistency
        if prev_eigvec is not None:
            try:
                overlap = np.dot(eigvecs[:, idx], prev_eigvec)
                if overlap < 0:
                    eigvecs[:, idx] *= -1
            except Exception as e:
                # Handle dimension mismatch or other errors
                self.log(f"Error in eigenvector consistency check: {str(e)}")
                # Continue without flipping
                
        eigval = eigvals[idx]
        eigvec = eigvecs[:, idx]
        
        # The last component is nu
        nu = eigvec[-1]
        
        # Add safeguard against very small nu values
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
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """
        Update the Hessian using the specified update method.
        WARNING: This version FORCES the update even if dot_product <= 0,
        which may lead to numerical instability or crashes.
        """
        displacement = np.asarray(current_geom - previous_geom).reshape(-1, 1)
        delta_grad = np.asarray(current_grad - previous_grad).reshape(-1, 1)
        
        disp_norm = np.linalg.norm(displacement)
        grad_diff_norm = np.linalg.norm(delta_grad)
        
        # This is a pre-check from the original code, kept for safety
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update due to small changes")
            return
            
        dot_product = np.dot(displacement.T, delta_grad)[0, 0]
        
        # === [IMPROVEMENT 3] Selective Hessian update ===
        # Uncomment the following lines if should_update_hessian method is implemented
        # if not self.should_update_hessian(displacement, delta_grad, dot_product):
        #     return
        # === [END IMPROVEMENT 3] ===
        
        # === [MODIFICATION] Safety check removed per user request ===
        if dot_product <= 0:
            self.log(f"WARNING: Forcing Hessian update despite poor alignment (dot_product={dot_product:.6f}).", force=True)
            self.log("This may cause instability or errors in the update function.", force=True)
        # =======================================================
        else:
            self.log(f"Hessian update: displacement norm={disp_norm:.6f}, gradient diff norm={grad_diff_norm:.6f}, dot product={dot_product:.6f}")
        
        method_key_lower = self.hessian_update_method.lower()
        method_name, update_function = self.default_update_method
        found_method = False

        for key, name, func in self.updater_dispatch_list:
            if key in method_key_lower:
                method_name = name
                update_function = func
                found_method = True
                break

        if not found_method:
             self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
        
        self.log(f"Hessian update method: {method_name}")
        
        try:
            delta_hess = update_function(
                self.hessian, displacement, delta_grad
            )
            self.hessian += delta_hess
            self.hessian = 0.5 * (self.hessian + self.hessian.T)
            self.log("Hessian update attempted.")
            
        except Exception as e:
            self.log(f"ERROR during forced Hessian update ({method_name}): {e}", force=True)
            self.log("Hessian may be corrupted. Proceeding with caution.", force=True)
    
    def log(self, message, force=False):
        """
        Print log message if display flag is enabled or force is True
        
        Parameters:
        message: str - Message to display
        force: bool - If True, display message regardless of display_flag
        """
        if self.display_flag or force:
            print(message)
    
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
        
    def set_trust_radius(self, radius):
        """
        Manually set the trust radius
        
        Parameters:
        radius: float - New trust radius value
        """
        old_value = self.trust_radius
        self.trust_radius = max(min(radius, self.trust_radius_max), self.trust_radius_min)
        self.log(f"Trust radius manually set from {old_value:.6f} to {self.trust_radius:.6f}")
        
    def get_reduction_ratios(self):
        """
        Get the history of reduction ratios
        
        Returns:
        list - Reduction ratios for each iteration
        """
        return self.reduction_ratios
        
    def get_trust_radius_history(self):
        """
        Get the history of trust radius values
        
        Returns:
        list - Trust radius values for each iteration
        """
        return self.trust_radius_history
