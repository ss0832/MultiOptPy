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
        
        This code is made based on the below codes.
        1, https://github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/TSHessianOptimizer.py
        2, https://github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/RSIRFOptimizer.py

        """
        # Configuration parameters
        self.alpha0 = config.get("alpha0", 1.0)
        self.max_micro_cycles = config.get("max_micro_cycles", 40)
        self.saddle_order = config.get("saddle_order", 1)
        self.hessian_update_method = config.get("method", "auto")
        self.small_eigval_thresh = config.get("small_eigval_thresh", 1e-6)
        
        self.alpha_max = config.get("alpha_max", 1000.0)
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
        
        # Adaptive Trust Radius Management Settings
        self.use_adaptive_trust_radius = config.get("use_adaptive_trust_radius", True)
        
        # === [NEW] Threshold to activate adaptive radius ===
        # Only use adaptive trust radius if grad_norm is *below* this value
        self.adaptive_trust_gradient_norm_threshold = config.get(
            "adaptive_trust_gradient_norm_threshold", 
            1e-2  # Default: activate when norm is 0.01 (adjust as needed)
        ) 
        # === [END NEW] ===
        
        self.max_curvature_factor = config.get("max_curvature_factor", 2.5)
        self.negative_curvature_safety = config.get("negative_curvature_safety", 0.8)
        self.min_eigenvalue_history = []
        
        
        # Enable/disable level-shifting manually
        # Default is False for conservative approach
        self.use_level_shift = config.get("use_level_shift", False)
        
        # Magnitude of the level shift
        # Should be much smaller than typical eigenvalue magnitudes
        self.level_shift_value = config.get("level_shift_value", 1e-5)
        
        # Automatic level-shifting based on condition number
        # Enabled by default for adaptive behavior
        self.auto_level_shift = config.get("auto_level_shift", True)
        
        # Threshold condition number for automatic level-shifting
        # If condition number exceeds this, automatically apply shift
        self.condition_number_threshold = config.get("condition_number_threshold", 1e8)
        
        # Track whether shift was applied in current iteration
        self.level_shift_applied = False
        # === [END MODIFICATION] ===

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
        
        # Build the prioritized list of Hessian updaters
        self._build_hessian_updater_list()
        
        # Initial alpha values to try - more memory efficient than np.linspace
        self.alpha_init_values = [0.001 + (10.0 - 0.001) * i / 14 for i in range(15)]
        self.NEB_mode = False
        
        

    def _build_hessian_updater_list(self):
        """
        Builds the prioritized dispatch list for Hessian updaters.
        The order of this list is CRITICAL as it mimics the original
        if/elif chain (most specific matches must come first).
        """
        
        # Define the default (fallback) method
        # We store this tuple (name, function)
        self.default_update_method = (
            "auto (default)",
            lambda h, d, g: self.hessian_updater.flowchart_hessian_update(h, d, g, "auto")
        )

        # List of (substring_key, display_name, function) tuples
        # The order MUST match the original if/elif logic exactly.
        self.updater_dispatch_list = [
            # (key to check with 'in', name for logging, function to call)
            
            ("flowchart", "flowchart", lambda h, d, g: self.hessian_updater.flowchart_hessian_update(h, d, g, "auto")),
            
            # --- Block methods (most specific first) ---
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

            # --- Standard methods (specific first) ---
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
        
        # === [MODIFIED] First eigendecomposition: Full Hessian H ===
        H = 0.5 * (H + H.T)  # Ensure symmetry
        # Use new method that applies/removes shift for numerical stability
        eigvals, eigvecs = self.compute_eigendecomposition_with_shift(H)

        # Always check conditioning (provides useful diagnostic information)
        condition_number, is_ill_conditioned = self.check_hessian_conditioning(eigvals)
        print(f"Condition number of Hessian: {condition_number:.2f}, Ill-conditioned: {is_ill_conditioned}")
        
        
        # Trust Radius Adjustment (Moved here to use eigenvalues)
        if not self.Initialization:
            if self.prev_energy is not None:
                actual_energy_change = B_e - self.prev_energy
                
                # Keep limited history
                if len(self.actual_energy_changes) >= 3:
                    self.actual_energy_changes.pop(0)
                self.actual_energy_changes.append(actual_energy_change)
                
                if self.predicted_energy_changes:
                    # Pass the minimum eigenvalue (which is the first one after eigh)
                    min_eigval = eigvals[0] if len(eigvals) > 0 else None
                    self.adjust_trust_radius(
                        actual_energy_change,
                        self.predicted_energy_changes[-1],
                        min_eigval,  # Pass minimum eigenvalue
                        gradient_norm # === [MODIFIED] Pass gradient norm ===
                    )

        # Count negative eigenvalues for diagnostic purposes
        neg_eigvals = np.sum(eigvals < -1e-10)
        self.log(f"Found {neg_eigvals} negative eigenvalues (target for saddle order: {self.saddle_order})", force=True)
        
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
        
        eigvals_star, eigvecs_star = self.compute_eigendecomposition_with_shift(H_star)
        
        # === Apply existing small eigenvalue filter ===
        # This is INDEPENDENT of level-shifting.
        # Level-shifting affects numerical stability during computation.
        # This filtering affects which eigenvalues are used in optimization.
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
        
        # Keep limited history - only store the last few values
        if len(self.predicted_energy_changes) >= 3:
            self.predicted_energy_changes.pop(0)
        self.predicted_energy_changes.append(predicted_energy_change)
        
        self.log(f"Predicted energy change: {predicted_energy_change:.6f}", force=True)
        
        # Evaluate step quality if we have history
        if self.actual_energy_changes and len(self.predicted_energy_changes) > 1:
            self.evaluate_step_quality()
            
        # Store current geometry, gradient and energy for next iteration (no deep copy)
        self.prev_geometry = geom_num_list
        self.prev_gradient = B_g
        self.prev_energy = current_energy
        
        # Increment iteration counter
        self.iteration += 1
        
        return -1 * move_vector.reshape(-1, 1)

    def check_hessian_conditioning(self, eigvals):
        """
        Check the condition number of the Hessian.
        
        The condition number κ = |λ_max| / |λ_min| indicates how ill-conditioned
        the matrix is. Large condition numbers suggest numerical instability.
        
        This method filters out near-zero eigenvalues (likely from projected-out
        modes like translation/rotation) before computing the condition number.
        
        Parameters:
            eigvals: np.ndarray
                Eigenvalues of the Hessian (sorted in ascending order)
        
        Returns:
            condition_number: float or None
                Condition number of the Hessian, or None if cannot be computed
            is_ill_conditioned: bool
                True if Hessian is considered ill-conditioned
        """
        if len(eigvals) < 2:
            self.log("Warning: Too few eigenvalues to compute condition number", force=True)
            return None, False
        
        # Filter eigenvalues: exclude those near zero
        # These are typically translation/rotation modes that were projected out
        # Note: This is different from filter_small_eigvals() which filters after all processing
        nonzero_mask = np.abs(eigvals) > 1e-10
        nonzero_eigvals = eigvals[nonzero_mask]
        
        if len(nonzero_eigvals) < 2:
            self.log("Warning: Insufficient non-zero eigenvalues for condition number", force=True)
            return None, True  # Likely ill-conditioned
        
        # Condition number = |λ_max| / |λ_min| among non-zero eigenvalues
        max_abs_eigval = np.max(np.abs(nonzero_eigvals))
        min_abs_eigval = np.min(np.abs(nonzero_eigvals))
        
        if min_abs_eigval < 1e-15:
            self.log("Warning: Extremely small minimum eigenvalue detected", force=True)
            return None, True
        
        condition_number = max_abs_eigval / min_abs_eigval
        
        # Classify conditioning
        is_ill_conditioned = condition_number > self.condition_number_threshold
        
        # Diagnostic output
        if condition_number > 1e10:
            self.log(f"WARNING: Hessian is severely ill-conditioned (κ={condition_number:.2e})", force=True)
            if not self.use_level_shift and not self.auto_level_shift:
                self.log("  Suggestion: Enable auto_level_shift=True for better stability", force=True)
        elif condition_number > 1e8:
            self.log(f"CAUTION: Hessian is ill-conditioned (κ={condition_number:.2e})", force=True)
        elif condition_number > 1e6:
            self.log(f"Hessian condition number is moderate (κ={condition_number:.2e})")
        else:
            self.log(f"Hessian is well-conditioned (κ={condition_number:.2e})")
        
        return condition_number, is_ill_conditioned

    def compute_eigendecomposition_with_shift(self, H):
        """
        Compute eigenvalue decomposition with optional level-shifting.
        
        Level-shifting temporarily improves numerical conditioning during the
        eigenvalue computation by adding a uniform shift to all diagonal elements:
            H_shifted = H + shift * I
        
        The shift is removed from eigenvalues afterward, so the returned eigenvalues
        are identical to those from standard eigendecomposition (in exact arithmetic).
        
        Key properties:
        - Eigenvalues: λ_shifted = λ + shift, so λ = λ_shifted - shift
        - Eigenvectors: Unchanged by uniform shift
        - Numerical stability: Improved during computation
        - Final result: Same as non-shifted (after shift removal)
        
        This is fully compatible with subsequent filter_small_eigvals():
            Workflow: shift → compute → remove shift → filter small eigenvalues
        
        The method works for ALL saddle orders:
        - saddle_order = 0: Shift improves positive eigenvalue conditioning
        - saddle_order > 0: Shift improves conditioning without affecting negative eigenvalues
                           (negative eigenvalues remain negative after shift removal)
        
        Parameters:
            H: np.ndarray
                Hessian matrix (symmetric, n×n)
        
        Returns:
            eigvals: np.ndarray
                Eigenvalues (with shift removed, identical to original)
            eigvecs: np.ndarray
                Eigenvectors (unchanged by shift)
        """
        n = H.shape[0]
        self.level_shift_applied = False
        
        # === Decide whether to apply level-shifting ===
        apply_shift = False
        shift_reason = ""
        eigvals_check = None # To store results from auto-check
        eigvecs_check = None
        
        if self.use_level_shift:
            # User explicitly requested level-shifting
            apply_shift = True
            shift_reason = "user-enabled"
            
        elif self.auto_level_shift:
            # Automatic level-shifting based on condition number
            try:
                # Quick eigendecomposition to check conditioning
                eigvals_check, eigvecs_check = np.linalg.eigh(H)
                condition_number, is_ill_conditioned = self.check_hessian_conditioning(eigvals_check)
                
                if is_ill_conditioned:
                    apply_shift = True
                    shift_reason = f"auto (κ={condition_number:.2e})"
                    self.log(f"Auto level-shifting triggered: κ={condition_number:.2e} > threshold={self.condition_number_threshold:.2e}", force=True)
            except Exception as e:
                self.log(f"Could not check condition number for auto level-shift: {e}")
                apply_shift = False
        
        # === Perform eigendecomposition ===
        if apply_shift:
            shift = self.level_shift_value
            self.log(f"Applying level shift: {shift:.2e} ({shift_reason})", force=True)
            
            # Add uniform shift to all diagonal elements
            H_shifted = H + shift * np.eye(n)
            
            # Eigendecomposition of shifted matrix
            eigvals_shifted, eigvecs = np.linalg.eigh(H_shifted)
            
            # Remove shift to restore original eigenvalues
            eigvals = eigvals_shifted - shift
            
            self.level_shift_applied = True
            
            # Diagnostic output
            if self.debug_mode:
                self.log(f"  Eigenvalue range (original):      [{eigvals[0]:.6e}, {eigvals[-1]:.6e}]")
                self.log(f"  Eigenvalue range (shifted):     [{eigvals_shifted[0]:.6e}, {eigvals_shifted[-1]:.6e}]")
                self.log(f"  Eigenvalue range (after removal): [{eigvals[0]:.6e}, {eigvals[-1]:.6e}]")
                self.log(f"  Note: Small eigenvalue filtering (if any) will be applied separately by filter_small_eigvals()")
            else:
                self.log(f"  Level shift applied during computation and removed from eigenvalues")
                self.log(f"  Final eigenvalues are identical to non-shifted computation")
        
        else:
            # Standard eigendecomposition without shift
            # Check if we already computed it during the auto-check
            if eigvals_check is not None:
                self.log("No level shift applied (auto-check passed)")
                eigvals, eigvecs = eigvals_check, eigvecs_check
            else:
                # Both use_level_shift and auto_level_shift were False
                self.log("No level shift applied (disabled)")
                eigvals, eigvecs = np.linalg.eigh(H)
                
            if self.debug_mode and not (eigvals_check is not None):
                self.log(f"No level shift applied")
        
        return eigvals, eigvecs
    

    def adjust_trust_radius_adaptive(self, actual_change, predicted_change, min_eigenvalue):
        """
        Adaptive trust radius update.
        Adjusts the trust radius considering Hessian curvature information (minimum eigenvalue).
        
        Parameters:
            actual_change: float
                Actual energy change (current energy - previous energy)
            predicted_change: float
                Predicted energy change from the RFO model
            min_eigenvalue: float
                Minimum eigenvalue of the Hessian (curvature information)
                - Positive value: curvature in the minimization direction
                - Negative value: curvature in the maximization/saddle point direction
        """
        # Skip if predicted change is too small
        if abs(predicted_change) < 1e-10:
            self.log("Skipping trust radius update: predicted change too small")
            return
        
        # === Step 1: Evaluate prediction accuracy ===
        # ratio = actual change / predicted change
        # Ideally ratio ≈ 1.0 (prediction is accurate)
        ratio = actual_change / predicted_change
        
        self.log(f"Step quality: actual={actual_change:.6e}, predicted={predicted_change:.6e}, ratio={ratio:.3f}")
        
        # === Step 2: Calculate adjustment factor based on curvature ===
        # Large curvature (steep) → small step is appropriate → smaller factor
        # Small curvature (flat) → large step is safe → larger factor
        
        abs_eigenvalue = abs(min_eigenvalue)
        
        if abs_eigenvalue > 1e-6:
            # When curvature is clearly present
            # Use relationship like curvature_factor = 1 / sqrt(|λ_min|)
            # But set an upper limit
            curvature_factor = min(
                self.max_curvature_factor,
                1.0 / max(abs_eigenvalue, 0.1)
            )
        else:
            # When curvature is nearly zero (very flat)
            # Allow larger steps
            curvature_factor = 1.5
        
        # === Step 3: Additional adjustment for transition state searches ===
        if self.saddle_order > 0:
            if min_eigenvalue < -1e-6:
                # Negative curvature direction (reaction coordinate of transition state)
                # Adjust step size more carefully
                curvature_factor *= self.negative_curvature_safety
                self.log(f"Negative curvature detected (λ_min={min_eigenvalue:.6e}), "
                         f"applying safety factor {self.negative_curvature_safety}")
            elif min_eigenvalue > 1e-6:
                # If minimum is positive curvature, may have crossed the transition state
                self.log(f"Warning: Positive minimum eigenvalue (λ_min={min_eigenvalue:.6e}) "
                         f"in transition state search", force=True)
        
        # === Step 4: Trust radius adjustment based on ratio ===
        old_trust_radius = self.trust_radius
        
        if ratio > 0.75:
            # === Excellent prediction accuracy (ratio > 0.75) ===
            # Model is very accurate → aggressively increase
            increase_factor = 1.5 * curvature_factor
            # Set upper limit (don't change too drastically at once)
            increase_factor = min(increase_factor, self.max_curvature_factor)
            
            self.trust_radius = min(
                self.trust_radius * increase_factor,
                self.trust_radius_max
            )
            status = "excellent"
            
        elif ratio > 0.5:
            # === Good prediction accuracy (0.5 < ratio ≤ 0.75) ===
            # Model is generally accurate → gradually increase
            increase_factor = 1.1 * curvature_factor
            increase_factor = min(increase_factor, 1.5)
            
            self.trust_radius = min(
                self.trust_radius * increase_factor,
                self.trust_radius_max
            )
            status = "good"
            
        elif ratio > 0.25:
            # === Acceptable prediction accuracy (0.25 < ratio ≤ 0.5) ===
            # Model accuracy is moderate
            
            if curvature_factor > 1.2:
                # Flat region (small curvature) → try increasing slightly
                self.trust_radius = min(
                    self.trust_radius * 1.05,
                    self.trust_radius_max
                )
                status = "acceptable (expanding slowly)"
            else:
                # Steep region or normal curvature → maintain
                status = "acceptable (maintaining)"
            
        elif ratio > 0.1:
            # === Poor prediction (0.1 < ratio ≤ 0.25) ===
            # Model accuracy is low → decrease
            self.trust_radius = max(
                self.trust_radius * 0.5,
                self.trust_radius_min
            )
            status = "poor"
            
        else:
            # === Very poor prediction (ratio ≤ 0.1 or ratio < 0) ===
            # Model is completely inaccurate, or energy increased → drastically decrease
            self.trust_radius = max(
                self.trust_radius * 0.25,
                self.trust_radius_min
            )
            status = "very poor"
        
        # === Step 5: Boundary check ===
        self.trust_radius = np.clip(
            self.trust_radius,
            self.trust_radius_min,
            self.trust_radius_max
        )
        
        # === Step 6: Log output ===
        if self.trust_radius != old_trust_radius:
            self.log(
                f"Trust radius adjusted: {old_trust_radius:.6f} → {self.trust_radius:.6f}",
                force=True
            )
            self.log(
                f"  Reason: ratio={ratio:.3f}, curvature_factor={curvature_factor:.3f}, "
                f"λ_min={min_eigenvalue:.6e}, status={status}"
            )
        else:
            self.log(f"Trust radius maintained: {self.trust_radius:.6f} (status={status})")
        
        # Optional: Save minimum eigenvalue history (for trend analysis)
        if len(self.min_eigenvalue_history) >= 10:
            self.min_eigenvalue_history.pop(0)
        self.min_eigenvalue_history.append(min_eigenvalue)

    def adjust_trust_radius(self, actual_change, predicted_change, min_eigenvalue=None, gradient_norm=None):
        """
        Trust radius adjustment.
        
        If the adaptive method is enabled, min_eigenvalue is provided,
        AND the gradient_norm is below the threshold,
        performs adjustment considering curvature information.
        Otherwise, uses the conventional simple method.
        
        Parameters:
            actual_change: float
                Actual energy change
            predicted_change: float
                Predicted energy change
            min_eigenvalue: float, optional
                Minimum eigenvalue of the Hessian (default: None)
            gradient_norm: float, optional
                Current L2 norm of the gradient. Used to conditionally
                activate the adaptive method.
        """
        
        # === [MODIFIED] Check conditions for using the ADAPTIVE method ===
        
        # 1. Must be globally enabled
        # 2. Must have the minimum eigenvalue
        can_use_adaptive = self.use_adaptive_trust_radius and min_eigenvalue is not None

        if can_use_adaptive:
            # If gradient norm was provided, check if it's below the threshold
            if gradient_norm is not None:
                if gradient_norm < self.adaptive_trust_gradient_norm_threshold:
                    # Gradient is small enough -> Use ADAPTIVE
                    self.log(f"Gradient norm ({gradient_norm:.6f}) < threshold "
                             f"({self.adaptive_trust_gradient_norm_threshold:.6f}). "
                             f"Using ADAPTIVE trust radius.", force=True)
                    self.adjust_trust_radius_adaptive(actual_change, predicted_change, min_eigenvalue)
                    return
                else:
                    # Gradient is still large -> Fallback to CONVENTIONAL
                    self.log(f"Gradient norm ({gradient_norm:.6f}) >= threshold "
                             f"({self.adaptive_trust_gradient_norm_threshold:.6f}). "
                             f"Using CONVENTIONAL trust radius.", force=True)
            else:
                # Gradient norm was *not* provided, but adaptive is on.
                # Default to using it (legacy behavior for backward compatibility).
                self.log("Gradient norm not provided. Defaulting to ADAPTIVE trust radius.")
                self.adjust_trust_radius_adaptive(actual_change, predicted_change, min_eigenvalue)
                return
        
        # === Conventional simple method (fallback) ===
        # (This block is reached if can_use_adaptive=False OR if gradient was too large)
        
        if abs(predicted_change) < 1e-10:
            self.log("Skipping trust radius update: predicted change too small")
            return
        
        ratio = actual_change / predicted_change
        
        self.log(f"Energy change: actual={actual_change:.6f}, predicted={predicted_change:.6f}, ratio={ratio:.3f}", force=True)
        
        old_trust_radius = self.trust_radius
        
        if ratio > self.good_step_threshold:
            # Good step
            self.trust_radius = min(
                self.trust_radius * self.trust_radius_increase_factor,
                self.trust_radius_max
            )
            if self.trust_radius != old_trust_radius:
                self.log(f"Good step quality (ratio={ratio:.3f}), increasing trust radius to {self.trust_radius:.6f}", force=True)
                
        elif ratio < self.poor_step_threshold:
            # Poor step
            self.trust_radius = max(
                self.trust_radius * self.trust_radius_decrease_factor,
                self.trust_radius_min
            )
            if self.trust_radius != old_trust_radius:
                self.log(f"Poor step quality (ratio={ratio:.3f}), decreasing trust radius to {self.trust_radius:.6f}", force=True)
                
        else:
            # Acceptable step
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
        
        try:
            # Calculate step with default alpha (alpha0) using the new O(N) solver
            initial_step, _, _, _ = self.solve_rfo(eigvals, gradient_trans, self.alpha0)
            initial_step_norm = np.linalg.norm(initial_step)
            
            self.log(f"Initial step with alpha={self.alpha0:.6f} has norm={initial_step_norm:.6f}", force=True)
            
            # If the step is already within trust radius, use it directly
            if initial_step_norm <= self.trust_radius:
                self.log(f"Initial step is within trust radius ({self.trust_radius:.6f}), using it directly", force=True)
                # Transform step back to original basis
                final_step = np.dot(eigvecs, initial_step)
                return final_step
                
            self.log(f"Initial step exceeds trust radius, optimizing alpha to match radius...", force=True)
            
            # --- MODIFICATION START ---
            # If the initial step is outside the trust radius, we must find the
            # alpha that puts the step *on* the trust radius boundary.
            # We call compute_rsprfo_step *once* to solve this.
            
            step, step_norm, final_alpha = self.compute_rsprfo_step(
                eigvals, gradient_trans, self.alpha0
            )
            
            self.log(f"Optimized alpha={final_alpha:.6f} to get step_norm={step_norm:.6f}", force=True)
            
            # Transform step back to original basis (use matrix multiplication for efficiency)
            step_original_basis = np.dot(eigvecs, step)
            
            step_norm_original = np.linalg.norm(step_original_basis)
            self.log(f"Final norm(step)={step_norm_original:.6f}", force=True)
        
            return step_original_basis
            
            # --- MODIFICATION END ---

        except Exception as e:
            self.log(f"Error during RS step calculation: {str(e)}", force=True)
            # If all else fails, use a steepest descent step
            self.log("Using steepest descent step as fallback", force=True)
            sd_step = -gradient_trans
            sd_norm = np.linalg.norm(sd_step)
            
            if sd_norm > self.trust_radius:
                best_overall_step = sd_step / sd_norm * self.trust_radius
            else:
                best_overall_step = sd_step
                
            # Transform step back to original basis
            step = np.dot(eigvecs, best_overall_step)
            
            step_norm = np.linalg.norm(step)
            self.log(f"Final norm(step)={step_norm:.6f}", force=True)
            
            return step

    def compute_rsprfo_step(self, eigvals, gradient_trans, alpha_init):
        """
        Compute an RS-P-RFO step using a specific initial alpha value.
        Prioritizes Brent's method (brentq) for finding the root 'alpha'
        that matches the trust radius, falling back to Newton iterations
        only if brentq fails or its result is not sufficiently precise.
        """
        
        # Pre-calculate squared gradient components for efficiency
        grad_trans_sq = gradient_trans**2
        
        # Create proxy functions for step norm calculation
        def calculate_step(alpha):
            """Calculate RFO step for a given alpha value"""
            try:
                # Use the new O(N) solver
                step, eigval_min, _, _ = self.solve_rfo(eigvals, gradient_trans, alpha)
                return step, eigval_min
            except Exception as e:
                self.log(f"Error in step calculation: {str(e)}")
                raise
                
        def step_norm_squared(alpha):
            """Calculate ||step||^2 for a given alpha value"""
            # This function is only used by brentq, which only needs the step norm
            step, _ = calculate_step(alpha)
            return np.dot(step, step)
            
        def objective_function(alpha):
            """U(a) = ||step||^2 - R^2"""
            return step_norm_squared(alpha) - self.trust_radius**2

        # --- MODIFICATION START ---
        # Prioritize Brent's method (brentq) as it does not rely on derivatives.
        
        alpha_lo = 1e-6  # Very small alpha gives large step
        alpha_hi = self.alpha_max  # Very large alpha gives small step
        
        try:
            # Check step norms at boundaries to establish bracket
            step_lo, _ = calculate_step(alpha_lo)
            norm_lo = np.linalg.norm(step_lo)
            obj_lo = norm_lo**2 - self.trust_radius**2
            
            step_hi, _ = calculate_step(alpha_hi) 
            norm_hi = np.linalg.norm(step_hi)
            obj_hi = norm_hi**2 - self.trust_radius**2
            
            self.log(f"Bracket search: alpha_lo={alpha_lo:.6e}, step_norm={norm_lo:.6f}, obj={obj_lo:.6e}")
            self.log(f"Bracket search: alpha_hi={alpha_hi:.6e}, step_norm={norm_hi:.6f}, obj={obj_hi:.6e}")
            
            # Check if we have a proper bracket (signs differ)
            if obj_lo * obj_hi < 0:
                # We have a bracket, use Brent's method for robust root finding
                self.log("Bracket established, using Brent's method (brentq) for root finding")
                
                alpha_brent = brentq(objective_function, alpha_lo, alpha_hi, 
                                     xtol=1e-6, rtol=1e-6, maxiter=50)
                
                self.log(f"Brent's method converged to alpha={alpha_brent:.6e}")
                
                # Calculate the step using the alpha from brentq
                step, _ = calculate_step(alpha_brent)
                step_norm = np.linalg.norm(step)
                norm_diff = abs(step_norm - self.trust_radius)
                
                # Check if the result from brentq is within the strict tolerance
                if norm_diff < self.step_norm_tolerance:
                    self.log(f"brentq result is within tolerance ({self.step_norm_tolerance:.2e}). Using this step (norm={step_norm:.6f}).")
                    # Return immediately, skipping the Newton loop
                    return step, step_norm, alpha_brent
                else:
                    self.log(f"brentq result norm={step_norm:.6f} (diff={norm_diff:.2e}) still outside tolerance. Proceeding to Newton refinement.")
                    # Use the brentq result as the starting point for Newton
                    alpha = alpha_brent
            
            else:
                # No bracket, so use initial alpha and proceed with Newton iterations
                self.log("Could not establish bracket with opposite signs, proceeding to Newton iterations")
                alpha = alpha_init
                
        except Exception as e:
            # Handle any error during bracketing or brentq
            self.log(f"Error during brentq attempt: {str(e)}. Falling back to Newton iterations with initial alpha.")
            alpha = alpha_init
            
        # --- MODIFICATION END ---


        # Fallback: Use Newton iterations to refine alpha (or if brentq was imprecise)
        # 'alpha' is either alpha_init (if brentq failed) or alpha_brent (if brentq succeeded but was imprecise)
        
        self.log(f"Starting Newton refinement loop with alpha={alpha:.6f}")

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
            self.log(f"RS-I-RFO (Newton) micro cycle {mu:02d}, alpha={alpha:.6f}")
            
            try:
                # Calculate current step and its properties
                # (Re-use eigval_min from calculate_step)
                step, eigval_min = calculate_step(alpha)
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
                    self.log(f"Step norm {step_norm:.6f} is sufficiently close to trust radius. Newton loop converged.")
                    # --- MODIFICATION ---
                    # (Original code had: if mu >= 1: break)
                    # We now break immediately upon convergence.
                    best_step = step # Ensure the final step is the one that converged
                    break
                
                # Track step norm history for convergence detection (use fixed size array)
                if history_count < self.max_micro_cycles:
                    step_norm_history[history_count] = step_norm
                    history_count += 1
                
                # Compute derivative of squared step norm with respect to alpha
                # (Pass computed step and eigval_min to avoid re-calculation)
                dstep2_dalpha = self.get_step_derivative(alpha, eigvals, gradient_trans, 
                                                         step=step, eigval_min=eigval_min)
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
            # === [MODIFIED] If we exhausted micro-cycles without converging ===
            self.log(f"RS-I-RFO (Newton) did not converge in {self.max_micro_cycles} cycles", force=True)
            
            # Check if the 'best_step' found is close enough to the trust radius
            if best_step is not None:
                best_step_norm = np.linalg.norm(best_step)
                # Use a slightly relaxed tolerance
                if abs(best_step_norm - self.trust_radius) < self.step_norm_tolerance * 1.1:
                     self.log(f"Using best step found during iterations (norm={best_step_norm:.6f} was close enough)")
                     step = best_step
                     step_norm = best_step_norm
                else:
                     # If 'best_step' is not close (e.g., norm=506),
                     # discard it as junk and fall back to safe steepest descent.
                     self.log(f"Best step found (norm={best_step_norm:.6f}) was NOT close to trust radius. Forcing steepest descent.", force=True)
                     step = -gradient_trans
                     step_norm = np.linalg.norm(step)
                     if step_norm > 1e-10:
                         step = step / step_norm * self.trust_radius
                     else:
                         step = np.zeros_like(gradient_trans) # Gradient is zero
                     step_norm = self.trust_radius
            else:
                # If no 'best_step' was ever found, fall back to steepest descent.
                self.log("No usable step found. Forcing steepest descent as a last resort.", force=True)
                step = -gradient_trans
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-10:
                    step = step / step_norm * self.trust_radius
                else:
                    step = np.zeros_like(gradient_trans) # Gradient is zero
                step_norm = self.trust_radius
            # === [END MODIFICATION] ===
        
        return step, step_norm, alpha
        
    def get_step_derivative(self, alpha, eigvals, gradient_trans, step=None, eigval_min=None):
        """
        Compute derivative of squared step norm with respect to alpha directly.
        Assumes eigval_min is (approximately) constant w.r.t alpha.
        """
        # If step or eigval_min was not provided, compute them
        if step is None or eigval_min is None:
            try:
                # Use the new O(N) solver
                step, eigval_min, _, _ = self.solve_rfo(eigvals, gradient_trans, alpha)
            except Exception as e:
                self.log(f"Error in step calculation for derivative: {str(e)}")
                return 1e-8  # Return a small value as fallback
        
        try:
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
        
        # --- [Refactored Method Dispatch (maintaining 'in' logic)] ---
        
        method_key_lower = self.hessian_update_method.lower()
        
        # Default values (fallback)
        method_name, update_function = self.default_update_method
        found_method = False

        # Iterate through the prioritized list
        for key, name, func in self.updater_dispatch_list:
            if key in method_key_lower:
                method_name = name
                update_function = func
                found_method = True
                break  # Found the first (highest priority) match

        if not found_method:
             self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
        
        self.log(f"Hessian update method: {method_name}")
        
        # Call the selected function (either found or default)
        delta_hess = update_function(
            self.hessian, displacement, delta_grad
        )
        
        # --- [End of Refactored Section] ---
            
        # Update the Hessian (in-place addition)
        self.hessian += delta_hess
    
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        # Use in-place operation for symmetrization
        self.hessian = 0.5 * (self.hessian + self.hessian.T)
        
    def _solve_secular_safeguarded(self, eigvals_prime, grad_comps_prime_sq, lambda_min_asymptote, initial_guess):
        """
        [NEW] Safeguarded Newton's Method for the RFO Secular Equation.
        
        This solver is specifically designed for the secular equation's structure.
        It combines the rapid convergence of Newton's method with the
        guaranteed convergence of bisection.
        
        It maintains a bracket [a, b] known to contain the root and uses
        Newton's method. If the Newton step would fall outside the bracket,
        it reverts to a bisection step.
        """
        
        # Define the secular function and its derivative
        def f_secular(lmd):
            denominators = eigvals_prime - lmd
            # Safety for division
            safe_denoms = np.where(
                np.abs(denominators) < 1e-30,
                np.sign(denominators) * 1e-30,
                denominators
            )
            safe_denoms[safe_denoms == 0] = 1e-30 # Handle exact zeros
            terms_f = grad_comps_prime_sq / safe_denoms
            return lmd + np.sum(terms_f)
            
        def f_prime_secular(lmd):
            denominators = eigvals_prime - lmd
            safe_denoms = np.where(
                np.abs(denominators) < 1e-30,
                np.sign(denominators) * 1e-30,
                denominators
            )
            safe_denoms[safe_denoms == 0] = 1e-30
            terms_f_prime = grad_comps_prime_sq / (safe_denoms**2)
            return 1.0 + np.sum(terms_f_prime)

        # --- Setup Bracket [a, b] ---
        # b is the upper bound (the first pole)
        b = lambda_min_asymptote
        
        # a is the lower bound. We need f(a) < 0.
        # Start with the initial guess.
        a = initial_guess
        f_a = f_secular(a)
        
        # If f(a) is not negative, step back until it is.
        g_norm = np.sqrt(np.sum(grad_comps_prime_sq))
        search_limit = 10
        while f_a > 0 and search_limit > 0:
            self.log(f"  Safeguard Solver: f(a) > 0 at a={a:.6e}. Stepping back.")
            step_back = max(g_norm, np.abs(a) * 0.1, 1e-8)
            a = a - step_back
            f_a = f_secular(a)
            search_limit -= 1
            
        if f_a > 0:
            self.log(f"  Safeguard Solver: Could not establish lower bound 'a'.", force=True)
            return initial_guess # Fallback

        # We don't calculate f(b) because it's +infinity.
        # We know the root is in [a, b).
        
        # Start iteration from the best initial guess
        lambda_k = initial_guess
        if lambda_k <= a or lambda_k >= b:
             lambda_k = (a + b) / 2.0 # Fallback to bisection if guess is out of bounds

        self.log(f"  Safeguard Solver: Starting search in [{a:.6e}, {b:.6e}]")

        max_iterations = 50
        # Use a tolerance relative to the pole
        tolerance = (1e-10 * abs(lambda_min_asymptote)) + 1e-12 

        for iteration in range(max_iterations):
            f_lambda = f_secular(lambda_k)
            
            # Check convergence
            if abs(f_lambda) < tolerance:
                self.log(f"  Safeguard Solver: Converged in {iteration + 1} iterations", force=True)
                self.log(f"  Final: lambda_aug={lambda_k:.6e}, f(λ)={f_lambda:.2e}")
                return lambda_k
            
            f_prime_lambda = f_prime_secular(lambda_k)
            
            # --- Calculate Newton Step ---
            delta_newton = 0.0
            if abs(f_prime_lambda) > 1e-20:
                delta_newton = -f_lambda / f_prime_lambda
            else:
                self.log(f"  Warning: f'(λ) too small. Switching to bisection.")

            lambda_newton = lambda_k + delta_newton
            
            # --- Calculate Bisection Step ---
            lambda_bisection = (a + b) / 2.0

            # --- Safeguard Check ---
            # Is the Newton step safe (i.e., within the bracket [a, b])?
            if (delta_newton != 0.0) and (lambda_newton > a) and (lambda_newton < b):
                # Yes: Use Newton step
                lambda_k_next = lambda_newton
                if self.debug_mode:
                    self.log(f"  Iter {iteration:2d} (Newton): λ={lambda_k_next:.6e}")
            else:
                # No: Use safe bisection step
                lambda_k_next = lambda_bisection
                if self.debug_mode:
                    self.log(f"  Iter {iteration:2d} (Bisection): λ={lambda_k_next:.6e}")
            
            # --- Update Bracket [a, b] for next iteration ---
            # (This is the key to safety)
            if f_lambda > 0:
                # Root is to the left, new upper bound is current lambda
                b = lambda_k
            else:
                # Root is to the right, new lower bound is current lambda
                a = lambda_k
                
            lambda_k = lambda_k_next
            
            # Check if bracket is too small
            if abs(b - a) < tolerance:
                 self.log(f"  Safeguard Solver: Bracket converged", force=True)
                 return (a + b) / 2.0

        else:
            # Max iterations reached
            self.log(f"Warning: Safeguard Solver did not converge in {max_iterations} iterations", force=True)
            return (a + b) / 2.0 # Return the center of the last known bracket
        
    def _solve_secular_more_sorensen(self, eigvals, grad_comps, alpha):
        """
        [MODIFIED] Robust solver for the RFO secular equation with fallback.
        
        Attempts to find the smallest root (lambda_aug) of the secular equation
        using brentq first for maximum robustness. If brentq fails (e.g.,
        cannot establish a bracket), it falls back to the Moré-Sorensen
        (Newton-style) solver.
        
        Secular equation:
            f(λ) = λ + Σ_i [g_i'^2 / (λ_i' - λ)] = 0
        
        Where: λ_i' = λ_i/α, g_i' = g_i/α
        
        Parameters:
            eigvals: np.ndarray (sorted ascending)
            grad_comps: np.ndarray
            alpha: float
        
        Returns:
            lambda_aug: float (smallest root)
        """
        
                # Define the secular function and its derivative
        def f_secular(lmd):
            denominators = eigvals_prime - lmd
            # Safety for division
            safe_denoms = np.where(
                np.abs(denominators) < 1e-30,
                np.sign(denominators) * 1e-30,
                denominators
            )
            safe_denoms[safe_denoms == 0] = 1e-30 # Handle exact zeros
            terms_f = grad_comps_prime_sq / safe_denoms
            return lmd + np.sum(terms_f)
            
   
        
        # 1. Scale values
        eigvals_prime = eigvals / alpha
        grad_comps_prime = grad_comps / alpha
        grad_comps_prime_sq = grad_comps_prime**2
        
        # 2. Find the first asymptote (smallest λ_i') where g_i' is non-zero
        lambda_min_asymptote = None
        g_norm_sq = 0.0
        
        for i in range(len(eigvals_prime)):
            g_sq = grad_comps_prime_sq[i]
            g_norm_sq += g_sq
            
            if lambda_min_asymptote is None and g_sq > 1e-20:
                lambda_min_asymptote = eigvals_prime[i]
                
        if lambda_min_asymptote is None:
            # Hard case: All gradient components are zero
            self.log("Hard case detected: All gradient components are zero.", force=True)
            return eigvals_prime[0]

        # 3. Initial guess (Baker, JCC 1986, Eq. 15)
        lambda_initial_guess = 0.5 * (lambda_min_asymptote - np.sqrt(max(0.0, lambda_min_asymptote**2 + 4 * g_norm_sq)))

        # 4. Call the dedicated solver
        try:
            lambda_aug = self._solve_secular_safeguarded(
                eigvals_prime,
                grad_comps_prime_sq,
                lambda_min_asymptote,
                lambda_initial_guess
            )
            return lambda_aug
            
        except Exception as e:
            self.log(f"CRITICAL ERROR in _solve_secular_safeguarded: {e}", force=True)
            self.log("Falling back to initial guess as last resort.", force=True)
             
        # --- Primary Strategy: brentq ---
        try:
            self.log(f"Normal case: solving RFO secular equation f(λ)=0 using brentq (Primary)")
            self.log(f"First asymptote (lambda_min_asymptote) = {lambda_min_asymptote:.6e}")
            
            # --- Establish bracket [a, b] ---
            
            # b (upper bound) is just below the asymptote, f(b) should be large positive
            b_margin = max(1e-12, np.abs(lambda_min_asymptote) * 1e-10)
            b = lambda_min_asymptote - b_margin 
            f_b = f_secular(b)
            
            if f_b < 0:
                self.log(f"  Warning: f(b) < 0 at {b:.6e}. Evaluating at asymptote limit.")
                b = lambda_min_asymptote
                f_b = f_secular(b) # This will be large and positive due to safe_denoms
            
            # a (lower bound), f(a) must be < 0
            a = lambda_initial_guess
            f_a = f_secular(a)
            
            search_limit = 10
            while f_a > 0 and search_limit > 0:
                self.log(f"  brentq bracket search: f(a) > 0 at a={a:.6e}. Stepping back.")
                step_back = max(g_norm, np.abs(a) * 0.1, 1e-8) # Ensure step back is non-zero
                a = a - step_back
                f_a = f_secular(a)
                search_limit -= 1

            if f_a * f_b >= 0:
                # Failed to find a bracket
                self.log(f"  Error: Could not establish a bracket for brentq. [a,b]=[{a:.2e},{b:.2e}], [f(a),f(b)]=[{f_a:.2e},{f_b:.2e}]", force=True)
                # This will raise an exception, triggering the fallback
                raise ValueError("brentq bracketing failed")
            
            self.log(f"  brentq bracket established: [a, b] = [{a:.6e}, {b:.6e}], [f(a), f(b)] = [{f_a:.2e}, {f_b:.2e}]")
            
            # Use brentq to find the root
            lambda_aug_brent = brentq(f_secular, a, b, xtol=1e-10, rtol=1e-10, maxiter=100)
            
            self.log(f"  brentq solver converged: lambda_aug = {lambda_aug_brent:.6e}", force=True)
            return lambda_aug_brent # Return the successful brentq result

        except Exception as e:
            self.log(f"brentq solver failed ({str(e)}). Falling back to Moré-Sorensen (Newton) solver.", force=True)
            
            # --- Fallback Strategy: Moré-Sorensen (Newton) ---
            # (This logic is from the original file, lines 1445-1502, with English comments)
            
            lambda_aug = lambda_initial_guess
            self.log(f"Fallback (Newton): Initial lambda_aug guess = {lambda_aug:.6e}")
            
            max_iterations = 50
            tolerance = (1e-10 * abs(lambda_min_asymptote)) + 1e-12 

            for iteration in range(max_iterations):
                # Denominators (λ_i' - λ)
                denominators = eigvals_prime - lambda_aug
                
                # Safe denominators
                safe_denoms = np.where(
                    np.abs(denominators) < 1e-30,
                    np.sign(denominators) * 1e-30,
                    denominators
                )
                safe_denoms[safe_denoms == 0] = 1e-30 # Handle exact zeros
                
                # f(λ) and f'(λ)
                terms_f = grad_comps_prime_sq / safe_denoms
                terms_f_prime = grad_comps_prime_sq / (safe_denoms**2)
                
                f_lambda = lambda_aug + np.sum(terms_f)
                f_prime_lambda = 1.0 + np.sum(terms_f_prime)
                
                # Check convergence
                if abs(f_lambda) < tolerance:
                    self.log(f"RFO Newton (Fallback) converged in {iteration + 1} iterations", force=True)
                    self.log(f"Final: lambda_aug={lambda_aug:.6e}, f(λ)={f_lambda:.2e}")
                    break
                
                if abs(f_prime_lambda) < 1e-20:
                    self.log(f"Warning: f'(λ) too small ({f_prime_lambda:.2e}) at iteration {iteration}", force=True)
                    break 

                # Newton update
                delta_lambda = -f_lambda / f_prime_lambda
                
                lambda_aug_old = lambda_aug
                lambda_aug += delta_lambda
                
                # Safeguard: must stay below the asymptote
                if lambda_aug >= lambda_min_asymptote:
                    self.log(f"Warning: lambda_aug ({lambda_aug:.6e}) >= asymptote ({lambda_min_asymptote:.6e}), adjusting")
                    lambda_aug = 0.5 * (lambda_aug_old + lambda_min_asymptote)
                
                if self.debug_mode:
                     self.log(f"  Iter {iteration:2d}: λ={lambda_aug:.6e}, f(λ)={f_lambda:.2e}, "
                              f"f'(λ)={f_prime_lambda:.2e}, Δλ={delta_lambda:.2e}")

            else:
                # Max iterations reached for Newton
                self.log(f"Warning: RFO Newton (Fallback) did not converge in {max_iterations} iterations", force=True)
                self.log(f"Final residual f(λ): {f_lambda:.2e}. Using last value.", force=True)
            
            # Return the result from the Newton solver (even if it didn't converge, it's the best guess)
            return lambda_aug

    def solve_rfo(self, eigvals, gradient_components, alpha, mode="min"):
        """
        Solve the RFO equations to get the step using the O(N) secular equation.
        """
        if mode != "min":
            raise NotImplementedError("Secular equation solver is only implemented for RFO minimization (mode='min')")
            
        # 1. Find the smallest eigenvalue (lambda_aug) of the augmented Hessian
        eigval_min = self._solve_secular_more_sorensen(eigvals, gradient_components, alpha)

        # 2. Calculate the step components directly. This is O(N).
        denominators = (eigvals / alpha) - eigval_min
        
        # Safety for division
        safe_denoms = np.where(
            np.abs(denominators) < 1e-20,
            np.sign(denominators) * 1e-20,
            denominators
        )
        
        # Handle exact zeros that slipped through (e.g., in the 'hard case')
        safe_denoms[safe_denoms == 0] = 1e-20
        
        # Calculate step s_i = -(g_i/alpha) / (denominators)
        step = -(gradient_components / alpha) / safe_denoms
        
        # Return dummy values for nu and eigvec, as they are no longer computed
        return step, eigval_min, 1.0, None

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