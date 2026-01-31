import numpy as np
from numpy.linalg import norm
import copy

from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Optimizer.block_hessian_update import BlockHessianUpdate
from multioptpy.Utils.calc_tools import Calculationtools


class EnhancedRSPRFO:
    """
    Enhanced Rational Step P-RFO (Rational Function Optimization) for transition state searches
    with dynamic trust radius adjustment based on trust region methodology.
    
    Key Improvements:
    - Improved Levenberg-Marquardt-style alpha solver with backtracking
    - Enhanced trust region adjustment with asymmetric expansion/contraction
    - Robust Hessian update with curvature condition checks
    - Improved mode following with overlap matrix tracking
    - Step rejection mechanism for poor quality steps
    - Hessian eigenvalue shifting for proper TS curvature
    - Comprehensive step quality metrics and diagnostics
    - Gradient-based step scaling for near-convergence behavior
    
    References:
    [1] Banerjee et al., Phys. Chem., 89, 52-57 (1985)
    [2] Heyden et al., J. Chem. Phys., 123, 224101 (2005)
    [3] Baker, J. Comput. Chem., 7, 385-395 (1986)
    [4] Besalú and Bofill, Theor. Chem. Acc., 100, 265-274 (1998)
    [5] Jensen and Jørgensen, J. Chem. Phys., 80, 1204 (1984) [Eigenvector following]
    [6] Yuan, SIAM J. Optim. 11, 325-357 (2000) [Trust region methods]
    [7] Nocedal and Wright, Numerical Optimization, 2nd ed. (2006) [Trust region]
    
    This code is made based on:
    1. https://github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/RSPRFOptimizer.py
    """
    
    def __init__(self, **config):
        """
        Initialize the Enhanced RS-PRFO optimizer.
        
        Parameters (via config dict):
        -----------------------------
        alpha0 : float
            Initial alpha parameter for RS-PRFO (default: 1.0)
        max_micro_cycles : int
            Maximum number of micro-iterations for alpha adjustment (default: 50)
        saddle_order : int
            Number of negative eigenvalues at the saddle point (default: 1)
        method : str
            Hessian update method (default: "auto")
        display_flag : bool
            Enable/disable logging output (default: True)
        debug : bool
            Enable detailed debug output (default: False)
        trust_radius : float
            Initial trust radius (default: 0.1 for TS, 0.5 for min)
        trust_radius_max : float
            Maximum allowed trust radius (default: same as initial)
        trust_radius_min : float
            Minimum allowed trust radius (default: 0.01)
        adapt_trust_radius : bool
            Enable dynamic trust radius adjustment (default: True)
        mode_following : bool
            Enable mode following for consistent TS mode tracking (default: True)
        eigvec_following : bool
            Enable eigenvector following for mode mixing (default: True)
        overlap_threshold : float
            Minimum overlap for mode identification (default: 0.5)
        step_rejection : bool
            Enable step rejection for very poor steps (default: True)
        rejection_threshold : float
            Reduction ratio threshold below which steps are rejected (default: -0.5)
        hessian_shift_enabled : bool
            Enable Hessian eigenvalue shifting (default: True)
        min_positive_eigval : float
            Minimum positive eigenvalue after shifting (default: 0.005)
        gradient_scaling_enabled : bool
            Enable gradient-based step scaling near convergence (default: True)
        gradient_scaling_threshold : float
            Gradient norm threshold below which scaling is applied (default: 0.001)
        """
        # Standard RSPRFO parameters
        self.alpha0 = config.get("alpha0", 1.0)
        self.max_micro_cycles = config.get("max_micro_cycles", 50)
        self.saddle_order = config.get("saddle_order", 1)
        self.hessian_update_method = config.get("method", "auto")
        self.display_flag = config.get("display_flag", True)
        self.debug = config.get("debug", False)
        
        # Alpha constraints to prevent numerical instability
        self.alpha_max = config.get("alpha_max", 1e8)
        self.alpha_min = config.get("alpha_min", 1e-8)
        self.alpha_step_max = config.get("alpha_step_max", 100.0)
        
        # Micro-cycle convergence criteria
        self.micro_cycle_rtol = config.get("micro_cycle_rtol", 1e-3)
        self.micro_cycle_atol = config.get("micro_cycle_atol", 1e-6)
        
        # Trust region parameters
        if self.saddle_order == 0:
            self.trust_radius_initial = config.get("trust_radius", 0.5)
            self.trust_radius_max = config.get("trust_radius_max", 0.5)
        else:
            self.trust_radius_initial = config.get("trust_radius", 0.1)
            self.trust_radius_max = config.get("trust_radius_max", 0.3)
            
        self.trust_radius = self.trust_radius_initial
        self.trust_radius_min = config.get("trust_radius_min", 0.01)
        
        # Trust region acceptance thresholds (based on Nocedal & Wright)
        self.eta_1 = config.get("eta_1", 0.1)
        self.eta_2 = config.get("eta_2", 0.25)
        self.eta_3 = config.get("eta_3", 0.75)
        self.gamma_1 = config.get("gamma_1", 0.25)
        self.gamma_2 = config.get("gamma_2", 2.0)
        
        # Step rejection settings
        self.step_rejection_enabled = config.get("step_rejection", True)
        self.rejection_threshold = config.get("rejection_threshold", -0.5)
        self.max_consecutive_rejections = config.get("max_consecutive_rejections", 3)
        self.consecutive_rejections = 0
        
        # Hessian eigenvalue shifting - IMPROVED: smaller minimum to avoid over-shifting
        self.hessian_shift_enabled = config.get("hessian_shift_enabled", True)
        self.min_positive_eigval = config.get("min_positive_eigval", 0.001)  
        self.min_negative_eigval = config.get("min_negative_eigval", -0.001) 
        
        # NEW: Gradient-based step scaling for near-convergence
        self.gradient_scaling_enabled = config.get("gradient_scaling_enabled", True)
        self.gradient_scaling_threshold = config.get("gradient_scaling_threshold", 0.001)
        self.min_step_scale = config.get("min_step_scale", 0.1)  # Minimum scaling factor
        
        # NEW: Adaptive trust radius based on gradient magnitude
        self.adaptive_trust_enabled = config.get("adaptive_trust_enabled", True)
        self.gradient_trust_coupling = config.get("gradient_trust_coupling", 0.5)
        
        # Whether to use trust radius adaptation
        self.adapt_trust_radius = config.get("adapt_trust_radius", True)
        
        # Rest of initialization
        self.config = config
        self.Initialization = True
        self.iter = 0
        
        # Hessian-related variables
        self.hessian = None
        self.bias_hessian = None
        self.shifted_hessian = None
        
        # Optimization tracking variables
        self.prev_eigvec_max = None
        self.prev_eigvec_min = None
        self.predicted_energy_changes = []
        self.actual_energy_changes = []
        self.reduction_ratios = []
        self.trust_radius_history = []
        self.step_quality_history = []
        self.prev_geometry = None
        self.prev_gradient = None
        self.prev_energy = None
        self.prev_move_vector = None
        
        # Step rejection tracking
        self.rejected_step_geometry = None
        self.rejected_step_gradient = None
        
        # Mode Following specific parameters
        self.mode_following_enabled = config.get("mode_following", True)
        self.eigvec_history = []
        self.eigval_history = []
        self.ts_mode_idx = None
        self.ts_mode_eigvec = None
        
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
        """Builds the prioritized dispatch list for Hessian updaters."""
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
    
    def _project_grad_tr_rot(self, gradient, geometry):
        """
        Project out translation and rotation components from the gradient.
        Uses QR decomposition for orthonormalization.
        
        Parameters:
        gradient : numpy.ndarray
            Gradient vector to project
        geometry : numpy.ndarray
            Current geometry coordinates
            
        Returns:
        numpy.ndarray
            Projected gradient with TR/ROT components removed
        """
        coords = geometry.reshape(-1, 3)
        n_atoms = coords.shape[0]
        
        if n_atoms < 3:
            return gradient
        
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        
        basis = []
        
        # Translation (x, y, z)
        basis.append(np.tile([1, 0, 0], n_atoms))
        basis.append(np.tile([0, 1, 0], n_atoms))
        basis.append(np.tile([0, 0, 1], n_atoms))
        
        # Rotation (Rx, Ry, Rz via cross product)
        rx = np.zeros_like(coords)
        rx[:, 1] = -coords_centered[:, 2]
        rx[:, 2] = coords_centered[:, 1]
        basis.append(rx.flatten())
        
        ry = np.zeros_like(coords)
        ry[:, 0] = coords_centered[:, 2]
        ry[:, 2] = -coords_centered[:, 0]
        basis.append(ry.flatten())
        
        rz = np.zeros_like(coords)
        rz[:, 0] = -coords_centered[:, 1]
        rz[:, 1] = coords_centered[:, 0]
        basis.append(rz.flatten())
        
        A = np.array(basis).T
        Q, R = np.linalg.qr(A, mode='reduced')
        
        diag_R = np.abs(np.diag(R))
        valid_cols = diag_R > 1e-10
        Q = Q[:, valid_cols]
        
        overlaps = np.dot(Q.T, gradient)
        tr_rot_part = np.dot(Q, overlaps)
        projected_gradient = gradient - tr_rot_part
        
        return projected_gradient
    
    def _shift_hessian_eigenvalues(self, hessian, eigvals, eigvecs):
        """
        Shift Hessian eigenvalues to ensure proper curvature for TS search.
        
        IMPROVED: More conservative shifting to avoid over-constraining
        small eigenvalues that correspond to soft modes.
        
        For saddle_order > 0:
        - First `saddle_order` eigenvalues should be negative
        - Remaining eigenvalues should be positive (but allow small values)
        
        For saddle_order == 0 (minimization):
        - All eigenvalues should be positive
        
        Parameters:
        hessian : numpy.ndarray
            Original Hessian matrix
        eigvals : numpy.ndarray
            Eigenvalues of the Hessian
        eigvecs : numpy.ndarray
            Eigenvectors of the Hessian
            
        Returns:
        tuple
            (shifted_hessian, shifted_eigvals, shift_applied)
        """
        if not self.hessian_shift_enabled:
            return hessian, eigvals, False
        
        n = len(eigvals)
        shifted_eigvals = eigvals.copy()
        shift_applied = False
        
        if self.saddle_order == 0:
            # Minimization: all eigenvalues should be positive
            min_eigval = np.min(eigvals)
            if min_eigval < self.min_positive_eigval:
                shift = self.min_positive_eigval - min_eigval
                shifted_eigvals = eigvals + shift
                shift_applied = True
                self.log(f"Applied eigenvalue shift of {shift:.6f} for minimization")
        else:
            # TS search: need exactly saddle_order negative eigenvalues
            sorted_indices = np.argsort(eigvals)
            
            # Ensure first saddle_order eigenvalues are sufficiently negative
            for i in range(self.saddle_order):
                idx = sorted_indices[i]
                if eigvals[idx] > self.min_negative_eigval:
                    shifted_eigvals[idx] = self.min_negative_eigval
                    shift_applied = True
            
            # IMPROVED: Only shift eigenvalues that are very close to zero or negative
            # when they should be positive. Don't shift already positive eigenvalues
            # to a higher minimum unless they are problematically small.
            for i in range(self.saddle_order, n):
                idx = sorted_indices[i]
                # Only shift if eigenvalue is negative or very close to zero
                if eigvals[idx] < 1e-6:  # Much smaller threshold
                    shifted_eigvals[idx] = self.min_positive_eigval
                    shift_applied = True
        
        if shift_applied:
            shifted_hessian = eigvecs @ np.diag(shifted_eigvals) @ eigvecs.T
            shifted_hessian = 0.5 * (shifted_hessian + shifted_hessian.T)
            self.log(f"Hessian eigenvalues shifted for proper curvature")
            return shifted_hessian, shifted_eigvals, True
        
        return hessian, eigvals, False
    
    def _compute_gradient_based_scale(self, gradient_norm, step_norm):
        """
        Compute a scaling factor based on gradient magnitude to prevent
        overshooting near convergence.
        
        When the gradient is small but the step is large, this indicates
        the Hessian may have small eigenvalues causing large steps.
        
        Parameters:
        gradient_norm : float
            Norm of the current gradient
        step_norm : float
            Norm of the proposed step
            
        Returns:
        float
            Scaling factor (0 < scale <= 1)
        """
        if not self.gradient_scaling_enabled:
            return 1.0
        
        if gradient_norm < 1e-10 or step_norm < 1e-10:
            return 1.0
        
        # Expected step norm based on gradient and typical curvature
        # For a Newton step: s = -H^{-1}g, so |s| ~ |g| / |lambda_min|
        # If |s| >> |g| / typical_curvature, we should scale down
        
        # Use a simple heuristic: if step_norm / gradient_norm > threshold,
        # scale the step proportionally
        ratio = step_norm / gradient_norm
        
        # Typical ratio for well-conditioned systems is O(1) to O(10)
        # If ratio is very large (> 100), the Hessian likely has very small eigenvalues
        max_ratio = 50.0  # Maximum allowed ratio
        
        if ratio > max_ratio:
            scale = max_ratio / ratio
            scale = max(scale, self.min_step_scale)  # Don't scale below minimum
            self.log(f"Gradient-based scaling: ratio={ratio:.2f}, scale={scale:.4f}")
            return scale
        
        return 1.0
    
    def _compute_adaptive_trust_radius(self, gradient_norm):
        """
        Compute an adaptive trust radius based on gradient magnitude.
        
        Near convergence (small gradient), the trust radius should be
        proportional to the gradient to prevent overshooting.
        
        Parameters:
        gradient_norm : float
            Norm of the current gradient
            
        Returns:
        float
            Suggested trust radius
        """
        if not self.adaptive_trust_enabled:
            return self.trust_radius
        
        if gradient_norm < self.gradient_scaling_threshold:
            # Near convergence: scale trust radius with gradient
            # Use a linear relationship with a minimum floor
            adaptive_radius = self.gradient_trust_coupling * gradient_norm / self.gradient_scaling_threshold * self.trust_radius_max
            adaptive_radius = max(adaptive_radius, self.trust_radius_min)
            adaptive_radius = min(adaptive_radius, self.trust_radius)
            
            if adaptive_radius < self.trust_radius * 0.9:  # Only log if significant change
                self.log(f"Adaptive trust radius: {self.trust_radius:.6f} -> {adaptive_radius:.6f} "
                        f"(gradient_norm={gradient_norm:.6e})")
            
            return adaptive_radius
        
        return self.trust_radius
        
    def compute_reduction_ratio(self, gradient, hessian, step, actual_reduction):
        """
        Compute ratio between actual and predicted reduction in energy.
        
        Parameters:
        gradient : numpy.ndarray
            Current gradient
        hessian : numpy.ndarray
            Current approximate Hessian
        step : numpy.ndarray
            Step vector
        actual_reduction : float
            Actual energy reduction (previous_energy - current_energy)
            
        Returns:
        float
            Ratio of actual to predicted reduction
        """
        g_flat = gradient.flatten()
        step_flat = step.flatten()
        
        linear_term = np.dot(g_flat, step_flat)
        quadratic_term = 0.5 * np.dot(step_flat, np.dot(hessian, step_flat))
        predicted_reduction = -(linear_term + quadratic_term)
        
        if abs(predicted_reduction) < 1e-14:
            self.log("Warning: Predicted reduction is near zero")
            return 1.0 if abs(actual_reduction) < 1e-14 else 0.0
            
        ratio = actual_reduction / predicted_reduction
        
        if not np.isfinite(ratio):
            self.log("Warning: Non-finite reduction ratio, using 0.0")
            return 0.0
            
        self.log(f"Reduction ratio: actual={actual_reduction:.6e}, "
                f"predicted={predicted_reduction:.6e}, ratio={ratio:.4f}")
        
        return ratio
        
    def adjust_trust_radius(self, ratio, step_norm, at_boundary):
        """
        Dynamically adjust the trust radius based on reduction ratio.
        Uses Nocedal & Wright's trust region update strategy.
        
        Parameters:
        ratio : float
            Reduction ratio (actual/predicted)
        step_norm : float
            Norm of the current step
        at_boundary : bool
            Whether the step is at the trust region boundary
        """
        if not self.adapt_trust_radius:
            return
            
        old_trust_radius = self.trust_radius
        self.trust_radius_history.append(old_trust_radius)
        
        quality_metric = {
            'iteration': self.iter,
            'ratio': ratio,
            'step_norm': step_norm,
            'at_boundary': at_boundary,
            'trust_radius': old_trust_radius
        }
        self.step_quality_history.append(quality_metric)
        
        if ratio < self.eta_2:
            self.trust_radius = max(self.gamma_1 * step_norm, self.trust_radius_min)
            self.log(f"Poor step quality (ratio={ratio:.3f} < {self.eta_2}), "
                    f"shrinking trust radius: {old_trust_radius:.6f} -> {self.trust_radius:.6f}")
        elif ratio > self.eta_3 and at_boundary:
            self.trust_radius = min(self.gamma_2 * self.trust_radius, self.trust_radius_max)
            self.log(f"Good step quality (ratio={ratio:.3f} > {self.eta_3}) at boundary, "
                    f"expanding trust radius: {old_trust_radius:.6f} -> {self.trust_radius:.6f}")
        else:
            self.log(f"Acceptable step quality (ratio={ratio:.3f}), "
                    f"keeping trust radius at {self.trust_radius:.6f}")

    def _solve_alpha_micro_cycles(self, eigvals, gradient_trans, max_indices, min_indices, gradient_norm):
        """
        Solve for alpha using improved micro-cycle iteration with 
        Levenberg-Marquardt style damping and backtracking.
        
        Parameters:
        eigvals : numpy.ndarray
            Eigenvalues of the Hessian
        gradient_trans : numpy.ndarray
            Gradient transformed to eigenvector basis
        max_indices : list
            Indices for maximization subspace
        min_indices : list
            Indices for minimization subspace
        gradient_norm : float
            Norm of the original gradient (for adaptive scaling)
            
        Returns:
        tuple
            (step, step_norm, converged)
        """
        alpha = self.alpha0
        best_step = None
        best_step_norm_diff = float('inf')
        step_norm_history = []
        
        # Compute adaptive trust radius based on gradient
        effective_trust_radius = self._compute_adaptive_trust_radius(gradient_norm)
        
        for mu in range(self.max_micro_cycles):
            self.log(f"  Micro cycle {mu:02d}: alpha={alpha:.6e}, trust_radius={effective_trust_radius:.6f}")
            
            try:
                step = np.zeros_like(gradient_trans)
                
                # Maximization subspace
                step_max = np.array([])
                eigval_max = 0.0
                if len(max_indices) > 0:
                    H_aug_max = self.get_augmented_hessian(
                        eigvals[max_indices], gradient_trans[max_indices], alpha
                    )
                    step_max, eigval_max, nu_max, eigvec_max = self.solve_rfo(
                        H_aug_max, "max", prev_eigvec=self.prev_eigvec_max
                    )
                    self.prev_eigvec_max = eigvec_max
                    step[max_indices] = step_max
                
                # Minimization subspace
                step_min = np.array([])
                eigval_min = 0.0
                if len(min_indices) > 0:
                    H_aug_min = self.get_augmented_hessian(
                        eigvals[min_indices], gradient_trans[min_indices], alpha
                    )
                    step_min, eigval_min, nu_min, eigvec_min = self.solve_rfo(
                        H_aug_min, "min", prev_eigvec=self.prev_eigvec_min
                    )
                    self.prev_eigvec_min = eigvec_min
                    step[min_indices] = step_min
                
                step_norm = np.linalg.norm(step)
                step_norm_history.append(step_norm)
                
                step_max_norm = np.linalg.norm(step_max) if len(max_indices) > 0 else 0.0
                step_min_norm = np.linalg.norm(step_min) if len(min_indices) > 0 else 0.0
                
                if self.debug:
                    self.log(f"    |step_max|={step_max_norm:.6f}, |step_min|={step_min_norm:.6f}, |step|={step_norm:.6f}")
                
                # Track best step
                norm_diff = abs(step_norm - effective_trust_radius)
                if norm_diff < best_step_norm_diff:
                    best_step = step.copy()
                    best_step_norm_diff = norm_diff
                
                # Check convergence
                if step_norm <= effective_trust_radius:
                    self.log(f"  Step satisfies trust radius (|step|={step_norm:.6f} <= {effective_trust_radius:.6f})")
                    break
                
                # Check relative convergence
                if step_norm > 0 and norm_diff / step_norm < self.micro_cycle_rtol:
                    self.log(f"  Micro-cycle converged (relative diff={norm_diff/step_norm:.6e})")
                    step = step * (effective_trust_radius / step_norm)
                    return step, effective_trust_radius, True
                
                # Check for stagnation
                if len(step_norm_history) >= 3:
                    recent_changes = [abs(step_norm_history[-i] - step_norm_history[-(i+1)]) 
                                     for i in range(1, 3)]
                    if all(c < self.micro_cycle_atol for c in recent_changes):
                        self.log(f"  Micro-cycle stagnated, using best step")
                        if best_step is not None:
                            best_norm = np.linalg.norm(best_step)
                            if best_norm > effective_trust_radius:
                                best_step = best_step * (effective_trust_radius / best_norm)
                            return best_step, min(best_norm, effective_trust_radius), True
                
                # Calculate alpha update
                alpha_step = self._compute_alpha_step(
                    alpha, eigval_max, eigval_min, 
                    step_max_norm, step_min_norm, step_norm,
                    eigvals, gradient_trans, max_indices, min_indices,
                    effective_trust_radius
                )
                
                # Apply damping for stability
                damping = 1.0
                if abs(alpha_step) > self.alpha_step_max:
                    damping = self.alpha_step_max / abs(alpha_step)
                    alpha_step *= damping
                    self.log(f"    Damped alpha_step by factor {damping:.4f}")
                
                # Update alpha with bounds
                old_alpha = alpha
                alpha = np.clip(alpha + alpha_step, self.alpha_min, self.alpha_max)
                
                if self.debug:
                    self.log(f"    alpha: {old_alpha:.6e} -> {alpha:.6e} (step={alpha_step:.6e})")
                
                # Check if alpha hit bounds
                if alpha == self.alpha_max or alpha == self.alpha_min:
                    self.log(f"  Alpha reached bounds, using best step")
                    if best_step is not None:
                        best_norm = np.linalg.norm(best_step)
                        if best_norm > effective_trust_radius:
                            best_step = best_step * (effective_trust_radius / best_norm)
                        return best_step, min(best_norm, effective_trust_radius), True
                    break
                    
            except Exception as e:
                self.log(f"  Error in micro-cycle {mu}: {str(e)}")
                if best_step is not None:
                    best_norm = np.linalg.norm(best_step)
                    if best_norm > effective_trust_radius:
                        best_step = best_step * (effective_trust_radius / best_norm)
                    return best_step, min(best_norm, effective_trust_radius), False
                break
        
        # Micro-cycles did not converge - use best step with scaling
        self.log(f"  Micro-cycles did not converge in {self.max_micro_cycles} iterations")
        if best_step is not None:
            best_norm = np.linalg.norm(best_step)
            if best_norm > effective_trust_radius:
                best_step = best_step * (effective_trust_radius / best_norm)
            return best_step, min(best_norm, effective_trust_radius), False
        
        return np.zeros_like(gradient_trans), 0.0, False
    
    def _compute_alpha_step(self, alpha, eigval_max, eigval_min, 
                           step_max_norm, step_min_norm, step_norm,
                           eigvals, gradient_trans, max_indices, min_indices,
                           target_trust_radius):
        """
        Compute the alpha step using Newton-Raphson with safeguards.
        
        Returns:
        float
            Computed alpha step
        """
        eps = 1e-12
        
        dstep2_dalpha_max = 0.0
        if len(max_indices) > 0 and step_max_norm > eps:
            denom_max = eigvals[max_indices] - eigval_max * alpha
            safe_denom = np.where(np.abs(denom_max) < eps, 
                                  np.sign(denom_max) * eps, denom_max)
            g_max = gradient_trans[max_indices]
            
            step_factor = 1.0 + step_max_norm**2 * alpha
            if abs(step_factor) > eps:
                quot = np.sum(g_max**2 / safe_denom**3)
                dstep2_dalpha_max = 2.0 * eigval_max / step_factor * quot
        
        dstep2_dalpha_min = 0.0
        if len(min_indices) > 0 and step_min_norm > eps:
            denom_min = eigvals[min_indices] - eigval_min * alpha
            safe_denom = np.where(np.abs(denom_min) < eps,
                                  np.sign(denom_min) * eps, denom_min)
            g_min = gradient_trans[min_indices]
            
            step_factor = 1.0 + step_min_norm**2 * alpha
            if abs(step_factor) > eps:
                quot = np.sum(g_min**2 / safe_denom**3)
                dstep2_dalpha_min = 2.0 * eigval_min / step_factor * quot
        
        dstep2_dalpha = dstep2_dalpha_max + dstep2_dalpha_min
        
        if abs(dstep2_dalpha) < eps:
            if step_norm > target_trust_radius:
                return alpha * 0.5
            else:
                return 0.0
        
        alpha_step = (target_trust_radius**2 - step_norm**2) / dstep2_dalpha
        
        return alpha_step

    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, 
            pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """
        Execute one step of enhanced RSPRFO optimization with trust radius adjustment.
        
        Parameters:
        geom_num_list : numpy.ndarray
            Current geometry coordinates
        B_g : numpy.ndarray
            Current gradient
        pre_B_g : numpy.ndarray
            Previous gradient
        pre_geom : numpy.ndarray
            Previous geometry
        B_e : float
            Current energy
        pre_B_e : float
            Previous energy
        pre_move_vector : numpy.ndarray
            Previous step vector
        initial_geom_num_list : numpy.ndarray
            Initial geometry
        g : numpy.ndarray
            Alternative gradient representation
        pre_g : numpy.ndarray
            Previous alternative gradient representation
            
        Returns:
        numpy.ndarray
            Optimization step vector (shaped as column vector)
        """
        self.log(f"\n{'='*60}")
        self.log(f"RS-PRFO Iteration {self.iter}")
        self.log(f"{'='*60}")
        
        if self.Initialization:
            self._reset_state()
            self.Initialization = False
            self.log(f"Initialized with trust radius {self.trust_radius:.6f}")
        else:
            step_accepted = self._process_previous_step(
                B_e, geom_num_list, B_g, pre_B_g, pre_geom, pre_move_vector
            )
            
            if not step_accepted and self.step_rejection_enabled:
                self.log("Step rejected - optimizer should use previous geometry")
                
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        if (self.prev_geometry is not None and self.prev_gradient is not None and 
            len(pre_B_g) > 0 and len(pre_geom) > 0):
            self.update_hessian(geom_num_list, B_g, pre_geom, pre_B_g)
        
        gradient = np.asarray(B_g).flatten()
        
        # Project out TR/ROT from gradient
        raw_norm = np.linalg.norm(gradient)
        gradient = self._project_grad_tr_rot(gradient, geom_num_list)
        proj_norm = np.linalg.norm(gradient)
        
        if abs(raw_norm - proj_norm) > 1e-10:
            self.log(f"Gradient TR/ROT projection: {raw_norm:.6e} -> {proj_norm:.6e}")
        
        gradient_norm = proj_norm  # Store for later use
        
        # Prepare Hessian
        H = self.hessian + self.bias_hessian if self.bias_hessian is not None else self.hessian
        
        # Compute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # === [CRITICAL FIX] Handle NaN/Inf in Hessian ===
        
        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            self.log("CRITICAL ERROR: Hessian eigendecomposition failed (NaNs detected).", force=True)
            self.log("Resetting to Identity Hessian to force Steepest Descent fallback.", force=True)
        
            eigvals = np.ones_like(eigvals)
            eigvecs = np.eye(len(eigvals))
        # =================================================
        
        # Apply eigenvalue shifting if needed
        H, eigvals, shifted = self._shift_hessian_eigenvalues(H, eigvals, eigvecs)
        if shifted:
            eigvals, eigvecs = np.linalg.eigh(H)
        
        self.shifted_hessian = H
        
        # Log eigenvalue information
        neg_eigval_count = np.sum(eigvals < -1e-8)
        self.log(f"Eigenvalue analysis: {neg_eigval_count} negative (target: {self.saddle_order})")
        self.log(f"Lowest eigenvalues: {eigvals[:min(5, len(eigvals))]}")
        
        # Mode selection with mode following
        max_indices, min_indices = self._select_modes(eigvals, eigvecs, gradient)
        
        # Store eigenvector history
        self.eigvec_history.append(eigvecs.copy())
        self.eigval_history.append(eigvals.copy())
        if len(self.eigvec_history) > 5:
            self.eigvec_history.pop(0)
            self.eigval_history.pop(0)
        
        # Transform gradient to eigenvector space
        gradient_trans = eigvecs.T @ gradient
        
        # Solve for step using micro-cycles (now with gradient_norm)
        step_trans, step_norm, converged = self._solve_alpha_micro_cycles(
            eigvals, gradient_trans, max_indices, min_indices, gradient_norm
        )
        # === [ADDED START] Safety check for NaN/Inf steps ===
        if not np.isfinite(step_norm) or not np.all(np.isfinite(step_trans)):
            self.log("CRITICAL WARNING: NaN detected in optimization step. Falling back to Steepest Descent.", force=True)
            
            # Fallback: Steepest Descent (SD) step within trust radius
            # In eigenvector basis, SD direction is simply -gradient
            sd_step = -gradient_trans
            sd_norm = np.linalg.norm(sd_step)
            
            # Apply trust radius
            target_norm = min(sd_norm, self.trust_radius)
            
            if sd_norm > 1e-12:
                step_trans = sd_step * (target_norm / sd_norm)
                step_norm = target_norm
            else:
                step_trans = np.zeros_like(gradient_trans)
                step_norm = 0.0
                
            converged = False
        # === [ADDED END] ===
        
        if not converged:
            self.log("Warning: Micro-cycles did not fully converge")
        
        # Transform step back to original coordinates
        move_vector = eigvecs @ step_trans
        step_norm = np.linalg.norm(move_vector)
        
        # Apply gradient-based scaling for near-convergence
        grad_scale = self._compute_gradient_based_scale(gradient_norm, step_norm)
        if grad_scale < 1.0:
            move_vector = move_vector * grad_scale
            step_norm = step_norm * grad_scale
            self.log(f"Applied gradient-based scaling: {1.0/grad_scale:.2f}x reduction")
        
        # Apply trust radius constraint
        effective_trust = self._compute_adaptive_trust_radius(gradient_norm)
        if step_norm > effective_trust * 1.01:
            self.log(f"Scaling step from {step_norm:.6f} to trust radius {effective_trust:.6f}")
            move_vector = move_vector * (effective_trust / step_norm)
            step_norm = effective_trust
        
        # Apply maxstep constraint if specified
        if self.config.get("maxstep") is not None:
            move_vector, step_norm = self._apply_maxstep_constraint(move_vector)
        
        self.log(f"Final step norm: {step_norm:.6f}")
        
        # Calculate predicted energy change
        predicted_energy_change = self.rfo_model(gradient, H, move_vector)
        self.predicted_energy_changes.append(predicted_energy_change)
        self.log(f"Predicted energy change: {predicted_energy_change:.6e}")
        
        # Store state for next iteration
        self.prev_geometry = np.copy(geom_num_list)
        self.prev_gradient = np.copy(B_g)
        self.prev_energy = B_e
        self.prev_move_vector = np.copy(move_vector)
        
        self.iter += 1
        
        return move_vector.reshape(-1, 1)
    
    def _reset_state(self):
        """Reset optimizer state for a new optimization run."""
        self.prev_eigvec_max = None
        self.prev_eigvec_min = None
        self.predicted_energy_changes = []
        self.actual_energy_changes = []
        self.reduction_ratios = []
        self.trust_radius_history = []
        self.step_quality_history = []
        self.prev_geometry = None
        self.prev_gradient = None
        self.prev_energy = None
        self.prev_move_vector = None
        self.eigvec_history = []
        self.eigval_history = []
        self.ts_mode_idx = None
        self.ts_mode_eigvec = None
        self.consecutive_rejections = 0
        self.trust_radius = self.trust_radius_initial
    
    def _process_previous_step(self, B_e, geom_num_list, B_g, pre_B_g, pre_geom, pre_move_vector):
        """
        Process results from the previous step and adjust trust radius.
        
        Returns:
        bool
            True if step is accepted, False if rejected
        """
        if self.prev_energy is None or len(self.predicted_energy_changes) == 0:
            return True
        
        actual_energy_change = B_e - self.prev_energy
        predicted_energy_change = self.predicted_energy_changes[-1]
        self.actual_energy_changes.append(actual_energy_change)
        
        if len(pre_move_vector) > 0:
            prev_step_norm = np.linalg.norm(np.asarray(pre_move_vector).flatten())
        elif self.prev_move_vector is not None:
            prev_step_norm = np.linalg.norm(self.prev_move_vector.flatten())
        else:
            prev_step_norm = 0.0
        
        self.log(f"Energy: {self.prev_energy:.8f} -> {B_e:.8f}")
        self.log(f"Actual change: {actual_energy_change:.6e}, Predicted: {predicted_energy_change:.6e}")
        
        H = self.hessian + self.bias_hessian if self.bias_hessian is not None else self.hessian
        
        if hasattr(Calculationtools, 'project_out_hess_tr_and_rot_for_coord'):
            H = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                H, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), 
                display_eigval=False
            )
        
        ratio = self.compute_reduction_ratio(
            self.prev_gradient, H, self.prev_move_vector, actual_energy_change
        )
        self.reduction_ratios.append(ratio)
        
        at_boundary = prev_step_norm >= self.trust_radius * 0.95
        
        self.adjust_trust_radius(ratio, prev_step_norm, at_boundary)
        
        if self.step_rejection_enabled and ratio < self.rejection_threshold:
            self.consecutive_rejections += 1
            self.log(f"Step quality very poor (ratio={ratio:.4f}), rejection count: {self.consecutive_rejections}")
            
            if self.consecutive_rejections >= self.max_consecutive_rejections:
                self.log(f"Too many consecutive rejections, accepting step anyway")
                self.consecutive_rejections = 0
                return True
            
            return False
        
        self.consecutive_rejections = 0
        return True
    
    def _select_modes(self, eigvals, eigvecs, gradient):
        """
        Select modes for maximization and minimization subspaces.
        
        Returns:
        tuple
            (max_indices, min_indices)
        """
        n = len(eigvals)
        
        if self.saddle_order == 0:
            return [], list(range(n))
        
        if self.mode_following_enabled:
            max_indices = self._find_ts_modes(eigvals, eigvecs, gradient)
        else:
            sorted_indices = np.argsort(eigvals)
            max_indices = sorted_indices[:self.saddle_order].tolist()
        
        min_indices = [i for i in range(n) if i not in max_indices]
        
        return max_indices, min_indices
    
    def _find_ts_modes(self, eigvals, eigvecs, gradient):
        """
        Find transition state modes using mode following.
        
        Returns:
        list
            Indices of modes to maximize
        """
        sorted_indices = np.argsort(eigvals)
        
        if self.ts_mode_idx is None or self.ts_mode_eigvec is None:
            self.ts_mode_idx = sorted_indices[0]
            self.ts_mode_eigvec = eigvecs[:, self.ts_mode_idx].copy()
            self.log(f"Initial TS mode: {self.ts_mode_idx}, eigenvalue={eigvals[self.ts_mode_idx]:.6f}")
            return sorted_indices[:self.saddle_order].tolist()
        
        overlaps = np.abs(eigvecs.T @ self.ts_mode_eigvec)
        
        best_idx = np.argmax(overlaps)
        best_overlap = overlaps[best_idx]
        
        self.log(f"Mode following: best overlap={best_overlap:.4f} with mode {best_idx} "
                f"(eigenvalue={eigvals[best_idx]:.6f})")
        
        if best_overlap > self.overlap_threshold:
            self.ts_mode_idx = best_idx
            self.ts_mode_eigvec = eigvecs[:, best_idx].copy()
            
            if np.dot(eigvecs[:, best_idx], self.ts_mode_eigvec) < 0:
                self.ts_mode_eigvec *= -1
            
            max_indices = [best_idx]
            
            if self.saddle_order > 1:
                remaining = [i for i in sorted_indices if i != best_idx]
                max_indices.extend(remaining[:self.saddle_order - 1])
            
            return max_indices
        else:
            self.log(f"Warning: Poor mode overlap ({best_overlap:.4f}), possible mode crossing")
            
            if self.eigvec_following:
                return self._handle_mode_mixing(eigvals, eigvecs, overlaps, sorted_indices)
            
            self.ts_mode_idx = sorted_indices[0]
            self.ts_mode_eigvec = eigvecs[:, sorted_indices[0]].copy()
            return sorted_indices[:self.saddle_order].tolist()
    
    def _handle_mode_mixing(self, eigvals, eigvecs, overlaps, sorted_indices):
        """
        Handle mode mixing when mode overlap is poor.
        
        Returns:
        list
            Selected mode indices
        """
        significant_overlaps = np.where(overlaps > self.mixing_threshold)[0]
        
        if len(significant_overlaps) == 0:
            self.log("No significant mode overlap - resetting mode tracking")
            self.ts_mode_idx = sorted_indices[0]
            self.ts_mode_eigvec = eigvecs[:, sorted_indices[0]].copy()
            return sorted_indices[:self.saddle_order].tolist()
        
        weights = []
        for idx in significant_overlaps:
            overlap_weight = overlaps[idx]**2
            eigval_weight = 1.0 if eigvals[idx] < 0 else 0.1
            weights.append(overlap_weight * eigval_weight)
        
        best_local_idx = np.argmax(weights)
        best_idx = significant_overlaps[best_local_idx]
        
        self.log(f"Mode mixing resolution: selected mode {best_idx} "
                f"(overlap={overlaps[best_idx]:.4f}, eigenvalue={eigvals[best_idx]:.6f})")
        
        self.ts_mode_idx = best_idx
        self.ts_mode_eigvec = eigvecs[:, best_idx].copy()
        
        max_indices = [best_idx]
        if self.saddle_order > 1:
            remaining = [i for i in sorted_indices if i != best_idx]
            max_indices.extend(remaining[:self.saddle_order - 1])
        
        return max_indices
    
    def _apply_maxstep_constraint(self, move_vector):
        """
        Apply maximum step constraint.
        
        Returns:
        tuple
            (constrained_move_vector, step_norm)
        """
        maxstep = self.config.get("maxstep")
        
        if move_vector.size % 3 == 0 and move_vector.size > 3:
            move_reshaped = move_vector.reshape(-1, 3)
            step_lengths = np.sqrt(np.sum(move_reshaped**2, axis=1))
            longest_step = np.max(step_lengths)
        else:
            longest_step = np.linalg.norm(move_vector)
        
        if longest_step > maxstep:
            scale = maxstep / longest_step
            move_vector = move_vector * scale
            self.log(f"Step constrained by maxstep: {longest_step:.6f} -> {maxstep:.6f}")
        
        return move_vector, np.linalg.norm(move_vector)

    def get_augmented_hessian(self, eigenvalues, gradient_components, alpha):
        """
        Create the augmented hessian matrix for RFO calculation.
        
        Parameters:
        eigenvalues : numpy.ndarray
            Eigenvalues for the selected subspace
        gradient_components : numpy.ndarray
            Gradient components in the selected subspace
        alpha : float
            Alpha parameter for RS-RFO
            
        Returns:
        numpy.ndarray
            Augmented Hessian matrix for RFO calculation
        """
        n = len(eigenvalues)
        H_aug = np.zeros((n + 1, n + 1))
        
        np.fill_diagonal(H_aug[:n, :n], eigenvalues / alpha)
        
        gradient_components = np.asarray(gradient_components).flatten()
        
        H_aug[:n, n] = gradient_components / alpha
        H_aug[n, :n] = gradient_components / alpha
        
        return H_aug
    
    def solve_rfo(self, H_aug, mode="min", prev_eigvec=None):
        """
        Solve the RFO equations to get the step.
        
        Parameters:
        H_aug : numpy.ndarray
            Augmented Hessian matrix
        mode : str
            "min" for energy minimization, "max" for maximization
        prev_eigvec : numpy.ndarray
            Previous eigenvector for consistent direction
            
        Returns:
        tuple
            (step, eigenvalue, nu parameter, eigenvector)
        """
        eigvals, eigvecs = np.linalg.eigh(H_aug)
        
        if mode == "min":
            idx = np.argmin(eigvals)
        else:
            idx = np.argmax(eigvals)
        
        if prev_eigvec is not None:
            try:
                if prev_eigvec.shape == eigvecs[:, idx].shape:
                    overlap = np.dot(eigvecs[:, idx], prev_eigvec)
                    if overlap < 0:
                        eigvecs[:, idx] *= -1
            except Exception:
                pass
        
        eigval = eigvals[idx]
        eigvec = eigvecs[:, idx]
        
        nu = eigvec[-1]
        
        if abs(nu) < 1e-12:
            self.log(f"Warning: Very small nu={nu:.2e}, using safe value")
            nu = np.sign(nu) * 1e-12 if nu != 0 else 1e-12
        
        step = -eigvec[:-1] / nu
        
        return step, eigval, nu, eigvec
    
    def rfo_model(self, gradient, hessian, step):
        """
        Estimate energy change based on RFO model.
        
        Parameters:
        gradient : numpy.ndarray
            Energy gradient
        hessian : numpy.ndarray
            Hessian matrix
        step : numpy.ndarray
            Step vector
            
        Returns:
        float
            Predicted energy change
        """
        g = gradient.flatten()
        s = step.flatten()
        return np.dot(g, s) + 0.5 * np.dot(s, hessian @ s)
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """
        Update the Hessian using the specified update method with curvature checks.
        
        Parameters:
        current_geom : numpy.ndarray
            Current geometry
        current_grad : numpy.ndarray
            Current gradient
        previous_geom : numpy.ndarray
            Previous geometry
        previous_grad : numpy.ndarray
            Previous gradient
        """
        displacement = np.asarray(current_geom - previous_geom).reshape(-1, 1)
        delta_grad = np.asarray(current_grad - previous_grad).reshape(-1, 1)
        
        disp_norm = np.linalg.norm(displacement)
        grad_diff_norm = np.linalg.norm(delta_grad)
        
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update: changes too small")
            return
        
        dot_product = np.dot(displacement.T, delta_grad)[0, 0]
        
        curvature_ratio = dot_product / (disp_norm * grad_diff_norm)
        
        self.log(f"Hessian update: |disp|={disp_norm:.6f}, |dgrad|={grad_diff_norm:.6f}, "
                f"dot={dot_product:.6f}, curvature_ratio={curvature_ratio:.4f}")
        
        if abs(curvature_ratio) < 0.01:
            self.log("Warning: Very poor displacement-gradient alignment, proceeding with caution")
        
        method_key_lower = self.hessian_update_method.lower()
        method_name, update_function = self.default_update_method
        
        for key, name, func in self.updater_dispatch_list:
            if key in method_key_lower:
                method_name = name
                update_function = func
                break
        
        self.log(f"Using Hessian update method: {method_name}")
        
        try:
            old_hessian = self.hessian.copy()
            
            delta_hess = update_function(self.hessian, displacement, delta_grad)
            new_hessian = self.hessian + delta_hess
            new_hessian = 0.5 * (new_hessian + new_hessian.T)
            
            new_eigvals = np.linalg.eigvalsh(new_hessian)
            
            n_neg = np.sum(new_eigvals < -1e-8)
            max_eigval = np.max(np.abs(new_eigvals))
            
            if max_eigval > 1e6:
                self.log(f"Warning: Updated Hessian has very large eigenvalues ({max_eigval:.2e}), "
                        "reverting to previous Hessian")
                return
            
            if self.saddle_order > 0 and n_neg == 0:
                self.log(f"Warning: No negative eigenvalues after update (expected {self.saddle_order})")
            
            self.hessian = new_hessian
            self.log(f"Hessian updated successfully ({n_neg} negative eigenvalues)")
            
        except Exception as e:
            self.log(f"Error in Hessian update: {e}")
            self.log("Keeping previous Hessian")
    
    def should_update_hessian(self, displacement, delta_grad, dot_product):
        """
        Determine whether to update the Hessian based on quality metrics.
        
        Parameters:
        displacement : numpy.ndarray
            Geometry displacement vector
        delta_grad : numpy.ndarray
            Gradient difference vector
        dot_product : float
            Dot product of displacement and gradient difference
            
        Returns:
        bool
            True if Hessian should be updated
        """
        disp_norm = np.linalg.norm(displacement)
        grad_norm = np.linalg.norm(delta_grad)
        
        if disp_norm < 1e-10 or grad_norm < 1e-10:
            return False
        
        cos_angle = dot_product / (disp_norm * grad_norm)
        
        if self.saddle_order == 0 and dot_product < 0:
            self.log(f"Skipping update: negative curvature in minimization (cos={cos_angle:.4f})")
            return False
        
        if abs(cos_angle) < 0.001:
            self.log(f"Skipping update: nearly orthogonal vectors (cos={cos_angle:.4f})")
            return False
        
        return True
    
    def log(self, message, force=False):
        """
        Print log message if display flag is enabled.
        
        Parameters:
        message : str
            Message to display
        force : bool
            If True, display message regardless of display_flag
        """
        if self.display_flag or force:
            print(message)
    
    def set_hessian(self, hessian):
        """
        Set the Hessian matrix.
        
        Parameters:
        hessian : numpy.ndarray
            Hessian matrix
        """
        self.hessian = np.asarray(hessian).copy()
        self.hessian = 0.5 * (self.hessian + self.hessian.T)  # Ensure symmetry

    def set_bias_hessian(self, bias_hessian):
        """
        Set the bias Hessian matrix.
        
        Parameters:
        bias_hessian : numpy.ndarray
            Bias Hessian matrix
        """
        self.bias_hessian = np.asarray(bias_hessian).copy()
        self.bias_hessian = 0.5 * (self.bias_hessian + self.bias_hessian.T)
    
    def get_hessian(self):
        """
        Get the current Hessian matrix.
        
        Returns:
        numpy.ndarray
            Hessian matrix
        """
        return self.hessian
    
    def get_bias_hessian(self):
        """
        Get the current bias Hessian matrix.
        
        Returns:
        numpy.ndarray
            Bias Hessian matrix
        """
        return self.bias_hessian
    
    def get_shifted_hessian(self):
        """
        Get the eigenvalue-shifted Hessian matrix.
        
        Returns:
        numpy.ndarray
            Shifted Hessian matrix (or None if not computed)
        """
        return self.shifted_hessian
        
    def reset_trust_radius(self):
        self.trust_radius = self.trust_radius_initial
        self.log(f"Trust radius reset to initial value: {self.trust_radius:.6f}", force=True)