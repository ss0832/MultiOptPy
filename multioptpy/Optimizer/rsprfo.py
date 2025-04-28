import numpy as np
from numpy.linalg import norm
import copy
from .hessian_update import ModelHessianUpdate

class RSPRFO:
    def __init__(self, **config):
        # Rational Step P-RFO (Rational Function Optimization) for transition state searches
        # References:
        # [1] Banerjee et al., Phys. Chem., 89, 52-57 (1985)
        # [2] Heyden et al., J. Chem. Phys., 123, 224101 (2005)
        # [3] Baker, J. Comput. Chem., 7, 385-395 (1986)
        # [4] Besalú and Bofill, Theor. Chem. Acc., 100, 265-274 (1998)
        
        
        
        
        # Initial alpha parameter for RS-RFO
        self.alpha0 = config.get("alpha0", 1.0)
        # Maximum number of micro-cycles
        self.max_micro_cycles = config.get("max_micro_cycles", 1)
        # Saddle order (0=minimum, 1=first-order saddle/TS, 2=second-order saddle, etc.)
        self.saddle_order = config.get("saddle_order", 1)
        # Hessian update method ('BFGS', 'SR1', 'FSB', 'Bofill', 'PSB', 'MSP', 'auto')
        self.hessian_update_method = config.get("method", "auto")
        
        self.display_flag = config.get("display_flag", True)
        self.config = config
        self.Initialization = True
        
        self.hessian = None
        self.bias_hessian = None
        
        # For tracking optimization
        self.prev_eigvec_max = None
        self.prev_eigvec_min = None
        self.predicted_energy_changes = []
        self.prev_geometry = None
        self.prev_gradient = None
        
        # Define which mode(s) to maximize along based on saddle order
        self.roots = list(range(self.saddle_order))
        
        # Trust radius for step restriction
        if self.saddle_order == 0:
            self.trust_radius = config.get("trust_radius", 0.5)
        else:
            self.trust_radius = config.get("trust_radius", 0.1)    
        # Initialize the hessian update module
        self.hessian_updater = ModelHessianUpdate()
        return
        
    def log(self, message):
        if self.display_flag:
            print(message)
            
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        if self.Initialization:
            self.prev_eigvec_max = None
            self.prev_eigvec_min = None
            self.predicted_energy_changes = []
            self.prev_geometry = None
            self.prev_gradient = None
            self.Initialization = False
            
        # If hessian isn't set, we can't perform the optimization
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set before running optimization")
        
        # Update Hessian if we have previous geometry and gradient information
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_B_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, B_g, pre_geom, pre_B_g)
            
        # Ensure gradient is properly shaped as a 1D array
        gradient = np.asarray(B_g).flatten()
        H = self.hessian + self.bias_hessian
            
        # Compute eigenvalues and eigenvectors of the hessian
        eigvals, eigvecs = np.linalg.eigh(H)
        
       
        # Transform gradient to eigensystem of hessian and ensure it's a 1D array
        gradient_trans = eigvecs.T.dot(gradient).flatten()
        
        # Special case for saddle_order=0
        if self.saddle_order == 0:
            min_indices = list(range(len(gradient_trans)))
            max_indices = []
        else:
            # Minimize energy along all modes, except those specified by saddle_order
            min_indices = [i for i in range(gradient_trans.size) if i not in self.roots]
            # Maximize energy along the modes determined by saddle_order
            max_indices = self.roots
        
        alpha = self.alpha0
        step = np.zeros_like(gradient_trans)  # This ensures step is 1D
        
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-PRFO micro cycle {mu:02d}, alpha={alpha:.6f}")
            
            # Only process max_indices if there are any
            if len(max_indices) > 0:
                # Maximize energy along the chosen saddle direction(s)
                H_aug_max = self.get_augmented_hessian(
                    eigvals[max_indices], gradient_trans[max_indices], alpha
                )
                step_max, eigval_max, nu_max, self.prev_eigvec_max = self.solve_rfo(
                    H_aug_max, "max", prev_eigvec=self.prev_eigvec_max
                )
                step[max_indices] = step_max
            else:
                eigval_max = 0
                step_max = np.array([])
            
            # Only process min_indices if there are any
            if len(min_indices) > 0:
                # Minimize energy along all other modes
                H_aug_min = self.get_augmented_hessian(
                    eigvals[min_indices], gradient_trans[min_indices], alpha
                )
                step_min, eigval_min, nu_min, self.prev_eigvec_min = self.solve_rfo(
                    H_aug_min, "min", prev_eigvec=self.prev_eigvec_min
                )
                step[min_indices] = step_min
            else:
                eigval_min = 0
                step_min = np.array([])
            
            # Calculate step norm
            step_norm = np.linalg.norm(step)
            
            if len(max_indices) > 0:
                self.log(f"norm(step_max)={np.linalg.norm(step_max):.6f}")
            if len(min_indices) > 0:
                self.log(f"norm(step_min)={np.linalg.norm(step_min):.6f}")
                
            self.log(f"norm(step)={step_norm:.6f}")
            
            # Check if step is within trust radius
            inside_trust = step_norm <= self.trust_radius
            if inside_trust:
                self.log(f"Restricted step satisfies trust radius of {self.trust_radius:.6f}")
                self.log(f"Micro-cycles converged in cycle {mu:02d} with alpha={alpha:.6f}!")
                break
                
            # If step is too large, adjust alpha to restrict step size
            # Calculate derivatives only if the corresponding subspaces exist
            dstep2_dalpha_max = 0
            if len(max_indices) > 0:
                dstep2_dalpha_max = (
                    2
                    * eigval_max
                    / (1 + np.dot(step_max, step_max) * alpha)
                    * np.sum(
                        gradient_trans[max_indices] ** 2
                        / (eigvals[max_indices] - eigval_max * alpha) ** 3
                    )
                )
                
            dstep2_dalpha_min = 0
            if len(min_indices) > 0:
                dstep2_dalpha_min = (
                    2
                    * eigval_min
                    / (1 + np.dot(step_min, step_min) * alpha)
                    * np.sum(
                        gradient_trans[min_indices] ** 2
                        / (eigvals[min_indices] - eigval_min * alpha) ** 3
                    )
                )
                
            dstep2_dalpha = dstep2_dalpha_max + dstep2_dalpha_min
            
            # Avoid division by zero
            if abs(dstep2_dalpha) < 1e-10:
                alpha *= 1.5  # Simple heuristic if derivative is very small
            else:
                # Update alpha
                alpha_step = (
                    2 * (self.trust_radius * step_norm - step_norm**2) / dstep2_dalpha
                )
                alpha += alpha_step
            
        # Transform step back to original coordinates
        move_vector = eigvecs.dot(step)
        step_norm = np.linalg.norm(move_vector)
        
        # Scale step if necessary
        if step_norm > self.trust_radius:
            move_vector = move_vector / step_norm * self.trust_radius
            
        self.log(f"Final norm(step)={np.linalg.norm(move_vector):.6f}")
        self.log("")
        
        # Calculate predicted energy change
        predicted_energy_change = self.rfo_model(gradient, H, move_vector)
        self.predicted_energy_changes.append(predicted_energy_change)
        
        # Store current geometry and gradient for the next Hessian update
        self.prev_geometry = copy.deepcopy(geom_num_list)
        self.prev_gradient = copy.deepcopy(B_g)
        
        return move_vector.reshape(-1, 1)
    
    def update_hessian(self, current_geom, current_grad, previous_geom, previous_grad):
        """Update the Hessian using the specified update method"""
        # Calculate displacement and gradient difference
        displacement = np.array(current_geom - previous_geom).reshape(-1, 1)
        delta_grad = np.array(current_grad - previous_grad).reshape(-1, 1)
        
        # Skip update if changes are too small
        disp_norm = np.linalg.norm(displacement)
        grad_diff_norm = np.linalg.norm(delta_grad)
        
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update due to small changes")
            return
            
        # Check if displacement and gradient difference are sufficiently aligned
        dot_product = np.dot(displacement.T, delta_grad)[0, 0]
        if dot_product <= 0:
            self.log("Skipping Hessian update due to poor alignment")
            return
        print(f"Hessian update: displacement norm={disp_norm:.6f}, gradient diff norm={grad_diff_norm:.6f}, dot product={dot_product:.6f}")
        # Apply the selected Hessian update method
        if "flowchart" in self.hessian_update_method.lower():
            print(f"Hessian update method: flowchart")            
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad, "auto"
            )
        elif "bfgs" in self.hessian_update_method.lower():
            print(f"Hessian update method: bfgs")            
            delta_hess = self.hessian_updater.BFGS_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "sr1" in self.hessian_update_method.lower():
            print(f"Hessian update method: sr1")            
            delta_hess = self.hessian_updater.SR1_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "fsb" in self.hessian_update_method.lower():
            print(f"Hessian update method: fsb")            
            delta_hess = self.hessian_updater.FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "bofill" in self.hessian_update_method.lower():
            print(f"Hessian update method: bofill")
            delta_hess = self.hessian_updater.Bofill_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "psb" in self.hessian_update_method.lower():
            print(f"Hessian update method: psb")            
            delta_hess = self.hessian_updater.PSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "msp" in self.hessian_update_method.lower():
            print(f"Hessian update method: msp")
            delta_hess = self.hessian_updater.MSP_hessian_update(
                self.hessian, displacement, delta_grad
            )
        else:
            print(f"Hessian update method: unknown")
            self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad
            )
            
        # Update the Hessian
        self.hessian += delta_hess
        
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        self.hessian = (self.hessian + self.hessian.T) / 2
    
    def get_augmented_hessian(self, eigenvalues, gradient_components, alpha):
        """Create the augmented hessian matrix for RFO calculation"""
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
        """Solve the RFO equations to get the step"""
        eigvals, eigvecs = np.linalg.eigh(H_aug)
        
        if mode == "min":
            idx = np.argmin(eigvals)
        else:  # mode == "max"
            idx = np.argmax(eigvals)
            
        # Check if we need to flip the eigenvector to maintain consistency with the previous step
        if prev_eigvec is not None:
            overlap = np.dot(eigvecs[:, idx], prev_eigvec)
            if overlap < 0:
                eigvecs[:, idx] *= -1
                
        eigval = eigvals[idx]
        eigvec = eigvecs[:, idx]
        
        # The last component is nu
        nu = eigvec[-1]
        
        # The step is -p/nu where p are the first n components of the eigenvector
        step = -eigvec[:-1] / nu
        
        return step, eigval, nu, eigvec
    
    def rfo_model(self, gradient, hessian, step):
        """Estimate energy change based on RFO model"""
        return np.dot(gradient, step) + 0.5 * np.dot(step, np.dot(hessian, step))
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian



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
        """
        # Standard RSPRFO parameters
        self.alpha0 = config.get("alpha0", 1.0)
        self.max_micro_cycles = config.get("max_micro_cycles", 1)
        self.saddle_order = config.get("saddle_order", 1)
        self.hessian_update_method = config.get("method", "auto")
        self.display_flag = config.get("display_flag", True)
        self.debug = config.get("debug", False)
        
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
        
        self.log(f"Initialized EnhancedRSPRFO with trust radius={self.trust_radius:.6f}, "
                f"bounds=[{self.trust_radius_min:.6f}, {self.trust_radius_max:.6f}]")
    
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
            
        alpha = self.alpha0
        step = np.zeros_like(gradient_trans)
        
        # Micro-cycle loop for trust radius adjustment
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-PRFO micro cycle {mu:02d}, alpha={alpha:.6f}, trust radius={self.trust_radius:.6f}")
            
            # Calculate step for maximization directions (usually the TS mode)
            if len(max_indices) > 0:
                H_aug_max = self.get_augmented_hessian(
                    eigvals[max_indices], gradient_trans[max_indices], alpha
                )
                step_max, eigval_max, nu_max, self.prev_eigvec_max = self.solve_rfo(
                    H_aug_max, "max", prev_eigvec=self.prev_eigvec_max
                )
                step[max_indices] = step_max
            else:
                eigval_max = 0
                step_max = np.array([])
            
            # Calculate step for minimization directions (all other modes)
            if len(min_indices) > 0:
                H_aug_min = self.get_augmented_hessian(
                    eigvals[min_indices], gradient_trans[min_indices], alpha
                )
                step_min, eigval_min, nu_min, self.prev_eigvec_min = self.solve_rfo(
                    H_aug_min, "min", prev_eigvec=self.prev_eigvec_min
                )
                step[min_indices] = step_min
            else:
                eigval_min = 0
                step_min = np.array([])
            
            # Calculate step norm
            step_norm = norm(step)
            
            if len(max_indices) > 0:
                self.log(f"norm(step_max)={norm(step_max):.6f}")
            if len(min_indices) > 0:
                self.log(f"norm(step_min)={norm(step_min):.6f}")
                
            self.log(f"norm(step)={step_norm:.6f}")
            
            # Check if step is within trust radius
            inside_trust = step_norm <= self.trust_radius
            if inside_trust:
                self.log(f"Restricted step satisfies trust radius of {self.trust_radius:.6f}")
                self.log(f"Micro-cycles converged in cycle {mu:02d} with alpha={alpha:.6f}!")
                break
                
            # If step is too large, adjust alpha parameter
            # Calculate derivatives only if the corresponding subspaces exist
            dstep2_dalpha_max = 0
            if len(max_indices) > 0:
                dstep2_dalpha_max = (
                    2
                    * eigval_max
                    / (1 + np.dot(step_max, step_max) * alpha)
                    * np.sum(
                        gradient_trans[max_indices] ** 2
                        / (eigvals[max_indices] - eigval_max * alpha) ** 3
                    )
                )
                
            dstep2_dalpha_min = 0
            if len(min_indices) > 0:
                dstep2_dalpha_min = (
                    2
                    * eigval_min
                    / (1 + np.dot(step_min, step_min) * alpha)
                    * np.sum(
                        gradient_trans[min_indices] ** 2
                        / (eigvals[min_indices] - eigval_min * alpha) ** 3
                    )
                )
                
            dstep2_dalpha = dstep2_dalpha_max + dstep2_dalpha_min
            
            # Update alpha value for next micro-cycle
            if abs(dstep2_dalpha) < 1e-10:
                alpha *= 1.5  # Simple heuristic if derivative is very small
            else:
                # Update alpha using analytical formula
                alpha_step = (
                    2 * (self.trust_radius * step_norm - step_norm**2) / dstep2_dalpha
                )
                alpha += alpha_step
        
        # Transform step back to original coordinates
        move_vector = eigvecs.dot(step)
        step_norm = norm(move_vector)
        
        # Scale step if necessary to enforce trust radius
        if step_norm > self.trust_radius:
            move_vector = move_vector / step_norm * self.trust_radius
            self.log(f"Step scaled to trust radius: {self.trust_radius:.6f}")
            
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
        
        # Calculate predicted energy change for convergence assessment and trust radius update
        predicted_energy_change = self.rfo_model(gradient, H, move_vector)
        self.predicted_energy_changes.append(predicted_energy_change)
        self.log(f"Predicted energy change: {predicted_energy_change:.6f}")
        
        # Store current geometry, gradient and energy for next iteration
        self.prev_geometry = copy.deepcopy(geom_num_list)
        self.prev_gradient = copy.deepcopy(B_g)
        self.prev_energy = B_e
        self.prev_move_vector = copy.deepcopy(move_vector)
        
        # Increment iteration counter
        self.iter += 1
        
        return move_vector.reshape(-1, 1)
        
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
            overlap = np.dot(eigvecs[:, idx], prev_eigvec)
            if overlap < 0:
                eigvecs[:, idx] *= -1
                
        eigval = eigvals[idx]
        eigvec = eigvecs[:, idx]
        
        # The last component is nu
        nu = eigvec[-1]
        
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
        Update the Hessian using the specified update method
        
        Parameters:
        current_geom: numpy.ndarray - Current geometry
        current_grad: numpy.ndarray - Current gradient
        previous_geom: numpy.ndarray - Previous geometry
        previous_grad: numpy.ndarray - Previous gradient
        """
        # Calculate displacement and gradient difference
        displacement = np.array(current_geom - previous_geom).reshape(-1, 1)
        delta_grad = np.array(current_grad - previous_grad).reshape(-1, 1)
        
        # Skip update if changes are too small
        disp_norm = norm(displacement)
        grad_diff_norm = norm(delta_grad)
        
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update due to small changes")
            return
            
        # Check if displacement and gradient difference are sufficiently aligned
        dot_product = np.dot(displacement.T, delta_grad)[0, 0]
        
        # Skip update if dot product is negative or zero
        if dot_product <= 0:
            self.log("Skipping Hessian update due to poor alignment (negative or zero dot product)")
            return
            
        # Calculate curvature for diagnostic
        curvature = dot_product / (disp_norm**2)
        
        self.log(f"Hessian update: displacement norm={disp_norm:.6f}, gradient diff norm={grad_diff_norm:.6f}")
        self.log(f"Dot product={dot_product:.6f}, curvature={curvature:.6f}")
        
        # Apply the selected Hessian update method
        if "flowchart" in self.hessian_update_method.lower():
            self.log(f"Using flowchart-based Hessian update selection")
            delta_hess = self.hessian_updater.flowchart_hessian_update(
                self.hessian, displacement, delta_grad, "auto"
            )
        elif "bfgs" in self.hessian_update_method.lower():
            self.log(f"Using BFGS Hessian update")
            delta_hess = self.hessian_updater.BFGS_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "sr1" in self.hessian_update_method.lower():
            self.log(f"Using SR1 Hessian update")
            delta_hess = self.hessian_updater.SR1_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "fsb" in self.hessian_update_method.lower():
            self.log(f"Using FSB Hessian update")
            delta_hess = self.hessian_updater.FSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "bofill" in self.hessian_update_method.lower():
            self.log(f"Using Bofill Hessian update")
            delta_hess = self.hessian_updater.Bofill_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "psb" in self.hessian_update_method.lower():
            self.log(f"Using PSB Hessian update")
            delta_hess = self.hessian_updater.PSB_hessian_update(
                self.hessian, displacement, delta_grad
            )
        elif "msp" in self.hessian_update_method.lower():
            self.log(f"Using MSP Hessian update")
            delta_hess = self.hessian_updater.MSP_hessian_update(
                self.hessian, displacement, delta_grad
            )
        else:
            self.log(f"Unknown Hessian update method: {self.hessian_update_method}. Using auto selection.")
            # Default to Bofill for saddle points, BFGS for minimization
            if self.saddle_order > 0:
                delta_hess = self.hessian_updater.Bofill_hessian_update(
                    self.hessian, displacement, delta_grad
                )
                self.log("Auto-selected Bofill update for saddle point")
            else:
                delta_hess = self.hessian_updater.BFGS_hessian_update(
                    self.hessian, displacement, delta_grad
                )
                self.log("Auto-selected BFGS update for minimization")
            
        # Update the Hessian
        self.hessian += delta_hess
        
        # Ensure Hessian symmetry (numerical errors might cause slight asymmetry)
        self.hessian = (self.hessian + self.hessian.T) / 2
    
    def log(self, message):
        """
        Print log message if display flag is enabled
        
        Parameters:
        message: str - Message to display
        """
        if self.display_flag:
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
    
    