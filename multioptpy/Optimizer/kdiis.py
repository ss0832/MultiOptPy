import numpy as np
from scipy.linalg import solve, lstsq
import warnings


class KrylovDIIS:
    """
    Implementation of Krylov-DIIS optimization method.
    
    This method combines Krylov subspace techniques with DIIS extrapolation
    for accelerated optimization, particularly effective for large-scale
    quantum chemical calculations.
    """
    
    def __init__(self):
        # Krylov subspace parameters
        self.krylov_dimension = 5         # Dimension of the Krylov subspace
        self.krylov_restart = 10          # Number of steps before Krylov restart
        self.orthogonalization = 'mgs'    # Orthogonalization method ('mgs' or 'arnoldi')
        
        # DIIS parameters
        self.diis_max_points = 6          # Maximum number of DIIS points to use
        self.diis_min_points = 3          # Minimum points before starting DIIS
        self.diis_error_metric = 'grad'   # Error metric ('grad', 'energy_change', 'combined')
        self.diis_start_iter = 2          # Iteration to start DIIS
        
        # Numerical parameters
        self.diis_regularization = 1e-8   # Regularization for DIIS matrix
        self.krylov_tolerance = 1e-10     # Tolerance for Krylov basis
        self.error_threshold = 1.0        # Error threshold for including points
        self.weight_factor = 0.9          # Weight factor for DIIS extrapolation
        
        # Adaptation parameters
        self.dynamic_subspace = True      # Whether to adapt Krylov dimension
        self.krylov_min_dim = 3           # Minimum Krylov dimension
        self.krylov_max_dim = 10          # Maximum Krylov dimension
        self.adapt_frequency = 5          # How often to adapt Krylov dimension
        
        # Fallback and recovery
        self.max_diis_failures = 2        # Maximum consecutive DIIS failures
        self.stabilize_step = False       # Whether to stabilize with SD step
        self.recovery_steps = 2           # Number of steps in recovery mode
        
        # History storage
        self.geom_history = []            # Geometry history
        self.grad_history = []            # Gradient history
        self.energy_history = []          # Energy history
        self.error_history = []           # Error vector history
        self.krylov_basis = []            # Current Krylov basis
        self.hess_proj = None             # Projected Hessian in Krylov subspace
        
        # Performance tracking
        self.diis_failure_count = 0       # Count of DIIS failures
        self.recovery_counter = 0         # Counter for recovery steps
        self.iter = 0                     # Iteration counter
    
    def _update_histories(self, geometry, gradient, energy):
        """
        Update the optimization histories with current data.
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry
        gradient : numpy.ndarray
            Current gradient
        energy : float
            Current energy value
        """
        # Add new data to histories
        self.geom_history.append(geometry.copy())
        self.grad_history.append(gradient.copy())
        self.energy_history.append(energy)
        
        # Calculate error vector based on selected metric
        error_vec = self._calculate_error_vector(gradient)
        self.error_history.append(error_vec)
        
        # Limit history size to maximum DIIS points
        if len(self.geom_history) > self.diis_max_points:
            self.geom_history.pop(0)
            self.grad_history.pop(0)
            self.energy_history.pop(0)
            self.error_history.pop(0)
    
    def _calculate_error_vector(self, gradient):
        """
        Calculate the error vector for DIIS extrapolation.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
            
        Returns:
        --------
        numpy.ndarray
            Error vector
        """
        if self.diis_error_metric == 'grad':
            # Use gradient as error vector
            return gradient.copy()
        
        elif self.diis_error_metric == 'energy_change' and len(self.energy_history) > 0:
            # Use energy change combined with gradient
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-10:
                return gradient.copy()
                
            # Scale by energy change if possible
            if len(self.energy_history) > 0:
                prev_energy = self.energy_history[-1]
                energy_diff = abs(self.energy_history[-1] - prev_energy)
                scale_factor = np.sqrt(1.0 + energy_diff)
                return gradient * scale_factor
            else:
                return gradient.copy()
                
        elif self.diis_error_metric == 'combined' and len(self.grad_history) > 0:
            # Combine current gradient with gradient difference
            prev_grad = self.grad_history[-1]
            grad_diff = gradient - prev_grad
            return gradient + 0.5 * grad_diff
            
        else:
            # Default to gradient
            return gradient.copy()
    
    def _build_krylov_subspace(self, gradient):
        """
        Build or update the Krylov subspace.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
            
        Returns:
        --------
        bool
            Whether the Krylov basis was successfully updated
        """
        # Check if we need to restart
        restart = (self.iter % self.krylov_restart == 0) or not self.krylov_basis
        
        if restart:
            # Start a new Krylov basis with the current gradient
            self.krylov_basis = []
            
            # Normalize gradient to start the basis
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < self.krylov_tolerance:
                # Gradient too small, use random vector
                warnings.warn("Gradient norm too small for Krylov basis, using random vector")
                v = np.random.randn(*gradient.shape)
                v = v / np.linalg.norm(v)
            else:
                # Use normalized gradient
                v = gradient / grad_norm
                
            # Start with normalized gradient
            self.krylov_basis = [v]
            self.hess_proj = None
            
        # Determine current Krylov dimension (may be adaptive)
        if self.dynamic_subspace and self.iter % self.adapt_frequency == 0:
            # Adapt dimension based on convergence or problem size
            grad_norm = np.linalg.norm(gradient)
            if len(self.grad_history) > 2:
                prev_grad_norm = np.linalg.norm(self.grad_history[-2])
                # If convergence is slow, increase dimension
                if grad_norm > 0.7 * prev_grad_norm:
                    self.krylov_dimension = min(self.krylov_dimension + 1, self.krylov_max_dim)
                # If convergence is fast, decrease dimension
                elif grad_norm < 0.3 * prev_grad_norm:
                    self.krylov_dimension = max(self.krylov_dimension - 1, self.krylov_min_dim)
        
        # Expand the Krylov basis if needed
        while len(self.krylov_basis) < self.krylov_dimension:
            try:
                # Generate next Krylov vector (approximately Hv)
                v_prev = self.krylov_basis[-1]
                
                # Approximate Hessian-vector product using finite differences
                # If we have enough history
                if len(self.grad_history) >= 2 and len(self.geom_history) >= 2:
                    # Get finite difference approximation of Hessian-vector product
                    g_diff = self.grad_history[-1] - self.grad_history[-2]
                    x_diff = self.geom_history[-1] - self.geom_history[-2]
                    
                    # Compute direction in x_diff parallel to v_prev
                    v_comp = np.dot(x_diff.flatten(), v_prev.flatten())
                    
                    # If sufficient component in this direction
                    if abs(v_comp) > 1e-10:
                        # Approximate Hessian-vector product
                        Hv = g_diff * (np.dot(v_prev.flatten(), v_prev.flatten()) / v_comp)
                    else:
                        # Fallback: use gradient difference directly
                        Hv = g_diff
                else:
                    # Not enough history, use gradient as approximation
                    Hv = gradient
                
                # Orthogonalize against existing basis
                if self.orthogonalization == 'mgs':
                    # Modified Gram-Schmidt orthogonalization
                    v_next = Hv.copy()
                    for v in self.krylov_basis:
                        proj = np.dot(v.flatten(), v_next.flatten())
                        v_next = v_next - proj * v
                else:
                    # Simple Arnoldi-like orthogonalization
                    v_next = Hv.copy()
                    for v in self.krylov_basis:
                        v_next = v_next - np.dot(v_next.flatten(), v.flatten()) * v
                
                # Normalize the new vector
                v_next_norm = np.linalg.norm(v_next)
                
                # Check if new vector is sufficiently linearly independent
                if v_next_norm < self.krylov_tolerance:
                    # Basis cannot be expanded further
                    print(f"Krylov basis saturated at dimension {len(self.krylov_basis)}")
                    break
                    
                # Add normalized vector to basis
                v_next = v_next / v_next_norm
                self.krylov_basis.append(v_next)
                
                # Update the projected Hessian
                self._update_projected_hessian(Hv, v_next, len(self.krylov_basis)-1)
                
            except Exception as e:
                print(f"Error expanding Krylov basis: {str(e)}")
                # Return current basis
                break
        
        return len(self.krylov_basis) > 1
    
    def _update_projected_hessian(self, Hv, v_next, idx):
        """
        Update the projected Hessian in the Krylov subspace.
        
        Parameters:
        -----------
        Hv : numpy.ndarray
            Hessian-vector product
        v_next : numpy.ndarray
            Next basis vector
        idx : int
            Index of the new vector
        """
        k = len(self.krylov_basis)
        
        # Initialize or expand projected Hessian
        if self.hess_proj is None:
            self.hess_proj = np.zeros((k, k))
        elif self.hess_proj.shape[0] < k:
            # Expand Hessian matrix
            new_hess = np.zeros((k, k))
            old_size = self.hess_proj.shape[0]
            new_hess[:old_size, :old_size] = self.hess_proj
            self.hess_proj = new_hess
        
        # Update the Hessian elements for the new vector
        for i in range(k):
            # H_ij = v_i^T * H * v_j
            if i < idx:
                self.hess_proj[i, idx] = np.dot(self.krylov_basis[i].flatten(), Hv.flatten())
                self.hess_proj[idx, i] = self.hess_proj[i, idx]  # Symmetry
            elif i == idx:
                self.hess_proj[idx, idx] = np.dot(v_next.flatten(), Hv.flatten())
    
    def _solve_krylov_system(self, gradient):
        """
        Solve the optimization problem in the Krylov subspace.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
            
        Returns:
        --------
        tuple
            (step, success)
        """
        k = len(self.krylov_basis)
        
        if k < 2:
            # Not enough basis vectors
            return None, False
        
        # Project gradient into Krylov subspace
        g_proj = np.zeros(k)
        for i in range(k):
            g_proj[i] = np.dot(self.krylov_basis[i].flatten(), gradient.flatten())
        
        # Check if projected Hessian is available
        if self.hess_proj is None or self.hess_proj.shape[0] < k:
            return None, False
        
        try:
            # Try to solve the system (H_proj * alpha = -g_proj)
            # Add regularization for stability
            H_reg = self.hess_proj + self.diis_regularization * np.eye(k)
            alpha = solve(H_reg, -g_proj)
            
            # Construct the step in the original space
            step = np.zeros_like(gradient)
            for i in range(k):
                step += alpha[i] * self.krylov_basis[i]
            
            return step, True
            
        except Exception as e:
            print(f"Failed to solve Krylov system: {str(e)}")
            return None, False
    
    def _solve_diis_system(self):
        """
        Solve the DIIS extrapolation problem.
        
        Returns:
        --------
        tuple
            (diis_geometry, coefficients, success)
        """
        n_points = len(self.geom_history)
        
        if n_points < self.diis_min_points:
            # Not enough points for DIIS
            return None, None, False
        
        # Construct the DIIS B matrix
        B = np.zeros((n_points + 1, n_points + 1))
        
        # Fill B matrix with error vector dot products
        for i in range(n_points):
            for j in range(n_points):
                B[i, j] = np.dot(self.error_history[i].flatten(), 
                                 self.error_history[j].flatten())
        
        # Add regularization to diagonal for numerical stability
        np.fill_diagonal(B[:n_points, :n_points], 
                         np.diag(B[:n_points, :n_points]) + self.diis_regularization)
        
        # Add Lagrange multiplier constraints
        B[n_points, :n_points] = 1.0
        B[:n_points, n_points] = 1.0
        B[n_points, n_points] = 0.0
        
        # Right-hand side vector with constraint
        rhs = np.zeros(n_points + 1)
        rhs[n_points] = 1.0
        
        try:
            # Try to solve the DIIS equations
            coeffs = solve(B, rhs)[:n_points]
            
            # Check for reasonable coefficients
            if np.any(np.isnan(coeffs)) or np.max(np.abs(coeffs)) > 10.0:
                raise ValueError("DIIS produced extreme coefficients")
                
            # Calculate the extrapolated geometry
            diis_geom = np.zeros_like(self.geom_history[0])
            for i in range(n_points):
                diis_geom += coeffs[i] * self.geom_history[i]
                
            return diis_geom, coeffs, True
            
        except Exception as e:
            print(f"DIIS system solve failed: {str(e)}")
            
            try:
                # Fallback: try least squares with non-negative constraints
                coeffs = np.zeros(n_points)
                coeffs[-1] = 1.0  # Start with most recent point
                
                return self.geom_history[-1].copy(), coeffs, False
                
            except:
                # Ultimate fallback: use most recent point
                coeffs = np.zeros(n_points)
                coeffs[-1] = 1.0
                return self.geom_history[-1].copy(), coeffs, False
    
    def _blend_steps(self, krylov_step, diis_step, original_step, gradient):
        """
        Blend different steps based on their quality.
        
        Parameters:
        -----------
        krylov_step : numpy.ndarray or None
            Step from Krylov subspace optimization
        diis_step : numpy.ndarray or None
            Step from DIIS extrapolation
        original_step : numpy.ndarray
            Original optimizer step
        gradient : numpy.ndarray
            Current gradient
            
        Returns:
        --------
        numpy.ndarray
            Blended step
        """
        # Use negative gradient direction for evaluating alignment
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-10:
            neg_grad_dir = -gradient / grad_norm
        else:
            # If gradient is nearly zero, any direction is fine
            return original_step
        
        # Start with original step
        blend_weights = {"original": 1.0, "krylov": 0.0, "diis": 0.0}
        
        # Evaluate Krylov step if available
        if krylov_step is not None:
            krylov_norm = np.linalg.norm(krylov_step)
            if krylov_norm > 1e-10:
                # Check alignment with negative gradient
                krylov_align = np.dot(krylov_step.flatten(), neg_grad_dir.flatten()) / krylov_norm
                
                # If reasonably aligned with descent direction
                if krylov_align > 0.1:
                    # Allocate weight to Krylov step
                    krylov_weight = min(0.7, max(0.3, krylov_align))
                    blend_weights["krylov"] = krylov_weight
                    blend_weights["original"] -= krylov_weight * 0.7  # Reduce original weight
        
        # Evaluate DIIS step if available
        if diis_step is not None:
            diis_norm = np.linalg.norm(diis_step)
            if diis_norm > 1e-10:
                # Calculate DIIS step (extrapolated geometry - current geometry)
                diis_step_vec = diis_step - self.geom_history[-1]
                diis_step_norm = np.linalg.norm(diis_step_vec)
                
                if diis_step_norm > 1e-10:
                    # Check alignment with negative gradient
                    diis_align = np.dot(diis_step_vec.flatten(), neg_grad_dir.flatten()) / diis_step_norm
                    
                    # If reasonably aligned with descent direction
                    if diis_align > 0.0:
                        # Allocate weight to DIIS step based on accuracy history
                        diis_weight = self.weight_factor * min(0.8, max(0.2, diis_align))
                        blend_weights["diis"] = diis_weight
                        # Reduce other weights proportionally
                        total_other = blend_weights["original"] + blend_weights["krylov"]
                        if total_other > 0:
                            factor = (1.0 - diis_weight) / total_other
                            blend_weights["original"] *= factor
                            blend_weights["krylov"] *= factor
        
        # Normalize weights (should sum to 1.0)
        total_weight = sum(blend_weights.values())
        if abs(total_weight - 1.0) > 1e-10:
            for k in blend_weights:
                blend_weights[k] /= total_weight
        
        print(f"Blend weights: Original={blend_weights['original']:.3f}, "
              f"Krylov={blend_weights['krylov']:.3f}, DIIS={blend_weights['diis']:.3f}")
        
        # Construct the blended step
        blended_step = np.zeros_like(original_step)
        
        # Add original step contribution
        blended_step += blend_weights["original"] * original_step
        
        # Add Krylov step contribution
        if krylov_step is not None and blend_weights["krylov"] > 0:
            # Scale to similar magnitude as original step
            orig_norm = np.linalg.norm(original_step)
            krylov_norm = np.linalg.norm(krylov_step)
            
            if krylov_norm > 1e-10 and orig_norm > 1e-10:
                # Scale Krylov step if it's much larger/smaller
                if krylov_norm > 2.0 * orig_norm:
                    scale = 2.0 * orig_norm / krylov_norm
                    krylov_step = krylov_step * scale
                elif krylov_norm < 0.5 * orig_norm:
                    scale = 0.5 * orig_norm / krylov_norm
                    krylov_step = krylov_step * scale
            
            blended_step += blend_weights["krylov"] * krylov_step
        
        # Add DIIS step contribution
        if diis_step is not None and blend_weights["diis"] > 0:
            # Convert DIIS geometry to step vector
            diis_step_vec = diis_step - self.geom_history[-1]
            
            # Scale to similar magnitude as original step
            orig_norm = np.linalg.norm(original_step)
            diis_norm = np.linalg.norm(diis_step_vec)
            
            if diis_norm > 1e-10 and orig_norm > 1e-10:
                # Scale DIIS step if it's much larger/smaller
                if diis_norm > 2.0 * orig_norm:
                    scale = 2.0 * orig_norm / diis_norm
                    diis_step_vec = diis_step_vec * scale
                elif diis_norm < 0.5 * orig_norm:
                    scale = 0.5 * orig_norm / diis_norm
                    diis_step_vec = diis_step_vec * scale
            
            blended_step += blend_weights["diis"] * diis_step_vec.reshape(original_step.shape)
        
        # Final safety check
        blended_norm = np.linalg.norm(blended_step)
        orig_norm = np.linalg.norm(original_step)
        
        if blended_norm > 3.0 * orig_norm:
            # Cap the step size for safety
            print(f"Capping step size: {blended_norm:.4f} → {3.0 * orig_norm:.4f}")
            blended_step = blended_step * (3.0 * orig_norm / blended_norm)
        
        return blended_step
    
    def run(self, geom_num_list, energy, gradient, original_move_vector):
        """
        Run Krylov-DIIS optimization step.
        
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
            Optimized step vector
        """
        print("Krylov-DIIS method")
        
        # Update histories
        self._update_histories(geom_num_list, gradient, energy)
        
        # Skip advanced methods in very early iterations
        if self.iter < self.diis_start_iter:
            print(f"Building history (iteration {self.iter+1}), using original step")
            self.iter += 1
            return original_move_vector
        
        # Skip if in recovery mode
        if self.recovery_counter > 0:
            print(f"In recovery mode ({self.recovery_counter} steps remaining)")
            self.recovery_counter -= 1
            self.iter += 1
            return original_move_vector
        
        # Build/update Krylov subspace
        krylov_success = self._build_krylov_subspace(gradient)
        print(f"Krylov basis dimension: {len(self.krylov_basis)}")
        
        # Attempt Krylov subspace optimization
        krylov_step = None
        if krylov_success:
            krylov_step, krylov_solved = self._solve_krylov_system(gradient)
            if krylov_solved:
                print("Krylov subspace optimization successful")
            else:
                print("Krylov subspace optimization failed")
        
        # Attempt DIIS extrapolation if we have enough history
        diis_geom = None
        if len(self.geom_history) >= self.diis_min_points and self.diis_failure_count < self.max_diis_failures:
            diis_geom, diis_coeffs, diis_success = self._solve_diis_system()
            
            if diis_success:
                print("DIIS extrapolation successful")
                print("DIIS coefficients:", ", ".join(f"{c:.4f}" for c in diis_coeffs))
                # Reset failure counter on success
                self.diis_failure_count = max(0, self.diis_failure_count - 1)
            else:
                print("DIIS extrapolation failed")
                self.diis_failure_count += 1
                
                # If too many failures, enter recovery mode
                if self.diis_failure_count >= self.max_diis_failures:
                    print(f"Too many DIIS failures ({self.diis_failure_count}), entering recovery mode")
                    self.recovery_counter = self.recovery_steps
        
        # Blend the steps
        blended_step = self._blend_steps(krylov_step, diis_geom, original_move_vector, gradient)
        
        # Add a small steepest descent component for stability if requested
        if self.stabilize_step:
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                sd_step = -0.05 * gradient / grad_norm
                step_norm = np.linalg.norm(blended_step)
                if step_norm > 1e-10:
                    # Scale SD step relative to blended step
                    sd_step = sd_step * (0.1 * step_norm / np.linalg.norm(sd_step))
                blended_step += sd_step
                print("Added stabilizing SD component")
        
        # Final checks
        if np.any(np.isnan(blended_step)) or np.any(np.isinf(blended_step)):
            print("Warning: Numerical issues in step, using original step")
            move_vector = original_move_vector
        else:
            move_vector = blended_step
        
        # Make sure step isn't too small
        if np.linalg.norm(move_vector) < 1e-10:
            print("Warning: Step too small, using scaled original step")
            move_vector = original_move_vector
            
            # If original also too small, use scaled negative gradient
            if np.linalg.norm(move_vector) < 1e-10:
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > 1e-10:
                    move_vector = -0.1 * gradient / grad_norm
        
        self.iter += 1
        return move_vector