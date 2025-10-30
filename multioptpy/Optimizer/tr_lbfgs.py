import numpy as np
from numpy.linalg import norm, inv, qr, eig, pinv

class TRLBFGS:
    """Trust-Region Limited-memory BFGS optimizer.
    
    A trust region variant of the L-BFGS algorithm that approximates the inverse Hessian
    matrix using a limited amount of memory while enforcing step length constraints
    through a trust region approach.
    
    Unlike standard L-BFGS with line search, this implementation adapts the trust region
    radius dynamically based on the agreement between predicted and actual function
    value reductions.
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.Initialization = True
        self.iter = 0
        
        # Set default parameters
        self.FC_COUNT = config.get("fc_count", -1)
        self.saddle_order = 0
        self.memory = config.get("memory", 30)  # Number of previous steps to remember
        
        # Trust region parameters
        self.delta_hat = config.get("delta_hat", 0.5)  # Upper bound for trust region radius (0.5)
        self.delta_min = config.get("delta_min", 0.01)  # Lower bound for trust region radius (0.01)
        self.delta_tr = config.get("initial_delta", self.delta_hat * 0.75)  # Current trust region radius
        self.eta = config.get("eta", 0.25 * 0.9)  # η ∈ [0, 1/4)
        
        # Powell damping parameters
        self.use_powell_damping = config.get("use_powell_damping", True)
        self.powell_theta = config.get("powell_theta", 0.2)  # Damping threshold
        
        # Newton solver parameters
        self.newton_max_iter = config.get("newton_max_iter", 50)
        self.newton_tol = config.get("newton_tol", 1e-6)
        self.newton_alpha_min = config.get("newton_alpha_min", 1e-8)
        
        # Storage for L-BFGS vectors
        self.s = []  # Position differences
        self.y = []  # Gradient differences
        self.rho = []  # 1 / (y_k^T s_k)
        
        # Initialize Hessian related variables
        self.hessian = None
        self.bias_hessian = None
        self.gamma = 1.0  # Initial scaling factor
        
        # Trust region internal variables
        self.P_ll = None  # P_parallel matrix
        self.Lambda_1 = None  # Eigenvalues
        self.lambda_min = 0.0  # Minimum eigenvalue
        self.prev_move_vector = None  # Previous step
        self.tr_subproblem_solved = False  # Flag to check if trust region subproblem has been solved
        
        print(f"Initialized TRLBFGS optimizer with memory={self.memory}, "
              f"initial trust region radius={self.delta_tr}, "
              f"bounds=[{self.delta_min}, {self.delta_hat}]")
        print(f"Powell damping: {self.use_powell_damping}, theta={self.powell_theta}")
        print(f"Newton solver: max_iter={self.newton_max_iter}, tol={self.newton_tol}")
        
    def set_hessian(self, hessian):
        """Set explicit Hessian matrix."""
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        """Set bias Hessian matrix."""
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        """Get Hessian matrix (if available)."""
        return self.hessian
    
    def get_bias_hessian(self):
        """Get bias Hessian matrix."""
        return self.bias_hessian
    
    def apply_powell_damping(self, s, y):
        """Apply Powell's damping strategy to ensure positive definiteness.
        
        This method modifies the gradient difference y to satisfy the curvature condition
        y^T s > 0, which is necessary for maintaining a positive definite Hessian approximation.
        
        Parameters:
        -----------
        s : ndarray
            Position difference vector (x_{k+1} - x_k)
        y : ndarray
            Gradient difference vector (g_{k+1} - g_k)
            
        Returns:
        --------
        y_corrected : ndarray
            Corrected gradient difference that satisfies curvature condition
        damped : bool
            True if damping was applied, False otherwise
        """
        s_flat = s.flatten()
        y_flat = y.flatten()
        
        s_dot_y = np.dot(s_flat, y_flat)
        s_dot_s = np.dot(s_flat, s_flat)
        
        # Check if curvature condition is satisfied
        threshold = self.powell_theta * s_dot_s
        
        if s_dot_y < threshold:
            # Apply damping correction
            # y_corrected = r * y + (1 - r) * B * s
            # where B * s ≈ s for simplicity (identity approximation)
            r = (1 - self.powell_theta) * s_dot_s / (s_dot_s - s_dot_y)
            y_corrected = r * y_flat + (1 - r) * self.gamma * s_flat
            
            print(f"Powell damping applied: s^T y = {s_dot_y:.4e} < {threshold:.4e}, r = {r:.4f}")
            return y_corrected, True
        
        return y_flat, False
    
    def check_curvature_condition(self, s, y, epsilon=1e-10):
        """Check if the curvature condition is satisfied.
        
        The curvature condition y^T s > 0 must be satisfied to maintain
        a positive definite Hessian approximation.
        
        Parameters:
        -----------
        s : ndarray
            Position difference vector
        y : ndarray
            Gradient difference vector
        epsilon : float
            Tolerance for checking positivity
            
        Returns:
        --------
        satisfied : bool
            True if curvature condition is satisfied
        dot_product : float
            The value of y^T s
        """
        s_flat = s.flatten()
        y_flat = y.flatten()
        
        dot_product = np.dot(y_flat, s_flat)
        
        # Also check for numerical issues
        s_norm = norm(s_flat)
        y_norm = norm(y_flat)
        
        if s_norm < epsilon or y_norm < epsilon:
            print(f"Warning: Very small vector norms (||s||={s_norm:.4e}, ||y||={y_norm:.4e})")
            return False, dot_product
        
        # Normalized check for better numerical stability
        cos_angle = dot_product / (s_norm * y_norm)
        
        satisfied = dot_product > epsilon
        
        if not satisfied:
            print(f"Curvature condition violated: y^T s = {dot_product:.4e}, cos(angle) = {cos_angle:.4f}")
        
        return satisfied, dot_product
    
    def update_vectors(self, displacement, delta_grad):
        """Update the vectors used for the L-BFGS approximation.
        
        This method incorporates improved curvature condition checking
        and optional Powell damping for better numerical stability.
        """
        # Flatten vectors if they're not already
        s = displacement.flatten()
        y = delta_grad.flatten()
        
        # Check curvature condition
        curvature_satisfied, dot_product = self.check_curvature_condition(s, y)
        
        # Apply Powell damping if enabled and curvature condition is not satisfied
        if self.use_powell_damping and not curvature_satisfied:
            y, damped = self.apply_powell_damping(s, y)
            if damped:
                # Recompute dot product after damping
                dot_product = np.dot(y, s)
                print(f"After damping: y^T s = {dot_product:.4e}")
        
        # Final check before updating
        if abs(dot_product) < 1e-10:
            print("Warning: y^T s is still too small after correction, skipping update")
            return False
        
        # Calculate rho = 1 / (y^T * s)
        rho = 1.0 / dot_product
        
        # Add to history
        self.s.append(s)
        self.y.append(y)
        self.rho.append(rho)
        
        # Remove oldest vectors if exceeding memory limit
        if len(self.s) > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
        
        # Update gamma (scaling factor for initial Hessian approximation)
        y_dot_y = np.dot(y, y)
        self.gamma = y_dot_y / dot_product
        print(f"Updated gamma = {self.gamma:.4f}, memory size = {len(self.s)}")
        
        return True
    
    def solve_trust_region_newton(self, g_ll, g_NL_norm, delta):
        """Solve the trust region subproblem using improved Newton's method.
        
        This method finds the Lagrange multiplier sigma that satisfies:
        ||p(sigma)|| = delta, where p(sigma) is the step in the trust region.
        
        Parameters:
        -----------
        g_ll : ndarray
            Projection of gradient onto eigenspace
        g_NL_norm : float
            Norm of gradient component orthogonal to eigenspace
        delta : float
            Trust region radius
            
        Returns:
        --------
        sigma_star : float
            Optimal Lagrange multiplier
        iterations : int
            Number of Newton iterations performed
        """
        # Define the trust region constraint function
        def phi_bar_func(sigma):
            """Compute phi_bar(sigma) = 1/||p(sigma)|| - 1/delta"""
            u = np.sum((g_ll ** 2) / ((self.Lambda_1 + sigma) ** 2)) + \
                (g_NL_norm ** 2) / ((self.gamma + sigma) ** 2)
            v = np.sqrt(u)
            return 1.0 / v - 1.0 / delta
        
        def phi_bar_prime_func(sigma):
            """Compute the derivative of phi_bar function."""
            lambda_sigma = self.Lambda_1 + sigma
            gamma_sigma = self.gamma + sigma
            
            u = np.sum(g_ll ** 2 / lambda_sigma ** 2) + g_NL_norm ** 2 / gamma_sigma ** 2
            u_prime = -2.0 * np.sum(g_ll ** 2 / lambda_sigma ** 3) - 2.0 * g_NL_norm ** 2 / gamma_sigma ** 3
            
            return -0.5 * u ** (-1.5) * u_prime
        
        # Initialize sigma
        sigma = max(0.0, -self.lambda_min)
        phi_bar_0 = phi_bar_func(sigma)
        
        print(f"Newton solver: initial sigma = {sigma:.6e}, phi_bar = {phi_bar_0:.6e}")
        
        # Check if we're already at the solution (interior case)
        if abs(phi_bar_0) < self.newton_tol:
            print("Interior solution found (sigma = 0 or -lambda_min)")
            return sigma, 0
        
        # Need to solve for positive sigma if phi_bar_0 < 0
        if phi_bar_0 < 0:
            # Initialize with a better starting guess
            sigma_hat = max(np.max(np.abs(g_ll) / delta - self.Lambda_1), 0.0)
            sigma = max(sigma, sigma_hat)
            
            print(f"Boundary case: starting Newton from sigma = {sigma:.6e}")
            
            # Newton iterations with backtracking line search
            for iteration in range(self.newton_max_iter):
                phi_bar = phi_bar_func(sigma)
                
                # Check convergence
                if abs(phi_bar) < self.newton_tol:
                    print(f"Newton converged in {iteration + 1} iterations, sigma = {sigma:.6e}")
                    return sigma, iteration + 1
                
                # Compute Newton direction
                phi_bar_prime = phi_bar_prime_func(sigma)
                
                # Check for zero derivative (should not happen in practice)
                if abs(phi_bar_prime) < 1e-15:
                    print(f"Warning: phi_bar_prime too small ({phi_bar_prime:.4e}), stopping Newton")
                    break
                
                # Newton step
                delta_sigma = -phi_bar / phi_bar_prime
                
                # Backtracking line search to ensure progress
                alpha = 1.0
                sigma_new = sigma + alpha * delta_sigma
                
                # Ensure sigma stays non-negative
                while sigma_new < 0:
                    alpha *= 0.5
                    sigma_new = sigma + alpha * delta_sigma
                    if alpha < self.newton_alpha_min:
                        sigma_new = sigma * 0.5
                        break
                
                # Simple backtracking: ensure |phi_bar| decreases
                phi_bar_new = phi_bar_func(sigma_new)
                backtrack_count = 0
                max_backtrack = 10
                
                while abs(phi_bar_new) > abs(phi_bar) and backtrack_count < max_backtrack:
                    alpha *= 0.5
                    if alpha < self.newton_alpha_min:
                        # Can't make progress, accept current sigma
                        print(f"Backtracking failed, accepting sigma = {sigma:.6e}")
                        return sigma, iteration + 1
                    
                    sigma_new = max(0.0, sigma + alpha * delta_sigma)
                    phi_bar_new = phi_bar_func(sigma_new)
                    backtrack_count += 1
                
                if backtrack_count > 0:
                    print(f"  Iter {iteration + 1}: backtracked {backtrack_count} times, alpha = {alpha:.4f}")
                
                # Update sigma
                sigma = sigma_new
                
                if iteration % 10 == 0 or iteration == self.newton_max_iter - 1:
                    print(f"  Iter {iteration + 1}: sigma = {sigma:.6e}, phi_bar = {phi_bar:.6e}, alpha = {alpha:.4f}")
            
            print(f"Newton reached max iterations ({self.newton_max_iter}), sigma = {sigma:.6e}")
            return sigma, self.newton_max_iter
        
        elif self.lambda_min < 0:
            # Hard case: negative curvature, set sigma to make Hessian PSD
            sigma_star = -self.lambda_min
            print(f"Hard case (negative curvature), using sigma = {sigma_star:.6e}")
            return sigma_star, 0
        else:
            # Interior solution
            print("Interior solution (no constraint active)")
            return 0.0, 0
    
    def compute_lbfgs_tr_step(self, gradient, delta):
        """Compute trust region step using L-BFGS approximation.
        
        Parameters:
        ----------
        gradient : ndarray
            Current gradient vector
        delta : float
            Current trust region radius
            
        Returns:
        -------
        ndarray
            Step vector satisfying the trust region constraint
        """
        n = gradient.size
        g = gradient.flatten()
        
        # If we have no history yet, just return scaled negative gradient
        if len(self.s) == 0:
            direction = -g
            step_length = norm(direction)
            if step_length > delta:
                direction = direction * (delta / step_length)
            # Initialize P_ll and Lambda_1 as None to indicate we're using steepest descent
            self.P_ll = None
            self.Lambda_1 = None
            self.tr_subproblem_solved = False
            return direction.reshape(gradient.shape)
        
        # Create S and Y matrices from history
        try:
            # Stack vectors as columns
            S_matrix = np.column_stack(self.s)
            Y_matrix = np.column_stack(self.y)
            
            # Construct Psi matrix
            Psi = np.hstack((self.gamma * S_matrix, Y_matrix))
            
            # Compute S^T Y and related matrices
            S_T_Y = np.dot(S_matrix.T, Y_matrix)
            L = np.tril(S_T_Y, k=-1)
            D = np.diag(np.diag(S_T_Y))
            
            # Construct the M matrix
            M_block = np.block([
                [self.gamma * np.dot(S_matrix.T, S_matrix), L],
                [L.T, -D]
            ])
            
            # Check if M is well-conditioned
            try:
                M = -inv(M_block)
            except np.linalg.LinAlgError:
                print("Warning: M matrix is singular, using pseudoinverse")
                M = -pinv(M_block)
            
            # QR factorization
            Q, R = qr(Psi, mode='reduced')
            
            # Check if R is invertible and handle accordingly
            try:
                # Try to compute eigendecomposition
                R_inv = pinv(R)  # Use pseudoinverse instead of inv for robustness
                eigen_decomp = np.dot(np.dot(R, M), R.T)
                eigen_values, eigen_vectors = eig(eigen_decomp)
                
                # Sort eigenvalues and eigenvectors
                idx = eigen_values.argsort()
                eigen_values_sorted = eigen_values[idx].real
                eigen_vectors_sorted = eigen_vectors[:, idx].real
                
                # Store for later use
                self.Lambda_1 = self.gamma + eigen_values_sorted
                self.lambda_min = min(np.min(self.Lambda_1), self.gamma)
                
                # Compute P_ll using the pseudoinverse for robustness
                self.P_ll = np.dot(Psi, np.dot(R_inv, eigen_vectors_sorted))
                
                g_ll = np.dot(self.P_ll.T, g)
                g_NL_norm_squared = max(0, norm(g)**2 - norm(g_ll)**2)
                g_NL_norm = np.sqrt(g_NL_norm_squared)
                
                # Solve trust region subproblem using improved Newton method
                sigma_star, newton_iters = self.solve_trust_region_newton(g_ll, g_NL_norm, delta)
                
                # Compute trust region step
                tau_star = self.gamma + sigma_star
                
                # Use pseudoinverse for robustness
                p_star = -1/tau_star * (g - np.dot(Psi, np.dot(pinv(tau_star * inv(M) + np.dot(Psi.T, Psi)), np.dot(Psi.T, g))))
                
                # Verify step is within trust region
                step_norm = norm(p_star)
                if step_norm > delta * (1 + 1e-6):  # Allow slight numerical error
                    print(f"Warning: Step length ({step_norm:.6e}) exceeds trust region radius ({delta:.6e})")
                    p_star = p_star * (delta / step_norm)
                    print(f"Rescaled step to length {norm(p_star):.6e}")
                
                self.tr_subproblem_solved = True
                print(f"Trust region step computed: ||p|| = {norm(p_star):.6e}, sigma = {sigma_star:.6e}")
                return p_star.reshape(gradient.shape)
                
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Error in L-BFGS trust region calculation: {e}")
                print("Falling back to steepest descent")
                
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Error constructing L-BFGS matrices: {e}")
            print("Falling back to steepest descent")
        
        # Fallback to steepest descent if any errors occur
        direction = -g
        step_length = norm(direction)
        if step_length > delta:
            direction = direction * (delta / step_length)
        
        # Reset internal variables
        self.P_ll = None
        self.Lambda_1 = None
        self.tr_subproblem_solved = False
        
        return direction.reshape(gradient.shape)
    
    def compute_reduction_ratio(self, g, p, actual_reduction):
        """Compute the ratio between actual and predicted reduction.
        
        Parameters:
        ----------
        g : ndarray
            Gradient at current point
        p : ndarray
            Step vector
        actual_reduction : float
            Actual reduction in function value f(x) - f(x+p)
            
        Returns:
        -------
        float
            Reduction ratio: actual reduction / predicted reduction
        """
        # Calculate predicted reduction from quadratic model
        p_flat = p.flatten()
        g_flat = g.flatten()
        
        # Basic predicted reduction (first-order term)
        linear_reduction = np.dot(g_flat, p_flat)
        
        if len(self.s) > 0 and self.tr_subproblem_solved and self.P_ll is not None and self.Lambda_1 is not None:
            try:
                # Use the built L-BFGS model
                # Compute parallel component
                p_ll = np.dot(self.P_ll.T, p_flat)
                p_NL_norm_squared = max(0, norm(p_flat)**2 - norm(p_ll)**2)
                p_NL_norm = np.sqrt(p_NL_norm_squared)
                
                # Compute p^T B p (p^T times Hessian approximation times p)
                p_T_B_p = np.sum(self.Lambda_1 * p_ll**2) + self.gamma * p_NL_norm**2
                
                # Predicted reduction: -g^T p - 0.5 p^T B p
                pred_reduction = -(linear_reduction + 0.5 * p_T_B_p)
            except Exception as e:
                print(f"Error computing quadratic model: {e}")
                # Fallback to linear model
                pred_reduction = -linear_reduction
        else:
            # For first iteration or when subproblem not fully solved, use simple model
            pred_reduction = -linear_reduction
        
        # Avoid division by zero or very small numbers
        if abs(pred_reduction) < 1e-10:
            print("Warning: Predicted reduction is near zero")
            return 0.0
            
        ratio = actual_reduction / pred_reduction
        print(f"Actual reduction: {actual_reduction:.6e}, "
              f"Predicted reduction: {pred_reduction:.6e}, "
              f"Ratio: {ratio:.4f}")
        
        return ratio
    
    def determine_step(self, dr):
        """Determine step to take according to maxstep.
        
        Parameters:
        ----------
        dr : ndarray
            Step vector
            
        Returns:
        -------
        ndarray
            Step vector constrained by maxstep if necessary
        """
        if self.config.get("maxstep") is None:
            return dr
        
        # Get maxstep from config
        maxstep = self.config.get("maxstep")
        
        # Calculate step lengths
        dr_reshaped = dr.reshape(-1, 3) if dr.size % 3 == 0 else dr.reshape(-1, dr.size)
        steplengths = np.sqrt((dr_reshaped**2).sum(axis=1))
        longest_step = np.max(steplengths)
        
        # Scale step if necessary
        if longest_step > maxstep:
            dr = dr * (maxstep / longest_step)
            print(f"Step constrained by maxstep={maxstep}")
        
        return dr
    
    def trust_region_step(self, g, B_e, pre_B_e, actual_reduction=None):
        """Perform a trust region optimization step.
        
        Parameters:
        ----------
        g : ndarray
            Current gradient
        B_e : float
            Current energy/function value
        pre_B_e : float
            Previous energy/function value
        actual_reduction : float, optional
            Actual reduction in function value
            
        Returns:
        -------
        ndarray
            Step vector
        """
        print("Trust region step calculation")
        
        # First iteration or if actual_reduction is None (before energy evaluation)
        if self.Initialization or actual_reduction is None:
            if self.Initialization:
                self.Initialization = False
                print("First iteration - using initial trust region radius")
            
            # Compute step using current trust region radius
            p = self.compute_lbfgs_tr_step(g, self.delta_tr)
            self.prev_move_vector = p
            return p
        
        # Compute reduction ratio
        reduction_ratio = self.compute_reduction_ratio(g, self.prev_move_vector, actual_reduction)
        
        # Update trust region radius based on reduction ratio
        if reduction_ratio < 0.25:
            self.delta_tr = max(0.25 * self.delta_tr, self.delta_min)  # Ensure we don't go below min radius
            print(f"Shrinking trust region radius to {self.delta_tr:.4f}")
        elif reduction_ratio > 0.75 and np.isclose(norm(self.prev_move_vector.flatten()), self.delta_tr, rtol=1e-2):
            self.delta_tr = min(2.0 * self.delta_tr, self.delta_hat)  # Ensure we don't exceed max radius
            print(f"Expanding trust region radius to {self.delta_tr:.4f}")
        else:
            print(f"Maintaining trust region radius at {self.delta_tr:.4f}")
        
        # Compute next step using updated trust region radius
        p = self.compute_lbfgs_tr_step(g, self.delta_tr)
        
        # Store step for later reduction ratio calculation
        self.prev_move_vector = p
        
        return p
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        """Run a single optimization step.
        
        Parameters:
        ----------
        geom_num_list : ndarray
            Current geometry/position
        B_g : ndarray
            Current gradient
        pre_B_g : ndarray
            Previous gradient
        pre_geom : ndarray
            Previous geometry/position
        B_e : float
            Current energy/function value
        pre_B_e : float
            Previous energy/function value
        pre_move_vector : ndarray
            Previous step vector
        initial_geom_num_list : ndarray
            Initial geometry
        g : ndarray
            Current gradient (unbranched)
        pre_g : ndarray
            Previous gradient (unbranched)
            
        Returns:
        -------
        ndarray
            Step vector for the next iteration
        """
        print(f"\n{'='*50}\nIteration {self.iter}\n{'='*50}")
        
        # Compute actual energy reduction if not first iteration
        actual_reduction = None
        if not self.Initialization and self.iter > 0:
            actual_reduction = pre_B_e - B_e
            print(f"Energy change: {actual_reduction:.6e}")
        
        # Compute trust region step
        try:
            move_vector = self.trust_region_step(B_g, B_e, pre_B_e, actual_reduction)
        except Exception as e:
            print(f"Error in trust region step: {e}")
            print("Falling back to steepest descent")
            # Fallback to simple steepest descent
            move_vector = -B_g
            if norm(move_vector) > self.delta_tr:
                move_vector = move_vector * (self.delta_tr / norm(move_vector))
        
        # Apply maxstep constraint if needed
        move_vector = self.determine_step(move_vector)
        
        # If this is not the first iteration, update L-BFGS vectors
        if self.iter > 0:
            # Calculate displacement and gradient difference
            delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
            
            # Update L-BFGS vectors with improved curvature checking
            update_success = self.update_vectors(displacement, delta_grad)
            
            if not update_success:
                print("Warning: L-BFGS update skipped due to curvature condition failure")
        
        self.iter += 1
        return -move_vector