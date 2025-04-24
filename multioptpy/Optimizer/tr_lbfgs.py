import numpy as np
from numpy.linalg import norm, inv, qr, eig, pinv
import math

class TRLBFGS:
    """Trust-Region Limited-memory BFGS optimizer.
    
    A trust region variant of the L-BFGS algorithm that approximates the inverse Hessian
    matrix using a limited amount of memory while enforcing step length constraints
    through a trust region approach.
    
    Unlike standard L-BFGS with line search, this implementation adapts the trust region
    radius dynamically based on the agreement between predicted and actual function
    value reductions.
    
    Parameters
    ----------
    memory : int, optional
        Number of previous steps to remember for L-BFGS approximation (default: 30)
    delta_hat : float, optional
        Upper bound for trust region radius (default: 0.5)
    delta_min : float, optional
        Lower bound for trust region radius (default: 0.01)
    initial_delta : float, optional
        Initial trust region radius (default: 0.75*delta_hat)
    eta : float, optional
        Acceptance threshold for step quality, must be in [0, 1/4) (default: 0.25*0.9)
    fc_count : int, optional
        Force constant counter (default: -1)
    maxstep : float, optional
        Maximum allowed step size
    debug : bool, optional
        Whether to print additional debug information (default: False)
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.initialization = True
        self.iter = 0
        
        # Set default parameters
        self.fc_count = config.get("fc_count", -1)
        self.saddle_order = 0
        self.memory = config.get("memory", 30)  # Number of previous steps to remember
        
        # Trust region parameters
        self.delta_hat = config.get("delta_hat", 0.5)  # Upper bound for trust region radius (0.5)
        self.delta_min = config.get("delta_min", 0.01)  # Lower bound for trust region radius (0.01)
        self.delta_tr = config.get("initial_delta", self.delta_hat * 0.75)  # Current trust region radius
        self.eta = config.get("eta", 0.25 * 0.9)  # η ∈ [0, 1/4)
        
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
        
        # Debug flag
        self.debug = config.get("debug", False)
        
        print(f"Initialized TRLBFGS optimizer with memory={self.memory}, "
              f"initial trust region radius={self.delta_tr}, "
              f"bounds=[{self.delta_min}, {self.delta_hat}]")
        
    def set_hessian(self, hessian):
        """Set explicit Hessian matrix.
        
        Parameters
        ----------
        hessian : ndarray
            The Hessian matrix to use
        """
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        """Set bias Hessian matrix.
        
        Parameters
        ----------
        bias_hessian : ndarray
            The bias Hessian matrix to use
        """
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        """Get Hessian matrix (if available).
        
        Returns
        -------
        ndarray or None
            The currently set Hessian matrix, or None if not set
        """
        return self.hessian
    
    def get_bias_hessian(self):
        """Get bias Hessian matrix.
        
        Returns
        -------
        ndarray or None
            The currently set bias Hessian matrix, or None if not set
        """
        return self.bias_hessian
    
    def update_vectors(self, displacement, delta_grad):
        """Update the vectors used for the L-BFGS approximation.
        
        This method updates the internal state of the L-BFGS approximation
        by adding new displacement and gradient difference vectors.
        
        Parameters
        ----------
        displacement : ndarray
            The difference between current and previous positions
        delta_grad : ndarray
            The difference between current and previous gradients
        """
        # Flatten vectors if they're not already
        s = displacement.flatten()
        y = delta_grad.flatten()
        
        # Calculate rho = 1 / (y^T * s)
        dot_product = np.dot(y, s)
        if abs(dot_product) < 1e-10:
            # Avoid division by very small numbers
            rho = 1000.0
            print("Warning: y^T s is very small, using default rho value")
        else:
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
        # This sets the scaling of the identity matrix in BFGS update
        if dot_product > 1e-10:
            self.gamma = np.dot(y, y) / dot_product
            if self.debug:
                print(f"Updated gamma = {self.gamma:.4f}")
        else:
            if self.debug:
                print("Warning: Not updating gamma due to small y^T s")
    
    def compute_lbfgs_tr_step(self, gradient, delta):
        """Compute trust region step using L-BFGS approximation.
        
        This solves the trust region subproblem:
        min_p   m(p) = f + g^T p + 0.5 p^T B p
        s.t.    ||p|| ≤ delta
        
        where B is the L-BFGS approximation of the Hessian.
        
        Parameters
        ----------
        gradient : ndarray
            Current gradient vector
        delta : float
            Current trust region radius
            
        Returns
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
            
            # Construct Psi matrix (for compact representation)
            Psi = np.hstack((self.gamma * S_matrix, Y_matrix))
            
            # Compute S^T Y and related matrices
            S_T_Y = np.dot(S_matrix.T, Y_matrix)
            L = np.tril(S_T_Y, k=-1)  # Lower triangular part without diagonal
            D = np.diag(np.diag(S_T_Y))  # Diagonal part
            
            # Construct the M matrix for compact representation of the Hessian
            M_block = np.block([
                [self.gamma * np.dot(S_matrix.T, S_matrix), L],
                [L.T, -D]
            ])
            
            # Check if M is well-conditioned
            try:
                M = -inv(M_block)
            except np.linalg.LinAlgError:
                if self.debug:
                    print("Warning: M matrix is singular, using pseudoinverse")
                M = -pinv(M_block)
            
            # QR factorization for numerical stability
            Q, R = qr(Psi, mode='reduced')
            
            # Check if R is invertible and handle accordingly
            try:
                # Use pseudoinverse for robustness
                R_inv = pinv(R)
                
                # Compute eigendecomposition for compact representation
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
                
                # Project the gradient onto the L-BFGS subspace
                g_ll = np.dot(self.P_ll.T, g)
                g_NL_norm_squared = max(0, norm(g)**2 - norm(g_ll)**2)
                g_NL_norm = np.sqrt(g_NL_norm_squared)
                
                # Trust region subproblem solution functions
                def phi_bar_func(sigma, delta):
                    """Compute the trust region constraint function.
                    
                    Find σ such that ||p(σ)|| = delta
                    
                    Parameters
                    ----------
                    sigma : float
                        Lagrange multiplier for trust region constraint
                    delta : float
                        Trust region radius
                        
                    Returns
                    -------
                    float
                        Value of constraint function (should be zero at solution)
                    """
                    u = np.sum((g_ll ** 2) / ((self.Lambda_1 + sigma) ** 2)) + \
                        (g_NL_norm ** 2) / ((self.gamma + sigma) ** 2)
                    v = np.sqrt(u)
                    return 1/v - 1/delta
                
                def phi_bar_prime_func(sigma):
                    """Compute the derivative of phi_bar function.
                    
                    Used in Newton's method to find optimal sigma.
                    
                    Parameters
                    ----------
                    sigma : float
                        Current sigma value
                        
                    Returns
                    -------
                    float
                        Derivative of phi_bar with respect to sigma
                    """
                    u = np.sum(g_ll ** 2 / (self.Lambda_1 + sigma) ** 2) + \
                        g_NL_norm ** 2 / (self.gamma + sigma) ** 2
                    u_prime = np.sum(-2 * g_ll ** 2 / (self.Lambda_1 + sigma) ** 3) - \
                              2 * g_NL_norm ** 2 / (self.gamma + sigma) ** 3
                    return u ** (-3/2) * u_prime
                
                # Find sigma that solves the trust region constraint
                tol = 1e-6  # Increased precision
                max_iter = 20  # More iterations for better convergence
                sigma = max(0, -self.lambda_min + 1e-6)  # Add small offset for stability
                
                if phi_bar_func(sigma, delta) < 0:
                    # Need to find a positive sigma
                    # Better initial guess for sigma
                    sigma_hat = max(np.max(np.abs(g_ll) / delta - self.Lambda_1), 
                                   np.abs(g_NL_norm) / delta - self.gamma, 0)
                    sigma = max(0, sigma_hat)
                    
                    # Newton iterations to find optimal sigma
                    converged = False
                    for i in range(max_iter):
                        phi_bar = phi_bar_func(sigma, delta)
                        if abs(phi_bar) < tol:
                            converged = True
                            break
                            
                        phi_bar_prime = phi_bar_prime_func(sigma)
                        # Safeguard Newton step
                        if abs(phi_bar_prime) < 1e-10:
                            # If derivative is too small, use bisection
                            sigma = sigma * 1.5
                        else:
                            sigma_new = sigma - phi_bar / phi_bar_prime
                            
                            # Strong safeguards to ensure sigma remains positive and reasonable
                            if sigma_new <= 0 or not np.isfinite(sigma_new):
                                sigma = sigma * 1.5
                            elif sigma_new > 100 * sigma:
                                sigma = sigma * 2
                            else:
                                sigma = sigma_new
                    
                    if converged:
                        sigma_star = sigma
                        if self.debug:
                            print(f"Found sigma_star = {sigma_star:.6e} after {i+1} iterations")
                    else:
                        # Fallback if Newton method doesn't converge
                        # Use bisection method as a more robust alternative
                        if self.debug:
                            print("Newton method did not converge, trying bisection")
                        
                        sigma_lo = max(0, -self.lambda_min + 1e-6)
                        sigma_hi = 1e6  # Large upper bound
                        
                        # Find an upper bound where phi_bar is positive
                        phi_lo = phi_bar_func(sigma_lo, delta)
                        phi_hi = phi_bar_func(sigma_hi, delta)
                        
                        # If we can't bracket a root, use a simple approach
                        if phi_lo * phi_hi > 0:
                            # Just use a scaled version of the gradient
                            sigma_star = sigma_lo
                            if self.debug:
                                print(f"Couldn't bracket a root, using sigma_star = {sigma_star:.6e}")
                        else:
                            # Bisection method
                            for i in range(50):  # More iterations for bisection
                                sigma_mid = (sigma_lo + sigma_hi) / 2
                                phi_mid = phi_bar_func(sigma_mid, delta)
                                
                                if abs(phi_mid) < tol:
                                    break
                                
                                if phi_mid * phi_lo < 0:
                                    sigma_hi = sigma_mid
                                    phi_hi = phi_mid
                                else:
                                    sigma_lo = sigma_mid
                                    phi_lo = phi_mid
                            
                            sigma_star = sigma_mid
                            if self.debug:
                                print(f"Found sigma_star = {sigma_star:.6e} using bisection after {i+1} iterations")
                elif self.lambda_min < 0:
                    # Hard case: negative curvature
                    # More careful handling of the hard case
                    sigma_star = -self.lambda_min + 1e-6  # Small offset for stability
                    if self.debug:
                        print(f"Hard case (negative curvature), using sigma_star = {sigma_star:.6e}")
                else:
                    # No need for regularization
                    sigma_star = 0
                    if self.debug:
                        print("No regularization needed (sigma_star = 0)")
                
                # Compute trust region step
                tau_star = self.gamma + sigma_star
                
                # More robust computation of the step
                try:
                    # First attempt: direct formula
                    M_reg = tau_star * inv(M) + np.dot(Psi.T, Psi)
                    M_reg_inv = pinv(M_reg)
                    p_star = -1/tau_star * (g - np.dot(Psi, np.dot(M_reg_inv, np.dot(Psi.T, g))))
                except np.linalg.LinAlgError:
                    # Fallback approach: solve component-wise
                    g_parallel = np.dot(self.P_ll, g_ll)
                    g_orth = g - g_parallel
                    
                    # Compute parallel component
                    p_parallel = np.zeros_like(g)
                    for j in range(len(self.Lambda_1)):
                        p_parallel += -g_ll[j] / (self.Lambda_1[j] + sigma_star) * self.P_ll[:, j]
                    
                    # Compute orthogonal component
                    p_orth = -g_orth / (self.gamma + sigma_star)
                    
                    # Combine components
                    p_star = p_parallel + p_orth
                
                # Verify step is within trust region
                step_norm = norm(p_star)
                if step_norm > delta * (1 + 1e-6):  # Allow slight numerical error
                    if self.debug:
                        print(f"Warning: Step length ({step_norm:.6e}) exceeds trust region radius ({delta:.6e})")
                    p_star = p_star * (delta / step_norm)
                
                self.tr_subproblem_solved = True
                return p_star.reshape(gradient.shape)
                
            except (np.linalg.LinAlgError, ValueError) as e:
                if self.debug:
                    print(f"Error in L-BFGS trust region calculation: {e}")
                    print("Falling back to steepest descent")
                
        except (ValueError, np.linalg.LinAlgError) as e:
            if self.debug:
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
        
        This ratio is used to determine whether to accept the step and
        how to adjust the trust region radius.
        
        Parameters
        ----------
        g : ndarray
            Gradient at current point
        p : ndarray
            Step vector
        actual_reduction : float
            Actual reduction in function value f(x) - f(x+p)
            
        Returns
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
                # Use the built L-BFGS model for more accurate prediction
                # Compute parallel component
                p_ll = np.dot(self.P_ll.T, p_flat)
                p_NL_norm_squared = max(0, norm(p_flat)**2 - norm(p_ll)**2)
                p_NL_norm = np.sqrt(p_NL_norm_squared)
                
                # Compute p^T B p (p^T times Hessian approximation times p)
                p_T_B_p = np.sum(self.Lambda_1 * p_ll**2) + self.gamma * p_NL_norm**2
                
                # Predicted reduction: -g^T p - 0.5 p^T B p
                pred_reduction = -(linear_reduction + 0.5 * p_T_B_p)
            except Exception as e:
                if self.debug:
                    print(f"Error computing quadratic model: {e}")
                # Fallback to linear model
                pred_reduction = -linear_reduction
        else:
            # For first iteration or when subproblem not fully solved, use simple model
            pred_reduction = -linear_reduction
        
        # Avoid division by zero or very small numbers
        if abs(pred_reduction) < 1e-10:
            if self.debug:
                print("Warning: Predicted reduction is near zero")
            # Return a conservative value
            return 0.0
            
        ratio = actual_reduction / pred_reduction
        
        # Safeguard against numerical issues
        if not np.isfinite(ratio):
            if self.debug:
                print("Warning: Non-finite reduction ratio, using 0.0")
            return 0.0
            
        if self.debug:
            print(f"Actual reduction: {actual_reduction:.6e}, "
                  f"Predicted reduction: {pred_reduction:.6e}, "
                  f"Ratio: {ratio:.4f}")
        
        return ratio
    
    def determine_step(self, dr):
        """Determine step to take according to maxstep constraint.
        
        Parameters
        ----------
        dr : ndarray
            Step vector
            
        Returns
        -------
        ndarray
            Step vector constrained by maxstep if necessary
        """
        if self.config.get("maxstep") is None:
            return dr
        
        # Get maxstep from config
        maxstep = self.config.get("maxstep")
        
        # Calculate step lengths (handle both coordinate and general vector formats)
        if dr.size % 3 == 0 and dr.size > 3:  # Likely atomic coordinates in 3D
            dr_reshaped = dr.reshape(-1, 3)
            steplengths = np.sqrt((dr_reshaped**2).sum(axis=1))
            longest_step = np.max(steplengths)
        else:
            # Generic vector - just compute total norm
            longest_step = norm(dr)
        
        # Scale step if necessary
        if longest_step > maxstep:
            dr = dr * (maxstep / longest_step)
            if self.debug:
                print(f"Step constrained by maxstep={maxstep}")
        
        return dr
    
    def trust_region_step(self, g, B_e, pre_B_e, actual_reduction=None):
        """Perform a trust region optimization step.
        
        This is the main method that implements the trust region strategy
        by computing the step and updating the trust region radius.
        
        Parameters
        ----------
        g : ndarray
            Current gradient
        B_e : float
            Current energy/function value
        pre_B_e : float
            Previous energy/function value
        actual_reduction : float, optional
            Actual reduction in function value
            
        Returns
        -------
        ndarray
            Step vector
        """
        if self.debug:
            print("Trust region step calculation")
        
        # First iteration or if actual_reduction is None (before energy evaluation)
        if self.initialization or actual_reduction is None:
            if self.initialization:
                self.initialization = False
                if self.debug:
                    print("First iteration - using initial trust region radius")
            
            # Compute step using current trust region radius
            p = self.compute_lbfgs_tr_step(g, self.delta_tr)
            self.prev_move_vector = p
            return p
        
        # Compute reduction ratio
        reduction_ratio = self.compute_reduction_ratio(g, self.prev_move_vector, actual_reduction)
        
        # Update trust region radius based on reduction ratio
        if reduction_ratio < 0.25:
            # Step was not effective, shrink trust region
            self.delta_tr = max(0.25 * self.delta_tr, self.delta_min)  # Ensure we don't go below min radius
            if self.debug:
                print(f"Shrinking trust region radius to {self.delta_tr:.4f}")
        elif reduction_ratio > 0.75 and np.isclose(norm(self.prev_move_vector.flatten()), self.delta_tr, rtol=1e-2):
            # Step was very effective and was constrained by trust region, expand radius
            self.delta_tr = min(2.0 * self.delta_tr, self.delta_hat)  # Ensure we don't exceed max radius
            if self.debug:
                print(f"Expanding trust region radius to {self.delta_tr:.4f}")
        else:
            # Step was reasonably effective, keep trust region the same
            if self.debug:
                print(f"Maintaining trust region radius at {self.delta_tr:.4f}")
        
        # Compute next step using updated trust region radius
        p = self.compute_lbfgs_tr_step(g, self.delta_tr)
        
        # Store step for later reduction ratio calculation
        self.prev_move_vector = p
        
        return p
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        """Run a single optimization step.
        
        This is the main method called from the optimization loop.
        
        Parameters
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
            
        Returns
        -------
        ndarray
            Step vector for the next iteration
        """
        if self.debug:
            print(f"\n{'='*50}\nIteration {self.iter}\n{'='*50}")
        
        # Compute actual energy reduction if not first iteration
        actual_reduction = None
        if not self.initialization and self.iter > 0:
            actual_reduction = pre_B_e - B_e
            if self.debug:
                print(f"Energy change: {actual_reduction:.6e}")
        
        # Compute trust region step
        try:
            move_vector = self.trust_region_step(B_g, B_e, pre_B_e, actual_reduction)
        except Exception as e:
            print(f"Error in trust region step: {e}")
            print("Falling back to steepest descent")
            # Fallback to simple steepest descent
            move_vector = -B_g
            move_norm = norm(move_vector)
            if move_norm > 0 and move_norm > self.delta_tr:
                move_vector = move_vector * (self.delta_tr / move_norm)
        
        # Apply maxstep constraint if needed
        move_vector = self.determine_step(move_vector)
        
        # If this is not the first iteration, update L-BFGS vectors
        if self.iter > 0:
            # Calculate displacement and gradient difference
            delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
            
            # Update L-BFGS vectors
            self.update_vectors(displacement, delta_grad)
        
        self.iter += 1
        return -move_vector
 