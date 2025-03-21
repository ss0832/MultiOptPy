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
    
    def update_vectors(self, displacement, delta_grad):
        """Update the vectors used for the L-BFGS approximation."""
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
        if dot_product > 1e-10:
            self.gamma = np.dot(y, y) / dot_product
            print(f"Updated gamma = {self.gamma:.4f}")
        else:
            print("Warning: Not updating gamma due to small y^T s")
    
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
                
                # Solve trust region subproblem
                def phi_bar_func(sigma, delta):
                    """Compute the trust region constraint function."""
                    u = np.sum((g_ll ** 2) / ((self.Lambda_1 + sigma) ** 2)) + \
                        (g_NL_norm ** 2) / ((self.gamma + sigma) ** 2)
                    v = np.sqrt(u)
                    return 1/v - 1/delta
                
                def phi_bar_prime_func(sigma):
                    """Compute the derivative of phi_bar function."""
                    u = np.sum(g_ll ** 2 / (self.Lambda_1 + sigma) ** 2) + \
                        g_NL_norm ** 2 / (self.gamma + sigma) ** 2
                    u_prime = np.sum(g_ll ** 2 / (self.Lambda_1 + sigma) ** 3) + \
                            g_NL_norm ** 2 / (self.gamma + sigma) ** 3
                    return u ** (-3/2) * u_prime
                
                # Find sigma that solves the trust region constraint
                tol = 1e-4
                sigma = max(0, -self.lambda_min)
                
                if phi_bar_func(sigma, delta) < 0:
                    # Need to find a positive sigma
                    sigma_hat = max(np.max(np.abs(g_ll) / delta - self.Lambda_1), 0)
                    sigma = max(0, sigma_hat)
                    
                    # Newton iterations to find optimal sigma
                    for i in range(10):  # limit iterations for safety
                        phi_bar = phi_bar_func(sigma, delta)
                        if abs(phi_bar) < tol:
                            break
                        phi_bar_prime = phi_bar_prime_func(sigma)
                        sigma_new = sigma - phi_bar / phi_bar_prime
                        
                        # Safeguard against negative sigma
                        if sigma_new <= 0:
                            sigma = sigma / 2
                        else:
                            sigma = sigma_new
                        
                    sigma_star = sigma
                    print(f"Found sigma_star = {sigma_star} after {i+1} iterations")
                elif self.lambda_min < 0:
                    # Hard case: negative curvature
                    sigma_star = -self.lambda_min
                    print(f"Hard case (negative curvature), using sigma_star = {sigma_star}")
                else:
                    # No need for regularization
                    sigma_star = 0
                    print("No regularization needed (sigma_star = 0)")
                
                # Compute trust region step
                tau_star = self.gamma + sigma_star
                
                # Use pseudoinverse for robustness
                p_star = -1/tau_star * (g - np.dot(Psi, np.dot(pinv(tau_star * inv(M) + np.dot(Psi.T, Psi)), np.dot(Psi.T, g))))
                
                # Verify step is within trust region
                step_norm = norm(p_star)
                if step_norm > delta * (1 + 1e-6):  # Allow slight numerical error
                    print(f"Warning: Step length ({step_norm}) exceeds trust region radius ({delta})")
                    p_star = p_star * (delta / step_norm)
                
                self.tr_subproblem_solved = True
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
            
            # Update L-BFGS vectors
            self.update_vectors(displacement, delta_grad)
        
        self.iter += 1
        return -move_vector