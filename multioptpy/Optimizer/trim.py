import numpy as np
from scipy.optimize import newton

from multioptpy.Utils.calc_tools import Calculationtools

class TRIM:
    def __init__(self, saddle_order=0):
        """
        Trust Region Image Minimization (TRIM) for transition state optimization.
        
        ref.: https://doi.org/10.1016/0009-2614(91)90115-P
              Helgaker, 1991
        """
        # Trust region parameters
        self.trust_radius = 0.3          # Default trust radius
        self.trust_radius_min = 0.01     # Minimum trust radius
        self.trust_radius_max = 0.5      # Maximum trust radius
        
        # Transition state mode parameters
        self.roots = [0]                 # Indices of eigenvalues to follow uphill (default: lowest mode)
        self.ts_mode_history = []        # History of transition state modes
        self.mode_following_threshold = 0.7  # Dot product threshold for mode following
        
        # Step control parameters
        self.max_step_norm = 0.5         # Maximum allowed step size
        self.min_eigenvalue_shift = 1e-8 # Minimum eigenvalue shift for numerical stability
        
        # Convergence tracking
        self.predicted_energy_changes = []
        self.iter = 0
        self.H = None                    # Store Hessian for energy prediction
        self.saddle_order = saddle_order  # Desired saddle order
        # Logging and diagnostics
        self.debug = False
        
    def log(self, message):
        """Print log message if debug is enabled."""
        if self.debug:
            print(f"TRIM: {message}")
        else:
            # Always print important information even when debug is disabled
            if "μ=" in message or "norm(step)" in message:
                print(f"TRIM: {message}")
    
    def update_ts_mode(self, eigvals, eigvecs):
        """
        Update the transition state mode to follow based on eigenvalues and eigenvectors.
        
        Parameters:
        -----------
        eigvals : numpy.ndarray
            Eigenvalues of the Hessian matrix
        eigvecs : numpy.ndarray
            Eigenvectors of the Hessian matrix
        """
        # Sort eigenvalues to find the lowest (potentially negative) ones
        idx = np.argsort(eigvals)
        
        # Default to following the lowest mode (idx[0])
        if not self.ts_mode_history:
            self.roots = [idx[0:self.saddle_order]]
            self.ts_mode_history.append(eigvecs[:, idx[0:self.saddle_order]])
            return
        
        # If we have history, ensure we follow the consistent mode
        prev_mode = self.ts_mode_history[-1]
        overlaps = np.abs([np.dot(prev_mode.flatten(), eigvecs[:, i].flatten()) for i in idx[:3]])
        
        # Find the mode with highest overlap with previous mode
        max_overlap_idx = np.argmax(overlaps)
        if overlaps[max_overlap_idx] > self.mode_following_threshold:
            # Use the mode with highest overlap
            self.roots = [idx[max_overlap_idx]]
        else:
            # Default to lowest eigenvalue if no good overlap
            self.roots = [idx[0:self.saddle_order]]
        
        self.ts_mode_history.append(eigvecs[:, self.roots[0:self.saddle_order]])
        self.log(f"Following mode with eigenvalue {eigvals[self.roots[0]]:.6f}")
    
    def quadratic_model(self, gradient, hessian, step):
        """
        Predict energy change using quadratic model.
        
        Parameters:
        -----------
        gradient : numpy.ndarray
            Current gradient
        hessian : numpy.ndarray
            Current Hessian matrix
        step : numpy.ndarray
            Step vector
            
        Returns:
        --------
        float
            Predicted energy change
        """
        step_flat = step.flatten()
        grad_flat = gradient.flatten()
        
        linear_term = np.dot(grad_flat, step_flat)
        quadratic_term = 0.5 * np.dot(step_flat, np.dot(hessian, step_flat))
        
        return linear_term + quadratic_term

    def get_step(self, forces, hessian, eigvals, eigvecs):
        """
        Calculate the TRIM step.
        
        Parameters:
        -----------
        
        forces : numpy.ndarray
            Current forces (-gradient)
        hessian : numpy.ndarray
            Current Hessian matrix
        eigvals : numpy.ndarray
            Eigenvalues of the Hessian
        eigvecs : numpy.ndarray
            Eigenvectors of the Hessian
        
        Returns:
        --------
        numpy.ndarray
            Calculated step vector
        """
        gradient = -forces
        self.H = hessian  # Store for energy prediction
        
        
        # Update transition state mode
        if self.saddle_order > 0:
            self.update_ts_mode(eigvals, eigvecs)
            print(f"Signs of eigenvalue and -vector of root(s) {self.roots} will be reversed!")
        
        
        # Transform gradient to basis of eigenvectors
        gradient_ = eigvecs.T.dot(gradient.flatten())
        
        if self.saddle_order > 0:
            # Construct image function by inverting the signs of the eigenvalue and
            # -vector of the mode to follow uphill.
            eigvals_ = eigvals.copy()
            eigvals_[self.roots] *= -1
            gradient_ = gradient_.copy()
            gradient_[self.roots] *= -1
        else:
            eigvals_ = eigvals.copy()
            gradient_ = gradient_.copy()
        
        def get_step(mu):
            """Calculate step with level shift parameter mu."""
            zetas = -gradient_ / (eigvals_ - mu)
            # Replace nan with 0.
            zetas = np.nan_to_num(zetas)
            # Transform to original basis
            step = np.dot(eigvecs, zetas[:, np.newaxis])
            return step
        
        def get_step_norm(mu):
            """Calculate norm of step with level shift parameter mu."""
            return np.linalg.norm(get_step(mu))
        
        def func(mu):
            """Function to find mu such that step norm equals trust radius."""
            return get_step_norm(mu) - self.trust_radius
        
        # Initialize level shift parameter
        mu = 0
        norm0 = get_step_norm(mu)
        
        # Apply trust radius constraint if needed
        if norm0 > self.trust_radius:
            try:
                mu, res = newton(func, x0=mu, full_output=True)
                if res.converged:
                    self.log(f"Using levelshift of μ={mu:.4f}")
                else:
                    self.log("Newton method for levelshift did not converge, using simple scaling")
                    mu = 0
                    step = get_step(mu)
                    step = step * (self.trust_radius / norm0)
                    return step
            except:
                self.log("Error in levelshift calculation, using simple scaling")
                mu = 0
                step = get_step(mu)
                step = step * (self.trust_radius / norm0)
                return step
        else:
            self.log("Took pure newton step without levelshift")
        
        # Calculate final step
        step = get_step(mu)
        step_norm = np.linalg.norm(step)
        self.log(f"norm(step)={step_norm:.6f}")
        
        # Store predicted energy change
        predicted_change = self.quadratic_model(gradient, self.H, step)
        self.predicted_energy_changes.append(predicted_change)
        
        return step
        
    def run(self, geom_num_list, B_g, hessian, trust_radius, original_move_vector):
        """
        Run TRIM optimization step
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry
        energy : float
            Current energy value
        B_g : numpy.ndarray
            Current gradient (forces with sign flipped)
        original_move_vector : numpy.ndarray
            Step calculated by the original method (ignored in TRIM)
        hessian : numpy.ndarray, optional
            Current Hessian matrix (required for TRIM)
        trust_radius : float
            Trust radius for the optimization step
        original_move_vector : numpy.ndarray
            Step calculated by the original method (ignored in TRIM)

        Returns:
        --------
        numpy.ndarray
            Optimized step vector
        """
        print("TRIM method")
        n_coords = len(geom_num_list)
        step_norm = np.linalg.norm(original_move_vector)
        print(f"Original step norm: {step_norm:.6f}, Trust radius: {trust_radius:.6f}")
        self.trust_radius = trust_radius
        if step_norm < self.trust_radius:
            print("Step is within trust radius, using original move vector")
            return original_move_vector
        
        # Check if Hessian is provided
        if hessian is None:
            print("Error: Hessian matrix is required for TRIM method")
            return original_move_vector  # Fallback to steepest descent
        
        hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(hessian, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), display_eigval=False)
        
        # Compute eigenvalues and eigenvectors of the Hessian
        try:
            eigvals, eigvecs = np.linalg.eigh(hessian)
        except np.linalg.LinAlgError:
            print("Warning: Eigenvalue computation failed, using diagonal approximation")
            # Use diagonal approximation if eigenvalue computation fails
            H_diag = np.diag(np.diag(hessian))
            eigvals = np.diag(H_diag)
            eigvecs = np.eye(len(eigvals))
        
        # Reshape forces for compatibility
        forces = -B_g.reshape(-1)  # Convert gradient to forces (sign flip)
        
        # Calculate TRIM step
        move_vector = self.get_step(forces, hessian, eigvals, eigvecs)
        
        # Ensure step has correct shape
        move_vector = -1*move_vector.reshape(n_coords, 1)
        
        # Final safety checks
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10 or np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Step issue detected, using scaled gradient instead")
            move_vector = original_move_vector
        
        self.iter += 1
        return move_vector