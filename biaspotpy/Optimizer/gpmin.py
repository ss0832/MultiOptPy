import numpy as np
from scipy.linalg import cholesky, cho_solve, LinAlgError
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

class GPmin:
    def __init__(self, **config):
        """
        Reference: https://doi.org/10.1063/1.5009347
        
        Gaussian Process Minimizer (GPmin) with inverse distance coordinates.
        Uses 1/R internal coordinates as input to predict energy and Cartesian forces.
        
        Uses RBF kernel and performs surrogate PES optimization with improved step size control.
        """
        # GP parameters
        self.n_points = 0
        self.kernel_func = 'RBF'
        self.noise_energy = 1e-8
        self.length_scale = 0.5
        self.sigma_f = 1.0
        
        # Trust radius parameters - larger values for bigger steps
        self.trust_radius = 1.0       # Increased initial trust radius 
        self.trust_radius_max = 5.0   # Increased maximum trust radius
        self.trust_radius_min = 2.0  # Minimum trust radius
        self.trust_increase = 5.0     # Larger trust radius increase factor 
        self.trust_decrease = 0.2     # Trust radius decrease factor
        
        # Step size control
        self.min_step_size = 0.1      # Minimum allowed step size for non-converged systems
        self.step_scale = 1.0         # Global scale factor for step sizes
        self.force_scale_factor = 0.3 # Scaling factor for force-based steps
        self.adaptive_step = True     # Enable adaptive step sizing
        
        # Internal coordinates parameters
        self.min_dist = 1e-7           # Minimum distance to avoid singularity in 1/R
        self.dist_scale = 1.0         # Scaling factor for distances
        
        # Surrogate PES optimization parameters
        self.surrogate_maxiter = 20   # Maximum iterations for surrogate optimization
        self.surrogate_ftol = 1e-1    # Function tolerance for surrogate optimization
        self.surrogate_gtol = 1e-1    # Gradient tolerance for surrogate optimization
        
        # Data selection parameters
        self.max_gp_pts = 100
        self.selection_method = 'recent'  # 'recent', 'best', 'diverse'
        
        # Optimization parameters
        self.alpha = 0.3              # Increased step size factor (was 0.1)
        
        # Flags
        self.display_flag = True
        self.Initialization = True
        
        # Override defaults with provided config
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.hessian = None
        self.bias_hessian = None
   
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, 
            pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """
        Perform one optimization step using GP regression with inverse distance coordinates.
        """
        # Start timer for performance measurement
        start_time = time.time()
        
        # Convert inputs to numpy arrays
        geom_array = np.asarray(geom_num_list)
        geom_shape = geom_array.shape
        geom_flat = geom_array.flatten()
        forces_flat = -np.asarray(B_g).flatten()  # Forces are negative of gradients
        
        if self.Initialization:
            if self.display_flag:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing GPmin with RBF kernel")
                print(f"Trust radius: {self.trust_radius:.3f}, Max trust radius: {self.trust_radius_max:.3f}")
            
            # Initialize data structures
            self.dim = len(geom_flat)
            self.n_atoms = len(geom_flat) // 3
            self.geom_shape = geom_shape
            
            # Calculate number of distances (n_atoms choose 2)
            self.n_dist = self.n_atoms * (self.n_atoms - 1) // 2
            
            # Convert to internal coordinates (inverse distances)
            coords_3d = geom_array.reshape(-1, 3)
            inv_dist = self._cart_to_inverse_dist(coords_3d)
            
            # Store historical data
            self.X_cart_all = np.array([geom_flat])    # Cartesian positions
            self.X_all = np.array([inv_dist])          # Inverse distances
            self.Y_all = np.array([float(B_e)])        # Energies
            self.F_all = np.array([forces_flat])       # Forces (Cartesian)
            
            # Selected training data
            self.X = np.array([inv_dist])
            self.Y = np.array([float(B_e)])
            self.X_cart = np.array([geom_flat])
            self.F = np.array([forces_flat])
            
            # Store Jacobians for coordinate conversion
            self.jacobians = [self._calc_jacobian(coords_3d)]
            
            # Initialize GP model
            self._init_gp()
            self.Initialization = False
            self.n_points = 1
            
            # First step: take larger force-based step
            force_norm = np.linalg.norm(forces_flat)
            if force_norm > 1e-10:
                # Use alpha * force direction * force_scale_factor
                move_vector_flat = self.alpha * (forces_flat / force_norm) * self.force_scale_factor
                
                # Scale the step to be at least min_step_size
                step_size = np.linalg.norm(move_vector_flat)
                if step_size < self.min_step_size:
                    move_vector_flat *= self.min_step_size / step_size
                
                # Cap at trust radius
                step_size = np.linalg.norm(move_vector_flat)
                if step_size > self.trust_radius:
                    move_vector_flat *= self.trust_radius / step_size
            else:
                # If forces are nearly zero, use small random step
                move_vector_flat = np.random.normal(0, 0.1, size=self.dim)
                move_vector_flat *= self.min_step_size / np.linalg.norm(move_vector_flat)
        else:
            # Update data with new point
            coords_3d = geom_array.reshape(-1, 3)
            inv_dist = self._cart_to_inverse_dist(coords_3d)
            jacobian = self._calc_jacobian(coords_3d)
            
            self._update_data(geom_flat, inv_dist, jacobian, float(B_e), forces_flat)
            
            # Optimize on the surrogate PES with improved step size control
            move_vector_flat = self._optimize_on_surrogate_pes(geom_flat, forces_flat, float(B_e), inv_dist, jacobian)
            
            # Apply adaptive step sizing if enabled
            if self.adaptive_step:
                move_vector_flat = self._adapt_step_size(move_vector_flat, forces_flat)
        
        # Apply global step scale factor
        move_vector_flat *= self.step_scale
        
        # Final check on step size (trust radius constraint)
        step_size = np.linalg.norm(move_vector_flat)
        if step_size > self.trust_radius:
            move_vector_flat *= self.trust_radius / step_size
        
        # Reshape move vector to match input geometry shape
        move_vector = move_vector_flat.reshape(geom_shape)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if self.display_flag:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPmin with RBF kernel")
            print(f"Points: {self.n_points}, Active points: {len(self.X)}")
            print(f"Trust radius: {self.trust_radius:.6f}, Step magnitude: {np.linalg.norm(move_vector_flat):.6f}")
            
            if self.n_points > 1:
                # Evaluate model quality
                pred_energy, pred_forces = self._predict(inv_dist, jacobian)
                energy_error = abs(pred_energy - float(B_e))
                force_error = np.linalg.norm(pred_forces - forces_flat)
                
                print(f"GP energy prediction error: {energy_error:.6f}")
                print(f"GP force prediction error: {force_error:.6f}")
                print(f"Computation time: {elapsed_time:.3f} seconds")
        
        return -1 * move_vector
    
    def _adapt_step_size(self, step, forces):
        """
        Adaptively adjust step size based on forces and minimum step size.
        
        Args:
            step: Current step vector
            forces: Current forces
            
        Returns:
            Adjusted step vector
        """
        step_size = np.linalg.norm(step)
        force_norm = np.linalg.norm(forces)
        
        # If step is too small but forces are significant, increase step size
        if step_size < self.min_step_size and force_norm > 1e-3:
            # Scale up to minimum step size while preserving direction
            new_step = step * (self.min_step_size / step_size)
            
            if self.display_flag:
                print(f"Step size increased: {step_size:.6f} → {np.linalg.norm(new_step):.6f}")
            
            return new_step
        
        # If forces are large and step is relatively small compared to trust radius,
        # consider taking a larger step in the force direction
        if force_norm > 0.1 and step_size < 0.3 * self.trust_radius:
            # Mix the current step with force direction
            force_dir = forces / force_norm
            mixed_step = 0.7 * step + 0.3 * force_dir * self.trust_radius
            mixed_step_size = np.linalg.norm(mixed_step)
            
            # Make sure we don't exceed trust radius
            if mixed_step_size > self.trust_radius:
                mixed_step *= self.trust_radius / mixed_step_size
            
            if self.display_flag and np.linalg.norm(mixed_step) > step_size * 1.2:
                print(f"Step enhanced with force direction: {step_size:.6f} → {np.linalg.norm(mixed_step):.6f}")
            
            return mixed_step
        
        return step
    
    def _cart_to_inverse_dist(self, coords):
        """Convert Cartesian coordinates to inverse distances."""
        # Calculate pairwise distances
        dist_matrix = squareform(pdist(coords))
        
        # Extract upper triangular part (excluding diagonal)
        upper_indices = np.triu_indices(len(coords), k=1)
        distances = dist_matrix[upper_indices]
        
        # Apply minimum distance threshold to avoid singularities
        distances = np.maximum(distances, self.min_dist)
        
        # Calculate inverse distances (1/R)
        inv_dist = 1.0 / (distances * self.dist_scale)
        
        return inv_dist
    
    def _calc_jacobian(self, coords):
        """
        Calculate the Jacobian matrix for mapping between internal and Cartesian coordinates.
        Returns a matrix of shape (n_dist, n_atoms*3).
        """
        n_atoms = len(coords)
        n_cart = n_atoms * 3
        n_dist = self.n_dist
        
        jacobian = np.zeros((n_dist, n_cart))
        
        # For each atom pair
        idx = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Calculate distance vector r_ij and its norm
                r_ij = coords[j] - coords[i]
                r_norm = np.linalg.norm(r_ij)
                
                # Avoid division by zero
                if r_norm < self.min_dist:
                    r_norm = self.min_dist
                
                # Unit vector in direction of r_ij
                r_unit = r_ij / r_norm
                
                # For inverse distance (1/R), the derivative is -1/R^2 * unit_vector
                factor = -1.0 / (r_norm**2 * self.dist_scale)
                
                # Fill in the Jacobian
                jacobian[idx, i*3:i*3+3] = -r_unit * factor  # For atom i
                jacobian[idx, j*3:j*3+3] = r_unit * factor   # For atom j
                
                idx += 1
        
        return jacobian
    
    def _init_gp(self):
        """Initialize the Gaussian Process model."""
        # Set length scale if not provided as array
        if isinstance(self.length_scale, (int, float)):
            self.length_scale = np.ones(self.n_dist) * self.length_scale
        
        self._update_kernel()
    
    def _update_data(self, cart_pos, inv_dist, jacobian, energy, forces):
        """Add new data point to GP model and select training data."""
        energy = float(energy)
        
        # Add to history
        self.X_cart_all = np.vstack([self.X_cart_all, [cart_pos]])
        self.X_all = np.vstack([self.X_all, [inv_dist]])
        self.Y_all = np.append(self.Y_all, energy)
        self.F_all = np.vstack([self.F_all, [forces]])
        self.jacobians.append(jacobian)
        self.n_points += 1
        
        # Select training data
        self._select_training_data()
        
        # Update kernel matrix
        self._update_kernel()
        
        # Update trust radius based on prediction accuracy
        if self.n_points > 1:
            prev_energy = float(self.Y_all[-2])
            current_energy = float(self.Y_all[-1])
            
            # Calculate relative energy improvement
            rel_improvement = (prev_energy - current_energy) / (abs(prev_energy) + 1e-10)
            
            # Update trust radius based on energy improvement
            if current_energy < prev_energy:
                # Good step - increase trust radius more aggressively
                if rel_improvement > 0.05:  # Significant improvement
                    self.trust_radius = min(self.trust_radius * self.trust_increase, self.trust_radius_max)
                else:  # Modest improvement
                    self.trust_radius = min(self.trust_radius * 1.2, self.trust_radius_max)
            else:
                # Energy increased - decrease trust radius
                self.trust_radius = max(self.trust_radius * self.trust_decrease, self.trust_radius_min)
    
    def _select_training_data(self):
        """Select data points for GP training."""
        n_total = len(self.X_all)
        max_pts = min(self.max_gp_pts, n_total)
        
        if self.selection_method == 'recent':
            # Keep most recent points
            indices = np.arange(n_total - max_pts, n_total)
        elif self.selection_method == 'best':
            # Keep lowest energy points
            indices = np.argsort(self.Y_all)[:max_pts]
        elif self.selection_method == 'diverse':
            # Select diverse points + most recent
            indices = self._select_diverse_points(max_pts)
        else:
            # Default to most recent
            indices = np.arange(n_total - max_pts, n_total)
        
        # Update active training data
        self.X = self.X_all[indices]
        self.X_cart = self.X_cart_all[indices]
        self.Y = self.Y_all[indices]
        self.F = self.F_all[indices]
        self.active_jacobians = [self.jacobians[i] for i in indices]
    
    def _select_diverse_points(self, max_pts):
        """Select diverse set of points."""
        n_total = len(self.X_all)
        
        # Always include most recent point
        selected_indices = [n_total - 1]
        
        if n_total <= max_pts:
            return np.arange(n_total)
        
        # Calculate distances in inverse distance space
        distances = squareform(pdist(self.X_all))
        
        # Greedy selection
        remaining = list(set(range(n_total)) - set(selected_indices))
        while len(selected_indices) < max_pts and remaining:
            # Find point with maximum minimum distance to selected points
            min_distances = []
            for idx in remaining:
                min_dist = min(distances[idx, sel_idx] for sel_idx in selected_indices)
                min_distances.append((min_dist, idx))
            
            _, next_idx = max(min_distances)
            selected_indices.append(next_idx)
            remaining.remove(next_idx)
        
        return np.array(sorted(selected_indices))
    
    def _update_kernel(self):
        """Update the kernel matrices for GP regression."""
        n = len(self.X)
        
        if n == 0:
            return
        
        # Create kernel matrix for energy prediction
        K_ee = np.zeros((n, n))
        
        # Calculate kernel between all pairs of points
        for i in range(n):
            for j in range(i+1):
                K_ee[i, j] = self._rbf_kernel(self.X[i], self.X[j])
                if i != j:
                    K_ee[j, i] = K_ee[i, j]
        
        # Add noise to diagonal
        np.fill_diagonal(K_ee, np.diag(K_ee) + self.noise_energy)
        
        self.K_ee = K_ee
        
        # Compute Cholesky decomposition with fallbacks for numerical stability
        try:
            self.L_ee = cholesky(self.K_ee, lower=True)
        except LinAlgError:
            np.fill_diagonal(self.K_ee, np.diag(self.K_ee) + self.noise_energy * 10.0)
            try:
                self.L_ee = cholesky(self.K_ee, lower=True)
            except LinAlgError:
                np.fill_diagonal(self.K_ee, np.diag(self.K_ee) + self.noise_energy * 100.0)
                self.L_ee = cholesky(self.K_ee, lower=True)
    
    def _rbf_kernel(self, x1, x2):
        """
        RBF (Squared Exponential) kernel function for inverse distance coordinates.
        k(x, x') = σ² * exp(-0.5 * r²)
        where r² is the squared distance between x and x'.
        """
        # Calculate squared distance with scaling
        sq_dist = np.sum(((x1 - x2) / self.length_scale) ** 2)
        
        # RBF kernel formula
        return self.sigma_f * np.exp(-0.5 * sq_dist)
    
    def _rbf_kernel_gradient(self, x1, x2):
        """Gradient of the RBF kernel with respect to x2."""
        diff = (x1 - x2) / (self.length_scale ** 2)
        k = self._rbf_kernel(x1, x2)
        return k * diff
    
    def _predict(self, inv_dist, jacobian):
        """
        Predict energy and forces using the GP model.
        
        Args:
            inv_dist: Inverse distances (1/R) for the test point
            jacobian: Jacobian matrix for coordinate conversion
            
        Returns:
            (pred_energy, pred_forces): Predicted energy and Cartesian forces
        """
        n = len(self.X)
        
        if n == 0:
            return 0.0, np.zeros(self.dim)
        
        # Create kernel vector between test point and training points
        k_e = np.zeros(n)
        for i in range(n):
            k_e[i] = self._rbf_kernel(inv_dist, self.X[i])
        
        # Predict energy
        alpha_e = cho_solve((self.L_ee, True), self.Y)
        pred_energy = float(np.dot(k_e, alpha_e))
        
        # Calculate partial derivatives dE/d(1/R)
        grad_internal = np.zeros(self.n_dist)
        for j in range(self.n_dist):
            for i in range(n):
                # Compute derivative of kernel wrt each inverse distance
                x1 = inv_dist.copy()
                x2 = self.X[i].copy()
                
                # Compute analytical gradient for RBF kernel
                grad_k = self._rbf_kernel_gradient(x1, x2)
                dk_dr = -grad_k[j]  # Negative because we're taking derivative wrt x1
                
                grad_internal[j] += alpha_e[i] * dk_dr
        
        # Convert from internal coordinate gradients to Cartesian forces
        # F = -dE/dR = -J^T * dE/d(1/R)
        pred_forces = -np.dot(jacobian.T, grad_internal)
        
        return pred_energy, pred_forces
    
    def _optimize_on_surrogate_pes(self, position, forces, energy, inv_dist, jacobian):
        """
        Perform optimization on the surrogate PES constructed by GPR.
        This is a more thorough optimization than a single step.
        
        Args:
            position: Current Cartesian coordinates
            forces: Current forces
            energy: Current energy
            inv_dist: Current inverse distances
            jacobian: Current Jacobian matrix
            
        Returns:
            best_step: Optimized step vector
        """
        if self.n_points <= 1:
            # If only one data point, use gradient-based step
            norm_f = np.linalg.norm(forces)
            if norm_f > 1e-10:
                # Take a larger step using force_scale_factor
                step = self.alpha * (forces / norm_f) * self.force_scale_factor
                step_size = np.linalg.norm(step)
                
                # Ensure minimum step size
                if step_size < self.min_step_size:
                    step *= self.min_step_size / step_size
                
                # Cap at trust radius
                if step_size > self.trust_radius:
                    step *= self.trust_radius / step_size
                    
                return step
            else:
                # Small random step
                step = np.random.normal(0, 0.1, size=self.dim)
                step *= self.min_step_size / np.linalg.norm(step)
                return step
        
        if self.display_flag:
            print(f"Starting optimization on surrogate PES (trust radius: {self.trust_radius:.4f})")
        
        # Objective function for minimization on surrogate PES
        def surrogate_objective(step):
            # Calculate inverse distances for new position
            new_coords = (position + step).reshape(-1, 3)
            new_inv_dist = self._cart_to_inverse_dist(new_coords)
            new_jacobian = self._calc_jacobian(new_coords)
            
            # Predict energy at new position
            pred_energy, _ = self._predict(new_inv_dist, new_jacobian)
            return float(pred_energy)
        
        # Gradient function (negative of predicted forces)
        def surrogate_gradient(step):
            # Calculate inverse distances and Jacobian for new position
            new_coords = (position + step).reshape(-1, 3)
            new_inv_dist = self._cart_to_inverse_dist(new_coords)
            new_jacobian = self._calc_jacobian(new_coords)
            
            # Predict forces (negative gradient)
            _, pred_forces = self._predict(new_inv_dist, new_jacobian)
            return -pred_forces  # Negative because we're minimizing energy
        
        # Initial guess: take larger step in force direction
        norm_f = np.linalg.norm(forces)
        if norm_f > 1e-10:
            initial_step = self.trust_radius * 0.8 * forces / norm_f  # 80% of trust radius
        else:
            # Random step with significant magnitude
            initial_step = np.random.normal(0, 0.1, size=self.dim)
            initial_step *= self.trust_radius * 0.5 / max(np.linalg.norm(initial_step), 1e-10)
        
        # Trust region as spherical constraint
        constraint = {'type': 'ineq', 
                      'fun': lambda x: self.trust_radius**2 - np.sum(x**2),
                      'jac': lambda x: -2*x}
        
        # Perform thorough optimization on surrogate PES
        try:
            result = minimize(
                surrogate_objective, 
                initial_step, 
                method='SLSQP',
                jac=surrogate_gradient,
                constraints=[constraint],
                options={
                    'maxiter': self.surrogate_maxiter,
                    'ftol': self.surrogate_ftol, 
                    'disp': False
                }
            )
            
            if self.display_flag:
                if result.success:
                    print(f"Surrogate optimization successful: {result.message}")
                    print(f"Final surrogate energy: {result.fun:.8f}")
                    print(f"Iterations: {result.nit}")
                else:
                    print(f"Surrogate optimization warning: {result.message}")
            
            move_vector = result.x
            
        except Exception as e:
            if self.display_flag:
                print(f"Surrogate optimization failed: {e}")
            # Use force-based step as fallback
            if norm_f > 1e-10:
                move_vector = self.trust_radius * 0.8 * forces / norm_f
            else:
                # Random step
                move_vector = np.random.normal(0, 0.1, size=self.dim)
                move_vector *= self.min_step_size / np.linalg.norm(move_vector)
        
        # Calculate predicted improvement
        new_coords = (position + move_vector).reshape(-1, 3)
        new_inv_dist = self._cart_to_inverse_dist(new_coords)
        new_jacobian = self._calc_jacobian(new_coords)
        pred_new_energy, _ = self._predict(new_inv_dist, new_jacobian)
        
        # Check if the optimization actually improved energy
        if pred_new_energy >= energy:
            # If no improvement, take force-based step instead
            if self.display_flag:
                print("Optimization didn't reduce energy, using force-based step")
            
            if norm_f > 1e-10:
                move_vector = self.trust_radius * 0.8 * forces / norm_f
            else:
                # Random step as last resort
                move_vector = np.random.normal(0, 0.1, size=self.dim)
                move_vector *= self.min_step_size / np.linalg.norm(move_vector)
        
        # Ensure we don't exceed trust radius
        step_size = np.linalg.norm(move_vector)
        if step_size > self.trust_radius:
            move_vector *= self.trust_radius / step_size
        
        # Ensure minimum step size when forces are significant
        if step_size < self.min_step_size and norm_f > 1e-3:
            move_vector *= self.min_step_size / step_size
            
        if self.display_flag:
            pred_improvement = energy - pred_new_energy
            final_step_size = np.linalg.norm(move_vector)
            print(f"Predicted energy improvement: {pred_improvement:.6f}")
            print(f"Step magnitude: {final_step_size:.6f}")
        
        return move_vector
    
    def set_hessian(self, hessian):
        """Store the Hessian matrix (required interface)"""
        self.hessian = hessian
    
    def set_bias_hessian(self, bias_hessian):
        """Store the bias Hessian matrix (required interface)"""
        self.bias_hessian = bias_hessian
        
    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian