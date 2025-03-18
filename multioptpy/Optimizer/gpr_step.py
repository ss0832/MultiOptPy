import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize


class GaussianProcessRegression:
    """Scratch implementation of Gaussian Process Regression"""
    
    def __init__(self, kernel='rbf', length_scale=1.0, amplitude=1.0, noise=1e-6):
        # Kernel parameters
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.amplitude = amplitude
        self.noise = noise
        
        # Model state
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
    
    def kernel(self, X1, X2=None):
        """Compute kernel matrix between X1 and X2"""
        if X2 is None:
            X2 = X1
            
        # RBF/Squared exponential kernel
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        K = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        K = self.amplitude**2 * np.exp(-K / (2 * self.length_scale**2))
        return K
    
    def _nll_fn(self, theta):
        """Negative log-likelihood for hyperparameter optimization"""
        # Extract hyperparameters (in log space)
        length_scale, amplitude, noise = np.exp(theta)
        
        # Store original parameters
        old_ls, old_amp, old_noise = self.length_scale, self.amplitude, self.noise
        self.length_scale, self.amplitude, self.noise = length_scale, amplitude, noise
        
        # Compute kernel with current hyperparameters
        K = self.kernel(self.X_train)
        K_noisy = K + np.eye(len(self.X_train)) * noise
        
        try:
            # Compute log-likelihood
            L = cholesky(K_noisy, lower=True)
            alpha = cho_solve((L, True), self.y_train)
            
            log_det = 2 * np.sum(np.log(np.diag(L)))
            data_fit = np.dot(self.y_train.T, alpha)
            nll = 0.5 * (data_fit + log_det + len(self.X_train) * np.log(2 * np.pi))
        except np.linalg.LinAlgError:
            nll = 1e10
        
        # Restore original parameters
        self.length_scale, self.amplitude, self.noise = old_ls, old_amp, old_noise
        return nll
    
    def optimize_parameters(self):
        """Optimize hyperparameters using maximum likelihood"""
        if len(self.X_train) < 5:
            return False
            
        # Initial guess in log space
        theta_init = [np.log(self.length_scale), np.log(self.amplitude), np.log(self.noise)]
        bounds = [(np.log(0.01), np.log(10.0)), (np.log(0.1), np.log(10.0)), (np.log(1e-10), np.log(1e-2))]
        
        try:
            res = minimize(self._nll_fn, theta_init, method='L-BFGS-B', bounds=bounds)
            if res.success:
                self.length_scale, self.amplitude, self.noise = np.exp(res.x)
                print(f"Optimized hyperparameters: length_scale={self.length_scale:.4f}, "
                      f"amplitude={self.amplitude:.4f}, noise={self.noise:.6f}")
                return True
        except Exception as e:
            print(f"Hyperparameter optimization failed: {str(e)}")
        return False
    
    def fit(self, X, y, optimize=True):
        """Fit GP model to training data"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Optimize hyperparameters if requested
        if optimize and len(X) >= 5:
            self.optimize_parameters()
        
        # Compute kernel matrix
        K = self.kernel(X)
        K_noisy = K + np.eye(len(X)) * self.noise
        
        try:
            self.L = cholesky(K_noisy, lower=True)
            self.alpha = cho_solve((self.L, True), y)
        except np.linalg.LinAlgError:
            # Add more regularization if Cholesky fails
            K_noisy = K + np.eye(len(X)) * (self.noise + 1e-6)
            self.alpha = np.linalg.solve(K_noisy, y)
            self.L = None
        
        return self
    
    def predict(self, X, return_std=False):
        """Make predictions at query points X"""
        if self.X_train is None or self.alpha is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Compute kernel between X and training data
        K_trans = self.kernel(X, self.X_train)
        
        # Mean prediction
        y_mean = np.dot(K_trans, self.alpha)
        
        if return_std:
            # Compute variance
            if self.L is not None:
                v = np.linalg.solve(self.L, K_trans.T)
                K_pred = self.kernel(X)
                y_var = K_pred - np.dot(v.T, v)
            else:
                K_inv = np.linalg.inv(self.kernel(self.X_train) + np.eye(len(self.X_train)) * self.noise)
                K_pred = self.kernel(X)
                y_var = K_pred - np.dot(K_trans, np.dot(K_inv, K_trans.T))
            
            # Ensure variance is positive
            diag = np.maximum(np.diag(y_var), 0)
            return y_mean, np.sqrt(diag)
        
        return y_mean


class GPRStep:
    """GPR-based optimization step generator"""
    
    def __init__(self):
        # GPR model parameters
        self.kernel_type = 'rbf'
        self.length_scale_init = 1.0
        self.amplitude_init = 1.0
        self.noise_init = 1e-6
        
        # Optimization strategy
        self.balance_factor = 0.1
        self.min_points_for_gpr = 5
        self.max_history_points = 25
        self.ucb_beta = 2.0
        self.num_candidates = 25
        self.candidate_scale = 1.0
        self.max_step_norm = 0.5
        
        # State variables
        self.geom_history = []
        self.energy_history = []
        self.gradient_history = []
        self.step_history = []
        self.gpr = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.iter = 0
        self.failures = 0
    
    def _update_histories(self, geometry, energy, gradient):
        """Update optimization histories"""
        self.geom_history.append(geometry.copy())
        self.energy_history.append(energy)
        self.gradient_history.append(gradient.copy())
        
        # Calculate and store step if possible
        if len(self.geom_history) > 1:
            step = geometry - self.geom_history[-2]
            self.step_history.append(step)
        
        # Limit history size
        if len(self.geom_history) > self.max_history_points:
            self.geom_history.pop(0)
            self.energy_history.pop(0)
            self.gradient_history.pop(0)
            if len(self.step_history) > 0:
                self.step_history.pop(0)
    
    def _fit_gpr_model(self):
        """Fit GPR model to current data"""
        if len(self.geom_history) < self.min_points_for_gpr:
            return False
        
        try:
            # Prepare training data
            X_train = np.array([g.flatten() for g in self.geom_history])
            
            # Normalize energy values
            y_train = np.array(self.energy_history)
            self.y_mean = np.mean(y_train)
            self.y_std = np.std(y_train) if np.std(y_train) > 1e-10 else 1.0
            y_train_norm = (y_train - self.y_mean) / self.y_std
            
            # Initialize or reuse GPR model
            if self.gpr is None:
                self.gpr = GaussianProcessRegression(
                    kernel='rbf',
                    length_scale=self.length_scale_init,
                    amplitude=self.amplitude_init,
                    noise=self.noise_init
                )
            
            # Fit model
            self.gpr.fit(X_train, y_train_norm)
            return True
            
        except Exception as e:
            print(f"Error fitting GPR model: {str(e)}")
            self.failures += 1
            return False
    
    def _generate_candidate_steps(self, current_geom, gradient):
        """Generate candidate steps for GPR evaluation"""
        n_coords = len(current_geom)
        candidates = []
        
        # Negative gradient direction
        if np.linalg.norm(gradient) > 1e-10:
            unit_grad = gradient / np.linalg.norm(gradient)
            neg_grad_step = -0.1 * self.candidate_scale * unit_grad
            candidates.append(current_geom + neg_grad_step)
        
        # Steps from previous history
        if len(self.step_history) > 0:
            for i in range(min(3, len(self.step_history))):
                scale = np.random.uniform(0.8, 1.2) * self.candidate_scale
                candidates.append(current_geom + scale * self.step_history[-(i+1)])
        
        # Random steps
        n_random = max(1, self.num_candidates - len(candidates))
        step_scale = 0.1 * self.candidate_scale
        if len(self.step_history) > 0:
            step_scale = np.mean([np.linalg.norm(s) for s in self.step_history[-3:]]) * self.candidate_scale
        
        for _ in range(n_random):
            rand_dir = np.random.randn(*current_geom.shape)
            rand_dir = rand_dir / max(1e-10, np.linalg.norm(rand_dir))
            scale = np.random.uniform(0.5, 1.5) * step_scale
            candidates.append(current_geom + scale * rand_dir)
        
        return candidates
    
    def _select_best_candidate(self, candidates, current_geom, current_energy):
        """Select best candidate based on GPR predictions"""
        if self.gpr is None or not candidates:
            return None, 0.0
        
        try:
            # Prepare candidates for GPR
            X_cand = np.array([g.flatten() for g in candidates])
            
            # Get predictions with uncertainty
            y_mean, y_std = self.gpr.predict(X_cand, return_std=True)
            
            # Denormalize predictions
            y_mean = y_mean * self.y_std + self.y_mean
            y_std = y_std * self.y_std
            
            # Acquisition function (Upper Confidence Bound)
            acquisition = -(y_mean - self.ucb_beta * self.balance_factor * y_std)
            
            # Select best candidate
            best_idx = np.argmax(acquisition)
            best_geom = candidates[best_idx]
            best_step = best_geom - current_geom
            
            # Expected improvement
            expected_improvement = current_energy - y_mean[best_idx]
            print(f"Expected energy: {y_mean[best_idx]:.6f}, uncertainty: {y_std[best_idx]:.6f}")
            print(f"Expected improvement: {expected_improvement:.6f}")
            
            return best_step, expected_improvement
            
        except Exception as e:
            print(f"Error in candidate selection: {str(e)}")
            self.failures += 1
            return None, 0.0
    
    def run(self, geom_num_list, energy, gradient, original_move_vector):
        """Run GPR-based optimization step"""
        print("GPR-Step optimization method")
        n_coords = len(geom_num_list)
        grad_rms = np.sqrt(np.mean(gradient ** 2))
        
        print(f"Energy: {energy:.8f}, Gradient RMS: {grad_rms:.8f}")
        
        # Update histories
        self._update_histories(geom_num_list, energy, gradient)
        
        # Use original step if not enough history
        if len(self.geom_history) < self.min_points_for_gpr:
            print(f"Building history ({len(self.geom_history)}/{self.min_points_for_gpr}), using original step")
            self.iter += 1
            return original_move_vector
        
        # Use original step if too many GPR failures
        if self.failures >= 3:
            print(f"Too many GPR failures ({self.failures}), using original step")
            self.iter += 1
            return original_move_vector
        
        # Fit GPR model
        if not self._fit_gpr_model():
            print("Failed to fit GPR model, using original step")
            self.iter += 1
            return original_move_vector
        
        # Generate candidate steps
        candidates = self._generate_candidate_steps(geom_num_list, gradient)
        print(f"Generated {len(candidates)} candidate steps")
        
        # Select best step
        gpr_step, expected_improvement = self._select_best_candidate(
            candidates, geom_num_list, energy)
        
        if gpr_step is None or expected_improvement < 0:
            print("GPR step selection failed or predicted energy increase, using original step")
            self.iter += 1
            return original_move_vector
        
        # Blend steps
        orig_norm = np.linalg.norm(original_move_vector)
        gpr_norm = np.linalg.norm(gpr_step)
        
        # Scale GPR step if it's too large
        if gpr_norm > self.max_step_norm:
            gpr_step = gpr_step * (self.max_step_norm / gpr_norm)
            gpr_norm = self.max_step_norm
        
        # Blend with original step
        if orig_norm > 1e-10:
            # Check if directions are similar
            cos_angle = np.dot(original_move_vector.flatten(), gpr_step.flatten()) / (orig_norm * gpr_norm)
            
            if cos_angle > 0.5:  # Directions are similar
                weight_gpr = 0.7
            elif cos_angle > 0:   # Somewhat aligned
                weight_gpr = 0.5
            else:                 # Different directions
                weight_gpr = 0.3
                
            # Ensure GPR step is not much larger than original
            if gpr_norm > 3.0 * orig_norm:
                gpr_step = gpr_step * (3.0 * orig_norm / gpr_norm)
                
            blended_step = -1 * weight_gpr * gpr_step + (1.0 - weight_gpr) * original_move_vector
            print(f"Using blended step: {weight_gpr:.2f}*GPR + {1.0-weight_gpr:.2f}*Original")
        else:
            blended_step = -1 * gpr_step
            print("Using pure GPR step (original step near zero)")
        
        # Final safety checks
        if np.any(np.isnan(blended_step)) or np.any(np.isinf(blended_step)):
            print("Numerical issues detected, using original step")
            self.iter += 1
            return original_move_vector
        
        self.iter += 1
        return blended_step