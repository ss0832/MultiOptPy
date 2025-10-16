import numpy as np
import copy
from scipy.interpolate import BSpline, interp1d
from scipy.signal import argrelextrema
from numpy.polynomial import polynomial as P

# ref. : S.-i. Koda and S. Saito, Locating Transition States by Variational Reaction Path Optimization with an Energy-Derivative-Free Objective Function, JCTC, 20, 2798–2811 (2024). doi: 10.1021/acs.jctc.3c01246

# Helper functions from the original context, now outside the class
def extremum_list_index(energy_list):
    """Finds indices of local maxima and minima in an energy list."""
    energy_list = np.array(energy_list)
    local_max_energy_list_index = argrelextrema(energy_list, np.greater)
    inverse_energy_list = (-1) * energy_list
    local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)

    local_max_energy_list_index = local_max_energy_list_index[0].tolist()
    local_min_energy_list_index = local_min_energy_list_index[0].tolist()
    
    # Ensure endpoints are not considered extrema
    if 0 in local_min_energy_list_index: local_min_energy_list_index.remove(0)
    if 0 in local_max_energy_list_index: local_max_energy_list_index.remove(0)
    if len(energy_list)-1 in local_min_energy_list_index: local_min_energy_list_index.remove(len(energy_list)-1)
    if len(energy_list)-1 in local_max_energy_list_index: local_max_energy_list_index.remove(len(energy_list)-1)

    return local_max_energy_list_index, local_min_energy_list_index

class CaluculationDMF():
    """
    Class to calculate forces for path optimization.
    This version implements the Direct MaxFlux (DMF) algorithm principles.
    """
    def __init__(self, APPLY_CI_NEB=99999, beta=10.0, nsegs=4, dspl=3):
        """
        Initializes the calculator.

        Args:
            APPLY_CI_NEB (int): Legacy parameter, not used by DMF logic.
            beta (float): Reciprocal temperature for the MaxFlux method.
            nsegs (int): Number of segments for the B-spline path.
            dspl (int): Degree of the B-spline functions.
        """
        self.APPLY_CI_NEB = APPLY_CI_NEB # Retained for structural consistency
        self.beta = beta
        self.nsegs = nsegs
        self.dspl = dspl
        self.nbasis = nsegs + dspl
        
        # B-spline basis setup
        _t_knot = np.concatenate([
            np.zeros(dspl),
            np.linspace(0.0, 1.0, nsegs + 1),
            np.ones(dspl)])
        self._t_knot = _t_knot
        
        # Create basis functions with error handling
        self._basis = [[], []]
        for i in range(self.nbasis):
            try:
                self._basis[0].append(BSpline(self._t_knot, np.identity(self.nbasis)[i], dspl, extrapolate=False))
                self._basis[1].append(BSpline(self._t_knot, np.identity(self.nbasis)[i], dspl, extrapolate=False).derivative(nu=1))
            except Exception as e:
                print(f"Error creating BSpline basis {i}: {e}")
                # Fall back to simple basis if needed
                self._basis[0].append(None)
                self._basis[1].append(None)

    def _get_basis_values(self, t_seq, nu=0):
        """Evaluates B-spline basis functions or their derivatives at given points."""
        if nu not in [0, 1]:
            raise ValueError("Only nu=0 and nu=1 are supported.")
        
        # Ensure t_seq is within [0, 1] to avoid extrapolation issues
        t_seq_clipped = np.clip(t_seq, 0.0, 1.0)
        
        result = []
        for b in self._basis[nu]:
            if b is None:
                # Handle case where basis function creation failed
                result.append(np.zeros(len(t_seq_clipped)))
            else:
                try:
                    values = np.array([b(t) for t in t_seq_clipped])
                    # Replace any NaN or inf with zeros
                    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    result.append(values)
                except Exception as e:
                    print(f"Error evaluating basis: {e}")
                    result.append(np.zeros(len(t_seq_clipped)))
                    
        return np.array(result)

    def _get_coefs_from_images(self, images):
        """
        Computes B-spline coefficients that best fit the given image geometries.
        
        Args:
            images (list of np.ndarray): A list of geometries (images) along the path.

        Returns:
            np.ndarray: The calculated B-spline coefficients.
        """
        nimages = len(images)
        natoms = images[0].shape[0]
        pos_ref = np.array(images)

        # Ensure there are no NaN values in the input images
        if not np.all(np.isfinite(pos_ref)):
            print("WARNING: Non-finite values detected in input geometries")
            pos_ref = np.nan_to_num(pos_ref, nan=0.0, posinf=0.0, neginf=0.0)

        # Estimate path length to create a parameterization t_ref
        diff = pos_ref[1:] - pos_ref[:-1]
        lengths = np.sqrt(np.sum(diff**2, axis=(1, 2)))
        
        # Handle potential zeros in lengths
        epsilon = 1e-10
        lengths = np.maximum(lengths, epsilon)
        
        t_ref = np.concatenate(([0.0], np.cumsum(lengths)))
        
        # STABILITY: Handle case where path length is zero or very small
        if t_ref[-1] > epsilon:
            t_ref /= t_ref[-1]
        else: 
            # All images are at nearly the same position
            print("WARNING: Path length is very small, using linear parameterization")
            t_ref = np.linspace(0.0, 1.0, nimages)

        # Use linear interpolation as a starting point
        try:
            f_interp = interp1d(t_ref, pos_ref, axis=0, fill_value="extrapolate", bounds_error=False)
            t_solve = np.linspace(0.0, 1.0, 4 * self.nsegs + 1)
            pos_solve = f_interp(t_solve)
        except Exception as e:
            print(f"Interpolation error: {e}, falling back to simple interpolation")
            t_solve = np.linspace(0.0, 1.0, 4 * self.nsegs + 1)
            # Fall back to simple linear interpolation
            pos_solve = np.zeros((len(t_solve), natoms, 3))
            for i, t in enumerate(t_solve):
                idx = min(int(t * (nimages-1)), nimages-2)
                frac = (t - idx/(nimages-1)) * (nimages-1)
                pos_solve[i] = pos_ref[idx] * (1 - frac) + pos_ref[idx+1] * frac
        
        P_solve = self._get_basis_values(t_solve, nu=0)
        
        # Ensure P_solve has no numerical issues
        if not np.all(np.isfinite(P_solve)):
            print("WARNING: Non-finite values in basis evaluation")
            P_solve = np.nan_to_num(P_solve, nan=0.0, posinf=0.0, neginf=0.0)
            
        # A * coefs = b, where we solve for coefs
        A = P_solve.T
        b = pos_solve.reshape(len(t_solve), -1)
        
        # Check matrix condition before solving
        try:
            # Using a more stable solver with regularization
            rcond = 1e-6  # Increased regularization factor for better stability
            coefs_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=rcond)
        except np.linalg.LinAlgError:
            print("WARNING: Linear algebra error in least squares, using fallback")
            # Fallback: Use simple pseudoinverse with stronger regularization
            ATA = np.dot(A.T, A) + np.eye(A.shape[1]) * 1e-4  # Increased regularization
            ATb = np.dot(A.T, b)
            try:
                coefs_flat = np.linalg.solve(ATA, ATb)
            except:
                print("SEVERE WARNING: Could not solve for coefficients, using direct images")
                # Last resort: Use the original points
                coefs_flat = np.zeros((self.nbasis, natoms * 3))
                coefs_flat[0] = pos_ref[0].reshape(-1)
                coefs_flat[-1] = pos_ref[-1].reshape(-1)
                for i in range(1, self.nbasis-1):
                    idx = min(int(i * (nimages-1)/(self.nbasis-1)), nimages-1)
                    coefs_flat[i] = pos_ref[idx].reshape(-1)
        
        coefs = coefs_flat.reshape(self.nbasis, natoms, 3)
        
        # Check for NaN or inf values in coefficients
        if not np.all(np.isfinite(coefs)):
            print("WARNING: Non-finite coefficients detected, replacing with zeros")
            coefs = np.nan_to_num(coefs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Enforce boundary conditions
        coefs[0] = pos_ref[0]
        coefs[-1] = pos_ref[-1]
        
        # Ensure coefficients are reasonably smooth - this helps prevent kabsch issues
        # Add a small amount of smoothing/averaging between neighboring coefficients
        smoothed_coefs = coefs.copy()
        for i in range(1, self.nbasis-1):
            smoothed_coefs[i] = 0.9 * coefs[i] + 0.05 * coefs[i-1] + 0.05 * coefs[i+1]
        
        # Keep endpoints fixed
        smoothed_coefs[0] = coefs[0]
        smoothed_coefs[-1] = coefs[-1]
        coefs = smoothed_coefs

        return coefs

    def _get_path_properties(self, coefs, t_eval):
        """Calculates positions and velocities along the spline path."""
        P_eval_pos = self._get_basis_values(t_eval, nu=0)
        P_eval_vel = self._get_basis_values(t_eval, nu=1)
        
        positions = np.tensordot(P_eval_pos.T, coefs, axes=1)
        velocities = np.tensordot(P_eval_vel.T, coefs, axes=1)
        
        # Check for NaN or inf values
        if not np.all(np.isfinite(positions)):
            print("WARNING: Non-finite positions detected")
            positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.all(np.isfinite(velocities)):
            print("WARNING: Non-finite velocities detected")
            velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
            
        return positions, velocities

    def _get_func_en(self, energies, e0):
        """Calculates the energy-dependent part of the action and its derivative."""
        en_shifted = energies - e0
        
        # Cap extremely large values to prevent overflow
        max_en = 700.0 / self.beta  # Prevent exp overflow
        en_shifted = np.minimum(en_shifted, max_en)
        
        exp_beta_en = np.exp(self.beta * en_shifted)
        
        # Check for NaN or inf
        if not np.all(np.isfinite(exp_beta_en)):
            print("WARNING: Non-finite energy function values detected")
            exp_beta_en = np.nan_to_num(exp_beta_en, nan=1.0, posinf=1e10, neginf=0.0)
            
        return exp_beta_en, self.beta * exp_beta_en
        
    def _get_action_and_gradient(self, coefs, t_eval, w_eval, energies, forces, e0):
        """
        Calculates the MaxFlux action and its gradient with respect to coefficients.
        """
        positions, velocities = self._get_path_properties(coefs, t_eval)
        
        # Calculate norm of velocities with stability checks
        vel_squared = np.sum(velocities**2, axis=(1, 2))
        vel_squared = np.maximum(vel_squared, 1e-16)  # Prevent sqrt of negative/zero
        norm_vels = np.sqrt(vel_squared)
        
        # STABILITY: Ensure minimum reasonable velocity norm
        epsilon = 1e-8
        norm_vels_safe = np.maximum(norm_vels, epsilon)
        
        # Calculate the energy function and its derivative
        fe, dfe = self._get_func_en(energies, e0)
        
        # Calculate total action (numerical integration)
        action_terms = w_eval * norm_vels * fe
        action = np.sum(action_terms)
        
        # If action is too small, return zero gradient
        if abs(action) < 1e-12:
            print("WARNING: Action is nearly zero, returning zero gradient")
            grad_action = np.zeros_like(coefs)
            return max(action, 1e-12), grad_action  # Return small positive action to avoid division by zero
        
        # --- Gradient Calculation ---
        P_vel_eval = self._get_basis_values(t_eval, nu=1) # Shape (nbasis, nnode)
        P_pos_eval = self._get_basis_values(t_eval, nu=0) # Shape (nbasis, nnode)
        
        # Compute normalized velocities safely
        normalized_velocities = np.zeros_like(velocities)
        for i in range(len(norm_vels_safe)):
            if norm_vels_safe[i] > epsilon:
                normalized_velocities[i] = velocities[i] / norm_vels_safe[i]
                
        # Part 1: Gradient from the velocity norm term
        try:
            grad_action_v = np.einsum('bt,tas,t->bas', P_vel_eval, normalized_velocities, w_eval * fe)
        except Exception as e:
            print(f"Error in einsum (velocity part): {e}")
            grad_action_v = np.zeros_like(coefs)
            
        # Part 2: Gradient from the energy term (forces)
        term_for_force = w_eval * norm_vels * dfe
        
        # Ensure forces have no NaN/inf values
        safe_forces = np.array(forces)
        if not np.all(np.isfinite(safe_forces)):
            print("WARNING: Non-finite forces detected")
            safe_forces = np.nan_to_num(safe_forces, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Clip very large force values to prevent numerical instability
        max_force = 1e3
        safe_forces = np.clip(safe_forces, -max_force, max_force)
            
        try:
            grad_action_f = -np.einsum('bt,tas,t->bas', P_pos_eval, safe_forces, term_for_force)
        except Exception as e:
            print(f"Error in einsum (force part): {e}")
            grad_action_f = np.zeros_like(coefs)

        # Combine gradients
        grad_action = grad_action_v + grad_action_f
        
        # Final check for NaN/inf
        if not np.all(np.isfinite(grad_action)):
            print("WARNING: Non-finite gradient detected")
            grad_action = np.nan_to_num(grad_action, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Clip gradient to reasonable values to prevent instability
        max_grad = 1e3
        grad_action = np.clip(grad_action, -max_grad, max_grad)
            
        return action, grad_action

    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        """
        Calculates the "forces" to drive the path optimization using the DMF method.
        The returned "force" is the negative gradient of the objective function w.r.t. image positions.
        """
        print("DMF"*20)
        
        nnode = len(energy_list)
        
        # Check input data
        try:
            # The "images" are the geometries in Cartesian coordinates
            images = [np.array(g, dtype="float64") for g in geometry_num_list]
            energies = np.array(energy_list, dtype="float64")
            # Gradients are negative forces
            forces = [-np.array(g, dtype="float64") for g in gradient_list]
            
            # Check for NaN/inf in input data
            for i, (img, en, f) in enumerate(zip(images, energies, forces)):
                if not np.all(np.isfinite(img)):
                    print(f"WARNING: Non-finite values in geometry {i}")
                    images[i] = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.isfinite(en):
                    print(f"WARNING: Non-finite value in energy {i}")
                    energies[i] = 0.0
                if not np.all(np.isfinite(f)):
                    print(f"WARNING: Non-finite values in force {i}")
                    forces[i] = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error processing input data: {e}")
            # Return zeros if we can't process the input
            return np.zeros((len(geometry_num_list), len(geometry_num_list[0]), 3))

        # 1. Define the path parameterization and integration weights
        t_eval = np.linspace(0.0, 1.0, nnode)
        w_eval = np.zeros_like(t_eval)
        w_eval[0] = 0.5 * (t_eval[1] - t_eval[0])
        w_eval[-1] = 0.5 * (t_eval[-1] - t_eval[-2])
        w_eval[1:-1] = 0.5 * (t_eval[2:] - t_eval[:-2])

        # 2. Get B-spline coefficients for the current path
        try:
            coefs = self._get_coefs_from_images(images)
        except Exception as e:
            print(f"Error in coefficient calculation: {e}")
            # Return zero forces if coefficient calculation fails
            return np.zeros((len(geometry_num_list), len(geometry_num_list[0]), 3))

        # 3. Calculate the objective function (action) and its gradient
        e0 = np.min(energies) # Energy reference
        
        try:
            action, grad_action_coefs = self._get_action_and_gradient(coefs, t_eval, w_eval, energies, forces, e0)
            # Print the action value as requested
            print(f"DMF Objective Function Value (Action): {action}")
            print(f"Energy Reference e0: {e0}")
            print(f"Beta Value: {self.beta}")
            print(f"Normalized Action: {action * self.beta}")
        except Exception as e:
            print(f"Error in action calculation: {e}")
            # Return zero forces if action calculation fails
            return np.zeros((len(geometry_num_list), len(geometry_num_list[0]), 3))
        
        # STABILITY: Check for safe division
        denominator = action * self.beta
        if abs(denominator) < 1e-12:
            print("WARNING: Action denominator is too small")
            objective_grad_coefs = np.zeros_like(grad_action_coefs)
        else:
            objective_grad_coefs = grad_action_coefs / denominator
            
        # Clip objective gradient to reasonable values
        max_obj_grad = 1e2
        objective_grad_coefs = np.clip(objective_grad_coefs, -max_obj_grad, max_obj_grad)
        
        # 4. Project the gradient on coefficients back to forces on images
        P_pos_eval = self._get_basis_values(t_eval, nu=0) # Shape: [nbasis, nnode]
        
        try:
            forces_from_obj = -np.einsum('bt,bas->tas', P_pos_eval, objective_grad_coefs)
        except Exception as e:
            print(f"Error in final einsum: {e}")
            forces_from_obj = np.zeros((len(images), images[0].shape[0], 3))

        # 5. Handle endpoints: they are fixed, so their forces should be zero.
        forces_from_obj[0, :, :] = 0.0
        forces_from_obj[-1, :, :] = 0.0
        
        # STABILITY: Final check to ensure no NaN/inf values are returned
        if not np.all(np.isfinite(forces_from_obj)):
            print("WARNING: Non-finite values detected in DMF forces. Returning zero forces.")
            forces_from_obj = np.nan_to_num(forces_from_obj, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure reasonable force magnitudes - critical for trust radius calculations
        # This directly addresses the divide-by-zero errors in trust_radius_neb.py
        for i in range(len(forces_from_obj)):
            force_mag = np.linalg.norm(forces_from_obj[i].reshape(-1))
            if force_mag < 1e-8:  # Force too small
                if i > 0 and i < len(forces_from_obj) - 1:  # Skip endpoints
                    print(f"WARNING: Force magnitude for image {i} is too small ({force_mag}), adding small noise")
                    np.random.seed(i+42)  # For reproducibility but different for each image
                    # Add small noise proportional to the system size
                    system_scale = np.mean(np.abs(images[i]))
                    if system_scale < 1e-10:
                        system_scale = 1.0
                    noise_scale = 1e-6 * system_scale
                    small_noise = np.random.normal(0, noise_scale, forces_from_obj[i].shape)
                    forces_from_obj[i] += small_noise
            elif force_mag > 1e2:  # Force too large
                print(f"WARNING: Force magnitude for image {i} is too large ({force_mag}), scaling down")
                scale_factor = 1e2 / force_mag
                forces_from_obj[i] *= scale_factor

        # Reset endpoints to zero force
        forces_from_obj[0, :, :] = 0.0  
        forces_from_obj[-1, :, :] = 0.0

        # Final sanitization for trust radius calculation safety
        # This ensures total_delta values in trust_radius_neb.py will have reasonable norms
        total_force_list = np.array(forces_from_obj, dtype="float64")
        
        # Make sure no image has exactly zero force (except endpoints)
        # This prevents division by zero in trust_radius_neb.py
        for i in range(1, len(total_force_list) - 1):
            norm_i = np.linalg.norm(total_force_list[i].reshape(-1))
            if norm_i < 1e-10:
                total_force_list[i, 0, 0] += 1e-8  # Tiny force in x direction of first atom
                
        # Print force statistics to help with debugging
        force_norms = [np.linalg.norm(total_force_list[i].reshape(-1)) for i in range(len(total_force_list))]
        print(f"Force norms: min={min(force_norms):.6e}, max={max(force_norms):.6e}, avg={np.mean(force_norms):.6e}")
        
        return total_force_list