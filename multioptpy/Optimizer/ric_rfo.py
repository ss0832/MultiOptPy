import numpy as np
from scipy import linalg
from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib
from .hessian_update import ModelHessianUpdate

"""
Redundant Internal Coordinate RFO implementation
Implementation without dihedral angles in primitive coordinates
"""

class RedundantInternalRFO:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        
        # Initialize RIC specific parameters
        self.Initialization = True
        self.saddle_order = self.config.get("saddle_order", 0)
        self.trust_radius = float(self.config.get("trust_radius", 0.1))
        self.trust_radius /= UnitValueLib().bohr2angstroms
        self.DELTA = 0.1  # Step scale factor
        
        # For internal coordinate handling
        self.B_matrix = None      # Wilson B matrix (transformation matrix)
        self.G_matrix = None      # G matrix (metric matrix)
        self.primitive_coords = None
        
        # RIC specific parameters
        self.redundant_thresh = 1e-5   # Threshold for numerical stability
        self.coord_type = self.config.get("coord_type", "redundant")
        self.iter = 0
        self.max_backsteps = 200
        self.max_micro_cycles = 1000
        self.FC_COUNT = self.config.get("FC_COUNT", -1)
        self.backconv_method = self.config.get("backconv_method", "scf").lower()
        
        # Tracking variables
        self.prev_cartesian = None
        self.prev_gradient = None
        self.internal_hessian = None
        self.hessian = None
        self.bias_hessian = None
        self.prev_connectivity = None

        self.element_list = self.config.get("element_list", None)
    
    def define_internal_coordinates(self, geometry):
        """
        Define primitive internal coordinates based on molecular connectivity
        Returns the connectivity table and primitive internal coordinates
        Using only bonds and angles (no dihedrals as requested)
        """
        atomic_numbers = self.element_list
        # Convert geometry from (3N, 1) to (N, 3) format
        natoms = len(atomic_numbers)
        geom_reshaped = geometry.reshape(natoms, 3)
        
        primitive_coords = []
        connectivity = np.zeros((natoms, natoms), dtype=int)
        
        # Build connectivity based on covalent radii
        for i in range(natoms):
            for j in range(i+1, natoms):
                # Calculate distance between atoms
                dist = np.linalg.norm(geom_reshaped[i] - geom_reshaped[j])
                
                # Get covalent radii based on atomic numbers
                r_i = covalent_radii_lib(atomic_numbers[i])
                r_j = covalent_radii_lib(atomic_numbers[j])
                
                # Check if atoms are bonded (using a bond tolerance factor of 1.2)
                if dist < 1.2 * (r_i + r_j):
                    connectivity[i, j] = 1
                    connectivity[j, i] = 1
                    
                    # Add bond as primitive coordinate
                    primitive_coords.append({
                        'type': 'bond',
                        'atoms': [i, j]
                    })
        
        # Add angle coordinates
        for j in range(natoms):
            bonded_to_j = np.where(connectivity[j] > 0)[0]
            for i in bonded_to_j:
                for k in bonded_to_j:
                    if i < k:  # Avoid duplicates
                        primitive_coords.append({
                            'type': 'angle',
                            'atoms': [i, j, k]
                        })
        
        # Note: As requested, no dihedral angles are added
        
        self.primitive_coords = primitive_coords
        return connectivity, primitive_coords



    def calc_center(self, geometry):
        """Calculate center of mass of the geometry"""
        n_atoms = len(geometry) // 3
        return np.mean(geometry.reshape(n_atoms, 3), axis=0)
    
    def build_B_matrix(self, geometry, primitive_coords=None):
        """
        Build Wilson B matrix (∂q/∂x) for coordinate transformation
        q: internal coordinates
        x: Cartesian coordinates
        
        The geometry is expected in (3N, 1) format
        """
        if primitive_coords is None:
            primitive_coords = self.primitive_coords
            
        natoms = len(geometry) // 3
        geom_reshaped = geometry.reshape(natoms, 3)
        ncoords = len(primitive_coords)
        
        B = np.zeros((ncoords, 3*natoms))
        
        for i, coord in enumerate(primitive_coords):
            if coord['type'] == 'bond':
                a1, a2 = coord['atoms']
                B[i] = self._bond_B_elements(geom_reshaped, a1, a2)
            elif coord['type'] == 'angle':
                a1, a2, a3 = coord['atoms']
                B[i] = self._angle_B_elements(geom_reshaped, a1, a2, a3)
      
        return B
    
    def _bond_B_elements(self, geometry, a1, a2):
        """Calculate B matrix elements for bond stretching"""
        r1 = geometry[a1]
        r2 = geometry[a2]
        
        # Vector pointing from atom a1 to atom a2
        vec = r2 - r1
        
        # Distance between the atoms
        distance = np.linalg.norm(vec)
        
        if distance < 1e-10:
            unit_vec = np.zeros(3)
        else:
            unit_vec = vec / distance
            
        # B matrix elements (derivatives of the bond distance w.r.t. Cartesian coordinates)
        B_elements = np.zeros(3 * len(geometry))
        
        # For atom a1
        B_elements[3*a1:3*a1+3] = -unit_vec
        
        # For atom a2
        B_elements[3*a2:3*a2+3] = unit_vec
        
        return B_elements
    
    def _angle_B_elements(self, geometry, a1, a2, a3):
        """Calculate B matrix elements for angle bending"""
        r1 = geometry[a1]
        r2 = geometry[a2]
        r3 = geometry[a3]
        
        # Vectors from central atom to the other two atoms
        v1 = r1 - r2
        v3 = r3 - r2
        
        # Normalize vectors
        d1 = np.linalg.norm(v1)
        d3 = np.linalg.norm(v3)
        
        if d1 < 1e-10 or d3 < 1e-10:
            return np.zeros(3 * len(geometry))
            
        v1_normalized = v1 / d1
        v3_normalized = v3 / d3
        
        # Cosine of the angle
        cos_angle = np.dot(v1_normalized, v3_normalized)
        
        # Ensure numerical stability
        if cos_angle > 1.0:
            cos_angle = 1.0
        elif cos_angle < -1.0:
            cos_angle = -1.0
            
        sin_angle = np.sqrt(1.0 - cos_angle**2)
        
        if sin_angle < 1e-10:
            return np.zeros(3 * len(geometry))
            
        # Calculate derivatives
        term1 = 1.0 / (d1 * sin_angle)
        term3 = 1.0 / (d3 * sin_angle)
        
        # Cross products
        cp1 = np.cross(v1_normalized, v3_normalized)
        cp3 = np.cross(v3_normalized, v1_normalized)
        
        # B matrix elements
        B_elements = np.zeros(3 * len(geometry))
        
        # For atom a1
        B_elements[3*a1:3*a1+3] = term1 * cp1
        
        # For atom a3
        B_elements[3*a3:3*a3+3] = term3 * cp3
        
        # For central atom a2
        B_elements[3*a2:3*a2+3] = -(B_elements[3*a1:3*a1+3] + B_elements[3*a3:3*a3+3])
        
        return B_elements
    
    def build_G_matrix(self, B_matrix):
        """
        Build G matrix (metric matrix) from Wilson B matrix
        G = B·B^T
        """
        G = np.dot(B_matrix, B_matrix.T)
        return G
    
    def check_connectivity_change(self, geometry):
        """
        Check if the molecular connectivity has changed significantly
        """
        if self.prev_connectivity is None:
            # First call, just store the current connectivity
            connectivity, _ = self.define_internal_coordinates(geometry)
            self.prev_connectivity = connectivity
            return False
        
        # Check current connectivity
        current_connectivity, _ = self.define_internal_coordinates(geometry)
        
        # Compare with previous connectivity matrix
        diff = np.sum(np.abs(current_connectivity - self.prev_connectivity))
        
        # If there are changes in connectivity, reset coordinate system
        if diff > 0:
            print(f"WARNING: Detected {diff} changes in molecular connectivity")
            print("Resetting internal coordinates due to bond rearrangement")
            self.prev_connectivity = current_connectivity
            return True
        
        return False
    
    def update_coordinates(self, geometry):
        """
        Update the internal coordinates for the current geometry
        """
        # Build B matrix for current geometry
        B = self.build_B_matrix(geometry)
        self.B_matrix = B
        
        # Build G matrix
        G = self.build_G_matrix(B)
        self.G_matrix = G
        
        return B
    
    def transform_gradient(self, cart_gradient):
        """
        Transform Cartesian gradient to internal coordinate gradient
        g_int = B·g_cart (direct transformation using Wilson B matrix)
        """
        B = self.B_matrix
        
        # Direct transformation using B matrix
        int_gradient = np.dot(B, cart_gradient)
        return int_gradient
    
    def transform_hessian(self, cart_hessian):
        """
        Transform Cartesian Hessian to internal coordinate Hessian
        H_int = B·H_cart·B^T (direct transformation)
        """
        B = self.B_matrix
        
        # Direct transformation
        int_hessian = np.dot(B, np.dot(cart_hessian, B.T))
        return int_hessian
    
    def internal_to_cartesian_scf(self, step_int, geometry, 
                                max_iterations=100,
                                convergence_threshold=1e-8):
        """
        SCF-inspired algorithm for internal to Cartesian coordinate conversion.
        Uses techniques from electronic structure theory: line search when far from
        convergence and Newton-Raphson near convergence, with DIIS acceleration.
        
        Parameters:
        -----------
        step_int : ndarray
            Target step in internal coordinates
        geometry : ndarray
            Starting geometry in Cartesian coordinates
        max_iterations : int
            Maximum number of iterations
        convergence_threshold : float
            Convergence criterion for RMS error
            
        Returns:
        --------
        ndarray
            Step in Cartesian coordinates
        """
        # Initialize variables
        reference_geom = geometry.copy()
        current_geom = geometry.copy()
        best_geom = current_geom.copy()
        best_error = float('inf')
        
        # DIIS parameters
        diis_start = 3
        diis_max = 8
        diis_errors = []
        diis_geometries = []
        
        # Dynamic algorithm control
        use_newton = False  # Start with line search, switch to Newton later
        damping_factor = 0.7  # Initial damping
        level_shift = 0.0  # Level shifting parameter
        
        # Last iteration data for backtracks
        prev_error = float('inf')
        prev_geom = None
        
        # Calculate initial values
        B = self.build_B_matrix(current_geom)
        current_q = np.dot(B, current_geom - reference_geom)
        delta_q = step_int - current_q
        error = np.linalg.norm(delta_q)
        rms_error = error / np.sqrt(len(delta_q))
        
        print(f"Starting SCF-like optimization: initial RMS error = {rms_error:.3e}")
        
        # Main iteration loop
        for iteration in range(max_iterations):
            # Save best solution
            if error < best_error:
                best_error = error
                best_geom = current_geom.copy()
            
            # Report progress
            if iteration % 20 == 0 or iteration < 2:
                print(f"Iteration {iteration}: RMS error = {rms_error:.3e}, "
                    f"{'Newton' if use_newton else 'LineSearch'}, damping={damping_factor:.2f}")
            
            # Check convergence
            if rms_error < convergence_threshold:
                print(f"Conversion converged in {iteration+1} iterations")
                break
            
            # Prepare for this iteration
            B = self.build_B_matrix(current_geom)
            
            # Calculate step based on current method
            if use_newton:
                # Calculate B+ (pseudoinverse of B) using SVD with level shifting
                try:
                    U, s, Vh = np.linalg.svd(B, full_matrices=False)
                    
                    # Apply level shifting to singular values (similar to SCF level shifting)
                    s_inv = np.where(s > 1e-7, 1.0 / (s + level_shift), 0.0)
                    
                    B_pinv = np.dot(Vh.T * s_inv, U.T)
                    
                    # Calculate Newton step
                    step = np.dot(B_pinv, delta_q)
                except np.linalg.LinAlgError:
                    # Fall back to a more stable approach
                    print("SVD failed, using pinv with increased regularization")
                    B_pinv = np.linalg.pinv(B, rcond=1e-6)
                    step = np.dot(B_pinv, delta_q)
            else:
                # Line search along steepest descent direction
                gradient = np.dot(B.T, delta_q)
                grad_norm = np.linalg.norm(gradient)
                
                if grad_norm < 1e-10:
                    # Gradient too small, try Newton step
                    use_newton = True
                    B_pinv = np.linalg.pinv(B, rcond=1e-7)
                    step = np.dot(B_pinv, delta_q)
                else:
                    # Normalize gradient and scale by dynamic step size
                    step_dir = gradient / grad_norm
                    
                    # Determine step size (larger when further from convergence)
                    # Similar to trust radius in SCF
                    step_size = min(0.2, 0.1 * (1.0 + 10.0 * rms_error))
                    
                    step = step_size * step_dir
            
            # Apply DIIS acceleration if we have enough iterations
            diis_step = None
            if iteration >= diis_start:
                # Store current error and geometry for DIIS
                diis_errors.append(delta_q.flatten())
                diis_geometries.append(current_geom.copy())
                
                # Limit DIIS vector storage
                if len(diis_errors) > diis_max:
                    diis_errors.pop(0)
                    diis_geometries.pop(0)
                
                # Apply DIIS if we have at least 2 vectors
                if len(diis_errors) >= 2:
                    try:
                        diis_step = self._compute_diis_solution(diis_errors, diis_geometries)
                    except Exception as e:
                        print(f"DIIS failed: {e}")
                        diis_step = None
            
            # Try step with current damping
            damped_step = step * damping_factor
            trial_geom = current_geom + damped_step
            
            # Evaluate trial geometry
            trial_B = self.build_B_matrix(trial_geom)
            trial_q = np.dot(trial_B, trial_geom - reference_geom)
            trial_delta_q = step_int - trial_q
            trial_error = np.linalg.norm(trial_delta_q)
            trial_rms = trial_error / np.sqrt(len(trial_delta_q))
            
            # If we have a DIIS solution, evaluate it too
            if diis_step is not None:
                diis_B = self.build_B_matrix(diis_step)
                diis_q = np.dot(diis_B, diis_step - reference_geom)
                diis_delta_q = step_int - diis_q
                diis_error = np.linalg.norm(diis_delta_q)
                diis_rms = diis_error / np.sqrt(len(diis_delta_q))
                
                # Use DIIS if it's better
                if diis_error < trial_error and diis_error < error:
                    #print(f"DIIS improvement: {trial_rms:.3e} -> {diis_rms:.3e}")
                    current_geom = diis_step
                    delta_q = diis_delta_q
                    error = diis_error
                    rms_error = diis_rms
                    
                    # DIIS was successful, try to reduce level shifting
                    if level_shift > 0:
                        level_shift = max(0, level_shift * 0.5)
                    continue  # Skip to next iteration
            
            # Dynamically adjust the algorithm based on progress
            if trial_error < error:
                # Step is good, accept it
                current_geom = trial_geom
                delta_q = trial_delta_q
                prev_error = error
                error = trial_error
                rms_error = trial_rms
                
                # Increase damping for more aggressive steps
                damping_factor = min(1.0, damping_factor * 1.2)
                
                # Switch to Newton method when close enough to solution
                if not use_newton and rms_error < 0.1:
                    #print(f"Switching to Newton-Raphson at iteration {iteration+1}")
                    use_newton = True
                    
                # Reduce level shifting since things are going well
                if level_shift > 0:
                    level_shift = max(0, level_shift * 0.5)
                    
            else:
                # Step is bad, adjust strategy
                if use_newton:
                    # Newton step made things worse
                    if level_shift == 0:
                        # Start with modest level shifting
                        level_shift = 0.1
                    else:
                        # Increase level shifting (similar to SCF when oscillating)
                        level_shift = min(1.0, level_shift * 2.0)
                        
                    #print(f"Increasing level shift to {level_shift:.3e}")
                    
                    # If level shift is getting too high, try line search
                    if level_shift > 0.5:
                        #print("Switching to line search due to unstable Newton steps")
                        use_newton = False
                
                # Reduce damping for more conservative steps
                damping_factor = max(0.2, damping_factor * 0.5)
                
                # If we have previous good geometry, backtrack halfway
                if prev_geom is not None:
                    #print(f"Backtracking: {error:.3e} -> {prev_error:.3e}")
                    current_geom = 0.5 * (current_geom + prev_geom)
                    
                    # Recalculate at backtracked position
                    B = self.build_B_matrix(current_geom)
                    current_q = np.dot(B, current_geom - reference_geom)
                    delta_q = step_int - current_q
                    error = np.linalg.norm(delta_q)
                    rms_error = error / np.sqrt(len(delta_q))
                
            # Save current position for potential backtracking
            prev_geom = current_geom.copy()
        
        # Handle non-convergence
        if iteration == max_iterations - 1 and rms_error > convergence_threshold:
            print(f"Warning: Conversion did not converge. Best RMS error = {best_error/np.sqrt(len(step_int)):.3e}")
            current_geom = best_geom
        
        # Return the Cartesian step
        return current_geom - reference_geom

    def _compute_diis_solution(self, error_vectors, geometries):
        """
        Compute DIIS extrapolated solution using Pulay's method
        
        Parameters:
        -----------
        error_vectors : list
            List of error vectors (flattened)
        geometries : list
            List of corresponding geometries
            
        Returns:
        --------
        ndarray
            DIIS extrapolated geometry
        """
        n_vecs = len(error_vectors)
        
        # Build DIIS B matrix for Pulay method
        B_diis = np.zeros((n_vecs + 1, n_vecs + 1))
        
        # Fill error vector dot products
        for i in range(n_vecs):
            for j in range(n_vecs):
                B_diis[i, j] = np.dot(error_vectors[i], error_vectors[j])
        
        # Add constraint rows/columns
        B_diis[n_vecs, :n_vecs] = 1.0
        B_diis[:n_vecs, n_vecs] = 1.0
        B_diis[n_vecs, n_vecs] = 0.0
        
        # RHS vector [0,0,...,0,1]
        rhs = np.zeros(n_vecs + 1)
        rhs[n_vecs] = 1.0
        
        # Add small regularization to diagonal for numerical stability
        for i in range(n_vecs):
            B_diis[i, i] += 1e-8 * (1.0 + abs(B_diis[i, i]))
        
        # Solve DIIS equations
        try:
            c = np.linalg.solve(B_diis, rhs)
        except np.linalg.LinAlgError:
            # Use SVD-based solution if direct solve fails
            c = np.linalg.lstsq(B_diis, rhs, rcond=1e-10)[0]
        
        # Construct DIIS solution
        diis_geom = np.zeros_like(geometries[0])
        for i in range(n_vecs):
            diis_geom += c[i] * geometries[i]
            
        return diis_geom
        
    
    def _evaluate_error(self, geom, reference_geom, target_step):
        """
        Helper function to evaluate the error for a given geometry
        """
        B = self.build_B_matrix(geom)
        current_q = np.dot(B, geom - reference_geom)
        delta_q = target_step - current_q
        return np.linalg.norm(delta_q)


    def internal_to_cartesian(self, step_int, geometry):
        """
        Transform step in internal coordinates to Cartesian coordinates
        using enhanced iterative methods
        """
        # Choose the algorithm based on configuration or problem characteristics
        
        if self.backconv_method == "scf":
            return self.internal_to_cartesian_scf(step_int, geometry)
        else:  
            return self.internal_to_cartesian_scf(step_int, geometry)
    
    def get_cleaned_hessian(self, hessian):
        """Ensure the Hessian is clean and well-conditioned"""
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        try:
            # Use more stable eigenvalue decomposition
            eigval, eigvec = linalg.eigh(hessian)
        except np.linalg.LinAlgError:
            # Fallback to more robust algorithm if standard fails
            print("Warning: Using more robust eigenvalue decomposition")
            eigval, eigvec = linalg.eigh(hessian, driver='evr')
        
        # Find valid eigenvalues (|λ| > 1e-7)
        valid_mask = np.abs(eigval) > 1e-7
        n_removed = np.sum(~valid_mask)
        
        # Create diagonal matrix with only valid eigenvalues
        # Replace small eigenvalues with small positive values
        cleaned_eigval = np.where(valid_mask, eigval, 1e-7)
        
        # Reconstruct Hessian using only valid components
        # H = U Λ U^T where Λ contains only valid eigenvalues
        cleaned_hessian = np.dot(np.dot(eigvec, np.diag(cleaned_eigval)), eigvec.T)
        
        # Ensure symmetry of final result
        cleaned_hessian = 0.5 * (cleaned_hessian + cleaned_hessian.T)
        
        return cleaned_hessian, n_removed
    
    def run_rfo_step(self, int_gradient, int_hessian):
        """
        Calculate the RFO step in internal coordinates with improved stability
        """
        n_coords = len(int_gradient)
        
        # Ensure symmetry and clean the Hessian
        new_hess = 0.5 * (int_hessian + int_hessian.T)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        
        # Construct RFO matrix
        matrix_for_RFO = np.block([
            [new_hess, int_gradient.reshape(n_coords, 1)],
            [int_gradient.reshape(1, n_coords), np.zeros((1, 1))]
        ])
        
        # Get eigenvalues of the RFO matrix
        try:
            RFO_eigenvalues, RFO_eigenvectors = linalg.eigh(matrix_for_RFO)
        except np.linalg.LinAlgError:
            print("Warning: Using more robust eigenvalue algorithm")
            RFO_eigenvalues, RFO_eigenvectors = linalg.eigh(matrix_for_RFO, driver='evr')
        
        # Sort eigenvalues
        idx = np.argsort(RFO_eigenvalues)
        RFO_eigenvalues = RFO_eigenvalues[idx]
        RFO_eigenvectors = RFO_eigenvectors[:, idx]
        
        # Select appropriate eigenvalue based on saddle order
        lambda_for_calc = float(RFO_eigenvalues[self.saddle_order])
        
        # Calculate step using direct RFO approach
        shifted_hessian = new_hess - lambda_for_calc * np.eye(n_coords)
        
        # Solve the RFO equations
        try:
            # LU decomposition for stable solving
            move_vector = -np.linalg.solve(shifted_hessian, int_gradient)
        except np.linalg.LinAlgError:
            print("Warning: Linear solve failed, using pseudoinverse")
            # Use pseudoinverse as fallback
            shifted_hessian_inv = np.linalg.pinv(shifted_hessian, rcond=1e-10)
            move_vector = -np.dot(shifted_hessian_inv, int_gradient)
        
        print(f"Lambda for RFO step: {lambda_for_calc}")
        print(f"Gradient RMS: {np.sqrt(np.mean(int_gradient**2))}")
        print(f"Step RMS: {np.sqrt(np.mean(move_vector**2))}")
        
        # Limit step size if it exceeds trust radius
        step_norm = np.linalg.norm(move_vector)
        if step_norm > self.trust_radius:
            scale_factor = self.trust_radius / step_norm
            move_vector *= scale_factor
            print(f"Step scaled by {scale_factor} to meet trust radius")
        
        return move_vector.reshape(-1, 1)
    
    def reset_system(self, geometry):
        """
        Reset internal coordinates when molecular structure changes significantly
        """
        print("Resetting redundant internal coordinate system")
        
        # Clear previous coordinates
        self.primitive_coords = None
        
        # Define new internal coordinates
        _, self.primitive_coords = self.define_internal_coordinates(geometry)
        print(f"Defined {len(self.primitive_coords)} new primitive coordinates")
        
        # Update B matrix
        B = self.update_coordinates(geometry)
        
        # Reset Hessian to identity
        self.internal_hessian = np.eye(len(self.primitive_coords))
        self.Initialization = False
        
        return B
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        """
        Main optimization step function using Redundant Internal Coordinates
        """
        print(f"======= RIC-RFO Iteration {self.iter} =======")
        
        # Define internal coordinates if not already defined
        if self.primitive_coords is None:
            _, self.primitive_coords = self.define_internal_coordinates(geom_num_list)
            print(f"Defined {len(self.primitive_coords)} primitive internal coordinates")
        
        # Check if molecular connectivity has changed
        coords_reset = False
        if pre_geom is not None:
            coords_reset = self.check_connectivity_change(geom_num_list)
        
        if coords_reset:
            # Reset coordinate system if connectivity changed
            B = self.reset_system(geom_num_list)
        else:
            # Update coordinate system for current geometry
            B = self.update_coordinates(geom_num_list)
        
        # Transform gradient to internal coordinates
        int_gradient = self.transform_gradient(B_g)
        
        # Initialize or update the Hessian
        if self.Initialization or self.internal_hessian is None:
            # Start with identity Hessian in internal coordinates
            self.internal_hessian = np.eye(len(self.primitive_coords))
            self.Initialization = False
        else:
            # Update the internal coordinate Hessian if we have previous geometry and gradient
            if pre_geom is not None and pre_B_g is not None and not coords_reset:
                # Calculate previous internal gradient
                prev_B = self.build_B_matrix(pre_geom)
                prev_G = self.build_G_matrix(prev_B)
                
                try:
                    U, s, Vh = np.linalg.svd(prev_G, full_matrices=False)
                    s_inv = np.array([1.0/x if x > self.redundant_thresh else 0.0 for x in s])
                    G_inv = np.dot(U * s_inv[:, np.newaxis], U.T)
                    
                    prev_B_pinv = np.dot(G_inv, prev_B)
                except np.linalg.LinAlgError:
                    prev_B_pinv = np.linalg.pinv(prev_B, rcond=1e-8)
                
                prev_int_gradient = np.dot(prev_B_pinv, pre_B_g)
                
                # Calculate displacement in internal coordinates
                cart_displacement = (geom_num_list - pre_geom).reshape(-1, 1)
                int_displacement = np.dot(B, cart_displacement)
                
                # Delta gradient in internal coordinates
                delta_grad = (int_gradient - prev_int_gradient).reshape(-1, 1)
                
                # Update Hessian using the hess_update methods
                if "msp" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using MSP Hessian update")
                    delta_hess = self.hess_update.MSP_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                elif "bfgs" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using BFGS Hessian update")
                    delta_hess = self.hess_update.BFGS_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                elif "fsb" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using FSB Hessian update")
                    delta_hess = self.hess_update.FSB_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                elif "bofill" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using Bofill Hessian update")
                    delta_hess = self.hess_update.Bofill_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                elif "sr1" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using SR1 Hessian update")
                    delta_hess = self.hess_update.SR1_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                elif "psb" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using PSB Hessian update")
                    delta_hess = self.hess_update.PSB_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                elif "flowchart" in self.config.get("method", "").lower():
                    print("RIC-RFO: Using flowchart Hessian update")
                    delta_hess = self.hess_update.flowchart_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad, self.config["method"]
                    )
                else:
                    # Default to BFGS if no method is specified
                    print("RIC-RFO: Using BFGS Hessian update (default)")
                    delta_hess = self.hess_update.BFGS_hessian_update(
                        self.internal_hessian, int_displacement, delta_grad
                    )
                
                # Apply the Hessian update
                self.internal_hessian += delta_hess
        
        # Ensure the Hessian is symmetric
        self.internal_hessian = 0.5 * (self.internal_hessian + self.internal_hessian.T)
        
        # Apply bias Hessian if provided
        working_hessian = self.internal_hessian.copy()
        if self.bias_hessian is not None:
            int_bias_hessian = self.transform_hessian(self.bias_hessian)
            working_hessian += int_bias_hessian
        
        # Calculate RFO step in internal coordinates
        int_step = self.run_rfo_step(int_gradient, working_hessian)
        
        # Transform step back to Cartesian coordinates
        cart_step = self.internal_to_cartesian(int_step, geom_num_list)
        
        # Store current state for next iteration
        self.prev_cartesian = geom_num_list.copy()
        self.prev_gradient = B_g.copy()
        
        # Increment iteration counter
        self.iter += 1
        
        # Apply DELTA scaling and reshape
        return -1 * self.DELTA * cart_step.reshape(-1, 1)
    
    def set_hessian(self, hessian):
        """Set Cartesian Hessian"""
        self.hessian = hessian.copy()
        if self.B_matrix is not None:
            # Transform to internal coordinates
            self.internal_hessian = self.transform_hessian(hessian)
        
    def set_bias_hessian(self, bias_hessian):
        """Set bias Hessian (in Cartesian coordinates)"""
        self.bias_hessian = bias_hessian.copy()
    
    def get_hessian(self):
        """Return current Hessian (in Cartesian coordinates)"""
        return self.hessian if self.hessian is not None else None
    
    
    def get_bias_hessian(self):
        return self.bias_hessian