import numpy as np
from scipy import linalg
from parameter import UnitValueLib, covalent_radii_lib
from .hessian_update import ModelHessianUpdate

"""
Hybrid Internal-Cartesian Coordinate RFO implementation
Based on:
1. Baker, J. Comput. Chem. Sci. 1993, 34, 118-127
2. Bakken and Helgaker, JCP 117, 9160 (2002)
3. Lindh et al., Chem. Phys. Lett. 241, 423-428 (1995)

Implementation combining redundant internal coordinates and Cartesian coordinates
"""

class HybridRFO:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        
        # Initialize specific parameters
        self.Initialization = True
        self.saddle_order = self.config.get("saddle_order", 0)
        self.trust_radius = float(self.config.get("trust_radius", 0.1))
        self.trust_radius /= UnitValueLib().bohr2angstroms
        self.DELTA = 0.1  # Step scale factor
        
        # For coordinate handling
        self.B_matrix = None      # Wilson B matrix (transformation matrix)
        self.G_matrix = None      # G matrix (metric matrix)
        self.primitive_coords = None
        
        # Hybrid specific parameters
        self.redundant_thresh = 1e-8   # Threshold for numerical stability
        self.coord_type = self.config.get("coord_type", "hybrid")
        self.iter = 0
        self.max_backsteps = 20
        self.max_micro_cycles = 100
        self.gradient_rms_threshold = 1e-4
        self.FC_COUNT = self.config.get("FC_COUNT", -1)
        self.backconv_method = self.config.get("backconv_method", "newton-diis").lower()
        # Initial weighting between coordinate systems
        self.internal_weight = self.config.get("internal_weight", 0.5)  # Default weight for internal coords
        self.cartesian_weight = 1.0 - self.internal_weight  # Weight for Cartesian coords
        
        # Tracking variables
        self.prev_cartesian = None
        self.prev_gradient = None
        self.internal_hessian = None
        self.cartesian_hessian = None
        self.hybrid_hessian = None
        self.hessian = None
        self.bias_hessian = None
        self.prev_connectivity = None
        self.last_backtransform_error = 0.0

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
            
            # Adjust weights to favor Cartesian coordinates during bond rearrangements
            self.internal_weight = max(0.5, self.internal_weight * 0.6)
            self.cartesian_weight = 1.0 - self.internal_weight
            print(f"Adjusted coordinate weights: Internal={self.internal_weight:.2f}, Cartesian={self.cartesian_weight:.2f}")
            
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
    
    def build_hybrid_system(self, geometry, cart_gradient=None):
        """
        Build the hybrid coordinate system combining internal and Cartesian
        """
        n_cart = len(geometry)
        
        # Build or update internal coordinate system
        B = self.update_coordinates(geometry)
        n_int = B.shape[0]
        
        # Create transformation matrices for the hybrid system
        cart_identity = np.eye(n_cart)
        
        # Calculate weights for the hybrid system
        w_int = np.sqrt(self.internal_weight)
        w_cart = np.sqrt(self.cartesian_weight)
        
        # Create hybrid transformation matrix: [w_int*B; w_cart*I]
        hybrid_transform = np.vstack([
            w_int * B,
            w_cart * cart_identity
        ])
        
        # Store dimensions for later use
        self.n_cart = n_cart
        self.n_int = n_int
        self.hybrid_transform = hybrid_transform
        
        return hybrid_transform, n_int + n_cart
    
    def transform_hybrid_gradient(self, cart_gradient):
        """
        Transform Cartesian gradient to hybrid coordinate gradient
        Returns: [w_int*B*g_cart; w_cart*g_cart]
        """
        # Transform gradient for internal coordinates: B*g_cart
        int_gradient = np.dot(self.B_matrix, cart_gradient)
        
        # Apply weights
        w_int = np.sqrt(self.internal_weight)
        w_cart = np.sqrt(self.cartesian_weight)
        
        # Create hybrid gradient vector
        hybrid_gradient = np.vstack([
            w_int * int_gradient,
            w_cart * cart_gradient
        ])
        
        return hybrid_gradient
    
    def transform_hybrid_hessian(self, cart_hessian):
        """
        Transform Cartesian Hessian to hybrid coordinate Hessian
        with proper dimension checking
        """
        B = self.B_matrix
        n_int = B.shape[0]
        n_cart = cart_hessian.shape[0]
        
        # Check dimensions
        if n_cart != self.n_cart:
            print(f"Warning: Bias Hessian dimension ({n_cart}) doesn't match current system ({self.n_cart})")
            # 必要に応じてサイズを調整するか、計算をスキップ
            return np.zeros((n_int + self.n_cart, n_int + self.n_cart))
        
        # Internal coordinate part: B*H_cart*B^T
        int_hessian = np.dot(B, np.dot(cart_hessian, B.T))
        
        # Apply weights
        w_int = np.sqrt(self.internal_weight)
        w_cart = np.sqrt(self.cartesian_weight)
        
        # Create hybrid Hessian with block structure
        hybrid_hessian = np.zeros((n_int + n_cart, n_int + n_cart))
        
        # Internal block: w_int^2 * B*H_cart*B^T
        hybrid_hessian[:n_int, :n_int] = w_int**2 * int_hessian
        
        # Cross terms: w_int*w_cart * B*H_cart and w_cart*w_int * H_cart*B^T
        hybrid_hessian[:n_int, n_int:] = w_int * w_cart * np.dot(B, cart_hessian)
        hybrid_hessian[n_int:, :n_int] = w_int * w_cart * np.dot(cart_hessian, B.T)
        
        # Cartesian block: w_cart^2 * H_cart
        hybrid_hessian[n_int:, n_int:] = w_cart**2 * cart_hessian
        
        return hybrid_hessian
    
    def hybrid_to_cartesian(self, step_hybrid, geometry):
        """
        Convert a step in hybrid coordinates back to Cartesian coordinates
        The hybrid step contains both internal and Cartesian contributions
        """
        n_int = self.n_int
        n_cart = self.n_cart
        
        # Extract and unweight the internal and Cartesian components
        w_int = np.sqrt(self.internal_weight)
        w_cart = np.sqrt(self.cartesian_weight)
        
        step_int = step_hybrid[:n_int] / w_int
        step_cart_direct = step_hybrid[n_int:] / w_cart
        
        # Step in Cartesian coordinates via internal coordinates
        cart_step_from_int = self.internal_to_cartesian(step_int, geometry)
        
        # Combine the two steps with proper weighting
        total_cart_step = self.internal_weight * cart_step_from_int + self.cartesian_weight * step_cart_direct
        
        # Print information about the contributions
        int_norm = np.linalg.norm(cart_step_from_int)
        cart_norm = np.linalg.norm(step_cart_direct)
        total_norm = np.linalg.norm(total_cart_step)
        
        print(f"Step norms - Internal: {int_norm:.6f}, Cartesian: {cart_norm:.6f}, Combined: {total_norm:.6f}")
        
        return total_cart_step
        
    def internal_to_cartesian_newton(self, step_int, geometry, use_diis=True, 
                                    max_iterations=50, convergence_threshold=1e-8):
        """
        Enhanced Newton-Raphson method for internal to Cartesian coordinate conversion
        with optional DIIS acceleration.
        
        Parameters:
        -----------
        step_int : ndarray
            Target step in internal coordinates
        geometry : ndarray
            Starting geometry in Cartesian coordinates
        use_diis : bool
            Whether to use DIIS acceleration
        max_iterations : int
            Maximum number of iterations
        convergence_threshold : float
            Convergence criterion for RMS error
            
        Returns:
        --------
        ndarray
            Step in Cartesian coordinates
        """
        # Initialize DIIS if requested
        diis_subspace = 8  # Maximum size of DIIS subspace
        diis_start = 3     # Start DIIS after this many steps
        
        if use_diis:
            error_vectors = []
            solution_vectors = []
        
        # Initialize tracking variables
        reference_geom = geometry.copy()
        current_geom = geometry.copy()
        best_geom = current_geom.copy()
        best_error = float('inf')
        prev_error = float('inf')
        
        # Pre-condition step_int to avoid large steps
        step_norm = np.linalg.norm(step_int)
        if step_norm > 1.0:
            print(f"Pre-conditioning large internal step (norm={step_norm:.3f})")
            step_int = step_int * (1.0 / step_norm)
        
        # Iterative Newton-Raphson with line search and DIIS acceleration
        for iteration in range(max_iterations):
            # Build B matrix for current geometry
            B = self.build_B_matrix(current_geom)
            
            # Calculate current internal coordinates relative to reference
            current_q = np.dot(B, current_geom - reference_geom)
            
            # Calculate error vector (difference from target)
            delta_q = step_int - current_q
            error = np.linalg.norm(delta_q)
            rms_error = error / np.sqrt(len(delta_q))
            
            # Debug output every few iterations
            if iteration % 5 == 0 or iteration < 3:
                print(f"Iteration {iteration}: RMS error = {rms_error:.3e}")
            
            # Save best solution
            if error < best_error:
                best_error = error
                best_geom = current_geom.copy()
            
            # Check convergence
            if rms_error < convergence_threshold:
                print(f"Newton-Raphson converged after {iteration+1} iterations")
                break
                
            # Early termination if we're not making progress
            if iteration > 5 and abs(error - prev_error) < convergence_threshold * 1e-10:
                print(f"Newton-Raphson: Early termination due to slow progress")
                break
            
            # Calculate B+ (pseudoinverse of B) using SVD for stability
            try:
                U, s, Vh = np.linalg.svd(B, full_matrices=False)
                s_inv = np.where(s > 1e-7, 1.0 / s, 0.0)
                B_pinv = np.dot(np.dot(Vh.T, np.diag(s_inv)), U.T)
            except np.linalg.LinAlgError:
                print("Warning: SVD failed, using direct pseudoinverse")
                B_pinv = np.linalg.pinv(B, rcond=1e-8)
            
            # Calculate Newton step
            delta_x = np.dot(B_pinv, delta_q)
            
            # Apply trust region constraint
            step_norm = np.linalg.norm(delta_x)
            trust_radius = 0.2  # Maximum step size in Angstroms
            
            if step_norm > trust_radius:
                delta_x = delta_x * (trust_radius / step_norm)
                step_norm = trust_radius
                
            # Store current error and position for DIIS
            if use_diis and iteration >= diis_start:
                # Important: Ensure vectors are flattened to one dimension
                error_vectors.append(delta_q.flatten())
                solution_vectors.append(current_geom.copy())
                
                # Limit DIIS subspace size
                if len(error_vectors) > diis_subspace:
                    error_vectors.pop(0)
                    solution_vectors.pop(0)
            
            # Regular Newton step by default
            new_geom = current_geom + delta_x
            
            # Apply DIIS if we have enough vectors
            if use_diis and iteration >= diis_start and len(error_vectors) >= 2:
                try:
                    # Build DIIS B matrix (Pulay method)
                    n_vecs = len(error_vectors)
                    B_diis = np.zeros((n_vecs + 1, n_vecs + 1))
                    
                    # Fill B matrix with error vector dot products
                    # Fix: Use already flattened vectors
                    for i in range(n_vecs):
                        for j in range(n_vecs):
                            # Explicitly check shapes to prevent errors
                            e_i = error_vectors[i]
                            e_j = error_vectors[j]
                            B_diis[i, j] = np.dot(e_i, e_j)
                    
                    # Last row and column are for constraint sum(c) = 1
                    B_diis[n_vecs, :n_vecs] = 1.0
                    B_diis[:n_vecs, n_vecs] = 1.0
                    B_diis[n_vecs, n_vecs] = 0.0
                    
                    # RHS vector [0,0,...,0,1]
                    rhs = np.zeros(n_vecs + 1)
                    rhs[n_vecs] = 1.0
                    
                    # Solve for DIIS coefficients
                    try:
                        c = np.linalg.solve(B_diis, rhs)
                    except np.linalg.LinAlgError:
                        c = np.linalg.lstsq(B_diis, rhs, rcond=1e-10)[0]
                    
                    # Form DIIS optimized geometry
                    diis_geom = np.zeros_like(current_geom)
                    for i in range(n_vecs):
                        diis_geom += c[i] * solution_vectors[i]
                    
                    # Line search between Newton and DIIS steps
                    newton_error = self._evaluate_error(new_geom, reference_geom, step_int)
                    diis_error = self._evaluate_error(diis_geom, reference_geom, step_int)
                    
                    if diis_error < newton_error:
                        print(f"DIIS correction applied at iteration {iteration+1}")
                        new_geom = diis_geom
                except Exception as e:
                    print(f"DIIS calculation failed: {e}")
                    # Add debug information
                    if len(error_vectors) > 0:
                        print(f"Error vector shape: {np.asarray(error_vectors[0]).shape}")
                    pass  # Fall back to regular Newton step
            
            # Apply adaptive damping based on progress
            if iteration > 0:
                if error > prev_error:
                    # We're getting worse, use more damping
                    alpha = 0.5
                    new_geom = current_geom + alpha * delta_x
                else:
                    # We're improving, can be more aggressive
                    alpha = min(1.0, 0.6 + 0.4 * np.exp(-iteration/10.0))
                    # If DIIS was used, alpha is already incorporated
                    if not (use_diis and iteration >= diis_start and len(error_vectors) >= 2):
                        new_geom = current_geom + alpha * delta_x
            
            # Update for next iteration
            current_geom = new_geom
            prev_error = error
        
        # If we didn't converge, warn and use best solution
        if iteration == max_iterations - 1:
            print(f"Warning: Newton-Raphson did not converge. Best RMS error = {best_error/np.sqrt(len(step_int)):.3e}")
            current_geom = best_geom
            
        # Store error for diagnostics
        self.last_backtransform_error = best_error
        
        # Return the total step from original geometry
        return current_geom - reference_geom

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
        
        if self.backconv_method == "newton-diis" or self.backconv_method == "diis":
            return self.internal_to_cartesian_newton(step_int, geometry, use_diis=True)

        else:  # Default to Newton method without DIIS
            return self.internal_to_cartesian_newton(step_int, geometry, use_diis=False)
    
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
    
    def run_hybrid_rfo_step(self, hybrid_gradient, hybrid_hessian):
        """
        Calculate the RFO step in hybrid coordinates
        """
        n_coords = len(hybrid_gradient)
        
        # Ensure symmetry and clean the Hessian
        new_hess = 0.5 * (hybrid_hessian + hybrid_hessian.T)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        
        # Construct RFO matrix
        matrix_for_RFO = np.block([
            [new_hess, hybrid_gradient.reshape(n_coords, 1)],
            [hybrid_gradient.reshape(1, n_coords), np.zeros((1, 1))]
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
            move_vector = -np.linalg.solve(shifted_hessian, hybrid_gradient)
        except np.linalg.LinAlgError:
            print("Warning: Linear solve failed, using pseudoinverse")
            # Use pseudoinverse as fallback
            shifted_hessian_inv = np.linalg.pinv(shifted_hessian, rcond=1e-10)
            move_vector = -np.dot(shifted_hessian_inv, hybrid_gradient)
        
        print(f"Lambda for RFO step: {lambda_for_calc}")
        print(f"Gradient RMS: {np.sqrt(np.mean(hybrid_gradient**2))}")
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
        print("Resetting hybrid coordinate system")
        
        # Clear previous coordinates
        self.primitive_coords = None
        
        # Define new internal coordinates
        _, self.primitive_coords = self.define_internal_coordinates(geometry)
        print(f"Defined {len(self.primitive_coords)} new primitive coordinates")
        
        # Update B matrix
        B = self.update_coordinates(geometry)
        
        # Reset hybrid system dimensions
        self.n_int = len(self.primitive_coords)
        self.n_cart = len(geometry)
        
        # Reset Hessian
        self.internal_hessian = np.eye(self.n_int)
        self.cartesian_hessian = np.eye(self.n_cart)
        self.hybrid_hessian = np.eye(self.n_int + self.n_cart)
        
        # Reset flags
        self.Initialization = False
        
        return B
    
    def update_weights(self, geometry, cart_gradient, cart_step=None):
        """
        Adaptively update weights for internal and Cartesian coordinates
        based on optimization progress
        """
        # Skip if no previous step
        if self.iter < 2 or cart_step is None:
            return
            
        # Update weights based on backtransformation error and step norm
        if self.last_backtransform_error > 0.1:
            # Increase weight of Cartesian coords if transformation is unstable
            self.internal_weight = max(0.2, self.internal_weight * 0.9)
        elif self.last_backtransform_error < 0.01 and self.internal_weight < 0.8:
            # Increase weight of internal if transformation is reliable
            self.internal_weight = min(0.5, self.internal_weight * 1.1)
            
        # Update Cartesian weight accordingly
        self.cartesian_weight = 1.0 - self.internal_weight
        
        print(f"Updated coordinate weights: Internal={self.internal_weight:.2f}, Cartesian={self.cartesian_weight:.2f}")
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        """
        Main optimization step function using Hybrid Cartesian-Internal Coordinates
        """
        print(f"======= Hybrid RFO Iteration {self.iter} =======")
        
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
            self.reset_system(geom_num_list)
        
        # Build hybrid coordinate system
        hybrid_transform, n_hybrid = self.build_hybrid_system(geom_num_list, B_g)
        
        # Transform gradient to hybrid coordinates
        hybrid_gradient = self.transform_hybrid_gradient(B_g)
        
        # Initialize or update the Hessian
        if self.Initialization or self.hybrid_hessian is None:
            # Start with identity Hessian in hybrid coordinates
            self.hybrid_hessian = np.eye(n_hybrid)
            self.internal_hessian = np.eye(self.n_int)
            self.cartesian_hessian = np.eye(self.n_cart)
            self.Initialization = False
        else:
            # Update the hybrid Hessian if we have previous geometry and gradient
            if pre_geom is not None and pre_B_g is not None and not coords_reset:
                # Get previous hybrid transform and gradient
                prev_B = self.build_B_matrix(pre_geom)
                prev_hybrid_gradient = self.transform_hybrid_gradient(pre_B_g)
                
                # Calculate displacement in hybrid coordinates
                cart_displacement = (geom_num_list - pre_geom).reshape(-1, 1)
                hybrid_displacement = np.dot(hybrid_transform, cart_displacement)
                
                # Delta gradient in hybrid coordinates
                delta_grad = (hybrid_gradient - prev_hybrid_gradient).reshape(-1, 1)
                
                # Update Hessian using the appropriate update method
                if "msp" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using MSP Hessian update")
                    delta_hess = self.hess_update.MSP_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                elif "bfgs" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using BFGS Hessian update")
                    delta_hess = self.hess_update.BFGS_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                elif "fsb" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using FSB Hessian update")
                    delta_hess = self.hess_update.FSB_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                elif "bofill" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using Bofill Hessian update")
                    delta_hess = self.hess_update.Bofill_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                elif "sr1" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using SR1 Hessian update")
                    delta_hess = self.hess_update.SR1_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                elif "psb" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using PSB Hessian update")
                    delta_hess = self.hess_update.PSB_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                elif "flowchart" in self.config.get("method", "").lower():
                    print("Hybrid RFO: Using flowchart Hessian update")
                    delta_hess = self.hess_update.flowchart_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad, self.config["method"]
                    )
                else:
                    # Default to BFGS if no method is specified
                    print("Hybrid RFO: Using BFGS Hessian update (default)")
                    delta_hess = self.hess_update.BFGS_hessian_update(
                        self.hybrid_hessian, hybrid_displacement, delta_grad
                    )
                
                # Apply the Hessian update
                self.hybrid_hessian += delta_hess
        
        # Ensure the Hessian is symmetric
        self.hybrid_hessian = 0.5 * (self.hybrid_hessian + self.hybrid_hessian.T)
        
        # Apply bias Hessian if provided
        working_hessian = self.hybrid_hessian.copy()
        if self.bias_hessian is not None:
            hybrid_bias_hessian = self.transform_hybrid_hessian(self.bias_hessian)
            working_hessian += hybrid_bias_hessian
        
        # Calculate RFO step in hybrid coordinates
        hybrid_step = self.run_hybrid_rfo_step(hybrid_gradient, working_hessian)
        
        # Transform step back to Cartesian coordinates
        cart_step = self.hybrid_to_cartesian(hybrid_step, geom_num_list)
        
        # Update weights for next iteration based on this step
        self.update_weights(geom_num_list, B_g, cart_step)
        
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
            self.internal_hessian = self.transform_hybrid_hessian(hessian)
        
    def set_bias_hessian(self, bias_hessian):
        """Set bias Hessian (in Cartesian coordinates)"""
        self.bias_hessian = bias_hessian.copy()
    
    def get_hessian(self):
        """Return current Hessian (in Cartesian coordinates)"""
        return self.hessian if self.hessian is not None else None
    