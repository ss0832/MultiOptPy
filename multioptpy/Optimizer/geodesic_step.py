import numpy as np
from scipy.integrate import solve_ivp
import logging

from multioptpy.Parameters.parameter import covalent_radii_lib

class GeodesicStepper:
    """
    Geodesic step calculation for geometry optimization based on
    J. Chem. Phys. 155, 094105 (2021) "Geometry optimization speedup through a geodesic coordinate system"
    
    This class implements geodesic-based step calculation to improve optimization convergence
    by following the intrinsic curved geometry of the internal coordinate space.
    """
    def __init__(self, element_list):
        """
        Initialize the GeodesicStep optimizer
        
        Parameters:
        -----------
        element_list : list
            List of element symbols as strings
        """
        self.element_list = element_list
        self.natoms = len(element_list)
        self.ndim = 3 * self.natoms
        self.logger = logging.getLogger(__name__)
        self.bond_scale = 1.5  # Scaling factor for covalent radii
        
    def determine_bonds(self):
        """
        Determine bonded atoms based on covalent radii
        Only considers bond lengths as internal coordinates
        
        Returns:
        --------
        list
            List of (i,j) tuples representing bonded atoms
        """
        
        
        # Get covalent radii for all atoms
        radii = [covalent_radii_lib(elem) for elem in self.element_list]
        
        bonds = []
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                # Calculate distance threshold based on covalent radii with scaling
                threshold = (radii[i] + radii[j]) * self.bond_scale
                # Store bond if it exists
                bonds.append((i, j, threshold))
                
        return bonds
    
    def calculate_internal_coordinates(self, geometry):
        """
        Calculate internal coordinates (bond lengths) from Cartesian coordinates
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Cartesian coordinates (natoms×3)
            
        Returns:
        --------
        tuple
            (bond_values, bond_pairs, bond_thresholds)
        """
        # Get bonds
        bonds = self.determine_bonds()
        
        # Calculate bond lengths and filter by threshold
        bond_values = []
        bond_pairs = []
        bond_thresholds = []
        
        for i, j, threshold in bonds:
            r_ij = np.linalg.norm(geometry[i] - geometry[j])
            if r_ij < threshold:
                bond_values.append(r_ij)
                bond_pairs.append((i, j))
                bond_thresholds.append(threshold)
                
        return np.array(bond_values), bond_pairs, bond_thresholds
    
    def calculate_b_matrix(self, geometry, bond_pairs):
        """
        Calculate Wilson's B-matrix (derivatives of internal coordinates w.r.t. Cartesian)
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Cartesian coordinates (natoms×3)
        bond_pairs : list
            List of (i,j) tuples representing bonded atoms
            
        Returns:
        --------
        numpy.ndarray
            B matrix
        """
        ncoords = len(bond_pairs)
        ndim = self.ndim
        B = np.zeros((ncoords, ndim))
        
        # Calculate B matrix elements
        for idx, (i, j) in enumerate(bond_pairs):
            # Vector pointing from atom j to atom i
            rij = geometry[i] - geometry[j]
            # Bond length
            r = np.linalg.norm(rij)
            # Unit vector
            if r > 1e-10:  # Avoid division by zero
                unit_vec = rij / r
            else:
                unit_vec = np.zeros(3)
                
            # Fill B matrix elements
            B[idx, 3*i:3*i+3] = unit_vec
            B[idx, 3*j:3*j+3] = -unit_vec
            
        return B
    
    def calculate_b_derivatives(self, geometry, bond_pairs):
        """
        Calculate derivatives of the B-matrix w.r.t. Cartesian coordinates
        Using vectorized operations
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Cartesian coordinates (natoms×3)
        bond_pairs : list
            List of (i,j) tuples representing bonded atoms
            
        Returns:
        --------
        numpy.ndarray
            Derivatives of B matrix
        """
        ncoords = len(bond_pairs)
        ndim = self.ndim
        dB = np.zeros((ncoords, ndim, ndim))
        
        coords = geometry.reshape(-1, 3)
        
        # Vectorized calculation for all bonds
        for idx, (i, j) in enumerate(bond_pairs):
            # Vector pointing from atom j to atom i
            rij = coords[i] - coords[j]
            r = np.linalg.norm(rij)
            
            if r < 1e-10:  # Avoid numerical issues
                continue
                
            # Precompute terms
            r3 = r**3
            
            # Create unit matrices for i and j blocks
            eye3 = np.eye(3)
            outer_prod = np.outer(rij, rij) / r3
            
            # Term for each 3x3 block is: δ_ab/r - r_a*r_b/r^3
            block = eye3 / r - outer_prod
            
            # Assign to the 4 blocks efficiently
            ii_idx = slice(3*i, 3*i+3)
            jj_idx = slice(3*j, 3*j+3)
            
            dB[idx, ii_idx, ii_idx] = block
            dB[idx, jj_idx, jj_idx] = block
            dB[idx, ii_idx, jj_idx] = -block
            dB[idx, jj_idx, ii_idx] = -block
            
        return dB
    
    def calculate_metric_tensor(self, B):
        """
        Calculate metric tensor G = B·Bᵀ
        
        Parameters:
        -----------
        B : numpy.ndarray
            Wilson's B matrix
            
        Returns:
        --------
        numpy.ndarray
            Metric tensor G
        """
        G = np.dot(B, B.T)
        return G
    
    def calculate_christoffel_symbols(self, B, dB, G_inv):
        """
        Calculate Christoffel symbols of the second kind using vectorized operations
        
        Parameters:
        -----------
        B : numpy.ndarray
            Wilson's B matrix
        dB : numpy.ndarray
            Derivatives of B matrix
        G_inv : numpy.ndarray
            Inverse of G matrix
            
        Returns:
        --------
        gamma : numpy.ndarray
            Christoffel symbols
        """
        ncoords = B.shape[0]
        
        # Initialize Christoffel symbols tensor
        gamma = np.zeros((ncoords, ncoords, ncoords))
        
        # Step 1: Calculate partial_j B_ia for all combinations
        partial_j_B_i_a = np.zeros((ncoords, ncoords, self.ndim))
        for i in range(ncoords):
            for j in range(ncoords):
                for a in range(self.ndim):
                    partial_j_B_i_a[i, j, a] = np.sum(dB[i, a, :] * B[j, :])
        
        # Step 2: Sum over dimension a and contract with G_inv
        for i in range(ncoords):
            for j in range(ncoords):
                for k in range(ncoords):
                    gamma[i, j, k] = G_inv[i, k] * np.sum(partial_j_B_i_a[i, j, :])
        
        # Step 3: Symmetrize over j,k indices
        for i in range(ncoords):
            for j in range(ncoords):
                for k in range(j+1, ncoords):
                    avg = (gamma[i, j, k] + gamma[i, k, j]) / 2
                    gamma[i, j, k] = avg
                    gamma[i, k, j] = avg
        
        return gamma
    
    def geodesic_equation(self, t, y, gamma, ncoords):
        """
        Vectorized implementation of the geodesic equation ODE function
        
        Parameters:
        -----------
        t : float
            Integration parameter (0 to 1)
        y : numpy.ndarray
            State vector [q, q_dot], where q are internal coordinates
        gamma : numpy.ndarray
            Christoffel symbols
        ncoords : int
            Number of internal coordinates
            
        Returns:
        --------
        dydt : numpy.ndarray
            Derivative of state vector
        """
        # Extract q and q_dot from state vector
        q = y[:ncoords]
        q_dot = y[ncoords:]
        
        # Calculate acceleration using vectorized operations
        # Compute the contraction of Christoffel symbols with velocities
        q_ddot = np.zeros(ncoords)
        for i in range(ncoords):
            for j in range(ncoords):
                for k in range(ncoords):
                    q_ddot[i] -= gamma[i, j, k] * q_dot[j] * q_dot[k]
        
        # Return combined derivative [q_dot, q_ddot]
        return np.concatenate([q_dot, q_ddot])
    
    def parallel_transport(self, v0, B0, G0_inv, B1, G1_inv):
        """
        Parallel transport a vector from one point to another
        
        Parameters:
        -----------
        v0 : numpy.ndarray
            Vector in Cartesian coordinates at the initial point
        B0, B1 : numpy.ndarray
            B matrices at initial and final points
        G0_inv, G1_inv : numpy.ndarray
            Inverse metric tensors at initial and final points
            
        Returns:
        --------
        numpy.ndarray
            Transported vector in Cartesian coordinates
        """
        # Transform to internal coordinates
        v0_int = np.dot(B0, v0)
        
        # Parallel transport in internal coordinates (simplified)
        # In this implementation, we use a direct transformation approach
        # For more accuracy, one should solve the parallel transport equations
        
        # Transform back to Cartesian coordinates
        v1 = np.dot(B1.T, np.dot(G1_inv, np.dot(G0_inv, v0_int)))
        
        return v1
    
    def solve_geodesic(self, q0, v0, gamma, ncoords):
        """
        Solve the geodesic equation using LSODA method
        
        Parameters:
        -----------
        q0 : numpy.ndarray
            Initial internal coordinates
        v0 : numpy.ndarray
            Initial velocity in internal coordinates
        gamma : numpy.ndarray
            Christoffel symbols
        ncoords : int
            Number of internal coordinates
            
        Returns:
        --------
        numpy.ndarray
            Final internal coordinates after following the geodesic
        """
        # Initial state [q0, v0]
        y0 = np.concatenate([q0, v0])
        
        # ODE solver with LSODA method
        solution = solve_ivp(
            lambda t, y: self.geodesic_equation(t, y, gamma, ncoords),
            [0, 1],  # Integration from t=0 to t=1
            y0,
            method='LSODA',
            rtol=1e-6,
            atol=1e-8
        )
        
        # Return final coordinates
        q_final = solution.y[:ncoords, -1]
        return q_final
    
    def transform_to_cartesian(self, geometry, q_final, B, bond_pairs):
        """
        Transform internal coordinate step to Cartesian coordinates
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current Cartesian geometry
        q_final : numpy.ndarray
            Final internal coordinates
        B : numpy.ndarray
            Wilson's B matrix
        bond_pairs : list
            List of (i,j) tuples for bonds
            
        Returns:
        --------
        numpy.ndarray
            Step vector in Cartesian coordinates
        """
        # Calculate current internal coordinates
        q_current = np.array([np.linalg.norm(geometry[i] - geometry[j]) 
                             for i, j in bond_pairs])
        
        # Calculate change in internal coordinates
        dq = q_final - q_current
        
        # Transform to Cartesian step using the pseudo-inverse of B
        B_pinv = np.linalg.pinv(B)
        cartesian_step = np.dot(B_pinv, dq)
        
        return cartesian_step
    
    def run(self, geometry, original_move_vector):
        """
        Run geodesic step calculation
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry in Cartesian coordinates
        original_move_vector : numpy.ndarray
            Original step vector from optimizer
            
        Returns:
        --------
        numpy.ndarray
            Modified step vector following geodesic path
        """
        # Reshape inputs if needed
        geom = geometry.reshape(self.natoms, 3)
        move_vec = original_move_vector.reshape(-1)
        
        # Calculate internal coordinates
        q, bond_pairs, _ = self.calculate_internal_coordinates(geom)
        ncoords = len(q)
        
        if ncoords == 0:
            self.logger.warning("No internal coordinates found. Using original step.")
            return original_move_vector
        
        # Calculate Wilson's B-matrix
        B = self.calculate_b_matrix(geom, bond_pairs)
        
        # Calculate metric tensor and its inverse
        G = self.calculate_metric_tensor(B)
        
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular G matrix. Using pseudo-inverse instead.")
            G_inv = np.linalg.pinv(G)
        
        # Calculate derivatives of B-matrix
        dB = self.calculate_b_derivatives(geom, bond_pairs)
        
        # Calculate Christoffel symbols
        gamma = self.calculate_christoffel_symbols(B, dB, G_inv)
        
        # Transform original move vector to internal coordinates
        v0_int = np.dot(B, move_vec)
        
        # Solve geodesic equation
        q_final = self.solve_geodesic(q, v0_int, gamma, ncoords)
        
        # Transform back to Cartesian coordinates
        geodesic_step = self.transform_to_cartesian(geom, q_final, B, bond_pairs)
        print(f"Geodesic step norm: {np.linalg.norm(geodesic_step):<.6f}")
        print(f"Original step norm: {np.linalg.norm(move_vec):<.6f}")
        print(f"Ratio (geodesic/original): {np.linalg.norm(geodesic_step)/np.linalg.norm(move_vec):<.6f}")
        # Reshape to match input format
        return geodesic_step.reshape(original_move_vector.shape)