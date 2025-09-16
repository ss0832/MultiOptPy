import numpy as np
from numpy.linalg import norm

from multioptpy.Parameters.parameter import UnitValueLib

class ConjugateGradientNEB:
    """Nonlinear Conjugate Gradient optimizer for Nudged Elastic Band calculations.
    
    This implementation provides multiple conjugate gradient update formulas:
    - Fletcher-Reeves (FR): Numerically stable but slower convergence
    - Polak-Ribière (PR): Faster convergence but can be unstable without safeguards
    - Hestenes-Stiefel (HS): Often performs better than FR and PR for nonlinear problems
    - Dai-Yuan (DY): Good global convergence properties and descent direction guarantee
    - Hager-Zhang (HZ): Modern, efficient update with strong theoretical properties
    
    All methods include automatic restart strategies for enhanced stability and performance.
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.Initialization = True
        self.iter = 0
        
        # Set default parameters
        self.maxstep = config.get("maxstep", 0.1)  # Maximum step size
        
        # Line search parameters
        self.initial_step_size = config.get("initial_step_size", 0.1)  # Initial step size for line search
        self.min_step_size = config.get("min_step_size", 0.01)  # Minimum step size
        self.max_step_size = config.get("max_step_size", 0.1)  # Maximum step size
        self.step_descent_factor = config.get("step_descent_factor", 0.5)  # Step size reduction factor
        
        # CG parameters
        # Valid options: "FR", "PR", "HS", "DY", "HZ"
        self.cg_method = config.get("cg_method", "HS")  # Conjugate gradient method
        self.restart_cycles = config.get("restart_cycles", 10)  # Restart CG every n iterations
        self.restart_threshold = config.get("restart_threshold", 0.2)  # Restart if orthogonality drops below this
        
        # Storage for previous step data
        self.prev_forces = None  # Previous forces for all nodes
        self.prev_directions = None  # Previous CG directions for all nodes
        self.prev_positions = None  # Previous positions for all nodes
        self.prev_energies = None  # Previous energies for all nodes
        
        # Node-specific step sizes (adaptive)
        self.node_step_sizes = None
        
        # Unit conversion
        self.bohr2angstroms = UnitValueLib().bohr2angstroms 
        if 'bohr2angstroms' in config:
            self.bohr2angstroms = config['bohr2angstroms']
        
        # Compatibility with the original code
        self.TR_NEB = config.get("TR_NEB", None)
        
        # Print initialization message based on selected method
        print(f"Initialized Conjugate Gradient NEB optimizer with method={self.cg_method}, maxstep={self.maxstep}")
        if self.cg_method == "FR":
            print("Using Fletcher-Reeves method with periodic restart")
        elif self.cg_method == "PR":
            print("Using Polak-Ribière method with automatic restart")
        elif self.cg_method == "HS":
            print("Using Hestenes-Stiefel method with automatic restart")
        elif self.cg_method == "DY":
            print("Using Dai-Yuan method with strong convergence properties")
        elif self.cg_method == "HZ":
            print("Using Hager-Zhang method with enhanced numerical stability")
        else:
            print(f"Warning: Unknown CG method '{self.cg_method}', falling back to Fletcher-Reeves")
            self.cg_method = "FR"
    
    def initialize_data(self, num_nodes):
        """Initialize data structures for the optimization.
        
        Parameters:
        ----------
        num_nodes : int
            Number of nodes in the NEB calculation
        """
        # Initialize storage for previous data
        self.prev_forces = [None] * num_nodes
        self.prev_directions = [None] * num_nodes
        self.prev_positions = [None] * num_nodes
        self.prev_energies = [None] * num_nodes
        
        # Initialize node-specific step sizes
        self.node_step_sizes = np.ones(num_nodes) * self.initial_step_size
        
        print(f"Initialized CG storage for {num_nodes} nodes")
    
    def compute_cg_direction(self, force, node_idx):
        """Compute the conjugate gradient direction for an node.
        
        Parameters:
        ----------
        force : ndarray
            Current force vector for the node
        node_idx : int
            Index of the node in the NEB chain
            
        Returns:
        -------
        ndarray
            Conjugate gradient direction
        """
        # If first iteration or no history, return the force direction (steepest descent)
        if self.prev_forces[node_idx] is None:
            return force.copy()
        
        # Get previous force and direction
        prev_force = self.prev_forces[node_idx]
        prev_direction = self.prev_directions[node_idx]
        
        # Compute change in force (negative gradient)
        force_change = force - prev_force
        
        # Flatten vectors for easier dot product calculations
        g = force.flatten()
        g_prev = prev_force.flatten()
        d_prev = prev_direction.flatten()
        y = force_change.flatten()
        
        # Compute beta coefficient according to selected method
        beta = 0.0  # Default value
        
        # Compute dot products that will be reused
        g_dot_g = np.dot(g, g)
        g_prev_dot_g_prev = np.dot(g_prev, g_prev)
        y_dot_d_prev = np.dot(y, d_prev)
        g_dot_y = np.dot(g, y)
        
        # Small constant to avoid division by zero
        epsilon = 1e-10
        
        if self.cg_method == "FR":  # Fletcher-Reeves
            beta = g_dot_g / max(epsilon, g_prev_dot_g_prev)
            
        elif self.cg_method == "PR":  # Polak-Ribière
            beta = g_dot_y / max(epsilon, g_prev_dot_g_prev)
            # Apply Polak-Ribière+ modification: max(beta, 0)
            beta = max(0.0, beta)
            
        elif self.cg_method == "HS":  # Hestenes-Stiefel
            beta = g_dot_y / max(epsilon, y_dot_d_prev)
            # Apply HS+ modification for guaranteed descent direction
            beta = max(0.0, beta)
            
        elif self.cg_method == "DY":  # Dai-Yuan
            beta = g_dot_g / max(epsilon, y_dot_d_prev)
            # DY method always yields descent directions
            
        elif self.cg_method == "HZ":  # Hager-Zhang
            y_dot_y = np.dot(y, y)
            g_prev_dot_y = np.dot(g_prev, y)
            
            # Hager-Zhang formula
            beta = (g_dot_y - 2 * np.dot(g, y) * np.dot(y, d_prev) / max(epsilon, y_dot_y)) / max(epsilon, y_dot_d_prev)
            
            # Apply HZ+ modification
            eta = 0.4  # Parameter from the HZ paper
            beta = max(-eta * g_prev_dot_g_prev / max(epsilon, np.dot(d_prev, d_prev)), beta)
        
        # Check orthogonality for restart
        orthogonality = np.abs(np.dot(g, g_prev)) / (norm(g) * norm(g_prev) + epsilon)
        
        # Reset beta to zero (restart CG) if:
        # 1. We're at a restart cycle iteration
        # 2. Orthogonality is high (gradient directions are too similar)
        # 3. Beta is negative (not an issue with PR+, HS+, HZ+, but added as safety)
        if (self.iter % self.restart_cycles == 0) or (orthogonality > (1.0 - self.restart_threshold)):
            print(f"Restarting CG for node {node_idx} (orthogonality: {orthogonality:.3f})")
            beta = 0.0
        
        # Compute new conjugate direction
        new_direction = force + beta * prev_direction
        
        # Ensure we're moving in a descent direction (dot product with force should be positive)
        if np.dot(new_direction.flatten(), g) < 0:
            print(f"Warning: CG direction not a descent direction for node {node_idx}, resetting to steepest descent")
            new_direction = force.copy()
        
        return new_direction
    
    def adjust_step_sizes(self, energies, prev_energies, forces):
        """Adjust step sizes for each node based on energy changes.
        
        Parameters:
        ----------
        energies : list of float
            Current energies for all nodes
        prev_energies : list of float
            Previous energies for all nodes
        forces : list of ndarray
            Current forces for all nodes
        """
        if prev_energies is None or None in prev_energies:
            return
        
        for i, (curr_e, prev_e) in enumerate(zip(energies, prev_energies)):
            if prev_e is None:
                continue
                
            # Calculate energy change
            energy_change = prev_e - curr_e
            force_magnitude = norm(forces[i].flatten())
            
            # Adjust step size based on energy change
            if energy_change > 0:  # Energy decreased (good)
                # Increase step size, but more conservatively for larger forces
                increase_factor = 1.2 * np.exp(-0.1 * force_magnitude)
                self.node_step_sizes[i] = min(
                    self.node_step_sizes[i] * (1.0 + increase_factor),
                    self.max_step_size
                )
                print(f"node {i}: Energy decreased by {energy_change:.6e}, increasing step size to {self.node_step_sizes[i]:.4f}")
            else:  # Energy increased (bad)
                # Decrease step size more aggressively for larger energy increases
                decrease_factor = self.step_descent_factor * (1.0 + 0.5 * abs(energy_change))
                self.node_step_sizes[i] = max(
                    self.node_step_sizes[i] * decrease_factor,
                    self.min_step_size
                )
                print(f"node {i}: Energy increased by {abs(energy_change):.6e}, decreasing step size to {self.node_step_sizes[i]:.4f}")
    
    def determine_step(self, directions, step_sizes):
        """Determine step vectors from directions and step sizes, constrained by maxstep.
        
        Parameters:
        ----------
        directions : list of ndarray
            Direction vectors for all nodes
        step_sizes : ndarray
            Step sizes for all nodes
            
        Returns:
        -------
        list of ndarray
            Step vectors for all nodes
        """
        steps = []
        
        for i, direction in enumerate(directions):
            # Normalize direction
            direction_norm = norm(direction.flatten())
            if direction_norm < 1e-10:  # Avoid division by zero
                steps.append(np.zeros_like(direction))
                continue
                
            normalized_direction = direction / direction_norm
            
            # Apply step size
            step = normalized_direction * step_sizes[i]
            
            # Apply maxstep constraint if needed
            step_length = norm(step.flatten())
            if step_length > self.maxstep:
                step = step * (self.maxstep / step_length)
                print(f"node {i}: Step constrained by maxstep={self.maxstep}")
            
            steps.append(step)
        
        return steps
    
    def CG_NEB_calc(self, geometry_num_list, total_force_list, pre_total_velocity, 
                   optimize_num, total_velocity, cos_list, 
                   biased_energy_list, pre_biased_energy_list, pre_geom):
        """Calculate step using Conjugate Gradient for NEB.
        
        Parameters:
        ----------
        geometry_num_list : ndarray
            Current geometry coordinates for all nodes
        total_force_list : ndarray
            Current forces for all nodes
        pre_total_velocity : ndarray
            Previous velocities for all nodes
        optimize_num : int
            Current optimization iteration number
        total_velocity : ndarray
            Current velocities for all nodes
        cos_list : ndarray
            Cosines between adjacent nodes
        biased_energy_list : ndarray
            Current energy for all nodes
        pre_biased_energy_list : ndarray
            Previous energy for all nodes
        pre_geom : ndarray
            Previous geometry coordinates for all nodes
            
        Returns:
        -------
        ndarray
            Updated geometry coordinates for all nodes
        """
        print(f"\n{'='*50}\nNEB-CG Iteration {self.iter}\n{'='*50}")
        
        # Get number of nodes
        num_nodes = len(geometry_num_list)
        
        # Initialize data structures if first iteration
        if self.Initialization:
            self.initialize_data(num_nodes)
            self.Initialization = False
            print("First iteration - using steepest descent")
        
        # Adjust step sizes based on energy changes (if not first iteration)
        if not self.Initialization and self.iter > 0 and self.prev_energies[0] is not None:
            self.adjust_step_sizes(biased_energy_list, self.prev_energies, total_force_list)
        
        # Compute conjugate gradient directions for each node
        cg_directions = []
        for i, force in enumerate(total_force_list):
            direction = self.compute_cg_direction(force, i)
            cg_directions.append(direction)
        
        # Determine step vectors from directions and step sizes
        move_vectors = self.determine_step(cg_directions, self.node_step_sizes)
        
        # Store current data for next iteration
        for i in range(num_nodes):
            self.prev_forces[i] = total_force_list[i].copy()
            self.prev_directions[i] = cg_directions[i].copy()
            self.prev_positions[i] = geometry_num_list[i].copy()
            self.prev_energies[i] = biased_energy_list[i]
        
        
        if self.TR_NEB:
            move_vectors = self.TR_NEB.TR_calc(geometry_num_list, total_force_list, move_vectors, 
                                             biased_energy_list, pre_biased_energy_list, pre_geom)
        
        # Update geometry using move vectors
        new_geometry = []
        for i, (geom, move) in enumerate(zip(geometry_num_list, move_vectors)):
            new_geom = geom + move
            new_geometry.append(new_geom)
        
        # Convert to numpy array and apply unit conversion
        new_geometry = np.array(new_geometry)
        new_geometry *= self.bohr2angstroms
        
        self.iter += 1
        return new_geometry