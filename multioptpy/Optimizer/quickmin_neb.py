import numpy as np
from numpy.linalg import norm

from multioptpy.Parameters.parameter import UnitValueLib

class QuickMin_NEB:
    """QuickMin optimizer for Nudged Elastic Band calculations.
    
    QuickMin is a minimization algorithm that combines molecular dynamics with 
    steepest descent optimization. It retains the velocity component in the direction
    of the force only when it's aligned with the force, providing efficient energy 
    minimization for complex systems.
    
    Reference:
    D. Sheppard, R. Terrell, and G. Henkelman, J. Chem. Phys. 128, 134106 (2008)
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.Initialization = True
        self.iter = 0
        
        # QuickMin parameters
        self.dt = config.get("dt", 0.1)  # Time step
        self.dt_max = config.get("dt_max", 0.2)  # Maximum time step
        self.dt_min = config.get("dt_min", 0.01)  # Minimum time step
        self.dt_grow = config.get("dt_grow", 1.1)  # Time step growth factor
        self.dt_shrink = config.get("dt_shrink", 0.5)  # Time step shrink factor
        self.velocity_mixing = config.get("velocity_mixing", 0.9)  # Velocity mixing factor
        
        # Adaptive time steps for each image
        self.dt_images = None
        
        # Maximum step size constraint
        self.maxstep = config.get("maxstep", 0.1)
        
        # Storage for velocities and previous data
        self.velocities = None
        self.prev_forces = None
        self.prev_positions = None
        self.prev_energies = None
        
        # Unit conversion
        self.bohr2angstroms = UnitValueLib().bohr2angstroms 
        if 'bohr2angstroms' in config:
            self.bohr2angstroms = config['bohr2angstroms']
        
        # For compatibility with other NEB implementations
        self.TR_NEB = config.get("TR_NEB", None)
        
        print(f"Initialized QuickMin optimizer for NEB with dt={self.dt}, maxstep={self.maxstep}")
        print(f"Using per-image adaptive time steps with growth factor={self.dt_grow}, shrink factor={self.dt_shrink}")
    
    def initialize_data(self, num_images):
        """Initialize data structures for the optimization.
        
        Parameters:
        ----------
        num_images : int
            Number of images in the NEB calculation
        """
        # Initialize time steps for each image
        self.dt_images = np.ones(num_images) * self.dt
        
        # Initialize velocities with zero (no initial momentum)
        self.velocities = [np.zeros_like(img) for img in range(num_images)]
        
        # Initialize storage for previous data
        self.prev_forces = [None] * num_images
        self.prev_positions = [None] * num_images
        self.prev_energies = [None] * num_images
        
        print(f"Initialized QuickMin data for {num_images} images")
    
    def update_velocities_and_positions(self, positions, forces, energies):
        """Update velocities and positions using the QuickMin algorithm.
        
        Parameters:
        ----------
        positions : list of ndarray
            Current positions for all images
        forces : list of ndarray
            Current forces for all images
        energies : list of float
            Current energies for all images
            
        Returns:
        -------
        tuple
            (new_velocities, steps) - Updated velocities and position steps for all images
        """
        new_velocities = []
        steps = []
        
        # Process each image independently
        for i, (position, force) in enumerate(zip(positions, forces)):
            # Initialize velocity if needed
            if self.velocities[i] is None or not isinstance(self.velocities[i], np.ndarray):
                self.velocities[i] = np.zeros_like(position)
            
            # Ensure correct shape
            if self.velocities[i].shape != force.shape:
                self.velocities[i] = np.zeros_like(force)
            
            velocity = self.velocities[i]
            
            # Calculate dot product of velocity and force
            v_dot_f = np.dot(velocity.flatten(), force.flatten())
            
            # Reset velocity if moving against the force
            if v_dot_f <= 0:
                velocity = np.zeros_like(velocity)
                v_dot_f = 0
            
            # Calculate force magnitude
            f_norm = norm(force.flatten())
            
            # Project velocity onto force direction
            if f_norm > 1e-10:
                f_hat = force.flatten() / f_norm
                v_parallel = v_dot_f * f_hat
                
                # Update velocity (only keep component parallel to force)
                new_v = v_parallel.reshape(velocity.shape)
                
                # Add force contribution for this time step
                new_v += self.dt_images[i] * force
            else:
                # No force, maintain zero velocity
                new_v = np.zeros_like(velocity)
            
            # Apply velocity mixing for stability
            if self.velocity_mixing < 1.0:
                new_v = self.velocity_mixing * new_v + (1.0 - self.velocity_mixing) * velocity
            
            # Calculate step
            step = self.dt_images[i] * new_v
            
            # Apply maxstep constraint
            step_length = norm(step.flatten())
            if step_length > self.maxstep:
                step = step * (self.maxstep / step_length)
            
            # Adjust time step based on energy change if we have previous data
            if self.prev_energies[i] is not None:
                energy_change = self.prev_energies[i] - energies[i]
                if energy_change > 0:  # Energy decreased (good)
                    # Increase time step
                    self.dt_images[i] = min(self.dt_images[i] * self.dt_grow, self.dt_max)
                else:  # Energy increased (bad)
                    # Decrease time step and reset velocity
                    self.dt_images[i] = max(self.dt_images[i] * self.dt_shrink, self.dt_min)
                    new_v = np.zeros_like(velocity)  # Reset velocity completely
            
            # Store results
            new_velocities.append(new_v)
            steps.append(step)
        
        return new_velocities, steps
    
    def QuickMin_NEB_calc(self, geometry_num_list, total_force_list, pre_total_velocity, 
                         optimize_num, total_velocity, cos_list, 
                         biased_energy_list, pre_biased_energy_list, pre_geom):
        """Calculate step using QuickMin for NEB.
        
        Parameters:
        ----------
        geometry_num_list : ndarray
            Current geometry coordinates for all images
        total_force_list : ndarray
            Current forces for all images
        pre_total_velocity : ndarray
            Previous velocities for all images
        optimize_num : int
            Current optimization iteration number
        total_velocity : ndarray
            Current velocities for all images
        cos_list : ndarray
            Cosines between adjacent images
        biased_energy_list : ndarray
            Current energy for all images
        pre_biased_energy_list : ndarray
            Previous energy for all images
        pre_geom : ndarray
            Previous geometry coordinates for all images
            
        Returns:
        -------
        ndarray
            Updated geometry coordinates for all images
        """
        print(f"\n{'='*50}\nNEB-QuickMin Iteration {self.iter}\n{'='*50}")
        
        # Get number of images
        num_images = len(geometry_num_list)
        
        # Initialize data structures if first iteration
        if self.Initialization:
            self.initialize_data(num_images)
            self.Initialization = False
            print("First iteration - initializing QuickMin")
            
            # Initialize velocities from pre_total_velocity if available
            if pre_total_velocity is not None and len(pre_total_velocity) >= num_images:
                self.velocities = [vel.copy() if vel is not None else np.zeros_like(geometry_num_list[i]) 
                                  for i, vel in enumerate(pre_total_velocity[:num_images])]
        
        # Print current time steps
        min_dt = np.min(self.dt_images)
        max_dt = np.max(self.dt_images)
        avg_dt = np.mean(self.dt_images)
        print(f"QuickMin time steps - range: [{min_dt:.4f}, {max_dt:.4f}], avg: {avg_dt:.4f}")
        
        # Update velocities and calculate steps
        new_velocities, move_vectors = self.update_velocities_and_positions(
            geometry_num_list, total_force_list, biased_energy_list
        )
        
        # Apply trust region correction if TR_NEB is provided
        if self.TR_NEB:
            move_vectors = self.TR_NEB.TR_calc(geometry_num_list, total_force_list, move_vectors,
                                             biased_energy_list, pre_biased_energy_list, pre_geom)
        
        # Store current data for next iteration
        for i in range(num_images):
            self.velocities[i] = new_velocities[i]
            self.prev_forces[i] = total_force_list[i].copy()
            self.prev_positions[i] = geometry_num_list[i].copy()
            self.prev_energies[i] = biased_energy_list[i]
        
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