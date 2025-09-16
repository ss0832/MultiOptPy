import numpy as np
from numpy.linalg import norm

from multioptpy.Parameters.parameter import UnitValueLib

class AFIRE_NEB:
    """Adaptive Fast Inertial Relaxation Engine for Nudged Elastic Band calculations.
    
    This implementation extends the standard FIRE algorithm with adaptive parameters
    for each image in the NEB chain, improving convergence for reaction paths that have
    regions with different curvatures and energy barriers.
    
    Reference: 
    E. Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006)
    with adaptive extensions for NEB.
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.Initialization = True
        self.iter = 0
        
        # FIRE parameters
        self.a_start = config.get("a_start", 0.1)
        self.f_inc = config.get("f_inc", 1.1)
        self.f_dec = config.get("f_dec", 0.5)
        self.f_a = config.get("f_a", 0.99)
        self.dt_start = config.get("dt_start", 0.1)
        self.dt_max = config.get("dt_max", 1.0)
        self.n_min = config.get("n_min", 5)
        
        # Per-image adaptive parameters
        self.a_images = None  # α values for each image
        self.dt_images = None  # Time steps for each image
        self.n_pos_images = None  # Consecutive positive dot products counter for each image
        
        # Storage for velocities and previous forces
        self.velocities = None
        self.prev_forces = None
        self.prev_positions = None
        
        # Maximum step size constraint
        self.maxstep = config.get("maxstep", 0.1)
        
        # Unit conversion
        self.bohr2angstroms = UnitValueLib().bohr2angstroms 
        if 'bohr2angstroms' in config:
            self.bohr2angstroms = config['bohr2angstroms']
        
        # For compatibility with other NEB implementations
        self.TR_NEB = config.get("TR_NEB", None)
        
        print(f"Initialized Adaptive FIRE optimizer for NEB with "
              f"a_start={self.a_start}, dt_start={self.dt_start}, maxstep={self.maxstep}")
    
    def initialize_data(self, num_images):
        """Initialize data structures for the optimization.
        
        Parameters:
        ----------
        num_images : int
            Number of images in the NEB calculation
        """
        # Initialize per-image parameters
        self.a_images = np.ones(num_images) * self.a_start
        self.dt_images = np.ones(num_images) * self.dt_start
        self.n_pos_images = np.zeros(num_images, dtype=int)
        
        # Initialize velocities to zero with correct shapes
        self.velocities = [np.zeros_like(img) for img in range(num_images)]
        
        # Initialize storage for previous values
        self.prev_forces = [None] * num_images
        self.prev_positions = [None] * num_images
        
        print(f"Initialized AFIRE parameters for {num_images} images")
    
    def update_velocities_and_positions(self, positions, forces, dt):
        """Update velocities and positions using the FIRE algorithm for multiple images.
        
        Parameters:
        ----------
        positions : list of ndarray
            Current positions for all images
        forces : list of ndarray
            Current forces for all images
        dt : ndarray
            Time steps for each image
            
        Returns:
        -------
        tuple
            (new_velocities, steps) - Updated velocities and position steps for all images
        """
        new_velocities = []
        steps = []
        
        # Process each image independently
        for i in range(len(positions)):
            # Get current force and position
            force = forces[i]
            position = positions[i]
            
            # Initialize velocity if needed
            if self.velocities[i] is None or not isinstance(self.velocities[i], np.ndarray):
                self.velocities[i] = np.zeros_like(position)
            
            velocity = self.velocities[i]
            
            # Ensure velocity has the same shape as force
            if velocity.shape != force.shape:
                velocity = np.zeros_like(force)
                print(f"Warning: Resetting velocity for image {i} due to shape mismatch")
            
            # Calculate power (P = F·v)
            power = np.dot(force.flatten(), velocity.flatten())
            
            # Calculate velocity magnitude and force magnitude
            v_norm = norm(velocity.flatten())
            f_norm = norm(force.flatten())
            
            # Update velocity using FIRE algorithm
            if f_norm > 0 and v_norm > 0:
                # Calculate normalized force and velocity vectors
                f_hat = force.flatten() / f_norm
                v_hat = velocity.flatten() / v_norm
                
                # Mix velocity and force direction according to α
                mixed_v = (1.0 - self.a_images[i]) * velocity + self.a_images[i] * v_norm * f_hat.reshape(velocity.shape)
            else:
                mixed_v = velocity.copy()
            
            # Perform velocity Verlet integration
            new_v = mixed_v + 0.5 * dt[i] * force
            
            # Calculate step
            step = dt[i] * new_v
            
            # Apply max step constraint
            step_length = norm(step.flatten())
            if step_length > self.maxstep:
                step = step * (self.maxstep / step_length)
            
            # Store results
            new_velocities.append(new_v)
            steps.append(step)
            
            # Update FIRE parameters based on power
            if power > 0:  # Moving in right direction
                self.n_pos_images[i] += 1
                if self.n_pos_images[i] > self.n_min:
                    # Increase time step and decrease mixing parameter
                    self.dt_images[i] = min(self.dt_images[i] * self.f_inc, self.dt_max)
                    self.a_images[i] *= self.f_a
            else:  # Moving in wrong direction
                # Reset parameters
                self.n_pos_images[i] = 0
                self.dt_images[i] *= self.f_dec
                self.a_images[i] = self.a_start
                # Reset velocity (stop inertial motion against force)
                new_velocities[i] = np.zeros_like(velocity)
        
        return new_velocities, steps
    
    def determine_step(self, dr):
        """Ensure step sizes are within maxstep.
        
        Parameters:
        ----------
        dr : list of ndarray
            Step vectors for all images
            
        Returns:
        -------
        list of ndarray
            Step vectors constrained by maxstep if necessary
        """
        if self.maxstep is None:
            return dr
        
        # Handle list of step vectors
        result = []
        for step in dr:
            # Calculate step length
            step_reshaped = step.reshape(-1, 3) if step.size % 3 == 0 else step
            steplengths = np.sqrt((step_reshaped**2).sum(axis=1))
            longest_step = np.max(steplengths)
            
            # Scale step if necessary
            if longest_step > self.maxstep:
                result.append(step * (self.maxstep / longest_step))
            else:
                result.append(step)
        return result
    
    def AFIRE_NEB_calc(self, geometry_num_list, total_force_list, pre_total_velocity, 
                      optimize_num, total_velocity, cos_list, 
                      biased_energy_list, pre_biased_energy_list, pre_geom):
        """Calculate step using Adaptive FIRE for NEB.
        
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
        print(f"\n{'='*50}\nNEB-AFIRE Iteration {self.iter}\n{'='*50}")
        
        # Get number of images
        num_images = len(geometry_num_list)
        
        # Initialize data structures if first iteration
        if self.Initialization:
            self.initialize_data(num_images)
            self.Initialization = False
            print("First iteration - initializing AFIRE")
            
            # Initialize velocities from pre_total_velocity if available
            if pre_total_velocity is not None and len(pre_total_velocity) >= num_images:
                self.velocities = [vel.copy() if vel is not None else np.zeros_like(geometry_num_list[i]) 
                                  for i, vel in enumerate(pre_total_velocity[:num_images])]
            else:
                self.velocities = [np.zeros_like(geom) for geom in geometry_num_list]
        
        # Track adaptive parameters
        min_dt = np.min(self.dt_images)
        max_dt = np.max(self.dt_images)
        avg_a = np.mean(self.a_images)
        print(f"FIRE parameters - dt range: [{min_dt:.4f}, {max_dt:.4f}], avg α: {avg_a:.4f}")
        
        # Update velocities and calculate steps
        new_velocities, move_vectors = self.update_velocities_and_positions(
            geometry_num_list, total_force_list, self.dt_images
        )
        
        # Apply trust region correction if TR_NEB is provided (for compatibility)
        if self.TR_NEB:
            move_vectors = self.TR_NEB.TR_calc(geometry_num_list, total_force_list, move_vectors,
                                             biased_energy_list, pre_biased_energy_list, pre_geom)
        
        # Store updated velocities
        self.velocities = new_velocities
        
        # Update geometry using move vectors
        new_geometry = []
        for i, (geom, move) in enumerate(zip(geometry_num_list, move_vectors)):
            new_geom = geom + move
            new_geometry.append(new_geom)
            
            # Store current force and position for next iteration
            self.prev_forces[i] = total_force_list[i].copy()
            self.prev_positions[i] = geometry_num_list[i].copy()
        
        # Convert to numpy array and apply unit conversion
        new_geometry = np.array(new_geometry)
        new_geometry *= self.bohr2angstroms
        
        self.iter += 1
        return new_geometry