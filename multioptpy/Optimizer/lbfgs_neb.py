import numpy as np
from numpy.linalg import norm

from multioptpy.Parameters.parameter import UnitValueLib

class LBFGS_NEB:
    """Limited-memory BFGS optimizer for Nudged Elastic Band calculations.
    
    This implementation uses standard L-BFGS algorithm to approximate the inverse Hessian
    matrix using a limited amount of memory, maintaining separate approximations
    for each image in the NEB chain.
    
    Step length is controlled through direct scaling rather than trust regions.
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.Initialization = True
        self.iter = 0
        
        # Set default parameters
        self.memory = config.get("memory", 40)  # Number of previous steps to remember
        self.maxstep = config.get("maxstep", 0.1)  # Maximum step size
        
        # Scaling parameters
        self.initial_step_scale = config.get("initial_step_scale", 1.0)
        self.step_scale = self.initial_step_scale  # Current step scaling factor
        
        # Storage for L-BFGS history per image
        self.s_images = []  # Position differences per image
        self.y_images = []  # Force differences per image
        self.rho_images = []  # 1 / (y_k^T s_k) per image
        self.gamma_images = []  # Scaling factors per image
        
        # Unit conversion
        self.bohr2angstroms = UnitValueLib().bohr2angstroms 
        if 'bohr2angstroms' in config:
            self.bohr2angstroms = config['bohr2angstroms']
        
        self.TR_NEB = config.get("TR_NEB", None)  # Keep this for compatibility
        
        print(f"Initialized LBFGS_NEB optimizer with memory={self.memory}, "
              f"maxstep={self.maxstep}")
    
    def initialize_per_node_data(self, num_images):
        """Initialize data structures for per-node L-BFGS history management.
        
        Parameters:
        ----------
        num_images : int
            Number of images in the NEB calculation
        """
        # Initialize L-BFGS history storage for each image
        self.s_images = [[] for _ in range(num_images)]
        self.y_images = [[] for _ in range(num_images)]
        self.rho_images = [[] for _ in range(num_images)]
        self.gamma_images = [1.0] * num_images  # Initial gamma value for each image
        
        # Previous data for updates
        self.prev_forces = [None] * num_images
        self.prev_positions = [None] * num_images
        self.prev_energies = [None] * num_images
        
        print(f"Initialized L-BFGS history storage for {num_images} images")
    
    def update_vectors(self, displacements, delta_forces):
        """Update the vectors used for the L-BFGS approximation for each image.
        
        Parameters:
        ----------
        displacements : list of ndarray
            Position displacement vectors for each image
        delta_forces : list of ndarray
            Force difference vectors for each image
        """
        # Initialize storage if this is the first update
        if not self.s_images:
            self.initialize_per_node_data(len(displacements))
            
        # Update vectors for each image
        for i, (s, y) in enumerate(zip(displacements, delta_forces)):
            # Flatten vectors
            s_flat = s.flatten()
            y_flat = y.flatten()
            
            # Calculate rho = 1 / (y^T * s)
            dot_product = np.dot(y_flat, s_flat)
            if abs(dot_product) < 1e-10:
                # Avoid division by very small numbers
                rho = 1000.0
                print(f"Warning: Image {i} has very small y^T·s value ({dot_product:.2e})")
            else:
                rho = 1.0 / dot_product
                
                # Update gamma (scaling factor for initial Hessian approximation)
                y_norm_squared = np.dot(y_flat, y_flat)
                if y_norm_squared > 1e-10:
                    self.gamma_images[i] = y_norm_squared / dot_product
                    print(f"Updated gamma for image {i} to {self.gamma_images[i]:.4f}")
            
            # Add to history
            self.s_images[i].append(s_flat)
            self.y_images[i].append(y_flat)
            self.rho_images[i].append(rho)
            
            # Remove oldest vectors if exceeding memory limit
            if len(self.s_images[i]) > self.memory:
                self.s_images[i].pop(0)
                self.y_images[i].pop(0)
                self.rho_images[i].pop(0)
    
    def compute_lbfgs_step(self, force, image_idx):
        """Compute the L-BFGS step direction for a single image.
        
        Parameters:
        ----------
        force : ndarray
            Current force vector for the image
        image_idx : int
            Index of the image in the NEB chain
            
        Returns:
        -------
        ndarray
            Step direction
        """
        # Flatten force vector
        f = force.flatten()
        orig_shape = force.shape
        
        # If we have no history yet for this image, just return force direction
        if not self.s_images or image_idx >= len(self.s_images) or len(self.s_images[image_idx]) == 0:
            return f.reshape(orig_shape)
        
        # Get history for this image
        s_list = self.s_images[image_idx]
        y_list = self.y_images[image_idx]
        rho_list = self.rho_images[image_idx]
        gamma = self.gamma_images[image_idx]
        
        # Apply two-loop recursion algorithm for L-BFGS
        q = -f  # Negative gradient (negative of negative force)
        k = len(s_list)
        alpha = np.zeros(k)
        
        try:
            # First loop
            for i in range(k-1, -1, -1):
                alpha[i] = rho_list[i] * np.dot(s_list[i], q)
                q = q - alpha[i] * y_list[i]
            
            # Initial Hessian approximation
            r = gamma * q
            
            # Second loop
            for i in range(k):
                beta = rho_list[i] * np.dot(y_list[i], r)
                r = r + s_list[i] * (alpha[i] - beta)
            
            # r is now the product B^(-1) * g where B is the Hessian approximation
            return r.reshape(orig_shape)
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Error in L-BFGS calculation for image {image_idx}: {e}")
            print("Falling back to steepest descent")
            # Fallback to steepest descent
            return f.reshape(orig_shape)
    
    def determine_step(self, dr):
        """Determine step to take according to maxstep.
        
        Parameters:
        ----------
        dr : ndarray or list of ndarray
            Step vector(s)
            
        Returns:
        -------
        ndarray or list of ndarray
            Step vector(s) constrained by maxstep if necessary
        """
        if self.maxstep is None:
            return dr
        
        # Handle single step vector or list of step vectors
        if isinstance(dr, list):
            result = []
            for step in dr:
                # Calculate step lengths
                step_reshaped = step.reshape(-1, 3) if step.size % 3 == 0 else step
                steplengths = np.sqrt((step_reshaped**2).sum(axis=1))
                longest_step = np.max(steplengths)
                
                # Scale step if necessary
                if longest_step > self.maxstep:
                    result.append(step * (self.maxstep / longest_step))
                else:
                    result.append(step)
            return result
        else:
            # Single step vector
            step_reshaped = dr.reshape(-1, 3) if dr.size % 3 == 0 else dr
            steplengths = np.sqrt((step_reshaped**2).sum(axis=1))
            longest_step = np.max(steplengths)
            
            # Scale step if necessary
            if longest_step > self.maxstep:
                return dr * (self.maxstep / longest_step)
            return dr
    
    def adjust_step_scale(self, energies, prev_energies):
        """Adjust the global step scaling factor based on energy changes.
        
        Parameters:
        ----------
        energies : list of float
            Current energies for all images
        prev_energies : list of float
            Previous energies for all images
        """
        if prev_energies is None or None in prev_energies:
            return
        
        # Calculate energy changes
        energy_changes = [prev - curr for prev, curr in zip(prev_energies, energies) if prev is not None and curr is not None]
        
        if not energy_changes:
            return
        
        # Count improvements and deteriorations
        improvements = sum(1 for change in energy_changes if change > 0)
        deteriorations = sum(1 for change in energy_changes if change < 0)
        
        # Adjust scale based on overall improvement ratio
        if improvements > deteriorations:
            # More improvements than deteriorations - increase step scale
            self.step_scale = min(self.step_scale * 1.1, 2.0)
            print(f"Energy improved for {improvements} images, increasing step scale to {self.step_scale:.4f}")
        elif deteriorations > improvements:
            # More deteriorations than improvements - decrease step scale
            self.step_scale = max(self.step_scale * 0.5, 0.1)
            print(f"Energy deteriorated for {deteriorations} images, decreasing step scale to {self.step_scale:.4f}")
        else:
            # Equal number - maintain current scale
            print(f"Maintaining current step scale at {self.step_scale:.4f}")
    
    def LBFGS_NEB_calc(self, geometry_num_list, total_force_list, pre_total_velocity, 
                       optimize_num, total_velocity, cos_list, 
                       biased_energy_list, pre_biased_energy_list, pre_geom):
        """Calculate step using standard L-BFGS for NEB.
        
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
        print(f"\n{'='*50}\nNEB-LBFGS Iteration {self.iter}\n{'='*50}")
        
        # Get number of images
        num_images = len(geometry_num_list)
        
        # Initialize data structures if first iteration
        if self.Initialization:
            self.initialize_per_node_data(num_images)
            self.Initialization = False
            print("First iteration")
        
        # Adjust step scale based on energy changes (if not first iteration)
        if not self.Initialization and self.iter > 0 and self.prev_energies[0] is not None:
            self.adjust_step_scale(biased_energy_list, self.prev_energies)
        
        # Store current energies for next iteration
        self.prev_energies = biased_energy_list.copy()
        
        # List to store move vectors for each image
        move_vectors = []
        
        # Process each image
        for i, force in enumerate(total_force_list):
            # Compute L-BFGS step direction
            step_direction = self.compute_lbfgs_step(force, i)
            
            # Scale step by the current step scaling factor
            step = self.step_scale * step_direction
            
            move_vectors.append(step)
        
        # Apply maxstep constraint if needed
        move_vectors = self.determine_step(move_vectors)
        
        # If this is not the first iteration, update L-BFGS vectors
        if optimize_num > 0:
            # Calculate displacement and force difference for each image
            displacements = []
            delta_forces = []
            
            for i in range(num_images):
                if pre_geom is not None and pre_total_velocity is not None and len(pre_total_velocity) > 0:
                    # Get previous force if available
                    if len(pre_total_velocity) > i:
                        prev_force = pre_total_velocity[i]
                        curr_force = total_force_list[i]
                        delta_force = curr_force - prev_force
                        
                        # Get displacement
                        curr_geom = geometry_num_list[i]
                        prev_geom = pre_geom[i]
                        displacement = curr_geom - prev_geom
                        
                        displacements.append(displacement)
                        delta_forces.append(delta_force)
            
            if displacements and delta_forces:
                self.update_vectors(displacements, delta_forces)
        
        # Apply trust region correction if TR_NEB is provided for compatibility
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