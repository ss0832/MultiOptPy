import numpy as np
import os

# Import from BiasPotPy
from parameter import UnitValueLib, covalent_radii_lib, element_number  
from Optimizer.fire import FIRE

class ONIOMCalculation:
    """
    Class for ONIOM calculations with microiterations.
    
    ONIOM method divides the system into high and low layers.
    The high layer calculations are performed externally,
    while this class handles layer division, capping, and low layer optimization.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize ONIOM calculation.
        
        Args:
            **kwargs: Configuration parameters
        """
        self.UVL = UnitValueLib()
        self.bohr2angstroms = self.UVL.bohr2angstroms
        
        # Default parameters
        self.low_layer_method = kwargs.get("LOW_METHOD", "GFN1-xTB")
        self.max_iterations = kwargs.get("MAX_ITERATIONS", 1000)
        self.gradient_conv = kwargs.get("GRADIENT_CONV", 3.0e-4)  # Maximum gradient threshold
        self.rms_gradient_conv = kwargs.get("RMS_GRADIENT_CONV", 2.0e-4)  # RMS gradient threshold
        self.displacement_conv = kwargs.get("DISPLACEMENT_CONV", 1.5e-3)  # Maximum displacement threshold
        self.rms_displacement_conv = kwargs.get("RMS_DISPLACEMENT_CONV", 1.0e-3)  # RMS displacement threshold
        
        self.output_dir = kwargs.get("OUTPUT_DIR", "oniom_output")
        self.verbose = kwargs.get("VERBOSE", True)
        
        
        # Initialize data structures
        self.coordinates = None  # Full system coordinates (Bohr)
        self.elements = None  # Full system elements
        self.high_layer_indices = None  # High layer atom indices (0-indexed)
        self.high_layer_coords = None  # High layer coordinates with H caps (Bohr)
        self.high_layer_elements = None  # High layer elements with H caps
        
        # Bond and cap information
        self.bonds = []
        self.boundary_bonds = []
        self.h_caps_info = []  # (cap_idx, high_atom_idx, low_atom_idx)
        
        # Calculation instances
        self.low_calc = None
        self.optimizer = FIRE()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For storing results
        self.iteration_energies = []
        self.iteration_gradients = []
        self.iteration_geometries = []
        
     
    
    def update_output_dir(self, output_dir):
        """Update the output directory for ONIOM calculations."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def setup(self, coordinates, elements, high_layer_indices, charge=0, multiplicity=1):
        """
        Set up ONIOM calculation by dividing the system into layers.
        
        Args:
            coordinates: Molecular coordinates in Bohr (shape: Nx3)
            elements: List of element symbols (length: N)
            high_layer_indices: Indices of atoms in high layer (1-indexed)
            charge: Total molecular charge
            multiplicity: Spin multiplicity
            
        Returns:
            Dictionary containing high layer model information
        """
        self.coordinates = coordinates
        self.elements = elements
        self.charge_and_multiplicity = [charge, multiplicity]
        
        # Convert 1-indexed to 0-indexed for internal use
        self.high_layer_indices = {i-1 for i in high_layer_indices}
        
        if self.verbose:
            print(f"Setting up ONIOM calculation")
            print(f"High layer: {len(high_layer_indices)} atoms")
            print(f"Total system: {len(elements)} atoms")
            print(f"Low layer method: {self.low_layer_method}")
        
        # Initialize calculation objects
        #### self.low_calc = Calculation(**self.calc_params)
        
        # Setup layers, detect bonds, and create caps
        self._setup_layers()
        
        
        return {
            "high_layer_coords": self.high_layer_coords,
            "high_layer_elements": self.high_layer_elements,
            "boundary_bonds": self.boundary_bonds,
            "h_caps_info": self.h_caps_info,
            "full_system_coords": self.coordinates,
            "full_system_elements": self.elements
        }
    
    def _setup_layers(self):
        """Set up high and low layers, detect bonds, and create hydrogen caps."""
        # Convert coordinates from Bohr to Angstrom for bond detection
        coords_angstrom = self.coordinates * self.bohr2angstroms
        
        # Detect bonds
        self.bonds = self._detect_bonds(coords_angstrom, self.elements)
        
        # Find boundary bonds
        self.boundary_bonds = self._find_boundary_bonds(self.bonds, self.high_layer_indices)
        
        # Generate capped high layer
        self.high_layer_coords, self.high_layer_elements, self.h_caps_info = self._generate_hydrogen_caps(
            coords_angstrom, self.elements, self.boundary_bonds, self.high_layer_indices
        )
        
        # Convert high layer coordinates back to Bohr
        self.high_layer_coords = self.high_layer_coords / self.bohr2angstroms
        
        if self.verbose:
            print(f"Detected {len(self.bonds)} bonds")
            print(f"Found {len(self.boundary_bonds)} boundary bonds")
            print(f"Added {len(self.h_caps_info)} hydrogen caps")
    
    def _detect_bonds(self, coordinates, elements, bond_tolerance=1.2):
        """
        Detect bonds based on distance criteria (vectorized version).
        
        Args:
            coordinates: Array of shape (N, 3) with atom coordinates (Angstrom)
            elements: List of element symbols
            bond_tolerance: Multiplier for sum of covalent radii to determine bond cutoff
            
        Returns:
            List of tuples (atom1_idx, atom2_idx) representing bonds (0-indexed)
        """
        num_atoms = len(elements)
        
        # Get covalent radii for all atoms (default to carbon if not found)
        radii = np.array([covalent_radii_lib(elem) for elem in elements]) * self.bohr2angstroms
        
        # Calculate pairwise distance threshold matrix
        threshold_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :]) * bond_tolerance
        
        # Calculate pairwise distances using vectorized operations
        delta_x = coordinates[:, 0, np.newaxis] - coordinates[:, 0]
        delta_y = coordinates[:, 1, np.newaxis] - coordinates[:, 1]
        delta_z = coordinates[:, 2, np.newaxis] - coordinates[:, 2]
        
        # Compute squared distances
        distances_squared = delta_x**2 + delta_y**2 + delta_z**2
        
        # Create a mask for bonds (where distance ≤ threshold)
        i, j = np.where(
            (distances_squared <= threshold_matrix**2) &  # distance ≤ threshold
            (np.triu(np.ones((num_atoms, num_atoms), dtype=bool), k=1))  # upper triangle only
        )
        
        # Create list of bond tuples
        bonds = list(zip(i.tolist(), j.tolist()))
        
        return bonds

    def _check_convergence(self, gradient, current_coords, prev_coords=None):
        """
        Check convergence using multiple criteria: gradient max/RMS and displacement max/RMS.
        
        Args:
            gradient: Current gradient array
            current_coords: Current coordinates array
            prev_coords: Previous coordinates array (None if first iteration)
        
        Returns:
            Tuple of (converged, metrics) where:
                converged: Boolean indicating if all convergence criteria are met
                metrics: Dictionary with convergence metrics
        """
        # Create mask for low layer atoms (those not in high layer)
        low_layer_mask = np.array([idx not in self.high_layer_indices for idx in range(len(self.elements))])
        
        # Calculate gradient metrics (only for low layer atoms)
        low_layer_gradient = gradient[low_layer_mask]
        max_gradient = np.max(np.abs(low_layer_gradient))
        rms_gradient = np.sqrt(np.mean(np.square(low_layer_gradient)))
        
        # Calculate displacement metrics (only if we have previous coordinates)
        max_displacement = 0.0
        rms_displacement = 0.0
        
        if prev_coords is not None:
            # Calculate displacement between current and previous coordinates
            displacements = current_coords - prev_coords
            # Only consider low layer atoms
            low_layer_displacements = displacements[low_layer_mask]
            # Calculate displacement metrics
            max_displacement = np.max(np.abs(low_layer_displacements))
            rms_displacement = np.sqrt(np.mean(np.square(low_layer_displacements)))
        
        # Check if all convergence criteria are met
        converged = (
            max_gradient < self.gradient_conv and
            rms_gradient < self.rms_gradient_conv and
            (prev_coords is None or max_displacement < self.displacement_conv) and
            (prev_coords is None or rms_displacement < self.rms_displacement_conv)
        )
        
        # Return convergence status and metrics for reporting
        metrics = {
            "max_gradient": max_gradient,
            "rms_gradient": rms_gradient,
            "max_displacement": max_displacement,
            "rms_displacement": rms_displacement
        }
        
        return converged, metrics
  
    def _find_boundary_bonds(self, bonds, high_layer_indices):
        """
        Find bonds that cross between high and low layers (vectorized version).
        
        Args:
            bonds: List of tuples (atom1_idx, atom2_idx) representing bonds (0-indexed)
            high_layer_indices: Set of atom indices in high layer (0-indexed)
            
        Returns:
            List of boundary bond tuples (atom1_idx, atom2_idx)
        """
        if not bonds:
            return []
        
        # Convert bonds to NumPy array for vectorized operations
        bonds_array = np.array(bonds)
        
        # Create boolean arrays indicating if atoms are in high layer
        atom1_in_high = np.isin(bonds_array[:, 0], list(high_layer_indices))
        atom2_in_high = np.isin(bonds_array[:, 1], list(high_layer_indices))
        
        # Find bonds where exactly one atom is in high layer (XOR operation)
        is_boundary = np.logical_xor(atom1_in_high, atom2_in_high)
        
        # Filter bonds using the boundary mask
        boundary_bonds = bonds_array[is_boundary].tolist()
        
        # Convert back to list of tuples
        boundary_bonds = [tuple(bond) for bond in boundary_bonds]
        
        return boundary_bonds
    
    def _generate_hydrogen_caps(self, coordinates, elements, boundary_bonds, high_layer_indices):
        """
        Generate hydrogen cap atoms for boundary bonds (vectorized version).
        
        Args:
            coordinates: Array of shape (N, 3) with atom coordinates (Angstrom)
            elements: List of element symbols
            boundary_bonds: List of boundary bond tuples
            high_layer_indices: Set of atom indices in high layer (0-indexed)
            
        Returns:
            Tuple containing:
                - Array of coordinates for high layer atoms + H caps (Angstrom)
                - List of element symbols for high layer atoms + H caps
                - List of (cap_idx, high_atom_idx, low_atom_idx) tuples
        """
        if not boundary_bonds:
            # If no boundary bonds, return only high layer atoms
            high_layer_mask = np.zeros(len(elements), dtype=bool)
            high_layer_indices_list = list(high_layer_indices)
            high_layer_mask[high_layer_indices_list] = True
            high_layer_coords = coordinates[high_layer_mask]
            high_layer_elements = [elements[i] for i in high_layer_indices]
            return high_layer_coords, high_layer_elements, []
        
        # Create high layer mask using vectorized operation
        high_layer_mask = np.zeros(len(elements), dtype=bool)
        high_layer_indices_list = list(high_layer_indices)
        high_layer_mask[high_layer_indices_list] = True
        
        # Extract high layer coordinates and elements
        high_layer_coords = coordinates[high_layer_mask]
        high_layer_elements = [elements[i] for i in range(len(elements)) if high_layer_mask[i]]
        
        # Create mapping from original indices to new indices in high layer array
        high_layer_indices_array = np.array(list(high_layer_indices))
        orig_to_high_map = {idx: i for i, idx in enumerate(high_layer_indices_array)}
        
        # Determine high and low atoms for each boundary bond
        high_atoms = []
        low_atoms = []
        
        for bond in boundary_bonds:
            i, j = bond
            if i in high_layer_indices:
                high_atoms.append(i)
                low_atoms.append(j)
            else:
                high_atoms.append(j)
                low_atoms.append(i)
        
        high_atoms = np.array(high_atoms)
        low_atoms = np.array(low_atoms)
        
        # Extract coordinates for high and low atoms
        high_atom_coords = coordinates[high_atoms]
        low_atom_coords = coordinates[low_atoms]
        
        # Calculate bond vectors from high to low atoms
        bond_vectors = low_atom_coords - high_atom_coords
        
        # Compute norms of bond vectors
        bond_vector_norms = np.linalg.norm(bond_vectors, axis=1, keepdims=True)
        
        # Normalize bond vectors
        bond_unit_vectors = bond_vectors / bond_vector_norms
        
        # Get H-X bond lengths based on high layer atom elements
        h_bond_lengths = np.array([covalent_radii_lib(elements[idx]) + covalent_radii_lib("H") for idx in high_atoms]) * self.bohr2angstroms
        
        # Position H caps along bond vectors
        h_cap_coords = high_atom_coords + bond_unit_vectors * h_bond_lengths[:, np.newaxis]
        
        # Create hydrogen cap info tuples
        h_caps_info = []
        for i, (high_atom_idx, low_atom_idx) in enumerate(zip(high_atoms, low_atoms)):
            cap_idx = len(high_layer_elements) + i
            high_atom_new_idx = orig_to_high_map[high_atom_idx]
            h_caps_info.append((cap_idx, high_atom_new_idx, low_atom_idx))
        
        # Create list of hydrogen element symbols
        h_caps_elements = ['H'] * len(boundary_bonds)
        
        # Combine high layer atoms with hydrogen caps
        combined_coords = np.vstack([high_layer_coords, h_cap_coords])
        combined_elements = high_layer_elements + h_caps_elements
        
        return combined_coords, combined_elements, h_caps_info
    

    
    def write_xyz(self, filename, coordinates, elements, comment="ONIOM method"):
        """Write molecular structure to XYZ file."""
        with open(filename, 'w') as f:
            f.write(f"{len(elements)}\n")
            f.write(f"{comment}\n")
            for i, (element, coord) in enumerate(zip(elements, coordinates)):
                f.write(f"{element}    {coord[0]:17.12f}     {coord[1]:17.12f}     {coord[2]:17.12f}\n")
    
    def run_microiterations(self, high_layer_results=None):
        """
        Run ONIOM microiterations.
        
        Args:
            high_layer_results: Optional dictionary with pre-optimized high layer results
                containing 'coordinates', 'energy', and 'gradient' keys
        
        Returns:
            Dictionary with final energy, gradient, and coordinates including
            the full optimized molecular structure
        """
        if self.verbose:
            print("\nStarting ONIOM microiterations")
        
        # Initialize variables for convergence check
        prev_energy = 0.0
        prev_coordinates = None
        iteration = 0
        converged = False
        
        
        # Step 1: Use provided high layer results or wait for external optimization
        if high_layer_results is not None:
            # Use pre-optimized high layer results
            high_coords = high_layer_results.get('coordinates')
            high_energy = high_layer_results.get('energy', 0.0)
            high_gradient = high_layer_results.get('gradient', np.zeros_like(high_coords))
            
            if self.verbose:
                print("Using provided high layer results")
        else:
            # In a real implementation, you might wait for external calculation here
            # For now, we'll just use the current high layer coordinates
            high_coords = self.high_layer_coords
            high_energy = 0.0  # Placeholder
            high_gradient = np.zeros_like(high_coords)  # Placeholder
            
            if self.verbose:
                print("No high layer results provided, using current high layer coordinates")
        
        # Update high layer coordinates
        self.high_layer_coords = high_coords
            
        # Step 2: Update full system coordinates based on high layer optimization
        self._update_coordinates_from_high_layer(high_coords)
        
        # Step 3: Optimize low layer with fixed high layer
        low_energy, low_gradient, low_coords = self._optimize_low_layer()
        
        # Total ONIOM energy = E_high(high) + E_low(full) - E_low(high)
        total_energy = high_energy + low_energy
        
        # Store results for this iteration
        self.iteration_energies.append(total_energy)
        
        # Check convergence using our new function
        converged, metrics = self._check_convergence(
            low_gradient, self.coordinates, prev_coordinates
        )
        
        # Store maximum gradient for consistency with previous code
        self.iteration_gradients.append(metrics["max_gradient"])
        self.iteration_geometries.append(np.copy(self.coordinates))
        
        
        
        # Update for next iteration
        prev_energy = total_energy
        prev_coordinates = np.copy(self.coordinates)
        
     
       
        # Write convergence data
        self._write_convergence_data()
        
        return {
            "energy": prev_energy,
            "full_system_coordinates": np.copy(self.coordinates),  # Full optimized structure
            "high_layer_coords": np.copy(self.high_layer_coords),
            "full_system_elements": self.elements,
            "high_layer_elements": self.high_layer_elements,
            "converged": converged,
            "iterations": iteration,
            "convergence_metrics": metrics
        }
    
    def update_high_layer_coords(self, high_layer_coords):
        """
        Update the high layer coordinates with externally optimized coordinates.
        
        Args:
            high_layer_coords: Optimized high layer coordinates (with H caps)
        
        Returns:
            Updated full system coordinates
        """
        self.high_layer_coords = high_layer_coords
        self._update_coordinates_from_high_layer(high_layer_coords)
        return np.copy(self.coordinates)
    
    def _update_coordinates_from_high_layer(self, high_coords):
        """
        Update full system coordinates with optimized high layer coordinates.
        
        Args:
            high_coords: Optimized high layer coordinates (with H caps)
        """
        # Update high layer atoms in full system
        for idx, orig_idx in enumerate(self.high_layer_indices):
            self.coordinates[orig_idx] = high_coords[idx]
    
    def _optimize_low_layer(self):
        """
        Optimize low layer with fixed high layer at low level.
        
        Returns:
            Tuple of (energy, gradient, optimized_coordinates)
        """
        if self.verbose:
            print("Optimizing low layer with fixed high layer...")
        
        # Convert element symbols to atomic numbers for tblite
        element_numbers = np.array([element_number(elem) for elem in self.elements])
        
        # Set up FIRE optimizer for low layer
        optimizer = FIRE()
        optimizer.Initialization = True
        
        # Initial energy and gradient
        energy, gradient, _, _ = self.low_calc.single_point(
            file_directory=os.path.join(self.output_dir, "full_system"),
            element_number_list=element_numbers,
            iter=0,  # Initial iteration
            electric_charge_and_multiplicity=self.charge_and_multiplicity,
            method=self.low_layer_method,
            geom_num_list=self.coordinates
        )
        
        # Create a mask for atoms that can be optimized (low layer only)
        free_atoms = np.ones(len(self.elements), dtype=bool)
        for idx in self.high_layer_indices:
            free_atoms[idx] = False
        
        
        for iteration in range(self.max_iterations):
            # Create a modified gradient where high layer atoms have zero gradient
            modified_gradient = np.copy(gradient)
            for idx in self.high_layer_indices:
                modified_gradient[idx] = np.zeros(3)
            
            # Compute optimization step using FIRE
            move_vector = optimizer.run(
                geom_num_list=self.coordinates.reshape(-1),
                B_g=modified_gradient.reshape(-1),
                B_e=energy
            )
            
            # Apply move vector (only to low layer atoms)
            move_vector = move_vector.reshape(-1, 3)
            move_vector[~free_atoms] = 0  # Zero movement for high layer atoms
            new_coords = self.coordinates + move_vector
            
            # Calculate new energy and gradient
            new_energy, new_gradient, _, finish_flag = self.low_calc.single_point(
                file_directory=os.path.join(self.output_dir, "full_system"),
                element_number_list=element_numbers,
                iter=iteration + 1,
                electric_charge_and_multiplicity=self.charge_and_multiplicity,
                method=self.low_layer_method,
                geom_num_list=new_coords
            )
            
            # Check for convergence (only consider low layer gradients)
            low_layer_gradient = new_gradient[~np.array([idx in self.high_layer_indices for idx in range(len(self.elements))])]
            gradient_norm = np.linalg.norm(low_layer_gradient)
            energy_change = abs(new_energy - energy)
            
            if self.verbose and iteration % 10 == 0:
                print(f"  Low layer opt step {iteration}: E={new_energy:.8f}, |grad|={gradient_norm:.6f}")
            
            # Update for next iteration
            energy = new_energy
            gradient = new_gradient
            self.coordinates = new_coords
            
            # Check convergence
            converged, metrics = self._check_convergence(gradient, self.coordinates, prev_coords=self.coordinates)
            # Check if optimization cannot continue
            if converged:
                if self.verbose:
                    print("  Low layer optimization converged")
                break
            
        
        return energy, gradient, self.coordinates
    
    def _write_convergence_data(self):
        """Write convergence data to file."""
        with open(os.path.join(self.output_dir, "convergence.dat"), 'w') as f:
            f.write("# Iteration  Energy(Hartree)  MaxGradient(Hartree/Bohr)\n")
            for i, (e, g) in enumerate(zip(self.iteration_energies, self.iteration_gradients)):
                f.write(f"{i+1}  {e:.12f}  {g:.12f}\n")


