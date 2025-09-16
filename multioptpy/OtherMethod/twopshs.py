import copy
import datetime
import os
import numpy as np

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.fileio import make_workspace, xyz2list
from multioptpy.Parameters.parameter import UnitValueLib, element_number


class twoPSHSlikeMethod:
    def __init__(self, config):
        """
        Implementation of 2PSHS method based on SHS4py approach
        # ref. : Journal of chemical theory and computation 16.6 (2020): 3869-3878. (https://pubs.acs.org/doi/10.1021/acs.jctc.0c00010)
        # ref. : Chem. Phys. Lett. 2004, 384 (4–6), 277–282.
        # ref. : Chem. Phys. Lett. 2005, 404 (1–3), 95–99.
        # ref. : Chemical Physics Letters 404 (2005) 95–99.
        """
        self.config = config
        self.addf_config = {
            'step_number': int(config.addf_step_num),
            'number_of_add': int(config.nadd),
            'IOEsphereA_initial': 0.01,  # Initial hypersphere radius (will be overridden)
            'IOEsphereA_dist': float(config.addf_step_size),    # Decrement for hypersphere radius
            'IOEthreshold': 0.01,        # Threshold for IOE
            'minimize_threshold': 1.0e-5,# Threshold for minimization
        }
        self.energy_list_1 = []
        self.energy_list_2 = []
        self.gradient_list_1 = []
        self.gradient_list_2 = []
        self.init_displacement = 0.03 / self.get_unit_conversion()  # Bohr
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.converge_criteria = 0.12  # Convergence criteria in Angstroms
        self.element_number_list_1 = None
        self.element_number_list_2 = None
        self.ADDths = []  # List to store ADD theta classes
        self.optimized_structures = {}  # Dictionary to store optimized structures by ADD ID
        self.max_iterations = 1  # Limit to 1 major iteration per hypersphere radius
        self.max_inner_iterations = 20  # Maximum inner iterations for minimization
        self.sp1_structure = None  # Will store SP_1's structure
        self.sp1_energy = None  # Will store SP_1's energy
        self.sp2_structure = None  # Will store SP_2's structure
        self.sp2_energy = None  # Will store SP_2's energy
        self.initial_distance = None  # Will store distance between SP_1 and SP_2
        self.stalled_count = 0  # Counter for detecting stalled optimization

    def get_unit_conversion(self):
        """Return bohr to angstrom conversion factor"""
        return UnitValueLib().bohr2angstroms  # Approximate value for bohr2angstroms
        
    def adjust_center2origin(self, coord):
        """Adjust coordinates to have center at origin"""
        center = np.mean(coord, axis=0)
        return coord - center

    def SuperSphere_cartesian(self, A, thetalist, SQ, dim):
        """
        Vector of super sphere by cartesian (basis transformation from polar to cartesian)
        {sqrt(2*A), theta_1,..., theta_n-1} ->  {q_1,..., q_n} -> {x_1, x_2,..., x_n}
        """
        n_components = min(dim, SQ.shape[1] if SQ.ndim > 1 else dim)
        
        qlist = np.zeros(n_components)
        
        # Fill q-list using hyperspherical coordinates
        a_k = np.sqrt(2.0 * A)
        for i in range(min(len(thetalist), n_components-1)):
            qlist[i] = a_k * np.cos(thetalist[i])
            a_k *= np.sin(thetalist[i])
        
        # Handle the last component
        if n_components > 0:
            qlist[n_components-1] = a_k
        
        # Transform to original space
        SSvec = np.dot(SQ, qlist)
        
        return SSvec  # This is a vector in the reduced space
    
    def calctheta(self, vec, eigVlist, eigNlist):
        """
        Calculate the polar coordinates (theta) from a vector
        """
        # Get actual dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector
        n_components = min(len(eigVlist), len(eigNlist))  # Number of eigenvectors
        
        # Check vector dimensions
        if len(vec) != n_features:
            # If dimensions don't match, truncate or pad the vector
            if len(vec) > n_features:
                vec = vec[:n_features]  # Truncate
            else:
                padded_vec = np.zeros(n_features)
                padded_vec[:len(vec)] = vec
                vec = padded_vec
        
        # Create SQ_inv matrix with correct dimensions
        SQ_inv = np.zeros((n_components, n_features))
        
        for i in range(n_components):
            SQ_inv[i] = eigVlist[i] / np.sqrt(abs(eigNlist[i]))
        
        # Perform the dot product
        qvec = np.dot(SQ_inv, vec)
        
        r = np.linalg.norm(qvec)
        if r < 1e-10:
            return np.zeros(n_components - 1)
            
        thetalist = []
        for i in range(len(qvec) - 1):
            # Handle possible numerical issues with normalization
            norm_q = np.linalg.norm(qvec[i:])
            if norm_q < 1e-10:
                theta = 0.0
            else:
                cos_theta = qvec[i] / norm_q
                cos_theta = max(-1.0, min(1.0, cos_theta))  # Ensure within bounds
                theta = np.arccos(cos_theta)
                if i == len(qvec) - 2 and qvec[-1] < 0:
                    theta = 2*np.pi - theta
            thetalist.append(theta)
        
        return np.array(thetalist)
    
    def SQaxes(self, eigNlist, eigVlist, dim):
        """Calculate the SQ matrix for transformation"""
        # Get actual available dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector
        n_components = min(len(eigVlist), len(eigNlist), dim)  # Number of eigenvectors to use
        
        # Initialize with correct dimensions
        SQ = np.zeros((n_features, n_components))
        
        # Only iterate up to the available components
        for i in range(n_components):
            SQ[:, i] = eigVlist[i] * np.sqrt(abs(eigNlist[i]))
        
        return SQ

    def SQaxes_inv(self, eigNlist, eigVlist, dim):
        """Calculate the inverse SQ matrix for transformation"""
        # Get actual available dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector 
        n_components = min(len(eigVlist), len(eigNlist), dim)  # Number of eigenvectors to use
        
        # Initialize with correct dimensions
        SQ_inv = np.zeros((n_components, n_features))
        
        # Only iterate up to the available components
        for i in range(n_components):
            SQ_inv[i] = eigVlist[i] / np.sqrt(abs(eigNlist[i]))
        
        return SQ_inv
    
    def angle(self, v1, v2):
        """Calculate angle between two vectors"""
        # Check for zero vectors or invalid inputs
        if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
            return 0.0
            
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        
        # Handle potential numerical issues
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        
        return np.arccos(dot_product)
    
    def angle_SHS(self, v1, v2, SQ_inv):
        """Calculate angle between two vectors in SHS space"""
        # Ensure vectors are flattened
        v1_flat = v1.flatten() if hasattr(v1, 'flatten') else v1
        v2_flat = v2.flatten() if hasattr(v2, 'flatten') else v2
        
        # Handle potential dimension mismatches
        min_dim = min(len(v1_flat), len(v2_flat), SQ_inv.shape[1])
        v1_flat = v1_flat[:min_dim]
        v2_flat = v2_flat[:min_dim]
        
        q_v1 = np.dot(SQ_inv[:, :min_dim], v1_flat)
        q_v2 = np.dot(SQ_inv[:, :min_dim], v2_flat)
        return self.angle(q_v1, q_v2)
    
    def calc_cartesian_distance(self, structure1, structure2):
        """Calculate the RMSD between two molecular structures in Cartesian coordinates"""
        if structure1.shape != structure2.shape:
            raise ValueError("Structures have different shapes")
        
        # Calculate squared differences
        squared_diff = np.sum((structure1 - structure2)**2)
        
        # Return RMSD
        return np.sqrt(squared_diff / structure1.shape[0])
    
    def grad_hypersphere(self, f, grad, eqpoint, IOEsphereA, thetalist):
        """Calculate gradient on hypersphere - MODIFIED for 2PSHS to minimize ADD"""
        # Generate nADD in the reduced space (this will be a vector in the Hessian eigenspace)
        nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
        
        # We need to convert the reduced space vector back to the full coordinate space
        # First create a zero vector in the full space
        n_atoms = eqpoint.shape[0]
        n_coords = n_atoms * 3
        nADD_full = np.zeros(n_coords)
        
        # Map the reduced vector to the full space (this is approximate)
        for i in range(min(len(nADD_reduced), n_coords)):
            nADD_full[i] = nADD_reduced[i]
            
        # Reshape to molecular geometry format
        nADD = nADD_full.reshape(n_atoms, 3)
        
        # Calculate the normalized direction vector for projecting out later
        EnADD = nADD_full / np.linalg.norm(nADD_full)
        
        # Apply the displacement to the initial geometry
        target_point = eqpoint + nADD
        target_point = self.periodicpoint(target_point)
        
        # Calculate gradient at this point
        grad_x = grad(target_point)
        if isinstance(grad_x, bool) and grad_x is False:
            return False, False
        
        # Flatten the gradient for vector operations
        grad_x_flat = grad_x.flatten()
        
        # Project gradient onto tangent space of hypersphere
        # We remove the component along the displacement vector
        returngrad_flat = grad_x_flat - np.dot(grad_x_flat, EnADD) * EnADD
        
        # 2PSHS modification: Calculate distance to SP_1 structure
        distance_to_sp1 = self.calc_cartesian_distance(target_point, self.sp1_structure)
        
        # Create a gradient component that points toward SP_1's structure
        # This is the key modification for 2PSHS - we want to minimize ADD
        direction_to_sp1 = self.sp1_structure - target_point
        direction_to_sp1_flat = direction_to_sp1.flatten()
        
        # Normalize the direction vector
        direction_norm = np.linalg.norm(direction_to_sp1_flat)
        if direction_norm > 1e-10:
            direction_to_sp1_flat /= direction_norm
            
            # Project this direction to be tangent to the hypersphere
            direction_component_along_nADD = np.dot(direction_to_sp1_flat, EnADD)
            direction_on_tangent = direction_to_sp1_flat - direction_component_along_nADD * EnADD
            direction_on_tangent_norm = np.linalg.norm(direction_on_tangent)
            
            if direction_on_tangent_norm > 1e-10:
                direction_on_tangent /= direction_on_tangent_norm
                
                # Add this component to our gradient
                # Weight decreases as we get closer to SP_1
                weight = min(1.0, distance_to_sp1 / self.converge_criteria)
                returngrad_flat += direction_on_tangent * weight * np.linalg.norm(returngrad_flat)
        
        # Reshape gradient back to molecular geometry format for easier handling
        returngrad = returngrad_flat.reshape(n_atoms, 3)
        
        return target_point, returngrad

    def periodicpoint(self, point):
        """Apply periodic boundary conditions if needed"""
        # Implement according to your specific requirements
        return point

    def minimizeTh_SD_SS(self, ADDth, initialpoint, f, grad, eqpoint, IOEsphereA):
        """
        Steepest descent optimization on hypersphere to minimize ADD
        For 2PSHS, we want to find minimum ADD on each hypersphere before reducing radius
        """
        whileN = 0
        thetalist = ADDth.thetalist + initialpoint
        stepsize = 0.1
        n_atoms = eqpoint.shape[0]
        n_coords = n_atoms * 3
        
        # Generate initial nADD
        nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
        
        # Convert reduced space vector to full coordinate space with proper dimensions
        nADD_full = np.zeros(n_coords)
        for i in range(min(len(nADD_reduced), n_coords)):
            nADD_full[i] = nADD_reduced[i]
        
        # Reshape to match eqpoint dimensions
        nADD = nADD_full.reshape(n_atoms, 3)
        
        # Keep track of best solution (for minimizing ADD)
        best_thetalist = thetalist.copy()
        best_add_value = float('inf')  # We want to minimize ADD, so start with positive infinity
        
        # Initial point
        targetpoint = eqpoint + nADD
        targetpoint = self.periodicpoint(targetpoint)
        
        # Try to calculate initial energy and ADD
        try:
            initial_energy = f(targetpoint)
            if isinstance(initial_energy, (int, float)) and not np.isnan(initial_energy):
                # Calculate initial ADD
                initial_add = initial_energy - IOEsphereA - self.sp1_energy
                best_add_value = initial_add
        except Exception:
            pass  # Continue even if initial energy calculation fails
        
        # Variables to detect convergence
        prev_add_values = []
        no_improvement_count = 0
        
        # Main optimization loop
        while whileN < self.max_inner_iterations:
            try:
                # Get gradient at current point
                grad_x = grad(targetpoint)
                
                # If gradient calculation fails, continue with smaller step or different approach
                if grad_x is False:
                    # Try a random perturbation and continue
                    print(f"Gradient calculation failed at iteration {whileN}, trying random perturbation")
                    random_perturbation = np.random.rand(n_atoms, 3) * 0.01 - 0.005  # Small random perturbation
                    targetpoint = targetpoint + random_perturbation
                    targetpoint = self.periodicpoint(targetpoint)
                    
                    # Calculate new nADD
                    nADD = targetpoint - eqpoint
                    thetalist = self.calctheta(nADD.flatten(), self.eigVlist, self.eigNlist)
                    
                    # Ensure we're on the hypersphere with correct radius
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    
                    targetpoint = eqpoint + nADD
                    targetpoint = self.periodicpoint(targetpoint)
                    
                    whileN += 1
                    continue
                
                # Calculate nADD norm
                nADD_norm = np.linalg.norm(nADD.flatten())
                if nADD_norm < 1e-10:
                    # If nADD is too small, generate a new one
                    print(f"nADD norm too small at iteration {whileN}, regenerating")
                    thetalist = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    targetpoint = eqpoint + nADD
                    targetpoint = self.periodicpoint(targetpoint)
                    whileN += 1
                    continue
                    
                # Project gradient onto tangent space of hypersphere
                nADD_unit = nADD.flatten() / nADD_norm
                grad_along_nADD = np.dot(grad_x.flatten(), nADD_unit)
                SSgrad_flat = grad_x.flatten() - grad_along_nADD * nADD_unit
                
                # For minimizing ADD, we follow the negative gradient
                # This is the key aspect - when minimizing, we move opposite to gradient direction
                SSgrad = -1.0 * SSgrad_flat.reshape(n_atoms, 3)
                
                # Calculate energy and current ADD
                current_energy = f(targetpoint)
                current_add = current_energy - IOEsphereA - self.sp1_energy
                
                # Store current ADD value for convergence detection
                prev_add_values.append(current_add)
                if len(prev_add_values) > 3:
                    prev_add_values.pop(0)
                
                # Check if gradient is small enough (local minimum on the hypersphere)
                if np.linalg.norm(SSgrad) < 1.0e-3:
                    print(f"Small gradient: {np.linalg.norm(SSgrad):.6f}, potential minimum ADD found: {current_add:.6f}")
                    # If gradient is small, we may have found a local minimum
                    if current_add < best_add_value:
                        best_add_value = current_add
                        best_thetalist = thetalist.copy()
                        ADDth.converged = True
                        print(f"New best ADD value: {best_add_value:.6f}")
                        return thetalist
                
                # Store current point
                _point_initial = copy.copy(targetpoint)
                
                # Line search
                stepsizedamp = stepsize
                found_valid_step = False
                
                # Try multiple step sizes
                for whileN2 in range(1, 5):  # Try up to 4 steps with varying sizes
                    try:
                        # Take step with dynamic step size
                        step_scale = whileN2 if whileN2 <= 2 else (whileN2 - 2) * 0.1
                        
                        # Follow the negative gradient to minimize ADD
                        targetpoint = _point_initial + step_scale * SSgrad / np.linalg.norm(SSgrad) * stepsizedamp
                        
                        # Calculate new nADD
                        nADD2 = targetpoint - eqpoint
                        
                        # Convert to theta parameters
                        thetalist_new = self.calctheta(nADD2.flatten(), self.eigVlist, self.eigNlist)
                        
                        # Ensure we're on the hypersphere with correct radius
                        nADD2_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist_new, self.SQ, self.dim)
                        
                        # Convert reduced space vector to full coordinate space
                        nADD2_full = np.zeros(n_coords)
                        for i in range(min(len(nADD2_reduced), n_coords)):
                            nADD2_full[i] = nADD2_reduced[i]
                        
                        # Reshape to match eqpoint dimensions
                        nADD2 = nADD2_full.reshape(n_atoms, 3)
                        
                        # Calculate new point on hypersphere
                        new_point = eqpoint + nADD2
                        new_point = self.periodicpoint(new_point)
                        
                        # Calculate energy and ADD at new point
                        new_energy = f(new_point)
                        new_add = new_energy - IOEsphereA - self.sp1_energy
                        
                        # Accept step if it decreases ADD (since we're minimizing)
                        if new_add < current_add:
                            found_valid_step = True
                            targetpoint = new_point
                            thetalist = thetalist_new
                            nADD = nADD2
                            
                            # Update best solution if better
                            if new_add < best_add_value:
                                best_add_value = new_add
                                best_thetalist = thetalist_new.copy()
                            
                            break
                        
                    except Exception as e:
                        print(f"Step calculation error: {e}, trying different step")
                        continue
                        
                # If no valid step found, try random steps to escape local minima
                if not found_valid_step:
                    # Check if we've been stuck for too long
                    if len(prev_add_values) >= 3 and all(abs(prev_add_values[i] - prev_add_values[i-1]) < 1e-4 for i in range(1, len(prev_add_values))):
                        no_improvement_count += 1
                        if no_improvement_count > 3:
                            print(f"No improvement in ADD for several iterations. Current ADD: {current_add:.6f}")
                            # If we're stuck, return the best solution found so far
                            if current_add < best_add_value:
                                best_add_value = current_add
                                best_thetalist = thetalist.copy()
                            ADDth.converged = True
                            return best_thetalist
                    
                    # Try a random step
                    random_theta_offset = np.random.uniform(-0.1, 0.1, len(thetalist))
                    thetalist_new = thetalist + random_theta_offset
                    
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist_new, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    
                    targetpoint = eqpoint + nADD
                    targetpoint = self.periodicpoint(targetpoint)
                    thetalist = thetalist_new
                    
                    # Check ADD at new random point
                    try:
                        new_energy = f(targetpoint)
                        new_add = new_energy - IOEsphereA - self.sp1_energy
                        
                        if new_add < best_add_value:
                            best_add_value = new_add
                            best_thetalist = thetalist.copy()
                    except Exception:
                        pass
                
                # Print progress periodically
                if whileN % 5 == 0:
                    print(f"Iteration {whileN}: Current ADD = {current_add:.6f}, Best ADD = {best_add_value:.6f}")
                
                whileN += 1
                
            except Exception as e:
                print(f"Error in optimization step {whileN}: {e}")
                whileN += 1
                # Try a random step to recover
                random_theta_offset = np.random.uniform(-0.1, 0.1, len(thetalist))
                thetalist = thetalist + random_theta_offset
                
                nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                nADD_full = np.zeros(n_coords)
                for i in range(min(len(nADD_reduced), n_coords)):
                    nADD_full[i] = nADD_reduced[i]
                nADD = nADD_full.reshape(n_atoms, 3)
                
                targetpoint = eqpoint + nADD
                targetpoint = self.periodicpoint(targetpoint)
        
        print(f"Max iterations ({self.max_inner_iterations}) reached, returning best solution with ADD = {best_add_value:.6f}")
        # Return the best solution found based on minimizing ADD
        return best_thetalist
    
    def save_optimized_structure(self, ADDth, iteration_num, IOEsphereA):
        """Save optimized structure for a specific ADD and iteration"""
        # Create directory path for ADD-specific structures
        add_dir = os.path.join(self.directory, "optimized_structures", f"ADD_{ADDth.IDnum}")
        os.makedirs(add_dir, exist_ok=True)
        
        # Create filename with radius and iteration information
        radius = np.sqrt(IOEsphereA)
        filename = f"iteration_{iteration_num}_r_{radius:.4f}.xyz"
        filepath = os.path.join(add_dir, filename)
        
        # Calculate distance to SP_1
        distance = self.calc_cartesian_distance(ADDth.x, self.sp1_structure)
        
        # Write XYZ file
        with open(filepath, 'w') as f:
            f.write(f"{len(self.element_list_1)}\n")
            f.write(f"ADD_{ADDth.IDnum} Iteration {iteration_num} Radius {radius:.4f} Distance_to_SP1 {distance:.6f}\n")
            for i, (element, coord) in enumerate(zip(self.element_list_1, ADDth.x)):
                f.write(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")

        # Store structure information in dictionary
        if ADDth.IDnum not in self.optimized_structures:
            self.optimized_structures[ADDth.IDnum] = []
            
        self.optimized_structures[ADDth.IDnum].append({
            'iteration': iteration_num,
            'radius': radius,
            'distance': distance,
            'file': filepath,
            'coords': ADDth.x.copy(),
            'comment': f"ADD_{ADDth.IDnum} Iteration {iteration_num} Radius {radius:.4f} Distance_to_SP1 {distance:.6f}"
        })
    
    def create_separate_xyz_files(self):
        """Create separate XYZ files for each ADD path"""
        created_files = []
        
        for add_id, structures in self.optimized_structures.items():
            # Sort structures by iteration number
            structures.sort(key=lambda x: x['iteration'])
            
            # Path for the ADD-specific file
            add_trajectory_file = os.path.join(self.directory, f"ADD_{add_id}_trajectory.xyz")
            
            # Write the trajectory file
            with open(add_trajectory_file, 'w') as f:
                for structure in structures:
                    f.write(f"{len(self.element_list_1)}\n")
                    f.write(f"{structure['comment']}\n")
                    for i, (element, coord) in enumerate(zip(self.element_list_1, structure['coords'])):
                        f.write(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
            
            created_files.append(add_trajectory_file)
            
        if created_files:
            paths_str = "\n".join(created_files)
            print(f"Created {len(created_files)} ADD trajectory files:\n{paths_str}")
    
    def create_distance_plots(self):
        """Create CSV files with distance data for plotting"""
        for add_id, structures in self.optimized_structures.items():
            # Sort structures by iteration number
            structures.sort(key=lambda x: x['iteration'])
            
            # Path for the distance data file
            distance_file = os.path.join(self.directory, f"ADD_{add_id}_distances.csv")
            
            # Write the distance data
            with open(distance_file, 'w') as f:
                f.write("iteration,radius,distance_to_sp1\n")
                for structure in structures:
                    f.write(f"{structure['iteration']},{structure['radius']:.4f},{structure['distance']:.6f}\n")
            
            print(f"Created distance data file: {distance_file}")

    def detect_add(self, SP_1, SP_2):
        """
        Calculate coordinate axes from SP_1's Hessian for creating the hypersphere
        Use the direction from SP_1 to SP_2 as the primary direction
        """
        # Get coordinates for SP_1
        coord_1 = self.coords_1
        coord_1 = self.adjust_center2origin(coord_1)
        n_atoms = coord_1.shape[0]
        n_coords = n_atoms * 3
        
        # Get coordinates for SP_2
        coord_2 = self.coords_2
        coord_2 = self.adjust_center2origin(coord_2)
        
        # Check that both structures have the same number of atoms
        if coord_1.shape != coord_2.shape:
            print("SP_1 and SP_2 structures have different shapes")
            return False
        
        element_number_list_1 = self.get_element_number_list_1()
        print("### Calculating SP_1 structure and Hessian to create coordinate system ###")
        
        SP_1.hessian_flag = True
        self.init_energy_1, self.init_gradient_1, _, iscalculationfailed = SP_1.single_point(
            None, element_number_list_1, "", self.electric_charge_and_multiplicity_1, 
            self.method, coord_1
        )
        
        if iscalculationfailed:
            print("Initial calculation with SP_1 failed.")
            return False
        
        # Apply bias potential if needed for SP_1
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, bias_energy_1, bias_gradient_1, bias_hess_1 = BPC.main(
            self.init_energy_1, self.init_gradient_1, coord_1, element_number_list_1, 
            self.config.force_data
        )
        self.init_energy_1 = bias_energy_1
        self.init_gradient_1 = bias_gradient_1
        
        SP_1.hessian_flag = False
        self.init_geometry = coord_1  # Shape: (n_atoms, 3)
        
        # Store SP_1 structure and energy
        self.sp1_structure = copy.deepcopy(coord_1)
        self.sp1_energy = self.init_energy_1
        
        print(f"SP_1 energy: {self.sp1_energy:.6f}")
        print("### Calculating Hessian matrix to set up coordinate system ###")
        
        hessian = SP_1.Model_hess + bias_hess_1
        
        # Project out translation and rotation
        projection_hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(hessian, self.element_list_1, coord_1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(projection_hessian)
        eigenvalues = eigenvalues.astype(np.float64)
        
        # Filter out near-zero eigenvalues
        nonzero_indices = np.where(np.abs(eigenvalues) > 1e-10)[0]
        nonzero_eigenvectors = eigenvectors[:, nonzero_indices].astype(np.float64)
        nonzero_eigenvalues = eigenvalues[nonzero_indices].astype(np.float64)
        
        sorted_idx = np.argsort(nonzero_eigenvalues)
        
        self.init_eigenvalues = nonzero_eigenvalues
        self.init_eigenvectors = nonzero_eigenvectors
        self.dim = len(nonzero_eigenvalues)
        self.n_atoms = n_atoms
        self.n_coords = n_coords
        
        # Store flattened versions for matrix operations
        self.init_geometry_flat = self.init_geometry.flatten()
        
        # Prepare SQ matrices
        self.SQ = self.SQaxes(nonzero_eigenvalues, nonzero_eigenvectors, self.dim)
        self.SQ_inv = self.SQaxes_inv(nonzero_eigenvalues, nonzero_eigenvectors, self.dim)
        
        self.eigNlist = nonzero_eigenvalues
        self.eigVlist = nonzero_eigenvectors
        
        # Calculate mode eigenvectors to try - focus on lower eigenvectors first
        search_idx = len(sorted_idx) # Start with all eigenvectors
        self.sorted_eigenvalues_idx = sorted_idx[0:search_idx]
        
        # Initialize ADDths with initial directions
        self.ADDths = []
        
        print("### Getting SP_2 structure and energy ###")
        # Calculate SP_2's energy at its initial structure
        element_number_list_2 = self.get_element_number_list_2()
        sp2_energy, sp2_gradient, _, iscalculationfailed = SP_2.single_point(
            None, element_number_list_2, "", self.electric_charge_and_multiplicity_2, 
            self.method, coord_2
        )
        
        if iscalculationfailed:
            print("Initial calculation with SP_2 failed.")
            return False
        
        # Apply bias potential if needed for SP_2
        _, bias_energy_2, bias_gradient_2, _ = BPC.main(
            sp2_energy, sp2_gradient, coord_2, element_number_list_2, 
            self.config.force_data
        )
        
        self.sp2_structure = copy.deepcopy(coord_2)
        self.sp2_energy = bias_energy_2
        
        print(f"SP_2 energy: {self.sp2_energy:.6f}")
        
        # Calculate vector from SP_1 to SP_2
        direction_vector = self.sp1_structure - self.sp2_structure
        direction_vector_flat = direction_vector.flatten()
        
        # Calculate distance between SP_1 and SP_2
        direction_norm = np.linalg.norm(direction_vector_flat)
        if direction_norm < 1e-10:
            print("SP_1 and SP_2 structures are too similar. Cannot establish a meaningful direction.")
            return False
        
        # Set the initial distance
        self.initial_distance = direction_norm
            
        # Normalize the direction vector
        direction_vector_flat = direction_vector_flat / direction_norm
        
        print("### Setting up initial point on the hypersphere ###")
        print(f"Distance between SP_1 and SP_2: {direction_norm:.6f} Å")
        
        with open(self.directory + "/direction_info.csv", "w") as f:
            f.write("direction,distance_between_sp1_sp2\n")
            f.write(f"SP_1_to_SP_2,{direction_norm:.6f}\n")
        
        # Use the distance between SP_1 and SP_2 as the initial sphere radius
        IOEsphereA = direction_norm ** 2  # Convert to A value (squared radius)
        
        # Create a single ADD point based on the SP_1 to SP_2 direction
        ADDth = type('ADDthetaClass', (), {})
        ADDth.IDnum = 0
        ADDth.dim = self.dim
        ADDth.SQ = self.SQ
        ADDth.SQ_inv = self.SQ_inv
        
        # Calculate theta parameters from the direction vector
        ADDth.thetalist = self.calctheta(direction_vector_flat, nonzero_eigenvectors, nonzero_eigenvalues)
        
        # Generate nADD (this will be a flattened vector in the Hessian eigenspace)
        ADDth.nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, self.SQ, self.dim)
        
        # Convert the reduced space vector back to the full coordinate space
        ADDth.nADD_full = np.zeros(n_coords)
        for i in range(min(len(ADDth.nADD_reduced), n_coords)):
            ADDth.nADD_full[i] = ADDth.nADD_reduced[i]
            
        # Reshape to molecular geometry format
        ADDth.nADD = ADDth.nADD_full.reshape(n_atoms, 3)
        
        # Apply the displacement to the initial geometry
        ADDth.x = self.init_geometry + ADDth.nADD
        ADDth.x = self.periodicpoint(ADDth.x)
        
        # Initialize flags
        ADDth.converged = False
        ADDth.ADDoptQ = True
        ADDth.ADDremoveQ = False
        ADDth.last_distance = float('inf')  # For tracking progress
        
        # Store the reference direction
        ADDth.direction_vector = direction_vector
        
        # Add to ADDths list
        self.ADDths = [ADDth]
        
        print(f"### Primary direction established with initial radius {np.sqrt(IOEsphereA):.6f} ###")
        
        # Initialize the optimized structures dictionary
        self.optimized_structures = {}
        
        # Create directory for optimized structures
        os.makedirs(os.path.join(self.directory, "optimized_structures"), exist_ok=True)
        
        print("### Coordinate system setup complete ###")
        return True

    def optimize_with_sp2(self, ADDths, SP_2, eqpoint, IOEsphereA, sphereN):
        """
        Optimize points on the hypersphere using SP_2
        Minimize ADD on the hypersphere
        """
        print(f"Starting optimization on sphere {sphereN} with radius {np.sqrt(IOEsphereA):.4f}")
        
        # Reset optimization flags
        for ADDth in ADDths:
            if not ADDth.converged and not ADDth.ADDremoveQ:
                ADDth.ADDoptQ = True
            else:
                ADDth.ADDoptQ = False
                
        # Create a directory for intermediate optimization steps
        sphere_dir = os.path.join(self.directory, "optimized_structures", f"sphere_{sphereN}")
        os.makedirs(sphere_dir, exist_ok=True)
        
        # One iteration per hypersphere
        iteration_num = 1
        print(f"Iteration {iteration_num}")
        
        # Process each ADD point
        for ADDth in ADDths:
            if not ADDth.ADDoptQ or ADDth.ADDremoveQ:
                continue
                
            # Optimize this ADD point
            self.current_id = ADDth.IDnum
            
            # Starting from zero displacement
            x_initial = np.zeros(len(ADDth.thetalist))
            
            # Minimize ADD on hypersphere using our modified steepest descent
            thetalist = self.minimizeTh_SD_SS(
                ADDth, x_initial, 
                lambda x: SP_2.single_point(None, self.get_element_number_list_2(), "", 
                                        self.electric_charge_and_multiplicity_2, self.method, x)[0],
                lambda x: self.calculate_gradient(SP_2, x, self.element_number_list_2, 
                                                self.electric_charge_and_multiplicity_2),
                eqpoint, IOEsphereA
            )
            
            if thetalist is False:
                ADDth.ADDremoveQ = True
                print(f"ADD {ADDth.IDnum} optimization failed, marking for removal")
                continue
            
            # Update ADD point with optimized position
            ADDth.thetalist = thetalist
            
            # Generate nADD in the reduced space
            n_atoms = eqpoint.shape[0]
            ADDth.nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, self.SQ, self.dim)
            
            # Map to full coordinate space
            ADDth.nADD_full = np.zeros(n_atoms * 3)
            for i in range(min(len(ADDth.nADD_reduced), n_atoms * 3)):
                ADDth.nADD_full[i] = ADDth.nADD_reduced[i]
            
            # Reshape to molecular geometry
            ADDth.nADD = ADDth.nADD_full.reshape(n_atoms, 3)
            
            # Calculate new coordinates
            ADDth.x = eqpoint + ADDth.nADD
            ADDth.x = self.periodicpoint(ADDth.x)
            
            # Calculate new energy with SP_2
            energy, grad, _, iscalculationfailed = SP_2.single_point(
                None, self.get_element_number_list_2(), "", 
                self.electric_charge_and_multiplicity_2, self.method, ADDth.x
            )
            
            if iscalculationfailed:
                ADDth.ADDremoveQ = True
                continue
                
            BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
            _, bias_energy, bias_grad, _ = BPC.main(
                energy, grad, ADDth.x, self.get_element_number_list_2(), 
                self.config.force_data
            )
            
            ADDth.A = bias_energy
            ADDth.ADD = ADDth.A - IOEsphereA - self.sp1_energy  # Use SP_1's energy as reference
            
            # Calculate distance to SP_1 structure (for information only)
            distance_to_sp1 = self.calc_cartesian_distance(ADDth.x, self.sp1_structure)
            ADDth.distance_to_sp1 = distance_to_sp1
            
            # Print results
            print(f"ADD {ADDth.IDnum}: Energy={energy:.6f}, ADD={ADDth.ADD:.6f}, ||Grad||={np.linalg.norm(grad):.6f}, Distance to SP_1={distance_to_sp1:.6f} Å")
            
            # Save structure
            self.save_optimized_structure(ADDth, iteration_num, IOEsphereA)
        
        # Filter ADDths list
        ADDths = [ADDth for ADDth in ADDths if not ADDth.ADDremoveQ]
        
        # Create a summary of final ADD values
        print("\n### Final ADD values for this hypersphere ###")
        min_add = float('inf')
        min_add_id = -1
        
        for ADDth in ADDths:
            if hasattr(ADDth, 'ADD'):
                print(f"ADD {ADDth.IDnum}: {ADDth.ADD:.6f}")
                if ADDth.ADD < min_add:
                    min_add = ADDth.ADD
                    min_add_id = ADDth.IDnum
        
        if min_add_id >= 0:
            print(f"\nMinimum ADD on this hypersphere: {min_add:.6f} (ADD {min_add_id})")
        
        return ADDths
    
    def calculate_gradient(self, SP, x, element_number_list, electric_charge_and_multiplicity):
        
        """Calculate gradient at point x"""
        _, grad_x, _, iscalculationfailed = SP.single_point(
            None, element_number_list, "", electric_charge_and_multiplicity, 
            self.method, x
        )
        
        if iscalculationfailed:
            return False
            
        # Apply bias if needed
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, _, bias_gradient, _ = BPC.main(
            0, grad_x, x, element_number_list, 
            self.config.force_data
        )
        
        return bias_gradient
    def run(self, file_directory, SP_1, SP_2, electric_charge_and_multiplicity, FIO_img_1, FIO_img_2):
        """
        Main function to run the 2PSHS method.
        SP_1 is used to create the hypersphere.
        SP_2 moves on the hypersphere to minimize ADD.
        The hypersphere radius starts at the distance between SP_1 and SP_2 
        and decreases gradually until it reaches zero.
        """
        print("### Start Two-Point Scaled Hypersphere Search (2PSHS) method ###")

        # Preparation
        base_file_name_1 = os.path.splitext(FIO_img_1.START_FILE)[0]
        base_file_name_2 = os.path.splitext(FIO_img_2.START_FILE)[0]
        self.set_mole_info(base_file_name_1, base_file_name_2, electric_charge_and_multiplicity)

        self.directory = make_workspace(file_directory)
        
        # Create main directory for optimized structures
        os.makedirs(os.path.join(self.directory, "optimized_structures"), exist_ok=True)
        
        # Detect initial ADD directions using SP_1 and SP_2
        print("Step 1: Using SP_1 and SP_2 to detect coordinate system and SP_1-SP_2 direction")
        success = self.detect_add(SP_1, SP_2)
        if not success:
            print("Failed to establish SP_1-SP_2 direction.")
            return False
        
        # Start with radius equal to the distance between SP_1 and SP_2
        radius = self.initial_distance
        IOEsphereA = radius ** 2  # Convert to A value (squared radius)
        
        print("\nStep 2: Optimizing SP_2 to minimize ADD on hypersphere")
        
        # Main optimization loop - decrease the radius in each iteration
        sphere_num = 1
        best_add_value = float('inf')
        best_structure = None
        best_radius = None
        
        while IOEsphereA > 0 and sphere_num <= self.addf_config['step_number'] and len(self.ADDths) > 0:
            print(f"\nStep {sphere_num+1}: Using hypersphere with radius {np.sqrt(IOEsphereA):.4f}")
            
            # Reset optimization flags
            for ADDth in self.ADDths:
                ADDth.converged = False
                ADDth.ADDoptQ = True
            
            # Optimize on the current sphere to minimize ADD
            self.ADDths = self.optimize_with_sp2(
                self.ADDths, SP_2, self.init_geometry, IOEsphereA, sphere_num
            )
            
            # Check if we found a better ADD value on this hypersphere
            for ADDth in self.ADDths:
                if hasattr(ADDth, 'ADD') and ADDth.ADD < best_add_value:
                    best_add_value = ADDth.ADD
                    best_structure = ADDth.x.copy()
                    best_radius = np.sqrt(IOEsphereA)
                    print(f"New best ADD value: {best_add_value:.6f} at radius {best_radius:.4f}")
            
            # Decrease the radius for next iteration
            radius -= self.addf_config['IOEsphereA_dist']
            IOEsphereA = radius ** 2  # Convert to A value
            
            # Check if the radius has become zero or negative
            if radius <= 0:
                print("Radius has reached zero or negative value. Stopping the search.")
                break
                
            sphere_num += 1
        
        # Create separate trajectory files for each ADD path
        self.create_separate_xyz_files()
        
        # Create distance plots
        self.create_distance_plots()
        
        if best_structure is not None:
            print(f"\n### Success! Found minimum ADD value {best_add_value:.6f} at radius {best_radius:.4f} ###")
            # Save the best structure
            best_file = os.path.join(self.directory, "best_add_structure.xyz")
            with open(best_file, 'w') as f:
                f.write(f"{len(self.element_list_1)}\n")
                f.write(f"Best ADD structure with value {best_add_value:.6f} at radius {best_radius:.4f}\n")
                for i, (element, coord) in enumerate(zip(self.element_list_1, best_structure)):
                    f.write(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
            print(f"Best structure saved to: {best_file}")
        else:
            print("\n### Warning: Could not find a good ADD minimum. ###")
        
        return best_structure is not None
            
    # Getters and setters
    def set_molecule_1(self, element_list, coords):
        self.element_list_1 = element_list
        self.coords_1 = coords

    def set_molecule_2(self, element_list, coords):
        self.element_list_2 = element_list
        self.coords_2 = coords

    def set_gradient_1(self, gradient):
        self.gradient_1 = gradient

    def set_gradient_2(self, gradient):
        self.gradient_2 = gradient

    def set_hessian_1(self, hessian):
        self.hessian_1 = hessian

    def set_hessian_2(self, hessian):
        self.hessian_2 = hessian

    def set_energy_1(self, energy):
        self.energy_1 = energy
    
    def set_energy_2(self, energy):
        self.energy_2 = energy

    def set_coords_1(self, coords):
        self.coords_1 = coords

    def set_coords_2(self, coords):
        self.coords_2 = coords

    def set_element_list_1(self, element_list):
        self.element_list_1 = element_list
        self.element_number_list_1 = [element_number(i) for i in self.element_list_1]

    def set_element_list_2(self, element_list):
        self.element_list_2 = element_list
        self.element_number_list_2 = [element_number(i) for i in self.element_list_2]

    def get_coords_1(self):
        return self.coords_1
        
    def get_coords_2(self):
        return self.coords_2

    def get_element_list_1(self):
        return self.element_list_1
        
    def get_element_list_2(self):
        return self.element_list_2

    def get_element_number_list_1(self):
        if self.element_number_list_1 is None:
            if self.element_list_1 is None:
                raise ValueError('Element list 1 is not set.')
            self.element_number_list_1 = [element_number(i) for i in self.element_list_1]
        return self.element_number_list_1

    def get_element_number_list_2(self):
        if self.element_number_list_2 is None:
            if self.element_list_2 is None:
                raise ValueError('Element list 2 is not set.')
            self.element_number_list_2 = [element_number(i) for i in self.element_list_2]
        return self.element_number_list_2

    def set_mole_info(self, base_file_name_1, base_file_name_2, electric_charge_and_multiplicity):
        """Load molecular information for both SP_1 and SP_2 structures"""
        coord_1, element_list_1, electric_charge_and_multiplicity = xyz2list(
            base_file_name_1 + ".xyz", electric_charge_and_multiplicity)

        coord_2, element_list_2, _ = xyz2list(
            base_file_name_2 + ".xyz", electric_charge_and_multiplicity)

        if self.config.usextb != "None":
            self.method = self.config.usextb
        elif self.config.usedxtb != "None":
            self.method = self.config.usedxtb
        else:
            self.method = "None"

        self.coords_1 = np.array(coord_1, dtype="float64")  
        self.element_list_1 = element_list_1
        self.electric_charge_and_multiplicity_1 = electric_charge_and_multiplicity

        self.coords_2 = np.array(coord_2, dtype="float64")  
        self.element_list_2 = element_list_2
        self.electric_charge_and_multiplicity_2 = electric_charge_and_multiplicity
