import numpy as np
import datetime
import os
import glob

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Visualization.visualization import Graph
from multioptpy.fileio import make_workspace, xyz2list
from multioptpy.Parameters.parameter import UnitValueLib, element_number


class DimerMethod:
    def __init__(self, config):
        """
        Implementation of the Dimer method for finding saddle points.
        
        The Dimer method is a minimum mode following technique that uses 
        two points (a dimer) to find the lowest curvature mode without
        explicitly calculating the Hessian matrix.
        
        References:
        - J. Chem. Phys. 111, 7010 (1999) - Original Dimer Method
        - J. Chem. Phys. 121, 9776 (2004) - Improvements
        - J. Chem. Phys. 123, 224101 (2005) - Additional improvements
        - J. Chem. Phys. 128, 014106 (2008) - Force extrapolation scheme
        """
        self.config = config
        self.energy_list = []
        self.gradient_list = []
        self.curvature_list = []
        self.init_displacement = 0.03 / self.get_unit_conversion()  # Bohr
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.converge_criteria = 0.001  # Force convergence criteria in a.u.
        self.element_number_list = None
        self.optimized_structures = {}  # Dictionary to store optimized structures
        
        # Dimer method specific parameters
        self.dimer_parameters = {
            'dimer_separation': 0.0001,  # Distance between dimer images (Bohr)
            'trial_angle': np.pi / 32.0,  # Trial rotation angle (radians)
            'f_rot_min': 0.1,            # Min rotational force for rotation
            'f_rot_max': 1.0,            # Max rotational force for only one rotation
            'max_num_rot': 5,            # Maximum number of rotations per step
            'extrapolate_forces': True,  # Use force extrapolation scheme
            'max_iterations': 1000,       # Maximum iterations for dimer optimization
            'max_step': 0.1,             # Maximum translation step size
            'trial_step': 0.001,         # Step size for curvature estimation
            'cg_translation': True,      # Use conjugate gradient for translation
            'cg_rotation': False,        # Use conjugate gradient for rotation
            'quickmin_rotation': True,   # Use quickmin for rotation
            'max_rot_iterations': 64,    # Max iterations for rotation
            'max_force_rotation': 1e-3,  # Max force for rotation convergence
            'potim': 0.1,                # Time step for quickmin
        }
        
        # For CG optimization (translation)
        self.cg_init_translation = True
        self.old_direction_translation = None
        self.cg_direction_translation = None

        # For CG optimization (rotation)
        self.cg_init_rotation = True
        self.old_rot_force = None
        self.old_rot_gradient = None
        self.current_rot_gradient_unit = None
        self.rot_velocity = None  # For QuickMin rotation

    def get_unit_conversion(self):
        """Return bohr to angstrom conversion factor"""
        return UnitValueLib().bohr2angstroms  # Approximate value for bohr2angstroms
        
    def adjust_center2origin(self, coord):
        """Adjust coordinates to have center at origin"""
        center = np.mean(coord, axis=0)
        return coord - center

    def normalize(self, vector):
        """Create a unit vector along *vector*"""
        vector_flat = vector.flatten()
        norm = np.linalg.norm(vector_flat)
        if norm < 1e-10:
            return vector  # Return original vector if it's too small
        return (vector_flat / norm).reshape(vector.shape)
        
    def parallel_vector(self, vector, base):
        """Extract the components of *vector* that are parallel to *base*"""
        vector_flat = vector.flatten()
        base_flat = base.flatten()
        base_norm = np.linalg.norm(base_flat)
        if base_norm < 1e-10:
            return np.zeros_like(vector)
        base_unit = base_flat / base_norm
        return (np.dot(vector_flat, base_unit) * base_unit).reshape(vector.shape)
        
    def perpendicular_vector(self, vector, base):
        """Remove the components of *vector* that are parallel to *base*"""
        return vector - self.parallel_vector(vector, base)
        
    def rotate_vector_around_axis(self, vec_to_rotate, axis, angle):
        """Rotates a vector around a given axis by a specified angle (Rodrigues' rotation formula)"""
        axis = self.normalize(axis)
        k = axis.flatten()
        v = vec_to_rotate.flatten()

        v_rot = v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))
        return v_rot.reshape(vec_to_rotate.shape)

    def print_status(self, iteration, energy, curvature, max_force, rot_force=None, rotation_angle=None):
        """Print status information during optimization"""
        status = f"Iteration {iteration}: Energy = {energy:.6f}, Curvature = {curvature:.6f}, Max Force = {max_force:.6f}"
        if rot_force is not None:
            status += f", Rotational Force = {rot_force:.6f}"
        if rotation_angle is not None:
            status += f", Rotation Angle = {rotation_angle:.6f} rad"
        print(status)
        
    def get_cg_direction_translation(self, direction):
        """Apply Conjugate Gradient algorithm to step direction for translation"""
        direction_shape = direction.shape
        direction_flat = direction.flatten()
        
        if self.cg_init_translation:
            self.cg_init_translation = False
            self.old_direction_translation = direction_flat.copy()
            self.cg_direction_translation = direction_flat.copy()
        
        old_norm = np.dot(self.old_direction_translation, self.old_direction_translation)
        
        # Polak-Ribiere formula for conjugate gradient
        if old_norm > 1e-10:
            betaPR = np.dot(direction_flat, 
                          (direction_flat - self.old_direction_translation)) / old_norm
        else:
            betaPR = 0.0
            
        if betaPR < 0.0:
            betaPR = 0.0
            
        self.cg_direction_translation = direction_flat + self.cg_direction_translation * betaPR
        self.old_direction_translation = direction_flat.copy()
        
        return self.cg_direction_translation.reshape(direction_shape)

    def calculate_gradient(self, QMC, x):
        """Calculate gradient at point x"""
        element_number_list = self.get_element_number_list()
        _, grad_x, _, iscalculationfailed = QMC.single_point(
            None, element_number_list, "", self.electric_charge_and_multiplicity, 
            self.method, x
        )
        
        if iscalculationfailed:
            return False
            
        # Apply bias if needed
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, _, bias_gradient, _ = BPC.main(
            0, grad_x, x, element_number_list, self.config.force_data
        )
        
        return bias_gradient

    def dimer_rotate(self, SP_obj, center_coords, dimer_axis, element_list, charge_multiplicity, method):
        """
        Perform dimer rotation to find the lowest curvature direction.
        
        Parameters:
        -----------
        SP_obj : Calculation object
            Object for performing single point calculations
        center_coords : ndarray
            Coordinates of the center point of the dimer
        dimer_axis : ndarray
            Current orientation of the dimer axis
        element_list : list
            List of element symbols
        charge_multiplicity : list
            [charge, multiplicity]
        method : str
            Calculation method
            
        Returns:
        --------
        ndarray, float, bool
            New dimer axis, curvature along this axis, and success flag
        """
        # Parameters for rotation
        dR = self.dimer_parameters['dimer_separation']
        trial_angle = self.dimer_parameters['trial_angle']
        max_rot_iterations = self.dimer_parameters['max_rot_iterations']
        max_force_rotation = self.dimer_parameters['max_force_rotation']
        cg_rotation = self.dimer_parameters['cg_rotation']
        quickmin_rotation = self.dimer_parameters['quickmin_rotation']
        potim = self.dimer_parameters['potim']

        # Ensure dimer_axis has correct shape and is normalized
        dimer_axis = self.normalize(np.array(dimer_axis).reshape(center_coords.shape))
        
        # Initial forces and energies at center_coords
        energy_center, forces_center, _, failed = SP_obj.single_point(
            None, element_list, "", charge_multiplicity, method, center_coords
        )
        if failed:
            return None, None, True

        # Calculate forces at the dimer endpoints
        pos1 = center_coords + dimer_axis * dR
        pos2 = center_coords - dimer_axis * dR

        energy1, forces1, _, failed = SP_obj.single_point(
            None, element_list, "", charge_multiplicity, method, pos1
        )
        if failed:
            return None, None, True

        energy2, forces2, _, failed = SP_obj.single_point(
            None, element_list, "", charge_multiplicity, method, pos2
        )
        if failed:
            return None, None, True

        # Apply bias potential
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, bias_energy_center, bias_forces_center, _ = BPC.main(
            energy_center, forces_center, center_coords, element_list, self.config.force_data
        )
        _, bias_energy1, bias_forces1, _ = BPC.main(
            energy1, forces1, pos1, element_list, self.config.force_data
        )
        _, bias_energy2, bias_forces2, _ = BPC.main(
            energy2, forces2, pos2, element_list, self.config.force_data
        )
        
        # Update with bias forces
        forces_center = bias_forces_center
        forces1 = bias_forces1
        forces2 = bias_forces2

        # Calculate rotational forces and curvature
        fd1 = np.sum(forces1 * dimer_axis)
        fd2 = np.sum(forces2 * dimer_axis)
        fn1 = forces1 - dimer_axis * fd1
        fn2 = forces2 - dimer_axis * fd2

        rotational_force_gradient = (fn1 - fn2) / dR
        rotational_force_magnitude = np.linalg.norm(rotational_force_gradient.flatten())

        curvature = np.dot((forces2 - forces1).flatten(), dimer_axis.flatten()) / (2.0 * dR)

        # Initialize for rotation
        current_dimer_axis = dimer_axis.copy()
        rot_iteration = 0

        if quickmin_rotation:
            # QuickMin for dimer rotation
            if self.rot_velocity is None:
                self.rot_velocity = np.zeros_like(center_coords)

            while rot_iteration < max_rot_iterations:
                # Calculate velocity update
                dv_rot = rotational_force_gradient * (dR * potim)
                vdv = np.sum(self.rot_velocity * dv_rot)

                # Update velocity
                if vdv > 0.0 and np.sum(dv_rot**2) > 1e-10:
                    self.rot_velocity = dv_rot * (1.0 + vdv / np.sum(dv_rot**2))
                else:
                    self.rot_velocity = dv_rot

                # Update dimer axis
                current_r1 = center_coords + current_dimer_axis * dR
                new_r1 = current_r1 + self.rot_velocity * potim
                
                new_dimer_axis_unnormalized = new_r1 - center_coords
                current_dimer_axis = self.normalize(new_dimer_axis_unnormalized)

                # Recalculate forces for new dimer axis
                pos1_new = center_coords + current_dimer_axis * dR
                energy1_new, forces1_new, _, failed = SP_obj.single_point(
                    None, element_list, "", charge_multiplicity, method, pos1_new
                )
                if failed: 
                    return None, None, True
                
                # Apply bias potential
                _, _, bias_forces1_new, _ = BPC.main(
                    energy1_new, forces1_new, pos1_new, element_list, self.config.force_data
                )
                forces1_new = bias_forces1_new
                
                forces2_new = 2 * forces_center - forces1_new

                fd1_new = np.sum(forces1_new * current_dimer_axis)
                fd2_new = np.sum(forces2_new * current_dimer_axis)
                fn1_new = forces1_new - current_dimer_axis * fd1_new
                fn2_new = forces2_new - current_dimer_axis * fd2_new
                rotational_force_gradient = (fn1_new - fn2_new) / dR
                rotational_force_magnitude = np.linalg.norm(rotational_force_gradient.flatten())

                # Update curvature
                curvature = np.dot((forces2_new - forces1_new).flatten(), current_dimer_axis.flatten()) / (2.0 * dR)

                self.print_status(rot_iteration, bias_energy_center, curvature, 
                                 np.max(np.abs(forces_center)), rotational_force_magnitude)

                if rotational_force_magnitude < max_force_rotation or rot_iteration >= max_rot_iterations - 1:
                    break
                
                rot_iteration += 1
                
        elif cg_rotation:
            # Conjugate Gradient for rotation
            if self.cg_init_rotation:
                self.old_rot_force = rotational_force_gradient.copy()
                self.old_rot_gradient = rotational_force_gradient.copy()
                self.current_rot_gradient_unit = self.normalize(rotational_force_gradient)
                self.cg_init_rotation = False
            
            while rot_iteration < max_rot_iterations:
                # Polak-Ribiere formula
                gam_n = 0.0
                if np.linalg.norm(self.old_rot_force.flatten()) > 1e-10:
                    gam_n = np.dot(rotational_force_gradient.flatten(), 
                                   (rotational_force_gradient - self.old_rot_force).flatten()) / np.dot(self.old_rot_force.flatten(), self.old_rot_force.flatten())
                if gam_n < 0: 
                    gam_n = 0.0

                # Update gradient
                current_rot_gradient = rotational_force_gradient + self.old_rot_gradient * gam_n
                self.current_rot_gradient_unit = self.normalize(current_rot_gradient)

                # Calculate force components
                fnp1 = self.current_rot_gradient_unit * np.sum(rotational_force_gradient * self.current_rot_gradient_unit)
                fnrp1 = np.sum(fnp1 * self.current_rot_gradient_unit)

                # Trial rotation
                n_tmp = current_dimer_axis.copy()
                gnu_tmp = self.current_rot_gradient_unit.copy()
                rotated_n_trial = self.rotate_vector_around_axis(n_tmp, gnu_tmp, trial_angle)
                
                # Calculate forces at trial position
                pos1_trial = center_coords + rotated_n_trial * dR
                energy1_trial, forces1_trial, _, failed = SP_obj.single_point(
                    None, element_list, "", charge_multiplicity, method, pos1_trial
                )
                if failed: 
                    return None, None, True
                
                # Apply bias potential
                _, _, bias_forces1_trial, _ = BPC.main(
                    energy1_trial, forces1_trial, pos1_trial, element_list, self.config.force_data
                )
                forces1_trial = bias_forces1_trial
                
                forces2_trial = 2 * forces_center - forces1_trial

                # Calculate rotational force for trial
                fd1_trial = np.sum(forces1_trial * rotated_n_trial)
                fd2_trial = np.sum(forces2_trial * rotated_n_trial)
                fn1_trial = forces1_trial - rotated_n_trial * fd1_trial
                fn2_trial = forces2_trial - rotated_n_trial * fd2_trial
                rotational_force_gradient_trial = (fn1_trial - fn2_trial) / dR

                # Calculate optimal rotation angle
                fnp2 = gnu_tmp * np.sum(rotational_force_gradient_trial * gnu_tmp)
                fnrp2 = np.sum(fnp2 * gnu_tmp)

                cth = (fnrp1 - fnrp2) / trial_angle
                fnrp = (fnrp1 + fnrp2) / 2.0

                rotation_angle = 0.0
                if abs(cth) > 1e-10:
                    rotation_angle = np.arctan((fnrp / cth) * 2.0) / 2.0 + trial_angle / 2.0
                    if cth < 0: 
                        rotation_angle += np.pi / 2.0
                else:
                    rotation_angle = trial_angle / 2.0

                # Apply optimal rotation
                current_dimer_axis = self.rotate_vector_around_axis(n_tmp, gnu_tmp, rotation_angle)
                current_dimer_axis = self.normalize(current_dimer_axis)

                # Update gradient history
                self.old_rot_force = rotational_force_gradient.copy()
                self.old_rot_gradient = current_rot_gradient.copy()

                # Recalculate forces for new axis
                pos1_new = center_coords + current_dimer_axis * dR
                energy1_new, forces1_new, _, failed = SP_obj.single_point(
                    None, element_list, "", charge_multiplicity, method, pos1_new
                )
                if failed: 
                    return None, None, True
                
                # Apply bias potential
                _, _, bias_forces1_new, _ = BPC.main(
                    energy1_new, forces1_new, pos1_new, element_list, self.config.force_data
                )
                forces1_new = bias_forces1_new
                
                forces2_new = 2 * forces_center - forces1_new

                # Update rotational forces and curvature
                fd1_new = np.sum(forces1_new * current_dimer_axis)
                fd2_new = np.sum(forces2_new * current_dimer_axis)
                fn1_new = forces1_new - current_dimer_axis * fd1_new
                fn2_new = forces2_new - current_dimer_axis * fd2_new
                rotational_force_gradient = (fn1_new - fn2_new) / dR
                rotational_force_magnitude = np.linalg.norm(rotational_force_gradient.flatten())

                curvature = np.dot((forces2_new - forces1_new).flatten(), current_dimer_axis.flatten()) / (2.0 * dR)

                self.print_status(rot_iteration, bias_energy_center, curvature, 
                                 np.max(np.abs(forces_center)), rotational_force_magnitude, rotation_angle)

                if rotational_force_magnitude < max_force_rotation or rot_iteration >= max_rot_iterations - 1:
                    break
                
                rot_iteration += 1
                
        else:
            # Original dimer rotation algorithm
            rotation_count = 0
            forces1A = forces1.copy()

            while rotation_count < max_rot_iterations:
                rot_force = self.perpendicular_vector((forces1 - forces2), current_dimer_axis)
                rot_force_magnitude = np.linalg.norm(rot_force.flatten())

                if rot_force_magnitude <= self.dimer_parameters["f_rot_min"]:
                    break
                if rot_force_magnitude <= self.dimer_parameters["f_rot_max"] and rotation_count > 0:
                    break

                n_A = current_dimer_axis.copy()
                rot_unit_A = self.normalize(rot_force)

                c0 = curvature
                c0d = np.dot((forces2 - forces1).flatten(), rot_unit_A.flatten()) / dR

                # Trial rotation
                n_B = self.rotate_vector_around_axis(n_A, rot_unit_A, trial_angle)
                n_B = self.normalize(n_B)

                pos1B = center_coords + n_B * dR
                energy1B, forces1B, _, failed = SP_obj.single_point(
                    None, element_list, "", charge_multiplicity, method, pos1B
                )
                if failed: 
                    return None, None, True
                
                # Apply bias potential
                _, _, bias_forces1B, _ = BPC.main(
                    energy1B, forces1B, pos1B, element_list, self.config.force_data
                )
                forces1B = bias_forces1B
                
                forces2B = 2 * forces_center - forces1B

                c1d = np.dot((forces2B - forces1B).flatten(), rot_unit_A.flatten()) / dR

                # Calculate optimal rotation angle
                a1 = c0d * np.cos(2 * trial_angle) - c1d / (2 * np.sin(2 * trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)

                rotation_angle = 0.0
                if abs(a1) > 1e-10:
                    rotation_angle = np.arctan(b1 / a1) / 2.0
                
                cmin = a0 / 2.0 + a1 * np.cos(2 * rotation_angle) + b1 * np.sin(2 * rotation_angle)
                if c0 < cmin:
                    rotation_angle += np.pi / 2.0

                # Apply optimal rotation
                current_dimer_axis = self.rotate_vector_around_axis(n_A, rot_unit_A, rotation_angle)
                current_dimer_axis = self.normalize(current_dimer_axis)

                curvature = cmin

                # Calculate forces at new orientation
                pos1 = center_coords + current_dimer_axis * dR
                energy1, forces1, _, failed = SP_obj.single_point(
                    None, element_list, "", charge_multiplicity, method, pos1
                )
                if failed: 
                    return None, None, True
                
                # Apply bias potential
                _, _, bias_forces1, _ = BPC.main(
                    energy1, forces1, pos1, element_list, self.config.force_data
                )
                forces1 = bias_forces1
                
                forces2 = 2 * forces_center - forces1

                self.print_status(rotation_count, bias_energy_center, curvature, 
                                 np.max(np.abs(forces_center)), rot_force_magnitude, rotation_angle)
                
                rotation_count += 1

        return current_dimer_axis, curvature, False

    def dimer_translate(self, SP_obj, coords, dimer_axis, curvature, element_list, charge_multiplicity, method):
        """
        Translate the dimer to find a saddle point.
        
        Parameters:
        -----------
        SP_obj : Calculation object
            Object for performing single point calculations
        coords : ndarray
            Current coordinates
        dimer_axis : ndarray
            Current dimer axis (normalized)
        curvature : float
            Current curvature along the dimer axis
        element_list : list
            List of element symbols
        charge_multiplicity : list
            [charge, multiplicity]
        method : str
            Calculation method
            
        Returns:
        --------
        ndarray, float, bool
            New coordinates, energy, and success flag
        """
        # Parameters for translation
        max_step = self.dimer_parameters["max_step"]
        cg_translation = self.dimer_parameters["cg_translation"]
        
        # Normalize dimer_axis
        dimer_axis = self.normalize(np.array(dimer_axis).reshape(coords.shape))
        
        # Get forces at current position
        energy, forces, _, failed = SP_obj.single_point(
            None, element_list, "", charge_multiplicity, method, coords
        )
        if failed:
            return None, None, True
            
        # Apply bias potential
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, bias_energy, bias_forces, _ = BPC.main(
            energy, forces, coords, element_list, self.config.force_data
        )
        forces = bias_forces
            
        # Modify forces according to dimer method
        f_parallel = self.parallel_vector(forces, dimer_axis)
        
        if curvature < 0:
            # Invert component parallel to dimer axis
            modified_forces = forces - 2 * f_parallel
        else:
            # Only consider inverted parallel component
            modified_forces = -f_parallel
            
        # Apply conjugate gradient if enabled
        if cg_translation:
            direction = self.get_cg_direction_translation(modified_forces)
        else:
            direction = modified_forces
            
        # Normalize direction and apply step size
        direction = self.normalize(direction)
        step_size = max_step
        
        # Calculate new coordinates
        new_coords = coords + direction * step_size
        
        # Calculate energy at new position
        new_energy, _, _, failed = SP_obj.single_point(
            None, element_list, "", charge_multiplicity, method, new_coords
        )
        if failed:
            return None, None, True
            
        # Apply bias potential
        _, bias_new_energy, _, _ = BPC.main(
            new_energy, np.zeros_like(coords), new_coords, element_list, self.config.force_data
        )
        new_energy = bias_new_energy
            
        return new_coords, new_energy, False
        
    def save_structure(self, coords, element_list, iteration, energy, curvature, label):
        """Save structure to XYZ file"""
        # Create filename
        filename = f"{label}_iter_{iteration}.xyz"
        filepath = os.path.join(self.directory, "dimer_structures", filename)
        
        # Convert coordinates to Angstroms
        coords_ang = coords * self.get_unit_conversion()
        
        # Write XYZ file
        with open(filepath, 'w') as f:
            f.write(f"{len(element_list)}\n")
            f.write(f"Dimer {label} - Iteration {iteration} - Energy {energy:.6f} - Curvature {curvature:.6f}\n")
            for i, (element, coord) in enumerate(zip(element_list, coords_ang)):
                f.write(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
                
    def save_eigenmode(self, eigenmode, element_list, iteration, curvature):
        """Save eigenmode to XYZ file"""
        # Create filename
        filename = f"eigenmode_iter_{iteration}.xyz"
        filepath = os.path.join(self.directory, "dimer_structures", filename)
        
        # Scale eigenmode for visualization
        scale_factor = 0.3 / np.max(np.abs(eigenmode)) if np.max(np.abs(eigenmode)) > 1e-10 else 0.3
        scaled_mode = eigenmode * scale_factor
        
        # Write XYZ file
        with open(filepath, 'w') as f:
            f.write(f"{len(element_list)}\n")
            f.write(f"Dimer eigenmode - Iteration {iteration} - Curvature {curvature:.6f}\n")
            for i, (element, mode) in enumerate(zip(element_list, scaled_mode)):
                f.write(f"{element} {mode[0]:.12f} {mode[1]:.12f} {mode[2]:.12f}\n")
                
    def create_trajectory_file(self, element_list):
        """Create a trajectory file from all saved structures"""
        # Get all structure files sorted by iteration
        structure_files = glob.glob(os.path.join(self.directory, "dimer_structures", "optimization_iter_*.xyz"))
        structure_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # If no files found, return
        if not structure_files:
            return
            
        # Create trajectory file
        trajectory_file = os.path.join(self.directory, "dimer_trajectory.xyz")
        
        with open(trajectory_file, 'w') as outfile:
            for file in structure_files:
                with open(file, 'r') as infile:
                    outfile.write(infile.read())
                    
        print(f"Created trajectory file: {trajectory_file}")
        
    def create_distance_plots(self):
        """Create CSV files with distance and energy data for plotting"""
        iterations = range(len(self.energy_list))
        
        # Path for the data file
        data_file = os.path.join(self.directory, "dimer_results.csv")
        
        # Write the data
        with open(data_file, 'w') as f:
            f.write("iteration,energy,curvature\n")
            for i in iterations:
                f.write(f"{i},{self.energy_list[i]:.6f},{self.curvature_list[i]:.6f}\n")
        
        print(f"Created data file: {data_file}")
        
    # Getters and setters - keep same style as ADDFlikeMethod
    def set_molecule(self, element_list, coords):
        self.element_list = element_list
        self.coords = coords
    
    def set_gradient(self, gradient):
        self.gradient = gradient
    
    def set_hessian(self, hessian):
        self.hessian = hessian
    
    def set_energy(self, energy):
        self.energy = energy
    
    def set_coords(self, coords):
        self.coords = coords
    
    def set_element_list(self, element_list):
        self.element_list = element_list
        self.element_number_list = [element_number(i) for i in self.element_list]
    
    def set_coord(self, coord):
        self.coords = coord
    
    def get_coord(self):
        return self.coords
    
    def get_element_list(self):
        return self.element_list
    
    def get_element_number_list(self):
        if self.element_number_list is None:
            if self.element_list is None:
                raise ValueError('Element list is not set.')
            self.element_number_list = [element_number(i) for i in self.element_list]
        return self.element_number_list
    
    def set_mole_info(self, base_file_name, electric_charge_and_multiplicity):
        """Load molecular information from XYZ file"""
        coord, element_list, electric_charge_and_multiplicity = xyz2list(
            base_file_name + ".xyz", electric_charge_and_multiplicity)

        if self.config.usextb != "None":
            self.method = self.config.usextb
        elif self.config.usedxtb != "None":
            self.method = self.config.usedxtb
        else:
            self.method = "None"

        self.coords = np.array(coord, dtype="float64")  
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        
    def run(self, file_directory, SP, electric_charge_and_multiplicity, FIO_img):
        """
        Main method to run Dimer optimization.
        
        Parameters:
        -----------
        file_directory : str
            Path to input file
        SP : SinglePointCalculation object
            Object for performing single point calculations
        electric_charge_and_multiplicity : list
            [charge, multiplicity]
        FIO_img : FileIO object
            Object for file I/O operations
            
        Returns:
        --------
        bool
            True if optimization succeeded, False otherwise
        """
        print("### Start Dimer Method ###")
        
        # Preparation
        base_file_name = os.path.splitext(FIO_img.START_FILE)[0]
        self.set_mole_info(base_file_name, electric_charge_and_multiplicity)
        
        self.directory = make_workspace(file_directory)
        
        # Create directory for dimer structures
        os.makedirs(os.path.join(self.directory, "dimer_structures"), exist_ok=True)
        
        # Initial coordinates
        initial_coords = self.get_coord()
        element_list = self.get_element_list()
        
        # Initial energy and forces
        energy, forces, _, failed = SP.single_point(
            None, element_list, "", electric_charge_and_multiplicity, self.method, initial_coords
        )
        if failed:
            print("Initial calculation failed.")
            return False
            
        # Apply bias potential
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, bias_energy, bias_forces, _ = BPC.main(
            energy, forces, initial_coords, element_list, self.config.force_data
        )
        
        print(f"Initial energy: {bias_energy:.6f}, Max force: {np.max(np.abs(bias_forces)):.6f}")
        
        # Save initial structure
        self.save_structure(initial_coords, element_list, 0, bias_energy, 0.0, "initial")
        
        # Initialize lists for tracking
        self.energy_list = [bias_energy]
        self.gradient_list = [np.linalg.norm(bias_forces)]
        self.curvature_list = [0.0]
        
        # Initialize dimer axis (random for now)
        coords_flat = initial_coords.flatten()
        dimer_axis = np.random.rand(*initial_coords.shape) - 0.5
        dimer_axis = self.normalize(dimer_axis)
        
        # Main iteration loop
        max_iterations = self.dimer_parameters["max_iterations"]
        converged = False
        iteration = 0
        current_coords = initial_coords.copy()
        
        while iteration < max_iterations and not converged:
            print(f"\n### Iteration {iteration+1} ###")
            
            # Step 1: Rotate the dimer to find lowest curvature mode
            print("Rotating dimer to find lowest curvature mode...")
            dimer_axis, curvature, failed = self.dimer_rotate(
                SP, current_coords, dimer_axis, element_list, 
                electric_charge_and_multiplicity, self.method
            )
            
            if failed:
                print(f"Dimer rotation failed at iteration {iteration+1}")
                break
                
            print(f"After rotation: Curvature = {curvature:.6f}")
            
            # Save the eigenmode
            self.save_eigenmode(dimer_axis, element_list, iteration, curvature)
            
            # Step 2: Translate the dimer
            print("Translating dimer...")
            new_coords, new_energy, failed = self.dimer_translate(
                SP, current_coords, dimer_axis, curvature, element_list,
                electric_charge_and_multiplicity, self.method
            )
            
            if failed:
                print(f"Dimer translation failed at iteration {iteration+1}")
                break
                
            # Calculate forces at new position
            _, new_forces, _, failed = SP.single_point(
                None, element_list, "", electric_charge_and_multiplicity, self.method, new_coords
            )
            if failed:
                print(f"Force calculation failed at iteration {iteration+1}")
                break
                
            # Apply bias potential
            _, _, bias_new_forces, _ = BPC.main(
                0, new_forces, new_coords, element_list, self.config.force_data
            )
            new_forces = bias_new_forces
                
            # Calculate maximum force component
            max_force = np.max(np.abs(new_forces))
            
            # Store results for this iteration
            self.energy_list.append(new_energy)
            self.curvature_list.append(curvature)
            self.gradient_list.append(max_force)
            
            # Print status
            energy_change = new_energy - self.energy_list[-2] if iteration > 0 else 0.0
            print(f"After translation: Energy = {new_energy:.6f} (Δ = {energy_change:.6f})")
            print(f"                   Max Force = {max_force:.6f}")
            print(f"                   Curvature = {curvature:.6f}")
            
            # Save structure for this iteration
            self.save_structure(new_coords, element_list, iteration+1, new_energy, curvature, "optimization")
            
            # Check convergence
            if max_force < self.converge_criteria and curvature < 0:
                converged = True
                print("\n### Dimer method converged to a saddle point! ###")
                self.save_structure(new_coords, element_list, iteration+1, new_energy, curvature, "final_saddle_point")
            
            # Store this structure
            structure_info = {
                'iteration': iteration+1,
                'energy': new_energy,
                'curvature': curvature,
                'max_force': max_force,
                'coords': new_coords.copy(),
                'comment': f"Dimer Iteration {iteration+1} Energy {new_energy:.6f} Curvature {curvature:.6f}"
            }
            self.optimized_structures[iteration+1] = structure_info
            
            # Update for next iteration
            current_coords = new_coords
            
            # Reset CG for next step (for translation)
            self.cg_init_translation = True
            
            iteration += 1
            
        # Create trajectory file
        self.create_trajectory_file(element_list)
        
        # Create data plots
        self.create_distance_plots()
        
        # Plot optimization progress using Graph class from ieip.py
        G = Graph(self.config.iEIP_FOLDER_DIRECTORY)
        iterations_plot = list(range(len(self.energy_list)))
        
        G.single_plot(iterations_plot, self.energy_list, file_directory, "dimer_energy", 
                    axis_name_2="Energy [Hartree]", name="dimer_energy")
        G.single_plot(iterations_plot, self.curvature_list, file_directory, "dimer_curvature", 
                    axis_name_2="Curvature", name="dimer_curvature")
        G.single_plot(iterations_plot, self.gradient_list, file_directory, "dimer_gradient", 
                    axis_name_2="Max Force [a.u.]", name="dimer_gradient")
        
        if converged:
            print(f"Dimer method converged after {iteration} iterations.")
            print(f"Final energy: {new_energy:.6f}")
            print(f"Final curvature: {curvature:.6f}")
            print(f"Final max force: {max_force:.6f}")
            return True
        else:
            print(f"Dimer method did not converge after {iteration} iterations.")
            if iteration > 0:
                print(f"Final energy: {new_energy:.6f}")
                print(f"Final curvature: {curvature:.6f}")
                print(f"Final max force: {max_force:.6f}")
            return False
