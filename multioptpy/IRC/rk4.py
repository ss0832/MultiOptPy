import numpy as np
import os
import copy
import csv

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass
from multioptpy.IRC.converge_criteria import convergence_check
from multioptpy.Visualization.visualization import Graph
from multioptpy.PESAnalyzer.calc_irc_curvature import calc_irc_curvature_properties, save_curvature_properties_to_file


class RK4:
    """Runge-Kutta 4th order method for IRC calculations
    
    References
    ----------
    [1] J. Chem. Phys. 95, 9, 6758–6763 (1991)
    [2] Chem. Phys. Lett. 437, 1–3, 120-125 (2007)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None, **kwargs):
        """Initialize RK4 IRC calculator
        
        Parameters
        ----------
        element_list : list
            List of atomic elements
        electric_charge_and_multiplicity : tuple
            Charge and multiplicity for the system
        FC_count : int
            Frequency of full hessian recalculation
        file_directory : str
            Working directory
        final_directory : str
            Directory for final output
        force_data : dict
            Force field data for bias potential
        max_step : int, optional
            Maximum number of steps
        step_size : float, optional
            Step size for the IRC
        init_coord : numpy.ndarray, optional
            Initial coordinates
        init_hess : numpy.ndarray, optional
            Initial hessian
        calc_engine : object, optional
            Calculator engine
        xtb_method : str, optional
            XTB method specification
        """
        self.max_step = max_step
        self.step_size = step_size
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
        # initial condition
        self.coords = init_coord
        self.init_hess = init_hess
        self.mw_hessian = init_hess  # Mass-weighted hessian
        self.xtb_method = xtb_method
        
        # convergence criteria
        self.MAX_FORCE_THRESHOLD = 0.0004
        self.RMS_FORCE_THRESHOLD = 0.0001

        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # IRC data storage for current calculation (needed for immediate operations)
        # These will no longer store the full trajectory but only recent points needed for calculations
        self.irc_bias_energy_list = []
        self.irc_energy_list = []
        self.irc_mw_coords = []
        self.irc_mw_gradients = []
        self.irc_mw_bias_gradients = []
        self.path_bending_angle_list = []
        
        # Create data files
        self.create_csv_file()
        self.create_xyz_file()
    
    def create_csv_file(self):
        """Create CSV file for energy and gradient data"""
        self.csv_filename = os.path.join(self.directory, "irc_energies_gradients.csv")
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 'RMS Gradient', 'RMS Bias Gradient'])
    
    def create_xyz_file(self):
        """Create XYZ file for structure data"""
        self.xyz_filename = os.path.join(self.directory, "irc_structures.xyz")
        # Create empty file (will be appended to later)
        open(self.xyz_filename, 'w').close()
    
    def save_to_csv(self, step, energy, bias_energy, gradient, bias_gradient):
        """Save energy and gradient data to CSV file
        
        Parameters
        ----------
        step : int
            Current step number
        energy : float
            Energy in Hartree
        bias_energy : float
            Bias energy in Hartree
        gradient : numpy.ndarray
            Gradient array
        bias_gradient : numpy.ndarray
            Bias gradient array
        """
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad])
    
    def save_xyz_structure(self, step, coords):
        """Save molecular structure to XYZ file
        
        Parameters
        ----------
        step : int
            Current step number
        coords : numpy.ndarray
            Atomic coordinates in Bohr
        """
        # Convert coordinates to Angstroms
        coords_angstrom = coords * UnitValueLib().bohr2angstroms
        
        with open(self.xyz_filename, 'a') as f:
            # Number of atoms and comment line
            f.write(f"{len(coords)}\n")
            f.write(f"IRC Step {step}\n")
            
            # Atomic coordinates
            for i, coord in enumerate(coords_angstrom):
                f.write(f"{self.element_list[i]:<3} {coord[0]:15.10f} {coord[1]:15.10f} {coord[2]:15.10f}\n")
    
    def get_mass_array(self):
        """Create arrays of atomic masses for mass-weighting operations"""
        elem_mass_list = np.array([atomic_mass(elem) for elem in self.element_list], dtype="float64")
        sqrt_mass_list = np.sqrt(elem_mass_list)
        
        # Create arrays for 3D operations (x,y,z for each atom)
        three_elem_mass_list = np.repeat(elem_mass_list, 3)
        three_sqrt_mass_list = np.repeat(sqrt_mass_list, 3)
        
        return elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list
    
    def mass_weight_hessian(self, hessian, three_sqrt_mass_list):
        """Apply mass-weighting to the hessian matrix
        
        Parameters
        ----------
        hessian : numpy.ndarray
            Hessian matrix in non-mass-weighted coordinates
        three_sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values repeated for x,y,z per atom
            
        Returns
        -------
        numpy.ndarray
            Mass-weighted hessian
        """
        mass_mat = np.diag(1.0 / three_sqrt_mass_list)
        return np.dot(mass_mat, np.dot(hessian, mass_mat))
    
    def mass_weight_coordinates(self, coordinates, sqrt_mass_list):
        """Convert coordinates to mass-weighted coordinates
        
        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinates in non-mass-weighted form
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
            
        Returns
        -------
        numpy.ndarray
            Mass-weighted coordinates
        """
        mw_coords = copy.deepcopy(coordinates)
        for i in range(len(coordinates)):
            mw_coords[i] = coordinates[i] * sqrt_mass_list[i]
        return mw_coords
    
    def mass_weight_gradient(self, gradient, sqrt_mass_list):
        """Convert gradient to mass-weighted form
        
        Parameters
        ----------
        gradient : numpy.ndarray
            Gradient in non-mass-weighted form
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
            
        Returns
        -------
        numpy.ndarray
            Mass-weighted gradient
        """
        mw_gradient = copy.deepcopy(gradient)
        for i in range(len(gradient)):
            mw_gradient[i] = gradient[i] / sqrt_mass_list[i]
        return mw_gradient
    
    def unmass_weight_step(self, step, sqrt_mass_list):
        """Convert a step vector from mass-weighted to non-mass-weighted coordinates
        
        Parameters
        ----------
        step : numpy.ndarray
            Step in mass-weighted form
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
            
        Returns
        -------
        numpy.ndarray
            Step in non-mass-weighted coordinates
        """
        unmw_step = copy.deepcopy(step)
        for i in range(len(step)):
            unmw_step[i] = step[i] / sqrt_mass_list[i]
        return unmw_step
    
    def check_energy_oscillation(self, energy_list):
        """Check if energy is oscillating (going up and down)
        
        Parameters
        ----------
        energy_list : list
            List of energy values
            
        Returns
        -------
        bool
            True if energy is oscillating, False otherwise
        """
        if len(energy_list) < 3:
            return False
        
        # Check if the energy changes direction (from increasing to decreasing or vice versa)
        last_diff = energy_list[-1] - energy_list[-2]
        prev_diff = energy_list[-2] - energy_list[-3]
        
        # Return True if the energy direction has changed
        return (last_diff * prev_diff) < 0
    
    def get_k(self, mw_coords, mw_gradient):
        """Calculate k value for RK4 integration
        
        Parameters
        ----------
        mw_coords : numpy.ndarray
            Mass-weighted coordinates
        mw_gradient : numpy.ndarray
            Mass-weighted gradient
            
        Returns
        -------
        numpy.ndarray
            k vector for RK4 step
        """
        # Flatten gradient for norm calculation
        flat_gradient = mw_gradient.flatten()
        norm = np.linalg.norm(flat_gradient)
        
        # Avoid division by zero
        if norm < 1e-12:
            direction = np.zeros_like(mw_gradient)
        else:
            direction = -mw_gradient / norm
            
        # Return step scaled by step_size
        return self.step_size * direction
        
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list):
        """Calculate a single RK4 IRC step
        
        Parameters
        ----------
        mw_gradient : numpy.ndarray
            Mass-weighted gradient
        geom_num_list : numpy.ndarray
            Current geometry coordinates
        mw_BPA_hessian : numpy.ndarray
            Mass-weighted bias potential hessian
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
            
        Returns
        -------
        numpy.ndarray
            New geometry coordinates
        """
        # Update Hessian if we have previous points
        if len(self.irc_mw_gradients) > 1 and len(self.irc_mw_coords) > 1:
            delta_g = (self.irc_mw_gradients[-1] - self.irc_mw_gradients[-2]).reshape(-1, 1)
            delta_x = (self.irc_mw_coords[-1] - self.irc_mw_coords[-2]).reshape(-1, 1)
           
            # Only update if the step and gradient difference are meaningful
            if np.dot(delta_x.T, delta_g)[0, 0] > 1e-10:
                delta_hess = self.ModelHessianUpdate.BFGS_hessian_update(self.mw_hessian, delta_x, delta_g)
                self.mw_hessian += delta_hess
        
        # Add bias potential hessian to Hessian
        combined_hessian = self.mw_hessian + mw_BPA_hessian
        
        # Mass-weight the current coordinates
        mw_coords = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
        
        # Runge-Kutta 4th order integration
        # k1 calculation
        k1 = self.get_k(mw_coords, mw_gradient)
        
        # k2 calculation - we need to evaluate gradient at coords + 0.5*k1
        # This requires a new energy/gradient calculation at this point
        mw_coords_k2 = mw_coords + 0.5 * k1
        coords_k2 = self.unmass_weight_step(mw_coords_k2, sqrt_mass_list)
        
        e_k2, g_k2, _, _ = self.CE.single_point(
            self.final_directory, 
            self.element_list, 
            -1,  # Temporary calculation, no step number
            self.electric_charge_and_multiplicity, 
            self.xtb_method,  
            UnitValueLib().bohr2angstroms * coords_k2
        )
        
        # Calculate bias potential at k2 point
        CalcBiaspot = BiasPotentialCalculation(self.directory)
        _, _, B_g_k2, _ = CalcBiaspot.main(
            e_k2, g_k2, coords_k2, self.element_list, 
            self.force_data, g_k2, -1, coords_k2
        )
        
        # Mass-weight k2 gradient
        mw_B_g_k2 = self.mass_weight_gradient(B_g_k2, sqrt_mass_list)
        k2 = self.get_k(mw_coords_k2, mw_B_g_k2)
        
        # k3 calculation
        mw_coords_k3 = mw_coords + 0.5 * k2
        coords_k3 = self.unmass_weight_step(mw_coords_k3, sqrt_mass_list)
        
        e_k3, g_k3, _, _ = self.CE.single_point(
            self.final_directory, 
            self.element_list, 
            -1,  # Temporary calculation, no step number
            self.electric_charge_and_multiplicity, 
            self.xtb_method,  
            UnitValueLib().bohr2angstroms * coords_k3
        )
        
        # Calculate bias potential at k3 point
        _, _, B_g_k3, _ = CalcBiaspot.main(
            e_k3, g_k3, coords_k3, self.element_list, 
            self.force_data, g_k3, -1, coords_k3
        )
        
        # Mass-weight k3 gradient
        mw_B_g_k3 = self.mass_weight_gradient(B_g_k3, sqrt_mass_list)
        k3 = self.get_k(mw_coords_k3, mw_B_g_k3)
        
        # k4 calculation
        mw_coords_k4 = mw_coords + k3
        coords_k4 = self.unmass_weight_step(mw_coords_k4, sqrt_mass_list)
        
        e_k4, g_k4, _, _ = self.CE.single_point(
            self.final_directory, 
            self.element_list, 
            -1,  # Temporary calculation, no step number
            self.electric_charge_and_multiplicity, 
            self.xtb_method,  
            UnitValueLib().bohr2angstroms * coords_k4
        )
        
        # Calculate bias potential at k4 point
        _, _, B_g_k4, _ = CalcBiaspot.main(
            e_k4, g_k4, coords_k4, self.element_list, 
            self.force_data, g_k4, -1, coords_k4
        )
        
        # Mass-weight k4 gradient
        mw_B_g_k4 = self.mass_weight_gradient(B_g_k4, sqrt_mass_list)
        k4 = self.get_k(mw_coords_k4, mw_B_g_k4)
        
        # Calculate step using RK4 formula
        mw_step = (k1 + 2*k2 + 2*k3 + k4) / 6
        step = self.unmass_weight_step(mw_step, sqrt_mass_list)
        
        # Update geometry
        new_geom = geom_num_list + step
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
        
    def run(self):
        """Run the RK4 IRC calculation"""
        print("Runge-Kutta 4th Order method")
        geom_num_list = self.coords
        CalcBiaspot = BiasPotentialCalculation(self.directory)
        
        # Initialize oscillation counter
        oscillation_counter = 0
        
        for iter in range(1, self.max_step):
            print("# STEP: ", iter)
            exit_file_detect = os.path.exists(self.directory+"end.txt")

            if exit_file_detect:
                break
             
            # Calculate energy, gradient and new geometry
            e, g, geom_num_list, finish_frag = self.CE.single_point(
                self.final_directory, 
                self.element_list, 
                iter, 
                self.electric_charge_and_multiplicity, 
                self.xtb_method,  
                UnitValueLib().bohr2angstroms * geom_num_list
            )
            
            # Calculate bias potential
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(
                e, g, geom_num_list, self.element_list, 
                self.force_data, g, iter-1, geom_num_list
            )
            
            if finish_frag:
                break
            
            # Recalculate Hessian if needed
            if iter % self.FC_count == 0:
                self.mw_hessian = self.CE.Model_hess
                self.mw_hessian = Calculationtools().project_out_hess_tr_and_rot(
                    self.mw_hessian, self.element_list, geom_num_list
                )
            
            # Get mass arrays for consistent mass-weighting
            elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list = self.get_mass_array()
            
            # Mass-weight the hessian
            mw_BPA_hessian = self.mass_weight_hessian(BPA_hessian, three_sqrt_mass_list)
            
            # Mass-weight the coordinates
            mw_geom_num_list = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
            
            # Mass-weight the gradients
            mw_g = self.mass_weight_gradient(g, sqrt_mass_list)
            mw_B_g = self.mass_weight_gradient(B_g, sqrt_mass_list)

            # Save structure to XYZ file
            self.save_xyz_structure(iter, geom_num_list)
            
            # Save energy and gradient data to CSV
            self.save_to_csv(iter, e, B_e, g, B_g)

            # Store IRC data for calculation purposes (limit to keep only necessary data)
            # Keep only last 3 points for calculations like path bending angles and hessian updates
            if len(self.irc_energy_list) >= 3:
                self.irc_energy_list.pop(0)
                self.irc_bias_energy_list.pop(0)
                self.irc_mw_coords.pop(0)
                self.irc_mw_gradients.pop(0)
                self.irc_mw_bias_gradients.pop(0)
            
            self.irc_energy_list.append(e)
            self.irc_bias_energy_list.append(B_e)
            self.irc_mw_coords.append(mw_geom_num_list)
            self.irc_mw_gradients.append(mw_g)
            self.irc_mw_bias_gradients.append(mw_B_g)
            
            # Check for energy oscillations
            if self.check_energy_oscillation(self.irc_bias_energy_list):
                oscillation_counter += 1
                print(f"Energy oscillation detected ({oscillation_counter}/5)")
                
                if oscillation_counter >= 5:
                    print("Terminating IRC: Energy oscillated for 5 consecutive steps")
                    break
            else:
                # Reset counter if no oscillation is detected
                oscillation_counter = 0
                 
            if iter > 1:
                # Take RK4 step
                geom_num_list = self.step(
                    mw_B_g, geom_num_list, mw_BPA_hessian, sqrt_mass_list
                )
                
                # Calculate path bending angle
                if iter > 2:
                    bend_angle = Calculationtools().calc_multi_dim_vec_angle(
                        self.irc_mw_coords[0]-self.irc_mw_coords[1], 
                        self.irc_mw_coords[2]-self.irc_mw_coords[1]
                    )
                    self.path_bending_angle_list.append(np.degrees(bend_angle))
                    print("Path bending angle: ", np.degrees(bend_angle))

                # Check for convergence
                if convergence_check(B_g, self.MAX_FORCE_THRESHOLD, self.RMS_FORCE_THRESHOLD) and iter > 10:
                    print("Convergence reached. (IRC)")
                    break
                
            else:
                # First step is simple scaling along the gradient direction
                normalized_grad = mw_B_g / np.linalg.norm(mw_B_g.flatten())
                step = -normalized_grad * self.step_size * 0.05
                step = self.unmass_weight_step(step, sqrt_mass_list)
                
                geom_num_list = geom_num_list + step
                geom_num_list -= Calculationtools().calc_center_of_mass(geom_num_list, self.element_list)
            
            # Print current geometry
            print()
            for i in range(len(geom_num_list)):    
                x = geom_num_list[i][0] * UnitValueLib().bohr2angstroms
                y = geom_num_list[i][1] * UnitValueLib().bohr2angstroms
                z = geom_num_list[i][2] * UnitValueLib().bohr2angstroms
                print(f"{self.element_list[i]:<3} {x:17.12f} {y:17.12f} {z:17.12f}")
              
            # Display information
            print()
            print("Energy         : ", e)
            print("Bias Energy    : ", B_e)
            print("RMS B. grad    : ", np.sqrt((B_g**2).mean()))
            
            if len(self.irc_mw_coords) > 1:
                # Calculate curvature properties
                unit_tangent_vector, curvature_vector, scalar_curvature, curvature_coupling = calc_irc_curvature_properties(
                    mw_B_g, self.irc_mw_gradients[-2], self.mw_hessian, self.step_size
                )
                
                print("Scalar curvature: ", scalar_curvature)
                print("Curvature coupling: ", curvature_coupling.ravel())
                
                # Save curvature properties to file
                save_curvature_properties_to_file(
                    os.path.join(self.directory, "irc_curvature_properties.csv"),
                    scalar_curvature,
                    curvature_coupling
                )
                print()
        
        
        # Save final data visualization
        G = Graph(self.directory)
        rms_gradient_list = []
        with open(self.csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                rms_gradient_list.append(float(row[3]))
        
        G.single_plot(
            np.array(range(len(self.path_bending_angle_list))),
            np.array(self.path_bending_angle_list),
            self.directory,
            atom_num=0,
            axis_name_1="# STEP",
            axis_name_2="bending angle [degrees]",
            name="IRC_bending"
        )
        
        return
