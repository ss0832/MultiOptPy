import numpy as np
import os
import copy
import glob
import csv

from Optimizer.hessian_update import ModelHessianUpdate
from potential import BiasPotentialCalculation
from parameter import atomic_mass, UnitValueLib
from calc_tools import Calculationtools
from visualization import Graph

### I recommend to use LQA method to calculate IRC path ###

def convergence_check(grad, MAX_FORCE_THRESHOLD, RMS_FORCE_THRESHOLD):
    """Check convergence based on maximum and RMS force thresholds.
    
    Parameters
    ----------
    grad : numpy.ndarray
        Gradient vector
    MAX_FORCE_THRESHOLD : float
        Maximum force threshold for convergence
    RMS_FORCE_THRESHOLD : float
        RMS force threshold for convergence
        
    Returns
    -------
    bool
        True if converged, False otherwise
    """
    max_force = abs(grad.max())
    rms_force = abs(np.sqrt((grad**2).mean()))
    if max_force < MAX_FORCE_THRESHOLD and rms_force < RMS_FORCE_THRESHOLD:
        return True
    else:
        return False

def taylor(energy, gradient, hessian, step):
    """Taylor series expansion of the energy to second order.
    
    Parameters
    ----------
    energy : float
        Energy at expansion point
    gradient : numpy.ndarray
        Gradient at expansion point
    hessian : numpy.ndarray
        Hessian at expansion point
    step : numpy.ndarray
        Displacement vector from expansion point
        
    Returns
    -------
    float
        Expanded energy value
    """
    flat_step = step.flatten()
    flat_grad = gradient.flatten()
    return energy + flat_step @ flat_grad + 0.5 * flat_step @ hessian @ flat_step


def taylor_grad(gradient, hessian, step):
    """Gradient of a Taylor series expansion of the energy to second order.
    
    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient at expansion point
    hessian : numpy.ndarray
        Hessian at expansion point
    step : numpy.ndarray
        Displacement vector from expansion point
        
    Returns
    -------
    numpy.ndarray
        Expanded gradient vector
    """
    flat_step = step.flatten()
    result = gradient.flatten() + hessian @ flat_step
    return result.reshape(gradient.shape)


class RK4:
    """Runge-Kutta 4th order method for IRC calculations
    
    References
    ----------
    [1] J. Chem. Phys. 95, 9, 6758–6763 (1991)
    [2] Chem. Phys. Lett. 437, 1–3, 120-125 (2007)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None):
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

class LQA:
    """Local quadratic approximation method for IRC calculations
    
    References
    ----------
    [1] J. Chem. Phys. 93, 5634–5642 (1990)
    [2] J. Chem. Phys. 120, 9918–9924 (2004)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None):
        """Initialize LQA IRC calculator
        
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
        self.N_euler = 20000  # Number of Euler integration steps
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
        
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list):
        """Calculate a single LQA IRC step
        
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

        # Add bias potential hessian and diagonalize
        combined_hessian = self.mw_hessian + mw_BPA_hessian
        eigenvalues, eigenvectors = np.linalg.eigh(combined_hessian)
        
        # Drop small eigenvalues and corresponding eigenvectors
        small_eigvals = np.abs(eigenvalues) < 1e-8
        eigenvalues = eigenvalues[~small_eigvals]
        eigenvectors = eigenvectors[:,~small_eigvals]
        
        # Reshape gradient for matrix operations
        flattened_gradient = mw_gradient.flatten()
        
        # Time step for numerical integration
        dt = 1 / self.N_euler * self.step_size / np.linalg.norm(flattened_gradient)

        # Transform gradient to eigensystem of the hessian
        mw_gradient_proj = np.dot(eigenvectors.T, flattened_gradient)

        # Integration of the step size
        t = dt
        current_length = 0
        for j in range(self.N_euler):
            dsdt = np.sqrt(np.sum(mw_gradient_proj**2 * np.exp(-2*eigenvalues*t)))
            current_length += dsdt * dt
            if current_length > self.step_size:
                break
            t += dt
        
        # Calculate alphas and the IRC step
        alphas = (np.exp(-eigenvalues*t) - 1) / eigenvalues
        A = np.dot(eigenvectors, np.dot(np.diag(alphas), eigenvectors.T))
        step = np.dot(A, flattened_gradient)
        
        # Reshape and un-mass-weight the step
        step = step.reshape(len(geom_num_list), 3)
        step = self.unmass_weight_step(step, sqrt_mass_list)
        
        # Update geometry
        new_geom = geom_num_list + step
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
        
    def run(self):
        """Run the LQA IRC calculation"""
        print("Local Quadratic Approximation method")
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
                # Take LQA step
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

class DVV:
    """Damped Velocity Verlet method for IRC calculations
    
    This method uses a damped classical trajectory algorithm with dynamic time step control.
    
    References
    ----------
    [1] J. Phys. Chem. A, 106, 11, 2657-2667 (2002)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None, v0=0.04, dt0=0.5, error_tol=0.003):
        """Initialize DVV IRC calculator
        
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
            Target velocity magnitude (v0)
        init_coord : numpy.ndarray, optional
            Initial coordinates
        init_hess : numpy.ndarray, optional
            Initial hessian
        calc_engine : object, optional
            Calculator engine
        xtb_method : str, optional
            XTB method specification
        v0 : float, optional
            Target velocity magnitude (in atomic units), default=0.04
        dt0 : float, optional
            Initial time step (in fs), default=0.5
        error_tol : float, optional
            Error tolerance for time step adjustments, default=0.003
        """
        self.max_step = max_step
        self.step_size = step_size  # Not directly used in DVV, but kept for consistency
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
        # DVV specific parameters
        self.v0 = v0
        self.dt0 = dt0
        self.error_tol = error_tol
        
        # Initial condition
        self.coords = init_coord
        self.init_hess = init_hess
        self.mw_hessian = init_hess  # Mass-weighted hessian
        self.xtb_method = xtb_method
        
        # Convergence criteria
        self.MAX_FORCE_THRESHOLD = 0.0004
        self.RMS_FORCE_THRESHOLD = 0.0001

        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # DVV trajectory data
        self.velocities = []  # Store velocities
        self.accelerations = []  # Store accelerations
        self.time_steps = []  # Store time steps
        
        # IRC data storage
        self.irc_bias_energy_list = []
        self.irc_energy_list = []
        self.irc_mw_coords = []
        self.irc_mw_gradients = []
        self.irc_mw_bias_gradients = []
        self.path_bending_angle_list = []
        
        # Constants for unit conversion
        self.BOHR2M = 5.29177210903e-11  # Bohr to meters
        self.AU2J = 4.359744650e-18  # Hartree to Joules
        self.AMU2KG = 1.660539040e-27  # AMU to kg
        
        # Create data files
        self.create_csv_file()
        self.create_xyz_file()
    
    def create_csv_file(self):
        """Create CSV file for energy and gradient data"""
        self.csv_filename = os.path.join(self.directory, "irc_energies_gradients.csv")
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 
                             'RMS Gradient', 'RMS Bias Gradient', 'Time Step (fs)', 'Damping Factor'])
    
    def create_xyz_file(self):
        """Create XYZ file for structure data"""
        self.xyz_filename = os.path.join(self.directory, "irc_structures.xyz")
        # Create empty file (will be appended to later)
        open(self.xyz_filename, 'w').close()
    
    def save_to_csv(self, step, energy, bias_energy, gradient, bias_gradient, time_step=None, damping=None):
        """Save energy and gradient data to CSV file"""
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad, 
                             time_step if time_step is not None else '',
                             damping if damping is not None else ''])
    
    def save_xyz_structure(self, step, coords):
        """Save molecular structure to XYZ file"""
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
        """Apply mass-weighting to the hessian matrix"""
        mass_mat = np.diag(1.0 / three_sqrt_mass_list)
        return np.dot(mass_mat, np.dot(hessian, mass_mat))
    
    def mass_weight_coordinates(self, coordinates, sqrt_mass_list):
        """Convert coordinates to mass-weighted coordinates"""
        mw_coords = copy.deepcopy(coordinates)
        for i in range(len(coordinates)):
            mw_coords[i] = coordinates[i] * sqrt_mass_list[i]
        return mw_coords
    
    def mass_weight_gradient(self, gradient, sqrt_mass_list):
        """Convert gradient to mass-weighted form"""
        mw_gradient = copy.deepcopy(gradient)
        for i in range(len(gradient)):
            mw_gradient[i] = gradient[i] / sqrt_mass_list[i]
        return mw_gradient
    
    def unmass_weight_step(self, step, sqrt_mass_list):
        """Convert a step vector from mass-weighted to non-mass-weighted coordinates"""
        unmw_step = copy.deepcopy(step)
        for i in range(len(step)):
            unmw_step[i] = step[i] / sqrt_mass_list[i]
        return unmw_step
    
    def mw_grad_to_acc(self, mw_gradient):
        """Convert mass-weighted gradient to acceleration with proper units
        
        Converts units of mass-weighted gradient [Hartree/(Bohr*sqrt(amu))]
        to units of acceleration [sqrt(amu)*Bohr/fs²]
        
        Parameters
        ----------
        mw_gradient : numpy.ndarray
            Mass-weighted gradient
            
        Returns
        -------
        numpy.ndarray
            Acceleration with proper units
        """
        # The 1e30 comes from converting second² to femto second²
        return mw_gradient * self.AU2J / self.AMU2KG / self.BOHR2M**2 / 1e30
    
    def damp_velocity(self, velocity):
        """Apply damping to velocity to maintain consistent step size
        
        Parameters
        ----------
        velocity : numpy.ndarray
            Velocity vector
            
        Returns
        -------
        tuple
            (damped_velocity, damping_factor)
        """
        # Compute damping factor to maintain consistent magnitude
        v_norm = np.linalg.norm(velocity)
        if v_norm < 1e-10:
            damping_factor = 1.0
        else:
            damping_factor = self.v0 / v_norm
            
        print(f"Damping factor={damping_factor:.6f}")
        damped_velocity = velocity * damping_factor
        return damped_velocity, damping_factor
    
    def estimate_error(self, new_mw_coords, sqrt_mass_list):
        """Estimate error for time step adjustment
        
        Parameters
        ----------
        new_mw_coords : numpy.ndarray
            New mass-weighted coordinates
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
            
        Returns
        -------
        float
            Estimated error for time step adjustment
        """
        print("Error estimation")
        
        if len(self.irc_mw_coords) < 2 or len(self.time_steps) < 2 or len(self.velocities) < 2:
            # Not enough history for error estimation
            return self.error_tol
        
        # Get current and previous time steps
        cur_time_step = self.time_steps[-1]
        prev_time_step = self.time_steps[-2]
        time_step_sum = prev_time_step + cur_time_step
        print(f"\tSum of current and previous timestep: {time_step_sum:.6f} fs")
        
        # Calculate reference coordinates for error estimation (x' in the paper)
        ref_coords = (
            self.irc_mw_coords[-2]  # x_i-2 in the paper
            + np.dot(self.velocities[-2].reshape(-1), time_step_sum).reshape(new_mw_coords.shape)  # v_i-2 * dt_sum
            + 0.5 * np.dot(self.accelerations[-2].reshape(-1), time_step_sum**2).reshape(new_mw_coords.shape)  # 0.5 * a_i-2 * dt_sum^2
        )
        
        # Calculate difference and scale by mass
        diff = new_mw_coords - ref_coords
        for i in range(len(diff)):
            diff[i] = diff[i] / sqrt_mass_list[i]
        
        # Calculate error metrics
        largest_component = np.max(np.abs(diff))
        norm = np.linalg.norm(diff)
        
        print(f"\tmax(|diff|)={largest_component:.6f}")
        print(f"\tnorm(diff)={norm:.6f}")
        
        # Use the larger of the two metrics as the estimated error
        estimated_error = max(largest_component, norm)
        print(f"\testimated error={estimated_error:.6f}")
        
        return estimated_error
    
    def check_energy_oscillation(self, energy_list):
        """Check if energy is oscillating (going up and down)"""
        if len(energy_list) < 3:
            return False
        
        # Check if the energy changes direction (from increasing to decreasing or vice versa)
        last_diff = energy_list[-1] - energy_list[-2]
        prev_diff = energy_list[-2] - energy_list[-3]
        
        # Return True if the energy direction has changed
        return (last_diff * prev_diff) < 0
    
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list):
        """Calculate a single DVV IRC step
        
        Parameters
        ----------
        mw_gradient : numpy.ndarray
            Mass-weighted gradient
        geom_num_list : numpy.ndarray
            Current geometry coordinates
        mw_BPA_hessian : numpy.ndarray
            Mass-weighted bias potential hessian (not used in DVV)
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
            
        Returns
        -------
        numpy.ndarray
            New geometry coordinates
        """
        # First-time setup if needed
        if not self.velocities:
            # Initialize with current time step
            self.time_steps.append(self.dt0)
            
            # Convert gradient to acceleration (negative because gradient points uphill)
            acceleration = -self.mw_grad_to_acc(mw_gradient)
            
            # Initialize velocity with damped acceleration
            initial_velocity, _ = self.damp_velocity(acceleration)
            
            self.velocities.append(initial_velocity)
            self.accelerations.append(acceleration)
            
            # Convert mw_coords to mass weighted
            mw_coords = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
            
            # For first step, don't update coords yet
            return geom_num_list
        
        # Get previous values
        prev_time_step = self.time_steps[-1]
        prev_acceleration = self.accelerations[-1]
        prev_velocity = self.velocities[-1]
        
        # Convert mw_coords to mass weighted
        mw_coords = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
        
        # Calculate new acceleration from current gradient
        acceleration = -self.mw_grad_to_acc(mw_gradient)
        
        # Check for velocity-acceleration alignment (for logging)
        acc_normed = acceleration / np.linalg.norm(acceleration)
        prev_vel_normed = prev_velocity / np.linalg.norm(prev_velocity)
        ovlp = np.sum(acc_normed * prev_vel_normed)
        print(f"a @ v_i-1={ovlp:.8f}")
        
        # Store acceleration
        self.accelerations.append(acceleration)
        
        # Update coordinates using Velocity Verlet algorithm (Eq. 2 in the paper)
        new_mw_coords = (mw_coords
                      + prev_velocity * prev_time_step
                      + 0.5 * prev_acceleration * prev_time_step**2)
        
        # Update velocity (Eq. 3 in the paper)
        velocity = prev_velocity + 0.5 * (prev_acceleration + acceleration) * prev_time_step
        
        # Damp velocity to maintain consistent step size
        damped_velocity, damping_factor = self.damp_velocity(velocity)
        self.velocities.append(damped_velocity)
        
        # Estimate error for time step adjustment
        estimated_error = self.estimate_error(new_mw_coords, sqrt_mass_list)
        
        # Adjust time step based on error (Eq. 6 in the paper)
        new_time_step = prev_time_step * (self.error_tol / estimated_error)**(1/3)
        
        # Constrain time step between 0.0025 fs and 3.0 fs
        new_time_step = min(new_time_step, 3.0)
        new_time_step = max(new_time_step, 0.025)
        
        print(f"\tCurrent time step={prev_time_step:.6f} fs")
        print(f"\tNext time step={new_time_step:.6f} fs")
        
        # Store the new time step
        self.time_steps.append(new_time_step)
        
        # Convert back to non-mass-weighted
        new_geom = self.unmass_weight_step(new_mw_coords, sqrt_mass_list)
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
        
    def run(self):
        """Run the DVV IRC calculation"""
        print("Damped Velocity Verlet method")
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
            current_dt = self.time_steps[-1] if self.time_steps else self.dt0
            current_damping = damping_factor if 'damping_factor' in locals() else None
            self.save_to_csv(iter, e, B_e, g, B_g, current_dt, current_damping)

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
                 
            # Take DVV step
            geom_num_list = self.step(mw_B_g, geom_num_list, mw_BPA_hessian, sqrt_mass_list)
            
            # Calculate path bending angle if we have enough points
            if iter > 2 and len(self.irc_mw_coords) >= 3:
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
            if self.time_steps:
                print(f"Time step      : {self.time_steps[-1]:.6f} fs")
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
        
        # Also plot the time step evolution
        G.single_plot(
            np.array(range(len(self.time_steps))),
            np.array(self.time_steps),
            self.directory,
            atom_num=0,
            axis_name_1="# STEP",
            axis_name_2="Time step (fs)",
            name="DVV_timesteps"
        )
        
        return


class IRC:
    """Main class for Intrinsic Reaction Coordinate calculations
    
    This class handles saddle point verification, forward/backward IRC calculations
    """
    
    def __init__(self, directory, final_directory, irc_method, QM_interface, element_list, 
                 electric_charge_and_multiplicity, force_data, xtb_method, FC_count=-1, hessian=None):
        """Initialize IRC calculator
        
        Parameters
        ----------
        directory : str
            Working directory
        final_directory : str
            Directory for final output
        irc_method : list
            [step_size, max_step, method_name]
        QM_interface : object
            Interface to quantum mechanical calculator
        element_list : list
            List of atomic elements
        electric_charge_and_multiplicity : tuple
            Charge and multiplicity for the system
        force_data : dict
            Force field data for bias potential
        xtb_method : str
            XTB method specification
        FC_count : int, optional
            Frequency of full hessian recalculation, default=-1
        hessian : numpy.ndarray, optional
            Initial hessian matrix, default=None
        """
        if hessian is None:
            self.hessian_flag = False
        else:
            self.hessian_flag = True
            self.hessian = hessian
        
        self.step_size = float(irc_method[0])
        self.max_step = int(irc_method[1])
        self.method = str(irc_method[2])
            
        self.file_directory = directory
        self.final_directory = final_directory
        self.QM_interface = QM_interface
        
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.xtb_method = xtb_method
        
        self.force_data = force_data
        self.FC_count = FC_count
        
        # Will be set in saddle_check
        self.IRC_flag = False
        self.initial_step = None
        self.geom_num_list = None
        self.ts_coords = None
    
    def saddle_check(self):
        """Check if starting point is a saddle point and calculate initial displacement
        
        Returns
        -------
        numpy.ndarray
            Initial displacement vector
        bool
            True if valid IRC starting point (has 1 imaginary mode)
        numpy.ndarray
            Current geometry coordinates
        bool
            True if calculation failed
        """
        # Setup hessian calculation
        if not self.hessian_flag:
            self.QM_interface.hessian_flag = True
            iter = 1
        else:
            self.QM_interface.FC_COUNT = -1
            iter = 1
        
        # Read input geometry
        fin_xyz = glob.glob(self.final_directory+"/*.xyz")
        with open(fin_xyz[0], "r") as f:
            words = f.read().splitlines()
        geom_num_list = []
        for i in range(len(words)):
            if len(words[i].split()) > 3:
                geom_num_list.append([
                    float(words[i].split()[1]), 
                    float(words[i].split()[2]), 
                    float(words[i].split()[3])
                ])
        geom_num_list = np.array(geom_num_list)
        # Calculate energy, gradient and hessian
        init_e, init_g, geom_num_list, finish_frag = self.QM_interface.single_point(
            self.final_directory, 
            self.element_list, 
            iter, 
            self.electric_charge_and_multiplicity, 
            self.xtb_method, 
            geom_num_list
        )
        
        # Reset QM interface settings
        self.QM_interface.hessian_flag = False
        self.QM_interface.FC_COUNT = self.FC_count
        
        if finish_frag:
            return 0, 0, 0, finish_frag
        
        # Get hessian from QM calculation
        self.hessian = self.QM_interface.Model_hess
        self.QM_interface.hessian_flag = False
        
        # Calculate bias potential
        CalcBiaspot = BiasPotentialCalculation(self.final_directory)
        _, init_B_e, init_B_g, BPA_hessian = CalcBiaspot.main(
            init_e, init_g, geom_num_list, self.element_list, 
            self.force_data, init_g, 0, geom_num_list
        )
        
        # Add bias potential hessian
        self.hessian += BPA_hessian
        
        # Project out translational and rotational modes
        self.hessian = Calculationtools().project_out_hess_tr_and_rot(
            self.hessian, self.element_list, geom_num_list
        )
       
        # Store initial state
        self.init_e = init_e
        self.init_g = init_g
        self.init_B_e = init_B_e
        self.init_B_g = init_B_g
        
        # Find imaginary modes (negative eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(self.hessian)
        neg_indices = np.where(eigenvalues < -1e-8)[0]
        imaginary_count = len(neg_indices)
        print("Number of imaginary eigenvalues: ", imaginary_count)
        
        # Determine initial step direction
        if imaginary_count == 1:
            print("Execute IRC")
            # True IRC: Use transition vector (imaginary mode)
            imaginary_idx = neg_indices[0]
            transition_vector = eigenvectors[:, imaginary_idx].reshape(len(geom_num_list), 3)
            initial_step = transition_vector / np.linalg.norm(transition_vector.flatten()) * self.step_size * 0.1
            IRC_flag = True
        else:
            print("Execute meta-IRC")
            # Meta-IRC: Use gradient direction
            gradient = self.QM_interface.gradient.reshape(len(geom_num_list), 3)
            initial_step = gradient / np.linalg.norm(gradient.flatten()) * self.step_size * 0.1
            
            # Mass-weight the initial step for meta-IRC
            sqrt_mass_list = np.sqrt([atomic_mass(elem) for elem in self.element_list])
            for i in range(len(initial_step)):
                initial_step[i] /= sqrt_mass_list[i]
                
            IRC_flag = False
            
        return initial_step, IRC_flag, geom_num_list, finish_frag
    
    def calc_IRCpath(self):
        """Calculate IRC path in forward and/or backward directions"""
        print("IRC carry out...")
        if self.method.upper() == "LQA":
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = LQA(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    fwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Backward direction (from TS to reactants)
                print("Backward IRC")
                init_geom = self.geom_num_list - self.initial_step
                
                # Create backward direction directory
                bwd_dir = os.path.join(self.file_directory, "irc_backward")
                os.makedirs(bwd_dir, exist_ok=True)
                
                IRCmethod = LQA(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    bwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Combine forward and backward CSV data into a single file
                self.combine_csv_data(fwd_dir, bwd_dir)
                
            else:
                # Meta-IRC (single direction)
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = LQA(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    self.file_directory, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()

        elif self.method.upper() == "RK4":
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = RK4(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    fwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Backward direction (from TS to reactants)
                print("Backward IRC")
                init_geom = self.geom_num_list - self.initial_step
                
                # Create backward direction directory
                bwd_dir = os.path.join(self.file_directory, "irc_backward")
                os.makedirs(bwd_dir, exist_ok=True)
                
                IRCmethod = RK4(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    bwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Combine forward and backward CSV data into a single file
                self.combine_csv_data(fwd_dir, bwd_dir)
                
            else:
                # Meta-IRC (single direction)
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = RK4(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    self.file_directory, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                

        elif self.method.upper() == "DVV":
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = DVV(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    fwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Backward direction (from TS to reactants)
                print("Backward IRC")
                init_geom = self.geom_num_list - self.initial_step
                
                # Create backward direction directory
                bwd_dir = os.path.join(self.file_directory, "irc_backward")
                os.makedirs(bwd_dir, exist_ok=True)
                
                IRCmethod = DVV(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    bwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Combine forward and backward CSV data into a single file
                self.combine_csv_data(fwd_dir, bwd_dir)
                
            else:
                # Meta-IRC (single direction)
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = DVV(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    self.file_directory, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()

        else:
            # Default to LQA method if method is not recognized
            print("Unexpected method. (default method is LQA.)")
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = LQA(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    fwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Backward direction (from TS to reactants)
                print("Backward IRC")
                init_geom = self.geom_num_list - self.initial_step
                
                # Create backward direction directory
                bwd_dir = os.path.join(self.file_directory, "irc_backward")
                os.makedirs(bwd_dir, exist_ok=True)
                
                IRCmethod = LQA(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    bwd_dir, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
                
                # Combine forward and backward CSV data into a single file
                self.combine_csv_data(fwd_dir, bwd_dir)
                
            else:
                # Meta-IRC (single direction)
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = LQA(
                    self.element_list, 
                    self.electric_charge_and_multiplicity, 
                    self.FC_count, 
                    self.file_directory, 
                    self.final_directory, 
                    self.force_data, 
                    max_step=self.max_step, 
                    step_size=self.step_size, 
                    init_coord=init_geom, 
                    init_hess=self.hessian, 
                    calc_engine=self.QM_interface,
                    xtb_method=self.xtb_method
                )
                IRCmethod.run()
        
        return
    
    def combine_csv_data(self, fwd_dir, bwd_dir):
        """Combine forward and backward CSV data into a single file
        
        Parameters
        ----------
        fwd_dir : str
            Forward directory path
        bwd_dir : str
            Backward directory path
        """
        # Final CSV file
        combined_csv = os.path.join(self.file_directory, "irc_combined_data.csv")
        
        # Read backward data (to be reversed)
        bwd_data = []
        with open(os.path.join(bwd_dir, "irc_energies_gradients.csv"), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                bwd_data.append(row)
        
        # Read forward data
        fwd_data = []
        with open(os.path.join(fwd_dir, "irc_energies_gradients.csv"), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                fwd_data.append(row)
        
        # Prepare TS point data
        ts_data = [0, self.init_e, self.init_B_e, np.sqrt((self.init_g**2).mean()), np.sqrt((self.init_B_g**2).mean())]
        
        # Write combined data
        with open(combined_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 'RMS Gradient', 'RMS Bias Gradient'])
            
            # Write reversed backward data with negative step numbers
            for i, row in enumerate(reversed(bwd_data)):
                step = -int(row[0])
                writer.writerow([step, row[1], row[2], row[3], row[4]])
            
            # Write TS point (step 0)
            writer.writerow(ts_data)
            
            # Write forward data with positive step numbers
            for row in fwd_data:
                writer.writerow(row)
        
        print(f"Combined IRC data saved to {combined_csv}")
    
    def run(self):
        """Main function to run IRC calculation"""
        # Check if starting point is a saddle point and get initial displacement
        self.initial_step, self.IRC_flag, self.geom_num_list, finish_flag = self.saddle_check()
        
        if finish_flag:
            print("IRC calculation failed.")
            return
            
        # Calculate the IRC path
        self.calc_IRCpath()
        
        print("IRC calculation is finished.")
        return