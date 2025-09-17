import numpy as np
import os
import copy
import csv

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass
from multioptpy.Visualization.visualization import Graph
from multioptpy.IRC.converge_criteria import convergence_check


class DVV:
    """Damped Velocity Verlet method for IRC calculations
    
    This method uses a damped classical trajectory algorithm with dynamic time step control.
    
    References
    ----------
    [1] J. Phys. Chem. A, 106, 11, 2657-2667 (2002)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None, v0=0.04, dt0=0.5, error_tol=0.003, **kwargs):
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

