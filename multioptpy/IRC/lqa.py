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


class LQA:
    """Local quadratic approximation method for IRC calculations
    
    References
    ----------
    [1] J. Chem. Phys. 93, 5634–5642 (1990)
    [2] J. Chem. Phys. 120, 9918–9924 (2004)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None, **kwargs):
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
        self.N_euler = 40000  # Number of Euler integration steps
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

            delta_hess = self.ModelHessianUpdate.Bofill_hessian_update(self.mw_hessian, delta_x, delta_g)
            self.mw_hessian += delta_hess

            # Parallel transport correction for the Hessian.
            #
            # As the IRC path curves through mass-weighted Cartesian space, the local
            # coordinate frame rotates from step to step. The Bofill update is expressed
            # in the fixed Cartesian frame and therefore accumulates a frame-rotation
            # error proportional to the path curvature. The parallel transport
            # (covariant derivative) correction removes this error to first order in κ:
            #
            #   ΔH_pt = -κ * ( H·n ⊗ t  +  t ⊗ H·n )
            #
            # where t is the unit tangent vector along the IRC path, n is the unit
            # curvature (principal normal) vector, and κ is the scalar curvature.
            # This is the discrete analogue of Lie-transporting the Hessian tensor
            # along the path and ensures that successive Bofill updates are consistent
            # with the same local frame, reducing drift between full Hessian recalculations.
            unit_tangent_vector, curvature_vector, scalar_curvature, _ = calc_irc_curvature_properties(
                self.irc_mw_gradients[-1], self.irc_mw_gradients[-2], self.mw_hessian, self.step_size
            )
            t_vec = unit_tangent_vector.flatten()
            n_vec = curvature_vector.flatten()
            Hn = np.dot(self.mw_hessian, n_vec)
            delta_hess_pt = -scalar_curvature * (np.outer(Hn, t_vec) + np.outer(t_vec, Hn))
            self.mw_hessian += delta_hess_pt
            print(f"  [LQA] Parallel transport correction applied: curvature={scalar_curvature:.4e}")

        # Add bias potential hessian and diagonalize
        combined_hessian = self.mw_hessian + mw_BPA_hessian
        eigenvalues, eigenvectors = np.linalg.eigh(combined_hessian)
        
        # Drop small eigenvalues and corresponding eigenvectors
        small_eigvals = np.abs(eigenvalues) < 1e-8
        eigenvalues = eigenvalues[~small_eigvals]
        eigenvectors = eigenvectors[:,~small_eigvals]
        
        # Reshape gradient for matrix operations
        flattened_gradient = mw_gradient.flatten()
        
        # --- START MODIFICATION (Fix for numerical stability) ---

        # Time step for numerical integration
        # Original: dt = 1 / self.N_euler * self.step_size / np.linalg.norm(flattened_gradient)
        # This can diverge if np.linalg.norm(flattened_gradient) -> 0
        
        epsilon = 1e-6  # Prevent divergence when gradient norm is near zero
        norm_g = np.linalg.norm(flattened_gradient)
        dt = 1 / self.N_euler * self.step_size / max(norm_g, epsilon)

        # --- END MODIFICATION ---

        # Transform gradient to eigensystem of the hessian
        mw_gradient_proj = np.dot(eigenvectors.T, flattened_gradient)

        # Integration of the step size
        # Use a for/else to detect if the arc length saturated before reaching step_size.
        # Saturation occurs near the EQ where the gradient norm is very small and
        # exp(-2*lambda*t) decays rapidly, causing the integrand to vanish before
        # the target arc length is reached. In this case the full N_euler loop
        # completes without breaking, so the else clause fires.
        t = dt
        current_length = 0
        saturated = False
        for j in range(self.N_euler):
            dsdt = np.sqrt(np.sum(mw_gradient_proj**2 * np.exp(-2*eigenvalues*t)))
            current_length += dsdt * dt
            if current_length > self.step_size:
                break
            t += dt
        else:
            # Loop completed without reaching step_size: arc length has saturated.
            # This indicates we are near the EQ. Scale the step proportionally to
            # avoid overshooting and oscillating around the minimum.
            saturated = True

        if saturated:
            # Near the EQ the arc length integrand decays to zero before reaching
            # step_size, meaning the distance to the EQ is smaller than one full step.
            # The LQA formula has a well-defined t -> infinity limit:
            #   alpha_i = (exp(-lambda_i * t) - 1) / lambda_i  ->  -1 / lambda_i
            # which is exactly the Newton step in the eigenbasis of the Hessian.
            # Using this limit directly is both numerically clean and physically correct:
            # on a local quadratic surface the Newton step lands at the minimum in one step.
            alphas = -1.0 / eigenvalues
            print(f"  [LQA] Arc length saturated at {current_length:.4e}: "
                  f"using Newton step (t->inf limit of LQA)")
        else:
            # Normal LQA: compute alphas from the integration time t.
            # Use np.expm1 to avoid catastrophic cancellation when eigenvalues*t ~ 0:
            #   (exp(-lambda*t) - 1) / lambda = expm1(-lambda*t) / lambda
            # Limit as lambda*t -> 0: alpha -> -t
            x = -eigenvalues * t
            small_x_mask = np.abs(x) < 1e-8
            alphas = np.where(
                small_x_mask,
                -t,                         # Limiting value as x -> 0
                np.expm1(x) / eigenvalues   # Numerically stable for finite x
            )
        
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

            # FIX (Bug 1): Move get_mass_array() before the Hessian recalculation block.
            # In the original code, get_mass_array() was called after this block, so
            # three_sqrt_mass_list was undefined when mass_weight_hessian() needed it.
            elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list = self.get_mass_array()

            # Recalculate Hessian if needed
            if iter % self.FC_count == 0 and iter > 0:
                # FIX (Bug 2): self.CE.Model_hess is a Cartesian-coordinate Hessian.
                # Assigning it directly to mw_hessian causes a unit mismatch that
                # corrupts all subsequent IRC steps. Apply mass_weight_hessian() first.
                self.mw_hessian = self.mass_weight_hessian(
                    self.CE.Model_hess, three_sqrt_mass_list
                )
                self.mw_hessian = Calculationtools().project_out_hess_tr_and_rot(
                    self.mw_hessian, self.element_list, geom_num_list
                )
            
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
                print(f"Energy oscillation detected ({oscillation_counter}/10)")
                if oscillation_counter >= 10:
                    print("Terminating IRC: Energy oscillated for 10 consecutive steps")
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
            
            """
            if len(self.irc_mw_coords) > 1:
                # Calculate curvature properties
                unit_tangent_vector, curvature_vector, scalar_curvature, curvature_coupling = calc_irc_curvature_properties(
                    mw_B_g, self.irc_mw_gradients[-2], self.mw_hessian, self.step_size
                )
                
                print("Scalar curvature: ", scalar_curvature)
                # Print curvature_coupling as 6 columns, 8 decimal places
                flat_cc = curvature_coupling.ravel()
                print("Curvature coupling:")
                for i in range(0, len(flat_cc), 6):
                    print(" ".join(f"{x: .8f}" for x in flat_cc[i:i+6]))
                
                # Save curvature properties to file
                save_curvature_properties_to_file(
                    os.path.join(self.directory, "irc_curvature_properties.csv"),
                    scalar_curvature,
                    curvature_coupling
                )
                print()
            """
            
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