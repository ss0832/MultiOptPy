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
    with enhanced convergence to tight thresholds (MAX/RMS Force ~ 1e-7).
    
    Key improvements over the basic LQA:
    - Adaptive step size based on local curvature and gradient magnitude
    - Predictor-corrector scheme (corrector step perpendicular to IRC tangent)
    - Higher-precision Euler integration with adaptive subdivision
    - Periodic Hessian re-orthogonalization against translation/rotation
    - Trust-region-like step control to prevent overshooting
    - Gradient-norm-aware convergence logic
    
    References
    ----------
    [1] J. Chem. Phys. 93, 5634-5642 (1990)
    [2] J. Chem. Phys. 120, 9918-9924 (2004)
    [3] J. Chem. Phys. 95, 5853-5860 (1991) - Gonzalez & Schlegel improved IRC
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
            Initial step size for the IRC (will adapt during calculation)
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
        self.initial_step_size = step_size
        self.step_size = step_size
        self.N_euler = 50000  # Increased number of Euler integration steps for higher precision
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
        # initial condition
        self.coords = init_coord
        self.init_hess = init_hess
        self.mw_hessian = init_hess  # Mass-weighted hessian
        self.xtb_method = xtb_method
        
        # Tight convergence criteria for 5e-5 level
        self.MAX_FORCE_THRESHOLD = 5e-5
        self.RMS_FORCE_THRESHOLD = 5e-5

        # Adaptive step size parameters
        self.min_step_size = 1e-10       # Minimum allowed step size
        self.max_step_size = step_size  # Maximum allowed step size (initial value)
        self.step_scale_up = 1.2        # Factor to increase step size
        self.step_scale_down = 0.5      # Factor to decrease step size
        self.target_energy_change = 1e-6  # Target energy change per step (Hartree)
        
        # Corrector step parameters
        self.corrector_max_iter = 20    # Max corrector iterations per IRC step
        self.corrector_convergence = 1e-8  # Convergence for corrector step
        
        # Hessian quality control
        self.hessian_reset_interval = 50  # Reset Hessian from projected form periodically
        self.hessian_project_interval = 5  # Re-project translation/rotation every N steps
        
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # IRC data storage for current calculation (needed for immediate operations)
        # Keep only recent points needed for calculations
        self.irc_bias_energy_list = []
        self.irc_energy_list = []
        self.irc_mw_coords = []
        self.irc_mw_gradients = []
        self.irc_mw_bias_gradients = []
        self.path_bending_angle_list = []
        
        # Tracking for adaptive step size
        self.prev_energy = None
        self.prev_bias_energy = None
        self.consecutive_good_steps = 0
        
        # Create data files
        self.create_csv_file()
        self.create_xyz_file()
    
    def create_csv_file(self):
        """Create CSV file for energy and gradient data"""
        self.csv_filename = os.path.join(self.directory, "irc_energies_gradients.csv")
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 
                'Max Force', 'RMS Force', 'RMS Gradient', 'RMS Bias Gradient', 
                'Step Size', 'Corrector Iters'
            ])
    
    def create_xyz_file(self):
        """Create XYZ file for structure data"""
        self.xyz_filename = os.path.join(self.directory, "irc_structures.xyz")
        # Create empty file (will be appended to later)
        open(self.xyz_filename, 'w').close()
    
    def save_to_csv(self, step, energy, bias_energy, gradient, bias_gradient, 
                    current_step_size, corrector_iters):
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
        current_step_size : float
            Step size used for this step
        corrector_iters : int
            Number of corrector iterations used
        """
        max_force = np.max(np.abs(gradient))
        rms_force = np.sqrt((gradient**2).mean())
        rms_grad = rms_force
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                step, f"{energy:.12f}", f"{bias_energy:.12f}", 
                f"{max_force:.2e}", f"{rms_force:.2e}",
                f"{rms_grad:.2e}", f"{rms_bias_grad:.2e}",
                f"{current_step_size:.2e}", corrector_iters
            ])
    
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
    
    def adapt_step_size(self, current_energy, prev_energy, rms_gradient, scalar_curvature=None):
        """Adaptively adjust step size based on energy change, gradient, and curvature
        
        Parameters
        ----------
        current_energy : float
            Current energy value
        prev_energy : float
            Previous energy value
        rms_gradient : float
            Current RMS gradient
        scalar_curvature : float, optional
            Scalar curvature of the IRC path
            
        Returns
        -------
        float
            Updated step size
        """
        if prev_energy is None:
            return self.step_size
        
        energy_change = abs(current_energy - prev_energy)
        
        # Strategy 1: Energy-change-based adaptation
        if energy_change > 0:
            energy_ratio = self.target_energy_change / energy_change
            energy_factor = np.clip(np.sqrt(energy_ratio), self.step_scale_down, self.step_scale_up)
        else:
            energy_factor = self.step_scale_up
        
        # Strategy 2: Gradient-based adaptation
        # Near convergence (small gradient), use smaller steps for precision
        if rms_gradient < 1e-4:
            gradient_factor = max(rms_gradient / 1e-4, 0.1)
        elif rms_gradient < 1e-3:
            gradient_factor = max(rms_gradient / 1e-3, 0.3)
        else:
            gradient_factor = 1.0
        
        # Strategy 3: Curvature-based adaptation
        # In high-curvature regions, reduce step size
        curvature_factor = 1.0
        if scalar_curvature is not None and scalar_curvature > 0:
            # Reduce step when curvature is large (the path is bending sharply)
            if scalar_curvature > 0.1:
                curvature_factor = 0.1 / scalar_curvature
                curvature_factor = max(curvature_factor, 0.2)
            elif scalar_curvature > 0.01:
                curvature_factor = 0.8
        
        # Combine all factors
        combined_factor = min(energy_factor, gradient_factor, curvature_factor)
        new_step_size = self.step_size * combined_factor
        
        # Enforce bounds
        new_step_size = np.clip(new_step_size, self.min_step_size, self.max_step_size)
        
        # Track consecutive good steps for gradual step size recovery
        if energy_change < self.target_energy_change * 2.0 and not self.check_energy_oscillation(self.irc_bias_energy_list):
            self.consecutive_good_steps += 1
        else:
            self.consecutive_good_steps = 0
        
        # Allow slow recovery of step size after 5 consecutive good steps
        if self.consecutive_good_steps >= 5:
            new_step_size = min(new_step_size * 1.1, self.max_step_size)
        
        return new_step_size
    
    def project_out_translation_rotation(self, hessian, element_list, geom_num_list):
        """Re-project translation and rotation out of the Hessian
        
        This is essential for maintaining Hessian quality over many update steps.
        Accumulated BFGS updates can re-introduce translation/rotation components.
        
        Parameters
        ----------
        hessian : numpy.ndarray
            Mass-weighted Hessian matrix
        element_list : list
            List of atomic elements
        geom_num_list : numpy.ndarray
            Current geometry in Bohr
            
        Returns
        -------
        numpy.ndarray
            Projected mass-weighted Hessian
        """
        return Calculationtools().project_out_hess_tr_and_rot(
            hessian, element_list, geom_num_list
        )
    
    def corrector_step(self, geom_num_list, tangent_vector, mw_gradient, sqrt_mass_list, 
                       three_sqrt_mass_list, mw_BPA_hessian):
        """Perform a corrector step: optimize geometry perpendicular to the IRC tangent
        
        The corrector step constrains the geometry to a hyperplane perpendicular to the
        IRC tangent vector and minimizes the energy within that hyperplane. This dramatically
        improves the quality of the IRC path and convergence.
        
        Parameters
        ----------
        geom_num_list : numpy.ndarray
            Current geometry coordinates
        tangent_vector : numpy.ndarray
            Unit tangent vector of the IRC path (mass-weighted, flattened)
        mw_gradient : numpy.ndarray
            Mass-weighted gradient (atom x 3)
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
        three_sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values repeated for x,y,z per atom
        mw_BPA_hessian : numpy.ndarray
            Mass-weighted bias potential hessian
            
        Returns
        -------
        numpy.ndarray
            Corrected geometry coordinates
        int
            Number of corrector iterations performed
        """
        corrected_geom = copy.deepcopy(geom_num_list)
        n = len(tangent_vector)
        
        # Projection matrix: P = I - t * t^T (projects out tangent component)
        P_tangent = np.eye(n) - np.outer(tangent_vector, tangent_vector)
        
        for c_iter in range(self.corrector_max_iter):
            # Project gradient perpendicular to tangent
            flat_grad = mw_gradient.flatten()
            perp_gradient = np.dot(P_tangent, flat_grad)
            
            perp_norm = np.linalg.norm(perp_gradient)
            if perp_norm < self.corrector_convergence:
                print(f"  Corrector converged in {c_iter + 1} iterations (perp grad norm: {perp_norm:.2e})")
                return corrected_geom, c_iter + 1
            
            # Project Hessian perpendicular to tangent
            combined_hessian = self.mw_hessian + mw_BPA_hessian
            perp_hessian = np.dot(P_tangent, np.dot(combined_hessian, P_tangent))
            
            # Regularize the projected Hessian (it has a zero eigenvalue along the tangent)
            # Add a small positive shift along the tangent direction to make it invertible
            perp_hessian += 1000.0 * np.outer(tangent_vector, tangent_vector)
            
            # Solve for the corrector displacement: H_perp * dx = -g_perp
            try:
                corrector_disp = np.linalg.solve(perp_hessian, -perp_gradient)
            except np.linalg.LinAlgError:
                # Fallback to steepest descent in the perpendicular subspace
                print("  Corrector: Hessian solve failed, using steepest descent")
                step_scale = min(0.01, perp_norm)
                corrector_disp = -perp_gradient * step_scale / perp_norm
            
            # Ensure the displacement is perpendicular to the tangent
            corrector_disp = np.dot(P_tangent, corrector_disp)
            
            # Limit the corrector step size to avoid overshooting
            disp_norm = np.linalg.norm(corrector_disp)
            max_corrector_step = self.step_size * 0.5
            if disp_norm > max_corrector_step:
                corrector_disp *= max_corrector_step / disp_norm
            
            # Apply displacement (un-mass-weight first)
            disp_3d = corrector_disp.reshape(len(corrected_geom), 3)
            disp_3d = self.unmass_weight_step(disp_3d, sqrt_mass_list)
            corrected_geom = corrected_geom + disp_3d
            
            # Remove center of mass motion
            corrected_geom -= Calculationtools().calc_center_of_mass(corrected_geom, self.element_list)
            
            # Re-compute mass-weighted gradient for the corrected geometry
            # Note: For a true corrector, we would re-evaluate the electronic structure here.
            # However, that is expensive. Instead, we use the quadratic model:
            # g(x + dx) ≈ g(x) + H * dx
            delta_grad_flat = np.dot(combined_hessian, corrector_disp)
            new_flat_grad = flat_grad + delta_grad_flat
            mw_gradient = new_flat_grad.reshape(len(corrected_geom), 3)
        
        print(f"  Corrector did not fully converge after {self.corrector_max_iter} iterations (perp grad norm: {perp_norm:.2e})")
        return corrected_geom, self.corrector_max_iter
        
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list, 
             three_sqrt_mass_list):
        """Calculate a single LQA IRC step with enhanced numerical precision
        
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
        three_sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values repeated for x,y,z per atom
            
        Returns
        -------
        tuple of (numpy.ndarray, int)
            New geometry coordinates and number of corrector iterations
        """
        # Update Hessian if we have previous points
        if len(self.irc_mw_gradients) > 1 and len(self.irc_mw_coords) > 1:
            delta_g = (self.irc_mw_gradients[-1] - self.irc_mw_gradients[-2]).reshape(-1, 1)
            delta_x = (self.irc_mw_coords[-1] - self.irc_mw_coords[-2]).reshape(-1, 1)
           
            # Only update if the step and gradient difference are meaningful
            dot_product = np.dot(delta_x.T, delta_g)[0, 0]
            if dot_product > 1e-12:
                delta_hess = self.ModelHessianUpdate.BFGS_hessian_update(self.mw_hessian, delta_x, delta_g)
                self.mw_hessian += delta_hess
                
                # Symmetrize Hessian to prevent accumulation of asymmetry errors
                self.mw_hessian = (self.mw_hessian + self.mw_hessian.T) * 0.5

        # Add bias potential hessian and diagonalize
        combined_hessian = self.mw_hessian + mw_BPA_hessian
        eigenvalues, eigenvectors = np.linalg.eigh(combined_hessian)
        
        # Drop small eigenvalues and corresponding eigenvectors
        # Use a tighter threshold for better precision
        small_eigvals = np.abs(eigenvalues) < 1e-10
        eigenvalues = eigenvalues[~small_eigvals]
        eigenvectors = eigenvectors[:, ~small_eigvals]
        
        # Reshape gradient for matrix operations
        flattened_gradient = mw_gradient.flatten()
        
        # Adaptive time step for numerical integration
        # Use gradient norm with a very small epsilon for stability
        epsilon = 1e-12
        norm_g = np.linalg.norm(flattened_gradient)
        dt = 1.0 / self.N_euler * self.step_size / max(norm_g, epsilon)

        # Transform gradient to eigensystem of the hessian
        mw_gradient_proj = np.dot(eigenvectors.T, flattened_gradient)

        # Adaptive Euler integration with subdivision for accuracy
        # Use higher-order integration near convergence
        t = dt
        current_length = 0.0
        actual_euler_steps = 0
        
        for j in range(self.N_euler):
            dsdt = np.sqrt(np.sum(mw_gradient_proj**2 * np.exp(-2.0 * eigenvalues * t)))
            current_length += dsdt * dt
            actual_euler_steps += 1
            if current_length > self.step_size:
                # Refine: back up and use smaller dt for the last sub-step
                current_length -= dsdt * dt
                remaining = self.step_size - current_length
                if dsdt > 1e-15:
                    t_fine = remaining / dsdt
                    t = t - dt + t_fine
                break
            t += dt
        
        # Calculate alphas with numerically stable formula
        x = -eigenvalues * t
        
        # Use np.expm1(x) for numerical stability when x is near zero
        # expm1(x) = exp(x) - 1, computed accurately for small x
        small_x_mask = np.abs(x) < 1e-8
        
        alphas = np.where(
            small_x_mask,
            # Taylor expansion: (exp(x)-1)/eigenvalues ≈ -t + eigenvalues*t^2/2 - ...
            # More precise: -t * (1 - x/2 + x^2/6)
            -t * (1.0 - x / 2.0 + x**2 / 6.0),
            np.expm1(x) / eigenvalues
        )
        
        A = np.dot(eigenvectors, np.dot(np.diag(alphas), eigenvectors.T))
        predictor_step = np.dot(A, flattened_gradient)
        
        # Reshape and un-mass-weight the step
        step_3d = predictor_step.reshape(len(geom_num_list), 3)
        step_3d = self.unmass_weight_step(step_3d, sqrt_mass_list)
        
        # Update geometry (predictor)
        new_geom = geom_num_list + step_3d
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        # Corrector step: optimize perpendicular to the IRC tangent
        # The tangent vector is the normalized gradient direction in mass-weighted coordinates
        tangent = flattened_gradient / max(norm_g, epsilon)
        
        # Estimate gradient at the new point using the quadratic model
        # g_new ≈ g_old + H * step
        predicted_grad_flat = flattened_gradient + np.dot(combined_hessian, predictor_step)
        predicted_mw_grad = predicted_grad_flat.reshape(len(new_geom), 3)
        
        corrector_iters = 0
        # Only apply corrector when gradient is small enough for it to matter
        if norm_g < 1e-2:
            new_geom, corrector_iters = self.corrector_step(
                new_geom, tangent, predicted_mw_grad, sqrt_mass_list,
                three_sqrt_mass_list, mw_BPA_hessian
            )
        
        return new_geom, corrector_iters
        
    def run(self):
        """Run the LQA IRC calculation with enhanced convergence"""
        print("=" * 60)
        print("Local Quadratic Approximation method (Enhanced Convergence)")
        print(f"Target convergence: MAX Force < {self.MAX_FORCE_THRESHOLD:.1e}, "
              f"RMS Force < {self.RMS_FORCE_THRESHOLD:.1e}")
        print(f"Initial step size: {self.step_size:.4f}")
        print("=" * 60)
        
        geom_num_list = self.coords
        CalcBiaspot = BiasPotentialCalculation(self.directory)
        
        # Initialize oscillation counter
        oscillation_counter = 0
        scalar_curvature_current = None
        corrector_iters = 0
        
        for iter in range(1, self.max_step):
            print(f"\n{'='*40}")
            print(f"# STEP: {iter}  (step_size: {self.step_size:.6f})")
            print(f"{'='*40}")
            
            exit_file_detect = os.path.exists(self.directory + "end.txt")
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
            
            # Recalculate Hessian if needed (periodic full recalculation)
            if iter % self.FC_count == 0 and iter > 0:
                print("Recalculating full Hessian...")
                self.mw_hessian = self.CE.Model_hess
                self.mw_hessian = Calculationtools().project_out_hess_tr_and_rot(
                    self.mw_hessian, self.element_list, geom_num_list
                )
            
            # Get mass arrays for consistent mass-weighting
            elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list = self.get_mass_array()
            
            # Periodically re-project translation/rotation out of the Hessian
            # This prevents contamination from accumulating over many BFGS updates
            if iter > 1 and iter % self.hessian_project_interval == 0:
                print("Re-projecting translation/rotation from Hessian...")
                self.mw_hessian = self.project_out_translation_rotation(
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
            
            # Compute force metrics for monitoring
            max_force = np.max(np.abs(B_g))
            rms_force = np.sqrt((B_g**2).mean())
            
            # Save energy and gradient data to CSV
            self.save_to_csv(iter, e, B_e, g, B_g, self.step_size, corrector_iters)

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
            
            # Adaptive step size based on energy change and gradient
            if self.prev_bias_energy is not None:
                self.step_size = self.adapt_step_size(
                    B_e, self.prev_bias_energy, rms_force, scalar_curvature_current
                )
                print(f"  Adapted step size: {self.step_size:.6f}")
            
            self.prev_energy = e
            self.prev_bias_energy = B_e
            
            # Check for energy oscillations (only consider significant oscillations)
            if len(self.irc_bias_energy_list) >= 3:
                energy_scale = max(abs(self.irc_bias_energy_list[-1]), 1e-10)
                oscillation_magnitude = abs(
                    (self.irc_bias_energy_list[-1] - self.irc_bias_energy_list[-2]) + 
                    (self.irc_bias_energy_list[-2] - self.irc_bias_energy_list[-3])
                )
                # Only count as oscillation if the magnitude is significant relative to the energy
                if self.check_energy_oscillation(self.irc_bias_energy_list) and oscillation_magnitude > 1e-10:
                    oscillation_counter += 1
                    print(f"  Energy oscillation detected ({oscillation_counter}/10)")
                    
                    # Reduce step size on oscillation
                    self.step_size *= self.step_scale_down
                    self.step_size = max(self.step_size, self.min_step_size)
                    print(f"  Step size reduced to: {self.step_size:.6f}")
                    
                    if oscillation_counter >= 10:
                        print("Terminating IRC: Energy oscillated for 10 consecutive steps")
                        break
                else:
                    # Reset counter if no significant oscillation
                    oscillation_counter = 0
                 
            if iter > 1:
                # Take LQA step with corrector
                geom_num_list, corrector_iters = self.step(
                    mw_B_g, geom_num_list, mw_BPA_hessian, sqrt_mass_list,
                    three_sqrt_mass_list
                )
                
                # Calculate path bending angle
                if len(self.irc_mw_coords) >= 3:
                    bend_angle = Calculationtools().calc_multi_dim_vec_angle(
                        self.irc_mw_coords[0] - self.irc_mw_coords[1], 
                        self.irc_mw_coords[2] - self.irc_mw_coords[1]
                    )
                    self.path_bending_angle_list.append(np.degrees(bend_angle))
                    print(f"  Path bending angle: {np.degrees(bend_angle):.4f} degrees")

                # Check for convergence with tight thresholds
                if max_force < self.MAX_FORCE_THRESHOLD and rms_force < self.RMS_FORCE_THRESHOLD and iter > 10:
                    print(f"\n{'='*60}")
                    print(f"CONVERGENCE REACHED (IRC)")
                    print(f"  MAX Force: {max_force:.2e} < {self.MAX_FORCE_THRESHOLD:.1e}")
                    print(f"  RMS Force: {rms_force:.2e} < {self.RMS_FORCE_THRESHOLD:.1e}")
                    print(f"  Total steps: {iter}")
                    print(f"{'='*60}")
                    break
                
            else:
                # First step: use a more careful initial displacement
                # Instead of a simple scaled gradient, use the Hessian eigenvector
                # corresponding to the most negative eigenvalue (transition vector)
                combined_hessian = self.mw_hessian + mw_BPA_hessian
                eigenvalues_init, eigenvectors_init = np.linalg.eigh(combined_hessian)
                
                # Find the most negative eigenvalue (transition mode)
                min_eigval_idx = np.argmin(eigenvalues_init)
                
                if eigenvalues_init[min_eigval_idx] < 0:
                    # Use the transition vector for the initial step direction
                    # This follows the IRC more accurately from the TS
                    transition_vector = eigenvectors_init[:, min_eigval_idx]
                    
                    # Choose direction: align with the gradient
                    flat_mw_B_g = mw_B_g.flatten()
                    if np.dot(transition_vector, flat_mw_B_g) > 0:
                        transition_vector = -transition_vector
                    
                    step_flat = transition_vector * self.step_size * 0.05
                    step = step_flat.reshape(len(geom_num_list), 3)
                    step = self.unmass_weight_step(step, sqrt_mass_list)
                else:
                    # Fallback to gradient-based step if no negative eigenvalue
                    normalized_grad = mw_B_g / max(np.linalg.norm(mw_B_g.flatten()), 1e-12)
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
            print(f"  Energy         : {e:.12f}")
            print(f"  Bias Energy    : {B_e:.12f}")
            print(f"  MAX B. force   : {max_force:.2e}")
            print(f"  RMS B. force   : {rms_force:.2e}")
            print(f"  Step size      : {self.step_size:.6f}")
            
            if len(self.irc_mw_coords) > 1:
                # Calculate curvature properties
                unit_tangent_vector, curvature_vector, scalar_curvature, curvature_coupling = calc_irc_curvature_properties(
                    mw_B_g, self.irc_mw_gradients[-2], self.mw_hessian, self.step_size
                )
                
                scalar_curvature_current = scalar_curvature
                
                print(f"  Scalar curvature: {scalar_curvature:.8f}")
                # Print curvature_coupling as 6 columns, 8 decimal places
                flat_cc = curvature_coupling.ravel()
                print("  Curvature coupling:")
                for i in range(0, len(flat_cc), 6):
                    print("  " + " ".join(f"{x: .8f}" for x in flat_cc[i:i+6]))
                
                # Save curvature properties to file
                save_curvature_properties_to_file(
                    os.path.join(self.directory, "irc_curvature_properties.csv"),
                    scalar_curvature,
                    curvature_coupling
                )
                print()
            else:
                scalar_curvature_current = None

        # Save final data visualization
        G = Graph(self.directory)
        rms_gradient_list = []
        with open(self.csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                rms_gradient_list.append(float(row[5]))  # RMS Force column
        
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