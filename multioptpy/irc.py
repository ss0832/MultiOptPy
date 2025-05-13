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


class ModeKill:
    """Mode killing method for IRC calculations
    
    This method specifically targets and eliminates unwanted imaginary frequencies
    from transition state structures or other stationary points.
    
    References
    ----------
    [1] J. Chem. Theory Comput., 13, 4, 1632-1649 (2017)
    """
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                init_hess=None, calc_engine=None, xtb_method=None, kill_inds=None, nu_thresh=-5.0):
        """Initialize ModeKill IRC calculator
        
        By default, ModeKill will automatically target all imaginary modes except for the
        primary transition vector (the mode with the most negative eigenvalue).
        
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
        kill_inds : list or array, optional
            Indices of modes to eliminate. If None (default), automatically targets all 
            imaginary modes except the primary transition vector.
        nu_thresh : float, optional
            Frequency threshold (cm^-1) for identifying imaginary modes
        """
        self.max_step = max_step
        self.step_size = step_size
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
        # Initial condition
        self.coords = init_coord
        self.init_hess = init_hess
        self.mw_hessian = init_hess  # Mass-weighted hessian
        self.xtb_method = xtb_method
        
        # Convergence criteria - tighter for ModeKill
        self.MAX_FORCE_THRESHOLD = 0.0002  # Tighter than standard IRC
        self.RMS_FORCE_THRESHOLD = 0.00005  # Tighter than standard IRC

        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # ModeKill specific parameters
        self.kill_inds = np.array(kill_inds, dtype=int) if kill_inds is not None else None
        self.nu_thresh = float(nu_thresh)  # Threshold for imaginary frequencies
        self.overlap_thresh = 0.3  # Threshold for mode overlap reporting
        
        # Storage for killed modes and frequencies
        self.kill_modes = None  # Will store eigenvectors to be killed
        self.mw_down_step = None  # Direction for downhill step
        self.neg_nus = []  # List of negative frequencies at each step
        
        # IRC data storage
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
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 
                             'RMS Gradient', 'RMS Bias Gradient', 'Imag. Frequencies (cm^-1)'])
    
    def create_xyz_file(self):
        """Create XYZ file for structure data"""
        self.xyz_filename = os.path.join(self.directory, "irc_structures.xyz")
        # Create empty file (will be appended to later)
        open(self.xyz_filename, 'w').close()
    
    def save_to_csv(self, step, energy, bias_energy, gradient, bias_gradient, imag_freqs=None):
        """Save energy and gradient data to CSV file"""
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if imag_freqs is not None:
                writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad, 
                                 ','.join([f"{f:.2f}" for f in imag_freqs])])
            else:
                writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad, ''])
    
    def save_xyz_structure(self, step, coords):
        """Save molecular structure to XYZ file"""
        # Convert coordinates to Angstroms
        coords_angstrom = coords * UnitValueLib().bohr2angstroms
        
        with open(self.xyz_filename, 'a') as f:
            # Number of atoms and comment line
            f.write(f"{len(coords)}\n")
            f.write(f"ModeKill Step {step}\n")
            
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
    
    def eigval_to_wavenumber(self, eigval):
        """Convert eigenvalue to wavenumber (cm^-1)
        
        Parameters
        ----------
        eigval : float
            Eigenvalue of the mass-weighted hessian
            
        Returns
        -------
        float
            Corresponding wavenumber in cm^-1
        """
        # Constants
        BOHR_TO_ANGSTROM = UnitValueLib().bohr2angstroms
        AMU_TO_KG = 1.66053906660e-27
        HART_TO_JOULE = 4.3597447222071e-18
        C_LIGHT = 299792458
        PLANCK = 6.62607015e-34
        
        # Convert to wavenumber
        # For imaginary frequencies, return negative wavenumbers
        sign = np.sign(eigval)
        if eigval == 0:
            return 0.0
        
        # sqrt(hartree/(bohr^2*amu)) to sqrt(J/(m^2*kg))
        conv = np.sqrt(HART_TO_JOULE) / (BOHR_TO_ANGSTROM * 1e-10 * np.sqrt(AMU_TO_KG))
        # sqrt(J/(m^2*kg)) to cm^-1
        conv /= 2 * np.pi * C_LIGHT * 100
        
        return sign * abs(eigval)**0.5 * conv
    
    def update_mw_down_step(self, mw_gradient):
        """Update the downhill step based on modes to be killed
        
        Parameters
        ----------
        mw_gradient : numpy.ndarray
            Current mass-weighted gradient
        """
        # Diagonalize Hessian
        w, v = np.linalg.eigh(self.mw_hessian)
        
        # Get frequencies in cm^-1
        frequencies = np.array([self.eigval_to_wavenumber(eigval) for eigval in w])
        
        # If kill_inds is not provided, identify all imaginary modes below threshold
        # except for the most negative one (primary transition vector)
        if self.kill_inds is None:
            # Find all imaginary modes
            imag_inds = np.where(frequencies < self.nu_thresh)[0]
            
            if len(imag_inds) > 0:
                # Find the index of the most negative eigenvalue (transition vector)
                most_negative_idx = imag_inds[np.argmin(frequencies[imag_inds])]
                
                # Filter out the most negative eigenvalue from kill_inds
                self.kill_inds = np.array([idx for idx in imag_inds if idx != most_negative_idx])
                
                print(f"Preserving primary transition mode (index {most_negative_idx}, frequency {frequencies[most_negative_idx]:.2f} cm^-1)")
                print(f"Automatically targeting imaginary modes: {self.kill_inds}")
                print(f"Targeted frequencies: {frequencies[self.kill_inds]}")
            else:
                # No imaginary frequencies to kill
                self.kill_inds = np.array([], dtype=int)
                print("No imaginary frequencies found below threshold.")
        
        # Check if we have modes to kill
        if len(self.kill_inds) == 0:
            self.kill_modes = None
            self.mw_down_step = np.zeros_like(mw_gradient)
            self.neg_nus.append(np.array([]))
            print("No modes to kill.")
            return
        
        # Check that the specified modes are imaginary
        assert all(frequencies[self.kill_inds] < self.nu_thresh), \
            f"ModeKill can only eliminate imaginary frequencies below {self.nu_thresh} cm^-1!"
        
        # Store modes to be killed
        self.kill_modes = v[:, self.kill_inds]
        
        # Determine correct sign of eigenvectors from overlap with gradient
        # We want to step downhill, so overlap with gradient should be negative
        mw_grad_normed = mw_gradient / np.linalg.norm(mw_gradient)
        overlaps = np.einsum("ij,i->j", self.kill_modes, mw_grad_normed)
        print("Overlaps between gradient and eigenvectors to kill:")
        print(overlaps)
        
        # Flip sign where overlap is positive
        flip = overlaps > 0
        print(f"Eigenvector signs to be flipped: {flip}")
        self.kill_modes[:, flip] *= -1
        
        # Create step as sum of downhill steps along modes to remove
        self.mw_down_step = (self.step_size * self.kill_modes).sum(axis=1)
        
        # Store frequencies of imaginary modes
        neg_inds = frequencies <= self.nu_thresh
        neg_freqs = frequencies[neg_inds]
        self.neg_nus.append(neg_freqs)
        print(f"All imaginary modes (<= {self.nu_thresh} cm^-1):")
        print(f"{neg_freqs} cm^-1")
    
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list):
        """Calculate a single ModeKill IRC step
        
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
        bool
            True if converged (all targeted modes eliminated)
        """
        # Update Hessian if we have previous points
        if len(self.irc_mw_gradients) > 1 and len(self.irc_mw_coords) > 1:
            delta_g = (self.irc_mw_gradients[-1] - self.irc_mw_gradients[-2]).reshape(-1, 1)
            delta_x = (self.irc_mw_coords[-1] - self.irc_mw_coords[-2]).reshape(-1, 1)
           
            # Only update if the step and gradient difference are meaningful
            if np.dot(delta_x.T, delta_g)[0, 0] > 1e-10:
                # Use Bofill update for ModeKill
                from scipy.linalg import norm
                
                # BFGS part
                dx_dot_dg = np.dot(delta_x.T, delta_g)
                dx_sq = np.dot(delta_x.T, delta_x)
                dg_dx = np.dot(delta_g, delta_x.T)
                dx_dg = np.dot(delta_x, delta_g.T)
                
                B_bfgs = (dg_dx + dx_dg) / dx_dot_dg - np.dot(np.dot(dx_dg.T, dx_dg), np.dot(delta_x, delta_x.T)) / (dx_dot_dg**2)
                
                # SR1 part
                y = delta_g - np.dot(self.mw_hessian, delta_x)
                y_sq = np.dot(y.T, y)
                dx_y = np.dot(delta_x.T, y)
                B_sr1 = np.dot(y, y.T) / dx_y
                
                # Bofill parameter
                dx_norm = norm(delta_x)
                y_norm = norm(y)
                dxy_norm = dx_norm * y_norm
                phi = (dx_y / dxy_norm)**2
                
                # Combined update
                delta_hess = phi * B_bfgs + (1 - phi) * B_sr1
                self.mw_hessian += delta_hess
                
                print(f"Performed Bofill hessian update: phi={phi:.4f}")
        
        # Add bias potential hessian
        combined_hessian = self.mw_hessian + mw_BPA_hessian
        
        # Mass-weight the coordinates
        mw_coords = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
        
        # If this is the first step or we need to update the modes to kill
        if self.mw_down_step is None or len(self.kill_inds) == 0:
            self.update_mw_down_step(mw_gradient.flatten())
        
        # Check if any eigenvalues became positive
        w, v = np.linalg.eigh(combined_hessian)
        frequencies = np.array([self.eigval_to_wavenumber(eigval) for eigval in w])
        
        # Check overlaps between current modes and modes we want to kill
        if hasattr(self, 'kill_modes') and self.kill_modes is not None:
            overlaps = np.abs(np.einsum("ij,ik->jk", self.kill_modes, v))
            print("Overlaps between original modes and current modes:")
            
            # Report overlaps above threshold
            for i, ovlps in enumerate(overlaps):
                above_thresh = ovlps > self.overlap_thresh
                if any(above_thresh):
                    indices = np.arange(len(ovlps))
                    ovlp_str = " ".join(
                        [f"{idx:02d}: {o:.4f}" for idx, o in zip(indices[above_thresh], ovlps[above_thresh])]
                    )
                    print(f"Original mode {i:02d}: {ovlp_str}")
            
            # Find if any targeted modes became positive
            argmax = overlaps.argmax(axis=1)
            eigvals = w[argmax]
            pos_eigvals = eigvals > 0
            
            if any(pos_eigvals):
                # Only keep negative eigenvalues
                flipped = self.kill_inds[pos_eigvals]
                print(f"Modes {flipped} became positive, removing from kill list!")
                self.kill_inds = self.kill_inds[~pos_eigvals]
                
                # Update step direction if there are still modes to kill
                if len(self.kill_inds) > 0:
                    self.update_mw_down_step(mw_gradient.flatten())
                else:
                    print("All targeted modes have been eliminated!")
                    return geom_num_list, True  # Converged!
        
        # Take the step if we have modes to kill
        if len(self.kill_inds) > 0:
            mw_coords += self.mw_down_step.reshape(mw_coords.shape)
            
            # Convert back to non-mass-weighted
            new_geom = self.unmass_weight_step(mw_coords, sqrt_mass_list)
            
            # Remove center of mass motion
            new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
            
            return new_geom, False  # Not converged yet
        else:
            # No modes to kill, we're done
            return geom_num_list, True
    
    def run(self):
        """Run the ModeKill IRC calculation"""
        print("ModeKill method")
        geom_num_list = self.coords
        CalcBiaspot = BiasPotentialCalculation(self.directory)
        
        # Flag for convergence
        converged = False
        
        for iter in range(1, self.max_step):
            print("# STEP: ", iter)
            exit_file_detect = os.path.exists(self.directory+"end.txt")

            if exit_file_detect or converged:
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
            imag_freqs = self.neg_nus[-1] if self.neg_nus else None
            self.save_to_csv(iter, e, B_e, g, B_g, imag_freqs)

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
                 
            # Take ModeKill step
            geom_num_list, converged = self.step(
                mw_B_g, geom_num_list, mw_BPA_hessian, sqrt_mass_list
            )
            
            if converged:
                print("All targeted imaginary modes have been eliminated!")
            
            # Check for standard convergence
            if convergence_check(B_g, self.MAX_FORCE_THRESHOLD, self.RMS_FORCE_THRESHOLD) and iter > 5:
                print("Gradient convergence reached.")
                converged = True
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
            if self.neg_nus and len(self.neg_nus[-1]) > 0:
                print(f"Imag. freqs    : {', '.join([f'{freq:.2f}' for freq in self.neg_nus[-1]])} cm^-1")
            print()
        
        # Print final frequencies
        print("\nFinal calculation to verify imaginary frequencies:")
        final_e, final_g, geom_num_list, _ = self.CE.single_point(
            self.final_directory, 
            self.element_list, 
            self.max_step + 1,  # Extra step for final analysis
            self.electric_charge_and_multiplicity, 
            self.xtb_method,  
            UnitValueLib().bohr2angstroms * geom_num_list,
            force_hessian=True  # Force calculation of Hessian
        )
        
        # Get final Hessian and calculate frequencies
        final_hessian = self.CE.Model_hess
        final_hessian = Calculationtools().project_out_hess_tr_and_rot(
            final_hessian, self.element_list, geom_num_list
        )
        
        # Get mass arrays
        _, sqrt_mass_list, three_sqrt_mass_list, _ = self.get_mass_array()
        
        # Mass-weight the hessian
        mw_final_hessian = self.mass_weight_hessian(final_hessian, three_sqrt_mass_list)
        
        # Get eigenvalues and frequencies
        w, _ = np.linalg.eigh(mw_final_hessian)
        final_freqs = np.array([self.eigval_to_wavenumber(eigval) for eigval in w])
        
        # Report remaining imaginary frequencies
        imag_inds = final_freqs <= self.nu_thresh
        if any(imag_inds):
            imag_freqs = final_freqs[imag_inds]
            print(f"WARNING: There are still {len(imag_freqs)} imaginary frequencies:")
            print(f"{imag_freqs} cm^-1")
        else:
            print("SUCCESS: No imaginary frequencies remain below the threshold!")
            
        return

class DWI_Interpolator:
    """Distance weighted interpolation for energy and gradient interpolation
    
    References
    ----------
    [1] J. Chem. Phys. 120, 2877 (2004)
    [2] J. Chem. Phys. 109, 2807 (1998)
    """

    def __init__(self, n=4, maxlen=2):
        """Initialize DWI interpolator
        
        Parameters
        ----------
        n : int, optional
            Exponent for distance weighting, default=4
        maxlen : int, optional
            Maximum number of points to store, default=2
        """
        self.n = int(n)
        assert self.n > 0
        assert (self.n % 2) == 0
        self.maxlen = maxlen
        assert self.maxlen == 2, "Only maxlen=2 is supported"

        # Using lists with limited length for storing data points
        self.coords = []
        self.energies = []
        self.gradients = []
        self.hessians = []
    
    def update(self, coords, energy, gradient, hessian):
        """Add a new data point
        
        Parameters
        ----------
        coords : numpy.ndarray
            Coordinates
        energy : float
            Energy
        gradient : numpy.ndarray
            Gradient
        hessian : numpy.ndarray
            Hessian
        """
        # If we already have maxlen points, remove the oldest one
        if len(self.coords) >= self.maxlen:
            self.coords.pop(0)
            self.energies.pop(0)
            self.gradients.pop(0)
            self.hessians.pop(0)
            
        self.coords.append(coords)
        self.energies.append(energy)
        self.gradients.append(gradient)
        self.hessians.append(hessian)
        
        assert len(self.coords) == len(self.energies) == len(self.gradients) == len(self.hessians)

    def interpolate(self, at_coords, gradient=False):
        """Interpolate energy and optionally gradient at given coordinates
        
        Parameters
        ----------
        at_coords : numpy.ndarray
            Coordinates at which to interpolate
        gradient : bool, optional
            Whether to also interpolate gradient, default=False
            
        Returns
        -------
        float or tuple
            Interpolated energy, or (energy, gradient) if gradient=True
        """
        # Need at least 2 points for interpolation
        if len(self.coords) < 2:
            raise ValueError("Need at least 2 points for interpolation")
            
        c1, c2 = self.coords

        dx1 = at_coords - c1
        dx2 = at_coords - c2

        dx1_norm = np.linalg.norm(dx1)
        dx2_norm = np.linalg.norm(dx2)
        dx1_norm_n = dx1_norm**self.n
        dx2_norm_n = dx2_norm**self.n

        denom = dx1_norm**self.n + dx2_norm**self.n
        w1 = dx2_norm_n / denom
        w2 = dx1_norm_n / denom

        e1, e2 = self.energies
        g1, g2 = self.gradients
        h1, h2 = self.hessians

        t1 = taylor(e1, g1, h1, dx1)
        t2 = taylor(e2, g2, h2, dx2)

        E_dwi = w1*t1 + w2*t2

        if not gradient:
            return E_dwi

        t1_grad = taylor_grad(g1, h1, dx1)
        t2_grad = taylor_grad(g2, h2, dx2)

        # Calculate gradient with n/2 to account for square root reduction
        n_2 = self.n // 2
        dx1_norm_n_grad = 2 * n_2 * dx1_norm**(2*n_2-2) * dx1
        dx2_norm_n_grad = 2 * n_2 * dx2_norm**(2*n_2-2) * dx2
        w1_grad = (dx2_norm_n_grad*dx1_norm_n - dx1_norm_n_grad*dx2_norm_n) / denom**2
        w2_grad = -w1_grad

        # Complete gradient calculation
        w1_grad_flat = w1_grad.flatten()
        w2_grad_flat = w2_grad.flatten()
        grad_dwi = w1_grad_flat*t1 + w1*t1_grad.flatten() + w2_grad_flat*t2 + w2*t2_grad.flatten()
        
        # Reshape to match input gradient shape
        grad_dwi = grad_dwi.reshape(g1.shape)

        return E_dwi, grad_dwi


class DWI:
    """Distance-Weighted Interpolation method for IRC calculations
    
    References
    ----------
    [1] J. Chem. Phys. 120, 2877 (2004)
    [2] J. Chem. Phys. 109, 2807 (1998)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None, dwi_n=4):
        """Initialize DWI IRC calculator
        
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
        dwi_n : int, optional
            Exponent for distance weighting in DWI, default=4
        """
        self.max_step = max_step
        self.step_size = step_size
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
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
        
        # DWI specific parameters
        self.dwi_n = dwi_n
        self.interpolator = DWI_Interpolator(n=dwi_n)
        
        # IRC data storage
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
        """Save energy and gradient data to CSV file"""
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad])
    
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
        
    def check_energy_oscillation(self, energy_list):
        """Check if energy is oscillating (going up and down)"""
        if len(energy_list) < 3:
            return False
        
        # Check if the energy changes direction (from increasing to decreasing or vice versa)
        last_diff = energy_list[-1] - energy_list[-2]
        prev_diff = energy_list[-2] - energy_list[-3]
        
        # Return True if the energy direction has changed
        return (last_diff * prev_diff) < 0
    
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list, energy, bias_energy):
        """Calculate a single DWI IRC step
        
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
        energy : float
            Current energy
        bias_energy : float
            Current bias energy
            
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
        
        # Mass-weight the coordinates
        mw_coords = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
        
        # Update the DWI interpolator with current point
        self.interpolator.update(mw_coords, bias_energy, mw_gradient, combined_hessian)
        
        # If we don't have enough points for interpolation yet, use Euler step
        if len(self.interpolator.coords) < 2:
            # Take a simple step against the gradient
            flat_gradient = mw_gradient.flatten()
            grad_norm = np.linalg.norm(flat_gradient)
            
            if grad_norm < 1e-12:
                return geom_num_list  # No movement if gradient is essentially zero
                
            step_direction = -flat_gradient / grad_norm
            step_direction = step_direction.reshape(mw_gradient.shape)
            
            # Calculate step using Euler method
            mw_step = step_direction * self.step_size
            
            # Un-mass-weight the step
            step = self.unmass_weight_step(mw_step, sqrt_mass_list)
            
            # Update geometry
            new_geom = geom_num_list + step
            
            # Remove center of mass motion
            new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
            
            return new_geom
        
        # Determine next IRC point using DWI interpolation
        # Get tangent at current point (negative gradient direction)
        tangent = -mw_gradient / np.linalg.norm(mw_gradient.flatten())
        
        # Initial guess for next point
        next_mw_coords = mw_coords + self.step_size * tangent
        
        # We need to do a constrained optimization to find the point that:
        # 1) Is at the desired step size from current point
        # 2) Follows the IRC path as defined by DWI
        
        # Use Newton-Raphson iterations to refine the next point
        max_iter = 10
        converged = False
        
        for i in range(max_iter):
            # Get interpolated gradient at current guess
            _, interp_grad = self.interpolator.interpolate(next_mw_coords, gradient=True)
            
            # Calculate constraint residual (should be on hypersphere with radius step_size)
            displacement = next_mw_coords - mw_coords
            disp_norm = np.linalg.norm(displacement)
            step_diff = disp_norm - self.step_size
            
            # Check convergence
            if abs(step_diff) < 1e-6 and np.linalg.norm(interp_grad) < 1e-4:
                converged = True
                break
                
            # Calculate new point based on interpolated gradient and constraint
            # Project gradient onto hypersphere
            displacement_unit = displacement / disp_norm if disp_norm > 1e-10 else np.zeros_like(displacement)
            grad_parallel = np.sum(interp_grad * displacement_unit) * displacement_unit
            grad_perp = interp_grad - grad_parallel
            
            # Step size along tangent to hypersphere
            alpha = 0.1
            step_perp = -alpha * grad_perp
            
            # Step to correct radius
            step_parallel = -step_diff * displacement_unit
            
            # Update next coordinates
            next_mw_coords = next_mw_coords + step_perp + step_parallel
            
        if not converged:
            print("Warning: DWI step optimization did not converge fully")
            
        # Convert optimized coordinates back to non-mass-weighted
        new_geom = self.unmass_weight_step(next_mw_coords, sqrt_mass_list)
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
    
    def run(self):
        """Run the DWI IRC calculation"""
        print("Distance-Weighted Interpolation method")
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
                # Take DWI step
                geom_num_list = self.step(
                    mw_B_g, geom_num_list, mw_BPA_hessian, sqrt_mass_list, e, B_e
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

class GS:
    """Gonzalez-Schlegel method for IRC calculations
    
    This method uses constrained optimization on hyperspheres for accurate IRC path following.
    
    References
    ----------
    [1] J. Chem. Phys. 90, 2154 (1989)
    [2] J. Am. Chem. Soc. 112, 4009-4017 (1990)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None):
        """Initialize GS IRC calculator
        
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
        
        # IRC data storage
        self.irc_bias_energy_list = []
        self.irc_energy_list = []
        self.irc_mw_coords = []
        self.irc_mw_gradients = []
        self.irc_mw_bias_gradients = []
        self.path_bending_angle_list = []
        
        # GS specific parameters
        self.max_micro_cycles = 20  # Maximum number of micro cycles
        self.micro_step_thresh = 1e-3  # Convergence threshold for micro steps
        self.pivot_coords = []  # Store pivot points
        self.micro_coords = []  # Store micro cycles coordinates
        
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
        """Save energy and gradient data to CSV file"""
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad])
    
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
        
    def check_energy_oscillation(self, energy_list):
        """Check if energy is oscillating (going up and down)"""
        if len(energy_list) < 3:
            return False
        
        # Check if the energy changes direction (from increasing to decreasing or vice versa)
        last_diff = energy_list[-1] - energy_list[-2]
        prev_diff = energy_list[-2] - energy_list[-3]
        
        # Return True if the energy direction has changed
        return (last_diff * prev_diff) < 0
    
    def perp_component(self, vec, perp_to):
        """Calculate the perpendicular component of vec with respect to perp_to
        
        Parameters
        ----------
        vec : numpy.ndarray
            Vector to find perpendicular component of
        perp_to : numpy.ndarray
            Vector to find component perpendicular to
            
        Returns
        -------
        numpy.ndarray
            Perpendicular component of vec with respect to perp_to
        """
        # Subtract parallel component
        return vec - perp_to.dot(vec) * perp_to / perp_to.dot(perp_to)
    
    def micro_step(self, mw_coords, mw_gradient, displacement, mw_hessian, constraint):
        """Perform a constrained optimization step on a hypersphere
        
        Parameters
        ----------
        mw_coords : numpy.ndarray
            Current mass-weighted coordinates
        mw_gradient : numpy.ndarray
            Current mass-weighted gradient
        displacement : numpy.ndarray
            Current displacement from pivot point
        mw_hessian : numpy.ndarray
            Current mass-weighted hessian
        constraint : float
            Radius of the hypersphere constraint
            
        Returns
        -------
        tuple
            (new_coords, new_displacement, dx)
        """
        # Calculate eigenvalues and eigenvectors of the Hessian
        eigvals, eigvecs = np.linalg.eigh(mw_hessian)
        
        # Filter out small eigenvalues that could cause numerical problems
        big = np.abs(eigvals) > 1e-8
        big_eigvals = eigvals[big]
        big_eigvecs = eigvecs[:, big]
        
        # Project gradient and displacement onto eigenvector basis
        grad_star = big_eigvecs.T.dot(mw_gradient.flatten())
        displ_star = big_eigvecs.T.dot(displacement.flatten())
        
        def get_dx(lambda_):
            """Calculate step in eigenvector basis for a given lambda"""
            return -(grad_star - lambda_ * displ_star) / (big_eigvals - lambda_)
        
        def on_sphere(lambda_):
            """Constraint function to ensure step stays on hypersphere"""
            p = displ_star + get_dx(lambda_)
            return p.dot(p) - constraint
        
        # Initial guess for λ - must be smaller than the smallest eigenvalue
        lambda_0 = big_eigvals[0]
        lambda_0 *= 1.5 if (lambda_0 < 0) else 0.5
        print(f"\tSmallest eigenvalue is {big_eigvals[0]:.4f}, λ_0={lambda_0:.4f}")
        
        # Find the root using Newton's method
        from scipy.optimize import newton
        try:
            lambda_ = newton(on_sphere, lambda_0, maxiter=500)
            print(f"\tDetermined λ={lambda_:.4f} from Newton's method")
        except RuntimeError:
            print("\tNewton's method failed, using initial guess")
            lambda_ = lambda_0
        
        # Calculate dx from optimized lambda in eigenvector basis
        # and transform back to mass-weighted coordinates
        dx_star = get_dx(lambda_)
        dx = big_eigvecs.dot(dx_star)
        dx = dx.reshape(displacement.shape)
        
        # Update displacement and coordinates
        new_displacement = displacement + dx
        new_coords = mw_coords + dx
        
        # Calculate gradient component tangent to sphere
        grad_tangent = self.perp_component(mw_gradient.flatten(), new_displacement.flatten())
        grad_tangent = grad_tangent.reshape(mw_gradient.shape)
        
        return new_coords, new_displacement, dx
    
    def step(self, mw_gradient, geom_num_list, mw_BPA_hessian, sqrt_mass_list):
        """Calculate a single GS IRC step
        
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
        
        # Mass-weight the coordinates
        mw_coords = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
        
        # Flatten gradient for norm calculation
        flat_gradient = mw_gradient.flatten()
        grad_norm = np.linalg.norm(flat_gradient)
        
        if grad_norm < 1e-12:
            return geom_num_list  # No movement if gradient is essentially zero
            
        # Take a half-step against the gradient to the pivot point
        pivot_step = 0.5 * self.step_size * -flat_gradient / grad_norm
        pivot_step = pivot_step.reshape(mw_gradient.shape)
        pivot_coords = mw_coords + pivot_step
        
        # Initial displacement from pivot point
        displacement = pivot_step
        
        # Initial guess for next point (full step from current position)
        current_coords = pivot_coords + displacement
        
        # Constraint is the squared radius of the hypersphere
        constraint = (0.5 * self.step_size) ** 2
        
        # Store initial coordinates and gradients for micro cycles
        prev_coords = mw_coords.copy()
        prev_grad = mw_gradient.copy()
        
        # Perform micro cycles for constrained optimization
        for i in range(self.max_micro_cycles):
            print(f"Micro cycle {i+1:02d}")
            
            # Calculate energy and gradient at current point
            # Convert to non-mass-weighted for QM calculation
            unmw_current = self.unmass_weight_step(current_coords, sqrt_mass_list)
            
            e, g, _, _ = self.CE.single_point(
                self.final_directory, 
                self.element_list, 
                -1,  # Temporary calculation, no step number
                self.electric_charge_and_multiplicity, 
                self.xtb_method,  
                UnitValueLib().bohr2angstroms * unmw_current
            )
            
            # Calculate bias potential
            CalcBiaspot = BiasPotentialCalculation(self.directory)
            _, _, B_g, _ = CalcBiaspot.main(
                e, g, unmw_current, self.element_list, 
                self.force_data, g, -1, unmw_current
            )
            
            # Mass-weight the gradient
            mw_B_g = self.mass_weight_gradient(B_g, sqrt_mass_list)
            
            # Update Hessian using BFGS
            if i > 0:  # Skip first iteration
                delta_g = mw_B_g.flatten() - prev_grad.flatten()
                delta_x = current_coords.flatten() - prev_coords.flatten()
                
                # Reshape for Hessian update
                delta_g = delta_g.reshape(-1, 1)
                delta_x = delta_x.reshape(-1, 1)
                
                # Only update if the step and gradient difference are meaningful
                if np.dot(delta_x.T, delta_g)[0, 0] > 1e-10:
                    delta_hess = self.ModelHessianUpdate.BFGS_hessian_update(combined_hessian, delta_x, delta_g)
                    combined_hessian += delta_hess
            
            # Store current state for next iteration
            prev_coords = current_coords.copy()
            prev_grad = mw_B_g.copy()
            
            # Perform micro step
            current_coords, displacement, dx = self.micro_step(
                current_coords, mw_B_g, displacement, combined_hessian, constraint
            )
            
            # Check for convergence of micro cycles
            dx_norm = np.linalg.norm(dx)
            print(f"\tnorm(dx)={dx_norm:.6f}")
            if dx_norm <= self.micro_step_thresh:
                print(f"Micro cycles converged after {i+1} iterations")
                break
        else:
            print("Warning: Maximum micro cycles reached without convergence")
            
        # Convert optimized coordinates back to non-mass-weighted
        new_geom = self.unmass_weight_step(current_coords, sqrt_mass_list)
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
    
    def run(self):
        """Run the GS IRC calculation"""
        print("Gonzalez-Schlegel method")
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
                # Take GS step
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

class Euler:
    """Euler method for IRC calculations
    
    This is the simplest integration method for following reaction paths.
    
    References
    ----------
    [1] Int. J. Quantum Chem., 29, 1877–1886 (1986)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None):
        """Initialize Euler IRC calculator
        
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
        """Calculate a single Euler IRC step
        
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
        
        # Flatten gradient for norm calculation
        flat_gradient = mw_gradient.flatten()
        grad_norm = np.linalg.norm(flat_gradient)
        
        # Avoid division by zero
        if grad_norm < 1e-12:
            step_direction = np.zeros_like(mw_gradient)
        else:
            # Step downhill, against the gradient
            step_direction = -flat_gradient / grad_norm * self.step_size
            step_direction = step_direction.reshape(mw_gradient.shape)
            
        # Calculate step using Euler method
        mw_step = step_direction 
        
        # Un-mass-weight the step
        step = self.unmass_weight_step(mw_step, sqrt_mass_list)
        
        # Update geometry
        new_geom = geom_num_list + step
        
        # Remove center of mass motion
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
        
    def run(self):
        """Run the Euler IRC calculation"""
        print("Euler integration method")
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
                # Take Euler step
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

        elif self.method.upper() == "GS":
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = GS(
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
                
                IRCmethod = GS(
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
                IRCmethod = GS(
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

        elif self.method.upper() == "EULER":
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = Euler(
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
                
                IRCmethod = Euler(
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
                IRCmethod = Euler(
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


        elif self.method.upper() == "MODEKILL":
            # For ModeKill, we don't need forward/backward directions 
            # as we're targeting specific eigenmodes
            print("ModeKill Method")
            
            # By default, preserve the primary transition vector and kill all other imaginary modes
            kill_inds = None
            
            IRCmethod = ModeKill(
                self.element_list, 
                self.electric_charge_and_multiplicity, 
                self.FC_count, 
                self.file_directory, 
                self.final_directory, 
                self.force_data, 
                max_step=self.max_step, 
                step_size=self.step_size, 
                init_coord=self.geom_num_list, 
                init_hess=self.hessian, 
                calc_engine=self.QM_interface,
                xtb_method=self.xtb_method,
                kill_inds=kill_inds
            )
            IRCmethod.run()

        elif self.method.upper() == "DWI":
            if self.IRC_flag:
                # Forward direction (from TS to products)
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                
                # Create forward direction directory
                fwd_dir = os.path.join(self.file_directory, "irc_forward")
                os.makedirs(fwd_dir, exist_ok=True)
                
                IRCmethod = DWI(
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
                
                IRCmethod = DWI(
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
                IRCmethod = DWI(
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