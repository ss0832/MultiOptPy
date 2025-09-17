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


class ModeKill:
    """ModeKill class for removing specific imaginary frequencies
    
    This class implements the ModeKill algorithm which selectively removes
    specific unwanted imaginary frequency modes by stepping downhill along 
    those eigenvectors.
    
    References
    ----------
    Based on the concept from pysisyphus package
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, kill_inds=None, nu_thresh=-5.0, max_step=1000, 
                 step_size=0.1, init_coord=None, init_hess=None, calc_engine=None, 
                 xtb_method=None, do_hess=True, hessian_update=True):
        """Initialize ModeKill calculator
        
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
        kill_inds : list or numpy.ndarray, optional
            Indices of modes to be removed (typically imaginary modes)
        nu_thresh : float, optional
            Threshold for considering a mode as imaginary (in cm^-1), default=-5.0
        max_step : int, optional
            Maximum number of steps
        step_size : float, optional
            Step size for the optimization
        init_coord : numpy.ndarray, optional
            Initial coordinates
        init_hess : numpy.ndarray, optional
            Initial hessian
        calc_engine : object, optional
            Calculator engine
        xtb_method : str, optional
            XTB method specification
        do_hess : bool, optional
            Whether to calculate final hessian, default=True
        hessian_update : bool, optional
            Whether to update hessian during optimization, default=True
        """
        self.max_step = max_step
        self.step_size = step_size
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
        # Kill mode parameters
        self.kill_inds = kill_inds  # Will be set in run() if None
        self.nu_thresh = float(nu_thresh)
        self.do_hess = do_hess
        self.hessian_update = hessian_update
        self.ovlp_thresh = 0.3
        
        # Initialize tracking variables
        self.prev_full_eigenvectors = None
        self.converged = False
        self.mw_down_step = None
        
        # initial condition
        self.coords = init_coord
        self.init_hess = init_hess
        self.mw_hessian = init_hess  # Mass-weighted hessian
        self.xtb_method = xtb_method
        
        # convergence criteria - using tight criteria
        self.MAX_FORCE_THRESHOLD = 0.0004
        self.RMS_FORCE_THRESHOLD = 0.0001

        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # Data storage for tracking the modes and their frequencies
        self.indices = np.arange(len(self.element_list) * 3)
        self.neg_nus = []
        self.kill_modes = None
        
        # IRC data storage for calculations
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
        self.csv_filename = os.path.join(self.directory, "modekill_energies_gradients.csv")
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 
                            'RMS Gradient', 'RMS Bias Gradient', 'Imaginary Frequencies'])
    
    def create_xyz_file(self):
        """Create XYZ file for structure data"""
        self.xyz_filename = os.path.join(self.directory, "modekill_structures.xyz")
        # Create empty file (will be appended to later)
        open(self.xyz_filename, 'w').close()
    
    def save_to_csv(self, step, energy, bias_energy, gradient, bias_gradient, imaginary_freqs=None):
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
        imaginary_freqs : list, optional
            List of imaginary frequencies
        """
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        if imaginary_freqs is None:
            imaginary_freqs = []
            
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad, str(imaginary_freqs)])
    
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
    
    def eigval_to_wavenumber(self, eigenvalue):
        """Convert eigenvalue to wavenumber (cm^-1)
        
        Parameters
        ----------
        eigenvalue : float or numpy.ndarray
            Eigenvalue(s) from hessian
            
        Returns
        -------
        float or numpy.ndarray
            Corresponding wavenumber(s) in cm^-1
        """
        # Constants
        au2rcm = 5140.48678
        
        # Convert eigenvalue to wavenumber
        sign = np.sign(eigenvalue)
        return sign * np.sqrt(np.abs(eigenvalue)) * au2rcm
    
    def update_mw_down_step(self, sqrt_mass_list, mw_gradient):
        """Update the downhill step direction based on current hessian
        
        Parameters
        ----------
        sqrt_mass_list : numpy.ndarray
            Array of sqrt(mass) values for each atom
        mw_gradient : numpy.ndarray
            Mass-weighted gradient
        """
        # Diagonalize the mass-weighted hessian
        w, v = np.linalg.eigh(self.mw_hessian)
        
        # Convert eigenvalues to wavenumbers
        nus = self.eigval_to_wavenumber(w)
        
        # Check if we have any negative eigenvalues below threshold
        neg_inds = nus < self.nu_thresh
        neg_nus = nus[neg_inds]
        
        # Check if we have modes to track
        if len(self.kill_inds) == 0:
            self.converged = True
            return
        
        # First time initialization
        if self.prev_full_eigenvectors is None:
            # Verify the kill indices point to imaginary modes
            try:
                assert all(nus[self.kill_inds] < self.nu_thresh), \
                      "ModeKill is intended for removal of imaginary frequencies " \
                     f"below {self.nu_thresh} cm^-1! The specified indices " \
                     f"{self.kill_inds} contain modes with positive frequencies " \
                     f"({nus[self.kill_inds]} cm^-1). Please choose different kill_inds!"
            except IndexError:
                print("Warning: Kill indices out of range. Using available negative modes.")
                self.kill_inds = np.where(nus < self.nu_thresh)[0]
                if len(self.kill_inds) == 0:
                    print("No negative modes found. ModeKill will exit.")
                    self.converged = True
                    return
                
            # Store the full eigenvector set for later comparison
            self.prev_full_eigenvectors = v
            self.kill_modes = v[:, self.kill_inds]
        else:
            # Calculate overlaps between previous eigenvectors and current eigenvectors
            new_kill_inds = []
            
            for idx in self.kill_inds:
                # Get the original kill mode from previous eigenvectors
                orig_mode = self.prev_full_eigenvectors[:, idx]
                
                # Find current mode with maximum overlap with original mode
                overlaps_with_orig = np.abs(np.dot(orig_mode, v))
                
                # Only consider negative eigenvalues (imaginary frequencies)
                neg_mask = w < 0
                overlaps_with_orig[~neg_mask] = 0
                
                # Find index with highest overlap
                max_overlap_idx = np.argmax(overlaps_with_orig)
                
                # Only keep the mode if it still has a significant overlap and is imaginary
                if overlaps_with_orig[max_overlap_idx] > self.ovlp_thresh and w[max_overlap_idx] < 0:
                    new_kill_inds.append(max_overlap_idx)
                    print(f"Mode {idx} tracked to current mode {max_overlap_idx} with overlap {overlaps_with_orig[max_overlap_idx]:.4f}")
            
            # If no modes to track anymore, we're done
            if len(new_kill_inds) == 0:
                print("No modes left to track. ModeKill will exit.")
                self.converged = True
                return
                
            # Update kill indices for the current iteration
            self.kill_inds = np.array(new_kill_inds, dtype=int)
            self.prev_full_eigenvectors = v
            self.kill_modes = v[:, self.kill_inds]
        
        # Determine correct sign for eigenvectors based on gradient overlap
        mw_grad_flatten = mw_gradient.flatten()
        mw_grad_normed = mw_grad_flatten / np.linalg.norm(mw_grad_flatten)
        overlaps = np.dot(self.kill_modes.T, mw_grad_normed)
        print("Overlaps between gradient and eigenvectors:")
        print(overlaps)
        
        # Flip eigenvector signs if needed (we want negative overlaps for downhill direction)
        flip = overlaps > 0
        print("Eigenvector signs to be flipped:")
        print(str(flip))
        self.kill_modes[:, flip] *= -1
        
        # Create the step as the sum of the downhill steps along the modes to remove
        self.mw_down_step = (self.step_size * self.kill_modes).sum(axis=1)
        
        # Log information about the current modes being killed
        print("\nCurrent modes to kill:")
        for i, mode_idx in enumerate(self.kill_inds):
            print(f"Mode {mode_idx}: {nus[mode_idx]:.2f} cm^-1")
            
    def get_additional_print(self):
        """Return additional information for printing during optimization
        
        Returns
        -------
        str
            String with additional information
        """
        if len(self.neg_nus) > 0:
            neg_nus = np.array2string(self.neg_nus[-1], precision=2)
            return f"\timag. ῦ: {neg_nus} cm^-1"
        return ""
    
    def run(self):
        """Run the ModeKill calculation"""
        print("ModeKill: Selective Imaginary Mode Removal")
        geom_num_list = self.coords
        CalcBiaspot = BiasPotentialCalculation(self.directory)
        
        # Initialize the Hessian if needed to identify modes to kill
        if self.mw_hessian is None or self.kill_inds is None:
            # Calculate initial Hessian
            self.CE.hessian_flag = True
            e, g, geom_num_list, finish_frag = self.CE.single_point(
                self.final_directory,
                self.element_list,
                0,
                self.electric_charge_and_multiplicity,
                self.xtb_method,
                UnitValueLib().bohr2angstroms * geom_num_list
            )
            self.mw_hessian = self.CE.Model_hess
            self.CE.hessian_flag = False
            
            # Project out translation and rotation
            self.mw_hessian = Calculationtools().project_out_hess_tr_and_rot(
                self.mw_hessian, self.element_list, geom_num_list
            )
            
            # Find negative eigenvalues if kill_inds not specified
            if self.kill_inds is None:
                w, v = np.linalg.eigh(self.mw_hessian)
                nus = self.eigval_to_wavenumber(w)
                neg_inds = np.where(nus < self.nu_thresh)[0]
                if len(neg_inds) > 0:
                    print(f"Found {len(neg_inds)} imaginary modes below {self.nu_thresh} cm^-1")
                    # By default, kill all imaginary modes except the first one (IRC mode)
                    if len(neg_inds) > 1:
                        self.kill_inds = neg_inds[1:]
                    else:
                        self.kill_inds = np.array([], dtype=int)
                        print("No secondary imaginary modes to remove.")
                        return
                else:
                    self.kill_inds = np.array([], dtype=int)
                    print("No imaginary modes found. Nothing to remove.")
                    return
                    
                print(f"Will attempt to remove modes: {self.kill_inds}")
        
        # First step is just preparation
        cur_cycle = 0
        
        # Get mass arrays for consistent mass-weighting
        elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list = self.get_mass_array()
        
        while not self.converged and cur_cycle < self.max_step:
            cur_cycle += 1
            print(f"# STEP: {cur_cycle}")
            
            # Check for early termination file
            exit_file_detect = os.path.exists(self.directory + "end.txt")
            if exit_file_detect:
                break
            
            # Calculate energy, gradient and update geometry
            e, g, geom_num_list, finish_frag = self.CE.single_point(
                self.final_directory,
                self.element_list,
                cur_cycle,
                self.electric_charge_and_multiplicity,
                self.xtb_method,
                UnitValueLib().bohr2angstroms * geom_num_list
            )
            
            # Calculate bias potential
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(
                e, g, geom_num_list, self.element_list,
                self.force_data, g, cur_cycle-1, geom_num_list
            )
            
            if finish_frag:
                break
                
            # Recalculate Hessian if needed
            if cur_cycle % self.FC_count == 0 or not self.hessian_update:
                self.CE.hessian_flag = True
                e, g, geom_num_list, finish_frag = self.CE.single_point(
                    self.final_directory,
                    self.element_list,
                    cur_cycle,
                    self.electric_charge_and_multiplicity,
                    self.xtb_method,
                    UnitValueLib().bohr2angstroms * geom_num_list
                )
                self.mw_hessian = self.CE.Model_hess
                self.CE.hessian_flag = False
                
                self.mw_hessian = Calculationtools().project_out_hess_tr_and_rot(
                    self.mw_hessian, self.element_list, geom_num_list
                )
                print("Recalculated exact hessian.")
            elif self.hessian_update and cur_cycle > 1:
                # Hessian update with mass-weighted values
                if len(self.irc_mw_coords) >= 2 and len(self.irc_mw_gradients) >= 2:
                    dx = self.irc_mw_coords[-1] - self.irc_mw_coords[-2]
                    dg = self.irc_mw_gradients[-1] - self.irc_mw_gradients[-2]
                    
                    dx_flat = dx.reshape(-1, 1)
                    dg_flat = dg.reshape(-1, 1)
                    
                    # Only update if the step and gradient difference are meaningful
                    inner_prod = np.dot(dx_flat.T, dg_flat)[0, 0]
                    if inner_prod > 1e-10:
                        delta_hess = self.ModelHessianUpdate.BFGS_hessian_update(
                            self.mw_hessian, dx_flat, dg_flat
                        )
                        self.mw_hessian += delta_hess
                        
                        norm_dx = np.linalg.norm(dx_flat)
                        norm_dg = np.linalg.norm(dg_flat)
                        print(f"Did BFGS hessian update: norm(dx)={norm_dx:.4e}, "
                             f"norm(dg)={norm_dg:.4e}.")
            
            # Mass-weight the bias potential hessian
            mw_BPA_hessian = self.mass_weight_hessian(BPA_hessian, three_sqrt_mass_list)
            
            # Mass-weight the coordinates
            mw_geom_num_list = self.mass_weight_coordinates(geom_num_list, sqrt_mass_list)
            
            # Mass-weight the gradients
            mw_g = self.mass_weight_gradient(g, sqrt_mass_list)
            mw_B_g = self.mass_weight_gradient(B_g, sqrt_mass_list)
            
            # Store data for next iteration
            if len(self.irc_mw_coords) >= 2:
                self.irc_mw_coords.pop(0)
                self.irc_mw_gradients.pop(0)
                self.irc_mw_bias_gradients.pop(0)
                
            self.irc_mw_coords.append(mw_geom_num_list.flatten())
            self.irc_mw_gradients.append(mw_g.flatten())
            self.irc_mw_bias_gradients.append(mw_B_g.flatten())
            
            # Save structure to XYZ file
            self.save_xyz_structure(cur_cycle, geom_num_list)
            
            # Update the downhill step
            self.update_mw_down_step(sqrt_mass_list, mw_B_g.flatten())
            
            # If converged flag was set in update_mw_down_step, break out
            if self.converged:
                print("All targeted modes have been removed.")
                break
            
            # Diagonalize the current hessian to check modes
            w, v = np.linalg.eigh(self.mw_hessian)
            
            # Calculate wavenumbers
            nus = self.eigval_to_wavenumber(w)
            neg_inds = nus <= self.nu_thresh
            neg_nus = nus[neg_inds]
            self.neg_nus.append(neg_nus)
            
            # Save energy and gradient data to CSV
            self.save_to_csv(cur_cycle, e, B_e, g, B_g, neg_nus.tolist())
            
            # Take a step
            if not self.converged and self.mw_down_step is not None:
                # Step along the downhill direction
                mw_step = self.mw_down_step
                unmw_step = self.unmass_weight_step(
                    mw_step.reshape(len(geom_num_list), 3), 
                    sqrt_mass_list
                )
                geom_num_list = geom_num_list + unmw_step
                
                # Remove center of mass motion
                geom_num_list -= Calculationtools().calc_center_of_mass(geom_num_list, self.element_list)
                
            # Calculate path bending angle if we have enough points
            if len(self.irc_mw_coords) >= 3:
                bend_angle = Calculationtools().calc_multi_dim_vec_angle(
                    self.irc_mw_coords[0] - self.irc_mw_coords[1], 
                    self.irc_mw_coords[2] - self.irc_mw_coords[1]
                )
                self.path_bending_angle_list.append(np.degrees(bend_angle))
                print("Path bending angle: ", np.degrees(bend_angle))
                
            # Check for convergence of gradients
            if convergence_check(B_g, self.MAX_FORCE_THRESHOLD, self.RMS_FORCE_THRESHOLD) and len(neg_nus) == 0:
                print("Converged: All imaginary modes removed and gradient criteria satisfied.")
                self.converged = True
                break
                
            # Print current geometry and info
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
            print(f"Imag. freqs    : {neg_nus} cm^-1")
            print()
        
        # Final hessian calculation if requested
        if self.do_hess and not finish_frag:
            print("Calculating final hessian...")
            self.CE.hessian_flag = True
            e_final, g_final, geom_num_list_final, _ = self.CE.single_point(
                self.final_directory,
                self.element_list,
                cur_cycle + 1,
                self.electric_charge_and_multiplicity,
                self.xtb_method,
                UnitValueLib().bohr2angstroms * geom_num_list
            )
            
            # Print final frequencies
            final_hess = self.CE.Model_hess
            final_hess = Calculationtools().project_out_hess_tr_and_rot(
                final_hess, self.element_list, geom_num_list
            )
            
            # Reset QM interface settings
            self.CE.hessian_flag = False
            self.CE.FC_COUNT = self.FC_count
            
            # Calculate and print final frequencies
            w_final, _ = np.linalg.eigh(final_hess)
            nus_final = self.eigval_to_wavenumber(w_final)
            neg_inds_final = nus_final <= self.nu_thresh
            neg_nus_final = nus_final[neg_inds_final]
            
            print(f"Final wavenumbers of imaginary modes (<= {self.nu_thresh} cm^-1):")
            print(f"{neg_nus_final} cm^-1")
            
            # Save final structure with frequencies
            self.save_to_csv(cur_cycle + 1, e_final, 0.0, g_final, g_final, neg_nus_final.tolist())
            self.save_xyz_structure(cur_cycle + 1, geom_num_list)
        
        # Save bending angle plot if we have enough data
        if len(self.path_bending_angle_list) > 0:
            G = Graph(self.directory)
            G.single_plot(
                np.array(range(len(self.path_bending_angle_list))),
                np.array(self.path_bending_angle_list),
                self.directory,
                atom_num=0,
                axis_name_1="# STEP",
                axis_name_2="bending angle [degrees]",
                name="ModeKill_bending"
            )
        
        print("ModeKill calculation finished.")
        return