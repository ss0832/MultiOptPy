import numpy as np
from pathlib import Path
import os

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Visualization.visualization import Graph

class NewtonTrajectory:
    """
    Implementation of the Growing Newton Trajectory (GNT) method for finding transition states.
    
    Reference:
    [1] Quapp, "Finding the Transition State without Initial Guess: The Growing 
        String Method for Newton Trajectory to Isomerization and Enantiomerization",
        J. Comput. Chem. 2005, 26, 1383-1399, DOI: 10.1063/1.1885467
    """
    def __init__(self, config):
        self.config = config
        self.step_len = config.gnt_step_len
        self.rms_thresh = config.gnt_rms_thresh
        self.gnt_vec = config.gnt_vec
        self.micro_iter_limit = config.gnt_microiter
        self.out_dir = Path(config.iEIP_FOLDER_DIRECTORY) if hasattr(Path, '__call__') else config.iEIP_FOLDER_DIRECTORY
        
        # Initialize storage for trajectory data
        self.images = []
        self.all_energies = []
        self.all_real_forces = []
        self.sp_images = []
        self.ts_images = []
        self.min_images = []
        self.ts_imag_freqs = []
        
        # Flags for tracking stationary points
        self.passed_min = False
        self.passed_ts = False
        self.did_reparametrization = False
        
    
    def rms(self, vector):
        """Calculate root mean square of a vector"""
        return np.sqrt(np.mean(np.square(vector)))
    
    def get_r(self, current_geom, final_geom=None):
        """Determine search direction vector"""
        if final_geom is not None:
            current_geom, _ = Calculationtools().kabsch_algorithm(current_geom, final_geom)
            r = final_geom - current_geom
        elif self.gnt_vec is not None:
            # Parse atom indices from gnt_vec string
            atom_indices = list(map(int, self.gnt_vec.split(",")))
            if len(atom_indices) % 2 != 0:
                raise ValueError("Invalid gnt_vec format. Need even number of atom indices.")
            
            r = np.zeros_like(current_geom)
            for i in range(len(atom_indices) // 2):
                atom_i = atom_indices[2*i] - 1  # Convert to 0-indexed
                atom_j = atom_indices[2*i+1] - 1
                # Create a displacement vector between these atoms
                r[atom_i] = current_geom[atom_j] - current_geom[atom_i]
                r[atom_j] = current_geom[atom_i] - current_geom[atom_j]
        else:
            raise ValueError("Need to specify either final_geom or gnt_vec")
            
        # Normalize the direction vector
        r = r / np.linalg.norm(r)
        return r
        
    def calc_projector(self, r):
        """Calculate projector that keeps perpendicular component"""
        flat_r = r.reshape(-1)
        return np.eye(flat_r.size) - np.outer(flat_r, flat_r)
        
    def grow_image(self, SP, FIO, geom, element_list, charge_multiplicity, r, iter_num, file_directory):
        """Grow a new image along the Newton trajectory"""
        # Store current image
        self.images.append(geom.copy())
        
        # Calculate new displacement along r
        step = self.step_len * r
        new_geom = geom + step
        
        # Prepare and run calculation at the new geometry
        new_geom_tolist = (new_geom * self.config.bohr2angstroms).tolist()
        for i, elem in enumerate(element_list):
            new_geom_tolist[i].insert(0, elem)
        
        new_geom_tolist.insert(0, charge_multiplicity)
        
        file_directory = FIO.make_psi4_input_file([new_geom_tolist], iter_num)
        energy, forces, geom_coords, error_flag = SP.single_point(
            file_directory, element_list, iter_num, charge_multiplicity, self.config.force_data["xtb"]
        )
        
        if error_flag:
            print("Error in QM calculation during trajectory growth.")
            with open(os.path.join(self.out_dir, "end.txt"), "w") as f:
                f.write("Error in QM calculation during trajectory growth.")
            return None, None, None, True, None
            
        # Store results
        self.all_energies.append(energy)
        self.all_real_forces.append(forces)
        
        return energy, forces, geom_coords, False, file_directory
        
    def initialize(self, SP, FIO, initial_geom, element_list, charge_multiplicity, file_directory, final_geom=None, iter_num=0):
        """Initialize the Newton trajectory
        
        Parameters:
        -----------
        SP : SinglePoint
            Object to perform single point calculations
        FIO : FileIO
            Object for file I/O operations
        initial_geom : ndarray
            Initial geometry coordinates
        element_list : list
            List of element symbols
        charge_multiplicity : list
            [charge, multiplicity]
        file_directory : str
            Path to current input file
        final_geom : ndarray, optional
            Final geometry coordinates
        iter_num : int, optional
            Current iteration number
        """
        # Use the provided file_directory instead of trying to get it from FIO
        energy, forces, geom_coords, error_flag = SP.single_point(
            file_directory, element_list, iter_num, charge_multiplicity, self.config.force_data["xtb"]
        )
        
        if error_flag:
            print("Error in QM calculation during initialization.")
            return None, None, True
            
        # Store initial data
        self.images.append(geom_coords.copy())
        self.all_energies.append(energy)
        self.all_real_forces.append(forces)
        
        # Calculate search direction
        self.r = self.get_r(geom_coords, final_geom)
        self.r_org = self.r.copy()
        
        # Calculate projector
        self.P = self.calc_projector(self.r)
        
        # Grow first image
        energy, forces, geom_coords, error_flag, new_file_directory = self.grow_image(
            SP, FIO, geom_coords, element_list, charge_multiplicity, self.r, iter_num, file_directory
        )
        
        return geom_coords, new_file_directory, error_flag
        
    def optimize_frontier_image(self, SP, FIO, geom, element_list, charge_multiplicity, iter_num, file_directory):
        """Optimize the frontier image using projected forces"""
        # Initialize BFGS variables
        num_atoms = len(element_list)
        num_coords = num_atoms * 3
        H_inv = np.eye(num_coords)  # Initial inverse Hessian approximation
        prev_geom = None
        prev_proj_grad = None
        
        # Get current energy and forces - use provided file_directory
        energy, forces, geom_coords, error_flag = SP.single_point(
            file_directory, element_list, iter_num, charge_multiplicity, self.config.force_data["xtb"]
        )
        
        if error_flag:
            print("Error in QM calculation during frontier optimization.")
            return None, None, None, True, None
        
        # Main optimization loop
        for micro_iter in range(self.micro_iter_limit):
            # Project forces onto perpendicular space
            flat_forces = forces.reshape(-1)
            proj_forces = np.dot(self.P, flat_forces).reshape(geom_coords.shape)
            
            # Calculate RMS of projected forces
            proj_rms = self.rms(proj_forces)
            
            if micro_iter % 5 == 0:
                print(f"Micro-iteration {micro_iter}: Projected force RMS = {proj_rms:.6f}, Energy = {energy:.8f}")
                
            # Check convergence
            if proj_rms <= self.rms_thresh:
                print(f"Frontier image converged after {micro_iter} micro-iterations")
                break
                
            # BFGS update
            flat_geom = geom_coords.flatten()
            flat_proj_forces = proj_forces.flatten()
            
            if prev_geom is not None:
                s = flat_geom - prev_geom  # Position difference
                y = prev_proj_grad - flat_proj_forces  # Force difference (note: forces = -gradient)
                
                # Check curvature condition
                sy = np.dot(s, y)
                if sy > 1e-10:
                    # BFGS update formula
                    rho = 1.0 / sy
                    V = np.eye(len(s)) - rho * np.outer(s, y)
                    H_inv = np.dot(V.T, np.dot(H_inv, V)) + rho * np.outer(s, s)
            
            # Store current values for next iteration
            prev_geom = flat_geom.copy()
            prev_proj_grad = flat_proj_forces.copy()
            
            # Calculate search direction
            search_dir = -np.dot(H_inv, flat_proj_forces).reshape(geom_coords.shape)
            
            # Determine step size (simple trust radius approach)
            trust_radius = 0.02  # Bohr
            step_norm = np.linalg.norm(search_dir)
            if step_norm > trust_radius:
                search_dir = search_dir * (trust_radius / step_norm)
                
            # Update geometry
            geom_coords = geom_coords + search_dir
            
            # Prepare and run calculation at the new geometry
            new_geom_tolist = (geom_coords * self.config.bohr2angstroms).tolist()
            for i, elem in enumerate(element_list):
                new_geom_tolist[i].insert(0, elem)
            
            new_geom_tolist.insert(0, charge_multiplicity)
            
            file_directory = FIO.make_psi4_input_file([new_geom_tolist], iter_num)
            energy, forces, geom_coords, error_flag = SP.single_point(
                file_directory, element_list, iter_num, charge_multiplicity, self.config.force_data["xtb"]
            )
            
            if error_flag:
                print("Error in QM calculation during frontier optimization.")
                return None, None, None, True, None
                
        # Return optimized geometry
        return energy, forces, geom_coords, False, file_directory
        
    def reparametrize(self, SP, FIO, geom, element_list, charge_multiplicity, iter_num, file_directory):
        """Check if NT can be grown and update trajectory"""
        # Get latest energy and forces
        energy = self.all_energies[-1]
        real_forces = self.all_real_forces[-1]
        
        # Get projected forces
        flat_forces = real_forces.reshape(-1)
        proj_forces = np.dot(self.P, flat_forces).reshape(real_forces.shape)
        
        # Check if we can grow the NT (convergence of frontier image)
        proj_rms = self.rms(proj_forces)
        can_grow = proj_rms <= self.rms_thresh
        
        if can_grow:
          
            # Check for stationary points
            ae = self.all_energies
            if len(ae) >= 3:
                self.passed_min = ae[-3] > ae[-2] < ae[-1]
                self.passed_ts = ae[-3] < ae[-2] > ae[-1]
                
                if self.passed_min or self.passed_ts:
                    sp_image = self.images[-2].copy()
                    sp_kind = "Minimum" if self.passed_min else "TS"
                    self.sp_images.append(sp_image)
                    print(f"Passed stationary point! It seems to be a {sp_kind}.")
                    
                    if self.passed_ts:
                        self.ts_images.append(sp_image)
                        # Calculate Hessian at TS if needed
                        # This would require additional implementation
                    elif self.passed_min:
                        self.min_images.append(sp_image)
            
            # Update search direction if needed
            r_new = self.get_r(geom)
            r_dot = np.dot(r_new.reshape(-1), self.r.reshape(-1))
            r_org_dot = np.dot(r_new.reshape(-1), self.r_org.reshape(-1))
            print(f"r.dot(r')={r_dot:.6f} r_org.dot(r')={r_org_dot:.6f}")
            
            # Update r if direction has changed significantly
            if r_org_dot <= 0.5 and self.passed_min:  # Using 0.5 as threshold
                self.r = r_new
                self.P = self.calc_projector(self.r)
                print("Updated r")
            
            # Grow new image
            energy, forces, geom_coords, error_flag, new_file_directory = self.grow_image(
                SP, FIO, geom, element_list, charge_multiplicity, self.r, iter_num, file_directory
            )
            
            if error_flag:
                return None, True, None
            
            self.did_reparametrization = True
            return geom_coords, False, new_file_directory
        else:
            # Optimize frontier image since it's not converged yet
            energy, forces, geom_coords, error_flag, new_file_directory = self.optimize_frontier_image(
                SP, FIO, geom, element_list, charge_multiplicity, iter_num, file_directory
            )
            
            if error_flag:
                return None, True, None
                
            # Update stored energy and forces
            self.all_energies[-1] = energy
            self.all_real_forces[-1] = forces
            
            self.did_reparametrization = False
            return geom_coords, False, new_file_directory
            
    def check_convergence(self):
        """Check if the Newton Trajectory calculation has converged"""
        if len(self.ts_images) == 0:
            return False
            
        # Consider converged if we've found a TS
        return True
        
    def get_additional_print(self):
        """Get additional information for printing"""
        if self.did_reparametrization:
            img_num = len(self.images)
            str_ = f"Grew Newton trajectory to {img_num} images."
            if self.passed_min:
                str_ += f" Passed minimum geometry at image {img_num-1}."
            elif self.passed_ts:
                str_ += f" Passed transition state geometry at image {img_num-1}."
        else:
            str_ = None
            
        # Reset flags
        self.did_reparametrization = False
        self.passed_min = False
        self.passed_ts = False
        
        return str_
        
    def main(self, file_directory_1, file_directory_2, SP1, SP2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, FIO1, FIO2):
        """Main method to run Newton Trajectory calculation"""
        G = Graph(self.config.iEIP_FOLDER_DIRECTORY)
        BIAS_GRAD_LIST_A = []
        BIAS_ENERGY_LIST_A = []
        GRAD_LIST_A = []
        ENERGY_LIST_A = []
        
        # Get initial geometry from first file
        energy_1, gradient_1, geom_num_list_1, error_flag_1 = SP1.single_point(
            file_directory_1, element_list, 0, init_electric_charge_and_multiplicity, self.config.force_data["xtb"]
        )
        
        if error_flag_1:
            print("Error in initial QM calculation.")
            with open(os.path.join(self.config.iEIP_FOLDER_DIRECTORY, "end.txt"), "w") as f:
                f.write("Error in initial QM calculation.")
            return
            
        # If using final geometry for direction, get it
        final_geom = None
        if self.gnt_vec is None:
            energy_2, gradient_2, geom_num_list_2, error_flag_2 = SP2.single_point(
                file_directory_2, element_list, 0, final_electric_charge_and_multiplicity, self.config.force_data["xtb"]
            )
            if error_flag_2:
                print("Error in second QM calculation.")
                with open(os.path.join(self.config.iEIP_FOLDER_DIRECTORY, "end.txt"), "w") as f:
                    f.write("Error in second QM calculation.")
                return
            final_geom = geom_num_list_2
        
        # Initialize Newton trajectory
        geom, file_directory, error_flag = self.initialize(
            SP1, FIO1, geom_num_list_1, element_list, init_electric_charge_and_multiplicity, file_directory_1, final_geom
        )
        
        if error_flag:
            return
        
        # Main iteration loop
        for iter in range(1, self.config.microiterlimit):
            print(f"==========================================================")
            print(f"Newton Trajectory Iteration ({iter}/{self.config.microiterlimit})")
            
            # Check for early termination
            if os.path.isfile(os.path.join(self.config.iEIP_FOLDER_DIRECTORY, "end.txt")):
                break
                
            # Grow trajectory or optimize frontier image
            geom, error_flag, file_directory = self.reparametrize(
                SP1, FIO1, geom, element_list, init_electric_charge_and_multiplicity, iter, file_directory
            )
            
            if error_flag:
                break
                
            # Get current energy and forces
            energy = self.all_energies[-1]
            forces = self.all_real_forces[-1]
            
            # Calculate bias potential if needed
            BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
            _, bias_energy, bias_gradient, _ = BPC.main(
                energy, forces, geom, element_list, self.config.force_data
            )
            
            # Record data for plotting
            ENERGY_LIST_A.append(energy * self.config.hartree2kcalmol)
            GRAD_LIST_A.append(np.sqrt(np.sum(forces**2)))
            BIAS_ENERGY_LIST_A.append(bias_energy * self.config.hartree2kcalmol)
            BIAS_GRAD_LIST_A.append(np.sqrt(np.sum(bias_gradient**2)))
            
            # Print current status
            add_info = self.get_additional_print() or ""
            print(f"Energy                : {energy}")
            print(f"Bias Energy           : {bias_energy}")
            print(f"Gradient  Norm        : {np.linalg.norm(forces)}")
            print(f"Bias Gradient Norm    : {np.linalg.norm(bias_gradient)}")
            print(add_info)
            print(f"==========================================================")
            
            # Check for convergence
            if self.check_convergence():
                print("Newton Trajectory converged to transition state!")
                break
                
        else:
            print("Reached maximum number of iterations. Newton trajectory calculation completed.")
            
        # Create energy and gradient profile plots
        NUM_LIST = list(range(len(ENERGY_LIST_A)))
        
        G.single_plot(NUM_LIST, ENERGY_LIST_A, file_directory_1, "energy", 
                      axis_name_2="energy [kcal/mol]", name="nt_energy")
        G.single_plot(NUM_LIST, GRAD_LIST_A, file_directory_1, "gradient", 
                      axis_name_2="grad (RMS) [a.u.]", name="nt_gradient")
        G.single_plot(NUM_LIST, BIAS_ENERGY_LIST_A, file_directory_1, "bias_energy", 
                      axis_name_2="energy [kcal/mol]", name="nt_bias_energy")
        G.single_plot(NUM_LIST, BIAS_GRAD_LIST_A, file_directory_1, "bias_gradient", 
                      axis_name_2="grad (RMS) [a.u.]", name="nt_bias_gradient")
        
        # Create trajectory file
        FIO1.make_traj_file_for_DM(img_1="A", img_2="B")
        
        # Identify critical points
        FIO1.argrelextrema_txt_save(ENERGY_LIST_A, "approx_TS", "max")
        FIO1.argrelextrema_txt_save(ENERGY_LIST_A, "approx_EQ", "min")
        FIO1.argrelextrema_txt_save(GRAD_LIST_A, "local_min_grad", "min")
        
        return
