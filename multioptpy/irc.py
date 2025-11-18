import numpy as np
import os
import glob
import csv

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Parameters.parameter import atomic_mass
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.IRC.lqa import LQA
from multioptpy.IRC.hpc import HPC
from multioptpy.IRC.rk4 import RK4
from multioptpy.IRC.dvv import DVV
from multioptpy.IRC.modekill import ModeKill
from multioptpy.IRC.euler import Euler
from multioptpy.IRC.converge_criteria import convergence_check
from multioptpy.fileio import traj2list

### I recommend to use LQA method to calculate IRC path ###

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
        tmp_method = irc_method[2].split(":")
        if len(tmp_method) > 1:
            self.method = tmp_method[0]
            self.method_options = tmp_method[1:]
        else:
            self.method = irc_method[2]
            self.method_options = []

        self.file_directory = os.path.abspath(directory)+"/"
        self.final_directory = os.path.abspath(final_directory)+"/"
        self.QM_interface = QM_interface
        
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.xtb_method = xtb_method
        
        self.force_data = force_data
        self.FC_count = FC_count

        # convergence criteria
        self.MAX_FORCE_THRESHOLD = 0.0004
        self.RMS_FORCE_THRESHOLD = 0.0001
        
        # Will be set in saddle_check
        self.IRC_flag = False
        self.initial_step = None
        self.geom_num_list = None
        self.ts_coords = None
        self.fin_xyz_base = None # Added
        self.terminal_struct_paths = [] # Added
    
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
        # --- Added ---
        if fin_xyz:
            self.fin_xyz_base = os.path.basename(fin_xyz[0]).split('.')[0]
        else:
            print("Warning: No XYZ file found in final_directory. Using default name 'input' for terminal structures.")
            self.fin_xyz_base = "input"
        # --- End of addition ---
            
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
        isconverged = convergence_check(init_g, self.MAX_FORCE_THRESHOLD, self.RMS_FORCE_THRESHOLD)
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
        if self.method.upper() == "MODEKILL":
            print("Execute ModeKill")
            # ModeKill: Remove imaginary modes using ModeKill class
            IRC_flag = False
            gradient = self.QM_interface.gradient.reshape(len(geom_num_list), 3)
            initial_step = np.zeros_like(gradient)
            
        
        elif imaginary_count == 1 and isconverged:
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
    
    # --- New method added ---
    def write_xyz(self, filename, geometry):
        """Write a single geometry to an XYZ file"""
        with open(filename, 'w') as outfile:
            outfile.write(f"{len(geometry)}\n")
            outfile.write("Terminal structure\n") # Comment line
            for i in range(len(geometry)):
                outfile.write(f"{self.element_list[i]}  {' '.join(map(str, geometry[i]))}\n")
    # --- End of new method ---

    def _get_irc_method_class(self):
        """Get the appropriate IRC method class based on method name
        
        Returns
        -------
        class
            IRC method class (LQA, RK4, ModeKill, Euler, or DVV)
        """
        method_map = {
            "LQA": LQA,
            "RK4": RK4,
            "MODEKILL": ModeKill,
            "EULER": Euler,
            "DVV": DVV,
            "HPC": HPC
        }
        
        method_key = self.method.upper()
        # Default to LQA if method is not recognized
        if method_key not in method_map:
            print(f"Unexpected method '{self.method}'. (default method is LQA.)")
            return LQA
            
        return method_map[method_key]
    
    def _run_single_irc(self, directory, initial_geometry):
        """Run a single IRC calculation
        
        Parameters
        ----------
        directory : str
            Directory to store results
        initial_geometry : numpy.ndarray
            Initial geometry coordinates
            
        Returns
        -------
        object
            IRC method instance
        """
        
        
        MethodClass = self._get_irc_method_class()
        
        kill_inds = None # Initialize kill_inds
        if len(self.method_options) > 0 and MethodClass == ModeKill:
            # Pass kill_inds if specified in method options
            try:
                kill_inds = [int(idx) for idx in self.method_options[0].split(",")]
                print(f"Using specified kill_inds: {kill_inds}")
            except ValueError:
                print("Invalid kill_inds format. It should be a comma-separated list of integers.")
                kill_inds = None
        
        irc_instance = MethodClass(
            self.element_list, 
            self.electric_charge_and_multiplicity, 
            self.FC_count, 
            directory, 
            self.final_directory, 
            self.force_data, 
            max_step=self.max_step, 
            step_size=self.step_size, 
            init_coord=initial_geometry, 
            init_hess=self.hessian, 
            calc_engine=self.QM_interface,
            xtb_method=self.xtb_method,
            kill_inds=kill_inds
        )
        
        irc_instance.run()
        return irc_instance
        
    def _run_irc(self):
        """Run IRC calculation in both forward and backward directions"""
        # Forward direction (from TS to products)
        print("Forward IRC")
        init_geom = self.geom_num_list + self.initial_step
        
        # Create forward direction directory
        fwd_dir = os.path.join(self.file_directory, "irc_forward")
        os.makedirs(fwd_dir, exist_ok=True)
        
        # Run forward IRC
        self._run_single_irc(fwd_dir, init_geom)
        
        # Backward direction (from TS to reactants)
        print("Backward IRC")
        init_geom = self.geom_num_list - self.initial_step
        
        # Create backward direction directory
        bwd_dir = os.path.join(self.file_directory, "irc_backward")
        os.makedirs(bwd_dir, exist_ok=True)
        
        # Run backward IRC
        self._run_single_irc(bwd_dir, init_geom)
        
        # Combine XYZ files from forward and backward directions
        self.combine_xyz_files(fwd_dir, bwd_dir)
        
        # Combine forward and backward CSV data into a single file
        self.combine_csv_data(fwd_dir, bwd_dir)
    
    def _run_meta_irc(self):
        """Run meta-IRC calculation in a single direction"""
        init_geom = self.geom_num_list - self.initial_step
        self._run_single_irc(self.file_directory, init_geom)
    
    def calc_IRCpath(self):
        """Calculate IRC path in forward and/or backward directions"""
        print("IRC carry out...")
        
        if self.IRC_flag:
            self._run_irc()
        else:
            self._run_meta_irc()
            # --- Added for terminal structure output (meta-IRC) ---
            meta_irc_xyz_file = os.path.join(self.file_directory, "irc_structures.xyz")
            if os.path.exists(meta_irc_xyz_file):
                try:
                    meta_irc_geometry_list, _, _ = traj2list(meta_irc_xyz_file, [0, 1])
                    if meta_irc_geometry_list: # Check if list is not empty
                        terminal_geom_meta = meta_irc_geometry_list[-1]
                        outfile_name_meta = os.path.join(self.file_directory, f"{self.fin_xyz_base}_irc_endpoint_1.xyz")
                        self.write_xyz(outfile_name_meta, terminal_geom_meta)
                        self.terminal_struct_paths = [os.path.abspath(outfile_name_meta)]
                        print(f"Meta-IRC terminal structure saved to {outfile_name_meta}")
                except Exception as e:
                    print(f"Error processing meta-IRC terminal structure: {e}")
            # --- End of addition ---
        
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
        bwd_csv_path = os.path.join(bwd_dir, "irc_energies_gradients.csv")
        if os.path.exists(bwd_csv_path):
            with open(bwd_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                try:
                    next(reader)  # Skip header
                    for row in reader:
                        bwd_data.append(row)
                except StopIteration:
                    print("Warning: Backward CSV file is empty.")
        else:
            print(f"Warning: Backward CSV file not found at {bwd_csv_path}")

        # Read forward data
        fwd_data = []
        fwd_csv_path = os.path.join(fwd_dir, "irc_energies_gradients.csv")
        if os.path.exists(fwd_csv_path):
            with open(fwd_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                try:
                    next(reader)  # Skip header
                    for row in reader:
                        fwd_data.append(row)
                except StopIteration:
                    print("Warning: Forward CSV file is empty.")
        else:
            print(f"Warning: Forward CSV file not found at {fwd_csv_path}")
        
        # Prepare TS point data
        ts_data = [0, self.init_e, self.init_B_e, np.sqrt((self.init_g**2).mean()), np.sqrt((self.init_B_g**2).mean())]
        
        # Write combined data
        with open(combined_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 'RMS Gradient', 'RMS Bias Gradient'])
            
            # Write reversed backward data with negative step numbers
            for i, row in enumerate(reversed(bwd_data)):
                try:
                    step = -int(row[0])
                    writer.writerow([step, row[1], row[2], row[3], row[4]])
                except (IndexError, ValueError) as e:
                    print(f"Warning: Skipping malformed row in backward CSV data: {row} ({e})")
            
            # Write TS point (step 0)
            writer.writerow(ts_data)
            
            # Write forward data with positive step numbers
            for row in fwd_data:
                try:
                    writer.writerow(row)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Skipping malformed row in forward CSV data: {row} ({e})")

        print(f"Combined IRC data saved to {combined_csv}")

    def combine_xyz_files(self, fwd_dir, bwd_dir):
        """Combine forward and backward XYZ structures into a single XYZ file
        and output terminal structures.
        
        Parameters
        ----------
        fwd_dir : str
            Forward directory path
        bwd_dir : str
            Backward directory path
        """
        # Path for combined XYZ file
        combined_xyz = os.path.join(self.file_directory, "irc_combined_path.xyz")
        
        # Find all XYZ files in forward and backward directories
        fwd_xyz_file = os.path.join(fwd_dir, "irc_structures.xyz")
        bwd_xyz_file = os.path.join(bwd_dir, "irc_structures.xyz")
        
        fwd_irc_geometry_list = []
        if os.path.exists(fwd_xyz_file):
            try:
                fwd_irc_geometry_list, _, _ = traj2list(fwd_xyz_file, [0, 1])
            except Exception as e:
                print(f"Error reading forward XYZ file: {e}")
        else:
            print(f"Warning: Forward XYZ file not found at {fwd_xyz_file}")

        bwd_irc_geometry_list = []
        if os.path.exists(bwd_xyz_file):
            try:
                bwd_irc_geometry_list, _, _ = traj2list(bwd_xyz_file, [0, 1])
            except Exception as e:
                print(f"Error reading backward XYZ file: {e}")
        else:
            print(f"Warning: Backward XYZ file not found at {bwd_xyz_file}")
        

      
        self.terminal_struct_paths = [] # Clear any previous paths
        if fwd_irc_geometry_list: # Check if list is not empty
            terminal_geom_fwd = fwd_irc_geometry_list[-1]
            outfile_name_fwd = os.path.join(self.file_directory, f"{self.fin_xyz_base}_irc_endpoint_1.xyz")
            self.write_xyz(outfile_name_fwd, terminal_geom_fwd)
            self.terminal_struct_paths.append(os.path.abspath(outfile_name_fwd))
            print(f"Forward terminal structure saved to {outfile_name_fwd}")

        if bwd_irc_geometry_list: # Check if list is not empty
            terminal_geom_bwd = bwd_irc_geometry_list[-1]
            outfile_name_bwd = os.path.join(self.file_directory, f"{self.fin_xyz_base}_irc_endpoint_2.xyz")
            self.write_xyz(outfile_name_bwd, terminal_geom_bwd)
            self.terminal_struct_paths.append(os.path.abspath(outfile_name_bwd))
            print(f"Backward terminal structure saved to {outfile_name_bwd}")
      

        fwd_irc_geometry_list = fwd_irc_geometry_list[::-1]  # Reverse forward list

        # TS structure from the final directory
        ts_xyz_file_path = glob.glob(os.path.join(self.final_directory, "*.xyz"))
        
        # Write combined XYZ file
        with open(combined_xyz, 'w') as outfile:
            # Forward structures (from last to first)
            for xyz in fwd_irc_geometry_list:
                outfile.write(f"{len(xyz)}\n")
                outfile.write("\n")
                for i in range(len(xyz)):
                    outfile.write(self.element_list[i] + "  " + " ".join(map(str, xyz[i])) + "\n")
                
            # TS structure
            if ts_xyz_file_path:
                with open(ts_xyz_file_path[0], 'r') as infile:
                    outfile.write(infile.read())
            else:
                print(f"Warning: TS XYZ file not found in {self.final_directory}")
            
            # Backward structures (from first to last)
            for xyz in bwd_irc_geometry_list:
                outfile.write(f"{len(xyz)}\n")
                outfile.write("\n")
                for i in range(len(xyz)):
                    outfile.write(self.element_list[i] + "  " + " ".join(map(str, xyz[i])) + "\n")
                
        
        print(f"Combined IRC path saved to {combined_xyz}")

    def run(self):
        """Main function to run IRC calculation"""
        # Check if starting point is a saddle point and get initial displacement
        self.initial_step, self.IRC_flag, self.geom_num_list, finish_flag = self.saddle_check()
        
        if finish_flag:
            print("IRC calculation failed.")
            self.terminal_struct_paths = []
            return
            
        # Calculate the IRC path
        self.calc_IRCpath()
        
        print("IRC calculation is finished.")
        # self.terminal_struct_paths can be accessed for terminal structure file paths.
        return