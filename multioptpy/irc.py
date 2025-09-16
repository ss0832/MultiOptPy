import numpy as np
import os
import glob
import csv

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.parameter import atomic_mass
from multioptpy.calc_tools import Calculationtools
from multioptpy.IRC.lqa import LQA
from multioptpy.IRC.rk4 import RK4
from multioptpy.IRC.dvv import DVV
from multioptpy.IRC.converge_criteria import convergence_check

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
            
        self.file_directory = directory
        self.final_directory = final_directory
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
        if imaginary_count == 1 and isconverged:
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