import sys
import os
import copy
import glob
import itertools
import datetime


import numpy as np

from multioptpy.optimizer import CalculateMoveVector
from multioptpy.Visualization.visualization import Graph
from multioptpy.fileio import FileIO
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.interface import force_data_parser
from multioptpy.ModelHessian.approx_hessian import ApproxHessian
from multioptpy.PESAnalyzer.cmds_analysis import CMDSPathAnalysis
from multioptpy.PESAnalyzer.pca_analysis import PCAPathAnalysis
from multioptpy.PESAnalyzer.koopman_analysis import KoopmanAnalyzer
from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import CalculationStructInfo, Calculationtools
from multioptpy.Constraint.constraint_condition import ProjectOutConstrain
from multioptpy.irc import IRC
from multioptpy.Utils.bond_connectivity import judge_shape_condition
from multioptpy.Utils.oniom import separate_high_layer_and_low_layer, specify_link_atom_pairs, link_number_high_layer_and_low_layer
from multioptpy.Utils.symmetry_analyzer import analyze_symmetry
from multioptpy.Thermo.normal_mode_analyzer import MolecularVibrations

# Responsibility 1: Holds the "Configuration" (immutable)
class OptimizationConfig:
    """
    Holds all "settings" that do not change during the run.
    Initialized from 'args'.
    """
    def __init__(self, args):
        # Constants like UVL
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol

        # Port the logic from _set_convergence_criteria
        self._set_convergence_criteria(args)

        # Port all "immutable" settings from _initialize_variables
        self.microiter_num = 100
        self.args = args  # Keep a reference to args
        self.FC_COUNT = args.calc_exact_hess
        self.temperature = 0.0
        self.CMDS = args.cmds
        self.PCA = args.pca
        self.DELTA = "x" if args.DELTA == "x" else float(args.DELTA)
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.NSTEP = args.NSTEP
        self.BASIS_SET = args.basisset
        self.FUNCTIONAL = args.functional
        self.excited_state = args.excited_state
        
        # Port the logic from _check_sub_basisset
        self._check_sub_basisset(args)
        
        self.mFC_COUNT = args.calc_model_hess
        self.DC_check_dist = float(args.dissociate_check)
        self.unrestrict = args.unrestrict
        self.irc = args.intrinsic_reaction_coordinates
        self.othersoft = args.othersoft
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        self.shape_conditions = args.shape_conditions
        self.oniom = args.oniom_flag
        self.use_model_hessian = args.use_model_hessian
        self.sqm1 = args.sqm1
        self.sqm2 = args.sqm2
        self.freq_analysis = args.frequency_analysis
        self.thermo_temperature = float(args.temperature)
        self.thermo_pressure = float(args.pressure)
        self.dft_grid = int(args.dft_grid)
        self.max_trust_radius = args.max_trust_radius
        self.min_trust_radius = args.min_trust_radius
        self.software_path_file = args.software_path_file
        self.koopman_analysis = args.koopman
        self.detect_negative_eigenvalues = args.detect_negative_eigenvalues

    def _set_convergence_criteria(self, args):
        # Original _set_convergence_criteria method code
        if args.tight_convergence_criteria and not args.loose_convergence_criteria:
            self.MAX_FORCE_THRESHOLD = 0.000015
            self.RMS_FORCE_THRESHOLD = 0.000010
            self.MAX_DISPLACEMENT_THRESHOLD = 0.000060
            self.RMS_DISPLACEMENT_THRESHOLD = 0.000040
        elif not args.tight_convergence_criteria and args.loose_convergence_criteria:
            self.MAX_FORCE_THRESHOLD = 0.0030
            self.RMS_FORCE_THRESHOLD = 0.0020
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0100
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0070
        else:
            self.MAX_FORCE_THRESHOLD = 0.0003
            self.RMS_FORCE_THRESHOLD = 0.0002
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0015
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0010

    def _check_sub_basisset(self, args):
        # Original _check_sub_basisset method code
        if len(args.sub_basisset) % 2 != 0:
            print("invalid input (-sub_bs)")
            sys.exit(0)
        self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        
        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET)
                for j in range(int(len(args.sub_basisset) / 2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2 * j]] = args.sub_basisset[2 * j + 1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
            else:
                self.SUB_BASIS_SET = {"default": self.BASIS_SET}
        else:
            self.SUB_BASIS_SET = ""
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET += "\nassign " + str(self.BASIS_SET) + "\n"
                for j in range(int(len(args.sub_basisset) / 2)):
                    self.SUB_BASIS_SET += "assign " + args.sub_basisset[2 * j] + " " + args.sub_basisset[2 * j + 1] + "\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
        
        if len(args.effective_core_potential) % 2 != 0:
            print("invaild input (-ecp)")
            sys.exit(0)

        if args.pyscf:
            self.ECP = {}
            if len(args.effective_core_potential) > 0:
                for j in range(int(len(args.effective_core_potential)/2)):
                    self.ECP[args.effective_core_potential[2*j]] = args.effective_core_potential[2*j+1]
        else:
            self.ECP = ""

# Responsibility 2: Manages the "State" (mutable)
class OptimizationState:
    """
    Holds all "state" variables that change during the optimization loop.
    """
    def __init__(self, element_list):
        natom = len(element_list)
        
        # Current step state
        self.iter = 0
        self.e = None  # Hartree
        self.B_e = None  # Hartree
        self.g = None  # Hartree/Bohr
        self.B_g = None  # Hartree/Bohr
        self.geom_num_list = None  # Bohr
        self.Model_hess = np.eye(natom * 3) # Model_hess is treated as state
        self.element_list = element_list

        # Previous step state
        self.pre_e = 0.0
        self.pre_B_e = 0.0
        self.pre_geom = np.zeros((natom, 3), dtype="float64")
        self.pre_g = np.zeros((natom, 3), dtype="float64")
        self.pre_B_g = np.zeros((natom, 3), dtype="float64")
        self.pre_move_vector = np.zeros((natom, 3), dtype="float64")

        # Plotting / result lists
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.BIAS_ENERGY_LIST_FOR_PLOTTING = []
        self.NUM_LIST = []
        self.grad_list = []
        self.bias_grad_list = []
        self.cos_list = [] # Initialized properly in Optimize class

        # Final result placeholders
        self.final_file_directory = None
        self.final_geometry = None
        self.final_energy = None
        self.final_bias_energy = None
        self.bias_pot_params_grad_list = None
        self.bias_pot_params_grad_name_list = None

        # Flags
        self.DC_check_flag = False
        self.optimized_flag = False
        self.exit_flag = False

# Responsibility 3: Performs the "Execution" (main logic)
class Optimize:
    """
    Main execution (Runner) class.
    It holds the Config, creates and updates the State,
    and runs the main optimization logic.
    """
    def __init__(self, args):
        # 1. Set the Configuration (immutable)
        self.config = OptimizationConfig(args)
        
        # 2. State will be created freshly inside the run() method for each job.
        self.state = None 

        # 3. Helper instances and job-specific variables
        self.BPA_FOLDER_DIRECTORY = None
        self.START_FILE = None
        self.element_list = None # Will be set in run()
        self.CalcBiaspot = None # Shared helper
        self.SP = None # Shared helper

        # 4. Final results (for external access, mirrors original design)
        self.final_file_directory = None
        self.final_geometry = None
        self.final_energy = None
        self.final_bias_energy = None
        self.irc_terminal_struct_paths = []
        self.optimized_struct_file = None
        self.traj_file = None
        self.symmetry = None

    # --- Helper Methods ---
    # (Ported from the original class)
    # These methods must now read from self.config
    # and read/write from self.state.

    def _make_init_directory(self, file):
        """
        Create initial directory for optimization results.
        Uses self.config to build the path.
        """
        self.START_FILE = file
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]
        date = datetime.datetime.now().strftime("%Y_%m_%d")
        base_dir = f"{date}/{self.START_FILE[:-4]}_OPT_"

        if self.config.othersoft != "None":
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}ASE_{timestamp}/"
        elif self.config.sqm2:
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}SQM2_{timestamp}/"
        elif self.config.sqm1:
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}SQM1_{timestamp}/"
        elif self.config.args.usextb == "None" and self.config.args.usedxtb == "None":
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}{self.config.FUNCTIONAL}_{self.config.BASIS_SET}_{timestamp}/"
        else:
            method = self.config.args.usedxtb if self.config.args.usedxtb != "None" else self.config.args.usextb
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}{method}_{timestamp}/"
        
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)

    def _save_input_data(self):
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.config.args))) # Read from config
        return

    def _constrain_flag_check(self, force_data):
        # (This method is pure, no changes needed)
        if len(force_data["projection_constraint_condition_list"]) > 0:
            projection_constrain = True
        else:
            projection_constrain = False
            
        if len(force_data["fix_atoms"]) == 0:
            allactive_flag = True
        else:
            allactive_flag = False

        if "x" in force_data["projection_constraint_condition_list"] or "y" in force_data["projection_constraint_condition_list"] or "z" in force_data["projection_constraint_condition_list"]:
            allactive_flag = False
        
        return projection_constrain, allactive_flag

    def _init_projection_constraint(self, PC, geom_num_list, iter, projection_constrain, hessian=None):
        # (This method is pure, no changes needed)
        if iter == 0:
            if projection_constrain:
                PC.initialize(geom_num_list, hessian=hessian)
            else:
                pass
            return PC
        else:
            return PC

    def _save_init_geometry(self, geom_num_list, element_list, allactive_flag):
        # (This method is pure, no changes needed)
        if allactive_flag:
            initial_geom_num_list = geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
            pre_geom = initial_geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
        else:
            initial_geom_num_list = geom_num_list 
            pre_geom = initial_geom_num_list 
        
        return initial_geom_num_list, pre_geom

    def _calc_eff_hess_for_fix_atoms_and_set_hess(self, allactive_flag, force_data, BPA_hessian, n_fix, optimizer_instances, geom_num_list, B_g, g, projection_constrain, PC):
        # (Reads self.state.Model_hess, self.config.FC_COUNT, etc.)
        if not allactive_flag:
            fix_num = []
            for fnum in force_data["fix_atoms"]:
                fix_num.extend([3*(fnum-1)+0, 3*(fnum-1)+1, 3*(fnum-1)+2])
            fix_num = np.array(fix_num, dtype="int64")
            #effective hessian
            tmp_fix_hess = self.state.Model_hess[np.ix_(fix_num, fix_num)] + np.eye((3*n_fix)) * 1e-10
            inv_tmp_fix_hess = np.linalg.pinv(tmp_fix_hess)
            tmp_fix_bias_hess = BPA_hessian[np.ix_(fix_num, fix_num)] + np.eye((3*n_fix)) * 1e-10
            inv_tmp_fix_bias_hess = np.linalg.pinv(tmp_fix_bias_hess)
            BPA_hessian -= np.dot(BPA_hessian[:, fix_num], np.dot(inv_tmp_fix_bias_hess, BPA_hessian[fix_num, :]))
        
        for i in range(len(optimizer_instances)):
                
            if projection_constrain:
                if np.all(np.abs(BPA_hessian) < 1e-20):
                    proj_bpa_hess = PC.calc_project_out_hess(geom_num_list, B_g - g, BPA_hessian)
                else:
                    proj_bpa_hess = BPA_hessian
                optimizer_instances[i].set_bias_hessian(proj_bpa_hess)
            else:
                optimizer_instances[i].set_bias_hessian(BPA_hessian)
            
            if self.state.iter % self.config.FC_COUNT == 0 or (self.config.use_model_hessian is not None and self.state.iter % self.config.mFC_COUNT == 0):
        
                if not allactive_flag:
                    self.state.Model_hess -= np.dot(self.state.Model_hess[:, fix_num], np.dot(inv_tmp_fix_hess, self.state.Model_hess[fix_num, :]))
                
                
                if projection_constrain:
                    proj_model_hess = PC.calc_project_out_hess(geom_num_list, g, self.state.Model_hess)
                    optimizer_instances[i].set_hessian(proj_model_hess)
                else:
                    optimizer_instances[i].set_hessian(self.state.Model_hess)
        
        return optimizer_instances
        
    def _apply_projection_constraints(self, projection_constrain, PC, geom_num_list, g, B_g):
        # (This method is pure, no changes needed)
        if projection_constrain:
            g = copy.deepcopy(PC.calc_project_out_grad(geom_num_list, g))
            proj_d_B_g = copy.deepcopy(PC.calc_project_out_grad(geom_num_list, B_g - g))
            B_g = copy.deepcopy(g + proj_d_B_g)
        
        return g, B_g, PC

    def _zero_fixed_atom_gradients(self, allactive_flag, force_data, g, B_g):
        # (This method is pure, no changes needed)
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                g[j-1] = copy.deepcopy(g[j-1]*0.0)
                B_g[j-1] = copy.deepcopy(B_g[j-1]*0.0)
        
        return g, B_g

    def _project_out_translation_rotation(self, new_geometry, geom_num_list, allactive_flag):
        # (Reads self.config.bohr2angstroms)
        if allactive_flag:
            # Convert to Bohr, apply Kabsch alignment algorithm, then convert back
            aligned_geometry, _ = Calculationtools().kabsch_algorithm(
                new_geometry/self.config.bohr2angstroms, geom_num_list)
            aligned_geometry *= self.config.bohr2angstroms
            return aligned_geometry
        else:
            # If not all atoms are active, return the original geometry
            return new_geometry

    def _apply_projection_constraints_to_geometry(self, projection_constrain, PC, new_geometry, hessian=None):
        # (Reads self.config.bohr2angstroms)
        if projection_constrain:
            tmp_new_geometry = new_geometry / self.config.bohr2angstroms
            adjusted_geometry = PC.adjust_init_coord(tmp_new_geometry, hessian=hessian) * self.config.bohr2angstroms
            return adjusted_geometry, PC
        
        return new_geometry, PC

    def _reset_fixed_atom_positions(self, new_geometry, initial_geom_num_list, allactive_flag, force_data):
        # (Reads self.config.bohr2angstroms)
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                new_geometry[j-1] = copy.deepcopy(initial_geom_num_list[j-1]*self.config.bohr2angstroms)
        
        return new_geometry

    def _initialize_optimization_tools(self, FIO, force_data):
        """
        Initializes all tools needed for the optimization loop.
        This replaces the old _initialize_optimization_variables.
        It assumes self.state is already created.
        """
        # Load modules
        Calculation, xtb_method = self._import_calculation_module()
        self._save_input_data() # Save input.txt
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        
        # Get atom info
        file_directory, electric_charge_and_multiplicity, element_list = self.write_input_files(FIO)
        self.element_list = element_list # Store on self for helper methods
        self.state.element_list = element_list # Store in state
        
        element_number_list = np.array([element_number(elem) for elem in element_list], dtype="int")
        natom = len(element_list)
        
        # Constraint setup
        PC = ProjectOutConstrain(force_data["projection_constraint_condition_list"], 
                                  force_data["projection_constraint_atoms"], 
                                  force_data["projection_constraint_constant"])
        projection_constrain, allactive_flag = self._constrain_flag_check(force_data)
        n_fix = len(force_data["fix_atoms"])

        # Bias potential and calculation setup
        self.CalcBiaspot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        self.SP = self.setup_calculation(Calculation) # SP is self.SP
        
        # Move vector calculation
        CMV = CalculateMoveVector(self.config.DELTA, element_list, self.config.args.saddle_order, 
                                  self.config.FC_COUNT, self.config.temperature, self.config.use_model_hessian, 
                                  max_trust_radius=self.config.max_trust_radius, min_trust_radius=self.config.min_trust_radius)
        optimizer_instances = CMV.initialization(force_data["opt_method"])
        
        # Check optimizer compatibility
        for i in range(len(optimizer_instances)):
            if CMV.newton_tag[i] is False and self.config.FC_COUNT > 0 and not "eigvec" in force_data["projection_constraint_condition_list"]:
                print("Error: This optimizer method does not support exact Hessian calculations.")
                print("Please either choose a different optimizer or set FC_COUNT=0 to disable exact Hessian calculations.")
                sys.exit(0)
        
        # Initialize optimizer instances
        for i in range(len(optimizer_instances)):
            optimizer_instances[i].set_hessian(self.state.Model_hess) # From state
            if self.config.DELTA != "x":
                optimizer_instances[i].DELTA = self.config.DELTA
                
        if self.config.koopman_analysis:
            KA = KoopmanAnalyzer(natom, file_directory=self.BPA_FOLDER_DIRECTORY)
        else:
            KA = None

        # Pack and return all initialized tools
        tools = {
            'Calculation': Calculation, 'xtb_method': xtb_method,
            'SP': self.SP, 'CMV': CMV, 'optimizer_instances': optimizer_instances,
            'FIO': FIO, 'G': G, 'file_directory': file_directory,
            'element_number_list': element_number_list, 'natom': natom,
            'electric_charge_and_multiplicity': electric_charge_and_multiplicity,
            'PC': PC, 'projection_constrain': projection_constrain,
            'allactive_flag': allactive_flag, 'force_data': force_data, 'n_fix': n_fix,
            'KA': KA
        }
        return tools

    def check_negative_eigenvalues(self, geom_num_list, hessian):
        # (This method is pure, no changes needed)
        proj_hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(hessian, geom_num_list, geom_num_list, display_eigval=False)
        if proj_hessian is not None:
            eigvals = np.linalg.eigvalsh(proj_hessian)
            if np.any(eigvals < -1e-10):
                print("Notice: Negative eigenvalues detected.")
                return True
        return False

    def judge_early_stop_due_to_no_negative_eigenvalues(self, geom_num_list, hessian):
        # (Reads self.config)
        if self.config.detect_negative_eigenvalues and self.config.FC_COUNT > 0:
            negative_eigenvalues_detected = self.check_negative_eigenvalues(geom_num_list, hessian)
            if not negative_eigenvalues_detected and self.config.args.saddle_order > 0:
                print("No negative eigenvalues detected while saddle_order > 0. Stopping optimization.")
                with open(self.BPA_FOLDER_DIRECTORY+"no_negative_eigenvalues_detected.txt", "w") as f:
                    f.write("No negative eigenvalues detected while saddle_order > 0. Stopping optimization.")
                return True
        return False

    def optimize(self):
        # 1. Initialize State.
        #    write_input_files needs FIO, FIO needs BPA_FOLDER_DIRECTORY.
        #    This is complex. Let's initialize FIO and element_list first.
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        
        # This will read the file and set self.element_list
        file_directory, electric_charge_and_multiplicity, element_list = self.write_input_files(FIO)
        self.element_list = element_list
        
        # Now we can create the State
        self.state = OptimizationState(element_list)
        self.state.cos_list = [[] for i in range(len(force_data_parser(self.config.args)["geom_info"]))] # Init cos_list
        
        # 2. Initialize all other tools, passing FIO
        force_data = force_data_parser(self.config.args)
        tools = self._initialize_optimization_tools(FIO, force_data)

        # 3. Unpack tools into local variables for the loop
        # (This is better than the giant vars_dict at the end)
        xtb_method = tools['xtb_method']
        SP = tools['SP']
        CMV = tools['CMV']
        optimizer_instances = tools['optimizer_instances']
        FIO = tools['FIO']
        G = tools['G']
        file_directory = tools['file_directory']
        element_number_list = tools['element_number_list']
        electric_charge_and_multiplicity = tools['electric_charge_and_multiplicity']
        PC = tools['PC']
        projection_constrain = tools['projection_constrain']
        allactive_flag = tools['allactive_flag']
        force_data = tools['force_data']
        n_fix = tools['n_fix']
        KA = tools['KA']
        
        # 4. Main Optimization Loop
        for iter in range(self.config.NSTEP):
            
            self.state.iter = iter
            self.state.exit_flag = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")
            if self.state.exit_flag:
                break
            
            self.state.exit_flag = judge_shape_condition(self.state.geom_num_list, self.config.shape_conditions)
            if self.state.exit_flag:
                break
                
            print(f"\n# ITR. {iter}\n")
            
            # --- Perform Single Point Calculation ---
            SP.Model_hess = copy.deepcopy(self.state.Model_hess)
            e, g, geom_num_list, exit_flag = SP.single_point(file_directory, element_number_list, iter, electric_charge_and_multiplicity, xtb_method)
            
            # Update state
            self.state.e = e
            self.state.g = g
            self.state.geom_num_list = geom_num_list
            self.state.exit_flag = exit_flag
            self.state.Model_hess = copy.deepcopy(SP.Model_hess)
            
            if self.state.exit_flag:
                break
                
            # --- Update Model Hessian (if needed) ---
            if iter % self.config.mFC_COUNT == 0 and self.config.use_model_hessian is not None and self.config.FC_COUNT < 1:
                SP.Model_hess = ApproxHessian().main(geom_num_list, self.element_list, g, self.config.use_model_hessian)
                self.state.Model_hess = SP.Model_hess 
                
            if iter == 0:
                initial_geom_num_list, pre_geom = self._save_init_geometry(geom_num_list, self.element_list, allactive_flag)
                # Save initial geometry to state
                self.state.pre_geom = pre_geom

            # --- Bias Potential Calculation ---
            _, B_e, B_g, BPA_hessian = self.CalcBiaspot.main(e, g, geom_num_list, self.element_list, force_data, self.state.pre_B_g, iter, initial_geom_num_list)
            # Update state
            self.state.B_e = B_e
            self.state.B_g = B_g

            # --- Check Eigenvalues (if first iter) ---
            Hess = BPA_hessian + self.state.Model_hess
            if iter == 0:
                if self.judge_early_stop_due_to_no_negative_eigenvalues(geom_num_list, Hess):
                    break
            
            # --- Constraints ---
            PC = self._init_projection_constraint(PC, geom_num_list, iter, projection_constrain, hessian=Hess)
            optimizer_instances = self._calc_eff_hess_for_fix_atoms_and_set_hess(allactive_flag, force_data, BPA_hessian, n_fix, optimizer_instances, geom_num_list, B_g, g, projection_constrain, PC)
            
            if not allactive_flag:
                B_g = copy.deepcopy(self.calc_fragement_grads(B_g, force_data["opt_fragment"]))
                g = copy.deepcopy(self.calc_fragement_grads(g, force_data["opt_fragment"]))
            
            self.save_tmp_energy_profiles(iter, e, g, B_g)
            
            g, B_g, PC = self._apply_projection_constraints(projection_constrain, PC, geom_num_list, g, B_g)
            g, B_g = self._zero_fixed_atom_gradients(allactive_flag, force_data, g, B_g)

            # Update state with final gradients for this step
            self.state.g = g
            self.state.B_g = B_g
            
            if self.config.koopman_analysis:
                _ = KA.run(iter, geom_num_list, B_g, self.element_list)

            # --- Calculate Move Vector ---
            new_geometry, move_vector, optimizer_instances = CMV.calc_move_vector(
                iter, geom_num_list, B_g, self.state.pre_B_g, self.state.pre_geom, B_e, self.state.pre_B_e,
                self.state.pre_move_vector, initial_geom_num_list, g, self.state.pre_g, optimizer_instances, projection_constrain)
            
            # --- Post-step Geometry Adjustments ---
            new_geometry = self._project_out_translation_rotation(new_geometry, geom_num_list, allactive_flag)
            new_geometry, PC = self._apply_projection_constraints_to_geometry(projection_constrain, PC, new_geometry, hessian=Hess)

            # --- Update State Lists ---
            self.state.ENERGY_LIST_FOR_PLOTTING.append(e * self.config.hartree2kcalmol)
            self.state.BIAS_ENERGY_LIST_FOR_PLOTTING.append(B_e * self.config.hartree2kcalmol)
            self.state.NUM_LIST.append(int(iter))
            
            self.geom_info_extract(force_data, file_directory, B_g, g) # This updates self.state.cos_list
            
            if self.state.iter == 0:
                displacement_vector = move_vector
            else:
                displacement_vector = new_geometry / self.config.bohr2angstroms - geom_num_list
            
            # --- Check Convergence ---
            converge_flag, max_displacement_threshold, rms_displacement_threshold = self._check_converge_criteria(B_g, displacement_vector)
            self.print_info(e, B_e, B_g, displacement_vector, self.state.pre_e, self.state.pre_B_e, max_displacement_threshold, rms_displacement_threshold)
            
            self.state.grad_list.append(self.calculate_rms_safely(g))
            self.state.bias_grad_list.append(self.calculate_rms_safely(B_g))
            
            new_geometry = self._reset_fixed_atom_positions(new_geometry, initial_geom_num_list, allactive_flag, force_data)
            
            # --- Dissociation Check ---
            DC_exit_flag = self.dissociation_check(new_geometry, self.element_list)

            if converge_flag:
                if projection_constrain and iter == 0:
                    pass
                else:
                    self.state.optimized_flag = True
                    print("\n=====================================================")
                    print("converged!!!")
                    print("=====================================================")
                    break
            
            if DC_exit_flag:
                self.state.DC_check_flag = True
                break
            
            # --- Save State for Next Iteration ---
            self.state.pre_B_e = B_e
            self.state.pre_e = e
            self.state.pre_B_g = B_g
            self.state.pre_g = g
            self.state.pre_geom = geom_num_list
            self.state.pre_move_vector = move_vector
            
            # --- Write Next Input File ---
            geometry_list = FIO.print_geometry_list(new_geometry, self.element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            
        else: # Loop ended (no break)
            self.state.optimized_flag = False
            print("Reached maximum number of iterations. This is not converged.")
            with open(self.BPA_FOLDER_DIRECTORY+"not_converged.txt", "w") as f:
                f.write("Reached maximum number of iterations. This is not converged.")

        # --- 5. Post-Optimization Analysis ---
        
        # Check if exact hessian is already computed.
        if self.config.FC_COUNT == -1:
            exact_hess_flag = False
        elif self.state.iter % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
            exact_hess_flag = True
        else:
            exact_hess_flag = False
        
        if self.state.DC_check_flag:
            print("Dissociation is detected. Optimization stopped.")
            with open(self.BPA_FOLDER_DIRECTORY+"dissociation_is_detected.txt", "w") as f:
                f.write("Dissociation is detected. Optimization stopped.")

        if self.config.freq_analysis and not self.state.exit_flag and not self.state.DC_check_flag:
            self._perform_vibrational_analysis(SP, geom_num_list, self.element_list, initial_geom_num_list, force_data, exact_hess_flag, file_directory, iter, electric_charge_and_multiplicity, xtb_method, e)

        # --- 6. Finalize and Save Results ---
        self._finalize_optimization(FIO, G, self.state.grad_list, self.state.bias_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, self.state.exit_flag)
        
        # Copy final results from state to self
        self._copy_final_results_from_state()
        return

    def _perform_vibrational_analysis(self, SP, geom_num_list, element_list, initial_geom_num_list, force_data, exact_hess_flag, file_directory, iter, electric_charge_and_multiplicity, xtb_method, e):
        # (Reads self.state, self.config)
        print("\n====================================================")
        print("Performing vibrational analysis...")
        print("====================================================\n")
        print("Is Exact Hessian calculated? : ", exact_hess_flag)
        
        if exact_hess_flag:
            g = np.zeros_like(geom_num_list, dtype="float64")
            exit_flag = False
        else:
            print("Calculate exact Hessian...")
            SP.hessian_flag = True
            e, g, geom_num_list, exit_flag = SP.single_point(file_directory, element_list, iter, electric_charge_and_multiplicity, xtb_method)
            SP.hessian_flag = False
            
        if exit_flag:
            print("Error: QM calculation failed.")
            return
            
        _, B_e, _, BPA_hessian = self.CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g="", iter=iter, initial_geom_num_list="")
        tmp_hess = copy.deepcopy(SP.Model_hess) # SP.Model_hess holds the latest hessian
        tmp_hess += BPA_hessian
        
        MV = MolecularVibrations(atoms=element_list, coordinates=geom_num_list, hessian=tmp_hess)
        results = MV.calculate_thermochemistry(e_tot=B_e, temperature=self.config.thermo_temperature, pressure=self.config.thermo_pressure)
        
        MV.print_thermochemistry(output_file=self.BPA_FOLDER_DIRECTORY+"/thermochemistry.txt")
        MV.print_normal_modes(output_file=self.BPA_FOLDER_DIRECTORY+"/normal_modes.txt")
        MV.create_vibration_animation(output_dir=self.BPA_FOLDER_DIRECTORY+"/vibration_animation")
        
        if not self.state.optimized_flag:
            print("Warning: Vibrational analysis was performed, but the optimization did not converge. The result of thermochemistry is useless.")
        
        return

    def _finalize_optimization(self, FIO, G, grad_list, bias_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, exit_flag):
        # (Writes to self.state)
        self._save_opt_results(FIO, G, grad_list, bias_grad_list, file_directory, force_data)
        
        self.state.bias_pot_params_grad_list = self.CalcBiaspot.bias_pot_params_grad_list
        self.state.bias_pot_params_grad_name_list = self.CalcBiaspot.bias_pot_params_grad_name_list
        self.state.final_file_directory = file_directory
        self.state.final_geometry = geom_num_list  # Bohr
        self.state.final_energy = e  # Hartree
        self.state.final_bias_energy = B_e  # Hartree
        
        if not exit_flag:
            self.symmetry = analyze_symmetry(self.element_list, self.state.final_geometry)
            self.state.symmetry = self.symmetry # Save to state too
            with open(self.BPA_FOLDER_DIRECTORY+"symmetry.txt", "w") as f:
                f.write(f"Symmetry of final structure: {self.symmetry}")
            print(f"Symmetry: {self.symmetry}")
    
    def _save_opt_results(self, FIO, G, grad_list, bias_grad_list, file_directory, force_data):
        # (Reads self.state)
        G.double_plot(self.state.NUM_LIST, self.state.ENERGY_LIST_FOR_PLOTTING, self.state.BIAS_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.state.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.state.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                G.single_plot(self.state.NUM_LIST, self.state.cos_list[num], file_directory, i)

        FIO.make_traj_file()
        FIO.argrelextrema_txt_save(self.state.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.state.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")

        self._save_energy_profiles()
        return

    def _copy_final_results_from_state(self):
        """Copy final results from the State object to the main Optimize object."""
        if self.state:
            self.final_file_directory = self.state.final_file_directory
            self.final_geometry = self.state.final_geometry
            self.final_energy = self.state.final_energy
            self.final_bias_energy = self.state.final_bias_energy
            self.symmetry = getattr(self.state, 'symmetry', None)
            
            # These were not in the original _finalize, but probably should be
            self.bias_pot_params_grad_list = self.state.bias_pot_params_grad_list
            self.bias_pot_params_grad_name_list = self.state.bias_pot_params_grad_name_list
            self.optimized_flag = self.state.optimized_flag

    def _check_converge_criteria(self, B_g, displacement_vector):
        # (Reads self.config)
        max_force = np.abs(B_g).max()
        max_force_threshold = self.config.MAX_FORCE_THRESHOLD
        
        rms_force = self.calculate_rms_safely(B_g)
        rms_force_threshold = self.config.RMS_FORCE_THRESHOLD
        
        delta_max_force_threshold = max(0.0, max_force_threshold -1 * max_force)
        delta_rms_force_threshold = max(0.0, rms_force_threshold -1 * rms_force)
        
        max_displacement = np.abs(displacement_vector).max()
        max_displacement_threshold = max(self.config.MAX_DISPLACEMENT_THRESHOLD, self.config.MAX_DISPLACEMENT_THRESHOLD + delta_max_force_threshold)
        rms_displacement = self.calculate_rms_safely(displacement_vector)
        rms_displacement_threshold = max(self.config.RMS_DISPLACEMENT_THRESHOLD, self.config.RMS_DISPLACEMENT_THRESHOLD + delta_rms_force_threshold)
        
        if max_force < max_force_threshold and rms_force < rms_force_threshold and max_displacement < max_displacement_threshold and rms_displacement < rms_displacement_threshold:
            return True, max_displacement_threshold, rms_displacement_threshold
        return False, max_displacement_threshold, rms_displacement_threshold
    
    def _import_calculation_module(self):
        # (Reads self.config)
        xtb_method = None
        if self.config.args.pyscf:
            from multioptpy.Calculator.pyscf_calculation_tools import Calculation
        elif self.config.sqm2:
            from multioptpy.Calculator.sqm2_calculation_tools import Calculation
            print("Use SQM2 potential.")
        elif self.config.sqm1:
            from multioptpy.Calculator.sqm1_calculation_tools import Calculation
        elif self.config.othersoft and self.config.othersoft != "None":
            if self.config.othersoft.lower() == "lj":
                from multioptpy.Calculator.lj_calculation_tools import Calculation
                print("Use Lennard-Jones cluster potential.")
            elif self.config.othersoft.lower() == "emt":
                from multioptpy.Calculator.emt_calculation_tools import Calculation
                print("Use ETM potential.")
            elif self.config.othersoft.lower() == "tersoff":
                from multioptpy.Calculator.tersoff_calculation_tools import Calculation
                print("Use Tersoff potential.")
            else:
                from multioptpy.Calculator.ase_calculation_tools import Calculation
                print("Use", self.config.othersoft)
                with open(self.BPA_FOLDER_DIRECTORY + "use_" + self.config.othersoft + ".txt", "w") as f:
                    f.write(self.config.othersoft + "\n")
                    f.write(self.config.BASIS_SET + "\n")
                    f.write(self.config.FUNCTIONAL + "\n")
        else:
            if self.config.args.usedxtb and self.config.args.usedxtb != "None":
                from multioptpy.Calculator.dxtb_calculation_tools import Calculation
                xtb_method = self.config.args.usedxtb
            elif self.config.args.usextb and self.config.args.usextb != "None":
                from multioptpy.Calculator.tblite_calculation_tools import Calculation
                xtb_method = self.config.args.usextb
            else:
                from multioptpy.Calculator.psi4_calculation_tools import Calculation
                
        return Calculation, xtb_method
    
    def setup_calculation(self, Calculation):
        # (Reads self.config, self.state)
        # Note: Model_hess is passed from state, but SP is re-created per job.
        # This assumes the initial Model_hess (eye) is what's needed.
        # This might be a flaw if SP needs the *current* Model_hess.
        # Let's assume self.state.Model_hess is correct at time of call.
        
        SP = Calculation(
            START_FILE=self.START_FILE,
            N_THREAD=self.config.N_THREAD,
            SET_MEMORY=self.config.SET_MEMORY,
            FUNCTIONAL=self.config.FUNCTIONAL,
            FC_COUNT=self.config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=self.BPA_FOLDER_DIRECTORY,
            Model_hess=self.state.Model_hess, # Reads from state
            software_type=self.config.othersoft,
            unrestrict=self.config.unrestrict,
            SUB_BASIS_SET=self.config.SUB_BASIS_SET,
            BASIS_SET=self.config.BASIS_SET,
            spin_multiplicity=self.config.spin_multiplicity,
            electronic_charge=self.config.electronic_charge,
            excited_state=self.config.excited_state,
            dft_grid=self.config.dft_grid,
            ECP = self.config.ECP,
            software_path_file = self.config.software_path_file
        )
        SP.cpcm_solv_model = self.config.cpcm_solv_model
        SP.alpb_solv_model = self.config.alpb_solv_model
        return SP

    def write_input_files(self, FIO):
        # (Reads self.config)
        # (This method sets self.element_list and self.state.Model_hess,
        #  which is a bit of a side-effect, but we'll keep it)
        
        if os.path.splitext(FIO.START_FILE)[1] == ".gjf":
            print("Gaussian input file (.gjf) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_gjf_file(self.config.electric_charge_and_multiplicity)
        elif os.path.splitext(FIO.START_FILE)[1] == ".inp":
            print("GAMESS/Orca/Q-Chem input file (.inp) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_gamess_inp_file(self.config.electric_charge_and_multiplicity)
        elif os.path.splitext(FIO.START_FILE)[1] == ".mol":
            print("MDL Molfile (.mol) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_mol_file(self.config.electric_charge_and_multiplicity)
        elif os.path.splitext(FIO.START_FILE)[1] == ".mol2":
            print("MOL2 file (.mol2) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_mol2_file(self.config.electric_charge_and_multiplicity)
        else:
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.config.electric_charge_and_multiplicity)
        
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        
        if self.config.args.pyscf:
            electric_charge_and_multiplicity = self.config.electric_charge_and_multiplicity
            
        self.element_list = element_list # Set self.element_list
        # self.Model_hess = np.eye(len(element_list) * 3) # This is now done in OptimizationState
        
        return file_directory, electric_charge_and_multiplicity, element_list

    def save_tmp_energy_profiles(self, iter, e, g, B_g):
        # (This method is pure, no changes needed, writes to files)
        if iter == 0:
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write("energy [hartree] \n")
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
            f.write(str(e)+"\n")
        #-------------------gradient profile
        if iter == 0:
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write("gradient (RMS) [hartree/Bohr] \n")
        with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
            f.write(str(self.calculate_rms_safely(g))+"\n")
        #-------------------
        if iter == 0:
            with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                f.write("bias gradient (RMS) [hartree/Bohr] \n")
        with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
            f.write(str(self.calculate_rms_safely(B_g))+"\n")
            #-------------------
        return
    
    def _save_energy_profiles(self):
        # (Reads self.state)
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.state.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.state.ENERGY_LIST_FOR_PLOTTING[i] - self.state.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        return

    def geom_info_extract(self, force_data, file_directory, B_g, g):
        # (Writes to self.state.cos_list)
        if len(force_data["geom_info"]) > 1:
            CSI = CalculationStructInfo()
            
            data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
            
            for num, i in enumerate(force_data["geom_info"]):
                cos = CSI.calculate_cos(B_g[i-1] - g[i-1], g[i-1])
                self.state.cos_list[num].append(cos)
            
            # Need to use self.state.iter to check
            if self.state.iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                    f.write(",".join(data_name_list)+"\n")
            
            with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                f.write(",".join(list(map(str,data_list)))+"\n")          
        return
    
    
    def dissociation_check(self, new_geometry, element_list):
        """
        Checks if the molecular geometry has dissociated into multiple fragments
        based on a distance threshold.
        """
        # (Reads self.config.DC_check_dist)
        atom_label_list = list(range(len(new_geometry)))
        fragm_atom_num_list = []

        # 1. Identify all molecular fragments (connected components)
        while len(atom_label_list) > 0:
            tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0])
            atom_label_list = list(set(atom_label_list) - set(tmp_fragm_list))
            fragm_atom_num_list.append(tmp_fragm_list)
        
        # 2. Check distances only if there is more than one fragment
        if len(fragm_atom_num_list) > 1:
            fragm_dist_list = []
            
            # Ensure geometry is a NumPy array for efficient slicing
            geom_np = np.asarray(new_geometry)
            
            # Iterate through all unique pairs of fragments
            for fragm_1_indices, fragm_2_indices in itertools.combinations(fragm_atom_num_list, 2):
                
                # Get the coordinates for all atoms in each fragment
                coords1 = geom_np[fragm_1_indices] # Shape (M, 3)
                coords2 = geom_np[fragm_2_indices] # Shape (K, 3)
                
                # Reshape coords1 to (M, 1, 3) and coords2 to (1, K, 3)
                # This allows NumPy broadcasting to create all pairs of differences
                # The result (diff_matrix) will have shape (M, K, 3)
                diff_matrix = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
                
                # Square the differences and sum along the last axis (axis=2)
                # This calculates the squared Euclidean distance for all pairs
                # The result (sq_dist_matrix) will have shape (M, K)
                sq_dist_matrix = np.sum(diff_matrix**2, axis=2)
                
                # Find the minimum value in the squared distance matrix
                min_sq_dist = np.min(sq_dist_matrix)
                
                # Take the square root of only the minimum value to get the final distance
                min_dist = np.sqrt(min_sq_dist)
           
                fragm_dist_list.append(min_dist)
                
            # 3. Check if the closest distance between any two fragments
            #    is greater than the dissociation threshold.
            min_interfragment_dist = min(fragm_dist_list)
            
            if min_interfragment_dist > self.config.DC_check_dist:
                print(f"Minimum fragment distance (ang.) {min_interfragment_dist:.4f} > {self.config.DC_check_dist}")
                print("These molecules are dissociated.")
                DC_exit_flag = True
            else:
                DC_exit_flag = False
        else:
            # Only one fragment, so it's not dissociated
            DC_exit_flag = False
            
        return DC_exit_flag
    
    def calculate_rms_safely(self, vector, threshold=1e-10):
        # (This method is pure, no changes needed)
        filtered_vector = vector[np.abs(vector) > threshold]
        if filtered_vector.size > 0:
            return np.sqrt((filtered_vector**2).mean())
        else:
            return 0.0
            
    def print_info(self, e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold):
        # (Reads self.config)
        rms_force = self.calculate_rms_safely(np.abs(B_g))
        rms_displacement = self.calculate_rms_safely(np.abs(displacement_vector))
        max_B_g = np.abs(B_g).max()
        max_displacement = np.abs(displacement_vector).max()
        print("caluculation results (unit a.u.):")
        print("                         Value                     Threshold ")
        print("ENERGY                 : {:>15.12f} ".format(e))
        print("BIAS  ENERGY           : {:>15.12f} ".format(B_e))
        print("Maximum  Force         : {0:>15.12f}                 {1:>15.12f} ".format(max_B_g, self.config.MAX_FORCE_THRESHOLD))
        print("RMS      Force         : {0:>15.12f}                 {1:>15.12f} ".format(rms_force, self.config.RMS_FORCE_THRESHOLD))
        print("Maximum  Displacement  : {0:>15.12f}                 {1:>15.12f} ".format(max_displacement, max_displacement_threshold))
        print("RMS      Displacement  : {0:>15.12f}                 {1:>15.12f} ".format(rms_displacement, rms_displacement_threshold))
        print("ENERGY SHIFT           : {:>15.12f} ".format(e - pre_e))
        print("BIAS ENERGY SHIFT      : {:>15.12f} ".format(B_e - pre_B_e))
        return
    
    def calc_fragement_grads(self, gradient, fragment_list):
        # (This method is pure, no changes needed)
        calced_gradient = gradient
        for fragment in fragment_list:
            tmp_grad = np.array([0.0, 0.0, 0.0], dtype="float64")
            for atom_num in fragment:
                tmp_grad += gradient[atom_num-1]
            tmp_grad /= len(fragment)

            for atom_num in fragment:
                calced_gradient[atom_num-1] = copy.deepcopy(tmp_grad)
        return calced_gradient

    def optimize_oniom(self):
        """
        Perform ONIOM optimization using a high-level QM method for a subset of atoms
        and a low-level method for the entire system.
        
        Refactored to use self.config and self.state.
        """
        # 1. Parse input parameters and initialize file IO
        force_data = force_data_parser(self.config.args)
        high_layer_atom_num = force_data["oniom_flag"][0]
        link_atom_num = force_data["oniom_flag"][1]
        calc_method = force_data["oniom_flag"][2]
        
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        
        # 2. Write input files and create the State object
        geometry_list, element_list, electric_charge_and_multiplicity = self.write_input_files(FIO)
        self.element_list = element_list # Set on self for helpers
        
        # Create the main State object for the "Real" system
        self.state = OptimizationState(element_list)
        self.state.cos_list = [[] for i in range(len(force_data["geom_info"]))]

        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        
        # 3. Import appropriate calculation modules
        if self.config.args.pyscf:
            from multioptpy.Calculator.pyscf_calculation_tools import Calculation as HL_Calculation
        else:
            from multioptpy.Calculator.psi4_calculation_tools import Calculation as HL_Calculation
        
        if calc_method in ["GFN2-xTB", "GFN1-xTB", "IPEA1-xTB"]:
            from multioptpy.Calculator.tblite_calculation_tools import Calculation as LL_Calculation
        else:
            from multioptpy.Calculator.ase_calculation_tools import Calculation as LL_Calculation
        
        # Save ONIOM configuration to file
        with open(self.BPA_FOLDER_DIRECTORY+"ONIOM2.txt", "w") as f:
            f.write("### Low layer ###\n")
            f.write(calc_method+"\n")
            f.write("### High layer ###\n")
            f.write(self.config.BASIS_SET+"\n")
            f.write(self.config.FUNCTIONAL+"\n")  
        
        # 4. Initialize geometries and ONIOM setup
        geom_num_list = []
        for i in range(2, len(geometry_list[0])):
            geom_num_list.append(geometry_list[0][i][1:4])
        geom_num_list = np.array(geom_num_list, dtype="float64") / self.config.bohr2angstroms
        self.state.geom_num_list = geom_num_list # Set initial geometry in state
        
        linker_atom_pair_num = specify_link_atom_pairs(geom_num_list, element_list, high_layer_atom_num, link_atom_num)
        print("Boundary of high layer and low layer:", linker_atom_pair_num)
        
        high_layer_geom_num_list, high_layer_element_list = separate_high_layer_and_low_layer(
            geom_num_list, linker_atom_pair_num, high_layer_atom_num, element_list)
        
        real_2_highlayer_label_connect_dict, highlayer_2_real_label_connect_dict = link_number_high_layer_and_low_layer(high_layer_atom_num)
        
        # 5. Initialize model Hessians (local state for ONIOM)
        LL_Model_hess = np.eye(len(element_list)*3)
        HL_Model_hess = np.eye((len(high_layer_element_list))*3)
        
        # Create mask for high layer atoms
        bool_list = []
        for i in range(len(element_list)):
            if i in high_layer_atom_num:
                bool_list.extend([True, True, True])
            else:
                bool_list.extend([False, False, False])
        
        # 6. Initialize bias potential calculators
        # (self.CalcBiaspot will be used for the "Real" system bias)
        LL_Calc_BiasPot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        # HL_Calc_BiasPot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY) # Seems unused in original
        self.CalcBiaspot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY) # For main state

        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.config.args)))
        
        # 7. Initialize ONIOM-specific previous-step variables (as local vars)
        pre_model_HL_B_e = 0.0
        pre_model_HL_B_g = np.zeros((len(high_layer_element_list), 3))
        pre_model_HL_g = np.zeros((len(high_layer_element_list), 3))
        # pre_model_LL_B_g = np.zeros((len(high_layer_element_list), 3)) # Seems unused
        pre_real_LL_B_e = 0.0
        pre_real_LL_e = 0.0
        pre_real_LL_B_g = np.zeros((len(element_list), 3))
        pre_real_LL_g = np.zeros((len(element_list), 3))
        pre_real_LL_move_vector = np.zeros((len(element_list), 3))
        pre_model_HL_move_vector = np.zeros((len(high_layer_element_list), 3))
        
        # 8. Initialize HL optimizer
        HL_CMV = CalculateMoveVector(self.config.DELTA, high_layer_element_list[:len(high_layer_atom_num)], 
                                     self.config.args.saddle_order, self.config.FC_COUNT, self.config.temperature, 
                                     max_trust_radius=self.config.max_trust_radius, min_trust_radius=self.config.min_trust_radius)
        HL_optimizer_instances = HL_CMV.initialization(force_data["opt_method"])
        
        for i in range(len(HL_optimizer_instances)):
            HL_optimizer_instances[i].set_hessian(HL_Model_hess[:len(high_layer_atom_num)*3, :len(high_layer_atom_num)*3])
            if self.config.DELTA != "x":
                HL_optimizer_instances[i].DELTA = self.config.DELTA     
        
        # 9. Initialize calculation instances
        HLSP = HL_Calculation(START_FILE=self.START_FILE,
                              SUB_BASIS_SET=self.config.SUB_BASIS_SET,
                              BASIS_SET=self.config.BASIS_SET,
                              N_THREAD=self.config.N_THREAD,
                              SET_MEMORY=self.config.SET_MEMORY,
                              FUNCTIONAL=self.config.FUNCTIONAL,
                              FC_COUNT=self.config.FC_COUNT,
                              BPA_FOLDER_DIRECTORY=self.BPA_FOLDER_DIRECTORY,
                              Model_hess=HL_Model_hess[:len(high_layer_atom_num)*3, :len(high_layer_atom_num)*3],
                              unrestrict=self.config.unrestrict,
                              excited_state=self.config.excited_state,
                              electronic_charge=self.config.electronic_charge,
                              spin_multiplicity=self.config.spin_multiplicity
                              )
        
        LLSP = LL_Calculation(START_FILE=self.START_FILE,
                              SUB_BASIS_SET=self.config.SUB_BASIS_SET,
                              BASIS_SET=self.config.BASIS_SET,
                              N_THREAD=self.config.N_THREAD,
                              SET_MEMORY=self.config.SET_MEMORY,
                              FUNCTIONAL=self.config.FUNCTIONAL,
                              FC_COUNT=self.config.FC_COUNT,
                              BPA_FOLDER_DIRECTORY=self.BPA_FOLDER_DIRECTORY,
                              Model_hess=LL_Model_hess,
                              unrestrict=self.config.unrestrict,
                              software_type=calc_method,
                              excited_state=self.config.excited_state)
        
        # 10. Initialize result tracking (uses self.state)
        real_grad_list = []
        real_bias_grad_list = []
        
        # 11. Main optimization loop
        for iter in range(self.config.NSTEP):
            self.state.iter = iter
            
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")
            if exit_file_detect:
                self.state.exit_flag = True
                break
            
            print(f"\n# ITR. {iter}\n")
            
            if iter == 0:
                high_layer_initial_geom_num_list = high_layer_geom_num_list.copy()  # Bohr
                high_layer_pre_geom = high_layer_initial_geom_num_list.copy()  # Bohr
                real_initial_geom_num_list = geom_num_list.copy()  # Bohr
                real_pre_geom = real_initial_geom_num_list.copy()  # Bohr
            
            # --- Model Low Layer Calc ---
            print("Model low layer calculation")
            model_LL_e, model_LL_g, high_layer_geom_num_list, finish_frag = LLSP.single_point(
                file_directory, high_layer_element_list, iter, electric_charge_and_multiplicity, 
                calc_method, geom_num_list=high_layer_geom_num_list*self.config.bohr2angstroms)
            
            if finish_frag:
                self.state.exit_flag = True
                break
            
            # --- Microiterations ---
            print("Processing microiteration...")
            LL_CMV = CalculateMoveVector(self.config.DELTA, element_list, self.config.args.saddle_order, self.config.FC_COUNT, self.config.temperature)
            LL_optimizer_instances = LL_CMV.initialization(["fire"])
            LL_optimizer_instances[0].display_flag = False
            
            low_layer_converged = False
            
            # Use geom_num_list from main state
            current_geom_num_list = self.state.geom_num_list.copy()
            
            for microiter in range(self.config.microiter_num):
                LLSP.Model_hess = LL_Model_hess
                
                real_LL_e, real_LL_g, current_geom_num_list, finish_frag = LLSP.single_point(
                    file_directory, element_list, microiter, electric_charge_and_multiplicity, 
                    calc_method, geom_num_list=current_geom_num_list*self.config.bohr2angstroms)
                
                LL_Model_hess = LLSP.Model_hess
                
                LL_Calc_BiasPot.Model_hess = LL_Model_hess
                _, real_LL_B_e, real_LL_B_g, LL_BPA_hessian = LL_Calc_BiasPot.main(
                    real_LL_e, real_LL_g, current_geom_num_list, element_list, 
                    force_data, pre_real_LL_B_g, microiter, real_initial_geom_num_list)
                
                for x in range(len(LL_optimizer_instances)):
                    LL_optimizer_instances[x].set_bias_hessian(LL_BPA_hessian)
                    if microiter % self.config.FC_COUNT == 0: # Using FC_COUNT, not mFC_COUNT
                        LL_optimizer_instances[x].set_hessian(LL_Model_hess)
                
                if len(force_data["opt_fragment"]) > 0:
                    real_LL_B_g = copy.deepcopy(self.calc_fragement_grads(real_LL_B_g, force_data["opt_fragment"]))
                    real_LL_g = copy.deepcopy(self.calc_fragement_grads(real_LL_g, force_data["opt_fragment"]))
                
                prev_geom = current_geom_num_list.copy()
                
                current_geom_num_list_ang, LL_move_vector, LL_optimizer_instances = LL_CMV.calc_move_vector(
                    microiter, current_geom_num_list, real_LL_B_g, pre_real_LL_B_g, 
                    real_pre_geom, real_LL_B_e, pre_real_LL_B_e, 
                    pre_real_LL_move_vector, real_initial_geom_num_list, 
                    real_LL_g, pre_real_LL_g, LL_optimizer_instances, print_flag=False)
                
                current_geom_num_list = current_geom_num_list_ang / self.config.bohr2angstroms

                # Fix high layer atoms
                for key, value in highlayer_2_real_label_connect_dict.items():
                    current_geom_num_list[value-1] = copy.deepcopy(high_layer_geom_num_list[key-1]) # Already in Bohr
                
                # Fix user-specified atoms
                if len(force_data["fix_atoms"]) > 0:
                    for j in force_data["fix_atoms"]:
                        current_geom_num_list[j-1] = copy.deepcopy(real_initial_geom_num_list[j-1]) # Already in Bohr
                
                displacement_vector = current_geom_num_list - prev_geom
                
                # Calculate convergence metrics for low layer atoms only
                low_layer_grads = []
                low_layer_displacements = []
                for i in range(len(element_list)):
                    if (i+1) not in high_layer_atom_num:
                        low_layer_grads.append(real_LL_B_g[i])
                        low_layer_displacements.append(displacement_vector[i])
                
                low_layer_grads = np.array(low_layer_grads)
                low_layer_displacements = np.array(low_layer_displacements)
                
                low_layer_rms_grad = self.calculate_rms_safely(low_layer_grads)
                max_displacement = np.abs(displacement_vector).max() if len(displacement_vector) > 0 else 0
                rms_displacement = self.calculate_rms_safely(displacement_vector)
                energy_shift = -1 * pre_real_LL_B_e + real_LL_B_e
            
                if microiter % 10 == 0:
                    print(f"M. ITR. {microiter}")
                    print("Microiteration results:")
                    print(f"LOW LAYER BIAS ENERGY : {float(real_LL_B_e):10.8f}")
                    print(f"LOW LAYER ENERGY      : {float(real_LL_e):10.8f}")
                    print(f"LOW LAYER MAX GRADIENT: {float(low_layer_grads.max() if len(low_layer_grads) > 0 else 0):10.8f}")
                    print(f"LOW LAYER RMS GRADIENT: {float(low_layer_rms_grad):10.8f}")
                    print(f"MAX DISPLACEMENT      : {float(max_displacement):10.8f}")
                    print(f"RMS DISPLACEMENT      : {float(rms_displacement):10.8f}")
                    print(f"ENERGY SHIFT          : {float(energy_shift):10.8f}")
                
                # Check convergence (using hardcoded values from original)
                if (low_layer_rms_grad < 0.0003) and \
                   (low_layer_grads.max() < 0.0006 if len(low_layer_grads) > 0 else True) and \
                   (max_displacement < 0.003) and \
                   (rms_displacement < 0.002):
                    print("Low layer converged... (microiteration)")
                    low_layer_converged = True
                    break
                
                # Update previous values for next microiteration
                pre_real_LL_B_e = real_LL_B_e
                pre_real_LL_g = real_LL_g
                pre_real_LL_B_g = real_LL_B_g
                pre_real_LL_move_vector = LL_move_vector
            
            # End of microiteration loop
            if not low_layer_converged:
                print("Reached maximum number of microiterations.")
            print("Microiteration complete.")
            
            # Update the main geometry state
            self.state.geom_num_list = current_geom_num_list
            geom_num_list = current_geom_num_list # Use for this iter
            
            # --- Model High Layer Calc ---
            print("Model system (high layer)")
            HLSP.Model_hess = HL_Model_hess
            model_HL_e, model_HL_g, high_layer_geom_num_list, finish_frag = HLSP.single_point(
                file_directory, high_layer_element_list, iter, electric_charge_and_multiplicity,
                method="", geom_num_list=high_layer_geom_num_list*self.config.bohr2angstroms)
            
            HL_Model_hess = HLSP.Model_hess
            
            if finish_frag:
                self.state.exit_flag = True
                break
            
            # --- Combine Gradients ---
            # Use LL_Calc_BiasPot to get bias gradient on "Real" system
            _, tmp_model_HL_B_e, tmp_model_HL_B_g, LL_BPA_hessian = LL_Calc_BiasPot.main(
                0.0, real_LL_g*0.0, geom_num_list, element_list, force_data, pre_real_LL_B_g*0.0, iter, real_initial_geom_num_list)
            
            tmp_model_HL_g = tmp_model_HL_B_g * 0.0
            
            for key, value in real_2_highlayer_label_connect_dict.items():
                tmp_model_HL_B_g[key-1] += model_HL_g[value-1] - model_LL_g[value-1]
                tmp_model_HL_g[key-1] += model_HL_g[value-1] - model_LL_g[value-1]
            
            HL_BPA_hessian = LL_BPA_hessian[np.ix_(bool_list, bool_list)]
            
            for i in range(len(HL_optimizer_instances)):
                HL_optimizer_instances[i].set_bias_hessian(HL_BPA_hessian)
                if iter % self.config.FC_COUNT == 0:
                    HL_optimizer_instances[i].set_hessian(HL_Model_hess[:len(high_layer_atom_num)*3, :len(high_layer_atom_num)*3])
            
            if len(force_data["opt_fragment"]) > 0:
                tmp_model_HL_B_g = copy.deepcopy(self.calc_fragement_grads(tmp_model_HL_B_g, force_data["opt_fragment"]))
                tmp_model_HL_g = copy.deepcopy(self.calc_fragement_grads(tmp_model_HL_g, force_data["opt_fragment"]))
            
            model_HL_B_g = copy.deepcopy(model_HL_g)
            model_HL_B_e = model_HL_e + tmp_model_HL_B_e
            
            for key, value in real_2_highlayer_label_connect_dict.items():
                model_HL_B_g[value-1] += tmp_model_HL_B_g[key-1] # This seems incorrect logic, but mirrors original
            
            pre_high_layer_geom_num_list = high_layer_geom_num_list
            
            # --- Calculate HL Move Vector ---
            high_layer_geom_num_list_ang, move_vector, HL_optimizer_instances = HL_CMV.calc_move_vector(
                iter, high_layer_geom_num_list[:len(high_layer_atom_num)], model_HL_B_g[:len(high_layer_atom_num)], pre_model_HL_B_g[:len(high_layer_atom_num)], 
                pre_high_layer_geom_num_list[:len(high_layer_atom_num)], model_HL_B_e, pre_model_HL_B_e, 
                pre_model_HL_move_vector[:len(high_layer_atom_num)], high_layer_pre_geom[:len(high_layer_atom_num)], 
                model_HL_g[:len(high_layer_atom_num)], pre_model_HL_g[:len(high_layer_atom_num)], HL_optimizer_instances)
            
            high_layer_geom_num_list = high_layer_geom_num_list_ang / self.config.bohr2angstroms
            
            # --- Update Full System Geometry ---
            for l in range(len(high_layer_geom_num_list) - len(linker_atom_pair_num)):
                geom_num_list[highlayer_2_real_label_connect_dict[l+1]-1] = copy.deepcopy(high_layer_geom_num_list[l])
            
            # Project out translation and rotation
            geom_num_list -= Calculationtools().calc_center_of_mass(geom_num_list, element_list)
            geom_num_list, _ = Calculationtools().kabsch_algorithm(geom_num_list, real_pre_geom)
            
            # Update high layer geometry after alignment
            high_layer_geom_num_list, high_layer_element_list = separate_high_layer_and_low_layer(
                geom_num_list, linker_atom_pair_num, high_layer_atom_num, element_list)
            
            # --- Combine Energies and Gradients for REAL system ---
            real_e = real_LL_e + model_HL_e - model_LL_e
            real_B_e = real_LL_B_e + model_HL_B_e - model_LL_e # Original uses model_LL_e, not B_e
            real_g = real_LL_g + tmp_model_HL_g
            real_B_g = real_LL_B_g + tmp_model_HL_g
            
            # --- Update Main State ---
            self.state.e = real_e
            self.state.B_e = real_B_e
            self.state.g = real_g
            self.state.B_g = real_B_g
            self.state.geom_num_list = geom_num_list
            
            self.save_tmp_energy_profiles(iter, real_e, real_g, real_B_g)
            self.state.ENERGY_LIST_FOR_PLOTTING.append(real_e*self.config.hartree2kcalmol)
            self.state.BIAS_ENERGY_LIST_FOR_PLOTTING.append(real_B_e*self.config.hartree2kcalmol)
            self.state.NUM_LIST.append(iter)
            
            self.geom_info_extract(force_data, file_directory, real_B_g, real_g)
            
            if len(linker_atom_pair_num) > 0:
                tmp_real_B_g = model_HL_B_g[:-len(linker_atom_pair_num)].reshape(-1,1)
            else:
                tmp_real_B_g = model_HL_B_g.reshape(-1,1)
            
            abjusted_high_layer_geom_num_list, _ = Calculationtools().kabsch_algorithm(high_layer_geom_num_list, pre_high_layer_geom_num_list)
            
            if len(linker_atom_pair_num) > 0:
                tmp_displacement_vector = (abjusted_high_layer_geom_num_list - pre_high_layer_geom_num_list)[:-len(linker_atom_pair_num)].reshape(-1,1)
            else:
                tmp_displacement_vector = (abjusted_high_layer_geom_num_list - pre_high_layer_geom_num_list).reshape(-1,1)
            
            # --- Check Convergence (on HL model) ---
            converge_flag, max_displacement_threshold, rms_displacement_threshold = self._check_converge_criteria(tmp_real_B_g, tmp_displacement_vector)
            
            self.print_info(real_e, real_B_e, tmp_real_B_g, tmp_displacement_vector, self.state.pre_e, self.state.pre_B_e, 
                            max_displacement_threshold, rms_displacement_threshold)
            
            real_grad_list.append(self.calculate_rms_safely(real_g))
            real_bias_grad_list.append(self.calculate_rms_safely(real_B_g))
            
            # Update state lists
            self.state.grad_list = real_grad_list
            self.state.bias_grad_list = real_bias_grad_list
            
            if converge_flag:
                self.state.optimized_flag = True
                print("\n=====================================================")
                print("converged!!!")
                print("=====================================================")
                break
            
            # Fix user-specified atoms
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    geom_num_list[j-1] = copy.deepcopy(real_initial_geom_num_list[j-1]) # Bohr
            
            DC_exit_flag = self.dissociation_check(geom_num_list, element_list)
            if DC_exit_flag:
                self.state.DC_check_flag = True
                break
            
            # --- Update Previous State Variables ---
            self.state.pre_B_e = real_B_e
            self.state.pre_e = real_e
            self.state.pre_geom = geom_num_list
            real_pre_geom = geom_num_list # Update local var
            
            pre_model_HL_B_g = model_HL_B_g
            pre_model_HL_g = model_HL_g
            pre_model_HL_B_e = model_HL_B_e
            pre_model_HL_move_vector = move_vector
            
            # Create input for next iteration
            geometry_list = FIO.print_geometry_list(geom_num_list*self.config.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
        
        else: # Loop finished without break
            self.state.optimized_flag = False
            print("Reached maximum number of iterations. This is not converged.")
            with open(self.BPA_FOLDER_DIRECTORY+"not_converged.txt", "w") as f:
                f.write("Reached maximum number of iterations. This is not converged.")
        
        if self.state.DC_check_flag:
            with open(self.BPA_FOLDER_DIRECTORY+"dissociation_is_detected.txt", "w") as f:
                f.write("These molecules are dissociated.")
        
        # --- 12. Finalize and Save Results ---
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        
        # Finalize plots and save results (using self.state lists)
        self._finalize_optimization(FIO, G, self.state.grad_list, self.state.bias_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, self.state.exit_flag) # Pass LLSP as dummy
        
        # Copy final results from state to self
        self._copy_final_results_from_state()
        return
        
    def get_result_file_path(self):
        """
        Sets the absolute file paths for optimization results.
        Relies on self.BPA_FOLDER_DIRECTORY and self.START_FILE
        which are set by _make_init_directory().
        """
        try:
            if (hasattr(self, 'BPA_FOLDER_DIRECTORY') and self.BPA_FOLDER_DIRECTORY and
                hasattr(self, 'START_FILE') and self.START_FILE):
                
                base_name = os.path.splitext(os.path.basename(self.START_FILE))[0]
                optimized_filename = f"{base_name}_optimized.xyz"
                traj_filename = f"{base_name}_traj.xyz"

                self.optimized_struct_file = os.path.abspath(os.path.join(self.BPA_FOLDER_DIRECTORY, optimized_filename))
                self.traj_file = os.path.abspath(os.path.join(self.BPA_FOLDER_DIRECTORY, traj_filename))
                
                print("Optimized structure file path:", self.optimized_struct_file)
                print("Trajectory file path:", self.traj_file)
            
            else:
                print("Error: BPA_FOLDER_DIRECTORY or START_FILE is not set. Please run optimize() or optimize_oniom() first.")
                self.optimized_struct_file = None
                self.traj_file = None
                
        except Exception as e:
            print(f"Error setting result file paths: {e}")
            self.optimized_struct_file = None
            self.traj_file = None

        return

    def run(self):
        # (Reads self.config)
        if type(self.config.args.INPUT) is str:
            START_FILE_LIST = [self.config.args.INPUT]
        else:
            START_FILE_LIST = self.config.args.INPUT
        
        job_file_list = []
        
        for job_file in START_FILE_LIST:
            print()
            if "*" in job_file:
                result_list = glob.glob(job_file)
                job_file_list = job_file_list + result_list
            else:
                job_file_list = job_file_list + [job_file]
        
        for file in job_file_list:
            print("********************************")
            print(file)
            print("********************************")
            if os.path.exists(file) == False:
                print(f"{file} does not exist.")
                continue
            
            # This creates the directory and sets self.START_FILE
            self._make_init_directory(file)
            
            # Run the main optimization, which will create its own state
            if len(self.config.args.oniom_flag) > 0:
                self.optimize_oniom()
            else:
                self.optimize()
            
            # Post-processing (relies on self.state being set by optimize())
            if self.state and self.config.CMDS:
                CMDPA = CMDSPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.state.ENERGY_LIST_FOR_PLOTTING, self.state.BIAS_ENERGY_LIST_FOR_PLOTTING)
                CMDPA.main()
            if self.state and self.config.PCA:
                PCAPA = PCAPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.state.ENERGY_LIST_FOR_PLOTTING, self.state.BIAS_ENERGY_LIST_FOR_PLOTTING)
                PCAPA.main()
            
            if self.state and len(self.config.irc) > 0:
                if self.config.args.usextb != "None":
                    xtb_method = self.config.args.usextb
                else:
                    xtb_method = "None"
                
                if self.state.iter % self.config.FC_COUNT == 0:
                    hessian = self.state.Model_hess
                else:
                    hessian = None
                    
                EXEC_IRC = IRC(self.BPA_FOLDER_DIRECTORY, self.state.final_file_directory, 
                               self.config.irc, self.SP, self.element_list, 
                               self.config.electric_charge_and_multiplicity, # This was from config
                               force_data_parser(self.config.args), # Re-parse force_data
                               xtb_method, FC_count=int(self.config.FC_COUNT), hessian=hessian) 
                EXEC_IRC.run()
                self.irc_terminal_struct_paths = EXEC_IRC.terminal_struct_paths
            else:
                self.irc_terminal_struct_paths = []
            
            print(f"Trial of geometry optimization ({file}) was completed.")
        
        print("All calculations were completed.")
        
        self.get_result_file_path()
        return