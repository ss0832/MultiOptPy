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

class Optimize:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol
        self._set_convergence_criteria(args)
        self._initialize_variables(args)

    def _set_convergence_criteria(self, args):
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

    def _initialize_variables(self, args):
        self.microiter_num = 100
        self.args = args
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
        self._check_sub_basisset(args)
        self.Model_hess = None
        self.mFC_COUNT = args.calc_model_hess
        self.DC_check_dist = float(args.dissociate_check)
        self.unrestrict = args.unrestrict
        self.irc = args.intrinsic_reaction_coordinates
        self.force_data = force_data_parser(self.args)
        self.final_file_directory = None
        self.final_geometry = None
        self.final_energy = None
        self.final_bias_energy = None
        self.othersoft = args.othersoft
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        self.shape_conditions = args.shape_conditions
        self.bias_pot_params_grad_list = None
        self.bias_pot_params_grad_name_list = None
        self.DC_check_flag = False
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

    def _check_sub_basisset(self, args):
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

    def _make_init_directory(self, file):
        """
        Create initial directory for optimization results.
        """
        self.START_FILE = file
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]
        date = datetime.datetime.now().strftime("%Y_%m_%d")
        base_dir = f"{date}/{self.START_FILE[:-4]}_OPT_"

        if self.othersoft != "None":
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}ASE_{timestamp}/"
        elif self.sqm2:
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}SQM2_{timestamp}/"
        elif self.sqm1:
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}SQM1_{timestamp}/"
        elif self.args.usextb == "None" and self.args.usedxtb == "None":
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}{self.FUNCTIONAL}_{self.BASIS_SET}_{timestamp}/"
        else:
            method = self.args.usedxtb if self.args.usedxtb != "None" else self.args.usextb
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}{method}_{timestamp}/"
        
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        
    def _save_input_data(self):
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        return

    def _constrain_flag_check(self, force_data):
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
        if iter == 0:
            if projection_constrain:
                PC.initialize(geom_num_list, hessian=hessian)
            else:
                pass
            return PC
        else:
            return PC

    def _save_init_geometry(self, geom_num_list, element_list, allactive_flag):
        
        if allactive_flag:
            initial_geom_num_list = geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
            pre_geom = initial_geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
        else:
            initial_geom_num_list = geom_num_list 
            pre_geom = initial_geom_num_list 
        
        return initial_geom_num_list, pre_geom


    def _calc_eff_hess_for_fix_atoms_and_set_hess(self, allactive_flag, force_data, BPA_hessian, n_fix, optimizer_instances, geom_num_list, B_g, g, projection_constrain, PC):
        if not allactive_flag:
            fix_num = []
            for fnum in force_data["fix_atoms"]:
                fix_num.extend([3*(fnum-1)+0, 3*(fnum-1)+1, 3*(fnum-1)+2])
            fix_num = np.array(fix_num, dtype="int64")
            #effective hessian
            tmp_fix_hess = self.Model_hess[np.ix_(fix_num, fix_num)] + np.eye((3*n_fix)) * 1e-10
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
            
            if self.iter % self.FC_COUNT == 0 or (self.use_model_hessian is not None and self.iter % self.mFC_COUNT == 0):
        
                if not allactive_flag:
                    self.Model_hess -= np.dot(self.Model_hess[:, fix_num], np.dot(inv_tmp_fix_hess, self.Model_hess[fix_num, :]))
                
                
                if projection_constrain:
                    proj_model_hess = PC.calc_project_out_hess(geom_num_list, g, self.Model_hess)
                    optimizer_instances[i].set_hessian(proj_model_hess)
                else:
                    optimizer_instances[i].set_hessian(self.Model_hess)
        
        return optimizer_instances
        

    def _apply_projection_constraints(self, projection_constrain, PC, geom_num_list, g, B_g):
        if projection_constrain:
            g = copy.copy(PC.calc_project_out_grad(geom_num_list, g))
            proj_d_B_g = copy.copy(PC.calc_project_out_grad(geom_num_list, B_g - g))
            B_g = copy.copy(g + proj_d_B_g)
        
        return g, B_g, PC

    def _zero_fixed_atom_gradients(self, allactive_flag, force_data, g, B_g):
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                g[j-1] = copy.copy(g[j-1]*0.0)
                B_g[j-1] = copy.copy(B_g[j-1]*0.0)
        
        return g, B_g

    def _project_out_translation_rotation(self, new_geometry, geom_num_list, allactive_flag):
        
        if allactive_flag:
            # Convert to Bohr, apply Kabsch alignment algorithm, then convert back
            aligned_geometry, _ = Calculationtools().kabsch_algorithm(
                new_geometry/self.bohr2angstroms, geom_num_list)
            aligned_geometry *= self.bohr2angstroms
            return aligned_geometry
        else:
            # If not all atoms are active, return the original geometry
            return new_geometry

    def _apply_projection_constraints_to_geometry(self, projection_constrain, PC, new_geometry, hessian=None):
        if projection_constrain:
            tmp_new_geometry = new_geometry / self.bohr2angstroms
            adjusted_geometry = PC.adjust_init_coord(tmp_new_geometry, hessian=hessian) * self.bohr2angstroms
            return adjusted_geometry, PC
        
        return new_geometry, PC

    def _reset_fixed_atom_positions(self, new_geometry, initial_geom_num_list, allactive_flag, force_data):
        
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
        
        return new_geometry


    def _initialize_optimization_variables(self):
        # Initialize return dictionary with categories
        vars_dict = {
            'calculation': {},  # Calculation modules and algorithms
            'io': {},           # File input/output handlers
            'energy': {},       # Energy-related variables
            'geometry': {},     # Geometry and structure variables
            'gradients': {},    # Gradient and force variables
            'constraints': {},  # Constraint related variables
            'optimization': {}, # Optimizer settings
            'misc': {},          # Miscellaneous
            'oniom': {}          # ONIOM related variables
        }
        
        # Calculation modules and file I/O
        Calculation, xtb_method = self._import_calculation_module()
        self._save_input_data()
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        
        # Initialize energy tracking arrays
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.BIAS_ENERGY_LIST_FOR_PLOTTING = []
        self.NUM_LIST = []
        
        # Force data and flags
        force_data = force_data_parser(self.args)
        exit_flag = False
        
        # Energy state variables
        geom_num_list = None  # Bohr
        e = None  # Hartree
        B_e = None  # Hartree
        
        # Previous step information
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        
        # Atom-related data
        n_fix = len(force_data["fix_atoms"])
        file_directory, electric_charge_and_multiplicity, element_list = self.write_input_files(FIO)
        
        # Initialize gradient arrays
        for i in range(len(element_list)):
            pre_B_g.append([0, 0, 0])
        pre_B_g = np.array(pre_B_g, dtype="float64")
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        
        # Analysis data structures
        self.cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        bias_grad_list = []

        
        # Element information
        element_number_list = np.array([element_number(elem) for elem in element_list], dtype="int")
        natom = len(element_list)
        
        # Constraint setup
        PC = ProjectOutConstrain(force_data["projection_constraint_condition_list"], 
                                force_data["projection_constraint_atoms"], 
                                force_data["projection_constraint_constant"])
        projection_constrain, allactive_flag = self._constrain_flag_check(force_data)
        
        # Bias potential and calculation setup
        self.CalcBiaspot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        SP = self.setup_calculation(Calculation)
        
        # Move vector calculation
        CMV = CalculateMoveVector(self.DELTA, element_list, self.args.saddle_order, 
                                self.FC_COUNT, self.temperature, self.use_model_hessian, max_trust_radius=self.max_trust_radius, min_trust_radius=self.min_trust_radius)
        optimizer_instances = CMV.initialization(force_data["opt_method"])
        
        for i in range(len(optimizer_instances)):
            if CMV.newton_tag[i] is False and self.FC_COUNT > 0 and not "eigvec" in force_data["projection_constraint_condition_list"]:
                print("Error: This optimizer method does not support exact Hessian calculations.")
                print("Please either choose a different optimizer or set FC_COUNT=0 to disable exact Hessian calculations.")
                sys.exit(0)
        
        
        # Initialize optimizer instances
        for i in range(len(optimizer_instances)):
            optimizer_instances[i].set_hessian(self.Model_hess)
            if self.DELTA != "x":
                optimizer_instances[i].DELTA = self.DELTA
            
                
        
    
        
        # Final status flag
        optimized_flag = False
        
        # Populate the dictionary with grouped variables
        vars_dict['calculation'] = {
            'Calculation': Calculation,
            'xtb_method': xtb_method,
            'SP': SP,
            'CMV': CMV,
            'optimizer_instances': optimizer_instances
        }
        
        vars_dict['io'] = {
            'FIO': FIO,
            'G': G,
            'file_directory': file_directory
        }
        
        vars_dict['energy'] = {
            'e': e,
            'B_e': B_e,
            'pre_e': pre_e,
            'pre_B_e': pre_B_e
        }
        
        vars_dict['geometry'] = {
            'geom_num_list': geom_num_list,
            'element_list': element_list,
            'element_number_list': element_number_list,
            'natom': natom,
            'electric_charge_and_multiplicity': electric_charge_and_multiplicity
        }
        
        vars_dict['gradients'] = {
            'pre_g': pre_g,
            'pre_B_g': pre_B_g,
            'grad_list': grad_list,
            'bias_grad_list': bias_grad_list,
            'pre_move_vector': pre_move_vector
        }
        
        vars_dict['constraints'] = {
            'PC': PC,
            'projection_constrain': projection_constrain,
            'allactive_flag': allactive_flag,
            'force_data': force_data,
            'n_fix': n_fix
        }
        
        vars_dict['misc'] = {
            'exit_flag': exit_flag,
            'optimized_flag': optimized_flag,
        }
        

        
        return vars_dict

    def check_negative_eigenvalues(self, geom_num_list, hessian):
        proj_hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(hessian, geom_num_list, geom_num_list, display_eigval=False)
        if proj_hessian is not None:
            eigvals = np.linalg.eigvalsh(proj_hessian)
            if np.any(eigvals < 0):
                print("Notice: Negative eigenvalues detected.")
                return True
        return False

    def judge_early_stop_due_to_no_negative_eigenvalues(self, geom_num_list, hessian):
        if self.detect_negative_eigenvalues and self.FC_COUNT > 0:
            negative_eigenvalues_detected = self.check_negative_eigenvalues(geom_num_list, hessian)
            if not negative_eigenvalues_detected and self.args.saddle_order > 0:
                print("No negative eigenvalues detected while saddle_order > 0. Stopping optimization.")
                with open(self.BPA_FOLDER_DIRECTORY+"no_negative_eigenvalues_detected.txt", "w") as f:
                    f.write("No negative eigenvalues detected while saddle_order > 0. Stopping optimization.")
                return True
        return False

    def optimize(self):
        # Initialize all variables needed for optimization
        vars_dict = self._initialize_optimization_variables()
        
        # Extract variables from dictionary
        # Calculation related
        Calculation = vars_dict['calculation']['Calculation']
        xtb_method = vars_dict['calculation']['xtb_method']
        SP = vars_dict['calculation']['SP']
        CMV = vars_dict['calculation']['CMV']
        optimizer_instances = vars_dict['calculation']['optimizer_instances']
        
        # File I/O related
        FIO = vars_dict['io']['FIO']
        G = vars_dict['io']['G']
        file_directory = vars_dict['io']['file_directory']
        
        # Energy related
        e = vars_dict['energy']['e']
        B_e = vars_dict['energy']['B_e']
        pre_e = vars_dict['energy']['pre_e']
        pre_B_e = vars_dict['energy']['pre_B_e']
        
        # Geometry related
        geom_num_list = vars_dict['geometry']['geom_num_list']
        element_list = vars_dict['geometry']['element_list']
        element_number_list = vars_dict['geometry']['element_number_list']
        natom = vars_dict['geometry']['natom']
        electric_charge_and_multiplicity = vars_dict['geometry']['electric_charge_and_multiplicity']
        
        # Gradient related
        pre_g = vars_dict['gradients']['pre_g']
        pre_B_g = vars_dict['gradients']['pre_B_g']
        grad_list = vars_dict['gradients']['grad_list']
        bias_grad_list = vars_dict['gradients']['bias_grad_list']
        pre_move_vector = vars_dict['gradients']['pre_move_vector']
        
        # Constraint related
        PC = vars_dict['constraints']['PC']
        projection_constrain = vars_dict['constraints']['projection_constrain']
        allactive_flag = vars_dict['constraints']['allactive_flag']
        force_data = vars_dict['constraints']['force_data']
        n_fix = vars_dict['constraints']['n_fix']
        
        # Miscellaneous
        exit_flag = vars_dict['misc']['exit_flag']
        optimized_flag = vars_dict['misc']['optimized_flag']
  
        exact_hess_flag = False
        
        
        if self.koopman_analysis:
            KA = KoopmanAnalyzer(natom, file_directory=self.BPA_FOLDER_DIRECTORY)

        for iter in range(self.NSTEP):

                
            self.iter = iter
            exit_flag = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")
            if exit_flag:
                break
            exit_flag = judge_shape_condition(geom_num_list, self.shape_conditions)
            if exit_flag:
                break
            print("\n# ITR. "+str(iter)+"\n")
            SP.Model_hess = copy.copy(self.Model_hess)
            e, g, geom_num_list, exit_flag = SP.single_point(file_directory, element_number_list, iter, electric_charge_and_multiplicity, xtb_method)
            self.Model_hess = copy.copy(SP.Model_hess)
            
            if exit_flag:
                break
                
            if iter % self.mFC_COUNT == 0 and self.use_model_hessian is not None and self.FC_COUNT < 1:
                SP.Model_hess = ApproxHessian().main(geom_num_list, element_list, g, self.use_model_hessian)
                self.Model_hess = SP.Model_hess 
            if iter == 0:
                initial_geom_num_list, pre_geom = self._save_init_geometry(geom_num_list, element_list, allactive_flag)

            _, B_e, B_g, BPA_hessian = self.CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)

          
            Hess = BPA_hessian + self.Model_hess

            if iter == 0:
                if self.judge_early_stop_due_to_no_negative_eigenvalues(geom_num_list, Hess) and iter == 0:
                    break
            
            
            PC = self._init_projection_constraint(PC, geom_num_list, iter, projection_constrain, hessian=Hess)

            optimizer_instances = self._calc_eff_hess_for_fix_atoms_and_set_hess(allactive_flag, force_data, BPA_hessian, n_fix, optimizer_instances, geom_num_list, B_g, g, projection_constrain, PC)
                    
            if not allactive_flag:
                B_g = copy.copy(self.calc_fragement_grads(B_g, force_data["opt_fragment"]))
                g = copy.copy(self.calc_fragement_grads(g, force_data["opt_fragment"]))
            
            #energy profile 
            self.save_tmp_energy_profiles(iter, e, g, B_g)
            
            g, B_g, PC = self._apply_projection_constraints(projection_constrain, PC, geom_num_list, g, B_g)
                
            g, B_g = self._zero_fixed_atom_gradients(allactive_flag, force_data, g, B_g)

            if self.koopman_analysis:
                _ = KA.run(iter, geom_num_list, B_g, element_list)

            new_geometry, move_vector, optimizer_instances = CMV.calc_move_vector(iter, geom_num_list,
                                                                                  B_g, pre_B_g, pre_geom, B_e, pre_B_e,
                                                                                  pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances, projection_constrain)
            
          
            new_geometry = self._project_out_translation_rotation(new_geometry, geom_num_list, allactive_flag)
            
            new_geometry, PC = self._apply_projection_constraints_to_geometry(projection_constrain, PC, new_geometry, hessian=Hess)

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.BIAS_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)   
            
            if self.iter == 0:
                displacement_vector = move_vector
            else:
                displacement_vector = new_geometry / self.bohr2angstroms - geom_num_list
            
            converge_flag, max_displacement_threshold, rms_displacement_threshold = self._check_converge_criteria(B_g, displacement_vector)
            
            self.print_info(e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold)
            
            grad_list.append(self.calculate_rms_safely(g))
            bias_grad_list.append(self.calculate_rms_safely(B_g))
            
            new_geometry = self._reset_fixed_atom_positions(new_geometry, initial_geom_num_list, allactive_flag, force_data)
            #dissociation check
            DC_exit_flag = self.dissociation_check(new_geometry, element_list)

            if converge_flag:
                if projection_constrain and iter == 0:
                    pass
                else:
                    optimized_flag = True
                    print("\n=====================================================")
                    print("converged!!!")
                    print("=====================================================")
                    break
            
            if DC_exit_flag:
                self.DC_check_flag = True
                break
           
            #Save previous gradient, movestep, and energy.
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = FIO.print_geometry_list(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            
        else:
            print("Reached maximum number of iterations. This is not converged.")
            with open(self.BPA_FOLDER_DIRECTORY+"not_converged.txt", "w") as f:
                f.write("Reached maximum number of iterations. This is not converged.")

        ## --------------------
        # Check if exact hessian is already computed.
        ## --------------------
        if self.FC_COUNT == -1:
            exact_hess_flag = False
        elif self.iter % self.FC_COUNT == 0 and self.FC_COUNT > 0:
            exact_hess_flag = True
        else:
            exact_hess_flag = False
        # --------------------
        
        if self.DC_check_flag:
            print("Dissociation is detected. Optimization stopped.")
            with open(self.BPA_FOLDER_DIRECTORY+"dissociation_is_detected.txt", "w") as f:
                f.write("Dissociation is detected. Optimization stopped.")
        self.optimized_flag = optimized_flag
        if self.freq_analysis and not exit_flag and not self.DC_check_flag:
            self._perform_vibrational_analysis(SP, geom_num_list, element_list, initial_geom_num_list, force_data, exact_hess_flag, file_directory, iter, electric_charge_and_multiplicity, xtb_method, e)

        self._finalize_optimization(FIO, G, grad_list, bias_grad_list,
                                   file_directory, self.force_data, geom_num_list, e, B_e, SP, exit_flag)
        
        
        return

    def _perform_vibrational_analysis(self, SP, geom_num_list, element_list, initial_geom_num_list, force_data, exact_hess_flag, file_directory, iter, electric_charge_and_multiplicity, xtb_method, e):
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
        tmp_hess = copy.copy(SP.Model_hess)
        tmp_hess += BPA_hessian
        MV = MolecularVibrations(atoms=element_list, coordinates=geom_num_list, hessian=tmp_hess)
        results = MV.calculate_thermochemistry(e_tot=B_e, temperature=self.thermo_temperature, pressure=self.thermo_pressure)
        MV.print_thermochemistry(output_file=self.BPA_FOLDER_DIRECTORY+"/thermochemistry.txt")
        MV.print_normal_modes(output_file=self.BPA_FOLDER_DIRECTORY+"/normal_modes.txt")
        MV.create_vibration_animation(output_dir=self.BPA_FOLDER_DIRECTORY+"/vibration_animation")
        if not self.optimized_flag:
            print("Warning: Vibrational analysis was performed, but the optimization did not converge. The result of thermochemistry is useless.")
            
        return

    def _finalize_optimization(self, FIO, G, grad_list, bias_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, exit_flag):
        self._save_opt_results(FIO, G, grad_list, bias_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP)
        self.bias_pot_params_grad_list = self.CalcBiaspot.bias_pot_params_grad_list
        self.bias_pot_params_grad_name_list = self.CalcBiaspot.bias_pot_params_grad_name_list
        self.final_file_directory = file_directory
        self.final_geometry = geom_num_list  # Bohr
        self.final_energy = e  # Hartree
        self.final_bias_energy = B_e  # Hartree
        if not exit_flag:
            self.symmetry = analyze_symmetry(self.element_list, self.final_geometry)
            with open(self.BPA_FOLDER_DIRECTORY+"symmetry.txt", "w") as f:
                f.write(f"Symmetry of final structure: {self.symmetry}")
            print(f"Symmetry: {self.symmetry}")
       
    def _save_opt_results(self, FIO, G, grad_list, bias_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP):
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.BIAS_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        
    
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                G.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)

        FIO.make_traj_file()
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")

        self._save_energy_profiles()

        self.SP = SP
        self.final_file_directory = file_directory
        self.final_geometry = geom_num_list  # Bohr
        self.final_energy = e  # Hartree
        self.final_bias_energy = B_e  # Hartree
        return

    def _check_converge_criteria(self, B_g, displacement_vector):
        
        max_force = np.abs(B_g).max()
        max_force_threshold = self.MAX_FORCE_THRESHOLD
        
        
        rms_force = self.calculate_rms_safely(B_g)
        rms_force_threshold = self.RMS_FORCE_THRESHOLD
        
        delta_max_force_threshold = max(0.0, max_force_threshold -1 * max_force)
        delta_rms_force_threshold = max(0.0, rms_force_threshold -1 * rms_force)
        
        max_displacement = np.abs(displacement_vector).max()
        max_displacement_threshold = max(self.MAX_DISPLACEMENT_THRESHOLD, self.MAX_DISPLACEMENT_THRESHOLD + delta_max_force_threshold)
        rms_displacement = self.calculate_rms_safely(displacement_vector)
        rms_displacement_threshold = max(self.RMS_DISPLACEMENT_THRESHOLD, self.RMS_DISPLACEMENT_THRESHOLD + delta_rms_force_threshold)
        
        if max_force < max_force_threshold and rms_force < rms_force_threshold and max_displacement < max_displacement_threshold and rms_displacement < rms_displacement_threshold:#convergent criteria
            return True, max_displacement_threshold, rms_displacement_threshold
        return False, max_displacement_threshold, rms_displacement_threshold
    
    def _import_calculation_module(self):
        xtb_method = None
        if self.args.pyscf:
            from multioptpy.Calculator.pyscf_calculation_tools import Calculation
        elif self.args.sqm2:
            from multioptpy.Calculator.sqm2_calculation_tools import Calculation
            print("Use SQM2 potential.")
        elif self.args.sqm1:
            from multioptpy.Calculator.sqm1_calculation_tools import Calculation
        elif self.args.othersoft and self.args.othersoft != "None":
            if self.args.othersoft.lower() == "lj":
                from multioptpy.Calculator.lj_calculation_tools import Calculation
                print("Use Lennard-Jones cluster potential.")
            elif self.args.othersoft.lower() == "emt":
                from multioptpy.Calculator.emt_calculation_tools import Calculation
                print("Use ETM potential.")
            elif self.args.othersoft.lower() == "tersoff":
                from multioptpy.Calculator.tersoff_calculation_tools import Calculation
                print("Use Tersoff potential.")
            else:
                from multioptpy.Calculator.ase_calculation_tools import Calculation
                print("Use", self.args.othersoft)
                with open(self.BPA_FOLDER_DIRECTORY + "use_" + self.args.othersoft + ".txt", "w") as f:
                    f.write(self.args.othersoft + "\n")
                    f.write(self.BASIS_SET + "\n")
                    f.write(self.FUNCTIONAL + "\n")
        else:
            if self.args.usedxtb and self.args.usedxtb != "None":
                from multioptpy.Calculator.dxtb_calculation_tools import Calculation
              
                xtb_method = self.args.usedxtb
            elif self.args.usextb and self.args.usextb != "None":
                from multioptpy.Calculator.tblite_calculation_tools import Calculation
               
                xtb_method = self.args.usextb
            else:
                from multioptpy.Calculator.psi4_calculation_tools import Calculation
               
        return Calculation, xtb_method
    
    def setup_calculation(self, Calculation):
        SP = Calculation(
            START_FILE=self.START_FILE,
            N_THREAD=self.N_THREAD,
            SET_MEMORY=self.SET_MEMORY,
            FUNCTIONAL=self.FUNCTIONAL,
            FC_COUNT=self.FC_COUNT,
            BPA_FOLDER_DIRECTORY=self.BPA_FOLDER_DIRECTORY,
            Model_hess=self.Model_hess,
            software_type=self.args.othersoft,
            unrestrict=self.unrestrict,
            SUB_BASIS_SET=self.SUB_BASIS_SET,
            BASIS_SET=self.BASIS_SET,
            spin_multiplicity=self.spin_multiplicity,
            electronic_charge=self.electronic_charge,
            excited_state=self.excited_state,
            dft_grid=self.dft_grid,
            ECP = self.ECP,
            software_path_file = self.software_path_file)
        SP.cpcm_solv_model = self.cpcm_solv_model
        SP.alpb_solv_model = self.alpb_solv_model
        return SP

    def write_input_files(self, FIO):
        if os.path.splitext(FIO.START_FILE)[1] == ".gjf":
            print("Gaussian input file (.gjf) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_gjf_file(self.electric_charge_and_multiplicity)
        elif os.path.splitext(FIO.START_FILE)[1] == ".inp":
            print("GAMESS/Orca/Q-Chem input file (.inp) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_gamess_inp_file(self.electric_charge_and_multiplicity)
        elif os.path.splitext(FIO.START_FILE)[1] == ".mol":
            print("MDL Molfile (.mol) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_mol_file(self.electric_charge_and_multiplicity)
        elif os.path.splitext(FIO.START_FILE)[1] == ".mol2":
            print("MOL2 file (.mol2) detected.")
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_mol2_file(self.electric_charge_and_multiplicity)
        else:
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        
        if self.args.pyscf:
            electric_charge_and_multiplicity = self.electric_charge_and_multiplicity
        self.element_list = element_list
        self.Model_hess = np.eye(len(element_list) * 3)
        return file_directory, electric_charge_and_multiplicity, element_list

    def save_tmp_energy_profiles(self, iter, e, g, B_g):
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
    
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        return

    def geom_info_extract(self, force_data, file_directory, B_g, g):
        if len(force_data["geom_info"]) > 1:
            CSI = CalculationStructInfo()
            
            data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
            
            for num, i in enumerate(force_data["geom_info"]):
                cos = CSI.calculate_cos(B_g[i-1] - g[i-1], g[i-1])
                self.cos_list[num].append(cos)
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                    f.write(",".join(data_name_list)+"\n")
            
            with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                f.write(",".join(list(map(str,data_list)))+"\n")                 
        return
    
    def dissociation_check(self, new_geometry, element_list):
        #dissociation check
        atom_label_list = [i for i in range(len(new_geometry))]
        fragm_atom_num_list = []
        while len(atom_label_list) > 0:
            tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0])
            
            atom_label_list = list(set(atom_label_list) - set(tmp_fragm_list))
            fragm_atom_num_list.append(tmp_fragm_list)
       
        if len(fragm_atom_num_list) > 1:
            fragm_dist_list = []
            for fragm_1_num, fragm_2_num in list(itertools.combinations(fragm_atom_num_list, 2)):
                dist = Calculationtools().calc_fragm_distance(new_geometry, fragm_1_num, fragm_2_num)
                fragm_dist_list.append(dist)
            
            
            if min(fragm_dist_list) > self.DC_check_dist:
                print("mean fragm distance (ang.)", min(fragm_dist_list), ">", self.DC_check_dist)
                print("These molecules are dissociated.")
                DC_exit_flag = True
            else:
                DC_exit_flag = False
        else:
            DC_exit_flag = False
            
        return DC_exit_flag
    
    def calculate_rms_safely(self, vector, threshold=1e-10):
            filtered_vector = vector[np.abs(vector) > threshold]
            if filtered_vector.size > 0:
                return np.sqrt((filtered_vector**2).mean())
            else:
                return 0.0
            
    def print_info(self, e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold):
        

        rms_force = self.calculate_rms_safely(np.abs(B_g))
        rms_displacement = self.calculate_rms_safely(np.abs(displacement_vector))
        max_B_g = np.abs(B_g).max()
        max_displacement = np.abs(displacement_vector).max()
        print("caluculation results (unit a.u.):")
        print("                         Value                         Threshold ")
        print("ENERGY                : {:>15.12f} ".format(e))
        print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
        print("Maximum  Force        : {0:>15.12f}             {1:>15.12f} ".format(max_B_g, self.MAX_FORCE_THRESHOLD))
        print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(rms_force, self.RMS_FORCE_THRESHOLD))
        print("Maximum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(max_displacement, max_displacement_threshold))
        print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(rms_displacement, rms_displacement_threshold))
        print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
        print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
        return
    
    def calc_fragement_grads(self, gradient, fragment_list):
        calced_gradient = gradient
        for fragment in fragment_list:
            tmp_grad = np.array([0.0, 0.0, 0.0], dtype="float64")
            for atom_num in fragment:
                tmp_grad += gradient[atom_num-1]
            tmp_grad /= len(fragment)

            for atom_num in fragment:
                calced_gradient[atom_num-1] = copy.copy(tmp_grad)
        return calced_gradient

    def optimize_oniom(self):
        """
        Perform ONIOM optimization using a high-level QM method for a subset of atoms
        and a low-level method for the entire system.
        
        High layer: psi4 (or other accurate QM method)
        Low layer: tblite, ASE(NNP), etc.
        """
        # Parse input parameters and initialize calculations
        force_data = force_data_parser(self.args)
        high_layer_atom_num = force_data["oniom_flag"][0]
        link_atom_num = force_data["oniom_flag"][1]
        calc_method = force_data["oniom_flag"][2]
        
        # Import appropriate calculation modules
        if self.args.pyscf:
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
            f.write(self.BASIS_SET+"\n")
            f.write(self.FUNCTIONAL+"\n")  
        
        # Initialize file IO and geometry
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        self.element_list = element_list
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        
        # Estimate number of microiterations based on system size
        self.microiter_num += 10 * len(element_list)
        
        # Extract coordinates and convert to Bohr
        geom_num_list = []
        for i in range(2, len(geometry_list[0])):
            geom_num_list.append(geometry_list[0][i][1:4])
        geom_num_list = np.array(geom_num_list, dtype="float64") / self.bohr2angstroms
        
        # Identify link atoms and separate layers
        linker_atom_pair_num = specify_link_atom_pairs(geom_num_list, element_list, high_layer_atom_num, link_atom_num)
        print("Boundary of high layer and low layer:", linker_atom_pair_num)
        
        high_layer_geom_num_list, high_layer_element_list = separate_high_layer_and_low_layer(
            geom_num_list, linker_atom_pair_num, high_layer_atom_num, element_list)
        
        # Create mapping between high layer and full system atom indices
        real_2_highlayer_label_connect_dict, highlayer_2_real_label_connect_dict = link_number_high_layer_and_low_layer(high_layer_atom_num)
        
        
        
        # Initialize model Hessians
        LL_Model_hess = np.eye(len(element_list)*3)
        HL_Model_hess = np.eye((len(high_layer_element_list))*3)
      
        self.microiter_num += 10 * len(element_list)
        # Create mask for high layer atoms in full system
        bool_list = []
        for i in range(len(element_list)):
            if i in high_layer_atom_num:
                bool_list.extend([True, True, True])
            else:
                bool_list.extend([False, False, False])
        
        # Initialize bias potential calculators
        LL_Calc_BiasPot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        HL_Calc_BiasPot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        
        # Save input arguments
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        
        # Initialize energy and gradient tracking variables
        pre_model_HL_B_e = 0.0
        pre_model_HL_B_g = np.zeros((len(high_layer_element_list), 3))
        pre_model_HL_g = np.zeros((len(high_layer_element_list), 3))
        pre_model_LL_B_g = np.zeros((len(high_layer_element_list), 3))
        pre_real_LL_B_e = 0.0
        pre_real_LL_e = 0.0
        pre_real_LL_B_g = np.zeros((len(element_list), 3))
        pre_real_LL_g = np.zeros((len(element_list), 3))
        pre_real_B_e = 0.0
        pre_real_e = 0.0
        pre_real_LL_move_vector = np.zeros((len(element_list), 3))
        pre_model_HL_move_vector = np.zeros((len(high_layer_element_list), 3))
        
        # Initialize high layer optimizer
        HL_CMV = CalculateMoveVector(self.DELTA, high_layer_element_list[:len(high_layer_atom_num)], self.args.saddle_order, self.FC_COUNT, self.temperature, max_trust_radius=self.max_trust_radius, min_trust_radius=self.min_trust_radius)
        HL_optimizer_instances = HL_CMV.initialization(force_data["opt_method"])
        
        for i in range(len(HL_optimizer_instances)):
            HL_optimizer_instances[i].set_hessian(HL_Model_hess[:len(high_layer_atom_num)*3, :len(high_layer_atom_num)*3])
            if self.DELTA != "x":
                HL_optimizer_instances[i].DELTA = self.DELTA      
        
        # Initialize calculation instances
        HLSP = HL_Calculation(START_FILE=self.START_FILE,
                        SUB_BASIS_SET=self.SUB_BASIS_SET,
                        BASIS_SET=self.BASIS_SET,
                        N_THREAD=self.N_THREAD,
                        SET_MEMORY=self.SET_MEMORY,
                        FUNCTIONAL=self.FUNCTIONAL,
                        FC_COUNT=self.FC_COUNT,
                        BPA_FOLDER_DIRECTORY=self.BPA_FOLDER_DIRECTORY,
                        Model_hess=HL_Model_hess[:len(high_layer_atom_num)*3, :len(high_layer_atom_num)*3],
                        unrestrict=self.unrestrict,
                        excited_state=self.excited_state,
                        electronic_charge=self.electronic_charge,
                        spin_multiplicity=self.spin_multiplicity
                        )
        
        LLSP = LL_Calculation(START_FILE=self.START_FILE,
                        SUB_BASIS_SET=self.SUB_BASIS_SET,
                        BASIS_SET=self.BASIS_SET,
                        N_THREAD=self.N_THREAD,
                        SET_MEMORY=self.SET_MEMORY,
                        FUNCTIONAL=self.FUNCTIONAL,
                        FC_COUNT=self.FC_COUNT,
                        BPA_FOLDER_DIRECTORY=self.BPA_FOLDER_DIRECTORY,
                        Model_hess=LL_Model_hess,
                        unrestrict=self.unrestrict,
                        software_type=calc_method,
                        excited_state=self.excited_state)
        
        
        
        # Initialize result tracking
        self.cos_list = [[] for i in range(len(force_data["geom_info"]))]
        real_grad_list = []
        real_bias_grad_list = []
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.BIAS_ENERGY_LIST_FOR_PLOTTING = []
          
        real_e = None
        real_B_e = None
        
        # Main optimization loop
        for iter in range(self.NSTEP):
            self.iter = iter
            
            # Check for exit file
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")
            if exit_file_detect:
                break
            
            print(f"\n# ITR. {iter}\n")
            
            # Initialize geometries on first iteration
            if iter == 0:
                high_layer_initial_geom_num_list = high_layer_geom_num_list.copy()  # Bohr
                high_layer_pre_geom = high_layer_initial_geom_num_list.copy()  # Bohr
                real_initial_geom_num_list = geom_num_list.copy()  # Bohr
                real_pre_geom = real_initial_geom_num_list.copy()  # Bohr
            
            # Calculate model low layer energy and gradients
            print("Model low layer calculation")
            model_LL_e, model_LL_g, high_layer_geom_num_list, finish_frag = LLSP.single_point(
                file_directory, high_layer_element_list, iter, electric_charge_and_multiplicity, 
                calc_method, geom_num_list=high_layer_geom_num_list*self.bohr2angstroms)
            
            if finish_frag:  # Exit if calculation failed
                break
           
           
            # Perform microiterations to optimize the low layer with fixed high layer
            print("Processing microiteration...")
            
            # Initialize low layer optimizer
            LL_CMV = CalculateMoveVector(self.DELTA, element_list, self.args.saddle_order, self.FC_COUNT, self.temperature)
            LL_optimizer_instances = LL_CMV.initialization(["fire"])
            LL_optimizer_instances[0].display_flag = False
            
            # Variables for tracking convergence
       
            low_layer_converged = False
            
            for microiter in range(self.microiter_num):
              
                # Update model Hessian
                LLSP.Model_hess = LL_Model_hess
                
                # Calculate low layer energy and gradients
                real_LL_e, real_LL_g, geom_num_list, finish_frag = LLSP.single_point(
                    file_directory, element_list, microiter, electric_charge_and_multiplicity, 
                    calc_method, geom_num_list=geom_num_list*self.bohr2angstroms)
                
                # Update model Hessian
                LL_Model_hess = LLSP.Model_hess
                
                
                # Calculate bias potential
                LL_Calc_BiasPot.Model_hess = LL_Model_hess
                _, real_LL_B_e, real_LL_B_g, LL_BPA_hessian = LL_Calc_BiasPot.main(
                    real_LL_e, real_LL_g, geom_num_list, element_list, 
                    force_data, pre_real_LL_B_g, microiter, real_initial_geom_num_list)
                
                # Update optimizer Hessians
                for x in range(len(LL_optimizer_instances)):
                    LL_optimizer_instances[x].set_bias_hessian(LL_BPA_hessian)
                    
                    if microiter % self.FC_COUNT == 0:
                        LL_optimizer_instances[x].set_hessian(LL_Model_hess)
                
                # Apply fragment constraints if specified
                if len(force_data["opt_fragment"]) > 0:
                    real_LL_B_g = copy.copy(self.calc_fragement_grads(real_LL_B_g, force_data["opt_fragment"]))
                    real_LL_g = copy.copy(self.calc_fragement_grads(real_LL_g, force_data["opt_fragment"]))
                
              
                # Save previous geometry for displacement calculation
                prev_geom = geom_num_list.copy()
                
                # Calculate move vector for low layer
                geom_num_list, LL_move_vector, LL_optimizer_instances = LL_CMV.calc_move_vector(
                    microiter, geom_num_list, real_LL_B_g, pre_real_LL_B_g, 
                    real_pre_geom, real_LL_B_e, pre_real_LL_B_e, 
                    pre_real_LL_move_vector, real_initial_geom_num_list, 
                    real_LL_g, pre_real_LL_g, LL_optimizer_instances, print_flag=False)
                
                # Fix high layer atoms to their original positions
                for key, value in highlayer_2_real_label_connect_dict.items():
                    geom_num_list[value-1] = copy.copy(high_layer_geom_num_list[key-1]*self.bohr2angstroms)
                
                # Fix user-specified atoms
                if len(force_data["fix_atoms"]) > 0:
                    for j in force_data["fix_atoms"]:
                        geom_num_list[j-1] = copy.copy(real_initial_geom_num_list[j-1]*self.bohr2angstroms)
                
                # Convert units
                geom_num_list /= self.bohr2angstroms
                
                # Calculate displacement vector for convergence check
                displacement_vector = geom_num_list - prev_geom
                
                # Calculate convergence metrics for low layer atoms only
                low_layer_grads = []
                low_layer_displacements = []
                
                for i in range(len(element_list)):
                    if (i+1) not in high_layer_atom_num:  # 0-indexed to 1-indexed
                        low_layer_grads.append(real_LL_B_g[i])
                        low_layer_displacements.append(displacement_vector[i])
                
                low_layer_grads = np.array(low_layer_grads)
                low_layer_displacements = np.array(low_layer_displacements)
                
                # Calculate RMS and max values for convergence checking
                low_layer_rms_grad = np.sqrt((low_layer_grads**2).mean()) if len(low_layer_grads) > 0 else 0
                max_displacement = np.abs(displacement_vector).max() if len(displacement_vector) > 0 else 0
                rms_displacement = np.sqrt((displacement_vector**2).mean()) if len(displacement_vector) > 0 else 0
                
                # Calculate changes from previous iteration
                energy_shift = -1 * pre_real_LL_B_e + real_LL_B_e
             
                if microiter % 10 == 0:
                    # Print current values
                    print(f"M. ITR. {microiter}")
                    print("Microiteration results:")
                    print(f"LOW LAYER BIAS ENERGY : {float(real_LL_B_e):10.8f}")
                    print(f"LOW LAYER ENERGY      : {float(real_LL_e):10.8f}")
                    print(f"LOW LAYER MAX GRADIENT: {float(low_layer_grads.max()):10.8f}")                   
                    print(f"LOW LAYER RMS GRADIENT: {float(low_layer_rms_grad):10.8f}")
                    print(f"MAX DISPLACEMENT      : {float(max_displacement):10.8f}")
                    print(f"RMS DISPLACEMENT      : {float(rms_displacement):10.8f}")
                    print(f"ENERGY SHIFT          : {float(energy_shift):10.8f}")
                
                # Check convergence of microiterations with defaults from __init__
                if (low_layer_rms_grad < 0.0003) and \
                   (low_layer_grads.max() < 0.0006) and \
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
              
            
            if not low_layer_converged:
                print("Reached maximum number of microiterations.")
            print("Microiteration complete.")
            
            # Calculate high layer energy and gradients
            print("Model system (high layer)")
           
            HLSP.Model_hess = HL_Model_hess
            model_HL_e, model_HL_g, high_layer_geom_num_list, finish_frag = HLSP.single_point(
                file_directory, high_layer_element_list, iter, electric_charge_and_multiplicity,
                method="", geom_num_list=high_layer_geom_num_list*self.bohr2angstroms)
            
            HL_Model_hess = HLSP.Model_hess
            
            if finish_frag:
                break
            
            # Transfer gradients from model to real system
            _, tmp_model_HL_B_e, tmp_model_HL_B_g, HL_BPA_hessian = LL_Calc_BiasPot.main(
                0.0, real_LL_g*0.0, geom_num_list, element_list, force_data, pre_real_LL_B_g*0.0, iter, real_initial_geom_num_list)
            
            tmp_model_HL_g = tmp_model_HL_B_g * 0.0
            
            # Apply high layer gradients to the real system
            for key, value in real_2_highlayer_label_connect_dict.items():
                tmp_model_HL_B_g[key-1] += model_HL_g[value-1] - model_LL_g[value-1]
                tmp_model_HL_g[key-1] += model_HL_g[value-1] - model_LL_g[value-1]
            
            # Extract high layer Hessian
            HL_BPA_hessian = LL_BPA_hessian[np.ix_(bool_list, bool_list)]
         
          
            # Update high layer optimizer Hessians
            for i in range(len(HL_optimizer_instances)):
                HL_optimizer_instances[i].set_bias_hessian(HL_BPA_hessian)
                
                if iter % self.FC_COUNT == 0:
                    HL_optimizer_instances[i].set_hessian(HL_Model_hess[:len(high_layer_atom_num)*3, :len(high_layer_atom_num)*3])
            
            # Apply fragment constraints if specified
            if len(force_data["opt_fragment"]) > 0:
                tmp_model_HL_B_g = copy.copy(self.calc_fragement_grads(tmp_model_HL_B_g, force_data["opt_fragment"]))
                tmp_model_HL_g = copy.copy(self.calc_fragement_grads(tmp_model_HL_g, force_data["opt_fragment"]))
            
            # Combine gradients for full ONIOM model
            model_HL_B_g = copy.copy(model_HL_g)
            model_HL_B_e = model_HL_e + tmp_model_HL_B_e
            
            for key, value in real_2_highlayer_label_connect_dict.items():
                model_HL_B_g[value-1] += tmp_model_HL_B_g[key-1]
            
            # Store previous high layer geometry
            pre_high_layer_geom_num_list = high_layer_geom_num_list
            
            # Calculate move vector for high layer
     
            high_layer_geom_num_list, move_vector, HL_optimizer_instances = HL_CMV.calc_move_vector(
                iter, high_layer_geom_num_list[:len(high_layer_atom_num)], model_HL_B_g[:len(high_layer_atom_num)], pre_model_HL_B_g[:len(high_layer_atom_num)], 
                pre_high_layer_geom_num_list[:len(high_layer_atom_num)], model_HL_B_e, pre_model_HL_B_e, 
                pre_model_HL_move_vector[:len(high_layer_atom_num)], high_layer_pre_geom[:len(high_layer_atom_num)], 
                model_HL_g[:len(high_layer_atom_num)], pre_model_HL_g[:len(high_layer_atom_num)], HL_optimizer_instances)
            
            # Update full system geometry with high layer changes
            for l in range(len(high_layer_geom_num_list) - len(linker_atom_pair_num)):
                geom_num_list[highlayer_2_real_label_connect_dict[l+1]-1] = copy.copy(high_layer_geom_num_list[l]/self.bohr2angstroms)
            
            # Project out translation and rotation
            geom_num_list -= Calculationtools().calc_center_of_mass(geom_num_list, element_list)
            geom_num_list, _ = Calculationtools().kabsch_algorithm(geom_num_list, real_pre_geom)
            
            # Update high layer geometry after alignment
            high_layer_geom_num_list, high_layer_element_list = separate_high_layer_and_low_layer(
                geom_num_list, linker_atom_pair_num, high_layer_atom_num, element_list)
            
            # Combine energies and gradients
            real_e = real_LL_e + model_HL_e - model_LL_e
            real_B_e = real_LL_B_e + model_HL_B_e - model_LL_e
            real_g = real_LL_g + tmp_model_HL_g
            real_B_g = real_LL_B_g + tmp_model_HL_B_g
            
            # Save energy profiles
            self.save_tmp_energy_profiles(iter, real_e, real_g, real_B_g)
            self.ENERGY_LIST_FOR_PLOTTING.append(real_e*self.hartree2kcalmol)
            self.BIAS_ENERGY_LIST_FOR_PLOTTING.append(real_B_e*self.hartree2kcalmol)
            
            
            # Extract geometry information
            self.geom_info_extract(force_data, file_directory, real_B_g, real_g)
            if len(linker_atom_pair_num) > 0:
                tmp_real_B_g = model_HL_B_g[:-len(linker_atom_pair_num)].reshape(-1,1)
            else:
                tmp_real_B_g = model_HL_B_g.reshape(-1,1)
            
            abjusted_high_layer_geom_num_list, _ = Calculationtools().kabsch_algorithm(high_layer_geom_num_list, pre_high_layer_geom_num_list)
            # Calculate displacement and check convergence
            if len(linker_atom_pair_num) > 0:
                tmp_displacement_vector = (abjusted_high_layer_geom_num_list - pre_high_layer_geom_num_list)[:-len(linker_atom_pair_num)].reshape(-1,1)
            else:
                tmp_displacement_vector = (abjusted_high_layer_geom_num_list - pre_high_layer_geom_num_list).reshape(-1,1)
            
            displacement_vector = geom_num_list - real_pre_geom
            converge_flag, max_displacement_threshold, rms_displacement_threshold = self._check_converge_criteria(tmp_real_B_g, tmp_displacement_vector)
            
            # Print optimization information
            self.print_info(real_e, real_B_e, tmp_real_B_g, tmp_displacement_vector, pre_real_e, pre_real_B_e, 
                            max_displacement_threshold, rms_displacement_threshold)
            
            # Track gradients for analysis
            real_grad_list.append(np.sqrt((real_g**2).mean()))
            real_bias_grad_list.append(np.sqrt((real_B_g**2).mean()))
            
            # Check if optimization has converged
            if converge_flag:
                break
            
            # Fix user-specified atoms
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    geom_num_list[j-1] = copy.copy(real_initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            # Check for molecular dissociation
            DC_exit_flag = self.dissociation_check(geom_num_list, element_list)
            if DC_exit_flag:
                break
            
            # Update previous values for next iteration
            pre_real_B_e = real_B_e  # Hartree
            pre_real_e = real_e
            real_pre_geom = geom_num_list  # Bohr
            pre_model_HL_B_g = model_HL_B_g
            pre_model_HL_g = model_HL_g
            pre_model_HL_B_e = model_HL_B_e
            pre_model_HL_move_vector = move_vector
            
            # Create input for next iteration
            geometry_list = FIO.print_geometry_list(geom_num_list*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
        else:
            print("Reached maximum number of iterations. This is not converged.")
            with open(self.BPA_FOLDER_DIRECTORY+"not_converged.txt", "w") as f:
                f.write("Reached maximum number of iterations. This is not converged.")
        
        if DC_exit_flag:
            with open(self.BPA_FOLDER_DIRECTORY+"dissociation_is_detected.txt", "w") as f:
                f.write("These molecules are dissociated.")
        
        # Generate plots and save results
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        self.NUM_LIST = np.array([i for i in range(len(self.ENERGY_LIST_FOR_PLOTTING))])
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.BIAS_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, real_grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, real_bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                G.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)
        
        # Save trajectory and critical points
        FIO.make_traj_file()
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(real_grad_list, "local_min_grad", "min")
        
        # Save energy profiles
        self._save_energy_profiles()
        
        # Store final results
        self.final_file_directory = file_directory
        self.final_geometry = geom_num_list  # Bohr
        self.final_energy = real_e  # Hartree
        self.final_bias_energy = real_B_e  # Hartree
        return
    
    def get_result_file_path(self):
        """
        Sets the absolute file paths for optimization results as instance variables after run() is executed.
        Before calling this method, run() must have been executed,
        and self.BPA_FOLDER_DIRECTORY and self.START_FILE must be set.
        The file names will be xxx_optimized.xyz / xxx_traj.xyz, where xxx is the input file name.
        """
        try:
            if (hasattr(self, 'BPA_FOLDER_DIRECTORY') and self.BPA_FOLDER_DIRECTORY and
                hasattr(self, 'START_FILE') and self.START_FILE):
                
                # Get the base name (xxx) from the input file name, removing the extension
                base_name = os.path.splitext(os.path.basename(self.START_FILE))[0]
                
                # Build file names based on user-specified naming convention
                # (Assuming xxx_optimized.py was a typo for .xyz)
                optimized_filename = f"{base_name}_optimized.xyz"
                traj_filename = f"{base_name}_traj.xyz"

                # Set the full, absolute file paths as instance variables
                # Use os.path.abspath to ensure the path is absolute
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
        if type(self.args.INPUT) is str:
            START_FILE_LIST = [self.args.INPUT]
        else:
            START_FILE_LIST = self.args.INPUT #
        
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
            
            
            self._make_init_directory(file)
            if len(self.args.oniom_flag) > 0:
                self.optimize_oniom()
            else:
                self.optimize()
        
            if self.CMDS:
                CMDPA = CMDSPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.ENERGY_LIST_FOR_PLOTTING, self.BIAS_ENERGY_LIST_FOR_PLOTTING)
                CMDPA.main()
            if self.PCA:
                PCAPA = PCAPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.ENERGY_LIST_FOR_PLOTTING, self.BIAS_ENERGY_LIST_FOR_PLOTTING)
                PCAPA.main()
            
            if len(self.irc) > 0:
                if self.args.usextb != "None":
                    xtb_method = self.args.usextb
                else:
                    xtb_method = "None"
                if self.iter % self.FC_COUNT == 0:
                    hessian = self.Model_hess
                else:
                    hessian = None
                EXEC_IRC = IRC(self.BPA_FOLDER_DIRECTORY, self.final_file_directory, self.irc, self.SP, self.element_list, self.electric_charge_and_multiplicity, self.force_data, xtb_method, FC_count=int(self.FC_COUNT), hessian=hessian) 
                EXEC_IRC.run()
                self.irc_terminal_struct_paths = EXEC_IRC.terminal_struct_paths
            else:
                self.irc_terminal_struct_paths = []
            
            print(f"Trial of geometry optimization ({file}) was completed.")
        print("All calculations were completed.")
        
        self.get_result_file_path()
        
        
        return
