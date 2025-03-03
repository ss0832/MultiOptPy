import sys
import os
import copy
import glob
import itertools
import datetime
import time

import numpy as np

from optimizer import CalculateMoveVector
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib, element_number
from interface import force_data_parser
from approx_hessian import ApproxHessian
from cmds_analysis import CMDSPathAnalysis
from pca_analysis import PCAPathAnalysis
from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from MO_analysis import NROAnalysis
from constraint_condition import ProjectOutConstrain
from irc import IRC
from bond_connectivity import judge_shape_condition


class Optimize:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol
        self.set_convergence_criteria(args)
        self.initialize_variables(args)

    def set_convergence_criteria(self, args):
        if args.tight_convergence_criteria and not args.loose_convergence_criteria:
            self.MAX_FORCE_THRESHOLD = 0.00012
            self.RMS_FORCE_THRESHOLD = 0.00008
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0006
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0003
        elif not args.tight_convergence_criteria and args.loose_convergence_criteria:
            self.MAX_FORCE_THRESHOLD = 0.0030
            self.RMS_FORCE_THRESHOLD = 0.0020
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0150
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0100
        else:
            self.MAX_FORCE_THRESHOLD = 0.0003
            self.RMS_FORCE_THRESHOLD = 0.0002
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0015
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0010

    def initialize_variables(self, args):
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
        self.check_sub_basisset(args)
        self.Model_hess = None
        self.mFC_COUNT = args.calc_model_hess
        self.DC_check_dist = float(args.dissociate_check)
        self.unrestrict = args.unrestrict
        self.NRO_analysis = args.NRO_analysis
        self.check_NRO_analysis(args)
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

    def check_sub_basisset(self, args):
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

    def check_NRO_analysis(self, args):
        if self.NRO_analysis:
            if args.usextb == "None":
                print("Currently, Natural Reaction Orbital analysis is only available for xTB method.")
                sys.exit(0)
        

    def make_init_directory(self, file):
        """
        Create initial directory for optimization results.
        """
        self.START_FILE = file
        timestamp = str(time.time()).replace(".", "_")
        date = str(datetime.datetime.now().date())
        base_dir = f"{date}/{self.START_FILE[:-4]}_OPT_"

        if self.args.othersoft != "None":
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}ASE_{timestamp}/"
        elif self.args.usextb == "None" and self.args.usedxtb == "None":
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}{self.FUNCTIONAL}_{self.BASIS_SET}_{timestamp}/"
        else:
            method = self.args.usedxtb if self.args.usedxtb != "None" else self.args.usextb
            self.BPA_FOLDER_DIRECTORY = f"{base_dir}{method}_{timestamp}/"
        
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        
    def save_input_data(self):
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        return

    def constrain_flag_check(self, force_data):
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

    def init_projection_constraint(self, PC, geom_num_list, iter, projection_constrain):
        if iter == 0:
            if projection_constrain:
                PC.initialize(geom_num_list)
            else:
                pass
            return PC
        else:
            return PC

    def save_init_geometry(self, geom_num_list, element_list, allactive_flag):
        
        if allactive_flag:
            initial_geom_num_list = geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
            pre_geom = initial_geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
        else:
            initial_geom_num_list = geom_num_list 
            pre_geom = initial_geom_num_list 
        
        return initial_geom_num_list, pre_geom


    def calc_eff_hess_for_fix_atoms_and_set_hess(self, allactive_flag, force_data, BPA_hessian, n_fix, optimizer_instances, geom_num_list, B_g, g, projection_constrain, PC):
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
                proj_bpa_hess = PC.calc_project_out_hess(geom_num_list, B_g - g, BPA_hessian)
                optimizer_instances[i].set_bias_hessian(proj_bpa_hess)
            else:
                optimizer_instances[i].set_bias_hessian(BPA_hessian)
            
            if self.iter % self.FC_COUNT == 0 or (self.args.use_model_hessian and self.iter % self.mFC_COUNT == 0):
                if not allactive_flag:
                    self.Model_hess -= np.dot(self.Model_hess[:, fix_num], np.dot(inv_tmp_fix_hess, self.Model_hess[fix_num, :]))
                
                
                if projection_constrain:
                    proj_model_hess = PC.calc_project_out_hess(geom_num_list, g, self.Model_hess)
                    optimizer_instances[i].set_hessian(proj_model_hess)
                else:
                    optimizer_instances[i].set_hessian(self.Model_hess)
        
        return optimizer_instances
        

    def apply_projection_constraints(self, projection_constrain, PC, geom_num_list, g, B_g):
        if projection_constrain:
            g = copy.copy(PC.calc_project_out_grad(geom_num_list, g))
            proj_d_B_g = copy.copy(PC.calc_project_out_grad(geom_num_list, B_g - g))
            B_g = copy.copy(g + proj_d_B_g)
        
        return g, B_g, PC

    def zero_fixed_atom_gradients(self, allactive_flag, force_data, g, B_g):
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                g[j-1] = copy.copy(g[j-1]*0.0)
                B_g[j-1] = copy.copy(B_g[j-1]*0.0)
        
        return g, B_g

    def project_out_translation_rotation(self, new_geometry, geom_num_list, allactive_flag):
        
        if allactive_flag:
            # Convert to Bohr, apply Kabsch alignment algorithm, then convert back
            aligned_geometry, _ = Calculationtools().kabsch_algorithm(
                new_geometry/self.bohr2angstroms, geom_num_list)
            aligned_geometry *= self.bohr2angstroms
            return aligned_geometry
        else:
            # If not all atoms are active, return the original geometry
            return new_geometry

    def apply_projection_constraints_to_geometry(self, projection_constrain, PC, new_geometry):
        if projection_constrain:
            tmp_new_geometry = new_geometry / self.bohr2angstroms
            adjusted_geometry = PC.adjust_init_coord(tmp_new_geometry) * self.bohr2angstroms
            return adjusted_geometry
        
        return new_geometry, PC

    def reset_fixed_atom_positions(self, new_geometry, initial_geom_num_list, allactive_flag, force_data):
        
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
        
        return new_geometry


    def initialize_optimization_variables(self):
        # Initialize return dictionary with categories
        vars_dict = {
            'calculation': {},  # Calculation modules and algorithms
            'io': {},           # File input/output handlers
            'energy': {},       # Energy-related variables
            'geometry': {},     # Geometry and structure variables
            'gradients': {},    # Gradient and force variables
            'constraints': {},  # Constraint related variables
            'optimization': {}, # Optimizer settings
            'misc': {}          # Miscellaneous
        }
        
        # Calculation modules and file I/O
        Calculation, xtb_method = self.import_calculation_module()
        self.save_input_data()
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        
        # Initialize energy tracking arrays
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
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
        orthogonal_bias_grad_list = []
        orthogonal_grad_list = []
        
        # Element information
        element_number_list = np.array([element_number(elem) for elem in element_list], dtype="int")
        natom = len(element_list)
        
        # Constraint setup
        PC = ProjectOutConstrain(force_data["projection_constraint_condition_list"], 
                                force_data["projection_constraint_atoms"], 
                                force_data["projection_constraint_constant"])
        projection_constrain, allactive_flag = self.constrain_flag_check(force_data)
        
        # Bias potential and calculation setup
        self.CalcBiaspot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        SP = self.setup_calculation(Calculation)
        
        # Move vector calculation
        CMV = CalculateMoveVector(self.DELTA, element_list, self.args.saddle_order, 
                                self.FC_COUNT, self.temperature, self.args.use_model_hessian)
        optimizer_instances = CMV.initialization(force_data["opt_method"])
        
        # Initialize optimizer instances
        for i in range(len(optimizer_instances)):
            optimizer_instances[i].set_hessian(self.Model_hess)
            if self.DELTA != "x":
                optimizer_instances[i].DELTA = self.DELTA
        
        # NRO analysis setup
        if self.NRO_analysis:
            NRO = NROAnalysis(file_directory=self.BPA_FOLDER_DIRECTORY, 
                            xtb=xtb_method, 
                            element_list=element_list, 
                            electric_charge_and_multiplicity=electric_charge_and_multiplicity)
        else:
            NRO = None
        
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
            'orthogonal_bias_grad_list': orthogonal_bias_grad_list,
            'orthogonal_grad_list': orthogonal_grad_list,
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
            'NRO': NRO
        }
        
        return vars_dict

    def optimize(self):
        # Initialize all variables needed for optimization
        vars_dict = self.initialize_optimization_variables()
        
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
        orthogonal_bias_grad_list = vars_dict['gradients']['orthogonal_bias_grad_list']
        orthogonal_grad_list = vars_dict['gradients']['orthogonal_grad_list']
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
        NRO = vars_dict['misc']['NRO']

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
                
            if iter % self.mFC_COUNT == 0 and self.args.use_model_hessian and self.FC_COUNT < 1:
                SP.Model_hess = ApproxHessian().main(geom_num_list, element_list, g)
            
            if iter == 0:
                initial_geom_num_list, pre_geom = self.save_init_geometry(geom_num_list, element_list, allactive_flag)

            _, B_e, B_g, BPA_hessian = self.CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)
            
            PC = self.init_projection_constraint(PC, geom_num_list, iter, projection_constrain)
            
            optimizer_instances = self.calc_eff_hess_for_fix_atoms_and_set_hess(allactive_flag, force_data, BPA_hessian, n_fix, optimizer_instances, geom_num_list, B_g, g, projection_constrain, PC)
                    
            if not allactive_flag:
                B_g = copy.copy(self.calc_fragement_grads(B_g, force_data["opt_fragment"]))
                g = copy.copy(self.calc_fragement_grads(g, force_data["opt_fragment"]))
            
            #energy profile 
            self.save_tmp_energy_profiles(iter, e, g, B_g)
            
            g, B_g, PC = self.apply_projection_constraints(projection_constrain, PC, geom_num_list, g, B_g)
                
            g, B_g = self.zero_fixed_atom_gradients(allactive_flag, force_data, g, B_g)

            new_geometry, move_vector, optimizer_instances = CMV.calc_move_vector(iter, geom_num_list,
                                                                                  B_g, pre_B_g, pre_geom, B_e, pre_B_e,
                                                                                  pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances, projection_constrain)
            
          
            new_geometry = self.project_out_translation_rotation(new_geometry, geom_num_list, allactive_flag)
            
            new_geometry, PC = self.apply_projection_constraints_to_geometry(projection_constrain, PC, new_geometry)
            
            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)   
            
            if self.iter == 0:
                displacement_vector = move_vector
            else:
                displacement_vector = new_geometry / self.bohr2angstroms - geom_num_list
            
            converge_flag, max_displacement_threshold, rms_displacement_threshold = self.check_converge_criteria(B_g, displacement_vector)
            
            self.print_info(e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold)
            
            grad_list.append(np.sqrt((g[g > 1e-10]**2).mean()))
            bias_grad_list.append(np.sqrt((B_g[B_g > 1e-10]**2).mean()))
            
            if iter > 0:
                RMS_ortho_B_g, RMS_ortho_g = self.calculate_orthogonal_gradients(pre_move_vector, B_g, g)
                orthogonal_bias_grad_list.append(RMS_ortho_B_g)
                orthogonal_grad_list.append(RMS_ortho_g)
            
            if self.NRO_analysis:
                NRO.run(SP, geom_num_list, move_vector)
            
            if converge_flag:
                if projection_constrain and iter == 0:
                    pass
                else:
                    optimized_flag = True
                    break

            new_geometry = self.reset_fixed_atom_positions(new_geometry, initial_geom_num_list, allactive_flag, force_data)
            #dissociation check
            DC_exit_flag = self.dissociation_check(new_geometry, element_list)
            
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

        self.finalize_optimization(FIO, G, grad_list, bias_grad_list,
                                   orthogonal_bias_grad_list, orthogonal_grad_list,
                                   file_directory, self.force_data, geom_num_list, e, B_e, SP, NRO)
        self.optimized_flag = optimized_flag
        return

    def finalize_optimization(self, FIO, G, grad_list, bias_grad_list, orthogonal_bias_grad_list, orthogonal_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, NRO):
        self.save_results(FIO, G, grad_list, bias_grad_list, orthogonal_bias_grad_list, orthogonal_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, NRO)
        self.bias_pot_params_grad_list = self.CalcBiaspot.bias_pot_params_grad_list
        self.bias_pot_params_grad_name_list = self.CalcBiaspot.bias_pot_params_grad_name_list
        self.final_file_directory = file_directory
        self.final_geometry = geom_num_list  # Bohr
        self.final_energy = e  # Hartree
        self.final_bias_energy = B_e  # Hartree

    def calculate_orthogonal_gradients(self, pre_move_vector, B_g, g):
        norm_pre_move_vec = (pre_move_vector / np.linalg.norm(pre_move_vector)).reshape(len(pre_move_vector) * 3, 1)
        orthogonal_bias_grad = B_g.reshape(len(B_g) * 3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
        orthogonal_grad = g.reshape(len(g) * 3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
        RMS_ortho_B_g = abs(np.sqrt((orthogonal_bias_grad**2).mean()))
        RMS_ortho_g = abs(np.sqrt((orthogonal_grad**2).mean()))
        return RMS_ortho_B_g, RMS_ortho_g

    def save_results(self, FIO, G, grad_list, bias_grad_list, orthogonal_bias_grad_list, orthogonal_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, NRO):
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        
        if self.NRO_analysis:
            NRO.save_results(self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)

        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                G.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)

        FIO.make_traj_file()
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")

        self.save_energy_profiles(orthogonal_bias_grad_list, orthogonal_grad_list, grad_list)

        print("Complete...")
        self.SP = SP
        self.final_file_directory = file_directory
        self.final_geometry = geom_num_list  # Bohr
        self.final_energy = e  # Hartree
        self.final_bias_energy = B_e  # Hartree
        return

    def check_converge_criteria(self, B_g, displacement_vector):
        max_force = abs(B_g.max())
        max_force_threshold = self.MAX_FORCE_THRESHOLD
        
        rms_force = abs(np.sqrt(np.mean(B_g[B_g > 1e-10]**2.0)))
        rms_force_threshold = self.RMS_FORCE_THRESHOLD
        
        delta_max_force_threshold = max(0.0, max_force_threshold -1 * max_force)
        delta_rms_force_threshold = max(0.0, rms_force_threshold -1 * rms_force)
        
        max_displacement = abs(displacement_vector.max())
        max_displacement_threshold = max(self.MAX_DISPLACEMENT_THRESHOLD, self.MAX_DISPLACEMENT_THRESHOLD + delta_max_force_threshold)
        rms_displacement = abs(np.sqrt((displacement_vector[displacement_vector > 1e-10]**2).mean()))
        rms_displacement_threshold = max(self.RMS_DISPLACEMENT_THRESHOLD, self.RMS_DISPLACEMENT_THRESHOLD + delta_rms_force_threshold)
        
        if max_force < max_force_threshold and rms_force < rms_force_threshold and max_displacement < max_displacement_threshold and rms_displacement < rms_displacement_threshold:#convergent criteria
            return True, max_displacement_threshold, rms_displacement_threshold
        return False, max_displacement_threshold, rms_displacement_threshold
    
    def import_calculation_module(self):
        xtb_method = None
        if self.args.pyscf:
            from pyscf_calculation_tools import Calculation
          
        elif self.args.othersoft and self.args.othersoft != "None":
            from ase_calculation_tools import Calculation
            
            print("Use", self.args.othersoft)
            with open(self.BPA_FOLDER_DIRECTORY + "use_" + self.args.othersoft + ".txt", "w") as f:
                f.write(self.args.othersoft + "\n")
                f.write(self.BASIS_SET + "\n")
                f.write(self.FUNCTIONAL + "\n")
        else:
            if self.args.usedxtb and self.args.usedxtb != "None":
                from dxtb_calculation_tools import Calculation
              
                xtb_method = self.args.usedxtb
            elif self.args.usextb and self.args.usextb != "None":
                from tblite_calculation_tools import Calculation
               
                xtb_method = self.args.usextb
            else:
                from psi4_calculation_tools import Calculation
               

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
            excited_state=self.excited_state)
        SP.cpcm_solv_model = self.cpcm_solv_model
        SP.alpb_solv_model = self.alpb_solv_model
        return SP

    def write_input_files(self, FIO):
        if os.path.splitext(FIO.START_FILE)[1] == ".gjf":
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.read_gjf_file(self.electric_charge_and_multiplicity)
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
            f.write(str(np.sqrt((g[g > 1e-10]**2).mean()))+"\n")
        #-------------------
        if iter == 0:
            with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                f.write("bias gradient (RMS) [hartree/Bohr] \n")
        with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
            f.write(str(np.sqrt((B_g[B_g > 1e-10]**2).mean()))+"\n")
            #-------------------
        return
    
    def save_energy_profiles(self, orthogonal_bias_grad_list, orthogonal_grad_list, grad_list):
        if len(orthogonal_bias_grad_list) != 0 and len(orthogonal_grad_list) != 0:
            with open(self.BPA_FOLDER_DIRECTORY+"orthogonal_bias_gradient_profile.csv","w") as f:
                f.write("ITER.,orthogonal bias gradient[a.u.]\n")
                for i in range(len(orthogonal_bias_grad_list)):
                    f.write(str(i+1)+","+str(float(orthogonal_bias_grad_list[i]))+"\n")
            
            with open(self.BPA_FOLDER_DIRECTORY+"orthogonal_gradient_profile.csv","w") as f:
                f.write("ITER.,orthogonal gradient[a.u.]\n")
                for i in range(len(orthogonal_bias_grad_list)):
                    f.write(str(i+1)+","+str(float(orthogonal_grad_list[i]))+"\n")
            
            with open(self.BPA_FOLDER_DIRECTORY+"orthogonal_gradient_diff_profile.csv","w") as f:
                f.write("ITER.,orthogonal gradient[a.u.]\n")
                for i in range(len(orthogonal_grad_list)):
                    f.write(str(i+1)+","+str(float(orthogonal_grad_list[i]-grad_list[i+1]))+"\n")
            with open(self.BPA_FOLDER_DIRECTORY+"orthogonal_bias_gradient_diff_profile.csv","w") as f:
                f.write("ITER.,orthogonal gradient[a.u.]\n")
                for i in range(len(orthogonal_bias_grad_list)):
                    f.write(str(i+1)+","+str(float(orthogonal_bias_grad_list[i]-grad_list[i+1]))+"\n")  
        
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
            
            for j in tmp_fragm_list:
                atom_label_list.remove(j)
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
    
    def print_info(self, e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold):
        
        rms_force = abs(np.sqrt((B_g[B_g > 1e-10]**2).mean())) + 1e-13
        rms_displacement = abs(np.sqrt((displacement_vector[displacement_vector > 1e-10]**2).mean())) + 1e-13
        
        print("caluculation results (unit a.u.):")
        print("                         Value                         Threshold ")
        print("ENERGY                : {:>15.12f} ".format(e))
        print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
        print("Maximum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
        print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(rms_force, self.RMS_FORCE_THRESHOLD))
        print("Maximum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), max_displacement_threshold))
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
    
    def run(self):
        if type(self.args.INPUT) is str:
            START_FILE_LIST = [self.args.INPUT]
        else:
            START_FILE_LIST = self.args.INPUT #
        for file in START_FILE_LIST:
            print("********************************")
            print(file)
            print("********************************")
            if os.path.exists(file) == False:
                print(f"{file} does not exist.")
                continue
            
            self.make_init_directory(file)
            self.optimize()
        
            if self.CMDS:
                CMDPA = CMDSPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
                CMDPA.main()
            if self.PCA:
                PCAPA = PCAPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
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
            print(f"Geometry optimization of {file} was completed.")
        print("All calculations are completed.")
        return
