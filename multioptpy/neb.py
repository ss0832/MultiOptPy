import os
import numpy as np
import sys
import glob
import datetime
import matplotlib.pyplot as plt
import random
import copy
import re
from scipy.signal import argrelextrema
from abc import ABC, abstractmethod

try:
    import psi4
except:
    psi4 = None
    print("You can't use psi4.")

try:
    from tblite.interface import Calculator
except:
    print("You can't use extended tight binding method.")

try:
    import pyscf
    from pyscf import tdscf
    from pyscf.hessian import thermo
except:
    print("You can't use pyscf.")

try:
    import dxtb
    dxtb.timer.disable()
except:
    print("You can't use dxtb.")

try:
    import torch
except:
    print("You can't use pytorch.")

from interface import force_data_parser
from parameter import element_number
from potential import BiasPotentialCalculation
from pathopt_bneb_force import CaluculationBNEB, CaluculationBNEB2
from pathopt_dneb_force import CaluculationDNEB
from pathopt_nesb_force import CaluculationNESB
from pathopt_lup_force import CaluculationLUP
from pathopt_om_force import CaluculationOM
from pathopt_ewbneb_force import CaluculationEWBNEB
from calc_tools import Calculationtools
from idpp import IDPP
from Optimizer.rfo import RationalFunctionOptimization 
from interpolation import spline_interpolation
from constraint_condition import ProjectOutConstrain
from fileio import xyz2list, traj2list, FileIO
from multioptpy.Optimizer import lbfgs_neb 
from multioptpy.Optimizer import conjugate_gradient_neb
from multioptpy.Optimizer import trust_radius_neb 
from multioptpy.Optimizer import rsirfo
from multioptpy.Optimizer.fire_neb import FIREOptimizer
from multioptpy.Optimizer.rfo_neb import RFOOptimizer
from multioptpy.Optimizer.gradientdescent_neb import SteepestDescentOptimizer
from approx_hessian import ApproxHessian
from visualization import NEBVisualizer
from tblite_calculation_tools import TBLiteEngine
from pyscf_calculation_tools import PySCFEngine
from psi4_calculation_tools import Psi4Engine
from dxtb_calculation_tools import DXTBEngine
from ase_calculation_tools import ASEEngine

class NEBConfig:
    """Configuration management class for NEB calculations"""
    
    def __init__(self, args):
        # Basic calculation settings
        self.functional = args.functional
        self.basisset = args.basisset
        self.basic_set_and_function = args.functional + "/" + args.basisset
        self.FUNCTIONAL = args.functional
        
        # Solvent model settings
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        
        # Computational settings
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.pyscf = args.pyscf
        self.usextb = args.usextb
        self.usedxtb = args.usedxtb
        
        # NEB specific settings
        self.NEB_NUM = args.NSTEP
        self.partition = args.partition
        self.APPLY_CI_NEB = args.apply_CI_NEB
        self.om = args.OM
        self.lup = args.LUP
        self.dneb = args.DNEB
        self.nesb = args.NESB
        self.bneb = args.BNEB
        self.ewbneb = args.EWBNEB
        
        # Optimization settings
        self.FC_COUNT = args.calc_exact_hess
        self.MFC_COUNT = int(args.calc_model_hess)
        self.model_hessian = args.use_model_hessian
        self.climbing_image_start = args.climbing_image[0]
        self.climbing_image_interval = args.climbing_image[1]
        self.sd = args.steepest_descent
        self.cg_method = args.conjugate_gradient
        self.lbfgs_method = args.memory_limited_BFGS
        
        # Flags
        self.IDPP_flag = args.use_image_dependent_pair_potential
        self.align_distances = args.align_distances
        self.excited_state = args.excited_state
        self.unrestrict = args.unrestrict
        self.save_pict = args.save_pict
        self.apply_convergence_criteria = args.apply_convergence_criteria
        self.node_distance = args.node_distance
        
        # Electronic state settings
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        
        # Constants
        self.bohr2angstroms = 0.52917721067
        self.hartree2kcalmol = 627.509
        
        # Additional settings
        self.dft_grid = int(args.dft_grid)
        self.spring_constant_k = 0.01
        self.force_const_for_cineb = 0.01
        self.othersoft = args.othersoft
        
        # FIRE method parameters
        self.FIRE_dt = 0.1
        self.dt = 0.5
        self.a = 0.10
        self.n_reset = 0
        self.FIRE_N_accelerate = 5
        self.FIRE_f_inc = 1.10
        self.FIRE_f_accelerate = 0.99
        self.FIRE_f_decelerate = 0.5
        self.FIRE_a_start = 0.1
        self.FIRE_dt_max = 1.0
        
        # Initialize derived settings
        self.set_sub_basisset(args)
        self.set_fixed_edges(args)
        
        # Input file and directory settings
        self.init_input = args.INPUT
        self.NEB_FOLDER_DIRECTORY = self.make_neb_work_directory(args.INPUT)
        
    def set_sub_basisset(self, args):
        """Set up basis set configuration"""
        if len(args.sub_basisset) % 2 != 0:
            print("invalid input (-sub_bs)")
            sys.exit(0)
        
        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(args.basisset)
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
            else:
                self.SUB_BASIS_SET = {"default": args.basisset}
        else:
            self.SUB_BASIS_SET = args.basisset
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET += "\nassign " + str(args.basisset) + "\n"
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign " + args.sub_basisset[2*j] + " " + args.sub_basisset[2*j+1] + "\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
        
        # ECP settings
        if len(args.effective_core_potential) % 2 != 0:
            print("invalid input (-ecp)")
            sys.exit(0)
            
        if args.pyscf:
            self.ECP = {}
            if len(args.effective_core_potential) > 0:
                for j in range(int(len(args.effective_core_potential)/2)):
                    self.ECP[args.effective_core_potential[2*j]] = args.effective_core_potential[2*j+1]
        else:
            self.ECP = ""
    
    def set_fixed_edges(self, args):
        """Set up edge fixing configuration"""
        if args.fixedges <= 0:
            self.fix_init_edge = False
            self.fix_end_edge = False
        elif args.fixedges == 1:
            self.fix_init_edge = True
            self.fix_end_edge = False
        elif args.fixedges == 2:
            self.fix_init_edge = False
            self.fix_end_edge = True
        else:
            self.fix_init_edge = True
            self.fix_end_edge = True
    
    def make_neb_work_directory(self, input_file):
        """Create NEB working directory path"""
        if os.path.splitext(input_file)[1] == ".xyz":
            tmp_name = os.path.splitext(input_file)[0] 
        else:
            tmp_name = input_file 
        
        timestamp = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])
        if self.othersoft != "None":
            return tmp_name + "_NEB_" + self.othersoft + "_" + timestamp + "/"


        elif self.usextb == "None" and self.usedxtb == "None":
            return tmp_name + "_NEB_" + self.basic_set_and_function.replace("/", "_") + "_" + timestamp + "/"
        else:
            if self.usextb != "None":
                return tmp_name + "_NEB_" + self.usextb + "_" + timestamp + "/"
            else:
                return tmp_name + "_NEB_" + self.usedxtb + "_" + timestamp + "/"


class CalculationEngineFactory:
    """Factory class for creating calculation engines"""
    
    @staticmethod
    def create_engine(config):
        """Create appropriate calculation engine based on configuration"""
        if config.othersoft != "None":
            return ASEEngine()    
        elif config.usextb != "None":
            return TBLiteEngine()
        elif config.usedxtb != "None":
            return DXTBEngine()
        elif config.pyscf:
            return PySCFEngine()
        else:
            return Psi4Engine()


class OptimizationFactory:
    """Factory class for creating optimization algorithms"""
    
    @staticmethod
    def create_optimizer(method, config):
        """Create appropriate optimizer based on method and configuration"""
        if method == "fire":
            return FIREOptimizer(config)
        elif method == "steepest_descent":
            return SteepestDescentOptimizer(config)
        elif method == "rfo":
            return RFOOptimizer(config)
        elif method == "lbfgs":
            tr_neb = trust_radius_neb.TR_NEB(
                NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY,
                fix_init_edge=config.fix_init_edge,
                fix_end_edge=config.fix_end_edge,
                apply_convergence_criteria=config.apply_convergence_criteria
            )
            return lbfgs_neb.LBFGS_NEB(TR_NEB=tr_neb)
        elif method == "conjugate_gradient":
            tr_neb = trust_radius_neb.TR_NEB(
                NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY,
                fix_init_edge=config.fix_init_edge,
                fix_end_edge=config.fix_end_edge,
                apply_convergence_criteria=config.apply_convergence_criteria
            )
            return conjugate_gradient_neb.ConjugateGradientNEB(TR_NEB=tr_neb, cg_method=config.cg_method)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")


class NEB:
    """Main NEB (Nudged Elastic Band) calculation class (Refactored version)"""
    
    def __init__(self, args):
        # Initialize configuration
        self.config = NEBConfig(args)
        
        # Create calculation engine
        self.calculation_engine = CalculationEngineFactory.create_engine(self.config)
        
        # Initialize visualizer if needed
        if self.config.save_pict:
            self.visualizer = NEBVisualizer(self.config)
        
        # Create working directory
        os.mkdir(self.config.NEB_FOLDER_DIRECTORY)
        
        # Store original args for backward compatibility
        self.args = args
        
        # Set element list (will be initialized in run method)
        self.element_list = None
    
    def run(self):
        """Execute NEB calculation"""
        # Load and prepare geometries
        geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list(
            self.config.init_input, self.config.partition)
        self.element_list = element_list
        self.config.element_list = element_list  # Add to config for optimizer access
        
        # Create initial input files
        file_directory = self.make_input_files(geometry_list, 0)
        
        # Initialize calculation variables
        pre_total_velocity = [[[]]]
        force_data = force_data_parser(self.args)
        
        # Check for projection constraints
        if len(force_data["projection_constraint_condition_list"]) > 0:
            projection_constraint_flag = True
        else:
            projection_constraint_flag = False        
        
        # Get element number list
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        
        # Save input configuration
        with open(self.config.NEB_FOLDER_DIRECTORY + "input.txt", "w") as f:
            f.write(str(vars(self.args)))
        
        # Setup force calculation method
        STRING_FORCE_CALC = self._setup_force_calculation()
        
        # Check for fixed atoms
        if len(force_data["fix_atoms"]) > 0:
            fix_atom_flag = True
        else:
            fix_atom_flag = False
        
        # Initialize optimization variables - FIXED: Proper initialization order
        pre_geom = None
        pre_total_force = None
        pre_biased_gradient_list = None
        pre_total_velocity = []
        total_velocity = []
        pre_biased_energy_list = None  # FIXED: Initialize this variable
        
        # Check for conflicting optimization methods
        if self.config.cg_method and self.config.lbfgs_method: 
            print("You can not use CG and LBFGS at the same time.")
            exit()
        
        # Setup optimizer
        optimizer = self._setup_optimizer()
        
        # Main NEB iteration loop
        for optimize_num in range(self.config.NEB_NUM):
            exit_file_detect = os.path.exists(self.config.NEB_FOLDER_DIRECTORY + "end.txt")
            if exit_file_detect:
                if psi4:
                    psi4.core.clean()
                break
            
            print(f"\n\n\n NEB: ITR.  {optimize_num}  \n\n\n")
            self.make_traj_file(file_directory)
            
            # Calculate energy and gradients
            energy_list, gradient_list, geometry_num_list, pre_total_velocity = \
                self.calculation_engine.calculate(file_directory, optimize_num, 
                                                pre_total_velocity, self.config)
            
            if optimize_num == 0:
                init_geometry_num_list = geometry_num_list
            
            # Apply bias potential - FIXED: Check if hessian files exist before using them
            biased_energy_list, biased_gradient_list = self._apply_bias_potential(
                energy_list, gradient_list, geometry_num_list, element_list, force_data, optimize_num)
            
            # FIXED: Initialize pre_biased_energy_list on first iteration
            if optimize_num == 0:
                pre_biased_energy_list = copy.copy(biased_energy_list)
            
            # Calculate model hessian if needed
            if (self.config.FC_COUNT == -1 and self.config.model_hessian and 
                optimize_num % self.config.MFC_COUNT == 0):
                for i in range(len(geometry_num_list)):
                    hessian = ApproxHessian().main(geometry_num_list[i], element_list, 
                                                 gradient_list[i], approx_hess_type=self.config.model_hessian)
                    np.save(self.config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(i) + ".npy", hessian)
            
            # Initialize projection constraints
            if projection_constraint_flag and optimize_num == 0:
                PC_list = []
                for i in range(len(energy_list)):
                    PC_list.append(ProjectOutConstrain(
                        force_data["projection_constraint_condition_list"], 
                        force_data["projection_constraint_atoms"], 
                        force_data["projection_constraint_constant"]))
                    PC_list[i].initialize(geometry_num_list[i])

            # Apply projection constraints
            if projection_constraint_flag:
                for i in range(len(energy_list)):
                    biased_gradient_list[i] = copy.copy(PC_list[i].calc_project_out_grad(
                        geometry_num_list[i], biased_gradient_list[i]))            

            # Calculate forces
            total_force = STRING_FORCE_CALC.calc_force(
                geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list)

            # Calculate analysis metrics
            cos_list, tot_force_rms_list, tot_force_max_list, bias_force_rms_list = \
                self._calculate_analysis_metrics(total_force, biased_gradient_list)
            
            # Save analysis data and create plots
            self._save_analysis_data(cos_list, tot_force_rms_list, tot_force_max_list, 
                                   bias_force_rms_list, file_directory, optimize_num)
            
            total_velocity = self.force2velocity(total_force, element_list)
            
            # Perform optimization step
            new_geometry = self._perform_optimization_step(
                optimizer, geometry_num_list, total_force, biased_gradient_list, 
                pre_geom, pre_biased_gradient_list, optimize_num, biased_energy_list, 
                pre_biased_energy_list, pre_total_velocity, total_velocity, 
                cos_list, pre_geom, STRING_FORCE_CALC)
            
            # Apply climbing image if needed
            if (optimize_num > self.config.climbing_image_start and 
                (optimize_num - self.config.climbing_image_start) % self.config.climbing_image_interval == 0):
                new_geometry = apply_climbing_image(new_geometry, biased_energy_list)
            
            # Apply constraints
            new_geometry = self._apply_constraints(
                new_geometry, fix_atom_flag, force_data, init_geometry_num_list, 
                projection_constraint_flag, PC_list if projection_constraint_flag else None)
            
            # Align geometries if needed
            if self.config.align_distances >= 1 and optimize_num % self.config.align_distances == 0 and optimize_num > 0:
                print("Aligning geometries...")
                tmp_new_geometry = distribute_geometry(np.array(new_geometry))
                for k in range(len(new_geometry)):
                    new_geometry[k] = copy.copy(tmp_new_geometry[k])
            
            # Save analysis files
            tmp_instance_fileio = FileIO(file_directory + "/", "dummy.txt")
            tmp_instance_fileio.argrelextrema_txt_save(biased_energy_list, "approx_TS_node", "max")
            tmp_instance_fileio.argrelextrema_txt_save(biased_energy_list, "approx_EQ_node", "min")
            tmp_instance_fileio.argrelextrema_txt_save(bias_force_rms_list, "local_min_bias_grad_node", "min")
            
            # Prepare for next iteration - FIXED: Proper variable assignment order
            pre_geom = copy.copy(geometry_num_list)
            geometry_list = self.print_geometry_list(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_input_files(geometry_list, optimize_num + 1)
            pre_total_force = copy.copy(total_force)
            pre_biased_gradient_list = copy.copy(biased_gradient_list)
            pre_total_velocity = copy.copy(total_velocity)
            pre_biased_energy_list = copy.copy(biased_energy_list)
            
            # Save energy data
            with open(self.config.NEB_FOLDER_DIRECTORY + "energy_plot.csv", "a") as f:
                f.write(",".join(list(map(str, biased_energy_list.tolist()))) + "\n")
            
        self.make_traj_file(file_directory) 
        print("Complete...")
        return
    
    def _setup_force_calculation(self):
        """Setup force calculation method"""
        if self.config.om:
            return CaluculationOM(self.config.APPLY_CI_NEB)
        elif self.config.lup:
            return CaluculationLUP(self.config.APPLY_CI_NEB)
        elif self.config.dneb:
            return CaluculationDNEB(self.config.APPLY_CI_NEB)
        elif self.config.nesb:
            return CaluculationNESB(self.config.APPLY_CI_NEB)
        elif self.config.bneb:
            return CaluculationBNEB2(self.config.APPLY_CI_NEB)
        elif self.config.ewbneb:
            return CaluculationEWBNEB(self.config.APPLY_CI_NEB)
        else:
            return CaluculationBNEB(self.config.APPLY_CI_NEB)
    
    def _setup_optimizer(self):
        """Setup optimization algorithm"""
        # Determine optimization method based on configuration
        if self.config.FC_COUNT != -1 or (self.config.MFC_COUNT != -1 and self.config.model_hessian):
            return OptimizationFactory.create_optimizer("rfo", self.config)
        elif self.config.lbfgs_method:
            return OptimizationFactory.create_optimizer("lbfgs", self.config)
        elif self.config.cg_method:
            return OptimizationFactory.create_optimizer("conjugate_gradient", self.config)
        else:
            return OptimizationFactory.create_optimizer("fire", self.config)
    
    def _apply_bias_potential(self, energy_list, gradient_list, geometry_num_list, element_list, force_data, optimize_num):
        """Apply bias potential to energies and gradients - FIXED: Check hessian file existence"""
        biased_energy_list = []
        biased_gradient_list = []
        
        for i in range(len(energy_list)):
            _, B_e, B_g, B_hess = BiasPotentialCalculation(
                self.config.NEB_FOLDER_DIRECTORY).main(
                energy_list[i], gradient_list[i], geometry_num_list[i], 
                element_list, force_data)
                
            # FIXED: Only load hessian files if they exist and are needed
            if self.config.FC_COUNT > 0 or (self.config.MFC_COUNT > 0 and self.config.model_hessian):
                hessian_file = self.config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(i) + ".npy"
                if os.path.exists(hessian_file):
                    hess = np.load(hessian_file)
                    np.save(hessian_file, B_hess + hess)
                else:
                    # If hessian file doesn't exist yet, just save the bias hessian
                    # This can happen on the first iteration when exact hessians haven't been calculated yet
                    if not np.allclose(B_hess, 0):  # Only save if bias hessian is non-zero
                        np.save(hessian_file, B_hess)
                
            biased_energy_list.append(B_e)
            biased_gradient_list.append(B_g)
            
        return np.array(biased_energy_list, dtype="float64"), np.array(biased_gradient_list, dtype="float64")
    
    def _calculate_analysis_metrics(self, total_force, biased_gradient_list):
        """Calculate analysis metrics for monitoring convergence"""
        cos_list = []
        tot_force_rms_list = []
        tot_force_max_list = []
        bias_force_rms_list = []
        
        for i in range(len(total_force)):
            # Calculate cosine between total force and biased gradient
            total_force_norm = np.linalg.norm(total_force[i])
            biased_grad_norm = np.linalg.norm(biased_gradient_list[i])
            
            if total_force_norm > 1e-10 and biased_grad_norm > 1e-10:
                cos = np.sum(total_force[i] * biased_gradient_list[i]) / (total_force_norm * biased_grad_norm)
            else:
                cos = 0.0
            cos_list.append(cos)
            
            tot_force_rms = np.sqrt(np.mean(total_force[i]**2))
            tot_force_rms_list.append(tot_force_rms)
            
            tot_force_max = np.max(np.abs(total_force[i]))
            tot_force_max_list.append(tot_force_max)
            
            bias_force_rms = np.sqrt(np.mean(biased_gradient_list[i]**2))
            bias_force_rms_list.append(bias_force_rms)
            
        return cos_list, tot_force_rms_list, tot_force_max_list, bias_force_rms_list
    
    def _save_analysis_data(self, cos_list, tot_force_rms_list, tot_force_max_list, 
                           bias_force_rms_list, file_directory, optimize_num):
        """Save analysis data and create plots"""
        # Save bias force RMS data
        with open(self.config.NEB_FOLDER_DIRECTORY + "bias_force_rms.csv", "a") as f:
            f.write(",".join(list(map(str, bias_force_rms_list))) + "\n")
        
        # Create orthogonality plot
        if self.config.save_pict:
            self.visualizer.plot_orthogonality([x for x in range(len(cos_list))], cos_list, optimize_num)
        
        # Save orthogonality data
        with open(self.config.NEB_FOLDER_DIRECTORY + "orthogonality.csv", "a") as f:
            f.write(",".join(list(map(str, cos_list))) + "\n")
        
        # Create perpendicular gradient RMS plot
        if self.config.save_pict:
            self.visualizer.plot_perpendicular_gradient(
                [x for x in range(len(tot_force_rms_list))][1:-1], 
                tot_force_rms_list[1:-1], optimize_num, "rms")
        
        # Save perpendicular gradient RMS data
        with open(self.config.NEB_FOLDER_DIRECTORY + "perp_rms_gradient.csv", "a") as f:
            f.write(",".join(list(map(str, tot_force_rms_list))) + "\n")
        
        # Create perpendicular gradient MAX plot
        if self.config.save_pict:
            self.visualizer.plot_perpendicular_gradient(
                [x for x in range(len(tot_force_max_list))], 
                tot_force_max_list, optimize_num, "max")
        
        # Save perpendicular gradient MAX data
        with open(self.config.NEB_FOLDER_DIRECTORY + "perp_max_gradient.csv", "a") as f:
            f.write(",".join(list(map(str, tot_force_max_list))) + "\n")
    
    def _perform_optimization_step(self, optimizer, geometry_num_list, total_force, 
                                  biased_gradient_list, pre_geom, pre_biased_gradient_list, 
                                  optimize_num, biased_energy_list, pre_biased_energy_list, 
                                  pre_total_velocity, total_velocity, cos_list, 
                                  pre_geom_param, STRING_FORCE_CALC):
        """Perform optimization step based on the selected method"""
        if isinstance(optimizer, RFOOptimizer):
            # RFO optimization
            return optimizer.optimize(
                geometry_num_list, biased_gradient_list, pre_geom, pre_biased_gradient_list, 
                optimize_num, biased_energy_list, pre_biased_energy_list, 
                pre_total_velocity, total_velocity, cos_list, pre_geom_param, STRING_FORCE_CALC)
        elif isinstance(optimizer, FIREOptimizer):
            # FIRE optimization
            if optimize_num < self.config.sd:
                return optimizer.optimize(
                    geometry_num_list, total_force, pre_total_velocity, optimize_num, 
                    total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom_param)
            else:
                # Switch to steepest descent
                sd_optimizer = SteepestDescentOptimizer(self.config)
                return sd_optimizer.optimize(geometry_num_list, total_force)
        elif isinstance(optimizer, SteepestDescentOptimizer):
            return optimizer.optimize(geometry_num_list, total_force)
        elif hasattr(optimizer, 'LBFGS_NEB_calc'):
            # LBFGS optimization
            return optimizer.LBFGS_NEB_calc(
                geometry_num_list, total_force, pre_total_velocity, optimize_num, 
                total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom_param)
        elif hasattr(optimizer, 'CG_NEB_calc'):
            # Conjugate gradient optimization
            return optimizer.CG_NEB_calc(
                geometry_num_list, total_force, pre_total_velocity, optimize_num, 
                total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom_param)
        else:
            # Default to FIRE
            fire_optimizer = FIREOptimizer(self.config)
            return fire_optimizer.optimize(
                geometry_num_list, total_force, pre_total_velocity, optimize_num, 
                total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom_param)
    
    def _apply_constraints(self, new_geometry, fix_atom_flag, force_data, 
                          init_geometry_num_list, projection_constraint_flag, PC_list):
        """Apply various constraints to the new geometry"""
        # Apply fixed atoms constraint
        if fix_atom_flag:
            for k in range(len(new_geometry)):
                for j in force_data["fix_atoms"]:
                    new_geometry[k][j-1] = copy.copy(init_geometry_num_list[k][j-1] * self.config.bohr2angstroms)
        
        # Apply projection constraints
        if projection_constraint_flag:
            for x in range(len(new_geometry)):
                tmp_new_geometry = new_geometry[x] / self.config.bohr2angstroms
                tmp_new_geometry = PC_list[x].adjust_init_coord(tmp_new_geometry) * self.config.bohr2angstroms    
                new_geometry[x] = copy.copy(tmp_new_geometry)
        
        # Apply Kabsch alignment if no fixed atoms
        if not fix_atom_flag:
            for k in range(len(new_geometry)-1):
                tmp_new_geometry, _ = Calculationtools().kabsch_algorithm(new_geometry[k], new_geometry[k+1])
                new_geometry[k] = copy.copy(tmp_new_geometry)
        
        return new_geometry
    
    def make_geometry_list(self, init_input, partition_function):
        """Create geometry list from input files"""
        if os.path.splitext(init_input)[1] == ".xyz":
            self.config.init_input = os.path.splitext(init_input)[0]
            xyz_flag = True
        else:
            xyz_flag = False 
            
        start_file_list = sum([sorted(glob.glob(os.path.join(init_input, f"*_" + "[0-9]" * i + ".xyz"))) 
                              for i in range(1, 7)], [])

        loaded_geometry_list = []

        if xyz_flag:
            geometry_list, elements, electric_charge_and_multiplicity = traj2list(
                init_input, [self.config.electronic_charge, self.config.spin_multiplicity])
            
            element_list = elements[0]
            
            for i in range(len(geometry_list)):
                loaded_geometry_list.append([electric_charge_and_multiplicity] + 
                    [[element_list[num]] + list(map(str, geometry)) 
                     for num, geometry in enumerate(geometry_list[i])])
        else:
            for start_file in start_file_list:
                tmp_geometry_list, element_list, electric_charge_and_multiplicity = xyz2list(
                    start_file, [self.config.electronic_charge, self.config.spin_multiplicity])
                tmp_data = [electric_charge_and_multiplicity]

                for i in range(len(tmp_geometry_list)):
                    tmp_data.append([element_list[i]] + list(map(str, tmp_geometry_list[i])))
                loaded_geometry_list.append(tmp_data)
        
        electric_charge_and_multiplicity = loaded_geometry_list[0][0]
        element_list = [row[0] for row in loaded_geometry_list[0][1:]]
        
        loaded_geometry_num_list = [[list(map(float, row[1:4])) for row in geometry[1:]] 
                                   for geometry in loaded_geometry_list]

        geometry_list = [loaded_geometry_list[0]] 

        tmp_data = []
        
        for k in range(len(loaded_geometry_list) - 1):
            delta_num_geom = (np.array(loaded_geometry_num_list[k + 1], dtype="float64") - 
                            np.array(loaded_geometry_num_list[k], dtype="float64")) / (partition_function + 1)
            
            for i in range(partition_function + 1):
                frame_geom = np.array(loaded_geometry_num_list[k], dtype="float64") + delta_num_geom * i
                tmp_data.append(frame_geom)
                
        tmp_data = np.array(tmp_data, dtype="float64")
        
        # Apply IDPP if requested
        if self.config.IDPP_flag:
            IDPP_obj = IDPP()
            tmp_data = IDPP_obj.opt_path(tmp_data, element_list)
        
        # Align distances if requested
        if self.config.align_distances > 0:
            tmp_data = distribute_geometry(tmp_data)
        
        # Apply node distance constraint if specified
        if self.config.node_distance is not None:
            tmp_data = distribute_geometry_by_length(tmp_data, self.config.node_distance)
        
        for data in tmp_data:
            geometry_list.append([electric_charge_and_multiplicity] + 
                [[element_list[num]] + list(map(str, geometry)) 
                 for num, geometry in enumerate(data)])        
        
        print("\n Geometries are loaded. \n")
        return geometry_list, element_list, electric_charge_and_multiplicity

    def print_geometry_list(self, new_geometry, element_list, electric_charge_and_multiplicity):
        """Convert geometry array back to list format"""
        new_geometry = new_geometry.tolist()
        geometry_list = []
        for geometries in new_geometry:
            new_data = [electric_charge_and_multiplicity]
            for num, geometry in enumerate(geometries):
                geometry_data = list(map(str, geometry))
                geometry_data = [element_list[num]] + geometry_data
                new_data.append(geometry_data)
            
            geometry_list.append(new_data)
        return geometry_list

    def make_input_files(self, geometry_list, optimize_num):
        """Create input files for calculations"""
        file_directory = self.config.NEB_FOLDER_DIRECTORY + "path_ITR_" + str(optimize_num) + "_" + str(self.config.init_input)
        try:
            os.mkdir(file_directory)
        except:
            pass
        tmp_cs = [self.config.electronic_charge, self.config.spin_multiplicity]
        float_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"
        
        for y, geometry in enumerate(geometry_list):
            tmp_geometry = []
            for geom in geometry:
                if len(geom) == 4 and re.match(r"[A-Za-z]+", str(geom[0])) \
                  and all(re.match(float_pattern, str(x)) for x in geom[1:]):
                        tmp_geometry.append(geom)

                if len(geom) == 2 and re.match(r"-*\d+", str(geom[0])) and re.match(r"-*\d+", str(geom[1])):
                    tmp_cs = geom   
                    
            with open(file_directory + "/" + self.config.init_input + "_" + str(y) + ".xyz", "w") as w:
                w.write(str(len(tmp_geometry)) + "\n")
                w.write(str(tmp_cs[0]) + " " + str(tmp_cs[1]) + "\n")
                for rows in tmp_geometry:
                    w.write(f"{rows[0]:2}   {float(rows[1]):>17.12f}   {float(rows[2]):>17.12f}   {float(rows[3]):>17.12f}\n")
        return file_directory
        
    def make_traj_file(self, file_directory):
        """Create trajectory file from current geometries"""
        print("\nprocessing geometry collecting ...\n")
        file_list = sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) 
                        for i in range(1, 7)], [])
        
        for m, file in enumerate(file_list):
            tmp_geometry_list, element_list, _ = xyz2list(file, None)
            atom_num = len(tmp_geometry_list)
            with open(file_directory + "/" + self.config.init_input + "_path.xyz", "a") as w:
                w.write(str(atom_num) + "\n")
                w.write("Frame " + str(m) + "\n")
                for i in range(len(tmp_geometry_list)):
                    w.write(f"{element_list[i]:2}   {float(tmp_geometry_list[i][0]):17.12f}  {float(tmp_geometry_list[i][1]):17.12f}  {float(tmp_geometry_list[i][2]):17.12f}\n")
        print("\ncollecting geometries was complete...\n")
        return

    def force2velocity(self, gradient_list, element_list):
        """Convert force to velocity"""
        velocity_list = gradient_list
        return np.array(velocity_list, dtype="float64")


def calc_path_length_list(geometry_list):
    """Calculate path length list for geometry distribution"""
    path_length_list = [0.0]
    for i in range(len(geometry_list)-1):
        tmp_geometry_list_j = geometry_list[i+1] - np.mean(geometry_list[i+1], axis=0)
        tmp_geometry_list_i = geometry_list[i] - np.mean(geometry_list[i], axis=0)
        
        path_length = path_length_list[-1] + np.linalg.norm(tmp_geometry_list_j - tmp_geometry_list_i)
        path_length_list.append(path_length)
    return path_length_list


def apply_climbing_image(geometry_list, energy_list):
    """Apply climbing image method to locate transition states"""
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    local_maxima, local_minima = spline_interpolation(path_length_list, energy_list)
    print(local_maxima)
    
    for distance, energy in local_maxima:
        print("Local maximum at distance: ", distance)
        for i in range(2, len(path_length_list)-2):
            if path_length_list[i] >= distance or distance >= path_length_list[i+1]:
                continue
            delta_t = (distance - path_length_list[i]) / (path_length_list[i+1] - path_length_list[i])
            tmp_geometry = geometry_list[i] + (geometry_list[i+1] - geometry_list[i]) * delta_t
            tmp_geom_list = [geometry_list[i], tmp_geometry, geometry_list[i+1]]
            idpp_instance = IDPP()
            tmp_geom_list = idpp_instance.opt_path(tmp_geom_list, element_list)
            geometry_list[i] = tmp_geom_list[1]
    return geometry_list


def distribute_geometry(geometry_list):
    """Distribute geometries evenly along the path"""
    nnode = len(geometry_list)
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    node_dist = total_length / (nnode-1)
    
    new_geometry_list = [geometry_list[0]]
    for i in range(1, nnode-1):
        dist = i * node_dist
        for j in range(nnode-1):
            if path_length_list[j] <= dist and dist <= path_length_list[j+1]:
                break
        delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
        new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
        new_geometry_list.append(new_geometry)
    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list


def distribute_geometry_by_length(geometry_list, angstrom_spacing):
    """Distribute geometries by specified distance spacing"""
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    new_geometry_list = [geometry_list[0]]
    
    max_steps = int(total_length // angstrom_spacing)
    for i in range(1, max_steps):
        dist = i * angstrom_spacing
       
        for j in range(len(path_length_list) - 1):
            if path_length_list[j] <= dist <= path_length_list[j+1]:
                break
      
        delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
        new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
        new_geometry_list.append(new_geometry)

    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list