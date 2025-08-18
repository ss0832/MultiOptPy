import os
import sys
import datetime
import glob
import numpy as np
from pathlib import Path
import logging

from potential import BiasPotentialCalculation
from optimizer import CalculateMoveVector 
from calc_tools import Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib
from interface import force_data_parser
from polar_coordinate import cart2polar, polar2cart, cart_grad_2_polar_grad
import ModelFunction as MF

class IEIPConfig:
    """
    Configuration class for Improved Elastic Image Pair (iEIP) method.
    
    References:
    - J. Chem. Theory. Comput. 2023, 19, 2410-2417
    - J. Comput. Chem. 2018, 39, 233–251 (DS-AFIR)
    """
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        
        # Unit conversion constants
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol
        
        # Displacement and convergence thresholds
        self.displacement_limit = 0.04  # Bohr
        self.maximum_ieip_disp = 0.2    # Bohr
        self.L_covergence = 0.03        # Bohr
        self.img_distance_convage_criterion = 0.15  # Bohr
        
        # Convergence thresholds
        self.MAX_FORCE_THRESHOLD = 0.0006
        self.RMS_FORCE_THRESHOLD = 0.0004
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0030
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0020
        
        # Iteration limits
        self.microiterlimit = int(args.NSTEP)
        self.microiter_num = args.microiter
        
        # Electronic state configuration
        self.initial_excite_state = args.excited_state[0]
        self.final_excite_state = args.excited_state[1]
        self.excite_state_list = args.excited_state
        
        # Charge and multiplicity
        self.init_electric_charge_and_multiplicity = [int(args.electronic_charge[0]), int(args.spin_multiplicity[0])]
        self.final_electric_charge_and_multiplicity = [int(args.electronic_charge[1]), int(args.spin_multiplicity[1])]
        self.electric_charge_and_multiplicity_list = []
        for i in range(len(args.electronic_charge)):
            self.electric_charge_and_multiplicity_list.append([int(args.electronic_charge[i]), int(args.spin_multiplicity[i])])
        
        # Solvation models
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        
        # Computation resources
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.START_FILE = args.INPUT + "/"
        
        # Quantum chemistry settings
        self.BASIS_SET = args.basisset
        self.FUNCTIONAL = args.functional
        self.usextb = args.usextb
        self.usedxtb = args.usedxtb
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        
        # Validate sub-basis set inputs
        if len(args.sub_basisset) % 2 != 0:
            print("invalid input (-sub_bs)")
            sys.exit(0)
        
        # Configure basis sets
        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET)
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
            else:
                self.SUB_BASIS_SET = {"default": self.BASIS_SET}
        else:  # psi4
            self.SUB_BASIS_SET = ""
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET += "\nassign " + str(self.BASIS_SET) + "\n"
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign " + args.sub_basisset[2*j] + " " + args.sub_basisset[2*j+1] + "\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
        
        # Validate effective core potential inputs
        if len(args.effective_core_potential) % 2 != 0:
            print("invalid input (-ecp)")
            sys.exit(0)
        
        # Configure effective core potentials
        if args.pyscf:
            self.ECP = {}
            if len(args.effective_core_potential) > 0:
                for j in range(int(len(args.effective_core_potential)/2)):
                    self.ECP[args.effective_core_potential[2*j]] = args.effective_core_potential[2*j+1]
        else:
            self.ECP = ""
        
        # Other settings
        self.othersoft = args.othersoft
        self.basic_set_and_function = args.functional + "/" + args.basisset
        self.force_data = force_data_parser(args)
        
        # Set up output directory
        if self.othersoft != "None":
            self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + args.othersoft + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
        elif args.usextb == "None" and args.usedxtb == "None":
            self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + self.basic_set_and_function.replace("/", "_") + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
        else:
            if args.usedxtb != "None":
                self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + self.usedxtb + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
                self.force_data["xtb"] = self.usedxtb
            else:
                self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + self.usextb + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
                self.force_data["xtb"] = self.usextb
        
        # Optimization parameters
        self.args = args
        self.BETA = args.BETA
        self.spring_const = 1e-8
        self.unrestrict = args.unrestrict
        self.mf_mode = args.model_function_mode
        self.FC_COUNT = int(args.calc_exact_hess)
        self.dft_grid = int(args.dft_grid)
        
        # Set config for Newton Trajectory Method
        self.use_gnt = getattr(args, 'use_gnt', False)
        self.gnt_step_len = getattr(args, 'gnt_step_len', 0.5)
        self.gnt_rms_thresh = getattr(args, 'gnt_rms_thresh', 1.7e-3)
        self.gnt_vec = getattr(args, 'gnt_vec', None)
        self.gnt_microiter = getattr(args, 'gnt_microiter', 100)
        # Create output directory
        os.mkdir(self.iEIP_FOLDER_DIRECTORY)
    
    def save_input_data(self):
        """Save the input configuration data to a file"""
        with open(self.iEIP_FOLDER_DIRECTORY + "input.txt", "w") as f:
            f.write(str(vars(self.args)))
        return

class ElasticImagePair:
    """
    Implementation of the Improved Elastic Image Pair (iEIP) method
    for finding minimum energy paths and transition states.
    """
    def __init__(self, config):
        self.config = config
        
    def RMS(self, mat):
        """Calculate root mean square of a matrix"""
        rms = np.sqrt(np.sum(mat**2))
        return rms
    
    def print_info(self, dat):
        """Print optimization information"""
        print("[[opt information]]")
        print("                                                image_1               image_2")
        print("energy  (normal)                       : "+str(dat["energy_1"])+"   "+str(dat["energy_2"]))
        print("energy  (bias)                         : "+str(dat["bias_energy_1"])+"   "+str(dat["bias_energy_2"]))
        print("gradient  (normal, RMS)                : "+str(self.RMS(dat["gradient_1"]))+"   "+str(self.RMS(dat["gradient_2"])))
        print("gradient  (bias, RMS)                  : "+str(self.RMS(dat["bias_gradient_1"]))+"   "+str(self.RMS(dat["bias_gradient_2"])))
        print("perpendicular_force (RMS)              : "+str(self.RMS(dat["perp_force_1"]))+"   "+str(self.RMS(dat["perp_force_2"])))
        print("energy_difference_dependent_force (RMS): "+str(self.RMS(dat["delta_energy_force_1"]))+"   "+str(self.RMS(dat["delta_energy_force_2"])))
        print("distance_dependent_force (RMS)         : "+str(self.RMS(dat["close_target_force"])))
        print("Total_displacement (RMS)               : "+str(self.RMS(dat["total_disp_1"]))+"   "+str(self.RMS(dat["total_disp_2"])))
        print("Image_distance                         : "+str(dat["delta_geometry"]))
        
        print("[[threshold]]")
        print("Image_distance                         : ", self.config.img_distance_convage_criterion)
       
        return

    def lbfgs_update(self, s_list, y_list, grad, m=10):
        """
        L-BFGS algorithm to compute the search direction.
        
        Parameters:
        -----------
        s_list : list of arrays
            List of position differences (x_{k+1} - x_k)
        y_list : list of arrays
            List of gradient differences (g_{k+1} - g_k)
        grad : array
            Current gradient
        m : int
            Number of corrections to store
        
        Returns:
        --------
        array
            Search direction
        """
        k = len(s_list)
        if k == 0:
            return -grad
        
        q = grad.copy()
        alphas = np.zeros(k)
        rhos = np.zeros(k)
        
        # Compute rho values
        for i in range(k):
            rhos[i] = 1.0 / (np.dot(y_list[i], s_list[i]) + 1e-10)
        
        # First loop (backward)
        for i in range(k-1, -1, -1):
            alphas[i] = rhos[i] * np.dot(s_list[i], q)
            q = q - alphas[i] * y_list[i]
        
        # Scaling factor
        gamma = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-10)
        r = gamma * q
        
        # Second loop (forward)
        for i in range(k):
            beta = rhos[i] * np.dot(y_list[i], r)
            r = r + s_list[i] * (alphas[i] - beta)
        
        return -r

    def microiteration(self, SP1, SP2, FIO1, FIO2, file_directory_1, file_directory_2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter):
        """
        Perform microiterations to optimize geometries with adaptive trust radius
        and polar coordinate representation.
        """
        # Initialize L-BFGS storage
        s_list_1, y_list_1 = [], []
        s_list_2, y_list_2 = [], []
        prev_grad_1, prev_grad_2 = None, None
        prev_pos_1, prev_pos_2 = None, None
        prev_energy_1, prev_energy_2 = None, None
        max_lbfgs_memory = 10  # Store up to 10 previous steps
        
        # Initialize trust region parameters with polar coordinate adaptations
        # Base trust radius settings
        trust_radius_1 = 0.015  # Initial trust radius (Bohr)
        trust_radius_2 = 0.015
        # Component weights for polar coordinates
        radial_weight = 1.0     # Weight for radial components
        angular_weight = 0.85    # Weight for angular components (smaller due to different scale)
        # Trust radius limits
        min_trust_radius = 0.01  # Minimum trust radius
        max_trust_radius = 0.1   # Maximum trust radius
        # Adjustment factors
        good_step_factor = 1.25    # Increase factor for good steps
        bad_step_factor = 0.5     # Decrease factor for bad steps
        # Performance thresholds
        excellent_ratio_threshold = 0.85  # Threshold for exceptionally good steps
        very_bad_ratio_threshold = 0.2    # Threshold for very poor steps
    
        for i in range(self.config.microiter_num):
            
            energy_1, gradient_1, geom_num_list_1, error_flag_1 = SP1.single_point(file_directory_1, element_list, iter, init_electric_charge_and_multiplicity, self.config.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, error_flag_2 = SP2.single_point(file_directory_2, element_list, iter, final_electric_charge_and_multiplicity, self.config.force_data["xtb"])
            
            BPC_1 = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
            BPC_2 = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.config.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.config.force_data)
        
            if error_flag_1 or error_flag_2:
                print("Error in QM calculation.")
                with open(self.config.iEIP_FOLDER_DIRECTORY+"end.txt", "w") as f:
                    f.write("Error in QM calculation.")
                break

            microiter_force_1 = -cart_grad_2_polar_grad(geom_num_list_1.reshape(-1), bias_gradient_1.reshape(-1))
            microiter_force_2 = -cart_grad_2_polar_grad(geom_num_list_2.reshape(-1), bias_gradient_2.reshape(-1))
            microiter_force_1[0] = 0.0
            microiter_force_2[0] = 0.0
            
            # Update L-BFGS memory
            if prev_grad_1 is not None and prev_pos_1 is not None:
                s_1 = geom_num_list_1.reshape(-1) - prev_pos_1
                y_1 = microiter_force_1 - prev_grad_1
                
                # Only add if step and gradient difference are significant and curvature is positive
                if np.linalg.norm(s_1) > 1e-10 and np.linalg.norm(y_1) > 1e-10 and np.dot(s_1, y_1) > 1e-10:
                    s_list_1.append(s_1)
                    y_list_1.append(y_1)
                    if len(s_list_1) > max_lbfgs_memory:
                        s_list_1.pop(0)
                        y_list_1.pop(0)
            
            if prev_grad_2 is not None and prev_pos_2 is not None:
                s_2 = geom_num_list_2.reshape(-1) - prev_pos_2
                y_2 = microiter_force_2 - prev_grad_2
                
                if np.linalg.norm(s_2) > 1e-10 and np.linalg.norm(y_2) > 1e-10 and np.dot(s_2, y_2) > 1e-10:
                    s_list_2.append(s_2)
                    y_list_2.append(y_2)
                    if len(s_list_2) > max_lbfgs_memory:
                        s_list_2.pop(0)
                        y_list_2.pop(0)
            
            # Store current values for next iteration
            prev_grad_1 = microiter_force_1.copy()
            prev_grad_2 = microiter_force_2.copy()
            prev_pos_1 = geom_num_list_1.reshape(-1).copy()
            prev_pos_2 = geom_num_list_2.reshape(-1).copy()
            
            if i % 10 == 0:
                print("# Microiteration "+str(i))
                print("Energy 1                 :", energy_1)
                print("Energy 2                 :", energy_2)
                print("Energy 1  (bias)         :", bias_energy_1)
                print("Energy 2  (bias)         :", bias_energy_2)
                print("RMS perpendicular force 1:", self.RMS(microiter_force_1[1:]))
                print("RMS perpendicular force 2:", self.RMS(microiter_force_2[1:]))
                print("Trust radius 1           :", trust_radius_1)
                print("Trust radius 2           :", trust_radius_2)
            
            polar_coord_1 = cart2polar(geom_num_list_1.reshape(-1), prev_geom_num_list_1.reshape(-1))
            polar_coord_2 = cart2polar(geom_num_list_2.reshape(-1), prev_geom_num_list_2.reshape(-1))
            
            # Apply L-BFGS to get initial step directions
            if len(s_list_1) > 0:
                total_disp_1 = self.lbfgs_update(s_list_1, y_list_1, -microiter_force_1)
            else:
                total_disp_1 = microiter_force_1.reshape(-1)
            total_disp_1[0] = 0.0  # Keep the first component fixed
            if len(s_list_2) > 0:
                total_disp_2 = self.lbfgs_update(s_list_2, y_list_2, -microiter_force_2)
            else:
                total_disp_2 = microiter_force_2.reshape(-1)
            total_disp_2[0] = 0.0  # Keep the first component fixed
            # Create component-wise weight masks for polar coordinates
            # Polar coordinate structure: [r, theta, phi, r, theta, phi, ...]
            weights_1 = np.ones_like(total_disp_1)
            weights_2 = np.ones_like(total_disp_2)
            
            # Set different weights for radial and angular components
            for j in range(0, len(weights_1), 3):
                weights_1[j] = radial_weight         # r component
                weights_1[j+1:j+3] = angular_weight  # theta, phi components
                
            for j in range(0, len(weights_2), 3):
                weights_2[j] = radial_weight         # r component
                weights_2[j+1:j+3] = angular_weight  # theta, phi components
            
            # Calculate weighted displacement norms for trust radius scaling
            weighted_disp_1 = total_disp_1 * weights_1
            weighted_disp_2 = total_disp_2 * weights_2
            
            weighted_norm_1 = np.linalg.norm(weighted_disp_1)
            weighted_norm_2 = np.linalg.norm(weighted_disp_2)
            
            # Apply weighted trust region constraints
            if weighted_norm_1 > trust_radius_1:
                scale_factor_1 = trust_radius_1 / weighted_norm_1
                total_disp_1 = total_disp_1 * scale_factor_1
                
            if weighted_norm_2 > trust_radius_2:
                scale_factor_2 = trust_radius_2 / weighted_norm_2
                total_disp_2 = total_disp_2 * scale_factor_2
            
            # Handle angular periodicities in polar coordinates
            for j in range(1, len(total_disp_1), 3):  # theta components
                # Constrain theta (0 to π)
                if polar_coord_1[j] + total_disp_1[j] > np.pi:
                    total_disp_1[j] = np.pi - polar_coord_1[j] - 1e-6
                elif polar_coord_1[j] + total_disp_1[j] < 0:
                    total_disp_1[j] = -polar_coord_1[j] + 1e-6
                
                # Constrain phi (0 to 2π) - ensure shortest path
                if j+1 < len(total_disp_1):
                    curr_phi = polar_coord_1[j+1]
                    target_phi = curr_phi + total_disp_1[j+1]
                    
                    # Normalize phi to [0, 2π]
                    while target_phi > 2*np.pi:
                        target_phi -= 2*np.pi
                    while target_phi < 0:
                        target_phi += 2*np.pi
                        
                    # Find shortest angular path
                    phi_diff = target_phi - curr_phi
                    if abs(phi_diff) > np.pi:
                        if phi_diff > 0:
                            phi_diff -= 2*np.pi
                        else:
                            phi_diff += 2*np.pi
                    total_disp_1[j+1] = phi_diff
            
            # Apply same periodic boundary handling for image 2
            for j in range(1, len(total_disp_2), 3):
                if polar_coord_2[j] + total_disp_2[j] > np.pi:
                    total_disp_2[j] = np.pi - polar_coord_2[j] - 1e-6
                elif polar_coord_2[j] + total_disp_2[j] < 0:
                    total_disp_2[j] = -polar_coord_2[j] + 1e-6
                
                if j+1 < len(total_disp_2):
                    curr_phi = polar_coord_2[j+1]
                    target_phi = curr_phi + total_disp_2[j+1]
                    
                    while target_phi > 2*np.pi:
                        target_phi -= 2*np.pi
                    while target_phi < 0:
                        target_phi += 2*np.pi
                        
                    phi_diff = target_phi - curr_phi
                    if abs(phi_diff) > np.pi:
                        if phi_diff > 0:
                            phi_diff -= 2*np.pi
                        else:
                            phi_diff += 2*np.pi
                    total_disp_2[j+1] = phi_diff
            
            # Calculate predicted energy reduction
            pred_reduction_1 = np.dot(microiter_force_1, total_disp_1)
            pred_reduction_2 = np.dot(microiter_force_2, total_disp_2)
            
            # Apply step
            new_polar_coord_1 = polar_coord_1 + total_disp_1
            new_polar_coord_2 = polar_coord_2 + total_disp_2
            
            tmp_geom_1 = polar2cart(new_polar_coord_1, prev_geom_num_list_1.reshape(-1))
            tmp_geom_2 = polar2cart(new_polar_coord_2, prev_geom_num_list_2.reshape(-1))
            
            geom_num_list_1 = tmp_geom_1.reshape(len(geom_num_list_1), 3)
            geom_num_list_2 = tmp_geom_2.reshape(len(geom_num_list_2), 3)
            
            # Create input files with new geometries
            new_geom_num_list_1_tolist = (geom_num_list_1*self.config.bohr2angstroms).tolist()
            new_geom_num_list_2_tolist = (geom_num_list_2*self.config.bohr2angstroms).tolist()
            for j, elem in enumerate(element_list):
                new_geom_num_list_1_tolist[j].insert(0, elem)
                new_geom_num_list_2_tolist[j].insert(0, elem)
            
            new_geom_num_list_1_tolist.insert(0, init_electric_charge_and_multiplicity)
            new_geom_num_list_2_tolist.insert(0, final_electric_charge_and_multiplicity)
                
            file_directory_1 = FIO1.make_psi4_input_file([new_geom_num_list_1_tolist], iter)
            file_directory_2 = FIO2.make_psi4_input_file([new_geom_num_list_2_tolist], iter)
            
            # Update trust radius based on actual vs. predicted reduction
            if prev_energy_1 is not None and prev_energy_2 is not None:
                actual_reduction_1 = prev_energy_1 - bias_energy_1
                actual_reduction_2 = prev_energy_2 - bias_energy_2
                
                # Calculate performance ratio
                ratio_1 = actual_reduction_1 / (pred_reduction_1 + 1e-10)
                ratio_2 = actual_reduction_2 / (pred_reduction_2 + 1e-10)
                
                # Update trust radius based on performance metrics
                if ratio_1 < very_bad_ratio_threshold:  # Very poor step
                    trust_radius_1 = max(trust_radius_1 * bad_step_factor, min_trust_radius)
                elif ratio_1 > excellent_ratio_threshold and weighted_norm_1 >= 0.9 * trust_radius_1:  # Excellent step
                    trust_radius_1 = min(trust_radius_1 * good_step_factor, max_trust_radius)
                
                if ratio_2 < very_bad_ratio_threshold:  # Very poor step
                    trust_radius_2 = max(trust_radius_2 * bad_step_factor, min_trust_radius)
                elif ratio_2 > excellent_ratio_threshold and weighted_norm_2 >= 0.9 * trust_radius_2:  # Excellent step
                    trust_radius_2 = min(trust_radius_2 * good_step_factor, max_trust_radius)
                
                # Auto-adjust weights if large angular displacements are occurring
                max_angle_disp_1 = max([abs(total_disp_1[j]) for j in range(1, len(total_disp_1), 3) if j < len(total_disp_1)])
                max_angle_disp_2 = max([abs(total_disp_2[j]) for j in range(1, len(total_disp_2), 3) if j < len(total_disp_2)])
                
                if max_angle_disp_1 > 0.3:  # Threshold for excessive angular displacement
                    angular_weight = max(angular_weight * 0.9, 0.1)  # Decrease angular weight
                
                if max_angle_disp_2 > 0.3:
                    angular_weight = max(angular_weight * 0.9, 0.1)
            
            # Store current energies for next iteration
            prev_energy_1 = bias_energy_1
            prev_energy_2 = bias_energy_2
            
            # Convergence check
            if self.RMS(microiter_force_1) < 0.01 and self.RMS(microiter_force_2) < 0.01:
                print("Optimization converged.")
                break
        
        return energy_1, gradient_1, bias_energy_1, bias_gradient_1, geom_num_list_1, energy_2, gradient_2, bias_energy_2, bias_gradient_2, geom_num_list_2

    def iteration(self, file_directory_1, file_directory_2, SP1, SP2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, FIO1, FIO2):
        """
        Main elastic image pair optimization iteration.
        """
        G = Graph(self.config.iEIP_FOLDER_DIRECTORY)
        beta_m = 0.9
        beta_v = 0.999 
        BIAS_GRAD_LIST_A = []
        BIAS_GRAD_LIST_B = []
        BIAS_ENERGY_LIST_A = []
        BIAS_ENERGY_LIST_B = []
        
        GRAD_LIST_A = []
        GRAD_LIST_B = []
        ENERGY_LIST_A = []
        ENERGY_LIST_B = []
        prev_delta_geometry = 0.0
        
        for iter in range(0, self.config.microiterlimit):
            if os.path.isfile(self.config.iEIP_FOLDER_DIRECTORY+"end.txt"):
                break
            print("# ITR. "+str(iter))
            
            energy_1, gradient_1, geom_num_list_1, error_flag_1 = SP1.single_point(file_directory_1, element_list, iter, init_electric_charge_and_multiplicity, self.config.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, error_flag_2 = SP2.single_point(file_directory_2, element_list, iter, final_electric_charge_and_multiplicity, self.config.force_data["xtb"])
            geom_num_list_1, geom_num_list_2 = Calculationtools().kabsch_algorithm(geom_num_list_1, geom_num_list_2)
            
            if error_flag_1 or error_flag_2:
                print("Error in QM calculation.")
                with open(self.config.iEIP_FOLDER_DIRECTORY+"end.txt", "w") as f:
                    f.write("Error in QM calculation.")
                break
            
            if iter == 0:
                m_1 = gradient_1 * 0.0
                m_2 = gradient_1 * 0.0
                v_1 = gradient_1 * 0.0
                v_2 = gradient_1 * 0.0
                ini_geom_1 = geom_num_list_1
                ini_geom_2 = geom_num_list_2
            
            BPC_1 = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
            BPC_2 = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.config.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.config.force_data)
        
            if self.config.microiter_num > 0 and iter > 0:
                energy_1, gradient_1, bias_energy_1, bias_gradient_1, geom_num_list_1, energy_2, gradient_2, bias_energy_2, bias_gradient_2, geom_num_list_2 = self.microiteration(SP1, SP2, FIO1, FIO2, file_directory_1, file_directory_2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter)
                if os.path.isfile(self.config.iEIP_FOLDER_DIRECTORY+"end.txt"):
                    break
            
            # Determine which image is higher in energy for proper force direction
            if energy_2 > energy_1:
                N = self.norm_dist_2imgs(geom_num_list_1, geom_num_list_2)
                L = self.dist_2imgs(geom_num_list_1, geom_num_list_2)
            else:
                N = self.norm_dist_2imgs(geom_num_list_2, geom_num_list_1)
                L = self.dist_2imgs(geom_num_list_2, geom_num_list_1)   
            
            Lt = self.target_dist_2imgs(L)
            
            force_disp_1 = self.displacement(bias_gradient_1) 
            force_disp_2 = self.displacement(bias_gradient_2) 
            
            perp_force_1 = self.perpendicular_force(bias_gradient_1, N)
            perp_force_2 = self.perpendicular_force(bias_gradient_2, N)
            
            delta_energy_force_1 = self.delta_energy_force(bias_energy_1, bias_energy_2, N, L)
            delta_energy_force_2 = self.delta_energy_force(bias_energy_1, bias_energy_2, N, L)
            
            close_target_force = self.close_target_force(L, Lt, geom_num_list_1, geom_num_list_2)

            perp_disp_1 = self.displacement(perp_force_1)
            perp_disp_2 = self.displacement(perp_force_2)

            delta_energy_disp_1 = self.displacement(delta_energy_force_1) 
            delta_energy_disp_2 = self.displacement(delta_energy_force_2) 
            
            close_target_disp = self.displacement(close_target_force)
            
            if iter == 0:
                ini_force_1 = perp_force_1 * 0.0
                ini_force_2 = perp_force_2 * 0.0
                ini_disp_1 = ini_force_1
                ini_disp_2 = ini_force_2
                close_target_disp_1 = close_target_disp
                close_target_disp_2 = close_target_disp
            else:
                ini_force_1 = self.initial_structure_dependent_force(geom_num_list_1, ini_geom_1)
                ini_force_2 = self.initial_structure_dependent_force(geom_num_list_2, ini_geom_2)
                ini_disp_1 = self.displacement_prime(ini_force_1)
                ini_disp_2 = self.displacement_prime(ini_force_2)
                
                # Based on DS-AFIR method
                Z_1 = np.linalg.norm(geom_num_list_1 - ini_geom_1) / np.linalg.norm(geom_num_list_1 - geom_num_list_2) + (np.sum((geom_num_list_1 - ini_geom_1) * (geom_num_list_1 - geom_num_list_2))) / (np.linalg.norm(geom_num_list_1 - ini_geom_1) * np.linalg.norm(geom_num_list_1 - geom_num_list_2)) 
                Z_2 = np.linalg.norm(geom_num_list_2 - ini_geom_2) / np.linalg.norm(geom_num_list_2 - geom_num_list_1) + (np.sum((geom_num_list_2 - ini_geom_2) * (geom_num_list_2 - geom_num_list_1))) / (np.linalg.norm(geom_num_list_2 - ini_geom_2) * np.linalg.norm(geom_num_list_2 - geom_num_list_1))
                
                if Z_1 > 0.0:
                    Y_1 = Z_1 / (Z_1 + 1) + 0.5
                else:
                    Y_1 = 0.5
                
                if Z_2 > 0.0:
                    Y_2 = Z_2 / (Z_2 + 1) + 0.5
                else:
                    Y_2 = 0.5
                
                u_1 = Y_1 * ((geom_num_list_1 - geom_num_list_2) / np.linalg.norm(geom_num_list_1 - geom_num_list_2)) - (1.0 - Y_1) * ((geom_num_list_1 - ini_geom_1) / np.linalg.norm(geom_num_list_1 - ini_geom_1))  
                u_2 = Y_2 * ((geom_num_list_2 - geom_num_list_1) / np.linalg.norm(geom_num_list_2 - geom_num_list_1)) - (1.0 - Y_2) * ((geom_num_list_2 - ini_geom_2) / np.linalg.norm(geom_num_list_2 - ini_geom_2)) 
                
                X_1 = self.config.BETA / np.linalg.norm(u_1) - (np.sum(gradient_1 * u_1) / np.linalg.norm(u_1) ** 2)
                X_2 = self.config.BETA / np.linalg.norm(u_2) - (np.sum(gradient_2 * u_2) / np.linalg.norm(u_2) ** 2)
               
                ini_disp_1 *= X_1 * (1.0 - Y_1)
                ini_disp_2 *= X_2 * (1.0 - Y_2)
                
                close_target_disp_1 = close_target_disp * X_1 * Y_1
                close_target_disp_2 = close_target_disp * X_2 * Y_2
       
            total_disp_1 = -perp_disp_1 + delta_energy_disp_1 + close_target_disp_1 - force_disp_1 + ini_disp_1
            total_disp_2 = -perp_disp_2 - delta_energy_disp_2 - close_target_disp_2 - force_disp_2 + ini_disp_2
            
            # AdaBelief optimizer: https://doi.org/10.48550/arXiv.2010.07468
            m_1 = beta_m * m_1 + (1 - beta_m) * total_disp_1
            m_2 = beta_m * m_2 + (1 - beta_m) * total_disp_2
            v_1 = beta_v * v_1 + (1 - beta_v) * (total_disp_1 - m_1)**2
            v_2 = beta_v * v_2 + (1 - beta_v) * (total_disp_2 - m_2)**2
            
            adabelief_1 = 0.01 * (m_1 / (np.sqrt(v_1) + 1e-8))
            adabelief_2 = 0.01 * (m_2 / (np.sqrt(v_2) + 1e-8))
            
            new_geom_num_list_1 = geom_num_list_1 + adabelief_1
            new_geom_num_list_2 = geom_num_list_2 + adabelief_2
            
            new_geom_num_list_1, new_geom_num_list_2 = Calculationtools().kabsch_algorithm(new_geom_num_list_1, new_geom_num_list_2)
    
            if iter != 0:
                prev_delta_geometry = delta_geometry
            
            delta_geometry = np.linalg.norm(new_geom_num_list_2 - new_geom_num_list_1)
            rms_perp_force = np.linalg.norm(np.sqrt(perp_force_1 ** 2 + perp_force_2 ** 2))
            
            info_dat = {
                "perp_force_1": perp_force_1, 
                "perp_force_2": perp_force_2, 
                "delta_energy_force_1": delta_energy_force_1, 
                "delta_energy_force_2": delta_energy_force_2,
                "close_target_force": close_target_force, 
                "perp_disp_1": perp_disp_1,
                "perp_disp_2": perp_disp_2,
                "delta_energy_disp_1": delta_energy_disp_1,
                "delta_energy_disp_2": delta_energy_disp_2,
                "close_target_disp": close_target_disp, 
                "total_disp_1": total_disp_1, 
                "total_disp_2": total_disp_2, 
                "bias_energy_1": bias_energy_1,
                "bias_energy_2": bias_energy_2,
                "bias_gradient_1": bias_gradient_1,
                "bias_gradient_2": bias_gradient_2,
                "energy_1": energy_1,
                "energy_2": energy_2,
                "gradient_1": gradient_1,
                "gradient_2": gradient_2, 
                "delta_geometry": delta_geometry, 
                "rms_perp_force": rms_perp_force
            }
            
            self.print_info(info_dat)
            
            # Prepare geometries for next iteration
            new_geom_num_list_1_tolist = (new_geom_num_list_1*self.config.bohr2angstroms).tolist()
            new_geom_num_list_2_tolist = (new_geom_num_list_2*self.config.bohr2angstroms).tolist()
            for i, elem in enumerate(element_list):
                new_geom_num_list_1_tolist[i].insert(0, elem)
                new_geom_num_list_2_tolist[i].insert(0, elem)
            
            new_geom_num_list_1_tolist.insert(0, init_electric_charge_and_multiplicity)
            new_geom_num_list_2_tolist.insert(0, final_electric_charge_and_multiplicity)
                
            file_directory_1 = FIO1.make_psi4_input_file([new_geom_num_list_1_tolist], iter+1)
            file_directory_2 = FIO2.make_psi4_input_file([new_geom_num_list_2_tolist], iter+1)
            
            # Record data for plotting
            BIAS_ENERGY_LIST_A.append(bias_energy_1*self.config.hartree2kcalmol)
            BIAS_ENERGY_LIST_B.append(bias_energy_2*self.config.hartree2kcalmol)
            BIAS_GRAD_LIST_A.append(np.sqrt(np.sum(bias_gradient_1**2)))
            BIAS_GRAD_LIST_B.append(np.sqrt(np.sum(bias_gradient_2**2)))
            
            ENERGY_LIST_A.append(energy_1*self.config.hartree2kcalmol)
            ENERGY_LIST_B.append(energy_2*self.config.hartree2kcalmol)
            GRAD_LIST_A.append(np.sqrt(np.sum(gradient_1**2)))
            GRAD_LIST_B.append(np.sqrt(np.sum(gradient_2**2)))
            
            prev_geom_num_list_1 = geom_num_list_1
            prev_geom_num_list_2 = geom_num_list_2
            
            # Check convergence
            if delta_geometry < self.config.img_distance_convage_criterion:  # Bohr
                print("Converged!!!")
                break
            
            # Adjust beta if images are diverging
            if delta_geometry > prev_delta_geometry:
                self.config.BETA *= 1.02
        else:
            print("Reached maximum number of iterations. This is not converged.")
            with open(self.config.iEIP_FOLDER_DIRECTORY+"not_converged.txt", "w") as f:
                f.write("Reached maximum number of iterations. This is not converged.")
        
        # Create energy and gradient profile plots
        bias_ene_list = BIAS_ENERGY_LIST_A + BIAS_ENERGY_LIST_B[::-1]
        bias_grad_list = BIAS_GRAD_LIST_A + BIAS_GRAD_LIST_B[::-1]
        
        ene_list = ENERGY_LIST_A + ENERGY_LIST_B[::-1]
        grad_list = GRAD_LIST_A + GRAD_LIST_B[::-1]
        NUM_LIST = [i for i in range(len(ene_list))]
        
        G.single_plot(NUM_LIST, ene_list, file_directory_1, "energy", axis_name_2="energy [kcal/mol]", name="energy")   
        G.single_plot(NUM_LIST, grad_list, file_directory_1, "gradient", axis_name_2="grad (RMS) [a.u.]", name="gradient")
        G.single_plot(NUM_LIST, bias_ene_list, file_directory_1, "bias_energy", axis_name_2="energy [kcal/mol]", name="energy")   
        G.single_plot(NUM_LIST, bias_grad_list, file_directory_1, "bias_gradient", axis_name_2="grad (RMS) [a.u.]", name="gradient")
        FIO1.make_traj_file_for_DM(img_1="A", img_2="B")
        
        # Identify critical points
        FIO1.argrelextrema_txt_save(ene_list, "approx_TS", "max")
        FIO1.argrelextrema_txt_save(ene_list, "approx_EQ", "min")
        FIO1.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        return
    
    def norm_dist_2imgs(self, geom_num_list_1, geom_num_list_2):
        """Calculate normalized distance vector between two images"""
        L = self.dist_2imgs(geom_num_list_1, geom_num_list_2)
        N = (geom_num_list_2 - geom_num_list_1) / L
        return N 
    
    def dist_2imgs(self, geom_num_list_1, geom_num_list_2):
        """Calculate distance between two images"""
        L = np.linalg.norm(geom_num_list_2 - geom_num_list_1) + 1e-10
        return L  # Bohr
   
    def target_dist_2imgs(self, L):
        """Calculate target distance between two images"""
        Lt = max(L * 0.9, self.config.L_covergence - 0.01)
        return Lt
    
    def force_R(self, L):
        """Calculate force magnitude based on distance"""
        F_R = min(max(L/self.config.L_covergence, 1)) * self.F_R_convage_criterion
        return F_R  

    def displacement(self, force):
        """Calculate displacement from force with magnitude limit"""
        n_force = np.linalg.norm(force)
        displacement = (force / (n_force + 1e-10)) * min(n_force, self.config.displacement_limit)
        return displacement
    
    def displacement_prime(self, force):
        """Calculate displacement from force with fixed magnitude"""
        n_force = np.linalg.norm(force)
        displacement = (force / (n_force + 1e-10)) * self.config.displacement_limit
        return displacement
    
    def initial_structure_dependent_force(self, geom, ini_geom):
        """Calculate force toward initial structure"""
        ini_force = geom - ini_geom
        return ini_force
    
    def perpendicular_force(self, gradient, N):
        """Calculate force component perpendicular to path"""
        perp_force = gradient.reshape(len(gradient)*3, 1) - np.dot(gradient.reshape(1, len(gradient)*3), N.reshape(len(gradient)*3, 1)) * N.reshape(len(gradient)*3, 1)
        return perp_force.reshape(len(gradient), 3)  # (atomnum×3, ndarray)
    
    def delta_energy_force(self, ene_1, ene_2, N, L):
        """Calculate force component due to energy difference"""
        d_ene_force = N * abs(ene_1 - ene_2) / L
        return d_ene_force
    
    def close_target_force(self, L, Lt, geom_num_list_1, geom_num_list_2):
        """Calculate force component to maintain target distance"""
        ct_force = (geom_num_list_2 - geom_num_list_1) * (L - Lt) / L
        return ct_force

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
    
    
class ModelFunctionOptimizer:
    """
    Implementation of model function optimization for iEIP method.
    Optimizes different model functions for locating transition states and
    crossing points between potential energy surfaces.
    """
    def __init__(self, config):
        self.config = config
        
    def print_info_for_model_func(self, optmethod, e, B_e, B_g, displacement_vector, pre_e, pre_B_e):
        """Print model function optimization information"""
        print("calculation results (unit a.u.):")
        print("OPT method            : {} ".format(optmethod))
        print("                         Value                         Threshold ")
        print("ENERGY                : {:>15.12f} ".format(e))
        print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
        print("Maximum  Force        : {0:>15.12f}             {1:>15.12f} ".format(
            abs(B_g.max()), self.config.MAX_FORCE_THRESHOLD))
        print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(
            abs(np.sqrt((B_g**2).mean())), self.config.RMS_FORCE_THRESHOLD))
        print("Maximum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(
            abs(displacement_vector.max()), self.config.MAX_DISPLACEMENT_THRESHOLD))
        print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(
            abs(np.sqrt((displacement_vector**2).mean())), self.config.RMS_DISPLACEMENT_THRESHOLD))
        print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
        print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
        return
    
    def check_converge_criteria(self, B_g, displacement_vector):
        """Check convergence criteria for model function optimization"""
        max_force = abs(B_g.max())
        max_force_threshold = self.config.MAX_FORCE_THRESHOLD
        rms_force = abs(np.sqrt((B_g**2).mean()))
        rms_force_threshold = self.config.RMS_FORCE_THRESHOLD

        max_displacement = abs(displacement_vector.max())
        max_displacement_threshold = self.config.MAX_DISPLACEMENT_THRESHOLD
        rms_displacement = abs(np.sqrt((displacement_vector**2).mean()))
        rms_displacement_threshold = self.config.RMS_DISPLACEMENT_THRESHOLD
        
        if max_force < max_force_threshold and rms_force < rms_force_threshold and \
           max_displacement < max_displacement_threshold and rms_displacement < rms_displacement_threshold:
            return True, max_displacement_threshold, rms_displacement_threshold
       
        return False, max_displacement_threshold, rms_displacement_threshold
    
    def model_function_optimization(self, file_directory_list, SP_list, element_list_list, electric_charge_and_multiplicity_list, FIO_img_list):
        """
        Perform model function optimization to locate specific points on PESs.
        
        Supported model functions:
        - seam: Finds seam between potential energy surfaces
        - avoiding: Finds avoided crossing points
        - conical: Finds conical intersections
        - mesx/mesx2: Finds minimum energy crossing points
        - meci: Finds minimum energy conical intersections
        """
        G = Graph(self.config.iEIP_FOLDER_DIRECTORY)
        BIAS_GRAD_LIST_LIST = [[] for i in range(len(SP_list))]
        BIAS_MF_GRAD_LIST = [[] for i in range(len(SP_list))]
        BIAS_ENERGY_LIST_LIST = [[] for i in range(len(SP_list))]
        BIAS_MF_ENERGY_LIST = []
        GRAD_LIST_LIST = [[] for i in range(len(SP_list))]
        MF_GRAD_LIST = [[] for i in range(len(SP_list))]
        ENERGY_LIST_LIST = [[] for i in range(len(SP_list))]
        MF_ENERGY_LIST = []

        for iter in range(0, self.config.microiterlimit):
            if os.path.isfile(self.config.iEIP_FOLDER_DIRECTORY+"end.txt"):
                break
            print("# ITR. "+str(iter))
            
            tmp_gradient_list = []
            tmp_energy_list = []
            tmp_geometry_list = []
            exit_flag = False
            
            # Compute energy, gradient, and geometry for all systems
            for j in range(len(SP_list)):
                energy, gradient, geom_num_list, exit_flag = SP_list[j].single_point(
                    file_directory_list[j], element_list_list[j], iter, 
                    electric_charge_and_multiplicity_list[j], self.config.force_data["xtb"])
                if exit_flag:
                    break
                tmp_gradient_list.append(gradient)
                tmp_energy_list.append(energy)
                tmp_geometry_list.append(geom_num_list)
            
            if exit_flag:
                break
            
            tmp_gradient_list = np.array(tmp_gradient_list)
            tmp_energy_list = np.array(tmp_energy_list)
            tmp_geometry_list = np.array(tmp_geometry_list)
            
            # Initialize on first iteration
            if iter == 0:
                PREV_GRAD_LIST = []
                PREV_BIAS_GRAD_LIST = []
                PREV_MOVE_VEC_LIST = []
                PREV_GEOM_LIST = []
                PREV_GRAD_LIST = []
                PREV_MF_BIAS_GRAD_LIST = []
                PREV_MF_GRAD_LIST = []
                PREV_B_e_LIST = []
                PREV_e_LIST = []
                PREV_MF_e = 0.0
                PREV_MF_B_e = 0.0
                CMV = None
                
                optimizer_instances = None
                for j in range(len(SP_list)):
                    PREV_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_BIAS_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_MOVE_VEC_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_MF_BIAS_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_MF_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_B_e_LIST.append(0.0)
                    PREV_e_LIST.append(0.0)
                   
                CMV = CalculateMoveVector("x", element_list_list[j], 0, SP_list[j].FC_COUNT, 0)
                    
                optimizer_instances = CMV.initialization(self.config.force_data["opt_method"])
                for i in range(len(optimizer_instances)):
                    optimizer_instances[i].set_hessian(np.eye((len(geom_num_list)*3)))
                    
                init_geom_list = tmp_geometry_list
                PREV_GEOM_LIST = tmp_geometry_list
                
                # Initialize appropriate model function
                if self.config.mf_mode == "seam":
                    SMF = MF.SeamModelFunction()
                elif self.config.mf_mode == "avoiding":
                    AMF = MF.AvoidingModelFunction()
                elif self.config.mf_mode == "conical":
                    CMF = MF.ConicalModelFunction()
                elif self.config.mf_mode == "mesx2":
                    MESX = MF.OptMESX2()
                elif self.config.mf_mode == "mesx":
                    MESX = MF.OptMESX()
                elif self.config.mf_mode == "meci":
                    MECI_bare = MF.OptMECI()
                    MECI_bias = MF.OptMECI()
                else:
                    print("Unexpected method. exit...")
                    raise ValueError(f"Unsupported model function: {self.config.mf_mode}")

            # Calculate bias potential and gradient
            BPC_LIST = []
            for j in range(len(SP_list)):
                BPC_LIST.append(BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY))
                
            tmp_bias_energy_list = []
            tmp_bias_gradient_list = []
            tmp_bias_hessian_list = []
            
            for j in range(len(SP_list)):
                _, bias_energy, bias_gradient, BPA_hessian = BPC_LIST[j].main(
                    tmp_energy_list[j], tmp_gradient_list[j], tmp_geometry_list[j], 
                    element_list_list[j], self.config.force_data)
                
                for l in range(len(optimizer_instances)):
                    optimizer_instances[l].set_bias_hessian(BPA_hessian)
                
                tmp_bias_hessian_list.append(BPA_hessian)
                tmp_bias_energy_list.append(bias_energy)
                tmp_bias_gradient_list.append(bias_gradient)
 
            tmp_bias_energy_list = np.array(tmp_bias_energy_list)
            tmp_bias_gradient_list = np.array(tmp_bias_gradient_list)
 
            ##-----
            ##  Calculate model function energy, gradient and hessian
            ##-----
            if self.config.mf_mode == "seam":
                mf_energy = SMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = SMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = SMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], 
                                                    tmp_gradient_list[0], tmp_gradient_list[1])
                smf_bias_grad_1, smf_bias_grad_2 = SMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], 
                                                              tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                
                # Calculate Hessian if needed
                if iter % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
                    hess_list = []
                    for l in range(len(SP_list)):
                        tmp_hess = 0.5 * (SP_list[l].Model_hess + SP_list[l].Model_hess.T)
                        hess_list.append(tmp_hess)
                    gp_hess = SMF.calc_hess(tmp_energy_list[0], tmp_energy_list[1], 
                                         tmp_gradient_list[0], tmp_gradient_list[1], 
                                         hess_list[0], hess_list[1])
                    
                    for l in range(len(optimizer_instances)):
                        optimizer_instances[l].set_hessian(gp_hess)
                   
                bias_gp_hess = SMF.calc_hess(
                    tmp_bias_energy_list[0] - tmp_energy_list[0], 
                    tmp_bias_energy_list[1] - tmp_energy_list[1], 
                    tmp_bias_gradient_list[0] - tmp_gradient_list[0], 
                    tmp_bias_gradient_list[1] - tmp_gradient_list[1], 
                    tmp_bias_hessian_list[0], tmp_bias_hessian_list[1])
                
                for l in range(len(optimizer_instances)):
                    optimizer_instances[l].set_bias_hessian(bias_gp_hess)

            elif self.config.mf_mode == "avoiding":
                mf_energy = AMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = AMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = AMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], 
                                                    tmp_gradient_list[0], tmp_gradient_list[1])
                smf_bias_grad_1, smf_bias_grad_2 = AMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], 
                                                              tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                
                if iter % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
                    raise NotImplementedError("Not implemented Hessian of AMF.")

            elif self.config.mf_mode == "conical":
                mf_energy = CMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = CMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = CMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], 
                                                    tmp_gradient_list[0], tmp_gradient_list[1])
                smf_bias_grad_1, smf_bias_grad_2 = CMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], 
                                                              tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                
                if iter % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
                    raise NotImplementedError("Not implemented Hessian of CMF.")

            elif self.config.mf_mode == "mesx" or self.config.mf_mode == "mesx2":
                mf_energy = MESX.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = MESX.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                gp_grad = MESX.calc_grad(tmp_energy_list[0], tmp_energy_list[1], 
                                      tmp_gradient_list[0], tmp_gradient_list[1])
                gp_bias_grad = MESX.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], 
                                           tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [gp_bias_grad, gp_bias_grad]
                tmp_smf_grad_list = [gp_grad, gp_grad]
                
                if iter % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
                    hess_list = []
                    for l in range(len(SP_list)):
                        tmp_hess = 0.5 * (SP_list[l].Model_hess + SP_list[l].Model_hess.T)
                        hess_list.append(tmp_hess)
                    gp_hess = MESX.calc_hess(tmp_gradient_list[0], tmp_gradient_list[1], 
                                          hess_list[0], hess_list[1])
                    
                    for l in range(len(optimizer_instances)):
                        optimizer_instances[l].set_hessian(gp_hess)
                   
            elif self.config.mf_mode == "meci":
                mf_energy = MECI_bare.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = MECI_bias.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                gp_grad = MECI_bare.calc_grad(tmp_energy_list[0], tmp_energy_list[1], 
                                           tmp_gradient_list[0], tmp_gradient_list[1])
                gp_bias_grad = MECI_bias.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], 
                                                tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [gp_bias_grad, gp_bias_grad]
                tmp_smf_grad_list = [gp_grad, gp_grad]
                
                if iter % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
                    hess_list = []
                    for l in range(len(SP_list)):
                        tmp_hess = 0.5 * (SP_list[l].Model_hess + SP_list[l].Model_hess.T)
                        hess_list.append(tmp_hess)
                    gp_hess = MECI_bare.calc_hess(tmp_gradient_list[0], tmp_gradient_list[1], 
                                               hess_list[0], hess_list[1])
                    
                    for l in range(len(optimizer_instances)):
                        optimizer_instances[l].set_hessian(gp_hess)
                    
            else:
                print("No model function is selected.")
                raise
            
            tmp_smf_bias_grad_list = np.array(tmp_smf_bias_grad_list)
            tmp_smf_grad_list = np.array(tmp_smf_grad_list)            
            tmp_move_vector_list = []
            tmp_new_geometry_list = []
            
            CMV.trust_radii = 0.1
                
            _, tmp_move_vector, _ = CMV.calc_move_vector(iter, tmp_geometry_list[0], 
                                                      tmp_smf_bias_grad_list[0], 
                                                      PREV_MF_BIAS_GRAD_LIST[0], 
                                                      PREV_GEOM_LIST[0], 
                                                      PREV_MF_e, 
                                                      PREV_MF_B_e, 
                                                      PREV_MOVE_VEC_LIST[0], 
                                                      init_geom_list[0], 
                                                      tmp_smf_grad_list[0], 
                                                      PREV_GRAD_LIST[0], 
                                                      optimizer_instances)
            
            for j in range(len(SP_list)):
                tmp_move_vector_list.append(tmp_move_vector)
                tmp_new_geometry_list.append((tmp_geometry_list[j]-tmp_move_vector)*self.config.bohr2angstroms)
                        
            tmp_move_vector_list = np.array(tmp_move_vector_list)
            tmp_new_geometry_list = np.array(tmp_new_geometry_list)

            for j in range(len(SP_list)):
                tmp_new_geometry_list[j] -= Calculationtools().calc_center_of_mass(
                    tmp_new_geometry_list[j], element_list_list[j])
                tmp_new_geometry_list[j], _ = Calculationtools().kabsch_algorithm(
                    tmp_new_geometry_list[j], PREV_GEOM_LIST[j])
                
            tmp_new_geometry_list_to_list = tmp_new_geometry_list.tolist()
            
            for j in range(len(SP_list)):
                for i, elem in enumerate(element_list_list[j]):
                    tmp_new_geometry_list_to_list[j][i].insert(0, elem)
                
            for j in range(len(SP_list)):
                tmp_new_geometry_list_to_list[j].insert(0, electric_charge_and_multiplicity_list[j])
                
            for j in range(len(SP_list)):
                print(f"Input: {j}")
                _ = FIO_img_list[j].print_geometry_list(
                    tmp_new_geometry_list[j], element_list_list[j], [])
                file_directory_list[j] = FIO_img_list[j].make_psi4_input_file(
                    [tmp_new_geometry_list_to_list[j]], iter+1)
                print()
              
            # Store values for next iteration
            PREV_GRAD_LIST = tmp_gradient_list
            PREV_BIAS_GRAD_LIST = tmp_bias_gradient_list
            PREV_MOVE_VEC_LIST = tmp_move_vector_list
            PREV_GEOM_LIST = tmp_new_geometry_list

            PREV_MF_BIAS_GRAD_LIST = tmp_bias_gradient_list
            PREV_MF_GRAD_LIST = tmp_smf_grad_list
            PREV_B_e_LIST = tmp_bias_energy_list
            PREV_e_LIST = tmp_energy_list
            
            # Record data for plotting
            BIAS_MF_ENERGY_LIST.append(mf_bias_energy)
            MF_ENERGY_LIST.append(mf_energy)
            for j in range(len(SP_list)):
                BIAS_GRAD_LIST_LIST[j].append(np.sqrt(np.sum(tmp_bias_gradient_list[j]**2)))
                BIAS_ENERGY_LIST_LIST[j].append(tmp_bias_energy_list[j])
                GRAD_LIST_LIST[j].append(np.sqrt(np.sum(tmp_gradient_list[j]**2)))
                ENERGY_LIST_LIST[j].append(tmp_energy_list[j])
                MF_GRAD_LIST[j].append(np.sqrt(np.sum(tmp_smf_grad_list[j]**2)))
                BIAS_MF_GRAD_LIST[j].append(np.sqrt(np.sum(tmp_smf_bias_grad_list[j]**2)))
            
            self.print_info_for_model_func(self.config.force_data["opt_method"], 
                                        mf_energy, mf_bias_energy, 
                                        tmp_smf_bias_grad_list, tmp_move_vector_list, 
                                        PREV_MF_e, PREV_MF_B_e)
            
            PREV_MF_e = mf_energy
            PREV_MF_B_e = mf_bias_energy
            
            # Check convergence
            converge_check_flag, _, _ = self.check_converge_criteria(tmp_smf_bias_grad_list, tmp_move_vector_list)
            if converge_check_flag:  # convergence criteria met
                print("Converged!!!")
                break

        # Generate plots and save data
        NUM_LIST = [i for i in range(len(BIAS_MF_ENERGY_LIST))]
        MF_ENERGY_LIST = np.array(MF_ENERGY_LIST)
        BIAS_MF_ENERGY_LIST = np.array(BIAS_MF_ENERGY_LIST)
        ENERGY_LIST_LIST = np.array(ENERGY_LIST_LIST)
        GRAD_LIST_LIST = np.array(GRAD_LIST_LIST)
        BIAS_ENERGY_LIST_LIST = np.array(BIAS_ENERGY_LIST_LIST)
        BIAS_GRAD_LIST_LIST = np.array(BIAS_GRAD_LIST_LIST)
        MF_GRAD_LIST = np.array(MF_GRAD_LIST)
        BIAS_MF_GRAD_LIST = np.array(BIAS_MF_GRAD_LIST)
        
        # Create model function energy plots
        G.single_plot(NUM_LIST, MF_ENERGY_LIST*self.config.hartree2kcalmol, 
                   file_directory_list[0], "model_function_energy", 
                   axis_name_2="energy [kcal/mol]", name="model_function_energy")   
        G.single_plot(NUM_LIST, BIAS_MF_ENERGY_LIST*self.config.hartree2kcalmol, 
                   file_directory_list[0], "model_function_bias_energy", 
                   axis_name_2="energy [kcal/mol]", name="model_function_bias_energy")   
        G.double_plot(NUM_LIST, MF_ENERGY_LIST*self.config.hartree2kcalmol, 
                   BIAS_MF_ENERGY_LIST*self.config.hartree2kcalmol, 
                   add_file_name="model_function_energy")
        
        # Save model function energy data to CSV files
        with open(self.config.iEIP_FOLDER_DIRECTORY+"model_function_energy_"+str(j+1)+".csv", "w") as f:
            for k in range(len(NUM_LIST)):
                f.write(str(NUM_LIST[k])+","+str(MF_ENERGY_LIST[k])+"\n") 
        with open(self.config.iEIP_FOLDER_DIRECTORY+"model_function_bias_energy_"+str(j+1)+".csv", "w") as f:
            for k in range(len(NUM_LIST)):
                f.write(str(NUM_LIST[k])+","+str(BIAS_MF_ENERGY_LIST[k])+"\n")
        
        # Create and save plots and data for each state
        for j in range(len(SP_list)):
            G.single_plot(NUM_LIST, ENERGY_LIST_LIST[j]*self.config.hartree2kcalmol, 
                       file_directory_list[j], "energy_"+str(j+1), 
                       axis_name_2="energy [kcal/mol]", name="energy_"+str(j+1))   
            G.single_plot(NUM_LIST, GRAD_LIST_LIST[j], 
                       file_directory_list[j], "gradient_"+str(j+1), 
                       axis_name_2="grad (RMS) [a.u.]", name="gradient_"+str(j+1))
            G.single_plot(NUM_LIST, BIAS_ENERGY_LIST_LIST[j]*self.config.hartree2kcalmol, 
                       file_directory_list[j], "bias_energy_"+str(j+1), 
                       axis_name_2="energy [kcal/mol]", name="bias_energy_"+str(j+1))   
            G.single_plot(NUM_LIST, BIAS_GRAD_LIST_LIST[j], 
                       file_directory_list[j], "bias_gradient_"+str(j+1), 
                       axis_name_2="grad (RMS) [a.u.]", name="bias_gradient_"+str(j+1))
            G.single_plot(NUM_LIST, MF_GRAD_LIST[j], 
                       file_directory_list[j], "model_func_gradient_"+str(j+1), 
                       axis_name_2="grad (RMS) [a.u.]", name="model_func_gradient_"+str(j+1))
            G.single_plot(NUM_LIST, BIAS_MF_GRAD_LIST[j], 
                       file_directory_list[j], "model_func_bias_gradient_"+str(j+1), 
                       axis_name_2="grad (RMS) [a.u.]", name="model_func_bias_gradient_"+str(j+1))
            
            # Save energy data to CSV files
            with open(self.config.iEIP_FOLDER_DIRECTORY+"energy_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(ENERGY_LIST_LIST[j][k])+"\n")
            with open(self.config.iEIP_FOLDER_DIRECTORY+"gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(GRAD_LIST_LIST[j][k])+"\n")
            with open(self.config.iEIP_FOLDER_DIRECTORY+"bias_energy_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(BIAS_ENERGY_LIST_LIST[j][k])+"\n")
            with open(self.config.iEIP_FOLDER_DIRECTORY+"bias_gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(BIAS_GRAD_LIST_LIST[j][k])+"\n")
            with open(self.config.iEIP_FOLDER_DIRECTORY+"model_func_gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(MF_GRAD_LIST[j][k])+"\n")
            with open(self.config.iEIP_FOLDER_DIRECTORY+"model_func_bias_gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(BIAS_MF_GRAD_LIST[j][k])+"\n")
                    
        # Generate trajectory files and identify critical points
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for j in range(len(SP_list)):
            FIO_img_list[j].argrelextrema_txt_save(ENERGY_LIST_LIST[j], "approx_TS_"+str(j+1), "max")
            FIO_img_list[j].argrelextrema_txt_save(ENERGY_LIST_LIST[j], "approx_EQ_"+str(j+1), "min")
            FIO_img_list[j].argrelextrema_txt_save(GRAD_LIST_LIST[j], "local_min_grad_"+str(j+1), "min")
            
            FIO_img_list[j].make_traj_file(name=alphabet[j])
        
        return


class iEIP:
    """
    Main class for Improved Elastic Image Pair (iEIP) method.
    
    Manages the overall optimization process by delegating to specialized
    component classes for configuration, elastic image pair optimization,
    model function optimization.
    
    
    References:
    - J. Chem. Theory. Comput. 2023, 19, 2410-2417
    - J. Comput. Chem. 2018, 39, 233–251 (DS-AFIR)
    """
    def __init__(self, args):
        self.config = IEIPConfig(args)
        self.elastic_image_pair = ElasticImagePair(self.config)
        self.model_function_optimizer = ModelFunctionOptimizer(self.config)
        self.newton_trajectory = NewtonTrajectory(self.config)
    
    def optimize(self):
        """Load calculation modules based on configuration and run optimization"""
        if self.config.othersoft != "None":
            from ase_calculation_tools import Calculation
        elif self.config.args.pyscf:
            from pyscf_calculation_tools import Calculation
        elif self.config.args.usextb != "None" and self.config.args.usedxtb == "None":
            from tblite_calculation_tools import Calculation
        elif self.config.args.usedxtb != "None" and self.config.args.usextb == "None":
            from dxtb_calculation_tools import Calculation
        else:
            from psi4_calculation_tools import Calculation
        
        file_path_list = glob.glob(self.config.START_FILE+"*_[A-Z].xyz")
        FIO_img_list = []

        for file_path in file_path_list:
            FIO_img_list.append(FileIO(self.config.iEIP_FOLDER_DIRECTORY, file_path))

        geometry_list_list = []
        element_list_list = []
        electric_charge_and_multiplicity_list = []

        for i in range(len(FIO_img_list)):
            geometry_list, element_list, electric_charge_and_multiplicity = FIO_img_list[i].make_geometry_list(
                self.config.electric_charge_and_multiplicity_list[i])
            
            if self.config.args.pyscf:
                electric_charge_and_multiplicity = [self.config.electronic_charge[i], 
                                                 self.config.spin_multiplicity[i]]
            
            geometry_list_list.append(geometry_list)
            element_list_list.append(element_list)
            electric_charge_and_multiplicity_list.append(electric_charge_and_multiplicity)
            
        # Save input configuration data
        self.config.save_input_data()
        
        # Initialize calculation objects
        SP_list = []
        file_directory_list = []
        for i in range(len(FIO_img_list)):
            SP_list.append(Calculation(
                START_FILE = self.config.START_FILE,
                N_THREAD = self.config.N_THREAD,
                SET_MEMORY = self.config.SET_MEMORY,
                FUNCTIONAL = self.config.FUNCTIONAL,
                FC_COUNT = self.config.FC_COUNT,
                BPA_FOLDER_DIRECTORY = self.config.iEIP_FOLDER_DIRECTORY,
                Model_hess = np.eye(3*len(geometry_list_list[i])),
                unrestrict = self.config.unrestrict, 
                BASIS_SET = self.config.BASIS_SET,
                SUB_BASIS_SET = self.config.SUB_BASIS_SET,
                electronic_charge = self.config.electronic_charge[i] or electric_charge_and_multiplicity_list[i][0],
                spin_multiplicity = self.config.spin_multiplicity[i] or electric_charge_and_multiplicity_list[i][1],
                excited_state = self.config.excite_state_list[i],
                dft_grid = self.config.dft_grid,
                ECP = self.config.ECP,
                software_type = self.config.othersoft
            ))

            SP_list[i].cpcm_solv_model = self.config.cpcm_solv_model
            SP_list[i].alpb_solv_model = self.config.alpb_solv_model
            
            file_directory = FIO_img_list[i].make_psi4_input_file(geometry_list_list[i], 0)
            file_directory_list.append(file_directory)
        
        # Run optimization with appropriate method
        if self.config.mf_mode != "None":
            self.model_function_optimizer.model_function_optimization(
                file_directory_list, SP_list, element_list_list, 
                self.config.electric_charge_and_multiplicity_list, FIO_img_list)
        elif self.config.use_gnt:
            self.newton_trajectory.main(file_directory_list[0], file_directory_list[1], 
                SP_list[0], SP_list[1], element_list_list[0], 
                self.config.electric_charge_and_multiplicity_list[0], 
                self.config.electric_charge_and_multiplicity_list[1], 
                FIO_img_list[0], FIO_img_list[1])
        
        else:
            self.elastic_image_pair.iteration(
                file_directory_list[0], file_directory_list[1], 
                SP_list[0], SP_list[1], element_list_list[0], 
                self.config.electric_charge_and_multiplicity_list[0], 
                self.config.electric_charge_and_multiplicity_list[1], 
                FIO_img_list[0], FIO_img_list[1])
        
        return

    def run(self):
        """Run the optimization process and display completion message"""
        self.optimize()
        print("completed...")
        return