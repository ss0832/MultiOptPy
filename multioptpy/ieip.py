import os
import sys
import datetime
import glob
import numpy as np
import copy
import shutil
from pathlib import Path
from scipy.optimize import minimize


from potential import BiasPotentialCalculation
from optimizer import CalculateMoveVector 
from calc_tools import Calculationtools
from visualization import Graph
from fileio import FileIO, make_workspace, xyz2list
from parameter import UnitValueLib, element_number
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
        
        # Set config for ADDF-like method
        self.use_addf = getattr(args, 'use_addf', False)
        self.addf_step_num = getattr(args, 'addf_step_num', 300)
        self.nadd = getattr(args, 'number_of_add', 5)
        self.addf_step_size = getattr(args, 'addf_step_size', 0.05)


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

class ADDFlikeMethod:
    def __init__(self, config):
        """
        Implementation of ADD (Anharmonic Downward Distortion) method based on SHS4py approach
        # ref. : Journal of chemical theory and computation 16.6 (2020): 3869-3878. (https://pubs.acs.org/doi/10.1021/acs.jctc.0c00010)
        # ref. : Chem. Phys. Lett. 2004, 384 (4–6), 277–282.
        # ref. : Chem. Phys. Lett. 2005, 404 (1–3), 95–99.
        """
        self.config = config
        self.addf_config = {
            'step_number': int(config.addf_step_num),
            'number_of_add': int(config.nadd),
            'IOEsphereA_initial': 0.01,  # Initial hypersphere radius
            'IOEsphereA_dist': float(config.addf_step_size),    # Increment for hypersphere radius
            'IOEthreshold': 0.01,        # Threshold for IOE
            'minimize_threshold': 1.0e-5,# Threshold for minimization
        }
        self.energy_list_1 = []
        self.energy_list_2 = []
        self.gradient_list_1 = []
        self.gradient_list_2 = []
        self.init_displacement = 0.03 / self.get_unit_conversion()  # Bohr
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.converge_criteria = 0.12
        self.element_number_list = None
        self.ADDths = []  # List to store ADD theta classes
        self.optimized_structures = {}  # Dictionary to store optimized structures by ADD ID
        self.max_iterations = 20

    def get_unit_conversion(self):
        """Return bohr to angstrom conversion factor"""
        return UnitValueLib().bohr2angstroms  # Approximate value for bohr2angstroms
        
    def adjust_center2origin(self, coord):
        """Adjust coordinates to have center at origin"""
        center = np.mean(coord, axis=0)
        return coord - center

    def SuperSphere_cartesian(self, A, thetalist, SQ, dim):
        """
        Vector of super sphere by cartesian (basis transformation from polar to cartesian)
        {sqrt(2*A), theta_1,..., theta_n-1} ->  {q_1,..., q_n} -> {x_1, x_2,..., x_n}
        """
        n_components = min(dim, SQ.shape[1] if SQ.ndim > 1 else dim)
        
        qlist = np.zeros(n_components)
        
        # Fill q-list using hyperspherical coordinates
        a_k = np.sqrt(2.0 * A)
        for i in range(min(len(thetalist), n_components-1)):
            qlist[i] = a_k * np.cos(thetalist[i])
            a_k *= np.sin(thetalist[i])
        
        # Handle the last component
        if n_components > 0:
            qlist[n_components-1] = a_k
        
        # Transform to original space
        SSvec = np.dot(SQ, qlist)
        
        return SSvec  # This is a vector in the reduced space
    
    def calctheta(self, vec, eigVlist, eigNlist):
        """
        Calculate the polar coordinates (theta) from a vector
        """
        # Get actual dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector
        n_components = min(len(eigVlist), len(eigNlist))  # Number of eigenvectors
        
        # Check vector dimensions
        if len(vec) != n_features:
            # If dimensions don't match, truncate or pad the vector
            if len(vec) > n_features:
                vec = vec[:n_features]  # Truncate
            else:
                padded_vec = np.zeros(n_features)
                padded_vec[:len(vec)] = vec
                vec = padded_vec
        
        # Create SQ_inv matrix with correct dimensions
        SQ_inv = np.zeros((n_components, n_features))
        
        for i in range(n_components):
            SQ_inv[i] = eigVlist[i] / np.sqrt(abs(eigNlist[i]))
        
        # Perform the dot product
        qvec = np.dot(SQ_inv, vec)
        
        r = np.linalg.norm(qvec)
        if r < 1e-10:
            return np.zeros(n_components - 1)
            
        thetalist = []
        for i in range(len(qvec) - 1):
            # Handle possible numerical issues with normalization
            norm_q = np.linalg.norm(qvec[i:])
            if norm_q < 1e-10:
                theta = 0.0
            else:
                cos_theta = qvec[i] / norm_q
                cos_theta = max(-1.0, min(1.0, cos_theta))  # Ensure within bounds
                theta = np.arccos(cos_theta)
                if i == len(qvec) - 2 and qvec[-1] < 0:
                    theta = 2*np.pi - theta
            thetalist.append(theta)
        
        return np.array(thetalist)
    
    def SQaxes(self, eigNlist, eigVlist, dim):
        """Calculate the SQ matrix for transformation"""
        # Get actual available dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector
        n_components = min(len(eigVlist), len(eigNlist), dim)  # Number of eigenvectors to use
        
        # Initialize with correct dimensions
        SQ = np.zeros((n_features, n_components))
        
        # Only iterate up to the available components
        for i in range(n_components):
            SQ[:, i] = eigVlist[i] * np.sqrt(abs(eigNlist[i]))
        
        return SQ

    def SQaxes_inv(self, eigNlist, eigVlist, dim):
        """Calculate the inverse SQ matrix for transformation"""
        # Get actual available dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector 
        n_components = min(len(eigVlist), len(eigNlist), dim)  # Number of eigenvectors to use
        
        # Initialize with correct dimensions
        SQ_inv = np.zeros((n_components, n_features))
        
        # Only iterate up to the available components
        for i in range(n_components):
            SQ_inv[i] = eigVlist[i] / np.sqrt(abs(eigNlist[i]))
        
        return SQ_inv
    
    def angle(self, v1, v2):
        """Calculate angle between two vectors"""
        # Check for zero vectors or invalid inputs
        if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
            return 0.0
            
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        
        # Handle potential numerical issues
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        
        return np.arccos(dot_product)
    
    def angle_SHS(self, v1, v2, SQ_inv):
        """Calculate angle between two vectors in SHS space"""
        # Ensure vectors are flattened
        v1_flat = v1.flatten() if hasattr(v1, 'flatten') else v1
        v2_flat = v2.flatten() if hasattr(v2, 'flatten') else v2
        
        # Handle potential dimension mismatches
        min_dim = min(len(v1_flat), len(v2_flat), SQ_inv.shape[1])
        v1_flat = v1_flat[:min_dim]
        v2_flat = v2_flat[:min_dim]
        
        q_v1 = np.dot(SQ_inv[:, :min_dim], v1_flat)
        q_v2 = np.dot(SQ_inv[:, :min_dim], v2_flat)
        return self.angle(q_v1, q_v2)
    
    def calc_onHS(self, deltaTH, func, eqpoint, thetalist, IOEsphereA, A_eq):
        """
        Calculate function value on hypersphere
        """
        thetalist_new = thetalist + deltaTH
        nADD = self.SuperSphere_cartesian(IOEsphereA, thetalist_new, self.SQ, self.dim)
        x = eqpoint + nADD
        x = self.periodicpoint(x)
        
        result = func(x) - IOEsphereA - A_eq
        result += self.IOE_total(nADD)
        return result
    
    def IOE_total(self, nADD):
        """Sum up all IOE illumination"""
        result = 0.0
        for ADDth in self.ADDths:
            if self.current_id == ADDth.IDnum:
                continue
            if ADDth.ADDoptQ:
                continue
            if ADDth.ADD_IOE <= -1000000 or ADDth.ADD_IOE > 10000000:
                continue
            if ADDth.ADD <= self.current_ADD:
                result -= self.IOE(nADD, ADDth)
        return result
    
    def IOE(self, nADD, neiborADDth):
        """Calculate IOE between current point and neighbor"""
        deltaTH = self.angle_SHS(nADD, neiborADDth.nADD, self.SQ_inv)
        if deltaTH <= np.pi * 0.5:
            cosdamp = np.cos(deltaTH)
            return neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
        else:
            return 0.0
    
    def grad_hypersphere(self, f, grad, eqpoint, IOEsphereA, thetalist):
        """Calculate gradient on hypersphere"""
        # Generate nADD in the reduced space (this will be a vector in the Hessian eigenspace)
        nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
        
        # We need to convert the reduced space vector back to the full coordinate space
        # First create a zero vector in the full space
        n_atoms = eqpoint.shape[0]
        n_coords = n_atoms * 3
        nADD_full = np.zeros(n_coords)
        
        # Map the reduced vector to the full space (this is approximate)
        for i in range(min(len(nADD_reduced), n_coords)):
            nADD_full[i] = nADD_reduced[i]
            
        # Reshape to molecular geometry format
        nADD = nADD_full.reshape(n_atoms, 3)
        
        # Calculate the normalized direction vector for projecting out later
        EnADD = nADD_full / np.linalg.norm(nADD_full)
        
        # Apply the displacement to the initial geometry
        target_point = eqpoint + nADD
        target_point = self.periodicpoint(target_point)
        
        # Calculate gradient at this point
        grad_x = grad(target_point)
        if isinstance(grad_x, bool) and grad_x is False:
            return False, False
        
        # Flatten the gradient for vector operations
        grad_x_flat = grad_x.flatten()
        
        # Project gradient onto tangent space of hypersphere
        # We remove the component along the displacement vector
        returngrad_flat = grad_x_flat - np.dot(grad_x_flat, EnADD) * EnADD
        
        # Apply IOE contributions (if implemented)
        for ADDth in self.ADDths:
            # Skip if this is the current ADD or if current_id is None
            if self.current_id is not None and self.current_id == ADDth.IDnum:
                continue
                
            # Skip if this ADD is being optimized
            if ADDth.ADDoptQ:
                continue
                
            # Only apply IOE for ADDs with lower energy (if current_ADD is set)
            if self.current_ADD is not None and hasattr(ADDth, 'ADD') and ADDth.ADD is not None:
                if ADDth.ADD <= self.current_ADD:
                    ioe_grad = self.IOE_grad(nADD_full, ADDth)
                    if ioe_grad is not None:
                        returngrad_flat -= ioe_grad
        
        # Reshape gradient back to molecular geometry format for easier handling
        returngrad = returngrad_flat.reshape(n_atoms, 3)
        
        return target_point, returngrad

    def IOE_grad(self, nADD, neiborADDth):
        """Calculate gradient of IOE"""
        # Make sure we're working with flattened arrays
        nADD_flat = nADD.flatten() if hasattr(nADD, 'flatten') else nADD
        nADD_neibor = neiborADDth.nADD_full if hasattr(neiborADDth, 'nADD_full') else neiborADDth.nADD.flatten()
        
        # Get minimum dimension we can work with
        min_dim = min(len(nADD_flat), len(nADD_neibor), self.SQ_inv.shape[1])
        nADD_flat = nADD_flat[:min_dim]
        nADD_neibor = nADD_neibor[:min_dim]
        
        # Transform to eigenspace
        q_x = np.dot(self.SQ_inv[:, :min_dim], nADD_flat)
        q_y = np.dot(self.SQ_inv[:, :min_dim], nADD_neibor)
        
        # Check for valid vectors before calculating angle
        if np.isnan(q_x).any() or np.isnan(q_y).any() or np.linalg.norm(q_x) < 1e-10 or np.linalg.norm(q_y) < 1e-10:
            return None
            
        # Calculate angle in eigenspace
        deltaTH = self.angle(q_x, q_y)
        
        # Check if deltaTH is valid (not None and not NaN)
        if deltaTH is None or np.isnan(deltaTH):
            return None
        
        # Initialize gradient vector
        returngrad = np.zeros(len(nADD_flat))
        eps = 1.0e-3
        
        # Calculate IOE gradient using finite differences
        if deltaTH <= np.pi * 0.5:
            cosdamp = np.cos(deltaTH)
            for i in range(len(nADD_flat)):
                nADD_eps = copy.copy(nADD_flat)
                nADD_eps[i] += eps
                
                # Transform to eigenspace
                qx_i = np.dot(self.SQ_inv[:, :min_dim], nADD_eps[:min_dim])
                deltaTH_eps = self.angle(qx_i, q_y)
                
                # Check if the new angle is valid
                if deltaTH_eps is None or np.isnan(deltaTH_eps):
                    continue
                    
                cosdamp_eps = np.cos(deltaTH_eps)
                IOE_center = neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
                IOE_eps = neiborADDth.ADD_IOE * cosdamp_eps * cosdamp_eps * cosdamp_eps
                
                returngrad[i] = (IOE_eps - IOE_center) / eps
            
            # Pad the gradient to the full space if needed
            full_grad = np.zeros(self.n_coords if hasattr(self, 'n_coords') else len(nADD))
            full_grad[:len(returngrad)] = returngrad
            return full_grad
        
        return None
    
    def periodicpoint(self, point):
        """Apply periodic boundary conditions if needed"""
        # Implement according to your specific requirements
        return point

    def minimizeTh_SD_SS(self, ADDth, initialpoint, f, grad, eqpoint, IOEsphereA):
        """
        Steepest descent optimization on hypersphere with step size control
        Following the implementation in SHS4py.ADD.py with added robustness
        """
        whileN = 0
        thetalist = ADDth.thetalist + initialpoint
        stepsize = 0.001
        n_atoms = eqpoint.shape[0]
        n_coords = n_atoms * 3
        
        # Generate initial nADD
        nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
        
        # Convert reduced space vector to full coordinate space with proper dimensions
        nADD_full = np.zeros(n_coords)
        for i in range(min(len(nADD_reduced), n_coords)):
            nADD_full[i] = nADD_reduced[i]
        
        # Reshape to match eqpoint dimensions
        nADD = nADD_full.reshape(n_atoms, 3)
        
        # Keep track of best solution
        best_thetalist = thetalist.copy()
        best_energy = float('inf')
        
        # Initial point
        tergetpoint = eqpoint + nADD
        tergetpoint = self.periodicpoint(tergetpoint)
        
        # Try to calculate initial energy
        try:
            initial_energy = f(tergetpoint)
            if isinstance(initial_energy, (int, float)) and not np.isnan(initial_energy):
                best_energy = initial_energy
        except Exception:
            pass  # Continue even if initial energy calculation fails
        
        # Main optimization loop
        while whileN < self.max_iterations:
            try:
                # Get gradient at current point
                grad_x = grad(tergetpoint)
                
                # If gradient calculation fails, continue with smaller step or different approach
                if grad_x is False:
                    # Try a random perturbation and continue
                    print(f"Gradient calculation failed at iteration {whileN}, trying random perturbation")
                    random_perturbation = np.random.rand(n_atoms, 3) * 0.01 - 0.005  # Small random perturbation
                    tergetpoint = tergetpoint + random_perturbation
                    tergetpoint = self.periodicpoint(tergetpoint)
                    
                    # Calculate new nADD
                    nADD = tergetpoint - eqpoint
                    thetalist = self.calctheta(nADD.flatten(), self.eigVlist, self.eigNlist)
                    
                    # Ensure we're on the hypersphere with correct radius
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    
                    tergetpoint = eqpoint + nADD
                    tergetpoint = self.periodicpoint(tergetpoint)
                    
                    whileN += 1
                    continue
                
                # Apply IOE contributions
                grad_flat = grad_x.flatten()
                for neiborADDth in self.ADDths:
                    if ADDth.IDnum == neiborADDth.IDnum:
                        continue
                    if neiborADDth.ADDoptQ:
                        continue
                    if neiborADDth.ADD <= ADDth.ADD:
                        ioe_grad = self.IOE_grad(nADD.flatten(), neiborADDth)
                        if ioe_grad is not None:
                            grad_flat = grad_flat - ioe_grad
                
                # Reshape back to molecular geometry format
                grad_x = grad_flat.reshape(n_atoms, 3)
                
                # Project gradient onto tangent space
                nADD_norm = np.linalg.norm(nADD.flatten())
                if nADD_norm < 1e-10:
                    # If nADD is too small, generate a new one
                    print(f"nADD norm too small at iteration {whileN}, regenerating")
                    thetalist = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    tergetpoint = eqpoint + nADD
                    tergetpoint = self.periodicpoint(tergetpoint)
                    whileN += 1
                    continue
                        
                nADD_unit = nADD.flatten() / nADD_norm
                
                # Project gradient component along nADD
                grad_along_nADD = np.dot(grad_x.flatten(), nADD_unit)
                
                # Subtract this component to get the tangent space gradient
                SSgrad_flat = grad_x.flatten() - grad_along_nADD * nADD_unit
                SSgrad = SSgrad_flat.reshape(n_atoms, 3)
                
                # Check convergence
                if np.linalg.norm(SSgrad) < 1.0e-1:
                    # Update best solution if better
                    try:
                        current_energy = f(tergetpoint)
                        if isinstance(current_energy, (int, float)) and not np.isnan(current_energy):
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_thetalist = thetalist.copy()
                    except Exception:
                        pass  # Just keep the current best if energy calculation fails
                    
                    return thetalist  # Converged successfully
                
                # Store current point
                _point_initial = copy.copy(tergetpoint)
                
                # Line search
                whileN2 = 0
                stepsizedamp = stepsize
                found_valid_step = False
                
                # Try multiple step sizes
                for whileN2 in range(1, 5):  # Try up to 10 steps with varying sizes
                    try:
                        # Take step with dynamic step size
                        step_scale = whileN2 if whileN2 <= 5 else (whileN2 - 5) * 0.1
                        tergetpoint = _point_initial - step_scale * SSgrad / np.linalg.norm(SSgrad) * stepsizedamp
                        
                        # Calculate new nADD
                        nADD2 = tergetpoint - eqpoint
                        
                        # Convert to theta parameters
                        thetalist_new = self.calctheta(nADD2.flatten(), self.eigVlist, self.eigNlist)
                        
                        # Ensure we're on the hypersphere with correct radius
                        nADD2_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist_new, self.SQ, self.dim)
                        
                        # Convert reduced space vector to full coordinate space
                        nADD2_full = np.zeros(n_coords)
                        for i in range(min(len(nADD2_reduced), n_coords)):
                            nADD2_full[i] = nADD2_reduced[i]
                        
                        # Reshape to match eqpoint dimensions
                        nADD2 = nADD2_full.reshape(n_atoms, 3)
                        
                        # Calculate new point on hypersphere
                        new_point = eqpoint + nADD2
                        new_point = self.periodicpoint(new_point)
                        
                        # Calculate step size
                        delta = np.linalg.norm(nADD.flatten() - nADD2.flatten())
                        
                        # Calculate energy at new point to check improvement
                        try:
                            new_energy = f(new_point)
                            if isinstance(new_energy, (int, float)) and not np.isnan(new_energy):
                                # Accept step if it improves energy or makes reasonable movement
                                if new_energy < best_energy or delta > 0.005:
                                    found_valid_step = True
                                    if new_energy < best_energy:
                                        best_energy = new_energy
                                        best_thetalist = thetalist_new.copy()
                                    tergetpoint = new_point
                                    thetalist = thetalist_new
                                    nADD = nADD2
                                    break
                        except Exception:
                            # If energy calculation fails, accept step if it's a reasonable move
                            if delta > 0.005 and delta < 0.1:
                                found_valid_step = True
                                tergetpoint = new_point
                                thetalist = thetalist_new
                                nADD = nADD2
                                break
                    except Exception as e:
                        print(f"Step calculation error: {e}, trying different step")
                        continue
                        
                # If no valid step found, try a random perturbation
                if not found_valid_step:
                    print(f"No valid step found at iteration {whileN}, trying random perturbation")
                    # Generate random perturbation but keep on hypersphere
                    random_theta = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                    random_nADD = self.SuperSphere_cartesian(IOEsphereA, random_theta, self.SQ, self.dim)
                    
                    # Interpolate between current point and random point
                    alpha = 0.1  # Small mixing factor
                    mixed_theta = thetalist * (1-alpha) + random_theta * alpha
                    
                    # Generate new point
                    mixed_nADD = self.SuperSphere_cartesian(IOEsphereA, mixed_theta, self.SQ, self.dim)
                    mixed_nADD_full = np.zeros(n_coords)
                    for i in range(min(len(mixed_nADD), n_coords)):
                        mixed_nADD_full[i] = mixed_nADD[i]
                    mixed_nADD = mixed_nADD_full.reshape(n_atoms, 3)
                    
                    # Update point
                    tergetpoint = eqpoint + mixed_nADD
                    tergetpoint = self.periodicpoint(tergetpoint)
                    nADD = mixed_nADD
                    thetalist = mixed_theta
                
                # Increment counter
                whileN += 1
                
                # Print progress periodically
                if whileN % 10 == 0:
                    print(f"Optimization step {whileN}: gradient norm = {np.linalg.norm(SSgrad):.6f}")
                    
            except Exception as e:
                print(f"Error in optimization step {whileN}: {e}, continuing with best solution")
                whileN += 1
                
                # Try to recover with random perturbation
                if whileN % 3 == 0:  # Every third error, try a more drastic change
                    try:
                        # Generate a completely new point on hypersphere
                        random_theta = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                        random_nADD = self.SuperSphere_cartesian(IOEsphereA, random_theta, self.SQ, self.dim)
                        
                        # Create full vector
                        random_nADD_full = np.zeros(n_coords)
                        for i in range(min(len(random_nADD), n_coords)):
                            random_nADD_full[i] = random_nADD[i]
                        random_nADD = random_nADD_full.reshape(n_atoms, 3)
                        
                        # Try the new point
                        new_point = eqpoint + random_nADD
                        new_point = self.periodicpoint(new_point)
                        
                        # Check if it's better
                        try:
                            random_energy = f(new_point)
                            if isinstance(random_energy, (int, float)) and not np.isnan(random_energy):
                                if random_energy < best_energy:
                                    best_energy = random_energy
                                    best_thetalist = random_theta.copy()
                                    tergetpoint = new_point
                                    nADD = random_nADD
                                    thetalist = random_theta
                        except Exception:
                            pass  # Ignore failed energy calculations
                    except Exception:
                        pass  # Ignore errors in recovery attempt
        
        print(f"Optimization completed with {whileN} iterations")
        # Return the best solution found
        return best_thetalist if best_energy < float('inf') else thetalist
    
    def detect_add(self, QMC):
        """Detect ADD directions from Hessian"""
        coord_1 = self.get_coord()
        coord_1 = self.adjust_center2origin(coord_1)
        n_atoms = coord_1.shape[0]
        n_coords = n_atoms * 3
        
        element_number_list_1 = self.get_element_number_list()
        print("### Checking whether initial structure is EQ. ###")
        
        QMC.hessian_flag = True
        self.init_energy, self.init_gradient, _, iscalculationfailed = QMC.single_point(
            None, element_number_list_1, "", self.electric_charge_and_multiplicity, 
            self.method, coord_1
        )
        
        if iscalculationfailed:
            print("Initial calculation failed.")
            return False
        
        # Apply bias potential if needed
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, bias_energy, bias_gradient, bias_hess = BPC.main(
            self.init_energy, self.init_gradient, coord_1, element_number_list_1, 
            self.config.force_data
        )
        self.init_energy = bias_energy
        self.init_gradient = bias_gradient
        
        QMC.hessian_flag = False
        self.init_geometry = coord_1  # Shape: (n_atoms, 3)
        self.set_coord(coord_1)
        
        if np.linalg.norm(self.init_gradient) > 1e-3:
            print("Norm of gradient is too large. Structure is not at equilibrium.")
            return False
                
        print("Initial structure is EQ.")
        print("### Start calculating Hessian matrix to detect ADD. ###")
        
        hessian = QMC.Model_hess + bias_hess
        
        # Project out translation and rotation
        projection_hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(hessian, self.element_list, coord_1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(projection_hessian)
        eigenvalues = eigenvalues.astype(np.float64)
        
        # Filter out near-zero eigenvalues
        nonzero_indices = np.where(np.abs(eigenvalues) > 1e-10)[0]
        nonzero_eigenvectors = eigenvectors[:, nonzero_indices].astype(np.float64)
        nonzero_eigenvalues = eigenvalues[nonzero_indices].astype(np.float64)
        
        sorted_idx = np.argsort(nonzero_eigenvalues)
        
        self.init_eigenvalues = nonzero_eigenvalues
        self.init_eigenvectors = nonzero_eigenvectors
        self.dim = len(nonzero_eigenvalues)
        self.n_atoms = n_atoms
        self.n_coords = n_coords
        
        # Store flattened versions for matrix operations
        self.init_geometry_flat = self.init_geometry.flatten()
        
        # Prepare SQ matrices
        self.SQ = self.SQaxes(nonzero_eigenvalues, nonzero_eigenvectors, self.dim)
        self.SQ_inv = self.SQaxes_inv(nonzero_eigenvalues, nonzero_eigenvectors, self.dim)
        
        self.eigNlist = nonzero_eigenvalues
        self.eigVlist = nonzero_eigenvectors
        
        # Calculate mode eigenvectors to try - focus on lower eigenvectors first
        # (corresponding to softer modes that are more likely to lead to transition states)
        search_idx = len(sorted_idx) # Start with more eigenvectors than needed
        self.sorted_eigenvalues_idx = sorted_idx[0:search_idx]
        
        # Initialize ADDths with initial directions
        self.ADDths = []
        IDnum = 0
        
        print("### Checking ADD energy. ###")
        with open(self.directory + "/add_energy_list.csv", "w") as f:
            f.write("index_of_principal_axis,eigenvalue,direction,add_energy,abs_add_energy\n")
        
        IOEsphereA = self.addf_config['IOEsphereA_initial']
        
        # Create candidate ADDs for eigenvectors
        candidate_ADDths = []
        
        for idx in self.sorted_eigenvalues_idx:
            for pm in [-1.0, 1.0]:
                eigV = self.init_eigenvectors[:, idx]
                
                # Create a new ADD point
                ADDth = type('ADDthetaClass', (), {})
                ADDth.IDnum = IDnum
                ADDth.dim = self.dim
                ADDth.SQ = self.SQ
                ADDth.SQ_inv = self.SQ_inv
                ADDth.thetalist = self.calctheta(pm * eigV, nonzero_eigenvectors, nonzero_eigenvalues)
                
                # Generate nADD (this will be a flattened vector in the Hessian eigenspace)
                ADDth.nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, self.SQ, self.dim)
                
                # We need to convert the reduced space vector back to the full coordinate space
                # First create a zero vector in the full space
                ADDth.nADD_full = np.zeros(n_coords)
                
                # Map the reduced vector to the full space (this is approximate)
                for i in range(min(len(ADDth.nADD_reduced), n_coords)):
                    ADDth.nADD_full[i] = ADDth.nADD_reduced[i]
                    
                # Reshape to molecular geometry format
                ADDth.nADD = ADDth.nADD_full.reshape(n_atoms, 3)
                
                # Apply the displacement to the initial geometry
                ADDth.x = self.init_geometry + ADDth.nADD
                ADDth.x = self.periodicpoint(ADDth.x)
                
                # Calculate energy
                energy, grad_x, _, iscalculationfailed = QMC.single_point(
                    None, element_number_list_1, "", self.electric_charge_and_multiplicity, 
                    self.method, ADDth.x
                )
                
                if iscalculationfailed:
                    continue
                    
                # Apply bias if needed
                _, bias_energy, bias_gradient, _ = BPC.main(
                    energy, grad_x, ADDth.x, element_number_list_1, 
                    self.config.force_data
                )
                
                ADDth.A = bias_energy
                ADDth.ADD = ADDth.A - IOEsphereA - self.init_energy
                ADDth.ADD_IOE = ADDth.ADD  # Initial value, will be updated later
                ADDth.grad = bias_gradient
                ADDth.grad_vec = np.dot(bias_gradient.flatten(), ADDth.nADD.flatten())
                ADDth.grad_vec /= np.linalg.norm(ADDth.nADD.flatten())
                ADDth.findTSQ = False
                ADDth.ADDoptQ = False
                ADDth.ADDremoveQ = False
                
                # Add to candidate list
                candidate_ADDths.append(ADDth)
                IDnum += 1
                
                with open(self.directory + "/add_energy_list.csv", "a") as f:
                    f.write(f"{idx},{self.init_eigenvalues[idx]},{pm},{ADDth.ADD},{abs(ADDth.ADD)}\n")
        
        # Sort candidate ADDths by negative ADD value in descending order (-ADD value)
        # This prioritizes more negative (favorable) paths first
        candidate_ADDths.sort(key=lambda x: -x.ADD, reverse=True)
        
        # Select only the top n ADDs according to config.nadd
        num_add = min(self.addf_config['number_of_add'], len(candidate_ADDths))
        self.ADDths = candidate_ADDths[:num_add]
        
        # Reassign IDs to be sequential
        for i, ADDth in enumerate(self.ADDths):
            ADDth.IDnum = i
        
        print(f"### Selected top {len(self.ADDths)} ADD paths (sign-inverted ADD values, most negative first) ###")
        for ADDth in self.ADDths:
            print(f"ADD {ADDth.IDnum}: {ADDth.ADD:.8f} (-ADD = {-ADDth.ADD:.8f})")
        
        # Initialize the optimized structures dictionary
        self.optimized_structures = {}
        
        # Create directory for optimized structures
        os.makedirs(os.path.join(self.directory, "optimized_structures"), exist_ok=True)
        
        print("### ADD detection complete. ###")
        return True

    def save_optimized_structure(self, ADDth, sphere_num, IOEsphereA):
        """Save optimized structure for a specific ADD and sphere"""
        # Create directory path for ADD-specific structures
        add_dir = os.path.join(self.directory, "optimized_structures", f"ADD_{ADDth.IDnum}")
        os.makedirs(add_dir, exist_ok=True)
        
        # Create filename with radius information
        radius = np.sqrt(IOEsphereA)
        filename = f"optimized_r_{radius:.4f}.xyz"
        filepath = os.path.join(add_dir, filename)
        
        # Write XYZ file
        with open(filepath, 'w') as f:
            f.write(f"{len(self.element_list)}\n")
            f.write(f"ADD_{ADDth.IDnum} Sphere {sphere_num} Radius {radius:.4f} Energy {ADDth.ADD:.6f}\n")
            for i, (element, coord) in enumerate(zip(self.element_list, ADDth.x)):
                f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        
        # Store structure information in dictionary
        if ADDth.IDnum not in self.optimized_structures:
            self.optimized_structures[ADDth.IDnum] = []
            
        self.optimized_structures[ADDth.IDnum].append({
            'sphere': sphere_num,
            'radius': radius,
            'energy': ADDth.ADD,
            'file': filepath,
            'coords': ADDth.x.copy(),
            'comment': f"ADD_{ADDth.IDnum} Sphere {sphere_num} Radius {radius:.4f} Energy {ADDth.ADD:.6f}"
        })

    def create_separate_xyz_files(self):
        """Create separate XYZ files for each ADD path"""
        created_files = []
        
        for add_id, structures in self.optimized_structures.items():
            # Sort structures by sphere number
            structures.sort(key=lambda x: x['sphere'])
            
            # Path for the ADD-specific file
            add_trajectory_file = os.path.join(self.directory, f"ADD_{add_id}_trajectory.xyz")
            
            # Write the trajectory file
            with open(add_trajectory_file, 'w') as f:
                for structure in structures:
                    f.write(f"{len(self.element_list)}\n")
                    f.write(f"{structure['comment']}\n")
                    for i, (element, coord) in enumerate(zip(self.element_list, structure['coords'])):
                        f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
            
            created_files.append(add_trajectory_file)
            
        if created_files:
            paths_str = "\n".join(created_files)
            print(f"Created {len(created_files)} ADD trajectory files:\n{paths_str}")

    def Opt_hyper_sphere(self, ADDths, QMC, eqpoint, IOEsphereA, IOEsphereA_r, A_eq, sphereN):
        """Optimize points on the hypersphere"""
        print(f"Starting optimization on sphere {sphereN} with radius {np.sqrt(IOEsphereA):.4f}")
        
        # Reset optimization flags
        for ADDth in ADDths:
            if not ADDth.findTSQ and not ADDth.ADDremoveQ:
                ADDth.ADDoptQ = True
            else:
                ADDth.ADDoptQ = False
                
        optturnN = 0
        newADDths = []
        n_atoms = eqpoint.shape[0]
        
        # Create a directory for intermediate optimization steps
        sphere_dir = os.path.join(self.directory, "optimized_structures", f"sphere_{sphereN}")
        os.makedirs(sphere_dir, exist_ok=True)
        
        # Optimization loop
        while any(ADDth.ADDoptQ for ADDth in ADDths):
            optturnN += 1
            if optturnN >= 100:
                print(f"Optimization exceeded 100 iterations, breaking.")
                break
                
            print(f"Optimization iteration {optturnN}")
            
            # Process each ADD point in order of negative ADD value (most negative first)
            for ADDth in sorted(ADDths, key=lambda x: -x.ADD, reverse=True):
                if not ADDth.ADDoptQ or ADDth.ADDremoveQ:
                    continue
                    
                # Optimize this ADD point
                self.current_id = ADDth.IDnum
                self.current_ADD = ADDth.ADD
                
                # Starting from zero displacement
                x_initial = np.zeros(len(ADDth.thetalist))
                
                # Minimize on hypersphere using our modified steepest descent
                thetalist = self.minimizeTh_SD_SS(
                    ADDth, x_initial, 
                    lambda x: QMC.single_point(None, self.get_element_number_list(), "", 
                                            self.electric_charge_and_multiplicity, self.method, x)[0],
                    lambda x: self.calculate_gradient(QMC, x),
                    eqpoint, IOEsphereA
                )
                
                if thetalist is False:
                    ADDth.ADDremoveQ = True
                    print(f"ADD {ADDth.IDnum} optimization failed, marking for removal")
                    continue
                
                # Update ADD point with optimized position
                ADDth.thetalist = thetalist
                
                # Generate nADD in the reduced space
                ADDth.nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, self.SQ, self.dim)
                
                # Map to full coordinate space
                ADDth.nADD_full = np.zeros(n_atoms * 3)
                for i in range(min(len(ADDth.nADD_reduced), n_atoms * 3)):
                    ADDth.nADD_full[i] = ADDth.nADD_reduced[i]
                
                # Reshape to molecular geometry
                ADDth.nADD = ADDth.nADD_full.reshape(n_atoms, 3)
                
                # Calculate new coordinates
                ADDth.x = eqpoint + ADDth.nADD
                ADDth.x = self.periodicpoint(ADDth.x)
                
                # Calculate new energy
                energy, grad, _, iscalculationfailed = QMC.single_point(
                    None, self.get_element_number_list(), "", 
                    self.electric_charge_and_multiplicity, self.method, ADDth.x
                )
                
                if iscalculationfailed:
                    ADDth.ADDremoveQ = True
                    continue
                    
                BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
                _, bias_energy, bias_grad, _ = BPC.main(
                    energy, grad, ADDth.x, self.get_element_number_list(), 
                    self.config.force_data
                )
                
                ADDth.A = bias_energy
                ADDth.ADD = ADDth.A - IOEsphereA - A_eq
                
                # Calculate ADD_IOE with IOE contributions
                self.current_id = ADDth.IDnum
                self.current_ADD = ADDth.ADD
                ADDth.ADD_IOE = ADDth.ADD + self.IOE_total(ADDth.nADD_full)
                
                # Mark as optimized
                ADDth.ADDoptQ = False
                
                print(f"ADD {ADDth.IDnum} optimized: ADD={ADDth.ADD:.4f} (-ADD = {-ADDth.ADD:.4f}), ADD_IOE={ADDth.ADD_IOE:.4f}")
                print(f"Grad {np.linalg.norm(grad):.6f}")
                print(f"Energy {energy:.6f}")
                print()
                
                # Save XYZ file after each ADD optimization step
                # Use both iteration number and ADD ID in filename to ensure uniqueness
                filename = f"iteration_{optturnN}_ADD_{ADDth.IDnum}.xyz"
                filepath = os.path.join(sphere_dir, filename)
                
                # Write XYZ file for this optimization step
                with open(filepath, 'w') as f:
                    f.write(f"{len(self.element_list)}\n")
                    f.write(f"Sphere {sphereN} Iteration {optturnN} ADD_{ADDth.IDnum} Radius {np.sqrt(IOEsphereA):.4f} Energy {ADDth.ADD:.6f}\n")
                    for i, (element, coord) in enumerate(zip(self.element_list, ADDth.x)):
                        f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
                
                print(f"Saved intermediate structure to {filepath}")
                
                # Check for similar points already found
                for existing_add in newADDths:
                    if self.angle_SHS(ADDth.nADD_full, existing_add.nADD_full, self.SQ_inv) < 0.01:
                        ADDth.ADDremoveQ = True
                        print(f"ADD {ADDth.IDnum} too similar to ADD {existing_add.IDnum}, marking for removal")
                        break
                
                if not ADDth.ADDremoveQ and ADDth.ADD_IOE < 0:  # Only keep negative ADD_IOE points
                    newADDths.append(ADDth)
                    
                    # Save optimized structure for this sphere (in the regular directory structure)
                    self.save_optimized_structure(ADDth, sphereN, IOEsphereA)
            
            # Filter ADDths list
            ADDths = [ADDth for ADDth in ADDths if not ADDth.ADDremoveQ]
            
            # Sort by negative ADD value (most negative first) for next iteration
            ADDths.sort(key=lambda x: -x.ADD, reverse=True)
            
        # Sort the final ADDths list by negative ADD value (most negative first)
        newADDths.sort(key=lambda x: -x.ADD, reverse=True)
        
        return newADDths if newADDths else ADDths
    
    def add_following(self, QMC):
        """Follow ADD paths to find transition states"""
        print("### Start ADD Following. ###")
        
        IOEsphereA = self.addf_config['IOEsphereA_initial']
        IOEsphereA_r = self.addf_config['IOEsphereA_dist']
        A_eq = self.init_energy
        
        TSinitialpoints = []
        sphereN = 0
        
        # Main ADD following loop
        while sphereN < self.addf_config["step_number"]:  # Limit to prevent infinite loops
            sphereN += 1
            print(f"\n### Sphere {sphereN} with radius {np.sqrt(IOEsphereA):.4f} ###\n")
            
            # Sort ADDths by absolute ADD value (largest magnitude first)
            self.ADDths.sort(key=lambda x: abs(x.ADD), reverse=True)
            
            # Optimize on current hypersphere
            self.ADDths = self.Opt_hyper_sphere(
                self.ADDths, QMC, self.init_geometry, IOEsphereA, IOEsphereA_r, A_eq, sphereN
            )
            
            # Check for TS points and update ADD status
            for ADDth in self.ADDths:
                if ADDth.ADDremoveQ:
                    continue
                    
                # Calculate gradient at current point
                grad_x = self.calculate_gradient(QMC, ADDth.x)
                if grad_x is False:
                    ADDth.ADDremoveQ = True
                    continue
                    
                ADDth.grad = grad_x
                
                # Calculate normalized displacement vector
                normalized_nADD = ADDth.nADD.flatten() / np.linalg.norm(ADDth.nADD.flatten())
                
                # Calculate projection of gradient onto displacement vector
                ADDth.grad_vec = np.dot(ADDth.grad.flatten(), normalized_nADD)
                
                # Check if we've found a TS (gradient points downward)
                if sphereN > 5 and ADDth.grad_vec < 0.0:
                    print(f"New TS point found at ADD {ADDth.IDnum}")
                    ADDth.findTSQ = True
                    TSinitialpoints.append(ADDth.x)
                    
            # If all ADDs are done, exit
            if all(ADDth.findTSQ or ADDth.ADDremoveQ for ADDth in self.ADDths):
                print("All ADD paths complete.")
                break
                
            # Increase sphere size for next iteration
            IOEsphereA = (np.sqrt(IOEsphereA) + IOEsphereA_r) ** 2
            print(f"Expanding sphere to radius {np.sqrt(IOEsphereA):.4f}")
            
            # Save displacement vectors for debugging
            with open(os.path.join(self.directory, f"displacement_vectors_sphere_{sphereN}.csv"), "w") as f:
                f.write("ADD_ID,x,y,z,ADD,ADD_IOE\n")
                for ADDth in self.ADDths:
                    if ADDth.ADDremoveQ:
                        continue
                    for i in range(len(ADDth.nADD)):
                        f.write(f"{ADDth.IDnum},{ADDth.nADD[i][0]},{ADDth.nADD[i][1]},{ADDth.nADD[i][2]},{ADDth.ADD},{ADDth.ADD_IOE}\n")
        
        # Create separate trajectory files for each ADD path
        self.create_separate_xyz_files()
        
        # Write TS points
        if TSinitialpoints:
            print(f"Found {len(TSinitialpoints)} potential transition states.")
            self.write_ts_points(TSinitialpoints)
            
        return len(TSinitialpoints) > 0
    
    def calculate_gradient(self, QMC, x):
        """Calculate gradient at point x"""
        element_number_list = self.get_element_number_list()
        _, grad_x, _, iscalculationfailed = QMC.single_point(
            None, element_number_list, "", self.electric_charge_and_multiplicity, 
            self.method, x
        )
        
        if iscalculationfailed:
            return False
            
        # Apply bias if needed
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, _, bias_gradient, _ = BPC.main(
            0, grad_x, x, element_number_list, self.config.force_data
        )
        
        return bias_gradient
        
    def write_ts_points(self, ts_points):
        """Write TS points to file"""
        with open(f"{self.directory}/TSpoints.xyz", "w") as f:
            for i, point in enumerate(ts_points):
                f.write(f"{len(self.element_list)}\n")
                f.write(f"TS candidate {i+1}\n")
                for j, (element, coord) in enumerate(zip(self.element_list, point)):
                    f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    # Getters and setters - keep as is
    def set_molecule(self, element_list, coords):
        self.element_list = element_list
        self.coords = coords
    
    def set_gradient(self, gradient):
        self.gradient = gradient
    
    def set_hessian(self, hessian):
        self.hessian = hessian
    
    def set_energy(self, energy):
        self.energy = energy
    
    def set_coords(self, coords):
        self.coords = coords
    
    def set_element_list(self, element_list):
        self.element_list = element_list
        self.element_number_list = [element_number(i) for i in self.element_list]
    
    def set_coord(self, coord):
        self.coords = coord
    
    def get_coord(self):
        return self.coords
    
    def get_element_list(self):
        return self.element_list
    
    def get_element_number_list(self):
        if self.element_number_list is None:
            if self.element_list is None:
                raise ValueError('Element list is not set.')
            self.element_number_list = [element_number(i) for i in self.element_list]
        return self.element_number_list
    
    def set_mole_info(self, base_file_name, electric_charge_and_multiplicity):
        coord, element_list, electric_charge_and_multiplicity = xyz2list(
            base_file_name + ".xyz", electric_charge_and_multiplicity)

        if self.config.usextb != "None":
            self.method = self.config.usextb
        elif self.config.usedxtb != "None":
            self.method = self.config.usedxtb
        else:
            self.method = "None"

        self.coords = np.array(coord, dtype="float64")  
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
    
    def run(self, file_directory, SP, electric_charge_and_multiplicity, FIO_img):
        print("### Start Anharmonic Downward Distortion (ADD) method ###")
        
        # Preparation
        base_file_name = os.path.splitext(FIO_img.START_FILE)[0]
        self.set_mole_info(base_file_name, electric_charge_and_multiplicity)
        
        self.directory = make_workspace(file_directory)
        
        # Create main directory for optimized structures
        os.makedirs(os.path.join(self.directory, "optimized_structures"), exist_ok=True)
        
        # Detect initial ADD directions
        success = self.detect_add(SP)
        if not success:
            return False
        
        # Follow ADD paths to find transition states
        isConverged = self.add_following(SP)
        
        print("### ADD Following is done. ###")
        
        return isConverged
        
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
        self.addf_like_method = ADDFlikeMethod(self.config)

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
        elif self.config.use_addf:
            self.addf_like_method.run(file_directory_list[0],
                SP_list[0],
                self.config.electric_charge_and_multiplicity_list[0], 
                FIO_img_list[0])
        
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