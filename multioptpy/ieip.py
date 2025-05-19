import os
import sys
import datetime
import glob
import numpy as np

from potential import BiasPotentialCalculation
from optimizer import CalculateMoveVector 
from calc_tools import Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib
from interface import force_data_parser
from polar_coordinate import cart2polar, polar2cart, cart_grad_2_polar_grad
import ModelFunction as MF

class iEIP:#based on Improved Elastic Image Pair (iEIP) method   
    def __init__(self, args):
        #Ref.: J. Chem. Theory. Comput. 2023, 19, 2410-2417
        #Ref.: J. Comput. Chem. 2018, 39, 233–251 (DS-AFIR)
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        
        self.displacement_limit = 0.04 #Bohr
        self.maximum_ieip_disp = 0.2 #Bohr
        self.L_covergence = 0.03 #Bohr
        
        self.microiterlimit = int(args.NSTEP)
        self.initial_excite_state = args.excited_state[0]
        self.final_excite_state = args.excited_state[1]
        self.excite_state_list = args.excited_state

        self.init_electric_charge_and_multiplicity = [int(args.electronic_charge[0]), int(args.spin_multiplicity[0])]
        self.final_electric_charge_and_multiplicity = [int(args.electronic_charge[1]), int(args.spin_multiplicity[1])]
        self.electric_charge_and_multiplicity_list = []
        for i in range(len(args.electronic_charge)):
            self.electric_charge_and_multiplicity_list.append([int(args.electronic_charge[i]), int(args.spin_multiplicity[i])])

        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        self.img_distance_convage_criterion = 0.15 #Bohr

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT+"/" #directory
       
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        self.usextb = args.usextb
        self.usedxtb = args.usedxtb
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        if args.pyscf:

            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET) # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            else:
                self.SUB_BASIS_SET = { "default" : self.BASIS_SET}
            
        else:#psi4
            self.SUB_BASIS_SET = "" # 
            
            
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET +="\nassign "+str(self.BASIS_SET)+"\n" # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
        
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
        
        

        self.basic_set_and_function = args.functional+"/"+args.basisset
        self.force_data = force_data_parser(args)
        if args.usextb == "None" and args.usedxtb == "None":
            self.iEIP_FOLDER_DIRECTORY = args.INPUT+"_iEIP_"+self.basic_set_and_function.replace("/","_")+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])+"/"
        else:
            if args.usedxtb != "None":
                self.iEIP_FOLDER_DIRECTORY = args.INPUT+"_iEIP_"+self.usedxtb+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])+"/"
                self.force_data["xtb"] = self.usedxtb
            
            else:
                self.iEIP_FOLDER_DIRECTORY = args.INPUT+"_iEIP_"+self.usextb+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])+"/"
                self.force_data["xtb"] = self.usextb
        
        
        self.args = args
        os.mkdir(self.iEIP_FOLDER_DIRECTORY)
        self.BETA = args.BETA
        
        self.spring_const = 1e-8
        self.microiter_num = args.microiter
        self.unrestrict = args.unrestrict
        self.mf_mode = args.model_function_mode
        self.MAX_FORCE_THRESHOLD = 0.0006 #0.0003
        self.RMS_FORCE_THRESHOLD = 0.0004 #0.0002
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0030 #0.0015 
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0020 #0.0010

        self.FC_COUNT = int(args.calc_exact_hess)
        self.dft_grid = int(args.dft_grid)
        return
        
        
    def RMS(self, mat):
        rms = np.sqrt(np.sum(mat**2))
        return rms
    
    def print_info(self, dat):

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
        print("Image_distance                         : ", self.img_distance_convage_criterion)
       
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
    
        for i in range(self.microiter_num):
            
            energy_1, gradient_1, geom_num_list_1, error_flag_1 = SP1.single_point(file_directory_1, element_list, iter, init_electric_charge_and_multiplicity, self.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, error_flag_2 = SP2.single_point(file_directory_2, element_list, iter, final_electric_charge_and_multiplicity, self.force_data["xtb"])
            
            BPC_1 = BiasPotentialCalculation(self.iEIP_FOLDER_DIRECTORY)
            BPC_2 = BiasPotentialCalculation(self.iEIP_FOLDER_DIRECTORY)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.force_data)
        
            if error_flag_1 or error_flag_2:
                print("Error in QM calculation.")
                with open(self.iEIP_FOLDER_DIRECTORY+"end.txt", "w") as f:
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
            new_geom_num_list_1_tolist = (geom_num_list_1*self.bohr2angstroms).tolist()
            new_geom_num_list_2_tolist = (geom_num_list_2*self.bohr2angstroms).tolist()
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
        G = Graph(self.iEIP_FOLDER_DIRECTORY)
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
        for iter in range(0, self.microiterlimit):
            if os.path.isfile(self.iEIP_FOLDER_DIRECTORY+"end.txt"):
                break
            print("# ITR. "+str(iter))
            
            energy_1, gradient_1, geom_num_list_1, error_flag_1 = SP1.single_point(file_directory_1, element_list, iter, init_electric_charge_and_multiplicity, self.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, error_flag_2 = SP2.single_point(file_directory_2, element_list, iter, final_electric_charge_and_multiplicity, self.force_data["xtb"])
            geom_num_list_1, geom_num_list_2 = Calculationtools().kabsch_algorithm(geom_num_list_1, geom_num_list_2)
            
            if error_flag_1 or error_flag_2:
                print("Error in QM calculation.")
                with open(self.iEIP_FOLDER_DIRECTORY+"end.txt", "w") as f:
                    f.write("Error in QM calculation.")
                break
            
            if iter == 0:
                m_1 = gradient_1 * 0.0
                m_2 = gradient_1 * 0.0
                v_1 = gradient_1 * 0.0
                v_2 = gradient_1 * 0.0
                ini_geom_1 = geom_num_list_1
                ini_geom_2 = geom_num_list_2
            
            BPC_1 = BiasPotentialCalculation(self.iEIP_FOLDER_DIRECTORY)
            BPC_2 = BiasPotentialCalculation(self.iEIP_FOLDER_DIRECTORY)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.force_data)
        
            if self.microiter_num > 0 and iter > 0:
                energy_1, gradient_1, bias_energy_1, bias_gradient_1, geom_num_list_1, energy_2, gradient_2, bias_energy_2, bias_gradient_2, geom_num_list_2 = self.microiteration(SP1, SP2, FIO1, FIO2, file_directory_1, file_directory_2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter)
                if os.path.isfile(self.iEIP_FOLDER_DIRECTORY+"end.txt"):
                    break
            
        
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
                #based on DS-AFIR method
                Z_1 = np.linalg.norm(geom_num_list_1 - ini_geom_1) / np.linalg.norm(geom_num_list_1 - geom_num_list_2) + (np.sum( (geom_num_list_1 - ini_geom_1) * (geom_num_list_1 - geom_num_list_2))) / (np.linalg.norm(geom_num_list_1 - ini_geom_1) * np.linalg.norm(geom_num_list_1 - geom_num_list_2)) 
                Z_2 = np.linalg.norm(geom_num_list_2 - ini_geom_2) / np.linalg.norm(geom_num_list_2 - geom_num_list_1) + (np.sum( (geom_num_list_2 - ini_geom_2) * (geom_num_list_2 - geom_num_list_1))) / (np.linalg.norm(geom_num_list_2 - ini_geom_2) * np.linalg.norm(geom_num_list_2 - geom_num_list_1))
                
                if Z_1 > 0.0:
                    Y_1 = Z_1 /(Z_1 + 1) + 0.5
                else:
                    Y_1 = 0.5
                
                if Z_2 > 0.0:
                    Y_2 = Z_2 /(Z_2 + 1) + 0.5
                else:
                    Y_2 = 0.5
                
                u_1 = Y_1 * ((geom_num_list_1 - geom_num_list_2) / np.linalg.norm(geom_num_list_1 - geom_num_list_2)) - (1.0 - Y_1) * ((geom_num_list_1 - ini_geom_1) / np.linalg.norm(geom_num_list_1 - ini_geom_1))  
                u_2 = Y_2 * ((geom_num_list_2 - geom_num_list_1) / np.linalg.norm(geom_num_list_2 - geom_num_list_1)) - (1.0 - Y_2) * ((geom_num_list_2 - ini_geom_2) / np.linalg.norm(geom_num_list_2 - ini_geom_2)) 
                
                X_1 = self.BETA / np.linalg.norm(u_1) - (np.sum(gradient_1 * u_1) / np.linalg.norm(u_1) ** 2)
                X_2 = self.BETA / np.linalg.norm(u_2) - (np.sum(gradient_2 * u_2) / np.linalg.norm(u_2) ** 2)
               
                ini_disp_1 *= X_1 * (1.0 - Y_1)
                ini_disp_2 *= X_2 * (1.0 - Y_2)
                
             
                close_target_disp_1 = close_target_disp * X_1 * Y_1
                close_target_disp_2 = close_target_disp * X_2 * Y_2
                
                
       
       
            total_disp_1 = - perp_disp_1 + delta_energy_disp_1 + close_target_disp_1 - force_disp_1 + ini_disp_1
            total_disp_2 = - perp_disp_2 - delta_energy_disp_2 - close_target_disp_2 - force_disp_2 + ini_disp_2
            
            #AdaBelief: https://doi.org/10.48550/arXiv.2010.07468

            m_1 = beta_m*m_1 + (1-beta_m)*total_disp_1
            m_2 = beta_m*m_2 + (1-beta_m)*total_disp_2
            v_1 = beta_v*v_1 + (1-beta_v)*(total_disp_1-m_1)**2
            v_2 = beta_v*v_2 + (1-beta_v)*(total_disp_2-m_2)**2
            

            
            adabelief_1 = 0.01*(m_1 / (np.sqrt(v_1) + 1e-8))
            adabelief_2 = 0.01*(m_2 / (np.sqrt(v_2) + 1e-8))

            
            new_geom_num_list_1 = geom_num_list_1 + adabelief_1
            new_geom_num_list_2 = geom_num_list_2 + adabelief_2
            
            new_geom_num_list_1, new_geom_num_list_2 = Calculationtools().kabsch_algorithm(new_geom_num_list_1, new_geom_num_list_2)
    
            if iter != 0:
                prev_delta_geometry = delta_geometry
            
            delta_geometry = np.linalg.norm(new_geom_num_list_2 - new_geom_num_list_1)
            rms_perp_force = np.linalg.norm(np.sqrt(perp_force_1 ** 2 + perp_force_2 ** 2))
            
            info_dat = {"perp_force_1": perp_force_1, "perp_force_2": perp_force_2, "delta_energy_force_1": delta_energy_force_1, "delta_energy_force_2": delta_energy_force_2,"close_target_force": close_target_force, "perp_disp_1": perp_disp_1 ,"perp_disp_2": perp_disp_2,"delta_energy_disp_1": delta_energy_disp_1,"delta_energy_disp_2": delta_energy_disp_2,"close_target_disp": close_target_disp, "total_disp_1": total_disp_1, "total_disp_2": total_disp_2, "bias_energy_1": bias_energy_1,"bias_energy_2": bias_energy_2,"bias_gradient_1":bias_gradient_1,"bias_gradient_2":bias_gradient_2,"energy_1": energy_1,"energy_2": energy_2,"gradient_1": gradient_1,"gradient_2": gradient_2, "delta_geometry":delta_geometry, "rms_perp_force":rms_perp_force }
            
            self.print_info(info_dat)
            
            new_geom_num_list_1_tolist = (new_geom_num_list_1*self.bohr2angstroms).tolist()
            new_geom_num_list_2_tolist = (new_geom_num_list_2*self.bohr2angstroms).tolist()
            for i, elem in enumerate(element_list):
                new_geom_num_list_1_tolist[i].insert(0, elem)
                new_geom_num_list_2_tolist[i].insert(0, elem)
           
            
            
            new_geom_num_list_1_tolist.insert(0, init_electric_charge_and_multiplicity)
            new_geom_num_list_2_tolist.insert(0, final_electric_charge_and_multiplicity)
                
            file_directory_1 = FIO1.make_psi4_input_file([new_geom_num_list_1_tolist], iter+1)
            file_directory_2 = FIO2.make_psi4_input_file([new_geom_num_list_2_tolist], iter+1)
            
            BIAS_ENERGY_LIST_A.append(bias_energy_1*self.hartree2kcalmol)
            BIAS_ENERGY_LIST_B.append(bias_energy_2*self.hartree2kcalmol)
            BIAS_GRAD_LIST_A.append(np.sqrt(np.sum(bias_gradient_1**2)))
            BIAS_GRAD_LIST_B.append(np.sqrt(np.sum(bias_gradient_2**2)))
            
            ENERGY_LIST_A.append(energy_1*self.hartree2kcalmol)
            ENERGY_LIST_B.append(energy_2*self.hartree2kcalmol)
            GRAD_LIST_A.append(np.sqrt(np.sum(gradient_1**2)))
            GRAD_LIST_B.append(np.sqrt(np.sum(gradient_2**2)))
            
            prev_geom_num_list_1 = geom_num_list_1
            prev_geom_num_list_2 = geom_num_list_2
            
            if delta_geometry < self.img_distance_convage_criterion:#Bohr
                print("Converged!!!")
                break
            
            if delta_geometry > prev_delta_geometry:
                self.BETA *= 1.02
        else:
            print("Reached maximum number of iterations. This is not converged.")
            with open(self.iEIP_FOLDER_DIRECTORY+"not_converged.txt", "w") as f:
                f.write("Reached maximum number of iterations. This is not converged.")
        
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
        
        FIO1.argrelextrema_txt_save(ene_list, "approx_TS", "max")
        FIO1.argrelextrema_txt_save(ene_list, "approx_EQ", "min")
        FIO1.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        return
    
    def norm_dist_2imgs(self, geom_num_list_1, geom_num_list_2):
        L = self.dist_2imgs(geom_num_list_1, geom_num_list_2)
        N = (geom_num_list_2 - geom_num_list_1) / L
        return N 
    
    def dist_2imgs(self, geom_num_list_1, geom_num_list_2):
        L = np.linalg.norm(geom_num_list_2 - geom_num_list_1) + 1e-10
        return L #Bohr
   
    def target_dist_2imgs(self, L):
        Lt = max(L * 0.9, self.L_covergence - 0.01)       
    
        return Lt
    
    def force_R(self, L):
        F_R = min(max(L/self.L_covergence, 1)) * self.F_R_convage_criterion
        return F_R  

    def displacement(self, force):
        n_force = np.linalg.norm(force)
        displacement = (force / (n_force + 1e-10)) * min(n_force, self.displacement_limit)
        return displacement
    
    def displacement_prime(self, force):
        n_force = np.linalg.norm(force)
        displacement = (force / (n_force + 1e-10)) * self.displacement_limit 
        return displacement
    
    def initial_structure_dependent_force(self, geom, ini_geom):
        ini_force = geom - ini_geom
        return ini_force
        
    
    def perpendicular_force(self, gradient, N):#gradient and N (atomnum×3, ndarray)
        perp_force = gradient.reshape(len(gradient)*3, 1) - np.dot(gradient.reshape(1, len(gradient)*3), N.reshape(len(gradient)*3, 1)) * N.reshape(len(gradient)*3, 1)
        return perp_force.reshape(len(gradient), 3) #(atomnum×3, ndarray)
        
    
    def delta_energy_force(self, ene_1, ene_2, N, L):
        d_ene_force = N * abs(ene_1 - ene_2) / L
        return d_ene_force
    
    
    def close_target_force(self, L, Lt, geom_num_list_1, geom_num_list_2):
        ct_force = (geom_num_list_2 - geom_num_list_1) * (L - Lt) / L
        return ct_force
    
    
    
    def optimize(self):
        if self.args.pyscf:
            from pyscf_calculation_tools import Calculation
        elif self.args.usextb != "None" and self.args.usedxtb == "None":
            from tblite_calculation_tools import Calculation
        elif self.args.usedxtb != "None" and self.args.usextb == "None":
            from dxtb_calculation_tools import Calculation
        else:
            from psi4_calculation_tools import Calculation

        
        
        file_path_list = glob.glob(self.START_FILE+"*_[A-Z].xyz")
        FIO_img_list = []

        for file_path in file_path_list:
            FIO_img_list.append(FileIO(self.iEIP_FOLDER_DIRECTORY, file_path))

        geometry_list_list = []
        element_list_list = []
        electric_charge_and_multiplicity_list = []

        for i in range(len(FIO_img_list)):
        
            geometry_list, element_list, electric_charge_and_multiplicity = FIO_img_list[i].make_geometry_list(self.electric_charge_and_multiplicity_list[i])
            
            if self.args.pyscf:
                electric_charge_and_multiplicity = [self.electronic_charge[i], self.spin_multiplicity[i]]
            
            geometry_list_list.append(geometry_list)
            element_list_list.append(element_list)
            electric_charge_and_multiplicity_list.append(electric_charge_and_multiplicity)
        self.save_input_data()
        SP_list = []
        file_directory_list = []
        for i in range(len(FIO_img_list)):
            SP_list.append(Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye(3*len(geometry_list_list[i])),
                         unrestrict=self.unrestrict, 
                         BASIS_SET = self.BASIS_SET,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         electronic_charge = self.electronic_charge[i] or electric_charge_and_multiplicity_list[i][0],
                         spin_multiplicity = self.spin_multiplicity[i] or electric_charge_and_multiplicity_list[i][1],
                         excited_state = self.excite_state_list[i],
                         dft_grid=self.dft_grid,
                         ECP = self.ECP))
            
            SP_list[i].cpcm_solv_model = self.cpcm_solv_model
            SP_list[i].alpb_solv_model = self.alpb_solv_model
            
            file_directory = FIO_img_list[i].make_psi4_input_file(geometry_list_list[i], 0)
            file_directory_list.append(file_directory)
       
       
        
        if self.mf_mode != "None":
            self.model_function_optimization(file_directory_list, SP_list, element_list_list, self.electric_charge_and_multiplicity_list, FIO_img_list)
        else:
            self.iteration(file_directory_list[0], file_directory_list[1], SP_list[0], SP_list[1], element_list, self.electric_charge_and_multiplicity_list[0], self.electric_charge_and_multiplicity_list[1], FIO_img_list[0], FIO_img_list[1])
        
        return
        

    def save_input_data(self):
        with open(self.iEIP_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        return
        
    def model_function_optimization(self, file_directory_list, SP_list, element_list_list, electric_charge_and_multiplicity_list, FIO_img_list):
        
        G = Graph(self.iEIP_FOLDER_DIRECTORY)
        BIAS_GRAD_LIST_LIST = [[] for i in range(len(SP_list))]
        BIAS_MF_GRAD_LIST = [[] for i in range(len(SP_list))]
        BIAS_ENERGY_LIST_LIST = [[] for i in range(len(SP_list))]
        BIAS_MF_ENERGY_LIST = []
        GRAD_LIST_LIST = [[] for i in range(len(SP_list))]
        MF_GRAD_LIST = [[] for i in range(len(SP_list))]
        ENERGY_LIST_LIST = [[] for i in range(len(SP_list))]
        MF_ENERGY_LIST = []

        for iter in range(0, self.microiterlimit):
            if os.path.isfile(self.iEIP_FOLDER_DIRECTORY+"end.txt"):
                break
            print("# ITR. "+str(iter))
            
            tmp_gradient_list = []
            tmp_energy_list = []
            tmp_geometry_list = []
            exit_flag = False
            for j in range(len(SP_list)):
                energy, gradient, geom_num_list, exit_flag = SP_list[j].single_point(file_directory_list[j], element_list_list[j], iter, electric_charge_and_multiplicity_list[j], self.force_data["xtb"])
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
                    
                optimizer_instances = CMV.initialization(self.force_data["opt_method"])
                for i in range(len(optimizer_instances)):

                    optimizer_instances[i].set_hessian(np.eye((len(geom_num_list)*3)))
                    
                init_geom_list = tmp_geometry_list
                PREV_GEOM_LIST = tmp_geometry_list
                if self.mf_mode == "seam":
                    SMF = MF.SeamModelFunction()
                    
                elif self.mf_mode == "avoiding":
                    AMF = MF.AvoidingModelFunction()

                elif self.mf_mode == "conical":
                    CMF = MF.ConicalModelFunction()
                elif self.mf_mode == "mesx2":
                    MESX = MF.OptMESX2()
                
                elif self.mf_mode == "mesx":
                    MESX = MF.OptMESX()
                
                elif self.mf_mode == "meci":
                    MECI_bare = MF.OptMECI()
                    MECI_bias = MF.OptMECI()

                else:
                    print("Unexpected method. exit...")
                    raise
            
                 

            BPC_LIST = []
            for j in range(len(SP_list)):
                BPC_LIST.append(BiasPotentialCalculation(self.iEIP_FOLDER_DIRECTORY))
                
            tmp_bias_energy_list = []
            tmp_bias_gradient_list = []
            tmp_bias_hessian_list = []
            for j in range(len(SP_list)):
                
                _, bias_energy, bias_gradient, BPA_hessian = BPC_LIST[j].main(tmp_energy_list[j], tmp_gradient_list[j], tmp_geometry_list[j], element_list_list[j], self.force_data)
                
                for l in range(len(optimizer_instances)):
                    optimizer_instances[l].set_bias_hessian(BPA_hessian)
                tmp_bias_hessian_list.append(BPA_hessian)
                tmp_bias_energy_list.append(bias_energy)
                tmp_bias_gradient_list.append(bias_gradient)
 
            tmp_bias_energy_list = np.array(tmp_bias_energy_list)
            tmp_bias_gradient_list = np.array(tmp_bias_gradient_list)
 
            ##-----
            ##  model function
            ##-----
            if self.mf_mode == "seam":
                mf_energy = SMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = SMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = SMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1])
                smf_bias_grad_1, smf_bias_grad_2 = SMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                if iter % self.FC_COUNT == 0 and self.FC_COUNT > 0:
                    hess_list = []
                    for l in range(len(SP_list)):
                        tmp_hess = 0.5 * (SP_list[l].Model_hess + SP_list[l].Model_hess.T)
                        hess_list.append(tmp_hess)
                    gp_hess = SMF.calc_hess(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1], hess_list[0], hess_list[1])
                    
                    for l in range(len(optimizer_instances)):
                        optimizer_instances[l].set_hessian(gp_hess)
                   
                bias_gp_hess = SMF.calc_hess(tmp_bias_energy_list[0] - tmp_energy_list[0], tmp_bias_energy_list[1] - tmp_energy_list[1], tmp_bias_gradient_list[0] - tmp_gradient_list[0], tmp_bias_gradient_list[1] - tmp_gradient_list[1], tmp_bias_hessian_list[0], tmp_bias_hessian_list[1])
                for l in range(len(optimizer_instances)):
                    optimizer_instances[l].set_bias_hessian(bias_gp_hess)

                
            elif self.mf_mode == "avoiding":
                mf_energy = AMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = AMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = AMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1])
                smf_bias_grad_1, smf_bias_grad_2 = AMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                
                
                if iter % self.FC_COUNT == 0 and self.FC_COUNT > 0:
                    raise NotImplementedError("Not implemented Hessian of AMF.")



            elif self.mf_mode == "conical":
                mf_energy = CMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = CMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = CMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1])
                smf_bias_grad_1, smf_bias_grad_2 = CMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                
                if iter % self.FC_COUNT == 0 and self.FC_COUNT > 0:
                    raise NotImplementedError("Not implemented Hessian of CMF.")


            elif self.mf_mode == "mesx" or self.mf_mode == "mesx2":
                mf_energy = MESX.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = MESX.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                gp_grad = MESX.calc_grad(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1])
                gp_bias_grad = MESX.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [gp_bias_grad, gp_bias_grad]
                tmp_smf_grad_list = [gp_grad, gp_grad]
                if iter % self.FC_COUNT == 0 and self.FC_COUNT > 0:
                    hess_list = []
                    for l in range(len(SP_list)):
                        tmp_hess = 0.5 * (SP_list[l].Model_hess + SP_list[l].Model_hess.T)
                        hess_list.append(tmp_hess)
                    gp_hess = MESX.calc_hess(tmp_gradient_list[0], tmp_gradient_list[1], hess_list[0], hess_list[1])
                    
                    for l in range(len(optimizer_instances)):
                        optimizer_instances[l].set_hessian(gp_hess)
                   
             
                       
                    
            elif self.mf_mode == "meci":
                mf_energy = MECI_bare.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = MECI_bias.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                gp_grad = MECI_bare.calc_grad(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1])
                gp_bias_grad = MECI_bias.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [gp_bias_grad, gp_bias_grad]
                tmp_smf_grad_list = [gp_grad, gp_grad]
                if iter % self.FC_COUNT == 0 and self.FC_COUNT > 0:
                    hess_list = []
                    for l in range(len(SP_list)):
                        tmp_hess = 0.5 * (SP_list[l].Model_hess + SP_list[l].Model_hess.T)
                        hess_list.append(tmp_hess)
                    gp_hess = MECI_bare.calc_hess(tmp_gradient_list[0], tmp_gradient_list[1], hess_list[0], hess_list[1])
                    
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
                
            _, tmp_move_vector, _ = CMV.calc_move_vector(iter, tmp_geometry_list[0], tmp_smf_bias_grad_list[0], PREV_MF_BIAS_GRAD_LIST[0], PREV_GEOM_LIST[0], PREV_MF_e, PREV_MF_B_e, PREV_MOVE_VEC_LIST[0], init_geom_list[0], tmp_smf_grad_list[0], PREV_GRAD_LIST[0], optimizer_instances)
            
            for j in range(len(SP_list)):
                tmp_move_vector_list.append(tmp_move_vector)
                tmp_new_geometry_list.append((tmp_geometry_list[j]-tmp_move_vector)*self.bohr2angstroms)
                        
            tmp_move_vector_list = np.array(tmp_move_vector_list)
            tmp_new_geometry_list = np.array(tmp_new_geometry_list)

            for j in range(len(SP_list)):
                tmp_new_geometry_list[j] -= Calculationtools().calc_center_of_mass(tmp_new_geometry_list[j], element_list_list[j])
                tmp_new_geometry_list[j], _ = Calculationtools().kabsch_algorithm(tmp_new_geometry_list[j], PREV_GEOM_LIST[j])
                

    
            tmp_new_geometry_list_to_list =  tmp_new_geometry_list.tolist()
            
            for j in range(len(SP_list)):
                for i, elem in enumerate(element_list_list[j]):
                    tmp_new_geometry_list_to_list[j][i].insert(0, elem)
                
          
            for j in range(len(SP_list)):
                tmp_new_geometry_list_to_list[j].insert(0, electric_charge_and_multiplicity_list[j])
                
            for j in range(len(SP_list)):
                print(f"Input: {j}")
                _ = FIO_img_list[j].print_geometry_list(tmp_new_geometry_list[j], element_list_list[j], [])
                file_directory_list[j] = FIO_img_list[j].make_psi4_input_file([tmp_new_geometry_list_to_list[j]], iter+1)
                print()
              
            PREV_GRAD_LIST = tmp_gradient_list
            PREV_BIAS_GRAD_LIST = tmp_bias_gradient_list
            PREV_MOVE_VEC_LIST = tmp_move_vector_list
            PREV_GEOM_LIST = tmp_new_geometry_list

            PREV_MF_BIAS_GRAD_LIST = tmp_bias_gradient_list
            PREV_MF_GRAD_LIST = tmp_smf_grad_list
            PREV_B_e_LIST = tmp_bias_energy_list
            PREV_e_LIST = tmp_energy_list
            
            
            BIAS_MF_ENERGY_LIST.append(mf_bias_energy)
            MF_ENERGY_LIST.append(mf_energy)
            for j in range(len(SP_list)):
                BIAS_GRAD_LIST_LIST[j].append(np.sqrt(np.sum(tmp_bias_gradient_list[j]**2)))
                BIAS_ENERGY_LIST_LIST[j].append(tmp_bias_energy_list[j])
                GRAD_LIST_LIST[j].append(np.sqrt(np.sum(tmp_gradient_list[j]**2)))
                ENERGY_LIST_LIST[j].append(tmp_energy_list[j])
                MF_GRAD_LIST[j].append(np.sqrt(np.sum(tmp_smf_grad_list[j]**2)))
                BIAS_MF_GRAD_LIST[j].append(np.sqrt(np.sum(tmp_smf_bias_grad_list[j]**2)))
            
            self.print_info_for_model_func(self.force_data["opt_method"], mf_energy, mf_bias_energy, tmp_smf_bias_grad_list, tmp_move_vector_list, PREV_MF_e, PREV_MF_B_e)
            
            PREV_MF_e = mf_energy
            PREV_MF_B_e = mf_bias_energy
            converge_check_flag, _, _ = self.check_converge_criteria(tmp_smf_bias_grad_list, tmp_move_vector_list)
            if converge_check_flag:#convergent criteria
                print("Converged!!!")
                break




        NUM_LIST = [i for i in range(len(BIAS_MF_ENERGY_LIST))]
        MF_ENERGY_LIST = np.array(MF_ENERGY_LIST)
        BIAS_MF_ENERGY_LIST = np.array(BIAS_MF_ENERGY_LIST)
        ENERGY_LIST_LIST = np.array(ENERGY_LIST_LIST)
        GRAD_LIST_LIST = np.array(GRAD_LIST_LIST)
        BIAS_ENERGY_LIST_LIST = np.array(BIAS_ENERGY_LIST_LIST)
        BIAS_GRAD_LIST_LIST = np.array(BIAS_GRAD_LIST_LIST)
        MF_GRAD_LIST = np.array(MF_GRAD_LIST)
        BIAS_MF_GRAD_LIST = np.array(BIAS_MF_GRAD_LIST)
        
        
        G.single_plot(NUM_LIST, MF_ENERGY_LIST*self.hartree2kcalmol, file_directory_list[j], "model_function_energy", axis_name_2="energy [kcal/mol]", name="model_function_energy")   
        G.single_plot(NUM_LIST, BIAS_MF_ENERGY_LIST*self.hartree2kcalmol, file_directory_list[j], "model_function_bias_energy", axis_name_2="energy [kcal/mol]", name="model_function_bias_energy")   
        G.double_plot(NUM_LIST, MF_ENERGY_LIST*self.hartree2kcalmol, BIAS_MF_ENERGY_LIST*self.hartree2kcalmol, add_file_name="model_function_energy")
        
        with open(self.iEIP_FOLDER_DIRECTORY+"model_function_energy_"+str(j+1)+".csv", "w") as f:
            for k in range(len(NUM_LIST)):
                f.write(str(NUM_LIST[k])+","+str(MF_ENERGY_LIST[k])+"\n") 
        with open(self.iEIP_FOLDER_DIRECTORY+"model_function_bias_energy_"+str(j+1)+".csv", "w") as f:
            for k in range(len(NUM_LIST)):
                f.write(str(NUM_LIST[k])+","+str(BIAS_MF_ENERGY_LIST[k])+"\n")
        
        
        for j in range(len(SP_list)):
            G.single_plot(NUM_LIST, ENERGY_LIST_LIST[j]*self.hartree2kcalmol, file_directory_list[j], "energy_"+str(j+1), axis_name_2="energy [kcal/mol]", name="energy_"+str(j+1))   
            G.single_plot(NUM_LIST, GRAD_LIST_LIST[j], file_directory_list[j], "gradient_"+str(j+1), axis_name_2="grad (RMS) [a.u.]", name="gradient_"+str(j+1))
            G.single_plot(NUM_LIST, BIAS_ENERGY_LIST_LIST[j]*self.hartree2kcalmol, file_directory_list[j], "bias_energy_"+str(j+1), axis_name_2="energy [kcal/mol]", name="bias_energy_"+str(j+1))   
            G.single_plot(NUM_LIST, BIAS_GRAD_LIST_LIST[j], file_directory_list[j], "bias_gradient_"+str(j+1), axis_name_2="grad (RMS) [a.u.]", name="bias_gradient_"+str(j+1))
            G.single_plot(NUM_LIST, MF_GRAD_LIST[j], file_directory_list[j], "model_func_gradient_"+str(j+1), axis_name_2="grad (RMS) [a.u.]", name="model_func_gradient_"+str(j+1))
            G.single_plot(NUM_LIST, BIAS_MF_GRAD_LIST[j], file_directory_list[j], "model_func_bias_gradient_"+str(j+1), axis_name_2="grad (RMS) [a.u.]", name="model_func_bias_gradient_"+str(j+1))
            
            with open(self.iEIP_FOLDER_DIRECTORY+"energy_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(ENERGY_LIST_LIST[j][k])+"\n")
            with open(self.iEIP_FOLDER_DIRECTORY+"gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(GRAD_LIST_LIST[j][k])+"\n")
            with open(self.iEIP_FOLDER_DIRECTORY+"bias_energy_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(BIAS_ENERGY_LIST_LIST[j][k])+"\n")
            with open(self.iEIP_FOLDER_DIRECTORY+"bias_gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(BIAS_GRAD_LIST_LIST[j][k])+"\n")
            with open(self.iEIP_FOLDER_DIRECTORY+"model_func_gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(MF_GRAD_LIST[j][k])+"\n")
            with open(self.iEIP_FOLDER_DIRECTORY+"model_func_bias_gradient_"+str(j+1)+".csv", "w") as f:
                for k in range(len(NUM_LIST)):
                    f.write(str(NUM_LIST[k])+","+str(BIAS_MF_GRAD_LIST[j][k])+"\n")
                    
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for j in range(len(SP_list)):
            FIO_img_list[j].argrelextrema_txt_save(ENERGY_LIST_LIST[j], "approx_TS_"+str(j+1), "max")
            FIO_img_list[j].argrelextrema_txt_save(ENERGY_LIST_LIST[j], "approx_EQ_"+str(j+1), "min")
            FIO_img_list[j].argrelextrema_txt_save(GRAD_LIST_LIST[j], "local_min_grad_"+str(j+1), "min")
            
            FIO_img_list[j].make_traj_file(name=alphabet[j])
        
        return
    
    def print_info_for_model_func(self, optmethod, e, B_e, B_g, displacement_vector, pre_e, pre_B_e):
        print("caluculation results (unit a.u.):")
        print("OPT method            : {} ".format(optmethod))
        print("                         Value                         Threshold ")
        print("ENERGY                : {:>15.12f} ".format(e))
        print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
        print("Maximum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
        print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt((B_g**2).mean())), self.RMS_FORCE_THRESHOLD))
        print("Maximum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), self.MAX_DISPLACEMENT_THRESHOLD))
        print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt((displacement_vector**2).mean())), self.RMS_DISPLACEMENT_THRESHOLD))
        print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
        print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
        return
    
    def check_converge_criteria(self, B_g, displacement_vector):
        max_force = abs(B_g.max())
        max_force_threshold = self.MAX_FORCE_THRESHOLD
        rms_force = abs(np.sqrt((B_g**2).mean()))
        rms_force_threshold = self.RMS_FORCE_THRESHOLD

        max_displacement = abs(displacement_vector.max())
        max_displacement_threshold = self.MAX_DISPLACEMENT_THRESHOLD
        rms_displacement = abs(np.sqrt((displacement_vector**2).mean()))
        rms_displacement_threshold = self.RMS_DISPLACEMENT_THRESHOLD
        if max_force < max_force_threshold and rms_force < rms_force_threshold and max_displacement < max_displacement_threshold and rms_displacement < rms_displacement_threshold:#convergent criteria
            return True, max_displacement_threshold, rms_displacement_threshold
       
        return False, max_displacement_threshold, rms_displacement_threshold
    
    def run(self):
        self.optimize()
        print("completed...")
        return
    
        
    