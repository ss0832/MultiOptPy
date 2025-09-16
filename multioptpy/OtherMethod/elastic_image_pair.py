import numpy as np
import os

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Visualization.visualization import Graph
from multioptpy.Coordinate.polar_coordinate import cart2polar, polar2cart, cart_grad_2_polar_grad


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
