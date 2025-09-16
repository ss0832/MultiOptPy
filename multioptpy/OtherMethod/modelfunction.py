import numpy as np
import os

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Visualization.visualization import Graph
from multioptpy.optimizer import CalculateMoveVector
import multioptpy.ModelFunction as MF


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
