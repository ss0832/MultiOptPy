import os
import sys
import time
import glob
import copy
import torch
import numpy as np

from potential import BiasPotentialCalculation
from optimizer import CalculateMoveVector 
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib, element_number
from interface import force_data_parser
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


        #self.force_perpendicularity_convage_criterion = 0.008 #Hartree/Bohr
        self.img_distance_convage_criterion = 0.15 #Bohr
        #self.F_R_convage_criterion = 0.012
        #self.DELTA = float(args.DELTA) # 

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT+"/" #directory
        #self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        self.usextb = args.usextb
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        
        if args.pyscf:
            self.electronic_charge = args.electronic_charge
            self.spin_multiplicity = args.spin_multiplicity
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
            

        self.basic_set_and_function = args.functional+"/"+args.basisset
        
        if args.usextb == "None":
            self.iEIP_FOLDER_DIRECTORY = args.INPUT+"_iEIP_"+self.basic_set_and_function.replace("/","_")+"_"+str(time.time())+"/"
        else:
            self.iEIP_FOLDER_DIRECTORY = args.INPUT+"_iEIP_"+self.usextb+"_"+str(time.time())+"/"
        self.args = args
        os.mkdir(self.iEIP_FOLDER_DIRECTORY)
        self.BETA = args.BETA
        self.force_data = force_data_parser(args)
        self.spring_const = 1e-8
        self.microiter_num = args.microiter
        self.unrestrict = args.unrestrict
        self.mf_mode = args.model_function_mode
        self.MAX_FORCE_THRESHOLD = 0.0003 #0.0003
        self.RMS_FORCE_THRESHOLD = 0.0002 #0.0002
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0015 #0.0015 
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0010 #0.0010
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

    def microiteration(self, SP1, SP2, FIO1, FIO2, file_directory_1, file_directory_2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter):
        #Add force to minimize potential along MEP. (based on nudged elestic bond method)
 
        for i in range(self.microiter_num):
            print("# Microiteration "+str(i))
            energy_1, gradient_1, geom_num_list_1, _ = SP1.single_point(file_directory_1, element_list, iter, init_electric_charge_and_multiplicity, self.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, _ = SP2.single_point(file_directory_2, element_list, iter, final_electric_charge_and_multiplicity, self.force_data["xtb"])
            
            BPC_1 = BiasPotentialCalculation(SP1.Model_hess, SP1.FC_COUNT, self.iEIP_FOLDER_DIRECTORY)
            BPC_2 = BiasPotentialCalculation(SP2.Model_hess, SP2.FC_COUNT, self.iEIP_FOLDER_DIRECTORY)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.force_data)
            

            
            N_1 = self.norm_dist_2imgs(geom_num_list_1, prev_geom_num_list_1)
            N_2 = self.norm_dist_2imgs(geom_num_list_2, prev_geom_num_list_2)
            L_1 = self.dist_2imgs(geom_num_list_1, prev_geom_num_list_1)
            L_2 = self.dist_2imgs(geom_num_list_2, prev_geom_num_list_2)
            perp_force_1 = self.perpendicular_force(bias_gradient_1, N_1)
            perp_force_2 = self.perpendicular_force(bias_gradient_2, N_2)
            
            paral_force_1 = self.spring_const * N_1
            paral_force_2 = self.spring_const * N_2
            
            print("Energy 1                 :", energy_1)
            print("Energy 2                 :", energy_2)
            print("Energy 1  (bias)         :", bias_energy_1)
            print("Energy 2  (bias)         :", bias_energy_2)
            print("RMS perpendicular force 1:", self.RMS(perp_force_1))
            print("RMS perpendicular force 2:", self.RMS(perp_force_2))
            
            total_force_1 = perp_force_1 + paral_force_1
            total_force_2 = perp_force_2 + paral_force_2
            
            #cg method
            if i > 0:
                alpha_1 = np.dot(total_force_1.reshape(1, len(total_force_1)*3), (d_1).reshape(len(total_force_1)*3, 1)) / np.dot(d_1.reshape(1, len(total_force_1)*3), d_1.reshape(len(total_force_1)*3, 1))
                total_disp_1 = alpha_1 * d_1
                beta_1 = np.dot(total_force_1.reshape(1, len(total_force_1)*3), (total_force_1 - prev_total_force_1).reshape(len(total_force_1)*3, 1)) / np.dot(d_1.reshape(1, len(total_force_1)*3), (total_force_1 - prev_total_force_1).reshape(len(total_force_1)*3, 1))#Hestenes-stiefel
                d_1 = copy.copy(-1 * total_force_1 + abs(beta_1) * d_1)
                
                alpha_2 = np.dot(total_force_2.reshape(1, len(total_force_2)*3), (d_2).reshape(len(total_force_2)*3, 1)) / np.dot(d_2.reshape(1, len(total_force_2)*3), d_2.reshape(len(total_force_2)*3, 1))
                total_disp_2 = alpha_2 * d_2
                beta_2 = np.dot(total_force_2.reshape(1, len(total_force_2)*3), (total_force_2 - prev_total_force_2).reshape(len(total_force_2)*3, 1)) / np.dot(d_2.reshape(1, len(total_force_2)*3), (total_force_2 - prev_total_force_2).reshape(len(total_force_2)*3, 1))#Hestenes-stiefel
                d_2 = copy.copy(-1 * total_force_2 + abs(beta_2) * d_2)  
            else:   
                d_1 = total_force_1
                d_2 = total_force_2
                perp_disp_1 = self.displacement(perp_force_1)
                perp_disp_2 = self.displacement(perp_force_2)
                paral_disp_1 = self.displacement(paral_force_1)
                paral_disp_2 = self.displacement(paral_force_2)
                
                total_disp_1 = perp_disp_1 + paral_disp_1
                total_disp_2 = perp_disp_2 + paral_disp_2
            

            
            total_disp_1 = (total_disp_1 / np.linalg.norm(total_disp_1)) * min(np.linalg.norm(total_disp_1), L_1/2)            
            total_disp_2 = (total_disp_2 / np.linalg.norm(total_disp_2)) * min(np.linalg.norm(total_disp_2), L_2/2)            

            geom_num_list_1 -= total_disp_1 
            geom_num_list_2 -= total_disp_2
            
            
            
            
            new_geom_num_list_1_tolist = (geom_num_list_1*self.bohr2angstroms).tolist()
            new_geom_num_list_2_tolist = (geom_num_list_2*self.bohr2angstroms).tolist()
            for i, elem in enumerate(element_list):
                new_geom_num_list_1_tolist[i].insert(0, elem)
                new_geom_num_list_2_tolist[i].insert(0, elem)
           
            if self.args.pyscf:
                
                file_directory_1 = FIO1.make_pyscf_input_file([new_geom_num_list_1_tolist], iter) 
                file_directory_2 = FIO2.make_pyscf_input_file([new_geom_num_list_2_tolist], iter) 
            
            else:
                new_geom_num_list_1_tolist.insert(0, init_electric_charge_and_multiplicity)
                new_geom_num_list_2_tolist.insert(0, final_electric_charge_and_multiplicity)
                
                file_directory_1 = FIO1.make_psi4_input_file([new_geom_num_list_1_tolist], iter)
                file_directory_2 = FIO2.make_psi4_input_file([new_geom_num_list_2_tolist], iter)
            
            if self.RMS(perp_force_1) < 0.01 and self.RMS(perp_force_2) < 0.01:
                print("enough to relax.")
                break
            
            prev_total_force_1 = total_force_1
            prev_total_force_2 = total_force_2
            
            
            
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
        #G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient [a.u.]", name="gradient")
        prev_delta_geometry = 0.0
        for iter in range(0, self.microiterlimit):
            if os.path.isfile(self.iEIP_FOLDER_DIRECTORY+"end.txt"):
                break
            print("# ITR. "+str(iter))
            
            energy_1, gradient_1, geom_num_list_1, _ = SP1.single_point(file_directory_1, element_list, iter, init_electric_charge_and_multiplicity, self.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, _ = SP2.single_point(file_directory_2, element_list, iter, final_electric_charge_and_multiplicity, self.force_data["xtb"])
            
            if iter == 0:
                m_1 = gradient_1 * 0.0
                m_2 = gradient_1 * 0.0
                v_1 = gradient_1 * 0.0
                v_2 = gradient_1 * 0.0
                ini_geom_1 = geom_num_list_1
                ini_geom_2 = geom_num_list_2
            
            BPC_1 = BiasPotentialCalculation(SP1.Model_hess, SP1.FC_COUNT, self.iEIP_FOLDER_DIRECTORY)
            BPC_2 = BiasPotentialCalculation(SP2.Model_hess, SP2.FC_COUNT, self.iEIP_FOLDER_DIRECTORY)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.force_data)
        
            if self.microiter_num > 0 and iter > 0:
                energy_1, gradient_1, bias_energy_1, bias_gradient_1, geom_num_list_1, energy_2, gradient_2, bias_energy_2, bias_gradient_2, geom_num_list_2 = self.microiteration(SP, FIO1, FIO2, file_directory_1, file_directory_2, element_list, init_electric_charge_and_multiplicity, final_electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter)
            
        
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
           
            
            if self.args.pyscf:
                
                file_directory_1 = FIO1.make_pyscf_input_file([new_geom_num_list_1_tolist], iter+1) 
                file_directory_2 = FIO2.make_pyscf_input_file([new_geom_num_list_2_tolist], iter+1) 
            
            else:
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
                self.BETA *= 1.01
        
        bias_ene_list = BIAS_ENERGY_LIST_A + BIAS_ENERGY_LIST_B[::-1]
        bias_grad_list = BIAS_GRAD_LIST_A + BIAS_GRAD_LIST_B[::-1]
        
        
        ene_list = ENERGY_LIST_A + ENERGY_LIST_B[::-1]
        grad_list = GRAD_LIST_A + GRAD_LIST_B[::-1]
        NUM_LIST = [i for i in range(len(ene_list))]
        G.single_plot(NUM_LIST, ene_list, file_directory_1, "energy", axis_name_2="energy [kcal/mol]", name="energy")   
        G.single_plot(NUM_LIST, grad_list, file_directory_1, "gradient", axis_name_2="grad (RMS) [a.u.]", name="gradient")
        G.single_plot(NUM_LIST, bias_ene_list, file_directory_1, "bias_energy", axis_name_2="energy [kcal/mol]", name="energy")   
        G.single_plot(NUM_LIST, bias_grad_list, file_directory_1, "bias_gradient", axis_name_2="grad (RMS) [a.u.]", name="gradient")
        FIO1.xyz_file_make_for_DM(img_1="A", img_2="B")
        
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
    
    
    
    def optimize_using_tblite(self):
        from tblite_calculation_tools import Calculation
        file_path_list = glob.glob(self.START_FILE+"*_[A-Z].xyz")
        FIO_img_list = []

        for file_path in file_path_list:
            FIO_img_list.append(FileIO(self.iEIP_FOLDER_DIRECTORY, file_path))

        geometry_list_list = []
        element_list_list = []
        electric_charge_and_multiplicity_list = []

        for i in range(len(FIO_img_list)):
            geometry_list, element_list, electric_charge_and_multiplicity = FIO_img_list[i].make_geometry_list(self.electric_charge_and_multiplicity_list[i])
            geometry_list_list.append(geometry_list)
            element_list_list.append(element_list)
            electric_charge_and_multiplicity_list.append(electric_charge_and_multiplicity)
    
        SP_list = []
        file_directory_list = []
        for i in range(len(FIO_img_list)):
            SP_list.append(Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = -1,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye(3*len(geometry_list_list[i])),
                         unrestrict=self.unrestrict, 
                         excited_state = self.excite_state_list[i]))
            file_directory = FIO_img_list[i].make_psi4_input_file(geometry_list_list[i], 0)
            file_directory_list.append(file_directory)
       
       
        
        if self.mf_mode != "None":
            self.model_function_optimization(file_directory_list, SP_list, element_list_list, self.electric_charge_and_multiplicity_list, FIO_img_list)
        else:
            self.iteration(file_directory_list[0], file_directory_list[1], SP_list[0], SP_list[1], element_list, self.electric_charge_and_multiplicity_list[0], self.electric_charge_and_multiplicity_list[1], FIO_img_list[0], FIO_img_list[1])
        
        
    def optimize_using_psi4(self):
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
            geometry_list_list.append(geometry_list)
            element_list_list.append(element_list)
            electric_charge_and_multiplicity_list.append(electric_charge_and_multiplicity)
    
        SP_list = []
        file_directory_list = []
        for i in range(len(FIO_img_list)):
            SP_list.append(Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY,
                         FUNCTIONAL = self.FUNCTIONAL,
                         BASIS_SET = self.BASIS_SET,
                         FC_COUNT = -1,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye((3*len(geometry_list_list[i]))),
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         unrestrict=self.unrestrict, 
                         excited_state = self.excite_state_list[i]))
            file_directory = FIO_img_list[i].make_psi4_input_file(geometry_list_list[i], 0)
            file_directory_list.append(file_directory)
       
       
        
        if self.mf_mode != "None":
            self.model_function_optimization(file_directory_list, SP_list, element_list_list, self.electric_charge_and_multiplicity_list, FIO_img_list)
        else:
            self.iteration(file_directory_list[0], file_directory_list[1], SP_list[0], SP_list[1], element_list, self.electric_charge_and_multiplicity_list[0], self.electric_charge_and_multiplicity_list[1], FIO_img_list[0], FIO_img_list[1])

        
    def optimize_using_pyscf(self):
        from pyscf_calculation_tools import Calculation

        file_path_list = glob.glob(self.START_FILE+"*_[A-Z].xyz")
        FIO_img_list = []

        for file_path in file_path_list:
            FIO_img_list.append(FileIO(self.iEIP_FOLDER_DIRECTORY, file_path))

        geometry_list_list = []
        element_list_list = []

        for i in range(len(FIO_img_list)):
            geometry_list, element_list = FIO_img_list[i].make_geometry_list_for_pyscf()
            geometry_list_list.append(geometry_list)
            element_list_list.append(element_list)
           
    
        SP_list = []
        file_directory_list = []
        for i in range(len(FIO_img_list)):
            SP_list.append(Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY,
                         FUNCTIONAL = self.FUNCTIONAL,
                         BASIS_SET = self.BASIS_SET,
                         FC_COUNT = -1,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye(3*len(geometry_list_list[i])),
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         electronic_charge = self.electronic_charge[i],
                         spin_multiplicity = self.spin_multiplicity[i],
                         unrestrict=self.unrestrict, 
                         excited_state = self.excite_state_list[i]))
            file_directory = FIO_img_list[i].make_pyscf_input_file(geometry_list_list[i], 0)
            file_directory_list.append(file_directory)
       
       
        
        if self.mf_mode != "None":
            self.model_function_optimization(file_directory_list, SP_list, element_list_list, self.electric_charge_and_multiplicity_list, FIO_img_list)
        else:
            self.iteration(file_directory_list[0], file_directory_list[1], SP_list[0], SP_list[1], element_list, self.electric_charge_and_multiplicity_list[0], self.electric_charge_and_multiplicity_list[1], FIO_img_list[0], FIO_img_list[1])

        
        
    def model_function_optimization(self, file_directory_list, SP_list, element_list_list, electric_charge_and_multiplicity_list, FIO_img_list):
       
        G = Graph(self.iEIP_FOLDER_DIRECTORY)
        trust_radii = 1.0
        
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
            
            for j in range(len(SP_list)):
                energy, gradient, geom_num_list, _ = SP_list[j].single_point(file_directory_list[j], element_list_list[j], iter, electric_charge_and_multiplicity_list[j], self.force_data["xtb"])
                tmp_gradient_list.append(gradient)
                tmp_energy_list.append(energy)
                tmp_geometry_list.append(geom_num_list)
            
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
                CMV_LIST = []
                
                OPTIMIZER_INSTANCE_LIST = []
                
                for j in range(len(SP_list)):
                    PREV_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_BIAS_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_MOVE_VEC_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_MF_BIAS_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_MF_GRAD_LIST.append(tmp_gradient_list[j] * 0.0)
                    PREV_B_e_LIST.append(0.0)
                    PREV_e_LIST.append(0.0)
                   
                    CMV_LIST.append(CalculateMoveVector("x", trust_radii, element_list_list[j], 0, SP_list[j].FC_COUNT, 0))
                    OPTIMIZER_INSTANCE_LIST.append(CMV_LIST[j].initialization(self.force_data["opt_method"]))
                    #OPTIMIZER_INSTANCE_LIST[j].set_hessian(SP_list[j].Model_hess)
                    
                init_geom_list = tmp_geometry_list
                PREV_GEOM_LIST = tmp_geometry_list 
                
            BPC_LIST = []
            for j in range(len(SP_list)):
                BPC_LIST.append(BiasPotentialCalculation(SP_list[j].Model_hess, SP_list[j].FC_COUNT, self.iEIP_FOLDER_DIRECTORY))

            tmp_bias_energy_list = []
            tmp_bias_gradient_list = []
            for j in range(len(SP_list)):
                _, bias_energy, bias_gradient, _ = BPC_LIST[j].main(tmp_energy_list[j], tmp_gradient_list[j], tmp_geometry_list[j], element_list_list[j], self.force_data)
                tmp_bias_energy_list.append(bias_energy)
                tmp_bias_gradient_list.append(bias_gradient)
 
            tmp_bias_energy_list = np.array(tmp_bias_energy_list)
            tmp_bias_gradient_list = np.array(tmp_bias_gradient_list)
 
            ##-----
            ##  model function
            ##-----
            if self.mf_mode == "seam":
                SMF = MF.SeamModelFunction()
                mf_energy = SMF.calc_energy(tmp_energy_list[0], tmp_energy_list[1])
                mf_bias_energy = SMF.calc_energy(tmp_bias_energy_list[0], tmp_bias_energy_list[1])
                smf_grad_1, smf_grad_2 = SMF.calc_grad(tmp_energy_list[0], tmp_energy_list[1], tmp_gradient_list[0], tmp_gradient_list[1])
                
                smf_bias_grad_1, smf_bias_grad_2 = SMF.calc_grad(tmp_bias_energy_list[0], tmp_bias_energy_list[1], tmp_bias_gradient_list[0], tmp_bias_gradient_list[1])
                tmp_smf_bias_grad_list = [smf_bias_grad_1, smf_bias_grad_2]
                tmp_smf_grad_list = [smf_grad_1, smf_grad_2]
                tmp_smf_bias_grad_list = np.array(tmp_smf_bias_grad_list)
                tmp_smf_grad_list = np.array(tmp_smf_grad_list)
                
            elif self.mf_mode == "avoided":
                pass
            
            else:
                print("No model function is selected.")
                raise
            
            tmp_move_vector_list = []
            tmp_new_geometry_list = []
            for j in range(len(SP_list)):
                tmp_new_geometry, tmp_move_vector, tmp_optimizer_instances, trust_radii = CMV_LIST[j].calc_move_vector(iter, tmp_geometry_list[j], tmp_smf_bias_grad_list[j], PREV_MF_BIAS_GRAD_LIST[j], PREV_GEOM_LIST[j], PREV_MF_e, PREV_MF_B_e, PREV_MOVE_VEC_LIST[j], init_geom_list[j], tmp_smf_grad_list[j], PREV_GRAD_LIST[j], OPTIMIZER_INSTANCE_LIST[j])
                tmp_move_vector_list.append(tmp_move_vector)
                tmp_new_geometry_list.append(tmp_new_geometry)
                OPTIMIZER_INSTANCE_LIST[j] = tmp_optimizer_instances
            
            tmp_move_vector_list = np.array(tmp_move_vector_list)
            tmp_new_geometry_list = np.array(tmp_new_geometry_list)

            for j in range(len(SP_list)):
                tmp_new_geometry_list[j] -= Calculationtools().calc_center_of_mass(tmp_new_geometry_list[j], element_list_list[j])
                tmp_new_geometry_list[j], _ = Calculationtools().kabsch_algorithm(tmp_new_geometry_list[j], PREV_GEOM_LIST[j])
                

    
            tmp_new_geometry_list_to_list =  tmp_new_geometry_list.tolist()
            
            for j in range(len(SP_list)):
                for i, elem in enumerate(element_list_list[j]):
                    tmp_new_geometry_list_to_list[j][i].insert(0, elem)
                
            if self.args.pyscf:
                for j in range(len(SP_list)):
                    file_directory_list[j] = FIO_img_list[j].make_pyscf_input_file([tmp_new_geometry_list_to_list[j]], iter+1)
            else:
                for j in range(len(SP_list)):
                    tmp_new_geometry_list_to_list[j].insert(0, electric_charge_and_multiplicity_list[j])
                
                for j in range(len(SP_list)):
                    file_directory_list[j] = FIO_img_list[j].make_psi4_input_file([tmp_new_geometry_list[j]], iter+1)
            
              
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
            if abs(tmp_bias_gradient_list.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt((tmp_bias_gradient_list**2).mean())) < self.RMS_FORCE_THRESHOLD and  abs(tmp_move_vector_list.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt((tmp_move_vector_list**2).mean())) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
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
            if self.args.pyscf:
                FIO_img_list[j].xyz_file_make_for_pyscf(name=alphabet[j])
            else:
                FIO_img_list[j].xyz_file_make(name=alphabet[j])
        
        return
    
    def print_info_for_model_func(self, optmethod, e, B_e, B_g, displacement_vector, pre_e, pre_B_e):
        print("caluculation results (unit a.u.):")
        print("OPT method            : {} ".format(optmethod))
        print("                         Value                         Threshold ")
        print("ENERGY                : {:>15.12f} ".format(e))
        print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
        print("Maxinum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
        print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt((B_g**2).mean())), self.RMS_FORCE_THRESHOLD))
        print("Maxinum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), self.MAX_DISPLACEMENT_THRESHOLD))
        print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt((displacement_vector**2).mean())), self.RMS_DISPLACEMENT_THRESHOLD))
        print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
        print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
        return
    
    def run(self):
        if self.args.pyscf:
            self.optimize_using_pyscf()
        elif self.args.usextb != "None":
            self.optimize_using_tblite()
        else:
            self.optimize_using_psi4()
        
        print("completed...")
        return
    
        
    