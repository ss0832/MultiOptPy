import os
import sys
import time
import glob
import copy

import numpy as np

from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from param import UnitValueLib, element_number
from interface import force_data_parser

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
        
        self.microiterlimit = 1000
        
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
            
        self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
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

    def microiteration(self, SP, FIO1, FIO2, file_directory_1, file_directory_2, element_list, electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter):
        #Add force to minimize potential along MEP. (based on nudged elestic bond method)
 
        for i in range(self.microiter_num):
            print("# Microiteration "+str(i))
            energy_1, gradient_1, geom_num_list_1, _ = SP.single_point(file_directory_1, element_list, iter, electric_charge_and_multiplicity, self.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, _ = SP.single_point(file_directory_2, element_list, iter, electric_charge_and_multiplicity, self.force_data["xtb"])
            
            BPC_1 = BiasPotentialCalculation(SP.Model_hess, SP.FC_COUNT)
            BPC_2 = BiasPotentialCalculation(SP.Model_hess, SP.FC_COUNT)
            
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
            

            
            total_disp_1 = (total_disp_1 / np.linalg.norm(total_disp_1)) * min(np.linalg.norm(total_disp_1), L_1)            
            total_disp_2 = (total_disp_2 / np.linalg.norm(total_disp_2)) * min(np.linalg.norm(total_disp_2), L_2)            

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
                new_geom_num_list_1_tolist.insert(0, electric_charge_and_multiplicity)
                new_geom_num_list_2_tolist.insert(0, electric_charge_and_multiplicity)
                
                file_directory_1 = FIO1.make_psi4_input_file([new_geom_num_list_1_tolist], iter)
                file_directory_2 = FIO2.make_psi4_input_file([new_geom_num_list_2_tolist], iter)
            
            if self.RMS(perp_force_1) < 0.01 and self.RMS(perp_force_2) < 0.01:
                print("enough to relax.")
                break
            
            prev_total_force_1 = total_force_1
            prev_total_force_2 = total_force_2
            
            
            
        return energy_1, gradient_1, bias_energy_1, bias_gradient_1, geom_num_list_1, energy_2, gradient_2, bias_energy_2, bias_gradient_2, geom_num_list_2


    def iteration(self, file_directory_1, file_directory_2, SP, element_list, electric_charge_and_multiplicity, FIO1, FIO2):
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
            
            energy_1, gradient_1, geom_num_list_1, _ = SP.single_point(file_directory_1, element_list, iter, electric_charge_and_multiplicity, self.force_data["xtb"])
            energy_2, gradient_2, geom_num_list_2, _ = SP.single_point(file_directory_2, element_list, iter, electric_charge_and_multiplicity, self.force_data["xtb"])
            
            if iter == 0:
                m_1 = gradient_1 * 0.0
                m_2 = gradient_1 * 0.0
                v_1 = gradient_1 * 0.0
                v_2 = gradient_1 * 0.0
                ini_geom_1 = geom_num_list_1
                ini_geom_2 = geom_num_list_2
            
            BPC_1 = BiasPotentialCalculation(SP.Model_hess, SP.FC_COUNT)
            BPC_2 = BiasPotentialCalculation(SP.Model_hess, SP.FC_COUNT)
            
            _, bias_energy_1, bias_gradient_1, _ = BPC_1.main(energy_1, gradient_1, geom_num_list_1, element_list, self.force_data)
            _, bias_energy_2, bias_gradient_2, _ = BPC_2.main(energy_2, gradient_2, geom_num_list_2, element_list, self.force_data)
        
            if self.microiter_num > 0 and iter > 0:
                energy_1, gradient_1, bias_energy_1, bias_gradient_1, geom_num_list_1, energy_2, gradient_2, bias_energy_2, bias_gradient_2, geom_num_list_2 = self.microiteration(SP, FIO1, FIO2, file_directory_1, file_directory_2, element_list, electric_charge_and_multiplicity, prev_geom_num_list_1, prev_geom_num_list_2, iter)
            
        
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
                new_geom_num_list_1_tolist.insert(0, electric_charge_and_multiplicity)
                new_geom_num_list_2_tolist.insert(0, electric_charge_and_multiplicity)
                
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
        L = np.linalg.norm(geom_num_list_2 - geom_num_list_1)
        return L #Bohr
   
    def target_dist_2imgs(self, L):
        Lt = max(L * 0.9, self.L_covergence - 0.01)       
    
        return Lt
    
    def force_R(self, L):
        F_R = min(max(L/self.L_covergence, 1)) * self.F_R_convage_criterion
        return F_R  

    def displacement(self, force):
        n_force = np.linalg.norm(force)
        displacement = (force / n_force) * min(n_force, self.displacement_limit)
        return displacement
    
    def displacement_prime(self, force):
        n_force = np.linalg.norm(force)
        displacement = (force / n_force) * self.displacement_limit 
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
        file_path_A = glob.glob(self.START_FILE+"*_A.xyz")[0]
        file_path_B = glob.glob(self.START_FILE+"*_B.xyz")[0]
        FIO_img1 = FileIO(self.iEIP_FOLDER_DIRECTORY, file_path_A)
        FIO_img2 = FileIO(self.iEIP_FOLDER_DIRECTORY, file_path_B)

        geometry_list_1, element_list, electric_charge_and_multiplicity = FIO_img1.make_geometry_list(self.electric_charge_and_multiplicity)
        geometry_list_2, _, _ = FIO_img2.make_geometry_list(self.electric_charge_and_multiplicity)
        
        SP = Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = -1,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye(3*len(geometry_list_1)))
        file_directory_1 = FIO_img1.make_psi4_input_file(geometry_list_1, 0)
        file_directory_2 = FIO_img2.make_psi4_input_file(geometry_list_2, 0)
        
        self.iteration(file_directory_1, file_directory_2, SP, element_list, electric_charge_and_multiplicity, FIO_img1, FIO_img2)
        
        
    def optimize_using_psi4(self):
        from psi4_calculation_tools import Calculation
        file_path_A = glob.glob(self.START_FILE+"*_A.xyz")[0]
        file_path_B = glob.glob(self.START_FILE+"*_B.xyz")[0]
        FIO_img1 = FileIO(self.iEIP_FOLDER_DIRECTORY, file_path_A)
        FIO_img2 = FileIO(self.iEIP_FOLDER_DIRECTORY, file_path_B)

        
        geometry_list_1, element_list, electric_charge_and_multiplicity = FIO_img1.make_geometry_list(self.electric_charge_and_multiplicity)
        geometry_list_2, _, _ = FIO_img2.make_geometry_list(self.electric_charge_and_multiplicity)
        
        
        SP = Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY,
                         FUNCTIONAL = self.FUNCTIONAL,
                         BASIS_SET = self.BASIS_SET,
                         FC_COUNT = -1,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye((3*len(geometry_list_1))),
                         SUB_BASIS_SET = self.SUB_BASIS_SET)
                         
        file_directory_1 = FIO_img1.make_psi4_input_file(geometry_list_1, 0)
        file_directory_2 = FIO_img2.make_psi4_input_file(geometry_list_2, 0)
        
        self.iteration(file_directory_1, file_directory_2, SP, element_list, electric_charge_and_multiplicity, FIO_img1, FIO_img2)
        
    def optimize_using_pyscf(self):
        from pyscf_calculation_tools import Calculation
        file_path_A = glob.glob(self.START_FILE+"*_A.xyz")[0]
        file_path_B = glob.glob(self.START_FILE+"*_B.xyz")[0]
        FIO_img1 = FileIO(self.iEIP_FOLDER_DIRECTORY, file_path_A)
        FIO_img2 = FileIO(self.iEIP_FOLDER_DIRECTORY, file_path_B)

        
        geometry_list_1, element_list = FIO_img1.make_geometry_list_for_pyscf()
        geometry_list_2, _ = FIO_img2.make_geometry_list_for_pyscf()
        
    
        SP = Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY,
                         FUNCTIONAL = self.FUNCTIONAL,
                         BASIS_SET = self.BASIS_SET,
                         FC_COUNT = -1,
                         BPA_FOLDER_DIRECTORY = self.iEIP_FOLDER_DIRECTORY,
                         Model_hess = np.eye(3*len(geometry_list_1)),
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         electronic_charge = self.electronic_charge,
                         spin_multiplicity = self.spin_multiplicity)
        file_directory_1 = FIO_img1.make_pyscf_input_file(geometry_list_1, 0)
        file_directory_2 = FIO_img2.make_pyscf_input_file(geometry_list_2, 0)
        
        self.iteration(file_directory_1, file_directory_2, SP, element_list, self.electric_charge_and_multiplicity, FIO_img1, FIO_img2)
        
        
        
        
    def run(self):
        if self.args.pyscf:
            self.optimize_using_pyscf()
        elif self.args.usextb != "None":
            self.optimize_using_tblite()
        else:
            self.optimize_using_psi4()
        
        print("completed...")
        return
    
        
    