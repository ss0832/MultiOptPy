import sys
import os
import copy
import glob
import itertools
import datetime
import time

import numpy as np

from optimizer import CalculateMoveVector, Opt_calc_tmps, Model_hess_tmp
from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib, element_number
from interface import force_data_parser
from approx_hessian import ApproxHessian
from cmds_analysis import CMDSPathAnalysis
from redundant_coordinations import RedundantInternalCoordinates
from riemann_curvature import CalculationCurvature

class Optimize:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
 
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #

        self.MAX_FORCE_THRESHOLD = 0.0003 #0.0003
        self.RMS_FORCE_THRESHOLD = 0.0002 #0.0002
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0015 #0.0015 
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0010 #0.0010

        
        self.args = args #
        self.FC_COUNT = args.calc_exact_hess # 
        #---------------------------
        self.temperature = float(args.md_like_perturbation)
        self.CMDS = args.cmds 
        self.ricci_curvature = args.ricci_curvature
        #---------------------------
        if len(args.opt_method) > 2:
            print("invaild input (-opt)")
            sys.exit(0)
        
        if args.DELTA == "x":
            if args.opt_method[0] == "FSB":
                args.DELTA = 0.1
            elif args.opt_method[0] == "FSB_LS":
                args.DELTA = 0.3
            elif args.opt_method[0] == "RFO_FSB":
                args.DELTA = 0.5
            elif args.opt_method[0] == "RFO2_FSB":
                args.DELTA = 0.5    
                
            elif args.opt_method[0] == "BFGS":
                args.DELTA = 0.5
            elif args.opt_method[0] == "BFGS_LS":
                args.DELTA = 0.2
            elif args.opt_method[0] == "RFO_BFGS":
                args.DELTA = 0.5
            elif args.opt_method[0] == "Bofill":
                args.DELTA = 1.0
            elif args.opt_method[0] == "RFO_Bofill":
                args.DELTA = 0.5
            elif args.opt_method[0] == "RFO2_Bofill":
                args.DELTA = 0.5 
            elif args.opt_method[0] == "MSP":
                args.DELTA = 1.0
            elif args.opt_method[0] == "RFO2_MSP":
                args.DELTA = 0.5
            elif args.opt_method[0] == "mBFGS":
                args.DELTA = 0.50
            elif args.opt_method[0] == "mFSB":
                args.DELTA = 0.50
            elif args.opt_method[0] == "RFO_mBFGS":
                args.DELTA = 0.30
            elif args.opt_method[0] == "RFO_mFSB":
                args.DELTA = 0.30
            elif args.opt_method[0] == "Adaderivative":
                args.DELTA = 0.002
            elif args.opt_method[0] == "Adabound":
                args.DELTA = 0.01
            elif args.opt_method[0] == "AdaMax":
                args.DELTA = 0.01
            elif args.opt_method[0] == "CG":
                args.DELTA = 1.0
            elif args.opt_method[0] == "CG_HS":
                args.DELTA = 1.0            
            elif args.opt_method[0] == "CG_DY":
                args.DELTA = 1.0
            elif args.opt_method[0] == "CG_FR":
                args.DELTA = 1.0
            elif args.opt_method[0] == "SADAM":
                args.DELTA = 1.0
            elif args.opt_method[0] == "SAMSGrad":
                args.DELTA = 1.0
            elif args.opt_method[0] == "YOGI":
                args.DELTA = 0.05
            elif args.opt_method[0] == "Adamod":
                args.DELTA = 1.0
            #elif args.opt_method[0] == "CG2":
            #    args.DELTA = 1.0    
            else:
                args.DELTA = 0.06
        else:
            pass 
        self.DELTA = float(args.DELTA) # 

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT #
        self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        
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
            self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
            
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET +="\nassign "+str(self.BASIS_SET)+"\n" # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            
        #-----------------------------
        if args.usextb == "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        else:
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+args.usextb+"_"+str(time.time())+"/"
        
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True) #
        
        self.Model_hess = None #
        self.Opt_params = None #
        self.DC_check_dist = 10.0#ang.
        self.unrestrict = args.unrestrict
        return

    def grad_fix_atoms(self, gradient, geom_num_list, fix_atom_list):
        #fix_atom_list: force_data["gradient_fix_atoms"]
        RIC = RedundantInternalCoordinates()
        b_mat = RIC.B_matrix(geom_num_list)
        int_grad = RIC.cartgrad2RICgrad(gradient.reshape(len(geom_num_list)*3, 1), b_mat)
        
        RIC_idx_list = list(itertools.combinations([i+1 for i in range(len(geom_num_list))], 2))
        for fix_atoms in fix_atom_list:
            fix_combination_list = list(itertools.combinations(sorted(fix_atoms), 2))
            for fix in fix_combination_list:
                int_grad_idx = RIC_idx_list.index(fix)
                int_grad[int_grad_idx] *= 0.0
               
        fixed_gradient = RIC.RICgrad2cartgrad(int_grad, b_mat).reshape(len(geom_num_list), 3)
        
        return fixed_gradient
    
    
    def optimize_using_tblite(self):
        from tblite_calculation_tools import Calculation
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         unrestrict = self.unrestrict)
        #-----------------------------------
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        #----------------------------------
        
        self.cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        bias_grad_list = []
        
        orthogonal_bias_grad_list = []
        orthogonal_grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_number_list, iter, electric_charge_and_multiplicity, force_data["xtb"])

            self.Model_hess = SP.Model_hess
            #---------------------------------------
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            else:
                pass


            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            CalcBiaspot.Model_hess = self.Model_hess
            
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            #-------------------energy profile 
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
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                    f.write("bias gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((B_g**2).mean()))+"\n")
            #-------------------
            #----------------------------
            if len(force_data["gradient_fix_atoms"]) > 0:
                #fix_atom_list: force_data["gradient_fix_atoms"]
                B_g = self.grad_fix_atoms(B_g, geom_num_list, force_data["gradient_fix_atoms"])
                g = self.grad_fix_atoms(g, geom_num_list, force_data["gradient_fix_atoms"])
                
            #----------------------------
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, element_list, self.args.saddle_order, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, Opt_params, Model_hess, trust_radii = CMV.calc_move_vector(iter, geom_num_list, B_g, force_data["opt_method"], pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)   
            
            #----------------------------
            displacement_vector = geom_num_list - pre_geom
            self.print_info(force_data["opt_method"], e, B_e, B_g, displacement_vector, pre_e, pre_B_e)
            
            
            grad_list.append(np.sqrt((g**2).mean()))
            bias_grad_list.append(np.sqrt((B_g**2).mean()))
            #----------------------
            if iter > 0:
                norm_pre_move_vec = (pre_move_vector / np.linalg.norm(pre_move_vector)).reshape(len(pre_move_vector)*3, 1)
                orthogonal_bias_grad = B_g.reshape(len(B_g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                orthogonal_grad = g.reshape(len(g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                RMS_ortho_B_g = abs(np.sqrt((orthogonal_bias_grad**2).mean()))
                RMS_ortho_g = abs(np.sqrt((orthogonal_grad**2).mean()))
                orthogonal_bias_grad_list.append(RMS_ortho_B_g)
                orthogonal_grad_list.append(RMS_ortho_g)
            
            
            #------------------------
            if abs(B_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt((B_g**2).mean())) < self.RMS_FORCE_THRESHOLD and  abs(displacement_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt((displacement_vector**2).mean())) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            
            #dissociation check
            DC_exit_flag = self.dissociation_check(new_geometry, element_list)
            if DC_exit_flag:
                break


                 
            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            #---------------------------
            
            
            geometry_list = FIO.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        G.single_plot(self.NUM_LIST[1:], (np.array(bias_grad_list[1:]) - np.array(orthogonal_bias_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal bias gradient diff (RMS) [a.u.]", name="orthogonal_bias_gradient_diff")
        G.single_plot(self.NUM_LIST[1:], (np.array(grad_list[1:]) - np.array(orthogonal_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal gradient diff (RMS) [a.u.]", name="orthogonal_gradient_diff")
        
        
        G.single_plot(self.NUM_LIST[1:], orthogonal_bias_grad_list, file_directory, "", axis_name_2="orthogonal bias gradient (RMS) [a.u.]", name="orthogonal_bias_gradient")
        G.single_plot(self.NUM_LIST[1:], orthogonal_grad_list, file_directory, "", axis_name_2="orthogonal gradient (RMS) [a.u.]", name="orthogonal_gradient")
        
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
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
        
        
        
        
        #----------------------
        print("Complete...")
        return

    def optimize_using_psi4(self):
        from psi4_calculation_tools import Calculation
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         unrestrict = self.unrestrict)
        #----------------------------------
        
        self.cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        bias_grad_list = []
        orthogonal_bias_grad_list = []
        orthogonal_grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_list, iter, electric_charge_and_multiplicity)
            
            proj_model_hess = ApproxHessian().main(geom_num_list, element_list, g)

            
            #---------------------------------------
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            else:
                pass


            
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            CalcBiaspot.Model_hess = self.Model_hess
            
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            #-------------------energy profile 
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
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                    f.write("bias gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((B_g**2).mean()))+"\n")
            #-------------------
            #----------------------------
            if len(force_data["gradient_fix_atoms"]) > 0:
                #fix_atom_list: force_data["gradient_fix_atoms"]
                B_g = self.grad_fix_atoms(B_g, geom_num_list, force_data["gradient_fix_atoms"])
                g = self.grad_fix_atoms(g, geom_num_list, force_data["gradient_fix_atoms"])
                
            #----------------------------

            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, element_list, self.args.saddle_order, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, Opt_params, Model_hess, trust_radii = CMV.calc_move_vector(iter, geom_num_list, B_g, force_data["opt_method"], pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)   
            #----------------------------
            displacement_vector = geom_num_list - pre_geom
            self.print_info(force_data["opt_method"], e, B_e, B_g, displacement_vector, pre_e, pre_B_e)
            
            
            grad_list.append(np.sqrt((g**2).mean()))
            bias_grad_list.append(np.sqrt((B_g**2).mean()))
            if iter > 0:
                norm_pre_move_vec = (pre_move_vector / np.linalg.norm(pre_move_vector)).reshape(len(pre_move_vector)*3, 1)
                orthogonal_bias_grad = B_g.reshape(len(B_g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                orthogonal_grad = g.reshape(len(g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                RMS_ortho_B_g = abs(np.sqrt((orthogonal_bias_grad**2).mean()))
                RMS_ortho_g = abs(np.sqrt((orthogonal_grad**2).mean()))
                orthogonal_bias_grad_list.append(RMS_ortho_B_g)
                orthogonal_grad_list.append(RMS_ortho_g)
            if abs(B_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt((B_g**2).mean())) < self.RMS_FORCE_THRESHOLD and  abs(displacement_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt((displacement_vector**2).mean())) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            
            #dissociation check
            DC_exit_flag = self.dissociation_check(new_geometry, element_list)
            if DC_exit_flag:
                break

            
            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = FIO.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        G.single_plot(self.NUM_LIST[1:], (np.array(bias_grad_list[1:]) - np.array(orthogonal_bias_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal bias gradient diff (RMS) [a.u.]", name="orthogonal_bias_gradient_diff")
        G.single_plot(self.NUM_LIST[1:], (np.array(grad_list[1:]) - np.array(orthogonal_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal gradient diff (RMS) [a.u.]", name="orthogonal_gradient_diff")
        
        G.single_plot(self.NUM_LIST[1:], orthogonal_bias_grad_list, file_directory, "", axis_name_2="orthogonal bias gradient (RMS) [a.u.]", name="orthogonal_bias_gradient")
        G.single_plot(self.NUM_LIST[1:], orthogonal_grad_list, file_directory, "", axis_name_2="orthogonal gradient (RMS) [a.u.]", name="orthogonal_gradient")
        
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
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
        
       
        #----------------------
        print("Complete...")
        return
    
    def optimize_using_pyscf(self):
        from pyscf_calculation_tools import Calculation
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = force_data_parser(self.args)
        finish_frag = False
        geometry_list, element_list = FIO.make_geometry_list_for_pyscf()
        file_directory = FIO.make_pyscf_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(self.args))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         spin_multiplicity = self.spin_multiplicity,
                         electronic_charge = self.electronic_charge,
                         unrestrict = self.unrestrict)
        #----------------------------------
        
        self.cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        bias_grad_list = []
        orthogonal_bias_grad_list = []
        orthogonal_grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_list, iter)

            self.Model_hess = SP.Model_hess
            #---------------------------------------
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            else:
                pass


            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            CalcBiaspot.Model_hess = self.Model_hess
            
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            #-------------------energy profile 
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
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                    f.write("bias gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"bias_gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((B_g**2).mean()))+"\n")
            #-------------------
            #----------------------------
            if len(force_data["gradient_fix_atoms"]) > 0:
                #fix_atom_list: force_data["gradient_fix_atoms"]
                B_g = self.grad_fix_atoms(B_g, geom_num_list, force_data["gradient_fix_atoms"])
                g = self.grad_fix_atoms(g, geom_num_list, force_data["gradient_fix_atoms"])
                
            #----------------------------
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, element_list, self.args.saddle_order, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, Opt_params, Model_hess, trust_radii = CMV.calc_move_vector(iter, geom_num_list, B_g, force_data["opt_method"], pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)     
            
            #----------------------------
            displacement_vector = geom_num_list - pre_geom
            self.print_info(force_data["opt_method"], e, B_e, B_g, displacement_vector, pre_e, pre_B_e)
            
            grad_list.append(np.sqrt((g**2).mean()))
            bias_grad_list.append(np.sqrt((B_g**2).mean()))
            if iter > 0:
                norm_pre_move_vec = (pre_move_vector / np.linalg.norm(pre_move_vector)).reshape(len(pre_move_vector)*3, 1)
                orthogonal_bias_grad = B_g.reshape(len(B_g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                orthogonal_grad = g.reshape(len(g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                RMS_ortho_B_g = abs(np.sqrt((orthogonal_bias_grad**2).mean()))
                RMS_ortho_g = abs(np.sqrt((orthogonal_grad**2).mean()))
                orthogonal_bias_grad_list.append(RMS_ortho_B_g)
                orthogonal_grad_list.append(RMS_ortho_g)
            if abs(B_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt((B_g**2).mean())) < self.RMS_FORCE_THRESHOLD and  abs(displacement_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt((displacement_vector**2).mean())) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            print("\ngeometry:")
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            #----------------------------
            #dissociation check
            DC_exit_flag = self.dissociation_check(new_geometry, element_list)
            if DC_exit_flag:
                break

             
            #----------------------------
            
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = FIO.make_geometry_list_2_for_pyscf(new_geometry, element_list)
            file_directory = FIO.make_pyscf_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        G.single_plot(self.NUM_LIST[1:], (np.array(bias_grad_list[1:]) - np.array(orthogonal_bias_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal bias gradient diff (RMS) [a.u.]", name="orthogonal_bias_gradient_diff")
        G.single_plot(self.NUM_LIST[1:], (np.array(grad_list[1:]) - np.array(orthogonal_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal gradient diff (RMS) [a.u.]", name="orthogonal_gradient_diff")
        
        G.single_plot(self.NUM_LIST[1:], orthogonal_bias_grad_list, file_directory, "", axis_name_2="orthogonal bias gradient (RMS) [a.u.]", name="orthogonal_bias_gradient")
        G.single_plot(self.NUM_LIST[1:], orthogonal_grad_list, file_directory, "", axis_name_2="orthogonal gradient (RMS) [a.u.]", name="orthogonal_gradient")
        
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
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
        
       
        #----------------------
        print("Complete...")
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
        
        print("\nfragm_list:", fragm_atom_num_list)
        
        if len(fragm_atom_num_list) > 1:
            fragm_dist_list = []
            for fragm_1_num, fragm_2_num in list(itertools.combinations(fragm_atom_num_list, 2)):
                dist = Calculationtools().calc_fragm_distance(new_geometry, fragm_1_num, fragm_2_num)
                fragm_dist_list.append(dist)
            
            
            if min(fragm_dist_list) > self.DC_check_dist:
                print("mean fragm distance (ang.)", min(fragm_dist_list), ">", self.DC_check_dist)
                print("This molecules are dissociated.")
                DC_exit_flag = True
            else:
                DC_exit_flag = False
        else:
            DC_exit_flag = False
            
        return DC_exit_flag
    
    
    def print_info(self, optmethod, e, B_e, B_g, displacement_vector, pre_e, pre_B_e):
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
    
        if self.CMDS:
            CMDPA = CMDSPathAnalysis(self.BPA_FOLDER_DIRECTORY, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
            CMDPA.main()
        if self.ricci_curvature:
            CC = CalculationCurvature(self.BPA_FOLDER_DIRECTORY)
            CC.main()
        
        return