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
from redundant_coordinations import RedundantInternalCoordinates
from riemann_curvature import CalculationCurvature
from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from MO_analysis import NROAnalysis
from constraint_condition import GradientSHAKE, shake_parser, LagrangeConstrain, ProjectOutConstrain
from irc import IRC
from bond_connectivity import judge_shape_condition


class Optimize:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
 
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

        self.microiter_num = 100
        self.args = args #
        self.FC_COUNT = args.calc_exact_hess # 
        #---------------------------
        self.temperature = float(args.md_like_perturbation)
        self.CMDS = args.cmds 
        self.PCA = args.pca
        #---------------------------
        if len(args.opt_method) > 2:
            print("invaild input (-opt)")
            sys.exit(0)
        
        if args.DELTA == "x":
            self.DELTA = "x"
        else:
            self.DELTA = float(args.DELTA) # 

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        
        self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        self.excited_state = args.excited_state
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
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
            
        #-----------------------------

        #-----------------------------
        self.Model_hess = None #
        self.mFC_COUNT = args.calc_model_hess
        self.Opt_params = None #
        self.DC_check_dist = 30.0#ang.
        self.unrestrict = args.unrestrict
        self.NRO_analysis = args.NRO_analysis
        if self.NRO_analysis:
            if args.usextb == "None":
                print("Currently, Natural Reaction Orbital analysis is only available for xTB method.")
                sys.exit(0)
        if len(args.constraint_condition) > 0:
            self.constraint_condition_list = shake_parser(args.constraint_condition)
        else:
            self.constraint_condition_list = []
            
        self.irc = args.intrinsic_reaction_coordinates
        self.force_data = force_data_parser(self.args)
        
        self.final_file_directory = None
        self.final_geometry = None#Bohr
        self.final_energy = None #Hartree
        self.final_bias_energy = None #Hartree
        self.othersoft = args.othersoft
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        self.shape_conditions = args.shape_conditions
        self.bias_pot_params_grad_list = None
        self.bias_pot_params_grad_name_list = None
        
        return

    def make_init_directory(self, file):
        self.START_FILE = file #
        if self.args.othersoft != "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_ASE_"+str(time.time()).replace(".","_")+"/"

        elif self.args.usextb == "None" and self.args.usedxtb == "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time()).replace(".","_")+"/"
        else:
            if self.args.usedxtb != "None":
                self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+self.args.usedxtb+"_"+str(time.time()).replace(".","_")+"/"
            else:
                self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+self.args.usextb+"_"+str(time.time()).replace(".","_")+"/"
        
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True) #
        
        return
    
    def save_input_data(self):
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        return

    def optimize(self):
        Calculation, xtb_method = self.import_calculation_module()
        self.save_input_data()
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geom_num_list = None#Bohr
        e = None #Hartree
        B_e = None #Hartree
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        file_directory, electric_charge_and_multiplicity, element_list = self.write_input_files(FIO)
        for i in range(len(element_list)):
            pre_B_g.append([0,0,0])
        pre_B_g = np.array(pre_B_g, dtype="float64")
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        self.cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        bias_grad_list = []
        orthogonal_bias_grad_list = []
        orthogonal_grad_list = []
        
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        #-------------------
        lagrange_lambda_movestep = [] #(M, 1)
        lagrange_lambda_prev_movestep = [] #(M, 1)
        lagrange_lambda_list = [] #(M, 1)
        lagrange_prev_lambda_list = [] #(M, 1)
        lagrange_lambda_grad_list = [] #(M, 1)
        lagrange_lambda_prev_grad_list = [] #(M, 1)
        init_lagrange_lambda_list = [] #(M, 1)
        
        lagrange_constraint_energy = 0.0 
        lagrange_constraint_prev_energy = 0.0
        LC = LagrangeConstrain(force_data["lagrange_constraint_condition_list"], force_data["lagrange_constraint_atoms"])
        natom = len(element_list)
        #-------------------
        PC = ProjectOutConstrain(force_data["projection_constraint_condition_list"], force_data["projection_constraint_atoms"])
        if len(force_data["projection_constraint_condition_list"]) > 0 or len(force_data["lagrange_constraint_condition_list"]) > 0:
            projection_constrain = True
        else:
            projection_constrain = False
        
        #-------------------

        if len(self.constraint_condition_list) > 0:
            class_GradientSHAKE = GradientSHAKE(self.constraint_condition_list)
        
        CalcBiaspot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)

        SP = self.setup_calculation(Calculation)        
        CMV = CalculateMoveVector(self.DELTA, element_list, self.args.saddle_order, self.FC_COUNT, self.temperature, self.args.use_model_hessian)
        optimizer_instances = CMV.initialization(force_data["opt_method"])
        
        for i in range(len(optimizer_instances)):
            optimizer_instances[i].set_hessian(self.Model_hess) #hessian is None.
            if self.DELTA != "x":
                optimizer_instances[i].DELTA = self.DELTA
        
 
            
        #----------------------------------
        if self.NRO_analysis:
            NRO = NROAnalysis(file_directory=self.BPA_FOLDER_DIRECTORY, xtb=xtb_method, element_list=element_list, electric_charge_and_multiplicity=electric_charge_and_multiplicity)
        else:
            NRO = None
        #---------------------------------
        for iter in range(self.NSTEP):
            self.iter = iter
            finish_frag = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")
            if finish_frag:
                break
            #---------------------------------------
            finish_frag = judge_shape_condition(geom_num_list, self.shape_conditions)
            if finish_frag:
                break
            #---------------------------------------

            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            
            SP.Model_hess = copy.copy(self.Model_hess)
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_number_list, iter, electric_charge_and_multiplicity, xtb_method)



            if iter % self.mFC_COUNT == 0 and self.args.use_model_hessian:
                SP.Model_hess = ApproxHessian().main(geom_num_list, element_list, g)
            self.Model_hess = copy.copy(SP.Model_hess)
            
            #if self.args.usedxtb != "None":
            #    common_freq, au_int = SP.ir(geom_num_list, element_list, electric_charge_and_multiplicity, xtb_method)
            #    G.stem_plot(common_freq, au_int, "IR_spectra_iteration_"+str(iter))
            #--------------------------------------
            



            if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                break   
            
            if iter == 0:
                if len(force_data["fix_atoms"]) == 0:
                    initial_geom_num_list = geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
                    pre_geom = initial_geom_num_list - Calculationtools().calc_center(geom_num_list, element_list)
                else:
                    initial_geom_num_list = geom_num_list 
                    pre_geom = initial_geom_num_list 

            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            if iter == 0:
                if len(force_data["lagrange_constraint_condition_list"]) > 0:
                    LC.initialize(geom_num_list)
                    lagrange_lambda_list = LC.lagrange_init_lambda_calc(B_g.reshape(3*natom, 1), geom_num_list)
                    init_lagrange_lambda_list = copy.copy(lagrange_lambda_list)
                    lagrange_prev_lambda_list = copy.copy(lagrange_lambda_list)
                    lagrange_lambda_prev_grad_list = np.array([0.0 for i in range(len(lagrange_lambda_list))], dtype="float64")
                    lagrange_lambda_movestep = np.array([0.0 for i in range(len(lagrange_lambda_list))], dtype="float64")
                    lagrange_lambda_prev_movestep = np.array([0.0 for i in range(len(lagrange_lambda_list))], dtype="float64")
                    lagrange_lambda_prev_grad_list = np.array([0.0 for i in range(len(lagrange_lambda_list))], dtype="float64")
                if len(force_data["projection_constraint_condition_list"]) > 0:
                    PC.initialize(geom_num_list)

            else:
                pass
            
            #----------
            
            print("=== Eigenvalue (Before Adding Bias potential) ===")
            _ = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.Model_hess, element_list, geom_num_list)
            
            print("=== Eigenvalue (After Adding Bias potential) ===")
            _ = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.Model_hess + BPA_hessian, element_list, geom_num_list)
            


            if len(force_data["lagrange_constraint_condition_list"]) > 0:
                lagrange_constraint_atom_addgrad = LC.lagrange_constraint_grad_calc(geom_num_list, lagrange_lambda_list)
                B_g += lagrange_constraint_atom_addgrad
                g += lagrange_constraint_atom_addgrad
                lagrange_lambda_grad_list = LC.lagrange_lambda_grad_calc(geom_num_list)
                lagrange_constraint_energy = LC.calc_lagrange_constraint_energy(geom_num_list, lagrange_lambda_list)
                
                lagrange_constraint_atom_addhess = LC.lagrange_constraint_atom_hess_calc(geom_num_list, lagrange_lambda_list)
                lagrange_constraint_coupling_hessian = LC.lagrange_constraint_couple_hess_calc(geom_num_list)
                
                BPA_hessian += lagrange_constraint_atom_addhess
                tmp_zero_mat = LC.make_couple_zero_hess(geom_num_list)
                combined_BPA_hessian = LC.make_combined_lagrangian_hess(BPA_hessian, lagrange_constraint_coupling_hessian)
                combined_hessian = LC.make_combined_lagrangian_hess(self.Model_hess, tmp_zero_mat)
                e += float(lagrange_constraint_energy)
                B_e += float(lagrange_constraint_energy)
                
            else:
                pass
            
            if len(force_data["fix_atoms"]) > 0:
                fix_num = []
                n_fix = len(force_data["fix_atoms"])
                for fnum in force_data["fix_atoms"]:
                    fix_num.extend([3*(fnum-1)+0, 3*(fnum-1)+1, 3*(fnum-1)+2])
                fix_num = np.array(fix_num, dtype="int64")

                tmp_fix_hess = self.Model_hess[np.ix_(fix_num, fix_num)] + np.eye((3*n_fix)) * 1e-10
                inv_tmp_fix_hess = np.linalg.pinv(tmp_fix_hess)
                tmp_fix_bias_hess = BPA_hessian[np.ix_(fix_num, fix_num)] + np.eye((3*n_fix)) * 1e-10
                inv_tmp_fix_bias_hess = np.linalg.pinv(tmp_fix_bias_hess)

                
                BPA_hessian -= np.dot(BPA_hessian[:, fix_num], np.dot(inv_tmp_fix_bias_hess, BPA_hessian[fix_num, :]))

            
            for i in range(len(optimizer_instances)):
                if len(force_data["lagrange_constraint_condition_list"]) > 0:
                    optimizer_instances[i].set_bias_hessian(combined_BPA_hessian)
                else:
                    if len(force_data["projection_constraint_condition_list"]) > 0:
                        proj_bpa_hess = PC.calc_project_out_hess(geom_num_list, B_g - g, BPA_hessian)
                        optimizer_instances[i].set_bias_hessian(proj_bpa_hess)
                        
                    
                    else:
                        optimizer_instances[i].set_bias_hessian(BPA_hessian)
                
                if iter % self.FC_COUNT == 0 or (self.args.use_model_hessian and iter % self.mFC_COUNT == 0):
                    self.Model_hess -= np.dot(self.Model_hess[:, fix_num], np.dot(inv_tmp_fix_hess, self.Model_hess[fix_num, :]))
                    
                    if len(force_data["lagrange_constraint_condition_list"]) > 0:
                        optimizer_instances[i].set_hessian(combined_hessian)
                    else:
                        if len(force_data["projection_constraint_condition_list"]) > 0:
                            proj_model_hess = PC.calc_project_out_hess(geom_num_list, g, self.Model_hess)
                            optimizer_instances[i].set_hessian(proj_model_hess)
                        else:
                            optimizer_instances[i].set_hessian(self.Model_hess)
                     
            
            #----------------------------
            if len(force_data["opt_fragment"]) > 0:
                B_g = copy.copy(self.calc_fragement_grads(B_g, force_data["opt_fragment"]))
                g = copy.copy(self.calc_fragement_grads(g, force_data["opt_fragment"]))
            
            
            #-------------------energy profile 
            self.save_tmp_energy_profiles(iter, e, g, B_g)
            #-------------------

            if len(self.constraint_condition_list) > 0 and iter > 0:
                B_g = class_GradientSHAKE.run_grad(pre_geom, B_g) 
                g = class_GradientSHAKE.run_grad(pre_geom, g) 
            
            if len(force_data["projection_constraint_condition_list"]) > 0:
                B_g = PC.calc_project_out_grad(geom_num_list, B_g)
                g = PC.calc_project_out_grad(geom_num_list, g)
                
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    g[j-1] = copy.copy(g[j-1]*0.0)
                    B_g[j-1] = copy.copy(B_g[j-1]*0.0)


            
            new_geometry, move_vector, optimizer_instances = CMV.calc_move_vector(iter, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances, lagrange_lambda_list, lagrange_prev_lambda_list, lagrange_lambda_grad_list, lagrange_lambda_prev_grad_list, lagrange_lambda_prev_movestep, init_lagrange_lambda_list, projection_constrain)
            lagrange_prev_lambda_list = copy.copy(lagrange_lambda_list)
            lagrange_lambda_prev_movestep = copy.copy(lagrange_lambda_movestep)
            lagrange_lambda_list = copy.copy(CMV.new_lambda_list)
            lagrange_lambda_movestep = copy.copy(CMV.lambda_movestep)
            #print(lagrange_lambda_list, lagrange_lambda_movestep)
            if len(self.constraint_condition_list) > 0 and iter > 0:
                tmp_new_geometry = class_GradientSHAKE.run_coord(pre_geom, new_geometry/self.bohr2angstroms, element_list) 
               
                new_geometry = tmp_new_geometry * self.bohr2angstroms
            ##------------
            ## project out translation and rotation
            ##------------
            if len(force_data["fix_atoms"]) == 0:
                new_geometry -= Calculationtools().calc_center_of_mass(new_geometry/self.bohr2angstroms, element_list) * self.bohr2angstroms
                new_geometry, _ = Calculationtools().kabsch_algorithm(new_geometry/self.bohr2angstroms, geom_num_list)
                new_geometry *= self.bohr2angstroms
            #if not PC.Isduplicated:
            #    tmp_new_geometry = new_geometry / self.bohr2angstroms
            #    new_geometry = PC.adjust_init_coord(tmp_new_geometry) * self.bohr2angstroms    
            
            #---------------------------------
            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)   
            
            #----------------------------
            if iter == 0:
                displacement_vector = move_vector
            else:
                displacement_vector = geom_num_list - pre_geom
            converge_flag, max_displacement_threshold, rms_displacement_threshold = self.check_converge_criteria(B_g, displacement_vector)
            
            
            self.print_info(e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold)
            
            
            grad_list.append(np.sqrt((g[g > 1e-10]**2).mean()))
            bias_grad_list.append(np.sqrt((B_g[B_g > 1e-10]**2).mean()))
            #----------------------
            if iter > 0:
                norm_pre_move_vec = (pre_move_vector / np.linalg.norm(pre_move_vector)).reshape(len(pre_move_vector)*3, 1)
                orthogonal_bias_grad = B_g.reshape(len(B_g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                orthogonal_grad = g.reshape(len(g)*3, 1) * (1.0 - np.dot(norm_pre_move_vec, norm_pre_move_vec.T))
                RMS_ortho_B_g = abs(np.sqrt((orthogonal_bias_grad**2).mean()))
                RMS_ortho_g = abs(np.sqrt((orthogonal_grad**2).mean()))
                orthogonal_bias_grad_list.append(RMS_ortho_B_g)
                orthogonal_grad_list.append(RMS_ortho_g)
            
            if self.NRO_analysis:
                NRO.run(SP, geom_num_list, move_vector)
            #------------------------
            if converge_flag:#convergent criteria
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
            #Save previous gradient, movestep, and energy.
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
           
            lagrange_lambda_prev_grad_list = copy.copy(lagrange_lambda_grad_list)
            
            lagrange_constraint_prev_energy = lagrange_constraint_energy
            
            #---------------------------
            if self.args.pyscf:
                geometry_list = FIO.make_geometry_list_2_for_pyscf(new_geometry, element_list)
                file_directory = FIO.make_pyscf_input_file(geometry_list, iter+1)
            else:
                geometry_list = FIO.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
                file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------


            #----------------------------
        #plot graph
        
        self.save_results(FIO, G, grad_list, bias_grad_list, orthogonal_bias_grad_list, orthogonal_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, NRO)
        self.bias_pot_params_grad_list = CalcBiaspot.bias_pot_params_grad_list
        self.bias_pot_params_grad_name_list = CalcBiaspot.bias_pot_params_grad_name_list
        
        return


    def save_results(self, FIO, G, grad_list, bias_grad_list, orthogonal_bias_grad_list, orthogonal_grad_list, file_directory, force_data, geom_num_list, e, B_e, SP, NRO):
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, bias_grad_list, file_directory, "", axis_name_2="bias gradient (RMS) [a.u.]", name="bias_gradient")
        G.single_plot(self.NUM_LIST[1:], (np.array(bias_grad_list[1:]) - np.array(orthogonal_bias_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal bias gradient diff (RMS) [a.u.]", name="orthogonal_bias_gradient_diff")
        G.single_plot(self.NUM_LIST[1:], (np.array(grad_list[1:]) - np.array(orthogonal_grad_list)).tolist(), file_directory, "", axis_name_2="orthogonal gradient diff (RMS) [a.u.]", name="orthogonal_gradient_diff")
        if self.NRO_analysis:
            NRO.save_results(self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)

        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                G.single_plot(self.NUM_LIST, self.cos_list[num], file_directory, i)

        if self.args.pyscf:
            FIO.xyz_file_make_for_pyscf()
        else:
            FIO.xyz_file_make()

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
            excited_state=self.excited_state
        )
        SP.cpcm_solv_model = self.cpcm_solv_model
        SP.alpb_solv_model = self.alpb_solv_model
        return SP

    def write_input_files(self, FIO):
        if self.args.pyscf:
            geometry_list, element_list = FIO.make_geometry_list_for_pyscf()
            file_directory = FIO.make_pyscf_input_file(geometry_list, 0)
            electric_charge_and_multiplicity = self.electric_charge_and_multiplicity
        else:
            geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, 0)
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
    
    
    def print_info(self, e, B_e, B_g, displacement_vector, pre_e, pre_B_e, max_displacement_threshold, rms_displacement_threshold):
        print("caluculation results (unit a.u.):")
        print("                         Value                         Threshold ")
        print("ENERGY                : {:>15.12f} ".format(e))
        print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
        print("Maxinum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
        print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt((B_g[B_g > 1e-10]**2).mean())), self.RMS_FORCE_THRESHOLD))
        print("Maxinum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), max_displacement_threshold))
        print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt((displacement_vector[displacement_vector > 1e-10]**2).mean())), rms_displacement_threshold))
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
        #print(calced_gradient)
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
            print(f"Optimization of {file} is completed.")
            
        return