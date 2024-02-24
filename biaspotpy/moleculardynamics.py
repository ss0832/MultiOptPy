import sys
import os
import shutil
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
from param import UnitValueLib, element_number, atomic_mass
from interface import force_data_parser
from approx_hessian import ApproxHessian
from cmds_analysis import CMDSPathAnalysis



class Thermostat:
    def __init__(self, momentum_list, temperature, pressure):
    
        self.momentum_list = momentum_list #list
        self.temperature = temperature
        self.initial_temperature = temperature
        self.pressure = pressure
        self.initial_pressure = pressure
        
        self.zeta = 0.0
        self.eta = 0.0
        self.scaling = 0.0
        
        self.g_value = len(momentum_list) * 3
        self.Q_value = 1.0
        #self.M_value = 1.0
        self.Boltzmann_constant = 3.16681 * 10 ** (-6) # hartree/K
        self.delta_timescale = 0.10
        self.volume = 1.00
        
        self.Instantaneous_temperatures_list = []
        self.Instantaneous_momentum_list = []
        
        return
    

    
    def Nose_Hoover_thermostat(self, geom_num_list, element_list, new_g):#fixed volume
        new_g *= -1
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta * 0.5)

        self.momentum_list += new_g * self.delta_timescale * 0.5
        print(np.sum(np.abs(self.momentum_list)))
        tmp_list = []
        for i, elem in enumerate(element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem))
        
        new_geometry = geom_num_list + tmp_list
        
        tmp_value = 0.0
        
        for i, elem in enumerate(element_list):
            tmp_value += (np.sum(self.momentum_list[i]) ** 2 / atomic_mass(elem))
        Instantaneous_temperature = tmp_value / (self.g_value * self.Boltzmann_constant)
        print("Instantaneous_temperature: ",Instantaneous_temperature ," K")

        self.Instantaneous_temperatures_list.append(tmp_value / (self.g_value * self.Boltzmann_constant))
        self.zeta += self.delta_timescale * (tmp_value - self.g_value * self.Boltzmann_constant * self.initial_temperature) / self.Q_value
        
        #print(tmp_value, self.g_value * self.Boltzmann_constant * self.temperature)
        
        
        self.momentum_list += new_g * self.delta_timescale * 0.5
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta * 0.5)
        
        
        return new_geometry



class MD:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.Boltzmann_constant = 3.16681 * 10 ** (-6) # hartree/K
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #
        
        self.args = args #
        self.FC_COUNT = -1 # 

        self.initial_temperature = args.temperature
        self.num_of_trajectory = args.TRAJECTORY
        self.condition = args.condition
        self.momentum_list = None
        self.initial_pressure = args.pressure * 1000 * ( UnitValueLib().bohr2m ** 3  / UnitValueLib().hartree2j )

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

        
        self.Model_hess = None #
        self.Opt_params = None #
        self.DC_check_dist = 10.0#ang.
        self.unrestrict = args.unrestrict
        return
    
    def init_purtubation(self, geometry):
        
        addtional_momentum = np.random.rand(len(geometry), 3)
        
        self.momentum_list += addtional_momentum * self.Boltzmann_constant * self.initial_temperature 
       
        return

    def md_tblite(self):
        from tblite_calculation_tools import Calculation
        
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.args.usextb+"_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        temperature_list = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure)
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
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

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
            if iter % 10 != 0:
                shutil.rmtree(file_directory)
                 
            #---------------------------------------
            if iter == 0:
                self.init_purtubation(geom_num_list)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass

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
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            

            
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            

            #----------------------------

            #----------------------------
            
            new_geometry = TM.Nose_Hoover_thermostat(geom_num_list, element_list, B_g)

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
        
        
        
        #----------------------
        print("Complete...")
        return

    def md_psi4(self):
        from psi4_calculation_tools import Calculation
        
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure)
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
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_list, iter, electric_charge_and_multiplicity)
            
   
            
            self.Model_hess = SP.Model_hess
            if iter % 10 != 0:
                shutil.rmtree(file_directory)
                 
            #---------------------------------------
            if iter == 0:
                self.init_purtubation(geom_num_list)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass

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
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            

            
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            

            #----------------------------

            #----------------------------
            
            new_geometry = TM.Nose_Hoover_thermostat(geom_num_list, element_list, B_g)

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
    
    def md_pyscf(self):
        from pyscf_calculation_tools import Calculation
        
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure)
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
                         spin_multiplicity = self.spin_multiplicity,
                         electronic_charge = self.electronic_charge,
                         unrestrict = self.unrestrict)
        #----------------------------------
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

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
            if iter % 10 != 0:
                shutil.rmtree(file_directory)
                 
            #---------------------------------------
            if iter == 0:
                self.init_purtubation(geom_num_list)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass

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
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            

            
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            

            #----------------------------

            #----------------------------
            
            new_geometry = TM.Nose_Hoover_thermostat(geom_num_list, element_list, B_g)

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make_for_pyscf()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
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
    
    def run(self):
        
        for i in range(self.num_of_trajectory):
            print("trajectory :", i)
            if self.args.pyscf:
                self.md_pyscf()
            elif self.args.usextb != "None":
                self.md_tblite()
            else:
                self.md_psi4()
    
        print("All complete...")
        
        return