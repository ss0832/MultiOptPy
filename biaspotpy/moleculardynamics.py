import sys
import os
import shutil
import copy
import glob
import itertools
import datetime
import time

import numpy as np

from optimizer import CalculateMoveVector
from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib, element_number, atomic_mass
from interface import force_data_parser
from approx_hessian import ApproxHessian
from cmds_analysis import CMDSPathAnalysis
from constraint_condition import shake_parser, SHAKE
from pbc import apply_periodic_boundary_condition


class Thermostat:
    def __init__(self, momentum_list, temperature, pressure, element_list=[]):

        self.momentum_list = momentum_list #list
        
        self.temperature = temperature #K
        self.initial_temperature = temperature #K
        self.pressure = pressure * (3.39893 * 10 ** (-11)) #kPa -> a.u.
        self.initial_pressure = pressure * (3.39893 * 10 ** (-11)) #kPa -> a.u.

        self.Langevin_zeta = 0.01
        self.zeta = 0.0
        self.eta = 0.0
        self.scaling = 1.0
        self.Ps_momentum = 0.0
        
        self.g_value = len(momentum_list) * 3
        self.Q_value = 1e-1
        
        
        self.M_value = 1e+12
        self.Boltzmann_constant = 3.16681 * 10 ** (-6) # hartree/K
        self.delta_timescale = 1e-1
        self.volume = 1e-23 * (1/UnitValueLib().bohr2m) ** 3#m^3 -> Bohr^3
        
        # Nose-Hoover-chain
        self.Q_value_chain = [1.0, 2.0, 3.0, 6.0, 10.0, 20, 40, 50, 100, 200]#mass of thermostat 
        self.zeta_chain = [0.0 for i in range(len(self.Q_value_chain))]
        self.timestep = None
        
        self.Instantaneous_temperatures_list = []
        self.Instantaneous_momentum_list = []
        
        self.element_list = element_list
        return
    
    
    def calc_tot_kinetic_energy(self):
        tot_kinetic_ene = 0.0
        
        for i, elem in enumerate(self.element_list):
            tot_kinetic_ene += (np.sum(self.momentum_list[i] ** 2) /(2 * atomic_mass(elem)))
        self.tot_kinetic_ene = tot_kinetic_ene
        return tot_kinetic_ene
    
    def calc_inst_temperature(self):
        #temperature
        tot_kinetic_ene = self.calc_tot_kinetic_energy()
        Instantaneous_temperature = 2 * tot_kinetic_ene / (self.g_value * self.Boltzmann_constant)
        print("Instantaneous_temperature: ",Instantaneous_temperature ," K")

        self.Instantaneous_temperature = Instantaneous_temperature
        #-----------------
        return Instantaneous_temperature
    
    def add_inst_temperature_list(self):
        #self.add_inst_temperature_list()
        self.Instantaneous_temperatures_list.append(self.Instantaneous_temperature)
        
    
    def Nose_Hoover_thermostat(self, geom_num_list, new_g):#fixed volume #NVT ensemble
        new_g *= -1
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta * 0.5)

        self.momentum_list += new_g * self.delta_timescale * 0.5
        print("Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem))
        
        new_geometry = geom_num_list + tmp_list
        #------------
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #----------
        self.zeta += self.delta_timescale * (2 * self.tot_kinetic_ene - self.g_value * self.Boltzmann_constant * self.initial_temperature) / self.Q_value
        
        #print(tmp_value, self.g_value * self.Boltzmann_constant * self.temperature)
        
        
        self.momentum_list += new_g * self.delta_timescale * 0.5
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta * 0.5)
        
        
        return new_geometry

    def Nose_Hoover_chain_thermostat(self, geom_num_list, new_g):#fixed volume #NVT ensemble
        #ref. J. Chem. Phys. 97, 2635-2643 (1992)
        new_g *= -1
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta_chain[0] * 0.5)

        self.momentum_list += new_g * self.delta_timescale * 0.5
        print("Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
        
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem))
        
        new_geometry = geom_num_list + tmp_list
        #------------
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #----------
        self.zeta_chain[0] += self.delta_timescale * (2 * self.tot_kinetic_ene - self.g_value * self.Boltzmann_constant * self.initial_temperature) / self.Q_value_chain[0] -1* self.delta_timescale * (self.zeta_chain[0] * self.zeta_chain[1])
        
        for j in range(1, len(self.zeta_chain)-1):
            self.zeta_chain[j] += self.delta_timescale * (self.Q_value_chain[j-1]*self.zeta_chain[j-1]**2 - self.Boltzmann_constant * self.initial_temperature) / self.Q_value_chain[j] -1* self.delta_timescale * (self.zeta_chain[j] * self.zeta_chain[j+1])
        
        self.zeta_chain[-1] += self.delta_timescale * (self.Q_value_chain[-2]*self.zeta_chain[-2]**2 -1*self.Boltzmann_constant * self.initial_temperature) / self.Q_value_chain[-1]
        
        #print(tmp_value, self.g_value * self.Boltzmann_constant * self.temperature)
    
        
        self.momentum_list += new_g * self.delta_timescale * 0.5
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta_chain[0] * 0.5)
        print("zeta_list (Coefficient of friction): ", self.zeta_chain)    
        return new_geometry

    def Velocity_Verlet(self, geom_num_list, new_g, prev_g, iter):#NVE ensemble 
        tmp_new_g = copy.copy(-1*new_g)
        tmp_prev_g = copy.copy(-1*prev_g)

        self.momentum_list += ( tmp_new_g + tmp_prev_g ) * (self.delta_timescale) * 0.5 #+ third_term + fourth_term
        #-----------
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem) + self.delta_timescale ** 2 * tmp_new_g[i] / (2.0 * atomic_mass(elem)))
        new_geometry = geom_num_list + tmp_list
        #------------    
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #-------------
 
        return new_geometry
    

    
    def generate_normal_random_variables(self, n_variables):
        random_variables = []
        for _ in range(n_variables // 2):
            u1, u2 = np.random.rand(2)
            #Box-Muller method
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            random_variables.extend([z1, z2])
        
        if n_variables % 2 == 1:
            u1, u2 = np.random.rand(2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            random_variables.append(z1)
        
        return np.array([random_variables], dtype="float64")

    def calc_rand_moment_based_on_boltzman_const(self, random_variables):
        rand_moment = random_variables
        
        for i in range(len(self.element_list)):
            random_variables[i] *= np.sqrt(self.Boltzmann_constant * self.temperature / atomic_mass(self.element_list[i]))
        
        
        return rand_moment
    
    def calc_DRC_init_momentum(self, coord, cart_gradient):
        #dynamic reaction coordinate (DRC)
        #ref. J. Chem. Phys. 93, 5902–5911 (1990) etc.
        AH = ApproxHessian()
        hess = AH.main(coord, self.element_list, cart_gradient)
 
        eigenvalue, eigenvector = np.linalg.eig(hess)
        
        drc_momentum = eigenvector.T[0].reshape(len(self.element_list), 3)
        for i in range(len(self.element_list)):
            drc_momentum[i] *= atomic_mass(self.element_list[i])

        return drc_momentum
    
    def init_purtubation(self, geometry, B_e, B_g):
        random_variables = self.generate_normal_random_variables(len(self.element_list)*3).reshape(len(self.element_list), 3)
        random_moment_flag = True
        
        if random_moment_flag:
            addtional_velocity = self.calc_rand_moment_based_on_boltzman_const(random_variables) # velocity
            init_momentum = addtional_velocity * 0.0
            
            for i in range(len(self.element_list)):
                init_momentum[i] += addtional_velocity[i] * atomic_mass(self.element_list[i])
        else:
            init_momentum = self.calc_DRC_init_momentum(geometry, B_g)
        
        
        self.momentum_list += init_momentum
        self.init_energy = B_e
        #self.init_hamiltonian = B_e
        #for i, elem in enumerate(element_list):
        #    self.init_hamiltonian += (np.sum(self.momentum_list[i]) ** 2 / (2 * atomic_mass(elem)))
        return


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
        self.time_atom_unit = 2.419 * 10 ** (-17)
        self.args = args #
        self.FC_COUNT = -1 # 
        self.excited_state = args.excited_state
        self.initial_temperature = args.temperature
        self.num_of_trajectory = args.TRAJECTORY
        self.change_temperature = args.change_temperature
        self.momentum_list = None
        self.initial_pressure = args.pressure
        self.timestep = args.timestep #
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
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
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

        self.mdtype = args.mdtype
        self.Model_hess = None #
        self.Opt_params = None #
        self.DC_check_dist = 10.0#ang.
        self.unrestrict = args.unrestrict
        if len(args.constraint_condition) > 0:
            self.constraint_condition_list = shake_parser(args.constraint_condition)
        else:
            self.constraint_condition_list = []
            
        if len(args.periodic_boundary_condition) > 0:
            if len(args.periodic_boundary_condition.split(",")) == 3:
                self.pbc_box = np.array(args.periodic_boundary_condition.split(","), dtype="float64") / self.bohr2angstroms
            else:
                self.pbc_box = []
        else:
            self.pbc_box = []
        self.args = args
        
        if args.othersoft != "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_ASE_"+str(time.time()).replace(".","_")+"/"

        elif args.usextb == "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time()).replace(".","_")+"/"
        else:
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+args.usextb+"_"+str(time.time()).replace(".","_")+"/"
        self.tot_energy_list = []
        return
    


    def exec_md(self, TM, geom_num_list, prev_geom_num_list, B_g, B_e, pre_B_g, iter):
        if iter == 0 and len(self.constraint_condition_list) > 0:
            self.class_SHAKE = SHAKE(TM.delta_timescale, self.constraint_condition_list)
        if self.mdtype == "nosehoover" or self.mdtype == "nvt": 
            new_geometry = TM.Nose_Hoover_thermostat(geom_num_list, B_g)
        elif self.mdtype == "nosehooverchain": 
            new_geometry = TM.Nose_Hoover_chain_thermostat(geom_num_list, B_g)
        elif self.mdtype == "velocityverlet" or self.mdtype == "nve":
            new_geometry = TM.Velocity_Verlet(geom_num_list, B_g, pre_B_g, iter)
        else:
            print("Unexpected method.", self.mdtype)
            raise
        
        if iter > 0 and len(self.constraint_condition_list) > 0:
            
            new_geometry, tmp_momentum_list = self.class_SHAKE.run(new_geometry, prev_geom_num_list, TM.momentum_list, TM.element_list)
            TM.momentum_list = copy.copy(tmp_momentum_list)

        kinetic_ene = 0.0
        
        for i in range(len(geom_num_list)):
            kinetic_ene += np.sum(TM.momentum_list[i] ** 2) / (2 * atomic_mass(TM.element_list[i]))
        
        tot_energy = B_e + kinetic_ene
        print("hamiltonian :", tot_energy, "hartree")
        
        self.tot_energy_list.append(tot_energy)
        
        if len(self.pbc_box) > 0:
            new_geometry = apply_periodic_boundary_condition(new_geometry, TM.element_list, self.pbc_box)
        return new_geometry

    def md(self):
        xtb_method = None
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        if self.args.pyscf:
            from pyscf_calculation_tools import Calculation
        elif self.args.othersoft != "None":
            from ase_calculation_tools import Calculation
            print("Use", self.args.othersoft)
            with open(self.BPA_FOLDER_DIRECTORY+"use_"+self.args.othersoft+".txt", "w") as f:
                f.write(self.args.othersoft+"\n")
                f.write(self.BASIS_SET+"\n")
                f.write(self.FUNCTIONAL+"\n")
        else:
            if self.args.usextb != "None":
                from tblite_calculation_tools import Calculation
                xtb_method = self.args.usextb
            else:
                from psi4_calculation_tools import Calculation
                
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
       

        temperature_list = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        initial_geom_num_list = np.zeros((len(element_list), 3))
        
        self.Model_hess = np.eye(len(element_list*3))
         
        CalcBiaspot = BiasPotentialCalculation(self.BPA_FOLDER_DIRECTORY)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_g = []

        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
        pre_B_g = np.array(pre_B_g, dtype="float64")

        #-------------------------------------
        finish_frag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         BASIS_SET = self.BASIS_SET,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         unrestrict = self.unrestrict,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         software_type = self.args.othersoft,
                         spin_multiplicity = self.spin_multiplicity,
                         electronic_charge = self.electronic_charge,
                         excited_state = self.excited_state
                         )
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure, element_list=element_list)
        TM.timestep = self.timestep
        #-----------------------------------
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        #----------------------------------
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

        #----------------------------------
        ct_count = 0
        
        
        for iter in range(self.NSTEP):
            #-----------------------------
            if ct_count < len(self.change_temperature):
                if int(self.change_temperature[ct_count]) == iter:
                    TM.initial_temperature = float(self.change_temperature[ct_count+1])
                    ct_count += 2
                    
            
            #------------------------------
            
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# STEP. "+str(iter)+" ("+str(TM.delta_timescale * iter * self.time_atom_unit * 10 ** 15)+" fs)\n")
            #---------------------------------------
            
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_number_list, iter, electric_charge_and_multiplicity, force_data["xtb"])
            self.Model_hess = SP.Model_hess
            if finish_frag:
                break
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            

             
            #---------------------------------------
            if iter == 0:
                TM.init_purtubation(geom_num_list, B_e, B_g)
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
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"tot_energy_profile.csv","a") as f:
                    f.write("total energy [hartree] \n")
            else:
                with open(self.BPA_FOLDER_DIRECTORY+"tot_energy_profile.csv","a") as f:
                    f.write(str(self.tot_energy_list[-1])+"\n")
            
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            #----------------------------
            #----------------------------
 
            
            new_geometry = self.exec_md(TM, geom_num_list, pre_geom, B_g, B_e, pre_B_g, iter)

            if iter % 10 != 0:
                shutil.rmtree(file_directory)
            

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
            pre_B_g = B_g#Hartree/Bohr
           
            pre_geom = geom_num_list#Bohr
            if self.args.pyscf:
                geometry_list = FIO.make_geometry_list_2_for_pyscf(new_geometry*self.bohr2angstroms, element_list)
                file_directory = FIO.make_pyscf_input_file(geometry_list, iter+1)
            else:
                geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
                file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        G.single_plot(self.NUM_LIST, self.tot_energy_list, file_directory, "", axis_name_2="total energy [a.u.]", name="tot_energy")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        if self.args.pyscf:
            FIO.xyz_file_make_for_pyscf()
        else:
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
            self.md()

    
        print("All complete...")
        
        return