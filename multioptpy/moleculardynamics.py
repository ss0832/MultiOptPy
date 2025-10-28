import sys
import os
import shutil
import copy
import glob
import datetime

import numpy as np

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import CalculationStructInfo
from multioptpy.Visualization.visualization import Graph
from multioptpy.fileio import FileIO
from multioptpy.Parameters.parameter import UnitValueLib, element_number, atomic_mass
from multioptpy.interface import force_data_parser
from multioptpy.Constraint.constraint_condition import shake_parser, SHAKE, ProjectOutConstrain
from multioptpy.Utils.pbc import apply_periodic_boundary_condition
from multioptpy.MD.thermostat import Thermostat


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
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        self.software_path_file = args.software_path_file
        self.dft_grid = int(args.dft_grid)
        self.sqm1 = args.sqm1
        self.sqm2 = args.sqm2
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
        if self.args.othersoft != "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().strftime("%Y_%m_%d"))+"/"+self.START_FILE[:-4]+"_MD_ASE_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])+"/"

        elif self.args.usextb == "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().strftime("%Y_%m_%d"))+"/"+self.START_FILE[:-4]+"_MD_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])+"/"
        else:
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().strftime("%Y_%m_%d"))+"/"+self.START_FILE[:-4]+"_MD_"+self.args.usextb+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2])+"/"
        
        xtb_method = None
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        if self.args.pyscf:
            from multioptpy.Calculator.pyscf_calculation_tools import Calculation
        elif self.args.sqm2:
            from multioptpy.Calculator.sqm2_calculation_tools import Calculation
            print("Use SQM2 potential.")
        
        elif self.args.sqm1:
            from multioptpy.Calculator.sqm1_calculation_tools import Calculation    
        
        elif self.args.othersoft != "None":
            if self.args.othersoft.lower() == "lj":
                from multioptpy.Calculator.lj_calculation_tools import Calculation
                print("Use Lennard-Jones cluster potential.")
            elif self.args.othersoft.lower() == "emt":
                from multioptpy.Calculator.emt_calculation_tools import Calculation
                print("Use EMT cluster potential.")
            elif self.args.othersoft.lower() == "tersoff":
                from multioptpy.Calculator.tersoff_calculation_tools import Calculation
                print("Use Tersoff cluster potential.")
                
            else:
                from multioptpy.Calculator.ase_calculation_tools import Calculation
                print("Use", self.args.othersoft)
                with open(self.BPA_FOLDER_DIRECTORY+"use_"+self.args.othersoft+".txt", "w") as f:
                    f.write(self.args.othersoft+"\n")
                    f.write(self.BASIS_SET+"\n")
                    f.write(self.FUNCTIONAL+"\n")
        else:
            if self.args.usextb != "None":
                from multioptpy.Calculator.tblite_calculation_tools import Calculation
                xtb_method = self.args.usextb
            else:
                from multioptpy.Calculator.psi4_calculation_tools import Calculation
                
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        self.tot_energy_list = []

        temperature_list = []
        force_data = force_data_parser(self.args)
        PC = ProjectOutConstrain(force_data["projection_constraint_condition_list"], force_data["projection_constraint_atoms"], force_data["projection_constraint_constant"])


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
                         excited_state = self.excited_state,
                         dft_grid=self.dft_grid,
                         ECP = self.ECP,
                         software_path_file = self.software_path_file
                         )
        SP.cpcm_solv_model = self.cpcm_solv_model
        SP.alpb_solv_model = self.alpb_solv_model

        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure, element_list=element_list)
        TM.delta_timescale = self.timestep
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
        if len(force_data["projection_constraint_condition_list"]) > 0:
            projection_constrain_flag = True
        else:
            projection_constrain_flag = False
        
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
                if projection_constrain_flag:
                    PC.initialize(geom_num_list)

            else:
                pass

            if projection_constrain_flag:
                B_g = PC.calc_project_out_grad(geom_num_list, B_g)
                g = PC.calc_project_out_grad(geom_num_list, g)


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
            
            if projection_constrain_flag:
                tmp_new_geometry = new_geometry / self.bohr2angstroms
                new_geometry = PC.adjust_init_coord(tmp_new_geometry) * self.bohr2angstroms    
                
            #----------------------------
            pre_B_g = B_g#Hartree/Bohr
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.print_geometry_list(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
           
            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        self.NUM_LIST = np.array(self.NUM_LIST, dtype=np.float64)
        #TM.delta_timescale * iter * self.time_atom_unit * 10 ** 15
        G.double_plot(self.NUM_LIST * TM.delta_timescale * self.time_atom_unit * 10 ** 15, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST * TM.delta_timescale * self.time_atom_unit * 10 ** 15, grad_list, file_directory, "", axis_name_1="STEP (fs)", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST * TM.delta_timescale * self.time_atom_unit * 10 ** 15, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_1="STEP (fs)", axis_name_2="temperature [K]", name="temperature")
        G.single_plot(self.NUM_LIST * TM.delta_timescale * self.time_atom_unit * 10 ** 15, self.tot_energy_list, file_directory, "", axis_name_1="STEP (fs)", axis_name_2="total energy [a.u.]", name="tot_energy")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
       
        FIO.make_traj_file()
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "maximum_value", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "minimum_value", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("STEP,energy[kcal/mol]\n")
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