import os
import sys
import datetime
import glob
import numpy as np


from multioptpy.fileio import FileIO
from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.interface import force_data_parser

from multioptpy.OtherMethod.dimer import DimerMethod
from multioptpy.OtherMethod.newton_traj import NewtonTrajectory
from multioptpy.OtherMethod.addf import ADDFlikeMethod
from multioptpy.OtherMethod.twopshs import twoPSHSlikeMethod
from multioptpy.OtherMethod.elastic_image_pair import ElasticImagePair
from multioptpy.OtherMethod.modelfunction import ModelFunctionOptimizer
from multioptpy.OtherMethod.spring_pair_method import SpringPairMethod

class IEIPConfig:
    """
    Configuration class for Improved Elastic Image Pair (iEIP) method.
    
    References:
    - J. Chem. Theory. Comput. 2023, 19, 2410-2417
    - J. Comput. Chem. 2018, 39, 233–251 (DS-AFIR)
    """
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        
        # Unit conversion constants
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol
        
        # Displacement and convergence thresholds
        self.displacement_limit = 0.04  # Bohr
        self.maximum_ieip_disp = 0.2    # Bohr
        self.L_covergence = 0.03        # Bohr
        self.img_distance_convage_criterion = 0.15  # Bohr
        
        # Convergence thresholds
        self.MAX_FORCE_THRESHOLD = 0.0006
        self.RMS_FORCE_THRESHOLD = 0.0004
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0030
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0020
        
        # Iteration limits
        self.microiterlimit = int(args.NSTEP)
        self.microiter_num = args.microiter
        
        # Electronic state configuration
        self.initial_excite_state = args.excited_state[0]
        self.final_excite_state = args.excited_state[1]
        self.excite_state_list = args.excited_state
        
        # Charge and multiplicity
        self.init_electric_charge_and_multiplicity = [int(args.electronic_charge[0]), int(args.spin_multiplicity[0])]
        self.final_electric_charge_and_multiplicity = [int(args.electronic_charge[1]), int(args.spin_multiplicity[1])]
        self.electric_charge_and_multiplicity_list = []
        for i in range(len(args.electronic_charge)):
            self.electric_charge_and_multiplicity_list.append([int(args.electronic_charge[i]), int(args.spin_multiplicity[i])])
        
        # Solvation models
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        
        # Computation resources
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.START_FILE = args.INPUT + "/"
        
        # Quantum chemistry settings
        self.BASIS_SET = args.basisset
        self.FUNCTIONAL = args.functional
        self.usextb = args.usextb
        self.usedxtb = args.usedxtb
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        
        # Validate sub-basis set inputs
        if len(args.sub_basisset) % 2 != 0:
            print("invalid input (-sub_bs)")
            sys.exit(0)
        
        # Configure basis sets
        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET)
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
            else:
                self.SUB_BASIS_SET = {"default": self.BASIS_SET}
        else:  # psi4
            self.SUB_BASIS_SET = ""
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET += "\nassign " + str(self.BASIS_SET) + "\n"
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign " + args.sub_basisset[2*j] + " " + args.sub_basisset[2*j+1] + "\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
        
        # Validate effective core potential inputs
        if len(args.effective_core_potential) % 2 != 0:
            print("invalid input (-ecp)")
            sys.exit(0)
        
        # Configure effective core potentials
        if args.pyscf:
            self.ECP = {}
            if len(args.effective_core_potential) > 0:
                for j in range(int(len(args.effective_core_potential)/2)):
                    self.ECP[args.effective_core_potential[2*j]] = args.effective_core_potential[2*j+1]
        else:
            self.ECP = ""
        
        # Other settings
        self.othersoft = args.othersoft
        self.software_path_file = args.software_path_file
        self.basic_set_and_function = args.functional + "/" + args.basisset
        self.force_data = force_data_parser(args)
        
        # Set up output directory
        if self.othersoft != "None":
            self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + args.othersoft + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
        elif args.sqm2:
            self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_SQM2_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
        
        elif args.sqm1:
            self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_SQM1_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
        elif args.usextb == "None" and args.usedxtb == "None":
            self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + self.basic_set_and_function.replace("/", "_") + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
        else:
            if args.usedxtb != "None":
                self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + self.usedxtb + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
                self.force_data["xtb"] = self.usedxtb
            else:
                self.iEIP_FOLDER_DIRECTORY = args.INPUT + "_iEIP_" + self.usextb + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]) + "/"
                self.force_data["xtb"] = self.usextb
        
        # Optimization parameters
        self.args = args
        self.BETA = args.BETA
        self.spring_const = 1e-8
        self.unrestrict = args.unrestrict
        self.mf_mode = args.model_function_mode
        self.FC_COUNT = int(args.calc_exact_hess)
        self.dft_grid = int(args.dft_grid)
        
        # Set config for Newton Trajectory Method
        self.use_gnt = getattr(args, 'use_gnt', False)
        self.gnt_step_len = getattr(args, 'gnt_step_len', 0.5)
        self.gnt_rms_thresh = getattr(args, 'gnt_rms_thresh', 1.7e-3)
        self.gnt_vec = getattr(args, 'gnt_vec', None)
        self.gnt_microiter = getattr(args, 'gnt_microiter', 100)
        
        # Set config for ADDF-like method
        self.use_addf = getattr(args, 'use_addf', False)
        self.addf_step_num = getattr(args, 'addf_step_num', 300)
        self.nadd = getattr(args, 'number_of_add', 5)
        self.addf_step_size = getattr(args, 'addf_step_size', 0.05)

        # Set config for 2PSHS-like method
        self.use_2pshs = getattr(args, 'use_2pshs', False)
        self.twoPshs_step_num = getattr(args, 'twoPshs_step_num', 300)
        self.twoPshs_step_size = getattr(args, 'twoPshs_step_size', 0.05)

        # Set config for Dimer Method
        self.use_dimer = getattr(args, 'use_dimer', False)
        self.dimer_separation = getattr(args, 'dimer_separation', 0.0001)
        self.dimer_trial_angle = getattr(args, 'dimer_trial_angle', np.pi / 32.0)
        self.dimer_max_iterations = getattr(args, 'dimer_max_iterations', 1000)

        # Add config flag check if needed
        self.use_spm = getattr(args, 'use_spm', False)

        # Create output directory
        os.mkdir(self.iEIP_FOLDER_DIRECTORY)
    
    def save_input_data(self):
        """Save the input configuration data to a file"""
        with open(self.iEIP_FOLDER_DIRECTORY + "input.txt", "w") as f:
            f.write(str(vars(self.args)))
        return
    
        
class iEIP:
    """
    Main class for Improved Elastic Image Pair (iEIP) method.
    
    Manages the overall optimization process by delegating to specialized
    component classes for configuration, elastic image pair optimization,
    model function optimization.
    
    
    References:
    - J. Chem. Theory. Comput. 2023, 19, 2410-2417
    - J. Comput. Chem. 2018, 39, 233–251 (DS-AFIR)
    """
    def __init__(self, args):
        self.config = IEIPConfig(args)
        self.elastic_image_pair = ElasticImagePair(self.config)
        self.model_function_optimizer = ModelFunctionOptimizer(self.config)
        self.newton_trajectory = NewtonTrajectory(self.config)
        self.addf_like_method = ADDFlikeMethod(self.config)
        self.twoPshs = twoPSHSlikeMethod(self.config)
        self.dimer_method = DimerMethod(self.config)
        self.spring_pair_method = SpringPairMethod(self.config)
        
        

    def optimize(self):
        """Load calculation modules based on configuration and run optimization"""
        if self.config.othersoft != "None":
            if self.config.othersoft.lower() == "lj":
                from multioptpy.Calculator.lj_calculation_tools import Calculation
                print("Use Lennard-Jones cluster potential.")
            elif self.config.othersoft.lower() == "emt":
                from multioptpy.Calculator.emt_calculation_tools import Calculation
                print("Use EMT cluster potential.")
            elif self.config.othersoft.lower() == "tersoff":
                from multioptpy.Calculator.tersoff_calculation_tools import Calculation
                print("Use Tersoff cluster potential.")
            else:
                print("Use", self.config.othersoft)
                with open(self.config.iEIP_FOLDER_DIRECTORY + "use_" + self.config.othersoft + ".txt", "w") as f:
                    f.write(self.config.othersoft + "\n")
                    f.write(self.config.BASIS_SET + "\n")
                    f.write(self.config.FUNCTIONAL + "\n")
                from multioptpy.Calculator.ase_calculation_tools import Calculation
        elif self.config.args.sqm2:
            from multioptpy.Calculator.sqm2_calculation_tools import Calculation
            print("Use SQM2 potential.")
        
        elif self.config.args.sqm1:
            from multioptpy.Calculator.sqm1_calculation_tools import Calculation
        elif self.config.args.pyscf:
            from multioptpy.Calculator.pyscf_calculation_tools import Calculation
        elif self.config.args.usextb != "None" and self.config.args.usedxtb == "None":
            from multioptpy.Calculator.tblite_calculation_tools import Calculation
        elif self.config.args.usedxtb != "None" and self.config.args.usextb == "None":
            from multioptpy.Calculator.dxtb_calculation_tools import Calculation
        else:
            from multioptpy.Calculator.psi4_calculation_tools import Calculation
        
        file_path_list = glob.glob(self.config.START_FILE+"*_[A-Z].xyz")
        FIO_img_list = []

        for file_path in file_path_list:
            FIO_img_list.append(FileIO(self.config.iEIP_FOLDER_DIRECTORY, file_path))

        geometry_list_list = []
        element_list_list = []
        electric_charge_and_multiplicity_list = []

        for i in range(len(FIO_img_list)):
            geometry_list, element_list, electric_charge_and_multiplicity = FIO_img_list[i].make_geometry_list(
                self.config.electric_charge_and_multiplicity_list[i])
            
            if self.config.args.pyscf:
                electric_charge_and_multiplicity = [self.config.electronic_charge[i], 
                                                 self.config.spin_multiplicity[i]]
            
            geometry_list_list.append(geometry_list)
            element_list_list.append(element_list)
            electric_charge_and_multiplicity_list.append(electric_charge_and_multiplicity)
            
        # Save input configuration data
        self.config.save_input_data()
        
        # Initialize calculation objects
        SP_list = []
        file_directory_list = []
        for i in range(len(FIO_img_list)):
            SP_list.append(Calculation(
                START_FILE = self.config.START_FILE,
                N_THREAD = self.config.N_THREAD,
                SET_MEMORY = self.config.SET_MEMORY,
                FUNCTIONAL = self.config.FUNCTIONAL,
                FC_COUNT = self.config.FC_COUNT,
                BPA_FOLDER_DIRECTORY = self.config.iEIP_FOLDER_DIRECTORY,
                Model_hess = np.eye(3*len(geometry_list_list[i])),
                unrestrict = self.config.unrestrict, 
                BASIS_SET = self.config.BASIS_SET,
                SUB_BASIS_SET = self.config.SUB_BASIS_SET,
                electronic_charge = self.config.electronic_charge[i] or electric_charge_and_multiplicity_list[i][0],
                spin_multiplicity = self.config.spin_multiplicity[i] or electric_charge_and_multiplicity_list[i][1],
                excited_state = self.config.excite_state_list[i],
                dft_grid = self.config.dft_grid,
                ECP = self.config.ECP,
                software_type = self.config.othersoft,
                software_path_file = self.config.software_path_file
            ))

            SP_list[i].cpcm_solv_model = self.config.cpcm_solv_model
            SP_list[i].alpb_solv_model = self.config.alpb_solv_model
            
            file_directory = FIO_img_list[i].make_psi4_input_file(geometry_list_list[i], 0)
            file_directory_list.append(file_directory)
        
        # Run optimization with appropriate method
        if self.config.mf_mode != "None":
            self.model_function_optimizer.model_function_optimization(
                file_directory_list, SP_list, element_list_list, 
                self.config.electric_charge_and_multiplicity_list, FIO_img_list)
        elif self.config.use_gnt:
            self.newton_trajectory.main(file_directory_list[0], file_directory_list[1], 
                SP_list[0], SP_list[1], element_list_list[0], 
                self.config.electric_charge_and_multiplicity_list[0], 
                self.config.electric_charge_and_multiplicity_list[1], 
                FIO_img_list[0], FIO_img_list[1])
        elif self.config.use_addf:
            self.addf_like_method.run(file_directory_list[0],
                SP_list[0],
                self.config.electric_charge_and_multiplicity_list[0], 
                FIO_img_list[0])
        elif self.config.use_2pshs:
            self.twoPshs.run(file_directory_list[0],
                SP_list[0], SP_list[1],
                self.config.electric_charge_and_multiplicity_list[0], 
                FIO_img_list[0], FIO_img_list[1])
        elif self.config.use_dimer:
            self.dimer_method.run(file_directory_list[0],
                SP_list[0],
                self.config.electric_charge_and_multiplicity_list[0], 
                FIO_img_list[0])
        elif getattr(self.config, 'use_spm', False):
            print("Using Spring Pair Method (SPM)")
            self.spring_pair_method.iteration(
                file_directory_list[0],
                SP_list[0], element_list_list[0], 
                self.config.electric_charge_and_multiplicity_list[0], 
                FIO_img_list[0])
        else:
            self.elastic_image_pair.iteration(
                file_directory_list[0], file_directory_list[1], 
                SP_list[0], SP_list[1], element_list_list[0], 
                self.config.electric_charge_and_multiplicity_list[0], 
                self.config.electric_charge_and_multiplicity_list[1], 
                FIO_img_list[0], FIO_img_list[1])
        
        return

    def run(self):
        """Run the optimization process and display completion message"""
        self.optimize()
        print("completed...")
        return