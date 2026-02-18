import glob
import os
import copy
import numpy as np
import torch

# --- SQM2 ---
try:
    from multioptpy.SQM.sqm2.sqm2_core import SQM2Calculator
except ImportError:
    pass
# ---

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
from multioptpy.ModelHessian.o1numhess import O1NumHessCalculator

# --- Constants ---
ANG2BOHR = 1.8897261246257704
ANGSTROM_TO_BOHR = ANG2BOHR
BOHR_TO_ANGSTROM = 1.0 / ANG2BOHR
# ---

"""
Experimental semiempirical electronic structure approach inspired by GFN-xTB (SQM2)
"""

class Calculation:
    """
    Handles SQM2 calculation logic.
    Supports direct in-memory execution.
    """
    def __init__(self, **kwarg):
        if UnitValueLib is not None:
            UVL = UnitValueLib()
            self.bohr2angstroms = UVL.bohr2angstroms
        else:
            self.bohr2angstroms = BOHR_TO_ANGSTROM
            
        # Configuration
        self.START_FILE = kwarg.get("START_FILE")
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL")
        self.FC_COUNT = kwarg.get("FC_COUNT", -1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", "./")
        self.Model_hess = kwarg.get("Model_hess")
        self.unrestrict = kwarg.get("unrestrict", False)
        self.dft_grid = kwarg.get("dft_grid")
        self.hessian_flag = False
        
        self.device = kwarg.get("device", "cpu")
        self.dtype = kwarg.get("dtype", torch.float64)

    def run_calculation(self, positions_ang, element_number_list, charge_mult):
        """
        Execute SQM2 calculation for a single geometry.
        
        Args:
            positions_ang (np.ndarray): Coordinates in Angstrom.
            element_number_list (np.ndarray): Array of atomic numbers.
            charge_mult (list): [charge, multiplicity].
            
        Returns:
            tuple: (energy, gradient)
        """
        total_charge = int(charge_mult[0])
        spin = int(charge_mult[1]) - 1 # SQM2 uses multiplicity - 1 (unpaired electrons?)
        # Verify spin logic: Original code used `int(electric_charge_and_multiplicity[1]) - 1`
        
        # Initialize SQM2Calculator (Input: Angstrom)
        calc = SQM2Calculator(
            xyz=positions_ang, 
            element_list=element_number_list, 
            charge=total_charge, 
            spin=spin
        )
        
        # Get energy and gradient
        # Input: Angstrom, Output: (Hartree, Hartree/Bohr)
        e, g = calc.total_gradient(positions_ang)
        
        return e, g

    def calc_exact_hess(self, positions_ang, element_number_list, charge_mult):
        """
        Calculate exact Hessian using automatic differentiation (SQM2).
        """
        total_charge = int(charge_mult[0])
        spin = int(charge_mult[1]) - 1
        
        # Initialize SQM2Calculator
        calc = SQM2Calculator(
            xyz=positions_ang,
            element_list=element_number_list,
            charge=total_charge,
            spin=spin
        )
        
        # Calculate Hessian
        # Input: Angstrom, Output: (Hartree/Bohr^2)
        exact_hess_np = calc.total_hessian(positions_ang)
        
        # Project out translation and rotation
        # Calculationtools expects Bohr for projection usually if Hessian is AU?
        # Original code passed `positions` (Angstrom) to projection tool.
        # Let's check `project_out_hess_tr_and_rot_for_coord` implementation...
        # If it converts internally or expects Bohr, we need to be careful.
        # Assuming original code was correct in passing Angstrom if the tool handles conversion.
        # However, usually mass-weighted projection needs consistent units.
        # For safety, let's pass Angstrom as per original code, or check tool.
        # Original code: `Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess_np, ..., positions)`
        
        if Calculationtools is not None:
            # element_number_list needs to be list
            elem_list_arg = element_number_list.tolist() if isinstance(element_number_list, np.ndarray) else element_number_list
            
            exact_hess_np = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                exact_hess_np, elem_list_arg, positions_ang, display_eigval=False
            )
            
        self.Model_hess = exact_hess_np
        return exact_hess_np

    def single_point(self, file_directory, element_number_list, iter_index, electric_charge_and_multiplicity, method="", geom_num_list=None):
        """
        Legacy method for directory-based execution.
        """
        print("Warning: SQM2 is an experimental method and does not be suitable for production use.")
        finish_frag = False

        if isinstance(element_number_list[0], str):
            tmp = copy.copy(element_number_list)
            element_number_list = []
            if element_number is not None:
                for elem in tmp:
                    element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)

        try:
            os.mkdir(file_directory)
        except Exception:
            pass

        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = glob.glob(file_directory+"/*_[0-9].xyz")

        e = 0.0
        g = np.zeros(1)
        positions_bohr = np.zeros(1)

        for num, input_file in enumerate(file_list):
            try:
                if geom_num_list is None and xyz2list is not None:
                    tmp_positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                    positions = np.array(tmp_positions, dtype="float64").reshape(-1, 3) # Angstrom
                else:
                    # Assuming geom_num_list is Angstrom based on original usage
                    positions = np.array(geom_num_list, dtype="float64").reshape(-1, 3)

                # Execute
                e, g = self.run_calculation(positions, element_number_list, electric_charge_and_multiplicity)
                
                # Convert to Bohr for output
                positions_bohr = positions * ANGSTROM_TO_BOHR
                
                # Hessian
                if self.FC_COUNT == -1 or isinstance(iter_index, str):
                    if self.hessian_flag:
                        self.calc_exact_hess(positions, element_number_list, electric_charge_and_multiplicity)
                elif iter_index % self.FC_COUNT == 0 or self.hessian_flag:
                    self.calc_exact_hess(positions, element_number_list, electric_charge_and_multiplicity)

            except Exception as error:
                print(error)
                return np.array([0]), np.array([0]), positions_bohr, finish_frag
        
        self.energy = e
        self.gradient = g
        self.coordinate = positions_bohr # Bohr
        return e, g, positions_bohr, finish_frag

    def single_point_no_directory(self, positions, element_number_list, electric_charge_and_multiplicity):
        """
        Legacy direct execution wrapper.
        """
        finish_frag = False
        if isinstance(element_number_list[0], str):
            element_number_list = np.array([element_number(e) for e in element_number_list])
            
        try:
            positions = np.array(positions, dtype='float64') # Angstrom
            e, g = self.run_calculation(positions, element_number_list, electric_charge_and_multiplicity)
            self.energy = e
            self.gradient = g
        except Exception as error:
            print(error)
            finish_frag = True
            return np.array([0]), np.array([0]), finish_frag
        
        return e, g, finish_frag


class CalculationEngine(object):
    """Base class for calculation engines"""
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        raise NotImplementedError

    def _get_file_list(self, file_directory):
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])

    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        try:
            if getattr(config, 'save_pict', False):
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype='float64') * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                gradient_norm_list = [np.sqrt(np.linalg.norm(g)**2/(len(g)*3)) for g in gradient_list]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")


class SQM2Engine(CalculationEngine):
    """SQM2 calculation engine"""
    def __init__(self, param_file=None, device="cpu", dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []

        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        if xyz2list is None:
            raise ImportError("xyz2list is required for this method")
            
        # Initialize Calculation Instance
        calc_instance = Calculation(
            START_FILE=config.init_input,
            FC_COUNT=config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY,
            Model_hess=config.model_hessian,
            unrestrict=config.unrestrict,
            device=self.device,
            dtype=self.dtype
        )

        # Parse Elements once
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        element_number_list = []
        if element_number is not None:
            for elem in element_list:
                element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype='int')
        
        hess_count = 0

        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                # Parse Geometry
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions_ang = np.array(positions, dtype='float64').reshape(-1, 3)  # Angstrom
                
                # --- Execute Calculation ---
                e, g = calc_instance.run_calculation(
                    positions_ang, 
                    element_number_list, 
                    electric_charge_and_multiplicity
                )
                # ---------------------------
                
                print("\n")
                energy_list.append(e)       # Hartree
                gradient_list.append(g)     # Hartree/Bohr
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                geometry_num_list.append(positions_ang) # Angstrom
                num_list.append(num)
                
                # Hessian
                if config.MFC_COUNT != -1 and optimize_num % config.MFC_COUNT == 0 and config.model_hessian.lower() == "o1numhess":
                    print(f" Calculating O1NumHess for image {num} using {config.model_hessian}...")
                    o1numhess = O1NumHessCalculator(calc_instance, 
                        element_list, 
                        electric_charge_and_multiplicity,
                        method="")
                    seminumericalhessian = o1numhess.compute_hessian(positions_ang)
                    np.save(os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{hess_count}.npy"), seminumericalhessian)
                    hess_count += 1
                
                elif config.FC_COUNT == -1 or isinstance(optimize_num, str):
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    print(f"  Calculating Exact Hessian (SQM2) for image {num}...")
                    
                    exact_hess = calc_instance.calc_exact_hess(
                        positions_ang, 
                        element_number_list, 
                        electric_charge_and_multiplicity
                    )
                    
                    np.save(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".npy", exact_hess)
                    hess_count += 1
                    
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)

        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)

        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype='float64').tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype='float64')

        # (Hartree, Hartree/Bohr, Angstrom, velocity)
        return (np.array(energy_list, dtype='float64'),
                np.array(gradient_list, dtype='float64'),
                np.array(geometry_num_list, dtype='float64'),
                pre_total_velocity)