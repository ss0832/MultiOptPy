import glob
import os
import copy
import numpy as np
import torch
from abc import ABC, abstractmethod

try:
    import dxtb
    dxtb.timer.disable()
except ImportError:
    pass

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
from multioptpy.ModelHessian.o1numhess import O1NumHessCalculator

"""
ref:
dxtb: M. Friede, et al., J. Chem. Phys., 2024, 161, 062501. 
DOI: https://doi.org/10.1063/5.0216715
"""

class Calculation:
    """
    Handles DXTB (differentiable xTB) calculation logic.
    Supports direct in-memory execution using PyTorch tensors.
    """
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        
        # Configuration
        self.START_FILE = kwarg.get("START_FILE", None)
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY", "2GB")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL", None)
        self.FC_COUNT = kwarg.get("FC_COUNT", 1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", "./")
        self.Model_hess = kwarg.get("Model_hess", None)
        self.unrestrict = kwarg.get("unrestrict", False)
        self.dft_grid = kwarg.get("dft_grid", None)
        self.hessian_flag = False
        self.method = kwarg.get("method", "GFN1-xTB") # Default method
        
        # Internal state for calculator reuse if needed, though dxtb is often stateless per call or cheap to re-init
        self.calc = None 

    def _get_calculator(self, element_numbers):
        """Internal helper to create or retrieve Calculator instance"""
        # Element numbers need to be tensor
        if not isinstance(element_numbers, torch.Tensor):
            torch_element_numbers = torch.tensor(element_numbers, dtype=torch.long)
        else:
            torch_element_numbers = element_numbers

        max_scf_iteration = len(element_numbers) * 50 + 1000
        settings = {"maxiter": max_scf_iteration}

        if self.method == "GFN1-xTB":
            calc = dxtb.calculators.GFN1Calculator(torch_element_numbers, opts=settings)
        elif self.method == "GFN2-xTB":
            calc = dxtb.calculators.GFN2Calculator(torch_element_numbers, opts=settings)
        else:
            raise ValueError(f"Unknown dxtb method: {self.method}")
            
        return calc

    def run_calculation(self, positions_bohr, element_number_list, charge_mult):
        """
        Execute DXTB calculation for a single geometry.
        
        Args:
            positions_bohr (np.ndarray): Coordinates in Bohr.
            element_number_list (np.ndarray): Array of atomic numbers.
            charge_mult (list): [charge, multiplicity].
            
        Returns:
            tuple: (energy_hartree, gradient_hartree_bohr, calculator_instance, torch_pos)
        """
        # Convert inputs to Tensor
        torch_positions = torch.tensor(positions_bohr, requires_grad=True, dtype=torch.float32)
        
        # Get Calculator
        calc = self._get_calculator(element_number_list)
        
        charge = int(charge_mult[0])
        mult = int(charge_mult[1])
      
        
        spin_arg = mult if mult > 1 else None # Original logic only passed spin if > 1
        
        # Compute Energy
        pos = torch_positions.clone().requires_grad_(True)
        if spin_arg is not None:
             e = calc.get_energy(pos, chrg=charge, spin=spin_arg)
        else:
             e = calc.get_energy(pos, chrg=charge)
        
        calc.reset() # Important for dxtb to clear cache/graph if needed

        # Compute Forces (Gradient)
     
        pos_force = torch_positions.clone().requires_grad_(True)
        if spin_arg is not None:
             g_tensor = -1 * calc.get_forces(pos_force, chrg=charge, spin=spin_arg)
        else:
             g_tensor = -1 * calc.get_forces(pos_force, chrg=charge)
        
        calc.reset()
        
        # Detach to numpy
        e_np = e.to('cpu').detach().numpy().copy()
        g_np = g_tensor.to('cpu').detach().numpy().copy()
        
        return e_np, g_np, calc, torch_positions

    def calc_exact_hess(self, calc, torch_positions, element_number_list, charge_mult):
        """
        Calculate Exact Hessian using Autograd.
        """
        charge = int(charge_mult[0])
        mult = int(charge_mult[1])
        spin_arg = mult if mult > 1 else None
        
        pos = torch_positions.clone().requires_grad_(True)
        
        # Get Hessian
        if spin_arg is not None:
            exact_hess = calc.get_hessian(pos, chrg=charge, spin=spin_arg)
        else:
            exact_hess = calc.get_hessian(pos, chrg=charge)
            
        # Reshape
        n_atoms = len(element_number_list)
        exact_hess = exact_hess.reshape(3 * n_atoms, 3 * n_atoms)
        
        # Convert to numpy
        return_exact_hess = exact_hess.to('cpu').detach().numpy().copy()
        
        np.save(self.BPA_FOLDER_DIRECTORY+"raw_hessian.npy", return_exact_hess)
        # Project out TR
        # positions for projection: needs to be numpy array.
        # torch_positions is tensor.
        positions_np = torch_positions.detach().cpu().numpy()
        
        # element_number_list needs to be list for tool?
        elem_list_arg = element_number_list.tolist() if isinstance(element_number_list, (np.ndarray, torch.Tensor)) else element_number_list

        projected_hess = copy.copy(Calculationtools().project_out_hess_tr_and_rot_for_coord(
            return_exact_hess, elem_list_arg, positions_np, display_eigval=False
        ))
        
        self.Model_hess = copy.copy(projected_hess)
        calc.reset()
        
        return projected_hess

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
        """
        Legacy method for directory-based execution.
        """
        element_number_list = element_list 
        finish_frag = False
        self.method = method
        
        # Ensure element list format
        if isinstance(element_number_list[0], str):
            element_number_list = np.array([element_number(e) for e in element_number_list])
            
        try:
            os.mkdir(file_directory)
        except:
            pass

        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = glob.glob(file_directory+"/*_[0-9].xyz")

        e = 0.0
        g = np.zeros(1)
        positions = np.zeros(1)

        for num, input_file in enumerate(file_list):
            try:
                if geom_num_list is None:
                    pos_ang, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                    positions = np.array(pos_ang, dtype="float64") / self.bohr2angstroms # Bohr
                else:
                    positions = np.array(geom_num_list, dtype="float64") / self.bohr2angstroms # Bohr

                # Execute
                e, g, calc, torch_pos = self.run_calculation(positions, element_number_list, electric_charge_and_multiplicity)
                
                # Hessian
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.calc_exact_hess(calc, torch_pos, element_number_list, electric_charge_and_multiplicity)
                      
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.calc_exact_hess(calc, torch_pos, element_number_list, electric_charge_and_multiplicity)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), positions, finish_frag 
        
        self.energy = e
        self.gradient = g
        self.coordinate = positions
        
        return e, g, positions, finish_frag

    # Note: ir() method and others kept as is if needed, can be refactored similarly but single_point is main focus.
    def ir(self, geom_num_list, element_number_list, electric_charge_and_multiplicity, method):
        """IR spectrum calculation (kept largely as is but cleaned up imports/logic if needed)"""
        # ... (Implementation kept compatible)
        # For brevity, retaining original logic structure here implicitly or explicit below:
        torch_positions = torch.tensor(geom_num_list, requires_grad=True, dtype=torch.float32)
        if isinstance(element_number_list[0], str):
             element_number_list = np.array([element_number(e) for e in element_number_list])
        torch_element_number_list = torch.tensor(element_number_list)
        
        max_scf_iteration = len(element_number_list) * 50 + 1000
        ef = dxtb.components.field.new_efield(torch.tensor([0.0, 0.0, 0.0], requires_grad=True))
        settings = {"maxiter": max_scf_iteration}
        
        if method == "GFN1-xTB":
            calc = dxtb.calculators.GFN1Calculator(torch_element_number_list, opts=settings, interaction=[ef])
        elif method == "GFN2-xTB":
            calc = dxtb.calculators.GFN2Calculator(torch_element_number_list, opts=settings, interaction=[ef])
        else:
            raise ValueError("method error")

        charge = int(electric_charge_and_multiplicity[0])
        mult = int(electric_charge_and_multiplicity[1])
        spin = mult if mult > 1 else None # Same logic as above

        pos = torch_positions.clone().requires_grad_(True)
        if spin:
             res = calc.ir(pos, chrg=charge, spin=spin)
        else:
             res = calc.ir(pos, chrg=charge)
        
        # au_int = res.ints # Deprecated or specific version?
        # Assuming res has .ints and .freqs as per dxtb API
        try:
             au_int = res.ints
             res.use_common_units()
             common_freqs = res.freqs.cpu().detach().numpy().copy()
             au_int = au_int.cpu().detach().numpy().copy()
        except:
             common_freqs = np.zeros(1)
             au_int = np.zeros(1)

        return common_freqs, au_int


class CalculationEngine(ABC):
    """Base class for calculation engines"""
    
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        """Calculate energy and gradients"""
        pass
    
    def _get_file_list(self, file_directory):
        """Get list of input files"""
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) 
                   for i in range(1, 7)], [])
    
    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        """Process common visualization tasks"""
        try:
            if config.save_pict:
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                
                gradient_norm_list = [np.sqrt(np.linalg.norm(g)**2/(len(g)*3)) for g in gradient_list]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")


class DXTBEngine(CalculationEngine):
    """DXTB calculation engine"""
    
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        # Initialize Calculation Instance
        calc_instance = Calculation(
            START_FILE=config.init_input,
            N_THREAD=config.N_THREAD,
            SET_MEMORY=config.SET_MEMORY,
            FC_COUNT=config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY,
            Model_hess=config.model_hessian,
            unrestrict=config.unrestrict,
            method=config.usedxtb # "GFN1-xTB" or "GFN2-xTB"
        )
        
        # Parse Elements once
        geometry_list_tmp, element_list_str, _ = xyz2list(file_list[0], None)
        element_number_list = np.array([element_number(e) for e in element_list_str], dtype="int")
        
        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                # Parse Geometry
                pos_ang, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions_bohr = np.array(pos_ang, dtype="float64") / config.bohr2angstroms
                
                # --- Execute Calculation ---
                e, g, calc, torch_pos = calc_instance.run_calculation(
                    positions_bohr, 
                    element_number_list, 
                    electric_charge_and_multiplicity
                )
                # ---------------------------
                
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                geometry_num_list.append(positions_bohr) # Bohr
                num_list.append(num)
                
                # Hessian
                if config.MFC_COUNT != -1 and optimize_num % config.MFC_COUNT == 0 and config.model_hessian.lower() == "o1numhess":
                    print(f" Calculating O1NumHess for image {num} using {config.model_hessian}...")
                    o1numhess = O1NumHessCalculator(calc_instance, 
                        element_list_str, 
                        electric_charge_and_multiplicity,
                        method=config.usedxtb)
                    seminumericalhessian = o1numhess.compute_hessian(pos_ang)
                    np.save(os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{hess_count}.npy"), seminumericalhessian)
                    hess_count += 1
                
                elif config.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    print(f"  Calculating Autograd Hessian for image {num}...")
                    
                    exact_hess = calc_instance.calc_exact_hess(
                        calc, torch_pos, element_number_list, electric_charge_and_multiplicity
                    )
                    
                    np.save(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".npy", exact_hess)
                    hess_count += 1
                
            except Exception as error:
                print(f"Error in {input_file}: {error}")
                # calc.reset() # handled inside methods usually, but safe to ignore here as next iter creates fresh state or methods handle it
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)

        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64").tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return (np.array(energy_list, dtype="float64"), 
                np.array(gradient_list, dtype="float64"), 
                np.array(geometry_num_list, dtype="float64"), 
                pre_total_velocity)