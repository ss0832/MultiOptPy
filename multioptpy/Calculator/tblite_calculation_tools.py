import glob
import os
import copy
import numpy as np
from abc import ABC, abstractmethod

try:
    from tblite.interface import Calculator
except ImportError:
    pass

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
from multioptpy.ModelHessian.o1numhess import O1NumHessCalculator

"""
GFN2-xTB(tblite)
J. Chem. Theory Comput. 2019, 15, 3, 1652–1671 
GFN1-xTB(tblite, dxtb)
J. Chem. Theory Comput. 2017, 13, 5, 1989–2009
"""

class Calculation:
    """
    Handles TBLite (xTB) calculation logic.
    Supports direct in-memory execution.
    """
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        
        # Configuration
        self.START_FILE = kwarg.get("START_FILE", None)
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY", "2GB")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL", None) # Used for method selection if needed
        self.FC_COUNT = kwarg.get("FC_COUNT", 1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", "./")
        self.Model_hess = kwarg.get("Model_hess", None)
        self.unrestrict = kwarg.get("unrestrict", False)
        self.dft_grid = kwarg.get("dft_grid", None)
        self.hessian_flag = False
        self.cpcm_solv_model = kwarg.get("cpcm_solv_model", None)
        self.alpb_solv_model = kwarg.get("alpb_solv_model", None)
        self.method = kwarg.get("method", "GFN2-xTB") # Default method
        self.verbosity = 0
        self.numerical_delivative_delta = 0.0001

    def _get_calculator(self, positions_bohr, element_numbers, charge, uhf):
        """Internal helper to create Calculator instance"""
        calc = Calculator(self.method, element_numbers, positions_bohr, charge=charge, uhf=uhf)
        calc.set("max-iter", 500) # Or dynamic based on atoms
        calc.set("verbosity", self.verbosity)
        
        if self.cpcm_solv_model is not None:
            calc.add("cpcm-solvation", self.cpcm_solv_model)
        if self.alpb_solv_model is not None:
            calc.add("alpb-solvation", self.alpb_solv_model)
            
        return calc

    def run_calculation(self, positions_bohr, element_number_list, charge_mult):
        """
        Execute TBLite calculation for a single geometry.
        
        Args:
            positions_bohr (np.ndarray): Coordinates in Bohr.
            element_number_list (np.ndarray): Array of atomic numbers.
            charge_mult (list): [charge, multiplicity].
            
        Returns:
            tuple: (energy, gradient)
        """
        charge = int(charge_mult[0])
        mult = int(charge_mult[1])
        # xTB logic: uhf is number of unpaired electrons (mult - 1)
        uhf = max(mult - 1, 0)
        
        # Override uhf if unrestrict flag is set but mult is 1 (though usually 0 for singlet)
        if self.unrestrict and uhf == 0:
             # Logic depends on specific needs, usually 0 is fine for RHF/RKS equivalent
             pass

        calc = self._get_calculator(positions_bohr, element_number_list, charge, uhf)
        
        # Execute
        res = calc.singlepoint()
        e = float(res.get("energy"))
        g = res.get("gradient") # Hartree/Bohr
        
        self.res = res # Store full result for later access to orbitals, etc.
        
        return e, g

    def numerical_hessian(self, geom_num_list, element_number_list, electric_charge_and_multiplicity):
        """
        Compute numerical Hessian using finite difference of gradients.
        """
        n_atoms = len(geom_num_list)
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
        
        charge = int(electric_charge_and_multiplicity[0])
        uhf = int(electric_charge_and_multiplicity[1]) - 1

        # Pre-calculation setup
        # Note: Optimization opportunity - parallelize this loop if possible in future
        count = 0
        for atom_num in range(n_atoms):
            for i in range(3): # x, y, z
                for atom_num_2 in range(n_atoms):
                    for j in range(3):
                        # Symmetric check to avoid double calculation
                        if count > 3 * atom_num_2 + j:
                            continue
                        
                        tmp_grad = []
                        for direction in [1, -1]:
                            # Perturb geometry
                            perturbed_geom = copy.copy(geom_num_list)
                            perturbed_geom[atom_num][i] += direction * self.numerical_delivative_delta
                            
                            calc = self._get_calculator(perturbed_geom, element_number_list, charge, uhf)
                            res = calc.singlepoint()
                            g = res.get("gradient")
                            tmp_grad.append(g[atom_num_2][j])
                        
                        # Central difference
                        val = (tmp_grad[0] - tmp_grad[1]) / (2 * self.numerical_delivative_delta)
                        hessian[3*atom_num+i][3*atom_num_2+j] = val
                        hessian[3*atom_num_2+j][3*atom_num+i] = val
                        
                count += 1
        return hessian

    def calc_exact_hess(self, positions_bohr, element_number_list, charge_mult):
        """
        Calculate Exact Hessian (Numerical) and project out TR modes.
        """
        # Compute raw numerical Hessian
        exact_hess = self.numerical_hessian(positions_bohr, element_number_list, charge_mult)
        np.save(self.BPA_FOLDER_DIRECTORY+"raw_hessian.npy", exact_hess)
        # Project out Translation/Rotation
        # Note: Calculationtools usually expects Bohr for projection if hessian is in au?
        # Re-checking previous context: `project_out_hess_tr_and_rot_for_coord` seems to handle it.
        # It takes `positions` which here are in Bohr.
        
        # Ensure element_number_list is a list for the tool if needed, or array
        # The original code passed `element_number_list.tolist()`
        elem_list_arg = element_number_list.tolist() if isinstance(element_number_list, np.ndarray) else element_number_list
        
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            exact_hess, elem_list_arg, positions_bohr, display_eigval=False
        )
        self.Model_hess = exact_hess
        return exact_hess

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
        """
        Legacy method for directory-based execution.
        """
        
        element_number_list = element_list
        finish_frag = False
        self.method = method # Update method from argument

        # Ensure element_number_list is numpy array of ints
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
                    positions = np.array(pos_ang, dtype="float64") / self.bohr2angstroms # Convert to Bohr
                else:
                    positions = np.array(geom_num_list, dtype="float64") / self.bohr2angstroms 
                    
                # Execute using new method
                e, g = self.run_calculation(positions, element_number_list, electric_charge_and_multiplicity)
                
       

                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.calc_exact_hess(positions, element_number_list, electric_charge_and_multiplicity)
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.calc_exact_hess(positions, element_number_list, electric_charge_and_multiplicity)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), positions, finish_frag 
            
        self.energy = e
        self.gradient = g
        self.coordinate = positions
        
        return e, g, positions, finish_frag

    def single_point_no_directory(self, positions, element_number_list, electric_charge_and_multiplicity, method):
        """
        Legacy method for direct execution.
        Wrapper around run_calculation.
        """
        self.method = method
        finish_frag = False
        
        # Ensure element numbers
        if isinstance(element_number_list[0], str):
            element_number_list = np.array([element_number(e) for e in element_number_list])
            
        try:
            e, g = self.run_calculation(positions, element_number_list, electric_charge_and_multiplicity)
            # Original code stored orbital info here.
        except Exception as error:
            print(error)
            finish_frag = True
            return np.array([0]), np.array([0]), finish_frag
            
        self.energy = e
        self.gradient = g
        return e, g, finish_frag


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


class TBLiteEngine(CalculationEngine):
    """TBLite (extended tight binding) calculation engine"""
    
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
        # Extract solvation models from config if they exist
        cpcm = getattr(config, 'cpcm_solv_model', None)
        alpb = getattr(config, 'alpb_solv_model', None)
        
        calc_instance = Calculation(
            START_FILE=config.init_input,
            N_THREAD=config.N_THREAD,
            SET_MEMORY=config.SET_MEMORY,
            FC_COUNT=config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY,
            Model_hess=config.model_hessian,
            unrestrict=config.unrestrict,
            cpcm_solv_model=cpcm,
            alpb_solv_model=alpb,
            method=config.usextb # "GFN2-xTB" etc.
        )
        
        # Prepare Element List once
        geometry_list_tmp, element_list_str, _ = xyz2list(file_list[0], None)
        element_number_list = np.array([element_number(e) for e in element_list_str], dtype="int")
        
        hess_count = 0

        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                
                # Load Geometry
                pos_ang, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions_bohr = np.array(pos_ang, dtype="float64") / config.bohr2angstroms
                
                # --- Execute Calculation ---
                e, g = calc_instance.run_calculation(
                    positions_bohr, 
                    element_number_list, 
                    electric_charge_and_multiplicity
                )
                # ---------------------------

             
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                geometry_num_list.append(positions_bohr) # Store in Bohr
                num_list.append(num)
                
                # Hessian Calculation
                if config.MFC_COUNT != -1 and optimize_num % config.MFC_COUNT == 0 and config.model_hessian.lower() == "o1numhess":
                    print(f" Calculating O1NumHess for image {num} using {config.model_hessian}...")
                    o1numhess = O1NumHessCalculator(calc_instance, 
                        element_list_str, 
                        electric_charge_and_multiplicity,
                        method=config.usextb)
                    seminumericalhessian = o1numhess.compute_hessian(pos_ang)
                    np.save(os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{hess_count}.npy"), seminumericalhessian)
                    hess_count += 1
                
                elif config.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    print(f"  Calculating Numerical Hessian for image {num}...")
                    
                    exact_hess = calc_instance.calc_exact_hess(
                        positions_bohr, 
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
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return (np.array(energy_list, dtype="float64"), 
                np.array(gradient_list, dtype="float64"), 
                np.array(geometry_num_list, dtype="float64"), 
                pre_total_velocity)
                
                
# Note: There but for the grace of God go I.