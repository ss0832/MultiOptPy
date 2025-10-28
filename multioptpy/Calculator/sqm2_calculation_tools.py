import glob
import os
import copy
import numpy as np
import torch

# --- SQM2 ---
from multioptpy.SQM.sqm2.sqm2_core import SQM2Calculator
# ---

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer

# --- Constants (from SQM2Calculator reference) ---
ANG2BOHR = 1.8897261246257704
ANGSTROM_TO_BOHR = ANG2BOHR
BOHR_TO_ANGSTROM = 1.0 / ANG2BOHR
# ---

"""
Experimental semiempirical electronic structure approach inspired by GFN-xTB (SQM2)

This module provides calculator utility helpers wrapping the Python implementation
of an experimental semiempirical electronic structure approach (SQM2).
"""

class Calculation:
    def __init__(self, **kwarg):
        if UnitValueLib is not None:
            UVL = UnitValueLib()
            self.bohr2angstroms = UVL.bohr2angstroms
        else:
            self.bohr2angstroms = BOHR_TO_ANGSTROM
            
        # Optional keys
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
        
        # The main calculator instance will be created on-the-fly
        # as it requires geometry (xyz) upon initialization.
        self.calculator = None
        
    def numerical_hessian(self, geom_num_list, element_list, total_charge):
        """
        Calculate numerical Hessian using finite differences of gradients (SQM2).
        
        Args:
            geom_num_list: Atomic positions in Angstrom (n_atoms, 3)
            element_list: Atomic numbers (n_atoms,)
            total_charge: Total molecular charge
            
        Returns:
            Hessian matrix (3*n_atoms, 3*n_atoms) in Hartree/Bohr^2
        """
        numerical_delivative_delta = 1.0e-4  # in Angstrom
        geom_num_list = np.array(geom_num_list, dtype="float64")
        n_atoms = len(geom_num_list)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        count = 0
        
        # SQM2Calculator also requires spin (assuming 0 for singlet)
        spin = 0 
        
        for a in range(n_atoms):
            for i in range(3):
                for b in range(n_atoms):
                    for j in range(3):
                        if count > 3*b + j:
                            continue
                        tmp_grad = []
                        for direction in [1, -1]:
                            shifted = geom_num_list.copy() # Angstrom
                            shifted[a, i] += direction * numerical_delivative_delta
                            
                            # --- Get SQM2 gradient ---
                            # Re-initialize SQM2Calculator (required as xyz is in __init__)
                            calc = SQM2Calculator(
                                xyz=shifted, 
                                element_list=element_list, 
                                charge=total_charge, 
                                spin=spin
                            )
                            
                            # .total_gradient returns (Energy, Gradient)
                            # Input: Angstrom, Output: (Hartree, Hartree/Bohr)
                            _, grad_np = calc.total_gradient(shifted)
                            
                            tmp_grad.append(grad_np[b, j])
                            
                        # Finite difference
                        # val = (Hartree/Bohr) / Angstrom
                        val = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        
                        # Convert units to Hartree/Bohr^2
                        hessian[3*a+i, 3*b+j] = val * ANGSTROM_TO_BOHR
                        hessian[3*b+j, 3*a+i] = val * ANGSTROM_TO_BOHR
                count += 1
        return hessian


    def exact_hessian(self, element_number_list, total_charge, positions):
        """
        Calculate exact Hessian using automatic differentiation (SQM2).
        
        Args:
            element_number_list: Atomic numbers (n_atoms,)
            total_charge: Total molecular charge
            positions: Atomic positions in Angstrom (n_atoms, 3)
            
        Returns:
            Projected Hessian matrix (3*n_atoms, 3*n_atoms) in Hartree/Bohr^2
        """
        
        # Assume spin=0 (singlet)
        spin = 0
        
        # Initialize SQM2Calculator (Input: Angstrom)
        calc = SQM2Calculator(
            xyz=positions,
            element_list=element_number_list,
            charge=total_charge,
            spin=spin
        )
        
        # Calculate Hessian
        # Input: Angstrom, Output: (Hartree/Bohr^2)
        exact_hess_np = calc.total_hessian(positions)
        
        # Project out translation and rotation
        if Calculationtools is not None:
            exact_hess_np = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                exact_hess_np, element_number_list.tolist(), positions, display_eigval=False
            )
        self.Model_hess = exact_hess_np


    def single_point(self, file_directory, element_number_list, iter_index, electric_charge_and_multiplicity, method="", geom_num_list=None):
        """
        Calculate single point energy and gradient (SQM2).
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

        total_charge = int(electric_charge_and_multiplicity[0])
        # SQM2 EHTCalculator requires spin (multiplicity - 1)
        spin = int(electric_charge_and_multiplicity[1]) - 1 

        for num, input_file in enumerate(file_list):
            try:
                if geom_num_list is None and xyz2list is not None:
                    tmp_positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    tmp_positions = geom_num_list

                positions = np.array(tmp_positions, dtype="float64").reshape(-1, 3)  # Angstrom
                
                # --- SQM2 Energy and Gradient Calculation ---
                # Initialize SQM2Calculator (Input: Angstrom)
                calc = SQM2Calculator(
                    xyz=positions,
                    element_list=element_number_list,
                    charge=total_charge,
                    spin=spin
                )
                
                # Get energy and gradient
                # Input: Angstrom, Output: (Hartree, Hartree/Bohr)
                e, g = calc.total_gradient(positions)
                
                # Output coordinates are in Bohr
                positions_bohr = positions * ANGSTROM_TO_BOHR
                
                S_mat = calc.get_overlap_matrix()  # Get overlap matrix S (torch.Tensor)
                natom = len(element_number_list)
                if S_mat is not None and natom <= 10:
                    print("Overlap matrix S:")
                    
                    for i in range(len(S_mat)):
                        print(" ".join([f"{S_mat[i, j].item():9.6f}" for j in range(len(S_mat))]))

                # Get EHT MO energies and coefficients
                #mo_energies = calc.get_eht_mo_energy().detach().cpu().numpy()
                #mo_coefficients = calc.get_eht_mo_coeff().detach().cpu().numpy()
                #if natom <= 10:
                #    print("Molecular Orbital Energies (Hartree):")
                #    for i, ene in enumerate(mo_energies):
                #        print(f"MO {i+1:2d}: {ene:12.6f} Hartree")
                #    print("Molecular Orbital Coefficients:")
                #    for i in range(len(mo_coefficients)):
                #        print(f"MO {i+1:2d}:"+" ".join([f"{mo_coefficients[i, j]:9.6f}" for j in range(len(mo_coefficients))]))
                    print("Notice: This output is displayed only for small molecules (n_atoms <= 10).")
                # Save results
                self.energy = e
                self.gradient = g
                self.coordinate = positions_bohr # Bohr
               
                if self.FC_COUNT == -1 or isinstance(iter_index, str):
                    if self.hessian_flag:
                        self.exact_hessian(element_number_list, total_charge, positions) # Angstrom
                elif iter_index % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_number_list, total_charge, positions) # Angstrom

            except Exception as error:
                print(error)
             
                return np.array([0]), np.array([0]), positions, finish_frag
        
        # e (Hartree), g (Hartree/Bohr), positions_bohr (Bohr)
        return e, g, positions_bohr, finish_frag

    def single_point_no_directory(self, positions, element_number_list, electric_charge_and_multiplicity):
        """
        Calculate single point energy and gradient without file I/O (SQM2).
        
        Args:
            positions: Atomic positions in Angstrom (n_atoms, 3)
        """
        finish_frag = False
        if isinstance(element_number_list[0], str):
            tmp = copy.copy(element_number_list)
            element_number_list = []
            if element_number is not None:
                for elem in tmp:
                    element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        try:
            positions = np.array(positions, dtype='float64') # Angstrom
            total_charge = int(electric_charge_and_multiplicity[0])
            spin = int(electric_charge_and_multiplicity[1]) - 1

            # --- SQM2 Energy and Gradient Calculation ---
            calc = SQM2Calculator(
                xyz=positions,
                element_list=element_number_list,
                charge=total_charge,
                spin=spin
            )
            # Input: Angstrom, Output: (Hartree, Hartree/Bohr)
            e, g = calc.total_gradient(positions) 
            
            self.energy = e
            self.gradient = g
        except Exception as error:
            print(error)
            print("This molecule could not be optimized.")
            finish_frag = True
            return np.array([0]), np.array([0]), finish_frag
        
        # e (Hartree), g (Hartree/Bohr)
        return e, g, finish_frag


class CalculationEngine:
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
    """SQM2 calculation engine wrapping SQM2Calculator"""
    def __init__(self, param_file=None, device="cpu", dtype=torch.float64):
        # SQM2Calculator holds parameters internally, not needed here
        self.device = device
        self.dtype = dtype
        # SQM2Calculator is instantiated in the calculate method as it requires xyz
        
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
            
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        element_number_list = []
        if element_number is not None:
            for elem in element_list:
                element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype='int')

        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions = np.array(positions, dtype='float64').reshape(-1, 3)  # Angstrom
                total_charge = int(electric_charge_and_multiplicity[0])
                spin = int(electric_charge_and_multiplicity[1]) - 1
                
                # --- SQM2 Energy and Gradient Calculation ---
                calc = SQM2Calculator(
                    xyz=positions,
                    element_list=element_number_list,
                    charge=total_charge,
                    spin=spin
                )
                # Input: Angstrom, Output: (Hartree, Hartree/Bohr)
                e, g = calc.total_gradient(positions) 
                
                print("\n")
                energy_list.append(e)       # Hartree
                gradient_list.append(g)     # Hartree/Bohr
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                geometry_num_list.append(positions) # Angstrom
                num_list.append(num)
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