import glob
import os
import copy
import numpy as np

from multioptpy.SQM.sqm1.sqm1_core import SQM1Calculator
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
"""
Experimental semiempirical electronic structure approach inspired by GFN0-xTB (SQM1)

This module provides calculator utility helpers wrapping the Python implementation
of an experimental semiempirical electronic structure approach inspired by GFN0-xTB.
It mirrors the interface style of tblite_calculation_tools for convenience.
"""

class Calculation:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()
        self.bohr2angstroms = UVL.bohr2angstroms
        # Optional keys kept for interface parity; use .get to avoid KeyError
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
        # SQM1 specific calculator instance
        self.calculator = SQM1Calculator()

    def numerical_hessian(self, geom_num_list, element_list, total_charge):
        numerical_delivative_delta = 1.0e-4
        geom_num_list = np.array(geom_num_list, dtype="float64")
        n_atoms = len(geom_num_list)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        count = 0
        for a in range(n_atoms):
            for i in range(3):
                for b in range(n_atoms):
                    for j in range(3):
                        if count > 3*b + j:
                            continue
                        tmp_grad = []
                        for direction in [1, -1]:
                            shifted = geom_num_list.copy()
                            shifted[a, i] += direction * numerical_delivative_delta
                            grad = self.calculator.calculate_gradient(shifted, element_list, total_charge, method='analytical')
                            tmp_grad.append(grad[b, j])
                        val = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        hessian[3*a+i, 3*b+j] = val
                        hessian[3*b+j, 3*a+i] = val
                count += 1
        return hessian

    def exact_hessian(self, element_number_list, total_charge, positions):
        exact_hess = self.numerical_hessian(positions, element_number_list, total_charge)
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_number_list.tolist(), positions, display_eigval=False)
        self.Model_hess = exact_hess

    def single_point(self, file_directory, element_number_list, iter_index, electric_charge_and_multiplicity, geom_num_list=None):
        print("Warning: This function is not fully tested. Please do not use this method.")
        raise NotImplementedError("This method is not fully implemented and tested.")
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        finish_frag = False

        if isinstance(element_number_list[0], str):
            tmp = copy.copy(element_number_list)
            element_number_list = []
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

        for num, input_file in enumerate(file_list):
            if True:#try:
                if geom_num_list is None:
                    tmp_positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    tmp_positions = geom_num_list

                positions = np.array(tmp_positions, dtype="float64").reshape(-1, 3) / self.bohr2angstroms  # angstrom
                results = self.calculator.calculate_energy_and_gradient(positions, element_number_list, total_charge, gradient_method='analytical')
                e = results['total']
                g = results['gradient']  # Hartree/Bohr

                # Save orbital-like placeholders (not available; keep compatibility attributes if needed)
                self.energy = e
                self.gradient = g
                self.coordinate = positions
             
                if self.FC_COUNT == -1 or isinstance(iter_index, str):
                    if self.hessian_flag:
                        self.exact_hessian(element_number_list, total_charge, positions)
                elif iter_index % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_number_list, total_charge, positions)

          

            #except Exception as error:
            #    print(error)
            #    print("This molecule could not be optimized.")
            #    print("Input file: ", file_list, "\n")
            #    finish_frag = True
            #    return np.array([0]), np.array([0]), positions, finish_frag
       
        return e, g, positions, finish_frag

    def single_point_no_directory(self, positions, element_number_list, electric_charge_and_multiplicity):
        finish_frag = False
        if isinstance(element_number_list[0], str):
            tmp = copy.copy(element_number_list)
            element_number_list = []
            for elem in tmp:
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        try:
            positions = np.array(positions, dtype='float64')
            total_charge = int(electric_charge_and_multiplicity[0])
            results = self.calculator.calculate_energy_and_gradient(positions, element_number_list, total_charge, gradient_method='analytical')
            e = results['total']
            g = results['gradient']
            self.energy = e
            self.gradient = g
        except Exception as error:
            print(error)
            print("This molecule could not be optimized.")
            finish_frag = True
            return np.array([0]), np.array([0]), finish_frag
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


class SQM1Engine(CalculationEngine):
    """SQM1 calculation engine wrapping SQM1Calculator"""
    def __init__(self):
        self.calculator = SQM1Calculator()

    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []

        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype='int')

        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions = np.array(positions, dtype='float64').reshape(-1, 3)  # angstrom
                total_charge = int(electric_charge_and_multiplicity[0])
                results = self.calculator.calculate_energy_and_gradient(positions, element_number_list, total_charge, gradient_method='analytical')
                e = results['total']
                g = results['gradient']
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                geometry_num_list.append(positions)
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

        return (np.array(energy_list, dtype='float64'),
                np.array(gradient_list, dtype='float64'),
                np.array(geometry_num_list, dtype='float64'),
                pre_total_velocity)
