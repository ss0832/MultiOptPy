import glob
import os
import numpy as np
from abc import ABC, abstractmethod

try:
    from ase import Atoms
    from ase.vibrations import Vibrations
    from gpaw import GPAW
except ImportError:
    print("ASE or GPAW is not installed. Please install them to use this module.")

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, number_element
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer

class Calculation:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2eV = UVL.hartree2eV

        self.START_FILE = kwarg["START_FILE"]
        self.N_THREAD = kwarg["N_THREAD"]
        self.SET_MEMORY = kwarg["SET_MEMORY"]
        self.FUNCTIONAL = kwarg["FUNCTIONAL"]
        self.FC_COUNT = kwarg["FC_COUNT"]
        self.BPA_FOLDER_DIRECTORY = kwarg["BPA_FOLDER_DIRECTORY"]
        self.Model_hess = kwarg["Model_hess"]
        self.hessian_flag = False

    def calc_exact_hess(self, atom_obj, positions, element_list):
        vib = Vibrations(atom_obj, delta=0.001)
        vib.run()
        result_vib = vib.get_vibrations()
        exact_hess = result_vib.get_hessian_2d()
        vib.clean()
        exact_hess = exact_hess / self.hartree2eV * (self.bohr2angstroms ** 2)
        if type(element_list[0]) is str:
            exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, positions)
        else:
            exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, [number_element(elem_num) for elem_num in element_list], positions)
        self.Model_hess = exact_hess
        return exact_hess

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass

        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = glob.glob(file_directory + "/*_[0-9].xyz")

        for num, input_file in enumerate(file_list):
            try:
                if geom_num_list is None:
                    positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    positions = geom_num_list

                positions = np.array(positions, dtype="float64")
                atom_obj = Atoms(element_list, positions)
                atom_obj.calc = GPAW(mode='lcao', xc=self.FUNCTIONAL, txt=None) # You can configure GPAW parameters as needed

                e = atom_obj.get_potential_energy(apply_constraint=False) / self.hartree2eV
                g = -1 * atom_obj.get_forces(apply_constraint=False) * self.bohr2angstroms / self.hartree2eV

                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        _ = self.calc_exact_hess(atom_obj, positions, element_list)
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    _ = self.calc_exact_hess(atom_obj, positions, element_list)
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), np.array([0]), finish_frag

        positions /= self.bohr2angstroms
        self.energy = e
        self.gradient = g
        self.coordinate = positions

        return e, g, positions, finish_frag

class GPAWEngine(ABC):
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        pass

    def _get_file_list(self, file_directory):
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz")))
                    for i in range(1, 7)], [])

    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        try:
            if config.save_pict:
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                gradient_norm_list = [np.sqrt(np.linalg.norm(g) ** 2 / (len(g) * 3)) for g in gradient_list]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")

class GPAWASEEngine(GPAWEngine):
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        num_list = []
        delete_pre_total_velocity = []

        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)

        if not file_list:
            print("No input files found in directory.")
            return (np.array([], dtype="float64"),
                    np.array([], dtype="float64"),
                    np.array([], dtype="float64"),
                    pre_total_velocity)

        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)

        hess_count = 0
        for num, input_file in enumerate(file_list):
            try:
                print(f"\n{input_file}\n")
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)

                positions = np.array(positions, dtype="float64")
                atom_obj = Atoms(element_list, positions)
                atom_obj.calc = GPAW(mode='lcao', xc=config.FUNCTIONAL, txt=None)

                e = atom_obj.get_potential_energy(apply_constraint=False) / config.hartree2eV
                g = -1 * atom_obj.get_forces(apply_constraint=False) * config.bohr2angstroms / config.hartree2eV

                energy_list.append(e)
                gradient_list.append(g)
                geometry_num_list.append(positions / config.bohr2angstroms)
                num_list.append(num)

                if config.FC_COUNT == -1 or isinstance(optimize_num, str):
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    vib = Vibrations(atom_obj, delta=0.001)
                    vib.run()
                    result_vib = vib.get_vibrations()
                    exact_hess = result_vib.get_hessian_2d()
                    vib.clean()
                    exact_hess = exact_hess / config.hartree2eV * (config.bohr2angstroms ** 2)
                    calc_tools = Calculationtools()
                    exact_hess = calc_tools.project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, positions)
                    np.save(os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{hess_count}.npy"), exact_hess)
                    hess_count += 1

            except Exception as error:
                print(f"Error: {error}")
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