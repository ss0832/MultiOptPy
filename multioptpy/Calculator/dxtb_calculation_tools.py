import glob
import os
import copy

import numpy as np
import torch

from abc import ABC, abstractmethod


try:
    import dxtb
    dxtb.timer.disable()
except:
    pass

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer

"""
ref:
dxtb
M. Friede, C. Hölzer, S. Ehlert, S. Grimme, dxtb -- An Efficient and Fully Differentiable Framework for Extended Tight-Binding, J. Chem. Phys., 2024, 161, 062501. 
DOI: https://doi.org/10.1063/5.0216715
"""


class Calculation:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        
        self.START_FILE = kwarg["START_FILE"]
        self.N_THREAD = kwarg["N_THREAD"]
        self.SET_MEMORY = kwarg["SET_MEMORY"]
        self.FUNCTIONAL = kwarg["FUNCTIONAL"]
        self.FC_COUNT = kwarg["FC_COUNT"]
        self.BPA_FOLDER_DIRECTORY = kwarg["BPA_FOLDER_DIRECTORY"]
        self.Model_hess = kwarg["Model_hess"]
        self.unrestrict = kwarg["unrestrict"]
        self.dft_grid = kwarg["dft_grid"]
        self.hessian_flag = False
    
    def single_point(self, file_directory, element_number_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
        """execute extended tight binding method calclation."""

        finish_frag = False
        
        if type(element_number_list[0]) is str:
            tmp = copy.copy(element_number_list)
            element_number_list = []
            
            for elem in tmp:    
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        torch_element_number_list = torch.tensor(element_number_list)

        try:
            os.mkdir(file_directory)
        except:
            pass

        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = glob.glob(file_directory+"/*_[0-9].xyz")

        for num, input_file in enumerate(file_list):
            try:
                
                if geom_num_list is None:
                    
                    positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    positions = geom_num_list        
            
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                torch_positions = torch.tensor(positions, requires_grad=True, dtype=torch.float32)
                
                max_scf_iteration = len(element_number_list) * 50 + 1000
                settings = {"maxiter": max_scf_iteration}
                
                
                if method == "GFN1-xTB":
                    calc = dxtb.calculators.GFN1Calculator(torch_element_number_list, opts=settings)
                elif method == "GFN2-xTB":
                    calc = dxtb.calculators.GFN2Calculator(torch_element_number_list, opts=settings)
                else:
                    print("method error")
                    raise

                if int(electric_charge_and_multiplicity[1]) > 1:

                    pos = torch_positions.clone().requires_grad_(True)
                    e = calc.get_energy(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1])) # hartree
                    calc.reset()
                    pos = torch_positions.clone().requires_grad_(True)
                    g = -1 * calc.get_forces(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1])) #hartree/Bohr
                    calc.reset()
                else:
                    pos = torch_positions.clone().requires_grad_(True)
                    e = calc.get_energy(pos, chrg=int(electric_charge_and_multiplicity[0])) # hartree
                    calc.reset()
                    pos = torch_positions.clone().requires_grad_(True)
                    g = -1 * calc.get_forces(pos, chrg=int(electric_charge_and_multiplicity[0])) #hartree/Bohr
                    calc.reset()
                
                #print("Orbital_energies :", self.orbital_energies)    
                #print("Orbital_occupations :", self.orbital_occupations)    
                #tmp = list(map(str, self.orbital_energies.tolist()))
                #with open(self.BPA_FOLDER_DIRECTORY+"orbital-energies.csv" ,"a") as f:
                #    f.write(",".join(tmp)+"\n")
                #tmp = list(map(str, self.orbital_occupations.tolist()))
                #with open(self.BPA_FOLDER_DIRECTORY+"orbital_occupations.csv" ,"a") as f:
                #    f.write(",".join(tmp)+"\n")
                #tmp = list(map(str, self.charges.tolist()))
                #with open(self.BPA_FOLDER_DIRECTORY+"charges.csv" ,"a") as f:
                #    f.write(",".join(tmp)+"\n")
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.exact_hessian(element_number_list, electric_charge_and_multiplicity, positions, torch_positions, calc)
                      

                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_number_list, electric_charge_and_multiplicity, positions, torch_positions, calc)
                   


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                print("Input file: ",file_list,"\n")
                finish_frag = True
                return np.array([0]), np.array([0]), positions, finish_frag 
        
        return_e = e.to('cpu').detach().numpy().copy()
        return_g = g.to('cpu').detach().numpy().copy()
        self.energy = return_e
        self.gradient = return_g
        self.coordinate = positions
        
        return return_e, return_g, positions, finish_frag

    def exact_hessian(self, element_number_list, electric_charge_and_multiplicity, positions, torch_positions, calc):
        """exact autograd hessian"""
                    
        pos = torch_positions.clone().requires_grad_(True)
        if int(electric_charge_and_multiplicity[1]) > 1:
            exact_hess = calc.get_hessian(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1]))
        else:
            exact_hess = calc.get_hessian(pos, chrg=int(electric_charge_and_multiplicity[0]))
        exact_hess = exact_hess.reshape(3*len(element_number_list), 3*len(element_number_list))
        return_exact_hess = exact_hess.to('cpu').detach().numpy().copy()
                    
                    #eigenvalues, _ = np.linalg.eigh(return_exact_hess)
                    #print("=== hessian (before add bias potential) ===")
                    #print("eigenvalues: ", eigenvalues)
                    
        return_exact_hess = copy.copy(Calculationtools().project_out_hess_tr_and_rot_for_coord(return_exact_hess, element_number_list.tolist(), positions, display_eigval=False))
        self.Model_hess = copy.copy(return_exact_hess)
        calc.reset()
    
    def ir(self, geom_num_list, element_number_list, electric_charge_and_multiplicity, method):
        finish_frag = False
        torch_positions = torch.tensor(geom_num_list, requires_grad=True, dtype=torch.float32)
        if type(element_number_list[0]) is str:
            tmp = copy.copy(element_number_list)
            element_number_list = []
            
            for elem in tmp:    
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        torch_element_number_list = torch.tensor(element_number_list)           
        max_scf_iteration = len(element_number_list) * 50 + 1000
        ef = dxtb.components.field.new_efield(torch.tensor([0.0, 0.0, 0.0], requires_grad=True))
        settings = {"maxiter": max_scf_iteration}
        if method == "GFN1-xTB":
            calc = dxtb.calculators.GFN1Calculator(torch_element_number_list, opts=settings, interaction=[ef])
        elif method == "GFN2-xTB":
            calc = dxtb.calculators.GFN2Calculator(torch_element_number_list, opts=settings, interaction=[ef])
        else:
            print("method error")
            raise

        if int(electric_charge_and_multiplicity[1]) > 1:
            pos = torch_positions.clone().requires_grad_(True)
            res = calc.ir(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1]))
            au_int = res.ints
        else:
            pos = torch_positions.clone().requires_grad_(True)
            res = calc.ir(pos, chrg=int(electric_charge_and_multiplicity[0]))
            au_int = res.ints
        res.use_common_units()
        common_freqs = res.freqs.cpu().detach().numpy().copy()
        au_int = au_int.cpu().detach().numpy().copy()
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
        method = config.usedxtb
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        # Get element number list from the first file
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        torch_element_number_list = torch.tensor(element_number_list)
        
        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                
                positions = np.array(positions, dtype="float64") / config.bohr2angstroms
                torch_positions = torch.tensor(positions, requires_grad=True, dtype=torch.float32)
                
                max_scf_iteration = len(element_number_list) * 50 + 1000
                settings = {"maxiter": max_scf_iteration}
                
                if method == "GFN1-xTB":
                    calc = dxtb.calculators.GFN1Calculator(torch_element_number_list, opts=settings)
                elif method == "GFN2-xTB":
                    calc = dxtb.calculators.GFN2Calculator(torch_element_number_list, opts=settings)
                else:
                    print("method error")
                    raise

                if int(electric_charge_and_multiplicity[1]) > 1:
                    pos = torch_positions.clone().requires_grad_(True)
                    e = calc.get_energy(pos, chrg=int(electric_charge_and_multiplicity[0]), 
                                      spin=int(electric_charge_and_multiplicity[1]))  # hartree
                    calc.reset()
                    pos = torch_positions.clone().requires_grad_(True)
                    g = -1 * calc.get_forces(pos, chrg=int(electric_charge_and_multiplicity[0]), 
                                           spin=int(electric_charge_and_multiplicity[1]))  # hartree/Bohr
                    calc.reset()
                else:
                    pos = torch_positions.clone().requires_grad_(True)
                    e = calc.get_energy(pos, chrg=int(electric_charge_and_multiplicity[0]))  # hartree
                    calc.reset()
                    pos = torch_positions.clone().requires_grad_(True)
                    g = -1 * calc.get_forces(pos, chrg=int(electric_charge_and_multiplicity[0]))  # hartree/Bohr
                    calc.reset()
                    
                return_e = e.to('cpu').detach().numpy().copy()
                return_g = g.to('cpu').detach().numpy().copy()
                
                energy_list.append(return_e)
                gradient_list.append(return_g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(return_g)**2/(len(return_g)*3)))  # RMS
                geometry_num_list.append(positions)
                num_list.append(num)
                
                if config.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    """exact autograd hessian"""
                    pos = torch_positions.clone().requires_grad_(True)
                    if int(electric_charge_and_multiplicity[1]) > 1:
                        exact_hess = calc.get_hessian(pos, chrg=int(electric_charge_and_multiplicity[0]), 
                                                    spin=int(electric_charge_and_multiplicity[1]))
                    else:
                        exact_hess = calc.get_hessian(pos, chrg=int(electric_charge_and_multiplicity[0]))
                    exact_hess = exact_hess.reshape(3*len(element_number_list), 3*len(element_number_list))
                    return_exact_hess = exact_hess.to('cpu').detach().numpy().copy()
                    
                    return_exact_hess = copy.copy(Calculationtools().project_out_hess_tr_and_rot_for_coord(
                        return_exact_hess, element_number_list.tolist(), positions))
                    np.save(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".npy", return_exact_hess)
                   
                    calc.reset()
                hess_count += 1
                
            except Exception as error:
                print(error)
                calc.reset()
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

