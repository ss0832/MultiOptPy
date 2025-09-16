import glob
import os
import copy

import numpy as np
from abc import ABC, abstractmethod


try:
    from tblite.interface import Calculator
except:
    pass

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
"""

GFN2-xTB(tblite)
J. Chem. Theory Comput. 2019, 15, 3, 1652–1671 
GFN1-xTB(tblite, dxtb)
J. Chem. Theory Comput. 2017, 13, 5, 1989–2009
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
        self.cpcm_solv_model = None
        self.alpb_solv_model = None
    
    def numerical_hessian(self, geom_num_list, element_list, method, electric_charge_and_multiplicity):#geom_num_list: 3*N (Bohr)
        numerical_delivative_delta = 0.0001
        
        count = 0
        hessian = np.zeros((3*len(geom_num_list), 3*len(geom_num_list)))
        for atom_num in range(len(geom_num_list)):
            for i in range(3):
                for atom_num_2 in range(len(geom_num_list)):
                    for j in range(3):
                        tmp_grad = []
                        if count > 3 * atom_num_2 + j:
                            continue
                        
                        for direction in [1, -1]:
                            geom_num_list = np.array(geom_num_list, dtype="float64")
                            max_scf_iteration = len(element_list) * 50 + 1000 
                            copy_geom_num_list = copy.copy(geom_num_list)
                            copy_geom_num_list[atom_num][i] += direction * numerical_delivative_delta
                            
                            if int(electric_charge_and_multiplicity[1]) > 1 or self.unrestrict:
                                calc = Calculator(method, element_list, copy_geom_num_list, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
                            else:
                                calc = Calculator(method, element_list, copy_geom_num_list, charge=int(electric_charge_and_multiplicity[0]))
                            
                            calc.set("max-iter", max_scf_iteration)
                            calc.set("verbosity", 0)
                            if not self.cpcm_solv_model is None:
                                calc.add("cpcm-solvation", self.cpcm_solv_model)
                            if not self.alpb_solv_model is None:
                                calc.add("alpb-solvation", self.alpb_solv_model)
                            res = calc.singlepoint()        
                            g = res.get("gradient") #hartree/Bohr
                            tmp_grad.append(g[atom_num_2][j])
                        
                        hessian[3*atom_num+i][3*atom_num_2+j] = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        hessian[3*atom_num_2+j][3*atom_num+i] = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        
                count += 1        
      
        
        return hessian
    
    def single_point(self, file_directory, element_number_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
        """execute extended tight binding method calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        
        if type(element_number_list[0]) is str:
            tmp = copy.copy(element_number_list)
            element_number_list = []
            
            for elem in tmp:    
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        
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
                max_scf_iteration = len(element_number_list) * 50 + 1000 
                if int(electric_charge_and_multiplicity[1]) > 1:
                                 
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
                else:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]))
                
                calc.set("max-iter", max_scf_iteration)           
                calc.set("verbosity", 0)
                calc.set("save-integrals", 1)
                if not self.cpcm_solv_model is None:        
                    print("Apply CPCM solvation model")
                    calc.add("cpcm-solvation", self.cpcm_solv_model)
                if not self.alpb_solv_model is None:
                    print("Apply ALPB solvation model")
                    calc.add("alpb-solvation", self.alpb_solv_model)
                            
                res = calc.singlepoint()
                
                e = float(res.get("energy"))  #hartree
                g = res.get("gradient") #hartree/Bohr
                
                self.orbital_coefficients = copy.deepcopy(res.get("orbital-coefficients"))
                self.overlap_matrix = copy.deepcopy(res.get("overlap-matrix"))
                self.density_matrix = copy.deepcopy(res.get("density-matrix"))
                self.orbital_energies = copy.deepcopy(res.get("orbital-energies"))
                self.orbital_occupations = copy.deepcopy(res.get("orbital-occupations"))
                self.charges = copy.deepcopy(res.get("charges"))
                
                #print("Orbital_energies :", self.orbital_energies)    
                #print("Orbital_occupations :", self.orbital_occupations)    
                tmp = list(map(str, self.orbital_energies.tolist()))
                with open(self.BPA_FOLDER_DIRECTORY+"orbital-energies.csv" ,"a") as f:
                    f.write(",".join(tmp)+"\n")
                tmp = list(map(str, self.orbital_occupations.tolist()))
                with open(self.BPA_FOLDER_DIRECTORY+"orbital_occupations.csv" ,"a") as f:
                    f.write(",".join(tmp)+"\n")
                tmp = list(map(str, self.charges.tolist()))
                with open(self.BPA_FOLDER_DIRECTORY+"charges.csv" ,"a") as f:
                    f.write(",".join(tmp)+"\n")
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.exact_hessian(element_number_list, electric_charge_and_multiplicity, method, positions)

                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_number_list, electric_charge_and_multiplicity, method, positions)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                print("Input file: ",file_list,"\n")
                finish_frag = True
                return np.array([0]), np.array([0]), positions, finish_frag 
            
        self.energy = e
        self.gradient = g
        self.coordinate = positions
        
        return e, g, positions, finish_frag

    def exact_hessian(self, element_number_list, electric_charge_and_multiplicity, method, positions):
        """exact numerical hessian"""
        exact_hess = self.numerical_hessian(positions, element_number_list, method, electric_charge_and_multiplicity)

                    #eigenvalues, _ = np.linalg.eigh(exact_hess)
                    #print("=== hessian (before add bias potential) ===")
                    #print("eigenvalues: ", eigenvalues)
                    
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_number_list.tolist(), positions, display_eigval=False)
        self.Model_hess = exact_hess
    
    def single_point_no_directory(self, positions, element_number_list, electric_charge_and_multiplicity, method):#positions:Bohr
        """execute extended tight binding method calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        
        if type(element_number_list[0]) is str:
            tmp = copy.copy(element_number_list)
            element_number_list = []
            
            for elem in tmp:    
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        
        
        try:
            
            positions = np.array(positions, dtype="float64") 
            max_scf_iteration = len(element_number_list) * 50 + 1000 
            if int(electric_charge_and_multiplicity[1]) > 1:
                calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
            else:
                calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]))
            
            calc.set("max-iter", max_scf_iteration)           
            calc.set("verbosity", 0)
            calc.set("save-integrals", 1)
            
            res = calc.singlepoint()
            
            e = float(res.get("energy"))  #hartree
            g = res.get("gradient") #hartree/Bohr
            self.orbital_coefficients = res.get("orbital-coefficients")
            self.overlap_matrix = res.get("overlap-matrix")
            self.density_matrix = res.get("density-matrix")
            self.orbital_energies = copy.deepcopy(res.get("orbital-energies"))                      
            print("\n")

        except Exception as error:
            print(error)
            print("This molecule could not be optimized.")
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
        method = config.usextb
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        # Get element number list from the first file
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                        
                positions = np.array(positions, dtype="float64") / config.bohr2angstroms
                if int(electric_charge_and_multiplicity[1]) > 1 or config.unrestrict:
                    calc = Calculator(method, element_number_list, positions, 
                                    charge=int(electric_charge_and_multiplicity[0]), 
                                    uhf=int(electric_charge_and_multiplicity[1]))
                else:
                    calc = Calculator(method, element_number_list, positions, 
                                    charge=int(electric_charge_and_multiplicity[0]))                
                calc.set("max-iter", 500)
                calc.set("verbosity", 0)
                if not config.cpcm_solv_model is None:        
                    print("Apply CPCM solvation model")
                    calc.add("cpcm-solvation", config.cpcm_solv_model)
                if not config.alpb_solv_model is None:
                    print("Apply ALPB solvation model")
                    calc.add("alpb-solvation", config.alpb_solv_model)
                            
                res = calc.singlepoint()
                e = float(res.get("energy"))  # hartree
                g = res.get("gradient")  # hartree/Bohr
                        
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))  # RMS
                geometry_num_list.append(positions)
                num_list.append(num)
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

