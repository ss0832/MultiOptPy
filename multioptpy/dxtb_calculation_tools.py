import glob
import os
import copy

import numpy as np
import torch

import dxtb

from calc_tools import Calculationtools
from parameter import UnitValueLib, element_number
from fileio import xyz2list

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
        
  