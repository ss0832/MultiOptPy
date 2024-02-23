import glob
import os
import copy

import numpy as np

from tblite.interface import Calculator

from calc_tools import Calculationtools
from optimizer import Model_hess_tmp
from param import UnitValueLib, element_number

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
                                calc = Calculator(method, element_list, geom_num_list, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
                            else:
                                calc = Calculator(method, element_list, geom_num_list, charge=int(electric_charge_and_multiplicity[0]))
                            
                            calc.set("max-iter", max_scf_iteration)
                            calc.set("verbosity", 0)
                            
                            res = calc.singlepoint()        
                            g = res.get("gradient") #hartree/Bohr
                            tmp_grad.append(g[atom_num_2][j])
                        hessian[3*atom_num+i][3*atom_num_2+j] = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        hessian[3*atom_num_2+j][3*atom_num+i] = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        
                count += 1        
      
        
        return hessian
    
    def single_point(self, file_directory, element_number_list, iter, electric_charge_and_multiplicity, method):
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
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                
                print("\n",input_file,"\n")

                with open(input_file,"r") as f:
                    input_data = f.readlines()
                
                positions = []
                if iter == 0:
                    for word in input_data[2:]:
                        positions.append(word.split()[1:4])
                else:
                    for word in input_data[1:]:
                        positions.append(word.split()[1:4])
                        
                
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                max_scf_iteration = len(element_number_list) * 50 + 1000 
                if int(electric_charge_and_multiplicity[1]) > 1:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
                else:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]))
                
                calc.set("max-iter", max_scf_iteration)           
                calc.set("verbosity", 0)
                
                res = calc.singlepoint()
                
                e = float(res.get("energy"))  #hartree
                g = res.get("gradient") #hartree/Bohr
                        
                print("\n")

                
                if self.FC_COUNT == -1 or type(iter) is str:
                    pass
                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    """exact numerical hessian"""
                    exact_hess = self.numerical_hessian(positions, element_number_list, method, electric_charge_and_multiplicity)

                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot(exact_hess, element_number_list.tolist(), positions)
                    self.Model_hess = Model_hess_tmp(exact_hess, momentum_disp=self.Model_hess.momentum_disp, momentum_grad=self.Model_hess.momentum_grad)
                   
                                 
                
                


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), np.array([0]), finish_frag 
            
        self.energy = e
        self.gradient = g
        self.coordinate = positions
        
        return e, g, positions, finish_frag