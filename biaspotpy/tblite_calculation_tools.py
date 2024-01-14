import glob
import os

import numpy as np

from tblite.interface import Calculator

from calc_tools import Calculationtools
from optimizer import Model_hess_tmp
from param import UnitValueLib

class SinglePoint:
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
    
    def tblite_calculation(self, file_directory, element_number_list, electric_charge_and_multiplicity, iter, method):
        """execute extended tight binding method calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
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
                max_scf_iteration = len(element_number_list) * 100 + 2500 
                calc = Calculator(method, element_number_list, positions)
                calc.set("max-iter", max_scf_iteration)
                calc.set("verbosity", 0)
                res = calc.singlepoint()
                e = float(res.get("energy"))  #hartree
                g = res.get("gradient") #hartree/Bohr
                        
                print("\n")

                
                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    print("error (cant calculate hessian)")
                    return 0, 0, 0, finish_frag 
                


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag 
        
        return e, g, positions, finish_frag