
from multioptpy.Parameters.parameter import UnitValueLib

import numpy as np
import torch
import itertools
  
      
class UniversalPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["universal_pot_const"], 
                             self.config["universal_pot_target"], 
        """
        energy = 0.0
        
        point = geom_num_list[self.config["universal_pot_target"][0]-1]
        for i in range(1, len(self.config["universal_pot_target"])):
            point = point + geom_num_list[self.config["universal_pot_target"][i]-1]
        
        point = point / len(self.config["universal_pot_target"])
        
        for i in self.config["universal_pot_target"]:
            energy = energy + self.config["universal_pot_const"] / self.hartree2kjmol / len(list(itertools.combinations(self.config["universal_pot_target"], 2))) * torch.linalg.norm(geom_num_list[i-1] - point)
        
        return energy