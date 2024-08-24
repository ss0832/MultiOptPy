
from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from calc_tools import Calculationtools

import numpy as np
import torch
import itertools
  
      
class FluxPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["flux_pot_const"], 
                             self.config["flux_pot_order"], 
                             self.config["flux_pot_direction"], 
                             self.config["flux_pot_target"], 
        """
        energy = 0.0
        direction = torch.tensor(self.config["flux_pot_direction"] / self.bohr2angstroms, dtype=torch.float64)
        order = torch.tensor(self.config["flux_pot_order"], dtype=torch.float64)
        const = torch.tensor(self.config["flux_pot_const"], dtype=torch.float64)
        
        for i in self.config["flux_pot_target"]:
            energy = energy + const * (geom_num_list[i-1] - direction) ** order
        energy = torch.sum(energy)
        
        return energy