from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from calc_tools import Calculationtools

import itertools
import math
import numpy as np
import torch

class LinearMechanoForcePotential:
    def __init__(self, **kwarg):
        #ref: J. Am. Chem. Soc. 2009, 131, 18, 6377–6379
		#https://doi.org/10.1021/ja8095834
        #FMPES
        # This implementation is not tested yet.
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol
        self.pN2au = 1.213 * 10 ** (-5)
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["linear_mechano_force"], 
                             self.config["linear_mechano_force_atoms_1"], 
                             self.config["linear_mechano_force_atoms_2"],
        bias_pot_params[0] : linear_mechano_force
        """
        direction_1 = geom_num_list[self.config["linear_mechano_force_atoms_1"][1] - 1] - geom_num_list[self.config["linear_mechano_force_atoms_1"][0] - 1]
        direction_2 = geom_num_list[self.config["linear_mechano_force_atoms_2"][1] - 1] - geom_num_list[self.config["linear_mechano_force_atoms_2"][0] - 1]
        norm_direction_1 = torch.linalg.norm(direction_1)
        norm_direction_2 = torch.linalg.norm(direction_2)
        unit_direction_1 = direction_1 / norm_direction_1
        unit_direction_2 = direction_2 / norm_direction_2
        
        if len(bias_pot_params) == 0:
            force_magnitude = 0.5 * self.config["linear_mechano_force"] * self.pN2au
            energy = force_magnitude * torch.sum(unit_direction_1) + force_magnitude * torch.sum(unit_direction_2)
        else:
            force_magnitude = 0.5 * self.pN2au * bias_pot_params[0]
            energy =  force_magnitude * torch.sum(unit_direction_1) + force_magnitude * torch.sum(unit_direction_2)
        return energy #hartree
        
        
        
        