
from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_partial_center

import torch

class ValueRangePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        # ref.:https://doi.org/10.1063/5.0197592 (bond range potential)
        """
        # required variables: self.config["value_range_upper_const"]
                              self.config["value_range_lower_const"]
                              self.config["value_range_upper_distance"]
                              self.config["value_range_lower_distance"]
                              self.config["value_range_fragm_1"]
                              self.config["value_range_fragm_2"]
        """
        fragm_1_center = torch_calc_partial_center(geom_num_list, self.config["value_range_fragm_1"])
        fragm_2_center = torch_calc_partial_center(geom_num_list, self.config["value_range_fragm_2"])
        
        distance = torch.linalg.norm(fragm_1_center - fragm_2_center)
        
        upper_distance = self.config["value_range_upper_distance"] / self.bohr2angstroms
        lower_distance = self.config["value_range_lower_distance"] / self.bohr2angstroms
        upper_const = self.config["value_range_upper_const"]
        lower_const = self.config["value_range_lower_const"]
        energy = torch.log((1 + torch.exp(upper_const * (distance - upper_distance))) * (1 + torch.exp(lower_const * (lower_distance - distance))))
        return energy