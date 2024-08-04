
from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from calc_tools import Calculationtools, torch_calc_outofplain_angle_from_vec


import numpy as np
import torch


class StructKeepOutofPlainAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_out_of_plain_angle_spring_const"],
                              self.config["keep_out_of_plain_angle_atom_pairs"]
                              self.config["keep_out_of_plain_angle_angle"]
                        
        """
        a1 = geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][1]-1] - geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][0]-1]
        a2 = geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][0]-1]
        a3 = geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][3]-1] - geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][0]-1]

        angle = torch_calc_outofplain_angle_from_vec(a1, a2, a3)
        energy = 0.5 * self.config["keep_out_of_plain_angle_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_out_of_plain_angle_angle"]))) ** 2
        
        return energy #hartree    
    
    def calc_energy_v2(self, geom_num_list):
        """
        # required variables: self.config["keep_out_of_plain_angle_v2_spring_const"], 
                             self.config["keep_out_of_plain_angle_v2_angle"], 
                             self.config["keep_out_of_plain_angle_v2_fragm1"],
                             self.config["keep_out_of_plain_angle_v2_fragm2"],
                             self.config["keep_out_of_plain_angle_v2_fragm3"],
                             self.config["keep_out_of_plain_angle_v2_fragm4"],
                             
        """
        fragm_1_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["keep_out_of_plain_angle_v2_fragm1"]:
            fragm_1_center = fragm_1_center + geom_num_list[i-1]
        
        fragm_1_center = fragm_1_center / len(self.config["keep_out_of_plain_angle_v2_fragm1"])
        
        fragm_2_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["keep_out_of_plain_angle_v2_fragm2"]:
            fragm_2_center = fragm_2_center + geom_num_list[i-1]
        
        fragm_2_center = fragm_2_center / len(self.config["keep_out_of_plain_angle_v2_fragm2"]) 
            
        fragm_3_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["keep_out_of_plain_angle_v2_fragm3"]:
            fragm_3_center = fragm_3_center + geom_num_list[i-1]
        
        fragm_3_center = fragm_3_center / len(self.config["keep_out_of_plain_angle_v2_fragm3"])   

        fragm_4_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["keep_out_of_plain_angle_v2_fragm4"]:
            fragm_4_center = fragm_4_center + geom_num_list[i-1]
        
        fragm_4_center = fragm_4_center / len(self.config["keep_out_of_plain_angle_v2_fragm4"])  
              
        a1 = fragm_2_center - fragm_1_center
        a2 = fragm_3_center - fragm_1_center
        a3 = fragm_4_center - fragm_1_center
        
        angle = torch_calc_outofplain_angle_from_vec(a1, a2, a3)

        energy = 0.5 * self.config["keep_out_of_plain_angle_v2_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_out_of_plain_angle_v2_angle"]))) ** 2
        return energy #hartree

