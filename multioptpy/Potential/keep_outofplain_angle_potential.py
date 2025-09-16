
from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_outofplain_angle_from_vec
import torch


class StructKeepOutofPlainAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_out_of_plain_angle_spring_const"],
                              self.config["keep_out_of_plain_angle_atom_pairs"]
                              self.config["keep_out_of_plain_angle_angle"]
        bias_pot_params[0] : keep_out_of_plain_angle_spring_const
        bias_pot_params[1] : keep_out_of_plain_angle_angle 
                        
        """
        a1 = geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][1]-1] - geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][0]-1]
        a2 = geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][0]-1]
        a3 = geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][3]-1] - geom_num_list[self.config["keep_out_of_plain_angle_atom_pairs"][0]-1]

        angle = torch_calc_outofplain_angle_from_vec(a1, a2, a3)
        if len(bias_pot_params) == 0:
            energy = 0.5 * self.config["keep_out_of_plain_angle_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_out_of_plain_angle_angle"]))) ** 2
        else:
            energy = 0.5 * bias_pot_params[0] * (angle - torch.deg2rad(bias_pot_params[1])) ** 2
        return energy #hartree 
       
class StructKeepOutofPlainAnglePotentialv2:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_out_of_plain_angle_v2_spring_const"], 
                             self.config["keep_out_of_plain_angle_v2_angle"], 
                             self.config["keep_out_of_plain_angle_v2_fragm1"],
                             self.config["keep_out_of_plain_angle_v2_fragm2"],
                             self.config["keep_out_of_plain_angle_v2_fragm3"],
                             self.config["keep_out_of_plain_angle_v2_fragm4"],
                             
        """
        fragm_1_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_out_of_plain_angle_v2_fragm1"]) - 1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_out_of_plain_angle_v2_fragm2"]) - 1], dim=0)
        fragm_3_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_out_of_plain_angle_v2_fragm3"]) - 1], dim=0)
        fragm_4_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_out_of_plain_angle_v2_fragm4"]) - 1], dim=0)

              
        a1 = fragm_2_center - fragm_1_center
        a2 = fragm_3_center - fragm_1_center
        a3 = fragm_4_center - fragm_1_center
        
        angle = torch_calc_outofplain_angle_from_vec(a1, a2, a3)
        if len(bias_pot_params) == 0:
            energy = 0.5 * self.config["keep_out_of_plain_angle_v2_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_out_of_plain_angle_v2_angle"]))) ** 2
        else:
            energy = 0.5 * bias_pot_params[0] * (angle - torch.deg2rad(bias_pot_params[1])) ** 2
        return energy #hartree

