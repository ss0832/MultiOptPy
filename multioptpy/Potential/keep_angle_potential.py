from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_angle_from_vec

import torch


class StructKeepAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_angle_atom_pairs"],
                              self.config["keep_angle_spring_const"]
                              self.config["keep_angle_angle"]
        bias_pot_params[0] : keep_angle_spring_const
        bias_pot_params[1] : keep_angle_angle
        """
       
        vector1 = geom_num_list[self.config["keep_angle_atom_pairs"][0]-1] - geom_num_list[self.config["keep_angle_atom_pairs"][1]-1]
        vector2 = geom_num_list[self.config["keep_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_angle_atom_pairs"][1]-1]
        theta = torch_calc_angle_from_vec(vector1, vector2)
        if len(bias_pot_params) == 0:
            energy = 0.5 * self.config["keep_angle_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["keep_angle_angle"]))) ** 2
        else:
            energy = 0.5 * bias_pot_params[0] * (theta - torch.deg2rad(bias_pot_params[1])) ** 2
        return energy #hartree



class StructKeepAnglePotentialv2:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_angle_v2_spring_const"], 
                             self.config["keep_angle_v2_angle"], 
                             self.config["keep_angle_v2_fragm1"],
                             self.config["keep_angle_v2_fragm2"],
                             self.config["keep_angle_v2_fragm3"],
        bias_pot_params[0] : keep_angle_v2_spring_const
        bias_pot_params[1] : keep_angle_v2_angle
               
        """
        fragm_1_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm1"]) - 1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm2"]) - 1], dim=0)
        fragm_3_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm3"]) - 1], dim=0)
           
        vector1 = fragm_1_center - fragm_2_center
        vector2 = fragm_3_center - fragm_2_center
        theta = torch_calc_angle_from_vec(vector1, vector2)
        if len(bias_pot_params) == 0:
            energy = 0.5 * self.config["keep_angle_v2_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["keep_angle_v2_angle"]))) ** 2
        else:
            energy = 0.5 * bias_pot_params[0] * (theta - torch.deg2rad(bias_pot_params[1])) ** 2
        return energy #hartree


class StructKeepAnglePotentialAtomDistDependent:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["aDD_keep_angle_spring_const"] 
                              self.config["aDD_keep_angle_min_angle"] 
                              self.config["aDD_keep_angle_max_angle"]
                              self.config["aDD_keep_angle_base_dist"]
                              self.config["aDD_keep_angle_reference_atom"] 
                              self.config["aDD_keep_angle_center_atom"] 
                              self.config["aDD_keep_angle_atoms"]
        
        """
        energy = 0.0
        self.config["keep_angle_spring_const"] = self.config["aDD_keep_angle_spring_const"] 
        max_angle = torch.tensor(self.config["aDD_keep_angle_max_angle"])
        min_angle = torch.tensor(self.config["aDD_keep_angle_min_angle"])
        ref_dist = torch.linalg.norm(geom_num_list[self.config["aDD_keep_angle_center_atom"]-1] - geom_num_list[self.config["aDD_keep_angle_reference_atom"]-1]) / self.bohr2angstroms
        base_dist = self.config["aDD_keep_angle_base_dist"] / self.bohr2angstroms
        eq_angle = min_angle + ((max_angle - min_angle)/(1 + torch.exp(-(ref_dist - base_dist))))
        
        bias_pot_params = [self.config["aDD_keep_angle_spring_const"], eq_angle]
        KAP = StructKeepAnglePotential(keep_angle_angle=eq_angle, keep_angle_spring_const=self.config["aDD_keep_angle_spring_const"], keep_angle_atom_pairs=[self.config["aDD_keep_angle_atoms"][0], self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]])
        
        energy += KAP.calc_energy(geom_num_list, bias_pot_params)
        
        KAP = StructKeepAnglePotential(keep_angle_angle=eq_angle, keep_angle_spring_const=self.config["aDD_keep_angle_spring_const"], keep_angle_atom_pairs=[self.config["aDD_keep_angle_atoms"][2], self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]])
        energy += KAP.calc_energy(geom_num_list, bias_pot_params)
        
        KAP = StructKeepAnglePotential(keep_angle_angle=eq_angle, keep_angle_spring_const=self.config["aDD_keep_angle_spring_const"], keep_angle_atom_pairs=[self.config["aDD_keep_angle_atoms"][0], self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][2]])
        energy += KAP.calc_energy(geom_num_list, bias_pot_params)
    
        return energy
    
    
class StructKeepAnglePotentialLonePairAngle:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["lone_pair_keep_angle_spring_const"] 
                              self.config["lone_pair_keep_angle_angle"] 
                              self.config["lone_pair_keep_angle_atom_pair_1"]
                              self.config["lone_pair_keep_angle_atom_pair_2"]        
        """
        lone_pair_1_vec_1 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][1]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][0]-1]) / self.bohr2angstroms
     
        lone_pair_1_vec_2 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][2]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][0]-1]) / self.bohr2angstroms
        lone_pair_1_vec_3 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][3]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][0]-1]) / self.bohr2angstroms
        lone_pair_1_vector = (lone_pair_1_vec_1/torch.linalg.norm(lone_pair_1_vec_1)) + (lone_pair_1_vec_2/torch.linalg.norm(lone_pair_1_vec_2)) + (lone_pair_1_vec_3/torch.linalg.norm(lone_pair_1_vec_3))
        
        lone_pair_1_vector_norm = lone_pair_1_vector / torch.linalg.norm(lone_pair_1_vector)
        
        lone_pair_2_vec_1 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][1]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][0]-1]) / self.bohr2angstroms
        lone_pair_2_vec_2 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][2]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][0]-1]) / self.bohr2angstroms
        lone_pair_2_vec_3 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][3]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][0]-1]) / self.bohr2angstroms
        
        lone_pair_2_vector = (lone_pair_2_vec_1/torch.linalg.norm(lone_pair_2_vec_1)) + (lone_pair_2_vec_2/torch.linalg.norm(lone_pair_2_vec_2)) + (lone_pair_2_vec_3/torch.linalg.norm(lone_pair_2_vec_3))
        
        lone_pair_2_vector_norm = lone_pair_2_vector / torch.linalg.norm(lone_pair_2_vector)
        
        theta = torch_calc_angle_from_vec(lone_pair_1_vector_norm, lone_pair_2_vector_norm)
        energy = 0.5 * self.config["lone_pair_keep_angle_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["lone_pair_keep_angle_angle"]))) ** 2

        return energy