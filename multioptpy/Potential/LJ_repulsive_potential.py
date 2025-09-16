
from multioptpy.Parameters.parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib

import itertools
import math
import torch


class LJRepulsivePotentialScale:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib #function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib #function
        else:
            raise "No MM potential type"
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol
        
        indices_1 = torch.tensor(self.config["repulsive_potential_Fragm_1"]) - 1
        indices_2 = torch.tensor(self.config["repulsive_potential_Fragm_2"]) - 1
        self.pairs_1, self.pairs_2 = torch.meshgrid(indices_1, indices_2, indexing='ij')
        self.pairs_1 = self.pairs_1.flatten()
        self.pairs_2 = self.pairs_2.flatten()
   
        VDW_well_depth_1 = [self.VDW_well_depth_lib(self.config["element_list"][elem]) for elem in self.pairs_1]
        VDW_well_depth_2 = [self.VDW_well_depth_lib(self.config["element_list"][elem]) for elem in self.pairs_2]
        
        self.VDW_well_depth_1 = torch.tensor(VDW_well_depth_1)
        self.VDW_well_depth_2 = torch.tensor(VDW_well_depth_2)
        
        VDW_distance_1 = [self.VDW_distance_lib(self.config["element_list"][elem]) for elem in self.pairs_1]
        VDW_distance_2 = [self.VDW_distance_lib(self.config["element_list"][elem]) for elem in self.pairs_2]
        
        self.VDW_distance_1 = torch.tensor(VDW_distance_1)
        self.VDW_distance_2 = torch.tensor(VDW_distance_2)
        
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):#geom_num_list: torch.float32
        """
        # required variables: self.config["repulsive_potential_well_scale"], 
                             self.config["repulsive_potential_dist_scale"], 
                             self.config["repulsive_potential_Fragm_1"],
                             self.config["repulsive_potential_Fragm_2"]
                             self.config["element_list"]
        """
        energy = 0.0
        UFF_VDW_well_depth = torch.sqrt(
            self.config["repulsive_potential_well_scale"] ** 2 * self.VDW_well_depth_1 * self.VDW_well_depth_2)
        UFF_VDW_distance = torch.sqrt(
            self.config["repulsive_potential_dist_scale"] ** 2 * self.VDW_distance_1 * self.VDW_distance_2)
        
        vectors = torch.linalg.norm(geom_num_list[self.pairs_1] - geom_num_list[self.pairs_2], dim=1)
        
        tot_energy = UFF_VDW_well_depth * (
            -2 * (UFF_VDW_distance / vectors) ** 6
            + (UFF_VDW_distance / vectors) ** 12)
        energy = torch.sum(tot_energy)
        return energy

class LJRepulsivePotentialValue:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib #function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib #function
        else:
            raise "No MM potential type"
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 

        indices_1 = torch.tensor(self.config["repulsive_potential_Fragm_1"]) - 1
        indices_2 = torch.tensor(self.config["repulsive_potential_Fragm_2"]) - 1
        self.pairs_1, self.pairs_2 = torch.meshgrid(indices_1, indices_2, indexing='ij')
        self.pairs_1 = self.pairs_1.flatten()
        self.pairs_2 = self.pairs_2.flatten()
   
        VDW_well_depth_1 = [self.VDW_well_depth_lib(self.config["element_list"][elem]) for elem in self.pairs_1]
        VDW_well_depth_2 = [self.VDW_well_depth_lib(self.config["element_list"][elem]) for elem in self.pairs_2]
        
        self.VDW_well_depth_1 = torch.tensor(VDW_well_depth_1)
        self.VDW_well_depth_2 = torch.tensor(VDW_well_depth_2)
        
        VDW_distance_1 = [self.VDW_distance_lib(self.config["element_list"][elem]) for elem in self.pairs_1]
        VDW_distance_2 = [self.VDW_distance_lib(self.config["element_list"][elem]) for elem in self.pairs_2]
        
        self.VDW_distance_1 = torch.tensor(VDW_distance_1)
        self.VDW_distance_2 = torch.tensor(VDW_distance_2)
        
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):#geom_num_list: torch.float32
        """
        # required variables: self.config["repulsive_potential_well_value"], 
                             self.config["repulsive_potential_dist_value"], 
                             self.config["repulsive_potential_Fragm_1"],
                             self.config["repulsive_potential_Fragm_2"]
                             self.config["element_list"]
        """
        
        energy = 0.0
        UFF_VDW_well_depth = self.config["repulsive_potential_well_value"]/self.hartree2kjmol
        UFF_VDW_distance = self.config["repulsive_potential_dist_value"]/self.bohr2angstroms
        vectors = torch.linalg.norm(geom_num_list[self.pairs_1] - geom_num_list[self.pairs_2], dim=1)
        tot_energy = UFF_VDW_well_depth * (
            -2 * (UFF_VDW_distance / vectors) ** 6
            + (UFF_VDW_distance / vectors) ** 12)
        energy = torch.sum(tot_energy)
        return energy


class LJRepulsivePotentialv2Scale:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib #function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib #function
        else:
            raise "No MM potential type"
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        self.center1 = self.config["repulsive_potential_v2_center"][1] - 1
        self.center0 = self.config["repulsive_potential_v2_center"][0] - 1

        self.target_indices = torch.tensor(self.config["repulsive_potential_v2_target"]) - 1

        VDW_well_depth_center = self.VDW_well_depth_lib(self.config["element_list"][self.center1])
        VDW_distance_center = self.VDW_distance_lib(self.config["element_list"][self.center1])

        VDW_well_depth_target = [self.VDW_well_depth_lib(self.config["element_list"][tgt]) for tgt in self.target_indices]
        VDW_distance_target = [self.VDW_distance_lib(self.config["element_list"][tgt]) for tgt in self.target_indices]
        
        self.VDW_well_depth_center = torch.tensor(VDW_well_depth_center)
        self.VDW_distance_center = torch.tensor(VDW_distance_center)
        self.VDW_well_depth_target = torch.tensor(VDW_well_depth_target)
        self.VDW_distance_target = torch.tensor(VDW_distance_target)
                
        
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["repulsive_potential_v2_well_scale"], 
                             self.config["repulsive_potential_v2_dist_scale"], 
                             self.config["repulsive_potential_v2_length"],
                             self.config["repulsive_potential_v2_const_rep"]
                             self.config["repulsive_potential_v2_const_attr"], 
                             self.config["repulsive_potential_v2_order_rep"], 
                             self.config["repulsive_potential_v2_order_attr"],
                             self.config["repulsive_potential_v2_center"]
                             self.config["repulsive_potential_v2_target"]
                             self.config["element_list"]
        """
        energy = 0.0
        
     
        center_vector = geom_num_list[self.center1] - geom_num_list[self.center0]
        center_norm = torch.linalg.norm(center_vector)
        LJ_pot_center = (
            geom_num_list[self.center1]
            + (self.config["repulsive_potential_v2_length"] / self.bohr2angstroms) * (center_vector / center_norm))

        UFF_VDW_well_depth = torch.sqrt(
            self.config["repulsive_potential_v2_well_scale"] * self.VDW_well_depth_center * self.VDW_well_depth_target)
        UFF_VDW_distance = torch.sqrt(
            self.config["repulsive_potential_v2_dist_scale"] * self.VDW_distance_center * self.VDW_distance_target)

        vectors = geom_num_list[self.target_indices] - LJ_pot_center
        distances = torch.linalg.norm(vectors, dim=1)

        rep_term = (
            abs(self.config["repulsive_potential_v2_const_rep"])
            * (UFF_VDW_distance / distances) ** self.config["repulsive_potential_v2_order_rep"])
        attr_term = (
            -1
            * abs(self.config["repulsive_potential_v2_const_attr"])
            * (UFF_VDW_distance / distances) ** self.config["repulsive_potential_v2_order_attr"])

        tot_energy = UFF_VDW_well_depth * (rep_term + attr_term)
        energy = torch.sum(tot_energy)
        return energy
    
class LJRepulsivePotentialv2Value:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib  # function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib  # function
        else:
            raise ValueError("No MM potential type")
        self.config = kwarg

        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol
        
        self.center1 = self.config["repulsive_potential_v2_center"][1] - 1
        self.center0 = self.config["repulsive_potential_v2_center"][0] - 1
        self.target_indices = torch.tensor(self.config["repulsive_potential_v2_target"]) - 1
        
        target_elements = [self.config["element_list"][idx] for idx in self.target_indices]
        self.VDW_well_depth_target = torch.tensor([
            math.sqrt(self.config["repulsive_potential_v2_well_value"] / self.hartree2kjmol
                      * self.VDW_well_depth_lib(element))
            for element in target_elements
        ])
        self.VDW_distance_target = torch.tensor([
            math.sqrt(self.config["repulsive_potential_v2_dist_value"] / self.bohr2angstroms
                      * self.VDW_distance_lib(element))
            for element in target_elements
        ])

        
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["repulsive_potential_v2_well_value"], 
                             self.config["repulsive_potential_v2_dist_value"], 
                             self.config["repulsive_potential_v2_length"],
                             self.config["repulsive_potential_v2_const_rep"],
                             self.config["repulsive_potential_v2_const_attr"], 
                             self.config["repulsive_potential_v2_order_rep"], 
                             self.config["repulsive_potential_v2_order_attr"],
                             self.config["repulsive_potential_v2_center"],
                             self.config["repulsive_potential_v2_target"],
                             self.config["element_list"]
        """
        energy = 0.0
        center_vector = geom_num_list[self.center1] - geom_num_list[self.center0]
        center_norm = torch.linalg.norm(center_vector)
        LJ_pot_center = (
            geom_num_list[self.center1]
            + (self.config["repulsive_potential_v2_length"] / self.bohr2angstroms)
            * (center_vector / center_norm))

        vectors = geom_num_list[self.target_indices] - LJ_pot_center
        distances = torch.linalg.norm(vectors, dim=1)

        rep_term = (
            abs(self.config["repulsive_potential_v2_const_rep"])
            * (self.VDW_distance_target / distances) ** self.config["repulsive_potential_v2_order_rep"])
        attr_term = (
            -1
            * abs(self.config["repulsive_potential_v2_const_attr"])
            * (self.VDW_distance_target / distances) ** self.config["repulsive_potential_v2_order_attr"])

        energy = torch.sum(self.VDW_well_depth_target * (rep_term + attr_term))
        return energy
    
class LJRepulsivePotentialGaussian:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib #function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib #function
        else:
            raise "No MM potential type"
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        return
    #calc_energy_gau
    def calc_energy(self, geom_num_list, bias_pot_params=[]):

        """
        # required variables: self.config["repulsive_potential_gaussian_LJ_well_depth"], 
                             self.config["repulsive_potential_gaussian_LJ_dist"], 
                             self.config["repulsive_potential_gaussian_gau_well_depth"],
                             self.config["repulsive_potential_gaussian_gau_dist"]
                             self.config["repulsive_potential_gaussian_gau_range"], 
                             self.config["repulsive_potential_gaussian_fragm_1"], 
                             self.config["repulsive_potential_gaussian_fragm_2"],
                             self.config["element_list"]
        """
        energy = 0.0
        gau_range_const = 0.03
        for i, j in itertools.product(self.config["repulsive_potential_gaussian_fragm_1"], self.config["repulsive_potential_gaussian_fragm_2"]):
            LJ_well_depth = self.config["repulsive_potential_gaussian_LJ_well_depth"]/self.hartree2kjmol
            LJ_distance = self.config["repulsive_potential_gaussian_LJ_dist"]/self.bohr2angstroms
            Gau_well_depth = self.config["repulsive_potential_gaussian_gau_well_depth"]/self.hartree2kjmol
            Gau_distance = self.config["repulsive_potential_gaussian_gau_dist"]/self.bohr2angstroms
            Gau_range = self.config["repulsive_potential_gaussian_gau_range"]/self.bohr2angstroms
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            energy += LJ_well_depth * ( -2 * ( LJ_distance / vector ) ** 6 + ( LJ_distance / vector ) ** 12) -1 * Gau_well_depth * torch.exp(-1 * (vector - Gau_distance) ** 2 / (gau_range_const * (Gau_range) ** 2))
       
        return energy

class LJRepulsivePotentialCone:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib #function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib #function
        else:
            raise "No MM potential type"
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):

        a_value = 1.0
        """
        # ref.  ACS Catal. 2022, 12, 7, 3752–3766
        # required variables: self.config["cone_potential_well_value"], 
                             self.config["cone_potential_dist_value"], 
                             self.config["cone_potential_cone_angle"],
                             self.config["cone_potential_center"], 
                             self.config["cone_potential_three_atoms"]
                             self.config["cone_potential_target"]   
                             self.config["element_list"]
        """
        apex_vector = geom_num_list[self.config["cone_potential_center"]-1] - (2.28/self.bohr2angstroms) * ((geom_num_list[self.config["cone_potential_three_atoms"][0]-1] + geom_num_list[self.config["cone_potential_three_atoms"][1]-1] + geom_num_list[self.config["cone_potential_three_atoms"][2]-1] -3.0 * geom_num_list[self.config["cone_potential_center"]-1]) / torch.linalg.norm(geom_num_list[self.config["cone_potential_three_atoms"][0]-1] + geom_num_list[self.config["cone_potential_three_atoms"][1]-1] + geom_num_list[self.config["cone_potential_three_atoms"][2]-1] -3.0 * geom_num_list[self.config["cone_potential_center"]-1]))
        cone_angle = torch.deg2rad(torch.tensor(self.config["cone_potential_cone_angle"], dtype=torch.float32))
        energy = 0.0
        for i in self.config["cone_potential_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["cone_potential_well_value"]/self.hartree2kjmol * self.VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(self.config["cone_potential_dist_value"]/self.bohr2angstroms * self.VDW_distance_lib(self.config["element_list"][i-1]))
            s_a_length = (geom_num_list[i-1] - apex_vector).view(1,3)
            c_a_length = (geom_num_list[self.config["cone_potential_center"]-1] - apex_vector).view(1,3)
            sub_angle = torch.arccos((torch.matmul(c_a_length, s_a_length.T)) / (torch.linalg.norm(c_a_length) * torch.linalg.norm(s_a_length)))#rad
            dist = torch.linalg.norm(s_a_length)
            
            if sub_angle - cone_angle / 2 <= torch.pi / 2:
                length = (dist * torch.sin(sub_angle - cone_angle / 2)).view(1,1)
            
            else:
                length = dist.view(1,1)
            
            energy += 4 * UFF_VDW_well_depth * ((UFF_VDW_distance / (length + a_value * UFF_VDW_distance)) ** 12 - (UFF_VDW_distance / (length + a_value * UFF_VDW_distance)) ** 6)
            
        
        return energy
