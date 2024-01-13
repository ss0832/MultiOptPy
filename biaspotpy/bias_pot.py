import torch
import itertools
import math
import numpy as np
import copy
import random

from param import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib


class LJRepulsivePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        return
    
    
    def calc_energy_scale(self, geom_num_list):#geom_num_list: torch.float64
        """
        # required variables: self.config["repulsive_potential_well_scale"], 
                             self.config["repulsive_potential_dist_scale"], 
                             self.config["repulsive_potential_Fragm_1"],
                             self.config["repulsive_potential_Fragm_2"]
                             self.config["element_list"]
        """
        energy = 0.0

        for i, j in itertools.product(self.config["repulsive_potential_Fragm_1"], self.config["repulsive_potential_Fragm_2"]):
            UFF_VDW_well_depth = math.sqrt(self.config["repulsive_potential_well_scale"]*UFF_VDW_well_depth_lib(self.config["element_list"][i-1]) + self.config["repulsive_potential_well_scale"]*UFF_VDW_well_depth_lib(self.config["element_list"][j-1]))
            UFF_VDW_distance = math.sqrt(UFF_VDW_distance_lib(self.config["element_list"][i-1])*self.config["repulsive_potential_dist_scale"] + UFF_VDW_distance_lib(self.config["element_list"][j-1])*self.config["repulsive_potential_dist_scale"])
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            energy += UFF_VDW_well_depth * ( -2 * ( UFF_VDW_distance / vector ) ** 6 + ( UFF_VDW_distance / vector ) ** 12)
            
        return energy
    
    def calc_energy_value(self, geom_num_list):#geom_num_list: torch.float64
        """
        # required variables: self.config["repulsive_potential_well_value"], 
                             self.config["repulsive_potential_dist_value"], 
                             self.config["repulsive_potential_Fragm_1"],
                             self.config["repulsive_potential_Fragm_2"]
                             self.config["element_list"]
        """
        energy = 0.0

        for i, j in itertools.product(self.config["repulsive_potential_Fragm_1"], self.config["repulsive_potential_Fragm_2"]):
            UFF_VDW_well_depth = self.config["repulsive_potential_well_value"]/self.hartree2kjmol
            UFF_VDW_distance = self.config["repulsive_potential_dist_value"]/self.bohr2angstroms
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            energy += UFF_VDW_well_depth * ( -2 * ( UFF_VDW_distance / vector ) ** 6 + ( UFF_VDW_distance / vector ) ** 12)
            
        return
    
    def calc_energy_scale_v2(self, geom_num_list):
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
        
        LJ_pot_center = geom_num_list[self.config["repulsive_potential_v2_center"][1]-1] + (self.config["repulsive_potential_v2_length"]/self.bohr2angstroms) * (geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1] / torch.linalg.norm(geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1])) 
        for i in self.config["repulsive_potential_v2_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["repulsive_potential_v2_well_scale"]*UFF_VDW_well_depth_lib(self.config["element_list"][self.config["repulsive_potential_v2_center"][1]-1]) * UFF_VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(UFF_VDW_distance_lib(self.config["element_list"][self.config["repulsive_potential_v2_center"][1]-1])*self.config["repulsive_potential_v2_dist_scale"] * UFF_VDW_distance_lib(self.config["repulsive_potential_v2_center"][i-1]))
            
            vector = torch.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
            energy += UFF_VDW_well_depth * ( abs(self.config["repulsive_potential_v2_const_rep"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_rep"] -1 * abs(self.config["repulsive_potential_v2_const_attr"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_attr"])
            
        return energy
    def calc_energy_value_v2(self, geom_num_list):

        """
        # required variables: self.config["repulsive_potential_v2_well_value"], 
                             self.config["repulsive_potential_v2_dist_value"], 
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
        
        LJ_pot_center = geom_num_list[self.config["repulsive_potential_v2_center"][1]-1] + (self.config["repulsive_potential_v2_length"]/self.bohr2angstroms) * (geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1] / torch.linalg.norm(geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1])) 
        for i in self.config["repulsive_potential_v2_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["repulsive_potential_v2_well_value"]/self.hartree2kjmol * UFF_VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(self.config["repulsive_potential_v2_dist_value"]/self.bohr2angstroms * UFF_VDW_distance_lib(self.config["repulsive_potential_v2_center"][i-1]))
            
            vector = torch.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
            energy += UFF_VDW_well_depth * ( abs(self.config["repulsive_potential_v2_const_rep"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_rep"] -1 * abs(self.config["repulsive_potential_v2_const_attr"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_attr"])
            
        return energy
    
 

    
    def calc_cone_potential_energy(self, geom_num_list):

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
        cone_angle = torch.deg2rad(torch.tensor(self.config["cone_potential_cone_angle"], dtype=torch.float64))
        energy = 0.0
        for i in self.config["cone_potential_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["cone_potential_well_value"]/self.hartree2kjmol * UFF_VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(self.config["cone_potential_dist_value"]/self.bohr2angstroms * UFF_VDW_distance_lib(self.config["element_list"][i-1]))
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
        
class AFIRPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["AFIR_gamma"], 
                             self.config["AFIR_Fragm_1"], 
                             self.config["AFIR_Fragm_2"],
                             self.config["element_list"]
        """
        """
        ###  Reference  ###
            Chem. Rec., 2016, 16, 2232
            J. Comput. Chem., 2018, 39, 233
            WIREs Comput. Mol. Sci., 2021, 11, e1538
        """
        R_0 = 3.8164/self.bohr2angstroms #ang.→bohr
        EPSIRON = 1.0061/self.hartree2kjmol #kj/mol→hartree
        if self.config["AFIR_gamma"] > 0.0 or self.config["AFIR_gamma"] < 0.0:
            alpha = (self.config["AFIR_gamma"]/self.hartree2kjmol) / ((2 ** (-1/6) - (1 + math.sqrt(1 + (abs(self.config["AFIR_gamma"]/self.hartree2kjmol) / EPSIRON))) ** (-1/6))*R_0) #hartree/Bohr
        else:
            alpha = 0.0
        A = 0.0
        B = 0.0
        
        p = 6.0

        for i, j in itertools.product(self.config["AFIR_Fragm_1"], self.config["AFIR_Fragm_2"]):
            R_i = covalent_radii_lib(self.config["element_list"][i-1])
            R_j = covalent_radii_lib(self.config["element_list"][j-1])
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            omega = ((R_i + R_j) / vector) ** p #no unit
            A += omega * vector
            B += omega
        
        energy = alpha*(A/B)#A/B:Bohr
        return energy #hartree
      
class StructKeepPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_pot_spring_const"], 
                             self.config["keep_pot_distance"], 
                             self.config["keep_pot_atom_pairs"],
                             
        """
        vector = torch.linalg.norm((geom_num_list[self.config["keep_pot_atom_pairs"][0]-1] - geom_num_list[self.config["keep_pot_atom_pairs"][1]-1]), ord=2)
        energy = 0.5 * self.config["keep_pot_spring_const"] * (vector - self.config["keep_pot_distance"]/self.bohr2angstroms) ** 2
        return energy #hartree

class StructAnharmonicKeepPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["anharmonic_keep_pot_spring_const"],
                              self.config["anharmonic_keep_pot_potential_well_depth"]
                              self.config["anharmonic_keep_pot_atom_pairs"]
                              self.config["anharmonic_keep_pot_distance"]

        """
        vector = torch.linalg.norm((geom_num_list[self.config["anharmonic_keep_pot_atom_pairs"][0]-1] - geom_num_list[self.config["anharmonic_keep_pot_atom_pairs"][1]-1]), ord=2)
        if self.config["anharmonic_keep_pot_potential_well_depth"] != 0.0:
            energy = self.config["anharmonic_keep_pot_potential_well_depth"] * ( 1.0 - torch.exp( - math.sqrt(self.config["anharmonic_keep_pot_spring_const"] / (2 * self.config["anharmonic_keep_pot_potential_well_depth"])) * (vector - self.config["anharmonic_keep_pot_distance"]/self.bohr2angstroms)) ) ** 2
        else:
            energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

        return energy

class StructKeepAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_angle_atom_pairs"],
                              self.config["keep_angle_spring_const"]
                              self.config["keep_angle_angle"]

        """
       
        vector1 = geom_num_list[self.config["keep_angle_atom_pairs"][0]-1] - geom_num_list[self.config["keep_angle_atom_pairs"][1]-1]
        vector2 = geom_num_list[self.config["keep_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_angle_atom_pairs"][1]-1]
        magnitude1 = torch.linalg.norm(vector1)
        magnitude2 = torch.linalg.norm(vector2)
        dot_product = torch.matmul(vector1, vector2)
        cos_theta = dot_product / (magnitude1 * magnitude2)
        theta = torch.arccos(cos_theta)
        energy = 0.5 * self.config["keep_angle_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["keep_angle_angle"]))) ** 2
        return energy #hartree
        
    def calc_atom_dist_dependent_energy(self, geom_num_list):
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
        
        self.config["keep_angle_angle"] = eq_angle
        
        
        self.config["keep_angle_atom_pairs"] = [self.config["aDD_keep_angle_atoms"][0] , self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]]
        energy += self.calc_energy(geom_num_list)
        self.config["keep_angle_atom_pairs"] = [self.config["aDD_keep_angle_atoms"][2] , self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]]
        energy += self.calc_energy(geom_num_list)
        self.config["keep_angle_atom_pairs"] = [self.config["aDD_keep_angle_atoms"][0] , self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][2]]
        energy += self.calc_energy(geom_num_list)
    
        return energy

class StructKeepDihedralAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_dihedral_angle_spring_const"],
                              self.config["keep_dihedral_angle_atom_pairs"]
                              self.config["keep_dihedral_angle_angle"]
                        
        """
        a1 = geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][1]-1] - geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][0]-1]
        a2 = geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][1]-1]
        a3 = geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][3]-1] - geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][2]-1]

        v1 = torch.cross(a1, a2)
        v1 = v1 / torch.linalg.norm(v1, ord=2)
        v2 = torch.cross(a2, a3)
        v2 = v2 / torch.linalg.norm(v2, ord=2)
        angle = torch.arccos((v1*v2).sum(-1) / ((v1**2).sum() * (v2**2).sum())**0.5)

        energy = 0.5 * self.config["keep_dihedral_angle_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_dihedral_angle_angle"]))) ** 2
        
        return energy #hartree    

class VoidPointPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["void_point_pot_spring_const"],
                              self.config["void_point_pot_atoms"]
                              self.config["void_point_pot_coord"]  #need to convert tensor type 
                              
                              self.config["void_point_pot_distance"]
                              self.config["void_point_pot_order"]
                        
        """
        vector = torch.linalg.norm((geom_num_list[self.config["void_point_pot_atoms"]-1] - self.config["void_point_pot_coord"]), ord=2)
        energy = (1 / self.config["void_point_pot_order"]) * self.config["void_point_pot_spring_const"] * (vector - self.config["void_point_pot_distance"]/self.bohr2angstroms) ** self.config["void_point_pot_order"]
        return energy #hartree

class WellPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["well_pot_wall_energy"]
                              self.config["well_pot_fragm_1"]
                              self.config["well_pot_fragm_2"]
                              self.config["well_pot_limit_dist"]
                              
                              
        """
        fragm_1_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["well_pot_fragm_1"]:
            fragm_1_center = fragm_1_center + geom_num_list[i-1]
        
        fragm_1_center = fragm_1_center / len(self.config["well_pot_fragm_1"])
        
        fragm_2_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["well_pot_fragm_2"]:
            fragm_2_center = fragm_2_center + geom_num_list[i-1]
        
        fragm_2_center = fragm_2_center / len(self.config["well_pot_fragm_2"])        
        
        vec_norm = torch.linalg.norm(fragm_1_center - fragm_2_center) 
        a = float(self.config["well_pot_limit_dist"][0]) / self.bohr2angstroms
        b = float(self.config["well_pot_limit_dist"][1]) / self.bohr2angstroms
        c = float(self.config["well_pot_limit_dist"][2]) / self.bohr2angstroms
        d = float(self.config["well_pot_limit_dist"][3]) / self.bohr2angstroms
        short_dist_linear_func_slope = 0.5 / (b - a)
        short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
        long_dist_linear_func_slope = 0.5 / (c - d)
        long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

        x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
        x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

        if vec_norm <= a:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
            
        elif a < vec_norm and vec_norm <= b:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                
        elif b < vec_norm and vec_norm < c:
            energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
            
        elif c <= vec_norm and vec_norm < d:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
            
        elif d <= vec_norm:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
            
        else:
            print("well pot error")
            raise "well pot error"
        #print(energy)
        return energy
        
    def calc_energy_wall(self, geom_num_list):
        """
        # required variables: self.config["wall_well_pot_wall_energy"]
                              self.config["wall_well_pot_direction"] 
                              self.config["wall_well_pot_limit_dist"]
                              self.config["wall_well_pot_target"]
                              
                              
        """
        
        if self.config["wall_well_pot_direction"] == "x":
            direction_num = 0
        elif self.config["wall_well_pot_direction"] == "y":
            direction_num = 1
        elif self.config["wall_well_pot_direction"] == "z":
            direction_num = 2
                     
        
        
        energy = 0.0
        for i in self.config["wall_well_pot_target"]:

            vec_norm = abs(torch.linalg.norm(geom_num_list[i-1][direction_num])) 
                     
            a = float(self.config["wall_well_pot_limit_dist"][0]) / self.bohr2angstroms
            b = float(self.config["wall_well_pot_limit_dist"][1]) / self.bohr2angstroms
            c = float(self.config["wall_well_pot_limit_dist"][2]) / self.bohr2angstroms
            d = float(self.config["wall_well_pot_limit_dist"][3]) / self.bohr2angstroms
            short_dist_linear_func_slope = 0.5 / (b - a)
            short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
            long_dist_linear_func_slope = 0.5 / (c - d)
            long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

            x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
            x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

            if vec_norm <= a:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
                
            elif a < vec_norm and vec_norm <= b:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                    
            elif b < vec_norm and vec_norm < c:
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
        
    def calc_energy_vp(self, geom_num_list):
        """
        # required variables: self.config["void_point_well_pot_wall_energy"]
                              self.config["void_point_well_pot_coordinate"] 
                              self.config["void_point_well_pot_limit_dist"]
                              self.config["void_point_well_pot_target"]
                              
                              
        """
        self.config["void_point_well_pot_coordinate"]  = torch.tensor(self.config["void_point_well_pot_coordinate"], dtype=torch.float64)

        
        energy = 0.0
        for i in self.config["void_point_well_pot_target"]:

            vec_norm = torch.linalg.norm(geom_num_list[i-1] - self.config["void_point_well_pot_coordinate"]) 
                     
            a = float(self.config["void_point_well_pot_limit_dist"][0]) / self.bohr2angstroms
            b = float(self.config["void_point_well_pot_limit_dist"][1]) / self.bohr2angstroms
            c = float(self.config["void_point_well_pot_limit_dist"][2]) / self.bohr2angstroms
            d = float(self.config["void_point_well_pot_limit_dist"][3]) / self.bohr2angstroms
            short_dist_linear_func_slope = 0.5 / (b - a)
            short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
            long_dist_linear_func_slope = 0.5 / (c - d)
            long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

            x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
            x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

            if vec_norm <= a:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
                
            elif a < vec_norm and vec_norm <= b:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                    
            elif b < vec_norm and vec_norm < c:
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
        
    def calc_energy_around(self, geom_num_list):
        """
        # required variables: self.config["around_well_pot_wall_energy"]
                              self.config["around_well_pot_center"] 
                              self.config["around_well_pot_limit_dist"]
                              self.config["around_well_pot_target"]
                              
                              
        """
        geom_center_coord = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["around_well_pot_center"]:
            geom_center_coord = geom_center_coord + geom_num_list[i-1]
        geom_center_coord = geom_center_coord/len(self.config["around_well_pot_center"])
        energy = 0.0
        for i in self.config["around_well_pot_target"]:

            vec_norm = torch.linalg.norm(geom_num_list[i-1] - geom_center_coord) 
                     
            a = float(self.config["around_well_pot_limit_dist"][0]) / self.bohr2angstroms
            b = float(self.config["around_well_pot_limit_dist"][1]) / self.bohr2angstroms
            c = float(self.config["around_well_pot_limit_dist"][2]) / self.bohr2angstroms
            d = float(self.config["around_well_pot_limit_dist"][3]) / self.bohr2angstroms
            short_dist_linear_func_slope = 0.5 / (b - a)
            short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
            long_dist_linear_func_slope = 0.5 / (c - d)
            long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

            x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
            x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

            if vec_norm <= a:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
                
            elif a < vec_norm and vec_norm <= b:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                    
            elif b < vec_norm and vec_norm < c:
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
      
class BiasPotentialCalculation:
    def __init__(self, Model_hess, FC_COUNT):
        torch.set_printoptions(precision=12)
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.Model_hess = Model_hess
        self.FC_COUNT = FC_COUNT
        self.JOBID = random.randint(0, 1000000)
    
    def ndarray2tensor(self, ndarray):
        tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64, requires_grad=True))

        return tensor

    def ndarray2nogradtensor(self, ndarray):
        tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64))

        return tensor

    def tensor2ndarray(self, tensor):
        ndarray = copy.copy(tensor.detach().numpy())
        return ndarray
    
    def main(self, e, g, geom_num_list, element_list,  force_data, pre_B_g="", iter="", initial_geom_num_list=""):
        numerical_derivative_delta = 0.0001 #unit:Bohr
        
        #g:hartree/Bohr
        #e:hartree
        #geom_num_list:Bohr

  
        #--------------------------------------------------
        B_e = e
        BPA_grad_list = g*0.0
        BPA_hessian = np.zeros((3*len(g), 3*len(g)))
        #debug_delta_BPA_grad_list = g*0.0
        geom_num_list = self.ndarray2tensor(geom_num_list)
        #print("rpv2")   torch.tensor(*** , dtype=torch.float64, requires_grad=True)




        for i in range(len(force_data["repulsive_potential_v2_well_scale"])):
            if force_data["repulsive_potential_v2_well_scale"][i] != 0.0:
                if force_data["repulsive_potential_v2_unit"][i] == "scale":
                    LJRP = LJRepulsivePotential(repulsive_potential_v2_well_scale=force_data["repulsive_potential_v2_well_scale"][i], 
                                                repulsive_potential_v2_dist_scale=force_data["repulsive_potential_v2_dist_scale"][i], 
                                                repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                                repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                                repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"], 
                                                repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"], 
                                                repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"],
                                                repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"],
                                                repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_scale_v2(geom_num_list)
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_scale_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_scale_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

                elif force_data["repulsive_potential_v2_unit"][i] == "value":
                    LJRP = LJRepulsivePotential(repulsive_potential_v2_well_value=force_data["repulsive_potential_v2_well_scale"][i], 
                                                repulsive_potential_v2_dist_value=force_data["repulsive_potential_v2_dist_scale"][i], 
                                                repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                                repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                                repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"], 
                                                repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"], 
                                                repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"],
                                                repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"],
                                                repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_value_v2(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_value_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_value_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    print("error -rpv2")
                    raise "error -rpv2"
            else:
                pass

        #print("rp")
        #------------------
        for i in range(len(force_data["repulsive_potential_dist_scale"])):
            if force_data["repulsive_potential_well_scale"][i] != 0.0:
                if force_data["repulsive_potential_unit"][i] == "scale":
                    LJRP = LJRepulsivePotential(repulsive_potential_well_scale=force_data["repulsive_potential_well_scale"][i], 
                                                repulsive_potential_dist_scale=force_data["repulsive_potential_dist_scale"][i], 
                                                repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                                repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_scale(geom_num_list)
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_scale)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_scale)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    
                    
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

                elif force_data["repulsive_potential_unit"][i] == "value":
                    LJRP = LJRepulsivePotential(repulsive_potential_well_value=force_data["repulsive_potential_well_scale"][i], 
                                                repulsive_potential_dist_value=force_data["repulsive_potential_dist_scale"][i], 
                                                repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                                repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_value(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_value)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_value)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    print("error -rpv2")
                    raise "error -rpv2"
            else:
                pass
        
    
        
        #------------------
        for i in range(len(force_data["cone_potential_well_value"])):
            if force_data["cone_potential_well_value"][i] != 0.0:
              
                LJRP = LJRepulsivePotential(cone_potential_well_value=force_data["cone_potential_well_value"][i], 
                                            cone_potential_dist_value=force_data["cone_potential_dist_value"][i], 
                                            cone_potential_cone_angle=force_data["cone_potential_cone_angle"][i],
                                            cone_potential_center=force_data["cone_potential_center"][i],
                                            cone_potential_three_atoms=force_data["cone_potential_three_atoms"][i],
                                            cone_potential_target=force_data["cone_potential_target"][i],
                                            element_list=element_list
                                            )

                B_e += LJRP.calc_cone_potential_energy(geom_num_list)
                tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_cone_potential_energy)(geom_num_list).view(len(geom_num_list), 3)
                
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(LJRP.calc_cone_potential_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))

                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
       
        #------------------
        
        for i in range(len(force_data["keep_pot_spring_const"])):
            if force_data["keep_pot_spring_const"][i] != 0.0:
                SKP = StructKeepPotential(keep_pot_spring_const=force_data["keep_pot_spring_const"][i], 
                                            keep_pot_distance=force_data["keep_pot_distance"][i], 
                                            keep_pot_atom_pairs=force_data["keep_pot_atom_pairs"][i])
                
                B_e += SKP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SKP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(SKP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            else:
                pass
        #print("akp")
        #------------------        
        for i in range(len(force_data["anharmonic_keep_pot_spring_const"])):
            if force_data["anharmonic_keep_pot_spring_const"][i] != 0.0:
                SAKP = StructAnharmonicKeepPotential(anharmonic_keep_pot_spring_const=force_data["anharmonic_keep_pot_spring_const"][i], 
                                            anharmonic_keep_pot_potential_well_depth=force_data["anharmonic_keep_pot_potential_well_depth"][i], 
                                            anharmonic_keep_pot_atom_pairs=force_data["anharmonic_keep_pot_atom_pairs"][i],
                                            anharmonic_keep_pot_distance=force_data["anharmonic_keep_pot_distance"][i])
                
                B_e += SAKP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SAKP.calc_energy)(geom_num_list)
            
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                
                tensor_BPA_hessian = torch.func.hessian(SAKP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
               
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
     
        #------------------
        

        #print("wp")
        for i in range(len(force_data["well_pot_wall_energy"])):
            if force_data["well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(well_pot_wall_energy=force_data["well_pot_wall_energy"][i], 
                                            well_pot_fragm_1=force_data["well_pot_fragm_1"][i], 
                                            well_pot_fragm_2=force_data["well_pot_fragm_2"][i], 
                                            well_pot_limit_dist=force_data["well_pot_limit_dist"][i])
                
                
                B_e += WP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------
        

        #print("wp")
        for i in range(len(force_data["wall_well_pot_wall_energy"])):
            if force_data["wall_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(wall_well_pot_wall_energy=force_data["wall_well_pot_wall_energy"][i],
                                            wall_well_pot_direction=force_data["wall_well_pot_direction"][i], 
                                            wall_well_pot_limit_dist=force_data["wall_well_pot_limit_dist"][i],
                                            wall_well_pot_target=force_data["wall_well_pot_target"][i])
                
                B_e += WP.calc_energy_wall(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_wall)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_wall)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------
        

        #print("wp")
        for i in range(len(force_data["void_point_well_pot_wall_energy"])):
            if force_data["void_point_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(void_point_well_pot_wall_energy=force_data["void_point_well_pot_wall_energy"][i], 
                                            void_point_well_pot_coordinate=force_data["void_point_well_pot_coordinate"][i], 
                                            void_point_well_pot_limit_dist=force_data["void_point_well_pot_limit_dist"][i],
                                            void_point_well_pot_target=force_data["void_point_well_pot_target"][i])
                
                B_e += WP.calc_energy_vp(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_vp)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_vp)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
                
            
        #------------------    

        for i in range(len(force_data["around_well_pot_wall_energy"])):
            if force_data["around_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(around_well_pot_wall_energy=force_data["around_well_pot_wall_energy"][i], 
                                            around_well_pot_center=force_data["around_well_pot_center"][i], 
                                            around_well_pot_limit_dist=force_data["around_well_pot_limit_dist"][i],
                                            around_well_pot_target=force_data["around_well_pot_target"][i])
                
                B_e += WP.calc_energy_around(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_around)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_around)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
                
            
        #------------------   
        
        if len(geom_num_list) > 2:
            for i in range(len(force_data["keep_angle_spring_const"])):
                if force_data["keep_angle_spring_const"][i] != 0.0:
                    SKAngleP = StructKeepAnglePotential(keep_angle_atom_pairs=force_data["keep_angle_atom_pairs"][i], 
                                                keep_angle_spring_const=force_data["keep_angle_spring_const"][i], 
                                                keep_angle_angle=force_data["keep_angle_angle"][i])
                    
                    B_e += SKAngleP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKAngleP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKAngleP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        
        if len(geom_num_list) > 2:
            for i in range(len(force_data["aDD_keep_angle_spring_const"])):
                if force_data["aDD_keep_angle_spring_const"][i] != 0.0:
                    aDDKAngleP = StructKeepAnglePotential(aDD_keep_angle_spring_const=force_data["aDD_keep_angle_spring_const"][i], 
                                                aDD_keep_angle_min_angle=force_data["aDD_keep_angle_min_angle"][i], 
                                                aDD_keep_angle_max_angle=force_data["aDD_keep_angle_max_angle"][i],
                                                aDD_keep_angle_base_dist=force_data["aDD_keep_angle_base_dist"][i],
                                                aDD_keep_angle_reference_atom=force_data["aDD_keep_angle_reference_atom"][i],
                                                aDD_keep_angle_center_atom=force_data["aDD_keep_angle_center_atom"][i],
                                                aDD_keep_angle_atoms=force_data["aDD_keep_angle_atoms"][i])

                    B_e += aDDKAngleP.calc_atom_dist_dependent_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(aDDKAngleP.calc_atom_dist_dependent_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(aDDKAngleP.calc_atom_dist_dependent_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_dihedral_angle_spring_const"])):
                if force_data["keep_dihedral_angle_spring_const"][i] != 0.0:
                    SKDAP = StructKeepDihedralAnglePotential(keep_dihedral_angle_spring_const=force_data["keep_dihedral_angle_spring_const"][i], 
                                                keep_dihedral_angle_atom_pairs=force_data["keep_dihedral_angle_atom_pairs"][i], 
                                                keep_dihedral_angle_angle=force_data["keep_dihedral_angle_angle"][i])
                    
                    B_e += SKDAP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKDAP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKDAP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    pass
        else:
            pass

        #------------------
        for i in range(len(force_data["void_point_pot_spring_const"])):
            if force_data["void_point_pot_spring_const"][i] != 0.0:
                for j in force_data["void_point_pot_atoms"][i]:
                    VPP = VoidPointPotential(void_point_pot_spring_const=force_data["void_point_pot_spring_const"][i], 
                                            void_point_pot_atoms=j, 
                                            void_point_pot_coord=self.ndarray2tensor(np.array(force_data["void_point_pot_coord"][i], dtype="float64")),
                                            void_point_pot_distance=force_data["void_point_pot_distance"][i],
                                            void_point_pot_order=force_data["void_point_pot_order"][i])
                                            
                    
                    B_e += VPP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(VPP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(VPP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
              
            else:
                pass
        
        #------------------
        for i in range(len(force_data["AFIR_gamma"])):
            if force_data["AFIR_gamma"][i] != 0.0:
                AP = AFIRPotential(AFIR_gamma=force_data["AFIR_gamma"][i], 
                                            AFIR_Fragm_1=force_data["AFIR_Fragm_1"][i], 
                                            AFIR_Fragm_2=force_data["AFIR_Fragm_2"][i],
                                            element_list=element_list)
                
                B_e += AP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(AP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                
                tensor_BPA_hessian = torch.func.hessian(AP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------        
        B_g = g + BPA_grad_list

        
        
        B_e = B_e.item()
        #new_geometry:ang. 
        #B_e:hartree

        return BPA_grad_list, B_e, B_g, BPA_hessian