
from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from calc_tools import Calculationtools

import numpy as np
import torch
import itertools  
import math
  
      
class SpacerModelPotential:
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
        
        self.a = 1.0
        self.num2tgtatomlabel = {}
        for num, tgt_atom_num in enumerate(self.config["spacer_model_potential_target"]):
            self.num2tgtatomlabel[num] = tgt_atom_num -1

        return

    def save_spacer_xyz_for_visualization(self, geom_num_list, particle_num_list):
        with open(self.config["directory"]+"/tmp_smp.xyz", "a") as f:
            f.write(str(len(geom_num_list)+len(particle_num_list))+"\n\n")
            for i in range(len(self.config["element_list"])):
                f.write(self.config["element_list"][i]+" "+str(geom_num_list[i][0].item()*self.bohr2angstroms)+" "+str(geom_num_list[i][1].item()*self.bohr2angstroms)+" "+str(geom_num_list[i][2].item()*self.bohr2angstroms)+"\n")
            
            for i in range(len(particle_num_list)):
                f.write("He "+str(particle_num_list[i][0].item()*self.bohr2angstroms)+" "+str(particle_num_list[i][1].item()*self.bohr2angstroms)+" "+str(particle_num_list[i][2].item()*self.bohr2angstroms)+"\n")
                
        return


    def morse_potential(self, distance, sigma, epsilon):
        ene = epsilon * (torch.exp(-2 * self.a * (distance - sigma)) -2 * torch.exp(-1 * self.a * (distance - sigma)))
        return ene

    def barrier_potential(self, distance, sigma):
        normalized_distance = distance / sigma
        if normalized_distance >= 0.9:
            ene = 100.0 * 10 ** 2 * (normalized_distance - 0.9) ** 2
        else:
            ene = 0.0 
        return ene

    def calc_energy(self, geom_num_list, particle_num_list):
        """
        # required variables:self.config["spacer_model_potential_target"]
                             self.config["spacer_model_potential_distance"]
                             self.config["spacer_model_potential_well_depth"]
                             self.config["spacer_model_potential_cavity_scaling"]
                             self.config["element_list"]
                             self.config["directory"]
        """
        energy = 0.0
        particle_sigma = self.config["spacer_model_potential_distance"] / self.bohr2angstroms
        particle_epsilon = self.config["spacer_model_potential_well_depth"] / self.hartree2kjmol
        #atom-particle interactions
        for i, j in itertools.product(range(len(self.config["spacer_model_potential_target"])), range(len(particle_num_list))):
            atom_sigma = self.VDW_distance_lib(self.config["element_list"][self.config["spacer_model_potential_target"][i]-1])
            atom_epsilon = self.VDW_well_depth_lib(self.config["element_list"][self.config["spacer_model_potential_target"][i]-1])
            sigma = math.sqrt(particle_sigma * atom_sigma)
            epsilon = math.sqrt(particle_epsilon * atom_epsilon)
            distance = torch.linalg.norm(geom_num_list[self.config["spacer_model_potential_target"][i]-1] - particle_num_list[j])
            energy = energy + self.morse_potential(distance, sigma, epsilon)
            
        #particle-particle interactions
        for i, j in itertools.combinations(range(len(particle_num_list)), 2):
            distance = torch.linalg.norm(particle_num_list[i] - particle_num_list[j]) 
            energy = energy + self.morse_potential(distance, particle_sigma, particle_epsilon)

        #avoid scattering particle to outside of cavity
        for i in range(len(particle_num_list)):
            norm_list = torch.abs(torch.linalg.norm(geom_num_list[np.array(self.config["spacer_model_potential_target"])-1], dim=1) - torch.linalg.norm(particle_num_list[i]))
            min_idx = torch.argmin(norm_list).item()
            min_dist = norm_list[min_idx]
            
            atom_sigma = self.config["spacer_model_potential_cavity_scaling"] * self.VDW_distance_lib(self.config["element_list"][self.num2tgtatomlabel[min_idx]])
            energy = energy + self.barrier_potential(min_dist, atom_sigma)

        return energy

