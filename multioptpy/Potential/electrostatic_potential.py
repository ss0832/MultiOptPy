
from multioptpy.Parameters.parameter import UnitValueLib, UFF_effective_charge_lib

import itertools
import torch

class ElectroStaticPotentialFragment:
    def __init__(self, pot_type="UFF", **kwarg):
        if pot_type == "UFF":
            self.effective_charge_lib = UFF_effective_charge_lib #function
        else:
            raise "Please input MM potential type."
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]): #For QM/MM interaction
        epsilon = 1.0
        """
        # required variables:self.config["es_charge_scale"], 
                             self.config["es_Fragm_1"],
                             self.config["es_Fragm_2"]
                             self.config["element_list"]
        """
        energy = 0.0

        for i, j in itertools.product(self.config["es_Fragm_1"], self.config["es_Fragm_2"]):
            electrostaticcharge = self.config["es_charge_scale"]*self.effective_charge_lib(self.config["element_list"][i-1])*self.effective_charge_lib(self.config["element_list"][j-1])
           
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) * self.bohr2angstroms #ang.
            energy += ((332.0637 * electrostaticcharge) / (epsilon * vector)) / self.hartree2kcalmol
            
        return energy
    
class ElectroStaticPotentialAtomPair:
    def __init__(self, pot_type="UFF", **kwarg):
        if pot_type == "UFF":
            self.effective_charge_lib = UFF_effective_charge_lib #function
        else:
            raise "Please input MM potential type."
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        epsilon = 1.0
        """
        # required variables:self.config["es_charge_scale"], 
                             self.config["es_atoms"],
                             self.config["element_list"]
        """
        energy = 0.0

        for i, j in itertools.combinations(self.config["es_atoms"], 2):
            electrostaticcharge = self.config["es_charge_scale"]*self.effective_charge_lib(self.config["element_list"][i-1])*self.effective_charge_lib(self.config["element_list"][j-1])
           
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) * self.bohr2angstroms #ang.
            energy += ((332.0637 * electrostaticcharge) / (epsilon * vector)) / self.hartree2kcalmol
            
        return energy