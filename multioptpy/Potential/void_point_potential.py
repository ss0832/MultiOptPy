
from multioptpy.Parameters.parameter import UnitValueLib
import torch

class VoidPointPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
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
