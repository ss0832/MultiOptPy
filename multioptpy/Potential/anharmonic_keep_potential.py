
from multioptpy.Parameters.parameter import UnitValueLib
import math
import torch

class StructAnharmonicKeepPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
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
            energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

        return energy