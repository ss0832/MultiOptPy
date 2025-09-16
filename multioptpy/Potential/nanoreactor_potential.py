
from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass

import torch
  
  
      
class NanoReactorPotential:
    def __init__(self, **kwarg):
        # ref.:https://doi.org/10.1038/nchem.2099, https://doi.org/10.1021/acs.jctc.4c00826
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        self.au2sec = UVL.au2sec
        self.inner_wall = torch.tensor(self.config["inner_wall"] / self.bohr2angstroms, dtype=torch.float64)
        self.outer_wall = torch.tensor(self.config["outer_wall"] / self.bohr2angstroms, dtype=torch.float64)
        self.contraction_time = torch.tensor(self.config["contraction_time"] * 10 ** -12 / self.au2sec, dtype=torch.float64) #pico-sec to au
        self.expansion_time = torch.tensor(self.config["expansion_time"] * 10 ** -12 / self.au2sec, dtype=torch.float64) #pico-sec to au
        
        self.contraction_force_const = self.config["contraction_force_const"] / self.hartree2kcalmol * self.bohr2angstroms ** 2 # kcal/mol/A^2 to hartree/bohr^2
        self.expansion_force_const = self.config["expansion_force_const"] / self.hartree2kcalmol * self.bohr2angstroms ** 2 # kcal/mol/A^2 to hartree/bohr^2
        self.element_list = self.config["element_list"]
        self.atom_mass_list = torch.tensor([[atomic_mass(element)] for element in self.element_list], dtype=torch.float64)

        return
    
    def calc_energy(self, geom_num_list, time):#geom_num_list: (n_atoms, 3), bohr time: au 
        """
        # required variables: self.inner_wall, 
                             self.outer_wall, 
                             self.contraction_time,
                             self.expansion_time,
                             self.contraction_force_const,
                             self.expansion_force_const,
                             self.element_list
        """
        distance_list = torch.linalg.norm(geom_num_list, ord=2, dim=1).reshape(-1, 1)
        distance_inner = distance_list - self.inner_wall
        distance_outer = distance_list - self.outer_wall
        
        f_t = torch.heaviside(torch.floor(time / (self.contraction_time + self.expansion_time)) - (time / (self.contraction_time + self.expansion_time)) + (self.contraction_time / (self.contraction_time + self.expansion_time)), torch.tensor(0.5, dtype=torch.float64))
        
        
        U_c = torch.where(distance_list < self.inner_wall, self.atom_mass_list * 0.5 * self.contraction_force_const * distance_inner ** 2, torch.zeros_like(distance_inner))
        
        U_e = torch.where(distance_list > self.outer_wall, self.atom_mass_list * 0.5 * self.contraction_force_const * distance_outer ** 2,
                          torch.where(distance_list < self.inner_wall, self.atom_mass_list * 0.5 * self.expansion_force_const * distance_inner ** 2, torch.zeros_like(distance_list)))
        energy = torch.sum(f_t * U_c + (1.0 - f_t) * U_e)
        return energy #hartree
    