
from multioptpy.Parameters.parameter import UnitValueLib
import torch
  
class StructKeepPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_pot_spring_const"], 
                             self.config["keep_pot_distance"], 
                             self.config["keep_pot_atom_pairs"],
        bias_pot_params[0] : keep_pot_spring_const
        bias_pot_params[1] : keep_pot_distance
        """
        vector = torch.linalg.norm((geom_num_list[self.config["keep_pot_atom_pairs"][0]-1] - geom_num_list[self.config["keep_pot_atom_pairs"][1]-1]), ord=2)
        if len(bias_pot_params) == 0:
            energy = 0.5 * self.config["keep_pot_spring_const"] * (vector - self.config["keep_pot_distance"]/self.bohr2angstroms) ** 2
        else:
            energy = 0.5 * bias_pot_params[0] * (vector - bias_pot_params[1]/self.bohr2angstroms) ** 2
        return energy #hartree
    
class StructKeepPotentialv2:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_pot_v2_spring_const"], 
                             self.config["keep_pot_v2_distance"], 
                             self.config["keep_pot_v2_fragm1"],
                             self.config["keep_pot_v2_fragm2"],
        bias_pot_params[0] : keep_pot_v2_spring_const
        bias_pot_params[1] : keep_pot_v2_distance    
        """
        fragm_1_indices = torch.tensor(self.config["keep_pot_v2_fragm1"]) - 1
        fragm_2_indices = torch.tensor(self.config["keep_pot_v2_fragm2"]) - 1

        fragm_1_center = torch.mean(geom_num_list[fragm_1_indices], dim=0)
        fragm_2_center = torch.mean(geom_num_list[fragm_2_indices], dim=0)
        
        distance = torch.linalg.norm(fragm_1_center - fragm_2_center, ord=2)
        if len(bias_pot_params) == 0:
            energy = 0.5 * self.config["keep_pot_v2_spring_const"] * (distance - self.config["keep_pot_v2_distance"]/self.bohr2angstroms) ** 2
        
        else:
            energy = 0.5 * bias_pot_params[0] * (distance - bias_pot_params[1]/self.bohr2angstroms) ** 2
        
        return energy #hartree
    
class StructKeepPotentialAniso:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables:   self.config["aniso_keep_pot_v2_spring_const_mat"]
                                self.config["aniso_keep_pot_v2_dist"] 
                                self.config["aniso_keep_pot_v2_fragm1"]
                                self.config["aniso_keep_pot_v2_fragm2"]
                             
        """
        fragm_1_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["aniso_keep_pot_v2_fragm1"]:
            fragm_1_center = fragm_1_center + geom_num_list[i-1]
        
        fragm_1_center = fragm_1_center / len(self.config["aniso_keep_pot_v2_fragm1"])
        
        fragm_2_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["aniso_keep_pot_v2_fragm2"]:
            fragm_2_center = fragm_2_center + geom_num_list[i-1]
        
        fragm_2_center = fragm_2_center / len(self.config["aniso_keep_pot_v2_fragm2"])     
        x_dist = torch.abs(fragm_1_center[0] - fragm_2_center[0])
        y_dist = torch.abs(fragm_1_center[1] - fragm_2_center[1])
        z_dist = torch.abs(fragm_1_center[2] - fragm_2_center[2])
        eq_dist = self.config["aniso_keep_pot_v2_dist"]  / (3 ** 0.5) / self.bohr2angstroms
        dist_vec = torch.stack([(x_dist - eq_dist) ** 2,(y_dist - eq_dist) ** 2,(z_dist - eq_dist) ** 2])
        dist_vec = torch.reshape(dist_vec, (3, 1))
        vec_pot = torch.matmul(torch.tensor(self.config["aniso_keep_pot_v2_spring_const_mat"], dtype=torch.float32), dist_vec)
        
        energy = torch.sum(vec_pot)
        
        
        return energy #hartree