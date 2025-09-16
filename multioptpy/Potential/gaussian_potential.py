from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_outofplain_angle_from_vec, torch_calc_dihedral_angle_from_vec, torch_calc_angle_from_vec

import torch

class GaussianPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol
        self.history_list = None
        self.add_history = True
        return
    
    def calc_energy(self, A, B, C, x):
        energy = A * torch.exp(-1 * (x - B) ** 2 / (2 * C ** 2))
        return energy
        
    
    def calc_energy_for_metadyn(self, geom_num_list):
        """
        # required variables: self.config["gaussian_potential_width"], # ang., deg, ...
                              self.config["gaussian_potential_height"], # kj/mol
                              self.config["gaussian_potential_target"]
                              self.config["gaussian_potential_tgt_atom"]
        """
        
        if self.history_list is None:
            print("error")
            raise
        
        energy = torch.tensor(0.0, requires_grad=True)
     
        for i in range(len(self.config["gaussian_potential_target"])):
            target = self.config["gaussian_potential_target"][i]
            #target: tgt variable [0], tgt atom [1]
         
            if target == "bond":
                vector = torch.linalg.norm((geom_num_list[self.config["gaussian_potential_tgt_atom"][i][0]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][1]-1]), ord=2)#Bohr
                #for history_bond in self.history_list[i]:
                gau_energy = self.calc_energy(torch.tensor(self.config["gaussian_potential_height"][i] / self.hartree2kjmol), torch.tensor(self.history_list[i]), self.config["gaussian_potential_width"][i] / self.bohr2angstroms, vector)
                
                if len(gau_energy) > 0:
                    energy = energy + torch.sum(gau_energy)
                    
                if self.add_history:
                    self.history_list[i].append(vector)
                
                
            elif target == "angle":
                vector1 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][0]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][1]-1]
                vector2 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][2]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][1]-1]
                theta = torch_calc_angle_from_vec(vector1, vector2)
                
  
                gau_energy = self.calc_energy(torch.tensor(self.config["gaussian_potential_height"][i] / self.hartree2kjmol), torch.tensor(self.history_list[i]), torch.deg2rad(torch.tensor(self.config["gaussian_potential_width"][i])), theta)
                if len(gau_energy) > 0:
                    energy = energy + torch.sum(gau_energy)
                
                if self.add_history:
                    self.history_list[i].append(theta)
                
            elif target == "dihedral":
                a1 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][1]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][0]-1]
                a2 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][2]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][1]-1]
                a3 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][3]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][2]-1]
                angle = torch_calc_dihedral_angle_from_vec(a1, a2, a3) % (2 * torch.pi)
                
                
                gau_energy = self.calc_energy(torch.tensor(self.config["gaussian_potential_height"][i] / self.hartree2kjmol), torch.tensor(self.history_list[i]), torch.deg2rad(torch.tensor(self.config["gaussian_potential_width"][i])), angle)
                if len(gau_energy) > 0:
                    energy = energy + torch.sum(gau_energy)
                    
                if self.add_history:
                    self.history_list[i].append(angle)
                
                
            elif target == "outofplain":
                a1 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][1]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][0]-1]
                a2 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][2]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][0]-1]
                a3 = geom_num_list[self.config["gaussian_potential_tgt_atom"][i][3]-1] - geom_num_list[self.config["gaussian_potential_tgt_atom"][i][0]-1]
                angle = torch_calc_outofplain_angle_from_vec(a1, a2, a3) % (2 * torch.pi)
                
          
                gau_energy = self.calc_energy(torch.tensor(self.config["gaussian_potential_height"][i] / self.hartree2kjmol), torch.tensor(self.history_list[i]), torch.deg2rad(torch.tensor(self.config["gaussian_potential_width"][i])), angle)
                if len(gau_energy) > 0:
                    energy = energy + torch.sum(gau_energy)
                    
                if self.add_history:
                    self.history_list[i].append(angle)
                
                
            else:
                print("target variable error")
                raise
            
    
        self.add_history = False
        return energy