from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib

import math
import torch

        
class AFIRPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        self.R_0 = 3.8164/self.bohr2angstroms #ang.→bohr
        self.EPSIRON = 1.0061/self.hartree2kjmol #kj/mol→hartree
        self.p = 6.0
        return
    def calc_energy(self, geom_num_list, bias_pot_params):
        """
        # required variables: 
                             self.config["AFIR_Fragm_1"], 
                             self.config["AFIR_Fragm_2"],
                             self.config["element_list"]
                             bias_pot_params[0] : AFIR_gamma
        """
        """
        ###  Reference  ###
            Chem. Rec., 2016, 16, 2232
            J. Comput. Chem., 2018, 39, 233
            WIREs Comput. Mol. Sci., 2021, 11, e1538
        """

        if bias_pot_params[0] > 0.0 or bias_pot_params[0] < 0.0:
            alpha = (bias_pot_params[0]/self.hartree2kjmol) / ((2 ** (-1/6) - (1 + math.sqrt(1 + (abs(bias_pot_params[0]/self.hartree2kjmol) / self.EPSIRON))) ** (-1/6))*self.R_0) #hartree/Bohr
        else:
            alpha = 0.0
        A = 0.0
        B = 0.0
        
        i_indices = torch.tensor(self.config["AFIR_Fragm_1"]) - 1  # 0-based index
        j_indices = torch.tensor(self.config["AFIR_Fragm_2"]) - 1  # 0-based index

        R_i = torch.tensor([covalent_radii_lib(self.config["element_list"][i.item()]) for i in i_indices])  # shape: (M,)
        R_j = torch.tensor([covalent_radii_lib(self.config["element_list"][j.item()]) for j in j_indices])  # shape: (M,)

        geom_diff = geom_num_list[i_indices].unsqueeze(1) - geom_num_list[j_indices].unsqueeze(0)  # shape: (M, N, 3)
        vector = torch.linalg.norm(geom_diff, dim=2)  # shape: (M, N)

        omega = ((R_i.unsqueeze(1) + R_j.unsqueeze(0)) / vector) ** self.p  # shape: (M, N)

        A = (omega * vector).sum() 
        B = omega.sum()  
        
        energy = alpha*(A/B)#A/B:Bohr
        return energy #hartree