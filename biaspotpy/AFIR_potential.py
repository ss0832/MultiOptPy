from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from calc_tools import Calculationtools

import itertools
import math
import numpy as np
import torch


        
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