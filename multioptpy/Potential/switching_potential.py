
from multioptpy.Parameters.parameter import UnitValueLib
import torch

class WellPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["well_pot_wall_energy"]
                              self.config["well_pot_fragm_1"]
                              self.config["well_pot_fragm_2"]
                              self.config["well_pot_limit_dist"]
                              
                              
        """
        fragm_1_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        for i in self.config["well_pot_fragm_1"]:
            fragm_1_center = fragm_1_center + geom_num_list[i-1]
        
        fragm_1_center = fragm_1_center / len(self.config["well_pot_fragm_1"])
        
        fragm_2_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
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
            energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
            
        elif c <= vec_norm and vec_norm < d:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
            
        elif d <= vec_norm:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
            
        else:
            print("well pot error")
            raise "well pot error"
        #print(energy)
        return energy


class WellPotentialWall:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
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
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
    
class WellPotentialVP:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["void_point_well_pot_wall_energy"]
                              self.config["void_point_well_pot_coordinate"] 
                              self.config["void_point_well_pot_limit_dist"]
                              self.config["void_point_well_pot_target"]
                              
                              
        """
        self.config["void_point_well_pot_coordinate"]  = torch.tensor(self.config["void_point_well_pot_coordinate"], dtype=torch.float32)

        
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
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
    
    
class WellPotentialAround:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return 
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["around_well_pot_wall_energy"]
                              self.config["around_well_pot_center"] 
                              self.config["around_well_pot_limit_dist"]
                              self.config["around_well_pot_target"]
                              
                              
        """
        geom_center_coord = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
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
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy