from parameter import UnitValueLib
from calc_tools import torch_calc_angle_from_vec
import torch

class StructKeepArchPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def save_arch_xyz_for_visualization(self, geom_num_list):
        with open(self.config["directory"]+"/tmp_arch.xyz", "a") as f:
            f.write(str(len(self.config["element_list"])+2)+"\n\n")
            for i in range(len(self.config["element_list"])):
                f.write(self.config["element_list"][i]+" "+str(geom_num_list[i][0].item()*self.bohr2angstroms)+" "+str(geom_num_list[i][1].item()*self.bohr2angstroms)+" "+str(geom_num_list[i][2].item()*self.bohr2angstroms)+"\n")
            f.write("X "+str(self.circle_center[0].item()*self.bohr2angstroms)+" "+str(self.circle_center[1].item()*self.bohr2angstroms)+" "+str(self.circle_center[2].item()*self.bohr2angstroms)+"\n")
            f.write("X "+str(self.center_H1_H2[0].item()*self.bohr2angstroms)+" "+str(self.center_H1_H2[1].item()*self.bohr2angstroms)+" "+str(self.center_H1_H2[2].item()*self.bohr2angstroms)+"\n")
                
        return
    
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        energy = 0.0 
        """
        # required variables: 
                              self.config["keep_arch_dist_spring_const"],
                              self.config["keep_arch_angle_spring_const"],
                              self.config["keep_arch_angle_tangent_spring_const"],
                              self.config["keep_arch_radius"],
                              self.config["keep_arch_angle"],
                              self.config["keep_arch_tangent_angle"],
                              self.config["keep_arch_atoms"]   (atom1(tangent),atom2(tangent),atom3,atom4)

        """
        center_H1_H2 = (geom_num_list[self.config["keep_arch_atoms"][2]-1] + geom_num_list[self.config["keep_arch_atoms"][3]-1]) / 2.0
        vec_P1_center = geom_num_list[self.config["keep_arch_atoms"][0]-1] - center_H1_H2
        vec_P2_center = geom_num_list[self.config["keep_arch_atoms"][1]-1] - center_H1_H2
        cross_vec_P12 = torch.linalg.cross(vec_P1_center, vec_P2_center)
        
        proj_unit_vec_x = vec_P1_center / torch.linalg.norm(vec_P1_center)
        proj_vec_y = torch.linalg.cross(proj_unit_vec_x, cross_vec_P12)
        proj_unit_vec_y = proj_vec_y / torch.linalg.norm(proj_vec_y)
        
        scalar_proj_unit_vec_x = torch.linalg.norm(vec_P1_center) / 2.0
        scalar_proj_unit_vec_y = ((torch.sum(vec_P2_center * proj_unit_vec_y)) ** 2 + (torch.sum(vec_P2_center * proj_unit_vec_x)) ** 2 -1 * torch.linalg.norm(vec_P1_center) * (torch.sum(vec_P2_center * proj_unit_vec_x))) / (2.0 * torch.sum(vec_P2_center * proj_unit_vec_y))
        circle_center = center_H1_H2 + proj_unit_vec_x * scalar_proj_unit_vec_x + proj_unit_vec_y * scalar_proj_unit_vec_y
        self.circle_center = circle_center
        self.center_H1_H2 = center_H1_H2
        #test
        #print("### debug info ###")
        #print(circle_center)
        #print(self.config["keep_arch_radius"] / self.bohr2angstroms )
        #print(torch.linalg.norm(geom_num_list[self.config["keep_arch_atoms"][0]-1] - circle_center))
        #print(torch.linalg.norm(center_H1_H2 - circle_center))
        #print(torch.linalg.norm(geom_num_list[self.config["keep_arch_atoms"][1]-1] - circle_center))

        vec_P1_O = geom_num_list[self.config["keep_arch_atoms"][0]-1] - circle_center
        vec_P2_O = geom_num_list[self.config["keep_arch_atoms"][1]-1] - circle_center
        vec_H1_H2_O = center_H1_H2 - circle_center
        
        #energy = energy + 0.5 * self.config["keep_arch_dist_spring_const"] * (torch.linalg.norm(vec_P1_O) - self.config["keep_arch_radius"] / self.bohr2angstroms) ** 2 
        #energy = energy + 0.5 * self.config["keep_arch_dist_spring_const"] * (torch.linalg.norm(vec_P2_O) - self.config["keep_arch_radius"] / self.bohr2angstroms) ** 2 
        energy = energy + 0.5 * self.config["keep_arch_dist_spring_const"] * (torch.linalg.norm(vec_H1_H2_O) - self.config["keep_arch_radius"] / self.bohr2angstroms) ** 2 
        
        
        theta = torch_calc_angle_from_vec(vec_P1_O, vec_P2_O)

        energy = energy + 0.5 * self.config["keep_arch_angle_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["keep_arch_angle"], dtype=torch.float64))) ** 2

        vec_H1_P1 = geom_num_list[self.config["keep_arch_atoms"][2]-1] - geom_num_list[self.config["keep_arch_atoms"][0]-1]
        vec_H2_P2 = geom_num_list[self.config["keep_arch_atoms"][3]-1] - geom_num_list[self.config["keep_arch_atoms"][1]-1]
        
        vec_circle_center_P1 = circle_center - geom_num_list[self.config["keep_arch_atoms"][0]-1]
        vec_circle_center_P2 = circle_center - geom_num_list[self.config["keep_arch_atoms"][1]-1]
        
        tangent_line_vec_P1 = torch.linalg.cross(cross_vec_P12, vec_circle_center_P1)
        tangent_line_vec_P2 = torch.linalg.cross(-1*cross_vec_P12, vec_circle_center_P2)

        theta_t1 = torch_calc_angle_from_vec(tangent_line_vec_P1, vec_H1_P1)
        theta_t2 = torch_calc_angle_from_vec(tangent_line_vec_P2, vec_H2_P2)

        energy = energy + 0.5 * self.config["keep_arch_angle_tangent_spring_const"] * (theta_t1 - torch.deg2rad(torch.tensor(self.config["keep_arch_tangent_angle"], dtype=torch.float64))) ** 2
        energy = energy + 0.5 * self.config["keep_arch_angle_tangent_spring_const"] * (theta_t2 - torch.deg2rad(torch.tensor(self.config["keep_arch_tangent_angle"], dtype=torch.float64))) ** 2

        
 
        return energy #hartree
    
class StructKeepArchPotentialv2:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def save_arch_xyz_for_visualization(self, geom_num_list):
        with open(self.config["directory"]+"/tmp_arch.xyz", "a") as f:
            f.write(str(len(self.config["element_list"])+2)+"\n\n")
            for i in range(len(self.config["element_list"])):
                f.write(self.config["element_list"][i]+" "+str(geom_num_list[i][0].item()*self.bohr2angstroms)+" "+str(geom_num_list[i][1].item()*self.bohr2angstroms)+" "+str(geom_num_list[i][2].item()*self.bohr2angstroms)+"\n")
            f.write("X "+str(self.circle_center[0].item()*self.bohr2angstroms)+" "+str(self.circle_center[1].item()*self.bohr2angstroms)+" "+str(self.circle_center[2].item()*self.bohr2angstroms)+"\n")
            f.write("X "+str(self.center_H1_H2[0].item()*self.bohr2angstroms)+" "+str(self.center_H1_H2[1].item()*self.bohr2angstroms)+" "+str(self.center_H1_H2[2].item()*self.bohr2angstroms)+"\n")
                
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        energy = 0.0 
        """
        # required variables: 
                              self.config["keep_arch_v2_dist_spring_const"],
                              self.config["keep_arch_v2_angle_spring_const"],
                              self.config["keep_arch_v2_angle"],
                              self.config["keep_arch_v2_radius"],
                              self.config["keep_arch_v2_atoms"]   (atom1(tangent),atom2(tangent),atom3,atom4)

        """
        center_H1_H2 = (geom_num_list[self.config["keep_arch_v2_atoms"][2]-1] + geom_num_list[self.config["keep_arch_v2_atoms"][3]-1]) / 2.0
        vec_P1_center = geom_num_list[self.config["keep_arch_v2_atoms"][0]-1] - center_H1_H2
        vec_P2_center = geom_num_list[self.config["keep_arch_v2_atoms"][1]-1] - center_H1_H2
        vec = vec_P2_center + vec_P1_center
        unit_vec = (vec) / torch.linalg.norm(vec)
        circle_center = center_H1_H2 + unit_vec * self.config["keep_arch_v2_radius"] / self.bohr2angstroms
        
        vec_1_from_cc = geom_num_list[self.config["keep_arch_v2_atoms"][0]-1] - circle_center
        vec_2_from_cc = geom_num_list[self.config["keep_arch_v2_atoms"][1]-1] - circle_center
        
        dist_1 = torch.linalg.norm(vec_1_from_cc)
        dist_2 = torch.linalg.norm(vec_2_from_cc)
        
        energy = energy + 0.5 * self.config["keep_arch_v2_dist_spring_const"] * (dist_1 - self.config["keep_arch_v2_radius"] / self.bohr2angstroms) ** 2
        energy = energy + 0.5 * self.config["keep_arch_v2_dist_spring_const"] * (dist_2 - self.config["keep_arch_v2_radius"] / self.bohr2angstroms) ** 2
        
        angle = torch_calc_angle_from_vec(vec_1_from_cc, vec_2_from_cc)
        energy = energy + 0.5 * self.config["keep_arch_v2_angle_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_arch_v2_angle"], dtype=torch.float64))) ** 2
        self.circle_center = circle_center
        self.center_H1_H2 = center_H1_H2
        return energy #hartree