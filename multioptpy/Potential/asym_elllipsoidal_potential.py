from multioptpy.Parameters.parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, GNB_VDW_radii_lib, GNB_VDW_well_depth_lib
from multioptpy.Utils.calc_tools import torch_rotate_around_axis, torch_align_vector_with_z
from multioptpy.Optimizer.fire import FIRE

import torch
import copy
import random
import math

class AsymmetricEllipsoidalLJPotential:
    def __init__(self, **kwarg):
        #ref.: https://doi.org/10.26434/chemrxiv-2024-6www6
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        self.element_list = self.config["element_list"]
        self.file_directory = self.config["file_directory"]
        self.rot_angle_list = []
        npot = len(self.config["asymmetric_ellipsoidal_repulsive_potential_eps"])
        for i in range(npot):
            self.rot_angle_list.append(random.uniform(0, 2*math.pi))
        self.nangle = npot
        self.rot_angle_list = torch.tensor([self.rot_angle_list], dtype=torch.float64, requires_grad=True)
        
        self.lj_repulsive_order = 12.0
        self.lj_attractive_order = 6.0
        
        self.micro_iteration = 15000 * npot
        self.rand_search_iteration = 1000 * npot
        self.threshold = 1e-7
        self.init = True

        self.energy_analysis_dict = None
        self.energy_save_flag = False
        return
    
    def save_state(self):      
        with open(self.file_directory + "/asym_ellipsoid_v1.xyz", "a") as f:
            f.write(str(len(self.tmp_geom_num_list_for_save)) + "\n")
            f.write("AsymmetricEllipsoid\n")
            for i in range(len(self.tmp_geom_num_list_for_save)):
                f.write(self.tmp_element_list_for_save[i] + " " + str(self.tmp_geom_num_list_for_save[i][0].item()) + " " + str(self.tmp_geom_num_list_for_save[i][1].item()) + " " + str(self.tmp_geom_num_list_for_save[i][2].item()) + "\n")
        
        return
    
    def save_part_ene(self):
        with open(self.file_directory + "/asym_ellipsoid_p_ene.txt", "a") as f:
            f.write(str(self.energy_analysis_dict)+"\n")
        
        return

    def calc_potential(self, geom_num_list, rot_angle_list, bias_pot_params):
        energy = 0.0
        """
        self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"],
        self.config["asymmetric_ellipsoidal_repulsive_potential_offtgt"],
    
        """
        rot_angle_list = rot_angle_list[0]
        tmp_geom_num_list_for_save = geom_num_list * self.bohr2angstroms # save the geometry with asymmetric ellipsoid
        tmp_element_list_for_save = self.element_list
        
        
        # interaction between substrate and asymmetric ellipsoid
        for pot_i in range(len(bias_pot_params)):
           
            tgt_atom_list = [i for i in range(len(geom_num_list)) if not i+1 in self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_i] + self.config["asymmetric_ellipsoidal_repulsive_potential_offtgt"][pot_i]]
            root_atom = self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_i][0] - 1
            LJ_atom = self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_i][1] - 1
           
            asym_elip_eps = bias_pot_params[pot_i][0] / self.hartree2kjmol
            asym_elip_sig_xp = bias_pot_params[pot_i][1] / self.bohr2angstroms
            asym_elip_sig_xm = bias_pot_params[pot_i][2] / self.bohr2angstroms
            asym_elip_sig_yp = bias_pot_params[pot_i][3] / self.bohr2angstroms
            asym_elip_sig_ym = bias_pot_params[pot_i][4] / self.bohr2angstroms
            asym_elip_sig_zp = bias_pot_params[pot_i][5] / self.bohr2angstroms
            asym_elip_sig_zm = bias_pot_params[pot_i][6] / self.bohr2angstroms
            asym_elip_dist = bias_pot_params[pot_i][7] / self.bohr2angstroms
            
            # rotate the asymmetric ellipsoid
            LJ_vec = geom_num_list[LJ_atom] - geom_num_list[root_atom]
            LJ_vec = LJ_vec / torch.norm(LJ_vec)
            LJ_center = geom_num_list[root_atom] + LJ_vec * asym_elip_dist
            z_axis_adjust_rot_mat = torch_align_vector_with_z(LJ_vec)
            z_axis_adjusted_tmp_geom_num_list = torch.matmul(z_axis_adjust_rot_mat, (geom_num_list - geom_num_list[root_atom]).T).T
            z_axis_adjusted_LJ_center = torch.matmul(z_axis_adjust_rot_mat, (LJ_center - geom_num_list[root_atom]).reshape(3, 1)).T
         
            rot_mat = torch_rotate_around_axis(rot_angle_list[pot_i], axis="z")
            rotated_z_axis_adjusted_tmp_geom_num_list = torch.matmul(rot_mat, z_axis_adjusted_tmp_geom_num_list.T).T
            
 
            LJ_vec_for_save = tmp_geom_num_list_for_save[LJ_atom] - tmp_geom_num_list_for_save[root_atom]
            LJ_vec_for_save = LJ_vec_for_save / torch.norm(LJ_vec_for_save)
            LJ_center_for_save = tmp_geom_num_list_for_save[root_atom] + LJ_vec_for_save * asym_elip_dist * self.bohr2angstroms
            z_axis_adjust_rot_mat_for_save = torch_align_vector_with_z(LJ_vec_for_save)
            z_axis_adjusted_tmp_geom_num_list_for_save = torch.matmul(z_axis_adjust_rot_mat_for_save, (tmp_geom_num_list_for_save - tmp_geom_num_list_for_save[root_atom]).T).T
            z_axis_adjusted_LJ_center_for_save = torch.matmul(z_axis_adjust_rot_mat_for_save, (LJ_center_for_save - tmp_geom_num_list_for_save[root_atom]).reshape(3, 1)).T
            rotated_z_axis_adjusted_tmp_geom_num_list_for_save = torch.matmul(rot_mat, z_axis_adjusted_tmp_geom_num_list_for_save.T).T  
            

            tgt_atom_pos = rotated_z_axis_adjusted_tmp_geom_num_list[tgt_atom_list] - z_axis_adjusted_LJ_center[0]
            tgt_atom_eps = torch.tensor([GNB_VDW_well_depth_lib(self.element_list[tgt_atom]) for tgt_atom in tgt_atom_list], dtype=torch.float64)
            tgt_atom_sig = torch.tensor([GNB_VDW_radii_lib(self.element_list[tgt_atom]) / 2.0 for tgt_atom in tgt_atom_list], dtype=torch.float64)

           
            x, y, z = tgt_atom_pos[:, 0], tgt_atom_pos[:, 1], tgt_atom_pos[:, 2]

          
            x_sig = torch.where(x > 0, torch.sqrt(2 ** (14 / 6) * asym_elip_sig_xp * tgt_atom_sig), torch.sqrt(2 ** (14 / 6) * asym_elip_sig_xm * tgt_atom_sig))
            y_sig = torch.where(y > 0, torch.sqrt(2 ** (14 / 6) * asym_elip_sig_yp * tgt_atom_sig), torch.sqrt(2 ** (14 / 6) * asym_elip_sig_ym * tgt_atom_sig))
            z_sig = torch.where(z > 0, torch.sqrt(2 ** (14 / 6) * asym_elip_sig_zp * tgt_atom_sig), torch.sqrt(2 ** (14 / 6) * asym_elip_sig_zm * tgt_atom_sig))

            x_eps = torch.sqrt(asym_elip_eps * tgt_atom_eps)
            y_eps = torch.sqrt(asym_elip_eps * tgt_atom_eps)
            z_eps = torch.sqrt(asym_elip_eps * tgt_atom_eps)

            
            r_ell = torch.sqrt((x / x_sig) ** 2 + (y / y_sig) ** 2 + (z / z_sig) ** 2)
            r_ell_norm = torch.linalg.norm(r_ell, dim=-1)

            lj_eps = 1 / torch.sqrt((x / r_ell_norm / x_eps) ** 2 + (y / r_ell_norm / y_eps) ** 2 + (z / r_ell_norm / z_eps) ** 2)
            eps = torch.sqrt(lj_eps * tgt_atom_eps)

            
            r_ell_inv = 1 / r_ell
            tmp_ene = eps * ((r_ell_inv ** self.lj_repulsive_order) - 2 * (r_ell_inv ** self.lj_attractive_order))

            
            if self.energy_save_flag:
                for i, tgt_atom in enumerate(tgt_atom_list):
                    self.energy_analysis_dict["ell_" + str(pot_i) + "_atom" + str(tgt_atom + 1)] = tmp_ene[i].item()

            energy = energy + torch.sum(tmp_ene)
            
            
            #--------------
            tmp_geom_num_list_for_save = torch.cat([rotated_z_axis_adjusted_tmp_geom_num_list_for_save, z_axis_adjusted_LJ_center_for_save], dim=0)
         
            ellipsoid_list = torch.tensor([[asym_elip_sig_xp, 0.0, 0.0+asym_elip_dist],
                                            [-1*asym_elip_sig_xm, 0.0, 0.0+asym_elip_dist],
                                            [0.0, asym_elip_sig_yp, 0.0+asym_elip_dist],
                                            [0.0, -1*asym_elip_sig_ym, 0.0+asym_elip_dist],
                                            [0.0, 0.0,  asym_elip_sig_zp+asym_elip_dist],
                                            [0.0, 0.0,  -1*asym_elip_sig_zm+asym_elip_dist]], dtype=torch.float64) * self.bohr2angstroms    
            tmp_geom_num_list_for_save = torch.cat((tmp_geom_num_list_for_save, ellipsoid_list), dim=0)
            tmp_element_list_for_save = tmp_element_list_for_save + ["x", "X", "X", "X", "X", "X", "X"]
            
            #--------------
          
        # interaction between asymmetric ellipsoid and asymmetric ellipsoid
        if len(bias_pot_params) > 1:
            for pot_i in range(len(bias_pot_params)):
                root_atom_i = self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_i][0] - 1
                LJ_atom_i = self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_i][1] - 1
                asym_elip_eps_i = bias_pot_params[pot_i][0] / self.hartree2kjmol
                asym_elip_sig_xp_i = bias_pot_params[pot_i][1] / self.bohr2angstroms
                asym_elip_sig_xm_i = bias_pot_params[pot_i][2] / self.bohr2angstroms
                asym_elip_sig_yp_i = bias_pot_params[pot_i][3] / self.bohr2angstroms
                asym_elip_sig_ym_i = bias_pot_params[pot_i][4] / self.bohr2angstroms
                asym_elip_sig_zp_i = bias_pot_params[pot_i][5] / self.bohr2angstroms
                asym_elip_sig_zm_i = bias_pot_params[pot_i][6] / self.bohr2angstroms
                asym_elip_dist_i = bias_pot_params[pot_i][7] / self.bohr2angstroms
                # rotate the asymmetric ellipsoid
                LJ_vec_i = geom_num_list[LJ_atom_i] - geom_num_list[root_atom_i]
                LJ_vec_i = LJ_vec_i / torch.norm(LJ_vec_i)
                LJ_center_i = geom_num_list[root_atom_i] + LJ_vec_i * asym_elip_dist_i

                for pot_j in range(pot_i+1, len(bias_pot_params)):
                    if pot_i > pot_j:
                        continue
                    
                    root_atom_j = self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_j][0] - 1
                    LJ_atom_j = self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"][pot_j][1] - 1
                    asym_elip_eps_j = bias_pot_params[pot_j][0] / self.hartree2kjmol
                    asym_elip_sig_xp_j = bias_pot_params[pot_j][1] / self.bohr2angstroms
                    asym_elip_sig_xm_j = bias_pot_params[pot_j][2] / self.bohr2angstroms
                    asym_elip_sig_yp_j = bias_pot_params[pot_j][3] / self.bohr2angstroms
                    asym_elip_sig_ym_j = bias_pot_params[pot_j][4] / self.bohr2angstroms
                    asym_elip_sig_zp_j = bias_pot_params[pot_j][5] / self.bohr2angstroms
                    asym_elip_sig_zm_j = bias_pot_params[pot_j][6] / self.bohr2angstroms
                    asym_elip_dist_j = bias_pot_params[pot_j][7] / self.bohr2angstroms
                    # rotate the asymmetric ellipsoid
                    LJ_vec_j = geom_num_list[LJ_atom_j] - geom_num_list[root_atom_j]
                    LJ_vec_j = LJ_vec_j / torch.norm(LJ_vec_j)
                    LJ_center_j = geom_num_list[root_atom_j] + LJ_vec_j * asym_elip_dist_j
                    
                    
                    #-----------------------------------
                    
                    z_axis_adjust_rot_mat_i = torch_align_vector_with_z(LJ_vec_i)
                    z_axis_adjusted_LJ_center_j_i = torch.matmul(z_axis_adjust_rot_mat_i, (LJ_center_j - geom_num_list[root_atom_i]).reshape(3, 1)).T
                 
                    z_axis_adjusted_LJ_center_i = torch.matmul(z_axis_adjust_rot_mat_i, (LJ_center_i - geom_num_list[root_atom_i]).reshape(3, 1)).T
                   
                    rot_mat_i = torch_rotate_around_axis(rot_angle_list[pot_i])
                    rotated_z_axis_adjusted_LJ_center_j_i = torch.matmul(rot_mat_i, z_axis_adjusted_LJ_center_j_i.T).T
                  
                        
                    
                    z_axis_adjust_rot_mat_j = torch_align_vector_with_z(LJ_vec_j)
                    z_axis_adjusted_LJ_center_i_j = torch.matmul(z_axis_adjust_rot_mat_j, (LJ_center_i - geom_num_list[root_atom_j]).reshape(3, 1)).T
                   
                    z_axis_adjusted_LJ_center_j = torch.matmul(z_axis_adjust_rot_mat_j, (LJ_center_j - geom_num_list[root_atom_j]).reshape(3, 1)).T
                    
                    rot_mat_j = torch_rotate_around_axis(rot_angle_list[pot_j])
                    rotated_z_axis_adjusted_LJ_center_i_j = torch.matmul(rot_mat_j, z_axis_adjusted_LJ_center_i_j.T).T
                   
                    pos_j = rotated_z_axis_adjusted_LJ_center_j_i[0] - z_axis_adjusted_LJ_center_i[0]
                 
                    x_j = pos_j[0]
                    y_j = pos_j[1]
                    z_j = pos_j[2]
                    
                    if x_j > 0:
                        x_i_sig = 2 ** (7 / 6) * asym_elip_sig_xp_i
                        x_i_eps = asym_elip_eps_i
                    else:
                        x_i_sig = 2 ** (7 / 6) * asym_elip_sig_xm_i
                        x_i_eps = asym_elip_eps_i
                    
                    if y_j > 0:
                        y_i_sig = 2 ** (7 / 6) * asym_elip_sig_yp_i
                        y_i_eps = asym_elip_eps_i
                    else:
                        y_i_sig = 2 ** (7 / 6) * asym_elip_sig_ym_i
                        y_i_eps = asym_elip_eps_i
                    
                    if z_j > 0:
                        z_i_sig = 2 ** (7 / 6) * asym_elip_sig_zp_i
                        z_i_eps = asym_elip_eps_i
                    else:
                        z_i_sig = 2 ** (7 / 6) * asym_elip_sig_zm_i
                        z_i_eps = asym_elip_eps_i
                
                    r_ell_i = torch.sqrt((x_j / x_i_sig) ** 2 + (y_j / y_i_sig) ** 2 + (z_j / z_i_sig) ** 2)
                    r_ell_i_norm = torch.linalg.norm(r_ell_i)
                    lj_eps_i = 1 / torch.sqrt((x_j / r_ell_i_norm / x_i_eps) ** 2 + (y_j / r_ell_i_norm / y_i_eps) ** 2 + (z_j / r_ell_i_norm / z_i_eps) ** 2 )

                    pos_i = rotated_z_axis_adjusted_LJ_center_i_j[0] - z_axis_adjusted_LJ_center_j[0]
                  
                    x_i = pos_i[0]
                    y_i = pos_i[1]
                    z_i = pos_i[2]
                    
                    if x_i > 0:
                        x_j_sig = 2 ** (7 / 6) * asym_elip_sig_xp_j
                        x_j_eps = asym_elip_eps_j
                    else:
                        x_j_sig = 2 ** (7 / 6) * asym_elip_sig_xm_j
                        x_j_eps = asym_elip_eps_j
                    
                    if y_i > 0:
                        y_j_sig = 2 ** (7 / 6) * asym_elip_sig_yp_j
                        y_j_eps = asym_elip_eps_j
                    else:
                        y_j_sig = 2 ** (7 / 6) * asym_elip_sig_ym_j
                        y_j_eps = asym_elip_eps_j
                    
                    if z_i > 0:
                        z_j_sig = 2 ** (7 / 6) * asym_elip_sig_zp_j
                        z_j_eps = asym_elip_eps_j
                    else:
                        z_j_sig = 2 ** (7 / 6) * asym_elip_sig_zm_j
                        z_j_eps = asym_elip_eps_j
                        
                    r_ell_j = torch.sqrt((x_i / x_j_sig) ** 2 + (y_i / y_j_sig) ** 2 + (z_i / z_j_sig) ** 2)
                    r_ell_j_norm = torch.linalg.norm(r_ell_j)
                    lj_eps_j = 1 / torch.sqrt((x_i / r_ell_j_norm / x_j_eps) ** 2 + (y_i / r_ell_j_norm / y_j_eps) ** 2 + (z_i / r_ell_j_norm / z_j_eps) ** 2 )

                    eps = torch.sqrt(lj_eps_i * lj_eps_j)
                    r_ell = torch.sqrt(r_ell_i * r_ell_j)
 
                    energy = energy + eps * (((1/r_ell) ** self.lj_repulsive_order) -2 * ((1/r_ell) ** self.lj_attractive_order))
                    
        self.tmp_geom_num_list_for_save = tmp_geom_num_list_for_save    
        self.tmp_element_list_for_save = tmp_element_list_for_save
        return energy
    
    def rand_search(self, geom_num_list, bias_pot_params):
        max_energy = 1e+10
        print("rand_search")
        
        for i in range(self.rand_search_iteration):
            tmp_rot_angle_list = [random.uniform(0, 2*math.pi) for j in range(len(self.config["asymmetric_ellipsoidal_repulsive_potential_eps"]))]
            tmp_rot_angle_list = torch.tensor([tmp_rot_angle_list], dtype=torch.float64, requires_grad=True)
            energy = self.calc_potential(geom_num_list, tmp_rot_angle_list, bias_pot_params)
            if energy < max_energy:
                print("energy: ", energy.item())
                max_energy = energy
                self.rot_angle_list = tmp_rot_angle_list
        print("rand_search done")
        print("max_energy: ", max_energy.item())
        return
    
    
    def microiteration(self, geom_num_list, bias_pot_params):
        if self.init:
            self.rand_search(geom_num_list, bias_pot_params)
            self.init = False
        
        prev_rot_grad = torch.zeros_like(self.rot_angle_list)
        Opt = FIRE()
        Opt.display_flag = False
        
        for j in range(self.micro_iteration):
            rot_grad = torch.func.jacrev(self.calc_potential, argnums=1)(geom_num_list, self.rot_angle_list, bias_pot_params)
            if torch.linalg.norm(rot_grad) < self.threshold:
                print("Converged!")
                print("M. itr: ", j)
                print("energy: ", self.calc_potential(geom_num_list, self.rot_angle_list, bias_pot_params).item())
                break
            
            tmp_rot_angle_list = copy.copy(self.rot_angle_list.clone().detach().numpy())
            tmp_rot_grad = copy.copy(rot_grad.clone().detach().numpy())
            tmp_prev_rot_grad = copy.copy(prev_rot_grad.clone().detach().numpy())
            move_vector = Opt.run(tmp_rot_angle_list[0], tmp_rot_grad[0], tmp_prev_rot_grad[0])
            move_vector = torch.tensor(move_vector, dtype=torch.float64)
            self.rot_angle_list = self.rot_angle_list - 1.0 * move_vector
            # update rot_angle_list
            if j % 100 == 0:
                print("M. itr: ", j)
                print("energy: ", self.calc_potential(geom_num_list, self.rot_angle_list, bias_pot_params).item())
                print("rot_angle_list: ", self.rot_angle_list.detach().numpy())
                print("rot_grad: ", rot_grad.detach().numpy())
            

            prev_rot_grad = rot_grad
        else:
            print("Not converged...")
            raise
        
        self.energy_analysis_dict = {}
        self.energy_save_flag = True
        energy = self.calc_potential(geom_num_list, self.rot_angle_list, bias_pot_params)
        self.energy_save_flag = False
        self.save_part_ene()
        return energy
    
    
    def calc_energy(self, geom_num_list, bias_pot_params):
        """
        # required variables: self.config["asymmetric_ellipsoidal_repulsive_potential_eps"], 
                             self.config["asymmetric_ellipsoidal_repulsive_potential_sig"], 
                             self.config["asymmetric_ellipsoidal_repulsive_potential_dist"],
                             self.config["asymmetric_ellipsoidal_repulsive_potential_atoms"],
                             self.config["asymmetric_ellipsoidal_repulsive_potential_offtgt"],
        bias_pot_params[n][0] : asymmetric_ellipsoidal_repulsive_potential_eps
        bias_pot_params[n][1:7] : asymmetric_ellipsoidal_repulsive_potential_sig
        bias_pot_params[n][7] : asymmetric_ellipsoidal_repulsive_potential_dist
        """
        energy = self.microiteration(geom_num_list, bias_pot_params)
        return energy
    
    def calc_pot_for_eff_hess(self, coord_and_ell_angle, bias_pot_params):
        geom_num_list = coord_and_ell_angle[:len(self.element_list)*3].reshape(-1, 3)
        rot_angle_list = coord_and_ell_angle[len(self.element_list)*3:].reshape(1, self.nangle)
        energy = self.calc_potential(geom_num_list, rot_angle_list, bias_pot_params)
        return energy


    def calc_eff_hessian(self, geom_num_list, bias_pot_params):# effective hessian
        transformed_geom_num_list = geom_num_list.reshape(-1, 1)
        transformed_angle_list = self.rot_angle_list.reshape(-1, 1)
        coord_and_ell_angle = torch.cat((transformed_geom_num_list, transformed_angle_list), dim=0)
        combined_hess = torch.func.hessian(self.calc_pot_for_eff_hess, argnums=0)(coord_and_ell_angle, bias_pot_params).reshape(len(self.element_list)*3+self.nangle, len(self.element_list)*3+self.nangle)
        coupling_hess_1 = combined_hess[:len(self.element_list)*3, len(self.element_list)*3:]
        coupling_hess_2 = combined_hess[len(self.element_list)*3:, :len(self.element_list)*3]
        angle_hess = combined_hess[len(self.element_list)*3:, len(self.element_list)*3:]
        eff_hess = -1 * torch.matmul(torch.matmul(coupling_hess_1, torch.linalg.inv(angle_hess)), coupling_hess_2)
        return eff_hess
    

class AsymmetricEllipsoidalLJPotentialv2:
    def __init__(self, **kwarg):
        #ref.: https://doi.org/10.26434/chemrxiv-2024-6www6
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        self.element_list = self.config["element_list"]
        self.file_directory = self.config["file_directory"]
        self.rot_angle_list = []
        for i in range(len(self.config["asymmetric_ellipsoidal_repulsive_potential_v2_eps"])):
            self.rot_angle_list.append(random.uniform(0, 2*math.pi))
        self.nangle = len(self.config["asymmetric_ellipsoidal_repulsive_potential_v2_eps"])
        self.rot_angle_list = torch.tensor([self.rot_angle_list], dtype=torch.float64, requires_grad=True)
        
        self.lj_repulsive_order = 12.0
        self.lj_attractive_order = 6.0
        
        self.micro_iteration = 15000 * len(self.config["asymmetric_ellipsoidal_repulsive_potential_v2_eps"])
        self.rand_search_iteration = 1000 * len(self.config["asymmetric_ellipsoidal_repulsive_potential_v2_eps"])
        self.threshold = 1e-7
        self.init = True
        self.save_flag = False
        return
    
    def save_state(self):      
        with open(self.file_directory + "/asym_ellipsoid_v2.xyz", "a") as f:
            f.write(str(len(self.tmp_geom_num_list_for_save)) + "\n")
            f.write("AsymmetricEllipsoid\n")
            for i in range(len(self.tmp_geom_num_list_for_save)):
                f.write(self.tmp_element_list_for_save[i] + " " + str(self.tmp_geom_num_list_for_save[i][0].item()) + " " + str(self.tmp_geom_num_list_for_save[i][1].item()) + " " + str(self.tmp_geom_num_list_for_save[i][2].item()) + "\n")
        
        return
    
    def calc_potential(self, geom_num_list, rot_angle_list, bias_pot_params):
        energy = 0.0
        """
        self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"],
        self.config["asymmetric_ellipsoidal_repulsive_potential_v2_offtgt"],
    
        """
        rot_angle_list = rot_angle_list[0]
        tmp_geom_num_list_for_save = geom_num_list * self.bohr2angstroms # save the geometry with asymmetric ellipsoid
        tmp_element_list_for_save = self.element_list
        
        
        # interaction between substrate and asymmetric ellipsoid
        for pot_i in range(len(bias_pot_params)):
            tgt_atom_list = [i for i in range(len(geom_num_list)) if not i+1 in self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_i] + self.config["asymmetric_ellipsoidal_repulsive_potential_v2_offtgt"][pot_i]]
            root_atom = self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_i][0] - 1
            LJ_atom = self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_i][1] - 1
           
            asym_elip_eps = bias_pot_params[pot_i][0] / self.hartree2kjmol
            asym_elip_sig_xp = bias_pot_params[pot_i][1] / self.bohr2angstroms
            asym_elip_sig_xm = bias_pot_params[pot_i][2] / self.bohr2angstroms
            asym_elip_sig_yp = bias_pot_params[pot_i][3] / self.bohr2angstroms
            asym_elip_sig_ym = bias_pot_params[pot_i][4] / self.bohr2angstroms
            asym_elip_sig_zp = bias_pot_params[pot_i][5] / self.bohr2angstroms
            asym_elip_sig_zm = bias_pot_params[pot_i][6] / self.bohr2angstroms
            asym_elip_dist = bias_pot_params[pot_i][7] / self.bohr2angstroms
            
            # rotate the asymmetric ellipsoid
            LJ_vec = geom_num_list[LJ_atom] - geom_num_list[root_atom]
            LJ_vec = LJ_vec / torch.norm(LJ_vec)
            LJ_center = geom_num_list[root_atom] + LJ_vec * asym_elip_dist
            z_axis_adjust_rot_mat = torch_align_vector_with_z(LJ_vec)
            z_axis_adjusted_tmp_geom_num_list = torch.matmul(z_axis_adjust_rot_mat, (geom_num_list - geom_num_list[root_atom]).T).T
            z_axis_adjusted_LJ_center = torch.matmul(z_axis_adjust_rot_mat, (LJ_center - geom_num_list[root_atom]).reshape(3, 1)).T
         
            rot_mat = torch_rotate_around_axis(rot_angle_list[pot_i], axis="z")
            rotated_z_axis_adjusted_tmp_geom_num_list = torch.matmul(rot_mat, z_axis_adjusted_tmp_geom_num_list.T).T

            if self.save_flag:
                LJ_vec_for_save = tmp_geom_num_list_for_save[LJ_atom] - tmp_geom_num_list_for_save[root_atom]
                LJ_vec_for_save = LJ_vec_for_save / torch.norm(LJ_vec_for_save)
                LJ_center_for_save = tmp_geom_num_list_for_save[root_atom] + LJ_vec_for_save * asym_elip_dist * self.bohr2angstroms
                z_axis_adjust_rot_mat_for_save = torch_align_vector_with_z(LJ_vec_for_save)
                z_axis_adjusted_tmp_geom_num_list_for_save = torch.matmul(z_axis_adjust_rot_mat_for_save, (tmp_geom_num_list_for_save - tmp_geom_num_list_for_save[root_atom]).T).T
                z_axis_adjusted_LJ_center_for_save = torch.matmul(z_axis_adjust_rot_mat_for_save, (LJ_center_for_save - tmp_geom_num_list_for_save[root_atom]).reshape(3, 1)).T
                rotated_z_axis_adjusted_tmp_geom_num_list_for_save = torch.matmul(rot_mat, z_axis_adjusted_tmp_geom_num_list_for_save.T).T  
                
            tgt_atom_pos = rotated_z_axis_adjusted_tmp_geom_num_list[tgt_atom_list] - z_axis_adjusted_LJ_center[0]
            tgt_atom_eps = torch.tensor([UFF_VDW_well_depth_lib(self.element_list[tgt_atom]) for tgt_atom in tgt_atom_list], dtype=torch.float64)
            tgt_atom_sig = torch.tensor([UFF_VDW_distance_lib(self.element_list[tgt_atom]) / 2.0 for tgt_atom in tgt_atom_list], dtype=torch.float64)

            x, y, z = tgt_atom_pos[:, 0], tgt_atom_pos[:, 1], tgt_atom_pos[:, 2]

            x_sig = torch.where(x > 0, (asym_elip_sig_xp + tgt_atom_sig) ** (7 / 6), (asym_elip_sig_xm + tgt_atom_sig) ** (7 / 6))
            y_sig = torch.where(y > 0, (asym_elip_sig_yp + tgt_atom_sig) ** (7 / 6), (asym_elip_sig_ym + tgt_atom_sig) ** (7 / 6))
            z_sig = torch.where(z > 0, (asym_elip_sig_zp + tgt_atom_sig) ** (7 / 6), (asym_elip_sig_zm + tgt_atom_sig) ** (7 / 6))

            x_eps = torch.sqrt(asym_elip_eps * tgt_atom_eps)
            y_eps = torch.sqrt(asym_elip_eps * tgt_atom_eps)
            z_eps = torch.sqrt(asym_elip_eps * tgt_atom_eps)

            r_ell = torch.sqrt((x / x_sig) ** 2 + (y / y_sig) ** 2 + (z / z_sig) ** 2)
            r_ell_norm = torch.linalg.norm(r_ell, dim=-1)

            lj_eps = 1 / torch.sqrt((x / r_ell_norm / x_eps) ** 2 + (y / r_ell_norm / y_eps) ** 2 + (z / r_ell_norm / z_eps) ** 2)
            eps = torch.sqrt(lj_eps * tgt_atom_eps)
            
            r_ell_inv = 1 / r_ell
            tmp_ene = eps * ((r_ell_inv ** self.lj_repulsive_order) - 2 * (r_ell_inv ** self.lj_attractive_order))
          
            energy = energy + torch.sum(tmp_ene)
            
            #--------------
            if self.save_flag:
                tmp_geom_num_list_for_save = torch.cat([rotated_z_axis_adjusted_tmp_geom_num_list_for_save, z_axis_adjusted_LJ_center_for_save], dim=0)
         
                ellipsoid_list = torch.tensor([[asym_elip_sig_xp, 0.0, 0.0+asym_elip_dist],
                                            [-1*asym_elip_sig_xm, 0.0, 0.0+asym_elip_dist],
                                            [0.0, asym_elip_sig_yp, 0.0+asym_elip_dist],
                                            [0.0, -1*asym_elip_sig_ym, 0.0+asym_elip_dist],
                                            [0.0, 0.0,  asym_elip_sig_zp+asym_elip_dist],
                                            [0.0, 0.0,  -1*asym_elip_sig_zm+asym_elip_dist]], dtype=torch.float64) * self.bohr2angstroms    
                tmp_geom_num_list_for_save = torch.cat((tmp_geom_num_list_for_save, ellipsoid_list), dim=0)
                tmp_element_list_for_save = tmp_element_list_for_save + ["x", "X", "X", "X", "X", "X", "X"]
            
            #--------------

        # interaction between asymmetric ellipsoid and  asymmetric ellipsoid
        if len(bias_pot_params) > 1:
            for pot_i in range(len(bias_pot_params)):
                asym_elip_eps_i = bias_pot_params[pot_i][0] / self.hartree2kjmol
                asym_elip_sig_xp_i = bias_pot_params[pot_i][1] / self.bohr2angstroms
                asym_elip_sig_xm_i = bias_pot_params[pot_i][2] / self.bohr2angstroms
                asym_elip_sig_yp_i = bias_pot_params[pot_i][3] / self.bohr2angstroms
                asym_elip_sig_ym_i = bias_pot_params[pot_i][4] / self.bohr2angstroms
                asym_elip_sig_zp_i = bias_pot_params[pot_i][5] / self.bohr2angstroms
                asym_elip_sig_zm_i = bias_pot_params[pot_i][6] / self.bohr2angstroms
                asym_elip_dist_i = bias_pot_params[pot_i][7] / self.bohr2angstroms

                root_atom_i = self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_i][0] - 1
                LJ_atom_i = self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_i][1] - 1

                # rotate the asymmetric ellipsoid
                LJ_vec_i = geom_num_list[LJ_atom_i] - geom_num_list[root_atom_i]
                LJ_vec_i = LJ_vec_i / torch.norm(LJ_vec_i)
                LJ_center_i = geom_num_list[root_atom_i] + LJ_vec_i * asym_elip_dist_i

                for pot_j in range(pot_i+1, len(bias_pot_params)):
                    if pot_i > pot_j:
                        continue
       

                    
                    root_atom_j = self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_j][0] - 1
                    LJ_atom_j = self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"][pot_j][1] - 1
                    asym_elip_eps_j = bias_pot_params[pot_j][0] / self.hartree2kjmol
                    asym_elip_sig_xp_j = bias_pot_params[pot_j][1] / self.bohr2angstroms
                    asym_elip_sig_xm_j = bias_pot_params[pot_j][2] / self.bohr2angstroms
                    asym_elip_sig_yp_j = bias_pot_params[pot_j][3] / self.bohr2angstroms
                    asym_elip_sig_ym_j = bias_pot_params[pot_j][4] / self.bohr2angstroms
                    asym_elip_sig_zp_j = bias_pot_params[pot_j][5] / self.bohr2angstroms
                    asym_elip_sig_zm_j = bias_pot_params[pot_j][6] / self.bohr2angstroms
                    asym_elip_dist_j = bias_pot_params[pot_j][7] / self.bohr2angstroms
                    # rotate the asymmetric ellipsoid
                    LJ_vec_j = geom_num_list[LJ_atom_j] - geom_num_list[root_atom_j]
                    LJ_vec_j = LJ_vec_j / torch.norm(LJ_vec_j)
                    LJ_center_j = geom_num_list[root_atom_j] + LJ_vec_j * asym_elip_dist_j
                    
                    
                    #-----------------------------------
                    
                    z_axis_adjust_rot_mat_i = torch_align_vector_with_z(LJ_vec_i)
                    z_axis_adjusted_LJ_center_j_i = torch.matmul(z_axis_adjust_rot_mat_i, (LJ_center_j - geom_num_list[root_atom_i]).reshape(3, 1)).T
                 
                    z_axis_adjusted_LJ_center_i = torch.matmul(z_axis_adjust_rot_mat_i, (LJ_center_i - geom_num_list[root_atom_i]).reshape(3, 1)).T
                   
                    rot_mat_i = torch_rotate_around_axis(rot_angle_list[pot_i])
                    rotated_z_axis_adjusted_LJ_center_j_i = torch.matmul(rot_mat_i, z_axis_adjusted_LJ_center_j_i.T).T
                  
                        
                    
                    z_axis_adjust_rot_mat_j = torch_align_vector_with_z(LJ_vec_j)
                    z_axis_adjusted_LJ_center_i_j = torch.matmul(z_axis_adjust_rot_mat_j, (LJ_center_i - geom_num_list[root_atom_j]).reshape(3, 1)).T
                   
                    z_axis_adjusted_LJ_center_j = torch.matmul(z_axis_adjust_rot_mat_j, (LJ_center_j - geom_num_list[root_atom_j]).reshape(3, 1)).T
                    
                    rot_mat_j = torch_rotate_around_axis(rot_angle_list[pot_j])
                    rotated_z_axis_adjusted_LJ_center_i_j = torch.matmul(rot_mat_j, z_axis_adjusted_LJ_center_i_j.T).T
                   
                    pos_j = rotated_z_axis_adjusted_LJ_center_j_i[0] - z_axis_adjusted_LJ_center_i[0]
                 
                    x_j = pos_j[0]
                    y_j = pos_j[1]
                    z_j = pos_j[2]
                    
                    if x_j > 0:
                        x_i_sig = asym_elip_sig_xp_i * 2 ** (7 / 6)
                        x_i_eps = asym_elip_eps_i
                    else:
                        x_i_sig = asym_elip_sig_xm_i * 2 ** (7 / 6)
                        x_i_eps = asym_elip_eps_i
                    
                    if y_j > 0:
                        y_i_sig = asym_elip_sig_yp_i * 2 ** (7 / 6)
                        y_i_eps = asym_elip_eps_i
                    else:
                        y_i_sig = asym_elip_sig_ym_i * 2 ** (7 / 6)
                        y_i_eps = asym_elip_eps_i
                    
                    if z_j > 0:
                        z_i_sig = asym_elip_sig_zp_i * 2 ** (7 / 6)
                        z_i_eps = asym_elip_eps_i
                    else:
                        z_i_sig = asym_elip_sig_zm_i * 2 ** (7 / 6)
                        z_i_eps = asym_elip_eps_i
                
                    r_ell_i = torch.sqrt((x_j / x_i_sig) ** 2 + (y_j / y_i_sig) ** 2 + (z_j / z_i_sig) ** 2)
                    r_ell_i_norm = torch.linalg.norm(r_ell_i)
                    lj_eps_i = 1 / torch.sqrt((x_j / r_ell_i_norm / x_i_eps) ** 2 + (y_j / r_ell_i_norm / y_i_eps) ** 2 + (z_j / r_ell_i_norm / z_i_eps) ** 2 )
                    
                    pos_i = rotated_z_axis_adjusted_LJ_center_i_j[0] - z_axis_adjusted_LJ_center_j[0]
                  
                    x_i = pos_i[0]
                    y_i = pos_i[1]
                    z_i = pos_i[2]
                    
                    if x_i > 0:
                        x_j_sig = asym_elip_sig_xp_j * 2 ** (7 / 6)
                        x_j_eps = asym_elip_eps_j
                    else:
                        x_j_sig = asym_elip_sig_xm_j * 2 ** (7 / 6)
                        x_j_eps = asym_elip_eps_j
                    
                    if y_i > 0:
                        y_j_sig = asym_elip_sig_yp_j * 2 ** (7 / 6)
                        y_j_eps = asym_elip_eps_j
                    else:
                        y_j_sig = asym_elip_sig_ym_j * 2 ** (7 / 6)
                        y_j_eps = asym_elip_eps_j
                    
                    if z_i > 0:
                        z_j_sig = asym_elip_sig_zp_j * 2 ** (7 / 6)
                        z_j_eps = asym_elip_eps_j
                    else:
                        z_j_sig = asym_elip_sig_zm_j * 2 ** (7 / 6)
                        z_j_eps = asym_elip_eps_j
                        
                    r_ell_j = torch.sqrt((x_i / x_j_sig) ** 2 + (y_i / y_j_sig) ** 2 + (z_i / z_j_sig) ** 2)
                    r_ell_j_norm = torch.linalg.norm(r_ell_j)
                    lj_eps_j = 1 / torch.sqrt((x_i / r_ell_j_norm / x_j_eps) ** 2 + (y_i / r_ell_j_norm / y_j_eps) ** 2 + (z_i / r_ell_j_norm / z_j_eps) ** 2 )

                    eps = torch.sqrt(lj_eps_i * lj_eps_j)
                    r_ell = torch.sqrt(r_ell_i * r_ell_j)
 
                    energy = energy + eps * (((1/r_ell) ** self.lj_repulsive_order) -2 * ((1/r_ell) ** self.lj_attractive_order))
                    
        self.tmp_geom_num_list_for_save = tmp_geom_num_list_for_save    
        self.tmp_element_list_for_save = tmp_element_list_for_save
        return energy
    
    def rand_search(self, geom_num_list, bias_pot_params):
        max_energy = 1e+10
        print("rand_search")
        for i in range(self.rand_search_iteration):
            tmp_rot_angle_list = [random.uniform(0, 2*math.pi) for j in range(len(self.config["asymmetric_ellipsoidal_repulsive_potential_v2_eps"]))]
            tmp_rot_angle_list = torch.tensor([tmp_rot_angle_list], dtype=torch.float64, requires_grad=True)
            energy = self.calc_potential(geom_num_list, tmp_rot_angle_list, bias_pot_params)
            if energy < max_energy:
                print("Energy: ", energy)
                max_energy = energy
                self.rot_angle_list = tmp_rot_angle_list
        print("rand_search done")
        print("max_energy: ", max_energy.item())
        return
    
    
    def microiteration(self, geom_num_list, bias_pot_params):
        if self.init:
            self.rand_search(geom_num_list, bias_pot_params)
            self.init = False
        
        prev_rot_grad = torch.zeros_like(self.rot_angle_list)
        Opt = FIRE()
        Opt.display_flag = False
        
        for j in range(self.micro_iteration):
            rot_grad = torch.func.jacrev(self.calc_potential, argnums=1)(geom_num_list, self.rot_angle_list, bias_pot_params)
            if torch.linalg.norm(rot_grad) < self.threshold:
                print("Converged!")
                print("M. itr: ", j)
                self.save_flag = True
                energy = self.calc_potential(geom_num_list, self.rot_angle_list, bias_pot_params)
                print("energy: ", energy.item())
                break
            
            tmp_rot_angle_list = copy.copy(self.rot_angle_list.clone().detach().numpy())
            tmp_rot_grad = copy.copy(rot_grad.clone().detach().numpy())
            tmp_prev_rot_grad = copy.copy(prev_rot_grad.clone().detach().numpy())
            move_vector = Opt.run(tmp_rot_angle_list[0], tmp_rot_grad[0], tmp_prev_rot_grad[0])
            move_vector = torch.tensor(move_vector, dtype=torch.float64)
            self.rot_angle_list = self.rot_angle_list - 1.0 * move_vector
            # update rot_angle_list
            if j % 100 == 0:
                print("M. itr: ", j)
                print("energy: ", self.calc_potential(geom_num_list, self.rot_angle_list, bias_pot_params).item())
                print("rot_angle_list: ", self.rot_angle_list.detach().numpy())
                print("rot_grad: ", rot_grad.detach().numpy())
            
            prev_rot_grad = rot_grad
        else:
            print("Not converged...")
            energy = None
            raise
        
        
        return energy
    
    
    def calc_energy(self, geom_num_list, bias_pot_params):
        """
        # required variables: self.config["asymmetric_ellipsoidal_repulsive_potential_v2_eps"], 
                             self.config["asymmetric_ellipsoidal_repulsive_potential_v2_sig"], 
                             self.config["asymmetric_ellipsoidal_repulsive_potential_v2_dist"],
                             self.config["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"],
                             self.config["asymmetric_ellipsoidal_repulsive_potential_v2_offtgt"],
        bias_pot_params[n][0] : asymmetric_ellipsoidal_repulsive_potential_v2_eps
        bias_pot_params[n][1:7] : asymmetric_ellipsoidal_repulsive_potential_v2_sig
        bias_pot_params[n][7] : asymmetric_ellipsoidal_repulsive_potential_v2_dist
        """
        energy = self.microiteration(geom_num_list, bias_pot_params)
        return energy

    def calc_pot_for_eff_hess(self, coord_and_ell_angle, bias_pot_params):
        geom_num_list = coord_and_ell_angle[:len(self.element_list)*3].reshape(-1, 3)
        rot_angle_list = coord_and_ell_angle[len(self.element_list)*3:].reshape(1, self.nangle)
        energy = self.calc_potential(geom_num_list, rot_angle_list, bias_pot_params)
        return energy


    def calc_eff_hessian(self, geom_num_list, bias_pot_params):# effective hessian
        transformed_geom_num_list = geom_num_list.reshape(-1, 1)
        transformed_angle_list = self.rot_angle_list.reshape(-1, 1)
        coord_and_ell_angle = torch.cat((transformed_geom_num_list, transformed_angle_list), dim=0)
        combined_hess = torch.func.hessian(self.calc_pot_for_eff_hess, argnums=0)(coord_and_ell_angle, bias_pot_params).reshape(len(self.element_list)*3+self.nangle, len(self.element_list)*3+self.nangle)
        coupling_hess_1 = combined_hess[:len(self.element_list)*3, len(self.element_list)*3:]
        coupling_hess_2 = combined_hess[len(self.element_list)*3:, :len(self.element_list)*3]
        angle_hess = combined_hess[len(self.element_list)*3:, len(self.element_list)*3:]
        eff_hess = -1 * torch.matmul(torch.matmul(coupling_hess_1, torch.linalg.inv(angle_hess)), coupling_hess_2)
        return eff_hess
