
from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from calc_tools import Calculationtools

import itertools
import math
import numpy as np
import copy
import random
import torch

from electrostatic_potential import ElectroStaticPotential
from LJ_repulsive_potential import LJRepulsivePotential
from AFIR_potential import AFIRPotential
from keep_potential import StructKeepPotential
from anharmonic_keep_potential import StructAnharmonicKeepPotential
from keep_angle_potential import StructKeepAnglePotential
from keep_dihedral_angle_potential import StructKeepDihedralAnglePotential
from keep_outofplain_angle_potential import StructKeepOutofPlainAnglePotential
from void_point_potential import VoidPointPotential
from switching_potential import WellPotential
from gaussian_potential import GaussianPotential
from spacer_model_potential import SpacerModelPotential

class BiasPotentialCalculation:
    def __init__(self, Model_hess, FC_COUNT, FOLDER_DIRECTORY="./"):
        torch.set_printoptions(precision=12)
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.Model_hess = Model_hess
        self.FC_COUNT = FC_COUNT
        self.JOBID = random.randint(0, 1000000)
        self.partition = 300
        self.microiteration_num = 300
        self.rand_search_num = 800
        self.BPA_FOLDER_DIRECTORY = FOLDER_DIRECTORY
        self.metaD_history_list = None
        self.miter_delta = 1.0
    
    def ndarray2tensor(self, ndarray):
        tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64, requires_grad=True))

        return tensor

    def ndarray2nogradtensor(self, ndarray):
        tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64))

        return tensor

    def tensor2ndarray(self, tensor):
        ndarray = copy.copy(tensor.detach().numpy())
        return ndarray
    
    def main(self, e, g, geom_num_list, element_list,  force_data, pre_B_g="", iter="", initial_geom_num_list=""):
        numerical_derivative_delta = 1e-5 #unit:Bohr
        
        #g:hartree/Bohr
        #e:hartree
        #geom_num_list:Bohr

  
        #--------------------------------------------------
        B_e = e
        BPA_grad_list = g*0.0
        BPA_hessian = np.zeros((3*len(g), 3*len(g)))
        geom_num_list = self.ndarray2tensor(geom_num_list)
        #------------------------------------------------
         
        if iter == 0 and len(force_data["spacer_model_potential_well_depth"]) > 0:
            self.smp_particle_coord_list = []
            for i in range(len(force_data["spacer_model_potential_well_depth"])):
                center = torch.mean(geom_num_list[np.array(force_data["spacer_model_potential_target"][i])-1], dim=0)
                smp_particle_coord = torch.normal(mean=0, std=5, size=(force_data["spacer_model_potential_particle_number"][i], 3)) + center
                self.smp_particle_coord_list.append(smp_particle_coord)
                
        for i in range(len(force_data["spacer_model_potential_well_depth"])):
        
            if force_data["spacer_model_potential_well_depth"][i] != 0.0:
                SMP = SpacerModelPotential(spacer_model_potential_target=force_data["spacer_model_potential_target"][i],
                                           spacer_model_potential_distance=force_data["spacer_model_potential_distance"][i],
                                           spacer_model_potential_well_depth=force_data["spacer_model_potential_well_depth"][i],
                                           spacer_model_potential_cavity_scaling=force_data["spacer_model_potential_cavity_scaling"][i],
                                           element_list=element_list,
                                           directory=self.BPA_FOLDER_DIRECTORY)
                
                self.microiteration_num = 100 * force_data["spacer_model_potential_particle_number"][i]
                #---------------------- microiteration
                print("processing microiteration...") 
                for jter in range(self.microiteration_num):#TODO: improvement of convergence

                    
                    ene = SMP.calc_energy(geom_num_list, self.smp_particle_coord_list[i])
                    particle_grad_list = torch.func.jacfwd(SMP.calc_energy, argnums=1)(geom_num_list, self.smp_particle_coord_list[i]).reshape(1, len(self.smp_particle_coord_list[i])*3)
                    if jter % 50 ==  0:   
                        particle_model_hess = torch.func.hessian(SMP.calc_energy, argnums=1)(geom_num_list, self.smp_particle_coord_list[i]).reshape(len(self.smp_particle_coord_list[i])*3, len(self.smp_particle_coord_list[i])*3)
                    print("ITR. ", jter, " energy:", ene)
                    
                    if jter > 0:
                        diff_grad = particle_grad_list - prev_particle_grad_list
                        diff_coord = self.smp_particle_coord_list[i].reshape(1, len(self.smp_particle_coord_list[i])*3) - prev_particle_list
                        
                        #BFGS method
                        diff_hess = (torch.matmul(torch.t(diff_grad), diff_grad)) / (torch.matmul(diff_grad, torch.t(diff_grad))) -1 * (torch.matmul(particle_model_hess, torch.matmul(torch.t(diff_coord), torch.matmul(diff_coord, particle_model_hess)))/torch.matmul(diff_coord, torch.matmul(particle_model_hess, torch.t(diff_coord))))
                       
                        particle_model_hess = particle_model_hess + diff_hess
                        move_vector = torch.linalg.solve(particle_model_hess, particle_grad_list.reshape(len(self.smp_particle_coord_list[i])*3, 1))
                        prev_particle_list = copy.copy(self.smp_particle_coord_list[i].reshape(1, len(self.smp_particle_coord_list[i])*3))
                        new_particle_list = self.smp_particle_coord_list[i].reshape(1, len(self.smp_particle_coord_list[i])*3) -1*self.miter_delta*particle_grad_list
                        self.smp_particle_coord_list[i] = copy.copy(new_particle_list)
                    else:
                        prev_particle_list = copy.copy(self.smp_particle_coord_list[i].reshape(1, len(self.smp_particle_coord_list[i])*3))
                        new_particle_list = self.smp_particle_coord_list[i].reshape(1, len(self.smp_particle_coord_list[i])*3) -1*self.miter_delta*particle_grad_list
                        self.smp_particle_coord_list[i] = copy.copy(new_particle_list)
                    prev_ene = copy.copy(ene)   
                    
                    self.smp_particle_coord_list[i] = self.smp_particle_coord_list[i].reshape(force_data["spacer_model_potential_particle_number"][i], 3)
                    prev_particle_grad_list = copy.copy(particle_grad_list)
                    
                    #print(self.smp_particle_coord_list[i], prev_particle_list)
                    
                    if torch.sum(prev_particle_grad_list**2)/len(prev_particle_grad_list) < 1e-4:#converge criteria
                        print("gradient: ",torch.sqrt(torch.sum(prev_particle_grad_list**2)/len(prev_particle_grad_list)).item())
                        print("Microiteration: ", jter)
                        print("microiteration completed...")
                        microitr_converged_flag = True
                        break
                    if jter % 5 == 0:
                        print("gradient:", torch.sum(prev_particle_grad_list**2)/len(prev_particle_grad_list)) 
                else:
                    print("not converged... (microiteration)")    
                    microitr_converged_flag = False
                
                B_e += SMP.calc_energy(geom_num_list, self.smp_particle_coord_list[i])
                
                tensor_BPA_grad = torch.func.jacfwd(SMP.calc_energy, argnums=0)(geom_num_list, self.smp_particle_coord_list[i])
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(SMP.calc_energy, argnums=0)(geom_num_list, self.smp_particle_coord_list[i])
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                SMP.save_spacer_xyz_for_visualization(geom_num_list, self.smp_particle_coord_list[i])
            else:
                pass
        

            
        #-----------------------------------------------
        for i in range(len(force_data["repulsive_potential_v2_well_scale"])):
            if force_data["repulsive_potential_v2_well_scale"][i] != 0.0:
                if force_data["repulsive_potential_v2_unit"][i] == "scale":
                    LJRP = LJRepulsivePotential(repulsive_potential_v2_well_scale=force_data["repulsive_potential_v2_well_scale"][i], 
                                                repulsive_potential_v2_dist_scale=force_data["repulsive_potential_v2_dist_scale"][i], 
                                                repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                                repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                                repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"][i], 
                                                repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"][i], 
                                                repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"][i],
                                                repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"][i],
                                                repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_scale_v2(geom_num_list)
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_scale_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_scale_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

                elif force_data["repulsive_potential_v2_unit"][i] == "value":
                    LJRP = LJRepulsivePotential(repulsive_potential_v2_well_value=force_data["repulsive_potential_v2_well_scale"][i], 
                                                repulsive_potential_v2_dist_value=force_data["repulsive_potential_v2_dist_scale"][i], 
                                                repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                                repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                                repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"][i], 
                                                repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"][i], 
                                                repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"][i],
                                                repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"][i],
                                                repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_value_v2(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_value_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_value_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    print("error -rpv2")
                    raise "error -rpv2"
            else:
                pass

        #print("rp")
        #------------------
        for i in range(len(force_data["repulsive_potential_dist_scale"])):
            if force_data["repulsive_potential_well_scale"][i] != 0.0:
                if force_data["repulsive_potential_unit"][i] == "scale":
                    LJRP = LJRepulsivePotential(repulsive_potential_well_scale=force_data["repulsive_potential_well_scale"][i], 
                                                repulsive_potential_dist_scale=force_data["repulsive_potential_dist_scale"][i], 
                                                repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                                repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_scale(geom_num_list)
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_scale)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_scale)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    
                    
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

                elif force_data["repulsive_potential_unit"][i] == "value":
                    LJRP = LJRepulsivePotential(repulsive_potential_well_value=force_data["repulsive_potential_well_scale"][i], 
                                                repulsive_potential_dist_value=force_data["repulsive_potential_dist_scale"][i], 
                                                repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                                repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_value(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_value)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_value)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    print("error -rpv2")
                    raise "error -rpv2"
            else:
                pass
        
        #-----------------
        
        for i in range(len(force_data["repulsive_potential_gaussian_LJ_well_depth"])):
            
            if force_data["repulsive_potential_gaussian_LJ_well_depth"][i] != 0.0 or force_data["repulsive_potential_gaussian_gau_well_depth"][i] != 0.0:
                
                LJRP = LJRepulsivePotential(repulsive_potential_gaussian_LJ_well_depth=force_data["repulsive_potential_gaussian_LJ_well_depth"][i], 
                                                repulsive_potential_gaussian_LJ_dist=force_data["repulsive_potential_gaussian_LJ_dist"][i], 
                                                repulsive_potential_gaussian_gau_well_depth=force_data["repulsive_potential_gaussian_gau_well_depth"][i],
                                                repulsive_potential_gaussian_gau_dist=force_data["repulsive_potential_gaussian_gau_dist"][i],
                                                repulsive_potential_gaussian_gau_range=force_data["repulsive_potential_gaussian_gau_range"][i], 
                                                repulsive_potential_gaussian_fragm_1=force_data["repulsive_potential_gaussian_fragm_1"][i], 
                                                repulsive_potential_gaussian_fragm_2=force_data["repulsive_potential_gaussian_fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                B_e += LJRP.calc_energy_gau(geom_num_list)
                tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_gau)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_gau)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
        #------------------
        for i in range(len(force_data["cone_potential_well_value"])):
            if force_data["cone_potential_well_value"][i] != 0.0:
              
                LJRP = LJRepulsivePotential(cone_potential_well_value=force_data["cone_potential_well_value"][i], 
                                            cone_potential_dist_value=force_data["cone_potential_dist_value"][i], 
                                            cone_potential_cone_angle=force_data["cone_potential_cone_angle"][i],
                                            cone_potential_center=force_data["cone_potential_center"][i],
                                            cone_potential_three_atoms=force_data["cone_potential_three_atoms"][i],
                                            cone_potential_target=force_data["cone_potential_target"][i],
                                            element_list=element_list
                                            )

                B_e += LJRP.calc_cone_potential_energy(geom_num_list)
                tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_cone_potential_energy)(geom_num_list).view(len(geom_num_list), 3)
                
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(LJRP.calc_cone_potential_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))

                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
       
        #------------------
        
        for i in range(len(force_data["keep_pot_spring_const"])):
            if force_data["keep_pot_spring_const"][i] != 0.0:
                SKP = StructKeepPotential(keep_pot_spring_const=force_data["keep_pot_spring_const"][i], 
                                            keep_pot_distance=force_data["keep_pot_distance"][i], 
                                            keep_pot_atom_pairs=force_data["keep_pot_atom_pairs"][i])
                
                B_e += SKP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SKP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(SKP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            else:
                pass
            
        for i in range(len(force_data["aniso_keep_pot_v2_spring_const_mat"])):
            if np.any(force_data["aniso_keep_pot_v2_spring_const_mat"][i] != 0.0):
                SKP = StructKeepPotential(aniso_keep_pot_v2_spring_const_mat=force_data["aniso_keep_pot_v2_spring_const_mat"][i], 
                                            aniso_keep_pot_v2_dist=force_data["aniso_keep_pot_v2_dist"][i], 
                                              aniso_keep_pot_v2_fragm1=force_data["aniso_keep_pot_v2_fragm1"][i],
                                            aniso_keep_pot_v2_fragm2=force_data["aniso_keep_pot_v2_fragm2"][i]
                                           )
                
                B_e += SKP.calc_energy_aniso_v2(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SKP.calc_energy_aniso_v2)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                

                tensor_BPA_hessian = torch.func.hessian(SKP.calc_energy_aniso_v2)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            else:
                pass
               
        for i in range(len(force_data["keep_pot_v2_spring_const"])):
            if not 0.0 in force_data["keep_pot_v2_spring_const"][i]:
                if len(force_data["keep_pot_v2_spring_const"][i]) == 2 and iter != "":
                    spring_const_tmp = self.gradually_change_param(force_data["keep_pot_v2_spring_const"][i][0], force_data["keep_pot_v2_spring_const"][i][1], iter)
                    print(spring_const_tmp)
                else:
                    spring_const_tmp = force_data["keep_pot_v2_spring_const"][i][0]
                    
                if len(force_data["keep_pot_v2_distance"][i]) == 2 and iter != "":
                    dist_tmp = self.gradually_change_param(force_data["keep_pot_v2_distance"][i][0], force_data["keep_pot_v2_distance"][i][1], iter)
                    print(dist_tmp)
                else:
                    dist_tmp = force_data["keep_pot_v2_distance"][i][0]      
                SKP = StructKeepPotential(keep_pot_v2_spring_const=spring_const_tmp, 
                                            keep_pot_v2_distance=dist_tmp, 
                                            keep_pot_v2_fragm1=force_data["keep_pot_v2_fragm1"][i],
                                            keep_pot_v2_fragm2=force_data["keep_pot_v2_fragm2"][i]
                                            )
                
                B_e += SKP.calc_energy_v2(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SKP.calc_energy_v2)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(SKP.calc_energy_v2)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            else:
                pass
                 
            
        #print("akp")
        #------------------        
        for i in range(len(force_data["anharmonic_keep_pot_spring_const"])):
            if force_data["anharmonic_keep_pot_spring_const"][i] != 0.0:
                SAKP = StructAnharmonicKeepPotential(anharmonic_keep_pot_spring_const=force_data["anharmonic_keep_pot_spring_const"][i], 
                                            anharmonic_keep_pot_potential_well_depth=force_data["anharmonic_keep_pot_potential_well_depth"][i], 
                                            anharmonic_keep_pot_atom_pairs=force_data["anharmonic_keep_pot_atom_pairs"][i],
                                            anharmonic_keep_pot_distance=force_data["anharmonic_keep_pot_distance"][i])
                
                B_e += SAKP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SAKP.calc_energy)(geom_num_list)
            
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                
                tensor_BPA_hessian = torch.func.hessian(SAKP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
               
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
     
        #------------------
        

        #print("wp")
        for i in range(len(force_data["well_pot_wall_energy"])):
            if force_data["well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(well_pot_wall_energy=force_data["well_pot_wall_energy"][i], 
                                            well_pot_fragm_1=force_data["well_pot_fragm_1"][i], 
                                            well_pot_fragm_2=force_data["well_pot_fragm_2"][i], 
                                            well_pot_limit_dist=force_data["well_pot_limit_dist"][i])
                
                
                B_e += WP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------
        

        #print("wp")
        for i in range(len(force_data["wall_well_pot_wall_energy"])):
            if force_data["wall_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(wall_well_pot_wall_energy=force_data["wall_well_pot_wall_energy"][i],
                                            wall_well_pot_direction=force_data["wall_well_pot_direction"][i], 
                                            wall_well_pot_limit_dist=force_data["wall_well_pot_limit_dist"][i],
                                            wall_well_pot_target=force_data["wall_well_pot_target"][i])
                
                B_e += WP.calc_energy_wall(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_wall)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_wall)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------
        

        #print("wp")
        for i in range(len(force_data["void_point_well_pot_wall_energy"])):
            if force_data["void_point_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(void_point_well_pot_wall_energy=force_data["void_point_well_pot_wall_energy"][i], 
                                            void_point_well_pot_coordinate=force_data["void_point_well_pot_coordinate"][i], 
                                            void_point_well_pot_limit_dist=force_data["void_point_well_pot_limit_dist"][i],
                                            void_point_well_pot_target=force_data["void_point_well_pot_target"][i])
                
                B_e += WP.calc_energy_vp(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_vp)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_vp)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
                
            
        #------------------    

        for i in range(len(force_data["around_well_pot_wall_energy"])):
            if force_data["around_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(around_well_pot_wall_energy=force_data["around_well_pot_wall_energy"][i], 
                                            around_well_pot_center=force_data["around_well_pot_center"][i], 
                                            around_well_pot_limit_dist=force_data["around_well_pot_limit_dist"][i],
                                            around_well_pot_target=force_data["around_well_pot_target"][i])
                
                B_e += WP.calc_energy_around(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_around)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_around)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
                
            
        #------------------   
        
        if len(geom_num_list) > 2:
            for i in range(len(force_data["keep_angle_spring_const"])):
                if force_data["keep_angle_spring_const"][i] != 0.0:
                    SKAngleP = StructKeepAnglePotential(keep_angle_atom_pairs=force_data["keep_angle_atom_pairs"][i], 
                                                keep_angle_spring_const=force_data["keep_angle_spring_const"][i], 
                                                keep_angle_angle=force_data["keep_angle_angle"][i])
                    
                    B_e += SKAngleP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKAngleP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKAngleP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        #-------------------
        if len(geom_num_list) > 7:
            for i in range(len(force_data["lone_pair_keep_angle_spring_const"])):
                if force_data["lone_pair_keep_angle_spring_const"][i] != 0.0:
                    SKAngleP = StructKeepAnglePotential(lone_pair_keep_angle_spring_const=force_data["lone_pair_keep_angle_spring_const"][i], 
                                                lone_pair_keep_angle_angle=force_data["lone_pair_keep_angle_angle"][i], 
                                                lone_pair_keep_angle_atom_pair_1=force_data["lone_pair_keep_angle_atom_pair_1"][i],
                                                lone_pair_keep_angle_atom_pair_2=force_data["lone_pair_keep_angle_atom_pair_2"][i]
                                                )
                    
                    B_e += SKAngleP.calc_lone_pair_angle_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKAngleP.calc_lone_pair_angle_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKAngleP.calc_lone_pair_angle_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        
        if len(geom_num_list) > 2:
            for i in range(len(force_data["keep_angle_v2_spring_const"])):
                if not 0.0 in force_data["keep_angle_v2_spring_const"][i]:
                    if len(force_data["keep_angle_v2_spring_const"][i]) == 2 and iter != "":
                        spring_const_tmp = self.gradually_change_param(force_data["keep_angle_v2_spring_const"][i][0], force_data["keep_angle_v2_spring_const"][i][1], iter)
                        print(spring_const_tmp)
                    else:
                        spring_const_tmp = force_data["keep_angle_v2_spring_const"][i][0]
                        
                    if len(force_data["keep_angle_v2_angle"][i]) == 2 and iter != "":
                        angle_tmp = self.gradually_change_param(force_data["keep_angle_v2_angle"][i][0], force_data["keep_angle_v2_angle"][i][1], iter)
                        print(angle_tmp)
                    else:
                        angle_tmp = force_data["keep_angle_v2_angle"][i][0]
                        
                    SKAngleP = StructKeepAnglePotential(
                        keep_angle_v2_fragm1=force_data["keep_angle_v2_fragm1"][i], 
                        keep_angle_v2_fragm2=force_data["keep_angle_v2_fragm2"][i], 
                        keep_angle_v2_fragm3=force_data["keep_angle_v2_fragm3"][i], 
                                                keep_angle_v2_spring_const=spring_const_tmp, 
                                                keep_angle_v2_angle=angle_tmp)
                    
                    B_e += SKAngleP.calc_energy_v2(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKAngleP.calc_energy_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKAngleP.calc_energy_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        if len(geom_num_list) > 2:
            for i in range(len(force_data["aDD_keep_angle_spring_const"])):
                if force_data["aDD_keep_angle_spring_const"][i] != 0.0:
                    aDDKAngleP = StructKeepAnglePotential(aDD_keep_angle_spring_const=force_data["aDD_keep_angle_spring_const"][i], 
                                                aDD_keep_angle_min_angle=force_data["aDD_keep_angle_min_angle"][i], 
                                                aDD_keep_angle_max_angle=force_data["aDD_keep_angle_max_angle"][i],
                                                aDD_keep_angle_base_dist=force_data["aDD_keep_angle_base_dist"][i],
                                                aDD_keep_angle_reference_atom=force_data["aDD_keep_angle_reference_atom"][i],
                                                aDD_keep_angle_center_atom=force_data["aDD_keep_angle_center_atom"][i],
                                                aDD_keep_angle_atoms=force_data["aDD_keep_angle_atoms"][i])

                    B_e += aDDKAngleP.calc_atom_dist_dependent_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(aDDKAngleP.calc_atom_dist_dependent_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(aDDKAngleP.calc_atom_dist_dependent_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_dihedral_angle_spring_const"])):
                if force_data["keep_dihedral_angle_spring_const"][i] != 0.0:
                    SKDAP = StructKeepDihedralAnglePotential(keep_dihedral_angle_spring_const=force_data["keep_dihedral_angle_spring_const"][i], 
                                                keep_dihedral_angle_atom_pairs=force_data["keep_dihedral_angle_atom_pairs"][i], 
                                                keep_dihedral_angle_angle=force_data["keep_dihedral_angle_angle"][i])
                    
                    B_e += SKDAP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKDAP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKDAP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    pass
        else:
            pass
        #------------------
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_out_of_plain_angle_spring_const"])):
                if force_data["keep_out_of_plain_angle_spring_const"][i] != 0.0:
                    SKOPAP = StructKeepOutofPlainAnglePotential(keep_out_of_plain_angle_spring_const=force_data["keep_out_of_plain_angle_spring_const"][i], 
                                                keep_out_of_plain_angle_atom_pairs=force_data["keep_out_of_plain_angle_atom_pairs"][i], 
                                                keep_out_of_plain_angle_angle=force_data["keep_out_of_plain_angle_angle"][i])
                    
                    B_e += SKOPAP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKOPAP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKOPAP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    pass
        else:
            pass
        #------------------
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_dihedral_angle_v2_spring_const"])):
                if not 0.0 in force_data["keep_dihedral_angle_v2_spring_const"][i]:
                    if len(force_data["keep_dihedral_angle_v2_spring_const"][i]) == 2 and iter != "":
                        spring_const_tmp = self.gradually_change_param(force_data["keep_dihedral_angle_v2_spring_const"][i][0], force_data["keep_dihedral_angle_v2_spring_const"][i][1], iter)
                        print(spring_const_tmp)
                    else:
                        spring_const_tmp = force_data["keep_dihedral_angle_v2_spring_const"][i][0]
                        
                    if len(force_data["keep_dihedral_angle_v2_angle"][i]) == 2 and iter != "":
                        angle_tmp = self.gradually_change_param(force_data["keep_dihedral_angle_v2_angle"][i][0], force_data["keep_dihedral_angle_v2_angle"][i][1], iter)
                        print(angle_tmp)
                    else:
                        angle_tmp = force_data["keep_dihedral_angle_v2_angle"][i][0]
                        
                    SKDAP = StructKeepDihedralAnglePotential(keep_dihedral_angle_v2_spring_const=spring_const_tmp, 
                                                keep_dihedral_angle_v2_fragm1=force_data["keep_dihedral_angle_v2_fragm1"][i], 
                                                keep_dihedral_angle_v2_fragm2=force_data["keep_dihedral_angle_v2_fragm2"][i], 
                                                keep_dihedral_angle_v2_fragm3=force_data["keep_dihedral_angle_v2_fragm3"][i], 
                                                keep_dihedral_angle_v2_fragm4=force_data["keep_dihedral_angle_v2_fragm4"][i], 
                                                keep_dihedral_angle_v2_angle=angle_tmp)
                    
                    B_e += SKDAP.calc_energy_v2(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKDAP.calc_energy_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKDAP.calc_energy_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    pass
        else:
            pass
        #------------------
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_out_of_plain_angle_v2_spring_const"])):
                if not 0.0 in force_data["keep_out_of_plain_angle_v2_spring_const"][i]:
                    if len(force_data["keep_out_of_plain_angle_v2_spring_const"][i]) == 2 and iter != "":
                        spring_const_tmp = self.gradually_change_param(force_data["keep_out_of_plain_angle_v2_spring_const"][i][0], force_data["keep_out_of_plain_angle_v2_spring_const"][i][1], iter)
                        print(spring_const_tmp)
                    else:
                        spring_const_tmp = force_data["keep_out_of_plain_angle_v2_spring_const"][i][0]
                        
                    if len(force_data["keep_out_of_plain_angle_v2_angle"][i]) == 2 and iter != "":
                        angle_tmp = self.gradually_change_param(force_data["keep_out_of_plain_angle_v2_angle"][i][0], force_data["keep_out_of_plain_angle_v2_angle"][i][1], iter)
                        print(angle_tmp)
                    else:
                        angle_tmp = force_data["keep_out_of_plain_angle_v2_angle"][i][0]
                        
                    SKOPAP = StructKeepOutofPlainAnglePotential(keep_out_of_plain_angle_v2_spring_const=spring_const_tmp, 
                                                keep_out_of_plain_angle_v2_fragm1=force_data["keep_out_of_plain_angle_v2_fragm1"][i], 
                                                keep_out_of_plain_angle_v2_fragm2=force_data["keep_out_of_plain_angle_v2_fragm2"][i], 
                                                keep_out_of_plain_angle_v2_fragm3=force_data["keep_out_of_plain_angle_v2_fragm3"][i], 
                                                keep_out_of_plain_angle_v2_fragm4=force_data["keep_out_of_plain_angle_v2_fragm4"][i], 
                                                keep_out_of_plain_angle_v2_angle=angle_tmp)
                    
                    B_e += SKOPAP.calc_energy_v2(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKOPAP.calc_energy_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKOPAP.calc_energy_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    pass
        else:
            pass
        #------------------
        for i in range(len(force_data["void_point_pot_spring_const"])):
            if force_data["void_point_pot_spring_const"][i] != 0.0:
                for j in force_data["void_point_pot_atoms"][i]:
                    VPP = VoidPointPotential(void_point_pot_spring_const=force_data["void_point_pot_spring_const"][i], 
                                            void_point_pot_atoms=j, 
                                            void_point_pot_coord=self.ndarray2tensor(np.array(force_data["void_point_pot_coord"][i], dtype="float64")),
                                            void_point_pot_distance=force_data["void_point_pot_distance"][i],
                                            void_point_pot_order=force_data["void_point_pot_order"][i])
                                            
                    
                    B_e += VPP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(VPP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(VPP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
              
            else:
                pass
        
        #------------------
        for i in range(len(force_data["AFIR_gamma"])):
            if not 0.0 in force_data["AFIR_gamma"][i]:
                if len(force_data["AFIR_gamma"][i]) == 2 and iter != "":
                    AFIR_gamma_tmp = self.gradually_change_param(force_data["AFIR_gamma"][i][0], force_data["AFIR_gamma"][i][1], iter)
                    print(AFIR_gamma_tmp)
                else:
                    AFIR_gamma_tmp = force_data["AFIR_gamma"][i][0]
                
                AP = AFIRPotential(AFIR_gamma=AFIR_gamma_tmp, 
                                            AFIR_Fragm_1=force_data["AFIR_Fragm_1"][i], 
                                            AFIR_Fragm_2=force_data["AFIR_Fragm_2"][i],
                                            element_list=element_list)
                
                B_e += AP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(AP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                
                tensor_BPA_hessian = torch.func.hessian(AP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------    
        if len(force_data["gaussian_potential_target"]) > 0:
            if self.metaD_history_list is None:
                self.metaD_history_list = [[] for i in range(len(force_data["gaussian_potential_target"]))]
            prev_metaD_history_list = self.metaD_history_list 
            METAD = GaussianPotential(gaussian_potential_target=force_data["gaussian_potential_target"], 
                                            gaussian_potential_height=force_data["gaussian_potential_height"], 
                                            gaussian_potential_width=force_data["gaussian_potential_width"],
                                            gaussian_potential_tgt_atom=force_data["gaussian_potential_tgt_atom"])
            METAD.history_list = prev_metaD_history_list
            
            B_e += METAD.calc_energy_for_metadyn(geom_num_list)
            
            tensor_BPA_grad = torch.func.jacfwd(METAD.calc_energy_for_metadyn)(geom_num_list)
            BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

            tensor_BPA_hessian = torch.func.hessian(METAD.calc_energy_for_metadyn)(geom_num_list)
            tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian) 
            
            self.metaD_history_list = METAD.history_list
            
        #------------------        
        B_g = g + BPA_grad_list

        
        
        B_e = B_e.item()
        #new_geometry:ang. 
        #B_e:hartree

        return BPA_grad_list, B_e, B_g, BPA_hessian
    
    
    def gradually_change_param(self, param_1, param_2, iter):
        parameter = param_1 + ((param_2 - param_1)/self.partition) * int(iter)
    
        return parameter