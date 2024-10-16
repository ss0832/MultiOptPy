
from parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, UFF_effective_charge_lib
from fileio import save_bias_pot_info, save_bias_param_grad_info

import numpy as np
import copy
import random
import torch

from LJ_repulsive_potential import LJRepulsivePotentialCone, LJRepulsivePotentialGaussian, LJRepulsivePotentialv2Value, LJRepulsivePotentialv2Scale, LJRepulsivePotentialValue, LJRepulsivePotentialScale
from AFIR_potential import AFIRPotential
from keep_potential import StructKeepPotential, StructKeepPotentialv2, StructKeepPotentialAniso
from anharmonic_keep_potential import StructAnharmonicKeepPotential
from keep_angle_potential import StructKeepAnglePotential, StructKeepAnglePotentialv2, StructKeepAnglePotentialAtomDistDependent, StructKeepAnglePotentialLonePairAngle
from keep_dihedral_angle_potential import StructKeepDihedralAnglePotential, StructKeepDihedralAnglePotentialv2, StructKeepDihedralAnglePotentialCos
from keep_outofplain_angle_potential import StructKeepOutofPlainAnglePotential, StructKeepOutofPlainAnglePotentialv2
from void_point_potential import VoidPointPotential
from switching_potential import WellPotential, WellPotentialAround, WellPotentialVP, WellPotentialWall
from gaussian_potential import GaussianPotential
from spacer_model_potential import SpacerModelPotential
from universal_potential import UniversalPotential
from flux_potential import FluxPotential
from value_range_potential import ValueRangePotential
from mechano_force_potential import LinearMechanoForcePotential, LinearMechanoForcePotentialv2
from asym_elllipsoidal_potential import AsymmetricEllipsoidalLJPotential, AsymmetricEllipsoidalLJPotentialv2

class BiasPotentialCalculation:
    def __init__(self, FOLDER_DIRECTORY="./"):
        torch.set_printoptions(precision=12)
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.JOBID = random.randint(0, 1000000)
        self.microiteration_num = 300
        self.rand_search_num = 800
        self.BPA_FOLDER_DIRECTORY = FOLDER_DIRECTORY
        self.metaD_history_list = None
        self.miter_delta = 1.0
        self.mi_bias_pot_obj_list = []
        self.mi_bias_pot_obj_id_list = []
        self.mi_bias_pot_params_list = []
        self.bias_pot_params_grad_list = None
        self.bias_pot_params_grad_name_list = None
    

    
    def main(self, e, g, geom_num_list, element_list,  force_data, pre_B_g="", iter="", initial_geom_num_list=""):
        numerical_derivative_delta = 1e-5 #unit:Bohr
        
        #g:hartree/Bohr
        #e:hartree
        #geom_num_list:Bohr

  
        #--------------------------------------------------
        B_e = torch.tensor(0.0, dtype=torch.float64)
        BPA_grad_list = g*0.0
        BPA_hessian = np.zeros((3*len(g), 3*len(g)))
        geom_num_list = ndarray2tensor(geom_num_list)
        #------------------------------------------------
        #if iter == "" or iter == 0:
        self.bias_pot_obj_list, self.bias_pot_obj_id_list, self.bias_pot_params_list = make_bias_pot_obj_list(force_data, element_list, self.BPA_FOLDER_DIRECTORY, self.JOBID, geom_num_list, iter)
        if iter == 0:
            self.mi_bias_pot_obj_list, self.mi_bias_pot_obj_id_list, self.mi_bias_pot_params_list = make_micro_iter_bias_pot_obj_list(force_data, element_list, self.BPA_FOLDER_DIRECTORY, self.JOBID, geom_num_list, iter)
        

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
            BPA_grad_list += tensor2ndarray(tensor_BPA_grad)

            tensor_BPA_hessian = torch.func.hessian(METAD.calc_energy_for_metadyn)(geom_num_list)
            tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            BPA_hessian += tensor2ndarray(tensor_BPA_hessian) 
            
            self.metaD_history_list = METAD.history_list

        tmp_bias_pot_params_grad_list = []
        tmp_bias_pot_params_grad_name_list = []
        
        
        #-----------------
        # combine almost all the bias potentials
        #-----------------
        for j in range(len(self.bias_pot_obj_list)):
            tmp_bias_pot_params = self.bias_pot_params_list[j]
            
            tmp_B_e = self.bias_pot_obj_list[j].calc_energy(geom_num_list, tmp_bias_pot_params)
            tmp_tensor_BPA_grad = torch.func.jacrev(self.bias_pot_obj_list[j].calc_energy, argnums=0)(geom_num_list, tmp_bias_pot_params)
            tmp_tensor_BPA_grad = tensor2ndarray(tmp_tensor_BPA_grad)
            tmp_tensor_BPA_hessian = torch.func.hessian(self.bias_pot_obj_list[j].calc_energy, argnums=0)(geom_num_list, tmp_bias_pot_params)
            tmp_tensor_BPA_hessian = torch.reshape(tmp_tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            tmp_tensor_BPA_hessian = tensor2ndarray(tmp_tensor_BPA_hessian)
            if len(tmp_bias_pot_params) > 0:
                results = torch.func.jacrev(self.bias_pot_obj_list[j].calc_energy, argnums=1)(geom_num_list, tmp_bias_pot_params)
                results = tensor2ndarray(results)
                print(self.bias_pot_obj_id_list[j],":dE_bias_pot/d_param: ", results)
                save_bias_param_grad_info(self.BPA_FOLDER_DIRECTORY, results, self.bias_pot_obj_id_list[j])
                tmp_bias_pot_params_grad_list.append(results)
                tmp_bias_pot_params_grad_name_list.append(self.bias_pot_obj_id_list[j])
            
            save_bias_pot_info(self.BPA_FOLDER_DIRECTORY, tmp_B_e.item(), tmp_tensor_BPA_grad, self.bias_pot_obj_id_list[j])
           
            B_e = B_e + tmp_B_e
            BPA_grad_list = BPA_grad_list + tmp_tensor_BPA_grad
            BPA_hessian = BPA_hessian + tmp_tensor_BPA_hessian
        
        ###-----------------
        # combine the bias potentials using microiteration
        ###-----------------
        for i in range(len(self.mi_bias_pot_obj_list)):
            tmp_bias_pot_params = self.mi_bias_pot_params_list[i]
            
            tmp_B_e = self.mi_bias_pot_obj_list[i].calc_energy(geom_num_list, tmp_bias_pot_params)
            tmp_tensor_BPA_grad = torch.func.jacrev(self.mi_bias_pot_obj_list[i].calc_energy, argnums=0)(geom_num_list, tmp_bias_pot_params)
            tmp_tensor_BPA_grad = tensor2ndarray(tmp_tensor_BPA_grad)
            tmp_tensor_BPA_hessian = torch.func.hessian(self.mi_bias_pot_obj_list[i].calc_energy, argnums=0)(geom_num_list, tmp_bias_pot_params)
            tmp_tensor_BPA_hessian = torch.reshape(tmp_tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            tmp_tensor_BPA_hessian = tensor2ndarray(tmp_tensor_BPA_hessian)
            if len(tmp_bias_pot_params) > 0:
                results = torch.func.jacrev(self.mi_bias_pot_obj_list[i].calc_energy, argnums=1)(geom_num_list, tmp_bias_pot_params)
                results = tensor2ndarray(results)
                print(self.mi_bias_pot_obj_id_list[i],":dE_bias_pot/d_param: ", results)
                save_bias_param_grad_info(self.BPA_FOLDER_DIRECTORY, results, self.mi_bias_pot_obj_id_list[i])
                tmp_bias_pot_params_grad_list.append(results)
                tmp_bias_pot_params_grad_name_list.append(self.mi_bias_pot_obj_id_list[i])
            
            save_bias_pot_info(self.BPA_FOLDER_DIRECTORY, tmp_B_e.item(), tmp_tensor_BPA_grad, self.mi_bias_pot_obj_id_list[i])
           
            B_e = B_e + tmp_B_e
            BPA_grad_list = BPA_grad_list + tmp_tensor_BPA_grad
            BPA_hessian = BPA_hessian + tmp_tensor_BPA_hessian
            eff_hess = self.mi_bias_pot_obj_list[i].calc_eff_hessian(geom_num_list, tmp_bias_pot_params)
            eff_hess = tensor2ndarray(eff_hess)
            BPA_hessian = BPA_hessian + eff_hess
            self.mi_bias_pot_obj_list[i].save_state()
            
        
        
         
        B_g = g + BPA_grad_list
        B_e = B_e.item() + e
        #new_geometry:ang. 
        #B_e:hartree

        self.bias_pot_params_grad_list = tmp_bias_pot_params_grad_list
        self.bias_pot_params_grad_name_list = tmp_bias_pot_params_grad_name_list
        
        return BPA_grad_list, B_e, B_g, BPA_hessian
    
def ndarray2tensor(ndarray):
    tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64, requires_grad=True))

    return tensor

def ndarray2nogradtensor(ndarray):
    tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64))

    return tensor

def tensor2ndarray(tensor):
    ndarray = copy.copy(tensor.detach().numpy())
    return ndarray

def gradually_change_param(param_1, param_2, iter):
    partition = 300
    parameter = param_1 + ((param_2 - param_1)/partition) * int(iter)
    
    return parameter

def make_micro_iter_bias_pot_obj_list(force_data, element_list, file_directory, JOBID, geom_num_list, iter):
    bias_pot_obj_list = []
    bias_pot_obj_id_list = []
    bias_pot_params_list = []
    if len(force_data["asymmetric_ellipsoidal_repulsive_potential_eps"]) > 0:
        AERP = AsymmetricEllipsoidalLJPotential(asymmetric_ellipsoidal_repulsive_potential_eps=force_data["asymmetric_ellipsoidal_repulsive_potential_eps"], 
                                    asymmetric_ellipsoidal_repulsive_potential_sig=force_data["asymmetric_ellipsoidal_repulsive_potential_sig"], 
                                    asymmetric_ellipsoidal_repulsive_potential_dist=force_data["asymmetric_ellipsoidal_repulsive_potential_dist"], 
                                    asymmetric_ellipsoidal_repulsive_potential_atoms=force_data["asymmetric_ellipsoidal_repulsive_potential_atoms"],
                                    asymmetric_ellipsoidal_repulsive_potential_offtgt=force_data["asymmetric_ellipsoidal_repulsive_potential_offtgt"],
                                    element_list=element_list,
                                    file_directory=file_directory)
        
        bias_pot_params = []
        
        for j in range(len(force_data["asymmetric_ellipsoidal_repulsive_potential_eps"])):
            tmp_list = [force_data["asymmetric_ellipsoidal_repulsive_potential_eps"][j]] + force_data["asymmetric_ellipsoidal_repulsive_potential_sig"][j] + [force_data["asymmetric_ellipsoidal_repulsive_potential_dist"][j]]
            bias_pot_params.append(tmp_list)

        
        bias_pot_params = torch.tensor(bias_pot_params, requires_grad=True, dtype=torch.float64)

        bias_pot_obj_list.append(AERP)
        bias_pot_obj_id_list.append("asymmetric_ellipsoidal_repulsive_potential")
        bias_pot_params_list.append(bias_pot_params) 
        
    if len(force_data["asymmetric_ellipsoidal_repulsive_potential_v2_eps"]) > 0:
        AERP2 = AsymmetricEllipsoidalLJPotentialv2(asymmetric_ellipsoidal_repulsive_potential_v2_eps=force_data["asymmetric_ellipsoidal_repulsive_potential_v2_eps"], 
                                    asymmetric_ellipsoidal_repulsive_potential_v2_sig=force_data["asymmetric_ellipsoidal_repulsive_potential_v2_sig"], 
                                    asymmetric_ellipsoidal_repulsive_potential_v2_dist=force_data["asymmetric_ellipsoidal_repulsive_potential_v2_dist"], 
                                    asymmetric_ellipsoidal_repulsive_potential_v2_atoms=force_data["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"],
                                    asymmetric_ellipsoidal_repulsive_potential_v2_offtgt=force_data["asymmetric_ellipsoidal_repulsive_potential_v2_offtgt"],
                                    element_list=element_list,
                                    file_directory=file_directory)
        
        bias_pot_params = []
        
        for j in range(len(force_data["asymmetric_ellipsoidal_repulsive_potential_v2_eps"])):
            tmp_list = [force_data["asymmetric_ellipsoidal_repulsive_potential_v2_eps"][j]] + force_data["asymmetric_ellipsoidal_repulsive_potential_v2_sig"][j] + [force_data["asymmetric_ellipsoidal_repulsive_potential_v2_dist"][j]]
            bias_pot_params.append(tmp_list)

        
        bias_pot_params = torch.tensor(bias_pot_params, requires_grad=True, dtype=torch.float64)

        bias_pot_obj_list.append(AERP2)
        bias_pot_obj_id_list.append("asymmetric_ellipsoidal_repulsive_potential_v2")
        bias_pot_params_list.append(bias_pot_params) 
    

    for i in range(len(force_data["spacer_model_potential_well_depth"])):
        
        if force_data["spacer_model_potential_well_depth"][i] != 0.0:
            SMP = SpacerModelPotential(spacer_model_potential_target=force_data["spacer_model_potential_target"][i],
                                           spacer_model_potential_distance=force_data["spacer_model_potential_distance"][i],
                                           spacer_model_potential_well_depth=force_data["spacer_model_potential_well_depth"][i],
                                           spacer_model_potential_cavity_scaling=force_data["spacer_model_potential_cavity_scaling"][i],
                                           spacer_model_potential_particle_number=force_data["spacer_model_potential_particle_number"][i],
                                           element_list=element_list,
                                           directory=file_directory)
            bias_pot_params = []
            bias_pot_obj_list.append(SMP)
            bias_pot_obj_id_list.append("spacer_model_potential_"+str(i))
            bias_pot_params_list.append(bias_pot_params) 


    return bias_pot_obj_list, bias_pot_obj_id_list, bias_pot_params_list


def make_bias_pot_obj_list(force_data, element_list, file_directory, JOBID, geom_num_list, iter):
    bias_pot_obj_list = []
    bias_pot_obj_id_list = []
    bias_pot_params_list = []


    for i in range(len(force_data["linear_mechano_force"])):
        if force_data["linear_mechano_force"][i] != 0.0:
            LMF = LinearMechanoForcePotential(linear_mechano_force=force_data["linear_mechano_force"][i], 
                                        linear_mechano_force_atoms_1=force_data["linear_mechano_force_atoms_1"][i], 
                                        linear_mechano_force_atoms_2=force_data["linear_mechano_force_atoms_2"][i],
                                        element_list=element_list)
            bias_pot_params = torch.tensor([force_data["linear_mechano_force"][i]], requires_grad=True, dtype=torch.float64) 
            bias_pot_obj_list.append(LMF)
            bias_pot_obj_id_list.append("linear_mechano_force_"+str(i))
            bias_pot_params_list.append(bias_pot_params)
            
        else:
            pass
        
    for i in range(len(force_data["linear_mechano_force_v2"])):
        if force_data["linear_mechano_force_v2"][i] != 0.0:
            LMF2 = LinearMechanoForcePotentialv2(linear_mechano_force_v2=force_data["linear_mechano_force_v2"][i], 
                                        linear_mechano_force_atom_v2=force_data["linear_mechano_force_atom_v2"][i], 
                                        element_list=element_list)
            bias_pot_params = torch.tensor([force_data["linear_mechano_force_v2"][i]], requires_grad=True, dtype=torch.float64) 
            bias_pot_obj_list.append(LMF2)
            bias_pot_obj_id_list.append("linear_mechano_force_v2_"+str(i))
            bias_pot_params_list.append(bias_pot_params)
            
        else:
            pass


    for i in range(len(force_data["AFIR_gamma"])):
        if not 0.0 in force_data["AFIR_gamma"][i]:
            if len(force_data["AFIR_gamma"][i]) == 2 and iter != "":
                AFIR_gamma_tmp = gradually_change_param(force_data["AFIR_gamma"][i][0], force_data["AFIR_gamma"][i][1], iter)
                print(AFIR_gamma_tmp)
            else:
                AFIR_gamma_tmp = force_data["AFIR_gamma"][i][0]
            
            bias_pot_params = torch.tensor([AFIR_gamma_tmp], requires_grad=True, dtype=torch.float64) 
            
            AP = AFIRPotential(AFIR_Fragm_1=force_data["AFIR_Fragm_1"][i], 
                                AFIR_Fragm_2=force_data["AFIR_Fragm_2"][i],
                                element_list=element_list)
            bias_pot_obj_list.append(AP)
            bias_pot_obj_id_list.append("AFIR_pot_"+str(i))
            bias_pot_params_list.append(bias_pot_params)
            
    for i in range(len(force_data["flux_pot_const"])):
        
        FP = FluxPotential(flux_pot_const=force_data["flux_pot_const"][i], 
                                        flux_pot_target=force_data["flux_pot_target"][i], 
                                        flux_pot_order=force_data["flux_pot_order"][i],
                                        flux_pot_direction=force_data["flux_pot_direction"][i],
                                        element_list=element_list,
                                        directory=file_directory)
        
        bias_pot_obj_list.append(FP)
        bias_pot_obj_id_list.append("flux_pot_"+str(i))
        bias_pot_params_list.append([])
        
    for i in range(len(force_data["value_range_upper_const"])):
        if force_data["value_range_upper_const"][i] != 0.0:
            VRP = ValueRangePotential(value_range_upper_const=force_data["value_range_upper_const"][i], 
                                        value_range_lower_const=force_data["value_range_lower_const"][i], 
                                        value_range_upper_distance=force_data["value_range_upper_distance"][i],
                                        value_range_lower_distance=force_data["value_range_lower_distance"][i],
                                        value_range_fragm_1=force_data["value_range_fragm_1"][i],
                                        value_range_fragm_2=force_data["value_range_fragm_2"][i],
                                        element_list=element_list,
                                        directory=file_directory)

            
            bias_pot_obj_list.append(VRP)
            bias_pot_obj_id_list.append("value_range_pot_"+str(i))
            bias_pot_params_list.append([])
        else:
            pass
    
    for i in range(len(force_data["universal_pot_const"])):
        if force_data["universal_pot_const"][i] != 0.0:
            UP = UniversalPotential(universal_pot_const=force_data["universal_pot_const"][i],
                                    universal_pot_target=force_data["universal_pot_target"][i],
                                        element_list=element_list,
                                        directory=file_directory)
            
            
            bias_pot_obj_list.append(UP)
            bias_pot_obj_id_list.append("flux_pot_"+str(i))
            bias_pot_params_list.append([])
        else:
            pass
 
    for i in range(len(force_data["repulsive_potential_v2_well_scale"])):
        if force_data["repulsive_potential_v2_well_scale"][i] != 0.0:
            if force_data["repulsive_potential_v2_unit"][i] == "scale":
                LJRP = LJRepulsivePotentialv2Scale(repulsive_potential_v2_well_scale=force_data["repulsive_potential_v2_well_scale"][i], 
                                            repulsive_potential_v2_dist_scale=force_data["repulsive_potential_v2_dist_scale"][i], 
                                            repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                            repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                            repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"][i], 
                                            repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"][i], 
                                            repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"][i],
                                            repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"][i],
                                            repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"][i],
                                            element_list=element_list,
                                            jobid=JOBID)
                
                
                bias_pot_obj_list.append(LJRP)
                bias_pot_obj_id_list.append("repulsive_pot_v2_scale_"+str(i))
                bias_pot_params_list.append([])

            elif force_data["repulsive_potential_v2_unit"][i] == "value":
                LJRP = LJRepulsivePotentialv2Value(repulsive_potential_v2_well_value=force_data["repulsive_potential_v2_well_scale"][i], 
                                            repulsive_potential_v2_dist_value=force_data["repulsive_potential_v2_dist_scale"][i], 
                                            repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                            repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                            repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"][i], 
                                            repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"][i], 
                                            repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"][i],
                                            repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"][i],
                                            repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"][i],
                                            element_list=element_list,
                                            jobid=JOBID)
                
                
                bias_pot_obj_list.append(LJRP)
                bias_pot_obj_id_list.append("repulsive_pot_v2_value_"+str(i))
                bias_pot_params_list.append([])
                
            else:
                print("error -rpv2")
                raise "error -rpv2"
        else:
            pass

    for i in range(len(force_data["repulsive_potential_dist_scale"])):
        if force_data["repulsive_potential_well_scale"][i] != 0.0:
            if force_data["repulsive_potential_unit"][i] == "scale":
                LJRP = LJRepulsivePotentialScale(repulsive_potential_well_scale=force_data["repulsive_potential_well_scale"][i], 
                                            repulsive_potential_dist_scale=force_data["repulsive_potential_dist_scale"][i], 
                                            repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                            repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                            element_list=element_list,
                                            jobid=JOBID)
                
                
                bias_pot_obj_list.append(LJRP)
                bias_pot_obj_id_list.append("repulsive_pot_scale_"+str(i))
                bias_pot_params_list.append([])

            elif force_data["repulsive_potential_unit"][i] == "value":
                LJRP = LJRepulsivePotentialValue(repulsive_potential_well_value=force_data["repulsive_potential_well_scale"][i], 
                                            repulsive_potential_dist_value=force_data["repulsive_potential_dist_scale"][i], 
                                            repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                            repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                            element_list=element_list,
                                            jobid=JOBID)
                
                bias_pot_obj_list.append(LJRP)
                bias_pot_obj_id_list.append("repulsive_pot_value_"+str(i))
                bias_pot_params_list.append([])
            else:
                print("error -rpv2")
                raise "error -rpv2"
        else:
            pass
    
    for i in range(len(force_data["repulsive_potential_gaussian_LJ_well_depth"])):
        
        if force_data["repulsive_potential_gaussian_LJ_well_depth"][i] != 0.0 or force_data["repulsive_potential_gaussian_gau_well_depth"][i] != 0.0:
            
            LJRP = LJRepulsivePotentialGaussian(repulsive_potential_gaussian_LJ_well_depth=force_data["repulsive_potential_gaussian_LJ_well_depth"][i], 
                                            repulsive_potential_gaussian_LJ_dist=force_data["repulsive_potential_gaussian_LJ_dist"][i], 
                                            repulsive_potential_gaussian_gau_well_depth=force_data["repulsive_potential_gaussian_gau_well_depth"][i],
                                            repulsive_potential_gaussian_gau_dist=force_data["repulsive_potential_gaussian_gau_dist"][i],
                                            repulsive_potential_gaussian_gau_range=force_data["repulsive_potential_gaussian_gau_range"][i], 
                                            repulsive_potential_gaussian_fragm_1=force_data["repulsive_potential_gaussian_fragm_1"][i], 
                                            repulsive_potential_gaussian_fragm_2=force_data["repulsive_potential_gaussian_fragm_2"][i],
                                            element_list=element_list,
                                            jobid=JOBID)
                
            bias_pot_obj_list.append(LJRP)
            bias_pot_obj_id_list.append("repulsive_gaussian_pot_"+str(i))
            bias_pot_params_list.append([])
            
    for i in range(len(force_data["cone_potential_well_value"])):
        if force_data["cone_potential_well_value"][i] != 0.0:
            
            LJRP = LJRepulsivePotentialCone(cone_potential_well_value=force_data["cone_potential_well_value"][i], 
                                        cone_potential_dist_value=force_data["cone_potential_dist_value"][i], 
                                        cone_potential_cone_angle=force_data["cone_potential_cone_angle"][i],
                                        cone_potential_center=force_data["cone_potential_center"][i],
                                        cone_potential_three_atoms=force_data["cone_potential_three_atoms"][i],
                                        cone_potential_target=force_data["cone_potential_target"][i],
                                        element_list=element_list
                                        )
            
            bias_pot_obj_list.append(LJRP)
            bias_pot_obj_id_list.append("repulsive_cone_pot_"+str(i))
            bias_pot_params_list.append([])
    
    for i in range(len(force_data["keep_pot_spring_const"])):
        if force_data["keep_pot_spring_const"][i] != 0.0:
            SKP = StructKeepPotential(keep_pot_spring_const=force_data["keep_pot_spring_const"][i], 
                                        keep_pot_distance=force_data["keep_pot_distance"][i], 
                                        keep_pot_atom_pairs=force_data["keep_pot_atom_pairs"][i])
            
            bias_pot_params = torch.tensor([force_data["keep_pot_spring_const"][i], force_data["keep_pot_distance"][i]], requires_grad=True, dtype=torch.float64) 
            
            bias_pot_obj_list.append(SKP)
            bias_pot_obj_id_list.append("keep_pot_"+str(i))
            bias_pot_params_list.append(bias_pot_params)
        else:
            pass
        
    for i in range(len(force_data["aniso_keep_pot_v2_spring_const_mat"])):
        if np.any(force_data["aniso_keep_pot_v2_spring_const_mat"][i] != 0.0):
            SKP = StructKeepPotentialAniso(aniso_keep_pot_v2_spring_const_mat=force_data["aniso_keep_pot_v2_spring_const_mat"][i], 
                                        aniso_keep_pot_v2_dist=force_data["aniso_keep_pot_v2_dist"][i], 
                                            aniso_keep_pot_v2_fragm1=force_data["aniso_keep_pot_v2_fragm1"][i],
                                        aniso_keep_pot_v2_fragm2=force_data["aniso_keep_pot_v2_fragm2"][i]
                                        )
            
            bias_pot_obj_list.append(SKP)
            bias_pot_obj_id_list.append("keep_pot_aniso_v2_"+str(i))
            bias_pot_params_list.append([])
            

        else:
            pass
            
    for i in range(len(force_data["keep_pot_v2_spring_const"])):
        if not 0.0 in force_data["keep_pot_v2_spring_const"][i]:
            if len(force_data["keep_pot_v2_spring_const"][i]) == 2 and iter != "":
                spring_const_tmp = gradually_change_param(force_data["keep_pot_v2_spring_const"][i][0], force_data["keep_pot_v2_spring_const"][i][1], iter)
                print(spring_const_tmp)
            else:
                spring_const_tmp = force_data["keep_pot_v2_spring_const"][i][0]
                
            if len(force_data["keep_pot_v2_distance"][i]) == 2 and iter != "":
                dist_tmp = gradually_change_param(force_data["keep_pot_v2_distance"][i][0], force_data["keep_pot_v2_distance"][i][1], iter)
                print(dist_tmp)
            else:
                dist_tmp = force_data["keep_pot_v2_distance"][i][0]      
            SKP = StructKeepPotentialv2(keep_pot_v2_spring_const=spring_const_tmp, 
                                        keep_pot_v2_distance=dist_tmp, 
                                        keep_pot_v2_fragm1=force_data["keep_pot_v2_fragm1"][i],
                                        keep_pot_v2_fragm2=force_data["keep_pot_v2_fragm2"][i]
                                        )
            bias_pot_params = torch.tensor([spring_const_tmp, dist_tmp], requires_grad=True, dtype=torch.float64) 
            
            bias_pot_obj_list.append(SKP)
            bias_pot_obj_id_list.append("keep_pot_v2_"+str(i))
            bias_pot_params_list.append(bias_pot_params)
        else:
            pass
                
    
    for i in range(len(force_data["anharmonic_keep_pot_spring_const"])):
        if force_data["anharmonic_keep_pot_spring_const"][i] != 0.0:
            SAKP = StructAnharmonicKeepPotential(anharmonic_keep_pot_spring_const=force_data["anharmonic_keep_pot_spring_const"][i], 
                                        anharmonic_keep_pot_potential_well_depth=force_data["anharmonic_keep_pot_potential_well_depth"][i], 
                                        anharmonic_keep_pot_atom_pairs=force_data["anharmonic_keep_pot_atom_pairs"][i],
                                        anharmonic_keep_pot_distance=force_data["anharmonic_keep_pot_distance"][i])
            
            bias_pot_obj_list.append(SAKP)
            bias_pot_obj_id_list.append("anharmonic_keep_pot_"+str(i))
            bias_pot_params_list.append([])
            
        else:
            pass
    
    for i in range(len(force_data["well_pot_wall_energy"])):
        if force_data["well_pot_wall_energy"][i] != 0.0:
            WP = WellPotential(well_pot_wall_energy=force_data["well_pot_wall_energy"][i], 
                                        well_pot_fragm_1=force_data["well_pot_fragm_1"][i], 
                                        well_pot_fragm_2=force_data["well_pot_fragm_2"][i], 
                                        well_pot_limit_dist=force_data["well_pot_limit_dist"][i])
            
            bias_pot_obj_list.append(WP)
            bias_pot_obj_id_list.append("well_pot_wall_"+str(i))
            bias_pot_params_list.append([])
        else:
            pass

    for i in range(len(force_data["wall_well_pot_wall_energy"])):
        if force_data["wall_well_pot_wall_energy"][i] != 0.0:
            WP = WellPotentialWall(wall_well_pot_wall_energy=force_data["wall_well_pot_wall_energy"][i],
                                        wall_well_pot_direction=force_data["wall_well_pot_direction"][i], 
                                        wall_well_pot_limit_dist=force_data["wall_well_pot_limit_dist"][i],
                                        wall_well_pot_target=force_data["wall_well_pot_target"][i])
            
            bias_pot_obj_list.append(WP)
            bias_pot_obj_id_list.append("wall_well_pot_wall_"+str(i))
            bias_pot_params_list.append([])
        else:
            pass

    for i in range(len(force_data["void_point_well_pot_wall_energy"])):
        if force_data["void_point_well_pot_wall_energy"][i] != 0.0:
            WP = WellPotentialVP(void_point_well_pot_wall_energy=force_data["void_point_well_pot_wall_energy"][i], 
                                        void_point_well_pot_coordinate=force_data["void_point_well_pot_coordinate"][i], 
                                        void_point_well_pot_limit_dist=force_data["void_point_well_pot_limit_dist"][i],
                                        void_point_well_pot_target=force_data["void_point_well_pot_target"][i])
            
            bias_pot_obj_list.append(WP)
            bias_pot_obj_id_list.append("void_point_well_pot_wall_"+str(i))
            bias_pot_params_list.append([])
            
        else:
            pass
            
    for i in range(len(force_data["around_well_pot_wall_energy"])):
        if force_data["around_well_pot_wall_energy"][i] != 0.0:
            WP = WellPotentialAround(around_well_pot_wall_energy=force_data["around_well_pot_wall_energy"][i], 
                                        around_well_pot_center=force_data["around_well_pot_center"][i], 
                                        around_well_pot_limit_dist=force_data["around_well_pot_limit_dist"][i],
                                        around_well_pot_target=force_data["around_well_pot_target"][i])
            
            bias_pot_obj_list.append(WP)
            bias_pot_obj_id_list.append("around_well_pot_wall_"+str(i))
            bias_pot_params_list.append([])
            
        else:
            pass
            
    if len(geom_num_list) > 2:
        for i in range(len(force_data["keep_angle_spring_const"])):
            if force_data["keep_angle_spring_const"][i] != 0.0:
                SKAngleP = StructKeepAnglePotential(keep_angle_atom_pairs=force_data["keep_angle_atom_pairs"][i], 
                                            keep_angle_spring_const=force_data["keep_angle_spring_const"][i], 
                                            keep_angle_angle=force_data["keep_angle_angle"][i])
                
                bias_pot_params = torch.tensor([force_data["keep_angle_spring_const"][i], force_data["keep_angle_angle"][i]], requires_grad=True, dtype=torch.float64) 
                bias_pot_obj_list.append(SKAngleP)
                bias_pot_obj_id_list.append("keep_angle_pot_"+str(i))
                bias_pot_params_list.append(bias_pot_params)

    else:
        pass

    if len(geom_num_list) > 7:
        for i in range(len(force_data["lone_pair_keep_angle_spring_const"])):
            if force_data["lone_pair_keep_angle_spring_const"][i] != 0.0:
                SKAngleP = StructKeepAnglePotentialLonePairAngle(lone_pair_keep_angle_spring_const=force_data["lone_pair_keep_angle_spring_const"][i], 
                                            lone_pair_keep_angle_angle=force_data["lone_pair_keep_angle_angle"][i], 
                                            lone_pair_keep_angle_atom_pair_1=force_data["lone_pair_keep_angle_atom_pair_1"][i],
                                            lone_pair_keep_angle_atom_pair_2=force_data["lone_pair_keep_angle_atom_pair_2"][i]
                                            )
                
                bias_pot_obj_list.append(SKAngleP)
                bias_pot_obj_id_list.append("lone_pair_keep_angle_"+str(i))   
                bias_pot_params_list.append([])

    else:
        pass
    
    if len(geom_num_list) > 2:
        for i in range(len(force_data["keep_angle_v2_spring_const"])):
            if not 0.0 in force_data["keep_angle_v2_spring_const"][i]:
                if len(force_data["keep_angle_v2_spring_const"][i]) == 2 and iter != "":
                    spring_const_tmp = gradually_change_param(force_data["keep_angle_v2_spring_const"][i][0], force_data["keep_angle_v2_spring_const"][i][1], iter)
                    print(spring_const_tmp)
                else:
                    spring_const_tmp = force_data["keep_angle_v2_spring_const"][i][0]
                    
                if len(force_data["keep_angle_v2_angle"][i]) == 2 and iter != "":
                    angle_tmp = gradually_change_param(force_data["keep_angle_v2_angle"][i][0], force_data["keep_angle_v2_angle"][i][1], iter)
                    print(angle_tmp)
                else:
                    angle_tmp = force_data["keep_angle_v2_angle"][i][0]
                bias_pot_params = torch.tensor([spring_const_tmp, angle_tmp], requires_grad=True, dtype=torch.float64) 
                SKAngleP = StructKeepAnglePotentialv2(
                    keep_angle_v2_fragm1=force_data["keep_angle_v2_fragm1"][i], 
                    keep_angle_v2_fragm2=force_data["keep_angle_v2_fragm2"][i], 
                    keep_angle_v2_fragm3=force_data["keep_angle_v2_fragm3"][i], 
                                            keep_angle_v2_spring_const=spring_const_tmp, 
                                            keep_angle_v2_angle=angle_tmp)
                
                bias_pot_obj_list.append(SKAngleP)
                bias_pot_obj_id_list.append("keep_angle_v2_"+str(i))   
                bias_pot_params_list.append(bias_pot_params)

    else:
        pass
    
    if len(geom_num_list) > 2:
        for i in range(len(force_data["aDD_keep_angle_spring_const"])):
            if force_data["aDD_keep_angle_spring_const"][i] != 0.0:
                aDDKAngleP = StructKeepAnglePotentialAtomDistDependent(aDD_keep_angle_spring_const=force_data["aDD_keep_angle_spring_const"][i], 
                                            aDD_keep_angle_min_angle=force_data["aDD_keep_angle_min_angle"][i], 
                                            aDD_keep_angle_max_angle=force_data["aDD_keep_angle_max_angle"][i],
                                            aDD_keep_angle_base_dist=force_data["aDD_keep_angle_base_dist"][i],
                                            aDD_keep_angle_reference_atom=force_data["aDD_keep_angle_reference_atom"][i],
                                            aDD_keep_angle_center_atom=force_data["aDD_keep_angle_center_atom"][i],
                                            aDD_keep_angle_atoms=force_data["aDD_keep_angle_atoms"][i])

                bias_pot_obj_list.append(aDDKAngleP)
                bias_pot_obj_id_list.append("aDD_keep_angle_"+str(i))   
                bias_pot_params_list.append([])

    else:
        pass
    
    if len(geom_num_list) > 3:
        for i in range(len(force_data["keep_dihedral_angle_spring_const"])):
            if force_data["keep_dihedral_angle_spring_const"][i] != 0.0:
                SKDAP = StructKeepDihedralAnglePotential(keep_dihedral_angle_spring_const=force_data["keep_dihedral_angle_spring_const"][i], 
                                            keep_dihedral_angle_atom_pairs=force_data["keep_dihedral_angle_atom_pairs"][i], 
                                            keep_dihedral_angle_angle=force_data["keep_dihedral_angle_angle"][i])
                
                
                bias_pot_params = torch.tensor([force_data["keep_dihedral_angle_spring_const"][i], force_data["keep_dihedral_angle_angle"][i]], requires_grad=True, dtype=torch.float64) 
                bias_pot_obj_list.append(SKDAP)
                bias_pot_obj_id_list.append("keep_dihedral_angle_"+str(i))   
                bias_pot_params_list.append(bias_pot_params)
            else:
                pass
    else:
        pass

    if len(geom_num_list) > 3:
        for i in range(len(force_data["keep_out_of_plain_angle_spring_const"])):
            if force_data["keep_out_of_plain_angle_spring_const"][i] != 0.0:
                SKOPAP = StructKeepOutofPlainAnglePotential(keep_out_of_plain_angle_spring_const=force_data["keep_out_of_plain_angle_spring_const"][i], 
                                            keep_out_of_plain_angle_atom_pairs=force_data["keep_out_of_plain_angle_atom_pairs"][i], 
                                            keep_out_of_plain_angle_angle=force_data["keep_out_of_plain_angle_angle"][i])
                bias_pot_params = torch.tensor([force_data["keep_out_of_plain_angle_spring_const"][i], force_data["keep_out_of_plain_angle_angle"][i]], requires_grad=True, dtype=torch.float64)
                bias_pot_obj_list.append(SKOPAP)
                bias_pot_obj_id_list.append("keep_out_of_plain_angle_"+str(i))   
                bias_pot_params_list.append(bias_pot_params)
            else:
                pass
    else:
        pass
    
    if len(geom_num_list) > 3:
        for i in range(len(force_data["keep_dihedral_angle_v2_spring_const"])):
            if not 0.0 in force_data["keep_dihedral_angle_v2_spring_const"][i]:
                if len(force_data["keep_dihedral_angle_v2_spring_const"][i]) == 2 and iter != "":
                    spring_const_tmp = gradually_change_param(force_data["keep_dihedral_angle_v2_spring_const"][i][0], force_data["keep_dihedral_angle_v2_spring_const"][i][1], iter)
                    print(spring_const_tmp)
                else:
                    spring_const_tmp = force_data["keep_dihedral_angle_v2_spring_const"][i][0]
                    
                if len(force_data["keep_dihedral_angle_v2_angle"][i]) == 2 and iter != "":
                    angle_tmp = gradually_change_param(force_data["keep_dihedral_angle_v2_angle"][i][0], force_data["keep_dihedral_angle_v2_angle"][i][1], iter)
                    print(angle_tmp)
                else:
                    angle_tmp = force_data["keep_dihedral_angle_v2_angle"][i][0]
                    
                SKDAP = StructKeepDihedralAnglePotentialv2(keep_dihedral_angle_v2_spring_const=spring_const_tmp, 
                                            keep_dihedral_angle_v2_fragm1=force_data["keep_dihedral_angle_v2_fragm1"][i], 
                                            keep_dihedral_angle_v2_fragm2=force_data["keep_dihedral_angle_v2_fragm2"][i], 
                                            keep_dihedral_angle_v2_fragm3=force_data["keep_dihedral_angle_v2_fragm3"][i], 
                                            keep_dihedral_angle_v2_fragm4=force_data["keep_dihedral_angle_v2_fragm4"][i], 
                                            keep_dihedral_angle_v2_angle=angle_tmp)
                bias_pot_params = torch.tensor([spring_const_tmp, angle_tmp], requires_grad=True, dtype=torch.float64) 

                bias_pot_obj_list.append(SKDAP)
                bias_pot_obj_id_list.append("keep_dihedral_angle_v2_"+str(i))  
                bias_pot_params_list.append(bias_pot_params)

            else:
                pass
    else:
        pass

    if len(geom_num_list) > 3:
        for i in range(len(force_data["keep_dihedral_angle_cos_potential_const"])):
            if not 0.0 in force_data["keep_dihedral_angle_cos_potential_const"][i]:
                if len(force_data["keep_dihedral_angle_cos_potential_const"][i]) == 2 and iter != "":
                    potential_const_tmp = gradually_change_param(force_data["keep_dihedral_angle_cos_spring_const"][i][0], force_data["keep_dihedral_angle_cos_spring_const"][i][1], iter)
                    print(potential_const_tmp)
                else:
                    potential_const_tmp = force_data["keep_dihedral_angle_cos_potential_const"][i][0]
                    
                if len(force_data["keep_dihedral_angle_cos_angle_const"][i]) == 2 and iter != "":
                    angle_const_tmp = gradually_change_param(force_data["keep_dihedral_angle_cos_angle_const"][i][0], force_data["keep_dihedral_angle_cos_angle_const"][i][1], iter)
                    print(angle_const_tmp)
                else:
                    angle_const_tmp = force_data["keep_dihedral_angle_cos_angle_const"][i][0]
                
                if len(force_data["keep_dihedral_angle_cos_angle"][i]) == 2 and iter != "":
                    angle_tmp = gradually_change_param(force_data["keep_dihedral_angle_cos_angle"][i][0], force_data["keep_dihedral_angle_cos_angle"][i][1], iter)
                    print(angle_tmp)
                else:
                    angle_tmp = force_data["keep_dihedral_angle_cos_angle"][i][0]
                
                    
                SKDAP = StructKeepDihedralAnglePotentialCos(keep_dihedral_angle_cos_potential_const=potential_const_tmp, 
                                            keep_dihedral_angle_cos_angle_const=angle_const_tmp,
                                            keep_dihedral_angle_cos_fragm1=force_data["keep_dihedral_angle_cos_fragm1"][i], 
                                            keep_dihedral_angle_cos_fragm2=force_data["keep_dihedral_angle_cos_fragm2"][i], 
                                            keep_dihedral_angle_cos_fragm3=force_data["keep_dihedral_angle_cos_fragm3"][i], 
                                            keep_dihedral_angle_cos_fragm4=force_data["keep_dihedral_angle_cos_fragm4"][i], 
                                            keep_dihedral_angle_cos_angle=angle_tmp)
                bias_pot_obj_list.append(SKDAP)
                bias_pot_obj_id_list.append("keep_dihedral_angle_cos_"+str(i))  
                bias_pot_params_list.append([])
            else:
                pass
    else:
        pass

    if len(geom_num_list) > 3:
        for i in range(len(force_data["keep_out_of_plain_angle_v2_spring_const"])):
            if not 0.0 in force_data["keep_out_of_plain_angle_v2_spring_const"][i]:
                if len(force_data["keep_out_of_plain_angle_v2_spring_const"][i]) == 2 and iter != "":
                    spring_const_tmp = gradually_change_param(force_data["keep_out_of_plain_angle_v2_spring_const"][i][0], force_data["keep_out_of_plain_angle_v2_spring_const"][i][1], iter)
                    print(spring_const_tmp)
                else:
                    spring_const_tmp = force_data["keep_out_of_plain_angle_v2_spring_const"][i][0]
                    
                if len(force_data["keep_out_of_plain_angle_v2_angle"][i]) == 2 and iter != "":
                    angle_tmp = gradually_change_param(force_data["keep_out_of_plain_angle_v2_angle"][i][0], force_data["keep_out_of_plain_angle_v2_angle"][i][1], iter)
                    print(angle_tmp)
                else:
                    angle_tmp = force_data["keep_out_of_plain_angle_v2_angle"][i][0]
                bias_pot_params = torch.tensor([spring_const_tmp, angle_tmp], requires_grad=True, dtype=torch.float64)
                SKOPAP = StructKeepOutofPlainAnglePotentialv2(keep_out_of_plain_angle_v2_spring_const=spring_const_tmp, 
                                            keep_out_of_plain_angle_v2_fragm1=force_data["keep_out_of_plain_angle_v2_fragm1"][i], 
                                            keep_out_of_plain_angle_v2_fragm2=force_data["keep_out_of_plain_angle_v2_fragm2"][i], 
                                            keep_out_of_plain_angle_v2_fragm3=force_data["keep_out_of_plain_angle_v2_fragm3"][i], 
                                            keep_out_of_plain_angle_v2_fragm4=force_data["keep_out_of_plain_angle_v2_fragm4"][i], 
                                            keep_out_of_plain_angle_v2_angle=angle_tmp)
                
                bias_pot_obj_list.append(SKOPAP)
                bias_pot_obj_id_list.append("keep_out_of_plain_angle_v2_"+str(i))  
                bias_pot_params_list.append(bias_pot_params)
            else:
                pass
    else:
        pass

    for i in range(len(force_data["void_point_pot_spring_const"])):
        if force_data["void_point_pot_spring_const"][i] != 0.0:
            for j in force_data["void_point_pot_atoms"][i]:
                VPP = VoidPointPotential(void_point_pot_spring_const=force_data["void_point_pot_spring_const"][i], 
                                        void_point_pot_atoms=j, 
                                        void_point_pot_coord=ndarray2tensor(np.array(force_data["void_point_pot_coord"][i], dtype="float64")),
                                        void_point_pot_distance=force_data["void_point_pot_distance"][i],
                                        void_point_pot_order=force_data["void_point_pot_order"][i])
                bias_pot_obj_list.append(VPP)
                bias_pot_obj_id_list.append("void_point_pot_"+str(i))  
                bias_pot_params_list.append([])
        else:
            pass      
    
    return bias_pot_obj_list, bias_pot_obj_id_list, bias_pot_params_list
