import numpy as np
from abc import ABC, abstractmethod
from multioptpy.Optimizer import trust_radius_neb
from multioptpy.Optimizer.fire_neb import FIREOptimizer
from scipy.signal import argrelextrema
from multioptpy.Optimizer import rsirfo
import os
import copy


class OptimizationAlgorithm(ABC):
    """Base class for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, geometry_num_list, total_force_list, **kwargs):
        """Execute optimization step"""
        pass



class RFOOptimizer(OptimizationAlgorithm):
    """Rational Function Optimization (RFO) optimizer"""
    
    def __init__(self, config):
        self.config = config
        # Initialize NEB trust radius
        self.NEB_TR = trust_radius_neb.TR_NEB(
            NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY, 
            fix_init_edge=config.fix_init_edge,
            fix_end_edge=config.fix_end_edge,
            apply_convergence_criteria=config.apply_convergence_criteria
        )
        self.apply_ts_opt = True
        self.ratio_of_rfo_step = config.ratio_of_rfo_step if hasattr(config, "ratio_of_rfo_step") else 0.5  # Ratio of RFO step in combined step
    
    def set_apply_ts_opt(self, apply_ts_opt):#apply_ts_opt:Boolean
        """Set whether to apply transition state optimization"""
        self.apply_ts_opt = apply_ts_opt
    
    def optimize(self, geometry_num_list, total_force_list, prev_geometry_num_list, 
                prev_total_force_list, optimize_num, biased_energy_list, 
                pre_biased_energy_list, pre_total_velocity, total_velocity, 
                cos_list, pre_geom, STRING_FORCE_CALC):
        """RFO optimization with FIRE combination"""
        
        natoms = len(geometry_num_list[0])
        
        proj_total_force_list = STRING_FORCE_CALC.calc_force(
            geometry_num_list, biased_energy_list, total_force_list, optimize_num, self.config.element_list)
        
        maxima_indices = argrelextrema(biased_energy_list, np.greater)[0]
        total_delta = []
        
        for num, total_force in enumerate(total_force_list):
            hessian_file = self.config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(num) + ".npy"
            if os.path.exists(hessian_file):
                hessian = np.load(hessian_file)
            else:
                # Create identity matrix as fallback
                print(f"Warning: Hessian file {hessian_file} not found. Using identity matrix.")
                hessian = np.eye(3 * natoms)
            
            if num == 0 or num == len(total_force_list) - 1:
                OPT = rsirfo.RSIRFO(method="rsirfo_fsb", saddle_order=0, trust_radius=0.2)
            else:
                OPT = rsirfo.RSIRFO(method="rsirfo_bofill", saddle_order=1, trust_radius=0.1)
                if num in maxima_indices and self.apply_ts_opt:
                    pass
                else:
                    OPT.switch_NEB_mode()
                    
            OPT.iteration = optimize_num
            OPT.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
            OPT.set_hessian(hessian)
           
            if optimize_num == 0:
                OPT.Initialization = True
                pre_B_g = None
                pre_geom = None
            else:
                OPT.Initialization = False
                pre_B_g = prev_total_force_list[num].reshape(-1, 1)
                pre_geom = prev_geometry_num_list[num].reshape(-1, 1)        
                
            geom_num_list = geometry_num_list[num].reshape(-1, 1)
            B_g = total_force.reshape(-1, 1)
            
            move_vec = OPT.run(geom_num_list, B_g, pre_B_g, pre_geom, 0.0, 0.0, [], [], B_g, pre_B_g)
            
            if num == 0 or num == len(total_force_list) - 1:
                move_vec_norm = np.linalg.norm(move_vec)
                if move_vec_norm > 1e-8:
                    move_vec = move_vec / move_vec_norm * min(0.2, move_vec_norm)
            else:
                move_vec_norm = np.linalg.norm(move_vec)
                if move_vec_norm > 1e-8:
                    move_vec = move_vec / move_vec_norm * min(0.1, move_vec_norm)
          
            total_delta.append(move_vec.reshape(-1, 3))
            hessian = OPT.get_hessian()
            np.save(self.config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(num) + ".npy", hessian)
            
      
        rfo_move_vector_list = self.NEB_TR.TR_calc(geometry_num_list, total_force_list, total_delta, 
                                                 biased_energy_list, pre_biased_energy_list, pre_geom)
      
        
        # Calculate FIRE movement for comparison
        fire_optimizer = FIREOptimizer(self.config)
        tmp_new_geom = fire_optimizer.optimize(geometry_num_list, proj_total_force_list, 
                                             pre_total_velocity, optimize_num, total_velocity, 
                                             cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
        tmp_new_geom = np.array(tmp_new_geom, dtype="float64") / self.config.bohr2angstroms
        fire_move_vector_list = tmp_new_geom - geometry_num_list
        
        # Combine RFO and FIRE movements
        move_vector_list = []
        for i in range(len(geometry_num_list)):
            if i == 0 or i == len(geometry_num_list) - 1:
                move_vector_list.append(-1.0 * rfo_move_vector_list[i])
            else:
                move_vector_list.append((1.0 - self.ratio_of_rfo_step) * fire_move_vector_list[i] - self.ratio_of_rfo_step * rfo_move_vector_list[i])
                
        move_vector_list = np.array(move_vector_list, dtype="float64")
        #move_vector_list = projection(move_vector_list, geometry_num_list)
        new_geometry_list = (geometry_num_list + move_vector_list) * self.config.bohr2angstroms
        
        return new_geometry_list


class RFOQSMOptimizer(OptimizationAlgorithm):
    """Rational Function Optimization (RFO) optimizer for QSM"""
    
    def __init__(self, config):
        self.config = config
        # Initialize NEB trust radius
        self.NEB_TR = trust_radius_neb.TR_NEB(
            NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY, 
            fix_init_edge=config.fix_init_edge,
            fix_end_edge=config.fix_end_edge,
            apply_convergence_criteria=config.apply_convergence_criteria
        )
        self.ratio_of_rfo_step = config.ratio_of_rfo_step if hasattr(config, "ratio_of_rfo_step") else 0.5  # Ratio of RFO step in combined step
    
    def optimize(self, geometry_num_list, total_force_list, prev_geometry_num_list, 
                prev_total_force_list, optimize_num, biased_energy_list, 
                pre_biased_energy_list, pre_total_velocity, total_velocity, 
                cos_list, pre_geom, STRING_FORCE_CALC):
        """RFO optimization with FIRE combination"""
        
        natoms = len(geometry_num_list[0])
        
        proj_total_force_list = STRING_FORCE_CALC.calc_force(
            geometry_num_list, biased_energy_list, total_force_list, optimize_num, self.config.element_list)
        
        maxima_indices = argrelextrema(biased_energy_list, np.greater)[0]
        total_delta = []
        
        for num, total_force in enumerate(total_force_list):
            hessian_file = self.config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(num) + ".npy"
            if os.path.exists(hessian_file):
                hessian = np.load(hessian_file)
            else:
                # Create identity matrix as fallback
                print(f"Warning: Hessian file {hessian_file} not found. Using identity matrix.")
                hessian = np.eye(3 * natoms)
            
            if num == 0 or num == len(total_force_list) - 1:
                OPT = rsirfo.RSIRFO(method="rsirfo_fsb", saddle_order=0, trust_radius=0.5)
            else:
                OPT = rsirfo.RSIRFO(method="rsirfo_bofill", saddle_order=0, trust_radius=0.1)
                
                    
            OPT.iteration = optimize_num
            OPT.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
            
            
            hessian = STRING_FORCE_CALC.calc_proj_hess(hessian, num, geometry_num_list)
            
            OPT.set_hessian(hessian)
           
            if optimize_num == 0:
                OPT.Initialization = True
                pre_B_g = None
                pre_geom = None
            else:
                OPT.Initialization = False
                pre_B_g = prev_total_force_list[num].reshape(-1, 1)
                pre_geom = prev_geometry_num_list[num].reshape(-1, 1)        
                
            geom_num_list = geometry_num_list[num].reshape(-1, 1)
            B_g = total_force.reshape(-1, 1)
            
            move_vec = OPT.run(geom_num_list, B_g, pre_B_g, pre_geom, 0.0, 0.0, [], [], B_g, pre_B_g)
            
            if num == 0 or num == len(total_force_list) - 1:
                move_vec_norm = np.linalg.norm(move_vec)
                if move_vec_norm > 1e-8:
                    move_vec = move_vec / move_vec_norm * min(0.1, move_vec_norm)
            else:
                move_vec_norm = np.linalg.norm(move_vec)
                if move_vec_norm > 1e-8:
                    move_vec = move_vec / move_vec_norm * min(0.1, move_vec_norm)
          
            total_delta.append(move_vec.reshape(-1, 3))
            hessian = OPT.get_hessian()
            np.save(self.config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(num) + ".npy", hessian)
            
      
        rfo_move_vector_list = self.NEB_TR.TR_calc(geometry_num_list, total_force_list, total_delta, 
                                                 biased_energy_list, pre_biased_energy_list, pre_geom)
      
        
        # Calculate FIRE movement for comparison
        fire_optimizer = FIREOptimizer(self.config)
        tmp_new_geom = fire_optimizer.optimize(geometry_num_list, proj_total_force_list, 
                                             pre_total_velocity, optimize_num, total_velocity, 
                                             cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
        tmp_new_geom = np.array(tmp_new_geom, dtype="float64") / self.config.bohr2angstroms
        fire_move_vector_list = tmp_new_geom - geometry_num_list
        
        # Combine RFO and FIRE movements
        move_vector_list = []
        for i in range(len(geometry_num_list)):
            if i == 0 or i == len(geometry_num_list) - 1:
                move_vector_list.append(fire_move_vector_list[i])
            else:
                move_vector_list.append((1.0 - self.ratio_of_rfo_step) * fire_move_vector_list[i] - self.ratio_of_rfo_step * rfo_move_vector_list[i])
                
        move_vector_list = np.array(move_vector_list, dtype="float64")
        new_geometry_list = (geometry_num_list + move_vector_list) * self.config.bohr2angstroms
        
        return new_geometry_list
