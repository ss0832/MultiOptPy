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

    def _load_or_init_hessian(self, num, natoms, config):
        """Load existing Hessian or initialize identity matrix"""
        hessian_file = os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{num}.npy")
        if os.path.exists(hessian_file):
            return np.load(hessian_file)
        else:
            # print(f"Warning: Hessian file {hessian_file} not found. Using identity matrix.")
            return np.eye(3 * natoms)

    def _setup_rfo_optimizer(self, num, total_nodes, optimize_num, is_qsm=False):
        """Configure RSIRFO instance based on node type"""
        is_endpoint = (num == 0 or num == total_nodes - 1)
        
        if is_endpoint:
            # Endpoints: Minimize (order 0), larger trust radius
            OPT = rsirfo.RSIRFO(method="rsirfo_fsb", saddle_order=0, trust_radius=0.5)
        else:
            # Intermediate
            # QSM usually targets saddle order 0 (path finding), NEB might target saddle order 1
            saddle_order = 0 if is_qsm else 1
            OPT = rsirfo.RSIRFO(method="rsirfo_bofill", saddle_order=saddle_order, trust_radius=0.1)
        
        OPT.iteration = optimize_num
        return OPT

    def _apply_ayala_hessian_update(self, hessian, num, total_nodes, geometry_num_list, 
                                  biased_energy_list, total_force_list, STRING_FORCE_CALC):
        """
        Common Method: Update Hessian using Quintic Polynomial Fit (Ayala Stage 1).
        Checks if STRING_FORCE_CALC supports the required methods.
        """
        # Skip endpoints
        if num == 0 or num == total_nodes - 1:
            return hessian
            
        # Check if the Force Calculator supports Ayala methods
        if not hasattr(STRING_FORCE_CALC, 'get_tau') or not hasattr(STRING_FORCE_CALC, 'calculate_gamma'):
            return hessian

        tangent = STRING_FORCE_CALC.get_tau(num)
        
        # Extract local triplet safely
        start_idx = max(0, num - 1)
        end_idx = min(len(geometry_num_list), num + 2)
        
        gamma = STRING_FORCE_CALC.calculate_gamma(
            geometry_num_list[start_idx:end_idx],
            biased_energy_list[start_idx:end_idx],
            total_force_list[start_idx:end_idx],
            tangent
        )
        
        # Add curvature along the tangent direction
        # H_new = H_old + gamma * |t><t|
        hessian += gamma * np.outer(tangent, tangent)
        
        return hessian

    def _limit_step_size(self, move_vec, is_endpoint):
        """Limit the norm of the movement vector"""
        move_vec_norm = np.linalg.norm(move_vec)
        limit = 0.2 if is_endpoint else 0.1
        
        if move_vec_norm > 1e-8:
            return move_vec / move_vec_norm * min(limit, move_vec_norm)
        return move_vec


class RFOOptimizer(OptimizationAlgorithm):
    """Rational Function Optimization (RFO) optimizer (Standard NEB)"""
    
    def __init__(self, config):
        self.config = config
        self.NEB_TR = trust_radius_neb.TR_NEB(
            NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY, 
            fix_init_edge=config.fix_init_edge,
            fix_end_edge=config.fix_end_edge,
            apply_convergence_criteria=config.apply_convergence_criteria
        )
        self.apply_ts_opt = True
        self.ratio_of_rfo_step = getattr(config, "ratio_of_rfo_step", 0.5)
    
    def set_apply_ts_opt(self, apply_ts_opt):
        self.apply_ts_opt = apply_ts_opt
    
    def optimize(self, geometry_num_list, total_force_list, prev_geometry_num_list, 
                prev_total_force_list, optimize_num, biased_energy_list, 
                pre_biased_energy_list, pre_total_velocity, total_velocity, 
                cos_list, pre_geom, STRING_FORCE_CALC):
        
        natoms = len(geometry_num_list[0])
        total_nodes = len(geometry_num_list)
        
        # 1. Calc Force
        proj_total_force_list = STRING_FORCE_CALC.calc_force(
            geometry_num_list, biased_energy_list, total_force_list, optimize_num, self.config.element_list)
        
        maxima_indices = argrelextrema(biased_energy_list, np.greater)[0]
        rfo_delta_list = []
        
        for num, total_force in enumerate(total_force_list):
            # A. Load Hessian
            hessian = self._load_or_init_hessian(num, natoms, self.config)
            
            # B. Setup Optimizer
            # Note: Standard RFOOptimizer logic for 'intermediate' nodes differs slightly (saddle_order=1)
            # We preserve the original logic here manually or via the helper with is_qsm=False
            if num == 0 or num == total_nodes - 1:
                OPT = rsirfo.RSIRFO(method="rsirfo_fsb", saddle_order=0, trust_radius=0.2)
            else:
                OPT = rsirfo.RSIRFO(method="rsirfo_bofill", saddle_order=0, trust_radius=0.1)
                if num in maxima_indices and self.apply_ts_opt:
                    pass
                else:
                    OPT.switch_NEB_mode()

            OPT.iteration = optimize_num
            OPT.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
            
            # C. Hessian Processing
           
            # [ADDED] Apply Ayala Curvature Correction if available
            hessian = self._apply_ayala_hessian_update(
                hessian, num, total_nodes, geometry_num_list, 
                biased_energy_list, total_force_list, STRING_FORCE_CALC
            )

            OPT.set_hessian(hessian)
           
            # D. Prepare Steps
            if optimize_num == 0:
                OPT.Initialization = True
                pre_B_g, pre_geom_node = None, None
            else:
                OPT.Initialization = False
                pre_B_g = prev_total_force_list[num].reshape(-1, 1)
                pre_geom_node = prev_geometry_num_list[num].reshape(-1, 1)        
                
            curr_geom_node = geometry_num_list[num].reshape(-1, 1)
            B_g = total_force.reshape(-1, 1)
            
            # E. Run
            move_vec = OPT.run(curr_geom_node, B_g, pre_B_g, pre_geom_node, 0.0, 0.0, [], [], B_g, pre_B_g)
            
            # F. Limit Step
            move_vec = self._limit_step_size(move_vec, num == 0 or num == total_nodes - 1)
            
            rfo_delta_list.append(move_vec.reshape(-1, 3))
            
            # G. Save
            new_hessian = OPT.get_hessian()
            np.save(os.path.join(self.config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{num}.npy"), new_hessian)
            
        # 3. TR Calc
        rfo_move_vector_list = self.NEB_TR.TR_calc(
            geometry_num_list, total_force_list, rfo_delta_list, 
            biased_energy_list, pre_biased_energy_list, pre_geom
        )
      
        # 4. FIRE
        fire_optimizer = FIREOptimizer(self.config)
        tmp_new_geom = fire_optimizer.optimize(
            geometry_num_list, proj_total_force_list, 
            pre_total_velocity, optimize_num, total_velocity, 
            cos_list, biased_energy_list, pre_biased_energy_list, pre_geom
        )
        tmp_new_geom = np.array(tmp_new_geom, dtype="float64") / self.config.bohr2angstroms
        fire_move_vector_list = tmp_new_geom - geometry_num_list
        
        # 5. Combine
        move_vector_list = []
        for i in range(len(geometry_num_list)):
            if i == 0 or i == len(geometry_num_list) - 1:
                move_vector_list.append(-1.0 * rfo_move_vector_list[i])
            else:
                move_vector_list.append((1.0 - self.ratio_of_rfo_step) * fire_move_vector_list[i] - 
                                      self.ratio_of_rfo_step * rfo_move_vector_list[i])
                
        move_vector_list = np.array(move_vector_list, dtype="float64")
        new_geometry_list = (geometry_num_list + move_vector_list) * self.config.bohr2angstroms
        
        return new_geometry_list


class RFOQSMOptimizer(OptimizationAlgorithm):
    """Rational Function Optimization (RFO) optimizer for QSM (Quadratic String Method)"""
    
    def __init__(self, config):
        self.config = config
        self.NEB_TR = trust_radius_neb.TR_NEB(
            NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY, 
            fix_init_edge=config.fix_init_edge,
            fix_end_edge=config.fix_end_edge,
            apply_convergence_criteria=config.apply_convergence_criteria
        )
        self.ratio_of_rfo_step = getattr(config, "ratio_of_rfo_step", 0.5)
        self.is_qsmv2 = getattr(config, "qsmv2", False)
    
    def optimize(self, geometry_num_list, total_force_list, prev_geometry_num_list, 
                prev_total_force_list, optimize_num, biased_energy_list, 
                pre_biased_energy_list, pre_total_velocity, total_velocity, 
                cos_list, pre_geom, STRING_FORCE_CALC):
        
        natoms = len(geometry_num_list[0])
        total_nodes = len(geometry_num_list)
        
        # 1. Calc Force
        proj_total_force_list = STRING_FORCE_CALC.calc_force(
            geometry_num_list, biased_energy_list, total_force_list, optimize_num, self.config.element_list)
        
        rfo_delta_list = []
        
        for num, total_force in enumerate(total_force_list):
            # A. Load Hessian
            hessian = self._load_or_init_hessian(num, natoms, self.config)
            
            # B. Setup Optimizer (QSM mode uses saddle_order=0 for intermediates usually)
            OPT = self._setup_rfo_optimizer(num, total_nodes, optimize_num, is_qsm=True)
            OPT.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
            
            # C. Hessian Processing
            hessian = STRING_FORCE_CALC.calc_proj_hess(hessian, num, geometry_num_list)
            
            # [ADDED] Apply Ayala Curvature Correction (Use the base class method)
            if self.is_qsmv2:
                hessian = self._apply_ayala_hessian_update(
                    hessian, num, total_nodes, geometry_num_list, 
                    biased_energy_list, total_force_list, STRING_FORCE_CALC
                )
            
            OPT.set_hessian(hessian)
           
            # D. Prepare Steps
            if optimize_num == 0:
                OPT.Initialization = True
                pre_B_g, pre_geom_node = None, None
            else:
                OPT.Initialization = False
                pre_B_g = prev_total_force_list[num].reshape(-1, 1)
                pre_geom_node = prev_geometry_num_list[num].reshape(-1, 1)
            
            curr_geom_node = geometry_num_list[num].reshape(-1, 1)
            B_g = total_force.reshape(-1, 1)
            
            # E. Run
            move_vec = OPT.run(curr_geom_node, B_g, pre_B_g, pre_geom_node, 0.0, 0.0, [], [], B_g, pre_B_g)
            
            # F. Limit Step
            move_vec = self._limit_step_size(move_vec, num == 0 or num == total_nodes - 1)
            
            rfo_delta_list.append(move_vec.reshape(-1, 3))
            
            # G. Save
            new_hessian = OPT.get_hessian()
            np.save(os.path.join(self.config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{num}.npy"), new_hessian)
            
        # 3. TR Calc
        rfo_move_vector_list = self.NEB_TR.TR_calc(
            geometry_num_list, total_force_list, rfo_delta_list, 
            biased_energy_list, pre_biased_energy_list, pre_geom
        )

        # 4. FIRE
        fire_optimizer = FIREOptimizer(self.config)
        tmp_new_geom = fire_optimizer.optimize(
            geometry_num_list, proj_total_force_list, 
            pre_total_velocity, optimize_num, total_velocity, 
            cos_list, biased_energy_list, pre_biased_energy_list, pre_geom
        )
        tmp_new_geom = np.array(tmp_new_geom, dtype="float64") / self.config.bohr2angstroms
        fire_move_vector_list = tmp_new_geom - geometry_num_list
        
        # 5. Combine
        move_vector_list = []
        for i in range(total_nodes):
            if i == 0 or i == total_nodes - 1:
                move_vector_list.append(fire_move_vector_list[i])
            else:
                move_vector_list.append((1.0 - self.ratio_of_rfo_step) * fire_move_vector_list[i] - 
                                      self.ratio_of_rfo_step * rfo_move_vector_list[i])
                
        move_vector_list = np.array(move_vector_list, dtype="float64")
        new_geometry_list = (geometry_num_list + move_vector_list) * self.config.bohr2angstroms
        
        return new_geometry_list