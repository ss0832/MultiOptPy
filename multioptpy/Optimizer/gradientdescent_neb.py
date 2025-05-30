import numpy as np
from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):
    """Base class for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, geometry_num_list, total_force_list, **kwargs):
        """Execute optimization step"""
        pass




class SteepestDescentOptimizer(OptimizationAlgorithm):
    """Steepest descent optimizer"""
    
    def __init__(self, config):
        self.config = config
    
    def optimize(self, geometry_num_list, total_force_list, **kwargs):
        """Steepest descent optimization"""
        total_delta = []
        delta = 0.5
        for i in range(len(total_force_list)):
            total_delta.append(delta * total_force_list[i])

        # Apply edge constraints
        if self.config.fix_init_edge:
            move_vector = [total_delta[0] * 0.0]
        else:
            move_vector = [total_delta[0]]
            
        for i in range(1, len(total_delta) - 1):
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            if np.linalg.norm(total_delta[i]) > trust_radii_1:
                move_vector.append(total_delta[i] * trust_radii_1 / np.linalg.norm(total_delta[i]))
            elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                move_vector.append(total_delta[i] * trust_radii_2 / np.linalg.norm(total_delta[i]))
            else:
                move_vector.append(total_delta[i])
                
        if self.config.fix_end_edge:
            move_vector.append(total_delta[-1] * 0.0)
        else:
            move_vector.append(total_delta[-1])
            
        new_geometry = (geometry_num_list + move_vector) * self.config.bohr2angstroms
        return new_geometry

