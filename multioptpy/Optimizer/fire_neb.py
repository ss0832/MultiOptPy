import numpy as np
from abc import ABC, abstractmethod
from multioptpy.Optimizer import trust_radius_neb

class OptimizationAlgorithm(ABC):
    """Base class for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, geometry_num_list, total_force_list, **kwargs):
        """Execute optimization step"""
        pass


class FIREOptimizer(OptimizationAlgorithm):
    """FIRE method optimizer"""
    
    def __init__(self, config):
        self.config = config
        # FIRE method parameters
        self.dt = config.dt
        self.a = config.a
        self.n_reset = config.n_reset
        self.FIRE_N_accelerate = config.FIRE_N_accelerate
        self.FIRE_f_inc = config.FIRE_f_inc
        self.FIRE_f_accelerate = config.FIRE_f_accelerate
        self.FIRE_f_decelerate = config.FIRE_f_decelerate
        self.FIRE_a_start = config.FIRE_a_start
        self.FIRE_dt_max = config.FIRE_dt_max
        
        # Initialize NEB trust radius
        self.NEB_TR = trust_radius_neb.TR_NEB(
            NEB_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY, 
            fix_init_edge=config.fix_init_edge,
            fix_end_edge=config.fix_end_edge,
            apply_convergence_criteria=config.apply_convergence_criteria
        )
    
    def optimize(self, geometry_num_list, total_force_list, pre_total_velocity, 
                optimize_num, total_velocity, cos_list, biased_energy_list, 
                pre_biased_energy_list, pre_geom):
        """FIRE method optimization"""
        velocity_neb = []

        for num in range(len(total_velocity)):
            part_velocity_neb = []
            for i in range(len(total_force_list[0])):
                force_norm = np.linalg.norm(total_force_list[num][i])
                velocity_norm = np.linalg.norm(total_velocity[num][i])
                if force_norm > 1e-10:  # avoid division by zero
                    part_velocity_neb.append(
                        (1.0 - self.a) * total_velocity[num][i] + 
                        self.a * (velocity_norm / force_norm) * total_force_list[num][i]
                    )
                else:
                    part_velocity_neb.append(total_velocity[num][i])
            velocity_neb.append(part_velocity_neb)

        velocity_neb = np.array(velocity_neb)

        np_dot_param = 0.0
        if optimize_num != 0 and len(pre_total_velocity) > 1:
            np_dot_param = np.sum([np.dot(pre_total_velocity[num_1][num_2], total_force_num.T) 
                                 for num_1, total_force in enumerate(total_force_list) 
                                 for num_2, total_force_num in enumerate(total_force)])
            print(np_dot_param)

        if optimize_num > 0 and np_dot_param > 0 and len(pre_total_velocity) > 1:
            if self.n_reset > self.FIRE_N_accelerate:
                self.dt = min(self.dt * self.FIRE_f_inc, self.FIRE_dt_max)
                self.a *= self.FIRE_f_inc
            self.n_reset += 1
        else:
            velocity_neb *= 0
            self.a = self.FIRE_a_start
            self.dt *= self.FIRE_f_decelerate
            self.n_reset = 0

        total_velocity = velocity_neb + self.dt * total_force_list

        if optimize_num != 0 and len(pre_total_velocity) > 1:
            total_delta = self.dt * (total_velocity + pre_total_velocity)
        else:
            total_delta = self.dt * total_velocity

        # Calculate the movement vector using TR_calc
        move_vector = self.NEB_TR.TR_calc(geometry_num_list, total_force_list, total_delta, 
                                        biased_energy_list, pre_biased_energy_list, pre_geom)

        # Update geometry using the move vector
        new_geometry = (geometry_num_list + move_vector) * self.config.bohr2angstroms

        return new_geometry
