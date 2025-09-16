import numpy as np
import copy

class LookAhead:
    def __init__(self, k=10, alpha=0.5, **config):
        #LookAhead algorithm
        #ref. arXiv:1907.08610
        self.iter = 0
        self.alpha = alpha
        self.k = k
        self.config = config
        self.slow_geom_num_list = []
        self.fast_geom_num_list_history = []
        self.fast_energy_history = []
        
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, move_vector):
        print("LookAhead")
        if self.iter % self.k != 0 or self.iter == 0:
            self.slow_geom_num_list = geom_num_list
            self.fast_geom_num_list_history.append(geom_num_list)
            self.fast_energy_history.append(B_e)
            
        else:
            print("update slow geometry...")
            move_vector = []
            self.fast_geom_num_list_history.append(geom_num_list)
            self.fast_energy_history.append(B_e)
            best_fast_geom_num_list = self.fast_geom_num_list_history[np.argmin(self.fast_energy_history)]
            new_geom_num_list = self.alpha * self.slow_geom_num_list + (1.0-self.alpha) * best_fast_geom_num_list
            
            move_vector = -1 * (new_geom_num_list - geom_num_list)
            
            self.slow_geom_num_list = best_fast_geom_num_list
            self.fast_geom_num_list_history = []
            self.fast_energy_history = []
        
        self.iter += 1
        return move_vector#Bohr