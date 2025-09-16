import numpy as np
import copy

class LARS:
    def __init__(self, beta=0.6, **config):
        # ref. arXiv:1708.03888
        # https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20
        self.beta = beta
        self.config = config
        self.iter = 0
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, tmp_move_vector):
        print("LARS")
        weight = np.clip(np.linalg.norm(geom_num_list), 0, 10)
        
        scaled_lr = (weight / (np.linalg.norm(tmp_move_vector) + weight * self.beta))
        
        self.iter += 1
        print("scaled learning rate: ", scaled_lr)
        return scaled_lr#Bohr