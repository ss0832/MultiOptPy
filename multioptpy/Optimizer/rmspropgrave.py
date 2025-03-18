import numpy as np
import copy

         
class RMSpropGrave:
    def __init__(self, **config):
        #arXiv:https://arxiv.org/abs/1308.0850v5
        self.RMSpropGrave_count = 1
        self.DELTA = 0.75
        self.beta_m = 0.95
        self.beta_v = 0.95
        self.Epsilon = 1e-10
        self.eta = 0.0001
        self.nue = 0.9
        self.Initialization = True
        self.config = config
        
        self.hessian = None
        self.bias_hessian = None
   
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        print("RMSpropGrave")
        if self.Initialization:
            self.RMSpropGrave_m = geom_num_list * 0.0
            self.RMSpropGrave_v = geom_num_list * 0.0
            self.prev_move_vector = geom_num_list * 0.0
            self.Initialization = False
        
        RMSpropGrave_count = self.RMSpropGrave_count
        RMSpropGrave_m = self.RMSpropGrave_m
        RMSpropGrave_v = self.RMSpropGrave_v
        new_RMSpropGrave_m = RMSpropGrave_m*0.0
        new_RMSpropGrave_v = RMSpropGrave_v*0.0
        
        for i in range(len(geom_num_list)):
            new_RMSpropGrave_m[i] = copy.copy(self.beta_m*RMSpropGrave_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_RMSpropGrave_v[i] = copy.copy(self.beta_v*RMSpropGrave_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
    
        move_vector = []
        
        for i in range(len(geom_num_list)):
            tmp = self.nue * self.prev_move_vector[i] + B_g[i] * self.eta / np.sqrt(np.abs(new_RMSpropGrave_m[i] -1 * new_RMSpropGrave_v[i] ** 2 + self.Epsilon))
            move_vector.append(self.DELTA*tmp)
                
        self.RMSpropGrave_m = new_RMSpropGrave_m
        self.RMSpropGrave_v = new_RMSpropGrave_v
        self.prev_move_vector = move_vector
        self.RMSpropGrave_count += 1

        return move_vector#Bohr.
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian