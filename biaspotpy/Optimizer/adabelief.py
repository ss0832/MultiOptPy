import numpy as np
import copy

class Adabelief:
    def __init__(self, **config):
        #AdaBelief
        #ref. arXiv:2010.07468v5
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.99
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("AdaBelief")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i]-new_adam_m[i])**2)
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+self.Epsilon))
        
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return