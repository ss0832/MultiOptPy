import numpy as np
import copy



class AdaMax:
    def __init__(self, **config):
        #arXiv:1412.6980v9
        #not worked well
        self.adam_count = 1
        self.DELTA = 0.005
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.adamax_u = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        print("AdaMax")
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
        new_adamax_u = max(self.beta_v*self.adamax_u, np.linalg.norm(B_g))
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append((self.DELTA / (self.beta_m ** adam_count + self.Epsilon)) * (adam_m[i] / (new_adamax_u + self.Epsilon)))
            
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