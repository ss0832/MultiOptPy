import numpy as np
import copy

 
class Prodigy:    
    def __init__(self, **config):
        #Prodigy
        #arXiv:2306.06101v1
        self.adam_count = 1
        self.d = 0.03
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.1
        self.Epsilon = 1e-12
        self.Initialization = True
        self.config = config
        self.hessian = None
        self.bias_hessian = None
   
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g=[], pre_g=[]):
        print("Prodigy")
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0        
            self.adam_s = geom_num_list * 0.0
            self.adam_r = 0.0
            new_d = self.d
            self.initial_geom_num_list = geom_num_list
            self.Initialization = False
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        new_adam_s = self.adam_s*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]*self.d))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i]*self.d)**2)
            new_adam_s[i] = np.sqrt(self.beta_v)*self.adam_s[i] + (1.0 - np.sqrt(self.beta_v))*self.DELTA*B_g[i]*self.d**2  
        
        new_adam_r = np.sqrt(self.beta_v)*self.adam_r + (1.0 - np.sqrt(self.beta_v))*(np.dot(B_g.reshape(1,len(B_g)), (self.initial_geom_num_list - geom_num_list).reshape(len(B_g),1)))*self.DELTA*self.d**2
        
        new_d = float(max((new_adam_r / np.linalg.norm(new_adam_s ,ord=1)), self.d))
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_d*new_adam_m[i]/(np.sqrt(new_adam_v[i])+self.Epsilon*self.d))
        
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_r = new_adam_r
        self.d = new_d
        self.adam_count += 1
        return move_vector
    
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