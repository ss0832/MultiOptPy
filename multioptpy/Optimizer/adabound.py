import numpy as np
import copy



   
class AdaBound:
    def __init__(self, **config):
        #AdaBound
        #arXiv:1902.09843v1
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.05
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
        self.hessian = None
        self.bias_hessian = None
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        print("AdaBound")
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = np.zeros((len(geom_num_list), 3, 3))
            
            self.Initialization = False
        
        move_vector = []
 
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        V = self.adam_m*0.0
        Eta = self.adam_m*0.0
        Eta_hat = self.adam_m*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(np.dot(np.array([B_g[i]]).T, np.array([B_g[i]]))))
            V[i] = copy.copy(np.diag(new_adam_v[i]))
            
            Eta_hat[i] = copy.copy(np.clip(self.DELTA/(np.sqrt(V[i]) + self.Epsilon), 0.1 - (0.1/((1.0 - self.beta_v) ** (self.adam_count + 1) + self.Epsilon)) ,0.1 + 0.1/((1.0 - self.beta_v) ** self.adam_count + self.Epsilon) ))
            Eta[i] = copy.copy(Eta_hat[i]/np.sqrt(self.adam_count))
                
        for i in range(len(geom_num_list)):
            move_vector.append(Eta[i] * new_adam_m[i])
            
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
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