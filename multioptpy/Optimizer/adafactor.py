import numpy as np
import copy



class Adafactor:
    def __init__(self, **config):
        #Adafactor
        #arXiv:1804.04235v1
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.06
        self.Epsilon_1 = 1e-08
        self.Epsilon_2 = self.DELTA
        self.Initialization = True
        self.config = config
        self.hessian = None
        self.bias_hessian = None
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        print("Adafactor")
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.adam_u = geom_num_list * 0.0
            
            self.Initialization = False

        beta = 1 - self.adam_count ** (-0.8)
        rho = min(0.01, 1/np.sqrt(self.adam_count))
        alpha = max(np.sqrt(np.square(geom_num_list).mean()),  self.Epsilon_2) * rho
        new_adam_m = self.adam_m
        new_adam_v = self.adam_v*0.0
        new_adam_u = self.adam_u*0.0
        new_adam_u_hat = self.adam_u*0.0
        for i in range(len(geom_num_list)):
            new_adam_v[i] = copy.copy(beta*self.adam_v[i] + (1.0-beta)*((B_g[i])**2 + np.array([1]) * self.Epsilon_1))
            new_adam_u[i] = copy.copy(B_g[i]/np.sqrt(new_adam_v[i]))
            
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_u_hat[i] = copy.copy(new_adam_u[i] / max(1, np.sqrt(np.square(new_adam_u).mean())))
        
        
        for i in range(len(geom_num_list)):
            move_vector.append(alpha*new_adam_u_hat[i])
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_u = new_adam_u
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