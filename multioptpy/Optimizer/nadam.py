import numpy as np
import copy



class NAdam:
    def __init__(self, **config):
        #https://cs229.stanford.edu/proj2015/054_report.pdf
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.mu = 0.975
        self.nu = 0.999
        self.config = config
        self.hessian = None
        self.bias_hessian = None
   
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        print("NAdam")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = []
        new_adam_v_hat = []
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.mu*adam_m[i] + (1.0 - self.mu)*(B_g[i]))
            new_adam_v[i] = copy.copy((self.nu*adam_v[i]) + (1.0 - self.nu)*(B_g[i]) ** 2)
            new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64") * ( self.mu / (1.0 - self.mu ** adam_count)) + np.array(B_g[i], dtype="float64") * ((1.0 - self.mu)/(1.0 - self.mu ** adam_count)))        
            new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64") * (self.nu / (1.0 - self.nu ** adam_count)))
        
        move_vector = []
        for i in range(len(geom_num_list)):
            move_vector.append( (self.DELTA*new_adam_m_hat[i]) / (np.sqrt(new_adam_v_hat[i] + self.Epsilon)))
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
    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian