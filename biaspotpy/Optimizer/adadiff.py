import numpy as np
import copy



    
class AdaDiff:
    def __init__(self, **config):
        #AdaDiff
        #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
        self.adam_count = 1
        self.DELTA = 0.03
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-12
        self.Initialization = True
        self.config = config
        self.hessian = None
        self.bias_hessian = None
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        print("AdaDiff")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
         
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2 + (1.0-self.beta_v) * (B_g[i] - pre_B_g[i]) ** 2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + self.Epsilon)/(1 - self.beta_v**adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+self.Epsilon))
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
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