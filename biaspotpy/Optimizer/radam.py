import numpy as np
import copy


    
class RADAM:
    def __init__(self, **config):
         #arXiv:1908.03265v4
        self.adam_count = 1
        self.DELTA = 0.03
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-12
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("RADAM")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        rho_inf = 2.0 / (1.0- self.beta_v) - 1.0 
         
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = []
        new_adam_v_hat = []
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy((self.beta_v*adam_v[i]) + (1.0-self.beta_v)*(B_g[i] - new_adam_m[i])**2) + self.Epsilon
            new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64")/(1.0-self.beta_m**adam_count))        
            new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64")/(1.0-self.beta_v**adam_count))
        rho = rho_inf - (2.0*adam_count*self.beta_v**adam_count)/(1.0 -self.beta_v**adam_count)
                    
        move_vector = []
        if rho > 4.0:
            l_alpha = []
            for j in range(len(new_adam_v)):
                l_alpha.append(np.sqrt((abs(1.0 - self.beta_v**adam_count))/new_adam_v[j]))
            l_alpha = np.array(l_alpha, dtype="float64")
            r = np.sqrt(((rho-4.0)*(rho-2.0)*rho_inf)/((rho_inf-4.0)*(rho_inf-2.0)*rho))
            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*r*new_adam_m_hat[i]*l_alpha[i])
        else:
            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i])
        
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