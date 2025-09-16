import numpy as np
import copy


class Adadelta:
    def __init__(self, **config):
        #Adadelta
        #arXiv:1212.5701v1
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.03
        self.Epsilon = 1e-06
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0
        self.RMS_FORCE_THRESHOLD = 1e+10
        self.Initialization = True
        self.config = config
        self.hessian = None
        self.bias_hessian = None
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):#delta is not required. This method tends to converge local minima. This class doesnt work well.
        print("Adadelta")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            
            self.Initialization = False
        rho = 0.9
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0

        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(rho * self.adam_m[i] + (1.0 - rho)*(B_g[i]) ** 2)
        move_vector = []
        
        for i in range(len(geom_num_list)):
            if self.adam_count > 1:
                move_vector.append(B_g[i] * (np.sqrt(np.square(self.adam_v).mean()) + self.Epsilon)/(np.sqrt(np.square(new_adam_m).mean()) + self.Epsilon))
            else:
                move_vector.append(B_g[i])
        if abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) > self.RMS_FORCE_THRESHOLD:
            move_vector = B_g

        for i in range(len(geom_num_list)):
            new_adam_v[i] = copy.copy(rho * self.adam_v[i] + (1.0 - rho) * (move_vector[i]) ** 2)
                
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