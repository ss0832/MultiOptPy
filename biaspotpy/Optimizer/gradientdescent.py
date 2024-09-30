import numpy as np
import copy



class GradientDescent:
    def __init__(self, **config):
        #Pseudo-IRC
        self.DELTA = 1.0
        self.Initialization = True
        self.config = config
        
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("SD")
        if self.Initialization:
            self.Initialization = False
            
        move_vector = self.DELTA * B_g
        
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return