import numpy as np
import copy



class GradientDescent:
    def __init__(self, **config):
        #Pseudo-IRC
        self.DELTA = 1.0
        self.Initialization = True
        self.config = config
        self.hessian = None
        self.bias_hessian = None
   
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
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
    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian
    

class MassWeightedGradientDescent:
    def __init__(self, **config):
        #For (meta-)IRC (Euler method)
        self.DELTA = 1.0
        self.Initialization = True
        self.config = config
        self.element_list = None
        self.atomic_mass = None
        
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("MWSD")
        if self.Initialization:
            self.elem_mass_list = []
            for elem in self.element_list:
                self.elem_mass_list.append([self.atomic_mass(elem)])
                self.elem_mass_list.append([self.atomic_mass(elem)])
                self.elem_mass_list.append([self.atomic_mass(elem)])
            self.elem_mass_list = np.array(self.elem_mass_list, dtype="float64")
            self.Initialization = False
            
        move_vector = self.DELTA * B_g / self.elem_mass_list
        
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