from .linesearch import LineSearch
import numpy as np
import copy


class ConjgateGradient:
    def __init__(self, **config):
        #required variables in config(dict): method 
        self.config = config
        self.epsilon = 1e-8
        self.DELTA = 1.0
        self.Initialization = True
        self.hessian = None
        self.bias_hessian = None
        return
    
    def calc_alpha(self):
        alpha = np.dot(self.gradient.reshape(1, len(self.geom_num_list)), (self.d_vector).reshape(len(self.geom_num_list), 1)) / (np.dot(self.d_vector.reshape(1, len(self.geom_num_list)), self.d_vector.reshape(len(self.geom_num_list), 1)) + self.epsilon)
        return alpha
    
    def calc_beta(self):#based on polak-ribiere
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list), 1)) / (np.dot(self.prev_gradient.reshape(1, len(self.geom_num_list)), self.prev_gradient.reshape(len(self.geom_num_list), 1)) ** 2 + self.epsilon)#based on polak-ribiere
        return beta
    
    def calc_beta_PR(self):#polak-ribiere
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list), 1)) / (np.dot(self.prev_gradient.reshape(1, len(self.geom_num_list)), self.prev_gradient.reshape(len(self.geom_num_list), 1)) + self.epsilon) #polak-ribiere
        return beta
    
    def calc_beta_FR(self):#fletcher-reeeves
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)), self.gradient.reshape(len(self.geom_num_list), 1)) / (np.dot(self.prev_gradient.reshape(1, len(self.geom_num_list)), self.prev_gradient.reshape(len(self.geom_num_list), 1)) + self.epsilon)
        return beta
    
    def calc_beta_HS(self):#Hestenes-stiefel
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list), 1)) / (np.dot(self.d_vector.reshape(1, len(self.geom_num_list)), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list), 1)) + self.epsilon)
        return beta
    
    def calc_beta_DY(self):#Dai-Yuan
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)), self.gradient.reshape(len(self.geom_num_list), 1)) / (np.dot(self.d_vector.reshape(1, len(self.geom_num_list)), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list), 1)) + self.epsilon)
        return beta
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        #cg method
        self.geom_num_list = np.array(geom_num_list)
        self.gradient = np.array(B_g)
        self.prev_gradient = np.array(pre_B_g)
        
        if self.Initialization:
            self.d_vector = geom_num_list * 0.0
            self.Initialization = False
            return self.DELTA*B_g
        
        alpha = self.calc_alpha()
        
        move_vector = self.DELTA * alpha * self.d_vector
        if self.config["method"].lower() == "cg_pr":
            beta = self.calc_beta_PR()
        elif self.config["method"].lower() == "cg_fr":
            beta = self.calc_beta_FR()
        elif self.config["method"].lower() == "cg_hs":
            beta = self.calc_beta_HS()
        elif self.config["method"].lower() == "cg_dy":
            beta = self.calc_beta_DY()
        else: 
            beta = self.calc_beta()
        
        self.d_vector = copy.copy(-1 * B_g + abs(beta) * self.d_vector)
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