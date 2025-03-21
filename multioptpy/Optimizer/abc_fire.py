import numpy as np
import copy


 
class ABC_FIRE:
    def __init__(self, **config):#MD-like optimization method. 
        #ABC_FIRE
        #Computational Materials Science Volume 218, 5 February 2023, 111978
        self.iter = 0
        self.sub_iter = 0
        self.N_acc = 5
        self.f_inc = 1.10
        self.f_acc = 0.99
        self.f_dec = 0.50
        self.dt_max = 0.8
        self.alpha_start = 0.1
        
        self.display_flag = True
        self.config = config
        self.Initialization = True
        self.hessian = None
        self.bias_hessian = None
   
        return
    
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        
        

        if self.Initialization:
            self.dt = 0.1
            self.alpha = self.alpha_start
            self.n_reset = 0
            self.pre_velocity = geom_num_list * 0.0
            self.Initialization = False
            
        
        
        velocity = (1.0 / (1.0 - (1.0 - self.alpha) ** (self.sub_iter) + 1e-10)) * (1.0 - self.alpha) * self.pre_velocity + self.alpha * (np.linalg.norm(self.pre_velocity, ord=2)/np.linalg.norm(B_g, ord=2)) * B_g
        
        if self.iter > 0 and np.dot(self.pre_velocity.reshape(1, len(geom_num_list)), B_g.reshape(len(geom_num_list), 1)) > 0:
            if self.n_reset > self.N_acc:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.alpha = self.alpha * self.f_acc
            self.n_reset += 1
        else:
           
            velocity *= 0.0
            self.alpha = self.alpha_start
            self.dt *= self.f_dec
            self.n_reset = 0
        
        velocity += self.dt*B_g
        
        move_vector = copy.copy(self.dt * velocity)
        
        if self.display_flag:
            print("FIRE")
            print("dt, alpha, n_reset :", self.dt, self.alpha, self.n_reset)
        
        self.pre_velocity = velocity
        self.iter += 1
        self.sub_iter += 1
        if self.iter > 0:
            if np.dot(velocity.reshape(1, len(geom_num_list)), B_g.reshape(len(geom_num_list), 1)) < 0:
                self.sub_iter = 0
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