from .linesearch import LineSearch
from .hessian_update import ModelHessianUpdate
import numpy as np

class Newton:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        self.Initialization = True
        self.linesearchflag = False
        
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = 0 #
        self.iter = 0 #
        self.beta = 0.5
        return
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def hessian_update(self, displacement, delta_grad):
        if "MSP" in self.config["method"]:
            print("MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(self.hessian, displacement, delta_grad)
        elif "BFGS" in self.config["method"]:
            print("BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(self.hessian, displacement, delta_grad)
        elif "FSB" in self.config["method"]:
            print("FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "Bofill" in self.config["method"]:
            print("Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(self.hessian, displacement, delta_grad)
        else:
            raise "method error"
        return delta_hess
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("normal mode")
        if self.linesearchflag:
            print("linesearch mode")
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        
        delta_hess = self.hessian_update(displacement, delta_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
            
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        if self.iter > 1 and self.linesearchflag:
            LS = LineSearch(self.prev_move_vector, move_vector, B_g, pre_B_g, B_e, pre_B_e)
            new_move_vector, optimal_step_flag = LS.linesearch()
        else:
            new_move_vector = move_vector
            optimal_step_flag = True
        
        print("step size: ",DELTA_for_QNM,"\n")


        if self.iter > 0 and self.linesearchflag:
            move_vector = new_move_vector.reshape(len(geom_num_list), 3)
            if optimal_step_flag or self.iter == 1:
                self.prev_move_vector = move_vector    
                
        if not self.linesearchflag or np.linalg.norm(move_vector) > 1e-4:
            self.hessian += delta_hess       
            
        self.iter += 1
        return move_vector#Bohr
  
    # arXiv:2307.13744v1
    def moment(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("moment mode")
        if self.Initialization:
            self.momentum_disp = geom_num_list * 0.0
            self.momentum_grad = geom_num_list * 0.0
            self.Initialization = False
            return -1*self.DELTA*B_g
        
        if self.iter == 1:
            self.momentum_disp = geom_num_list - pre_geom
            self.momentum_grad = B_g - pre_B_g

        
        delta_grad = (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        

        new_momentum_disp = self.beta * self.momentum_disp + (1.0 - self.beta) * geom_num_list
        new_momentum_grad = self.beta * self.momentum_grad + (1.0 - self.beta) * B_g
        
        delta_momentum_disp = (new_momentum_disp - self.momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - self.momentum_grad).reshape(len(geom_num_list)*3, 1)
        

        delta_hess = self.hessian_update(delta_momentum_disp, delta_momentum_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)

        
        print("step size: ",DELTA_for_QNM,"\n")
        self.hessian += delta_hess
        self.iter += 1
        self.momentum_disp = new_momentum_disp
        self.momentum_grad = new_momentum_grad
        return move_vector 
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        if self.config["method"] == "mBFGS" or self.config["method"] == "mFSB" or self.config["method"] == "mMSP" or self.config["method"] == "mBofill":
            move_vector = self.moment(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        else:
            move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector