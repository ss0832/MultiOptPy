from .linesearch import LineSearch
from .hessian_update import ModelHessianUpdate
import numpy as np

class Newton:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        self.Initialization = True
        self.linesearchflag = False
        self.optimal_step_flag = False
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = 0 #
        self.iter = 0 #
        self.beta = 0.5
        self.hessian = None
        self.bias_hessian = None
   
        return
            
    def project_out_hess_tr_and_rot_for_coord(self, hessian, geomerty):#do not consider atomic mass
        natoms = len(geomerty)
       
        geomerty -= self.calc_center(geomerty)
        
    
        tr_x = (np.tile(np.array([1, 0, 0]), natoms)).reshape(-1, 3)
        tr_y = (np.tile(np.array([0, 1, 0]), natoms)).reshape(-1, 3)
        tr_z = (np.tile(np.array([0, 0, 1]), natoms)).reshape(-1, 3)

        rot_x = np.cross(geomerty, tr_x).flatten()
        rot_y = np.cross(geomerty, tr_y).flatten() 
        rot_z = np.cross(geomerty, tr_z).flatten()
        tr_x = tr_x.flatten()
        tr_y = tr_y.flatten()
        tr_z = tr_z.flatten()

        TR_vectors = np.vstack([tr_x, tr_y, tr_z, rot_x, rot_y, rot_z])
        
        Q, R = np.linalg.qr(TR_vectors.T)
        keep_indices = ~np.isclose(np.diag(R), 0, atol=1e-6, rtol=0)
        TR_vectors = Q.T[keep_indices]
        n_tr = len(TR_vectors)

        P = np.identity(natoms * 3)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)

        hess_proj = np.dot(np.dot(P.T, hessian), P)

        return hess_proj    
    
    def calc_center(self, geomerty, element_list=[]):#geomerty:Bohr
        center = np.array([0.0, 0.0, 0.0], dtype="float64")
        for i in range(len(geomerty)):
            
            center += geomerty[i] 
        center /= float(len(geomerty))
        
        return center
    
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
    
    def hessian_update(self, displacement, delta_grad):
        if "msp" in self.config["method"].lower():
            print("MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(self.hessian, displacement, delta_grad)
        elif "bfgs" in self.config["method"].lower():
            print("BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(self.hessian, displacement, delta_grad)
        elif "fsb" in self.config["method"].lower():
            print("FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "bofill" in self.config["method"].lower():
            print("Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(self.hessian, displacement, delta_grad)
        else:
            raise "method error"
        return delta_hess
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        
        if self.linesearchflag:
            print("linesearch mode")
        else:
            print("normal mode")
        if self.Initialization:
            self.Initialization = False
            return self.DELTA*B_g
        
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        
        
        delta_hess = self.hessian_update(displacement, delta_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
            
        DELTA_for_QNM = self.DELTA
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list), 1))
        
        if self.iter > 1 and self.linesearchflag:
            if self.FC_COUNT != -1:
                tmp_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list.reshape(-1, 3))  
            else:
                tmp_hess = None
            if self.optimal_step_flag or self.iter == 2:
                self.LS = LineSearch(self.prev_move_vector, move_vector, B_g, pre_B_g, B_e, pre_B_e, tmp_hess)
            
            new_move_vector, self.optimal_step_flag = self.LS.linesearch(self.prev_move_vector, move_vector, B_g, pre_B_g, B_e, pre_B_e, tmp_hess)
        else:
            new_move_vector = move_vector
            self.optimal_step_flag = True
        
        print("step size: ",DELTA_for_QNM,"\n")


        if self.iter > 0 and self.linesearchflag:
            move_vector = -new_move_vector
            if self.optimal_step_flag or self.iter == 1:
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
            return self.DELTA*B_g
        
        if self.iter == 1:
            self.momentum_disp = geom_num_list - pre_geom
            self.momentum_grad = B_g - pre_B_g

        
        delta_grad = (B_g - pre_B_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        
        

        new_momentum_disp = self.beta * self.momentum_disp + (1.0 - self.beta) * geom_num_list
        new_momentum_grad = self.beta * self.momentum_grad + (1.0 - self.beta) * B_g
        
        delta_momentum_disp = (new_momentum_disp - self.momentum_disp).reshape(len(geom_num_list), 1)
        delta_momentum_grad = (new_momentum_grad - self.momentum_grad).reshape(len(geom_num_list), 1)
        

        delta_hess = self.hessian_update(delta_momentum_disp, delta_momentum_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list), 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list), 1))

        
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