from .linesearch import LineSearch
from .hessian_update import ModelHessianUpdate
import numpy as np


"""
RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
 Theor chim Acta (1992) 82: 189-205
"""




class RationalFunctionOptimization:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        self.Initialization = True
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = self.config["saddle_order"] #
        self.iter = 0 #
        self.beta = 0.5
        return
    
    def calc_center(self, geomerty, element_list=[]):#geomerty:Bohr
        center = np.array([0.0, 0.0, 0.0], dtype="float64")
        for i in range(len(geomerty)):
            
            center += geomerty[i] 
        center /= float(len(geomerty))
        
        return center
            
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
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def hessian_update(self, displacement, delta_grad):
        if "MSP" in self.config["method"]:
            print("RFO_MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(self.hessian, displacement, delta_grad)
        elif "BFGS" in self.config["method"]:
            print("RFO_BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(self.hessian, displacement, delta_grad)
        elif "FSB" in self.config["method"]:
            print("RFO_FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "Bofill" in self.config["method"]:
            print("RFO_Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(self.hessian, displacement, delta_grad)
        else:
            raise "method error"
        return delta_hess
    
    
    def normal_v2(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv2")
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        DELTA_for_QNM = self.DELTA
        
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            delta_hess = self.hessian_update(displacement, delta_grad)
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[self.saddle_order])

        hess_eigenvalue, hess_eigenvector = np.linalg.eigh(new_hess)
        hess_eigenvector = hess_eigenvector.T
        move_vector = np.zeros((len(geom_num_list), 1))
        DELTA_for_QNM = self.DELTA
        
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        hess_eigenvector = hess_eigenvector.astype(np.float64)
        
        for i in range(len(hess_eigenvalue)):
            
            tmp_vector = np.array([hess_eigenvector[i].T], dtype="float64")
            move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list), 1)) * tmp_vector.T / (hess_eigenvalue[i] - lambda_for_calc + 1e-8)
            
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector
        self.hessian += delta_hess 
        self.iter += 1
     
        return move_vector#Bohr.
        
    def normal_v3(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv3")
        #ref.:Culot, P., Dive, G., Nguyen, V.H. et al. A quasi-Newton algorithm for first-order saddle-point location. Theoret. Chim. Acta 82, 189–205 (1992). https://doi.org/10.1007/BF01113492
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        DELTA_for_QNM = self.DELTA
        

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            delta_hess = self.hessian_update(displacement, delta_grad)
            delta_hess = self.project_out_hess_tr_and_rot_for_coord(delta_hess, geom_num_list.reshape(int(len(geom_num_list)/3), 3))
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eigh(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[max(self.saddle_order-1, 0)])

        hess_eigenvalue, hess_eigenvector = np.linalg.eigh(new_hess)
        hess_eigenvector = hess_eigenvector.T
        hess_eigenval_indices = np.argsort(hess_eigenvalue)
        
        move_vector = np.zeros((len(geom_num_list), 1))
        DELTA_for_QNM = self.DELTA
        
        for i in range(len(hess_eigenvalue)):
            tmp_vector = np.array([hess_eigenvector[hess_eigenval_indices[i]].T], dtype="float64")
            if i < self.saddle_order:
                move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list), 1)) * tmp_vector.T / (hess_eigenvalue[hess_eigenval_indices[i]] + lambda_for_calc + 1e-8) 
            else:
                move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list), 1)) * tmp_vector.T / (hess_eigenvalue[hess_eigenval_indices[i]] - lambda_for_calc + 1e-8)
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector
        self.hessian += delta_hess 
        self.iter += 1
     
        return move_vector#Bohr.
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv1")
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        DELTA_for_QNM = self.DELTA
        

        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            delta_hess = self.hessian_update(displacement, delta_grad)
           # delta_hess = self.project_out_hess_tr_and_rot_for_coord(delta_hess, geom_num_list.reshape(int(len(geom_num_list)/3), 3))
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list))), B_g.reshape(len(geom_num_list), 1))
            
        #move_vector = DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)))), B_g.reshape(len(geom_num_list), 1))
        
        DELTA_for_QNM = self.DELTA
        
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        
            
        move_vector = move_vector
        self.hessian += delta_hess 
        self.iter += 1
        return move_vector
        
    def moment(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("moment mode")
        print("saddle order:", self.saddle_order)

        if self.Initialization:
            self.momentum_disp = geom_num_list * 0.0
            self.momentum_grad = geom_num_list * 0.0
            self.Initialization = False
            return -1*self.DELTA*B_g
            
        
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

        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        

        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)))), B_g.reshape(len(geom_num_list), 1)))
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list))), B_g.reshape(len(geom_num_list), 1))
    
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM,"\n")
        self.hessian += delta_hess
        self.momentum_disp = new_momentum_disp
        self.momentum_grad = new_momentum_grad
        self.iter += 1
        return move_vector#Bohr.   

    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        if "mrfo" in self.config["method"].lower():
            move_vector = self.moment(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo2" in self.config["method"].lower():
            move_vector = self.normal_v2(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo3" in self.config["method"].lower():
            move_vector = self.normal_v3(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        else:
            move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector
 
