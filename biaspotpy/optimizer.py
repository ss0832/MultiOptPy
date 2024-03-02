import numpy as np
import copy

from param import UnitValueLib
from calc_tools import Calculationtools

"""
RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
 Theor chim Acta (1992) 82: 189-205
FSB, Bofill
 J. Chem. Phys. 1999, 111, 10806
MSP
 Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
"""

class Opt_calc_tmps:
    def __init__(self, adam_m, adam_v, adam_count, eve_d_tilde=0.0):
        self.adam_m = adam_m
        self.adam_v = adam_v
        self.adam_count = 1 + adam_count
        self.eve_d_tilde = eve_d_tilde
            
class Model_hess_tmp:
    def __init__(self, model_hess, momentum_disp=0, momentum_grad=0):
        self.model_hess = model_hess
        self.momentum_disp = momentum_disp
        self.momentum_grad = momentum_grad


class LineSearch:
    def __init__(self, prev_move_vector, move_vector, gradient, prev_gradient, energy, prev_energy,  hessian=None):
        
        self.move_vector = move_vector
        self.prev_move_vector = prev_move_vector
        self.gradient = gradient
        self.prev_gradient = prev_gradient
        self.energy = energy
        self.prev_energy = prev_energy
        self.hessian = hessian
        self.convergence_criterion = 0.2
        self.order = 6.0
        

    def linesearch(self):
        self.prev_gradient = self.prev_gradient.reshape(len(self.prev_gradient)*3, 1)
        self.gradient = self.prev_gradient.reshape(len(self.gradient)*3, 1)
        self.prev_move_vector = self.prev_move_vector.reshape(len(self.prev_move_vector)*3, 1)
        
        #self.gradient = self.gradient/np.linalg.norm(self.gradient)
        #self.prev_move_vector = self.prev_move_vector/np.linalg.norm(self.prev_move_vector)
        
        cos = np.sum(self.gradient*self.prev_move_vector)/(np.linalg.norm(self.gradient)*np.linalg.norm(self.prev_move_vector)+1e-8)
        print("orthogonality", cos)
        if abs(cos) < self.convergence_criterion:
            new_move_vector = self.move_vector
            print("optimal step is found.")
            optimal_step_flag = True
        else:
            if self.prev_energy < self.energy:
                new_move_vector = abs(cos) ** self.order * self.prev_move_vector# / np.linalg.norm(self.prev_move_vector)
            
            else:
                new_move_vector = -1 * abs(cos) ** self.order * self.prev_move_vector# / np.linalg.norm(self.prev_move_vector)
            
            print("linesearching...")
            optimal_step_flag = False
            
        return new_move_vector, optimal_step_flag



class CalculateMoveVector:
    def __init__(self, DELTA, Opt_params, Model_hess, BPA_hessian, trust_radii, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0):
        self.Opt_params = Opt_params 
        self.DELTA = DELTA
        self.Model_hess = Model_hess
        self.temperature = temperature
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.FC_COUNT = FC_COUNT
        self.MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.RMS_FORCE_SWITCHING_THRESHOLD = 0.0008
        self.BPA_hessian = BPA_hessian
        self.trust_radii = trust_radii
        self.saddle_order = saddle_order
        self.iter = 0
        self.element_list = element_list
        self.trust_radii_update = "legacy"
        
        
    def update_trust_radii(self, trust_radii, B_e, pre_B_e, pre_B_g, pre_move_vector):
    
        if self.trust_radii_update == "trust":
            Sc = 2.0
            Ce = (np.dot(pre_B_g.reshape(1, len(self.geom_num_list)*3), pre_move_vector.reshape(len(self.geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot(pre_move_vector.reshape(1, len(self.geom_num_list)*3), self.Model_hess.model_hess), pre_move_vector.reshape(len(self.geom_num_list)*3, 1)))
            r = (B_e - pre_B_e) / Ce
            
            if r < 0.25:
                trust_radii /= 2*Sc
            
            elif r > 0.25 and (trust_radii - np.linalg.norm(pre_move_vector)) < 1e-3:
                trust_radii *= Sc ** 0.5
            else:
                pass
        elif self.trust_radii_update == "legacy":
            if pre_B_e >= B_e:
                trust_radii *= 3.0
            else:
                trust_radii *= 0.1
        else:
            pass
                                   
        return np.clip(trust_radii, 0.01, 1.0)

    def BFGS_hessian_update(self, hess, displacement, delta_grad):
        
        A = delta_grad - np.dot(hess, displacement)

        delta_hess = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        return delta_hess
    
    def FSB_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
        delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        return delta_hess

    def Bofill_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
        delta_hess_PSB = -1 * np.dot(A.T, displacement) * np.dot(displacement, displacement.T) / np.dot(displacement.T, displacement) ** 2 - (np.dot(A, displacement.T) + np.dot(displacement, A.T)) / np.dot(displacement.T, displacement)
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        return delta_hess  
    
    def MSP_hessian_update(self, hess, displacement, delta_grad):
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_MS = np.dot(A, A.T) / np.dot(A.T, displacement) #SR1
        delta_hess_P = -1 * np.dot(A.T, displacement) * np.dot(displacement, displacement.T) / np.dot(displacement.T, displacement) ** 2 - (np.dot(A, displacement.T) + np.dot(displacement, A.T)) / np.dot(displacement.T, displacement) #PSB
        A_norm = np.linalg.norm(A) + 1e-8
        displacement_norm = np.linalg.norm(displacement) + 1e-8
        
        phi = np.sin(np.arccos(np.dot(displacement.T, A) / (A_norm * displacement_norm))) ** 2
        delta_hess = phi*delta_hess_P + (1 - phi)*delta_hess_MS
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        
        return delta_hess


    def RFOv2_MSP_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv2_MSP_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.MSP_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[self.saddle_order])

        hess_eigenvalue, hess_eigenvector = np.linalg.eig(new_hess)
        hess_eigenvector = hess_eigenvector.T
        move_vector = np.zeros((len(geom_num_list)*3, 1))
        DELTA_for_QNM = self.DELTA
        
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        hess_eigenvector = hess_eigenvector.astype(np.float64)
        
        
        
        for i in range(len(hess_eigenvalue)):
            
            tmp_vector = np.array([hess_eigenvector[i].T], dtype="float64")
            move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list)*3, 1)) * tmp_vector.T / (hess_eigenvalue[i] - lambda_for_calc + 1e-8)
            
     
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector.reshape(len(geom_num_list), 3)
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def RFOv2_FSB_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv2_FSB_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.FSB_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[self.saddle_order])

        hess_eigenvalue, hess_eigenvector = np.linalg.eig(new_hess)
        hess_eigenvector = hess_eigenvector.T
        move_vector = np.zeros((len(geom_num_list)*3, 1))
        DELTA_for_QNM = self.DELTA
        
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        hess_eigenvector = hess_eigenvector.astype(np.float64)
        
        
        
        for i in range(len(hess_eigenvalue)):
            
            tmp_vector = np.array([hess_eigenvector[i].T], dtype="float64")
            move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list)*3, 1)) * tmp_vector.T / (hess_eigenvalue[i] - lambda_for_calc + 1e-8)
            
     
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector.reshape(len(geom_num_list), 3)
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def RFOv2_BFGS_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv2_BFGS_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.BFGS_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[self.saddle_order])

        hess_eigenvalue, hess_eigenvector = np.linalg.eig(new_hess)
        hess_eigenvector = hess_eigenvector.T
        move_vector = np.zeros((len(geom_num_list)*3, 1))
        DELTA_for_QNM = self.DELTA
        
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        hess_eigenvector = hess_eigenvector.astype(np.float64)
        
        
        
        for i in range(len(hess_eigenvalue)):
            
            tmp_vector = np.array([hess_eigenvector[i].T], dtype="float64")
            move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list)*3, 1)) * tmp_vector.T / (hess_eigenvalue[i] - lambda_for_calc + 1e-8)
            
     
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector.reshape(len(geom_num_list), 3)
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def RFOv2_Bofill_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv2_Bofill_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.Bofill_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[self.saddle_order])

        hess_eigenvalue, hess_eigenvector = np.linalg.eig(new_hess)
        hess_eigenvector = hess_eigenvector.T
        move_vector = np.zeros((len(geom_num_list)*3, 1))
        DELTA_for_QNM = self.DELTA
        
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        hess_eigenvector = hess_eigenvector.astype(np.float64)
        
        
        
        for i in range(len(hess_eigenvalue)):
            
            tmp_vector = np.array([hess_eigenvector[i].T], dtype="float64")
            move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list)*3, 1)) * tmp_vector.T / (hess_eigenvalue[i] - lambda_for_calc + 1e-8)
            
     
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector.reshape(len(geom_num_list), 3)
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def RFO_MSP_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO_MSP_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.Bofill_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
            
        #move_vector = DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        DELTA_for_QNM = self.DELTA
        
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        
            

        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    def MSP_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("MSP_quasi_newton_method")
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        
        
        delta_hess_BFGS = self.Bofill_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)

        
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
  
    def RFO_Bofill_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO_Bofill_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.Bofill_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        

        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)   
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        
        
        
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        
            

        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    def Bofill_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("Bofill_quasi_newton_method")
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        
        
        delta_hess_BFGS = self.Bofill_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)

        
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
              
    def RFO_BFGS_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO_BFGS_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.BFGS_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
            
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        
    
        
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        
            

        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    def BFGS_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("BFGS_quasi_newton_method")
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        
        
        delta_hess_BFGS = self.BFGS_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)

        
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    

    def RFO_FSB_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO_FSB_quasi_newton_method")
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        
            

        delta_hess = self.FSB_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
            
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        
        
            
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    def FSB_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("FSB_quasi_newton_method")
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        

        delta_hess = self.FSB_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def FSB_linesearch_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("FSB_quasi_newton_method (linesearch)")
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        prev_move_vector = geom_num_list - pre_geom
        displacement = prev_move_vector.reshape(len(geom_num_list)*3, 1)
        

        delta_hess = self.FSB_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        LS = LineSearch(prev_move_vector, move_vector, B_g, pre_B_g, B_e, pre_B_e)
        new_move_vector, optimal_step_flag = LS.linesearch()

        
        print("step size: ",DELTA_for_QNM,"\n")
        if np.linalg.norm(move_vector) > 1e-5:
            self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - new_move_vector.reshape(len(geom_num_list), 3)) * self.bohr2angstroms #ang.
        
        if not optimal_step_flag:
            move_vector = new_move_vector.reshape(len(geom_num_list), 3)
        
        
        return move_vector

    def BFGS_linesearch_quasi_newton_method(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("BFGS_quasi_newton_method (linesearch)")
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        prev_move_vector = geom_num_list - pre_geom
        displacement = prev_move_vector.reshape(len(geom_num_list)*3, 1)
        

        delta_hess = self.BFGS_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        LS = LineSearch(prev_move_vector, move_vector, B_g, pre_B_g, B_e, pre_B_e)
        new_move_vector, optimal_step_flag = LS.linesearch()

        
        print("step size: ",DELTA_for_QNM,"\n")
        
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian)
        self.new_geometry = (geom_num_list - new_move_vector.reshape(len(geom_num_list), 3)) * self.bohr2angstroms #ang.
        if not optimal_step_flag:
            move_vector = new_move_vector.reshape(len(geom_num_list), 3)
        
        return move_vector
    
    # arXiv:2307.13744v1
    def momentum_based_BFGS(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("momentum_based_BFGS")
        adam_count = self.Opt_params.adam_count
        if adam_count == 1:
            momentum_disp = pre_geom
            momentum_grad = pre_g
        else:

            momentum_disp = self.Model_hess.momentum_disp
            momentum_grad = self.Model_hess.momentum_grad
            
        beta = 0.50
        
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        

        new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
        new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
        
        delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
        

        delta_hess_BFGS = self.BFGS_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)

        
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian, new_momentum_disp, new_momentum_grad)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector 
    
    def momentum_based_FSB(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("momentum_based_FSB")
        adam_count = self.Opt_params.adam_count
        if adam_count == 1:
            momentum_disp = pre_geom
            momentum_grad = pre_g
        else:
            momentum_disp = self.Model_hess.momentum_disp
            momentum_grad = self.Model_hess.momentum_grad
        beta = 0.50
        
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        

        new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
        new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
        
        delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
        

        delta_hess = self.FSB_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian, new_momentum_disp, new_momentum_grad)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector 
            
        
    def RFO_momentum_based_BFGS(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO_momentum_based_BFGS")
        print("saddle order:", self.saddle_order)
        adam_count = self.Opt_params.adam_count
        if adam_count == 1:
            momentum_disp = pre_geom
            momentum_grad = pre_g
        else:
            momentum_disp = self.Model_hess.momentum_disp
            momentum_grad = self.Model_hess.momentum_grad
        beta = 0.5
        
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        

        new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
        new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
        
        delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
        
    
        
        
        delta_hess_BFGS = self.BFGS_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
        DELTA_for_QNM = self.DELTA

        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        

        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
    
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian, new_momentum_disp, new_momentum_grad)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector 
    
    def RFO_momentum_based_FSB(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO_momentum_based_FSB")
        print("saddle order:", self.saddle_order)
        adam_count = self.Opt_params.adam_count
        if adam_count == 1:
            momentum_disp = pre_geom
            momentum_grad = pre_g
        else:
            momentum_disp = self.Model_hess.momentum_disp
            momentum_grad = self.Model_hess.momentum_grad
        beta = 0.5
        print("beta :", beta)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        

        new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
        new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
        
        delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
        
        delta_hess = self.FSB_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
        else:
            new_hess = self.Model_hess.model_hess + self.BPA_hessian

        DELTA_for_QNM = self.DELTA
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        

        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM,"\n")
        self.Model_hess = Model_hess_tmp(new_hess - self.BPA_hessian, new_momentum_disp, new_momentum_grad)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector 
    
    def conjugate_gradient_descent(self, geom_num_list, pre_move_vector, B_g, pre_B_g):
        #cg method
        
        alpha = np.dot(B_g.reshape(1, len(geom_num_list)*3), (self.Opt_params.adam_v).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), self.Opt_params.adam_v.reshape(len(geom_num_list)*3, 1))
        
        move_vector = self.DELTA * alpha * self.Opt_params.adam_v
        
        beta = np.dot(B_g.reshape(1, len(geom_num_list)*3), (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)) / np.dot(pre_B_g.reshape(1, len(geom_num_list)*3), pre_B_g.reshape(len(geom_num_list)*3, 1)) ** 2#based on polak-ribiere
        
        self.Opt_params.adam_v = copy.copy(-1 * B_g + abs(beta) * self.Opt_params.adam_v)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def conjugate_gradient_descent_PR(self, geom_num_list, pre_move_vector, B_g, pre_B_g):
        #cg method
        
        alpha = np.dot(B_g.reshape(1, len(geom_num_list)*3), (self.Opt_params.adam_v).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), self.Opt_params.adam_v.reshape(len(geom_num_list)*3, 1))
        
        move_vector = self.DELTA * alpha * self.Opt_params.adam_v
        
        beta = np.dot(B_g.reshape(1, len(geom_num_list)*3), (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)) / np.dot(pre_B_g.reshape(1, len(geom_num_list)*3), pre_B_g.reshape(len(geom_num_list)*3, 1)) #polak-ribiere
        
        self.Opt_params.adam_v = copy.copy(-1 * B_g + abs(beta) * self.Opt_params.adam_v)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector

    def conjugate_gradient_descent_FR(self, geom_num_list, pre_move_vector, B_g, pre_B_g):
        #cg method
        
        alpha = np.dot(B_g.reshape(1, len(geom_num_list)*3), (self.Opt_params.adam_v).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), self.Opt_params.adam_v.reshape(len(geom_num_list)*3, 1))
        
        move_vector = self.DELTA * alpha * self.Opt_params.adam_v
        
        beta = np.dot(B_g.reshape(1, len(geom_num_list)*3), B_g.reshape(len(geom_num_list)*3, 1)) / np.dot(pre_B_g.reshape(1, len(geom_num_list)*3), pre_B_g.reshape(len(geom_num_list)*3, 1)) #fletcher-reeeves
        
        self.Opt_params.adam_v = copy.copy(-1 * B_g + abs(beta) * self.Opt_params.adam_v)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    def conjugate_gradient_descent_HS(self, geom_num_list, pre_move_vector, B_g, pre_B_g):
        #cg method
        
        alpha = np.dot(B_g.reshape(1, len(geom_num_list)*3), (self.Opt_params.adam_v).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), self.Opt_params.adam_v.reshape(len(geom_num_list)*3, 1))
        
        move_vector = self.DELTA * alpha * self.Opt_params.adam_v
        
        beta = np.dot(B_g.reshape(1, len(geom_num_list)*3), (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1))#Hestenes-stiefel
        
        self.Opt_params.adam_v = copy.copy(-1 * B_g + abs(beta) * self.Opt_params.adam_v)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    
    def conjugate_gradient_descent_DY(self, geom_num_list, pre_move_vector, B_g, pre_B_g):
        #cg method
        
        alpha = np.dot(B_g.reshape(1, len(geom_num_list)*3), (self.Opt_params.adam_v).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), self.Opt_params.adam_v.reshape(len(geom_num_list)*3, 1))
        
        move_vector = self.DELTA * alpha * self.Opt_params.adam_v
        
        beta = np.dot(B_g.reshape(1, len(geom_num_list)*3), B_g.reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)) #Dai-Yuan
        
        self.Opt_params.adam_v = copy.copy(-1 * B_g + abs(beta) * self.Opt_params.adam_v)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector 
    
    #Engineering Applications of Artificial Intelligence 2023, 119, 105755. https://doi.org/10.1016/j.engappai.2022.105755
    def Adaderivative(self, geom_num_list, B_g, pre_B_g):
        print("Adaderivative")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy((beta_m * adam_m[i] + (1.0-beta_m) * (B_g[i]) ))
            new_adam_v[i] = copy.copy((beta_v * adam_v[i] + (1.0-beta_v) * (B_g[i] - pre_B_g[i]) ** 2))
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+Epsilon))
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    #arXiv:1908.00700v2
    def SAdam(self, geom_num_list, B_g):
        print("SAdam")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
    
        move_vector = []
        
        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/(np.log(1.0 + np.exp(np.sqrt(new_adam_v[i])))+Epsilon))
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    #arXiv:1908.00700v2
    def SAMSGrad(self, geom_num_list, B_g):
        print("SAMSGrad")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
            for j in range(len(new_adam_v[i])):
                
                new_adam_v[i][j] = max(new_adam_v[i][j], adam_v[i][j])
                    
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/(np.log(1.0 + np.exp(np.sqrt(new_adam_v[i])))+Epsilon))
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    #arXiv:1810.06801v4    
    def QHAdam(self, geom_num_list, B_g):
        print("QHAdam")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        nu_m = 1.0
        nu_v = 1.0

        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
    
                    
        move_vector = []

        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*((1.0 - nu_m) * B_g[i] + new_adam_m[i] * nu_m)/np.sqrt(((1.0 - nu_v) * B_g[i] ** 2 + new_adam_v[i] * nu_v)+Epsilon))
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    #https://doi.org/10.1145/3441501.3441507
    def YOGI(self, geom_num_list, B_g):
        print("YOGI")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(adam_v[i] - (1.0 - beta_v) * np.sign(adam_v[i] - B_g[i] ** 2) * (B_g[i])**2)
    
                    
        move_vector = []

        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*(new_adam_m[i])/np.sqrt((new_adam_v[i])+Epsilon))
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    
    #arXiv:1412.6980v9
    def AdaMax(self, geom_num_list, B_g):#not worked well
        print("AdaMax")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        if adam_count == 1:
            adamax_u = 1e-8
            self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, adamax_u)
            
        else:
            adamax_u = self.Opt_params.eve_d_tilde#eve_d_tilde = adamax_u 
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
        new_adamax_u = max(beta_v*adamax_u, np.linalg.norm(B_g))
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append((self.DELTA / (beta_m ** adam_count)) * (adam_m[i] / new_adamax_u))
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, new_adamax_u)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    #https://cs229.stanford.edu/proj2015/054_report.pdf
    def NAdam(self, geom_num_list, B_g):
        print("NAdam")
        mu = 0.975
        nu = 0.999
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = []
        new_adam_v_hat = []
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(mu*adam_m[i] + (1.0 - mu)*(B_g[i]))
            new_adam_v[i] = copy.copy((nu*adam_v[i]) + (1.0 - nu)*(B_g[i]) ** 2)
            new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64") * ( mu / (1.0 - mu ** adam_count)) + np.array(B_g[i], dtype="float64") * ((1.0 - mu)/(1.0 - mu ** adam_count)))        
            new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64") * (nu / (1.0 - nu ** adam_count)))
        
        move_vector = []
        for i in range(len(geom_num_list)):
            move_vector.append( (self.DELTA*new_adam_m_hat[i]) / (np.sqrt(new_adam_v_hat[i] + Epsilon)))
            
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    
    #FIRE
    #Physical Review Letters, Vol. 97, 170201 (2006)
    def FIRE(self, geom_num_list, B_g):#MD-like optimization method. This method tends to converge local minima.
        print("FIRE")
        adam_count = self.Opt_params.adam_count
        N_acc = 5
        f_inc = 1.10
        f_acc = 0.99
        f_dec = 0.50
        dt_max = 0.8
        alpha_start = 0.1
        if adam_count == 1:
            self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, [0.1, alpha_start, 0])
            #valuable named 'eve_d_tilde' is parameters for FIRE.
            # [0]:dt [1]:alpha [2]:n_reset
        dt = self.Opt_params.eve_d_tilde[0]
        alpha = self.Opt_params.eve_d_tilde[1]
        n_reset = self.Opt_params.eve_d_tilde[2]
        
        pre_velocity = self.Opt_params.adam_v
        
        velocity = (1.0 - alpha) * pre_velocity + alpha * (np.linalg.norm(pre_velocity, ord=2)/np.linalg.norm(B_g, ord=2)) * B_g
        
        if adam_count > 1 and np.dot(pre_velocity.reshape(1, len(geom_num_list)*3), B_g.reshape(len(geom_num_list)*3, 1)) > 0:
            if n_reset > N_acc:
                dt = min(dt * f_inc, dt_max)
                alpha = alpha * f_acc
            n_reset += 1
        else:
            velocity *= 0.0
            alpha = alpha_start
            dt *= f_dec
            n_reset = 0
        
        velocity += dt*B_g
        
        move_vector = velocity * 0.0
        move_vector = copy.copy(dt * velocity)
        
        print("dt, alpha, n_reset :", dt, alpha, n_reset)
        self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, velocity, adam_count, [dt, alpha, n_reset])
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    #RAdam
    #arXiv:1908.03265v4
    def RADAM(self, geom_num_list, B_g):
        print("RADAM")
        beta_m = 0.9
        beta_v = 0.99
        rho_inf = 2.0 / (1.0- beta_v) - 1.0 
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = []
        new_adam_v_hat = []
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy((beta_v*adam_v[i]) + (1.0-beta_v)*(B_g[i] - new_adam_m[i])**2) + Epsilon
            new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64")/(1.0-beta_m**adam_count))        
            new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64")/(1.0-beta_v**adam_count))
        rho = rho_inf - (2.0*adam_count*beta_v**adam_count)/(1.0 -beta_v**adam_count)
                    
        move_vector = []
        if rho > 4.0:
            l_alpha = []
            for j in range(len(new_adam_v)):
                l_alpha.append(np.sqrt((abs(1.0 - beta_v**adam_count))/new_adam_v[j]))
            l_alpha = np.array(l_alpha, dtype="float64")
            r = np.sqrt(((rho-4.0)*(rho-2.0)*rho_inf)/((rho_inf-4.0)*(rho_inf-2.0)*rho))
            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*r*new_adam_m_hat[i]*l_alpha[i])
        else:
            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i])
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    #AdaBelief
    #ref. arXiv:2010.07468v5
    def AdaBelief(self, geom_num_list, B_g):
        print("AdaBelief")
        beta_m = 0.9
        beta_v = 0.99
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i]-new_adam_m[i])**2)
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+Epsilon))
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
        
    #AdaDiff
    #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
    def AdaDiff(self, geom_num_list, B_g, pre_B_g):
        print("AdaDiff")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2 + (1.0-beta_v) * (B_g[i] - pre_B_g[i]) ** 2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    #arXiv:1910.12249v1 
    def Adamod(self, geom_num_list, B_g):
        print("Adamod")
        
        beta_m = 0.9
        beta_v = 0.999
        beta_s = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count

        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        if adam_count == 1:
            adam_r = adam_m*0.0
            adam_s = adam_m*0.0
            self.Opt_params = Opt_calc_tmps(adam_m, adam_v, adam_count - 1, [adam_r, adam_s])
        else:
            adam_r = self.Opt_params.eve_d_tilde[0]
            adam_s = self.Opt_params.eve_d_tilde[1]
            
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        new_adam_s = adam_s*0.0
        new_adam_r = adam_s*0.0
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        new_adam_r_hat = adam_r*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
            
            
            
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            new_adam_s[i] =  self.DELTA / (np.sqrt(beta_v) + Epsilon)
            new_adam_r[i] = beta_s * adam_r[i] + (1 - beta_s) * new_adam_s[i]
        
        for i in range(len(geom_num_list)):
            for j in range(len(new_adam_r_hat[i])):
                new_adam_r_hat[i][j] = min(new_adam_r[i][j], new_adam_s[i][j])
        
        move_vector = []

        for i in range(len(geom_num_list)):
                move_vector.append(new_adam_r_hat[i]*new_adam_m_hat[i])
        
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, [new_adam_r, new_adam_s])
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    
    #EVE
    #ref.arXiv:1611.01505v3
    def EVE(self, geom_num_list, B_g, B_e, pre_B_e, pre_B_g):
        print("EVE")
        beta_m = 0.9
        beta_v = 0.999
        beta_d = 0.999
        c = 10
        Epsilon = 1e-08 
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        if adam_count > 1:
            eve_d_tilde = self.Opt_params.eve_d_tilde
        else:
            eve_d_tilde = 1.0
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i])/(1 - beta_v**adam_count))
            
        if adam_count > 1:
            eve_d = abs(B_e - pre_B_e)/ min(B_e, pre_B_e)
            eve_d_hat = np.clip(eve_d, 1/c , c)
            eve_d_tilde = beta_d*eve_d_tilde + (1.0 - beta_d)*eve_d_hat
            
        else:
            pass
        
        for i in range(len(geom_num_list)):
                move_vector.append((self.DELTA/eve_d_tilde)*new_adam_m_hat[i]/(np.sqrt(new_adam_v_hat[i])+Epsilon))
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, eve_d_tilde)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    #AdamW
    #arXiv:2302.06675v4
    def AdamW(self, geom_num_list, B_g):
        print("AdamW")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        AdamW_lambda = 0.001
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon) + AdamW_lambda * geom_num_list[i])
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    #Adam
    #arXiv:1412.6980
    def Adam(self, geom_num_list, B_g):
        print("Adam")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
        
    def third_order_momentum_Adam(self, geom_num_list, B_g):
        print("third_order_momentum_Adam")
        #self.Opt_params.eve_d_tilde is 3rd-order momentum
        beta_m = 0.9
        beta_v = 0.999
        beta_s = 0.9999999999
        
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        if adam_count > 1:                
            adam_s = self.Opt_params.eve_d_tilde
        else:
            adam_s = adam_m*0.0
            
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        new_adam_s = adam_s*0.0
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        new_adam_s_hat = adam_s*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
            new_adam_s[i] = copy.copy(beta_s*adam_s[i] + (1.0-beta_s)*(B_g[i])**3)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            new_adam_s_hat[i] = copy.copy((new_adam_s[i] + Epsilon)/(1 - beta_s**adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA * new_adam_m_hat[i] / np.abs(np.sqrt(new_adam_v_hat[i]+Epsilon) - ((new_adam_m_hat[i] * (new_adam_s_hat[i]) ** (1 / 3)) / (2.0 * np.sqrt(new_adam_v_hat[i]+Epsilon)) )) )
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, new_adam_s)
        return move_vector
                
    #Adafactor
    #arXiv:1804.04235v1
    def Adafactor(self, geom_num_list, B_g):
        print("Adafactor")
        Epsilon_1 = 1e-08
        Epsilon_2 = self.DELTA

        adam_count = self.Opt_params.adam_count
        beta = 1 - adam_count ** (-0.8)
        rho = min(0.01, 1/np.sqrt(adam_count))
        alpha = max(np.sqrt(np.square(geom_num_list).mean()),  Epsilon_2) * rho
        adam_v = self.Opt_params.adam_m
        adam_u = self.Opt_params.adam_v
        new_adam_v = adam_v*0.0
        new_adam_u = adam_u*0.0
        new_adam_v = adam_v*0.0
        new_adam_u = adam_u*0.0
        new_adam_u_hat = adam_u*0.0
        for i in range(len(geom_num_list)):
            new_adam_v[i] = copy.copy(beta*adam_v[i] + (1.0-beta)*((B_g[i])**2 + np.array([1,1,1]) * Epsilon_1))
            new_adam_u[i] = copy.copy(B_g[i]/np.sqrt(new_adam_v[i]))
            
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_u_hat[i] = copy.copy(new_adam_u[i] / max(1, np.sqrt(np.square(new_adam_u).mean())))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(alpha*new_adam_u_hat[i])
                
        self.Opt_params = Opt_calc_tmps(new_adam_v, new_adam_u, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    #Prodigy
    #arXiv:2306.06101v1
    def Prodigy(self, geom_num_list, B_g, initial_geom_num_list):
        print("Prodigy")
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        adam_count = self.Opt_params.adam_count

        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        if adam_count == 1:
            adam_r = 0.0
            adam_s = adam_m*0.0
            d = 1e-1
            new_d = d
            self.Opt_params = Opt_calc_tmps(adam_m, adam_v, adam_count - 1, [d, adam_r, adam_s])
        else:
            d = self.Opt_params.eve_d_tilde[0]
            adam_r = self.Opt_params.eve_d_tilde[1]
            adam_s = self.Opt_params.eve_d_tilde[2]
            
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        new_adam_s = adam_s*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]*d))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i]*d)**2)
            
            new_adam_s[i] = np.sqrt(beta_v)*adam_s[i] + (1.0 - np.sqrt(beta_v))*self.DELTA*B_g[i]*d**2  
        new_adam_r = np.sqrt(beta_v)*adam_r + (1.0 - np.sqrt(beta_v))*(np.dot(B_g.reshape(1,len(B_g)*3), (initial_geom_num_list - geom_num_list).reshape(len(B_g)*3,1)))*self.DELTA*d**2
        
        new_d = float(max((new_adam_r / np.linalg.norm(new_adam_s ,ord=1)), d))
        move_vector = []

        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_d*new_adam_m[i]/(np.sqrt(new_adam_v[i])+Epsilon*d))
        
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, [new_d, new_adam_r, new_adam_s])
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    
    #AdaBound
    #arXiv:1902.09843v1
    def Adabound(self, geom_num_list, B_g):
        print("AdaBound")
        adam_count = self.Opt_params.adam_count
        move_vector = []
        beta_m = 0.9
        beta_v = 0.999
        Epsilon = 1e-08
        if adam_count == 1:
            adam_m = self.Opt_params.adam_m
            adam_v = np.zeros((len(geom_num_list),3,3))
        else:
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        V = adam_m*0.0
        Eta = adam_m*0.0
        Eta_hat = adam_m*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(np.dot(np.array([B_g[i]]).T, np.array([B_g[i]]))))
            V[i] = copy.copy(np.diag(new_adam_v[i]))
            
            Eta_hat[i] = copy.copy(np.clip(self.DELTA/np.sqrt(V[i]), 0.1 - (0.1/(1.0 - beta_v) ** (adam_count + 1)) ,0.1 + (0.1/(1.0 - beta_v) ** adam_count) ))
            Eta[i] = copy.copy(Eta_hat[i]/np.sqrt(adam_count))
                
        for i in range(len(geom_num_list)):
            move_vector.append(Eta[i] * new_adam_m[i])
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector    
    
    #Adadelta
    #arXiv:1212.5701v1
    def Adadelta(self, geom_num_list, B_g):#delta is not required. This method tends to converge local minima.
        print("Adadelta")
        rho = 0.9
        adam_count = self.Opt_params.adam_count
        adam_m = self.Opt_params.adam_m
        adam_v = self.Opt_params.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        Epsilon = 1e-06
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(rho * adam_m[i] + (1.0 - rho)*(B_g[i]) ** 2)
        move_vector = []
        
        for i in range(len(geom_num_list)):
            if adam_count > 1:
                move_vector.append(B_g[i] * (np.sqrt(np.square(adam_v).mean()) + Epsilon)/(np.sqrt(np.square(new_adam_m).mean()) + Epsilon))
            else:
                move_vector.append(B_g[i])
        if abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) > self.RMS_FORCE_THRESHOLD:
            move_vector = B_g

        for i in range(len(geom_num_list)):
            new_adam_v[i] = copy.copy(rho * adam_v[i] + (1.0 - rho) * (move_vector[i]) ** 2)
                
        self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
        self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        return move_vector
    

    def Perturbation(self, move_vector):#This function is just for fun. Thus, it is no scientific basis.
        """Langevin equation"""
        Boltzmann_constant = 3.16681*10**(-6) # hartree/K
        damping_coefficient = 10.0

        temperature = self.temperature
        perturbation = self.DELTA * np.sqrt(2.0 * damping_coefficient * Boltzmann_constant * temperature) * np.random.normal(loc=0.0, scale=1.0, size=3*len(move_vector)).reshape(len(move_vector), 3)

        return perturbation
  
    def calc_move_vector(self, iter, geom_num_list, B_g, opt_method_list, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):#geom_num_list:Bohr
        self.iter = iter
        self.geom_num_list = geom_num_list
        move_vector_list = []
        #-------------------------------------------------------------
        #update trust radii
        #-------------------------------------------------------------
        if self.iter % self.FC_COUNT == 0 and self.FC_COUNT != -1:
            self.trust_radii = 0.01
        elif self.FC_COUNT == -1:
            self.trust_radii = 1.0
        else:
            self.trust_radii = self.update_trust_radii(self.trust_radii, B_e, pre_B_e, pre_B_g, pre_move_vector)
            
        #---------------------------------
        #calculate move vector
        #---------------------------------
        for opt_method in opt_method_list:
            # group of steepest descent
            if opt_method == "RADAM":
                tmp_move_vector = self.RADAM(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adamod":
                tmp_move_vector = self.Adamod(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)      
                          
            elif opt_method == "YOGI":
                tmp_move_vector = self.YOGI(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)            
                  
            elif opt_method == "SADAM":
                tmp_move_vector = self.SAdam(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "QHADAM":
                tmp_move_vector = self.QHAdam(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "SAMSGrad":
                tmp_move_vector = self.SAMSGrad(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)     
                               
            elif opt_method == "Adam":
                tmp_move_vector = self.Adam(geom_num_list, B_g) 
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adadelta":
                tmp_move_vector = self.Adadelta(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdamW":
                tmp_move_vector = self.AdamW(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaDiff":
                tmp_move_vector = self.AdaDiff(geom_num_list, B_g, pre_B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adafactor":
                tmp_move_vector = self.Adafactor(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaBelief":
                tmp_move_vector = self.AdaBelief(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adabound":
                tmp_move_vector = self.Adabound(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "EVE":
                tmp_move_vector = self.EVE(geom_num_list, B_g, B_e, pre_B_e, pre_B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Prodigy":
                tmp_move_vector = self.Prodigy(geom_num_list, B_g, initial_geom_num_list)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaMax":
                tmp_move_vector = self.AdaMax(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "NAdam":    
                tmp_move_vector = self.NAdam(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "FIRE":
                tmp_move_vector = self.FIRE(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
            
            elif opt_method == "third_order_momentum_Adam":
                tmp_move_vector = self.third_order_momentum_Adam(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adaderivative":
                if iter != 0:
                    tmp_move_vector = self.Adaderivative(geom_num_list, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)    
                
            elif opt_method == "CG":
                if iter != 0:
                    tmp_move_vector = self.conjugate_gradient_descent(geom_num_list, pre_move_vector, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    self.Opt_params.adam_v = copy.copy(-1 * B_g)
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)

            elif opt_method == "CG_PR":
                if iter != 0:
                    tmp_move_vector = self.conjugate_gradient_descent_PR(geom_num_list, pre_move_vector, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    self.Opt_params.adam_v = copy.copy(-1 * B_g)
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
  
            elif opt_method == "CG_FR":
                if iter != 0:
                    tmp_move_vector = self.conjugate_gradient_descent_FR(geom_num_list, pre_move_vector, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    self.Opt_params.adam_v = copy.copy(-1 * B_g)
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)

            elif opt_method == "CG_HS":
                if iter != 0:
                    tmp_move_vector = self.conjugate_gradient_descent_HS(geom_num_list, pre_move_vector, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    self.Opt_params.adam_v = copy.copy(-1 * B_g)
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)


            elif opt_method == "CG_DY":
                if iter != 0:
                    tmp_move_vector = self.conjugate_gradient_descent_DY(geom_num_list, pre_move_vector, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    self.Opt_params.adam_v = copy.copy(-1 * B_g)
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
             
            
            # group of quasi-Newton method
        
                    
            elif opt_method == "BFGS":
                if iter != 0:
                    tmp_move_vector = self.BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "BFGS_LS":
                if iter != 0:
                    tmp_move_vector = self.BFGS_linesearch_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)              
            elif opt_method == "RFO_BFGS":
                if iter != 0:
                    tmp_move_vector = self.RFO_BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "RFO2_BFGS":
                if iter != 0:
                    tmp_move_vector = self.RFOv2_BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)           
                    
            elif opt_method == "Bofill":
                if iter != 0:
                    tmp_move_vector = self.Bofill_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_Bofill":
                if iter != 0:
                    tmp_move_vector = self.RFO_Bofill_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)     

            elif opt_method == "RFO2_Bofill":
                if iter != 0:
                    tmp_move_vector = self.RFOv2_Bofill_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)     
                               
            elif opt_method == "MSP":
                if iter != 0:
                    tmp_move_vector = self.MSP_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_MSP":
                if iter != 0:
                    tmp_move_vector = self.RFO_MSP_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)     

            elif opt_method == "RFO2_MSP":
                if iter != 0:
                    tmp_move_vector = self.RFOv2_MSP_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "FSB":
                if iter != 0:
                    tmp_move_vector = self.FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            
            elif opt_method == "FSB_LS":
                if iter != 0:
                    tmp_move_vector = self.FSB_linesearch_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)        
                    
            elif opt_method == "RFO_FSB":
                if iter != 0:
                    tmp_move_vector = self.RFO_FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "RFO2_FSB":
                if iter != 0:
                    tmp_move_vector = self.RFOv2_FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "mBFGS":
                if iter != 0:
                    tmp_move_vector = self.momentum_based_BFGS(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mFSB":
                if iter != 0:
                    tmp_move_vector = self.momentum_based_FSB(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mBFGS":
                if iter != 0:
                    tmp_move_vector = self.RFO_momentum_based_BFGS(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mFSB":
                if iter != 0:
                    tmp_move_vector = self.RFO_momentum_based_FSB(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            else:
                print("optimization method that this program is not sppourted is selected... thus, default method is selected.")
                tmp_move_vector = self.AdaBelief(geom_num_list, B_g)
        #---------------------------------
        # switich step update method
        if len(move_vector_list) > 1:
            if abs(B_g.max()) < self.MAX_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) < self.RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method: ", opt_method_list[1])
            else:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method: ", opt_method_list[0])
        else:
            move_vector = copy.copy(move_vector_list[0])
        
        # add perturbation (toy function)
        perturbation = self.Perturbation(move_vector)
        
        move_vector += perturbation
        print("perturbation: ", np.linalg.norm(perturbation))

        #-------------------------------------------------------------
        #display trust radii
        #-------------------------------------------------------------
        if np.linalg.norm(move_vector) > self.trust_radii:
            move_vector = self.trust_radii * move_vector/np.linalg.norm(move_vector)
        print("trust radii: ", self.trust_radii)
        print("step  radii: ", np.linalg.norm(move_vector))
        #------------------------------------------------------------
        # diagonize hessian matrix and display eigenvalues
        #-----------------------------------------------------------
        hess_eigenvalue, _ = np.linalg.eig(self.Model_hess.model_hess + self.BPA_hessian)
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        print("NORMAL MODE EIGENVALUE:\n",np.sort(hess_eigenvalue),"\n")
        
        #---------------------------------
        if iter == 0:
            self.new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms #ang.
        
        return self.new_geometry, np.array(move_vector, dtype="float64"), self.Opt_params, self.Model_hess, self.trust_radii


