import numpy as np
import copy

"""
FSB, Bofill
 J. Chem. Phys. 1999, 111, 10806
MSP
 Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
"""

class ModelHessianUpdate:
    def __init__(self):
        self.Initialization = True
        self.denom_threshold = 1e-7
        return
    def flowchart_hessian_update(self, hess, displacement, delta_grad, method):
        #Theor Chem Acc  (2016) 135:84 
        z = delta_grad - np.dot(hess, delta_grad)
        
        zs_denominator = np.linalg.norm(displacement) * np.linalg.norm(z)
        if abs(zs_denominator) < self.denom_threshold:
            zs_denominator += self.denom_threshold
        zs = np.dot(z.T, displacement) / zs_denominator
        ys_denominator = np.linalg.norm(displacement) * np.linalg.norm(delta_grad)
        if abs(ys_denominator) < self.denom_threshold:
            ys_denominator += self.denom_threshold
        
        ys = np.dot(delta_grad.T, displacement) / ys_denominator
        if zs < -0.1:
            print("SR1")
            delta_hess = self.SR1_hessian_update(hess, displacement, delta_grad)
        elif ys > 0.1:
            print("BFGS")
            delta_hess = self.BFGS_hessian_update(hess, displacement, delta_grad)
        else:
            print("PSB")
            delta_hess = self.FSB_hessian_update(hess, displacement, delta_grad)
        
        return delta_hess
    
    
    
    def BFGS_hessian_update(self, hess, displacement, delta_grad):
        demon_1 = np.dot(displacement.T, delta_grad)
        if np.abs(demon_1) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        demon_2 = np.dot(np.dot(displacement.T, hess), displacement)
        if np.abs(demon_2) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        
        delta_hess = (np.dot(delta_grad, delta_grad.T)) / demon_1 - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ demon_2)
        return delta_hess
    
    def FSB_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        delta_hess_BFGS_denominator_1 = np.dot(displacement.T, delta_grad)
        if np.abs(delta_hess_BFGS_denominator_1) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS_denominator_2 = np.dot(np.dot(displacement.T, hess), displacement)
        if np.abs(delta_hess_BFGS_denominator_2) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        
        delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / delta_hess_BFGS_denominator_1) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ delta_hess_BFGS_denominator_2)
        Bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        if np.abs(Bofill_const_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / Bofill_const_denominator
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        return delta_hess

    def Bofill_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
            
        delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
         
        block_1 = delta_grad - 1 * np.dot(hess, displacement) 
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        
        block_2 = np.dot(displacement, displacement.T) / block_2_denominator
        
        delta_hess_PSB = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / np.dot(displacement.T, displacement) 
            
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        return delta_hess  
    
    def SR1_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        return delta_hess_SR1
    
    def PSB_hessian_update(self, hess, displacement, delta_grad):
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        block_1 = delta_grad - 1 * np.dot(hess, displacement) 
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        block_2 = np.dot(displacement, displacement.T) / block_2_denominator ** 2
        
        delta_hess_P = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator  #PSB
        return delta_hess_P
    
    def MSP_hessian_update(self, hess, displacement, delta_grad):
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_MS_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_MS_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        
        delta_hess_MS = np.dot(A, A.T) / delta_hess_MS_denominator #SR1
        block_1 = delta_grad - 1 * np.dot(hess, displacement) 
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
        
        block_2 = np.dot(displacement, displacement.T) / block_2_denominator ** 2
        
        delta_hess_P = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator  #PSB
        A_norm = np.linalg.norm(A)
        displacement_norm = np.linalg.norm(displacement)
        if displacement_norm * A_norm < self.denom_threshold:
            print("denominator is too small!!! Stop performing Hessian update.")
            return np.zeros((len(delta_grad), len(delta_grad)))
            
        
        phi = np.sin(np.arccos(np.dot(displacement.T, A) / (A_norm * displacement_norm))) ** 2
        delta_hess = phi*delta_hess_P + (1 - phi)*delta_hess_MS
        
        return delta_hess