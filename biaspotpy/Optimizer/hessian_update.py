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
        return
    def BFGS_hessian_update(self, hess, displacement, delta_grad):
        
        A = delta_grad - np.dot(hess, displacement)

        delta_hess = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        return delta_hess
    
    def FSB_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
        delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        return delta_hess

    def Bofill_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement)
         
        block_1 = delta_grad - 1 * np.dot(hess, displacement) 
        block_2 = np.dot(displacement, displacement.T) / (np.dot(displacement.T, displacement)) ** 2
        
        delta_hess_PSB = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / np.dot(displacement.T, displacement) 
            
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        return delta_hess  
    
    def MSP_hessian_update(self, hess, displacement, delta_grad):
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_MS = np.dot(A, A.T) / np.dot(A.T, displacement) #SR1
        block_1 = delta_grad - 1 * np.dot(hess, displacement) 
        block_2 = np.dot(displacement, displacement.T) / (np.dot(displacement.T, displacement)) ** 2
        
        delta_hess_P = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / np.dot(displacement.T, displacement)  #PSB
        A_norm = np.linalg.norm(A)
        displacement_norm = np.linalg.norm(displacement) + 1e-10
        
        phi = np.sin(np.arccos(np.dot(displacement.T, A) / (A_norm * displacement_norm))) ** 2
        delta_hess = phi*delta_hess_P + (1 - phi)*delta_hess_MS
        
        return delta_hess