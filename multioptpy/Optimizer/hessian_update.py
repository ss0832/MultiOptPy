import numpy as np
from scipy.linalg import null_space
"""
FSB, Bofill
 J. Chem. Phys. 1999, 111, 10806
MSP
 Journal of Molecular Structure: THEOCHEM 2002, 591 (1-3), 35-57.
 
CFD (compact finite difference) Hessian approximation approach
 J. Chem. Theory Comput. 2013, 9, 54-64
 J. Chem. Phys. 2010, 133, 074101
 
"""

class ModelHessianUpdate:
    def __init__(self):
        self.Initialization = True
        self.denom_threshold = 1e-10
        return
    
    def _auto_scale_hessian(self, hess, displacement, delta_grad):
        """
         Heuristic to scale matrix at first iteration.
         Described in Nocedal and Wright "Numerical Optimization"
         p.143 formula (6.20).
        """
        if self.Initialization and np.all(hess - np.eye(len(delta_grad))) < 1e-8:
            print("Auto scaling Hessian")
            s_norm_2 = np.dot(displacement.T, displacement)
            y_norm_2 = np.dot(delta_grad.T, delta_grad)
            ys = np.abs(np.dot(delta_grad.T, displacement))
            if np.abs(s_norm_2) < 1e-10 or np.abs(y_norm_2) < 1e-10 or np.abs(ys) < 1e-10:
                print("Norms too small, skip scaling.")
                return hess
            scale_factor = y_norm_2 / ys
            print("Scale factor:", scale_factor)
            hess = hess * scale_factor
            self.Initialization = False
        
        return hess
    
    def flowchart_hessian_update(self, hess, displacement, delta_grad, method):
        print("Flowchart Hessian Update")
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
        print("BFGS Hessian Update")
        demon_1 = np.dot(displacement.T, delta_grad)
        demon_2 = np.dot(np.dot(displacement.T, hess), displacement)
        delta_hess = np.zeros((len(delta_grad), len(delta_grad)))
        if np.abs(demon_1) >= self.denom_threshold:
            term1 = np.dot(delta_grad, delta_grad.T) / demon_1
        else:
            print("demon_1 is too small, term1 set to zero.")
            term1 = np.zeros((len(delta_grad), len(delta_grad)))
        if np.abs(demon_2) >= self.denom_threshold:
            term2 = np.dot(np.dot(np.dot(hess, displacement), displacement.T), hess.T) / demon_2
        else:
            print("demon_2 is too small, term2 set to zero.")
            term2 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess = term1 - term2
        return delta_hess
    
    def FSB_hessian_update(self, hess, displacement, delta_grad):
        print("FSB Hessian Update")
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("SR1 denominator is too small, term set to zero.")
            delta_hess_SR1 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS_denominator_1 = np.dot(displacement.T, delta_grad)
        if np.abs(delta_hess_BFGS_denominator_1) >= self.denom_threshold:
            term1 = np.dot(delta_grad, delta_grad.T) / delta_hess_BFGS_denominator_1
        else:
            print("BFGS denominator 1 is too small, term set to zero.")
            term1 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS_denominator_2 = np.dot(np.dot(displacement.T, hess), displacement)
        if np.abs(delta_hess_BFGS_denominator_2) >= self.denom_threshold:
            term2 = np.dot(np.dot(np.dot(hess, displacement), displacement.T), hess.T) / delta_hess_BFGS_denominator_2
        else:
            print("BFGS denominator 2 is too small, term set to zero.")
            term2 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS = term1 - term2
        Bofill_const_numerator = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement)
        Bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        Bofill_const = Bofill_const_numerator / Bofill_const_denominator
        
        if np.abs(Bofill_const_denominator) >= self.denom_threshold:
            Bofill_const = Bofill_const_numerator / Bofill_const_denominator
        else: # For minimization, BFGS is chosen to guarantee positive definiteness
            Bofill_const = 0.0
            print("Bofill_const denominator is too small, set to zero.")
            
        print("Bofill_const:", Bofill_const)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        return delta_hess

    
    def CFD_FSB_hessian_update(self, hess, displacement, delta_grad):
        print("CFD FSB Hessian Update")
        #J. Chem. Phys. 1999, 111, 10806
        A = 2.0 * (delta_grad - np.dot(hess, displacement))
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("CFD SR1 denominator is too small, term set to zero.")
            delta_hess_SR1 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS_denominator_1 = np.dot(displacement.T, delta_grad)
        if np.abs(delta_hess_BFGS_denominator_1) >= self.denom_threshold:
            term1 = np.dot(delta_grad, delta_grad.T) / delta_hess_BFGS_denominator_1
        else:
            print("BFGS denominator 1 is too small, term set to zero.")
            term1 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS_denominator_2 = np.dot(np.dot(displacement.T, hess), displacement)
        if np.abs(delta_hess_BFGS_denominator_2) >= self.denom_threshold:
            term2 = np.dot(np.dot(np.dot(hess, displacement), displacement.T), hess.T) / delta_hess_BFGS_denominator_2
        else:
            print("BFGS denominator 2 is too small, term set to zero.")
            term2 = np.zeros((len(delta_grad), len(delta_grad)))
        delta_hess_BFGS = term1 - term2
        Bofill_const_numerator = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement)
        Bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        Bofill_const = Bofill_const_numerator / Bofill_const_denominator
        
        if np.abs(Bofill_const_denominator) >= self.denom_threshold:
            Bofill_const = Bofill_const_numerator / Bofill_const_denominator
        else: # For minimization, BFGS is chosen to guarantee positive definiteness
            Bofill_const = 0.0
            print("Bofill_const denominator is too small, set to zero.")
            
        print("Bofill_const:", Bofill_const)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        return delta_hess


    def Bofill_hessian_update(self, hess, displacement, delta_grad):
        print("Bofill Hessian Update")
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("SR1 denominator is too small, term set to zero.")
            delta_hess_SR1 = np.zeros((len(delta_grad), len(delta_grad)))
        block_1 = 2.0 * (delta_grad - 1 * np.dot(hess, displacement))
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) >= self.denom_threshold:
            block_2 = np.dot(displacement, displacement.T) / block_2_denominator
        else:
            print("block_2 denominator is too small, term set to zero.")
            block_2 = np.zeros((len(delta_grad), len(delta_grad)))
        if np.abs(block_2_denominator) >= self.denom_threshold:
            delta_hess_PSB = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator
        else:
            print("PSB denominator is too small, term set to zero.")
            delta_hess_PSB = np.zeros((len(delta_grad), len(delta_grad)))
        bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        bofill_const_numerator = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement)
        Bofill_const = bofill_const_numerator / bofill_const_denominator
        if np.abs(bofill_const_denominator) >= self.denom_threshold:
            pass
        else: 
            Bofill_const = 0.0
            print("Bofill_const denominator is too small, set to zero.")
        print("Bofill_const:", Bofill_const)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        return delta_hess


    def CFD_Bofill_hessian_update(self, hess, displacement, delta_grad):
        # SR1 and Bofill constant with CFD
        print("CFD Bofill Hessian Update (CFD apply to SR1 update method and Bofill constant)")
        #J. Chem. Phys. 1999, 111, 10806
        A = 2.0 * (delta_grad - np.dot(hess, displacement))
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("CFD SR1 denominator is too small, term set to zero.")
            delta_hess_SR1 = np.zeros((len(delta_grad), len(delta_grad)))
        block_1 = (delta_grad - 1 * np.dot(hess, displacement))
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) >= self.denom_threshold:
            block_2 = np.dot(displacement, displacement.T) / block_2_denominator
        else:
            print("block_2 denominator is too small, term set to zero.")
            block_2 = np.zeros((len(delta_grad), len(delta_grad)))
        if np.abs(block_2_denominator) >= self.denom_threshold:
            delta_hess_PSB = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator
        else:
            print("CFD PSB denominator is too small, term set to zero.")
            delta_hess_PSB = np.zeros((len(delta_grad), len(delta_grad)))
        bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        bofill_const_numerator = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement)
        Bofill_const = bofill_const_numerator / bofill_const_denominator
        if np.abs(bofill_const_denominator) >= self.denom_threshold:
            pass
        else: 
            Bofill_const = 0.0
            print("Bofill_const denominator is too small, set to zero.")
        print("Bofill_const:", Bofill_const)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        return delta_hess


    def pCFD_Bofill_hessian_update(self, hess, displacement, delta_grad):
        print("Perturbed CFD Bofill Hessian Update (CFD apply to SR1 update method and Bofill constant)")
        #J. Chem. Phys. 1999, 111, 10806
        A = 2.0 * (delta_grad - np.dot(hess, displacement))
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("CFD SR1 denominator is too small, term set to zero.")
            delta_hess_SR1 = np.zeros((len(delta_grad), len(delta_grad)))
        block_1 = (delta_grad - 1 * np.dot(hess, displacement))
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) >= self.denom_threshold:
            block_2 = np.dot(displacement, displacement.T) / block_2_denominator
        else:
            print("block_2 denominator is too small, term set to zero.")
            block_2 = np.zeros((len(delta_grad), len(delta_grad)))
        if np.abs(block_2_denominator) >= self.denom_threshold:
            delta_hess_PSB = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator
        else:
            print("CFD PSB denominator is too small, term set to zero.")
            delta_hess_PSB = np.zeros((len(delta_grad), len(delta_grad)))
        bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        bofill_const_numerator = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement)
        Bofill_const = bofill_const_numerator / bofill_const_denominator
        if np.abs(bofill_const_denominator) >= self.denom_threshold:
            pass
        else: 
            Bofill_const = 0.0
            print("Bofill_const denominator is too small, set to zero.")
        print("Bofill_const:", Bofill_const)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        print("Calculating perturbation term...")
        tmp_perturb_term_matrix = np.zeros_like(delta_hess)
        ortho_vecs = null_space(displacement.T)
        for ortho_vec_i in ortho_vecs.T:
            for ortho_vec_j in ortho_vecs.T:
                tmp_perturb_term_matrix += (np.dot(ortho_vec_j.T, np.dot(delta_hess, ortho_vec_i))) * (np.dot(ortho_vec_i, ortho_vec_j.T) + np.dot(ortho_vec_j, ortho_vec_i.T))
        delta_hess = delta_hess + tmp_perturb_term_matrix
        return delta_hess

    
    def SR1_hessian_update(self, hess, displacement, delta_grad):
        print("SR1 Hessian Update")
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("SR1 denominator is too small, term set to zero.")
            delta_hess_SR1 = np.zeros((len(delta_grad), len(delta_grad)))
        return delta_hess_SR1
    
    def PSB_hessian_update(self, hess, displacement, delta_grad):
        print("PSB Hessian Update")
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        block_1 = delta_grad - 1 * np.dot(hess, displacement)
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) >= self.denom_threshold:
            block_2 = np.dot(displacement, displacement.T) / block_2_denominator ** 2
            delta_hess_P = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator
        else:
            print("PSB denominator is too small, term set to zero.")
            delta_hess_P = np.zeros((len(delta_grad), len(delta_grad)))
        return delta_hess_P
    
    def MSP_hessian_update(self, hess, displacement, delta_grad):
        print("MSP Hessian Update")
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_MS_denominator = np.dot(A.T, displacement)
        if np.abs(delta_hess_MS_denominator) >= self.denom_threshold:
            delta_hess_MS = np.dot(A, A.T) / delta_hess_MS_denominator #SR1
        else:
            print("MS denominator is too small, term set to zero.")
            delta_hess_MS = np.zeros((len(delta_grad), len(delta_grad)))
        block_1 = delta_grad - 1 * np.dot(hess, displacement)
        block_2_denominator = np.dot(displacement.T, displacement)
        if np.abs(block_2_denominator) >= self.denom_threshold:
            block_2 = np.dot(displacement, displacement.T) / block_2_denominator ** 2
            delta_hess_P = -1 * np.dot(block_1.T, displacement) * block_2 + (np.dot(block_1, displacement.T) + np.dot(displacement, block_1.T)) / block_2_denominator
        else:
            print("PSB denominator is too small, term set to zero.")
            delta_hess_P = np.zeros((len(delta_grad), len(delta_grad)))
        A_norm = np.linalg.norm(A)
        displacement_norm = np.linalg.norm(displacement)
        if displacement_norm * A_norm >= self.denom_threshold:
            phi = np.sin(np.arccos(np.dot(displacement.T, A) / (A_norm * displacement_norm))) ** 2
        else:
            print("phi denominator is too small, set phi=0.")
            phi = 0.0
        delta_hess = phi*delta_hess_P + (1 - phi)*delta_hess_MS
        return delta_hess