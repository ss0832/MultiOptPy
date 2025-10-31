import numpy as np
from scipy.linalg import null_space

"""
References:

FSB, Bofill
 J. Chem. Phys. 1999, 111, 10806

MSP
 Journal of Molecular Structure: THEOCHEM 2002, 591 (1-3), 35-57.
 
CFD (compact finite difference) Hessian approximation approach
 J. Chem. Theory Comput. 2013, 9, 54-64
 J. Chem. Phys. 2010, 133, 074101

Double Damping (DD)
 arXiv:2006.08877v3 [cs.LG] 7 Jan 2021
 
"""

class ModelHessianUpdate:
    def __init__(self):
        self.Initialization = True
        self.denom_threshold = 1e-10
        # Default parameters for Double Damping
        self.dd_mu1 = 0.2
        self.dd_mu2 = 0.2
        return

    # -----------------------------------------------------------------
    # Private Helper Methods (Strictly based on original code)
    # -----------------------------------------------------------------

    def _calculate_bfgs_delta(self, hess, s, y):
        """
        Calculates the BFGS update terms exactly as written in the original functions.
        delta_B = (y * y.T / y.T*s) - (B*s*s.T*B.T / s.T*B*s)
        
        Note: This implementation strictly follows the original np.dot usage.
        If s and y are 1D arrays, np.dot(y, y.T) computes an inner product (scalar).
        """
        n = len(y)
        delta_hess = np.zeros((n, n))

        # Term 1
        demon_1 = np.dot(s.T, y)
        term1 = np.zeros((n, n))
        if np.abs(demon_1) >= self.denom_threshold:
            # Strictly using np.dot(y, y.T) as in the original code
            term1 = np.dot(y, y.T) / demon_1
        else:
            print("BFGS denominator 1 (y.T*s) is too small, term1 set to zero.")
            
        # Term 2
        demon_2 = np.dot(np.dot(s.T, hess), s)
        term2 = np.zeros((n, n))
        if np.abs(demon_2) >= self.denom_threshold:
            # Strictly using the original complex np.dot chain
            term2 = np.dot(np.dot(np.dot(hess, s), s.T), hess.T) / demon_2
        else:
            print("BFGS denominator 2 (s.T*B*s) is too small, term2 set to zero.")
            
        delta_hess = term1 - term2
        return delta_hess

    def _calculate_sr1_delta(self, A, s):
        """
        Calculates the SR1 update term exactly as written in the original functions.
        A = (y - B*s) or A = 2.0 * (y - B*s)
        delta_B = A*A.T / A.T*s
        
        Note: This implementation strictly follows the original np.dot usage.
        If A is a 1D array, np.dot(A, A.T) computes an inner product (scalar).
        """
        delta_hess_SR1 = np.zeros((len(s), len(s)))
        delta_hess_SR1_denominator = np.dot(A.T, s)
        
        if np.abs(delta_hess_SR1_denominator) >= self.denom_threshold:
            # Strictly using np.dot(A, A.T) as in the original code
            delta_hess_SR1 = np.dot(A, A.T) / delta_hess_SR1_denominator
        else:
            print("SR1 denominator (A.T*s) is too small, term set to zero.")
            
        return delta_hess_SR1

    def _calculate_psb_delta(self, hess, s, y):
        """
        Calculates the PSB update term exactly as written in the original functions.
        """
        n = len(y)
        delta_hess_P = np.zeros((n, n))
        block_1 = y - 1 * np.dot(hess, s)
        block_2_denominator = np.dot(s.T, s)
        
        if np.abs(block_2_denominator) >= self.denom_threshold:
            # Logic from original PSB_hessian_update
            block_2 = np.dot(s, s.T) / block_2_denominator ** 2
            delta_hess_P = -1 * np.dot(block_1.T, s) * block_2 + \
                           (np.dot(block_1, s.T) + np.dot(s, block_1.T)) / block_2_denominator
        else:
            print("PSB denominator (s.T*s) is too small, term set to zero.")
            
        return delta_hess_P

    def _calculate_bofill_const(self, A, s):
        """
        Calculates the Bofill constant (phi^2) exactly as written in the original functions.
        phi^2 = ( (A.T*s)*(A.T*s) ) / ( (A.T*A)*(s.T*s) )
        
        Note: This implementation strictly follows the original np.dot usage.
        If A and s are 1D arrays, this correctly computes (A.T@s)**2 / ((A.T@A)*(s.T@s)).
        """
        Bofill_const = 0.0
        
        # Original calculation (assuming 1D arrays):
        # Numerator: (A.T @ s) * (A.T @ s)
        # Denominator: (A.T @ A) * (s.T @ s)
        
        Bofill_const_numerator = np.dot(np.dot(np.dot(A.T, s), A.T), s)
        Bofill_const_denominator = np.dot(np.dot(np.dot(A.T, A), s.T), s)
        
        if np.abs(Bofill_const_denominator) >= self.denom_threshold:
            Bofill_const = Bofill_const_numerator / Bofill_const_denominator
        else: 
            Bofill_const = 0.0
            print("Bofill_const denominator is too small, set to zero.")
            
        print("Bofill_const:", Bofill_const)
        return Bofill_const

    # -----------------------------------------------------------------
    # Initialization / Scaling
    # -----------------------------------------------------------------
    
    def _auto_scale_hessian(self, hess, displacement, delta_grad):
        """
         Heuristic to scale matrix at first iteration.
         Described in Nocedal and Wright "Numerical Optimization"
         p.143 formula (6.20).
        """
        if self.Initialization and np.allclose(hess, np.eye(len(delta_grad)), atol=1e-8):
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

    # -----------------------------------------------------------------
    # Flowchart Method
    # -----------------------------------------------------------------

    def flowchart_hessian_update(self, hess, displacement, delta_grad, method):
        print("Flowchart Hessian Update")
        #Theor Chem Acc  (2016) 135:84 
        
        # Note: Strictly adhering to the original code: z = y - B*y
        # The paper (TCA 2016, eq 19) suggests z = y - B*s
        z = delta_grad - np.dot(hess, delta_grad)
        
        zs_denominator = np.linalg.norm(displacement) * np.linalg.norm(z)
        if abs(zs_denominator) < self.denom_threshold:
            zs_denominator += self.denom_threshold
        zs = np.dot(z.T, displacement) / zs_denominator
        
        ys_denominator = np.linalg.norm(displacement) * np.linalg.norm(delta_grad)
        if abs(ys_denominator) < self.denom_threshold:
            ys_denominator += self.denom_threshold
        
        ys = np.dot(delta_grad.T, displacement) / ys_denominator
        
        delta_hess = np.zeros_like(hess)
        if zs < -0.1:
            print("Flowchart -> SR1")
            delta_hess = self.SR1_hessian_update(hess, displacement, delta_grad)
        elif ys > 0.1:
            print("Flowchart -> BFGS")
            delta_hess = self.BFGS_hessian_update(hess, displacement, delta_grad)
        else:
            print("Flowchart -> FSB")
            # Note: The paper suggests PSB, but the original code used FSB.
            delta_hess = self.FSB_hessian_update(hess, displacement, delta_grad)
        
        return delta_hess

    # -----------------------------------------------------------------
    # Double Damping (DD) Method
    # -----------------------------------------------------------------
    
    def double_damping_step2_only(self, s, y, mu2=0.2):
        """
        Implements ONLY Step 2 of the Double Damping (DD) procedure .
        This step does NOT require the inverse Hessian H.
        It is equivalent to Powell's damping with B=I [cite: 365-367].
        
        Args:
            s: displacement vector
            y: delta_grad vector
            mu2: Damping parameter (e.g., self.dd_mu2)

        Returns:
            s_tilde: (s is returned unmodified in this version)
            y_tilde: Damped delta_grad vector
        """
        print("Applying Double Damping (Step 2 only, H-independent)")
        
        s_tilde = s  # Step 1 is skipped
        y_tilde = y
        
        # --- Step 2: Powell's damping with B=I  ---
        s_tilde_y = np.dot(s_tilde.T, y)
        s_tilde_s_tilde = np.dot(s_tilde.T, s_tilde)

        # Check if damping is needed (also ensures s_tilde_y > 0 if mu2 > 0)
        if s_tilde_y < mu2 * s_tilde_s_tilde:
            print(f"DD Step 2 active: s_tilde.T*y ({float(s_tilde_y):.4e}) < mu2*s_tilde.T*s_tilde ({float(mu2 * s_tilde_s_tilde):.4e})")
            denominator = s_tilde_s_tilde - s_tilde_y
            
            if np.abs(denominator) < self.denom_threshold:
                 theta2 = 0.1 # Fallback
                 print("Warning: DD Step 2 denominator near zero. Using default theta2=0.1.")
            else:
                theta2 = (1.0 - mu2) * s_tilde_s_tilde / denominator
            
            theta2 = np.clip(theta2, 0.0, 1.0)
            y_tilde = theta2 * y + (1.0 - theta2) * s_tilde
            
        final_sy = np.dot(s_tilde.T, y_tilde)
        if final_sy <= 0:
            print(f"Warning: Damping (Step 2 only) resulted in s.T * y_tilde = {final_sy:.4e} <= 0.")
            
        return s_tilde, y_tilde # s_tilde is the original s
    
    # -----------------------------------------------------------------
    # Standard Hessian Update Methods
    # -----------------------------------------------------------------
    
    def BFGS_hessian_update(self, hess, displacement, delta_grad):
        print("BFGS Hessian Update")
        return self._calculate_bfgs_delta(hess, displacement, delta_grad)
    
    def SR1_hessian_update(self, hess, displacement, delta_grad):
        print("SR1 Hessian Update")
        A = delta_grad - np.dot(hess, displacement)
        return self._calculate_sr1_delta(A, displacement)
    
    def PSB_hessian_update(self, hess, displacement, delta_grad):
        print("PSB Hessian Update")
        return self._calculate_psb_delta(hess, displacement, delta_grad)

    def FSB_hessian_update(self, hess, displacement, delta_grad):
        print("FSB Hessian Update")
        A = delta_grad - np.dot(hess, displacement)
        
        delta_hess_SR1 = self._calculate_sr1_delta(A, displacement)
        delta_hess_BFGS = self._calculate_bfgs_delta(hess, displacement, delta_grad)
        Bofill_const = self._calculate_bofill_const(A, displacement)
        
        # Per original code, mix with sqrt(Bofill_const)
        phi = np.sqrt(Bofill_const)
        delta_hess = (1.0 - phi) * delta_hess_BFGS + phi * delta_hess_SR1
        return delta_hess

    def CFD_FSB_hessian_update(self, hess, displacement, delta_grad):
        print("CFD FSB Hessian Update")
        A = 2.0 * (delta_grad - np.dot(hess, displacement))
        
        delta_hess_SR1 = self._calculate_sr1_delta(A, displacement)
        delta_hess_BFGS = self._calculate_bfgs_delta(hess, displacement, delta_grad)
        Bofill_const = self._calculate_bofill_const(A, displacement)
        
        phi = np.sqrt(Bofill_const)
        delta_hess = (1.0 - phi) * delta_hess_BFGS + phi * delta_hess_SR1
        return delta_hess

    def Bofill_hessian_update(self, hess, displacement, delta_grad):
        print("Bofill Hessian Update")
        A = delta_grad - np.dot(hess, displacement)
        
        delta_hess_SR1 = self._calculate_sr1_delta(A, displacement)
        delta_hess_PSB = self._calculate_psb_delta(hess, displacement, delta_grad)
        Bofill_const = self._calculate_bofill_const(A, displacement)

        # Bofill (SR1/PSB) mixes with the constant directly (phi^2)
        delta_hess = (1.0 - Bofill_const) * delta_hess_PSB + Bofill_const * delta_hess_SR1
        return delta_hess

    def CFD_Bofill_hessian_update(self, hess, displacement, delta_grad):
        print("CFD Bofill Hessian Update")
        A = 2.0 * (delta_grad - np.dot(hess, displacement))
        
        delta_hess_SR1 = self._calculate_sr1_delta(A, displacement)
        delta_hess_PSB = self._calculate_psb_delta(hess, displacement, delta_grad)
        Bofill_const = self._calculate_bofill_const(A, displacement)

        delta_hess = (1.0 - Bofill_const) * delta_hess_PSB + Bofill_const * delta_hess_SR1
        return delta_hess

    def pCFD_Bofill_hessian_update(self, hess, displacement, delta_grad):
        print("Perturbed CFD Bofill Hessian Update")
        
        # 1. CFD Bofill part
        A = 2.0 * (delta_grad - np.dot(hess, displacement))
        
        delta_hess_SR1 = self._calculate_sr1_delta(A, displacement)
        delta_hess_PSB = self._calculate_psb_delta(hess, displacement, delta_grad)
        Bofill_const = self._calculate_bofill_const(A, displacement)
        
        delta_hess = (1.0 - Bofill_const) * delta_hess_PSB + Bofill_const * delta_hess_SR1

        # 2. Perturbation Term
        print("Calculating perturbation term...")
        tmp_perturb_term_matrix = np.zeros_like(delta_hess)
        
        # Ensure displacement is 2D for null_space
        if displacement.ndim == 1:
            displacement_2d = displacement[np.newaxis, :]
        else:
            displacement_2d = displacement.T
            
        ortho_vecs = null_space(displacement_2d) # (N, N-1)
        
        # Iterate over column vectors (which are (N,) 1D arrays)
        for ortho_vec_i in ortho_vecs.T:
            for ortho_vec_j in ortho_vecs.T:
                # scalar = j.T @ (delta_B @ i)
                scalar_term = np.dot(ortho_vec_j.T, np.dot(delta_hess, ortho_vec_i))
                # matrix = (i @ j.T) + (j @ i.T)
                matrix_term = np.outer(ortho_vec_i, ortho_vec_j) + np.outer(ortho_vec_j, ortho_vec_i)
                tmp_perturb_term_matrix += scalar_term * matrix_term
                
        delta_hess = delta_hess + tmp_perturb_term_matrix
        return delta_hess

    def MSP_hessian_update(self, hess, displacement, delta_grad):
        print("MSP Hessian Update")
        A = delta_grad - np.dot(hess, displacement)
        
        delta_hess_MS = self._calculate_sr1_delta(A, displacement) # MS = SR1
        delta_hess_P = self._calculate_psb_delta(hess, displacement, delta_grad) # P = PSB
        
        A_norm = np.linalg.norm(A)
        displacement_norm = np.linalg.norm(displacement)
        phi_denominator = A_norm * displacement_norm
        
        phi_cos_arg = 0.0
        if phi_denominator >= self.denom_threshold:
            phi_cos_arg = np.dot(displacement.T, A) / phi_denominator
            # Clip argument for arccos to valid range [-1, 1]
            phi_cos_arg = np.clip(phi_cos_arg, -1.0, 1.0)
        else:
            print("phi denominator is too small, set phi=0.")
        
        # phi = sin(arccos(arg))^2 = 1 - cos(arccos(arg))^2 = 1 - arg^2
        phi = 1.0 - phi_cos_arg**2
        
        delta_hess = phi * delta_hess_P + (1.0 - phi) * delta_hess_MS
        return delta_hess
    
    # -----------------------------------------------------------------
    # DD-Enabled Hessian Update Methods 
    # -----------------------------------------------------------------

    def BFGS_hessian_update_dd(self, hess, displacement, delta_grad):
        """
        BFGS Hessian (B) update with Double Damping (DD).
        """
        print("--- BFGS Hessian Update with Double Damping ---")
        
        # 1. Apply Double Damping to get (s_tilde, y_tilde)
        s_tilde, y_tilde = self.double_damping_step2_only(
            displacement, delta_grad, self.dd_mu2
        )

        # 2. Call the helper function with the new (s_tilde, y_tilde)
        print("Calling damped BFGS update logic...")
        return self._calculate_bfgs_delta(hess, s_tilde, y_tilde)

    def FSB_hessian_update_dd(self, hess, displacement, delta_grad):
        """
        FSB Hessian (B) update with Double Damping (DD).
        """
        print("--- FSB Hessian Update with Double Damping ---")

        # 1. Apply Double Damping
        s_tilde, y_tilde = self.double_damping_step2_only(
            displacement, delta_grad,  self.dd_mu2
        )

        # 2. Call damped logic
        print("Calling damped FSB update logic...")
        A_tilde = y_tilde - np.dot(hess, s_tilde)
        
        delta_hess_SR1 = self._calculate_sr1_delta(A_tilde, s_tilde)
        delta_hess_BFGS = self._calculate_bfgs_delta(hess, s_tilde, y_tilde)
        Bofill_const = self._calculate_bofill_const(A_tilde, s_tilde)
        
        phi = np.sqrt(Bofill_const)
        delta_hess = (1.0 - phi) * delta_hess_BFGS + phi * delta_hess_SR1
        return delta_hess

    def CFD_FSB_hessian_update_dd(self, hess, displacement, delta_grad):
        """
        CFD FSB Hessian (B) update with Double Damping (DD).
        """
        print("--- CFD FSB Hessian Update with Double Damping ---")

        # 1. Apply Double Damping
        s_tilde, y_tilde = self.double_damping_step2_only(
            displacement, delta_grad, self.dd_mu2
        )

        # 2. Call damped logic
        print("Calling damped CFD FSB update logic...")
        A_tilde = 2.0 * (y_tilde - np.dot(hess, s_tilde))

        delta_hess_SR1 = self._calculate_sr1_delta(A_tilde, s_tilde)
        delta_hess_BFGS = self._calculate_bfgs_delta(hess, s_tilde, y_tilde)
        Bofill_const = self._calculate_bofill_const(A_tilde, s_tilde)

        phi = np.sqrt(Bofill_const)
        delta_hess = (1.0 - phi) * delta_hess_BFGS + phi * delta_hess_SR1
        return delta_hess