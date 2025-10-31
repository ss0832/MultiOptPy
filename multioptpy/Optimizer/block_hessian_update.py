import numpy as np

"""
Block Hessian Update Class

ref: https://arxiv.org/pdf/1609.00318
"""

def symm(A):
    return 0.5 * (A + A.T)

def safe_inv(A, reg=1e-10):
    """Invert A with small regularization fallback, then pinv."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Areg = A + reg * np.eye(A.shape[0])
        try:
            return np.linalg.inv(Areg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(Areg)

class BlockHessianUpdate:
    def __init__(self, block_size=4, max_window=8, denom_threshold=1e-12, inv_reg=1e-10):
        """
        block_size: number of stored steps to use when performing a block update
        max_window: maximum history length to retain (>= block_size)
        """
        assert max_window >= block_size
        self.block_size = int(block_size)
        self.max_window = int(max_window)
        self.denom_threshold = denom_threshold
        self.inv_reg = inv_reg

        # history stored as lists of vectors (each vector shape (n,))
        self.S_list = []
        self.Y_list = []

        # Default parameters for Double Damping
        self.dd_mu1 = 0.2
        self.dd_mu2 = 0.2


    def delete_old_data(self):
        """Drop the oldest history item (if any)."""
        if self.S_list:
            self.S_list.pop(0)
            self.Y_list.pop(0)

    def _push_history(self, s, y):
        """Append new step (s,y), maintain window."""
        self.S_list.append(s.copy())
        self.Y_list.append(y.copy())
        if len(self.S_list) > self.max_window:
            self.S_list.pop(0); self.Y_list.pop(0)

    def _assemble_block(self, use_last_k=None):
        """Return S (n x q) and Y (n x q) matrices from most recent columns."""
        if use_last_k is None:
            use_last_k = min(self.block_size, len(self.S_list))
        k = min(use_last_k, len(self.S_list))
        if k == 0:
            return None, None
        # take last k entries
        Scols = [self.S_list[-k + i] for i in range(k)]
        Ycols = [self.Y_list[-k + i] for i in range(k)]
        S = np.column_stack(Scols)  # n x k
        Y = np.column_stack(Ycols)
        return S, Y

    # -----------------------------------------------------------------
    # --- Base Block Update Methods ---
    # -----------------------------------------------------------------

    def _block_BFGS_update(self, B, S, Y):
        """
        B <- B - B S (S^T B S)^{-1} S^T B + Y (S^T Y)^{-1} Y^T
        S,Y are n x q with columns as steps.
        """
        if S is None or Y is None or S.shape[1] == 0:
            return B.copy()
            
        # filter near linear dependence in S by SVD (drop tiny singular values)
        U, svals, Vt = np.linalg.svd(S, full_matrices=False)
        keep = svals > 1e-8
        if not np.any(keep):
            return B.copy()
        rank = np.sum(keep)
        col_norms = np.linalg.norm(S, axis=0)
        idx_sorted = np.argsort(-col_norms)
        keep_idx = np.sort(idx_sorted[:rank])
        Sf = S[:, keep_idx]
        Yf = Y[:, keep_idx]

        # Further filter columns based on curvature condition y^T s > threshold
        keep_cols = []
        for i in range(Sf.shape[1]):
            s = Sf[:, i]
            y = Yf[:, i]
            denom = np.dot(y, s)
            if denom <= self.denom_threshold:
                continue
            keep_cols.append(i)

        if len(keep_cols) == 0:
            return B.copy()


        M1 = np.dot(np.dot(Sf.T, B), Sf)      # q x q
        M2 = np.dot(Sf.T, Yf)                # q x q

        invM1 = safe_inv(M1, reg=self.inv_reg)
        invM2 = safe_inv(M2, reg=self.inv_reg)

        term1 = np.dot(np.dot(np.dot(B, Sf), invM1), np.dot(Sf.T, B))
        term2 = np.dot(np.dot(Yf, invM2), Yf.T)
        Bp = B - term1 + term2
        return symm(Bp)

    def _block_PSB_update(self, B, S, Y, denom_threshold=1e-8):
        """
        Block PSB Hessian update. Applies single-step PSB for each column.
        """
        if S is None or Y is None or S.shape[1] == 0:
            return B.copy()

        # SVD filtering for near linear dependence in S
        _, svals, _ = np.linalg.svd(S, full_matrices=False)
        keep = svals > denom_threshold
        if not np.any(keep):
            return B.copy()

        rank = np.sum(keep)
        col_norms = np.linalg.norm(S, axis=0)
        idx_sorted = np.argsort(-col_norms)
        keep_idx = np.sort(idx_sorted[:rank])
        Sf = S[:, keep_idx]
        Yf = Y[:, keep_idx]

        # Calculate block_1 and denominator for each column
        n, q = Sf.shape
        delta_hess_total = np.zeros((n,n))

        for i in range(q):
            s = Sf[:, i:i+1]  # column vector, shape (n, 1)
            y = Yf[:, i:i+1]
            block_1 = y - np.dot(B, s)
            block_2_denominator = float(np.dot(s.T, s))
            if np.abs(block_2_denominator) >= denom_threshold:
                block_2 = np.dot(s, s.T) / (block_2_denominator ** 2)
                delta_hess_P = (-np.dot(block_1.T, s) * block_2 +
                                (np.dot(block_1, s.T) + np.dot(s, block_1.T)) / block_2_denominator)
                delta_hess_total += delta_hess_P
            # else: delta_hess_P is zero, so do nothing

        Bp = B + delta_hess_total
        return Bp # Already symmetric if B is

    def _block_SR1_update(self, B, S, Y):
        """
        Block SR1 generalization: R = Y - B S
        Delta = R (S^T R)^{-1} R^T
        """
        if S is None or Y is None or S.shape[1] == 0:
            return B.copy()
        R = Y - np.dot(B, S)
        M = np.dot(S.T, R)
        invM = safe_inv(M, reg=self.inv_reg)
        Delta = np.dot(np.dot(R, invM), R.T)
        Bp = B + Delta
        return symm(Bp)

    def _block_CFD_SR1_update(self, B, S, Y):
        """
        Block CFD-SR1 generalization: R = 2.0 * (Y - B S)
        """
        if S is None or Y is None or S.shape[1] == 0:
            return B.copy()
        R = 2.0 * (Y - np.dot(B, S))
        M = np.dot(S.T, R)
        invM = safe_inv(M, reg=self.inv_reg)
        Delta = np.dot(np.dot(R, invM), R.T)
        Bp = B + Delta
        return symm(Bp)

    # -----------------------------------------------------------------
    # --- Helper for Calculating Weights ---
    # -----------------------------------------------------------------

    def _get_individual_weights(self, B, S, Y, is_cfd=False, use_bofill_logic=False):
        """
        Internal helper to calculate individual weights for mixing.
        
        Args:
            is_cfd (bool): If True, use A = 2.0 * (y - Bs)
            use_bofill_logic (bool): 
                If True (for Bofill/CFD-FSB), returns w_j = c_j
                If False (for FSB), returns w_j = sqrt(c_j)

        Returns:
            c_list (list of phi^2), w_list (list of mixing weights)
        """
        if S is None or Y is None:
            return [], []

        q = S.shape[1]
        c_list = [] # phi^2
        w_list = [] # mixing weight (phi or phi^2)

        for j in range(q):
            s = S[:, j]
            y = Y[:, j]
            
            A = y - np.dot(B, s)
            if is_cfd:
                A = 2.0 * A

            num = (np.dot(A.T, s)) ** 2
            denom = (np.dot(A.T, A)) * (np.dot(s.T, s))
            c = num / denom if np.abs(denom) > self.denom_threshold else 0.0
            if np.isnan(c):
                c = 0.0
            c = float(max(0.0, min(1.0, c)))
            c_list.append(c)

            if use_bofill_logic:
                w_list.append(c) # Use c (phi^2)
            else:
                w_list.append(np.sqrt(c)) # Use sqrt(c) (phi)
            
        return c_list, w_list

    # -----------------------------------------------------------------
    # --- "Mean Weight" Mixed Methods ---
    # -----------------------------------------------------------------

    def _block_FSB_update(self, B, S, Y):
        """
        Original Block-FSB: Mix block-SR1 and block-BFGS updates
        using the *mean* of individual sqrt(c_j) weights.
        """
        if S is None or Y is None:
            return B.copy()

        # Build block SR1 and block BFGS deltas
        Delta_sr1 = self._block_SR1_update(B, S, Y) - B
        Delta_bfgs = self._block_BFGS_update(B, S, Y) - B

        # Get individual weights (w_j = sqrt(c_j) for FSB)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=False, use_bofill_logic=False)

        w_mean = float(np.mean(w_list)) if len(w_list) > 0 else 0.0
        Bp = B + w_mean * Delta_sr1 + (1.0 - w_mean) * Delta_bfgs
        return symm(Bp)

    def _block_CFD_FSB_update(self, B, S, Y):
        """
        Original Block-CFD-FSB: Mix block-CFD-SR1 and block-BFGS updates
        using the *mean* of individual c_j weights.
        """
        if S is None or Y is None:
            return B.copy()

        # Build block CFD_SR1 and block BFGS deltas
        Delta_sr1 = self._block_CFD_SR1_update(B, S, Y) - B
        Delta_bfgs = self._block_BFGS_update(B, S, Y) - B

        # Get individual weights (w_j = c_j for CFD-FSB)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=True, use_bofill_logic=True)
        
        w_mean = float(np.mean(w_list)) if len(w_list) > 0 else 0.0
        Bp = B + w_mean * Delta_sr1 + (1.0 - w_mean) * Delta_bfgs
        return symm(Bp)

    def _block_Bofill_update(self, B, S, Y):
        """
        Original Block-Bofill: Mix block-SR1 and block-PSB updates
        using the *mean* of individual c_j weights.
        """
        if S is None or Y is None:
            return B.copy()
            
        Delta_psb = self._block_PSB_update(B, S, Y) - B
        Delta_sr1 = self._block_SR1_update(B, S, Y) - B

        # Get individual weights (w_j = c_j for Bofill)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=False, use_bofill_logic=True)

        w_mean = float(np.mean(w_list)) if len(w_list) > 0 else 0.0
        Bp = B + w_mean * (Delta_sr1) + (1.0 - w_mean) * (Delta_psb)
        return symm(Bp)

    def _block_CFD_Bofill_update(self, B, S, Y):
        """
        Original Block-CFD-Bofill: Mix block-CFD-SR1 and block-PSB updates
        using the *mean* of individual c_j weights.
        """
        if S is None or Y is None:
            return B.copy()
            
        Delta_psb = self._block_PSB_update(B, S, Y) - B
        Delta_sr1 = self._block_CFD_SR1_update(B, S, Y) - B

        # Get individual weights (w_j = c_j for CFD-Bofill)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=True, use_bofill_logic=True)

        w_mean = float(np.mean(w_list)) if len(w_list) > 0 else 0.0
        Bp = B + w_mean * (Delta_sr1) + (1.0 - w_mean) * (Delta_psb)
        return symm(Bp)

    # -----------------------------------------------------------------
    # --- "Weighted Subspace" Mixed Methods ---
    # -----------------------------------------------------------------

    def _block_FSB_update_weighted(self, B, S, Y):
        """
        Block-FSB update using the "Weighted Subspace" approach.
        """
        if S is None or Y is None:
            return B.copy()
        
        print("Calculating Weighted Subspace FSB update")
        # 1. Get individual weights (w_j = sqrt(c_j) for FSB)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=False, use_bofill_logic=False)
        W_sr1 = np.diag(w_list)
        W_bfgs = np.diag([1.0 - w for w in w_list])

        # 2. Build weighted subspace matrices
        S_sr1 = np.dot(S, W_sr1)
        Y_sr1 = np.dot(Y, W_sr1)
        S_bfgs = np.dot(S, W_bfgs)
        Y_bfgs = np.dot(Y, W_bfgs)
        
        # 3. Calculate updates in each subspace
        print("... calculating weighted SR1 subspace ...")
        Delta_sr1 = self._block_SR1_update(B, S_sr1, Y_sr1) - B
        print("... calculating weighted BFGS subspace ...")
        Delta_bfgs = self._block_BFGS_update(B, S_bfgs, Y_bfgs) - B

        # 4. Combine the updates
        Bp = B + Delta_sr1 + Delta_bfgs
        return symm(Bp)

    def _block_CFD_FSB_update_weighted(self, B, S, Y):
        """
        Block-CFD-FSB update using the "Weighted Subspace" approach.
        """
        if S is None or Y is None:
            return B.copy()
            
        print("Calculating Weighted Subspace CFD-FSB update")
        # 1. Get individual weights (w_j = c_j for CFD-FSB)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=True, use_bofill_logic=True)
        W_sr1 = np.diag(w_list)
        W_bfgs = np.diag([1.0 - w for w in w_list])

        # 2. Build weighted subspace matrices
        S_sr1 = np.dot(S, W_sr1)
        Y_sr1 = np.dot(Y, W_sr1)
        S_bfgs = np.dot(S, W_bfgs)
        Y_bfgs = np.dot(Y, W_bfgs)
        
        # 3. Calculate updates in each subspace
        print("... calculating weighted CFD-SR1 subspace ...")
        Delta_sr1 = self._block_CFD_SR1_update(B, S_sr1, Y_sr1) - B
        print("... calculating weighted BFGS subspace ...")
        Delta_bfgs = self._block_BFGS_update(B, S_bfgs, Y_bfgs) - B

        # 4. Combine the updates
        Bp = B + Delta_sr1 + Delta_bfgs
        return symm(Bp)

    def _block_Bofill_update_weighted(self, B, S, Y):
        """
        Block-Bofill update using the "Weighted Subspace" approach.
        """
        if S is None or Y is None:
            return B.copy()
            
        print("Calculating Weighted Subspace Bofill update")
        # 1. Get individual weights (w_j = c_j for Bofill)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=False, use_bofill_logic=True)
        W_sr1 = np.diag(w_list)
        W_psb = np.diag([1.0 - w for w in w_list])

        # 2. Build weighted subspace matrices
        S_sr1 = np.dot(S, W_sr1)
        Y_sr1 = np.dot(Y, W_sr1)
        S_psb = np.dot(S, W_psb)
        Y_psb = np.dot(Y, W_psb)
        
        # 3. Calculate updates in each subspace
        print("... calculating weighted SR1 subspace ...")
        Delta_sr1 = self._block_SR1_update(B, S_sr1, Y_sr1) - B
        print("... calculating weighted PSB subspace ...")
        Delta_psb = self._block_PSB_update(B, S_psb, Y_psb) - B

        # 4. Combine the updates
        Bp = B + Delta_sr1 + Delta_psb
        return symm(Bp)

    def _block_CFD_Bofill_update_weighted(self, B, S, Y):
        """
        Block-CFD-Bofill update using the "Weighted Subspace" approach.
        """
        if S is None or Y is None:
            return B.copy()
            
        print("Calculating Weighted Subspace CFD-Bofill update")
        # 1. Get individual weights (w_j = c_j for CFD-Bofill)
        c_list, w_list = self._get_individual_weights(B, S, Y, 
                                            is_cfd=True, use_bofill_logic=True)
        W_sr1 = np.diag(w_list)
        W_psb = np.diag([1.0 - w for w in w_list])

        # 2. Build weighted subspace matrices
        S_sr1 = np.dot(S, W_sr1)
        Y_sr1 = np.dot(Y, W_sr1)
        S_psb = np.dot(S, W_psb)
        Y_psb = np.dot(Y, W_psb)
        
        # 3. Calculate updates in each subspace
        print("... calculating weighted CFD-SR1 subspace ...")
        Delta_sr1 = self._block_CFD_SR1_update(B, S_sr1, Y_sr1) - B
        print("... calculating weighted PSB subspace ...")
        Delta_psb = self._block_PSB_update(B, S_psb, Y_psb) - B

        # 4. Combine the updates
        Bp = B + Delta_sr1 + Delta_psb
        return symm(Bp)
        
    # -----------------------------------------------------------------
    # --- Public Methods ---
    # -----------------------------------------------------------------

    def block_BFGS_hessian_update(self, B, displacement, delta_grad):
        print("Block BFGS update method")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_BFGS_update(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_FSB_hessian_update(self, B, displacement, delta_grad):
        print("Block FSB update method (Mean Weight)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_FSB_update(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_CFD_FSB_hessian_update(self, B, displacement, delta_grad):
        print("Block CFD_FSB update method (Mean Weight)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_CFD_FSB_update(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_Bofill_hessian_update(self, B, displacement, delta_grad):
        print("Block Bofill update method (Mean Weight)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_Bofill_update(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_CFD_Bofill_hessian_update(self, B, displacement, delta_grad):
        print("Block CFD_Bofill update method (Mean Weight)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_CFD_Bofill_update(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    # -----------------------------------------------------------------
    # --- Public Methods (Weighted Subspace) ---
    # -----------------------------------------------------------------

    def block_FSB_hessian_update_weighted(self, B, displacement, delta_grad):
        """
        Public entry point for "Weighted Subspace" FSB update.
        (Alternative to block_FSB_hessian_update)
        """
        print("Block FSB update method (Weighted Subspace)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        
        Bp = self._block_FSB_update_weighted(B, S, Y)
        
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_CFD_FSB_hessian_update_weighted(self, B, displacement, delta_grad):
        """
        Public entry point for "Weighted Subspace" CFD-FSB update.
        (Alternative to block_CFD_FSB_hessian_update)
        """
        print("Block CFD_FSB update method (Weighted Subspace)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        
        Bp = self._block_CFD_FSB_update_weighted(B, S, Y)
        
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_Bofill_hessian_update_weighted(self, B, displacement, delta_grad):
        """
        Public entry point for "Weighted Subspace" Bofill update.
        (Alternative to block_Bofill_hessian_update)
        """
        print("Block Bofill update method (Weighted Subspace)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        
        Bp = self._block_Bofill_update_weighted(B, S, Y)
        
        self.delete_old_data()
        return Bp - B # Return deltaB
        
    def block_CFD_Bofill_hessian_update_weighted(self, B, displacement, delta_grad):
        """
        Public entry point for "Weighted Subspace" CFD-Bofill update.
        (Alternative to block_CFD_Bofill_hessian_update)
        """
        print("Block CFD_Bofill update method (Weighted Subspace)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        
        Bp = self._block_CFD_Bofill_update_weighted(B, S, Y)
        
        self.delete_old_data()
        return Bp - B # Return deltaB

    # -----------------------------------------------------------------
    # --- Public Methods (DD-Enabled) ---
    # -----------------------------------------------------------------
    
    def double_damping_step2_only(self, s, y, mu2):
        """
        Implements ONLY Step 2 of the Double Damping (DD) procedure [cite: 102, 362-364].
        This step does NOT require the inverse Hessian H.
        It is equivalent to Powell's damping with B=I [cite: 365-367].
        """
        s_tilde = s
        y_tilde = y
        
        s_tilde_y = np.dot(s_tilde.T, y)
        s_tilde_s_tilde = np.dot(s_tilde.T, s_tilde)

        # Check if damping is needed
        if s_tilde_y < mu2 * s_tilde_s_tilde:
            print(f"DD Step 2 active: s_tilde.T*y ({s_tilde_y:.4e}) < mu2*s_tilde.T*s_tilde ({mu2 * s_tilde_s_tilde:.4e})")
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

    def _apply_block_damping_step2(self, S, Y):
        """
        Helper to apply H-free DD Step 2 [cite: 102, 362-364] to all columns of S, Y.
        """
        if S is None or Y is None:
            return None, None
            
        q = S.shape[1]
        S_tilde = S.copy()
        Y_tilde = Y.copy()
        
        print(f"Applying H-free DD (Step 2) to {q} pairs...")
        for i in range(q):
            s_i = S[:, i]
            y_i = Y[:, i]
            # Apply H-free DD Step 2
            s_tilde_i, y_tilde_i = self.double_damping_step2_only(s_i, y_i, self.dd_mu2)
            S_tilde[:, i] = s_tilde_i
            Y_tilde[:, i] = y_tilde_i
        
        return S_tilde, Y_tilde

    def _block_BFGS_update_dd(self, B, S, Y):
        # (Internal logic for DD-BFGS)
        if S is None or Y is None:
            return B.copy()
        U, svals, Vt = np.linalg.svd(S, full_matrices=False)
        keep = svals > 1e-8
        if not np.any(keep):
            return B.copy()
        rank = np.sum(keep)
        col_norms = np.linalg.norm(S, axis=0)
        idx_sorted = np.argsort(-col_norms)
        keep_idx = np.sort(idx_sorted[:rank])
        Sf = S[:, keep_idx]
        Yf = Y[:, keep_idx]
        Sf_tilde, Yf_tilde = self._apply_block_damping_step2(Sf, Yf)
        M1 = np.dot(np.dot(Sf_tilde.T, B), Sf_tilde)
        M2 = np.dot(Sf_tilde.T, Yf_tilde)
        invM1 = safe_inv(M1, reg=self.inv_reg)
        invM2 = safe_inv(M2, reg=self.inv_reg)
        term1 = np.dot(np.dot(np.dot(B, Sf_tilde), invM1), np.dot(Sf_tilde.T, B))
        term2 = np.dot(np.dot(Yf_tilde, invM2), Yf_tilde.T)
        Bp = B - term1 + term2
        return symm(Bp)
        
    def _block_FSB_update_dd(self, B, S, Y):
        # (Internal logic for DD-FSB)
        if S is None or Y is None:
            return B.copy()
        S_tilde, Y_tilde = self._apply_block_damping_step2(S, Y)
        q = S_tilde.shape[1]
        c_list, w_list = self._get_individual_weights(B, S_tilde, Y_tilde, 
                                            is_cfd=False, use_bofill_logic=False)
        B_sr1_delta = self._block_SR1_update(B, S_tilde, Y_tilde) - B
        B_bfgs_delta = self._block_BFGS_update(B, S_tilde, Y_tilde) - B
        w_mean = float(np.mean(w_list)) if len(w_list) > 0 else 0.0
        Bp = B + w_mean * B_sr1_delta + (1.0 - w_mean) * B_bfgs_delta
        return symm(Bp)

    def _block_CFD_FSB_update_dd(self, B, S, Y):
        # (Internal logic for DD-CFD-FSB)
        if S is None or Y is None:
            return B.copy()
        S_tilde, Y_tilde = self._apply_block_damping_step2(S, Y)
        q = S_tilde.shape[1]
        c_list, w_list = self._get_individual_weights(B, S_tilde, Y_tilde, 
                                            is_cfd=True, use_bofill_logic=True)
        B_sr1_delta = self._block_CFD_SR1_update(B, S_tilde, Y_tilde) - B
        B_bfgs_delta = self._block_BFGS_update(B, S_tilde, Y_tilde) - B
        w_mean = float(np.mean(w_list)) if len(w_list) > 0 else 0.0
        Bp = B + w_mean * B_sr1_delta + (1.0 - w_mean) * B_bfgs_delta
        return symm(Bp)


    def block_BFGS_hessian_update_dd(self, B, displacement, delta_grad):
        """
        Public entry point for Block BFGS update with H-free DD (Step 2).
        """
        print("Block BFGS update method with DD (Step 2 only)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_BFGS_update_dd(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_FSB_hessian_update_dd(self, B, displacement, delta_grad):
        """
        Public entry point for Block FSB update with H-free DD (Step 2).
        """
        print("Block FSB update method with DD (Step 2 only)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_FSB_update_dd(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB

    def block_CFD_FSB_hessian_update_dd(self, B, displacement, delta_grad):
        """
        Public entry point for Block CFD-FSB update with H-free DD (Step 2).
        """
        print("Block CFD_FSB update method with DD (Step 2 only)")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        S, Y = self._assemble_block(self.block_size)
        Bp = self._block_CFD_FSB_update_dd(B, S, Y)
        self.delete_old_data()
        return Bp - B # Return deltaB