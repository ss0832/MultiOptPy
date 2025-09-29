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
        S = np.column_stack(Scols)   # n x k
        Y = np.column_stack(Ycols)
        return S, Y


    def _block_BFGS_update(self, B, S, Y):
        """
        B <- B - B S (S^T B S)^{-1} S^T B + Y (S^T Y)^{-1} Y^T
        S,Y are n x q with columns as steps.
        """
        if S is None or Y is None:
            return B.copy()
        # filter near linear dependence in S by SVD (drop tiny singular values)
        U, svals, Vt = np.linalg.svd(S, full_matrices=False)
        keep = svals > 1e-8
        if not np.any(keep):
            return B.copy()
        # choose columns corresponding to largest contributions
        # simple approach: keep first rank columns from SVD basis converted back to original column indices
        rank = np.sum(keep)
        # project S onto leading rank singular vectors to get an orthonormal basis for subspace
        #Ssub = np.dot(np.dot(U[:, :rank], np.diag(svals[:rank])), Vt[:rank, :])
        # Alternatively pick original columns with largest norms (preserve physical steps)
        col_norms = np.linalg.norm(S, axis=0)
        idx_sorted = np.argsort(-col_norms)
        keep_idx = np.sort(idx_sorted[:rank])
        Sf = S[:, keep_idx]
        Yf = Y[:, keep_idx]

        M1 = np.dot(np.dot(Sf.T, B), Sf)       # q x q
        M2 = np.dot(Sf.T, Yf)              # q x q

        invM1 = safe_inv(M1, reg=self.inv_reg)
        invM2 = safe_inv(M2, reg=self.inv_reg)

        term1 = np.dot(np.dot(np.dot(B, Sf), invM1), np.dot(Sf.T, B))
        term2 = np.dot(np.dot(Yf, invM2), Yf.T)
        Bp = B - term1 + term2
        return symm(Bp)

    def _block_PSB_update(self, B, S, Y, denom_threshold=1e-8):
        """
        Block PSB Hessian update based on:
        Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.

        B: Hessian matrix (n x n)
        S: Displacement matrix (n x q), columns are step vectors
        Y: Gradient difference matrix (n x q), columns are delta_grad vectors
        denom_threshold: Threshold for denominator to avoid instability

        Returns updated Hessian matrix.
        """

        if S is None or Y is None:
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
        delta_hess_blocks = []

        for i in range(q):
            s = Sf[:, i:i+1]  # column vector, shape (n, 1)
            y = Yf[:, i:i+1]
            block_1 = y - np.dot(B, s)
            block_2_denominator = float(np.dot(s.T, s))
            if np.abs(block_2_denominator) >= denom_threshold:
                block_2 = np.dot(s, s.T) / (block_2_denominator ** 2)
                delta_hess_P = (-np.dot(block_1.T, s) * block_2 +
                                (np.dot(block_1, s.T) + np.dot(s, block_1.T)) / block_2_denominator)
            else:
                delta_hess_P = np.zeros((n, n))
            delta_hess_blocks.append(delta_hess_P)

        # Sum all block updates
        delta_hess_total = sum(delta_hess_blocks)
        Bp = B + delta_hess_total
        return Bp

    def _block_SR1_update(self, B, S, Y):
        """
        Block SR1 generalization: R = Y - B S
        Delta = R (S^T R)^{-1} R^T   (if invertible)
        B <- B + Delta
        """
        if S is None or Y is None:
            return B.copy()
        R = Y - np.dot(B, S)
        M = np.dot(S.T, R)
        invM = safe_inv(M, reg=self.inv_reg)
        Delta = np.dot(np.dot(R, invM), R.T)
        Bp = B + Delta
        return symm(Bp)

    def _block_CFD_SR1_update(self, B, S, Y):
        """
        Block CFD-SR1 generalization: R = Y - B S
        Delta = R (S^T R)^{-1} R^T   (if invertible)
        B <- B + Delta
        """
        if S is None or Y is None:
            return B.copy()
        R = 2.0 * (Y - np.dot(B, S))
        M = np.dot(S.T, R)
        invM = safe_inv(M, reg=self.inv_reg)
        Delta = np.dot(np.dot(R, invM), R.T)
        Bp = B + Delta
        return symm(Bp)

    def _block_FSB_update(self, B, S, Y):
        """
        Block-FSB: Mix block-SR1 and block-BFGS updates.
        Weight computed per-column using single-step Bofill-like scalar, then averaged.

        Parameters:
        - B: Hessian matrix (n x n)
        - S: Displacement matrix (n x q), columns are step vectors
        - Y: Gradient difference matrix (n x q), columns are delta_grad vectors
        - denom_threshold: Small value for denominator stabilization
        - block_SR1_update: Function handle for block SR1 update (required)
        - block_BFGS_update: Function handle for block BFGS update (required)

        Returns:
        - Updated Hessian matrix
        """

        if S is None or Y is None:
            return B.copy()

        q = S.shape[1]
        c_list = []

        # Build block SR1 and block BFGS
        B_sr1 = self._block_SR1_update(B, S, Y) - B
        B_bfgs = self._block_BFGS_update(B, S, Y) - B

        # Compute per-column c_i (Bofill-like coefficient)
        for j in range(q):
            s = S[:, j]
            y = Y[:, j]
            A = y - np.dot(B, s)
            num = (np.dot(A.T, s)) ** 2
            denom = (np.dot(A.T, A)) * (np.dot(s.T, s))
            c = num / denom if np.abs(denom) > self.denom_threshold else 0.0
            if np.isnan(c):
                c = 0.0
            c = float(max(0.0, min(1.0, c)))
            c_list.append(c)

        c_mean = float(np.mean(c_list)) if len(c_list) > 0 else 0.0
        # Use sqrt weighting as in single-step FSB
        w = np.sqrt(c_mean)
        Bp = B + w * B_sr1 + (1.0 - w) * B_bfgs
        return symm(Bp)

    def _block_CFD_FSB_update(self, B, S, Y):
        """
        Block-CFD_FSB: Mix block-CFD_SR1 and block-BFGS updates.
        Weight computed per-column using single-step CFD_Bofill-like scalar, then averaged.

        Parameters:
        - B: Hessian matrix (n x n)
        - S: Displacement matrix (n x q), columns are step vectors
        - Y: Gradient difference matrix (n x q), columns are delta_grad vectors
        - denom_threshold: Small value for denominator stabilization
        - block_CFD_SR1_update: Function handle for block CFD_SR1 update (required)
        - block_BFGS_update: Function handle for block BFGS update (required)

        Returns:
        - Updated Hessian matrix
        """

        if S is None or Y is None:
            return B.copy()

        q = S.shape[1]
        c_list = []

        # Build block CFD_SR1 and block BFGS
        B_sr1 = self._block_CFD_SR1_update(B, S, Y) - B
        B_bfgs = self._block_BFGS_update(B, S, Y) - B

        # Compute per-column c_i (CFD_Bofill-like coefficient)
        for j in range(q):
            s = S[:, j]
            y = Y[:, j]
            A = 2.0 * (y - np.dot(B, s))
            num = (np.dot(A.T, s)) ** 2
            denom = (np.dot(A.T, A)) * (np.dot(s.T, s))
            c = num / denom if np.abs(denom) > self.denom_threshold else 0.0
            if np.isnan(c):
                c = 0.0
            c = float(max(0.0, min(1.0, c)))
            c_list.append(c)

        c_mean = float(np.mean(c_list)) if len(c_list) > 0 else 0.0
        # Use sqrt weighting as in single-step FSB
        w = np.sqrt(c_mean)
        Bp = B + w * B_sr1 + (1.0 - w) * B_bfgs
        return symm(Bp)

    def _block_Bofill_update(self, B, S, Y):
        """
        Block-Bofill: mix block-SR1-like and a PSB-like block term.
        Implementation choice: use block_BFGS and a simple PSB-inspired block,
        then mix using averaged per-column bofill-like coefficient (no guarantee it's exactly the classical Bofill).
        """
        if S is None or Y is None:
            return B.copy()
        # block-PSB-like delta
        B_psb = self._block_PSB_update(B, S, Y)
        Delta_psb = B_psb - B
        # block-SR1-like delta
        B_sr1 = self._block_SR1_update(B, S, Y)
        Delta_sr1 = B_sr1 - B

        # compute per-column bofill-like coefficient (average of single-step ones)
        q = S.shape[1]
        c_list = []
        for j in range(q):
            s = S[:, j]; y = Y[:, j]
            A = y - np.dot(B, s)
            num = (np.dot(A.T, s))**2
            denom = (np.dot(A.T, A)) * (np.dot(s.T, s))
            c = num / denom if np.abs(denom) > self.denom_threshold else 0.0
            c = float(max(0.0, min(1.0, c))) if not np.isnan(c) else 0.0
            c_list.append(c)
        c_mean = float(np.mean(c_list)) if len(c_list) > 0 else 0.0

        # Bofill mixing: use c_mean linearly
        Bp = B + c_mean * (Delta_sr1) + (1.0 - c_mean) * (Delta_psb)
        return symm(Bp)


    def _block_CFD_Bofill_update(self, B, S, Y):
        """
        Block-CFD_Bofill: mix block-CFD_SR1-like and a PSB-like block term.
        Implementation choice: use block_BFGS and a simple PSB-inspired block,
        then mix using averaged per-column CFD_bofill-like coefficient (no guarantee it's exactly the classical Bofill).
        """
        if S is None or Y is None:
            return B.copy()
        # block-PSB-like delta
        B_psb = self._block_PSB_update(B, S, Y)
        Delta_psb = B_psb - B
        # block-SR1-like delta
        B_sr1 = self._block_CFD_SR1_update(B, S, Y)
        Delta_sr1 = B_sr1 - B

        # compute per-column bofill-like coefficient (average of single-step ones)
        q = S.shape[1]
        c_list = []
        for j in range(q):
            s = S[:, j]; y = Y[:, j]
            A = 2.0 * (y - np.dot(B, s))
            num = (np.dot(A.T, s))**2
            denom = (np.dot(A.T, A)) * (np.dot(s.T, s))
            c = num / denom if np.abs(denom) > self.denom_threshold else 0.0
            c = float(max(0.0, min(1.0, c))) if not np.isnan(c) else 0.0
            c_list.append(c)
        c_mean = float(np.mean(c_list)) if len(c_list) > 0 else 0.0

        # Bofill mixing: use c_mean linearly
        Bp = B + c_mean * (Delta_sr1) + (1.0 - c_mean) * (Delta_psb)
        return symm(Bp)


    def block_BFGS_hessian_update(self, B, displacement, delta_grad):
        print("block BFGS update method")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        
        S, Y = self._assemble_block(self.block_size)
        deltaB = self._block_BFGS_update(B, S, Y) - B
        # rotate history: drop oldest
        self.delete_old_data()
        return deltaB

    def block_FSB_hessian_update(self, B, displacement, delta_grad):
        print("block FSB update method")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
    
        S, Y = self._assemble_block(self.block_size)
        deltaB = self._block_FSB_update(B, S, Y) - B
        self.delete_old_data()
        return deltaB

    def block_CFD_FSB_hessian_update(self, B, displacement, delta_grad):
        print("block CFD_FSB update method")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
    
        S, Y = self._assemble_block(self.block_size)
        deltaB = self._block_CFD_FSB_update(B, S, Y) - B
        self.delete_old_data()
        return deltaB

    def block_Bofill_hessian_update(self, B, displacement, delta_grad):
        print("block Bofill update method")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        
        S, Y = self._assemble_block(self.block_size)
        deltaB = self._block_Bofill_update(B, S, Y) - B
        self.delete_old_data()
        return deltaB

    def block_CFD_Bofill_hessian_update(self, B, displacement, delta_grad):
        print("block CFD_Bofill update method")
        s = displacement.reshape(-1)
        y = delta_grad.reshape(-1)
        self._push_history(s, y)
        
        S, Y = self._assemble_block(self.block_size)
        deltaB = self._block_CFD_Bofill_update(B, S, Y) - B
        self.delete_old_data()
        return deltaB