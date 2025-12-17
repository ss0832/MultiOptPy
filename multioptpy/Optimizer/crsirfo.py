import numpy as np
import scipy.linalg
from multioptpy.Optimizer.rsirfo import RSIRFO

class CRSIRFO(RSIRFO):
    def __init__(self, constraints=None, **config):
        """
        Constrained RS-I-RFO Optimizer (CRS-I-RFO)
        """
        super().__init__(**config)
        self.constraints_obj = constraints
        self.null_space_basis = None
        self.svd_threshold = config.get("svd_threshold", 1e-5)

    def _get_null_space_basis(self, geom):
        if self.constraints_obj is None:
            return np.eye(len(geom) * 3)
            
        geom_reshaped = geom.reshape(-1, 3)
        B_mat = self.constraints_obj._get_all_constraint_vectors(geom_reshaped)
        
        if B_mat is None or len(B_mat) == 0:
            return np.eye(len(geom) * 3)

        norms = np.linalg.norm(B_mat, axis=1)
        norms[norms < 1e-12] = 1.0 
        B_mat_normalized = B_mat / norms[:, np.newaxis]

        try:
            U, S, Vt = scipy.linalg.svd(B_mat_normalized.T, full_matrices=True)
            max_s = S[0] if len(S) > 0 else 1.0
            threshold = max(self.svd_threshold, max_s * 1e-6)
            rank = np.sum(S > threshold)
            null_space_basis = U[:, rank:]
            
            if null_space_basis.shape[1] == 0:
                self.log("Warning: System is fully constrained.", force=True)
                return np.zeros((len(geom)*3, 0))

            return null_space_basis
            
        except np.linalg.LinAlgError:
            return np.eye(len(geom) * 3)

    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        self.log(f"\n{'='*50}\nCRS-I-RFO Iteration {self.iteration}\n{'='*50}", force=True)
        
        if self.Initialization:
            self.prev_eigvec_min = None
            self.prev_eigvec_size = None
            self.predicted_energy_changes = []
            self.actual_energy_changes = []
            self.prev_geometry = None
            self.prev_gradient = None
            self.prev_energy = None
            self.proj_grad_converged = False
            self.iteration = 0
            self.Initialization = False
        
        # --- 0. SHAKE-like Correction & Gradient Transport ---
        gradient_full = np.asarray(B_g).ravel()
        original_shape = geom_num_list.shape
        geom_flat = geom_num_list.ravel()
        
        if self.constraints_obj is not None:
            geom_reshaped = geom_num_list.reshape(-1, 3)
            corrected_geom_3d = self.constraints_obj.adjust_init_coord(geom_reshaped)
            corrected_geom_flat = corrected_geom_3d.ravel()
            
            shake_displacement = corrected_geom_flat - geom_flat
            diff_norm = np.linalg.norm(shake_displacement)
            
            if diff_norm > 1e-6:
                self.log(f"SHAKE Correction: {diff_norm:.6e}", force=True)
                H_eff = self.hessian
                if self.bias_hessian is not None:
                    H_eff += self.bias_hessian
                grad_correction = np.dot(H_eff, shake_displacement)
                gradient_full += grad_correction
            
            geom_num_list = corrected_geom_3d.reshape(original_shape)

        # --- 1. Hessian Update ---
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, g, pre_geom, pre_g)

        hessian_full = self.hessian
        if self.bias_hessian is not None:
             hessian_full += self.bias_hessian
             
        current_energy = B_e

        # --- 2. Projection to Subspace ---
        U = self._get_null_space_basis(geom_num_list.reshape(-1, 3))
        
        if U.shape[1] == 0:
             return np.zeros_like(gradient_full).reshape(-1, 1)

        gradient_sub = np.dot(U.T, gradient_full)
        hessian_sub = np.dot(U.T, np.dot(hessian_full, U))
        
        subspace_dim = len(gradient_sub)
        grad_sub_norm = np.linalg.norm(gradient_sub)
        
        self.log(f"Subspace Dim: {subspace_dim}, Projected Grad Norm: {grad_sub_norm:.6e}", force=True)

        # === CRITICAL FIX: Explicit Convergence Check in Subspace ===
        # If the projected gradient is effectively zero, we are done.
        # Don't try to calculate RFO step, it will be numerically unstable.
        if grad_sub_norm < self.gradient_norm_threshold:
            self.log(f"*** CONVERGED in Subspace (Grad: {grad_sub_norm:.6e}) ***", force=True)
            self.proj_grad_converged = True
            
            # Reset history to clean state
            self.prev_geometry = geom_num_list
            self.prev_gradient = B_g
            self.prev_energy = current_energy
            
            return np.zeros_like(gradient_full).reshape(-1, 1)
        # ============================================================

        # --- 3. RFO in Subspace ---
        hessian_sub = 0.5 * (hessian_sub + hessian_sub.T)
        
        eigvals_sub, eigvecs_sub = self.compute_eigendecomposition_with_shift(hessian_sub)
        
        # Trust Radius
        if not self.Initialization and self.prev_energy is not None:
            actual_energy_change = B_e - self.prev_energy
            if len(self.actual_energy_changes) >= 3:
                self.actual_energy_changes.pop(0)
            self.actual_energy_changes.append(actual_energy_change)
            
            if self.predicted_energy_changes:
                min_eigval = eigvals_sub[0] if len(eigvals_sub) > 0 else None
                self.adjust_trust_radius(
                    actual_energy_change,
                    self.predicted_energy_changes[-1],
                    min_eigval,
                    grad_sub_norm
                )

        P_rfo = np.eye(subspace_dim)
        root_num = 0
        i = 0
        while root_num < len(self.roots) and i < len(eigvals_sub):
            if np.abs(eigvals_sub[i]) > 1e-10:
                trans_vec = eigvecs_sub[:, i]
                if self.NEB_mode:
                    P_rfo -= np.outer(trans_vec, trans_vec)
                else:
                    P_rfo -= 2 * np.outer(trans_vec, trans_vec)
                root_num += 1
            i += 1
            
        H_star_sub = np.dot(P_rfo, hessian_sub)
        H_star_sub = 0.5 * (H_star_sub + H_star_sub.T)
        grad_star_sub = np.dot(P_rfo, gradient_sub)
        
        eigvals_star, eigvecs_star = self.compute_eigendecomposition_with_shift(H_star_sub)
        eigvals_star, eigvecs_star = self.filter_small_eigvals(eigvals_star, eigvecs_star)
        
        step_sub = self.get_rs_step(eigvals_star, eigvecs_star, grad_star_sub)
        
        # --- 4. Reconstruct Step ---
        step_full = np.dot(U, step_sub)
        
        predicted_energy_change = self.rfo_model(gradient_sub, hessian_sub, step_sub)
        
        if len(self.predicted_energy_changes) >= 3:
            self.predicted_energy_changes.pop(0)
        self.predicted_energy_changes.append(predicted_energy_change)
        
        if self.actual_energy_changes and len(self.predicted_energy_changes) > 1:
            self.evaluate_step_quality()
            
        self.prev_geometry = geom_num_list
        self.prev_gradient = B_g
        self.prev_energy = current_energy
        self.iteration += 1
        
        return -1 * step_full.reshape(-1, 1)