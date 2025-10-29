import numpy as np
from scipy.optimize import brentq
from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Optimizer.block_hessian_update import BlockHessianUpdate
import sys


class InternalCoordinates:
    """
    Handles the construction and transformation of Delocalized Internal Coordinates (DIC)
    as described by Baker, Kessi, and Delley (1996).
    """
    def __init__(self, **config):
        self.log_func = config.get("log_func", print)
        self.g_tol = config.get("g_tol", 1e-6)
        # B, U, Lambda_inv are calculated once and stored in the instance
        self.B_prim = None  # Primitive B-Matrix (n x 3N)
        self.U = None       # Active Eigenvectors (n x k)
        self.Lambda_inv = None # Inverse Eigenvalues (k x k)
        self.k = 0          # Number of active coordinates

    def log(self, message, force=False):
        if self.log_func and force: self.log_func(message, force=True)
        elif self.log_func: self.log_func(message)

    def _build_primitive_B_stretches(self, geom_cart_3N):
        """ (Unchanged) Build B-matrix for all stretches (N*(N-1)/2) """
        geom_cart_N3 = geom_cart_3N.reshape(-1, 3); N = geom_cart_N3.shape[0]; M = N * (N - 1) // 2
        if M <= 0: return np.zeros((0, 3*N))
        B_prim_ic = np.zeros((M, 3 * N)); row_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                ri = geom_cart_N3[i, :]; rj = geom_cart_N3[j, :]
                rij_vec = ri - rj; rij_norm = np.linalg.norm(rij_vec)
                s_vec = np.zeros(3)
                if rij_norm > 1e-8: s_vec = rij_vec / rij_norm
                B_prim_ic[row_idx, 3*i : 3*i+3] = s_vec
                B_prim_ic[row_idx, 3*j : 3*j+3] = -s_vec
                row_idx += 1
        return B_prim_ic

    def _build_primitive_B_bends(self, geom_cart_3N):
        """
        【TODO】Build B-matrix for all planar bends
        Calculate B-matrix for all i-j-k tuples based on molecular connectivity.
        """
        N_atoms = len(geom_cart_3N) // 3
        self.log("Warning: _build_primitive_B_bends is not implemented.", force=True)
        # Placeholder:
        return np.zeros((0, 3 * N_atoms))

    def _build_primitive_B_torsions(self, geom_cart_3N):
        """
        【TODO】Build B-matrix for all proper torsions
        Calculate B-matrix for all i-j-k-l tuples based on molecular connectivity.
        """
        N_atoms = len(geom_cart_3N) // 3
        self.log("Warning: _build_primitive_B_torsions is not implemented.", force=True)
        # Placeholder:
        return np.zeros((0, 3 * N_atoms))

    def build_active_subspace(self, geom_cart_3N):
        """
        Build B-matrix from stretches, bends, torsions based on the paper.
        Removed Cartesian coordinates (B_CC).
        """
        N_atoms = len(geom_cart_3N) // 3
        if N_atoms == 0:
            self.log("Warning: No atoms.", force=True); self.k=0; self.B_prim=np.zeros((0,0)); self.U=np.zeros((0,0)); self.Lambda_inv=np.zeros((0,0)); return 0

        # Stack primitive internal coordinates according to the paper
        B_Stretches = self._build_primitive_B_stretches(geom_cart_3N)
        B_Bends = self._build_primitive_B_bends(geom_cart_3N)
        B_Torsions = self._build_primitive_B_torsions(geom_cart_3N)
        
        # Remove B_CC (Cartesian) and construct B_prim using only internal coordinates
        self.B_prim = np.vstack((B_Stretches, B_Bends, B_Torsions))
        
        if self.B_prim.shape[0] == 0:
             self.log("FATAL: Primitive B-Matrix is empty. (Bends/Torsions not implemented?)", force=True)
             # Fallback to Cartesian (differs from the paper's intent)
             # self.B_prim = np.eye(3 * N_atoms)
             raise ValueError("Primitive B-Matrix is empty. Implement Bends and Torsions.")

        M_total, N_cart = self.B_prim.shape
        self.log(f"Building G-Matrix ({M_total} x {M_total}) from {M_total} primitives...")
        
        # G = B * B.T
        G = np.dot(self.B_prim, self.B_prim.T); G = 0.5 * (G + G.T)
        
        self.log("Diagonalizing G-Matrix...")
        try:
            eigvals_g, U_g = np.linalg.eigh(G)
        except np.linalg.LinAlgError as e:
            self.log(f"FATAL: G-matrix eigh failed: {e}. Check geom.", force=True); raise

        active_indices = eigvals_g > self.g_tol; k = np.sum(active_indices)
        
        # According to the paper, k should be 3N-6 (linear) or 3N-5 (non-linear)
        expected_k = 3*N_atoms
        if N_atoms == 1: expected_k = 3
        elif N_atoms > 1: expected_k = 3*N_atoms - 6 # (Assuming non-linear for simplicity)
        if expected_k < 1: expected_k = 1
        
        if k == 0:
            self.log("No active coords! Check g_tol/geom. Forcing k=1.", force=True); k = 1
            active_indices = np.array([G.shape[0] - 1])
            
        self.log(f"DIC: Found {k} active coordinates (Expected ~{expected_k})")
        
        self.k = k
        self.U = U_g[:, active_indices] # (n x k)
        active_eigvals = eigvals_g[active_indices]

        if np.any(active_eigvals <= 0):
             num_neg = np.sum(active_eigvals <= 0); self.log(f"Warning: {num_neg} non-positive active eigvals. Clamping.", force=True)
             active_eigvals[active_eigvals <= 0] = 1e-12
             
        self.Lambda_inv = np.diag(1.0 / active_eigvals) # (k x k)
        return k

    # ===================================================================
    # Coordinate Transformation Methods
    # ===================================================================

    def project_cart_to_dic(self, vec_cart_3N):
        """
        Projects a Cartesian vector (3N,) to a DIC vector (k,).
        g_q = T @ g_x = (Lambda_inv @ U.T @ B_prim) @ g_x
        """
        if self.B_prim is None: raise ValueError("Coordinate system not built.")
        # 1. B_prim @ g_x
        vec_prim = np.dot(self.B_prim, vec_cart_3N)
        # 2. U.T @ (B_prim @ g_x)
        vec_dic_temp = np.dot(self.U.T, vec_prim)
        # 3. Lambda_inv @ (U.T @ B_prim @ g_x)
        vec_dic = np.dot(self.Lambda_inv, vec_dic_temp)
        return vec_dic

    def back_transform_dic_to_cart(self, vec_dic_k):
        """
        Back-transforms a DIC vector (k,) to a Cartesian vector (3N,).
        dx = T_dagger @ dq = (B_prim.T @ U @ Lambda_inv) @ dq
        """
        if self.B_prim is None: raise ValueError("Coordinate system not built.")
        # 1. Lambda_inv @ dq
        tmp_vec = np.dot(self.Lambda_inv, vec_dic_k)
        # 2. U @ (Lambda_inv @ dq)
        vec_prim = np.dot(self.U, tmp_vec)
        # 3. B_prim.T @ (U @ Lambda_inv @ dq)
        vec_cart_3N = np.dot(self.B_prim.T, vec_prim)
        return vec_cart_3N

    def transform_hessian_cart_to_dic(self, H_cart_3N):
        """
        Transforms a Cartesian Hessian (3N x 3N) to a DIC Hessian (k x k).
        H_q = T @ H_x @ T_dagger = (Lambda_inv @ U.T @ B_prim) @ H_x @ (B_prim.T @ U @ Lambda_inv)
        """
        if self.B_prim is None: raise ValueError("Coordinate system not built.")
        k = self.k
        
        if H_cart_3N is None:
            self.log("No Cartesian Hessian provided, initializing DIC Hessian as Identity.", force=True)
            return np.eye(k)
            
        dim_cart = H_cart_3N.shape[0]
        if dim_cart != self.B_prim.shape[1]:
            self.log(f"ERROR: Cartesian Hessian dimension ({dim_cart}) mismatch with B_prim ({self.B_prim.shape[1]}). Using Identity.", force=True)
            return np.eye(k)

        self.log("Transforming Cartesian Hessian to DIC...", force=True)
        try:
            # T = Lambda_inv @ U.T @ B_prim
            T_part1 = np.dot(self.U.T, self.B_prim) # (k x 3N)
            T = np.dot(self.Lambda_inv, T_part1)  # (k x 3N)
            
            # T_dagger = B_prim.T @ U @ Lambda_inv
            T_dagger_part1 = np.dot(self.B_prim.T, self.U) # (3N x k)
            T_dagger = np.dot(T_dagger_part1, self.Lambda_inv) # (3N x k)
            
            # H_q = T @ H_cart @ T_dagger
            H_q_temp = np.dot(T, H_cart_3N) # (k x 3N)
            H_q = np.dot(H_q_temp, T_dagger) # (k x k)
            
            H_q = 0.5 * (H_q + H_q.T) # Ensure symmetry
            self.log("Transformation complete.", force=True)
        except Exception as e:
            self.log(f"WARNING: Cartesian Hessian to DIC transformation failed: {e}. Using Identity.", force=True)
            H_q = np.eye(k)

        return H_q

# ===================================================================
# Modified DIC_RSIRFO Class
# ===================================================================

class DIC_RSIRFO:
    def __init__(self, **config):
        """
        Delocalized Internal Coordinates (DIC) RS-I-RFO Optimizer.
        """
        # --- Common RSIRFO configuration ---
        self.alpha0 = config.get("alpha0", 1.0)
        self.max_micro_cycles = config.get("max_micro_cycles", 40)
        self.saddle_order = config.get("saddle_order", 1)
        self.hessian_update_method = config.get("method", "auto")
        self.small_eigval_thresh = config.get("small_eigval_thresh", 1e-6)
        self.alpha_max = config.get("alpha_max", 1e6)
        self.alpha_step_max = config.get("alpha_step_max", 10.0)
        if self.saddle_order == 0:
            self.trust_radius_initial = config.get("trust_radius", 0.5)
            self.trust_radius_max = config.get("trust_radius_max", 0.5)
        else:
            self.trust_radius_initial = config.get("trust_radius", 0.1)
            self.trust_radius_max = config.get("trust_radius_max", 0.1)
        self.trust_radius = self.trust_radius_initial
        self.trust_radius_min = config.get("trust_radius_min", 0.01)
        self.good_step_threshold = config.get("good_step_threshold", 0.75)
        self.poor_step_threshold = config.get("poor_step_threshold", 0.25)
        self.trust_radius_increase_factor = config.get("trust_radius_increase_factor", 1.2)
        self.trust_radius_decrease_factor = config.get("trust_radius_decrease_factor", 0.5)
        self.energy_change_threshold = config.get("energy_change_threshold", 1e-6)
        self.gradient_norm_threshold = config.get("gradient_norm_threshold", 1e-4)
        self.step_norm_tolerance = config.get("step_norm_tolerance", 1e-3)
        self.debug_mode = config.get("debug_mode", False)
        self.display_flag = config.get("display_flag", True)
        self.Initialization = True
        
        # --- Hessian storage (Cartesian) ---
        self.hessian = None          # Stores the INITIAL/CURRENT Cartesian Hessian (3N x 3N)
        self.bias_hessian = None     # Stores the Cartesian Bias Hessian (3N x 3N)
        
        # --- DIC-specific storage ---
        self.dic_hessian = None      # The operational Hessian in DIC space (k x k)
        self.dic_bias_hessian = None # The operational Bias Hessian in DIC space (k x k)

        # --- State variables ---
        self.prev_eigvec_min = None
        self.prev_eigvec_size = None
        self.predicted_energy_changes = []
        self.actual_energy_changes = []
        self.prev_geometry = None # Cartesian
        self.prev_gradient = None # Cartesian
        self.prev_energy = None
        self.converged = False
        self.iteration = 0
        self.roots = list(range(self.saddle_order))
        
        # --- Updaters and Helpers ---
        self.hessian_updater = ModelHessianUpdate()
        self.block_hessian_updater = BlockHessianUpdate()
        self.alpha_init_values = [0.001 + (10.0 - 0.001) * i / 14 for i in range(15)]
        self.NEB_mode = False
        
        # --- DIC coordinate system ---
        config["log_func"] = self.log
        self.coord_system = InternalCoordinates(**config) # Instance to hold the coordinate system

    def log(self, message, force=False):
        display = getattr(self, 'display_flag', True)
        debug = getattr(self, 'debug_mode', False)
        if display and (force or debug):
            print(message)

    # === Main Methods ===

    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        self.log(f"\n{'='*50}\nDIC-RS-I-RFO Iteration {self.iteration}\n{'='*50}", force=True)
        
        geom_cart_3N = np.asarray(geom_num_list).ravel()
        g_cart_for_step = np.asarray(B_g).ravel()

        # --- 1. Build Coordinate System (Once) ---
        try:
            # Build coordinate system only on the first step
            if self.coord_system.B_prim is None or self.Initialization:
                self.log("Building DIC coordinate system (first step)...", force=True)
                k = self.coord_system.build_active_subspace(geom_cart_3N)
                if k <= 0: raise ValueError("Invalid number of active coordinates (k<=0).")
                
                # Initialize DIC Hessian
                self.dic_hessian = self.coord_system.transform_hessian_cart_to_dic(self.hessian)
                
                self.Initialization = False
                self.predicted_energy_changes = []; self.actual_energy_changes = []
                self.converged = False; self.iteration = 0
            else:
                k = self.coord_system.k # Use existing coordinate system
                
                # --- 2. Hessian Update (Subsequent steps) ---
                if self.prev_geometry is not None and len(pre_g) > 0 and len(pre_geom) > 0:
                    g_cart_for_update = np.asarray(g).ravel()
                    pre_g_cart_for_update = np.asarray(pre_g).ravel()
                    
                    self.update_hessian(geom_cart_3N, g_cart_for_update,
                                        np.asarray(pre_geom).ravel(), pre_g_cart_for_update)
                
                # --- Trust Radius Update ---
                if self.prev_energy is not None:
                    actual_energy_change = B_e - self.prev_energy
                    if len(self.actual_energy_changes) >= 3: self.actual_energy_changes.pop(0)
                    self.actual_energy_changes.append(actual_energy_change)
                    if self.predicted_energy_changes:
                        self.adjust_trust_radius(actual_energy_change, self.predicted_energy_changes[-1])

        except Exception as e:
            self.log(f"FATAL: DIC coordinate generation/update failed: {e}", force=True)
            self.log("Aborting optimization.", force=True); self.converged = True
            return np.zeros_like(geom_num_list).reshape(-1, 1)

        # --- Bias Hessian Prep ---
        if self.bias_hessian is not None and self.dic_bias_hessian is None:
            self.log("Transforming Cartesian Bias Hessian to DIC...")
            self.dic_bias_hessian = self.coord_system.transform_hessian_cart_to_dic(self.bias_hessian)
            if self.dic_bias_hessian.shape[0] != k:
                self.log(f"Warn: Transformed Bias shape != k. Ignoring.", force=True)
                self.dic_bias_hessian = None
        elif self.bias_hessian is None:
             self.dic_bias_hessian = None

        # --- 3. Convergence Check (Cartesian) ---
        gradient_norm = np.linalg.norm(g_cart_for_step)
        self.log(f"Gradient norm (Cartesian): {gradient_norm:.6f}", force=True)
        if gradient_norm < self.gradient_norm_threshold:
            self.log(f"Converged: Gradient norm {gradient_norm:.6f} < {self.gradient_norm_threshold:.6f}", force=True)
            self.converged = True
        
        if self.actual_energy_changes:
            last_energy_change = abs(self.actual_energy_changes[-1])
            if last_energy_change < self.energy_change_threshold:
                self.log(f"Converged: Energy change {last_energy_change:.6f} < {self.energy_change_threshold:.6f}", force=True)
                self.converged = True
                
        if self.converged:
            return np.zeros_like(geom_num_list).reshape(-1, 1)

        # --- 4. RFO Step (in DIC space) ---
        
        # g_q = T @ g_x
        g_q = self.coord_system.project_cart_to_dic(g_cart_for_step)
        
        H_q = self.dic_hessian
        
        if self.dic_bias_hessian is not None:
            if H_q.shape == self.dic_bias_hessian.shape:
                H_q = H_q + self.dic_bias_hessian
            else:
                self.log(f"Warn: DIC Bias shape mismatch. Ignoring.", force=True)

        H_q = 0.5 * (H_q + H_q.T)
        
        try:
            eigvals_q, eigvecs_q = np.linalg.eigh(H_q)
        except np.linalg.LinAlgError:
            self.log("FATAL: DIC Hessian diagonalization failed. Using Identity.", force=True)
            H_q = np.eye(k); self.dic_hessian = H_q
            eigvals_q, eigvecs_q = np.linalg.eigh(H_q)


        if self.display_flag:
            self.log(f"--- DIC Hessian Eigenvalues (k={k}) ---", force=True)
            chunk_size = 6
            for i in range(0, k, chunk_size):
                chunk = eigvals_q[i:i + chunk_size]
                line_str = " ".join([f"{val:10.6f}" for val in chunk])
                self.log(f"  {line_str}", force=True)
            self.log(f"----------------------------------------", force=True)
        


        neg_eigvals = np.sum(eigvals_q < -1e-10)
        self.log(f"Found {neg_eigvals} negative eigenvalues in DIC Hessian (target: {self.saddle_order})", force=True)



        # (RFO/Image Projection logic - Unchanged)
        P_q = np.eye(k)
        root_num = 0; i = 0
        while root_num < len(self.roots) and i < k:
              if i < len(eigvals_q) and np.abs(eigvals_q[i]) > 1e-10:
                  trans_vec_q = eigvecs_q[:, i]
                  if self.NEB_mode: P_q -= np.outer(trans_vec_q, trans_vec_q)
                  else: P_q -= 2 * np.outer(trans_vec_q, trans_vec_q)
                  root_num += 1
              elif i >= len(eigvals_q): self.log(f"Warn: Index i={i} out of bounds for eigvals_q ({len(eigvals_q)}).", force=True); break
              i += 1
              
        H_q_star = np.dot(P_q, H_q); H_q_star = 0.5 * (H_q_star + H_q_star.T)
        g_q_star = np.dot(P_q, g_q)
        eigvals_q_star, eigvecs_q_star = np.linalg.eigh(H_q_star)
        eigvals_q_star_filt, eigvecs_q_star_filt = self.filter_small_eigvals(eigvals_q_star, eigvecs_q_star)
        
        current_eigvec_size = eigvecs_q_star_filt.shape[1]
        if current_eigvec_size == 0:
            self.log("ERROR: No eigenvalues after filtering. Using steepest descent.", force=True)
            step_q = -g_q
            step_norm_q = np.linalg.norm(step_q)
            if step_norm_q > self.trust_radius: step_q *= self.trust_radius / step_norm_q
        else:
            self.log(f"Using {current_eigvec_size} eigenvalues/vectors after filtering")
            if self.prev_eigvec_size is not None and self.prev_eigvec_size != current_eigvec_size:
                self.log(f"Resetting prev eigvec info (dim change: {self.prev_eigvec_size} -> {current_eigvec_size})")
                self.prev_eigvec_min = None

            g_q_star_in_filt_basis = np.dot(eigvecs_q_star_filt.T, g_q_star)
            step_q_filt_trans = self.get_rs_step(eigvals_q_star_filt, g_q_star_in_filt_basis)
            step_q = np.dot(eigvecs_q_star_filt, step_q_filt_trans)
            self.prev_eigvec_size = current_eigvec_size

        # --- 5. Back-transform & Save State ---
        
        # dx = T_dagger @ dq
        move_vector_cart_3N = self.coord_system.back_transform_dic_to_cart(step_q)
        
        step_norm_cart = np.linalg.norm(move_vector_cart_3N)
        max_step_allowed = 2.0 * self.trust_radius_max
        if step_norm_cart > max_step_allowed:
            self.log(f"Warn: Cart step norm {step_norm_cart:.4f} > limit {max_step_allowed:.4f}. Scaling.", force=True)
            move_vector_cart_3N *= max_step_allowed / step_norm_cart

        predicted_energy_change = self.rfo_model(g_q, H_q, step_q)

        if len(self.predicted_energy_changes) >= 3: self.predicted_energy_changes.pop(0)
        self.predicted_energy_changes.append(predicted_energy_change)
        self.log(f"Predicted energy change (DIC): {predicted_energy_change:.6f}", force=True)

        self.prev_geometry = geom_cart_3N
        self.prev_gradient = g_cart_for_step
        self.prev_energy = B_e
        self.iteration += 1
        
        return -1 * move_vector_cart_3N.reshape(-1, 1)

    def update_hessian(self, current_geom_cart, current_grad_cart, previous_geom_cart, previous_grad_cart):
        """
        Updates the DIC Hessian (self.dic_hessian).
        Uses the *fixed* coordinate system (self.coord_system).
        """
        
        # Check if coordinate system is built
        if self.coord_system.B_prim is None or self.dic_hessian is None:
            self.log("Warning: Coordinate system or DIC Hessian not ready. Skipping Hessian update.")
            return

        displacement_cart = np.asarray(current_geom_cart - previous_geom_cart).ravel()
        delta_grad_cart = np.asarray(current_grad_cart - previous_grad_cart).ravel()

        # 【P2 Fix】Use correct projection (T @ dx)
        displacement_q = self.coord_system.project_cart_to_dic(displacement_cart)
        delta_grad_q = self.coord_system.project_cart_to_dic(delta_grad_cart)
        
        disp_norm = np.linalg.norm(displacement_q)
        grad_diff_norm = np.linalg.norm(delta_grad_q)
        if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
            self.log("Skipping Hessian update (DIC changes too small)")
            return
            
        dot_product_q = np.dot(displacement_q, delta_grad_q)
        if dot_product_q <= 1e-8:
            self.log(f"Skipping Hessian update (DIC poor alignment: dot={dot_product_q:.2e})")
            return
            
        self.log(f"Hessian update (DIC): disp_norm={disp_norm:.6f}, grad_diff_norm={grad_diff_norm:.6f}, dot={dot_product_q:.6f}")
        
        displacement_q = displacement_q.reshape(-1, 1)
        delta_grad_q = delta_grad_q.reshape(-1, 1)

        try:
            # --- (This part is unchanged) Select and call the update method ---
            method_name = self.hessian_update_method.lower()
            delta_hess = None

            if "flowchart" in method_name:
                # ... (Logic below is same as original update_hessian) ...
                self.log(f"Hessian update method: flowchart")
                delta_hess = self.hessian_updater.flowchart_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q, "auto"
                )
            elif "block_cfd_fsb" in method_name:
                self.log(f"Hessian update method: block_cfd_fsb")
                delta_hess = self.block_hessian_updater.block_CFD_FSB_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "block_cfd_bofill" in method_name:
                self.log(f"Hessian update method: block_cfd_bofill")
                delta_hess = self.block_hessian_updater.block_CFD_Bofill_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "block_bfgs" in method_name:
                self.log(f"Hessian update method: block_bfgs")
                delta_hess = self.block_hessian_updater.block_BFGS_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "block_fsb" in method_name:
                self.log(f"Hessian update method: block_fsb")
                delta_hess = self.block_hessian_updater.block_FSB_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "block_bofill" in method_name:
                self.log(f"Hessian update method: block_bofill")
                delta_hess = self.block_hessian_updater.block_Bofill_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "bfgs" in method_name:
                self.log(f"Hessian update method: bfgs")
                delta_hess = self.hessian_updater.BFGS_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "sr1" in method_name:
                self.log(f"Hessian update method: sr1")
                delta_hess = self.hessian_updater.SR1_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "pcfd_bofill" in method_name:
                self.log(f"Hessian update method: pcfd_bofill")
                delta_hess = self.hessian_updater.pCFD_Bofill_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "cfd_fsb" in method_name:
                self.log(f"Hessian update method: cfd_fsb")
                delta_hess = self.hessian_updater.CFD_FSB_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "cfd_bofill" in method_name:
                self.log(f"Hessian update method: cfd_bofill")
                delta_hess = self.hessian_updater.CFD_Bofill_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "fsb" in method_name:
                self.log(f"Hessian update method: fsb")
                delta_hess = self.hessian_updater.FSB_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "bofill" in method_name:
                self.log(f"Hessian update method: bofill")
                delta_hess = self.hessian_updater.Bofill_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "psb" in method_name:
                self.log(f"Hessian update method: psb")
                delta_hess = self.hessian_updater.PSB_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            elif "msp" in method_name:
                self.log(f"Hessian update method: msp")
                delta_hess = self.hessian_updater.MSP_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q
                )
            else: # Default fallback
                self.log(f"Unknown Hessian update method: '{self.hessian_update_method}'. Using flowchart/auto.")
                delta_hess = self.hessian_updater.flowchart_hessian_update(
                    self.dic_hessian, displacement_q, delta_grad_q, "auto"
                )

            # --- Apply the update ---
            if delta_hess is None:
                self.log("Error: delta_hess is None. Skipping update.", force=True)
            elif not np.all(np.isfinite(delta_hess)):
                self.log("Warning: Hessian update resulted in non-finite values. Skipping update.", force=True)
            else:
                self.dic_hessian += delta_hess
                self.dic_hessian = 0.5 * (self.dic_hessian + self.dic_hessian.T)
        
        except Exception as e:
            self.log(f"Error during Hessian update method call for '{method_name}': {e}. Skipping update.")
            if self.debug_mode: raise e

    # --- (Helper methods below are unchanged) ---
    # (Unchanged helper methods are omitted for brevity)
    # (Please copy from the original code)

    def set_hessian(self, hessian_cart):
        self.log("Cartesian Hessian received.")
        if hessian_cart is not None:
            self.hessian = np.asarray(hessian_cart)
        else:
            self.hessian = None
        # DIC Hessian will be transformed in run() (if needed)
        self.dic_hessian = None 
        # Reset coordinate system as well
        self.coord_system = InternalCoordinates(**self.coord_system.__dict__)
        self.Initialization = True
        return

    def get_hessian(self):
        return self.hessian 

    def set_bias_hessian(self, bias_hessian_cart):
        self.log("Cartesian Bias Hessian received.")
        if bias_hessian_cart is not None:
            self.bias_hessian = np.asarray(bias_hessian_cart)
        else:
            self.bias_hessian = None
        self.dic_bias_hessian = None # Recalculated in run()
        return

    def get_bias_hessian(self):
        return self.bias_hessian

    # (Copy from original code)
    def switch_NEB_mode(self):
        if self.NEB_mode: self.NEB_mode = False
        else: self.NEB_mode = True
        
    def filter_small_eigvals(self, eigvals, eigvecs, mask=False):
        small_inds = np.abs(eigvals) < self.small_eigval_thresh
        small_num = np.sum(small_inds)
        
        if small_num > 0:
            self.log(f"Found {small_num} small eigenvalues. Removed corresponding.")
            
        filtered_eigvals = eigvals[~small_inds]
        filtered_eigvecs = eigvecs[:, ~small_inds]
        
        if small_num > 6 and eigvals.shape[0] > 10: # Only warn if DoF is large
            self.log(f"Warning: Found {small_num} small eigenvalues (>6).", force=True)
        
        if mask:
            return filtered_eigvals, filtered_eigvecs, small_inds
        else:
            return filtered_eigvals, filtered_eigvecs
            
    def adjust_trust_radius(self, actual_change, predicted_change):
        if abs(predicted_change) < 1e-10:
            self.log("Skipping trust radius update: predicted change too small")
            return
        # Avoid division by zero if actual_change is also tiny
        if abs(actual_change) < 1e-10:
            ratio = 1.0 # Treat as perfect agreement
        else:
            ratio = actual_change / predicted_change
            
        self.log(f"Energy change: actual={actual_change:.6f}, predicted={predicted_change:.6f}, ratio={ratio:.3f}", force=True)
        old_trust_radius = self.trust_radius
        
        if ratio > self.good_step_threshold:
            self.trust_radius = min(self.trust_radius * self.trust_radius_increase_factor, 
                                    self.trust_radius_max)
            if self.trust_radius != old_trust_radius:
                self.log(f"Good step (ratio={ratio:.3f}), increasing trust radius to {self.trust_radius:.6f}", force=True)
        elif ratio < self.poor_step_threshold:
            self.trust_radius = max(self.trust_radius * self.trust_radius_decrease_factor, 
                                    self.trust_radius_min)
            if self.trust_radius != old_trust_radius:
                self.log(f"Poor step (ratio={ratio:.3f}), decreasing trust radius to {self.trust_radius:.6f}", force=True)
        else:
            self.log(f"Acceptable step (ratio={ratio:.3f}), keeping trust radius at {self.trust_radius:.6f}", force=True)

    def evaluate_step_quality(self):
        if len(self.predicted_energy_changes) < 2 or len(self.actual_energy_changes) < 2:
            return "unknown"
        ratios = []
        for actual, predicted in zip(self.actual_energy_changes[-2:], self.predicted_energy_changes[-2:]):
            if abs(predicted) > 1e-10:
                if abs(actual) < 1e-10: # Handle tiny actual change
                    ratios.append(1.0)
                else:
                    ratios.append(actual / predicted)
        if not ratios: return "unknown"
        avg_ratio = sum(ratios) / len(ratios)
        # Check if steps generally decrease energy (allow small positive actual changes)
        generally_downhill = all(a < 1e-6 or (a > 0 and abs(a/p) < 0.1)
                                    for a, p in zip(self.actual_energy_changes[-2:], self.predicted_energy_changes[-2:]) if abs(p) > 1e-10)

        if 0.8 < avg_ratio < 1.2 and generally_downhill: quality = "good"
        elif 0.5 < avg_ratio < 1.5 and generally_downhill: quality = "acceptable"
        else: quality = "poor"
        self.log(f"Step quality assessment: {quality} (avg ratio: {avg_ratio:.3f})", force=True)
        return quality

    def get_rs_step(self, eigvals, gradient_trans):
        try:
            initial_step, _, _, _ = self.solve_rfo(eigvals, gradient_trans, self.alpha0)
            initial_step_norm = np.linalg.norm(initial_step)
            self.log(f"Initial step with alpha={self.alpha0:.6f} has norm={initial_step_norm:.6f}", force=True)
            
            if initial_step_norm <= self.trust_radius:
                self.log(f"Initial step is within trust radius ({self.trust_radius:.6f}), using it directly", force=True)
                return initial_step # Return the step in the eigenvector basis
                
            self.log(f"Initial step exceeds trust radius, optimizing alpha...", force=True)
        except Exception as e:
            self.log(f"Error calculating initial step: {str(e)}", force=True)
            
        best_overall_step = None
        best_overall_norm_diff = float('inf')
        best_alpha_value = None
        
        self.log(f"Trying {len(self.alpha_init_values)} different initial alpha values:", force=True)
        
        for trial_idx, alpha_init in enumerate(self.alpha_init_values):
            if self.debug_mode:
                self.log(f"\n--- Alpha Trial {trial_idx+1}/{len(self.alpha_init_values)}: alpha_init = {alpha_init:.6f} ---")
            
            try:
                step_, step_norm, alpha_final = self.compute_rsprfo_step(
                    eigvals, gradient_trans, alpha_init
                )
                norm_diff = abs(step_norm - self.trust_radius)
                
                if self.debug_mode:
                    self.log(f"Alpha trial {trial_idx+1}: ... step_norm={step_norm:.6f}, diff={norm_diff:.6f}")
                
                is_better = False
                if best_overall_step is None: is_better = True
                elif step_norm <= self.trust_radius and best_overall_norm_diff > self.trust_radius: is_better = True
                elif (step_norm <= self.trust_radius) == (best_overall_norm_diff <= self.trust_radius):
                    if norm_diff < best_overall_norm_diff: is_better = True
                        
                if is_better:
                    best_overall_step = step_
                    best_overall_norm_diff = norm_diff
                    best_alpha_value = alpha_init
                
            except Exception as e:
                if self.debug_mode:
                    self.log(f"Error in alpha trial {trial_idx+1}: {str(e)}")
                
        if best_overall_step is None:
            self.log("All alpha trials failed, using steepest descent step as fallback", force=True)
            sd_step = -gradient_trans
            sd_norm = np.linalg.norm(sd_step)
            best_overall_step = sd_step / sd_norm * self.trust_radius if sd_norm > self.trust_radius else sd_step
        else:
            self.log(f"Selected alpha value: {best_alpha_value:.6f}", force=True)
            
        step = best_overall_step
        step_norm = np.linalg.norm(step)
        self.log(f"Final norm(step)={step_norm:.6f}", force=True)
        
        return step # Return the step in the eigenvector basis

    def compute_rsprfo_step(self, eigvals, gradient_trans, alpha_init):
        def calculate_step(alpha):
            try:
                step, eigval_min, _, _ = self.solve_rfo(eigvals, gradient_trans, alpha)
                return step, eigval_min
            except Exception as e:
                self.log(f"Error in step calculation: {str(e)}")
                raise
        def step_norm_squared(alpha):
            step, _ = calculate_step(alpha)
            return np.dot(step, step)
        def objective_function(alpha):
            return step_norm_squared(alpha) - self.trust_radius**2

        alpha_lo, alpha_hi = 1e-6, self.alpha_max
        try:
            step_lo, _ = calculate_step(alpha_lo)
            obj_lo = np.dot(step_lo, step_lo) - self.trust_radius**2
            step_hi, _ = calculate_step(alpha_hi) 
            obj_hi = np.dot(step_hi, step_hi) - self.trust_radius**2
            
            self.log(f"Bracket search: alpha_lo={alpha_lo:.6e}, obj={obj_lo:.6e}")
            self.log(f"Bracket search: alpha_hi={alpha_hi:.6e}, obj={obj_hi:.6e}")
            
            if obj_lo * obj_hi >= 0:
                self.log("Could not establish bracket, proceeding with Newton")
                alpha = alpha_init
            else:
                self.log("Bracket established, using Brent's method")
                try:
                    alpha = brentq(objective_function, alpha_lo, alpha_hi, 
                                   xtol=1e-6, rtol=1e-6, maxiter=50)
                    self.log(f"Brent's method converged to alpha={alpha:.6e}")
                except Exception as e:
                    self.log(f"Brent's method failed: {str(e)}, using initial alpha")
                    alpha = alpha_init
        except Exception as e:
            self.log(f"Error establishing bracket: {str(e)}, using initial alpha")
            alpha = alpha_init
            
        alpha = alpha_init if 'alpha' not in locals() else alpha
        step_norm_history = np.zeros(self.max_micro_cycles)
        history_count = 0
        best_step = None
        best_step_norm_diff = float('inf')
        alpha_left, alpha_right = None, None
        
        for mu in range(self.max_micro_cycles):
            self.log(f"RS-I-RFO micro cycle {mu:02d}, alpha={alpha:.6f}")
            try:
                step, eigval_min = calculate_step(alpha)
                step_norm = np.linalg.norm(step)
                self.log(f"norm(step)={step_norm:.6f}")
                
                norm_diff = abs(step_norm - self.trust_radius)
                if norm_diff < best_step_norm_diff:
                    if best_step is None:
                        best_step = step.copy()
                    else:
                        best_step = np.copyto(best_step, step)
                    best_step_norm_diff = norm_diff
                
                objval = step_norm**2 - self.trust_radius**2
                self.log(f"U(a)={objval:.6e}")
                
                if objval < 0 and (alpha_left is None or alpha > alpha_left): alpha_left = alpha
                elif objval > 0 and (alpha_right is None or alpha < alpha_right): alpha_right = alpha
                
                if abs(objval) < 1e-8 or norm_diff < self.step_norm_tolerance:
                    self.log(f"Step norm {step_norm:.6f} is sufficiently close to trust radius")
                    if mu >= 1: break
                
                if history_count < self.max_micro_cycles:
                    step_norm_history[history_count] = step_norm
                    history_count += 1
                
                dstep2_dalpha = self.get_step_derivative(alpha, eigvals, gradient_trans, 
                                                        step=step, eigval_min=eigval_min)
                self.log(f"d(||step||^2)/dα={dstep2_dalpha:.6e}")
                
                if abs(dstep2_dalpha) < 1e-10:
                    if alpha_left is not None and alpha_right is not None:
                        alpha_new = (alpha_left + alpha_right) / 2
                        self.log(f"Small derivative, using bisection")
                    else:
                        alpha_new = max(alpha / 2, 1e-6) if objval > 0 else min(alpha * 2, self.alpha_max)
                        self.log(f"Small derivative, no bracket, using heuristic")
                else:
                    alpha_step_raw = -objval / dstep2_dalpha
                    alpha_step = np.clip(alpha_step_raw, -self.alpha_step_max, self.alpha_step_max)
                    if abs(alpha_step) != abs(alpha_step_raw):
                        self.log(f"Limited alpha step from {alpha_step_raw:.6f} to {alpha_step:.6f}")
                    alpha_new = alpha + alpha_step
                    if alpha_left is not None and alpha_right is not None:
                        alpha_new = max(min(alpha_new, alpha_right * 0.99), alpha_left * 1.01)
                
                old_alpha = alpha
                alpha = min(max(alpha_new, 1e-6), self.alpha_max)
                self.log(f"Updated alpha: {old_alpha:.6f} -> {alpha:.6f}")
                
                if alpha == self.alpha_max or alpha == 1e-6:
                    self.log(f"Alpha hit boundary at {alpha:.6e}, stopping iterations")
                    break
                
                if history_count >= 3:
                    idx = history_count - 1
                    changes = [abs(step_norm_history[idx] - step_norm_history[idx-1]),
                               abs(step_norm_history[idx-1] - step_norm_history[idx-2])]
                    if all(c < 1e-6 for c in changes):
                        self.log("Step norm not changing significantly, stopping iterations")
                        break
            except Exception as e:
                self.log(f"Error in micro-cycle {mu}: {str(e)}")
                if best_step is not None:
                    self.log("Using best step found so far due to error")
                    step, step_norm = best_step, np.linalg.norm(best_step)
                    break
                else:
                    self.log("Falling back to steepest descent due to errors")
                    step = -gradient_trans
                    step_norm = np.linalg.norm(step)
                    if step_norm > self.trust_radius:
                        step = step / step_norm * self.trust_radius
                        step_norm = self.trust_radius
                    break
        else:
            self.log(f"RS-I-RFO did not converge in {self.max_micro_cycles} cycles")
            if best_step is not None:
                self.log("Using best step found during iterations")
                step, step_norm = best_step, np.linalg.norm(best_step)
        
        return step, step_norm, alpha

    def get_step_derivative(self, alpha, eigvals, gradient_trans, step=None, eigval_min=None):
        if step is None or eigval_min is None:
            try:
                step, eigval_min, _, _ = self.solve_rfo(eigvals, gradient_trans, alpha)
            except Exception as e:
                self.log(f"Error in step calculation for derivative: {str(e)}")
                return 1e-8
        
        try:
            denominators = eigvals - eigval_min * alpha
            small_denoms = np.abs(denominators) < 1e-8
            if np.any(small_denoms):
                safe_denoms = denominators.copy()
                safe_denoms[small_denoms] = np.sign(safe_denoms[small_denoms]) * np.maximum(1e-8, np.abs(safe_denoms[small_denoms]))
                zero_mask = safe_denoms[small_denoms] == 0
                if np.any(zero_mask): safe_denoms[small_denoms][zero_mask] = 1e-8
                denominators = safe_denoms
                
            numerator = gradient_trans**2
            denominator = denominators**3
            valid_indices = np.abs(denominator) > 1e-10
            if not np.any(valid_indices): return 1e-8
                
            sum_terms = np.zeros_like(numerator)
            sum_terms[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
            
            max_magnitude = 1e20
            large_values = np.abs(sum_terms) > max_magnitude
            if np.any(large_values):
                sum_terms[large_values] = np.sign(sum_terms[large_values]) * max_magnitude
            sum_term = np.sum(sum_terms)
            
            dstep2_dalpha = 2.0 * eigval_min * sum_term
            
            if not np.isfinite(dstep2_dalpha) or abs(dstep2_dalpha) > max_magnitude:
                dstep2_dalpha = np.sign(dstep2_dalpha) * max_magnitude if dstep2_dalpha != 0 else 1e-8
                
            return dstep2_dalpha
            
        except Exception as e:
            self.log(f"Error in derivative calculation: {str(e)}")
            return 1e-8

    def _solve_secular_equation(self, eigvals, grad_comps, alpha):
        # 1. Prepare scaled values
        eigvals_prime = eigvals / alpha
        grad_comps_prime = grad_comps / alpha
        grad_comps_prime_sq = grad_comps_prime**2
        
        # 2. Define the secular function f(lambda)
        def f(lambda_aug):
            denoms = eigvals_prime - lambda_aug
            # Strictly avoid division by zero
            denoms[np.abs(denoms) < 1e-30] = np.sign(denoms[np.abs(denoms) < 1e-30]) * 1e-30
            terms = grad_comps_prime_sq / denoms
            return lambda_aug + np.sum(terms)

        # --- 3. Robust bracket (asymptote) search ---
        
        # Sort eigenvalues and rearrange corresponding gradients
        sort_indices = np.argsort(eigvals_prime)
        eigvals_sorted = eigvals_prime[sort_indices]
        grad_comps_sorted_sq = grad_comps_prime_sq[sort_indices]

        b_upper = None
        min_eig_val_overall = eigvals_sorted[0] # Fallback value

        # Find the "first" asymptote where the gradient is non-zero
        for i in range(len(eigvals_sorted)):
            if grad_comps_sorted_sq[i] > 1e-20: # Gradient is non-zero
                # This is the first asymptote
                b_upper = eigvals_sorted[i] - 1e-10 
                break
        
        if b_upper is None:
            # All gradient components are zero (already at a stationary point)
            self.log("All gradient components in RFO space are zero.", force=True)
            return 0.0 # Step will be zero

        # --- 4. Set the lower bracket bound (b_lower) ---
        g_norm_sq = np.sum(grad_comps_prime_sq)
        # Add small constant to avoid b_lower == b_upper if g_norm_sq is tiny
        b_lower = b_upper - (1e6 + g_norm_sq) # A robust heuristic lower bound

        # --- 5. Check bracket validity ---
        try:
            f_upper = f(b_upper)
            f_lower = f(b_lower)
        except Exception as e:
            self.log(f"f(lambda) calculation failed: {e}. Using fallback.", force=True)
            return min_eig_val_overall - 1e-6 # Worst-case fallback

        if f_lower * f_upper >= 0:
            # Bracket is invalid (meaning f(b_upper) did not go to +inf or b_lower not low enough)
            self.log(f"brentq bracket invalid: f(lower)={f_lower:.2e}, f(upper)={f_upper:.2e}", force=True)
            
            # Try a much lower b_lower
            b_lower = b_upper - 1e12 # Even lower
            try:
                f_lower = f(b_lower)
            except Exception as e:
                 self.log(f"f(lambda) calculation failed for lower fallback: {e}. Using fallback.", force=True)
                 return min_eig_val_overall - 1e-6

            if f_lower * f_upper >= 0:
                self.log("FATAL: Could not find valid bracket. Using fallback.", force=True)
                return min_eig_val_overall - 1e-6 # Worst-case fallback
        
        # --- 6. Root finding ---
        try:
            root = brentq(f, b_lower, b_upper, xtol=1e-10, rtol=1e-10, maxiter=100)
            return root
        except Exception as e:
            self.log(f"brentq failed: {e}. Using fallback.", force=True)
            return min_eig_val_overall - 1e-6

    def solve_rfo(self, eigvals, gradient_components, alpha, mode="min"):
        if mode != "min":
            raise NotImplementedError("Secular equation solver is only implemented for RFO minimization (mode='min')")
            
        eigval_min = self._solve_secular_equation(eigvals, gradient_components, alpha)
        denominators = (eigvals / alpha) - eigval_min
        
        safe_denoms = denominators
        small_denoms = np.abs(safe_denoms) < 1e-10
        if np.any(small_denoms):
            # Use copy() to avoid modifying original denominators if it's passed by reference elsewhere
            safe_denoms = safe_denoms.copy() 
            safe_denoms[small_denoms] = np.sign(safe_denoms[small_denoms]) * np.maximum(1e-10, np.abs(safe_denoms[small_denoms]))
            zero_mask = safe_denoms[small_denoms] == 0
            if np.any(zero_mask): safe_denoms[small_denoms][zero_mask] = 1e-10
        
        step = -(gradient_components / alpha) / safe_denoms
        return step, eigval_min, 1.0, None

    def rfo_model(self, gradient, hessian, step):
        return np.dot(gradient, step) + 0.5 * np.dot(step, np.dot(hessian, step))

    def is_converged(self):
        return self.converged
        
    def get_predicted_energy_changes(self):
        return self.predicted_energy_changes
        
    def get_actual_energy_changes(self):
        return self.actual_energy_changes
        
    def reset_trust_radius(self):
        self.trust_radius = self.trust_radius_initial
        self.log(f"Trust radius reset to initial value: {self.trust_radius:.6f}", force=True)