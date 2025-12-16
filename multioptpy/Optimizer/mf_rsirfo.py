import numpy as np
from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Optimizer.block_hessian_update import BlockHessianUpdate
from multioptpy.Utils.calc_tools import Calculationtools

# Import the original RSIRFO class for inheritance
from multioptpy.Optimizer.rsirfo import RSIRFO
from multioptpy.Optimizer.mode_following import ModeFollowing


class MF_RSIRFO(RSIRFO):
    """
    Mode-Following RS-I-RFO Optimizer.
    
        References:
        [1] Banerjee et al., Phys. Chem., 89, 52-57 (1985)
        [2] Heyden et al., J. Chem. Phys., 123, 224101 (2005)
        [3] Baker, J. Comput. Chem., 7, 385-395 (1986)
        [4] Besalú and Bofill, Theor. Chem. Acc., 100, 265-274 (1998)
        
        This code is made based on the below codes.
        1, https://github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/TSHessianOptimizer.py
        2, https://github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/RSIRFOptimizer.py


    Extended 'method' string support:
      "method_name : target_index : ema<val> : grad<val>"
      
    Examples:
      "block_fsb"              -> Default (Index 0, EMA=1.0 if adaptive, Grad=0.0)
      "block_fsb:1"            -> Track Mode 1
      "block_fsb:0:ema0.5"     -> Track Mode 0 with EMA alpha=0.5
      "block_fsb:ema0.1:grad0.3" -> Track Mode 0, EMA=0.1, Gradient Bias=0.3
    """
    def __init__(self, **config):
        # 1. Parse 'method' string for advanced configs
        raw_method_str = config.get("method", "auto")
        
        # Initial Defaults
        update_method = raw_method_str
        target_mode_index = 0
        
        # Check config for fallback defaults
        is_adaptive_config = config.get("adaptive_mode_following", True)
        
        # Placeholders for parsed values
        parsed_update_rate = None
        parsed_gradient_weight = 0.0
        
        # Parse logic
        if ":" in raw_method_str:
            parts = raw_method_str.split(":")
            update_method = parts[0].strip() # First part is always method name
            
            for part in parts[1:]:
                part = part.strip().lower()
                if not part: continue
                
                if part.isdigit():
                    # "1", "2" -> Target Index
                    target_mode_index = int(part)
                    
                elif part.startswith("ema"):
                    # "ema0.5" -> Update Rate
                    try:
                        val = float(part[3:])
                        parsed_update_rate = val
                    except ValueError:
                        print(f"Warning: Invalid ema value in '{part}'. Ignoring.")
                        
                elif part.startswith("grad"):
                    # "grad0.5" -> Gradient Weight
                    try:
                        val = float(part[4:])
                        parsed_gradient_weight = val
                    except ValueError:
                        print(f"Warning: Invalid grad value in '{part}'. Ignoring.")
        
        # Resolve Update Rate (EMA) and Adaptive Flag
        if parsed_update_rate is not None:
            # Explicit string config overrides config dict
            update_rate = parsed_update_rate
            # If ema > 0, we must enable adaptive mode
            adaptive = (update_rate > 1e-12)
        else:
            # Fallback to config dict or defaults
            # "If ema not specified: 0 if static, 1 if adaptive"
            adaptive = is_adaptive_config
            update_rate = 1.0 if adaptive else 0.0
            
        # Resolve Gradient Weight
        gradient_weight = parsed_gradient_weight
        
        # Update config for parent class
        config['method'] = update_method
        
        # Initialize parent RSIRFO
        super().__init__(**config)
        self.hessian_update_method = update_method
        
        self.use_mode_following = config.get("use_mode_following", True)
        
        # Other configs
        use_hungarian = config.get("use_hungarian", True)
        element_list = config.get("element_list", None)
        
        # Initialize Mode Following with resolved parameters
        self.mode_follower = ModeFollowing(
            self.saddle_order, 
            atoms=element_list,
            initial_target_index=target_mode_index,
            adaptive=adaptive,
            update_rate=update_rate,
            use_hungarian=use_hungarian,
            gradient_weight=gradient_weight,
            debug_mode=config.get("debug_mode", False)
        )
        
        if self.display_flag:
            print(f"MF-RS-I-RFO Initialized:")
            print(f"  - Update Method: {self.hessian_update_method}")
            print(f"  - Target Index: {target_mode_index}")
            print(f"  - Mode Following: Adaptive={adaptive} (EMA Rate={update_rate})")
            print(f"  - Gradient Bias: {gradient_weight}")
            print(f"  - Matching: {'Hungarian' if use_hungarian else 'Greedy'}")
            print(f"  - Mass-Weighted: {'Yes' if element_list else 'No'}")

    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """
        Execute one step of RS-I-RFO with Advanced Mode Following.
        """
        self.log(f"\n{'='*50}\nMF-RS-I-RFO Iteration {self.iteration}\n{'='*50}", force=True)
        
        if self.Initialization:
            self.prev_eigvec_min = None
            self.prev_eigvec_size = None
            self.predicted_energy_changes = []
            self.actual_energy_changes = []
            self.prev_geometry = None
            self.prev_gradient = None
            self.prev_energy = None
            self.converged = False
            self.iteration = 0
            self.Initialization = False
        
        if self.hessian is None:
            raise ValueError("Hessian matrix must be set")
        
        if self.prev_geometry is not None and self.prev_gradient is not None and len(pre_g) > 0 and len(pre_geom) > 0:
            self.update_hessian(geom_num_list, g, pre_geom, pre_g)
            
        gradient_norm = np.linalg.norm(B_g)
        self.log(f"Gradient norm: {gradient_norm:.6f}", force=True)
        
        if gradient_norm < self.gradient_norm_threshold:
            self.log(f"Converged: Gradient norm", force=True)
            self.converged = True
        
        if self.actual_energy_changes and abs(self.actual_energy_changes[-1]) < self.energy_change_threshold:
            self.log(f"Converged: Energy change", force=True)
            self.converged = True

        current_energy = B_e
        gradient = np.asarray(B_g).ravel()
        
        tmp_hess = self.hessian
        if self.bias_hessian is not None:
            H_base = tmp_hess + self.bias_hessian
        else:
            H_base = tmp_hess
            
        H = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            H_base, geom_num_list.reshape(-1, 3), geom_num_list.reshape(-1, 3), False
        )
        H = 0.5 * (H + H.T)

        eigvals, eigvecs = self.compute_eigendecomposition_with_shift(H)
        self.check_hessian_conditioning(eigvals)

        # =========================================================================
        # Mode Following: Identify Targets
        # =========================================================================
        target_indices = []
        
        if self.saddle_order > 0:
            if self.use_mode_following:
                if self.iteration == 0:
                    self.log(f"Init Mode Following...")
                    self.mode_follower.set_references(eigvecs, eigvals)
                    # For iter 0, use start indices directly
                    start = self.mode_follower.target_offset
                    target_indices = list(range(start, start + self.saddle_order))
                else:
                    # Find matching modes (pass gradient for optional bias)
                    target_indices = self.mode_follower.get_matched_indices(
                        eigvecs, eigvals, current_gradient=gradient
                    )
            else:
                target_indices = list(range(self.saddle_order))
        
        # =========================================================================
        # Trust Radius
        # =========================================================================
        if self.iteration > 0 and self.prev_energy is not None:
            actual_change = B_e - self.prev_energy
            if len(self.actual_energy_changes) >= 3:
                self.actual_energy_changes.pop(0)
            self.actual_energy_changes.append(actual_change)
            
            if self.predicted_energy_changes:
                # Use curvature of the TRACKED mode
                min_eigval_for_tr = eigvals[0]
                if target_indices and target_indices[0] < len(eigvals):
                    min_eigval_for_tr = eigvals[target_indices[0]]
                
                self.adjust_trust_radius(
                    actual_change,
                    self.predicted_energy_changes[-1],
                    min_eigval_for_tr,
                    gradient_norm
                )

        # =========================================================================
        # Image Surface Construction
        # =========================================================================
        P = np.eye(gradient.size)
        
        for idx in target_indices:
            if idx < len(eigvals) and np.abs(eigvals[idx]) > 1e-10:
                trans_vec = eigvecs[:, idx]
                if self.NEB_mode:
                     P -= np.outer(trans_vec, trans_vec)
                else:
                     P -= 2 * np.outer(trans_vec, trans_vec)

        H_star = np.dot(P, H)
        H_star = 0.5 * (H_star + H_star.T)
        grad_star = np.dot(P, gradient)
        
        eigvals_star, eigvecs_star = self.compute_eigendecomposition_with_shift(H_star)
        eigvals_star, eigvecs_star = self.filter_small_eigvals(eigvals_star, eigvecs_star)
        
        current_eigvec_size = eigvecs_star.shape[1]
        if self.prev_eigvec_size is not None and self.prev_eigvec_size != current_eigvec_size:
            self.prev_eigvec_min = None
        self.prev_eigvec_size = current_eigvec_size
        
        move_vector = self.get_rs_step(eigvals_star, eigvecs_star, grad_star)
        
        predicted_change = self.rfo_model(gradient, H, move_vector)
        
        if len(self.predicted_energy_changes) >= 3:
            self.predicted_energy_changes.pop(0)
        self.predicted_energy_changes.append(predicted_change)
        
        self.log(f"Predicted energy change: {predicted_change:.6f}", force=True)
        
        if self.actual_energy_changes and len(self.predicted_energy_changes) > 1:
            self.evaluate_step_quality()
            
        self.prev_geometry = geom_num_list
        self.prev_gradient = B_g
        self.prev_energy = current_energy
        self.iteration += 1
        
        return -1 * move_vector.reshape(-1, 1)