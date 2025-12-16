import numpy as np
from scipy.optimize import linear_sum_assignment
# Import atomic_mass from your package
from multioptpy.Parameters.atomic_mass import atomic_mass

class ModeFollowing:
    """
    Mode Tracking Class.
    
    Features:
    1. Mass-Weighted Overlap (MWO): Physically correct projection.
    2. Adaptive / Static Reference: Follows mode rotation.
    3. Gradient Overlap: Biases selection towards the current force direction.
    4. Maximum Weight Matching (Hungarian Algo): Solves the global assignment problem
       to prevent mode swapping in dense spectra.
    5. Exponential Moving Average (EMA): Filters noise while adapting to mode rotation.
    """
    def __init__(self, saddle_order, atoms=None, initial_target_index=0, 
                 adaptive=True, update_rate=1.0, 
                 use_hungarian=True, gradient_weight=0.0, debug_mode=False):
        """
        Parameters:
            saddle_order (int): Number of modes to track.
            atoms (list): List of atomic numbers/symbols for MWO.
            initial_target_index (int): 0-based index of the starting mode.
            adaptive (bool): Update reference vectors (True) or keep static (False).
            update_rate (float): EMA coefficient (alpha) for adaptive update.
                                 1.0 = Full replacement (Standard Adaptive MOM).
                                 0.5 = Balanced (Half old, half new).
                                 0.0 = No update (Same as adaptive=False).
            use_hungarian (bool): Use Kuhn-Munkres algorithm for global matching.
            gradient_weight (float): Weight for Gradient Overlap (0.0 to 1.0).
            debug_mode (bool): Verbose logging.
        """
        self.saddle_order = saddle_order
        self.debug_mode = debug_mode
        self.reference_modes = [] 
        self.reference_indices = []
        self.is_initialized = False
        self.target_offset = initial_target_index
        
        self.adaptive = adaptive
        self.update_rate = update_rate  # EMA alpha
        self.use_hungarian = use_hungarian
        self.gradient_weight = gradient_weight
        
        # Prepare Mass Weights
        self.mass_weights = None
        self.mass_sqrt = None
        strategies = []
        
        if atoms is not None:
            try:
                masses = [atomic_mass(a) for a in atoms]
                weights_list = []
                for m in masses:
                    weights_list.extend([m, m, m])
                self.mass_weights = np.array(weights_list)
                # Sqrt weights for norm calculation: |v|_M = |v * sqrt(M)|
                self.mass_sqrt = np.sqrt(self.mass_weights)
                self.log(f"Config: MWO enabled ({len(atoms)} atoms).")
                strategies.append("Mass-Weighted")
            except Exception as e:
                print(f"[ModeFollowing] Warning: Mass init failed ({e}). Using Cartesian.")
                self.mass_weights = None
                strategies.append("Cartesian")
        else:
            strategies.append("Cartesian")
        
        if self.adaptive: 
            strategies.append(f"Adaptive(EMA={self.update_rate})")
        else: 
            strategies.append("Static")
            
        if self.use_hungarian: 
            strategies.append("Hungarian")
        else: 
            strategies.append("Greedy")
            
        if self.gradient_weight > 0: 
            strategies.append(f"GradBias({self.gradient_weight})")
        
        self.log(f"Config: {', '.join(strategies)}")
        self.strategies = strategies
        
    def log(self, message):
        if self.debug_mode:
            print(f"[ModeFollowing] {message}")

    def _calc_overlap(self, v1, v2):
        """
        Calculate normalized Overlap (S_ij).
        Returns signed value (-1.0 to 1.0).
        """
        if self.mass_weights is None:
            dot_val = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
        else:
            dot_val = np.dot(v1 * self.mass_weights, v2)
            # Use pre-calculated sqrt weights for efficiency if available
            if self.mass_sqrt is not None:
                norm1 = np.linalg.norm(v1 * self.mass_sqrt)
                norm2 = np.linalg.norm(v2 * self.mass_sqrt)
            else:
                norm1 = np.sqrt(np.dot(v1 * self.mass_weights, v1))
                norm2 = np.sqrt(np.dot(v2 * self.mass_weights, v2))
        
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0
        return dot_val / (norm1 * norm2)

    def _normalize(self, v):
        """Normalize vector according to the current metric (Mass-Weighted or Cartesian)."""
        if self.mass_weights is None:
            norm = np.linalg.norm(v)
        else:
            if self.mass_sqrt is not None:
                norm = np.linalg.norm(v * self.mass_sqrt)
            else:
                norm = np.sqrt(np.dot(v * self.mass_weights, v))
            
        if norm < 1e-12:
            return v # Avoid division by zero
        return v / norm

    def set_references(self, eigvecs, eigvals=None):
        """Set initial reference modes."""
        self.reference_modes = []
        self.reference_indices = []
        n_modes = eigvecs.shape[1]
        
        start_idx = self.target_offset
        end_idx = start_idx + self.saddle_order
        
        if end_idx > n_modes:
            start_idx = 0
            end_idx = self.saddle_order
            print(f"[ModeFollowing] Error: Index out of bounds. Fallback to 0.")

        self.log(f"Initializing references using modes [{start_idx} to {end_idx-1}]")
        
        for i in range(start_idx, end_idx):
            self.reference_modes.append(eigvecs[:, i].copy())
            self.reference_indices.append(i)
            val_str = f"{eigvals[i]:.6f}" if eigvals is not None else "N/A"
            print(f"  [ModeFollowing] Target: Mode {i} (Val: {val_str})")
            
        self.is_initialized = True

    def get_matched_indices(self, current_eigvecs, current_eigvals=None, current_gradient=None):
        """
        Find best matching modes using configured strategies.
        Updates references using EMA if adaptive=True.
        """
        if not self.is_initialized:
            raise RuntimeError("References not set.")

        n_refs = len(self.reference_modes)
        n_curr = current_eigvecs.shape[1]
        
        # 1. Build Similarity Matrix (Cost Matrix for Hungarian)
        # Rows: References, Cols: Current Modes
        # Values: Absolute Overlap (0.0 to 1.0)
        similarity_matrix = np.zeros((n_refs, n_curr))
        sign_matrix = np.zeros((n_refs, n_curr)) # Store signs for phase correction

        # Pre-calculate Gradient Overlaps if enabled
        grad_overlaps = np.zeros(n_curr)
        if self.gradient_weight > 0 and current_gradient is not None:
            g_norm = np.linalg.norm(current_gradient)
            if g_norm > 1e-10:
                normalized_grad = current_gradient / g_norm
                for j in range(n_curr):
                    # Use same metric (MWO/Cartesian) for consistency
                    ov = abs(self._calc_overlap(normalized_grad, current_eigvecs[:, j]))
                    grad_overlaps[j] = ov

        for i in range(n_refs):
            ref_vec = self.reference_modes[i]
            for j in range(n_curr):
                overlap_signed = self._calc_overlap(ref_vec, current_eigvecs[:, j])
                overlap_abs = abs(overlap_signed)
                
                # Base Score: Eigenvector Overlap
                score = overlap_abs
                
                # Add Gradient Bias
                if self.gradient_weight > 0:
                    score += self.gradient_weight * grad_overlaps[j]
                
                similarity_matrix[i, j] = score
                sign_matrix[i, j] = 1.0 if overlap_signed >= 0 else -1.0

        matched_indices = []
        matched_pairs = [] # List of (ref_idx, curr_idx)

        # 2. Solve Matching Problem
        if self.use_hungarian:
            # Hungarian Algorithm (Minimizes cost, so we neglect similarity)
            cost_matrix = -1.0 * similarity_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Extract pairs
            for r, c in zip(row_ind, col_ind):
                matched_pairs.append((r, c))
                
            # Sort by reference index order to keep list consistent
            matched_pairs.sort(key=lambda x: x[0])
            
        else:
            # Greedy Algorithm
            used_cols = set()
            for i in range(n_refs):
                best_col = -1
                max_sim = -1.0
                for j in range(n_curr):
                    if j in used_cols: continue
                    if similarity_matrix[i, j] > max_sim:
                        max_sim = similarity_matrix[i, j]
                        best_col = j
                
                if best_col != -1:
                    matched_pairs.append((i, best_col))
                    used_cols.add(best_col)
                else:
                    # Fallback (Lost track)
                    print(f"    [ModeFollowing] LOST TRACK of Ref {i}")
                    matched_pairs.append((i, self.reference_indices[i] if self.reference_indices[i] < n_curr else 0))

        # 3. Process Matches and Update References
        print(f"  [ModeFollowing] --- Tracking Status ---")
        print(" Strategies: ", self.strategies)
        
        for ref_i, curr_j in matched_pairs:
            matched_indices.append(curr_j)
            
            # Stats for logging
            sim_score = similarity_matrix[ref_i, curr_j]
            prev_idx = self.reference_indices[ref_i]
            val_str = f"{current_eigvals[curr_j]:.5f}" if current_eigvals is not None else ""
            
            # Phase correction sign
            best_sign = sign_matrix[ref_i, curr_j]
            
            print(f"    Ref {ref_i} (was {prev_idx}) -> Mode {curr_j} (Score: {sim_score:.4f}) {val_str}")
            
            if sim_score < 0.3:
                print(f"    WARNING: Very low match score!")

            # --- Adaptive Update (EMA) ---
            if self.adaptive:
                alpha = self.update_rate
                
                old_vec = self.reference_modes[ref_i]
                new_vec_aligned = current_eigvecs[:, curr_j] * best_sign
                
                # Linear combination: v_new = (1-a)*v_old + a*v_curr
                if alpha >= 1.0:
                    updated_vec = new_vec_aligned
                elif alpha <= 0.0:
                    updated_vec = old_vec
                else:
                    updated_vec = (1.0 - alpha) * old_vec + alpha * new_vec_aligned
                
                # Normalize (Important: Length must be 1 for next overlap calc)
                # This normalization respects mass-weighting if enabled
                self.reference_modes[ref_i] = self._normalize(updated_vec)
            
            self.reference_indices[ref_i] = curr_j

        print(f"  [ModeFollowing] -----------------------")
        return matched_indices