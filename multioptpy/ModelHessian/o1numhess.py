import numpy as np
import copy
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.sparse.linalg import LinearOperator, cg as scipy_cg, gmres as scipy_gmres
from scipy.spatial.distance import cdist

from multioptpy.ModelHessian.swart import SwartApproxHessian
from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib


class O1NumHessCalculator:
    """
    O1NumHess:
    semi-numerical Hessian generation using Optimal 1-sided numerical differentiation.
    Ref: https://doi.org/10.1021/acs.jctc.5c01354
    
    Notice:
    This implementation is experimental and lacks formal validation against the performance benchmarks established in the original paper. 
    It is intended for reference only and does not serve as a basis for contesting the validity of the referenced methodology, regardless of the performance outcomes.
    
    UPDATES:
    1. Adaptive Cutoff based on Covalent Radii (rcov_scale * (Ri + Rj))
    """

    def __init__(self, calculation_engine, element_list, charge_mult, method, 
                 rcov_scale=2.5, delta_bohr=0.005, verbosity=1):
        """
        Initialize the O1NumHess calculator with Adaptive Cutoff.

        Parameters
        ----------
        calculation_engine : Calculation
            Instance of tblite_calculation_tools.Calculation.
        element_list : list
            List of element symbols.
        charge_mult : list or np.array
            [charge, multiplicity].
        method : str
            Calculation method.
        rcov_scale : float
            Scaling factor for covalent radii sum to determine cutoff (default: 2.5).
            Cutoff_ij = rcov_scale * (R_i + R_j) + 1.0
        delta_bohr : float
            Step size for numerical differentiation (default: 0.005 Bohr).
        verbosity : int
            Level of output detail.
        """
        self.calc_engine = calculation_engine
        self.element_list = element_list
        self.charge_mult = charge_mult
        self.method = method
        
        # FIXED: Use scaling factor instead of fixed dmax
        self.rcov_scale = rcov_scale
        self.delta = delta_bohr
        self.verbosity = verbosity
        
        # Constants manager
        self.uvl = UnitValueLib()
        self.bohr2ang = self.uvl.bohr2angstroms
        
        # FIXED: Pre-calculate Covalent Radii for all atoms (in Bohr)
        # covalent_radii_lib returns Bohr based on the provided file content
        self.atom_radii = np.array([covalent_radii_lib(el) for el in element_list])
        
        # Fischer Hessian Calculator
        self.fischer_calc = SwartApproxHessian()
        
        # Paper parameters
        self.lam = 1.0e-2    # Regularization parameter lambda
        self.bet = 1.5       # Penalty exponent beta
        self.ddmax = 5.0     # Buffer for masking (Bohr)
        
        # LR loop parameters 
        self.mingrad_lr = 1.0e-3
        self.thresh_lr = 1.0e-5
        self.maxiter_lr = 1000

    def _get_gradient(self, coords_ang):
        """Call external calculation engine."""
        energy, gradient, positions, error_flag = self.calc_engine.single_point(
            file_directory=None,
            element_list=self.element_list,
            iter=1,
            electric_charge_and_multiplicity=self.charge_mult,
            method=self.method,
            geom_num_list=coords_ang
        )
        
        if error_flag:
            raise RuntimeError("SCF Calculation failed during gradient evaluation.")
            
        return np.array(gradient)

    def _guess_topology_protection(self, distmat_bohr):
        """
        Guess 1-2 (bond) and 1-3 (angle) connectivity based on distances.
        Returns a boolean mask of pairs that should be protected.
        """
        N = distmat_bohr.shape[0]
        
        # 1. Define bonding threshold (e.g., 1.3 * sum of covalent radii)
        # Using broadcasting for radii sums
        radii_sum = self.atom_radii[:, np.newaxis] + self.atom_radii[np.newaxis, :]
        bond_thresh = 1.3 * radii_sum
        
        # 2. Identify 1-2 bonds (exclude self-loop)
        # distmat < threshold AND distmat > 1e-3 (avoid self)
        bond_adj = (distmat_bohr < bond_thresh) & (distmat_bohr > 1e-3)
        
        # 3. Identify 1-3 angles using matrix multiplication
        # If A-B is bonded and B-C is bonded, (bond_adj @ bond_adj)[A, C] > 0
        # converting to float for matmul, then back to bool
        bond_float = bond_adj.astype(float)
        angle_adj = (np.dot(bond_float, bond_float)) > 0.1
        
        # Remove self-loops (A-B-A) introduced by matmul
        np.fill_diagonal(angle_adj, False)
        
        # 4. Combine masks (Protects both Bonds and Angles)
        protected_mask = bond_adj | angle_adj
        
        return protected_mask

    def compute_hessian(self, current_coords_ang):
        """
        Execute the O1NumHess algorithm.
        """
        # 1. Prepare coordinates
        coords_ang = np.array(current_coords_ang, dtype=np.float64)
        coords_bohr = coords_ang / self.bohr2ang
        N_atom = len(coords_ang)
        N_dof = 3 * N_atom
        x0_bohr = coords_bohr.flatten()

        # FIXED: Generate Adaptive Cutoff Matrix
        # cutoff_mat[i, j] = scale * (R_i + R_j) + 1.0
        # Use broadcasting: (N, 1) + (1, N) -> (N, N)
        radii_col = self.atom_radii[:, np.newaxis]
        radii_row = self.atom_radii[np.newaxis, :]
        cutoff_mat = self.rcov_scale * (radii_col + radii_row) + 1.0

        if self.verbosity > 0:
            print(f"\n=== Starting O1NumHess Calculation ===")
            print(f"  Adaptive Cutoff Strategy: {self.rcov_scale} * (R_i + R_j) + 1.0 Bohr")
            print(f"  Min Cutoff: {np.min(cutoff_mat):.2f} Bohr, Max Cutoff: {np.max(cutoff_mat):.2f} Bohr")
            print(f"  Step size (eta): {self.delta} Bohr")

        # 2. Compute reference gradient (g0)
        if self.verbosity > 0:
            print("  [1/6] Computing reference gradient...")
        g0_vec = self._get_gradient(coords_ang).flatten()

        # 3. Generate initial model Hessian (H0)
        if self.verbosity > 0:
            print("  [2/6] Generating initial model Hessian (Swart)...")
        dummy_grad = np.zeros_like(g0_vec)
        h0_bohr_units = self.fischer_calc.main(coords_bohr, self.element_list, dummy_grad)

        # 4. Build neighbor list (using adaptive cutoff)
        if self.verbosity > 0:
            print("  [3/6] Building connectivity graph...")
        distmat_bohr = cdist(coords_bohr, coords_bohr)
        
        if self.verbosity > 0:
            print("        -> Applying topology-based protection (1-2 & 1-3 pairs)...")
            
        protected_mask = self._guess_topology_protection(distmat_bohr)
        
        safe_margin = 2.0  # Bohr
        
        # update only where protection is needed AND current cutoff is too small
        cutoff_mat[protected_mask] = np.maximum(
            cutoff_mat[protected_mask], 
            distmat_bohr[protected_mask] + safe_margin
        )
        # FIXED: Pass cutoff_mat instead of scalar dmax
        nblist, nbcounts = self._build_neighbor_list(N_atom, distmat_bohr, cutoff_mat)
        
        max_nb = max(nbcounts) if len(nbcounts) > 0 else 0

        # 5. Generate optimal displacement directions
        if self.verbosity > 0:
            print("  [4/6] Generating optimal displacement directions...")
        displdir, ndispl_final = self._generate_displacement_directions_corrected(
            N_dof, coords_bohr, nblist, nbcounts, h0_bohr_units, max_nb
        )
        
        if self.verbosity > 0:
            print(f"        -> Generated {ndispl_final} directions (Full DOF: {N_dof})")

        # 6. Compute gradients
        if self.verbosity > 0:
            print("  [5/6] Computing gradients at displaced geometries...")
        
        g_displ = np.zeros((N_dof, ndispl_final))
        
        for i in range(ndispl_final):
            d_vec = displdir[:, i]
            d_vec_norm = np.linalg.norm(d_vec)
            
            if i < 3:  # Translations
                g_displ[:, i] = 0.0
                continue
            
            if i == 6:  # Breathing mode (Double-sided)
                x_fwd_bohr = x0_bohr + self.delta * d_vec / d_vec_norm
                x_fwd_ang = x_fwd_bohr.reshape(-1, 3) * self.bohr2ang
                g_fwd = self._get_gradient(x_fwd_ang).flatten()
                
                x_bwd_bohr = x0_bohr - self.delta * d_vec / d_vec_norm
                x_bwd_ang = x_bwd_bohr.reshape(-1, 3) * self.bohr2ang
                g_bwd = self._get_gradient(x_bwd_ang).flatten()
                
                g_displ[:, i] = (g_fwd - g_bwd) / (2.0 * self.delta)
            else:  # Others (Single-sided)
                x_new_bohr = x0_bohr + self.delta * d_vec / d_vec_norm
                x_new_ang = x_new_bohr.reshape(-1, 3) * self.bohr2ang
                g_new = self._get_gradient(x_new_ang).flatten()
                
                g_displ[:, i] = (g_new - g0_vec) / self.delta

        # 7. Reconstruct Hessian
        if self.verbosity > 0:
            print("  [6/6] Reconstructing Hessian (ODLR solver + LR loop)...")
        
        dof_distmat = np.zeros((N_dof, N_dof))
        # FIXED: Also need dof_cutoff_mat for ODLR
        dof_cutoff_mat = np.zeros((N_dof, N_dof))
        
        for i in range(N_atom):
            for j in range(N_atom):
                dof_distmat[3*i:3*i+3, 3*j:3*j+3] = distmat_bohr[i, j]
                dof_cutoff_mat[3*i:3*i+3, 3*j:3*j+3] = cutoff_mat[i, j]

        # FIXED: Pass dof_cutoff_mat to solver
        hess_local = self._solve_odlr_problem_corrected(
            dof_distmat, displdir, g_displ, ndispl_final, dof_cutoff_mat
        )
        
        # FIXED: Use Strict LR Loop
        hessian, final_err = self._lr_loop_strict(
            ndispl_final, g_displ, hess_local, displdir
        )
        
        if self.verbosity > 0:
            print(f"  Final residual error: {final_err:.2e}")
            print("=== O1NumHess Calculation Finished ===")
            
        return hessian

    # =========================================================================
    # Internal Algorithm Implementation
    # =========================================================================

    def _build_neighbor_list(self, N_atom, distmat, cutoff_mat):
        """
        Create neighbor list using adaptive cutoff matrix.
        adj[i, j] = 1 if dist[i, j] < cutoff[i, j]
        """
        # FIXED: Use cutoff_mat for adjacency
        adj = (distmat < cutoff_mat).astype(int)
        np.fill_diagonal(adj, 1)
        
        n_comp, labels = connected_components(adj, directed=False)
        
        # MST Logic
        if n_comp > 1:
            max_val = np.max(distmat) * 10.0
            comp_dist = np.full((n_comp, n_comp), max_val)
            bridge_pairs = {}

            for i in range(N_atom):
                c_i = labels[i]
                for j in range(i + 1, N_atom):
                    c_j = labels[j]
                    if c_i != c_j:
                        d = distmat[i, j]
                        if d < comp_dist[c_i, c_j]:
                            comp_dist[c_i, c_j] = d
                            comp_dist[c_j, c_i] = d
                            bridge_pairs[(c_i, c_j)] = (i, j)
                            bridge_pairs[(c_j, c_i)] = (j, i)
            
            mst = minimum_spanning_tree(comp_dist).toarray()
            
            for c1 in range(n_comp):
                for c2 in range(c1 + 1, n_comp):
                    if mst[c1, c2] > 0 and mst[c1, c2] < max_val:
                        atom_i, atom_j = bridge_pairs[(c1, c2)]
                        adj[atom_i, atom_j] = 1
                        adj[atom_j, atom_i] = 1

        nblist = [[] for _ in range(3 * N_atom)]
        nbcounts = np.zeros(3 * N_atom, dtype=int)
        
        rows, cols = np.nonzero(adj)
        
        for atom_i, atom_j in zip(rows, cols):
            for k in range(3):
                dof_i = 3 * atom_i + k
                for l in range(3):
                    dof_j = 3 * atom_j + l
                    nblist[dof_i].append(dof_j)
        
        for i in range(3 * N_atom):
            nbcounts[i] = len(nblist[i])

        return nblist, nbcounts

    def _generate_displacement_directions_corrected(self, N_dof, coords_bohr, 
                                                    nblist, nbcounts, H0, max_nb):
        """
        Generate optimal displacement directions using iterative local eigenvector analysis.
        Optimized for performance: 
        - Atom-based iteration (3x speedup)
        - Vectorized phase alignment
        - Vectorized initial mode generation
        - Using np.dot instead of @ operator
        """
        N_atom = N_dof // 3
        displdir = np.zeros((N_dof, N_dof))
        eps = 1.0e-6
        eps2 = 1.0e-8

        # --- 1. Initial 7 directions (Vectorized) ---
        
        # Translations (Cols 0,1,2)
        # 1.0 for x, y, z respectively across all atoms
        for i in range(3):
            displdir[i::3, i] = 1.0

        # Rotations (Cols 3,4,5)
        center = np.mean(coords_bohr, axis=0)
        # Vectorized Inertia Tensor calculation
        rel_coords = coords_bohr - center  # (N_atom, 3)
        
        # I = sum( (r^2)I - r x rT )
        r_sq = np.sum(rel_coords**2, axis=1)
        # Use np.dot for matrix multiplication
        I_tensor = np.eye(3) * np.sum(r_sq) - np.dot(rel_coords.T, rel_coords)
        
        try:
            _, rot_axes = np.linalg.eigh(I_tensor)
        except np.linalg.LinAlgError:
            rot_axes = np.eye(3)

        # Vectorized cross product for all atoms
        for i in range(3):
            axis = rot_axes[:, i]
            # Cross product of axis with all r vectors
            # np.cross(A, B) where A=(3,), B=(N,3) broadcasts A
            # rot_vec = axis x r
            rot_vecs = np.cross(axis, rel_coords) # (N_atom, 3)
            displdir[:, 3+i] = rot_vecs.flatten()

        # Breathing (Col 6)
        displdir[:, 6] = rel_coords.flatten()

        # Normalize initial directions
        norms = np.linalg.norm(displdir[:, :7], axis=0)
        # Avoid division by zero
        valid_mask = norms > eps2
        # Use broadcasting for division
        displdir[:, :7] = np.divide(displdir[:, :7], norms[None, :], where=valid_mask[None, :])
        
        ndispl_final = 7

        # --- 2. Iterative directions (Atom-based Loop) ---
        
        # Iterate over ATOMS instead of DOFs to reduce overhead
        for n_curr in range(7, N_dof):
            ev = np.zeros(N_dof)
            coverage = np.zeros(N_dof)
            
            # Inner loop: Atom-based
            for i_atom in range(N_atom):
                dof_idx = 3 * i_atom
                nnb = nbcounts[dof_idx]
                
                if nnb == 0: 
                    continue
                
                # If neighborhood is smaller than current subspace dimension, skip
                if nnb <= n_curr: 
                    continue

                # Retrieve neighbor indices (DOFs)
                nb_idx = np.array(nblist[dof_idx][:nnb])
                
                # Extract local Hessian block
                subH = H0[np.ix_(nb_idx, nb_idx)]
                
                # Project out existing directions
                if n_curr > 0:
                    vec_subset = displdir[np.ix_(nb_idx, range(n_curr))]
                    
                    try:
                        q, _ = np.linalg.qr(vec_subset)
                        
                        # Project: H_new = P H P, where P = I - Q Q.T
                        # Using np.dot instead of @
                        # P = I - np.dot(q, q.T)
                        projmat = np.eye(nnb) - np.dot(q, q.T)
                        
                        # subH = np.dot(np.dot(projmat, subH), projmat.T)
                        temp = np.dot(projmat, subH)
                        subH = np.dot(temp, projmat.T)
                        
                        # Ensure symmetry
                        subH = 0.5 * (subH + subH.T)
                        
                    except np.linalg.LinAlgError:
                        continue

                # Diagonalize
                try:
                    loceigs, locevecs = np.linalg.eigh(subH)
                    # Select eigenvector with largest absolute eigenvalue (stiffest mode)
                    locind = np.argmax(np.abs(loceigs))
                    locev = locevecs[:, locind]
                except np.linalg.LinAlgError:
                    continue

                # Phase Alignment (Greedy Sum)
                # Calculate dot product to determine sign alignment
                # Extract current accumulated vector components for this neighborhood
                current_accum = coverage[nb_idx] * ev[nb_idx]
                dot_prod = np.dot(current_accum, locev)
                
                sign = -1.0 if dot_prod < -eps else 1.0
                
                # Update global vector and coverage
                denom = coverage[nb_idx] + 1.0
                ev[nb_idx] = (current_accum + sign * locev) / denom
                coverage[nb_idx] += 1.0

            # Orthogonalize against all previous directions (Gram-Schmidt)
            # overlaps = displdir.T @ ev
            overlaps = np.dot(displdir[:, :n_curr].T, ev)
            
            # ev -= displdir @ overlaps
            ev -= np.dot(displdir[:, :n_curr], overlaps)

            # Check norm and terminate
            v_norm = np.linalg.norm(ev)
            if v_norm < eps2:
                ndispl_final = n_curr
                break

            displdir[:, n_curr] = ev / v_norm
            ndispl_final = n_curr + 1

        return displdir[:, :ndispl_final], ndispl_final
        
    def _solve_odlr_problem_corrected(self, distmat, displdir, g, ndispl_final, cutoff_mat):
        """
        ODLR solver using a Lightweight Cascade Strategy (CG -> GMRES).
        """
        N = distmat.shape[0]
        
        # --- 1. System Setup ---
        # W^2 weighting for regularization
        W2 = self.lam * np.maximum(0.0, distmat - cutoff_mat) ** (2.0 * self.bet)
        
        # RHS computation
        rhs = np.dot(g[:, :ndispl_final], displdir[:, :ndispl_final].T)
        rhs = 0.5 * (rhs + rhs.T)
        
        # Mask setup for sparse packing
        mask = distmat < (cutoff_mat + self.ddmax)
        for i in range(N):
            mask[i, :i] = False
        
        def pack_sym(m):
            return ((m + m.T) * 0.5)[mask]
        
        def unpack_sym(v):
            H = np.zeros((N, N))
            H[mask] = v
            H = H + H.T
            for i in range(N):
                H[i, i] /= 2.0
            return H
        
        rhs_vec = pack_sym(rhs)
        ndim = len(rhs_vec)
        
        if ndim == 0:
            return np.zeros((N, N))
        
        def matvec(x_vec):
            H_tmp = unpack_sym(x_vec)
            # A(H) = H * D * D^T + W^2 * H
            tmp2 = np.dot(H_tmp, displdir[:, :ndispl_final])
            f1 = np.dot(tmp2, displdir[:, :ndispl_final].T)
            f1 = 0.5 * (f1 + f1.T)
            f2 = W2 * H_tmp
            return pack_sym(f1 + f2)
        
        op = LinearOperator((ndim, ndim), matvec=matvec, dtype=float)
        
        # --- 2. Lightweight Cascade Loop ---
        
        # Strategy: Try fast CG first. If it fails/stalls, try robust GMRES.
        solvers = [
            ("CG", scipy_cg, {"maxiter": 1000, "atol": 1e-14}),
            ("GMRES", scipy_gmres, {"maxiter": 1000, "atol": 1e-14, "restart": 30}),
        ]
        
        best_sol = None
        best_res = np.inf
        best_name = "None"
        
        if self.verbosity > 0:
            print(f"  [ODLR] Solving sparse system (Dim: {ndim}) with Fast Cascade...")

        for name, solver_func, kwargs in solvers:
            try:
                # Run solver
                sol, info = solver_func(op, rhs_vec, **kwargs)
                
                # Check residual explicitly
                Ax = matvec(sol)
                res_norm = np.linalg.norm(rhs_vec - Ax)
                
                if self.verbosity > 0:
                    status = "Converged" if info == 0 else f"Failed(info={info})"
                    print(f"    - {name:5s}: {status} | Residual: {res_norm:.4e}")
                
                # Keep the best result found so far
                if res_norm < best_res:
                    best_res = res_norm
                    best_sol = sol
                    best_name = name
                
                # If we have a good enough solution, stop cascading to save time
                if res_norm < 1e-6:
                    break
                    
            except Exception as e:
                print(f"    - {name:5s}: Exception occurred ({str(e)})")
                continue

        # --- 3. Finalize ---
        if best_sol is None:
            print("  [ODLR] CRITICAL: All solvers failed. Returning zero Hessian.")
            return np.zeros((N, N))
        
        if self.verbosity > 0:
            print(f"  [ODLR] Selected solution from {best_name} (Residual: {best_res:.4e})")
            
        hess_out = unpack_sym(best_sol)
        return hess_out
        
    def _lr_loop_strict(self, ndispl, g, hess_out, displdir):
        """
        Low-rank correction loop: Momentum + Adaptive Step + Best Keeper.
        The "Engineering Sweet Spot" - Fast, Low Memory, Robust.
        """
        # --- 1. Scaling Setup ---
        g_active = g[:, :ndispl]
        d_active = displdir[:, :ndispl]
        
        epsilon = 1.0e-3
        g_norms = np.linalg.norm(g_active, axis=0)
        scales = epsilon / np.maximum(epsilon, g_norms)
        
        g_scaled = g_active * scales[np.newaxis, :]
        d_scaled = d_active * scales[np.newaxis, :]
        
        # --- 2. Parameters ---
        dampfac = 1.0       # Current step size multiplier
        momentum = 0.5      # Inertia (starts cautious)
        
        N = hess_out.shape[0]
        prev_update = np.zeros((N, N)) # Only 1 extra matrix needed
        
        # --- 3. Best Solution Keeper ---
        best_hess = hess_out.copy()
        best_err = np.inf
        
        err0 = np.inf
        norm_g_scaled = np.linalg.norm(g_scaled)
        
        if self.verbosity > 0:
            print(f"  [LR-Loop] Starting Correction (Momentum + Adaptive Step)")
            print(f"  {'Iter':>5} | {'Residual':>12} | {'Ratio':>8} | {'Damp':>6} | {'Status'}")
            print("  " + "-"*60)

        for it in range(1, self.maxiter_lr + 1):
            # Compute Residual
            tmp = np.dot(hess_out, d_scaled)
            resid = g_scaled - tmp
            err = np.linalg.norm(resid)
            
            # --- Best Keeper ---
            if err < best_err:
                best_err = err
                best_hess = hess_out.copy() # Backup the best
            
            # --- Convergence Check ---
            if err < self.thresh_lr:
                if self.verbosity > 0:
                    print(f"  {it:5d} | {err:.4e} | {'CONVERGED':>16}")
                break
            
            # --- Adaptive Logic (Bold Driver) ---
            status = ""
            ratio = err / err0 if err0 != np.inf else 0.0
            
            if err > err0 and err > norm_g_scaled:
                # 1. Divergence detected: Brake hard
                dampfac *= 0.5
                momentum = 0.0      # Kill inertia
                prev_update[:] = 0  # Clear history
                status = "DAMP DOWN"
                
                # Optional: Restore best if things went really wrong
                if err > best_err * 2.0:
                    hess_out = best_hess.copy()
                    status = "RESET"
            
            elif ratio < 0.999:
                # 2. Improving: Accelerate
                dampfac = min(1.2, dampfac * 1.05)
                momentum = min(0.9, momentum + 0.05) # Build momentum up to 0.9
                if it % 10 == 0: status = "ACCEL"
            
            else:
                # 3. Stagnation: Check if we are stuck at noise floor
                if abs(err - err0) < 1.0e-7:
                    if self.verbosity > 0:
                        print(f"  {it:5d} | {err:.4e} | {'STAGNATED (Noise Floor)':>24}")
                    break
            
            # --- Update with Momentum ---
            # Correction direction
            hcorr = np.dot(resid, d_scaled.T)
            hcorr = 0.5 * (hcorr + hcorr.T)
            
            # Update = Damp * Correction + Momentum * Previous
            current_update = dampfac * hcorr + momentum * prev_update
            
            hess_out = hess_out + current_update
            
            # Store history
            prev_update = current_update
            err0 = err
            
            # Log
            if self.verbosity > 0:
                if it % 100 == 0 or it == 1 or status in ["DAMP DOWN", "RESET"]:
                    print(f"  {it:5d} | {err:.4e} | {ratio:8.4f} | {dampfac:6.4f} | {status}")

        # Return best solution found, not necessarily the last one
        return best_hess, best_err