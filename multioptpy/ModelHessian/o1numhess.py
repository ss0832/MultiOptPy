import numpy as np
import copy
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.sparse.linalg import LinearOperator, cg as scipy_cg 
from scipy.spatial.distance import cdist


from multioptpy.ModelHessian.swartd3 import SwartD3ApproxHessian
from multioptpy.Parameters.parameter import UnitValueLib


class O1NumHessCalculator:
    """
    O1NumHess: Semi-numerical Hessian generation using Optimal 1-sided differentiation.
    Ref: https://doi.org/10.1021/acs.jctc.5c01354
    
    CORRECTED VERSION - Fixes:
    1. Added initial 7 displacement directions (translations, rotations, breathing)
    2. Fixed phase fixing algorithm with coverage-weighted averaging
    3. Added adaptive damping in LR loop
    4. Fixed ODLR matrix packing/unpacking
    5. Improved numerical differentiation strategy
    """

    def __init__(self, calculation_engine, element_list, charge_mult, method, 
                 dmax_bohr=1.0, delta_bohr=0.005, verbosity=1):
        """
        Initialize the O1NumHess calculator.

        Parameters
        ----------
        calculation_engine : Calculation
            Instance of tblite_calculation_tools.Calculation.
        element_list : list
            List of element symbols or atomic numbers.
        charge_mult : list or np.array
            [charge, multiplicity].
        method : str
            Calculation method (e.g., "GFN2-xTB").
        dmax_bohr : float
            Cutoff distance Δr^(1) for near-range pairs (default: 1.0 Bohr, Paper value).
        delta_bohr : float
            Step size for numerical differentiation (default: 0.005 Bohr, Paper value).
        verbosity : int
            Level of output detail.
        """
        self.calc_engine = calculation_engine
        self.element_list = element_list
        self.charge_mult = charge_mult
        self.method = method
        self.dmax = dmax_bohr
        self.delta = delta_bohr
        self.verbosity = verbosity
        
        # Constants manager
        self.uvl = UnitValueLib()
        self.bohr2ang = self.uvl.bohr2angstroms
        
        # Fischer Hessian Calculator (for initial guess)
        self.fischer_calc = SwartD3ApproxHessian()
        
        # Paper parameters
        self.lam = 1.0e-2    # Regularization parameter λ (Eq. 28)
        self.bet = 1.5       # Penalty exponent β (Eq. 29)
        self.ddmax = 5.0     # Buffer for masking Δr^(2) - Δr^(1)
        
        # LR loop parameters 
        self.mingrad_lr = 1.0e-3
        self.thresh_lr = 1.0e-8
        self.maxiter_lr = 100

    def _get_gradient(self, coords_ang):
        """
        Call the external calculation engine to get the gradient.
        
        Parameters
        ----------
        coords_ang : np.array
            Coordinates in Angstroms.
            
        Returns
        -------
        gradient : np.array
            Gradient in Hartree/Bohr.
        """
        # Call according to Calculation.single_point specifications
        # With file_directory=None, geom_num_list is used preferentially
        energy, gradient, positions, error_flag = self.calc_engine.single_point(
            file_directory=None,
            element_list=self.element_list,
            iter=1,  # Dummy iteration number
            electric_charge_and_multiplicity=self.charge_mult,
            method=self.method,
            geom_num_list=coords_ang
        )
        
        if error_flag:
            raise RuntimeError("SCF Calculation failed during gradient evaluation.")
            
        return np.array(gradient)

    def compute_hessian(self, current_coords_ang):
        """
        Execute the O1NumHess algorithm.

        Returns
        -------
        hessian : np.array
            Hessian matrix in atomic units (Hartree/Bohr^2).
        """
        if self.verbosity > 0:
            print(f"\n=== Starting O1NumHess Calculation ===")
            print(f"  Δr^(1) (dmax): {self.dmax} Bohr")
            print(f"  Δr^(2): {self.dmax + self.ddmax} Bohr")
            print(f"  Step size (η): {self.delta} Bohr")

        # 1. Prepare coordinates (distinguish Bohr and Angstrom for physical consistency)
        coords_ang = np.array(current_coords_ang, dtype=np.float64)
        coords_bohr = coords_ang / self.bohr2ang
        N_atom = len(coords_ang)
        N_dof = 3 * N_atom
        x0_bohr = coords_bohr.flatten()

        # 2. Compute reference gradient (g0)
        if self.verbosity > 0:
            print("  [1/6] Computing reference gradient...")
        g0_vec = self._get_gradient(coords_ang).flatten()

        # 3. Generate initial model Hessian (H0) using Fischer-D3
        if self.verbosity > 0:
            print("  [2/6] Generating initial model Hessian (Fischer-D3)...")
        
        # Fischer method expects Angstrom input. Gradient info is not essential for H0 but required as argument
        dummy_grad = np.zeros_like(g0_vec)
        # FischerD3ApproxHessianOld.main returns projected Hessian,
        # but Cartesian space curvature is important for displacement direction generation
        h0_bohr_units = self.fischer_calc.main(coords_bohr, self.element_list, dummy_grad)
        
     

        # 4. Build neighbor list and graph connectivity
        if self.verbosity > 0:
            print("  [3/6] Building connectivity graph...")
        distmat_bohr = cdist(coords_bohr, coords_bohr)
        nblist, nbcounts = self._build_neighbor_list(N_atom, distmat_bohr, self.dmax)
        
        max_nb = max(nbcounts) if len(nbcounts) > 0 else 0

        # 5. Generate optimal displacement directions
        # FIXED: Now generates initial 7 directions + iterative directions
        if self.verbosity > 0:
            print("  [4/6] Generating optimal displacement directions...")
        displdir, ndispl_final = self._generate_displacement_directions_corrected(
            N_dof, coords_bohr, nblist, nbcounts, h0_bohr_units, max_nb
        )
        
        if self.verbosity > 0:
            print(f"        -> Generated {ndispl_final} directions (Full DOF: {N_dof})")

        # 6. Compute gradients by numerical differentiation
        # FIXED: Added double-sided differentiation for breathing mode
        if self.verbosity > 0:
            print("  [5/6] Computing gradients at displaced geometries...")
        
        g_displ = np.zeros((N_dof, ndispl_final))
        
        for i in range(ndispl_final):
            # Displacement: x_new = x0 + delta * d
            d_vec = displdir[:, i]
            d_vec_norm = np.linalg.norm(d_vec)
            
            # Skip translations (indices 0-2): gradient is always zero
            if i < 3:
                g_displ[:, i] = 0.0
                continue
            
            # For rotations (indices 3-5), could use analytical formula
            # but for simplicity, we compute numerically here
            # (Paper mentions skipping if g0 != 0, but implementation is optional)
            
            # FIXED: Double-sided differentiation for breathing mode (index 6)
            if i == 6:
                x_fwd_bohr = x0_bohr + self.delta * d_vec / d_vec_norm
                x_fwd_ang = x_fwd_bohr.reshape(-1, 3) * self.bohr2ang
                g_fwd = self._get_gradient(x_fwd_ang).flatten()
                
                x_bwd_bohr = x0_bohr - self.delta * d_vec / d_vec_norm
                x_bwd_ang = x_bwd_bohr.reshape(-1, 3) * self.bohr2ang
                g_bwd = self._get_gradient(x_bwd_ang).flatten()
                
                # Central difference
                g_displ[:, i] = (g_fwd - g_bwd) / (2.0 * self.delta)
            else:
                # Single-sided differentiation for other directions
                x_new_bohr = x0_bohr + self.delta * d_vec / d_vec_norm
                x_new_ang = x_new_bohr.reshape(-1, 3) * self.bohr2ang
                g_new = self._get_gradient(x_new_ang).flatten()
                
                # Forward difference: H*d ≈ (g_new - g0) / delta
                g_displ[:, i] = (g_new - g0_vec) / self.delta

        # 7. Reconstruct Hessian (ODLR + Low Rank Update)
        if self.verbosity > 0:
            print("  [6/6] Reconstructing Hessian (ODLR solver + LR loop)...")
        
        # Create DOF-level distance matrix (for regularization term calculation)
        dof_distmat = np.zeros((N_dof, N_dof))
        for i in range(N_atom):
            for j in range(N_atom):
                dof_distmat[3*i:3*i+3, 3*j:3*j+3] = distmat_bohr[i, j]

        # FIXED: Corrected ODLR solver with proper packing/unpacking
        hess_local = self._solve_odlr_problem_corrected(
            dof_distmat, displdir, g_displ, ndispl_final
        )
        
        # FIXED: Apply low-rank correction loop with adaptive damping
        hessian, final_err = self._lr_loop_corrected(
            ndispl_final, g_displ, hess_local, displdir
        )
        
        if self.verbosity > 0:
            print(f"  Final residual error: {final_err:.2e}")
            print("=== O1NumHess Calculation Finished ===")
            
        return hessian

    # =========================================================================
    # Internal Algorithm Implementation (CORRECTED)
    # =========================================================================

    def _build_neighbor_list(self, N_atom, distmat, dmax):
        """
        Create neighbor list for each DOF from atomic distance matrix.
        If graph is disconnected, forcibly connect using MST (Minimum Spanning Tree).
        
        Returns neighbor list and neighbor counts (added for corrected algorithm).
        """
        # Atomic adjacency matrix
        adj = (distmat < dmax).astype(int)
        np.fill_diagonal(adj, 1)  # Include self
        
        n_comp, labels = connected_components(adj, directed=False)
        
        # If multiple components, add bridges using MST
        if n_comp > 1:
            max_val = np.max(distmat) * 10.0
            # Distance matrix between components
            comp_dist = np.full((n_comp, n_comp), max_val)
            # Record atom pairs that give minimum distance
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
            
            # Create MST between components
            mst = minimum_spanning_tree(comp_dist).toarray()
            
            # Add atom pairs corresponding to edges on MST to adjacency matrix
            for c1 in range(n_comp):
                for c2 in range(c1 + 1, n_comp):
                    if mst[c1, c2] > 0 and mst[c1, c2] < max_val:
                        atom_i, atom_j = bridge_pairs[(c1, c2)]
                        adj[atom_i, atom_j] = 1
                        adj[atom_j, atom_i] = 1

        # Expand to DOF level (if atoms i, j are connected, all their xyz components are connected)
        nblist = [[] for _ in range(3 * N_atom)]
        nbcounts = np.zeros(3 * N_atom, dtype=int)
        
        rows, cols = np.nonzero(adj)
        
        for atom_i, atom_j in zip(rows, cols):
            # For all DOFs of atom_i, add all DOFs of atom_j
            for k in range(3):
                dof_i = 3 * atom_i + k
                for l in range(3):
                    dof_j = 3 * atom_j + l
                    nblist[dof_i].append(dof_j)
        
        # Count neighbors
        for i in range(3 * N_atom):
            nbcounts[i] = len(nblist[i])

        return nblist, nbcounts

    def _generate_displacement_directions_corrected(self, N_dof, coords_bohr, 
                                                    nblist, nbcounts, H0, max_nb):
        """
        CORRECTED: Generate displacement directions following Paper Algorithm.
        
        Key fixes:
        1. Added initial 7 directions (translations, rotations, breathing)
        2. Fixed phase fixing with coverage-weighted norm comparison
        3. Proper orthogonalization
        """
        N_atom = N_dof // 3
        displdir = np.zeros((N_dof, N_dof))
        
        eps = 1.0e-6   # Sign determination threshold
        eps2 = 1.0e-8  # Convergence threshold (Paper: eps2 in gen_displdir)
        
        # ====================================================================
        # STEP 1: Generate initial 7 displacement directions (Paper Section III.D)
        # ====================================================================
        
        # 1a. Translations (indices 0-2)
        for i in range(3):
            for j in range(N_atom):
                displdir[3*j + i, i] = 1.0
        
        # 1b. Rotations (indices 3-5) - from moment of inertia
        # Compute center (using equal masses as mentioned in Paper Appendix A)
        center = np.mean(coords_bohr, axis=0)
        
        # Compute moment of inertia tensor (equal masses)
        I_tensor = np.zeros((3, 3))
        for j in range(N_atom):
            r = coords_bohr[j] - center
            I_tensor += np.eye(3) * np.dot(r, r) - np.outer(r, r)
        
        # Eigenvectors of moment of inertia = rotation axes
        try:
            _, rot_axes = np.linalg.eigh(I_tensor)
        except:
            # Fallback to coordinate axes if inertia tensor is singular
            rot_axes = np.eye(3)
        
        for i in range(3):
            axis = rot_axes[:, i]
            for j in range(N_atom):
                r = coords_bohr[j] - center
                rot_vec = np.cross(axis, r)
                displdir[3*j:3*j+3, 3+i] = rot_vec
        
        # 1c. Symmetric breathing mode (index 6)
        for j in range(N_atom):
            r = coords_bohr[j] - center
            displdir[3*j:3*j+3, 6] = r
        
        # Normalize initial 7 directions
        for i in range(7):
            norm = np.linalg.norm(displdir[:, i])
            if norm > eps2:
                displdir[:, i] /= norm
        
        ndispl_final = 7
        
        # ====================================================================
        # STEP 2: Generate additional directions iteratively (Paper Section III.D)
        # ====================================================================
        
        for n_curr in range(7, N_dof):
            ev = np.zeros(N_dof)
            coverage = np.zeros(N_dof)
            
            # For each DOF, compute local eigenvector
            for j in range(N_dof):
                nnb = nbcounts[j]
                if nnb == 0:
                    continue
                
                nb_idx = np.array(nblist[j][:nnb])
                
                # Skip if subspace saturated
                if nnb <= n_curr:
                    continue
                
                # Extract submatrix
                subH = H0[np.ix_(nb_idx, nb_idx)]
                
                # Project out existing directions
                if n_curr > 0:
                    vec_subset = displdir[np.ix_(nb_idx, range(n_curr))]
                    
                    # Orthogonalize vec_subset (QR decomposition)
                    try:
                        q, r = np.linalg.qr(vec_subset)
                    except:
                        continue
                    
                    # Projection matrix P = I - QQ^T
                    projmat = np.eye(nnb) - np.dot(q, q.T)
                    
                    # Project submatrix: P * H * P^T
                    subH = np.dot(projmat, np.dot(subH, projmat.T))
                    subH = 0.5 * (subH + subH.T)
                
                # Diagonalize
                try:
                    loceigs, locevecs = np.linalg.eigh(subH)
                    # FIXED: Take eigenvector with maximum absolute eigenvalue
                    # (Paper: "eigenvector with the largest eigenvalue")
                    locind = np.argmax(np.abs(loceigs))
                    locev = locevecs[:, locind]
                except np.linalg.LinAlgError:
                    continue
                
                # ============================================================
                # FIXED: Phase fixing with coverage-weighted averaging
                # (Paper Section III.D)
                # ============================================================
                norm_ev1 = 0.0
                norm_ev2 = 0.0
                for p in range(nnb):
                    idx = nb_idx[p]
                    # Weighted average if we add locev
                    val1 = (coverage[idx] * ev[idx] + locev[p]) / (coverage[idx] + 1.0)
                    # Weighted average if we subtract locev
                    val2 = (coverage[idx] * ev[idx] - locev[p]) / (coverage[idx] + 1.0)
                    norm_ev1 += val1**2
                    norm_ev2 += val2**2
                
                norm_ev1 = np.sqrt(norm_ev1)
                norm_ev2 = np.sqrt(norm_ev2)
                
                # Determine sign
                if norm_ev1 > norm_ev2 + eps:
                    sign = 1.0
                elif norm_ev1 < norm_ev2 - eps:
                    sign = -1.0
                else:
                    # Deterministic tie-breaking (Paper: "based on max element")
                    locind_max = np.argmax(np.abs(locev))
                    sign = 1.0 if locev[locind_max] > 0 else -1.0
                
                # Apply update
                for p in range(nnb):
                    idx = nb_idx[p]
                    ev[idx] = (coverage[idx] * ev[idx] + sign * locev[p]) / (coverage[idx] + 1.0)
                
                # Update coverage
                for p in range(nnb):
                    coverage[nb_idx[p]] += 1.0
            
            # Project out previous columns (Gram-Schmidt)
            for k in range(n_curr):
                d_dot = np.dot(ev, displdir[:, k])
                ev -= d_dot * displdir[:, k]
            
            # Check norm (Paper: if norm < eps2, exit)
            v_norm = np.linalg.norm(ev)
            if v_norm < eps2:
                ndispl_final = n_curr
                break
            
            # Normalize and store
            ev /= v_norm
            displdir[:, n_curr] = ev
            ndispl_final = n_curr + 1
        
        return displdir[:, :ndispl_final], ndispl_final

    def _solve_odlr_problem_corrected(self, distmat, displdir, g, ndispl_final):
        """
        CORRECTED: ODLR (Off-Diagonal Low-Rank) solver with proper packing.
        
        Key fixes:
        1. Correct matrix packing (upper triangle only)
        2. Proper symmetrization in unpack
        3. Changed solver from GMRES to CG (symmetric positive definite)
        """
        N = distmat.shape[0]
        
        # Regularization term W^2 (Paper Eq. 29)
        W2 = self.lam * np.maximum(0.0, distmat - self.dmax) ** (2.0 * self.bet)
        
        # RHS = g * displdir^T (symmetrized) (Paper Eq. 30)
        rhs = np.dot(g[:, :ndispl_final], displdir[:, :ndispl_final].T)
        rhs = 0.5 * (rhs + rhs.T)
        
        # FIXED: Mask - upper triangle only within distance cutoff
        mask = distmat < (self.dmax + self.ddmax)
        for i in range(N):
            mask[i, :i] = False  # Upper triangle only
        
        # FIXED: Proper pack/unpack for symmetric matrix
        def pack_sym(m):
            """Pack symmetric matrix using mask (symmetrize first)"""
            return ((m + m.T) * 0.5)[mask]
        
        def unpack_sym(v):
            """Unpack vector to symmetric matrix"""
            H = np.zeros((N, N))
            H[mask] = v
            # Symmetrize: H_ij = H_ji for i < j
            H = H + H.T
            # Diagonal was added twice, so divide by 2
            for i in range(N):
                H[i, i] /= 2.0
            return H
        
        rhs_vec = pack_sym(rhs)
        ndim = len(rhs_vec)
        
        if ndim == 0:
            return np.zeros((N, N))
        
        # FIXED: Define linear operator matching Paper Eq. (30)
        # A(H) = (H*D*D^T + (H*D*D^T)^T)/2 + W^2*H
        def matvec(x_vec):
            H_tmp = unpack_sym(x_vec)
            
            # Term 1: H * displdir * displdir^T (symmetrized)
            tmp2 = np.dot(H_tmp, displdir[:, :ndispl_final])
            f1 = np.dot(tmp2, displdir[:, :ndispl_final].T)
            f1 = 0.5 * (f1 + f1.T)
            
            # Term 2: W^2 * H (regularization)
            f2 = W2 * H_tmp
            
            return pack_sym(f1 + f2)
        
        op = LinearOperator((ndim, ndim), matvec=matvec)
        
        # FIXED: Use Conjugate Gradient (CG) instead of GMRES
        # (System is symmetric positive definite)
        sol, info = scipy_cg(op, rhs_vec, maxiter=1000, atol=1e-14)
        
        if info != 0:
            if self.verbosity > 0:
                print(f"  Warning: CG did not converge (info={info})")
        
        hess_out = unpack_sym(sol)
        
        return hess_out

    def _lr_loop_corrected(self, ndispl, g, hess_out, displdir):
        """
        CORRECTED: Low-rank correction loop with adaptive damping.
        
        Key fixes:
        1. Added adaptive damping factor
        2. Added stagnation detection
        3. Added divergence detection
        4. Proper convergence criteria 
        """
        N = g.shape[0]
        
        dampfac = 1.0  # FIXED: Added damping factor
        err0 = np.inf
        
        # Compute norm of gradients (for divergence detection)
        norm_g = np.linalg.norm(g[:, :ndispl])
        
        for it in range(self.maxiter_lr):
            # Compute residual: R = g - H * displdir
            tmp = np.dot(hess_out, displdir[:, :ndispl])
            resid = g[:, :ndispl] - tmp
            
            err = np.linalg.norm(resid)
            
            # Check convergence 
            if err < self.thresh_lr:
                break
            
            # FIXED: Check stagnation 
            if abs(err - err0) < self.thresh_lr * err0:
                break
            
            # FIXED: Check divergence and reduce damping
            if err > err0 and err > norm_g:
                dampfac *= 0.5
            
            # Compute symmetric correction
            hcorr = np.dot(resid, displdir[:, :ndispl].T)
            hcorr = 0.5 * (hcorr + hcorr.T)
            
            # FIXED: Apply damped update 
            hess_out = hess_out + dampfac * hcorr
            
            err0 = err
        
        final_err = err
        
        return hess_out, final_err

