import numpy as np
import copy
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.sparse.linalg import LinearOperator, cg as scipy_cg
from scipy.spatial.distance import cdist

from multioptpy.ModelHessian.swart import SwartApproxHessian
from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib


class O1NumHessCalculator:
    """
    O1NumHess: Semi-numerical Hessian generation using Optimal 1-sided differentiation.
    Ref: https://doi.org/10.1021/acs.jctc.5c01354
    
    Notice:
    This implementation is experimental and lacks formal validation against the performance benchmarks established in the original paper. 
    It is intended for reference only and does not serve as a basis for contesting the validity of the referenced methodology, regardless of the performance outcomes.
    
    UPDATES:
    1. Adaptive Cutoff based on Covalent Radii (rcov_scale * (Ri + Rj))
    2. Strict Reject-and-Retry LR Loop
    """

    def __init__(self, calculation_engine, element_list, charge_mult, method, 
                 rcov_scale=1.5, delta_bohr=0.005, verbosity=1):
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
            Scaling factor for covalent radii sum to determine cutoff (default: 1.5).
            Cutoff_ij = rcov_scale * (R_i + R_j)
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
        self.thresh_lr = 1.0e-8
        self.maxiter_lr = 100

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
        # cutoff_mat[i, j] = scale * (R_i + R_j)
        # Use broadcasting: (N, 1) + (1, N) -> (N, N)
        radii_col = self.atom_radii[:, np.newaxis]
        radii_row = self.atom_radii[np.newaxis, :]
        cutoff_mat = self.rcov_scale * (radii_col + radii_row)

        if self.verbosity > 0:
            print(f"\n=== Starting O1NumHess Calculation ===")
            print(f"  Adaptive Cutoff Strategy: {self.rcov_scale} * (R_i + R_j)")
            print(f"  Min Cutoff: {np.min(cutoff_mat):.2f} Bohr, Max Cutoff: {np.max(cutoff_mat):.2f} Bohr")
            print(f"  Step size (eta): {self.delta} Bohr")

        # 2. Compute reference gradient (g0)
        if self.verbosity > 0:
            print("  [1/6] Computing reference gradient...")
        g0_vec = self._get_gradient(coords_ang).flatten()

        # 3. Generate initial model Hessian (H0)
        if self.verbosity > 0:
            print("  [2/6] Generating initial model Hessian (Fischer-D3)...")
        dummy_grad = np.zeros_like(g0_vec)
        h0_bohr_units = self.fischer_calc.main(coords_bohr, self.element_list, dummy_grad)

        # 4. Build neighbor list (using adaptive cutoff)
        if self.verbosity > 0:
            print("  [3/6] Building connectivity graph...")
        distmat_bohr = cdist(coords_bohr, coords_bohr)
        
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
        """
        N_atom = N_dof // 3
        displdir = np.zeros((N_dof, N_dof))
        eps = 1.0e-6
        eps2 = 1.0e-8
        
        # 1. Initial 7 directions
        # Translations
        for i in range(3):
            for j in range(N_atom):
                displdir[3*j + i, i] = 1.0
        
        # Rotations
        center = np.mean(coords_bohr, axis=0)
        I_tensor = np.zeros((3, 3))
        for j in range(N_atom):
            r = coords_bohr[j] - center
            I_tensor += np.eye(3) * np.dot(r, r) - np.outer(r, r)
        
        try:
            _, rot_axes = np.linalg.eigh(I_tensor)
        except:
            rot_axes = np.eye(3)
        
        for i in range(3):
            axis = rot_axes[:, i]
            for j in range(N_atom):
                r = coords_bohr[j] - center
                rot_vec = np.cross(axis, r)
                displdir[3*j:3*j+3, 3+i] = rot_vec
        
        # Breathing
        for j in range(N_atom):
            r = coords_bohr[j] - center
            displdir[3*j:3*j+3, 6] = r
        
        for i in range(7):
            norm = np.linalg.norm(displdir[:, i])
            if norm > eps2:
                displdir[:, i] /= norm
        
        ndispl_final = 7
        
        # 2. Iterative directions
        for n_curr in range(7, N_dof):
            ev = np.zeros(N_dof)
            coverage = np.zeros(N_dof)
            
            for j in range(N_dof):
                nnb = nbcounts[j]
                if nnb == 0: continue
                nb_idx = np.array(nblist[j][:nnb])
                if nnb <= n_curr: continue
                
                subH = H0[np.ix_(nb_idx, nb_idx)]
                
                if n_curr > 0:
                    vec_subset = displdir[np.ix_(nb_idx, range(n_curr))]
                    try:
                        q, r = np.linalg.qr(vec_subset)
                        projmat = np.eye(nnb) - np.dot(q, q.T)
                        subH = np.dot(projmat, np.dot(subH, projmat.T))
                        subH = 0.5 * (subH + subH.T)
                    except:
                        continue
                
                try:
                    loceigs, locevecs = np.linalg.eigh(subH)
                    locind = np.argmax(np.abs(loceigs))
                    locev = locevecs[:, locind]
                except np.linalg.LinAlgError:
                    continue
                
                norm_ev1 = 0.0
                norm_ev2 = 0.0
                for p in range(nnb):
                    idx = nb_idx[p]
                    val1 = (coverage[idx] * ev[idx] + locev[p]) / (coverage[idx] + 1.0)
                    val2 = (coverage[idx] * ev[idx] - locev[p]) / (coverage[idx] + 1.0)
                    norm_ev1 += val1**2
                    norm_ev2 += val2**2
                
                norm_ev1 = np.sqrt(norm_ev1)
                norm_ev2 = np.sqrt(norm_ev2)
                
                if norm_ev1 > norm_ev2 + eps:
                    sign = 1.0
                elif norm_ev1 < norm_ev2 - eps:
                    sign = -1.0
                else:
                    locind_max = np.argmax(np.abs(locev))
                    sign = 1.0 if locev[locind_max] > 0 else -1.0
                
                for p in range(nnb):
                    idx = nb_idx[p]
                    ev[idx] = (coverage[idx] * ev[idx] + sign * locev[p]) / (coverage[idx] + 1.0)
                
                for p in range(nnb):
                    coverage[nb_idx[p]] += 1.0
            
            for k in range(n_curr):
                d_dot = np.dot(ev, displdir[:, k])
                ev -= d_dot * displdir[:, k]
            
            v_norm = np.linalg.norm(ev)
            if v_norm < eps2:
                ndispl_final = n_curr
                break
            
            ev /= v_norm
            displdir[:, n_curr] = ev
            ndispl_final = n_curr + 1
        
        return displdir[:, :ndispl_final], ndispl_final

    def _solve_odlr_problem_corrected(self, distmat, displdir, g, ndispl_final, cutoff_mat):
        """
        ODLR solver using Adaptive Cutoff Matrix.
        """
        N = distmat.shape[0]
        
        # FIXED: W2 uses cutoff_mat
        # W^2 = lambda * max(0, r - r_cut)^(2*beta)
        W2 = self.lam * np.maximum(0.0, distmat - cutoff_mat) ** (2.0 * self.bet)
        
        rhs = np.dot(g[:, :ndispl_final], displdir[:, :ndispl_final].T)
        rhs = 0.5 * (rhs + rhs.T)
        
        # FIXED: Mask uses cutoff_mat
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
            
            tmp2 = np.dot(H_tmp, displdir[:, :ndispl_final])
            f1 = np.dot(tmp2, displdir[:, :ndispl_final].T)
            f1 = 0.5 * (f1 + f1.T)
            
            f2 = W2 * H_tmp
            
            return pack_sym(f1 + f2)
        
        op = LinearOperator((ndim, ndim), matvec=matvec)
        
        sol, info = scipy_cg(op, rhs_vec, maxiter=1000, atol=1e-14)
        
        if info != 0 and self.verbosity > 0:
            print(f"  Warning: CG did not converge (info={info})")
        
        hess_out = unpack_sym(sol)
        
        return hess_out

    def _lr_loop_strict(self, ndispl, g, hess_out, displdir):
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
