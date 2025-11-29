import torch
import torch.nn.functional as F
import math
import numpy as np


class OverlapCalculator:
    def __init__(self, element_list=None, param=None, wf=None):
        """
        Initializes the OverlapCalculator with the given parameters.
        """
        self.element_list = element_list
        self.param = param
        self.wf = wf
        self.basis = wf.basis

        # --- Constants Initialization ---
        self.MAXL = 6
        self.MAXL2 = self.MAXL * 2

        # --- Cartesian Exponent Lookup Tables (Numpy arrays) ---
        self.LX = np.array([0, 
                            1,0,0, 
                            2,0,0,1,1,0, 
                            3,0,0,2,2,1,0,1,0,1, 
                            4,0,0,3,3,1,0,1,0,2,2,0,2,1,1, 
                            5,0,0,3,3,2,2,0,0,4,4,1,0,0,1,1,3,1,2,2,1, 
                            6,0,0,3,3,0,5,5,1,0,0,1,4,4,2,0,2,0,3,3,1,2,2,1,4,1,1,2], dtype=int)
        self.LY = np.array([0, 
                            0,1,0, 
                            0,2,0,1,0,1, 
                            0,3,0,1,0,2,2,0,1,1, 
                            0,4,0,1,0,3,3,0,1,2,0,2,1,2,1, 
                            0,5,0,2,0,3,0,3,2,1,0,4,4,1,0,1,1,3,2,1,2, 
                            0,6,0,3,0,3,1,0,0,1,5,5,2,0,0,2,4,4,2,1,3,1,3,2,1,4,1,2], dtype=int)
        self.LZ = np.array([0, 
                            0,0,1, 
                            0,0,2,0,1,1,
                            0,0,3,0,1,0,1,2,2,1, 
                            0,0,4,0,1,0,1,3,3,0,2,2,1,1,2,
                            0,0,5,0,2,0,3,2,3,0,1,0,1,4,4,3,1,1,1,2,2,
                            0,0,6,0,3,3,0,1,5,5,1,0,0,2,4,4,0,2,1,2,2,3,1,3,1,1,4,2], dtype=int)
        self.LXYZ_TYPE_NP = np.stack([self.LX, self.LY, self.LZ], axis=0).T
        self.LXYZ_TYPE_LIST = self.LXYZ_TYPE_NP.tolist()

        tmp_LXYZ = self.basis["ao_type_id_list"]
        
        self.LXYZ_LIST = []
        for tmp in tmp_LXYZ:
            self.LXYZ_LIST.append(self.LXYZ_TYPE_LIST[tmp - 1])
        
        self.LXYZ_TENSOR = torch.tensor(self.LXYZ_LIST, dtype=torch.int64)
        self.LXYZ_TYPE_TENSOR = torch.tensor(self.LXYZ_TYPE_LIST, dtype=torch.int64)
        
        # --- Shell Start Index Offsets (Tensor) ---
        # L=0, 1, 2, 3 (s, p, d, f)
        self.ITT = torch.tensor([0, 1, 4, 10], dtype=torch.int64)

        # --- Cartesian d -> Spherical d Transformation Matrix (Tensor) ---
        s5 = math.sqrt(1.0 / 5.0)
        s3 = math.sqrt(3.0)
        self.TRAFO_NP = np.array([
            [s5, s5, s5, 0.0, 0.0, 0.0],
            [0.5 * s3, -0.5 * s3, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        
        self.TRAFO = None
        self.TRAFO_T = None

        self.LLAO = np.array([1, 3, 6, 10], dtype=np.int64)  # Cartesian counts
        self.LLAO2 = np.array([1, 3, 5, 7], dtype=np.int64)  # Spherical counts
        
        # --- Double Factorial Table for 1D overlap integral ---
        # DFTR(lh) = (2*lh - 1)!!  for lh = 0, 1, 2, ...
        # This matches Fortran: dftr(0:7) = [1, 1, 3, 15, 105, 945, 10395, 135135]
        # where dftr(lh) = (2*lh - 1)!! 
        #   lh=0: (-1)!! = 1
        #   lh=1: 1!! = 1
        #   lh=2: 3!! = 3
        #   lh=3: 5!! = 15
        #   lh=4: 7!!  = 105
        #   lh=5: 9!! = 945
        #   lh=6: 11!! = 10395
        #   lh=7: 13!! = 135135
        self.DFTR = torch.tensor([
            1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0
        ], dtype=torch.float64)
        
        # --- Binomial Coefficients ---
        # self.BINO[l, m] = l!  / (m! * (l-m)!)
        m = torch.arange(self.MAXL + 1, dtype=torch.float64)
        l_fact = torch.lgamma(m + 1)
        l_vec = m.view(-1, 1)
        m_vec = m.view(1, -1)
        
        log_bino = l_fact[l_vec.long()] - l_fact[m_vec.long()] - l_fact[(l_vec - m_vec).long()]
        bino = torch.exp(log_bino).round()
        self.BINO = torch.where(l_vec < m_vec, 0.0, bino).to(torch.float64)


    def calc_1d_overlap_constants(self, l, gamma): 
        """
        Calculates the 1D overlap integral factor (double factorial part).
        Corresponds to Fortran: olapp(l, gama)
        
        Returns 0 for odd l, otherwise returns (0.5/gamma)^(l/2) * (l-1)!!
        """
        if l % 2 != 0:
            s = 0.0
        else:
            lh = l // 2
            gm = 0.5 / gamma
            # Use DFTR[lh] which gives (2*lh - 1)!! = (l - 1)!! 
            if self.DFTR.device != (gamma.device if isinstance(gamma, torch.Tensor) else 'cpu'):
                self.DFTR = self.DFTR.to(gamma.device if isinstance(gamma, torch.Tensor) else 'cpu')
            dftr_val = self.DFTR[lh]
            s = (gm ** lh) * dftr_val
        if isinstance(gamma, torch.Tensor) and not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=gamma.dtype, device=gamma.device)
        return s
    
   
    def calc_1d_overlap_constants_vec(self, l: int, gamma: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of calc_1d_overlap_constants.
        Corresponds to Fortran: olapp(l, gama)
        """
        if l % 2 != 0:
            return torch.zeros_like(gamma)
        else:
            lh = l // 2
            gm = 0.5 / gamma
            
            # Ensure DFTR is on the correct device
            if self.DFTR.device != gamma.device:
                self.DFTR = self.DFTR.to(gamma.device)
                
            dftr_val = self.DFTR[lh]  # (2*lh - 1)!! = (l - 1)!!
            return (gm ** lh) * dftr_val
            
    def calc_product_exponent_and_overlap_vec(self, alpha_i, r_i, alpha_j, r_j): 
        """
        Vectorized: Calculates product exponent gamma and s-orbital overlap Kab.
        Corresponds to Fortran: build_kab(ra, alp, rb, bet, gama, kab)
        """
        # alpha_i: (Ni, 1), alpha_j: (1, Nj)
        # r_i, r_j: (3,)
        gamma = alpha_i + alpha_j  # (Ni, Nj)
        inv_gamma = 1.0 / gamma
        
        r_ij = r_i - r_j
        r_ij_2 = torch.dot(r_ij, r_ij)  # scalar
        
        est = r_ij_2 * alpha_i * alpha_j * inv_gamma  # (Ni, Nj)
        # kab = exp(-est) * (sqrt(pi) * sqrt(1/gamma))^3
        k_ab = torch.exp(-est) * (torch.pi * inv_gamma) ** 1.5  # (Ni, Nj)
        return gamma, k_ab

    def calc_gau_product_center_vec(self, alpha_i, r_i, alpha_j, r_j, gamma):
        """
        Vectorized: Calculates Gaussian product center P = (aA + bB) / (a + b).
        Corresponds to Fortran: gpcenter(alp, ra, bet, rb)
        """
        # alpha_i: (Ni, 1, 1), alpha_j: (1, Nj, 1)
        # r_i, r_j: (1, 1, 3)
        # gamma: (Ni, Nj, 1)
        r_gpc = (alpha_i * r_i + alpha_j * r_j) / gamma  # (Ni, Nj, 3)
        return r_gpc

    def _primitive_norm(self, alphas: torch.Tensor, lmn_mat: torch.Tensor) -> torch.Tensor:
        """
        Vectorized normalization factor for Cartesian Gaussian primitives.
        N = (2*alpha/pi)^(3/4) * (4*alpha)^(L/2) / sqrt((2lx-1)!! * (2ly-1)!! * (2lz-1)! !)
        """
        # (Ncomp,)
        lx, ly, lz = lmn_mat[:, 0], lmn_mat[:, 1], lmn_mat[:, 2]
        L = lx + ly + lz  # (Ncomp,)
        
        # Ensure DFTR is on the correct device
        if self.DFTR.device != alphas.device:
            self.DFTR = self.DFTR.to(alphas.device)
        
        # DFTR[lx] = (2*lx - 1)!! (with DFTR[0] = (-1)!! = 1)
        dftr_x = self.DFTR[lx.long()]  # (Ncomp,)
        dftr_y = self.DFTR[ly.long()]
        dftr_z = self.DFTR[lz.long()]
        
        # Add epsilon to avoid sqrt(0) -> NaN gradients
        den = torch.sqrt(dftr_x * dftr_y * dftr_z + 1e-30)  # (Ncomp,)
        
        pre = (2.0 * alphas / math.pi) ** 0.75  # (Ni,)
        
        # (Ni, 1) ** (1, Ncomp) -> (Ni, Ncomp)
        pow_term = (4.0 * alphas[:, None]) ** (L[None, :] / 2.0)
        
        # (Ni, 1) * (Ni, Ncomp) / (1, Ncomp) -> (Ni, Ncomp)
        return (pre[:, None] * pow_term) / den[None, :]

    def _ensure_trafo_device(self, device):
        """ Helper to move TRAFO matrix to the correct device once. """
        if self.TRAFO is None or self.TRAFO.device != device:
            self.TRAFO = torch.tensor(self.TRAFO_NP, dtype=torch.float64, device=device).contiguous()
            self.TRAFO_T = self.TRAFO.T.contiguous()

    def transform_d_cartesian_to_spherical(self, int_mat, l_i, l_j):
        """
        Transforms d-orbital block from Cartesian to Spherical basis.
        Corresponds to Fortran: dtrf2(s, li, lj)
        """
        if l_i < 2 and l_j < 2:
            return int_mat
            
        self._ensure_trafo_device(int_mat.device)
        trafo = self.TRAFO
        trafo_t = self.TRAFO_T
        
        if l_i == 0:  # s-d
            s_out = torch.matmul(trafo, int_mat[0:6, 0:1])
        elif l_i == 1:  # p-d
            s_out = torch.matmul(trafo, int_mat[0:6, 0:3])
        elif l_j == 0:  # d-s
            s_out = torch.matmul(int_mat[0:1, 0:6], trafo_t)
        elif l_j == 1:  # d-p
            s_out = torch.matmul(int_mat[0:3, 0:6], trafo_t)
        else:  # d-d
            dum = torch.matmul(trafo, int_mat[0:6, 0:6])
            s_out = torch.matmul(dum, trafo_t)

        return s_out
    
    def calculate_center_shift_coefficients_vec(self, cfs_in, a, e, l):
        """ 
        Vectorized: Calculates coefficients for center shift using binomial expansion.
        Corresponds to Fortran: build_hshift2(cfs, a, e, l) / horizontal_shift
        
        (x - a)^l = ((x - e) + (e - a))^l = sum_m [ C(l,m) * (e-a)^(l-m) * (x-e)^m ]
        Returns coefficients c_m = C(l,m) * (e-a)^(l-m)
        """
        if l > self.MAXL:
            raise NotImplementedError(f"calculate_center_shift_coefficients_vec not implemented for l = {l} > MAXL = {self.MAXL}")

        ae = e - a  # (Ni, Nj)
        val_l = cfs_in[l]  # scalar (1.0)
        
        if l == 0:
            return val_l.expand_as(ae).unsqueeze(-1)  # (Ni, Nj, 1)

        # Ensure BINO is on the correct device
        if self.BINO.device != ae.device:
            self.BINO = self.BINO.to(ae.device)
        
        # m = 0, 1, ..., l
        m = torch.arange(l + 1, device=ae.device)  # (l+1,)
        
        # Binomial coefficients C(l, m)
        bino_coeffs = self.BINO[l, m].view(1, 1, -1)  # (1, 1, l+1)
        
        # Powers (e-a)^(l-m)
        l_minus_m = (l - m).view(1, 1, -1)  # (1, 1, l+1)
        ae_b = ae.unsqueeze(-1)  # (Ni, Nj, 1)
        ae_powers = ae_b ** l_minus_m  # (Ni, Nj, l+1)
        
        # c_m = C(l,m) * (e-a)^(l-m) * val_l (where val_l is 1.0)
        coeffs = bino_coeffs * ae_powers * val_l
        
        return coeffs  # (Ni, Nj, l+1)

    def calculate_center_shift_coefficients_batch_l(
        self, l_vec: torch.Tensor, a: torch.Tensor, e: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Vectorized: Calculates center shift coefficients for a *batch* of l values.
        """
        Ni, Nj = e.shape[0:2]
        Ncomp = l_vec.shape[0]
        
        if Ncomp == 0:
            return torch.empty((Ni, Nj, 0, 0), dtype=e.dtype, device=e.device)
            
        if l_vec.max().item() > self.MAXL:
            raise NotImplementedError(f"l_vec max {l_vec.max()} > MAXL {self.MAXL}")

        ae = e - a  # (Ni, Nj, 3) or (Ni, Nj)
        if ae.dim() == 2:
            ae = ae.unsqueeze(-1)  # (Ni, Nj, 1)
        
        L_max = l_vec.max().item()
        if L_max < 0:
            return torch.empty((Ni, Nj, Ncomp, 0), dtype=e.dtype, device=e.device)
             
        m = torch.arange(L_max + 1, device=ae.device)  # (L_max+1,)
        
        # Ensure BINO is on the correct device
        if self.BINO.device != ae.device:
            self.BINO = self.BINO.to(ae.device)

        # C(l, m) -> (Ncomp, L_max+1)
        bino_coeffs = self.BINO[l_vec.long()][:, :L_max+1]
        
        # (e-a)^(l-m)
        l_vec_b = l_vec.view(Ncomp, 1)  # (Ncomp, 1)
        m_b = m.view(1, L_max+1)       # (1, L_max+1)
        
        l_minus_m_raw = l_vec_b - m_b  # (Ncomp, L_max+1)
        
        # Create mask for m > l, where l_minus_m is negative
        mask = m_b > l_vec_b  # (Ncomp, L_max+1)
        
        l_minus_m = torch.where(mask, 0.0, l_minus_m_raw)
        
        ae_b = ae.unsqueeze(-1)  # (Ni, Nj, 3 or 1, 1)
        ae_powers_raw = ae_b ** l_minus_m[None, None, :, :]
        ae_powers = torch.where(mask[None, None, :, :], 0.0, ae_powers_raw)

        # C(l,m) * (e-a)^(l-m)
        coeffs = bino_coeffs[None, None, :, :] * ae_powers
        
        return coeffs  # (Ni, Nj, Ncomp, L_max+1)

    def compute_1d_product_coefficients_vec(self, coeff_a, coeff_b, la, lb):
        """ 
        Vectorized 1D convolution of coefficient arrays using F.conv1d.
        Corresponds to Fortran: form_product / prod3
        """
        Ni, Nj, Na = coeff_a.shape
        _, _, Nb = coeff_b.shape
        output_size = la + lb + 1
        
        N_batch = Ni * Nj
        if N_batch == 0:
            return torch.empty((Ni, Nj, output_size), dtype=coeff_a.dtype, device=coeff_a.device)
            
        coeff_a_flat = coeff_a.view(1, N_batch, Na)
        coeff_b_flat_rev = torch.flip(coeff_b, dims=[2]).view(N_batch, 1, Nb)
        
        conv_out = F.conv1d(
            coeff_a_flat,
            coeff_b_flat_rev,
            padding=lb, 
            groups=N_batch
        )
        
        return conv_out.view(Ni, Nj, output_size)

    def compute_1d_product_coefficients_batch_outer(
        self, coeff_a: torch.Tensor, coeff_b: torch.Tensor, 
        la_vec: torch.Tensor, lb_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized 1D convolution for an outer product of coefficient arrays.
        """
        Ni, Nj, naoi, Na_max = coeff_a.shape
        _, _, naoj, Nb_max = coeff_b.shape
        
        max_la = la_vec.max().item() if naoi > 0 else -1
        max_lb = lb_vec.max().item() if naoj > 0 else -1
        N_out_max = max_la + max_lb + 1
        
        if Ni*Nj*naoi*naoj == 0:
            return torch.empty((Ni, Nj, naoi, naoj, N_out_max), dtype=coeff_a.dtype, device=coeff_a.device)

        a_b = coeff_a.unsqueeze(3).expand(-1, -1, -1, naoj, -1)
        b_b = coeff_b.unsqueeze(2).expand(-1, -1, naoi, -1, -1)
        
        N_batch = Ni * Nj * naoi * naoj
        a_flat = a_b.reshape(1, N_batch, Na_max)
        b_flat_rev = b_b.reshape(N_batch, 1, Nb_max).flip(dims=[-1])
        
        conv_out = F.conv1d(
            a_flat,
            b_flat_rev,
            padding=Nb_max - 1,
            groups=N_batch
        )
        
        return conv_out.view(Ni, Nj, naoi, naoj, -1)[..., 0:N_out_max]
    
    def _build_s1d_factors_vec(self, max_l: int, gama_mat: torch.Tensor) -> torch.Tensor:
        """
        Vectorized: Creates array of 1D integral factors (olapp results).
        """
        s1d_factors = []
        for l in range(max_l + 1):
            factor = self.calc_1d_overlap_constants_vec(l, gama_mat)
            s1d_factors.append(factor)
        return torch.stack(s1d_factors, dim=-1)  # (Ni, Nj, max_l+1)

    def _assemble_3d_multipole_factors_vec(
        self,
        ri: torch.Tensor, rj: torch.Tensor, rp: torch.Tensor, rc: torch.Tensor,
        li_mat: torch.Tensor, lj_mat: torch.Tensor,
        s1d_factors: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Vectorized Python (PyTorch) version of multipole_3d subroutine.
        Corresponds to Fortran: multipole_3d(ri, rj, rc, rp, ai, aj, li, lj, s1d, s3d)
        """
        Ni, Nj, _ = rp.shape
        naoi = li_mat.shape[0]
        naoj = lj_mat.shape[0]
        tensor_opts = {'dtype': ri.dtype, 'device': ri.device}
        
        if naoi == 0 or naoj == 0:
            return torch.empty((Ni, Nj, naoi, naoj, 10), **tensor_opts)

        val_S_k = torch.zeros((Ni, Nj, naoi, naoj, 3), **tensor_opts)
        val_M_k = torch.zeros((Ni, Nj, naoi, naoj, 3), **tensor_opts)
        val_Q_k = torch.zeros((Ni, Nj, naoi, naoj, 3), **tensor_opts)
        
        for k in range(3):
            l_i_k_vec = li_mat[:, k]  # (naoi,)
            l_j_k_vec = lj_mat[:, k]  # (naoj,)
            
            max_l_i_k = l_i_k_vec.max().item()
            max_l_j_k = l_j_k_vec.max().item()
            
            center_i_k = ri[k]
            center_j_k = rj[k]
            center_p_k = rp[..., k]
            center_c_k = rc[k]
            rpc_k = center_p_k - center_c_k

            vi_shifted = self.calculate_center_shift_coefficients_batch_l(
                l_i_k_vec, center_i_k, center_p_k)
            vj_shifted = self.calculate_center_shift_coefficients_batch_l(
                l_j_k_vec, center_j_k, center_p_k)
            
            l_total_max = max_l_i_k + max_l_j_k
            pad_out = l_total_max + 1
            
            vv = self.compute_1d_product_coefficients_batch_outer(
                vi_shifted, vj_shifted, l_i_k_vec, l_j_k_vec)
            
            l_total_mat = l_i_k_vec[:, None] + l_j_k_vec[None, :] 
            
            vv_subset_raw = vv[..., 0:pad_out]
            s1d_L = s1d_factors[..., 0:pad_out]
            s1d_Lplus1 = s1d_factors[..., 1:pad_out + 1]
            s1d_Lplus2 = s1d_factors[..., 2:pad_out + 2]
            
            m_vec = torch.arange(pad_out, device=ri.device)[None, None, :]
            l_total_b = l_total_mat.unsqueeze(-1)
            mask = m_vec > l_total_b
            mask_b = mask[None, None, :, :, :]
            
            vv_subset = torch.where(mask_b, 0.0, vv_subset_raw)

            rpc_b = rpc_k[..., None, None, None]
            s1d_L_b = s1d_L[..., None, None, :]
            s1d_Lplus1_b = s1d_Lplus1[..., None, None, :]
            s1d_Lplus2_b = s1d_Lplus2[..., None, None, :]
            
            val_S_k[..., k] = torch.sum(vv_subset * s1d_L_b, dim=-1)
            
            dipole_s1d_vec = s1d_Lplus1_b + rpc_b * s1d_L_b
            val_M_k[..., k] = torch.sum(vv_subset * dipole_s1d_vec, dim=-1)
            
            quad_s1d_vec = s1d_Lplus2_b + 2.0 * rpc_b * s1d_Lplus1_b + (rpc_b**2) * s1d_L_b
            val_Q_k[..., k] = torch.sum(vv_subset * quad_s1d_vec, dim=-1)
            
        S_x, S_y, S_z = val_S_k[..., 0], val_S_k[..., 1], val_S_k[..., 2]
        M_x, M_y, M_z = val_M_k[..., 0], val_M_k[..., 1], val_M_k[..., 2]
        Q_x, Q_y, Q_z = val_Q_k[..., 0], val_Q_k[..., 1], val_Q_k[..., 2]

        s3d_factors = torch.stack([
            S_x * S_y * S_z, M_x * S_y * S_z, S_x * M_y * S_z, S_x * S_y * M_z,
            Q_x * S_y * S_z, S_x * Q_y * S_z, S_x * S_y * Q_z,
            M_x * M_y * S_z, M_x * S_y * M_z, S_x * M_y * M_z
        ], dim=-1)
        
        return s3d_factors


    def compute_contracted_overlap_matrix(
        self,
        ish: int, jsh: int, naoi_cart: int, naoj_cart: int,
        ishtyp: int, jshtyp: int, ri: torch.Tensor, rj: torch.Tensor,
        point: torch.Tensor, intcut: float,
        basis: dict, alp_tensor: torch.Tensor, cont_tensor: torch.Tensor,
        prim_indices_i: list, prim_indices_j: list,
        itt_device: torch.Tensor, lxyz_type_device: torch.Tensor,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Computes the overlap matrix block between two *Shells* (ish, jsh).
        Corresponds to Fortran: get_overlap(...)
        """
        if device is None:
            device = ri.device
        tensor_opts = {'dtype': ri.dtype, 'device': device}
        
        rij = ri - rj
        rij2 = torch.dot(rij, rij)
        max_r2 = 2000.0
        if rij2 > max_r2: 
            return torch.zeros((naoj_cart, naoi_cart), **tensor_opts)

        iptyp = itt_device[ishtyp].item()
        jptyp = itt_device[jshtyp].item()

        Ni = len(prim_indices_i)
        Nj = len(prim_indices_j)
        if Ni == 0 or Nj == 0:
            return torch.zeros((naoj_cart, naoi_cart), **tensor_opts)
            
        alp_i_vec = alp_tensor[prim_indices_i]
        cont_i_vec = cont_tensor[prim_indices_i]
        alp_j_vec = alp_tensor[prim_indices_j]
        cont_j_vec = cont_tensor[prim_indices_j]
        
        alp_i_b = alp_i_vec.unsqueeze(1)
        alp_j_b = alp_j_vec.unsqueeze(0)
        
        gamma_mat, kab_mat = self.calc_product_exponent_and_overlap_vec(alp_i_b, ri, alp_j_b, rj)
        
        rp_mat = self.calc_gau_product_center_vec(
            alp_i_b.unsqueeze(2), ri.view(1, 1, 3),
            alp_j_b.unsqueeze(2), rj.view(1, 1, 3),
            gamma_mat.unsqueeze(2)
        )
        
        max_l_k_prim = ishtyp + jshtyp
        max_l_needed_prim = max_l_k_prim + 2
        t_factors_mat = self._build_s1d_factors_vec(max_l_needed_prim, gamma_mat)

        li_mat = lxyz_type_device[iptyp:iptyp + naoi_cart]
        lj_mat = lxyz_type_device[jptyp:jptyp + naoj_cart]
        
        n_i_mat = self._primitive_norm(alp_i_vec, li_mat)
        n_j_mat = self._primitive_norm(alp_j_vec, lj_mat)
        
        c_n_i = cont_i_vec[:, None] * n_i_mat
        c_n_j = cont_j_vec[:, None] * n_j_mat
        
        cc_full_mat = c_n_i[:, None, :, None] * c_n_j[None, :, None, :]
        
        saw_factors_mat = self._assemble_3d_multipole_factors_vec(
            ri, rj, rp_mat, point,
            li_mat, lj_mat,
            t_factors_mat
        )
        
        s_prim_full = kab_mat[..., None, None] * saw_factors_mat[..., 0] 
        
        sint_block_cart_T = torch.sum(s_prim_full * cc_full_mat, dim=(0, 1))
        
        sint_block_cart = sint_block_cart_T.T
        
        return sint_block_cart


    def calculate_overlap_matrix_full(self, xyz: torch.Tensor, intcut) -> torch.Tensor:
        """ Calculates the full molecular overlap matrix S (nao, nao). """
        if xyz.shape[0] == 3 and xyz.shape[1] != 3:
            xyz = xyz.T
        n_atoms = xyz.shape[0]
        current_device = xyz.device

        n_ao = self.basis['number_of_ao']
        n_shells = self.basis['number_of_shells']
        tensor_opts = {'dtype': xyz.dtype, 'device': current_device}
        sint = torch.zeros((n_ao, n_ao), **tensor_opts)

        LLAO = torch.tensor(self.LLAO, dtype=torch.int64, device=current_device)
        point = torch.zeros(3, **tensor_opts)

        shell_amqn_list = torch.tensor(self.basis['shell_amqn_list'], dtype=torch.int64, device=current_device)
        shell_atom_list = torch.tensor(self.basis['shell_atom_list'], dtype=torch.int64, device=current_device)
        shell_ao_map = torch.tensor(self.basis['shell_ao_map'], dtype=torch.int64, device=current_device)
        alp = self.basis.get('primitive_alpha_list', [])
        alp_tensor = torch.tensor(alp, **tensor_opts)
        cont = self.basis.get('primitive_coeff_list', [])
        cont_tensor = torch.tensor(cont, **tensor_opts)
        
        itt_device = self.ITT.to(current_device)
        lxyz_type_device = self.LXYZ_TYPE_TENSOR.to(current_device)
        self.DFTR = self.DFTR.to(current_device)

        shell_cgf_map = torch.tensor(self.basis['shell_cgf_map'], dtype=torch.int64, device=current_device)
        cgf_primitive_count_list = torch.tensor(self.basis['cgf_primitive_count_list'], dtype=torch.int64, device=current_device)
        cgf_primitive_start_idx_list = torch.tensor(self.basis['cgf_primitive_start_idx_list'], dtype=torch.int64, device=current_device)
        
        shell_prim_indices = []
        for ish in range(n_shells):
            indices = []
            icgf_start, icgf_count = shell_cgf_map[ish]
            for icgf_local in range(icgf_count.item()):
                icgf = icgf_start.item() + icgf_local
                nprim = cgf_primitive_count_list[icgf].item()
                start = cgf_primitive_start_idx_list[icgf].item()
                indices.extend(range(start, start + nprim))
            shell_prim_indices.append(indices)

        for ish in range(n_shells):
            iat_idx = shell_atom_list[ish]
            ri = xyz[iat_idx]
            ishtyp = shell_amqn_list[ish].item()
            naoi_cart = LLAO[ishtyp].item()
            iao_start, naoi_spher = shell_ao_map[ish]
            iao_start, naoi_spher = iao_start.item(), naoi_spher.item()
            iao_slice = slice(iao_start, iao_start + naoi_spher)
            
            prim_indices_i = shell_prim_indices[ish]

            for jsh in range(ish + 1):
                jat_idx = shell_atom_list[jsh]
                rj = xyz[jat_idx]
                jshtyp = shell_amqn_list[jsh].item()
                naoj_cart = LLAO[jshtyp].item()
                jao_start, naoj_spher = shell_ao_map[jsh]
                jao_start, naoj_spher = jao_start.item(), naoj_spher.item()
                jao_slice = slice(jao_start, jao_start + naoj_spher)
                
                prim_indices_j = shell_prim_indices[jsh]
                
                ss_cart_shell = self.compute_contracted_overlap_matrix(
                    ish=ish, jsh=jsh,
                    naoi_cart=naoi_cart, naoj_cart=naoj_cart,
                    ishtyp=ishtyp, jshtyp=jshtyp,
                    ri=ri, rj=rj, point=point, intcut=intcut,
                    basis=self.basis,
                    alp_tensor=alp_tensor, cont_tensor=cont_tensor,
                    prim_indices_i=prim_indices_i, prim_indices_j=prim_indices_j,
                    itt_device=itt_device, lxyz_type_device=lxyz_type_device,
                    device=current_device
                )

                ss_spher_padded = self.transform_d_cartesian_to_spherical(ss_cart_shell, ishtyp, jshtyp)
                
                i_slice_start = 1 if ishtyp == 2 else 0
                j_slice_start = 1 if jshtyp == 2 else 0
                i_slice_end = i_slice_start + naoi_spher
                j_slice_end = j_slice_start + naoj_spher
                
                ss_spher_shell = ss_spher_padded[j_slice_start:j_slice_end, i_slice_start:i_slice_end]

                if ish == jsh:
                    sint[jao_slice, iao_slice] = sint[jao_slice, iao_slice] + ss_spher_shell
                else:
                    sint[jao_slice, iao_slice] = sint[jao_slice, iao_slice] + ss_spher_shell
                    sint[iao_slice, jao_slice] = sint[iao_slice, jao_slice] + ss_spher_shell.T

        sint = self.normalize_overlap_matrix(sint)
        return sint

    def normalize_overlap_matrix(self, sint: torch.Tensor) -> torch.Tensor:
        """ Normalizes the overlap matrix S to have ones on the diagonal."""
        diag = torch.diagonal(sint)
        diag_safe = diag + 1e-20
        inv_sqrt_diag = 1.0 / torch.sqrt(diag_safe)
        inv_sqrt_diag = torch.where(diag > 1e-12, inv_sqrt_diag, torch.zeros_like(diag))
        
        D_inv_sqrt = torch.diag(inv_sqrt_diag)
        sint_normalized = torch.matmul(D_inv_sqrt, torch.matmul(sint, D_inv_sqrt))
        return sint_normalized


    def calculation(self, xyz: torch.Tensor, intcut=40):
        """ Main calculation entry point, returns S, D(zeros), Q(zeros)."""
        sint = self.calculate_overlap_matrix_full(xyz, intcut)
        n_ao = sint.shape[0]
        dpint = torch.zeros((3, n_ao, n_ao), dtype=sint.dtype, device=sint.device)
        qpint = torch.zeros((6, n_ao, n_ao), dtype=sint.dtype, device=sint.device)
        
        self.sint = sint
        return sint, dpint, qpint

    def get_overlap_integral_matrix(self) -> torch.Tensor:
        """ Returns the last computed overlap integral matrix S."""
        if not hasattr(self, 'sint'):
            print("Warning: Overlap integral matrix S has not been computed yet. Returning None.")
            return None
        return self.sint

    def overlap_int(self, xyz: np.ndarray) -> torch.Tensor:
        """ Calculates overlap matrix from NumPy coordinate array."""
        xyz_torch = torch.tensor(xyz, dtype=torch.float64, requires_grad=False)
        sint, _, _ = self.calculation(xyz_torch)
        return sint

    def overlap_int_torch(self, xyz: torch.Tensor) -> torch.Tensor:
        """ Calculates overlap matrix from Torch Tensor coordinates."""
        sint = self._get_sint_only(xyz)
        return sint
    
    def _get_sint_only(self, xyz: torch.Tensor) -> torch.Tensor:
        """ Helper function for jacrev/hessian to get S. """
        sint, _, _ = self.calculation(xyz)
        return sint

    def d_overlap_int_dxyz(self, xyz: np.ndarray) -> torch.Tensor:
        """ Calculates the Jacobian of the overlap matrix w.r.t.nuclear coordinates."""
        xyz_torch = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        d_sint_dxyz = torch.func.jacrev(self._get_sint_only, argnums=0)(xyz_torch)
        return d_sint_dxyz

    def d2_overlap_int_dxyz2(self, xyz: np.ndarray) -> torch.Tensor:
        """ Calculates the Hessian of the overlap matrix w.r.t.nuclear coordinates. """
        xyz_torch = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        d2_sint_sum_dxyz2 = torch.func.hessian(self._get_sint_only, argnums=0)(xyz_torch)
        return d2_sint_sum_dxyz2