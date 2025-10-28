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
        self.LXYZ_TYPE_NP = np.stack([self.LX, self.LY, self.LZ], axis=0).T # Keep numpy for list comprehension
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
        
        # We will convert these to tensors on the correct device later
        self.TRAFO = None
        self.TRAFO_T = None

        self.LLAO = np.array([1, 3, 6, 10], dtype=np.int64) # Cartesian counts
        self.LLAO2 = np.array([1, 3, 5, 7], dtype=np.int64) # Spherical counts
        
     
        self.F2 = torch.tensor([
            1.0, 1.0, 2.0, 3.0, 8.0, 15.0, 48.0, 105.0, 384.0, 945.0, 3840.0,
            10395.0, 46080.0, 135135.0, 645120.0, 2027025.0, 10321920.0
        ], dtype=torch.float64)
        # self.F2[l] = (l-1)!!  (with F2[0] = (-1)!! = 1)
        
      
        # self.BINO[l, m] = l! / (m! * (l-m)!)
        m = torch.arange(self.MAXL + 1, dtype=torch.float64)
        l_fact = torch.lgamma(m + 1)
        l_vec = m.view(-1, 1)
        m_vec = m.view(1, -1)
        
        # Use log-gamma for numerical stability
        log_bino = l_fact[l_vec.long()] - l_fact[m_vec.long()] - l_fact[(l_vec - m_vec).long()]
        bino = torch.exp(log_bino).round()
        # Set C(l, m) = 0 if m > l
        self.BINO = torch.where(l_vec < m_vec, 0.0, bino).to(torch.float64)


   
    def calc_1d_overlap_constants(self, l, gamma): 
        """ Calculates the 1D overlap integral factor (double factorial part). """
        if l % 2 != 0: # Returns 0 for odd l
            s = 0.0
        else:      # Calculation for even l
            lh = l // 2
            gm = 0.5 / gamma
            f2 = self.F2[l] # Use precomputed
            s = (gm ** lh) * f2
        if isinstance(gamma, torch.Tensor) and not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=gamma.dtype, device=gamma.device)
        return s
    
   
    def calc_1d_overlap_constants_vec(self, l: int, gamma: torch.Tensor) -> torch.Tensor:
        """ Vectorized version of calc_1d_overlap_constants. """
        if l % 2 != 0:
            return torch.zeros_like(gamma)
        else:
            lh = l // 2
            gm = 0.5 / gamma
            
            # Ensure F2 is on the correct device
            if self.F2.device != gamma.device:
                self.F2 = self.F2.to(gamma.device)
                
            f2 = self.F2[l]
            return (gm ** lh) * f2
            
    def calc_product_exponent_and_overlap_vec(self, alpha_i, r_i, alpha_j, r_j): 
        """ Vectorized: Calculates product exponent gamma and s-orbital overlap Kab. """
        # alpha_i: (Ni, 1), alpha_j: (1, Nj)
        # r_i, r_j: (3,)
        gamma = alpha_i + alpha_j # (Ni, Nj)
        inv_gamma = 1.0 / gamma
        
        r_ij = r_i - r_j
        r_ij_2 = torch.dot(r_ij, r_ij) # scalar
        
        est = r_ij_2 * alpha_i * alpha_j * inv_gamma # (Ni, Nj)
        k_ab = torch.exp(-est) * (torch.pi * inv_gamma) ** 1.5 # (Ni, Nj)
        return gamma, k_ab

    def calc_gau_product_center_vec(self, alpha_i, r_i, alpha_j, r_j, gamma):
        """ Vectorized: Calculates Gaussian product center P = (aA + bB) / (a + b). """
        # alpha_i: (Ni, 1, 1), alpha_j: (1, Nj, 1)
        # r_i, r_j: (1, 1, 3)
        # gamma: (Ni, Nj, 1)
        r_gpc = (alpha_i * r_i + alpha_j * r_j) / gamma # (Ni, Nj, 3)
        return r_gpc

    def _primitive_norm(self, alphas: torch.Tensor, lmn_mat: torch.Tensor) -> torch.Tensor:
        """
        Vectorized over both alphas and multiple lmn.
        alphas: (Ni,)
        lmn_mat: (Ncomp, 3)
        Returns: (Ni, Ncomp)
        """
        # (Ncomp,)
        lx, ly, lz = lmn_mat[:, 0], lmn_mat[:, 1], lmn_mat[:, 2]
        L = lx + ly + lz  # (Ncomp,)
        
        # Ensure F2 is on the correct device
        if self.F2.device != alphas.device:
            self.F2 = self.F2.to(alphas.device)
            
        f2_x = self.F2[2 * lx.long()]  # (Ncomp,)
        f2_y = self.F2[2 * ly.long()]
        f2_z = self.F2[2 * lz.long()]
        
        # Add epsilon to avoid sqrt(0) -> NaN gradients
        den = torch.sqrt(f2_x * f2_y * f2_z + 1e-30)  # (Ncomp,)
        
        if torch.any(den == 0.0):
            # This should not happen with F2 table, but good to check
            raise ValueError(f"Denominator is zero for lmn={lmn_mat[den == 0.0]}")

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
        """ Transforms d-orbital block from Cartesian to Spherical basis. """
        if l_i < 2 and l_j < 2:
            return int_mat
            
        self._ensure_trafo_device(int_mat.device)
        trafo = self.TRAFO
        trafo_t = self.TRAFO_T
        
        if l_i == 0: # s-d
            s_out = torch.matmul(trafo, int_mat[0:6, 0:1])
        elif l_i == 1: # p-d
            s_out = torch.matmul(trafo, int_mat[0:6, 0:3])
        elif l_j == 0: # d-s
            s_out = torch.matmul(int_mat[0:1, 0:6], trafo_t)
        elif l_j == 1: # d-p
            s_out = torch.matmul(int_mat[0:3, 0:6], trafo_t)
        else: # d-d
            dum = torch.matmul(trafo, int_mat[0:6, 0:6])
            s_out = torch.matmul(dum, trafo_t)

        return s_out
    
    def calculate_center_shift_coefficients_vec(self, cfs_in, a, e, l): # build_hshift2 (vectorized)
        """ 
        Vectorized: Calculates coefficients for center shift using binomial expansion.
        (x - a)^l = ((x - e) + (e - a))^l = sum_m [ C(l,m) * (e-a)^(l-m) * (x-e)^m ]
        Returns coefficients c_m = C(l,m) * (e-a)^(l-m)
        """
        # cfs_in: (l+1,) - one-hot vector (only cfs_in[l]=1.0 matters)
        # a, e: scalars or (Ni, Nj) tensors
        # l: int
        
        if l > self.MAXL:
            raise NotImplementedError(f"calculate_center_shift_coefficients_vec not implemented for l = {l} > MAXL = {self.MAXL}")

        ae = e - a # (Ni, Nj)
        val_l = cfs_in[l] # scalar (1.0)
        
        if l == 0:
            # (e-a)^0 = 1.0. Coeff array is [1.0]
            return val_l.expand_as(ae).unsqueeze(-1) # (Ni, Nj, 1)

        # Ensure BINO is on the correct device
        if self.BINO.device != ae.device:
            self.BINO = self.BINO.to(ae.device)
        
        # m = 0, 1, ..., l
        m = torch.arange(l + 1, device=ae.device) # (l+1,)
        
        # Binomial coefficients C(l, m)
        bino_coeffs = self.BINO[l, m].view(1, 1, -1) # (1, 1, l+1)
        
        # Powers (e-a)^(l-m)
        l_minus_m = (l - m).view(1, 1, -1) # (1, 1, l+1)
        ae_b = ae.unsqueeze(-1) # (Ni, Nj, 1)
        ae_powers = ae_b ** l_minus_m # (Ni, Nj, l+1)
        
        # c_m = C(l,m) * (e-a)^(l-m) * val_l (where val_l is 1.0)
        coeffs = bino_coeffs * ae_powers * val_l
        
        return coeffs # (Ni, Nj, l+1)

    def calculate_center_shift_coefficients_batch_l(
        self, l_vec: torch.Tensor, a: torch.Tensor, e: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Vectorized: Calculates center shift coefficients for a *batch* of l values.
        (x - a)^l = ((x - e) + (e - a))^l = sum_m [ C(l,m) * (e-a)^(l-m) * (x-e)^m ]
        
        Args:
            l_vec: (Ncomp,) tensor of angular momenta.
            a: (3,) or scalar center A.
            e: (Ni, Nj, 3) or (Ni, Nj) center P.
        
        Returns:
            coeffs: (Ni, Nj, Ncomp, L_max+1) tensor, padded to max(l_vec)+1.
        """
        Ni, Nj = e.shape[0:2]
        Ncomp = l_vec.shape[0]
        
        if Ncomp == 0:
            return torch.empty((Ni, Nj, 0, 0), dtype=e.dtype, device=e.device)
            
        if l_vec.max().item() > self.MAXL:
            raise NotImplementedError(f"l_vec max {l_vec.max()} > MAXL {self.MAXL}")

        ae = e - a # (Ni, Nj, 3) or (Ni, Nj)
        if ae.dim() == 2: # Handle 1D case (k-loop)
            ae = ae.unsqueeze(-1) # (Ni, Nj, 1)
        
        L_max = l_vec.max().item()
        if L_max < 0: # Handle empty shell case
             return torch.empty((Ni, Nj, Ncomp, 0), dtype=e.dtype, device=e.device)
             
        m = torch.arange(L_max + 1, device=ae.device) # (L_max+1,)
        
        # Ensure BINO is on the correct device
        if self.BINO.device != ae.device:
            self.BINO = self.BINO.to(ae.device)

        # C(l, m) -> (Ncomp, L_max+1)
        # FIX from previous bug:
        bino_coeffs = self.BINO[l_vec.long()][:, :L_max+1]
        
        # (e-a)^(l-m)
        l_vec_b = l_vec.view(Ncomp, 1) # (Ncomp, 1)
        m_b = m.view(1, L_max+1)      # (1, L_max+1)
        
        l_minus_m_raw = l_vec_b - m_b     # (Ncomp, L_max+1)
        
        # Create mask for m > l, where l_minus_m is negative
        mask = m_b > l_vec_b          # (Ncomp, L_max+1)
        
    
        l_minus_m = torch.where(mask, 0.0, l_minus_m_raw)
      
        
   
        ae_b = ae.unsqueeze(-1) # (Ni, Nj, 3 or 1, 1)
        ae_powers_raw = ae_b ** l_minus_m[None, None, :, :]
        ae_powers = torch.where(mask[None, None, :, :], 0.0, ae_powers_raw)

        # val_l (from one-hot) is implicitly handled by the binomial expansion
        # C(l,m) * (e-a)^(l-m)
        # (1, 1, Ncomp, L_max+1) * (Ni, Nj, Ncomp, L_max+1)
        coeffs = bino_coeffs[None, None, :, :] * ae_powers
        
        return coeffs # (Ni, Nj, Ncomp, L_max+1)

    def compute_1d_product_coefficients_vec(self, coeff_a, coeff_b, la, lb): # (vectorized)
        """ 
        Vectorized 1D convolution of coefficient arrays using F.conv1d.
        (d_k = sum_{i+j=k} a_i * b_j)
        """
        # coeff_a: (Ni, Nj, la+1)
        # coeff_b: (Ni, Nj, lb+1)
        Ni, Nj, Na = coeff_a.shape
        _, _, Nb = coeff_b.shape
        # la = Na - 1, lb = Nb - 1
        output_size = la + lb + 1
        
        # Reshape for conv1d: (N, Cin, Lin)
        # Batch size N = Ni * Nj
        N_batch = Ni * Nj
        if N_batch == 0:
            return torch.empty((Ni, Nj, output_size), dtype=coeff_a.dtype, device=coeff_a.device)
            
        # --- FIX ---
        # Reshape for grouped conv:
        # We treat the batch as channels in a single-item batch
        # Input shape (N=1, C_in=N_batch, L_in=Na)
        coeff_a_flat = coeff_a.view(1, N_batch, Na)
        
        # Kernel (weight) for conv1d: (Cout, Cin/groups, Lk)
        # We flip coeff_b to act as a convolution kernel (correlation -> convolution)
        # Weight shape (C_out=N_batch, C_in/groups=1, L_k=Nb)
        coeff_b_flat_rev = torch.flip(coeff_b, dims=[2]).view(N_batch, 1, Nb)
        
        # padding=lb ensures the output has size (Na + Nb - 1) = (la + lb + 1)
        conv_out = F.conv1d(
            coeff_a_flat,       # input: (1, N_batch, Na)
            coeff_b_flat_rev,   # weight: (N_batch, 1, Nb)
            padding=lb, 
            groups=N_batch
        )
        
        # Output shape will be (N=1, C_out=N_batch, L_out=output_size)
        # Reshape back to (Ni, Nj, output_size)
        return conv_out.view(Ni, Nj, output_size)

    def compute_1d_product_coefficients_batch_outer(
        self, coeff_a: torch.Tensor, coeff_b: torch.Tensor, 
        la_vec: torch.Tensor, lb_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized 1D convolution for an outer product of coefficient arrays.
        
        Args:
            coeff_a: (Ni, Nj, naoi, Na_max)
            coeff_b: (Ni, Nj, naoj, Nb_max)
            la_vec: (naoi,)
            lb_vec: (naoj,)
            
        Returns:
            (Ni, Nj, naoi, naoj, N_out_max)
        """
        Ni, Nj, naoi, Na_max = coeff_a.shape
        _, _, naoj, Nb_max = coeff_b.shape
        
        # Output size is padded to the maximum possible
        max_la = la_vec.max().item() if naoi > 0 else -1
        max_lb = lb_vec.max().item() if naoj > 0 else -1
        N_out_max = max_la + max_lb + 1
        
        if Ni*Nj*naoi*naoj == 0:
             return torch.empty((Ni, Nj, naoi, naoj, N_out_max), dtype=coeff_a.dtype, device=coeff_a.device)

        # Expand and reshape for batched outer-product convolution
        # (Ni, Nj, naoi, 1, Na_max) -> (Ni, Nj, naoi, naoj, Na_max)
        a_b = coeff_a.unsqueeze(3).expand(-1, -1, -1, naoj, -1)
        
        # (Ni, Nj, 1, naoj, Nb_max) -> (Ni, Nj, naoi, naoj, Nb_max)
        b_b = coeff_b.unsqueeze(2).expand(-1, -1, naoi, -1, -1)
        
        # Flatten into a single batch dimension for conv1d
        N_batch = Ni * Nj * naoi * naoj
        # (1, N_batch, Na_max)
        a_flat = a_b.reshape(1, N_batch, Na_max)
        
        # (N_batch, 1, Nb_max)
        b_flat_rev = b_b.reshape(N_batch, 1, Nb_max).flip(dims=[-1])
        
        # padding=Nb_max-1 ensures the output has size (Na_max + Nb_max - 1)
        conv_out = F.conv1d(
            a_flat,
            b_flat_rev,
            padding = Nb_max - 1,
            groups = N_batch
        ) # (1, N_batch, Na_max + Nb_max - 1)
        
        # Truncate to the required N_out_max and reshape
        # (Ni, Nj, naoi, naoj, N_out_max)
        return conv_out.view(Ni, Nj, naoi, naoj, -1)[..., 0:N_out_max]
    
    # --- Internal helper methods ---
    def _build_s1d_factors_vec(self, max_l: int, gama_mat: torch.Tensor) -> torch.Tensor:
        """ Vectorized: Creates array of 1D integral factors (olapp results). """
        # gama_mat: (Ni, Nj)
        s1d_factors = []
        for l in range(max_l + 1):
            factor = self.calc_1d_overlap_constants_vec(l, gama_mat)
            s1d_factors.append(factor)
        return torch.stack(s1d_factors, dim=-1) # (Ni, Nj, max_l+1)

    def _assemble_3d_multipole_factors_vec(
        self,
        ri: torch.Tensor, rj: torch.Tensor, rp: torch.Tensor, rc: torch.Tensor,
        li_mat: torch.Tensor, lj_mat: torch.Tensor,  # (naoi, 3), (naoj, 3)
        s1d_factors: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Vectorized Python (PyTorch) version of multipole_3d subroutine.
        Batched over (naoi, naoj) components, loops over k=3 (x,y,z).
        """
        # ri, rj, rc: (3,)
        # rp: (Ni, Nj, 3)
        # s1d_factors: (Ni, Nj, Lmax+3)
        Ni, Nj, _ = rp.shape
        naoi = li_mat.shape[0]
        naoj = lj_mat.shape[0]
        tensor_opts = {'dtype': ri.dtype, 'device': ri.device}
        
        if naoi == 0 or naoj == 0:
            return torch.empty((Ni, Nj, naoi, naoj, 10), **tensor_opts)

        # (Ni, Nj, naoi, naoj, 3)
        val_S_k = torch.zeros((Ni, Nj, naoi, naoj, 3), **tensor_opts)
        val_M_k = torch.zeros((Ni, Nj, naoi, naoj, 3), **tensor_opts)
        val_Q_k = torch.zeros((Ni, Nj, naoi, naoj, 3), **tensor_opts)
        
        # --- Loop over k=x,y,z (small, fixed loop) ---
        for k in range(3):
            l_i_k_vec = li_mat[:, k]  # (naoi,)
            l_j_k_vec = lj_mat[:, k]  # (naoj,)
            
            max_l_i_k = l_i_k_vec.max().item()
            max_l_j_k = l_j_k_vec.max().item()
            pad_i = max_l_i_k + 1
            pad_j = max_l_j_k + 1
            
            # 2. Define centers
            center_i_k = ri[k] # scalar
            center_j_k = rj[k] # scalar
            center_p_k = rp[..., k] # (Ni, Nj)
            center_c_k = rc[k] # scalar
            rpc_k = center_p_k - center_c_k # (Ni, Nj)

            # 3. Perform horizontal shift (batched over naoi and naoj)
            # vi_shifted -> (Ni, Nj, naoi, pad_i)
            vi_shifted = self.calculate_center_shift_coefficients_batch_l(
                l_i_k_vec, center_i_k, center_p_k)
                
            # vj_shifted -> (Ni, Nj, naoj, pad_j)
            vj_shifted = self.calculate_center_shift_coefficients_batch_l(
                l_j_k_vec, center_j_k, center_p_k)
            
            # 4. Convolve coefficients (batched outer product)
            # vv -> (Ni, Nj, naoi, naoj, pad_out)
            l_total_max = max_l_i_k + max_l_j_k
            pad_out = l_total_max + 1
            
            vv = self.compute_1d_product_coefficients_batch_outer(
                vi_shifted, vj_shifted, l_i_k_vec, l_j_k_vec)
            
            # 5. Calculate 1D integral factors (dot products)
            # (naoi, 1) + (1, naoj) -> (naoi, naoj)
            l_total_mat = l_i_k_vec[:, None] + l_j_k_vec[None, :] 
            
            # We must use a common slice size for all components
            vv_subset_raw = vv[..., 0 : pad_out]
            s1d_L = s1d_factors[..., 0 : pad_out]
            s1d_Lplus1 = s1d_factors[..., 1 : pad_out + 1]
            s1d_Lplus2 = s1d_factors[..., 2 : pad_out + 2]
            
            # Create mask for terms l > l_total (from padding)
            # m (0...pad_out-1) > l_total (naoi, naoj)
            m_vec = torch.arange(pad_out, device=ri.device)[None, None, :] # (1, 1, pad_out)
            l_total_b = l_total_mat.unsqueeze(-1) # (naoi, naoj, 1)
            mask = m_vec > l_total_b # (naoi, naoj, pad_out)
            mask_b = mask[None, None, :, :, :] # (1, 1, naoi, naoj, pad_out)
            
            # --- FIX 3: Replace in-place operation ---
            # Original: vv_subset.masked_fill_(mask_b, 0.0)
            # Mask out invalid terms from padded convolution
            vv_subset = torch.where(mask_b, 0.0, vv_subset_raw)
            # --- End Fix 3 ---

            # Broadcast for dot product
            # rpc_b -> (Ni, Nj, 1, 1, 1)
            rpc_b = rpc_k[..., None, None, None]
            # s1d_b -> (Ni, Nj, 1, 1, pad_out)
            s1d_L_b = s1d_L[..., None, None, :]
            s1d_Lplus1_b = s1d_Lplus1[..., None, None, :]
            s1d_Lplus2_b = s1d_Lplus2[..., None, None, :]
            
            # Batched dot product: sum over last dimension (pad_out)
            val_S_k[..., k] = torch.sum(vv_subset * s1d_L_b, dim=-1)
            
            dipole_s1d_vec = s1d_Lplus1_b + rpc_b * s1d_L_b
            val_M_k[..., k] = torch.sum(vv_subset * dipole_s1d_vec, dim=-1)
            
            quad_s1d_vec = s1d_Lplus2_b + 2.0 * rpc_b * s1d_Lplus1_b + (rpc_b**2) * s1d_L_b
            val_Q_k[..., k] = torch.sum(vv_subset * quad_s1d_vec, dim=-1)
            
        # 6. Assemble 3D integral factors
        # val_S_k (Ni, Nj, naoi, naoj, 3)
        S_x, S_y, S_z = val_S_k[..., 0], val_S_k[..., 1], val_S_k[..., 2]
        M_x, M_y, M_z = val_M_k[..., 0], val_M_k[..., 1], val_M_k[..., 2]
        Q_x, Q_y, Q_z = val_Q_k[..., 0], val_Q_k[..., 1], val_Q_k[..., 2]

        # (Ni, Nj, naoi, naoj, 10)
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
        # Basis set information (passed from caller)
        basis: dict, alp_tensor: torch.Tensor, cont_tensor: torch.Tensor,
        prim_indices_i: list, prim_indices_j: list,
        itt_device: torch.Tensor, lxyz_type_device: torch.Tensor,
        device: torch.device = None
    ) -> torch.Tensor:
        """
         Computes the overlap matrix block between two *Shells* (ish, jsh)
        using vectorized primitive AND component loops.
        """
        if device is None: device = ri.device
        tensor_opts = {'dtype': ri.dtype, 'device': device}
        
        rij = ri - rj
        rij2 = torch.dot(rij, rij)
        max_r2 = 2000.0
        if rij2 > max_r2: 
            return torch.zeros((naoj_cart, naoi_cart), **tensor_opts)

        # Cartesian component start indices
        iptyp = itt_device[ishtyp].item()
        jptyp = itt_device[jshtyp].item()

        # --- Get all primitive data at once ---
        Ni = len(prim_indices_i)
        Nj = len(prim_indices_j)
        if Ni == 0 or Nj == 0:
            return torch.zeros((naoj_cart, naoi_cart), **tensor_opts)
            
        alp_i_vec = alp_tensor[prim_indices_i] # (Ni,)
        cont_i_vec = cont_tensor[prim_indices_i]
        alp_j_vec = alp_tensor[prim_indices_j] # (Nj,)
        cont_j_vec = cont_tensor[prim_indices_j]
        
        # --- Broadcast for (Ni, Nj) matrix calculations ---
        alp_i_b = alp_i_vec.unsqueeze(1) # (Ni, 1)
        alp_j_b = alp_j_vec.unsqueeze(0) # (1, Nj)
        
        # --- Calculate all primitive pair properties ---
        # gamma_mat, kab_mat: (Ni, Nj)
        gamma_mat, kab_mat = self.calc_product_exponent_and_overlap_vec(alp_i_b, ri, alp_j_b, rj)
        
        # rp_mat: (Ni, Nj, 3)
        rp_mat = self.calc_gau_product_center_vec(
            alp_i_b.unsqueeze(2), ri.view(1,1,3), # (Ni, 1, 1), (1, 1, 3)
            alp_j_b.unsqueeze(2), rj.view(1,1,3), # (1, Nj, 1), (1, 1, 3)
            gamma_mat.unsqueeze(2)                # (Ni, Nj, 1)
        )
        
        # --- Pre-calculate 1D factors ---
        max_l_k_prim = ishtyp + jshtyp
        max_l_needed_prim = max_l_k_prim + 2
        # t_factors_mat: (Ni, Nj, max_l+3)
        t_factors_mat = self._build_s1d_factors_vec(max_l_needed_prim, gamma_mat)

        # ---  Batch over components (mli, mlj) ---
        
        # 1. Get component lmn vectors
        # (naoi_cart, 3)
        li_mat = lxyz_type_device[iptyp : iptyp + naoi_cart]
        # (naoj_cart, 3)
        lj_mat = lxyz_type_device[jptyp : jptyp + naoj_cart]
        
        # 2. Calculate normalization matrices
        # n_i_mat: (Ni, naoi_cart)
        n_i_mat = self._primitive_norm(alp_i_vec, li_mat)
        # n_j_mat: (Nj, naoj_cart)
        n_j_mat = self._primitive_norm(alp_j_vec, lj_mat)
        
        # 3. Combine with contraction coefficients
        # c_n_i: (Ni, naoi_cart)
        c_n_i = cont_i_vec[:, None] * n_i_mat
        # c_n_j: (Nj, naoj_cart)
        c_n_j = cont_j_vec[:, None] * n_j_mat
        
        # 4. Create full (Ni, Nj, naoi, naoj) coefficient matrix
        # (Ni, 1, naoi, 1) * (1, Nj, 1, naoj) -> (Ni, Nj, naoi, naoj)
        cc_full_mat = c_n_i[:, None, :, None] * c_n_j[None, :, None, :]
        
        # 5. Calculate HRR factors (batched over naoi, naoj)
        # saw_factors_mat: (Ni, Nj, naoi, naoj, 10)
        saw_factors_mat = self._assemble_3d_multipole_factors_vec(
            ri, rj, rp_mat, point,
            li_mat, lj_mat,
            t_factors_mat
        )
        
        # 6. Unnormalized primitive overlap matrix (S component only)
        # (Ni, Nj, 1, 1) * (Ni, Nj, naoi, naoj)
        s_prim_full = kab_mat[..., None, None] * saw_factors_mat[..., 0] 
        
        # 7. Contract by summing over all primitives (dim 0 and 1)
        # (Ni, Nj, naoi, naoj) * (Ni, Nj, naoi, naoj) -> sum(0,1) -> (naoi, naoj)
        sint_block_cart_T = torch.sum(s_prim_full * cc_full_mat, dim=(0, 1))
        
        # Transpose to (naoj, naoi) as expected by caller
        sint_block_cart = sint_block_cart_T.T
        
        return sint_block_cart # Return (naoj_cart, naoi_cart)


    def calculate_overlap_matrix_full(self, xyz: torch.Tensor, intcut) -> torch.Tensor:
        """ Calculates the full molecular overlap matrix S (nao, nao). """
        if xyz.shape[0] == 3 and xyz.shape[1] != 3: xyz = xyz.T
        n_atoms = xyz.shape[0]
        current_device = xyz.device

        # --- Access basis info ---
        n_ao = self.basis['number_of_ao']
        n_shells = self.basis['number_of_shells']
        tensor_opts = {'dtype': xyz.dtype, 'device': current_device}
        sint = torch.zeros((n_ao, n_ao), **tensor_opts)

        LLAO = torch.tensor(self.LLAO, dtype=torch.int64, device=current_device)
        point = torch.zeros(3, **tensor_opts)

        # --- Pre-fetch basis data ---
        shell_amqn_list = torch.tensor(self.basis['shell_amqn_list'], dtype=torch.int64, device=current_device)
        shell_atom_list = torch.tensor(self.basis['shell_atom_list'], dtype=torch.int64, device=current_device)
        shell_ao_map = torch.tensor(self.basis['shell_ao_map'], dtype=torch.int64, device=current_device)
        alp = self.basis.get('primitive_alpha_list', [])
        alp_tensor = torch.tensor(alp, **tensor_opts)
        cont = self.basis.get('primitive_coeff_list', [])
        cont_tensor = torch.tensor(cont, **tensor_opts)
        
        #  Pre-fetch device-specific lookups
        itt_device = self.ITT.to(current_device)
        lxyz_type_device = self.LXYZ_TYPE_TENSOR.to(current_device)
        self.F2 = self.F2.to(current_device) # Ensure F2 is on device
        # BINO tensor will be moved to device on-demand by calculate_center_shift...

        #  Pre-calculate primitive indices for all shells
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
            shell_prim_indices.append(indices) # Use list of lists

        # --- Shell Pair Loops ---
        for ish in range(n_shells):
            iat_idx = shell_atom_list[ish]
            ri = xyz[iat_idx]
            ishtyp = shell_amqn_list[ish].item()
            naoi_cart = LLAO[ishtyp].item()
            iao_start, naoi_spher = shell_ao_map[ish]
            iao_start, naoi_spher = iao_start.item(), naoi_spher.item()
            iao_slice = slice(iao_start, iao_start + naoi_spher)
            
            #  Get pre-calculated primitive indices
            prim_indices_i = shell_prim_indices[ish]

            for jsh in range(ish + 1):
                jat_idx = shell_atom_list[jsh]
                rj = xyz[jat_idx]
                jshtyp = shell_amqn_list[jsh].item()
                naoj_cart = LLAO[jshtyp].item()
                jao_start, naoj_spher = shell_ao_map[jsh]
                jao_start, naoj_spher = jao_start.item(), naoj_spher.item()
                jao_slice = slice(jao_start, jao_start + naoj_spher)
                
                #  Get pre-calculated primitive indices
                prim_indices_j = shell_prim_indices[jsh]
                
                # --- 1. Calculate Cartesian Shell Block ---
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
                # ss_cart_shell shape: (naoj_cart, naoi_cart)

                # --- 2. Transform Shell Block to Spherical ---
                ss_spher_padded = self.transform_d_cartesian_to_spherical(ss_cart_shell, ishtyp, jshtyp)
                
                # --- 3. Extract the correct spherical block ---
                i_slice_start = 1 if ishtyp == 2 else 0
                j_slice_start = 1 if jshtyp == 2 else 0
                i_slice_end = i_slice_start + naoi_spher
                j_slice_end = j_slice_start + naoj_spher
                
                ss_spher_shell = ss_spher_padded[j_slice_start:j_slice_end, i_slice_start:i_slice_end]

                # --- 4. Accumulate into global sint matrix ---
                if ish == jsh: # Diagonal shell block
                    sint[jao_slice, iao_slice] = sint[jao_slice, iao_slice] + ss_spher_shell
                else: # Off-diagonal shell block
                    sint[jao_slice, iao_slice] = sint[jao_slice, iao_slice] + ss_spher_shell
                    sint[iao_slice, jao_slice] = sint[iao_slice, jao_slice] + ss_spher_shell.T # Assign transpose S_ji

        sint = self.normalize_overlap_matrix(sint)
        return sint

    def normalize_overlap_matrix(self, sint: torch.Tensor) -> torch.Tensor:
        """ Normalizes the overlap matrix S to have ones on the diagonal. """
        diag = torch.diagonal(sint)
        # Add epsilon to diag to prevent sqrt(0) or 1/0
        diag_safe = diag + 1e-20
        inv_sqrt_diag = 1.0 / torch.sqrt(diag_safe)
        # Zero out entries where the original diagonal was near zero
        inv_sqrt_diag = torch.where(diag > 1e-12, inv_sqrt_diag, torch.zeros_like(diag))
        
        D_inv_sqrt = torch.diag(inv_sqrt_diag)
        sint_normalized = torch.matmul(D_inv_sqrt, torch.matmul(sint, D_inv_sqrt))
        return sint_normalized


    def calculation(self, xyz: torch.Tensor, intcut=40):
        """ Main calculation entry point, returns S, D(zeros), Q(zeros). """
        sint = self.calculate_overlap_matrix_full(xyz, intcut)
        n_ao = sint.shape[0]
        dpint = torch.zeros((3, n_ao, n_ao), dtype=sint.dtype, device=sint.device)
        qpint = torch.zeros((6, n_ao, n_ao), dtype=sint.dtype, device=sint.device)
        
        self.sint = sint
        return sint, dpint, qpint

    def get_overlap_integral_matrix(self) -> torch.Tensor:
        """ Returns the last computed overlap integral matrix S. """
        if not hasattr(self, 'sint'):
            print("Warning: Overlap integral matrix S has not been computed yet. Returning None.")
            return None
        return self.sint

    def overlap_int(self, xyz: np.ndarray) -> torch.Tensor:
        """ Calculates overlap matrix from NumPy coordinate array. """
        xyz_torch = torch.tensor(xyz, dtype=torch.float64, requires_grad=False)
        sint, _, _ = self.calculation(xyz_torch)
        return sint # Return tensor

    def overlap_int_torch(self, xyz: torch.Tensor) -> torch.Tensor:
        """ Calculates overlap matrix from Torch Tensor coordinates. """
        sint = self._get_sint_only(xyz)
        return sint # Return tensor
    
    def _get_sint_only(self, xyz: torch.Tensor) -> torch.Tensor:
        """ Helper function for jacrev/hessian to get S. """
        sint, _, _ = self.calculation(xyz)
        return sint

    def d_overlap_int_dxyz(self, xyz: np.ndarray) -> torch.Tensor:
        """ Calculates the Jacobian of the overlap matrix w.r.t. nuclear coordinates. """
        xyz_torch = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        d_sint_dxyz = torch.func.jacrev(self._get_sint_only, argnums=0)(xyz_torch)
        return d_sint_dxyz

    def d2_overlap_int_dxyz2(self, xyz: np.ndarray) -> torch.Tensor:
        """ Calculates the Hessian of the overlap matrix w.r.t. nuclear coordinates. """
        xyz_torch = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        d2_sint_sum_dxyz2 = torch.func.hessian(self._get_sint_only, argnums=0)(xyz_torch)
        return d2_sint_sum_dxyz2