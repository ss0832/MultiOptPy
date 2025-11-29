import numpy as np
import torch
import math

from multioptpy.SQM.sqm2.sqm2_overlapint import OverlapCalculator
from multioptpy.SQM.sqm2.sqm2_basis import BasisSet
from multioptpy.SQM.sqm2.calc_tools import factorial2, dfactorial


class EHTCalculator:
    def __init__(self, element_list, charge, spin, param, wf_instance):
        # element_list: list[int]
        # param: SQM2Parameters object containing EHT parameters
        # wf_instance: BasisSet object (or similar containing 'basis' dict and atom type info)
        
        # --- Device and Type ---
        self.device = "cpu" # Or get from wf_instance/param if available
        self.dtype = torch.float64

        # --- Core Properties ---
        self.element_list = element_list
        self.params = param  
        self.wf = wf_instance
        self.basis = wf_instance.basis  # Basis dictionary
        self.charge = charge
        self.charge_t = torch.tensor(self.charge, dtype=self.dtype, device=self.device) # Pre-tensorize
        self.spin = spin
        self.n_atoms = len(self.element_list)
        self.n_ao = self.basis["number_of_ao"]
        
        # --- Constants (non-tensor) ---
        self.PI = math.pi
        self.SQRTPI = math.sqrt(self.PI)
        self.LMAX_INTEGRAL = 8 # Max L for 1D integrals (l+l'+2)

        # --- Initialize Overlap Calculator ---
        self.overlap_calc = OverlapCalculator(self.element_list, self.params, self.wf)

        # --- Pre-load Basis Info as Tensors ---
        self.amqn_list = torch.tensor(self.basis["shell_amqn_list"], dtype=torch.int64, device=self.device)
        self.atom_shells_map = torch.tensor(self.basis["atom_shells_map"], dtype=torch.int64, device=self.device)
        self.shell_ao_map = torch.tensor(self.basis["shell_ao_map"], dtype=torch.int64, device=self.device)
        self.ang_shells_list = self.wf.ang_shells_list # Keep as list of lists for flattening
        

        # Build python lists first
        paulingEN_list = []
        kQatom_list = []
        kQShell_list = [] 
        shellpoly_list = [] 
        atomicRad_list = [] 
        nshells_list = []
        self_energy_list = []
        kCN_list = [] 
        slaterexponent_list = []
        referenceOcc = []
        
        for i in range(len(self.element_list)):
            atn = self.element_list[i]
            paulingEN_list.append(param.paulingEN[atn])
            kQatom_list.append(param.kQAtom[atn])
            kQShell_list.append(param.kQShell[atn])
            shellpoly_list.append(param.shellPoly[atn])          
            atomicRad_list.append(param.atomicRad[atn])
            nshells_list.append(param.nShell[atn])
            self_energy_list.append(param.selfEnergy[atn])
            kCN_list.append(param.kCN[atn])
            slaterexponent_list.append(param.slaterExponent[atn])
            referenceOcc.append(param.referenceOcc[atn])
        
        # ---  Pre-compute total valence electrons ---
        self.total_valence_e = sum(sum(occ) for occ in referenceOcc)
        self.total_valence_e = torch.tensor(self.total_valence_e, dtype=self.dtype, device=self.device)
        
        # ---  Pad per-atom lists to (n_atoms, max_nshell) arrays ---
        max_nshell = max(ns for ns in nshells_list)

        def pad_list_of_lists(lst, max_len, fill=0.0):
            # Helper to pad lists of lists/arrays to a uniform 2D NumPy array
            padded = np.full((self.n_atoms, max_len), fill, dtype=np.float64)
            for i, sublst in enumerate(lst):
                n_to_copy = min(len(sublst), max_len)
                padded[i, :n_to_copy] = sublst[:n_to_copy]
               
            return padded

        kQShell_padded = pad_list_of_lists(kQShell_list, max_nshell)
        shellpoly_padded = pad_list_of_lists(shellpoly_list, max_nshell)
        self_energy_padded = pad_list_of_lists(self_energy_list, max_nshell)
        kCN_padded = pad_list_of_lists(kCN_list, max_nshell)
        slaterexponent_padded = pad_list_of_lists(slaterexponent_list, max_nshell)

        # ---  Convert all to tensors ---
        # 1D tensors (per-atom)
        self.paulingEN_list = torch.tensor(paulingEN_list, dtype=self.dtype, device=self.device)
        self.kQatom_list = torch.tensor(kQatom_list, dtype=self.dtype, device=self.device)
        self.atomicRad_list = torch.tensor(atomicRad_list, dtype=self.dtype, device=self.device)
        self.nshells_list = torch.tensor(nshells_list, dtype=torch.int64, device=self.device)
        
        # 2D tensors (per-atom, per-shell)
        self.kQShell_tensor = torch.tensor(kQShell_padded, dtype=self.dtype, device=self.device)
        self.shellpoly_tensor = torch.tensor(shellpoly_padded, dtype=self.dtype, device=self.device)
        self.self_energy_tensor = torch.tensor(self_energy_padded, dtype=self.dtype, device=self.device)
        self.kCN_tensor = torch.tensor(kCN_padded, dtype=self.dtype, device=self.device)
        self.slaterexponent_tensor = torch.tensor(slaterexponent_padded, dtype=self.dtype, device=self.device)

        # --- EHT K-Factors (scalar) ---
        self.k_ss_eht = param.k_ss_eht
        self.k_pp_eht = param.k_pp_eht
        self.k_dd_eht = param.k_dd_eht
        self.k_sp_eht = param.k_sp_eht
        self.k_sd_eht = param.k_sd_eht
        self.k_pd_eht = param.k_pd_eht
        self.k_hh_2s22 = param.k_hh_2s2s 
        self.k_ss_en_eht = param.k_ss_en_eht
        self.k_pp_en_eht = param.k_pp_en_eht
        self.k_dd_en_eht = param.k_dd_en_eht
        self.k_sp_en_eht = param.k_sp_en_eht
        self.k_sd_en_eht = param.k_sd_en_eht
        self.k_pd_en_eht = param.k_pd_en_eht # <-- Fixed typo
        self.b_en_eht = param.b_en_eht 
        self.k_MM_pair = param.k_MM_pair 
        self.k_g11_pair = param.k_g11_pair 
        
        # Convert bool lists to tensors
        self.is_tm_tensor = torch.tensor(wf_instance.is_tm_list, dtype=torch.bool, device=self.device)
        self.is_g11_tensor = torch.tensor(wf_instance.is_g11_element_list, dtype=torch.bool, device=self.device)

        # ---  Pre-compute static shell properties (Vectorized) ---
        self.n_shell = self.shell_ao_map.shape[0]

        # Map: shell_index -> atom_index
        # e.g., [0, 0, 0, 1, 1, 2, 2, 2, 2, ...]
        self.shell_atom_map = torch.repeat_interleave(
            torch.arange(self.n_atoms, device=self.device), 
            self.nshells_list
        )
        
        # Map: shell_index -> local_shell_index (within its atom)
        # e.g., [0, 1, 2, 0, 1, 0, 1, 2, 3, ...]
        self.shell_local_idx_map = torch.cat(
            [torch.arange(n.item(), dtype=torch.int64, device=self.device) for n in self.nshells_list]
        )
        
        # Map: shell_index -> shell_type (0=s, 1=p, 2=d)
        # e.g., [0, 1, 0, 1, 2, 0, 1, 2, 3, ...]
        ang_flat = [item for sublist in self.ang_shells_list for item in sublist]
        self.shell_type_map = torch.tensor(ang_flat, dtype=torch.int64, device=self.device)

        # ---  Gather shell properties using maps (NO LOOPS) ---
        
        # (n_shell,) tensors gathered from (n_atoms, max_nshell) tensors
        self.shell_poly_const_map = self.shellpoly_tensor[self.shell_atom_map, self.shell_local_idx_map]
        self.shell_slater_exp_map = self.slaterexponent_tensor[self.shell_atom_map, self.shell_local_idx_map]

        # (n_shell,) tensors gathered from (n_atoms,) tensors
        self.shell_rad_map = self.atomicRad_list[self.shell_atom_map]
        self.shell_en_map = self.paulingEN_list[self.shell_atom_map]
        self.shell_is_tm_map = self.is_tm_tensor[self.shell_atom_map]
        self.shell_is_g11_map = self.is_g11_tensor[self.shell_atom_map]
        
        # ---  Pre-compute pair indices ---
        i_indices_upper, j_indices_upper = torch.triu_indices(self.n_shell, self.n_shell, 1, device=self.device)
        
        iat_pairs = self.shell_atom_map[i_indices_upper]
        jat_pairs = self.shell_atom_map[j_indices_upper]
        
        off_atom_mask = (iat_pairs != jat_pairs)
        self.i_off_atom_pairs = i_indices_upper[off_atom_mask]
        self.j_off_atom_pairs = j_indices_upper[off_atom_mask]
        
        on_atom_mask = ~off_atom_mask
        self.i_on_atom_pairs = i_indices_upper[on_atom_mask]
        self.j_on_atom_pairs = j_indices_upper[on_atom_mask]
        
        # ---  Pre-load AO slice info ---
        self.shell_ao_starts = self.shell_ao_map[:, 0]
        self.shell_ao_nao = self.shell_ao_map[:, 1]
        
        # ---  Create AO -> Shell map (Vectorized) ---
        # Map: ao_index -> shell_index
        # e.g., [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, ...]
        self.ao_shell_map = torch.repeat_interleave(
            torch.arange(self.n_shell, device=self.device),
            self.shell_ao_nao
        )
        self.holistic_k_factor = 1.4 #1.0#  # Overall scaling factor
        return
        
    def _get_eht_k_factor(self, ishtyp, jshtyp, iat, jat, delta_en):
        """ Calculates the EHT K-factor based on shell and atom types. (Scalar version) """

        # Determine base K based on shell types
        if ishtyp == 0 and jshtyp == 0:     # s-s
            k_base = self.k_ss_eht
            k_en = self.k_ss_en_eht
        elif ishtyp == 1 and jshtyp == 1:   # p-p
            k_base = self.k_pp_eht
            k_en = self.k_pp_en_eht
        elif ishtyp == 2 and jshtyp == 2:   # d-d
            k_base = self.k_dd_eht
            k_en = self.k_dd_en_eht
        elif (ishtyp == 0 and jshtyp == 1) or (ishtyp == 1 and jshtyp == 0): # s-p
            k_base = self.k_sp_eht
            k_en = self.k_sp_en_eht
        elif (ishtyp == 0 and jshtyp == 2) or (ishtyp == 2 and jshtyp == 0): # s-d
            k_base = self.k_sd_eht
            k_en = self.k_sd_en_eht
        elif (ishtyp == 1 and jshtyp == 2) or (ishtyp == 2 and jshtyp == 1): # p-d
            k_base = self.k_pd_eht
            k_en = self.k_pd_en_eht
        else: # f-shells or higher
            k_base = 1.0 
            k_en = 0.0

        en_factor = 1.0 + k_en * delta_en ** 2.0 + k_en * self.b_en_eht * delta_en ** 4.0
        
        # Atom type scaling (TM/G11)
        atom_factor = 1.0
        # Use pre-loaded boolean tensors
        is_tm_i = self.is_tm_tensor[iat]
        is_tm_j = self.is_tm_tensor[jat]
        is_g11_i = self.is_g11_tensor[iat]
        is_g11_j = self.is_g11_tensor[jat]

        if is_tm_i and is_tm_j:
            atom_factor = self.k_MM_pair
        elif is_g11_i and is_g11_j:
            atom_factor = self.k_g11_pair

        

        return k_base * en_factor * atom_factor
    
    def _get_eht_k_factor_vec(self, ishtyp, jshtyp, is_tm_i, is_tm_j, is_g11_i, is_g11_j, delta_en):
        """ (Vectorized) Calculates EHT K-factors for pairs. """
        # ishtyp, jshtyp, ... are all 1D tensors of shape (n_pairs,)
        
        k_base = torch.full_like(delta_en, 1.0)
        k_en = torch.zeros_like(delta_en)

        # s-s
        mask = (ishtyp == 0) & (jshtyp == 0)
        k_base[mask] = self.k_ss_eht
        k_en[mask] = self.k_ss_en_eht
        # p-p
        mask = (ishtyp == 1) & (jshtyp == 1)
        k_base[mask] = self.k_pp_eht
        k_en[mask] = self.k_pp_en_eht
        # d-d
        mask = (ishtyp == 2) & (jshtyp == 2)
        k_base[mask] = self.k_dd_eht
        k_en[mask] = self.k_dd_en_eht
        # s-p
        mask = ((ishtyp == 0) & (jshtyp == 1)) | ((ishtyp == 1) & (jshtyp == 0))
        k_base[mask] = self.k_sp_eht
        k_en[mask] = self.k_sp_en_eht
        # s-d
        mask = ((ishtyp == 0) & (jshtyp == 2)) | ((ishtyp == 2) & (jshtyp == 0))
        k_base[mask] = self.k_sd_eht
        k_en[mask] = self.k_sd_en_eht
        # p-d
        mask = ((ishtyp == 1) & (jshtyp == 2)) | ((ishtyp == 2) & (jshtyp == 1))
        k_base[mask] = self.k_pd_eht
        k_en[mask] = self.k_pd_en_eht

        en_factor = 1.0 + k_en * delta_en ** 2.0 + k_en * self.b_en_eht * delta_en ** 4.0
        
        atom_factor = torch.ones_like(delta_en)
        atom_factor[is_tm_i & is_tm_j] = self.k_MM_pair
        atom_factor[is_g11_i & is_g11_j] = self.k_g11_pair

        return k_base * en_factor * atom_factor * self.holistic_k_factor
    
    def _get_self_energy(self, q, cn):
        # q: torch.Tensor (N_atoms,)
        # cn: torch.Tensor (N_atoms,)
        """ 
        Calculates the self-energy matrix (N_atoms, max_n_shell). 
        (Vectorized implementation using padded tensors)
        """
        # Reshape (N_atoms,) to (N_atoms, 1) for broadcasting
        q_col = q.unsqueeze(-1)
        cn_col = cn.unsqueeze(-1)
        kqatom_col = self.kQatom_list.unsqueeze(-1)
        
        # All tensors are (N_atoms, max_nshell) or broadcast to it
        cn_corr = -self.kCN_tensor * cn_col 
        q_corr = -self.kQShell_tensor * q_col
        q_corr_2 = -kqatom_col * (q_col ** 2)
        
        # (N_atoms, max_nshell) + (N_atoms, max_nshell) + ...
        self_energy_matrix = self.self_energy_tensor + cn_corr + q_corr + q_corr_2
        
        return self_energy_matrix

    def _get_shellpoly_corr(self, iat, jat, ish_local, jsh_local, vec_i, vec_j, rad_ij):
        """ (Original scalar function, kept for reference/debugging if needed) """
        a = 0.5
        r_ij_vec = vec_i - vec_j
        # Add epsilon for numerical stability in norm
        r_ij_norm = torch.sqrt(torch.dot(r_ij_vec, r_ij_vec) + 1e-20) 
        
        atomic_rad_ij = rad_ij
        ratio = r_ij_norm / atomic_rad_ij
        
        #  Accessing from padded tensor requires atom and local shell index
        shellpoly_const_i = self.shellpoly_tensor[iat, ish_local]
        shellpoly_const_j = self.shellpoly_tensor[jat, jsh_local]

        shellpoly_corr_i = 1.0 + (0.01 * shellpoly_const_i) * ratio ** a
        shellpoly_corr_j = 1.0 + (0.01 * shellpoly_const_j) * ratio ** a

        return shellpoly_corr_i * shellpoly_corr_j

    def _get_shellpoly_corr_vec(self, vec_i_pairs, vec_j_pairs, rad_ij_pairs, poly_i_pairs, poly_j_pairs):
        """ (Vectorized) Calculates shellpoly correction for pairs. """
        # vec_i_pairs, vec_j_pairs are (n_pairs, 3)
        # rad_ij_pairs, poly_i_pairs, poly_j_pairs are (n_pairs,)
        a = 0.5
        r_ij_vec = vec_i_pairs - vec_j_pairs # (n_pairs, 3)
        # Add epsilon for numerical stability
        r_ij_norm = torch.linalg.norm(r_ij_vec, dim=1) + 1e-20 # (n_pairs,)
        
        ratio = r_ij_norm / rad_ij_pairs
        
        shellpoly_corr_i = 1.0 + (0.01 * poly_i_pairs) * ratio ** a
        shellpoly_corr_j = 1.0 + (0.01 * poly_j_pairs) * ratio ** a

        return shellpoly_corr_i * shellpoly_corr_j

    def get_hamiltonian(self, xyz, q, cn, sint):
        
        # 1. Get Self Energy
        # self_energy_matrix (N_atoms, max_n_shell)
        self_energy_matrix = self._get_self_energy(q, cn) 
        
        # Create a flat (n_shell,) tensor of self-energies
        # hii_all[k] = self_energy_matrix[atom_of_shell_k, local_idx_of_shell_k]
        hii_all = self_energy_matrix[self.shell_atom_map, self.shell_local_idx_map]
        
        # --- 2. DIAGONAL ELEMENTS (Vectorized) ---
        # Map hii from shells (n_shell,) to aos (n_ao,)
        hii_ao = hii_all[self.ao_shell_map] # (n_ao,)
        # Assign to diagonal of H0
        H0 = torch.diag_embed(hii_ao)
        
        # --- 3. OFF-DIAGONAL BLOCKS (On-Atom and Off-Atom) ---
        
        # Initialize (n_shell, n_shell) matrix for H_av values
        Hav_shell = torch.zeros((self.n_shell, self.n_shell), dtype=self.dtype, device=self.device)

        # --- 3a. On-Atom Pairs (Vectorized) ---
        i_pairs_on = self.i_on_atom_pairs
        j_pairs_on = self.j_on_atom_pairs
        
        if i_pairs_on.numel() > 0:
            # Gather parameters
            hii_pairs_on = hii_all[i_pairs_on]
            hjj_pairs_on = hii_all[j_pairs_on]
            ishtyp_pairs_on = self.shell_type_map[i_pairs_on]
            jshtyp_pairs_on = self.shell_type_map[j_pairs_on]
            slater_i_pairs_on = self.shell_slater_exp_map[i_pairs_on]
            slater_j_pairs_on = self.shell_slater_exp_map[j_pairs_on]
            
            is_tm_i_pairs_on = self.shell_is_tm_map[i_pairs_on]
            is_g11_i_pairs_on = self.shell_is_g11_map[i_pairs_on]
            delta_en_pairs_on = torch.zeros_like(hii_pairs_on) # On-atom, delta_en = 0
            
            # Calculate k_eht
            k_eht_vec_on = self._get_eht_k_factor_vec(
                ishtyp_pairs_on, jshtyp_pairs_on,
                is_tm_i_pairs_on, is_tm_i_pairs_on,      # Same atom
                is_g11_i_pairs_on, is_g11_i_pairs_on, # Same atom
                delta_en_pairs_on
            )
            
            # Calculate slater_exp_corr
            slater_exp_corr_vec_on = (2.0 * torch.sqrt(slater_i_pairs_on * slater_j_pairs_on)) / (slater_i_pairs_on + slater_j_pairs_on)
            
            # shellpoly_corr is 1.0 for on-atom pairs.
            
            # Calculate hav
            hav_vec_on = 0.5 * k_eht_vec_on * (hii_pairs_on + hjj_pairs_on) * slater_exp_corr_vec_on
            
            # Scatter into shell matrix (upper triangle)
            Hav_shell[i_pairs_on, j_pairs_on] = hav_vec_on

        # --- 3b. Off-Atom Pairs (Vectorized) ---
        i_pairs_off = self.i_off_atom_pairs
        j_pairs_off = self.j_off_atom_pairs
        
        if i_pairs_off.numel() > 0:
            # Gather all parameters for all pairs
            iat_pairs_off = self.shell_atom_map[i_pairs_off]
            jat_pairs_off = self.shell_atom_map[j_pairs_off]
            
            hii_pairs_off = hii_all[i_pairs_off]
            hjj_pairs_off = hii_all[j_pairs_off]
            
            ishtyp_pairs_off = self.shell_type_map[i_pairs_off]
            jshtyp_pairs_off = self.shell_type_map[j_pairs_off]
            
            slater_i_pairs_off = self.shell_slater_exp_map[i_pairs_off]
            slater_j_pairs_off = self.shell_slater_exp_map[j_pairs_off]
            
            poly_i_pairs_off = self.shell_poly_const_map[i_pairs_off]
            poly_j_pairs_off = self.shell_poly_const_map[j_pairs_off]

            en_i_pairs_off = self.shell_en_map[i_pairs_off]
            en_j_pairs_off = self.shell_en_map[j_pairs_off]
            
            rad_i_pairs_off = self.shell_rad_map[i_pairs_off]
            rad_j_pairs_off = self.shell_rad_map[j_pairs_off]
            rad_ij_pairs_off = rad_i_pairs_off + rad_j_pairs_off
            
            is_tm_i_pairs_off = self.shell_is_tm_map[i_pairs_off]
            is_tm_j_pairs_off = self.shell_is_tm_map[j_pairs_off]
            is_g11_i_pairs_off = self.shell_is_g11_map[i_pairs_off]
            is_g11_j_pairs_off = self.shell_is_g11_map[j_pairs_off]
            
            vec_i_pairs_off = xyz[iat_pairs_off] # (n_pairs, 3)
            vec_j_pairs_off = xyz[jat_pairs_off] # (n_pairs, 3)
            
            # Perform all calculations vectorized
            delta_en_pairs_off = torch.abs(en_i_pairs_off - en_j_pairs_off)
            
            k_eht_vec_off = self._get_eht_k_factor_vec(
                ishtyp_pairs_off, jshtyp_pairs_off, 
                is_tm_i_pairs_off, is_tm_j_pairs_off, 
                is_g11_i_pairs_off, is_g11_j_pairs_off, 
                delta_en_pairs_off
            )
            
            slater_exp_corr_vec_off = (2.0 * torch.sqrt(slater_i_pairs_off * slater_j_pairs_off)) / (slater_i_pairs_off + slater_j_pairs_off)

            shellpoly_corr_vec_off = self._get_shellpoly_corr_vec(
                vec_i_pairs_off, vec_j_pairs_off, rad_ij_pairs_off, poly_i_pairs_off, poly_j_pairs_off
            )
            
            hav_vec_off = 0.5 * k_eht_vec_off * (hii_pairs_off + hjj_pairs_off) * slater_exp_corr_vec_off * shellpoly_corr_vec_off
            
            # Scatter into shell matrix (upper triangle)
            Hav_shell[i_pairs_off, j_pairs_off] = hav_vec_off

        # --- 4. Assemble Final H0 (Vectorized) ---
        
        # Symmetrize the Hav_shell matrix
        Hav_shell = Hav_shell + Hav_shell.T
        
        # Expand Hav from (n_shell, n_shell) to (n_ao, n_ao) using ao_shell_map
        # Hav_ao[i, j] = Hav_shell[shell_of_ao_i, shell_of_ao_j]
        Hav_ao = Hav_shell[self.ao_shell_map, :][:, self.ao_shell_map]
        
        # Add the off-diagonal part (H_ij = Hav_ij * S_ij) to the diagonal H0
        # Since the diagonal of Hav_ao is 0.0, this does not affect
        # the diagonal elements of H0.
        H0 = H0 + Hav_ao * sint
            
        return H0

    def calculation(self, xyz, q, cn): 
        # xyz: torch.Tensor (N, 3) 
        # q: torch.Tensor (N, 1) or (N,)
        # cn: torch.Tensor (N, 1) or (N,)
        
        # Ensure q and cn are 1D
        q_1d = q.squeeze()
        cn_1d = cn.squeeze()
        
        # 1. Calculate Overlap
        sint = self.overlap_calc.overlap_int_torch(xyz) 

        # 2. Calculate Hamiltonian H0
        h0 = self.get_hamiltonian(xyz, q_1d, cn_1d, sint)
        
        # 3. Solve Generalized Eigenvalue Problem H0 C = S C E
        w_s, v_s = torch.linalg.eigh(sint)
        
        thresh = 1e-8
        mask = w_s > thresh
        w_s_inv_sqrt = torch.zeros_like(w_s)
        w_s_inv_sqrt[mask] = 1.0 / torch.sqrt(w_s[mask])
        
        s_inv_sqrt = torch.matmul(v_s, torch.matmul(torch.diag(w_s_inv_sqrt), v_s.T))
        f_tilde = torch.matmul(s_inv_sqrt, torch.matmul(h0, s_inv_sqrt))
        f_tilde = 0.5 * (f_tilde + f_tilde.T) # Ensure symmetry
        mo_ene, mo_eff_tilde = torch.linalg.eigh(f_tilde)

        C = torch.matmul(s_inv_sqrt, mo_eff_tilde)
        
        n_elec = self.total_valence_e - self.charge_t
        
        # Use torch.floor for differentiability, though this part is usually constant
        n_occ = (n_elec / 2.0).floor().long()
        
        # Proper EHT energy: 2 * sum occupied mo_ene (assuming closed shell)
        energy = 2.0 * torch.sum(mo_ene[:n_occ]) 

        self.mo_energy = mo_ene
        self.mo_coeff = C

        # Return energy
        return energy

    def get_mo_energy(self):
        return self.mo_energy
    
    def get_mo_coeff(self):
        return self.mo_coeff

    def get_overlap_integral_matrix(self):
        return self.overlap_calc.get_overlap_integral_matrix()


    def energy(self, xyz, q, cn):
        xyz_t = torch.tensor(xyz, dtype=self.dtype, device=self.device, requires_grad=True)
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device, requires_grad=True)
        cn_t = torch.tensor(cn, dtype=self.dtype, device=self.device, requires_grad=True)
        
        energy_val = self.calculation(xyz_t, q_t, cn_t) 
        return energy_val 

    def gradient(self, xyz, q, cn, d_eeq_charge, d_cn):
        xyz_t = torch.tensor(xyz, dtype=self.dtype, device=self.device, requires_grad=True)
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device, requires_grad=True)
        cn_t = torch.tensor(cn, dtype=self.dtype, device=self.device, requires_grad=True)
        d_eeq_charge_t = torch.tensor(d_eeq_charge, dtype=self.dtype, device=self.device)
        d_cn_t = torch.tensor(d_cn, dtype=self.dtype, device=self.device)

        gradient_1 = torch.func.jacrev(self.calculation, argnums=0)(xyz_t, q_t, cn_t)
        q_grad = torch.func.jacrev(self.calculation, argnums=1)(xyz_t, q_t, cn_t)
        cn_grad = torch.func.jacrev(self.calculation, argnums=2)(xyz_t, q_t, cn_t)
        
        gradient_2 = torch.einsum('i,ijk->jk', q_grad.squeeze(), d_eeq_charge_t)
        gradient_3 = torch.einsum('i,ijk->jk', cn_grad.squeeze(), d_cn_t)
        
        gradient = gradient_1 + gradient_2 + gradient_3 
        energy = self.energy(xyz, q, cn)
        return energy, gradient

    def hessian(self, xyz, q, cn, d_eeq_charge, dd_eeq_charge, d_cn, dd_cn):
        xyz_t = torch.tensor(xyz, dtype=self.dtype, device=self.device, requires_grad=True)
        q_t = torch.tensor(q, dtype=self.dtype, device=self.device, requires_grad=True)
        cn_t = torch.tensor(cn, dtype=self.dtype, device=self.device, requires_grad=True)
        
        d_eeq_charge_t = torch.tensor(d_eeq_charge, dtype=self.dtype, device=self.device)
        dd_eeq_charge_t = torch.tensor(dd_eeq_charge, dtype=self.dtype, device=self.device)
        d_cn_t = torch.tensor(d_cn, dtype=self.dtype, device=self.device)
        dd_cn_t = torch.tensor(dd_cn, dtype=self.dtype, device=self.device)
        
        n_atoms = xyz_t.shape[0]
        n_dim = n_atoms * 3

        hessian_1_raw = torch.func.hessian(self.calculation, argnums=0)(xyz_t, q_t, cn_t)
        hessian_1 = hessian_1_raw.reshape(n_dim, n_dim)
        
        q_hessian = torch.func.hessian(self.calculation, argnums=1)(xyz_t, q_t, cn_t)
        q_hessian = q_hessian.reshape(n_atoms, n_atoms)
        
        cn_hessian = torch.func.hessian(self.calculation, argnums=2)(xyz_t, q_t, cn_t)
        cn_hessian = cn_hessian.reshape(n_atoms, n_atoms)
        
        q_grad = torch.func.jacrev(self.calculation, argnums=1)(xyz_t, q_t, cn_t).squeeze()
        cn_grad = torch.func.jacrev(self.calculation, argnums=2)(xyz_t, q_t, cn_t).squeeze()
        
        dq_dr = d_eeq_charge_t.permute(0, 2, 1).reshape(n_atoms, n_dim) # (N, N*3)
        dcn_dr = d_cn_t.permute(0, 2, 1).reshape(n_atoms, n_dim) # (N, N*3)
        
        hessian_2 = torch.matmul(dq_dr.T, torch.matmul(q_hessian, dq_dr))
        hessian_3 = torch.matmul(dcn_dr.T, torch.matmul(cn_hessian, dcn_dr))

        hessian_4 = torch.einsum('i,ijk->jk', q_grad, dd_eeq_charge_t)
        hessian_5 = torch.einsum('i,ijk->jk', cn_grad, dd_cn_t)

        hessian = hessian_1 + hessian_2 + hessian_3 + hessian_4 + hessian_5
        
        return hessian
