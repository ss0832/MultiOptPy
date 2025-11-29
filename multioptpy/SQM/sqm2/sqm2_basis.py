import numpy as np


from multioptpy.SQM.sqm2.calc_tools import factorial2, dfactorial
        
class BasisSet:
    def __init__(self, element_list, param):
        self.element_list = element_list
        self.param = param 
        
        # --- Pre-calculate atom properties ---
        # These properties depend only on the element_list, 
        # so they can be set once.
        self.nshells_list = []
        self.ang_shells_list = []
        self.principal_qn_list = []
        self.slater_exponents_list = []
        self.self_energy_list = []
        self.kCN_list = []
        self.en_shell_list = []
        self.atomic_radius_list = []
        self.ref_occ_list = []
        
        for atn in self.element_list:
            self.nshells_list.append(param.nShell[atn])
            self.ang_shells_list.append(param.angShell[atn])
            self.principal_qn_list.append(param.principalQuantumNumber[atn])
            self.slater_exponents_list.append(param.slaterExponent[atn])
            self.self_energy_list.append(param.selfEnergy[atn])
            self.kCN_list.append(param.kCN[atn])
            self.en_shell_list.append(param.electronegativity[atn])
            self.atomic_radius_list.append(param.atomicRad[atn])
            self.ref_occ_list.append(param.referenceOcc[atn])

        self.is_tm_list, self.is_g11_element_list = self.check_transition_metals()
        
        # --- Run the main basis setup ---
        self._set_basis(param)
        return

    def _dim_basis(self, param):
        """
        Calculates the total number of shells (nshell), AOs (nao), and CGFs (nbf).
        Corresponds to Fortran subroutine: dim_basis
        """
        n_atoms = len(self.element_list)
        nao = 0
        nbf = 0
        nshell = 0
        
        for i in range(n_atoms):
            atn = self.element_list[i] # 0-indexed
            k = 0
            
            # Directly reference nShell and angShell from param
            n_shell_for_atom = param.nShell[atn]
            ang_shell_for_atom = param.angShell[atn]

            for j in range(n_shell_for_atom): 
                l = ang_shell_for_atom[j] 
                k += 1
                nshell += 1
                if l == 0:   # s
                    nbf += 1
                    nao += 1
                elif l == 1: # p
                    nbf += 3
                    nao += 3
                elif l == 2: # d
                    nbf += 6 # 6 Cartesian CGFs
                    nao += 5 # 5 spherical AOs
                elif l == 3: # f
                    nbf += 10 # 10 Cartesian CGFs
                    nao += 7  # 7 spherical AOs
                elif l == 4: # g
                    nbf += 15
                    nao += 9
            if k == 0:
                raise Exception(f"No basis found for atom {i} Z={atn+1}")
        
        return nshell, nao, nbf

    def _get_max_shells_per_atom(self):
        """
        Returns the maximum number of shells for any atom in the system.
        Used for allocating caoshell/saoshell arrays.
        """
        max_shells = 0
        for i in range(len(self.element_list)):
            n_shells = self.nshells_list[i]
            if n_shells > max_shells:
                max_shells = n_shells
        return max_shells

    def _set_basis(self, param):
        """
        Main basis set setup routine.
        Corresponds to Fortran subroutine: newBasisset
        """
        n_atoms = len(self.element_list)

        self.basis = {}

        # --- Global Counts (calculated by _dim_basis) ---
        n_shell_total, n_ao_total, n_cgf_total = self._dim_basis(param)
        max_shells_per_atom = self._get_max_shells_per_atom()
        
        self.basis["number_of_atoms"] = n_atoms
        self.basis["number_of_cgf"] = n_cgf_total      # Corresponds to: nbf
        self.basis["number_of_ao"] = n_ao_total       # Corresponds to: nao
        self.basis["number_of_shells"] = n_shell_total  # Corresponds to: nshell
        self.basis["total_number_of_primitives"] = 0   # Final 'ipr' value

        # --- Per-Atom Indices ---
        # Corresponds to Fortran: shells(2,n), fila(2,n), fila2(2,n)
        self.basis["atom_shells_map"] = np.zeros((n_atoms, 2), dtype=int) # [iat, 0]=start, [iat, 1]=end
        self.basis["atom_cgf_map"] = np.zeros((n_atoms, 2), dtype=int)    # [iat, 0]=start, [iat, 1]=end (fila)
        self.basis["atom_ao_map"] = np.zeros((n_atoms, 2), dtype=int)     # [iat, 0]=start, [iat, 1]=end (fila2)

        # --- Per-Atom Per-Shell Mappings (NEW - corresponds to caoshell/saoshell) ---
        # caoshell(m, iat) = CGF index at start of shell m for atom iat
        # saoshell(m, iat) = AO index at start of shell m for atom iat
        # Using shape (n_atoms, max_shells_per_atom) with -1 for unused slots
        self.basis["atom_shell_cgf_offset"] = np.full((n_atoms, max_shells_per_atom), -1, dtype=int)  # caoshell
        self.basis["atom_shell_ao_offset"] = np.full((n_atoms, max_shells_per_atom), -1, dtype=int)   # saoshell

        # --- Per-Shell Data (Lists) ---
        self.basis["shell_amqn_list"] = []           # lsh(ish)
        self.basis["shell_atom_list"] = []           # ash(ish)
        self.basis["shell_cgf_map"] = []             # sh2bf(1:2, ish) -> [start, count]
        self.basis["shell_ao_map"] = []              # sh2ao(1:2, ish) -> [start, count]
        self.basis["shell_level_list"] = []          # level(ish)
        self.basis["shell_zeta_list"] = []           # zeta(ish)
        self.basis["shell_valence_flag_list"] = []   # valsh(ish)
        self.basis["shell_min_alpha_list"] = []      # minalp(ish)

        # --- Per-CGF (Contracted Gaussian Function / Cartesian BF) Data (Lists) ---
        self.basis["cgf_primitive_start_idx_list"] = [] # primcount(ibf)
        self.basis["cgf_valence_flag_list"] = []        # valao(ibf)
        self.basis["cgf_atom_list"] = []                # aoat(ibf)
        self.basis["cgf_type_id_list"] = []             # lao(ibf)
        self.basis["cgf_primitive_count_list"] = []     # nprim(ibf)
        self.basis["cgf_energy_list"] = []              # hdiag(ibf)

        # --- Per-AO (Spherical Atomic Orbital) Data (Lists) ---
        self.basis["ao_valence_flag_list"] = [] # valao2(iao)
        self.basis["ao_atom_list"] = []         # aoat2(iao)
        self.basis["ao_type_id_list"] = []      # lao2(iao)
        self.basis["ao_energy_list"] = []       # hdiag2(iao)
        self.basis["ao_zeta_list"] = []         # aoexp(iao)
        self.basis["ao_shell_list"] = []        # ao2sh(iao)

        # --- Per-PGF (Primitive Gaussian Function) Data (Lists) ---
        self.basis["primitive_alpha_list"] = [] # alp(ipr)
        self.basis["primitive_coeff_list"] = [] # cont(ipr)
        
        # --- Transformation factors for d/f functions ---
        # Corresponds to Fortran trafo arrays in set_d_function and set_f_function
        d_trafo = { 5: 1.0, 6: 1.0, 7: 1.0, 8: np.sqrt(3.0), 9: np.sqrt(3.0), 10: np.sqrt(3.0) }
        f_trafo = { 11: 1.0, 12: 1.0, 13: 1.0, 14: np.sqrt(5.0), 15: np.sqrt(5.0),
                    16: np.sqrt(5.0), 17: np.sqrt(5.0), 18: np.sqrt(5.0), 19: np.sqrt(5.0),
                    20: np.sqrt(15.0) }

        # --- Global 0-indexed Counters ---
        ibf = 0  # CGF counter
        iao = 0  # AO counter
        ipr = 0  # Primitive counter
        ish = 0  # Shell counter

        # === Main Loop (Corresponds to: atoms: do iat=1,n) ===
        for iat in range(n_atoms):
            atn = self.element_list[iat] # 0-indexed atomic number
            ati = atn + 1                # 1-indexed atomic number (for H/He checks)
            
            # These are held within the iat loop and used for diffuse s processing
            nprim_s_val = 0
            alpha_s_val = np.array([])
            coeff_s_val = np.array([])

            # Set Per-Atom Start Indices
            # Corresponds to: basis%shells(1,iat)=ish+1, basis%fila(1,iat)=ibf+1, basis%fila2(1,iat)=iao+1
            self.basis["atom_shells_map"][iat, 0] = ish 
            self.basis["atom_cgf_map"][iat, 0] = ibf
            self.basis["atom_ao_map"][iat, 0] = iao

            # Corresponds to: shells: do m=1,xtbData%nShell(ati)
            nshell_i = self.nshells_list[iat]
            for m in range(nshell_i):
                # --- Get Shell Parameters ---
                l = self.ang_shells_list[iat][m]
                npq = self.principal_qn_list[iat][m]
                level = self.self_energy_list[iat][m]
                zeta = self.slater_exponents_list[iat][m]
                
                valao = self.ref_occ_list[iat][m] # The valence flag
                is_valence = (valao != 0)

                current_nprim_or_R = self.get_number_of_primitives(atn, l, npq, is_valence=is_valence)

                ibf_start_shell = ibf
                iao_start_shell = iao

                # --- Store Per-Atom Per-Shell Offsets (caoshell/saoshell) ---
                self.basis["atom_shell_cgf_offset"][iat, m] = ibf
                self.basis["atom_shell_ao_offset"][iat, m] = iao

                # --- Store Per-Shell Data (at index ish) ---
                self.basis["shell_amqn_list"].append(l)
                self.basis["shell_atom_list"].append(iat)
                self.basis["shell_cgf_map"].append([ibf_start_shell, 0]) # [start, count]
                self.basis["shell_ao_map"].append([iao_start_shell, 0])  # [start, count]
                self.basis["shell_level_list"].append(level)
                self.basis["shell_zeta_list"].append(zeta)
                self.basis["shell_valence_flag_list"].append(valao)
                
                min_alpha_for_shell = np.inf

                # === Process Shells by Type ===

                # --- H-He s (Valence) ---
                if l == 0 and ati <= 2 and is_valence:
                    
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    nprim_s_val = current_nprim_or_R
                    alpha_s_val = alpha.copy()
                    coeff_s_val = coeff.copy()
                    
                    # Add CGF (1)
                    self.basis["cgf_primitive_start_idx_list"].append(ipr)
                    self.basis["cgf_valence_flag_list"].append(valao)
                    self.basis["cgf_atom_list"].append(iat)
                    self.basis["cgf_type_id_list"].append(1) # 1=s
                    self.basis["cgf_primitive_count_list"].append(current_nprim_or_R)
                    self.basis["cgf_energy_list"].append(level)
                    ibf += 1
                    
                    # Add Primitives
                    for p in range(current_nprim_or_R):
                        self.basis["primitive_alpha_list"].append(alpha[p])
                        self.basis["primitive_coeff_list"].append(coeff[p])
                        ipr += 1
                        
                    # Add AO (1)
                    self.basis["ao_valence_flag_list"].append(valao)
                    self.basis["ao_atom_list"].append(iat)
                    self.basis["ao_type_id_list"].append(1) # 1=s
                    self.basis["ao_energy_list"].append(level)
                    self.basis["ao_zeta_list"].append(zeta)
                    self.basis["ao_shell_list"].append(ish)
                    iao += 1

                # --- H-He s (Diffuse) ---
                elif l == 0 and ati <= 2 and not is_valence:
                  
                    alpha_R, coeff_R = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    
                    # Get overlap with valence s 
                    ss = self.calc_overlap_int_for_diffuse_func(0, alpha_s_val, alpha_R, coeff_s_val, coeff_R)
                    min_alpha_for_shell = min(np.min(alpha_s_val), np.min(alpha_R))

                    nprim_total = current_nprim_or_R + nprim_s_val
                    
                    # Add CGF (1)
                    self.basis["cgf_primitive_start_idx_list"].append(ipr)
                    self.basis["cgf_valence_flag_list"].append(valao)
                    self.basis["cgf_atom_list"].append(iat)
                    self.basis["cgf_type_id_list"].append(1) # 1=s
                    self.basis["cgf_primitive_count_list"].append(nprim_total)
                    self.basis["cgf_energy_list"].append(level)
                    ibf += 1
                    
                    # Add Primitives (Diffuse part first, then valence orthogonalized)
                    all_alphas = []
                    all_coeffs = []
                    prim_start_idx_for_this_cgf = ipr
                    
                    for p in range(current_nprim_or_R):
                        alpha_p = alpha_R[p]
                        coeff_p = coeff_R[p]
                        self.basis["primitive_alpha_list"].append(alpha_p)
                        self.basis["primitive_coeff_list"].append(coeff_p)
                        all_alphas.append(alpha_p)
                        all_coeffs.append(coeff_p)
                        ipr += 1

                    # Add Primitives (Valence part, orthogonalized)
                    for p in range(nprim_s_val):
                        alpha_p = alpha_s_val[p]
                        coeff_p = -ss * coeff_s_val[p]
                        self.basis["primitive_alpha_list"].append(alpha_p)
                        self.basis["primitive_coeff_list"].append(coeff_p)
                        all_alphas.append(alpha_p)
                        all_coeffs.append(coeff_p)
                        ipr += 1

                    # Renormalize the new CGF
                    all_alphas = np.array(all_alphas)
                    all_coeffs = np.array(all_coeffs)
                    ss_norm = self.calc_overlap_int_for_diffuse_func(0, all_alphas, all_alphas, all_coeffs, all_coeffs)
                    norm_factor = 1.0 / np.sqrt(ss_norm)
                    
                    # Update the coefficients in the global list
                    for p in range(nprim_total):
                        self.basis["primitive_coeff_list"][prim_start_idx_for_this_cgf + p] *= norm_factor
                    
                    # Add AO (1)
                    self.basis["ao_valence_flag_list"].append(valao)
                    self.basis["ao_atom_list"].append(iat)
                    self.basis["ao_type_id_list"].append(1) # 1=s
                    self.basis["ao_energy_list"].append(level)
                    self.basis["ao_zeta_list"].append(zeta)
                    self.basis["ao_shell_list"].append(ish)
                    iao += 1
                    
                # --- H-He p (Polarization) ---
                elif l == 1 and ati <= 2:
                    valao_p_pol = -valao 
                    
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    for j_type in [2, 3, 4]: # px, py, pz
                        # Add CGF (1)
                        self.basis["cgf_primitive_start_idx_list"].append(ipr)
                        self.basis["cgf_valence_flag_list"].append(valao_p_pol)
                        self.basis["cgf_atom_list"].append(iat)
                        self.basis["cgf_type_id_list"].append(j_type)
                        self.basis["cgf_primitive_count_list"].append(current_nprim_or_R)
                        self.basis["cgf_energy_list"].append(level)
                        ibf += 1

                        # Add Primitives
                        for p in range(current_nprim_or_R):
                            self.basis["primitive_alpha_list"].append(alpha[p])
                            self.basis["primitive_coeff_list"].append(coeff[p])
                            ipr += 1

                        # Add AO (1)
                        self.basis["ao_valence_flag_list"].append(valao_p_pol)
                        self.basis["ao_atom_list"].append(iat)
                        self.basis["ao_type_id_list"].append(j_type)
                        self.basis["ao_energy_list"].append(level)
                        self.basis["ao_zeta_list"].append(zeta)
                        self.basis["ao_shell_list"].append(ish)
                        iao += 1

                # --- General s (Valence) ---
                elif l == 0 and ati > 2 and is_valence:
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    # Store for use by diffuse shell
                    nprim_s_val = current_nprim_or_R
                    alpha_s_val = alpha.copy()
                    coeff_s_val = coeff.copy()
                    
                    # Add CGF (1)
                    self.basis["cgf_primitive_start_idx_list"].append(ipr)
                    self.basis["cgf_valence_flag_list"].append(valao)
                    self.basis["cgf_atom_list"].append(iat)
                    self.basis["cgf_type_id_list"].append(1) # 1=s
                    self.basis["cgf_primitive_count_list"].append(current_nprim_or_R)
                    self.basis["cgf_energy_list"].append(level)
                    ibf += 1
                    
                    # Add Primitives
                    for p in range(current_nprim_or_R):
                        self.basis["primitive_alpha_list"].append(alpha[p])
                        self.basis["primitive_coeff_list"].append(coeff[p])
                        ipr += 1
                        
                    # Add AO (1)
                    self.basis["ao_valence_flag_list"].append(valao)
                    self.basis["ao_atom_list"].append(iat)
                    self.basis["ao_type_id_list"].append(1) # 1=s
                    self.basis["ao_energy_list"].append(level)
                    self.basis["ao_zeta_list"].append(zeta)
                    self.basis["ao_shell_list"].append(ish)
                    iao += 1

                # --- General p ---
                elif l == 1 and ati > 2:
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    for j_type in [2, 3, 4]: # px, py, pz
                        # Add CGF (1)
                        self.basis["cgf_primitive_start_idx_list"].append(ipr)
                        self.basis["cgf_valence_flag_list"].append(valao)
                        self.basis["cgf_atom_list"].append(iat)
                        self.basis["cgf_type_id_list"].append(j_type)
                        self.basis["cgf_primitive_count_list"].append(current_nprim_or_R)
                        self.basis["cgf_energy_list"].append(level)
                        ibf += 1

                        # Add Primitives
                        for p in range(current_nprim_or_R):
                            self.basis["primitive_alpha_list"].append(alpha[p])
                            self.basis["primitive_coeff_list"].append(coeff[p])
                            ipr += 1

                        # Add AO (1)
                        self.basis["ao_valence_flag_list"].append(valao)
                        self.basis["ao_atom_list"].append(iat)
                        self.basis["ao_type_id_list"].append(j_type)
                        self.basis["ao_energy_list"].append(level)
                        self.basis["ao_zeta_list"].append(zeta)
                        self.basis["ao_shell_list"].append(ish)
                        iao += 1
                        
                # --- General s (Diffuse) ---
                elif l == 0 and ati > 2 and not is_valence:
                    alpha_R, coeff_R = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    # Get overlap with valence s
                    ss = self.calc_overlap_int_for_diffuse_func(0, alpha_s_val, alpha_R, coeff_s_val, coeff_R)
                    min_alpha_for_shell = min(np.min(alpha_s_val), np.min(alpha_R))

                    nprim_total = current_nprim_or_R + nprim_s_val
                    
                    # Add CGF (1)
                    self.basis["cgf_primitive_start_idx_list"].append(ipr)
                    self.basis["cgf_valence_flag_list"].append(valao)
                    self.basis["cgf_atom_list"].append(iat)
                    self.basis["cgf_type_id_list"].append(1) # 1=s
                    self.basis["cgf_primitive_count_list"].append(nprim_total)
                    self.basis["cgf_energy_list"].append(level)
                    ibf += 1
                    
                    # Add Primitives (Diffuse part)
                    all_alphas = []
                    all_coeffs = []
                    prim_start_idx_for_this_cgf = ipr

                    for p in range(current_nprim_or_R):
                        alpha_p = alpha_R[p]
                        coeff_p = coeff_R[p]
                        self.basis["primitive_alpha_list"].append(alpha_p)
                        self.basis["primitive_coeff_list"].append(coeff_p)
                        all_alphas.append(alpha_p)
                        all_coeffs.append(coeff_p)
                        ipr += 1
                    
                    # Add Primitives (Valence part, orthogonalized)
                    for p in range(nprim_s_val):
                        alpha_p = alpha_s_val[p]
                        coeff_p = -ss * coeff_s_val[p]
                        self.basis["primitive_alpha_list"].append(alpha_p)
                        self.basis["primitive_coeff_list"].append(coeff_p)
                        all_alphas.append(alpha_p)
                        all_coeffs.append(coeff_p)
                        ipr += 1

                    # Renormalize the new CGF
                    all_alphas = np.array(all_alphas)
                    all_coeffs = np.array(all_coeffs)
                    ss_norm = self.calc_overlap_int_for_diffuse_func(0, all_alphas, all_alphas, all_coeffs, all_coeffs)
                    norm_factor = 1.0 / np.sqrt(ss_norm)
                    
                    # Update the coefficients in the global list
                    for p in range(nprim_total):
                        self.basis["primitive_coeff_list"][prim_start_idx_for_this_cgf + p] *= norm_factor

                    # Add AO (1)
                    self.basis["ao_valence_flag_list"].append(valao)
                    self.basis["ao_atom_list"].append(iat)
                    self.basis["ao_type_id_list"].append(1) # 1=s
                    self.basis["ao_energy_list"].append(level)
                    self.basis["ao_zeta_list"].append(zeta)
                    self.basis["ao_shell_list"].append(ish)
                    iao += 1

                # --- d functions ---
                elif l == 2:
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)

                    for j_type in range(5, 11): # 5..10 (6 CGFs: dxx, dyy, dzz, dxy, dxz, dyz)
                        # Add CGF (1)
                        self.basis["cgf_primitive_start_idx_list"].append(ipr)
                        self.basis["cgf_valence_flag_list"].append(valao)
                        self.basis["cgf_atom_list"].append(iat)
                        self.basis["cgf_type_id_list"].append(j_type)
                        self.basis["cgf_primitive_count_list"].append(current_nprim_or_R)
                        self.basis["cgf_energy_list"].append(level)
                        ibf += 1
                        
                        # Add Primitives (with transformation factor)
                        trafo = d_trafo[j_type]
                        for p in range(current_nprim_or_R):
                            self.basis["primitive_alpha_list"].append(alpha[p])
                            self.basis["primitive_coeff_list"].append(coeff[p] * trafo)
                            ipr += 1
                        
                        # Add AO (5 spherical AOs for 6 Cartesian CGFs)
                        # Skip for j_type=5 (dxx) - Fortran: if (j .eq.5) cycle
                        if j_type != 5:
                            ao_type_id = j_type - 1 # AO IDs: 4,5,6,7,8,9 for j_type 6,7,8,9,10
                            self.basis["ao_valence_flag_list"].append(valao)
                            self.basis["ao_atom_list"].append(iat)
                            self.basis["ao_type_id_list"].append(ao_type_id) 
                            self.basis["ao_energy_list"].append(level)
                            self.basis["ao_zeta_list"].append(zeta)
                            self.basis["ao_shell_list"].append(ish)
                            iao += 1

                # --- f functions ---
                elif l == 3:
                    # Fortran hardcodes valao=1 for f functions
                    valao_f = 1 
                    
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    for j_type in range(11, 21): # 11..20 (10 Cartesian CGFs)
                        # Add CGF (1)
                        self.basis["cgf_primitive_start_idx_list"].append(ipr)
                        self.basis["cgf_valence_flag_list"].append(valao_f)
                        self.basis["cgf_atom_list"].append(iat)
                        self.basis["cgf_type_id_list"].append(j_type)
                        self.basis["cgf_primitive_count_list"].append(current_nprim_or_R)
                        self.basis["cgf_energy_list"].append(level)
                        ibf += 1
                        
                        # Add Primitives (with transformation factor)
                        trafo = f_trafo[j_type]
                        for p in range(current_nprim_or_R):
                            self.basis["primitive_alpha_list"].append(alpha[p])
                            self.basis["primitive_coeff_list"].append(coeff[p] * trafo)
                            ipr += 1

                        # Add AO (7 spherical AOs for 10 Cartesian CGFs)
                        # Skip for j_type 11,12,13 (fxxx, fyyy, fzzz)
                        # Fortran: if (j.ge.11 .and.j.le.13) cycle
                        if j_type > 13:
                            ao_type_id = j_type - 3 # AO IDs: 11,12,13,14,15,16,17 for j_type 14..20
                            self.basis["ao_valence_flag_list"].append(valao_f)
                            self.basis["ao_atom_list"].append(iat)
                            self.basis["ao_type_id_list"].append(ao_type_id) 
                            self.basis["ao_energy_list"].append(level)
                            self.basis["ao_zeta_list"].append(zeta)
                            self.basis["ao_shell_list"].append(ish)
                            iao += 1
                            
                # --- End of shell type processing ---
                
                # Store min alpha for this shell
                self.basis["shell_min_alpha_list"].append(min_alpha_for_shell)
                
                # --- Set Per-Shell Counts ---
                self.basis["shell_cgf_map"][ish][1] = ibf - ibf_start_shell
                self.basis["shell_ao_map"][ish][1] = iao - iao_start_shell

                # --- Increment global shell counter ---
                ish += 1 
            
            # --- Set Per-Atom End Indices ---
            self.basis["atom_shells_map"][iat, 1] = ish 
            self.basis["atom_cgf_map"][iat, 1] = ibf
            self.basis["atom_ao_map"][iat, 1] = iao

        # --- Finalize Global Counts ---
        self.basis["total_number_of_primitives"] = ipr
        
        # --- Final Sanity Checks (corresponds to Fortran ok check) ---
        all_alphas_positive = all(a > 0.0 for a in self.basis["primitive_alpha_list"])
        if n_cgf_total != ibf:
            print(f"Warning: CGF count mismatch. Calculated={n_cgf_total}, Found={ibf}")
        if n_ao_total != iao:
            print(f"Warning: AO count mismatch.Calculated={n_ao_total}, Found={iao}")
        if n_shell_total != ish:
            print(f"Warning: Shell count mismatch. Calculated={n_shell_total}, Found={ish}")
        if not all_alphas_positive:
            print("Warning: Some primitive exponents are non-positive!")
        
        return


    def get_number_of_primitives(self, element, angmn, pqn, is_valence):
        """
        Returns the number of primitive Gaussians for STO-nG expansion.
        
        This function determines nprim based on element type and orbital type.
        Note: In the original Fortran, this would be read from xtbData%hamiltonian%numberOfPrimitives.
        Here we use hardcoded rules matching typical GFN0 behavior.
        """
        atom_num = element + 1 # Convert 0-indexed to 1-indexed atomic number
        
        if atom_num <= 2: # H, He
            if is_valence:
                n = 3 # STO-3G 
            else:
                n = 2 # STO-2G (thisprimR for diffuse)
        else:
            if angmn == 0: # s orbital
                if pqn > 5:
                    n = 6 # STO-6G
                else:
                    n = 4 # STO-4G
            elif angmn == 1: # p orbital
                if pqn > 5:
                    n = 6 # STO-6G
                else:
                    n = 3 # STO-3G
            elif angmn == 2 or angmn == 3: # d orbital or f orbital
                n = 4 # STO-4G
            else:
                raise ValueError(f"Error in get_number_of_primitives: angmn={angmn} > 3")

        return n


    def check_transition_metals(self):
        """
        Identifies transition metals and Group 11 elements in the element list.
        """
        is_tm_list = []
        is_g11_element_list = []
        
        g11_z_numbers = {29, 47, 79, 111} # Cu, Ag, Au, Rg

        for atn in self.element_list:
            z = atn + 1
            
            if z in g11_z_numbers:
                is_g11_element_list.append(True)
                is_tm_list.append(False) 
            elif (z >= 21 and z <= 30) or \
                 (z >= 39 and z <= 48) or \
                 (z >= 57 and z <= 80) or \
                 (z >= 89 and z <= 112):
                is_tm_list.append(True)
                is_g11_element_list.append(False)
            else:
                is_tm_list.append(False)
                is_g11_element_list.append(False)
                
        return is_tm_list, is_g11_element_list

    def set_valence_orbitals(self):
        """
        Determines which shells are valence vs diffuse based on angular momentum.
        NOTE: This method is not called by _set_basis. 
        It may be used elsewhere for analysis.
        """
        self.valence_orbitals_list = []
        for i in range(len(self.element_list)):
            seen_ang_mom = set()
            for shell_idx in range(self.nshells_list[i]):
                ang = self.ang_shells_list[i][shell_idx]
                if ang not in seen_ang_mom:
                    self.valence_orbitals_list.append(True)
                    seen_ang_mom.add(ang)
                else:
                    self.valence_orbitals_list.append(False)
                        
        return self.valence_orbitals_list
    
    def slater2gauss(self, nprim, pqn, angmn, slater_exp, param, need_normalization=True):
        """
        Converts Slater-type orbital to a sum of Gaussian primitives (STO-nG).
        Corresponds to Fortran subroutine: slaterToGauss
        
        Parameters:
            nprim: Number of primitive Gaussians
            pqn: Principal quantum number
            angmn: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f)
            slater_exp: Slater exponent (zeta)
            param: Parameter object containing STO-nG coefficients
            need_normalization: Whether to normalize the primitives
            
        Returns:
            alpha: Array of Gaussian exponents
            coeff: Array of contraction coefficients
        """
        alpha = np.zeros(nprim)
        coeff = np.zeros(nprim)
        
        if pqn < angmn + 1:
            raise ValueError(f"Error in slater2gauss: pqn ({pqn}) < angmn + 1 ({angmn + 1})")
        
        if slater_exp <= 0.0:
            raise ValueError(f"Error in slater2gauss: non-positive slater exponent ({slater_exp})")
        
        # Determine index into parameter tables based on orbital type
        tmp_num = -1
        if angmn == 0: # s orbital
            tmp_num = pqn - 1
        elif angmn == 1: # p orbital
            tmp_num = 4 + pqn - 1
        elif angmn == 2: # d orbital
            tmp_num = 7 + pqn - 1
        elif angmn == 3: # f orbital
            tmp_num = 9 + pqn - 1
        elif angmn == 4: # g orbital
            tmp_num = 10 + pqn - 1
        else:
            raise ValueError(f"Error in slater2gauss: angmn ({angmn}) > 4")
        
        slater_exp_sq = slater_exp ** 2.0
        
        if nprim == 1:
            alpha[0] = param.pAlpha1[tmp_num] * slater_exp_sq
            coeff[0] = 1.0
        elif nprim == 2:
            alpha = param.pAlpha2[tmp_num, :].copy() * slater_exp_sq
            coeff = param.pCoeff2[tmp_num, :].copy()
        elif nprim == 3:
            alpha = param.pAlpha3[tmp_num, :].copy() * slater_exp_sq
            coeff = param.pCoeff3[tmp_num, :].copy()
        elif nprim == 4:
            alpha = param.pAlpha4[tmp_num, :].copy() * slater_exp_sq
            coeff = param.pCoeff4[tmp_num, :].copy()
        elif nprim == 5:
            alpha = param.pAlpha5[tmp_num, :].copy() * slater_exp_sq
            coeff = param.pCoeff5[tmp_num, :].copy()
        elif nprim == 6:
            if pqn == 6:
                if angmn == 0:
                    alpha = param.pAlpha6s[:].copy() * slater_exp_sq
                    coeff = param.pCoeff6s[:].copy()
                elif angmn == 1:
                    alpha = param.pAlpha6p[:].copy() * slater_exp_sq
                    coeff = param.pCoeff6p[:].copy()
                else:
                    raise ValueError(f"Error in slater2gauss: invalid angmn ({angmn}) for pqn=6 with nprim=6")
            else:
                alpha = param.pAlpha6[tmp_num, :].copy() * slater_exp_sq
                coeff = param.pCoeff6[tmp_num, :].copy()
        else:
            raise ValueError(f"Error in slater2gauss: invalid number of primitives ({nprim})")
        
        if need_normalization:
            # Normalize primitives
            # Formula: coeff = coeff * (2/pi * alpha)^(3/4) * (4*alpha)^(l/2) / sqrt((2l-1)! !)
            # where (2l-1)!! is the double factorial
            top = 2.0 / np.pi
            # dfactorial(angmn + 1) should give (2*angmn - 1)!!  
            # For l=0: (2*0-1)!! = (-1)!! = 1
            # For l=1: (2*1-1)!! = 1!! = 1
            # For l=2: (2*2-1)!! = 3!! = 3
            # For l=3: (2*3-1)!!  = 5!! = 15
            coeff = coeff * (top * alpha) ** 0.75 * np.sqrt(4.0 * alpha) ** angmn / np.sqrt(dfactorial(angmn + 1))
                
        return alpha, coeff

    
    def calc_overlap_int_for_diffuse_func(self, angmn, alpha_A, alpha_B, coeff_A, coeff_B):
        """ 
        Calculates the overlap integral <A|B> for two CGFs on the same center.
        Corresponds to Fortran subroutine: atovlp
        
        Parameters:
            angmn: Angular momentum quantum number (0=s, 1=p)
            alpha_A, alpha_B: Primitive exponent arrays
            coeff_A, coeff_B: Contraction coefficient arrays
            
        Returns:
            overlap_int: The overlap integral value
        """
        overlap_int = 0.0
        
        n_prim_A = len(coeff_A)
        n_prim_B = len(coeff_B)

        if n_prim_A == 0 or n_prim_B == 0:
            print("Warning: calc_overlap_int_for_diffuse_func called with empty primitive list.")
            return 0.0

        for ii in range(n_prim_A):
            for jj in range(n_prim_B):
                ab = 1.0 / (alpha_A[ii] + alpha_B[jj])
                s00 = (np.pi * ab) ** 1.5
                
                sss = 0.0
                if angmn == 0: # s-s overlap
                    sss = s00
                elif angmn == 1: # p-p overlap (same component)
                    ab05 = ab * 0.5 
                    sss = s00 * ab05
                
                overlap_int += sss * coeff_A[ii] * coeff_B[jj]           
                
        return overlap_int


    # === Accessor methods for convenient data retrieval ===
    
    def get_cgf_offset_for_atom_shell(self, iat, m):
        """
        Returns the CGF index at the start of shell m for atom iat.
        Corresponds to Fortran: caoshell(m, iat)
        """
        return self.basis["atom_shell_cgf_offset"][iat, m]
    
    def get_ao_offset_for_atom_shell(self, iat, m):
        """
        Returns the AO index at the start of shell m for atom iat.
        Corresponds to Fortran: saoshell(m, iat)
        """
        return self.basis["atom_shell_ao_offset"][iat, m]
    
    def get_shell_range_for_atom(self, iat):
        """
        Returns (start_shell, end_shell) for atom iat.
        Corresponds to Fortran: shells(1:2, iat)
        """
        return (self.basis["atom_shells_map"][iat, 0], 
                self.basis["atom_shells_map"][iat, 1])
    
    def get_cgf_range_for_atom(self, iat):
        """
        Returns (start_cgf, end_cgf) for atom iat.
        Corresponds to Fortran: fila(1:2, iat)
        """
        return (self.basis["atom_cgf_map"][iat, 0], 
                self.basis["atom_cgf_map"][iat, 1])
    
    def get_ao_range_for_atom(self, iat):
        """
        Returns (start_ao, end_ao) for atom iat.
        Corresponds to Fortran: fila2(1:2, iat)
        """
        return (self.basis["atom_ao_map"][iat, 0], 
                self.basis["atom_ao_map"][iat, 1])
    
    def get_cgf_range_for_shell(self, ish):
        """
        Returns (start_cgf, count) for shell ish.
        Corresponds to Fortran: sh2bf(1:2, ish)
        """
        return tuple(self.basis["shell_cgf_map"][ish])
    
    def get_ao_range_for_shell(self, ish):
        """
        Returns (start_ao, count) for shell ish.
        Corresponds to Fortran: sh2ao(1:2, ish)
        """
        return tuple(self.basis["shell_ao_map"][ish])
    
    def get_primitives_for_cgf(self, ibf):
        """
        Returns (start_prim_idx, nprim, alphas, coeffs) for CGF ibf.
        """
        start_idx = self.basis["cgf_primitive_start_idx_list"][ibf]
        nprim = self.basis["cgf_primitive_count_list"][ibf]
        alphas = self.basis["primitive_alpha_list"][start_idx:start_idx + nprim]
        coeffs = self.basis["primitive_coeff_list"][start_idx:start_idx + nprim]
        return start_idx, nprim, alphas, coeffs