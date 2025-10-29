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
                    nao += 5 # 5 AOs
                elif l == 3: # f
                    nbf += 10 # 10 Cartesian CGFs
                    nao += 7  # 7 AOs
                elif l == 4: # g
                    nbf += 15
                    nao += 9
            if k == 0:
                raise Exception(f"No basis found for atom {i} Z={atn+1}")
        
        return nshell, nao, nbf

    def _set_basis(self, param):
        # NOTE: Per-atom parameter lists (like self.nshells_list)
        # are now set in __init__
        
        ### dict of Orbital Information
        n_atoms = len(self.element_list)

        self.basis = {}

        # --- Global Counts (calculated by _dim_basis) ---
        n_shell_total, n_ao_total, n_cgf_total = self._dim_basis(param)
        
        self.basis["number_of_atoms"] = n_atoms
        self.basis["number_of_cgf"] = n_cgf_total      # Corresponds to: nbf
        self.basis["number_of_ao"] = n_ao_total       # Corresponds to: nao
        self.basis["number_of_shells"] = n_shell_total  # Corresponds to: nsh
        self.basis["total_number_of_primitives"] = 0   # Final 'ipr' value

        # --- Per-Atom Indices ---
        self.basis["atom_shells_map"] = np.zeros((n_atoms, 2), dtype=int) # [iat, 0]=start, [iat, 1]=end
        self.basis["atom_cgf_map"] = np.zeros((n_atoms, 2), dtype=int)    # [iat, 0]=start, [iat, 1]=end
        self.basis["atom_ao_map"] = np.zeros((n_atoms, 2), dtype=int)     # [iat, 0]=start, [iat, 1]=end

        # --- Per-Shell Data (Lists) ---
        self.basis["shell_amqn_list"] = []           # lsh(ish)
        self.basis["shell_atom_list"] = []           # ash(ish)
        self.basis["shell_cgf_map"] = []             # sh2bf(1:2, ish) -> [start, count]
        self.basis["shell_ao_map"] = []              # sh2ao(1:2, ish) -> [start, count]
        self.basis["shell_level_list"] = []          # level(ish)
        self.basis["shell_zeta_list"] = []           # zeta(ish)
        self.basis["shell_valence_flag_list"] = []   # valsh(ish) (valao goes here)
        self.basis["shell_min_alpha_list"] = []      # minalp(ish)

        # --- Per-CGF (Contracted Gaussian Function) Data (Lists) ---
        self.basis["cgf_primitive_start_idx_list"] = [] # primcount(ibf)
        self.basis["cgf_valence_flag_list"] = []        # valao(ibf)
        self.basis["cgf_atom_list"] = []                # aoat(ibf)
        self.basis["cgf_type_id_list"] = []             # lao(ibf)
        self.basis["cgf_primitive_count_list"] = []     # nprim(ibf)
        self.basis["cgf_energy_list"] = []              # hdiag(ibf)

        # --- Per-AO (Atomic Orbital) Data (Lists) ---
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
            self.basis["atom_shells_map"][iat, 0] = ish 
            self.basis["atom_cgf_map"][iat, 0] = ibf
            self.basis["atom_ao_map"][iat, 0] = iao

            # Corresponds to: shells: do m=1,xtbData%nShell(ati)
            nshell_i = self.nshells_list[iat]
            for m in range(nshell_i):
                # 'ish' is the current 0-indexed global shell index (0, 1, 2, ...)
                
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

                # --- Store Per-Shell Data (at index ish) ---
                self.basis["shell_amqn_list"].append(l)
                self.basis["shell_atom_list"].append(iat)
                self.basis["shell_cgf_map"].append([ibf_start_shell, 0]) # [start, count]
                self.basis["shell_ao_map"].append([iao_start_shell, 0])  # [start, count]
                self.basis["shell_level_list"].append(level)
                self.basis["shell_zeta_list"].append(zeta)
                self.basis["shell_valence_flag_list"].append(valao) # Store valao (int) directly
                
                min_alpha_for_shell = np.inf

                # === Process Shells by Type ===

                # --- H-He s (Valence) ---
                if l == 0 and ati <= 2 and is_valence: # valao != 0
                    
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    nprim_s_val = current_nprim_or_R
                    alpha_s_val = alpha
                    coeff_s_val = coeff
                    
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
                    self.basis["ao_shell_list"].append(ish) # Link to current shell
                    iao += 1

                # --- H-He s (Diffuse) ---
                # orthogonalize and re-normalize.
                elif l == 0 and ati <= 2 and not is_valence: # valao == 0
                  
                    # current_nprim_or_R is 2 
                    alpha_R, coeff_R = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    
                    # Get overlap with valence s 
                    ss = self.calc_overlap_int_for_diffuse_func(0, alpha_s_val, alpha_R, coeff_s_val, coeff_R)
                    min_alpha_for_shell = min(np.min(alpha_s_val), np.min(alpha_R))

                    nprim_total = current_nprim_or_R + nprim_s_val # current_nprim_or_R = 2
                    
                    # Add CGF (1)
                    self.basis["cgf_primitive_start_idx_list"].append(ipr)
                    self.basis["cgf_valence_flag_list"].append(valao) # valao is 0
                    self.basis["cgf_atom_list"].append(iat)
                    self.basis["cgf_type_id_list"].append(1) # 1=s
                    self.basis["cgf_primitive_count_list"].append(nprim_total) # 2 + 3 = 5
                    self.basis["cgf_energy_list"].append(level)
                    ibf += 1
                    
                    # Add Primitives (Diffuse part)
                    all_alphas = []
                    all_coeffs = []
                    prim_start_idx_for_this_cgf = ipr # idum=ipr+1
                    
                    for p in range(current_nprim_or_R): # nprim = thisprimR (2)
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
                    
                    # Update the coefficients in the global list (idum-1+p)
                    for p in range(nprim_total):
                        self.basis["primitive_coeff_list"][prim_start_idx_for_this_cgf + p] *= norm_factor
                    
                    # Add AO (1)
                    self.basis["ao_valence_flag_list"].append(valao) # valao is 0
                    self.basis["ao_atom_list"].append(iat)
                    self.basis["ao_type_id_list"].append(1) # 1=s
                    self.basis["ao_energy_list"].append(level)
                    self.basis["ao_zeta_list"].append(zeta)
                    self.basis["ao_shell_list"].append(ish)
                    iao += 1
                    
                # --- H-He p (Polarization) ---
                elif l == 1 and ati <= 2:
                    # Here, valao is the flag for the p-pol shell (e.g., 1)
                    valao_p_pol = -valao 
                    
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    for j_type in [2, 3, 4]: # px, py, pz
                        # Add CGF (1)
                        self.basis["cgf_primitive_start_idx_list"].append(ipr)
                        self.basis["cgf_valence_flag_list"].append(valao_p_pol) # Use negative flag
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
                        self.basis["ao_valence_flag_list"].append(valao_p_pol) # Use negative flag
                        self.basis["ao_atom_list"].append(iat)
                        self.basis["ao_type_id_list"].append(j_type)
                        self.basis["ao_energy_list"].append(level)
                        self.basis["ao_zeta_list"].append(zeta)
                        self.basis["ao_shell_list"].append(ish)
                        iao += 1

                # --- General s (Valence) ---
                elif l == 0 and ati > 2 and is_valence: # valao != 0
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    # Store for use by diffuse shell (as, cs, nprim)
                    nprim_s_val = current_nprim_or_R
                    alpha_s_val = alpha
                    coeff_s_val = coeff
                    
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
                    # or !=0 (valence p))
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
                elif l == 0 and ati > 2 and not is_valence: # valao == 0
                    # current_nprim_or_R is 'thisprimR'
                   
                    alpha_R, coeff_R = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    # Get overlap with valence s (as, cs, nprim)
                    ss = self.calc_overlap_int_for_diffuse_func(0, alpha_s_val, alpha_R, coeff_s_val, coeff_R)
                    min_alpha_for_shell = min(np.min(alpha_s_val), np.min(alpha_R))

                    nprim_total = current_nprim_or_R + nprim_s_val # current_nprim = thisprimR
                    
                    # Add CGF (1)
                    self.basis["cgf_primitive_start_idx_list"].append(ipr)
                    self.basis["cgf_valence_flag_list"].append(valao) # valao is 0
                    self.basis["cgf_atom_list"].append(iat)
                    self.basis["cgf_type_id_list"].append(1) # 1=s
                    self.basis["cgf_primitive_count_list"].append(nprim_total)
                    self.basis["cgf_energy_list"].append(level)
                    ibf += 1
                    
                    # Add Primitives (Diffuse part)
                    all_alphas = []
                    all_coeffs = []
                    prim_start_idx_for_this_cgf = ipr # idum=ipr+1

                    for p in range(current_nprim_or_R): # current_nprim = thisprimR
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
                    self.basis["ao_valence_flag_list"].append(valao) # valao is 0
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
                        
                        # Add AO (5 AOs for 6 CGFs)
                        if j_type != 5: # Skip for dxx (j_type=5)
                            ao_type_id = j_type - 1 # (AO IDs: 5,6,7,8,9)
                            self.basis["ao_valence_flag_list"].append(valao)
                            self.basis["ao_atom_list"].append(iat)
                            self.basis["ao_type_id_list"].append(ao_type_id) 
                            self.basis["ao_energy_list"].append(level)
                            self.basis["ao_zeta_list"].append(zeta)
                            self.basis["ao_shell_list"].append(ish)
                            iao += 1

                # --- f functions ---
                elif l == 3:
                    valao_f = 1 
                    
                    alpha, coeff = self.slater2gauss(current_nprim_or_R, npq, l, zeta, param, need_normalization=True)
                    min_alpha_for_shell = np.min(alpha)
                    
                    for j_type in range(11, 21): # 11..20 (10 CGFs)
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

                        # Add AO (7 AOs for 10 CGFs)
                        if j_type > 13: # Skip for fxxx, fyyy, fzzz (11,12,13)
                            ao_type_id = j_type - 3 # (AO IDs: 11..17)
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
        
        # --- Final Sanity Checks ---
        if n_cgf_total != ibf:
            print(f"Warning: CGF count mismatch. Calculated={n_cgf_total}, Found={ibf}")
        if n_ao_total != iao:
            print(f"Warning: AO count mismatch. Calculated={n_ao_total}, Found={iao}")
        if n_shell_total != ish:
            print(f"Warning: Shell count mismatch. Calculated={n_shell_total}, Found={ish}")
        
        
        return


    def get_number_of_primitives(self, element, angmn, pqn, is_valence):
        
        atom_num = element + 1 # Convert 0-indexed to 1-indexed atomic number
        
        
        if atom_num <= 2: # H, He
            if is_valence:
                n = 3 # STO-3G 
            else:
                n = 2 # STO-2G
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
                print("Error in get_number_of_primitives: angmn > 3")
                raise ValueError

        return n # STO-nG


    def check_transition_metals(self):
        is_tm_list = []
        is_g11_element_list = []
        
        g11_z_numbers = {29, 47, 79, 111}

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
        NOTE: This method is not called by _set_basis in your example.
        You may be using it elsewhere.
        """
        self.valence_orbitals_list = [] # If valence orbital, True
        for i in range(len(self.element_list)):
            atn = self.element_list[i]
            
            seen_ang_mom = set()
            for shell_idx in range(self.nshells_list[i]):
                ang = self.ang_shells_list[i][shell_idx]
                if ang not in seen_ang_mom:
                    self.valence_orbitals_list.append(True) # Mark as valence
                    seen_ang_mom.add(ang)
                else:
                    self.valence_orbitals_list.append(False) # Mark as diffuse
                        
        return self.valence_orbitals_list
    
    def slater2gauss(self, nprim, pqn, angmn, slater_exp, param, need_normalization=True):
        # param is already available via self.param
        # No need to pass it as an argument, but we will keep 
        # the signature the same as your code.
        
        alpha = np.zeros(nprim)
        coeff = np.zeros(nprim)
        
        if pqn < angmn + 1:
            print("Error in slater2gauss: pqn < angmn + 1")
            raise ValueError
        
        if slater_exp <= 0.0:
            print("Error in slater2gauss: non-positive slater exponent")
            raise ValueError
        
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
            print("Error in slater2gauss: angmn > 4")
            raise ValueError
        
        slater_exp_sq = slater_exp ** 2.0
        
        if nprim == 1:
            alpha[0] = param.pAlpha1[tmp_num] * slater_exp_sq
            coeff[0] = 1.0
        elif nprim == 2:
            alpha = param.pAlpha2[tmp_num, :] * slater_exp_sq
            coeff = param.pCoeff2[tmp_num, :]
        elif nprim == 3:
            alpha = param.pAlpha3[tmp_num, :] * slater_exp_sq
            coeff = param.pCoeff3[tmp_num, :]
        elif nprim == 4:
            alpha = param.pAlpha4[tmp_num, :] * slater_exp_sq
            coeff = param.pCoeff4[tmp_num, :]
        elif nprim == 5:
            alpha = param.pAlpha5[tmp_num, :] * slater_exp_sq
            coeff = param.pCoeff5[tmp_num, :]
        elif nprim == 6:
            if pqn == 6:
                if angmn == 0:
                    alpha = param.pAlpha6s[:] * slater_exp_sq
                    coeff = param.pCoeff6s[:]
                elif angmn == 1:
                    alpha = param.pAlpha6p[:] * slater_exp_sq
                    coeff = param.pCoeff6p[:]
                else:
                    print("Error in slater2gauss: invalid angmn for pqn=6 with nprim=6")
                    raise ValueError
            else:
                alpha = param.pAlpha6[tmp_num, :] * slater_exp_sq
                coeff = param.pCoeff6[tmp_num, :]
        else:
            print("Error in slater2gauss: invalid number of primitives")
            raise ValueError
        

        if need_normalization:
            top = 2.0 / np.pi
            coeff = coeff * (top * alpha) ** 0.75 * np.sqrt(4.0 * alpha) ** angmn / np.sqrt(dfactorial(angmn + 1))
                
        return alpha, coeff

    
    def calc_overlap_int_for_diffuse_func(self, angmn, alpha_A, alpha_B, coeff_A, coeff_B):
        """ 
        Calculates the overlap <A|B> for two CGFs on the same center.
        """
        overlap_int = 0.0
        
        n_prim_A = len(coeff_A)
        n_prim_B = len(coeff_B)

        if n_prim_A == 0 or n_prim_B == 0:
            # This would happen if the diffuse shell is processed before
            # the valence shell, and state is not managed correctly.
            print("Warning: calc_overlap_int_for_diffuse_func called with empty primitive list.")
            return 0.0

        for i in range(n_prim_A):
            for j in range(n_prim_B):
                ab = 1.0 / (alpha_A[i] + alpha_B[j])
                s00 = (np.pi * ab) ** 1.5
                
                sss = 0.0
                if angmn == 0: # s orbital - s orbital
                    sss = s00
                elif angmn == 1: # p_a orbital - p_a orbital
                    ab05 = ab * 0.5 
                    sss = s00 * ab05
                # Add angmn==2 for d-d overlap if needed, etc.
                # S(d,d) = s00 * (ab05*ab05*3) for <xx|xx>
                # S(d,d) = s00 * (ab05*ab05) for <xy|xy>
                
                overlap_int += sss * coeff_A[i] * coeff_B[j]           
                
        return overlap_int

