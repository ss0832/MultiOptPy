import numpy as np
from scipy.spatial.distance import cdist
from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import D3Parameters, D2_C6_coeff_lib, D2_VDW_radii_lib
from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.ModelHessian.calc_params import torsion2, bend2

class FischerD3ApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation with Dynamic D3 dispersion correction.
        Robust against linear molecules and singularities.
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        
        # D3 dispersion correction parameters
        self.d3_params = D3Parameters()
        self.cart_hess = None
        
        # Grimme D3 scaling factors for CN calculation
        self.k1 = 16.0
        self.k2 = 4.0 / 3.0

        # Expanded Reference Coordination Numbers (Default averages)
        # Covers H through Xe roughly. Values are heuristic/typical valencies or coordination.
        self.ref_cn_map = {
            # Period 1
            'H': 1, 'He': 0,
            # Period 2
            'Li': 4, 'Be': 4, 'B': 3, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Ne': 0,
            # Period 3
            'Na': 6, 'Mg': 6, 'Al': 6, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 1, 'Ar': 0,
            # Period 4 (Transition Metals set to ~6-12 depending on typical packing/complexes)
            'K': 8, 'Ca': 6, 
            'Sc': 12, 'Ti': 12, 'V': 12, 'Cr': 6, 'Mn': 6, 'Fe': 6, 'Co': 6, 'Ni': 4, 'Cu': 4, 'Zn': 4,
            'Ga': 4, 'Ge': 4, 'As': 3, 'Se': 2, 'Br': 1, 'Kr': 0,
            # Period 5
            'Rb': 8, 'Sr': 6,
            'Y': 12, 'Zr': 12, 'Nb': 12, 'Mo': 6, 'Tc': 6, 'Ru': 6, 'Rh': 6, 'Pd': 4, 'Ag': 4, 'Cd': 4,
            'In': 6, 'Sn': 4, 'Sb': 3, 'Te': 2, 'I': 1, 'Xe': 0
        }

    def calc_coordination_numbers(self, coord, element_list):
        """Calculate fractional coordination numbers (CN) for Dynamic D3."""
        # Pre-fetch covalent radii
        cov_r = np.array([covalent_radii_lib(e) for e in element_list])
        
        # Vectorized distance matrix
        r_mat = cdist(coord, coord)
        np.fill_diagonal(r_mat, np.inf)
        
        # rcov_sum[i, j] = cov_r[i] + cov_r[j]
        rcov_sum = cov_r[:, None] + cov_r[None, :]
        
        # D3 CN formula
        term = -self.k1 * (self.k2 * (r_mat / rcov_sum) - 1.0)
        term = np.clip(term, -100, 100) # Prevent overflow
        cutoff_func = 1.0 / (1.0 + np.exp(term))
        
        return np.sum(cutoff_func, axis=1)

    # --- Fischer Terms (Robust) ---

    def calc_bond_force_const(self, r_ab, r_ab_cov):
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        val = r_ab_cov * r_ac_cov
        if abs(val) < 1e-10: return 0.0
        return 0.089 + 0.11 / (val)**(-0.42) * np.exp(-0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov))
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        val = r_ab * r_ab_cov
        if abs(val) < 1e-10: return 0.0
        return 0.0015 + 14.0 * max(bond_sum, 0)**0.57 / (val)**4.0 * np.exp(-2.85 * (r_ab - r_ab_cov))

    def _add_block(self, i, j, block):
        start_i, end_i = 3*i, 3*i+3
        start_j, end_j = 3*j, 3*j+3
        self.cart_hess[start_i:end_i, start_j:end_j] += block

    def fischer_bond(self, coord, element_list, bond_indices):
        for idx in bond_indices:
            i, j = idx
            r_vec = coord[i] - coord[j]
            r_ij = np.linalg.norm(r_vec)
            if r_ij < 0.1: continue

            r_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            fc = self.calc_bond_force_const(r_ij, r_cov)
            
            # Use safe unit vector calculation
            u = r_vec / r_ij
            block = fc * np.outer(u, u)
            
            self._add_block(i, i, block)
            self._add_block(j, j, block)
            self._add_block(i, j, -block)
            self._add_block(j, i, -block)

    def fischer_angle(self, coord, element_list, angle_indices):
        for idx in angle_indices:
            i, j, k = idx
            r_ij_vec = coord[i] - coord[j]
            r_jk_vec = coord[k] - coord[j] # Center is j
            
            r_ij = np.linalg.norm(r_ij_vec)
            r_jk = np.linalg.norm(r_jk_vec)
            
            if r_ij < 0.1 or r_jk < 0.1: 
                continue

            # Check for linearity (180 deg) or overlap (0 deg)
            cos_theta = np.dot(r_ij_vec, r_jk_vec) / (r_ij * r_jk)
            if abs(cos_theta) > 0.9999:
                 continue

            r_cov_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_cov_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            fc = self.calc_bend_force_const(r_ij, r_jk, r_cov_ij, r_cov_jk)
            
            try:
                # bend2 expects: [atom_end1, atom_center, atom_end2]
                t_xyz = np.array([coord[i], coord[j], coord[k]])
                _, b_vec = bend2(t_xyz)
                
                atoms = [i, j, k]
                for m_idx, m_atom in enumerate(atoms):
                    for n_idx, n_atom in enumerate(atoms):
                        block = fc * np.outer(b_vec[m_idx], b_vec[n_idx])
                        self._add_block(m_atom, n_atom, block)
            except Exception:
                continue

    def fischer_dihedral(self, coord, element_list, dihedral_indices, bond_mat):
        neighbor_counts = np.sum(bond_mat, axis=1)

        for idx in dihedral_indices:
            i, j, k, l = idx
            
            # Check linearity of the axis bond j-k and its connections
            vec_ji = coord[i] - coord[j]
            vec_jk = coord[k] - coord[j]
            vec_kl = coord[l] - coord[k]
            
            n_ji, n_jk, n_kl = np.linalg.norm(vec_ji), np.linalg.norm(vec_jk), np.linalg.norm(vec_kl)
            if min(n_ji, n_jk, n_kl) < 0.1: continue

            # Cosine check for linearity: i-j-k and j-k-l
            cos1 = np.dot(vec_ji, vec_jk) / (n_ji * n_jk)
            cos2 = np.dot(-vec_jk, vec_kl) / (n_jk * n_kl)
            
            # Use sin^2 to dampen force constant as it approaches linearity
            sin2_1 = 1.0 - min(cos1**2, 1.0)
            sin2_2 = 1.0 - min(cos2**2, 1.0)
            
            if sin2_1 < 1e-3 or sin2_2 < 1e-3:
                continue

            # Damping factor
            scaling = sin2_1 * sin2_2
            
            bond_sum = neighbor_counts[j] + neighbor_counts[k] - 2
            r_cov_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            fc = self.calc_dihedral_force_const(n_jk, r_cov_jk, bond_sum)
            fc *= scaling

            try:
                t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
                _, b_vec = torsion2(t_xyz)
                
                if not np.all(np.isfinite(b_vec)):
                    continue

                atoms = [i, j, k, l]
                for m_idx, m_atom in enumerate(atoms):
                    for n_idx, n_atom in enumerate(atoms):
                        block = fc * np.outer(b_vec[m_idx], b_vec[n_idx])
                        self._add_block(m_atom, n_atom, block)
            except Exception:
                continue

    def main(self, coord, element_list, cart_gradient):
        print("Generating Hessian using Fischer model with Dynamic D3 correction (Robust)...")
        
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # 1. Topology
        BC = BondConnectivity()
        bond_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(bond_mat)
        angle_indices = BC.angle_connect_table(bond_mat)
        dihedral_indices = BC.dihedral_angle_connect_table(bond_mat)
        
        # 2. Dynamic D3 Parameters (Coordination Numbers)
        cn_list = self.calc_coordination_numbers(coord, element_list)
        
        # 3. Fischer Contributions (with strict linearity checks)
        self.fischer_bond(coord, element_list, bond_indices)
        self.fischer_angle(coord, element_list, angle_indices)
        self.fischer_dihedral(coord, element_list, dihedral_indices, bond_mat)
        
        # 4. Dynamic D3 Dispersion (Vectorized)
        mask = np.tril(np.ones((n_atoms, n_atoms), dtype=bool), k=-1) & ~bond_mat
        
        diff_tensor = coord[:, None, :] - coord[None, :, :]  # (N, N, 3)
        dist_matrix = np.linalg.norm(diff_tensor, axis=-1)
        
        mask &= (dist_matrix > 0.1)
        i_idx, j_idx = np.where(mask)
        
        if len(i_idx) > 0:
            # --- Pre-compute atomic parameters for all atoms (O(N)) ---
            c6_atoms = np.array([D2_C6_coeff_lib(e) for e in element_list])
            r4r2_atoms = np.array([self.d3_params.get_r4r2(e) for e in element_list])
            ref_cn_atoms = np.array([self.ref_cn_map.get(e, 4) for e in element_list])
            
            # Safe VDW radii retrieval
            vdw_atoms = np.zeros(n_atoms)
            for k, e in enumerate(element_list):
                try: vdw_atoms[k] = D2_VDW_radii_lib(e)
                except: vdw_atoms[k] = covalent_radii_lib(e) * 1.5

            # --- Gather Pair Data (O(M)) ---
            r_vecs = diff_tensor[i_idx, j_idx]  # Vector r_ij (M, 3)
            r_ijs = dist_matrix[i_idx, j_idx]   # Distance scalar (M,)
            
            # Dynamic C6 Scaling
            cn_i = cn_list[i_idx]
            cn_j = cn_list[j_idx]
            scale_i = np.clip(1.0 - 0.05 * (cn_i - ref_cn_atoms[i_idx]), 0.75, 1.25)
            scale_j = np.clip(1.0 - 0.05 * (cn_j - ref_cn_atoms[j_idx]), 0.75, 1.25)
            
            c6_ijs = np.sqrt((c6_atoms[i_idx] * scale_i) * (c6_atoms[j_idx] * scale_j))
            
            # C8 and R0
            c8_ijs = 3.0 * c6_ijs * np.sqrt(r4r2_atoms[i_idx] * r4r2_atoms[j_idx])
            r0s = vdw_atoms[i_idx] + vdw_atoms[j_idx]
            
            # --- Vectorized Gradient & Hessian Calculation ---
            a1 = self.d3_params.a1
            a2_6 = self.d3_params.a2
            a2_8 = self.d3_params.a2 + 2.0
            
            # Damping terms
            denom6 = r_ijs**6 + (a1 * r0s + a2_6)**6
            denom8 = r_ijs**8 + (a1 * r0s + a2_8)**8
            
            f_damp6 = r_ijs**6 / denom6
            f_damp8 = r_ijs**8 / denom8
            
            df_damp6 = 6 * r_ijs**5 / denom6 - 6 * r_ijs**12 / denom6**2
            df_damp8 = 8 * r_ijs**7 / denom8 - 8 * r_ijs**16 / denom8**2
            
            # Gradients (Forces)
            g6 = -self.d3_params.s6 * c6_ijs * ((-6.0/r_ijs**7)*f_damp6 + (1.0/r_ijs**6)*df_damp6)
            g8 = -self.d3_params.s8 * c8_ijs * ((-8.0/r_ijs**9)*f_damp8 + (1.0/r_ijs**8)*df_damp8)
            
            # Second Derivatives (Projection coefficients)
            h6_proj = self.d3_params.s6 * c6_ijs / r_ijs**8 * (42.0*f_damp6 - r_ijs*df_damp6)
            h8_proj = self.d3_params.s8 * c8_ijs / r_ijs**10 * (72.0*f_damp8 - r_ijs*df_damp8)
            
            h_proj_vals = h6_proj + h8_proj
            h_perp_vals = (g6 + g8) / r_ijs
            
            # Construct Hessian Blocks (M, 3, 3)
            # P = u * u.T
            unit_vecs = r_vecs / r_ijs[:, None]
            proj_ops = np.einsum('bi,bj->bij', unit_vecs, unit_vecs) # (M, 3, 3)
            identity = np.eye(3)[None, :, :]
            
            hess_blocks = h_proj_vals[:, None, None] * proj_ops + h_perp_vals[:, None, None] * (identity - proj_ops)
            
            # --- Accumulate into Cartesian Hessian ---
            for k, (idx_i, idx_j) in enumerate(zip(i_idx, j_idx)):
                block = hess_blocks[k]
                self._add_block(idx_i, idx_i, block)
                self._add_block(idx_j, idx_j, block)
                self._add_block(idx_i, idx_j, -block)
                self._add_block(idx_j, idx_i, -block)
        
        # 5. Symmetrize & Clean
        self.cart_hess = (self.cart_hess + self.cart_hess.T) / 2.0
        
        # Project out TR/ROT modes
        try:
             hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                 self.cart_hess, element_list, coord
             )
             if not np.all(np.isfinite(hess_proj)):
                 print("Warning: Projection produced NaNs. Using unprojected Hessian.")
                 hess_proj = self.cart_hess
        except Exception as e:
             print(f"Warning: Projection failed ({e}). Using unprojected Hessian.")
             hess_proj = self.cart_hess
             
        if not np.all(np.isfinite(hess_proj)):
             print("CRITICAL: Hessian contains NaNs after generation. Resetting to Identity.")
             hess_proj = np.eye(n_atoms*3)

        return hess_proj