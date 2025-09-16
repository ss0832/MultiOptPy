import numpy as np

from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import GFN0Parameters


class GFN0XTBApproxHessian:
    """GFN0-xTB approximate Hessian with special handling for cyano groups"""
    def __init__(self):
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.params = GFN0Parameters()
        self.bond_factor = 1.3  # Bond detection threshold factor
    
    def detect_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix and related data"""
        n_atoms = len(coord)
        dist_mat = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coord[i] - coord[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
        
        # Bond connectivity based on covalent radii
        bond_mat = np.zeros((n_atoms, n_atoms), dtype=bool)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_i = self.params.get_radius(element_list[i])
                r_j = self.params.get_radius(element_list[j])
                
                # Use covalent radii to determine bonding
                r_cov = r_i + r_j
                
                if dist_mat[i, j] < r_cov * self.bond_factor:
                    bond_mat[i, j] = bond_mat[j, i] = True
        
        return bond_mat, dist_mat
    
    def analyze_molecular_structure(self, coord, element_list):
        """
        Analyze molecular structure to identify bond types, cyano groups, and hybridization
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            
        Returns:
            topology: dictionary with molecular structure information
        """
        # Get bond connectivity
        bond_mat, dist_mat = self.detect_bond_connectivity(coord, element_list)
        
        # Build list of bonds
        n_atoms = len(coord)
        bonds = []
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bond_mat[i, j]:
                    bonds.append((i, j))
        
        # Count neighbors for each atom
        neighbor_counts = np.sum(bond_mat, axis=1)
        
        # Determine hybridization based on neighbor count and element type
        hybridization = {}
        
        for i in range(n_atoms):
            elem = element_list[i]
            n_neighbors = neighbor_counts[i]
            
            if elem == 'C':
                if n_neighbors == 4:
                    hybridization[i] = 'sp3'
                elif n_neighbors == 3:
                    hybridization[i] = 'sp2'
                elif n_neighbors == 2:
                    # Check angle to decide between sp and sp2
                    neighbors = [j for j in range(n_atoms) if bond_mat[i, j]]
                    if len(neighbors) == 2:
                        v1 = coord[neighbors[0]] - coord[i]
                        v2 = coord[neighbors[1]] - coord[i]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        hybridization[i] = 'sp' if angle > 160 else 'sp2'
                    else:
                        hybridization[i] = 'sp2'  # Default
                else:
                    hybridization[i] = 'sp3'  # Default
            
            elif elem == 'N':
                if n_neighbors == 3:
                    hybridization[i] = 'sp2'
                elif n_neighbors == 2:
                    hybridization[i] = 'sp2'  # Most common
                elif n_neighbors == 1:
                    # Check bond length to determine if it's a triple bond (CN)
                    neighbors = [j for j in range(n_atoms) if bond_mat[i, j]]
                    if len(neighbors) == 1 and element_list[neighbors[0]] == 'C':
                        bond_length = dist_mat[i, neighbors[0]]
                        ref_cn_triple = self.params.get_bond_length('C', 'N', 'triple') / self.bohr2angstroms
                        if abs(bond_length - ref_cn_triple) < 0.15:
                            hybridization[i] = 'sp'
                        else:
                            hybridization[i] = 'sp2'
                    else:
                        hybridization[i] = 'sp2'
                else:
                    hybridization[i] = 'sp3'
            
            elif elem == 'O':
                if n_neighbors == 1:
                    # Check if it's a carbonyl
                    neighbors = [j for j in range(n_atoms) if bond_mat[i, j]]
                    if element_list[neighbors[0]] == 'C':
                        bond_length = dist_mat[i, neighbors[0]]
                        ref_co_double = self.params.get_bond_length('C', 'O', 'double') / self.bohr2angstroms
                        if abs(bond_length - ref_co_double) < 0.15:
                            hybridization[i] = 'sp2'
                        else:
                            hybridization[i] = 'sp3'
                    else:
                        hybridization[i] = 'sp3'
                else:
                    hybridization[i] = 'sp3'
            
            else:
                # Default hybridization for other elements
                hybridization[i] = 'sp3'
        
        # Identify bond types
        bond_types = {}
        
        for i, j in bonds:
            # Determine bond type based on elements and hybridization
            hyb_i = hybridization.get(i, 'sp3')
            hyb_j = hybridization.get(j, 'sp3')
            
            # Default is single bond
            bond_type = 'single'
            
            # Special case: CN triple bond
            if ((element_list[i] == 'C' and element_list[j] == 'N') or 
                (element_list[i] == 'N' and element_list[j] == 'C')):
                if ((hyb_i == 'sp' and hyb_j == 'sp') or 
                    (hyb_i == 'sp' and neighbor_counts[i] == 2 and neighbor_counts[j] == 1) or 
                    (hyb_j == 'sp' and neighbor_counts[j] == 2 and neighbor_counts[i] == 1)):
                    # This looks like a cyano group
                    bond_type = 'triple'
            
            # Carbon-carbon bonds
            elif element_list[i] == 'C' and element_list[j] == 'C':
                if hyb_i == 'sp' and hyb_j == 'sp':
                    bond_type = 'triple'
                elif hyb_i == 'sp2' and hyb_j == 'sp2':
                    # Could be double bond or aromatic
                    bond_type = 'double'  # Simplified
            
            # Carbon-oxygen bonds
            elif ((element_list[i] == 'C' and element_list[j] == 'O') or 
                  (element_list[i] == 'O' and element_list[j] == 'C')):
                if ((hyb_i == 'sp2' and hyb_j == 'sp2') or 
                    (hyb_i == 'sp2' and neighbor_counts[j] == 1) or 
                    (hyb_j == 'sp2' and neighbor_counts[i] == 1)):
                    # Carbonyl group
                    bond_type = 'double'
            
            bond_types[(i, j)] = bond_types[(j, i)] = bond_type
        
        # Identify cyano groups
        cyano_groups = []
        
        for i in range(n_atoms):
            if element_list[i] == 'C' and hybridization.get(i, '') == 'sp':
                n_partners = [j for j in range(n_atoms) if bond_mat[i, j] and element_list[j] == 'N']
                other_partners = [j for j in range(n_atoms) if bond_mat[i, j] and element_list[j] != 'N']
                
                if n_partners and len(other_partners) == 1:
                    n_idx = n_partners[0]
                    if bond_types.get((i, n_idx)) == 'triple':
                        cyano_groups.append((i, n_idx))
        
        # Build lists of angles and dihedrals
        angles = []
        dihedrals = []
        
        # Extract angles
        for j in range(n_atoms):
            bonded_to_j = [i for i in range(n_atoms) if bond_mat[i, j] and i != j]
            for i in bonded_to_j:
                for k in bonded_to_j:
                    if i < k:  # Avoid duplicates
                        angles.append((i, j, k))
        
        # Extract dihedrals
        for j, k in bonds:
            bonded_to_j = [i for i in range(n_atoms) if bond_mat[i, j] and i != j and i != k]
            bonded_to_k = [l for l in range(n_atoms) if bond_mat[k, l] and l != k and l != j]
            
            for i in bonded_to_j:
                for l in bonded_to_k:
                    if i != l:  # Avoid improper dihedrals here
                        dihedrals.append((i, j, k, l))
        
        # Collect structure information
        topology = {
            'bonds': bonds,
            'bond_types': bond_types,
            'angles': angles,
            'dihedrals': dihedrals,
            'bond_mat': bond_mat,
            'dist_mat': dist_mat,
            'hybridization': hybridization,
            'cyano_groups': cyano_groups,
            'neighbor_counts': neighbor_counts
        }
        
        return topology
    
    def compute_partial_charges(self, coord, element_list, topology):
        """
        Calculate partial charges using GFN0-xTB electronegativity equilibration
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular structure information
            
        Returns:
            charges: array of partial charges
        """
        n_atoms = len(coord)
        bond_mat = topology['bond_mat']
        dist_mat = topology['dist_mat']
        
        # Initialize charges
        charges = np.zeros(n_atoms)
        
        # Step 1: Initial charge distribution based on electronegativity difference
        for i, j in topology['bonds']:
            en_i = self.params.get_en(element_list[i])
            en_j = self.params.get_en(element_list[j])
            
            # EN difference determines charge flow direction
            en_diff = en_j - en_i
            
            # Basic charge transfer based on EN difference
            transfer = 0.05 * np.tanh(0.2 * en_diff)
            
            # Apply charge transfer
            charges[i] += transfer
            charges[j] -= transfer
        
        # Step 2: Special treatment for cyano groups
        for c_idx, n_idx in topology['cyano_groups']:
            # Cyano groups have strong polarization
            charges[n_idx] -= 0.3  # Negative charge on N
            charges[c_idx] += 0.3  # Positive charge on C
        
        # Step 3: Normalize charges to ensure neutrality
        charges -= np.mean(charges)
        
        return charges
    
    def gfn0_bond_hessian(self, coord, element_list, topology):
        """
        Calculate bond stretching contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular structure information
        """
        bonds = topology['bonds']
        bond_types = topology['bond_types']
        
        for i, j in bonds:
            r_vec = coord[j] - coord[i]
            r_ij = np.linalg.norm(r_vec)
            
            # Get bond type
            bond_type = bond_types.get((i, j), 'single')
            
            # Get force constant
            force_const = self.params.get_bond_force_constant(
                element_list[i], element_list[j], bond_type)
            
            # Get reference bond length (convert to Bohr)
            r0 = self.params.get_bond_length(
                element_list[i], element_list[j], bond_type) / self.bohr2angstroms
            
            # Calculate unit vector and projection operator
            if r_ij > 1e-10:
                u_ij = r_vec / r_ij
                proj_op = np.outer(u_ij, u_ij)
            else:
                # Avoid division by zero
                proj_op = np.eye(3) / 3.0
            
            # Add to Cartesian Hessian
            h_diag = force_const * proj_op
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += h_diag[n, m]
                    self.cart_hess[3*j+n, 3*j+m] += h_diag[n, m]
                    self.cart_hess[3*i+n, 3*j+m] -= h_diag[n, m]
                    self.cart_hess[3*j+n, 3*i+m] -= h_diag[n, m]
    
    def gfn0_angle_hessian(self, coord, element_list, topology):
        """
        Calculate angle bending contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular structure information
        """
        angles = topology['angles']
        hybridization = topology['hybridization']
        cyano_groups = topology['cyano_groups']
        
        # Create lookup for cyano groups
        cyano_carbons = [c for c, _ in cyano_groups]
        cyano_nitrogens = [n for _, n in cyano_groups]
        
        for i, j, k in angles:
            # Calculate vectors and distances
            r_ji = coord[i] - coord[j]
            r_jk = coord[k] - coord[j]
            r_ji_len = np.linalg.norm(r_ji)
            r_jk_len = np.linalg.norm(r_jk)
            
            # Skip if atoms are too close
            if r_ji_len < 1e-10 or r_jk_len < 1e-10:
                continue
            
            # Special handling for X-C≡N angles
            is_cyano_angle = False
            
            if j in cyano_carbons:
                n_idx = None
                for c, n in cyano_groups:
                    if c == j:
                        n_idx = n
                        break
                        
                if n_idx is not None and (i == n_idx or k == n_idx):
                    is_cyano_angle = True
                    force_const = self.params.CNParams['kBend']
                    theta0 = np.pi  # 180 degrees (linear)
            
            # Regular angle if not cyano
            if not is_cyano_angle:
                # Get hybridization of central atom
                hyb_j = hybridization.get(j, 'sp3')
                
                # Get natural angle based on hybridization
                theta0 = self.params.naturalAngles.get(hyb_j, self.params.naturalAngles['sp3'])
                
                # Base force constant
                force_const = self.params.kAngleBase
                
                # Scale force constant based on central atom
                if element_list[j] == 'C':
                    force_const *= 1.0
                elif element_list[j] == 'N':
                    force_const *= 0.9
                elif element_list[j] == 'O':
                    force_const *= 0.8
                else:
                    force_const *= 0.7
            
            # Calculate angle
            cos_theta = np.dot(r_ji, r_jk) / (r_ji_len * r_jk_len)
            cos_theta = np.clip(cos_theta, -0.999999, 0.999999)  # Avoid numerical issues
            theta = np.arccos(cos_theta)
            
            # Calculate derivatives
            sin_theta = np.sin(theta)
            if sin_theta < 1e-10:
                # Handle nearly linear case
                continue
                
            # Simplified derivatives for angle
            d_i = (np.cross(np.cross(r_ji, r_jk), r_ji)) / (r_ji_len**2 * r_jk_len * sin_theta)
            d_k = (np.cross(np.cross(r_jk, r_ji), r_jk)) / (r_ji_len * r_jk_len**2 * sin_theta)
            d_j = -d_i - d_k
            
            # Scale derivatives by force constant
            d_i *= np.sqrt(force_const)
            d_j *= np.sqrt(force_const)
            d_k *= np.sqrt(force_const)
            
            # Add to Hessian
            for a in range(3):
                for b in range(3):
                    # i-i block
                    self.cart_hess[3*i+a, 3*i+b] += d_i[a] * d_i[b]
                    
                    # j-j block
                    self.cart_hess[3*j+a, 3*j+b] += d_j[a] * d_j[b]
                    
                    # k-k block
                    self.cart_hess[3*k+a, 3*k+b] += d_k[a] * d_k[b]
                    
                    # Cross terms
                    self.cart_hess[3*i+a, 3*j+b] += d_i[a] * d_j[b]
                    self.cart_hess[3*i+a, 3*k+b] += d_i[a] * d_k[b]
                    self.cart_hess[3*j+a, 3*i+b] += d_j[a] * d_i[b]
                    self.cart_hess[3*j+a, 3*k+b] += d_j[a] * d_k[b]
                    self.cart_hess[3*k+a, 3*i+b] += d_k[a] * d_i[b]
                    self.cart_hess[3*k+a, 3*j+b] += d_k[a] * d_j[b]
    
    def gfn0_torsion_hessian(self, coord, element_list, topology):
        """
        Calculate torsion (dihedral) contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular structure information
        """
        dihedrals = topology['dihedrals']
        bond_types = topology['bond_types']
        cyano_groups = topology['cyano_groups']
        # Create lookup for cyano groups
        cyano_bonds = set()
        for c, n in cyano_groups:
            cyano_bonds.add((c, n))
            cyano_bonds.add((n, c))
        
        for i, j, k, l in dihedrals:
            # Calculate vectors along the bonds
            r_ij = coord[j] - coord[i]
            r_jk = coord[k] - coord[j]
            r_kl = coord[l] - coord[k]
            
            # Calculate cross products for the dihedral
            n1 = np.cross(r_ij, r_jk)
            n2 = np.cross(r_jk, r_kl)
            
            # Skip if any of the cross products are too small
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            r_jk_norm = np.linalg.norm(r_jk)
            
            if n1_norm < 1e-10 or n2_norm < 1e-10 or r_jk_norm < 1e-10:
                continue
            
            # Calculate the dihedral angle
            cos_phi = np.dot(n1, n2) / (n1_norm * n2_norm)
            cos_phi = np.clip(cos_phi, -0.999999, 0.999999)  # Avoid numerical issues
            
            sin_phi = np.dot(np.cross(n1, n2), r_jk) / (n1_norm * n2_norm * r_jk_norm)
            phi = np.arctan2(sin_phi, cos_phi)
            
            # Get torsion parameters
            # Check if this is a cyano torsion
            if (j, k) in cyano_bonds or (k, j) in cyano_bonds:
                # Cyano group torsion - very small barrier
                V2 = V3 = self.params.CNParams['kTorsion']
            else:
                # Normal torsion
                bond_type = bond_types.get((j, k), 'single')
                
                # Adjust based on bond type
                if bond_type == 'triple':
                    # Triple bonds have near-zero torsion barriers
                    V2 = V3 = 0.001
                elif bond_type == 'double':
                    # Double bonds have significant V2 (two-fold) term
                    V2 = self.params.V2Base * 2.0
                    V3 = self.params.V3Base * 0.5
                elif bond_type == 'aromatic':
                    # Aromatic bonds have mixed character
                    V2 = self.params.V2Base * 1.5
                    V3 = self.params.V3Base
                else:
                    # Single bonds have V3 (three-fold) term
                    V2 = self.params.V2Base * 0.5
                    V3 = self.params.V3Base * 1.5
            
            # Calculate second derivatives
            # V = V2/2 * (1-cos(2*phi)) + V3/2 * (1+cos(3*phi))
            # For second derivatives, we need:
            # d²V/dphi² = 2*V2*cos(2*phi) - 4.5*V3*cos(3*phi)
            d2V = 2.0 * V2 * np.cos(2*phi) - 4.5 * V3 * np.cos(3*phi)
            
            # Calculate derivatives of phi w.r.t. Cartesian coordinates
            # (simplified approach)
            # Calculate unit vectors
            e_ij = r_ij / np.linalg.norm(r_ij) if np.linalg.norm(r_ij) > 1e-10 else np.zeros(3)
            e_jk = r_jk / r_jk_norm
            e_kl = r_kl / np.linalg.norm(r_kl) if np.linalg.norm(r_kl) > 1e-10 else np.zeros(3)
            
            # Simplified derivatives calculation
            n1_u = n1 / n1_norm if n1_norm > 1e-10 else np.zeros(3)
            n2_u = n2 / n2_norm if n2_norm > 1e-10 else np.zeros(3)
            
            # Calculate derivatives (this is a simplified approach)
            # We calculate d(phi)/dr for each atom
            g_i = np.cross(e_ij, n1_u) / (np.linalg.norm(r_ij) * sin_phi) if sin_phi > 1e-10 else np.zeros(3)
            g_l = -np.cross(e_kl, n2_u) / (np.linalg.norm(r_kl) * sin_phi) if sin_phi > 1e-10 else np.zeros(3)
            
            # Use conservation of angular momentum to get the middle terms
            # These are simplified and not analytically perfect
            g_j = -g_i - (r_jk_norm / np.linalg.norm(r_ij)) * g_i
            g_k = -g_l - (r_jk_norm / np.linalg.norm(r_kl)) * g_l
            
            # Scale derivatives by second derivative of potential
            g_i *= np.sqrt(abs(d2V))
            g_j *= np.sqrt(abs(d2V))
            g_k *= np.sqrt(abs(d2V))
            g_l *= np.sqrt(abs(d2V))
            
            # Add to Hessian
            atoms = [i, j, k, l]
            derivatives = [g_i, g_j, g_k, g_l]
            
            for a, g_a in enumerate(derivatives):
                for b, g_b in enumerate(derivatives):
                    atom_a = atoms[a]
                    atom_b = atoms[b]
                    for x in range(3):
                        for y in range(3):
                            self.cart_hess[3*atom_a+x, 3*atom_b+y] += g_a[x] * g_b[y]
    
    def gfn0_nonbonded_hessian(self, coord, element_list, topology):
        """
        Calculate non-bonded interaction contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular structure information
        """
        n_atoms = len(coord)
        bond_mat = topology['bond_mat']
        charges = self.compute_partial_charges(coord, element_list, topology)
        
        # Simplified non-bonded model for Hessian approximation
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Skip bonded atoms and 1-3 interactions
                if bond_mat[i, j]:
                    continue
                
                # Skip 1-3 interactions (atoms sharing a bonded neighbor)
                has_common_neighbor = False
                for k in range(n_atoms):
                    if bond_mat[i, k] and bond_mat[j, k]:
                        has_common_neighbor = True
                        break
                
                if has_common_neighbor:
                    continue
                
                # Calculate distance vector and magnitude
                r_vec = coord[j] - coord[i]
                r_ij = np.linalg.norm(r_vec)
                
                if r_ij < 0.5:  # Avoid too small distances
                    continue
                    
                # Calculate unit vector and projection operator
                u_ij = r_vec / r_ij
                proj_op = np.outer(u_ij, u_ij)
                
                # Get atomic radii
                r_i = self.params.get_radius(element_list[i])
                r_j = self.params.get_radius(element_list[j])
                
                # Simple repulsion term (r^-12)
                rep_scale = 0.05  # Scale factor for repulsion
                rep_sum = r_i + r_j
                rep_term = rep_scale * ((rep_sum / r_ij)**12)
                
                # Electrostatic term (q_i * q_j / r)
                elec_scale = 0.1  # Scale factor for electrostatics
                elec_term = elec_scale * charges[i] * charges[j] / r_ij
                
                # Combine terms for total non-bonded Hessian contribution
                hess_factor = (12.0 * rep_term / r_ij**2) + (2.0 * elec_term / r_ij**2)
                
                # Add to Cartesian Hessian
                for n in range(3):
                    for m in range(3):
                        self.cart_hess[3*i+n, 3*i+m] += hess_factor * proj_op[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_factor * proj_op[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_factor * proj_op[n, m]
                        self.cart_hess[3*j+n, 3*i+m] -= hess_factor * proj_op[n, m]
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian using GFN0-xTB model
        
        Parameters:
            coord: Atomic coordinates (N×3 array, Bohr)
            element_list: List of element symbols
            cart_gradient: Gradient in Cartesian coordinates
            
        Returns:
            hess_proj: Hessian with rotational and translational modes projected out
        """
        print("Generating Hessian using GFN0-xTB model...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Analyze molecular structure
        topology = self.analyze_molecular_structure(coord, element_list)
        
        # Calculate different Hessian components
        self.gfn0_bond_hessian(coord, element_list, topology)
        self.gfn0_angle_hessian(coord, element_list, topology)
        self.gfn0_torsion_hessian(coord, element_list, topology)
        self.gfn0_nonbonded_hessian(coord, element_list, topology)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj
