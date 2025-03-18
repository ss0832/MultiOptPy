import itertools

import numpy as np

from bond_connectivity import BondConnectivity
from parameter import UnitValueLib, number_element, element_number, covalent_radii_lib, UFF_effective_charge_lib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, atomic_mass, D2_VDW_radii_lib, D2_C6_coeff_lib, D2_S6_parameter, double_covalent_radii_lib, triple_covalent_radii_lib, D3Parameters, D4Parameters, GFNFFParameters, GFN0Parameters, GNB_radii_lib
from redundant_coordinations import RedundantInternalCoordinates
from calc_tools import Calculationtools
from scipy.special import erf

import numpy as np
import itertools
from collections import defaultdict

from bond_connectivity import BondConnectivity
from parameter import UnitValueLib, number_element, element_number, covalent_radii_lib
from calc_tools import Calculationtools
from redundant_coordinations import RedundantInternalCoordinates



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

class GFNFFApproxHessian:
    def __init__(self):
        """
        GFNFF-based model Hessian implementation
        Based on the GFN-FF method by Grimme et al., JCTC 2017, 13, 1989-2009
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.params = GFNFFParameters()
        self.bond_factor = 1.3  # Bond detection threshold factor
        
    def estimate_atomic_charges(self, coord, element_list, bond_mat):
        """
        Estimate atomic partial charges using a simplified EEQ model based on GFNFF approach
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            bond_mat: bond connectivity matrix
            
        Returns:
            charges: array of estimated atomic charges
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Calculate coordination numbers for charge dependence
        cn = self.calculate_coordination_numbers(coord, element_list)
        
        # Calculate electronegativity-based charge transfer
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j or not bond_mat[i, j]:
                    continue
                
                # Get electronegativity values
                en_i = self.params.get_electronegativity(element_list[i])
                en_j = self.params.get_electronegativity(element_list[j])
                
                # Calculate distance in Angstroms
                r_ij = np.linalg.norm(coord[i] - coord[j]) * self.bohr2angstroms
                
                # CN-dependent electronegativity adjustment (simplified from GFNFF)
                cn_ref_i = self.params.ref_cn.get(element_list[i], 1.0)
                cn_ref_j = self.params.ref_cn.get(element_list[j], 1.0)
                
                cn_factor_i = np.exp(-0.1 * (cn[i] - cn_ref_i)**2)
                cn_factor_j = np.exp(-0.1 * (cn[j] - cn_ref_j)**2)
                
                en_eff_i = en_i * cn_factor_i
                en_eff_j = en_j * cn_factor_j
                
                # Simple electronegativity-based charge transfer
                charge_transfer = 0.1 * (en_eff_j - en_eff_i) / (r_ij * (en_eff_i + en_eff_j))
                charges[i] += charge_transfer
                charges[j] -= charge_transfer
        
        # Normalize charges to ensure neutrality
        charges -= np.mean(charges)
        
        return charges
    
    def calculate_coordination_numbers(self, coord, element_list):
        """
        Calculate atomic coordination numbers using the counting function from GFNFF
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            
        Returns:
            cn: array of coordination numbers for each atom
        """
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                # Get covalent radii
                r_cov_i = self.params.get_cov_radius(element_list[i])
                r_cov_j = self.params.get_cov_radius(element_list[j])
                
                # Calculate distance in Angstroms
                dist_ij = np.linalg.norm(coord[i] - coord[j]) * self.bohr2angstroms
                
                # Calculate coordination number contribution (GFNFF counting function)
                r_cov = r_cov_i + r_cov_j
                k_cn = 16.0  # Steepness of the counting function
                cn_contrib = 1.0 / (1.0 + np.exp(-k_cn * (r_cov * 1.2 / dist_ij - 1.0)))
                cn[i] += cn_contrib
                
        return cn
    
    def get_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix and related data"""
        n_atoms = len(coord)
        dist_mat = np.zeros((n_atoms, n_atoms))
        pair_cov_radii_mat = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coord[i] - coord[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
                
                r_cov_i = self.params.get_cov_radius(element_list[i])
                r_cov_j = self.params.get_cov_radius(element_list[j])
                cov_sum = (r_cov_i + r_cov_j) / self.bohr2angstroms  # Convert to bohr
                
                pair_cov_radii_mat[i, j] = pair_cov_radii_mat[j, i] = cov_sum
        
        # Bond connectivity matrix (True if bond exists between atoms)
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)  # No self-bonds
        
        return bond_mat, dist_mat, pair_cov_radii_mat
    
    def detect_hydrogen_bonds(self, coord, element_list, bond_mat, charges):
        """
        Detect hydrogen bonds in the structure based on GFNFF criteria
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            bond_mat: bond connectivity matrix
            charges: atomic partial charges
            
        Returns:
            hbonds: list of hydrogen bonds as (donor, H, acceptor) triplets
        """
        n_atoms = len(coord)
        hbonds = []
        
        # Define potential donors (electronegative elements that can have H attached)
        donors = ['O', 'N', 'F', 'Cl', 'Br', 'I', 'S']
        acceptors = ['O', 'N', 'F', 'Cl', 'Br', 'I', 'S']
        
        # Find all hydrogen bonds
        for i in range(n_atoms):
            if element_list[i] != 'H':
                continue
                
            # Find potential donor (atom bonded to H)
            donor_idx = -1
            for j in range(n_atoms):
                if bond_mat[i, j] and element_list[j] in donors:
                    donor_idx = j
                    break
                    
            if donor_idx == -1:
                continue  # No suitable donor
                
            donor_element = element_list[donor_idx]
            
            # Look for acceptors
            for acceptor_idx in range(n_atoms):
                if acceptor_idx == donor_idx or bond_mat[i, acceptor_idx]:
                    continue  # Skip donor and directly bonded atoms
                    
                acceptor_element = element_list[acceptor_idx]
                if acceptor_element not in acceptors:
                    continue
                    
                # Check if this acceptor-H-donor combination is valid
                if (acceptor_element, 'H', donor_element) not in self.params.hbond_params and \
                   (donor_element, 'H', acceptor_element) not in self.params.hbond_params:
                    continue
                    
                # Calculate H...acceptor distance
                h_acc_dist = np.linalg.norm(coord[i] - coord[acceptor_idx]) * self.bohr2angstroms
                
                # Calculate donor-H...acceptor angle
                d_h_vec = coord[i] - coord[donor_idx]
                h_acc_vec = coord[acceptor_idx] - coord[i]
                d_h_len = np.linalg.norm(d_h_vec)
                h_acc_len = np.linalg.norm(h_acc_vec)
                
                if d_h_len > 0 and h_acc_len > 0:
                    cos_angle = np.dot(d_h_vec, h_acc_vec) / (d_h_len * h_acc_len)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180.0 / np.pi
                    
                    # Get parameters for this H-bond
                    h_params = self.params.get_hbond_params(donor_element, 'H', acceptor_element)
                    r0_hb = h_params[0] * self.bohr2angstroms  # Convert to Angstroms
                    
                    # Check hydrogen bond criteria: distance and angle
                    # Distance criterion: within 30% of optimal distance
                    # Angle criterion: > 120 degrees (approximately linear)
                    if h_acc_dist < 1.3 * r0_hb and angle > 120.0:
                        # Verify with partial charges (acceptor should be negative)
                        if charges[acceptor_idx] < -0.05:
                            hbonds.append((donor_idx, i, acceptor_idx))
        
        return hbonds
    
    def build_topology(self, coord, element_list):
        """
        Build molecular topology including bonds, angles, dihedrals, and non-bonded interactions
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            
        Returns:
            topology: dictionary with molecular topology information
        """
        # Get bond connectivity
        bond_mat, dist_mat, cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Estimate atomic charges
        charges = self.estimate_atomic_charges(coord, element_list, bond_mat)
        
        # Detect hydrogen bonds
        hbonds = self.detect_hydrogen_bonds(coord, element_list, bond_mat, charges)
        
        # Calculate coordination numbers
        cn = self.calculate_coordination_numbers(coord, element_list)
        
        # Build lists of bonds, angles, and dihedrals
        n_atoms = len(coord)
        bonds = []
        angles = []
        dihedrals = []
        
        # Extract bonds
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bond_mat[i, j]:
                    bonds.append((i, j))
        
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
        
        # Build non-bonded pairs (atoms separated by at least 3 bonds)
        nb_pairs = []
        
        # Calculate shortest path lengths between atoms
        bond_graph = defaultdict(list)
        for i, j in bonds:
            bond_graph[i].append(j)
            bond_graph[j].append(i)
            
        # For each atom pair, determine if they're separated by at least 3 bonds
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if not bond_mat[i, j]:  # Not directly bonded
                    # Check if they share a bonded neighbor (1-3 interaction)
                    common_neighbors = set(bond_graph[i]).intersection(bond_graph[j])
                    if not common_neighbors:  # No common neighbors
                        nb_pairs.append((i, j))
                    else:
                        # Check if they're involved in a 1-4 interaction (part of a dihedral)
                        is_14 = False
                        for k in common_neighbors:
                            for l in bond_graph[j]:
                                if l != k and l in bond_graph[k] and l != i:
                                    is_14 = True
                                    break
                            if is_14:
                                break
                        
                        if not is_14:
                            nb_pairs.append((i, j))
        
        # Collect topology information
        topology = {
            'bonds': bonds,
            'angles': angles,
            'dihedrals': dihedrals,
            'nb_pairs': nb_pairs,
            'hbonds': hbonds,
            'charges': charges,
            'cn': cn,
            'bond_mat': bond_mat,
            'dist_mat': dist_mat
        }
        
        return topology
    
    def gfnff_bond_hessian(self, coord, element_list, topology):
        """
        Calculate bond stretching contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular topology information
        """
        bonds = topology['bonds']
        cn = topology['cn']
        n_atoms = len(coord)
        
        for i, j in bonds:
            r_vec = coord[j] - coord[i]
            r_ij = np.linalg.norm(r_vec)
            
            # Get base bond parameters (r0 in bohr, k in atomic units)
            r0, k_bond = self.params.get_bond_params(element_list[i], element_list[j])
            
            # Apply CN-dependent scaling (as in GFNFF)
            cn_ref_i = self.params.ref_cn.get(element_list[i], 1.0)
            cn_ref_j = self.params.ref_cn.get(element_list[j], 1.0)
            
            cn_factor = np.exp(-self.params.bond_decay * ((cn[i] - cn_ref_i)**2 + (cn[j] - cn_ref_j)**2))
            force_const = k_bond * cn_factor * self.params.bond_scaling
            
            # Calculate unit vector and projection operator
            if r_ij > 1e-10:
                u_ij = r_vec / r_ij
                proj_op = np.outer(u_ij, u_ij)
            else:
                # Avoid division by zero
                proj_op = np.eye(3) / 3.0
            
            # Calculate force constant with exponential term for deviation from equilibrium
            # (GFNFF uses a Morse-like function for bonds, but here we use a simpler approach)
            exp_factor = np.exp(-2.0 * (r_ij - r0)**2)
            force_const *= exp_factor
            
            # Calculate Hessian blocks
            h_diag = force_const * proj_op
            
            # Add to Cartesian Hessian
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += h_diag[n, m]
                    self.cart_hess[3*j+n, 3*j+m] += h_diag[n, m]
                    self.cart_hess[3*i+n, 3*j+m] -= h_diag[n, m]
                    self.cart_hess[3*j+n, 3*i+m] -= h_diag[n, m]
    
    def gfnff_angle_hessian(self, coord, element_list, topology):
        """
        Calculate angle bending contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular topology information
        """
        angles = topology['angles']
        cn = topology['cn']
        n_atoms = len(coord)
        
        for i, j, k in angles:
            # Calculate vectors and distances
            r_ji = coord[i] - coord[j]
            r_jk = coord[k] - coord[j]
            r_ji_len = np.linalg.norm(r_ji)
            r_jk_len = np.linalg.norm(r_jk)
            
            # Skip if atoms are too close
            if r_ji_len < 1e-10 or r_jk_len < 1e-10:
                continue
            
            # Calculate angle
            cos_theta = np.dot(r_ji, r_jk) / (r_ji_len * r_jk_len)
            cos_theta = np.clip(cos_theta, -0.999999, 0.999999)  # Avoid numerical issues
            theta = np.arccos(cos_theta)
            
            # Get angle parameters (theta0 in degrees, k in atomic units)
            theta0, k_angle = self.params.get_angle_params(element_list[i], element_list[j], element_list[k])
            theta0 = theta0 * np.pi / 180.0  # Convert to radians
            
            # Apply coordination number scaling
            cn_ref_j = self.params.ref_cn.get(element_list[j], 1.0)
            cn_factor = np.exp(-0.1 * (cn[j] - cn_ref_j)**2)
            force_const = k_angle * cn_factor
            
            # Calculate unit vectors
            u_ji = r_ji / r_ji_len
            u_jk = r_jk / r_jk_len
            
            # Calculate derivatives of the angle w.r.t. Cartesian coordinates
            # This is a simplified approach for the Hessian
            p_i = (u_ji - cos_theta * u_jk) / (r_ji_len * np.sin(theta))
            p_k = (u_jk - cos_theta * u_ji) / (r_jk_len * np.sin(theta))
            p_j = -p_i - p_k
            
            # Build derivative vectors
            deriv_i = p_i
            deriv_j = p_j
            deriv_k = p_k
            
            # Calculate Hessian blocks (simplified approach)
            for a in range(3):
                for b in range(3):
                    # i-i block
                    self.cart_hess[3*i+a, 3*i+b] += force_const * deriv_i[a] * deriv_i[b]
                    
                    # j-j block
                    self.cart_hess[3*j+a, 3*j+b] += force_const * deriv_j[a] * deriv_j[b]
                    
                    # k-k block
                    self.cart_hess[3*k+a, 3*k+b] += force_const * deriv_k[a] * deriv_k[b]
                    
                    # Cross terms
                    self.cart_hess[3*i+a, 3*j+b] += force_const * deriv_i[a] * deriv_j[b]
                    self.cart_hess[3*i+a, 3*k+b] += force_const * deriv_i[a] * deriv_k[b]
                    self.cart_hess[3*j+a, 3*i+b] += force_const * deriv_j[a] * deriv_i[b]
                    self.cart_hess[3*j+a, 3*k+b] += force_const * deriv_j[a] * deriv_k[b]
                    self.cart_hess[3*k+a, 3*i+b] += force_const * deriv_k[a] * deriv_i[b]
                    self.cart_hess[3*k+a, 3*j+b] += force_const * deriv_k[a] * deriv_j[b]
    
    def gfnff_torsion_hessian(self, coord, element_list, topology):
        """
        Calculate torsion (dihedral) contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular topology information
        """
        dihedrals = topology['dihedrals']
        cn = topology['cn']
        n_atoms = len(coord)
        
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
            
            # Get torsion parameters (V1, V2, V3 in hartree)
            v1, v2, v3 = self.params.get_torsion_params(
                element_list[i], element_list[j], element_list[k], element_list[l]
            )
            
            # Apply coordination number scaling (simplified)
            cn_ref_j = self.params.ref_cn.get(element_list[j], 1.0)
            cn_ref_k = self.params.ref_cn.get(element_list[k], 1.0)
            cn_factor = np.exp(-0.05 * ((cn[j] - cn_ref_j)**2 + (cn[k] - cn_ref_k)**2))
            
            v1 *= cn_factor
            v2 *= cn_factor
            v3 *= cn_factor
            
            # Calculate forces based on derivatives of the potential
            # V = v1*(1+cos(phi)) + v2*(1+cos(2*phi)) + v3*(1+cos(3*phi-pi))
            f1 = -v1 * np.sin(phi)
            f2 = -2 * v2 * np.sin(2*phi)
            f3 = -3 * v3 * np.sin(3*phi)
            
            force = f1 + f2 + f3
            
            # Calculate second derivatives (simplified for Hessian)
            # This is a simplified approach; a complete implementation would 
            # include all second derivatives of the torsion angles
            k2 = v1 * np.cos(phi) + 4 * v2 * np.cos(2*phi) + 9 * v3 * np.cos(3*phi)
            
            # Calculate derivatives of phi w.r.t. Cartesian coordinates
            # (Simplified approach - for a full treatment, see the properly derived formulas)
            # This approximation might not be accurate for all geometries
            
            # Normalize bond vectors
            e_ij = r_ij / np.linalg.norm(r_ij)
            e_jk = r_jk / r_jk_norm
            e_kl = r_kl / np.linalg.norm(r_kl)
            
            # Normalized cross products
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm
            
            # Calculate derivatives (simplified)
            # For a complete treatment, derive the derivatives of the dihedral angle
            # with respect to all atomic positions
            
            # Apply a simple projection approach for the derivatives
            # These are simplified derivatives and not analytically correct for all geometries
            deriv_i = np.cross(e_ij, n1) / np.linalg.norm(r_ij)
            deriv_l = np.cross(n2, e_kl) / np.linalg.norm(r_kl)
            deriv_j = -deriv_i - np.cross(e_jk, n1) / np.linalg.norm(r_ij)
            deriv_k = -deriv_l - np.cross(n2, e_jk) / np.linalg.norm(r_kl)
            
            # Scale derivatives by force constant
            deriv_i *= force
            deriv_j *= force
            deriv_k *= force
            deriv_l *= force
            
            # Add to Cartesian Hessian
            # This is a very simplified Hessian contribution
            atoms = [i, j, k, l]
            derivs = [deriv_i, deriv_j, deriv_k, deriv_l]
            
            for m in range(4):
                for n in range(4):
                    if m <= n:  # Only calculate upper triangular part
                        for a in range(3):
                            for b in range(3):
                                self.cart_hess[3*atoms[m]+a, 3*atoms[n]+b] += k2 * derivs[m][a] * derivs[n][b]
            
            # Make Hessian symmetric
            for m in range(3*n_atoms):
                for n in range(m):
                    self.cart_hess[m, n] = self.cart_hess[n, m]
    
    def gfnff_hbond_hessian(self, coord, element_list, topology):
        """
        Calculate hydrogen bond contributions to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular topology information
        """
        hbonds = topology['hbonds']
        
        for donor_idx, h_idx, acceptor_idx in hbonds:
            # Get atom elements
            donor_element = element_list[donor_idx]
            acceptor_element = element_list[acceptor_idx]
            
            # Get H-bond parameters
            r0_hb, k_hb = self.params.get_hbond_params(donor_element, 'H', acceptor_element)
            
            # Calculate vectors and distances
            r_dh = coord[h_idx] - coord[donor_idx]
            r_ha = coord[acceptor_idx] - coord[h_idx]
            r_dh_len = np.linalg.norm(r_dh)
            r_ha_len = np.linalg.norm(r_ha)
            
            # Skip if atoms are too close
            if r_dh_len < 1e-10 or r_ha_len < 1e-10:
                continue
            
            # Calculate angle
            cos_angle = np.dot(r_dh, r_ha) / (r_dh_len * r_ha_len)
            cos_angle = np.clip(cos_angle, -0.999999, 0.999999)  # Avoid numerical issues
            angle = np.arccos(cos_angle)
            
            # H-bond force constant depends on distance and angle
            # Optimal H-bond is linear (180 degrees) and at equilibrium distance
            
            # Distance-dependent term
            dist_factor = np.exp(-(r_ha_len - r0_hb)**2 / (2.0 * 0.3**2))  # Gaussian shape
            
            # Angle-dependent term (preference for linear H-bonds)
            angle_factor = (1.0 + np.cos(angle - np.pi))**2 / 4.0  # Peaks at 180 degrees
            
            # Combined force constant
            force_const = k_hb * dist_factor * angle_factor
            
            # Calculate unit vectors for H-A bond
            u_ha = r_ha / r_ha_len
            proj_op = np.outer(u_ha, u_ha)
            
            # Calculate Hessian blocks for H-bond stretching
            # This is a simplified approach focusing on the H-A distance
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*h_idx+n, 3*h_idx+m] += force_const * proj_op[n, m]
                    self.cart_hess[3*acceptor_idx+n, 3*acceptor_idx+m] += force_const * proj_op[n, m]
                    self.cart_hess[3*h_idx+n, 3*acceptor_idx+m] -= force_const * proj_op[n, m]
                    self.cart_hess[3*acceptor_idx+n, 3*h_idx+m] -= force_const * proj_op[n, m]
            
            # Angle component is more complex and not included in this simplified model
    
    def gfnff_nonbonded_hessian(self, coord, element_list, topology):
        """
        Calculate non-bonded interactions (dispersion and repulsion) to the Hessian
        
        Parameters:
            coord: atomic coordinates (Bohr)
            element_list: list of element symbols
            topology: molecular topology information
        """
        n_atoms = len(coord)
        nb_pairs = topology['nb_pairs']
        charges = topology['charges']
        
        for i, j in nb_pairs:
            # Calculate distance vector and magnitude
            r_vec = coord[i] - coord[j]
            r_ij = np.linalg.norm(r_vec)
            
            # Skip if atoms are too close
            if r_ij < 0.1:
                continue
            
            # Get element properties
            alpha_i = self.params.get_polarizability(element_list[i])
            alpha_j = self.params.get_polarizability(element_list[j])
            
            # Calculate simple dispersion C6 coefficient (simplified DFT-D3 style)
            c6_ij = 2.0 * alpha_i * alpha_j / (alpha_i/alpha_j + alpha_j/alpha_i) * 0.05
            
            # Calculate van der Waals radii
            vdw_i = self.params.get_vdw_radius(element_list[i]) / self.bohr2angstroms
            vdw_j = self.params.get_vdw_radius(element_list[j]) / self.bohr2angstroms
            vdw_sum = vdw_i + vdw_j
            
            # Calculate dispersion energy and derivatives
            # Repulsive term: simple exponential
            rep_const = 0.3  # Repulsion strength
            rep_energy = rep_const * np.exp(-(r_ij/vdw_sum - 0.6) * 12.0)
            rep_deriv = -12.0 * rep_const * np.exp(-(r_ij/vdw_sum - 0.6) * 12.0) * (1.0/vdw_sum) / r_ij
            
            # Attractive dispersion term (D3-like with BJ-damping)
            r0_ij = 0.5 * vdw_sum
            a1, a2 = 0.4, 3.0  # BJ-damping parameters
            damp_factor = r_ij**6 / (r_ij**6 + (a1*r0_ij + a2)**6)
            disp_energy = -self.params.d4_s6 * c6_ij * damp_factor / r_ij**6
            
            # Compute the derivative of the damping factor
            damp_deriv = 6 * r_ij**5 * (a1*r0_ij + a2)**6 / (r_ij**6 + (a1*r0_ij + a2)**6)**2
            disp_deriv = self.params.d4_s6 * c6_ij * (6 * damp_factor / r_ij**7 - damp_deriv / r_ij**6)
            
            # Total force
            force = rep_deriv + disp_deriv
            
            # Calculate unit vector and projection operator
            u_ij = r_vec / r_ij
            proj_op = np.outer(u_ij, u_ij)
            
            # Simplified second derivatives for Hessian
            # This is not analytically correct but provides an approximation
            # For correct treatment, one would need the full second derivatives
            hess_factor = (force / r_ij) + 0.2  # Adding a small positive constant for stability
            
            # Add to Cartesian Hessian
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += hess_factor * proj_op[n, m]
                    self.cart_hess[3*j+n, 3*j+m] += hess_factor * proj_op[n, m]
                    self.cart_hess[3*i+n, 3*j+m] -= hess_factor * proj_op[n, m]
                    self.cart_hess[3*j+n, 3*i+m] -= hess_factor * proj_op[n, m]
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian using GFNFF-based model
        
        Parameters:
            coord: Atomic coordinates (N×3 array, Bohr)
            element_list: List of element symbols
            cart_gradient: Gradient in Cartesian coordinates
            
        Returns:
            hess_proj: Hessian with rotational and translational modes projected out
        """
        print("Generating Hessian using GFNFF model...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Build molecular topology
        topology = self.build_topology(coord, element_list)
        
        # Calculate different Hessian components
        self.gfnff_bond_hessian(coord, element_list, topology)
        self.gfnff_angle_hessian(coord, element_list, topology)
        self.gfnff_torsion_hessian(coord, element_list, topology)
        self.gfnff_hbond_hessian(coord, element_list, topology)
        self.gfnff_nonbonded_hessian(coord, element_list, topology)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj


class FischerD4ApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation with D4 dispersion correction
        Ref: Fischer and Almlöf, J. Phys. Chem., 1992, 96, 24, 9768–9774
        D4 Ref: Caldeweyher et al., J. Chem. Phys., 2019, 150, 154122
        Implementation Ref.:pysisyphus.optimizers.guess_hessians
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.bond_factor = 1.3  # Bond detection threshold factor
        
        # D4 dispersion correction parameters (default: PBE0)
        self.d4_params = D4Parameters()
        
    def calc_bond_force_const(self, r_ab, r_ab_cov):
        """Calculate force constant for bond stretching using Fischer formula"""
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        """Calculate force constant for angle bending using Fischer formula"""
        return 0.089 + 0.11 / (r_ab_cov * r_ac_cov) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        )
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        """Calculate force constant for dihedral torsion using Fischer formula"""
        return 0.0015 + 14.0 * np.maximum(bond_sum, 0) ** 0.57 / (r_ab * r_ab_cov) ** 4.0 * np.exp(
            -2.85 * (r_ab - r_ab_cov)
        )
        
    def calculate_coordination_numbers(self, coord, element_list):
        """
        Calculate atomic coordination numbers
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            
        Returns:
            cn: array of coordination numbers for each atom
        """
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                # Get covalent radii
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                
                # Calculate distance
                dist_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Calculate coordination number contribution using cutoff function
                r_cov = r_cov_i + r_cov_j
                tmp = np.exp(-16.0 * ((dist_ij / (r_cov * 1.2)) - 1.0))
                cn[i] += 1.0 / (1.0 + tmp)
                
        return cn
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic partial charges using a simple electronegativity model
        In a real implementation, this would use proper EEQ method or external charges
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            
        Returns:
            charges: array of estimated atomic charges
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Simple estimation using bond distances and electronegativity differences
        bond_mat, dist_mat, _ = self.get_bond_connectivity(coord, element_list)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bond_mat[i, j]:
                    en_i = self.d4_params.get_electronegativity(element_list[i])
                    en_j = self.d4_params.get_electronegativity(element_list[j])
                    
                    # Simple electronegativity-based charge transfer
                    # The 0.1 factor is a simplification; real EEQ would be more complex
                    charge_transfer = 0.1 * (en_j - en_i) / (en_i + en_j) * 2.0
                    charges[i] += charge_transfer
                    charges[j] -= charge_transfer
        
        return charges
    
    def get_charge_scaling_factor(self, element, charge):
        """
        Calculate charge scaling factor for D4 dispersion coefficients
        
        Parameters:
            element: element symbol
            charge: atomic charge
            
        Returns:
            charge_factor: exponential charge scaling factor
        """
        ga = self.d4_params.ga
        charge_factor = np.exp(-ga * charge * charge)
        return charge_factor
    
    def get_c6_coefficient(self, element_i, element_j, q_i=0.0, q_j=0.0):
        """
        Get C6 coefficient based on D4 model with charge scaling
        
        Parameters:
            element_i, element_j: element symbols for atoms
            q_i, q_j: atomic charges
            
        Returns:
            c6_ij: dispersion coefficient for atom pair
        """
        # Get reference polarizabilities
        alpha_i = self.d4_params.get_polarizability(element_i)
        alpha_j = self.d4_params.get_polarizability(element_j)
        
        # Apply charge scaling
        scale_i = self.get_charge_scaling_factor(element_i, q_i)
        scale_j = self.get_charge_scaling_factor(element_j, q_j)
        
        # D4 approach using dynamic polarizabilities
        # The 0.75 factor is an empirical scaling constant
        c6_ij = 2.0 * alpha_i * alpha_j / (alpha_i / scale_i + alpha_j / scale_j) * 0.75
        
        return c6_ij
    
    def get_c8_coefficient(self, element_i, element_j, q_i=0.0, q_j=0.0):
        """
        Calculate C8 coefficient based on D4 model
        
        Parameters:
            element_i, element_j: element symbols for atoms
            q_i, q_j: atomic charges
            
        Returns:
            c8_ij: higher-order dispersion coefficient
        """
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        r4r2_i = self.d4_params.get_r4r2(element_i)
        r4r2_j = self.d4_params.get_r4r2(element_j)
        
        # C8 = 3 * C6 * sqrt(r4r2_i * r4r2_j)
        c8_ij = 3.0 * c6_ij * np.sqrt(r4r2_i * r4r2_j)
        return c8_ij
    
    def get_r0_value(self, element_i, element_j):
        """
        Calculate R0 value for D4 model (characteristic distance for atom pair)
        
        Parameters:
            element_i, element_j: element symbols for atoms
            
        Returns:
            r0: reference distance for damping function
        """
        # Using covalent radii as a base for reference distance
        try:
            r_i = covalent_radii_lib(element_i) * 4.0/3.0
            r_j = covalent_radii_lib(element_j) * 4.0/3.0
            return r_i + r_j
        except:
            # If exception occurs, estimate from covalent radii
            r_i = covalent_radii_lib(element_i) * 2.0
            r_j = covalent_radii_lib(element_j) * 2.0
            return r_i + r_j
    
    def d4_damping_function(self, r_ij, r0, order=6):
        """
        BJ (Becke-Johnson) damping function for D4
        
        Parameters:
            r_ij: Interatomic distance
            r0: Reference radius
            order: 6 for C6 term, 8 for C8 term
        
        Returns:
            f_damp: value of damping function
        """
        if order == 6:
            a1, a2 = self.d4_params.a1, self.d4_params.a2
        else:  # order == 8
            a1, a2 = self.d4_params.a1, self.d4_params.a2 + 2.0  # C8 damping is slightly different
            
        # BJ-damping (Becke-Johnson)
        denominator = r_ij**order + (a1 * r0 + a2)**order
        return r_ij**order / denominator
    
    def three_body_damping(self, r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki):
        """
        Three-body damping function for the ATM term
        
        Parameters:
            r_ij, r_jk, r_ki: Interatomic distances for the triangle
            r0_ij, r0_jk, r0_ki: Reference radii for the atom pairs
            
        Returns:
            f_damp: value of three-body damping function
        """
        # The geometric average of the three damping functions
        f_ij = self.d4_damping_function(r_ij, r0_ij, order=6)
        f_jk = self.d4_damping_function(r_jk, r0_jk, order=6)
        f_ki = self.d4_damping_function(r_ki, r0_ki, order=6)
        
        return f_ij * f_jk * f_ki
    
    def d4_energy_contribution(self, r_ij, element_i, element_j, q_i, q_j):
        """
        Calculate D4 pairwise dispersion energy
        
        Parameters:
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
            q_i, q_j: Atomic charges
        
        Returns:
            energy: dispersion energy contribution for the atom pair
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return 0.0
            
        # Get C6 and C8 coefficients with charge-dependence
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        # Energy calculation
        e6 = -self.d4_params.s6 * c6_ij / r_ij**6 * f_damp6
        e8 = -self.d4_params.s8 * c8_ij / r_ij**8 * f_damp8
        
        return e6 + e8
    
    def calculate_three_body_term(self, coord, element_list, charges):
        """
        Calculate the three-body dispersion energy term (ATM)
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            charges: array of atomic charges
            
        Returns:
            energy: three-body dispersion energy
        """
        energy = 0.0
        n_atoms = len(coord)
        
        # Calculate for all atom triplets
        for i, j, k in itertools.combinations(range(n_atoms), 3):
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ki = np.linalg.norm(coord[k] - coord[i])
            
            # Skip if any atoms are too close
            if min(r_ij, r_jk, r_ki) < 0.1:
                continue
                
            # Charge-dependent three-body C9 coefficient
            c6_ij = self.get_c6_coefficient(element_list[i], element_list[j], charges[i], charges[j])
            c6_jk = self.get_c6_coefficient(element_list[j], element_list[k], charges[j], charges[k])
            c6_ki = self.get_c6_coefficient(element_list[k], element_list[i], charges[k], charges[i])
            c9_ijk = np.sqrt(c6_ij * c6_jk * c6_ki)
            
            # Calculate angular dependent factor for ATM term
            r_vec_ij = coord[j] - coord[i]
            r_vec_jk = coord[k] - coord[j]
            r_vec_ki = coord[i] - coord[k]
            
            cos_i = np.dot(-r_vec_ki, r_vec_ij) / (r_ki * r_ij)
            cos_j = np.dot(-r_vec_ij, r_vec_jk) / (r_ij * r_jk)
            cos_k = np.dot(-r_vec_jk, r_vec_ki) / (r_jk * r_ki)
            
            angle_factor = 1.0 + 3.0 * cos_i * cos_j * cos_k
            
            # Calculate damping function
            r0_ij = self.get_r0_value(element_list[i], element_list[j])
            r0_jk = self.get_r0_value(element_list[j], element_list[k])
            r0_ki = self.get_r0_value(element_list[k], element_list[i])
            damp_factor = self.three_body_damping(r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki)
            
            # Axilrod-Teller-Muto term
            e_atm = angle_factor * c9_ijk * damp_factor / (r_ij * r_jk * r_ki)**3
            energy += e_atm
            
        return energy * self.d4_params.s9
    
    def d4_gradient_contribution(self, r_vec, r_ij, element_i, element_j, q_i, q_j):
        """
        Calculate D4 pairwise dispersion gradient
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
            q_i, q_j: Atomic charges
        
        Returns:
            gradient: dispersion gradient contribution for the atom pair
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros(3)
            
        # Get C6 and C8 coefficients with charge-dependence
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d4_params.a1, self.d4_params.a2
        a1_8, a2_8 = self.d4_params.a1, self.d4_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Gradient calculation
        g6 = -self.d4_params.s6 * c6_ij * ((-6 / r_ij**7) * f_damp6 + (1 / r_ij**6) * df_damp6)
        g8 = -self.d4_params.s8 * c8_ij * ((-8 / r_ij**9) * f_damp8 + (1 / r_ij**8) * df_damp8)
        
        unit_vec = r_vec / r_ij
        return (g6 + g8) * unit_vec
    
    def d4_three_body_gradient(self, coord, element_list, charges):
        """
        Calculate gradient contributions from the three-body dispersion term
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            charges: array of atomic charges
            
        Returns:
            gradients: gradient contributions from three-body term (N×3 array)
        """
        n_atoms = len(coord)
        gradients = np.zeros((n_atoms, 3))
        
        # Calculate for all atom triplets
        for i, j, k in itertools.combinations(range(n_atoms), 3):
            # Distances between atoms
            r_vec_ij = coord[j] - coord[i]
            r_vec_jk = coord[k] - coord[j]
            r_vec_ki = coord[i] - coord[k]
            
            r_ij = np.linalg.norm(r_vec_ij)
            r_jk = np.linalg.norm(r_vec_jk)
            r_ki = np.linalg.norm(r_vec_ki)
            
            # Skip if any atoms are too close
            if min(r_ij, r_jk, r_ki) < 0.1:
                continue
                
            # Unit vectors
            u_ij = r_vec_ij / r_ij
            u_jk = r_vec_jk / r_jk
            u_ki = r_vec_ki / r_ki
            
            # Charge-dependent three-body C9 coefficient
            c6_ij = self.get_c6_coefficient(element_list[i], element_list[j], charges[i], charges[j])
            c6_jk = self.get_c6_coefficient(element_list[j], element_list[k], charges[j], charges[k])
            c6_ki = self.get_c6_coefficient(element_list[k], element_list[i], charges[k], charges[i])
            c9_ijk = np.sqrt(c6_ij * c6_jk * c6_ki)
            
            # Get cosines for the angles
            cos_i = np.dot(-u_ki, u_ij)
            cos_j = np.dot(-u_ij, u_jk)
            cos_k = np.dot(-u_jk, u_ki)
            
            # Angular factor and its derivatives
            angle_factor = 1.0 + 3.0 * cos_i * cos_j * cos_k
            
            # Calculate damping
            # Calculate damping function
            r0_ij = self.get_r0_value(element_list[i], element_list[j])
            r0_jk = self.get_r0_value(element_list[j], element_list[k])
            r0_ki = self.get_r0_value(element_list[k], element_list[i])
            damp_factor = self.three_body_damping(r_ij, r_jk, r_ki, r0_ij, r0_jk, r0_ki)
            
            # Prefactor for gradient calculation
            pre_factor = self.d4_params.s9 * c9_ijk * damp_factor * angle_factor
            
            # Base gradient components (without angular derivatives)
            g_base = -3.0 * pre_factor / (r_ij * r_jk * r_ki)**3
            
            # Gradient contributions for each atom from distances
            g_i = g_base * (u_ij / r_ij - u_ki / r_ki)
            g_j = g_base * (u_jk / r_jk - u_ij / r_ij)
            g_k = g_base * (u_ki / r_ki - u_jk / r_jk)
            
            # Add angular derivative components (simplified approximation)
            # This is a simplified approach; a full analytical derivation would be more complex
            g_ang_i = 3.0 * pre_factor * (cos_j * cos_k / (r_ij * r_jk * r_ki)**3) * (-u_ij - u_ki)
            g_ang_j = 3.0 * pre_factor * (cos_i * cos_k / (r_ij * r_jk * r_ki)**3) * (-u_jk - u_ij)
            g_ang_k = 3.0 * pre_factor * (cos_i * cos_j / (r_ij * r_jk * r_ki)**3) * (-u_ki - u_jk)
            
            # Combine distance and angular contributions
            gradients[i] += g_i + g_ang_i
            gradients[j] += g_j + g_ang_j
            gradients[k] += g_k + g_ang_k
            
        return gradients
    
    def d4_hessian_contribution(self, r_vec, r_ij, element_i, element_j, q_i, q_j):
        """
        Calculate D4 dispersion contribution to Hessian for a pair of atoms
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
            q_i, q_j: Atomic charges
        
        Returns:
            hessian: 3×3 Hessian block for the atom pair
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros((3, 3))
            
        # Get C6 and C8 coefficients with charge-dependence
        c6_ij = self.get_c6_coefficient(element_i, element_j, q_i, q_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j, q_i, q_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d4_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d4_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d4_params.a1, self.d4_params.a2
        a1_8, a2_8 = self.d4_params.a1, self.d4_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Second derivatives of damping functions (approximate)
        d2f_damp6 = (30 * r_ij**4 / denom6 - 60 * r_ij**11 / denom6**2) - df_damp6 / r_ij
        d2f_damp8 = (56 * r_ij**6 / denom8 - 112 * r_ij**15 / denom8**2) - df_damp8 / r_ij
        
        # Unit vector and projection operator
        unit_vec = r_vec / r_ij
        proj_op = np.outer(unit_vec, unit_vec)
        
        # C6 term contribution to Hessian
        h6_coeff = self.d4_params.s6 * c6_ij * (
            (42 / r_ij**8) * f_damp6 - 
            (6 / r_ij**7) * df_damp6 + 
            (1 / r_ij**6) * d2f_damp6
        )
        
        # C8 term contribution to Hessian
        h8_coeff = self.d4_params.s8 * c8_ij * (
            (72 / r_ij**10) * f_damp8 - 
            (8 / r_ij**9) * df_damp8 + 
            (1 / r_ij**8) * d2f_damp8
        )
        
        # Calculation of projection and perpendicular parts
        h_proj = h6_coeff + h8_coeff
        h_perp = (
            self.d4_params.s6 * c6_ij * (6 / r_ij**8) * f_damp6 + 
            self.d4_params.s8 * c8_ij * (8 / r_ij**10) * f_damp8
        )
        
        # Construct Hessian matrix
        identity = np.eye(3)
        hessian = h_proj * proj_op + h_perp * (identity - proj_op)
        
        return hessian
    
    def d4_three_body_hessian(self, coord, element_list, charges):
        """
        Calculate Hessian contributions from the three-body dispersion term
        This is a simplified version using finite differences
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            charges: array of atomic charges
            
        Returns:
            hessian: Hessian matrix contribution from three-body term (3N×3N array)
        """
        n_atoms = len(coord)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        
        # Calculate base gradients
        base_gradients = self.d4_three_body_gradient(coord, element_list, charges)
        
        # Numerical Hessian calculation using finite differences
        delta = 1e-5  # Small displacement for finite difference
        
        # For each atom and coordinate
        for i in range(n_atoms):
            for j in range(3):
                # Create displaced coordinates
                coord_plus = np.copy(coord)
                coord_plus[i, j] += delta
                
                # Calculate gradients at displaced coordinates
                grad_plus = self.d4_three_body_gradient(coord_plus, element_list, charges)
                
                # Calculate Hessian elements using finite difference
                for k in range(n_atoms):
                    for l in range(3):
                        hessian[3*i+j, 3*k+l] += (grad_plus[k, l] - base_gradients[k, l]) / delta
        
        # Make Hessian symmetric
        for i in range(3*n_atoms):
            for j in range(i):
                hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def get_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix and related data"""
        n_atoms = len(coord)
        dist_mat = np.zeros((n_atoms, n_atoms))
        pair_cov_radii_mat = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coord[i] - coord[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
                
                cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                pair_cov_radii_mat[i, j] = pair_cov_radii_mat[j, i] = cov_sum
        
        # Bond connectivity matrix (True if bond exists between atoms)
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)  # No self-bonds
        
        return bond_mat, dist_mat, pair_cov_radii_mat
    
    def count_bonds_for_dihedral(self, bond_mat, central_atoms):
        """Count number of bonds for central atoms in a dihedral"""
        a, b = central_atoms
        # Sum bonds for both central atoms and subtract 2 (the bond between them is counted twice)
        bond_sum = bond_mat[a].sum() + bond_mat[b].sum() - 2
        return bond_sum
    
    def fischer_bond(self, coord, element_list):
        """Calculate Hessian components for bond stretching"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(b_c_mat)
        
        for idx in bond_indices:
            i, j = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_bond_force_const(r_ij, r_ij_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j]])
            r, b_vec = stretch2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
    
    def fischer_angle(self, coord, element_list):
        """Calculate Hessian components for angle bending"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        angle_indices = BC.angle_connect_table(b_c_mat)
        
        for idx in angle_indices:
            i, j, k = idx  # i-j-k angle
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_bend_force_const(r_ij, r_jk, r_ij_cov, r_jk_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k]])
            theta, b_vec = bend2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
    
    def fischer_dihedral(self, coord, element_list, bond_mat):
        """Calculate Hessian components for dihedral torsion"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        # Calculate bond count for central atoms in dihedrals
        tors_atom_bonds = {}
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            bond_sum = self.count_bonds_for_dihedral(bond_mat, (j, k))
            tors_atom_bonds[(j, k)] = bond_sum
        
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            
            # Central bond
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Get bond count for central atoms
            bond_sum = tors_atom_bonds.get((j, k), 0)
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            tau, b_vec = torsion2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    self.cart_hess[3*l+n, 3*l+m] += force_const * b_vec[3][n] * b_vec[3][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*i+n, 3*l+m] += force_const * b_vec[0][n] * b_vec[3][m]
                    
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*l+m] += force_const * b_vec[1][n] * b_vec[3][m]
                    
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*l+m] += force_const * b_vec[2][n] * b_vec[3][m]
                    
                    self.cart_hess[3*l+n, 3*i+m] += force_const * b_vec[3][n] * b_vec[0][m]
                    self.cart_hess[3*l+n, 3*j+m] += force_const * b_vec[3][n] * b_vec[1][m]
                    self.cart_hess[3*l+n, 3*k+m] += force_const * b_vec[3][n] * b_vec[2][m]
    
    def d4_dispersion_hessian(self, coord, element_list, bond_mat):
        """
        Calculate Hessian correction based on D4 dispersion forces
        
        Parameters:
            coord: atomic coordinates (N×3 array, Bohr)
            element_list: list of element symbols
            bond_mat: bond connectivity matrix
        """
        n_atoms = len(coord)
        
        # Estimate atomic charges
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate pairwise D4 dispersion correction for all atom pairs
        for i in range(n_atoms):
            for j in range(i):
                # Skip bonded atom pairs (already accounted for in Fischer model)
                if bond_mat[i, j]:
                    continue
                    
                # Calculate distance vector and magnitude
                r_vec = coord[i] - coord[j]
                r_ij = np.linalg.norm(r_vec)
                
                # Skip if atoms are too close
                if r_ij < 0.1:
                    continue
                
                # Calculate D4 Hessian contribution (pairwise)
                hess_block = self.d4_hessian_contribution(r_vec, r_ij, element_list[i], element_list[j], charges[i], charges[j])
                
                # Add to the Hessian matrix
                for n in range(3):
                    for m in range(3):
                        self.cart_hess[3*i+n, 3*i+m] += hess_block[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_block[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_block[n, m]
                        self.cart_hess[3*j+n, 3*i+m] -= hess_block[n, m]
        
        # Calculate three-body D4 contribution
        # Note: This can be computationally expensive for large systems
        # For production use, this could be made optional or use additional cutoffs
        three_body_hessian = self.d4_three_body_hessian(coord, element_list, charges)
        self.cart_hess += three_body_hessian
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian combining Fischer model and D4 dispersion correction
        
        Parameters:
            coord: Atomic coordinates (N×3 array, Bohr)
            element_list: List of element symbols
            cart_gradient: Gradient in Cartesian coordinates
            
        Returns:
            hess_proj: Hessian with rotational and translational modes projected out
        """
        print("Generating Hessian using Fischer model with D4 dispersion correction...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate bond connectivity matrix
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Calculate Hessian components from Fischer model
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        # Calculate Hessian components from D4 dispersion correction
        self.d4_dispersion_hessian(coord, element_list, bond_mat)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj
    
class FischerD3ApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation with D3 dispersion correction
        Ref: Fischer and Almlöf, J. Phys. Chem., 1992, 96, 24, 9768–9774
        Implementation Ref.: pysisyphus.optimizers.guess_hessians
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.bond_factor = 1.3  # Bond detection threshold factor
        
        # D3 dispersion correction parameters (default: PBE0)
        self.d3_params = D3Parameters()
        
    def calc_bond_force_const(self, r_ab, r_ab_cov):
        """Calculate force constant for bond stretching using Fischer formula"""
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        """Calculate force constant for angle bending using Fischer formula"""
        return 0.089 + 0.11 / (r_ab_cov * r_ac_cov) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        )
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        """Calculate force constant for dihedral torsion using Fischer formula"""
        return 0.0015 + 14.0 * np.maximum(bond_sum, 0) ** 0.57 / (r_ab * r_ab_cov) ** 4.0 * np.exp(
            -2.85 * (r_ab - r_ab_cov)
        )
    
    def get_c6_coefficient(self, element_i, element_j):
        """Get C6 coefficient based on D3 model (simplified)"""
        # Use D2's C6 coefficients as base
        c6_i = D2_C6_coeff_lib(element_i)
        c6_j = D2_C6_coeff_lib(element_j)
        
        # In D3, this becomes environment-dependent, but for simplicity, use the same calculation method as D2
        c6_ij = np.sqrt(c6_i * c6_j)
        return c6_ij
    
    def get_c8_coefficient(self, element_i, element_j):
        """Calculate C8 coefficient based on D3 model using reference r4r2 values"""
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        r4r2_i = self.d3_params.get_r4r2(element_i)
        r4r2_j = self.d3_params.get_r4r2(element_j)
        
        # C8 = 3 * C6 * sqrt(r4r2_i * r4r2_j)
        c8_ij = 3.0 * c6_ij * np.sqrt(r4r2_i * r4r2_j)
        return c8_ij
    
    def get_r0_value(self, element_i, element_j):
        """Calculate R0 value for D3 model (characteristic distance for atom pair)"""
        # Use existing D2 van der Waals radii
        try:
            r_i = D2_VDW_radii_lib(element_i)
            r_j = D2_VDW_radii_lib(element_j)
            return r_i + r_j
        except:
            # If exception occurs, estimate from covalent radii
            r_i = covalent_radii_lib(element_i) * 1.5
            r_j = covalent_radii_lib(element_j) * 1.5
            return r_i + r_j
    
    def d3_damping_function(self, r_ij, r0, order=6):
        """
        BJ (Becke-Johnson) damping function for D3
        
        Parameters:
            r_ij: Interatomic distance
            r0: Reference radius
            order: 6 for C6 term, 8 for C8 term
        """
        if order == 6:
            a1, a2 = self.d3_params.a1, self.d3_params.a2
        else:  # order == 8
            a1, a2 = self.d3_params.a1, self.d3_params.a2 + 2.0  # C8 damping is slightly different
            
        # BJ-damping (Becke-Johnson)
        denominator = r_ij**order + (a1 * r0 + a2)**order
        return r_ij**order / denominator
    
    def d3_energy_contribution(self, r_ij, element_i, element_j):
        """
        Calculate D3 dispersion energy
        
        Parameters:
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return 0.0
            
        # Get C6 and C8 coefficients
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Energy calculation
        e6 = -self.d3_params.s6 * c6_ij / r_ij**6 * f_damp6
        e8 = -self.d3_params.s8 * c8_ij / r_ij**8 * f_damp8
        
        return e6 + e8
    
    def d3_gradient_contribution(self, r_vec, r_ij, element_i, element_j):
        """
        Calculate D3 dispersion gradient
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros(3)
            
        # Get C6 and C8 coefficients
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d3_params.a1, self.d3_params.a2
        a1_8, a2_8 = self.d3_params.a1, self.d3_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Gradient calculation
        g6 = -self.d3_params.s6 * c6_ij * ((-6 / r_ij**7) * f_damp6 + (1 / r_ij**6) * df_damp6)
        g8 = -self.d3_params.s8 * c8_ij * ((-8 / r_ij**9) * f_damp8 + (1 / r_ij**8) * df_damp8)
        
        unit_vec = r_vec / r_ij
        return (g6 + g8) * unit_vec
    
    def d3_hessian_contribution(self, r_vec, r_ij, element_i, element_j):
        """
        Calculate D3 dispersion contribution to Hessian
        
        Parameters:
            r_vec: Distance vector
            r_ij: Interatomic distance
            element_i, element_j: Element symbols for atoms
        """
        if r_ij < 0.1:  # Exclude atoms that are too close
            return np.zeros((3, 3))
            
        # Get C6 and C8 coefficients
        c6_ij = self.get_c6_coefficient(element_i, element_j)
        c8_ij = self.get_c8_coefficient(element_i, element_j)
        
        # Get R0 value
        r0 = self.get_r0_value(element_i, element_j)
        
        # Damping functions
        f_damp6 = self.d3_damping_function(r_ij, r0, order=6)
        f_damp8 = self.d3_damping_function(r_ij, r0, order=8)
        
        # Derivatives of damping functions
        a1, a2 = self.d3_params.a1, self.d3_params.a2
        a1_8, a2_8 = self.d3_params.a1, self.d3_params.a2 + 2.0
        
        denom6 = r_ij**6 + (a1 * r0 + a2)**6
        denom8 = r_ij**8 + (a1_8 * r0 + a2_8)**8
        
        df_damp6 = 6 * r_ij**5 / denom6 - 6 * r_ij**12 / denom6**2
        df_damp8 = 8 * r_ij**7 / denom8 - 8 * r_ij**16 / denom8**2
        
        # Unit vector and projection operator
        unit_vec = r_vec / r_ij
        proj_op = np.outer(unit_vec, unit_vec)
        
        # C6 term contribution to Hessian
        h6_coeff = self.d3_params.s6 * c6_ij / r_ij**8 * (42.0 * f_damp6 - r_ij * df_damp6)
        
        # C8 term contribution to Hessian
        h8_coeff = self.d3_params.s8 * c8_ij / r_ij**10 * (72.0 * f_damp8 - r_ij * df_damp8)
        
        # Calculation of projection and perpendicular parts
        h_proj = h6_coeff + h8_coeff
        h_perp = (self.d3_params.s6 * c6_ij * 6.0 / r_ij**8 + self.d3_params.s8 * c8_ij * 8.0 / r_ij**10) * f_damp6
        
        # Construct Hessian matrix
        identity = np.eye(3)
        hessian = h_proj * proj_op + h_perp * (identity - proj_op)
        
        return hessian
    
    def get_bond_connectivity(self, coord, element_list):
        """Calculate bond connectivity matrix and related data"""
        n_atoms = len(coord)
        dist_mat = np.zeros((n_atoms, n_atoms))
        pair_cov_radii_mat = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coord[i] - coord[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
                
                cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                pair_cov_radii_mat[i, j] = pair_cov_radii_mat[j, i] = cov_sum
        
        # Bond connectivity matrix (True if bond exists between atoms)
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)  # No self-bonds
        
        return bond_mat, dist_mat, pair_cov_radii_mat
    
    def count_bonds_for_dihedral(self, bond_mat, central_atoms):
        """Count number of bonds for central atoms in a dihedral"""
        a, b = central_atoms
        # Sum bonds for both central atoms and subtract 2 (the bond between them is counted twice)
        bond_sum = bond_mat[a].sum() + bond_mat[b].sum() - 2
        return bond_sum
    
    def fischer_bond(self, coord, element_list):
        """Calculate Hessian components for bond stretching"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(b_c_mat)
        
        for idx in bond_indices:
            i, j = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_bond_force_const(r_ij, r_ij_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j]])
            r, b_vec = stretch2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
    
    def fischer_angle(self, coord, element_list):
        """Calculate Hessian components for angle bending"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        angle_indices = BC.angle_connect_table(b_c_mat)
        
        for idx in angle_indices:
            i, j, k = idx  # i-j-k angle
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_bend_force_const(r_ij, r_jk, r_ij_cov, r_jk_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k]])
            theta, b_vec = bend2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
    
    def fischer_dihedral(self, coord, element_list, bond_mat):
        """Calculate Hessian components for dihedral torsion"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        # Calculate bond count for central atoms in dihedrals
        tors_atom_bonds = {}
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            bond_sum = self.count_bonds_for_dihedral(bond_mat, (j, k))
            tors_atom_bonds[(j, k)] = bond_sum
        
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            
            # Central bond
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Get bond count for central atoms
            bond_sum = tors_atom_bonds.get((j, k), 0)
            
            # Calculate force constant using Fischer formula
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            tau, b_vec = torsion2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    self.cart_hess[3*l+n, 3*l+m] += force_const * b_vec[3][n] * b_vec[3][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*i+n, 3*l+m] += force_const * b_vec[0][n] * b_vec[3][m]
                    
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*l+m] += force_const * b_vec[1][n] * b_vec[3][m]
                    
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*l+m] += force_const * b_vec[2][n] * b_vec[3][m]
                    
                    self.cart_hess[3*l+n, 3*i+m] += force_const * b_vec[3][n] * b_vec[0][m]
                    self.cart_hess[3*l+n, 3*j+m] += force_const * b_vec[3][n] * b_vec[1][m]
                    self.cart_hess[3*l+n, 3*k+m] += force_const * b_vec[3][n] * b_vec[2][m]
    
    def d3_dispersion_hessian(self, coord, element_list, bond_mat):
        """Calculate Hessian correction based on D3 dispersion forces"""
        n_atoms = len(coord)
        
        # Calculate D3 dispersion correction for all atom pairs
        for i in range(n_atoms):
            for j in range(i):
                # Skip bonded atom pairs (already accounted for in Fischer model)
                if bond_mat[i, j]:
                    continue
                    
                # Calculate distance vector and magnitude
                r_vec = coord[i] - coord[j]
                r_ij = np.linalg.norm(r_vec)
                
                # Skip if atoms are too close
                if r_ij < 0.1:
                    continue
                
                # Calculate D3 Hessian contribution
                hess_block = self.d3_hessian_contribution(r_vec, r_ij, element_list[i], element_list[j])
                
                # Add to the Hessian matrix
                for n in range(3):
                    for m in range(3):
                        self.cart_hess[3*i+n, 3*i+m] += hess_block[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_block[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_block[n, m]
                        self.cart_hess[3*j+n, 3*j+m] += hess_block[n, m]
                        self.cart_hess[3*i+n, 3*j+m] -= hess_block[n, m]
                        self.cart_hess[3*j+n, 3*i+m] -= hess_block[n, m]
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate Hessian combining Fischer model and D3 dispersion correction
        
        Parameters:
            coord: Atomic coordinates (N×3 array, Bohr)
            element_list: List of element symbols
            cart_gradient: Gradient in Cartesian coordinates
            
        Returns:
            hess_proj: Hessian with rotational and translational modes projected out
        """
        print("Generating Hessian using Fischer model with D3 dispersion correction...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate bond connectivity matrix
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Calculate Hessian components from Fischer model
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        # Calculate Hessian components from D3 dispersion correction
        self.d3_dispersion_hessian(coord, element_list, bond_mat)
        
        # Symmetrize the lower triangle of the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out rotational and translational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj
    
class LindhApproxHessian:
    def __init__(self):
        #Lindh's approximate hessian
        #Ref: https://doi.org/10.1016/0009-2614(95)00646-L
        #Lindh, R., Chemical Physics Letters 1995, 241 (4), 423–428.
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.force_const_list = [0.45, 0.15, 0.005]  #bond, angle, dihedral_angle
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        return
    def LJ_force_const(self, elem_1, elem_2, coord_1, coord_2):
        eps_1 = UFF_VDW_well_depth_lib(elem_1)
        eps_2 = UFF_VDW_well_depth_lib(elem_2)
        sigma_1 = UFF_VDW_distance_lib(elem_1)
        sigma_2 = UFF_VDW_distance_lib(elem_2)
        eps = np.sqrt(eps_1 * eps_2)
        sigma = np.sqrt(sigma_1 * sigma_2)
        distance = np.linalg.norm(coord_1 - coord_2)
        LJ_force_const = -12 * eps * (-7*(sigma ** 6 / distance ** 8) + 13*(sigma ** 12 / distance ** 14))
        
        return LJ_force_const
    
    def electrostatic_force_const(self, elem_1, elem_2, coord_1, coord_2):
        effective_elec_charge = UFF_effective_charge_lib(elem_1) * UFF_effective_charge_lib(elem_2)
        distance = np.linalg.norm(coord_1 - coord_2)
        
        ES_force_const = 664.12 * (effective_elec_charge / distance ** 3) * (self.bohr2angstroms ** 2 / self.hartree2kcalmol)
        
        return ES_force_const#atom unit
    
    
    
    def return_lindh_const(self, element_1, element_2):
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        const_R_list = [[1.35, 2.10, 2.53],
                        [2.10, 2.87, 3.40],
                        [2.53, 3.40, 3.40]]
        
        const_alpha_list = [[1.0000, 0.3949, 0.3949],
                            [0.3949, 0.2800, 0.2800],
                            [0.3949, 0.2800, 0.2800]]      
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        
        if element_1 in first_period_table:
            idx_1 = 0
        elif element_1 in second_period_table:
            idx_1 = 1
        else:
            idx_1 = 2 
        
        if element_2 in first_period_table:
            idx_2 = 0
        elif element_2 in second_period_table:
            idx_2 = 1
        else:
            idx_2 = 2    
        
        #const_R = const_R_list[idx_1][idx_2]
        const_R = covalent_radii_lib(element_1) + covalent_radii_lib(element_2)
        const_alpha = const_alpha_list[idx_1][idx_2]
        
        return const_R, const_alpha

    def guess_lindh_hessian(self, coord, element_list):
        #coord: cartecian coord, Bohr (atom num × 3)
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        val_num = len(coord)*3
        connectivity_table = [BC.bond_connect_table(b_c_mat), BC.angle_connect_table(b_c_mat), BC.dihedral_angle_connect_table(b_c_mat)]
        #RIC_approx_diag_hessian = []
        RIC_approx_diag_hessian = [0.0 for i in range(self.RIC_variable_num)]
        RIC_idx_list = [[i[0], i[1]] for i in itertools.combinations([j for j in range(len(coord))] , 2)]
        
        for idx_list in connectivity_table:
            for idx in idx_list:
                force_const = self.force_const_list[len(idx)-2]
                for i in range(len(idx)-1):
                    elem_1 = element_list[idx[i]]
                    elem_2 = element_list[idx[i+1]]
                    const_R, const_alpha = self.return_lindh_const(elem_1, elem_2)
                    
                    R = np.linalg.norm(coord[idx[i]] - coord[idx[i+1]])
                    force_const *= np.exp(const_alpha * (const_R**2 - R**2)) 
                
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    mass_1 = atomic_mass(element_list[tmp_idx[0]]) 
                    mass_2 = atomic_mass(element_list[tmp_idx[1]])
                    
                    reduced_mass = (mass_1 * mass_2) / (mass_1 + mass_2)
                    
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    RIC_approx_diag_hessian[tmpnum] += force_const/reduced_mass
                  
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    RIC_approx_diag_hessian[tmpnum_1] += force_const
                    RIC_approx_diag_hessian[tmpnum_2] += force_const
                    
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    RIC_approx_diag_hessian[tmpnum_1] += force_const
                    RIC_approx_diag_hessian[tmpnum_2] += force_const
                    RIC_approx_diag_hessian[tmpnum_3] += force_const
                
                else:
                    print("error")
                    raise
                
        for num, pair in enumerate(RIC_idx_list):
            if pair in connectivity_table[0]:#bond connectivity
                continue#non bonding interaction
            RIC_approx_diag_hessian[num] += self.LJ_force_const(element_list[pair[0]], element_list[pair[1]], coord[pair[0]], coord[pair[1]])
            RIC_approx_diag_hessian[num] += self.electrostatic_force_const(element_list[pair[0]], element_list[pair[1]], coord[pair[0]], coord[pair[1]])
            
       
        
        RIC_approx_hessian = np.array(np.diag(RIC_approx_diag_hessian), dtype="float64")
        
        return RIC_approx_hessian

    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        
        print("generating Lindh's approximate hessian...")
        cart_gradient = cart_gradient.reshape(3*(len(cart_gradient)), 1)
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        int_grad = RedundantInternalCoordinates().cartgrad2RICgrad(cart_gradient, b_mat)
        int_approx_hess = self.guess_lindh_hessian(coord, element_list)
        BC = BondConnectivity()
        
        connnectivity = BC.connectivity_table(coord, element_list)
        #print(connnectivity, len(connnectivity[0])+len(connnectivity[1])+len(connnectivity[2]))
        cart_hess = RedundantInternalCoordinates().RIChess2carthess(coord, connnectivity, 
                                                                    int_approx_hess, b_mat, int_grad)
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        #eigenvalue, _ = np.linalg.eig(cart_hess)
        #print(sorted(eigenvalue))
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        return hess_proj#cart_hess

class SchlegelApproxHessian:
    def __init__(self):
        #Lindh's approximate hessian
        #ref: Journal of Molecular Structure: THEOCHEM Volumes 398–399, 30 June 1997, Pages 55-61
        #ref: Theoret. Chim. Acta (Berl.) 66, 333–340 (1984)
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
    
    def return_schlegel_const(self, element_1, element_2):
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        parameter_B_matrix = [[0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
                              [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
                              [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
                              [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
                              [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
                              [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
                              [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110]]# Bohr
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        third_period_table = ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br","Kr"]
        fourth_period_table = ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc","Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te","I", "Xe"]
        fifth_period_table = ["Cs", "Ba", "La","Ce","Pr","Nd","Pm","Sm", "Eu", "Gd", "Tb", "Dy" ,"Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
        if element_1 in first_period_table:
            idx_1 = 0
        elif element_1 in second_period_table:
            idx_1 = 1
        elif element_1 in third_period_table:
            idx_1 = 2
        elif element_1 in fourth_period_table:
            idx_1 = 3
        elif element_1 in fifth_period_table:
            idx_1 = 4
        else:
            idx_1 = 5 
        
        if element_2 in first_period_table:
            idx_2 = 0
        elif element_2 in second_period_table:
            idx_2 = 1
        elif element_2 in third_period_table:
            idx_2 = 2
        elif element_2 in fourth_period_table:
            idx_2 = 3
        elif element_2 in fifth_period_table:
            idx_2 = 4
        else:
            idx_2 = 5 
        
        
        const_b = parameter_B_matrix[idx_1][idx_2]
        
        return const_b
    
    
    def guess_schlegel_hessian(self, coord, element_list):
        #coord: cartecian coord, Bohr (atom num × 3)
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        val_num = len(coord)*3
        connectivity_table = [BC.bond_connect_table(b_c_mat), BC.angle_connect_table(b_c_mat), BC.dihedral_angle_connect_table(b_c_mat)]
        #RIC_approx_diag_hessian = []
        RIC_approx_diag_hessian = [0.0 for i in range(self.RIC_variable_num)]
        RIC_idx_list = [[i[0], i[1]] for i in itertools.combinations([j for j in range(len(coord))] , 2)]
        
        for idx_list in connectivity_table:
            for idx in idx_list:
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    distance = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    const_b = self.return_schlegel_const(elem_1, elem_2)   
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    F = 1.734 / (distance - const_b) ** 3
                    RIC_approx_diag_hessian[tmpnum] += F
                    
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    if elem_1 == "H" or elem_3 == "H":
                        RIC_approx_diag_hessian[tmpnum_1] += 0.160
                        RIC_approx_diag_hessian[tmpnum_2] += 0.160
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += 0.250
                        RIC_approx_diag_hessian[tmpnum_2] += 0.250
                                 
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    distance = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    bond_length = covalent_radii_lib(elem_2) + covalent_radii_lib(elem_3)
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    RIC_approx_diag_hessian[tmpnum_1] += 0.0023 -1* 0.07 * (distance - bond_length)
                    RIC_approx_diag_hessian[tmpnum_2] += 0.0023 -1* 0.07 * (distance - bond_length)
                    RIC_approx_diag_hessian[tmpnum_3] += 0.0023 -1* 0.07 * (distance - bond_length)

        
        RIC_approx_hessian = np.array(np.diag(RIC_approx_diag_hessian), dtype="float64")        
        return RIC_approx_hessian
    
    
    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        print("generating Schlegel's approximate hessian...")
        
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        
        int_approx_hess = self.guess_schlegel_hessian(coord, element_list)
        cart_hess = np.dot(b_mat.T, np.dot(int_approx_hess, b_mat))
        
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        #eigenvalue, _ = np.linalg.eig(cart_hess)
        #print(sorted(eigenvalue))
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        return hess_proj#cart_hess


class SchlegelD3ApproxHessian:
    def __init__(self):
        """
        Schlegel's approximate Hessian with D3 dispersion corrections and special handling for cyano groups
        References:
        - Schlegel: Journal of Molecular Structure: THEOCHEM Volumes 398–399, 30 June 1997, Pages 55-61
        - D3: S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys., 2010, 132, 154104
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # D3 dispersion parameters
        self.d3_params = D3Parameters()
        
        # Cyano group parameters - enhanced force constants
        self.cn_stretch_factor = 2.0  # Enhance stretch force constants for C≡N triple bond
        self.cn_angle_factor = 1.5    # Enhance angle force constants involving C≡N
        self.cn_torsion_factor = 0.5  # Reduce torsion force constants involving C≡N (more flexible)
        
    def detect_cyano_groups(self, coord, element_list):
        """Detect C≡N triple bonds in the structure"""
        cyano_atoms = []  # List of (C_idx, N_idx) tuples
        
        for i in range(len(coord)):
            if element_list[i] != 'C':
                continue
                
            for j in range(len(coord)):
                if i == j or element_list[j] != 'N':
                    continue
                    
                # Calculate distance between C and N
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Check if distance is close to a triple bond length
                cn_triple_bond = triple_covalent_radii_lib('C') + triple_covalent_radii_lib('N')
                
                if abs(r_ij - cn_triple_bond) < 0.3:  # Within 0.3 bohr of ideal length
                    # Check if C is connected to only one other atom (besides N)
                    connections_to_c = 0
                    for k in range(len(coord)):
                        if k == i or k == j:
                            continue
                            
                        r_ik = np.linalg.norm(coord[i] - coord[k])
                        cov_dist = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        if r_ik < 1.3 * cov_dist:  # Using 1.3 as a factor to account for bond length variations
                            connections_to_c += 1
                    
                    # If C has only one other connection, it's likely a terminal cyano group
                    if connections_to_c <= 1:
                        cyano_atoms.append((i, j))
        
        return cyano_atoms
    
    def return_schlegel_const(self, element_1, element_2):
        """Return Schlegel's constant for a given element pair"""
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        parameter_B_matrix = [
            [0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
            [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
            [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
            [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
            [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
            [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
            [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110]
        ]  # Bohr
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        third_period_table = ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
        fourth_period_table = ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br","Kr"]
        fifth_period_table = ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc","Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te","I", "Xe"]
        sixth_period_table = ["Cs", "Ba", "La","Ce","Pr","Nd","Pm","Sm", "Eu", "Gd", "Tb", "Dy" ,"Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
        
        if element_1 in first_period_table:
            idx_1 = 0
        elif element_1 in second_period_table:
            idx_1 = 1
        elif element_1 in third_period_table:
            idx_1 = 2
        elif element_1 in fourth_period_table:
            idx_1 = 3
        elif element_1 in fifth_period_table:
            idx_1 = 4
        elif element_1 in sixth_period_table:
            idx_1 = 5
        else:
            idx_1 = 6
        
        if element_2 in first_period_table:
            idx_2 = 0
        elif element_2 in second_period_table:
            idx_2 = 1
        elif element_2 in third_period_table:
            idx_2 = 2
        elif element_2 in fourth_period_table:
            idx_2 = 3
        elif element_2 in fifth_period_table:
            idx_2 = 4
        elif element_2 in sixth_period_table:
            idx_2 = 5
        else:
            idx_2 = 6
        
        const_b = parameter_B_matrix[idx_1][idx_2]
        return const_b
    
    def d3_damping_function(self, r_ij, r0):
        """Calculate D3 rational damping function"""
        a1 = self.d3_params.a1
        a2 = self.d3_params.a2
        
        # Rational damping function for C6 term
        damp = 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        return damp
    
    def get_d3_parameters(self, elem1, elem2):
        """Get D3 parameters for a pair of elements"""
        # Get R4/R2 values
        r4r2_1 = self.d3_params.get_r4r2(elem1)
        r4r2_2 = self.d3_params.get_r4r2(elem2)
        
        # C6 coefficients
        c6_1 = r4r2_1 ** 2
        c6_2 = r4r2_2 ** 2
        c6 = np.sqrt(c6_1 * c6_2)
        
        # C8 coefficients
        c8 = 3.0 * c6 * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (vdW radii)
        r0 = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6, c8, r0
    
    def calc_d3_correction(self, r_ij, elem1, elem2):
        """Calculate D3 dispersion correction to the force constant"""
        # Get D3 parameters
        c6, c8, r0 = self.get_d3_parameters(elem1, elem2)
        
        # Damping functions
        damp6 = self.d3_damping_function(r_ij, r0)
        
        # D3 energy contribution (simplified)
        e_disp = -self.d3_params.s6 * c6 / r_ij**6 * damp6
        
        # Approximate second derivative (force constant)
        fc_disp = self.d3_params.s6 * c6 * (42.0 / r_ij**8) * damp6
        
        return fc_disp * 0.01  # Scale factor to match overall Hessian scale
    
    def guess_schlegel_hessian(self, coord, element_list):
        """
        Calculate approximate Hessian using Schlegel's approach augmented with D3 dispersion
        and special handling for cyano groups
        """
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        # Setup connectivity tables using BondConnectivity utility
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        connectivity_table = [BC.bond_connect_table(b_c_mat), 
                              BC.angle_connect_table(b_c_mat), 
                              BC.dihedral_angle_connect_table(b_c_mat)]
        
        # Initialize RIC index list for all atom pairs
        RIC_idx_list = [[i[0], i[1]] for i in itertools.combinations(range(len(coord)), 2)]
        self.RIC_variable_num = len(RIC_idx_list)
        RIC_approx_diag_hessian = [0.0] * self.RIC_variable_num
        
        # Process connectivity table to build Hessian
        for idx_list in connectivity_table:
            for idx in idx_list:
                # Bond stretching terms
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    distance = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    const_b = self.return_schlegel_const(elem_1, elem_2)
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    
                    # Base Schlegel force constant
                    F = 1.734 / (distance - const_b) ** 3
                    
                    # Check if this is a cyano bond
                    is_cyano_bond = False
                    for c_idx, n_idx in cyano_atoms:
                        if (idx[0] == c_idx and idx[1] == n_idx) or (idx[0] == n_idx and idx[1] == c_idx):
                            is_cyano_bond = True
                            break
                    
                    # Add D3 dispersion contribution
                    d3_correction = self.calc_d3_correction(distance, elem_1, elem_2)
                    
                    if is_cyano_bond:
                        # Enhanced force constant for C≡N triple bond
                        RIC_approx_diag_hessian[tmpnum] += self.cn_stretch_factor * F + d3_correction
                    else:
                        RIC_approx_diag_hessian[tmpnum] += F + d3_correction
                
                # Angle bending terms
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    
                    # Check if angle involves cyano group
                    is_cyano_angle = (idx[0] in cyano_set or idx[1] in cyano_set or idx[2] in cyano_set)
                    
                    # Base Schlegel force constant
                    if elem_1 == "H" or elem_3 == "H":
                        F_angle = 0.160
                    else:
                        F_angle = 0.250
                    
                    # Add D3 dispersion contribution
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    d3_correction_1 = self.calc_d3_correction(d3_r1, elem_1, elem_2) * 0.2
                    d3_correction_2 = self.calc_d3_correction(d3_r2, elem_2, elem_3) * 0.2
                    
                    if is_cyano_angle:
                        # Enhanced angle force constants for angles involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_angle_factor * F_angle + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_angle_factor * F_angle + d3_correction_2
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_angle + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_angle + d3_correction_2
                
                # Torsion (dihedral) terms
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    elem_4 = element_list[idx[3]]
                    distance = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    bond_length = covalent_radii_lib(elem_2) + covalent_radii_lib(elem_3)
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    
                    # Base Schlegel torsion force constant
                    F_torsion = 0.0023 - 0.07 * (distance - bond_length)
                    
                    # Check if torsion involves cyano group
                    is_cyano_torsion = (idx[0] in cyano_set or idx[1] in cyano_set or 
                                        idx[2] in cyano_set or idx[3] in cyano_set)
                    
                    # Add D3 dispersion contribution
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    d3_r3 = np.linalg.norm(coord[idx[2]] - coord[idx[3]])
                    d3_correction_1 = self.calc_d3_correction(d3_r1, elem_1, elem_2) * 0.05
                    d3_correction_2 = self.calc_d3_correction(d3_r2, elem_2, elem_3) * 0.05
                    d3_correction_3 = self.calc_d3_correction(d3_r3, elem_3, elem_4) * 0.05
                    
                    if is_cyano_torsion:
                        # Reduced torsion force constants for torsions involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_torsion_factor * F_torsion + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_torsion_factor * F_torsion + d3_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += self.cn_torsion_factor * F_torsion + d3_correction_3
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_torsion + d3_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_torsion + d3_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += F_torsion + d3_correction_3
        
        # Convert to numpy array
        RIC_approx_hessian = np.diag(RIC_approx_diag_hessian).astype("float64")
        return RIC_approx_hessian
    
    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian"""
        print("Generating Schlegel's approximate Hessian with D3 dispersion correction...")
        
        # Calculate B matrix for redundant internal coordinates
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        # Calculate approximate Hessian in internal coordinates
        int_approx_hess = self.guess_schlegel_hessian(coord, element_list)
        
        # Convert to Cartesian coordinates
        cart_hess = np.dot(b_mat.T, np.dot(int_approx_hess, b_mat))
        
        # Handle NaN values
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        
        return hess_proj



class SchlegelD4ApproxHessian:
    def __init__(self):
        """
        Schlegel's approximate Hessian with D4 dispersion corrections and special handling for cyano groups
        References:
        - Schlegel: Journal of Molecular Structure: THEOCHEM Volumes 398–399, 30 June 1997, Pages 55-61
        - D4: E. Caldeweyher, C. Bannwarth, S. Grimme, J. Chem. Phys., 2017, 147, 034112
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # D4 dispersion parameters
        self.d4_params = D4Parameters()
        
        # Cyano group parameters - enhanced force constants
        self.cn_stretch_factor = 2.0   # Enhance stretch force constants for C≡N triple bond
        self.cn_angle_factor = 1.5     # Enhance angle force constants involving C≡N
        self.cn_torsion_factor = 0.5   # Reduce torsion force constants involving C≡N (more flexible)
        
    def detect_cyano_groups(self, coord, element_list):
        """Detect C≡N triple bonds in the structure"""
        cyano_atoms = []  # List of (C_idx, N_idx) tuples
        
        for i in range(len(coord)):
            if element_list[i] != 'C':
                continue
                
            for j in range(len(coord)):
                if i == j or element_list[j] != 'N':
                    continue
                    
                # Calculate distance between C and N
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Check if distance is close to a triple bond length
                cn_triple_bond = triple_covalent_radii_lib('C') + triple_covalent_radii_lib('N')
                
                if abs(r_ij - cn_triple_bond) < 0.3:  # Within 0.3 bohr of ideal length
                    # Check if C is connected to only one other atom (besides N)
                    connections_to_c = 0
                    for k in range(len(coord)):
                        if k == i or k == j:
                            continue
                            
                        r_ik = np.linalg.norm(coord[i] - coord[k])
                        cov_dist = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        if r_ik < 1.3 * cov_dist:  # Using 1.3 as a factor to account for bond length variations
                            connections_to_c += 1
                    
                    # If C has only one other connection, it's likely a terminal cyano group
                    if connections_to_c <= 1:
                        cyano_atoms.append((i, j))
        
        return cyano_atoms
    
    def calculate_coordination_numbers(self, coord, element_list):
        """Calculate atomic coordination numbers for D4 scaling"""
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                # Calculate distance
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Get covalent radii
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                
                # Coordination number contribution using counting function
                # k1 = 16.0, k2 = 4.0/3.0 (standard values from DFT-D4)
                k1 = 16.0
                k2 = 4.0/3.0
                r0 = r_cov_i + r_cov_j
                
                # Avoid overflow in exp
                if k1 * (r_ij / r0 - 1.0) > 25.0:
                    continue
                
                cn_contrib = 1.0 / (1.0 + np.exp(-k1 * (k2 * r0 / r_ij - 1.0)))
                cn[i] += cn_contrib
        
        return cn
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic charges using electronegativity equalization
        Simplified version for Hessian generation
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Calculate reference electronegativities
        en_list = [self.d4_params.get_electronegativity(elem) for elem in element_list]
        
        # Simple charge estimation based on electronegativity differences
        # This is a very simplified model
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coord[i] - coord[j])
                r_cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Only consider bonded atoms
                if r_ij < 1.3 * r_cov_sum:
                    en_diff = en_list[j] - en_list[i]
                    charge_transfer = 0.1 * en_diff  # Simple approximation
                    charges[i] += charge_transfer
                    charges[j] -= charge_transfer
        
        return charges
    
    def return_schlegel_const(self, element_1, element_2):
        """Return Schlegel's constant for a given element pair"""
        if type(element_1) is int:
            element_1 = number_element(element_1)
        if type(element_2) is int:
            element_2 = number_element(element_2)
        
        parameter_B_matrix = [
            [0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 0.9491],
            [0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190, 1.7190],
            [0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.3185],
            [0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206, 2.5206],
            [0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.5110],
            [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110],
            [0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110, 2.5110]
        ]  # Bohr
        
        first_period_table = ["H", "He"]
        second_period_table = ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        third_period_table = ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
        fourth_period_table = ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br","Kr"]
        fifth_period_table = ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc","Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te","I", "Xe"]
        sixth_period_table = ["Cs", "Ba", "La","Ce","Pr","Nd","Pm","Sm", "Eu", "Gd", "Tb", "Dy" ,"Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
        
        if element_1 in first_period_table:
            idx_1 = 0
        elif element_1 in second_period_table:
            idx_1 = 1
        elif element_1 in third_period_table:
            idx_1 = 2
        elif element_1 in fourth_period_table:
            idx_1 = 3
        elif element_1 in fifth_period_table:
            idx_1 = 4
        elif element_1 in sixth_period_table:
            idx_1 = 5
        else:
            idx_1 = 6
        
        if element_2 in first_period_table:
            idx_2 = 0
        elif element_2 in second_period_table:
            idx_2 = 1
        elif element_2 in third_period_table:
            idx_2 = 2
        elif element_2 in fourth_period_table:
            idx_2 = 3
        elif element_2 in fifth_period_table:
            idx_2 = 4
        elif element_2 in sixth_period_table:
            idx_2 = 5
        else:
            idx_2 = 6
        
        const_b = parameter_B_matrix[idx_1][idx_2]
        return const_b
    
    def d4_damping_function(self, r_ij, r0, order=6):
        """D4 rational damping function"""
        a1 = self.d4_params.a1
        a2 = self.d4_params.a2
        
        if order == 6:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        elif order == 8:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a2 * r0)) ** a1)
        return 0.0
    
    def charge_scale_factor(self, charge, element):
        """Calculate charge scaling factor for D4"""
        ga = self.d4_params.ga  # D4 charge scaling parameter (default=3.0)
        q_ref = self.d4_params.get_electronegativity(element)
        
        # Prevent numerical issues with large exponents
        exp_arg = -ga * abs(charge)
        if exp_arg < -50.0:  # Avoid underflow
            return 0.0
            
        return np.exp(exp_arg)
    
    def get_d4_parameters(self, elem1, elem2, q1=0.0, q2=0.0, cn1=None, cn2=None):
        """Get D4 parameters for a pair of elements with charge scaling"""
        # Get polarizabilities
        alpha1 = self.d4_params.get_polarizability(elem1)
        alpha2 = self.d4_params.get_polarizability(elem2)
        
        # Charge scaling
        qscale1 = self.charge_scale_factor(q1, elem1)
        qscale2 = self.charge_scale_factor(q2, elem2)
        
        # Get R4/R2 values
        r4r2_1 = self.d4_params.get_r4r2(elem1)
        r4r2_2 = self.d4_params.get_r4r2(elem2)
        
        # C6 coefficients with charge scaling
        c6_1 = alpha1 * r4r2_1 * qscale1
        c6_2 = alpha2 * r4r2_2 * qscale2
        c6_param = 2.0 * c6_1 * c6_2 / (c6_1 + c6_2)  # Effective C6 using harmonic mean
        
        # C8 coefficients
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (combined vdW radii)
        r0_param = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6_param, c8_param, r0_param
    
    def calc_d4_correction(self, r_ij, elem1, elem2, q1=0.0, q2=0.0, cn1=None, cn2=None):
        """Calculate D4 dispersion correction to the force constant"""
        # Get D4 parameters with charge scaling
        c6_param, c8_param, r0_param = self.get_d4_parameters(
            elem1, elem2, q1=q1, q2=q2, cn1=cn1, cn2=cn2
        )
        
        # Damping functions
        damp6 = self.d4_damping_function(r_ij, r0_param, order=6)
        damp8 = self.d4_damping_function(r_ij, r0_param, order=8)
        
        # D4 energy contribution 
        s6 = self.d4_params.s6
        s8 = self.d4_params.s8
        e_disp = -s6 * c6_param / r_ij**6 * damp6 - s8 * c8_param / r_ij**8 * damp8
        
        # Approximate second derivative (force constant)
        fc_disp = s6 * c6_param * (42.0 / r_ij**8) * damp6 + s8 * c8_param * (72.0 / r_ij**10) * damp8
        
        return fc_disp * 0.01  # Scale factor to match overall Hessian scale
    
    def calculate_three_body_term(self, coord, element_list, charges, cn):
        """Calculate three-body dispersion contribution"""
        n_atoms = len(coord)
        s9 = self.d4_params.s9
        
        # Skip if three-body term is turned off
        if abs(s9) < 1e-12:
            return np.zeros((3 * n_atoms, 3 * n_atoms))
        
        # Initialize three-body Hessian contribution
        three_body_hess = np.zeros((3 * n_atoms, 3 * n_atoms))
        
        # Loop over all atom triplets
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                for k in range(j+1, n_atoms):
                    # Get positions
                    r_i = coord[i]
                    r_j = coord[j]
                    r_k = coord[k]
                    
                    # Calculate interatomic distances
                    r_ij = np.linalg.norm(r_i - r_j)
                    r_jk = np.linalg.norm(r_j - r_k)
                    r_ki = np.linalg.norm(r_k - r_i)
                    
                    # Get coordination-number scaled C6 coefficients
                    c6_ij, _, r0_ij = self.get_d4_parameters(
                        element_list[i], element_list[j], 
                        q1=charges[i], q2=charges[j], 
                        cn1=cn[i], cn2=cn[j]
                    )
                    
                    c6_jk, _, r0_jk = self.get_d4_parameters(
                        element_list[j], element_list[k], 
                        q1=charges[j], q2=charges[k], 
                        cn1=cn[j], cn2=cn[k]
                    )
                    
                    c6_ki, _, r0_ki = self.get_d4_parameters(
                        element_list[k], element_list[i], 
                        q1=charges[k], q2=charges[i], 
                        cn1=cn[k], cn2=cn[i]
                    )
                    
                    # Calculate geometric mean of C6 coefficients
                    c9 = np.cbrt(c6_ij * c6_jk * c6_ki)
                    
                    # Calculate damping
                    damp_ij = self.d4_damping_function(r_ij, r0_ij)
                    damp_jk = self.d4_damping_function(r_jk, r0_jk)
                    damp_ki = self.d4_damping_function(r_ki, r0_ki)
                    damp = damp_ij * damp_jk * damp_ki
                    
                    # Skip if damping is too small
                    if damp < 1e-8:
                        continue
                    
                    # Calculate angle factor
                    r_ij_vec = r_j - r_i
                    r_jk_vec = r_k - r_j
                    r_ki_vec = r_i - r_k
                    
                    cos_ijk = np.dot(r_ij_vec, r_jk_vec) / (r_ij * r_jk)
                    cos_jki = np.dot(r_jk_vec, -r_ki_vec) / (r_jk * r_ki)
                    cos_kij = np.dot(-r_ki_vec, r_ij_vec) / (r_ki * r_ij)
                    
                    angle_factor = 1.0 + 3.0 * cos_ijk * cos_jki * cos_kij
                    
                    # Calculate three-body energy term
                    e_3 = -s9 * angle_factor * c9 * damp / (r_ij * r_jk * r_ki) ** 3
                    
                    # Approximate Hessian contribution (simplified)
                    # We use a small scaling factor to avoid dominating the Hessian
                    fc_scale = 0.002 * s9 * angle_factor * c9 * damp
                    
                    # Add approximate three-body contributions to Hessian
                    for n in range(3):
                        for m in range(3):
                            # Properly define all indices
                            idx_i_n = i * 3 + n
                            idx_j_n = j * 3 + n
                            idx_k_n = k * 3 + n
                            
                            idx_i_m = i * 3 + m
                            idx_j_m = j * 3 + m
                            idx_k_m = k * 3 + m
                            
                            # Diagonal blocks (diagonal atoms)
                            if n == m:
                                three_body_hess[idx_i_n, idx_i_n] += fc_scale / (r_ij**6) + fc_scale / (r_ki**6)
                                three_body_hess[idx_j_n, idx_j_n] += fc_scale / (r_ij**6) + fc_scale / (r_jk**6)
                                three_body_hess[idx_k_n, idx_k_n] += fc_scale / (r_jk**6) + fc_scale / (r_ki**6)
                            
                            # Off-diagonal blocks (between atoms)
                            three_body_hess[idx_i_n, idx_j_m] -= fc_scale / (r_ij**6)
                            three_body_hess[idx_j_m, idx_i_n] -= fc_scale / (r_ij**6)
                            
                            three_body_hess[idx_j_n, idx_k_m] -= fc_scale / (r_jk**6)
                            three_body_hess[idx_k_m, idx_j_n] -= fc_scale / (r_jk**6)
                            
                            three_body_hess[idx_k_n, idx_i_m] -= fc_scale / (r_ki**6)
                            three_body_hess[idx_i_m, idx_k_n] -= fc_scale / (r_ki**6)
        
        return three_body_hess
    
    def guess_schlegel_hessian(self, coord, element_list, charges, cn):
        """
        Calculate approximate Hessian using Schlegel's approach augmented with D4 dispersion
        and special handling for cyano groups
        """
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        # Setup connectivity tables using BondConnectivity utility
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        connectivity_table = [BC.bond_connect_table(b_c_mat), 
                              BC.angle_connect_table(b_c_mat), 
                              BC.dihedral_angle_connect_table(b_c_mat)]
        
        # Initialize RIC index list for all atom pairs
        RIC_idx_list = [[i[0], i[1]] for i in itertools.combinations(range(len(coord)), 2)]
        self.RIC_variable_num = len(RIC_idx_list)
        RIC_approx_diag_hessian = [0.0] * self.RIC_variable_num
        
        # Process connectivity table to build Hessian
        for idx_list in connectivity_table:
            for idx in idx_list:
                # Bond stretching terms
                if len(idx) == 2:
                    tmp_idx = sorted([idx[0], idx[1]])
                    distance = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    const_b = self.return_schlegel_const(elem_1, elem_2)
                    tmpnum = RIC_idx_list.index(tmp_idx)
                    
                    # Base Schlegel force constant
                    F = 1.734 / (distance - const_b) ** 3
                    
                    # Check if this is a cyano bond
                    is_cyano_bond = False
                    for c_idx, n_idx in cyano_atoms:
                        if (idx[0] == c_idx and idx[1] == n_idx) or (idx[0] == n_idx and idx[1] == c_idx):
                            is_cyano_bond = True
                            break
                    
                    # Add D4 dispersion contribution with charge scaling
                    d4_correction = self.calc_d4_correction(
                        distance, elem_1, elem_2, 
                        q1=charges[idx[0]], q2=charges[idx[1]],
                        cn1=cn[idx[0]], cn2=cn[idx[1]]
                    )
                    
                    if is_cyano_bond:
                        # Enhanced force constant for C≡N triple bond
                        RIC_approx_diag_hessian[tmpnum] += self.cn_stretch_factor * F + d4_correction
                    else:
                        RIC_approx_diag_hessian[tmpnum] += F + d4_correction
                
                # Angle bending terms
                elif len(idx) == 3:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    
                    # Check if angle involves cyano group
                    is_cyano_angle = (idx[0] in cyano_set or idx[1] in cyano_set or idx[2] in cyano_set)
                    
                    # Base Schlegel force constant
                    if elem_1 == "H" or elem_3 == "H":
                        F_angle = 0.160
                    else:
                        F_angle = 0.250
                    
                    # Add D4 dispersion contribution with charge scaling
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    
                    d4_correction_1 = self.calc_d4_correction(
                        d3_r1, elem_1, elem_2, 
                        q1=charges[idx[0]], q2=charges[idx[1]],
                        cn1=cn[idx[0]], cn2=cn[idx[1]]
                    ) * 0.2
                    
                    d4_correction_2 = self.calc_d4_correction(
                        d3_r2, elem_2, elem_3, 
                        q1=charges[idx[1]], q2=charges[idx[2]],
                        cn1=cn[idx[1]], cn2=cn[idx[2]]
                    ) * 0.2
                    
                    if is_cyano_angle:
                        # Enhanced angle force constants for angles involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_angle_factor * F_angle + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_angle_factor * F_angle + d4_correction_2
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_angle + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_angle + d4_correction_2
                
                # Torsion (dihedral) terms
                elif len(idx) == 4:
                    tmp_idx_1 = sorted([idx[0], idx[1]])
                    tmp_idx_2 = sorted([idx[1], idx[2]])
                    tmp_idx_3 = sorted([idx[2], idx[3]])
                    elem_1 = element_list[idx[0]]
                    elem_2 = element_list[idx[1]]
                    elem_3 = element_list[idx[2]]
                    elem_4 = element_list[idx[3]]
                    distance = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    bond_length = covalent_radii_lib(elem_2) + covalent_radii_lib(elem_3)
                    tmpnum_1 = RIC_idx_list.index(tmp_idx_1)
                    tmpnum_2 = RIC_idx_list.index(tmp_idx_2)
                    tmpnum_3 = RIC_idx_list.index(tmp_idx_3)
                    
                    # Base Schlegel torsion force constant
                    F_torsion = 0.0023 - 0.07 * (distance - bond_length)
                    
                    # Check if torsion involves cyano group
                    is_cyano_torsion = (idx[0] in cyano_set or idx[1] in cyano_set or 
                                        idx[2] in cyano_set or idx[3] in cyano_set)
                    
                    # Add D4 dispersion contribution with charge scaling
                    d3_r1 = np.linalg.norm(coord[idx[0]] - coord[idx[1]])
                    d3_r2 = np.linalg.norm(coord[idx[1]] - coord[idx[2]])
                    d3_r3 = np.linalg.norm(coord[idx[2]] - coord[idx[3]])
                    
                    d4_correction_1 = self.calc_d4_correction(
                        d3_r1, elem_1, elem_2, 
                        q1=charges[idx[0]], q2=charges[idx[1]],
                        cn1=cn[idx[0]], cn2=cn[idx[1]]
                    ) * 0.05
                    
                    d4_correction_2 = self.calc_d4_correction(
                        d3_r2, elem_2, elem_3, 
                        q1=charges[idx[1]], q2=charges[idx[2]],
                        cn1=cn[idx[1]], cn2=cn[idx[2]]
                    ) * 0.05
                    
                    d4_correction_3 = self.calc_d4_correction(
                        d3_r3, elem_3, elem_4, 
                        q1=charges[idx[2]], q2=charges[idx[3]],
                        cn1=cn[idx[2]], cn2=cn[idx[3]]
                    ) * 0.05
                    
                    if is_cyano_torsion:
                        # Reduced torsion force constants for torsions involving C≡N
                        RIC_approx_diag_hessian[tmpnum_1] += self.cn_torsion_factor * F_torsion + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += self.cn_torsion_factor * F_torsion + d4_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += self.cn_torsion_factor * F_torsion + d4_correction_3
                    else:
                        RIC_approx_diag_hessian[tmpnum_1] += F_torsion + d4_correction_1
                        RIC_approx_diag_hessian[tmpnum_2] += F_torsion + d4_correction_2
                        RIC_approx_diag_hessian[tmpnum_3] += F_torsion + d4_correction_3
        
        # Convert to numpy array
        RIC_approx_hessian = np.diag(RIC_approx_diag_hessian).astype("float64")
        return RIC_approx_hessian
    
    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian"""
        print("Generating Schlegel's approximate Hessian with D4 dispersion correction...")
        
        # Calculate coordination numbers and atomic charges for D4
        cn = self.calculate_coordination_numbers(coord, element_list)
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate B matrix for redundant internal coordinates
        b_mat = RedundantInternalCoordinates().B_matrix(coord)
        self.RIC_variable_num = len(b_mat)
        
        # Calculate approximate Hessian in internal coordinates
        int_approx_hess = self.guess_schlegel_hessian(coord, element_list, charges, cn)
        
        # Convert to Cartesian coordinates
        cart_hess = np.dot(b_mat.T, np.dot(int_approx_hess, b_mat))
        
        # Add three-body contribution (specific to D4)
        three_body_hess = self.calculate_three_body_term(coord, element_list, charges, cn)
        cart_hess += three_body_hess
        
        # Handle NaN values
        cart_hess = np.nan_to_num(cart_hess, nan=0.0)
        
        # Ensure Hessian is symmetric
        n = len(coord) * 3
        for i in range(n):
            for j in range(i):
                avg = (cart_hess[i, j] + cart_hess[j, i]) / 2
                cart_hess[i, j] = avg
                cart_hess[j, i] = avg
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(cart_hess, element_list, coord)
        
        print("D4 dispersion correction applied successfully")
        return hess_proj
    
def calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij):
   
    x_ij = x_ij + 1.0e-12
    y_ij = y_ij + 1.0e-12
    z_ij = z_ij + 1.0e-12
    a_vdw = 20.0
    t_1 = D2_S6_parameter * C6_param_ij
    t_2 = x_ij ** 2 
    t_3 = y_ij ** 2 
    t_4 = z_ij ** 2
    t_5 = t_2 + t_3 + t_4
    t_6 = t_5 ** 2
    t_7 = t_6 ** 2 
    t_10 = t_5 ** 0.5
    t_11 = 0.1 / C6_VDW_ij
   
    t_15 = np.exp(-1 * a_vdw * (t_10 * t_11 - 0.1)) + 1.0e-12
    t_16 = 0.1 + t_15
    t_17 = 0.1 / t_16
    t_24 = t_16 ** 2
    t_25 = 0.1 / t_24
    t_29 = t_11 * t_15
    t_33 = 0.1 / t_7
    t_41 = a_vdw ** 2
    t_42 = C6_VDW_ij ** 2
    t_44 = t_41 / t_42
    t_45 = t_15 ** 2
    #print(t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_10, t_11, t_15, t_16, t_17, t_24, t_25, t_29, t_33, t_41, t_42, t_44, t_45)
    t_62 = -0.48 * t_1 / t_7 / t_5 * t_17 * t_2 + 0.13 * t_1 / t_10 / t_7 *  t_25 * t_2 * a_vdw * t_29 + 0.6 * t_1 * t_33 * t_17 - 0.2 * t_1 * t_33 / t_24 / t_16 * t_44 * t_2 * t_45 - t_1 / t_10 / t_6 / t_5 * t_25 * a_vdw * t_29 + t_1 * t_33 * t_25 * t_44 * t_2 * t_15
    
    return t_62

def calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij):
    a_vdw = 20.0
    x_ij = x_ij + 1.0e-12
    y_ij = y_ij + 1.0e-12
    z_ij = z_ij + 1.0e-12
    
    t_1 = D2_S6_parameter * C6_param_ij
    t_2 = x_ij ** 2
    t_3 = y_ij ** 2 
    t_4 = z_ij ** 2 
    t_5 = t_2 + t_3 + t_4
    t_6 = t_5 ** 2
    t_7 = t_6 ** 2
    t_11 = t_5 ** 0.5
    t_12 = 0.1 / C6_VDW_ij
   
    t_16 = np.exp(-1 * a_vdw * (t_11 * t_12 - 0.1)) + 1.0e-12
    t_17 = 0.1 + t_16 
    t_25 = t_17 ** 2 
    t_26 = 0.1 / t_25
    t_35 = 0.1 / t_7
    t_40 = a_vdw ** 2
    t_41 = C6_VDW_ij ** 2
    t_43 = t_40 / t_41
    t_44 = t_16 ** 2
    #print(t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_11, t_12, t_16, t_17, t_25, t_26, t_35, t_40, t_41, t_43, t_44)
    t_56 = -0.48 * t_1 / t_7 / t_5 / t_17 * x_ij * y_ij + 0.13 * t_1 / t_11 / t_7 * t_26 * x_ij * a_vdw * t_12 * y_ij * t_16 - 0.2 * t_1 * t_35 / t_25 / t_17 * t_43 * x_ij * t_44 * y_ij + t_1 * t_35 * t_26 * t_43 * x_ij * y_ij * t_16
    #print(t_16, t_56)
    return t_56
    
    
def outofplane2(t_xyz):
    r_1 = t_xyz[0] - t_xyz[3]
    q_41 = np.linalg.norm(r_1)
    e_41 = r_1 / q_41

    r_2 = t_xyz[1] - t_xyz[3]
    q_42 = np.linalg.norm(r_2)
    e_42 = r_2 / q_42

    r_3 = t_xyz[2] - t_xyz[3]
    q_43 = np.linalg.norm(r_3)
    e_43 = r_3 / q_43

    cosfi1 = np.dot(e_43, e_42)

    fi_1 = np.arccos(cosfi1)

    if abs(fi_1 - np.pi) < 1e-12:
        teta = 0.0
        bt = 0.0
        return teta, bt

    cosfi2 = np.dot(e_41, e_43)
    fi_2 = np.arccos(cosfi2)
    #deg_fi_2 = 180.0 * fi_2 / np.pi

    cosfi3 = np.dot(e_41, e_42)

    fi_3 = np.arccos(cosfi3)
    #deg_fi_3 = 180.0 * fi_3 / np.pi

    c14 = np.zeros((3, 3))

    c14[0] = t_xyz[0]
    c14[1] = t_xyz[3]

    r_42 = t_xyz[1] - t_xyz[3]
    r_43 = t_xyz[2] - t_xyz[3]
    c14[2][0] = r_42[1] * r_43[2] - r_42[2] * r_43[1]
    c14[2][1] = r_42[2] * r_43[0] - r_42[0] * r_43[2]
    c14[2][2] = r_42[0] * r_43[1] - r_42[1] * r_43[0]

    if ((c14[2][0] ** 2 + c14[2][1] ** 2 + c14[2][2] ** 2) < 1e-12):
        teta = 0.0
        bt = 0.0
        return teta, bt
    c14[2][0] = c14[2][0] + t_xyz[3][0]
    c14[2][1] = c14[2][1] + t_xyz[3][1]
    c14[2][2] = c14[2][2] + t_xyz[3][2]

    teta, br_14 = bend2(c14)

    teta = teta -0.5 * np.pi

    bt = np.zeros((4, 3))

    for i_x in range(1, 4):
        i_y = (i_x + 1) % 4 + (i_x + 1) // 4
        i_z = (i_y + 1) % 4 + (i_y + 1) // 4
        
        bt[0][i_x-1] = -1 * br_14[2][i_x-1]
        bt[1][i_x-1] = -1 * br_14[2][i_y-1]
        bt[2][i_x-1] = -1 * br_14[2][i_z-1]
        bt[3][i_x-1] = -1 * (bt[0][i_x-1] + bt[1][i_x-1] + bt[2][i_x-1])
        
    bt *= -1.0 

    return teta, bt# angle, move_vector

def torsion2(t_xyz):
    r_ij_1, b_r_ij = stretch2(t_xyz[0:2])
    r_ij_2, b_r_jk = stretch2(t_xyz[1:3])
    r_ij_3, b_r_kl = stretch2(t_xyz[2:4])

    fi_2, b_fi_2 = bend2(t_xyz[0:3])
    fi_3, b_fi_3 = bend2(t_xyz[1:4])
    sin_fi_2 = np.sin(fi_2)
    sin_fi_3 = np.sin(fi_3)
    cos_fi_2 = np.cos(fi_2)
    cos_fi_3 = np.cos(fi_3)
    costau = ( (b_r_ij[0][1] * b_r_jk[1][2] - b_r_jk[1][1] * b_r_ij[0][2])) * (b_r_jk[0][1] * b_r_kl[1][2] - b_r_jk[0][2] * b_r_kl[1][1]) + (b_r_ij[0][2] * b_r_jk[1][0] - b_r_ij[0][0] * b_r_jk[1][2]) * (b_r_jk[0][2] * b_r_kl[1][0] - b_r_jk[0][0] * b_r_kl[1][2]) + (b_r_ij[0][0] * b_r_jk[1][1] - b_r_ij[0][1] * b_r_jk[1][0]) * (b_r_jk[0][0] * b_r_kl[1][1] - b_r_jk[0][1] * b_r_kl[1][0]) / (sin_fi_2 * sin_fi_3)
    sintau = (b_r_ij[1][0] * (b_r_jk[0][1] * b_r_kl[1][2] - b_r_jk[0][2] * b_r_kl[1][1]) + b_r_ij[1][1] * (b_r_jk[0][2] * b_r_kl[1][0] - b_r_jk[0][0] * b_r_kl[1][2]) + b_r_ij[1][2] * (b_r_jk[0][0] * b_r_kl[1][1] - b_r_jk[0][1] * b_r_kl[1][0])) / (sin_fi_2 * sin_fi_3)

    tau = np.arctan2(sintau, costau)

    if abs(tau) == np.pi:
        tau = np.pi

    bt = np.zeros((4, 3))
    for i_x in range(1,4):
        i_y = i_x + 1
        if i_y > 3:
            i_y = i_y - 3
        i_z = i_y + 1
        if i_z > 3:
            i_z = i_z - 3
        bt[0][i_x-1] = (b_r_ij[1][i_y-1] * b_r_jk[1][i_z-1] - b_r_ij[1][i_z-1] * b_r_jk[1][i_y-1]) / (r_ij_1 * sin_fi_2 ** 2)
        bt[3][i_x-1] = (b_r_kl[0][i_y-1] * b_r_jk[0][i_z-1] - b_r_kl[0][i_z-1] * b_r_jk[0][i_y-1]) / (r_ij_3 * sin_fi_3 ** 2)
        bt[1][i_x-1] = -1 * ((r_ij_2 - r_ij_1 * cos_fi_2) * bt[0][i_x-1] + r_ij_3 * cos_fi_3 * bt[3][i_x-1]) / r_ij_2
        bt[2][i_x-1] = -1 * (bt[0][i_x-1] + bt[1][i_x-1] + bt[3][i_x-1])

    for ix in range(1, 4):
        iy = ix + 1
        if iy > 3:
            iy = iy - 3
        iz = iy + 1
        if iz > 3:
            iz = iz - 3
        bt[0][ix-1] = (b_r_ij[1][iy-1] * b_r_jk[1][iz-1] - b_r_ij[1][iz-1] * b_r_jk[1][iy-1]) / (r_ij_1 * sin_fi_2 ** 2)
        bt[3][ix-1] = (b_r_kl[0][iy-1] * b_r_jk[0][iz-1] - b_r_kl[0][iz-1] * b_r_jk[0][iy-1]) / (r_ij_3 * sin_fi_3 ** 2)
        bt[1][ix-1] = -1 * ((r_ij_2 - r_ij_1 * cos_fi_2) * bt[0][ix-1] + r_ij_3 * cos_fi_3 * bt[3][ix-1]) / r_ij_2
        bt[2][ix-1] = -1 * (bt[0][ix-1] + bt[1][ix-1] + bt[3][ix-1])

    return tau, bt # angle, move_vector

def bend2(t_xyz):
    r_ij_1, b_r_ij = stretch2(t_xyz[0:2])
    r_jk_1, b_r_jk = stretch2(t_xyz[1:3])
    c_o = 0.0
    crap = 0.0
    for i in range(3):
        c_o += b_r_ij[0][i] * b_r_jk[1][i]
        crap += b_r_ij[0][i] ** 2
        crap += b_r_jk[1][i] ** 2

    if np.sqrt(crap) < 1e-12:
        fir = np.pi - np.arcsin(np.sqrt(crap))
        si = np.sqrt(crap)
    else:
        fir = np.arccos(c_o)
        si = np.sqrt(1 - c_o ** 2)

    if np.abs(fir - np.pi) < 1e-12:
        fir = np.pi

    bf = np.zeros((3, 3))
    for i in range(3):
        bf[0][i] = (c_o * b_r_ij[0][i] - b_r_jk[1][i]) / (r_ij_1 * si)
        bf[2][i] = (c_o * b_r_jk[1][i] - b_r_ij[0][i]) / (r_jk_1 * si)
        bf[1][i] = -1 * (bf[0][i] + bf[2][i])

    return fir, bf # angle, move_vector

def stretch2(t_xyz):
    dist = t_xyz[0] - t_xyz[1]
    norm_dist = np.linalg.norm(dist)
    b = np.zeros((2,3))
    b[0] = -1 * dist / norm_dist
    b[1] = dist / norm_dist
    return norm_dist, b # distance, move_vectors (unit_vector)


class SwartD4ApproxHessian:
    def __init__(self):
        #Swart's Model Hessian augmented with D4 dispersion
        #ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536–2544.
        #ref.: E. Caldeweyher, C. Bannwarth, S. Grimme, J. Chem. Phys., 2017, 147, 034112
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00  # Dispersion scaling factor
        
        self.kr = 0.35  # Bond stretching force constant scaling
        self.kf = 0.15  # Angle bending force constant scaling
        self.kt = 0.005 # Torsional force constant scaling
        
        self.cutoff = 70.0  # Cutoff for long-range interactions
        self.eps = 1.0e-12  # Small number for numerical stability
        
        # D4 parameters
        self.d4_params = D4Parameters()
        
        # Cyano group parameters
        self.cn_kr = 0.70  # Enhanced force constant for C≡N triple bond
        self.cn_kf = 0.20  # Enhanced force constant for angles involving C≡N
        self.cn_kt = 0.002 # Reduced force constant for torsions involving C≡N
        return
    
    def detect_cyano_groups(self, coord, element_list):
        """Detect C≡N triple bonds in the structure"""
        cyano_atoms = []  # List of (C_idx, N_idx) tuples
        
        for i in range(len(coord)):
            if element_list[i] != 'C':
                continue
                
            for j in range(len(coord)):
                if i == j or element_list[j] != 'N':
                    continue
                    
                # Calculate distance between C and N
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij = np.sqrt(x_ij**2 + y_ij**2 + z_ij**2)
                
                # Check if distance is close to a triple bond length
                cn_triple_bond = triple_covalent_radii_lib('C') + triple_covalent_radii_lib('N')
                
                if abs(r_ij - cn_triple_bond) < 0.3:  # Within 0.3 bohr of ideal length
                    # Check if C is connected to only one other atom (besides N)
                    connections_to_c = 0
                    for k in range(len(coord)):
                        if k == i or k == j:
                            continue
                            
                        x_ik = coord[i][0] - coord[k][0]
                        y_ik = coord[i][1] - coord[k][1]
                        z_ik = coord[i][2] - coord[k][2]
                        r_ik = np.sqrt(x_ik**2 + y_ik**2 + z_ik**2)
                        
                        cov_dist = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        if r_ik < 1.3 * cov_dist:  # Using 1.3 as a factor to account for bond length variations
                            connections_to_c += 1
                    
                    # If C has only one other connection, it's likely a terminal cyano group
                    if connections_to_c <= 1:
                        cyano_atoms.append((i, j))
        
        return cyano_atoms
    
    def calculate_coordination_numbers(self, coord, element_list):
        """Calculate atomic coordination numbers for D4 scaling"""
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                # Calculate distance
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Get covalent radii
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                
                # Coordination number contribution using counting function
                # k1 = 16.0, k2 = 4.0/3.0 (standard values from DFT-D4)
                k1 = 16.0
                k2 = 4.0/3.0
                r0 = r_cov_i + r_cov_j
                
                # Avoid overflow in exp
                if k1 * (r_ij / r0 - 1.0) > 25.0:
                    continue
                
                cn_contrib = 1.0 / (1.0 + np.exp(-k1 * (k2 * r0 / r_ij - 1.0)))
                cn[i] += cn_contrib
        
        return cn
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic charges using electronegativity equalization
        Simplified version for Hessian generation
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Calculate reference electronegativities
        en_list = [self.d4_params.get_electronegativity(elem) for elem in element_list]
        
        # Simple charge estimation based on electronegativity differences
        # This is a very simplified model
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coord[i] - coord[j])
                r_cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Only consider bonded atoms
                if r_ij < 1.3 * r_cov_sum:
                    en_diff = en_list[j] - en_list[i]
                    charge_transfer = 0.1 * en_diff  # Simple approximation
                    charges[i] += charge_transfer
                    charges[j] -= charge_transfer
        
        return charges
    
    def calc_force_const(self, alpha, covalent_length, distance):
        """Calculate force constant with exponential damping"""
        force_const = np.exp(-1 * alpha * (distance / covalent_length - 1.0))
        return force_const
        
    def calc_vdw_force_const(self, alpha, r0, distance):
        """Calculate van der Waals force constant with exponential damping"""
        vdw_force_const = np.exp(-1 * alpha * (r0 - distance) ** 2)
        return vdw_force_const
        
    def d4_damping_function(self, r_ij, r0, order=6):
        """D4 rational damping function"""
        a1 = self.d4_params.a1
        a2 = self.d4_params.a2
        
        if order == 6:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        elif order == 8:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a2 * r0)) ** a1)
        return 0.0
    
    def charge_scale_factor(self, charge, element):
        """Calculate charge scaling factor for D4"""
        ga = self.d4_params.ga  # D4 charge scaling parameter
        q_ref = self.d4_params.get_electronegativity(element)
        
        # Prevent numerical issues with large exponents
        exp_arg = -ga * abs(charge)
        if exp_arg < -50.0:  # Avoid underflow
            return 0.0
            
        return np.exp(exp_arg)
    
    def get_d4_parameters(self, elem1, elem2, q1=0.0, q2=0.0, cn1=None, cn2=None):
        """Get D4 parameters for a pair of elements with charge scaling"""
        # Get polarizabilities
        alpha1 = self.d4_params.get_polarizability(elem1)
        alpha2 = self.d4_params.get_polarizability(elem2)
        
        # Charge scaling
        qscale1 = self.charge_scale_factor(q1, elem1)
        qscale2 = self.charge_scale_factor(q2, elem2)
        
        # Get R4/R2 values
        r4r2_1 = self.d4_params.get_r4r2(elem1)
        r4r2_2 = self.d4_params.get_r4r2(elem2)
        
        # C6 coefficients with charge scaling
        c6_1 = alpha1 * r4r2_1 * qscale1
        c6_2 = alpha2 * r4r2_2 * qscale2
        c6_param = 2.0 * c6_1 * c6_2 / (c6_1 + c6_2)  # Effective C6 using harmonic mean
        
        # C8 coefficients
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (combined vdW radii)
        r0_param = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6_param, c8_param, r0_param
    
    def calc_d4_force_const(self, r_ij, c6_param, c8_param, r0_param):
        """Calculate D4 dispersion force constant"""
        s6 = self.d4_params.s6
        s8 = self.d4_params.s8
        
        # Apply damping functions
        damp6 = self.d4_damping_function(r_ij, r0_param, order=6)
        damp8 = self.d4_damping_function(r_ij, r0_param, order=8)
        
        # Energy terms (negative because dispersion is attractive)
        e6 = -s6 * c6_param / r_ij ** 6 * damp6
        e8 = -s8 * c8_param / r_ij ** 8 * damp8
        
        # Force constant is the second derivative of energy
        fc6 = s6 * c6_param * (42.0 / r_ij ** 8) * damp6
        fc8 = s8 * c8_param * (72.0 / r_ij ** 10) * damp8
        
        return fc6 + fc8
    
    def swart_bond(self, coord, element_list, charges, cn):
        """Calculate bond stretching contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for i in range(len(coord)):
            for j in range(i):
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get D4 parameters with charge scaling
                c6_param, c8_param, r0_param = self.get_d4_parameters(
                    element_list[i], element_list[j], 
                    q1=charges[i], q2=charges[j], 
                    cn1=cn[i], cn2=cn[j]
                )
                
                # Calculate D4 dispersion contribution
                d4_force_const = self.calc_d4_force_const(r_ij, c6_param, c8_param, r0_param)
                
                # Check if this is a cyano bond
                is_cyano_bond = False
                for c_idx, n_idx in cyano_atoms:
                    if (i == c_idx and j == n_idx) or (i == n_idx and j == c_idx):
                        is_cyano_bond = True
                        break
                
                # Apply appropriate force constant
                if is_cyano_bond:
                    # Special force constant for C≡N triple bond
                    g_mm = self.cn_kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * d4_force_const
                else:
                    # Regular Swart force constant with D4 dispersion
                    g_mm = self.kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * d4_force_const
                
                # Calculate Hessian components
                hess_xx = g_mm * x_ij ** 2 / r_ij_2
                hess_xy = g_mm * x_ij * y_ij / r_ij_2
                hess_xz = g_mm * x_ij * z_ij / r_ij_2
                hess_yy = g_mm * y_ij ** 2 / r_ij_2
                hess_yz = g_mm * y_ij * z_ij / r_ij_2
                hess_zz = g_mm * z_ij ** 2 / r_ij_2
                
                # Fill the Hessian matrix
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3] -= hess_xz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
                
                self.cart_hess[j * 3][i * 3] -= hess_xx
                self.cart_hess[j * 3][i * 3 + 1] -= hess_xy
                self.cart_hess[j * 3][i * 3 + 2] -= hess_xz
                self.cart_hess[j * 3 + 1][i * 3] -= hess_xy
                self.cart_hess[j * 3 + 1][i * 3 + 1] -= hess_yy
                self.cart_hess[j * 3 + 1][i * 3 + 2] -= hess_yz
                self.cart_hess[j * 3 + 2][i * 3] -= hess_xz
                self.cart_hess[j * 3 + 2][i * 3 + 1] -= hess_yz
                self.cart_hess[j * 3 + 2][i * 3 + 2] -= hess_zz
        
        return
    
    def swart_angle(self, coord, element_list, charges, cn):
        """Calculate angle bending contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        for i in range(len(coord)):
            for j in range(len(coord)):
                if i == j:
                    continue
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get D4 parameters with charge scaling for i-j pair
                c6_ij, c8_ij, r0_ij = self.get_d4_parameters(
                    element_list[i], element_list[j], 
                    q1=charges[i], q2=charges[j], 
                    cn1=cn[i], cn2=cn[j]
                )
                
                for k in range(j):
                    if i == k:
                        continue
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    # Check for linear arrangement (cos_theta ~ 1.0)
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    # Get D4 parameters with charge scaling for i-k pair
                    c6_ik, c8_ik, r0_ik = self.get_d4_parameters(
                        element_list[i], element_list[k], 
                        q1=charges[i], q2=charges[k], 
                        cn1=cn[i], cn2=cn[k]
                    )
                    
                    # Calculate D4 dispersion contributions
                    d4_ij = self.calc_d4_force_const(r_ij, c6_ij, c8_ij, r0_ij)
                    d4_ik = self.calc_d4_force_const(r_ik, c6_ik, c8_ik, r0_ik)
                    
                    # Calculate bond force constants with D4 dispersion
                    g_ij = self.calc_force_const(1.0, covalent_length_ij, r_ij) + 0.5 * self.kd * d4_ij
                    g_ik = self.calc_force_const(1.0, covalent_length_ik, r_ik) + 0.5 * self.kd * d4_ik
                    
                    # Check if angle involves cyano group
                    is_cyano_angle = (i in cyano_set or j in cyano_set or k in cyano_set)
                    
                    # Apply appropriate force constant
                    if is_cyano_angle:
                        # Special force constant for angles involving cyano groups
                        g_jk = self.cn_kf * g_ij * g_ik
                    else:
                        # Regular Swart force constant
                        g_jk = self.kf * g_ij * g_ik
                    
                    # Calculate cross product for sin(theta)
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:
                        cos_theta = (r_ij_2 + r_ik_2 - r_jk_2) / (2.0 * r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                       
                        if sin_theta > self.eps: # non-linear
                            # Calculate derivatives for non-linear case
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            # Update Hessian for non-linear case
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                    
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            # Update diagonal blocks
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            # Special handling for linear arrangements
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            # Iterate over two perpendicular directions
                            for ii in range(2):
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                # Update Hessian for linear case
                                for l in range(3):
                                    for m in range(3):
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                # Update diagonal blocks for linear case
                                for l in range(3):
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass  # Skip if any distance is too small
        
        # Make the Hessian symmetric for angle terms
        n_basis = len(coord) * 3
        for i in range(n_basis):
            for j in range(i):
                if abs(self.cart_hess[i][j] - self.cart_hess[j][i]) > 1.0e-10:
                    avg = (self.cart_hess[i][j] + self.cart_hess[j][i]) / 2.0
                    self.cart_hess[i][j] = avg
                    self.cart_hess[j][i] = avg
                
        return
    
    def swart_dihedral_angle(self, coord, element_list, charges, cn):
        """Calculate dihedral angle contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        r_jk = coord[j] - coord[k]
                        r_kl = coord[k] - coord[l]
                        
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        # Calculate vector magnitudes
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        # Skip if angle is too shallow (less than 35 degrees)
                        a35 = (35.0/180)* np.pi
                        cosfi_max = np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        
                        # Get D4 parameters for each atom pair with charge scaling
                        c6_ij, c8_ij, r0_ij = self.get_d4_parameters(
                            element_list[i], element_list[j], 
                            q1=charges[i], q2=charges[j], 
                            cn1=cn[i], cn2=cn[j]
                        )
                        
                        c6_jk, c8_jk, r0_jk = self.get_d4_parameters(
                            element_list[j], element_list[k], 
                            q1=charges[j], q2=charges[k], 
                            cn1=cn[j], cn2=cn[k]
                        )
                        
                        c6_kl, c8_kl, r0_kl = self.get_d4_parameters(
                            element_list[k], element_list[l], 
                            q1=charges[k], q2=charges[l], 
                            cn1=cn[k], cn2=cn[l]
                        )
                        
                        # Calculate D4 dispersion contributions
                        d4_ij = self.calc_d4_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        d4_jk = self.calc_d4_force_const(norm_r_jk, c6_jk, c8_jk, r0_jk)
                        d4_kl = self.calc_d4_force_const(norm_r_kl, c6_kl, c8_kl, r0_kl)
                        
                        # Calculate bond force constants with D4 dispersion
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * d4_ij
                        g_jk = self.calc_force_const(1.0, covalent_length_jk, norm_r_jk) + 0.5 * self.kd * d4_jk
                        g_kl = self.calc_force_const(1.0, covalent_length_kl, norm_r_kl) + 0.5 * self.kd * d4_kl
                        
                        # Check if torsion involves cyano group
                        is_cyano_torsion = False
                        if i in cyano_set or j in cyano_set or k in cyano_set or l in cyano_set:
                            is_cyano_torsion = True
                        
                        # Adjust force constant for cyano groups - they have flatter torsional potentials
                        if is_cyano_torsion:
                            t_ij = self.cn_kt * g_ij * g_jk * g_kl
                        else:
                            t_ij = self.kt * g_ij * g_jk * g_kl
                        
                        # Calculate torsion angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        # Update Hessian with torsional contributions
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] += t_ij * s_k[n] * s_l[m]
                            
                        # Update diagonal blocks
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] += t_ij * s_l[n] * s_l[m]
                       
        return
    
    def swart_out_of_plane(self, coord, element_list, charges, cn):
        """Calculate out-of-plane bending contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        r_ik = coord[i] - coord[k]
                        r_il = coord[i] - coord[l]
                        
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        # Skip if atoms are nearly collinear
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        # Get D4 parameters with charge scaling for each atom pair
                        c6_ij, c8_ij, r0_ij = self.get_d4_parameters(
                            element_list[i], element_list[j], 
                            q1=charges[i], q2=charges[j], 
                            cn1=cn[i], cn2=cn[j]
                        )
                        
                        c6_ik, c8_ik, r0_ik = self.get_d4_parameters(
                            element_list[i], element_list[k], 
                            q1=charges[i], q2=charges[k], 
                            cn1=cn[i], cn2=cn[k]
                        )
                        
                        c6_il, c8_il, r0_il = self.get_d4_parameters(
                            element_list[i], element_list[l], 
                            q1=charges[i], q2=charges[l], 
                            cn1=cn[i], cn2=cn[l]
                        )
                        
                        # Calculate D4 dispersion contributions
                        d4_ij = self.calc_d4_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        d4_ik = self.calc_d4_force_const(norm_r_ik, c6_ik, c8_ik, r0_ik)
                        d4_il = self.calc_d4_force_const(norm_r_il, c6_il, c8_il, r0_il)
                        
                        # Calculate bond force constants with D4 dispersion
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * d4_ij
                        g_ik = self.calc_force_const(1.0, covalent_length_ik, norm_r_ik) + 0.5 * self.kd * d4_ik
                        g_il = self.calc_force_const(1.0, covalent_length_il, norm_r_il) + 0.5 * self.kd * d4_il
                        
                        # Check if any atom is part of a cyano group
                        is_cyano_involved = (i in cyano_set or j in cyano_set or 
                                            k in cyano_set or l in cyano_set)
                        
                        # Adjust force constant if cyano group is involved
                        if is_cyano_involved:
                            t_ij = 0.5 * self.kt * g_ij * g_ik * g_il  # Reduce force constant for cyano
                        else:
                            t_ij = self.kt * g_ij * g_ik * g_il
                        
                        # Calculate out-of-plane angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update Hessian with out-of-plane contributions
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                        # Update diagonal blocks
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                        
        return
    
    def calculate_three_body_term(self, coord, element_list, charges, cn):
        """Calculate three-body dispersion contribution to the Hessian (D4 specific)"""
        s9 = self.d4_params.s9  # Scaling parameter for three-body term
        if abs(s9) < 1e-12:
            return  # Skip if three-body term is turned off
        
        n_atoms = len(coord)
        
        # Loop over all atom triplets
        for i in range(n_atoms):
            for j in range(i):
                for k in range(j):
                    # Get positions
                    r_i = coord[i]
                    r_j = coord[j]
                    r_k = coord[k]
                    
                    # Calculate interatomic distances
                    r_ij = np.linalg.norm(r_i - r_j)
                    r_jk = np.linalg.norm(r_j - r_k)
                    r_ki = np.linalg.norm(r_k - r_i)
                    
                    # Get coordination-number scaled C6 coefficients
                    c6_ij, _, r0_ij = self.get_d4_parameters(
                        element_list[i], element_list[j], 
                        q1=charges[i], q2=charges[j], 
                        cn1=cn[i], cn2=cn[j]
                    )
                    
                    c6_jk, _, r0_jk = self.get_d4_parameters(
                        element_list[j], element_list[k], 
                        q1=charges[j], q2=charges[k], 
                        cn1=cn[j], cn2=cn[k]
                    )
                    
                    c6_ki, _, r0_ki = self.get_d4_parameters(
                        element_list[k], element_list[i], 
                        q1=charges[k], q2=charges[i], 
                        cn1=cn[k], cn2=cn[i]
                    )
                    
                    # Calculate geometric mean of C6 coefficients
                    c9 = (c6_ij * c6_jk * c6_ki) ** (1.0/3.0)
                    
                    # Calculate three-body damping
                    damp_ij = self.d4_damping_function(r_ij, r0_ij)
                    damp_jk = self.d4_damping_function(r_jk, r0_jk)
                    damp_ki = self.d4_damping_function(r_ki, r0_ki)
                    damp = damp_ij * damp_jk * damp_ki
                    
                    # Calculate angle factor
                    cos_ijk = np.dot(r_i - r_j, r_k - r_j) / (r_ij * r_jk)
                    cos_jki = np.dot(r_j - r_k, r_i - r_k) / (r_jk * r_ki)
                    cos_kij = np.dot(r_k - r_i, r_j - r_i) / (r_ki * r_ij)
                    angle_factor = 1.0 + 3.0 * cos_ijk * cos_jki * cos_kij
                    
                    # Calculate three-body energy term
                    e_3 = s9 * angle_factor * c9 * damp / (r_ij * r_jk * r_ki) ** 3
                    
                    # Calculate force constants (second derivatives)
                    # This is a simplified approximation - full D4 three-body Hessian is complex
                    fc_scale = 0.01 * s9 * angle_factor * c9 * damp
                    
                    # Add approximate three-body contributions to Hessian
                    for n in range(3):
                        for m in range(3):
                            # Diagonal blocks (diagonal atoms)
                            if n == m:
                                self.cart_hess[i * 3 + n][i * 3 + m] += fc_scale / r_ij**6 + fc_scale / r_ki**6
                                self.cart_hess[j * 3 + n][j * 3 + m] += fc_scale / r_ij**6 + fc_scale / r_jk**6
                                self.cart_hess[k * 3 + n][k * 3 + m] += fc_scale / r_jk**6 + fc_scale / r_ki**6
                            
                            # Off-diagonal blocks (between atoms)
                            self.cart_hess[i * 3 + n][j * 3 + m] -= fc_scale / r_ij**6 
                            self.cart_hess[j * 3 + n][i * 3 + m] -= fc_scale / r_ij**6
                            
                            self.cart_hess[j * 3 + n][k * 3 + m] -= fc_scale / r_jk**6
                            self.cart_hess[k * 3 + n][j * 3 + m] -= fc_scale / r_jk**6
                            
                            self.cart_hess[k * 3 + n][i * 3 + m] -= fc_scale / r_ki**6
                            self.cart_hess[i * 3 + n][k * 3 + m] -= fc_scale / r_ki**6
        
        return
    
    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian using Swart's model with D4 dispersion"""
        print("Generating Swart's approximate hessian with D4 dispersion correction...")
        self.cart_hess = np.zeros((len(coord) * 3, len(coord) * 3), dtype="float64")
        
        # Calculate coordination numbers and atomic charges for D4
        cn = self.calculate_coordination_numbers(coord, element_list)
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate all contributions to the Hessian
        self.swart_bond(coord, element_list, charges, cn)
        self.swart_angle(coord, element_list, charges, cn)
        self.swart_dihedral_angle(coord, element_list, charges, cn)
        self.swart_out_of_plane(coord, element_list, charges, cn)
        
        # Add D4-specific three-body term
        self.calculate_three_body_term(coord, element_list, charges, cn)
        
        # Ensure symmetry of the Hessian matrix
        n_basis = len(coord) * 3
        for i in range(n_basis):
            for j in range(i):
                if abs(self.cart_hess[i][j] - self.cart_hess[j][i]) > 1.0e-10:
                    avg = (self.cart_hess[i][j] + self.cart_hess[j][i]) / 2.0
                    self.cart_hess[i][j] = avg
                    self.cart_hess[j][i] = avg
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj


class SwartD3ApproxHessian:
    def __init__(self):
        #Swart's Model Hessian augmented with D3 dispersion
        #ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536–2544.
        #ref.: S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys., 2010, 132, 154104
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        
        self.kr = 0.35
        self.kf = 0.15
        self.kt = 0.005
        
        self.cutoff = 70.0
        self.eps = 1.0e-12
        
        # D3 parameters
        self.d3_params = D3Parameters()
        
        # Cyano group parameters
        self.cn_kr = 0.70  # Enhanced force constant for C≡N triple bond
        self.cn_kf = 0.20  # Enhanced force constant for angles involving C≡N
        self.cn_kt = 0.002 # Reduced force constant for torsions involving C≡N
        return
    
    def detect_cyano_groups(self, coord, element_list):
        """Detect C≡N triple bonds in the structure"""
        cyano_atoms = []  # List of (C_idx, N_idx) tuples
        
        for i in range(len(coord)):
            if element_list[i] != 'C':
                continue
                
            for j in range(len(coord)):
                if i == j or element_list[j] != 'N':
                    continue
                    
                # Calculate distance between C and N
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij = np.sqrt(x_ij**2 + y_ij**2 + z_ij**2)
                
                # Check if distance is close to a triple bond length
                cn_triple_bond = triple_covalent_radii_lib('C') + triple_covalent_radii_lib('N')
                
                if abs(r_ij - cn_triple_bond) < 0.3:  # Within 0.3 bohr of ideal length
                    # Check if C is connected to only one other atom (besides N)
                    connections_to_c = 0
                    for k in range(len(coord)):
                        if k == i or k == j:
                            continue
                            
                        x_ik = coord[i][0] - coord[k][0]
                        y_ik = coord[i][1] - coord[k][1]
                        z_ik = coord[i][2] - coord[k][2]
                        r_ik = np.sqrt(x_ik**2 + y_ik**2 + z_ik**2)
                        
                        cov_dist = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        if r_ik < 1.3 * cov_dist:  # Using 1.3 as a factor to account for bond length variations
                            connections_to_c += 1
                    
                    # If C has only one other connection, it's likely a terminal cyano group
                    if connections_to_c <= 1:
                        cyano_atoms.append((i, j))
        
        return cyano_atoms
    
    def calc_force_const(self, alpha, covalent_length, distance):
        force_const = np.exp(-1 * alpha * (distance / covalent_length - 1.0))
        return force_const
        
    def d3_damping_function(self, r_ij, r0, order=6):
        """D3 rational damping function"""
        a1 = self.d3_params.a1
        a2 = self.d3_params.a2
        
        if order == 6:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        elif order == 8:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a2 * r0)) ** a1)
        return 0.0
        
    def get_d3_parameters(self, elem1, elem2):
        """Get D3 parameters for a pair of elements"""
        # Get R4/R2 values
        r4r2_1 = self.d3_params.get_r4r2(elem1)
        r4r2_2 = self.d3_params.get_r4r2(elem2)
        
        # C6 coefficients based on averaging rules
        c6_1 = self.d3_params.get_r4r2(elem1) ** 2
        c6_2 = self.d3_params.get_r4r2(elem2) ** 2
        c6_param = np.sqrt(c6_1 * c6_2)
        
        # C8 coefficients using r^4/r^2 ratio
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (combined vdW radii)
        r0_param = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6_param, c8_param, r0_param
    
    def calc_d3_force_const(self, r_ij, c6_param, c8_param, r0_param):
        """Calculate D3 dispersion force constant"""
        s6 = self.d3_params.s6
        s8 = self.d3_params.s8
        
        # Apply damping functions
        damp6 = self.d3_damping_function(r_ij, r0_param, order=6)
        damp8 = self.d3_damping_function(r_ij, r0_param, order=8)
        
        # Energy terms (negative because dispersion is attractive)
        e6 = -s6 * c6_param / r_ij ** 6 * damp6
        e8 = -s8 * c8_param / r_ij ** 8 * damp8
        
        # Force constant is the second derivative of energy
        # Simplified approximation of second derivatives
        fc6 = s6 * c6_param * (42.0 / r_ij ** 8) * damp6
        fc8 = s8 * c8_param * (72.0 / r_ij ** 10) * damp8
        
        return fc6 + fc8
    
    def swart_bond(self, coord, element_list):
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for i in range(len(coord)):
            for j in range(i):
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get D3 parameters
                c6_param, c8_param, r0_param = self.get_d3_parameters(element_list[i], element_list[j])
                
                # Calculate D3 dispersion contribution
                d3_force_const = self.calc_d3_force_const(r_ij, c6_param, c8_param, r0_param)
                
                # Check if this is a cyano bond
                is_cyano_bond = False
                for c_idx, n_idx in cyano_atoms:
                    if (i == c_idx and j == n_idx) or (i == n_idx and j == c_idx):
                        is_cyano_bond = True
                        break
                
                # Apply appropriate force constant
                if is_cyano_bond:
                    # Special force constant for C≡N triple bond
                    g_mm = self.cn_kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * d3_force_const
                else:
                    # Regular Swart force constant with D3 dispersion
                    g_mm = self.kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * d3_force_const
                
                # Calculate Hessian components
                hess_xx = g_mm * x_ij ** 2 / r_ij_2
                hess_xy = g_mm * x_ij * y_ij / r_ij_2
                hess_xz = g_mm * x_ij * z_ij / r_ij_2
                hess_yy = g_mm * y_ij ** 2 / r_ij_2
                hess_yz = g_mm * y_ij * z_ij / r_ij_2
                hess_zz = g_mm * z_ij ** 2 / r_ij_2
                
                # Fill the Hessian matrix
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3] -= hess_xz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
                
                self.cart_hess[j * 3][i * 3] -= hess_xx
                self.cart_hess[j * 3][i * 3 + 1] -= hess_xy
                self.cart_hess[j * 3][i * 3 + 2] -= hess_xz
                self.cart_hess[j * 3 + 1][i * 3] -= hess_xy
                self.cart_hess[j * 3 + 1][i * 3 + 1] -= hess_yy
                self.cart_hess[j * 3 + 1][i * 3 + 2] -= hess_yz
                self.cart_hess[j * 3 + 2][i * 3] -= hess_xz
                self.cart_hess[j * 3 + 2][i * 3 + 1] -= hess_yz
                self.cart_hess[j * 3 + 2][i * 3 + 2] -= hess_zz
        
        return
    
    def swart_angle(self, coord, element_list):
        """Calculate angle bending contributions to the Hessian with D3 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for i in range(len(coord)):
            for j in range(len(coord)):
                if i == j:
                    continue
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get D3 parameters for i-j pair
                c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                
                for k in range(j):
                    if i == k:
                        continue
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    # Check for linear arrangement (cos_theta ~ 1.0)
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    # Get D3 parameters for i-k pair
                    c6_ik, c8_ik, r0_ik = self.get_d3_parameters(element_list[i], element_list[k])
                    
                    # Calculate D3 dispersion contributions
                    d3_ij = self.calc_d3_force_const(r_ij, c6_ij, c8_ij, r0_ij)
                    d3_ik = self.calc_d3_force_const(r_ik, c6_ik, c8_ik, r0_ik)
                    
                    # Calculate bond force constants with D3 dispersion
                    g_ij = self.calc_force_const(1.0, covalent_length_ij, r_ij) + 0.5 * self.kd * d3_ij
                    g_ik = self.calc_force_const(1.0, covalent_length_ik, r_ik) + 0.5 * self.kd * d3_ik
                    
                    # Check if angle involves cyano group
                    is_cyano_angle = (i in cyano_set or j in cyano_set or k in cyano_set)
                    
                    # Apply appropriate force constant
                    if is_cyano_angle:
                        # Special force constant for angles involving cyano groups
                        g_jk = self.cn_kf * g_ij * g_ik
                    else:
                        # Regular Swart force constant
                        g_jk = self.kf * g_ij * g_ik
                    
                    # Calculate cross product for sin(theta)
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:
                        cos_theta = (r_ij_2 + r_ik_2 - r_jk_2) / (2.0 * r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                       
                        if sin_theta > self.eps: # non-linear
                            # Calculate derivatives for non-linear case
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            # Update Hessian for non-linear case
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                    
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            # Update diagonal blocks
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            # Special handling for linear arrangements
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            # Iterate over two perpendicular directions
                            for ii in range(2):
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                # Update Hessian for linear case
                                for l in range(3):
                                    for m in range(3):
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                # Update diagonal blocks for linear case
                                for l in range(3):
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass  # Skip if any distance is too small
        
        # Make the Hessian symmetric for angle terms
        n_basis = len(coord) * 3
        for i in range(n_basis):
            for j in range(i):
                if abs(self.cart_hess[i][j] - self.cart_hess[j][i]) > 1.0e-10:
                    avg = (self.cart_hess[i][j] + self.cart_hess[j][i]) / 2.0
                    self.cart_hess[i][j] = avg
                    self.cart_hess[j][i] = avg
                
        return

    def swart_dihedral_angle(self, coord, element_list):
        """Calculate dihedral angle contributions to the Hessian with D3 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        r_jk = coord[j] - coord[k]
                        r_kl = coord[k] - coord[l]
                        
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        # Calculate vector magnitudes
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        # Skip if angle is too shallow (less than 35 degrees)
                        a35 = (35.0/180)* np.pi
                        cosfi_max = np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        
                        # Get D3 parameters for each atom pair
                        c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                        c6_jk, c8_jk, r0_jk = self.get_d3_parameters(element_list[j], element_list[k])
                        c6_kl, c8_kl, r0_kl = self.get_d3_parameters(element_list[k], element_list[l])
                        
                        # Calculate D3 dispersion contributions
                        d3_ij = self.calc_d3_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        d3_jk = self.calc_d3_force_const(norm_r_jk, c6_jk, c8_jk, r0_jk)
                        d3_kl = self.calc_d3_force_const(norm_r_kl, c6_kl, c8_kl, r0_kl)
                        
                        # Calculate bond force constants with D3 dispersion
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * d3_ij
                        g_jk = self.calc_force_const(1.0, covalent_length_jk, norm_r_jk) + 0.5 * self.kd * d3_jk
                        g_kl = self.calc_force_const(1.0, covalent_length_kl, norm_r_kl) + 0.5 * self.kd * d3_kl
                        
                        # Check if torsion involves cyano group
                        is_cyano_torsion = False
                        if i in cyano_set or j in cyano_set or k in cyano_set or l in cyano_set:
                            is_cyano_torsion = True
                        
                        # Adjust force constant for cyano groups - they have flatter torsional potentials
                        if is_cyano_torsion:
                            t_ij = self.cn_kt * g_ij * g_jk * g_kl
                        else:
                            t_ij = self.kt * g_ij * g_jk * g_kl
                        
                        # Calculate torsion angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        # Update Hessian with torsional contributions
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] += t_ij * s_k[n] * s_l[m]
                            
                        # Update diagonal blocks
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] += t_ij * s_l[n] * s_l[m]
                       
        return
      
    def swart_out_of_plane(self, coord, element_list):
        """Calculate out-of-plane bending contributions to the Hessian"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        # Skip if atoms are nearly collinear
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        # Get D3 parameters for each atom pair
                        c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                        c6_ik, c8_ik, r0_ik = self.get_d3_parameters(element_list[i], element_list[k])
                        c6_il, c8_il, r0_il = self.get_d3_parameters(element_list[i], element_list[l])
                        
                        # Calculate D3 dispersion contributions
                        d3_ij = self.calc_d3_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        d3_ik = self.calc_d3_force_const(norm_r_ik, c6_ik, c8_ik, r0_ik)
                        d3_il = self.calc_d3_force_const(norm_r_il, c6_il, c8_il, r0_il)
                        
                        # Calculate bond force constants with D3 dispersion
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * d3_ij
                        g_ik = self.calc_force_const(1.0, covalent_length_ik, norm_r_ik) + 0.5 * self.kd * d3_ik
                        g_il = self.calc_force_const(1.0, covalent_length_il, norm_r_il) + 0.5 * self.kd * d3_il
                        
                        # Check if any atom is part of a cyano group
                        is_cyano_involved = (i in cyano_set or j in cyano_set or 
                                            k in cyano_set or l in cyano_set)
                        
                        # Adjust force constant if cyano group is involved
                        if is_cyano_involved:
                            t_ij = 0.5 * self.kt * g_ij * g_ik * g_il  # Reduce force constant for cyano
                        else:
                            t_ij = self.kt * g_ij * g_ik * g_il
                        
                        # Calculate out-of-plane angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update Hessian with out-of-plane contributions
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                        # Update diagonal blocks
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                        
        return

    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian using Swart's model with D3 dispersion"""
        print("Generating Swart's approximate hessian with D3 dispersion correction...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        
        self.swart_bond(coord, element_list)
        self.swart_angle(coord, element_list)
        self.swart_dihedral_angle(coord, element_list)
        self.swart_out_of_plane(coord, element_list)
        
        # Ensure symmetry of the Hessian matrix
        for i in range(len(coord)*3):
            for j in range(i):
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
                elif abs(self.cart_hess[j][i]) < 1.0e-10:
                    self.cart_hess[j][i] = self.cart_hess[i][j]
                else:
                    # Average if both elements are non-zero but different
                    avg = (self.cart_hess[i][j] + self.cart_hess[j][i]) / 2.0
                    self.cart_hess[i][j] = avg
                    self.cart_hess[j][i] = avg
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj
    
class SwartD2ApproxHessian:
    def __init__(self):
        #Swart's Model Hessian augmented with D2
        #ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536–2544.
        #ref.: https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        
        self.kr = 0.35
        self.kf = 0.15
        self.kt = 0.005
        
        self.cutoff = 70.0
        self.eps = 1.0e-12
        
        #self.s6 = 20.0
        return
    
    def calc_force_const(self, alpha, covalent_length, distance):
        force_const = np.exp(-1 * alpha * (distance / covalent_length - 1.0))
        
        return force_const
        
    def calc_vdw_force_const(self, alpha, vdw_length, distance):
        vdw_force_const = np.exp(-1 * alpha * (vdw_length - distance) ** 2)
        return vdw_force_const
    
    def swart_bond(self, coord, element_list):
        for i in range(len(coord)):
            for j in range(i):
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                vdw_length = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                C6_param_i = D2_C6_coeff_lib(element_list[i])
                C6_param_j = D2_C6_coeff_lib(element_list[j])
                C6_param_ij = np.sqrt(C6_param_i * C6_param_j)
                C6_VDW_ij = D2_VDW_radii_lib(element_list[i]) + D2_VDW_radii_lib(element_list[j])
                VDW_xx = calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xy = calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xz = calc_vdw_anisotropic(x_ij, z_ij, y_ij,C6_param_ij, C6_VDW_ij)
                VDW_yy = calc_vdw_isotopic(y_ij, x_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_yz = calc_vdw_anisotropic(y_ij, z_ij, x_ij, C6_param_ij, C6_VDW_ij)
                VDW_zz = calc_vdw_isotopic(z_ij, x_ij, y_ij, C6_param_ij, C6_VDW_ij)
        
                g_mm = self.kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * self.calc_vdw_force_const(5.0, vdw_length, r_ij)
                
                hess_xx = g_mm * x_ij ** 2 / r_ij_2 - VDW_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - VDW_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - VDW_xz
                hess_yy = g_mm * y_ij ** 2 / r_ij_2 - VDW_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - VDW_yz
                hess_zz = g_mm * z_ij ** 2 / r_ij_2 - VDW_zz
                
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
       
        return 
    
    def swart_angle(self, coord, element_list):
        for i in range(len(coord)):
            for j in range(len(coord)):
                if i == j:
                    continue
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                
                for k in range(j):
                    if i == k:
                        continue
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                    
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    g_ij = self.calc_force_const(1.0, covalent_length_ij, r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, r_ij)
                    g_ik = self.calc_force_const(1.0, covalent_length_ik, r_ik) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ik, r_ik)
                    
                    g_jk = self.kf * g_ij * g_ik
                    
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:
                        cos_theta = (r_ij_2 + r_ik_2 - r_jk_2) / (2.0 * r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                       
                        if sin_theta > self.eps: # non-linear
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        # m = i, i = j, j = k
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                       
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        
                                     
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val

                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            for ii in range(2):
                                # m = i, i = j, j = k
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                for l in range(3):
                                    for m in range(3):#Under construction
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                for l in range(3):#Under construction
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass
                    
                
        return

    def swart_dihedral_angle(self, coord, element_list):
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
    
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_kl = coord[k] - coord[l]
                        vdw_length_kl = UFF_VDW_distance_lib(element_list[k]) + UFF_VDW_distance_lib(element_list[l])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        a35 = (35.0/180)* np.pi
                        cosfi_max=np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, norm_r_ij)
                        g_jk = self.calc_force_const(1.0, covalent_length_jk, norm_r_jk) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_jk, norm_r_jk)
                        g_kl = self.calc_force_const(1.0, covalent_length_kl, norm_r_kl) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_kl, norm_r_kl) 
                       
                        t_ij = self.kt * g_ij * g_jk * g_kl
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] = self.cart_hess[3 * i + n][3 * j + m] + t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] = self.cart_hess[3 * i + n][3 * k + m] + t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] = self.cart_hess[3 * i + n][3 * l + m] + t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] = self.cart_hess[3 * j + n][3 * k + m] + t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] = self.cart_hess[3 * j + n][3 * l + m] + t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] = self.cart_hess[3 * k + n][3 * l + m] + t_ij * s_k[n] * s_l[m]
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] = self.cart_hess[3 * i + n][3 * i + m] + t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] = self.cart_hess[3 * j + n][3 * j + m] + t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] = self.cart_hess[3 * k + n][3 * k + m] + t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] = self.cart_hess[3 * l + n][3 * l + m] + t_ij * s_l[n] * s_l[m]
                       
        return
       
    def swart_out_of_plane(self, coord, element_list):
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        vdw_length_il = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[l])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, norm_r_ij)
                        
                        g_ik = self.calc_force_const(1.0, covalent_length_ik, norm_r_ik) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ik, norm_r_ik)
                        
                        g_il = self.calc_force_const(1.0, covalent_length_il, norm_r_il) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_il, norm_r_il)
                        
                        
                        t_ij = self.kt * g_ij * g_ik * g_il #self.ko * g_ij * g_ik * g_il
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                          
        return



    def main(self, coord, element_list, cart_gradient):
        #coord: Bohr
        print("generating Swart's approximate hessian...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        
        self.swart_bond(coord, element_list)
        self.swart_angle(coord, element_list)
        self.swart_dihedral_angle(coord, element_list)
        self.swart_out_of_plane(coord, element_list)
        
        for i in range(len(coord)*3):
            for j in range(len(coord)*3):
                
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj#cart_hess

class Lindh2007D2ApproxHessian:
    def __init__(self):
        #Lindh's Model Hessian (2007) augmented with D2
        #ref.: https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        self.bond_threshold_scale = 1.0
        self.kr = 0.45
        self.kf = 0.10
        self.kt = 0.0025
        self.ko = 0.16
        self.kd = 0.05
        self.cutoff = 50.0
        self.eps = 1.0e-12
        
        self.rAv = np.array([[1.3500,   2.1000,   2.5300],
                    [2.1000,   2.8700,   3.8000],
                    [2.5300,   3.8000,   4.5000]]) 
        
        self.aAv = np.array([[1.0000,   0.3949,   0.3949],
                    [0.3949,   0.2800,   0.1200],
                    [0.3949,   0.1200,   0.0600]]) 
        
        self.dAv = np.array([[0.0000,   3.6000,   3.6000],
                    [3.6000,   5.3000,   5.3000],
                    [3.6000,   5.3000,   5.3000]])
        
        #self.s6 = 20.0
        return
    
    def select_idx(self, elem_num):
        if type(elem_num) == str:
            elem_num = element_number(elem_num)

        if (elem_num > 0 and elem_num < 2):
            idx = 0
        elif (elem_num >= 2 and elem_num < 10):
            idx = 1
        elif (elem_num >= 10 and elem_num < 18):
            idx = 2
        elif (elem_num >= 18 and elem_num < 36):
            idx = 2
        elif (elem_num >= 36 and elem_num < 54):
            idx = 2
        elif (elem_num >= 54 and elem_num < 86):
            idx = 2
        elif (elem_num >= 86):
            idx = 2
        else:
            idx = 2

        return idx
    
    
    def calc_force_const(self, alpha, r_0, distance_2):
        force_const = np.exp(alpha * (r_0 ** 2 -1.0 * distance_2))
        
        return force_const
        
    def calc_vdw_force_const(self, alpha, vdw_length, distance):
        vdw_force_const = np.exp(-4 * alpha * (vdw_length - distance) ** 2)
        return vdw_force_const
    
    def lindh2007_bond(self, coord, element_list):
        for i in range(len(coord)):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(i):
                j_idx = self.select_idx(element_list[j])
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                r_0 = self.rAv[i_idx][j_idx]
                d_0 = self.dAv[i_idx][j_idx]
                alpha = self.aAv[i_idx][j_idx]
                
                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                
                #if self.bond_threshold_scale * triple_bond > r_ij:
                #    covalent_length = triple_bond
                #elif self.bond_threshold_scale * double_bond > r_ij:

                #    covalent_length = double_bond
                #else:
                covalent_length = single_bond
               
                vdw_length = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                C6_param_i = D2_C6_coeff_lib(element_list[i])
                C6_param_j = D2_C6_coeff_lib(element_list[j])
                C6_param_ij = np.sqrt(C6_param_i * C6_param_j)
                C6_VDW_ij = D2_VDW_radii_lib(element_list[i]) + D2_VDW_radii_lib(element_list[j])
                VDW_xx = calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xy = calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xz = calc_vdw_anisotropic(x_ij, z_ij, y_ij,C6_param_ij, C6_VDW_ij)
                VDW_yy = calc_vdw_isotopic(y_ij, x_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_yz = calc_vdw_anisotropic(y_ij, z_ij, x_ij, C6_param_ij, C6_VDW_ij)
                VDW_zz = calc_vdw_isotopic(z_ij, x_ij, y_ij, C6_param_ij, C6_VDW_ij)
        
                g_mm = self.kr * self.calc_force_const(alpha, covalent_length, r_ij_2) + self.kd * self.calc_vdw_force_const(4.0, vdw_length, r_ij)
                
                hess_xx = g_mm * x_ij ** 2 / r_ij_2 - VDW_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - VDW_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - VDW_xz
                hess_yy = g_mm * y_ij ** 2 / r_ij_2 - VDW_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - VDW_yz
                hess_zz = g_mm * z_ij ** 2 / r_ij_2 - VDW_zz
                
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
       
        return 
    
    def lindh2007_angle(self, coord, element_list):
        for i in range(len(coord)):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(len(coord)):
                if i == j:
                    continue
                j_idx = self.select_idx(element_list[j])
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)

                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                
                #if self.bond_threshold_scale * triple_bond > r_ij:
                #    covalent_length_ij = triple_bond
                #elif self.bond_threshold_scale * double_bond > r_ij:
                #    covalent_length_ij = double_bond
                #else:
                covalent_length_ij = single_bond

                vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                    
                r_ij_0 = self.rAv[i_idx][j_idx]
                d_ij_0 = self.dAv[i_idx][j_idx]
                alpha_ij = self.aAv[i_idx][j_idx]
                
                for k in range(j):
                    if i == k:
                        continue
                    k_idx = self.select_idx(element_list[k])
                    
                    r_ik_0 = self.rAv[i_idx][k_idx]
                    d_ik_0 = self.dAv[i_idx][k_idx]
                    alpha_ik = self.aAv[i_idx][k_idx]
                    
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)


                    single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[k])
                    triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[k])
                    
                    #if self.bond_threshold_scale * triple_bond > r_ik:
                    #    covalent_length_ik = triple_bond
                    #elif self.bond_threshold_scale * double_bond > r_ik:
                    #    covalent_length_ik = double_bond
                    #else:
                    covalent_length_ik = single_bond

                    #covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                    
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ij, r_ij)
                    g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ik, r_ik)
                    
                    g_jk = self.kf * (g_ij + 0.5 * self.kd / self.kr * d_ij_0) * (g_ik * 0.5 * self.kd / self.kr * d_ik_0)
                    
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:

                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                        cos_theta = dot_product_r_ij_r_ik / (r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        if sin_theta > self.eps: # non-linear
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        # m = i, i = j, j = k
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                       
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        
                                     
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val

                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            for ii in range(2):
                                # m = i, i = j, j = k
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                for l in range(3):
                                    for m in range(3):#Under construction
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                for l in range(3):#Under construction
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass
                    
                
        return

    def lindh2007_dihedral_angle(self, coord, element_list):
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
                        i_idx = self.select_idx(element_list[i])
                        j_idx = self.select_idx(element_list[j])
                        k_idx = self.select_idx(element_list[k])
                        l_idx = self.select_idx(element_list[l])
                        
                        r_ij_0 = self.rAv[i_idx][j_idx]
                        d_ij_0 = self.dAv[i_idx][j_idx]
                        alpha_ij = self.aAv[i_idx][j_idx]
                        
                        
                        r_jk_0 = self.rAv[j_idx][k_idx]
                        d_jk_0 = self.dAv[j_idx][k_idx]
                        alpha_jk = self.aAv[j_idx][k_idx]
                        
                        r_kl_0 = self.rAv[k_idx][l_idx]
                        d_kl_0 = self.dAv[k_idx][l_idx]
                        alpha_kl = self.aAv[k_idx][l_idx]
                        
                        
                        
                        
                        
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = double_bond
                        #else:
                        covalent_length_ij = single_bond

                        #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])

                        #single_bond = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[j])
                        #double_bond = double_covalent_radii_lib(element_list[k]) + double_covalent_radii_lib(element_list[j])
                        #triple_bond = triple_covalent_radii_lib(element_list[k]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_jk):
                        #    covalent_length_jk = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_jk):
                        #    covalent_length_jk = double_bond
                        #else:
                        covalent_length_jk = single_bond

                        #covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        #r_jk = coord[j] - coord[k]
                        #vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        #covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_kl = coord[k] - coord[l]

                        #single_bond = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        #double_bond = double_covalent_radii_lib(element_list[k]) + double_covalent_radii_lib(element_list[l])
                        #triple_bond = triple_covalent_radii_lib(element_list[k]) + triple_covalent_radii_lib(element_list[l])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_kl):
                        #    covalent_length_kl = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_kl):
                        #    covalent_length_kl = double_bond
                        #else:
                        covalent_length_kl = single_bond

                        vdw_length_kl = UFF_VDW_distance_lib(element_list[k]) + UFF_VDW_distance_lib(element_list[l])
                        #covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        a35 = (35.0/180)* np.pi
                        cosfi_max=np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ij, norm_r_ij)
                        g_jk = self.calc_force_const(alpha_jk, covalent_length_jk, r_jk_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_jk, norm_r_jk)
                        g_kl = self.calc_force_const(alpha_kl, covalent_length_kl, r_kl_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_kl, norm_r_kl) 
                       
                        t_ij = self.kt * (g_ij * 0.5 * self.kd / self.kr * d_ij_0) * (g_jk * 0.5 * self.kd / self.kr * d_jk_0) * (g_kl * 0.5 * self.kd / self.kr * d_kl_0)
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] = self.cart_hess[3 * i + n][3 * j + m] + t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] = self.cart_hess[3 * i + n][3 * k + m] + t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] = self.cart_hess[3 * i + n][3 * l + m] + t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] = self.cart_hess[3 * j + n][3 * k + m] + t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] = self.cart_hess[3 * j + n][3 * l + m] + t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] = self.cart_hess[3 * k + n][3 * l + m] + t_ij * s_k[n] * s_l[m]
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] = self.cart_hess[3 * i + n][3 * i + m] + t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] = self.cart_hess[3 * j + n][3 * j + m] + t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] = self.cart_hess[3 * k + n][3 * k + m] + t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] = self.cart_hess[3 * l + n][3 * l + m] + t_ij * s_l[n] * s_l[m]
                       
        return
       
    def lindh2007_out_of_plane(self, coord, element_list):
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = double_bond
                        #else:
                        covalent_length_ij = single_bond

                        #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[k])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[k])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ik):
                        #    covalent_length_ik = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ik):
                        #    covalent_length_ik = double_bond
                        #else:
                        covalent_length_ik = single_bond

                        #covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        vdw_length_il = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[l])

                        #single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        #double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[l])
                        #triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[l])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_il):
                        #    covalent_length_il = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_il):
                        #    covalent_length_il = double_bond
                        #else:
                        covalent_length_il = single_bond

                        #covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        idx_i = self.select_idx(element_list[i])
                        idx_j = self.select_idx(element_list[j])
                        idx_k = self.select_idx(element_list[k])
                        idx_l = self.select_idx(element_list[l])
                        
                        d_ij_0 = self.dAv[idx_i][idx_j]
                        r_ij_0 = self.rAv[idx_i][idx_j]
                        alpha_ij = self.aAv[idx_i][idx_j]
                        
                        d_ik_0 = self.dAv[idx_i][idx_k]
                        r_ik_0 = self.rAv[idx_i][idx_k]
                        alpha_ik = self.aAv[idx_i][idx_k]
                        
                        d_il_0 = self.dAv[idx_i][idx_l]
                        r_il_0 = self.rAv[idx_i][idx_l]
                        alpha_il = self.aAv[idx_i][idx_l]
                        
                        
                        
                        
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        kd = 0.0
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_ij, norm_r_ij)
                        
                        g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_ik, norm_r_ik)
                        
                        g_il = self.calc_force_const(alpha_il, covalent_length_il, r_il_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_il, norm_r_il)
                        
                        
                        t_ij = self.ko * g_ij * g_ik * g_il #self.ko * g_ij * g_ik * g_il
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                          
        return

    def main(self, coord, element_list, cart_gradient):
        norm_grad = np.linalg.norm(cart_gradient)
        scale = 0.1
        eigval_scale = scale * np.exp(-1 * norm_grad ** 2.0)
        #coord: Bohr
        print("generating Lindh's (2007) approximate hessian...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        self.lindh2007_bond(coord, element_list)
        self.lindh2007_angle(coord, element_list)
        self.lindh2007_dihedral_angle(coord, element_list)
        self.lindh2007_out_of_plane(coord, element_list)
        
        for i in range(len(coord)*3):
            for j in range(len(coord)*3):
                
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        hess_proj = np.dot(np.dot(eigenvectors, np.diag(np.abs(eigenvalues) * eigval_scale)), np.linalg.inv(eigenvectors))
        
        return hess_proj#cart_hess


class Lindh2007D3ApproxHessian:
    """
    Lindh's Model Hessian (2007) augmented with D3 dispersion correction.
    
    This class implements Lindh's 2007 approximate Hessian model with D3 dispersion
    corrections for improved accuracy in describing non-covalent interactions.
    
    References:
        - Lindh et al., Chem. Phys. Lett. 2007, 241, 423.
        - Grimme et al., J. Chem. Phys. 2010, 132, 154104 (DFT-D3).
        - https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
    """
    def __init__(self):
        # Unit conversion constants
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # Force constant parameters
        self.bond_threshold_scale = 1.0
        self.kr = 0.45  # Bond stretching force constant
        self.kf = 0.10  # Angle bend force constant
        self.kt = 0.0025  # Torsion force constant
        self.ko = 0.16  # Out-of-plane force constant
        self.kd = 0.05  # Dispersion force constant
        
        # Numerical parameters
        self.cutoff = 50.0  # Cutoff for long-range interactions (Bohr)
        self.eps = 1.0e-12  # Numerical threshold for avoiding division by zero
        
        # Reference parameters (element type matrices)
        self.rAv = np.array([
            [1.3500, 2.1000, 2.5300],
            [2.1000, 2.8700, 3.8000],
            [2.5300, 3.8000, 4.5000]
        ])
        
        self.aAv = np.array([
            [1.0000, 0.3949, 0.3949],
            [0.3949, 0.2800, 0.1200],
            [0.3949, 0.1200, 0.0600]
        ])
        
        self.dAv = np.array([
            [0.0000, 3.6000, 3.6000],
            [3.6000, 5.3000, 5.3000],
            [3.6000, 5.3000, 5.3000]
        ])
        
        # D3 dispersion parameters
        self.d3params = D3Parameters()
        
    def select_idx(self, elem_num):
        """
        Determine element group index for parameter selection.
        
        Args:
            elem_num (str or int): Element symbol or atomic number
            
        Returns:
            int: Group index (0-2) for parameter lookup
        """
        if isinstance(elem_num, str):
            elem_num = element_number(elem_num)

        # Group 1: H
        if elem_num > 0 and elem_num < 2:
            return 0
        # Group 2: First row elements (Li-Ne)
        elif elem_num >= 2 and elem_num < 10:
            return 1
        # Group 3: All others
        else:
            return 2
    
    def calc_force_const(self, alpha, r_0, distance_2):
        """
        Calculate bond stretching force constant based on Lindh's model.
        
        Args:
            alpha: Exponential parameter
            r_0: Reference bond length
            distance_2: Squared distance between atoms
            
        Returns:
            float: Force constant
        """
        return np.exp(alpha * (r_0**2 - distance_2))
    
    def get_c6_coefficient(self, element):
        """
        Get C6 dispersion coefficient for an element.
        
        Args:
            element: Element symbol
            
        Returns:
            float: C6 coefficient in atomic units
        """
        return D2_C6_coeff_lib(element)
    
    def calc_d3_force_const(self, r_ij, c6_param, c8_param, r0_param):
        """
        Calculate D3 dispersion force constant with Becke-Johnson damping.
        
        Args:
            r_ij: Distance between atoms
            c6_param: C6 dispersion coefficient
            c8_param: C8 dispersion coefficient
            r0_param: van der Waals radius sum parameter
            
        Returns:
            float: D3 dispersion force constant
        """
        # Becke-Johnson damping function for C6 term
        r0_plus_a1 = r0_param + self.d3params.a1
        f_damp_6 = r_ij**6 / (r_ij**6 + (r0_plus_a1 * self.d3params.a2)**6)
        
        # Becke-Johnson damping function for C8 term
        f_damp_8 = r_ij**8 / (r_ij**8 + (r0_plus_a1 * self.d3params.a2)**8)
        
        # D3 dispersion energy contributions
        e6 = -self.d3params.s6 * c6_param * f_damp_6 / r_ij**6
        e8 = -self.d3params.s8 * c8_param * f_damp_8 / r_ij**8
        
        # Combined force constant (negative of energy for attractive contribution)
        return -(e6 + e8)
    
    def get_d3_parameters(self, elem1, elem2):
        """
        Get D3 parameters for a pair of elements.
        
        Args:
            elem1: First element symbol
            elem2: Second element symbol
            
        Returns:
            tuple: (c6_param, c8_param, r0_param) for the element pair
        """
        # Get C6 coefficients
        c6_1 = self.get_c6_coefficient(elem1)
        c6_2 = self.get_c6_coefficient(elem2)
        
        # Combine C6 coefficients
        c6_param = np.sqrt(c6_1 * c6_2)
        
        # Get r4r2 values for C8 coefficient calculation
        r4r2_1 = self.d3params.get_r4r2(elem1)
        r4r2_2 = self.d3params.get_r4r2(elem2)
        
        # Calculate C8 coefficient (3.0 is the conversion factor in Grimme's D3 formulation)
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # Calculate R0 parameter (vdW radii sum)
        r0_1 = UFF_VDW_distance_lib(elem1) / self.bohr2angstroms
        r0_2 = UFF_VDW_distance_lib(elem2) / self.bohr2angstroms
        r0_param = r0_1 + r0_2
        
        return c6_param, c8_param, r0_param
    
    def calc_d3_gradient_components(self, x_ij, y_ij, z_ij, c6_param, c8_param, r0_param):
        """
        Calculate D3 dispersion gradient components.
        
        Args:
            x_ij, y_ij, z_ij: Distance components
            c6_param: C6 dispersion coefficient
            c8_param: C8 dispersion coefficient
            r0_param: van der Waals radius sum parameter
            
        Returns:
            tuple: (xx, xy, xz, yy, yz, zz) gradient components
        """
        r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
        r_ij = np.sqrt(r_ij_2)
        
        if r_ij < 0.1:  # Avoid numerical issues
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # BJ damping parameters
        r0_plus_a1 = r0_param + self.d3params.a1
        a2_term = self.d3params.a2
        bj_term_6 = (r0_plus_a1 * a2_term)**6
        bj_term_8 = (r0_plus_a1 * a2_term)**8
        
        # Calculate damping functions and their derivatives
        r_ij_6 = r_ij**6
        r_ij_8 = r_ij**8
        
        # C6 term: damping and derivatives
        damp_6 = r_ij_6 / (r_ij_6 + bj_term_6)
        d_damp_6_dr = 6.0 * r_ij_6 * bj_term_6 / ((r_ij_6 + bj_term_6)**2 * r_ij)
        
        # C8 term: damping and derivatives
        damp_8 = r_ij_8 / (r_ij_8 + bj_term_8)
        d_damp_8_dr = 8.0 * r_ij_8 * bj_term_8 / ((r_ij_8 + bj_term_8)**2 * r_ij)
        
        # Force (negative derivative of energy)
        f6 = self.d3params.s6 * c6_param * (6.0 * damp_6 / r_ij**7 + d_damp_6_dr / r_ij**6)
        f8 = self.d3params.s8 * c8_param * (8.0 * damp_8 / r_ij**9 + d_damp_8_dr / r_ij**8)
        
        # Total force
        force = f6 + f8
        
        # Calculate derivative components
        deriv_scale = force / r_ij
        
        # Calculate gradient components
        xx = deriv_scale * x_ij**2 / r_ij_2
        xy = deriv_scale * x_ij * y_ij / r_ij_2
        xz = deriv_scale * x_ij * z_ij / r_ij_2
        yy = deriv_scale * y_ij**2 / r_ij_2
        yz = deriv_scale * y_ij * z_ij / r_ij_2
        zz = deriv_scale * z_ij**2 / r_ij_2
        
        return xx, xy, xz, yy, yz, zz
    
    def lindh2007_bond(self, coord, element_list):
        """
        Calculate bond stretching contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(i):
                j_idx = self.select_idx(element_list[j])
                
                # Calculate distance components and magnitude
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                # Get Lindh parameters
                r_0 = self.rAv[i_idx][j_idx]
                d_0 = self.dAv[i_idx][j_idx]
                alpha = self.aAv[i_idx][j_idx]
                
                # Determine appropriate bond length based on bond type
                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                covalent_length = single_bond  # Default to single bond
                
                # Get D3 parameters and calculate dispersion contribution
                c6_param, c8_param, r0_param = self.get_d3_parameters(element_list[i], element_list[j])
                
                # Calculate force constants
                lindh_force = self.kr * self.calc_force_const(alpha, covalent_length, r_ij_2)
                
                # Add D3 dispersion if atoms are far apart
                d3_factor = 0.0
                if r_ij > 2.0 * covalent_length:
                    d3_factor = self.kd * self.calc_d3_force_const(r_ij, c6_param, c8_param, r0_param)
                
                # Combined force constant
                g_mm = lindh_force + d3_factor
                
                # Calculate D3 gradient components
                d3_xx, d3_xy, d3_xz, d3_yy, d3_yz, d3_zz = self.calc_d3_gradient_components(
                    x_ij, y_ij, z_ij, c6_param, c8_param, r0_param)
                
                # Calculate Hessian elements
                hess_xx = g_mm * x_ij**2 / r_ij_2 - d3_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - d3_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - d3_xz
                hess_yy = g_mm * y_ij**2 / r_ij_2 - d3_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - d3_yz
                hess_zz = g_mm * z_ij**2 / r_ij_2 - d3_zz
                
                # Update diagonal blocks
                i_offset = i * 3
                j_offset = j * 3
                
                # i-i block
                self.cart_hess[i_offset, i_offset] += hess_xx
                self.cart_hess[i_offset + 1, i_offset] += hess_xy
                self.cart_hess[i_offset + 1, i_offset + 1] += hess_yy
                self.cart_hess[i_offset + 2, i_offset] += hess_xz
                self.cart_hess[i_offset + 2, i_offset + 1] += hess_yz
                self.cart_hess[i_offset + 2, i_offset + 2] += hess_zz
                
                # j-j block
                self.cart_hess[j_offset, j_offset] += hess_xx
                self.cart_hess[j_offset + 1, j_offset] += hess_xy
                self.cart_hess[j_offset + 1, j_offset + 1] += hess_yy
                self.cart_hess[j_offset + 2, j_offset] += hess_xz
                self.cart_hess[j_offset + 2, j_offset + 1] += hess_yz
                self.cart_hess[j_offset + 2, j_offset + 2] += hess_zz
                
                # i-j block
                self.cart_hess[i_offset, j_offset] -= hess_xx
                self.cart_hess[i_offset, j_offset + 1] -= hess_xy
                self.cart_hess[i_offset, j_offset + 2] -= hess_xz
                self.cart_hess[i_offset + 1, j_offset] -= hess_xy
                self.cart_hess[i_offset + 1, j_offset + 1] -= hess_yy
                self.cart_hess[i_offset + 1, j_offset + 2] -= hess_yz
                self.cart_hess[i_offset + 2, j_offset] -= hess_xz
                self.cart_hess[i_offset + 2, j_offset + 1] -= hess_yz
                self.cart_hess[i_offset + 2, j_offset + 2] -= hess_zz
    
    def lindh2007_angle(self, coord, element_list):
        """
        Calculate angle bending contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                j_idx = self.select_idx(element_list[j])
                
                # Vector from j to i
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                # Get bond parameters
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get Lindh parameters
                r_ij_0 = self.rAv[i_idx][j_idx]
                d_ij_0 = self.dAv[i_idx][j_idx]
                alpha_ij = self.aAv[i_idx][j_idx]
                
                # Loop through potential third atoms to form an angle
                for k in range(j):
                    if i == k:
                        continue
                    
                    k_idx = self.select_idx(element_list[k])
                    
                    # Get parameters for i-k interaction
                    r_ik_0 = self.rAv[i_idx][k_idx]
                    d_ik_0 = self.dAv[i_idx][k_idx]
                    alpha_ik = self.aAv[i_idx][k_idx]
                    
                    # Vector from k to i
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    
                    # Get bond parameters
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    # Check if angle is well-defined (not linear)
                    cos_angle = (x_ij * x_ik + y_ij * y_ik + z_ij * z_ik) / (r_ij * r_ik)
                    
                    if abs(cos_angle - 1.0) < self.eps:
                        continue  # Skip near-linear angles
                    
                    # Vector from k to j
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    # Calculate force constants with D3 contributions
                    c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                    c6_ik, c8_ik, r0_ik = self.get_d3_parameters(element_list[i], element_list[k])
                    
                    g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                    if r_ij > 2.0 * covalent_length_ij:
                        g_ij += 0.5 * self.kd * self.calc_d3_force_const(r_ij, c6_ij, c8_ij, r0_ij)
                    
                    g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2)
                    if r_ik > 2.0 * covalent_length_ik:
                        g_ik += 0.5 * self.kd * self.calc_d3_force_const(r_ik, c6_ik, c8_ik, r0_ik)
                    
                    # Angular force constant
                    g_jk = self.kf * (g_ij + 0.5 * self.kd / self.kr * d_ij_0) * (g_ik + 0.5 * self.kd / self.kr * d_ik_0)
                    
                    # Cross product magnitude for sin(theta)
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik)**2 + (z_ij * x_ik - x_ij * z_ik)**2 + (x_ij * y_ik - y_ij * x_ik)**2
                    r_cross = np.sqrt(r_cross_2) if r_cross_2 > 1.0e-12 else 0.0
                    
                    # Skip if distances are too small
                    if r_ik <= self.eps or r_ij <= self.eps or r_jk <= self.eps:
                        continue
                    
                    # Calculate angle and its derivatives
                    dot_product = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    cos_theta = dot_product / (r_ij * r_ik)
                    sin_theta = r_cross / (r_ij * r_ik)
                    
                    if sin_theta > self.eps:  # Non-linear case
                        # Calculate derivatives
                        s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                        s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                        s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                        
                        s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                        s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                        s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                        
                        s_xi = -s_xj - s_xk
                        s_yi = -s_yj - s_yk
                        s_zi = -s_zj - s_zk
                        
                        s_i = [s_xi, s_yi, s_zi]
                        s_j = [s_xj, s_yj, s_zj]
                        s_k = [s_xk, s_yk, s_zk]
                        
                        # Update Hessian elements
                        for l in range(3):
                            for m in range(3):
                                # i-j block
                                if i > j:
                                    self.cart_hess[i*3+l, j*3+m] += g_jk * s_i[l] * s_j[m]
                                else:
                                    self.cart_hess[j*3+l, i*3+m] += g_jk * s_j[l] * s_i[m]
                                
                                # i-k block
                                if i > k:
                                    self.cart_hess[i*3+l, k*3+m] += g_jk * s_i[l] * s_k[m]
                                else:
                                    self.cart_hess[k*3+l, i*3+m] += g_jk * s_k[l] * s_i[m]
                                
                                # j-k block
                                if j > k:
                                    self.cart_hess[j*3+l, k*3+m] += g_jk * s_j[l] * s_k[m]
                                else:
                                    self.cart_hess[k*3+l, j*3+m] += g_jk * s_k[l] * s_j[m]
                        
                        # Diagonal blocks
                        for l in range(3):
                            for m in range(l):
                                self.cart_hess[j*3+l, j*3+m] += g_jk * s_j[l] * s_j[m]
                                self.cart_hess[i*3+l, i*3+m] += g_jk * s_i[l] * s_i[m]
                                self.cart_hess[k*3+l, k*3+m] += g_jk * s_k[l] * s_k[m]
                    
                    else:  # Linear case
                        # Handle linear angles using arbitrary perpendicular vectors
                        if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                            x_1, y_1, z_1 = -y_ij, x_ij, 0.0
                            x_2, y_2, z_2 = -x_ij * z_ij, -y_ij * z_ij, x_ij**2 + y_ij**2
                        else:
                            x_1, y_1, z_1 = 1.0, 0.0, 0.0
                            x_2, y_2, z_2 = 0.0, 1.0, 0.0
                        
                        x = [x_1, x_2]
                        y = [y_1, y_2]
                        z = [z_1, z_2]
                        
                        # Calculate derivatives for two perpendicular directions
                        for ii in range(2):
                            r_1 = np.sqrt(x[ii]**2 + y[ii]**2 + z[ii]**2)
                            cos_theta_x = x[ii] / r_1
                            cos_theta_y = y[ii] / r_1
                            cos_theta_z = z[ii] / r_1
                            
                            # Derivatives
                            s_xj = -cos_theta_x / r_ij
                            s_yj = -cos_theta_y / r_ij
                            s_zj = -cos_theta_z / r_ij
                            s_xk = -cos_theta_x / r_ik
                            s_yk = -cos_theta_y / r_ik
                            s_zk = -cos_theta_z / r_ik
                            
                            s_xi = -s_xj - s_xk
                            s_yi = -s_yj - s_yk
                            s_zi = -s_zj - s_zk
                            
                            s_i = [s_xi, s_yi, s_zi]
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            
                            # Update Hessian elements
                            for l in range(3):
                                for m in range(3):
                                    # i-j block
                                    if i > j:
                                        self.cart_hess[i*3+l, j*3+m] += g_jk * s_i[l] * s_j[m]
                                    else:
                                        self.cart_hess[j*3+l, i*3+m] += g_jk * s_j[l] * s_i[m]
                                    
                                    # i-k block
                                    if i > k:
                                        self.cart_hess[i*3+l, k*3+m] += g_jk * s_i[l] * s_k[m]
                                    else:
                                        self.cart_hess[k*3+l, i*3+m] += g_jk * s_k[l] * s_i[m]
                                    
                                    # j-k block
                                    if j > k:
                                        self.cart_hess[j*3+l, k*3+m] += g_jk * s_j[l] * s_k[m]
                                    else:
                                        self.cart_hess[k*3+l, j*3+m] += g_jk * s_k[l] * s_j[m]
                            
                            # Diagonal blocks
                            for l in range(3):
                                for m in range(l):
                                    self.cart_hess[j*3+l, j*3+m] += g_jk * s_j[l] * s_j[m]
                                    self.cart_hess[i*3+l, i*3+m] += g_jk * s_i[l] * s_i[m]
                                    self.cart_hess[k*3+l, k*3+m] += g_jk * s_k[l] * s_k[m]
    
    def lindh2007_dihedral_angle(self, coord, element_list):
        """
        Calculate dihedral angle (torsion) contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for j in range(n_atoms):
            t_xyz_2 = coord[j]
            
            for k in range(j+1, n_atoms):
                t_xyz_3 = coord[k]
                
                for i in range(j):
                    if i == k:
                        continue
                        
                    t_xyz_1 = coord[i]
                    
                    for l in range(k+1, n_atoms):
                        if l == i or l == j:
                            continue
                        
                        t_xyz_4 = coord[l]
                        
                        # Get element indices for parameter lookup
                        i_idx = self.select_idx(element_list[i])
                        j_idx = self.select_idx(element_list[j])
                        k_idx = self.select_idx(element_list[k])
                        l_idx = self.select_idx(element_list[l])
                        
                        # Get Lindh parameters
                        r_ij_0 = self.rAv[i_idx][j_idx]
                        d_ij_0 = self.dAv[i_idx][j_idx]
                        alpha_ij = self.aAv[i_idx][j_idx]
                        
                        r_jk_0 = self.rAv[j_idx][k_idx]
                        d_jk_0 = self.dAv[j_idx][k_idx]
                        alpha_jk = self.aAv[j_idx][k_idx]
                        
                        r_kl_0 = self.rAv[k_idx][l_idx]
                        d_kl_0 = self.dAv[k_idx][l_idx]
                        alpha_kl = self.aAv[k_idx][l_idx]
                        
                        # Calculate bond vectors and lengths
                        r_ij = coord[i] - coord[j]
                        r_jk = coord[j] - coord[k]
                        r_kl = coord[k] - coord[l]
                        
                        # Get bond parameters
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        # Calculate squared distances
                        r_ij_2 = np.sum(r_ij**2)
                        r_jk_2 = np.sum(r_jk**2)
                        r_kl_2 = np.sum(r_kl**2)
                        
                        # Calculate norms
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                        
                        # Check if near-linear angles would cause numerical issues
                        a35 = (35.0/180) * np.pi
                        cosfi_max = np.cos(a35)
                        
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                            
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        
                        # Get D3 parameters for bond pairs
                        c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                        c6_jk, c8_jk, r0_jk = self.get_d3_parameters(element_list[j], element_list[k])
                        c6_kl, c8_kl, r0_kl = self.get_d3_parameters(element_list[k], element_list[l])
                        
                        # Calculate force constants with D3 contributions
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                        if norm_r_ij > 2.0 * covalent_length_ij:
                            g_ij += 0.5 * self.kd * self.calc_d3_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        
                        g_jk = self.calc_force_const(alpha_jk, covalent_length_jk, r_jk_2)
                        if norm_r_jk > 2.0 * covalent_length_jk:
                            g_jk += 0.5 * self.kd * self.calc_d3_force_const(norm_r_jk, c6_jk, c8_jk, r0_jk)
                        
                        g_kl = self.calc_force_const(alpha_kl, covalent_length_kl, r_kl_2)
                        if norm_r_kl > 2.0 * covalent_length_kl:
                            g_kl += 0.5 * self.kd * self.calc_d3_force_const(norm_r_kl, c6_kl, c8_kl, r0_kl)
                        
                        # Calculate torsion force constant
                        t_ij = self.kt * (g_ij * 0.5 * self.kd / self.kr * d_ij_0) * \
                              (g_jk * 0.5 * self.kd / self.kr * d_jk_0) * \
                              (g_kl * 0.5 * self.kd / self.kr * d_kl_0)
                        
                        # Calculate torsion angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        
                        # Extract derivatives
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update off-diagonal blocks
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3*i+n, 3*j+m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3*i+n, 3*k+m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3*i+n, 3*l+m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3*j+n, 3*k+m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3*j+n, 3*l+m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3*k+n, 3*l+m] += t_ij * s_k[n] * s_l[m]
                        
                        # Update diagonal blocks (lower triangle)
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3*i+n, 3*i+m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3*j+n, 3*j+m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3*k+n, 3*k+m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3*l+n, 3*l+m] += t_ij * s_l[n] * s_l[m]
    
    def lindh2007_out_of_plane(self, coord, element_list):
        """
        Calculate out-of-plane bending contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            t_xyz_4 = coord[i]
            
            for j in range(i+1, n_atoms):
                t_xyz_1 = coord[j]
                
                for k in range(j+1, n_atoms):
                    t_xyz_2 = coord[k]
                    
                    for l in range(k+1, n_atoms):
                        t_xyz_3 = coord[l]
                        
                        # Calculate bond vectors
                        r_ij = coord[i] - coord[j]
                        r_ik = coord[i] - coord[k]
                        r_il = coord[i] - coord[l]
                        
                        # Get bond parameters
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        # Get element indices for parameter lookup
                        idx_i = self.select_idx(element_list[i])
                        idx_j = self.select_idx(element_list[j])
                        idx_k = self.select_idx(element_list[k])
                        idx_l = self.select_idx(element_list[l])
                        
                        # Get Lindh parameters
                        d_ij_0 = self.dAv[idx_i][idx_j]
                        r_ij_0 = self.rAv[idx_i][idx_j]
                        alpha_ij = self.aAv[idx_i][idx_j]
                        
                        d_ik_0 = self.dAv[idx_i][idx_k]
                        r_ik_0 = self.rAv[idx_i][idx_k]
                        alpha_ik = self.aAv[idx_i][idx_k]
                        
                        d_il_0 = self.dAv[idx_i][idx_l]
                        r_il_0 = self.rAv[idx_i][idx_l]
                        alpha_il = self.aAv[idx_i][idx_l]
                        
                        # Calculate squared distances
                        r_ij_2 = np.sum(r_ij**2)
                        r_ik_2 = np.sum(r_ik**2)
                        r_il_2 = np.sum(r_il**2)
                        
                        # Calculate norms
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        # Check for near-linear angles that would cause numerical issues
                        cosfi2 = np.dot(r_ij, r_ik) / (norm_r_ij * norm_r_ik)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                            
                        cosfi3 = np.dot(r_ij, r_il) / (norm_r_ij * norm_r_il)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                            
                        cosfi4 = np.dot(r_ik, r_il) / (norm_r_ik * norm_r_il)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        # Get D3 parameters for each pair
                        c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                        c6_ik, c8_ik, r0_ik = self.get_d3_parameters(element_list[i], element_list[k])
                        c6_il, c8_il, r0_il = self.get_d3_parameters(element_list[i], element_list[l])
                        
                        # Disable direct D3 contributions to out-of-plane terms
                        kd = 0.0
                        
                        # Calculate force constants for each bond
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                        if norm_r_ij > 2.0 * covalent_length_ij:
                            g_ij += 0.5 * kd * self.calc_d3_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        
                        g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2)
                        if norm_r_ik > 2.0 * covalent_length_ik:
                            g_ik += 0.5 * kd * self.calc_d3_force_const(norm_r_ik, c6_ik, c8_ik, r0_ik)
                        
                        g_il = self.calc_force_const(alpha_il, covalent_length_il, r_il_2)
                        if norm_r_il > 2.0 * covalent_length_il:
                            g_il += 0.5 * kd * self.calc_d3_force_const(norm_r_il, c6_il, c8_il, r0_il)
                        
                        # Combined force constant for out-of-plane motion
                        t_ij = self.ko * g_ij * g_ik * g_il
                        
                        # Calculate out-of-plane angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        # Extract derivatives
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update off-diagonal blocks
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i*3+n, j*3+m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i*3+n, k*3+m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i*3+n, l*3+m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j*3+n, k*3+m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j*3+n, l*3+m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k*3+n, l*3+m] += t_ij * s_k[n] * s_l[m]
                        
                        # Update diagonal blocks (lower triangle)
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i*3+n, i*3+m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j*3+n, j*3+m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k*3+n, k*3+m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l*3+n, l*3+m] += t_ij * s_l[n] * s_l[m]
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate approximate Hessian using Lindh's 2007 model with D3 dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            cart_gradient: Cartesian gradient vector
            
        Returns:
            hess_proj: Projected approximate Hessian matrix
        """
        print("Generating Lindh's (2007) approximate Hessian with D3 dispersion...")
        
        # Scale eigenvalues based on gradient norm (smaller scale for larger gradients)
        norm_grad = np.linalg.norm(cart_gradient)
        scale = 0.1
        eigval_scale = scale * np.exp(-1 * norm_grad**2.0)
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate individual contributions
        self.lindh2007_bond(coord, element_list)
        self.lindh2007_angle(coord, element_list)
        self.lindh2007_dihedral_angle(coord, element_list)
        self.lindh2007_out_of_plane(coord, element_list)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                if abs(self.cart_hess[i, j]) < 1.0e-10:
                    self.cart_hess[i, j] = self.cart_hess[j, i]
                else:
                    self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out translational and rotational degrees of freedom
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        # Adjust eigenvalues for stability based on gradient magnitude
        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        hess_proj = np.dot(np.dot(eigenvectors, np.diag(np.abs(eigenvalues) * eigval_scale)), 
                          np.transpose(eigenvectors))
        
        return hess_proj

class Lindh2007D4ApproxHessian:
    """
    Lindh's Model Hessian (2007) augmented with D4 dispersion correction.
    
    This class implements Lindh's 2007 approximate Hessian model with D4 dispersion
    corrections for improved accuracy in describing non-covalent interactions.
    
    References:
        - Lindh et al., Chem. Phys. Lett. 2007, 241, 423.
        - Caldeweyher et al., J. Chem. Phys. 2019, 150, 154122 (DFT-D4).
        - https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
    """

    def __init__(self):
        # Unit conversion constants
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # Force constant parameters
        self.bond_threshold_scale = 1.0
        self.kr = 0.45  # Bond stretching force constant
        self.kf = 0.10  # Angle bend force constant
        self.kt = 0.0025  # Torsion force constant
        self.ko = 0.16  # Out-of-plane force constant
        self.kd = 0.05  # Dispersion force constant
        
        # Numerical parameters
        self.cutoff = 50.0  # Cutoff for long-range interactions (Bohr)
        self.eps = 1.0e-12  # Numerical threshold for avoiding division by zero
        
        # Reference parameters (element type matrices)
        self.rAv = np.array([
            [1.3500, 2.1000, 2.5300],
            [2.1000, 2.8700, 3.8000],
            [2.5300, 3.8000, 4.5000]
        ])
        
        self.aAv = np.array([
            [1.0000, 0.3949, 0.3949],
            [0.3949, 0.2800, 0.1200],
            [0.3949, 0.1200, 0.0600]
        ])
        
        self.dAv = np.array([
            [0.0000, 3.6000, 3.6000],
            [3.6000, 5.3000, 5.3000],
            [3.6000, 5.3000, 5.3000]
        ])
        
        # D4 dispersion parameters
        self.d4params = D4Parameters()
        
    def select_idx(self, elem_num):
        """
        Determine element group index for parameter selection.
        
        Args:
            elem_num (str or int): Element symbol or atomic number
            
        Returns:
            int: Group index (0-2) for parameter lookup
        """
        if isinstance(elem_num, str):
            elem_num = element_number(elem_num)

        # Group 1: H
        if elem_num > 0 and elem_num < 2:
            return 0
        # Group 2: First row elements (Li-Ne)
        elif elem_num >= 2 and elem_num < 10:
            return 1
        # Group 3: All others
        else:
            return 2
    
    def calc_force_const(self, alpha, r_0, distance_2):
        """
        Calculate bond stretching force constant based on Lindh's model.
        
        Args:
            alpha: Exponential parameter
            r_0: Reference bond length
            distance_2: Squared distance between atoms
            
        Returns:
            float: Force constant
        """
        return np.exp(alpha * (r_0**2 - distance_2))
    
    def get_c6_coefficient(self, element):
        """
        Get C6 dispersion coefficient for an element.
        
        Args:
            element: Element symbol
            
        Returns:
            float: C6 coefficient in atomic units
        """
        return D2_C6_coeff_lib(element)
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic partial charges using electronegativity equilibration.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            
        Returns:
            charges: Array of partial charges
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Detect bonds based on distance
        bonds = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coord[i] - coord[j])
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                r_sum = (r_cov_i + r_cov_j) * self.bond_threshold_scale
                
                if r_ij < r_sum * 1.3:  # 1.3 is a common threshold for bond detection
                    bonds.append((i, j))
        
        # Estimate charges based on electronegativity differences
        for i, j in bonds:
            en_i = self.d4params.get_electronegativity(element_list[i])
            en_j = self.d4params.get_electronegativity(element_list[j])
            
            # Simple electronegativity-based charge transfer
            en_diff = en_j - en_i
            charge_transfer = 0.1 * np.tanh(0.2 * en_diff)  # Sigmoidal scaling for stability
            
            charges[i] += charge_transfer
            charges[j] -= charge_transfer
        
        # Normalize to ensure total charge is zero
        charges -= np.mean(charges)
        
        return charges
    
    def calculate_coordination_numbers(self, coord, element_list):
        """
        Calculate atomic coordination numbers for scaling dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            
        Returns:
            cn: Array of coordination numbers
        """
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        # D4 uses a counting function based on interatomic distances
        for i in range(n_atoms):
            r_cov_i = covalent_radii_lib(element_list[i])
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                r_cov_j = covalent_radii_lib(element_list[j])
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Coordination number contribution with Gaussian-like counting function
                r_cov = r_cov_i + r_cov_j
                k1 = 16.0  # Steepness parameter
                cn_contrib = 1.0 / (1.0 + np.exp(-k1 * (r_cov/r_ij - 1.0)))
                cn[i] += cn_contrib
        
        return cn
    
    def calc_d4_force_const(self, r_ij, c6_param, c8_param, r0_param, q_scaling=1.0):
        """
        Calculate D4 dispersion force constant with Becke-Johnson damping and charge scaling.
        
        Args:
            r_ij: Distance between atoms
            c6_param: C6 dispersion coefficient
            c8_param: C8 dispersion coefficient
            r0_param: van der Waals radius sum parameter
            q_scaling: Charge-dependent scaling factor
            
        Returns:
            float: D4 dispersion force constant
        """
        # Becke-Johnson damping function for C6 term
        r0_plus_a1 = r0_param + self.d4params.a1
        f_damp_6 = r_ij**6 / (r_ij**6 + (r0_plus_a1 * self.d4params.a2)**6)
        
        # Becke-Johnson damping function for C8 term
        f_damp_8 = r_ij**8 / (r_ij**8 + (r0_plus_a1 * self.d4params.a2)**8)
        
        # Apply charge scaling to C6 and C8 coefficients
        c6_scaled = c6_param * q_scaling
        c8_scaled = c8_param * q_scaling
        
        # D4 dispersion energy contributions
        e6 = -self.d4params.s6 * c6_scaled * f_damp_6 / r_ij**6
        e8 = -self.d4params.s8 * c8_scaled * f_damp_8 / r_ij**8
        
        # Combined force constant (negative of energy for attractive contribution)
        return -(e6 + e8)
    
    def get_d4_parameters(self, elem1, elem2, q1=0.0, q2=0.0):
        """
        Get D4 parameters for a pair of elements with charge scaling.
        
        Args:
            elem1: First element symbol
            elem2: Second element symbol
            q1: Partial charge on first atom
            q2: Partial charge on second atom
            
        Returns:
            tuple: (c6_param, c8_param, r0_param, q_scaling) for the element pair
        """
        # Get reference polarizabilities
        alpha_1 = self.d4params.get_polarizability(elem1)
        alpha_2 = self.d4params.get_polarizability(elem2)
        
        # Get base C6 coefficients
        c6_1 = self.get_c6_coefficient(elem1)
        c6_2 = self.get_c6_coefficient(elem2)
        
        # Combine C6 coefficients with Casimir-Polder formula
        c6_param = 2.0 * c6_1 * c6_2 / (c6_1 + c6_2)
        
        # Get r4r2 values for C8 coefficient calculation
        r4r2_1 = self.d4params.get_r4r2(elem1)
        r4r2_2 = self.d4params.get_r4r2(elem2)
        
        # Calculate C8 coefficient using D4 formula
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # Calculate R0 parameter (vdW radii sum)
        r0_1 = UFF_VDW_distance_lib(elem1) / self.bohr2angstroms
        r0_2 = UFF_VDW_distance_lib(elem2) / self.bohr2angstroms
        r0_param = r0_1 + r0_2
        
        # Apply charge scaling (D4-specific feature)
        # Gaussian charge scaling function
        q_scaling = np.exp(-self.d4params.ga * (q1**2 + q2**2))
        
        return c6_param, c8_param, r0_param, q_scaling
    
    def calc_d4_gradient_components(self, x_ij, y_ij, z_ij, c6_param, c8_param, r0_param, q_scaling=1.0):
        """
        Calculate D4 dispersion gradient components.
        
        Args:
            x_ij, y_ij, z_ij: Distance components
            c6_param: C6 dispersion coefficient
            c8_param: C8 dispersion coefficient
            r0_param: van der Waals radius sum parameter
            q_scaling: Charge-dependent scaling factor
            
        Returns:
            tuple: (xx, xy, xz, yy, yz, zz) gradient components
        """
        r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
        r_ij = np.sqrt(r_ij_2)
        
        if r_ij < 0.1:  # Avoid numerical issues
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # BJ damping parameters
        r0_plus_a1 = r0_param + self.d4params.a1
        a2_term = self.d4params.a2
        bj_term_6 = (r0_plus_a1 * a2_term)**6
        bj_term_8 = (r0_plus_a1 * a2_term)**8
        
        # Calculate damping functions and their derivatives
        r_ij_6 = r_ij**6
        r_ij_8 = r_ij**8
        
        # C6 term: damping and derivatives
        damp_6 = r_ij_6 / (r_ij_6 + bj_term_6)
        d_damp_6_dr = 6.0 * r_ij_6 * bj_term_6 / ((r_ij_6 + bj_term_6)**2 * r_ij)
        
        # C8 term: damping and derivatives
        damp_8 = r_ij_8 / (r_ij_8 + bj_term_8)
        d_damp_8_dr = 8.0 * r_ij_8 * bj_term_8 / ((r_ij_8 + bj_term_8)**2 * r_ij)
        
        # Apply charge scaling
        c6_scaled = c6_param * q_scaling
        c8_scaled = c8_param * q_scaling
        
        # Force (negative derivative of energy)
        f6 = self.d4params.s6 * c6_scaled * (6.0 * damp_6 / r_ij**7 + d_damp_6_dr / r_ij**6)
        f8 = self.d4params.s8 * c8_scaled * (8.0 * damp_8 / r_ij**9 + d_damp_8_dr / r_ij**8)
        
        # Total force
        force = f6 + f8
        
        # Calculate derivative components
        deriv_scale = force / r_ij
        
        # Calculate gradient components
        xx = deriv_scale * x_ij**2 / r_ij_2
        xy = deriv_scale * x_ij * y_ij / r_ij_2
        xz = deriv_scale * x_ij * z_ij / r_ij_2
        yy = deriv_scale * y_ij**2 / r_ij_2
        yz = deriv_scale * y_ij * z_ij / r_ij_2
        zz = deriv_scale * z_ij**2 / r_ij_2
        
        return xx, xy, xz, yy, yz, zz
    
    def lindh2007_bond(self, coord, element_list, charges, cn):
        """
        Calculate bond stretching contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            charges: Atomic partial charges
            cn: Coordination numbers
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(i):
                j_idx = self.select_idx(element_list[j])
                
                # Calculate distance components and magnitude
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                # Get Lindh parameters
                r_0 = self.rAv[i_idx][j_idx]
                d_0 = self.dAv[i_idx][j_idx]
                alpha = self.aAv[i_idx][j_idx]
                
                # Determine appropriate bond length based on bond type
                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                covalent_length = single_bond  # Default to single bond
                
                # Get D4 parameters with charge scaling
                c6_param, c8_param, r0_param, q_scaling = self.get_d4_parameters(
                    element_list[i], element_list[j], charges[i], charges[j])
                
                # Calculate force constants
                lindh_force = self.kr * self.calc_force_const(alpha, covalent_length, r_ij_2)
                
                # Add D4 dispersion if atoms are far apart
                d4_factor = 0.0
                if r_ij > 2.0 * covalent_length:
                    d4_factor = self.kd * self.calc_d4_force_const(r_ij, c6_param, c8_param, r0_param, q_scaling)
                
                # Combined force constant
                g_mm = lindh_force + d4_factor
                
                # Calculate D4 gradient components
                d4_xx, d4_xy, d4_xz, d4_yy, d4_yz, d4_zz = self.calc_d4_gradient_components(
                    x_ij, y_ij, z_ij, c6_param, c8_param, r0_param, q_scaling)
                
                # Calculate Hessian elements
                hess_xx = g_mm * x_ij**2 / r_ij_2 - d4_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - d4_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - d4_xz
                hess_yy = g_mm * y_ij**2 / r_ij_2 - d4_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - d4_yz
                hess_zz = g_mm * z_ij**2 / r_ij_2 - d4_zz
                
                # Update diagonal blocks
                i_offset = i * 3
                j_offset = j * 3
                
                # i-i block
                self.cart_hess[i_offset, i_offset] += hess_xx
                self.cart_hess[i_offset + 1, i_offset] += hess_xy
                self.cart_hess[i_offset + 1, i_offset + 1] += hess_yy
                self.cart_hess[i_offset + 2, i_offset] += hess_xz
                self.cart_hess[i_offset + 2, i_offset + 1] += hess_yz
                self.cart_hess[i_offset + 2, i_offset + 2] += hess_zz
                
                # j-j block
                self.cart_hess[j_offset, j_offset] += hess_xx
                self.cart_hess[j_offset + 1, j_offset] += hess_xy
                self.cart_hess[j_offset + 1, j_offset + 1] += hess_yy
                self.cart_hess[j_offset + 2, j_offset] += hess_xz
                self.cart_hess[j_offset + 2, j_offset + 1] += hess_yz
                self.cart_hess[j_offset + 2, j_offset + 2] += hess_zz
         
                # i-j block
                self.cart_hess[i_offset, j_offset] -= hess_xx
                self.cart_hess[i_offset, j_offset + 1] -= hess_xy
                self.cart_hess[i_offset, j_offset + 2] -= hess_xz
                self.cart_hess[i_offset + 1, j_offset] -= hess_xy
                self.cart_hess[i_offset + 1, j_offset + 1] -= hess_yy
                self.cart_hess[i_offset + 1, j_offset + 2] -= hess_yz
                self.cart_hess[i_offset + 2, j_offset] -= hess_xz
                self.cart_hess[i_offset + 2, j_offset + 1] -= hess_yz
                self.cart_hess[i_offset + 2, j_offset + 2] -= hess_zz
    
    def lindh2007_angle(self, coord, element_list, charges, cn):
        """
        Calculate angle bending contributions to the Hessian with D4 dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            charges: Atomic partial charges
            cn: Coordination numbers
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                j_idx = self.select_idx(element_list[j])
                
                # Vector from j to i
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                # Get bond parameters
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get Lindh parameters
                r_ij_0 = self.rAv[i_idx][j_idx]
                d_ij_0 = self.dAv[i_idx][j_idx]
                alpha_ij = self.aAv[i_idx][j_idx]
                
                # Loop through potential third atoms to form an angle
                for k in range(j):
                    if i == k:
                        continue
                    
                    k_idx = self.select_idx(element_list[k])
                    
                    # Get parameters for i-k interaction
                    r_ik_0 = self.rAv[i_idx][k_idx]
                    d_ik_0 = self.dAv[i_idx][k_idx]
                    alpha_ik = self.aAv[i_idx][k_idx]
                    
                    # Vector from k to i
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    
                    # Get bond parameters
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    # Check if angle is well-defined (not linear)
                    cos_angle = (x_ij * x_ik + y_ij * y_ik + z_ij * z_ik) / (r_ij * r_ik)
                    
                    if abs(cos_angle - 1.0) < self.eps:
                        continue  # Skip near-linear angles
                    
                    # Vector from k to j
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    # Calculate force constants with D4 contributions
                    c6_ij, c8_ij, r0_ij, q_ij = self.get_d4_parameters(
                        element_list[i], element_list[j], charges[i], charges[j])
                    c6_ik, c8_ik, r0_ik, q_ik = self.get_d4_parameters(
                        element_list[i], element_list[k], charges[i], charges[k])
                    
                    g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                    if r_ij > 2.0 * covalent_length_ij:
                        g_ij += 0.5 * self.kd * self.calc_d4_force_const(r_ij, c6_ij, c8_ij, r0_ij, q_ij)
                    
                    g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2)
                    if r_ik > 2.0 * covalent_length_ik:
                        g_ik += 0.5 * self.kd * self.calc_d4_force_const(r_ik, c6_ik, c8_ik, r0_ik, q_ik)
                    
                    # Angular force constant
                    g_jk = self.kf * (g_ij + 0.5 * self.kd / self.kr * d_ij_0) * (g_ik + 0.5 * self.kd / self.kr * d_ik_0)
                    
                    # Cross product magnitude for sin(theta)
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik)**2 + (z_ij * x_ik - x_ij * z_ik)**2 + (x_ij * y_ik - y_ij * x_ik)**2
                    r_cross = np.sqrt(r_cross_2) if r_cross_2 > 1.0e-12 else 0.0
                    
                    # Skip if distances are too small
                    if r_ik <= self.eps or r_ij <= self.eps or r_jk <= self.eps:
                        continue
                    
                    # Calculate angle and its derivatives
                    dot_product = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    cos_theta = dot_product / (r_ij * r_ik)
                    sin_theta = r_cross / (r_ij * r_ik)
                    
                    if sin_theta > self.eps:  # Non-linear case
                        # Calculate derivatives
                        s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                        s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                        s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                        
                        s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                        s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                        s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                        
                        s_xi = -s_xj - s_xk
                        s_yi = -s_yj - s_yk
                        s_zi = -s_zj - s_zk
                        
                        s_i = [s_xi, s_yi, s_zi]
                        s_j = [s_xj, s_yj, s_zj]
                        s_k = [s_xk, s_yk, s_zk]
                        
                        # Update Hessian elements
                        for l in range(3):
                            for m in range(3):
                                # i-j block
                                if i > j:
                                    self.cart_hess[i*3+l, j*3+m] += g_jk * s_i[l] * s_j[m]
                                else:
                                    self.cart_hess[j*3+l, i*3+m] += g_jk * s_j[l] * s_i[m]
                                
                                # i-k block
                                if i > k:
                                    self.cart_hess[i*3+l, k*3+m] += g_jk * s_i[l] * s_k[m]
                                else:
                                    self.cart_hess[k*3+l, i*3+m] += g_jk * s_k[l] * s_i[m]
                                
                                # j-k block
                                if j > k:
                                    self.cart_hess[j*3+l, k*3+m] += g_jk * s_j[l] * s_k[m]
                                else:
                                    self.cart_hess[k*3+l, j*3+m] += g_jk * s_k[l] * s_j[m]
                        
                        # Diagonal blocks
                        for l in range(3):
                            for m in range(l):
                                self.cart_hess[j*3+l, j*3+m] += g_jk * s_j[l] * s_j[m]
                                self.cart_hess[i*3+l, i*3+m] += g_jk * s_i[l] * s_i[m]
                                self.cart_hess[k*3+l, k*3+m] += g_jk * s_k[l] * s_k[m]
                    
                    else:  # Linear case
                        # Handle linear angles using arbitrary perpendicular vectors
                        if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                            x_1, y_1, z_1 = -y_ij, x_ij, 0.0
                            x_2, y_2, z_2 = -x_ij * z_ij, -y_ij * z_ij, x_ij**2 + y_ij**2
                        else:
                            x_1, y_1, z_1 = 1.0, 0.0, 0.0
                            x_2, y_2, z_2 = 0.0, 1.0, 0.0
                        
                        x = [x_1, x_2]
                        y = [y_1, y_2]
                        z = [z_1, z_2]
                        
                        # Calculate derivatives for two perpendicular directions
                        for ii in range(2):
                            r_1 = np.sqrt(x[ii]**2 + y[ii]**2 + z[ii]**2)
                            cos_theta_x = x[ii] / r_1
                            cos_theta_y = y[ii] / r_1
                            cos_theta_z = z[ii] / r_1
                            
                            # Derivatives
                            s_xj = -cos_theta_x / r_ij
                            s_yj = -cos_theta_y / r_ij
                            s_zj = -cos_theta_z / r_ij
                            s_xk = -cos_theta_x / r_ik
                            s_yk = -cos_theta_y / r_ik
                            s_zk = -cos_theta_z / r_ik
                            
                            s_xi = -s_xj - s_xk
                            s_yi = -s_yj - s_yk
                            s_zi = -s_zj - s_zk
                            
                            s_i = [s_xi, s_yi, s_zi]
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            
                            # Update Hessian elements
                            for l in range(3):
                                for m in range(3):
                                    # i-j block
                                    if i > j:
                                        self.cart_hess[i*3+l, j*3+m] += g_jk * s_i[l] * s_j[m]
                                    else:
                                        self.cart_hess[j*3+l, i*3+m] += g_jk * s_j[l] * s_i[m]
                                    
                                    # i-k block
                                    if i > k:
                                        self.cart_hess[i*3+l, k*3+m] += g_jk * s_i[l] * s_k[m]
                                    else:
                                        self.cart_hess[k*3+l, i*3+m] += g_jk * s_k[l] * s_i[m]
                                    
                                    # j-k block
                                    if j > k:
                                        self.cart_hess[j*3+l, k*3+m] += g_jk * s_j[l] * s_k[m]
                                    else:
                                        self.cart_hess[k*3+l, j*3+m] += g_jk * s_k[l] * s_j[m]
                            
                            # Diagonal blocks
                            for l in range(3):
                                for m in range(l):
                                    self.cart_hess[j*3+l, j*3+m] += g_jk * s_j[l] * s_j[m]
                                    self.cart_hess[i*3+l, i*3+m] += g_jk * s_i[l] * s_i[m]
                                    self.cart_hess[k*3+l, k*3+m] += g_jk * s_k[l] * s_k[m]
    
    def lindh2007_dihedral_angle(self, coord, element_list, charges, cn):
        """
        Calculate dihedral angle (torsion) contributions to the Hessian with D4 dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            charges: Atomic partial charges
            cn: Coordination numbers
        """
        n_atoms = len(coord)
        
        for j in range(n_atoms):
            t_xyz_2 = coord[j]
            
            for k in range(j+1, n_atoms):
                t_xyz_3 = coord[k]
                
                for i in range(j):
                    if i == k:
                        continue
                        
                    t_xyz_1 = coord[i]
                    
                    for l in range(k+1, n_atoms):
                        if l == i or l == j:
                            continue
                        
                        t_xyz_4 = coord[l]
                        
                        # Get element indices for parameter lookup
                        i_idx = self.select_idx(element_list[i])
                        j_idx = self.select_idx(element_list[j])
                        k_idx = self.select_idx(element_list[k])
                        l_idx = self.select_idx(element_list[l])
                        
                        # Get Lindh parameters
                        r_ij_0 = self.rAv[i_idx][j_idx]
                        d_ij_0 = self.dAv[i_idx][j_idx]
                        alpha_ij = self.aAv[i_idx][j_idx]
                        
                        r_jk_0 = self.rAv[j_idx][k_idx]
                        d_jk_0 = self.dAv[j_idx][k_idx]
                        alpha_jk = self.aAv[j_idx][k_idx]
                        
                        r_kl_0 = self.rAv[k_idx][l_idx]
                        d_kl_0 = self.dAv[k_idx][l_idx]
                        alpha_kl = self.aAv[k_idx][l_idx]
                        
                        # Calculate bond vectors and lengths
                        r_ij = coord[i] - coord[j]
                        r_jk = coord[j] - coord[k]
                        r_kl = coord[k] - coord[l]
                        
                        # Get bond parameters
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        # Calculate squared distances
                        r_ij_2 = np.sum(r_ij**2)
                        r_jk_2 = np.sum(r_jk**2)
                        r_kl_2 = np.sum(r_kl**2)
                        
                        # Calculate norms
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                        
                        # Check if near-linear angles would cause numerical issues
                        a35 = (35.0/180) * np.pi
                        cosfi_max = np.cos(a35)
                        
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                            
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        
                        # Get D4 parameters for bond pairs with charge scaling
                        c6_ij, c8_ij, r0_ij, q_ij = self.get_d4_parameters(
                            element_list[i], element_list[j], charges[i], charges[j])
                        c6_jk, c8_jk, r0_jk, q_jk = self.get_d4_parameters(
                            element_list[j], element_list[k], charges[j], charges[k])
                        c6_kl, c8_kl, r0_kl, q_kl = self.get_d4_parameters(
                            element_list[k], element_list[l], charges[k], charges[l])
                        
                        # Calculate force constants with D4 contributions
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                        if norm_r_ij > 2.0 * covalent_length_ij:
                            g_ij += 0.5 * self.kd * self.calc_d4_force_const(
                                norm_r_ij, c6_ij, c8_ij, r0_ij, q_ij)
                        
                        g_jk = self.calc_force_const(alpha_jk, covalent_length_jk, r_jk_2)
                        if norm_r_jk > 2.0 * covalent_length_jk:
                            g_jk += 0.5 * self.kd * self.calc_d4_force_const(
                                norm_r_jk, c6_jk, c8_jk, r0_jk, q_jk)
                        
                        g_kl = self.calc_force_const(alpha_kl, covalent_length_kl, r_kl_2)
                        if norm_r_kl > 2.0 * covalent_length_kl:
                            g_kl += 0.5 * self.kd * self.calc_d4_force_const(
                                norm_r_kl, c6_kl, c8_kl, r0_kl, q_kl)
                        
                        # Calculate torsion force constant
                        t_ij = self.kt * (g_ij * 0.5 * self.kd / self.kr * d_ij_0) * \
                              (g_jk * 0.5 * self.kd / self.kr * d_jk_0) * \
                              (g_kl * 0.5 * self.kd / self.kr * d_kl_0)
                        
                        # Calculate torsion angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        
                        # Extract derivatives
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Apply coordination number scaling (D4-specific)
                        # Torsions involving sp3 centers get higher weight
                        cn_scaling = 1.0
                        if 3.9 <= cn[j] <= 4.1 and 3.9 <= cn[k] <= 4.1:  # Both sp3
                            cn_scaling = 1.2
                        elif (cn[j] <= 3.1 and cn[k] <= 3.1):  # Both sp2 or lower
                            cn_scaling = 0.8
                        
                        # Apply scaling to force constant
                        t_ij *= cn_scaling
                        
                        # Update off-diagonal blocks
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3*i+n, 3*j+m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3*i+n, 3*k+m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3*i+n, 3*l+m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3*j+n, 3*k+m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3*j+n, 3*l+m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3*k+n, 3*l+m] += t_ij * s_k[n] * s_l[m]
                        
                        # Update diagonal blocks (lower triangle)
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3*i+n, 3*i+m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3*j+n, 3*j+m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3*k+n, 3*k+m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3*l+n, 3*l+m] += t_ij * s_l[n] * s_l[m]
    
    def lindh2007_out_of_plane(self, coord, element_list, charges, cn):
        """
        Calculate out-of-plane bending contributions to the Hessian with D4 dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            charges: Atomic partial charges
            cn: Coordination numbers
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            t_xyz_4 = coord[i]
            
            for j in range(i+1, n_atoms):
                t_xyz_1 = coord[j]
                
                for k in range(j+1, n_atoms):
                    t_xyz_2 = coord[k]
                    
                    for l in range(k+1, n_atoms):
                        t_xyz_3 = coord[l]
                        
                        # Calculate bond vectors
                        r_ij = coord[i] - coord[j]
                        r_ik = coord[i] - coord[k]
                        r_il = coord[i] - coord[l]
                        
                        # Get bond parameters
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        # Get element indices for parameter lookup
                        idx_i = self.select_idx(element_list[i])
                        idx_j = self.select_idx(element_list[j])
                        idx_k = self.select_idx(element_list[k])
                        idx_l = self.select_idx(element_list[l])
                        
                        # Get Lindh parameters
                        d_ij_0 = self.dAv[idx_i][idx_j]
                        r_ij_0 = self.rAv[idx_i][idx_j]
                        alpha_ij = self.aAv[idx_i][idx_j]
                        
                        d_ik_0 = self.dAv[idx_i][idx_k]
                        r_ik_0 = self.rAv[idx_i][idx_k]
                        alpha_ik = self.aAv[idx_i][idx_k]
                        
                        d_il_0 = self.dAv[idx_i][idx_l]
                        r_il_0 = self.rAv[idx_i][idx_l]
                        alpha_il = self.aAv[idx_i][idx_l]
                        
                        # Calculate squared distances
                        r_ij_2 = np.sum(r_ij**2)
                        r_ik_2 = np.sum(r_ik**2)
                        r_il_2 = np.sum(r_il**2)
                        
                        # Calculate norms
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        # Check for near-linear angles that would cause numerical issues
                        cosfi2 = np.dot(r_ij, r_ik) / (norm_r_ij * norm_r_ik)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                            
                        cosfi3 = np.dot(r_ij, r_il) / (norm_r_ij * norm_r_il)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                            
                        cosfi4 = np.dot(r_ik, r_il) / (norm_r_ik * norm_r_il)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        # Get D4 parameters for each pair with charge scaling
                        c6_ij, c8_ij, r0_ij, q_ij = self.get_d4_parameters(
                            element_list[i], element_list[j], charges[i], charges[j])
                        c6_ik, c8_ik, r0_ik, q_ik = self.get_d4_parameters(
                            element_list[i], element_list[k], charges[i], charges[k])
                        c6_il, c8_il, r0_il, q_il = self.get_d4_parameters(
                            element_list[i], element_list[l], charges[i], charges[l])
                        
                        # Disable direct D4 contributions to out-of-plane terms
                        kd = 0.0
                        
                        # Calculate force constants for each bond
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                        if norm_r_ij > 2.0 * covalent_length_ij:
                            g_ij += 0.5 * kd * self.calc_d4_force_const(
                                              norm_r_ij, c6_ij, c8_ij, r0_ij, q_ij)
                        
                        g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2)
                        if norm_r_ik > 2.0 * covalent_length_ik:
                            g_ik += 0.5 * kd * self.calc_d4_force_const(
                                norm_r_ik, c6_ik, c8_ik, r0_ik, q_ik)
                        
                        g_il = self.calc_force_const(alpha_il, covalent_length_il, r_il_2)
                        if norm_r_il > 2.0 * covalent_length_il:
                            g_il += 0.5 * kd * self.calc_d4_force_const(
                                norm_r_il, c6_il, c8_il, r0_il, q_il)
                        
                        # Combined force constant for out-of-plane motion
                        t_ij = self.ko * g_ij * g_ik * g_il
                        
                        # Apply special treatment for planar centers (D4 enhancement)
                        if 2.9 <= cn[i] <= 3.1:  # Planar center (CN=3 typical for sp2)
                            t_ij *= 1.2  # Increase force constant for planar centers
                        
                        # Calculate out-of-plane angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        # Extract derivatives
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update off-diagonal blocks
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i*3+n, j*3+m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i*3+n, k*3+m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i*3+n, l*3+m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j*3+n, k*3+m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j*3+n, l*3+m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k*3+n, l*3+m] += t_ij * s_k[n] * s_l[m]
                        
                        # Update diagonal blocks (lower triangle)
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i*3+n, i*3+m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j*3+n, j*3+m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k*3+n, k*3+m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l*3+n, l*3+m] += t_ij * s_l[n] * s_l[m]
    
    def calculate_three_body_terms(self, coord, element_list, charges, cn):
        """
        Calculate D4-specific three-body dispersion contributions.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            charges: Atomic partial charges
            cn: Coordination numbers
        """
        n_atoms = len(coord)
        
        # Only apply three-body terms for larger systems to improve efficiency
        if n_atoms < 3:
            return
        
        # Apply D4 three-body terms
        # This is a simplified implementation for Hessian calculations
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Skip if atoms are too close (likely bonded)
                cov_cutoff = (covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])) * 1.5
                if r_ij < cov_cutoff:
                    continue
                    
                for k in range(j+1, n_atoms):
                    r_ik = np.linalg.norm(coord[i] - coord[k])
                    r_jk = np.linalg.norm(coord[j] - coord[k])
                    
                    # Skip if atoms are too close (likely bonded)
                    if r_ik < (covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])) * 1.5:
                        continue
                    if r_jk < (covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])) * 1.5:
                        continue
                    
                    # Get D4 parameters
                    c6_ij, c8_ij, r0_ij, q_ij = self.get_d4_parameters(
                        element_list[i], element_list[j], charges[i], charges[j])
                    c6_ik, c8_ik, r0_ik, q_ik = self.get_d4_parameters(
                        element_list[i], element_list[k], charges[i], charges[k])
                    c6_jk, c8_jk, r0_jk, q_jk = self.get_d4_parameters(
                        element_list[j], element_list[k], charges[j], charges[k])
                    
                    # Calculate C9 coefficient for three-body term (Axilrod-Teller-Muto)
                    c9_ijk = np.sqrt(c6_ij * c6_ik * c6_jk)
                    
                    # Apply charge scaling
                    q_scaling = np.exp(-self.d4params.gc * (charges[i]**2 + charges[j]**2 + charges[k]**2))
                    
                    # Skip for very long-range interactions (r > 15 Bohr)
                    if r_ij > 15.0 or r_ik > 15.0 or r_jk > 15.0:
                        continue
                    
                    # Calculate three-body term (simplified for Hessian implementation)
                    # Note: This is a very simplified approximation
                    threebody_scale = 0.002 * self.d4params.s9 * q_scaling * c9_ijk / (r_ij * r_ik * r_jk)**3
                    
                    # Add minimal contribution to Hessian - just main diagonal elements
                    # For proper implementation, full derivatives would be needed
                    for idx in [i, j, k]:
                        for n in range(3):
                            self.cart_hess[idx*3+n, idx*3+n] += threebody_scale
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate approximate Hessian using Lindh's 2007 model with D4 dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            cart_gradient: Cartesian gradient vector
            
        Returns:
            hess_proj: Projected approximate Hessian matrix
        """
        print("Generating Lindh's (2007) approximate Hessian with D4 dispersion...")
        
        # Scale eigenvalues based on gradient norm (smaller scale for larger gradients)
        norm_grad = np.linalg.norm(cart_gradient)
        scale = 0.1
        eigval_scale = scale * np.exp(-1 * norm_grad**2.0)
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate atomic charges for D4 scaling
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate coordination numbers for D4 scaling
        cn = self.calculate_coordination_numbers(coord, element_list)
        
        # Calculate individual contributions
        self.lindh2007_bond(coord, element_list, charges, cn)
        self.lindh2007_angle(coord, element_list, charges, cn)
        self.lindh2007_dihedral_angle(coord, element_list, charges, cn)
        self.lindh2007_out_of_plane(coord, element_list, charges, cn)
        
        # Add D4-specific three-body terms
        self.calculate_three_body_terms(coord, element_list, charges, cn)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                if abs(self.cart_hess[i, j]) < 1.0e-10:
                    self.cart_hess[i, j] = self.cart_hess[j, i]
                else:
                    self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out translational and rotational degrees of freedom
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        # Adjust eigenvalues for stability based on gradient magnitude
        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        hess_proj = np.dot(np.dot(eigenvectors, np.diag(np.abs(eigenvalues) * eigval_scale)), 
                          np.transpose(eigenvectors))
        
        return hess_proj

class FischerApproxHessian:
    def __init__(self):
        """
        Fischer's Model Hessian implementation
        Ref: Fischer and Almlöf, J. Phys. Chem., 1992, 96, 24, 9768–9774
        """
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.bond_factor = 1.3  # Bond detection threshold factor
        
    def calc_bond_force_const(self, r_ab, r_ab_cov):
        """Calculate force constant for bond stretching"""
        return 0.3601 * np.exp(-1.944 * (r_ab - r_ab_cov))
        
    def calc_bend_force_const(self, r_ab, r_ac, r_ab_cov, r_ac_cov):
        """Calculate force constant for angle bending"""
        return 0.089 + 0.11 / (r_ab_cov * r_ac_cov) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        )
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        """Calculate force constant for dihedral torsion"""
        return 0.0015 + 14.0 * max(bond_sum, 0) ** 0.57 / (r_ab * r_ab_cov) ** 4.0 * np.exp(
            -2.85 * (r_ab - r_ab_cov)
        )
    
    def get_bond_connectivity(self, coord, element_list):
        """
        Calculate bond connectivity matrix and related data
        Returns:
            bond_mat: Bond connectivity matrix
            dist_mat: Distance matrix between atoms
            pair_cov_radii_mat: Matrix of covalent radii sums
        """
        n_atoms = len(coord)
        dist_mat = np.zeros((n_atoms, n_atoms))
        pair_cov_radii_mat = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coord[i] - coord[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
                
                cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                pair_cov_radii_mat[i, j] = pair_cov_radii_mat[j, i] = cov_sum
        
        # Bond connectivity matrix (True if bond exists between atoms)
        bond_mat = dist_mat <= (pair_cov_radii_mat * self.bond_factor)
        np.fill_diagonal(bond_mat, False)  # No self-bonds
        
        return bond_mat, dist_mat, pair_cov_radii_mat
    
    def count_bonds_for_dihedral(self, bond_mat, central_atoms):
        """Count bonds connected to the central atoms of a dihedral"""
        a, b = central_atoms
        # Sum bonds for both central atoms and subtract 2 (the bond between them is counted twice)
        bond_sum = bond_mat[a].sum() + bond_mat[b].sum() - 2
        return bond_sum
    
    def fischer_bond(self, coord, element_list):
        """Calculate Hessian components for bond stretching"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        bond_indices = BC.bond_connect_table(b_c_mat)
        
        for idx in bond_indices:
            i, j = idx
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            
            # Calculate force constant using Fischer's formula
            force_const = self.calc_bond_force_const(r_ij, r_ij_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j]])
            r, b_vec = stretch2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
    
    def fischer_angle(self, coord, element_list):
        """Calculate Hessian components for angle bending"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        angle_indices = BC.angle_connect_table(b_c_mat)
        
        for idx in angle_indices:
            i, j, k = idx  # i-j-k angle
            r_ij = np.linalg.norm(coord[i] - coord[j])
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_ij_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Calculate force constant using Fischer's formula
            force_const = self.calc_bend_force_const(r_ij, r_jk, r_ij_cov, r_jk_cov)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k]])
            theta, b_vec = bend2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
    
    def fischer_dihedral(self, coord, element_list, bond_mat):
        """Calculate Hessian components for dihedral torsions"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        for idx in dihedral_indices:
            i, j, k, l = idx  # i-j-k-l dihedral
            
            # Central bond in dihedral
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Count bonds to central atoms
            bond_sum = self.count_bonds_for_dihedral(bond_mat, (j, k))
            
            # Calculate force constant using Fischer's formula
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            tau, b_vec = torsion2(t_xyz)
            
            for n in range(3):
                for m in range(3):
                    self.cart_hess[3*i+n, 3*i+m] += force_const * b_vec[0][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*j+m] += force_const * b_vec[1][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*k+m] += force_const * b_vec[2][n] * b_vec[2][m]
                    self.cart_hess[3*l+n, 3*l+m] += force_const * b_vec[3][n] * b_vec[3][m]
                    
                    self.cart_hess[3*i+n, 3*j+m] += force_const * b_vec[0][n] * b_vec[1][m]
                    self.cart_hess[3*i+n, 3*k+m] += force_const * b_vec[0][n] * b_vec[2][m]
                    self.cart_hess[3*i+n, 3*l+m] += force_const * b_vec[0][n] * b_vec[3][m]
                    
                    self.cart_hess[3*j+n, 3*i+m] += force_const * b_vec[1][n] * b_vec[0][m]
                    self.cart_hess[3*j+n, 3*k+m] += force_const * b_vec[1][n] * b_vec[2][m]
                    self.cart_hess[3*j+n, 3*l+m] += force_const * b_vec[1][n] * b_vec[3][m]
                    
                    self.cart_hess[3*k+n, 3*i+m] += force_const * b_vec[2][n] * b_vec[0][m]
                    self.cart_hess[3*k+n, 3*j+m] += force_const * b_vec[2][n] * b_vec[1][m]
                    self.cart_hess[3*k+n, 3*l+m] += force_const * b_vec[2][n] * b_vec[3][m]
                    
                    self.cart_hess[3*l+n, 3*i+m] += force_const * b_vec[3][n] * b_vec[0][m]
                    self.cart_hess[3*l+n, 3*j+m] += force_const * b_vec[3][n] * b_vec[1][m]
                    self.cart_hess[3*l+n, 3*k+m] += force_const * b_vec[3][n] * b_vec[2][m]
    
    def main(self, coord, element_list, cart_gradient):
        """Main function to generate Fischer's approximate Hessian"""
        print("Generating Fischer's approximate hessian...")
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Get bond connectivity and distance information
        bond_mat, dist_mat, pair_cov_radii_mat = self.get_bond_connectivity(coord, element_list)
        
        # Calculate Hessian components
        self.fischer_bond(coord, element_list)
        self.fischer_angle(coord, element_list)
        self.fischer_dihedral(coord, element_list, bond_mat)
        
        # Symmetrize the Hessian
        for i in range(n_atoms*3):
            for j in range(i):
                self.cart_hess[i, j] = self.cart_hess[j, i]
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        return hess_proj


class TransitionStateHessian:
    """
    A class for modifying an existing Hessian matrix to introduce a negative eigenvalue 
    for transition state optimizations, but without using neg_eigenvalue directly as the replacement.
    Instead, it applies a procedure to the targeted eigenvalue: first take its absolute value,
    multiply by -1, and then add neg_eigenvalue.
    """

    def __init__(self):
        pass

    def create_ts_hessian(self, model_hessian, cart_gradient,
                          mode="auto", ts_mode_index=None, neg_eigenvalue=-0.000):
        """
        Generate a transition state Hessian matrix by introducing at least one negative eigenvalue 
        using the following procedure for the targeted eigenvalue:

            target_eig = -(|original_eig|) + neg_eigenvalue

        Parameters
        ----------
        model_hessian : numpy.ndarray
            The Hessian matrix already generated by an existing approximate method.
        cart_gradient : numpy.ndarray
            Cartesian gradient array.
        mode : str
            Method of choosing which eigenvalue to adjust:
            - "auto": Use the mode corresponding to the smallest eigenvalue.
            - "gradient": Use the mode that best overlaps with the gradient vector.
            - "custom": Use the mode specified by ts_mode_index.
        ts_mode_index : int
            If mode is "custom", this specifies which eigenvalue index to adjust.
        neg_eigenvalue : float
            A value added after flipping the eigenvalue to negative. Default is -0.000.

        Returns
        -------
        numpy.ndarray
            A Hessian matrix suitable for initiating a transition state search.
        """
        # Diagonalize the supplied Hessian
        eigenvalues, eigenvectors = np.linalg.eigh(model_hessian)

        # Select which mode to adjust
        if mode == "auto":
            mode_index = np.argmin(eigenvalues)
        elif mode == "gradient":
            flat_gradient = cart_gradient.flatten()
            grad_norm = np.linalg.norm(flat_gradient)
            if grad_norm > 1e-8:
                flat_gradient = flat_gradient / grad_norm
                overlaps = []
                for i in range(len(eigenvalues)):
                    overlaps.append(abs(np.dot(flat_gradient, eigenvectors[:, i])))
                mode_index = int(np.argmax(overlaps))
            else:
                mode_index = 0
        elif mode == "custom" and ts_mode_index is not None:
            mode_index = ts_mode_index
        else:
            mode_index = 0

        modified_eigenvalues = eigenvalues.copy()
        original_value = modified_eigenvalues[mode_index]
        modified_eigenvalues[mode_index] = -abs(original_value) + neg_eigenvalue

        diag_matrix = np.diag(modified_eigenvalues)
        temp_product = np.dot(eigenvectors, diag_matrix)
        ts_hessian = np.dot(temp_product, eigenvectors.T)

        # Enforce symmetry
        ts_hessian = 0.5 * (ts_hessian + ts_hessian.T)

        return ts_hessian


class MorseApproxHessian:
    """
    A simple class to generate a model Hessian based on the second derivative
    of a Morse potential, using GNB_radii_lib for covalent radii to estimate
    equilibrium bond distances. This is a highly simplified illustration.

    In this version, the covalent radii are obtained from GNB_radii_lib(element).
    """

    def __init__(self, De=0.10, a=0.20):
        """
        Parameters
        ----------
        De : float
            Dissociation energy in arbitrary units (e.g., Hartree).
        a : float
            Range parameter for the Morse potential.
        """
        self.De = De
        self.a = a

    def estimate_bond_length(self, elem1, elem2):
        """
        Estimate equilibrium bond length using GNB_radii_lib for each element.
        """
        r1 = GNB_radii_lib(elem1)
        r2 = GNB_radii_lib(elem2)
        return r1 + r2

    def compute_morse_second_derivative(self, r_current, r_eq):
        """
        Compute the second derivative of the Morse potential with respect to r,
        evaluated at r_current.

        V(r) = De * [1 - exp(-a * (r - r_eq))]^2
        
        For simplicity, use a general expanded form for the second derivative:
          d^2V/dr^2 = De * a^2 [ -2 e^{-x} + 4 e^{-2x} ]
        where x = a (r - r_eq).
        """
        x = self.a * (r_current - r_eq)
        # Expanded form for d^2V/dr^2
        second_derivative = self.De * (self.a ** 2) * (-2.0 * np.exp(-x) + 4.0 * np.exp(-2.0 * x))
        return second_derivative

    def create_model_hessian(self, coord, element_list):
        """
        Create a simple Hessian matrix for pairwise bonds as if
        each interaction is an independent Morse potential.

        Parameters
        ----------
        coord : numpy.ndarray
            Shape (N, 3) array of 3D coordinates for N atoms (in Å).
        element_list : list
            List of element symbols corresponding to the coordinates.

        Returns
        -------
        numpy.ndarray
            Hessian matrix of shape (3N, 3N).
        """
        n_atoms = len(element_list)
        hessian_size = 3 * n_atoms
        hessian = np.zeros((hessian_size, hessian_size), dtype=float)

        # Pairwise approach to generate naive bond Hessian elements
        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                # Estimate the equilibrium bond length
                r_eq = self.estimate_bond_length(element_list[i], element_list[j])

                # Current distance
                vec_ij = coord[j] - coord[i]
                dist_ij = np.linalg.norm(vec_ij)

                # Compute second derivative for the Morse potential
                d2V = self.compute_morse_second_derivative(dist_ij, r_eq)

                # Handle direction vector
                if dist_ij > 1.0e-12:
                    direction = vec_ij / dist_ij
                else:
                    direction = np.zeros(3)

                # Construct the 3x3 block k_ij * (direction outer direction)
                bond_k = d2V * np.outer(direction, direction)

                # Indices in the full Hessian
                block_i = slice(3 * i, 3 * i + 3)
                block_j = slice(3 * j, 3 * j + 3)

                # Update diagonal blocks
                hessian[block_i, block_i] += bond_k
                hessian[block_j, block_j] += bond_k
                # Update off-diagonal blocks
                hessian[block_i, block_j] -= bond_k
                hessian[block_j, block_i] -= bond_k

        # Symmetrize just in case
        hessian = 0.5 * (hessian + hessian.T)
        return hessian



class ShortRangeCorrectionHessian:
    """
    Class for calculating short-range correction to model Hessians, excluding bonded atom pairs.
    
    This class computes the second derivatives of the short-range part of
    the Coulomb operator used in range-separated hybrid functionals (e.g., ωB97X-D).
    The short-range part is defined as (1-erf(ω*r))/r, where ω is the range-separation parameter.
    
    References:
    [1] J.-D. Chai and M. Head-Gordon, J. Chem. Phys., 2008, 128, 084106 (ωB97X)
    [2] J.-D. Chai and M. Head-Gordon, Phys. Chem. Chem. Phys., 2008, 10, 6615 (ωB97X-D)
    """
    def __init__(self, omega=0.2, cx_sr=0.78, scaling_factor=0.5):
        """Initialize the ShortRangeCorrectionHessian class.
        
        Parameters:
        -----------
        omega : float
            Range-separation parameter in Bohr^-1 (default: 0.2 for ωB97X-D)
        cx_sr : float
            Short-range DFT exchange coefficient (default: 0.78 for ωB97X-D)
        scaling_factor : float
            Overall scaling factor for the correction (default: 0.5)
        """
        self.omega = omega                # Range-separation parameter (Bohr^-1)
        self.cx_sr = cx_sr                # Short-range exchange coefficient
        self.scaling_factor = scaling_factor  # Overall scaling factor
        self.sr_cutoff = 15.0             # Cutoff distance for short-range interactions (Bohr)
        
    def detect_bonds(self, coord, element_list):
        """Detect bonded atom pairs in the molecule.
        
        Parameters:
        -----------
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
            
        Returns:
        --------
        set
            Set of tuples (i,j) representing bonded atom pairs
        """
        # Use BondConnectivity class from MultiOptPy to detect bonds
        bc = BondConnectivity()
        bond_matrix = bc.bond_connect_matrix(element_list, coord)
        
        # Create a set of bonded atom pairs
        bonded_pairs = set()
        for i in range(len(coord)):
            for j in range(i+1, len(coord)):
                if bond_matrix[i, j] == 1:
                    bonded_pairs.add((i, j))
                    bonded_pairs.add((j, i))
        
        return bonded_pairs
        
    def sr_coulomb(self, r):
        """Calculate short-range Coulomb potential.
        
        V_SR(r) = (1 - erf(ω*r)) / r
        
        Parameters:
        -----------
        r : float
            Distance between two atoms (Bohr)
            
        Returns:
        --------
        float
            Short-range Coulomb potential
        """
        if r < 1e-10:
            # Use limit as r→0 (Taylor expansion)
            return 2 * self.omega / np.sqrt(np.pi)
        return (1.0 - erf(self.omega * r)) / r
    
    def sr_coulomb_first_derivative(self, r):
        """Calculate first derivative of short-range Coulomb potential.
        
        dV_SR(r)/dr = -V_SR(r)/r - 2ω/√π * exp(-ω²r²)/r
        
        Parameters:
        -----------
        r : float
            Distance between two atoms
            
        Returns:
        --------
        float
            First derivative of short-range Coulomb potential
        """
        if r < 1e-10:
            # Use limit as r→0 (Taylor expansion)
            return -2 * self.omega**3 / (3 * np.sqrt(np.pi))
        
        # Error function term
        erf_term = erf(self.omega * r)
        
        # Exponential term
        exp_term = 2 * self.omega * np.exp(-(self.omega * r)**2) / (np.sqrt(np.pi) * r)
        
        # Coulomb term
        coulomb_term = (erf_term - 1.0) / r**2
        
        return exp_term + coulomb_term
    
    def sr_coulomb_second_derivative(self, r):
        """Calculate second derivative of short-range Coulomb potential.
        
        d²V_SR(r)/dr² = 2(1-erf(ω*r))/r³ + 2erf(ω*r)/r³ + 4ω/(√π*r²)*e^(-ω²r²) + 2ω³/√π*e^(-ω²r²)
        
        Parameters:
        -----------
        r : float
            Distance between two atoms
            
        Returns:
        --------
        float
            Second derivative of short-range Coulomb potential
        """
        if r < 1e-10:
            # Use limit as r→0 (Taylor expansion)
            return 0.0
        
        # Error function term
        erf_term = erf(self.omega * r)
        
        # Exponential terms
        exp_factor = np.exp(-(self.omega * r)**2) / np.sqrt(np.pi)
        exp_term1 = 4 * self.omega * exp_factor / r**2
        exp_term2 = 2 * (self.omega**3) * exp_factor
        
        # Coulomb term
        coulomb_term = 2 * (2 * erf_term - 1) / r**3
        
        return coulomb_term + exp_term1 + exp_term2
    
    def estimate_atomic_charges(self, element_list):
        """Estimate atomic charges based on Pauling electronegativity.
        
        Parameters:
        -----------
        element_list : list
            List of element symbols
            
        Returns:
        --------
        numpy.ndarray
            Estimated atomic charges
        """
        # Pauling electronegativity values
        electronegativity = {
            'H': 2.20, 'He': 0.00,
            'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 
            'O': 3.44, 'F': 3.98, 'Ne': 0.00,
            'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 
            'S': 2.58, 'Cl': 3.16, 'Ar': 0.00,
            'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 
            'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 
            'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 
            'Se': 2.55, 'Br': 2.96, 'Kr': 0.00
        }
        
        n_atoms = len(element_list)
        charges = np.zeros(n_atoms)
        
        # Calculate average electronegativity (reference value)
        en_values = [electronegativity.get(element, 2.0) for element in element_list]
        avg_en = sum(en_values) / len(en_values)
        
        # Assign charges based on electronegativity differences
        for i, element in enumerate(element_list):
            en = electronegativity.get(element, 2.0)
            charges[i] = 0.2 * (avg_en - en)  # Scale by 0.2
        
        return charges
    
    def calculate_pair_hessian(self, r_vec, r_ij, atomic_charges, atom_i, atom_j):
        """Calculate Hessian contribution from short-range Coulomb between atom pair.
        
        Parameters:
        -----------
        r_vec : numpy.ndarray
            Relative position vector from atom i to atom j
        r_ij : float
            Distance between atoms i and j
        atomic_charges : numpy.ndarray
            Array of atomic charges
        atom_i, atom_j : int
            Atom indices
            
        Returns:
        --------
        numpy.ndarray
            3x3 Hessian block matrix
        """
        # Return zeros if beyond cutoff distance
        if r_ij > self.sr_cutoff:
            return np.zeros((3, 3))
        
        # Unit direction vector
        r_unit = r_vec / r_ij
        
        # Charge-based coefficient
        q_i = atomic_charges[atom_i]
        q_j = atomic_charges[atom_j]
        q_factor = q_i * q_j * self.cx_sr * self.scaling_factor
        
        # Calculate second derivative
        d2v = self.sr_coulomb_second_derivative(r_ij)
        
        # Calculate tensor using outer product
        r_outer = np.outer(r_unit, r_unit)
        
        # Calculate Hessian block
        identity = np.eye(3)
        hessian_block = q_factor * (d2v * r_outer + 
                           self.sr_coulomb_first_derivative(r_ij) / r_ij * (identity - r_outer))
        
        return hessian_block
    
    def calculate_correction_hessian(self, coord, element_list):
        """Calculate complete short-range correction Hessian, excluding bonded pairs.
        
        Parameters:
        -----------
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
            
        Returns:
        --------
        numpy.ndarray
            Short-range correction Hessian
        """
        n_atoms = len(coord)
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
        
        # Detect bonded atom pairs
        bonded_pairs = self.detect_bonds(coord, element_list)
        
        # Estimate atomic charges
        atomic_charges = self.estimate_atomic_charges(element_list)
        
        # Number of bonded pairs and total pairs for statistics
        num_total_pairs = n_atoms * (n_atoms - 1) // 2
        num_bonded_pairs = len(bonded_pairs) // 2  # Divide by 2 because we stored both (i,j) and (j,i)
        print(f"Detected {num_bonded_pairs} bonded pairs out of {num_total_pairs} total pairs")
        print(f"Short-range correction will be applied to {num_total_pairs - num_bonded_pairs} non-bonded pairs only")
        
        # Loop over all atom pairs
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Skip bonded atom pairs
                if (i, j) in bonded_pairs or (j, i) in bonded_pairs:
                    continue
                
                # Calculate interatomic vector and distance
                r_vec = coord[j] - coord[i]
                r_ij = np.linalg.norm(r_vec)
                
                # Calculate Hessian block for this pair
                hess_block = self.calculate_pair_hessian(
                    r_vec, r_ij, atomic_charges, i, j
                )
                
                # Add to the Hessian matrix
                for a in range(3):
                    for b in range(3):
                        # Diagonal blocks
                        hessian[3*i+a, 3*i+b] += hess_block[a, b]
                        hessian[3*j+a, 3*j+b] += hess_block[a, b]
                        
                        # Off-diagonal blocks
                        hessian[3*i+a, 3*j+b] -= hess_block[a, b]
                        hessian[3*j+a, 3*i+b] -= hess_block[a, b]
        
        return hessian
    
    def apply_correction(self, base_hessian, coord, element_list):
        """Apply short-range correction to an existing Hessian.
        
        Parameters:
        -----------
        base_hessian : numpy.ndarray
            Base model Hessian
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
            
        Returns:
        --------
        numpy.ndarray
            Corrected Hessian
        """
        tools = Calculationtools()
        # Calculate short-range correction
        correction = self.calculate_correction_hessian(coord, element_list)
        correction = tools.project_out_hess_tr_and_rot_for_coord(correction, element_list, coord, display_eigval=False)
        # Add correction to base Hessian
        corrected_hessian = base_hessian + correction
        
        corrected_hessian = 0.5 * (corrected_hessian + corrected_hessian.T)  # Symmetrize
        # Remove translational and rotational modes
        
        return corrected_hessian
    
    def main(self, coord, element_list, base_hessian):
        """Main method to apply short-range correction to a model Hessian.
        
        Parameters:
        -----------
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
        base_hessian : numpy.ndarray
            Base model Hessian
            
        Returns:
        --------
        numpy.ndarray
            Hessian with short-range correction
        """
        print(f"Applying short-range correction (ω={self.omega:.3f}) to model Hessian...")
        print("The correction will be applied only to non-bonded atom pairs.")
        
        # Apply correction
        corrected_hessian = self.apply_correction(base_hessian, coord, element_list)
        
        # Handle NaN values
        corrected_hessian = np.nan_to_num(corrected_hessian, nan=0.0)
        
        print("Short-range correction applied successfully")
        return corrected_hessian

class ApproxHessian:
    def __init__(self):
        return
    
    def main(self, coord, element_list, cart_gradient, approx_hess_type="lindh2007d3"):
        #coord: Bohr
        
        
        if "gfnff" in approx_hess_type.lower():
            GFNFFAH = GFNFFApproxHessian()
            hess_proj = GFNFFAH.main(coord, element_list, cart_gradient)
        elif "gfn0xtb" in approx_hess_type.lower():
            GFN0AH = GFN0XTBApproxHessian()
            hess_proj = GFN0AH.main(coord, element_list, cart_gradient)
        elif "fischerd3" in approx_hess_type.lower():
            FAHD3 = FischerD3ApproxHessian()
            hess_proj = FAHD3.main(coord, element_list, cart_gradient)
        elif "fischerd4" in approx_hess_type.lower():
            FAHD4 = FischerD4ApproxHessian()
            hess_proj = FAHD4.main(coord, element_list, cart_gradient)
        
        elif "schlegeld3" in approx_hess_type.lower():
            SAHD3 = SchlegelD3ApproxHessian()
            hess_proj = SAHD3.main(coord, element_list, cart_gradient)
        elif "schlegeld4" in approx_hess_type.lower():
            SAHD4 = SchlegelD4ApproxHessian()
            hess_proj = SAHD4.main(coord, element_list, cart_gradient)
        elif "schlegel" in approx_hess_type.lower():
            SAH = SchlegelApproxHessian()
            hess_proj = SAH.main(coord, element_list, cart_gradient)
        
        elif "swartd3" in approx_hess_type.lower():
            SWHD3 = SwartD3ApproxHessian()
            hess_proj = SWHD3.main(coord, element_list, cart_gradient)
        elif "swartd4" in approx_hess_type.lower():
            SWHD4 = SwartD4ApproxHessian()
            hess_proj = SWHD4.main(coord, element_list, cart_gradient)
        elif "swart" in approx_hess_type.lower():
            SWH = SwartD2ApproxHessian()
            hess_proj = SWH.main(coord, element_list, cart_gradient)
        elif "lindh2007d3" in approx_hess_type.lower():
            LH2007D3 = Lindh2007D3ApproxHessian()
            hess_proj = LH2007D3.main(coord, element_list, cart_gradient)
        elif "lindh2007d4" in approx_hess_type.lower():
            LH2007D4 = Lindh2007D4ApproxHessian()
            hess_proj = LH2007D4.main(coord, element_list, cart_gradient)
        elif "lindh2007" in approx_hess_type.lower():
            LH2007 = Lindh2007D2ApproxHessian()
            hess_proj = LH2007.main(coord, element_list, cart_gradient)
        elif "lindh" in approx_hess_type.lower():
            LAH = LindhApproxHessian()
            hess_proj = LAH.main(coord, element_list, cart_gradient)
        elif "fischer" in approx_hess_type.lower():
            FH = FischerApproxHessian()
            hess_proj = FH.main(coord, element_list, cart_gradient)
        elif "morse" in approx_hess_type.lower():
            MH = MorseApproxHessian()
            hess_proj = MH.create_model_hessian(coord, element_list)
        else:
            print("Approximate Hessian type not recognized. Using default Lindh (2007) D3 model...")
            LH2007D3 = Lindh2007D3ApproxHessian()
            hess_proj = LH2007D3.main(coord, element_list, cart_gradient)
            
        if "ts" in approx_hess_type.lower():
            TSH = TransitionStateHessian()
            hess_proj = TSH.create_ts_hessian(hess_proj, cart_gradient)
        
        if "sr" in approx_hess_type.lower():
            SRCH = ShortRangeCorrectionHessian()
            hess_proj = SRCH.main(coord, element_list, hess_proj)
        
        return hess_proj#cart_hess


def test():
    AH = ApproxHessian()
    words = ["O        1.607230637      0.000000000     -4.017111134",
             "O        1.607230637      0.463701826     -2.637210910",
             "H        2.429229637      0.052572461     -2.324941515",
             "H        0.785231637     -0.516274287     -4.017735703"]
    
    elements = []
    coord = []
    
    for word in words:
        sw = word.split()
        elements.append(sw[0])
        coord.append(sw[1:4])
    
    coord = np.array(coord, dtype="float64")/UnitValueLib().bohr2angstroms#Bohr
    gradient = np.array([[-0.0028911  ,  -0.0015559   ,  0.0002471],
                         [ 0.0028769  ,  -0.0013954   ,  0.0007272],
                         [-0.0025737   ,  0.0013921   , -0.0007226],
                         [ 0.0025880   ,  0.0015592  ,  -0.0002518]], dtype="float64")#a. u.
    
    hess_proj = AH.main(coord, elements, gradient)
    
    return hess_proj



if __name__ == "__main__":#test
    test()
    
    
    
    