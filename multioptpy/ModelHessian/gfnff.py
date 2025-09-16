import numpy as np
from collections import defaultdict

from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import GFNFFParameters


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
