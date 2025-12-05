
import numpy as np

from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.ModelHessian.calc_params import stretch2, bend2, torsion2

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
        val = r_ab_cov * r_ac_cov
        if abs(val) < 1.0e-10:
            return 0.0  # Avoid division by zero
        
        return 0.089 + 0.11 / (val) ** (-0.42) * np.exp(
            -0.44 * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        ) 
        
    def calc_dihedral_force_const(self, r_ab, r_ab_cov, bond_sum):
        """Calculate force constant for dihedral torsion"""
        val = r_ab * r_ab_cov
        if abs(val) < 1.0e-10:
            return 0.0  # Avoid division by zero
        return 0.0015 + 14.0 * max(bond_sum, 0) ** 0.57 / (val) ** 4.0 * np.exp(
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
        """Calculate Hessian components for dihedral torsions with linearity check"""
        BC = BondConnectivity()
        b_c_mat = BC.bond_connect_matrix(element_list, coord)
        dihedral_indices = BC.dihedral_angle_connect_table(b_c_mat)
        
        # Helper to calculate sin^2 of bond angle
        def get_sin_sq_angle(idx1, idx2, idx3):
            v1 = coord[idx1] - coord[idx2]
            v2 = coord[idx3] - coord[idx2]
            # Cross product magnitude squared
            cross_prod = np.cross(v1, v2)
            cross_sq = np.dot(cross_prod, cross_prod)
            # Squared norms
            n1_sq = np.dot(v1, v1)
            n2_sq = np.dot(v2, v2)
            if n1_sq * n2_sq < 1e-12:
                return 0.0
            return cross_sq / (n1_sq * n2_sq)

        for idx in dihedral_indices:
            # i-j-k-l dihedral
            i, j, k, l = idx 
            
            # --- Linearity Check ---
            # Check bond angles I-J-K and J-K-L.
            # If atoms are linear, the torsion definition is singular (B-matrix blows up).
            # We use sin^2(theta) because it avoids acos and is efficient.
            sin_sq_ijk = get_sin_sq_angle(i, j, k)
            sin_sq_jkl = get_sin_sq_angle(j, k, l)
            
            # Threshold: if sin(theta) < 0.17 (approx 10 degrees from linear), dampen or skip.
            # Using sin^2 < 0.03 as cutoff.
            # The Wilson B-matrix scales as 1/sin(theta). 
            # If we are too close to linear, we must skip to avoid numerical explosion.
            if sin_sq_ijk < 1.0e-3 or sin_sq_jkl < 1.0e-3:
                continue
            # -----------------------

            # Central bond in dihedral (j-k)
            r_jk = np.linalg.norm(coord[j] - coord[k])
            r_jk_cov = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
            
            # Count bonds to central atoms
            bond_sum = self.count_bonds_for_dihedral(bond_mat, (j, k))
            
            # Calculate force constant
            force_const = self.calc_dihedral_force_const(r_jk, r_jk_cov, bond_sum)
            
            # Optional: Additional damping can be applied here if desired, 
            # but the skip above is the most robust fix for the singularity.
            
            # Convert to Cartesian coordinates
            t_xyz = np.array([coord[i], coord[j], coord[k], coord[l]])
            
            # Note: torsion2 might still be unstable if not skipped
            try:
                tau, b_vec = torsion2(t_xyz)
            except Exception:
                # Fallback if torsion2 fails numerically
                continue
            
            # Iterate over all pairs of atoms in the dihedral
            atom_indices = [i, j, k, l]
            
            for a in range(4):
                for b in range(4):
                    atom_a = atom_indices[a]
                    atom_b = atom_indices[b]
                    
                    vec_a = b_vec[a]
                    vec_b = b_vec[b]
                    
                    # Update the 3x3 Hessian block
                    for n in range(3):
                        for m in range(3):
                            val = force_const * vec_a[n] * vec_b[m]
                            self.cart_hess[3*atom_a+n, 3*atom_b+m] += val
                            
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
