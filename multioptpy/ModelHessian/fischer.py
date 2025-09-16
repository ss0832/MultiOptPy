
import numpy as np

from multioptpy.bond_connectivity import BondConnectivity
from multioptpy.parameter import UnitValueLib, covalent_radii_lib
from multioptpy.calc_tools import Calculationtools
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
