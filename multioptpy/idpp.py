import numpy as np
import copy

from parameter import (
                covalent_radii_lib,
                UFF_VDW_distance_lib,
                number_element,
                UnitValueLib,)

class IDPP:
    def __init__(self):
        #ref.: arXiv:1406.1512v1
        self.iteration = 2000
        self.lr = 0.01
        self.threshold = 1e-5
        return
    
    def calc_obj_func(self, idpp_dist_matrix, dist_matrix):
        idpp_upper_triangle_indices = np.triu_indices(idpp_dist_matrix.shape[0], k=1)
        idpp_upper_triangle_distances = idpp_dist_matrix[idpp_upper_triangle_indices]
        dist_upper_triangle_indices = np.triu_indices(dist_matrix.shape[0], k=1)
        dist_upper_triangle_distances = dist_matrix[dist_upper_triangle_indices]
        weight_func = (dist_upper_triangle_distances + 1e-15) ** (-4)
        obj_func = np.sum(weight_func * (idpp_upper_triangle_distances - dist_upper_triangle_distances) ** 2.0)
        return obj_func
    
    def calc_obj_func_1st_deriv(self, pos, idpp_dist_matrix, dist_matrix):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # Shape: (N, N, 3)
        distances = np.linalg.norm(diff, axis=-1)  # Shape: (N, N)
        valid_mask = distances > 0
        unit_diff = np.zeros_like(diff)
        unit_diff[valid_mask] = diff[valid_mask] / (distances[valid_mask][:, np.newaxis] + 1e-15)
        w = (distances + 1e-15)**(-4)
        dw_dr = -4 * (distances + 1e-15)**(-5)
        
        diff_matrix = idpp_dist_matrix - dist_matrix
        d_obj_func_d_qij = (
            (dw_dr * diff_matrix**2 - 2.0 * w * diff_matrix)[:, :, np.newaxis]
            * unit_diff
        )  # Shape: (N, N, 3)
  
        i_indices, j_indices = np.triu_indices(len(pos), k=1)
        
        first_deriv = np.zeros_like(pos)
        np.add.at(first_deriv, i_indices, d_obj_func_d_qij[i_indices, j_indices])
        np.subtract.at(first_deriv, j_indices, d_obj_func_d_qij[i_indices, j_indices])
        return first_deriv
    
    def calc_idpp_dist_matrix(self, pos_list, n_node, number_of_node):
        init_pos = pos_list[0]
        term_pos = pos_list[-1]
        init_pos_diff = init_pos[:, np.newaxis, :] - init_pos[np.newaxis, :, :]
        init_pos_dist_matrix = np.sqrt(np.sum(init_pos_diff**2, axis=-1))
        term_pos_diff = term_pos[:, np.newaxis, :] - term_pos[np.newaxis, :, :]
        term_pos_dist_matrix = np.sqrt(np.sum(term_pos_diff**2, axis=-1))
        idpp_dist_matrix = init_pos_dist_matrix + number_of_node * (term_pos_dist_matrix - init_pos_dist_matrix) / (n_node - 1)
       
        return idpp_dist_matrix
    
    def calc_dist_matrix(self, pos):
        pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(pos_diff**2, axis=-1))
        return dist_matrix
    
    def get_func_and_deriv(self, pos_list, n_node, number_of_node):
        dist_matrix = self.calc_dist_matrix(pos_list[number_of_node])
        idpp_dist_matrix = self.calc_idpp_dist_matrix(pos_list, n_node, number_of_node)
        obj_func = self.calc_obj_func(idpp_dist_matrix, dist_matrix)
        first_deriv = self.calc_obj_func_1st_deriv(pos_list[number_of_node], idpp_dist_matrix, dist_matrix)
        
        return obj_func, first_deriv
        
    
    def opt_path(self, geometry_list, element_list):
        print("IDPP Optimization")
        
        for i in range(self.iteration):
            obj_func_list = []
            #FBENM_instance = FBENM(element_list, geometry_list)
            for j in range(len(geometry_list)):
                
                if j == 0 or j == len(geometry_list) - 1:
                    continue
                
                obj_func_idpp, first_deriv_idpp = self.get_func_and_deriv(geometry_list, len(geometry_list), j)
                
                #obj_func_fbenm, first_deriv_fbenm = FBENM_instance.get_func_and_deriv(geometry_list[j])
                
                obj_func = obj_func_idpp# + obj_func_fbenm
                first_deriv = first_deriv_idpp# + first_deriv_fbenm
                
                step_norm = min(self.lr, np.linalg.norm(first_deriv))
                norm_step = first_deriv / np.linalg.norm(first_deriv)
                
                geometry_list[j] -= step_norm * norm_step
            
                obj_func_list.append(obj_func)
            if i % 200 == 0:
                print("ITR: ", i)
                print("Objective function (Max): ", max(obj_func_list))
        
            if max(obj_func_list) < self.threshold:
                print("ITR: ", i)
                print("IDPP Converged!!!")
                break
        print("IDPP Optimization Done.")
        return geometry_list


class FBENM:# This class is under construction. 
    """
    Flat-Bottom Elastic Network Model (FB-ENM) potential.
    ref.: J.Chem.TheoryComput.2024,20,7176−7187
    
    Potential per pair (i, j), with current distance r = |r_i - r_j|:
        V_FB-ENM = (r - d_min)^2 / (dd_min)^2  (if r < d_min)
                   0                           (if d_min <= r <= d_max)
                   (r - d_max)^2 / (dd_max)^2  (if r > d_max)

    Constraints:
    - Strong constraints (for bonded pairs): 
      d_min = d_max = distance in the endpoint structure that is closer to the current structure
    - Weak constraints (for non-bonded pairs): 
      d_min = 0.5 * (UFF vdW radii sum)
      d_max = maximum distance across the path
    """

    def __init__(
        self,
        elements,
        geometry_list,
        bond_scale=1.3,
        delta_scale=2.0,
        eps=1e-12,
    ):
        """
        Initialize FB-ENM model.

        Args:
            elements: list of element symbols (e.g., ['C','H',...]) or atomic numbers.
            geometry_list: list of (N,3) positions arrays. Must include endpoints [0] and [-1].
            bond_scale: multiplier for sum of covalent radii to determine bonding.
            delta_scale: scale factor for dd_min and dd_max.
            eps: small number for numerical stability.
        """
        self.elements = self.convert_to_symbols(elements)
        self.N = len(self.elements)
        assert len(geometry_list) >= 2, "geometry_list must include at least two images (endpoints)."
        self.geometry_list = [np.array(g, dtype=float, copy=True) for g in geometry_list]
        self.bond_scale = float(bond_scale)
        self.delta_scale = float(delta_scale)
        self.eps = float(eps)

        self.covalent_radii_lib = covalent_radii_lib
        self.UFF_VDW_distance_lib = UFF_VDW_distance_lib
        self.number_element = number_element

        # Precompute endpoint distances
        self.d0 = self.pairwise_distances(self.geometry_list[0])
        self.dL = self.pairwise_distances(self.geometry_list[-1])

        # Precompute per-pair maximum distance across path
        self.d_max_path = self.calculate_max_distances(self.geometry_list)

        # Precompute per-pair covalent radii sum and UFF vdW radii sum
        self.cov_sum = self.calculate_covalent_sum(self.elements)
        self.vdw_sum = self.calculate_vdw_sum(self.elements)

    def convert_to_symbols(self, elements):
        """Convert a list of element identifiers to symbols."""
        syms = []
        for e in elements:
            if isinstance(e, str):
                syms.append(e)
            else:
                syms.append(number_element(int(e)))
        return syms

    def pairwise_distances(self, pos):
        """Compute pairwise distance matrix (N,N)."""
        dr = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(dr, axis=-1)

    def calculate_max_distances(self, geometry_list):
        """Compute per-pair maximum distance across a list of images."""
        N = geometry_list[0].shape[0]
        dmax = np.zeros((N, N), dtype=float)
        
        for k, g in enumerate(geometry_list):
            d = self.pairwise_distances(g)
            if k == 0:
                dmax = d
            else:
                dmax = np.maximum(dmax, d)
        # Symmetrize and zero diagonals
        dmax = 0.5 * (dmax + dmax.T)
        np.fill_diagonal(dmax, 0.0)
        return dmax * 2.0

    def calculate_covalent_sum(self, elements):
        """Sum of covalent radii per pair."""
        N = len(elements)
        r = np.zeros((N,), dtype=float)
        for i, e in enumerate(elements):
            r[i] = self.covalent_radii_lib(e) * UnitValueLib().bohr2angstroms
        rs = r[:, None] + r[None, :]
        np.fill_diagonal(rs, 0.0)
        return rs

    def calculate_vdw_sum(self, elements):
        """Sum of UFF vdW radii per pair."""
        N = len(elements)
        r = np.zeros((N,), dtype=float)
        for i, e in enumerate(elements):
            r[i] = self.UFF_VDW_distance_lib(e) * UnitValueLib().bohr2angstroms
        rs = r[:, None] + r[None, :]
        np.fill_diagonal(rs, 0.0)
        return rs

    def build_band_parameters(self, pos):
        """
        Build per-pair band parameters (d_min, d_max, dd_min, dd_max) using vectorized operations.
        """
        r = self.pairwise_distances(pos)
        N = len(r)

        # Determine which pairs are bonded (covalent)
        bonded_threshold = self.bond_scale * self.cov_sum
        bonded = r <= bonded_threshold
        
        # Initialize d_min and d_max arrays
        d_min = np.zeros((N, N), dtype=float)
        d_max = np.zeros((N, N), dtype=float)
        
        # Create masks for upper triangle (to avoid duplicate work)
        triu_indices = np.triu_indices(N, k=1)
        
        # Extract relevant distances for all upper triangle pairs
        current_dists = r[triu_indices]
        d0_dists = self.d0[triu_indices]
        dL_dists = self.dL[triu_indices]
        
        # Determine which endpoint is closer for each pair
        use_d0 = np.abs(current_dists - d0_dists) <= np.abs(current_dists - dL_dists)
        
        # Create target distances array based on the closer endpoint
        target_dists = np.where(use_d0, d0_dists, dL_dists)
        
        # Create mask arrays for bonded and non-bonded pairs in upper triangle
        bonded_pairs = bonded[triu_indices]
        bonded_pairs[:] = False
        nonbonded_pairs = ~bonded_pairs
        
        # Create temporary arrays for upper triangle values
        d_min_upper = np.zeros_like(current_dists)
        d_max_upper = np.zeros_like(current_dists)
        
        # Set values for bonded pairs (strong constraint)
        d_min_upper[bonded_pairs] = target_dists[bonded_pairs]
        d_max_upper[bonded_pairs] = target_dists[bonded_pairs]
        
        # Set values for non-bonded pairs (weak constraint)
        vdw_values = self.vdw_sum[triu_indices]
        dmax_values = self.d_max_path[triu_indices]
        
        d_min_upper[nonbonded_pairs] = vdw_values[nonbonded_pairs]
        d_max_upper[nonbonded_pairs] = dmax_values[nonbonded_pairs]
        
        # Fill the upper triangle of the matrices
        d_min[triu_indices] = d_min_upper
        d_max[triu_indices] = d_max_upper
        
        # Make symmetric by adding transpose (diagonal will be doubled, but we zero it later)
        d_min = d_min + d_min.T
        d_max = d_max + d_max.T
        
        # Zero the diagonal to handle division safely
        np.fill_diagonal(d_min, 0.0)
        np.fill_diagonal(d_max, 0.0)
        
        # Calculate delta parameters
        dd_min = self.delta_scale * np.maximum(d_min, self.eps)
        dd_max = self.delta_scale * np.maximum(d_max, self.eps)
        np.fill_diagonal(dd_min, 1.0)
        np.fill_diagonal(dd_max, 1.0)
        
        return d_min, d_max, dd_min, dd_max

    def get_func_and_deriv(self, pos):
        """
        Compute FB-ENM energy and Cartesian gradient.

        Args:
            pos: (N,3) array of positions.

        Returns:
            energy: float
            grad: (N,3) array, gradient dE/dR
        """
        pos = np.asarray(pos, dtype=float)
        N = pos.shape[0]
        assert N == self.N, f"Position size {N} does not match the number of elements {self.N}."

        # Pairwise differences and distances
        dr = pos[:, None, :] - pos[None, :, :]  # (N,N,3)
        r = np.linalg.norm(dr, axis=-1)         # (N,N)
        
        # Safe reciprocal distance (for gradient calculation)
        rinv = np.zeros_like(r)
        mask = r > 0.0
        rinv[mask] = 1.0 / r[mask]

        # Build band parameters for the current configuration
        d_min, d_max, dd_min, dd_max = self.build_band_parameters(pos)

        # Compute energy contributions per pair
        energy_rep = np.zeros_like(r)
        energy_att = np.zeros_like(r)
        
        # Repulsive contribution (r < d_min)
        rep_mask = r < d_min
        if np.any(rep_mask):
            energy_rep[rep_mask] = ((r[rep_mask] - d_min[rep_mask]) / dd_min[rep_mask]) ** 2
            
        # Attractive contribution (r > d_max)
        att_mask = r > d_max
        if np.any(att_mask):
            energy_att[att_mask] = ((r[att_mask] - d_max[att_mask]) / dd_max[att_mask]) ** 2

        # Total pair energy
        energy_pairs = energy_rep + energy_att
        
        # Calculate derivatives dV/dr for gradient
        dVdr = np.zeros_like(r)
        
        # Derivative for repulsion (r < d_min)
        if np.any(rep_mask):
            dVdr[rep_mask] = 2.0 * (r[rep_mask] - d_min[rep_mask]) / (dd_min[rep_mask] ** 2)
            
        # Derivative for attraction (r > d_max)
        if np.any(att_mask):
            dVdr[att_mask] = 2.0 * (r[att_mask] - d_max[att_mask]) / (dd_max[att_mask] ** 2)
        
        # Compute gradient: dE/dr_i = sum_j [(dV_ij/dr_ij) * (r_i - r_j) / r_ij]
        # Vectorized version: multiply weights by direction vectors and sum
        weights = dVdr * rinv  # (N,N)
        grad = np.einsum('ij,ijk->ik', weights, dr)  # Einstein summation to compute weighted sum

        # Total energy (0.5 * sum to avoid double-counting)
        total_energy = 0.5 * np.sum(energy_pairs)

        return total_energy, grad


