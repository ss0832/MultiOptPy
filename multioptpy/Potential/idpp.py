import numpy as np
from scipy.optimize import minimize

from multioptpy.Parameters.parameter import (
                covalent_radii_lib,
                UFF_VDW_distance_lib,
                number_element,
                UnitValueLib,)

class IDPP:
    def __init__(self):
        #ref.: arXiv:1406.1512v1
        self.iteration = 2000
        self.lr = 0.01
        self.threshold = 1e-4
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
        
    
    def opt_path(self, geometry_list, element_list, memory_size=30):
        """
        Optimize the path using L-BFGS algorithm with the original step size limiting.
        
        Parameters:
        -----------
        geometry_list : list of numpy arrays
            List of geometries to optimize
        element_list : list
            List of elements (preserved for compatibility)
        memory_size : int
            Number of correction pairs to store for L-BFGS
        
        Returns:
        --------
        list of numpy arrays
            Optimized geometries
        """
        print("IDPP Optimization with L-BFGS")
        
        # Initialize L-BFGS memory for each image
        s_list = [[] for _ in range(len(geometry_list))]
        y_list = [[] for _ in range(len(geometry_list))]
        rho_list = [[] for _ in range(len(geometry_list))]
        
        def lbfgs_direction(gradient, j):
            """Compute the L-BFGS search direction using two-loop recursion"""
            if len(s_list[j]) == 0:
                return -gradient
            
            q = gradient.copy()
            alpha_list = []
            
            # First loop: compute alpha values and update q
            for i in range(len(s_list[j])-1, -1, -1):
                alpha = rho_list[j][i] * np.sum(s_list[j][i] * q)
                alpha_list.insert(0, alpha)
                q = q - alpha * y_list[j][i]
            
            # Scale with gamma
            i = len(s_list[j]) - 1
            denominator = np.sum(y_list[j][i] * y_list[j][i]) # Avoid division by zero
            if np.abs(denominator) > 1e-10:
                gamma = np.sum(s_list[j][i] * y_list[j][i]) / denominator
            else:
                gamma = 0
            r = gamma * q
            
            # Second loop: compute search direction
            for i in range(len(s_list[j])):
                beta = rho_list[j][i] * np.sum(y_list[j][i] * r)
                r = r + s_list[j][i] * (alpha_list[i] - beta)
            
            return -r
        
        for i in range(self.iteration):
            obj_func_list = []
            
            for j in range(len(geometry_list)):
                if j == 0 or j == len(geometry_list) - 1:
                    continue
                
                # Save current position for computing displacement later
                current_pos = geometry_list[j].copy()
                
                # Get objective function and gradient
                obj_func, gradient = self.get_func_and_deriv(geometry_list, len(geometry_list), j)
                obj_func_list.append(obj_func)
                gradient *= -1
                # Compute search direction using L-BFGS
                search_dir = lbfgs_direction(gradient, j)
                
                # Apply the original step size limiting algorithm
                step_norm = min(self.lr, np.linalg.norm(search_dir))
                if np.linalg.norm(search_dir) > 1e-10:  # Avoid division by zero
                    norm_step = search_dir / np.linalg.norm(search_dir)
                    geometry_list[j] -= step_norm * norm_step
                
                # Update L-BFGS memory after taking the step
                new_obj_func, new_gradient = self.get_func_and_deriv(geometry_list, len(geometry_list), j)
                
                # Compute s and y vectors
                s = geometry_list[j] - current_pos
                y = new_gradient - gradient
                
                # Only update memory if curvature condition is satisfied
                sy_product = np.sum(s * y)
                if sy_product > 1e-10:
                    # Manage memory size
                    if len(s_list[j]) >= memory_size:
                        s_list[j].pop(0)
                        y_list[j].pop(0)
                        rho_list[j].pop(0)
                    
                    s_list[j].append(s)
                    y_list[j].append(y)
                    rho_list[j].append(1.0 / sy_product)
            
            if i % 200 == 0:
                print("ITR: ", i)
                print("Objective function (Max): ", max(obj_func_list))
            
            if max(obj_func_list) < self.threshold:
                print("ITR: ", i)
                print("IDPP Converged!!!")
                break
        
        print("IDPP Optimization Done.")
        return geometry_list



class CFB_ENM:
    """
    Implements a standalone Correlated Flat-Bottom Elastic Network Model (CFB-ENM)
    for optimizing reaction paths. The potential is based on the logic from dmf.py.

    This class identifies quartets of atoms involved in bond-making and -breaking
    events between a reactant and product structure. It then applies a specialized
    potential function to these quartets to guide the path optimization.

    The path is optimized using an L-BFGS algorithm implemented from scratch.
    ref. : S.-i. Koda and S. Saito, Flat-bottom Elastic Network Model for Generating Improved Plausible Reaction Paths, JCTC, 20, 7176−7187 (2024). doi: 10.1021/acs.jctc.4c00792
           S.-i. Koda and S. Saito, Correlated Flat-bottom Elastic Network Model for Improved Bond Rearrangement in Reaction Paths, JCTC, 21, 3513−3522 (2025). doi: 10.1021/acs.jctc.4c01549
    
    """

    def __init__(self, iteration=2000, lr=0.01, threshold=1e-4, bond_scale=1.25,
                 corr0_scale=1.10, corr1_scale=1.50, corr2_scale=1.60,
                 eps=0.05, pivotal=True, single=True, remove_fourmembered=True):
        """
        Initializes the CFB_ENM optimizer.

        Parameters:
        -----------
        iteration : int
            Maximum number of optimization iterations.
        lr : float
            Learning rate or maximum step size for the L-BFGS update.
        threshold : float
            Convergence threshold for the objective function.
        bond_scale : float
            Factor to determine bonding from covalent radii.
        corr0_scale, corr1_scale, corr2_scale : float
            Scaling factors for correlation distance thresholds.
        eps : float
            Smoothing parameter for the potential function.
        pivotal, single, remove_fourmembered : bool
            Flags to control quartet identification logic.
        """
        self.iteration = int(iteration) # FIX: Ensure iteration is an integer
        self.lr = lr
        self.threshold = threshold
        
        # Parameters for CFB_ENM potential from dmf.py
        self.bond_scale = bond_scale
        self.corr0_scale = corr0_scale
        self.corr1_scale = corr1_scale
        self.corr2_scale = corr2_scale
        self.eps = eps
        self.pivotal = pivotal
        self.single = single
        self.remove_fourmembered = remove_fourmembered
        
        # These will be populated by _initialize_potential
        self.quartets = []
        self.d_corr0 = None
        self.d_corr1 = None
        self.d_corr2 = None
        self.bohr2ang = UnitValueLib().bohr2angstroms
        return

    def _get_connectivity_matrix(self, pos, element_list):
        """
        Determines the adjacency matrix for a given geometry.
        """
        radii = np.array([covalent_radii_lib(el) * self.bohr2ang for el in element_list])
        r_cov = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        dist_matrix = self.calc_dist_matrix(pos)
        
        J = (dist_matrix / r_cov) < self.bond_scale
        np.fill_diagonal(J, False)
        return J, dist_matrix

    def _get_quartets(self, J_only_r, J_only_p, J_both,
                      pivotal=True, single=True, remove_fourmembered=True):
        """
        Identifies quartets of atoms involved in correlated motion.
        This is a direct adaptation of the logic from dmf.py.
        """
        J2 = np.dot(J_both, J_both)
        quartets = []

        if pivotal:
            if single:
                pivots = np.where((np.sum(J_only_r, axis=1) == 1)
                                & (np.sum(J_only_p, axis=1) == 1))[0]
            else:
                pivots = np.where(np.any(J_only_r, axis=1)
                                & np.any(J_only_p, axis=1))[0]
            for i in pivots:
                only_r = np.where(J_only_r[i])[0]
                only_p = np.where(J_only_p[i])[0]
                for j in only_r:
                    for k in only_p:
                        if not (remove_fourmembered and J2[j, k]):
                            quartets.append([i, j, i, k])
        else:
            # Non-pivotal logic (adapted from dmf.py)
            pairs_only_r = list(zip(*np.where(np.triu(J_only_r, k=1))))
            pairs_only_p = list(zip(*np.where(np.triu(J_only_p, k=1))))

            for pr in pairs_only_r:
                for pp in pairs_only_p:
                    q = list(pr) + list(pp)
                    is_fourmembered = False
                    if remove_fourmembered:
                        unique_atoms = set(q)
                        if len(unique_atoms) == 4:
                            is_fourmembered = (J_both[q[0], q[2]] and J_both[q[1], q[3]]) or \
                                              (J_both[q[0], q[3]] and J_both[q[1], q[2]])
                        elif len(unique_atoms) == 3:
                             # Find the two atoms that appear once
                            counts = {atom: q.count(atom) for atom in unique_atoms}
                            uniq_idxs = [atom for atom, count in counts.items() if count == 1]
                            if len(uniq_idxs) == 2:
                                is_fourmembered = J2[uniq_idxs[0], uniq_idxs[1]]

                    if not is_fourmembered:
                        quartets.append(q)
        return quartets

    def _initialize_potential(self, reactant_pos, product_pos, element_list):
        """
        Initializes the parameters for the CFB-ENM potential function based on
        reactant and product structures.
        """
        natoms = len(element_list)
        images = [reactant_pos, product_pos]
        
        Js = []
        d_bonds_list = []
        for pos in images:
            J, d = self._get_connectivity_matrix(pos, element_list)
            Js.append(J)
            d_bonds_list.append(np.where(J, d, 0.0))
        
        d_bond = np.max(np.array(d_bonds_list), axis=0)

        J_only_r = Js[0] & (~Js[1])
        J_only_p = Js[1] & (~Js[0])
        J_both = Js[0] & Js[1]

        self.quartets = self._get_quartets(J_only_r, J_only_p, J_both,
                                           self.pivotal, self.single, self.remove_fourmembered)
        
        self.d_corr0 = self.corr0_scale * d_bond
        self.d_corr1 = self.corr1_scale * d_bond
        self.d_corr2 = self.corr2_scale * d_bond
        
        # Ensure diagonal is zero
        I = np.identity(natoms, dtype='bool')
        self.d_corr0[I] = 0.0
        self.d_corr1[I] = 0.0
        self.d_corr2[I] = 0.0

        print(f"CFB-ENM: Initialized potential with {len(self.quartets)} quartets.")

    def calc_dist_matrix(self, pos):
        """
        Calculates the pairwise distance matrix for a given geometry.
        """
        pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(pos_diff**2, axis=-1))
        return dist_matrix

    def get_func_and_deriv(self, pos):
        """
        Calculates the CFB-ENM objective function and its analytical gradient
        for a single image, based on the quartet potential from dmf.py.
        """
        natoms = pos.shape[0]
        r = pos
        dr = r[:, np.newaxis, :] - r
        d = np.sqrt(np.sum(dr**2, axis=-1))

        energy = 0.0
        forces = np.zeros_like(pos)

        d_d0 = d - self.d_corr0
        d1_d0 = self.d_corr1 - self.d_corr0
        d2_d0 = self.d_corr2 - self.d_corr0
        
        for t in self.quartets:
            # t = [atom1, atom2, atom3, atom4]
            # Bond pair 1: (t[0], t[1]), Bond pair 2: (t[2], t[3])
            
            # Check if atoms are in the repulsive region
            if (d_d0[t[0], t[1]] > 0.0 and d_d0[t[2], t[3]] > 0.0):
                
                pp = (d_d0[t[0], t[1]] * d_d0[t[2], t[3]]
                      - d1_d0[t[0], t[1]] * d1_d0[t[2], t[3]])
                
                # Check if potential is active
                if pp > 0.0:
                    dnm = (d2_d0[t[0], t[1]] * d2_d0[t[2], t[3]]
                           - d1_d0[t[0], t[1]] * d1_d0[t[2], t[3]])
                    
                    # Avoid division by zero
                    if abs(dnm) < 1e-10: continue

                    pp_norm = pp / dnm
                    sqrt_pp2_eps2 = np.sqrt(pp_norm**2 + self.eps**2)
                    
                    energy += sqrt_pp2_eps2 - self.eps
                    
                    # Common factor for gradient
                    alpha = pp_norm / sqrt_pp2_eps2
                    
                    # Gradient vectors
                    v1 = d_d0[t[2], t[3]] / d[t[0], t[1]] * (r[t[0]] - r[t[1]])
                    v2 = d_d0[t[0], t[1]] / d[t[2], t[3]] * (r[t[2]] - r[t[3]])
                    
                    v1_norm = v1 / dnm
                    v2_norm = v2 / dnm

                    # Accumulate forces
                    forces[t[0]] -= alpha * v1_norm
                    forces[t[1]] += alpha * v1_norm
                    forces[t[2]] -= alpha * v2_norm
                    forces[t[3]] += alpha * v2_norm
        
        # The optimizer expects the gradient of the objective function.
        # Force is the negative of the gradient.
        gradient = -forces
        
        return energy, gradient

    def opt_path(self, geometry_list, element_list, memory_size=30):
        """
        Optimize the path using L-BFGS algorithm.

        Parameters:
        -----------
        geometry_list : list of np.ndarray
            List of geometries (images) to optimize. The first and last are
            fixed as reactant and product.
        element_list : list of str
            List of element symbols for the atoms.
        memory_size : int
            Number of correction pairs to store for L-BFGS.

        Returns:
        --------
        list of np.ndarray
            The list of optimized geometries.
        """
        print("CFB-ENM Optimization with L-BFGS")

        # Initialize the potential based on the start and end points of the path
        self._initialize_potential(geometry_list[0], geometry_list[-1], element_list)
        
        # Initialize L-BFGS memory for each image
        s_list = [[] for _ in range(len(geometry_list))]
        y_list = [[] for _ in range(len(geometry_list))]
        rho_list = [[] for _ in range(len(geometry_list))]

        def lbfgs_direction(gradient, j):
            """Compute the L-BFGS search direction using two-loop recursion"""
            if len(s_list[j]) == 0:
                return -gradient
            
            q = gradient.copy()
            alpha_list = []
            
            for i in range(len(s_list[j])-1, -1, -1):
                alpha = rho_list[j][i] * np.sum(s_list[j][i] * q)
                alpha_list.insert(0, alpha)
                q -= alpha * y_list[j][i]
            
            i = len(s_list[j]) - 1
            denominator = np.sum(y_list[j][i] * y_list[j][i])
            if np.abs(denominator) > 1e-10:
                gamma = np.sum(s_list[j][i] * y_list[j][i]) / denominator
            else:
                gamma = 1.0 # Fallback to steepest descent scaling
            r = gamma * q
            
            for i in range(len(s_list[j])):
                beta = rho_list[j][i] * np.sum(y_list[j][i] * r)
                r += s_list[j][i] * (alpha_list[i] - beta)
            
            return -r
        
        # FIX: Ensure self.iteration is an integer before using in range()
        for i in range(int(self.iteration)):
            obj_func_list = []
            
            # Iterate over intermediate images (endpoints are fixed)
            for j in range(1, len(geometry_list) - 1):
                current_pos = geometry_list[j].copy()
                
                obj_func, gradient = self.get_func_and_deriv(current_pos)
                obj_func_list.append(obj_func)
                
                search_dir = lbfgs_direction(gradient, j)
                
                # Simple step size control
                step_norm = self.lr
                if np.linalg.norm(search_dir) > 1e-10:
                    norm_step = search_dir / np.linalg.norm(search_dir)
                    geometry_list[j] += step_norm * norm_step
                
                # Update L-BFGS memory
                new_pos = geometry_list[j]
                _, new_gradient = self.get_func_and_deriv(new_pos)
                
                s = new_pos - current_pos
                y = new_gradient - gradient
                
                sy_product = np.sum(s * y)
                if sy_product > 1e-10: # Curvature condition
                    if len(s_list[j]) >= memory_size:
                        s_list[j].pop(0)
                        y_list[j].pop(0)
                        rho_list[j].pop(0)
                    
                    s_list[j].append(s)
                    y_list[j].append(y)
                    rho_list[j].append(1.0 / sy_product)
            
            if i % 200 == 0:
                max_obj = max(obj_func_list) if obj_func_list else 0.0
                print(f"ITR: {i}, Objective function (Max): {max_obj:.6e}")
            
            if not obj_func_list or max(obj_func_list) < self.threshold:
                print(f"ITR: {i}")
                print("CFB-ENM Converged!!!")
                break
        
        print("CFB-ENM Optimization Done.")
        return geometry_list