import numpy as np
import copy

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
        
    
    def opt_path(self, geometry_list):
        print("IDPP Optimization")
        for i in range(self.iteration):
            obj_func_list = []
            for j in range(len(geometry_list)):
                
                if j == 0 or j == len(geometry_list) - 1:
                    continue
                
                obj_func, first_deriv = self.get_func_and_deriv(geometry_list, len(geometry_list), j)
                geometry_list[j] -= self.lr * first_deriv
            
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


    