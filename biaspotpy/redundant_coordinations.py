import itertools

import numpy as np



class RedundantInternalCoordinates:
    def __init__(self):
        return
        
    def B_matrix(self, coord):#cartecian coord (atom num × 3)
        
        idx_list = [i for i in range(len(coord))]
        internal_coord_idx = list(itertools.combinations(idx_list, 2))
        b_mat = np.zeros((len(internal_coord_idx), 3*len(coord)))
        internal_coord_count = 0
        
        for i, j in internal_coord_idx:
            norm = np.linalg.norm(coord[i] - coord[j])
            dr_dxi = (coord[i][0] - coord[j][0]) / norm  
            dr_dyi = (coord[i][1] - coord[j][1]) / norm  
            dr_dzi = (coord[i][2] - coord[j][2]) / norm  
            
            dr_dxj = -(coord[i][0] - coord[j][0]) / norm  
            dr_dyj = -(coord[i][1] - coord[j][1]) / norm  
            dr_dzj = -(coord[i][2] - coord[j][2]) / norm  

            b_mat[internal_coord_count][3*i+0] = dr_dxi
            b_mat[internal_coord_count][3*i+1] = dr_dyi
            b_mat[internal_coord_count][3*i+2] = dr_dzi
            
            b_mat[internal_coord_count][3*j+0] = dr_dxj
            b_mat[internal_coord_count][3*j+1] = dr_dyj
            b_mat[internal_coord_count][3*j+2] = dr_dzj
                    
            internal_coord_count += 1
             
            
        return b_mat
    
    def G_matrix(self, b_mat):
        return np.dot(b_mat, b_mat.T)
    
    def RICgrad2cartgrad(self, RICgrad, b_mat):
        g_mat = self.G_matrix(b_mat)
        cartgrad = np.dot(np.dot(np.linalg.inv(b_mat), g_mat), RICgrad)
        return cartgrad.reshape(int(len(cartgrad)/3), 3) #atom_num × 3
    
    def cartgrad2RICgrad(self, cartgrad, b_mat):
        g_mat = self.G_matrix(b_mat)
        g_mat_inv = np.linalg.inv(g_mat)
        
        RICgrad = np.dot(np.dot(g_mat_inv, b_mat), cartgrad)
        return RICgrad
        
    
    def RIChess2carthess(self, cart_coord, connectivity, RIChess, b_mat, RICgrad):
        return