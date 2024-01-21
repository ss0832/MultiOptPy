import itertools
import torch
import copy

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
        #cart_coord: Bohr (natom × 3)
        #b_mat: Bohr
        #RIChess: Hartree/Bohr**2
        #RICgrad: Hartree/Bohr 
        #connectivity: bond, angle. dihedralangle

        natom = len(cart_coord)
        K_mat = np.zeros((natom*3, natom*3), dtype="float64")
        count = 0
        bond_connectivity_table = connectivity[0]
        angle_connectivity_table = connectivity[1]
        dihedral_angle_connectivity_table = connectivity[2]
        for idx_list in bond_connectivity_table:
            atom1 = cart_coord[idx_list[0]]
            atom2 = cart_coord[idx_list[1]]
            coord = torch.tensor(np.array([atom1, atom2]) , dtype=torch.float64, requires_grad=True)
            tensor_2nd_derivative_dist = torch.func.hessian(TorchDerivatives().distance)(coord)
            tensor_2nd_derivative_dist = tensor_2nd_derivative_dist.reshape(1, 36)
            second_derivative_dist = copy.copy(tensor_2nd_derivative_dist.detach().numpy())
            second_derivative_dist = np.squeeze(second_derivative_dist)
            
            K_mat_idx_list_A = []
            K_mat_idx_list_B = []
            new_idx_list = []
            for idx in idx_list:
                new_idx_list.extend([idx*3, idx*3+1, idx*3+2])
            length = len(new_idx_list)
            for j in new_idx_list:
                K_mat_idx_list_A.extend([j for i in range(length)])
                K_mat_idx_list_B.extend(new_idx_list)
           
            K_mat[K_mat_idx_list_A, K_mat_idx_list_B] += second_derivative_dist * RICgrad[count]
       
            
            count += 1
            
        for idx_list in angle_connectivity_table:
            atom1 = cart_coord[idx_list[0]]
            atom2 = cart_coord[idx_list[1]]
            atom3 = cart_coord[idx_list[2]]
            coord = torch.tensor(np.array([atom1, atom2, atom3]) , dtype=torch.float64, requires_grad=True)
            tensor_2nd_derivative_angle = torch.func.hessian(TorchDerivatives().angle)(coord)
            tensor_2nd_derivative_angle = tensor_2nd_derivative_angle.reshape(1, 81)
            second_derivative_angle = copy.copy(tensor_2nd_derivative_angle.detach().numpy())
            second_derivative_angle = np.squeeze(second_derivative_angle)
            K_mat_idx_list_A = []
            K_mat_idx_list_B = []
            new_idx_list = []
            for idx in idx_list:
                new_idx_list.extend([idx*3, idx*3+1, idx*3+2])
            length = len(new_idx_list)
            for j in new_idx_list:
                K_mat_idx_list_A.extend([j for i in range(length)])
                K_mat_idx_list_B.extend(new_idx_list)
            K_mat[K_mat_idx_list_A, K_mat_idx_list_B] += second_derivative_angle * RICgrad[count]
            
            count += 1
        
        for idx_list in dihedral_angle_connectivity_table:
            atom1 = cart_coord[idx_list[0]]
            atom2 = cart_coord[idx_list[1]]
            atom3 = cart_coord[idx_list[2]]
            atom4 = cart_coord[idx_list[3]]
            coord = torch.tensor(np.array([atom1, atom2, atom3, atom4]) , dtype=torch.float64, requires_grad=True)
            
            tensor_2nd_derivative_dangle = torch.func.hessian(TorchDerivatives().dihedral_angle)(coord)
            tensor_2nd_derivative_dangle = tensor_2nd_derivative_dangle.reshape(1, 144)
            second_derivative_dangle = copy.copy(tensor_2nd_derivative_dangle.detach().numpy())
            second_derivative_dangle = np.squeeze(second_derivative_dangle)
            K_mat_idx_list_A = []
            K_mat_idx_list_B = []
            new_idx_list = []
            for idx in idx_list:
                new_idx_list.extend([idx*3, idx*3+1, idx*3+2])
            length = len(new_idx_list)
            for j in new_idx_list:
                K_mat_idx_list_A.extend([j for i in range(length)])
                K_mat_idx_list_B.extend(new_idx_list)
            K_mat[K_mat_idx_list_A, K_mat_idx_list_B] += second_derivative_dangle * RICgrad[count]           
            count += 1
       
        cart_hessian = np.dot(np.dot(b_mat.T, RIChess), b_mat) + K_mat
        return cart_hessian


class TorchDerivatives:
    def __init__(self):
        return
    
    def distance(self, coord):
        dist = torch.linalg.norm(coord[0] - coord[1])
        return dist
    
    def angle(self, coord):
        atom1, atom2, atom3 = coord[0], coord[1], coord[2]
        vector1 = atom1 - atom2
        vector2 = atom3 - atom2

        cos_angle = torch.matmul(vector1, vector2) / (torch.linalg.norm(vector1) * torch.linalg.norm(vector2))
        angle = torch.arccos(cos_angle)
      
        return angle
 
    
    def dihedral_angle(self, coord):
        atom1, atom2, atom3, atom4 = coord[0], coord[1], coord[2], coord[3]
        
        a1 = atom2 - atom1
        a2 = atom3 - atom2
        a3 = atom4 - atom3

        v1 = torch.cross(a1, a2)
        v1 = v1 / torch.linalg.norm(v1, ord=2)
        v2 = torch.cross(a2, a3)
        v2 = v2 / torch.linalg.norm(v2, ord=2)
        
        dihedral_angle = torch.arccos(torch.sum(v1*v2) / torch.sum((v1**2) * torch.sum(v2**2)) ** 0.5)
        
        dihedral_angle = torch.abs(dihedral_angle)    
 
        return dihedral_angle


