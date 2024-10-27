import itertools
import torch
import copy

import numpy as np



class RedundantInternalCoordinates:
    def __init__(self):
        
        return
        
    def B_matrix(self, coord):#cartecian coord (atom num × 3)#only constract distance B matrix 
        
        idx_list = [i for i in range(len(coord))]#1-2, 1-3, ..., (N-2)-N, (N-1)-N
        internal_coord_idx = list(itertools.combinations(idx_list, 2))
        b_mat = np.zeros((len(internal_coord_idx), 3*len(coord)))
        internal_coord_count = 0
        
        for i, j in internal_coord_idx:
            norm = np.linalg.norm(coord[i] - coord[j])
            dr_dxi = (coord[i][0] - coord[j][0]) / norm  
            dr_dyi = (coord[i][1] - coord[j][1]) / norm  
            dr_dzi = (coord[i][2] - coord[j][2]) / norm  
            
            dr_dxj = -1*(coord[i][0] - coord[j][0]) / norm  
            dr_dyj = -1*(coord[i][1] - coord[j][1]) / norm  
            dr_dzj = -1*(coord[i][2] - coord[j][2]) / norm  

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
        b_mat_T = b_mat.T
        cartgrad = np.dot(b_mat_T, RICgrad)
        return cartgrad
    

    def cartgrad2RICgrad(self, cartgrad, b_mat):
        g_mat = self.G_matrix(b_mat)
        #g_mat_inv = np.linalg.inv(g_mat)
        RICgrad = np.linalg.solve(g_mat, np.dot(b_mat, cartgrad)) 
        #RICgrad = np.dot(g_mat_inv, np.dot(b_mat, cartgrad)) Calculating inverse matrix using np.linalg.inv gives Low-precision results. 
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


def stack_B_matrix():
    return

def partial_stretch_B_matirx(coord, atom_label_1, atom_label_2):#coord:Bohr
    partial_B = []
    natom = len(coord)
    atom_label_1 -= 1
    atom_label_2 -= 1
    norm = np.linalg.norm(coord[atom_label_1] - coord[atom_label_2])
    dr_dxi = (coord[atom_label_1][0] - coord[atom_label_2][0]) / norm  
    dr_dyi = (coord[atom_label_1][1] - coord[atom_label_2][1]) / norm  
    dr_dzi = (coord[atom_label_1][2] - coord[atom_label_2][2]) / norm  
            
    dr_dxj = -1*(coord[atom_label_1][0] - coord[atom_label_2][0]) / norm  
    dr_dyj = -1*(coord[atom_label_1][1] - coord[atom_label_2][1]) / norm  
    dr_dzj = -1*(coord[atom_label_1][2] - coord[atom_label_2][2]) / norm  
    
    for n in range(natom):
        if n == atom_label_1:
            partial_B.extend([dr_dxi, dr_dyi, dr_dzi])
        elif n == atom_label_2:
            partial_B.extend([dr_dxj, dr_dyj, dr_dzj])
        else:
            partial_B.extend([0.0, 0.0, 0.0])
    
    partial_B = np.array([partial_B], dtype="float64")
    return partial_B # Bohr/Bohr

def partial_bend_B_matrix(coord, atom_label_1, atom_label_2, atom_label_3):#coord:Bohr
    partial_B = []
    i = atom_label_1 - 1
    j = atom_label_2 - 1
    k = atom_label_3 - 1 
    natom = len(coord)

    vec_ij = coord[i] - coord[j]
    vec_kj = coord[k] - coord[j]
    norm_vec_ij = np.linalg.norm(vec_ij)
    norm_vec_kj = np.linalg.norm(vec_kj)

    dot_prod = np.dot(vec_ij, vec_kj) / (norm_vec_ij * norm_vec_kj)
    if dot_prod < -1:
        dot_prod = -1
    elif dot_prod > 1:
        dot_prod = 1

    theta = np.arccos(dot_prod)

    if abs(theta) > np.pi - 1e-6:
        d_theta_dx = [(np.pi - theta) / (2 * norm_vec_ij ** 2) * vec_ij,
                      (1 / norm_vec_ij - 1 / norm_vec_kj) * (np.pi - theta) / (2 * norm_vec_ij) * vec_ij,
                      (np.pi -theta) / (2 * norm_vec_kj ** 2) * vec_kj]
    else:
        d_theta_dx = [1 / np.tan(theta) * vec_ij / norm_vec_ij ** 2 
                      - vec_kj / (norm_vec_ij * norm_vec_kj * np.sin(theta)),
                      (vec_ij + vec_kj) / (norm_vec_ij * norm_vec_kj * np.sin(theta))
                        - 1 / np.tan(theta) * (vec_ij / norm_vec_ij ** 2 + vec_kj / norm_vec_kj ** 2),
                        1 / np.tan(theta) * vec_kj / norm_vec_kj ** 2 - vec_ij / (norm_vec_ij * norm_vec_kj * np.sin(theta))]
    dr_dxi = d_theta_dx[0][0]
    dr_dyi = d_theta_dx[0][1]
    dr_dzi = d_theta_dx[0][2]

    dr_dxj = d_theta_dx[1][0]
    dr_dyj = d_theta_dx[1][1]
    dr_dzj = d_theta_dx[1][2]

    dr_dxk = d_theta_dx[2][0]
    dr_dyk = d_theta_dx[2][1]
    dr_dzk = d_theta_dx[2][2]
    
    for n in range(natom):
        if n == i:
            partial_B.extend([dr_dxi, dr_dyi, dr_dzi])
        elif n == j:
            partial_B.extend([dr_dxj, dr_dyj, dr_dzj])
        elif n == k:
            partial_B.extend([dr_dxk, dr_dyk, dr_dzk])
        else:
            partial_B.extend([0.0, 0.0, 0.0])
    
    partial_B = np.array([partial_B], dtype="float64")
    return partial_B # radian/Bohr


def partial_torsion_B_matrix(coord, atom_label_1, atom_label_2, atom_label_3, atom_label_4):
    partial_B = []
    natom = len(coord)
    i = atom_label_1 - 1
    j = atom_label_2 - 1
    k = atom_label_3 - 1
    l = atom_label_4 - 1

    vec_ij = coord[i] - coord[j]
    vec_lk = coord[l] - coord[k]
    vec_kj = coord[k] - coord[j]

    norm_vec_kj = np.linalg.norm(vec_kj)
    unit_vec_kj = vec_kj / norm_vec_kj

    a_1 = vec_ij - np.dot(vec_ij, unit_vec_kj) * unit_vec_kj
    a_2 = vec_lk - np.dot(vec_lk, unit_vec_kj) * unit_vec_kj

    norm_a_1 = np.linalg.norm(a_1)
    norm_a_2 = np.linalg.norm(a_2)

    sgn = np.sign(np.linalg.det(np.array([vec_lk, vec_ij, vec_kj])))
    sgn = sgn or 1
    dot_prod = np.dot(a_1, a_2) / (norm_a_1 * norm_a_2)
    if dot_prod < -1:
        dot_prod = -1
    elif dot_prod > 1:
        dot_prod = 1
    phi = np.arccos(dot_prod) * sgn
    
    if abs(phi) > np.pi - 1e-6:
        G = np.cross(vec_kj, a_1)
        norm_G = np.linalg.norm(G)
        unit_G = G / norm_G
        A = np.dot(vec_ij, unit_vec_kj) / norm_vec_kj
        B = np.dot(vec_lk, unit_vec_kj) / norm_vec_kj 

        d_phi_dx = [unit_G / norm_a_1,
                    - ((1 - A) / norm_a_1 - B / norm_a_2) * unit_G,
                    - ((1 + B) / norm_a_2 + A / norm_a_1) * unit_G,
                    unit_G / norm_a_2]
    elif abs(phi) < 1e-6:
        G = np.cross(vec_kj, a_1)
        norm_G = np.linalg.norm(G)
        unit_G = G / norm_G
        A = np.dot(vec_ij, unit_vec_kj) / norm_vec_kj
        B = np.dot(vec_lk, unit_vec_kj) / norm_vec_kj 

        d_phi_dx = [unit_G / norm_a_1,
                    - ((1 - A) / norm_a_1 - B / norm_a_2) * unit_G,
                    - ((1 + B) / norm_a_2 + A / norm_a_1) * unit_G,
                    -1 * unit_G / norm_a_2]
    else:
        A = np.dot(vec_ij, unit_vec_kj) / norm_vec_kj
        B = np.dot(vec_lk, unit_vec_kj) / norm_vec_kj 
        d_phi_dx = [1/ np.tan(phi) * a_1 / norm_a_1 ** 2 - a_2 / (norm_a_1 * norm_a_2 * np.sin(phi)),
                    ((1 - A) * a_2 - B * a_1) / (norm_a_1 * norm_a_2 * np.sin(phi)) - 1 / np.tan(phi) * ((1 - A) * a_1 / norm_a_1 ** 2 - B * a_2 / norm_a_2 ** 2),
                    ((1 + B) * a_1 + A * a_2) / (norm_a_1 * norm_a_2 * np.sin(phi)) - 1 / np.tan(phi) * ((1 + B) * a_2 / norm_a_2 ** 2 + A * a_1 / norm_a_1 ** 2),
                    1 / np.tan(phi) * a_2 / norm_a_2 ** 2 - a_1 / (norm_a_1 * norm_a_2 * np.sin(phi))]
    
    dr_dxi = d_phi_dx[0][0]
    dr_dyi = d_phi_dx[0][1]
    dr_dzi = d_phi_dx[0][2]

    dr_dxj = d_phi_dx[1][0]
    dr_dyj = d_phi_dx[1][1]
    dr_dzj = d_phi_dx[1][2]

    dr_dxk = d_phi_dx[2][0]
    dr_dyk = d_phi_dx[2][1]
    dr_dzk = d_phi_dx[2][2]

    dr_dxl = d_phi_dx[3][0]
    dr_dyl = d_phi_dx[3][1]
    dr_dzl = d_phi_dx[3][2]


    for n in range(natom):
        if n == i:
            partial_B.extend([dr_dxi, dr_dyi, dr_dzi])
        elif n == j:
            partial_B.extend([dr_dxj, dr_dyj, dr_dzj])
        elif n == k:
            partial_B.extend([dr_dxk, dr_dyk, dr_dzk])
        elif n == l:
            partial_B.extend([dr_dxl, dr_dyl, dr_dzl])
        else:
            partial_B.extend([0.0, 0.0, 0.0])
    
    partial_B = np.array([partial_B], dtype="float64")
    return partial_B # radian/Bohr


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

        cos_angle = torch.matmul(vector1, vector2) / (torch.linalg.norm(vector1) * torch.linalg.norm(vector2) + 1e-15)
        angle = torch.arccos(cos_angle)
      
        return angle
 
    
    def dihedral_angle(self, coord):
        atom1, atom2, atom3, atom4 = coord[0], coord[1], coord[2], coord[3]
        
        a1 = atom2 - atom1
        a2 = atom3 - atom2
        a3 = atom4 - atom3

        v1 = torch.linalg.cross(a1, a2)
        v1 = v1 / torch.linalg.norm(v1, ord=2)
        v2 = torch.linalg.cross(a2, a3)
        v2 = v2 / torch.linalg.norm(v2, ord=2)
        cos_angle = torch.sum(v1*v2) / torch.sum((v1**2) * torch.sum(v2**2) + 1e-15) ** 0.5
        dihedral_angle = torch.arccos(cos_angle)

        dihedral_angle = torch.abs(dihedral_angle)    
        
        return dihedral_angle


