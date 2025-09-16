import itertools
import torch
import copy

import numpy as np

from multioptpy.Utils.calc_tools import Calculationtools, output_partial_hess
from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass

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

def torch_calc_distance(coord, atom_label_1, atom_label_2):#coord:Bohr
    vec_1 = coord[atom_label_1 - 1]
    vec_2 = coord[atom_label_2 - 1]
    norm = torch.linalg.norm(vec_1 - vec_2)
    return norm

def torch_calc_fragm_distance(coord, atom_fragm_1, atom_fragm_2):#fragm_n:tensor
    vec_1 = torch.mean(input=coord[atom_fragm_1 - 1], dim=0)
    vec_2 = torch.mean(input=coord[atom_fragm_2 - 1], dim=0)
    norm = torch.linalg.norm(vec_1 - vec_2)
    return norm

def torch_calc_angle(coord, atom_label_1, atom_label_2, atom_label_3):#coord:Bohr
    vec_1 = coord[atom_label_1 - 1]
    vec_2 = coord[atom_label_2 - 1]
    vec_3 = coord[atom_label_3 - 1]
    
    vec_12 = vec_1 - vec_2
    vec_32 = vec_3 - vec_2
    norm_vec_12 = torch.linalg.norm(vec_12) + 1e-15
    norm_vec_32 = torch.linalg.norm(vec_32) + 1e-15
    
    dot_prod = torch.matmul(vec_12, vec_32) / (norm_vec_12 * norm_vec_32)
    theta = torch.acos(dot_prod)
    return theta

def torch_calc_dihedral_angle(coord, atom_label_1, atom_label_2, atom_label_3, atom_label_4):#coord:Bohr
    vec_1 = coord[atom_label_1 - 1]
    vec_2 = coord[atom_label_2 - 1]
    vec_3 = coord[atom_label_3 - 1]
    vec_4 = coord[atom_label_4 - 1]
    
    vec_12 = vec_1 - vec_2
    vec_23 = vec_2 - vec_3
    vec_34 = vec_3 - vec_4
    
    v1 = torch.linalg.cross(vec_12, vec_23)
    norm_v1 = torch.linalg.norm(v1) + 1e-15
    v2 = torch.linalg.cross(vec_23, vec_34)
    norm_v2 = torch.linalg.norm(v2) + 1e-15
    cos_theta = torch.sum(v1 * v2) / (norm_v1 * norm_v2)
    angle = torch.acos(cos_theta)
    return angle


def torch_B_matrix(coord, atom_labels, calc_int_coord_func):#coord:Bohr
    B_mat = torch.func.jacrev(calc_int_coord_func)(coord, *atom_labels)
    return B_mat

def torch_B_matrix_derivative(coord, atom_labels, calc_int_coord_func):
    B_mat_derivative = torch.func.jacrev(torch_B_matrix)(coord, atom_labels, calc_int_coord_func)
    B_mat_derivative = B_mat_derivative.reshape(3*len(coord), 3*len(coord), 1)
    return B_mat_derivative


def calc_G_mat(Bmat):
    Gmat = np.dot(Bmat.T, Bmat)
    return Gmat

def calc_inv_G_mat(Gmat, threshold=1e-6):
    #Gmat += 1e-12*np.eye(len(Gmat)) 
    #print(np.linalg.cond(Gmat))
    U, s, VT = np.linalg.svd(Gmat)
    s_inv = []
    for value in s:
        if value > threshold:
            s_inv.append(1/value)
        else:
            s_inv.append(value)
    s_inv = np.array(s_inv, dtype="float64")
    s_inv = np.diag(s_inv)
    Gmat_inv = np.dot(VT.T, np.dot(s_inv, U.T))
    return Gmat_inv

def calc_inv_B_mat(Bmat):
    Gmat = calc_G_mat(Bmat)
    Gmat_inv = calc_inv_G_mat(Gmat)
    Bmat_inv = np.dot(Gmat_inv, Bmat.T)
    return Bmat_inv.T


def calc_dot_B_deriv_int_grad(B_mat_1st_derivative, int_grad):#B_mat_1st_derivative: (3N, 3N, M), int_grad: (M, 1)
    natom3 = len(B_mat_1st_derivative)
    tmp_list = []
    for i in range(natom3):
        tmp_list.append(np.dot(B_mat_1st_derivative[i], int_grad))
    dot_B_deriv_int_grad = np.array(tmp_list, dtype="float64").reshape(natom3, natom3)
    return dot_B_deriv_int_grad


def calc_int_hess_from_pBmat_for_non_stationary_point(cart_hess, pBmat, pBmat_1st_derivative, int_grad):
    Bmat_inv = calc_inv_B_mat(pBmat)
    dot_B_deriv_int_grad = calc_dot_B_deriv_int_grad(pBmat_1st_derivative, int_grad)
    
    int_hess = np.dot(Bmat_inv, np.dot(cart_hess - dot_B_deriv_int_grad, Bmat_inv.T))
    return int_hess

def calc_int_cart_coupling_hess_from_pBmat_for_non_stationary_point(cart_hess, pBmat, pBmat_1st_derivative, int_grad):
    Bmat_inv = calc_inv_B_mat(pBmat)
    dot_B_deriv_int_grad = calc_dot_B_deriv_int_grad(pBmat_1st_derivative, int_grad)
    couple_hess = np.dot(Bmat_inv, cart_hess - dot_B_deriv_int_grad)
    return couple_hess


def calc_cart_hess_from_pBmat_for_non_stationary_point(int_hess, pBmat, pBmat_1st_derivative, int_grad):
    dot_B_deriv_int_grad = calc_dot_B_deriv_int_grad(pBmat_1st_derivative, int_grad)
    cart_hess = np.dot(pBmat.T, np.dot(int_hess, pBmat)) + dot_B_deriv_int_grad
    return cart_hess


def calc_int_grad_from_pBmat(cart_grad, pBmat):
    Bmat_inv = calc_inv_B_mat(pBmat)
    int_grad = np.dot(Bmat_inv, cart_grad)
    return int_grad

def calc_cart_grad_from_pBmat(int_grad, pBmat):
    cart_grad = np.dot(pBmat.T, int_grad)
    return cart_grad


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


def calc_local_fc_from_pBmat(cart_hess, pBmat):#This method is not good since hessian is ill-condition.
    #hessian projected out transion and rotation is needed.
    inv_cart_hess = np.linalg.inv(cart_hess)
    inv_local_fc = np.dot(pBmat, np.dot(inv_cart_hess, pBmat.T))
    local_fc_matrix = np.linalg.inv(inv_local_fc)
    non_diagonal_ufc = np.triu(local_fc_matrix) - np.diag(np.diag(local_fc_matrix))
    non_diagonal_lfc = np.tril(local_fc_matrix) - np.diag(np.diag(local_fc_matrix))
    local_fc = np.diag(local_fc_matrix)
    return local_fc, non_diagonal_ufc, non_diagonal_lfc#a.u.

def calc_local_fc_from_pBmat_2(cart_hess, pBmat):#This method is only available to stationary point
    B_inv = np.dot(np.linalg.inv(np.dot(pBmat, pBmat.T)), pBmat)
    local_fc = np.dot(B_inv, np.dot(cart_hess, B_inv.T))
    diag_local_fc = np.diag(local_fc)
    return diag_local_fc

def calc_local_fc_from_pBmat_3(cart_hess, pBmat):#This method is only available to stationary point
    #ref.:https://geometric.readthedocs.io/en/latest/how-it-works.html
    B_inv = calc_inv_B_mat(pBmat)
    local_fc = np.dot(B_inv, np.dot(cart_hess, B_inv.T))

    return local_fc

def cartesian_to_z_matrix(cart_coords):
    def calculate_torsion_angle(A, B, C, D):
        AB = B - A
        BC = C - B
        CD = D - C

        normal_ABC = np.cross(AB, BC)
        normal_BCD = np.cross(BC, CD)
        
        normal_ABC = normal_ABC / np.linalg.norm(normal_ABC)
        normal_BCD = normal_BCD / np.linalg.norm(normal_BCD)
        
        cosine_angle = np.dot(normal_ABC, normal_BCD)
        angle = np.arccos(cosine_angle)

        sign = np.dot(np.cross(normal_ABC, normal_BCD), BC)  
        if sign < 0:
            angle = -angle
        
        return np.degrees(angle)
    n_atoms = len(cart_coords)
    z_matrix = []
    
    distance_1_2 = np.linalg.norm(np.array(cart_coords[1]) - np.array(cart_coords[0])) + 1e-15
    z_matrix.append([distance_1_2])

    distance_1_3 = np.linalg.norm(np.array(cart_coords[2]) - np.array(cart_coords[0])) + 1e-15
    distance_2_3 = np.linalg.norm(np.array(cart_coords[2]) - np.array(cart_coords[1])) + 1e-15
    angle_1_2_3 = np.degrees(np.arccos(np.dot(np.array(cart_coords[1]) - np.array(cart_coords[0]),
                                              np.array(cart_coords[2]) - np.array(cart_coords[0])) /
                                        (distance_1_2 * distance_1_3)))
    z_matrix.append([distance_2_3])
    z_matrix.append([angle_1_2_3])
    for i in range(3, n_atoms):
    
        distance_i_minus_1 = np.linalg.norm(np.array(cart_coords[i]) - np.array(cart_coords[i-1])) + 1e-15
        distance_i_minus_2 = np.linalg.norm(np.array(cart_coords[i]) - np.array(cart_coords[i-2])) + 1e-15
        clipped_value = np.clip(np.dot(np.array(cart_coords[i-1]) - np.array(cart_coords[i-2]),
                                                                   np.array(cart_coords[i]) - np.array(cart_coords[i-1])) /
                                                           (distance_i_minus_2 * distance_i_minus_1), -1, 1)
        angle_i_minus_2_i_minus_1_i = np.degrees(np.arccos(clipped_value))
        
        
        torsion_angle = calculate_torsion_angle(cart_coords[i - 3], cart_coords[i - 2], cart_coords[i - 1], cart_coords[i])
        z_matrix.append([distance_i_minus_1])  
        z_matrix.append([angle_i_minus_2_i_minus_1_i])
        z_matrix.append([torsion_angle])
    z_matrix = np.array(z_matrix)
    return z_matrix


if __name__ == "__main__":#test
    
    test_coord = np.array( [[0.075000142905,          0.075000142905,         -0.000000000000],
                            [ 1.027799531262,         -0.180310974599,          0.000000000000],
                            [-0.180310974599,          1.027799531262,          0.000000000000],
                            [-0.622488699568,         -0.622488699568,          0.000000000000]], dtype="float64") / UnitValueLib().bohr2angstroms
    test_hess = np.array([[ 0.955797621, 0.000024060, -0.000000000, -0.518670978, 0.115520742, 0.000000000, -0.118512045, 0.115518826, 0.000000000, -0.318614598, -0.231063629, -0.000000000],
                          [ 0.000024060, 0.955797621,  0.000000000, 0.115518826, -0.118512045, -0.000000000, 0.115520742,-0.518670978 ,-0.000000000,-0.231063629, -0.318614598, 0.000000000],
                          [-0.000000000, 0.000000000,  -0.016934167, 0.000000000, -0.000000000 , 0.005642447, 0.000000000,-0.000000000, 0.005642447, 0.000000000, 0.000000000, 0.005649274 ],
                          [-0.518670978,  0.115518826,0.000000000,0.534121419,-0.123663333,-0.000000000,-0.000671477,-0.007212539, -0.000000000,-0.014778964,0.015357046,-0.000000000],
                          [ 0.115520742, -0.118512045,-0.000000000 ,-0.123663333,0.105749196, 0.000000000,0.039787518, -0.000671477,0.000000000,-0.031644928, 0.013434326,-0.000000000],
                          [ 0.000000000,  -0.000000000,0.005642447,-0.000000000,0.000000000,-0.001877261,-0.000000000,0.000000000,-0.001883273,-0.000000000,0.000000000,-0.001881913],
                          [-0.118512045, 0.115520742,0.000000000,-0.000671477,0.039787518, -0.000000000, 0.105749196, -0.123663333,0.000000000,0.013434326,-0.031644928,0.000000000],
                          [ 0.115518826, -0.518670978 ,-0.000000000, -0.007212539,-0.000671477,0.000000000, -0.123663333,0.534121419,0.000000000,0.015357046,-0.014778964,0.000000000],
                          [ 0.000000000, -0.000000000,0.005642447, -0.000000000,0.000000000,-0.001883273,0.000000000,0.000000000,-0.001877261,0.000000000,0.000000000,-0.001881913],
                          [-0.318614598, -0.231063629, 0.000000000,-0.014778964,-0.031644928, -0.000000000,0.013434326,0.015357046,0.000000000,0.319959236,0.247351511,-0.000000000],
                          [-0.231063629, -0.318614598, 0.000000000,0.015357046,0.013434326,0.000000000,-0.031644928,-0.014778964,0.000000000,0.247351511, 0.319959236, -0.000000000],
                          [-0.000000000, 0.000000000,  0.005649274,-0.000000000, -0.000000000,-0.001881913,0.000000000, 0.000000000,-0.001881913,-0.000000000, -0.000000000,-0.001885447]], dtype="float64")
    test_element_list = ["N", "H", "H", "H"]
    partial_hess, partial_geom, partial_element_list = output_partial_hess(test_hess, [1,2], test_element_list, test_coord)
    p_partial_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(partial_hess, partial_element_list, partial_geom)
    partial_eigenvalue, partial_eigenvector = np.linalg.eigh(p_partial_hess)
    print(partial_eigenvalue)
    
    mw_partial_hess = Calculationtools().project_out_hess_tr_and_rot(partial_hess, partial_element_list, partial_geom)
    partial_eigenvalue, partial_eigenvector = np.linalg.eigh(mw_partial_hess)
    print(partial_eigenvalue)


    test_mw_hess = copy.copy(Calculationtools().project_out_hess_tr_and_rot(test_hess, test_element_list, test_coord))
    tmp_test_mw_hess = copy.copy(Calculationtools().project_out_hess_tr_and_rot(test_hess, test_element_list, test_coord))
    print("normal coordinate")
    bond_pBmat = partial_stretch_B_matirx(test_coord, 1, 2)
    bend_pBmat = partial_bend_B_matrix(test_coord, 2, 1, 3)
    torsion_pBmat = partial_torsion_B_matrix(test_coord, 2, 1, 3, 4)
    print(bond_pBmat, bend_pBmat, torsion_pBmat)
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(test_hess, bend_pBmat)
    lfc2 = calc_local_fc_from_pBmat_2(test_hess, bend_pBmat)
    print(lfc, undfc, lndfc, lfc2)
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(test_hess, bond_pBmat)
    lfc2 = calc_local_fc_from_pBmat_2(test_hess, bond_pBmat)
    print(lfc, undfc, lndfc, lfc2)
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(test_hess, torsion_pBmat)
    lfc2 = calc_local_fc_from_pBmat_2(test_hess, torsion_pBmat)
    print(lfc, undfc, lndfc, lfc2)
    print("mass-weighted coordinate")
    elem_mass = np.array([[atomic_mass(elem)**(0.5)] for elem in test_element_list], dtype="float64")
    
    bond_pBmat = partial_stretch_B_matirx(test_coord * elem_mass, 1, 2)
    bend_pBmat = partial_bend_B_matrix(test_coord * elem_mass, 2, 1, 3)
    torsion_pBmat = partial_torsion_B_matrix(test_coord * elem_mass, 2, 1, 3, 4)
    print(bond_pBmat, bend_pBmat, torsion_pBmat)
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(tmp_test_mw_hess, bend_pBmat)
    lfc2 = calc_local_fc_from_pBmat_2(tmp_test_mw_hess, bend_pBmat)
    lfc3 = calc_local_fc_from_pBmat_3(tmp_test_mw_hess, bend_pBmat)
    print(lfc, undfc, lndfc, lfc2 * 5140.48, lfc3, lfc2)
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(tmp_test_mw_hess, bond_pBmat)
    lfc2 = calc_local_fc_from_pBmat_2(tmp_test_mw_hess, bond_pBmat)
    lfc3 = calc_local_fc_from_pBmat_3(tmp_test_mw_hess, bond_pBmat)
    print(lfc, undfc, lndfc, lfc2 * 5140.48, lfc3, lfc2)
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(tmp_test_mw_hess, torsion_pBmat)
    lfc2 = calc_local_fc_from_pBmat_2(tmp_test_mw_hess, torsion_pBmat)
    lfc3 = calc_local_fc_from_pBmat_3(tmp_test_mw_hess, torsion_pBmat)
    print(lfc, undfc, lndfc, lfc2 * 5140.48, lfc3, lfc2)

    stack_Bmat = np.vstack([bond_pBmat, bend_pBmat, torsion_pBmat])
    lfc, undfc, lndfc = calc_local_fc_from_pBmat(tmp_test_mw_hess, stack_Bmat)
    lfc2 = calc_local_fc_from_pBmat_2(tmp_test_mw_hess, stack_Bmat)
    lfc3 = calc_local_fc_from_pBmat_3(tmp_test_mw_hess, stack_Bmat)
    print(lfc, undfc, lndfc, lfc3 * 5140.48, lfc2 * 5140.48, lfc3, lfc2)

    angle_B_matrix_derivative = torch_B_matrix_derivative(torch.tensor(test_coord * elem_mass, dtype=torch.float64), [2, 1, 3], torch_calc_angle)
    angle_B_matrix_derivative = angle_B_matrix_derivative.detach().numpy()
    print(angle_B_matrix_derivative)
    angle_B_matrix = torch_B_matrix(torch.tensor(test_coord * elem_mass, dtype=torch.float64), [2, 1, 3], torch_calc_angle)
    angle_B_matrix = angle_B_matrix.detach().numpy()
    angle_B_matrix = angle_B_matrix.reshape(1, 12)
    print(angle_B_matrix.reshape(1, 12))
    #For test, test_coord is used as gradient.
    int_grad = calc_int_grad_from_pBmat(test_coord.reshape(12, 1), angle_B_matrix)
    cart_grad_from_int_grad = calc_cart_grad_from_pBmat(-int_grad, angle_B_matrix)
    print(((cart_grad_from_int_grad - test_coord.reshape(12, 1))))
    int_hess = calc_int_hess_from_pBmat_for_non_stationary_point(test_hess, angle_B_matrix, angle_B_matrix_derivative, int_grad)
    cart_hess_from_int_hess = calc_cart_hess_from_pBmat_for_non_stationary_point(int_hess, angle_B_matrix, angle_B_matrix_derivative, int_grad)
    print(((cart_hess_from_int_hess - test_hess)))