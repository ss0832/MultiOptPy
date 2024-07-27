import itertools
import math
import numpy as np
import copy

from parameter import UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, element_number, number_element, atomic_mass, UnitValueLib

try:
    import torch
except:
    print("You cannot import pyTorch.")


class CalculationStructInfo:
    def __init__(self):
        return
    
    def calculate_cos(self, bg, g):
        if np.linalg.norm(bg) == 0.0 or np.linalg.norm(g) == 0.0:
            cos = 2.0
        else:
            cos = np.sum(bg * g) / (np.linalg.norm(g) * np.linalg.norm(bg))
        return cos
     
    
    def calculate_distance(self, atom1, atom2):
        atom1, atom2 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64")
        distance = np.linalg.norm(atom2 - atom1)
        return distance

    
    def calculate_bond_angle(self, atom1, atom2, atom3):
        atom1, atom2, atom3 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64"), np.array(atom3, dtype="float64")
        vector1 = atom1 - atom2
        vector2 = atom3 - atom2

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cos_angle)
        angle_deg = np.degrees(angle)

        return angle_deg
        
    def calculate_dihedral_angle(self, atom1, atom2, atom3, atom4):
        atom1, atom2, atom3, atom4 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64"), np.array(atom3, dtype="float64"), np.array(atom4, dtype="float64")
        
        a1 = atom2 - atom1
        a2 = atom3 - atom2
        a3 = atom4 - atom3

        v1 = np.cross(a1, a2)
        v1 = v1 / np.linalg.norm(v1, ord=2)
        v2 = np.cross(a2, a3)
        v2 = v2 / np.linalg.norm(v2, ord=2)
        porm = np.sign((v1 * a3).sum(-1))
        angle = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
        if not porm == 0:
            angle = angle * porm
            
        dihedral_angle_deg = np.degrees(angle)

        return dihedral_angle_deg
        

    def read_xyz_file(self, file_name):
        with open(file_name,"r") as f:
            words = f.readlines()
        mole_struct_list = []
            
        for word in words[1:]:
            mole_struct_list.append(word.split())
        return mole_struct_list

    def Data_extract(self, file, atom_numbers):
        data_list = []
        data_name_list = [] 
         
        
        
        mole_struct_list = self.read_xyz_file(file)
        DBD_list = []
        DBD_name_list = []
        #print(file, atom_numbers)
        if len(atom_numbers) > 1:
            for a1, a2 in list(itertools.combinations(atom_numbers,2)):
                #print(a1, a2)
                try:
                    distance = self.calculate_distance(mole_struct_list[int(a1) - 1][1:4], mole_struct_list[int(a2) - 1][1:4])
                    DBD_name_list.append("Distance ("+str(a1)+"-"+str(a2)+")  [ang.]")
                    DBD_list.append(distance)
                        
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Distance ("+str(a1)+"-"+str(a2)+")  [ang.]")
                    DBD_list.append("nan")
                
        if len(atom_numbers) > 2:
            for a1, a2, a3 in list(itertools.permutations(atom_numbers,3)):
                try:
                    bond_angle = self.calculate_bond_angle(mole_struct_list[int(a1)-1][1:4], mole_struct_list[int(a2)-1][1:4], mole_struct_list[int(a3)-1][1:4])
                    DBD_name_list.append("Bond_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+") [deg.]")
                    DBD_list.append(bond_angle)
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Bond_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+") [deg.]")
                    DBD_list.append("nan")            
        
        if len(atom_numbers) > 3:
            for a1, a2, a3, a4 in list(itertools.permutations(atom_numbers,4)):
                try:
                    dihedral_angle = self.calculate_dihedral_angle(mole_struct_list[int(a1)-1][1:4], mole_struct_list[int(a2)-1][1:4],mole_struct_list[int(a3)-1][1:4], mole_struct_list[int(a4)-1][1:4])
                    DBD_name_list.append("Dihedral_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+"-"+str(a4)+") [deg.]")
                    DBD_list.append(dihedral_angle)
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Dihedral_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+"-"+str(a4)+") [deg.]")
                    DBD_list.append("nan")        

        data_list = DBD_list 
        
        data_name_list = DBD_name_list    
        return data_list, data_name_list


class Calculationtools:
    def __init__(self):
        return
    
    def calc_center(self, geomerty, element_list=[]):#geomerty:Bohr
        center = np.array([0.0, 0.0, 0.0], dtype="float64")
        for i in range(len(geomerty)):
            
            center += geomerty[i] 
        center /= float(len(geomerty))
        
        return center
            
            
    def calc_center_of_mass(self, geomerty, element_list):#geomerty:Bohr
        center_of_mass = np.array([0.0, 0.0, 0.0], dtype="float64")
        elem_mass = np.array([atomic_mass(elem) for elem in element_list], dtype="float64")
        
        for i in range(len(elem_mass)):
           
            center_of_mass += geomerty[i] * elem_mass[i]
        
        center_of_mass /= np.sum(elem_mass)
       
        return center_of_mass
    
    def coord2massweightedcoord(self, geomerty, element_list):
        #output: Mass-weighted coordinates adjusted to the origin of the mass-weighted point
        center_of_mass = self.calc_center_of_mass(geomerty, element_list)
        geomerty -= center_of_mass
        elem_mass = np.array([atomic_mass(elem) for elem in element_list], dtype="float64")
        mass_weighted_coord = geomerty * 0.0
        for i in range(len(geomerty)):
            mass_weighted_coord[i] = copy.copy(geomerty[i] * np.sqrt(elem_mass[i]))
        return mass_weighted_coord
    
    def project_out_hess_tr_and_rot(self, hessian, element_list, geomerty):#covert coordination to mass-weighted coordination
        natoms = len(element_list)
        
        geomerty -= self.calc_center_of_mass(geomerty, element_list)
        
        elem_mass = np.array([atomic_mass(elem) for elem in element_list], dtype="float64")
        
        M = np.diag(np.repeat(elem_mass, 3))
        #M_plus_sqrt = np.diag(np.repeat(elem_mass, 3) ** (0.5))
        M_minus_sqrt = np.diag(np.repeat(elem_mass, 3) ** (-0.5))

        m_plus_sqrt = np.repeat(elem_mass, 3) ** (0.5)
        #m_minus_sqrt = np.repeat(elem_mass, 3) ** (-0.5)

        mw_hessian = np.dot(np.dot(M_minus_sqrt, hessian), M_minus_sqrt)#mw = mass weighted
        
        tr_x = (np.tile(np.array([1, 0, 0]), natoms)).reshape(-1, 3)
        tr_y = (np.tile(np.array([0, 1, 0]), natoms)).reshape(-1, 3)
        tr_z = (np.tile(np.array([0, 0, 1]), natoms)).reshape(-1, 3)

        mw_rot_x = np.cross(geomerty, tr_x).flatten() * m_plus_sqrt
        mw_rot_y = np.cross(geomerty, tr_y).flatten() * m_plus_sqrt
        mw_rot_z = np.cross(geomerty, tr_z).flatten() * m_plus_sqrt

        mw_tr_x = tr_x.flatten() * m_plus_sqrt
        mw_tr_y = tr_y.flatten() * m_plus_sqrt
        mw_tr_z = tr_z.flatten() * m_plus_sqrt

        TR_vectors = np.vstack([mw_tr_x, mw_tr_y, mw_tr_z, mw_rot_x, mw_rot_y, mw_rot_z])
        
        Q, R = np.linalg.qr(TR_vectors.T)
        keep_indices = ~np.isclose(np.diag(R), 0, atol=1e-6, rtol=0)
        TR_vectors = Q.T[keep_indices]
        n_tr = len(TR_vectors)

        P = np.identity(natoms * 3)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)

        mw_hess_proj = np.dot(np.dot(P.T, mw_hessian), P)

        eigenvalues, eigenvectors = np.linalg.eigh(mw_hess_proj)
        idx_eigenvalues = np.where((eigenvalues > 1e-10) | (eigenvalues < -1e-10))
        print("=== hessian projected out transition and rotation (mass-weighted coordination) ===")
        print("eigenvalues: ", eigenvalues[idx_eigenvalues])
        return mw_hess_proj

    def project_out_hess_tr_and_rot_for_coord(self, hessian, element_list, geomerty):#do not consider atomic mass
        natoms = len(element_list)
       
        geomerty -= self.calc_center(geomerty, element_list)
        
    
        tr_x = (np.tile(np.array([1, 0, 0]), natoms)).reshape(-1, 3)
        tr_y = (np.tile(np.array([0, 1, 0]), natoms)).reshape(-1, 3)
        tr_z = (np.tile(np.array([0, 0, 1]), natoms)).reshape(-1, 3)

        rot_x = np.cross(geomerty, tr_x).flatten()
        rot_y = np.cross(geomerty, tr_y).flatten() 
        rot_z = np.cross(geomerty, tr_z).flatten()
        tr_x = tr_x.flatten()
        tr_y = tr_y.flatten()
        tr_z = tr_z.flatten()

        TR_vectors = np.vstack([tr_x, tr_y, tr_z, rot_x, rot_y, rot_z])
        
        Q, R = np.linalg.qr(TR_vectors.T)
        keep_indices = ~np.isclose(np.diag(R), 0, atol=1e-6, rtol=0)
        TR_vectors = Q.T[keep_indices]
        n_tr = len(TR_vectors)

        P = np.identity(natoms * 3)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)

        hess_proj = np.dot(np.dot(P.T, hessian), P)

        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        idx_eigenvalues = np.where((eigenvalues > 1e-10) | (eigenvalues < -1e-10))
        print("=== hessian projected out transition and rotation (normal coordination) ===")
        print("eigenvalues: ", eigenvalues[idx_eigenvalues])
        return hess_proj    
    
  
    def check_atom_connectivity(self, mol_list, element_list, atom_num, covalent_radii_threshold_scale=1.2):
        connected_atoms = [atom_num]
        searched_atoms = []
        while True:
            for i in connected_atoms:
                if i in searched_atoms:
                    continue
                
                for j in range(len(mol_list)):
                    dist = np.linalg.norm(np.array(mol_list[i], dtype="float64") - np.array(mol_list[j], dtype="float64"))
                    
                    covalent_dist_threshold = covalent_radii_threshold_scale * (covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j]))
                    
                    if dist < covalent_dist_threshold:
                        if not j in connected_atoms:
                            connected_atoms.append(j)
                
                searched_atoms.append(i)
            
            if len(connected_atoms) == len(searched_atoms):
                break
     
        return sorted(connected_atoms)
    
    def calc_fragm_distance_matrix(self, fragm_coord_list):
        distance_matrix = np.zeros((len(fragm_coord_list), len(fragm_coord_list)))
        for i in range(len(fragm_coord_list)):
            for j in range(len(fragm_coord_list)):
                if i < j:
                    continue
                dist = np.linalg.norm(self.calc_center(fragm_coord_list[i], []) - self.calc_center(fragm_coord_list[j], []))
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        
        return distance_matrix
    
    
    def calc_fragm_distance(self, geom_num_list, fragm_1_num, fragm_2_num):
        fragm_1_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        fragm_2_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        
        for num in fragm_1_num:
            fragm_1_coord += geom_num_list[num]
        
        fragm_1_coord /= len(fragm_1_num)
            
        for num in fragm_2_num:
            fragm_2_coord += geom_num_list[num]
        
        fragm_2_coord /= len(fragm_2_num)
        
        dist = np.linalg.norm(fragm_1_coord - fragm_2_coord)
        
        return dist

    def calc_geodesic_distance(self, geom_num_list_1, geom_num_list_2):
        #doi:10.1002/jcc.27030
        geodesic_dist_mat = np.ones((len(geom_num_list_1), 3))
        dist = np.linalg.norm(geom_num_list_2 - geom_num_list_1)
        geodesic_dist_mat *= dist / np.sqrt(3 * len(geom_num_list_1))
        return geodesic_dist_mat
    
    def calc_euclidean_distance(self, geom_num_list_1, geom_num_list_2):
        #doi:10.1002/jcc.27030
        euclidean_dist_mat = geom_num_list_2 - geom_num_list_1
        return euclidean_dist_mat

    def kabsch_algorithm(self, P, Q):
        #scipy.spatial.transform.Rotation.align_vectors
        centroid_P = np.array([np.mean(P.T[0]), np.mean(P.T[1]), np.mean(P.T[2])], dtype="float64")
        centroid_Q = np.array([np.mean(Q.T[0]), np.mean(Q.T[1]), np.mean(Q.T[2])], dtype="float64")
        P -= centroid_P
        Q -= centroid_Q
        H = np.dot(P.T, Q)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = np.dot(Vt.T, U.T)
        P = np.dot(R, P.T).T
        return P, Q
    
    def torch_affine_transformation(self, geom_num_list, tr, LJ_center_vec, delta_angle):
        natoms = len(geom_num_list)
        ones = torch.ones(natoms, 1, requires_grad=True)
        tmp_geom_num_list = torch.t(torch.cat((geom_num_list, ones), dim=1))
        
        
        # affine translation
        affine_tr_matrix = torch.tensor([[1.0 , 0.0, 0.0, -tr[0]],
                                         [0.0 , 1.0, 0.0, -tr[1]],
                                         [0.0 , 0.0, 1.0, -tr[2]],
                                         [0.0 , 0.0, 0.0, 1.0 ]], dtype=torch.float64, requires_grad=True)
                                         
        if torch.linalg.norm(LJ_center_vec[1:3]) != 0.0:                      
            cos_x_angle = torch.matmul(LJ_center_vec[1:3], torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)) / (torch.linalg.norm(LJ_center_vec[1:3]) * torch.linalg.norm(torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)))
        else:
            cos_x_angle = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        
        
        if LJ_center_vec[2] < 0:
            x_angle = -torch.pi/2 + torch.arccos(cos_x_angle)
        else:
            x_angle = torch.pi/2 - torch.arccos(cos_x_angle)   

        
        affine_x_rot_matrix = torch.tensor([[1.0              , 0.0          , 0.0          , 0.0],
                                            [0.0              , torch.cos(x_angle),-torch.sin(x_angle), 0.0],
                                            [0.0              , torch.sin(x_angle), torch.cos(x_angle), 0.0],
                                            [0.0              , 0.0          , 0.0          , 1.0]], dtype=torch.float64, requires_grad=True)
        
        tmp_LJ_center_vec = torch.cat((LJ_center_vec, torch.tensor([1.0], requires_grad=True)), dim=0)
        
        
        after_x_rot_LJ_center_vec = torch.t(torch.matmul(affine_x_rot_matrix, torch.t(tmp_LJ_center_vec.reshape(1, 4))))
        

        xz = torch.cat((after_x_rot_LJ_center_vec[0][0].reshape(1), after_x_rot_LJ_center_vec[0][2].reshape(1)))
        
        if torch.linalg.norm(xz) != 0.0:
            cos_y_angle = torch.matmul(xz, torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)) / (torch.linalg.norm(xz) * torch.linalg.norm(torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)))
        else:
            cos_y_angle = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            
        if LJ_center_vec[2] < 0:
            y_angle = -torch.pi/2 - torch.arccos(cos_y_angle)
        else:
            y_angle = torch.pi/2 + torch.arccos(cos_y_angle)

        affine_y_rot_matrix = torch.tensor([[torch.cos(y_angle)  , 0.0, torch.sin(y_angle), 0.0],
                                            [0.0              , 1.0, 0.0            , 0.0],
                                            [-torch.sin(y_angle) , 0.0, torch.cos(y_angle), 0.0],
                                            [0.0              , 0.0, 0.0            , 1.0]], dtype=torch.float64, requires_grad=True)
        
        
        after_y_x_rot_LJ_center_vec = torch.t(torch.matmul(affine_y_rot_matrix, torch.t(after_x_rot_LJ_center_vec.reshape(1, 4))))
        
        # to adjust LJ center vector to z axis of specific direction.
        if torch.linalg.norm(after_y_x_rot_LJ_center_vec[0:3]) != 0.0:                      
            cos_x_angle_2 = torch.matmul(after_y_x_rot_LJ_center_vec[0][0:3], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)) / (torch.linalg.norm(after_y_x_rot_LJ_center_vec[0][0:3]) * torch.linalg.norm(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)))
            
        else:
            cos_x_angle_2 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            
        
        x_angle_2 = torch.arccos(cos_x_angle_2)
       
        affine_x_rot_matrix_2 = torch.tensor([[1.0             , 0.0                , 0.0                 , 0.0],
                                             [0.0              , torch.cos(x_angle_2), -torch.sin(x_angle_2), 0.0],
                                             [0.0              , torch.sin(x_angle_2),  torch.cos(x_angle_2), 0.0],
                                             [0.0              , 0.0                , 0.0                 , 1.0]], dtype=torch.float64, requires_grad=True)
        

        affine_z_rot_matrix = torch.tensor([[torch.cos(delta_angle) , -torch.sin(delta_angle), 0.0, 0.0],
                                            [torch.sin(delta_angle) , torch.cos(delta_angle) , 0.0, 0.0],
                                            [0.0             , 0.0             , 1.0, 0.0],
                                            [0.0             , 0.0             , 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
                                            
                                           
        tr_rot_matrix = torch.matmul(affine_z_rot_matrix, torch.matmul(affine_x_rot_matrix_2, torch.matmul(affine_y_rot_matrix, torch.matmul(affine_x_rot_matrix, affine_tr_matrix))))
        
        tmp_transformed_geom_num_list = torch.t(torch.matmul(tr_rot_matrix, tmp_geom_num_list))# [[x,y,z,1] ...]
        transformed_geom_num_list = torch.tensor_split(tmp_transformed_geom_num_list, (0, 3), dim=1)[1]
        # [[x,y,z] ...]
        
        return transformed_geom_num_list
        

    def gen_n_dinensional_rot_matrix(self, vector_1, vector_2):
        #Zhelezov NRMG algorithm (doi:10.5923/j.ajcam.20170702.04)
        dimension_1 = len(vector_1)
        dimension_2 = len(vector_2)
        assert dimension_1 == dimension_2
        R_1 = np.eye((dimension_1))
        R_2 = np.eye((dimension_2))
        
        
        
        step = 1
        
        while step < dimension_1:
            A_1 = np.eye((dimension_1))
            n = 1
            #print(step)
            while n <= dimension_1 - step:
                #print(n)
                #print(vector_1[n + step - 1])
                r2 = vector_1[n - 1] ** 2 + vector_1[n + step - 1] ** 2
                if r2 > 0:
                    r = r2 ** 0.5
                    p_cos = vector_1[n - 1] / r
                    
                    p_sin = -1 * vector_1[n + step - 1] / r
                    A_1[n - 1][n - 1] = p_cos.item()
                    A_1[n - 1][n + step - 1] = -1 * p_sin.item()
                    A_1[n + step - 1][n - 1] = p_sin.item()
                    A_1[n + step - 1][n + step - 1] = p_cos.item()
                n += 2 * step
            step *= 2
            vector_1 = np.dot(A_1, vector_1)
            R_1 = np.dot(A_1, R_1)

        step = 1
        
        while step < dimension_2:
            A_2 = np.eye((dimension_2))
            n = 1
            while n <= dimension_2 - step:
                r2 = vector_2[n - 1] ** 2 + vector_2[n + step - 1] ** 2
                if r2 > 0:
                    r = r2 ** 0.5
                    p_cos = vector_2[n - 1] / r
                    p_sin = -1 * vector_2[n + step - 1] / r
                    A_2[n - 1][n - 1] = p_cos.item()
                    A_2[n - 1][n + step - 1] = -1 * p_sin.item()
                    A_2[n + step - 1][n - 1] = p_sin.item()
                    A_2[n + step - 1][n + step - 1] = p_cos.item()
                n += 2 * step
            step *= 2
            vector_2 = np.dot(A_2, vector_2)
            R_2 = np.dot(A_2, R_2)
        #print(R_1, R_2)
        R_12 = np.dot(R_2.T, R_1)
        #vector_1 -> vector_2's direction
        return R_12

    def calc_multi_dim_vec_angle(self, vec_1, vec_2):
        
        angle = np.arccos(np.sum(vec_1 * vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)) + 1e-8)
        
        return angle

def calc_normalized_distance_list(geom_num_list, element_list, tgt_atoms=None):
    if tgt_atoms is not None:
        atom_list = [i for i in tgt_atoms]
    else:
        atom_list = [i for i in range(len(geom_num_list))]
    
    norm_distance_list = []
    
    for i, j in itertools.combinations(atom_list, 2):#(0, 1) (0, 2) ... (natoms-2, natoms-1)
        elem_i = element_list[i]
        elem_j = element_list[j]
        covalent_length = covalent_radii_lib(elem_i) + covalent_radii_lib(elem_j)
        norm_distance = np.linalg.norm(geom_num_list[i] - geom_num_list[j]) / covalent_length
        norm_distance_list.append(norm_distance)
    norm_distance_list = np.array(norm_distance_list)
    return norm_distance_list

def return_pair_idx(i, j):
    ii = max(i, j) + 1
    jj = min(i, j) + 1
    pair_idx = int(ii * (ii - 1) / 2 - (ii - jj)) -1
    return pair_idx


def torch_calc_angle_from_vec(vector1, vector2):
    magnitude1 = torch.linalg.norm(vector1)
    magnitude2 = torch.linalg.norm(vector2)
    dot_product = torch.matmul(vector1, vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = torch.arccos(cos_theta)
    return theta

def torch_calc_dihedral_angle_from_vec(vector1, vector2, vector3):
    v1 = torch.linalg.cross(vector1, vector2)
    v2 = torch.linalg.cross(vector2, vector3)
    norm_v1 = torch.linalg.norm(v1) + 1e-15
    norm_v2 = torch.linalg.norm(v2) + 1e-15
    cos_theta = torch.sum(v1*v2) / (norm_v1 * norm_v2)
    angle = torch.arccos(cos_theta)
    return angle


def torch_calc_outofplain_angle_from_vec(vector1, vector2, vector3):
    v1 = torch.linalg.cross(vector1, vector2)
    magnitude1 = torch.linalg.norm(v1)
    magnitude2 = torch.linalg.norm(vector3)
    dot_product = torch.matmul(v1, vector3)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle = torch.arccos(cos_theta)
    return angle



def output_partial_hess(hessian, atom_num_list, element_list, geometry):#hessian: ndarray 3N*3N, atom_num_list: list
    partial_hess = np.zeros((3*len(atom_num_list), 3*len(atom_num_list)))
    partial_geom = np.zeros((len(atom_num_list), 3))
    partial_element_list = []
   
    # Copy the relevant parts of the geometry and element list
    for i in range(len(atom_num_list)):
        partial_geom[i] = copy.copy(geometry[atom_num_list[i]-1])
        partial_element_list.append(element_list[atom_num_list[i]-1])
    
    # Copy the relevant parts of the Hessian matrix
    for i, j in itertools.product(range(len(atom_num_list)), repeat=2):
        for k in range(3):
            for l in range(3):
                partial_hess[3*i+k][3*j+l] = copy.copy(hessian[3*(atom_num_list[i]-1)+k][3*(atom_num_list[j]-1)+l])
    
    return partial_hess, partial_geom, partial_element_list

def fragment_check(new_geometry, element_list):
    atom_label_list = [i for i in range(len(new_geometry))]
    fragm_atom_num_list = []
    while len(atom_label_list) > 0:
        tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0], covalent_radii_threshold_scale=1.2)
        
        for j in tmp_fragm_list:
            atom_label_list.remove(j)
        fragm_atom_num_list.append(tmp_fragm_list)
    
    print("\nfragment_list:", fragm_atom_num_list)
    
    return fragm_atom_num_list    

def rotate_molecule(geom, axis, angle):
    #geom: ndarray, axis: str, angle: float (radian)
    #axis: "x", "y", "z"
    if axis == "x":
        rot_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, np.cos(angle), -np.sin(angle)],
                               [0.0, np.sin(angle), np.cos(angle)]], dtype="float64")
    elif axis == "y":
        rot_matrix = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                               [0.0, 1.0, 0.0],
                               [-np.sin(angle), 0.0, np.cos(angle)]], dtype="float64")
    elif axis == "z":
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                               [np.sin(angle), np.cos(angle), 0.0],
                               [0.0, 0.0, 1.0]], dtype="float64")
    else:
        print("Invalid axis.")
        return
    
    new_geom = np.dot(geom, rot_matrix)
    
    return new_geom


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
    partial_hess, partial_geom, partial_element_list = output_partial_hess(test_hess, [1,2,3,4], test_element_list, test_coord)
    p_partial_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(partial_hess, partial_element_list, partial_geom)
    partial_eigenvalue, partial_eigenvector = np.linalg.eigh(p_partial_hess)
    print(partial_eigenvalue)
    
    mw_partial_hess = Calculationtools().project_out_hess_tr_and_rot(partial_hess, partial_element_list, partial_geom)
    partial_eigenvalue, partial_eigenvector = np.linalg.eigh(mw_partial_hess)
    print(partial_eigenvalue)
    