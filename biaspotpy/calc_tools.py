import itertools
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

def hess2mwhess(hessian, element_list):
    elem_mass = np.array([atomic_mass(elem) for elem in element_list], dtype="float64")    
    M = np.diag(np.repeat(elem_mass, 3))
    M_minus_sqrt = np.diag(np.repeat(elem_mass, 3) ** (-0.5))
    mw_hessian = np.dot(np.dot(M_minus_sqrt, hessian), M_minus_sqrt)#mw = mass weighted
    return mw_hessian


class Calculationtools:
    def __init__(self):
        return
    
    def calc_center(self, geomerty, element_list=[]):#geomerty:Bohr
        center = np.array([0.0, 0.0, 0.0], dtype="float64")
        for i in range(len(geomerty)):
            
            center += geomerty[i] 
        center /= len(geomerty)
        
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
        
        geomerty = geomerty - self.calc_center_of_mass(geomerty, element_list)
        
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

        eigenvalues, eigenvectors = np.linalg.eig(mw_hess_proj)
        eigenvalues = eigenvalues.astype(np.float64)
        eigenvalues = np.sort(eigenvalues)
        idx_eigenvalues = np.where((eigenvalues > 1e-10) | (eigenvalues < -1e-10))
        print("=== hessian projected out transition and rotation (mass-weighted coordination) ===")
        print(f"eigenvalues (NUMBER OF VALUES: {len(eigenvalues[idx_eigenvalues])}): \n", eigenvalues[idx_eigenvalues])
        return mw_hess_proj

    def project_out_hess_tr_and_rot_for_coord(self, hessian, element_list, geomerty):#do not consider atomic mass
        natoms = len(element_list)
       
        geomerty = geomerty - self.calc_center(geomerty, element_list)
        
    
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

        eigenvalues, eigenvectors = np.linalg.eig(hess_proj)
        eigenvalues = eigenvalues.astype(np.float64)
        eigenvalues = np.sort(eigenvalues)
        idx_eigenvalues = np.where((eigenvalues > 1e-10) | (eigenvalues < -1e-10))
        print("=== hessian projected out transition and rotation (normal coordination) ===")
        print(f"eigenvalues (NUMBER OF VALUES: {len(eigenvalues[idx_eigenvalues])}): \n", eigenvalues[idx_eigenvalues])
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
        
    def gen_n_dinensional_rot_matrix(self, vector_1, vector_2):
        #Zhelezov NRMG algorithm (doi:10.5923/j.ajcam.20170702.04) This implementation may be not correct.
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

def calc_bond_length_from_vec(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    return distance

def torch_calc_angle_from_vec(vector1, vector2):
    magnitude1 = torch.linalg.norm(vector1) + 1e-15
    magnitude2 = torch.linalg.norm(vector2) + 1e-15
    dot_product = torch.matmul(vector1, vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = torch.arccos(cos_theta)
    return theta

def calc_angle_from_vec(vector1, vector2):
    magnitude1 = np.linalg.norm(vector1) + 1e-15
    magnitude2 = np.linalg.norm(vector2) + 1e-15
    dot_product = np.matmul(vector1, vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = np.arccos(cos_theta)
    return theta

def torch_calc_dihedral_angle_from_vec(vector1, vector2, vector3):
    v1 = torch.linalg.cross(vector1, vector2)
    v2 = torch.linalg.cross(vector2, vector3)
    norm_v1 = torch.linalg.norm(v1) + 1e-15
    norm_v2 = torch.linalg.norm(v2) + 1e-15
    cos_theta = torch.sum(v1*v2) / (norm_v1 * norm_v2)
    angle = torch.arccos(cos_theta)
    return angle

def change_torsion_angle_both_side(coordinates, atom_idx1, atom_idx2, atom_idx3, atom_idx4, target_torsion):#rad:target_torsion
    A = coordinates[atom_idx1]
    B = coordinates[atom_idx2]
    C = coordinates[atom_idx3]
    D = coordinates[atom_idx4]
    current_torsion = calc_dihedral_angle_from_vec(A - B, B - C, C - D)
    torsion_diff = target_torsion - current_torsion 

    BC = C - B
    new_D = rotate_atom(D, C, BC, torsion_diff * 0.5)
    new_A = rotate_atom(A, B, BC, -1*torsion_diff * 0.5)     
    coordinates[atom_idx4] = new_D
    coordinates[atom_idx1] = new_A

    return coordinates

def change_bond_angle_both_side(coordinates, atom_idx1, atom_idx2, atom_idx3, target_angle):#rad:target_angle
    A = coordinates[atom_idx1]
    B = coordinates[atom_idx2]
    C = coordinates[atom_idx3]
    BA = A - B
    BC = C - B
    current_angle_rad = calc_angle_from_vec(BA, BC)
    rotation_axis = np.cross(BA, BC) 
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle_diff = target_angle - current_angle_rad
    new_A = rotate_atom(A, B, rotation_axis, -angle_diff / 2.0)
    new_C = rotate_atom(C, B, rotation_axis, angle_diff / 2.0) 
    coordinates[atom_idx1] = new_A
    coordinates[atom_idx3] = new_C
    return coordinates

def calc_dihedral_angle_from_vec(vector1, vector2, vector3):
    v1 = np.cross(vector1, vector2)
    v2 = np.cross(vector2, vector3)
    norm_v1 = np.linalg.norm(v1) + 1e-15
    norm_v2 = np.linalg.norm(v2) + 1e-15
    cos_theta = np.sum(v1*v2) / (norm_v1 * norm_v2)
    angle = np.arccos(cos_theta)
    return angle

def rotate_atom(coord, axis_point, axis_direction, angle):
    axis_unit = axis_direction / np.linalg.norm(axis_direction)
    translated_coord = coord - axis_point
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + axis_unit[0]**2 * (1 - cos_angle), axis_unit[0]*axis_unit[1]*(1 - cos_angle) - axis_unit[2]*sin_angle, axis_unit[0]*axis_unit[2]*(1 - cos_angle) + axis_unit[1]*sin_angle],
        [axis_unit[1]*axis_unit[0]*(1 - cos_angle) + axis_unit[2]*sin_angle, cos_angle + axis_unit[1]**2 * (1 - cos_angle), axis_unit[1]*axis_unit[2]*(1 - cos_angle) - axis_unit[0]*sin_angle],
        [axis_unit[2]*axis_unit[0]*(1 - cos_angle) - axis_unit[1]*sin_angle, axis_unit[2]*axis_unit[1]*(1 - cos_angle) + axis_unit[0]*sin_angle, cos_angle + axis_unit[2]**2 * (1 - cos_angle)]
    ])
    rotated_coord = np.dot(rotation_matrix, translated_coord)
    return rotated_coord + axis_point

def torch_calc_outofplain_angle_from_vec(vector1, vector2, vector3):
    v1 = torch.linalg.cross(vector1, vector2)
    magnitude1 = torch.linalg.norm(v1) + 1e-15 
    magnitude2 = torch.linalg.norm(vector3) + 1e-15
    dot_product = torch.matmul(v1, vector3)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle = torch.arccos(cos_theta)
    return angle

def calc_outofplain_angle_from_vec(vector1, vector2, vector3):
    v1 = np.cross(vector1, vector2)
    magnitude1 = np.linalg.norm(v1) + 1e-15 
    magnitude2 = np.linalg.norm(vector3) + 1e-15
    dot_product = np.matmul(v1, vector3)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle = np.arccos(cos_theta)
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

def calc_partial_center(geometry, atom_num_list):
    partial_center = np.array([0.0, 0.0, 0.0], dtype="float64")
    
    for i in atom_num_list:
        partial_center += geometry[i-1]
    partial_center /= len(atom_num_list)
    
    return partial_center

def torch_calc_partial_center(geometry, atom_num_list):
    partial_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    for i in atom_num_list:
        partial_center = partial_center + geometry[i-1]
    partial_center /= len(atom_num_list)
    return partial_center

def project_optional_vector_for_grad(gradient, vector):#gradient:ndarray (3*natoms, 1), vector:ndarray (3*natoms, 1)
    unit_vec = vector / np.linalg.norm(vector)
    P_matrix = np.eye((len(gradient))) -1 * np.dot(unit_vec, unit_vec.T)
    gradient_proj = np.dot(P_matrix, gradient).reshape(len(vector), 1)
    return gradient_proj #gradient_proj:ndarray (3*natoms, 1)

def project_optional_vector_for_hess(hessian, vector):#hessian:ndarray (3*natoms, 3*natoms), vector:ndarray (3*natoms, 1)
    hess_length = len(vector)
    identity_matrix = np.eye(hess_length)
    LL = np.dot(vector, vector.T)
    E_LL = identity_matrix - LL
    hessian_proj = np.dot(np.dot(E_LL, hessian), E_LL)
    return hessian_proj #hessian:ndarray (3*natoms, 3*natoms)

def project_fragm_pair_vector_for_grad(gradient, geom_num_list, fragm_1, fragm_2):
    natom = len(geom_num_list)
    fragm_1_center = calc_partial_center(geom_num_list, fragm_1)
    fragm_2_center = calc_partial_center(geom_num_list, fragm_2)
    fragm_vec = fragm_2_center - fragm_1_center / (np.linalg.norm(fragm_2_center - fragm_1_center))
    fragm_vec_x = np.array([fragm_vec[0], 0.0, 0.0], dtype="float64")
    fragm_vec_y = np.array([0.0, fragm_vec[1], 0.0], dtype="float64")
    fragm_vec_z = np.array([0.0, 0.0, fragm_vec[2]], dtype="float64")
    
    tmp_fragm_vec_x = np.array([], dtype="float64")
    tmp_fragm_vec_y = np.array([], dtype="float64")
    tmp_fragm_vec_z = np.array([], dtype="float64")
    for i in range(len(geom_num_list)):
        if i + 1 in fragm_1:
            tmp_fragm_vec_x = np.append(tmp_fragm_vec_x, -1 * fragm_vec_x, axis=0)
            tmp_fragm_vec_y = np.append(tmp_fragm_vec_y, -1 * fragm_vec_y, axis=0)
            tmp_fragm_vec_z = np.append(tmp_fragm_vec_z, -1 * fragm_vec_z, axis=0)
        elif i + 1 in fragm_2:
            tmp_fragm_vec_x = np.append(tmp_fragm_vec_x, fragm_vec_x, axis=0)
            tmp_fragm_vec_y = np.append(tmp_fragm_vec_y, fragm_vec_y, axis=0)
            tmp_fragm_vec_z = np.append(tmp_fragm_vec_z, fragm_vec_z, axis=0)
        
        else:
            tmp_fragm_vec_x = np.append(tmp_fragm_vec_x, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y = np.append(tmp_fragm_vec_y, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z = np.append(tmp_fragm_vec_z, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
    
    tmp_fragm_vec_x = tmp_fragm_vec_x.reshape(3*natom, 1)
    tmp_fragm_vec_y = tmp_fragm_vec_y.reshape(3*natom, 1)
    tmp_fragm_vec_z = tmp_fragm_vec_z.reshape(3*natom, 1)
    tmp_fragm_vec_x = tmp_fragm_vec_x / (np.linalg.norm(tmp_fragm_vec_x) + 1e-15)
    tmp_fragm_vec_y = tmp_fragm_vec_y / (np.linalg.norm(tmp_fragm_vec_y) + 1e-15)
    tmp_fragm_vec_z = tmp_fragm_vec_z / (np.linalg.norm(tmp_fragm_vec_z) + 1e-15)
    tmp_gradient = gradient.reshape(3*natom, 1)
    
    gradient_proj = project_optional_vector_for_grad(tmp_gradient, tmp_fragm_vec_x)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z)
    gradient_proj = gradient_proj.reshape(natom, 3)
    return gradient_proj # ndarray (natoms, 3)

def project_fragm_pair_vector_for_hess(hessian, geom_num_list, fragm_1, fragm_2):
    natom = len(geom_num_list)
    fragm_1_center = calc_partial_center(geom_num_list, fragm_1)
    fragm_2_center = calc_partial_center(geom_num_list, fragm_2)
    fragm_vec = fragm_2_center - fragm_1_center / (np.linalg.norm(fragm_2_center - fragm_1_center))

    fragm_vec_x = np.array([1.0, 0.0, 0.0], dtype="float64")
    fragm_vec_y = np.array([0.0, 1.0, 0.0], dtype="float64")
    fragm_vec_z = np.array([0.0, 0.0, 1.0], dtype="float64")

    fragm_rot_vec_x = np.array([0.0, fragm_vec[2], -1*fragm_vec[1]], dtype="float64") #np.cross(fragm_vec, np.array([1.0, 0.0, 0.0], dtype="float64"))
    fragm_rot_vec_y = np.array([-1*fragm_vec[2], 0.0, -fragm_vec[0]], dtype="float64") #np.cross(fragm_vec, np.array([1.0, 0.0, 0.0], dtype="float64"))
    fragm_rot_vec_z = np.array([fragm_vec[1], -1*fragm_vec[0], 0.0], dtype="float64") #np.cross(fragm_vec, np.array([1.0, 0.0, 0.0], dtype="float64"))
    
    tmp_fragm_vec_x = np.array([], dtype="float64")
    tmp_fragm_vec_y = np.array([], dtype="float64")
    tmp_fragm_vec_z = np.array([], dtype="float64")
    tmp_fragm_rot_vec_x = np.array([], dtype="float64")
    tmp_fragm_rot_vec_y = np.array([], dtype="float64")
    tmp_fragm_rot_vec_z = np.array([], dtype="float64")
    
    
    for i in range(len(geom_num_list)):
        if i + 1 in fragm_1:
            tmp_fragm_vec_x = np.append(tmp_fragm_vec_x, -1 * fragm_vec_x, axis=0)
            tmp_fragm_vec_y = np.append(tmp_fragm_vec_y, -1 * fragm_vec_y, axis=0)
            tmp_fragm_vec_z = np.append(tmp_fragm_vec_z, -1 * fragm_vec_z, axis=0)
            tmp_fragm_rot_vec_x = np.append(tmp_fragm_rot_vec_x, fragm_rot_vec_x, axis=0)
            tmp_fragm_rot_vec_y = np.append(tmp_fragm_rot_vec_y, fragm_rot_vec_y, axis=0)
            tmp_fragm_rot_vec_z = np.append(tmp_fragm_rot_vec_z, fragm_rot_vec_z, axis=0)

        elif i + 1 in fragm_2:
            tmp_fragm_vec_x = np.append(tmp_fragm_vec_x, fragm_vec_x, axis=0)
            tmp_fragm_vec_y = np.append(tmp_fragm_vec_y, fragm_vec_y, axis=0)
            tmp_fragm_vec_z = np.append(tmp_fragm_vec_z, fragm_vec_z, axis=0)
            tmp_fragm_rot_vec_x = np.append(tmp_fragm_rot_vec_x, fragm_rot_vec_x, axis=0)
            tmp_fragm_rot_vec_y = np.append(tmp_fragm_rot_vec_y, fragm_rot_vec_y, axis=0)
            tmp_fragm_rot_vec_z = np.append(tmp_fragm_rot_vec_z, fragm_rot_vec_z, axis=0)
        
        else:
            tmp_fragm_vec_x = np.append(tmp_fragm_vec_x, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y = np.append(tmp_fragm_vec_y, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z = np.append(tmp_fragm_vec_z, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_rot_vec_x = np.append(tmp_fragm_rot_vec_x, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_rot_vec_y = np.append(tmp_fragm_rot_vec_y, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_rot_vec_z = np.append(tmp_fragm_rot_vec_z, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)    
            
    tmp_fragm_vec_x = tmp_fragm_vec_x.reshape(3*natom, 1)
    tmp_fragm_vec_y = tmp_fragm_vec_y.reshape(3*natom, 1)
    tmp_fragm_vec_z = tmp_fragm_vec_z.reshape(3*natom, 1)
    tmp_fragm_rot_vec_x = tmp_fragm_rot_vec_x.reshape(3*natom, 1)
    tmp_fragm_rot_vec_y = tmp_fragm_rot_vec_y.reshape(3*natom, 1)
    tmp_fragm_rot_vec_z = tmp_fragm_rot_vec_z.reshape(3*natom, 1)
    tmp_fragm_vec_x = tmp_fragm_vec_x / (np.linalg.norm(tmp_fragm_vec_x) + 1e-15)
    tmp_fragm_vec_y = tmp_fragm_vec_y / (np.linalg.norm(tmp_fragm_vec_y) + 1e-15)
    tmp_fragm_vec_z = tmp_fragm_vec_z / (np.linalg.norm(tmp_fragm_vec_z) + 1e-15)
    tmp_fragm_rot_vec_x = tmp_fragm_rot_vec_x / (np.linalg.norm(tmp_fragm_rot_vec_x) + 1e-15)
    tmp_fragm_rot_vec_y = tmp_fragm_rot_vec_y / (np.linalg.norm(tmp_fragm_rot_vec_y) + 1e-15)
    tmp_fragm_rot_vec_z = tmp_fragm_rot_vec_z / (np.linalg.norm(tmp_fragm_rot_vec_z) + 1e-15)
    
    hessian_proj = project_optional_vector_for_hess(hessian, tmp_fragm_vec_x)
    hessian_proj = project_optional_vector_for_hess(hessian_proj, tmp_fragm_vec_y)
    hessian_proj = project_optional_vector_for_hess(hessian_proj, tmp_fragm_vec_z)
    hessian_proj = project_optional_vector_for_hess(hessian_proj, tmp_fragm_rot_vec_x)
    hessian_proj = project_optional_vector_for_hess(hessian_proj, tmp_fragm_rot_vec_y)
    hessian_proj = project_optional_vector_for_hess(hessian_proj, tmp_fragm_rot_vec_z)
    
    return hessian_proj # ndarray (3*natoms, 3*natoms)


def project_fragm_bend_vector_for_grad(gradient, geom_num_list, fragm_1, fragm_2, fragm_3):
    natom = len(geom_num_list)
    fragm_1_center = calc_partial_center(geom_num_list, fragm_1)
    fragm_2_center = calc_partial_center(geom_num_list, fragm_2)
    fragm_3_center = calc_partial_center(geom_num_list, fragm_3)
    #  3
    # / \
    #1   2
    vector_31 = fragm_1_center - fragm_3_center
    vector_32 = fragm_2_center - fragm_3_center
    norm_vector_31 = np.linalg.norm(vector_31)
    norm_vector_32 = np.linalg.norm(vector_32)
    unit_vector_31 = vector_31 / (norm_vector_31)
    unit_vector_32 = vector_32 / (norm_vector_32)
    theta = calc_angle_from_vec(unit_vector_31, unit_vector_32)
    
    center_1_bend_vec = (np.cos(theta) * unit_vector_31 - 1 * unit_vector_32) / (norm_vector_31 * np.sin(theta) + 1e-15)
    center_2_bend_vec = (np.cos(theta) * unit_vector_32 - 1 * unit_vector_31) / (norm_vector_32 * np.sin(theta) + 1e-15)
    center_3_bend_vec = ((norm_vector_31 -1 * norm_vector_32 * np.cos(theta)) * unit_vector_31 + (norm_vector_32 -1 * norm_vector_31 * np.cos(theta)) * unit_vector_32) / (norm_vector_31 * norm_vector_32 * np.sin(theta) + 1e-15)
    
    center_1_bend_vec /= (np.linalg.norm(center_1_bend_vec))
    center_2_bend_vec /= (np.linalg.norm(center_2_bend_vec))
    center_3_bend_vec /= (np.linalg.norm(center_3_bend_vec))
    
    center_1_bend_vec_x = np.array([center_1_bend_vec[0], 0.0, 0.0], dtype="float64")
    center_1_bend_vec_y = np.array([0.0, center_1_bend_vec[1], 0.0], dtype="float64")
    center_1_bend_vec_z = np.array([0.0, 0.0, center_1_bend_vec[2]], dtype="float64")
    
    center_2_bend_vec_x = np.array([center_2_bend_vec[0], 0.0, 0.0], dtype="float64")
    center_2_bend_vec_y = np.array([0.0, center_2_bend_vec[1], 0.0], dtype="float64")
    center_2_bend_vec_z = np.array([0.0, 0.0, center_2_bend_vec[2]], dtype="float64")
    
    center_3_bend_vec_x = np.array([center_3_bend_vec[0], 0.0, 0.0], dtype="float64")
    center_3_bend_vec_y = np.array([0.0, center_3_bend_vec[1], 0.0], dtype="float64")
    center_3_bend_vec_z = np.array([0.0, 0.0, center_3_bend_vec[2]], dtype="float64")
    
    tmp_fragm_vec_x_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_1 = np.array([], dtype="float64")
    
    tmp_fragm_vec_x_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_2 = np.array([], dtype="float64")
    
    tmp_fragm_vec_x_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_3 = np.array([], dtype="float64")
    
    for i in range(len(geom_num_list)):
        if i + 1 in fragm_1:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, center_1_bend_vec_x, axis=0)
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, center_1_bend_vec_y, axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, center_1_bend_vec_z, axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0]),  axis=0)
            
        elif i + 1 in fragm_2:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0]), axis=0)
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0]), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0]), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, center_2_bend_vec_x,  axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, center_2_bend_vec_y,  axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, center_2_bend_vec_z,  axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0]),  axis=0)
        
        elif i + 1 in fragm_3:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0]), axis=0)
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0]), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0]), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0]),  axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, center_3_bend_vec_x,  axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, center_3_bend_vec_y,  axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, center_3_bend_vec_z,  axis=0)
        else:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)            
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)    
    
    tmp_fragm_vec_x_center_1 = tmp_fragm_vec_x_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_1 = tmp_fragm_vec_y_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_1 = tmp_fragm_vec_z_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_2 = tmp_fragm_vec_x_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_2 = tmp_fragm_vec_y_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_2 = tmp_fragm_vec_z_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_3 = tmp_fragm_vec_x_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_3 = tmp_fragm_vec_y_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_3 = tmp_fragm_vec_z_center_3.reshape(3*natom, 1)
    
    tmp_fragm_vec_x_center_1 = tmp_fragm_vec_x_center_1 / (np.linalg.norm(tmp_fragm_vec_x_center_1) + 1e-15)
    tmp_fragm_vec_y_center_1 = tmp_fragm_vec_y_center_1 / (np.linalg.norm(tmp_fragm_vec_y_center_1) + 1e-15)
    tmp_fragm_vec_z_center_1 = tmp_fragm_vec_z_center_1 / (np.linalg.norm(tmp_fragm_vec_z_center_1) + 1e-15)
    tmp_fragm_vec_x_center_2 = tmp_fragm_vec_x_center_2 / (np.linalg.norm(tmp_fragm_vec_x_center_2) + 1e-15)
    tmp_fragm_vec_y_center_2 = tmp_fragm_vec_y_center_2 / (np.linalg.norm(tmp_fragm_vec_y_center_2) + 1e-15)
    tmp_fragm_vec_z_center_2 = tmp_fragm_vec_z_center_2 / (np.linalg.norm(tmp_fragm_vec_z_center_2) + 1e-15)
    tmp_fragm_vec_x_center_3 = tmp_fragm_vec_x_center_3 / (np.linalg.norm(tmp_fragm_vec_x_center_3) + 1e-15)
    tmp_fragm_vec_y_center_3 = tmp_fragm_vec_y_center_3 / (np.linalg.norm(tmp_fragm_vec_y_center_3) + 1e-15)
    tmp_fragm_vec_z_center_3 = tmp_fragm_vec_z_center_3 / (np.linalg.norm(tmp_fragm_vec_z_center_3) + 1e-15)
    
    
    tmp_gradient = gradient.reshape(3*natom, 1)
    
    gradient_proj = project_optional_vector_for_grad(tmp_gradient, tmp_fragm_vec_x_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_3)
    
    gradient_proj = gradient_proj.reshape(natom, 3)
    
    return gradient_proj # ndarray (natoms, 3)

def project_fragm_torsion_vector_for_grad(gradient, geom_num_list, fragm_1, fragm_2, fragm_3, fragm_4):
    natom = len(geom_num_list)
    fragm_1_center = calc_partial_center(geom_num_list, fragm_1)
    fragm_2_center = calc_partial_center(geom_num_list, fragm_2)
    fragm_3_center = calc_partial_center(geom_num_list, fragm_3)
    fragm_4_center = calc_partial_center(geom_num_list, fragm_4)
    
    # 1
    #  \
    #   2---3
    #        \
    #         4
    
    
    vector_12 = fragm_1_center - fragm_2_center
    vector_23 = fragm_2_center - fragm_3_center
    vector_34 = fragm_3_center - fragm_4_center
    
    norm_vector_12 = np.linalg.norm(vector_12)
    norm_vector_23 = np.linalg.norm(vector_23)
    norm_vector_34 = np.linalg.norm(vector_34)
    
    unit_vector_12 = vector_12 / (norm_vector_12 + 1e-15)
    unit_vector_23 = vector_23 / (norm_vector_23 + 1e-15)
    unit_vector_34 = vector_34 / (norm_vector_34 + 1e-15)
    
    bend_2 = calc_angle_from_vec(unit_vector_12, -1 * unit_vector_23)
    bend_3 = calc_angle_from_vec(unit_vector_23, -1 * unit_vector_34)

    center_1_torsion_vec = -1 * np.cross(unit_vector_12, unit_vector_23) / (norm_vector_12 * np.sin(bend_2) + 1e-15)

    center_2_torsion_vec = ((norm_vector_23 -1 * norm_vector_12 * np.cos(bend_2)) / (norm_vector_23 * norm_vector_12 * np.sin(bend_2) + 1e-15) * (np.cross(unit_vector_12, unit_vector_23)/(np.sin(bend_2) + 1e-15))) + (np.cos(bend_3)/(norm_vector_23 * np.sin(bend_3) + 1e-15)) * (np.cross(-1 * unit_vector_34, -1 * unit_vector_23)/(np.sin(bend_3) + 1e-15))
    center_3_torsion_vec = ((norm_vector_23 -1 * norm_vector_34 * np.cos(bend_3)) / (norm_vector_23 * norm_vector_34 * np.sin(bend_3) + 1e-15) * (np.cross(unit_vector_34, unit_vector_23)/(np.sin(bend_3) + 1e-15))) + (np.cos(bend_2)/(norm_vector_23 * np.sin(bend_2) + 1e-15)) * (np.cross(-1 * unit_vector_34, -1 * unit_vector_23)/(np.sin(bend_2) + 1e-15))
    
    center_4_torsion_vec = -1 * np.cross(-1 * unit_vector_34, -1 * unit_vector_23) / (-1 * norm_vector_34 * np.sin(bend_3) + 1e-15)
    center_1_torsion_vec /= (np.linalg.norm(center_1_torsion_vec) + 1e-15)
    center_2_torsion_vec /= (np.linalg.norm(center_2_torsion_vec) + 1e-15)
    center_3_torsion_vec /= (np.linalg.norm(center_3_torsion_vec) + 1e-15)
    center_4_torsion_vec /= (np.linalg.norm(center_4_torsion_vec) + 1e-15)
    
    
    center_1_torsion_vec_x = np.array([center_1_torsion_vec[0], 0.0, 0.0], dtype="float64")
    center_1_torsion_vec_y = np.array([0.0, center_1_torsion_vec[1], 0.0], dtype="float64")
    center_1_torsion_vec_z = np.array([0.0, 0.0, center_1_torsion_vec[2]], dtype="float64")
    
    center_2_torsion_vec_x = np.array([center_2_torsion_vec[0], 0.0, 0.0], dtype="float64")
    center_2_torsion_vec_y = np.array([0.0, center_2_torsion_vec[1], 0.0], dtype="float64")
    center_2_torsion_vec_z = np.array([0.0, 0.0, center_2_torsion_vec[2]], dtype="float64")
    
    center_3_torsion_vec_x = np.array([center_3_torsion_vec[0], 0.0, 0.0], dtype="float64")
    center_3_torsion_vec_y = np.array([0.0, center_3_torsion_vec[1], 0.0], dtype="float64")
    center_3_torsion_vec_z = np.array([0.0, 0.0, center_3_torsion_vec[2]], dtype="float64")
    
    center_4_torsion_vec_x = np.array([center_4_torsion_vec[0], 0.0, 0.0], dtype="float64")
    center_4_torsion_vec_y = np.array([0.0, center_4_torsion_vec[1], 0.0], dtype="float64")
    center_4_torsion_vec_z = np.array([0.0, 0.0, center_4_torsion_vec[2]], dtype="float64")
    
    
    tmp_fragm_vec_x_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_x_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_x_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_x_center_4 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_4 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_4 = np.array([], dtype="float64")
    
    
    for i in range(len(geom_num_list)):
        if i + 1 in fragm_1:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, center_1_torsion_vec_x, axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, center_1_torsion_vec_y, axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, center_1_torsion_vec_z, axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            
        elif i + 1 in fragm_2:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, center_2_torsion_vec_x, axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, center_2_torsion_vec_y, axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, center_2_torsion_vec_z, axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
        
        elif i + 1 in fragm_3:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, center_3_torsion_vec_x, axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, center_3_torsion_vec_y, axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, center_3_torsion_vec_z, axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            
        
        elif i + 1 in fragm_4:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, center_4_torsion_vec_x, axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, center_4_torsion_vec_y, axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, center_4_torsion_vec_z, axis=0)
            
        else:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            
            
            
    tmp_fragm_vec_x_center_1 = tmp_fragm_vec_x_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_1 = tmp_fragm_vec_y_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_1 = tmp_fragm_vec_z_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_2 = tmp_fragm_vec_x_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_2 = tmp_fragm_vec_y_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_2 = tmp_fragm_vec_z_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_3 = tmp_fragm_vec_x_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_3 = tmp_fragm_vec_y_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_3 = tmp_fragm_vec_z_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_4 = tmp_fragm_vec_x_center_4.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_4 = tmp_fragm_vec_y_center_4.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_4 = tmp_fragm_vec_z_center_4.reshape(3*natom, 1)
    
    
    tmp_fragm_vec_x_center_1 = tmp_fragm_vec_x_center_1 / (np.linalg.norm(tmp_fragm_vec_x_center_1) + 1e-15)
    tmp_fragm_vec_y_center_1 = tmp_fragm_vec_y_center_1 / (np.linalg.norm(tmp_fragm_vec_y_center_1) + 1e-15)
    tmp_fragm_vec_z_center_1 = tmp_fragm_vec_z_center_1 / (np.linalg.norm(tmp_fragm_vec_z_center_1) + 1e-15)
    tmp_fragm_vec_x_center_2 = tmp_fragm_vec_x_center_2 / (np.linalg.norm(tmp_fragm_vec_x_center_2) + 1e-15)
    tmp_fragm_vec_y_center_2 = tmp_fragm_vec_y_center_2 / (np.linalg.norm(tmp_fragm_vec_y_center_2) + 1e-15)
    tmp_fragm_vec_z_center_2 = tmp_fragm_vec_z_center_2 / (np.linalg.norm(tmp_fragm_vec_z_center_2) + 1e-15)
    tmp_fragm_vec_x_center_3 = tmp_fragm_vec_x_center_3 / (np.linalg.norm(tmp_fragm_vec_x_center_3) + 1e-15)
    tmp_fragm_vec_y_center_3 = tmp_fragm_vec_y_center_3 / (np.linalg.norm(tmp_fragm_vec_y_center_3) + 1e-15)
    tmp_fragm_vec_z_center_3 = tmp_fragm_vec_z_center_3 / (np.linalg.norm(tmp_fragm_vec_z_center_3) + 1e-15)
    tmp_fragm_vec_x_center_4 = tmp_fragm_vec_x_center_4 / (np.linalg.norm(tmp_fragm_vec_x_center_4) + 1e-15)
    tmp_fragm_vec_y_center_4 = tmp_fragm_vec_y_center_4 / (np.linalg.norm(tmp_fragm_vec_y_center_4) + 1e-15)
    tmp_fragm_vec_z_center_4 = tmp_fragm_vec_z_center_4 / (np.linalg.norm(tmp_fragm_vec_z_center_4) + 1e-15)
    
    
    
    tmp_gradient = gradient.reshape(3*natom, 1)
    
    gradient_proj = project_optional_vector_for_grad(tmp_gradient, tmp_fragm_vec_x_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_4)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_4)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_4)
    gradient_proj = gradient_proj.reshape(natom, 3)
    return gradient_proj # ndarray (natoms, 3)

def project_fragm_outofplain_vector_for_grad(gradient, geom_num_list, fragm_1, fragm_2, fragm_3, fragm_4):
    natom = len(geom_num_list)
    fragm_1_center = calc_partial_center(geom_num_list, fragm_1)
    fragm_2_center = calc_partial_center(geom_num_list, fragm_2)
    fragm_3_center = calc_partial_center(geom_num_list, fragm_3)
    fragm_4_center = calc_partial_center(geom_num_list, fragm_4)
    #   3
    #   |
    #1--4--2
    
    vector_41 = fragm_4_center - fragm_1_center
    vector_42 = fragm_4_center - fragm_2_center
    vector_43 = fragm_4_center - fragm_3_center
    
    norm_vector_41 = np.linalg.norm(vector_41)
    norm_vector_42 = np.linalg.norm(vector_42)
    norm_vector_43 = np.linalg.norm(vector_43)
    
    unit_vector_41 = vector_41 / (norm_vector_41)
    unit_vector_42 = vector_42 / (norm_vector_42)
    unit_vector_43 = vector_43 / (norm_vector_43)
    
    bend_1 = calc_angle_from_vec(unit_vector_43, unit_vector_42)
    bend_2 = calc_angle_from_vec(unit_vector_41, unit_vector_43)
    bend_3 = calc_angle_from_vec(unit_vector_42, unit_vector_41)
    
    outofplain_angle = np.abs(np.pi/2 - calc_outofplain_angle_from_vec(unit_vector_42, unit_vector_43, unit_vector_41))
    
    center_1_outofplain_vec = (1.0 / norm_vector_41) * (np.cross(unit_vector_42, unit_vector_43)/(1e-15 + np.cos(outofplain_angle) * np.sin(bend_1))) -1 * (np.tan(outofplain_angle) * unit_vector_41)
    center_2_outofplain_vec = (1.0 / norm_vector_42) * (np.cross(unit_vector_43, unit_vector_41)/(1e-15 + np.cos(outofplain_angle) * np.sin(bend_2))) -1 * (np.tan(outofplain_angle) / (1e-15 + np.sin(bend_1) ** 2)) * (unit_vector_42 -1 * np.cos(bend_1) * unit_vector_43)
    
    center_3_outofplain_vec = (1.0 / norm_vector_43) * (np.cross(unit_vector_41, unit_vector_42)/(1e-15 + np.cos(outofplain_angle) * np.sin(bend_3))) -1 * (np.tan(outofplain_angle) / (1e-15 + np.sin(bend_1) ** 2)) * (unit_vector_43 -1 * np.cos(bend_1) * unit_vector_42)
    
    center_4_outofplain_vec = -1 * (center_1_outofplain_vec + center_2_outofplain_vec + center_3_outofplain_vec)
    
    
    center_1_outofplain_vec /= (np.linalg.norm(center_1_outofplain_vec))
    center_2_outofplain_vec /= (np.linalg.norm(center_2_outofplain_vec))
    center_3_outofplain_vec /= (np.linalg.norm(center_3_outofplain_vec))
    center_4_outofplain_vec /= (np.linalg.norm(center_4_outofplain_vec))
    
    center_1_outofplain_vec_x = np.array([center_1_outofplain_vec[0], 0.0, 0.0], dtype="float64")
    center_1_outofplain_vec_y = np.array([0.0, center_1_outofplain_vec[1], 0.0], dtype="float64")
    center_1_outofplain_vec_z = np.array([0.0, 0.0, center_1_outofplain_vec[2]], dtype="float64")
    
    center_2_outofplain_vec_x = np.array([center_2_outofplain_vec[0], 0.0, 0.0], dtype="float64")
    center_2_outofplain_vec_y = np.array([0.0, center_2_outofplain_vec[1], 0.0], dtype="float64")
    center_2_outofplain_vec_z = np.array([0.0, 0.0, center_2_outofplain_vec[2]], dtype="float64")
    
    center_3_outofplain_vec_x = np.array([center_3_outofplain_vec[0], 0.0, 0.0], dtype="float64")
    center_3_outofplain_vec_y = np.array([0.0, center_3_outofplain_vec[1], 0.0], dtype="float64")
    center_3_outofplain_vec_z = np.array([0.0, 0.0, center_3_outofplain_vec[2]], dtype="float64")
    
    center_4_outofplain_vec_x = np.array([center_4_outofplain_vec[0], 0.0, 0.0], dtype="float64")
    center_4_outofplain_vec_y = np.array([0.0, center_4_outofplain_vec[1], 0.0], dtype="float64")
    center_4_outofplain_vec_z = np.array([0.0, 0.0, center_4_outofplain_vec[2]], dtype="float64")
    
    
    tmp_fragm_vec_x_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_1 = np.array([], dtype="float64")
    tmp_fragm_vec_x_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_2 = np.array([], dtype="float64")
    tmp_fragm_vec_x_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_3 = np.array([], dtype="float64")
    tmp_fragm_vec_x_center_4 = np.array([], dtype="float64")
    tmp_fragm_vec_y_center_4 = np.array([], dtype="float64")
    tmp_fragm_vec_z_center_4 = np.array([], dtype="float64")
    
    
    for i in range(len(geom_num_list)):
        if i + 1 in fragm_1:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, center_1_outofplain_vec_x, axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, center_1_outofplain_vec_y, axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, center_1_outofplain_vec_z, axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            
        elif i + 1 in fragm_2:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, center_2_outofplain_vec_x, axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, center_2_outofplain_vec_y, axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, center_2_outofplain_vec_z, axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
        
        elif i + 1 in fragm_3:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, center_3_outofplain_vec_x, axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, center_3_outofplain_vec_y, axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, center_3_outofplain_vec_z, axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            
        
        elif i + 1 in fragm_4:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, center_4_outofplain_vec_x, axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, center_4_outofplain_vec_y, axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, center_4_outofplain_vec_z, axis=0)
            
        else:
            tmp_fragm_vec_x_center_1 = np.append(tmp_fragm_vec_x_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0) 
            tmp_fragm_vec_y_center_1 = np.append(tmp_fragm_vec_y_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_1 = np.append(tmp_fragm_vec_z_center_1, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_2 = np.append(tmp_fragm_vec_x_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_2 = np.append(tmp_fragm_vec_y_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_2 = np.append(tmp_fragm_vec_z_center_2, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_3 = np.append(tmp_fragm_vec_x_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_3 = np.append(tmp_fragm_vec_y_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_3 = np.append(tmp_fragm_vec_z_center_3, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_x_center_4 = np.append(tmp_fragm_vec_x_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_y_center_4 = np.append(tmp_fragm_vec_y_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            tmp_fragm_vec_z_center_4 = np.append(tmp_fragm_vec_z_center_4, np.array([0.0, 0.0, 0.0], dtype="float64"), axis=0)
            
            
            
    tmp_fragm_vec_x_center_1 = tmp_fragm_vec_x_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_1 = tmp_fragm_vec_y_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_1 = tmp_fragm_vec_z_center_1.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_2 = tmp_fragm_vec_x_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_2 = tmp_fragm_vec_y_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_2 = tmp_fragm_vec_z_center_2.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_3 = tmp_fragm_vec_x_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_3 = tmp_fragm_vec_y_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_3 = tmp_fragm_vec_z_center_3.reshape(3*natom, 1)
    tmp_fragm_vec_x_center_4 = tmp_fragm_vec_x_center_4.reshape(3*natom, 1)
    tmp_fragm_vec_y_center_4 = tmp_fragm_vec_y_center_4.reshape(3*natom, 1)
    tmp_fragm_vec_z_center_4 = tmp_fragm_vec_z_center_4.reshape(3*natom, 1)
    
    
    tmp_fragm_vec_x_center_1 = tmp_fragm_vec_x_center_1 / (np.linalg.norm(tmp_fragm_vec_x_center_1) + 1e-15)
    tmp_fragm_vec_y_center_1 = tmp_fragm_vec_y_center_1 / (np.linalg.norm(tmp_fragm_vec_y_center_1) + 1e-15)
    tmp_fragm_vec_z_center_1 = tmp_fragm_vec_z_center_1 / (np.linalg.norm(tmp_fragm_vec_z_center_1) + 1e-15)
    tmp_fragm_vec_x_center_2 = tmp_fragm_vec_x_center_2 / (np.linalg.norm(tmp_fragm_vec_x_center_2) + 1e-15)
    tmp_fragm_vec_y_center_2 = tmp_fragm_vec_y_center_2 / (np.linalg.norm(tmp_fragm_vec_y_center_2) + 1e-15)
    tmp_fragm_vec_z_center_2 = tmp_fragm_vec_z_center_2 / (np.linalg.norm(tmp_fragm_vec_z_center_2) + 1e-15)
    tmp_fragm_vec_x_center_3 = tmp_fragm_vec_x_center_3 / (np.linalg.norm(tmp_fragm_vec_x_center_3) + 1e-15)
    tmp_fragm_vec_y_center_3 = tmp_fragm_vec_y_center_3 / (np.linalg.norm(tmp_fragm_vec_y_center_3) + 1e-15)
    tmp_fragm_vec_z_center_3 = tmp_fragm_vec_z_center_3 / (np.linalg.norm(tmp_fragm_vec_z_center_3) + 1e-15)
    tmp_fragm_vec_x_center_4 = tmp_fragm_vec_x_center_4 / (np.linalg.norm(tmp_fragm_vec_x_center_4) + 1e-15)
    tmp_fragm_vec_y_center_4 = tmp_fragm_vec_y_center_4 / (np.linalg.norm(tmp_fragm_vec_y_center_4) + 1e-15)
    tmp_fragm_vec_z_center_4 = tmp_fragm_vec_z_center_4 / (np.linalg.norm(tmp_fragm_vec_z_center_4) + 1e-15)
    
    
    tmp_gradient = gradient.reshape(3*natom, 1)
    
    gradient_proj = project_optional_vector_for_grad(tmp_gradient, tmp_fragm_vec_x_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_1)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_2)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_3)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_x_center_4)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_y_center_4)
    gradient_proj = project_optional_vector_for_grad(gradient_proj, tmp_fragm_vec_z_center_4)
    gradient_proj = gradient_proj.reshape(natom, 3)
    return gradient_proj # ndarray (natoms, 3)


def move_atom_distance_one_side(geom_num_list, atom1, atom2, distance):
    vec = geom_num_list[atom2] - geom_num_list[atom1]
    norm_vec = np.linalg.norm(vec)
    unit_vec = vec / norm_vec
    geom_num_list[atom2] = geom_num_list[atom2] + distance * unit_vec
    return geom_num_list
    
def change_atom_distance_both_side(geom_num_list, atom1, atom2, distance):
    vec = geom_num_list[atom2] - geom_num_list[atom1]
    norm_vec = np.linalg.norm(vec)
    unit_vec = vec / norm_vec
    dist_diff = distance - norm_vec

    geom_num_list[atom1] = geom_num_list[atom1] - dist_diff * unit_vec * 0.5
    geom_num_list[atom2] = geom_num_list[atom2] + dist_diff * unit_vec * 0.5
    return geom_num_list



def calc_bond_matrix(geom_num_list, element_list, threshold=1.2):
    bond_matrix = np.zeros((len(geom_num_list), len(geom_num_list)))
    
    for i in range(len(element_list)):
        for j in range(i+1, len(element_list)):
            r = np.linalg.norm(geom_num_list[i], geom_num_list[j])
            r_cov = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
            if r < threshold * r_cov:
                bond_matrix[i][j] = 1
                bond_matrix[j][i] = 1
    
    return bond_matrix






def calc_RMS(data):
    return np.sqrt(np.mean(data ** 2))


def torch_rotate_around_axis(theta, axis='z'):
    cos_theta = torch.cos(theta).reshape(1)
    sin_theta = torch.sin(theta).reshape(1)
    tensor_zero = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    tensor_one = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    if axis == 'x':
       
        R = torch.stack([torch.cat([tensor_one, tensor_zero, tensor_zero]),
                          torch.cat([tensor_zero, cos_theta, -sin_theta]),
                          torch.cat([tensor_zero, sin_theta, cos_theta])])
    elif axis == 'y':
       
        R = torch.stack([torch.cat([cos_theta, tensor_zero, sin_theta]),
                          torch.cat([tensor_zero, tensor_one, tensor_zero]),
                          torch.cat([-sin_theta, tensor_zero, cos_theta])])
    elif axis == 'z':
       
        R = torch.stack([torch.cat([cos_theta, -sin_theta, tensor_zero]),
                          torch.cat([sin_theta, cos_theta, tensor_zero]),
                          torch.cat([tensor_zero, tensor_zero, tensor_one])])
    else:
        raise ValueError

    return R


def torch_align_vector_with_z(v):
    v = v / torch.linalg.norm(v) 
    z = torch.tensor([0.0, 0.0, 1.0], dtype=v.dtype, requires_grad=True) 
    tensor_zero = torch.tensor([0.0], dtype=v.dtype, requires_grad=True)
    axis = torch.linalg.cross(v, z)
    axis_len = torch.linalg.norm(axis)

    cos_theta = torch.dot(v, z)
    sin_theta = axis_len

    axis = axis / axis_len 
    axis = axis.reshape(3, 1)
    K = torch.stack([
        torch.cat([tensor_zero, -axis[2], axis[1]]),
        torch.cat([axis[2], tensor_zero, -axis[0]]),
        torch.cat([-axis[1], axis[0], tensor_zero])
    ])

    R = torch.eye(3, dtype=v.dtype, requires_grad=True) + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)
    return R







    