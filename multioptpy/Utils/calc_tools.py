import itertools
import numpy as np
import copy
from scipy.spatial import cKDTree
from scipy.interpolate import make_interp_spline

from multioptpy.Interpolation.interpolation import spline_interpolation
from multioptpy.Potential.idpp import IDPP
from multioptpy.Parameters.parameter import covalent_radii_lib, atomic_mass, UnitValueLib

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
    
    def project_out_hess_tr_and_rot(self, hessian, element_list, geometry, display_eigval=True):#covert coordination to mass-weighted coordination
        natoms = len(element_list)
        
        # Move to center of mass
        geometry = geometry - self.calc_center_of_mass(geometry, element_list)
        
        # Calculate atomic masses
        elem_mass = np.array([atomic_mass(elem) for elem in element_list], dtype="float64")
        
        # Mass-weighting matrices
        m_sqrt = np.repeat(elem_mass, 3) ** 0.5
        M_minus_sqrt = np.diag(np.repeat(elem_mass, 3) ** (-0.5))
        
        # Convert to mass-weighted Hessian
        mw_hessian = np.dot(np.dot(M_minus_sqrt, hessian), M_minus_sqrt)
        
        # Initialize arrays for translation and rotation vectors
        tr_vectors = np.zeros((3, 3 * natoms))
        rot_vectors = np.zeros((3, 3 * natoms))
        
        # Create mass-weighted translation vectors
        for i in range(3):
            tr_vectors[i, i::3] = m_sqrt[i::3]
        
        # Create mass-weighted rotation vectors
        for atom in range(natoms):
            x, y, z = geometry[atom]
            mass_sqrt = m_sqrt[3*atom]
            
            # Rotation around x-axis: (0, -z, y)
            rot_vectors[0, 3*atom:3*atom+3] = np.array([0.0, -z, y]) * mass_sqrt
            
            # Rotation around y-axis: (z, 0, -x)
            rot_vectors[1, 3*atom:3*atom+3] = np.array([z, 0.0, -x]) * mass_sqrt
            
            # Rotation around z-axis: (-y, x, 0)
            rot_vectors[2, 3*atom:3*atom+3] = np.array([-y, x, 0.0]) * mass_sqrt
        
        # Combine translation and rotation vectors
        TR_vectors = np.vstack([tr_vectors, rot_vectors])
        
        # Gram-Schmidt orthonormalization with improved numerical stability
        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v.copy()
                for b in basis:
                    w -= np.dot(v, b) * b
                norm = np.linalg.norm(w)
                if norm > 1e-10:  # Threshold for linear independence
                    basis.append(w / norm)
            return np.array(basis)
        
        # Orthonormalize the translation and rotation vectors
        TR_vectors = gram_schmidt(TR_vectors)
        
        # Calculate projection matrix
        P = np.eye(3 * natoms)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)
        
        # Project the mass-weighted Hessian
        mw_hess_proj = np.dot(np.dot(P.T, mw_hessian), P)
        
        # Ensure symmetry (numerical stability)
        mw_hess_proj = (mw_hess_proj + mw_hess_proj.T) / 2
        
        if display_eigval:
            eigenvalues, _ = np.linalg.eigh(mw_hess_proj)
            eigenvalues = np.sort(eigenvalues)
            # Stricter threshold for eigenvalue filtering
            idx_eigenvalues = np.where(np.abs(eigenvalues) > 1e-7)[0]
            print(f"EIGENVALUES (MASS-WEIGHTED COORDINATE, NUMBER OF VALUES: {len(idx_eigenvalues)}):")
            for i in range(0, len(idx_eigenvalues), 6):
                tmp_arr = eigenvalues[idx_eigenvalues[i:i+6]]
                print(" ".join(f"{val:12.8f}" for val in tmp_arr))
        
        return mw_hess_proj

    def project_out_hess_tr_and_rot_for_coord(self, hessian, element_list, geometry, display_eigval=True):#do not consider atomic mass
        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v.copy()
                for b in basis:
                    w -= np.dot(v, b) * b
                norm = np.linalg.norm(w)
                if norm > 1e-10:
                    basis.append(w / norm)
            return np.array(basis)
        
        natoms = len(element_list)
        # Center the geometry
        geometry = geometry - self.calc_center(geometry, element_list)
        
        # Initialize arrays for translation and rotation vectors
        tr_vectors = np.zeros((3, 3 * natoms))
        rot_vectors = np.zeros((3, 3 * natoms))
        
        # Create translation vectors (mass-weighted normalization is not used as specified)
        for i in range(3):
            tr_vectors[i, i::3] = 1.0
        
        # Create rotation vectors
        for atom in range(natoms):
            # Get atom coordinates
            x, y, z = geometry[atom]
            
            # Rotation around x-axis: (0, -z, y)
            rot_vectors[0, 3*atom:3*atom+3] = np.array([0.0, -z, y])
            
            # Rotation around y-axis: (z, 0, -x)
            rot_vectors[1, 3*atom:3*atom+3] = np.array([z, 0.0, -x])
            
            # Rotation around z-axis: (-y, x, 0)
            rot_vectors[2, 3*atom:3*atom+3] = np.array([-y, x, 0.0])

        # Combine translation and rotation vectors
        TR_vectors = np.vstack([tr_vectors, rot_vectors])
        

        
        # Orthonormalize the translation and rotation vectors
        TR_vectors = gram_schmidt(TR_vectors)
        
        # Calculate projection matrix
        P = np.eye(3 * natoms)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)
        
        # Project the Hessian
        hess_proj = np.dot(np.dot(P.T, hessian), P)
        
        # Make the projected Hessian symmetric (numerical stability)
        hess_proj = (hess_proj + hess_proj.T) / 2
        
        if display_eigval:
            eigenvalues, _ = np.linalg.eigh(hess_proj)
            eigenvalues = np.sort(eigenvalues)
            # Filter out near-zero eigenvalues
            idx_eigenvalues = np.where(np.abs(eigenvalues) > 1e-10)[0]
            print(f"EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: {len(idx_eigenvalues)}):")
            for i in range(0, len(idx_eigenvalues), 6):
                tmp_arr = eigenvalues[idx_eigenvalues[i:i+6]]
                print(" ".join(f"{val:12.8f}" for val in tmp_arr))
        
        return hess_proj

    def check_atom_connectivity(self, mol_list, element_list, atom_num, covalent_radii_threshold_scale=1.2):#mol_list:ang.
        # Convert molecular coordinates to numpy array
        coords = np.array(mol_list, dtype=np.float64)
        
        # Build KD-Tree for efficient nearest neighbor searches
        kdtree = cKDTree(coords)
        
        # Initialize arrays for tracking
        n_atoms = len(mol_list)
        connected_atoms = [atom_num]
        searched_atoms = []
        
        # Pre-calculate covalent radii for each element to avoid repeated lookups
        cov_radii = [covalent_radii_lib(element) * UnitValueLib().bohr2angstroms for element in element_list]
        
        while True:
            search_progress = False
            for i in connected_atoms:
                if i in searched_atoms:
                    continue
                
                # Calculate max possible bond distance for this atom
                # This is a conservative estimate to limit initial search radius
                max_cov_radius = max([cov_radii[i] + cov_radii[j] for j in range(n_atoms)]) * covalent_radii_threshold_scale
                
                # Query the KD-Tree for potential neighbors within the max bond distance
                potential_neighbors = kdtree.query_ball_point(coords[i], max_cov_radius)
                
                # Check each potential neighbor more precisely
                for j in potential_neighbors:
                    if j == i or j in connected_atoms:
                        continue
                    
                    # Calculate exact threshold for this specific pair
                    covalent_dist_threshold = covalent_radii_threshold_scale * (cov_radii[i] + cov_radii[j])
                    
                    # Calculate distance
                    dist = np.linalg.norm(coords[i] - coords[j])
                    
                    if dist < covalent_dist_threshold:
                        connected_atoms.append(j)
                        search_progress = True
                
                searched_atoms.append(i)
                search_progress = True
            
            if not search_progress or len(connected_atoms) == len(searched_atoms):
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
    magnitude1 = torch.linalg.norm(vector1)
    if torch.abs(magnitude1) < 1e-15:
        magnitude1 = magnitude1 + 1e-15
    magnitude2 = torch.linalg.norm(vector2)
    if torch.abs(magnitude2) < 1e-15:
        magnitude2 = magnitude2 + 1e-15

    dot_product = torch.matmul(vector1, vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = torch.arccos(cos_theta)
    return theta

def calc_angle_from_vec(vector1, vector2):
    magnitude1 = np.linalg.norm(vector1)
    if np.abs(magnitude1) < 1e-15:
        magnitude1 += 1e-15
    magnitude2 = np.linalg.norm(vector2)
    if np.abs(magnitude2) < 1e-15:
        magnitude2 += 1e-15
    dot_product = np.matmul(vector1, vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = np.arccos(cos_theta)
    return theta

def torch_calc_dihedral_angle_from_vec(vector1, vector2, vector3):
    v1 = torch.linalg.cross(vector1, vector2)
    v2 = torch.linalg.cross(vector2, vector3)
    norm_v1 = torch.linalg.norm(v1)
    if torch.abs(norm_v1) < 1e-15:
        norm_v1 = norm_v1 + 1e-15
    norm_v2 = torch.linalg.norm(v2)
    if torch.abs(norm_v2) < 1e-15:
        norm_v2 = norm_v2 + 1e-15
 
    cos_theta = torch.sum(v1*v2) / (norm_v1 * norm_v2)
    angle = torch.arccos(cos_theta)
    sign = torch.sign(torch.sum(torch.linalg.cross(v1 / norm_v1, v2 / norm_v2) * vector2))
    if sign != 0:
        angle = -1 * angle * sign
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
    norm_v1 = np.linalg.norm(v1)
    if np.abs(norm_v1) < 1e-15:
        norm_v1 += 1e-15 
    norm_v2 = np.linalg.norm(v2)
    if np.abs(norm_v2) < 1e-15:
        norm_v2 += 1e-15
    
    cos_theta = np.sum(v1*v2) / (norm_v1 * norm_v2)
    angle = np.abs(np.arccos(cos_theta))
    sign = np.sign(np.dot(np.cross(v1 / norm_v1, v2 / norm_v2), vector2))
    if sign != 0:
        angle = -1 * angle * sign
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
    magnitude1 = torch.linalg.norm(v1)
    if torch.abs(magnitude1) < 1e-15:
        magnitude1 = magnitude1 + 1e-15
    magnitude2 = torch.linalg.norm(vector3)
    if torch.abs(magnitude2) < 1e-15:
        magnitude2 = magnitude2 + 1e-15
    
    dot_product = torch.matmul(v1, vector3)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle = torch.arccos(cos_theta)
    return angle

def calc_outofplain_angle_from_vec(vector1, vector2, vector3):
    v1 = np.cross(vector1, vector2)
    
    magnitude1 = np.linalg.norm(v1)
    if np.abs(magnitude1) < 1e-15:
        magnitude1 += 1e-15
    magnitude2 = np.linalg.norm(vector3)
    if np.abs(magnitude2) < 1e-15:
        magnitude2 += 1e-15
    
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

def change_fragm_distance_both_side(geom_num_list, fragm1, fragm2, distance):
    center_1 = np.array([0.0, 0.0, 0.0], dtype="float64")
    for i in fragm1:
        center_1 += geom_num_list[i]
    center_1 /= len(fragm1)
    
    center_2 = np.array([0.0, 0.0, 0.0], dtype="float64")
    for i in fragm2:
        center_2 += geom_num_list[i]
    center_2 /= len(fragm2)
    
    vec = center_2 - center_1
    norm_vec = np.linalg.norm(vec)
    unit_vec = vec / norm_vec
    dist_diff = distance - norm_vec
    
    for i in fragm1:
        geom_num_list[i] = geom_num_list[i] - dist_diff * unit_vec * 0.5
    
    for i in fragm2:
        geom_num_list[i] = geom_num_list[i] + dist_diff * unit_vec * 0.5
        
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



def calc_path_length_list(geometry_list):
    """Calculate path length list for geometry distribution"""
    path_length_list = [0.0]
    for i in range(len(geometry_list)-1):
        tmp_geometry_list_j = geometry_list[i+1] - np.mean(geometry_list[i+1], axis=0)
        tmp_geometry_list_i = geometry_list[i] - np.mean(geometry_list[i], axis=0)
        
        path_length = path_length_list[-1] + np.linalg.norm(tmp_geometry_list_j - tmp_geometry_list_i)
        path_length_list.append(path_length)
    return path_length_list


def apply_climbing_image(geometry_list, energy_list, element_list):
    """Apply climbing image method to locate transition states"""
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    local_maxima, local_minima = spline_interpolation(path_length_list, energy_list)
    print(local_maxima)
    
    for distance, energy in local_maxima:
        print("Local maximum at distance: ", distance)
        for i in range(2, len(path_length_list)-2):
            if path_length_list[i] >= distance or distance >= path_length_list[i+1]:
                continue
            delta_t = (distance - path_length_list[i]) / (path_length_list[i+1] - path_length_list[i])
            tmp_geometry = geometry_list[i] + (geometry_list[i+1] - geometry_list[i]) * delta_t
            tmp_geom_list = [geometry_list[i], tmp_geometry, geometry_list[i+1]]
            idpp_instance = IDPP()
            tmp_geom_list = idpp_instance.opt_path(tmp_geom_list, element_list)
            geometry_list[i] = tmp_geom_list[1]
    return geometry_list


