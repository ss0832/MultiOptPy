import itertools

import numpy as np

from parameter import UFF_VDW_distance_lib, UFF_VDW_well_depth_lib, covalent_radii_lib, element_number, number_element, atomic_mass, UnitValueLib



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
        print(file)
        if len(atom_numbers) > 1:
            for a1, a2 in list(itertools.combinations(atom_numbers,2)):
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
    
    def project_out_hess_tr_and_rot(self, hessian, element_list, geomerty):
        natoms = len(element_list)

        
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

        hess_proj = np.dot(np.dot(P.T, mw_hessian), P)

        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        eigenvalues = eigenvalues[n_tr:]
        eigenvectors = eigenvectors[:, n_tr:]
        print("=== hessian projected out transition and rotation ===")
        print("eigenvalues: ", eigenvalues)
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



    
    