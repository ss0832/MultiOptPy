
import numpy as np

from param import covalent_radii_lib


class BondConnectivity:#Bohr
    def __init__(self):
        self.covalent_radii_lib_func = covalent_radii_lib#function
        self.covalent_radii_threshold = 1.2
        
        return
    
    def distance_matrix(self, coord):#coord:natoms × 3, ndarray (Bohr unit)
        natom = len(coord)
        
        distance_mat = np.zeros((natom, natom), dtype="float64")
        tmp_mat = np.zeros((natom), dtype="float64")
        for i in range(natom):
            tmp_mat = coord - coord[i]
            distance_mat[i] = np.linalg.norm(tmp_mat, axis=1)
            distance_mat[i][i] = 0.0
        return distance_mat
    
    def bondlength_matrix(self, element_list):
        natom = len(element_list)
        bond_length_mat = np.zeros((natom, natom), dtype="float64")
        radii_list = np.array([self.covalent_radii_lib_func(elem) for elem in element_list], dtype="float64")
        for i in range(natom):
            bond_length_mat[i] = (radii_list + radii_list[i]) * self.covalent_radii_threshold
            bond_length_mat[i][i] = -1.0
        return bond_length_mat#Bohr

    def bond_connect_matrix(self, element_list, coord):
        
        distance_mat = self.distance_matrix(coord)
        bond_length_matrix = self.bondlength_matrix(element_list)
        connectivity_mat = np.where(distance_mat <= bond_length_matrix, 1, 0)
                  
        return connectivity_mat


    def bond_connect_table(self, connectivity_mat):
        natom = len(connectivity_mat)

        bond_table = []
        for i in range(natom):
            for j in range(natom):
                if i > j:
                    continue
                if connectivity_mat[i, j] == 1:
                    bond_table.append([i, j])
                    
        return bond_table
    
    def angle_connect_table(self, bond_connect_matrix):
        angle_connect_table = []

        for i in range(len(bond_connect_matrix)):
            for j in range(len(bond_connect_matrix)):
                if bond_connect_matrix[i][j] == 1:
                    for n in range(j+1, len(bond_connect_matrix[i])):
                        if bond_connect_matrix[i][n] == 1 and bond_connect_matrix[j][n] == 0:
                            angle_connect_table.append([j, i, n])
                        
                
        return angle_connect_table
    
    def dihedral_angle_connect_table(self, bond_connect_matrix):
        dihedral_angle_connect_table = []
        angle_connect_table = self.angle_connect_table(bond_connect_matrix)
        for i in range(len(angle_connect_table)):
            for j in range(i+1, len(angle_connect_table)):
                if (angle_connect_table[i][1] == angle_connect_table[j][1] and angle_connect_table[i][2] == angle_connect_table[j][2]) or (angle_connect_table[i][1] == angle_connect_table[j][2] and angle_connect_table[i][2] == angle_connect_table[j][1]):
                    
                    candidate_dac_table = [angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2], angle_connect_table[j][0]]
                    
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[3]] == 1 or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[3]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)
                        continue
                        
                    candidate_dac_table = [angle_connect_table[j][0], angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2]]
                    
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[0]] == 1  or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[0]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)   
                        continue
                        
                
                if (angle_connect_table[i][1] == angle_connect_table[j][1] and angle_connect_table[i][0] == angle_connect_table[j][0]) or (angle_connect_table[i][1] == angle_connect_table[j][0] and angle_connect_table[i][0] == angle_connect_table[j][1]):
                    
                    candidate_dac_table = [angle_connect_table[j][2], angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2]]
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[0]] == 1  or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[0]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)            
                        continue    
                
                    candidate_dac_table = [angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2], angle_connect_table[j][2]]
                    
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[3]] == 1 or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[3]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)   
                        continue
                
                if (angle_connect_table[i][1] == angle_connect_table[j][0] and angle_connect_table[i][2] == angle_connect_table[j][1]) or (angle_connect_table[i][1] == angle_connect_table[j][1] and angle_connect_table[i][2] == angle_connect_table[j][0]):
                    
                    candidate_dac_table = [angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2], angle_connect_table[j][2]]
                    
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[3]] == 1 or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[3]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)   
                        continue       
                        
                    candidate_dac_table = [angle_connect_table[j][2], angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2]]
                    
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[0]] == 1  or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[0]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)     
                        continue
                
                if (angle_connect_table[i][0] == angle_connect_table[j][1] and angle_connect_table[i][1] == angle_connect_table[j][2]) or (angle_connect_table[i][0] == angle_connect_table[j][2] and angle_connect_table[i][1] == angle_connect_table[j][1]):
                  
                    candidate_dac_table = [angle_connect_table[j][0], angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2]]
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[0]] == 1  or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[0]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)    
                        continue
                        
                    candidate_dac_table = [angle_connect_table[i][0], angle_connect_table[i][1], angle_connect_table[i][2], angle_connect_table[j][0]]
                    if bond_connect_matrix[candidate_dac_table[2]][candidate_dac_table[3]] == 1 or bond_connect_matrix[candidate_dac_table[1]][candidate_dac_table[3]] == 1:
                        dihedral_angle_connect_table.append(candidate_dac_table)              
                        continue
                
        return dihedral_angle_connect_table
    
    def connectivity_table(self, coord, element_list):
        b_C_mat = self.bond_connect_matrix(element_list, coord)
        connectivity_table = [self.bond_connect_table(b_C_mat), self.angle_connect_table(b_C_mat), self.dihedral_angle_connect_table(b_C_mat)]
        
        return connectivity_table
    

if __name__ == "__main__":#test
    BC = BondConnectivity()
    words = ["C     0.184130020000      1.253523500000      0.322496160000"
            ,"H     0.649324370000      0.314864420000      0.540242210000"
            ,"H     0.649339210000      1.534270620000      1.244266980000"
            ,"H     0.540802860000      1.757921690000     -0.551155350000"
            ,"H   -10.075869980000      1.253539030000      0.322496160000"
            ,"H   -10.475869980000      1.253539030000      0.322496160000"
            ,"C    -1.355869979883      1.253542481111      0.322496160000"
            ,"F    -1.850403239571      2.449475576521     -0.063047840000"
            ,"F    -1.850429465316      0.321694576682     -0.520435840000"
            ,"F    -1.850421234650      0.989475576632      1.550971160000"]
    elements = []
    coord = []
    for word in words:
        splited_word = word.split()
       
        elements.append(splited_word[0])
        coord.append(list(map(float, splited_word[1:4])))
    
    coord = np.array(coord)/0.52917721067 #Bohr
    print(BC.distance_matrix(coord))
    print(BC.bondlength_matrix(elements))
    
    b_c_mat = BC.bond_connect_matrix(elements, coord)
    print(b_c_mat)
    print(BC.bond_connect_table(b_c_mat))
    print(BC.angle_connect_table(b_c_mat))
    print(BC.dihedral_angle_connect_table(b_c_mat))
    