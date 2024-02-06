import glob
import itertools
import matplotlib.pyplot as plt

import numpy as np


from calc_tools import Calculationtools #kabsch_algorithm
from param import atomic_mass, UnitValueLib

#ref. Chem. Commun. 2021, 57 (89), 11734–11750.
#https://doi.org/10.1039/D1CC04667E
#CMDS = classical multidimensional scaling 

class CMDSPathAnalysis:
    def __init__(self, directory, energy_list, bias_energy_list):
        self.directory = directory
        energy_list = np.array(energy_list)
        self.energy_list = energy_list - energy_list[0]
        bias_energy_list = np.array(bias_energy_list)
        self.bias_energy_list = bias_energy_list - bias_energy_list[0]
        return
    
    def read_xyz_file(self, struct_path_1, struct_path_2):
        
        with open(struct_path_1, "r") as f:
            words_1 = f.read().splitlines()
        
        with open(struct_path_2, "r") as f:
            words_2 = f.read().splitlines()
        
        mass_weight_coord_1 = []
        
        for word in words_1:
            splited_word = word.split()
            if len(splited_word) != 4:
                continue
            tmp = np.sqrt(atomic_mass(splited_word[0])) * np.array(splited_word[1:4], dtype="float64")
            mass_weight_coord_1.append(tmp)
        mass_weight_coord_2 = []
        for word in words_2:
            splited_word = word.split()
            if len(splited_word) != 4:
                continue
            tmp = np.sqrt(atomic_mass(splited_word[0])) * np.array(splited_word[1:4], dtype="float64")
            mass_weight_coord_2.append(tmp)        
        
        mass_weight_coord_1 = np.array(mass_weight_coord_1, dtype="float64")
        mass_weight_coord_2 = np.array(mass_weight_coord_2, dtype="float64")
        
        return mass_weight_coord_1, mass_weight_coord_2
    
    
    def double_centering(self, distance_matrix, struct_num):
        distance_matrix_2 = distance_matrix * distance_matrix
        length_dist_mat = len(distance_matrix)
        Q_matrix = -0.5 * np.dot((np.eye(length_dist_mat) - (1 / struct_num) * np.ones((length_dist_mat, length_dist_mat))), np.dot(distance_matrix_2, (np.eye(length_dist_mat) - (1 / struct_num) * np.ones((length_dist_mat, length_dist_mat))).T))
        
        return Q_matrix
    
    
    def cmds_visualization(self, result_list, energy_list, name=""):
        plt.xlabel("PCo1 (ang. / amu^0.5)")#("LTEP $\mathsf{(cm^{–1})}$")
        plt.ylabel("PCo2 (ang. / amu^0.5)")#("He8_wedge (kcal/mol)")
        plt.title("CMDS result ("+name+")")
        
        x_array = np.array(result_list[1])
        y_array = np.array(result_list[2])
        xmin = min(x_array)
        xmax = max(x_array)
        ymin = min(y_array)
        ymax = max(y_array)
        delta_x = xmax - xmin
        delta_y = ymax - ymin
        plt.xlim(xmin-(delta_x/5), xmax+(delta_x/5))
        plt.ylim(ymin-(delta_y/5), ymax+(delta_y/5))
        for i in range(len(energy_list[:-1])):
            data = plt.scatter(x_array[i], y_array[i], c=energy_list[i], vmin=min(energy_list), vmax=max(energy_list), cmap='jet', s=25, marker="o", linewidths=0.1, edgecolors="black")
        plt.colorbar(data, label=name+" (kcal/mol)")
        plt.savefig(self.directory+"cmds_result_visualization_"+str(name)+".png" ,dpi=300,format="png")
        plt.close()
        return
    
    def main(self):
        print("processing CMDS analysis to aprrox. reaction path ...")
        file_list = sorted(glob.glob(self.directory+"samples_*_[0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz"))  
        
        idx_list = itertools.combinations([i for i in range(len(file_list))], 2)
        struct_num = len(file_list)
        D_matrix = np.zeros((struct_num, struct_num))
        
        for i, j in idx_list:
            
            mass_weight_coord_1, mass_weight_coord_2 = self.read_xyz_file(file_list[i], file_list[j])
            modified_coord_1, modified_coord_2 = Calculationtools().kabsch_algorithm(mass_weight_coord_1, mass_weight_coord_2)
            dist = np.linalg.norm(modified_coord_1 - modified_coord_2)
            D_matrix[i][j] = dist
            D_matrix[j][i] = dist
        
        Q_matrix = self.double_centering(D_matrix, struct_num)
        
        Q_eigenvalue, Q_eigenvector = np.linalg.eig(Q_matrix)
        Q_eigenvector = Q_eigenvector.T
        
        Q_eigenvalue = np.real_if_close(Q_eigenvalue, tol=1000)
        Q_eigenvector = np.real_if_close(Q_eigenvector, tol=1000)
        
        sorted_Q_eigenvalue = np.sort(Q_eigenvalue)
        rank_1_value = sorted_Q_eigenvalue[-1]
        rank_2_value = sorted_Q_eigenvalue[-2]
        #print(Q_eigenvalue)
        
        rank1_idx = np.where(Q_eigenvalue == rank_1_value)[0][0]
        rank2_idx = np.where(Q_eigenvalue == rank_2_value)[0][0]
        
        sum_of_eigenvalue = 0.0
        for value in Q_eigenvalue:
            if value < 0:
                continue
            sum_of_eigenvalue += value
        print("dimensional reproducibility:", np.sum(Q_eigenvalue)/sum_of_eigenvalue)
        print("Percentage contribution 1:", Q_eigenvalue[rank1_idx]/sum_of_eigenvalue)
        print("Percentage contribution 2:", Q_eigenvalue[rank2_idx]/sum_of_eigenvalue)
        PCo1 = np.sqrt(Q_eigenvalue[rank1_idx]) * Q_eigenvector[rank1_idx]
        PCo2 = np.sqrt(Q_eigenvalue[rank2_idx]) * Q_eigenvector[rank2_idx]
        
        result_list = []
        
        with open(self.directory+"cmds_analysis_result.csv", "w") as f:
            f.write("itr.,   PCo1,     PCo2,     energy[kcal/mol],    energy(bias)[kcal/mol]\n")
            for i in range(len(PCo1)-1):
                f.write(str(i)+", "+str(float(PCo1[i]))+", "+str(float(PCo2[i]))+","+str(self.energy_list[i])+", "+str(self.bias_energy_list[i])+"\n")
                tmp = [i, float(PCo1[i]), float(PCo2[i])]
                result_list.append(tmp)
        result_list = np.array(result_list, dtype="float64").T
        self.cmds_visualization(result_list, self.energy_list, name="energy")
        self.cmds_visualization(result_list, self.bias_energy_list, name="bias_energy")
        
        return
    