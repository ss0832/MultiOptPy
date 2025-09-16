import glob
import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import atomic_mass
from multioptpy.Coordinate.redundant_coordinate import cartesian_to_z_matrix

### These are toy functions. ###
class PCAPathAnalysis:
    def __init__(self, directory, energy_list, bias_energy_list):
        self.directory = directory
        energy_list = np.array(energy_list)
        self.energy_list = energy_list - energy_list[0]
        bias_energy_list = np.array(bias_energy_list)
        self.bias_energy_list = bias_energy_list - bias_energy_list[0]
        self.modredundant = True

        return
    
    def read_xyz_file(self, struct_path_1, struct_path_2):
        
        with open(struct_path_1, "r") as f:
            words_1 = f.read().splitlines()
        
        with open(struct_path_2, "r") as f:
            words_2 = f.read().splitlines()
        
        mass_weight_coord_1 = []
        element_list = []
        for word in words_1:
            splited_word = word.split()
            if len(splited_word) != 4:
                continue
            tmp = np.sqrt(atomic_mass(splited_word[0])) * np.array(splited_word[1:4], dtype="float64")
            mass_weight_coord_1.append(tmp)
            element_list.append(splited_word[0])
        mass_weight_coord_2 = []
        for word in words_2:
            splited_word = word.split()
            if len(splited_word) != 4:
                continue
            tmp = np.sqrt(atomic_mass(splited_word[0])) * np.array(splited_word[1:4], dtype="float64")
            mass_weight_coord_2.append(tmp)        
        
        mass_weight_coord_1 = np.array(mass_weight_coord_1, dtype="float64")
        mass_weight_coord_2 = np.array(mass_weight_coord_2, dtype="float64")
        
        return mass_weight_coord_1, mass_weight_coord_2, element_list
    
    def pca_visualization(self, result_list, energy_list, name=""):
        plt.xlabel("PC1 (ang. / amu^0.5)")
        plt.ylabel("PC2 (ang. / amu^0.5)")
        plt.title("PCA result ("+name+")")
        
        x_array = np.array(result_list[0])
        y_array = np.array(result_list[1])
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
        plt.savefig(self.directory+"pca_result_visualization_"+str(name)+".png" ,dpi=300,format="png")
        plt.close()
        return

    def main(self):
        print("processing PCA analysis to aprrox. reaction path ...")
        file_list = sorted(glob.glob(self.directory+"samples_*_[0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.directory+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz"))  
        file_list = file_list[1:]
        struct_num = len(file_list)

        stack_coord = None
        dist_stack_coord = None
        z_mat_stack_coord = None

        for i in range(struct_num - 1):            
            mass_weight_coord_1, mass_weight_coord_2, element_list = self.read_xyz_file(file_list[i], file_list[i+1])
            modified_coord_1, modified_coord_2 = Calculationtools().kabsch_algorithm(mass_weight_coord_1, mass_weight_coord_2)
            tmp_dist = np.linalg.norm(modified_coord_1[:, np.newaxis] - modified_coord_1[np.newaxis, :], axis=2)
            tmp_dist_2 = np.linalg.norm(modified_coord_2[:, np.newaxis] - modified_coord_2[np.newaxis, :], axis=2)
            
            modified_coord_1 = modified_coord_1.reshape(1, 3*len(modified_coord_1)) # (1, N)
            modified_coord_2 = modified_coord_2.reshape(1, 3*len(modified_coord_2))
            if i == 0:
                stack_coord = copy.copy(modified_coord_1)
                upper_triangle_indices = np.triu_indices_from(tmp_dist, k=1)
                upper_triangle_distances = tmp_dist[upper_triangle_indices]    
                upper_triangle_distances = upper_triangle_distances.reshape(1, -1)
                dist_stack_coord = copy.copy(upper_triangle_distances)
                z_mat_stack_coord = cartesian_to_z_matrix(modified_coord_1.reshape(-1, 3)).reshape(1, -1)
                
            
            upper_triangle_indices = np.triu_indices_from(tmp_dist_2, k=1)
            upper_triangle_distances = tmp_dist_2[upper_triangle_indices]    
            upper_triangle_distances = upper_triangle_distances.reshape(1, -1)
            stack_coord = np.vstack((stack_coord, modified_coord_2)) # (M, N)
            dist_stack_coord = np.vstack((dist_stack_coord, upper_triangle_distances))    
            z_mat_stack_coord = np.vstack((z_mat_stack_coord, cartesian_to_z_matrix(modified_coord_2.reshape(-1, 3)).reshape(1, -1)))
          
        stack_coord_meams = np.mean(stack_coord, axis=0)
        stack_coord_std = np.std(stack_coord, axis=0)
        stack_coord_standardized = (stack_coord - stack_coord_meams) / stack_coord_std # (M, N)
        stack_coord_cov = 1 / (struct_num - 1) * np.dot(stack_coord_standardized.T, stack_coord_standardized) # (N, M) * (M, N)
        eigenvalues, eigenvectors = np.linalg.eig(stack_coord_cov)

        eigenvalues = np.real_if_close(eigenvalues, tol=1000)
        eigenvectors = np.real_if_close(eigenvectors, tol=1000).T
        
        eigval_sorted_indices = np.argsort(eigenvalues)

        sum_of_eigenvalue = 0.0
        for value in eigenvalues:
            if value < 0:
                continue
            sum_of_eigenvalue += value

        print("### cartesian coordinate PCA analysis ###")
        print("dimensional reproducibility:", np.sum(eigenvalues)/sum_of_eigenvalue)
        print("Percentage contribution 1:", eigenvalues[eigval_sorted_indices[-1]]/sum_of_eigenvalue)
        print("Percentage contribution 2:", eigenvalues[eigval_sorted_indices[-2]]/sum_of_eigenvalue)
        print("Percentage contribution 3:", eigenvalues[eigval_sorted_indices[-3]]/sum_of_eigenvalue)

        PC1 = np.sum(eigenvectors[eigval_sorted_indices[-1]] * stack_coord, axis=1)
        PC2 = np.sum(eigenvectors[eigval_sorted_indices[-2]] * stack_coord, axis=1)
        
        result_list = np.vstack((PC1, PC2))
        
        self.save_log_result(eigval_sorted_indices, eigenvalues, eigenvectors, element_list)

        self.pca_visualization(result_list, self.energy_list, name="energy")
        self.pca_visualization(result_list, self.bias_energy_list, name="bias_energy")
        
        print("### redundant coordinate (only atom distance) PCA analysis ###")
        dist_stack_coord_means = np.mean(dist_stack_coord, axis=0)
        dist_stack_coord_std = np.std(dist_stack_coord, axis=0)
        dist_stack_coord_standardized = (dist_stack_coord - dist_stack_coord_means) / dist_stack_coord_std # (M, N)
        dist_stack_coord_cov = 1 / (struct_num - 1) * np.dot(dist_stack_coord_standardized.T, dist_stack_coord_standardized) # (N, M) * (M, N)
        dist_eigenvalues, dist_eigenvectors = np.linalg.eig(dist_stack_coord_cov)
        
        dist_eigenvalues = np.real_if_close(dist_eigenvalues, tol=1000)
        dist_eigenvectors = np.real_if_close(dist_eigenvectors, tol=1000).T
        
        dist_eigval_sorted_indices = np.argsort(dist_eigenvalues)
        
        dist_sum_of_eigenvalue = 0.0
        for value in dist_eigenvalues:
            if value < 0:
                continue
            dist_sum_of_eigenvalue += value
        
        print("dimensional reproducibility:", np.sum(dist_eigenvalues)/dist_sum_of_eigenvalue)
        print("Percentage contribution 1:", dist_eigenvalues[dist_eigval_sorted_indices[-1]]/dist_sum_of_eigenvalue)
        print("Percentage contribution 2:", dist_eigenvalues[dist_eigval_sorted_indices[-2]]/dist_sum_of_eigenvalue)
        print("Percentage contribution 3:", dist_eigenvalues[dist_eigval_sorted_indices[-3]]/dist_sum_of_eigenvalue)
        
        PC1 = np.sum(dist_eigenvectors[dist_eigval_sorted_indices[-1]] * dist_stack_coord, axis=1)
        PC2 = np.sum(dist_eigenvectors[dist_eigval_sorted_indices[-2]] * dist_stack_coord, axis=1)
        
        result_list = np.vstack((PC1, PC2))
        self.save_log_result_for_redundant(dist_eigval_sorted_indices, dist_eigenvalues, dist_eigenvectors, element_list)
        self.pca_visualization(result_list, self.energy_list, name="redundant_energy")
        self.pca_visualization(result_list, self.bias_energy_list, name="redundant_bias_energy")
        
        print("### redundant coordinate (z-matrix) PCA analysis ###")
        
        z_mat_stack_coord_means = np.mean(z_mat_stack_coord, axis=0)
        z_mat_stack_coord_std = np.std(z_mat_stack_coord, axis=0) + 1e-15
        z_mat_stack_coord_standardized = (z_mat_stack_coord - z_mat_stack_coord_means) / z_mat_stack_coord_std # (M, N)
        z_mat_stack_coord_cov = 1 / (struct_num - 1) * np.dot(z_mat_stack_coord_standardized.T, z_mat_stack_coord_standardized) # (N, M) * (M, N)
        z_mat_eigenvalues, z_mat_eigenvectors = np.linalg.eig(z_mat_stack_coord_cov)
        
        z_mat_eigenvalues = np.real_if_close(z_mat_eigenvalues, tol=1000)
        z_mat_eigenvectors = np.real_if_close(z_mat_eigenvectors, tol=1000).T
        
        z_mat_eigval_sorted_indices = np.argsort(z_mat_eigenvalues)
        
        z_mat_sum_of_eigenvalue = 0.0
        for value in z_mat_eigenvalues:
            if value < 0:
                continue
            z_mat_sum_of_eigenvalue += value
        
        print("dimensional reproducibility:", np.sum(z_mat_eigenvalues)/z_mat_sum_of_eigenvalue)
        print("Percentage contribution 1:", z_mat_eigenvalues[z_mat_eigval_sorted_indices[-1]]/z_mat_sum_of_eigenvalue)
        print("Percentage contribution 2:", z_mat_eigenvalues[z_mat_eigval_sorted_indices[-2]]/z_mat_sum_of_eigenvalue)
        print("Percentage contribution 3:", z_mat_eigenvalues[z_mat_eigval_sorted_indices[-3]]/z_mat_sum_of_eigenvalue)
        
        PC1 = np.sum(z_mat_eigenvectors[z_mat_eigval_sorted_indices[-1]] * z_mat_stack_coord, axis=1)
        PC2 = np.sum(z_mat_eigenvectors[z_mat_eigval_sorted_indices[-2]] * z_mat_stack_coord, axis=1)
        
        result_list = np.vstack((PC1, PC2))
        self.save_log_result_for_z_mat(z_mat_eigval_sorted_indices, z_mat_eigenvalues, z_mat_eigenvectors, element_list)
        self.pca_visualization(result_list, self.energy_list, name="z_matrix_energy")
        self.pca_visualization(result_list, self.bias_energy_list, name="z_matrix_bias_energy")
        
        print("PCA analysis completed...")
        return

    def save_log_result(self, eigval_sorted_indices, eigenvalues, eigenvectors, element_list):
        contribution_list = eigenvalues / np.sum(eigenvalues)

        with open(self.directory+"pca_analysis_result.log", "w") as f:
            f.write("********************************************\n")
            f.write("*                                          *\n")
            f.write("*  PCA analysis for approx. reaction path  *\n")
            f.write("*                                          *\n")
            f.write("********************************************\n\n")
            for i in range(len(contribution_list)):
                f.write("-----\n")
                f.write("basis "+str(i)+"\n")
                contribution_of_eigvec = np.abs(eigenvectors[eigval_sorted_indices[-(i+1)]]) / np.sum(np.abs(eigenvectors[eigval_sorted_indices[-(i+1)]]))
                f.write("contribution: "+str(contribution_list[eigval_sorted_indices[-(i+1)]])+"\n")
                f.write("  vector    contribution of vector\n")
                for j in range(len(element_list)):
                    f.write(str(3*j+0)+"  "+str(j+1)+"  "+element_list[j]+" - x : "+str(eigenvectors[eigval_sorted_indices[-(i+1)]][3*j+0])+"   "+str(contribution_of_eigvec[3*j+0])+"\n")
                    f.write(str(3*j+1)+"  "+str(j+1)+"  "+element_list[j]+" - y : "+str(eigenvectors[eigval_sorted_indices[-(i+1)]][3*j+1])+"   "+str(contribution_of_eigvec[3*j+1])+"\n")
                    f.write(str(3*j+2)+"  "+str(j+1)+"  "+element_list[j]+" - z : "+str(eigenvectors[eigval_sorted_indices[-(i+1)]][3*j+2])+"   "+str(contribution_of_eigvec[3*j+2])+"\n")
                tmp_argsort = np.argsort(contribution_of_eigvec)
                f.write(f"{tmp_argsort[-1]}  {contribution_of_eigvec[tmp_argsort[-1]]}  {tmp_argsort[-2]}  {contribution_of_eigvec[tmp_argsort[-2]]} {tmp_argsort[-3]}  {contribution_of_eigvec[tmp_argsort[-3]]} \n")
                if len(tmp_argsort) > 3:
                    f.write(f"{tmp_argsort[-4]}  {contribution_of_eigvec[tmp_argsort[-4]]}  {tmp_argsort[-5]}  {contribution_of_eigvec[tmp_argsort[-5]]} {tmp_argsort[-6]}  {contribution_of_eigvec[tmp_argsort[-6]]} \n")
                if len(tmp_argsort) > 6:
                    f.write(f"{tmp_argsort[-7]}  {contribution_of_eigvec[tmp_argsort[-7]]}  {tmp_argsort[-8]}  {contribution_of_eigvec[tmp_argsort[-8]]} {tmp_argsort[-9]}  {contribution_of_eigvec[tmp_argsort[-9]]} \n")



            f.write("-----\n")

                
        return
    
    def save_log_result_for_redundant(self, eigval_sorted_indices, eigenvalues, eigenvectors, element_list):
        contribution_list = eigenvalues / np.sum(eigenvalues)

        with open(self.directory+"pca_analysis_result_redundant.log", "w") as f:
            f.write("********************************************\n")
            f.write("*                                          *\n")
            f.write("*  PCA analysis for approx. reaction path  *\n")
            f.write("*           redundant coordinates          *\n")
            f.write("*           (only atom distances)          *\n")
            f.write("*                                          *\n")
            f.write("********************************************\n\n")
            for i in range(len(contribution_list)):
                f.write("-----\n")
                f.write("basis "+str(i)+"\n")
                contribution_of_eigvec = np.abs(eigenvectors[eigval_sorted_indices[-(i+1)]]) / np.sum(np.abs(eigenvectors[eigval_sorted_indices[-(i+1)]]))
                f.write("contribution: "+str(contribution_list[eigval_sorted_indices[-(i+1)]])+"\n")
                f.write("       vector         contribution of vector\n")
                count = 0
                for j, k in list(itertools.combinations([l for l in range(len(element_list))], 2)):
                    f.write(str(count)+"  "+str(j+1)+" "+element_list[j]+" - "+str(k+1)+" "+element_list[k]+" : "+str(eigenvectors[eigval_sorted_indices[-(i+1)]][count])+"   "+str(contribution_of_eigvec[count])+"\n")
                    count += 1
                    
                tmp_argsort = np.argsort(contribution_of_eigvec)
                f.write(f"{tmp_argsort[-1]}  {contribution_of_eigvec[tmp_argsort[-1]]}  {tmp_argsort[-2]}  {contribution_of_eigvec[tmp_argsort[-2]]} {tmp_argsort[-3]}  {contribution_of_eigvec[tmp_argsort[-3]]} \n")

            f.write("-----\n")

                
        return
    def save_log_result_for_z_mat(self, eigval_sorted_indices, eigenvalues, eigenvectors, element_list):
        contribution_list = eigenvalues / np.sum(eigenvalues)

        with open(self.directory+"pca_analysis_result_z_matrix.log", "w") as f:
            f.write("********************************************\n")
            f.write("*                                          *\n")
            f.write("*  PCA analysis for approx. reaction path  *\n")
            f.write("*           redundant coordinates          *\n")
            f.write("*                (z-matrix)                *\n")
            f.write("*                                          *\n")
            f.write("********************************************\n\n")
            for i in range(len(contribution_list)):
                f.write("-----\n")
                f.write("basis "+str(i)+"\n")
                contribution_of_eigvec = np.abs(eigenvectors[eigval_sorted_indices[-(i+1)]]) / np.sum(np.abs(eigenvectors[eigval_sorted_indices[-(i+1)]]))
                f.write("contribution: "+str(contribution_list[eigval_sorted_indices[-(i+1)]])+"\n")
                f.write("       vector         contribution of vector\n")
                bond_count = 0
                angle_count = 0
                dihedral_count = 0


                for count in range(len(eigenvectors[eigval_sorted_indices[-(i+1)]])):
                    if count == 0 or count == 1 or (count % 3 == 0 and count > 2):
                        label = "bond "+str(bond_count+1)+" "+str(bond_count+2)
                        bond_count += 1

                    elif count == 2 or (count % 3 == 1 and count > 2):
                        label = "angle "+str(angle_count+1)+" "+str(angle_count+2)+" "+str(angle_count+3)
                        angle_count += 1

                    else:
                        label = "dihedral "+str(dihedral_count+1)+" "+str(dihedral_count+2)+" "+str(dihedral_count+3)+" "+str(dihedral_count+4)
                        dihedral_count += 1


                    f.write(str(count)+" "+label+" : "+str(eigenvectors[eigval_sorted_indices[-(i+1)]][count])+"   "+str(contribution_of_eigvec[count])+"\n")
                    
                tmp_argsort = np.argsort(contribution_of_eigvec)
                f.write(f"{tmp_argsort[-1]}  {contribution_of_eigvec[tmp_argsort[-1]]}  {tmp_argsort[-2]}  {contribution_of_eigvec[tmp_argsort[-2]]} {tmp_argsort[-3]}  {contribution_of_eigvec[tmp_argsort[-3]]} \n")

            f.write("-----\n")

                
        return