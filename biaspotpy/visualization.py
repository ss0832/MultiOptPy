import matplotlib.pyplot as plt
import numpy as np
class Graph:
    def __init__(self, folder_directory):
        self.BPA_FOLDER_DIRECTORY = folder_directory
        return
    
    def stem_plot(self, freq_list, int_list, add_file_name=""):
        fig, ax = plt.subplots()
        ax.stem(freq_list, int_list)
        ax.set_title(add_file_name)
        y_max = np.max(int_list) + (np.max(int_list) * 0.1)
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('Wave number [/ nm]')
        ax.set_ylabel('intensity [a.u.]')
        fig.tight_layout()
        fig.savefig(self.BPA_FOLDER_DIRECTORY+"stem_plot_"+add_file_name+".png", format="png", dpi=300)
        plt.close()
        return
    
    def double_plot(self, num_list, energy_list, energy_list_2, add_file_name=""):
        
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.plot(num_list, energy_list, "g--.")
        ax2.plot(num_list, energy_list_2, "b--.")

        ax1.set_xlabel('ITR.')
        ax2.set_xlabel('ITR.')

        ax1.set_ylabel('Electronic Energy [kcal/mol]')
        ax2.set_ylabel('Electronic Energy [kcal/mol]')
        plt.title('normal_above AFIR_below')
        plt.tight_layout()
        plt.savefig(self.BPA_FOLDER_DIRECTORY+"Energy_plot_"+add_file_name+".png", format="png", dpi=300)
        plt.close()
        return
        
    def single_plot(self, num_list, energy_list, file_directory, atom_num, axis_name_1="ITR. ", axis_name_2="cosθ", name="orthogonality"):
        fig, ax = plt.subplots()
        ax.plot(num_list,energy_list, "r--o" , markersize=2)

        ax.set_title(str(atom_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(self.BPA_FOLDER_DIRECTORY+"Plot_"+name+"_"+str(atom_num)+".png", format="png", dpi=200)
        plt.close()
         
        return


