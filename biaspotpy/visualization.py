import matplotlib.pyplot as plt

class Graph:
    def __init__(self, folder_directory):
        self.BPA_FOLDER_DIRECTORY = folder_directory
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