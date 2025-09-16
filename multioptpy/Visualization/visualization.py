import matplotlib.pyplot as plt
import numpy as np

from multioptpy.Parameters.parameter import UnitValueLib

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
        """
        Plot two energy profiles on a single figure using primary and secondary y-axes
        with proper legends.
        
        Args:
            num_list: Iteration numbers (x-axis)
            energy_list: First energy profile (primary y-axis)
            energy_list_2: Second energy profile (secondary y-axis)
            add_file_name: Additional text for the output file name
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Primary axis (left) - Normal energy
        color1 = 'green'
        line1, = ax1.plot(num_list, energy_list, color=color1, linestyle='--', marker='.', label='Normal Energy')
        ax1.set_xlabel('ITR.')
        ax1.set_ylabel('Electronic Energy [kcal/mol]', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Secondary axis (right) - Bias energy
        color2 = 'blue'
        ax2 = ax1.twinx()
        line2, = ax2.plot(num_list, energy_list_2, color=color2, linestyle='--', marker='.', label='Bias Energy')
        ax2.set_ylabel('Electronic Energy [kcal/mol]', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')
        
        # Title and layout
        plt.title('Energy Profile')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.BPA_FOLDER_DIRECTORY + "energy_plot_" + add_file_name + ".png", format="png", dpi=300)
        plt.close()
        return
        
    def single_plot(self, num_list, energy_list, file_directory, atom_num, axis_name_1="ITR. ", axis_name_2="cosθ", name="orthogonality"):
        fig, ax = plt.subplots()
        ax.plot(num_list,energy_list, "b--o" , markersize=3)

        ax.set_title(str(atom_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(self.BPA_FOLDER_DIRECTORY+"plot_"+name+"_"+str(atom_num)+".png", format="png", dpi=200)
        plt.close()
         
        return



class NEBVisualizer:
    """Visualization functionality for NEB calculations"""
    
    def __init__(self, config):
        self.config = config
        self.color_list = ["b"]  # for matplotlib
    
    def simple_plot(self, num_list, data_list, file_directory, optimize_num, 
                   axis_name_1="NODE #", axis_name_2="Value", name="data"):
        """Create a simple plot"""
        fig, ax = plt.subplots()
        ax.plot(num_list, data_list, 
               self.color_list[0] + "--o")
        
        ax.set_title(str(optimize_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(f"{self.config.NEB_FOLDER_DIRECTORY}plot_{name}_{optimize_num}.png", 
                   format="png", dpi=200)
        plt.close()

    def simple_scatter_plot(self, num_list, data_list, file_directory, optimize_num, 
                           axis_name_1="NODE #", axis_name_2="Value", name="data"):
        """Create a simple scatter plot"""
        fig, ax = plt.subplots()
        ax.scatter(num_list, data_list, color=self.color_list[0], marker='o')
        ax.plot(num_list, data_list, self.color_list[0]+'--o')
        ax.set_title(str(optimize_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(f"{self.config.NEB_FOLDER_DIRECTORY}plot_{name}_{optimize_num}.png", 
                   format="png", dpi=200)
        plt.close()

    def plot_energy(self, num_list, energy_list, optimize_num, 
                   axis_name_1="NODE #", axis_name_2="Electronic Energy [kcal/mol]", name="energy"):
        """Plot energy profile"""
        self.simple_plot(num_list, energy_list, "", optimize_num, axis_name_1, axis_name_2, name)
    
    def plot_gradient(self, num_list, gradient_norm_list, optimize_num, 
                     axis_name_1="NODE #", axis_name_2="Gradient (RMS) [a.u.]", name="gradient"):
        """Plot gradient profile"""
        self.simple_plot(num_list, gradient_norm_list, "", optimize_num, axis_name_1, axis_name_2, name)
    
    def plot_orthogonality(self, num_list, cos_list, optimize_num):
        """Plot orthogonality profile"""
        self.simple_plot(num_list, cos_list, "", optimize_num, 
                        axis_name_1="NODE #", axis_name_2="cosθ", name="orthogonality")
    
    def plot_perpendicular_gradient(self, num_list, force_list, optimize_num, force_type="rms"):
        """Plot perpendicular gradient profile"""
        if force_type == "rms":
            axis_name_2 = "Perpendicular Gradient (RMS) [a.u.]"
            name = "perp_rms_gradient"
        else:
            axis_name_2 = "Perpendicular Gradient (MAX) [a.u.]"
            name = "perp_max_gradient"
        
        self.simple_plot(num_list, force_list, "", optimize_num, 
                        axis_name_1="NODE #", axis_name_2=axis_name_2, name=name)

# For ADDF-like method in ieip.py
def plot_potential_energy_path(energy_list, path, additional_name=""):
    min_energy = np.min(energy_list)
    energy_list -= min_energy
    energy_list *= UnitValueLib().hartree2kcalmol
    plt.plot(energy_list, marker='.', linestyle='--', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("Potential Energy Path")
    plt.savefig(path+"/"+additional_name+"_energy_profile.png")
    plt.close()
    return
