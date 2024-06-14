import numpy as np
import matplotlib.pyplot as plt
# NRO analysis
# ref. S. Ebisawa, M. Hasebe, T. Tsutsumi, T. Tsuneda, and T. Taketsugu, Phys. Chem. Chem. Phys. 24, 3532 (2022).


class NROAnalysis:
    def __init__(self, **config):
        # Currently this method is only available for tblite.
        # Visualization of NRO is not implemented.
        self.config = config
        self.method = self.config["xtb"]
        self.element_list = self.config["element_list"]
        self.electric_charge_and_multiplicity = self.config["electric_charge_and_multiplicity"]
        
        self.numerical_delta = 0.0001
        self.LAMBDA_list = []
        self.file_directory = self.config["file_directory"]
        return
        
    def save_results(self, energy_list, bias_energy_list):
        # LAMBDA is singular value describing magnitude of change in a molecular orbitals when atoms are moved in a particular direction.
      
        num_list = np.array(range(len(self.LAMBDA_list)))
        
        self.LAMBDA_list = np.array(self.LAMBDA_list)
        self.LAMBDA_list = (self.LAMBDA_list - np.min(self.LAMBDA_list)) / (np.max(self.LAMBDA_list) - np.min(self.LAMBDA_list))
  
        fig = plt.figure()
        ax1 = fig.add_subplot(111)  
        ax1.plot(num_list, energy_list, label='Energy [kcal/mol]',color="g")
        ax1.plot(num_list, bias_energy_list, label='Bias Energy [kcal/mol]',color="b")
        ax2 = ax1.twinx()
        ax2.plot(num_list, self.LAMBDA_list,label='LAMBDA', color="r")
        #ax2.set_yscale('log')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')

        ax1.set_xlabel('# iteration')
        ax1.set_ylabel("Energy [kcal/mol]")
        ax2.set_ylabel('LAMBDA')
        #if np.max(self.LAMBDA_list[i]) > 1.0:
        #    ax2.axis([num_list[0], num_list[-1], 0, 1.0])
        plt.title("LAMBDA ")
        plt.tight_layout()
        plt.savefig(self.file_directory+"NRO_lambda_plot.png", format="png", dpi=300)
        plt.close()
        with open(self.file_directory+"NRO_lambda_plot.csv", "w") as f:
            f.write("#ITR. , energy [kcal/mol], bias energy [kcal/mol], LAMBDA\n")
            for j in range(len(self.LAMBDA_list)):
                f.write(str(num_list[j])+","+str(energy_list[j])+","+str(bias_energy_list[j])+","+str(self.LAMBDA_list[j])+"\n")
                
            
        return
        
    def run(self, SP, geom_num_list, move_vector):
        print("### Natural Reaction Orbital analysis ###")
        print("# This method is only available for tblite (xTB).")
        displacement_for_numerical_differential = self.numerical_delta * move_vector * np.linalg.norm(move_vector)
        neutral_orbital_coefficients = SP.orbital_coefficients
        overlap_matrix = SP.overlap_matrix
        #plus  positions, element_number_list, electric_charge_and_multiplicity, method
        _, _, _ = SP.single_point_no_directory(geom_num_list+displacement_for_numerical_differential, self.element_list, self.electric_charge_and_multiplicity, self.method)
        p_orbital_coefficients = SP.orbital_coefficients
        
        #minus
        _, _, _ = SP.single_point_no_directory(geom_num_list-displacement_for_numerical_differential, self.element_list, self.electric_charge_and_multiplicity, self.method)
        m_orbital_coefficients = SP.orbital_coefficients
        
        first_derivative_orbital_coefficients = (p_orbital_coefficients - m_orbital_coefficients) / (self.numerical_delta * 2)
        first_response_matrix = np.dot(np.conjugate(neutral_orbital_coefficients.T), np.dot(overlap_matrix, first_derivative_orbital_coefficients))
        L, LAMBDA, R = np.linalg.svd(first_response_matrix)
        
        N_R = np.conjugate(R.T)
        N_L = np.conjugate(L.T)
        #print(N_L, LAMBDA, N_R)
        #print("LAMBDA:", LAMBDA)
        sum_of_LAMBDA = np.sum(LAMBDA)
        self.LAMBDA_list.append(sum_of_LAMBDA)
        # temporary save
        with open(self.file_directory+"NRO_lambda_plot.csv", "a") as f:
             f.write(str(sum_of_LAMBDA)+"\n")
       
        return
    
    
