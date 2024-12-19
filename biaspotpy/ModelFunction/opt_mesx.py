import numpy as np

class OptMESX:
    def __init__(self):
        #ref.: Chemical Physics Letters 674 (2017) 141-145
       
        self.switch_threshold = 5e-4
        self.alpha = 1e-3
        return
    
    def calc_energy(self, energy_1, energy_2):
        tot_energy = (energy_1 + energy_2) / 2.0 
        print("energy_1:", energy_1, "hartree")
        print("energy_2:", energy_2, "hartree")
        print("energy_1 - energy_2:", abs(energy_1 - energy_2), "hartree")
        return tot_energy
        
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        delta_grad = grad_1 - grad_2
        dgv_vec = delta_grad / np.linalg.norm(delta_grad)
        dgv_vec = dgv_vec.reshape(-1, 1)
        
        P_matrix = np.eye((len(dgv_vec))) -1 * np.dot(dgv_vec, dgv_vec.T)
        gp_grad =  2 * (energy_1 - energy_2) * dgv_vec + np.dot(P_matrix, 0.5 * (grad_1.reshape(-1, 1) + grad_2.reshape(-1, 1)))
        
        gp_grad = gp_grad.reshape(len(grad_1), 3)
        return gp_grad