import numpy as np

class OptMECI:
    def __init__(self):
        # ref.:https://doi.org/10.1021/ct1000268
       
        self.switch_threshold = 5e-4
        self.alpha = 1e-3
        self.approx_cdv_vec = None
        self.prev_dgv_vec = None
        self.dgv_vec = None
        
        return
    
    def calc_energy(self, energy_1, energy_2):
        tot_energy = (energy_1 + energy_2) / 2.0 
        print("energy_1:", energy_1, "hartree")
        print("energy_2:", energy_2, "hartree")
        print("energy_1 - energy_2:", abs(energy_1 - energy_2), "hartree")
        return tot_energy
        
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        if self.approx_cdv_vec is None:
            self.approx_cdv_vec = np.ones((len(grad_1)*3, 1))

        delta_grad = grad_1 - grad_2
        dgv_vec = delta_grad / np.linalg.norm(delta_grad)
        dgv_vec = dgv_vec.reshape(-1, 1)
        
        if self.prev_dgv_vec is None:
            self.prev_dgv_vec = dgv_vec
        
        self.approx_cdv_vec = (np.dot(self.approx_cdv_vec.T, dgv_vec) * self.prev_dgv_vec -1 * np.dot(self.prev_dgv_vec.T, dgv_vec) * self.approx_cdv_vec) / np.sqrt(np.dot(self.approx_cdv_vec.T, dgv_vec) ** 2 + np.dot(self.prev_dgv_vec.T, dgv_vec) ** 2) 

        P_matrix = np.eye((len(dgv_vec))) -1 * np.dot(dgv_vec, dgv_vec.T) -1 * np.dot(self.approx_cdv_vec, self.approx_cdv_vec.T)
        P_matrix = 0.5 * (P_matrix + P_matrix.T)
        gp_grad =  2 * (energy_1 - energy_2) * dgv_vec + np.dot(P_matrix, 0.5 * (grad_1.reshape(-1, 1) + grad_2.reshape(-1, 1)))
        
        self.prev_dgv_vec = dgv_vec
        gp_grad = gp_grad.reshape(len(grad_1), 3)
        return gp_grad


    def calc_hess(self, hess_1, hess_2):
        mean_hess = 0.5 * (hess_1 + hess_2)
        P_matrix = np.eye((len(self.prev_dgv_vec))) -1 * np.dot(self.prev_dgv_vec, self.prev_dgv_vec.T) -1 * np.dot(self.approx_cdv_vec, self.approx_cdv_vec.T)
        P_matrix = 0.5 * (P_matrix + P_matrix.T)
        
        gp_hess = np.dot(P_matrix, np.dot(mean_hess, P_matrix))
        return gp_hess