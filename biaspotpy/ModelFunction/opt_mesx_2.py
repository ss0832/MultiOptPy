import numpy as np

class OptMESX2:
    def __init__(self):
        #ref.: Theor Chem Acc 99, 95–99 (1998)
       
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
        grad_1 = grad_1.reshape(-1, 1)
        grad_2 = grad_2.reshape(-1, 1)
        
        delta_grad = grad_1 - grad_2
        norm_delta_grad = np.linalg.norm(delta_grad)
        projection = np.sum(grad_1 * delta_grad) / norm_delta_grad
        parallel = grad_1 - delta_grad * projection / norm_delta_grad
        
        gp_grad = (energy_1 - energy_2) * 140 * delta_grad + 1.0 * parallel
        
        gp_grad = gp_grad.reshape(-1, 3)
        return gp_grad