import numpy as np


class SeamModelFunction:
    def __init__(self):
        #ref.: J. Phys. Chem. A 2009, 113, 9, 1704-1710
        self.alpha = 0.05
        return
    
    def calc_energy(self, energy_1, energy_2):
        tot_energy = 0.5 * (energy_1 + energy_2) + (energy_1 - energy_2) ** 2 / (self.alpha)
        return tot_energy
    
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        
        smf_grad_1 = 0.5 * (grad_1) + 0.5 * (grad_2) + 2 * (energy_1 - energy_2) * (grad_1 - grad_2) / (self.alpha)
        smf_grad_2 = 0.5 * (grad_1) + 0.5 * (grad_2) + 2 * (energy_1 - energy_2) * (grad_1 - grad_2) / (self.alpha)
        return smf_grad_1, smf_grad_2
    
    def calc_hess(self, energy_1, energy_2, grad_1, grad_2, hess_1, hess_2):
        grad_1 = grad_1.reshape(-1, 1)
        grad_2 = grad_2.reshape(-1, 1)
        delta_grad = grad_1 - grad_2
        delta_energy = energy_1 - energy_2
        smf_hess = 0.5 * (hess_1 + hess_2) + (2.0 / self.alpha) * np.dot(delta_grad, delta_grad.T) + (2.0 / self.alpha) * delta_energy * (hess_1 - hess_2)
        
        return smf_hess