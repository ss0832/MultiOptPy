import numpy as np


class ConicalModelFunction:
    def __init__(self):
        #ref.: J. Phys. Chem. B 2008, 112, 2, 405–413
        self.alpha = 0.025
        self.sigma = 3.5
        return
    
    def calc_energy(self, energy_1, energy_2):
        delta_ene = energy_1 - energy_2
        
        tot_energy = 0.5 * (energy_1 + energy_2) + self.sigma * delta_ene ** 2 / (delta_ene + self.alpha)
        return tot_energy
    
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        delta_ene = energy_1 - energy_2
        mf_grad_1 = 0.5 * (grad_1) + 0.5 * (grad_2) + self.sigma * (delta_ene ** 2.0 + 2.0 * self.alpha * delta_ene) / (delta_ene + self.alpha) ** 2.0 * (grad_1 - grad_2)
        mf_grad_2 = 0.5 * (grad_1) + 0.5 * (grad_2) + self.sigma * (delta_ene ** 2.0 + 2.0 * self.alpha * delta_ene) / (delta_ene + self.alpha) ** 2.0 * (grad_1 - grad_2)
        return mf_grad_1, mf_grad_2
    
    def calc_hess(self, energy_1, energy_2, grad_1, grad_2, hess_1, hess_2):
        
        
        return