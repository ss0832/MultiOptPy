import torch
import numpy as np


class SeamModelFunction:
    def __init__(self):
        #ref.: J. Phys. Chem. A 2009, 113, 9, 1704-1710
        self.alpha = 0.01
        return
    
    def calc_energy(self, energy_1, energy_2):
        tot_energy = 0.5 * (energy_1 + energy_2) + (energy_1 - energy_2) ** 2 / (self.alpha)
        return tot_energy
    
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        if energy_2 > energy_1:
            energy_2, energy_1 = energy_1, energy_2
            grad_1, grad_2 = grad_1, grad_2
        
        smf_grad_1 = 0.5 * (grad_1) + 0.5 * (grad_2) + 2 * (energy_1 - energy_2) * (grad_1 - grad_2) / (self.alpha)
        smf_grad_2 = 0.5 * (grad_1) + 0.5 * (grad_2) + 2 * (energy_1 - energy_2) * (grad_1 - grad_2) / (self.alpha)
        return smf_grad_1, smf_grad_2