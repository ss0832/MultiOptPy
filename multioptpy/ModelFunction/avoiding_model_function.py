import numpy as np

class AvoidingModelFunction:
    def __init__(self):
        #ref.: Advances in Physical Chemistry 2012, 2012, 1-13.
        self.alpha = 0.01
        return
    
    def calc_energy(self, energy_1, energy_2):
        U = self.alpha / 2.0 * np.exp(-1 * (energy_1 - energy_2) ** 2 / self.alpha)
        tot_energy = 0.5 * (energy_1 + energy_2) + 0.5 * np.sqrt((energy_1 - energy_2) ** 2 + 4.0 * U)
        return tot_energy
    
    def calc_grad(self, energy_1, energy_2, grad_1, grad_2):
        b = np.exp(-1 * (energy_1 - energy_2) ** 2 / self.alpha)
        U = self.alpha / 2.0 * b
        a = np.sqrt((energy_1 - energy_2) ** 2 + 4.0 * U)
        dU_dq1 = -1 * (energy_1 - energy_2) * grad_1 * b 
        dU_dq2 = (energy_1 - energy_2) * grad_2 * b

        smf_grad_1 = 0.5 * (grad_1 + grad_2) + 0.5 * (1.0 /a) * ( 2.0 * (energy_1 - energy_2) * (grad_1) + 8.0 * U * dU_dq1) + 0.5 * (1.0 /a) * (-2.0 * (energy_1 - energy_2) * (grad_2) + 8.0 * U * dU_dq2)
        smf_grad_2 = 0.5 * (grad_1 + grad_2) + 0.5 * (1.0 /a) * ( 2.0 * (energy_1 - energy_2) * (grad_1) + 8.0 * U * dU_dq1) + 0.5 * (1.0 /a) * (-2.0 * (energy_1 - energy_2) * (grad_2) + 8.0 * U * dU_dq2)
        
        return smf_grad_1, smf_grad_2

    def calc_hess(self, energy_1, energy_2, grad_1, grad_2, hess_1, hess_2):
        
        
        return