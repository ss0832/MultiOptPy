import numpy as np

#J. Chem. Phys. 157, 124107 (2022)
#https://doi.org/10.1063/5.0102145

class BITSSModelFunction:
    def __init__(self, geom_num_list_1, geom_num_list_2):
        self.f = 0.5
        self.alpha = 10.0
        self.beta = 0.1
        self.d = np.linalg.norm(geom_num_list_1 - geom_num_list_2)
        
        return
        
    def calc_energy(self, energy_1, energy_2, geom_num_list_1, geom_num_list_2, gradient_1, gradient_2, iter):
        current_distance = np.linalg.norm(geom_num_list_1 - geom_num_list_2)
        if iter % 500 == 0:

            self.E_B = abs(energy_1 - energy_2)
            self.kappa_e = self.alpha / (2.0 * self.E_B + 1e-10)
        
            unit_vec = (geom_num_list_1 - geom_num_list_2) / current_distance
            
     
            
            proj_grad_1 = gradient_1 * unit_vec * (-1)
            proj_grad_2 = gradient_2 * unit_vec

            a = np.sqrt(np.linalg.norm(proj_grad_1) + np.linalg.norm(proj_grad_2)) / (2 ** 1.5 * self.beta * self.d + 1e-10)
            b = self.E_B / (self.beta * self.d ** 2 + 1e-10)
            self.kappa_d = max(a, b)
            self.d = np.linalg.norm(geom_num_list_1 - geom_num_list_2)
        
        self.d = max((1.0 - self.f) * self.d, 1e-10)
        energy = energy_1 + energy_2 + self.kappa_e * (energy_1 + energy_2) ** 2 + self.kappa_d * (current_distance - self.d) ** 2
        
        return energy
        
    def calc_grad(self, energy_1, energy_2, geom_num_list_1, geom_num_list_2, gradient_1, gradient_2):
        current_vec = geom_num_list_1 - geom_num_list_2
        current_dist = np.linalg.norm(current_vec) + 1e-10

        bitss_grad_1 = gradient_1 + gradient_2 + 2.0 * self.kappa_e * (energy_1 - energy_2) * (gradient_1 - gradient_2) + current_vec * 2.0 * self.kappa_d * (current_dist - self.d) / current_dist

        bitss_grad_2 = gradient_1 + gradient_2 + 2.0 * self.kappa_e * (energy_1 - energy_2) * (gradient_1 - gradient_2) - current_vec * 2.0 * self.kappa_d * (current_dist - self.d) / current_dist
        
        return bitss_grad_1, bitss_grad_2