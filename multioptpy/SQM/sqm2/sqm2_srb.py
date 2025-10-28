import torch

class SRBCalculator:
    def __init__(self, element_list, params):
        
        self.k_srb = params.k_srb
        self.eta_srb = params.eta_srb
        self.g_scal_srb = params.g_scal_srb
        self.c_1_srb = params.c_1_srb
        self.c_2_srb = params.c_2_srb
        
        self.en_list = []
        self.ro_list = []
        for elem in element_list:
            self.en_list.append(params.en_data_srb[elem])
            self.ro_list.append(params.r0_data_srb[elem])

        self.en_list = torch.tensor(self.en_list, dtype=torch.float64)
        self.ro_list = torch.tensor(self.ro_list, dtype=torch.float64)

    def calculation(self, xyz):
        device = xyz.device
        en_list = self.en_list.to(device)
        ro_list = self.ro_list.to(device)
        
        diff = xyz.unsqueeze(1) - xyz.unsqueeze(0)
        dist_sq_matrix = torch.sum(diff**2, dim=-1)
        dist_matrix = torch.sqrt(dist_sq_matrix + 1e-12)
        
        delta_en_matrix = torch.abs(en_list.unsqueeze(1) - en_list.unsqueeze(0))
        delta_en_sq_matrix = delta_en_matrix ** 2.0
        
        ro_sum_matrix = ro_list.unsqueeze(1) + ro_list.unsqueeze(0)
        
        correction_factor = 1.0 - self.c_1_srb * delta_en_matrix - self.c_2_srb * delta_en_sq_matrix
        
        r_cov_matrix = ro_sum_matrix * correction_factor
        
        exponent_term1 = 1.0 + self.g_scal_srb * delta_en_sq_matrix
        exponent_term2 = (dist_matrix - r_cov_matrix) ** 2.0
        
        exponent = -1.0 * self.eta_srb * exponent_term1 * exponent_term2
        
        energy_matrix = self.k_srb * torch.exp(exponent)
        
        total_energy = torch.sum(torch.triu(energy_matrix, diagonal=1))
        
        return total_energy

    def energy(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=False)
        energy = self.calculation(xyz)
        return energy 
    
    def gradient(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        energy = self.calculation(xyz)
        gradient = torch.func.jacrev(self.calculation)(xyz)
        
        return energy, gradient
    
    def hessian(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        energy = self.calculation(xyz)
        hessian = torch.func.hessian(self.calculation)(xyz)
        hessian = hessian.reshape(xyz.shape[0] * 3, xyz.shape[0] * 3)
        return energy, hessian
    
    

