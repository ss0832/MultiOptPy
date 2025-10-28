import torch

class DispersionCalculator:
    def __init__(self, element_list, params):
        self.element_list = element_list
        
        self.c6_list = []
        self.r4r2_list = []
        self.d2_vdw_list = []
        
        for element in element_list:
            self.c6_list.append(params.c6[element])
            self.r4r2_list.append(params.r4r2[element])
            self.d2_vdw_list.append(params.d2_vdw[element])

        self.c6_list = torch.tensor(self.c6_list, dtype=torch.float64)
        self.r4r2_list = torch.tensor(self.r4r2_list, dtype=torch.float64)
        self.d2_vdw_list = torch.tensor(self.d2_vdw_list, dtype=torch.float64)
        
        self.s6 = params.s6
        self.s8 = params.s8
        self.beta_6 = params.beta_6
        self.beta_8 = params.beta_8
        
        return

    def calculation(self, xyz):  # xyz: (N,3) torch tensor
        import torch
        N = xyz.shape[0]
        diff = xyz.unsqueeze(1) - xyz.unsqueeze(0)
        r = torch.linalg.norm(diff, dim=-1)
        # Avoid division by zero on diagonal by setting a large value
        r = r + torch.eye(N, device=xyz.device) * 1e10

        c6_ij = torch.sqrt(self.c6_list.unsqueeze(1) * self.c6_list.unsqueeze(0))
        r4r2_ij = torch.sqrt(self.r4r2_list.unsqueeze(1) * self.r4r2_list.unsqueeze(0))
        c8_ij = 3.0 * c6_ij * r4r2_ij
        d2_sum = self.d2_vdw_list.unsqueeze(1) + self.d2_vdw_list.unsqueeze(0)

        tmp6 = 6.0 * (d2_sum / r) * self.beta_6
        damp_6 = 1.0 / (1.0 + tmp6)
        tmp8 = 6.0 * (d2_sum / r) * self.beta_8
        damp_8 = 1.0 / (1.0 + tmp8)

        term_6 = self.s6 * (c6_ij / (r ** 6)) * damp_6
        term_8 = self.s8 * (c8_ij / (r ** 8)) * damp_8

        energy = -torch.sum(term_6 + term_8) / 2.0
        return energy  # shape: scalar

    def energy(self, xyz):
        energy = self.calculation(torch.tensor(xyz, dtype=torch.float64))
        

        return energy

    def gradient(self, xyz):
        energy = self.calculation(torch.tensor(xyz, dtype=torch.float64))
        gradient = torch.func.jacrev(self.calculation)(torch.tensor(xyz, dtype=torch.float64))
        return energy, gradient 
    
    def hessian(self, xyz):
        energy = self.calculation(torch.tensor(xyz, dtype=torch.float64))
        hessian = torch.func.hessian(self.calculation)(torch.tensor(xyz, dtype=torch.float64)).reshape(xyz.shape[0] * 3, xyz.shape[0] * 3)
        return energy, hessian