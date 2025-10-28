import torch

class RepulsionCalculator:
    def __init__(self, element_list, params):
        
        self.rep_alpha_list = []
        self.rep_zeff_list = []
        
        for elem in element_list:
            self.rep_alpha_list.append(params.repAlpha[elem])
            self.rep_zeff_list.append(params.repZeff[elem])
        
        self.rep_alpha_list = torch.tensor(self.rep_alpha_list, dtype=torch.float64)
        self.rep_zeff_list = torch.tensor(self.rep_zeff_list, dtype=torch.float64)
        
        return
    
    
    def calculation(self, xyz):
        """
        This is the vectorized version of your calculation method.
        It removes the nested Python loops for efficiency.
        
        Args:
            xyz (torch.Tensor): A tensor of atomic coordinates, 
                                shape [n_atoms, 3].
        
        Returns:
            torch.Tensor: A scalar tensor containing the total energy.
        """
        device = xyz.device
        zeff_list = self.rep_zeff_list.to(device)
        alpha_list = self.rep_alpha_list.to(device)
        diff = xyz.unsqueeze(1) - xyz.unsqueeze(0)
        dist_sq_matrix = torch.sum(diff**2, dim=-1)
        dist_matrix = torch.sqrt(dist_sq_matrix + 1e-12)
        zeff_matrix = torch.outer(zeff_list, zeff_list)
        alpha_matrix = torch.outer(alpha_list, alpha_list)
        dist_cubed = dist_matrix ** 3.0
        exp_term = torch.exp(-1.0 * torch.sqrt(alpha_matrix * dist_cubed))
        inv_dist = 1.0 / dist_matrix 
        energy_matrix = zeff_matrix * inv_dist * exp_term
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
        hessian = hessian.reshape(xyz.shape[0]*3, xyz.shape[0]*3)
        return energy, hessian


