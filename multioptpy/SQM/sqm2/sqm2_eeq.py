import torch
    
class IESEnergyCalculator():
    def __init__(self, element_list, tot_charge, params):
        
        self.tot_charge = tot_charge
        self.element_list = element_list
        
        eeq_alpha = params.eeqAlp # atomic radius alpha
        eeq_kcn = params.eeqkCN # element_dependent_factor for the element_vector X
        eeq_gamma = params.eeqGam # chemical hardness J_AA
        eeq_kappa = params.eeqChi # electronegativity for eeq model 
        covalent_radii = params.eeq_covalent_radii
        self.cov_radii_list = []
        self.eeq_alpha_list = []
        self.eeq_kcn_list = []
        self.eeq_gamma_list = []
        self.eeq_en_list = []
        
        for elem in element_list:# 0-indexed
            self.cov_radii_list.append(covalent_radii[elem])
            self.eeq_alpha_list.append(eeq_alpha[elem])
            self.eeq_kcn_list.append(eeq_kcn[elem])
            self.eeq_gamma_list.append(eeq_gamma[elem])
            self.eeq_en_list.append(eeq_kappa[elem])
        
        self.cov_radii_list = torch.tensor(self.cov_radii_list, dtype=torch.float64)
        self.eeq_alpha_list = torch.tensor(self.eeq_alpha_list, dtype=torch.float64)
        self.eeq_kcn_list = torch.tensor(self.eeq_kcn_list, dtype=torch.float64)
        self.eeq_gamma_list = torch.tensor(self.eeq_gamma_list, dtype=torch.float64)
        self.eeq_en_list = torch.tensor(self.eeq_en_list, dtype=torch.float64)


    def get_coulomb_matrix(self, xyz):
        
        N = xyz.shape[0]
        
        r_ij_vec = (xyz.unsqueeze(1) - xyz.unsqueeze(0)).clone()
        r_ij_sq = torch.sum(r_ij_vec**2, dim=2)
        eps = torch.finfo(r_ij_sq.dtype).eps**0.5 

        r_ij = torch.sqrt(r_ij_sq.clone() + eps)
        gammas = self.eeq_gamma_list
        tmp_gamma_ij_sq = (gammas.unsqueeze(1)**2.0 + gammas.unsqueeze(0)**2.0).clone()
        tmp_gamma_ij = torch.sqrt(tmp_gamma_ij_sq)
        
        erf_term = torch.erf(tmp_gamma_ij * r_ij)
        
        eye_mask = torch.eye(N, dtype=torch.bool, device=xyz.device)
        

        r_ij_safe = torch.where(eye_mask, 1.0, r_ij)
        
        coulomb_matrix = torch.where(
            eye_mask,
            0.0,
            erf_term / r_ij_safe
        )
        
        
        diag_values = self.eeq_gamma_list + (2.0 * self.eeq_alpha_list / (torch.pi ** 0.5))
        
        coulomb_matrix = coulomb_matrix + torch.diag(diag_values)
        
        return coulomb_matrix.clone()


    def get_coulomb_matrix_legacy(self, xyz):# xyz: torch tensor, shape (N,3)
        coulomb_matrix = torch.zeros((len(self.element_list), len(self.element_list)), dtype=torch.float64)
        
        for i in range(len(self.element_list)):
            for j in range(len(self.element_list)):
                if i == j:
                    tmp_Jii = self.eeq_gamma_list[i]
                    tmp_Gii = 2.0 * self.eeq_alpha_list[i] / (torch.pi ** 0.5)
                    coulomb_matrix[i, j] = tmp_Jii + tmp_Gii
                else:
                    r_ij_vec = xyz[i] - xyz[j]
                    r_ij = torch.linalg.norm(r_ij_vec)
                    tmp_gamma_ij = torch.sqrt(self.eeq_gamma_list[i] ** 2.0 + self.eeq_gamma_list[j] ** 2.0)
                    erf_term = torch.erf(tmp_gamma_ij * r_ij)
                    coulomb_matrix[i, j] = erf_term / r_ij
                    coulomb_matrix[j, i] = coulomb_matrix[i, j] 
        
        return coulomb_matrix # shape (N,N)

    def get_cn_modified(self, xyz):
        """
        Calculates the vectorized modified coordination number.
        """
        
        N = xyz.shape[0]
        r_ij_vec = (xyz.unsqueeze(1) - xyz.unsqueeze(0)).clone()
        r_ij_sq = torch.sum(r_ij_vec**2, dim=2)
        eps = torch.finfo(r_ij_sq.dtype).eps**0.5
        r_ij = torch.sqrt(r_ij_sq.clone() + eps)
        rij_cov = (self.cov_radii_list.unsqueeze(1) + self.cov_radii_list.unsqueeze(0)).clone()
        ratio = r_ij / rij_cov
        arg = -7.5 * (ratio - 1.0)
        tmp_mcn_matrix = 0.5 * (1.0 + torch.erf(arg))
        tmp_mcn_matrix.fill_diagonal_(0.0)
        cn_mod_1d = torch.sum(tmp_mcn_matrix, dim=1).clone()
        return cn_mod_1d.reshape(N, 1).clone() * 2.0


    def get_cn_modified_legacy(self, xyz): # xyz: torch tensor, shape (N,3)
        cn_mod = torch.zeros((len(self.element_list), 1), dtype=torch.float64)
        
        for i in range(len(self.element_list)):
            for j in range(len(self.element_list)):
                if i != j:
                    r_ij_vec = xyz[i] - xyz[j]
                    r_ij = torch.linalg.norm(r_ij_vec)
                    rij_cov = self.cov_radii_list[i] + self.cov_radii_list[j]
                    tmp_mcn = 0.5 * (1.0 + torch.erf(-7.5 * ((r_ij / rij_cov) - 1.0)))
                    cn_mod[i, 0] += tmp_mcn
        
        return cn_mod # shape (N,1)

    def get_x_vector(self, xyz):
        """
        Calculates the vectorized x_vector, cloning the result
        to prevent UnsafeViewBackward0 errors.
        """
        
        m_CN_list = self.get_cn_modified(xyz)
        
        sqrt_cn = torch.sqrt(m_CN_list)
        
        kcn_list_col = self.eeq_kcn_list.reshape(-1, 1)
        en_list_col = self.eeq_en_list.reshape(-1, 1)

        x_vector = kcn_list_col * sqrt_cn - en_list_col

        return x_vector.clone()

    def get_x_vector_legacy(self, xyz): # xyz: torch tensor, shape (N,3)
        x_vector = torch.zeros((len(self.element_list), 1), dtype=torch.float64)

        m_CN_list = self.get_cn_modified(xyz)
        
        for i in range(len(self.element_list)):
            cn_i = m_CN_list[i]
            x_vector[i, 0] = self.eeq_kcn_list[i] * (cn_i ** 0.5) - self.eeq_en_list[i]
        
        return x_vector # shape (N,1)

    def get_q(self, coulomb_matrix, x_vector):
        
        N = len(self.element_list)
        

        Q_matrix = torch.zeros((N + 1, N + 1), dtype=torch.float64, device=coulomb_matrix.device)
        
        Q_matrix[:N, :N] = coulomb_matrix
        Q_matrix[:N, N] = 1.0  # Broadcasting works
        Q_matrix[N, :N] = 1.0  # Broadcasting works

        X_vector = torch.zeros((N + 1, 1), dtype=torch.float64, device=x_vector.device)

        X_vector[:N] = x_vector

        if isinstance(self.tot_charge, (int, float)):
            X_vector[N, 0] = torch.tensor(self.tot_charge, dtype=torch.float64, device=x_vector.device)
        else:
            X_vector[N, 0] = self.tot_charge


        q_vector = torch.linalg.solve(Q_matrix, X_vector)
        q = q_vector[:N].clone()  # shape (N, 1)
        # lambda_value = q_vector[N].clone() # scalar
        return q

    def energy_calculation(self, xyz):
        x_vector = self.get_x_vector(xyz)
        coulomb_matrix = self.get_coulomb_matrix(xyz)
        q_vector = self.get_q(coulomb_matrix, x_vector)
        energy = torch.matmul(q_vector.T, (0.5 * torch.matmul(coulomb_matrix, q_vector) - x_vector))
        return energy

    def q_calculation(self, xyz):
        x_vector = self.get_x_vector(xyz)
        coulomb_matrix = self.get_coulomb_matrix(xyz)
        q_vector = self.get_q(coulomb_matrix, x_vector)
        return q_vector

    def energy(self, xyz): # xyz: numpy array, shape (N,3)
        xyz = torch.tensor(xyz, dtype=torch.float64)
        energy = self.energy_calculation(xyz)
        return energy.item()
    
    def gradient(self, xyz): # xyz: numpy array, shape (N,3)
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        energy = self.energy_calculation(xyz)
        gradient = torch.func.jacrev(self.energy_calculation)(xyz)
        gradient = gradient.reshape(xyz.shape[0], 3)
        return energy, gradient
    
    def hessian(self, xyz): # xyz: numpy array, shape (N,3)
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        energy = self.energy_calculation(xyz)
        hessian = torch.func.hessian(self.energy_calculation)(xyz)
        hessian = hessian.reshape(xyz.shape[0] * 3, xyz.shape[0] * 3)
        
        return energy, hessian
    
    def eeq_charge(self, xyz):
        q_vector = self.q_calculation(torch.tensor(xyz, dtype=torch.float64, requires_grad=True))
        return q_vector 
    
    def d_eeq_charge_d_xyz(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        q_vector = self.q_calculation(xyz)
        dq_dxyz = torch.func.jacrev(self.q_calculation)(xyz)
        dq_dxyz = dq_dxyz.reshape(xyz.shape[0], xyz.shape[0], 3)
        return q_vector, dq_dxyz
    
    def d2_eeq_charge_d_xyz2(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        q_vector = self.q_calculation(xyz)
        d2q_dxyz2 = torch.func.hessian(self.q_calculation)(xyz)
        d2q_dxyz2 = d2q_dxyz2.reshape(len(self.element_list), xyz.shape[0] * 3, xyz.shape[0] * 3)
        
        return q_vector, d2q_dxyz2
    
    def cn(self, xyz):
        cn_mod = self.get_cn_modified(torch.tensor(xyz, dtype=torch.float64, requires_grad=True))
        return cn_mod  
    
    def d_cn_d_xyz(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        cn_mod = self.get_cn_modified(xyz)
        dcn_dxyz = torch.func.jacrev(self.get_cn_modified)(xyz)
        dcn_dxyz = dcn_dxyz.reshape(xyz.shape[0], xyz.shape[0], 3)
        return cn_mod, dcn_dxyz
    
    def d2_cn_d_xyz2(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        cn_mod = self.get_cn_modified(xyz)
        d2cn_dxyz2 = torch.func.hessian(self.get_cn_modified)(xyz)
        d2cn_dxyz2 = d2cn_dxyz2.reshape(len(self.element_list), xyz.shape[0] * 3, xyz.shape[0] * 3)

        return cn_mod, d2cn_dxyz2