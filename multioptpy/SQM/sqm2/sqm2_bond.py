import torch


class BondCalculator:
    def __init__(self, element_list, params):
        self.element_list = element_list
        
        #self.k_bond_list = []
        self.r0_bond_list = []
        self.paulingEN_list = []
        
        for element in element_list:
            #self.k_bond_list.append(params.k_bond[element])
            self.r0_bond_list.append(params.eeq_covalent_radii[element])
            self.paulingEN_list.append(params.paulingEN[element])
            
        #self.k_bond_list = torch.tensor(self.k_bond_list, dtype=torch.float64)
        self.r0_bond_list = torch.tensor(self.r0_bond_list, dtype=torch.float64)
        self.paulingEN_list = torch.tensor(self.paulingEN_list, dtype=torch.float64)

        self.en_polynominal_param = params.en_polynominal_param
        self._calc_corrected_r0_and_en_diff2_matrix()
        self.eta = params.eta_bond
        self.k_en = params.k_en_bond
        
        return

    def get_en_poly_index(self, atomic_number):
        if not isinstance(atomic_number, int) or atomic_number <= 0:
            raise ValueError("Atomic number must be a positive integer.")
        if atomic_number <= 2:   # Z = 1-2 (H, He)
            return 0  
        elif atomic_number <= 10:  # Z = 3-10 (Li-Ne)
            return 1
        elif atomic_number <= 18:  # Z = 11-18 (Na-Ar)
            return 2
        elif atomic_number <= 36:  # Z = 19-36 (K-Kr)
            return 3
        elif atomic_number <= 54:  # Z = 37-54 (Rb-Xe)
            return 4
        else:                      # Z = 55- (Cs-)
            return 5

    def _calc_corrected_r0_and_en_diff2_matrix(self):
        en_diff2_matrix = torch.zeros((len(self.element_list), len(self.element_list)), dtype=torch.float64)
        corrected_r0_matrix = torch.zeros((len(self.element_list), len(self.element_list)), dtype=torch.float64)
        for i in range(len(self.element_list)):
            for j in range(len(self.element_list)):
                r0_ij = self.r0_bond_list[i] + self.r0_bond_list[j]
                
                c_i = self.en_polynominal_param[self.get_en_poly_index(int(self.element_list[i])+1)]
                c_j = self.en_polynominal_param[self.get_en_poly_index(int(self.element_list[j])+1)]
                c = 0.5 * (c_i + c_j)
                c_1 = c[0]
                c_2 = c[1]

                en_diff = torch.abs(self.paulingEN_list[i] - self.paulingEN_list[j])
                corrected_r0_matrix[i, j] = r0_ij * (1.0 - c_1 * en_diff + c_2 * (en_diff ** 2))
                en_diff2_matrix[i, j] = en_diff ** 2
        
        self.corrected_r0_matrix = corrected_r0_matrix
        self.en_diff2_matrix = en_diff2_matrix
        return corrected_r0_matrix, en_diff2_matrix

    def calculation(self, xyz):  # xyz: (N, 3) torch tensor
        N = xyz.shape[0]
        diff = xyz.unsqueeze(1) - xyz.unsqueeze(0)
        # r is the (N, N) matrix of pairwise distances. Diagonal is 0.0
        r = torch.linalg.norm(diff, dim=-1)

        k_bond_ij = 1.0 #torch.sqrt(self.k_bond_list.unsqueeze(1) * self.k_bond_list.unsqueeze(0))
        r0_ij = self.corrected_r0_matrix
        en_diff2_ij = self.en_diff2_matrix

        # --- Modifications ---
        
        # 1. Calculate the cutoff matrix for bond determination
        cutoff_matrix = 1.25 * r0_ij
        
        # 2. Create a mask for pairs where the current distance r is <= the cutoff
        mask = (r <= cutoff_matrix)
        
        # 3. Always set the diagonal (i=i) to False to exclude self-interaction
        mask.fill_diagonal_(False)

        # 4. Calculate the energy term for all pairs
        energy_term = -1.0 * k_bond_ij * torch.exp(self.eta * (1.0 * self.k_en * en_diff2_ij) * (r - r0_ij) ** 2.0)
        
        # 5. Use torch.where to select the energy term only for masked pairs.
        #    Set 0.0 for False pairs (non-bonded pairs and the diagonal).
        bond_energy_matrix = torch.where(mask, energy_term, 0.0)

        # The previous .fill_diagonal_(0.0) is no longer needed.
        # --- End of modifications ---
        
        energy = torch.sum(bond_energy_matrix) / 2.0
        return energy  # shape: scalar
    
    
    def energy(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64)
        energy = self.calculation(xyz)
        return energy
    
    def gradient(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64)    
        energy = self.calculation(xyz)
        gradient = torch.func.jacrev(self.calculation)(xyz)
        return energy, gradient
    
    def hessian(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float64)
        energy = self.calculation(xyz)
        hessian = torch.func.hessian(self.calculation)(xyz)
        return energy, hessian
    
    
    
    