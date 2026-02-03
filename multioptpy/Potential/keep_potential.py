from multioptpy.Parameters.parameter import UnitValueLib
import torch

class StructKeepPotential:
    """
    Class for calculating harmonic distance potential between two atoms.
    
    Robustness Improvements:
    - Uses robust norm calculation (sqrt(sum^2 + eps)) to avoid NaN gradients 
      if atoms perfectly overlap (r=0).
    - Automatically adapts to the device and dtype of the input geometry.
    """
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_pot_spring_const"], 
                             self.config["keep_pot_distance"], 
                             self.config["keep_pot_atom_pairs"],
        bias_pot_params[0] : keep_pot_spring_const
        bias_pot_params[1] : keep_pot_distance
        """
        # 1. Parameter Retrieval & Device/Dtype handling
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype

        if len(bias_pot_params) == 0:
            k = self.config["keep_pot_spring_const"]
            r0_ang = self.config["keep_pot_distance"]
        else:
            k = bias_pot_params[0]
            r0_ang = bias_pot_params[1]

        # Convert r0 to Bohr (assuming parameter is in Angstroms) and ensure tensor
        if not isinstance(r0_ang, torch.Tensor):
            r0_ang = torch.tensor(r0_ang, device=device, dtype=dtype)
        r0 = r0_ang / self.bohr2angstroms

        # 2. Vector Calculation
        # Indices are 1-based in config
        idx1 = self.config["keep_pot_atom_pairs"][0] - 1
        idx2 = self.config["keep_pot_atom_pairs"][1] - 1
        
        vec = geom_num_list[idx1] - geom_num_list[idx2]
        
        # 3. Robust Distance Calculation
     
        dist = torch.clamp(torch.sqrt(torch.sum(vec**2)), min=1e-12)
        
        # 4. Energy Calculation
        # E = 0.5 * k * (r - r0)^2
        energy = 0.5 * k * (dist - r0) ** 2
        
        return energy # hartree


class StructKeepPotentialv2:
    """
    Class for calculating harmonic distance potential between two fragment centers.
    
    Robustness Improvements:
    - Uses robust norm calculation.
    - vectorized center of mass calculation.
    """
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
        
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_pot_v2_spring_const"], 
                             self.config["keep_pot_v2_distance"], 
                             self.config["keep_pot_v2_fragm1"],
                             self.config["keep_pot_v2_fragm2"],
        """
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype

        # 1. Parameter Retrieval
        if len(bias_pot_params) == 0:
            k = self.config["keep_pot_v2_spring_const"]
            r0_ang = self.config["keep_pot_v2_distance"]
        else:
            k = bias_pot_params[0]
            r0_ang = bias_pot_params[1]

        if not isinstance(r0_ang, torch.Tensor):
            r0_ang = torch.tensor(r0_ang, device=device, dtype=dtype)
        r0 = r0_ang / self.bohr2angstroms

        # 2. Center Calculation (Vectorized)
        # Convert indices lists to tensors for efficient indexing
        idx1 = torch.tensor(self.config["keep_pot_v2_fragm1"], device=device, dtype=torch.long) - 1
        idx2 = torch.tensor(self.config["keep_pot_v2_fragm2"], device=device, dtype=torch.long) - 1

        fragm_1_center = torch.mean(geom_num_list[idx1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[idx2], dim=0)
        
        # 3. Robust Distance Calculation
        vec = fragm_1_center - fragm_2_center
        distance = torch.clamp(torch.sqrt(torch.sum(vec**2)), min=1e-12)
        
        # 4. Energy
        energy = 0.5 * k * (distance - r0) ** 2
        
        return energy # hartree


class StructKeepPotentialAniso:
    """
    Class for calculating anisotropic distance potential between fragment centers.
    
    Robustness Improvements:
    - Vectorized centroid calculation (replaced inefficient loops).
    - Device/Dtype safety for matrix operations.
    """
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
        
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables:   self.config["aniso_keep_pot_v2_spring_const_mat"]
                                self.config["aniso_keep_pot_v2_dist"] 
                                self.config["aniso_keep_pot_v2_fragm1"]
                                self.config["aniso_keep_pot_v2_fragm2"]
        """
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype

        # 1. Centroid Calculation (Vectorized)
        idx1 = torch.tensor(self.config["aniso_keep_pot_v2_fragm1"], device=device, dtype=torch.long) - 1
        idx2 = torch.tensor(self.config["aniso_keep_pot_v2_fragm2"], device=device, dtype=torch.long) - 1
        
        fragm_1_center = torch.mean(geom_num_list[idx1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[idx2], dim=0)
        
        # 2. Anisotropic Distance Components
        # Distance diff vector: |x1-x2|, |y1-y2|, |z1-z2|
        # Note: Derivative of abs(x) at 0 is 0 in PyTorch subgradient, usually safe enough here.
        xyz_dist = torch.abs(fragm_1_center - fragm_2_center)
        
        # Equilibrium distance logic (from original code)
        # Original: eq_dist = dist / sqrt(3)
        # This implies the target distance is distributed equally among x,y,z components?
        ref_dist_val = self.config["aniso_keep_pot_v2_dist"]
        if not isinstance(ref_dist_val, torch.Tensor):
            ref_dist_val = torch.tensor(ref_dist_val, device=device, dtype=dtype)
            
        eq_dist_component = (ref_dist_val / (3 ** 0.5)) / self.bohr2angstroms
        
        # Squared deviation vector: [(dx - d0)^2, (dy - d0)^2, (dz - d0)^2]
        dist_vec = (xyz_dist - eq_dist_component) ** 2
        dist_vec = dist_vec.unsqueeze(1) # Shape (3, 1)
        
        # 3. Matrix Multiplication
        # Ensure spring constant matrix matches device/dtype
        spring_mat = torch.tensor(self.config["aniso_keep_pot_v2_spring_const_mat"], device=device, dtype=dtype)
        
        vec_pot = torch.matmul(spring_mat, dist_vec)
        
        energy = torch.sum(vec_pot)
        
        return energy # hartree