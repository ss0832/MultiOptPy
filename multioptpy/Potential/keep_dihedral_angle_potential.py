
from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_dihedral_angle_from_vec
import torch
import math



class StructKeepDihedralAnglePotential:
    """
    Class for calculating dihedral angle potential energy with robust singularity handling.
    
    This class handles the geometric singularity where constituent bond angles become linear.
    In such cases, the dihedral angle is undefined, and gradients normally explode.
    This implementation clamps the potential/gradient to zero (or a finite value) 
    in these collinear regions to prevents NaN/Inf.

    Energy Function: E = 0.5 * k * (phi - phi_0)^2
    (Where phi is the signed dihedral angle in radians, [-pi, pi])
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold for collinearity check
        # If the sine of a constituent bond angle is smaller than this,
        # the dihedral is considered undefined.
        # 1e-4 corresponds to bond angles approx < 0.0057 deg or > 179.99 deg.
        self.COLLINEAR_CUT = 1e-4
        
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates dihedral potential energy.
        
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias parameters [k, phi_0]
        
        Returns:
            energy: Scalar tensor
            
        Methodology:
            1. Compute bond vectors b1, b2, b3.
            2. Compute normal vectors n1 = b1 x b2, n2 = b2 x b3.
            3. **Safeguard**: Check norms of n1 and n2. If atoms are collinear 
               (|n| ~ 0), the dihedral is undefined. Mask these regions to 0 energy/force.
            4. Use atan2 for stable angle calculation across full range [-pi, pi].
            5. Handle periodicity for the difference (phi - phi_0).
        """
        
        # ========================================
        # 1. Parameter Retrieval
        # ========================================
        if len(bias_pot_params) == 0:
            k = self.config["keep_dihedral_angle_spring_const"]
            phi_0_deg = torch.tensor(self.config["keep_dihedral_angle_angle"])
            phi_0 = torch.deg2rad(phi_0_deg)
        else:
            k = bias_pot_params[0]
            phi_0_deg = bias_pot_params[1]
            if isinstance(phi_0_deg, torch.Tensor):
                phi_0 = torch.deg2rad(phi_0_deg)
            else:
                phi_0 = torch.deg2rad(torch.tensor(phi_0_deg))

        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        phi_0 = phi_0.to(device=device, dtype=dtype)

        # ========================================
        # 2. Vector Calculations
        # ========================================
        # Indices are 1-based in config
        i1 = self.config["keep_dihedral_angle_atom_pairs"][0] - 1
        i2 = self.config["keep_dihedral_angle_atom_pairs"][1] - 1
        i3 = self.config["keep_dihedral_angle_atom_pairs"][2] - 1
        i4 = self.config["keep_dihedral_angle_atom_pairs"][3] - 1
        
        # Bond vectors: b1(1->2), b2(2->3), b3(3->4)
        b1 = geom_num_list[i2] - geom_num_list[i1]
        b2 = geom_num_list[i3] - geom_num_list[i2]
        b3 = geom_num_list[i4] - geom_num_list[i3]
        
        # Normal vectors to the planes defined by bonds
        # n1 is normal to plane (1,2,3)
        # n2 is normal to plane (2,3,4)
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        
        # ========================================
        # 3. Collinearity Guard (Singularity Handling)
        # ========================================
        # The norm of the cross product |a x b| = |a||b|sin(theta).
        # If bond angle theta -> 0 or 180, this norm -> 0.
        # Dividing by this norm causes NaN and Infinite Gradients.
        
        n1_sq_norm = torch.sum(n1**2, dim=-1)
        n2_sq_norm = torch.sum(n2**2, dim=-1)
        
        # Threshold squared for comparison
        cut_sq = self.COLLINEAR_CUT ** 2
        
        # Mask: True if EITHER plane is undefined (atoms are collinear)
        is_linear = (n1_sq_norm < cut_sq) | (n2_sq_norm < cut_sq)
        
        # Normalize vectors safely
        # We add a small epsilon to denominator to avoid NaN in the "linear" branch,
        # even though we will overwrite the result with 0 later.
        n1_norm = torch.clamp(torch.sqrt(n1_sq_norm), min=1e-12)
        n2_norm = torch.clamp(torch.sqrt(n2_sq_norm), min=1e-12)
        
       
        n1_hat = n1 / (n1_norm.unsqueeze(-1))
        n2_hat = n2 / (n2_norm.unsqueeze(-1))
        
        # Normalize b2 to define the reference frame for sign
        b2_norm = torch.clamp(torch.linalg.norm(b2), min=1e-12)
        b2_hat = b2 / (b2_norm.unsqueeze(-1))
        
        # ========================================
        # 4. Angle Calculation (atan2)
        # ========================================
        # Using atan2 is numerically stable for phi -> 0 or pi.
        # x = cos(phi) = n1_hat . n2_hat
        # y = sin(phi) = (n1_hat x n2_hat) . b2_hat
        
        x = torch.sum(n1_hat * n2_hat, dim=-1)
        
        # (n1 x n2) is parallel to b2
        m1 = torch.linalg.cross(n1_hat, n2_hat)
        y = torch.sum(m1 * b2_hat, dim=-1)
        
        phi = torch.atan2(y, x)
        
        # ========================================
        # 5. Energy Calculation with Periodicity
        # ========================================
        
        # Calculate difference and wrap to [-pi, pi]
        diff = phi - phi_0
        diff = diff - 2.0 * PI * torch.round(diff / (2.0 * PI))
        
        # Standard harmonic potential
        energy = 0.5 * k * diff**2
        
        # ========================================
        # 6. Apply Collinearity Guard
        # ========================================
        # If the geometry is collinear (undefined dihedral), force energy/gradient to 0 (or constant).
        # This acts as the "Linear Extrapolation/Clamping" for the singularity.
        # The bond angle potential (separate term) is responsible for pushing atoms 
        # out of linearity; this term simply prevents crashing.
        
        energy = torch.where(is_linear, torch.tensor(0.0, device=device, dtype=dtype), energy)
        
        return energy


class StructKeepDihedralAnglePotentialv2:
    """
    Class for calculating dihedral angle potential energy between fragment centers
    with robust singularity handling.
    
    This class calculates the dihedral angle formed by the geometric centers (centroids)
    of four specified atom groups (fragments).
    
    Singularity Handling:
    It handles the geometric singularity where the fragment centers become collinear 
    (bond angles between centers approach 0 or 180 degrees). In these regions, 
    the dihedral plane is undefined, causing gradient explosion. 
    A 'Collinearity Guard' is applied to force gradients to zero in these regions.
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold for collinearity check (Squared Norm)
        # 1e-4 corresponds to bond angles approx < 0.0057 deg or > 179.99 deg.
        self.COLLINEAR_CUT_SQ = 1e-8  # (1e-4)^2
        
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates dihedral potential energy for fragment centers.
        
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias parameters
                bias_pot_params[0] : keep_dihedral_angle_v2_spring_const
                bias_pot_params[1] : keep_dihedral_angle_v2_angle
        """
        
        # ========================================
        # 1. Parameter Retrieval
        # ========================================
        if len(bias_pot_params) == 0:
            k = self.config["keep_dihedral_angle_v2_spring_const"]
            phi_0_deg = torch.tensor(self.config["keep_dihedral_angle_v2_angle"])
            phi_0 = torch.deg2rad(phi_0_deg)
        else:
            k = bias_pot_params[0]
            phi_0_deg = bias_pot_params[1]
            if isinstance(phi_0_deg, torch.Tensor):
                phi_0 = torch.deg2rad(phi_0_deg)
            else:
                phi_0 = torch.deg2rad(torch.tensor(phi_0_deg))

        # Device and dtype handling
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        phi_0 = phi_0.to(device=device, dtype=dtype)

        # ========================================
        # 2. Vector Calculations (Fragment Centers)
        # ========================================
        
        # Helper to get indices on device
        def get_indices(key):
            return torch.tensor(self.config[key], device=device, dtype=torch.long) - 1

        # Calculate centroids for each fragment
        # torch.mean preserves gradients, allowing backprop to individual atoms
        fragm_1_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm1")], dim=0)
        fragm_2_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm2")], dim=0)
        fragm_3_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm3")], dim=0)
        fragm_4_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm4")], dim=0)
           
        # Bond vectors between centers: b1(1->2), b2(2->3), b3(3->4)
        b1 = fragm_2_center - fragm_1_center
        b2 = fragm_3_center - fragm_2_center
        b3 = fragm_4_center - fragm_3_center

        # Normal vectors to the planes defined by fragment centers
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)

        # ========================================
        # 3. Collinearity Guard (Robustness Step)
        # ========================================
        
        n1_sq_norm = torch.sum(n1**2, dim=-1)
        n2_sq_norm = torch.sum(n2**2, dim=-1)
        
        # Mask: True if fragment centers are collinear (undefined dihedral)
        is_linear = (n1_sq_norm < self.COLLINEAR_CUT_SQ) | (n2_sq_norm < self.COLLINEAR_CUT_SQ)
        
        # Safe normalization
        n1_norm = torch.clamp(torch.sqrt(n1_sq_norm), min=1e-12)
        n2_norm = torch.clamp(torch.sqrt(n2_sq_norm), min=1e-12)
        
        n1_hat = n1 / (n1_norm.unsqueeze(-1))
        n2_hat = n2 / (n2_norm.unsqueeze(-1))
        
        b2_norm = torch.clamp(torch.linalg.norm(b2), min=1e-12)
        b2_hat = b2 / (b2_norm.unsqueeze(-1))

        # ========================================
        # 4. Angle Calculation (atan2)
        # ========================================
        
        # x = cos(phi)
        x = torch.sum(n1_hat * n2_hat, dim=-1)
        
        # y = sin(phi)
        m1 = torch.linalg.cross(n1_hat, n2_hat)
        y = torch.sum(m1 * b2_hat, dim=-1)
        
        phi = torch.atan2(y, x)

        # ========================================
        # 5. Energy Calculation & Clamping
        # ========================================
        
        # Handle periodicity [-pi, pi]
        diff = phi - phi_0
        diff = diff - 2.0 * PI * torch.round(diff / (2.0 * PI))
        
        energy_harmonic = 0.5 * k * diff**2
        
        # Apply Guard: Force energy to 0.0 if geometry is undefined
        energy = torch.where(is_linear, torch.tensor(0.0, device=device, dtype=dtype), energy_harmonic)
        
        return energy
    
class StructKeepDihedralAnglePotentialCos:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["keep_dihedral_angle_cos_potential_const"],
                             self.config["keep_dihedral_angle_cos_angle_const"], 
                             self.config["keep_dihedral_angle_cos_angle"], 
                             self.config["keep_dihedral_angle_cos_fragm1"],
                             self.config["keep_dihedral_angle_cos_fragm2"],
                             self.config["keep_dihedral_angle_cos_fragm3"],
                             self.config["keep_dihedral_angle_cos_fragm4"],
                             
        """
        potential_const = float(self.config["keep_dihedral_angle_cos_potential_const"])
        angle_const = float(self.config["keep_dihedral_angle_cos_angle_const"])
        
        
        fragm_1_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_dihedral_angle_cos_fragm1"]) - 1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_dihedral_angle_cos_fragm2"]) - 1], dim=0)
        fragm_3_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_dihedral_angle_cos_fragm3"]) - 1], dim=0)
        fragm_4_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_dihedral_angle_cos_fragm4"]) - 1], dim=0)
            
        a1 = fragm_2_center - fragm_1_center
        a2 = fragm_3_center - fragm_2_center
        a3 = fragm_4_center - fragm_3_center

        angle = torch_calc_dihedral_angle_from_vec(a1, a2, a3)
        energy = 0.5 * potential_const * (1.0 -1* torch.cos(angle_const * angle - (torch.deg2rad(torch.tensor(self.config["keep_dihedral_angle_cos_angle"])))))
              
        return energy #hartree