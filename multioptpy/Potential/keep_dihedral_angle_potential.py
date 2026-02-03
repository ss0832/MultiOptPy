from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_dihedral_angle_from_vec
import torch
import math

class StructKeepDihedralAnglePotential:
    """
    Computes the harmonic potential energy for a dihedral angle defined by four atoms.
    
    This class incorporates a robust singularity handling mechanism for cases where
    three consecutive atoms become collinear (i.e., the bond angle approaches 0 or 180 degrees).
    In such configurations, the dihedral angle is mathematically undefined, leading to 
    numerical instabilities and exploding gradients.

    To resolve this, a smooth switching function (smoothstep) is applied to the squared norm 
    of the cross products of the bond vectors. This function smoothly attenuates the 
    potential energy and forces to zero as the geometry enters the collinear region.

    Energy Function:
        E_total = E_harmonic * S(|n1|^2) * S(|n2|^2)
    
    Where:
        E_harmonic = 0.5 * k * (phi - phi_0)^2
        S(x) is a cubic Hermite interpolation function (smoothstep) mapping [0, 1].
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Thresholds for collinearity smoothing (Squared Norm of Cross Product).
        # These values determine the range over which the potential is switched off.
        # COLLINEAR_CUT_MIN: Below this value, the weighting factor is 0.0.
        # COLLINEAR_CUT_MAX: Above this value, the weighting factor is 1.0.
        # A squared norm of 1e-8 corresponds to a sine of approx 1e-4.
        self.COLLINEAR_CUT_MIN = 1e-10
        self.COLLINEAR_CUT_MAX = 1e-8
        
        return

    def _compute_switching(self, val):
        """
        Computes the smoothstep switching factor.

        Args:
            val (Tensor): The squared norm of the cross product vector.

        Returns:
            Tensor: A scalar factor between 0.0 and 1.0 using cubic interpolation.
                    Returns 0.0 if val < min, 1.0 if val > max.
        """
        t = (val - self.COLLINEAR_CUT_MIN) / (self.COLLINEAR_CUT_MAX - self.COLLINEAR_CUT_MIN)
        t = torch.clamp(t, 0.0, 1.0)
        # Smoothstep function: 3t^2 - 2t^3
        return t * t * (3.0 - 2.0 * t)

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates the potential energy with collinearity smoothing.

        Args:
            geom_num_list (Tensor): Atomic coordinates tensor of shape (N_atoms, 3).
            bias_pot_params (list, optional): Optional override for parameters [k, phi_0].

        Returns:
            Tensor: The calculated potential energy (scalar).
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
        i1 = self.config["keep_dihedral_angle_atom_pairs"][0] - 1
        i2 = self.config["keep_dihedral_angle_atom_pairs"][1] - 1
        i3 = self.config["keep_dihedral_angle_atom_pairs"][2] - 1
        i4 = self.config["keep_dihedral_angle_atom_pairs"][3] - 1
        
        b1 = geom_num_list[i2] - geom_num_list[i1]
        b2 = geom_num_list[i3] - geom_num_list[i2]
        b3 = geom_num_list[i4] - geom_num_list[i3]
        
        # Normal vectors to the planes defined by bond pairs
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        
        # ========================================
        # 3. Collinearity Guard (Singularity Smoothing)
        # ========================================
        n1_sq_norm = torch.sum(n1**2, dim=-1)
        n2_sq_norm = torch.sum(n2**2, dim=-1)
        
        # Compute switching factors to dampen energy in collinear regions
        switch_1 = self._compute_switching(n1_sq_norm)
        switch_2 = self._compute_switching(n2_sq_norm)
        
        # Safe normalization for angle calculation.
        # Clamping prevents NaN in the graph, while the switching factor ensures
        # that the resulting gradients in the clamped region are zeroed out.
        n1_norm = torch.clamp(torch.sqrt(n1_sq_norm), min=1e-12)
        n2_norm = torch.clamp(torch.sqrt(n2_sq_norm), min=1e-12)
        
        n1_hat = n1 / (n1_norm.unsqueeze(-1))
        n2_hat = n2 / (n2_norm.unsqueeze(-1))
        
        b2_norm = torch.clamp(torch.linalg.norm(b2), min=1e-12)
        b2_hat = b2 / (b2_norm.unsqueeze(-1))
        
        # ========================================
        # 4. Angle Calculation
        # ========================================
        x = torch.sum(n1_hat * n2_hat, dim=-1)
        # (n1 x n2) is parallel to b2
        m1 = torch.linalg.cross(n1_hat, n2_hat)
        y = torch.sum(m1 * b2_hat, dim=-1)
        
        phi = torch.atan2(y, x)
        
        # ========================================
        # 5. Energy Calculation with Smoothing
        # ========================================
        diff = phi - phi_0
        # Wrap difference to [-pi, pi]
        diff = diff - 2.0 * PI * torch.round(diff / (2.0 * PI))
        
        raw_energy = 0.5 * k * diff**2
        
        # Apply smoothing factors
        energy = raw_energy * switch_1 * switch_2
        
        return energy


class StructKeepDihedralAnglePotentialv2:
    """
    Computes the dihedral angle potential energy defined by the centroids of four atom fragments.
    
    This class is designed for coarse-grained constraints where the dihedral is defined
    by the geometric centers of specified groups of atoms rather than single atoms.
    It includes the same singularity smoothing mechanism as the standard atom-based potential
    to handle cases where the fragment centroids become collinear.
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Thresholds for collinearity smoothing
        self.COLLINEAR_CUT_MIN = 1e-10
        self.COLLINEAR_CUT_MAX = 1e-8
        
        return

    def _compute_switching(self, val):
        """
        Computes the smoothstep switching factor.
        """
        t = (val - self.COLLINEAR_CUT_MIN) / (self.COLLINEAR_CUT_MAX - self.COLLINEAR_CUT_MIN)
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates the potential energy for fragment centroids with collinearity smoothing.
        """
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

        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        phi_0 = phi_0.to(device=device, dtype=dtype)

        # Vector Calculations for Fragment Centers
        def get_indices(key):
            return torch.tensor(self.config[key], device=device, dtype=torch.long) - 1

        fragm_1_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm1")], dim=0)
        fragm_2_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm2")], dim=0)
        fragm_3_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm3")], dim=0)
        fragm_4_center = torch.mean(geom_num_list[get_indices("keep_dihedral_angle_v2_fragm4")], dim=0)
           
        b1 = fragm_2_center - fragm_1_center
        b2 = fragm_3_center - fragm_2_center
        b3 = fragm_4_center - fragm_3_center

        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)

        # Collinearity Guard (Smoothing)
        n1_sq_norm = torch.sum(n1**2, dim=-1)
        n2_sq_norm = torch.sum(n2**2, dim=-1)
        
        switch_1 = self._compute_switching(n1_sq_norm)
        switch_2 = self._compute_switching(n2_sq_norm)
        
        n1_norm = torch.clamp(torch.sqrt(n1_sq_norm), min=1e-12)
        n2_norm = torch.clamp(torch.sqrt(n2_sq_norm), min=1e-12)
        
        n1_hat = n1 / (n1_norm.unsqueeze(-1))
        n2_hat = n2 / (n2_norm.unsqueeze(-1))
        
        b2_norm = torch.clamp(torch.linalg.norm(b2), min=1e-12)
        b2_hat = b2 / (b2_norm.unsqueeze(-1))

        # Angle Calculation
        x = torch.sum(n1_hat * n2_hat, dim=-1)
        m1 = torch.linalg.cross(n1_hat, n2_hat)
        y = torch.sum(m1 * b2_hat, dim=-1)
        
        phi = torch.atan2(y, x)

        # Energy Calculation
        diff = phi - phi_0
        diff = diff - 2.0 * PI * torch.round(diff / (2.0 * PI))
        
        raw_energy = 0.5 * k * diff**2
        
        # Apply Smoothing
        energy = raw_energy * switch_1 * switch_2
        
        return energy
    
class StructKeepDihedralAnglePotentialCos:
    """
    Computes a cosine-based dihedral potential energy for fragment centroids.
    
    Includes singularity smoothing to prevent gradient explosion when fragment centers
    become collinear.
    """
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        self.COLLINEAR_CUT_MIN = 1e-10
        self.COLLINEAR_CUT_MAX = 1e-8
        return

    def _compute_switching(self, val):
        """
        Computes the smoothstep switching factor.
        """
        t = (val - self.COLLINEAR_CUT_MIN) / (self.COLLINEAR_CUT_MAX - self.COLLINEAR_CUT_MIN)
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates the cosine-based potential energy.
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

        # Explicitly compute cross products to determine switching factors
        n1 = torch.linalg.cross(a1, a2)
        n2 = torch.linalg.cross(a2, a3)
        n1_sq_norm = torch.sum(n1**2, dim=-1)
        n2_sq_norm = torch.sum(n2**2, dim=-1)

        switch_1 = self._compute_switching(n1_sq_norm)
        switch_2 = self._compute_switching(n2_sq_norm)

        angle = torch_calc_dihedral_angle_from_vec(a1, a2, a3)
        raw_energy = 0.5 * potential_const * (1.0 - 1 * torch.cos(angle_const * angle - (torch.deg2rad(torch.tensor(self.config["keep_dihedral_angle_cos_angle"])))))
        
        # Apply smoothing
        energy = raw_energy * switch_1 * switch_2
              
        return energy