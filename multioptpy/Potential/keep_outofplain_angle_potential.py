from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_outofplain_angle_from_vec
import torch
import math

class StructKeepOutofPlainAnglePotential:
    """
    Class for calculating Out-of-Plane (Wilson) angle potential with robust singularity handling.
    
    Singularity Handling:
    The Out-of-Plane angle measures the elevation of vector a1 from the plane defined by a2 and a3.
    A singularity occurs when vectors a2 and a3 are collinear (angle 0 or 180).
    In this case, the reference plane is undefined, and the normal vector vanishes.
    
    This implementation applies a 'Collinearity Guard' to force gradients to zero 
    when the reference plane is undefined.
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold for plane definition stability (Squared Norm of cross product)
        # If the cross product of base vectors is smaller than this, 
        # the plane is considered undefined.
        self.COLLINEAR_CUT_SQ = 1e-8
        
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates Out-of-Plane angle energy.
        
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias parameters [k, theta_0]
            
        Definition:
            Center atom: i (index 0 in pair list)
            Neighbors: j, k, l (indices 1, 2, 3)
            Vectors: a1 = r_j - r_i
                     a2 = r_k - r_i
                     a3 = r_l - r_i
            
            Angle is the deviation of a1 from the plane spanned by a2 and a3.
        """
        
        # ========================================
        # 1. Parameter Retrieval
        # ========================================
        if len(bias_pot_params) == 0:
            k = self.config["keep_out_of_plain_angle_spring_const"]
            theta_0_deg = torch.tensor(self.config["keep_out_of_plain_angle_angle"])
            theta_0 = torch.deg2rad(theta_0_deg)
        else:
            k = bias_pot_params[0]
            theta_0_deg = bias_pot_params[1]
            if isinstance(theta_0_deg, torch.Tensor):
                theta_0 = torch.deg2rad(theta_0_deg)
            else:
                theta_0 = torch.deg2rad(torch.tensor(theta_0_deg))

        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        theta_0 = theta_0.to(device=device, dtype=dtype)

        # ========================================
        # 2. Vector Calculations
        # ========================================
        # Indices are 1-based in config
        # Atom 0 is the Central Atom based on the vector definition in your snippet
        c_idx = self.config["keep_out_of_plain_angle_atom_pairs"][0] - 1
        i1_idx = self.config["keep_out_of_plain_angle_atom_pairs"][1] - 1
        i2_idx = self.config["keep_out_of_plain_angle_atom_pairs"][2] - 1
        i3_idx = self.config["keep_out_of_plain_angle_atom_pairs"][3] - 1
        
        center_pos = geom_num_list[c_idx]
        
        # Vectors from Center to Neighbors
        a1 = geom_num_list[i1_idx] - center_pos # The vector to measure (Probe)
        a2 = geom_num_list[i2_idx] - center_pos # Plane definition vector 1
        a3 = geom_num_list[i3_idx] - center_pos # Plane definition vector 2
        
        # ========================================
        # 3. Plane Definition & Singularity Guard
        # ========================================
        # The plane is defined by a2 and a3.
        # Normal vector n = a2 x a3
        n = torch.linalg.cross(a2, a3)
        
        # Check squared norm of normal vector
        # If n is near zero, a2 and a3 are collinear -> Plane Undefined
        n_sq_norm = torch.sum(n**2, dim=-1)
        
        # Guard Mask
        is_undefined_plane = (n_sq_norm < self.COLLINEAR_CUT_SQ)
        
        # Safe normalization
        n_norm = torch.sqrt(n_sq_norm)
        n_hat_demon = torch.clamp(n_norm.unsqueeze(-1), min=1e-12)
        n_hat = n / n_hat_demon
        
        # ========================================
        # 4. Angle Calculation (Robust atan2)
        # ========================================
        # We want the angle 'phi' between a1 and the plane (a2, a3).
        # This is equivalent to 90 - angle(a1, n), or simply:
        # sin(phi) = (a1 . n_hat) / |a1|
        
        # Height of a1 relative to the plane (Projection onto normal)
        # h = |a1| * sin(phi)
        h = torch.sum(a1 * n_hat, dim=-1)
        
        # Length of a1
        a1_norm = torch.linalg.norm(a1) # add epsilon if atom overlap is a concern
        
        # Projected length of a1 onto the plane
        # r_proj = |a1| * cos(phi) = sqrt(|a1|^2 - h^2)
        # We clamp inside sqrt to avoid negative values due to numerical noise
        r_proj_sq = a1_norm**2 - h**2
        r_proj = torch.sqrt(torch.clamp(r_proj_sq, min=0.0))
        
        # Calculate angle using atan2(y, x) = atan2(height, projected_distance)
        # This is valid for -90 to +90 degrees and stable at 90.
        # Note: If a1 is perpendicular to plane, r_proj is 0, atan2(h, 0) gives +/- 90 correctly.
        angle = torch.atan2(h, r_proj)
        
        # ========================================
        # 5. Energy & Clamping
        # ========================================
        
        # Difference from equilibrium
        # Out-of-plane angles are usually non-periodic (limited to -90 to 90), 
        # but if using generalized definition, simple subtraction is usually sufficient.
        diff = angle - theta_0
        
        energy_harmonic = 0.5 * k * diff**2
        
        # Apply Guard:
        # If plane is undefined (collinear base vectors), force energy/force to 0.
        energy = torch.where(is_undefined_plane, torch.tensor(0.0, device=device, dtype=dtype), energy_harmonic)
        
        return energy
       
class StructKeepOutofPlainAnglePotentialv2:
    """
    Class for calculating Out-of-Plane (Wilson) angle potential for fragment centroids
    with robust singularity handling.
    
    This class calculates the angle of vector a1 (Frag1->Frag2) out of the plane 
    defined by vectors a2 (Frag1->Frag3) and a3 (Frag1->Frag4).
    
    Singularity Handling:
    - Plane Undefined: If Frag1, Frag3, and Frag4 are collinear, the reference plane 
      cannot be defined (normal vector vanishes). A 'Collinearity Guard' forces 
      gradients to zero in this region.
    - Vertical Instability: Uses atan2(h, r_proj) instead of asin(h/r) to maintain 
      numerical stability when the angle approaches +/- 90 degrees.
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold for plane definition stability (Squared Norm of cross product)
        # If the cross product of plane-defining vectors is smaller than this, 
        # the plane is considered undefined.
        self.COLLINEAR_CUT_SQ = 1e-8
        
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates Out-of-Plane angle energy for fragments.
        
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias parameters [k, theta_0]
        """
        
        # ========================================
        # 1. Parameter Retrieval
        # ========================================
        if len(bias_pot_params) == 0:
            k = self.config["keep_out_of_plain_angle_v2_spring_const"]
            theta_0_deg = torch.tensor(self.config["keep_out_of_plain_angle_v2_angle"])
            theta_0 = torch.deg2rad(theta_0_deg)
        else:
            k = bias_pot_params[0]
            theta_0_deg = bias_pot_params[1]
            if isinstance(theta_0_deg, torch.Tensor):
                theta_0 = torch.deg2rad(theta_0_deg)
            else:
                theta_0 = torch.deg2rad(torch.tensor(theta_0_deg))

        # Device/Dtype handling
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        theta_0 = theta_0.to(device=device, dtype=dtype)

        # ========================================
        # 2. Vector Calculations (Fragment Centroids)
        # ========================================
        
        # Helper to get indices
        def get_indices(key):
            return torch.tensor(self.config[key], device=device, dtype=torch.long) - 1

        # Calculate centroids (Frag 1 is the Vertex/Center)
        fragm_1_center = torch.mean(geom_num_list[get_indices("keep_out_of_plain_angle_v2_fragm1")], dim=0)
        fragm_2_center = torch.mean(geom_num_list[get_indices("keep_out_of_plain_angle_v2_fragm2")], dim=0)
        fragm_3_center = torch.mean(geom_num_list[get_indices("keep_out_of_plain_angle_v2_fragm3")], dim=0)
        fragm_4_center = torch.mean(geom_num_list[get_indices("keep_out_of_plain_angle_v2_fragm4")], dim=0)

        # Define vectors originating from Fragment 1
        # a1: The "Probe" vector (whose angle we are measuring)
        # a2, a3: The "Base" vectors (defining the reference plane)
        a1 = fragm_2_center - fragm_1_center
        a2 = fragm_3_center - fragm_1_center
        a3 = fragm_4_center - fragm_1_center
        
        # ========================================
        # 3. Plane Definition & Singularity Guard
        # ========================================
        
        # Normal vector to the plane spanned by a2 and a3
        n = torch.linalg.cross(a2, a3)
        
        # Check if plane is defined (a2 and a3 are not collinear)
        n_sq_norm = torch.sum(n**2, dim=-1)
        is_undefined_plane = (n_sq_norm < self.COLLINEAR_CUT_SQ)
        
        # Safe normalization
        n_norm = torch.sqrt(n_sq_norm)
        n_hat_demon = torch.clamp(n_norm.unsqueeze(-1), min=1e-12)
        n_hat = n / n_hat_demon
        
        # ========================================
        # 4. Angle Calculation (Robust atan2)
        # ========================================
        
        # Height of a1 relative to the plane (Projection onto normal)
        # h = |a1| * sin(phi)
        h = torch.sum(a1 * n_hat, dim=-1)
        
        # Length of a1
        a1_norm = torch.linalg.norm(a1)
        
        # Projected length of a1 onto the plane
        # r_proj = sqrt(|a1|^2 - h^2)
        # Clamp to avoid negative sqrt due to numerical noise
        r_proj_sq = a1_norm**2 - h**2
        r_proj = torch.sqrt(torch.clamp(r_proj_sq, min=0.0))
        
        # Calculate angle using atan2(height, projected_distance)
        # Valid range: -90 to +90 degrees (or -pi/2 to pi/2)
        angle = torch.atan2(h, r_proj)
        
        # ========================================
        # 5. Energy & Clamping
        # ========================================
        
        diff = angle - theta_0
        
        energy_harmonic = 0.5 * k * diff**2
        
        # Apply Guard:
        # If the reference plane is undefined, force energy to 0.0 to prevent gradient explosion.
        energy = torch.where(is_undefined_plane, torch.tensor(0.0, device=device, dtype=dtype), energy_harmonic)
        
        return energy