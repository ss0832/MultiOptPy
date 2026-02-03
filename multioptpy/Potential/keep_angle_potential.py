from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_angle_from_vec

import torch
import math


class StructKeepAnglePotential:
    """
    Class for calculating angle restraint potential energy between atoms.
    
    This class calculates the angle formed by the geometric centers (centroids) of three 
    specified atoms. It ensures numerical stability and gradient 
    accuracy at singularities (theta -> 0, theta -> pi).
    
    Energy Function: E = 0.5 * k * (theta - theta_0)^2
    
    Processing Cases:
    1. theta -> 0, theta_0 != 0 : Quadratic Extrapolation - Maintains positive curvature (Hessian) to prevent optimizer explosion (step size divergence).
    2. theta -> 0, theta_0 ~= 0 : Taylor Expansion - Maintains physical accuracy for near-linear equilibrium.
    3. theta -> pi, theta_0 != pi : Quadratic Extrapolation - Maintains positive curvature (Hessian) to prevent optimizer explosion.
    4. theta -> pi, theta_0 ~= pi : Taylor Expansion - Maintains physical accuracy for near-linear equilibrium.
    5. Normal Region: Standard acos calculation.
    """
    
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold: Below this angle (rad), switch to extrapolation.
        # 1e-3 rad is conservative enough to ensure stability.
        self.THETA_CUT = 1e-3
        
        # Threshold for Equilibrium check
        self.EPSILON_PARAM = 1e-3
        
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias potential parameters
                bias_pot_params[0] : keep_angle_spring_const
                bias_pot_params[1] : keep_angle_angle
        
        Returns:
            energy: Angle potential energy (Scalar Tensor)
        """
        # 1. Parameter Retrieval
        if len(bias_pot_params) == 0:
            k = self.config["keep_angle_spring_const"]
            theta_0_deg = torch.tensor(self.config["keep_angle_angle"])
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
        
        # Constants
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        theta_0 = theta_0.to(device=device, dtype=dtype)
        theta_cut_val = torch.tensor(self.THETA_CUT, device=device, dtype=dtype)
        epsilon_param = torch.tensor(self.EPSILON_PARAM, device=device, dtype=dtype)

        # 2. Vector & Cosine Calculation
        idx1 = self.config["keep_angle_atom_pairs"][0] - 1
        idx2 = self.config["keep_angle_atom_pairs"][1] - 1
        idx3 = self.config["keep_angle_atom_pairs"][2] - 1
        
        vec1 = geom_num_list[idx1] - geom_num_list[idx2]
        vec2 = geom_num_list[idx3] - geom_num_list[idx2]
        
        norm1 = torch.linalg.norm(vec1)
        norm2 = torch.linalg.norm(vec2)
        
        # u = cos(theta)
        u = torch.dot(vec1, vec2) / (norm1 * norm2 + 1e-12)
        u = torch.clamp(u, -1.0, 1.0)

        # 3. Case Branching
        
        # --- Case 2 & 4: Equilibrium at singularity (Linear Molecules) ---
        # Taylor expansion (Physical accuracy)
        if torch.abs(theta_0) < epsilon_param:
            delta = 1.0 - u
            theta_sq = delta * (2.0 + delta * (1.0/3.0 + delta * 4.0/45.0))
            return 0.5 * k * theta_sq

        elif torch.abs(theta_0 - PI) < epsilon_param:
            delta = 1.0 + u
            diff_sq = delta * (2.0 + delta * (1.0/3.0 + delta * 4.0/45.0))
            return 0.5 * k * diff_sq

        # --- Case 1, 3, 5: General Angle ---
        else:
            u_cut_pos = torch.cos(theta_cut_val)
            u_cut_neg = torch.cos(PI - theta_cut_val)
            
            is_singular_0 = (u > u_cut_pos)    # theta -> 0
            is_singular_pi = (u < u_cut_neg)   # theta -> pi
            is_safe = ~(is_singular_0 | is_singular_pi)
            
            # A. Normal Region (Safe acos)
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0) ** 2
            
            # Helper for Quadratic Extrapolation
            # E(u) approx E_cut + E'_cut * (u - u_cut) + 0.5 * E''_cut * (u - u_cut)^2
            def get_quad_params(th_cut, u_bnd):
                # Analytical derivatives at the boundary
                # E = 0.5*k*(th - th0)^2
                # dE/du = k(th - th0) * dth/du
                # d2E/du2 = k * (dth/du)^2 + k(th - th0) * d2th/du2
                
                # th_cut is scalar tensor
                sin_cut = torch.sin(th_cut)
                cos_cut = torch.cos(th_cut)
                
                # Derivatives of theta w.r.t u = cos(theta)
                # dth/du = -1/sin(th)
                dth_du = -1.0 / sin_cut
                # d2th/du2 = -cos(th) / sin^3(th)
                d2th_du2 = -cos_cut / (sin_cut**3)
                
                # Energy and derivatives
                val = 0.5 * k * (th_cut - theta_0)**2
                d1 = k * (th_cut - theta_0) * dth_du
                d2 = k * (dth_du**2) + k * (th_cut - theta_0) * d2th_du2
                
                return val.detach(), d1.detach(), d2.detach()

            # B. Quadratic Extrapolation: theta -> 0
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val, u_cut_pos)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)
            
            # C. Quadratic Extrapolation: theta -> pi
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi, u_cut_neg)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            # Integration
            energy = torch.where(
                is_singular_0,
                E_quad_0,
                torch.where(is_singular_pi, E_quad_pi, E_safe)
            )
            
            return energy

class StructKeepAnglePotentialv2:
    """
    Class for calculating angle restraint potential energy between fragment centers.
    
    This class calculates the angle formed by the geometric centers (centroids) of three 
    specified atom groups (fragments). It ensures numerical stability and gradient 
    accuracy at singularities (theta -> 0, theta -> pi).
    
    Energy Function: E = 0.5 * k * (theta - theta_0)^2
    
    Processing Cases:
    1. theta -> 0, theta_0 != 0 : Quadratic Extrapolation - Maintains positive curvature (Hessian) to prevent optimizer explosion (step size divergence).
    2. theta -> 0, theta_0 ~= 0 : Taylor Expansion - Maintains physical accuracy for near-linear equilibrium.
    3. theta -> pi, theta_0 != pi : Quadratic Extrapolation - Maintains positive curvature (Hessian) to prevent optimizer explosion.
    4. theta -> pi, theta_0 ~= pi : Taylor Expansion - Maintains physical accuracy for near-linear equilibrium.
    5. Normal Region: Standard acos calculation.
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold: Below this angle (rad), switch to extrapolation.
        # 1e-3 rad is conservative enough to ensure stability for optimizers.
        self.THETA_CUT = 1e-3
        
        # Threshold to determine if the equilibrium angle is near a singularity (rad)
        self.EPSILON_PARAM = 1e-3
        
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias potential parameters
                bias_pot_params[0] : keep_angle_spring_const_v2
                bias_pot_params[1] : keep_angle_angle_v2
        
        Returns:
            energy: Angle potential energy (Scalar Tensor)
        """
        # ========================================
        # 1. Parameter Retrieval
        # ========================================
        if len(bias_pot_params) == 0:
            k = self.config["keep_angle_v2_spring_const"]
            theta_0_deg = torch.tensor(self.config["keep_angle_v2_angle"])
            theta_0 = torch.deg2rad(theta_0_deg)
        else:
            k = bias_pot_params[0]
            theta_0_deg = bias_pot_params[1]
            if isinstance(theta_0_deg, torch.Tensor):
                theta_0 = torch.deg2rad(theta_0_deg)
            else:
                theta_0 = torch.deg2rad(torch.tensor(theta_0_deg))

        # Device and dtype handling
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        # Pre-calculation of constants
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        theta_0 = theta_0.to(device=device, dtype=dtype)
        theta_cut_val = torch.tensor(self.THETA_CUT, device=device, dtype=dtype)
        epsilon_param = torch.tensor(self.EPSILON_PARAM, device=device, dtype=dtype)

        # ========================================
        # 2. Vector and Cosine (u) Calculation (Fragment Centers)
        # ========================================
        
        # Calculate centroids for each fragment
        fragm_1_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm1"], device=device) - 1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm2"], device=device) - 1], dim=0)
        fragm_3_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm3"], device=device) - 1], dim=0)
            
        vector1 = fragm_1_center - fragm_2_center
        vector2 = fragm_3_center - fragm_2_center
        
        # Norm calculation
        norm1 = torch.linalg.norm(vector1)
        norm2 = torch.linalg.norm(vector2)
        
        # u = cos(theta) calculation with safety clamp
        # Adding 1e-12 to denominator to prevent NaN if fragment centers overlap exactly
        u = torch.dot(vector1, vector2) / (norm1 * norm2 + 1e-12)
        u = torch.clamp(u, -1.0, 1.0)

        # ========================================
        # 3. Case Branching Processing
        # ========================================
        
        # --- Case 2 & 4: Equilibrium angle is near a singularity ---
        # Taylor expansion (Physical accuracy) - No change needed here
        
        # Case 2: theta_0 approx 0
        if torch.abs(theta_0) < epsilon_param:
            delta = 1.0 - u
            theta_sq = delta * (2.0 + delta * (1.0/3.0 + delta * 4.0/45.0))
            energy = 0.5 * k * theta_sq
            return energy

        # Case 4: theta_0 approx PI
        elif torch.abs(theta_0 - PI) < epsilon_param:
            delta = 1.0 + u
            diff_sq = delta * (2.0 + delta * (1.0/3.0 + delta * 4.0/45.0))
            energy = 0.5 * k * diff_sq
            return energy

        # --- Case 1, 3, 5: General Angle ---
        # Replaced Linear Extrapolation with Quadratic Extrapolation
        else:
            # Threshold calculation
            u_cut_pos = torch.cos(theta_cut_val)
            u_cut_neg = torch.cos(PI - theta_cut_val)
            
            # Mask creation
            is_singular_0 = (u > u_cut_pos)    # theta -> 0
            is_singular_pi = (u < u_cut_neg)   # theta -> pi
            is_safe = ~(is_singular_0 | is_singular_pi)
            
            # A. Normal Region (Case 5)
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0) ** 2
            
            # ---------------------------------------------------------
            # Helper for Quadratic Extrapolation Parameters
            # E(u) approx E_cut + E'_cut * (u - u_cut) + 0.5 * E''_cut * (u - u_cut)^2
            # ---------------------------------------------------------
            def get_quad_params(th_cut, u_bnd):
                # Analytical derivatives w.r.t u = cos(theta)
                
                sin_cut = torch.sin(th_cut)
                cos_cut = torch.cos(th_cut)
                
                # Derivatives of theta w.r.t u
                # dth/du = -1/sin(th)
                dth_du = -1.0 / sin_cut
                # d2th/du2 = -cos(th) / sin^3(th)
                d2th_du2 = -cos_cut / (sin_cut**3)
                
                # Energy derivatives w.r.t u
                # E = 0.5 * k * (th - th0)^2
                # dE/du = k(th - th0) * dth/du
                # d2E/du2 = k * (dth/du)^2 + k(th - th0) * d2th/du2
                
                val = 0.5 * k * (th_cut - theta_0)**2
                d1 = k * (th_cut - theta_0) * dth_du
                d2 = k * (dth_du**2) + k * (th_cut - theta_0) * d2th_du2
                
                return val.detach(), d1.detach(), d2.detach()

            # B. Quadratic Extrapolation: theta -> 0 (Case 1)
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val, u_cut_pos)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)
            
            # C. Quadratic Extrapolation: theta -> pi (Case 3)
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi, u_cut_neg)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            # D. Integration
            energy = torch.where(
                is_singular_0,
                E_quad_0,
                torch.where(is_singular_pi, E_quad_pi, E_safe)
            )
            
            return energy

class StructKeepAnglePotentialv2:
    """
    Class for calculating angle restraint potential energy between fragment centers.
    
    This class calculates the angle formed by the geometric centers (centroids) of three 
    specified atom groups (fragments). It ensures numerical stability and gradient 
    accuracy at singularities (theta -> 0, theta -> pi).
    
    Energy Function: E = 0.5 * k * (theta - theta_0)^2
    
    Processing Cases:
    1. theta -> 0, theta_0 != 0 : Linear Extrapolation (Clamping) - Ensures gradient stability.
    2. theta -> 0, theta_0 ~= 0 : Taylor Expansion - Maintains physical accuracy.
    3. theta -> pi, theta_0 != pi : Linear Extrapolation (Clamping) - Ensures gradient stability.
    4. theta -> pi, theta_0 ~= pi : Taylor Expansion - Maintains physical accuracy.
    5. Normal Region: Standard acos calculation.
    """

    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold setting (Recommended for double precision: 1e-4 rad)
        self.THETA_CUT = 1e-4
        
        # Threshold to determine if the equilibrium angle is near a singularity (rad)
        self.EPSILON_PARAM = 1e-4
        
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates angle potential energy between fragment centers with singularity handling.
        
        Args:
            geom_num_list: Tensor of atomic coordinates (N_atoms, 3)
            bias_pot_params: Optional bias potential parameters
                bias_pot_params[0] : keep_angle_v2_spring_const
                bias_pot_params[1] : keep_angle_v2_angle
        """
        
        # ========================================
        # 1. Parameter Retrieval
        # ========================================
        if len(bias_pot_params) == 0:
            k = self.config["keep_angle_v2_spring_const"]
            theta_0_deg = torch.tensor(self.config["keep_angle_v2_angle"])
            theta_0 = torch.deg2rad(theta_0_deg)
        else:
            k = bias_pot_params[0]
            theta_0_deg = bias_pot_params[1]
            if isinstance(theta_0_deg, torch.Tensor):
                theta_0 = torch.deg2rad(theta_0_deg)
            else:
                theta_0 = torch.deg2rad(torch.tensor(theta_0_deg))

        # Device and dtype handling
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        # Pre-calculation of constants
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        theta_0 = theta_0.to(device=device, dtype=dtype)
        theta_cut_val = torch.tensor(self.THETA_CUT, device=device, dtype=dtype)
        epsilon_param = torch.tensor(self.EPSILON_PARAM, device=device, dtype=dtype)

        # ========================================
        # 2. Vector and Cosine (u) Calculation (Fragment Centers)
        # ========================================
        
        # Calculate centroids for each fragment
        # Note: indices are 1-based in config, so we subtract 1
        fragm_1_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm1"], device=device) - 1], dim=0)
        fragm_2_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm2"], device=device) - 1], dim=0)
        fragm_3_center = torch.mean(geom_num_list[torch.tensor(self.config["keep_angle_v2_fragm3"], device=device) - 1], dim=0)
           
        vector1 = fragm_1_center - fragm_2_center
        vector2 = fragm_3_center - fragm_2_center
        
        # Norm calculation
        norm1 = torch.linalg.norm(vector1)
        norm2 = torch.linalg.norm(vector2)
        
        # u = cos(theta) calculation with safety clamp
        # Adding 1e-12 to denominator to prevent NaN if fragment centers overlap exactly
        u = torch.dot(vector1, vector2) / (norm1 * norm2 + 1e-12)
        u = torch.clamp(u, -1.0, 1.0)

        # ========================================
        # 3. Case Branching Processing
        # ========================================
        
        # --- Case 2 & 4: Equilibrium angle is near a singularity ---
        
        # Case 2: theta_0 approx 0
        if torch.abs(theta_0) < epsilon_param:
            delta = 1.0 - u
            theta_sq = delta * (2.0 + delta * (1.0/3.0 + delta * 4.0/45.0))
            energy = 0.5 * k * theta_sq
            return energy

        # Case 4: theta_0 approx PI
        elif torch.abs(theta_0 - PI) < epsilon_param:
            delta = 1.0 + u
            diff_sq = delta * (2.0 + delta * (1.0/3.0 + delta * 4.0/45.0))
            energy = 0.5 * k * diff_sq
            return energy

        # --- Case 1, 3, 5: General Angle ---
        else:
            # Threshold calculation
            u_cut_pos = torch.cos(theta_cut_val)
            u_cut_neg = torch.cos(PI - theta_cut_val)
            
            # Mask creation
            is_singular_0 = (u > u_cut_pos)    # theta -> 0
            is_singular_pi = (u < u_cut_neg)   # theta -> pi
            is_safe = ~(is_singular_0 | is_singular_pi)
            
            # A. Normal Region (Case 5)
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0) ** 2
            
            # B. Linear Extrapolation: theta -> 0 (Case 1)
            V_cut_0 = 0.5 * k * (theta_cut_val - theta_0) ** 2
            sin_cut_0 = torch.sin(theta_cut_val)
            Slope_0 = k * (theta_cut_val - theta_0) * (-1.0 / sin_cut_0)
            
            V_cut_0 = V_cut_0.detach()
            Slope_0 = Slope_0.detach()
            
            E_linear_0 = V_cut_0 + Slope_0 * (u - u_cut_pos)
            
            # C. Linear Extrapolation: theta -> pi (Case 3)
            theta_cut_pi = PI - theta_cut_val
            V_cut_pi = 0.5 * k * (theta_cut_pi - theta_0) ** 2
            sin_cut_pi = torch.sin(theta_cut_pi)
            Slope_pi = k * (theta_cut_pi - theta_0) * (-1.0 / sin_cut_pi)
            
            V_cut_pi = V_cut_pi.detach()
            Slope_pi = Slope_pi.detach()
            
            E_linear_pi = V_cut_pi + Slope_pi * (u - u_cut_neg)

            # D. Integration
            energy = torch.where(
                is_singular_0,
                E_linear_0,
                torch.where(is_singular_pi, E_linear_pi, E_safe)
            )
            
            return energy

class StructKeepAnglePotentialAtomDistDependent:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["aDD_keep_angle_spring_const"] 
                              self.config["aDD_keep_angle_min_angle"] 
                              self.config["aDD_keep_angle_max_angle"]
                              self.config["aDD_keep_angle_base_dist"]
                              self.config["aDD_keep_angle_reference_atom"] 
                              self.config["aDD_keep_angle_center_atom"] 
                              self.config["aDD_keep_angle_atoms"]
        
        """
        energy = 0.0
        self.config["keep_angle_spring_const"] = self.config["aDD_keep_angle_spring_const"] 
        max_angle = torch.tensor(self.config["aDD_keep_angle_max_angle"])
        min_angle = torch.tensor(self.config["aDD_keep_angle_min_angle"])
        ref_dist = torch.linalg.norm(geom_num_list[self.config["aDD_keep_angle_center_atom"]-1] - geom_num_list[self.config["aDD_keep_angle_reference_atom"]-1]) / self.bohr2angstroms
        base_dist = self.config["aDD_keep_angle_base_dist"] / self.bohr2angstroms
        eq_angle = min_angle + ((max_angle - min_angle)/(1 + torch.exp(-(ref_dist - base_dist))))
        
        bias_pot_params = [self.config["aDD_keep_angle_spring_const"], eq_angle]
        KAP = StructKeepAnglePotential(keep_angle_angle=eq_angle, keep_angle_spring_const=self.config["aDD_keep_angle_spring_const"], keep_angle_atom_pairs=[self.config["aDD_keep_angle_atoms"][0], self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]])
        
        energy += KAP.calc_energy(geom_num_list, bias_pot_params)
        
        KAP = StructKeepAnglePotential(keep_angle_angle=eq_angle, keep_angle_spring_const=self.config["aDD_keep_angle_spring_const"], keep_angle_atom_pairs=[self.config["aDD_keep_angle_atoms"][2], self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]])
        energy += KAP.calc_energy(geom_num_list, bias_pot_params)
        
        KAP = StructKeepAnglePotential(keep_angle_angle=eq_angle, keep_angle_spring_const=self.config["aDD_keep_angle_spring_const"], keep_angle_atom_pairs=[self.config["aDD_keep_angle_atoms"][0], self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][2]])
        energy += KAP.calc_energy(geom_num_list, bias_pot_params)
    
        return energy
    
    
class StructKeepAnglePotentialLonePairAngle:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables: self.config["lone_pair_keep_angle_spring_const"] 
                              self.config["lone_pair_keep_angle_angle"] 
                              self.config["lone_pair_keep_angle_atom_pair_1"]
                              self.config["lone_pair_keep_angle_atom_pair_2"]        
        """
        lone_pair_1_vec_1 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][1]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][0]-1]) / self.bohr2angstroms
     
        lone_pair_1_vec_2 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][2]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][0]-1]) / self.bohr2angstroms
        lone_pair_1_vec_3 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][3]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_1"][0]-1]) / self.bohr2angstroms
        lone_pair_1_vector = (lone_pair_1_vec_1/torch.linalg.norm(lone_pair_1_vec_1)) + (lone_pair_1_vec_2/torch.linalg.norm(lone_pair_1_vec_2)) + (lone_pair_1_vec_3/torch.linalg.norm(lone_pair_1_vec_3))
        
        lone_pair_1_vector_norm = lone_pair_1_vector / torch.linalg.norm(lone_pair_1_vector)
        
        lone_pair_2_vec_1 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][1]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][0]-1]) / self.bohr2angstroms
        lone_pair_2_vec_2 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][2]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][0]-1]) / self.bohr2angstroms
        lone_pair_2_vec_3 = (geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][3]-1] - geom_num_list[self.config["lone_pair_keep_angle_atom_pair_2"][0]-1]) / self.bohr2angstroms
        
        lone_pair_2_vector = (lone_pair_2_vec_1/torch.linalg.norm(lone_pair_2_vec_1)) + (lone_pair_2_vec_2/torch.linalg.norm(lone_pair_2_vec_2)) + (lone_pair_2_vec_3/torch.linalg.norm(lone_pair_2_vec_3))
        
        lone_pair_2_vector_norm = lone_pair_2_vector / torch.linalg.norm(lone_pair_2_vector)
        
        theta = torch_calc_angle_from_vec(lone_pair_1_vector_norm, lone_pair_2_vector_norm)
        energy = 0.5 * self.config["lone_pair_keep_angle_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["lone_pair_keep_angle_angle"]))) ** 2

        return energy