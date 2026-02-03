from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_angle_from_vec

import torch
import math


class StructKeepAnglePotential:
    """
    Calculates the harmonic angle restraint potential energy between three atoms with full numerical robustness.

    This class implements a potential of the form :math:`E = 0.5 k (\\theta - \\theta_0)^2`.
    It employs specialized strategies to handle numerical singularities associated with the chain rule
    of :math:`\\arccos` at :math:`\\theta = 0` and :math:`\\theta = \\pi`.

    Singularity Handling Strategies:
        1. **Taylor Expansion (Physical Accuracy)**:
           Applied when the equilibrium angle :math:`\\theta_0` is EXACTLY 0 or :math:`\\pi`.
           This uses a high-order Taylor expansion to maintain physical accuracy for linear
           molecules or planar transition states without gradient explosion.

        2. **Quadratic Extrapolation (Numerical Stability)**:
           Applied in two cases:
           - When a normally bent molecule (:math:`\\theta_0 \\neq 0, \\pi`) is forced into linearity (0 or 180 deg).
           - When a linear molecule (:math:`\\theta_0 = 0`) bends all the way to the antipodal singularity (:math:`180` deg), or vice versa.
           
           This uses a Gauss-Newton Approximation for the Hessian (:math:`d^2E/du^2 \\approx k (d\\theta/du)^2`),
           ensuring positive curvature and preventing optimizer instability at the poles.

    Attributes:
        config (dict): Configuration dictionary containing potential parameters.
        THETA_CUT (float): Angle threshold (radians) to switch to extrapolation. Default is 1e-3.
        EPSILON_PARAM (float): Threshold (radians) to distinguish exactly linear/planar equilibrium. Default is 1e-9.
    """

    def __init__(self, **kwarg):
        """
        Initializes the StructKeepAnglePotential instance.

        Args:
            **kwarg: Arbitrary keyword arguments containing configuration parameters.
                     Expected keys include 'keep_angle_spring_const', 'keep_angle_angle', 'keep_angle_atom_pairs', etc.
        """
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Threshold: Below this angle (rad), switch to extrapolation/Taylor.
        self.THETA_CUT = 1e-3
        
        # Threshold for Equilibrium check.
        self.EPSILON_PARAM = 1e-9
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates the angle potential energy.

        Args:
            geom_num_list (torch.Tensor): Tensor of atomic coordinates with shape (N_atoms, 3).
            bias_pot_params (list, optional): List containing dynamic parameters [k, theta_0].
                - k: Spring constant (Hartree/rad^2).
                - theta_0: Equilibrium angle (Degrees).
                If empty, values from `self.config` are used.

        Returns:
            torch.Tensor: Scalar tensor representing the potential energy in Hartree.
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

        # Device/Dtype handling
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
        # Add epsilon to denominator to prevent NaN if atoms overlap exactly
        norm1_2 = norm1 * norm2
        if norm1_2 < 1e-12:
            norm1_2 = norm1_2 + 1e-12
        
        u = torch.dot(vec1, vec2) / (norm1_2)
        u = torch.clamp(u, -1.0, 1.0)

        # ========================================
        # 3. Singularity Handling Logic
        # ========================================

        # Thresholds
        u_cut_pos = torch.cos(theta_cut_val)        # Corresponds to theta ~ 0
        u_cut_neg = torch.cos(PI - theta_cut_val)   # Corresponds to theta ~ pi

        # --- Helper for Quadratic Extrapolation ---
        # Used for any region where the angle approaches a singularity that is NOT the equilibrium.
        def get_quad_params(th_cut):
            # Calculate properties at the cutoff boundary
            sin_cut = torch.sin(th_cut)
            dth_du = -1.0 / sin_cut  # Chain rule: d(acos(u))/du = -1/sqrt(1-u^2) = -1/sin(theta)
            
            # Energy value at cutoff
            val = 0.5 * k * (th_cut - theta_0)**2
            
            # First derivative w.r.t u at cutoff: dE/du = dE/dth * dth/du
            dE_dth = k * (th_cut - theta_0)
            d1 = dE_dth * dth_du
            
            # Second derivative w.r.t u (Gauss-Newton Approx): d2E/du2 approx k * (dth/du)^2
            # We ignore the d2th/du2 term to ensure positive curvature (convexity)
            d2 = k * (dth_du**2) 
            
            return val.detach(), d1.detach(), d2.detach()


        # --- BRANCH A: EXACTLY Linear Equilibrium (theta_0 ~ 0) ---
        if torch.abs(theta_0) < epsilon_param:
            # Region 1: Singularity at Equilibrium (theta ~ 0) -> Taylor Expansion
            delta = 1.0 - u
            # Corrected Taylor expansion for theta^2 around u=1
            theta_sq_taylor = delta * (2.0 + delta * (1.0/3.0 + delta * 8.0/45.0))
            E_taylor = 0.5 * k * theta_sq_taylor

            # Region 2: Singularity at Opposite Pole (theta ~ pi) -> Quadratic Extrapolation
            # Even if equilibrium is 0, we must protect against 180 degrees.
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            # Region 3: Normal -> Exact Analytical
            u_safe = torch.clamp(u, -1.0, u_cut_pos) 
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact ** 2)

            # Combine
            return torch.where(
                u > u_cut_pos, 
                E_taylor, 
                torch.where(u < u_cut_neg, E_quad_pi, E_exact)
            )

        # --- BRANCH B: EXACTLY Planar Equilibrium (theta_0 ~ pi) ---
        elif torch.abs(theta_0 - PI) < epsilon_param:
            # Region 1: Singularity at Equilibrium (theta ~ pi) -> Taylor Expansion
            delta = 1.0 + u
            # Corrected Taylor expansion for (theta-pi)^2 around u=-1
            diff_sq_taylor = delta * (2.0 + delta * (1.0/3.0 + delta * 8.0/45.0))
            E_taylor = 0.5 * k * diff_sq_taylor

            # Region 2: Singularity at Opposite Pole (theta ~ 0) -> Quadratic Extrapolation
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)

            # Region 3: Normal -> Exact Analytical
            u_safe = torch.clamp(u, u_cut_neg, 1.0)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact - theta_0) ** 2

            # Combine
            return torch.where(
                u < u_cut_neg, 
                E_taylor, 
                torch.where(u > u_cut_pos, E_quad_0, E_exact)
            )

        # --- BRANCH C: General Angle ---
        else:
            is_singular_0 = (u > u_cut_pos)    # theta -> 0
            is_singular_pi = (u < u_cut_neg)   # theta -> pi
            
            # 1. Normal Region (Safe acos)
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0) ** 2
            
            # 2. Extrapolation: theta -> 0
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)
            
            # 3. Extrapolation: theta -> pi
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            # Integration using masks
            return torch.where(
                is_singular_0,
                E_quad_0,
                torch.where(is_singular_pi, E_quad_pi, E_safe)
            )


class StructKeepAnglePotentialv2:
    """
    Class for calculating angle restraint potential energy between fragment centers (centroids)
    with full numerical robustness.

    This class applies the same robust singularity handling logic as `StructKeepAnglePotential`,
    but operates on the geometric centers of three specified atom fragments.

    Singularity Handling Strategies:
        1. **Taylor Expansion (Physical Accuracy)**:
           Applied when the equilibrium angle :math:`\\theta_0` is EXACTLY 0 or :math:`\\pi`.
           Maintains accuracy for linear/planar equilibrium states.

        2. **Quadratic Extrapolation (Numerical Stability)**:
           Applied to singularities (0 or 180 deg) that are NOT the equilibrium angle.
           This includes the antipodal pole for linear/planar molecules (e.g., theta ~ 180 when theta_0 ~ 0),
           ensuring the gradient never explodes even in extreme bending configurations.

    Attributes:
        config (dict): Configuration dictionary containing potential parameters.
        THETA_CUT (float): Angle threshold (radians) to switch to extrapolation. Default is 1e-3.
        EPSILON_PARAM (float): Threshold (radians) to distinguish exactly linear/planar equilibrium. Default is 1e-9.
    """

    def __init__(self, **kwarg):
        """
        Initializes the StructKeepAnglePotentialv2 instance.

        Args:
            **kwarg: Arbitrary keyword arguments. Expected keys include:
                     'keep_angle_v2_spring_const', 'keep_angle_v2_angle',
                     'keep_angle_v2_fragm1', 'keep_angle_v2_fragm2', 'keep_angle_v2_fragm3'.
        """
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        self.THETA_CUT = 1e-3
        self.EPSILON_PARAM = 1e-9
        return

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Calculates the angle potential energy for fragment centroids.

        Args:
            geom_num_list (torch.Tensor): Tensor of atomic coordinates with shape (N_atoms, 3).
            bias_pot_params (list, optional): List containing [k, theta_0].

        Returns:
            torch.Tensor: Scalar tensor representing the potential energy in Hartree.
        """
        # 1. Parameter Retrieval
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

        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        theta_0 = theta_0.to(device=device, dtype=dtype)
        theta_cut_val = torch.tensor(self.THETA_CUT, device=device, dtype=dtype)
        epsilon_param = torch.tensor(self.EPSILON_PARAM, device=device, dtype=dtype)

        # 2. Vector & Cosine Calculation (Fragment Centroids)
        def get_centroid(key):
            # Indices in config are 1-based, convert to 0-based
            indices = torch.tensor(self.config[key], device=device, dtype=torch.long) - 1
            return torch.mean(geom_num_list[indices], dim=0)

        fragm_1_center = get_centroid("keep_angle_v2_fragm1")
        fragm_2_center = get_centroid("keep_angle_v2_fragm2") # Vertex (Center Fragment)
        fragm_3_center = get_centroid("keep_angle_v2_fragm3")
            
        vec1 = fragm_1_center - fragm_2_center
        vec2 = fragm_3_center - fragm_2_center
        
        norm1 = torch.linalg.norm(vec1)
        norm2 = torch.linalg.norm(vec2)
        
        norm1_2 = norm1 * norm2
        if norm1_2 < 1e-12:
            norm1_2 = norm1_2 + 1e-12
            
        u = torch.dot(vec1, vec2) / (norm1_2)
        u = torch.clamp(u, -1.0, 1.0)

        # ========================================
        # 3. Singularity Handling Logic
        # ========================================

        # Thresholds
        u_cut_pos = torch.cos(theta_cut_val)
        u_cut_neg = torch.cos(PI - theta_cut_val)

        # --- Helper for Quadratic Extrapolation ---
        def get_quad_params(th_cut):
            sin_cut = torch.sin(th_cut)
            dth_du = -1.0 / sin_cut
            
            val = 0.5 * k * (th_cut - theta_0)**2
            
            dE_dth = k * (th_cut - theta_0)
            d1 = dE_dth * dth_du
            
            # Gauss-Newton Approximation
            d2 = k * (dth_du**2)
            
            return val.detach(), d1.detach(), d2.detach()

        # --- BRANCH A: EXACTLY Linear Equilibrium (theta_0 ~ 0) ---
        if torch.abs(theta_0) < epsilon_param:
            # Region 1: Singularity at Equilibrium (theta ~ 0) -> Taylor Expansion
            delta = 1.0 - u
            theta_sq_taylor = delta * (2.0 + delta * (1.0/3.0 + delta * 8.0/45.0))
            E_taylor = 0.5 * k * theta_sq_taylor

            # Region 2: Singularity at Opposite Pole (theta ~ pi) -> Quadratic Extrapolation
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            # Region 3: Normal -> Exact Analytical
            u_safe = torch.clamp(u, -1.0, u_cut_pos)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact ** 2)

            return torch.where(
                u > u_cut_pos, 
                E_taylor, 
                torch.where(u < u_cut_neg, E_quad_pi, E_exact)
            )

        # --- BRANCH B: EXACTLY Planar Equilibrium (theta_0 ~ pi) ---
        elif torch.abs(theta_0 - PI) < epsilon_param:
            # Region 1: Singularity at Equilibrium (theta ~ pi) -> Taylor Expansion
            delta = 1.0 + u
            diff_sq_taylor = delta * (2.0 + delta * (1.0/3.0 + delta * 8.0/45.0))
            E_taylor = 0.5 * k * diff_sq_taylor

            # Region 2: Singularity at Opposite Pole (theta ~ 0) -> Quadratic Extrapolation
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)

            # Region 3: Normal -> Exact Analytical
            u_safe = torch.clamp(u, u_cut_neg, 1.0)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact - theta_0) ** 2

            return torch.where(
                u < u_cut_neg, 
                E_taylor, 
                torch.where(u > u_cut_pos, E_quad_0, E_exact)
            )

        # --- BRANCH C: General Angle ---
        else:
            is_singular_0 = (u > u_cut_pos)
            is_singular_pi = (u < u_cut_neg)
            
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0) ** 2
            
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)
            
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            return torch.where(
                is_singular_0,
                E_quad_0,
                torch.where(is_singular_pi, E_quad_pi, E_safe)
            )        
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