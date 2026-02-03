from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.Utils.calc_tools import torch_calc_angle_from_vec

import torch
import math

class StructKeepAnglePotential:
    """
    Harmonic angle potential implementation ensuring C1 continuity and numerical stability at linear boundaries.

    Calculates the potential energy:
        E = 0.5 * k * (theta - theta_0)^2

    Standard implementations using `acos(cos_theta)` suffer from gradient instabilities and loss of precision
    near theta = 0 (cos_theta = 1) and theta = pi (cos_theta = -1). This class addresses these issues by
    switching to a 5th-order Taylor expansion of the arc-cosine function squared in these singular regions.

    Attributes:
        config (dict): Configuration parameters including spring constants and target angles.
        THETA_CUT (float): The angular threshold (in radians) defining the boundary regions near 0 and 180 degrees.
                           Inside this threshold, the Taylor expansion is used. Default is 1e-3.
        EPSILON_PARAM (float): Small epsilon for floating-point comparisons (e.g., checking if theta_0 is 0).
    """

    def __init__(self, **kwarg):
        """
        Initialize the potential parameters.

        Args:
            **kwarg: Keyword arguments containing:
                - keep_angle_spring_const (float): The force constant (k).
                - keep_angle_angle (float): The equilibrium angle (theta_0) in degrees.
                - keep_angle_atom_pairs (list of int): Indices of the three atoms forming the angle (1-based).
        """
        self.config = kwarg
        
        # Unit conversion placeholders (assuming UnitValueLib is defined elsewhere)
        # UVL = UnitValueLib()
        # self.hartree2kcalmol = UVL.hartree2kcalmol 
        # self.bohr2angstroms = UVL.bohr2angstroms 
        # self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Numerical stability parameters
        # 1e-3 rad is approximately 0.05 degrees.
        # This threshold is sufficiently large to avoid 'acos' catastrophic cancellation
        # while being small enough for the Taylor approximation to remain highly accurate.
        self.THETA_CUT = 1e-3
        self.EPSILON_PARAM = 1e-8

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Compute the potential energy for the current geometry.

        Args:
            geom_num_list (torch.Tensor): Tensor of atomic coordinates (shape: [N_atoms, 3]).
            bias_pot_params (list, optional): Overriding parameters [k, theta_0_deg]. 
                                              If empty, uses values from self.config.

        Returns:
            torch.Tensor: The calculated potential energy (scalar).
        """
        # 1. Parse parameters
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

        # 2. Setup device and constants
        device = geom_num_list.device if isinstance(geom_num_list, torch.Tensor) else torch.device("cpu")
        dtype = geom_num_list.dtype
        
        PI = torch.tensor(math.pi, device=device, dtype=dtype)
        theta_0 = theta_0.to(device=device, dtype=dtype)
        theta_cut_val = torch.tensor(self.THETA_CUT, device=device, dtype=dtype)
        epsilon_param = torch.tensor(self.EPSILON_PARAM, device=device, dtype=dtype)

        # 3. Calculate cosine of the angle (u)
        idx1 = self.config["keep_angle_atom_pairs"][0] - 1
        idx2 = self.config["keep_angle_atom_pairs"][1] - 1
        idx3 = self.config["keep_angle_atom_pairs"][2] - 1
        
        vec1 = geom_num_list[idx1] - geom_num_list[idx2]
        vec2 = geom_num_list[idx3] - geom_num_list[idx2]
        
        norm1 = torch.linalg.norm(vec1)
        norm2 = torch.linalg.norm(vec2)
        
        # Avoid division by zero
        norm1_2 = torch.clamp(norm1 * norm2, min=1e-12)
        
        u = torch.dot(vec1, vec2) / norm1_2
        u = torch.clamp(u, -1.0, 1.0) # Numerical clamp to stay within valid acos domain

        # 4. Define Thresholds for Taylor Expansion
        u_cut_pos = torch.cos(theta_cut_val)      # Threshold near 0 degrees (u ~ 1)
        u_cut_neg = torch.cos(PI - theta_cut_val) # Threshold near 180 degrees (u ~ -1)

        # ==============================================
        # Helper Functions: High-Order Taylor Expansions
        # ==============================================
        
        def theta_sq_taylor_at_0(u_val):
            """
            Compute theta^2 using a 5th-order Taylor expansion near u=1 (theta=0).
            Expansion of (arccos(1-x))^2:
            theta^2 approx 2x + x^2/3 + 8x^3/45 + 4x^4/35 + 128x^5/1575 + ...
            where x = 1 - u.
            """
            delta = 1.0 - u_val
            # Horner's method for efficiency and precision
            term = 128.0/1575.0
            term = 4.0/35.0 + delta * term
            term = 8.0/45.0 + delta * term
            term = 1.0/3.0 + delta * term
            term = 2.0 + delta * term
            return delta * term

        def theta_sq_taylor_at_pi(u_val):
            """
            Compute (pi - theta)^2 using a 5th-order Taylor expansion near u=-1 (theta=pi).
            The coefficients are identical to the u=1 case, but delta = 1 + u.
            """
            delta = 1.0 + u_val
            # Horner's method
            term = 128.0/1575.0
            term = 4.0/35.0 + delta * term
            term = 8.0/45.0 + delta * term
            term = 1.0/3.0 + delta * term
            term = 2.0 + delta * term
            return delta * term

        # ==============================================
        # Energy Calculation Logic (3 Branches)
        # ==============================================

        # --- BRANCH A: Equilibrium Angle is approx 0 degrees ---
        if torch.abs(theta_0) < epsilon_param:
            # Region 1 (u > u_cut_pos): Near 0. Use Taylor expansion of theta^2.
            theta_sq = theta_sq_taylor_at_0(u)
            E_taylor_0 = 0.5 * k * theta_sq
            
            # Region 2 (u < u_cut_neg): Near pi. 
            # theta approx pi - sqrt((pi-theta)^2).
            diff_sq = theta_sq_taylor_at_pi(u)
            sqrt_diff_sq = torch.sqrt(torch.clamp(diff_sq, min=1e-30))
            theta_approx = PI - sqrt_diff_sq
            E_taylor_pi = 0.5 * k * theta_approx**2
            
            # Region 3: Normal region. Use acos.
            u_safe = torch.clamp(u, u_cut_neg, u_cut_pos)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * theta_exact**2

            return torch.where(
                u > u_cut_pos,
                E_taylor_0,
                torch.where(u < u_cut_neg, E_taylor_pi, E_exact)
            )

        # --- BRANCH B: Equilibrium Angle is approx 180 degrees ---
        elif torch.abs(theta_0 - PI) < epsilon_param:
            # Region 1 (u < u_cut_neg): Near pi. Use Taylor expansion of (pi-theta)^2.
            diff_sq = theta_sq_taylor_at_pi(u)
            E_taylor_pi = 0.5 * k * diff_sq
            
            # Region 2 (u > u_cut_pos): Near 0.
            # theta approx sqrt(theta^2).
            theta_sq = theta_sq_taylor_at_0(u)
            sqrt_theta_sq = torch.sqrt(torch.clamp(theta_sq, min=1e-30))
            theta_approx = sqrt_theta_sq
            E_taylor_0 = 0.5 * k * (theta_approx - PI)**2
            
            # Region 3: Normal region. Use acos.
            u_safe = torch.clamp(u, u_cut_neg, u_cut_pos)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact - PI)**2

            return torch.where(
                u < u_cut_neg,
                E_taylor_pi,
                torch.where(u > u_cut_pos, E_taylor_0, E_exact)
            )

        # --- BRANCH C: General Equilibrium Angle (e.g., 109.5 deg) ---
        else:
            is_singular_0 = (u > u_cut_pos)
            is_singular_pi = (u < u_cut_neg)
            
            # Normal Region
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0)**2
            
            # Singular Region 0 (u ~ 1)
            # Use Taylor to get high-precision theta approx, then compute energy.
            theta_sq = theta_sq_taylor_at_0(u)
            sqrt_theta_sq = torch.sqrt(torch.clamp(theta_sq, min=1e-30))
            E_taylor_0 = 0.5 * k * (sqrt_theta_sq - theta_0)**2
            
            # Singular Region Pi (u ~ -1)
            diff_sq = theta_sq_taylor_at_pi(u)
            sqrt_diff_sq = torch.sqrt(torch.clamp(diff_sq, min=1e-30))
            theta_approx_pi = PI - sqrt_diff_sq
            E_taylor_pi = 0.5 * k * (theta_approx_pi - theta_0)**2

            return torch.where(
                is_singular_0,
                E_taylor_0,
                torch.where(is_singular_pi, E_taylor_pi, E_safe)
            )

class StructKeepAnglePotentialv2:
    r"""
    Angle restraint potential operating on fragment centroids with robust singularity handling.
    
    This class calculates the angle potential :math:`E = 0.5 \cdot k \cdot (\theta - \theta_0)^2`
    defined by the geometric centers (centroids) of three atom fragments.

    It implements a hybrid singularity handling strategy to ensure numerical stability
    (C1 continuity) and physical accuracy across all configurations, particularly
    when the angle approaches 0 or 180 degrees.

    **Singularity Handling Strategies:**

    1.  **High-Order Taylor Expansion (Physical Accuracy)**:
        Applied when the angle approaches a singularity (0 or 180) that **COINCIDES** with
        the equilibrium angle :math:`\\theta_0`.
        * *Context:* Linear (:math:`\\theta_0=180`) or hypothetical collapsed (:math:`\\theta_0=0`) molecules.
        * *Method:* Uses a 5th-order Taylor expansion of :math:`\\arccos(u)^2` to strictly preserve
            the physical curvature (Hessian) without precision loss from `acos`.

    2.  **Quadratic Extrapolation (Numerical Robustness)**:
        Applied when the angle approaches a singularity (0 or 180) that is **FAR** from
        the equilibrium angle.
        * *Context:* A linear molecule (:math:`\\theta_0=180`) being bent towards 0 degrees.
        * *Method:* Replaces the physically diverging force of the harmonic potential (where
            force -> infinity as sin(theta) -> 0) with a finite quadratic barrier in cos-space.
            This prevents gradient explosions in MD simulations.

    Attributes:
        config (dict): Configuration dictionary containing potential parameters.
        THETA_CUT (float): The angular threshold (in radians) defining the boundary regions.
                           Set to 1e-3 (~0.05 deg) to ensure stable transition before `acos` precision loss.
        EPSILON_PARAM (float): Tolerance for determining if theta_0 is exactly 0 or 180.
    """

    def __init__(self, **kwarg):
        """
        Initialize the potential parameters.

        Args:
            **kwarg: Keyword arguments. Must include:
                - keep_angle_v2_spring_const (float): Force constant (k).
                - keep_angle_v2_angle (float): Equilibrium angle in degrees.
                - keep_angle_v2_fragm1 (list[int]): Atom indices for fragment 1.
                - keep_angle_v2_fragm2 (list[int]): Atom indices for fragment 2 (vertex).
                - keep_angle_v2_fragm3 (list[int]): Atom indices for fragment 3.
        """
        self.config = kwarg
        
        # Unit conversion placeholders (assuming UnitValueLib context)
        # UVL = UnitValueLib()
        # self.hartree2kcalmol = UVL.hartree2kcalmol 
        # self.bohr2angstroms = UVL.bohr2angstroms 
        # self.hartree2kjmol = UVL.hartree2kjmol 
        
        # Thresholds
        self.THETA_CUT = 1e-3
        self.EPSILON_PARAM = 1e-8

    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        Compute the potential energy based on fragment centroids.

        Args:
            geom_num_list (torch.Tensor): Atomic coordinates (N_atoms, 3).
            bias_pot_params (list, optional): [k, theta_0] override.

        Returns:
            torch.Tensor: Potential energy.
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

        # 2. Centroid & Vector Calculation
        def get_centroid(key):
            # Convert 1-based config indices to 0-based
            indices = torch.tensor(self.config[key], device=device, dtype=torch.long) - 1
            return torch.mean(geom_num_list[indices], dim=0)

        fragm_1_center = get_centroid("keep_angle_v2_fragm1")
        fragm_2_center = get_centroid("keep_angle_v2_fragm2") # Vertex
        fragm_3_center = get_centroid("keep_angle_v2_fragm3")
            
        vec1 = fragm_1_center - fragm_2_center
        vec2 = fragm_3_center - fragm_2_center
        
        norm1 = torch.linalg.norm(vec1)
        norm2 = torch.linalg.norm(vec2)
        
        # Prevent division by zero if centroids overlap
        norm1_2 = torch.clamp(norm1 * norm2, min=1e-12)
            
        u = torch.dot(vec1, vec2) / norm1_2
        u = torch.clamp(u, -1.0, 1.0) # Numerical clamp for acos domain

        # ========================================
        # 3. Expansion Helper Functions
        # ========================================

        # Thresholds in cosine space
        u_cut_pos = torch.cos(theta_cut_val)      # Near 0 deg
        u_cut_neg = torch.cos(PI - theta_cut_val) # Near 180 deg

        def theta_sq_taylor_at_0(u_val):
            """5th-order Taylor expansion of theta^2 near 0 (u=1)."""
            delta = 1.0 - u_val
            # Horner's method
            term = 128.0/1575.0
            term = 4.0/35.0 + delta * term
            term = 8.0/45.0 + delta * term
            term = 1.0/3.0 + delta * term
            term = 2.0 + delta * term
            return delta * term

        def theta_sq_taylor_at_pi(u_val):
            """5th-order Taylor expansion of (pi-theta)^2 near 180 (u=-1)."""
            delta = 1.0 + u_val
            # Horner's method
            term = 128.0/1575.0
            term = 4.0/35.0 + delta * term
            term = 8.0/45.0 + delta * term
            term = 1.0/3.0 + delta * term
            term = 2.0 + delta * term
            return delta * term

        def get_quad_params(th_cut):
            """
            Compute parameters for Quadratic Extrapolation V(u) = A + B*du + 0.5*C*du^2.
            This matches the Value (V) and Slope (dV/du) at the cutoff point.
            """
            sin_cut = torch.sin(th_cut)
            # Chain rule: d(theta)/du = -1/sin(theta)
            dth_du = -1.0 / sin_cut
            
            # Value at cutoff
            val = 0.5 * k * (th_cut - theta_0)**2
            
            # First derivative wrt theta
            dE_dth = k * (th_cut - theta_0)
            
            # First derivative wrt u (Slope B)
            d1 = dE_dth * dth_du
            
            # Second derivative approximation (Gauss-Newton style: H ~ J^T J)
            # We approximate d2E/du2 to ensure positive curvature and stability.
            d2 = k * (dth_du**2)
            
            return val, d1, d2

        # ========================================
        # 4. Energy Calculation Branches
        # ========================================

        # --- BRANCH A: EXACTLY Linear Equilibrium (theta_0 ~ 0) ---
        if torch.abs(theta_0) < epsilon_param:
            # Region 1 (u ~ 1): Equilibrium Singularity -> Taylor Expansion
            # We want V = 0.5*k*theta^2.
            theta_sq = theta_sq_taylor_at_0(u)
            E_taylor = 0.5 * k * theta_sq

            # Region 2 (u ~ -1): Antipodal Singularity -> Quadratic Extrapolation
            # Molecule is bent to 180, but wants to be 0. Force is huge.
            theta_cut_pi = PI - theta_cut_val
            val_pi, d1_pi, d2_pi = get_quad_params(theta_cut_pi)
            diff_pi = u - u_cut_neg
            E_quad_pi = val_pi + d1_pi * diff_pi + 0.5 * d2_pi * (diff_pi**2)

            # Region 3: Normal
            u_safe = torch.clamp(u, -1.0, u_cut_pos)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact ** 2)

            return torch.where(
                u > u_cut_pos, 
                E_taylor, 
                torch.where(u < u_cut_neg, E_quad_pi, E_exact)
            )

        # --- BRANCH B: EXACTLY Planar Equilibrium (theta_0 ~ 180) ---
        elif torch.abs(theta_0 - PI) < epsilon_param:
            # Region 1 (u ~ -1): Equilibrium Singularity -> Taylor Expansion
            # We want V = 0.5*k*(pi - theta)^2.
            diff_sq_taylor = theta_sq_taylor_at_pi(u)
            E_taylor = 0.5 * k * diff_sq_taylor

            # Region 2 (u ~ 1): Antipodal Singularity -> Quadratic Extrapolation
            # Molecule is bent to 0, but wants to be 180.
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)

            # Region 3: Normal
            u_safe = torch.clamp(u, u_cut_neg, 1.0)
            theta_exact = torch.acos(u_safe)
            E_exact = 0.5 * k * (theta_exact - theta_0) ** 2

            return torch.where(
                u < u_cut_neg, 
                E_taylor, 
                torch.where(u > u_cut_pos, E_quad_0, E_exact)
            )

        # --- BRANCH C: General Angle (e.g., 109.5) ---
        else:
            is_singular_0 = (u > u_cut_pos)
            is_singular_pi = (u < u_cut_neg)
            
            # Normal calculation
            theta_safe = torch.acos(u)
            E_safe = 0.5 * k * (theta_safe - theta_0) ** 2
            
            # Extrapolation near 0
            val_0, d1_0, d2_0 = get_quad_params(theta_cut_val)
            diff_0 = u - u_cut_pos
            E_quad_0 = val_0 + d1_0 * diff_0 + 0.5 * d2_0 * (diff_0**2)
            
            # Extrapolation near 180
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