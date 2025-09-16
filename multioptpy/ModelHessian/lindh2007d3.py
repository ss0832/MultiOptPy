import numpy as np

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib, element_number
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.ModelHessian.calc_params import torsion2, outofplane2
from multioptpy.Parameters.parameter import D2_C6_coeff_lib, UFF_VDW_distance_lib, D3Parameters


class Lindh2007D3ApproxHessian:
    """
    Lindh's Model Hessian (2007) augmented with D3 dispersion correction.
    
    This class implements Lindh's 2007 approximate Hessian model with D3 dispersion
    corrections for improved accuracy in describing non-covalent interactions.
    
    References:
        - Lindh et al., Chem. Phys. Lett. 2007, 241, 423.
        - Grimme et al., J. Chem. Phys. 2010, 132, 154104 (DFT-D3).
        - https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
    """
    def __init__(self):
        # Unit conversion constants
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        
        # Force constant parameters
        self.bond_threshold_scale = 1.0
        self.kr = 0.45  # Bond stretching force constant
        self.kf = 0.10  # Angle bend force constant
        self.kt = 0.0025  # Torsion force constant
        self.ko = 0.16  # Out-of-plane force constant
        self.kd = 0.05  # Dispersion force constant
        
        # Numerical parameters
        self.cutoff = 50.0  # Cutoff for long-range interactions (Bohr)
        self.eps = 1.0e-12  # Numerical threshold for avoiding division by zero
        
        # Reference parameters (element type matrices)
        self.rAv = np.array([
            [1.3500, 2.1000, 2.5300],
            [2.1000, 2.8700, 3.8000],
            [2.5300, 3.8000, 4.5000]
        ])
        
        self.aAv = np.array([
            [1.0000, 0.3949, 0.3949],
            [0.3949, 0.2800, 0.1200],
            [0.3949, 0.1200, 0.0600]
        ])
        
        self.dAv = np.array([
            [0.0000, 3.6000, 3.6000],
            [3.6000, 5.3000, 5.3000],
            [3.6000, 5.3000, 5.3000]
        ])
        
        # D3 dispersion parameters
        self.d3params = D3Parameters()
        
    def select_idx(self, elem_num):
        """
        Determine element group index for parameter selection.
        
        Args:
            elem_num (str or int): Element symbol or atomic number
            
        Returns:
            int: Group index (0-2) for parameter lookup
        """
        if isinstance(elem_num, str):
            elem_num = element_number(elem_num)

        # Group 1: H
        if elem_num > 0 and elem_num < 2:
            return 0
        # Group 2: First row elements (Li-Ne)
        elif elem_num >= 2 and elem_num < 10:
            return 1
        # Group 3: All others
        else:
            return 2
    
    def calc_force_const(self, alpha, r_0, distance_2):
        """
        Calculate bond stretching force constant based on Lindh's model.
        
        Args:
            alpha: Exponential parameter
            r_0: Reference bond length
            distance_2: Squared distance between atoms
            
        Returns:
            float: Force constant
        """
        return np.exp(alpha * (r_0**2 - distance_2))
    
    def get_c6_coefficient(self, element):
        """
        Get C6 dispersion coefficient for an element.
        
        Args:
            element: Element symbol
            
        Returns:
            float: C6 coefficient in atomic units
        """
        return D2_C6_coeff_lib(element)
    
    def calc_d3_force_const(self, r_ij, c6_param, c8_param, r0_param):
        """
        Calculate D3 dispersion force constant with Becke-Johnson damping.
        
        Args:
            r_ij: Distance between atoms
            c6_param: C6 dispersion coefficient
            c8_param: C8 dispersion coefficient
            r0_param: van der Waals radius sum parameter
            
        Returns:
            float: D3 dispersion force constant
        """
        # Becke-Johnson damping function for C6 term
        r0_plus_a1 = r0_param + self.d3params.a1
        f_damp_6 = r_ij**6 / (r_ij**6 + (r0_plus_a1 * self.d3params.a2)**6)
        
        # Becke-Johnson damping function for C8 term
        f_damp_8 = r_ij**8 / (r_ij**8 + (r0_plus_a1 * self.d3params.a2)**8)
        
        # D3 dispersion energy contributions
        e6 = -self.d3params.s6 * c6_param * f_damp_6 / r_ij**6
        e8 = -self.d3params.s8 * c8_param * f_damp_8 / r_ij**8
        
        # Combined force constant (negative of energy for attractive contribution)
        return -(e6 + e8)
    
    def get_d3_parameters(self, elem1, elem2):
        """
        Get D3 parameters for a pair of elements.
        
        Args:
            elem1: First element symbol
            elem2: Second element symbol
            
        Returns:
            tuple: (c6_param, c8_param, r0_param) for the element pair
        """
        # Get C6 coefficients
        c6_1 = self.get_c6_coefficient(elem1)
        c6_2 = self.get_c6_coefficient(elem2)
        
        # Combine C6 coefficients
        c6_param = np.sqrt(c6_1 * c6_2)
        
        # Get r4r2 values for C8 coefficient calculation
        r4r2_1 = self.d3params.get_r4r2(elem1)
        r4r2_2 = self.d3params.get_r4r2(elem2)
        
        # Calculate C8 coefficient (3.0 is the conversion factor in Grimme's D3 formulation)
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # Calculate R0 parameter (vdW radii sum)
        r0_1 = UFF_VDW_distance_lib(elem1) / self.bohr2angstroms
        r0_2 = UFF_VDW_distance_lib(elem2) / self.bohr2angstroms
        r0_param = r0_1 + r0_2
        
        return c6_param, c8_param, r0_param
    
    def calc_d3_gradient_components(self, x_ij, y_ij, z_ij, c6_param, c8_param, r0_param):
        """
        Calculate D3 dispersion gradient components.
        
        Args:
            x_ij, y_ij, z_ij: Distance components
            c6_param: C6 dispersion coefficient
            c8_param: C8 dispersion coefficient
            r0_param: van der Waals radius sum parameter
            
        Returns:
            tuple: (xx, xy, xz, yy, yz, zz) gradient components
        """
        r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
        r_ij = np.sqrt(r_ij_2)
        
        if r_ij < 0.1:  # Avoid numerical issues
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # BJ damping parameters
        r0_plus_a1 = r0_param + self.d3params.a1
        a2_term = self.d3params.a2
        bj_term_6 = (r0_plus_a1 * a2_term)**6
        bj_term_8 = (r0_plus_a1 * a2_term)**8
        
        # Calculate damping functions and their derivatives
        r_ij_6 = r_ij**6
        r_ij_8 = r_ij**8
        
        # C6 term: damping and derivatives
        damp_6 = r_ij_6 / (r_ij_6 + bj_term_6)
        d_damp_6_dr = 6.0 * r_ij_6 * bj_term_6 / ((r_ij_6 + bj_term_6)**2 * r_ij)
        
        # C8 term: damping and derivatives
        damp_8 = r_ij_8 / (r_ij_8 + bj_term_8)
        d_damp_8_dr = 8.0 * r_ij_8 * bj_term_8 / ((r_ij_8 + bj_term_8)**2 * r_ij)
        
        # Force (negative derivative of energy)
        f6 = self.d3params.s6 * c6_param * (6.0 * damp_6 / r_ij**7 + d_damp_6_dr / r_ij**6)
        f8 = self.d3params.s8 * c8_param * (8.0 * damp_8 / r_ij**9 + d_damp_8_dr / r_ij**8)
        
        # Total force
        force = f6 + f8
        
        # Calculate derivative components
        deriv_scale = force / r_ij
        
        # Calculate gradient components
        xx = deriv_scale * x_ij**2 / r_ij_2
        xy = deriv_scale * x_ij * y_ij / r_ij_2
        xz = deriv_scale * x_ij * z_ij / r_ij_2
        yy = deriv_scale * y_ij**2 / r_ij_2
        yz = deriv_scale * y_ij * z_ij / r_ij_2
        zz = deriv_scale * z_ij**2 / r_ij_2
        
        return xx, xy, xz, yy, yz, zz
    
    def lindh2007_bond(self, coord, element_list):
        """
        Calculate bond stretching contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(i):
                j_idx = self.select_idx(element_list[j])
                
                # Calculate distance components and magnitude
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                # Get Lindh parameters
                r_0 = self.rAv[i_idx][j_idx]
                d_0 = self.dAv[i_idx][j_idx]
                alpha = self.aAv[i_idx][j_idx]
                
                # Determine appropriate bond length based on bond type
                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                covalent_length = single_bond  # Default to single bond
                
                # Get D3 parameters and calculate dispersion contribution
                c6_param, c8_param, r0_param = self.get_d3_parameters(element_list[i], element_list[j])
                
                # Calculate force constants
                lindh_force = self.kr * self.calc_force_const(alpha, covalent_length, r_ij_2)
                
                # Add D3 dispersion if atoms are far apart
                d3_factor = 0.0
                if r_ij > 2.0 * covalent_length:
                    d3_factor = self.kd * self.calc_d3_force_const(r_ij, c6_param, c8_param, r0_param)
                
                # Combined force constant
                g_mm = lindh_force + d3_factor
                
                # Calculate D3 gradient components
                d3_xx, d3_xy, d3_xz, d3_yy, d3_yz, d3_zz = self.calc_d3_gradient_components(
                    x_ij, y_ij, z_ij, c6_param, c8_param, r0_param)
                
                # Calculate Hessian elements
                hess_xx = g_mm * x_ij**2 / r_ij_2 - d3_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - d3_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - d3_xz
                hess_yy = g_mm * y_ij**2 / r_ij_2 - d3_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - d3_yz
                hess_zz = g_mm * z_ij**2 / r_ij_2 - d3_zz
                
                # Update diagonal blocks
                i_offset = i * 3
                j_offset = j * 3
                
                # i-i block
                self.cart_hess[i_offset, i_offset] += hess_xx
                self.cart_hess[i_offset + 1, i_offset] += hess_xy
                self.cart_hess[i_offset + 1, i_offset + 1] += hess_yy
                self.cart_hess[i_offset + 2, i_offset] += hess_xz
                self.cart_hess[i_offset + 2, i_offset + 1] += hess_yz
                self.cart_hess[i_offset + 2, i_offset + 2] += hess_zz
                
                # j-j block
                self.cart_hess[j_offset, j_offset] += hess_xx
                self.cart_hess[j_offset + 1, j_offset] += hess_xy
                self.cart_hess[j_offset + 1, j_offset + 1] += hess_yy
                self.cart_hess[j_offset + 2, j_offset] += hess_xz
                self.cart_hess[j_offset + 2, j_offset + 1] += hess_yz
                self.cart_hess[j_offset + 2, j_offset + 2] += hess_zz
                
                # i-j block
                self.cart_hess[i_offset, j_offset] -= hess_xx
                self.cart_hess[i_offset, j_offset + 1] -= hess_xy
                self.cart_hess[i_offset, j_offset + 2] -= hess_xz
                self.cart_hess[i_offset + 1, j_offset] -= hess_xy
                self.cart_hess[i_offset + 1, j_offset + 1] -= hess_yy
                self.cart_hess[i_offset + 1, j_offset + 2] -= hess_yz
                self.cart_hess[i_offset + 2, j_offset] -= hess_xz
                self.cart_hess[i_offset + 2, j_offset + 1] -= hess_yz
                self.cart_hess[i_offset + 2, j_offset + 2] -= hess_zz
    
    def lindh2007_angle(self, coord, element_list):
        """
        Calculate angle bending contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                j_idx = self.select_idx(element_list[j])
                
                # Vector from j to i
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                # Get bond parameters
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get Lindh parameters
                r_ij_0 = self.rAv[i_idx][j_idx]
                d_ij_0 = self.dAv[i_idx][j_idx]
                alpha_ij = self.aAv[i_idx][j_idx]
                
                # Loop through potential third atoms to form an angle
                for k in range(j):
                    if i == k:
                        continue
                    
                    k_idx = self.select_idx(element_list[k])
                    
                    # Get parameters for i-k interaction
                    r_ik_0 = self.rAv[i_idx][k_idx]
                    d_ik_0 = self.dAv[i_idx][k_idx]
                    alpha_ik = self.aAv[i_idx][k_idx]
                    
                    # Vector from k to i
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    
                    # Get bond parameters
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    # Check if angle is well-defined (not linear)
                    cos_angle = (x_ij * x_ik + y_ij * y_ik + z_ij * z_ik) / (r_ij * r_ik)
                    
                    if abs(cos_angle - 1.0) < self.eps:
                        continue  # Skip near-linear angles
                    
                    # Vector from k to j
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    # Calculate force constants with D3 contributions
                    c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                    c6_ik, c8_ik, r0_ik = self.get_d3_parameters(element_list[i], element_list[k])
                    
                    g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                    if r_ij > 2.0 * covalent_length_ij:
                        g_ij += 0.5 * self.kd * self.calc_d3_force_const(r_ij, c6_ij, c8_ij, r0_ij)
                    
                    g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2)
                    if r_ik > 2.0 * covalent_length_ik:
                        g_ik += 0.5 * self.kd * self.calc_d3_force_const(r_ik, c6_ik, c8_ik, r0_ik)
                    
                    # Angular force constant
                    g_jk = self.kf * (g_ij + 0.5 * self.kd / self.kr * d_ij_0) * (g_ik + 0.5 * self.kd / self.kr * d_ik_0)
                    
                    # Cross product magnitude for sin(theta)
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik)**2 + (z_ij * x_ik - x_ij * z_ik)**2 + (x_ij * y_ik - y_ij * x_ik)**2
                    r_cross = np.sqrt(r_cross_2) if r_cross_2 > 1.0e-12 else 0.0
                    
                    # Skip if distances are too small
                    if r_ik <= self.eps or r_ij <= self.eps or r_jk <= self.eps:
                        continue
                    
                    # Calculate angle and its derivatives
                    dot_product = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    cos_theta = dot_product / (r_ij * r_ik)
                    sin_theta = r_cross / (r_ij * r_ik)
                    
                    if sin_theta > self.eps:  # Non-linear case
                        # Calculate derivatives
                        s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                        s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                        s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                        
                        s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                        s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                        s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                        
                        s_xi = -s_xj - s_xk
                        s_yi = -s_yj - s_yk
                        s_zi = -s_zj - s_zk
                        
                        s_i = [s_xi, s_yi, s_zi]
                        s_j = [s_xj, s_yj, s_zj]
                        s_k = [s_xk, s_yk, s_zk]
                        
                        # Update Hessian elements
                        for l in range(3):
                            for m in range(3):
                                # i-j block
                                if i > j:
                                    self.cart_hess[i*3+l, j*3+m] += g_jk * s_i[l] * s_j[m]
                                else:
                                    self.cart_hess[j*3+l, i*3+m] += g_jk * s_j[l] * s_i[m]
                                
                                # i-k block
                                if i > k:
                                    self.cart_hess[i*3+l, k*3+m] += g_jk * s_i[l] * s_k[m]
                                else:
                                    self.cart_hess[k*3+l, i*3+m] += g_jk * s_k[l] * s_i[m]
                                
                                # j-k block
                                if j > k:
                                    self.cart_hess[j*3+l, k*3+m] += g_jk * s_j[l] * s_k[m]
                                else:
                                    self.cart_hess[k*3+l, j*3+m] += g_jk * s_k[l] * s_j[m]
                        
                        # Diagonal blocks
                        for l in range(3):
                            for m in range(l):
                                self.cart_hess[j*3+l, j*3+m] += g_jk * s_j[l] * s_j[m]
                                self.cart_hess[i*3+l, i*3+m] += g_jk * s_i[l] * s_i[m]
                                self.cart_hess[k*3+l, k*3+m] += g_jk * s_k[l] * s_k[m]
                    
                    else:  # Linear case
                        # Handle linear angles using arbitrary perpendicular vectors
                        if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                            x_1, y_1, z_1 = -y_ij, x_ij, 0.0
                            x_2, y_2, z_2 = -x_ij * z_ij, -y_ij * z_ij, x_ij**2 + y_ij**2
                        else:
                            x_1, y_1, z_1 = 1.0, 0.0, 0.0
                            x_2, y_2, z_2 = 0.0, 1.0, 0.0
                        
                        x = [x_1, x_2]
                        y = [y_1, y_2]
                        z = [z_1, z_2]
                        
                        # Calculate derivatives for two perpendicular directions
                        for ii in range(2):
                            r_1 = np.sqrt(x[ii]**2 + y[ii]**2 + z[ii]**2)
                            cos_theta_x = x[ii] / r_1
                            cos_theta_y = y[ii] / r_1
                            cos_theta_z = z[ii] / r_1
                            
                            # Derivatives
                            s_xj = -cos_theta_x / r_ij
                            s_yj = -cos_theta_y / r_ij
                            s_zj = -cos_theta_z / r_ij
                            s_xk = -cos_theta_x / r_ik
                            s_yk = -cos_theta_y / r_ik
                            s_zk = -cos_theta_z / r_ik
                            
                            s_xi = -s_xj - s_xk
                            s_yi = -s_yj - s_yk
                            s_zi = -s_zj - s_zk
                            
                            s_i = [s_xi, s_yi, s_zi]
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            
                            # Update Hessian elements
                            for l in range(3):
                                for m in range(3):
                                    # i-j block
                                    if i > j:
                                        self.cart_hess[i*3+l, j*3+m] += g_jk * s_i[l] * s_j[m]
                                    else:
                                        self.cart_hess[j*3+l, i*3+m] += g_jk * s_j[l] * s_i[m]
                                    
                                    # i-k block
                                    if i > k:
                                        self.cart_hess[i*3+l, k*3+m] += g_jk * s_i[l] * s_k[m]
                                    else:
                                        self.cart_hess[k*3+l, i*3+m] += g_jk * s_k[l] * s_i[m]
                                    
                                    # j-k block
                                    if j > k:
                                        self.cart_hess[j*3+l, k*3+m] += g_jk * s_j[l] * s_k[m]
                                    else:
                                        self.cart_hess[k*3+l, j*3+m] += g_jk * s_k[l] * s_j[m]
                            
                            # Diagonal blocks
                            for l in range(3):
                                for m in range(l):
                                    self.cart_hess[j*3+l, j*3+m] += g_jk * s_j[l] * s_j[m]
                                    self.cart_hess[i*3+l, i*3+m] += g_jk * s_i[l] * s_i[m]
                                    self.cart_hess[k*3+l, k*3+m] += g_jk * s_k[l] * s_k[m]
    
    def lindh2007_dihedral_angle(self, coord, element_list):
        """
        Calculate dihedral angle (torsion) contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for j in range(n_atoms):
            t_xyz_2 = coord[j]
            
            for k in range(j+1, n_atoms):
                t_xyz_3 = coord[k]
                
                for i in range(j):
                    if i == k:
                        continue
                        
                    t_xyz_1 = coord[i]
                    
                    for l in range(k+1, n_atoms):
                        if l == i or l == j:
                            continue
                        
                        t_xyz_4 = coord[l]
                        
                        # Get element indices for parameter lookup
                        i_idx = self.select_idx(element_list[i])
                        j_idx = self.select_idx(element_list[j])
                        k_idx = self.select_idx(element_list[k])
                        l_idx = self.select_idx(element_list[l])
                        
                        # Get Lindh parameters
                        r_ij_0 = self.rAv[i_idx][j_idx]
                        d_ij_0 = self.dAv[i_idx][j_idx]
                        alpha_ij = self.aAv[i_idx][j_idx]
                        
                        r_jk_0 = self.rAv[j_idx][k_idx]
                        d_jk_0 = self.dAv[j_idx][k_idx]
                        alpha_jk = self.aAv[j_idx][k_idx]
                        
                        r_kl_0 = self.rAv[k_idx][l_idx]
                        d_kl_0 = self.dAv[k_idx][l_idx]
                        alpha_kl = self.aAv[k_idx][l_idx]
                        
                        # Calculate bond vectors and lengths
                        r_ij = coord[i] - coord[j]
                        r_jk = coord[j] - coord[k]
                        r_kl = coord[k] - coord[l]
                        
                        # Get bond parameters
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        # Calculate squared distances
                        r_ij_2 = np.sum(r_ij**2)
                        r_jk_2 = np.sum(r_jk**2)
                        r_kl_2 = np.sum(r_kl**2)
                        
                        # Calculate norms
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                        
                        # Check if near-linear angles would cause numerical issues
                        a35 = (35.0/180) * np.pi
                        cosfi_max = np.cos(a35)
                        
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                            
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        
                        # Get D3 parameters for bond pairs
                        c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                        c6_jk, c8_jk, r0_jk = self.get_d3_parameters(element_list[j], element_list[k])
                        c6_kl, c8_kl, r0_kl = self.get_d3_parameters(element_list[k], element_list[l])
                        
                        # Calculate force constants with D3 contributions
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                        if norm_r_ij > 2.0 * covalent_length_ij:
                            g_ij += 0.5 * self.kd * self.calc_d3_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        
                        g_jk = self.calc_force_const(alpha_jk, covalent_length_jk, r_jk_2)
                        if norm_r_jk > 2.0 * covalent_length_jk:
                            g_jk += 0.5 * self.kd * self.calc_d3_force_const(norm_r_jk, c6_jk, c8_jk, r0_jk)
                        
                        g_kl = self.calc_force_const(alpha_kl, covalent_length_kl, r_kl_2)
                        if norm_r_kl > 2.0 * covalent_length_kl:
                            g_kl += 0.5 * self.kd * self.calc_d3_force_const(norm_r_kl, c6_kl, c8_kl, r0_kl)
                        
                        # Calculate torsion force constant
                        t_ij = self.kt * (g_ij * 0.5 * self.kd / self.kr * d_ij_0) * \
                              (g_jk * 0.5 * self.kd / self.kr * d_jk_0) * \
                              (g_kl * 0.5 * self.kd / self.kr * d_kl_0)
                        
                        # Calculate torsion angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        
                        # Extract derivatives
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update off-diagonal blocks
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3*i+n, 3*j+m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3*i+n, 3*k+m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3*i+n, 3*l+m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3*j+n, 3*k+m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3*j+n, 3*l+m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3*k+n, 3*l+m] += t_ij * s_k[n] * s_l[m]
                        
                        # Update diagonal blocks (lower triangle)
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3*i+n, 3*i+m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3*j+n, 3*j+m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3*k+n, 3*k+m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3*l+n, 3*l+m] += t_ij * s_l[n] * s_l[m]
    
    def lindh2007_out_of_plane(self, coord, element_list):
        """
        Calculate out-of-plane bending contributions to the Hessian.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
        """
        n_atoms = len(coord)
        
        for i in range(n_atoms):
            t_xyz_4 = coord[i]
            
            for j in range(i+1, n_atoms):
                t_xyz_1 = coord[j]
                
                for k in range(j+1, n_atoms):
                    t_xyz_2 = coord[k]
                    
                    for l in range(k+1, n_atoms):
                        t_xyz_3 = coord[l]
                        
                        # Calculate bond vectors
                        r_ij = coord[i] - coord[j]
                        r_ik = coord[i] - coord[k]
                        r_il = coord[i] - coord[l]
                        
                        # Get bond parameters
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        # Get element indices for parameter lookup
                        idx_i = self.select_idx(element_list[i])
                        idx_j = self.select_idx(element_list[j])
                        idx_k = self.select_idx(element_list[k])
                        idx_l = self.select_idx(element_list[l])
                        
                        # Get Lindh parameters
                        d_ij_0 = self.dAv[idx_i][idx_j]
                        r_ij_0 = self.rAv[idx_i][idx_j]
                        alpha_ij = self.aAv[idx_i][idx_j]
                        
                        d_ik_0 = self.dAv[idx_i][idx_k]
                        r_ik_0 = self.rAv[idx_i][idx_k]
                        alpha_ik = self.aAv[idx_i][idx_k]
                        
                        d_il_0 = self.dAv[idx_i][idx_l]
                        r_il_0 = self.rAv[idx_i][idx_l]
                        alpha_il = self.aAv[idx_i][idx_l]
                        
                        # Calculate squared distances
                        r_ij_2 = np.sum(r_ij**2)
                        r_ik_2 = np.sum(r_ik**2)
                        r_il_2 = np.sum(r_il**2)
                        
                        # Calculate norms
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        # Check for near-linear angles that would cause numerical issues
                        cosfi2 = np.dot(r_ij, r_ik) / (norm_r_ij * norm_r_ik)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                            
                        cosfi3 = np.dot(r_ij, r_il) / (norm_r_ij * norm_r_il)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                            
                        cosfi4 = np.dot(r_ik, r_il) / (norm_r_ik * norm_r_il)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        # Get D3 parameters for each pair
                        c6_ij, c8_ij, r0_ij = self.get_d3_parameters(element_list[i], element_list[j])
                        c6_ik, c8_ik, r0_ik = self.get_d3_parameters(element_list[i], element_list[k])
                        c6_il, c8_il, r0_il = self.get_d3_parameters(element_list[i], element_list[l])
                        
                        # Disable direct D3 contributions to out-of-plane terms
                        kd = 0.0
                        
                        # Calculate force constants for each bond
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2)
                        if norm_r_ij > 2.0 * covalent_length_ij:
                            g_ij += 0.5 * kd * self.calc_d3_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        
                        g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2)
                        if norm_r_ik > 2.0 * covalent_length_ik:
                            g_ik += 0.5 * kd * self.calc_d3_force_const(norm_r_ik, c6_ik, c8_ik, r0_ik)
                        
                        g_il = self.calc_force_const(alpha_il, covalent_length_il, r_il_2)
                        if norm_r_il > 2.0 * covalent_length_il:
                            g_il += 0.5 * kd * self.calc_d3_force_const(norm_r_il, c6_il, c8_il, r0_il)
                        
                        # Combined force constant for out-of-plane motion
                        t_ij = self.ko * g_ij * g_ik * g_il
                        
                        # Calculate out-of-plane angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        # Extract derivatives
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update off-diagonal blocks
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i*3+n, j*3+m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i*3+n, k*3+m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i*3+n, l*3+m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j*3+n, k*3+m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j*3+n, l*3+m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k*3+n, l*3+m] += t_ij * s_k[n] * s_l[m]
                        
                        # Update diagonal blocks (lower triangle)
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i*3+n, i*3+m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j*3+n, j*3+m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k*3+n, k*3+m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l*3+n, l*3+m] += t_ij * s_l[n] * s_l[m]
    
    def main(self, coord, element_list, cart_gradient):
        """
        Calculate approximate Hessian using Lindh's 2007 model with D3 dispersion.
        
        Args:
            coord: Atomic coordinates (Bohr)
            element_list: List of element symbols
            cart_gradient: Cartesian gradient vector
            
        Returns:
            hess_proj: Projected approximate Hessian matrix
        """
        print("Generating Lindh's (2007) approximate Hessian with D3 dispersion...")
        
        # Scale eigenvalues based on gradient norm (smaller scale for larger gradients)
        norm_grad = np.linalg.norm(cart_gradient)
        scale = 0.1
        eigval_scale = scale * np.exp(-1 * norm_grad**2.0)
        
        # Initialize Hessian matrix
        n_atoms = len(coord)
        self.cart_hess = np.zeros((n_atoms*3, n_atoms*3), dtype="float64")
        
        # Calculate individual contributions
        self.lindh2007_bond(coord, element_list)
        self.lindh2007_angle(coord, element_list)
        self.lindh2007_dihedral_angle(coord, element_list)
        self.lindh2007_out_of_plane(coord, element_list)
        
        # Symmetrize the Hessian matrix
        for i in range(n_atoms*3):
            for j in range(i):
                if abs(self.cart_hess[i, j]) < 1.0e-10:
                    self.cart_hess[i, j] = self.cart_hess[j, i]
                else:
                    self.cart_hess[j, i] = self.cart_hess[i, j]
        
        # Project out translational and rotational degrees of freedom
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        
        # Adjust eigenvalues for stability based on gradient magnitude
        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        hess_proj = np.dot(np.dot(eigenvectors, np.diag(np.abs(eigenvalues) * eigval_scale)), 
                          np.transpose(eigenvectors))
        
        return hess_proj
