import numpy as np

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib, UFF_VDW_distance_lib, D4Parameters, triple_covalent_radii_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.ModelHessian.calc_params import torsion2, outofplane2
from multioptpy.Parameters.parameter import UFF_VDW_distance_lib


class SwartD4ApproxHessian:
    def __init__(self):
        #Swart's Model Hessian augmented with D4 dispersion
        #ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536–2544.
        #ref.: E. Caldeweyher, C. Bannwarth, S. Grimme, J. Chem. Phys., 2017, 147, 034112
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00  # Dispersion scaling factor
        
        self.kr = 0.35  # Bond stretching force constant scaling
        self.kf = 0.15  # Angle bending force constant scaling
        self.kt = 0.005 # Torsional force constant scaling
        
        self.cutoff = 70.0  # Cutoff for long-range interactions
        self.eps = 1.0e-12  # Small number for numerical stability
        
        # D4 parameters
        self.d4_params = D4Parameters()
        
        # Cyano group parameters
        self.cn_kr = 0.70  # Enhanced force constant for C≡N triple bond
        self.cn_kf = 0.20  # Enhanced force constant for angles involving C≡N
        self.cn_kt = 0.002 # Reduced force constant for torsions involving C≡N
        return
    
    def detect_cyano_groups(self, coord, element_list):
        """Detect C≡N triple bonds in the structure"""
        cyano_atoms = []  # List of (C_idx, N_idx) tuples
        
        for i in range(len(coord)):
            if element_list[i] != 'C':
                continue
                
            for j in range(len(coord)):
                if i == j or element_list[j] != 'N':
                    continue
                    
                # Calculate distance between C and N
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij = np.sqrt(x_ij**2 + y_ij**2 + z_ij**2)
                
                # Check if distance is close to a triple bond length
                cn_triple_bond = triple_covalent_radii_lib('C') + triple_covalent_radii_lib('N')
                
                if abs(r_ij - cn_triple_bond) < 0.3:  # Within 0.3 bohr of ideal length
                    # Check if C is connected to only one other atom (besides N)
                    connections_to_c = 0
                    for k in range(len(coord)):
                        if k == i or k == j:
                            continue
                            
                        x_ik = coord[i][0] - coord[k][0]
                        y_ik = coord[i][1] - coord[k][1]
                        z_ik = coord[i][2] - coord[k][2]
                        r_ik = np.sqrt(x_ik**2 + y_ik**2 + z_ik**2)
                        
                        cov_dist = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        if r_ik < 1.3 * cov_dist:  # Using 1.3 as a factor to account for bond length variations
                            connections_to_c += 1
                    
                    # If C has only one other connection, it's likely a terminal cyano group
                    if connections_to_c <= 1:
                        cyano_atoms.append((i, j))
        
        return cyano_atoms
    
    def calculate_coordination_numbers(self, coord, element_list):
        """Calculate atomic coordination numbers for D4 scaling"""
        n_atoms = len(coord)
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                # Calculate distance
                r_ij = np.linalg.norm(coord[i] - coord[j])
                
                # Get covalent radii
                r_cov_i = covalent_radii_lib(element_list[i])
                r_cov_j = covalent_radii_lib(element_list[j])
                
                # Coordination number contribution using counting function
                # k1 = 16.0, k2 = 4.0/3.0 (standard values from DFT-D4)
                k1 = 16.0
                k2 = 4.0/3.0
                r0 = r_cov_i + r_cov_j
                
                # Avoid overflow in exp
                if k1 * (r_ij / r0 - 1.0) > 25.0:
                    continue
                
                cn_contrib = 1.0 / (1.0 + np.exp(-k1 * (k2 * r0 / r_ij - 1.0)))
                cn[i] += cn_contrib
        
        return cn
    
    def estimate_atomic_charges(self, coord, element_list):
        """
        Estimate atomic charges using electronegativity equalization
        Simplified version for Hessian generation
        """
        n_atoms = len(coord)
        charges = np.zeros(n_atoms)
        
        # Calculate reference electronegativities
        en_list = [self.d4_params.get_electronegativity(elem) for elem in element_list]
        
        # Simple charge estimation based on electronegativity differences
        # This is a very simplified model
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coord[i] - coord[j])
                r_cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Only consider bonded atoms
                if r_ij < 1.3 * r_cov_sum:
                    en_diff = en_list[j] - en_list[i]
                    charge_transfer = 0.1 * en_diff  # Simple approximation
                    charges[i] += charge_transfer
                    charges[j] -= charge_transfer
        
        return charges
    
    def calc_force_const(self, alpha, covalent_length, distance):
        """Calculate force constant with exponential damping"""
        force_const = np.exp(-1 * alpha * (distance / covalent_length - 1.0))
        return force_const
        
    def calc_vdw_force_const(self, alpha, r0, distance):
        """Calculate van der Waals force constant with exponential damping"""
        vdw_force_const = np.exp(-1 * alpha * (r0 - distance) ** 2)
        return vdw_force_const
        
    def d4_damping_function(self, r_ij, r0, order=6):
        """D4 rational damping function"""
        a1 = self.d4_params.a1
        a2 = self.d4_params.a2
        
        if order == 6:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a1 * r0)) ** a2)
        elif order == 8:
            return 1.0 / (1.0 + 6.0 * (r_ij / (a2 * r0)) ** a1)
        return 0.0
    
    def charge_scale_factor(self, charge, element):
        """Calculate charge scaling factor for D4"""
        ga = self.d4_params.ga  # D4 charge scaling parameter
        q_ref = self.d4_params.get_electronegativity(element)
        
        # Prevent numerical issues with large exponents
        exp_arg = -ga * abs(charge)
        if exp_arg < -50.0:  # Avoid underflow
            return 0.0
            
        return np.exp(exp_arg)
    
    def get_d4_parameters(self, elem1, elem2, q1=0.0, q2=0.0, cn1=None, cn2=None):
        """Get D4 parameters for a pair of elements with charge scaling"""
        # Get polarizabilities
        alpha1 = self.d4_params.get_polarizability(elem1)
        alpha2 = self.d4_params.get_polarizability(elem2)
        
        # Charge scaling
        qscale1 = self.charge_scale_factor(q1, elem1)
        qscale2 = self.charge_scale_factor(q2, elem2)
        
        # Get R4/R2 values
        r4r2_1 = self.d4_params.get_r4r2(elem1)
        r4r2_2 = self.d4_params.get_r4r2(elem2)
        
        # C6 coefficients with charge scaling
        c6_1 = alpha1 * r4r2_1 * qscale1
        c6_2 = alpha2 * r4r2_2 * qscale2
        c6_param = 2.0 * c6_1 * c6_2 / (c6_1 + c6_2)  # Effective C6 using harmonic mean
        
        # C8 coefficients
        c8_param = 3.0 * c6_param * np.sqrt(r4r2_1 * r4r2_2)
        
        # r0 parameter (combined vdW radii)
        r0_param = np.sqrt(UFF_VDW_distance_lib(elem1) * UFF_VDW_distance_lib(elem2))
        
        return c6_param, c8_param, r0_param
    
    def calc_d4_force_const(self, r_ij, c6_param, c8_param, r0_param):
        """Calculate D4 dispersion force constant"""
        s6 = self.d4_params.s6
        s8 = self.d4_params.s8
        
        # Apply damping functions
        damp6 = self.d4_damping_function(r_ij, r0_param, order=6)
        damp8 = self.d4_damping_function(r_ij, r0_param, order=8)
        
        # Energy terms (negative because dispersion is attractive)
        e6 = -s6 * c6_param / r_ij ** 6 * damp6
        e8 = -s8 * c8_param / r_ij ** 8 * damp8
        
        # Force constant is the second derivative of energy
        fc6 = s6 * c6_param * (42.0 / r_ij ** 8) * damp6
        fc8 = s8 * c8_param * (72.0 / r_ij ** 10) * damp8
        
        return fc6 + fc8
    
    def swart_bond(self, coord, element_list, charges, cn):
        """Calculate bond stretching contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for i in range(len(coord)):
            for j in range(i):
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get D4 parameters with charge scaling
                c6_param, c8_param, r0_param = self.get_d4_parameters(
                    element_list[i], element_list[j], 
                    q1=charges[i], q2=charges[j], 
                    cn1=cn[i], cn2=cn[j]
                )
                
                # Calculate D4 dispersion contribution
                d4_force_const = self.calc_d4_force_const(r_ij, c6_param, c8_param, r0_param)
                
                # Check if this is a cyano bond
                is_cyano_bond = False
                for c_idx, n_idx in cyano_atoms:
                    if (i == c_idx and j == n_idx) or (i == n_idx and j == c_idx):
                        is_cyano_bond = True
                        break
                
                # Apply appropriate force constant
                if is_cyano_bond:
                    # Special force constant for C≡N triple bond
                    g_mm = self.cn_kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * d4_force_const
                else:
                    # Regular Swart force constant with D4 dispersion
                    g_mm = self.kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * d4_force_const
                
                # Calculate Hessian components
                hess_xx = g_mm * x_ij ** 2 / r_ij_2
                hess_xy = g_mm * x_ij * y_ij / r_ij_2
                hess_xz = g_mm * x_ij * z_ij / r_ij_2
                hess_yy = g_mm * y_ij ** 2 / r_ij_2
                hess_yz = g_mm * y_ij * z_ij / r_ij_2
                hess_zz = g_mm * z_ij ** 2 / r_ij_2
                
                # Fill the Hessian matrix
                self.cart_hess[i * 3][i * 3] += hess_xx
                self.cart_hess[i * 3 + 1][i * 3] += hess_xy
                self.cart_hess[i * 3 + 1][i * 3 + 1] += hess_yy
                self.cart_hess[i * 3 + 2][i * 3] += hess_xz
                self.cart_hess[i * 3 + 2][i * 3 + 1] += hess_yz
                self.cart_hess[i * 3 + 2][i * 3 + 2] += hess_zz
                
                self.cart_hess[j * 3][j * 3] += hess_xx
                self.cart_hess[j * 3 + 1][j * 3] += hess_xy
                self.cart_hess[j * 3 + 1][j * 3 + 1] += hess_yy
                self.cart_hess[j * 3 + 2][j * 3] += hess_xz
                self.cart_hess[j * 3 + 2][j * 3 + 1] += hess_yz
                self.cart_hess[j * 3 + 2][j * 3 + 2] += hess_zz
                
                self.cart_hess[i * 3][j * 3] -= hess_xx
                self.cart_hess[i * 3][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3][j * 3 + 2] -= hess_xz
                self.cart_hess[i * 3 + 1][j * 3] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3] -= hess_xz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
                
                self.cart_hess[j * 3][i * 3] -= hess_xx
                self.cart_hess[j * 3][i * 3 + 1] -= hess_xy
                self.cart_hess[j * 3][i * 3 + 2] -= hess_xz
                self.cart_hess[j * 3 + 1][i * 3] -= hess_xy
                self.cart_hess[j * 3 + 1][i * 3 + 1] -= hess_yy
                self.cart_hess[j * 3 + 1][i * 3 + 2] -= hess_yz
                self.cart_hess[j * 3 + 2][i * 3] -= hess_xz
                self.cart_hess[j * 3 + 2][i * 3 + 1] -= hess_yz
                self.cart_hess[j * 3 + 2][i * 3 + 2] -= hess_zz
        
        return
    
    def swart_angle(self, coord, element_list, charges, cn):
        """Calculate angle bending contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        for i in range(len(coord)):
            for j in range(len(coord)):
                if i == j:
                    continue
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                
                # Get D4 parameters with charge scaling for i-j pair
                c6_ij, c8_ij, r0_ij = self.get_d4_parameters(
                    element_list[i], element_list[j], 
                    q1=charges[i], q2=charges[j], 
                    cn1=cn[i], cn2=cn[j]
                )
                
                for k in range(j):
                    if i == k:
                        continue
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    # Check for linear arrangement (cos_theta ~ 1.0)
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    # Get D4 parameters with charge scaling for i-k pair
                    c6_ik, c8_ik, r0_ik = self.get_d4_parameters(
                        element_list[i], element_list[k], 
                        q1=charges[i], q2=charges[k], 
                        cn1=cn[i], cn2=cn[k]
                    )
                    
                    # Calculate D4 dispersion contributions
                    d4_ij = self.calc_d4_force_const(r_ij, c6_ij, c8_ij, r0_ij)
                    d4_ik = self.calc_d4_force_const(r_ik, c6_ik, c8_ik, r0_ik)
                    
                    # Calculate bond force constants with D4 dispersion
                    g_ij = self.calc_force_const(1.0, covalent_length_ij, r_ij) + 0.5 * self.kd * d4_ij
                    g_ik = self.calc_force_const(1.0, covalent_length_ik, r_ik) + 0.5 * self.kd * d4_ik
                    
                    # Check if angle involves cyano group
                    is_cyano_angle = (i in cyano_set or j in cyano_set or k in cyano_set)
                    
                    # Apply appropriate force constant
                    if is_cyano_angle:
                        # Special force constant for angles involving cyano groups
                        g_jk = self.cn_kf * g_ij * g_ik
                    else:
                        # Regular Swart force constant
                        g_jk = self.kf * g_ij * g_ik
                    
                    # Calculate cross product for sin(theta)
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:
                        cos_theta = (r_ij_2 + r_ik_2 - r_jk_2) / (2.0 * r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                       
                        if sin_theta > self.eps: # non-linear
                            # Calculate derivatives for non-linear case
                            s_xj = (x_ij / r_ij * cos_theta - x_ik / r_ik) / (r_ij * sin_theta)
                            s_yj = (y_ij / r_ij * cos_theta - y_ik / r_ik) / (r_ij * sin_theta)
                            s_zj = (z_ij / r_ij * cos_theta - z_ik / r_ik) / (r_ij * sin_theta)
                            
                            s_xk = (x_ik / r_ik * cos_theta - x_ij / r_ij) / (r_ik * sin_theta)
                            s_yk = (y_ik / r_ik * cos_theta - y_ij / r_ij) / (r_ik * sin_theta)
                            s_zk = (z_ik / r_ik * cos_theta - z_ij / r_ij) / (r_ik * sin_theta)
                            
                            s_xi = -1 * s_xj - s_xk
                            s_yi = -1 * s_yj - s_yk
                            s_zi = -1 * s_zj - s_zk
                            
                            s_j = [s_xj, s_yj, s_zj]
                            s_k = [s_xk, s_yk, s_zk]
                            s_i = [s_xi, s_yi, s_zi]
                            
                            # Update Hessian for non-linear case
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        tmp_val = g_jk * s_i[l] * s_j[m]
                                        self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                    else:
                                        tmp_val = g_jk * s_j[l] * s_i[m]
                                        self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                    
                                    #-------------------------------------
                                    if i > k:
                                        tmp_val = g_jk * s_i[l] * s_k[m]
                                        self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_i[m]
                                        self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                            
                                    #-------------------------------------
                                    if j > k:
                                        tmp_val = g_jk * s_j[l] * s_k[m]
                                        self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                    else:
                                        tmp_val = g_jk * s_k[l] * s_j[m]
                                        self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                    #-------------------------------------
                                    
                            # Update diagonal blocks
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
                            # Special handling for linear arrangements
                            if abs(y_ij) < self.eps and abs(z_ij) < self.eps:
                                x_1 = -1 * y_ij
                                y_1 = x_ij
                                z_1 = 0.0
                                x_2 = -1 * x_ij * z_ij
                                y_2 = -1 * y_ij * z_ij
                                z_2 = x_ij ** 2 + y_ij ** 2
                            else:
                                x_1 = 1.0
                                y_1 = 0.0
                                z_1 = 0.0
                                x_2 = 0.0
                                y_2 = 1.0
                                z_2 = 0.0
                            
                            x = [x_1, x_2]
                            y = [y_1, y_2]
                            z = [z_1, z_2]
                            
                            # Iterate over two perpendicular directions
                            for ii in range(2):
                                r_1 = np.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                                cos_theta_x = x[ii] / r_1
                                cos_theta_y = y[ii] / r_1
                                cos_theta_z = z[ii] / r_1
                                
                                s_xj = -1 * cos_theta_x / r_ij
                                s_yj = -1 * cos_theta_y / r_ij
                                s_zj = -1 * cos_theta_z / r_ij
                                s_xk = -1 * cos_theta_x / r_ik
                                s_yk = -1 * cos_theta_y / r_ik
                                s_zk = -1 * cos_theta_z / r_ik
                                
                                s_xi = -1 * s_xj - s_xk
                                s_yi = -1 * s_yj - s_yk
                                s_zi = -1 * s_zj - s_zk
                                
                                s_j = [s_xj, s_yj, s_zj]
                                s_k = [s_xk, s_yk, s_zk]
                                s_i = [s_xi, s_yi, s_zi]
                                
                                # Update Hessian for linear case
                                for l in range(3):
                                    for m in range(3):
                                        #-------------------------------------
                                        if i > j:
                                            tmp_val = g_jk * s_i[l] * s_j[m]
                                            self.cart_hess[i * 3 + l][j * 3 + m] += tmp_val     
                                        else:
                                            tmp_val = g_jk * s_j[l] * s_i[m]
                                            self.cart_hess[j * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if i > k:
                                            tmp_val = g_jk * s_i[l] * s_k[m]
                                            self.cart_hess[i * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_i[m]
                                            self.cart_hess[k * 3 + l][i * 3 + m] += tmp_val
                                        #-------------------------------------
                                        if j > k:
                                            tmp_val = g_jk * s_j[l] * s_k[m]
                                            self.cart_hess[j * 3 + l][k * 3 + m] += tmp_val
                                        else:
                                            tmp_val = g_jk * s_k[l] * s_j[m]
                                            self.cart_hess[k * 3 + l][j * 3 + m] += tmp_val
                                        #-------------------------------------
                                
                                # Update diagonal blocks for linear case
                                for l in range(3):
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass  # Skip if any distance is too small
        
        # Make the Hessian symmetric for angle terms
        n_basis = len(coord) * 3
        for i in range(n_basis):
            for j in range(i):
                if abs(self.cart_hess[i][j] - self.cart_hess[j][i]) > 1.0e-10:
                    avg = (self.cart_hess[i][j] + self.cart_hess[j][i]) / 2.0
                    self.cart_hess[i][j] = avg
                    self.cart_hess[j][i] = avg
                
        return
    
    def swart_dihedral_angle(self, coord, element_list, charges, cn):
        """Calculate dihedral angle contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
        
        for j in range(len(coord)):
            t_xyz_2 = coord[j] 
            
            for k in range(len(coord)):
                if j >= k:
                    continue
                t_xyz_3 = coord[k]
                for i in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= j:
                        continue
                    if i >= k:
                        continue
                    
                    t_xyz_1 = coord[i]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                       
                        if ij <= kl:
                            continue
                        if l >= i:
                            continue
                        if l >= j:
                            continue
                        if l >= k:
                            continue
                    
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        r_jk = coord[j] - coord[k]
                        r_kl = coord[k] - coord[l]
                        
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        # Calculate vector magnitudes
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        # Skip if angle is too shallow (less than 35 degrees)
                        a35 = (35.0/180)* np.pi
                        cosfi_max = np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        
                        # Get D4 parameters for each atom pair with charge scaling
                        c6_ij, c8_ij, r0_ij = self.get_d4_parameters(
                            element_list[i], element_list[j], 
                            q1=charges[i], q2=charges[j], 
                            cn1=cn[i], cn2=cn[j]
                        )
                        
                        c6_jk, c8_jk, r0_jk = self.get_d4_parameters(
                            element_list[j], element_list[k], 
                            q1=charges[j], q2=charges[k], 
                            cn1=cn[j], cn2=cn[k]
                        )
                        
                        c6_kl, c8_kl, r0_kl = self.get_d4_parameters(
                            element_list[k], element_list[l], 
                            q1=charges[k], q2=charges[l], 
                            cn1=cn[k], cn2=cn[l]
                        )
                        
                        # Calculate D4 dispersion contributions
                        d4_ij = self.calc_d4_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        d4_jk = self.calc_d4_force_const(norm_r_jk, c6_jk, c8_jk, r0_jk)
                        d4_kl = self.calc_d4_force_const(norm_r_kl, c6_kl, c8_kl, r0_kl)
                        
                        # Calculate bond force constants with D4 dispersion
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * d4_ij
                        g_jk = self.calc_force_const(1.0, covalent_length_jk, norm_r_jk) + 0.5 * self.kd * d4_jk
                        g_kl = self.calc_force_const(1.0, covalent_length_kl, norm_r_kl) + 0.5 * self.kd * d4_kl
                        
                        # Check if torsion involves cyano group
                        is_cyano_torsion = False
                        if i in cyano_set or j in cyano_set or k in cyano_set or l in cyano_set:
                            is_cyano_torsion = True
                        
                        # Adjust force constant for cyano groups - they have flatter torsional potentials
                        if is_cyano_torsion:
                            t_ij = self.cn_kt * g_ij * g_jk * g_kl
                        else:
                            t_ij = self.kt * g_ij * g_jk * g_kl
                        
                        # Calculate torsion angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        # Update Hessian with torsional contributions
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] += t_ij * s_k[n] * s_l[m]
                            
                        # Update diagonal blocks
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] += t_ij * s_l[n] * s_l[m]
                       
        return
    
    def swart_out_of_plane(self, coord, element_list, charges, cn):
        """Calculate out-of-plane bending contributions to the Hessian with D4 dispersion"""
        # Detect cyano groups
        cyano_atoms = self.detect_cyano_groups(coord, element_list)
        cyano_set = set()
        for c_idx, n_idx in cyano_atoms:
            cyano_set.add(c_idx)
            cyano_set.add(n_idx)
            
        for i in range(len(coord)):
            t_xyz_4 = coord[i]
            for j in range(len(coord)):
                if i >= j:
                    continue
                t_xyz_1 = coord[j]
                for k in range(len(coord)):
                    ij = (len(coord) * j) + (i + 1)
                    if i >= k:
                        continue
                    if j >= k:
                        continue
                    t_xyz_2 = coord[k]
                    
                    for l in range(len(coord)):
                        kl = (len(coord) * k) + (l + 1)
                        if i >= l:
                            continue
                        if j >= l:
                            continue
                        if k >= l:
                            continue
                        if ij <= kl:
                            continue
                        t_xyz_3 = coord[l]
                        
                        r_ij = coord[i] - coord[j]
                        r_ik = coord[i] - coord[k]
                        r_il = coord[i] - coord[l]
                        
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        # Skip if atoms are nearly collinear
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        
                        # Get D4 parameters with charge scaling for each atom pair
                        c6_ij, c8_ij, r0_ij = self.get_d4_parameters(
                            element_list[i], element_list[j], 
                            q1=charges[i], q2=charges[j], 
                            cn1=cn[i], cn2=cn[j]
                        )
                        
                        c6_ik, c8_ik, r0_ik = self.get_d4_parameters(
                            element_list[i], element_list[k], 
                            q1=charges[i], q2=charges[k], 
                            cn1=cn[i], cn2=cn[k]
                        )
                        
                        c6_il, c8_il, r0_il = self.get_d4_parameters(
                            element_list[i], element_list[l], 
                            q1=charges[i], q2=charges[l], 
                            cn1=cn[i], cn2=cn[l]
                        )
                        
                        # Calculate D4 dispersion contributions
                        d4_ij = self.calc_d4_force_const(norm_r_ij, c6_ij, c8_ij, r0_ij)
                        d4_ik = self.calc_d4_force_const(norm_r_ik, c6_ik, c8_ik, r0_ik)
                        d4_il = self.calc_d4_force_const(norm_r_il, c6_il, c8_il, r0_il)
                        
                        # Calculate bond force constants with D4 dispersion
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * d4_ij
                        g_ik = self.calc_force_const(1.0, covalent_length_ik, norm_r_ik) + 0.5 * self.kd * d4_ik
                        g_il = self.calc_force_const(1.0, covalent_length_il, norm_r_il) + 0.5 * self.kd * d4_il
                        
                        # Check if any atom is part of a cyano group
                        is_cyano_involved = (i in cyano_set or j in cyano_set or 
                                            k in cyano_set or l in cyano_set)
                        
                        # Adjust force constant if cyano group is involved
                        if is_cyano_involved:
                            t_ij = 0.5 * self.kt * g_ij * g_ik * g_il  # Reduce force constant for cyano
                        else:
                            t_ij = self.kt * g_ij * g_ik * g_il
                        
                        # Calculate out-of-plane angle and derivatives
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        # Update Hessian with out-of-plane contributions
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                        # Update diagonal blocks
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                        
        return
    
    def calculate_three_body_term(self, coord, element_list, charges, cn):
        """Calculate three-body dispersion contribution to the Hessian (D4 specific)"""
        s9 = self.d4_params.s9  # Scaling parameter for three-body term
        if abs(s9) < 1e-12:
            return  # Skip if three-body term is turned off
        
        n_atoms = len(coord)
        
        # Loop over all atom triplets
        for i in range(n_atoms):
            for j in range(i):
                for k in range(j):
                    # Get positions
                    r_i = coord[i]
                    r_j = coord[j]
                    r_k = coord[k]
                    
                    # Calculate interatomic distances
                    r_ij = np.linalg.norm(r_i - r_j)
                    r_jk = np.linalg.norm(r_j - r_k)
                    r_ki = np.linalg.norm(r_k - r_i)
                    
                    # Get coordination-number scaled C6 coefficients
                    c6_ij, _, r0_ij = self.get_d4_parameters(
                        element_list[i], element_list[j], 
                        q1=charges[i], q2=charges[j], 
                        cn1=cn[i], cn2=cn[j]
                    )
                    
                    c6_jk, _, r0_jk = self.get_d4_parameters(
                        element_list[j], element_list[k], 
                        q1=charges[j], q2=charges[k], 
                        cn1=cn[j], cn2=cn[k]
                    )
                    
                    c6_ki, _, r0_ki = self.get_d4_parameters(
                        element_list[k], element_list[i], 
                        q1=charges[k], q2=charges[i], 
                        cn1=cn[k], cn2=cn[i]
                    )
                    
                    # Calculate geometric mean of C6 coefficients
                    c9 = (c6_ij * c6_jk * c6_ki) ** (1.0/3.0)
                    
                    # Calculate three-body damping
                    damp_ij = self.d4_damping_function(r_ij, r0_ij)
                    damp_jk = self.d4_damping_function(r_jk, r0_jk)
                    damp_ki = self.d4_damping_function(r_ki, r0_ki)
                    damp = damp_ij * damp_jk * damp_ki
                    
                    # Calculate angle factor
                    cos_ijk = np.dot(r_i - r_j, r_k - r_j) / (r_ij * r_jk)
                    cos_jki = np.dot(r_j - r_k, r_i - r_k) / (r_jk * r_ki)
                    cos_kij = np.dot(r_k - r_i, r_j - r_i) / (r_ki * r_ij)
                    angle_factor = 1.0 + 3.0 * cos_ijk * cos_jki * cos_kij
                    
                    # Calculate three-body energy term
                    e_3 = s9 * angle_factor * c9 * damp / (r_ij * r_jk * r_ki) ** 3
                    
                    # Calculate force constants (second derivatives)
                    # This is a simplified approximation - full D4 three-body Hessian is complex
                    fc_scale = 0.01 * s9 * angle_factor * c9 * damp
                    
                    # Add approximate three-body contributions to Hessian
                    for n in range(3):
                        for m in range(3):
                            # Diagonal blocks (diagonal atoms)
                            if n == m:
                                self.cart_hess[i * 3 + n][i * 3 + m] += fc_scale / r_ij**6 + fc_scale / r_ki**6
                                self.cart_hess[j * 3 + n][j * 3 + m] += fc_scale / r_ij**6 + fc_scale / r_jk**6
                                self.cart_hess[k * 3 + n][k * 3 + m] += fc_scale / r_jk**6 + fc_scale / r_ki**6
                            
                            # Off-diagonal blocks (between atoms)
                            self.cart_hess[i * 3 + n][j * 3 + m] -= fc_scale / r_ij**6 
                            self.cart_hess[j * 3 + n][i * 3 + m] -= fc_scale / r_ij**6
                            
                            self.cart_hess[j * 3 + n][k * 3 + m] -= fc_scale / r_jk**6
                            self.cart_hess[k * 3 + n][j * 3 + m] -= fc_scale / r_jk**6
                            
                            self.cart_hess[k * 3 + n][i * 3 + m] -= fc_scale / r_ki**6
                            self.cart_hess[i * 3 + n][k * 3 + m] -= fc_scale / r_ki**6
        
        return
    
    def main(self, coord, element_list, cart_gradient):
        """Main method to calculate the approximate Hessian using Swart's model with D4 dispersion"""
        print("Generating Swart's approximate hessian with D4 dispersion correction...")
        self.cart_hess = np.zeros((len(coord) * 3, len(coord) * 3), dtype="float64")
        
        # Calculate coordination numbers and atomic charges for D4
        cn = self.calculate_coordination_numbers(coord, element_list)
        charges = self.estimate_atomic_charges(coord, element_list)
        
        # Calculate all contributions to the Hessian
        self.swart_bond(coord, element_list, charges, cn)
        self.swart_angle(coord, element_list, charges, cn)
        self.swart_dihedral_angle(coord, element_list, charges, cn)
        self.swart_out_of_plane(coord, element_list, charges, cn)
        
        # Add D4-specific three-body term
        self.calculate_three_body_term(coord, element_list, charges, cn)
        
        # Ensure symmetry of the Hessian matrix
        n_basis = len(coord) * 3
        for i in range(n_basis):
            for j in range(i):
                if abs(self.cart_hess[i][j] - self.cart_hess[j][i]) > 1.0e-10:
                    avg = (self.cart_hess[i][j] + self.cart_hess[j][i]) / 2.0
                    self.cart_hess[i][j] = avg
                    self.cart_hess[j][i] = avg
        
        # Project out translational and rotational modes
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj
