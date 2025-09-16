import numpy as np

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib, element_number
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.ModelHessian.calc_params import torsion2, outofplane2, calc_vdw_isotopic, calc_vdw_anisotropic
from multioptpy.Parameters.parameter import D2_C6_coeff_lib, UFF_VDW_distance_lib, double_covalent_radii_lib, triple_covalent_radii_lib, D2_VDW_radii_lib


class Lindh2007D2ApproxHessian:
    def __init__(self):
        #Lindh's Model Hessian (2007) augmented with D2
        #ref.: https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        self.bond_threshold_scale = 1.0
        self.kr = 0.45
        self.kf = 0.10
        self.kt = 0.0025
        self.ko = 0.16
        self.kd = 0.05
        self.cutoff = 50.0
        self.eps = 1.0e-12
        
        self.rAv = np.array([[1.3500,   2.1000,   2.5300],
                    [2.1000,   2.8700,   3.8000],
                    [2.5300,   3.8000,   4.5000]]) 
        
        self.aAv = np.array([[1.0000,   0.3949,   0.3949],
                    [0.3949,   0.2800,   0.1200],
                    [0.3949,   0.1200,   0.0600]]) 
        
        self.dAv = np.array([[0.0000,   3.6000,   3.6000],
                    [3.6000,   5.3000,   5.3000],
                    [3.6000,   5.3000,   5.3000]])
        
        #self.s6 = 20.0
        return
    
    def select_idx(self, elem_num):
        if type(elem_num) == str:
            elem_num = element_number(elem_num)

        if (elem_num > 0 and elem_num < 2):
            idx = 0
        elif (elem_num >= 2 and elem_num < 10):
            idx = 1
        elif (elem_num >= 10 and elem_num < 18):
            idx = 2
        elif (elem_num >= 18 and elem_num < 36):
            idx = 2
        elif (elem_num >= 36 and elem_num < 54):
            idx = 2
        elif (elem_num >= 54 and elem_num < 86):
            idx = 2
        elif (elem_num >= 86):
            idx = 2
        else:
            idx = 2

        return idx
    
    
    def calc_force_const(self, alpha, r_0, distance_2):
        force_const = np.exp(alpha * (r_0 ** 2 -1.0 * distance_2))
        
        return force_const
        
    def calc_vdw_force_const(self, alpha, vdw_length, distance):
        vdw_force_const = np.exp(-4 * alpha * (vdw_length - distance) ** 2)
        return vdw_force_const
    
    def lindh2007_bond(self, coord, element_list):
        for i in range(len(coord)):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(i):
                j_idx = self.select_idx(element_list[j])
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                
                r_0 = self.rAv[i_idx][j_idx]
                d_0 = self.dAv[i_idx][j_idx]
                alpha = self.aAv[i_idx][j_idx]
                
                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                
                #if self.bond_threshold_scale * triple_bond > r_ij:
                #    covalent_length = triple_bond
                #elif self.bond_threshold_scale * double_bond > r_ij:

                #    covalent_length = double_bond
                #else:
                covalent_length = single_bond
               
                vdw_length = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                C6_param_i = D2_C6_coeff_lib(element_list[i])
                C6_param_j = D2_C6_coeff_lib(element_list[j])
                C6_param_ij = np.sqrt(C6_param_i * C6_param_j)
                C6_VDW_ij = D2_VDW_radii_lib(element_list[i]) + D2_VDW_radii_lib(element_list[j])
                VDW_xx = calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xy = calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_xz = calc_vdw_anisotropic(x_ij, z_ij, y_ij,C6_param_ij, C6_VDW_ij)
                VDW_yy = calc_vdw_isotopic(y_ij, x_ij, z_ij, C6_param_ij, C6_VDW_ij)
                VDW_yz = calc_vdw_anisotropic(y_ij, z_ij, x_ij, C6_param_ij, C6_VDW_ij)
                VDW_zz = calc_vdw_isotopic(z_ij, x_ij, y_ij, C6_param_ij, C6_VDW_ij)
        
                g_mm = self.kr * self.calc_force_const(alpha, covalent_length, r_ij_2) + self.kd * self.calc_vdw_force_const(4.0, vdw_length, r_ij)
                
                hess_xx = g_mm * x_ij ** 2 / r_ij_2 - VDW_xx
                hess_xy = g_mm * x_ij * y_ij / r_ij_2 - VDW_xy
                hess_xz = g_mm * x_ij * z_ij / r_ij_2 - VDW_xz
                hess_yy = g_mm * y_ij ** 2 / r_ij_2 - VDW_yy
                hess_yz = g_mm * y_ij * z_ij / r_ij_2 - VDW_yz
                hess_zz = g_mm * z_ij ** 2 / r_ij_2 - VDW_zz
                
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
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_xy
                self.cart_hess[i * 3 + 1][j * 3 + 1] -= hess_yy
                self.cart_hess[i * 3 + 1][j * 3 + 2] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 1] -= hess_yz
                self.cart_hess[i * 3 + 2][j * 3 + 2] -= hess_zz
       
        return 
    
    def lindh2007_angle(self, coord, element_list):
        for i in range(len(coord)):
            i_idx = self.select_idx(element_list[i])
            
            for j in range(len(coord)):
                if i == j:
                    continue
                j_idx = self.select_idx(element_list[j])
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)

                single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                
                #if self.bond_threshold_scale * triple_bond > r_ij:
                #    covalent_length_ij = triple_bond
                #elif self.bond_threshold_scale * double_bond > r_ij:
                #    covalent_length_ij = double_bond
                #else:
                covalent_length_ij = single_bond

                vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                    
                r_ij_0 = self.rAv[i_idx][j_idx]
                d_ij_0 = self.dAv[i_idx][j_idx]
                alpha_ij = self.aAv[i_idx][j_idx]
                
                for k in range(j):
                    if i == k:
                        continue
                    k_idx = self.select_idx(element_list[k])
                    
                    r_ik_0 = self.rAv[i_idx][k_idx]
                    d_ik_0 = self.dAv[i_idx][k_idx]
                    alpha_ik = self.aAv[i_idx][k_idx]
                    
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)


                    single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[k])
                    triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[k])
                    
                    #if self.bond_threshold_scale * triple_bond > r_ik:
                    #    covalent_length_ik = triple_bond
                    #elif self.bond_threshold_scale * double_bond > r_ik:
                    #    covalent_length_ik = double_bond
                    #else:
                    covalent_length_ik = single_bond

                    #covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                    
                    vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                    
                    error_check = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                    error_check = error_check / (r_ij * r_ik)
                    
                    if abs(error_check - 1.0) < self.eps:
                        continue
                    
                    x_jk = coord[j][0] - coord[k][0]
                    y_jk = coord[j][1] - coord[k][1]
                    z_jk = coord[j][2] - coord[k][2]
                    r_jk_2 = x_jk**2 + y_jk**2 + z_jk**2
                    r_jk = np.sqrt(r_jk_2)
                    
                    g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ij, r_ij)
                    g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ik, r_ik)
                    
                    g_jk = self.kf * (g_ij + 0.5 * self.kd / self.kr * d_ij_0) * (g_ik * 0.5 * self.kd / self.kr * d_ik_0)
                    
                    r_cross_2 = (y_ij * z_ik - z_ij * y_ik) ** 2 + (z_ij * x_ik - x_ij * z_ik) ** 2 + (x_ij * y_ik - y_ij * x_ik) ** 2
                    
                    if r_cross_2 < 1.0e-12:
                        r_cross = 0.0
                    else:
                        r_cross = np.sqrt(r_cross_2)
                    
                    if r_ik > self.eps and r_ij > self.eps and r_jk > self.eps:

                        
                        dot_product_r_ij_r_ik = x_ij * x_ik + y_ij * y_ik + z_ij * z_ik
                        cos_theta = dot_product_r_ij_r_ik / (r_ij * r_ik)
                        sin_theta = r_cross / (r_ij * r_ik)
                        if sin_theta > self.eps: # non-linear
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
                            
                            for l in range(3):
                                for m in range(3):
                                    #-------------------------------------
                                    if i > j:
                                        # m = i, i = j, j = k
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
                                    
                            for l in range(3):
                                for m in range(l):
                                    tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                    tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                    tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                    
                                    self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                    self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                    self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                            
                        else: # linear 
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
                            
                            for ii in range(2):
                                # m = i, i = j, j = k
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
                                
                                for l in range(3):
                                    for m in range(3):#Under construction
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
                                
                                for l in range(3):#Under construction
                                    for m in range(l):
                                        tmp_val_1 = g_jk * s_j[l] * s_j[m]
                                        tmp_val_2 = g_jk * s_i[l] * s_i[m]
                                        tmp_val_3 = g_jk * s_k[l] * s_k[m]
                                        
                                        self.cart_hess[j * 3 + l][j * 3 + m] += tmp_val_1
                                        self.cart_hess[i * 3 + l][i * 3 + m] += tmp_val_2
                                        self.cart_hess[k * 3 + l][k * 3 + m] += tmp_val_3
                    else:
                        pass
                    
                
        return

    def lindh2007_dihedral_angle(self, coord, element_list):
        
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
                    
                        i_idx = self.select_idx(element_list[i])
                        j_idx = self.select_idx(element_list[j])
                        k_idx = self.select_idx(element_list[k])
                        l_idx = self.select_idx(element_list[l])
                        
                        r_ij_0 = self.rAv[i_idx][j_idx]
                        d_ij_0 = self.dAv[i_idx][j_idx]
                        alpha_ij = self.aAv[i_idx][j_idx]
                        
                        
                        r_jk_0 = self.rAv[j_idx][k_idx]
                        d_jk_0 = self.dAv[j_idx][k_idx]
                        alpha_jk = self.aAv[j_idx][k_idx]
                        
                        r_kl_0 = self.rAv[k_idx][l_idx]
                        d_kl_0 = self.dAv[k_idx][l_idx]
                        alpha_kl = self.aAv[k_idx][l_idx]
                        
                        
                        
                        
                        
                        t_xyz_4 = coord[l]
                        r_ij = coord[i] - coord[j]
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = double_bond
                        #else:
                        covalent_length_ij = single_bond

                        #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])

                        #single_bond = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[j])
                        #double_bond = double_covalent_radii_lib(element_list[k]) + double_covalent_radii_lib(element_list[j])
                        #triple_bond = triple_covalent_radii_lib(element_list[k]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_jk):
                        #    covalent_length_jk = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_jk):
                        #    covalent_length_jk = double_bond
                        #else:
                        covalent_length_jk = single_bond

                        #covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        #r_jk = coord[j] - coord[k]
                        #vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        #covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_kl = coord[k] - coord[l]

                        #single_bond = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        #double_bond = double_covalent_radii_lib(element_list[k]) + double_covalent_radii_lib(element_list[l])
                        #triple_bond = triple_covalent_radii_lib(element_list[k]) + triple_covalent_radii_lib(element_list[l])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_kl):
                        #    covalent_length_kl = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_kl):
                        #    covalent_length_kl = double_bond
                        #else:
                        covalent_length_kl = single_bond

                        vdw_length_kl = UFF_VDW_distance_lib(element_list[k]) + UFF_VDW_distance_lib(element_list[l])
                        #covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_jk_2 = np.sum(r_jk ** 2)
                        r_kl_2 = np.sum(r_kl ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_jk = np.sqrt(r_jk_2)
                        norm_r_kl = np.sqrt(r_kl_2)
                       
                        a35 = (35.0/180)* np.pi
                        cosfi_max=np.cos(a35)
                        cosfi2 = np.dot(r_ij, r_jk) / np.sqrt(r_ij_2 * r_jk_2)
                        if abs(cosfi2) > cosfi_max:
                            continue
                        cosfi3 = np.dot(r_kl, r_jk) / np.sqrt(r_kl_2 * r_jk_2)
                        if abs(cosfi3) > cosfi_max:
                            continue
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_ij, norm_r_ij)
                        g_jk = self.calc_force_const(alpha_jk, covalent_length_jk, r_jk_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_jk, norm_r_jk)
                        g_kl = self.calc_force_const(alpha_kl, covalent_length_kl, r_kl_2) + 0.5 * self.kd * self.calc_vdw_force_const(4.0, vdw_length_kl, norm_r_kl) 
                       
                        t_ij = self.kt * (g_ij * 0.5 * self.kd / self.kr * d_ij_0) * (g_jk * 0.5 * self.kd / self.kr * d_jk_0) * (g_kl * 0.5 * self.kd / self.kr * d_kl_0)
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        tau, c = torsion2(t_xyz)
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                       
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[3 * i + n][3 * j + m] = self.cart_hess[3 * i + n][3 * j + m] + t_ij * s_i[n] * s_j[m]
                                self.cart_hess[3 * i + n][3 * k + m] = self.cart_hess[3 * i + n][3 * k + m] + t_ij * s_i[n] * s_k[m]
                                self.cart_hess[3 * i + n][3 * l + m] = self.cart_hess[3 * i + n][3 * l + m] + t_ij * s_i[n] * s_l[m]
                                self.cart_hess[3 * j + n][3 * k + m] = self.cart_hess[3 * j + n][3 * k + m] + t_ij * s_j[n] * s_k[m]
                                self.cart_hess[3 * j + n][3 * l + m] = self.cart_hess[3 * j + n][3 * l + m] + t_ij * s_j[n] * s_l[m]
                                self.cart_hess[3 * k + n][3 * l + m] = self.cart_hess[3 * k + n][3 * l + m] + t_ij * s_k[n] * s_l[m]
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[3 * i + n][3 * i + m] = self.cart_hess[3 * i + n][3 * i + m] + t_ij * s_i[n] * s_i[m]
                                self.cart_hess[3 * j + n][3 * j + m] = self.cart_hess[3 * j + n][3 * j + m] + t_ij * s_j[n] * s_j[m]
                                self.cart_hess[3 * k + n][3 * k + m] = self.cart_hess[3 * k + n][3 * k + m] + t_ij * s_k[n] * s_k[m]
                                self.cart_hess[3 * l + n][3 * l + m] = self.cart_hess[3 * l + n][3 * l + m] + t_ij * s_l[n] * s_l[m]
                       
        return
       
    def lindh2007_out_of_plane(self, coord, element_list):
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
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[j])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[j])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ij):
                        #    covalent_length_ij = double_bond
                        #else:
                        covalent_length_ij = single_bond

                        #covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])

                        single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[k])
                        triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[k])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_ik):
                        #    covalent_length_ik = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_ik):
                        #    covalent_length_ik = double_bond
                        #else:
                        covalent_length_ik = single_bond

                        #covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        vdw_length_il = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[l])

                        #single_bond = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        #double_bond = double_covalent_radii_lib(element_list[i]) + double_covalent_radii_lib(element_list[l])
                        #triple_bond = triple_covalent_radii_lib(element_list[i]) + triple_covalent_radii_lib(element_list[l])
                        
                        #if self.bond_threshold_scale * triple_bond > np.linalg.norm(r_il):
                        #    covalent_length_il = triple_bond
                        #elif self.bond_threshold_scale * double_bond > np.linalg.norm(r_il):
                        #    covalent_length_il = double_bond
                        #else:
                        covalent_length_il = single_bond

                        #covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
                        idx_i = self.select_idx(element_list[i])
                        idx_j = self.select_idx(element_list[j])
                        idx_k = self.select_idx(element_list[k])
                        idx_l = self.select_idx(element_list[l])
                        
                        d_ij_0 = self.dAv[idx_i][idx_j]
                        r_ij_0 = self.rAv[idx_i][idx_j]
                        alpha_ij = self.aAv[idx_i][idx_j]
                        
                        d_ik_0 = self.dAv[idx_i][idx_k]
                        r_ik_0 = self.rAv[idx_i][idx_k]
                        alpha_ik = self.aAv[idx_i][idx_k]
                        
                        d_il_0 = self.dAv[idx_i][idx_l]
                        r_il_0 = self.rAv[idx_i][idx_l]
                        alpha_il = self.aAv[idx_i][idx_l]
                        
                        
                        
                        
                        
                        r_ij_2 = np.sum(r_ij ** 2)
                        r_ik_2 = np.sum(r_ik ** 2)
                        r_il_2 = np.sum(r_il ** 2)
                        norm_r_ij = np.sqrt(r_ij_2)
                        norm_r_ik = np.sqrt(r_ik_2)
                        norm_r_il = np.sqrt(r_il_2)
                        
                        cosfi2 = np.dot(r_ij, r_ik) / np.sqrt(r_ij_2 * r_ik_2)
                        if abs(abs(cosfi2) - 1.0) < 1.0e-1:
                            continue
                        cosfi3 = np.dot(r_ij, r_il) / np.sqrt(r_ij_2 * r_il_2)
                        if abs(abs(cosfi3) - 1.0) < 1.0e-1:
                            continue
                        cosfi4 = np.dot(r_ik, r_il) / np.sqrt(r_ik_2 * r_il_2)
                        if abs(abs(cosfi4) - 1.0) < 1.0e-1:
                            continue
                        kd = 0.0
                        g_ij = self.calc_force_const(alpha_ij, covalent_length_ij, r_ij_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_ij, norm_r_ij)
                        
                        g_ik = self.calc_force_const(alpha_ik, covalent_length_ik, r_ik_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_ik, norm_r_ik)
                        
                        g_il = self.calc_force_const(alpha_il, covalent_length_il, r_il_2) + 0.5 * kd * self.calc_vdw_force_const(4.0, vdw_length_il, norm_r_il)
                        
                        
                        t_ij = self.ko * g_ij * g_ik * g_il #self.ko * g_ij * g_ik * g_il
                        
                        t_xyz = np.array([t_xyz_1, t_xyz_2, t_xyz_3, t_xyz_4])
                        theta, c = outofplane2(t_xyz)
                        
                        s_i = c[0]
                        s_j = c[1]
                        s_k = c[2]
                        s_l = c[3]
                        
                        for n in range(3):
                            for m in range(3):
                                self.cart_hess[i * 3 + n][j * 3 + m] += t_ij * s_i[n] * s_j[m]
                                self.cart_hess[i * 3 + n][k * 3 + m] += t_ij * s_i[n] * s_k[m]
                                self.cart_hess[i * 3 + n][l * 3 + m] += t_ij * s_i[n] * s_l[m]
                                self.cart_hess[j * 3 + n][k * 3 + m] += t_ij * s_j[n] * s_k[m]
                                self.cart_hess[j * 3 + n][l * 3 + m] += t_ij * s_j[n] * s_l[m]
                                self.cart_hess[k * 3 + n][l * 3 + m] += t_ij * s_k[n] * s_l[m]    
                            
                            
                        for n in range(3):
                            for m in range(n):
                                self.cart_hess[i * 3 + n][i * 3 + m] += t_ij * s_i[n] * s_i[m]
                                self.cart_hess[j * 3 + n][j * 3 + m] += t_ij * s_j[n] * s_j[m]
                                self.cart_hess[k * 3 + n][k * 3 + m] += t_ij * s_k[n] * s_k[m]
                                self.cart_hess[l * 3 + n][l * 3 + m] += t_ij * s_l[n] * s_l[m]
                          
        return

    def main(self, coord, element_list, cart_gradient):
        norm_grad = np.linalg.norm(cart_gradient)
        scale = 0.1
        eigval_scale = scale * np.exp(-1 * norm_grad ** 2.0)
        #coord: Bohr
        print("generating Lindh's (2007) approximate hessian...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        self.lindh2007_bond(coord, element_list)
        self.lindh2007_angle(coord, element_list)
        self.lindh2007_dihedral_angle(coord, element_list)
        self.lindh2007_out_of_plane(coord, element_list)
        
        for i in range(len(coord)*3):
            for j in range(len(coord)*3):
                
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        hess_proj = np.dot(np.dot(eigenvectors, np.diag(np.abs(eigenvalues) * eigval_scale)), np.linalg.inv(eigenvectors))
        
        return hess_proj#cart_hess

