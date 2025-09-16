import numpy as np

from multioptpy.Parameters.parameter import UnitValueLib, covalent_radii_lib, D2_C6_coeff_lib, D2_VDW_radii_lib, UFF_VDW_distance_lib
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.ModelHessian.calc_params import torsion2, outofplane2, calc_vdw_isotopic, calc_vdw_anisotropic
from multioptpy.Parameters.parameter import UFF_VDW_distance_lib


class SwartD2ApproxHessian:
    def __init__(self):
        #Swart's Model Hessian augmented with D2
        #ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536–2544.
        #ref.: https://github.com/grimme-lab/xtb/blob/main/src/model_hessian.f90
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.kd = 2.00
        
        self.kr = 0.35
        self.kf = 0.15
        self.kt = 0.005
        
        self.cutoff = 70.0
        self.eps = 1.0e-12
        
        #self.s6 = 20.0
        return
    
    def calc_force_const(self, alpha, covalent_length, distance):
        force_const = np.exp(-1 * alpha * (distance / covalent_length - 1.0))
        
        return force_const
        
    def calc_vdw_force_const(self, alpha, vdw_length, distance):
        vdw_force_const = np.exp(-1 * alpha * (vdw_length - distance) ** 2)
        return vdw_force_const
    
    def swart_bond(self, coord, element_list):
        for i in range(len(coord)):
            for j in range(i):
                
                x_ij = coord[i][0] - coord[j][0]
                y_ij = coord[i][1] - coord[j][1]
                z_ij = coord[i][2] - coord[j][2]
                r_ij_2 = x_ij**2 + y_ij**2 + z_ij**2
                r_ij = np.sqrt(r_ij_2)
                covalent_length = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
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
        
                g_mm = self.kr * self.calc_force_const(1.0, covalent_length, r_ij) + self.kd * self.calc_vdw_force_const(5.0, vdw_length, r_ij)
                
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
    
    def swart_angle(self, coord, element_list):
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
                vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                
                
                for k in range(j):
                    if i == k:
                        continue
                    x_ik = coord[i][0] - coord[k][0]
                    y_ik = coord[i][1] - coord[k][1]
                    z_ik = coord[i][2] - coord[k][2]
                    r_ik_2 = x_ik**2 + y_ik**2 + z_ik**2
                    r_ik = np.sqrt(r_ik_2)
                    covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
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
                    
                    g_ij = self.calc_force_const(1.0, covalent_length_ij, r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, r_ij)
                    g_ik = self.calc_force_const(1.0, covalent_length_ik, r_ik) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ik, r_ik)
                    
                    g_jk = self.kf * g_ij * g_ik
                    
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

    def swart_dihedral_angle(self, coord, element_list):
        
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
                        vdw_length_ij = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[j])
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_jk = coord[j] - coord[k]
                        vdw_length_jk = UFF_VDW_distance_lib(element_list[j]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_jk = covalent_radii_lib(element_list[j]) + covalent_radii_lib(element_list[k])
                        
                        r_kl = coord[k] - coord[l]
                        vdw_length_kl = UFF_VDW_distance_lib(element_list[k]) + UFF_VDW_distance_lib(element_list[l])
                        covalent_length_kl = covalent_radii_lib(element_list[k]) + covalent_radii_lib(element_list[l])
                        
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
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, norm_r_ij)
                        g_jk = self.calc_force_const(1.0, covalent_length_jk, norm_r_jk) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_jk, norm_r_jk)
                        g_kl = self.calc_force_const(1.0, covalent_length_kl, norm_r_kl) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_kl, norm_r_kl) 
                       
                        t_ij = self.kt * g_ij * g_jk * g_kl
                        
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
       
    def swart_out_of_plane(self, coord, element_list):
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
                        covalent_length_ij = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                        
                        r_ik = coord[i] - coord[k]
                        vdw_length_ik = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[k])
                        covalent_length_ik = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[k])
                        
                        r_il = coord[i] - coord[l]
                        vdw_length_il = UFF_VDW_distance_lib(element_list[i]) + UFF_VDW_distance_lib(element_list[l])
                        covalent_length_il = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[l])
                        
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
                        
                        g_ij = self.calc_force_const(1.0, covalent_length_ij, norm_r_ij) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ij, norm_r_ij)
                        
                        g_ik = self.calc_force_const(1.0, covalent_length_ik, norm_r_ik) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_ik, norm_r_ik)
                        
                        g_il = self.calc_force_const(1.0, covalent_length_il, norm_r_il) + 0.5 * self.kd * self.calc_vdw_force_const(5.0, vdw_length_il, norm_r_il)
                        
                        
                        t_ij = self.kt * g_ij * g_ik * g_il #self.ko * g_ij * g_ik * g_il
                        
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
        #coord: Bohr
        print("generating Swart's approximate hessian...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        
        self.swart_bond(coord, element_list)
        self.swart_angle(coord, element_list)
        self.swart_dihedral_angle(coord, element_list)
        self.swart_out_of_plane(coord, element_list)
        
        for i in range(len(coord)*3):
            for j in range(len(coord)*3):
                
                if abs(self.cart_hess[i][j]) < 1.0e-10:
                    self.cart_hess[i][j] = self.cart_hess[j][i]
        
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj#cart_hess
