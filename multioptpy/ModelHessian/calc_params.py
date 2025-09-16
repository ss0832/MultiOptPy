import numpy as np
from multioptpy.Parameters.parameter import D2_S6_parameter


def calc_vdw_isotopic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij):
   
    x_ij = x_ij + 1.0e-12
    y_ij = y_ij + 1.0e-12
    z_ij = z_ij + 1.0e-12
    a_vdw = 20.0
    t_1 = D2_S6_parameter * C6_param_ij
    t_2 = x_ij ** 2 
    t_3 = y_ij ** 2 
    t_4 = z_ij ** 2
    t_5 = t_2 + t_3 + t_4
    t_6 = t_5 ** 2
    t_7 = t_6 ** 2 
    t_10 = t_5 ** 0.5
    t_11 = 0.1 / C6_VDW_ij
   
    t_15 = np.exp(-1 * a_vdw * (t_10 * t_11 - 0.1)) + 1.0e-12
    t_16 = 0.1 + t_15
    t_17 = 0.1 / t_16
    t_24 = t_16 ** 2
    t_25 = 0.1 / t_24
    t_29 = t_11 * t_15
    t_33 = 0.1 / t_7
    t_41 = a_vdw ** 2
    t_42 = C6_VDW_ij ** 2
    t_44 = t_41 / t_42
    t_45 = t_15 ** 2
    #print(t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_10, t_11, t_15, t_16, t_17, t_24, t_25, t_29, t_33, t_41, t_42, t_44, t_45)
    t_62 = -0.48 * t_1 / t_7 / t_5 * t_17 * t_2 + 0.13 * t_1 / t_10 / t_7 *  t_25 * t_2 * a_vdw * t_29 + 0.6 * t_1 * t_33 * t_17 - 0.2 * t_1 * t_33 / t_24 / t_16 * t_44 * t_2 * t_45 - t_1 / t_10 / t_6 / t_5 * t_25 * a_vdw * t_29 + t_1 * t_33 * t_25 * t_44 * t_2 * t_15
    
    return t_62

def calc_vdw_anisotropic(x_ij, y_ij, z_ij, C6_param_ij, C6_VDW_ij):
    a_vdw = 20.0
    x_ij = x_ij + 1.0e-12
    y_ij = y_ij + 1.0e-12
    z_ij = z_ij + 1.0e-12
    
    t_1 = D2_S6_parameter * C6_param_ij
    t_2 = x_ij ** 2
    t_3 = y_ij ** 2 
    t_4 = z_ij ** 2 
    t_5 = t_2 + t_3 + t_4
    t_6 = t_5 ** 2
    t_7 = t_6 ** 2
    t_11 = t_5 ** 0.5
    t_12 = 0.1 / C6_VDW_ij
   
    t_16 = np.exp(-1 * a_vdw * (t_11 * t_12 - 0.1)) + 1.0e-12
    t_17 = 0.1 + t_16 
    t_25 = t_17 ** 2 
    t_26 = 0.1 / t_25
    t_35 = 0.1 / t_7
    t_40 = a_vdw ** 2
    t_41 = C6_VDW_ij ** 2
    t_43 = t_40 / t_41
    t_44 = t_16 ** 2
    #print(t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_11, t_12, t_16, t_17, t_25, t_26, t_35, t_40, t_41, t_43, t_44)
    t_56 = -0.48 * t_1 / t_7 / t_5 / t_17 * x_ij * y_ij + 0.13 * t_1 / t_11 / t_7 * t_26 * x_ij * a_vdw * t_12 * y_ij * t_16 - 0.2 * t_1 * t_35 / t_25 / t_17 * t_43 * x_ij * t_44 * y_ij + t_1 * t_35 * t_26 * t_43 * x_ij * y_ij * t_16
    #print(t_16, t_56)
    return t_56
    
    
def outofplane2(t_xyz):
    r_1 = t_xyz[0] - t_xyz[3]
    q_41 = np.linalg.norm(r_1)
    e_41 = r_1 / q_41

    r_2 = t_xyz[1] - t_xyz[3]
    q_42 = np.linalg.norm(r_2)
    e_42 = r_2 / q_42

    r_3 = t_xyz[2] - t_xyz[3]
    q_43 = np.linalg.norm(r_3)
    e_43 = r_3 / q_43

    cosfi1 = np.dot(e_43, e_42)

    fi_1 = np.arccos(cosfi1)

    if abs(fi_1 - np.pi) < 1e-12:
        teta = 0.0
        bt = 0.0
        return teta, bt

    cosfi2 = np.dot(e_41, e_43)
    fi_2 = np.arccos(cosfi2)
    #deg_fi_2 = 180.0 * fi_2 / np.pi

    cosfi3 = np.dot(e_41, e_42)

    fi_3 = np.arccos(cosfi3)
    #deg_fi_3 = 180.0 * fi_3 / np.pi

    c14 = np.zeros((3, 3))

    c14[0] = t_xyz[0]
    c14[1] = t_xyz[3]

    r_42 = t_xyz[1] - t_xyz[3]
    r_43 = t_xyz[2] - t_xyz[3]
    c14[2][0] = r_42[1] * r_43[2] - r_42[2] * r_43[1]
    c14[2][1] = r_42[2] * r_43[0] - r_42[0] * r_43[2]
    c14[2][2] = r_42[0] * r_43[1] - r_42[1] * r_43[0]

    if ((c14[2][0] ** 2 + c14[2][1] ** 2 + c14[2][2] ** 2) < 1e-12):
        teta = 0.0
        bt = 0.0
        return teta, bt
    c14[2][0] = c14[2][0] + t_xyz[3][0]
    c14[2][1] = c14[2][1] + t_xyz[3][1]
    c14[2][2] = c14[2][2] + t_xyz[3][2]

    teta, br_14 = bend2(c14)

    teta = teta -0.5 * np.pi

    bt = np.zeros((4, 3))

    for i_x in range(1, 4):
        i_y = (i_x + 1) % 4 + (i_x + 1) // 4
        i_z = (i_y + 1) % 4 + (i_y + 1) // 4
        
        bt[0][i_x-1] = -1 * br_14[2][i_x-1]
        bt[1][i_x-1] = -1 * br_14[2][i_y-1]
        bt[2][i_x-1] = -1 * br_14[2][i_z-1]
        bt[3][i_x-1] = -1 * (bt[0][i_x-1] + bt[1][i_x-1] + bt[2][i_x-1])
        
    bt *= -1.0 

    return teta, bt# angle, move_vector

def torsion2(t_xyz):
    r_ij_1, b_r_ij = stretch2(t_xyz[0:2])
    r_ij_2, b_r_jk = stretch2(t_xyz[1:3])
    r_ij_3, b_r_kl = stretch2(t_xyz[2:4])

    fi_2, b_fi_2 = bend2(t_xyz[0:3])
    fi_3, b_fi_3 = bend2(t_xyz[1:4])
    sin_fi_2 = np.sin(fi_2)
    sin_fi_3 = np.sin(fi_3)
    cos_fi_2 = np.cos(fi_2)
    cos_fi_3 = np.cos(fi_3)
    costau = ( (b_r_ij[0][1] * b_r_jk[1][2] - b_r_jk[1][1] * b_r_ij[0][2])) * (b_r_jk[0][1] * b_r_kl[1][2] - b_r_jk[0][2] * b_r_kl[1][1]) + (b_r_ij[0][2] * b_r_jk[1][0] - b_r_ij[0][0] * b_r_jk[1][2]) * (b_r_jk[0][2] * b_r_kl[1][0] - b_r_jk[0][0] * b_r_kl[1][2]) + (b_r_ij[0][0] * b_r_jk[1][1] - b_r_ij[0][1] * b_r_jk[1][0]) * (b_r_jk[0][0] * b_r_kl[1][1] - b_r_jk[0][1] * b_r_kl[1][0]) / (sin_fi_2 * sin_fi_3)
    sintau = (b_r_ij[1][0] * (b_r_jk[0][1] * b_r_kl[1][2] - b_r_jk[0][2] * b_r_kl[1][1]) + b_r_ij[1][1] * (b_r_jk[0][2] * b_r_kl[1][0] - b_r_jk[0][0] * b_r_kl[1][2]) + b_r_ij[1][2] * (b_r_jk[0][0] * b_r_kl[1][1] - b_r_jk[0][1] * b_r_kl[1][0])) / (sin_fi_2 * sin_fi_3)

    tau = np.arctan2(sintau, costau)

    if abs(tau) == np.pi:
        tau = np.pi

    bt = np.zeros((4, 3))
    for i_x in range(1,4):
        i_y = i_x + 1
        if i_y > 3:
            i_y = i_y - 3
        i_z = i_y + 1
        if i_z > 3:
            i_z = i_z - 3
        bt[0][i_x-1] = (b_r_ij[1][i_y-1] * b_r_jk[1][i_z-1] - b_r_ij[1][i_z-1] * b_r_jk[1][i_y-1]) / (r_ij_1 * sin_fi_2 ** 2)
        bt[3][i_x-1] = (b_r_kl[0][i_y-1] * b_r_jk[0][i_z-1] - b_r_kl[0][i_z-1] * b_r_jk[0][i_y-1]) / (r_ij_3 * sin_fi_3 ** 2)
        bt[1][i_x-1] = -1 * ((r_ij_2 - r_ij_1 * cos_fi_2) * bt[0][i_x-1] + r_ij_3 * cos_fi_3 * bt[3][i_x-1]) / r_ij_2
        bt[2][i_x-1] = -1 * (bt[0][i_x-1] + bt[1][i_x-1] + bt[3][i_x-1])

    for ix in range(1, 4):
        iy = ix + 1
        if iy > 3:
            iy = iy - 3
        iz = iy + 1
        if iz > 3:
            iz = iz - 3
        bt[0][ix-1] = (b_r_ij[1][iy-1] * b_r_jk[1][iz-1] - b_r_ij[1][iz-1] * b_r_jk[1][iy-1]) / (r_ij_1 * sin_fi_2 ** 2)
        bt[3][ix-1] = (b_r_kl[0][iy-1] * b_r_jk[0][iz-1] - b_r_kl[0][iz-1] * b_r_jk[0][iy-1]) / (r_ij_3 * sin_fi_3 ** 2)
        bt[1][ix-1] = -1 * ((r_ij_2 - r_ij_1 * cos_fi_2) * bt[0][ix-1] + r_ij_3 * cos_fi_3 * bt[3][ix-1]) / r_ij_2
        bt[2][ix-1] = -1 * (bt[0][ix-1] + bt[1][ix-1] + bt[3][ix-1])

    return tau, bt # angle, move_vector

def bend2(t_xyz):
    r_ij_1, b_r_ij = stretch2(t_xyz[0:2])
    r_jk_1, b_r_jk = stretch2(t_xyz[1:3])
    c_o = 0.0
    crap = 0.0
    for i in range(3):
        c_o += b_r_ij[0][i] * b_r_jk[1][i]
        crap += b_r_ij[0][i] ** 2
        crap += b_r_jk[1][i] ** 2

    if np.sqrt(crap) < 1e-12:
        fir = np.pi - np.arcsin(np.sqrt(crap))
        si = np.sqrt(crap)
    else:
        fir = np.arccos(c_o)
        si = np.sqrt(1 - c_o ** 2)

    if np.abs(fir - np.pi) < 1e-12:
        fir = np.pi

    bf = np.zeros((3, 3))
    denom_1 = r_ij_1 * si
    denom_2 = r_jk_1 * si
    
    for i in range(3):
        if denom_1 < 1e-12:
            bf[0][i] = 0.0
        else:
            bf[0][i] = (c_o * b_r_ij[0][i] - b_r_jk[1][i]) / denom_1
        if denom_2 < 1e-12:
            bf[2][i] = 0.0
        else:
            bf[2][i] = (c_o * b_r_jk[1][i] - b_r_ij[0][i]) / denom_2
        bf[1][i] = -1 * (bf[0][i] + bf[2][i])

    return fir, bf # angle, move_vector

def stretch2(t_xyz):
    dist = t_xyz[0] - t_xyz[1]
    norm_dist = np.linalg.norm(dist)
    b = np.zeros((2,3))
    b[0] = -1 * dist / norm_dist
    b[1] = dist / norm_dist
    return norm_dist, b # distance, move_vectors (unit_vector)

