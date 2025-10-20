#!/usr/bin/env python3
"""
SQM1 Implementation - Rigorous and Theory-Compliant

Complete implementation of SQM1 following the theoretical framework 
based on ChemRxiv: 60c742abbdbb890c7ba3851a.

This corrected implementation addresses:
1. Parameter defects - reads from external param_sqm1.txt file
2. Physical model simplifications - proper EEQ, SimpleDispersion, SRB implementations
3. Functional/algorithmic defects - proper unit handling and calculations

Total Energy Expression:
    E_total = E_EHT + E_IES + E_rep + E_dispSimple + E_SRB

Supported Elements: H, C, N, O, F, Cl, Br (Z=1, 6, 7, 8, 9, 17, 35)
"""

import torch
import torch.nn.functional as F

# --- Constants ---
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM  
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# --- SimpleDispersion Coordination Number Parameters ---
# Reference: Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019)
# and D3 method: Grimme et al., J. Chem. Phys. 132, 154104 (2010)
SIMPLE_DISP_CN_K1 = 16.0  # Steepness parameter for CN counting function
SIMPLE_DISP_CN_K2 = 4.0 / 3.0  # Scaling factor for covalent radii sum
SIMPLE_DISP_CN_CUTOFF = 20.0  # Cutoff radius in Angstrom for CN calculation

# Covalent radii for SimpleDispersion (in Angstrom)
# Reference: Pyykkö and Atsumi, Chem. Eur. J. 15, 186 (2009)
# These values are consistent with xTB parameterization
SIMPLE_DISP_COVALENT_RADII = {
    1: 0.32,   # H
    2: 0.46,   # He
    3: 1.33,   # Li
    4: 1.02,   # Be
    5: 0.85,   # B
    6: 0.75,   # C
    7: 0.71,   # N
    8: 0.63,   # O
    9: 0.64,   # F
    10: 0.67,  # Ne
    11: 1.55,  # Na
    12: 1.39,  # Mg
    13: 1.26,  # Al
    14: 1.16,  # Si
    15: 1.11,  # P
    16: 1.03,  # S
    17: 0.99,  # Cl
    18: 0.96,  # Ar
    19: 1.96,  # K
    20: 1.71,  # Ca
    21: 1.48,  # Sc
    22: 1.36,  # Ti
    23: 1.34,  # V
    24: 1.22,  # Cr
    25: 1.19,  # Mn
    26: 1.16,  # Fe
    27: 1.11,  # Co
    28: 1.10,  # Ni
    29: 1.12,  # Cu
    30: 1.18,  # Zn
    31: 1.24,  # Ga
    32: 1.21,  # Ge
    33: 1.21,  # As
    34: 1.16,  # Se
    35: 1.14,  # Br
    36: 1.17,  # Kr
    37: 2.10,  # Rb
    38: 1.85,  # Sr
    39: 1.63,  # Y
    40: 1.54,  # Zr
    41: 1.47,  # Nb
    42: 1.38,  # Mo
    43: 1.28,  # Tc
    44: 1.25,  # Ru
    45: 1.25,  # Rh
    46: 1.20,  # Pd
    47: 1.28,  # Ag
    48: 1.36,  # Cd
    49: 1.42,  # In
    50: 1.40,  # Sn
    51: 1.40,  # Sb
    52: 1.36,  # Te
    53: 1.33,  # I
    54: 1.31,  # Xe
    55: 2.32,  # Cs
    56: 1.96,  # Ba
    57: 1.80,  # La
    58: 1.63,  # Ce
    59: 1.76,  # Pr
    60: 1.74,  # Nd
    61: 1.73,  # Pm
    62: 1.72,  # Sm
    63: 1.68,  # Eu
    64: 1.69,  # Gd
    65: 1.68,  # Tb
    66: 1.67,  # Dy
    67: 1.66,  # Ho
    68: 1.65,  # Er
    69: 1.64,  # Tm
    70: 1.70,  # Yb
    71: 1.62,  # Lu
    72: 1.52,  # Hf
    73: 1.46,  # Ta
    74: 1.37,  # W
    75: 1.31,  # Re
    76: 1.29,  # Os
    77: 1.22,  # Ir
    78: 1.23,  # Pt
    79: 1.24,  # Au
    80: 1.33,  # Hg
    81: 1.44,  # Tl
    82: 1.44,  # Pb
    83: 1.51,  # Bi
    84: 1.45,  # Po
    85: 1.47,  # At
    86: 1.42,  # Rn
}


class SQM1Parameters:
    """
    Stores all SQM1 parameters embedded directly in the class.
    Parameters from param_gfn0_xtb.txt are now included in the code,
    eliminating the need for an external parameter file.
    """
    def __init__(self):
        # Initialize parameter dictionaries with embedded data
        self.element_params = {}
        self.sk_params = {}
        self.rep_params = {}
        self.simple_disp_params = {}
        self.srb_params = {}
        self.global_params = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize all parameters from embedded data."""
        # Embedded parameter data from param_gfn0_xtb.txt
        # Format: (Z, symbol, valence_e, h_s, h_p, Z_eff, EN_A, J_AA_param, alpha, C6_ref, alpha_ref)
        element_data = [
            (1, 'H', 1, -11.92, -2.81, 1.25, 1.92, -0.3023, 0.749, 0.81, 2.7),
            (2, 'He', 2, -20.95, -1.13, 1.2912, 2.0, 0.7743, 0.4197, 0.2, 1.4),
            (3, 'Li', 1, -7.0, -3.27, 0.854, 2.0, 0.5303, 1.4256, 164.0, 164.0),
            (4, 'Be', 2, -9.81, -4.17, 1.1724, 2.0, 0.2176, 2.0699, 75.0, 38.0),
            (5, 'B', 3, -11.53, -7.18, 1.1094, 2.0, 0.1956, 1.7359, 25.0, 13.0),
            (6, 'C', 4, -15.75, -9.8, 1.386, 2.48, 0.0308, 1.71, 28.0, 10.0),
            (7, 'N', 5, -18.84, -11.54, 1.5342, 2.97, 0.056, 1.8256, 19.0, 7.5),
            (8, 'O', 6, -17.93, -11.84, 1.5379, 2.0, 0.0581, 1.5927, 13.0, 5.8),
            (9, 'F', 7, -21.18, -12.1, 1.5891, 3.5, 0.1574, 0.8986, 9.5, 4.2),
            (10, 'Ne', 8, -23.81, -12.73, 1.2894, 3.5, 0.6826, 0.6138, 6.5, 2.7),
            (11, 'Na', 1, -8.02, -3.54, 0.7891, 2.0, 0.3922, 1.7294, 163.0, 163.0),
            (12, 'Mg', 2, -8.9, -3.39, 0.9983, 2.0, 0.5582, 1.7925, 147.0, 71.0),
            (13, 'Al', 3, -11.42, -5.5, 0.9621, 2.0, 0.3018, 1.2157, 65.0, 58.0),
            (14, 'Si', 4, -14.13, -8.28, 1.0441, 2.0, 0.1039, 1.5314, 80.0, 37.0),
            (15, 'P', 5, -15.71, -9.87, 1.479, 2.0, 0.2125, 1.3731, 57.0, 25.0),
            (16, 'S', 6, -20.16, -11.19, 1.3926, 2.0, 0.0581, 1.7936, 60.0, 20.0),
            (17, 'Cl', 7, -26.27, -12.37, 1.4749, 2.0, 0.2537, 2.6682, 60.0, 20.0),
            (18, 'Ar', 8, -22.03, -14.31, 1.225, 2.0, 0.578, 1.5892, 50.0, 16.0),
            (19, 'K', 1, -6.69, -3.11, 0.8162, 1.45, 0.3921, 2.183, 309.0, 290.0),
            (20, 'Ca', 2, -8.05, -2.18, 1.1252, 1.8, -0.0025, 1.4178, 210.0, 160.0),
            (21, 'Sc', 3, -8.71, -9.02, 0.9641, 1.73, -0.0062, 1.5181, 155.0, 80.0),
            (22, 'Ti', 4, -8.57, -9.49, 0.881, 2.0, 0.1663, 1.992, 125.0, 70.0),
            (23, 'V', 5, -8.76, -9.87, 0.9742, 2.0, 0.1052, 1.7172, 115.0, 65.0),
            (24, 'Cr', 6, -8.82, -7.1, 1.1029, 2.0, 0.001, 2.0655, 105.0, 60.0),
            (25, 'Mn', 7, -9.58, -6.08, 1.0077, 2.0, 0.0977, 1.3318, 100.0, 55.0),
            (26, 'Fe', 8, -10.15, -5.54, 0.7744, 2.0, 0.0612, 1.366, 95.0, 50.0),
            (27, 'Co', 9, -10.53, -4.96, 0.7554, 2.0, 0.0562, 1.5694, 90.0, 45.0),
            (28, 'Ni', 10, -10.59, -6.64, 1.0183, 2.0, 0.09, 1.2763, 85.0, 40.0),
            (29, 'Cu', 11, -11.36, -8.46, 1.0316, 2.0, 0.1313, 1.004, 120.0, 50.0),
            (30, 'Zn', 12, -11.05, -2.78, 1.6317, 2.0, 0.5728, 0.7339, 120.0, 50.0),
            (31, 'Ga', 3, -11.23, -4.64, 1.1187, 2.0, 0.1742, 3.2596, 90.0, 45.0),
            (32, 'Ge', 4, -15.56, -9.18, 1.0346, 2.0, 0.2672, 1.753, 85.0, 40.0),
            (33, 'As', 5, -16.8, -10.2, 1.3091, 2.0, 0.2352, 1.5282, 65.0, 35.0),
            (34, 'Se', 6, -20.69, -11.35, 1.4119, 2.0, 0.0718, 2.1838, 75.0, 30.0),
            (35, 'Br', 7, -19.9, -11.63, 1.4501, 2.0, 0.3458, 2.3806, 90.0, 28.0),
            (36, 'Kr', 8, -17.74, -13.32, 1.1747, 2.0, 0.8203, 2.7281, 60.0, 25.0),
            (37, 'Rb', 1, -6.66, -3.3, 0.6686, 1.5, 0.4288, 0.7838, 20.0, 10.0),
            (38, 'Sr', 2, -6.36, -1.69, 1.0745, 1.5, 0.2667, 1.4275, 20.0, 10.0),
            (39, 'Y', 3, -7.33, -10.52, 0.9108, 1.55, 0.0874, 1.8024, 20.0, 10.0),
            (40, 'Zr', 4, -8.35, -9.42, 0.7876, 2.0, 0.0599, 1.6093, 20.0, 10.0),
            (41, 'Nb', 5, -8.99, -9.38, 1.004, 2.0, 0.1582, 1.3834, 20.0, 10.0),
            (42, 'Mo', 6, -8.34, -5.04, 0.9225, 2.0, 0.1716, 1.1741, 20.0, 10.0),
            (43, 'Tc', 7, -9.59, -4.12, 0.9036, 2.0, 0.2722, 1.5768, 20.0, 10.0),
            (44, 'Ru', 8, -10.42, -4.67, 1.0332, 2.0, 0.2818, 1.3205, 20.0, 10.0),
            (45, 'Rh', 9, -11.24, -6.32, 1.0294, 2.0, 0.1392, 1.4259, 20.0, 10.0),
            (46, 'Pd', 10, -11.05, -7.52, 1.055, 2.0, 0.1176, 1.15, 20.0, 10.0),
            (47, 'Ag', 11, -12.21, -7.81, 1.1301, 2.0, 0.0668, 1.1423, 20.0, 10.0),
            (48, 'Cd', 12, -11.27, -2.61, 1.3935, 2.0, 0.5725, 0.6877, 20.0, 10.0),
            (49, 'In', 3, -11.8, -4.99, 1.2124, 2.0, 0.2002, 2.6507, 20.0, 10.0),
            (50, 'Sn', 4, -16.13, -9.32, 1.1587, 2.0, 0.1603, 1.9834, 20.0, 10.0),
            (51, 'Sb', 5, -17.16, -9.32, 1.2824, 2.0, 0.1716, 1.7405, 20.0, 10.0),
            (52, 'Te', 6, -18.5, -10.49, 1.3608, 2.0, 0.1016, 2.1537, 20.0, 10.0),
            (53, 'I', 7, -13.69, -10.14, 1.4131, 2.0, 0.3082, 2.0992, 20.0, 10.0),
            (54, 'Xe', 8, -11.7, -11.59, 1.188, 2.0, 0.7857, 2.6331, 20.0, 10.0),
            (55, 'Cs', 1, -6.18, -3.52, 0.5875, 1.4, 0.5055, 0.4975, 20.0, 10.0),
            (56, 'Ba', 2, -5.85, -1.72, 0.9976, 1.55, 0.2916, 1.184, 20.0, 10.0),
            (57, 'La', 3, -7.25, -10.38, 0.8626, 1.6, 0.0814, 1.6773, 20.0, 10.0),
            (58, 'Ce', 4, -7.4, -10.36, 0.8596, 1.65, 0.0742, 1.6882, 20.0, 10.0),
            (59, 'Pr', 5, -7.54, -10.34, 0.8566, 1.7, 0.0669, 1.6991, 20.0, 10.0),
            (60, 'Nd', 6, -7.69, -10.32, 0.8537, 1.75, 0.0597, 1.71, 20.0, 10.0),
            (61, 'Pm', 7, -7.84, -10.29, 0.8507, 1.8, 0.0524, 1.7209, 20.0, 10.0),
            (62, 'Sm', 8, -7.99, -10.27, 0.8478, 1.85, 0.0452, 1.7318, 20.0, 10.0),
            (63, 'Eu', 9, -8.13, -10.25, 0.8448, 1.9, 0.0379, 1.7427, 20.0, 10.0),
            (64, 'Gd', 10, -8.28, -10.23, 0.8419, 1.95, 0.0307, 1.7536, 20.0, 10.0),
            (65, 'Tb', 11, -8.43, -10.21, 0.8389, 2.0, 0.0234, 1.7645, 20.0, 10.0),
            (66, 'Dy', 12, -8.5, -10.12, 0.8359, 2.0, 0.0243, 1.8433, 20.0, 10.0),
            (67, 'Ho', 13, -8.55, -10.06, 0.8329, 2.0, 0.0276, 1.9221, 20.0, 10.0),
            (68, 'Er', 14, -8.6, -10.0, 1.0158, 1.5, 0.0301, 2.677, 20.0, 10.0),
            (69, 'Tm', 15, -8.63, -9.87, 1.0117, 1.5, 0.0307, 2.7533, 20.0, 10.0),
            (70, 'Yb', 16, -8.66, -9.74, 1.0075, 1.5, 0.0313, 2.8297, 20.0, 10.0),
            (71, 'Lu', 3, -8.7, -9.61, 1.0034, 1.5, 0.032, 2.906, 20.0, 10.0),
            (72, 'Hf', 4, -8.33, -9.27, 0.8613, 2.0, 0.0263, 1.6423, 20.0, 10.0),
            (73, 'Ta', 5, -9.15, -10.52, 1.0422, 2.0, 0.1715, 1.3568, 20.0, 10.0),
            (74, 'W', 6, -9.64, -8.4, 0.7633, 2.0, 0.1804, 1.8967, 20.0, 10.0),
            (75, 'Re', 7, -10.24, -4.94, 0.602, 2.0, 0.3632, 0.8253, 20.0, 10.0),
            (76, 'Os', 8, -10.01, -5.48, 0.7499, 2.0, 0.3011, 0.7412, 20.0, 10.0),
            (77, 'Ir', 9, -11.14, -7.58, 0.9512, 2.0, 0.11, 1.0351, 20.0, 10.0),
            (78, 'Pt', 10, -11.32, -8.89, 0.9357, 2.0, 0.0278, 0.9692, 20.0, 10.0),
            (79, 'Au', 11, -12.1, -9.51, 1.3555, 2.0, 0.0555, 1.0048, 20.0, 10.0),
            (80, 'Hg', 12, -12.17, -2.67, 1.2007, 2.0, 0.7723, 2.3139, 20.0, 10.0),
            (81, 'Tl', 3, -20.16, -4.99, 1.2092, 2.0, 0.1288, 2.8056, 20.0, 10.0),
            (82, 'Pb', 4, -22.07, -8.12, 1.1737, 2.0, 0.1035, 3.0969, 20.0, 10.0),
            (83, 'Bi', 5, -19.85, -8.18, 1.1937, 2.0, 0.0115, 1.6598, 20.0, 10.0),
            (84, 'Po', 6, -22.73, -10.66, 1.3045, 2.0, 0.0161, 3.2192, 20.0, 10.0),
            (85, 'At', 7, -16.22, -10.58, 1.1965, 2.0, 0.337, 1.5388, 20.0, 10.0),
            (86, 'Rn', 8, -13.64, -12.17, 1.2654, 2.0, 0.1844, 2.1222, 20.0, 10.0),
        ]
        
        # Parse element parameters
        for data in element_data:
            Z, symbol, valence_e, h_s, h_p, Z_eff, EN_A, J_AA_param, alpha, C6_ref, alpha_ref = data
            self.element_params[Z] = {
                'symbol': symbol,
                'valence_e': valence_e,
                'h_s': h_s,
                'h_p': h_p,
                'Z_eff': Z_eff,
                'EN_A': EN_A,
                'J_AA_param': J_AA_param,
                'alpha': alpha,
                'C6_ref': C6_ref,
                'alpha_ref': alpha_ref
            }
        
        # SK integral data
        # Format: (z1, z2, type, A, alpha)
        sk_data = [
            (1, 1, 'ss_sigma', 2.5, 3.0),
            (1, 6, 'ss_sigma', 2.8, 3.3),
            (1, 6, 'sp_sigma', 3.7, 3.45),
            (6, 6, 'ss_sigma', 3.1, 3.6),
            (6, 6, 'sp_sigma', 4.6, 3.75),
            (6, 6, 'pp_sigma', 6.2, 3.9),
            (6, 6, 'pp_pi', 4.6, 4.05),
            (1, 8, 'ss_sigma', 2.9, 3.36),
            (1, 8, 'sp_sigma', 3.8, 3.51),
            (6, 8, 'ss_sigma', 3.2, 3.66),
            (6, 8, 'sp_sigma', 4.7, 3.81),
            (6, 8, 'pp_sigma', 6.8, 3.96),
            (6, 8, 'pp_pi', 5.2, 4.11),
            (1, 7, 'ss_sigma', 2.8, 3.33),
            (1, 7, 'sp_sigma', 3.7, 3.48),
            (6, 7, 'ss_sigma', 3.1, 3.63),
            (6, 7, 'sp_sigma', 4.6, 3.78),
            (6, 7, 'pp_sigma', 6.2, 3.9),
            (6, 7, 'pp_pi', 4.7, 4.05),
            (7, 7, 'ss_sigma', 3.1, 3.6),
            (7, 7, 'sp_sigma', 4.4, 3.75),
            (7, 7, 'pp_sigma', 6.5, 3.93),
            (7, 7, 'pp_pi', 4.9, 4.08),
            (7, 8, 'ss_sigma', 3.0, 3.57),
            (7, 8, 'sp_sigma', 4.3, 3.72),
            (7, 8, 'pp_sigma', 6.6, 3.945),
            (7, 8, 'pp_pi', 5.0, 4.095),
            (8, 8, 'ss_sigma', 3.1, 3.54),
            (8, 8, 'sp_sigma', 4.5, 3.69),
            (8, 8, 'pp_sigma', 6.9, 3.99),
            (8, 8, 'pp_pi', 5.3, 4.14),
            (1, 35, 'ss_sigma', 2.7, 3.24),
            (1, 35, 'sp_sigma', 3.5, 3.42),
            (6, 35, 'ss_sigma', 3.0, 3.54),
            (6, 35, 'sp_sigma', 4.4, 3.72),
            (6, 35, 'pp_sigma', 6.3, 3.87),
            (6, 35, 'pp_pi', 4.9, 4.02),
            (35, 35, 'ss_sigma', 2.9, 3.45),
            (35, 35, 'sp_sigma', 4.3, 3.66),
            (35, 35, 'pp_sigma', 6.2, 3.84),
            (35, 35, 'pp_pi', 4.7, 3.99),
        ]
        
        for z1, z2, sk_type, A, alpha in sk_data:
            self.sk_params[(z1, z2, sk_type)] = {'A': A, 'alpha': alpha}
        
        # Repulsive parameters
        # Format: (z1, z2, a, b, c)
        rep_data = [
            (1, 1, 0.8, 2.0, 1.0),
            (1, 6, 1.0, 2.8, 1.15),
            (6, 6, 0.9, 2.2, 1.1),
            (1, 8, 0.85, 2.05, 1.02),
            (6, 8, 0.95, 2.35, 1.1),
            (8, 8, 1.0, 2.25, 1.12),
            (1, 7, 0.875, 2.075, 1.035),
            (6, 7, 0.9, 2.125, 1.085),
            (7, 7, 0.975, 2.225, 1.115),
            (7, 8, 0.925, 2.2, 1.095),
            (1, 35, 1.1, 2.4, 1.15),
            (6, 35, 1.15, 2.5, 1.2),
            (35, 35, 1.2, 2.6, 1.25),
        ]
        
        for z1, z2, a, b, c in rep_data:
            self.rep_params[(z1, z2)] = {'a': a, 'b': b, 'c': c}
        
        # D4 dispersion parameters
        self.simple_disp_params = {'s6': 1.0, 's8': 2.97, 'a1': 0.546, 'a2': 5.0}
        
        # SRB parameters
        # Format: (z1, z2, k, R0, alpha)
        srb_data = [
            (1, 6, -0.115, 1.9, 1.0),
            (6, 6, -0.173, 2.66, 1.2),
            (1, 8, -0.138, 1.71, 1.1),
            (6, 8, -0.138, 2.47, 1.3),
            (8, 8, -0.173, 2.28, 1.4),
            (1, 7, -0.127, 1.81, 1.05),
            (6, 7, -0.184, 2.57, 1.25),
            (7, 7, -0.219, 2.38, 1.35),
            (7, 8, -0.15, 2.42, 1.32),
            (1, 35, -0.092, 2.66, 1.05),
            (6, 35, -0.115, 3.04, 1.15),
            (35, 35, -0.092, 3.42, 1.2),
        ]
        
        for z1, z2, k, R0, alpha in srb_data:
            self.srb_params[(z1, z2)] = {'k': k, 'R0': R0, 'alpha': alpha}
        
        # Global parameters
        self.global_params = {'k_wh': 1.925}


def is_covalently_bonded(z1, z2, distance_angstrom, tolerance=1.2):
    """
    Determine if two atoms are covalently bonded based on their distance and covalent radii.
    
    Args:
        z1: Atomic number of first atom
        z2: Atomic number of second atom
        distance_angstrom: Distance between atoms in Angstrom (can be scalar or tensor)
        tolerance: Multiplier for the sum of covalent radii (default: 1.2)
                   If distance <= tolerance * (r1 + r2), atoms are considered bonded
    
    Returns:
        Boolean or boolean tensor indicating if atoms are covalently bonded
    """
    # Get covalent radii from the SIMPLE_DISP_COVALENT_RADII dictionary
    r1 = SIMPLE_DISP_COVALENT_RADII.get(z1, 1.5)  # Default to 1.5 Å if not found
    r2 = SIMPLE_DISP_COVALENT_RADII.get(z2, 1.5)
    
    # Calculate threshold distance
    threshold = tolerance * (r1 + r2)
    
    # Handle both scalar and tensor inputs
    if isinstance(distance_angstrom, torch.Tensor):
        return distance_angstrom <= threshold
    else:
        return distance_angstrom <= threshold


class SQM1Calculator:
    """
    Main SQM1 calculator implementing the complete theory.
    
    All internal calculations use atomic units (Hartree, Bohr).
    Conversions to eV and Angstrom are only for output.
    """
    # All elements with parameters in param_sqm1.txt (86 elements from Z=1 to Z=86)
    SUPPORTED_ELEMENTS = set(range(1, 87))  # H through Rn

    def __init__(self, atomic_numbers, positions, charge=0, uhf=0, params=None, device='cpu', dtype=torch.float64):
        """
        Initialize SQM1 calculator.
        
        Args:
            atomic_numbers: List of atomic numbers
            positions: Nx3 array of positions in Angstrom
            charge: Total molecular charge
            uhf: Number of unpaired electrons (0 for closed-shell)
            params: SQM1Parameters object
            device: PyTorch device ('cpu' or 'cuda')
            dtype: PyTorch data type (torch.float32 or torch.float64)
        """
        self.device = device
        self.dtype = dtype
        
        # Convert to tensors
        self.atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
        positions_tensor = torch.tensor(positions, dtype=dtype, device=device)
        self.positions = positions_tensor * ANGSTROM_TO_BOHR  # Convert to Bohr
        self.positions.requires_grad_(True)  # Enable gradients for forces
        
        self.charge = charge
        self.uhf = uhf
        self.params = params

        if uhf != 0:
            raise NotImplementedError("This implementation only supports closed-shell systems (uhf=0).")
        
        for z in atomic_numbers:
            if z not in self.SUPPORTED_ELEMENTS:
                raise ValueError(f"Element with atomic number {z} is not supported.")

        self.n_atoms = len(atomic_numbers)
        self.valence_electrons = sum(self.params.element_params[z]['valence_e'] for z in atomic_numbers)
        self.n_electrons = self.valence_electrons - self.charge
        if self.n_electrons % 2 != 0:
            raise ValueError("Odd number of electrons for a closed-shell calculation.")
        self.n_occ = self.n_electrons // 2

        self._build_basis_map()

    def _build_basis_map(self):
        """Build mapping from basis functions to atoms."""
        self.basis_map = []
        self.atom_map = []
        current_ao = 0
        for i, z in enumerate(self.atomic_numbers.tolist()):
            self.atom_map.append(current_ao)
            # s orbital
            self.basis_map.append({'atom_idx': i, 'type': 's'})
            current_ao += 1
            # p orbitals
            if self.params.element_params[z]['h_p'] is not None:
                # Store specific p-orbital types ('px', 'py', 'pz')
                # as required by the Slater-Koster transformation function.
                for orbital_type in ['px', 'py', 'pz']:
                    self.basis_map.append({'atom_idx': i, 'type': orbital_type})
                    current_ao += 1
        self.n_basis = len(self.basis_map)
        self.atom_map.append(self.n_basis)

    def _get_sk_integral(self, z1, z2, sk_type, R):
        """
        Get Slater-Koster integral value for given element pair and distance.
        
        Args:
            z1, z2: Atomic numbers
            sk_type: Type of SK integral (e.g., 'ss_sigma', 'pp_sigma')
            R: Distance in Bohr (tensor)
        
        Returns:
            Integral value (tensor)
        """
        z1_key, z2_key = sorted((z1, z2))
        key = (z1_key, z2_key, sk_type)
        if key in self.params.sk_params:
            p = self.params.sk_params[key]
            return p['A'] * torch.exp(-p['alpha'] * R)
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @staticmethod
    def _slater_koster_transform(orbital_i_type, orbital_j_type, cosines, H_integrals, S_integrals):
        """
        Applies Slater-Koster transformation to get H and S matrix elements.

        This is the rigorous implementation that handles:
        - s, p, and d orbitals with complete angular dependence
        - Proper sigma, pi, and delta bonding contributions
        - Direction cosine-based angular factors

        Args:
            orbital_i_type, orbital_j_type: orbital types (e.g., 's', 'px', 'dxy')
            cosines: direction cosines (l, m, n) - tensor
            H_integrals, S_integrals: dictionaries of sigma, pi, delta integrals (tensors)

        Returns:
            H_ij, S_ij: Hamiltonian and overlap matrix elements (tensors)
        """
        l, m, n = cosines
        type_i = orbital_i_type[0]
        type_j = orbital_j_type[0]

        # Get integral values, defaulting to 0 if not found
        def get_H_integral(key): 
            return H_integrals.get(key, torch.tensor(0.0, dtype=l.dtype, device=l.device))
        def get_S_integral(key): 
            return S_integrals.get(key, torch.tensor(0.0, dtype=l.dtype, device=l.device))

        # s-s
        if type_i == 's' and type_j == 's':
            return get_H_integral('ss_sigma'), get_S_integral('ss_sigma')

        # s-p
        if type_i == 's' and type_j == 'p':
            if orbital_j_type == 'px': return l * get_H_integral('sp_sigma'), l * get_S_integral('sp_sigma')
            if orbital_j_type == 'py': return m * get_H_integral('sp_sigma'), m * get_S_integral('sp_sigma')
            if orbital_j_type == 'pz': return n * get_H_integral('sp_sigma'), n * get_S_integral('sp_sigma')

        if type_i == 'p' and type_j == 's':
            if orbital_i_type == 'px': return l * get_H_integral('sp_sigma'), l * get_S_integral('sp_sigma')
            if orbital_i_type == 'py': return m * get_H_integral('sp_sigma'), m * get_S_integral('sp_sigma')
            if orbital_i_type == 'pz': return n * get_H_integral('sp_sigma'), n * get_S_integral('sp_sigma')

        # p-p
        if type_i == 'p' and type_j == 'p':
            V_pp_sigma = get_H_integral('pp_sigma')
            V_pp_pi = get_H_integral('pp_pi')
            S_pp_sigma = get_S_integral('pp_sigma')
            S_pp_pi = get_S_integral('pp_pi')

            if orbital_i_type == 'px' and orbital_j_type == 'px':
                return l*l*V_pp_sigma + (1-l*l)*V_pp_pi, l*l*S_pp_sigma + (1-l*l)*S_pp_pi
            if orbital_i_type == 'py' and orbital_j_type == 'py':
                return m*m*V_pp_sigma + (1-m*m)*V_pp_pi, m*m*S_pp_sigma + (1-m*m)*S_pp_pi
            if orbital_i_type == 'pz' and orbital_j_type == 'pz':
                return n*n*V_pp_sigma + (1-n*n)*V_pp_pi, n*n*S_pp_sigma + (1-n*n)*S_pp_pi
            if (orbital_i_type == 'px' and orbital_j_type == 'py') or (orbital_i_type == 'py' and orbital_j_type == 'px'):
                return l*m*(V_pp_sigma - V_pp_pi), l*m*(S_pp_sigma - S_pp_pi)
            if (orbital_i_type == 'px' and orbital_j_type == 'pz') or (orbital_i_type == 'pz' and orbital_j_type == 'px'):
                return l*n*(V_pp_sigma - V_pp_pi), l*n*(S_pp_sigma - S_pp_pi)
            if (orbital_i_type == 'py' and orbital_j_type == 'pz') or (orbital_i_type == 'pz' and orbital_j_type == 'py'):
                return m*n*(V_pp_sigma - V_pp_pi), m*n*(S_pp_sigma - S_pp_pi)

        # s-d
        if type_i == 's' and type_j == 'd':
            V_sd_sigma = get_H_integral('sd_sigma')
            S_sd_sigma = get_S_integral('sd_sigma')
            sqrt3 = torch.tensor(3.0, dtype=l.dtype, device=l.device).sqrt()
            if orbital_j_type == 'dxy': return sqrt3*l*m*V_sd_sigma, sqrt3*l*m*S_sd_sigma
            if orbital_j_type == 'dyz': return sqrt3*m*n*V_sd_sigma, sqrt3*m*n*S_sd_sigma
            if orbital_j_type == 'dzx': return sqrt3*n*l*V_sd_sigma, sqrt3*n*l*S_sd_sigma
            if orbital_j_type == 'dx2-y2': return sqrt3/2*(l*l-m*m)*V_sd_sigma, sqrt3/2*(l*l-m*m)*S_sd_sigma
            if orbital_j_type == 'd3z2-r2': return (n*n-0.5*(l*l+m*m))*V_sd_sigma, (n*n-0.5*(l*l+m*m))*S_sd_sigma

        if type_i == 'd' and type_j == 's':
            return SQM1Calculator._slater_koster_transform(orbital_j_type, orbital_i_type, cosines, H_integrals, S_integrals)

        # p-d
        if type_i == 'p' and type_j == 'd':
            V_pd_sigma = get_H_integral('pd_sigma')
            V_pd_pi = get_H_integral('pd_pi')
            S_pd_sigma = get_S_integral('pd_sigma')
            S_pd_pi = get_S_integral('pd_pi')
            sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=l.dtype, device=l.device))

            if orbital_i_type == 'px':
                if orbital_j_type == 'dxy': return sqrt3*l*l*m*V_pd_sigma + m*(1-2*l*l)*V_pd_pi, sqrt3*l*l*m*S_pd_sigma + m*(1-2*l*l)*S_pd_pi
                if orbital_j_type == 'dyz': return sqrt3*l*m*n*V_pd_sigma - 2*l*m*n*V_pd_pi, sqrt3*l*m*n*S_pd_sigma - 2*l*m*n*S_pd_pi
                if orbital_j_type == 'dzx': return sqrt3*l*l*n*V_pd_sigma + n*(1-2*l*l)*V_pd_pi, sqrt3*l*l*n*S_pd_sigma + n*(1-2*l*l)*S_pd_pi
                if orbital_j_type == 'dx2-y2': return sqrt3/2*l*(l*l-m*m)*V_pd_sigma + l*(1-(l*l-m*m))*V_pd_pi, sqrt3/2*l*(l*l-m*m)*S_pd_sigma + l*(1-(l*l-m*m))*S_pd_pi
                if orbital_j_type == 'd3z2-r2': return l*(n*n-0.5*(l*l+m*m))*V_pd_sigma - sqrt3*l*n*n*V_pd_pi, l*(n*n-0.5*(l*l+m*m))*S_pd_sigma - sqrt3*l*n*n*S_pd_pi
            if orbital_i_type == 'py':
                if orbital_j_type == 'dxy': return sqrt3*l*m*m*V_pd_sigma + l*(1-2*m*m)*V_pd_pi, sqrt3*l*m*m*S_pd_sigma + l*(1-2*m*m)*S_pd_pi
                if orbital_j_type == 'dyz': return sqrt3*m*m*n*V_pd_sigma + n*(1-2*m*m)*V_pd_pi, sqrt3*m*m*n*S_pd_sigma + n*(1-2*m*m)*S_pd_pi
                if orbital_j_type == 'dzx': return sqrt3*l*m*n*V_pd_sigma - 2*l*m*n*V_pd_pi, sqrt3*l*m*n*S_pd_sigma - 2*l*m*n*S_pd_pi
                if orbital_j_type == 'dx2-y2': return sqrt3/2*m*(l*l-m*m)*V_pd_sigma - m*(1+(l*l-m*m))*V_pd_pi, sqrt3/2*m*(l*l-m*m)*S_pd_sigma - m*(1+(l*l-m*m))*S_pd_pi
                if orbital_j_type == 'd3z2-r2': return m*(n*n-0.5*(l*l+m*m))*V_pd_sigma - sqrt3*m*n*n*V_pd_pi, m*(n*n-0.5*(l*l+m*m))*S_pd_sigma - sqrt3*m*n*n*S_pd_pi
            if orbital_i_type == 'pz':
                if orbital_j_type == 'dxy': return sqrt3*l*m*n*V_pd_sigma - 2*l*m*n*V_pd_pi, sqrt3*l*m*n*S_pd_sigma - 2*l*m*n*S_pd_pi
                if orbital_j_type == 'dyz': return sqrt3*m*n*n*V_pd_sigma + m*(1-2*n*n)*V_pd_pi, sqrt3*m*n*n*S_pd_sigma + m*(1-2*n*n)*S_pd_pi
                if orbital_j_type == 'dzx': return sqrt3*n*n*l*V_pd_sigma + l*(1-2*n*n)*V_pd_pi, sqrt3*n*n*l*S_pd_sigma + l*(1-2*n*n)*S_pd_pi
                if orbital_j_type == 'dx2-y2': return sqrt3/2*n*(l*l-m*m)*V_pd_sigma - n*(l*l-m*m)*V_pd_pi, sqrt3/2*n*(l*l-m*m)*S_pd_sigma - n*(l*l-m*m)*S_pd_pi
                if orbital_j_type == 'd3z2-r2': return n*(n*n-0.5*(l*l+m*m))*V_pd_sigma + sqrt3*n*(l*l+m*m)*V_pd_pi, n*(n*n-0.5*(l*l+m*m))*S_pd_sigma + sqrt3*n*(l*l+m*m)*S_pd_pi

        if type_i == 'd' and type_j == 'p':
            return SQM1Calculator._slater_koster_transform(orbital_j_type, orbital_i_type, cosines, H_integrals, S_integrals)

        # d-d
        if type_i == 'd' and type_j == 'd':
            V_dd_sigma = get_H_integral('dd_sigma')
            V_dd_pi = get_H_integral('dd_pi')
            V_dd_delta = get_H_integral('dd_delta')
            S_dd_sigma = get_S_integral('dd_sigma')
            S_dd_pi = get_S_integral('dd_pi')
            S_dd_delta = get_S_integral('dd_delta')
            sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=l.dtype, device=l.device))

            if orbital_i_type == 'dxy':
                if orbital_j_type == 'dxy': return 3*l*l*m*m*V_dd_sigma + (l*l+m*m-4*l*l*m*m)*V_dd_pi + (n*n+l*l*m*m)*V_dd_delta, 3*l*l*m*m*S_dd_sigma + (l*l+m*m-4*l*l*m*m)*S_dd_pi + (n*n+l*l*m*m)*S_dd_delta
                if orbital_j_type == 'dyz': return 3*l*m*m*n*V_dd_sigma + l*n*(1-4*m*m)*V_dd_pi + l*n*(m*m-1)*V_dd_delta, 3*l*m*m*n*S_dd_sigma + l*n*(1-4*m*m)*S_dd_pi + l*n*(m*m-1)*S_dd_delta
                if orbital_j_type == 'dzx': return 3*l*l*m*n*V_dd_sigma + m*n*(1-4*l*l)*V_dd_pi + m*n*(l*l-1)*V_dd_delta, 3*l*l*m*n*S_dd_sigma + m*n*(1-4*l*l)*S_dd_pi + m*n*(l*l-1)*S_dd_delta
                if orbital_j_type == 'dx2-y2': return 1.5*l*m*(l*l-m*m)*V_dd_sigma + 2*l*m*(m*m-l*l)*V_dd_pi + 0.5*l*m*(l*l-m*m)*V_dd_delta, 1.5*l*m*(l*l-m*m)*S_dd_sigma + 2*l*m*(m*m-l*l)*S_dd_pi + 0.5*l*m*(l*l-m*m)*S_dd_delta
                if orbital_j_type == 'd3z2-r2': return sqrt3*l*m*(n*n-0.5*(l*l+m*m))*V_dd_sigma - 2*sqrt3*l*m*n*n*V_dd_pi + 0.5*sqrt3*l*m*(l*l+m*m)*V_dd_delta, sqrt3*l*m*(n*n-0.5*(l*l+m*m))*S_dd_sigma - 2*sqrt3*l*m*n*n*S_dd_pi + 0.5*sqrt3*l*m*(l*l+m*m)*S_dd_delta
            if orbital_i_type == 'dyz':
                if orbital_j_type == 'dyz': return 3*m*m*n*n*V_dd_sigma + (m*m+n*n-4*m*m*n*n)*V_dd_pi + (l*l+m*m*n*n)*V_dd_delta, 3*m*m*n*n*S_dd_sigma + (m*m+n*n-4*m*m*n*n)*S_dd_pi + (l*l+m*m*n*n)*S_dd_delta
                if orbital_j_type == 'dzx': return 3*l*m*n*n*V_dd_sigma + l*m*(1-4*n*n)*V_dd_pi + l*m*(n*n-1)*V_dd_delta, 3*l*m*n*n*S_dd_sigma + l*m*(1-4*n*n)*S_dd_pi + l*m*(n*n-1)*S_dd_delta
                if orbital_j_type == 'dx2-y2': return 1.5*m*n*(l*l-m*m)*V_dd_sigma - m*n*(1+2*(l*l-m*m))*V_dd_pi + 0.5*m*n*(2-(l*l-m*m))*V_dd_delta, 1.5*m*n*(l*l-m*m)*S_dd_sigma - m*n*(1+2*(l*l-m*m))*S_dd_pi + 0.5*m*n*(2-(l*l-m*m))*S_dd_delta
                if orbital_j_type == 'd3z2-r2': return sqrt3*m*n*(n*n-0.5*(l*l+m*m))*V_dd_sigma + sqrt3*m*n*(l*l+m*m-n*n)*V_dd_pi - 0.5*sqrt3*m*n*(l*l+m*m)*V_dd_delta, sqrt3*m*n*(n*n-0.5*(l*l+m*m))*S_dd_sigma + sqrt3*m*n*(l*l+m*m-n*n)*S_dd_pi - 0.5*sqrt3*m*n*(l*l+m*m)*S_dd_delta
            if orbital_i_type == 'dzx':
                if orbital_j_type == 'dzx': return 3*n*n*l*l*V_dd_sigma + (n*n+l*l-4*n*n*l*l)*V_dd_pi + (m*m+n*n*l*l)*V_dd_delta, 3*n*n*l*l*S_dd_sigma + (n*n+l*l-4*n*n*l*l)*S_dd_pi + (m*m+n*n*l*l)*S_dd_delta
                if orbital_j_type == 'dx2-y2': return 1.5*n*l*(l*l-m*m)*V_dd_sigma + n*l*(1-2*(l*l-m*m))*V_dd_pi - 0.5*n*l*(2+(l*l-m*m))*V_dd_delta, 1.5*n*l*(l*l-m*m)*S_dd_sigma + n*l*(1-2*(l*l-m*m))*S_dd_pi - 0.5*n*l*(2+(l*l-m*m))*S_dd_delta
                if orbital_j_type == 'd3z2-r2': return sqrt3*n*l*(n*n-0.5*(l*l+m*m))*V_dd_sigma + sqrt3*n*l*(m*m+n*n-l*l)*V_dd_pi - 0.5*sqrt3*n*l*(l*l+m*m)*V_dd_delta, sqrt3*n*l*(n*n-0.5*(l*l+m*m))*S_dd_sigma + sqrt3*n*l*(m*m+n*n-l*l)*S_dd_pi - 0.5*sqrt3*n*l*(l*l+m*m)*S_dd_delta
            if orbital_i_type == 'dx2-y2':
                if orbital_j_type == 'dx2-y2': return 0.75*(l*l-m*m)*(l*l-m*m)*V_dd_sigma + (l*l+m*m)*(1-0.5*(l*l-m*m)*(l*l-m*m))*V_dd_pi + (1-0.5*(l*l+m*m)*(l*l+m*m))*V_dd_delta, 0.75*(l*l-m*m)*(l*l-m*m)*S_dd_sigma + (l*l+m*m)*(1-0.5*(l*l-m*m)*(l*l-m*m))*S_dd_pi + (1-0.5*(l*l+m*m)*(l*l+m*m))*S_dd_delta
                if orbital_j_type == 'd3z2-r2': return 0.5*sqrt3*(l*l-m*m)*(n*n-0.5*(l*l+m*m))*V_dd_sigma - sqrt3*(l*l-m*m)*n*n*V_dd_pi - 0.5*sqrt3*(l*l-m*m)*(l*l+m*m)*V_dd_delta, 0.5*sqrt3*(l*l-m*m)*(n*n-0.5*(l*l+m*m))*S_dd_sigma - sqrt3*(l*l-m*m)*n*n*S_dd_pi - 0.5*sqrt3*(l*l-m*m)*(l*l+m*m)*S_dd_delta
            if orbital_i_type == 'd3z2-r2':
                if orbital_j_type == 'd3z2-r2': return (n*n-0.5*(l*l+m*m))*(n*n-0.5*(l*l+m*m))*V_dd_sigma + 3*n*n*(l*l+m*m)*V_dd_pi + 0.75*(l*l+m*m)*(l*l+m*m)*V_dd_delta, (n*n-0.5*(l*l+m*m))*(n*n-0.5*(l*l+m*m))*S_dd_sigma + 3*n*n*(l*l+m*m)*S_dd_pi + 0.75*(l*l+m*m)*(l*l+m*m)*S_dd_delta
        
        zero = torch.tensor(0.0, dtype=l.dtype, device=l.device)
        return zero, zero

    def _build_matrices(self):
        """
        Build Hamiltonian and Overlap matrices using rigorous Slater-Koster method.
        
        Returns:
            Tuple of (H, S) matrices in atomic units (tensors)
        """
        S = torch.zeros((self.n_basis, self.n_basis), dtype=self.dtype, device=self.device)
        H = torch.zeros((self.n_basis, self.n_basis), dtype=self.dtype, device=self.device)

        # On-site (diagonal) blocks
        for i in range(self.n_atoms):
            z = self.atomic_numbers[i].item()
            p = self.params.element_params[z]
            start, end = self.atom_map[i], self.atom_map[i+1]
            S[start:end, start:end] = torch.eye(end-start, dtype=self.dtype, device=self.device)
            H[start, start] = p['h_s'] * EV_TO_HARTREE  # Convert to Hartree
            if end - start == 4:  # s and p
                H[start+1:end, start+1:end] = torch.eye(3, dtype=self.dtype, device=self.device) * p['h_p'] * EV_TO_HARTREE

        # Off-site (off-diagonal) blocks
        sk_types = [
            'ss_sigma', 'sp_sigma', 'pp_sigma', 'pp_pi',
            'sd_sigma', 'pd_sigma', 'pd_pi',
            'dd_sigma', 'dd_pi', 'dd_delta'
        ]
        
        for i in range(self.n_atoms):
            zi = self.atomic_numbers[i].item()
            i_start, i_end = self.atom_map[i], self.atom_map[i+1]
            
            for j in range(i + 1, self.n_atoms):
                zj = self.atomic_numbers[j].item()
                j_start, j_end = self.atom_map[j], self.atom_map[j+1]

                R_vec = self.positions[i] - self.positions[j]
                dist = torch.linalg.norm(R_vec)
                
                if dist < 1e-9:
                    continue
                    
                cosines = R_vec / dist
                
                # Build the dictionary of two-center integrals
                sk_integrals = {}
                for sk_type in sk_types:
                    val = self._get_sk_integral(zi, zj, sk_type, dist)
                    sk_integrals[sk_type] = val
                
                # Iterate over basis functions for this atom pair
                for mu in range(i_start, i_end):
                    type_i = self.basis_map[mu]['type']
                    
                    for nu in range(j_start, j_end):
                        type_j = self.basis_map[nu]['type']
                        
                        # Calculate H_ij and S_ij using the rigorous SK transform
                        H_ij, S_ij = self._slater_koster_transform(
                            type_i, type_j, cosines, 
                            sk_integrals, sk_integrals
                        )
                        
                        S[mu, nu] = S_ij
                        S[nu, mu] = S_ij
                        H[mu, nu] = H_ij
                        H[nu, mu] = H_ij

        return H, S

    def _solve_eht(self):
        """
        Solve Extended Hückel Theory eigenvalue problem.
        
        Returns:
            E_EHT in Hartree (relative to isolated atoms)
        """
        H, S = self._build_matrices()
        
        # Solve generalized eigenvalue problem HC = SCE
        # Use Cholesky decomposition: S = L L^T
        # Then solve L^{-1} H L^{-T} y = y E, where C = L^{-T} y
        try:
            L = torch.linalg.cholesky(S)
            L_inv = torch.linalg.inv(L)
            H_prime = L_inv @ H @ L_inv.T
            eigvals, y = torch.linalg.eigh(H_prime)
            eigvecs = L_inv.T @ y
        except RuntimeError:
            # If Cholesky fails, use alternative approach
            # Transform to standard eigenvalue problem using eigendecomposition of S
            eigvals_S, eigvecs_S = torch.linalg.eigh(S)
            # Filter out near-zero eigenvalues
            threshold = 1e-10
            idx = eigvals_S > threshold
            eigvals_S = eigvals_S[idx]
            eigvecs_S = eigvecs_S[:, idx]
            # Transform H
            S_sqrt_inv = eigvecs_S @ torch.diag(1.0 / torch.sqrt(eigvals_S)) @ eigvecs_S.T
            H_prime = S_sqrt_inv @ H @ S_sqrt_inv
            eigvals, y = torch.linalg.eigh(H_prime)
            eigvecs = S_sqrt_inv @ y
        
        # Sort eigenvalues
        idx = eigvals.argsort()
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx]

        # Density matrix for occupied orbitals
        self.density_matrix = 2 * self.eigvecs[:, :self.n_occ] @ self.eigvecs[:, :self.n_occ].T
        
        # Band structure energy
        e_bs = torch.sum(self.eigvals[:self.n_occ]) * 2
        
        # Calculate atomic reference energy
        e_ref = self._calculate_atomic_reference_energy()
        
        # Return molecular energy relative to isolated atoms
        e_eht = e_bs - e_ref
        return e_eht

    def _calculate_atomic_reference_energy(self):
        """
        Calculate the sum of isolated atomic energies.
        
        Returns:
            E_ref in Hartree (tensor)
        """
        e_ref = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for z in self.atomic_numbers.tolist():
            p = self.params.element_params[z]
            valence_e = p['valence_e']
            
            e_s = p['h_s'] * EV_TO_HARTREE
            remaining_e = valence_e
            
            # Fill s orbital first (max 2 electrons)
            if remaining_e >= 2:
                e_ref += 2 * e_s
                remaining_e -= 2
            else:
                e_ref += remaining_e * e_s
                remaining_e = 0
            
            # Fill p orbitals if present and if there are remaining electrons
            if p['h_p'] is not None and remaining_e > 0:
                e_p = p['h_p'] * EV_TO_HARTREE
                e_ref += min(remaining_e, 6) * e_p
        
        return e_ref

    def _solve_eeq(self):
        """
        Solve Electronegativity Equilibration Model (EEQ) equations.
        
        Reference: SQM1 Paper, Section 2.1.1
        
        Returns:
            E_IES in Hartree (tensor)
        """
        A = torch.zeros((self.n_atoms, self.n_atoms), dtype=self.dtype, device=self.device)
        b = torch.zeros(self.n_atoms, dtype=self.dtype, device=self.device)

        for i in range(self.n_atoms):
            zi = self.atomic_numbers[i].item()
            p_i = self.params.element_params[zi]
            b[i] = -p_i['EN_A'] * EV_TO_HARTREE
            
            gamma_AA = p_i['J_AA_param'] * (p_i['Z_eff'] ** p_i['alpha'])
            A[i, i] = gamma_AA * EV_TO_HARTREE

            for j in range(i + 1, self.n_atoms):
                zj = self.atomic_numbers[j].item()
                pj = self.params.element_params[zj]
                R_ij = torch.linalg.norm(self.positions[i] - self.positions[j])
                
                sigma_A = 0.7 / torch.sqrt(torch.tensor(max(abs(p_i['EN_A']), 0.5), dtype=self.dtype, device=self.device))
                sigma_B = 0.7 / torch.sqrt(torch.tensor(max(abs(pj['EN_A']), 0.5), dtype=self.dtype, device=self.device))
                sigma_sum = sigma_A + sigma_B
                
                gamma_AB = 1.0 / torch.sqrt(R_ij**2 + sigma_sum**2)
                A[i, j] = A[j, i] = gamma_AB
        
        # Lagrange multiplier for charge constraint
        A_ext = torch.ones((self.n_atoms + 1, self.n_atoms + 1), dtype=self.dtype, device=self.device)
        A_ext[:self.n_atoms, :self.n_atoms] = A
        A_ext[self.n_atoms, self.n_atoms] = 0
        
        b_ext = torch.zeros(self.n_atoms + 1, dtype=self.dtype, device=self.device)
        b_ext[:self.n_atoms] = b
        b_ext[self.n_atoms] = self.charge

        x = torch.linalg.solve(A_ext, b_ext)
        self.eeq_charges = x[:self.n_atoms]

        # Calculate electrostatic energy (IES)
        E_ies = 0.5 * self.eeq_charges @ A @ self.eeq_charges + b @ self.eeq_charges
        return E_ies

    def _calculate_coordination_numbers(self):
        """
        Calculate fractional coordination numbers for all atoms.
        Uses SimpleDispersion-style coordination number definition.
        
        Returns:
            Array of coordination numbers (tensor)
        """
        cn = torch.zeros(self.n_atoms, dtype=self.dtype, device=self.device)
        for i in range(self.n_atoms):
            zi = self.atomic_numbers[i].item()
            r_cov_i = SIMPLE_DISP_COVALENT_RADII.get(zi, 1.5)
            
            for j in range(self.n_atoms):
                if i == j:
                    continue
                    
                zj = self.atomic_numbers[j].item()
                R_ij_bohr = torch.linalg.norm(self.positions[i] - self.positions[j])
                R_ij_ang = R_ij_bohr * BOHR_TO_ANGSTROM
                
                if R_ij_ang > SIMPLE_DISP_CN_CUTOFF:
                    continue
                
                r_cov_j = SIMPLE_DISP_COVALENT_RADII.get(zj, 1.5)
                r_cov_sum = r_cov_i + r_cov_j
                argument = SIMPLE_DISP_CN_K1 * (SIMPLE_DISP_CN_K2 * r_cov_sum / R_ij_ang - 1.0)
                cn[i] += 1.0 / (1.0 + torch.exp(-argument))
        
        return cn

    def _calculate_repulsive_energy(self):
        """
        Calculate repulsive potential energy with environment-dependent scaling.
        
        Returns:
            E_rep in Hartree (tensor)
        """
        E_rep = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
        cn = self._calculate_coordination_numbers()
        
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                zi, zj = sorted((self.atomic_numbers[i].item(), self.atomic_numbers[j].item()))
                if (zi, zj) not in self.params.rep_params:
                    continue
                
                p = self.params.rep_params[(zi, zj)]
                R_ij = torch.linalg.norm(self.positions[i] - self.positions[j])
                
                Z_eff_i = self.params.element_params[self.atomic_numbers[i].item()]['Z_eff']
                Z_eff_j = self.params.element_params[self.atomic_numbers[j].item()]['Z_eff']

                # Exponential repulsive potential
                term = torch.exp(p['b'] * (1.0 - (R_ij / (p['a'] * (1/Z_eff_i + 1/Z_eff_j)))**p['c']))
                base_rep = (Z_eff_i * Z_eff_j / R_ij) * term
                
                # Environment-dependent scaling
                cn_i = cn[i] if self.atomic_numbers[i].item() == zi else cn[j]
                cn_j = cn[j] if self.atomic_numbers[j].item() == zj else cn[i]
                
                cn_scale_i = 1.0 + 0.5 * torch.exp(-(cn_i - 1.0) / 2.0)
                cn_scale_j = 1.0 + 0.5 * torch.exp(-(cn_j - 1.0) / 2.0)
                cn_scaling = torch.sqrt(cn_scale_i * cn_scale_j)
                
                E_rep += base_rep * cn_scaling
        
        return E_rep * EV_TO_HARTREE

    def _calculate_simple_dispersion(self):
        """
        Calculate SimpleDispersion energy with coordination number dependence and charge scaling.
        
        Reference: Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019)
        
        Returns:
            E_SimpleDisp in Hartree (tensor)
        """
        E_simple_disp = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        p_simple_disp = self.params.simple_disp_params
        
        # Calculate coordination numbers
        cn = torch.zeros(self.n_atoms, dtype=self.dtype, device=self.device)
        for i in range(self.n_atoms):
            zi = self.atomic_numbers[i].item()
            r_cov_i = SIMPLE_DISP_COVALENT_RADII.get(zi, 1.5)
            
            for j in range(self.n_atoms):
                if i == j:
                    continue
                    
                zj = self.atomic_numbers[j].item()
                R_ij_bohr = torch.linalg.norm(self.positions[i] - self.positions[j])
                R_ij_ang = R_ij_bohr * BOHR_TO_ANGSTROM
                
                if R_ij_ang > SIMPLE_DISP_CN_CUTOFF:
                    continue
                
                r_cov_j = SIMPLE_DISP_COVALENT_RADII.get(zj, 1.5)
                r_cov_sum = r_cov_i + r_cov_j
                argument = SIMPLE_DISP_CN_K1 * (SIMPLE_DISP_CN_K2 * r_cov_sum / R_ij_ang - 1.0)
                cn[i] += 1.0 / (1.0 + torch.exp(-argument))

        # Calculate charge-dependent C6 coefficients
        c6 = torch.zeros(self.n_atoms, dtype=self.dtype, device=self.device)
        for i in range(self.n_atoms):
            zi = self.atomic_numbers[i].item()
            p_i = self.params.element_params[zi]
            
            c6_ref = p_i['C6_ref']
            q_i = self.eeq_charges[i]
            
            charge_scaling = 1.0 / (1.0 + 0.5 * torch.abs(q_i))
            cn_scaling = 1.0 / (1.0 + 0.08 * cn[i])
            
            c6[i] = c6_ref * charge_scaling * cn_scaling

        # Identify bonded pairs
        bonded_pairs = set()
        for i in range(self.n_atoms):
            zi = self.atomic_numbers[i].item()
            r_cov_i = SIMPLE_DISP_COVALENT_RADII.get(zi, 1.5)
            
            for j in range(i + 1, self.n_atoms):
                zj = self.atomic_numbers[j].item()
                r_cov_j = SIMPLE_DISP_COVALENT_RADII.get(zj, 1.5)
                R_ij_ang = torch.linalg.norm(self.positions[i] - self.positions[j]) * BOHR_TO_ANGSTROM
                
                if R_ij_ang < 1.3 * (r_cov_i + r_cov_j):
                    bonded_pairs.add((i, j))
        
        # Calculate dispersion energy
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                R_ij_bohr = torch.linalg.norm(self.positions[i] - self.positions[j])
                
                c6_ij = torch.sqrt(c6[i] * c6[j])
                
                alpha_i = self.params.element_params[self.atomic_numbers[i].item()]['alpha_ref']
                alpha_j = self.params.element_params[self.atomic_numbers[j].item()]['alpha_ref']
                c8_ij = 3.0 * c6_ij * torch.sqrt(torch.tensor(alpha_i * alpha_j, dtype=self.dtype, device=self.device))

                if c6_ij > 1e-10:
                    sqrt3 = torch.tensor(3.0, dtype=self.dtype, device=self.device).sqrt()
                    R0_ij_base = p_simple_disp['a1'] * torch.sqrt(sqrt3 * c8_ij / c6_ij) + p_simple_disp['a2']
                else:
                    R0_ij_base = p_simple_disp['a2']
                
                # Enhanced damping for intramolecular interactions
                if (i, j) in bonded_pairs:
                    zi = self.atomic_numbers[i].item()
                    zj = self.atomic_numbers[j].item()
                    EN_i = self.params.element_params[zi]['EN_A']
                    EN_j = self.params.element_params[zj]['EN_A']
                    delta_EN = abs(EN_i - EN_j)
                    
                    avg_alpha = 0.5 * (alpha_i + alpha_j)
                    Pol_AB = min(1.0, (delta_EN / 3.0) * (avg_alpha / 20.0))
                    
                    k_damp = 0.5
                    R0_ij = R0_ij_base * (1.0 + k_damp * Pol_AB)
                else:
                    R0_ij = R0_ij_base
                
                R6 = R_ij_bohr**6
                R8 = R_ij_bohr**8
                R0_6 = R0_ij**6
                R0_8 = R0_ij**8
                
                term6 = p_simple_disp['s6'] * c6_ij / (R6 + R0_6)
                term8 = p_simple_disp['s8'] * c8_ij / (R8 + R0_8)
                
                E_simple_disp -= (term6 + term8)
        
        return E_simple_disp

    def _calculate_srb_energy(self):
        """
        Calculate Short-Range Basis (SRB) correction energy.
        
        Returns:
            E_SRB in Hartree (tensor)
        """
        E_srb = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                zi, zj = sorted((self.atomic_numbers[i].item(), self.atomic_numbers[j].item()))
                if (zi, zj) not in self.params.srb_params:
                    continue
                
                p = self.params.srb_params[(zi, zj)]
                R_ij = torch.linalg.norm(self.positions[i] - self.positions[j])
                
                p_i = self.params.element_params[self.atomic_numbers[i].item()]
                p_j = self.params.element_params[self.atomic_numbers[j].item()]
                
                delta_EN = abs(p_i['EN_A'] - p_j['EN_A'])
                
                alpha_sum = p_i['alpha_ref'] + p_j['alpha_ref']
                k_pol_damp = 0.02
                
                g_scal_base = delta_EN * delta_EN
                g_scal = g_scal_base / (1.0 + k_pol_damp * alpha_sum)
                
                E_srb += p['k'] * g_scal * torch.exp(-p['alpha'] * (R_ij - p['R0'])**2)
        
        return E_srb * EV_TO_HARTREE

    def calculate_total_energy(self, coords=None, atomic_numbers=None, total_charge=None, external_electric_field=None):
        """
        Calculate total SQM1 energy.
        
        E_total = E_EHT + E_IES + E_rep + E_SimpleDisp + E_SRB + E_field
        
        Args:
            coords: Optional coordinates tensor (n_atoms, 3) in Bohr with requires_grad=True
            atomic_numbers: Optional atomic numbers tensor
            total_charge: Optional total charge
            external_electric_field: Optional external electric field vector (3,) in atomic units
        
        Returns:
            Total energy as a scalar torch.Tensor
        """
        # Use instance attributes if not provided
        if coords is None:
            coords = self.positions
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers
        if total_charge is None:
            total_charge = self.charge
            
        # Temporarily update positions for calculation
        original_positions = self.positions
        self.positions = coords
        
        e_eht = self._solve_eht()
        e_ies = self._solve_eeq()
        e_rep = self._calculate_repulsive_energy()
        e_simple_disp = self._calculate_simple_dispersion()
        e_srb = self._calculate_srb_energy()
        
        # Calculate electric field interaction if field is provided
        e_field = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if external_electric_field is not None:
            # E_field = - sum_i (q_i * r_i) · F_ext
            # where q_i are the EEQ charges and r_i are positions
            dipole_component = torch.sum(self.eeq_charges.unsqueeze(1) * coords, dim=0)
            e_field = -torch.dot(dipole_component, external_electric_field)
        
        # Restore original positions
        self.positions = original_positions
        
        total_energy = e_eht + e_ies + e_rep + e_simple_disp + e_srb + e_field 
        
        return total_energy
    
    def calculate_energies(self):
        """
        Calculate and return energy components as a dictionary (for backward compatibility).
        
        Returns:
            Dictionary of energy components in Hartree (tensors)
        """
        e_eht = self._solve_eht()
        e_ies = self._solve_eeq()
        e_rep = self._calculate_repulsive_energy()
        e_simple_disp = self._calculate_simple_dispersion()
        e_srb = self._calculate_srb_energy()
        
        # Calculate dipole moment using AD
        dipole = self._calculate_dipole_moment_ad()

        self.energies = {
            'EHT': e_eht,
            'IES': e_ies,
            'Repulsive': e_rep,
            'SimpleDispersion': e_simple_disp,
            'SRB': e_srb,
            'Total': e_eht + e_ies + e_rep + e_simple_disp + e_srb,
            'dipole_moment': dipole
        }
        return self.energies
    
    def calculate_energy_and_gradient(self, coords=None, atomic_numbers=None, total_charge=None):
        """
        Calculate total energy and gradient using PyTorch autograd.
        
        Args:
            coords: Coordinates tensor (n_atoms, 3) in Bohr with requires_grad=True.
                   If None, uses self.positions (which must have requires_grad=True)
            atomic_numbers: Optional atomic numbers tensor
            total_charge: Optional total charge
        
        Returns:
            Tuple of (total_energy, gradient):
                - total_energy: Total energy as a scalar torch.Tensor
                - gradient: Gradient of energy w.r.t. coords, shape (n_atoms, 3)
        """
        # Use instance positions if coords not provided
        if coords is None:
            coords = self.positions
            if not coords.requires_grad:
                coords.requires_grad_(True)
        
        # Ensure coords requires gradients
        if not coords.requires_grad:
            coords = coords.clone().detach().requires_grad_(True)
        
        # Calculate total energy
        total_energy = self.calculate_total_energy(coords, atomic_numbers, total_charge)
        
        # Calculate gradient using autograd.grad
        gradient = torch.autograd.grad(
            outputs=total_energy,
            inputs=coords,
            create_graph=False
        )[0]
        
        return total_energy, gradient
    
    def _calculate_dipole_moment_ad(self):
        """
        Calculate dipole moment using automatic differentiation.
        
        The dipole moment is defined as the negative gradient of the total energy
        with respect to an external electric field, evaluated at zero field:
        μ = -∂E_total/∂F_ext |_{F_ext=0}
        
        Returns:
            Dipole moment vector (3,) in atomic units (e·Bohr)
        """
        # Create a zero electric field tensor with gradient tracking
        field_tensor = torch.zeros(3, dtype=self.dtype, device=self.device, requires_grad=True)
        
        # Calculate energy with this field
        energy = self.calculate_total_energy(external_electric_field=field_tensor)
        
        # Calculate dipole as negative gradient
        dipole = -torch.autograd.grad(
            outputs=energy,
            inputs=field_tensor,
            create_graph=False
        )[0]
        
        return dipole
    
    def calculate_hessian(self, coords=None, atomic_numbers=None, total_charge=None, method='analytical'):
        """
        Calculate the Hessian matrix of the total energy with respect to atomic coordinates.
        
        Args:
            coords: Coordinates tensor (n_atoms, 3) in Bohr. If None, uses self.positions
            atomic_numbers: Optional atomic numbers tensor
            total_charge: Optional total charge
            method: 'analytical' uses torch.autograd.functional.hessian,
                   'numerical' uses finite differences
        
        Returns:
            Hessian matrix of shape (3*n_atoms, 3*n_atoms) in Hartree/Bohr^2
        """
        if coords is None:
            coords = self.positions.detach().clone()
        else:
            coords = coords.detach().clone()
        
        n_atoms = coords.shape[0]
        
        if method == 'analytical':
            # Use PyTorch's native hessian function
            # Create a wrapper that computes energy for flat coordinates
            def energy_func(coords_flat):
                coords_reshaped = coords_flat.reshape(n_atoms, 3)
                return self.calculate_total_energy(coords_reshaped, atomic_numbers, total_charge)
            
            # Flatten coordinates for hessian computation
            coords_flat = coords.flatten()
            coords_flat.requires_grad_(True)
            
            # Calculate Hessian - output shape will be (3*n_atoms, 3*n_atoms)
            try:
                hessian = torch.autograd.functional.hessian(energy_func, coords_flat)
            except RuntimeError as e:
                print(f"Warning: Analytical Hessian calculation failed: {e}")
                print("Falling back to numerical method")
                return self.calculate_hessian(coords, atomic_numbers, total_charge, method='numerical')
            
            return hessian
            
        elif method == 'numerical':
            # Numerical finite difference method
            hessian = torch.zeros((n_atoms * 3, n_atoms * 3), dtype=self.dtype, device=self.device)
            h = 1e-5  # Finite difference step in Bohr
            
            for i in range(n_atoms):
                for j in range(3):
                    idx = i * 3 + j
                    coords_copy = coords.clone()
                    
                    # Forward difference
                    coords_copy[i, j] = coords[i, j] + h
                    _, grad_plus = self.calculate_energy_and_gradient(coords_copy, atomic_numbers, total_charge)
                    
                    # Backward difference
                    coords_copy[i, j] = coords[i, j] - h
                    _, grad_minus = self.calculate_energy_and_gradient(coords_copy, atomic_numbers, total_charge)
                    
                    # Central difference for Hessian row
                    hessian[idx, :] = (grad_plus - grad_minus).flatten() / (2 * h)
            
            return hessian
        else:
            raise ValueError(f"Unknown method: {method}. Use 'analytical' or 'numerical'.")

    def get_forces(self, use_numerical=False):
        """
        Calculate forces via automatic differentiation or numerical differentiation.
        
        Args:
            use_numerical: If True, use numerical differentiation as fallback
        
        Returns:
            Forces in Hartree/Bohr (tensor)
        """
        if not use_numerical:
            # Use automatic differentiation via calculate_energy_and_gradient
            try:
                _, gradient = self.calculate_energy_and_gradient()
                # Forces are negative gradient
                forces = -gradient
                return forces
            except (RuntimeError, ValueError) as e:
                print(f"Warning: Autograd failed ({e}), falling back to numerical differentiation")
                use_numerical = True
        
        if use_numerical:
            # Numerical differentiation fallback
            forces = torch.zeros_like(self.positions)
            h = 1e-5  # Finite difference step in Bohr
            
            for i in range(self.n_atoms):
                for j in range(3):
                    original_pos = self.positions[i, j].item()
                    
                    self.positions.data[i, j] = original_pos + h
                    e_plus = self.calculate_total_energy()
                    
                    self.positions.data[i, j] = original_pos - h
                    e_minus = self.calculate_total_energy()
                    
                    self.positions.data[i, j] = original_pos  # Restore
                    
                    forces[i, j] = -(e_plus - e_minus) / (2 * h)
            
            # Recalculate energy at original position
            self.calculate_total_energy()
            return forces

    def optimize_geometry(self, method='Adam', lr=0.01, max_steps=1000, gtol=1e-3, max_distance_deviation=0.10, disp=False):
        """
        Optimize molecular geometry using PyTorch optimizer.
        
        Args:
            method: Optimization method ('LBFGS' or 'Adam'). Default is 'Adam' for stability.
            lr: Learning rate for optimizer (default: 0.01 for Adam, will be scaled down for LBFGS)
            max_steps: Maximum optimization steps
            gtol: Gradient tolerance for convergence
            max_distance_deviation: Maximum allowed deviation of interatomic distances
            disp: If True, display verbose output during optimization
        
        Returns:
            Optimized positions in Angstrom (tensor)
        """
        # Store initial positions and distances
        initial_positions = self.positions.detach().clone()
        initial_distances = {}
        bonded_pairs = set()
        
        # Only store distances for covalently bonded pairs
        for i in range(self.n_atoms):
            for j in range(i+1, self.n_atoms):
                d = torch.linalg.norm(initial_positions[i] - initial_positions[j])
                d_angstrom = d * BOHR_TO_ANGSTROM
                
                # Check if atoms are covalently bonded
                if is_covalently_bonded(self.atomic_numbers[i].item(), self.atomic_numbers[j].item(), d_angstrom.item()):
                    initial_distances[(i, j)] = d
                    bonded_pairs.add((i, j))
        
        # Set up optimizer based on method with more conservative parameters
        if method.upper() == 'LBFGS':
            # Use very conservative LBFGS settings to prevent gradient explosion
            # Scale down learning rate for LBFGS
            lbfgs_lr = min(lr * 0.1, 0.001)
            optimizer = torch.optim.LBFGS(
                [self.positions], 
                lr=lbfgs_lr, 
                max_iter=5,  # Very conservative max iterations per step
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                history_size=10  # Limit memory
            )
        elif method.upper() == 'ADAM':
            # Adam is generally more stable for molecular optimization
            optimizer = torch.optim.Adam([self.positions], lr=lr)
        else:
            raise ValueError(f"Unknown optimization method: {method}. Use 'LBFGS' or 'Adam'")
        
        # Track consecutive failed steps for early stopping
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Store best state for recovery
        best_loss = float('inf')
        best_positions = self.positions.detach().clone()
        
        def closure():
            optimizer.zero_grad()
            try:
                E = self.calculate_total_energy()
                
                # Check for NaN or inf in energy
                if not torch.isfinite(E):
                    if disp:
                        print("Warning: Energy is NaN or inf, returning large penalty")
                    return torch.tensor(1e10, dtype=self.dtype, device=self.device)
                
                # Sanity check: energy shouldn't be extremely negative (indication of numerical issues)
                if E < -1000.0:  # -1000 Hartree is extremely negative for small molecules
                    if disp:
                        print(f"Warning: Energy is unrealistically negative ({E.item():.2f} Hartree)")
                    # Restore best positions and return penalty
                    self.positions.data.copy_(best_positions)
                    return torch.tensor(1e10, dtype=self.dtype, device=self.device)
                
                # Add penalty for distance constraint violations with adaptive scaling
                penalty = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                max_deviation = 0.0
                for (i, j), d_init in initial_distances.items():
                    d_current = torch.linalg.norm(self.positions[i] - self.positions[j])
                    deviation = torch.abs(d_current - d_init) / d_init
                    max_deviation = max(max_deviation, deviation.item())
                    if deviation > max_distance_deviation:
                        # Adaptive penalty: stronger penalty as deviation increases
                        penalty += 100.0 * (deviation - max_distance_deviation)**2
                
                total_loss = E + penalty
                
                # Check for NaN or inf in total loss before backward
                if not torch.isfinite(total_loss):
                    if disp:
                        print("Warning: Total loss is NaN or inf, returning large penalty")
                    self.positions.data.copy_(best_positions)
                    return torch.tensor(1e10, dtype=self.dtype, device=self.device)
                
                total_loss.backward()
                
                # Gradient clipping to prevent explosion
                if self.positions.grad is not None:
                    # Check for NaN or inf in gradients
                    if not torch.all(torch.isfinite(self.positions.grad)):
                        if disp:
                            print("Warning: Gradient contains NaN or inf")
                        self.positions.grad.zero_()
                        self.positions.data.copy_(best_positions)
                        return torch.tensor(1e10, dtype=self.dtype, device=self.device)
                    
                    # Clip gradients to prevent explosion
                    grad_norm = torch.linalg.norm(self.positions.grad)
                    max_grad_norm = 1.0  # Maximum allowed gradient norm in Hartree/Bohr
                    if grad_norm > max_grad_norm:
                        clip_factor = max_grad_norm / grad_norm
                        self.positions.grad *= clip_factor
                        if disp:
                            print(f"  Gradient clipped: {grad_norm:.2e} -> {max_grad_norm:.2e}")
                
                return total_loss
            except (RuntimeError, ValueError) as e:
                if disp:
                    print(f"Error during energy calculation: {e}")
                    print("Returning large penalty and restoring best positions")
                # Restore best positions on error
                self.positions.data.copy_(best_positions)
                return torch.tensor(1e10, dtype=self.dtype, device=self.device)
        
        # Optimization loop
        prev_loss = None
        for step in range(max_steps):
            try:
                if method.upper() == 'LBFGS':
                    loss = optimizer.step(closure)
                elif method.upper() == 'ADAM':
                    # For Adam, we need to manually call closure and step
                    loss = closure()
                    if loss.item() < 1e9:  # Only step if not a penalty value
                        optimizer.step()
                
                # Check for NaN or inf in loss
                if not torch.isfinite(loss):
                    consecutive_failures += 1
                    if disp:
                        print(f"Step {step}: Loss is NaN or inf")
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"Optimization stopped at step {step}: too many consecutive failures")
                        # Restore best positions
                        self.positions.data.copy_(best_positions)
                        break
                    continue
                
                loss_val = loss.item()
                
                # Update best state if this is better
                if loss_val < best_loss and loss_val < 1e9:  # Not a penalty value
                    best_loss = loss_val
                    best_positions = self.positions.detach().clone()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                # Check for loss explosion (sudden large increase)
                if prev_loss is not None and prev_loss < 1e9:  # Previous wasn't a penalty
                    loss_ratio = abs(loss_val / prev_loss) if prev_loss != 0 else 1.0
                    if loss_ratio > 100:  # Loss increased by more than 100x
                        print(f"Optimization stopped at step {step}: loss explosion detected")
                        print(f"  Previous loss: {prev_loss:.8f}")
                        print(f"  Current loss: {loss_val:.8f}")
                        # Restore best positions
                        self.positions.data.copy_(best_positions)
                        break
                
                # Check for unrealistic energy
                if loss_val < -1000.0:
                    print(f"Optimization stopped at step {step}: unrealistic energy detected")
                    print(f"  Current loss: {loss_val:.8f}")
                    # Restore best positions
                    self.positions.data.copy_(best_positions)
                    break
                
                prev_loss = loss_val
                
                # Check convergence
                if self.positions.grad is not None:
                    max_grad = torch.max(torch.abs(self.positions.grad))
                    if disp and step % 10 == 0:
                        print(f"Step {step}: Loss = {loss_val:.8f}, Max gradient = {max_grad.item():.6f}")
                    if max_grad < gtol and loss_val < 1e9:  # Converged and not a penalty
                        if disp:
                            print(f"Converged at step {step}")
                        break
                
                # Stop if too many consecutive non-improving steps
                if consecutive_failures >= max_consecutive_failures:
                    if disp:
                        print(f"Step {step}: No improvement for {max_consecutive_failures} steps")
                    # Restore best positions
                    self.positions.data.copy_(best_positions)
                    break
                    
            except (RuntimeError, ValueError) as e:
                consecutive_failures += 1
                if disp:
                    print(f"Optimization error at step {step}: {e}")
                # Restore best positions on error
                self.positions.data.copy_(best_positions)
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Optimization stopped at step {step} due to repeated errors")
                    break
        
        final_energy_dict = self.calculate_energies()
        print("\n--- Optimization Finished ---")
        print(f"Final Energy: {final_energy_dict['Total'].item():.8f} Hartree")
        print(f"Final Energy: {final_energy_dict['Total'].item() * HARTREE_TO_EV:.4f} eV")
        
        print(f"\nCovalent bonds identified: {len(bonded_pairs)}")
        print(f"Total atom pairs: {self.n_atoms * (self.n_atoms - 1) // 2}")
        
        # Report distance deviations for bonded pairs
        print("\nDistance Deviations (Bonded Pairs Only):")
        max_dev = 0.0
        for (i, j), d_init in initial_distances.items():
            d_final = torch.linalg.norm(self.positions[i] - self.positions[j])
            deviation = torch.abs(d_final - d_init) / d_init * 100.0
            if deviation > 1.0:
                print(f"  Atoms {i}-{j}: {deviation.item():.2f}%")
            max_dev = max(max_dev, deviation.item())
        print(f"Maximum deviation: {max_dev:.2f}%")
        
        return self.positions.detach() * BOHR_TO_ANGSTROM



def optimize_molecule(name, atoms, coords, params, device='cpu', dtype=torch.float64):
    """Helper function to optimize a molecule and print results."""
    print("\n" + "="*70)
    print(f"{name} (Geometry Optimization)")
    print("="*70)
    
    print("Initial Geometry (Angstrom):")
    for i, coord in enumerate(coords):
        symbol = params.element_params[atoms[i]]['symbol']
        print(f"  {symbol}  {coord[0]:10.6f}  {coord[1]:10.6f}  {coord[2]:10.6f}")
    
    calc = SQM1Calculator(atoms, coords, params=params, device=device, dtype=dtype)
    
    # Test automatic differentiation before optimization
    print("\n--- Testing Automatic Differentiation ---")
    test_coords = torch.tensor(coords, dtype=dtype, device=device) * ANGSTROM_TO_BOHR
    test_coords.requires_grad_(True)
    
    # Calculate energy and gradient using autograd
    energy, gradient = calc.calculate_energy_and_gradient(test_coords)
    
    print(f"Energy (Hartree): {energy.item():.8f}")
    print(f"Gradient norm (Hartree/Bohr): {torch.linalg.norm(gradient).item():.8f}")
    
    # Verify gradient using numerical differentiation
    numerical_grad = torch.zeros_like(test_coords)
    h = 1e-5  # Small step for numerical differentiation
    for i in range(test_coords.shape[0]):
        for j in range(3):
            # Create new tensor for positive step
            test_coords_plus = test_coords.detach().clone()
            test_coords_plus[i, j] = test_coords_plus[i, j] + h
            test_coords_plus.requires_grad_(False)
            e_plus = calc.calculate_total_energy(test_coords_plus)
            
            # Create new tensor for negative step
            test_coords_minus = test_coords.detach().clone()
            test_coords_minus[i, j] = test_coords_minus[i, j] - h
            test_coords_minus.requires_grad_(False)
            e_minus = calc.calculate_total_energy(test_coords_minus)
            
            numerical_grad[i, j] = (e_plus - e_minus) / (2 * h)
    
    grad_diff = torch.linalg.norm(gradient - numerical_grad)
    print(f"Gradient difference (AD vs Numerical): {grad_diff.item():.2e}")
    
    if grad_diff < 1e-6:
        print("✓ Automatic differentiation validated successfully!")
    else:
        print("⚠ Warning: Gradient difference is larger than expected")
    
    try:
        optimized_coords = calc.optimize_geometry()
        
        print("\nOptimized Geometry (Angstrom):")
        for i, coord in enumerate(optimized_coords):
            symbol = params.element_params[atoms[i]]['symbol']
            print(f"  {symbol}  {coord[0].item():10.6f}  {coord[1].item():10.6f}  {coord[2].item():10.6f}")
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        print("Calculating single-point energy instead...")
        energies = calc.calculate_energies()
        print(f"Total Energy: {energies['Total'].item():.8f} Hartree ({energies['Total'].item() * HARTREE_TO_EV:.4f} eV)")


def main():
    """Main function to run test cases."""
    # Parameters are now embedded in SQM1Parameters class
    params = SQM1Parameters()

    print("\n" + "="*70)
    print("SQM1 Implementation - Comprehensive Test Suite")
    print("Testing geometry optimization for various molecules")
    print("="*70)

    # --- Test Case 1: Water (H2O) ---
    water_atoms = [8, 1, 1]
    water_coords = [
        [0.000000,  0.000000,  0.117300],
        [0.000000,  0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200]
    ]
    optimize_molecule("Water (H2O)", water_atoms, water_coords, params)

    # --- Test Case 2: Ammonia (NH3) ---
    ammonia_atoms = [7, 1, 1, 1]
    ammonia_coords = [
        [ 0.0000,  0.0000,  0.1   ],
        [ 0.9400,  0.0000, -0.3   ],
        [-0.4700,  0.8141, -0.3   ],
        [-0.4700, -0.8141, -0.3   ]
    ]
    optimize_molecule("Ammonia (NH3)", ammonia_atoms, ammonia_coords, params)

    # --- Test Case 3: Methane (CH4) ---
    methane_atoms = [6, 1, 1, 1, 1]
    methane_coords = [
        [0.000,  0.000,  0.000],
        [0.629,  0.629,  0.629],
        [-0.629, -0.629,  0.629],
        [-0.629,  0.629, -0.629],
        [0.629, -0.629, -0.629]
    ]
    optimize_molecule("Methane (CH4)", methane_atoms, methane_coords, params)

    # --- Test Case 4: Ethanol (C2H5OH) ---
    ethanol_atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
    ethanol_coords = [
        [ 1.185,  -0.184,  0.000],  # C
        [-0.274,  0.095,  0.000],  # C
        [-1.048, -1.089,  0.000],  # O
        [ 1.554, -0.719,  0.886],  # H
        [ 1.554, -0.719, -0.886],  # H
        [ 1.621,  0.796,  0.000],  # H
        [-0.630,  0.646, -0.884],  # H
        [-0.630,  0.646,  0.884],  # H
        [-1.982, -0.861,  0.000],  # H
    ]
    optimize_molecule("Ethanol (C2H5OH)", ethanol_atoms, ethanol_coords, params)

    # --- Test Case 5: Glycine (C2H5NO2) ---
    glycine_atoms = [6, 6, 7, 8, 8, 1, 1, 1, 1, 1]
    glycine_coords = [
        [-0.714,  1.280,  0.000],  # C
        [ 0.000,  0.000,  0.000],  # C
        [ 1.420,  0.000,  0.000],  # N
        [-0.593, -1.116, -0.203],  # O
        [-0.276,  2.346,  0.173],  # O
        [-1.773,  1.244,  0.000],  # H
        [ 1.819, -0.003,  0.928],  # H
        [ 1.792,  0.790, -0.501],  # H
        [-1.573, -1.040, -0.203],  # H
        [-0.062,  2.356, -0.663],  # H (OH hydrogen)
    ]
    optimize_molecule("Glycine (C2H5NO2)", glycine_atoms, glycine_coords, params)

    # --- Test Case 6: Benzene (C6H6) ---
    benzene_atoms = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]
    r = 1.39  # C-C bond length in benzene
    a = 1.08  # C-H bond length
    benzene_coords = [
        [ r,      0.0,    0.0],  # C
        [ r/2,    r*0.866, 0.0],  # C
        [-r/2,    r*0.866, 0.0],  # C
        [-r,      0.0,    0.0],  # C
        [-r/2,   -r*0.866, 0.0],  # C
        [ r/2,   -r*0.866, 0.0],  # C
        [ (r+a),  0.0,    0.0],  # H
        [ (r+a)/2,  (r+a)*0.866, 0.0],  # H
        [-(r+a)/2,  (r+a)*0.866, 0.0],  # H
        [-(r+a),  0.0,    0.0],  # H
        [-(r+a)/2, -(r+a)*0.866, 0.0],  # H
        [ (r+a)/2, -(r+a)*0.866, 0.0],  # H
    ]
    optimize_molecule("Benzene (C6H6)", benzene_atoms, benzene_coords, params)

    # --- Test Case 7: Acetylene (C2H2) ---
    acetylene_atoms = [6, 6, 1, 1]
    acetylene_coords = [
        [0.000,  0.000,  0.600],  # C
        [0.000,  0.000, -0.600],  # C
        [0.000,  0.000,  1.665],  # H
        [0.000,  0.000, -1.665],  # H
    ]
    optimize_molecule("Acetylene (C2H2)", acetylene_atoms, acetylene_coords, params)

    # --- Test Case 8: Dichloromethane (CH2Cl2) ---
    dichloromethane_atoms = [6, 17, 17, 1, 1]
    dichloromethane_coords = [
        [0.000,  0.000,  0.000],  # C
        [0.000,  1.772,  0.000],  # Cl
        [0.000, -1.772,  0.000],  # Cl
        [1.030,  0.000,  0.000],  # H
        [-1.030,  0.000,  0.000],  # H
    ]
    optimize_molecule("Dichloromethane (CH2Cl2)", dichloromethane_atoms, dichloromethane_coords, params)

    # --- Test Case 9: Bromomethane (CH3Br) ---
    bromomethane_atoms = [6, 35, 1, 1, 1]
    bromomethane_coords = [
        [0.000,  0.000,  0.000],  # C
        [0.000,  0.000,  1.939],  # Br
        [1.025,  0.000, -0.377],  # H
        [-0.512,  0.887, -0.377],  # H
        [-0.512, -0.887, -0.377],  # H
    ]
    optimize_molecule("Bromomethane (CH3Br)", bromomethane_atoms, bromomethane_coords, params)

    # --- Test Case 10: Cyclohexane (C6H12) ---
    # Chair conformation
    cyclohexane_atoms = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cyclohexane_coords = [
        # C atoms in chair conformation
        [ 1.261,  0.728,  0.000],  # C1
        [ 1.261, -0.728,  0.000],  # C2
        [ 0.000, -1.457,  0.000],  # C3
        [-1.261, -0.728,  0.000],  # C4
        [-1.261,  0.728,  0.000],  # C5
        [ 0.000,  1.457,  0.000],  # C6
        # H atoms (axial and equatorial)
        [ 2.180,  1.259,  0.000],  # H (axial on C1)
        [ 1.261,  0.728,  0.990],  # H (equatorial on C1)
        [ 2.180, -1.259,  0.000],  # H (axial on C2)
        [ 1.261, -0.728,  0.990],  # H (equatorial on C2)
        [ 0.000, -2.518,  0.000],  # H (axial on C3)
        [ 0.000, -1.457,  0.990],  # H (equatorial on C3)
        [-2.180, -1.259,  0.000],  # H (axial on C4)
        [-1.261, -0.728,  0.990],  # H (equatorial on C4)
        [-2.180,  1.259,  0.000],  # H (axial on C5)
        [-1.261,  0.728,  0.990],  # H (equatorial on C5)
        [ 0.000,  2.518,  0.000],  # H (axial on C6)
        [ 0.000,  1.457,  0.990],  # H (equatorial on C6)
    ]
    optimize_molecule("Cyclohexane (C6H12)", cyclohexane_atoms, cyclohexane_coords, params)

    # --- Test Case 11: Bromobenzene (C6H5Br) ---
    r = 1.39  # C-C bond length
    a = 1.08  # C-H bond length
    bromobenzene_atoms = [6, 6, 6, 6, 6, 6, 35, 1, 1, 1, 1, 1]
    bromobenzene_coords = [
        [ r,      0.0,    0.0],  # C1
        [ r/2,    r*0.866, 0.0],  # C2
        [-r/2,    r*0.866, 0.0],  # C3
        [-r,      0.0,    0.0],  # C4
        [-r/2,   -r*0.866, 0.0],  # C5
        [ r/2,   -r*0.866, 0.0],  # C6
        [ (r+1.9),  0.0,    0.0],  # Br on C1
        [ (r+a)/2,  (r+a)*0.866, 0.0],  # H on C2
        [-(r+a)/2,  (r+a)*0.866, 0.0],  # H on C3
        [-(r+a),  0.0,    0.0],  # H on C4
        [-(r+a)/2, -(r+a)*0.866, 0.0],  # H on C5
        [ (r+a)/2, -(r+a)*0.866, 0.0],  # H on C6
    ]
    optimize_molecule("Bromobenzene (C6H5Br)", bromobenzene_atoms, bromobenzene_coords, params)

    # --- Test Case 12: 1,2-Dichloroethane (C2H4Cl2) ---
    dichloroethane_atoms = [6, 6, 17, 17, 1, 1, 1, 1]
    dichloroethane_coords = [
        [ 0.765,  0.000,  0.000],  # C1
        [-0.765,  0.000,  0.000],  # C2
        [ 1.265,  1.772,  0.000],  # Cl on C1
        [-1.265, -1.772,  0.000],  # Cl on C2
        [ 1.134,  0.000,  1.027],  # H on C1
        [ 1.134,  0.000, -1.027],  # H on C1
        [-1.134,  0.000,  1.027],  # H on C2
        [-1.134,  0.000, -1.027],  # H on C2
    ]
    optimize_molecule("1,2-Dichloroethane (C2H4Cl2)", dichloroethane_atoms, dichloroethane_coords, params)

    # --- Test Case 13: Acetone (C3H6O) ---
    acetone_atoms = [6, 6, 6, 8, 1, 1, 1, 1, 1, 1]
    acetone_coords = [
        [ 0.000,  0.000,  0.000],  # C (carbonyl)
        [ 1.520,  0.000,  0.000],  # C (methyl)
        [-1.520,  0.000,  0.000],  # C (methyl)
        [ 0.000,  0.000,  1.220],  # O (carbonyl)
        [ 1.900,  0.000,  1.027],  # H on C2
        [ 1.900,  0.887, -0.513],  # H on C2
        [ 1.900, -0.887, -0.513],  # H on C2
        [-1.900,  0.000,  1.027],  # H on C3
        [-1.900,  0.887, -0.513],  # H on C3
        [-1.900, -0.887, -0.513],  # H on C3
    ]
    optimize_molecule("Acetone (C3H6O)", acetone_atoms, acetone_coords, params)

    # --- Test Case 14: n-Butane (C4H10) ---
    nbutane_atoms = [6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    nbutane_coords = [
        [ 1.950,  0.000,  0.000],  # C1
        [ 0.650,  0.000,  0.000],  # C2
        [-0.650,  0.000,  0.000],  # C3
        [-1.950,  0.000,  0.000],  # C4
        [ 2.333,  0.000,  1.027],  # H on C1
        [ 2.333,  0.887, -0.513],  # H on C1
        [ 2.333, -0.887, -0.513],  # H on C1
        [ 0.650,  0.000,  1.090],  # H on C2
        [ 0.650,  0.890, -0.545],  # H on C2
        [-0.650,  0.000,  1.090],  # H on C3
        [-0.650, -0.890, -0.545],  # H on C3
        [-2.333,  0.000,  1.027],  # H on C4
        [-2.333,  0.887, -0.513],  # H on C4
        [-2.333, -0.887, -0.513],  # H on C4
    ]
    optimize_molecule("n-Butane (C4H10)", nbutane_atoms, nbutane_coords, params)

    print("\n" + "="*70)
    print("SQM1 Implementation Test Complete!")
    print("All test cases processed.")
    print("="*70)


if __name__ == "__main__":
    main()
