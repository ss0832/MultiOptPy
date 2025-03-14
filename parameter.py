
### General Nonbonded Force Field Parameters ###
def GNB_s_lib(element):
    #ref.: DOI: 10.1021/acs.jctc.4c01435
    if element is int:
        element = number_element(element)
    # C is C_3
    # N is N_3
    # The parameters of Lanthanides are same as La. (The parameters of Lanthanides except for La are not available in the paper.)
    GNB_s = {'H' : 0.2772, 'He': 0.2425,
             'Li': 0.3266, 'Be': 0.3964, 'B' : 0.3121, 'C' : 0.2455, 'N': 0.2743, 'O' : 0.2577, 'F' : 0.2791, 'Ne': 0.2378,
             'Na': 0.2592, 'Mg': 0.3837, 'Al': 0.3013, 'Si': 0.3202, 'P': 0.3399, 'S' : 0.3215, 'Cl': 0.3071, 'Ar': 0.2752,
             'K' : 0.2264, 'Ca': 0.4015, 'Sc': 0.9388, 'Ti': 0.6048, 'V': 0.3678, 'Cr': 0.2848, 'Mn': 0.3550, 'Fe': 0.4406, 'Co': 0.4124, 'Ni': 0.3139, 'Cu': 0.3345, 'Zn': 0.3391, 'Ga': 0.3146, 'Ge': 0.3327, 'As': 0.4040, 'Se': 0.3343, 'Br': 0.3242, 'Kr': 0.2920,
             'Rb': 0.2677, 'Sr': 0.3446, 'Y': 1.0038, 'Zr': 0.6146, 'Nb': 0.4244, 'Mo': 0.4882, 'Tc': 0.4439, 'Ru': 0.4736, 'Rh': 0.4039, 'Pd': 0.3188, 'Ag': 0.2948, 'Cd': 0.3012, 'In': 0.3351, 'Sn': 0.3134, 'Sb': 0.4081, 'Te': 0.3585, 'I': 0.3574, 'Xe': 0.3134,
             'Cs': 0.2872, 'Ba': 0.3788, 'La': 1.0851, 'Ce': 1.0851, 'Pr': 1.0851, 'Nd': 1.0851, 'Pm': 1.0851, 'Sm': 1.0851, 'Eu': 1.0851, 'Gd': 1.0851, 'Tb': 1.0851, 'Dy': 1.0851, 'Ho': 1.0851, 'Er': 1.0851, 'Tm': 1.0851, 'Yb': 1.0851, 'Lu': 1.0851, 'Hf': 0.5577, 'Ta': 0.5308, 'W': 0.3926, 'Re': 0.5351, 'Os': 0.4701, 'Ir': 0.3661, 'Pt': 0.3411, 'Au': 0.3164, 'Hg': 0.3047, 'Tl': 0.2802, 'Pb': 0.2682, 'Bi': 0.3876, 'Po': 0.3679, 'At': 0.3680, 'Rn': 0.3195, 'X': 1.000}#ang.
    
    return GNB_s[element] / UnitValueLib().bohr2angstroms #Bohr

def GNB_beta_lib(element):
    #ref.: DOI: 10.1021/acs.jctc.4c01435
    if element is int:
        element = number_element(element)
    # C is C_3
    # N is N_3
    # The parameters of Lanthanides are same as La. (The parameters of Lanthanides except for La are not available in the paper.)
    GNB_beta = {'H': 1.7504, 'He': 2.0870,
                'Li': 2.3073, 'Be': 2.8811, 'B': 2.6366, 'C': 3.0189, 'N': 2.9432, 'O': 2.7407, 'F': 2.5863, 'Ne': 2.5307,
                'Na': 2.5679, 'Mg': 2.9679, 'Al': 3.2377, 'Si': 3.2749, 'P': 3.7491, 'S': 3.6718, 'Cl': 3.4180, 'Ar': 3.3306,
                'K': 2.9736, 'Ca': 3.4464, 'Sc': 4.5955, 'Ti': 3.5334, 'V': 3.0680, 'Cr': 2.8393, 'Mn': 3.0550, 'Fe': 3.2628, 'Co': 3.2622, 'Ni': 2.7490, 'Cu': 2.9097, 'Zn': 2.9024, 'Ga': 3.3924, 'Ge': 3.2804, 'As': 3.8757, 'Se': 3.8562, 'Br': 3.7067, 'Kr': 3.6457,
                'Rb': 3.4421, 'Sr': 3.5590, 'Y': 5.7293, 'Zr': 5.0750, 'Nb': 3.7156, 'Mo': 3.6853, 'Tc': 3.6987, 'Ru': 3.7826, 'Rh': 3.4706, 'Pd': 3.3433, 'Ag': 3.1186, 'Cd': 3.0853, 'In': 3.5452, 'Sn': 3.2997, 'Sb': 4.1277, 'Te': 4.1795, 'I': 4.0988, 'Xe': 4.0387,
                'Cs': 3.8468, 'Ba': 4.0368, 'La': 6.1365, 'Ce': 6.1365, 'Pr': 6.1365, 'Nd': 6.1365, 'Pm': 6.1365, 'Sm': 6.1365, 'Eu': 6.1365, 'Gd': 6.1365, 'Tb': 6.1365, 'Dy': 6.1365, 'Ho': 6.1365, 'Er': 6.1365, 'Tm': 6.1365, 'Yb': 6.1365, 'Lu': 6.1365, 'Hf': 5.2130, 'Ta': 4.7637, 'W': 3.6018, 'Re': 3.9552, 'Os': 4.0278, 'Ir': 3.6123, 'Pt': 3.6066, 'Au': 3.3822, 'Hg': 3.3750, 'Tl': 3.3964, 'Pb': 3.2481, 'Bi': 4.1225, 'Po': 4.2313, 'At': 4.2455, 'Rn': 4.1784, 'X': 0.000}#ang.
    
    return GNB_beta[element] / UnitValueLib().bohr2angstroms #Bohr


def GNB_radii_lib(element):
    #ref.: DOI: 10.1021/acs.jctc.4c01435
    if element is int:
        element = number_element(element)
    # C is C_3
    # N is N_3
    # The parameters of Lanthanides are same as La. (The parameters of Lanthanides except for La are not available in the paper.) 
    GNB_radii = {'H': 3.6516, 'He': 2.1843,
             'Li': 1.2711, 'Be': 3.3497, 'B': 2.7079, 'C': 1.8219, 'N': 2.4667, 'O': 2.3650, 'F': 1.5062, 'Ne': 1.8233,
             'Na': 1.3974, 'Mg': 3.3515, 'Al': 3.0102, 'Si': 3.1629, 'P': 3.2554, 'S': 2.9539, 'Cl': 3.0368, 'Ar': 2.6598,
             'K': 4.0877, 'Ca': 4.1275, 'Sc': 9.7282, 'Ti': 8.5322, 'V': 7.2344, 'Cr': 5.3605, 'Mn': 3.7180, 'Fe': 3.6408, 'Co': 3.4961, 'Ni': 3.5108, 'Cu': 3.0537, 'Zn': 3.0261, 'Ga': 3.1735, 'Ge': 3.1773, 'As': 3.8357, 'Se': 3.1109, 'Br': 3.2122, 'Kr': 2.8263,
             'Rb': 2.4120, 'Sr': 1.8940, 'Y': 11.2061, 'Zr': 6.8210, 'Nb': 7.2367, 'Mo': 3.9010, 'Tc': 4.0857, 'Ru': 4.0450, 'Rh': 3.4813, 'Pd': 3.0487, 'Ag': 2.7795, 'Cd': 2.8673, 'In': 3.3339, 'Sn': 3.0086, 'Sb': 3.9919, 'Te': 3.4209, 'I': 3.5649, 'Xe': 3.0288,
             'Cs': 2.2620, 'Ba': 1.3837, 'La': 12.1710, 'Ce': 12.1710, 'Pr': 12.1710, 'Nd': 12.1710, 'Pm': 12.1710, 'Sm': 12.1710, 'Eu': 12.1710, 'Gd': 12.1710, 'Tb': 12.1710, 'Dy': 12.1710, 'Ho': 12.1710, 'Er': 12.1710, 'Tm': 12.1710, 'Yb': 12.1710, 'Lu': 12.1710, 'Hf': 6.0791, 'Ta': 5.7661, 'W': 3.6366, 'Re': 4.2410, 'Os': 4.1348, 'Ir': 3.4213, 'Pt': 3.2486, 'Au': 2.9588, 'Hg': 2.9381, 'Tl': 2.7711, 'Pb': 2.5816, 'Bi': 3.7850, 'Po': 3.5381, 'At': 3.6985, 'Rn': 3.0551, 'X': 0.000}#ang.
    
    return GNB_radii[element] / UnitValueLib().bohr2angstroms #Bohr

def GNB_C6_lib(element):
    #ref.: DOI: 10.1021/acs.jctc.4c01435
    if element is int:
        element = number_element(element)
    # C is C_3
    # N is N_3
    # The parameters of Lanthanides are same as La. (The parameters of Lanthanides except for La are not available in the paper.)   
    GNB_C6 = {'H': 95.99, 'He': 40.67,
             'Li': 70.21, 'Be': 114.51, 'B': 152.36, 'C': 184.28, 'N': 482.54, 'O': 405.57, 'F': 218.45, 'Ne': 174.81,
             'Na': 181.70, 'Mg': 263.02, 'Al': 228.10, 'Si': 359.43, 'P': 3222.12, 'S': 2144.49, 'Cl': 2072.46, 'Ar': 1357.42,
             'K': 1406.65, 'Ca': 1058.36, 'Sc': 11498.73, 'Ti': 3361.33, 'V': 2095.91, 'Cr': 1049.31, 'Mn': 966.27, 'Fe': 1571.36, 'Co': 1183.59, 'Ni': 787.76, 'Cu': 563.93, 'Zn': 592.91, 'Ga': 430.82, 'Ge': 812.57, 'As': 4533.53, 'Se': 3440.92, 'Br': 3859.82, 'Kr': 2729.60,
             'Rb': 1864.19, 'Sr': 1175.73, 'Y': 32141.18, 'Zr': 27655.14, 'Nb': 2864.20, 'Mo': 3563.45, 'Tc': 3266.43, 'Ru': 3967.23, 'Rh': 2233.82, 'Pd': 1393.49, 'Ag': 1315.09, 'Cd': 1311.47, 'In': 1460.56, 'Sn': 1662.99, 'Sb': 8089.97, 'Te': 6887.05, 'I': 8799.32, 'Xe': 6136.50,
             'Cs': 3757.31, 'Ba': 2561.18, 'La': 66580.83, 'Ce': 66580.83, 'Pr': 66580.83, 'Nd': 66580.83, 'Pm': 66580.83, 'Sm': 66580.83, 'Eu': 66580.83, 'Gd': 66580.83, 'Tb': 66580.83, 'Dy': 66580.83, 'Ho': 66580.83, 'Er': 66580.83, 'Tm': 66580.83, 'Yb': 66580.83, 'Lu': 66580.83, 'Hf': 27593.76, 'Ta': 15364.65, 'W': 2734.50, 'Re': 4801.82, 'Os': 5685.94, 'Ir': 2786.00, 'Pt': 2699.79, 'Au': 2282.60, 'Hg': 2476.79, 'Tl': 2988.70, 'Pb': 2506.63, 'Bi': 8916.84, 'Po': 8694.22, 'At': 11821.61, 'Rn': 8410.64, 'X': 0.000}#kcal ang^6 mol^-1
    
    return GNB_C6[element] / UnitValueLib().hartree2kcalmol / (UnitValueLib().bohr2angstroms) ** 6 #hartree bohr^6

def GNB_VDW_radii_lib(element):
    #ref.: DOI: 10.1021/acs.jctc.4c01435
    if element is int:
        element = number_element(element)
    # C is C_3
    # N is N_3
    # The parameters of Lanthanides are same as La. (The parameters of Lanthanides except for La are not available in the paper.)
    # Please check the SI of the paper for the details of the parameters.
    GNB_VDW_distance = {'H':3.2431,'He':3.0533,
                        'Li' : 3.6711 ,'Be': 5.3659, 'B': 3.9219,'C': 4.0516, 'N':3.6456,'O':3.3001, 'F': 3.2433,'Ne': 3.1416, 
                        'Na': 3.2429,'Mg': 4.8010 ,'Al':4.7457 ,'Si': 4.7121, 'P': 4.3825, 'S': 4.3735,'Cl':3.9557,'Ar': 3.8692,
                        'K': 3.8025 ,'Ca':5.0620 ,'Sc': 10.586 ,'Ti':7.7490 ,'V': 5.6617, 'Cr': 4.4761,'Mn': 4.1887, 'Fe': 4.4113,'Co':4.4575 ,'Ni': 3.6711,'Cu': 3.8716,'Zn': 3.8327,'Ga': 4.7820,'Ge': 4.3316,'As': 4.7036 ,'Se': 4.4826,'Br':4.1816,'Kr':4.1261,
                        'Rb': 3.8623,'Sr': 4.5095,'Y': 11.9894,'Zr': 7.1388,'Nb': 6.4121,'Mo': 4.757,'Tc': 4.8495,'Ru': 4.8882,'Rh':4.3388 ,'Pd': 4.061,'Ag':3.5832 ,'Cd': 3.5717,'In': 4.5002,'Sn': 3.8721,'Sb': 4.8066,'Te': 4.7337, 'I': 4.5014, 'Xe': 4.4360, 
                        'Cs': 4.2468,'Ba': 5.0441, 'La': 12.586, 'Ce': 12.586,'Pr': 12.586,'Nd': 12.586,'Pm': 12.586,'Sm': 12.586,'Eu': 12.586,'Gd': 12.586,'Tb': 12.586,'Dy': 12.586,'Ho': 12.586,'Er': 12.586,'Tm': 12.586,'Yb': 12.586,'Lu': 12.586,'Hf': 6.7740,
                        'Ta': 6.3793,'W': 4.4757,'Re': 5.2841,'Os': 5.0541,'Ir': 4.339,'Pt': 4.2436,'Au': 3.8280,'Hg': 3.7598,'Tl': 3.6437,'Pb': 3.4216,'Bi': 4.6308,'Po': 4.7192,'At': 4.6158,'Rn': 4.5115}
    return GNB_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr

def GNB_VDW_well_depth_lib(element):
    #ref.: DOI: 10.1021/acs.jctc.4c01435
    if element is int:
        element = number_element(element)    
    # C is C_3
    # N is N_3
    # The parameters of Lanthanides are same as La. (The parameters of Lanthanides except for La are not available in the paper.)
    # Please check the SI of the paper for the details of the parameters.    
    GNB_VDW_well_depth = {'H':0.0226, 'He':0.0257,
                          'Li':0.0133 ,'Be':0.0026 ,'B':0.0215,'C': 0.0264, 'N':0.1103, 'O':0.1624,'F':0.0908,'Ne':0.0985, 
                          'Na':0.0813, 'Mg':0.0110 ,'Al':0.0120,'Si': 0.0188, 'P':0.2342, 'S':0.1671, 'Cl':0.2754,  'Ar':0.2247,
                          'K':0.1573,'Ca':0.0307,'Sc':0.0034,'Ti': 0.0046,'V': 0.0110, 'Cr':0.0298, 'Mn':0.0791,'Fe': 0.0883,'Co':0.0673,'Ni':0.1293,'Cu':0.0786,'Zn':0.0862,'Ga':0.0211 ,'Ge':0.0640, 'As':0.1947,'Se':0.2280,'Br':0.3678,'Kr':0.3084,
                          'Rb':0.3220,'Sr':0.0756,'Y':0.0045 ,'Zr':0.0838,'Nb':0.0117,'Mo':0.1245,'Tc':0.1101,'Ru':0.1233 ,'Rh':0.1478,'Pd':0.1582,'Ag':0.3034,'Cd':0.2994,'In':0.0930,'Sn':0.2434,'Sb':0.3045,'Te':0.3227, 'I':0.5242,'Xe':0.4498, 
                          'Cs':0.3778,'Ba':0.0854 , 'La':0.0066 , 'Ce':0.0066 ,'Pr':0.0066 ,'Nd':0.0066 ,'Pm':0.0066 ,'Sm':0.0066 ,'Eu':0.0066 ,'Gd':0.0066 ,'Tb':0.0066 ,'Dy':0.0066 ,'Ho':0.0066 ,'Er':0.0066 ,'Tm':0.0066 ,'Yb':0.0066 ,'Lu':0.0066,'Hf':0.1267 ,
                          'Ta':0.0999 ,'W':0.1562 ,'Re':0.0906 ,'Os':0.1498,'Ir':0.1992,'Pt':0.2303,'Au':0.3535 ,'Hg':0.4313 ,'Tl':0.6563 ,'Pb':0.7952 ,'Bi':0.4271 ,'Po':0.4029 ,'At':0.6010 ,'Rn':0.5572, 'X':0.000}
                
    return GNB_VDW_well_depth[element] / UnitValueLib().hartree2kcalmol #hartree
    
### Universal Force Field Parameters ###
def UFF_bond_distance_lib(element):
    if element is int:
        element = number_element(element)
    UFF_bond_distance = {'H_':0.354, 'H_b':0.460, 'He':0.849,
                        'Li' : 1.336, 'Be': 1.074, 'B_3':0.838, 'B_2':0.828, 'C_3': 0.757, 'C_R': 0.729, 'C_2': 0.732, 'C_1': 0.706,###
                          'N':3.660,'O':3.500 , 'F':3.364,'Ne': 3.243, 
                        'Na':2.983,'Mg': 3.021 ,'Al':4.499 ,'Si': 4.295, 'P':4.147, 'S':4.035 ,'Cl':3.947,'Ar':3.868 ,
                        'K':3.812 ,'Ca':3.399 ,'Sc':3.295 ,'Ti':3.175 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 2.912,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':2.763 ,'Ga': 4.383,'Ge':4.280,'As':4.230 ,'Se':4.205,'Br':4.189,'Kr':4.141 ,
                        'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':2.963 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.463 ,'Sn':4.392 ,'Sb':4.420 ,'Te':4.470 , 'I':4.50, 'Xe':4.404 , 
                        'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,
                        'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_bond_distance[element] / UnitValueLib().bohr2angstroms#Bohr



def UFF_bondangle_lib(element):# not implemented
    if element is int:
        element = number_element(element)
    UFF_bondangle = {'H_':180.0, 'H_b':83.5, 'He':90.0,
                        'Li' : 180.0, 'Be': 109.47, 'B_3':109.47, 'B_2':120.0, 'C_3': 109.47, 'C_R': 120.0, 'C_2': 120.0, 'C_1': 180.0,###
                          'N':3.660,'O':3.500 , 'F':3.364,'Ne': 3.243, 
                        'Na':2.983,'Mg': 3.021 ,'Al':4.499 ,'Si': 4.295, 'P':4.147, 'S':4.035 ,'Cl':3.947,'Ar':3.868 ,
                        'K':3.812 ,'Ca':3.399 ,'Sc':3.295 ,'Ti':3.175 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 2.912,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':2.763 ,'Ga': 4.383,'Ge':4.280,'As':4.230 ,'Se':4.205,'Br':4.189,'Kr':4.141 ,
                        'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':2.963 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.463 ,'Sn':4.392 ,'Sb':4.420 ,'Te':4.470 , 'I':4.50, 'Xe':4.404 , 
                        'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,
                        'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_bondangle[element] * UnitValueLib().deg2rad #rad


def UFF_effective_charge_lib(element):
    if element is int:
        element = number_element(element)
    UFF_EC = {'H':0.712,'He': 0.098,
                        'Li' : 1.026 ,'Be': 1.565, 'B': 1.755,'C': 1.912, 'N': 2.544,'O': 2.300, 'F': 1.735,'Ne': 0.194, 
                        'Na': 1.081, 'Mg': 1.787,'Al': 1.792,'Si': 2.323, 'P': 2.863, 'S': 2.703,'Cl': 2.348,'Ar': 0.300,
                        'K': 1.165 ,'Ca': 2.141,'Sc': 2.592,'Ti': 2.659,'V': 2.679, 'Cr': 2.463,'Mn': 2.430, 'Fe': 2.430,'Co': 2.430 ,'Ni': 2.430 ,'Cu': 1.756 ,'Zn': 1.308,'Ga': 1.821, 'Ge': 2.789,'As': 2.864,'Se': 2.764,'Br': 2.519,'Kr': 0.452,
                        'Rb': 1.592,'Sr': 2.449,'Y': 3.257,'Zr': 3.667,'Nb': 3.618,'Mo': 3.400, 'Tc': 3.400,'Ru': 3.400,'Rh': 3.508, 'Pd': 3.210,'Ag': 1.956,'Cd': 1.650,'In': 2.070,'Sn': 2.961,'Sb': 2.704,'Te': 2.882, 'I': 2.650, 'Xe': 0.556, 
                        'Cs': 1.573,'Ba': 2.727, 'La': 3.300, 'Ce': 3.300,'Pr': 3.300,'Nd':3.300,'Pm':3.300,'Sm':3.300,'Eu':3.300,'Gd':3.300,'Tb':3.300,'Dy':3.300,'Ho': 3.416 ,'Er': 3.300,'Tm': 3.300,'Yb': 2.618,'Lu': 3.271,'Hf': 3.921,
                        'Ta': 4.075,'W': 3.70,'Re': 3.70,'Os': 3.70,'Ir': 3.731,'Pt': 3.382,'Au': 2.625,'Hg': 1.750,'Tl': 2.068,'Pb': 2.846,'Bi': 2.470,'Po': 2.330,'At': 2.240,'Rn': 0.583}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #charge
    return UFF_EC[element]

def UFF_VDW_distance_lib(element):
    if element is int:
        element = number_element(element)
    UFF_VDW_distance = {'H':2.886,'He':2.362 ,
                        'Li' : 2.451 ,'Be': 2.745, 'B':4.083 ,'C': 3.851, 'N':3.660,'O':3.500 , 'F':3.364,'Ne': 3.243, 
                        'Na':2.983,'Mg': 3.021 ,'Al':4.499 ,'Si': 4.295, 'P':4.147, 'S':4.035 ,'Cl':3.947,'Ar':3.868 ,
                        'K':3.812 ,'Ca':3.399 ,'Sc':3.295 ,'Ti':3.175 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 2.912,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':2.763 ,'Ga': 4.383,'Ge':4.280,'As':4.230 ,'Se':4.205,'Br':4.189,'Kr':4.141 ,
                        'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':2.963 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.463 ,'Sn':4.392 ,'Sb':4.420 ,'Te':4.470 , 'I':4.50, 'Xe':4.404 , 
                        'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,
                        'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr

def UFF_VDW_well_depth_lib(element):
    if element is int:
        element = number_element(element)         
    UFF_VDW_well_depth = {'H':0.0152, 'He':0.056 ,
                          'Li':0.025 ,'Be':0.085 ,'B':0.095,'C': 0.0951, 'N':0.0774, 'O':0.0957,'F':0.0725,'Ne':0.042 , 
                          'Na':0.50, 'Mg':0.111 ,'Al':0.31 ,'Si': 0.31, 'P':0.3200, 'S':0.3440, 'Cl':0.2833,  'Ar':0.185 ,
                          'K':0.035 ,'Ca':0.05 ,'Sc':0.019 ,'Ti':0.0550 ,'V':0.016 , 'Cr':0.015, 'Mn':0.013 ,'Fe': 0.0550,'Co':0.014 ,'Ni':0.015 ,'Cu':0.005 ,'Zn':0.055 ,'Ga':0.40 ,'Ge':0.40, 'As':0.41 ,'Se':0.43,'Br':0.37,'Kr':0.220 ,
                          'Rb':0.04 ,'Sr':0.235 ,'Y':0.072 ,'Zr':0.069 ,'Nb':0.059 ,'Mo':0.056 ,'Tc':0.048 ,'Ru':0.0500 ,'Rh':0.053 ,'Pd':0.048 ,'Ag':0.036 ,'Cd':0.228 ,'In':0.55 ,'Sn':0.55 ,'Sb':0.55 ,'Te':0.57 , 'I':0.51,'Xe':0.332 , 
                          'Cs':0.045 ,'Ba':0.364 , 'La':0.017 , 'Ce':0.013 ,'Pr':0.010 ,'Nd':0.010 ,'Pm':0.009 ,'Sm':0.008 ,'Eu':0.008 ,'Gd':0.009 ,'Tb':0.007 ,'Dy':0.007 ,'Ho':0.007 ,'Er':0.007 ,'Tm':0.006 ,'Yb':0.228 ,'Lu':0.041 ,'Hf':0.072 ,
                          'Ta':0.081 ,'W':0.067 ,'Re':0.066 ,'Os':0.037 ,'Ir':0.073 ,'Pt':0.080 ,'Au':0.039 ,'Hg':0.385 ,'Tl':0.680 ,'Pb':0.663 ,'Bi':0.518 ,'Po':0.325 ,'At':0.284 ,'Rn':0.248, 'X':0.010}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 # kcal/mol
                
    return UFF_VDW_well_depth[element] / UnitValueLib().hartree2kcalmol #hartree

### D2 Dispersion Model Parameters ###
def D2_VDW_radii_lib(element):
    # VDW radii for D2 dispersion model
    if element is int:
        element = number_element(element)
    """
    D2_VDW_distance = {'H':0.91,'He':0.92,
                        'Li' : 0.75 ,'Be': 1.28, 'B':1.35 ,'C': 1.32, 'N': 1.27,'O': 1.22, 'F':1.17,'Ne': 1.13, 
                        'Na':1.04,'Mg': 1.24 ,'Al':1.49 ,'Si': 1.56, 'P':1.55, 'S':1.53 ,'Cl':1.49,'Ar':1.45 ,
                        'K': 1.35,'Ca':1.34 ,'Sc':1.42 ,'Ti':1.42 ,'V': 1.42, 'Cr':1.42 ,'Mn': 1.42, 'Fe': 1.42,'Co':1.42 ,'Ni':1.42 ,'Cu':1.42 ,'Zn':1.42 ,'Ga': 1.50,'Ge':1.57,'As':1.60 ,'Se':1.61,'Br':1.59,'Kr':1.57 ,
                        'Rb':1.48 ,'Sr': 1.46,'Y':1.49 ,'Zr':1.49 ,'Nb':1.49 ,'Mo':1.49,'Tc':1.49 ,'Ru':1.49 ,'Rh':1.49 ,'Pd':1.49 ,'Ag':1.49 ,'Cd':1.49 ,'In':1.52 ,'Sn':1.64 ,'Sb':1.71 ,'Te':1.72 , 'I':1.72, 'Xe':1.71 , 
                        'Cs':2.00 ,'Ba':2.00 , 'La':2.00 , 'Ce':2.00 ,'Pr':2.00 ,'Nd':2.00 ,'Pm':2.00 ,'Sm':2.00 ,'Eu':2.00 ,'Gd':2.00 ,'Tb':2.00 ,'Dy':2.00 ,'Ho':2.00 ,'Er':2.00 ,'Tm':2.00 ,'Yb':2.00,'Lu':2.00 ,'Hf': 2.00,
                        'Ta':2.00 ,'W':2.00 ,'Re':2.00 ,'Os':2.00 ,'Ir':2.00 ,'Pt':2.00 ,'Au':2.00 ,'Hg':2.00 ,'Tl':2.00 ,'Pb':2.00 ,'Bi':2.00 ,'Po':2.00 ,'At':2.00 ,'Rn': 2.00}
    """
    D2_VDW_distance = {
    'H': 1.001, 'He': 1.012,
    'Li': 0.825, 'Be': 1.408, 'B': 1.485, 'C': 1.452, 'N': 1.397, 'O': 1.342, 'F': 1.287, 'Ne': 1.243,
    'Na': 1.144, 'Mg': 1.364, 'Al': 1.639, 'Si': 1.716, 'P': 1.705, 'S': 1.683, 'Cl': 1.639, 'Ar': 1.595,
    'K': 1.485, 'Ca': 1.474, 'Sc': 1.562, 'Ti': 1.562, 'V': 1.562, 'Cr': 1.562, 'Mn': 1.562, 'Fe': 1.562, 
    'Co': 1.562, 'Ni': 1.562, 'Cu': 1.562, 'Zn': 1.562, 'Ga': 1.650, 'Ge': 1.727, 'As': 1.760, 'Se': 1.771, 
    'Br': 1.749, 'Kr': 1.727, 'Rb': 1.628, 'Sr': 1.606, 'Y': 1.639, 'Zr': 1.639, 'Nb': 1.639, 'Mo': 1.639, 
    'Tc': 1.639, 'Ru': 1.639, 'Rh': 1.639, 'Pd': 1.639, 'Ag': 1.639, 'Cd': 1.639, 'In': 1.672, 'Sn': 1.804, 
    'Sb': 1.881, 'Te': 1.892, 'I': 1.892, 'Xe': 1.881, 'Cs': 1.802, 'Ba': 1.762, 'La': 1.720, 'Ce': 1.753, 
    'Pr': 1.753, 'Nd': 1.753, 'Pm': 1.753, 'Sm': 1.753, 'Eu': 1.753, 'Gd': 1.753, 'Tb': 1.753, 'Dy': 1.753, 
    'Ho': 1.753, 'Er': 1.753, 'Tm': 1.753, 'Yb': 1.753, 'Lu': 1.753, 'Hf': 1.788, 'Ta': 1.772, 'W': 1.772, 
    'Re': 1.772, 'Os': 1.772, 'Ir': 1.772, 'Pt': 1.772, 'Au': 1.772, 'Hg': 1.758, 'Tl': 1.989, 'Pb': 1.944, 
    'Bi': 1.898, 'Po': 2.005, 'At': 1.991, 'Rn': 1.924
    }   
    return D2_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr

def D2_C6_coeff_lib(element):
    if element is int:
        element = number_element(element)
    C6_coefficients = {'H': 0.14, 'He': 0.08,
                   'Li': 1.61, 'Be': 1.61, 'B': 3.13, 'C': 1.75, 'N': 1.23, 'O': 0.70, 'F': 0.75, 'Ne': 0.63,
                   'Na': 5.71, 'Mg': 5.71, 'Al': 10.79, 'Si': 9.23, 'P': 7.84, 'S': 5.57, 'Cl': 5.07, 'Ar': 4.61,
                   'K': 10.80, 'Ca': 10.80,
                   'Sc': 10.80, 'Ti': 10.80, 'V': 10.80, 'Cr': 10.80, 'Mn': 10.80,
                   'Fe': 10.80, 'Co': 10.80, 'Ni': 10.80, 'Cu': 10.80, 'Zn': 10.80,
                   'Ga': 16.99, 'Ge': 17.10, 'As': 16.37, 'Se': 12.64, 'Br': 12.47, 'Kr': 12.01,
                   'Rb': 24.67, 'Sr': 24.67,
                   'Y': 24.67, 'Zr': 24.67, 'Nb': 24.67, 'Mo': 24.67, 'Tc': 24.67,
                   'Ru': 24.67, 'Rh': 24.67, 'Pd': 24.67, 'Ag': 24.67, 'Cd': 24.67,
                   'In': 37.32, 'Sn': 38.71, 'Sb': 38.44, 'Te': 31.74, 'I': 31.50, 'Xe': 29.99,
                   'Cs': 50.00, 'Ba': 50.00,
                   'La': 50.00, 'Ce': 50.00, 'Pr': 50.00, 'Nd': 50.00, 'Pm': 50.00, 'Sm': 50.00,
                   'Eu': 50.00, 'Gd': 50.00, 'Tb': 50.00, 'Dy': 50.00, 'Ho': 50.00, 'Er': 50.00,
                   'Tm': 50.00, 'Yb': 50.00, 'Lu': 50.00, 
                   'Hf': 50.00, 'Ta': 50.00, 'W': 50.00, 'Re': 50.00, 'Os': 50.00, 
                   'Ir': 50.00, 'Pt': 50.00, 'Au': 50.00, 'Hg': 50.00,
                   'Tl': 50.00, 'Pb': 50.00, 'Bi': 50.00, 'Po': 50.00, 'At': 50.00, 'Rn': 50.00}# J nm^6 mol^-1
    
    # J nm^6 mol^-1 -> hartree bohr^6
    param = C6_coefficients[element] * 10 ** 6 / (UnitValueLib().bohr2angstroms) ** 6 / UnitValueLib().hartree2j / UnitValueLib().mol2au
    
    return param#hartree bohr^6

D2_S6_parameter = 1.00 #dimensionless


class GFNFFParameters:
    """Parameters for GFNFF-based approximation Hessian"""
    def __init__(self):
        # Unit conversion
        self.bohr2ang = UnitValueLib().bohr2angstroms  # 0.529177
        self.kcalmol2hartree = 1.0 / UnitValueLib().hartree2kcalmol  # 1/627.5095
        
        # Reference atomic CN
        self.ref_cn = {
            'H': 1.0, 'C': 4.0, 'N': 3.0, 'O': 2.0, 'F': 1.0, 'Si': 4.0, 'P': 3.0, 
            'S': 2.0, 'Cl': 1.0, 'Br': 1.0, 'I': 1.0
        }
        
        # Element radii and vdW parameters (from gfnff_param.f90)
        # Format: [r_cov, r_vdw, en, alpha]
        self.element_params = {
            'H':  [0.32, 1.09, 2.20, 4.5],
            'He': [0.46, 1.3,  0.00, 1.0],
            'Li': [1.29, 1.80, 0.98, 164.0],
            'Be': [0.99, 1.53, 1.57, 38.0],
            'B':  [0.84, 1.92, 2.04, 21.0],
            'C':  [0.75, 1.70, 2.55, 12.0],
            'N':  [0.71, 1.55, 3.04, 7.4],
            'O':  [0.64, 1.52, 3.44, 5.4],
            'F':  [0.60, 1.47, 3.98, 3.8],
            'Ne': [0.67, 1.54, 0.00, 2.67],
            'Na': [1.60, 2.27, 0.93, 163.0],
            'Mg': [1.40, 1.73, 1.31, 71.0],
            'Al': [1.24, 1.84, 1.61, 60.0],
            'Si': [1.14, 2.10, 1.90, 37.0],
            'P':  [1.09, 1.80, 2.19, 25.0],
            'S':  [1.04, 1.80, 2.58, 19.6],
            'Cl': [0.99, 1.75, 3.16, 15.0],
            'Ar': [0.96, 1.88, 0.00, 11.1],
            'K':  [2.00, 2.75, 0.82, 293.0],
            'Ca': [1.70, 2.31, 1.00, 161.0],
            'Sc': [1.44, 2.15, 1.36, 120.0],
            'Ti': [1.32, 2.11, 1.54, 98.0],
            'V':  [1.22, 2.07, 1.63, 84.0],
            'Cr': [1.18, 2.06, 1.66, 78.0],
            'Mn': [1.17, 2.05, 1.55, 63.0],
            'Fe': [1.17, 2.00, 1.83, 56.0],
            'Co': [1.16, 2.00, 1.88, 50.0],
            'Ni': [1.15, 1.97, 1.91, 48.0],
            'Cu': [1.17, 1.96, 1.90, 42.0],
            'Zn': [1.25, 2.01, 1.65, 40.0],
            'Ga': [1.25, 1.87, 1.81, 60.0],
            'Ge': [1.21, 2.11, 2.01, 41.0],
            'As': [1.21, 1.85, 2.18, 29.0],
            'Se': [1.17, 1.90, 2.55, 25.0],
            'Br': [1.14, 1.83, 2.96, 20.0],
            'Kr': [1.17, 2.02, 0.00, 16.8],
            'Rb': [2.15, 3.03, 0.82, 320.0],
            'Sr': [1.90, 2.49, 0.95, 199.0],
            'Y':  [1.62, 2.40, 1.22, 126.7],
            'Zr': [1.45, 2.23, 1.33, 119.97],
            'Nb': [1.34, 2.18, 1.60, 101.603],
            'Mo': [1.30, 2.17, 2.16, 88.425],
            'Tc': [1.27, 2.16, 1.90, 80.083],
            'Ru': [1.25, 2.13, 2.20, 65.895],
            'Rh': [1.25, 2.10, 2.28, 56.1],
            'Pd': [1.28, 2.10, 2.20, 23.68],
            'Ag': [1.34, 2.11, 1.93, 50.6],
            'Cd': [1.48, 2.18, 1.69, 39.7],
            'In': [1.44, 1.93, 1.78, 70.2],
            'Sn': [1.40, 2.17, 1.96, 55.0],
            'Sb': [1.40, 2.06, 2.05, 43.7],
            'Te': [1.37, 2.06, 2.10, 37.65],
            'I':  [1.33, 1.98, 2.66, 35.0],
            'Xe': [1.31, 2.16, 0.00, 27.3],
            'Cs': [2.38, 3.43, 0.79, 400.0],
            'Ba': [2.00, 2.68, 0.89, 280.0],
            'La': [1.80, 2.40, 1.10, 215.0],
            'Ce': [1.65, 2.35, 1.12, 210.0],
            'Pr': [1.65, 2.35, 1.13, 205.0],
            'Nd': [1.64, 2.35, 1.14, 200.0],
            'Pm': [1.63, 2.35, 1.13, 200.0],
            'Sm': [1.62, 2.35, 1.17, 180.0],
            'Eu': [1.85, 2.35, 1.20, 180.0],
            'Gd': [1.61, 2.35, 1.20, 180.0],
            'Tb': [1.59, 2.35, 1.20, 180.0],
            'Dy': [1.59, 2.35, 1.22, 180.0],
            'Ho': [1.58, 2.35, 1.23, 180.0],
            'Er': [1.57, 2.35, 1.24, 180.0],
            'Tm': [1.56, 2.35, 1.25, 180.0],
            'Yb': [1.70, 2.35, 1.10, 180.0],
            'Lu': [1.56, 2.35, 1.27, 180.0],
            'Hf': [1.44, 2.23, 1.30, 110.0],
            'Ta': [1.34, 2.22, 1.50, 100.0],
            'W':  [1.30, 2.18, 2.36, 90.0],
            'Re': [1.28, 2.16, 1.90, 80.0],
            'Os': [1.26, 2.16, 2.20, 70.0],
            'Ir': [1.27, 2.13, 2.20, 60.0],
            'Pt': [1.30, 2.13, 2.28, 50.0],
            'Au': [1.34, 2.14, 2.54, 40.0],
            'Hg': [1.49, 2.23, 2.00, 35.0],
            'Tl': [1.48, 2.09, 2.04, 70.0],
            'Pb': [1.47, 2.02, 2.33, 55.0],
            'Bi': [1.46, 2.00, 2.02, 50.0],
            'Po': [1.46, 2.00, 2.00, 45.0],
            'At': [1.45, 2.00, 2.20, 40.0],
            'Rn': [1.43, 2.00, 0.00, 35.0]
        }

        # Bond parameters - scaling factors as in original GFNFF
        self.bond_scaling = 0.4195  # kcal/mol to hartree
        self.bond_decay = 0.10
        
        # Reference bond lengths and force constants for selected bonds
        # Values are taken from GFNFF parameters (bondkonst array in gfnff_param.f90)
        # Format: [r0 (bohr), kb (au)]
        self.bond_params = {
            ('C', 'C'): [2.8464, 0.3601],
            ('C', 'H'): [2.0697, 0.3430],
            ('C', 'N'): [2.7394, 0.3300],
            ('C', 'O'): [2.6794, 0.3250],
            ('C', 'F'): [2.5646, 0.4195],
            ('C', 'S'): [3.3926, 0.2050],
            ('C', 'Cl'): [3.2740, 0.2150],
            ('C', 'Br'): [3.5260, 0.1800],
            ('C', 'I'): [3.8467, 0.1600],
            ('N', 'H'): [1.9079, 0.4150],
            ('N', 'N'): [2.5363, 0.2660],
            ('N', 'O'): [2.6379, 0.2800],
            ('N', 'F'): [2.5155, 0.2950],
            ('O', 'H'): [1.8200, 0.4770],
            ('O', 'O'): [2.7358, 0.1525],
            ('O', 'S'): [3.1786, 0.2270],
            ('S', 'H'): [2.5239, 0.2750],
            ('S', 'S'): [3.6599, 0.1375],
            ('S', 'F'): [3.0108, 0.2200],
            ('S', 'Cl'): [3.4798, 0.1625]
        }

        # Angle bend parameters
        # Values from benkonst array in gfnff_param.f90
        # Format: [theta0 (degrees), ka (au)]
        self.angle_params = {
            ('C', 'C', 'C'): [112.7, 0.0800],
            ('C', 'C', 'H'): [110.7, 0.0590],
            ('C', 'C', 'N'): [111.0, 0.0740],
            ('C', 'C', 'O'): [109.5, 0.0950],
            ('H', 'C', 'H'): [109.5, 0.0400],
            ('H', 'C', 'N'): [109.5, 0.0670],
            ('H', 'C', 'O'): [109.5, 0.0580],
            ('N', 'C', 'N'): [109.5, 0.0700],
            ('N', 'C', 'O'): [110.5, 0.0750],
            ('O', 'C', 'O'): [109.5, 0.0990],
            ('C', 'N', 'C'): [109.5, 0.0680],
            ('C', 'N', 'H'): [109.5, 0.0560],
            ('H', 'N', 'H'): [106.4, 0.0450],
            ('C', 'O', 'C'): [111.0, 0.0880],
            ('C', 'O', 'H'): [107.0, 0.0980],
            ('H', 'O', 'H'): [104.5, 0.0550],
            ('C', 'S', 'C'): [96.0, 0.0850],
            ('C', 'S', 'H'): [96.0, 0.0680],
            ('H', 'S', 'H'): [93.0, 0.0380]
        }

        # Torsion parameters
        # Values from torskonst array in gfnff_param.f90
        # Format: [V1, V2, V3] (kcal/mol, converted to hartree in the getter)
        self.torsion_params = {
            ('C', 'C', 'C', 'C'): [0.20, 0.25, 0.18],
            ('C', 'C', 'C', 'H'): [0.00, 0.00, 0.30],
            ('C', 'C', 'C', 'N'): [0.10, 0.40, 0.70],
            ('C', 'C', 'C', 'O'): [-0.55, 0.10, 0.50],
            ('H', 'C', 'C', 'H'): [0.00, 0.00, 0.30],
            ('H', 'C', 'C', 'N'): [0.00, 0.00, 0.40],
            ('H', 'C', 'C', 'O'): [0.00, 0.00, 0.35],
            ('N', 'C', 'C', 'N'): [-0.60, -0.10, 0.50],
            ('N', 'C', 'C', 'O'): [0.50, 0.45, 0.00],
            ('O', 'C', 'C', 'O'): [-0.55, -0.10, 0.00],
            ('C', 'C', 'N', 'C'): [-0.54, -0.10, 0.32],
            ('C', 'C', 'N', 'H'): [0.00, 0.00, 0.30],
            ('H', 'C', 'N', 'C'): [0.00, 0.00, 0.40],
            ('H', 'C', 'N', 'H'): [0.00, 0.00, 0.30],
            ('C', 'C', 'O', 'C'): [0.65, -0.25, 0.67],
            ('C', 'C', 'O', 'H'): [0.00, 0.00, 0.45],
            ('H', 'C', 'O', 'C'): [0.00, 0.00, 0.45],
            ('H', 'C', 'O', 'H'): [0.00, 0.00, 0.27]
        }
        
        # Hydrogen bond parameters
        # Based on hbtyppar in gfnff_param.f90
        # Format: [r0 (Å), k (kcal/mol)]
        self.hbond_params = {
            ('O', 'H', 'N'): [1.9, 4.0],
            ('O', 'H', 'O'): [1.8, 4.0],
            ('N', 'H', 'N'): [2.0, 4.0],
            ('N', 'H', 'O'): [1.9, 4.0],
            ('F', 'H', 'N'): [1.8, 3.5],
            ('F', 'H', 'O'): [1.7, 3.5],
            ('S', 'H', 'N'): [2.5, 3.5],
            ('S', 'H', 'O'): [2.4, 3.5],
            ('Cl', 'H', 'N'): [2.3, 3.0],
            ('Cl', 'H', 'O'): [2.2, 3.0],
            ('Br', 'H', 'N'): [2.5, 2.5],
            ('Br', 'H', 'O'): [2.4, 2.5],
            ('I', 'H', 'N'): [2.7, 2.0],
            ('I', 'H', 'O'): [2.6, 2.0]
        }

        # Dispersion parameters
        self.d4_s6 = 1.0
        self.d4_s8 = 1.03683
        self.d4_s9 = 1.0
        self.d4_a1 = 0.4171
        self.d4_a2 = 4.5337

        # Default parameters for missing entries
        self.default_bond_k = 0.3000
        self.default_angle_k = 0.0700
        self.default_torsion_v = [0.0, 0.0, 0.2]
        
    def get_vdw_radius(self, element):
        """Get van der Waals radius for an element (in Angstrom)"""
        if element in self.element_params:
            return self.element_params[element][1]
        return 2.0  # Default value
    
    def get_cov_radius(self, element):
        """Get covalent radius for an element (in Angstrom)"""
        if element in self.element_params:
            return self.element_params[element][0]
        return 1.0  # Default value
    
    def get_electronegativity(self, element):
        """Get electronegativity for an element"""
        if element in self.element_params:
            return self.element_params[element][2]
        return 2.0  # Default value
    
    def get_polarizability(self, element):
        """Get polarizability for an element (in a.u.)"""
        if element in self.element_params:
            return self.element_params[element][3]
        return 10.0  # Default value
    
    def get_bond_params(self, element1, element2):
        """Get bond parameters for a given pair of elements"""
        key = tuple(sorted([element1, element2]))
        if key in self.bond_params:
            # Return [r0 (bohr), kb (au)]
            return self.bond_params[key]
        
        # Estimate based on covalent radii if not explicitly defined
        r_cov1 = self.get_cov_radius(element1)
        r_cov2 = self.get_cov_radius(element2)
        r0 = (r_cov1 + r_cov2) / self.bohr2ang  # Convert Å to bohr
        
        return [r0, self.default_bond_k]
    
    def get_angle_params(self, element1, element2, element3):
        """Get angle parameters for a given triplet of elements"""
        key = (element1, element2, element3)
        if key in self.angle_params:
            return self.angle_params[key]
        
        # Reverse order
        key_rev = (element3, element2, element1)
        if key_rev in self.angle_params:
            return self.angle_params[key_rev]
        
        # Default angle based on element2's expected coordination
        if element2 in ['C', 'Si']:
            theta0 = 109.5  # tetrahedral
        elif element2 in ['N', 'P']:
            theta0 = 107.0  # slightly reduced tetrahedral
        elif element2 in ['O', 'S']:
            theta0 = 104.5  # bent
        else:
            theta0 = 120.0  # default
        
        return [theta0, self.default_angle_k]
    
    def get_torsion_params(self, element1, element2, element3, element4):
        """Get torsion parameters for a given quartet of elements"""
        key = (element1, element2, element3, element4)
        if key in self.torsion_params:
            # Convert kcal/mol to hartree
            v1, v2, v3 = self.torsion_params[key]
            return [v1 * self.kcalmol2hartree, 
                    v2 * self.kcalmol2hartree, 
                    v3 * self.kcalmol2hartree]
        
        # Reverse order
        key_rev = (element4, element3, element2, element1)
        if key_rev in self.torsion_params:
            v1, v2, v3 = self.torsion_params[key_rev]
            return [v1 * self.kcalmol2hartree, 
                    v2 * self.kcalmol2hartree, 
                    v3 * self.kcalmol2hartree]
        
        # Default parameters
        return [v * self.kcalmol2hartree for v in self.default_torsion_v]
    
    def get_hbond_params(self, donor, h, acceptor):
        """Get hydrogen bond parameters for a donor-H-acceptor triplet"""
        key = (donor, h, acceptor)
        if key in self.hbond_params:
            r0, k = self.hbond_params[key]
            return [r0 / self.bohr2ang, k * self.kcalmol2hartree]  # Convert to bohr, hartree
        
        # Try reverse order (some H-bonds can be bidirectional)
        key_rev = (acceptor, h, donor)
        if key_rev in self.hbond_params:
            r0, k = self.hbond_params[key_rev]
            return [r0 / self.bohr2ang, k * self.kcalmol2hartree]
        
        # Default weak hydrogen bond parameters
        return [2.0 / self.bohr2ang, 2.0 * self.kcalmol2hartree]


class GFN0Parameters:
    """GFN0-xTB model parameters based on official implementation"""
    def __init__(self):
        # Unit conversion
        self.bohr2ang = UnitValueLib().bohr2angstroms
        self.kcalmol2hartree = 1.0 / UnitValueLib().hartree2kcalmol
        
        # --- Atomic parameters from gfn0_param.f90 ---
        
        # Atomic radii (Bohr)
        self.rad = {
            'H': 0.75, 'He': 0.75, 'Li': 1.23, 'Be': 1.01, 'B': 0.90, 'C': 0.85, 'N': 0.84,
            'O': 0.83, 'F': 0.83, 'Ne': 0.75, 'Na': 1.60, 'Mg': 1.40, 'Al': 1.25, 'Si': 1.14,
            'P': 1.09, 'S': 1.04, 'Cl': 1.00, 'Ar': 0.75, 'K': 1.90, 'Ca': 1.71, 'Sc': 1.48,
            'Ti': 1.36, 'V': 1.34, 'Cr': 1.22, 'Mn': 1.19, 'Fe': 1.17, 'Co': 1.16, 'Ni': 1.15,
            'Cu': 1.14, 'Zn': 1.23, 'Ga': 1.25, 'Ge': 1.21, 'As': 1.16, 'Se': 1.14, 'Br': 1.12,
            'Kr': 0.75, 'Rb': 2.06, 'Sr': 1.85, 'Y': 1.61, 'Zr': 1.48, 'Nb': 1.37, 'Mo': 1.31,
            'Tc': 1.23, 'Ru': 1.24, 'Rh': 1.24, 'Pd': 1.19, 'Ag': 1.26, 'Cd': 1.36, 'In': 1.47,
            'Sn': 1.40, 'Sb': 1.39, 'Te': 1.35, 'I': 1.33, 'Xe': 0.75
        }
        
        # Electronegativity parameters (Mulliken EN)
        self.en = {
            'H': 2.20, 'He': 0.00, 'Li': 0.97, 'Be': 1.47, 'B': 2.01, 'C': 2.50, 'N': 3.07,
            'O': 3.50, 'F': 4.10, 'Ne': 0.00, 'Na': 1.01, 'Mg': 1.23, 'Al': 1.47, 'Si': 1.74,
            'P': 2.06, 'S': 2.44, 'Cl': 2.83, 'Ar': 0.00, 'K': 0.91, 'Ca': 1.04, 'Sc': 1.20,
            'Ti': 1.32, 'V': 1.45, 'Cr': 1.56, 'Mn': 1.60, 'Fe': 1.64, 'Co': 1.70, 'Ni': 1.75,
            'Cu': 1.75, 'Zn': 1.66, 'Ga': 1.82, 'Ge': 2.02, 'As': 2.20, 'Se': 2.48, 'Br': 2.74,
            'Kr': 0.00, 'Rb': 0.89, 'Sr': 0.99, 'Y': 1.11, 'Zr': 1.22, 'Nb': 1.23, 'Mo': 1.30,
            'Tc': 1.36, 'Ru': 1.42, 'Rh': 1.45, 'Pd': 1.35, 'Ag': 1.42, 'Cd': 1.46, 'In': 1.49,
            'Sn': 1.72, 'Sb': 1.82, 'Te': 2.01, 'I': 2.21, 'Xe': 0.00
        }
        
        # Charge scaling and interaction parameters
        self.kCN = 2.0  # Charge-scaling exponent
        self.shellPoly = 1.5  # Shell-charge polynomial
        
        # GFN0 specific bond parameters
        # Reference bond orders and stretching constants
        self.referenceBondLength = {
            ('C', 'C'): 1.53, ('C', 'N'): 1.42, ('C', 'O'): 1.42, ('C', 'H'): 1.10,
            ('N', 'N'): 1.41, ('N', 'O'): 1.40, ('N', 'H'): 1.03,
            ('O', 'O'): 1.45, ('O', 'H'): 0.98,
            ('H', 'H'): 0.80,
            # Special bonds for cyano groups 
            ('C', 'N', 'triple'): 1.16, # C≡N triple bond
            ('C', 'C', 'triple'): 1.20, # C≡C triple bond
            ('C', 'O', 'double'): 1.25, # C=O double bond
            ('C', 'N', 'double'): 1.29, # C=N double bond
        }
        
        # Force constant scaling factors for different bond types
        self.bondForceFactors = {
            'single': 1.0,
            'aromatic': 1.2, 
            'double': 1.5,
            'triple': 2.0
        }
        
        # Base force constants in mDyne/Å (converted to atomic units)
        self.kStretchBase = 0.35
        
        # Angle parameters - natural angles in radians
        self.naturalAngles = {
            'sp3': 109.5 * np.pi/180,  # Tetrahedral
            'sp2': 120.0 * np.pi/180,   # Trigonal planar
            'sp': 180.0 * np.pi/180     # Linear
        }
        
        # Base angle force constants (mDyne·Å/rad²)
        self.kAngleBase = 0.07
        
        # Base torsion barriers (kcal/mol)
        self.V2Base = 0.1 * self.kcalmol2hartree
        self.V3Base = 0.01 * self.kcalmol2hartree
        
        # Special parameters for CN triple bonds
        self.CNParams = {
            'kStretch': 0.9,   # Stronger force constant for CN triple bond
            'kBend': 0.15,     # Force constant for X-C≡N bending
            'kTorsion': 0.002  # Very weak torsional barrier
        }
    
    def get_radius(self, element):
        """Get atomic radius for an element (in Bohr)"""
        if element in self.rad:
            return self.rad[element]
        return 1.0  # Default radius
    
    def get_en(self, element):
        """Get electronegativity for an element"""
        if element in self.en:
            return self.en[element]
        return 2.0  # Default EN
    
    def get_bond_length(self, element1, element2, bond_type='single'):
        """Get reference bond length for a given pair of elements and bond type"""
        key = tuple(sorted([element1, element2]))
        if bond_type != 'single':
            key_with_type = key + (bond_type,)
            if key_with_type in self.referenceBondLength:
                return self.referenceBondLength[key_with_type]
        
        if key in self.referenceBondLength:
            return self.referenceBondLength[key]
        
        # If not found, estimate from radii
        r1 = self.get_radius(element1)
        r2 = self.get_radius(element2)
        bond_length = r1 + r2
        
        # Adjust for bond type
        if bond_type == 'double':
            bond_length *= 0.85
        elif bond_type == 'triple':
            bond_length *= 0.78
        elif bond_type == 'aromatic':
            bond_length *= 0.90
            
        return bond_length
    
    def get_bond_force_constant(self, element1, element2, bond_type='single'):
        """Get bond force constant for a pair of elements and bond type"""
        # Special case for CN triple bond
        if ((element1 == 'C' and element2 == 'N') or 
            (element1 == 'N' and element2 == 'C')) and bond_type == 'triple':
            return self.CNParams['kStretch']
        
        # Regular case - scale by bond type
        factor = self.bondForceFactors.get(bond_type, 1.0)
        return self.kStretchBase * factor


class D4Parameters:
    """Parameters class for D4 dispersion correction"""
    def __init__(self, s6=1.0, s8=1.03683, s9=1.0, a1=0.4171, a2=4.5337):
        # Default parameters for PBE0/def2-QZVP
        self.s6 = s6    # Scaling constant for C6 term (typically 1.0)
        self.s8 = s8    # Scaling constant for C8 term
        self.s9 = s9    # Scaling constant for three-body term
        self.a1 = a1    # Parameter for damping function of C6 term
        self.a2 = a2    # Parameter for damping function of C8 term
        
        # Charge scaling constants
        self.ga = 3.0   # Charge scaling factor
        self.gc = 2.0   # Three-body charge scaling factor
        
        # Reference polarizabilities based on PBE0/def2-QZVP calculations
        self.ref_polarizabilities = {
            'H': 4.50, 'He': 1.38, 
            'Li': 164.20, 'Be': 38.40, 'B': 21.10, 'C': 12.00, 'N': 7.40, 'O': 5.40, 'F': 3.80, 'Ne': 2.67,
            'Na': 162.70, 'Mg': 71.00, 'Al': 57.80, 'Si': 37.00, 'P': 25.00, 'S': 19.60, 'Cl': 15.00, 'Ar': 11.10,
            'K': 292.80, 'Ca': 160.80, 'Sc': 120.00, 'Ti': 98.00, 'V': 84.00, 'Cr': 72.00, 'Mn': 63.00, 'Fe': 56.00,
            'Co': 50.00, 'Ni': 44.00, 'Cu': 42.00, 'Zn': 40.00, 'Ga': 60.00, 'Ge': 41.00, 'As': 29.00, 'Se': 25.00,
            'Br': 20.00, 'Kr': 16.80, 'Rb': 320.20, 'Sr': 199.30, 'Y': 126.70, 'Zr': 119.97, 'Nb': 101.60, 
            'Mo': 88.42, 'Tc': 80.08, 'Ru': 65.89, 'Rh': 56.10, 'Pd': 23.68, 'Ag': 46.00, 'Cd': 39.72,
            'In': 70.22, 'Sn': 55.95, 'Sb': 43.67, 'Te': 37.65, 'I': 35.00, 'Xe': 27.30
        }
        
        # r4/r2 values from tad-dftd3 library
        self.r4r2_values = {
            # H, He
            'H': 8.0589, 'He': 3.4698,
            # Li-Ne
            'Li': 29.0974, 'Be': 14.8517, 'B': 11.8799, 'C': 7.8715, 'N': 5.5588, 
            'O': 4.7566, 'F': 3.8025, 'Ne': 3.1036,
            # Na-Ar
            'Na': 26.1552, 'Mg': 17.2304, 'Al': 17.7210, 'Si': 12.7442, 'P': 9.5361, 
            'S': 8.1652, 'Cl': 6.7463, 'Ar': 5.6004,
            # K, Ca
            'K': 29.2012, 'Ca': 22.3934,
            # Sc-Zn
            'Sc': 19.0598, 'Ti': 16.8590, 'V': 15.4023, 'Cr': 12.5589, 'Mn': 13.4788,
            'Fe': 12.2309, 'Co': 11.2809, 'Ni': 10.5569, 'Cu': 10.1428, 'Zn': 9.4907,
            # Ga-Kr
            'Ga': 13.4606, 'Ge': 10.8544, 'As': 8.9386, 'Se': 8.1350, 'Br': 7.1251, 'Kr': 6.1971,
            # Rb, Sr
            'Rb': 30.0162, 'Sr': 24.4103,
            # Y-Cd
            'Y': 20.3537, 'Zr': 17.4780, 'Nb': 13.5528, 'Mo': 11.8451, 'Tc': 11.0355,
            'Ru': 10.1997, 'Rh': 9.5414, 'Pd': 9.0061, 'Ag': 8.6417, 'Cd': 8.9975,
            # In-Xe
            'In': 14.0834, 'Sn': 11.8333, 'Sb': 10.0179, 'Te': 9.3844, 'I': 8.4110, 'Xe': 7.5152,
            # Cs, Ba
            'Cs': 32.7622, 'Ba': 27.5708
        }
        
        # Electronegativity values used for charge scaling
        self.electronegativity = {
            'H': 2.20, 'He': 0.00,
            'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.00,
            'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.00,
            'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83,
            'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55,
            'Br': 2.96, 'Kr': 0.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
            'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
            'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 0.00, 'Cs': 0.79, 'Ba': 0.89
        }

        # Reference coordination numbers for different hybridization states
        self.ref_cn = {
            'H': [0.0, 1.0],
            'C': [0.0, 2.0, 3.0, 4.0],
            'N': [0.0, 1.0, 2.0, 3.0],
            'O': [0.0, 1.0, 2.0],
            'P': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'S': [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
        }

        # Default values for unlisted elements
        self.default_r4r2 = 10.0
        self.default_polarizability = 20.0
        self.default_electronegativity = 2.0
        
    def get_r4r2(self, element):
        """Get r^4/r^2 ratio for each element"""
        if element in self.r4r2_values:
            return self.r4r2_values[element]
        return self.default_r4r2
    
    def get_polarizability(self, element):
        """Get reference polarizability for an element"""
        if element in self.ref_polarizabilities:
            return self.ref_polarizabilities[element]
        return self.default_polarizability
    
    def get_electronegativity(self, element):
        """Get electronegativity for an element"""
        if element in self.electronegativity:
            return self.electronegativity[element]
        return self.default_electronegativity

class D3Parameters:
    """Parameters class for D3 dispersion correction"""
    def __init__(self, s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407):
        # Default parameters for PBE0
        self.s6 = s6  # Scaling constant for C6 term (typically 1.0)
        self.s8 = s8  # Scaling constant for C8 term
        self.a1 = a1  # Parameter for damping function of C6 term
        self.a2 = a2  # Parameter for damping function of C8 term
        
        # r4_over_r2 values from tad-dftd3 library
        # PBE0/def2-QZVP atomic values calculated by S. Grimme in Gaussian (2010)
        # with updates for rare gases and super-heavy elements
        self.r4r2_values = {
            # None
            # H, He
            'H': 8.0589, 'He': 3.4698,
            # Li-Ne
            'Li': 29.0974, 'Be': 14.8517, 'B': 11.8799, 'C': 7.8715, 'N': 5.5588, 
            'O': 4.7566, 'F': 3.8025, 'Ne': 3.1036,
            # Na-Ar
            'Na': 26.1552, 'Mg': 17.2304, 'Al': 17.7210, 'Si': 12.7442, 'P': 9.5361, 
            'S': 8.1652, 'Cl': 6.7463, 'Ar': 5.6004,
            # K, Ca
            'K': 29.2012, 'Ca': 22.3934,
            # Sc-Zn
            'Sc': 19.0598, 'Ti': 16.8590, 'V': 15.4023, 'Cr': 12.5589, 'Mn': 13.4788,
            'Fe': 12.2309, 'Co': 11.2809, 'Ni': 10.5569, 'Cu': 10.1428, 'Zn': 9.4907,
            # Ga-Kr
            'Ga': 13.4606, 'Ge': 10.8544, 'As': 8.9386, 'Se': 8.1350, 'Br': 7.1251, 'Kr': 6.1971,
            # Rb, Sr
            'Rb': 30.0162, 'Sr': 24.4103,
            # Y-Cd
            'Y': 20.3537, 'Zr': 17.4780, 'Nb': 13.5528, 'Mo': 11.8451, 'Tc': 11.0355,
            'Ru': 10.1997, 'Rh': 9.5414, 'Pd': 9.0061, 'Ag': 8.6417, 'Cd': 8.9975,
            # In-Xe
            'In': 14.0834, 'Sn': 11.8333, 'Sb': 10.0179, 'Te': 9.3844, 'I': 8.4110, 'Xe': 7.5152,
            # Cs, Ba
            'Cs': 32.7622, 'Ba': 27.5708,
            # La-Eu
            'La': 23.1671, 'Ce': 21.6003, 'Pr': 20.9615, 'Nd': 20.4562, 'Pm': 20.1010, 
            'Sm': 19.7475, 'Eu': 19.4828,
            # Gd-Yb
            'Gd': 15.6013, 'Tb': 19.2362, 'Dy': 17.4717, 'Ho': 17.8321, 'Er': 17.4237, 
            'Tm': 17.1954, 'Yb': 17.1631,
            # Lu-Hg
            'Lu': 14.5716, 'Hf': 15.8758, 'Ta': 13.8989, 'W': 12.4834, 'Re': 11.4421,
            'Os': 10.2671, 'Ir': 8.3549, 'Pt': 7.8496, 'Au': 7.3278, 'Hg': 7.4820,
            # Tl-Rn
            'Tl': 13.5124, 'Pb': 11.6554, 'Bi': 10.0959, 'Po': 9.7340, 'At': 8.8584, 'Rn': 8.0125
        }
        
        # Default value for unlisted elements
        self.default_r4r2 = 10.0
        
    def get_r4r2(self, element):
        """Get r^4/r^2 ratio for each element"""
        if element in self.r4r2_values:
            return self.r4r2_values[element]
        return self.default_r4r2


def DREIDING_VDW_distance_lib(element):#https://doi.org/10.1021/j100389a010 
    #Atoms for which no parameters exist use UFF parameters
    if element is int:
        element = number_element(element)
    UFF_VDW_distance = {'H':3.195,'He':2.362 ,
                        'Li' : 2.451 ,'Be': 2.745, 'B':4.02 ,'C': 3.8983, 'N':3.6621,'O':3.4046 , 'F':3.4720,'Ne': 3.243, 
                        'Na':3.1440,'Mg': 3.021 ,'Al':4.39 ,'Si': 4.27, 'P':4.1500, 'S':4.0300 ,'Cl':3.9503,'Ar':3.868 ,
                        'K':3.812 ,'Ca':3.472 ,'Sc':3.295 ,'Ti':4.5400 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 4.5400,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':4.54 ,'Ga': 4.39,'Ge':4.27,'As':4.15 ,'Se':4.03,'Br':3.95,'Kr':4.141 ,
                        'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':4.5400 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.59 ,'Sn':4.47 ,'Sb':4.35 ,'Te':4.23 , 'I':4.15, 'Xe':4.404 , 
                        'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,
                        'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr


def DREIDING_VDW_well_depth_lib(element):
    #Atoms for which no parameters exist use UFF parameters
    if element is int:
        element = number_element(element)         
    UFF_VDW_well_depth = {'H':0.044, 'He':0.056 ,
                          'Li':0.025 ,'Be':0.085 ,'B':0.180,'C': 0.105, 'N':0.069, 'O':0.060,'F':0.050,'Ne':0.042 , 
                          'Na':0.030, 'Mg':0.111 ,'Al':0.505 ,'Si': 0.402, 'P':0.305, 'S':0.274, 'Cl':0.227,  'Ar':0.185 ,
                          'K':0.035 ,'Ca':0.238 ,'Sc':0.019 ,'Ti':0.017 ,'V':0.016 , 'Cr':0.015, 'Mn':0.013 ,'Fe': 0.013,'Co':0.014 ,'Ni':0.015 ,'Cu':0.005 ,'Zn':0.124 ,'Ga':0.415 ,'Ge':0.379, 'As':0.309 ,'Se':0.291,'Br':0.251,'Kr':0.220 ,
                          'Rb':0.04 ,'Sr':0.235 ,'Y':0.072 ,'Zr':0.069 ,'Nb':0.059 ,'Mo':0.056 ,'Tc':0.048 ,'Ru':0.056 ,'Rh':0.053 ,'Pd':0.048 ,'Ag':0.036 ,'Cd':0.228 ,'In':0.599 ,'Sn':0.567 ,'Sb':0.449 ,'Te':0.398 , 'I':0.339,'Xe':0.332 , 
                          'Cs':0.045 ,'Ba':0.364 , 'La':0.017 , 'Ce':0.013 ,'Pr':0.010 ,'Nd':0.010 ,'Pm':0.009 ,'Sm':0.008 ,'Eu':0.008 ,'Gd':0.009 ,'Tb':0.007 ,'Dy':0.007 ,'Ho':0.007 ,'Er':0.007 ,'Tm':0.006 ,'Yb':0.228 ,'Lu':0.041 ,'Hf':0.072 ,
                          'Ta':0.081 ,'W':0.067 ,'Re':0.066 ,'Os':0.037 ,'Ir':0.073 ,'Pt':0.080 ,'Au':0.039 ,'Hg':0.385 ,'Tl':0.680 ,'Pb':0.663 ,'Bi':0.518 ,'Po':0.325 ,'At':0.284 ,'Rn':0.248, 'X':0.010}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 # kcal/mol
                
    return UFF_VDW_well_depth[element] / UnitValueLib().hartree2kcalmol #hartree


def covalent_radii_lib(element):#single bond
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, 
           "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, 
           "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, 
           "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,
           "Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42, 'X':1.000}#ang.
    # ref. Pekka Pyykkö; Michiko Atsumi (2009). “Molecular single-bond covalent radii for elements 1 - 118”. Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. (H...Rn)
            
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr

def double_covalent_radii_lib(element):#double bond
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.24, "Be": 0.90, "B": 0.78, "C": 0.67, "N": 0.60, "O": 0.57, "F": 0.59, "Ne": 0.96, 
           "Na": 1.60, "Mg": 1.32, "Al":1.13, "Si": 1.07, "P": 1.02, "S": 0.94, "Cl": 0.95, "Ar": 1.07, 
           "K": 1.93, "Ca": 1.47, "Sc": 1.16, "Ti": 1.17, "V": 1.12, "Cr": 1.11, "Mn": 1.05, "Fe": 1.09, "Co": 1.03, "Ni": 1.01, "Cu": 1.15, "Zn": 1.20, "Ga": 1.17, "Ge": 1.11, "As": 1.14, "Se": 1.07, "Br": 1.09, "Kr": 1.21, 
           "Rb": 2.02, "Sr": 1.57, "Y": 1.30, "Zr": 1.27,"Nb": 1.25,"Mo": 1.21,"Tc": 1.20,"Ru": 1.14,"Rh": 1.10,"Pd": 1.17,"Ag": 1.39,"Cd": 1.44,"In": 1.36,"Sn": 1.30,"Sb": 1.33,"Te": 1.28,"I": 1.29,"Xe": 1.35,
           "Cs": 2.09,"Ba": 1.61,"La":1.39,"Ce": 1.37,"Pr": 1.38,"Nd": 1.37,"Pm": 1.35,"Sm": 1.34,"Eu": 1.34,"Gd": 1.35 ,"Tb": 1.35,"Dy": 1.33,"Ho": 1.33,"Er": 1.33,"Tm": 1.31,"Yb": 1.29,"Lu": 1.31,"Hf": 1.28,"Ta": 1.26,"W": 1.20,"Re": 1.19,"Os": 1.16,"Ir": 1.15,"Pt": 1.12,"Au": 1.21,"Hg": 1.42,"Tl": 1.42,"Pb":1.35,"Bi":1.41,"Po":1.35,"At":1.38,"Rn":1.45, 'X':1.000}#ang.
    # ref. P. Pyykkö; M. Atsumi (2009). "Molecular Double-Bond Covalent Radii for Elements Li–E112". Chemistry: A European Journal. 15 (46): 12770–12779. doi:10.1002/chem.200901472.  (H...Rn)
    #If double bond length is not available, single bond length is used. (H, He)
            
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr

def triple_covalent_radii_lib(element):#triple bond
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.24, "Be": 0.85, "B": 0.73, "C": 0.60, "N": 0.54, "O": 0.53, "F": 0.53, "Ne": 0.96, 
            "Na": 1.60, "Mg": 1.27, "Al":1.11, "Si": 1.02, "P": 0.94, "S": 0.95, "Cl": 0.93, "Ar": 0.96, 
            "K": 1.93, "Ca": 1.33, "Sc": 1.14, "Ti": 1.08, "V": 1.06, "Cr": 1.03, "Mn": 1.03, "Fe": 1.02, "Co": 0.96, "Ni": 1.01, "Cu": 1.20, "Zn": 1.20, "Ga": 1.21, "Ge": 1.21, "As": 1.06, "Se": 1.07, "Br": 1.10, "Kr": 1.08, 
           "Rb": 2.02, "Sr": 1.39, "Y": 1.24, "Zr": 1.21,"Nb": 1.16,"Mo": 1.13,"Tc": 1.10,"Ru": 1.03,"Rh": 1.06,"Pd": 1.12,"Ag": 1.37,"Cd": 1.44,"In": 1.46,"Sn": 1.32,"Sb": 1.27,"Te": 1.21,"I": 1.25,"Xe": 1.22,
          "Cs": 2.09,"Ba": 1.49,"La":1.39,"Ce": 1.31,"Pr": 1.28,"Nd": 1.37,"Pm": 1.35,"Sm": 1.34,"Eu": 1.34,"Gd": 1.32 ,"Tb": 1.35,"Dy": 1.33,"Ho": 1.33,"Er": 1.33,"Tm": 1.31,"Yb": 1.29,"Lu": 1.31,"Hf": 1.21,"Ta": 1.19,"W": 1.15,"Re": 1.10,"Os": 1.09,"Ir": 1.07,"Pt": 1.10,"Au": 1.23,"Hg": 1.42,"Tl": 1.50,"Pb":1.37,"Bi":1.35,"Po":1.29,"At":1.38,"Rn":1.33, 'X':1.000}#ang.
    # ref. P. Pyykkö; S. Riedel; M. Patzschke (2005). "Triple-Bond Covalent Radii". Chemistry: A European Journal. 11 (12): 3511–3520. doi:10.1002/chem.200401299. (H...Rn)
    #If there is no triple bond length, use the double bond length; if there is no double bond length, use the single bond length. (H, He, Li, Ne, Na, K, Zn, Rb, Cd, Cs, Nd, Pm, Sm, Eu, Tb, Dy, Ho, Er, Tm, Yb, Hg)
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr

def element_number(elem):
    num = {"H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, 
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,"In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53,"Xe": 54,
        "Cs": 55 ,"Ba": 56, "La": 57,"Ce":58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,"Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,"Hf": 72,"Ta": 73,"W": 74,"Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,"Tl": 81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86}
        
    return num[elem]

def number_element(num):
    elem = {1: "H",  2:"He",
         3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
        11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
        19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr",
        37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",43:"Tc",44:"Ru",45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe",
        55:"Cs", 56:"Ba", 57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy" ,67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn"}
        
    return elem[num]

def atomic_mass(elem):
    
    if type(elem) is int:
        elem_num = elem
    else:
        
        elem_num = element_number(elem)
    
    mass = {1: 1.00782503223, 2: 4.00260325413,
    3: 7.0160034366, 4: 9.012183065, 5: 11.00930536, 6: 12.0, 7: 14.00307400443, 8: 15.99491461957, 9: 18.99840316273, 10: 19.9924401762,
    11: 22.989769282, 12: 23.985041697, 13: 26.98153853, 14: 27.97692653465, 15: 30.97376199842, 16: 31.9720711744, 17: 34.968852682, 18: 39.9623831237,
    19: 38.9637064864, 20: 39.962590863, 21: 44.95590828, 22: 47.94794198, 23: 50.94395704, 24: 51.94050623, 25: 54.93804391, 26: 55.93493633, 27: 58.93319429, 28: 57.93534241, 29: 62.92959772, 30: 63.92914201, 31: 68.9255735, 32: 73.921177761, 33: 74.92159457, 34: 79.9165218, 35: 78.9183376, 36: 83.9114977282,
    37: 84.9117897379, 38: 87.9056125, 39: 88.9058403, 40: 89.9046977, 41: 92.906373, 42: 97.90540482, 43: 96.9063667, 44: 101.9043441, 45: 102.905498, 46: 105.9034804, 47: 106.9050916, 48: 113.90336509, 49: 114.903878776, 50: 119.90220163, 51: 120.903812, 52: 129.906222748, 53: 126.9044719, 54: 131.9041550856,
    55: 132.905451961, 56: 137.905247, 57: 138.9063563, 58: 139.9054431, 59: 140.9076576, 60: 141.907729, 61: 144.9127559, 62: 151.9197397, 63: 152.921238, 64: 157.9241123, 65: 158.9253547, 66: 163.9291819, 67: 164.9303288,
    68: 165.9302995, 69: 168.9342179, 70: 173.9388664, 71: 174.9407752, 72: 179.946557, 73: 180.9479958, 74: 183.95093092, 75: 186.9557501, 76: 191.961477, 77: 192.9629216, 78: 194.9647917, 79: 196.96656879, 80: 201.9706434, 81: 204.9744278,
    82: 207.9766525, 83: 208.9803991, 84: 208.9824308, 85: 209.9871479, 86: 222.0175782}# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    return mass[int(elem_num)]




#grimme_C6Parameter_list = [0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63, 5.71, 5.71, 10.79, 9.23, 7.84, 5.57, 5.07, 4.61, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 16.99, 17.10, 16.37, 12.64, 12.47, 12.01, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 37.32, 38.71, 38.44, 31.74, 31.50, 29.99, 29.99, 315.275, 226.994, 176.252, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 105.112, 81.24, 81.24, 81.24, 81.24, 81.24, 81.24, 81.24, 57.364, 57.254, 63.162, 63.540, 55.283, 57.171, 56.64]
#grimme_vdw_list = [1.001, 1.012, 0.825, 1.408, 1.485, 1.452, 1.397, 1.342, 1.287, 1.243, 1.144, 1.364, 1.639, 1.716, 1.705, 1.683, 1.639, 1.595, 1.485, 1.474, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.650, 1.727, 1.760, 1.771, 1.749, 1.727, 1.628, 1.606, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.672, 1.804, 1.881, 1.892, 1.892, 1.881, 1.881, 1.802, 1.762, 1.720, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753,  1.753, 1.753, 1.753, 1.753, 1.788, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.758, 1.989, 1.944, 1.898, 2.005, 1.991, 1.924]



class UnitValueLib: 
    def __init__(self):
        self.hartree2kcalmol = 627.509 #
        self.bohr2angstroms = 0.52917721067 #
        self.hartree2kjmol = 2625.500 #
        self.hartree2eV = 27.211396127707
        self.amu2kg = 1.66053906660 * 10 ** (-27)
        self.au2kg = 9.1093837015 * 10 ** (-31)
        self.hartree2j =  4.3597447222071 * 10 ** (-18) 
        self.bohr2m = 5.29177210903 * 10 ** (-11)
        self.mol2au = 6.02214076 * 10 ** 23
        self.deg2rad = 0.017453292519943295
        self.au2sec = 2.418884326505 * 10 ** (-17)
        return
        