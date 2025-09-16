from multioptpy.Parameters.atomic_number import number_element, element_number
from multioptpy.Parameters.unit_values import UnitValueLib



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