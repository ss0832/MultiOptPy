from multioptpy.Parameters.atomic_number import number_element, element_number
from multioptpy.Parameters.unit_values import UnitValueLib



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

