from multioptpy.Parameters.atomic_number import number_element, element_number
from multioptpy.Parameters.unit_values import UnitValueLib



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
