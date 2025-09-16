import numpy as np

from multioptpy.Parameters.atomic_number import number_element, element_number
from multioptpy.Parameters.unit_values import UnitValueLib


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
