




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
