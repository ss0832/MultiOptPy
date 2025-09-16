



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
