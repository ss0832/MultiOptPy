from multioptpy.Parameters.unit_values import UnitValueLib


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
