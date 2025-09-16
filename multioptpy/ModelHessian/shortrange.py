import numpy as np
from scipy.special import erf
from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.Utils.calc_tools import Calculationtools


class ShortRangeCorrectionHessian:
    """
    Class for calculating short-range correction to model Hessians, excluding bonded atom pairs.
    
    This class computes the second derivatives of the short-range part of
    the Coulomb operator used in range-separated hybrid functionals (e.g., ωB97X-D).
    The short-range part is defined as (1-erf(ω*r))/r, where ω is the range-separation parameter.
    
    References:
    [1] J.-D. Chai and M. Head-Gordon, J. Chem. Phys., 2008, 128, 084106 (ωB97X)
    [2] J.-D. Chai and M. Head-Gordon, Phys. Chem. Chem. Phys., 2008, 10, 6615 (ωB97X-D)
    """
    def __init__(self, omega=0.2, cx_sr=0.78, scaling_factor=0.5):
        """Initialize the ShortRangeCorrectionHessian class.
        
        Parameters:
        -----------
        omega : float
            Range-separation parameter in Bohr^-1 (default: 0.2 for ωB97X-D)
        cx_sr : float
            Short-range DFT exchange coefficient (default: 0.78 for ωB97X-D)
        scaling_factor : float
            Overall scaling factor for the correction (default: 0.5)
        """
        self.omega = omega                # Range-separation parameter (Bohr^-1)
        self.cx_sr = cx_sr                # Short-range exchange coefficient
        self.scaling_factor = scaling_factor  # Overall scaling factor
        self.sr_cutoff = 15.0             # Cutoff distance for short-range interactions (Bohr)
        
    def detect_bonds(self, coord, element_list):
        """Detect bonded atom pairs in the molecule.
        
        Parameters:
        -----------
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
            
        Returns:
        --------
        set
            Set of tuples (i,j) representing bonded atom pairs
        """
        # Use BondConnectivity class from MultiOptPy to detect bonds
        bc = BondConnectivity()
        bond_matrix = bc.bond_connect_matrix(element_list, coord)
        
        # Create a set of bonded atom pairs
        bonded_pairs = set()
        for i in range(len(coord)):
            for j in range(i+1, len(coord)):
                if bond_matrix[i, j] == 1:
                    bonded_pairs.add((i, j))
                    bonded_pairs.add((j, i))
        
        return bonded_pairs
        
    def sr_coulomb(self, r):
        """Calculate short-range Coulomb potential.
        
        V_SR(r) = (1 - erf(ω*r)) / r
        
        Parameters:
        -----------
        r : float
            Distance between two atoms (Bohr)
            
        Returns:
        --------
        float
            Short-range Coulomb potential
        """
        if r < 1e-10:
            # Use limit as r→0 (Taylor expansion)
            return 2 * self.omega / np.sqrt(np.pi)
        return (1.0 - erf(self.omega * r)) / r
    
    def sr_coulomb_first_derivative(self, r):
        """Calculate first derivative of short-range Coulomb potential.
        
        dV_SR(r)/dr = -V_SR(r)/r - 2ω/√π * exp(-ω²r²)/r
        
        Parameters:
        -----------
        r : float
            Distance between two atoms
            
        Returns:
        --------
        float
            First derivative of short-range Coulomb potential
        """
        if r < 1e-10:
            # Use limit as r→0 (Taylor expansion)
            return -2 * self.omega**3 / (3 * np.sqrt(np.pi))
        
        # Error function term
        erf_term = erf(self.omega * r)
        
        # Exponential term
        exp_term = 2 * self.omega * np.exp(-(self.omega * r)**2) / (np.sqrt(np.pi) * r)
        
        # Coulomb term
        coulomb_term = (erf_term - 1.0) / r**2
        
        return exp_term + coulomb_term
    
    def sr_coulomb_second_derivative(self, r):
        """Calculate second derivative of short-range Coulomb potential.
        
        d²V_SR(r)/dr² = 2(1-erf(ω*r))/r³ + 2erf(ω*r)/r³ + 4ω/(√π*r²)*e^(-ω²r²) + 2ω³/√π*e^(-ω²r²)
        
        Parameters:
        -----------
        r : float
            Distance between two atoms
            
        Returns:
        --------
        float
            Second derivative of short-range Coulomb potential
        """
        if r < 1e-10:
            # Use limit as r→0 (Taylor expansion)
            return 0.0
        
        # Error function term
        erf_term = erf(self.omega * r)
        
        # Exponential terms
        exp_factor = np.exp(-(self.omega * r)**2) / np.sqrt(np.pi)
        exp_term1 = 4 * self.omega * exp_factor / r**2
        exp_term2 = 2 * (self.omega**3) * exp_factor
        
        # Coulomb term
        coulomb_term = 2 * (2 * erf_term - 1) / r**3
        
        return coulomb_term + exp_term1 + exp_term2
    
    def estimate_atomic_charges(self, element_list):
        """Estimate atomic charges based on Pauling electronegativity.
        
        Parameters:
        -----------
        element_list : list
            List of element symbols
            
        Returns:
        --------
        numpy.ndarray
            Estimated atomic charges
        """
        # Pauling electronegativity values
        electronegativity = {
            'H': 2.20, 'He': 0.00,
            'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 
            'O': 3.44, 'F': 3.98, 'Ne': 0.00,
            'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 
            'S': 2.58, 'Cl': 3.16, 'Ar': 0.00,
            'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 
            'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 
            'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 
            'Se': 2.55, 'Br': 2.96, 'Kr': 0.00
        }
        
        n_atoms = len(element_list)
        charges = np.zeros(n_atoms)
        
        # Calculate average electronegativity (reference value)
        en_values = [electronegativity.get(element, 2.0) for element in element_list]
        avg_en = sum(en_values) / len(en_values)
        
        # Assign charges based on electronegativity differences
        for i, element in enumerate(element_list):
            en = electronegativity.get(element, 2.0)
            charges[i] = 0.2 * (avg_en - en)  # Scale by 0.2
        
        return charges
    
    def calculate_pair_hessian(self, r_vec, r_ij, atomic_charges, atom_i, atom_j):
        """Calculate Hessian contribution from short-range Coulomb between atom pair.
        
        Parameters:
        -----------
        r_vec : numpy.ndarray
            Relative position vector from atom i to atom j
        r_ij : float
            Distance between atoms i and j
        atomic_charges : numpy.ndarray
            Array of atomic charges
        atom_i, atom_j : int
            Atom indices
            
        Returns:
        --------
        numpy.ndarray
            3x3 Hessian block matrix
        """
        # Return zeros if beyond cutoff distance
        if r_ij > self.sr_cutoff:
            return np.zeros((3, 3))
        
        # Unit direction vector
        r_unit = r_vec / r_ij
        
        # Charge-based coefficient
        q_i = atomic_charges[atom_i]
        q_j = atomic_charges[atom_j]
        q_factor = q_i * q_j * self.cx_sr * self.scaling_factor
        
        # Calculate second derivative
        d2v = self.sr_coulomb_second_derivative(r_ij)
        
        # Calculate tensor using outer product
        r_outer = np.outer(r_unit, r_unit)
        
        # Calculate Hessian block
        identity = np.eye(3)
        hessian_block = q_factor * (d2v * r_outer + 
                           self.sr_coulomb_first_derivative(r_ij) / r_ij * (identity - r_outer))
        
        return hessian_block
    
    def calculate_correction_hessian(self, coord, element_list):
        """Calculate complete short-range correction Hessian, excluding bonded pairs.
        
        Parameters:
        -----------
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
            
        Returns:
        --------
        numpy.ndarray
            Short-range correction Hessian
        """
        n_atoms = len(coord)
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
        
        # Detect bonded atom pairs
        bonded_pairs = self.detect_bonds(coord, element_list)
        
        # Estimate atomic charges
        atomic_charges = self.estimate_atomic_charges(element_list)
        
        # Number of bonded pairs and total pairs for statistics
        num_total_pairs = n_atoms * (n_atoms - 1) // 2
        num_bonded_pairs = len(bonded_pairs) // 2  # Divide by 2 because we stored both (i,j) and (j,i)
        print(f"Detected {num_bonded_pairs} bonded pairs out of {num_total_pairs} total pairs")
        print(f"Short-range correction will be applied to {num_total_pairs - num_bonded_pairs} non-bonded pairs only")
        
        # Loop over all atom pairs
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Skip bonded atom pairs
                if (i, j) in bonded_pairs or (j, i) in bonded_pairs:
                    continue
                
                # Calculate interatomic vector and distance
                r_vec = coord[j] - coord[i]
                r_ij = np.linalg.norm(r_vec)
                
                # Calculate Hessian block for this pair
                hess_block = self.calculate_pair_hessian(
                    r_vec, r_ij, atomic_charges, i, j
                )
                
                # Add to the Hessian matrix
                for a in range(3):
                    for b in range(3):
                        # Diagonal blocks
                        hessian[3*i+a, 3*i+b] += hess_block[a, b]
                        hessian[3*j+a, 3*j+b] += hess_block[a, b]
                        
                        # Off-diagonal blocks
                        hessian[3*i+a, 3*j+b] -= hess_block[a, b]
                        hessian[3*j+a, 3*i+b] -= hess_block[a, b]
        
        return hessian
    
    def apply_correction(self, base_hessian, coord, element_list):
        """Apply short-range correction to an existing Hessian.
        
        Parameters:
        -----------
        base_hessian : numpy.ndarray
            Base model Hessian
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
            
        Returns:
        --------
        numpy.ndarray
            Corrected Hessian
        """
        tools = Calculationtools()
        # Calculate short-range correction
        correction = self.calculate_correction_hessian(coord, element_list)
        correction = tools.project_out_hess_tr_and_rot_for_coord(correction, element_list, coord, display_eigval=False)
        # Add correction to base Hessian
        corrected_hessian = base_hessian + correction
        
        corrected_hessian = 0.5 * (corrected_hessian + corrected_hessian.T)  # Symmetrize
        # Remove translational and rotational modes
        
        return corrected_hessian
    
    def main(self, coord, element_list, base_hessian):
        """Main method to apply short-range correction to a model Hessian.
        
        Parameters:
        -----------
        coord : numpy.ndarray
            Atomic coordinates (Bohr)
        element_list : list
            List of element symbols
        base_hessian : numpy.ndarray
            Base model Hessian
            
        Returns:
        --------
        numpy.ndarray
            Hessian with short-range correction
        """
        print(f"Applying short-range correction (ω={self.omega:.3f}) to model Hessian...")
        print("The correction will be applied only to non-bonded atom pairs.")
        
        # Apply correction
        corrected_hessian = self.apply_correction(base_hessian, coord, element_list)
        
        # Handle NaN values
        corrected_hessian = np.nan_to_num(corrected_hessian, nan=0.0)
        
        print("Short-range correction applied successfully")
        return corrected_hessian
