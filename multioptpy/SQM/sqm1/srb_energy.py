import numpy as np
from typing import Tuple, Optional

class SRBEnergyCalculator:
    """
    Short-Range Bond (SRB) energy calculator
    
    Handles short-range bond correction energy computation and derivatives
    for the experimental semiempirical approach inspired by GFN0-xTB method.
    """
    
    def __init__(self, data):
        """
        Initialize SRB energy calculator
        
        Parameters:
        -----------
        data : SQM1Data
            Parametrization data containing SRB parameters
        """
        self.data = data
        self.BOHR_TO_ANG = 0.52917721092  # Bohr to Angstrom conversion
    
    def _get_srb_parameters(self, z_i: int, z_j: int) -> Tuple[float, float, float]:
        """
        Get SRB parameters for atom pair
        
        Parameters:
        -----------
        z_i, z_j : int
            Atomic numbers
            
        Returns:
        --------
        k_srb : float
            SRB force constant
        r0_srb : float  
            SRB equilibrium distance
        alpha_srb : float
            SRB exponential parameter
        """
        idx_i = z_i - 1
        idx_j = z_j - 1
        
        # SRB parameters from data
        k_i = self.data.srb.k[idx_i]
        k_j = self.data.srb.k[idx_j]
        r0_i = self.data.srb.r0[idx_i]
        r0_j = self.data.srb.r0[idx_j]
        alpha_i = self.data.srb.alpha[idx_i]
        alpha_j = self.data.srb.alpha[idx_j]
        
        # Combine parameters for pair
        k_srb = np.sqrt(k_i * k_j)
        r0_srb = r0_i + r0_j
        alpha_srb = 0.5 * (alpha_i + alpha_j)
        
        return k_srb, r0_srb, alpha_srb
    
    def _srb_damping_function(self, r: float, alpha: float) -> Tuple[float, float, float]:
        """
        Calculate SRB damping function and its derivatives
        
        f(r) = 1 / (1 + exp(alpha * r))
        
        Parameters:
        -----------
        r : float
            Distance in Bohr
        alpha : float
            Exponential parameter
            
        Returns:
        --------
        f : float
            Damping function value
        df_dr : float
            First derivative
        d2f_dr2 : float
            Second derivative
        """
        exp_term = np.exp(alpha * r)
        denom = 1.0 + exp_term
        
        f = 1.0 / denom
        df_dr = -alpha * exp_term / (denom**2)
        d2f_dr2 = alpha**2 * exp_term * (exp_term - 1.0) / (denom**3)
        
        return f, df_dr, d2f_dr2
    
    def calculate_energy(self,
                        coords: np.ndarray,
                        atomic_numbers: np.ndarray,
                        bond_pairs: Optional[list] = None) -> float:
        """
        Calculate short-range bond correction energy
        
        E_srb = sum_{bonded i,j} k_ij * exp(-alpha_ij * r_ij) * f(r_ij)
        
        where f(r) is a damping function and the sum runs over bonded pairs.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        bond_pairs : list, optional
            List of bonded atom pairs. If None, uses all pairs within cutoff.
            
        Returns:
        --------
        energy : float
            SRB correction energy in Hartree
        """
        n_atoms = len(atomic_numbers)
        energy = 0.0
        
        # If no bond pairs provided, use all pairs within cutoff
        if bond_pairs is None:
            bond_pairs = self._get_bonded_pairs(coords, atomic_numbers)
        
        for i, j in bond_pairs:
            r_ij = np.linalg.norm(coords[i] - coords[j]) / self.BOHR_TO_ANG  # Convert to Bohr
            
            if r_ij < 1e-12:
                continue
            
            z_i = atomic_numbers[i]
            z_j = atomic_numbers[j]
            
            # Get SRB parameters
            k_srb, r0_srb, alpha_srb = self._get_srb_parameters(z_i, z_j)
            
            # SRB damping function
            f_damp, _, _ = self._srb_damping_function(r_ij - r0_srb, alpha_srb)
            
            # SRB correction energy
            srb_energy = k_srb * np.exp(-alpha_srb * r_ij) * f_damp
            energy += srb_energy
        
        return energy
    
    def calculate_gradient(self,
                          coords: np.ndarray,
                          atomic_numbers: np.ndarray,
                          bond_pairs: Optional[list] = None) -> np.ndarray:
        """
        Calculate analytical gradient of SRB correction energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        bond_pairs : list, optional
            List of bonded atom pairs
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            SRB energy gradient in Hartree/Bohr
        """
        n_atoms = len(atomic_numbers)
        gradient = np.zeros((n_atoms, 3))
        
        # If no bond pairs provided, use all pairs within cutoff
        if bond_pairs is None:
            bond_pairs = self._get_bonded_pairs(coords, atomic_numbers)
        
        for i, j in bond_pairs:
            r_vec = coords[i] - coords[j]
            r_ij = np.linalg.norm(r_vec) / self.BOHR_TO_ANG  # Convert to Bohr
            
            if r_ij < 1e-12:
                continue
            
            z_i = atomic_numbers[i]
            z_j = atomic_numbers[j]
            
            # Get SRB parameters
            k_srb, r0_srb, alpha_srb = self._get_srb_parameters(z_i, z_j)
            
            # SRB functions and derivatives
            exp_term = np.exp(-alpha_srb * r_ij)
            f_damp, df_dr, _ = self._srb_damping_function(r_ij - r0_srb, alpha_srb)
            
            # Derivative of SRB energy
            dsrb_dr = k_srb * (-alpha_srb * exp_term * f_damp + exp_term * df_dr)
            
            # Convert to Cartesian coordinates
            r_unit = r_vec / (r_ij * self.BOHR_TO_ANG)  # Unit vector
            force_vec = dsrb_dr * r_unit / self.BOHR_TO_ANG
            
            gradient[i] += force_vec
            gradient[j] -= force_vec
        
        return gradient
    
    def calculate_hessian(self,
                         coords: np.ndarray,
                         atomic_numbers: np.ndarray,
                         bond_pairs: Optional[list] = None) -> np.ndarray:
        """
        Calculate analytical Hessian of SRB correction energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        bond_pairs : list, optional
            List of bonded atom pairs
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            SRB energy Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        # If no bond pairs provided, use all pairs within cutoff
        if bond_pairs is None:
            bond_pairs = self._get_bonded_pairs(coords, atomic_numbers)
        
        for i, j in bond_pairs:
            r_vec = coords[i] - coords[j]
            r_ij = np.linalg.norm(r_vec) / self.BOHR_TO_ANG  # Convert to Bohr
            
            if r_ij < 1e-12:
                continue
            
            z_i = atomic_numbers[i]
            z_j = atomic_numbers[j]
            
            # Get SRB parameters
            k_srb, r0_srb, alpha_srb = self._get_srb_parameters(z_i, z_j)
            
            # SRB functions and derivatives
            exp_term = np.exp(-alpha_srb * r_ij)
            f_damp, df_dr, d2f_dr2 = self._srb_damping_function(r_ij - r0_srb, alpha_srb)
            
            # First derivative
            dsrb_dr = k_srb * (-alpha_srb * exp_term * f_damp + exp_term * df_dr)
            
            # Second derivative
            d2srb_dr2 = k_srb * (alpha_srb**2 * exp_term * f_damp - 
                                2.0 * alpha_srb * exp_term * df_dr + 
                                exp_term * d2f_dr2)
            
            # Unit vector and outer product
            unit = r_vec / (r_ij * self.BOHR_TO_ANG)
            outer = np.outer(unit, unit)
            I = np.identity(3)
            
            # Hessian block
            block = ((d2srb_dr2 - dsrb_dr / r_ij) * outer + 
                    (dsrb_dr / r_ij) * I) / (self.BOHR_TO_ANG**2)
            
            # Add to Hessian matrix
            for a in range(3):
                for b in range(3):
                    hessian[3*i + a, 3*i + b] += block[a, b]
                    hessian[3*j + a, 3*j + b] += block[a, b]
                    hessian[3*i + a, 3*j + b] -= block[a, b]
                    hessian[3*j + a, 3*i + b] -= block[a, b]
        
        return hessian
    
    def _get_bonded_pairs(self, coords: np.ndarray, atomic_numbers: np.ndarray) -> list:
        """
        Get bonded atom pairs based on distance cutoff
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
            
        Returns:
        --------
        bond_pairs : list
            List of bonded atom pairs (i, j) with i < j
        """
        n_atoms = len(atomic_numbers)
        bond_pairs = []
        
        # Simple distance-based bonding criterion
        # This could be replaced with more sophisticated topology detection
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = np.linalg.norm(coords[i] - coords[j])
                
                # Rough covalent radii (in Angstrom) for bond detection
                cov_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 
                           15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20}
                
                z_i = atomic_numbers[i]
                z_j = atomic_numbers[j]
                
                r_cov_i = cov_radii.get(z_i, 1.5)  # Default to 1.5 Å
                r_cov_j = cov_radii.get(z_j, 1.5)
                
                # Bond if distance is within 1.3 times sum of covalent radii
                if r_ij <= 1.3 * (r_cov_i + r_cov_j):
                    bond_pairs.append((i, j))
        
        return bond_pairs