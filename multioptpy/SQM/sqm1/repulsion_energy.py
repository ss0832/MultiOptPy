import numpy as np
from typing import Tuple, Optional

class RepulsionEnergyCalculator:
    """
    Nuclear repulsion energy calculator
    
    Handles nuclear-nuclear repulsion energy computation and derivatives
    for the experimental semiempirical approach inspired by GFN0-xTB method.
    """
    
    def __init__(self, data):
        """
        Initialize nuclear repulsion energy calculator
        
        Parameters:
        -----------
        data : SQM1Data
            Parametrization data containing nuclear repulsion parameters
        """
        self.data = data
        self.BOHR_TO_ANG = 0.52917721092  # Bohr to Angstrom conversion
    
    def calculate_energy(self,
                        coords: np.ndarray,
                        atomic_numbers: np.ndarray) -> float:
        """
        Calculate nuclear repulsion energy
        
        E_rep = sum_{i<j} Z_i * Z_j / r_ij * f_rep(r_ij)
        
        where f_rep is a repulsion damping function.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
            
        Returns:
        --------
        energy : float
            Nuclear repulsion energy in Hartree
        """
        n_atoms = len(atomic_numbers)
        energy = 0.0
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = np.linalg.norm(coords[i] - coords[j]) / self.BOHR_TO_ANG  # Convert to Bohr
                
                if r_ij < 1e-12:
                    continue
                
                z_i = atomic_numbers[i]
                z_j = atomic_numbers[j]
                
                # Get repulsion parameters
                idx_i = z_i - 1
                idx_j = z_j - 1
                
                # Nuclear repulsion with exponential damping
                alpha_i = self.data.repulsion.alpha[idx_i]
                alpha_j = self.data.repulsion.alpha[idx_j]
                zeff_i = self.data.repulsion.zeff[idx_i]
                zeff_j = self.data.repulsion.zeff[idx_j]
                
                # Average alpha parameter for pair
                alpha_ij = 0.5 * (alpha_i + alpha_j)
                
                # Exponential damping function (typical for semiempirical methods)
                damping = np.exp(-alpha_ij * r_ij)
                
                # Nuclear repulsion term
                coulomb_rep = zeff_i * zeff_j / r_ij
                energy += coulomb_rep * damping
        
        return energy
    
    def calculate_gradient(self,
                          coords: np.ndarray,
                          atomic_numbers: np.ndarray) -> np.ndarray:
        """
        Calculate analytical gradient of nuclear repulsion energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            Nuclear repulsion gradient in Hartree/Bohr
        """
        n_atoms = len(atomic_numbers)
        gradient = np.zeros((n_atoms, 3))
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = coords[i] - coords[j]
                r_ij = np.linalg.norm(r_vec) / self.BOHR_TO_ANG  # Convert to Bohr
                
                if r_ij < 1e-12:
                    continue
                
                z_i = atomic_numbers[i]
                z_j = atomic_numbers[j]
                
                # Get repulsion parameters
                idx_i = z_i - 1
                idx_j = z_j - 1
                
                alpha_i = self.data.repulsion.alpha[idx_i]
                alpha_j = self.data.repulsion.alpha[idx_j]
                zeff_i = self.data.repulsion.zeff[idx_i]
                zeff_j = self.data.repulsion.zeff[idx_j]
                
                # Average alpha parameter for pair
                alpha_ij = 0.5 * (alpha_i + alpha_j)
                
                # Exponential damping and its derivative
                damping = np.exp(-alpha_ij * r_ij)
                ddamping_dr = -alpha_ij * damping
                
                # Nuclear repulsion and its derivatives
                coulomb_rep = zeff_i * zeff_j / r_ij
                dcoulomb_dr = -coulomb_rep / r_ij
                
                # Total derivative: d/dr (coulomb * damping)
                total_deriv = dcoulomb_dr * damping + coulomb_rep * ddamping_dr
                
                # Convert to Cartesian coordinates
                r_unit = r_vec / (r_ij * self.BOHR_TO_ANG)  # Unit vector
                force_vec = total_deriv * r_unit / self.BOHR_TO_ANG
                
                gradient[i] += force_vec
                gradient[j] -= force_vec
        
        return gradient
    
    def calculate_hessian(self,
                         coords: np.ndarray,
                         atomic_numbers: np.ndarray) -> np.ndarray:
        """
        Calculate analytical Hessian of nuclear repulsion energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            Nuclear repulsion Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = coords[i] - coords[j]
                r_ij = np.linalg.norm(r_vec) / self.BOHR_TO_ANG  # Convert to Bohr
                
                if r_ij < 1e-12:
                    continue
                
                z_i = atomic_numbers[i]
                z_j = atomic_numbers[j]
                
                # Get repulsion parameters
                idx_i = z_i - 1
                idx_j = z_j - 1
                
                alpha_i = self.data.repulsion.alpha[idx_i]
                alpha_j = self.data.repulsion.alpha[idx_j]
                zeff_i = self.data.repulsion.zeff[idx_i]
                zeff_j = self.data.repulsion.zeff[idx_j]
                
                # Average alpha parameter for pair
                alpha_ij = 0.5 * (alpha_i + alpha_j)
                
                # Functions and derivatives
                damping = np.exp(-alpha_ij * r_ij)
                ddamping_dr = -alpha_ij * damping
                d2damping_dr2 = alpha_ij**2 * damping
                
                coulomb_rep = zeff_i * zeff_j / r_ij
                dcoulomb_dr = -coulomb_rep / r_ij
                d2coulomb_dr2 = 2.0 * coulomb_rep / r_ij**2
                
                # Second derivative of the product
                d2energy_dr2 = (d2coulomb_dr2 * damping + 
                               2.0 * dcoulomb_dr * ddamping_dr + 
                               coulomb_rep * d2damping_dr2)
                
                # First derivative for the off-diagonal terms
                denergy_dr = dcoulomb_dr * damping + coulomb_rep * ddamping_dr
                
                # Unit vector and outer product
                unit = r_vec / (r_ij * self.BOHR_TO_ANG)
                outer = np.outer(unit, unit)
                I = np.identity(3)
                
                # Hessian block
                block = ((d2energy_dr2 - denergy_dr / r_ij) * outer + 
                        (denergy_dr / r_ij) * I) / (self.BOHR_TO_ANG**2)
                
                # Add to Hessian matrix
                for a in range(3):
                    for b in range(3):
                        hessian[3*i + a, 3*i + b] += block[a, b]
                        hessian[3*j + a, 3*j + b] += block[a, b]
                        hessian[3*i + a, 3*j + b] -= block[a, b]
                        hessian[3*j + a, 3*i + b] -= block[a, b]
        
        return hessian
