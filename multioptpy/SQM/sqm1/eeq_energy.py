import numpy as np
import warnings
from typing import Tuple, Optional

class EEQEnergyCalculator:
    """
    Electronegativity Equilibration (EEQ) energy calculator
    
    Handles EEQ charge calculation and electrostatic energy computation
    and their derivatives for the experimental semiempirical approach inspired by GFN0-xTB method.
    """
    
    def __init__(self, data):
        """
        Initialize EEQ energy calculator
        
        Parameters:
        -----------
        data : SQM1Data
            Parametrization data containing EEQ and Coulomb parameters
        """
        self.data = data
        self.BOHR_TO_ANG = 0.52917721092  # Bohr to Angstrom conversion
    
    def calculate_charges_eeq(self, 
                             coords: np.ndarray,
                             atomic_numbers: np.ndarray,
                             cn: np.ndarray,
                             total_charge: float = 0.0) -> np.ndarray:
        """
        Calculate atomic charges using electronegativity equilibration (EEQ)
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        cn : np.ndarray, shape (n_atoms,)
            Coordination numbers
        total_charge : float
            Total molecular charge
            
        Returns:
        --------
        charges : np.ndarray, shape (n_atoms,)
            Atomic partial charges
        """
        n_atoms = len(atomic_numbers)
        
        # Setup EEQ matrix equation A*q = b
        A = np.zeros((n_atoms + 1, n_atoms + 1))
        b = np.zeros(n_atoms + 1)
        
        # Fill diagonal elements (chemical hardnesses)
        for i in range(n_atoms):
            z = atomic_numbers[i]
            idx = z - 1  # Convert to 0-based indexing
            
            # Chemical hardness with CN dependence
            gamma = self.data.coulomb.chemicalHardness[idx]
            chi = self.data.eeq_chi[idx] + self.data.eeq_kcn[idx] * cn[i]
            
            A[i, i] = gamma + self.data.coulomb.chargeWidth[idx] ** 2
            b[i] = -chi
        
        # Fill off-diagonal elements (Coulomb interactions)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = np.linalg.norm(coords[i] - coords[j]) / self.BOHR_TO_ANG  # Convert to Bohr
                
                # Coulomb interaction with charge width correction
                z_i = atomic_numbers[i] - 1
                z_j = atomic_numbers[j] - 1
                alpha_i = self.data.coulomb.chargeWidth[z_i]
                alpha_j = self.data.coulomb.chargeWidth[z_j]
                
                # Gaussian damped Coulomb interaction
                gamma_ij = 1.0 / np.sqrt(r_ij**2 + (alpha_i + alpha_j)**2)
                
                A[i, j] = gamma_ij
                A[j, i] = gamma_ij
        
        # Charge constraint
        A[n_atoms, :n_atoms] = 1.0
        A[:n_atoms, n_atoms] = 1.0
        b[n_atoms] = total_charge
        
        # Solve the linear system
        try:
            solution = np.linalg.solve(A, b)
            charges = solution[:n_atoms]
        except np.linalg.LinAlgError:
            warnings.warn("EEQ charge calculation failed, using zero charges")
            charges = np.zeros(n_atoms)
        
        return charges
    
    def calculate_energy(self,
                        coords: np.ndarray,
                        atomic_numbers: np.ndarray,
                        charges: np.ndarray) -> float:
        """
        Calculate EEQ electrostatic energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,)
            Atomic charges
            
        Returns:
        --------
        energy : float
            EEQ electrostatic energy in Hartree
        """
        n_atoms = len(atomic_numbers)
        energy = 0.0
        
        # Self-energy contributions (diagonal terms)
        for i in range(n_atoms):
            z_i = atomic_numbers[i] - 1
            gamma_ii = self.data.coulomb.chemicalHardness[z_i] + self.data.coulomb.chargeWidth[z_i] ** 2
            energy += 0.5 * charges[i]**2 * gamma_ii
        
        # Interaction terms (off-diagonal)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = np.linalg.norm(coords[i] - coords[j]) / self.BOHR_TO_ANG
                
                z_i = atomic_numbers[i] - 1
                z_j = atomic_numbers[j] - 1
                alpha_i = self.data.coulomb.chargeWidth[z_i]
                alpha_j = self.data.coulomb.chargeWidth[z_j]
                
                gamma_ij = 1.0 / np.sqrt(r_ij**2 + (alpha_i + alpha_j)**2)
                energy += charges[i] * charges[j] * gamma_ij
        
        return energy
    
    def calculate_gradient(self,
                          coords: np.ndarray,
                          atomic_numbers: np.ndarray,
                          charges: np.ndarray) -> np.ndarray:
        """
        Calculate analytical gradient of EEQ electrostatic energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,)
            Atomic charges
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            EEQ energy gradient in Hartree/Bohr
        """
        n_atoms = len(atomic_numbers)
        gradient = np.zeros((n_atoms, 3))
        
        # Only the interaction terms contribute to the gradient
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = coords[i] - coords[j]
                r_ij = np.linalg.norm(r_vec) / self.BOHR_TO_ANG  # Convert to Bohr
                
                # Get EEQ parameters
                z_i = atomic_numbers[i] - 1
                z_j = atomic_numbers[j] - 1
                alpha_i = self.data.coulomb.chargeWidth[z_i]
                alpha_j = self.data.coulomb.chargeWidth[z_j]
                
                # Corrected damped Coulomb interaction
                alpha_sum_sq = (alpha_i + alpha_j)**2
                gamma_ij = 1.0 / np.sqrt(r_ij**2 + alpha_sum_sq)
                
                # Gradient of gamma_ij
                dgamma_dr = -r_ij * gamma_ij**3
                
                # EEQ interaction gradient: d/dr (q_i * q_j * gamma_ij)
                interaction_deriv = charges[i] * charges[j] * dgamma_dr
                
                # Chain rule: convert to Cartesian coordinates
                r_unit = r_vec / (r_ij * self.BOHR_TO_ANG)  # Unit vector
                force_vec = interaction_deriv * r_unit / self.BOHR_TO_ANG
                
                gradient[i] += force_vec
                gradient[j] -= force_vec
        
        return gradient
    
    def calculate_hessian(self,
                         coords: np.ndarray,
                         atomic_numbers: np.ndarray,
                         charges: np.ndarray) -> np.ndarray:
        """
        Calculate analytical Hessian of EEQ electrostatic energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,)
            Atomic charges
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            EEQ energy Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        # Calculate second derivatives of interaction terms
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = coords[i] - coords[j]
                r_ij = np.linalg.norm(r_vec) / self.BOHR_TO_ANG
                
                if r_ij < 1e-12:
                    continue
                
                z_i = atomic_numbers[i] - 1
                z_j = atomic_numbers[j] - 1
                alpha_i = self.data.coulomb.chargeWidth[z_i]
                alpha_j = self.data.coulomb.chargeWidth[z_j]
                
                alpha_sum_sq = (alpha_i + alpha_j)**2
                gamma_ij = 1.0 / np.sqrt(r_ij**2 + alpha_sum_sq)
                
                # First and second derivatives
                dgamma_dr = -r_ij * gamma_ij**3
                d2gamma_dr2 = gamma_ij**3 * (-1.0 + 3.0 * r_ij**2 * gamma_ij**2)
                
                # Energy prefactor
                q_product = charges[i] * charges[j]
                
                # Unit vector and outer product
                unit = r_vec / (r_ij * self.BOHR_TO_ANG)
                outer = np.outer(unit, unit)
                I = np.identity(3)
                
                # Hessian block
                block = q_product * ((d2gamma_dr2 - dgamma_dr / r_ij) * outer + (dgamma_dr / r_ij) * I) / (self.BOHR_TO_ANG**2)
                
                # Add to Hessian matrix
                for a in range(3):
                    for b in range(3):
                        hessian[3*i + a, 3*i + b] += block[a, b]
                        hessian[3*j + a, 3*j + b] += block[a, b]
                        hessian[3*i + a, 3*j + b] -= block[a, b]
                        hessian[3*j + a, 3*i + b] -= block[a, b]
        
        return hessian
    
    def calculate_charge_gradient(self, 
                                  coords: np.ndarray, 
                                  atomic_numbers: np.ndarray,
                                  charges: np.ndarray) -> np.ndarray:
        """
        Calculate analytical gradient of EEQ charges with respect to coordinates
        
        The EEQ charges are determined by solving the linear system:
        A * q = b
        where A is the Coulomb matrix and b contains electronegativity terms.
        
        The gradient is: ∂q/∂R = -A^(-1) * (∂A/∂R * q - ∂b/∂R)
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,)
            Current EEQ charges
            
        Returns:
        --------
        charge_gradients : np.ndarray, shape (n_atoms, n_atoms, 3)
            Charge gradients: ∂q_i/∂R_j
        """
        n_atoms = len(atomic_numbers)
        charge_gradients = np.zeros((n_atoms, n_atoms, 3))
        
        # For now, use a simplified numerical approach for charge gradients
        # This could be implemented analytically, but it's quite complex
        delta = 1e-6
        
        for j in range(n_atoms):
            for k in range(3):
                # Calculate charges at displaced coordinates
                coords_plus = coords.copy()
                coords_minus = coords.copy()
                coords_plus[j, k] += delta
                coords_minus[j, k] -= delta
                
                # Dummy CN for charge calculation (will be provided by main calculator)
                cn_dummy = np.ones(n_atoms)
                
                charges_plus = self.calculate_charges_eeq(coords_plus, atomic_numbers, cn_dummy)
                charges_minus = self.calculate_charges_eeq(coords_minus, atomic_numbers, cn_dummy)
                
                # Numerical derivative
                charge_gradients[:, j, k] = (charges_plus - charges_minus) / (2.0 * delta)
        
        return charge_gradients