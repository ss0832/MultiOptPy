import numpy as np
from typing import Tuple, Optional
from multioptpy.SQM.sqm1.dftd4 import D4DispersionModel, D4Parameters

class D4DispersionEnergyCalculator:
    """
    D4 Dispersion energy calculator wrapper
    
    Wrapper class for the existing D4 dispersion model that provides
    a consistent interface for energy, gradient, and Hessian calculations
    in the component-based SQM1 calculator architecture.
    """
    
    def __init__(self, data):
        """
        Initialize D4 dispersion energy calculator
        
        Parameters:
        -----------
        data : SQM1Data
            Parametrization data containing D4 dispersion parameters
        """
        self.data = data
        self.d4_calculator = D4DispersionModel()
        self.params = D4Parameters()  # Initialize with default D4 parameters
        self.BOHR_TO_ANG = 0.52917721092  # Bohr to Angstrom conversion
    
    def calculate_energy(self,
                        coords: np.ndarray,
                        atomic_numbers: np.ndarray,
                        charges: Optional[np.ndarray] = None) -> float:
        """
        Calculate D4 dispersion energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,), optional
            Atomic charges. If None, uses zero charges.
            
        Returns:
        --------
        energy : float
            D4 dispersion energy in Hartree
        """
        if charges is None:
            charges = np.zeros(len(atomic_numbers))
        
        # Use existing D4 dispersion calculator
        energy, _ = self.d4_calculator.calculate_dispersion_energy_gradient(
            coords, atomic_numbers, charges, self.params
        )
        
        return energy
    
    def calculate_gradient(self,
                          coords: np.ndarray,
                          atomic_numbers: np.ndarray,
                          charges: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate analytical gradient of D4 dispersion energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,), optional
            Atomic charges. If None, uses zero charges.
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            D4 dispersion gradient in Hartree/Bohr
        """
        if charges is None:
            charges = np.zeros(len(atomic_numbers))
        
        # Use existing D4 dispersion calculator
        _, gradient = self.d4_calculator.calculate_dispersion_energy_gradient(
            coords, atomic_numbers, charges, self.params
        )
        
        return gradient
    
    def calculate_hessian(self,
                         coords: np.ndarray,
                         atomic_numbers: np.ndarray,
                         charges: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate analytical Hessian of D4 dispersion energy
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,), optional
            Atomic charges. If None, uses zero charges.
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            D4 dispersion Hessian in Hartree/Bohr²
        """
        if charges is None:
            charges = np.zeros(len(atomic_numbers))
        
        # Check if analytical Hessian is available
        if hasattr(self.d4_calculator, 'calculate_dispersion_hessian'):
            hessian = self.d4_calculator.calculate_dispersion_hessian(
                coords, atomic_numbers, charges
            )
        else:
            # Fall back to numerical Hessian
            hessian = self._calculate_numerical_hessian(coords, atomic_numbers, charges)
        
        return hessian
    
    def _calculate_numerical_hessian(self,
                                   coords: np.ndarray,
                                   atomic_numbers: np.ndarray,
                                   charges: np.ndarray) -> np.ndarray:
        """
        Calculate numerical Hessian of D4 dispersion energy
        
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
            Numerical D4 dispersion Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        delta = 1e-5  # Finite difference step size in Angstrom
        
        # Central finite difference for each coordinate pair
        for i in range(n_coords):
            atom_i = i // 3
            coord_i = i % 3
            
            # Calculate gradient at x_i + delta
            coords_plus = coords.copy()
            coords_plus[atom_i, coord_i] += delta
            grad_plus = self.calculate_gradient(coords_plus, atomic_numbers, charges)
            
            # Calculate gradient at x_i - delta
            coords_minus = coords.copy()
            coords_minus[atom_i, coord_i] -= delta
            grad_minus = self.calculate_gradient(coords_minus, atomic_numbers, charges)
            
            # Numerical second derivative
            grad_diff = (grad_plus - grad_minus) / (2.0 * delta)
            
            # Convert from Hartree/Ang² to Hartree/Bohr²
            hessian[i, :] = grad_diff.flatten() / self.BOHR_TO_ANG
        
        return hessian
    
    def set_damping_parameters(self, s6: float = None, s8: float = None, a1: float = None, a2: float = None):
        """
        Set D4 damping parameters
        
        Parameters:
        -----------
        s6 : float, optional
            s6 damping parameter
        s8 : float, optional
            s8 damping parameter
        a1 : float, optional
            a1 damping parameter
        a2 : float, optional
            a2 damping parameter
        """
        if hasattr(self.d4_calculator, 'set_damping_parameters'):
            self.d4_calculator.set_damping_parameters(s6, s8, a1, a2)
    
    def get_dispersion_coefficients(self,
                                   atomic_numbers: np.ndarray,
                                   charges: Optional[np.ndarray] = None) -> dict:
        """
        Get D4 dispersion coefficients
        
        Parameters:
        -----------
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,), optional
            Atomic charges
            
        Returns:
        --------
        coefficients : dict
            Dictionary containing C6, C8 coefficients and other parameters
        """
        if charges is None:
            charges = np.zeros(len(atomic_numbers))
        
        if hasattr(self.d4_calculator, 'get_dispersion_coefficients'):
            return self.d4_calculator.get_dispersion_coefficients(atomic_numbers, charges)
        else:
            return {}
    
    def calculate_coordination_numbers(self,
                                     coords: np.ndarray,
                                     atomic_numbers: np.ndarray) -> np.ndarray:
        """
        Calculate coordination numbers used in D4 model
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
            
        Returns:
        --------
        cn : np.ndarray, shape (n_atoms,)
            Coordination numbers
        """
        if hasattr(self.d4_calculator, 'calculate_coordination_numbers'):
            return self.d4_calculator.calculate_coordination_numbers(coords, atomic_numbers)
        else:
            # Fallback: return zeros
            return np.zeros(len(atomic_numbers))
    
    def get_cutoff_radii(self) -> Tuple[float, float]:
        """
        Get D4 dispersion cutoff radii
        
        Returns:
        --------
        r_cutoff : float
            Interaction cutoff radius in Angstrom
        cn_cutoff : float
            Coordination number cutoff radius in Angstrom
        """
        if hasattr(self.d4_calculator, 'get_cutoff_radii'):
            return self.d4_calculator.get_cutoff_radii()
        else:
            return 50.0, 25.0  # Default values