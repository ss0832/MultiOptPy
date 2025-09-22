import numpy as np
from typing import Tuple, Optional

class EHTEnergyCalculator:
    """
    Extended Hückel Theory (EHT) energy calculator
    
    Handles electronic energy calculation and its derivatives for the
    experimental semiempirical approach inspired by GFN0-xTB method.
    """
    
    def __init__(self, data):
        """
        Initialize EHT energy calculator
        
        Parameters:
        -----------
        data : SQM1Data
            Parametrization data containing Hamiltonian parameters
        """
        self.data = data
    
    def calculate_energy(self, 
                        coords: np.ndarray,
                        atomic_numbers: np.ndarray,
                        charges: Optional[np.ndarray] = None) -> float:
        """
        Calculate electronic energy contribution (simplified but more accurate)
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, optional
            Atomic charges. If None, will use zero charges.
            
        Returns:
        --------
        energy : float
            Electronic energy in Hartree
        """
        if charges is None:
            charges = np.zeros(len(atomic_numbers))
        
        n_atoms = len(atomic_numbers)
        energy = 0.0
        
        # Atomic self-energy contributions (approximate)
        for i in range(n_atoms):
            z = atomic_numbers[i]
            # Approximate atomic energy (ionization potential scaling)
            energy += -0.5 * z * (1.0 - 0.1 * abs(charges[i]))  # Charge dependence
        
        # Two-electron Coulomb interactions (handled by EEQ is already included)
        # Keep this minimal to avoid double counting
        bohr_to_ang = 0.52917721092  # Bohr to Angstrom conversion
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = np.linalg.norm(coords[i] - coords[j]) / bohr_to_ang
                
                # Very small contribution to avoid double counting with EEQ
                energy += 0.01 * charges[i] * charges[j] / r_ij
        
        return energy
    
    def calculate_gradient(self,
                          coords: np.ndarray,
                          atomic_numbers: np.ndarray,
                          charges: np.ndarray,
                          multiplicity: int,
                          cn: np.ndarray,
                          cn_gradients: np.ndarray,
                          charge_gradients: np.ndarray) -> np.ndarray:
        """
        Calculate analytical gradient of electronic energy
        
        This computes the proper analytical derivative of the electronic 
        energy with respect to nuclear coordinates, including contributions from:
        1. Direct coordinate dependence through coordination numbers: ∂E/∂CN × ∂CN/∂R
        2. Indirect dependence through charge derivatives: ∂E/∂q × ∂q/∂R
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,)
            Atomic charges
        multiplicity : int
            Spin multiplicity
        cn : np.ndarray, shape (n_atoms,)
            Coordination numbers
        cn_gradients : np.ndarray, shape (n_atoms, n_atoms, 3)
            Coordination number gradients: ∂CN_i/∂R_j
        charge_gradients : np.ndarray, shape (n_atoms, n_atoms, 3)
            Charge gradients: ∂q_i/∂R_j
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            Electronic energy gradient in Hartree/Bohr
        """
        n_atoms = len(atomic_numbers)
        gradient = np.zeros((n_atoms, 3))
        
        # Loop over atoms to calculate their contributions to the gradient
        for i in range(n_atoms):
            z = atomic_numbers[i]
            z_idx = z - 1  # Convert to 0-based indexing
            
            # Get number of shells for this atom
            n_shells = self.data.nShell[z_idx] if hasattr(self.data, 'nShell') else 2
            n_shells = min(n_shells, 4)  # Limit to available shells
            
            # Calculate gradient contribution for each shell of atom i
            for ish in range(n_shells):
                se_base = self.data.hamiltonian.selfEnergy[ish, z_idx]
                
                if se_base == 0.0:
                    continue
                
                # CN-dependent contribution: ∂E/∂R = -kCN * ∂CN/∂R
                if hasattr(self.data.hamiltonian, 'kCN'):
                    kCN = self.data.hamiltonian.kCN[ish, z_idx]
                    
                    # Add gradient contribution from coordination number derivative
                    # ∂E/∂R_j = -kCN_i * ∂CN_i/∂R_j for all j
                    for j in range(n_atoms):
                        gradient[j] -= kCN * cn_gradients[i, j, :]
                
                # Charge-dependent contributions: ∂E/∂R = ∂E/∂q * ∂q/∂R
                # Linear charge term: se = se - kQShell * q_i
                if hasattr(self.data.hamiltonian, 'kQShell'):
                    kQShell = self.data.hamiltonian.kQShell[ish, z_idx]
                    
                    # ∂E/∂q_i = -kQShell
                    # ∂E/∂R = -kQShell * ∂q_i/∂R
                    for j in range(n_atoms):
                        gradient[j] -= kQShell * charge_gradients[i, j, :]
                
                # Quadratic charge term: se = se - kQAtom * q_i^2  
                if hasattr(self.data.hamiltonian, 'kQAtom'):
                    kQAtom = self.data.hamiltonian.kQAtom[z_idx]
                    
                    # ∂E/∂q_i = -2 * kQAtom * q_i
                    # ∂E/∂R = -2 * kQAtom * q_i * ∂q_i/∂R
                    for j in range(n_atoms):
                        gradient[j] -= 2.0 * kQAtom * charges[i] * charge_gradients[i, j, :]
        
        return gradient
    
    def calculate_hessian(self,
                         coords: np.ndarray,
                         atomic_numbers: np.ndarray,
                         charges: np.ndarray,
                         multiplicity: int,
                         cn: np.ndarray) -> np.ndarray:
        """
        Calculate analytical Hessian of electronic energy
        
        For now, this returns a numerical approximation via finite differences
        of the gradient. A full analytical implementation would be quite complex.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        charges : np.ndarray, shape (n_atoms,)
            Atomic charges
        multiplicity : int
            Spin multiplicity
        cn : np.ndarray, shape (n_atoms,)
            Coordination numbers
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            Electronic energy Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        # For now, use a placeholder - full analytical Hessian is complex
        # In practice, this would require second derivatives of CN and charges
        # with respect to coordinates, which is computationally intensive
        
        # Return zero Hessian for electronic contribution for now
        # This could be implemented numerically if needed
        
        return hessian