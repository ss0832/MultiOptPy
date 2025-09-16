import numpy as np
from multioptpy.Parameters.parameter import GNB_radii_lib

class MorseApproxHessian:
    """
    A simple class to generate a model Hessian based on the second derivative
    of a Morse potential, using GNB_radii_lib for covalent radii to estimate
    equilibrium bond distances. This is a highly simplified illustration.

    In this version, the covalent radii are obtained from GNB_radii_lib(element).
    """

    def __init__(self, De=0.10, a=0.20):
        """
        Parameters
        ----------
        De : float
            Dissociation energy in arbitrary units (e.g., Hartree).
        a : float
            Range parameter for the Morse potential.
        """
        self.De = De
        self.a = a

    def estimate_bond_length(self, elem1, elem2):
        """
        Estimate equilibrium bond length using GNB_radii_lib for each element.
        """
        r1 = GNB_radii_lib(elem1)
        r2 = GNB_radii_lib(elem2)
        return r1 + r2

    def compute_morse_second_derivative(self, r_current, r_eq):
        """
        Compute the second derivative of the Morse potential with respect to r,
        evaluated at r_current.

        V(r) = De * [1 - exp(-a * (r - r_eq))]^2
        
        For simplicity, use a general expanded form for the second derivative:
          d^2V/dr^2 = De * a^2 [ -2 e^{-x} + 4 e^{-2x} ]
        where x = a (r - r_eq).
        """
        x = self.a * (r_current - r_eq)
        # Expanded form for d^2V/dr^2
        second_derivative = self.De * (self.a ** 2) * (-2.0 * np.exp(-x) + 4.0 * np.exp(-2.0 * x))
        return second_derivative

    def create_model_hessian(self, coord, element_list):
        """
        Create a simple Hessian matrix for pairwise bonds as if
        each interaction is an independent Morse potential.

        Parameters
        ----------
        coord : numpy.ndarray
            Shape (N, 3) array of 3D coordinates for N atoms (in Å).
        element_list : list
            List of element symbols corresponding to the coordinates.

        Returns
        -------
        numpy.ndarray
            Hessian matrix of shape (3N, 3N).
        """
        n_atoms = len(element_list)
        hessian_size = 3 * n_atoms
        hessian = np.zeros((hessian_size, hessian_size), dtype=float)

        # Pairwise approach to generate naive bond Hessian elements
        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                # Estimate the equilibrium bond length
                r_eq = self.estimate_bond_length(element_list[i], element_list[j])

                # Current distance
                vec_ij = coord[j] - coord[i]
                dist_ij = np.linalg.norm(vec_ij)

                # Compute second derivative for the Morse potential
                d2V = self.compute_morse_second_derivative(dist_ij, r_eq)

                # Handle direction vector
                if dist_ij > 1.0e-12:
                    direction = vec_ij / dist_ij
                else:
                    direction = np.zeros(3)

                # Construct the 3x3 block k_ij * (direction outer direction)
                bond_k = d2V * np.outer(direction, direction)

                # Indices in the full Hessian
                block_i = slice(3 * i, 3 * i + 3)
                block_j = slice(3 * j, 3 * j + 3)

                # Update diagonal blocks
                hessian[block_i, block_i] += bond_k
                hessian[block_j, block_j] += bond_k
                # Update off-diagonal blocks
                hessian[block_i, block_j] -= bond_k
                hessian[block_j, block_i] -= bond_k

        # Symmetrize just in case
        hessian = 0.5 * (hessian + hessian.T)
        return hessian

