import glob
import os
import numpy as np
from abc import ABC, abstractmethod
import warnings

# Suppress specific warnings during calculation
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, number_element
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer

class TersoffCore:
    """
    Core calculator for Tersoff potential.
    Handles both homo- and hetero-atomic clusters with appropriate parameters.
    """
    # Tersoff parameters for common elements
    # Source: Tersoff, J. Physical Review B, 1989, 39, 5566-5568 (Si)
    #         Tersoff, J. Physical Review Letters, 1988, 61, 2879-2882 (C)
    TERSOFF_PARAMETERS = {
        'Si': {
            'A': 1830.8, 'B': 471.18, 'lambda': 2.4799, 'mu': 1.7322, 
            'beta': 1.1e-6, 'n': 0.78734, 'c': 1.0039e5, 'd': 16.217, 
            'h': -0.59825, 'R': 2.7, 'D': 0.3
        },
        'C': {
            'A': 1393.6, 'B': 346.74, 'lambda': 3.4879, 'mu': 2.2119,
            'beta': 1.5724e-7, 'n': 0.72751, 'c': 3.8049e4, 'd': 4.3484,
            'h': -0.57058, 'R': 1.95, 'D': 0.15
        },
        'Ge': {
            'A': 1769.0, 'B': 419.23, 'lambda': 2.4451, 'mu': 1.7047,
            'beta': 9.0166e-7, 'n': 0.75627, 'c': 1.0643e5, 'd': 15.652,
            'h': -0.43884, 'R': 2.95, 'D': 0.15
        },
    }
    
    # Mixing rules parameters for hetero-atomic interactions
    # Based on common approaches in literature
    MIX_PARAMETERS = {
        ('Si', 'C'): {
            'A': 1612.2, 'B': 395.15, 'lambda': 2.9839, 'mu': 1.9720,
            'beta': 1.1e-6, 'n': 0.75743, 'c': 6.0e4, 'd': 13.0,
            'h': -0.585, 'R': 2.4, 'D': 0.2
        },
        ('Si', 'Ge'): {
            'A': 1800.0, 'B': 445.0, 'lambda': 2.46, 'mu': 1.72,
            'beta': 1.0e-6, 'n': 0.77, 'c': 1.03e5, 'd': 15.9,
            'h': -0.52, 'R': 2.8, 'D': 0.2
        },
        ('C', 'Ge'): {
            'A': 1580.0, 'B': 380.0, 'lambda': 2.97, 'mu': 1.96,
            'beta': 1.0e-6, 'n': 0.74, 'c': 7.0e4, 'd': 12.0,
            'h': -0.5, 'R': 2.5, 'D': 0.2
        },
    }

    # Constants for numerical stability - much more restrictive limits
    MAX_EXPONENT = 50.0    # Much lower maximum exponent to prevent overflow
    MIN_DISTANCE = 1e-8    # Minimum distance to avoid division by zero
    EPSILON = 1e-8         # Small value to avoid division by zero
    MAX_VALUE = 1e6        # Maximum value for any intermediate result

    def __init__(self):
        """Initializes a general Tersoff potential calculator."""
        self.UVL = UnitValueLib()
        # Cache for memoizing parameters of atom types
        self._param_cache = {}
        # Conversion factors
        self.angstrom_to_bohr = self.UVL.bohr2angstroms
        self.ev_to_hartree = 1.0 / self.UVL.hartree2eV

    def get_parameters(self, atom_i, atom_j):
        """
        Retrieves Tersoff parameters for a pair of atoms.
        Returns a dictionary of parameters for the interaction.
        """
        # Check cache first
        pair_key = (atom_i, atom_j)
        if pair_key in self._param_cache:
            return self._param_cache[pair_key]
            
        # For homo-atomic pairs, use direct parameters
        if atom_i == atom_j:
            if atom_i not in self.TERSOFF_PARAMETERS:
                raise ValueError(f"Atom symbol '{atom_i}' is not supported. "
                                 f"Supported: {list(self.TERSOFF_PARAMETERS.keys())}")
            params = self.TERSOFF_PARAMETERS[atom_i].copy()
            
        # For hetero-atomic pairs, use mixing parameters if available
        else:
            sorted_pair = tuple(sorted([atom_i, atom_j]))
            if sorted_pair in self.MIX_PARAMETERS:
                params = self.MIX_PARAMETERS[sorted_pair].copy()
            else:
                # Apply simple mixing rules if specific parameters not available
                if atom_i not in self.TERSOFF_PARAMETERS or atom_j not in self.TERSOFF_PARAMETERS:
                    raise ValueError(f"One or both atoms {atom_i}, {atom_j} are not supported.")
                    
                params_i = self.TERSOFF_PARAMETERS[atom_i]
                params_j = self.TERSOFF_PARAMETERS[atom_j]
                
                params = {
                    'A': np.sqrt(params_i['A'] * params_j['A']),
                    'B': np.sqrt(params_i['B'] * params_j['B']),
                    'lambda': 0.5 * (params_i['lambda'] + params_j['lambda']),
                    'mu': 0.5 * (params_i['mu'] + params_j['mu']),
                    'beta': np.sqrt(params_i['beta'] * params_j['beta']),
                    'n': 0.5 * (params_i['n'] + params_j['n']),
                    'c': np.sqrt(params_i['c'] * params_j['c']),
                    'd': np.sqrt(params_i['d'] * params_j['d']),
                    'h': 0.5 * (params_i['h'] + params_j['h']),
                    'R': 0.5 * (params_i['R'] + params_j['R']),
                    'D': 0.5 * (params_i['D'] + params_j['D']),
                }

        # Convert parameters from eV to Hartree and Å to Bohr
        # Energy parameters: A, B
        params['A'] *= self.ev_to_hartree
        params['B'] *= self.ev_to_hartree
        
        # Distance parameters: lambda, mu, R, D
        params['lambda'] /= self.angstrom_to_bohr
        params['mu'] /= self.angstrom_to_bohr
        params['R'] /= self.angstrom_to_bohr
        params['D'] /= self.angstrom_to_bohr
        
        self._param_cache[pair_key] = params
        return params

    def safe_exp(self, x):
        """
        Safely compute exponential function to avoid overflow.
        Uses a much more restrictive cap on the exponent.
        """
        if isinstance(x, np.ndarray):
            # For arrays, clip element-wise
            clipped_x = np.clip(x, -self.MAX_EXPONENT, self.MAX_EXPONENT)
            return np.exp(clipped_x)
        else:
            # For scalar values
            if x > self.MAX_EXPONENT:
                return np.exp(self.MAX_EXPONENT)
            elif x < -self.MAX_EXPONENT:
                return np.exp(-self.MAX_EXPONENT)
            return np.exp(x)

    def safe_value(self, x):
        """
        Clip any value to a safe range to prevent overflow in subsequent calculations.
        """
        if isinstance(x, np.ndarray):
            return np.clip(x, -self.MAX_VALUE, self.MAX_VALUE)
        else:
            return max(min(x, self.MAX_VALUE), -self.MAX_VALUE)

    def calculate_cutoff(self, r, R, D):
        """Calculate the cutoff function f_c(r)."""
        if r <= (R - D):
            return 1.0
        elif r >= (R + D):
            return 0.0
        else:
            return 0.5 - 0.5 * np.sin(np.pi * (r - R) / (2 * D))
            
    def calculate_cutoff_derivative(self, r, R, D):
        """Calculate the derivative of the cutoff function df_c(r)/dr."""
        if r <= (R - D) or r >= (R + D):
            return 0.0
        else:
            return -0.5 * np.pi / (2 * D) * np.cos(np.pi * (r - R) / (2 * D))

    def calculate_bond_angle(self, r_ij, r_ik):
        """Calculate the cosine of the angle between bonds ij and ik."""
        # Add small values to avoid division by zero
        norm_rij = np.linalg.norm(r_ij)
        norm_rik = np.linalg.norm(r_ik)
        
        # Check for extremely small values
        if norm_rij < self.MIN_DISTANCE or norm_rik < self.MIN_DISTANCE:
            return 0.0
        
        cos_theta = np.dot(r_ij, r_ik) / (norm_rij * norm_rik)
        # Ensure numerical stability by clamping to [-1, 1]
        return np.clip(cos_theta, -1.0, 1.0)
    
    def calculate_g(self, cos_theta, c, d, h):
        """Calculate the angular function g(theta)."""
        # Ensure d is not too small to avoid division by very small numbers
        d_safe = max(d, self.EPSILON)
        denom = d_safe**2 + (h - cos_theta)**2
        
        # Avoid division by zero
        if denom < self.EPSILON:
            denom = self.EPSILON
            
        result = 1.0 + (c**2 / d_safe**2) - (c**2 / denom)
        return self.safe_value(result)
    
    def calculate_g_derivative(self, cos_theta, c, d, h):
        """Calculate the derivative of g(theta) with respect to cos_theta."""
        # Ensure d is not too small
        d_safe = max(d, self.EPSILON)
        term = d_safe**2 + (h - cos_theta)**2
        
        # Avoid division by very small numbers
        if term < self.EPSILON:
            term = self.EPSILON
        
        result = 2.0 * c**2 * (h - cos_theta) / (term**2)
        return self.safe_value(result)

    def safe_bond_order_term(self, beta, zeta, n):
        """Safely calculate bond order term to avoid numerical issues."""
        # Avoid issues with very small zeta values
        if zeta < self.EPSILON:
            return 1.0
            
        # Limit the exponent to avoid overflow
        try:
            # Use logarithmic calculation for numerical stability
            log_power_term = n * np.log(beta) + n * np.log(max(zeta, self.EPSILON))
            
            # If exponent is too large, cap the result
            if log_power_term > np.log(self.MAX_VALUE):
                power_term = self.MAX_VALUE
            else:
                power_term = np.exp(log_power_term)
                
            # Ensure we don't divide by zero
            denom = 1.0 + power_term
            if denom < self.EPSILON:
                denom = self.EPSILON
                
            # Apply the power with safe exponent
            exponent = -1.0/(2.0*n)
            if exponent < 0 and denom < self.EPSILON:
                return 0.0  # Avoid division by zero for negative exponents
                
            result = denom**exponent
            return self.safe_value(result)
            
        except (OverflowError, FloatingPointError, RuntimeWarning):
            # If any numerical issues occur, return a safe default
            if n > 0:
                return 0.0  # Small value for positive n
            else:
                return 1.0  # Default for negative n
    
    def calculate_three_body_term(self, r_ij, r_ik, lambda1):
        """
        Safely calculate the three-body exponential term to avoid overflow.
        Uses a much more aggressive approach to prevent overflow.
        """
        # Calculate cubic term with tight bounds to prevent overflow
        diff = r_ij - r_ik
        
        # Very aggressive limiting to prevent overflow
        # Limit diff to prevent cubic overflow
        diff_limited = np.clip(diff, -2.0, 2.0)
        diff_cubed = diff_limited**3
        
        # Scale lambda to prevent overflow when cubed
        lambda_scaled = min(lambda1, np.cbrt(self.MAX_EXPONENT/8.0))
        
        # Calculate exponent with strict limits
        exponent = lambda_scaled**3 * diff_cubed
        exponent_limited = np.clip(exponent, -self.MAX_EXPONENT, self.MAX_EXPONENT)
        
        # Use safe exponential
        result = self.safe_exp(exponent_limited)
        
        # Ensure result is finite and within bounds
        if not np.isfinite(result):
            if exponent_limited > 0:
                return self.MAX_VALUE
            else:
                return 0.0
                
        return min(result, self.MAX_VALUE)

    def calculate_energy_and_gradient(self, coords_bohr, atom_symbols):
        """
        Calculates the Tersoff energy and gradient with aggressive numerical safeguards.
        
        Args:
            coords_bohr: Atomic coordinates in Bohr
            atom_symbols: List of atomic symbols
            
        Returns:
            Dictionary containing energy and gradient
        """
        num_atoms = coords_bohr.shape[0]
        if num_atoms <= 1:
            return {"energy": 0.0, "gradient": np.zeros_like(coords_bohr)}
        
        # Initialize energy and gradient
        total_energy = 0.0
        gradient = np.zeros_like(coords_bohr)
        
        # Precompute all distances and direction vectors
        diffs = {}
        dists = {}
        unit_vectors = {}
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                r_ij = coords_bohr[j] - coords_bohr[i]
                dist = np.linalg.norm(r_ij)
                # Ensure distance is not too small
                dist = max(dist, self.MIN_DISTANCE)
                diffs[(i, j)] = r_ij
                dists[(i, j)] = dist
                unit_vectors[(i, j)] = r_ij / dist
                
        # Main Tersoff calculation loop
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                    
                atom_i = atom_symbols[i]
                atom_j = atom_symbols[j]
                params = self.get_parameters(atom_i, atom_j)
                
                A = params['A']
                B = params['B']
                lambda1 = params['lambda']
                mu = params['mu']
                beta = params['beta']
                n = params['n']
                c = params['c']
                d = params['d']
                h = params['h']
                R = params['R']
                D = params['D']
                
                r_ij = dists[(i, j)]
                e_ij = unit_vectors[(i, j)]
                
                # Calculate cutoff
                fc_ij = self.calculate_cutoff(r_ij, R, D)
                if fc_ij < self.EPSILON:
                    continue  # Skip if beyond cutoff
                
                # Calculate repulsive and attractive terms with safe exponential
                exp_lambda_r = self.safe_exp(-lambda1 * r_ij)
                exp_mu_r = self.safe_exp(-mu * r_ij)
                
                # Calculate bond order term
                b_ij = 1.0  # Default if no neighbors
                db_ij_dcos = np.zeros(num_atoms)
                zeta_ij = 0.0
                
                for k in range(num_atoms):
                    if k == i or k == j:
                        continue
                        
                    atom_k = atom_symbols[k]
                    params_ik = self.get_parameters(atom_i, atom_k)
                    
                    r_ik = dists[(i, k)]
                    fc_ik = self.calculate_cutoff(r_ik, params_ik['R'], params_ik['D'])
                    
                    if fc_ik < self.EPSILON:
                        continue
                    
                    cos_theta = self.calculate_bond_angle(-diffs[(i, j)], -diffs[(i, k)])
                    g_ijk = self.calculate_g(cos_theta, c, d, h)
                    
                    # Calculate three-body term with strict overflow prevention
                    exp_term = self.calculate_three_body_term(r_ij, r_ik, lambda1)
                    
                    # Calculate zeta term with bounded multiplication
                    zeta_term = 0.0
                    try:
                        # Use log arithmetic to prevent overflow
                        if exp_term > 0:
                            log_term = np.log(fc_ik) + np.log(g_ijk) + np.log(exp_term)
                            if log_term < np.log(self.MAX_VALUE):
                                zeta_term = np.exp(log_term)
                            else:
                                zeta_term = self.MAX_VALUE
                    except (ValueError, RuntimeWarning, OverflowError):
                        # If log calculation fails, try direct multiplication with safeguards
                        if fc_ik < 1.0 and g_ijk < self.MAX_VALUE and exp_term < self.MAX_VALUE:
                            # Start with smallest value and multiply cautiously
                            temp = fc_ik * g_ijk
                            if temp < self.MAX_VALUE:
                                zeta_term = temp * min(exp_term, self.MAX_VALUE/temp if temp > 0 else self.MAX_VALUE)
                    
                    # Only add if the result is finite
                    if np.isfinite(zeta_term):
                        zeta_ij += min(zeta_term, self.MAX_VALUE - zeta_ij)
                    
                    # Store derivatives for later gradient calculation - with safeguards
                    dg_dcos = self.calculate_g_derivative(cos_theta, c, d, h)
                    db_term = 0.0
                    
                    # Use similar approach for derivative term
                    try:
                        if fc_ik > 0 and dg_dcos != 0 and exp_term > 0:
                            # Calculate using log arithmetic if possible
                            log_db = np.log(abs(fc_ik)) + np.log(abs(dg_dcos)) + np.log(exp_term)
                            if log_db < np.log(self.MAX_VALUE):
                                db_term = np.sign(dg_dcos) * np.exp(log_db)
                            else:
                                db_term = np.sign(dg_dcos) * self.MAX_VALUE
                    except (ValueError, RuntimeWarning, OverflowError):
                        # Fall back to cautious multiplication
                        if abs(fc_ik) < 1.0 and abs(dg_dcos) < self.MAX_VALUE and exp_term < self.MAX_VALUE:
                            temp = fc_ik * dg_dcos
                            if abs(temp) < self.MAX_VALUE:
                                db_term = temp * min(exp_term, self.MAX_VALUE/abs(temp) if abs(temp) > 0 else self.MAX_VALUE)
                    
                    # Store if finite
                    if np.isfinite(db_term):
                        db_ij_dcos[k] = self.safe_value(db_term)
                
                # Cap zeta value to prevent overflow in bond order calculation
                zeta_ij = min(zeta_ij, self.MAX_VALUE)
                
                # Safely compute bond order with zeta
                b_ij = self.safe_bond_order_term(beta, zeta_ij, n)
                
                # Calculate pair energy contribution with safeguards
                repulsive = min(A * exp_lambda_r, self.MAX_VALUE)
                attractive = max(-min(b_ij * B * exp_mu_r, self.MAX_VALUE), -self.MAX_VALUE)
                pair_energy = fc_ij * (repulsive + attractive)
                
                # Check for NaN or infinity in energy
                if not np.isfinite(pair_energy):
                    pair_energy = 0.0
                
                # Half the energy to avoid double counting
                total_energy += 0.5 * pair_energy
                
                # Calculate force components
                dfc_dr = self.calculate_cutoff_derivative(r_ij, R, D)
                drepulsive_dr = -lambda1 * repulsive
                dattractive_dr = mu * attractive
                
                # Direct force term (without bond-order derivatives)
                direct_force = fc_ij * (drepulsive_dr + dattractive_dr) + dfc_dr * (repulsive + attractive)
                direct_force = self.safe_value(direct_force)
                
                # Apply direct forces
                gradient[i] -= 0.5 * direct_force * e_ij
                gradient[j] += 0.5 * direct_force * e_ij
                
                # Bond-order derivative contributions (many-body forces)
                if zeta_ij > self.EPSILON and b_ij > self.EPSILON:
                    # Calculate derivative of bond order term safely
                    try:
                        # Using logarithmic approach for better numerical stability
                        log_term = np.log(beta) * n + np.log(zeta_ij) * (n-1) + np.log(b_ij) * (1+2*n)
                        if log_term < np.log(self.MAX_VALUE):
                            db_dzeta = -0.5 * np.exp(log_term)
                        else:
                            db_dzeta = -0.5 * self.MAX_VALUE
                    except (ValueError, RuntimeWarning, OverflowError):
                        # Fall back to direct calculation with safeguards
                        beta_n = min(beta**n, self.MAX_VALUE)
                        zeta_n_1 = min(zeta_ij**(n-1), self.MAX_VALUE)
                        b_ij_term = min(b_ij**(1+2*n), self.MAX_VALUE)
                        
                        # Apply bounds at each step
                        temp1 = min(beta_n * zeta_n_1, self.MAX_VALUE)
                        temp2 = min(temp1 * b_ij_term, self.MAX_VALUE)
                        db_dzeta = -0.5 * temp2
                    
                    # Ensure value is finite and bounded
                    db_dzeta = self.safe_value(db_dzeta)
                    
                    # Calculate bond force with bounds
                    dbond_force = self.safe_value(fc_ij * B * exp_mu_r * db_dzeta)
                    
                    for k in range(num_atoms):
                        if k == i or k == j:
                            continue
                            
                        atom_k = atom_symbols[k]
                        params_ik = self.get_parameters(atom_i, atom_k)
                        
                        r_ik = dists[(i, k)]
                        fc_ik = self.calculate_cutoff(r_ik, params_ik['R'], params_ik['D'])
                        
                        if fc_ik < self.EPSILON:
                            continue
                        
                        cos_theta = self.calculate_bond_angle(-diffs[(i, j)], -diffs[(i, k)])
                        g_ijk = self.calculate_g(cos_theta, c, d, h)
                        
                        # Calculate exponential term with extreme safeguards
                        exp_term = self.calculate_three_body_term(r_ij, r_ik, lambda1)
                        
                        # Angular derivative terms with strict bounds
                        if dists[(i, j)] * dists[(i, k)] > self.MIN_DISTANCE:
                            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
                            
                            if sin_theta > self.EPSILON:
                                # Angular forces on i, j, k with numerical safeguards
                                dcos_di = (e_ij / dists[(i, j)]) + (unit_vectors[(i, k)] / dists[(i, k)])
                                dcos_dj = -e_ij / dists[(i, j)]
                                dcos_dk = -unit_vectors[(i, k)] / dists[(i, k)]
                                
                                # Safe calculation of angular force
                                angular_force = 0.0
                                if np.isfinite(dbond_force) and np.isfinite(db_ij_dcos[k]):
                                    angular_force = self.safe_value(dbond_force * db_ij_dcos[k])
                                
                                # Apply angular forces with bounds checks
                                if np.all(np.isfinite(dcos_di)) and np.isfinite(angular_force):
                                    force_i = self.safe_value(angular_force) * dcos_di
                                    gradient[i] -= force_i
                                
                                if np.all(np.isfinite(dcos_dj)) and np.isfinite(angular_force):
                                    force_j = self.safe_value(angular_force) * dcos_dj
                                    gradient[j] -= force_j
                                
                                if np.all(np.isfinite(dcos_dk)) and np.isfinite(angular_force):
                                    force_k = self.safe_value(angular_force) * dcos_dk
                                    gradient[k] -= force_k
                        
                        # Radial derivative terms from three-body interactions
                        # Use very strict bounds for the derivative calculation
                        dexp_factor = 0.0
                        if abs(r_ij - r_ik) < 1.0:  # Stricter threshold
                            dexp_factor = self.safe_value(3 * lambda1**3 * (r_ij - r_ik)**2)
                        else:
                            dexp_factor = self.safe_value(3 * lambda1**3 * np.sign(r_ij - r_ik))
                            
                        # Calculate derivatives safely
                        if exp_term > 0 and exp_term < self.MAX_VALUE and dexp_factor < self.MAX_VALUE:
                            dexp_drij = self.safe_value(dexp_factor * exp_term)
                            dexp_drik = self.safe_value(-dexp_factor * exp_term)
                        else:
                            dexp_drij = 0.0
                            dexp_drik = 0.0
                        
                        # Safe calculation of radial forces
                        radial_force_ij = 0.0
                        radial_force_ik = 0.0
                        
                        # Compute forces cautiously
                        if dbond_force != 0.0:
                            # Calculate each term separately and check bounds
                            temp1 = self.safe_value(fc_ik * g_ijk)
                            if temp1 != 0.0 and dexp_drij != 0.0:
                                radial_force_ij = self.safe_value(dbond_force * temp1 * dexp_drij)
                                
                            if temp1 != 0.0 and dexp_drik != 0.0:
                                radial_force_ik = self.safe_value(dbond_force * temp1 * dexp_drik)
                        
                        # Apply radial forces with strict bounds checking
                        if np.isfinite(radial_force_ij) and abs(radial_force_ij) < self.MAX_VALUE:
                            gradient[i] -= radial_force_ij * e_ij
                            gradient[j] += radial_force_ij * e_ij
                        
                        if np.isfinite(radial_force_ik) and abs(radial_force_ik) < self.MAX_VALUE:
                            gradient[i] -= radial_force_ik * unit_vectors[(i, k)]
                            gradient[k] += radial_force_ik * unit_vectors[(i, k)]
                        
                        # Cutoff derivative for three-body
                        dfc_ik_dr = self.calculate_cutoff_derivative(r_ik, params_ik['R'], params_ik['D'])
                        
                        # Safe calculation of cutoff force
                        cutoff_force = 0.0
                        if dbond_force != 0.0 and dfc_ik_dr != 0.0:
                            temp1 = self.safe_value(g_ijk * exp_term)
                            if temp1 != 0.0:
                                cutoff_force = self.safe_value(dbond_force * dfc_ik_dr * temp1)
                        
                        # Apply cutoff forces with bounds checking
                        if np.isfinite(cutoff_force) and abs(cutoff_force) < self.MAX_VALUE:
                            gradient[i] -= cutoff_force * unit_vectors[(i, k)]
                            gradient[k] += cutoff_force * unit_vectors[(i, k)]
        
        # Final check of energy and gradient for numerical stability
        if not np.isfinite(total_energy):
            total_energy = 0.0
            
        # Ensure gradient is finite and within bounds
        for i in range(num_atoms):
            for j in range(3):
                if not np.isfinite(gradient[i, j]):
                    gradient[i, j] = 0.0
                else:
                    gradient[i, j] = self.safe_value(gradient[i, j])
        
        return {"energy": total_energy, "gradient": gradient}

    def calculate_hessian(self, coords_bohr, atom_symbols):
        """
        Calculates the Tersoff Hessian matrix using finite difference of gradients.
        Uses aggressive numerical safeguards to avoid overflow issues.
        
        Args:
            coords_bohr: Atomic coordinates in Bohr
            atom_symbols: List of atomic symbols
            
        Returns:
            Dictionary containing the Hessian matrix
        """
        print("Warning: Hessian calculation via finite difference is not tested well.")
        num_atoms = coords_bohr.shape[0]
        hessian = np.zeros((num_atoms * 3, num_atoms * 3))
        
        if num_atoms <= 1:
            return {"hessian": hessian}
        
        # Compute base energy and gradient
        base_results = self.calculate_energy_and_gradient(coords_bohr, atom_symbols)
        base_gradient = base_results["gradient"]
        
        # Small displacement for finite difference
        delta = 1e-5
        
        # For each degree of freedom
        for i in range(num_atoms):
            for j in range(3):  # x, y, z
                # Create displaced coordinates
                displaced_coords = coords_bohr.copy()
                displaced_coords[i, j] += delta
                
                # Calculate gradient at displaced position
                displaced_results = self.calculate_energy_and_gradient(displaced_coords, atom_symbols)
                displaced_gradient = displaced_results["gradient"]
                
                # Finite difference approximation of Hessian
                gradient_diff = (displaced_gradient - base_gradient) / delta
                
                # Fill in the Hessian matrix
                for k in range(num_atoms):
                    for l in range(3):
                        # Ensure value is finite and bounded
                        hess_val = gradient_diff[k, l]
                        if not np.isfinite(hess_val):
                            hess_val = 0.0
                        # Cap extremely large values
                        hess_val = self.safe_value(hess_val)
                        hessian[3*i+j, 3*k+l] = hess_val
        
        # Ensure Hessian is symmetric
        hessian = 0.5 * (hessian + hessian.T)
        
        return {"hessian": hessian}

class Calculation:
    """
    High-level wrapper for Tersoff calculations.
    """
    def __init__(self, **kwarg):
        UVL = UnitValueLib()
        self.bohr2angstroms = UVL.bohr2angstroms
        self.atom_symbol = kwarg.get("atom_symbol", None) # Can be None initially
        self.FC_COUNT = kwarg.get("FC_COUNT", -1)
        self.Model_hess = kwarg.get("Model_hess")
        self.hessian_flag = kwarg.get("hessian_flag", False)
        self.calculator = TersoffCore()
        self.energy = None
        self.gradient = None
        self.coordinate = None

    def exact_hessian(self, element_list, positions_bohr):
        """Calculates and projects the Hessian."""
        results = self.calculator.calculate_hessian(positions_bohr, self.atom_symbol)
        exact_hess = results['hessian']

        self.Model_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            exact_hess, element_list, positions_bohr, display_eigval=False
        )

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, geom_num_list=None):
        """
        Executes a Tersoff single point calculation, reading from a file
        or using a provided geometry.
        """
        finish_frag = False
        e, g, positions_bohr = None, None, None

        try:
            os.makedirs(file_directory, exist_ok=True)
        except (OSError, TypeError): # TypeError if file_directory is None
            pass

        if file_directory is None:
            file_list = ["dummy"] # To run the loop once for geom_num_list
        else:
            file_list = sorted(glob.glob(os.path.join(file_directory, "*_[0-9].xyz")))
            if not file_list and geom_num_list is None:
                 raise FileNotFoundError(f"No XYZ files found in {file_directory}")

        for num, input_file in enumerate(file_list):
            try:
                positions_angstrom = None
                if geom_num_list is None:
                    positions_angstrom, read_elements, _ = xyz2list(input_file, electric_charge_and_multiplicity)
                    # **FIX**: Check if element_list is None or empty.
                    if element_list is None or len(element_list) == 0:
                         element_list = read_elements
                else:
                    positions_angstrom = geom_num_list

                if self.atom_symbol is None:
                    if element_list is None or len(element_list) == 0:
                        raise ValueError("Element list is empty. Cannot determine atom symbol.")
                    first_element = element_list[0]
                    if type(element_list[0]) is not str:
                        first_element = []
                        for i in range(len(element_list)):
                            first_element.append(number_element(element_list[i]))

                    self.atom_symbol = first_element
                    print(f"Atom symbol set to '{self.atom_symbol}' based on the first structure.")

                positions_bohr = np.array(positions_angstrom, dtype="float64") / self.bohr2angstroms
                results = self.calculator.calculate_energy_and_gradient(positions_bohr, self.atom_symbol)
                e = results['energy']
                g = results['gradient']

                if self.FC_COUNT == -1 or isinstance(iter, str):
                    if self.hessian_flag:
                        self.exact_hessian(element_list, positions_bohr)
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_list, positions_bohr)
                
                break

            except Exception as error:
                print(f"Error during Tersoff calculation for {input_file}: {error}")
                finish_frag = True
                return np.array([0]), np.array([0]), np.array([0]), finish_frag

        self.energy = e
        self.gradient = g
        self.coordinate = positions_bohr
        return e, g, positions_bohr, finish_frag

class CalculationEngine(ABC):
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        pass
    def _get_file_list(self, file_directory):
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        try:
            if hasattr(config, 'save_pict') and config.save_pict:
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                # Fix for overflow in square calculation
                gradient_norm_list = []
                for g in gradient_list:
                    if g.size > 0:
                        # Safely calculate gradient norm to avoid overflow
                        g_squared = np.square(np.clip(g, -1e3, 1e3))  # Clip before squaring
                        mean_squared = g_squared.mean()
                        gradient_norm_list.append(np.sqrt(mean_squared))
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")

class TersoffEngine(CalculationEngine):
    def __init__(self, atom_symbol=None):
        super().__init__()
        self.atom_symbol = atom_symbol
        self.calculator = TersoffCore()
        self.bohr2angstroms = UnitValueLib().bohr2angstroms

    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list, energy_list, geometry_num_list, num_list = [], [], [], []
        delete_pre_total_velocity = []
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)

        if not file_list:
            print(f"No XYZ files found in directory: {file_directory}")
            return np.array([]), np.array([]), np.array([]), pre_total_velocity

        for num, input_file in enumerate(file_list):
            try:
                print(f"Processing file: {input_file}")
                positions_angstrom, element_list, _ = xyz2list(input_file, None)
                
                if self.atom_symbol is None:
                    if element_list is None or len(element_list) == 0:
                         raise ValueError("Element list from file is empty.")
                    self.atom_symbol = element_list
                    print(f"Engine atom symbols set based on the first file.")
                
                positions_bohr = np.array(positions_angstrom, dtype='float64').reshape(-1, 3) / self.bohr2angstroms
                results = self.calculator.calculate_energy_and_gradient(positions_bohr, self.atom_symbol)
                
                energy_list.append(results['energy'])
                gradient_list.append(results['gradient'])
                geometry_num_list.append(positions_angstrom)
                num_list.append(num)
            except Exception as error:
                print(f"Error processing {input_file}: {error}")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)

        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)
        if optimize_num != 0 and len(pre_total_velocity) > 0 and delete_pre_total_velocity:
            pre_total_velocity = np.delete(np.array(pre_total_velocity), delete_pre_total_velocity, axis=0)
        return (np.array(energy_list, dtype='float64'),
                np.array(gradient_list, dtype='float64'),
                np.array(geometry_num_list, dtype='float64'),
                pre_total_velocity)