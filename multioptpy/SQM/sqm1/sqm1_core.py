import numpy as np

from multioptpy.SQM.sqm1.sqm1_data import SQM1Data
from multioptpy.SQM.sqm1.dftd4 import D4DispersionModel, D4Parameters
from multioptpy.Parameters.unit_values import UnitValueLib
from multioptpy.Parameters.uff import UFF_VDW_distance_lib, UFF_VDW_well_depth_lib
from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.Parameters.atomic_number import number_element
from multioptpy.SQM.sqm1.srb_energy import SRBEnergyCalculator
from multioptpy.SQM.sqm1.eht_energy import EHTEnergyCalculator
from multioptpy.SQM.sqm1.eeq_energy import EEQEnergyCalculator
from multioptpy.SQM.sqm1.repulsion_energy import RepulsionEnergyCalculator
from multioptpy.SQM.sqm1.d4_dispersion_energy import D4DispersionEnergyCalculator
 
"""
Experimental semiempirical electronic structure approach inspired by GFN0-xTB calculator implementation

This module provides the main calculator class for performing calculations
with an experimental semiempirical electronic structure approach inspired by GFN0-xTB in Python.
ref.: https://doi.org/10.26434/chemrxiv.8326202.v1

This method is under active development. Thus, you must not use it for any user applications.
"""


class SQM1Calculator:
    """
    Main calculator class for an experimental semiempirical electronic structure approach inspired by GFN0-xTB
    
    This class implements the core functionality of an experimental semiempirical electronic structure approach inspired by GFN0-xTB,
    including energy calculations, gradient computations, and property
    evaluations.
    """
    
    def __init__(self, data=None):
        """
        Initialize the SQM1 calculator
        
        Parameters:
        -----------
        data : SQM1Data, optional
            Parametrization data. If None, default SQM1Data will be used.
        """
        self.data = data if data is not None else SQM1Data()
        self._setup_constants()
        
        # Initialize energy component calculators
        self.eht_calculator = EHTEnergyCalculator(self.data)
        self.eeq_calculator = EEQEnergyCalculator(self.data)
        self.repulsion_calculator = RepulsionEnergyCalculator(self.data)
        self.srb_calculator = SRBEnergyCalculator(self.data)
        self.d4_calculator = D4DispersionEnergyCalculator(self.data)
        
        # Initialize D4 dispersion model (for compatibility)
        self.d4_model = D4DispersionModel()
        self.d4_params = D4Parameters(
            s6=1.0, s8=2.7, s10=0.0, 
            a1=0.4, a2=5.0, s9=1.0, alp=16
        )
        # vdW control parameters
        self.enable_vdw = False
        self.vdw_scale_rep = 1.0  # scaling of repulsive r^-12 term
        self.vdw_scale_attr = 1.0  # scaling of attractive r^-6 term (multiplies the 2 factor)
        self.vdw_scale_14 = 0.5  # scaling applied to 1-4 interactions (path length 3)
        self.vdw_min_path_exclude = 3  # exclude path lengths < this from vdW (1-2,1-3 already hard excluded)
        self.vdw_debug = False  # set True to print per-pair contributions
        self.exclude_vdw_14 = False  # if True, 1-4 pairs are completely excluded instead of scaled
        
        # Harmonic bond restraint parameters
        self.enable_harmonic_bonds = False
        self.harmonic_bond_k = 2.0  # force constant in Hartree/Bohr^2
        self._reference_bond_distances = None  # will store initial bonded distances
    
    def _setup_constants(self):
        """Setup physical and mathematical constants"""
        # Physical constants
        self.BOHR_TO_ANG = UnitValueLib().bohr2angstroms
        self.HARTREE_TO_EV = UnitValueLib().hartree2eV
        self.HARTREE_TO_KCALMOL = UnitValueLib().hartree2kcalmol

        # Mathematical constants
        self.PI = np.pi
        self.SQRT_PI = np.sqrt(np.pi)
        
        # Cutoff parameters
        self.CN_CUTOFF = 40.0  # Coordination number cutoff in Bohr
        self.DISP_CUTOFF = 50.0  # Dispersion cutoff in Bohr
        self.VDW_CUTOFF = 30.0  # vdW cutoff in Bohr
        self._bc = BondConnectivity()

    # ================= Van der Waals (UFF-based Lennard-Jones) =================
    def _build_vdw_pairs(self, coords, atomic_numbers):
        """Build non-bonded vdW pairs with topology-based filtering.

        Exclusions / scaling:
          - Exclude 1-2 (bonded) and 1-3 pairs completely.
          - Detect 1-4 pairs (path length = 3 bonds) and apply scaling (vdw_scale_14).
          - Optionally exclude additional close path lengths via self.vdw_min_path_exclude.

        Returns list of (i, j, r_bohr, is14)
        """
        nat = len(atomic_numbers)
        coord_bohr = coords 
        symbols = [number_element(int(z)) for z in atomic_numbers]
        # Binary bond adjacency (0/1)
        conn = self._bc.bond_connect_matrix(symbols, coord_bohr)

        # Build adjacency list
        adjacency = [list(np.where(conn[i] == 1)[0]) for i in range(nat)]

        # Compute shortest path lengths up to 4 using multi-source BFS per atom (truncate beyond 4)
        # path_len[i,j] = 1 (bond), 2, 3, 4, or large (unconnected within cutoff)
        max_track = 4
        path_len = np.full((nat, nat), np.inf)
        for s in range(nat):
            path_len[s, s] = 0
            frontier = [s]
            depth = 0
            visited = {s}
            while frontier and depth < max_track:
                next_frontier = []
                depth += 1
                for v in frontier:
                    for nb in adjacency[v]:
                        if nb in visited:
                            continue
                        visited.add(nb)
                        path_len[s, nb] = depth
                        next_frontier.append(nb)
                frontier = next_frontier

        # Collect pairs: exclude path < vdw_min_path_exclude (default 3) meaning exclude 1-2 (1), 1-3 (2)
        # 1-4 (3) may be scaled or excluded depending on flag. If vdw_min_path_exclude changes, logic adapts.
        exclude_threshold = max(1, self.vdw_min_path_exclude)  # safety
        pairs = []
        for i in range(nat):
            for j in range(i + 1, nat):
                pl = path_len[i, j]
                if pl <= 0:  # same atom
                    continue
                if pl < exclude_threshold:
                    # Exclude 1-2 and 1-3 for default threshold=3
                    continue
                is14 = (pl == 3)
                if is14 and self.exclude_vdw_14:
                    continue
                r_vec = coord_bohr[i] - coord_bohr[j]
                r = np.linalg.norm(r_vec)
                if r > self.VDW_CUTOFF or r < 1e-6:
                    continue
                pairs.append((i, j, r, is14))
       
        return pairs

    # ================= Harmonic Bond Restraints =================
    def _build_bond_pairs(self, coords, atomic_numbers):
        """Build list of bonded atom pairs (1-2 connections) for harmonic restraints.
        
        Returns list of (i, j, r_bohr) for all bonded pairs.
        """
        nat = len(atomic_numbers)
        coord_bohr = coords / self.BOHR_TO_ANG
        symbols = [number_element(int(z)) for z in atomic_numbers]
        conn = self._bc.bond_connect_matrix(symbols, coord_bohr)
        
        pairs = []
        for i in range(nat):
            for j in range(i + 1, nat):
                if conn[i, j] == 1:  # bonded
                    r_vec = coord_bohr[i] - coord_bohr[j]
                    r = np.linalg.norm(r_vec)
                    pairs.append((i, j, r))
        return pairs

    def _cache_reference_bond_distances(self, coords, atomic_numbers):
        """Store reference bond distances from initial geometry for harmonic restraints."""
        if self._reference_bond_distances is not None:
            return  # already cached
        
        bond_pairs = self._build_bond_pairs(coords, atomic_numbers)
        self._reference_bond_distances = {}
        for i, j, r in bond_pairs:
            self._reference_bond_distances[(i, j)] = r

    def calculate_harmonic_bond_energy(self, coords, atomic_numbers):
        """Calculate harmonic bond restraint energy: E = 0.5 * k * (r - r0)^2
        
        Uses initial geometry bond distances as equilibrium lengths (r0).
        Returns 0 if harmonic bonds disabled.
        """
        if not self.enable_harmonic_bonds:
            return 0.0
            
        # Cache reference distances on first call
        self._cache_reference_bond_distances(coords, atomic_numbers)
        
        energy = 0.0
        bond_pairs = self._build_bond_pairs(coords, atomic_numbers)
        
        for i, j, r in bond_pairs:
            r0 = self._reference_bond_distances.get((i, j), r)  # fallback to current if not found
            delta_r = (r - r0) / self.BOHR_TO_ANG  # convert to Bohr
            energy += 0.5 * self.harmonic_bond_k * delta_r**2
        
        return energy

    def _calculate_harmonic_bond_gradient(self, coords, atomic_numbers):
        """Analytical gradient of harmonic bond restraints (Hartree/Bohr)."""
        if not self.enable_harmonic_bonds:
            return np.zeros((len(atomic_numbers), 3))
            
        # Cache reference distances
        self._cache_reference_bond_distances(coords, atomic_numbers)
        
        nat = len(atomic_numbers)
        grad = np.zeros((nat, 3))
        coord_bohr = coords / self.BOHR_TO_ANG
        bond_pairs = self._build_bond_pairs(coords, atomic_numbers)
        
        for i, j, r in bond_pairs:
            r0 = self._reference_bond_distances.get((i, j), r)
            if r < 1e-12:
                continue
            delta_r = (r - r0) / self.BOHR_TO_ANG  # convert to Bohr
            r_vec = coord_bohr[i] - coord_bohr[j]
            unit = r_vec / r
            
            # dE/dr = k * (r - r0), force = -dE/dr * unit_vector
            force_magnitude = -self.harmonic_bond_k * delta_r
            force_vec = force_magnitude * unit
            
            grad[i] += force_vec
            grad[j] -= force_vec
        
        return grad

    def _calculate_harmonic_bond_hessian(self, coords, atomic_numbers):
        """Analytical Hessian of harmonic bond restraints."""
        if not self.enable_harmonic_bonds:
            return np.zeros((3 * len(atomic_numbers), 3 * len(atomic_numbers)))
            
        # Cache reference distances
        self._cache_reference_bond_distances(coords, atomic_numbers)
        
        nat = len(atomic_numbers)
        hess = np.zeros((3 * nat, 3 * nat))
        coord_bohr = coords / self.BOHR_TO_ANG
        bond_pairs = self._build_bond_pairs(coords, atomic_numbers)
        
        for i, j, r in bond_pairs:
            r0 = self._reference_bond_distances.get((i, j), r)
            if r < 1e-12:
                continue
            
            r_vec = coord_bohr[i] - coord_bohr[j]
            unit = r_vec / r
            
            # For harmonic potential: d2E/dr2 = k, dE/dr = k*(r-r0)
            d2Edr2 = self.harmonic_bond_k
            dEdr = self.harmonic_bond_k * (r - r0)
            
            # Hessian block: (d2E/dr2 - dE/dr/r) * u*u^T + (dE/dr/r) * I
            outer = np.outer(unit, unit)
            I = np.identity(3)
            block = (d2Edr2 - dEdr / r) * outer + (dEdr / r) * I
            
            for a in range(3):
                for b in range(3):
                    hess[3 * i + a, 3 * i + b] += block[a, b]
                    hess[3 * j + a, 3 * j + b] += block[a, b]
                    hess[3 * i + a, 3 * j + b] -= block[a, b]
                    hess[3 * j + a, 3 * i + b] -= block[a, b]
        
        return hess

    def calculate_vdw_energy(self, coords, atomic_numbers):
        """Calculate vdW (12-6) energy with topology scaling.

        E_ij = s_topo * [ s_rep * eps_ij * (R_ij/r)^12 - 2 s_attr * eps_ij * (R_ij/r)^6 ]
        where s_topo = 1 for normal pairs, = vdw_scale_14 for 1-4 pairs.

        Returns 0 if vdW disabled.
        """
        if not self.enable_vdw:
            return 0.0
        energy = 0.0
        pairs = self._build_vdw_pairs(coords, atomic_numbers)
        if not pairs:
            return 0.0
        debug_lines = [] if self.vdw_debug else None
        for i, j, r, is14 in pairs:
            zi = int(atomic_numbers[i]); zj = int(atomic_numbers[j])
            sym_i = number_element(zi); sym_j = number_element(zj)
            R_i = UFF_VDW_distance_lib(sym_i)
            R_j = UFF_VDW_distance_lib(sym_j)
            eps_i = UFF_VDW_well_depth_lib(sym_i)
            eps_j = UFF_VDW_well_depth_lib(sym_j)
            # UFF combining rule: sigma_ij (here R_ij) is the sum R_i + R_j (parameters already Bohr)
            R_ij = (R_i + R_j) / 2.0
            eps_ij = np.sqrt(eps_i * eps_j)
            inv = R_ij / r
            inv6 = inv ** 6
            inv12 = inv6 ** 2
            topo_scale = self.vdw_scale_14 if is14 else 1.0
            term_rep = self.vdw_scale_rep * inv12
            term_attr = self.vdw_scale_attr * inv6
            e_ij = topo_scale * eps_ij * (term_rep - 2.0 * term_attr)
            energy += e_ij
            if debug_lines is not None:
                debug_lines.append(f"vdW pair ({i},{j}) r={r:.3f}Bohr is14={is14} eps={eps_ij:.4e} E={e_ij:.4e}")
        if debug_lines is not None:
            print("\n".join(debug_lines))
        return energy

    def _calculate_vdw_gradient(self, coords, atomic_numbers):
        """Analytical vdW gradient including topology scaling (Hartree/Bohr)."""
        if not self.enable_vdw:
            return np.zeros((len(atomic_numbers), 3))
        nat = len(atomic_numbers)
        grad = np.zeros((nat, 3))
        coord_bohr = coords / self.BOHR_TO_ANG
        pairs = self._build_vdw_pairs(coords, atomic_numbers)
        for i, j, r, is14 in pairs:
            zi = int(atomic_numbers[i])
            zj = int(atomic_numbers[j])
            sym_i = number_element(zi)
            sym_j = number_element(zj)
            R_i = UFF_VDW_distance_lib(sym_i)
            R_j = UFF_VDW_distance_lib(sym_j)
            eps_i = UFF_VDW_well_depth_lib(sym_i)
            eps_j = UFF_VDW_well_depth_lib(sym_j)
            R_ij = (R_i + R_j) / 2.0
            eps_ij = np.sqrt(eps_i * eps_j)
            r_vec = coord_bohr[i] - coord_bohr[j]
            if r < 1e-12:
                continue
            inv = R_ij / r
            inv2 = inv * inv
            inv6 = inv2 ** 3
            inv12 = inv6 ** 2
            topo_scale = self.vdw_scale_14 if is14 else 1.0
            # With scaling: E = topo * eps * ( s_rep*inv12 - 2 s_attr*inv6 )
            # dE/dr = topo*eps*( s_rep*(-12)inv12 + (-2 s_attr)*(-6)inv6 )/r
            dEdr = topo_scale * eps_ij * ( self.vdw_scale_rep * (-12.0) * inv12 + 12.0 * self.vdw_scale_attr * inv6 ) / r
            unit = r_vec / r
            g_vec = dEdr * unit
            grad[i] += g_vec
            grad[j] -= g_vec
        return grad

    def _calculate_vdw_hessian(self, coords, atomic_numbers):
        """Analytical vdW Hessian (approx) including topology scaling.

        Uses standard central-force Hessian formula with scaled first/second radial derivatives.
        """
        if not self.enable_vdw:
            return np.zeros((3 * len(atomic_numbers), 3 * len(atomic_numbers)))
        nat = len(atomic_numbers)
        hess = np.zeros((3 * nat, 3 * nat))
        coord_bohr = coords / self.BOHR_TO_ANG
        pairs = self._build_vdw_pairs(coords, atomic_numbers)
        for i, j, r, is14 in pairs:
            zi = int(atomic_numbers[i]); zj = int(atomic_numbers[j])
            sym_i = number_element(zi); sym_j = number_element(zj)
            R_i = UFF_VDW_distance_lib(sym_i)
            R_j = UFF_VDW_distance_lib(sym_j)
            eps_i = UFF_VDW_well_depth_lib(sym_i)
            eps_j = UFF_VDW_well_depth_lib(sym_j)
            R_ij = (R_i + R_j) / 2.0
            eps_ij = np.sqrt(eps_i * eps_j)
            r_vec = coord_bohr[i] - coord_bohr[j]
            if r < 1e-12:
                continue
            inv = R_ij / r
            inv2 = inv * inv
            inv6 = inv2 ** 3
            inv12 = inv6 ** 2
            topo_scale = self.vdw_scale_14 if is14 else 1.0
            # dE/dr with scaling
            dEdr = topo_scale * eps_ij * ( self.vdw_scale_rep * (-12.0) * inv12 + 12.0 * self.vdw_scale_attr * inv6 ) / r
            # Second derivative: differentiate dE/dr expression
            # Starting from E = topo*eps*( s_rep*R^12 r^-12 - 2 s_attr * R^6 r^-6 )
            # d2E/dr2 = topo*eps*( s_rep*156 R^12 r^-14 - 84 s_attr R^6 r^-8 )
            R6 = R_ij ** 6
            R12 = R6 * R6
            d2Edr2 = topo_scale * eps_ij * ( self.vdw_scale_rep * 156.0 * R12 / (r ** 14) - 84.0 * self.vdw_scale_attr * R6 / (r ** 8) )
            unit = r_vec / r
            outer = np.outer(unit, unit)
            I = np.identity(3)
            block = (d2Edr2 - dEdr / r) * outer + (dEdr / r) * I
            for a in range(3):
                for b in range(3):
                    hess[3 * i + a, 3 * i + b] += block[a, b]
                    hess[3 * j + a, 3 * j + b] += block[a, b]
                    hess[3 * i + a, 3 * j + b] -= block[a, b]
                    hess[3 * j + a, 3 * i + b] -= block[a, b]
        return hess
    
    def calculate_coordination_numbers(self, 
                                     coords, 
                                     atomic_numbers,
                                     cutoff=None):
        """
        Calculate coordination numbers for all atoms
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        cutoff : float, optional
            Cutoff distance for CN calculation
            
        Returns:
        --------
        cn : np.ndarray, shape (n_atoms,)
            Coordination numbers
        """
        n_atoms = len(atomic_numbers)
        if cutoff is None:
            cutoff = self.CN_CUTOFF
        
        # Ensure cutoff is a scalar
        if hasattr(cutoff, '__len__'):
            cutoff = float(cutoff[0]) if len(cutoff) > 0 else 10.0
        else:
            cutoff = float(cutoff)
        
        cn = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate distance
                r_ij = np.linalg.norm(coords[i] - coords[j])
                
                # Ensure r_ij is a scalar
                if hasattr(r_ij, '__len__'):
                    r_ij = float(r_ij[0]) if len(r_ij) > 0 else 0.0
                else:
                    r_ij = float(r_ij)
                
                # Safe comparison with scalar values
                cutoff_val = float(cutoff)
                if r_ij > cutoff_val:
                    continue
                
                # Get covalent radii
                r_cov_i = self.data.atomic_radii[atomic_numbers[i] - 1]
                r_cov_j = self.data.atomic_radii[atomic_numbers[j] - 1]
                r_cov = r_cov_i + r_cov_j
                
                # Calculate coordination number contribution
                if r_ij < r_cov * 4.0:  # Only consider reasonable distances
                    cn_ij = 1.0 / (1.0 + np.exp(-16.0 * (r_cov / r_ij - 1.0)))
                    cn[i] += cn_ij
                    cn[j] += cn_ij
        
        return cn
    
    def _calculate_coordination_number_gradient(self, 
                                              coords, 
                                              atomic_numbers,
                                              cutoff=None):
        """
        Calculate analytical gradient of coordination numbers
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        cutoff : float, optional
            Cutoff distance for CN calculation
            
        Returns:
        --------
        cn_gradient : np.ndarray, shape (n_atoms, n_atoms, 3)
            Coordination number gradients: ∂CN_i/∂R_j
        """
        n_atoms = len(atomic_numbers)
        if cutoff is None:
            cutoff = self.CN_CUTOFF
        
        # Ensure cutoff is a scalar
        if hasattr(cutoff, '__len__'):
            cutoff = float(cutoff[0]) if len(cutoff) > 0 else 10.0
        else:
            cutoff = float(cutoff)
        
        cn_gradient = np.zeros((n_atoms, n_atoms, 3))
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                # Calculate distance vector and distance
                r_vec = coords[i] - coords[j]
                r_ij = np.linalg.norm(r_vec)
                
                # Ensure r_ij is a scalar
                if hasattr(r_ij, '__len__'):
                    r_ij = float(r_ij[0]) if len(r_ij) > 0 else 0.0
                else:
                    r_ij = float(r_ij)
                
                # Safe comparison with scalar values
                cutoff_val = float(cutoff)
                if r_ij > cutoff_val or r_ij < 1e-8:
                    continue
                
                # Get covalent radii
                r_cov_i = self.data.atomic_radii[atomic_numbers[i] - 1]
                r_cov_j = self.data.atomic_radii[atomic_numbers[j] - 1]
                r_cov = r_cov_i + r_cov_j
                
                # Calculate coordination number contribution and its derivative
                if r_ij < r_cov * 4.0:  # Only consider reasonable distances
                    # CN_ij = 1 / (1 + exp(-16 * (r_cov/r_ij - 1)))
                    # Let x = r_cov/r_ij - 1, then CN_ij = 1 / (1 + exp(-16*x))
                    # d(CN_ij)/dr_ij = 16 * r_cov/r_ij^2 * exp(-16*x) / (1 + exp(-16*x))^2
                    
                    x = r_cov / r_ij - 1.0
                    exp_term = np.exp(-16.0 * x)
                    cn_ij = 1.0 / (1.0 + exp_term)
                    
                    # Derivative of CN_ij with respect to r_ij
                    dcn_dr = 16.0 * r_cov / (r_ij * r_ij) * exp_term / ((1.0 + exp_term)**2)
                    
                    # Unit vector from j to i
                    r_unit = r_vec / r_ij
                    
                    # Gradient of CN_i with respect to coordinates of atom j
                    # ∂CN_i/∂R_j = (∂CN_i/∂r_ij) * (∂r_ij/∂R_j)
                    # Since r_ij = |R_i - R_j|, we have ∂r_ij/∂R_j = -(R_i - R_j)/r_ij
                    cn_gradient[i, j, :] = -dcn_dr * r_unit
                    
                    # Self-contribution: ∂CN_i/∂R_i
                    cn_gradient[i, i, :] += dcn_dr * r_unit
        
        return cn_gradient
    
    def _calculate_charge_gradient(self, 
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
                
                charges_plus = self.calculate_charges_eeq(coords_plus, atomic_numbers)
                charges_minus = self.calculate_charges_eeq(coords_minus, atomic_numbers)
                
                # Numerical derivative
                charge_gradients[:, j, k] = (charges_plus - charges_minus) / (2.0 * delta)
        
        return charge_gradients
    

    
    def calculate_total_energy(self,
                              coords,
                              atomic_numbers,
                              total_charge=0.0):
        """
        Calculate total energy for the experimental semiempirical approach inspired by GFN0-xTB
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
            
        Returns:
        --------
        results : dict
            Dictionary containing energy components and total energy
        """
        # Calculate charges and coordination numbers  
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        charges = self.eeq_calculator.calculate_charges_eeq(coords, atomic_numbers, cn, total_charge)
        
        # Calculate energy components using specialized calculators
        e_electronic = self.eht_calculator.calculate_energy(coords, atomic_numbers, charges)
        e_eeq = self.eeq_calculator.calculate_energy(coords, atomic_numbers, charges)
        e_repulsion = self.repulsion_calculator.calculate_energy(coords, atomic_numbers)
        e_dispersion = self.d4_calculator.calculate_energy(coords, atomic_numbers, charges)
        e_srb = self.srb_calculator.calculate_energy(coords, atomic_numbers)
        
        # Legacy energy components (vdW and harmonic bonds)
        e_vdw = self.calculate_vdw_energy(coords, atomic_numbers)
        e_harmonic_bonds = self.calculate_harmonic_bond_energy(coords, atomic_numbers)
        
        # Total energy
        e_total = e_electronic + e_eeq + e_repulsion + e_dispersion + e_srb + e_harmonic_bonds
        if self.enable_vdw:
            e_total += e_vdw

        return {
            'total': e_total,
            'electronic': e_electronic,
            'eeq': e_eeq,
            'repulsion': e_repulsion,
            'dispersion': e_dispersion,
            'srb': e_srb,
            'vdw': e_vdw,
            'harmonic_bonds': e_harmonic_bonds,
            'charges': charges
        }
    
    def calculate_charges_eeq(self, 
                             coords,
                             atomic_numbers,
                             total_charge=0.0):
        """
        Calculate atomic charges using electronegativity equilibration (EEQ)
        
        This is a compatibility wrapper around the EEQ calculator component.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
            
        Returns:
        --------
        charges : np.ndarray, shape (n_atoms,)
            Atomic partial charges
        """
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        return self.eeq_calculator.calculate_charges_eeq(coords, atomic_numbers, cn, total_charge)
    
    def calculate_density_matrix(self,
                               coords,
                               atomic_numbers,
                               total_charge=0.0):
        """
    Calculate density matrix for the experimental semiempirical approach inspired by GFN0-xTB
        
        This implements a simplified density matrix calculation consistent with
    the experimental semiempirical approach inspired by GFN0-xTB. The density matrix is constructed
        using a minimal basis set model with atomic overlap considerations.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
            
        Returns:
        --------
        density_matrix : np.ndarray, shape (n_atoms, n_atoms)
            Density matrix in atomic orbital basis
        """
        n_atoms = len(atomic_numbers)
        
        # Calculate charges and coordination numbers
        charges = self.calculate_charges_eeq(coords, atomic_numbers, total_charge)
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        
        # Initialize density matrix
        density_matrix = np.zeros((n_atoms, n_atoms))
        
        # Calculate total number of electrons
        total_electrons = np.sum(atomic_numbers) - total_charge
        
    # Calculate orbital occupation numbers (simplified approach)
    # For the experimental semiempirical approach inspired by GFN0-xTB, we use a simplified model based on atomic populations
        for i in range(n_atoms):
            z_i = atomic_numbers[i]
            q_i = charges[i]
            
            # Effective electron population on atom i
            n_electrons_i = z_i - q_i
            
            # Diagonal element: self-interaction density
            density_matrix[i, i] = n_electrons_i / 2.0  # Factor of 2 for closed shell
            
            # Off-diagonal elements: bonding density based on overlap and CN
            for j in range(i + 1, n_atoms):
                z_j = atomic_numbers[j]
                q_j = charges[j]
                n_electrons_j = z_j - q_j
                
                # Distance-dependent overlap
                r_ij = np.linalg.norm(coords[i] - coords[j]) / self.BOHR_TO_ANG
                
                # Get covalent radii for overlap estimation
                r_cov_i = self.data.atomic_radii[atomic_numbers[i] - 1]
                r_cov_j = self.data.atomic_radii[atomic_numbers[j] - 1]
                r_cov = (r_cov_i + r_cov_j) / self.BOHR_TO_ANG
                
                if r_ij < 3.0 * r_cov:  # Only consider bonded atoms
                    # Overlap-based bonding density
                    overlap = np.exp(-1.5 * (r_ij / r_cov - 1.0))
                    
                    # Bond order based on coordination numbers and atomic sizes
                    bond_strength = overlap * np.sqrt(n_electrons_i * n_electrons_j) / (cn[i] + cn[j] + 1.0)
                    
                    density_matrix[i, j] = bond_strength * 0.1  # Scale factor for semi-empirical method
                    density_matrix[j, i] = density_matrix[i, j]  # Symmetry
        
        return density_matrix
    
    def calculate_density_matrix_derivative(self,
                                          coords: np.ndarray,
                                          atomic_numbers: np.ndarray,
                                          total_charge: float = 0.0,
                                          delta: float = 1e-5) -> np.ndarray:
        """
        Calculate derivative of density matrix with respect to nuclear coordinates
        
        Uses numerical differentiation to calculate accurate derivatives of the 
        density matrix elements with respect to atomic positions. This provides
        reliable derivatives for testing and validation purposes.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        delta : float, optional
            Finite difference step size (default: 1e-5)
            
        Returns:
        --------
        dP_dR : np.ndarray, shape (n_atoms, n_atoms, n_atoms, 3)
            Density matrix derivative: dP_dR[μ, ν, atom, coord] = ∂P_μν/∂R_atom,coord
        """
        n_atoms = len(atomic_numbers)
        dP_dR = np.zeros((n_atoms, n_atoms, n_atoms, 3))
        
        # Calculate reference density matrix
        P_ref = self.calculate_density_matrix(coords, atomic_numbers, total_charge)
        
        # For each atom and coordinate
        for atom_idx in range(n_atoms):
            for coord_idx in range(3):
                # Forward step
                coords_plus = coords.copy()
                coords_plus[atom_idx, coord_idx] += delta
                P_plus = self.calculate_density_matrix(coords_plus, atomic_numbers, total_charge)
                
                # Backward step
                coords_minus = coords.copy()
                coords_minus[atom_idx, coord_idx] -= delta
                P_minus = self.calculate_density_matrix(coords_minus, atomic_numbers, total_charge)
                
                # Central difference
                dP_dR[:, :, atom_idx, coord_idx] = (P_plus - P_minus) / (2.0 * delta)
        
        return dP_dR
    
    def calculate_density_matrix_derivative_analytical(self,
                                                     coords: np.ndarray,
                                                     atomic_numbers: np.ndarray,
                                                     total_charge: float = 0.0) -> np.ndarray:
        """
        Calculate analytical derivative of density matrix with respect to nuclear coordinates
        
        This method implements analytical differentiation of the density matrix elements.
        Currently experimental and may have accuracy issues for complex molecules.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
            
        Returns:
        --------
        dP_dR : np.ndarray, shape (n_atoms, n_atoms, n_atoms, 3)
            Density matrix derivative: dP_dR[μ, ν, atom, coord] = ∂P_μν/∂R_atom,coord
        """
        # For now, fall back to numerical method for reliability
        # Future work could implement true analytical derivatives
        return self.calculate_density_matrix_derivative(coords, atomic_numbers, total_charge)
    
    def calculate_energy_and_gradient(self,
                                    coords,
                                    atomic_numbers,
                                    total_charge=0.0,
                                    gradient_method='analytical'):
        """
        Calculate both total energy and gradient in one call for efficiency
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        gradient_method : str
            Gradient calculation method: 'analytical' or 'numerical'
            
        Returns:
        --------
        results : dict
            Dictionary containing energy components, total energy, and gradient
        """
        # Calculate energy components
        energy_results = self.calculate_total_energy(coords, atomic_numbers, total_charge)
        
        # Calculate gradient
        gradient = self.calculate_gradient(coords, atomic_numbers, total_charge, method=gradient_method)
        
        # Combine results
        results = energy_results.copy()
        results['gradient'] = gradient
        results['gradient_norm'] = np.linalg.norm(gradient)
        results['max_gradient_component'] = np.max(np.abs(gradient))
        
        return results
    
    def calculate_gradient_analytical(self,
                                     coords,
                                     atomic_numbers,
                                     total_charge=0.0,
                                     multiplicity=1):
        """
        Calculate energy gradient using analytical differentiation
        
        This provides much faster and more accurate gradients than numerical
        differentiation. Implements analytical gradients for all components of the experimental semiempirical approach inspired by GFN0-xTB
        energy components:
        - Electronic energy (EHT) gradient
        - EEQ electrostatic energy gradient  
        - Repulsion energy gradient
        - Short-ranged bond (SRB) correction gradient
        - D4 dispersion energy gradient
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        multiplicity : int
            Spin multiplicity
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            Energy gradient in Hartree/Bohr
        """
        n_atoms = len(atomic_numbers)
        gradient = np.zeros((n_atoms, 3))
        
        # Calculate charges and coordination numbers for use in derivatives
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        charges = self.eeq_calculator.calculate_charges_eeq(coords, atomic_numbers, cn, total_charge)
        
        # Calculate additional derivatives needed for EHT gradient
        cn_gradients = self._calculate_coordination_number_gradient(coords, atomic_numbers, cn)
        charge_gradients = self._calculate_charge_gradient(coords, atomic_numbers, total_charge)
        
        # Calculate gradients using specialized calculators
        # 1. Electronic energy (EHT) gradient
        gradient += self.eht_calculator.calculate_gradient(
            coords, atomic_numbers, charges, multiplicity, cn, cn_gradients, charge_gradients
        )
        
        # 2. EEQ electrostatic energy gradient
        gradient += self.eeq_calculator.calculate_gradient(coords, atomic_numbers, charges)
        
        # 3. Nuclear repulsion energy gradient
        gradient += self.repulsion_calculator.calculate_gradient(coords, atomic_numbers)
        
        # 4. D4 dispersion energy gradient
        gradient += self.d4_calculator.calculate_gradient(coords, atomic_numbers, charges)
        
        # 5. Short-ranged bond correction gradient
        gradient += self.srb_calculator.calculate_gradient(coords, atomic_numbers)

        # 6. Harmonic bond restraint gradient (legacy method)
        gradient += self._calculate_harmonic_bond_gradient(coords, atomic_numbers)

        # 7. vdW gradient (legacy method) if enabled
        if self.enable_vdw:
            gradient += self._calculate_vdw_gradient(coords, atomic_numbers)
        
        return gradient
    
    
    def calculate_gradient(self,
                          coords: np.ndarray,
                          atomic_numbers: np.ndarray,
                          total_charge: float = 0.0,
                          delta: float = 1e-5,
                          method: str = 'analytical') -> np.ndarray:
        """
        Calculate energy gradient
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        delta : float
            Step size for numerical differentiation (if method='numerical')
        method : str
            Gradient calculation method: 'analytical' or 'numerical'
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            Energy gradient in Hartree/Bohr
        """
        if method.lower() == 'analytical':
            return self.calculate_gradient_analytical(coords, atomic_numbers, total_charge)
        elif method.lower() == 'numerical':
            return self.calculate_gradient_numerical(coords, atomic_numbers, total_charge, delta)
        else:
            raise ValueError(f"Unknown gradient method: {method}. Use 'analytical' or 'numerical'")
    
    def calculate_gradient_numerical(self,
                                   coords: np.ndarray,
                                   atomic_numbers: np.ndarray,
                                   total_charge: float = 0.0,
                                   delta: float = 1e-5) -> np.ndarray:
        """
        Calculate energy gradient using numerical differentiation
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        delta : float
            Step size for numerical differentiation
            
        Returns:
        --------
        gradient : np.ndarray, shape (n_atoms, 3)
            Energy gradient in Hartree/Bohr
        """
        n_atoms = len(atomic_numbers)
        gradient = np.zeros((n_atoms, 3))
        
        # Central difference approximation
        for i in range(n_atoms):
            for j in range(3):
                coords_plus = coords.copy()
                coords_minus = coords.copy()
                coords_plus[i, j] += delta
                coords_minus[i, j] -= delta
                
                e_plus = self.calculate_total_energy(coords_plus, atomic_numbers, total_charge)['total']
                e_minus = self.calculate_total_energy(coords_minus, atomic_numbers, total_charge)['total']
                
                gradient[i, j] = (e_plus - e_minus) / (2.0 * delta)
        
        # Convert to Hartree/Bohr
        gradient *= self.BOHR_TO_ANG
        
        return gradient
    
    def calculate_molecular_properties(self,
                                     coords,
                                     atomic_numbers,
                                     total_charge=0.0,
                                     include_gradient=True,
                                     gradient_method='analytical'):
        """
        Calculate various molecular properties including gradients
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        include_gradient : bool
            Whether to calculate and include gradient information
        gradient_method : str
            Gradient calculation method: 'analytical' or 'numerical'
            
        Returns:
        --------
        properties : dict
            Dictionary of calculated properties
        """
        # Basic energy calculation
        if include_gradient:
            energy_results = self.calculate_energy_and_gradient(
                coords, atomic_numbers, total_charge, gradient_method)
        else:
            energy_results = self.calculate_total_energy(coords, atomic_numbers, total_charge)
        
        # Coordination numbers
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        
        # Molecular properties
        n_atoms = len(atomic_numbers)
        total_electrons = sum(atomic_numbers) - total_charge
        
        # Center of mass
        masses = np.array([atomic_numbers[i] for i in range(n_atoms)])  # Approximate masses
        center_of_mass = np.average(coords, axis=0, weights=masses)
        
        # Radius of gyration
        r_squared = np.sum((coords - center_of_mass)**2 * masses[:, np.newaxis]) / np.sum(masses)
        radius_of_gyration = np.sqrt(r_squared)
        
        properties = {
            'energy_total': energy_results['total'],
            'energy_components': {
                'electronic': energy_results['electronic'],
                'repulsion': energy_results['repulsion'],
                'dispersion': energy_results['dispersion']
            },
            'charges': energy_results['charges'],
            'coordination_numbers': cn,
            'n_atoms': n_atoms,
            'n_electrons': total_electrons,
            'center_of_mass': center_of_mass,
            'radius_of_gyration': radius_of_gyration,
            'total_charge': total_charge
        }
        
        # Add gradient information if calculated
        if include_gradient and 'gradient' in energy_results:
            properties['gradient'] = energy_results['gradient']
            properties['gradient_norm'] = energy_results['gradient_norm']
            properties['max_gradient_component'] = energy_results['max_gradient_component']
        
        return properties
    
    def calculate_total_energy(self,
                              coords,
                              atomic_numbers,
                              total_charge=0.0,
                              multiplicity=1):
        """
    Calculate total energy with all 5 components for the experimental semiempirical approach inspired by GFN0-xTB
        and support for formal charge and spin multiplicity.
        
    The 5 energy components of the experimental semiempirical approach inspired by GFN0-xTB are:
        1. Electronic energy (EHT - Extended Hückel Theory)
        2. Electrostatic energy (EEQ - Electronegativity Equalization)  
        3. Repulsion energy (classical nuclear repulsion with damping)
        4. Dispersion energy (D4 dispersion)
        5. Short-range bond correction (SRB)
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        multiplicity : int
            Spin multiplicity (2S+1)
            
        Returns:
        --------
        results : dict
            Dictionary containing all 5 energy components and properties
        """
        # Calculate coordination numbers first
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        
        # Calculate EEQ charges with proper charge constraint
        charges = self.eeq_calculator.calculate_charges_eeq(coords, atomic_numbers, cn, total_charge)
        
        # 1. Electronic energy (EHT) - with multiplicity support
        e_electronic = self.eht_calculator.calculate_energy(
            coords, atomic_numbers, charges
        )
        
        # 2. Electrostatic energy (EEQ)
        e_electrostatic = self.eeq_calculator.calculate_energy(
            coords, atomic_numbers, charges
        )
        
        # 3. Repulsion energy
        e_repulsion = self.repulsion_calculator.calculate_energy(
            coords, atomic_numbers
        )
        
        # 4. Dispersion energy (D4)
        e_dispersion = self.d4_calculator.calculate_energy(coords, atomic_numbers, charges)
        e_vdw = self.calculate_vdw_energy(coords, atomic_numbers)
        
        # 5. Short-range bond correction (SRB)
        e_srb = self.srb_calculator.calculate_energy(coords, atomic_numbers)
        
        # Total energy
        e_total = e_electronic + e_electrostatic + e_repulsion + e_dispersion + e_srb + e_vdw
        
        # Calculate dipole moment
        dipole_moment = self._calculate_dipole_moment(coords, atomic_numbers, charges)
        
        return {
            'total': e_total,
            'electronic': e_electronic,
            'electrostatic': e_electrostatic, 
            'repulsion': e_repulsion,
            'dispersion': e_dispersion,
            'vdw': e_vdw,
            'srb': e_srb,
            'charges': charges,
            'coordination_numbers': cn,
            'dipole_moment': dipole_moment,
            'multiplicity': multiplicity
        }
    
    
    def _calculate_dipole_moment(self,
                               coords: np.ndarray,
                               atomic_numbers: np.ndarray,
                               charges: np.ndarray) -> np.ndarray:
        """
        Calculate molecular dipole moment in Debye
        """
        # Charge contribution to dipole moment
        dipole_charge = np.sum(charges[:, np.newaxis] * coords, axis=0)
        
        # Convert from e*Angstrom to Debye (1 Debye = 0.2081943 e*Angstrom)
        dipole_debye = dipole_charge / 0.2081943
        
        return dipole_debye
    
        
    def calculate_hessian(self,
                         coords: np.ndarray,
                         atomic_numbers: np.ndarray,
                         total_charge: float = 0.0,
                         method: str = 'analytical') -> np.ndarray:
        """
        Calculate energy Hessian (second derivatives)
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        method : str
            Hessian calculation method: 'analytical' or 'numerical'
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            Energy Hessian in Hartree/Bohr²
        """
        if method.lower() == 'analytical':
            return self.calculate_hessian_analytical(coords, atomic_numbers, total_charge)
        elif method.lower() == 'numerical':
            return self.calculate_hessian_numerical(coords, atomic_numbers, total_charge)
        else:
            raise ValueError(f"Unknown Hessian method: {method}. Use 'analytical' or 'numerical'")

    def calculate_hessian_numerical(self,
                                   coords: np.ndarray,
                                   atomic_numbers: np.ndarray,
                                   total_charge: float = 0.0,
                                   delta: float = 1e-5) -> np.ndarray:
        """
        Calculate energy Hessian using numerical differentiation of analytical gradients
        
        This method uses the already implemented analytical gradients to compute
        the numerical Hessian, providing a reliable reference for validating
        the analytical Hessian implementation.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        delta : float
            Step size for numerical differentiation
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            Energy Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        # Central difference approximation of gradient
        for i in range(n_atoms):
            for j in range(3):
                coord_idx = 3 * i + j
                
                # Displace coordinate forward
                coords_plus = coords.copy()
                coords_plus[i, j] += delta
                grad_plus = self.calculate_gradient_analytical(coords_plus, atomic_numbers, total_charge)
                
                # Displace coordinate backward
                coords_minus = coords.copy()
                coords_minus[i, j] -= delta
                grad_minus = self.calculate_gradient_analytical(coords_minus, atomic_numbers, total_charge)
                
                # Second derivative by finite difference
                hess_col = (grad_plus - grad_minus) / (2.0 * delta)
                
                # Convert from 3D array to 1D column
                hess_col_flat = hess_col.flatten()
                
                # Convert units from Hartree/Angstrom/Angstrom to Hartree/Bohr/Bohr
                hess_col_flat *= (self.BOHR_TO_ANG ** 2)
                
                hessian[:, coord_idx] = hess_col_flat
        
        return hessian

    def calculate_hessian_analytical(self,
                                   coords,
                                   atomic_numbers,
                                   total_charge=0.0,
                                   multiplicity=1):
        """
        Calculate energy Hessian using analytical second derivatives
        
        This method calculates the analytical Hessian by differentiating the
        gradient expressions analytically. The Hessian includes contributions from
        all components of the experimental semiempirical approach inspired by GFN0-xTB:
        - Electronic energy (EHT) Hessian
        - EEQ electrostatic energy Hessian  
        - Repulsion energy Hessian
        - D4 dispersion energy Hessian
        - Short-ranged bond (SRB) correction Hessian
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_atoms, 3)
            Atomic coordinates in Angstrom
        atomic_numbers : np.ndarray, shape (n_atoms,)
            Atomic numbers
        total_charge : float
            Total molecular charge
        multiplicity : int
            Spin multiplicity
            
        Returns:
        --------
        hessian : np.ndarray, shape (3*n_atoms, 3*n_atoms)
            Energy Hessian in Hartree/Bohr²
        """
        n_atoms = len(atomic_numbers)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))
        
        # Calculate charges and coordination numbers for use in second derivatives
        cn = self.calculate_coordination_numbers(coords, atomic_numbers)
        charges = self.eeq_calculator.calculate_charges_eeq(coords, atomic_numbers, cn, total_charge)
        
        # Calculate Hessians using specialized calculators
        # 1. Electronic energy (EHT) Hessian
        hessian += self.eht_calculator.calculate_hessian(coords, atomic_numbers, charges)
        
        # 2. EEQ electrostatic energy Hessian
        hessian += self.eeq_calculator.calculate_hessian(coords, atomic_numbers, charges)
        
        # 3. Nuclear repulsion energy Hessian
        hessian += self.repulsion_calculator.calculate_hessian(coords, atomic_numbers)
        
        # 4. D4 dispersion energy Hessian
        hessian += self.d4_calculator.calculate_hessian(coords, atomic_numbers, charges)
        
        # 5. Short-ranged bond correction Hessian
        hessian += self.srb_calculator.calculate_hessian(coords, atomic_numbers)

        # 6. Harmonic bond restraint Hessian (legacy method)
        hessian += self._calculate_harmonic_bond_hessian(coords, atomic_numbers)

        # 7. vdW Hessian (legacy method)
        if self.enable_vdw:
            hessian += self._calculate_vdw_hessian(coords, atomic_numbers)

        return hessian

    # Helper methods for analytical Hessian calculations
    
