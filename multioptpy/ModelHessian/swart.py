import numpy as np
import math

from multioptpy.Utils.calc_tools import Calculationtools

# ============================================================================
# Library / Helper Functions (Acting as multioptpy replacements for this logic)
# ============================================================================

# Periodic table (element symbols)
PERIODIC_TABLE = [
    'X',  # Dummy element at index 0
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
]

# Covalent radii in Bohr (from Pyykkö & Atsumi, Chem. Eur. J. 2009, 15, 186)
COVALENT_RADII = {
    'H': 0.59, 'He': 0.54,
    'Li': 2.43, 'Be': 1.72, 'B': 1.53, 'C': 1.40, 'N': 1.34, 'O': 1.25, 'F': 1.18, 'Ne': 1.14,
    'Na': 2.89, 'Mg': 2.53, 'Al': 2.19, 'Si': 2.10, 'P': 2.04, 'S': 1.97, 'Cl': 1.87, 'Ar': 1.82,
    'K': 3.42, 'Ca': 3.06, 'Sc': 2.85, 'Ti': 2.70, 'V': 2.55, 'Cr': 2.49, 'Mn': 2.49,
    'Fe': 2.44, 'Co': 2.38, 'Ni': 2.32, 'Cu': 2.42, 'Zn': 2.40,
    'Ga': 2.27, 'Ge': 2.19, 'As': 2.17, 'Se': 2.10, 'Br': 2.04, 'Kr': 2.06,
    'Rb': 3.70, 'Sr': 3.40, 'Y': 3.21, 'Zr': 2.98, 'Nb': 2.85, 'Mo': 2.72, 'Tc': 2.61,
    'Ru': 2.55, 'Rh': 2.51, 'Pd': 2.55, 'Ag': 2.68, 'Cd': 2.72,
    'In': 2.61, 'Sn': 2.55, 'Sb': 2.51, 'Te': 2.48, 'I': 2.44, 'Xe': 2.48,
    'Cs': 4.03, 'Ba': 3.59,
    'La': 3.34, 'Ce': 3.25, 'Pr': 3.23, 'Nd': 3.21, 'Pm': 3.19, 'Sm': 3.17, 'Eu': 3.17,
    'Gd': 3.15, 'Tb': 3.13, 'Dy': 3.13, 'Ho': 3.11, 'Er': 3.11, 'Tm': 3.09, 'Yb': 3.09, 'Lu': 3.06,
    'Hf': 2.89, 'Ta': 2.76, 'W': 2.61, 'Re': 2.49, 'Os': 2.46, 'Ir': 2.42, 'Pt': 2.42, 'Au': 2.55, 'Hg': 2.72,
    'Tl': 2.68, 'Pb': 2.68, 'Bi': 2.68, 'Po': 2.61, 'At': 2.57, 'Rn': 2.63,
}

def covalent_radii_lib(element):
    """Mimics multioptpy covalent_radii_lib"""
    elem = element.capitalize()
    if elem not in COVALENT_RADII:
        raise ValueError(f"Unknown element: {element}")
    return COVALENT_RADII[elem]

def bmat_bond(xyz: np.ndarray, i: int, j: int) -> np.ndarray:
    vec = xyz[i, :] - xyz[j, :]
    l = np.linalg.norm(vec)
    B = np.zeros(6)
    B[0:3] = vec / l
    B[3:6] = -vec / l
    return B

def bmat_angle(xyz: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    vec1 = xyz[i, :] - xyz[j, :]
    vec2 = xyz[k, :] - xyz[j, :]
    l1 = np.linalg.norm(vec1)
    l2 = np.linalg.norm(vec2)
    
    if l1 < 1e-10 or l2 < 1e-10:
        return np.zeros(9)
    
    nvec1 = vec1 / l1
    nvec2 = vec2 / l2
    
    dot_prod = np.dot(nvec1, nvec2)
    sin_theta_sq = max(1e-15, 1.0 - dot_prod**2)
    
    dl = np.zeros((2, 6))
    dl[0, 0:3] = nvec1
    dl[0, 3:6] = -nvec1
    dl[1, 0:3] = nvec2
    dl[1, 3:6] = -nvec2
    
    dnvec = np.zeros((2, 3, 6))
    for ii in range(6):
        dnvec[0, :, ii] = -nvec1 * dl[0, ii] / l1
        dnvec[1, :, ii] = -nvec2 * dl[1, ii] / l2
    
    for ii in range(3):
        dnvec[0, ii, ii] += 1.0 / l1
        dnvec[1, ii, ii] += 1.0 / l2
        dnvec[0, ii, ii + 3] -= 1.0 / l1
        dnvec[1, ii, ii + 3] -= 1.0 / l2
    
    dinprod = np.zeros(9)
    for ii in range(3):
        dinprod[ii] = np.dot(dnvec[0, :, ii], nvec2)
        dinprod[ii + 3] = np.dot(dnvec[0, :, ii + 3], nvec2) + np.dot(dnvec[1, :, ii + 3], nvec1)
        dinprod[ii + 6] = np.dot(dnvec[1, :, ii], nvec1)
    
    B = -dinprod / math.sqrt(sin_theta_sq)
    return B

def bmat_linear_angle(xyz: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    vec1 = xyz[i, :] - xyz[j, :]
    vec2 = xyz[k, :] - xyz[j, :]
    l1 = np.linalg.norm(vec1)
    l2 = np.linalg.norm(vec2)
    
    if l1 < 1e-10 or l2 < 1e-10:
        return np.zeros((2, 9))
    
    nvec1 = vec1 / l1
    nvec2 = vec2 / l2
    
    vn = np.cross(vec1, vec2)
    nvn = np.linalg.norm(vn)
    
    if nvn < 1e-15:
        vn = np.array([1.0, 0.0, 0.0])
        vn = vn - np.dot(vn, vec1) / l1**2 * vec1
        nvn = np.linalg.norm(vn)
        
        if nvn < 1e-15:
            vn = np.array([0.0, 1.0, 0.0])
            vn = vn - np.dot(vn, vec1) / l1**2 * vec1
            nvn = np.linalg.norm(vn)
    
    vn = vn / nvn
    vn2 = np.cross(vec1 - vec2, vn)
    vn2 = vn2 / np.linalg.norm(vn2)
    
    B = np.zeros((2, 9))
    
    B[1, 0:3] = vn / l1
    B[1, 6:9] = vn / l2
    B[1, 3:6] = -B[1, 0:3] - B[1, 6:9]
    
    B[0, 0:3] = vn2 / l1
    B[0, 6:9] = vn2 / l2
    B[0, 3:6] = -B[0, 0:3] - B[0, 6:9]
    
    return B

def bond_length(xyz: np.ndarray, i: int, j: int) -> float:
    return np.linalg.norm(xyz[i, :] - xyz[j, :])


# ============================================================================
# Main Class
# ============================================================================

class SwartApproxHessian:
    def __init__(self):
        # Swart's model Hessian parameters from swart.py
        # ref.: M. Swart, F. M. Bickelhaupt, Int. J. Quantum Chem., 2006, 106, 2536
        # ref.: O1NumHess paper (arXiv:2508.07544v1) Appendix B
        
        self.wthr = 0.3
        self.f = 0.12
        self.tolth = 0.2
        
        # Derived parameters
        self.eps1 = self.wthr**2
        self.eps2 = self.wthr**2 / math.exp(1)
        
        self.cart_hess = None
        return

    def screening_function(self, distance, cov_radius_sum):
        """Screening function ρ_AB = exp(1 - R_AB / (r_A + r_B))"""
        return math.exp(1.0 - distance / cov_radius_sum)
    
    def _cos_angle(self, xyz, i, j, k):
        """Calculate cos(angle i-j-k)."""
        vec1 = xyz[i, :] - xyz[j, :]
        vec2 = xyz[k, :] - xyz[j, :]
        
        l1 = np.linalg.norm(vec1)
        l2 = np.linalg.norm(vec2)
        
        if l1 < 1e-10 or l2 < 1e-10:
            return 1.0
        
        cos_theta = np.dot(vec1, vec2) / (l1 * l2)
        return np.clip(cos_theta, -1.0, 1.0)

    def swart_bond(self, coord, element_list):
        """
        Calculate bond stretching contributions.
        Paper: All atom pairs are treated as bonds, with force constant H_int = 0.35 * ρ_AB^3
        """
        N = len(coord)
        # Re-calculate screening matrix locally as per original flow, or reuse if optimized. 
        # To strictly follow algorithmic configuration of swart.py, we perform calculations as needed.
        
        for i in range(N):
            for j in range(i + 1, N):
                cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                dist = bond_length(coord, i, j)
                
                screen = self.screening_function(dist, cov_sum)
                
                # Force constant (Paper: 0.35 * ρ^3)
                H_int = 0.35 * screen**3
                
                # Wilson B matrix for bond
                B = bmat_bond(coord, i, j)
                
                # Indices for atoms i and j
                range_i = list(range(3 * i, 3 * (i + 1)))
                range_j = list(range(3 * j, 3 * (j + 1)))
                range_ij = range_i + range_j
                
                # Add contribution: H += H_int * B * B^T
                # Using np.ix_ to match logic of swart.py efficient update
                self.cart_hess[np.ix_(range_ij, range_ij)] += H_int * np.outer(B, B)
        return

    def swart_angle(self, coord, element_list):
        """
        Calculate angle bending contributions.
        Paper: Force constant H_int = 0.075 * s_ij,jk^2 * (f + (1-f)*sin(θ))^2
        """
        N = len(coord)
        
        # Precompute screening for efficiency inside the angle loop (consistent with swart.py logic)
        screen_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j: continue
                cov_sum = covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j])
                dist = bond_length(coord, i, j)
                screen_matrix[i, j] = self.screening_function(dist, cov_sum)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Check if i-j screening is sufficient
                if screen_matrix[i, j] < self.eps2:
                    continue
                
                for k in range(i + 1, N):
                    if k == j:
                        continue
                    
                    # Combined screening
                    s_ij_jk = screen_matrix[i, j] * screen_matrix[j, k]
                    if s_ij_jk < self.eps1:
                        continue
                    
                    # Calculate angle
                    cos_theta = self._cos_angle(coord, i, j, k)
                    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
                    
                    # Force constant (Paper Eq. with modified coefficient 0.075)
                    H_int = 0.075 * s_ij_jk**2 * (self.f + (1 - self.f) * sin_theta)**2
                    
                    # Check for linear or zero angle
                    if cos_theta > 1 - self.tolth:
                        th1 = 1.0 - cos_theta
                    else:
                        th1 = 1.0 + cos_theta
                    
                    range_i = list(range(3 * i, 3 * (i + 1)))
                    range_j = list(range(3 * j, 3 * (j + 1)))
                    range_k = list(range(3 * k, 3 * (k + 1)))
                    range_ijk = range_i + range_j + range_k
                    
                    if th1 < self.tolth:
                        # Near-linear or near-zero angle
                        scale_lin = (1.0 - (th1 / self.tolth)**2)**2
                        
                        if cos_theta > 1 - self.tolth:
                            # Near 180 degrees (linear)
                            B_lin = bmat_linear_angle(coord, i, j, k)
                            B = bmat_angle(coord, i, j, k)
                            
                            # Scale between linear and normal
                            B_combined = scale_lin * B_lin[0, :] + (1.0 - scale_lin) * B
                            
                            # Add linear bending mode
                            self.cart_hess[np.ix_(range_ijk, range_ijk)] += H_int * np.outer(B_lin[1, :], B_lin[1, :])
                            # Add combined mode
                            self.cart_hess[np.ix_(range_ijk, range_ijk)] += H_int * np.outer(B_combined, B_combined)
                        else:
                            # Near 0 degrees
                            B = bmat_angle(coord, i, j, k)
                            B_scaled = (1.0 - scale_lin) * B
                            self.cart_hess[np.ix_(range_ijk, range_ijk)] += H_int * np.outer(B_scaled, B_scaled)
                    else:
                        # Normal angle
                        B = bmat_angle(coord, i, j, k)
                        self.cart_hess[np.ix_(range_ijk, range_ijk)] += H_int * np.outer(B, B)
        return

    def swart_dihedral_angle(self, coord, element_list):
        """
        Calculate dihedral angle contributions.
        Note: swart.py logic states 'No dihedral terms (paper notes they are not necessary)'.
        Method exists to maintain interface compatibility with SwartD3ApproxHessian.
        """
        pass
      
    def swart_out_of_plane(self, coord, element_list):
        """
        Calculate out-of-plane bending contributions.
        Note: swart.py logic states 'No out-of-plane terms'.
        Method exists to maintain interface compatibility with SwartD3ApproxHessian.
        """
        pass

    def main(self, coord, element_list, cart_gradient=None):
        """
        Main method to calculate the approximate Hessian using Swart's model.
        Note: logic expects 'coord' in Bohr.
        """
        print("Generating Swart's approximate hessian (O1NumHess variant)...")
        self.cart_hess = np.zeros((len(coord)*3, len(coord)*3), dtype="float64")
        
        self.swart_bond(coord, element_list)
        self.swart_angle(coord, element_list)
        self.swart_dihedral_angle(coord, element_list)
        self.swart_out_of_plane(coord, element_list)
        
        # Note: swart.py does not implement projection of translational/rotational modes
        # in the main logic, returning the raw internal Hessian matrix directly.
        hess_proj = Calculationtools().project_out_hess_tr_and_rot_for_coord(self.cart_hess, element_list, coord)
        return hess_proj
       