import numpy as np
from scipy.spatial import distance_matrix


def rotation_matrix(axis, theta):
    """
    Create rotation matrix for rotation around an axis by angle theta
    
    Parameters
    ----------
    axis : np.ndarray
        Axis of rotation (3D unit vector)
    theta : float
        Angle of rotation in radians
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Components of axis
    x, y, z = axis
    
    # Compute rotation matrix using axis-angle formula
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    
    matrix = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])
    
    return matrix


class SymmetryAnalyzer:
    """Analyzer for molecular symmetry elements and point group determination"""

    def __init__(self, atoms, coordinates, dist_tol=0.25, max_n_fold=6):
        """
        Parameters
        ----------
        atoms : list of str
            List of atomic symbols
        coordinates : np.ndarray
            Atomic coordinates (n_atoms, 3)
        dist_tol : float
            Distance tolerance for symmetry operations
        max_n_fold : int
            Maximum n-fold rotation to check
        """
        self.atoms = atoms
        self.coordinates = np.array(coordinates)
        self.n_atoms = len(atoms)
        self.dist_tol = dist_tol
        self.max_n_fold = max_n_fold
        
        # Center molecule at COM
        self.com = np.mean(self.coordinates, axis=0)
        self.coordinates -= self.com
        
        # Symmetry elements
        self.cn_axes = {}
        self.reflection_planes = []
        self.has_inversion = False
        self.sn_axes = {}
        
        # Properties
        self.is_linear_mol = self._check_linearity()
        self.pcoords = self._create_pcoords()
        
    def _check_linearity(self):
        """Check if molecule is linear"""
        if self.n_atoms <= 2:
            return True
            
        # Calculate principal moments of inertia
        inertia = np.zeros((3, 3))
        for i in range(self.n_atoms):
            r = self.coordinates[i]
            r2 = np.sum(r * r)
            inertia += r2 * np.eye(3) - np.outer(r, r)
            
        evals = np.linalg.eigvalsh(inertia)
        return abs(evals[0]) < 1e-6 and abs(evals[1] - evals[2]) < 1e-6
    
    def _create_pcoords(self):
        """Group coordinates by atom type"""
        atom_symbols = list(set(self.atoms))
        n_symbols = len(atom_symbols)
        pcoords = np.zeros((n_symbols, self.n_atoms, 3))
        
        for i in range(n_symbols):
            for j in range(self.n_atoms):
                if self.atoms[j] != atom_symbols[i]:
                    continue
                pcoords[i, j, :] = self.coordinates[j]
                
        return pcoords
    
    def analyze(self):
        """Analyze molecular symmetry and determine point group"""
        self._find_rotation_axes()
        self._find_reflection_planes()
        self._find_inversion_center()
        self._find_improper_rotation_axes()
        
        return self._determine_point_group()
        
    def _find_rotation_axes(self):
        """Detect rotation axes (Cn)"""
        axes = get_possible_axes(self.coordinates)
        self.cn_axes = {i: [] for i in range(2, self.max_n_fold + 1)}
        
        for axis in axes:
            for n in range(2, self.max_n_fold + 1):
                if is_same_under_n_fold(self.pcoords, axis, n=n, tol=self.dist_tol):
                    self.cn_axes[n].append(axis)
        
    def _find_reflection_planes(self):
        """Detect reflection planes (σ)"""
        axes = get_possible_axes(self.coordinates)
        self.reflection_planes = [axis for axis in axes if self._is_reflection_plane(axis)]
        self.reflection_planes = strip_identical_and_inv_axes(self.reflection_planes, self.dist_tol)
        
    def _is_reflection_plane(self, normal):
        """Check if structure is invariant under reflection through plane with given normal"""
        reflected_coords = np.array(self.coordinates, copy=True)
        
        # Reflection: r' = r - 2(r·n)n
        for i in range(self.n_atoms):
            r = self.coordinates[i]
            reflected_coords[i] = r - 2 * np.dot(r, normal) * normal
            
        for atom_type in set(self.atoms):
            orig_indices = [i for i, a in enumerate(self.atoms) if a == atom_type]
            refl_indices = [i for i, a in enumerate(self.atoms) if a == atom_type]
            
            if len(orig_indices) == 0:
                continue
                
            orig_coords = self.coordinates[orig_indices]
            refl_coords = reflected_coords[refl_indices]
            
            dist_mat = distance_matrix(orig_coords, refl_coords)
            if np.linalg.norm(np.min(dist_mat, axis=1)) > self.dist_tol:
                return False
                
        return True
        
    def _find_inversion_center(self):
        """Detect inversion center (i)"""
        inverted_coords = -self.coordinates
        
        for atom_type in set(self.atoms):
            orig_indices = [i for i, a in enumerate(self.atoms) if a == atom_type]
            inv_indices = [i for i, a in enumerate(self.atoms) if a == atom_type]
            
            if len(orig_indices) == 0:
                continue
                
            orig_coords = self.coordinates[orig_indices]
            inv_coords = inverted_coords[inv_indices]
            
            dist_mat = distance_matrix(orig_coords, inv_coords)
            if np.linalg.norm(np.min(dist_mat, axis=1)) > self.dist_tol:
                self.has_inversion = False
                return
                
        self.has_inversion = True
        
    def _find_improper_rotation_axes(self):
        """Detect improper rotation axes (Sn)"""
        axes = get_possible_axes(self.coordinates)
        self.sn_axes = {i: [] for i in range(2, self.max_n_fold + 1)}
        
        for axis in axes:
            for n in range(2, self.max_n_fold + 1):
                if self._is_improper_rotation_axis(axis, n):
                    self.sn_axes[n].append(axis)
                    
    def _is_improper_rotation_axis(self, axis, n):
        """Check if structure is invariant under improper rotation"""
        rotated_reflected_coords = np.array(self.coordinates, copy=True)
        rot_mat = rotation_matrix(axis, theta=(2.0 * np.pi / n))
        
        for i in range(self.n_atoms):
            rotated = rot_mat.dot(self.coordinates[i])
            rotated_reflected_coords[i] = rotated - 2 * np.dot(rotated, axis) * axis
            
        for atom_type in set(self.atoms):
            orig_indices = [i for i, a in enumerate(self.atoms) if a == atom_type]
            sr_indices = [i for i, a in enumerate(self.atoms) if a == atom_type]
            
            if len(orig_indices) == 0:
                continue
                
            orig_coords = self.coordinates[orig_indices]
            sr_coords = rotated_reflected_coords[sr_indices]
            
            dist_mat = distance_matrix(orig_coords, sr_coords)
            if np.linalg.norm(np.min(dist_mat, axis=1)) > self.dist_tol:
                return False
                
        return True
        
    def _determine_point_group(self):
        """Determine point group based on detected symmetry elements"""
        # Linear molecules
        if self.is_linear_mol:
            return "D∞h" if self.has_inversion else "C∞v"
                
        # High symmetry groups
        if self._has_icosahedral_symmetry():
            return "Ih" if self.has_inversion else "I"
                
        if self._has_octahedral_symmetry():
            return "Oh" if self.has_inversion else "O"
                
        if self._has_tetrahedral_symmetry():
            if self.has_inversion:
                return "Th"
            elif any(len(self.sn_axes[n]) > 0 for n in [4, 8]):
                return "Td"
            else:
                return "T"
                
        # Find highest order rotation axis
        max_cn = 1
        for n in range(self.max_n_fold, 1, -1):
            if len(self.cn_axes[n]) > 0:
                max_cn = n
                break
                
        # No principal axis
        if max_cn == 1:
            if self.has_inversion:
                return "Ci"
            elif len(self.reflection_planes) == 1:
                return "Cs"
            else:
                return "C1"
                
        # With principal axis
        principal_axis = self.cn_axes[max_cn][0]
        
        # Count C2 axes perpendicular to principal axis
        perp_c2_count = sum(1 for axis in self.cn_axes.get(2, []) 
                            if abs(np.dot(axis, principal_axis)) < 0.1)
                
        if perp_c2_count == max_cn:
            # Check for horizontal reflection plane
            has_horizontal_plane = any(abs(np.dot(plane, principal_axis)) > 0.9 
                                     for plane in self.reflection_planes)
                    
            if has_horizontal_plane:
                return f"D{max_cn}h"
            elif len(self.reflection_planes) > 0:
                return f"D{max_cn}d"
            else:
                return f"D{max_cn}"
                
        # Only C2 axes along principal axis
        if len(self.reflection_planes) > 0:
            has_horizontal_plane = any(abs(np.dot(plane, principal_axis)) > 0.9 
                                     for plane in self.reflection_planes)
                    
            if has_horizontal_plane:
                return f"C{max_cn}h"
            else:
                return f"C{max_cn}v"
        elif any(len(self.sn_axes[n]) > 0 for n in [max_cn*2, max_cn]):
            return f"S{2*max_cn}"
        else:
            return f"C{max_cn}"
            
    def _has_icosahedral_symmetry(self):
        """Check for icosahedral symmetry"""
        c5_count = len(self.cn_axes.get(5, []))
        c3_count = len(self.cn_axes.get(3, []))
        c2_count = len(self.cn_axes.get(2, []))
        return c5_count >= 6 and c3_count >= 10 and c2_count >= 15
        
    def _has_octahedral_symmetry(self):
        """Check for octahedral symmetry"""
        c4_count = len(self.cn_axes.get(4, []))
        c3_count = len(self.cn_axes.get(3, []))
        c2_count = len(self.cn_axes.get(2, []))
        return c4_count >= 3 and c3_count >= 4 and c2_count >= 6
        
    def _has_tetrahedral_symmetry(self):
        """Check for tetrahedral symmetry"""
        c3_count = len(self.cn_axes.get(3, []))
        c2_count = len(self.cn_axes.get(2, []))
        return c3_count >= 4 and c2_count >= 3


def analyze_symmetry(atoms, coordinates, dist_tol=0.25, max_n_fold=6):
    """
    Analyze molecular symmetry and determine point group
    
    Parameters
    ----------
    atoms : list of str
        List of atomic symbols
    coordinates : np.ndarray
        Atomic coordinates (n_atoms, 3)
    dist_tol : float
        Distance tolerance for symmetry operations
    max_n_fold : int
        Maximum n-fold rotation to check
        
    Returns
    -------
    str
        Molecular point group
    """
    analyzer = SymmetryAnalyzer(atoms, coordinates, dist_tol, max_n_fold)
    return analyzer.analyze()


# Support functions
def strip_identical_and_inv_axes(axes, sim_axis_tol):
    """Remove similar or inverse axes within tolerance"""
    unique_axes = []
    for axis in axes:
        if not any(np.linalg.norm(axis - unique_axis) < sim_axis_tol or 
                  np.linalg.norm(-axis - unique_axis) < sim_axis_tol 
                  for unique_axis in unique_axes):
            unique_axes.append(axis)
    return unique_axes


def get_possible_axes(coords, max_triple_dist=2.0, sim_axis_tol=0.1):
    """Get possible rotation axes in a molecule"""
    possible_axes = []
    n_atoms = len(coords)

    # Add atom pair vectors
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            vec = coords[j] - coords[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                possible_axes.append(vec/norm)

    # Add triple vectors and cross products
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            for k in range(j+1, n_atoms):
                if i == k:
                    continue
                    
                vec1 = coords[j] - coords[i]
                vec2 = coords[k] - coords[i]
                
                if all(np.linalg.norm(vec) < max_triple_dist for vec in (vec1, vec2)):
                    # Average vector
                    avg_vec = vec1 + vec2
                    avg_norm = np.linalg.norm(avg_vec)
                    if avg_norm > 1e-6:
                        possible_axes.append(avg_vec/avg_norm)
                    
                    # Perpendicular vector
                    perp_vec = np.cross(vec1, vec2)
                    perp_norm = np.linalg.norm(perp_vec)
                    if perp_norm > 1e-6:
                        possible_axes.append(perp_vec/perp_norm)

    return strip_identical_and_inv_axes(possible_axes, sim_axis_tol)


def is_same_under_n_fold(pcoords, axis, n, m=1, tol=0.25, excluded_pcoords=None):
    """Check if structure is invariant under n-fold rotation about axis"""
    n_unique, n_atoms, _ = pcoords.shape
    rotated_coords = np.array(pcoords, copy=True)
    rot_mat = rotation_matrix(axis, theta=(2.0 * np.pi * m / n))
    excluded = [False for _ in range(n_unique)]

    for i in range(n_unique):
        # Rotate coordinates
        rotated_coords[i] = rot_mat.dot(rotated_coords[i].T).T
        dist_mat = distance_matrix(pcoords[i], rotated_coords[i])

        # Check if structures match
        if np.linalg.norm(dist_mat) < tol:
            continue
        if np.linalg.norm(np.min(dist_mat, axis=1)) > tol:
            return False

        # Check against excluded coordinates
        if excluded_pcoords is not None:
            if any(np.linalg.norm(rotated_coords[i] - pcoords[i]) < tol
                   for pcoords in excluded_pcoords):
                excluded[i] = True

    if excluded_pcoords is not None:
        if all(excluded):
            return False
        excluded_pcoords.append(rotated_coords)

    return True