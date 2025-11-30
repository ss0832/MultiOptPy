import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


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
    # Ensure axis is normalized
    axis = axis / np.linalg.norm(axis)
    
    # Components of axis
    x, y, z = axis
    
    # Compute rotation matrix using axis-angle formula
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    
    matrix = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])
    
    return matrix


class SymmetryAnalyzer:
    """Analyzer for molecular symmetry elements and point group determination"""

    def __init__(self, atoms, coordinates, tol=1e-2, angle_tol=1e-4, max_n_fold=6):
        """
        Parameters
        ----------
        atoms : list of str
            List of atomic symbols
        coordinates : np.ndarray
            Atomic coordinates (n_atoms, 3)
        tol : float
            Distance tolerance for matching atoms (e.g., in Angstroms)
        angle_tol : float
            Tolerance for comparing axis vectors (dot product deviation from 1.0)
        max_n_fold : int
            Maximum n-fold rotation to check
        """
        
        self.atoms = atoms
        if coordinates is None or len(coordinates) == 0:
            raise ValueError("Valid coordinates must be provided.")
            
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.n_atoms = len(atoms)
        self.tol = tol
        self.angle_tol = angle_tol
        self.inertia_tol = 1e-6
        self.max_n_fold = max_n_fold
        
        # Center molecule at COM
        self.com = np.mean(self.coordinates, axis=0)
        self.coordinates -= self.com
        
        # Group coordinates by atom type for efficient checking
        self.pcoords = self._create_pcoords()
        
        # Symmetry elements
        self.cn_axes = {}
        self.reflection_planes = []
        self.has_inversion = False
        self.sn_axes = {}
        
        # Properties
        self.is_linear_mol = self._check_linearity()
        
    def _check_linearity(self):
        """Check if molecule is linear based on moments of inertia"""
        if self.n_atoms <= 2:
            return True
            
        # Calculate principal moments of inertia
        inertia = np.zeros((3, 3))
        for i in range(self.n_atoms):
            r = self.coordinates[i]
            r2 = np.sum(r * r)
            # We don't have atomic masses, so assume all are 1. 
            # This is fine for symmetry, as we only care about the *ratio* of moments.
            inertia += r2 * np.eye(3) - np.outer(r, r)
            
        evals = np.linalg.eigvalsh(inertia)
        
        # For a linear molecule, one moment is zero, and the other two are equal.
        # evals[0] is the smallest.
        return evals[0] < self.inertia_tol

    def _create_pcoords(self):
        """
        Group coordinates by atom type into a dictionary of dense arrays.
        This is much faster than the previous sparse array method.
        """
        pcoords = {}
        unique_atoms = set(self.atoms)
        for atom_type in unique_atoms:
            indices = [i for i, atom in enumerate(self.atoms) if atom == atom_type]
            pcoords[atom_type] = self.coordinates[indices]
        return pcoords

    def _check_op(self, op_matrix):
        """
        Check if the molecule is invariant under a given 3x3 operation matrix.
        This is the new, robust core of the analyzer.
        """
        for atom_type, coords in self.pcoords.items():
            if coords.shape[0] == 0:
                continue
            
            # Apply the operation: (N_atoms, 3) @ (3, 3) -> (N_atoms, 3)
            transformed_coords = coords @ op_matrix.T
            
            # Find the distance between all original and transformed atoms
            dist_mat = distance_matrix(coords, transformed_coords)
            
            try:
                # Use the Hungarian algorithm to find the *best one-to-one mapping*
                # This enforces a correct permutation.
                row_ind, col_ind = linear_sum_assignment(dist_mat)
                
                # Find the largest distance in this optimal mapping
                max_dist = dist_mat[row_ind, col_ind].max()
                
                # If the largest distance is over tolerance, this operation is not a symmetry
                if max_dist > self.tol:
                    return False
            except ValueError:
                # linear_sum_assignment can fail if matrix is empty or NaN
                return False
                
        # If all atom types pass, it's a valid symmetry operation
        return True

    def analyze(self):
        """Analyze molecular symmetry and determine point group"""
        # Get candidate axes *once*
        candidate_axes = get_possible_axes(self.coordinates, self.com, self.angle_tol)
        
        # Find all symmetry elements
        self._find_inversion_center()
        self._find_rotation_axes(candidate_axes)
        self._find_reflection_planes(candidate_axes)
        self._find_improper_rotation_axes(candidate_axes)
        
        return self._determine_point_group()
        
    def _find_rotation_axes(self, axes):
        """Detect rotation axes (Cn)"""
        self.cn_axes = {i: [] for i in range(2, self.max_n_fold + 1)}
        
        for n in range(self.max_n_fold, 1, -1):
            theta = 2.0 * np.pi / n
            for axis in axes:
                op = rotation_matrix(axis, theta)
                if self._check_op(op):
                    self.cn_axes[n].append(axis)
            # De-duplicate axes
            self.cn_axes[n] = strip_identical_axes(self.cn_axes[n], self.angle_tol)

    def _find_reflection_planes(self, axes):
        """Detect reflection planes (σ), using axes as plane normals"""
        planes = []
        for normal in axes:
            # Reflection matrix: I - 2 * (n @ n.T)
            op = np.eye(3) - 2 * np.outer(normal, normal)
            if self._check_op(op):
                planes.append(normal)
        self.reflection_planes = strip_identical_axes(planes, self.angle_tol)
        
    def _find_inversion_center(self):
        """Detect inversion center (i)"""
        op = -np.eye(3)
        self.has_inversion = self._check_op(op)
        
    def _find_improper_rotation_axes(self, axes):
        """Detect improper rotation axes (Sn)"""
        self.sn_axes = {i: [] for i in range(2, self.max_n_fold + 1)}
        
        for n in range(self.max_n_fold, 1, -1):
            theta = 2.0 * np.pi / n
            for axis in axes:
                # S_n = C_n followed by sigma_h (reflection in plane perp to axis)
                rot_op = rotation_matrix(axis, theta)
                refl_op = np.eye(3) - 2 * np.outer(axis, axis)
                op = refl_op @ rot_op
                
                if self._check_op(op):
                    self.sn_axes[n].append(axis)
            self.sn_axes[n] = strip_identical_axes(self.sn_axes[n], self.angle_tol)

    def _determine_point_group(self):
        """Determine point group based on detected symmetry elements (standard flowchart)"""
        
        # 1. Linear molecules
        if self.is_linear_mol:
            return "D∞h" if self.has_inversion else "C∞v"
            
        # 2. High symmetry groups
        if self._has_icosahedral_symmetry():
            return "Ih" # I (no inversion) is very rare
        
        if self._has_octahedral_symmetry():
            return "Oh" if self.has_inversion else "O"
            
        if self._has_tetrahedral_symmetry():
            if self.has_inversion:
                return "Th"
            elif len(self.reflection_planes) > 0:
                # More robust check: Td has 6 reflection planes
                if len(self.reflection_planes) >= 6:
                    return "Td"
                else:
                    return "T" # Or Th if S4 axes exist but no planes
            else:
                return "T"
                
        # 3. Find highest order rotation axis
        max_n = 1
        for n in range(self.max_n_fold, 1, -1):
            if len(self.cn_axes[n]) > 0:
                max_n = n
                break
                
        # 4. No principal axis (max_n = 1)
        if max_n == 1:
            if self.has_inversion:
                return "Ci"
            elif len(self.reflection_planes) > 0:
                return "Cs"
            else:
                return "C1"
                
        # 5. Has a principal C_n axis
        principal_axis = self.cn_axes[max_n][0]
        
        # Check for C2 axes perpendicular to principal axis (D groups)
        perp_c2_axes = [ax for ax in self.cn_axes.get(2, []) 
                        if abs(np.dot(ax, principal_axis)) < self.angle_tol]
        
        if len(perp_c2_axes) >= max_n:
            # --- D Groups ---
            # Check for horizontal plane (sigma_h)
            has_sigma_h = any(abs(np.dot(pl, principal_axis)) > (1.0 - self.angle_tol) 
                              for pl in self.reflection_planes)
            if has_sigma_h:
                return f"D{max_n}h"
            
            # Check for dihedral planes (sigma_d)
            # These are vertical planes that bisect the perp. C2 axes
            n_dihedral_planes = sum(1 for pl in self.reflection_planes 
                                    if abs(np.dot(pl, principal_axis)) < self.angle_tol)
            if n_dihedral_planes >= max_n:
                return f"D{max_n}d"
            
            return f"D{max_n}"
            
        else:
            # --- C and S Groups ---
            # Check for horizontal plane (sigma_h)
            has_sigma_h = any(abs(np.dot(pl, principal_axis)) > (1.0 - self.angle_tol) 
                              for pl in self.reflection_planes)
            if has_sigma_h:
                return f"C{max_n}h"
                
            # Check for vertical planes (sigma_v)
            n_vertical_planes = sum(1 for pl in self.reflection_planes 
                                    if abs(np.dot(pl, principal_axis)) < self.angle_tol)
            if n_vertical_planes >= max_n:
                return f"C{max_n}v"
                
            # Check for S_2n axis coincident with C_n
            n_s2n = 2 * max_n
            if n_s2n in self.sn_axes:
                 has_s2n = any(abs(np.dot(sn_ax, principal_axis)) > (1.0 - self.angle_tol) 
                               for sn_ax in self.sn_axes.get(n_s2n, []))
                 if has_s2n:
                     return f"S{n_s2n}"
                     
            return f"C{max_n}"
            
    def _has_icosahedral_symmetry(self):
        """Check for icosahedral symmetry"""
        c5_count = len(self.cn_axes.get(5, []))
        c3_count = len(self.cn_axes.get(3, []))
        return c5_count >= 6 and c3_count >= 10
        
    def _has_octahedral_symmetry(self):
        """Check for octahedral symmetry"""
        c4_count = len(self.cn_axes.get(4, []))
        c3_count = len(self.cn_axes.get(3, []))
        return c4_count >= 3 and c3_count >= 4
        
    def _has_tetrahedral_symmetry(self):
        """Check for tetrahedral symmetry"""
        c3_count = len(self.cn_axes.get(3, []))
        c2_count = len(self.cn_axes.get(2, []))
        return c3_count >= 4 and c2_count >= 3


def analyze_symmetry(atoms, coordinates, tol=1e-2, angle_tol=1e-4, max_n_fold=6):
    """
    Analyze molecular symmetry and determine point group
    
    Parameters
    ----------
    atoms : list of str
        List of atomic symbols
    coordinates : np.ndarray
        Atomic coordinates (n_atoms, 3)
    tol : float
        Distance tolerance for matching atoms (e.g., in Angstroms)
    angle_tol : float
        Tolerance for comparing axis vectors (dot product deviation from 1.0)
    max_n_fold : int
        Maximum n-fold rotation to check
        
    Returns
    -------
    str
        Molecular point group
    """
    try:
        analyzer = SymmetryAnalyzer(atoms, coordinates, tol, angle_tol, max_n_fold)
        return analyzer.analyze()
    except Exception as e:
        print(f"Error during symmetry analysis: {e}")
        return "Unknown"


# --- Support functions ---

def strip_identical_axes(axes, tol):
    """Remove similar or inverse axes within tolerance using dot product"""
    unique_axes = []
    for axis in axes:
        # Check if axis is parallel or anti-parallel to any already found
        is_duplicate = False
        for unique_axis in unique_axes:
            dot_product = abs(np.dot(axis, unique_axis))
            if dot_product > (1.0 - tol):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_axes.append(axis)
    return unique_axes


def get_possible_axes(coords, com, tol):
    """
    Get possible rotation axes in a molecule.
    Now includes body-diagonals and vector sums/differences.
    """
    possible_axes = []
    n_atoms = len(coords)

    # 1. Cartesian axes
    possible_axes.append(np.array([1., 0., 0.]))
    possible_axes.append(np.array([0., 1., 0.]))
    possible_axes.append(np.array([0., 0., 1.]))
    
    # 2. Body diagonals (for cubic symmetries)
    diag_axes = [
        [1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]
    ]
    for ax in diag_axes:
        possible_axes.append(np.array(ax) / np.linalg.norm(ax))

    # 3. Vectors from COM to each atom
    for i in range(n_atoms):
        vec = coords[i] # Coords are already COM-centered
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            possible_axes.append(vec / norm)

    # 4. Vectors between atom pairs and their cross/sum/diff
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Atom-atom pair vector
            vec = coords[j] - coords[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                possible_axes.append(vec / norm)
                
            # Cross, sum, and diff of COM-atom vectors
            vec1 = coords[i]
            vec2 = coords[j]
            
            cross_vec = np.cross(vec1, vec2)
            norm = np.linalg.norm(cross_vec)
            if norm > 1e-6:
                possible_axes.append(cross_vec / norm)

            sum_vec = vec1 + vec2
            norm = np.linalg.norm(sum_vec)
            if norm > 1e-6:
                possible_axes.append(sum_vec / norm)
                
            diff_vec = vec1 - vec2
            norm = np.linalg.norm(diff_vec)
            if norm > 1e-6:
                possible_axes.append(diff_vec / norm)

    return strip_identical_axes(possible_axes, tol)


# --- Example Usage ---

if __name__ == '__main__':
    # Water (C2v)
    atoms_water = ['O', 'H', 'H']
    coords_water = np.array([
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200]
    ])
    print(f"Molecule: Water")
    print(f"Point Group: {analyze_symmetry(atoms_water, coords_water)}\n")

    # Methane (Td)
    atoms_methane = ['C', 'H', 'H', 'H', 'H']
    coords_methane = np.array([
        [0.0, 0.0, 0.0],
        [0.6291, 0.6291, 0.6291],
        [-0.6291, -0.6291, 0.6291],
        [-0.6291, 0.6291, -0.6291],
        [0.6291, -0.6291, -0.6291]
    ])
    print(f"Molecule: Methane")
    print(f"Point Group: {analyze_symmetry(atoms_methane, coords_methane)}\n")

    # Benzene (D6h)
    atoms_benzene = ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    coords_benzene = np.array([
        [ 0.0000,  1.397, 0.0], [ 1.210,  0.698, 0.0], [ 1.210, -0.698, 0.0],
        [ 0.0000, -1.397, 0.0], [-1.210, -0.698, 0.0], [-1.210,  0.698, 0.0],
        [ 0.0000,  2.484, 0.0], [ 2.151,  1.242, 0.0], [ 2.151, -1.242, 0.0],
        [ 0.0000, -2.484, 0.0], [-2.151, -1.242, 0.0], [-2.151,  1.242, 0.0]
    ])
    print(f"Molecule: Benzene")
    print(f"Point Group: {analyze_symmetry(atoms_benzene, coords_benzene)}\n")


    # Allene (D2d) 
    atoms_allene = ['C', 'C', 'C', 'H', 'H', 'H', 'H']
    coords_allene_new = np.array([
        [ 0.0,  0.0,  0.0], [ 0.0,  0.0,  1.308], [ 0.0,  0.0, -1.308],
        [ 0.0,  0.95, 1.848], [ 0.0, -0.95, 1.848],
        [ 0.95, 0.0, -1.848], [-0.95, 0.0, -1.848]
    ])
    print(f"Molecule: Allene")
    print(f"Point Group: {analyze_symmetry(atoms_allene, coords_allene_new)}\n")
    
    # SF6 (Oh)
    atoms_sf6 = ['S'] + ['F'] * 6
    coords_sf6 = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0], [0.0, -1.5, 0.0],
        [0.0, 0.0, 1.5], [0.0, 0.0, -1.5]
    ])
    print(f"Molecule: SF6")
    print(f"Point Group: {analyze_symmetry(atoms_sf6, coords_sf6)}\n")