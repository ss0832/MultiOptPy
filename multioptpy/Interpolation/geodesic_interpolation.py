import numpy as np
from scipy.optimize import least_squares
from multioptpy.Parameters.parameter import covalent_radii_lib

#ref.: https://github.com/virtualzx-nad/geodesic-interpolate

def distribute_geometry_geodesic(geometry_list, n_points=None, spacing=None, spline_degree=3,
                          max_iterations=50, tolerance=1e-4, element_list=None, verbose=True):
    """
    Performs geodesic interpolation between molecular geometries using exponential and logarithmic maps.
    
    This function distributes geometries along a geodesic path in the internal coordinate space,
    resulting in more realistic interpolations compared to linear methods in Cartesian space.
    
    Parameters:
    -----------
    geometry_list : list
        List of geometry arrays/objects with shape (n_atoms, 3)
    n_points : int, optional
        Number of points to generate along the path (including endpoints)
    spacing : float, optional
        If specified, points will be distributed with this spacing (in Angstroms)
        If both n_points and spacing are specified, n_points takes precedence
    spline_degree : int, optional
        Degree of the spline for initial path estimation (default=3)
    max_iterations : int, optional
        Maximum number of geodesic optimization iterations
    tolerance : float, optional
        Convergence tolerance for geodesic optimization
    element_list : list, optional
        List of element symbols for each atom, used for better distance weighting
    verbose : bool, optional
        If True, print optimization progress (default=True)
        
    Returns:
    --------
    list
        List of geometries distributed along the geodesic path
    """
    if verbose:
        print("="*60)
        print("Starting Geodesic Interpolation")
        print("="*60)
    
    # Handle input validation
    if len(geometry_list) < 2:
        if verbose:
            print("Warning: Less than 2 geometries provided. Returning original list.")
        return geometry_list.copy()
    
    # Convert all geometries to numpy arrays
    geoms = [np.array(geom, dtype=np.float64) for geom in geometry_list]
    n_atoms = geoms[0].shape[0]
    
    if verbose:
        print(f"Input: {len(geometry_list)} geometries with {n_atoms} atoms each")
    
    # Use default atoms if none provided
    atoms = element_list if element_list is not None else ['C'] * n_atoms
    
    # Align geometries to minimize RMSD
    if verbose:
        print("Aligning geometries to minimize RMSD...")
    max_rmsd, aligned_geoms = align_path(geoms, verbose=verbose)
    
    # Determine the number of points to generate
    if n_points is None and spacing is None:
        n_points = len(geometry_list)
    elif n_points is None:
        # Estimate path length for spacing-based distribution
        path_length = estimate_path_length(aligned_geoms)
        n_points = max(3, int(np.ceil(path_length / spacing)) + 1)
        if verbose:
            print(f"Estimated path length: {path_length:.3f} Å")
            print(f"Using spacing {spacing:.3f} Å -> {n_points} points")
    
    if verbose:
        print(f"Target number of geometries: {n_points}")
    
    # Step 1: Generate initial path with correct number of points using redistribution
    if verbose:
        print("\nStep 1: Redistributing geometries...")
    redistributed_path = redistribute(atoms, aligned_geoms, n_points, tol=tolerance, verbose=verbose)
    
    # Step 2: Apply geodesic smoothing to optimize the path
    if verbose:
        print(f"\nStep 2: Geodesic optimization (max_iter={max_iterations}, tol={tolerance})")
    geodesic = Geodesic(atoms, redistributed_path, scaler=1.7, 
                      threshold=3, min_neighbors=4, 
                      verbose=verbose, friction=1e-3)
    
    # Perform geodesic optimization
    optimized_path = geodesic.smooth(tol=tolerance, max_iter=max_iterations, verbose=verbose)
    
    if verbose:
        print("="*60)
        print("Geodesic Interpolation Completed Successfully!")
        print("="*60)
    
    return [geom.copy() for geom in optimized_path]


def align_path(path, verbose=False):
    """
    Rotate and translate images to minimize RMSD movements along the path.
    Also moves the geometric center of all images to the origin.
    Optimized with vectorized operations where possible.
    """
    path = np.array(path, dtype=np.float64)
    # Center the first geometry
    path[0] -= np.mean(path[0], axis=0)
    
    max_rmsd = 0
    # Sequential alignment (cannot be fully vectorized due to dependency)
    for i in range(len(path) - 1):
        rmsd, path[i + 1] = align_geom(path[i], path[i + 1])
        max_rmsd = max(max_rmsd, rmsd)
        if verbose:
            print(f"  Aligned geometry {i+1}: RMSD = {rmsd:.4f} Å")
    
    if verbose:
        print(f"Alignment completed. Maximum RMSD: {max_rmsd:.4f} Å")
    
    return max_rmsd, path


def align_geom(refgeom, geom):
    """
    Find translation/rotation that moves a given geometry to maximally overlap
    with a reference geometry using the Kabsch algorithm.
    """
    center = np.mean(refgeom, axis=0)   # Find the geometric center
    ref2 = refgeom - center
    geom2 = geom - np.mean(geom, axis=0)
    cov = np.dot(geom2.T, ref2)
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    new_geom = np.dot(geom2, u) + center
    rmsd = np.sqrt(np.mean((new_geom - refgeom) ** 2))
    return rmsd, new_geom


def estimate_path_length(geometries):
    """
    Estimates the total path length in Cartesian space.
    """
    geometries = np.array(geometries)
    # Vectorized calculation of consecutive differences
    diffs = geometries[1:] - geometries[:-1]
    # Calculate RMSD for each consecutive pair
    rmsds = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=1))
    return np.sum(rmsds)



def get_bond_list(geom, atoms=None, threshold=4, min_neighbors=4, snapshots=30, bond_threshold=1.8,
                  enforce=()):
    """
    Get the list of all the important atom pairs.
    Samples a number of snapshots from a list of geometries to generate all
    distances that are below a given threshold in any of them.
    """
    from scipy.spatial import KDTree
    
    # Type casting and value checks on input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # If there is only one geometry or it is flattened, promote to 3d
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Determine which images to be used to determine distances
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    
    # Get neighbor list for included geometry and merge them (fully vectorized)
    rijset = set(enforce)
    
    for image in images:
        tree = KDTree(geom[image])
        pairs = tree.query_pairs(threshold)
        rijset.update(pairs)
        bonded = tree.query_pairs(bond_threshold)
        
        if bonded:
            bonded_array = np.array(list(bonded))
            n_atoms = geom.shape[1]
            
            # Choose optimal algorithm based on system size and bonding density
            n_bonds = len(bonded_array)
            density = n_bonds / (n_atoms * (n_atoms - 1) / 2)
            
            if n_atoms > 500 or density < 0.1:
                # Sparse approach for large or sparse systems
                from scipy.sparse import csr_matrix
                
                # Build sparse adjacency matrix with self-connections
                all_i = np.concatenate([bonded_array[:, 0], bonded_array[:, 1], np.arange(n_atoms)])
                all_j = np.concatenate([bonded_array[:, 1], bonded_array[:, 0], np.arange(n_atoms)])
                adj_sparse = csr_matrix((np.ones(len(all_i)), (all_i, all_j)), 
                                      shape=(n_atoms, n_atoms), dtype=bool)
                
                # Compute extended connections via sparse matrix multiplication
                extended_sparse = np.dot(adj_sparse, adj_sparse)
                extended_coo = extended_sparse.tocoo()
                
                # Filter to upper triangular (i < j) and add to rijset
                mask = extended_coo.row < extended_coo.col
                new_pairs = set(zip(extended_coo.row[mask], extended_coo.col[mask]))
                rijset.update(new_pairs)
            else:
                # Dense approach for smaller dense systems
                adj_matrix = np.eye(n_atoms, dtype=bool)  # Start with identity (self-connections)
                adj_matrix[bonded_array[:, 0], bonded_array[:, 1]] = True
                adj_matrix[bonded_array[:, 1], bonded_array[:, 0]] = True
                
                # Extended connections via matrix multiplication
                extended = np.dot(adj_matrix, adj_matrix)

                # Extract upper triangular indices efficiently
                triu_indices = np.triu_indices(n_atoms, k=1)
                valid_mask = extended[triu_indices]
                i_valid = triu_indices[0][valid_mask]
                j_valid = triu_indices[1][valid_mask]
                
                # Add to rijset
                new_pairs = set(zip(i_valid, j_valid))
                rijset.update(new_pairs)
    rijlist = sorted(rijset)
    
    # Check neighbor count to make sure `min_neighbors` is satisfied using vectorized operations
    rijlist = sorted(rijset)
    if not rijlist:
        re = np.array([])
        return rijlist, re
    
    # Vectorized neighbor counting
    rij_array = np.array(rijlist)
    count = np.zeros(geom.shape[1], dtype=int)
    unique_i, counts_i = np.unique(rij_array[:, 0], return_counts=True)
    unique_j, counts_j = np.unique(rij_array[:, 1], return_counts=True)
    
    count[unique_i] += counts_i
    count[unique_j] += counts_j
    
    # Find atoms with insufficient neighbors
    insufficient_atoms = np.where(count < min_neighbors)[0]
    
    for idx in insufficient_atoms:
        tree = KDTree(geom[-1])
        _, neighbors = tree.query(geom[-1, idx], k=min_neighbors + 1)
        for i in neighbors:
            if i == idx:
                continue
            pair = tuple(sorted([i, idx]))
            if pair in rijset:
                continue
            else:
                rijset.add(pair)
                rijlist.append(pair)
                count[i] += 1
                count[idx] += 1
    
    if atoms is None:
        re = np.full(len(rijlist), 2.0)
    else:
        radius = np.array([covalent_radii_lib(atom.capitalize()) for atom in atoms])
        re = np.array([radius[i] + radius[j] for i, j in rijlist])
    
    return rijlist, re


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    """
    Returns a scaling function that determines the metric of the internal
    coordinates using morse potential. Optimized for vectorized operations.
    """
    # Ensure re is a numpy array for vectorized operations
    re = np.asarray(re)
    
    def scaler(x):
        x = np.asarray(x)
        ratio = x / re
        
        # Vectorized exponential and ratio calculations
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        
        # Vectorized derivative calculation
        dval = -alpha / re * val1 - val2 / x
        
        return val1 + val2, dval
    return scaler


def compute_wij(geom, rij_list, func):
    """
    Calculate a list of scaled distances and their derivatives using vectorized operations
    """
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    
    if nrij == 0:
        return np.array([]), np.zeros((0, geom.size))
    
    rij, bmat = compute_rij(geom, rij_list)
    wij, dwdr = func(rij)
    
    # Vectorized multiplication of gradients
    bmat_reshaped = bmat.reshape(nrij, -1)
    bmat_scaled = bmat_reshaped * dwdr[:, np.newaxis]
    
    return wij, bmat_scaled


def compute_rij(geom, rij_list):
    """
    Calculate a list of distances and their derivatives using vectorized operations
    """
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    
    if nrij == 0:
        return np.array([]), np.zeros((0, len(geom), 3))
    
    # Convert rij_list to numpy arrays for vectorized operations
    rij_array = np.array(rij_list)
    i_indices = rij_array[:, 0]
    j_indices = rij_array[:, 1]
    
    # Vectorized calculation of displacement vectors
    dvec = geom[i_indices] - geom[j_indices]
    
    # Vectorized distance calculation
    rij = np.linalg.norm(dvec, axis=1)
    
    # Avoid division by zero
    safe_rij = np.where(rij > 1e-12, rij, 1e-12)
    grad = dvec / safe_rij[:, np.newaxis]
    
    # Initialize B matrix
    bmat = np.zeros((nrij, len(geom), 3))
    
    # Vectorized gradient assignment
    bmat[np.arange(nrij), i_indices] = grad
    bmat[np.arange(nrij), j_indices] = -grad
    
    return rij, bmat


def mid_point(atoms, geom1, geom2, tol=1e-2, nudge=0.01, threshold=4, verbose=False):
    """
    Find the Cartesian geometry that has internal coordinate values closest to the average of
    two geometries.
    """
    # Process the initial geometries, construct coordinate system and obtain average internals
    geom1, geom2 = np.array(geom1), np.array(geom2)
    add_pair = set()
    geom_list = [geom1, geom2]
    
    iteration = 0
    while True:
        iteration += 1
        rijlist, re = get_bond_list(geom_list, threshold=threshold + 1, enforce=add_pair)
        scaler = morse_scaler(alpha=0.7, re=re)
        w1, _ = compute_wij(geom1, rijlist, scaler)
        w2, _ = compute_wij(geom2, rijlist, scaler)
        w = (w1 + w2) / 2
        d_min, x_min = np.inf, None
        friction = 0.1 / np.sqrt(geom1.shape[0])
        
        if verbose:
            print(f"    Mid-point iteration {iteration}: {len(rijlist)} coordinate pairs")
        
        # The inner loop performs minimization using either end-point as the starting guess
        for coef_idx, coef in enumerate([0.02, 0.98]):
            x0 = (geom1 * coef + (1 - coef) * geom2).ravel()
            x0 += nudge * np.random.random_sample(x0.shape)
            if verbose:
                print(f"      Starting optimization from coef={coef:.2f}")
            
            result = least_squares(
                lambda x: np.concatenate([compute_wij(x, rijlist, scaler)[0] - w, (x-x0)*friction]), 
                x0,
                lambda x: np.vstack([compute_wij(x, rijlist, scaler)[1], np.identity(x.size) * friction]), 
                ftol=tol, 
                gtol=tol
            )
            
            x_mid = result['x'].reshape(-1, 3)
            
            # Take the interpolated geometry, construct new pair list and check for new contacts
            new_list = geom_list + [x_mid]
            new_rij, _ = get_bond_list(new_list, threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rijlist)
            
            if extras: 
                if verbose:
                    print(f"      New contacts detected. Adding {len(extras)} pairs.")
                # Update pair list then go back to the minimization loop if new contacts are found
                geom_list = new_list
                add_pair |= extras
                break
                
            # Perform local geodesic optimization for the new image
            smoother = Geodesic(atoms, [geom1, x_mid, geom2], 0.7, threshold=threshold, verbose=False, friction=1)
            smoother.compute_disps()
            
            # Vectorized width calculation
            geom_array = np.array([geom1, geom2])
            diffs = geom_array - smoother.path[1]
            widths = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=1))
            width = np.max(widths)
            
            dist, x_mid = width + smoother.length, smoother.path[1]
            if verbose:
                print(f"      Path length: {dist:.6f} after {result['nfev']} evaluations")
            
            if dist < d_min:
                d_min, x_min = dist, x_mid
                
        else:   # Both starting guesses finished without new atom pairs. Minimization successful
            if verbose:
                print(f"    Mid-point converged with path length: {d_min:.6f}")
            break
            
    return x_min


def redistribute(atoms, geoms, nimages, tol=1e-2, verbose=False):
    """
    Add or remove images so that the path length matches the desired number.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)
    
    if verbose:
        print(f"  Initial path has {len(geoms)} geometries, target: {nimages}")
    
    # If there are too few images, add bisection points
    while len(geoms) < nimages:
        # Vectorized distance calculation
        geoms_array = np.array(geoms)
        diffs = geoms_array[1:] - geoms_array[:-1]
        dists = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=1))
        max_i = np.argmax(dists)
        
        if verbose:
            print(f"  Inserting geometry between {max_i} and {max_i + 1} (RMSD={dists[max_i]:.3f})")
        
        insertion = mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol, verbose=verbose)
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])
        
        if verbose:
            print(f"  New path length: {len(geoms)}")
        
    # If there are too many images, remove points
    while len(geoms) > nimages:
        # Vectorized distance calculation for removal
        geoms_array = np.array(geoms)
        diffs = geoms_array[2:] - geoms_array[:-2]
        dists = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=1))
        min_i = np.argmin(dists)
        
        if verbose:
            print(f"  Removing geometry {min_i + 1} (merged RMSD={dists[min_i]:.3f})")
        
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])
        
        if verbose:
            print(f"  New path length: {len(geoms)}")
        
    return geoms


class Geodesic(object):
    """
    Optimizer to obtain geodesic in redundant internal coordinates.
    Core part is the calculation of the path length in the internal metric.
    """
    def __init__(self, atoms, path, scaler=1.7, threshold=3, min_neighbors=4, verbose=True,
                 friction=1e-3):
        """Initialize the interpolater"""
        rmsd0, self.path = align_path(path)
        if verbose:
            print(f"  Maximum RMSD change in initial path: {rmsd0:.4f} Å")
        
        if self.path.ndim != 3:
            raise ValueError('The path to be interpolated must have 3 dimensions')
        self.nimages, self.natoms, _ = self.path.shape
        
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(path, atoms, threshold=threshold, min_neighbors=min_neighbors)
        if isinstance(scaler, float):
            self.scaler = morse_scaler(re=self.re, alpha=1.7)
        else:
            self.scaler = scaler
        self.nrij = len(self.rij_list)
        self.friction = friction
        self.verbose = verbose
        
        # Initialize internal storages
        if verbose:
            print(f"  Geodesic setup: {self.nimages} images, {self.natoms} atoms, {self.nrij} coordinate pairs")
        
        self.neval = 0
        self.w = [None] * len(path)
        self.dwdR = [None] * len(path)
        self.X_mid = [None] * (len(path) - 1)
        self.w_mid = [None] * (len(path) - 1)
        self.dwdR_mid = [None] * (len(path) - 1)
        self.disps = self.grad = self.segment = None
        self.conv_path = []
        
        # Track optimization progress
        self.optimization_history = {
            'iterations': [],
            'path_lengths': [],
            'optimalities': [],
            'rmsds': []
        }

    def get_optimization_summary(self):
        """
        Return a summary of the optimization process
        """
        if not self.optimization_history['iterations']:
            return "No optimization performed yet."
        
        summary = f"""
Geodesic Optimization Summary:
==============================
Total iterations: {len(self.optimization_history['iterations'])}
Initial path length: {self.optimization_history['path_lengths'][0]:.6f}
Final path length: {self.optimization_history['path_lengths'][-1]:.6f}
Path length reduction: {self.optimization_history['path_lengths'][0] - self.optimization_history['path_lengths'][-1]:.6f}
Final optimality: {self.optimization_history['optimalities'][-1]:.3e}
Final RMSD: {self.optimization_history['rmsds'][-1]:.4f} Å
"""
        return summary

    def print_optimization_progress(self):
        """
        Print detailed optimization progress
        """
        if not self.optimization_history['iterations']:
            print("No optimization data available.")
            return
        
        print("\nDetailed Optimization Progress:")
        print("-" * 50)
        print("Iter |   Path Length   | Optimality  | RMSD (Å)")
        print("-" * 50)
        
        for i, (iteration, length, opt, rmsd) in enumerate(zip(
            self.optimization_history['iterations'],
            self.optimization_history['path_lengths'],
            self.optimization_history['optimalities'],
            self.optimization_history['rmsds']
        )):
            if i % 5 == 0 or i == len(self.optimization_history['iterations']) - 1:
                print(f"{iteration:4d} | {length:13.6f} | {opt:9.3e} | {rmsd:8.4f}")
        print("-" * 50)

    def update_intc(self):
        """
        Adjust unknown locations of mid points and compute missing values of 
        internal coordinates and their derivatives.
        """
        for i, (X, w, dwdR) in enumerate(zip(self.path, self.w, self.dwdR)):
            if w is None:
                self.w[i], self.dwdR[i] = compute_wij(X, self.rij_list, self.scaler)
        for i, (X0, X1, w) in enumerate(zip(self.path, self.path[1:], self.w_mid)):
            if w is None:
                self.X_mid[i] = Xm = (X0 + X1) / 2
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(Xm, self.rij_list, self.scaler)

    def update_geometry(self, X, start, end):
        """
        Update the geometry of a segment of the path, then set the corresponding internal
        coordinate, derivatives and midpoint locations to unknown
        """
        X = X.reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        self.path[start:end] = X
        for i in range(start, end):
            self.w_mid[i] = self.w[i] = None
        self.w_mid[start - 1] = None
        return True

    def compute_disps(self, start=1, end=-1, dx=None, friction=1e-3):
        """
        Compute displacement vectors and total length between two images using vectorized operations.
        Only recalculate internal coordinates if they are unknown.
        """
        if end < 0:
            end += self.nimages
        self.update_intc()
        
        # Vectorized calculation of displacement vectors in each segment
        w_left = np.array(self.w[start - 1:end])
        w_mid_left = np.array(self.w_mid[start - 1:end])
        w_right = np.array(self.w[start:end + 1])
        w_mid_right = np.array(self.w_mid[start - 1:end])
        
        vecs_l = w_mid_left - w_left
        vecs_r = w_right - w_mid_right
        
        # Vectorized norm calculation
        norms_l = np.linalg.norm(vecs_l, axis=1)
        norms_r = np.linalg.norm(vecs_r, axis=1)
        self.length = np.sum(norms_l) + np.sum(norms_r)
        
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry with friction term
        
        self.disps = np.concatenate([vecs_l.ravel(), vecs_r.ravel(), trans])
        self.disps0 = self.disps[:len(vecs_l.ravel()) + len(vecs_r.ravel())]

    def compute_disp_grad(self, start, end, friction=1e-3):
        """
        Compute derivatives of the displacement vectors with respect to the Cartesian coordinates
        using vectorized operations
        """
        # Calculate derivatives of displacement vectors with respect to image Cartesians
        l = end - start + 1
        total_size = l * 2 * self.nrij + 3 * (end - start) * self.natoms
        dof_size = (end - start) * 3 * self.natoms
        
        self.grad = np.zeros((total_size, dof_size))
        self.grad0 = self.grad[:l * 2 * self.nrij]
        
        grad_shape = (l, self.nrij, end - start, 3 * self.natoms)
        grad_l = self.grad[:l * self.nrij].reshape(grad_shape)
        grad_r = self.grad[l * self.nrij:l * self.nrij * 2].reshape(grad_shape)
        
        # Vectorized gradient calculation
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dwdR_mid[image - 1] / 2
            dmid2 = self.dwdR_mid[image] / 2
            dwdR_image = self.dwdR[image]
            
            if i + 1 < l:
                grad_l[i + 1, :, i, :] = dmid2 - dwdR_image
            grad_l[i, :, i, :] = dmid1
            
            if i + 1 < l:
                grad_r[i + 1, :, i, :] = -dmid2
            grad_r[i, :, i, :] = dwdR_image - dmid1
        
        # Vectorized friction term assignment
        friction_indices = np.arange((end - start) * 3 * self.natoms)
        self.grad[l * self.nrij * 2 + friction_indices, friction_indices] = friction

    def compute_target_func(self, X=None, start=1, end=-1, x0=None, friction=1e-3):
        """
        Compute the vectorized target function, which is then used for least squares minimization.
        """
        if end < 0:
            end += self.nimages
        if X is not None and not self.update_geometry(X, start, end) and self.segment == (start, end):
            return
            
        self.segment = start, end
        dx = np.zeros(self.path[start:end].size) if x0 is None else self.path[start:end].ravel() - x0.ravel()
        self.compute_disps(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.linalg.norm(np.einsum('i,i...', self.disps, self.grad), ord=np.inf)
        
        # Calculate current RMSD for tracking
        current_rmsd = 0.0
        for i in range(len(self.path) - 1):
            rmsd = np.sqrt(np.mean((self.path[i+1] - self.path[i]) ** 2))
            current_rmsd = max(current_rmsd, rmsd)
        
        # Store optimization history
        self.optimization_history['iterations'].append(self.neval)
        self.optimization_history['path_lengths'].append(self.length)
        self.optimization_history['optimalities'].append(self.optimality)
        self.optimization_history['rmsds'].append(current_rmsd)
        
        if self.verbose:
            # Check if we should print this iteration
            should_print = True
            if hasattr(self, '_progress_freq') and self._progress_freq > 1:
                should_print = (self.neval % self._progress_freq == 0) or (self.neval == 0)
            
            if should_print:
                print(f"    Iteration {self.neval:3d}: Length={self.length:10.6f}, |dL|={self.optimality:8.3e}, RMSD={current_rmsd:.4f}")
        
        self.conv_path.append(self.path[1].copy())
        self.neval += 1

    def target_func(self, X, **kwargs):
        """
        Wrapper around `compute_target_func` to prevent repeated evaluation at the same geometry
        """
        self.compute_target_func(X, **kwargs)
        return self.disps

    def target_deriv(self, X, **kwargs):
        """
        Wrapper around `compute_target_func` to prevent repeated evaluation at the same geometry
        """
        self.compute_target_func(X, **kwargs)
        return self.grad

    def smooth(self, tol=1e-3, max_iter=50, start=1, end=-1, verbose=None, friction=None,
               xref=None, progress_freq=1):
        """
        Minimize the path length as an overall function of the coordinates of all the images.
        
        Parameters:
        -----------
        progress_freq : int, optional
            Print progress every N iterations (default=1, set to 0 to disable progress)
        """
        if verbose is None:
            verbose = self.verbose
        
        # Temporarily adjust verbosity for iteration printing
        original_verbose = self.verbose
        if progress_freq == 0:
            self.verbose = False
        elif progress_freq > 1:
            self._progress_freq = progress_freq
            self._iteration_count = 0
            
        X0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref = X0
        self.disps = self.grad = self.segment = None
        
        if verbose:
            print(f"  Starting geodesic optimization with {len(X0)} degrees of freedom")
            if progress_freq > 1:
                print(f"  Progress will be shown every {progress_freq} iterations")
        
        if friction is None:
            friction = self.friction
            
        # Configure the keyword arguments that will be sent to the target function
        kwargs = dict(start=start, end=end, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        
        if self.optimality > tol:
            if verbose:
                print("  Starting least-squares optimization...")
            
            result = least_squares(self.target_func, X0, self.target_deriv, ftol=tol, gtol=tol,
                                 max_nfev=max_iter, kwargs=kwargs, loss='soft_l1')
            self.update_geometry(result['x'], start, end)
            
            if verbose:
                print(f"  Optimization converged after {result['nfev']} iterations")
                print(f"  Success: {result['success']}, Message: {result['message']}")
        else:
            if verbose:
                print("  Path already optimal, skipping optimization")
        
        # Restore original verbosity
        self.verbose = original_verbose
            
        rmsd, self.path = align_path(self.path)
        
        if verbose:
            print(f"  Final path length: {self.length:.6f}")
            print(f"  Maximum RMSD in path: {rmsd:.4f} Å")
            
            # Print optimization summary
            if len(self.optimization_history['iterations']) > 1:
                print(self.get_optimization_summary())
        
        return self.path