import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline, PchipInterpolator

from multioptpy.Utils.calc_tools import calc_path_length_list
from multioptpy.Interpolation.linear_interpolation import distribute_geometry_by_length, distribute_geometry

def spline_interpolation(
    structures,
    n_points=20,
    method='hermite',  # Options: 'linear', 'quadratic', 'cubic', 'b-spline', 'hermite'
    bc_type='natural',
    spline_degree=5,
    window=None
):
    """
    Interpolates between atomic structures using various spline interpolation methods.
    Supports global and local (windowed) interpolation. For local interpolation,
    only the nearest 'window' structures before and after each node are used.

    Args:
        structures: list of np.ndarray, each of shape (n_atoms, 3)
        n_points: int, number of interpolated structures to generate
        method: str, interpolation method to use
            Options:
                'linear'   : Piecewise linear interpolation
                'quadratic': Quadratic spline (degree 2, via make_interp_spline)
                'cubic'    : Cubic spline (via CubicSpline)
                'b-spline' : B-spline of arbitrary degree (use spline_degree)
                'hermite'  : Hermite spline (PCHIP monotonic cubic)
        bc_type: str or tuple, boundary condition type for cubic spline
        spline_degree: int or None, degree for B-spline interpolation
        window: int or None, number of structures before and after each node to use for local interpolation.
            If None, uses global interpolation.

    Returns:
        path: np.ndarray of shape (n_points, n_atoms, 3)
    """
    structures = np.array(structures)  # (n_structures, n_atoms, 3)
    n_structures = structures.shape[0]
    n_atoms = structures.shape[1]

    if window is None:
        # Global interpolation
        x = np.linspace(0, 1, n_structures)
        t_values = np.linspace(0, 1, n_points)
        path = np.zeros((n_points, n_atoms, 3))

        for atom_idx in range(n_atoms):
            for coord_idx in range(3):
                y = structures[:, atom_idx, coord_idx]
                if method == 'linear':
                    path[:, atom_idx, coord_idx] = np.interp(t_values, x, y)
                elif method == 'quadratic':
                    spline = make_interp_spline(x, y, k=2)
                    path[:, atom_idx, coord_idx] = spline(t_values)
                elif method == 'cubic':
                    spline = CubicSpline(x, y, bc_type=bc_type)
                    path[:, atom_idx, coord_idx] = spline(t_values)
                elif method == 'b-spline':
                    deg = spline_degree if spline_degree is not None else 3
                    spline = make_interp_spline(x, y, k=deg)
                    path[:, atom_idx, coord_idx] = spline(t_values)
                elif method == 'hermite':
                    interpolator = PchipInterpolator(x, y)
                    path[:, atom_idx, coord_idx] = interpolator(t_values)
                else:
                    raise ValueError(
                        f"Unknown method '{method}'. Supported methods are: 'linear', 'quadratic', 'cubic', 'b-spline', 'hermite'."
                    )
        print(f"Using global '{method}' spline interpolation"
              f"{' degree ' + str(spline_degree) if method == 'b-spline' else ''}"
              f"{' and bc_type ' + str(bc_type) if method == 'cubic' else ''}.")
        return path

    # Local (windowed) interpolation
    segments = []
    for idx in range(n_structures - 1):
        start = max(0, idx - window)
        end = min(n_structures, idx + window + 2)  # +2 to include idx+1
        local_structures = structures[start:end]
        local_n_structures = local_structures.shape[0]
        local_x = np.linspace(0, 1, local_n_structures)
        if idx - window >= 0:
            t0 = local_x[window]
            t1 = local_x[window + 1]
        else:
            t0 = local_x[idx]
            t1 = local_x[idx + 1]
        t_values = np.linspace(t0, t1, n_points)

        local_path = np.zeros((n_points, n_atoms, 3))
        for atom_idx in range(n_atoms):
            for coord_idx in range(3):
                y = local_structures[:, atom_idx, coord_idx]
                if method == 'linear':
                    local_path[:, atom_idx, coord_idx] = np.interp(t_values, local_x, y)
                elif method == 'quadratic':
                    spline = make_interp_spline(local_x, y, k=2)
                    local_path[:, atom_idx, coord_idx] = spline(t_values)
                elif method == 'cubic':
                    spline = CubicSpline(local_x, y, bc_type=bc_type)
                    local_path[:, atom_idx, coord_idx] = spline(t_values)
                elif method == 'b-spline':
                    deg = spline_degree if spline_degree is not None else 3
                    spline = make_interp_spline(local_x, y, k=deg)
                    local_path[:, atom_idx, coord_idx] = spline(t_values)
                elif method == 'hermite':
                    interpolator = PchipInterpolator(local_x, y)
                    local_path[:, atom_idx, coord_idx] = interpolator(t_values)
                else:
                    raise ValueError(
                        f"Unknown method '{method}'. Supported methods are: 'linear', 'quadratic', 'cubic', 'b-spline', 'hermite'."
                    )
        segments.append(local_path)

    # Concatenate all segments, removing duplicate points at boundaries
    result_path = [segments[0][0]]
    for seg in segments:
        result_path.extend(seg[1:])
    result_path = np.array(result_path)

    print(f"Using local '{method}' spline interpolation with window={window},"
          f"{' degree ' + str(spline_degree) if method == 'b-spline' else ''}"
          f"{' and bc_type ' + str(bc_type) if method == 'cubic' else ''}.")

    # Resample to exactly n_points along the full path
    result_path = resample_path(result_path, n_points)
    return result_path

def resample_path(path, n_points):
    """
    Resamples a path to have exactly n_points structures.
    path: np.ndarray of shape (N, n_atoms, 3)
    n_points: int, desired number of output structures
    Returns: np.ndarray of shape (n_points, n_atoms, 3)
    """
    N = path.shape[0]
    indices = np.linspace(0, N-1, n_points)
    resampled = np.array([path[int(round(idx))] for idx in indices])
    return resampled

# Example usage:
# S0, S1, S2, S3, S4, S5 = np.array(...), np.array(...), np.array(...), np.array(...), np.array(...), np.array(...)
# path_global_cubic = spline_interpolation([S0, S1, S2, S3, S4, S5], n_points=20, method='cubic')
# path_local_bspline = spline_interpolation([S0, S1, S2, S3, S4, S5], n_points=10, method='b-spline', spline_degree=5, window=3)
# path_local_hermite = spline_interpolation([S0, S1, S2, S3, S4, S5], n_points=10, method='hermite', window=3)




def distribute_geometry_by_length_spline(geometry_list, angstrom_spacing, spline_degree=3):
    """
    Distribute geometries by specified distance spacing using B-spline interpolation
    
    Parameters:
    -----------
    geometry_list : list
        List of geometry arrays/objects
    angstrom_spacing : float
        Desired spacing in Angstroms between distributed geometries
    spline_degree : int, optional
        Degree of the spline interpolation (default=3 for cubic splines)
        
    Returns:
    --------
    list
        New list of geometries distributed at regular intervals using spline interpolation
    """
    # Handle edge cases
    if len(geometry_list) <= 1:
        return geometry_list.copy()
    
    # Calculate path lengths
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    
    # Handle case with extremely short paths
    if total_length < angstrom_spacing:
        return [geometry_list[0], geometry_list[-1]]
    
    # Ensure spline degree is not greater than number of points minus 1
    k = min(spline_degree, len(geometry_list) - 1)
    
    # Check for duplicate path lengths and handle them
    # First create a list of unique indices to keep
    unique_indices = []
    unique_path_lengths = []
    for i, length in enumerate(path_length_list):
        if i == 0 or abs(length - path_length_list[i-1]) > 1e-10:  # Use small epsilon for floating-point comparison
            unique_indices.append(i)
            unique_path_lengths.append(length)
   
    # If we have duplicates, filter the geometry list and path lengths
    if len(unique_indices) < len(geometry_list):
        print(f"Warning: Removed {len(geometry_list) - len(unique_indices)} duplicate geometries from the path")
        unique_geometries = [geometry_list[i] for i in unique_indices]
        path_length_list = unique_path_lengths
        geometry_list = unique_geometries
        
    
    # If after removing duplicates we have too few points for the requested spline degree,
    # reduce the degree accordingly
    k = min(k, len(geometry_list) - 1)
    
    # If we have too few points for splines, fall back to linear interpolation
    if len(geometry_list) <= 2 or k < 1:
        print("Warning: Insufficient points for spline interpolation. Falling back to linear interpolation.")
        return distribute_geometry_by_length(geometry_list, angstrom_spacing)
    
    # Convert geometries to numpy arrays
    geom_arrays = [np.asarray(geom) for geom in geometry_list]
    original_shape = geom_arrays[0].shape
    flattened_geoms = [g.flatten() for g in geom_arrays]
    
    # Determine the number of coordinates to interpolate
    n_coords = len(flattened_geoms[0])
    
    # Create new geometry list with the first point
    new_geometry_list = []
    
    try:
        # Create splines for each coordinate
        splines = []
        for i in range(n_coords):
            # Extract the i-th coordinate from each geometry
            coord_values = [g[i] for g in flattened_geoms]
            # Create a spline for this coordinate
            spline = make_interp_spline(path_length_list, coord_values, k=k)
            splines.append(spline)
        
        # Determine sample points along the path including endpoints
        num_points = max(2, int(np.ceil(total_length / angstrom_spacing)) + 1)
        sample_distances = np.linspace(0, total_length, num_points)
        
        # Generate new geometries
        for dist in sample_distances:
            # Evaluate all splines at this distance
            interpolated_coords = [spline(dist) for spline in splines]
            # Reshape back to original geometry shape
            new_geom = np.array(interpolated_coords).reshape(original_shape)
            new_geometry_list.append(new_geom)
    
    except Exception as e:
        print(f"Warning: Spline interpolation failed with error: {e}. Falling back to linear interpolation.")
        return distribute_geometry_by_length(geometry_list, angstrom_spacing)
    
    # Replace first and last points with original endpoints to guarantee exact endpoints
    new_geometry_list[0] = geom_arrays[0]
    new_geometry_list[-1] = geom_arrays[-1]
    new_geometry_list.append(geometry_list[-1])
    
    return new_geometry_list



def distribute_geometry_spline(geometry_list, spline_degree=3):
    """
    Distribute geometries evenly along the path using spline interpolation
    
    Parameters:
    -----------
    geometry_list : list
        List of geometry arrays/objects
    spline_degree : int, optional
        Degree of the spline interpolation (default=3 for cubic splines)
        
    Returns:
    --------
    list
        New list of geometries distributed at regular intervals using spline interpolation
    """
    nnode = len(geometry_list)
    
    # Handle edge cases
    if nnode <= 2:
        return geometry_list.copy()
    
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    node_dist = total_length / (nnode-1)
    
    # Ensure spline degree is not greater than number of points minus 1
    k = min(spline_degree, nnode - 1)
    
    # Check for duplicate path lengths and handle them
    unique_indices = []
    unique_path_lengths = []
    for i, length in enumerate(path_length_list):
        if i == 0 or abs(length - path_length_list[i-1]) > 1e-10:
            unique_indices.append(i)
            unique_path_lengths.append(length)
    
    # If we have duplicates, filter the geometry list and path lengths
    if len(unique_indices) < nnode:
        filtered_geometries = [geometry_list[i] for i in unique_indices]
        unique_path_lengths_array = np.array(unique_path_lengths)
        
        # If after filtering we have too few points for spline, fall back to linear
        if len(filtered_geometries) <= k:
            return distribute_geometry(geometry_list)
            
        # Convert geometries to arrays for interpolation
        geom_arrays = [np.asarray(geom) for geom in filtered_geometries]
        
    else:
        # No duplicates found
        geom_arrays = [np.asarray(geom) for geom in geometry_list]
        unique_path_lengths_array = np.array(path_length_list)
    
    # Determine shape of geometries
    original_shape = geom_arrays[0].shape
    flattened_geoms = [g.flatten() for g in geom_arrays]
    n_coords = len(flattened_geoms[0])
    
    # Create new geometry list with the first point
    new_geometry_list = [geometry_list[0]]
    
    try:
        # Create splines for each coordinate
        splines = []
        for i in range(n_coords):
            # Extract the i-th coordinate from each geometry
            coord_values = [g[i] for g in flattened_geoms]
            # Create a spline for this coordinate
            spline = make_interp_spline(unique_path_lengths_array, coord_values, k=k)
            splines.append(spline)
        
        # Generate new geometries at evenly spaced positions
        for i in range(1, nnode-1):
            dist = i * node_dist
            
            # Evaluate all splines at this distance
            interpolated_coords = [spline(dist) for spline in splines]
            # Reshape back to original geometry shape
            new_geom = np.array(interpolated_coords).reshape(original_shape)
            new_geometry_list.append(new_geom)
        
    except Exception as e:
        # Fall back to linear interpolation if spline fails
        print(f"Warning: Spline interpolation failed with error: {e}. Falling back to linear interpolation.")
        return distribute_geometry(geometry_list)
    
    # Always include the last geometry
    new_geometry_list.append(geometry_list[-1])
    
    return new_geometry_list






    