import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d

from multioptpy.Utils.calc_tools import calc_path_length_list

def bernstein_interpolation(structures, n_points=20):
    """
    Interpolate between arbitrary number of structures using Bernstein polynomials.
    """
    print("Using Bernstein polynomial interpolation.")
    N = len(structures)
    t_values = np.linspace(0, 1, n_points)
    path = []
    structures = np.array(structures)
    for t in t_values:
        B_t = np.zeros_like(structures[0])
        for k in range(N):
            coef = comb(N-1, k) * (1-t)**(N-1-k) * t**k
            B_t += coef * structures[k]
        path.append(B_t)
    return np.array(path)


def distribute_geometry_by_length_bernstein(geometry_list, angstrom_spacing):
    """
    Distribute geometries by specified distance spacing using Bernstein polynomial interpolation.
    """
    print("Distributing geometries using Bernstein polynomial interpolation.")
    path_length_list = calc_path_length_list(geometry_list)
    
    # Avoid error if total length is 0
    total_length = path_length_list[-1]
    if total_length < 1e-8:
        return np.array(geometry_list)

    interpolate_dist_list = np.arange(0, total_length, angstrom_spacing)
    # Add the last point if it's missing, taking care not to duplicate
    if interpolate_dist_list[-1] < total_length:
        interpolate_dist_list = np.append(interpolate_dist_list, total_length)
        
    t_values = interpolate_dist_list / total_length
    N = len(geometry_list)
    new_geometry_list = []
    
    # Cast to array for speed
    geometry_arr = np.array(geometry_list)
    
    for t in t_values:
        B_t = np.zeros_like(geometry_arr[0])
        for k in range(N):
            coef = comb(N-1, k) * (1-t)**(N-1-k) * t**k
            B_t += coef * geometry_arr[k]
        new_geometry_list.append(B_t)

    return np.array(new_geometry_list)

def distribute_geometry_by_energy_bernstein(geometry_list, energy_list, gradient_list=None, n_points=None, smoothing=0.1):
    """
    Distribute geometries concentrating on ALL high-energy regions (Multiple Peaks)
    using Bernstein polynomial interpolation.
    
    Improvements:
    - Uses 'gradient_list' (if provided) for high-precision Peak Detection.
    - Concentrates nodes on secondary transition states as well as the global maximum.
    
    Args:
        geometry_list: list of np.ndarray
        energy_list: list of float
        gradient_list: list of np.ndarray (dE/dx). Optional.
        n_points: int, number of output nodes
        smoothing: float, prevents node density from becoming zero
        
    Returns:
        new_geometry_list: np.ndarray
    """
    print("Distributing geometries with Energy-Weighting (Multi-Peak) using Bernstein polynomial interpolation.")
    
    if len(geometry_list) != len(energy_list):
        raise ValueError("Length of geometry_list and energy_list must be the same.")
        
    if n_points is None:
        n_points = len(geometry_list)

    geometry_arr = np.array(geometry_list)
    energies = np.array(energy_list)
    n_nodes = len(energies)

    # 1. Calculate the current physical path length
    path_length_list = calc_path_length_list(geometry_list) 
    total_physical_length = path_length_list[-1]
    
    if total_physical_length < 1e-8:
        return geometry_arr

    current_s = path_length_list / total_physical_length

    # 2. Calculate Weights (Global + Local)
    
    # --- A. Global Energy Weights ---
    E_min = np.min(energies)
    E_max = np.max(energies)
    
    if E_max - E_min < 1e-6:
        w_global = np.zeros_like(energies)
    else:
        w_global = (energies - E_min) / (E_max - E_min)
        
    # --- B. Local Peak Weights ---
    w_local = np.zeros_like(energies)
    
    if n_nodes > 2:
        E_center = energies[1:-1]
        E_neighbors = (energies[:-2] + energies[2:]) / 2.0
        is_hill = E_center > E_neighbors
        
        if gradient_list is not None:
            # Gradient-based Curvature
            grad_arr = np.array(gradient_list)
            
            vecs = geometry_arr[1:] - geometry_arr[:-1]
            vec_norms = np.linalg.norm(vecs, axis=(1,2))
            valid = vec_norms > 1e-10
            tangents = np.zeros_like(geometry_arr)
            tangents[:-1][valid] = vecs[valid] / vec_norms[valid][:, np.newaxis, np.newaxis]
            tangents[-1] = tangents[-2]
            
            slopes = np.sum(grad_arr * tangents, axis=(1,2))
            slope_change = np.zeros_like(slopes)
            slope_change[1:-1] = slopes[2:] - slopes[:-2]
            
            peak_metric = np.abs(slope_change[1:-1])
            w_local[1:-1][is_hill] = peak_metric[is_hill]
            
        else:
            # Energy-based Convexity
            convexity = E_center - E_neighbors
            peak_score = np.maximum(convexity, 0.0)
            w_local[1:-1] = peak_score
            
        # Normalize
        p_max = np.max(w_local)
        if p_max > 1e-6:
            w_local /= p_max
            
        w_local[0] = w_local[1]
        w_local[-1] = w_local[-2]

    # --- C. Combine Weights ---
    weights = 0.5 * w_global + 0.5 * w_local + smoothing

    # 3. Integration of Weighted Arc Length
    segment_dists = np.diff(path_length_list)
    segment_weights = (weights[:-1] + weights[1:]) / 2.0
    weighted_segments = segment_dists * segment_weights
    
    cum_weighted_dist = np.concatenate(([0.0], np.cumsum(weighted_segments)))
    total_weighted_length = cum_weighted_dist[-1]

    # 4. Create new grid in Weighted Space
    target_weighted_grid = np.linspace(0, total_weighted_length, n_points)

    # 5. Inverse mapping: Weighted grid -> Physical distance (s)
    target_physical_s = np.interp(target_weighted_grid, cum_weighted_dist, current_s)
    
    t_values = target_physical_s

    # 6. Coordinate generation using Bernstein polynomials
    N = len(geometry_arr)
    new_geometry_list = []
    
    for t in t_values:
        B_t = np.zeros_like(geometry_arr[0])
        for k in range(N):
            coef = comb(N-1, k) * (1-t)**(N-1-k) * t**k
            B_t += coef * geometry_arr[k]
        new_geometry_list.append(B_t)

    return np.array(new_geometry_list)