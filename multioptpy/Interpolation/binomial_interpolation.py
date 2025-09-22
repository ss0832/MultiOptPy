import numpy as np
from scipy.special import comb

from multioptpy.Utils.calc_tools import calc_path_length_list

def bernstein_interpolation(structures, n_points=20):
    """
    Interpolate between arbitrary number of structures using Bernstein polynomials.
    
    Args:
        structures: list of np.ndarray, each of shape (n_atoms, 3)
        n_points: int, number of points to interpolate
        
    Returns:
        path: np.ndarray of shape (n_points, n_atoms, 3)
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
    
    Args:
        geometry_list: list of np.ndarray, each of shape (n_atoms, 3)
        angstrom_spacing: float, desired spacing
        
    Returns:
        new_geometry_list: list of np.ndarray, each of shape (n_atoms, 3)
    """
    print("Distributing geometries using Bernstein polynomial interpolation.")
    path_length_list = calc_path_length_list(geometry_list)
    interpolate_dist_list = np.arange(0, path_length_list[-1], angstrom_spacing)
    interpolate_dist_list = np.append(interpolate_dist_list, path_length_list[-1])
    t_values = interpolate_dist_list / path_length_list[-1]
    N = len(geometry_list)
    new_geometry_list = []
    for t in t_values:
        B_t = np.zeros_like(geometry_list[0])
        for k in range(N):
            coef = comb(N-1, k) * (1-t)**(N-1-k) * t**k
            B_t += coef * geometry_list[k]
        new_geometry_list.append(B_t)

    return np.array(new_geometry_list)

