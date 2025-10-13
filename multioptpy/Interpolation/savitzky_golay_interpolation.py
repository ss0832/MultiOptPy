import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from multioptpy.Utils.calc_tools import calc_path_length_list

def savitzky_golay_interpolation(structures, n_points=20, window_length=5, polyorder=2):
    """
    Interpolate between arbitrary number of structures using Savitzky-Golay filtering.
    
    Args:
        structures: list of np.ndarray, each of shape (n_atoms, 3)
        n_points: int, number of points to interpolate
        window_length: int, the length of the filter window (must be odd and greater than polyorder)
        polyorder: int, the order of the polynomial used to fit the samples
        
    Returns:
        path: np.ndarray of shape (n_points, n_atoms, 3)
    """
    print("Using Savitzky-Golay filter interpolation.")
    N = len(structures)
    if N < window_length:
        print("Not enough points for Savitzky-Golay filtering, falling back to linear interpolation.")
        # Fallback to linear interpolation if not enough points
        t_original = np.linspace(0, 1, N)
        t_interp = np.linspace(0, 1, n_points)
        structures = np.array(structures)
        path = []
        for atom_idx in range(structures.shape[1]):
            for coord in range(3):
                interp_func = interp1d(t_original, structures[:, atom_idx, coord], kind='linear')
                path.append(interp_func(t_interp))
        path = np.array(path).reshape(n_points, structures.shape[1], 3)
        return path
    
    structures = np.array(structures)
    # Apply Savitzky-Golay filter along the structure sequence
    smoothed_structures = np.zeros_like(structures)
    for atom_idx in range(structures.shape[1]):
        for coord in range(3):
            smoothed_structures[:, atom_idx, coord] = savgol_filter(structures[:, atom_idx, coord], window_length, polyorder)
    
    # Interpolate to n_points
    t_original = np.linspace(0, 1, N)
    t_interp = np.linspace(0, 1, n_points)
    path = []
    for atom_idx in range(structures.shape[1]):
        for coord in range(3):
            interp_func = interp1d(t_original, smoothed_structures[:, atom_idx, coord], kind='linear')
            path.append(interp_func(t_interp))
    path = np.array(path).reshape(n_points, structures.shape[1], 3)
    return path


def distribute_geometry_by_length_savgol(geometry_list, angstrom_spacing, window_length=5, polyorder=2):
    """
    Distribute geometries by specified distance spacing using Savitzky-Golay filtering.
    
    Args:
        geometry_list: list of np.ndarray, each of shape (n_atoms, 3)
        angstrom_spacing: float, desired spacing
        window_length: int, the length of the filter window (must be odd and greater than polyorder)
        polyorder: int, the order of the polynomial used to fit the samples
        
    Returns:
        new_geometry_list: list of np.ndarray, each of shape (n_atoms, 3)
    """
    print("Distributing geometries using Savitzky-Golay filtering.")
    path_length_list = calc_path_length_list(geometry_list)
    interpolate_dist_list = np.arange(0, path_length_list[-1], angstrom_spacing)
    interpolate_dist_list = np.append(interpolate_dist_list, path_length_list[-1])
    t_values = interpolate_dist_list / path_length_list[-1]
    
    N = len(geometry_list)
    if N < window_length:
        print("Not enough points for Savitzky-Golay filtering, falling back to linear interpolation.")
        # Fallback to linear interpolation if not enough points
        t_original = np.linspace(0, 1, N)
        geometry_list = np.array(geometry_list)
        new_geometry_list = []
        for t in t_values:
            interp_point = []
            for atom_idx in range(geometry_list.shape[1]):
                for coord in range(3):
                    interp_func = interp1d(t_original, geometry_list[:, atom_idx, coord], kind='linear')
                    interp_point.append(interp_func(t))
            new_geometry_list.append(np.array(interp_point).reshape(geometry_list.shape[1], 3))
        return np.array(new_geometry_list)
    
    geometry_list = np.array(geometry_list)
    # Apply Savitzky-Golay filter along the structure sequence
    smoothed_geometries = np.zeros_like(geometry_list)
    for atom_idx in range(geometry_list.shape[1]):
        for coord in range(3):
            smoothed_geometries[:, atom_idx, coord] = savgol_filter(geometry_list[:, atom_idx, coord], window_length, polyorder)
    
    # Interpolate to desired spacing
    t_original = np.linspace(0, 1, N)
    new_geometry_list = []
    for t in t_values:
        interp_point = []
        for atom_idx in range(geometry_list.shape[1]):
            for coord in range(3):
                interp_func = interp1d(t_original, smoothed_geometries[:, atom_idx, coord], kind='linear')
                interp_point.append(interp_func(t))
        new_geometry_list.append(np.array(interp_point).reshape(geometry_list.shape[1], 3))
    return np.array(new_geometry_list)


def savitzky_golay_interpolation_with_derivatives(structures, n_points=20, window_length=5, polyorder=2, deriv_order=1):
    """
    Interpolate between arbitrary number of structures using Savitzky-Golay filtering and compute derivatives.
    
    Args:
        structures: list of np.ndarray, each of shape (n_atoms, 3)
        n_points: int, number of points to interpolate
        window_length: int, the length of the filter window (must be odd and greater than polyorder)
        polyorder: int, the order of the polynomial used to fit the samples
        deriv_order: int, the order of the derivative to compute (0 for smoothed, 1 for first derivative, etc.)
        
    Returns:
        path: np.ndarray of shape (n_points, n_atoms, 3) if deriv_order == 0, or derivative array if deriv_order > 0
        derivatives: dict containing derivative arrays for each derivative order up to deriv_order
    """
    print(f"Using Savitzky-Golay filter interpolation with derivatives up to order {deriv_order}.")
    N = len(structures)
    if N < window_length:
        # Fallback to linear interpolation if not enough points
        t_original = np.linspace(0, 1, N)
        t_interp = np.linspace(0, 1, n_points)
        structures = np.array(structures)
        path = []
        derivatives = {}
        for d in range(deriv_order + 1):
            deriv_path = []
            for atom_idx in range(structures.shape[1]):
                for coord in range(3):
                    interp_func = interp1d(t_original, structures[:, atom_idx, coord], kind='linear')
                    if d == 0:
                        deriv_path.append(interp_func(t_interp))
                    else:
                        # Simple finite difference for derivatives in fallback
                        deriv_values = np.gradient(interp_func(t_interp), t_interp[1] - t_interp[0], edge_order=2)
                        for _ in range(d - 1):
                            deriv_values = np.gradient(deriv_values, t_interp[1] - t_interp[0], edge_order=2)
                        deriv_path.append(deriv_values)
            derivatives[f'deriv_{d}'] = np.array(deriv_path).reshape(n_points, structures.shape[1], 3)
            if d == 0:
                path = derivatives[f'deriv_{d}']
        return path, derivatives
    
    structures = np.array(structures)
    derivatives = {}
    for d in range(deriv_order + 1):
        deriv_path = []
        for atom_idx in range(structures.shape[1]):
            for coord in range(3):
                smoothed = savgol_filter(structures[:, atom_idx, coord], window_length, polyorder, deriv=d)
                deriv_path.append(smoothed)
        derivatives[f'deriv_{d}'] = np.array(deriv_path).reshape(N, structures.shape[1], 3)
    
    # Interpolate to n_points for each derivative
    t_original = np.linspace(0, 1, N)
    t_interp = np.linspace(0, 1, n_points)
    for d in range(deriv_order + 1):
        deriv_interp = []
        for atom_idx in range(structures.shape[1]):
            for coord in range(3):
                interp_func = interp1d(t_original, derivatives[f'deriv_{d}'][:, atom_idx, coord], kind='linear')
                deriv_interp.append(interp_func(t_interp))
        derivatives[f'deriv_{d}'] = np.array(deriv_interp).reshape(n_points, structures.shape[1], 3)
    
    path = derivatives['deriv_0']
    return path, derivatives


def distribute_geometry_by_length_savgol_with_derivatives(geometry_list, angstrom_spacing, window_length=5, polyorder=2, deriv_order=1):
    """
    Distribute geometries by specified distance spacing using Savitzky-Golay filtering and compute derivatives.
    
    Args:
        geometry_list: list of np.ndarray, each of shape (n_atoms, 3)
        angstrom_spacing: float, desired spacing
        window_length: int, the length of the filter window (must be odd and greater than polyorder)
        polyorder: int, the order of the polynomial used to fit the samples
        deriv_order: int, the order of the derivative to compute (0 for smoothed, 1 for first derivative, etc.)
        
    Returns:
        new_geometry_list: list of np.ndarray, each of shape (n_atoms, 3)
        derivatives: dict containing derivative arrays for each derivative order up to deriv_order
    """
    print(f"Distributing geometries using Savitzky-Golay filtering with derivatives up to order {deriv_order}.")
    path_length_list = calc_path_length_list(geometry_list)
    interpolate_dist_list = np.arange(0, path_length_list[-1], angstrom_spacing)
    interpolate_dist_list = np.append(interpolate_dist_list, path_length_list[-1])
    t_values = interpolate_dist_list / path_length_list[-1]
    
    N = len(geometry_list)
    if N < window_length:
        # Fallback to linear interpolation if not enough points
        t_original = np.linspace(0, 1, N)
        geometry_list = np.array(geometry_list)
        new_geometry_list = []
        derivatives = {}
        for d in range(deriv_order + 1):
            deriv_list = []
            for t in t_values:
                interp_point = []
                for atom_idx in range(geometry_list.shape[1]):
                    for coord in range(3):
                        interp_func = interp1d(t_original, geometry_list[:, atom_idx, coord], kind='linear')
                        if d == 0:
                            interp_point.append(interp_func(t))
                        else:
                            # Simple finite difference for derivatives in fallback
                            values = interp_func(t_original)
                            deriv_values = np.gradient(values, t_original[1] - t_original[0], edge_order=2)
                            for _ in range(d - 1):
                                deriv_values = np.gradient(deriv_values, t_original[1] - t_original[0], edge_order=2)
                            interp_func_deriv = interp1d(t_original, deriv_values, kind='linear')
                            interp_point.append(interp_func_deriv(t))
                deriv_list.append(np.array(interp_point).reshape(geometry_list.shape[1], 3))
            derivatives[f'deriv_{d}'] = np.array(deriv_list)
            if d == 0:
                new_geometry_list = derivatives[f'deriv_{d}']
        return new_geometry_list, derivatives
    
    geometry_list = np.array(geometry_list)
    derivatives = {}
    for d in range(deriv_order + 1):
        deriv_path = []
        for atom_idx in range(geometry_list.shape[1]):
            for coord in range(3):
                smoothed = savgol_filter(geometry_list[:, atom_idx, coord], window_length, polyorder, deriv=d)
                deriv_path.append(smoothed)
        derivatives[f'deriv_{d}'] = np.array(deriv_path).reshape(N, geometry_list.shape[1], 3)
    
    # Interpolate to desired spacing for each derivative
    t_original = np.linspace(0, 1, N)
    for d in range(deriv_order + 1):
        deriv_interp = []
        for t in t_values:
            interp_point = []
            for atom_idx in range(geometry_list.shape[1]):
                for coord in range(3):
                    interp_func = interp1d(t_original, derivatives[f'deriv_{d}'][:, atom_idx, coord], kind='linear')
                    interp_point.append(interp_func(t))
            deriv_interp.append(np.array(interp_point).reshape(geometry_list.shape[1], 3))
        derivatives[f'deriv_{d}'] = np.array(deriv_interp)
    
    new_geometry_list = derivatives['deriv_0']
    return new_geometry_list, derivatives