import numpy as np
import os

def calc_unit_tangent_vector(gradient):
    """
    Calculate the unit tangent vector from the gradient of a path.

    Parameters:
    gradient (np.ndarray): A 1D array representing the gradient of the path.

    Returns:
    np.ndarray: A 1D array representing the unit tangent vector.
    """
    norm = np.linalg.norm(gradient)
    if norm == 0:
        raise ValueError("The gradient vector has zero magnitude; cannot compute unit tangent vector.")
    unit_tangent_vector = gradient / norm
    return unit_tangent_vector

def calc_curvature_vector(gradient, prev_gradient, step_size):
    """
    Calculate the curvature vector from the current and previous gradients of a path.

    Parameters:
    gradient (np.ndarray): A 1D array representing the current gradient of the path.
    prev_gradient (np.ndarray): A 1D array representing the previous gradient of the path.
    step_size (float): The step size between the current and previous points.

    Returns:
    np.ndarray: A 1D array representing the curvature vector.
    """
    if step_size <= 0:
        raise ValueError("Step size must be a positive value.")
    
    curvature_vector = (gradient - prev_gradient) / step_size
    return curvature_vector


def calc_scalar_curvature(curvature_vector):
    """
    Calculate the scalar curvature from the current and previous gradients of a path.

    Parameters:
    gradient (np.ndarray): A 1D array representing the current gradient of the path.
    prev_gradient (np.ndarray): A 1D array representing the previous gradient of the path.
    step_size (float): The step size between the current and previous points.

    Returns:
    float: The scalar curvature.
    """
    scalar_curvature = np.linalg.norm(curvature_vector)
    return scalar_curvature


def calc_curvature_coupling(curvature_vector, hessian):
    """
    Calculate the curvature coupling from the curvature vector and Hessian matrix.

    Parameters:
    curvature_vector (np.ndarray): A 1D array representing the curvature vector.
    hessian (np.ndarray): A 2D array representing the Hessian matrix.

    Returns:
    float: The curvature coupling.
    """
    eigvals, eigvecs = np.linalg.eigh(hessian)

    sorted_indices = np.argsort(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    
    curvature_vector = curvature_vector.reshape(-1, 1)  # Ensure it's a column vector
    print("Only considering positive eigenvalue modes for curvature coupling.")
    # Mask out eigenvalues <= 1e-8 (positive eigenvalue components)
    mask = eigvals > 1e-8
    eigvecs_masked = eigvecs[:, mask]
    curvature_coupling = np.dot(eigvecs_masked.T, curvature_vector)
    return curvature_coupling

def calc_irc_curvature_properties(gradient, prev_gradient, hessian, step_size):
    """
    Calculate the unit tangent vector, curvature vector, scalar curvature, and curvature coupling
    for a given point along a path.

    Parameters:
    gradient (np.ndarray): A 1D array representing the current gradient of the path.
    prev_gradient (np.ndarray): A 1D array representing the previous gradient of the path.
    hessian (np.ndarray): A 2D array representing the Hessian matrix at the current point.
    step_size (float): The step size between the current and previous points.

    Returns:
    tuple: A tuple containing:
        - unit_tangent_vector (np.ndarray): The unit tangent vector.
        - curvature_vector (np.ndarray): The curvature vector.
        - scalar_curvature (float): The scalar curvature.   
        - curvature_coupling (np.ndarray): The curvature coupling.      
    """
    unit_tangent_vector = calc_unit_tangent_vector(gradient)
    curvature_vector = calc_curvature_vector(gradient, prev_gradient, step_size)
    scalar_curvature = calc_scalar_curvature(curvature_vector)
    curvature_coupling = calc_curvature_coupling(curvature_vector, hessian)
    
    return unit_tangent_vector, curvature_vector, scalar_curvature, curvature_coupling


def save_curvature_properties_to_file(filename, scalar_curvature, curvature_coupling):
    """
    Append curvature properties to a CSV file.

    Parameters:
    filename (str): Path to the CSV file to append to.
    scalar_curvature (float): The scalar curvature value.
    curvature_coupling (np.ndarray): The curvature coupling vector/array.
    """
    
    if not os.path.isfile(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            header = ["Scalar_Curvature"] + [f"Curvature_Coupling_{i+1}" for i in range(len(curvature_coupling))]
            f.write(",".join(header) + "\n")
    
    row = [f"{scalar_curvature:.6f}"]
    row += [f"{val:.6f}" for val in np.asarray(curvature_coupling).ravel()]
    line = ",".join(row) + "\n"
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(line)