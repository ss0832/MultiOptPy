import numpy as np


def cart2polar(point, reference_point=None):
    point = np.array(point, dtype=float)
    if reference_point is not None:
        point = point - reference_point
    
    n = len(point)
    polar_coords = np.zeros(n)
    
    r = np.linalg.norm(point)
    polar_coords[0] = r
    
    if r < 1e-9:
        return polar_coords
    
    for i in range(n-2):
        norm_partial = np.linalg.norm(point[i:])
        if norm_partial == 0:
            polar_coords[i+1] = 0
        else:
            polar_coords[i+1] = np.arccos(point[i] / norm_partial)
    
    if n > 1:
        last_angle = np.arctan2(point[-1], point[-2])
        if last_angle < 0:
            last_angle += 2 * np.pi
        polar_coords[-1] = last_angle
    
    return polar_coords

def polar2cart(polar_coords, reference_point=None):
    r = polar_coords[0]
    n = len(polar_coords)
    
    if abs(r) < 1e-9:
        if reference_point is not None:
            return np.array(reference_point)
        return np.zeros(n)
    
    cartesian = np.zeros(n)
    cartesian[0] = r * np.cos(polar_coords[1]) if n > 1 else r
    
    for i in range(1, n-1):
        prod = r
        for j in range(1, i+1):
            prod *= np.sin(polar_coords[j])
        
        if i < n-1:
            cartesian[i] = prod * np.cos(polar_coords[i+1])
        else:
            cartesian[i] = prod
    
    if n > 1:
        prod = r
        for j in range(1, n-1):
            prod *= np.sin(polar_coords[j])
        cartesian[-1] = prod * np.sin(polar_coords[-1])
    
    if reference_point is not None:
        cartesian += np.array(reference_point)
    
    return cartesian

def compute_analytical_jacobian(p):
    """
    Compute the Jacobian matrix analytically for the transformation 
    from polar to Cartesian coordinates.
    
    Parameters:
    p (numpy.ndarray): Polar coordinates [r, θ₁, θ₂, ..., θₙ₋₁]
    
    Returns:
    numpy.ndarray: Jacobian matrix J where J[i,j] = ∂xᵢ/∂pⱼ
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    r = p[0]
    J = np.zeros((n, n))
    
    # Handle the special case of zero radius
    if r < 1e-10:
        J[0, 0] = 1  # ∂x₁/∂r = 1 (all others zero)
        return J
    
    # Derivatives with respect to r (first column of Jacobian)
    # For all coordinates: ∂xᵢ/∂r = xᵢ/r
    x = cart2polar(p)
    for i in range(n):
        J[i, 0] = x[i] / r
    
    if n <= 1:
        return J  # Only radius in 1D case
    
    # Derivatives with respect to θ₁ (second column)
    # ∂x₁/∂θ₁ = -r sin(θ₁)
    J[0, 1] = -r * np.sin(p[1])
    
    # For other coordinates, ∂xᵢ/∂θ₁ for i>1
    for i in range(1, n):
        # Replace sin(θ₁) with cos(θ₁) in the formula for xᵢ
        deriv = r * np.cos(p[1])
        
        # Multiply by the remaining terms sin(θⱼ) and cos/sin terms
        for j in range(2, i+1):
            deriv *= np.sin(p[j])
        
        if i < n-1:
            deriv *= np.cos(p[i+1])
        else:  # Last coordinate
            deriv *= np.sin(p[n-1])
        
        J[i, 1] = deriv
    
    # Derivatives with respect to other angles (remaining columns)
    for j in range(2, n):  # For each angle θⱼ, j=2...n-1
        # xₖ doesn't depend on θⱼ for k < j-1
        for k in range(0, j-1):
            J[k, j] = 0
        
        # For coordinates xₖ where k ≥ j-1
        for k in range(j-1, n):
            if k == j-1:
                # ∂x_{j-1}/∂θⱼ = -r·sin(θ₁)·...·sin(θⱼ₋₁)·sin(θⱼ)
                deriv = -r
                for m in range(1, j):
                    deriv *= np.sin(p[m])
                deriv *= np.sin(p[j])
                J[k, j] = deriv
            else:  # k > j-1
                # Start with radius
                deriv = r
                
                # Multiply by sin terms for angles θ₁...θⱼ₋₁
                for m in range(1, j):
                    deriv *= np.sin(p[m])
                
                # Replace sin(θⱼ) with cos(θⱼ) in the formula
                deriv *= np.cos(p[j])
                
                # Multiply by remaining sin terms for θⱼ₊₁...θₖ
                for m in range(j+1, k+1):
                    deriv *= np.sin(p[m])
                
                # For intermediate coordinates, multiply by cos(θₖ₊₁)
                if k < n-1:
                    deriv *= np.cos(p[k+1])
                else:  # Last coordinate
                    deriv *= np.sin(p[n-1])
                
                J[k, j] = deriv
    
    # Special handling for the last angle θₙ₋₁
    if n >= 3:
        # Only the last two coordinates depend on the last angle
        for i in range(0, n-2):
            J[i, n-1] = 0
        
        # Second-to-last coordinate: ∂x_{n-1}/∂θ_{n-1} = -r·sin(θ₁)·...·sin(θ_{n-2})·sin(θ_{n-1})
        if n >= 3:
            deriv = -r
            for j in range(1, n-1):
                deriv *= np.sin(p[j])
            J[n-2, n-1] = deriv
        
        # Last coordinate: ∂xₙ/∂θ_{n-1} = r·sin(θ₁)·...·sin(θ_{n-2})·cos(θ_{n-1})
        deriv = r
        for j in range(1, n-1):
            deriv *= np.sin(p[j])
        deriv *= np.cos(p[n-1])
        J[n-1, n-1] = deriv
    
    return J

def cart_grad_2_polar_grad(x, grad_x):
    """
    Transform gradient from Cartesian to polar coordinates.
    
    Parameters:
    x (numpy.ndarray): Cartesian coordinates where gradient is evaluated
    grad_x (numpy.ndarray): Gradient in Cartesian coordinates [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    Returns:
    numpy.ndarray: Gradient in polar coordinates [∂f/∂r, ∂f/∂θ₁, ..., ∂f/∂θₙ₋₁]
    """
    x = np.asarray(x, dtype=float)
    grad_x = np.asarray(grad_x, dtype=float)
    
    # Convert to polar coordinates
    p = cart2polar(x)
    
    # Compute the Jacobian matrix analytically
    J = compute_analytical_jacobian(p)
    
    # Transform gradient: ∇ₚf = J^T · ∇ₓf
    grad_p = np.dot(J.T, grad_x)
    
    return grad_p
