import numpy as np
from scipy.interpolate import make_interp_spline, PPoly
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

def _fit_bspline_energy_path(s_vals, E_vals, gradient_norms=None, k=3):
    """
    Fits the energy profile using B-spline basis functions (Ritz approximation).
    Automatically handles boundary conditions for Cubic (k=3) and Quintic (k=5) splines.
    
    """
    bc_type = None
    if gradient_norms is not None:
        start_bc = [(1, gradient_norms[0])]
        end_bc = [(1, gradient_norms[-1])]
        
        # Consistent with higher order derivatives requirements
        if k >= 5:
            start_bc.append((2, 0.0))
            end_bc.append((2, 0.0))

        bc_type = (start_bc, end_bc)
    
    spline = make_interp_spline(s_vals, E_vals, k=k, bc_type=bc_type)
    return spline

def _find_spline_maxima(spline, s_min=0.05, s_max=0.95):
    """
    Finds local maxima on the B-spline curve.
    Uses a robust fallback (Grid Search + Brent's method) if 
    BSpline.roots() is unavailable in the environment.
    """
    # 1. Calculate derivative spline E'(s)
    deriv_spline = spline.derivative(nu=1)
    
    roots = []
    
    # --- Attempt 1: Native .roots() method (Newer SciPy) ---
    try:
        roots = deriv_spline.roots()
    except AttributeError:
        # --- Attempt 2: Fallback for older SciPy (Grid Search + Brentq) ---
        # Create a grid to detect sign changes
        grid_points = 200
        s_grid = np.linspace(0.0, 1.0, grid_points)
        y_grid = deriv_spline(s_grid)
        
        for i in range(len(s_grid) - 1):
            y1 = y_grid[i]
            y2 = y_grid[i+1]
            
            # Check for sign change
            if y1 * y2 < 0:
                try:
                    # Find exact root in this interval
                    root = brentq(deriv_spline, s_grid[i], s_grid[i+1])
                    roots.append(root)
                except ValueError:
                    continue
    
    # 3. Filter valid roots within range and check for maximum (E''(s) < 0)
    valid_maxima = []
    second_deriv_spline = spline.derivative(nu=2)
    
    for r in roots:
        # Check if root is real and within range
        if np.isreal(r) and s_min <= r.real <= s_max:
            r_real = r.real
            # Check concavity (Second derivative must be negative for a maximum)
            curvature = second_deriv_spline(r_real)
            if curvature < -1e-6: # Concave down -> Maximum
                energy = spline(r_real)
                valid_maxima.append((r_real, energy))
                
    return valid_maxima

def distribute_geometry_bspline_ritz(
    geometry_list,
    energy_list,
    gradient_list=None,
    n_points=None,
    spline_degree=3,
    use_gradient_corrections=True,
    concentration_factor=0.0
):
    """
    Redistributes geometry nodes based on B-spline Ritz approximation.
    
    Features:
    1. Fits continuous path using B-splines.
    2. Identifies the exact TS location on the spline.
    3. Optionally concentrates nodes around the high-energy region.

    Args:
        concentration_factor (float): 
            Controls how much nodes gather around the peak energy.
            0.0 = Uniform spacing (arc-length).
            > 0.0 = Higher density at high energy regions (mimicking MaxFlux behavior).
            Recommended values: 0.0 to 5.0.
    """
    geom = np.asarray(geometry_list, dtype=float)
    energies = np.asarray(energy_list, dtype=float)
    n_old = len(geom)
    
    if n_points is None:
        n_points = n_old

    # 1. Path Parameterization (Normalized Arc Length s in [0, 1])
    geom_flat = geom.reshape(n_old, -1)
    diffs = np.diff(geom_flat, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = s_cum[-1]
    
    if total_length < 1e-12 or n_old < 4:
        return geom.copy()
        
    s_norm = s_cum / total_length

    # 2. Fit Geometry Spline X(s) (Cubic for smooth path)
    geom_spline = make_interp_spline(s_norm, geom_flat, k=3)

    # 3. Fit Energy Spline E(s) (Ritz Ansatz)
    # Calculate projected gradients dE/ds
    grad_proj = None
    if gradient_list is not None and use_gradient_corrections:
        grads = np.asarray(gradient_list).reshape(n_old, -1)
        tangents = np.gradient(geom_flat, s_cum, axis=0)
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        t_norms[t_norms < 1e-12] = 1.0
        # Project and scale by L (since s is normalized)
        grad_proj = np.sum(grads * (tangents / t_norms), axis=1) * total_length

    energy_spline = _fit_bspline_energy_path(
        s_norm, 
        energies, 
        gradient_norms=grad_proj, 
        k=spline_degree
    )

    # 4. Determine New Node Distribution
    # Create a fine grid to evaluate the "Density Function"
    s_fine = np.linspace(0, 1, 1000)
    E_fine = energy_spline(s_fine)
    
    if concentration_factor > 1e-3:
        # --- Weighted Distribution (High density at high energy) ---
        # Ref:  "Evaluation points gather near the transition state"
        
        # Weight function W(s) ~ 1 + alpha * normalized_Energy(s)
        E_min = np.min(E_fine)
        E_max = np.max(E_fine)
        if E_max - E_min > 1e-6:
            E_scaled = (E_fine - E_min) / (E_max - E_min)
            # Use exponential weighting similar to Flux ~ exp(beta*E) but milder for stability
            weights = 1.0 + concentration_factor * (np.exp(2.0 * E_scaled) - 1.0)
        else:
            weights = np.ones_like(E_fine)
            
        # Calculate Cumulative Density Function (CDF)
        cdf = cumulative_trapezoid(weights, s_fine, initial=0)
        cdf /= cdf[-1] # Normalize to [0, 1]
        
        # Inverse Transform Sampling to find new s values
        # We want uniform steps in CDF space -> translates to variable steps in s space
        target_cdf = np.linspace(0, 1, n_points)
        s_new = np.interp(target_cdf, cdf, s_fine)
        
    else:
        # --- Align Peak only (Previous Logic) ---
        # Find exact TS
        maxima = _find_spline_maxima(energy_spline)
        if maxima:
            s_ts, _ = max(maxima, key=lambda x: x[1])
        else:
            s_ts = s_norm[np.argmax(energies)]
            
        # Align closest integer node to TS
        target_ts_idx = max(1, min(n_points - 2, int(round(s_ts * (n_points - 1)))))
        
        # Piecewise linear distribution ensuring TS hit
        s_new_1 = np.linspace(0.0, s_ts, target_ts_idx + 1)
        s_new_2 = np.linspace(s_ts, 1.0, n_points - target_ts_idx)
        s_new = np.concatenate([s_new_1[:-1], s_new_2])

    # 5. Generate New Geometry
    new_flat_geom = geom_spline(s_new)
    new_geom = new_flat_geom.reshape(n_points, geom.shape[1], 3)
    
    # Fix endpoints
    new_geom[0] = geom[0]
    new_geom[-1] = geom[-1]

    return new_geom