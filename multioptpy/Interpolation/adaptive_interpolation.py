import numpy as np
from scipy.special import comb
from multioptpy.Utils.calc_tools import calc_path_length_list

def _get_bernstein_coords_at_t(structures, t_values):
    """
    Generate Bernstein interpolated coordinates at specific t (0.0 to 1.0) values.
    
    Bernstein polynomials provide a global smoothing effect, useful for 
    removing noise from the path in flat energy regions.
    """
    structures = np.array(structures)
    N = len(structures)
    path = []
    
    # Calculate Bernstein polynomial for each t
    for t in t_values:
        B_t = np.zeros_like(structures[0])
        for k in range(N):
            # nCr * (1-t)^(n-k) * t^k
            coef = comb(N-1, k) * (1-t)**(N-1-k) * t**k
            B_t += coef * structures[k]
        path.append(B_t)
        
    return np.array(path)

def _get_linear_coords_at_s(geometry_list, path_length_list, target_s):
    """
    Generate Linear interpolated coordinates at specific physical distances (s).
    
    Linear interpolation preserves local geometric features (kinks/curves) 
    strictly, preventing corner-cutting in curved valleys.
    """
    geometry_list = np.array(geometry_list)
    current_s = path_length_list
    
    flat_geom = geometry_list.reshape(len(geometry_list), -1)
    new_flat = np.zeros((len(target_s), flat_geom.shape[1]))
    
    # Interpolate each coordinate dimension independently
    for dim in range(flat_geom.shape[1]):
        new_flat[:, dim] = np.interp(target_s, current_s, flat_geom[:, dim])
        
    return new_flat.reshape(len(target_s), geometry_list.shape[1], 3)

def predict_hidden_ts_weights(geometry_list, energy_list, gradient_list, boost_factor=2.0):
    """
    Predict if a Transition State (TS) is hidden strictly between two nodes 
    using Cubic Hermite Interpolation.
    
    f(x) = a3*x^3 + a2*x^2 + a1*x + a0
    
    If a local maximum (f'(x)=0, f''(x)<0) is detected within a segment, 
    the weight of that segment is boosted to attract more nodes.
    
    Args:
        geometry_list: Coordinates
        energy_list: Energies
        gradient_list: Gradients (dE/dx). NOT Forces. If Forces, flip sign before passing.
        boost_factor: Weight multiplier if a hidden TS is found.
    """
    n_nodes = len(geometry_list)
    geometry_arr = np.array(geometry_list)
    energies = np.array(energy_list)
    gradients = np.array(gradient_list)
    
    # Initialize with zeros (additive weight)
    ts_weights = np.zeros(n_nodes)
    
    for i in range(n_nodes - 1):
        # 1. Segment properties
        vec = geometry_arr[i+1] - geometry_arr[i]
        L = np.linalg.norm(vec)
        
        if L < 1e-8: continue
        
        tangent = vec / L
        
        # 2. Boundary conditions
        y0 = energies[i]
        y1 = energies[i+1]
        
        # Project gradient onto tangent (slope)
        m0 = np.sum(gradients[i] * tangent)
        m1 = np.sum(gradients[i+1] * tangent)
        
        # 3. Cubic Hermite Coefficients
        # f(x) = a3*x^3 + a2*x^2 + a1*x + a0
        # Derived from: f(0)=y0, f(L)=y1, f'(0)=m0, f'(L)=m1
        a0 = y0
        a1 = m0
        a2 = (3 * (y1 - y0) / (L**2)) - ((2 * m0 + m1) / L)
        a3 = ((m0 + m1) / (L**2)) - (2 * (y1 - y0) / (L**3))
        
        # 4. Find Roots of Derivative (Stationary Points)
        # f'(x) = 3*a3*x^2 + 2*a2*x + a1 = 0
        roots = []
        if abs(a3) > 1e-10:
            discriminant = (2 * a2)**2 - 4 * (3 * a3) * a1
            if discriminant >= 0:
                sqrt_d = np.sqrt(discriminant)
                roots.append((-2 * a2 + sqrt_d) / (6 * a3))
                roots.append((-2 * a2 - sqrt_d) / (6 * a3))
        elif abs(a2) > 1e-10:
            # Quadratic case
            roots.append(-a1 / (2 * a2))
            
        # 5. Check if roots indicate a valid hidden TS
        found_ts = False
        for x in roots:
            # Check if strictly inside the segment (with small buffer)
            if 0.05 * L < x < 0.95 * L:
                # Check convexity (2nd derivative)
                # f''(x) = 6*a3*x + 2*a2
                curvature = 6 * a3 * x + 2 * a2
                if curvature < 0: # Concave down -> Maximum
                    found_ts = True
                    break
        
        if found_ts:
            # Boost both nodes connected to this segment
            ts_weights[i] += boost_factor
            ts_weights[i+1] += boost_factor
            
    return ts_weights

def adaptive_geometry_energy_interpolation(geometry_list, energy_list, gradient_list=None,
                                         n_points=None, smoothing=None, angle_threshold_deg=15.0):
    """
    Advanced Adaptive Interpolation for Reaction Paths.
    
    This function performs two major tasks:
    1. Node Distribution Control (Density):
       Concentrates nodes in high-energy regions and regions with high curvature 
       (using gradients and cubic prediction).
    
    2. Coordinate Mixing (Shape):
       Blends 'Bernstein' (Smooth) and 'Linear' (Accurate) interpolation.
       - Uses Linear when geometric curvature is high AND energy is steep (prevents corner-cutting).
       - Uses Bernstein when the path is flat or noisy (smoothes optimization).

    Args:
        geometry_list: List of atomic coordinates.
        energy_list: List of energies.
        gradient_list: List of energy gradients (dE/dx). 
                       NOTE: Pass Gradients, not Forces. (Force = -Gradient)
        n_points: Number of output nodes (default: same as input).
        smoothing: Base weight factor for density (None = Auto-calculated).
        angle_threshold_deg: Angle threshold to trigger Linear interpolation.
    """
    geometry_arr = np.array(geometry_list)
    energies = np.array(energy_list)
    
    if n_points is None:
        n_points = len(geometry_arr)
    
    n_nodes = len(geometry_arr)
    path_length_list = calc_path_length_list(geometry_arr)
    total_length = path_length_list[-1]
    
    if total_length < 1e-8:
        return geometry_arr

    # =========================================================================
    # STEP 1: Weight Calculation (Node Density Control)
    # =========================================================================
    
    if smoothing is None:
        # Heuristic: ensure at least ~1 node's worth of weight in valleys
        smoothing = 1.5 / n_nodes if n_nodes > 0 else 0.1

    # --- A. Global Energy Height ---
    E_min, E_max = np.min(energies), np.max(energies)
    if E_max - E_min < 1e-6:
        w_global = np.zeros_like(energies)
    else:
        w_global = (energies - E_min) / (E_max - E_min)

    # --- B. Local Peak Convexity (via Gradient) ---
    w_local = np.zeros_like(energies)
    if gradient_list is not None and n_nodes > 2:
        grad_arr = np.array(gradient_list)
        
        # Calculate Tangents
        vecs = geometry_arr[1:] - geometry_arr[:-1]
        vec_norms = np.linalg.norm(vecs, axis=(1,2))
        valid = vec_norms > 1e-10
        tangents = np.zeros_like(geometry_arr)
        tangents[:-1][valid] = vecs[valid] / vec_norms[valid][:, np.newaxis, np.newaxis]
        tangents[-1] = tangents[-2]
        
        # Project Gradient -> Slope
        slopes = np.sum(grad_arr * tangents, axis=(1,2))
        
        # Change in Slope ~ Curvature (2nd derivative)
        slope_change = np.zeros_like(slopes)
        slope_change[1:-1] = slopes[2:] - slopes[:-2]
        
        # Identify Hills (Center is higher than neighbors)
        E_neighbors = (energies[:-2] + energies[2:]) / 2.0
        is_hill = energies[1:-1] > E_neighbors
        
        # We value high negative slope change (convex cap) in hill regions
        peak_metric = np.abs(slope_change[1:-1])
        w_local[1:-1][is_hill] = peak_metric[is_hill]
        
        # Normalize
        if np.max(w_local) > 1e-6:
            w_local /= np.max(w_local)
        w_local[0], w_local[-1] = w_local[1], w_local[-2]

    # --- C. Hidden TS Prediction (Cubic Hermite) ---
    w_hidden_ts = np.zeros_like(energies)
    if gradient_list is not None:
        w_hidden_ts = predict_hidden_ts_weights(
            geometry_arr, energies, gradient_list, boost_factor=2.0
        )

    # Combine Weights
    # 30% Global Height, 40% Local Convexity, + Hidden TS Boost + Smoothing
    weights = 0.3 * w_global + 0.4 * w_local + w_hidden_ts + smoothing
    
    # Calculate New Node Positions (s)
    segment_dists = np.diff(path_length_list)
    segment_weights = (weights[:-1] + weights[1:]) / 2.0
    weighted_segments = segment_dists * segment_weights
    
    cum_weighted_dist = np.concatenate(([0.0], np.cumsum(weighted_segments)))
    target_weighted_grid = np.linspace(0, cum_weighted_dist[-1], n_points)
    target_s = np.interp(target_weighted_grid, cum_weighted_dist, path_length_list)
    
    # =========================================================================
    # STEP 2: Determine Mixing Ratio (Bernstein vs Linear)
    # =========================================================================
    
    # --- Geometric Curvature (Angle) ---
    vecs = geometry_arr[1:] - geometry_arr[:-1]
    norms = np.linalg.norm(vecs, axis=(1,2))
    valid_mask = norms > 1e-10
    tangents = np.zeros_like(vecs)
    tangents[valid_mask] = vecs[valid_mask] / norms[valid_mask][:, np.newaxis, np.newaxis]
    
    angle_scores = np.zeros(n_nodes)
    for i in range(1, n_nodes - 1):
        v_prev = tangents[i-1]
        v_next = tangents[i]
        dot_val = np.clip(np.sum(v_prev * v_next), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_val))
        angle_scores[i] = np.clip(angle_deg / (2.0 * angle_threshold_deg), 0.0, 1.0)
    angle_scores[0], angle_scores[-1] = angle_scores[1], angle_scores[-2]
    
    # --- Energy Steepness ---
    # Only use Linear if energy is changing rapidly (Wall/Slope)
    steepness_scores = np.zeros(n_nodes)
    if (E_max - E_min) > 1e-6:
        dE = np.abs(energies[2:] - energies[:-2])
        # Sensitivity: 20% of barrier height counts as steep
        steepness_scores[1:-1] = np.clip((dE / (E_max - E_min)) * 5.0, 0.0, 1.0)
        steepness_scores[0], steepness_scores[-1] = steepness_scores[1], steepness_scores[-2]
        
    # Mixing Alpha (0=Bernstein, 1=Linear)
    # Logic: AND condition. Must be Curved AND Steep to enforce Linear.
    # Otherwise, prefer Bernstein for smoothing.
    alpha_original = angle_scores * steepness_scores
    
    # =========================================================================
    # STEP 3: Generate & Blend
    # =========================================================================
    
    # Generate candidates at NEW positions
    coords_linear = _get_linear_coords_at_s(geometry_arr, path_length_list, target_s)
    
    target_t = target_s / total_length
    coords_bernstein = _get_bernstein_coords_at_t(geometry_arr, target_t)
    
    # Interpolate alpha to new positions
    alpha_resampled = np.interp(target_s, path_length_list, alpha_original)
    alpha_resampled = alpha_resampled[:, np.newaxis, np.newaxis]
    
    # Final Blend
    new_geometry = alpha_resampled * coords_linear + (1.0 - alpha_resampled) * coords_bernstein
    
    return new_geometry