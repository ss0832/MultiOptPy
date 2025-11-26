import numpy as np
from multioptpy.Utils.calc_tools import calc_path_length_list


def _cumulative_path_length(geom):
    """Calculates cumulative path length along the string."""
    diffs = np.diff(geom, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    seg_lengths = np.nan_to_num(seg_lengths, nan=0.0)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])

def _estimate_curvature_and_tangents(gradients, geom):
    """
    Calculates curvature and tangent vectors at each node.
    Returns:
        curvatures: np.array (N,) - Second derivative along path
        tangents: np.array (N, atoms*3) - Normalized tangent vectors
    """
    n = len(geom)
    gradients = gradients.reshape(n, -1)
    geom = geom.reshape(n, -1)
    
    # 1. Calculate Tangents (Central Difference)
    tangents = np.zeros_like(geom)
    
    # Internal nodes: T_i ~ (x_{i+1} - x_{i-1})
    if n > 2:
        vecs = geom[2:] - geom[:-2]
        norms = np.linalg.norm(vecs, axis=1)
        norms = np.maximum(norms, 1e-12)
        tangents[1:-1] = vecs / norms[:, None]
        
    # Endpoints: Forward/Backward difference
    t_start = geom[1] - geom[0]
    tangents[0] = t_start / max(np.linalg.norm(t_start), 1e-12)
    
    t_end = geom[-1] - geom[-2]
    tangents[-1] = t_end / max(np.linalg.norm(t_end), 1e-12)

    # 2. Calculate Gradient Projections (Slope)
    grad_along = np.sum(gradients * tangents, axis=1)

    # 3. Calculate Curvature (Numerical differentiation of slope)
    curvature = np.zeros(n)
    dists = np.linalg.norm(geom[1:] - geom[:-1], axis=1) # Segment lengths
    
    for k in range(1, n - 1):
        ds_prev = dists[k-1]
        ds_next = dists[k]
        total_ds = ds_prev + ds_next
        
        if total_ds > 1e-10:
            # Centered finite difference of the first derivative
            curvature[k] = (grad_along[k+1] - grad_along[k-1]) / total_ds
            
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    
    return curvature, tangents, grad_along

def _solve_polynomial_max(s_vals, E_vals, g_vals, gamma_vals=None):
    """
    Fits a polynomial to 3 points and finds the local maximum.
    Calculates 3rd derivative to estimate asymmetry.
    
    Args:
        s_vals: [s_prev, 0, s_next] (relative arc lengths)
        E_vals: [E_prev, E_curr, E_next]
        g_vals: [g_prev, g_curr, g_next] (projected gradients)
        gamma_vals: [c_prev, c_curr, c_next] (curvatures, optional)
        
    Returns:
        s_max: Predicted arc length of TS (relative to center), or None if invalid.
    """
    # Coordinate scaling to avoid numerical issues with small s
    scale = max(abs(s_vals[0]), abs(s_vals[2]))
    if scale < 1e-12: return None
    
    s = np.array(s_vals) / scale
    E = np.array(E_vals)
    g = np.array(g_vals) * scale # Derivative scales with 1/scale
    
    # Construct Linear System: Ac = b
    # E(s) = sum(c_k * s^k)
    
    rows = []
    rhs = []
    
    # Degree of polynomial
    # If gamma is provided: 3 points * 3 constraints = 9 constraints -> Degree 8 (Octic)
    # If gamma is None: 3 points * 2 constraints = 6 constraints -> Degree 5 (Quintic)
    use_curvature = (gamma_vals is not None)
    degree = 8 if use_curvature else 5
    
    if use_curvature:
        gamma = np.array(gamma_vals) * (scale**2) # 2nd deriv scales with 1/scale^2

    for i in range(3):
        si = s[i]
        # Energy constraint: E(si)
        rows.append([si**k for k in range(degree + 1)])
        rhs.append(E[i])
        
        # Gradient constraint: E'(si)
        row_g = [0.0] * (degree + 1)
        for k in range(1, degree + 1):
            row_g[k] = k * si**(k-1)
        rows.append(row_g)
        rhs.append(g[i])
        
        # Curvature constraint: E''(si)
        if use_curvature:
            row_c = [0.0] * (degree + 1)
            for k in range(2, degree + 1):
                row_c[k] = k * (k-1) * si**(k-2)
            rows.append(row_c)
            rhs.append(gamma[i])

    try:
        # Solve for coefficients
        coeffs = np.linalg.solve(np.array(rows), np.array(rhs))
    except np.linalg.LinAlgError:
        return None

    # Find roots of the derivative (E'(s) = 0)
    deriv_coeffs = [k * coeffs[k] for k in range(1, degree + 1)]
    roots = np.roots(deriv_coeffs[::-1])
    
    # Filter real roots within the interval
    valid_roots = []
    s_min, s_max = s[0], s[2]
    
    for r in roots:
        if np.isreal(r):
            r_real = r.real
            # Allow slightly outside for prediction, but clamp tightly for safety
            if s_min * 1.1 <= r_real <= s_max * 1.1:
                # Check curvature (E''(s) < 0 for maximum)
                # Calculate 2nd derivative value at this root
                curvature_val = 0.0
                for k in range(2, degree + 1):
                    curvature_val += k * (k-1) * coeffs[k] * (r_real**(k-2))
                
                if curvature_val < -1e-5: # Must be concave (maximum)
                    # --- Added: Calculate 3rd Derivative for Asymmetry Check ---
                    deriv3_val = 0.0
                    for k in range(3, degree + 1):
                        deriv3_val += k * (k-1) * (k-2) * coeffs[k] * (r_real**(k-3))
                    
                    # Scale back to physical units
                    # 3rd derivative scales with 1/scale^3
                    true_deriv3 = deriv3_val / (scale**3)
                    true_curvature = curvature_val / (scale**2)
                    
                    fit_type = "Octic" if use_curvature else "Quintic"
                    print(f"  [{fit_type}] TS candidate found at s={r_real*scale:.4f}. "
                          f"Curvature={true_curvature:.4e}, 3rd Deriv={true_deriv3:.4e}")
                    
                    # Calculate Energy at this root
                    energy_val = np.polynomial.polynomial.polyval(r_real, coeffs)
                    valid_roots.append((r_real, energy_val))

    if not valid_roots:
        return None
        
    # Return the root with the highest energy
    best_s = max(valid_roots, key=lambda x: x[1])[0]
    
    return best_s * scale

def distribute_geometry_by_predicted_energy(
    geometry_list,
    energy_list,
    gradient_list,
    n_points=None,
    method="octic" # Options: 'quintic', 'octic'
):
    """
    Redistributes geometry nodes to align with predicted Transition State (TS).
    """
    geom = np.asarray(geometry_list, dtype=float)
    energies = np.asarray(energy_list, dtype=float)
    n_old = len(geom)

    if n_points is None:
        n_points = n_old
    
    natom = geom.shape[1]
    geom_flat = geom.reshape(n_old, -1)
    gradients = np.asarray(gradient_list, dtype=float).reshape(n_old, -1)

    # 1. Path length calculation
    s_cum = _cumulative_path_length(geom_flat)
    total_length = s_cum[-1]
    
    if total_length < 1e-12 or n_old < 3:
        return geom.copy()

    # 2. Calculate properties along the path
    curvatures, _, projected_gradients = _estimate_curvature_and_tangents(gradients, geom)

    # 3. Identify Anchors
    anchors = [(0, 0.0), (n_points - 1, total_length)]

    # Scan for local maxima
    for i in range(1, n_old - 1):
        if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
            
            s_prev = s_cum[i-1] - s_cum[i]
            s_curr = 0.0
            s_next = s_cum[i+1] - s_cum[i]
            
            s_vals = [s_prev, s_curr, s_next]
            E_vals = [energies[i-1], energies[i], energies[i+1]]
            g_vals = [projected_gradients[i-1], projected_gradients[i], projected_gradients[i+1]]
            c_vals = [curvatures[i-1], curvatures[i], curvatures[i+1]]
            
            s_ts_local = None
            print(f"NODE {i}: Detected local maximum at s={s_cum[i]:.4f}, E={energies[i]:.4f}")
            # --- Attempt Method 2: Octic (9-point) ---
            if method == "octic":
                s_ts_local = _solve_polynomial_max(s_vals, E_vals, g_vals, c_vals)
            
            # --- Fallback/Method 1: Quintic (6-point) ---
            if s_ts_local is None:
                # print(f"Node {i}: Fallback to Quintic fitting.")
                s_ts_local = _solve_polynomial_max(s_vals, E_vals, g_vals, None)

            # --- Finalize Position ---
            if s_ts_local is not None:
                s_ts_global = s_cum[i] + s_ts_local
                
                # Map old index 'i' to new index 'j'
                j = int(round(i * (n_points - 1) / (n_old - 1)))
                
                if 0 < j < n_points - 1:
                    anchors.append((j, s_ts_global))

    # 4. Process Anchors & Interpolate
    anchors.sort(key=lambda x: x[0])
    
    unique_anchors = [anchors[0]]
    for k in range(1, len(anchors)):
        curr_idx, curr_s = anchors[k]
        prev_idx, prev_s = unique_anchors[-1]
        
        if curr_idx > prev_idx:
            if curr_s <= prev_s: 
                curr_s = prev_s + 1e-6
            unique_anchors.append((curr_idx, curr_s))
            
    if unique_anchors[-1][0] != n_points - 1:
         unique_anchors.append((n_points - 1, total_length))

    # Construct Target Grid
    target_s = np.zeros(n_points)
    for k in range(len(unique_anchors) - 1):
        idx_start, s_start = unique_anchors[k]
        idx_end, s_end = unique_anchors[k+1]
        count = idx_end - idx_start + 1
        target_s[idx_start : idx_end + 1] = np.linspace(s_start, s_end, count)

    # Interpolate Geometry
    new_flat_geom = np.zeros((n_points, geom_flat.shape[1]))
    for dim in range(geom_flat.shape[1]):
        new_flat_geom[:, dim] = np.interp(target_s, s_cum, geom_flat[:, dim])

    new_geom = new_flat_geom.reshape(n_points, natom, 3)
    new_geom[0] = geom[0]
    new_geom[-1] = geom[-1]

    return new_geom

def distribute_geometry_by_length(geometry_list, angstrom_spacing):
    """Distribute geometries by specified distance spacing"""
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    new_geometry_list = []
    
    # Avoid zero length
    if total_length < 1e-8:
        return [geometry_list[0]]
    
    max_steps = int(total_length // angstrom_spacing)
    # Start point
    new_geometry_list.append(geometry_list[0])
    
    for i in range(1, max_steps + 1):
        dist = i * angstrom_spacing
        if dist >= total_length:
            break
       
        for j in range(len(path_length_list) - 1):
            if path_length_list[j] <= dist <= path_length_list[j+1]:
                delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
                new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
                new_geometry_list.append(new_geometry)
                break

    # Add the last point if it is far enough from the previous one,
    # following the original specification of appending the final geometry.
    if len(new_geometry_list) == 0 or np.linalg.norm(new_geometry_list[-1] - geometry_list[-1]) > 1e-4:
        new_geometry_list.append(geometry_list[-1])
        
    return new_geometry_list


def distribute_geometry(geometry_list):
    """Distribute geometries evenly along the path"""
    nnode = len(geometry_list)
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    
    if total_length < 1e-8:
        return list(geometry_list)

    node_dist = total_length / (nnode-1)
    
    new_geometry_list = [geometry_list[0]]
    for i in range(1, nnode-1):
        dist = i * node_dist
        # Search logic
        found = False
        for j in range(len(path_length_list)-1):
            if path_length_list[j] <= dist <= path_length_list[j+1]:
                delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
                new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
                new_geometry_list.append(new_geometry)
                found = True
                break
        if not found:
            # Safeguard for cases out of range due to numerical error
            new_geometry_list.append(geometry_list[-1])

    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list

def distribute_geometry_by_energy(geometry_list, energy_list, gradient_list=None, n_points=None, smoothing=0.1):
    """
    Distribute geometries concentrating on ALL high-energy regions (Multiple Peaks) 
    using Linear Interpolation.
    
    Improvements:
    - Uses 'gradient_list' (if provided) to calculate Energy Curvature for precise Peak Detection.
    - Concentrates nodes on secondary transition states as well as the global maximum.
    
    Args:
        geometry_list: list of np.ndarray
        energy_list: list of float
        gradient_list: list of np.ndarray (dE/dx). Optional.
        n_points: int, number of output nodes (default: same as input)
        smoothing: float, weighting factor for low energy regions
    
    Returns:
        new_geometry_list: list of np.ndarray
    """
    if len(geometry_list) != len(energy_list):
        raise ValueError("Length of geometry_list and energy_list must be the same.")

    if n_points is None:
        n_points = len(geometry_list)

    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    
    if total_length < 1e-8:
        return list(geometry_list)

    geometry_arr = np.array(geometry_list)
    energies = np.array(energy_list)
    n_nodes = len(energies)
    
    # --- 1. Calculate Global Energy Weights (Height) ---
    E_min = np.min(energies)
    E_max = np.max(energies)
    
    if E_max - E_min < 1e-6:
        w_global = np.zeros_like(energies)
    else:
        w_global = (energies - E_min) / (E_max - E_min)
        
    # --- 2. Calculate Local Peak Weights (Convexity) ---
    w_local = np.zeros_like(energies)
    
    if n_nodes > 2:
        # Identify Hills (Energy Higher than Neighbors)
        E_center = energies[1:-1]
        E_neighbors = (energies[:-2] + energies[2:]) / 2.0
        # Boolean mask for hill regions
        is_hill = E_center > E_neighbors
        
        if gradient_list is not None:
            # --- A. Use Gradients for Precision ---
            grad_arr = np.array(gradient_list)
            
            # Calculate Tangents
            vecs = geometry_arr[1:] - geometry_arr[:-1]
            vec_norms = np.linalg.norm(vecs, axis=(1,2))
            valid = vec_norms > 1e-10
            tangents = np.zeros_like(geometry_arr)
            tangents[:-1][valid] = vecs[valid] / vec_norms[valid][:, np.newaxis, np.newaxis]
            tangents[-1] = tangents[-2]
            
            # Project Gradient -> Slope (dE/ds)
            slopes = np.sum(grad_arr * tangents, axis=(1,2))
            
            # Curvature ~ Change in Slope
            slope_change = np.zeros_like(slopes)
            slope_change[1:-1] = slopes[2:] - slopes[:-2]
            
            # Assign weight based on curvature magnitude ONLY in Hill regions
            peak_metric = np.abs(slope_change[1:-1])
            w_local[1:-1][is_hill] = peak_metric[is_hill]
            
        else:
            # --- B. Fallback to Energy Convexity ---
            convexity = E_center - E_neighbors
            peak_score = np.maximum(convexity, 0.0)
            w_local[1:-1] = peak_score

        # Normalize local score
        p_max = np.max(w_local)
        if p_max > 1e-6:
            w_local /= p_max
            
        # Pad endpoints
        w_local[0] = w_local[1]
        w_local[-1] = w_local[-2]

    # --- 3. Combine Weights ---
    # Mix 50% Global Height + 50% Local Shape + Smoothing
    weights = 0.5 * w_global + 0.5 * w_local + smoothing
    
    # --- Calculate weighted cumulative distance ---
    segment_dists = np.diff(path_length_list)
    segment_weights = (weights[:-1] + weights[1:]) / 2.0
    weighted_segments = segment_dists * segment_weights
    
    cum_weighted_dist = np.concatenate(([0.0], np.cumsum(weighted_segments)))
    total_weighted_length = cum_weighted_dist[-1]
    
    # --- Determine new target distances ---
    target_weighted_grid = np.linspace(0, total_weighted_length, n_points)
    target_physical_dists = np.interp(target_weighted_grid, cum_weighted_dist, path_length_list)
    
    # --- Coordinate redistribution using linear interpolation ---
    new_geometry_list = []
    
    for dist in target_physical_dists:
        if dist <= 0:
            new_geometry_list.append(geometry_list[0])
            continue
        if dist >= total_length:
            new_geometry_list.append(geometry_list[-1])
            continue
            
        found = False
        for j in range(len(path_length_list) - 1):
            if path_length_list[j] <= dist <= path_length_list[j+1]:
                seg_len = path_length_list[j+1] - path_length_list[j]
                delta_t = 0 if seg_len < 1e-10 else (dist - path_length_list[j]) / seg_len
                
                vec = geometry_list[j+1] - geometry_list[j]
                new_geometry = geometry_list[j] + vec * delta_t
                new_geometry_list.append(new_geometry)
                found = True
                break
        
        if not found:
            new_geometry_list.append(geometry_list[-1])
            
    return new_geometry_list

