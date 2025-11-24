import numpy as np

from multioptpy.Utils.calc_tools import calc_path_length_list


def _cumulative_path_length(geom):
    diffs = np.diff(geom, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    seg_lengths = np.nan_to_num(seg_lengths, nan=0.0)  # NaN safety
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])

def _find_candidate_segments(energies, fraction=0.20):
    if len(energies) < 3:
        return list(range(len(energies)-1))
    e_range = np.ptp(energies)
    if e_range < 1e-12:
        return list(range(len(energies)-1))
    threshold = np.max(energies) - fraction * e_range
    cand = set()
    for i in range(len(energies)-1):
        if max(energies[i], energies[i+1]) >= threshold:
            cand.add(i)
        if 0 < i < len(energies)-2:
            if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
                cand.add(i-1)
                cand.add(i)
    return sorted(cand)

def _estimate_curvature(gradients, geom):
    """Second derivative of energy along path – fully safe version."""
    n = len(geom)
    if n < 3:
        return np.zeros(n)

    # Flatten coordinates for vector arithmetic
    gradients = gradients.reshape(n, -1)
    geom = geom.reshape(n, -1)
    
    diffs = np.diff(geom, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    norms = np.maximum(norms, 1e-12)

    tangents = diffs / norms[:, None]
    
    # Pad tangent array for centralized calculation
    tangents_padded = np.vstack([tangents[0], tangents, tangents[-1]]) 

    grad_along = np.zeros(n)
    for i in range(n):
        # Use central difference approximation for tangent projection (i-1 to i+1)
        # Note: This is an approximation of the true tangent at i
        # We use simple projection for this calculation's needs
        if 0 < i < n-1:
             tangent_vec = geom[i+1] - geom[i-1]
             tangent_norm = np.linalg.norm(tangent_vec)
             if tangent_norm > 1e-12:
                 tangent_vec /= tangent_norm
                 grad_along[i] = np.sum(gradients[i] * tangent_vec)
        else: # Boundary: Use adjacent segment vector
             tangent_vec = tangents[i-1] if i > 0 else tangents[i]
             grad_along[i] = np.sum(gradients[i] * tangent_vec)


    curvature = np.zeros(n)
    # Use distance of segments for ds
    dists = np.linalg.norm(geom[1:] - geom[:-1], axis=1)
    
    for k in range(1, n - 1):
        ds_prev = dists[k-1]
        ds_next = dists[k]
        total_ds = ds_prev + ds_next
        
        if total_ds > 1e-10:
             # Central difference of slopes: (slope[k+1] - slope[k-1]) / ds_total
             curvature[k] = (grad_along[k+1] - grad_along[k-1]) / total_ds
        
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature

# =============================================================================
# TS prediction – returns global arc length or None (Functions remain unchanged)
# =============================================================================

def _predict_ts_position_cubic(y0, y1, m0, m1, L):
    if L < 1e-12:
        return None

    h00 = lambda t: 2*t**3 - 3*t**2 + 1
    h10 = lambda t: t**3 - 2*t**2 + t
    h01 = lambda t: -2*t**3 + 3*t**2
    h11 = lambda t: t**3 - t**2

    def E(t):  return y0*h00(t) + L*m0*h10(t) + y1*h01(t) + L*m1*h11(t)
    def dE(t):
        return (y0*(6*t**2-6*t) + L*m0*(3*t**2-4*t+1) +
                y1*(-6*t**2+6*t) + L*m1*(3*t**2-2*t)) / L

    candidates = []
    for t0 in np.linspace(0.12, 0.88, 9):
        t = t0
        for _ in range(20):
            f = dE(t)
            if abs(f) < 1e-10: break
            fp = (dE(t+1e-7) - dE(t-1e-7)) / 2e-7
            if abs(fp) < 1e-14: break
            t -= f / fp
            t = np.clip(t, 0.0, 1.0)
        if 0.06 <= t <= 0.94 and abs(dE(t)) < 1e-6:
            if E(t) > max(y0, y1) + 1e-5:
                candidates.append(t)

    if not candidates:
        return None
    return max(candidates, key=E) * L


def _predict_ts_position_quintic(y0, y1, m0, m1, c0, c1, L):
    if L < 1e-12:
        return None

    h00 = lambda t: 1 - 10*t**3 + 15*t**4 - 6*t**5
    h10 = lambda t: t - 6*t**3 + 8*t**4 - 3*t**5
    h01 = lambda t: 10*t**3 - 15*t**4 + 6*t**5
    h11 = lambda t: -4*t**3 + 7*t**4 - 3*t**5
    h02 = lambda t: 0.5*t**2 - 1.5*t**3 + 1.5*t**4 - 0.5*t**5
    h12 = lambda t: 0.5*t**3 - t**4 + 0.5*t**5

    def E(t):
        return (y0*h00(t) + L*m0*h10(t) + y1*h01(t) + L*m1*h11(t) +
                L**2*c0*h02(t) + L**2*c1*h12(t))

    def dE(t):
        return (
            y0*(-30*t**2 + 60*t**3 - 30*t**4) +
            L*m0*(1 - 18*t**2 + 32*t**3 - 15*t**4) +
            y1*(30*t**2 - 60*t**3 + 30*t**4) +
            L*m1*(-12*t**2 + 28*t**3 - 15*t**4) +
            L**2*c0*(t - 4.5*t**2 + 6*t**3 - 2.5*t**4) +
            L**2*c1*(1.5*t**2 - 4*t**3 + 2.5*t**4)
        ) / L

    candidates = []
    for t0 in np.linspace(0.1, 0.9, 11):
        t = t0
        for _ in range(25):
            f = dE(t)
            if abs(f) < 1e-12: break
            fp = (dE(t+1e-7) - dE(t-1e-7)) / 2e-7
            if abs(fp) < 1e-14: break
            t -= f / fp
            t = np.clip(t, 0.0, 1.0)
        if 0.06 <= t <= 0.94 and abs(dE(t)) < 1e-7:
            if E(t) > max(y0, y1) + 2e-5:
                candidates.append(t)

    if not candidates:
        return None
    return max(candidates, key=E) * L

def distribute_geometry_by_predicted_energy(
    geometry_list,
    energy_list,
    gradient_list=None,
    n_points=None,
    method="quintic"
):
    """
    Redistributes geometry using Piecewise Linear Interpolation.
    
    1. Identifies local maxima (TS candidates).
    2. Predicts the exact TS arc-length position for each maximum.
    3. "Pins" the corresponding node index to this exact predicted position.
    4. Linearly interpolates (uniformly distributes) the nodes between these pinned points.

    Parameters
    ----------
    geometry_list : list or np.ndarray
        List of geometries (N, atoms, 3).
    energy_list : list or np.ndarray
        List of energies.
    gradient_list : list or np.ndarray
        List of gradients.
    n_points : int, optional
        Target number of nodes. Defaults to len(geometry_list).
    method : str
        'cubic' or 'quintic'.

    Returns
    -------
    np.ndarray
        The interpolated geometries.
    """
    geom = np.asarray(geometry_list, dtype=float)
    energies = np.asarray(energy_list, dtype=float)
    n_old = len(geom)

    if n_points is None:
        n_points = n_old
    
    natom = geom.shape[1]
    geom_flat = geom.reshape(n_old, -1)

    if gradient_list is None:
        gradients = np.zeros_like(geom_flat)
    else:
        gradients = np.asarray(gradient_list, dtype=float).reshape(n_old, -1)

    # 1. Path length calculation
    s_cum = _cumulative_path_length(geom_flat)
    total_length = s_cum[-1]
    
    if total_length < 1e-12 or n_old < 3:
        return geom.copy()

    # 2. Identify Anchors (Fixed Points)
    # An anchor is a tuple: (Target_Node_Index, Target_Arc_Length)
    # We start with the endpoints.
    anchors = [(0, 0.0), (n_points - 1, total_length)]

    # Calculate curvature once if needed
    curvatures = None
    if method == "quintic":
        curvatures = _estimate_curvature(gradients, geom)

    # Loop through internal nodes to find local maxima
    for i in range(1, n_old - 1):
        if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
            
            # --- Prediction Logic ---
            # Predict TS using segment (i-1) -> (i)
            i_prev = i - 1
            i_curr = i
            
            vec = geom_flat[i_curr] - geom_flat[i_prev]
            L = np.linalg.norm(vec)
            
            s_ts = None
            if L > 1e-12:
                tangent = vec / L
                y0, y1 = energies[i_prev], energies[i_curr]
                m0 = np.dot(gradients[i_prev], tangent)
                m1 = np.dot(gradients[i_curr], tangent)

                s_local = None
                if method == "quintic" and curvatures is not None:
                    s_local = _predict_ts_position_quintic(
                        y0, y1, m0, m1, curvatures[i_prev], curvatures[i_curr], L
                    )
                else:
                    s_local = _predict_ts_position_cubic(y0, y1, m0, m1, L)
                
                if s_local is not None:
                    s_ts = s_cum[i_prev] + s_local
            
            # Fallback: if prediction fails, use the current node's position
            if s_ts is None:
                s_ts = s_cum[i]

            # --- Mapping Logic ---
            # Map the old index 'i' to the new index 'j'
            # We preserve the relative position of the node in the chain.
            j = int(round(i * (n_points - 1) / (n_old - 1)))
            
            # Ensure we don't overwrite endpoints
            if 0 < j < n_points - 1:
                anchors.append((j, s_ts))

    # 3. Process Anchors
    # Sort anchors by index to ensure correct sequential processing
    anchors.sort(key=lambda x: x[0])

    # Remove duplicates (keep the one with higher energy or simply the first encountered?)
    # Here we perform a simple cleanup to ensure strictly increasing indices
    unique_anchors = [anchors[0]]
    for k in range(1, len(anchors)):
        curr_idx, curr_s = anchors[k]
        prev_idx, prev_s = unique_anchors[-1]
        
        # Only add if index is greater (avoid collision)
        if curr_idx > prev_idx:
            # Enforce physical ordering of arc length (TS must be after previous anchor)
            if curr_s <= prev_s: 
                curr_s = prev_s + 1e-6 # Tiny nudge to prevent collapse
            unique_anchors.append((curr_idx, curr_s))
            
    # Ensure the last anchor is strictly the endpoint
    if unique_anchors[-1][0] != n_points - 1:
         unique_anchors.append((n_points - 1, total_length))

    # 4. Construct Target Grid (Piecewise Linear)
    target_s = np.zeros(n_points)
    
    for k in range(len(unique_anchors) - 1):
        idx_start, s_start = unique_anchors[k]
        idx_end, s_end = unique_anchors[k+1]
        
        # Number of points in this segment (inclusive)
        count = idx_end - idx_start + 1
        
        # Linear interpolation for this segment
        segment_values = np.linspace(s_start, s_end, count)
        
        # Assign to the target array
        target_s[idx_start : idx_end + 1] = segment_values

    # 5. Interpolate Geometry
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

