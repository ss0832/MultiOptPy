import numpy as np


def compute_natural_spline_coefficients(x, y):
    """
    Computes the coefficients for the natural cubic spline interpolation for data points (x, y).
    Returns:
      a: array of a coefficients (equals y values for each interval start)
      b: array of b coefficients for each interval
      c: array of c coefficients for each interval (from 0 to n-2)
      d: array of d coefficients for each interval
      h: array of interval lengths, where h[i] = x[i+1] - x[i]
    """
    n = len(x)
    a = np.array(y, dtype=float)
    h = np.diff(x)
    
    # Build the tri-diagonal system for c: second derivative coefficients.
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    
    # Natural spline boundary conditions: second derivatives at endpoints are 0.
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 3 * ((a[i+1]-a[i])/h[i] - (a[i]-a[i-1])/h[i-1])
    
    # Solve for c (second derivatives at data points)
    c_full = np.linalg.solve(A, rhs)
    
    # Compute b and d for each segment
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    # For each interval [x[i], x[i+1]]
    for i in range(n-1):
        b[i] = (a[i+1]-a[i])/h[i] - h[i]*(2*c_full[i] + c_full[i+1])/3
        d[i] = (c_full[i+1]-c_full[i])/(3*h[i])
    
    # We will use c coefficients per segment as c[i] (from point i)
    # Note: For evaluating spline on interval i, c_full[i] is used.
    return a, b, c_full, d, h

def evaluate_spline(x, a, b, c_full, d, xi):
    """
    Evaluate the natural cubic spline S(x) at a given point xi.
    Identifies the appropriate interval and computes:
      S(x) = a[i] + b[i]*(xi - x[i]) + c_full[i]*(xi-x[i])^2 + d[i]*(xi-x[i])^3
    """
    n = len(x)
    # Find the appropriate interval
    if xi <= x[0]:
        i = 0
    elif xi >= x[-1]:
        i = n - 2
    else:
        # find i such that x[i] <= xi < x[i+1]
        i = np.searchsorted(x, xi) - 1
    dx = xi - x[i]
    return a[i] + b[i]*dx + c_full[i]*dx**2 + d[i]*dx**3

def evaluate_spline_deriv(x, a, b, c_full, d, xi):
    """
    Evaluate the first derivative S'(x) of the spline at xi.
    S'(x) = b[i] + 2*c_full[i]*(xi-x[i]) + 3*d[i]*(xi-x[i])^2
    """
    n = len(x)
    if xi <= x[0]:
        i = 0
    elif xi >= x[-1]:
        i = n - 2
    else:
        i = np.searchsorted(x, xi) - 1
    dx = xi - x[i]
    return b[i] + 2*c_full[i]*dx + 3*d[i]*dx**2

def evaluate_spline_second_deriv(x, c_full, d, xi):
    """
    Evaluate the second derivative S''(x) of the spline at xi.
    S''(x) = 2*c_full[i] + 6*d[i]*(xi-x[i])
    """
    n = len(x)
    if xi <= x[0]:
        i = 0
    elif xi >= x[-1]:
        i = n - 2
    else:
        i = np.searchsorted(x, xi) - 1
    dx = xi - x[i]
    return 2*c_full[i] + 6*d[i]*dx

def newton_method(f, df, x0, tol=1e-10, max_iter=500):
    """
    Apply Newton's method to solve f(x)=0.
    Starts from initial guess x0, and iterates until convergence.
    """
    x_current = x0
    for _ in range(max_iter):
        f_val = f(x_current)
        df_val = df(x_current)
        if abs(df_val) < 1e-12:
            break
        x_new = x_current - f_val/df_val
        if abs(x_new - x_current) < tol:
            return x_new
        x_current = x_new
    return x_current

def find_extrema(x, a, b, c_full, d):
    """
    Find local extrema for the natural spline interpolation.
    For each spline segment [x[i], x[i+1]], we apply Newton's method to solve:
      S'_i(x) = 0,
    starting from the midpoint of the interval.
    
    We then classify the extremum using the second derivative:
      - If S''(x) < 0, it's a local maximum.
      - If S''(x) > 0, it's a local minimum.
      
    Returns two lists: local_maxima and local_minima.
    Each element is a tuple (xi, S(xi)).
    """
    local_maxima = []
    local_minima = []
    n = len(x)
    
    for i in range(n-1):
        def f(xi):
            return b[i] + 2*c_full[i]*(xi - x[i]) + 3*d[i]*(xi - x[i])**2
        def df(xi):
            return 2*c_full[i] + 6*d[i]*(xi - x[i])
        
       
        x0 = (x[i] + x[i+1]) / 2
        root = newton_method(f, df, x0)
     
        if root >= x[i] - 1e-10 and root <= x[i+1] + 1e-10:
            S_value = a[i] + b[i]*(root - x[i]) + c_full[i]*(root - x[i])**2 + d[i]*(root - x[i])**3
            second_deriv = 2*c_full[i] + 6*d[i]*(root - x[i])
            if second_deriv < 0:
                local_maxima.append((root, S_value))
            elif second_deriv > 0:
                local_minima.append((root, S_value))
    return local_maxima, local_minima

def spline_interpolation(x_data, y_data):
    resolution = 100000
    a, b, c_full, d, h = compute_natural_spline_coefficients(x_data, y_data)
    t_fine = np.linspace(x_data[0], x_data[-1], resolution)
    S_values = [evaluate_spline(x_data, a, b, c_full, d, t) for t in t_fine]
    local_maxima, local_minima = find_extrema(x_data, a, b, c_full, d)
    # distance, energy = zip(*local_maxima or *local_minima)
    return local_maxima, local_minima

