import numpy as np


def convergence_check(grad, MAX_FORCE_THRESHOLD, RMS_FORCE_THRESHOLD):
    """Check convergence based on maximum and RMS force thresholds.
    
    Parameters
    ----------
    grad : numpy.ndarray
        Gradient vector
    MAX_FORCE_THRESHOLD : float
        Maximum force threshold for convergence
    RMS_FORCE_THRESHOLD : float
        RMS force threshold for convergence
        
    Returns
    -------
    bool
        True if converged, False otherwise
    """
    max_force = abs(grad.max())
    rms_force = abs(np.sqrt((grad**2).mean()))
    if max_force < MAX_FORCE_THRESHOLD and rms_force < RMS_FORCE_THRESHOLD:
        return True
    else:
        return False


