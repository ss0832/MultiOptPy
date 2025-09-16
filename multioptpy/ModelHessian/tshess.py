import numpy as np

class TransitionStateHessian:
    """
    A class for modifying an existing Hessian matrix to introduce a negative eigenvalue 
    for transition state optimizations, but without using neg_eigenvalue directly as the replacement.
    Instead, it applies a procedure to the targeted eigenvalue: first take its absolute value,
    multiply by -1, and then add neg_eigenvalue.
    """

    def __init__(self):
        pass

    def create_ts_hessian(self, model_hessian, cart_gradient):
        
        # Diagonalize the supplied Hessian
        eigenvalues, eigenvectors = np.linalg.eigh(model_hessian)
     
        if np.any(eigenvalues < -1e-8):
            return model_hessian  # No need to modify if negative eigenvalues already exist
        
                
        count = 0
        for i in range(len(eigenvalues)):
            if abs(eigenvalues[i]) < 1e-8:
                count += 1
            else:
                break
   
        
        target_eigvec = eigenvectors[:, count]
   
        P = np.eye(len(eigenvalues)) - 2.0 * np.outer(target_eigvec, target_eigvec) 

        ts_hessian = np.dot(P, model_hessian)

        # Enforce symmetry
        ts_hessian = 0.5 * (ts_hessian + ts_hessian.T)

        return ts_hessian
