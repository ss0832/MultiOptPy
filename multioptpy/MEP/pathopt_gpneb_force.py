import numpy as np
import copy
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from multioptpy.Coordinate.redundant_coordinate import calc_int_grad_from_pBmat, calc_cart_grad_from_pBmat

def extremum_list_index(energy_list):
    local_max_energy_list_index = argrelextrema(energy_list, np.greater)
    inverse_energy_list = (-1)*energy_list
    local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)

    local_max_energy_list_index = local_max_energy_list_index[0].tolist()
    local_min_energy_list_index = local_min_energy_list_index[0].tolist()
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    return local_max_energy_list_index, local_min_energy_list_index

#########################
# Chunked RBF Kernel Utilities
#########################
def rbf_kernel_chunked(X1, X2, sigma_f, length_scale, chunk_size=1024):
    """
    Compute the RBF kernel matrix between X1 (N1, d) and X2 (N2, d) in chunks
    to reduce memory usage.
    Args:
        X1: (N1, d) array
        X2: (N2, d) array
        sigma_f: scalar, kernel amplitude
        length_scale: scalar, kernel length scale
        chunk_size: int, number of X1 rows to process in each chunk
    Returns:
        (N1, N2) RBF kernel matrix
    """
    N1, d = X1.shape
    N2 = X2.shape[0]
    K = np.zeros((N1, N2), dtype=np.float64)
    
    # Precompute norms for X2 to help with chunk-based distance calculations
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    
    start = 0
    while start < N1:
        end = min(start + chunk_size, N1)
        X1_chunk = X1[start:end]
        
        # Compute squared distances for chunk
        X1_sq = np.sum(X1_chunk**2, axis=1).reshape(-1, 1)
        dist_sq = X1_sq + X2_sq - 2.0 * np.dot(X1_chunk, X2.T)
        
        exponent = -0.5 * dist_sq / (length_scale**2)
        # Clip the exponent to avoid overflow in exp. The upper bound 700 is conservative.
        exponent_clipped = np.clip(exponent, -1000, 1000)
        K[start:end, :] = (sigma_f**2) * np.exp(exponent_clipped)
        start = end
    
    return K

def rbf_kernel_grad_x_chunked(X1, X2, sigma_f, length_scale, chunk_size=1024):
    """
    Compute gradient of the RBF kernel with respect to X1 in a chunked manner.
    Output shape: (N1, N2, d).
    
    Args:
        X1: (N1, d)
        X2: (N2, d)
        sigma_f: float
        length_scale: float
        chunk_size: int
    
    Returns:
        grad: (N1, N2, d) array
    """
    N1, d = X1.shape
    N2 = X2.shape[0]
    grad = np.zeros((N1, N2, d), dtype=np.float64)
    
    start = 0
    while start < N1:
        end = min(start + chunk_size, N1)
        X1_chunk = X1[start:end]  # (chunk_section, d)
        
        # Kernel block for chunk
        K_chunk = rbf_kernel_chunked(X1_chunk, X2, sigma_f, length_scale, chunk_size=chunk_size)
        
        # diff array for chunk => shape (chunk_section, N2, d)
        diff_chunk = X1_chunk[:, np.newaxis, :] - X2[np.newaxis, :, :]
        
        # Gradient formula: (∂k/∂x1) = -diff / l^2 * k
        K_chunk_3d = K_chunk[..., np.newaxis]  # shape becomes (chunk_section, N2, 1)
        grad_chunk = -diff_chunk / (length_scale**2) * K_chunk_3d
        
        grad[start:end, :, :] = grad_chunk
        start = end
    
    return grad

def rbf_kernel_hessian_chunked(X1, X2, sigma_f, length_scale, chunk_size=512):
    """
    Compute Hessian of the RBF kernel with respect to x1, x2:
    Each entry H[a,b] = ∂^2 k(x1, x2) / (∂x1_a ∂x2_b),
    for all pairs in X1, X2. Returns array of shape (N1, N2, d, d).
    
    Args:
        X1: (N1, d)
        X2: (N2, d)
        sigma_f: float
        length_scale: float
        chunk_size: int
    
    Returns:
        hessian: (N1, N2, d, d)
    """
    N1, d = X1.shape
    N2 = X2.shape[0]
    hessian = np.zeros((N1, N2, d, d), dtype=np.float64)
    
    # We'll compute the kernel block and then apply the known Hessian formula:
    # H[a,b] = k * [ δ(a,b) / l^2 - (x1[a] - x2[a])*(x1[b] - x2[b]) / l^4 ]
    start = 0
    while start < N1:
        end = min(start + chunk_size, N1)
        X1_chunk = X1[start:end]  # shape (chunk_section, d)
        
        # k values (chunk_section, N2)
        K_chunk = rbf_kernel_chunked(X1_chunk, X2, sigma_f, length_scale, chunk_size=chunk_size)
        diff_chunk = X1_chunk[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (chunk_section, N2, d)
        
        # Expand for shape (chunk_section, N2, 1, 1)
        K_4d = K_chunk[:, :, np.newaxis, np.newaxis]
        
        # diff outer product => (chunk_section, N2, d, d)
        diff_outer = np.einsum('...i,...j->...ij', diff_chunk, diff_chunk)
        
        # Identity for each (d, d)
        eye_d = np.eye(d, dtype=np.float64).reshape(1, 1, d, d)
        
        # Hessian formula
        # H = K * [ (I / l^2) - (diff_outer / l^4) ]
        factor1 = eye_d / (length_scale**2)
        factor2 = diff_outer / (length_scale**4)
        
        hess_block = K_4d * (factor1 - factor2)
        hessian[start:end] = hess_block
        start = end
    
    return hessian

##################################
# Full Gaussian Process Regressor for Energy and Force
##################################

class GaussianProcessRegressor:
    """
    Gaussian Process Regressor for energies and forces with full kernel
    computations. The training block matrix is built as:
        K = [ K_EE    K_EF ]
            [ K_FE    K_FF ]
    where:
      - K_EE[i,j] = k(x_i, x_j)
      - K_EF[i, j] = -∂k(x_i, x_j)/∂x_j   (each block is 1 x d)
      - K_FE[i, j] = -∂k(x_i, x_j)/∂x_i   (each block is d x 1)
      - K_FF[i,j] = ∂^2k(x_i, x_j)/(∂x_i ∂x_j)   (each block is d x d)
    The target vector is: Y = [E; F] where energies E are (N,) and forces F are (N,d).
    """

    def __init__(self):
        """
        Args:
            sigma_f: RBF kernel amplitude.
            length_scale: RBF kernel length scale.
            noise_e: Noise standard deviation for energies.
            noise_f: Noise standard deviation for forces.
            chunk_size: Chunk size for kernel computations.
        """
        # Training data
        self.X = None  # shape (N, d)
        self.E = None  # shape (N,)
        self.F = None  # shape (N, d)
        
        # Optimized hyperparameters and alpha for predictions
        self.theta_opt = None
        self.alpha = None

    def _build_block_matrix_chunked(self, X, E, F, sigma_f, length_scale, noise_e, noise_f):
        """
        Build the full block kernel matrix in a chunked manner.
        Returns:
            K_full: (N + N*d, N + N*d) kernel matrix.
            Y: (N + N*d,) target vector.
        """
        N, d = X.shape
        
        # K_EE: Energy vs Energy, shape (N, N)
        K_EE = rbf_kernel_chunked(X, X, sigma_f, length_scale, chunk_size=self.chunk_size)
        
        # K_EF: Energy vs Force, shape (N, N*d)
        # For each training pair (i,j), block = -∂k(x_i, x_j)/∂x_j.
        # Compute gradient with respect to first argument for (X, X) then use symmetry.
        grad_X = rbf_kernel_grad_x_chunked(X, X, sigma_f, length_scale, chunk_size=self.chunk_size)  # shape (N, N, d)
        # For derivative w.r.t second argument, we have: ∂k(x_i, x_j)/∂x_j = -∂k(x_j, x_i)/∂x_j.
        # Therefore, K_EF[i, j] = -∂k(x_i, x_j)/∂x_j = grad_X[j, i]
        K_EF = np.zeros((N, N * d), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                K_EF[i, j * d:(j + 1) * d] = grad_X[j, i]
        
        # K_FE: Force vs Energy, shape (N*d, N)
        # For each training pair (i,j), block = -∂k(x_i, x_j)/∂x_i.
        # That is simply: K_FE[i, j] = -grad_X[i, j]
        K_FE = np.zeros((N * d, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                K_FE[i * d:(i + 1) * d, j] = -grad_X[i, j]
        
        # K_FF: Force vs Force, shape (N*d, N*d)
        # For each pair (i,j), block = ∂^2 k(x_i, x_j)/(∂x_i ∂x_j)
        H = rbf_kernel_hessian_chunked(X, X, sigma_f, length_scale, chunk_size=self.chunk_size)  # shape (N, N, d, d)
        K_FF = np.zeros((N * d, N * d), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                K_FF[i * d:(i + 1) * d, j * d:(j + 1) * d] = H[i, j]
        
        # Add noise to diagonal blocks
        K_EE += (noise_e**2) * np.eye(N, dtype=np.float64)
        K_FF += (noise_f**2) * np.eye(N * d, dtype=np.float64)
        
        # Combine blocks into full matrix:
        # K_full = [ K_EE    K_EF ]
        #          [ K_FE    K_FF ]
        top = np.hstack((K_EE, K_EF))
        bottom = np.hstack((K_FE, K_FF))
        K_full = np.vstack((top, bottom))
        
        # Construct target vector Y = [E; F_vectorized]
        Y = np.concatenate([E, F.reshape(-1)], axis=0)
        return K_full, Y

    def _neg_log_marginal_likelihood(self, params):
        """
        Negative log marginal likelihood function.
        params: [sigma_f, length_scale, noise_e, noise_f]
        """
        sigma_f, length_scale, noise_e, noise_f = params
        K, Y = self._build_block_matrix_chunked(self.X, self.E, self.F,
                                                 sigma_f, length_scale, noise_e, noise_f)
        N, d = self.X.shape
        n_data = N + N * d
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            logdetK = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            return 1e20
        
        nll = 0.5 * np.dot(Y, alpha) + 0.5 * logdetK + 0.5 * n_data * np.log(2.0 * np.pi)
        print("NLML func:", nll)
        return nll

    def fit(self, X, E, F, initial_params=(1.0, 1.0, 1e-3, 1e-3)):
        """
        Fit the GPR model to energy and force data.
        Args:
            X: (N, d) array of positions.
            E: (N,) array of energy values.
            F: (N, d) array of force values.
            initial_params: Tuple of initial hyperparameters (sigma_f, length_scale, noise_e, noise_f).
        """
        self.X = X
        self.E = E
        self.F = F
        
        bounds = [(1e-9, None), (1e-9, None), (1e-9, None), (1e-9, None)]
        res = minimize(self._neg_log_marginal_likelihood, x0=initial_params, bounds=bounds, method='L-BFGS-B', tol=1e-10)
        self.theta_opt = res.x
        self._compute_alpha()
        
        self.sigma_f, self.length_scale, self.noise_e, self.noise_f = self.theta_opt

    def _compute_alpha(self):
        """
        Compute alpha = K^{-1} Y with the optimized hyperparameters.
        """
        sigma_f, length_scale, noise_e, noise_f = self.theta_opt
        K, Y = self._build_block_matrix_chunked(self.X, self.E, self.F,
                                                 sigma_f, length_scale, noise_e, noise_f)
        L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))

    def predict_energy_and_forces(self, X_star):
        """
        Predict the energy and forces at new positions X_star.
        For each test point s:
          E(s) = k(s, X) * alpha_E + [-∂k(s, X)/∂x] * alpha_F,
          F(s) = [-∂k(s, X)/∂s] * alpha_E + [∂^2k(s, X)/(∂s ∂x)] * alpha_F,
        where alpha_E = self.alpha[:N] and alpha_F = self.alpha[N:].reshape(N, d).
        Args:
            X_star: (M, d) array of positions.
        Returns:
            E_pred: (M,) predicted energies.
            F_pred: (M, d) predicted forces.
        """
        sigma_f, length_scale, noise_e, noise_f = self.theta_opt
        X_train = self.X
        N, d = X_train.shape
        M = X_star.shape[0]
        
        # Compute K_star_EE: k(X_star, X_train), shape (M, N)
        K_star_EE = rbf_kernel_chunked(X_star, X_train, sigma_f, length_scale, chunk_size=self.chunk_size)
        
        # Compute derivative of k w.r.t test input: ∇_{s} k(s, x)
        # This is used for K_star_FE: -∇_s k(s, x)
        grad_test = rbf_kernel_grad_x_chunked(X_star, X_train, sigma_f, length_scale, chunk_size=self.chunk_size)
        # K_star_FE = -∇_{s} k(s,x), shape (M, N, d)
        K_star_FE = -grad_test
        
        # Compute derivative of k w.r.t training input: ∇_{x} k(x, s)
        # Using symmetry, ∇_{x} k(s, x) = -∇_{x} k(x, s)
        grad_train = rbf_kernel_grad_x_chunked(X_train, X_star, sigma_f, length_scale, chunk_size=self.chunk_size)
        # Transpose grad_train to shape (M, N, d)
        grad_train_T = np.transpose(grad_train, (1, 0, 2))
        # K_star_EF = -∂k(s,x)/∂x. But note: ∂k(s,x)/∂x = -∇_{x} k(x,s), so:
        K_star_EF = grad_train_T  # shape (M, N, d)
        
        # Compute Hessian cross covariance: ∂^2k(s, x)/(∂s ∂x), shape (M, N, d, d)
        K_star_FF = rbf_kernel_hessian_chunked(X_star, X_train, sigma_f, length_scale, chunk_size=self.chunk_size)
        
        # Reshape alpha into alpha_E and alpha_F
        alpha_E = self.alpha[:N]          # shape (N,)
        alpha_F = self.alpha[N:].reshape(N, d)  # shape (N, d)
        
        # Predicted energy: E_pred = K_star_EE * alpha_E + sum_{j,l} K_star_EF[:, j, l] * alpha_F[j,l]
        E_part1 = np.dot(K_star_EE, alpha_E)  # shape (M,)
        E_part2 = np.einsum('mjd,jd->m', K_star_EF, alpha_F)  # shape (M,)
        E_pred = E_part1 + E_part2
        
        # Predicted force: F_pred = sum_j [K_star_FE[:,j,:]*alpha_E[j]] + sum_j [K_star_FF[:,j,:,:] dot alpha_F[j]]
        F_part1 = np.einsum('mnd,n->md', K_star_FE, alpha_E)  # shape (M, d)
        F_part2 = np.einsum('mnij,nj->mi', K_star_FF, alpha_F)  # shape (M, d)
        F_pred = F_part1 + F_part2
        
        return E_pred, F_pred

    
class CaluculationGPNEB:
    def __init__(self, directory, APPLY_CI_NEB=99999):
        self.APPLY_CI_NEB = APPLY_CI_NEB
        self.base_dir = directory
        self.init_param = None
        self.spes_iter = 50
        self.chunk_size = 64
        return
    
    def calc_quickmin_step(self, positions, velocities, forces, mass=1.0, dt=0.01, alpha=0.0):
        dot_vf = np.sum(velocities * forces, axis=(1, 2), keepdims=True)
        mask = dot_vf < 0
        velocities[mask] = 0.0
        velocities = (1 - alpha) * velocities + (dt / mass) * forces
        positions = positions + dt * velocities
        return positions, velocities
    
    def judge_early_stopping(self, spes_energy_list, prev_spes_energy_list, spes_force_list, prev_spes_force_list):
        boolean_list = []
        nnode = len(spes_energy_list) 
        for i in range(nnode):
            pass
            ### implement early stopping conditions


        return boolean_list

    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        nnode = len(energy_list)
        prev_geometry_num_list = copy.copy(geometry_num_list)
        print("Start GPR fitting.")
        if optimize_num == 0:
            GPR = GaussianProcessRegressor()
            init_param = {"sigma_f": 1.0, "length_scale": 1.0, "noise_e": 1e-3, "noise_f": 1e-3, "chunk_size": self.chunk_size}
            train_geom_list = []
            train_force_list = []
            train_energy_list = []
            
            for i in range(nnode):
                train_geom_list.append(geometry_num_list[i].reshape(-1))
                train_energy_list.append(energy_list[i])
                train_force_list.append(gradient_list[i].reshape(-1))
            train_geom_list = np.array(train_geom_list, dtype = "float64")
            train_energy_list = np.array(train_energy_list, dtype = "float64")
            train_force_list = np.array(train_force_list, dtype = "float64")
            GPR.fit(train_geom_list, train_energy_list, train_force_list, init_param)
        
        else:
            GPR = GaussianProcessRegressor()
            train_geom_list = np.load(self.base_dir + "/train_geometry_num_list.npy")
            train_energy_list = np.load(self.base_dir + "/train_energy_list.npy")
            train_force_list = np.load(self.base_dir + "/train_force_list.npy")
            for i in range(nnode):
                np.vstack(train_geom_list, geometry_num_list[i].reshape(-1))
                np.hstack(train_energy_list, energy_list[i])
                np.vstack(train_force_list, gradient_list[i].reshape(-1))
                
            GPR.fit(train_geom_list, train_energy_list, train_force_list, self.init_param)
        
        print("GPR fitting was done.")
        print("Start optimization path on SPES (Surrogate Potential Energy Surface).")

        velocities_list = np.zeros_like(gradient_list)
        for l in range(self.spes_iter):
            print("SPES Opt ITR.:", l)
            input_pos = geometry_num_list
            ### reshape positions (3N, nnode) input_pos
            spes_energy_list, spes_force_list = GPR.predict_energy_and_forces(input_pos)
            ### reshape positions (nnode, N, 3) input_pos
            total_neb_spes_force_list = self.calc_force_for_gpr(input_pos, spes_energy_list, spes_force_list)
            ### reshape total_neb_spes_force_list (nnode, N, 3)
            for j in range(nnode):
                input_pos[j], velocities_list[j] = self.calc_quickmin_step(input_pos[j], velocities_list[j], total_neb_spes_force_list[j], mass=1.0, dt=0.01, alpha=0.0)
            if l > 0:
                ### implement conditions of early stopping. is_early_stopping_list = self.judge_early_stopping(spes_energy_list, prev_spes_energy_list, spes_force_list, prev_spes_force_list)
                pass
            else:
                pass

            for j in range(nnode):
                if is_early_stopping_list[j]:
                    print("Detect abnormal force and energy. Stop updating geometry.")
                    
                else:
                    geometry_num_list[j] = input_pos[j]
            prev_spes_energy_list = spes_energy_list
            prev_spes_force_list = spes_force_list

        print("Optimization on SPES was done.")
        np.save(self.base_dir + "/train_geometry_num_list.npy", train_geom_list)
        np.save(self.base_dir + "/train_energy_list.npy", train_energy_list)
        np.save(self.base_dir + "/train_force_list.npy", train_force_list)
        self.init_param = {"sigma_f": GPR.sigma_f, "length_scale": GPR.length_scale, "noise_e": GPR.noise_e, "noise_f": GPR.noise_f, "chunk_size": self.chunk_size}
        total_force_list = geometry_num_list - prev_geometry_num_list
        return np.array(total_force_list, dtype = "float64")
    
    def calc_force_for_gpr(self, geometry_num_list, energy_list, gradient_list):
        print("GPNEBGPNEBGPNEBGPNEBGPNEBGPNEBGPNEBGPNEBGPNEBGPNEBGPNEB")
        nnode = len(energy_list)
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        total_force_list = []
        for i in range(nnode):
            if i == 0:
                total_force_list.append(-1*np.array(gradient_list[0], dtype = "float64"))
                continue
            elif i == nnode-1:
                total_force_list.append(-1*np.array(gradient_list[nnode-1], dtype = "float64"))
                continue
            tmp_grad = copy.copy(gradient_list[i]).reshape(-1, 1)
            force, tangent_grad = self.calc_project_out_grad(geometry_num_list[i-1], geometry_num_list[i], geometry_num_list[i+1], tmp_grad, energy_list[i-1:i+2])     
            total_force_list.append(-1*force.reshape(-1, 3)) 
        return np.array(total_force_list, dtype = "float64")
    
    def calc_project_out_grad(self, coord_1, coord_2, coord_3, grad_2, energy_list):# grad: (3N, 1), geom_num_list: (N, 3)
        natom = len(coord_2)
        tmp_grad = copy.copy(grad_2)
        if energy_list[0] < energy_list[1] and energy_list[1] < energy_list[2]:
            B_mat = self.calc_B_matrix_for_NEB_tangent(coord_2, coord_3)
            int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)
            projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
            tangent_grad = projection_grad
        elif energy_list[0] > energy_list[1] and energy_list[1] > energy_list[2]:
            B_mat = self.calc_B_matrix_for_NEB_tangent(coord_1, coord_2)
            int_grad = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat)
            projection_grad = calc_cart_grad_from_pBmat(-1*int_grad, B_mat)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad
            tangent_grad = projection_grad
        else:
            B_mat_plus = self.calc_B_matrix_for_NEB_tangent(coord_2, coord_3)
            B_mat_minus = self.calc_B_matrix_for_NEB_tangent(coord_1, coord_2)
            int_grad_plus = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat_plus)
            int_grad_minus = calc_int_grad_from_pBmat(tmp_grad.reshape(3*natom, 1), B_mat_minus)
            max_ene = max(abs(energy_list[2] - energy_list[1]), abs(energy_list[1] - energy_list[0]))
            min_ene = min(abs(energy_list[2] - energy_list[1]), abs(energy_list[1] - energy_list[0]))
            a = (max_ene + 1e-15) / (max_ene + min_ene + 1e-15)
            b = (min_ene + 1e-15) / (max_ene + min_ene + 1e-15)
            
            if energy_list[0] < energy_list[2]:
                projection_grad_plus = calc_cart_grad_from_pBmat(-a*int_grad_plus, B_mat_plus)
                projection_grad_minus = calc_cart_grad_from_pBmat(-b*int_grad_minus, B_mat_minus)
            
            else:
                projection_grad_plus = calc_cart_grad_from_pBmat(-b*int_grad_plus, B_mat_plus)
                projection_grad_minus = calc_cart_grad_from_pBmat(-a*int_grad_minus, B_mat_minus)
            proj_grad = tmp_grad.reshape(3*natom, 1) + projection_grad_plus + projection_grad_minus
            tangent_grad = projection_grad_plus + projection_grad_minus
        return proj_grad, tangent_grad

    
    def calc_B_matrix_for_NEB_tangent(self, coord_1, coord_2):
        natom = len(coord_2)
        B_mat = np.zeros((natom, 3*natom))
        
        for i in range(natom):
            norm_12 = np.linalg.norm(coord_1[i] - coord_2[i]) + 1e-15
            dr12_dx2 = (coord_2[i][0] - coord_1[i][0]) / norm_12
            dr12_dy2 = (coord_2[i][1] - coord_1[i][1]) / norm_12
            dr12_dz2 = (coord_2[i][2] - coord_1[i][2]) / norm_12
            B_mat[i][3*i] = dr12_dx2
            B_mat[i][3*i+1] = dr12_dy2
            B_mat[i][3*i+2] = dr12_dz2

        return B_mat
    
    

