import numpy as np

from scipy import linalg
from parameter import UnitValueLib

from .hessian_update import ModelHessianUpdate
"""
RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
 Theor chim Acta (1992) 82: 189-205
"""

class RationalFunctionOptimization:
    def __init__(self, **config):
        
        self.config = config
        self.hess_update = ModelHessianUpdate()
        self.Initialization = True
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = self.config["saddle_order"] #
        
        self.trust_radius = 0.1
        if self.config["trust_radius"] is None:
            pass
        else:
            self.trust_radius = float(self.config["trust_radius"])
        self.trust_radius /= UnitValueLib().bohr2angstroms
        self.iter = 0 #
        self.max_rfo_iter = 2000 #
        self.rfo_threshold = 1e-10 #
        self.beta = 0.10
        self.grad_rms_threshold = 0.01
        self.tight_grad_rms_threshold = 0.0002 
        self.combine_eigen_vec_num = 3
        self.combine_eigvec_flag = False
        self.lambda_s_scale = 1.0
        self.lambda_clip = 1000.0
        self.lambda_clip_flag = False
        self.projection_eigenvector_flag = False
        
        # GDIIS parameters with enhanced defaults
        self.gdiis_history_size = 4        # Reduced history size for better stability
        self.gdiis_min_points = 3          # Require more points before starting GDIIS
        self.gdiis_error_threshold = 0.5   # More conservative error threshold
        self.gdiis_weight_initial = 0.3    # Start with lower GDIIS contribution
        self.gdiis_weight_max = 0.7        # Maximum GDIIS weight
        
        # Robust coefficient handling
        self.gdiis_coeff_min = -0.2        # Stricter minimum coefficient value
        self.gdiis_coeff_max = 1.3         # Stricter maximum coefficient value
        self.gdiis_regularization = 1e-8   # Increased regularization parameter
        
        # Enhanced error recovery
        self.gdiis_failure_count = 0       # Counter for consecutive GDIIS failures
        self.gdiis_max_failures = 2        # Reset history after fewer failures
        self.gdiis_recovery_steps = 3      # Number of steps in recovery mode
        self.gdiis_current_recovery = 0    # Current recovery step counter
        
        # Aggressive outlier detection
        self.gdiis_step_ratio_max = 2.0    # Maximum allowed ratio between GDIIS and RFO steps
        self.gdiis_outlier_threshold = 3.0 # Standard deviations for outlier detection
        
        # Dynamic weight adjustment
        self.gdiis_weight_current = self.gdiis_weight_initial  # Current weight
        self.gdiis_weight_increment = 0.05  # Increment for successful iterations
        self.gdiis_weight_decrement = 0.15  # Larger decrement for failures
        
        # GDIIS history storage with quality metrics
        self.geom_history = []
        self.grad_history = []
        self.quality_history = []          # Track quality of each point
        
        # Convergence monitoring
        self.prev_grad_rms = float('inf')
        self.non_improving_count = 0
        
        self.hessian = None
        self.bias_hessian = None
   
        return
    
    def calc_center(self, geomerty, element_list=[]):#geomerty:Bohr
        center = np.array([0.0, 0.0, 0.0], dtype="float64")
        for i in range(len(geomerty)):
            
            center += geomerty[i] 
        center /= float(len(geomerty))
        
        return center
                
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def reset_hessian(self, geometry):
        self.hessian = np.eye((len(geometry)))
        return

    
    def get_hessian(self):
        return self.hessian
    
    def get_bias_hessian(self):
        return self.bias_hessian
    
    def hessian_update(self, displacement, delta_grad):
        if "msp" in self.config["method"].lower():
            print("RFO_MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(self.hessian, displacement, delta_grad)
        elif "bfgs" in self.config["method"].lower():
            print("RFO_BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(self.hessian, displacement, delta_grad)
        elif "fsb" in self.config["method"].lower():
            print("RFO_FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "bofill" in self.config["method"].lower():
            print("RFO_Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(self.hessian, displacement, delta_grad)
        elif "sr1" in self.config["method"].lower():
            print("RFO_SR1_quasi_newton_method")
            delta_hess = self.hess_update.SR1_hessian_update(self.hessian, displacement, delta_grad)
        elif "psb" in self.config["method"].lower():
            print("RFO_PSB_quasi_newton_method")
            delta_hess = self.hess_update.PSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "flowchart" in self.config["method"].lower():
            print("RFO_flowchart_quasi_newton_method")
            delta_hess = self.hess_update.flowchart_hessian_update(self.hessian, displacement, delta_grad, self.config["method"])
        else:
            raise "method error"
        return delta_hess
    
    def get_cleaned_hessian(self, hessian):

        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        try:
            # Use more stable eigenvalue decomposition
            eigval, eigvec = linalg.eigh(hessian)
        except np.linalg.LinAlgError:
            # Fallback to more robust algorithm if standard fails
            print("Warning: Using more robust eigenvalue decomposition")
            eigval, eigvec = linalg.eigh(hessian, driver='evr')
        
        # Find valid eigenvalues (|λ| > 1e-7)
        valid_mask = np.abs(eigval) > 1e-7
        n_removed = np.sum(~valid_mask)
        
        # Create diagonal matrix with only valid eigenvalues
        # Replace small eigenvalues with zeros
        cleaned_eigval = np.where(valid_mask, eigval, 1e-7)
        
        # Reconstruct Hessian using only valid components
        # H = U Λ U^T where Λ contains only valid eigenvalues
        cleaned_hessian = np.dot(np.dot(eigvec, np.diag(cleaned_eigval)), eigvec.T)
        
        # Ensure symmetry of final result
        cleaned_hessian = 0.5 * (cleaned_hessian + cleaned_hessian.T)
        
        # Additional numerical stability check
        if not np.allclose(cleaned_hessian, cleaned_hessian.T, rtol=1e-13, atol=1e-13):
            print("Warning: Symmetry check failed, forcing exact symmetry")
            cleaned_hessian = 0.5 * (cleaned_hessian + cleaned_hessian.T)
        
        return cleaned_hessian, n_removed

    def project_out_hess_tr_and_rot_for_coord(self, hessian, geometry, display_eigval=True):#do not consider atomic mass
        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v.copy()
                for b in basis:
                    w -= np.dot(v, b) * b
                norm = np.linalg.norm(w)
                if norm > 1e-10:
                    basis.append(w / norm)
            return np.array(basis)
        
        natoms = len(geometry) // 3
        # Center the geometry
        geometry = geometry - self.calc_center(geometry, element_list=[])
        
        # Initialize arrays for translation and rotation vectors
        tr_vectors = np.zeros((3, 3 * natoms))
        rot_vectors = np.zeros((3, 3 * natoms))
        
        # Create translation vectors (mass-weighted normalization is not used as specified)
        for i in range(3):
            tr_vectors[i, i::3] = 1.0
        
        # Create rotation vectors
        for atom in range(natoms):
            # Get atom coordinates
            x, y, z = geometry[atom]
            
            # Rotation around x-axis: (0, -z, y)
            rot_vectors[0, 3*atom:3*atom+3] = np.array([0.0, -z, y])
            
            # Rotation around y-axis: (z, 0, -x)
            rot_vectors[1, 3*atom:3*atom+3] = np.array([z, 0.0, -x])
            
            # Rotation around z-axis: (-y, x, 0)
            rot_vectors[2, 3*atom:3*atom+3] = np.array([-y, x, 0.0])

        # Combine translation and rotation vectors
        TR_vectors = np.vstack([tr_vectors, rot_vectors])
        

        
        # Orthonormalize the translation and rotation vectors
        TR_vectors = gram_schmidt(TR_vectors)
        
        # Calculate projection matrix
        P = np.eye(3 * natoms)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)
        P = 0.5 * (P + P.T)
        # Project the Hessian
        hess_proj = np.dot(np.dot(P.T, hessian), P)
        
        # Make the projected Hessian symmetric (numerical stability)
        hess_proj = (hess_proj + hess_proj.T) / 2
        
        if display_eigval:
            eigenvalues, _ = np.linalg.eigh(hess_proj)
            eigenvalues = np.sort(eigenvalues)
            # Filter out near-zero eigenvalues
            idx_eigenvalues = np.where(np.abs(eigenvalues) > 1e-6)[0]
            print(f"EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: {len(idx_eigenvalues)}):")
            for i in range(0, len(idx_eigenvalues), 6):
                tmp_arr = eigenvalues[idx_eigenvalues[i:i+6]]
                print(" ".join(f"{val:12.8f}" for val in tmp_arr))
         
        return hess_proj    
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv1")
        if self.Initialization:
            self.lambda_s_scale = 0.1
            self.Initialization = False
            return self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        DELTA_for_QNM = self.DELTA
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            delta_hess = self.hessian_update(displacement, delta_grad)
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        new_hess = 0.5 * (new_hess + new_hess.T)
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, _ = np.linalg.eigh(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        
        print("lambda   : ",lambda_for_calc)
       
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - self.lambda_s_scale*lambda_for_calc*(np.eye(len(geom_num_list))), B_g.reshape(len(geom_num_list), 1))
        
        if np.linalg.norm(move_vector) < 1e-10:
            print("Warning: The step size is too small!!!")
           
        else:
            self.hessian += delta_hess 
        self.iter += 1
            
        return move_vector#Bohr.
    
    def normal_v2(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv2")    
        # This implementation is almost same as RFOv3 but lambda is not considered non-linear effect. 
        #ref.:Culot, P., Dive, G., Nguyen, V.H. et al. A quasi-Newton algorithm for first-order saddle-point location. Theoret. Chim. Acta 82, 189–205 (1992). https://doi.org/10.1007/BF01113492
        n_coords = len(geom_num_list)
        
        if self.Initialization:
            self.Initialization = False
            new_hess = self.hessian + self.bias_hessian
        else:
            # Calculate geometry and gradient differences with improved precision
            delta_grad = (g - pre_g).reshape(n_coords, 1)
            displacement = (geom_num_list - pre_geom).reshape(n_coords, 1)
        
            # Update Hessian if needed
            if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                try:
                    delta_hess = self.hessian_update(displacement, delta_grad)
                    new_hess = self.hessian + delta_hess + self.bias_hessian
                except np.linalg.LinAlgError:
                    print("Warning: Hessian update failed, using previous Hessian")
                    new_hess = self.hessian + self.bias_hessian
            else:
                new_hess = self.hessian + self.bias_hessian
        
        
        print("saddle order:", self.saddle_order)
        
        # Ensure symmetry and remove small eigenvalues
        new_hess = 0.5 * (new_hess + new_hess.T)
        #new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        # Construct RFO matrix with improved stability
        grad_norm = np.linalg.norm(B_g)
        
        scaled_grad = B_g
        
        matrix_for_RFO = np.block([
            [new_hess, scaled_grad.reshape(n_coords, 1)],
            [scaled_grad.reshape(1, n_coords), np.zeros((1, 1))]
        ])
        
        # Compute RFO eigenvalues using more stable algorithm
        try:
            RFO_eigenvalue, _ = linalg.eigh(matrix_for_RFO)
        except np.linalg.LinAlgError:
            print("Warning: Using more robust eigenvalue algorithm")
            RFO_eigenvalue, _ = linalg.eigh(matrix_for_RFO, driver='evr')
        
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        
        # Compute Hessian eigensystem with improved stability
        try:
            hess_eigenvalue, hess_eigenvector = linalg.eigh(new_hess)
        except np.linalg.LinAlgError:
            print("Warning: Using more robust eigenvalue algorithm for Hessian")
            hess_eigenvalue, hess_eigenvector = linalg.eigh(new_hess, driver='evr')
        
        hess_eigenvector = hess_eigenvector.T
        hess_eigenval_indices = np.argsort(hess_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[max(self.saddle_order, 0)])
        # Initialize move vector
        move_vector = np.zeros((n_coords, 1))
        saddle_order_count = 0
        
        
        
        # Constants for numerical stability
        EIGENVAL_THRESHOLD = 1e-7
        DENOM_THRESHOLD = 1e-10
        
        # Calculate move vector with improved stability
        for i in range(len(hess_eigenvalue)):
            idx = hess_eigenval_indices[i]
            
            # Skip processing if eigenvalue is too small
            if np.abs(hess_eigenvalue[idx]) < EIGENVAL_THRESHOLD:
                continue
                
            tmp_vector = hess_eigenvector[idx].reshape(1, n_coords)
            proj_magnitude = np.dot(tmp_vector, B_g.reshape(n_coords, 1))
            
            if saddle_order_count < self.saddle_order:
                if self.projection_eigenvector_flag:
                    continue
                    
                step_scaling = 1.0
                tmp_eigval = hess_eigenvalue[idx]
                denom = tmp_eigval + lambda_for_calc
                
                # Stabilize denominator
                if np.abs(denom) > DENOM_THRESHOLD:
                    contribution = (step_scaling * self.DELTA * proj_magnitude * tmp_vector.T) / denom
                    move_vector += contribution
                    saddle_order_count += 1
                
            else:
                step_scaling = 1.0
                
                # Handle combination of eigenvectors for unwanted saddle points
                if (self.grad_rms_threshold > np.sqrt(np.mean(B_g ** 2)) and 
                    hess_eigenvalue[idx] < -1e-9 and 
                    self.combine_eigvec_flag):
                        
                    print(f"Combining {self.combine_eigen_vec_num} eigenvectors to avoid unwanted saddle point...")
                    combined_vector = tmp_vector.copy()
                    count = 1
                    
                    for j in range(1, self.combine_eigen_vec_num):
                        next_idx = idx + j
                        if next_idx >= len(hess_eigenvalue):
                            break
                        combined_vector += hess_eigenvector[hess_eigenval_indices[next_idx]].reshape(1, n_coords)
                        count += 1
                    
                    tmp_vector = combined_vector / count
                
                denom = hess_eigenvalue[idx] - lambda_for_calc
                
                # Stabilize denominator
                if np.abs(denom) > DENOM_THRESHOLD:
                    contribution = (step_scaling * self.DELTA * proj_magnitude * tmp_vector.T) / denom
                    move_vector += contribution
        
        print("lambda   : ", lambda_for_calc)
        
        # Check step size and update
        step_norm = np.linalg.norm(move_vector)
        if step_norm < 1e-10:
            print("Warning: The step size is too small!")
        elif self.iter == 0:
            pass
        else:
            self.hessian += delta_hess  
        
        self.iter += 1        
        return move_vector  # in Bohr        

    def normal_v3(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFOv3")
        #ref.:Culot, P., Dive, G., Nguyen, V.H. et al. A quasi-Newton algorithm for first-order saddle-point location. Theoret. Chim. Acta 82, 189–205 (1992). https://doi.org/10.1007/BF01113492
        n_coords = len(geom_num_list)
        
        if self.Initialization:
            self.Initialization = False
            new_hess = self.hessian + self.bias_hessian
        else:
            # Calculate geometry and gradient differences with improved precision
            delta_grad = (g - pre_g).reshape(n_coords, 1)
            displacement = (geom_num_list - pre_geom).reshape(n_coords, 1)
        
            # Update Hessian if needed
            if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                try:
                    delta_hess = self.hessian_update(displacement, delta_grad)
                    new_hess = self.hessian + delta_hess + self.bias_hessian
                except np.linalg.LinAlgError:
                    print("Warning: Hessian update failed, using previous Hessian")
                    new_hess = self.hessian + self.bias_hessian
            else:
                new_hess = self.hessian + self.bias_hessian
        
        
        print("saddle order:", self.saddle_order)
        
        # Ensure symmetry and remove small eigenvalues
        new_hess = 0.5 * (new_hess + new_hess.T)
        #new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        # Construct RFO matrix with improved stability
        grad_norm = np.linalg.norm(B_g)
        
        scaled_grad = B_g
        
        matrix_for_RFO = np.block([
            [new_hess, scaled_grad.reshape(n_coords, 1)],
            [scaled_grad.reshape(1, n_coords), np.zeros((1, 1))]
        ])
        
        # Compute RFO eigenvalues using more stable algorithm
        try:
            RFO_eigenvalue, _ = linalg.eigh(matrix_for_RFO)
        except np.linalg.LinAlgError:
            print("Warning: Using more robust eigenvalue algorithm")
            RFO_eigenvalue, _ = linalg.eigh(matrix_for_RFO, driver='evr')
        
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        
        # Compute Hessian eigensystem with improved stability
        try:
            hess_eigenvalue, hess_eigenvector = linalg.eigh(new_hess)
        except np.linalg.LinAlgError:
            print("Warning: Using more robust eigenvalue algorithm for Hessian")
            hess_eigenvalue, hess_eigenvector = linalg.eigh(new_hess, driver='evr')
        
        hess_eigenvector = hess_eigenvector.T
        hess_eigenval_indices = np.argsort(hess_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[max(self.saddle_order, 0)])
        # Initialize move vector
        move_vector = np.zeros((n_coords, 1))
        saddle_order_count = 0
        
        lambda_for_calc = RFOSecularSolverIterative().calc_rfo_lambda_and_step(hess_eigenvector, hess_eigenvalue, lambda_for_calc, B_g, self.saddle_order) # consider non-linear effect
        
        # Constants for numerical stability
        EIGENVAL_THRESHOLD = 1e-7
        DENOM_THRESHOLD = 1e-10
        
        # Calculate move vector with improved stability
        for i in range(len(hess_eigenvalue)):
            idx = hess_eigenval_indices[i]
            
            # Skip processing if eigenvalue is too small
            if np.abs(hess_eigenvalue[idx]) < EIGENVAL_THRESHOLD:
                continue
                
            tmp_vector = hess_eigenvector[idx].reshape(1, n_coords)
            proj_magnitude = np.dot(tmp_vector, B_g.reshape(n_coords, 1))
            
            if saddle_order_count < self.saddle_order:
                if self.projection_eigenvector_flag:
                    continue
                    
                step_scaling = 1.0
                tmp_eigval = hess_eigenvalue[idx]
                denom = tmp_eigval + lambda_for_calc
                
                # Stabilize denominator
                if np.abs(denom) > DENOM_THRESHOLD:
                    contribution = (step_scaling * self.DELTA * proj_magnitude * tmp_vector.T) / denom
                    move_vector += contribution
                    saddle_order_count += 1
                
            else:
                step_scaling = 1.0
                
                # Handle combination of eigenvectors for unwanted saddle points
                if (self.grad_rms_threshold > np.sqrt(np.mean(B_g ** 2)) and 
                    hess_eigenvalue[idx] < -1e-9 and 
                    self.combine_eigvec_flag):
                        
                    print(f"Combining {self.combine_eigen_vec_num} eigenvectors to avoid unwanted saddle point...")
                    combined_vector = tmp_vector.copy()
                    count = 1
                    
                    for j in range(1, self.combine_eigen_vec_num):
                        next_idx = idx + j
                        if next_idx >= len(hess_eigenvalue):
                            break
                        combined_vector += hess_eigenvector[hess_eigenval_indices[next_idx]].reshape(1, n_coords)
                        count += 1
                    
                    tmp_vector = combined_vector / count
                
                denom = hess_eigenvalue[idx] - lambda_for_calc
                
                # Stabilize denominator
                if np.abs(denom) > DENOM_THRESHOLD:
                    contribution = (step_scaling * self.DELTA * proj_magnitude * tmp_vector.T) / denom
                    move_vector += contribution
        
        print("lambda   : ", lambda_for_calc)
        
        # Check step size and update
        step_norm = np.linalg.norm(move_vector)
        if step_norm < 1e-10:
            print("Warning: The step size is too small!")
        elif self.iter == 0:
            pass
        else:
            self.hessian += delta_hess  
        
        self.iter += 1
        return move_vector  # in Bohr
        
    def neb(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("RFO4neb")
        #ref.:Culot, P., Dive, G., Nguyen, V.H. et al. A quasi-Newton algorithm for first-order saddle-point location. Theoret. Chim. Acta 82, 189–205 (1992). https://doi.org/10.1007/BF01113492
        if self.Initialization:
            self.Initialization = False
            return self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        DELTA_for_QNM = self.DELTA
        

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            delta_hess = self.hessian_update(displacement, delta_grad)
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        new_hess = 0.5 * (new_hess + new_hess.T)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eigh(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[max(self.saddle_order-1, 0)])

        if self.lambda_clip_flag:
            lambda_for_calc = np.clip(lambda_for_calc, -self.lambda_clip, self.lambda_clip)

        hess_eigenvalue, hess_eigenvector = np.linalg.eigh(new_hess)
        hess_eigenvector = hess_eigenvector.T
        hess_eigenval_indices = np.argsort(hess_eigenvalue)
        
        move_vector = np.zeros((len(geom_num_list), 1))
        DELTA_for_QNM = self.DELTA
        
        for i in range(len(hess_eigenvalue)):
            tmp_vector = np.array([hess_eigenvector[hess_eigenval_indices[i]].T], dtype="float64")
            if i < self.saddle_order:
                if self.projection_eigenvector_flag:
                    continue
                step_scaling = 1.0
                tmp_eigval = np.clip(hess_eigenvalue[hess_eigenval_indices[i]], -10.0, 10.0)
                move_vector += step_scaling * DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list), 1)) * tmp_vector.T / (tmp_eigval + lambda_for_calc + 1e-12) 
            else:
                step_scaling = 1.0
                if self.grad_rms_threshold > np.sqrt(np.mean(B_g ** 2)) and hess_eigenvalue[i] < -1e-9 and self.combine_eigvec_flag:
                    print(f"To locate geometry away from unwanted saddle point, combine {self.combine_eigen_vec_num} other eigenvectors ...")# Gaussian 16 Rev.C 01
                    for j in range(self.saddle_order + 1, min(self.saddle_order + 1 + self.combine_eigen_vec_num, len(hess_eigenvalue))):
                       
                        tmp_vector += np.array([hess_eigenvector[hess_eigenval_indices[j+i]].T], dtype="float64")
                    tmp_vector /= self.combine_eigen_vec_num * (1 / (step_scaling))
                

                tmp_eigval = np.clip(hess_eigenvalue[hess_eigenval_indices[i]], -10.0, 10.0)
                move_vector += step_scaling * DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list), 1)) * tmp_vector.T / (tmp_eigval - lambda_for_calc + 1e-12)
        print("lambda   : ",lambda_for_calc)
         
        
        if np.linalg.norm(move_vector) < 1e-10:
            print("Warning: The step size is too small!!!")
        else:
            self.hessian += delta_hess 
        self.iter += 1
     
        return move_vector#Bohr.

  
    def moment(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("moment mode")
        print("saddle order:", self.saddle_order)

        if self.Initialization:
            self.momentum_disp = geom_num_list * 0.0
            self.momentum_grad = geom_num_list * 0.0
            self.Initialization = False
            return self.DELTA*B_g
            
        
        if self.iter == 1:
            self.momentum_disp = geom_num_list - pre_geom
            self.momentum_grad = B_g - pre_B_g

        
        delta_grad = (B_g - pre_B_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        
        new_momentum_disp = self.beta * self.momentum_disp + (1.0 - self.beta) * displacement
        new_momentum_grad = self.beta * self.momentum_grad + (1.0 - self.beta) * delta_grad
        
        delta_momentum_disp = (new_momentum_disp - self.momentum_disp).reshape(len(geom_num_list), 1)
        delta_momentum_grad = (new_momentum_grad - self.momentum_grad).reshape(len(geom_num_list), 1)
        
    
        
        delta_hess = self.hessian_update(delta_momentum_disp, delta_momentum_grad)
        

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        new_hess = 0.5 * (new_hess + new_hess.T)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        
        DELTA_for_QNM = self.DELTA

        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list), 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eigh(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        

        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)))), B_g.reshape(len(geom_num_list), 1)))
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - self.lambda_s_scale*lambda_for_calc*(np.eye(len(geom_num_list))), B_g.reshape(len(geom_num_list), 1))
    
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM,"\n")
        self.hessian += delta_hess
        self.momentum_disp = new_momentum_disp
        self.momentum_grad = new_momentum_grad
        self.iter += 1
        return move_vector#Bohr.   

    def _update_gdiis_history(self, geometry, gradient, step_quality=1.0):
        """
        Update the GDIIS history with quality-based filtering
        
        Parameters:
        -----------
        geometry : numpy.ndarray
            Current geometry
        gradient : numpy.ndarray
            Current gradient
        step_quality : float
            Quality metric for this point (1.0 = good, <1.0 = lower quality)
        """
        # Add current point to history with quality metric
        self.geom_history.append(geometry.copy())
        self.grad_history.append(gradient.copy())
        self.quality_history.append(step_quality)
        
        # If in recovery mode, only keep the most recent points
        if self.gdiis_current_recovery > 0:
            self.gdiis_current_recovery -= 1
            if len(self.geom_history) > 2:
                self.geom_history = self.geom_history[-2:]
                self.grad_history = self.grad_history[-2:]
                self.quality_history = self.quality_history[-2:]
            return
        
        # Limit history size
        if len(self.geom_history) > self.gdiis_history_size:
            # Remove lowest quality point (except for the newest point)
            if len(self.geom_history) > 2:
                # Don't consider the most recent point for removal
                oldest_qualities = self.quality_history[:-1]
                worst_idx = np.argmin(oldest_qualities)
                
                # Remove the lowest quality point
                self.geom_history.pop(worst_idx)
                self.grad_history.pop(worst_idx)
                self.quality_history.pop(worst_idx)
            else:
                # Default to removing oldest point if we only have 2 points
                self.geom_history.pop(0)
                self.grad_history.pop(0)
                self.quality_history.pop(0)

    def _condition_b_matrix(self, B, n_points):
        """
        Apply advanced conditioning to improve B matrix stability
        
        Parameters:
        -----------
        B : numpy.ndarray
            The B matrix to condition
        n_points : int
            Number of actual data points
            
        Returns:
        --------
        numpy.ndarray
            Conditioned B matrix
        """
        # 1. Add regularization to diagonal for numerical stability
        np.fill_diagonal(B[:n_points, :n_points], 
                        np.diag(B[:n_points, :n_points]) + self.gdiis_regularization)
        
        # 2. Apply weighted regularization based on point quality
        if hasattr(self, 'quality_history') and len(self.quality_history) == n_points:
            for i in range(n_points):
                # Lower quality points get more regularization
                quality_factor = self.quality_history[i]
                B[i, i] += self.gdiis_regularization * (2.0 - quality_factor) / quality_factor
        
        # 3. Improve conditioning with SVD-based truncation
        try:
            # Apply SVD to the main block
            u, s, vh = np.linalg.svd(B[:n_points, :n_points])
            
            # Truncate small singular values (improves condition number)
            s_max = np.max(s)
            s_cutoff = s_max * 1e-10
            s_fixed = np.array([max(sv, s_cutoff) for sv in s])
            
            # Reconstruct with improved conditioning
            B_improved = np.dot(u * s_fixed, vh)
            
            # Put the improved block back
            B[:n_points, :n_points] = B_improved
        except:
            # If SVD fails, use simpler Tikhonov regularization
            identity = np.eye(n_points)
            B[:n_points, :n_points] += 1e-7 * identity
        
        return B

    def _solve_gdiis_equations(self, error_vectors, qualities=None):
        """
        Solve GDIIS equations with multiple robustness techniques
        """
        n_points = len(error_vectors)
        
        # Handle case of too few points
        if n_points < 2:
            return np.array([1.0])
        
        # Use quality weighting if available
        if qualities is None:
            qualities = np.ones(n_points)
        
        # Construct the B matrix with dot products of error vectors
        B = np.zeros((n_points + 1, n_points + 1))
        
        # Fill B matrix with weighted error vector dot products
        for i in range(n_points):
            for j in range(n_points):
                # Weight error dot products by quality
                weight_factor = np.sqrt(qualities[i] * qualities[j])
                B[i, j] = weight_factor * np.dot(error_vectors[i].T, error_vectors[j])
        
        # Apply advanced conditioning to the B matrix
        B = self._condition_b_matrix(B, n_points)
        
        # Add Lagrange multiplier constraints
        B[n_points, :n_points] = 1.0
        B[:n_points, n_points] = 1.0
        B[n_points, n_points] = 0.0
        
        # Right-hand side vector with constraint
        rhs = np.zeros(n_points + 1)
        rhs[n_points] = 1.0
        
        # Multi-stage solver with progressive fallbacks
        methods = [
            ("Standard solve", lambda: np.linalg.solve(B, rhs)),
            ("SVD solve", lambda: self._svd_solve(B, rhs, 1e-12)),
            ("Regularized solve", lambda: np.linalg.solve(B + np.diag([1e-6]*(n_points+1)), rhs)),
            ("Least squares", lambda: np.linalg.lstsq(B, rhs, rcond=1e-8)[0]),
            ("Minimal solution", lambda: self._minimal_solution(n_points))
        ]
        
        coefficients = None
        for method_name, solver in methods:
            try:
                coefficients = solver()
                # Check if solution is reasonable
                if not np.any(np.isnan(coefficients)) and np.abs(np.sum(coefficients[:n_points]) - 1.0) < 0.01:
                    print(f"GDIIS using {method_name}")
                    break
            except Exception as e:
                print(f"{method_name} failed: {str(e)}")
        
        # If all methods failed, default to using the most recent point
        if coefficients is None or np.any(np.isnan(coefficients)):
            print("All GDIIS solvers failed, using last point only")
            coefficients = np.zeros(n_points + 1)
            coefficients[n_points-1] = 1.0  # Use the most recent point
            coefficients[n_points] = 0.0    # Zero Lagrange multiplier
        
        # Extract actual coefficients (without Lagrange multiplier)
        return coefficients[:n_points]

    def _svd_solve(self, A, b, rcond=1e-15):
        """
        Solve linear system using SVD with improved handling of small singular values
        """
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        
        # More sophisticated singular value filtering
        s_max = np.max(s)
        mask = s > rcond * s_max
        
        # Create pseudo-inverse with smooth cutoff for small singular values
        s_inv = np.zeros_like(s)
        for i, (val, use) in enumerate(zip(s, mask)):
            if use:
                s_inv[i] = 1.0/val
            else:
                # Smooth transition to zero for small values
                ratio = val/(rcond * s_max)
                s_inv[i] = ratio/(val * (1.0 + (1.0 - ratio)**2))
        
        # Calculate solution using pseudo-inverse
        return np.dot(np.dot(np.dot(vh.T, np.diag(s_inv)), u.T), b)

    def _minimal_solution(self, n_points):
        """
        Fallback solution when all numerical methods fail
        """
        # Create a solution that gives higher weight to more recent points
        result = np.zeros(n_points + 1)
        
        # Linear ramp with highest weight to most recent point
        total_weight = 0
        for i in range(n_points):
            # Linear weighting: i+1 gives more weight to later points
            result[i] = i + 1
            total_weight += result[i]
        
        # Normalize to sum=1 and add zero Lagrange multiplier
        result[:n_points] /= total_weight
        return result

    def _filter_gdiis_coefficients(self, coeffs, strict=False):
        """
        Advanced filtering of extreme coefficient values
        
        Parameters:
        -----------
        coeffs : numpy.ndarray
            DIIS coefficients
        strict : bool
            Whether to use stricter filtering limits
        
        Returns:
        --------
        tuple
            (filtered_coeffs, was_filtered, quality_metric)
        """
        # Adjust bounds based on strictness
        coeff_min = self.gdiis_coeff_min * (1.5 if strict else 1.0)
        coeff_max = self.gdiis_coeff_max * (0.9 if strict else 1.0)
        
        # Check for extreme values
        extreme_values = np.logical_or(coeffs < coeff_min, coeffs > coeff_max)
        has_extreme_values = np.any(extreme_values)
        
        # Calculate quality metric (1.0 = perfect, lower values indicate problems)
        quality = 1.0
        if has_extreme_values:
            # Reduce quality based on how extreme the coefficients are
            extreme_ratio = np.sum(np.abs(coeffs[extreme_values])) / np.sum(np.abs(coeffs))
            quality = max(0.1, 1.0 - extreme_ratio)
            
            print(f"Warning: Extreme GDIIS coefficients detected: {[f'{c:.3f}' for c in coeffs]}")
            
            # Apply multi-stage filtering
            
            # 1. First attempt: Simple clipping and renormalization
            clipped_coeffs = np.clip(coeffs, coeff_min, coeff_max)
            sum_clipped = np.sum(clipped_coeffs)
            
            if abs(sum_clipped - 1.0) > 1e-10 and sum_clipped > 1e-10:
                normalized_coeffs = clipped_coeffs / sum_clipped
            else:
                # 2. If simple clipping failed, try redistribution approach
                print("Warning: Simple coefficient normalization failed, using redistribution")
                
                # Start with minimum values
                adjusted_coeffs = np.full_like(coeffs, coeff_min)
                
                # Distribute available weight (1.0 - sum(mins)) proportionally to valid coefficients
                valid_indices = ~extreme_values
                if np.any(valid_indices):
                    # Use only valid coefficients for distribution
                    valid_sum = np.sum(coeffs[valid_indices])
                    if abs(valid_sum) > 1e-10:
                        remaining = 1.0 - len(coeffs) * coeff_min
                        adjusted_coeffs[valid_indices] += remaining * (coeffs[valid_indices] / valid_sum)
                    else:
                        # If all valid coefficients sum to near zero, use uniform distribution
                        adjusted_coeffs = np.ones_like(coeffs) / len(coeffs)
                else:
                    # If all coefficients are extreme, use exponentially weighted recent points
                    n = len(coeffs)
                    for i in range(n):
                        adjusted_coeffs[i] = 0.5**min(n-i-1, 3)  # Exponentially weighted recent points
                    adjusted_coeffs /= np.sum(adjusted_coeffs)
                    
                normalized_coeffs = adjusted_coeffs
            
            # 3. Check if coefficients still have issues
            if np.any(np.isnan(normalized_coeffs)) or abs(np.sum(normalized_coeffs) - 1.0) > 1e-8:
                # Final fallback: use most recent point with small contributions from others
                print("Warning: Advanced filtering failed, falling back to recent-point dominated solution")
                n = len(coeffs)
                last_dominated = np.zeros_like(coeffs)
                last_dominated[-1] = 0.7  # 70% weight to most recent point
                
                # Distribute remaining 30% to other points
                remaining_weight = 0.3
                if n > 1:
                    for i in range(n-1):
                        last_dominated[i] = remaining_weight / (n-1)
                
                normalized_coeffs = last_dominated
            
            self.gdiis_failure_count += 1
            return normalized_coeffs, True, quality
        else:
            # Calculate quality based on coefficient distribution
            # Prefer solutions where coefficients are more evenly distributed
            n = len(coeffs)
            if n > 1:
                # Shannon entropy as a measure of coefficient distribution
                entropy = 0
                for c in coeffs:
                    if c > 0:
                        entropy -= c * np.log(c)
                
                # Normalize to [0,1] range
                max_entropy = np.log(n)
                if max_entropy > 0:
                    distribution_quality = min(1.0, entropy / max_entropy)
                    quality = 0.5 + 0.5 * distribution_quality
            
            self.gdiis_failure_count = max(0, self.gdiis_failure_count - 1)  # Reduce failure count on success
            return coeffs, False, quality

    def _calculate_gdiis_geometry(self):
        """
        Calculate a new geometry using GDIIS with comprehensive robustness measures
        """
        n_points = len(self.geom_history)
        
        if n_points < self.gdiis_min_points:
            return None, None, False, 0.0
        
        # Reset history if we've had too many failures
        if self.gdiis_failure_count >= self.gdiis_max_failures:
            print(f"Warning: {self.gdiis_failure_count} consecutive GDIIS failures, resetting history")
            # Keep only the most recent point
            if len(self.geom_history) > 0:
                self.geom_history = [self.geom_history[-1]]
                self.grad_history = [self.grad_history[-1]]
                self.quality_history = [1.0] if hasattr(self, 'quality_history') else []
            
            self.gdiis_failure_count = 0
            self.gdiis_current_recovery = self.gdiis_recovery_steps
            self.gdiis_weight_current = max(0.2, self.gdiis_weight_current / 2)  # Reduce weight
            
            return None, None, False, 0.0
        
        try:
            # Calculate GDIIS coefficients with comprehensive robustness measures
            if hasattr(self, 'quality_history') and len(self.quality_history) == n_points:
                qualities = self.quality_history
            else:
                qualities = np.ones(n_points)
            
            # First pass with standard filtering
            coeffs = self._solve_gdiis_equations(self.grad_history, qualities)
            coeffs, was_filtered, quality = self._filter_gdiis_coefficients(coeffs, strict=False)
            
            # If first pass needed filtering, try again with stricter limits
            if was_filtered:
                strict_coeffs = self._solve_gdiis_equations(self.grad_history, qualities)
                strict_coeffs, strict_filtered, strict_quality = self._filter_gdiis_coefficients(strict_coeffs, strict=True)
                
                # Use the better quality result
                if strict_quality > quality:
                    coeffs = strict_coeffs
                    quality = strict_quality
                    print("Using stricter coefficient filtering (better quality)")
            
            # Calculate the new geometry as a linear combination
            extrapolated_geometry = np.zeros_like(self.geom_history[0])
            for i in range(n_points):
                extrapolated_geometry += coeffs[i] * self.geom_history[i]
            
            # Check for NaN values in the result
            if np.any(np.isnan(extrapolated_geometry)):
                print("Warning: NaN values in extrapolated geometry, GDIIS calculation failed")
                self.gdiis_failure_count += 1
                return None, None, False, 0.0
            
            # Print coefficients (only if they're reasonable)
            print("GDIIS coefficients:", ", ".join(f"{c:.4f}" for c in coeffs))
            print(f"GDIIS quality metric: {quality:.4f}")
            
            return extrapolated_geometry, coeffs, True, quality
            
        except Exception as e:
            print(f"GDIIS extrapolation failed: {str(e)}")
            self.gdiis_failure_count += 1
            return None, None, False, 0.0

    def _validate_gdiis_step(self, rfo_step, gdiis_step, B_g, quality):
        """
        Comprehensive validation of the GDIIS step
        
        Parameters:
        -----------
        rfo_step : numpy.ndarray
            Step calculated by the RFO method
        gdiis_step : numpy.ndarray
            Step calculated by the GDIIS method
        B_g : numpy.ndarray
            Current gradient
        quality : float
            Quality metric from coefficient calculation
            
        Returns:
        --------
        tuple
            (is_valid, validation_quality)
        """
        # 1. Check gradient alignment
        grad_norm = np.linalg.norm(B_g)
        if grad_norm > 1e-10:
            # Calculate normalized dot products with negative gradient
            neg_grad = -B_g / grad_norm
            rfo_alignment = np.dot(rfo_step.flatten(), neg_grad.flatten()) / np.linalg.norm(rfo_step)
            gdiis_alignment = np.dot(gdiis_step.flatten(), neg_grad.flatten()) / np.linalg.norm(gdiis_step)
            
            # GDIIS should point in a reasonable direction compared to RFO
            if rfo_alignment > 0.3 and gdiis_alignment < 0:
                print(f"GDIIS step rejected: opposing gradient direction (RFO: {rfo_alignment:.4f}, GDIIS: {gdiis_alignment:.4f})")
                return False, 0.0
        
        # 2. Check step size ratio
        rfo_norm = np.linalg.norm(rfo_step)
        gdiis_norm = np.linalg.norm(gdiis_step)
        
        if rfo_norm > 1e-10:
            step_ratio = gdiis_norm / rfo_norm
            if step_ratio > self.gdiis_step_ratio_max:
                print(f"GDIIS step too large: {step_ratio:.2f} times RFO step")
                return False, 0.0
            
            # Calculate quality based on step ratio (closer to 1.0 is better)
            ratio_quality = 1.0 - min(1.0, abs(np.log10(step_ratio)))
        else:
            ratio_quality = 0.5  # Neutral if RFO step is near zero
        
        # 3. Check for outliers in the step components
        step_diff = gdiis_step - rfo_step
        mean_diff = np.mean(step_diff)
        std_diff = np.std(step_diff)
        
        if std_diff > 1e-10:
            # Check for components that are far from the mean difference
            outliers = np.abs(step_diff - mean_diff) > self.gdiis_outlier_threshold * std_diff
            outlier_fraction = np.sum(outliers) / len(step_diff)
            
            if outlier_fraction > 0.1:  # More than 10% of components are outliers
                print(f"GDIIS step rejected: {outlier_fraction*100:.1f}% of components are outliers")
                return False, 0.0
        
        # 4. Overall validation quality (combine multiple factors)
        validation_quality = (ratio_quality + quality) / 2.0
        
        return True, validation_quality

    def gdiis_rfo(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        """
        Enhanced RFO with robust GDIIS: Combines basic RFO method with GDIIS extrapolation
        with comprehensive stability measures and dynamic adaptation
        """
        print("Enhanced GDIIS-RFO: Rational Function Optimization with robust GDIIS acceleration")
        n_coords = len(geom_num_list)
        
        # Track gradient RMS for convergence monitoring
        grad_rms = np.sqrt(np.mean(B_g ** 2))
        
        if self.Initialization:
            self.lambda_s_scale = 0.1
            self.Initialization = False
            self.geom_history = []  # Reset GDIIS history
            self.grad_history = []
            self.quality_history = []
            self.gdiis_failure_count = 0
            self.gdiis_weight_current = self.gdiis_weight_initial
            self.prev_grad_rms = grad_rms
            self.non_improving_count = 0
            return self.DELTA * B_g
            
        print("saddle order:", self.saddle_order)
        print(f"Gradient RMS: {grad_rms:.8f}")
        
        # Check convergence progress
        improving = grad_rms < self.prev_grad_rms * 0.95
        if improving:
            self.non_improving_count = 0
        else:
            self.non_improving_count += 1
            if self.non_improving_count > 2:
                # Reduce GDIIS weight if optimization is stalling
                self.gdiis_weight_current = max(0.1, self.gdiis_weight_current - 0.1)
                print(f"Optimization stalling, reducing GDIIS weight to {self.gdiis_weight_current:.2f}")
                self.non_improving_count = 0
        
        self.prev_grad_rms = grad_rms
            
        # Calculate geometry and gradient differences
        delta_grad = (g - pre_g).reshape(n_coords, 1)
        displacement = (geom_num_list - pre_geom).reshape(n_coords, 1)
        DELTA_for_QNM = self.DELTA
        
        # Update Hessian
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            try:
                delta_hess = self.hessian_update(displacement, delta_grad)
                new_hess = self.hessian + delta_hess + self.bias_hessian
            except np.linalg.LinAlgError:
                print("Warning: Hessian update failed, using previous Hessian")
                new_hess = self.hessian + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
            
        new_hess = 0.5 * (new_hess + new_hess.T)
        
        try:
            # Standard RFO step calculation
            matrix_for_RFO = np.append(new_hess, B_g.reshape(n_coords, 1), axis=1)
            tmp = np.array([np.append(B_g.reshape(1, n_coords), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, _ = np.linalg.eigh(matrix_for_RFO)
            eigenvalue = np.sort(eigenvalue)
            lambda_for_calc = float(eigenvalue[self.saddle_order])
            
            print("RFO lambda: ", lambda_for_calc)
            
            # Calculate RFO step
            rfo_step = DELTA_for_QNM * np.linalg.solve(
                new_hess - self.lambda_s_scale*lambda_for_calc*(np.eye(n_coords)), 
                B_g.reshape(n_coords, 1)
            )
        except np.linalg.LinAlgError:
            # Fallback if RFO step calculation fails
            print("Warning: RFO step calculation failed, using scaled gradient")
            rfo_step = -0.1 * B_g.reshape(n_coords, 1)
        
        # Update GDIIS history with quality information
        step_quality = 1.0  # Default quality
        if self.iter > 0 and np.linalg.norm(pre_B_g) > 1e-10:
            # Estimate quality based on gradient reduction
            grad_change_ratio = np.linalg.norm(B_g) / np.linalg.norm(pre_B_g)
            if grad_change_ratio < 1.0:
                # Gradient decreased, good quality
                step_quality = 1.0
            else:
                # Gradient increased, lower quality
                step_quality = max(0.3, 1.0 / (1.0 + 2*np.log(grad_change_ratio)))
        
        self._update_gdiis_history(geom_num_list, B_g, step_quality)
        
        # Skip GDIIS if in recovery mode
        if self.gdiis_current_recovery > 0:
            self.gdiis_current_recovery -= 1
            print(f"In GDIIS recovery mode ({self.gdiis_current_recovery} steps remaining), skipping GDIIS")
            move_vector = rfo_step
        # Apply GDIIS if enough history has been accumulated
        elif len(self.geom_history) >= self.gdiis_min_points:
            # Calculate GDIIS geometry with robust coefficient handling
            gdiis_geom, gdiis_coeffs, success, quality = self._calculate_gdiis_geometry()
            
            if success and gdiis_geom is not None:
                # Calculate GDIIS step
                gdiis_step = (gdiis_geom - geom_num_list).reshape(n_coords, 1)
                
                # Validate GDIIS step
                is_valid, validation_quality = self._validate_gdiis_step(rfo_step, gdiis_step, B_g, quality)
                
                if is_valid:
                    # Calculate adaptive weight based on quality metrics
                    if self.gdiis_failure_count > 0:
                        # Reduce GDIIS weight if we've had failures
                        gdiis_weight = max(0.1, self.gdiis_weight_current - self.gdiis_failure_count * self.gdiis_weight_decrement)
                    elif grad_rms < 0.01:
                        # Increase GDIIS weight as we converge
                        gdiis_weight = min(self.gdiis_weight_max, self.gdiis_weight_current + self.gdiis_weight_increment)
                    elif self.non_improving_count > 0:
                        # Reduce weight if progress is stalling
                        gdiis_weight = max(0.1, self.gdiis_weight_current - 0.05 * self.non_improving_count)
                    else:
                        gdiis_weight = self.gdiis_weight_current
                    
                    # Scale weight by validation quality
                    gdiis_weight *= validation_quality
                    
                    rfo_weight = 1.0 - gdiis_weight
                    
                    # Calculate blended step
                    move_vector = rfo_weight * rfo_step + gdiis_weight * gdiis_step
                    print(f"Using blended step: {rfo_weight:.2f}*RFO + {gdiis_weight:.2f}*GDIIS")
                    
                    # Safety check: verify step size is reasonable
                    rfo_norm = np.linalg.norm(rfo_step)
                    blended_norm = np.linalg.norm(move_vector)
                    
                    if blended_norm > 2.0 * rfo_norm and blended_norm > 0.3:
                        # Cap step size to avoid large jumps
                        print("Warning: GDIIS step too large, scaling down")
                        scale_factor = 2.0 * rfo_norm / blended_norm
                        move_vector = rfo_step + scale_factor * (move_vector - rfo_step)
                        print(f"Step scaled by {scale_factor:.3f}")
                    
                    # Update current weight for next iteration (with moderate memory)
                    self.gdiis_weight_current = 0.7 * self.gdiis_weight_current + 0.3 * gdiis_weight
                else:
                    print("GDIIS step validation failed, using RFO step only")
                    move_vector = rfo_step
                    self.gdiis_failure_count += 1
            else:
                # GDIIS failed
                move_vector = rfo_step
                if not success:  # Only increment failure count for actual failures, not insufficient history
                    self.gdiis_failure_count += 1
        else:
            # Not enough history points yet, use standard RFO
            print(f"Building GDIIS history ({len(self.geom_history)}/{self.gdiis_min_points} points), using RFO step")
            move_vector = rfo_step
        
        # Final safety check for step size and numerical issues
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10:
            print("Warning: Step size too small, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
        elif np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Numerical issues detected in step, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
            # Reset GDIIS history on numerical failure
            self.geom_history = []
            self.grad_history = []
            self.quality_history = []
            self.gdiis_failure_count = 0
        
        # Update Hessian if all went well
        if self.iter > 0 and not np.any(np.isnan(delta_hess)):
            self.hessian += delta_hess
            
        self.iter += 1
        return move_vector

    def gdiis_rfo3(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        """
        Enhanced GDIIS-RFO3: Combines RFO3 with robust GDIIS extrapolation
        with comprehensive stability measures for challenging optimizations
        """
        print("Enhanced GDIIS-RFO3: Advanced Rational Function Optimization with robust GDIIS acceleration")
        n_coords = len(geom_num_list)
        
        # Track gradient RMS for convergence monitoring
        grad_rms = np.sqrt(np.mean(B_g ** 2))
        
        if self.Initialization:
            self.Initialization = False
            self.geom_history = []  # Reset GDIIS history
            self.grad_history = []
            self.quality_history = []
            self.gdiis_failure_count = 0
            self.gdiis_weight_current = self.gdiis_weight_initial
            self.prev_grad_rms = grad_rms
            self.non_improving_count = 0
            new_hess = self.hessian + self.bias_hessian
            return self.DELTA * B_g
        else:
            # Calculate geometry and gradient differences
            delta_grad = (g - pre_g).reshape(n_coords, 1)
            displacement = (geom_num_list - pre_geom).reshape(n_coords, 1)
        
            # Update Hessian if needed
            if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                try:
                    delta_hess = self.hessian_update(displacement, delta_grad)
                    new_hess = self.hessian + delta_hess + self.bias_hessian
                except np.linalg.LinAlgError:
                    print("Warning: Hessian update failed, using previous Hessian")
                    new_hess = self.hessian + self.bias_hessian
            else:
                new_hess = self.hessian + self.bias_hessian
        
        print("saddle order:", self.saddle_order)
        print(f"Gradient RMS: {grad_rms:.8f}")
        
        # Check convergence progress
        improving = grad_rms < self.prev_grad_rms * 0.95
        if improving:
            self.non_improving_count = 0
        else:
            self.non_improving_count += 1
            if self.non_improving_count > 2:
                # Reduce GDIIS weight if optimization is stalling
                self.gdiis_weight_current = max(0.1, self.gdiis_weight_current - 0.1)
                print(f"Optimization stalling, reducing GDIIS weight to {self.gdiis_weight_current:.2f}")
                self.non_improving_count = 0
        
        self.prev_grad_rms = grad_rms
        
        # Ensure symmetry and clean the Hessian
        new_hess = 0.5 * (new_hess + new_hess.T)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        
        try:
            # Construct RFO matrix
            scaled_grad = B_g
            matrix_for_RFO = np.block([
                [new_hess, scaled_grad.reshape(n_coords, 1)],
                [scaled_grad.reshape(1, n_coords), np.zeros((1, 1))]
            ])
            
            # Calculate RFO eigenvalues
            try:
                RFO_eigenvalue, _ = linalg.eigh(matrix_for_RFO)
            except np.linalg.LinAlgError:
                print("Warning: Using more robust eigenvalue algorithm")
                RFO_eigenvalue, _ = linalg.eigh(matrix_for_RFO, driver='evr')
            
            RFO_eigenvalue = np.sort(RFO_eigenvalue)
            
            # Calculate Hessian eigensystem
            try:
                hess_eigenvalue, hess_eigenvector = linalg.eigh(new_hess)
            except np.linalg.LinAlgError:
                print("Warning: Using more robust eigenvalue algorithm for Hessian")
                hess_eigenvalue, hess_eigenvector = linalg.eigh(new_hess, driver='evr')
            
            hess_eigenvector = hess_eigenvector.T
            hess_eigenval_indices = np.argsort(hess_eigenvalue)
            
            # Get the appropriate lambda value
            lambda_for_calc = float(RFO_eigenvalue[max(self.saddle_order, 0)])
            
            # Refine lambda considering non-linear effects (RFO3 feature)
            try:
                lambda_for_calc = RFOSecularSolverIterative().calc_rfo_lambda_and_step(
                    hess_eigenvector, hess_eigenvalue, lambda_for_calc, B_g, self.saddle_order)
                print("Refined RFO lambda: ", lambda_for_calc)
            except Exception as e:
                print(f"Lambda refinement failed: {str(e)}, using initial lambda")
            
            # Calculate RFO step using eigendecomposition
            rfo_step = np.zeros((n_coords, 1))
            saddle_order_count = 0
            
            EIGENVAL_THRESHOLD = 1e-7
            DENOM_THRESHOLD = 1e-10
            
            for i in range(len(hess_eigenvalue)):
                idx = hess_eigenval_indices[i]
                
                if np.abs(hess_eigenvalue[idx]) < EIGENVAL_THRESHOLD:
                    continue
                    
                tmp_vector = hess_eigenvector[idx].reshape(1, n_coords)
                proj_magnitude = np.dot(tmp_vector, B_g.reshape(n_coords, 1))
                
                if saddle_order_count < self.saddle_order:
                    if self.projection_eigenvector_flag:
                        continue
                        
                    step_scaling = 1.0
                    tmp_eigval = hess_eigenvalue[idx]
                    denom = tmp_eigval + lambda_for_calc
                    
                    if np.abs(denom) > DENOM_THRESHOLD:
                        contribution = (step_scaling * self.DELTA * proj_magnitude * tmp_vector.T) / denom
                        rfo_step += contribution
                        saddle_order_count += 1
                    
                else:
                    step_scaling = 1.0
                    
                    # Handle combination of eigenvectors for unwanted saddle points
                    if (self.grad_rms_threshold > np.sqrt(np.mean(B_g ** 2)) and 
                        hess_eigenvalue[idx] < -1e-9 and 
                        self.combine_eigvec_flag):
                            
                        print(f"Combining {self.combine_eigen_vec_num} eigenvectors to avoid unwanted saddle point...")
                        combined_vector = tmp_vector.copy()
                        count = 1
                        
                        for j in range(1, self.combine_eigen_vec_num):
                            next_idx = idx + j
                            if next_idx >= len(hess_eigenvalue):
                                break
                            combined_vector += hess_eigenvector[hess_eigenval_indices[next_idx]].reshape(1, n_coords)
                            count += 1
                        
                        tmp_vector = combined_vector / count
                    
                    denom = hess_eigenvalue[idx] - lambda_for_calc
                    
                    if np.abs(denom) > DENOM_THRESHOLD:
                        contribution = (step_scaling * self.DELTA * proj_magnitude * tmp_vector.T) / denom
                        rfo_step += contribution
            
        except Exception as e:
            # Fallback if RFO step calculation fails
            print(f"Warning: RFO3 step calculation failed ({str(e)}), using scaled gradient")
            rfo_step = -0.1 * B_g.reshape(n_coords, 1)
        
        # Update GDIIS history with quality information
        step_quality = 1.0  # Default quality
        if self.iter > 0 and np.linalg.norm(pre_B_g) > 1e-10:
            # Estimate quality based on gradient reduction
            grad_change_ratio = np.linalg.norm(B_g) / np.linalg.norm(pre_B_g)
            if grad_change_ratio < 1.0:
                # Gradient decreased, good quality
                step_quality = 1.0
            else:
                # Gradient increased, lower quality
                step_quality = max(0.3, 1.0 / (1.0 + 2*np.log(grad_change_ratio)))
        
        self._update_gdiis_history(geom_num_list, B_g, step_quality)
        
        # Skip GDIIS if in recovery mode
        if self.gdiis_current_recovery > 0:
            self.gdiis_current_recovery -= 1
            print(f"In GDIIS recovery mode ({self.gdiis_current_recovery} steps remaining), skipping GDIIS")
            move_vector = rfo_step
        # Apply GDIIS if enough history has been accumulated
        elif len(self.geom_history) >= self.gdiis_min_points:
            # Calculate GDIIS geometry with robust coefficient handling
            gdiis_geom, gdiis_coeffs, success, quality = self._calculate_gdiis_geometry()
            
            if success and gdiis_geom is not None:
                # Calculate GDIIS step
                gdiis_step = (gdiis_geom - geom_num_list).reshape(n_coords, 1)
                
                # Extra validation for RFO3 + GDIIS (more stringent than regular GDIIS-RFO)
                is_valid, validation_quality = self._validate_gdiis_step(rfo_step, gdiis_step, B_g, quality)
                
                if is_valid:
                    # Calculate adaptive weight based on quality metrics and optimization progress
                    if self.gdiis_failure_count > 0:
                        # Reduce GDIIS weight if we've had failures
                        gdiis_weight = max(0.05, self.gdiis_weight_current - self.gdiis_failure_count * self.gdiis_weight_decrement)
                    elif grad_rms < self.grad_rms_threshold:
                        # Increase GDIIS weight as we converge
                        gdiis_weight = min(self.gdiis_weight_max, self.gdiis_weight_current + self.gdiis_weight_increment)
                    elif self.non_improving_count > 0:
                        # Reduce weight if progress is stalling
                        gdiis_weight = max(0.05, self.gdiis_weight_current - 0.1 * self.non_improving_count)
                    else:
                        gdiis_weight = self.gdiis_weight_current
                    
                    # Scale weight by validation quality and current progress
                    gdiis_weight *= validation_quality
                    
                    # For saddle searches, reduce GDIIS influence near critical regions
                    if self.saddle_order > 0 and grad_rms < 0.05:
                        gdiis_weight *= 0.5
                        print("Near saddle point, reducing GDIIS influence")
                    
                    rfo_weight = 1.0 - gdiis_weight
                    
                    # Calculate blended step
                    move_vector = rfo_weight * rfo_step + gdiis_weight * gdiis_step
                    print(f"Using blended step: {rfo_weight:.2f}*RFO3 + {gdiis_weight:.2f}*GDIIS")
                    
                    # Safe-guard for step size
                    rfo_norm = np.linalg.norm(rfo_step)
                    blended_norm = np.linalg.norm(move_vector)
                    
                    # Stricter step size control for RFO3+GDIIS
                    max_ratio = 1.5  # More conservative than GDIIS-RFO
                    if blended_norm > max_ratio * rfo_norm and blended_norm > 0.2:
                        scale_factor = max_ratio * rfo_norm / blended_norm
                        print(f"Scaling down GDIIS step by factor {scale_factor:.3f}")
                        move_vector = rfo_step + scale_factor * (move_vector - rfo_step)
                    
                    # Update current weight for next iteration (with slower adaptation for stability)
                    self.gdiis_weight_current = 0.8 * self.gdiis_weight_current + 0.2 * gdiis_weight
                else:
                    print("GDIIS step validation failed, using RFO3 step only")
                    move_vector = rfo_step
                    self.gdiis_failure_count += 1
            else:
                # GDIIS failed
                move_vector = rfo_step
                if not success:  # Only increment failure count for actual failures
                    self.gdiis_failure_count += 1
        else:
            # Not enough history points yet, use standard RFO
            print(f"Building GDIIS history ({len(self.geom_history)}/{self.gdiis_min_points} points), using RFO3 step")
            move_vector = rfo_step
        
        # Final safety check for step size and numerical issues
        move_norm = np.linalg.norm(move_vector)
        if move_norm < 1e-10:
            print("Warning: Step size too small, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
        elif np.any(np.isnan(move_vector)) or np.any(np.isinf(move_vector)):
            print("Warning: Numerical issues detected in step, using scaled gradient instead")
            move_vector = -0.1 * B_g.reshape(n_coords, 1)
            # Reset GDIIS history on numerical failure
            self.geom_history = []
            self.grad_history = []
            self.quality_history = []
            self.gdiis_failure_count = 0
        
        # Check step size and update hessian
        if self.iter > 0 and not np.any(np.isnan(delta_hess)):
            self.hessian += delta_hess  
        
        self.iter += 1
        return move_vector  # in Bohr
        
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        if "mrfo" in self.config["method"].lower():
            move_vector = self.moment(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "gdiis_rfo3" in self.config["method"].lower():
            move_vector = self.gdiis_rfo3(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "gdiis_rfo" in self.config["method"].lower():
            move_vector = self.gdiis_rfo(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo2" in self.config["method"].lower():
            move_vector = self.normal_v2(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo3" in self.config["method"].lower():
            move_vector = self.normal_v3(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo_neb" in self.config["method"].lower():
            move_vector = self.neb(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        else:
            move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector
 

class RFOSecularSolverIterative:
    """
    Implements a robust iterative method for the Rational Function Optimization (RFO) secular equation:
    F(λ) = ∑ (sigma_j^2 / (scale*λ - eigval_j)) - λ = 0
    
    Fully vectorized implementation for high performance.
    """
    def __init__(self,
                 scale=1.0,
                 max_iter=10000,
                 tol=1e-10,
                 delta=1e-4):
        """
        :param scale: Scalar factor applied to λ in the denominator (commonly 1.0).
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence threshold for the iterative method.
        :param delta: Determines how aggressively to adjust λ if the Newton step is unstable.
        """
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta

    def calc_rfo_lambda_and_step(self, hess_eigvec, hess_eigval, init_lambda_val, B_g, order):
        """
        Computes λ and the updated geometry step vector using a vectorized iterative approach.
        
        :param hess_eigvec: Eigenvectors of the Hessian, shape (n, n).
        :param hess_eigval: Eigenvalues of the Hessian, length n.
        :param init_lambda_val: Initial guess for λ.
        :param B_g: Gradient vector in the chosen basis, shape (n, 1) or (n,).
        :param order: Number of eigenvalues to treat with positive sign.
        :return: (final_lambda, move_vector)
        """
        print("Perform lambda optimization to calculate appropriate step size...")
        # Flatten gradient vector for consistent operations
        b_g = B_g.ravel()
        n = len(hess_eigval)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(hess_eigval)
        hess_eigval = hess_eigval[sorted_indices]
        hess_eigvec = hess_eigvec[:, sorted_indices]
        
        # Calculate sigma values: σⱼ = eigvecⱼᵀ · gradient
        sigma_values = np.dot(hess_eigvec.T, b_g)
        sigma_squared = sigma_values**2
        
        # Pre-compute sign array for eigenvalue terms
        # +1 for j < order, -1 for j >= order
        sign_array = np.ones(n)
        sign_array[order:] = -1
        
        # Try multiple starting points to ensure global solution
        lambda_candidates = np.linspace(init_lambda_val + 10, init_lambda_val - 100.0, 15)
        best_lambda = None
        best_residual = float('inf')
        
        for lambda_start in lambda_candidates:
            current_lambda = lambda_start
            
            for iteration in range(self.max_iter):
                # Compute denominators for all terms vectorized
                # scale*λ + eigval_j for j < order
                # scale*λ - eigval_j for j >= order
                denominators = self.scale * current_lambda + sign_array * hess_eigval
                
                # Add small value to avoid division by zero (vectorized)
                safe_denom = np.where(np.abs(denominators) < 1e-10, 
                                     np.sign(denominators) * 1e-10, 
                                     denominators)
                
                # Evaluate secular function F(λ) = ∑(σ²/(scale*λ±eigval)) - λ
                f_value = np.sum(sigma_squared / safe_denom) - current_lambda
                
                # First derivative: F'(λ) = -∑(scale*σ²/(scale*λ±eigval)²) - 1
                f_prime = -np.sum(self.scale * sigma_squared / safe_denom**2) - 1.0
                
                # Second derivative (Hessian): F''(λ) = 2*∑(scale²*σ²/(scale*λ±eigval)³)
                f_double_prime = 2.0 * np.sum(self.scale**2 * sigma_squared / safe_denom**3)
                
                # Check for convergence
                if np.abs(f_value) < self.tol:
                    if np.abs(f_value) < best_residual:
                        best_residual = np.abs(f_value)
                        best_lambda = current_lambda
                        print(f"Converged in {iteration} iterations with λ = {current_lambda:.10f}, residual = {f_value:.10e}")
                    break
                
                # Newton step using second derivative (Hessian)
                if np.abs(f_prime) < 1e-12 or np.abs(f_double_prime) < 1e-12:
                    # If derivatives are near zero, use a simple shift
                    step = self.delta * np.sign(f_value)
                else:
                    # Full Newton step with second derivative
                    step = -f_value * f_prime / (f_prime**2 - f_value * f_double_prime)
                    
                    # If the Newton step is large or in the wrong direction, fall back to regular Newton
                    if np.abs(step) > 1.0 or np.sign(step) != np.sign(-f_value / f_prime):
                        step = -f_value / f_prime
                
                # Damping for large steps
                if np.abs(step) > 0.1:
                    step = 0.1 * np.sign(step)
                
                # Update lambda
                new_lambda = current_lambda + step
                
                # Safety check for numerical issues
                if np.isnan(new_lambda) or np.isinf(new_lambda):
                    new_lambda = current_lambda + self.delta * np.sign(f_value)
                
                current_lambda = new_lambda
            
            # Store result if this is the best we've seen
            if iteration < self.max_iter - 1 and (best_lambda is None or np.abs(f_value) < best_residual):
                best_lambda = current_lambda
                best_residual = np.abs(f_value)
        
        if best_lambda is None:
            print(f"Warning: Failed to converge. Using initial value: {init_lambda_val:.10f}")
            best_lambda = init_lambda_val
        else:
            print(f"Optimization result: {init_lambda_val:.10f} -> {best_lambda:.10f}")
        
        # Calculate the step vector using the optimal lambda (vectorized)
        
        return best_lambda
   