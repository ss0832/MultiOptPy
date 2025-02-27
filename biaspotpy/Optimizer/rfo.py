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
                if np.abs(denom) < DENOM_THRESHOLD:
                    denom = np.sign(denom) * DENOM_THRESHOLD if denom != 0 else DENOM_THRESHOLD
                
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
                if np.abs(denom) < DENOM_THRESHOLD:
                    denom = np.sign(denom) * DENOM_THRESHOLD if denom != 0 else DENOM_THRESHOLD
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
                if np.abs(denom) < DENOM_THRESHOLD:
                    denom = np.sign(denom) * DENOM_THRESHOLD if denom != 0 else DENOM_THRESHOLD
                
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
                if np.abs(denom) < DENOM_THRESHOLD:
                    denom = np.sign(denom) * DENOM_THRESHOLD if denom != 0 else DENOM_THRESHOLD
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

    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        if "mrfo" in self.config["method"].lower():
            move_vector = self.moment(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo2" in self.config["method"].lower():
            move_vector = self.normal_v2(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "rfo3" in self.config["method"].lower():
            move_vector = self.normal_v3(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
               
        elif "rfo_neb" in self.config["method"].lower():
            move_vector = self.neb(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        else:
            move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector
 
import numpy as np

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
        print("Perform lambda optimization to define appropriate step size...")
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
        lambda_candidates = np.linspace(init_lambda_val, init_lambda_val - 100.0, 10)
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
   