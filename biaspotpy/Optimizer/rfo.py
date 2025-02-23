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
        cleaned_eigval = np.where(valid_mask, eigval, 0.0)
        
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
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
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
        
        # Initial step handling
        if self.Initialization:
            self.Initialization = False
            return self.DELTA * B_g
        
        n_coords = len(geom_num_list)
        print(f"Searching for saddle point of order: {self.saddle_order}")
        
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
        
        # Ensure Hessian symmetry
        new_hess = 0.5 * (new_hess + new_hess.T)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        # Construct augmented RFO matrix with improved numerical stability
        grad_norm = np.linalg.norm(B_g)
        
        scaled_grad = B_g / grad_norm
            
        # Construct the RFO matrix
        rfo_matrix = np.zeros((n_coords + 1, n_coords + 1))
        rfo_matrix[:n_coords, :n_coords] = new_hess
        rfo_matrix[:n_coords, -1] = scaled_grad.reshape(n_coords)
        rfo_matrix[-1, :n_coords] = scaled_grad.reshape(n_coords)
        
        # Solve eigenvalue problem with enhanced precision
        
        try:
            rfo_eigenvalues, _ = linalg.eigh(rfo_matrix)

        except np.linalg.LinAlgError:
            print("Warning: Eigenvalue computation failed, using more stable algorithm")
            rfo_eigenvalues, _ = linalg.eigh(rfo_matrix, driver='evr')

        # Sort eigenvalues and find appropriate one for the desired saddle order
       
        mask = np.abs(rfo_eigenvalues) > 1e-5
        sorted_indices = np.argsort(rfo_eigenvalues[mask])
        lambda_index = sorted_indices[self.saddle_order]
        lambda_for_calc = float(rfo_eigenvalues[mask][lambda_index])
       
        
        print(f"Selected eigenvalue: {lambda_for_calc:12.9f}")
        
        # Calculate step with trust radius control
        try:
            # Solve the linear system with improved conditioning
            H = new_hess - self.lambda_s_scale * lambda_for_calc * np.eye(n_coords)
            # Use more stable solver
            move_vector = linalg.solve(H, B_g.reshape(n_coords, 1), assume_a='sym')
        except np.linalg.LinAlgError:
            print("Warning: Linear system solve failed, using pseudoinverse")
            H_pinv = linalg.pinvh(H)
            move_vector = np.dot(H_pinv, B_g.reshape(n_coords, 1))
        
        step_norm = np.linalg.norm(move_vector)
        # Check for convergence and update
        if step_norm < 1e-10:
            print("Warning: Step size is too small!")
            if np.linalg.norm(B_g) > 1e-6:
                print("But gradient is still significant - might be stuck!")
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
            return self.DELTA * B_g
        
        print("saddle order:", self.saddle_order)
        
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
        
        # Ensure symmetry and remove small eigenvalues
        new_hess = 0.5 * (new_hess + new_hess.T)
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
        # Construct RFO matrix with improved stability
        grad_norm = np.linalg.norm(B_g)
        if grad_norm > 1e-10:
            scaled_grad = B_g / grad_norm
        else:
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
        EIGENVAL_THRESHOLD = 1e-10
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
                if np.abs(hess_eigenvalue[idx]) < EIGENVAL_THRESHOLD:
                    continue
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
        else:
            
            self.hessian += delta_hess  # Store cleaned Hessian
        
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
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
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
            self.iter += 1
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
        new_hess = self.project_out_hess_tr_and_rot_for_coord(new_hess, geom_num_list, display_eigval=False)
        new_hess, _ = self.get_cleaned_hessian(new_hess)
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
 