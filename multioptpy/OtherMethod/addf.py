import copy
import numpy as np
import datetime
import os

from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.fileio import make_workspace, xyz2list
from multioptpy.Parameters.parameter import UnitValueLib, element_number


class ADDFlikeMethod:
    def __init__(self, config):
        """
        Implementation of ADD (Anharmonic Downward Distortion) method based on SHS4py approach
        # ref. : Journal of chemical theory and computation 16.6 (2020): 3869-3878. (https://pubs.acs.org/doi/10.1021/acs.jctc.0c00010)
        # ref. : Chem. Phys. Lett. 2004, 384 (4–6), 277–282.
        # ref. : Chem. Phys. Lett. 2005, 404 (1–3), 95–99.
        """
        self.config = config
        self.addf_config = {
            'step_number': int(config.addf_step_num),
            'number_of_add': int(config.nadd),
            'IOEsphereA_initial': 0.01,  # Initial hypersphere radius
            'IOEsphereA_dist': float(config.addf_step_size),    # Increment for hypersphere radius
            'IOEthreshold': 0.01,        # Threshold for IOE
            'minimize_threshold': 1.0e-5,# Threshold for minimization
        }
        self.energy_list_1 = []
        self.energy_list_2 = []
        self.gradient_list_1 = []
        self.gradient_list_2 = []
        self.init_displacement = 0.03 / self.get_unit_conversion()  # Bohr
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.converge_criteria = 0.12
        self.element_number_list = None
        self.ADDths = []  # List to store ADD theta classes
        self.optimized_structures = {}  # Dictionary to store optimized structures by ADD ID
        self.max_iterations = 10

    def get_unit_conversion(self):
        """Return bohr to angstrom conversion factor"""
        return UnitValueLib().bohr2angstroms  # Approximate value for bohr2angstroms
        
    def adjust_center2origin(self, coord):
        """Adjust coordinates to have center at origin"""
        center = np.mean(coord, axis=0)
        return coord - center

    def SuperSphere_cartesian(self, A, thetalist, SQ, dim):
        """
        Vector of super sphere by cartesian (basis transformation from polar to cartesian)
        {sqrt(2*A), theta_1,..., theta_n-1} ->  {q_1,..., q_n} -> {x_1, x_2,..., x_n}
        """
        n_components = min(dim, SQ.shape[1] if SQ.ndim > 1 else dim)
        
        qlist = np.zeros(n_components)
        
        # Fill q-list using hyperspherical coordinates
        a_k = np.sqrt(2.0 * A)
        for i in range(min(len(thetalist), n_components-1)):
            qlist[i] = a_k * np.cos(thetalist[i])
            a_k *= np.sin(thetalist[i])
        
        # Handle the last component
        if n_components > 0:
            qlist[n_components-1] = a_k
        
        # Transform to original space
        SSvec = np.dot(SQ, qlist)
        
        return SSvec  # This is a vector in the reduced space
    
    def calctheta(self, vec, eigVlist, eigNlist):
        """
        Calculate the polar coordinates (theta) from a vector
        """
        # Get actual dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector
        n_components = min(len(eigVlist), len(eigNlist))  # Number of eigenvectors
        
        # Check vector dimensions
        if len(vec) != n_features:
            # If dimensions don't match, truncate or pad the vector
            if len(vec) > n_features:
                vec = vec[:n_features]  # Truncate
            else:
                padded_vec = np.zeros(n_features)
                padded_vec[:len(vec)] = vec
                vec = padded_vec
        
        # Create SQ_inv matrix with correct dimensions
        SQ_inv = np.zeros((n_components, n_features))
        
        for i in range(n_components):
            SQ_inv[i] = eigVlist[i] / np.sqrt(abs(eigNlist[i]))
        
        # Perform the dot product
        qvec = np.dot(SQ_inv, vec)
        
        r = np.linalg.norm(qvec)
        if r < 1e-10:
            return np.zeros(n_components - 1)
            
        thetalist = []
        for i in range(len(qvec) - 1):
            # Handle possible numerical issues with normalization
            norm_q = np.linalg.norm(qvec[i:])
            if norm_q < 1e-10:
                theta = 0.0
            else:
                cos_theta = qvec[i] / norm_q
                cos_theta = max(-1.0, min(1.0, cos_theta))  # Ensure within bounds
                theta = np.arccos(cos_theta)
                if i == len(qvec) - 2 and qvec[-1] < 0:
                    theta = 2*np.pi - theta
            thetalist.append(theta)
        
        return np.array(thetalist)
    
    def SQaxes(self, eigNlist, eigVlist, dim):
        """Calculate the SQ matrix for transformation"""
        # Get actual available dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector
        n_components = min(len(eigVlist), len(eigNlist), dim)  # Number of eigenvectors to use
        
        # Initialize with correct dimensions
        SQ = np.zeros((n_features, n_components))
        
        # Only iterate up to the available components
        for i in range(n_components):
            SQ[:, i] = eigVlist[i] * np.sqrt(abs(eigNlist[i]))
        
        return SQ

    def SQaxes_inv(self, eigNlist, eigVlist, dim):
        """Calculate the inverse SQ matrix for transformation"""
        # Get actual available dimensions
        n_features = eigVlist[0].shape[0]  # Length of each eigenvector 
        n_components = min(len(eigVlist), len(eigNlist), dim)  # Number of eigenvectors to use
        
        # Initialize with correct dimensions
        SQ_inv = np.zeros((n_components, n_features))
        
        # Only iterate up to the available components
        for i in range(n_components):
            SQ_inv[i] = eigVlist[i] / np.sqrt(abs(eigNlist[i]))
        
        return SQ_inv
    
    def angle(self, v1, v2):
        """Calculate angle between two vectors"""
        # Check for zero vectors or invalid inputs
        if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
            return 0.0
            
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        
        # Handle potential numerical issues
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        
        return np.arccos(dot_product)
    
    def angle_SHS(self, v1, v2, SQ_inv):
        """Calculate angle between two vectors in SHS space"""
        # Ensure vectors are flattened
        v1_flat = v1.flatten() if hasattr(v1, 'flatten') else v1
        v2_flat = v2.flatten() if hasattr(v2, 'flatten') else v2
        
        # Handle potential dimension mismatches
        min_dim = min(len(v1_flat), len(v2_flat), SQ_inv.shape[1])
        v1_flat = v1_flat[:min_dim]
        v2_flat = v2_flat[:min_dim]
        
        q_v1 = np.dot(SQ_inv[:, :min_dim], v1_flat)
        q_v2 = np.dot(SQ_inv[:, :min_dim], v2_flat)
        return self.angle(q_v1, q_v2)
    
    def calc_onHS(self, deltaTH, func, eqpoint, thetalist, IOEsphereA, A_eq):
        """
        Calculate function value on hypersphere
        """
        thetalist_new = thetalist + deltaTH
        nADD = self.SuperSphere_cartesian(IOEsphereA, thetalist_new, self.SQ, self.dim)
        x = eqpoint + nADD
        x = self.periodicpoint(x)
        
        result = func(x) - IOEsphereA - A_eq
        result += self.IOE_total(nADD)
        return result
    
    def IOE_total(self, nADD):
        """Sum up all IOE illumination"""
        result = 0.0
        for ADDth in self.ADDths:
            if self.current_id == ADDth.IDnum:
                continue
            if ADDth.ADDoptQ:
                continue
            if ADDth.ADD_IOE <= -1000000 or ADDth.ADD_IOE > 10000000:
                continue
            if ADDth.ADD <= self.current_ADD:
                result -= self.IOE(nADD, ADDth)
        return result
    
    def IOE(self, nADD, neiborADDth):
        """Calculate IOE between current point and neighbor"""
        deltaTH = self.angle_SHS(nADD, neiborADDth.nADD, self.SQ_inv)
        if deltaTH <= np.pi * 0.5:
            cosdamp = np.cos(deltaTH)
            return neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
        else:
            return 0.0
    
    def grad_hypersphere(self, f, grad, eqpoint, IOEsphereA, thetalist):
        """Calculate gradient on hypersphere"""
        # Generate nADD in the reduced space (this will be a vector in the Hessian eigenspace)
        nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
        
        # We need to convert the reduced space vector back to the full coordinate space
        # First create a zero vector in the full space
        n_atoms = eqpoint.shape[0]
        n_coords = n_atoms * 3
        nADD_full = np.zeros(n_coords)
        
        # Map the reduced vector to the full space (this is approximate)
        for i in range(min(len(nADD_reduced), n_coords)):
            nADD_full[i] = nADD_reduced[i]
            
        # Reshape to molecular geometry format
        nADD = nADD_full.reshape(n_atoms, 3)
        
        # Calculate the normalized direction vector for projecting out later
        EnADD = nADD_full / np.linalg.norm(nADD_full)
        
        # Apply the displacement to the initial geometry
        target_point = eqpoint + nADD
        target_point = self.periodicpoint(target_point)
        
        # Calculate gradient at this point
        grad_x = grad(target_point)
        if isinstance(grad_x, bool) and grad_x is False:
            return False, False
        
        # Flatten the gradient for vector operations
        grad_x_flat = grad_x.flatten()
        
        # Project gradient onto tangent space of hypersphere
        # We remove the component along the displacement vector
        returngrad_flat = grad_x_flat - np.dot(grad_x_flat, EnADD) * EnADD
        
        # Apply IOE contributions (if implemented)
        for ADDth in self.ADDths:
            # Skip if this is the current ADD or if current_id is None
            if self.current_id is not None and self.current_id == ADDth.IDnum:
                continue
                
            # Skip if this ADD is being optimized
            if ADDth.ADDoptQ:
                continue
                
            # Only apply IOE for ADDs with lower energy (if current_ADD is set)
            if self.current_ADD is not None and hasattr(ADDth, 'ADD') and ADDth.ADD is not None:
                if ADDth.ADD <= self.current_ADD:
                    ioe_grad = self.IOE_grad(nADD_full, ADDth)
                    if ioe_grad is not None:
                        returngrad_flat -= ioe_grad
        
        # Reshape gradient back to molecular geometry format for easier handling
        returngrad = returngrad_flat.reshape(n_atoms, 3)
        
        return target_point, returngrad

    def IOE_grad(self, nADD, neiborADDth):
        """Calculate gradient of IOE"""
        # Make sure we're working with flattened arrays
        nADD_flat = nADD.flatten() if hasattr(nADD, 'flatten') else nADD
        nADD_neibor = neiborADDth.nADD_full if hasattr(neiborADDth, 'nADD_full') else neiborADDth.nADD.flatten()
        
        # Get minimum dimension we can work with
        min_dim = min(len(nADD_flat), len(nADD_neibor), self.SQ_inv.shape[1])
        nADD_flat = nADD_flat[:min_dim]
        nADD_neibor = nADD_neibor[:min_dim]
        
        # Transform to eigenspace
        q_x = np.dot(self.SQ_inv[:, :min_dim], nADD_flat)
        q_y = np.dot(self.SQ_inv[:, :min_dim], nADD_neibor)
        
        # Check for valid vectors before calculating angle
        if np.isnan(q_x).any() or np.isnan(q_y).any() or np.linalg.norm(q_x) < 1e-10 or np.linalg.norm(q_y) < 1e-10:
            return None
            
        # Calculate angle in eigenspace
        deltaTH = self.angle(q_x, q_y)
        
        # Check if deltaTH is valid (not None and not NaN)
        if deltaTH is None or np.isnan(deltaTH):
            return None
        
        # Initialize gradient vector
        returngrad = np.zeros(len(nADD_flat))
        eps = 1.0e-3
        
        # Calculate IOE gradient using finite differences
        if deltaTH <= np.pi * 0.5:
            cosdamp = np.cos(deltaTH)
            for i in range(len(nADD_flat)):
                nADD_eps = copy.copy(nADD_flat)
                nADD_eps[i] += eps
                
                # Transform to eigenspace
                qx_i = np.dot(self.SQ_inv[:, :min_dim], nADD_eps[:min_dim])
                deltaTH_eps = self.angle(qx_i, q_y)
                
                # Check if the new angle is valid
                if deltaTH_eps is None or np.isnan(deltaTH_eps):
                    continue
                    
                cosdamp_eps = np.cos(deltaTH_eps)
                IOE_center = neiborADDth.ADD_IOE * cosdamp * cosdamp * cosdamp
                IOE_eps = neiborADDth.ADD_IOE * cosdamp_eps * cosdamp_eps * cosdamp_eps
                
                returngrad[i] = (IOE_eps - IOE_center) / eps
            
            # Pad the gradient to the full space if needed
            full_grad = np.zeros(self.n_coords if hasattr(self, 'n_coords') else len(nADD))
            full_grad[:len(returngrad)] = returngrad
            return full_grad
        
        return None
    
    def periodicpoint(self, point):
        """Apply periodic boundary conditions if needed"""
        # Implement according to your specific requirements
        return point

    def minimizeTh_SD_SS(self, ADDth, initialpoint, f, grad, eqpoint, IOEsphereA):
        """
        Steepest descent optimization on hypersphere with step size control
        Following the implementation in SHS4py.ADD.py with added robustness
        """
        whileN = 0
        thetalist = ADDth.thetalist + initialpoint
        stepsize = 0.001
        n_atoms = eqpoint.shape[0]
        n_coords = n_atoms * 3
        
        # Generate initial nADD
        nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
        
        # Convert reduced space vector to full coordinate space with proper dimensions
        nADD_full = np.zeros(n_coords)
        for i in range(min(len(nADD_reduced), n_coords)):
            nADD_full[i] = nADD_reduced[i]
        
        # Reshape to match eqpoint dimensions
        nADD = nADD_full.reshape(n_atoms, 3)
        
        # Keep track of best solution
        best_thetalist = thetalist.copy()
        best_energy = float('inf')
        
        # Initial point
        tergetpoint = eqpoint + nADD
        tergetpoint = self.periodicpoint(tergetpoint)
        
        # Try to calculate initial energy
        try:
            initial_energy = f(tergetpoint)
            if isinstance(initial_energy, (int, float)) and not np.isnan(initial_energy):
                best_energy = initial_energy
        except Exception:
            pass  # Continue even if initial energy calculation fails
        
        # Main optimization loop
        while whileN < self.max_iterations:
            try:
                # Get gradient at current point
                grad_x = grad(tergetpoint)
                
                # If gradient calculation fails, continue with smaller step or different approach
                if grad_x is False:
                    # Try a random perturbation and continue
                    print(f"Gradient calculation failed at iteration {whileN}, trying random perturbation")
                    random_perturbation = np.random.rand(n_atoms, 3) * 0.01 - 0.005  # Small random perturbation
                    tergetpoint = tergetpoint + random_perturbation
                    tergetpoint = self.periodicpoint(tergetpoint)
                    
                    # Calculate new nADD
                    nADD = tergetpoint - eqpoint
                    thetalist = self.calctheta(nADD.flatten(), self.eigVlist, self.eigNlist)
                    
                    # Ensure we're on the hypersphere with correct radius
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    
                    tergetpoint = eqpoint + nADD
                    tergetpoint = self.periodicpoint(tergetpoint)
                    
                    whileN += 1
                    continue
                
                # Apply IOE contributions
                grad_flat = grad_x.flatten()
                for neiborADDth in self.ADDths:
                    if ADDth.IDnum == neiborADDth.IDnum:
                        continue
                    if neiborADDth.ADDoptQ:
                        continue
                    if neiborADDth.ADD <= ADDth.ADD:
                        ioe_grad = self.IOE_grad(nADD.flatten(), neiborADDth)
                        if ioe_grad is not None:
                            grad_flat = grad_flat - ioe_grad
                
                # Reshape back to molecular geometry format
                grad_x = grad_flat.reshape(n_atoms, 3)
                
                # Project gradient onto tangent space
                nADD_norm = np.linalg.norm(nADD.flatten())
                if nADD_norm < 1e-10:
                    # If nADD is too small, generate a new one
                    print(f"nADD norm too small at iteration {whileN}, regenerating")
                    thetalist = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                    nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist, self.SQ, self.dim)
                    nADD_full = np.zeros(n_coords)
                    for i in range(min(len(nADD_reduced), n_coords)):
                        nADD_full[i] = nADD_reduced[i]
                    nADD = nADD_full.reshape(n_atoms, 3)
                    tergetpoint = eqpoint + nADD
                    tergetpoint = self.periodicpoint(tergetpoint)
                    whileN += 1
                    continue
                        
                nADD_unit = nADD.flatten() / nADD_norm
                
                # Project gradient component along nADD
                grad_along_nADD = np.dot(grad_x.flatten(), nADD_unit)
                
                # Subtract this component to get the tangent space gradient
                SSgrad_flat = grad_x.flatten() - grad_along_nADD * nADD_unit
                SSgrad = SSgrad_flat.reshape(n_atoms, 3)
                
                # Check convergence
                if np.linalg.norm(SSgrad) < 1.0e-1:
                    # Update best solution if better
                    try:
                        current_energy = f(tergetpoint)
                        if isinstance(current_energy, (int, float)) and not np.isnan(current_energy):
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_thetalist = thetalist.copy()
                    except Exception:
                        pass  # Just keep the current best if energy calculation fails
                    
                    return thetalist  # Converged successfully
                
                # Store current point
                _point_initial = copy.copy(tergetpoint)
                
                # Line search
                whileN2 = 0
                stepsizedamp = stepsize
                found_valid_step = False
                
                # Try multiple step sizes
                for whileN2 in range(1, 5):  # Try up to 10 steps with varying sizes
                    try:
                        # Take step with dynamic step size
                        step_scale = whileN2 if whileN2 <= 5 else (whileN2 - 5) * 0.1
                        tergetpoint = _point_initial - step_scale * SSgrad / np.linalg.norm(SSgrad) * stepsizedamp
                        
                        # Calculate new nADD
                        nADD2 = tergetpoint - eqpoint
                        
                        # Convert to theta parameters
                        thetalist_new = self.calctheta(nADD2.flatten(), self.eigVlist, self.eigNlist)
                        
                        # Ensure we're on the hypersphere with correct radius
                        nADD2_reduced = self.SuperSphere_cartesian(IOEsphereA, thetalist_new, self.SQ, self.dim)
                        
                        # Convert reduced space vector to full coordinate space
                        nADD2_full = np.zeros(n_coords)
                        for i in range(min(len(nADD2_reduced), n_coords)):
                            nADD2_full[i] = nADD2_reduced[i]
                        
                        # Reshape to match eqpoint dimensions
                        nADD2 = nADD2_full.reshape(n_atoms, 3)
                        
                        # Calculate new point on hypersphere
                        new_point = eqpoint + nADD2
                        new_point = self.periodicpoint(new_point)
                        
                        # Calculate step size
                        delta = np.linalg.norm(nADD.flatten() - nADD2.flatten())
                        
                        # Calculate energy at new point to check improvement
                        try:
                            new_energy = f(new_point)
                            if isinstance(new_energy, (int, float)) and not np.isnan(new_energy):
                                # Accept step if it improves energy or makes reasonable movement
                                if new_energy < best_energy or delta > 0.005:
                                    found_valid_step = True
                                    if new_energy < best_energy:
                                        best_energy = new_energy
                                        best_thetalist = thetalist_new.copy()
                                    tergetpoint = new_point
                                    thetalist = thetalist_new
                                    nADD = nADD2
                                    break
                        except Exception:
                            # If energy calculation fails, accept step if it's a reasonable move
                            if delta > 0.005 and delta < 0.1:
                                found_valid_step = True
                                tergetpoint = new_point
                                thetalist = thetalist_new
                                nADD = nADD2
                                break
                    except Exception as e:
                        print(f"Step calculation error: {e}, trying different step")
                        continue
                        
                # If no valid step found, try a random perturbation
                if not found_valid_step:
                    print(f"No valid step found at iteration {whileN}, trying random perturbation")
                    # Generate random perturbation but keep on hypersphere
                    random_theta = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                    random_nADD = self.SuperSphere_cartesian(IOEsphereA, random_theta, self.SQ, self.dim)
                    
                    # Interpolate between current point and random point
                    alpha = 0.1  # Small mixing factor
                    mixed_theta = thetalist * (1-alpha) + random_theta * alpha
                    
                    # Generate new point
                    mixed_nADD = self.SuperSphere_cartesian(IOEsphereA, mixed_theta, self.SQ, self.dim)
                    mixed_nADD_full = np.zeros(n_coords)
                    for i in range(min(len(mixed_nADD), n_coords)):
                        mixed_nADD_full[i] = mixed_nADD[i]
                    mixed_nADD = mixed_nADD_full.reshape(n_atoms, 3)
                    
                    # Update point
                    tergetpoint = eqpoint + mixed_nADD
                    tergetpoint = self.periodicpoint(tergetpoint)
                    nADD = mixed_nADD
                    thetalist = mixed_theta
                
                # Increment counter
                whileN += 1
                
                # Print progress periodically
                if whileN % 10 == 0:
                    print(f"Optimization step {whileN}: gradient norm = {np.linalg.norm(SSgrad):.6f}")
                    
            except Exception as e:
                print(f"Error in optimization step {whileN}: {e}, continuing with best solution")
                whileN += 1
                
                # Try to recover with random perturbation
                if whileN % 3 == 0:  # Every third error, try a more drastic change
                    try:
                        # Generate a completely new point on hypersphere
                        random_theta = self.calctheta(np.random.rand(n_coords) - 0.5, self.eigVlist, self.eigNlist)
                        random_nADD = self.SuperSphere_cartesian(IOEsphereA, random_theta, self.SQ, self.dim)
                        
                        # Create full vector
                        random_nADD_full = np.zeros(n_coords)
                        for i in range(min(len(random_nADD), n_coords)):
                            random_nADD_full[i] = random_nADD[i]
                        random_nADD = random_nADD_full.reshape(n_atoms, 3)
                        
                        # Try the new point
                        new_point = eqpoint + random_nADD
                        new_point = self.periodicpoint(new_point)
                        
                        # Check if it's better
                        try:
                            random_energy = f(new_point)
                            if isinstance(random_energy, (int, float)) and not np.isnan(random_energy):
                                if random_energy < best_energy:
                                    best_energy = random_energy
                                    best_thetalist = random_theta.copy()
                                    tergetpoint = new_point
                                    nADD = random_nADD
                                    thetalist = random_theta
                        except Exception:
                            pass  # Ignore failed energy calculations
                    except Exception:
                        pass  # Ignore errors in recovery attempt
        
        print(f"Optimization completed with {whileN} iterations")
        # Return the best solution found
        return best_thetalist if best_energy < float('inf') else thetalist
    
    def detect_add(self, QMC):
        """Detect ADD directions from Hessian"""
        coord_1 = self.get_coord()
        coord_1 = self.adjust_center2origin(coord_1)
        n_atoms = coord_1.shape[0]
        n_coords = n_atoms * 3
        
        element_number_list_1 = self.get_element_number_list()
        print("### Checking whether initial structure is EQ. ###")
        
        QMC.hessian_flag = True
        self.init_energy, self.init_gradient, _, iscalculationfailed = QMC.single_point(
            None, element_number_list_1, "", self.electric_charge_and_multiplicity, 
            self.method, coord_1
        )
        
        if iscalculationfailed:
            print("Initial calculation failed.")
            return False
        
        # Apply bias potential if needed
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, bias_energy, bias_gradient, bias_hess = BPC.main(
            self.init_energy, self.init_gradient, coord_1, element_number_list_1, 
            self.config.force_data
        )
        self.init_energy = bias_energy
        self.init_gradient = bias_gradient
        
        QMC.hessian_flag = False
        self.init_geometry = coord_1  # Shape: (n_atoms, 3)
        self.set_coord(coord_1)
        
        if np.linalg.norm(self.init_gradient) > 1e-3:
            print("Norm of gradient is too large. Structure is not at equilibrium.")
            return False
                
        print("Initial structure is EQ.")
        print("### Start calculating Hessian matrix to detect ADD. ###")
        
        hessian = QMC.Model_hess + bias_hess
        
        # Project out translation and rotation
        projection_hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(hessian, self.element_list, coord_1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(projection_hessian)
        eigenvalues = eigenvalues.astype(np.float64)
        
        # Filter out near-zero eigenvalues
        nonzero_indices = np.where(np.abs(eigenvalues) > 1e-10)[0]
        nonzero_eigenvectors = eigenvectors[:, nonzero_indices].astype(np.float64)
        nonzero_eigenvalues = eigenvalues[nonzero_indices].astype(np.float64)
        
        sorted_idx = np.argsort(nonzero_eigenvalues)
        
        self.init_eigenvalues = nonzero_eigenvalues
        self.init_eigenvectors = nonzero_eigenvectors
        self.dim = len(nonzero_eigenvalues)
        self.n_atoms = n_atoms
        self.n_coords = n_coords
        
        # Store flattened versions for matrix operations
        self.init_geometry_flat = self.init_geometry.flatten()
        
        # Prepare SQ matrices
        self.SQ = self.SQaxes(nonzero_eigenvalues, nonzero_eigenvectors, self.dim)
        self.SQ_inv = self.SQaxes_inv(nonzero_eigenvalues, nonzero_eigenvectors, self.dim)
        
        self.eigNlist = nonzero_eigenvalues
        self.eigVlist = nonzero_eigenvectors
        
        # Calculate mode eigenvectors to try - focus on lower eigenvectors first
        # (corresponding to softer modes that are more likely to lead to transition states)
        search_idx = len(sorted_idx) # Start with more eigenvectors than needed
        self.sorted_eigenvalues_idx = sorted_idx[0:search_idx]
        
        # Initialize ADDths with initial directions
        self.ADDths = []
        IDnum = 0
        
        print("### Checking ADD energy. ###")
        with open(self.directory + "/add_energy_list.csv", "w") as f:
            f.write("index_of_principal_axis,eigenvalue,direction,add_energy,abs_add_energy\n")
        
        IOEsphereA = self.addf_config['IOEsphereA_initial']
        
        # Create candidate ADDs for eigenvectors
        candidate_ADDths = []
        
        for idx in self.sorted_eigenvalues_idx:
            for pm in [-1.0, 1.0]:
                eigV = self.init_eigenvectors[:, idx]
                
                # Create a new ADD point
                ADDth = type('ADDthetaClass', (), {})
                ADDth.IDnum = IDnum
                ADDth.dim = self.dim
                ADDth.SQ = self.SQ
                ADDth.SQ_inv = self.SQ_inv
                ADDth.thetalist = self.calctheta(pm * eigV, nonzero_eigenvectors, nonzero_eigenvalues)
                
                # Generate nADD (this will be a flattened vector in the Hessian eigenspace)
                ADDth.nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, self.SQ, self.dim)
                
                # We need to convert the reduced space vector back to the full coordinate space
                # First create a zero vector in the full space
                ADDth.nADD_full = np.zeros(n_coords)
                
                # Map the reduced vector to the full space (this is approximate)
                for i in range(min(len(ADDth.nADD_reduced), n_coords)):
                    ADDth.nADD_full[i] = ADDth.nADD_reduced[i]
                    
                # Reshape to molecular geometry format
                ADDth.nADD = ADDth.nADD_full.reshape(n_atoms, 3)
                
                # Apply the displacement to the initial geometry
                ADDth.x = self.init_geometry + ADDth.nADD
                ADDth.x = self.periodicpoint(ADDth.x)
                
                # Calculate energy
                energy, grad_x, _, iscalculationfailed = QMC.single_point(
                    None, element_number_list_1, "", self.electric_charge_and_multiplicity, 
                    self.method, ADDth.x
                )
                
                if iscalculationfailed:
                    continue
                    
                # Apply bias if needed
                _, bias_energy, bias_gradient, _ = BPC.main(
                    energy, grad_x, ADDth.x, element_number_list_1, 
                    self.config.force_data
                )
                
                ADDth.A = bias_energy
                ADDth.ADD = ADDth.A - IOEsphereA - self.init_energy
                ADDth.ADD_IOE = ADDth.ADD  # Initial value, will be updated later
                ADDth.grad = bias_gradient
                ADDth.grad_vec = np.dot(bias_gradient.flatten(), ADDth.nADD.flatten())
                ADDth.grad_vec /= np.linalg.norm(ADDth.nADD.flatten())
                ADDth.findTSQ = False
                ADDth.ADDoptQ = False
                ADDth.ADDremoveQ = False
                
                # Add to candidate list
                candidate_ADDths.append(ADDth)
                IDnum += 1
                
                with open(self.directory + "/add_energy_list.csv", "a") as f:
                    f.write(f"{idx},{self.init_eigenvalues[idx]},{pm},{ADDth.ADD},{abs(ADDth.ADD)}\n")
        
        # Sort candidate ADDths by negative ADD value in descending order (-ADD value)
        # This prioritizes more negative (favorable) paths first
        candidate_ADDths.sort(key=lambda x: -x.ADD, reverse=True)
        
        # Select only the top n ADDs according to config.nadd
        num_add = min(self.addf_config['number_of_add'], len(candidate_ADDths))
        self.ADDths = candidate_ADDths[:num_add]
        
        # Reassign IDs to be sequential
        for i, ADDth in enumerate(self.ADDths):
            ADDth.IDnum = i
        
        print(f"### Selected top {len(self.ADDths)} ADD paths (sign-inverted ADD values, most negative first) ###")
        for ADDth in self.ADDths:
            print(f"ADD {ADDth.IDnum}: {ADDth.ADD:.8f} (-ADD = {-ADDth.ADD:.8f})")
        
        # Initialize the optimized structures dictionary
        self.optimized_structures = {}
        
        # Create directory for optimized structures
        os.makedirs(os.path.join(self.directory, "optimized_structures"), exist_ok=True)
        
        print("### ADD detection complete. ###")
        return True

    def save_optimized_structure(self, ADDth, sphere_num, IOEsphereA):
        """Save optimized structure for a specific ADD and sphere"""
        # Create directory path for ADD-specific structures
        add_dir = os.path.join(self.directory, "optimized_structures", f"ADD_{ADDth.IDnum}")
        os.makedirs(add_dir, exist_ok=True)
        
        # Create filename with radius information
        radius = np.sqrt(IOEsphereA)
        filename = f"optimized_r_{radius:.4f}.xyz"
        filepath = os.path.join(add_dir, filename)
        
        # Write XYZ file
        with open(filepath, 'w') as f:
            f.write(f"{len(self.element_list)}\n")
            f.write(f"ADD_{ADDth.IDnum} Sphere {sphere_num} Radius {radius:.4f} Energy {ADDth.ADD:.6f}\n")
            for i, (element, coord) in enumerate(zip(self.element_list, ADDth.x)):
                f.write(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")

        # Store structure information in dictionary
        if ADDth.IDnum not in self.optimized_structures:
            self.optimized_structures[ADDth.IDnum] = []
            
        self.optimized_structures[ADDth.IDnum].append({
            'sphere': sphere_num,
            'radius': radius,
            'energy': ADDth.ADD,
            'file': filepath,
            'coords': ADDth.x.copy(),
            'comment': f"ADD_{ADDth.IDnum} Sphere {sphere_num} Radius {radius:.4f} Energy {ADDth.ADD:.6f}"
        })

    def create_separate_xyz_files(self):
        """Create separate XYZ files for each ADD path"""
        created_files = []
        
        for add_id, structures in self.optimized_structures.items():
            # Sort structures by sphere number
            structures.sort(key=lambda x: x['sphere'])
            
            # Path for the ADD-specific file
            add_trajectory_file = os.path.join(self.directory, f"ADD_{add_id}_trajectory.xyz")
            
            # Write the trajectory file
            with open(add_trajectory_file, 'w') as f:
                for structure in structures:
                    f.write(f"{len(self.element_list)}\n")
                    f.write(f"{structure['comment']}\n")
                    for i, (element, coord) in enumerate(zip(self.element_list, structure['coords'])):
                        f.write(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
            
            created_files.append(add_trajectory_file)
            
        if created_files:
            paths_str = "\n".join(created_files)
            print(f"Created {len(created_files)} ADD trajectory files:\n{paths_str}")

    def Opt_hyper_sphere(self, ADDths, QMC, eqpoint, IOEsphereA, IOEsphereA_r, A_eq, sphereN):
        """Optimize points on the hypersphere"""
        print(f"Starting optimization on sphere {sphereN} with radius {np.sqrt(IOEsphereA):.4f}")
        
        # Reset optimization flags
        for ADDth in ADDths:
            if not ADDth.findTSQ and not ADDth.ADDremoveQ:
                ADDth.ADDoptQ = True
            else:
                ADDth.ADDoptQ = False
                
        optturnN = 0
        newADDths = []
        n_atoms = eqpoint.shape[0]
        
        # Create a directory for intermediate optimization steps
        sphere_dir = os.path.join(self.directory, "optimized_structures", f"sphere_{sphereN}")
        os.makedirs(sphere_dir, exist_ok=True)
        
        # Optimization loop
        while any(ADDth.ADDoptQ for ADDth in ADDths):
            optturnN += 1
            if optturnN >= 100:
                print(f"Optimization exceeded 100 iterations, breaking.")
                break
                
            print(f"Optimization iteration {optturnN}")
            
            # Process each ADD point in order of negative ADD value (most negative first)
            for ADDth in sorted(ADDths, key=lambda x: -x.ADD, reverse=True):
                if not ADDth.ADDoptQ or ADDth.ADDremoveQ:
                    continue
                    
                # Optimize this ADD point
                self.current_id = ADDth.IDnum
                self.current_ADD = ADDth.ADD
                
                # Starting from zero displacement
                x_initial = np.zeros(len(ADDth.thetalist))
                
                # Minimize on hypersphere using our modified steepest descent
                thetalist = self.minimizeTh_SD_SS(
                    ADDth, x_initial, 
                    lambda x: QMC.single_point(None, self.get_element_number_list(), "", 
                                            self.electric_charge_and_multiplicity, self.method, x)[0],
                    lambda x: self.calculate_gradient(QMC, x),
                    eqpoint, IOEsphereA
                )
                
                if thetalist is False:
                    ADDth.ADDremoveQ = True
                    print(f"ADD {ADDth.IDnum} optimization failed, marking for removal")
                    continue
                
                # Update ADD point with optimized position
                ADDth.thetalist = thetalist
                
                # Generate nADD in the reduced space
                ADDth.nADD_reduced = self.SuperSphere_cartesian(IOEsphereA, ADDth.thetalist, self.SQ, self.dim)
                
                # Map to full coordinate space
                ADDth.nADD_full = np.zeros(n_atoms * 3)
                for i in range(min(len(ADDth.nADD_reduced), n_atoms * 3)):
                    ADDth.nADD_full[i] = ADDth.nADD_reduced[i]
                
                # Reshape to molecular geometry
                ADDth.nADD = ADDth.nADD_full.reshape(n_atoms, 3)
                
                # Calculate new coordinates
                ADDth.x = eqpoint + ADDth.nADD
                ADDth.x = self.periodicpoint(ADDth.x)
                
                # Calculate new energy
                energy, grad, _, iscalculationfailed = QMC.single_point(
                    None, self.get_element_number_list(), "", 
                    self.electric_charge_and_multiplicity, self.method, ADDth.x
                )
                
                if iscalculationfailed:
                    ADDth.ADDremoveQ = True
                    continue
                    
                BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
                _, bias_energy, bias_grad, _ = BPC.main(
                    energy, grad, ADDth.x, self.get_element_number_list(), 
                    self.config.force_data
                )
                
                ADDth.A = bias_energy
                ADDth.ADD = ADDth.A - IOEsphereA - A_eq
                
                # Calculate ADD_IOE with IOE contributions
                self.current_id = ADDth.IDnum
                self.current_ADD = ADDth.ADD
                ADDth.ADD_IOE = ADDth.ADD + self.IOE_total(ADDth.nADD_full)
                
                # Mark as optimized
                ADDth.ADDoptQ = False
                
                print(f"ADD {ADDth.IDnum} optimized: ADD={ADDth.ADD:.4f} (-ADD = {-ADDth.ADD:.4f}), ADD_IOE={ADDth.ADD_IOE:.4f}")
                print(f"Grad {np.linalg.norm(grad):.6f}")
                print(f"Energy {energy:.6f}")
                print()
                
                # Save XYZ file after each ADD optimization step
                # Use both iteration number and ADD ID in filename to ensure uniqueness
                filename = f"iteration_{optturnN}_ADD_{ADDth.IDnum}.xyz"
                filepath = os.path.join(sphere_dir, filename)
                
                # Write XYZ file for this optimization step
                with open(filepath, 'w') as f:
                    f.write(f"{len(self.element_list)}\n")
                    f.write(f"Sphere {sphereN} Iteration {optturnN} ADD_{ADDth.IDnum} Radius {np.sqrt(IOEsphereA):.4f} Energy {ADDth.ADD:.6f}\n")
                    for i, (element, coord) in enumerate(zip(self.element_list, ADDth.x)):
                        f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
                
                print(f"Saved intermediate structure to {filepath}")
                
                # Check for similar points already found
                for existing_add in newADDths:
                    if self.angle_SHS(ADDth.nADD_full, existing_add.nADD_full, self.SQ_inv) < 0.01:
                        ADDth.ADDremoveQ = True
                        print(f"ADD {ADDth.IDnum} too similar to ADD {existing_add.IDnum}, marking for removal")
                        break
                
                if not ADDth.ADDremoveQ and ADDth.ADD_IOE < 0:  # Only keep negative ADD_IOE points
                    newADDths.append(ADDth)
                    
                    # Save optimized structure for this sphere (in the regular directory structure)
                    self.save_optimized_structure(ADDth, sphereN, IOEsphereA)
            
            # Filter ADDths list
            ADDths = [ADDth for ADDth in ADDths if not ADDth.ADDremoveQ]
            
            # Sort by negative ADD value (most negative first) for next iteration
            ADDths.sort(key=lambda x: -x.ADD, reverse=True)
            
        # Sort the final ADDths list by negative ADD value (most negative first)
        newADDths.sort(key=lambda x: -x.ADD, reverse=True)
        
        return newADDths if newADDths else ADDths
    
    def add_following(self, QMC):
        """Follow ADD paths to find transition states"""
        print("### Start ADD Following. ###")
        
        IOEsphereA = self.addf_config['IOEsphereA_initial']
        IOEsphereA_r = self.addf_config['IOEsphereA_dist']
        A_eq = self.init_energy
        
        TSinitialpoints = []
        sphereN = 0
        
        # Main ADD following loop
        while sphereN < self.addf_config["step_number"]:  # Limit to prevent infinite loops
            sphereN += 1
            print(f"\n### Sphere {sphereN} with radius {np.sqrt(IOEsphereA):.4f} ###\n")
            
            # Sort ADDths by absolute ADD value (largest magnitude first)
            self.ADDths.sort(key=lambda x: abs(x.ADD), reverse=True)
            
            # Optimize on current hypersphere
            self.ADDths = self.Opt_hyper_sphere(
                self.ADDths, QMC, self.init_geometry, IOEsphereA, IOEsphereA_r, A_eq, sphereN
            )
            
            # Check for TS points and update ADD status
            for ADDth in self.ADDths:
                if ADDth.ADDremoveQ:
                    continue
                    
                # Calculate gradient at current point
                grad_x = self.calculate_gradient(QMC, ADDth.x)
                if grad_x is False:
                    ADDth.ADDremoveQ = True
                    continue
                    
                ADDth.grad = grad_x
                
                # Calculate normalized displacement vector
                normalized_nADD = ADDth.nADD.flatten() / np.linalg.norm(ADDth.nADD.flatten())
                
                # Calculate projection of gradient onto displacement vector
                ADDth.grad_vec = np.dot(ADDth.grad.flatten(), normalized_nADD)
                
                # Check if we've found a TS (gradient points downward)
                if sphereN > 5 and ADDth.grad_vec < 0.0:
                    print(f"New TS point found at ADD {ADDth.IDnum}")
                    ADDth.findTSQ = True
                    TSinitialpoints.append(ADDth.x)
                    
            # If all ADDs are done, exit
            if all(ADDth.findTSQ or ADDth.ADDremoveQ for ADDth in self.ADDths):
                print("All ADD paths complete.")
                break
                
            # Increase sphere size for next iteration
            IOEsphereA = (np.sqrt(IOEsphereA) + IOEsphereA_r) ** 2
            print(f"Expanding sphere to radius {np.sqrt(IOEsphereA):.4f}")
            
            # Save displacement vectors for debugging
            with open(os.path.join(self.directory, f"displacement_vectors_sphere_{sphereN}.csv"), "w") as f:
                f.write("ADD_ID,x,y,z,ADD,ADD_IOE\n")
                for ADDth in self.ADDths:
                    if ADDth.ADDremoveQ:
                        continue
                    for i in range(len(ADDth.nADD)):
                        f.write(f"{ADDth.IDnum},{ADDth.nADD[i][0]},{ADDth.nADD[i][1]},{ADDth.nADD[i][2]},{ADDth.ADD},{ADDth.ADD_IOE}\n")
        
        # Create separate trajectory files for each ADD path
        self.create_separate_xyz_files()
        
        # Write TS points
        if TSinitialpoints:
            print(f"Found {len(TSinitialpoints)} potential transition states.")
            self.write_ts_points(TSinitialpoints)
            
        return len(TSinitialpoints) > 0
    
    def calculate_gradient(self, QMC, x):
        """Calculate gradient at point x"""
        element_number_list = self.get_element_number_list()
        _, grad_x, _, iscalculationfailed = QMC.single_point(
            None, element_number_list, "", self.electric_charge_and_multiplicity, 
            self.method, x
        )
        
        if iscalculationfailed:
            return False
            
        # Apply bias if needed
        BPC = BiasPotentialCalculation(self.config.iEIP_FOLDER_DIRECTORY)
        _, _, bias_gradient, _ = BPC.main(
            0, grad_x, x, element_number_list, self.config.force_data
        )
        
        return bias_gradient
        
    def write_ts_points(self, ts_points):
        """Write TS points to file"""
        with open(f"{self.directory}/TSpoints.xyz", "w") as f:
            for i, point in enumerate(ts_points):
                f.write(f"{len(self.element_list)}\n")
                f.write(f"TS candidate {i+1}\n")
                for j, (element, coord) in enumerate(zip(self.element_list, point)):
                    f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    # Getters and setters - keep as is
    def set_molecule(self, element_list, coords):
        self.element_list = element_list
        self.coords = coords
    
    def set_gradient(self, gradient):
        self.gradient = gradient
    
    def set_hessian(self, hessian):
        self.hessian = hessian
    
    def set_energy(self, energy):
        self.energy = energy
    
    def set_coords(self, coords):
        self.coords = coords
    
    def set_element_list(self, element_list):
        self.element_list = element_list
        self.element_number_list = [element_number(i) for i in self.element_list]
    
    def set_coord(self, coord):
        self.coords = coord
    
    def get_coord(self):
        return self.coords
    
    def get_element_list(self):
        return self.element_list
    
    def get_element_number_list(self):
        if self.element_number_list is None:
            if self.element_list is None:
                raise ValueError('Element list is not set.')
            self.element_number_list = [element_number(i) for i in self.element_list]
        return self.element_number_list
    
    def set_mole_info(self, base_file_name, electric_charge_and_multiplicity):
        coord, element_list, electric_charge_and_multiplicity = xyz2list(
            base_file_name + ".xyz", electric_charge_and_multiplicity)

        if self.config.usextb != "None":
            self.method = self.config.usextb
        elif self.config.usedxtb != "None":
            self.method = self.config.usedxtb
        else:
            self.method = "None"

        self.coords = np.array(coord, dtype="float64")  
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
    
    def run(self, file_directory, SP, electric_charge_and_multiplicity, FIO_img):
        print("### Start Anharmonic Downward Distortion (ADD) method ###")
        
        # Preparation
        base_file_name = os.path.splitext(FIO_img.START_FILE)[0]
        self.set_mole_info(base_file_name, electric_charge_and_multiplicity)
        
        self.directory = make_workspace(file_directory)
        
        # Create main directory for optimized structures
        os.makedirs(os.path.join(self.directory, "optimized_structures"), exist_ok=True)
        
        # Detect initial ADD directions
        success = self.detect_add(SP)
        if not success:
            return False
        
        # Follow ADD paths to find transition states
        isConverged = self.add_following(SP)
        
        print("### ADD Following is done. ###")
        
        return isConverged
