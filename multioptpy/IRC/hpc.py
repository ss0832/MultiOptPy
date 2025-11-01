import numpy as np
import os
import copy
import csv

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Optimizer.hessian_update import ModelHessianUpdate
from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass
from multioptpy.IRC.converge_criteria import convergence_check
from multioptpy.Visualization.visualization import Graph
from multioptpy.PESAnalyzer.calc_irc_curvature import calc_irc_curvature_properties, save_curvature_properties_to_file

# --- Helper Class for HPC (DWI) ---

class DWISurface:
    """
    Distance Weighted Interpolant (DWI) surface using data from two points
    (position, energy, gradient, Hessian) [cite_start][cite: 79-80].
    This surface can calculate the energy and gradient at any arbitrary point x.
    
    Ref: J. Chem. Phys. 120, 9918 (2004), Sec II. [cite_start]D [cite: 239-251]
    """
    def __init__(self, x1, e1, g1, h1, x2, e2, g2, h2):
        self.natoms = x1.shape[0]
        self.dim = self.natoms * 3
        
        # Store data as flat (3N,) arrays
        self.x = [x1.flatten(), x2.flatten()]
        self.e = [e1, e2]
        self.g = [g1.flatten(), g2.flatten()]
        self.h = [h1, h2] # (3N, 3N)

    def get_taylor(self, i, x_flat):
        """Calculates the Taylor expansion T_i(x) using data from point i"""
        dx = x_flat - self.x[i]
        e_taylor = self.e[i] \
                   + np.dot(self.g[i].T, dx) \
                   + 0.5 * np.dot(dx.T, np.dot(self.h[i], dx))
        return e_taylor

    def get_taylor_grad(self, i, x_flat):
        """Calculates the gradient ∇T_i(x) of the Taylor expansion T_i(x)"""
        dx = x_flat - self.x[i]
        # ∇T_i = g_i + H_i * (x - x_i)
        g_taylor = self.g[i] + np.dot(self.h[i], dx)
        return g_taylor

    def get_weights(self, x_flat):
        """Calculate weights w1, w2 at point x [cite: 247-248]"""
        dx1 = x_flat - self.x[0]
        dx2 = x_flat - self.x[1]
        
        norm_sq_1 = np.dot(dx1.T, dx1)
        norm_sq_2 = np.dot(dx2.T, dx2)
        
        denom = norm_sq_1 + norm_sq_2
        
        if denom < 1e-12: # If points are very close (or identical)
            return 0.5, 0.5
            
        w1 = norm_sq_2 / denom
        w2 = norm_sq_1 / denom
        return w1, w2

    def get_weight_grads(self, x_flat):
        """Calculate gradients of the weights ∇w1, ∇w2 at point x"""
        dx1 = x_flat - self.x[0]
        dx2 = x_flat - self.x[1]
        
        n1 = np.dot(dx1.T, dx1) # |Δx1|^2
        n2 = np.dot(dx2.T, dx2) # |Δx2|^2
        d = n1 + n2
        
        if d < 1e-12:
            return np.zeros(self.dim), np.zeros(self.dim)
            
        # ∇n1 = 2 * Δx1
        # ∇n2 = 2 * Δx2
        # ∇d = 2 * (Δx1 + Δx2)
        grad_n1 = 2 * dx1
        grad_n2 = 2 * dx2
        grad_d = grad_n1 + grad_n2
        
        # ∇w1 = ∇(n2 / d) = ( (∇n2) * d - n2 * (∇d) ) / d^2
        grad_w1 = (grad_n2 * d - n2 * grad_d) / (d**2)
        
        # ∇w2 = ∇(n1 / d) = ( (∇n1) * d - n1 * (∇d) ) / d^2
        grad_w2 = (grad_n1 * d - n1 * grad_d) / (d**2)
        
        return grad_w1, grad_w2

    def get_energy(self, x_flat):
        """Calculate the energy E_DWI(x) on the DWI surface"""
        w1, w2 = self.get_weights(x_flat)
        t1 = self.get_taylor(0, x_flat)
        t2 = self.get_taylor(1, x_flat)
        return w1 * t1 + w2 * t2

    def get_gradient(self, x_flat):
        """Calculate the gradient ∇E_DWI(x) on the DWI surface"""
        w1, w2 = self.get_weights(x_flat)
        gw1, gw2 = self.get_weight_grads(x_flat)
        
        t1 = self.get_taylor(0, x_flat)
        t2 = self.get_taylor(1, x_flat)
        
        gt1 = self.get_taylor_grad(0, x_flat)
        gt2 = self.get_taylor_grad(1, x_flat)
        
        # ∇(w1*T1 + w2*T2) = (∇w1)T1 + w1(∇T1) + (∇w2)T2 + w2(∇T2)
        grad = (gw1 * t1) + (w1 * gt1) + (gw2 * t2) + (w2 * gt2)
        return grad.reshape(self.natoms, 3)

# --- Corrector Step for HPC ---

def corrector_step(dwi_surface, x_start, total_s, n_steps=100):
    """
    Finds the corrected point x_corr by integrating on the DWI surface
    using Euler's method.
    dx/ds = -g / |g|
    """
    h = total_s / n_steps # Arc length per step
    x = x_start
    
    for _ in range(n_steps):
        g_flat = dwi_surface.get_gradient(x.flatten())
        g = g_flat.flatten()
        norm_g = np.linalg.norm(g)
        
        if norm_g < 1e-9: # Reached a minimum
            break
            
        # Euler step
        step_vec = - (g / norm_g) * h
        x = x + step_vec.reshape(dwi_surface.natoms, 3)
        
    return x

# --- Main HPC Class ---

class HPC:
    """Hessian-based Predictor-Corrector (HPC) method for IRC calculations
    
    This class implements the HPC algorithm which uses
    the LQA method as the predictor and a 
    DWI-based corrector.
    
    References
    ----------
    [1] J. Chem. Phys. 93, 5634–5642 (1990) (LQA)
    [2] J. Chem. Phys. 120, 9918–9924 (2004) (HPC)
    """
    
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, 
                 final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, 
                 init_hess=None, calc_engine=None, xtb_method=None, **kwargs):
        """Initialize HPC IRC calculator"""
        self.max_step = max_step
        self.step_size = step_size
        self.N_euler = 20000  # Number of Euler integration steps for LQA predictor
        self.N_corrector = 100 # Number of Euler integration steps for DWI corrector
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        
        # initial condition
        self.coords = init_coord
        self.init_hess = init_hess # This is non-mass-weighted
        self.mw_hessian = None     # This is mass-weighted
        self.xtb_method = xtb_method
        
        # convergence criteria
        self.MAX_FORCE_THRESHOLD = 0.0004
        self.RMS_FORCE_THRESHOLD = 0.0001

        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # Previous step's data for HPC (non-mass-weighted, bias-inclusive)
        self.prev_data = {
            'coords': None,
            'energy': None,
            'gradient': None,
            'hessian': None # (3N, 3N)
        }
        
        # Store only necessary data
        self.irc_bias_energy_list = []
        self.irc_energy_list = []
        self.irc_mw_coords = []
        self.irc_mw_gradients = []
        self.path_bending_angle_list = []
        
        # Create data files
        self.create_csv_file()
        self.create_xyz_file()
    
    def create_csv_file(self):
        """Create CSV file for energy and gradient data"""
        self.csv_filename = os.path.join(self.directory, "irc_energies_gradients.csv")
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Energy (Hartree)', 'Bias Energy (Hartree)', 'RMS Gradient', 'RMS Bias Gradient'])
    
    def create_xyz_file(self):
        """Create XYZ file for structure data"""
        self.xyz_filename = os.path.join(self.directory, "irc_structures.xyz")
        # Create empty file (will be appended to later)
        open(self.xyz_filename, 'w').close()
    
    def save_to_csv(self, step, energy, bias_energy, gradient, bias_gradient):
        """Save energy and gradient data to CSV file"""
        rms_grad = np.sqrt((gradient**2).mean())
        rms_bias_grad = np.sqrt((bias_gradient**2).mean())
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, energy, bias_energy, rms_grad, rms_bias_grad])
    
    def save_xyz_structure(self, step, coords):
        """Save molecular structure to XYZ file"""
        coords_angstrom = coords * UnitValueLib().bohr2angstroms
        
        with open(self.xyz_filename, 'a') as f:
            f.write(f"{len(coords)}\n")
            f.write(f"IRC Step {step}\n")
            for i, coord in enumerate(coords_angstrom):
                f.write(f"{self.element_list[i]:<3} {coord[0]:15.10f} {coord[1]:15.10f} {coord[2]:15.10f}\n")
    
    def get_mass_array(self):
        """Create arrays of atomic masses for mass-weighting operations"""
        elem_mass_list = np.array([atomic_mass(elem) for elem in self.element_list], dtype="float64")
        sqrt_mass_list = np.sqrt(elem_mass_list)
        
        three_elem_mass_list = np.repeat(elem_mass_list, 3)
        three_sqrt_mass_list = np.repeat(sqrt_mass_list, 3)
        
        return elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list
    
    def mass_weight_hessian(self, hessian, three_sqrt_mass_list):
        """Apply mass-weighting to the hessian matrix"""
        mass_mat = np.diag(1.0 / three_sqrt_mass_list)
        return np.dot(mass_mat, np.dot(hessian, mass_mat))
    
    def mass_weight_coordinates(self, coordinates, sqrt_mass_list):
        """Convert coordinates to mass-weighted coordinates"""
        mw_coords = copy.deepcopy(coordinates)
        for i in range(len(coordinates)):
            mw_coords[i] = coordinates[i] * sqrt_mass_list[i]
        return mw_coords
    
    def mass_weight_gradient(self, gradient, sqrt_mass_list):
        """Convert gradient to mass-weighted form"""
        mw_gradient = copy.deepcopy(gradient)
        for i in range(len(gradient)):
            mw_gradient[i] = gradient[i] / sqrt_mass_list[i]
        return mw_gradient
    
    def unmass_weight_step(self, step, sqrt_mass_list):
        """Convert a step vector from mass-weighted to non-mass-weighted coordinates"""
        unmw_step = copy.deepcopy(step)
        for i in range(len(step)):
            unmw_step[i] = step[i] / sqrt_mass_list[i]
        return unmw_step
    
    def check_energy_oscillation(self, energy_list):
        """Check if energy is oscillating (going up and down)"""
        if len(energy_list) < 3:
            return False
        last_diff = energy_list[-1] - energy_list[-2]
        prev_diff = energy_list[-2] - energy_list[-3]
        return (last_diff * prev_diff) < 0
        
    def step(self, mw_gradient, geom_num_list, mw_combined_hessian, sqrt_mass_list):
        """
        Calculate a single LQA predictor step.
        
        Note: This function receives the *combined* (model + bias)
        mass-weighted Hessian.
        """
        
        # BFGS update (Hessian is updated in the run loop *before* calling this)
        # self.mw_hessian is now the updated, combined, mass-weighted Hessian
        
        eigenvalues, eigenvectors = np.linalg.eigh(mw_combined_hessian)
        
        # Drop small eigenvalues
        small_eigvals = np.abs(eigenvalues) < 1e-8
        eigenvalues = eigenvalues[~small_eigvals]
        eigenvectors = eigenvectors[:,~small_eigvals]
        
        flattened_gradient = mw_gradient.flatten()
        
        # --- MODIFICATION (Fix for numerical stability) ---
        epsilon = 1e-6  # Prevent divergence when gradient norm is near zero
        norm_g = np.linalg.norm(flattened_gradient)
        dt = 1 / self.N_euler * self.step_size / max(norm_g, epsilon)
        # --- END MODIFICATION ---

        mw_gradient_proj = np.dot(eigenvectors.T, flattened_gradient)

        # Integration of the step size
        t = dt
        current_length = 0
        for j in range(self.N_euler):
            dsdt = np.sqrt(np.sum(mw_gradient_proj**2 * np.exp(-2*eigenvalues*t)))
            current_length += dsdt * dt
            if current_length > self.step_size:
                break
            t += dt
        
        # --- MODIFICATION (Fix for numerical stability) ---
        x = -eigenvalues * t
        small_x_mask = np.abs(x) < 1e-8
        alphas = np.where(
            small_x_mask,
            -t,
            np.expm1(x) / eigenvalues # Numerically stable (exp(x)-1)/eig
        )
        # --- END MODIFICATION ---
        
        A = np.dot(eigenvectors, np.dot(np.diag(alphas), eigenvectors.T))
        step = np.dot(A, flattened_gradient)
        
        step = step.reshape(len(geom_num_list), 3)
        step = self.unmass_weight_step(step, sqrt_mass_list)
        
        new_geom = geom_num_list + step
        new_geom -= Calculationtools().calc_center_of_mass(new_geom, self.element_list)
        
        return new_geom
        
    def run(self):
        """Run the HPC IRC calculation"""
        print("Hessian-based Predictor-Corrector (HPC) method")
        geom_num_list = self.coords # This is x_k_corr from previous step
        CalcBiaspot = BiasPotentialCalculation(self.directory)
        
        oscillation_counter = 0
        
        # Get mass arrays for mass-weighting
        elem_mass_list, sqrt_mass_list, three_elem_mass_list, three_sqrt_mass_list = self.get_mass_array()

        # --- HPC ALGORITHM FLOW ---
        # At start of loop 'iter', we are at point x_{k-1} (corrected)
        # 1. (LQA Predictor) Use data(k-1) to predict x_k_pred
        # 2. (Ab Initio) Calculate E, g, H at x_k_pred
        # 3. (DWI Surface) Build DWI surface using data(k-1) and data(k_pred)
        # 4. (Corrector) Integrate on DWI from x_{k-1} to get x_k_corr
        # 5. (Store) Save x_k_corr data for next iteration
        
        # --- Initialization (iter = 0) ---
        # Perform calculation at the starting point (x_0)
        print("# STEP: 0 (Initialization)")
        e, g, geom_num_list, finish_frag = self.CE.single_point(
            self.final_directory, self.element_list, 0, 
            self.electric_charge_and_multiplicity, self.xtb_method,  
            UnitValueLib().bohr2angstroms * geom_num_list
        )
        if finish_frag:
            print("Initial calculation failed.")
            return

        _, B_e, B_g, BPA_hessian = CalcBiaspot.main(
            e, g, geom_num_list, self.element_list, 
            self.force_data, g, 0, geom_num_list
        )
        
        # Get initial model Hessian
        model_hessian = self.init_hess # Assumes init_hess is provided
        model_hessian = Calculationtools().project_out_hess_tr_and_rot(
            model_hessian, self.element_list, geom_num_list
        )
        
        # Store initial (k=0) data
        self.prev_data = {
            'coords': copy.deepcopy(geom_num_list),
            'energy': B_e,
            'gradient': copy.deepcopy(B_g),
            'hessian': model_hessian + BPA_hessian, # non-MW, bias-inclusive
            'bpa_hessian': BPA_hessian
        }
        
        # Save initial data to files
        self.save_xyz_structure(0, geom_num_list)
        self.save_to_csv(0, e, B_e, g, B_g)
        
        # Store for BFGS/oscillation
        self.irc_energy_list.append(e)
        self.irc_bias_energy_list.append(B_e)
        self.irc_mw_coords.append(self.mass_weight_coordinates(geom_num_list, sqrt_mass_list))
        self.irc_mw_gradients.append(self.mass_weight_gradient(B_g, sqrt_mass_list))

        # --- Main Integration Loop (iter = 1 to max_step) ---
        for iter in range(1, self.max_step):
            print(f"# STEP: {iter}")
            exit_file_detect = os.path.exists(self.directory+"end.txt")
            if exit_file_detect:
                break
            
            # --- 1. Predictor Step ---
            # Get data from previous step (k-1) (corrected)
            x_km1 = self.prev_data['coords']
            e_km1 = self.prev_data['energy']
            g_km1 = self.prev_data['gradient'] # non-MW, bias-inclusive
            h_km1 = self.prev_data['hessian']  # non-MW, bias-inclusive
            
            # Mass-weight and apply BFGS update
            mw_g_km1 = self.mass_weight_gradient(g_km1, sqrt_mass_list)
            mw_h_km1 = self.mass_weight_hessian(h_km1, three_sqrt_mass_list)
            self.mw_hessian = mw_h_km1 # Start with last step's Hessian
            
            if len(self.irc_mw_gradients) > 1:
                delta_g = (self.irc_mw_gradients[-1] - self.irc_mw_gradients[-2]).reshape(-1, 1)
                delta_x = (self.irc_mw_coords[-1] - self.irc_mw_coords[-2]).reshape(-1, 1)
                if np.dot(delta_x.T, delta_g)[0, 0] > 1e-10:
                    delta_hess = self.ModelHessianUpdate.BFGS_hessian_update(self.mw_hessian, delta_x, delta_g)
                    self.mw_hessian += delta_hess
            
            # Predict x_k_pred from x_km1 using LQA
            x_k_pred = self.step(mw_g_km1, x_km1, self.mw_hessian, sqrt_mass_list)
            
            # --- 2. Ab Initio Calculation at x_k_pred ---
            e, g, x_k_pred_geom, finish_frag = self.CE.single_point(
                self.final_directory, self.element_list, iter, 
                self.electric_charge_and_multiplicity, self.xtb_method,  
                UnitValueLib().bohr2angstroms * x_k_pred
            )
            if finish_frag: break
            
            # Bias calculation at x_k_pred
            _, e_k_pred, g_k_pred, h_bpa_k_pred = CalcBiaspot.main(
                e, g, x_k_pred_geom, self.element_list, 
                self.force_data, g, iter, x_k_pred_geom
            )
            
            # Model Hessian calculation at x_k_pred
            if iter % self.FC_count == 0:
                model_h_k_pred = self.CE.Model_hess
            else:
                # Re-use the BFGS-updated Hessian (approximate)
                # Convert mass-weighted back to non-mass-weighted
                inv_mass_mat = np.diag(three_sqrt_mass_list)
                combined_h_non_mw = np.dot(inv_mass_mat, np.dot(self.mw_hessian, inv_mass_mat))
                model_h_k_pred = combined_h_non_mw - self.prev_data['bpa_hessian']

            model_h_k_pred = Calculationtools().project_out_hess_tr_and_rot(
                model_h_k_pred, self.element_list, x_k_pred_geom
            )
            h_k_pred = model_h_k_pred + h_bpa_k_pred # non-MW, bias-inclusive
            
            # --- 3. Build DWI Surface ---
            dwi = DWISurface(
                x_km1, e_km1, g_km1, h_km1,               # Point k-1
                x_k_pred_geom, e_k_pred, g_k_pred, h_k_pred # Point k (predicted)
            )

            # --- 4. Corrector Step ---
            # Integrate on the DWI surface starting from x_km1
            x_k_corr = corrector_step(dwi, x_km1, self.step_size, self.N_corrector)
            
            # --- 5. Store and Save ---
            geom_num_list = x_k_corr # This is the new starting point
            self.save_xyz_structure(iter, x_k_corr) # Save the corrected structure
            
            # Save the *ab initio* values from the *predicted* point to CSV
            self.save_to_csv(iter, e, e_k_pred, g, g_k_pred) 

            # --- 6. Prepare for next step (k+1) ---
            # Get the energy and gradient at the corrected point from the DWI surface
            e_k_corr = dwi.get_energy(x_k_corr.flatten())
            g_k_corr = dwi.get_gradient(x_k_corr.flatten())
            
            # [cite_start]Per HPC paper, use the Hessian from the predicted end point [cite: 81]
            h_k_corr = h_k_pred 
            
            self.prev_data = {
                'coords': x_k_corr,
                'energy': e_k_corr,
                'gradient': g_k_corr,
                'hessian': h_k_corr,
                'bpa_hessian': h_bpa_k_pred # Save for next BFGS approx.
            }
            
            # --- Update Data History (for BFGS and oscillation) ---
            if len(self.irc_energy_list) >= 3:
                self.irc_energy_list.pop(0)
                self.irc_bias_energy_list.pop(0)
                self.irc_mw_coords.pop(0)
                self.irc_mw_gradients.pop(0)
            
            self.irc_energy_list.append(e) # ab initio E
            self.irc_bias_energy_list.append(e_k_corr) # Corrected bias E
            self.irc_mw_coords.append(self.mass_weight_coordinates(x_k_corr, sqrt_mass_list))
            self.irc_mw_gradients.append(self.mass_weight_gradient(g_k_corr, sqrt_mass_list))

            # --- Checks ---
            if self.check_energy_oscillation(self.irc_bias_energy_list):
                oscillation_counter += 1
                print(f"Energy oscillation detected ({oscillation_counter}/5)")
                if oscillation_counter >= 5:
                    print("Terminating IRC: Energy oscillated for 5 consecutive steps")
                    break
            else:
                oscillation_counter = 0

            # Check convergence using the corrected gradient (DWI gradient)
            if convergence_check(g_k_corr, self.MAX_FORCE_THRESHOLD, self.RMS_FORCE_THRESHOLD) and iter > 10:
                print("Convergence reached. (HPC-IRC)")
                break

            # --- Common Output ---
            
            # Calculate path bending angle
            if iter > 1: # Needs at least 3 points
                bend_angle = Calculationtools().calc_multi_dim_vec_angle(
                    self.irc_mw_coords[0]-self.irc_mw_coords[1], 
                    self.irc_mw_coords[2]-self.irc_mw_coords[1]
                )
                self.path_bending_angle_list.append(np.degrees(bend_angle))
                print(f"Path bending angle: {np.degrees(bend_angle):.4f}")
            
            # Print current (corrected) geometry
            print()
            for i in range(len(geom_num_list)):    
                x = geom_num_list[i][0] * UnitValueLib().bohr2angstroms
                y = geom_num_list[i][1] * UnitValueLib().bohr2angstroms
                z = geom_num_list[i][2] * UnitValueLib().bohr2angstroms
                print(f"{self.element_list[i]:<3} {x:17.12f} {y:17.12f} {z:17.12f}")
              
            # Display information
            print()
            print("Energy (ab initio) : ", e)
            print("Bias Energy (pred) : ", e_k_pred)
            print("Bias Energy (corr) : ", e_k_corr)
            print("RMS B. grad (pred) : ", np.sqrt((g_k_pred**2).mean()))
            print("RMS B. grad (corr) : ", np.sqrt((g_k_corr**2).mean()))
            print("-" * 30)

        # Save final data visualization
        G = Graph(self.directory)
        rms_gradient_list = []
        with open(self.csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                rms_gradient_list.append(float(row[3]))
        
        if self.path_bending_angle_list:
            G.single_plot(
                np.array(range(len(self.path_bending_angle_list))),
                np.array(self.path_bending_angle_list),
                self.directory,
                atom_num=0,
                axis_name_1="# STEP",
                axis_name_2="bending angle [degrees]",
                name="IRC_bending"
            )
        
        return