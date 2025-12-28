import numpy as np
import os
import shutil

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Visualization.visualization import Graph

class SpringPairMethod:
    """
    Implementation of the Spring Pair Method (SPM) with Adaptive Step Size & Momentum.
    Modified to accept a SINGLE initial structure and auto-generate the second one via perturbation.
    """
    def __init__(self, config):
        self.config = config
        
        # --- TUNING PARAMETERS ---
        self.k_spring = 10.0 
        self.l_s = max(getattr(self.config, 'L_covergence', 0.1), 0.1)
        self.initial_drift_step = 0.01
        self.climb_step_size = 0.50 
        self.drift_limit = 100
        self.momentum = 0.3
        self.MAX_FORCE_THRESHOLD = 0.00100
        self.RMS_FORCE_THRESHOLD = 0.00005
        
        # Magnitude of random perturbation to generate Image 2 (Bohr)
        self.perturbation_scale = 0.1 
        
    def RMS(self, mat):
        return np.sqrt(np.mean(mat**2))
    
    def print_info(self, dat, phase):
        print(f"[[{phase} information]]")
        print(f"                                                image_1               image_2")
        print(f"energy (Hartree)                       : {dat['energy_1']:.8f}   {dat['energy_2']:.8f}")
        print(f"gradient RMS                           : {self.RMS(dat['gradient_1']):.6f}   {self.RMS(dat['gradient_2']):.6f}")
        
        if "perp_force_1" in dat:
            print(f"perp_force (Drift)                     : {self.RMS(dat['perp_force_1']):.6f}   {self.RMS(dat['perp_force_2']):.6f}")
            print(f"spring_force                           : {self.RMS(dat['spring_force_1']):.6f}   {self.RMS(dat['spring_force_2']):.6f}")
            
        print(f"spring_length (Bohr)                   : {dat['spring_length']:.6f}")
        step_info = dat.get('step_size', 0)
        print(f"DEBUG: k={self.k_spring:.1f}, step={step_info:.6f}")
        print("-" * 80)
        return

    def get_spring_vectors(self, geom_1, geom_2):
        diff = geom_2 - geom_1
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            rand_vec = np.random.randn(*diff.shape)
            unit_vec = rand_vec / np.linalg.norm(rand_vec)
            dist = 1e-10 
        else:
            unit_vec = diff / dist
        return dist, unit_vec

    def decompose_gradient(self, gradient, unit_vec):
        grad_flat = gradient.flatten()
        vec_flat = unit_vec.flatten()
        grad_par_mag = np.dot(grad_flat, vec_flat)
        grad_par = grad_par_mag * unit_vec
        grad_perp = gradient - grad_par
        return grad_par, grad_perp

    def _generate_perturbed_structure(self, geom, scale):
        """Generate a new structure by adding random perturbation to the given geometry."""
        # Generate random noise
        noise = np.random.randn(*geom.shape)
        # Normalize to unit vectors per atom to distribute perturbation evenly
        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        noise = noise / (norms + 1e-10)
        # Scale the noise
        perturbation = noise * scale
        return geom + perturbation

    def iteration(self, file_directory_1, SP1, element_list, electric_charge_and_multiplicity, FIO1):
        """
        Main SPM Optimization Loop.
        Accepts ONE initial structure. Image 2 is auto-generated.
        """
        G = Graph(self.config.iEIP_FOLDER_DIRECTORY)
        ENERGY_LIST_1, ENERGY_LIST_2 = [], []
        GRAD_LIST_1, GRAD_LIST_2 = [], []

        # --- Initialize Image 2 from Image 1 ---
        print("### Initializing SPM ###")
        print(f"Base Structure (Image 1): {file_directory_1}")
        
        # 1. Get initial geometry of Image 1
        # We perform a dummy single point calculation or just read the geom if possible.
        # Here we run SP1 to get the geometry and energy reliably.
        init_energy, init_grad, init_geom, err = SP1.single_point(
            file_directory_1, element_list, "init_check", 
            electric_charge_and_multiplicity, self.config.force_data["xtb"]
        )
        if err:
            print("[Error] Failed to read initial structure.")
            return

        # 2. Generate Image 2 geometry
        print(f"Generating Image 2 with random perturbation (Scale: {self.perturbation_scale} Bohr)...")
        init_geom_2 = self._generate_perturbed_structure(init_geom, self.perturbation_scale)
        
        # Create a directory for Image 2 (internally managed)
        # We can just use the same SP1 object but we need a file path for it.
        # We will generate input files for Image 2 on the fly using FIO1 logic.
        # Note: We reuse SP1 and FIO1 for both images since they share physics/settings.
        
        # Initialize loop variables
        geom_num_list_1 = init_geom
        geom_num_list_2 = init_geom_2
        
        velocity_1 = np.zeros((len(element_list), 3))
        velocity_2 = np.zeros((len(element_list), 3))
        
        current_drift_step = self.initial_drift_step
        
        # Variables to store the latest geometry
        new_geom_1 = None
        new_geom_2 = None

        for cycle in range(0, self.config.microiterlimit):
            if os.path.isfile(self.config.iEIP_FOLDER_DIRECTORY+"end.txt"):
                break
            
            print(f"### Cycle {cycle} Start ###")

            # =========================================================================
            # 1. Drifting Phase
            # =========================================================================
            print(f"--- Drifting Phase (Cycle {cycle}) ---")
            
            prev_force_drift_1 = None
            prev_force_drift_2 = None
            
            drift_temp_label = f"{cycle}_drift_temp"

            for d_step in range(self.drift_limit):
                iter_label = drift_temp_label
                
                # Make input files for this step
                # Note: file_directory_1/2 here are just strings for the input file path
                # generated by _make_next_input.
                # However, SP1.single_point expects the directory path or file path depending on implementation.
                # Assuming _make_next_input returns the PATH to the input file.
                
                input_path_1 = self._make_next_input(FIO1, geom_num_list_1, element_list, electric_charge_and_multiplicity, iter_label + "_img1")
                input_path_2 = self._make_next_input(FIO1, geom_num_list_2, element_list, electric_charge_and_multiplicity, iter_label + "_img2")
                
                # 1. QM Calculation
                # We reuse SP1 for both. It's just a calculator.
                energy_1, gradient_1, g1_read, error_flag_1 = SP1.single_point(input_path_1, element_list, iter_label + "_img1", electric_charge_and_multiplicity, self.config.force_data["xtb"])
                energy_2, gradient_2, g2_read, error_flag_2 = SP1.single_point(input_path_2, element_list, iter_label + "_img2", electric_charge_and_multiplicity, self.config.force_data["xtb"])
                
                # Update geom from read result to be safe, or stick to internal numpy array
                # Using internal array (geom_num_list_1/2) is safer for consistency unless optimizer changes it.
                # But let's align them just in case output orientation changed.
                # g1_read, g2_read = Calculationtools().kabsch_algorithm(g1_read, g2_read)
                # geom_num_list_1, geom_num_list_2 = g1_read, g2_read
                
                # Align current internal geometries
                geom_num_list_1, geom_num_list_2 = Calculationtools().kabsch_algorithm(geom_num_list_1, geom_num_list_2)

                if error_flag_1 or error_flag_2:
                    return

                # 2. Vector & Force Calculation
                ds, vs = self.get_spring_vectors(geom_num_list_1, geom_num_list_2)
                
                _, grad_perp_1 = self.decompose_gradient(gradient_1, vs)
                _, grad_perp_2 = self.decompose_gradient(gradient_2, vs)
                
                force_perp_1 = -grad_perp_1
                force_perp_2 = -grad_perp_2
                
                spring_force_mag = self.k_spring * (ds - self.l_s)
                spring_force_1 = spring_force_mag * vs 
                spring_force_2 = spring_force_mag * (-vs)
                
                total_force_1 = force_perp_1 + spring_force_1
                total_force_2 = force_perp_2 + spring_force_2
                
                # --- Adaptive Step Logic ---
                if prev_force_drift_1 is not None:
                    dot_1 = np.sum(prev_force_drift_1 * total_force_1)
                    dot_2 = np.sum(prev_force_drift_2 * total_force_2)
                    
                    if dot_1 < 0 or dot_2 < 0:
                        current_drift_step *= 0.5
                        velocity_1 *= 0.1 
                        velocity_2 *= 0.1
                        print(f"  [Auto-Brake] Oscillation detected at step {d_step}. Reduced drift step to {current_drift_step:.6f}")
                    else:
                        current_drift_step = min(current_drift_step * 1.05, self.initial_drift_step)
                
                prev_force_drift_1 = total_force_1.copy()
                prev_force_drift_2 = total_force_2.copy()

                # Update Position
                velocity_1 = self.momentum * velocity_1 + current_drift_step * total_force_1
                velocity_2 = self.momentum * velocity_2 + current_drift_step * total_force_2
                
                geom_num_list_1 += velocity_1
                geom_num_list_2 += velocity_2
                
                # Check Convergence
                drift_metric = max(self.RMS(force_perp_1), self.RMS(force_perp_2))
                
                if d_step % 5 == 0: 
                    info_dat = {
                        "energy_1": energy_1, "energy_2": energy_2,
                        "gradient_1": gradient_1, "gradient_2": gradient_2,
                        "perp_force_1": force_perp_1, "perp_force_2": force_perp_2,
                        "spring_force_1": spring_force_1, "spring_force_2": spring_force_2,
                        "spring_length": ds, 
                        "step_size": current_drift_step,
                        "convergence_metric": drift_metric
                    }
                    self.print_info(info_dat, f"Cycle {cycle} - Drifting {d_step}")

                if drift_metric < self.RMS_FORCE_THRESHOLD:
                    print(f"  >> Drifting converged at step {d_step}")
                    break
            
            # =========================================================================
            # 2. Climbing Phase
            # =========================================================================
            print(f"--- Climbing Phase (Cycle {cycle}) ---")
            
            iter_label = f"{cycle}_climb"
            
            input_path_1 = self._make_next_input(FIO1, geom_num_list_1, element_list, electric_charge_and_multiplicity, iter_label + "_img1")
            input_path_2 = self._make_next_input(FIO1, geom_num_list_2, element_list, electric_charge_and_multiplicity, iter_label + "_img2")
            
            energy_1, gradient_1, g1_read, error_flag_1 = SP1.single_point(input_path_1, element_list, iter_label + "_img1", electric_charge_and_multiplicity, self.config.force_data["xtb"])
            energy_2, gradient_2, g2_read, error_flag_2 = SP1.single_point(input_path_2, element_list, iter_label + "_img2", electric_charge_and_multiplicity, self.config.force_data["xtb"])
            
            geom_num_list_1, geom_num_list_2 = Calculationtools().kabsch_algorithm(geom_num_list_1, geom_num_list_2)
            
            ds, vs = self.get_spring_vectors(geom_num_list_1, geom_num_list_2)
            grad_par_1, _ = self.decompose_gradient(gradient_1, vs)
            grad_par_2, _ = self.decompose_gradient(gradient_2, vs)
            
            active_climb_step = self.climb_step_size
            
            # Move UP
            geom_num_list_1 += active_climb_step * grad_par_1
            geom_num_list_2 += active_climb_step * grad_par_2
            
            # Update 'new_geom' for final output reference
            new_geom_1 = geom_num_list_1.copy()
            new_geom_2 = geom_num_list_2.copy()
            
            grad_norm_1 = np.linalg.norm(gradient_1)
            grad_norm_2 = np.linalg.norm(gradient_2)
            global_metric = min(grad_norm_1, grad_norm_2)

            info_dat = {
                "energy_1": energy_1, "energy_2": energy_2,
                "gradient_1": gradient_1, "gradient_2": gradient_2,
                "par_force_1": grad_par_1, "par_force_2": grad_par_2, 
                "spring_length": ds, "convergence_metric": global_metric,
                "step_size": active_climb_step
            }
            self.print_info(info_dat, f"Cycle {cycle} - Climbing")
            
            ENERGY_LIST_1.append(energy_1 * self.config.hartree2kcalmol)
            ENERGY_LIST_2.append(energy_2 * self.config.hartree2kcalmol)
            GRAD_LIST_1.append(grad_norm_1)
            GRAD_LIST_2.append(grad_norm_2)

            if cycle > 5 and global_metric < self.MAX_FORCE_THRESHOLD:
                print("!!! Global Convergence Reached !!!")
                print(f"Saddle point candidate found around cycle {cycle}")
                break

        # --- END OF OPTIMIZATION LOOP ---
        
        # Save the optimized structure
        if new_geom_1 is not None and new_geom_2 is not None:
            avg_geom = (new_geom_1 + new_geom_2) / 2.0
            avg_geom_ang = avg_geom * self.config.bohr2angstroms
            
            dir_name = os.path.basename(os.path.normpath(self.config.iEIP_FOLDER_DIRECTORY))
            output_xyz_name = f"{dir_name}_optimized.xyz"
            
            try:
                with open(output_xyz_name, "w") as f:
                    num_atoms = len(element_list)
                    f.write(f"{num_atoms}\n")
                    f.write(f"SPM Optimized Saddle Point (Average)\n")
                    for i, elem in enumerate(element_list):
                        x, y, z = avg_geom_ang[i]
                        f.write(f"{elem} {x:.10f} {y:.10f} {z:.10f}\n")
                print(f"\n[Success] Final optimized structure saved to: {output_xyz_name}")
            except Exception as e:
                print(f"\n[Error] Failed to save optimized structure: {e}")

        return

    def _make_next_input(self, FIO, geom, element_list, charge_mult, iter_label):
        """
        Creates a PSI4 input file and returns its path.
        """
        geom_tolist = (geom * self.config.bohr2angstroms).tolist()
        for i, elem in enumerate(element_list):
            geom_tolist[i].insert(0, elem)
        geom_tolist.insert(0, charge_mult)
        
        # FIO.make_psi4_input_file usually returns the directory or path.
        # Ensure this matches your FIO implementation.
        return FIO.make_psi4_input_file([geom_tolist], iter_label)