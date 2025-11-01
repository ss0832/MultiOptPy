
import os
import shutil
import numpy as np
import sys
import copy
import traceback

# Try importing Matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Energy profile plots will not be generated.")
    print("Please run 'pip install matplotlib' to enable plotting.")

# Use relative imports as this file is inside the 'Wrapper' package
try:
    from multioptpy.Wrapper.optimize_wrapper import OptimizationJob
    from multioptpy.Wrapper.neb_wrapper import NEBJob
except ImportError:
    print("Error: Could not import OptimizationJob or NEBJob from relative paths.")
    print("Ensure autots.py and the wrappers are in the same 'Wrapper' directory.")
    sys.exit(1)


class AutoTSWorkflow:
    """
    Manages the 4-step (AFIR -> NEB -> TS -> IRC) automated workflow.
    """
    def __init__(self, config):
        self.config = config
        self.work_dir = config.get("work_dir", "autots_workflow")
        self.initial_mol_file = config.get("initial_mol_file")
        self.conf_file_source = config.get("software_path_file_source")
        
        self.top_n_candidates = config.get("top_n_candidates", 3)
        self.skip_step1 = config.get("skip_step1", False) 
        self.run_step4 = config.get("run_step4", False)
        self.skip_to_step4 = config.get("skip_to_step4", False)
        
        self.input_base_name = os.path.splitext(os.path.basename(self.initial_mol_file))[0]
        # This will be populated by Step 3
        self.ts_final_files = []

    def setup_workspace(self):
        """Prepares the working directory and copies necessary input files."""
        if os.path.exists(self.work_dir):
            print(f"Warning: Working directory '{self.work_dir}' already exists.")
        os.makedirs(self.work_dir, exist_ok=True)
        
        try:
            abs_initial_path = os.path.abspath(self.initial_mol_file)
            local_mol_name = os.path.basename(self.initial_mol_file)
            shutil.copy(abs_initial_path, os.path.join(self.work_dir, local_mol_name))
            self.initial_mol_file = local_mol_name 
        except shutil.Error as e:
            print(f"Warning: Could not copy initial file (may be the same file): {e}")
        except FileNotFoundError:
            print(f"Error: Initial molecule file not found: {self.initial_mol_file}")
            raise
            
        if not self.conf_file_source or not os.path.exists(self.conf_file_source):
             raise FileNotFoundError(
                 f"Software config file not found at: {self.conf_file_source}"
             )
        try:
            local_conf_name = os.path.basename(self.conf_file_source)
            shutil.copy(self.conf_file_source, os.path.join(self.work_dir, local_conf_name))
            print(f"Copied '{self.conf_file_source}' to '{self.work_dir}'")
        except shutil.Error as e:
            print(f"Warning: Could not copy software_path.conf (may be the same file): {e}")

        os.chdir(self.work_dir)
        print(f"Changed directory to: {os.getcwd()}")
        

    def _run_step1_afir_scan(self):
        """Runs the AFIR scan and copies the resulting trajectory."""
        print("\n--- 1. STARTING STEP 1: AFIR SCAN ---")
        job1_settings = self.config.get("step1_settings", {})
            
        if not job1_settings.get("manual_AFIR"):
             raise ValueError("Step 1 settings must contain 'manual_AFIR' (-ma) options.")

        job = OptimizationJob(input_file=self.initial_mol_file)
        job.set_options(**job1_settings)
        job.run()
        
        optimizer_instance = job.get_results()
        if optimizer_instance is None:
             raise RuntimeError("Step 1 failed to produce an optimizer instance.")
             
        optimizer_instance.get_result_file_path() 
        
        source_traj_path = optimizer_instance.traj_file
        if not source_traj_path or not os.path.exists(source_traj_path):
            raise FileNotFoundError(f"Step 1 finished, but 'traj_file' was not found at: {source_traj_path}")

        local_traj_name = f"{self.input_base_name}_step1_traj.xyz"
        shutil.copy(source_traj_path, local_traj_name)
        
        print(f"Copied AFIR trajectory to: {os.path.abspath(local_traj_name)}")
        print("--- STEP 1: AFIR SCAN COMPLETE ---")
        return local_traj_name 

    def _run_step2_neb_optimization(self, afir_traj_path):
        """Runs NEB, filters candidates by energy, and copies the top N."""
        print("\n--- 2. STARTING STEP 2: NEB OPTIMIZATION ---")
        
        job = NEBJob(input_files=[afir_traj_path])
        job2_settings = self.config.get("step2_settings", {})
        job.set_options(**job2_settings)
        job.run()

        neb_instance = job.get_results()
        if neb_instance is None:
            raise RuntimeError("Step 2 failed to produce an NEB instance.")

        neb_instance.get_result_file() 
        source_ts_paths = neb_instance.ts_guess_file_list

        if not source_ts_paths:
            print("Step 2 (NEB) did not find any TS candidate files (ts_guess_file_list is empty).")
            return [] 

        energy_csv_path = os.path.join(neb_instance.config.NEB_FOLDER_DIRECTORY, "energy_plot.csv")
        selected_paths = self._filter_candidates_by_energy(source_ts_paths, energy_csv_path)

        refinement_dir = f"{self.input_base_name}_step3_TS_Opt_Inputs"
        os.makedirs(refinement_dir, exist_ok=True)
        local_ts_paths = []
        
        print(f"Copying {len(selected_paths)} highest energy candidates for refinement...")
        for i, source_path in enumerate(selected_paths):
            if not os.path.exists(source_path):
                print(f"Warning: Source file not found, skipping: {source_path}")
                continue
                
            local_guess_name = f"{self.input_base_name}_ts_guess_{i+1}.xyz"
            local_path = os.path.join(refinement_dir, local_guess_name)
            shutil.copy(source_path, local_path)
            print(f"  Copied {os.path.basename(source_path)} to {local_path}")
            local_ts_paths.append(local_path)
            
        print("--- STEP 2: NEB OPTIMIZATION COMPLETE ---")
        return local_ts_paths 

    def _filter_candidates_by_energy(self, file_paths, energy_csv_path):
        """
        Parses the energy_plot.csv, correlates it with candidate file paths,
        and returns the paths for the Top N highest energy candidates.
        """
        print(f"Filtering {len(file_paths)} candidates down to a max of {self.top_n_candidates} by energy...")
        
        try:
            with open(energy_csv_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    raise ValueError(f"{energy_csv_path} is empty.")
                last_line = lines[-1].strip()
            energies = np.array([float(e) for e in last_line.split(',') if e.strip()])
        except Exception as e:
            print(f"Warning: Could not read or parse energy file '{energy_csv_path}': {e}")
            print("Proceeding with all found candidates (unsorted).")
            return file_paths[:self.top_n_candidates]

        candidates = []
        for path in file_paths:
            try:
                base_name = os.path.splitext(os.path.basename(path))[0]
                index_str = base_name.split('_')[-1]
                z = int(index_str)
                if z >= len(energies):
                    print(f"Warning: Index {z} from file '{path}' is out of bounds for energy list (len {len(energies)}).")
                    continue
                candidates.append((energies[z], path))
            except Exception as e:
                print(f"Warning: Could not parse index from '{path}': {e}. Skipping.")

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_n_list = candidates[:self.top_n_candidates]
        
        print(f"Identified {len(candidates)} candidates, selecting top {len(top_n_list)}:")
        for (energy, path) in top_n_list:
            print(f"  - Path: {os.path.basename(path)}, Energy: {energy:.6f} Hartree")
            
        return [path for energy, path in top_n_list]

    def _run_step3_ts_refinement(self, local_ts_guess_paths):
        """Runs saddle_order=1 OptimizationJob on all local candidates."""
        print("\n--- 3. STARTING STEP 3: TS REFINEMENT ---")
        
        if not local_ts_guess_paths:
            print("No TS candidates provided. Skipping refinement.")
            return []
            
        job3_settings = self.config.get("step3_settings", {})
        final_ts_files = []

        for i, guess_file_path in enumerate(local_ts_guess_paths):
            print(f"\n--- Running TS refinement for candidate {i+1}/{len(local_ts_guess_paths)} ({guess_file_path}) ---")
            
            job = OptimizationJob(input_file=guess_file_path)
            current_job_settings = job3_settings.copy()
            current_job_settings['saddle_order'] = 1
            job.set_options(**current_job_settings)
            job.run()
            
            optimizer_instance = job.get_results()
            if optimizer_instance is None:
                print(f"Warning: Refinement for {guess_file_path} failed.")
                continue

            optimizer_instance.get_result_file_path()
            source_final_ts_path = optimizer_instance.optimized_struct_file
            if not source_final_ts_path or not os.path.exists(source_final_ts_path):
                print(f"Warning: Refinement for {guess_file_path} finished, but 'optimized_struct_file' was not found.")
                continue
                
            local_final_name = f"{self.input_base_name}_ts_final_{i+1}.xyz"
            shutil.copy(source_final_ts_path, local_final_name)
            
            print(f"Copied final TS structure to: {os.path.abspath(local_final_name)}")
            final_ts_files.append(local_final_name)

        print("--- STEP 3: TS REFINEMENT COMPLETE ---")
        return final_ts_files

    def _run_step4_irc_and_opt(self, ts_final_files):
        """
        Runs Step 4: IRC calculation, endpoint optimization, and visualization.
        """
        print("\n--- 4. STARTING STEP 4: IRC & EQ OPTIMIZATION ---")
        
        if not ts_final_files:
            print("No TS files provided from Step 3 (or --skip_to_step4). Skipping Step 4.")
            return

        step4_settings = self.config.get("step4_settings", {})
        if "intrinsic_reaction_coordinates" not in step4_settings:
            raise ValueError("Step 4 requires 'intrinsic_reaction_coordinates' settings in config.json")

        for i, ts_path in enumerate(ts_final_files):
            ts_index = i + 1
            # If skipping, the base name might be long, so create a short ID
            if self.skip_to_step4:
                ts_name_base = f"{self.input_base_name}_TS_{ts_index}"
            else:
                ts_name_base = f"{self.input_base_name}_ts_final_{ts_index}"
                
            print(f"\n--- Running Step 4 for TS Candidate {ts_index} ({ts_path}) ---")
            
            
            step4_settings["saddle_order"] = 1
            
            # --- 4A: Run IRC ---
            print(f"  4A: Running IRC for {ts_path}...")
            job_irc = OptimizationJob(input_file=ts_path)
            # Pass the full Step 4 settings (including -irc)
            job_irc.set_options(**step4_settings)
            
            job_irc.run()

            irc_instance = job_irc.get_results()
            if irc_instance is None:
                print(f"  Warning: IRC job for {ts_path} failed.")
                continue
                
            # Get TS energy (as per Q2, this is set before IRC runs)
            ts_e = irc_instance.final_energy
            ts_bias_e = irc_instance.final_bias_energy
            
            # Get IRC endpoint paths
            endpoint_paths = irc_instance.irc_terminal_struct_paths
            if not endpoint_paths or len(endpoint_paths) != 2:
                print(f"  Warning: IRC job for {ts_path} did not return 2 endpoint files. Aborting Step 4 for this candidate.")
                continue

            print(f"  IRC found endpoints: {endpoint_paths}")

            # --- 4B: Run Endpoint Optimization ---
            endpoint_results = []
            for j, end_path in enumerate(endpoint_paths):
                
                shutil.copy(end_path, f"{ts_name_base}_IRC_Endpoint_{j+1}.xyz")
                end_path = f"{ts_name_base}_IRC_Endpoint_{j+1}.xyz"
                
                
                print(f"  4B: Optimizing endpoint {j+1} from {end_path}...")
                job_opt = OptimizationJob(input_file=end_path)
                    
                # Prepare settings for endpoint optimization
                opt_settings = copy.deepcopy(step4_settings)
                
                # Use the new dedicated opt_method for step 4B
                opt_settings["opt_method"] = opt_settings.get("step4b_opt_method", ["rsirfo_block_fsb"])
                
                # Remove IRC flag (as per Q3)
                opt_settings.pop("intrinsic_reaction_coordinates", None)
                
                # Set saddle_order=0 (minimization)
                opt_settings['saddle_order'] = 0
                
                job_opt.set_options(**opt_settings)
                job_opt.run()
                
                opt_instance = job_opt.get_results()
                if opt_instance is None:
                    print(f"  Warning: Optimization for endpoint {end_path} failed.")
                    continue
                
                opt_instance.get_result_file_path()
                final_opt_path = opt_instance.optimized_struct_file
                if not final_opt_path or not os.path.exists(final_opt_path):
                     print(f"  Warning: Optimization for {end_path} finished, but 'optimized_struct_file' was not found.")
                     continue
                
                endpoint_results.append({
                    "path": final_opt_path,
                    "e": opt_instance.final_energy,
                    "bias_e": opt_instance.final_bias_energy,
                    "label": f"Endpoint_{j+1}"
                })

            if not endpoint_results:
                print(f"  Warning: Failed to optimize any endpoints for {ts_path}. Aborting result collection.")
                continue
            
            if len(endpoint_results) > 2:
                print(f"  Warning: More than 2 optimized endpoints found.")
                raise ValueError("More than 2 endpoints found after optimization.")

            # --- 4C: Collect Results & Visualize ---
            print(f"  4C: Collecting results for TS Candidate {ts_index}...")
            # Prepare result directory
            result_dir = f"{ts_name_base}_Step4_Profile"
            os.makedirs(result_dir, exist_ok=True)
           
            e_profile = {
                "TS": {"e": ts_e, "bias_e": ts_bias_e, "path": ts_path},
            }
            if len(endpoint_results) >= 1:
                e_profile["End1"] = endpoint_results[0]
            if len(endpoint_results) >= 2:
                e_profile["End2"] = endpoint_results[1]
        
 
            # Create plot (as per Q1)
            plot_path = os.path.join(result_dir, "energy_profile.png")
            self._create_energy_profile_plot(e_profile, plot_path, ts_name_base)
 
            # Write text file
            text_path = os.path.join(result_dir, "energy_profile.txt")
            self._write_energy_profile_text(e_profile, text_path, ts_name_base)
            
            # Copy final XYZ files 
            shutil.copy(ts_path, os.path.join(result_dir, f"{ts_name_base}_ts_final.xyz"))
            if "End1" in e_profile:
                shutil.copy(e_profile["End1"]["path"], os.path.join(result_dir, "endpoint_1_opt.xyz"))
            if "End2" in e_profile:
                shutil.copy(e_profile["End2"]["path"], os.path.join(result_dir, "endpoint_2_opt.xyz"))
            
            print(f"  Successfully saved profile and structures to: {result_dir}")

        print("\n--- STEP 4: IRC & EQ OPTIMIZATION COMPLETE ---") 

    def _create_energy_profile_plot(self, e_profile, output_path, title_name):
        """Generates an energy profile plot using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            print(f"  Skipping plot generation: matplotlib not installed.")
            return
 
        try:
            labels = []
            energies = []
            biased_energies = []
 
       
            if "End1" in e_profile:
                labels.append("End1")
                energies.append(e_profile["End1"]["e"])
                biased_energies.append(e_profile["End1"]["bias_e"])
 
            labels.append("TS")
            energies.append(e_profile["TS"]["e"])
            biased_energies.append(e_profile["TS"]["bias_e"])
 
            if "End2" in e_profile:
                labels.append("End2")
                energies.append(e_profile["End2"]["e"])
                biased_energies.append(e_profile["End2"]["bias_e"])
            
         
            if not energies:
                print("  Warning: No energies found to plot.")
                return
 
            # Convert to relative kcal/mol
            min_e = min(energies)
            min_bias_e = min(biased_energies)
            
            rel_energies = (np.array(energies) - min_e) * 627.509
            rel_biased_energies = (np.array(biased_energies) - min_bias_e) * 627.509
 
            x = list(range(len(labels))) 
            plt.figure(figsize=(8, 6))
            
            # Plot both energies as requested in Q1
            plt.plot(x, rel_energies, 'o-', c='blue', label='Energy (E_final)')
            plt.plot(x, rel_biased_energies, 'o--', c='red', label='Bias Energy (E_bias_final)')
            
            plt.xticks(x, labels)
            plt.ylabel("Relative Energy (kcal/mol)")
            plt.title(f"Reaction Profile for {title_name}")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"  Generated energy plot: {output_path}")
 
        except Exception as e:
            print(f"  Warning: Failed to generate energy plot: {e}")

    def _write_energy_profile_text(self, e_profile, output_path, title_name):
        """Writes the final energies to a text file."""
        try:
            with open(output_path, 'w') as f:
                f.write(f"# Energy Profile for {title_name}\n")
                f.write("# All energies in Hartree\n")
                f.write("# -------------------------------------------------------------------\n")
                f.write(f"Structure,    File_Path,                          Final_Energy,     Final_Bias_Energy\n")
                
                
                ts_total = e_profile['TS']['bias_e']
                f.write(f"TS,           {e_profile['TS']['path']}, {e_profile['TS']['e']:.12f}, {e_profile['TS']['bias_e']:.12f}\n")

                end1_total = None
                if "End1" in e_profile:
                    end1_total = e_profile['End1']['bias_e']
                    f.write(f"Endpoint_1,   {e_profile['End1']['path']}, {e_profile['End1']['e']:.12f}, {e_profile['End1']['bias_e']:.12f}\n")
                
                end2_total = None
                if "End2" in e_profile:
                    end2_total = e_profile['End2']['bias_e']
                    f.write(f"Endpoint_2,   {e_profile['End2']['path']}, {e_profile['End2']['e']:.12f}, {e_profile['End2']['bias_e']:.12f}\n")
                
                f.write("# -------------------------------------------------------------------\n")
                 # Calculate and write barriers and reaction energies if possible
                if end1_total is not None:
                    barrier_1 = (ts_total - end1_total) * 627.509
                    f.write(f"Activation Energy (End1 -> TS): {barrier_1: .2f} kcal/mol\n")
 
                if end2_total is not None:
                    barrier_2 = (ts_total - end2_total) * 627.509
                    f.write(f"Activation Energy (End2 -> TS): {barrier_2: .2f} kcal/mol\n")
 
                if end1_total is not None and end2_total is not None:
                    reaction_e = (end2_total - end1_total) * 627.509
                    f.write(f"Reaction Energy (End1 -> End2): {reaction_e: .2f} kcal/mol\n")
                elif end1_total is not None:
                    pass
 
 
            print(f"  Generated energy text file: {output_path}")
        except Exception as e:
            print(f"  Warning: Failed to write energy text file: {e}")
            
           
    def run_workflow(self):
        """Executes the full automated workflow."""
        original_cwd = os.getcwd()
        try:
            if not os.path.exists(self.initial_mol_file):
                raise FileNotFoundError(f"Initial molecule file not found: {self.initial_mol_file}")
            
            self.setup_workspace() 
            
            if self.skip_to_step4:
                # --- Run Step 4 Only ---
                print("\n--- Skipping to Step 4 ---")
                # The input file is the TS file
                self.ts_final_files = [self.initial_mol_file]
                # Ensure base name is from the TS file
                self.input_base_name = os.path.splitext(os.path.basename(self.initial_mol_file))[0]
            
            else:
                # --- Run Steps 1-3 ---
                if self.skip_step1:
                    print("\n--- 1. STEP 1 (AFIR) SKIPPED ---")
                    local_afir_traj = self.initial_mol_file
                else:
                    local_afir_traj = self._run_step1_afir_scan()
                
                local_ts_paths = self._run_step2_neb_optimization(local_afir_traj)
                
                if not local_ts_paths:
                    print("Step 2 found 0 candidates. Workflow terminated.")
                    print(f"\n --- AUTO-TS WORKFLOW COMPLETED (NO TS FOUND) --- ")
                    return

                self.ts_final_files = self._run_step3_ts_refinement(local_ts_paths)

            # --- Run Step 4 (if flagged) ---
            if (self.run_step4 or self.skip_to_step4) and self.ts_final_files:
                self._run_step4_irc_and_opt(self.ts_final_files)
            
            print(f"\n --- AUTO-TS WORKFLOW COMPLETED SUCCESSFULLY --- ")
            print(f"All results are in: {os.path.realpath(os.getcwd())}")

        except Exception as e:
            print(f"\n --- AUTO-TS WORKFLOW FAILED --- ")
            print(f"Error: {e}")
            
            traceback.print_exc()
        finally:
            os.chdir(original_cwd)
            print(f"Returned to directory: {original_cwd}")

