
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

            if not optimizer_instance.optimized_flag:
                print(f"Warning: Refinement for {guess_file_path} did not converge (optimized_flag=False). Skipping.")
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


class AutoTSWorkflow_v2(AutoTSWorkflow):
    """
    AutoTSWorkflow (v2) - (FIXED)
    
    Manages a dynamic, repeatable, multi-step workflow based on a 
    "workflow" block in the configuration file.
    
    """

    def __init__(self, config):
        """
        Initializes the v2 workflow.
        """
        # Call the parent __init__ to load basic settings
        super().__init__(config)
        
        # v2-specific attributes
        self.data_cache = {}
        self.workflow_steps = self.config.get("workflow", [])
        
        # Validate the workflow configuration immediately
        try:
            self._validate_workflow_config()
        except ValueError as e:
            print(f"Error: Workflow configuration is invalid.")
            print(f"Details: {e}")
            sys.exit(1)

    def _validate_workflow_config(self):
        """
        Validates the 'workflow' block in config.json.
        """
        print("Validating workflow configuration...")
        if not self.workflow_steps:
            print("Warning: 'workflow' block is empty or missing. No steps will be run.")
            return

        for i, entry in enumerate(self.workflow_steps):
            if "step" not in entry:
                raise ValueError(f"Workflow entry {i} is missing required key 'step'.")
            
            step_name = entry["step"]
            if not hasattr(self, f"_run_{step_name}"):
                raise ValueError(f"Workflow entry {i} specifies invalid step: '{step_name}'. No method '_run_{step_name}' found.")
            
            repeat = entry.get("repeat", 1)
            if not isinstance(repeat, int) or repeat < 1:
                 raise ValueError(f"Workflow entry {i} ({step_name}): 'repeat' must be a positive integer.")
                 
            repeat_settings = entry.get("repeat_settings", [])
            
            if repeat_settings and len(repeat_settings) > repeat:
                raise ValueError(f"Workflow entry {i} ({step_name}): 'repeat_settings' list (len {len(repeat_settings)}) is longer than 'repeat' value ({repeat}).")
            
            base_key = entry.get("settings_key", f"{step_name}_settings")
            if base_key not in self.config:
                 raise ValueError(f"Workflow entry {i} ({step_name}): Base settings key '{base_key}' (or default) not found in main config.")
        
        print("Workflow validation successful.")

    def run_workflow(self):
        """
        Overrides the parent 'run_workflow' to call the
        v2 dynamic workflow engine.
        """
        original_cwd = os.getcwd()
        try:
            if not os.path.exists(self.initial_mol_file):
                if not self.config.get("skip_to_step4"):
                     raise FileNotFoundError(f"Initial molecule file not found: {self.initial_mol_file}")
            
            self.setup_workspace()
            self.run_dynamic_workflow()

            print(f"\n --- AUTO-TS WORKFLOW (V2) COMPLETED SUCCESSFULLY --- ")
            print(f"All results are in: {os.path.realpath(os.getcwd())}")

        except Exception as e:
            print(f"\n --- AUTO-TS WORKFLOW (V2) FAILED --- ")
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            os.chdir(original_cwd)
            print(f"Returned to directory: {original_cwd}")

    def _get_settings_for_repeat(self, wf_entry, repeat_index):
        """
        Gets the correct settings dictionary for a specific repeat.
        """
        step_name = wf_entry["step"]
        repeat_settings = wf_entry.get("repeat_settings", [])
        
        base_settings_key = wf_entry.get("settings_key", f"{step_name}_settings")
        
        if base_settings_key not in self.config:
            raise ValueError(f"Failed to find base settings key '{base_settings_key}' in config for {step_name}, repeat {repeat_index+1}.")
            
        param_override = {}
        r_setting = None 

        if repeat_index < len(repeat_settings):
            r_setting = repeat_settings[repeat_index]
        elif repeat_settings:
            r_setting = repeat_settings[-1]
            if repeat_index == len(repeat_settings): 
                print(f"  Info: 'repeat_settings' list (len {len(repeat_settings)}) is shorter than 'repeat' for {step_name}. Re-using last entry for repeat {repeat_index+1} and beyond.")
        
        if r_setting:
            param_override = r_setting.get("param_override", {})

        final_settings = copy.deepcopy(self.config[base_settings_key])
        final_settings.update(param_override) 
        
        return final_settings

    def run_dynamic_workflow(self):
        """
        Executes the dynamic workflow defined in config['workflow'].
        """
        print("\n--- 🚀 STARTING DYNAMIC WORKFLOW (V2) ---")
        
        for entry in self.workflow_steps:
            self.data_cache[entry["step"]] = {"runs": []}

        for wf_entry in self.workflow_steps:
            if not wf_entry.get("enabled", True):
                print(f"\n--- SKIPPING STEP: {wf_entry['step']} (disabled) ---")
                continue
            
            
            step_name = wf_entry["step"]
            method = getattr(self, f"_run_{step_name}")
            repeat = wf_entry.get("repeat", 1)
            
            if step_name == "step4" and self.run_step4 is not True:
                print(f"\n--- SKIPPING STEP: {step_name} (run_step4 flag not set) ---")
                continue
            if step_name == "step1" and self.skip_step1 is True:
                print(f"\n--- SKIPPING STEP: {step_name} (skip_step1 flag set) ---")
                continue
            
            if step_name != "step4" and self.skip_to_step4 is True:
                print(f"\n--- SKIPPING STEP: {step_name} (skip_to_step4 flag set) ---")
                continue
            
            
            print(f"\n--- 🏁 EXECUTING STEP: {step_name} (Repeat={repeat}) ---")

            for i in range(repeat):
                print(f"  --- {step_name} | Run {i+1}/{repeat} ---")
                
                try:
                    settings = self._get_settings_for_repeat(wf_entry, i)
                    input_data = self._determine_input_for_run(step_name, i, wf_entry)
                    result = method(settings, input_data, run_index=i)
                    self.data_cache[step_name]["runs"].append(result)
                    print(f"  --- {step_name} | Run {i+1}/{repeat} COMPLETED ---")
                
                except Exception as e:
                    print(f"  --- ❌ {step_name} | Run {i+1}/{repeat} FAILED ---")
                    print(f"  Error: {e}")
                    traceback.print_exc()
                    print(f"  Aborting remaining repeats for {step_name}.")
                    break 

            if "runs" in self.data_cache[step_name] and self.data_cache[step_name]["runs"]:
                self._run_post_processing(step_name, wf_entry)
            
            print(f"--- ✅ STEP: {step_name} COMPLETE ---")

    def _determine_input_for_run(self, step_name, run_index, wf_entry):
        """
        Determines the input for a specific run based on data dependency logic.
        (FIXED: Now uses relative paths and explicit copy for step2 sequential)
        """
        previous_runs_this_step = self.data_cache[step_name]["runs"]
        
        if step_name == "step1":
            if run_index == 0:
                print(f"  Info: '{step_name}' run 1 using initial_mol_file.")
                return {"input_file": self.initial_mol_file}
            else:
                if not previous_runs_this_step:
                    raise RuntimeError(f"Step 1, run {run_index+1}: Cannot start, previous run (0) failed or produced no output.")
                prev_result = previous_runs_this_step[-1]
                if "final_struct_file" not in prev_result:
                     raise RuntimeError(f"Step 1, run {run_index+1}: Previous run did not produce a 'final_struct_file'.")
                print(f"  Using previous run's output: {prev_result['final_struct_file']}")
                # This path is already relative (fixed in _run_step1)
                return {"input_file": prev_result['final_struct_file']}

        elif step_name == "step2":
            mode = wf_entry.get("mode", "sequential") 
            
            if "step1" not in self.data_cache or "combined_path" not in self.data_cache["step1"]:
                if run_index == 0 or mode == "independent":
                     raise RuntimeError(f"Step 2 ({mode}): data_cache[\"step1\"][\"combined_path\"] not found. Did Step 1 run and post-process?")
            
            if mode == "sequential":
                if run_index == 0:
                    # --- FIX: Explicit copy for Run 1 ---
                    source_path = self.data_cache["step1"]["combined_path"]
                    new_input_path = f"{self.input_base_name}_step2_run1_init_path.xyz"
                    shutil.copy(source_path, new_input_path)
                    print(f"  Copied '{source_path}' to '{new_input_path}' for Run 1.")
                    return {"input_files": [new_input_path]}
                else:
                    # --- FIX: Explicit copy for Run 2 and beyond ---
                    if not previous_runs_this_step:
                         raise RuntimeError(f"Step 2 (sequential), run {run_index+1}: Cannot start, previous run (0) failed.")
                    prev_result = previous_runs_this_step[-1]
                    
                    if "final_relaxed_path" not in prev_result or not prev_result["final_relaxed_path"]:
                         raise RuntimeError(f"Step 2 (sequential), run {run_index+1}: Previous run produced no 'final_relaxed_path' to refine.")
                    
                    source_path = prev_result["final_relaxed_path"]
                    new_input_path = f"{self.input_base_name}_step2_run{run_index+1}_init_path.xyz"
                    shutil.copy(source_path, new_input_path)
                    print(f"  Copied '{source_path}' to '{new_input_path}' for Run {run_index+1}.")
                    return {"input_files": [new_input_path]}

            elif mode == "independent":
                # Independent mode still uses the combined path directly
                return {"input_files": [self.data_cache["step1"]["combined_path"]]}
            else:
                 raise ValueError(f"Step 2: Unknown mode '{mode}'. Use 'sequential' or 'independent'.")
        
        elif step_name == "step3":
            if "step2" not in self.data_cache or "candidates" not in self.data_cache["step2"]:
                raise RuntimeError("Step 3: data_cache[\"step2\"][\"candidates\"] not found. Did Step 2 run and post-process?")
            return {"input_files": self.data_cache["step2"]["candidates"]}

        elif step_name == "step4":
            if self.skip_to_step4:
                return {"input_files": [self.initial_mol_file]}
            
            if "step3" not in self.data_cache or "ts_final" not in self.data_cache["step3"]:
                raise RuntimeError("Step 4: data_cache[\"step3\"][\"ts_final\"] not found. Did Step 3 run and post-process?")
                
            return {"input_files": self.data_cache["step3"]["ts_final"]}
        else:
            print(f"Warning: Input determination logic not implemented for {step_name}. Using 'initial_mol_file'.")
            return {"input_file": self.initial_mol_file}

    def _run_post_processing(self, step_name, wf_entry):
        """
        Runs the post-processing logic (merge, select, consolidate).
        """
        print(f"  Post-processing results for {step_name}...")
        runs_list = self.data_cache[step_name]["runs"]
        mode = wf_entry.get("mode", "sequential") 
        if not runs_list:
            print(f"  No successful runs found for {step_name}. Skipping post-processing.")
            return

        if step_name == "step1":
            traj_files = [run["traj_file"] for run in runs_list if "traj_file" in run]
            if not traj_files:
                print("  Step 1: No traj_files found in any run. Cannot create combined_path.")
                return
            combined_path = self.merge_paths(traj_files)
            self.data_cache[step_name]["combined_path"] = combined_path
            print(f"  Step 1: Created combined path: {combined_path}")

        elif step_name == "step2":
            if mode == "sequential":
                print("  Step 2 (Sequential mode): Using candidates from the *last* run only for Step 3.")
                if not runs_list: 
                     print("  Step 2 (Sequential mode): No runs found, candidates list is empty.")
                     all_candidates_flat = []
                     energy_csvs = []
                else:
                    last_run_result = runs_list[-1]
                    all_candidates_flat = last_run_result.get("candidates", [])
                    energy_csvs = [last_run_result["energy_csv_path"]] if last_run_result.get("energy_csv_path") else []
            
            else:# independent
                all_candidates_flat = [
                    path for run in runs_list for path in run.get("candidates", [])
                ]
                energy_csvs = [
                    run["energy_csv_path"] for run in runs_list if run.get("energy_csv_path")
                ]
            
            if not all_candidates_flat:
                print("  Step 2: No candidates found in any run. Final list is empty.")
                self.data_cache[step_name]["candidates"] = []
                return

            top_n = self.config.get("top_n_candidates", 3)
            final_candidates = self.select_candidates(all_candidates_flat, energy_csvs, top_n)
            
            refinement_dir = f"{self.input_base_name}_step3_TS_Opt_Inputs"
            os.makedirs(refinement_dir, exist_ok=True)
            copied_candidates = []
            for i, source_path in enumerate(final_candidates):
                local_guess_name = f"{self.input_base_name}_ts_guess_{i+1}.xyz"
                local_path = os.path.join(refinement_dir, local_guess_name)
                shutil.copy(source_path, local_path)
                copied_candidates.append(local_path)

            self.data_cache[step_name]["candidates"] = copied_candidates
            print(f"  Step 2: Selected top {len(copied_candidates)} candidates -> {refinement_dir}")

        elif step_name == "step3":
            final_ts_list = self.consolidate_ts(runs_list)
            self.ts_final_files = final_ts_list 
            self.data_cache[step_name]["ts_final"] = final_ts_list
            print(f"  Step 3: Consolidated results into {len(final_ts_list)} final TS files.")
        
        elif step_name == "step4":
            print(f"  Step 4: Post-processing complete (results saved by each run).")

    # --- 3. Helper Functions (Consolidation Logic) ---

    def merge_paths(self, traj_files):
        """
        Merges results from Step 1 (sequential) runs.
        
        MODIFIED: This now concatenates all trajectory files from the
        sequential runs (run1_traj.xyz + run2_traj.xyz + ...) into a 
        single combined trajectory file.
        """
        if not traj_files:
            raise ValueError("merge_paths called with no trajectory files.")
        
        merged_path = f"{self.input_base_name}_step1_combined_traj.xyz"
        
        print(f"    (merge_paths): Concatenating {len(traj_files)} trajectory files into '{merged_path}'...")

        try:
            with open(merged_path, 'wb') as outfile:
                for i, traj_file in enumerate(traj_files):
                    if not os.path.exists(traj_file):
                        print(f"    Warning: Trajectory file not found, skipping: {traj_file}")
                        continue
                    
                    print(f"    -> Appending file {i+1}/{len(traj_files)}: {traj_file}")
                    with open(traj_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
        
        except IOError as e:
            print(f"    Error during trajectory concatenation: {e}")
            raise RuntimeError(f"Failed to merge trajectory files into {merged_path}")

        print(f"    (merge_paths): Concatenation complete.")
        # --- FIX: Return relative path ---
        return merged_path

    def select_candidates(self, all_candidates_flat, energy_csvs, top_n):
        """
        Selects the 'top_n' best candidates from all Step 2 runs.
        """
        print(f"    (select_candidates): Filtering {len(all_candidates_flat)} total candidates down to {top_n}.")
        
        if not energy_csvs:
            print(f"    Warning: No energy_plot.csv files found. Returning first {top_n} candidates (unsorted).")
            return all_candidates_flat[:top_n]
            
        best_csv_path = None
        max_energies = -1
        all_energies = []
        
        for csv_path in energy_csvs:
            try:
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    if not lines: continue
                    last_line = lines[-1].strip()
                energies = np.array([float(e) for e in last_line.split(',') if e.strip()])
                if len(energies) > max_energies:
                    max_energies = len(energies)
                    all_energies = energies
                    best_csv_path = csv_path
            except Exception:
                continue 
        
        if best_csv_path is None:
             print("    Warning: Failed to parse any energy_plot.csv file. Returning unsorted candidates.")
             return all_candidates_flat[:top_n]

        print(f"    Using energy reference: {best_csv_path} (found {max_energies} energies)")
        
        candidates_with_energy = []
        for path in all_candidates_flat:
            try:
                base_name = os.path.splitext(os.path.basename(path))[0]
                index_str = base_name.split('_')[-1]
                z = int(index_str) # 1-based
                z_idx = z - 1 # 0-based index
                
                if z_idx < 0 or z_idx >= len(all_energies):
                    print(f"    Warning: Index {z} from '{path}' out of bounds for energy list (len {len(all_energies)}). Skipping.")
                    continue
                candidates_with_energy.append((all_energies[z_idx], path))
            except Exception as e:
                print(f"    Warning: Could not parse index from '{path}': {e}. Skipping.")

        candidates_with_energy.sort(key=lambda x: x[0], reverse=True) 
        top_n_list = candidates_with_energy[:top_n]
        
        selected_paths = [path for energy, path in top_n_list]
        for energy, path in top_n_list:
            print(f"    - Selected: {os.path.basename(path)} (Energy: {energy:.6f} Hartree)")
            
        return selected_paths

    def consolidate_ts(self, runs_list):
        """
        Consolidates results from multiple Step 3 runs.
        Adopts the results from the *last* run.
        """
        if not runs_list:
            return []
        
        last_run_results = runs_list[-1]
        final_files = last_run_results.get("optimized_ts_files", [])
        
        print(f"    (consolidate_ts): Adopting {len(final_files)} TS files from the *last* Step 3 run.")
        
        return final_files

    # --- 4. V2 Adapter Methods for _run_stepX ---

    def _run_step1(self, settings, input_data, run_index=0):
        """
        Runs a *single* Step 1 (AFIR) scan.
        (FIXED: Sets WORK_DIR in settings before set_options)
        """
        input_file = input_data["input_file"]
        print(f"    Running Step 1 (AFIR) on: {input_file}")
        
        if "manual_AFIR" not in settings:
             raise ValueError(f"Step 1 settings (run {run_index+1}) must contain 'manual_AFIR'.")

        job = OptimizationJob(input_file=input_file)
        
        # **FIX**: Modify settings dict to create a unique WORK_DIR
        base_work_dir = settings.get("WORK_DIR", ".") 
        settings["WORK_DIR"] = os.path.join(base_work_dir, f"step1_run_{run_index+1}")
        
        job.set_options(**settings)
        job.run()
        
        optimizer_instance = job.get_results()
        if optimizer_instance is None:
             raise RuntimeError(f"Step 1 (run {run_index+1}) failed to produce an optimizer instance.")
             
        optimizer_instance.get_result_file_path() 
        source_traj_path = optimizer_instance.traj_file
        source_final_struct = optimizer_instance.optimized_struct_file

        if not source_traj_path or not os.path.exists(source_traj_path):
            raise FileNotFoundError(f"Step 1 (run {run_index+1}) 'traj_file' not found at '{source_traj_path}'")
        if not source_final_struct or not os.path.exists(source_final_struct):
             raise FileNotFoundError(f"Step 1 (run {run_index+1}) 'optimized_struct_file' not found at '{source_final_struct}'")

        local_traj_name = f"{self.input_base_name}_step1_run{run_index+1}_traj.xyz"
        shutil.copy(source_traj_path, local_traj_name)
        local_final_name = f"{self.input_base_name}_step1_run{run_index+1}_final.xyz"
        shutil.copy(source_final_struct, local_final_name)
        
        print(f"    Step 1 (run {run_index+1}) results saved.")
        # --- FIX: Return relative paths ---
        return {
            "traj_file": local_traj_name,
            "final_struct_file": local_final_name
        }

    def _run_step2(self, settings, input_data, run_index=0):
        """
        Runs a *single* Step 2 (NEB) optimization.
        (FIXED: Sets WORK_DIR in settings)
        (FIXED: Returns 'last_itr_traj_file_path' as 'final_relaxed_path')
        """
        input_files = input_data["input_files"]
        if not input_files:
             raise ValueError(f"Step 2 (run {run_index+1}) received no input files.")
        print(f"    Running Step 2 (NEB) on {len(input_files)} input file(s).")
        
        job = NEBJob(input_files=input_files)
        
        # **FIX**: Modify settings dict to create a unique WORK_DIR
        base_work_dir = settings.get("WORK_DIR", ".")
        settings["WORK_DIR"] = os.path.join(base_work_dir, f"step2_run_{run_index+1}")
        
        job.set_options(**settings)
        job.run()

        neb_instance = job.get_results()
        if neb_instance is None:
            raise RuntimeError(f"Step 2 (run {run_index+1}) failed to produce an NEB instance.")

        neb_instance.get_result_file() 
        source_ts_paths = neb_instance.ts_guess_file_list
        energy_csv_path = os.path.join(neb_instance.config.NEB_FOLDER_DIRECTORY, "energy_plot.csv")
        
        # **FIX**: Get the final relaxed path based on user's attribute name
        final_relaxed_path = getattr(neb_instance, 'last_itr_traj_file_path', None)
        local_final_path = None # Initialize
        
        if not final_relaxed_path or not os.path.exists(final_relaxed_path):
            print(f"    Warning: 'last_itr_traj_file_path' not found or invalid: '{final_relaxed_path}'. Sequential refinement may fail.")
            final_relaxed_path = None
        else:
            local_final_path_name = f"{self.input_base_name}_step2_run{run_index+1}_final_path.xyz"
            shutil.copy(final_relaxed_path, local_final_path_name)
            # --- FIX: Store relative path ---
            local_final_path = local_final_path_name


        if not source_ts_paths:
            print(f"    Step 2 (run {run_index+1}) did not find any TS candidates.")
            return {
                "candidates": [], 
                "energy_csv_path": None,
                "final_relaxed_path": local_final_path
            }
        
        candidate_dir = f"{self.input_base_name}_step2_run{run_index+1}_candidates"
        os.makedirs(candidate_dir, exist_ok=True)
        local_ts_paths = []
        
        for i, source_path in enumerate(source_ts_paths):
            if not os.path.exists(source_path):
                print(f"    Warning: Source file not found, skipping: {source_path}")
                continue
            local_guess_name = f"{self.input_base_name}_s2_run{run_index+1}_guess_{i+1}.xyz"
            local_path = os.path.join(candidate_dir, local_guess_name)
            shutil.copy(source_path, local_path)
            # --- FIX: Store relative path ---
            local_ts_paths.append(local_path)
        
        print(f"    Step 2 (run {run_index+1}) found {len(local_ts_paths)} candidates.")
        return {
            "candidates": local_ts_paths,
            # --- FIX: Store relative path ---
            "energy_csv_path": energy_csv_path if os.path.exists(energy_csv_path) else None,
            "final_relaxed_path": local_final_path
        }

    def _run_step3(self, settings, input_data, run_index=0):
        """
        Runs a *single* Step 3 (TS Refinement) pass.
        (FIXED: Sets WORK_DIR in settings)
        """
        candidate_files = input_data["input_files"]
        print(f"    Running Step 3 (TS Refine) on {len(candidate_files)} candidates (Run {run_index+1}).")
        
        if not candidate_files:
            print("    No candidates provided. Skipping refinement.")
            return {"optimized_ts_files": [], "energies": {}}
            
        final_ts_files = []
        final_ts_energies = {}
        settings['saddle_order'] = 1
        base_work_dir = settings.get("WORK_DIR", ".") # Get base dir once

        for i, guess_file_path in enumerate(candidate_files):
            print(f"    Refining candidate {i+1}/{len(candidate_files)} ({os.path.basename(guess_file_path)})")
            
            job = OptimizationJob(input_file=guess_file_path)
            
            # **FIX**: Modify settings dict for this specific candidate
            current_settings = copy.deepcopy(settings)
            cand_basename = os.path.splitext(os.path.basename(guess_file_path))[0]
            current_settings["WORK_DIR"] = os.path.join(base_work_dir, f"step3_run_{run_index+1}", cand_basename)
            
            job.set_options(**current_settings)
            job.run()
            
            optimizer_instance = job.get_results()
            if optimizer_instance is None:
                print(f"    Warning: Refinement for {guess_file_path} failed.")
                continue

            optimizer_instance.get_result_file_path()
            source_final_ts_path = optimizer_instance.optimized_struct_file
            if not source_final_ts_path or not os.path.exists(source_final_ts_path):
                print(f"    Warning: Refinement for {guess_file_path} finished, but 'optimized_struct_file' was not found.")
                continue
            
            if not optimizer_instance.optimized_flag:
                print(f"Warning: Refinement for {guess_file_path} did not converge (optimized_flag=False). Skipping.")
                continue
            
            local_final_name = f"{self.input_base_name}_s3_run{run_index+1}_ts_final_{i+1}.xyz"
            shutil.copy(source_final_ts_path, local_final_name)
            
            # --- FIX: Store relative path ---
            rel_path = local_final_name
            final_ts_files.append(rel_path)
            final_ts_energies[rel_path] = optimizer_instance.final_energy

        print(f"    Step 3 (run {run_index+1}) successfully refined {len(final_ts_files)} structures.")
        return {
            "optimized_ts_files": final_ts_files,
            "energies": final_ts_energies
        }

    def _run_step4(self, settings, input_data, run_index=0):
        """
        Runs a *single* Step 4 (IRC + Opt) pass.
        (FIXED: Sets WORK_DIR in settings)
        """
        ts_final_files = input_data["input_files"]
        print(f"     Running Step 4 (IRC) on {len(ts_final_files)} TS files (Run {run_index+1}).")
        
        if not ts_final_files:
            print("     No TS files provided. Skipping Step 4.")
            return {"profile_dirs": []}
            
        if "intrinsic_reaction_coordinates" not in settings:
            raise ValueError(f"Step 4 (run {run_index+1}) requires 'intrinsic_reaction_coordinates' settings.")

        profile_dirs = []
        base_work_dir = settings.get("WORK_DIR", ".")
        
        for i, ts_path in enumerate(ts_final_files):
            ts_name_base = f"{self.input_base_name}_s4_run{run_index+1}_TS_{i+1}"
            print(f"     Running Step 4 for TS {i+1}/{len(ts_final_files)} ({ts_path})")
            
            # --- 4A: Run IRC ---
            
           
            relative_ts_path = os.path.relpath(ts_path)
            job_irc = OptimizationJob(input_file=relative_ts_path)
            
            irc_settings = copy.deepcopy(settings)
            irc_settings["saddle_order"] = 1
            # **FIX**: Set WORK_DIR in settings
            irc_settings["WORK_DIR"] = os.path.join(base_work_dir, f"step4_run_{run_index+1}", f"ts_{i+1}_irc")
            job_irc.set_options(**irc_settings)
            job_irc.run()

            irc_instance = job_irc.get_results()
            if irc_instance is None:
                print(f"     Warning: IRC job for {ts_path} failed.")
                continue
                
            ts_e = irc_instance.final_energy
            ts_bias_e = irc_instance.final_bias_energy
            endpoint_paths = irc_instance.irc_terminal_struct_paths

            if not endpoint_paths or len(endpoint_paths) != 2:
                print(f"     Warning: IRC job for {ts_path} did not return 2 endpoint files.")
                continue

            # --- 4B: Run Endpoint Optimization ---
            endpoint_results = []
            for j, end_path in enumerate(endpoint_paths):
                opt_settings = copy.deepcopy(settings)
                opt_settings["opt_method"] = opt_settings.get("opt_method", ["rsirfo_block_fsb"])
                opt_settings.pop("intrinsic_reaction_coordinates", None)
                opt_settings['saddle_order'] = 0 # Minimization
                
               
                relative_end_path = os.path.relpath(end_path)
                base_end_path = os.path.basename(relative_end_path)
                shutil.copy(relative_end_path, base_end_path)
                job_opt = OptimizationJob(input_file=base_end_path)
                
                # **FIX**: Set WORK_DIR in settings
                opt_settings["WORK_DIR"] = os.path.join(base_work_dir, f"step4_run_{run_index+1}", f"ts_{i+1}_endpt_{j+1}")
                job_opt.set_options(**opt_settings)
                job_opt.run()
                
                opt_instance = job_opt.get_results()
                if opt_instance is None: continue
                
                opt_instance.get_result_file_path()
                final_opt_path = opt_instance.optimized_struct_file
                if not final_opt_path or not os.path.exists(final_opt_path): continue
                
                endpoint_results.append({
                    "path": final_opt_path,
                    "e": opt_instance.final_energy,
                    "bias_e": opt_instance.final_bias_energy,
                })

            if not endpoint_results:
                print(f"     Warning: Failed to optimize any endpoints for {ts_path}.")
                continue

            # --- 4C: Collect Results & Visualize (using parent methods) ---
            result_dir = f"{ts_name_base}_Step4_Profile"
            os.makedirs(result_dir, exist_ok=True)
            
            e_profile = {"TS": {"e": ts_e, "bias_e": ts_bias_e, "path": ts_path}}
            if len(endpoint_results) >= 1: e_profile["End1"] = endpoint_results[0]
            if len(endpoint_results) >= 2: e_profile["End2"] = endpoint_results[1]
        
            plot_path = os.path.join(result_dir, "energy_profile.png")
            self._create_energy_profile_plot(e_profile, plot_path, ts_name_base)

            text_path = os.path.join(result_dir, "energy_profile.txt")
            self._write_energy_profile_text(e_profile, text_path, ts_name_base)
            
        
            shutil.copy(os.path.relpath(ts_path), os.path.join(result_dir, f"{ts_name_base}_ts_final.xyz"))
            if "End1" in e_profile: 
                shutil.copy(os.path.relpath(e_profile["End1"]["path"]), os.path.join(result_dir, "endpoint_1_opt.xyz"))
            if "End2" in e_profile: 
                shutil.copy(os.path.relpath(e_profile["End2"]["path"]), os.path.join(result_dir, "endpoint_2_opt.xyz"))
            
            print(f"     Successfully saved profile to: {result_dir}")
            # --- FIX: Store relative path ---
            profile_dirs.append(result_dir)
        
        return {"profile_dirs": profile_dirs}
        
        