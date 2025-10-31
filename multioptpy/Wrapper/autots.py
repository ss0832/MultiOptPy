
import os
import shutil
import numpy as np

from multioptpy.Wrapper.optimize_wrapper import OptimizationJob
from multioptpy.Wrapper.neb_wrapper import NEBJob


class AutoTSWorkflow:
    """
    Manages the 3-step (AFIR -> NEB -> TS) automated workflow.
    """
    def __init__(self, config):
        self.config = config
        self.work_dir = config.get("work_dir", "autots_workflow")
        self.initial_mol_file = config.get("initial_mol_file")
        self.conf_file_source = config.get("software_path_file_source")
        # Store Top N, defaulting to 3
        self.top_n_candidates = config.get("top_n_candidates", 3)

    def setup_workspace(self):
        """Prepares the working directory and copies necessary input files."""
        if os.path.exists(self.work_dir):
            print(f"Warning: Working directory '{self.work_dir}' already exists.")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # 1. Copy molecule file
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
            
        # 2. Copy conf file
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

        # 3. Change directory
        os.chdir(self.work_dir)
        print(f"Changed directory to: {os.getcwd()}")
        

    def _run_step1_afir_scan(self):
        """Runs the AFIR scan and copies the resulting trajectory."""
        print("\n--- 1. STARTING STEP 1: AFIR SCAN ---")
        job1_settings = self.config.get("step1_settings", {})
            
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

        # Use the input file base name for consistent naming
        self.input_base_name = os.path.splitext(os.path.basename(self.initial_mol_file))[0]
        local_traj_name = f"{self.input_base_name}_step1_afir.xyz"
        
        shutil.copy(source_traj_path, local_traj_name)
        
        print(f"Copied AFIR trajectory to: {os.path.abspath(local_traj_name)}")
        print("--- STEP 1: AFIR SCAN COMPLETE ---")
        return local_traj_name 

    # --- THIS METHOD IS MODIFIED ---
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

        # This populates neb_instance.ts_guess_file_list
        neb_instance.get_result_file() 
        source_ts_paths = neb_instance.ts_guess_file_list

        # --- MODIFIED LOGIC: Handle 0 candidates ---
        if not source_ts_paths:
            print("Step 2 (NEB) did not find any TS candidate files (ts_guess_file_list is empty).")
            return [] # Return empty list

        # --- MODIFIED LOGIC: Filter by Top N energy ---
        # Find the energy.csv file path from the instance
        energy_csv_path = os.path.join(neb_instance.config.NEB_FOLDER_DIRECTORY, "energy_plot.csv")
        
        selected_paths = self._filter_candidates_by_energy(
            source_ts_paths, energy_csv_path
        )

        # --- Copy the selected top N candidates ---
        refinement_dir = f"{self.input_base_name}_03_TS_Refinement_Inputs"
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

    # --- NEW HELPER FUNCTION ---
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
            
            # Get the final energy profile
            energies = np.array([float(e) for e in last_line.split(',') if e.strip()])
            
        except Exception as e:
            print(f"Warning: Could not read or parse energy file '{energy_csv_path}': {e}")
            print("Proceeding with all found candidates (unsorted).")
            # Return up to N candidates, unsorted
            return file_paths[:self.top_n_candidates]

        candidates = []
        for path in file_paths:
            try:
                # Extract the index 'z' from the filename (e.g., ..._5.xyz)
                base_name = os.path.splitext(os.path.basename(path))[0]
                index_str = base_name.split('_')[-1]
                z = int(index_str)
                
                if z >= len(energies):
                    print(f"Warning: Index {z} from file '{path}' is out of bounds for energy list (len {len(energies)}).")
                    continue
                    
                # Store (energy, path)
                candidates.append((energies[z], path))
                
            except Exception as e:
                print(f"Warning: Could not parse index from '{path}': {e}. Skipping.")

        # Sort by energy (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Get the top N candidates
        top_n_list = candidates[:self.top_n_candidates]
        
        print(f"Identified {len(candidates)} candidates, selecting top {len(top_n_list)}:")
        for (energy, path) in top_n_list:
            print(f"  - Path: {os.path.basename(path)}, Energy: {energy:.6f} Hartree")
            
        # Return just the file paths
        return [path for energy, path in top_n_list]


    def _run_step3_ts_refinement(self, local_ts_guess_paths):
        """Runs saddle_order=1 OptimizationJob on all local candidates."""
        print("\n--- 3. STARTING STEP 3: TS REFINEMENT ---")
        
        if not local_ts_guess_paths:
            # This check is now redundant because of run_workflow, but safe
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
                
            # Use the base name for the final output file
            local_final_name = f"{self.input_base_name}_ts_final_{i+1}.xyz"
            shutil.copy(source_final_ts_path, local_final_name)
            
            print(f"Copied final TS structure to: {os.path.abspath(local_final_name)}")
            final_ts_files.append(local_final_name)

        print("\n--- STEP 3: TS REFINEMENT COMPLETE ---")
        return final_ts_files

    # --- THIS METHOD IS MODIFIED ---
    def run_workflow(self):
        """Executes the full 3-step workflow."""
        original_cwd = os.getcwd()
        try:
            if not os.path.exists(self.initial_mol_file):
                raise FileNotFoundError(f"Initial molecule file not found: {self.initial_mol_file}")
            
            self.setup_workspace() 
            
            # Step 1: Run AFIR
            local_afir_traj = self._run_step1_afir_scan()
            
            # Step 2: Run NEB and filter Top N
            local_ts_paths = self._run_step2_neb_optimization(local_afir_traj)
            
            # --- MODIFIED LOGIC: Handle 0 candidates ---
            if not local_ts_paths:
                print("Step 2 found 0 candidates. Workflow terminated as requested.")
                print(f"\n✅ --- AUTO-TS WORKFLOW COMPLETED (NO TS FOUND) --- ✅")
                print(f"All results are in: {os.path.realpath(os.getcwd())}")
                return # Stop workflow

            # Step 3: Run Refinement
            self._run_step3_ts_refinement(local_ts_paths)
            
            print(f"\n --- AUTO-TS WORKFLOW COMPLETED SUCCESSFULLY --- ")
            print(f"All results are in: {os.path.realpath(os.getcwd())}")

        except Exception as e:
            print(f"\n --- AUTO-TS WORKFLOW FAILED --- ")
            print(f"Error: {e}")
        finally:
            os.chdir(original_cwd)
            print(f"Returned to directory: {original_cwd}")
            
            