
import sys
import argparse
import os

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multioptpy.Wrapper.autots import AutoTSWorkflow
except ImportError as e:
    print("Error: Could not import AutoTSWorkflow.")
    print(f"Details: {e}")
    sys.exit(1)

def get_default_config():
    """
    Returns the default configuration dictionary for the workflow.
    """
    # (Default settings for steps 1, 2, 3 remain the same...)
    config = {
        "work_dir": "autots_run",
        
        "step1_settings": {
            "othersoft": "uma-s-1p1",
            "opt_method": ["rsirfo_block_fsb"],
            "use_model_hessian": 'fischerd3'
        },
        "step2_settings": {
            "othersoft": "uma-s-1p1", "NSTEP": 15, "ANEB": [3, 5], "QSM": True,
            "use_model_hessian": 'fischerd3', "save_pict": True,
            "node_distance_bernstein": 0.75, "align_distances": 9999,
            "spin_multiplicity": 1, "electronic_charge": 0
        },
        "step3_settings": {
            "othersoft": "uma-s-1p1", "opt_method": ["rsirfo_block_bofill"],
            "calc_exact_hess": 5, "tight_convergence_criteria": True,
            "max_trust_radius": 0.2, "frequency_analysis": True
        }
    }
    return config

def main():
    parser = argparse.ArgumentParser(
        description="Run the Automated Transition State (AutoTS) workflow."
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the initial molecular structure file (e.g., molecule.xyz)"
    )
    
    parser.add_argument(
        "-ma", "--manual_AFIR",
        nargs="*",
        required=True,
        help="Manual AFIR parameters for Step 1 (e.g., 150 6 8 150 1 9)"
    )
    
    parser.add_argument(
        "-osp", "--software_path_file",
        type=str,
        default="./software_path.conf",
        help="Path to the 'software_path.conf' file. Defaults to './software_path.conf'"
    )
    
    # --- ADDED ARGUMENT FOR TOP N ---
    parser.add_argument(
        "-n", "--top_n",
        type=int,
        default=3,
        help="Refine the top N highest energy candidates from NEB. Default is 3."
    )
    
    args = parser.parse_args()

    # --- 1. Get Default Configuration ---
    workflow_config = get_default_config()
    
    # --- 2. Override Config with CMD Arguments ---
    workflow_config["initial_mol_file"] = args.input_file
    workflow_config["step1_settings"]["manual_AFIR"] = args.manual_AFIR
    workflow_config["software_path_file_source"] = os.path.abspath(args.software_path_file)
    
    # --- ADDED: Store Top N ---
    workflow_config["top_n_candidates"] = args.top_n

    local_conf_name = os.path.basename(args.software_path_file)
    workflow_config["step1_settings"]["software_path_file"] = local_conf_name
    workflow_config["step2_settings"]["software_path_file"] = local_conf_name
    workflow_config["step3_settings"]["software_path_file"] = local_conf_name

    print("--- AutoTS Workflow Starting ---")
    print(f"Input File: {workflow_config['initial_mol_file']}")
    print(f"AFIR Params: {workflow_config['step1_settings']['manual_AFIR']}")
    print(f"Software Conf: {workflow_config['software_path_file_source']}")
    print(f"Refining Top {workflow_config['top_n_candidates']} NEB candidates.")
    
    # --- 3. Create and Run the Workflow ---
    workflow = AutoTSWorkflow(config=workflow_config)
    workflow.run_workflow()

if __name__ == "__main__":
    main()