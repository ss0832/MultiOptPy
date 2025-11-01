import sys
import argparse
import os
import json 

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multioptpy.Wrapper.autots import AutoTSWorkflow
except ImportError as e:
    print("Error: Could not import AutoTSWorkflow.")
    print(f"Details: {e}")
    sys.exit(1)

# ======================================================================
# AUTO-TS WORKFLOW CONFIGURATION GUIDE
# ======================================================================
# Default parameters are now loaded from 'config.json'.
#
# CRITICAL GUIDELINE:
# To understand or modify any option (e.g., 'opt_method', 'NSTEP'),
# always refer to the detailed help strings in 'multioptpy/interface.py':
# 
# 1. Step 1, 3, & 4 Settings: call_optimizeparser()
# 2. Step 2 Settings: call_nebparser()
# ======================================================================


def load_config_from_file(config_path):
    """Loads configuration settings from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Successfully loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file {config_path}. Check file format.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}")
        sys.exit(1)


def launch_workflow(config):
    """
    Launches the AutoTSWorkflow with a prepared configuration dictionary.
    This function is callable from other Python scripts.
    
    Args:
        config (dict): The complete configuration dictionary.
    """
    
    # --- 1. Apply settings from the config dict ---
    
    # Ensure 'software_path_file' is applied to all steps
    local_conf_name = os.path.basename(config.get("software_path_file_source", "software_path.conf"))
    for i in range(1, 5): # Steps 1, 2, 3, and 4
        step_key = f"step{i}_settings"
        if step_key in config:
            config[step_key]["software_path_file"] = local_conf_name
        elif i == 4 and "step4_settings" not in config: # Create step4 settings if not in config
             config["step4_settings"] = {"software_path_file": local_conf_name}

    # --- 2. Print Summary ---
    print("--- AutoTS Workflow Starting ---")
    print(f"Input File: {config.get('initial_mol_file')}")
    print(f"Skip Step 1: {config.get('skip_step1', False)}")
    print(f"Skip to Step 4: {config.get('skip_to_step4', False)}")
    print(f"Run Step 4 (IRC): {config.get('run_step4', False)}")
    
    if not config.get('skip_step1', False) and not config.get('skip_to_step4', False):
        print(f"AFIR Params: {config.get('step1_settings', {}).get('manual_AFIR')}")
    
    # --- 3. Create and Run the Workflow ---
    workflow = AutoTSWorkflow(config=config)
    workflow.run_workflow()


def main():
    """
    Main function for command-line execution.
    Parses CMD arguments, loads config, merges them, and calls launch_workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run the Automated Transition State (AutoTS) workflow."
    )
    
    # --- (Parser arguments remain the same) ---
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the initial structure file. If --skip_to_step4 is used, this must be the TS file."
    )
    parser.add_argument(
        "-cfg", "--config_file",
        type=str,
        default="./config.json",
        help="Path to the configuration JSON file. Default is './config.json'."
    )
    parser.add_argument(
        "-ma", "--manual_AFIR",
        nargs="*",
        required=False,
        help="Manual AFIR parameters for Step 1 (e.g., 150 6 8 150 1 9)"
    )
    parser.add_argument(
        "-osp", "--software_path_file",
        type=str,
        default="./software_path.conf",
        help="Path to the 'software_path.conf' file. Defaults to './software_path.conf'"
    )
    parser.add_argument(
        "-n", "--top_n",
        type=int,
        default=None, # Default will be read from JSON
        help="Refine the top N highest energy candidates from NEB. Overrides config file."
    )
    parser.add_argument(
        "--skip_step1",
        action="store_true",
        help="Skip the AFIR scan (Step 1). The input_file must be the NEB trajectory file."
    )
    parser.add_argument(
        "--run_step4",
        action="store_true",
        help="Run Step 4 (IRC + Endpoint Optimization) after Step 3 completes."
    )
    parser.add_argument(
        "--skip_to_step4",
        action="store_true",
        help="Skip Steps 1-3 and run only Step 4. The 'input_file' must be the TS structure file."
    )
    
    args = parser.parse_args()

    # --- 1. Load Base Configuration from File ---
    workflow_config = load_config_from_file(args.config_file)
    
    # --- 2. Override Config with CMD Arguments ---
    workflow_config["initial_mol_file"] = args.input_file
    workflow_config["software_path_file_source"] = os.path.abspath(args.software_path_file)
    workflow_config["skip_step1"] = args.skip_step1
    workflow_config["run_step4"] = args.run_step4
    workflow_config["skip_to_step4"] = args.skip_to_step4

    if args.top_n is not None:
        workflow_config["top_n_candidates"] = args.top_n

    # --- Validation ---
    if not args.skip_step1 and not args.skip_to_step4 and not args.manual_AFIR:
        print("\nError: The -ma/--manual_AFIR argument is required unless skipping Step 1 or Step 4.")
        sys.exit(1)

    if args.manual_AFIR:
        workflow_config["step1_settings"]["manual_AFIR"] = args.manual_AFIR
    elif not args.skip_step1 and not args.skip_to_step4:
        # Ensure 'manual_AFIR' key exists if running Step 1
        workflow_config.setdefault("step1_settings", {})["manual_AFIR"] = []

    # --- 3. Call the launcher function ---
    launch_workflow(workflow_config)


if __name__ == "__main__":
    main()

