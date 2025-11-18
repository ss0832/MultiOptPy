import sys
import argparse
import os
import json 

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import both v1 and v2 workflow classes
    from multioptpy.Wrapper.autots import AutoTSWorkflow
    # Assuming v2 is added to the same autots.py file
    from multioptpy.Wrapper.autots import AutoTSWorkflow_v2
except ImportError as e:
    print("Error: Could not import AutoTSWorkflow or AutoTSWorkflow_v2.")
    print("       Ensure autots.py contains both classes.")
    print(f"Details: {e}")
    sys.exit(1)

# ======================================================================
# AUTO-TS WORKFLOW (V1/V2) CONFIGURATION GUIDE
# ======================================================================
# Config is loaded from 'config.json'.
#
# V1 (Legacy): Uses top-level keys like 'step1_settings', 'skip_step1'.
# V2 (Dynamic): Uses the "workflow": [...] block to define execution.
#
# CRITICAL GUIDELINE:
# To understand options (e.g., 'opt_method', 'NSTEP'),
# refer to 'multioptpy/interface.py':
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
    Launches the AutoTSWorkflow (v1 or v2) based on config.
    
    Args:
        config (dict): The complete configuration dictionary.
    """
    
    # --- 1. Apply settings (for v1 compatibility) ---
    # This is still useful for v2 if 'stepX_settings' are used as base keys.
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
    
    # --- 3. NEW: Dynamic v1/v2 Selection ---
    if "workflow" in config:
        print(">>> Detected 'workflow' key. Initializing AutoTSWorkflow_v2.")
        workflow = AutoTSWorkflow_v2(config=config)
    else:
        print(">>> No 'workflow' key found. Initializing AutoTSWorkflow (v1).")
        # Print v1-specific flags
        print(f"Skip Step 1: {config.get('skip_step1', False)}")
        print(f"Skip to Step 4: {config.get('skip_to_step4', False)}")
        print(f"Run Step 4 (IRC): {config.get('run_step4', False)}")
        
        if not config.get('skip_step1', False) and not config.get('skip_to_step4', False):
            print(f"AFIR Params: {config.get('step1_settings', {}).get('manual_AFIR')}")
        
        workflow = AutoTSWorkflow(config=config)

    # --- 4. Run the selected workflow ---
    # Both v1 and v2 classes have a .run_workflow() method
    workflow.run_workflow()

def main():
    """
    Main function for command-line execution.
    Parses CMD arguments, loads config, merges them, and calls launch_workflow.
    (Modified for v1/v2 compatibility)
    """
    parser = argparse.ArgumentParser(
        description="Run the Automated Transition State (AutoTS) workflow (v1 or v2)."
    )
    
    # --- (Parser arguments remain the same) ---
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the initial structure file. If --skip_to_step4 is used (v1), this must be the TS file."
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
        help="Manual AFIR parameters for Step 1. Overrides config file's 'step1_settings'."
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
    
    # --- V1-specific flags ---
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
    # These apply to both v1 and v2
    workflow_config["initial_mol_file"] = args.input_file
    workflow_config["software_path_file_source"] = os.path.abspath(args.software_path_file)
    
    # Merge V1-specific CMD flags (v2 will ignore them)
    workflow_config["skip_step1"] = args.skip_step1
    workflow_config["run_step4"] = args.run_step4
    workflow_config["skip_to_step4"] = args.skip_to_step4

    if args.top_n is not None:
        workflow_config["top_n_candidates"] = args.top_n

    # --- 3. AFIR Validation (v1/v2 compatibility logic) ---
    
    # Ensure 'step1_settings' key exists for v1 compatibility
    workflow_config.setdefault("step1_settings", {})
    
    # Check if v1 is running step 1
    is_v1_running_step1 = (
        "workflow" not in workflow_config and 
        not args.skip_step1 and 
        not args.skip_to_step4
    )
    
    # Check if v2 is running step 1
    is_v2_running_step1 = False
    if "workflow" in workflow_config:
        for entry in workflow_config.get("workflow", []):
            if entry.get("step") == "step1" and entry.get("enabled", True):
                is_v2_running_step1 = True
                break

    # Check AFIR status in config (base 'step1_settings' only) and CMD
    config_has_afir = workflow_config["step1_settings"].get("manual_AFIR")
    cmd_has_afir = args.manual_AFIR is not None

    if cmd_has_afir:
        # Case 1: CMD argument is given. It *always* overrides the base 'step1_settings'.
        workflow_config["step1_settings"]["manual_AFIR"] = args.manual_AFIR
        print(f"Using 'manual_AFIR' from command line (overrides 'step1_settings'): {args.manual_AFIR}")
        # This will be used by v1, or by v2 if 'step1_settings' is its base key.
    
    elif not config_has_afir:
        # Case 2: No AFIR in CMD, and *also* no AFIR in the base 'step1_settings'.
        
        if is_v1_running_step1:
            # For v1, this is a fatal error.
            print("\nError (v1 mode): 'manual_AFIR' is not defined in 'step1_settings' and was not provided via -ma.")
            print("       Please add 'manual_AFIR' to your JSON or use the -ma argument.")
            sys.exit(1)
            
        elif is_v2_running_step1:
            # For v2, this is a warning, as it might be defined in 'param_override'.
             print(f"Warning: 'manual_AFIR' not found in 'step1_settings' or via -ma.")
             print("       (v2 mode): Ensure 'manual_AFIR' is defined in your 'workflow' entry")
             print("       (either in the base 'settings_key' block or 'param_override').")
             
    elif is_v1_running_step1 and config_has_afir:
        # Case 3: v1 is running, no CMD override, but config has it. Just print confirmation.
        print(f"Using 'manual_AFIR' from config file: {config_has_afir}")

    # --- 4. Call the launcher function ---
    launch_workflow(workflow_config)

if __name__ == "__main__":
    main()