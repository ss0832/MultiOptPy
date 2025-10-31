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
# 1. Step 1 (AFIR Scan) & Step 3 (TS Refinement): call_optimizeparser()
# 2. Step 2 (NEB Optimization): call_nebparser()
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the Automated Transition State (AutoTS) workflow."
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the initial molecular structure file. If --skip_step1 is used, this file must be the NEB trajectory."
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
        required=False, # No longer required if --skip_step1 is used
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
    
    # --- ADDED FLAG: Skip Step 1 ---
    parser.add_argument(
        "--skip_step1",
        action="store_true",
        help="Skip the AFIR scan (Step 1). The input_file must be the NEB trajectory file."
    )
    
    args = parser.parse_args()

    # --- 1. Load Configuration from File ---
    workflow_config = load_config_from_file(args.config_file)
    
    # --- 2. Override Config with CMD Arguments ---
    workflow_config["initial_mol_file"] = args.input_file
    workflow_config["software_path_file_source"] = os.path.abspath(args.software_path_file)
    workflow_config["skip_step1"] = args.skip_step1

    # Override top_n only if explicitly set on command line
    if args.top_n is not None:
        workflow_config["top_n_candidates"] = args.top_n

    # Validation: Check for required AFIR args if not skipping
    if not args.skip_step1 and not args.manual_AFIR:
        print("\nError: The -ma/--manual_AFIR argument is required unless --skip_step1 is used.")
        sys.exit(1)

    if args.manual_AFIR:
        workflow_config["step1_settings"]["manual_AFIR"] = args.manual_AFIR
    elif not args.skip_step1:
        # Ensure the key exists if we are running step 1 but -ma wasn't provided
        # (This case is caught by the validation above, but defensive programming)
        workflow_config["step1_settings"]["manual_AFIR"] = []


    local_conf_name = os.path.basename(args.software_path_file)
    workflow_config["step1_settings"]["software_path_file"] = local_conf_name
    workflow_config["step2_settings"]["software_path_file"] = local_conf_name
    workflow_config["step3_settings"]["software_path_file"] = local_conf_name

    print("--- AutoTS Workflow Starting ---")
    print(f"Input File: {workflow_config['initial_mol_file']}")
    print(f"Skip Step 1: {workflow_config['skip_step1']}")
    if not args.skip_step1:
        print(f"AFIR Params: {workflow_config['step1_settings']['manual_AFIR']}")
    print(f"Software Conf: {workflow_config['software_path_file_source']}")
    print(f"Refining Top {workflow_config['top_n_candidates']} NEB candidates.")
    
    # --- 3. Create and Run the Workflow ---
    workflow = AutoTSWorkflow(config=workflow_config)
    workflow.run_workflow()

if __name__ == "__main__":
    main()