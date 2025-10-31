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

# ======================================================================
# AUTO-TS WORKFLOW CONFIGURATION GUIDE
# ======================================================================
# This section defines the default parameters for the 3-step workflow
# (AFIR Scan -> NEB Opt -> TS Refinement).
#
# CRITICAL GUIDELINE:
# To understand or modify any option (e.g., 'opt_method', 'NSTEP', 'calc_exact_hess'),
# always refer to the detailed help strings in the following functions 
# within 'multioptpy/interface.py':
# 
# 1. Step 1 (AFIR Scan) & Step 3 (TS Refinement): call_optimizeparser()
# 2. Step 2 (NEB Optimization): call_nebparser()
# 
# Parameter Hierarchy:
# Command Line Argument > workflow_config > multioptpy Default
# The 'manual_AFIR', 'input_file', and 'software_path_file' are always 
# controlled by command-line arguments.
# ======================================================================

def get_default_config():
    """
    Returns the default configuration dictionary for the workflow,
    including comments based on interface.py for future adjustment.
    """
    config = {
        "work_dir": "autots_run", # Working directory name for the job

        # --- Step 1: AFIR Scan (OptimizationJob) ---
        "step1_settings": {
            # Use other QM software (e.g., orca, gaussian, uma-s-1p1)
            "othersoft": "uma-s-1p1", 
            # Optimization method for QM calculation (e.g., FIRELARS, rsirfo_block_fsb)
            "opt_method": ["rsirfo_block_fsb"], 
            # Use model hessian (Default: fischerd3)
            "use_model_hessian": 'fischerd3',
            # Spin multiplicity (S=2*M-1)
            "spin_multiplicity": 1, 
            # Formal electronic charge
            "electronic_charge": 0 
            # 'manual_AFIR' is set via CMD arg
        },

        # --- Step 2: NEB Optimization (NEBJob) ---
        "step2_settings": {
            # Use other QM software
            "othersoft": "uma-s-1p1", 
            # Iteration number
            "NSTEP": 15,
            # Adaptic NEB method: [interpolation_num, frequency]
            "ANEB": [3, 5],
            # Use Quadratic String Method (QSM)
            "QSM": True,
            # Use model hessian (Default: fischerd3)
            "use_model_hessian": 'fischerd3', 
            # Save picture for visualization (bool)
            "save_pict": True,
            # Distribute images using Bernstein interpolation based on specific distance (ang.)
            "node_distance_bernstein": 0.75, 
            # Distribute images at equal intervals on the reaction coordinate (0=default)
            "align_distances": 9999, 
            # Spin multiplicity
            "spin_multiplicity": 1, 
            # Formal electronic charge
            "electronic_charge": 0
        },

        # --- Step 3: TS Refinement (OptimizationJob) ---
        "step3_settings": {
            # Use other QM software
            "othersoft": "uma-s-1p1", 
            # Optimization method for QM calculation (e.g., rsirfo_block_bofill)
            "opt_method": ["rsirfo_block_bofill"],
            # Calculate exact hessian per steps (e.g., 5 steps)
            "calc_exact_hess": 5, 
            # Apply tight opt criteria (bool)
            "tight_convergence_criteria": True,
            # Max trust radius to restrict step size (unit: ang.)
            "max_trust_radius": 0.2, 
            # Perform normal vibrational analysis after converging geometry optimization
            "frequency_analysis": True,
            # Spin multiplicity
            "spin_multiplicity": 1,
            # Formal electronic charge
            "electronic_charge": 0
        }
        # Note: 'saddle_order': 1 is set automatically by AutoTSWorkflow._run_step3_ts_refinement
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