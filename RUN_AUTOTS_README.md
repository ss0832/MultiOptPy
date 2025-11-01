# **🚀 AutoTS Workflow (run\_autots.py) Execution Guide**

run\_autots.py is the primary launcher script for the **MultiOptPy** automated Transition State (TS) search workflow.

This script reads default settings from config.json and executes a 4-step workflow based on command-line arguments.

## **🧩 Workflow Overview**

### **Step 1: AFIR Scan (OptimizationJob)**

* Scans a reaction path from an initial structure based on the \-ma (manual\_AFIR) options.  
* Generates a trajectory file (e.g., \_step1\_result.xyz).

### **Step 2: NEB Optimization (NEBJob)**

* Uses the trajectory from Step 1 as an initial path for a Nudged Elastic Band (NEB) calculation.  
* Identifies potential TS candidates (energy maxima) along the path.

### **Step 3: TS Refinement (OptimizationJob)**

* Performs a saddle point optimization (saddle\_order=1) on the top N highest-energy candidates found in Step 2\.  
* Generates the final, refined TS structures (e.g., \_ts\_final\_1.xyz).

### **Step 4: IRC & Validation (Optional) (OptimizationJob)**

* **Step 4A (IRC):** Runs an IRC calculation starting from a refined TS structure to find the two connected endpoints.  
* **Step 4B (Endpoint Opt):** Runs a standard geometry optimization (saddle\_order=0) on the two endpoints found by the IRC.  
* **Step 4C (Collect):** Gathers the TS, End1, and End2 structures and energies into a dedicated directory and generates a reaction profile plot.

## **⚙️ Usage**

Execute the script from the **root directory** (same level as the multioptpy folder).

### **✅ Standard Run (Steps 1-3)**

python run\_autots.py \[input\_file\] \-ma \[AFIR\_params\] \[options\]

### **🌀 Full Workflow (Steps 1-4)**

Use the \--run\_step4 flag to automatically run the validation step after Step 3\.

python run\_autots.py \[input\_file\] \-ma \[AFIR\_params\] \--run\_step4 \[options\]

### **⏩ Skip to NEB (Steps 2-4)**

If you have a trajectory file, use \--skip\_step1.

python run\_autots.py path/to/my\_trajectory.xyz \--skip\_step1 \--run\_step4 \[options\]

### **⏩ Skip to IRC (Step 4 Only)**

If you have a refined TS structure, use \--skip\_to\_step4.

\# The input\_file must be the single TS structure file  
python run\_autots.py path/to/my\_ts\_final.xyz \--skip\_to\_step4 \[options\]

## **🧾 Command-Line Arguments**

| Argument | Description | Required / Optional |
| :---- | :---- | :---- |
| input\_file | Path to the initial file. \- **Default:** molecule.xyz \- **If \--skip\_step1:** Path to NEB trajectory \- **If \--skip\_to\_step4:** Path to single TS file | **Required** |
| \-ma, \--manual\_AFIR | Manual AFIR parameters for Step 1 (e.g., 150 6 8). | **Required** (unless \--skip\_step1 or \--skip\_to\_step4 is used) |
| \--skip\_step1 | Skips Step 1 (AFIR) and passes the input\_file to Step 2 (NEB). | Optional |
| \--run\_step4 | Runs Step 4 (IRC/Validation) after Step 3 completes. Default: False. | Optional |
| \--skip\_to\_step4 | Skips Steps 1-3 and runs only Step 4 on the input\_file. | Optional |
| \-cfg, \--config\_file | Path to the JSON configuration file. Default: ./config.json | Optional |
| \-osp, \--software\_path\_file | Path to the software definition file (used by othersoft). Default: ./software\_path.conf | Optional |
| \-n, \--top\_n | Refine the top N highest-energy candidates from NEB. Overrides config.json. | Optional |

## **🧠 Configuration File (config.json)**

Default settings for each workflow step are defined in config.json.

### **⚠️ CRITICAL CONFIGURATION GUIDE ⚠️**

To understand, modify, or add any option in config.json (e.g., opt\_method, NSTEP, calc\_exact\_hess),  
refer to the argument definitions in multioptpy/interface.py:

* **Step 1, 3, & 4 Settings:** see call\_optimizeparser()  
* **Step 2 Settings:** see call\_nebparser()

The interface.py file is the **single source of truth** for all option names.

### **🧩 Example config.json Structure**

{  
  "work\_dir": "autots\_run",  
  "top\_n\_candidates": 3,  
    
  "step1\_settings": {  
    "othersoft": "uma-s-1p1",  
    "opt\_method": \["rsirfo\_block\_fsb"\],  
    "use\_model\_hessian": "fischerd3",  
    "spin\_multiplicity": 1,  
    "electronic\_charge": \0  
  },  
    
  "step2\_settings": {  
    "othersoft": "uma-s-1p1",  
    "NSTEP": 15,  
    "ANEB": \[3, 5\],  
    "QSM": true,  
    "use\_model\_hessian": "fischerd3",  
    "save\_pict": true,  
	"spin\_multiplicity": 1,  
    "electronic\_charge": \0
  },  
    
  "step3\_settings": {  
    "othersoft": "uma-s-1p1",  
    "opt\_method": \["rsirfo\_block\_bofill"\],  
    "calc\_exact\_hess": 5,  
    "tight\_convergence\_criteria": true,  
    "frequency\_analysis": true,
    "spin\_multiplicity": 1,  
    "electronic\_charge": \0	
  },

  "step4\_settings": {  
    "othersoft": "uma-s-1p1",  
	"opt_method": ["rsirfo_block_bofill"],
    "spin\_multiplicity": 1,  
    "electronic\_charge": \0,  
	"calc_exact_hess": 10,
    "tight\_convergence\_criteria": true,  
    "frequency\_analysis": true,  
      
    "intrinsic\_reaction\_coordinates": \["0.5", "200", "lqa"\],

    "step4b\_opt\_method": \["rsirfo\_block\_fsb"\]  
  }  
}

## **⚖️ Setting Precedence**

Settings are applied in the following order (highest priority first):

1. **Command-Line Arguments** (e.g., \-n 5, \-ma ...)  
2. **config.json Settings** (e.g., "NSTEP": 15\)  
3. **multioptpy/interface.py Defaults** (lowest priority)


## Appendix

If you want to call `run_autots.py` from your source code, please read the below example.

```
import os
import sys
import copy # Added deepcopy for safe config modification

# Add the directory containing run_autots.py to the Python path
# This allows us to import launch_workflow directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the workflow launcher function from run_autots.py
    from run_autots import launch_workflow, load_config_from_file
except ImportError:
    print("Error: Could not import launch_workflow from run_autots.py.")
    print("Ensure run_autots.py is in the current directory and its dependencies are correct.")
    sys.exit(1)


# --- Example: Programmatic Workflow Launch ---

# 1. Define the input file path (these files must exist for the program to run)
INPUT_FILE = "path/to/my_initial_molecule.xyz"
TS_INPUT_FILE = "path/to/existing_ts_final.xyz"
CONFIG_FILE_PATH = os.path.abspath("./config.json") # Assumed location


def define_and_launch_workflow(input_file, skip_step1=False, run_step4=True, skip_to_step4=False):
    """
    Loads base config, applies programmatic overrides, and launches the workflow.
    """
    
    # Load the base configuration from the JSON file
    base_config = load_config_from_file(CONFIG_FILE_PATH)
    
    # 1. Initialize configuration with base settings
    config = base_config
    
    # 2. Apply dynamic input and flag settings
    config["initial_mol_file"] = input_file
    config["software_path_file_source"] = os.path.abspath("./software_path.conf")
    config["skip_step1"] = skip_step1
    config["run_step4"] = run_step4
    config["skip_to_step4"] = skip_to_step4
    
    # 3. Apply specific step settings for demonstration
    
    # Ensure manual_AFIR is provided if we're starting from Step 1
    if not skip_step1 and not skip_to_step4:
        config["step1_settings"]["manual_AFIR"] = ['100', '1', '2', '200', '3', '4'] 
    
    # Overwrite top_n for this demo
    config["top_n_candidates"] = 2 

    # Launch the workflow
    launch_workflow(config)


def main():
    # --- NOTE: Placeholder files must be created for successful execution ---
    
    # --- Example A: Run Full Workflow (Step 1 -> 4) ---
    print("\n=============================================")
    print("--- EXAMPLE A: FULL WORKFLOW (Step 1 -> 4) ---")
    print("=============================================")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Skipping Example A.")
    else:
        # Step 1, 2, 3, and 4 execution
        define_and_launch_workflow(INPUT_FILE, run_step4=True)


    # --- Example B: Skip to Step 4 (Validation Only) ---
    print("\n=======================================================")
    print("--- EXAMPLE B: VALIDATION ONLY (Skip to Step 4) ---")
    print("=======================================================")
    
    if not os.path.exists(TS_INPUT_FILE):
        print(f"Error: TS input file {TS_INPUT_FILE} not found. Skipping Example B.")
    else:
        # Run only Step 4 validation on an existing TS file
        # NOTE: This requires creating a separate work directory to prevent collision
        validation_config = define_workflow_config(TS_INPUT_FILE, skip_to_step4=True)
        validation_config['work_dir'] = "program_test_validation_run"
        
        # Load and update the actual config object before launching
        base_config = load_config_from_file(CONFIG_FILE_PATH)
        base_config.update(validation_config) # Merge demo overrides
        
        launch_workflow(base_config)


if __name__ == "__main__":
    # NOTE: In a real environment, you would ensure the dummy files 
    # and config.json exist before calling main().
    print("--- Starting Programmatic Call Test ---")
    main()

```