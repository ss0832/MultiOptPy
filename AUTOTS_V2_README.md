# **AutoTSWorkflow_v2: config.json Configuration Guide**

AutoTSWorkflow_v2 is a new workflow engine that dynamically and flexibly executes calculation steps by reading the "workflow" block in the config.json file.

## **1. Basic Concept: Settings Vault and Execution Script**

The v2 configuration file is divided into two main parts:

1. **Settings Vault (Common with v1)**:  
   * These are the top-level blocks like "step1_settings", "step2_settings", "step3_settings", etc.  
   * They serve as a "vault" to store the parameters (calculation level, thresholds, algorithms, etc.) used for each step.  
   * In v2, these are used as the "base settings".  
2. **Execution Script ("workflow" block)**:  
   * **This is the core of v2.** It is an "execution script" that defines which steps to run, in what order, using which settings ("settings_key"), and how many times ("repeat").  
   * You can temporarily override the "vault" settings using param_override.

## **2. Common Keys in the "workflow" Block**

The "workflow" block is a list ([]) of jobs (steps) you want to execute. Each job is defined as a dictionary ({}) with the following keys:
```json
  "workflow": [  
    {  
      "step": "step1",  
      "enabled": true,  
      "settings_key": "step1_settings",  
      "repeat": 1,  
      "repeat_settings": [  
        { "param_override": { ... } }  
      ]  
    },  
    { ... }  
  ]
```
* "step" (Required): The name of the step to run (e.g., "step1", "step2", "step3", "step4").  
* "enabled" (Optional): true (default) to run, false to skip this job.  
* "settings_key" (Required): The key name of the "vault" block to use as the base settings for this job (e.g., "step1_settings").  
* "repeat" (Optional): The number of times to repeat this job (default: 1).  
* "repeat_settings" (Optional): A list defining settings for each repeat.  
  * "param_override": A dictionary of parameters to override the base settings loaded from "settings_key".

## **3. Step 1 (AFIR) Configuration**

step1 is the path exploration step. In v2, if step1 has repeat set to 2 or more, it **always runs in Sequential mode**.

* **Behavior**: The second execution (run 2) starts **using the final structure (..._final.xyz) from the first execution (run 1) as its input**.  
* **Primary Use**: To divide the AFIR search into multiple stages, continuing the search sequentially with different parameters (e.g., search a wide area -> explore deeper from a found structure).

### **Example Configuration: Run Step 1 twice (2 stages)**

This overrides the manual_AFIR from step1_settings for each run.
```json
    {  
      "step": "step1",  
      "enabled": true,  
      "repeat": 2,  
      "settings_key": "step1_settings",  
      "repeat_settings": [  
        {  
          "param_override": {  
            "manual_AFIR": ["600", "6", "22"]  
          }  
        },  
        {  
          "param_override": {  
            "manual_AFIR": ["300", "6", "23"]  
          }  
        }  
      ]  
    }
```
* **Run 1**: Executes based on step1_settings with manual_AFIR: ["600", "6", "22"].  
* **Run 2**: Executes based on step1_settings with manual_AFIR: ["300", "6", "23"]. **The input is the final structure from Run 1.**  
* **Note**: If the repeat_settings list is shorter than the repeat count, the **last element** in the list will be reused for all remaining runs.

## **4. Step 2 (NEB) Configuration**

step2 is the path optimization (e.g., NEB) step. Unlike step1, step2 can control its repeat behavior using the "mode" key.

### **4.1. Mode 1: Sequential Execution ("mode": "sequential") - Default**

* **Behavior**: Run 2 takes the **final relaxed path (..._final_path.xyz)** from Run 1 as its input and **performs relaxation (refinement) again**.  
* **Primary Use**: Hierarchical refinement of the NEB path. (e.g., Run 1 relaxes coarsely with few nodes -> Run 2 increases nodes and refines the path).

### **Example Configuration: Refine NEB path in 2 stages**
```json
    {  
      "step": "step2",  
      "enabled": true,  
      "repeat": 2,  
      "settings_key": "step2_settings",  
      "mode": "sequential",   
      "repeat_settings": [  
        {  
          "param_override": {  
             "NSTEP": 10   
          }  
        },  
        {  
          "param_override": {  
             "NSTEP": 50,  
             "node_distance_bernstein": 0.50   
          }  
        }  
      ]  
    }
```
* **Run 1**: Takes the step1 path as input, relaxes it based on step2_settings with NSTEP: 10.  
* **Run 2**: Takes the **final relaxed path from Run 1** as input, and **relaxes it again** with more precise settings (e.g., NSTEP: 50).

### **4.2. Mode 2: Independent Execution ("mode": "independent")**

* **Behavior**: All executions specified by repeat run **independently (as if in parallel)**, all using the **same input (the final path from step1)**.  
* **Primary Use**: To test different NEB parameters (like NSTEP or ANEB) on the same path and select the best result.  
* **Post-processing**: **All TS candidates** found from all runs (Run 1, Run 2, ...) are gathered by select_candidates, sorted by energy, and the top_n are selected.

### **Example Configuration: Run with 2 different NEB parameters independently**
```json
    {  
      "step": "step2",  
      "enabled": true,  
      "repeat": 2,  
      "settings_key": "step2_settings",  
      "mode": "independent",   
      "repeat_settings": [  
        {  
          "param_override": {  
             "NSTEP": 20   
          }  
        },  
        {  
          "param_override": {  
             "NSTEP": 40,  
             "ANEB": [3, 3]  
          }  
        }  
      ]  
    }
```
* **Run 1**: Takes the step1 path as input, executes based on step2_settings with NSTEP: 20.  
* **Run 2**: Takes the **same step1 path** as input, executes based on step2_settings with NSTEP: 40 and ANEB: [3, 3].

## **5. Step 3 (TS Refinement) Configuration**

step3 is the TS refinement step. When step3 repeats, it **always uses the final candidate list from step2 (data_cache["step2"]["candidates"]) as input**.

* **Behavior**: All executions specified by repeat run **on the exact same list of TS candidates**.  
* **Primary Use**: **Hierarchical Refinement**.  
  * Run 1: Optimize the candidates from step2 with a low-cost calculation level (e.g., B3LYP).  
  * Run 2: Optimize the same candidates from step2 with a **high-cost** calculation level (e.g., MP2).  
  * ...  
* **Post-processing**: consolidate_ts adopts the results from the **last execution** (assumed to be the highest level) as the final TS.

### **Example Configuration: Refine TS at 2 different levels of theory**

This overrides othersoft (specifying calculation level) and opt_method from step3_settings.
```json
    {  
      "step": "step3",  
      "enabled": true,  
      "repeat": 2,  
      "settings_key": "step3_settings",  
      "repeat_settings": [  
        {  
          "param_override": {  
             "othersoft": "uma-s-1p1"   
          }  
        },  
        {  
          "param_override": {  
             "othersoft": "g16-mp2-1p1",  
             "opt_method": ["rsirfo_block_mp2"],  
             "calc_exact_hess": true  
          }  
        }  
      ]  
    }
```
* **Run 1**: Optimizes all candidates from step2 using the base step3_settings (e.g., othersoft: "uma-s-1p1").  
* **Run 2**: Optimizes the **same candidates from step2** again, but this time based on step3_settings overridden with higher-precision settings (e.g., MP2 level).  
* **Final Result**: The structures that successfully optimized during Run 2 (MP2 level) will be used as input for step4.

## **6. Full v2 Configuration Example (based on config_bh9_4_21.json)**

This is a complete example of a v2 configuration, based on the v1 config_bh9_4_21.json, leveraging the advanced features of the workflow block.
```json
{  
  "work_dir": "bh9_4_21_v2",  
  "top_n_candidates": 3,  
  "multioptpy_version": "v1.19.4b",  
    
  "step1_settings": {  
    "othersoft": "uma-s-1p1",  
    "opt_method": ["rsirfo_block_fsb"],  
    "use_model_hessian": "fischerd3",  
    "spin_multiplicity": 2,  
    "electronic_charge": 0,  
	"manual_AFIR": ["400", "6", "22"]  
  },  
    
  "step2_settings": {  
    "othersoft": "uma-s-1p1",  
    "NSTEP": 15,  
    "ANEB": [3, 5],  
    "QSM": true,  
    "use_model_hessian": "fischerd3",  
    "save_pict": true,  
    "node_distance_bernstein": 0.80,  
    "align_distances": 9999,  
    "spin_multiplicity": 2,  
    "electronic_charge": 0  
  },  
    
  "step3_settings": {  
    "othersoft": "uma-s-1p1",  
    "opt_method": ["rsirfo_block_bofill"],  
    "calc_exact_hess": 5,  
    "tight_convergence_criteria": true,  
    "max_trust_radius": 0.2,  
    "frequency_analysis": true,  
    "spin_multiplicity": 2,  
    "electronic_charge": 0  
  },

  "step4_settings": {  
    "othersoft": "uma-s-1p1",  
	"opt_method": ["rsirfo_block_bofill"],  
    "spin_multiplicity": 2,  
    "electronic_charge": 0,  
	"calc_exact_hess": 10,  
    "tight_convergence_criteria": true,  
    "frequency_analysis": true,  
    "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"],  
    "step4b_opt_method": ["rsirfo_block_fsb"]  
  },

  "workflow": [  
    {  
      "step": "step1",  
      "enabled": true,  
      "repeat": 2,  
      "settings_key": "step1_settings",  
      "repeat_settings": [  
        { "param_override": { "manual_AFIR": ["600","6","22"] } },  
        { "param_override": { "manual_AFIR": ["300","6","23"] } }  
      ]  
    },  
    {  
      "step": "step2",  
      "enabled": true,  
      "repeat": 2,  
      "settings_key": "step2_settings",  
      "repeat_settings": [  
        { "param_override": { "NSTEP": 20 } },  
        { "param_override": { "NSTEP": 40, "ANEB": [3,3] } }  
      ]  
    },  
    {  
      "step": "step3",  
      "enabled": true,  
      "repeat": 1,  
      "settings_key": "step3_settings",  
      "repeat_settings": [  
        {  
          "param_override": {}  
        },  
        {  
          "param_override": {  
             "basis_set": "6-31G",  
             "opt_method": ["rsirfo_block_bofill"],  
             "calc_exact_hess": true  
          }  
        }  
      ]  
    },  
    {  
      "step": "step4",  
      "enabled": true,  
      "repeat": 1,  
      "settings_key": "step4_settings"  
    }  
  ]  
}  
```