# 🚀 AutoTS Workflow (`run_autots.py`) Execution Guide

`run_autots.py` is the primary launcher script for the **MultiOptPy** automated Transition State (TS) search workflow.

This script reads default settings from `config.json` and executes a 3-step workflow based on command-line arguments, such as the input file and AFIR parameters (or the `--skip_step1` flag).

---

## 🧩 Workflow Overview

### **Step 1: AFIR Scan** (`OptimizationJob`)
- Scans a reaction path from an initial structure based on the `-ma` (manual_AFIR) options.
- Generates a trajectory file (e.g., `_step1_result_traj.xyz`).

### **Step 2: NEB Optimization** (`NEBJob`)
- Uses the trajectory from Step 1 as an initial path for a Nudged Elastic Band (NEB) calculation.
- Identifies potential TS candidates (energy maxima) along the path.

### **Step 3: TS Refinement** (`OptimizationJob`)
- Performs a saddle point optimization (`saddle_order=1`) on the top N highest-energy candidates found in Step 2.
- Generates the final, refined TS structures.

---

## ⚙️ Usage

Execute the script from the **root directory** (same level as the `multioptpy` folder).

### ✅ Standard Run (Starting with AFIR)

```bash
python run_autots.py [input_file] -ma [AFIR_params] [options]
```

### 🌀 Skipping Step 1 (Starting with NEB)

If you already have a trajectory file (`.xyz` or `.traj`) from a previous scan, you can use the `--skip_step1` flag.

```bash
# Provide the trajectory file as input and skip Step 1
python run_autots.py path/to/my_trajectory.xyz --skip_step1 [options]
```

---

## 🧾 Command-Line Arguments

| Argument | Description | Required / Optional |
|---|---|---|
| `input_file` | Path to the initial structure file (`.xyz`, etc.). If `--skip_step1` is used, this must be the NEB trajectory file. | **Required** |
| `-ma`, `--manual_AFIR` | Manual AFIR parameters for Step 1 (e.g., `150 6 8 150 1 9`). | **Required** (unless `--skip_step1` is used) |
| `--skip_step1` | Skips Step 1 (AFIR) and passes the input directly to Step 2 (NEB). | Optional |
| `-cfg`, `--config_file` | Path to the JSON configuration file. Default: `./config.json` | Optional |
| `-osp`, `--software_path_file` | Path to the software definition file (used by `othersoft`). Default: `./software_path.conf` | Optional |
| `-n`, `--top_n` | Refine the top N highest-energy candidates from NEB. Overrides `top_n_candidates` in `config.json`. | Optional |

---

## 🧠 Configuration File (`config.json`)

Default settings for each workflow step are defined in `config.json`.

### ⚠️ CRITICAL CONFIGURATION GUIDE ⚠️

To **understand, modify, or add** any option in `config.json` (e.g., `opt_method`, `NSTEP`, `calc_exact_hess`),  
refer to the argument definitions in **`multioptpy/interface.py`**:

- **Step 1 & 3 Settings:** see `call_optimizeparser()`  
- **Step 2 Settings:** see `call_nebparser()`

The `interface.py` file is the **single source of truth** for option names and defaults.

---

### 🧩 Example `config.json`

```json
{
  "work_dir": "autots_run",
  "top_n_candidates": 3,
  
  "step1_settings": {
    "othersoft": "uma-s-1p1",
    "opt_method": ["rsirfo_block_fsb"],
    "use_model_hessian": "fischerd3",
    "spin_multiplicity": 1,
    "electronic_charge": -1
  },
  
  "step2_settings": {
    "othersoft": "uma-s-1p1",
    "NSTEP": 15,
    "ANEB": [3, 5],
    "QSM": true,
    "use_model_hessian": "fischerd3",
    "save_pict": true
  },
  
  "step3_settings": {
    "othersoft": "uma-s-1p1",
    "opt_method": ["rsirfo_block_bofill"],
    "calc_exact_hess": 5,
    "tight_convergence_criteria": true,
    "frequency_analysis": true
  }
}
```

---

## ⚖️ Setting Precedence

Settings are applied in the following order (highest priority first):

1. **Command-Line Arguments** (e.g., `-n 5`, `-ma ...`)  
2. **`config.json` Settings** (e.g., `"NSTEP": 15`)  
3. **`multioptpy/interface.py` Defaults** (lowest priority)

---

## 📘 References

- MultiOptPy repository (source & examples).  
- MultiOptPy documentation (BiasPotPy docs) for usage examples and installation notes.  
(See repository and docs for more details.)
