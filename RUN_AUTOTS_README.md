# `run_autots.py` Execution Guide

`run_autots.py` is the primary "launcher" script for the `autots` workflow. It is designed to be placed in the same root directory as the `multioptpy` package and executed from there.

---

## 1. How to Run (Command-Line Arguments)

This script accepts a combination of **required** and **optional** arguments to define the workflow.

**Example Command:**
```bash
python run_autots.py molecule.xyz -ma 150 6 8 150 1 9 -n 2 -osp /path/to/my/software_path.conf
```
Required Arguments
input_file

Description: The single molecular structure file (.xyz, etc.) that serves as the starting point for the workflow.

Format: str (File path)

Example: molecule.xyz

-ma / --manual_AFIR

Description: Specifies the parameters for the Manual AFIR scan in Step 1. This is a list of Force, Atom1, Atom2 sets, repeated as needed.

Format: list[str] (A list of strings)

Example: -ma 150 6 8 150 1 9

Optional Arguments
-n / --top_n

Description: The number of highest-energy candidates from the Step 2 (NEB) results to pass on to Step 3 (Refinement).

Format: int (Integer)

Default: 3

-osp / --software_path_file

Description: The path to the software path definition file, which is required by the othersoft option (e.g., uma-s-1p1).

Format: str (File path)

Default: ./software_path.conf (Assumes the file is in the same directory as run_autots.py)

2. Internal Configuration Guide (get_default_config)
The get_default_config() function inside run_autots.py defines the default settings for each step of the workflow.

If you need to change detailed calculation conditions (like the optimization method, NEB parameters, convergence criteria, etc.) that are not covered by the command-line arguments, you should edit this function directly.

step1_settings: AFIR Scan (Step 1)
These settings are passed to the OptimizationJob for Step 1.

```Python

"step1_settings": {
    # QM software to use (str)
    "othersoft": "uma-s-1p1", 
    # Optimization method (list[str])
    "opt_method": ["rsirfo_block_fsb"], 
    # Model hessian (str)
    "use_model_hessian": 'fischerd3'
    # "manual_AFIR" is overridden by the command-line argument
}
```
step2_settings: NEB Path Optimization (Step 2)
These settings are passed to the NEBJob for Step 2.

```Python

"step2_settings": {
    "othersoft": "uma-s-1p1",
    # Max steps (int)
    "NSTEP": 15,
    # ANEB settings (list[int]) - [interpolation_num, frequency]
    "ANEB": [3, 5],
    # Use QSM (bool)
    "QSM": True,
    "use_model_hessian": 'fischerd3',
    # Save pictures (bool)
    "save_pict": True,
    # Node distance (float)
    "node_distance_bernstein": 0.75,
    # (int)
    "align_distances": 9999,
    # Spin multiplicity (int)
    "spin_multiplicity": 1,
    # Charge (int)
    "electronic_charge": 0
}
```
step3_settings: TS Refinement (Step 3)
These settings are passed to the OptimizationJob for Step 3.

```Python

"step3_settings": {
    "othersoft": "uma-s-1p1",
    # TS refinement optimization method (list[str])
    "opt_method": ["rsirfo_block_bofill"],
    # Hessian calculation frequency (int)
    "calc_exact_hess": 5,
    # Convergence criteria (bool)
    "tight_convergence_criteria": True,
    # Max trust radius (float)
    "max_trust_radius": 0.2,
    # Run frequency analysis (bool)
    "frequency_analysis": True
    # "saddle_order": 1 is set automatically by autots.py

}
```
