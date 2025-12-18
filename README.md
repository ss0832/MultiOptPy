# MultiOptPy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wpW8YO8r9gq20GACyzdaEsFK4Va1JQs4?usp=sharing) (Test 1, only use GFN2-xTB)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lfvyd7lv6ChjRC7xfPdrBZtGME4gakhz?usp=sharing) (Test 2, GFN2-xTB + PySCF(HF/STO-3G))


[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ss0832)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/multioptpy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/multioptpy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17973395.svg)](https://doi.org/10.5281/zenodo.17973395)

If this tool helped your studies, education, or saved your time, I'd appreciate a coffee!
Your support serves as a great encouragement for this personal project and fuels my next journey.
I also welcome contributions, bug reports, and pull requests to improve this tool.

Note on Contributions: While bug reports and pull requests are welcome, please note that this is a personal project maintained in my spare time. Responses to issues and PRs may be delayed or not guaranteed. I appreciate your patience and understanding.

Multifunctional geometry optimization tools for quantum chemical calculations 

This program implements many geometry optimization methods in Python for learning purposes.

This program can also automatically calculate the transition-state structure from a single equilibrium geometry.

**Notice:** This program has NOT been experimentally validated in laboratory settings. I release this code to enable community contributions and collaborative development. Use at your own discretion and validate results independently. 

(Caution: Using Japanese to explain) Instructions on how to use: 
- https://ss0832.github.io/
- https://ss0832.github.io/posts/20251130_mop_usage_menschutkin_reaction_uma_en/ (In English, auto-translated)

## Video Demo

[![MultiOptPy Demo](https://img.youtube.com/vi/AE61iY2HZ8Y/0.jpg)](https://www.youtube.com/watch?v=AE61iY2HZ8Y)


## Features

- It is intended to be used in a linux environment.
- It can be used not only with AFIR functions, but also with other bias potentials.



## Quick Start (for Linux)
```
# Below is an example showing how to use GFN2-xTB to calculate a transition-state structure.
# These commands are intended for users who want a straightforward, ready-to-run setup on Linux.

## 1. Download and install Anaconda:
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash Anaconda3-2025.06-1-Linux-x86_64.sh
source .bashrc
# if the conda command is not available, you need to manually add Anaconda to your PATH:
# (example command) echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

## 2. Create and activate a conda environment:
conda create -n test_mop python=3.12.7
conda activate test_mop

## 3. Download and install MultiOptPy:
wget https://github.com/ss0832/MultiOptPy/archive/refs/tags/v1.20.4.zip
unzip v1.20.4.zip
cd MultiOptPy-1.20.4
pip install -r requirements.txt

## 4. Copy the test configuration file and run the AutoTS workflow:
cp test/config_autots_run_xtb_test.json .
python run_autots.py aldol_rxn.xyz -cfg config_autots_run_xtb_test.json


# Installation via environment.yml (Linux / conda-forge)

## 1. Download and install MultiOptPy:
git clone -b stable-v1.0 https://github.com/ss0832/MultiOptPy.git
cd MultiOptPy

## 2. Create and activate a conda environment:
conda env create -f environment.yml
conda activate test_mop

## 3. Copy the test configuration file and run the AutoTS workflow:
cp test/config_autots_run_xtb_test.json .
python run_autots.py aldol_rxn.xyz -cfg config_autots_run_xtb_test.json

# Installation via pip (Linux)
conda create -n <env-name> python=3.12 pip
conda activate <env-name>
pip install git+https://github.com/ss0832/MultiOptPy.git@v1.20.4
wget https://github.com/ss0832/MultiOptPy/archive/refs/tags/v1.20.4.zip
unzip v1.20.4.zip
cd MultiOptPy-1.20.4

## 💻 Command Line Interface (CLI) Functionality (v1.20.2)
# The following eight core functionalities are available as direct executable commands in your terminal after installation:
# optmain (Logic from optmain.py):
# Function: Executes the Core Geometry Optimization functionality.
# nebmain (Logic from nebmain.py):
# Function: Executes the Nudged Elastic Band (NEB) path optimization tool for transition state searches.
# confsearch (Logic from conformation_search.py):
# Function: Utilizes the comprehensive Conformational Search routine.
# run_autots (Logic from run_autots.py):
# Function: Launches the Automated Transition State (AutoTS) workflow.
# mdmain (Logic from mdmain.py):
# Function: Initiates Molecular Dynamics (MD) simulation functionality.
# relaxedscan (Logic from relaxed_scan.py):
# Function: Executes the Relaxed Potential Energy Surface (PES) Scanning functionality.
# orientsearch (Logic from orientation_search.py):
# Function: Executes the molecular Orientation Sampling and Search utility.

```

## Required Modules
```
cd <directory of repository files>
pip install -r requirements.txt
```
 - psi4 (Official page:https://psicode.org/) or PySCF 
 - numpy
 - matplotlib
 - scipy
 - pytorch (for calculating derivatives)

Optional
 
 - tblite (If you use extended tight binding (xTB) method, this module is required.)
 - dxtb (same as above)
 - ASE 
   
## References

References are given in the source code.

## Usage

After downloading the repository using git clone or similar commands, move to the generated directory and run the following:
python command
```
python optmain.py SN2.xyz -ma 150 1 6 -pyscf -elec 0 -spin 0 -opt rsirfo_block_fsb -modelhess
```
CLI command (arbitrary directory)
```
optmain SN2.xyz -ma 150 1 6 -pyscf -elec 0 -spin 0 -opt rsirfo_block_fsb -modelhess
```
python command
```
python optmain.py aldol_rxn.xyz -ma 95 1 5 50 3 11 -pyscf -elec 0 -spin 0 -opt rsirfo_block_fsb -modelhess
```
CLI command (arbitrary directory)
```
optmain aldol_rxn.xyz -ma 95 1 5 50 3 11 -pyscf -elec 0 -spin 0 -opt rsirfo_block_fsb -modelhess
```

For SADDLE calculation 

python command
```
python optmain.py aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rsirfo_block_bofill -order 1 -fc 5
```
CLI command (arbitrary directory)
```
optmain aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rsirfo_block_bofill -order 1 -fc 5
```
##### For NEB method
python command
```
python nebmain.py aldol_rxn -xtb GFN2-xTB -ns 50 -adpred 1 -nd 0.5
```
CLI command (arbitrary directory)
```
nebmain aldol_rxn -xtb GFN2-xTB -ns 50 -adpred 1 -nd 0.5
```

##### For iEIP method
python command
```
python ieipmain.py ieip_test -xtb GFN2-xTB 
```
CLI command (arbitrary directory)
```
ieipmain ieip_test -xtb GFN2-xTB 
```
##### For Molecular Dynamics (MD)
python command
```
python mdmain.py aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000
```
CLI command (arbitrary directory)
```
mdmain aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000
```
(Default deterministic algorithm for MD is Nosé–Hoover thermostat.)

For orientation search 
```
python orientation_search.py aldol_rxn.xyz -part 1-4 -ma 95 1 5 50 3 11 -nsample 5 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess
```
For conformation search
```
python conformation_search.py s8_for_confomation_search_test.xyz -xtb GFN2-xTB -ns 2000
```
For relaxed scan (Similar to functions implemented in Gaussian)
```
python relaxed_scan.py SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -elec -1 -spin 0 -xtb GFN2-xTB -opt crsirfo_block_fsb -modelhess
```
## Options
(optmain.py)

**`-opt`**

Specify the algorithm to be used for structural optimization.

example 1) `-opt FIRE`.

Perform structural optimization using the FIRE method.


Available optimization methods:

Recommended optimization methods:

- FIRE (Robust method)
- TR_LBFGS (Limited-memory BFGS method with trust radius method, Faster convergence than FIRE without Hessian)
- rsirfo_block_fsb 
- rsirfo_block_bofill (for calculation of saddle point)

`-ma`

Add the potential by AFIR function.
Energy (kJ/mol) Atom 1 or fragment 1 to which potential is added Atom 2 or fragment 2 to which potential is added.

Example 1) `-ma 195 1 5`

Apply a potential of 195 kJ/mol (pushing force) to the first atom and the fifth atom as a pair.

Example 2) `-ma 195 1 5 195 3 11`

Multiply the potential of 195 kJ/mol (pushing force) by the pair of the first atom and the fifth atom. Then multiply the potential of 195 kJ/mol (pushing force) by the pair of the third atom and the eleventh atom.

Example 3) `-ma -195 1-3 5,6`

Multiply the potential of -195 kJ/mol (pulling force) by the fragment consisting of the 1st-3rd atoms paired with the fragments consisting of the 5th and 6th atoms.


`-bs`

Specifies the basis function. The default is 6-31G*.

Example 1) `-bs 6-31G*`

Calculate using 6-31G* as the basis function.

Example 2) `-bs sto-3g`

Calculate using STO-3G as the basis function.

`-func`

Specify the functionals in the DFT (specify the calculation method). The default is b3lyp.

Example 1) `-func b3lyp`

Calculate using B3LYP as the functional.

Example 2) `-func hf`

Calculate using the Hartree-Fock method.

`-sub_bs`

Specify a specific basis function for a given atom.

Example 1) `-sub_bs I LanL2DZ`

Assign the basis function LanL2DZ to the iodine atom, and if -bs is the default, assign 6-31G* to non-iodine atoms for calculation.

`-ns`

Specifies the maximum number of times the gradient is calculated for structural optimization. The default is a maximum of 300 calculations.

Example 1) `-ns 400`

Calculate gradient up to 400 iterations.



`-core`

Specify the number of CPU cores to be used in the calculation. By default, 8 cores are used. (Adjust according to your own environment.)

Example 1) `-core 4`

Calculate using 4 CPU cores.

`-mem`

Specify the memory to be used for calculations. The default is 1GB. (Adjust according to your own environment.)

Example 1) `-mem 2GB`

Calculate using 2GB of memory.

`-d`

Specifies the size of the step width after gradient calculation. The larger the value, the faster the convergence, but it is not possible to follow carefully on the potential hypersurface. 

Example 1) `-d 0.05`



`-kp`

Multiply the potential calculated from the following equation (a potential based on the harmonic approximation) by the two atom pairs. This is used when you want to fix the distance between atoms to some extent.

$V(r) = 0.5k(r - r_0)^2$

`spring const. k (a.u.) keep distance [$ r_0] (ang.) atom1,atom2 ...`

Example 1) `-kp 2.0 1.0 1,2`

Apply harmonic approximation potentials to the 1st and 2nd atoms with spring constant 2.0 a.u. and equilibrium distance 1.0 Å.

`-akp`

The potential (based on anharmonic approximation, Morse potential) calculated from the following equation is applied to two atomic pairs. This is used when you want to fix the distance between atoms to some extent. Unlike -kp, the depth of the potential is adjustable.

$V(r) = D_e [1 - exp(- \sqrt(\frac{k}{2D_e})(r - r_0))]^2$

`potential well depth (a.u.) spring const.(a.u.) keep distance (ang.) atom1,atom2 ...`

Example 1) `-ukp 2.0 2.0 1.0 1,2`

Anharmonic approximate potential (Mohs potential) is applied to the first and second atoms as equilibrium distance 1.0 Å with a potential depth of 2.0 a.u. and a spring constant of 2.0 a.u.

`-ka`

The potential calculated from the following equation (potential based on the harmonic approximation) is applied to a group of three atoms, which is used when you want to fix the angle (bond angle) between the three atoms to some extent.

$V(\theta) = 0.5k(\theta - \theta_0)^2$

`spring const.(a.u.) keep angle (degrees) atom1,atom2,atom3`

Example 1) `-ka 2.0 60 1,2,3`

Assuming a spring constant of 2.0 a.u. and an equilibrium angle of 60 degrees, apply a potential so that the angle between the first, second, and third atoms approaches 60 degrees.

`-kda`

The potential (based on the harmonic approximation) calculated from the following equation is applied to a group of 4 atoms to fix the dihedral angle of the 4 atoms to a certain degree.

$V(\phi) = 0.5k(\phi - \phi_0)^2$

`spring const.(a.u.) keep dihedral angle (degrees) atom1,atom2,atom3,atom4 ...`

Example 1) `-kda 2.0 60 1,2,3,4`

With a spring constant of 2.0 a.u. and an equilibrium angle of 60 degrees, apply a potential so that the dihedral angles of the planes formed by the 1st, 2nd, and 3rd atoms and the 2nd, 3rd, and 4th atoms approach 60 degrees.

`-xtb`

Use extended tight binding method. (It is required tblite (python module).)

Example 1) `-xtb GFN2-xTB`

Use GFN2-xTB method to optimize molecular structure.

 - Other options are experimental.


## Author

Author of this program is ss0832.

## License

GNU Affero General Public License v3.0


## Contact

highlighty876[at]gmail.com

## Citation

If you use MultiOptPy in your research, please cite it as follows:

```bibtex
@software{ss0832_multioptpy_2025,
  author       = {ss0832},
  title        = {MultiOptPy: Multifunctional geometry optimization tools for quantum chemical calculations},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.20.4},
  doi          = {10.5281/zenodo.17973395},
  url          = {https://doi.org/10.5281/zenodo.17973395}
}
```
```
ss0832. (2025). MultiOptPy: Multifunctional geometry optimization tools for quantum chemical calculations (v1.20.4). Zenodo. https://doi.org/10.5281/zenodo.17973395
```

## Setting Up an Environment for Using NNP(UMA) on Windows 11

### 1. Install Anaconda

Download and install **Anaconda3-2025.06-1-Windows-x86_64.exe** from:

[https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/)

### 2. Launch the Anaconda PowerShell Prompt

Open **"Anaconda PowerShell Prompt"** from the Windows Start menu.

### 3. Create a New Virtual Environment

```
conda create -n <env_name> python=3.12.7
```

### 4. Activate the Environment

```
conda activate <env_name>
```

### 5. Install Required Libraries

```
pip install ase==3.26.0 fairchem-core==2.7.1 torch==2.6.0
```

* **fairchem-core**: Required for running NNP models provided by FAIR Chemistry.
* **ase**: Interface for passing molecular structures to the NNP.
* **torch**: PyTorch library for neural network execution.

---

## Setting Up the Model File (.pt) for Your NNP Library

### 1. Download the Model File

Download **uma-s-1p1.pt** from the following page:

[https://huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA)

(Ensure that you have permission to use the file.)

### 2. Add the Model Path to MultiOptPy

Open the file `software_path.conf` inside the **MultiOptPy** directory.

Add the following line using the absolute path to the model file:

```
uma-s-1p1::<absolute_path_to/uma-s-1p1.pt>
```

This enables **MultiOptPy** to use the **uma-s-1p1 NNP model**.

### references of UMA
- arXiv preprint arXiv:2505.08762 (2025).
- https://github.com/facebookresearch/fairchem

## Create environment for Win11 / UMA(NNP) using conda

```
conda env create -f environment_win11uma.yml
conda activate test_mop_win11_uma
```


---

> **Status: Maintenance Mode / Frozen**
> *This project has reached its initial stability goals (v1.20.2) and is currently frozen. No new features are planned by the original author, but the codebase remains open for the community to fork and explore the roadmap above.*
