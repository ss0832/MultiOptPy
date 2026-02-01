MultiOptPy Documentation
========================

    **Version**: v1.20.5  
    **Repository**: `https://github.com/ss0832/MultiOptPy <https://github.com/ss0832/MultiOptPy>`_  
    **Status**: Maintenance Mode / Frozen  
    *Multifunctional geometry optimization tools for quantum chemical calculations (AFIR, NEB, MD, TS search, NNP/UMA support).*

.. note::
   The project is frozen after reaching stability goals (v1.20.2). No new features are planned by the original author, but community contributions are welcome.

.. warning::
   This program has **not** been experimentally validated in laboratory settings. Use at your own discretion.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   quickstart
   cli
   autots_workflow
   configuration
   bias_potentials
   examples
   references
   license_contact

----------------
Overview
--------

MultiOptPy is a Python toolkit providing:

* Geometry optimization (minima and saddle points) with numerous optimizers (FIRE, TR\_LBFGS, rsirfo\_block\_fsb, rsirfo\_block\_bofill, etc.).
* Reaction path exploration (AFIR, NEB, QSM, IEIP), automated TS discovery (AutoTS), and IRC validation.
* Molecular dynamics (AIMD) with thermostats (e.g., Nosé–Hoover).
* Bias potentials and constraints (AFIR, harmonic/Morse restraints, dihedral/angle restraints, mechanochemical forces, metadynamics, nano-reactor potentials, etc.).
* Extended Tight Binding (xTB) and PySCF/Psi4 support; optional NNP via UMA (fairchem-core).
* Command-line interface (CLI) commands installable as console scripts after ``pip install``.
* Cross-platform guidance: Linux (primary), Windows 11 instructions for UMA (NNP).

----------------
Installation
------------

Recommended (Linux, conda, release zip)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # 1. Install Anaconda (example)
    cd ~
    wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
    bash Anaconda3-2025.06-1-Linux-x86_64.sh
    source ~/.bashrc

    # 2. Create env
    conda create -n test_mop python=3.12.7
    conda activate test_mop

    # 3. Get MultiOptPy v1.20.5 release
    wget https://github.com/ss0832/MultiOptPy/archive/refs/tags/v1.20.5.zip
    unzip v1.20.5.zip
    cd MultiOptPy-1.20.5
    pip install -r requirements.txt

Alternative: git + environment.yml (Linux / conda-forge)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone -b stable-v1.0 https://github.com/ss0832/MultiOptPy.git
    cd MultiOptPy
    conda env create -f environment.yml
    conda activate test_mop
    cp test/config_autots_run_xtb_test.json .
    python run_autots.py aldol_rxn.xyz -cfg config_autots_run_xtb_test.json

Alternative: pip (Linux, latest tag)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda create -n <env-name> python=3.12 pip
    conda activate <env-name>
    pip install git+https://github.com/ss0832/MultiOptPy.git@v1.20.5
    wget https://github.com/ss0832/MultiOptPy/archive/refs/tags/v1.20.5.zip
    unzip v1.20.5.zip
    cd MultiOptPy-1.20.5

Windows 11 + UMA (NNP) environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the provided conda environment file for UMA/NNP.

.. code-block:: bash

    # 1. Install Anaconda3-2025.06-1-Windows-x86_64.exe
    # 2. Open "Anaconda PowerShell Prompt"
    git clone https://github.com/ss0832/MultiOptPy.git
    cd MultiOptPy
    conda env create -f environment_win11uma.yml
    conda activate test_mop_win11_uma

UMA model setup
^^^^^^^^^^^^^^^

1. Download **uma-s-1p1.pt** from https://huggingface.co/facebook/UMA  
2. Edit ``software_path.conf`` (in the MultiOptPy root) and add:

   .. code-block:: text

       uma-s-1p1::<absolute_path_to/uma-s-1p1.pt>

Requirements (core)
~~~~~~~~~~~~~~~~~~~
* psi4 **or** PySCF
* numpy, matplotlib, scipy, pytorch
* Optional: tblite / dxtb (xTB), ASE, fairchem-core (NNP)

----------------
Quick Start (Linux, GFN2-xTB TS search)
---------------------------------------

.. code-block:: bash

    conda create -n test_mop python=3.12.7
    conda activate test_mop
    wget https://github.com/ss0832/MultiOptPy/archive/refs/tags/v1.20.5.zip
    unzip v1.20.5.zip
    cd MultiOptPy-1.20.5
    pip install -r requirements.txt

    # Copy sample config and run AutoTS
    cp test/config_autots_run_xtb_test.json .
    python run_autots.py aldol_rxn.xyz -cfg config_autots_run_xtb_test.json

----------------
Command Line Interface (CLI)
----------------------------

After installation, these console scripts are available:

* ``optmain`` — Core geometry optimization (minima / saddle)
* ``nebmain`` — NEB path optimization
* ``run_autots`` — Automated TS workflow (Generate path using Bias Potential → NEB → TS refine → optional IRC)
* ``mdmain`` — Molecular dynamics
* ``confsearch`` — Conformational search
* ``relaxedscan`` — Relaxed PES scan
* ``orientsearch`` — Orientation sampling
* ``ieipmain`` — Initial–End point Interpolation Path

Essential options (common)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Basis / functional: ``-bs``, ``-func`` (defaults: 6-31G(d), b3lyp)
* Charge / spin: ``-elec``, ``-spin``
* xTB: ``-xtb GFN2-xTB`` (requires tblite/dxtb)
* PySCF: ``-pyscf`` (use PySCF instead of psi4)
* Model Hessian: ``-modelhess`` (e.g., ``fischerd3``)
* Iterations: ``-ns`` / ``--NSTEP``
* Optimizer: ``-opt`` (e.g., FIRE, TR_LBFGS, rsirfo_block_fsb, rsirfo_block_bofill for saddle)
* AFIR/bias: ``-ma <Energy> <Frag1> <Frag2>`` (bias potential)

----------------
AutoTS Workflow (run_autots.py)
-------------------------------

Purpose
~~~~~~~
Automated TS search from a single equilibrium geometry: **Generate path using Bias Potential** → NEB → TS refinement → optional IRC/endpoints.

Steps
~~~~~
1. **Generate path using Bias Potential (Step 1)** — Apply ``-ma`` (or other bias) to generate a trajectory.
2. **NEB (Step 2)** — Use the trajectory; locate TS candidates.
3. **TS Refinement (Step 3)** — Saddle optimization (``saddle_order=1``) on top-N candidates.
4. **IRC & Validation (Step 4, optional)** — IRC, endpoint optimizations, reaction profile.

Usage
~~~~~

.. code-block:: bash

    # Standard (Steps 1–3)
    python run_autots.py input.xyz -ma 150 1 5

    # Full (Steps 1–4)
    python run_autots.py input.xyz -ma 150 1 5 --run_step4

    # Skip Bias path (start from trajectory)
    python run_autots.py path/to/trajectory.xyz --skip_step1 --run_step4

    # Skip to validation (IRC only, TS file input)
    python run_autots.py path/to/ts_final.xyz --skip_to_step4

Configuration (config.json, precedence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Priority: **CLI args > config.json > interface.py defaults**.

Example skeleton:

.. code-block:: json

    {
      "work_dir": "autots_run",
      "top_n_candidates": 3,
      "step1_settings": {
        "opt_method": ["rsirfo_block_fsb"],
        "manual_AFIR": ["150", "1", "5"],
        "use_model_hessian": "fischerd3"
      },
      "step2_settings": {
        "NSTEP": 15,
        "ANEB": [3, 5],
        "QSM": true,
        "use_model_hessian": "fischerd3",
        "save_pict": true
      },
      "step3_settings": {
        "opt_method": ["rsirfo_block_bofill"],
        "calc_exact_hess": 5,
        "tight_convergence_criteria": true,
        "frequency_analysis": true
      },
      "step4_settings": {
        "opt_method": ["rsirfo_block_bofill"],
        "calc_exact_hess": 10,
        "tight_convergence_criteria": true,
        "frequency_analysis": true,
        "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"],
        "step4b_opt_method": ["rsirfo_block_fsb"]
      }
    }

Key AutoTS flags
~~~~~~~~~~~~~~~~
* ``--run_step4`` — enable IRC/validation.
* ``--skip_step1`` — start from NEB with an existing trajectory.
* ``--skip_to_step4`` — run only validation starting from a TS file.
* ``-n / --top_n`` — refine top-N candidates from NEB.
* ``-cfg`` — path to config.json.
* ``-osp`` — path to software_path.conf.

----------------
Configuration Reference (highlights)
------------------------------------

Optimization (optmain)
~~~~~~~~~~~~~~~~~~~~~~
* ``-opt`` / ``--opt_method``: FIRE, TR_LBFGS, rsirfo_block_fsb, rsirfo_block_bofill (saddle), RFO, etc.
* ``-fc`` / ``--calc_exact_hess``: Exact Hessian every N steps.
* ``-mfc`` / ``--calc_model_hess``: Model Hessian every N steps (with ``-modelhess``).
* ``-order`` / ``--saddle_order``: Saddle order (0 = minimum, 1 = TS).
* ``-elec``, ``-spin``: Charge and multiplicity.
* ``-xtb`` / ``-dxtb``: xTB backend (GFN1/2-xTB); dxtb uses autograd Hessian.
* ``-pyscf``: Use PySCF QM engine.
* ``-tcc`` / ``-lcc``: Tight/loose convergence criteria.
* ``-pc``: Projection constraints.
* ``-fix``: Fix atoms; ``-gi``: geometry info every step.
* ``-oniom``: ONIOM (low layer GFN1-xTB).

NEB (nebmain)
~~~~~~~~~~~~~
* Methods: ``-om`` (Onsager–Machlup), ``-qsm`` (QSM), ``-dneb``, ``-bneb``, ``-bneb2``, ``-idpp`` (better initial path), ``-ci`` (climbing image).
* Path spacing: ``-ad``, ``-ads``, ``-adg``, ``-adb``, ``-nd`` (distance), spline/bernstein/savgol variants.
* Nodes: ``-p`` (partition), ``-fixedges``.
* Convergence: ``-aconv``, ``-apply_CI_NEB``; ``-calc_exact_hess`` for NEB.

IEIP (ieipmain)
~~~~~~~~~~~~~~~
* ``-opt`` methods, ``-mi`` micro-iteration, ``-beta`` force, excited-state pairs, model_function_mode (seam, conical, etc.), GNT / ADDF / 2PSHS / dimer options.

MD (mdmain)
~~~~~~~~~~~
* ``-mt`` thermostat (nosehoover default), ``-ts`` timestep, ``-time`` steps, ``-traj`` trajectories.
* ``-constraint_condition`` for bonds/angles/dihedrals; PBC support (``-pbc``).
* Temperature schedule: ``-ct``; PCA / CMDs options.

----------------
Bias Potentials & Constraints (common)
--------------------------------------

Core examples (all parsers share these, values as strings in JSON):
* **AFIR / Bias potential**: ``-ma GAMMA FRAG1 FRAG2`` (e.g., ``-ma 195 1 5``; fragments can be ranges/comma lists).
* **Harmonic distance (keep)**: ``-kp k r0 atoms`` (e.g., ``-kp 2.0 1.0 1,2``).
* **Anharmonic (Morse)**: ``-akp De k r0 atoms``.
* **Angle**: ``-ka k theta0 atoms``; **Dihedral**: ``-kda k phi0 atoms``; cosine form: ``-kdac``.
* **Repulsive potentials**: ``-rp``, ``-rpv2``, Gaussian variants.
* **Mechanochemical forces**: ``-lmefp`` (between fragments), ``-lmefpv2`` (directional).
* **Metadynamics**: ``-metad`` on bonds/angles/dihedrals.
* **Nano-reactor / walls / wells**: ``-nrp``, ``-wp``, ``-wwp``, ``-vpwp``, ``-awp``.
* **Void/point restraints**: ``-vpp``, ``-flux_potential``; distance/angle-dependent variants.

----------------
Examples
--------

Geometry optimization (minimum)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    optmain SN2.xyz -ma 150 1 6 -pyscf -elec -1 -spin 0 -opt rsirfo_block_fsb -modelhess

Saddle point (TS) search
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    optmain aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rsirfo_block_bofill -order 1 -fc 5

NEB path search
~~~~~~~~~~~~~~~

.. code-block:: bash

    nebmain aldol_rxn -xtb GFN2-xTB -ns 50 -adpred 1 -nd 0.5

Molecular dynamics
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    mdmain aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000

Relaxed scan
~~~~~~~~~~~~

.. code-block:: bash

    relaxedscan SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -elec -1 -spin 0 -xtb GFN2-xTB -opt crsirfo_block_fsb -modelhess

Orientation search
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    orientsearch aldol_rxn.xyz -part 1-4 -ma 95 1 5 50 3 11 -nsample 5 -xtb GFN2-xTB -opt rsirfo_block_fsb -modelhess

Conformation search
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    confsearch s8_for_confomation_search_test.xyz -xtb GFN2-xTB -ns 2000

----------------
References & Citation
---------------------

If you use MultiOptPy, please cite:

.. code-block:: text

    ss0832. (2025). MultiOptPy: Multifunctional geometry optimization tools for quantum chemical calculations (v1.20.4).
    Zenodo. https://doi.org/10.5281/zenodo.17973395

References are embedded in source code comments.

----------------
License & Contact
-----------------

* License: **GNU Affero General Public License v3.0**
* Author: ss0832
* Contact: highlighty876[at]gmail.com
