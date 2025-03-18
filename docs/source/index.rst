MultiOptPy Documentation
=======================

    **Version**: v1.4.5  
    **Repository**: `https://github.com/ss0832/MultiOptPy <https://github.com/ss0832/MultiOptPy>`_  
    *An optimizer for quantum chemical calculation including artificial force induced reaction method*

MultiOptPy is a Python program that supports various types of calculations (e.g., geometry optimization, NEB, MD), allowing the application of bias potentials and external forces to facilitate advanced modeling in computational chemistry.

Table of Contents
----------------
1. `Installation`_
2. `Examples`_
3. `Main Commands`_
   - `Optimize`_
   - `NEB`_
   - `MD`_
   - `IEIP`_
4. `Bias Potentials and Forces`_
5. `Extended Tight Binding Options`_
6. `References`_
7. `License`_

------------

Installation
-----------

.. code-block:: bash

    git clone https://github.com/ss0832/MultiOptPy.git

or download the latest version of this program (v1.4.5 (2025/3/5) ).


Requirements
~~~~~~~~~~~
- psi4 or PySCF (for quantum mechanics calculation)
- numpy 
- matplotlib (for visualization)
- scipy 
- pytorch (for calculating differentials)

Optional

 - tblite (If you use extended tight binding (xTB) method, this module is required.)
 - dxtb (same as above)
 - ASE 

------------

Examples 
--------

Below are some examples of MultiOptPy usage.

After ``git clone``,

.. code-block:: bash

    # Basic geometry optimization
    python optmain.py SN2.xyz

    # Transition state optimization (1st-order saddle)
    python optmain.py aldol_rxn_PT.xyz -xtb GFN2-xTB -opt rfo3_bofill -order 1 -fc 5

    # NEB calculation
    python nebmain.py aldol_rxn -xtb GFN2-xTB -ns 50 

    # MD simulation
    python mdmain.py aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000

    # Reaction path with AFIR
    python optmain.py aldol_rxn.xyz -ma 95 1 5 50 3 11
    python optmain.py SN2.xyz -ma 150 1 6

    python optmain.py aldol_rxn.xyz -ma 95 1 5 50 3 11 -modelhess
    python optmain.py SN2.xyz -ma 150 1 6 -modelhess

    # You can practice AFIR method to analyze other reactions by using .xyz files in "test" directory.

    # Orientation search 
    python orientation_search.py aldol_rxn.xyz -part 1-4 -ma 95 1 5 50 3 11 -nsample 5 -xtb GFN2-xTB 

    # Conformation search
    python conformation_search.py s8_for_confomation_search_test.xyz -xtb GFN2-xTB -ns 2000

    # Relaxed scan (Similar to functions implemented in Gaussian)
    python relaxed_scan.py SN2.xyz -nsample 8 -scan bond 1,2 1.3,2.6 -elec -1 -spin 0 -pyscf

    # Constraint optimization (fix the distance between 1st-atom and 5th atom)
    python optmain.py aldol_rxn.xyz -xtb GFN2-xTB -ns 50 -pc bond 1,5 -ma 95 1 5 50 3 11

    # Constraint optimization (fix ∠1st_atom-5th_atom-6th_atom)
    python optmain.py aldol_rxn.xyz -xtb GFN2-xTB -ns 50 -pc angle 1,5,6 -ma 95 1 5 50 3 11

    # Constraint optimization (fix dihedral angle of φ(8-6-5-7))
    python optmain.py aldol_rxn.xyz -xtb GFN2-xTB -ns 50 -pc dihedral 8,6,5,7 -ma 95 1 5 50 3 11


Main Commands
------------

Optimize Command
~~~~~~~~~~~~~~~

Run structure optimization with various methods and bias potentials.

.. code-block:: bash

    python optmain.py input.xyz [options]

Basic Options
^^^^^^^^^^^^

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-bs``, ``--basisset``
     - Basis set for QM calculation
     - ``6-31G(d)``
   * - ``-func``, ``--functional``
     - Functional for QM calculation
     - ``b3lyp``
   * - ``-sub_bs``, ``--sub_basisset``
     - Sub basis set for specific atoms
     - None
   * - ``-es``, ``--excited_state``
     - Calculate excited state (e.g., S1 => ``1``)
     - ``0``
   * - ``-ns``, ``--NSTEP``
     - Maximum number of optimization iterations
     - ``1000``
   * - ``-core``, ``--N_THREAD``
     - Number of CPU threads to use
     - ``8``
   * - ``-mem``, ``--SET_MEMORY``
     - Memory allocation for calculation
     - ``2GB``
   * - ``-d``, ``--DELTA``
     - Move step
     - ``x``
   * - ``-u``, ``--unrestrict``
     - Use unrestricted method (radical reactions)
     - False
   * - ``-fix``, ``--fix_atoms``
     - Fix atoms during optimization (e.g., ``1,2,3-6``)
     - None
   * - ``-elec``, ``--electronic_charge``
     - Formal electronic charge
     - ``0``
   * - ``-spin``, ``--spin_multiplicity``
     - Spin multiplicity
     - ``1``

Advanced Options
^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-opt``, ``--opt_method``
     - Optimization method (e.g. ``FIRELARS``, ``RFO``)
     - ``FIRELARS``
   * - ``-fc``, ``--calc_exact_hess``
     - Calculate exact Hessian every N steps
     - ``-1``
   * - ``-mfc``, ``--calc_model_hess``
     - Calculate model Hessian every N steps (this option is available by using this with ``-modelhess``)
     - ``50``
   * - ``-saddle``, ``--saddle_order``
     - Optimize to nth-order saddle point
     - ``0``
   * - ``-pyscf``, ``--pyscf``
     - Use PySCF instead of Psi4
     - False
   * - ``-tcc``, ``--tight_convergence_criteria``
     - Use tight optimization criteria
     - False
   * - ``-lcc``, ``--loose_convergence_criteria``
     - Use loose optimization criteria
     - False
   * - ``-modelhess``, ``--use_model_hessian``
     - Use model Hessian instead of exact
     - False
   * - ``-pc``, ``--projection_constrain``
     - Constrain gradient/Hessian via projection
     - None

------------

NEB Command
~~~~~~~~~~

Perform Nudged Elastic Band calculations for reaction path.

.. code-block:: bash

    python nebmain.py input_folder [options]

Basic Options
^^^^^^^^^^^^

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-bs``, ``--basisset``
     - Basis set for QM calculation
     - ``6-31G(d)``
   * - ``-func``, ``--functional``
     - Functional for QM calculation
     - ``b3lyp``
   * - ``-sub_bs``, ``--sub_basisset``
     - Sub basis set for specific atoms
     - None
   * - ``-u``, ``--unrestrict``
     - Use unrestricted method
     - False
   * - ``-es``, ``--excited_state``
     - Calculate excited state
     - ``0``
   * - ``-ns``, ``--NSTEP``
     - Number of iterations
     - ``10``
   * - ``-p``, ``--partition``
     - Number of nodes
     - ``0``
   * - ``-core``, ``--N_THREAD``
     - Number of CPU threads
     - ``8``
   * - ``-mem``, ``--SET_MEMORY``
     - Memory allocation for calculation
     - ``1GB``
   * - ``-elec``, ``--electronic_charge``
     - Formal electronic charge
     - ``0``
   * - ``-spin``, ``--spin_multiplicity``
     - Spin multiplicity
     - ``1``

NEB Method Options
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 35 50 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-om``, ``--OM``
     - Use Onsager-Machlup NEB method
     - False
   * - ``-lup``, ``--LUP``
     - Use locally updated planes method
     - False
   * - ``-dneb``, ``--DNEB``
     - Use doubly NEB method
     - False
   * - ``-idpp``, ``--use_image_dependent_pair_potential``
     - Use IDPP method to generate better initial path than LST (linear synchronous transit) method
     - False

------------

MD Command
~~~~~~~~~

Run *Ab initio* molecular dynamics (AIMD) simulations.

.. code-block:: bash

    python mdmain.py input.xyz [options]

Basic Options
^^^^^^^^^^^^

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-bs``, ``--basisset``
     - Basis set for QM calculation
     - ``6-31G(d)``
   * - ``-func``, ``--functional``
     - Functional for QM calculation
     - ``b3lyp``
   * - ``-sub_bs``, ``--sub_basisset``
     - Sub basis set for specific atoms
     - None
   * - ``-es``, ``--excited_state``
     - Calculate excited state (PySCF)
     - ``0``
   * - ``-time``, ``--NSTEP``
     - Total simulation time steps
     - ``100000``
   * - ``-traj``, ``--TRAJECTORY``
     - Number of trajectories to generate
     - ``1``
   * - ``-temp``, ``--temperature``
     - Temperature in Kelvin
     - ``298.15``
   * - ``-ts``, ``--timestep``
     - Time step in atomic units
     - ``0.1``
   * - ``-mt``, ``--mdtype``
     - MD thermostat type (``nosehoover`` or ``nvt``, ``nve``, etc.)
     - ``nosehoover``

------------

IEIP Command
~~~~~~~~~~~

Perform Initial-End point Interpolation Path calculations.

.. code-block:: bash

    python ieipmain.py input_folder [options]

Basic Options
^^^^^^^^^^^^

.. list-table::
   :widths: 25 60 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-bs``, ``--basisset``
     - Basis set for QM calculation
     - ``6-31G(d)``
   * - ``-func``, ``--functional``
     - Functional for QM calculation
     - ``b3lyp``
   * - ``-ns``, ``--NSTEP``
     - Number of iterations
     - ``999``
   * - ``-opt``, ``--opt_method``
     - Optimization method
     - ``FIRELARS``
   * - ``-sub_bs``, ``--sub_basisset``
     - Sub basis set for specific atoms
     - None
   * - ``-mi``, ``--microiter``
     - Microiteration for relaxing reaction pathways
     - ``0``
   * - ``-beta``, ``--BETA``
     - Force for optimization
     - ``1.0``

------------

Bias Potentials and Forces
-------------------------

MultiOptPy supports a variety of bias potentials and forces.

Artificial Force-Induced Reaction (AFIR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    -ma GAMMA FRAGM1 FRAGM2

- Example 1:

  .. code-block:: bash

      -ma 195 1 5

  Apply a potential of 195 kJ/mol (pushing force) to the first atom and the fifth atom as a pair.

- Example 2:

  .. code-block:: bash

      -ma 195 1 5 195 3 11

  Add the potential of 195 kJ/mol (pushing force) by the pair of the first atom and the fifth atom. Then add the potential of 195 kJ/mol (pushing force) by the pair of the third atom and the eleventh atom.

- Example 3:

  .. code-block:: bash

      -ma -195 1-3 5,6

  Add the potential of -195 kJ/mol (pulling force) by the fragment consisting of the 1st-3rd atoms paired with the fragments consisting of the 5th and 6th atoms.


Keep Potential (Harmonic Restraint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V(r) = 0.5k(r - r_0)^2

``spring const. k (a.u.) keep distance [$ r_0] (ang.) atom1,atom2 ...``

.. code-block:: bash

    -kp SPRING_CONST DISTANCE ATOMS

- Example:

  .. code-block:: bash

      -kp 0.1 2.5 1,2

Keep Angle Potential
~~~~~~~~~~~~~~~~~~

V(θ) = 0.5k(θ - θ_0)^2

``spring const.(a.u.) keep angle (degrees) atom1,atom2,atom3``

.. code-block:: bash

    -ka SPRING_CONST ANGLE ATOMS

- Example:

  .. code-block:: bash

      -ka 2.0 60 1,2,3

Keep Dihedral Angle Potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~

V(φ) = 0.5k(φ - φ_0)^2

``spring const.(a.u.) keep dihedral angle (degrees) atom1,atom2,atom3,atom4 ...``

.. code-block:: bash

    -kda SPRING_CONST ANGLE ATOMS

- Example:

  .. code-block:: bash

      -kda 2.0 60 1,2,3,4

------------

Extended Tight Binding Options
-----------------------------

.. list-table::
   :widths: 35 50 15
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-xtb``, ``--usextb``
     - Use extended tight binding method
     - ``None``
   * - ``-dxtb``, ``--usedxtb``
     - Use dxtb implementation of xTB
     - ``None``
   * - ``-cpcm``, ``--cpcm_solv_model``
     - Use CPCM solvent model for xTB
     - None
   * - ``-alpb``, ``--alpb_solv_model``
     - Use ALPB solvent model for xTB
     - None

------------

References
---------

The references for this program are embedded within the source code. Please refer to the comments and documentation within the code files for detailed citations and attributions.


License
------

MultiOptPy is licensed under the **GNU General Public License v3.0**.

(C) 2023-2025 ss0832

Contact
~~~~~~~
highlighty876 [at] gmail.com
