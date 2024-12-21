BiasPotPy
=========

test page

An optimizer for quantum chemical calculations including the artificial force induced reaction (AFIR) method.

This program reproduces the AFIR method in Python for learning purposes.

.. note::
    This program contains many experimental features, which may not work as intended. Please use them at your own risk.

Features
--------

- Intended for use in a Linux environment.
- Can be used with AFIR functions and other bias potentials.

Required Modules
----------------

- `psi4 <https://psicode.org/>`_ or PySCF
- numpy
- matplotlib
- scipy
- pytorch (for calculating derivatives)

Optional Modules
~~~~~~~~~~~~~~~~

- tblite (Required for the extended tight binding (xTB) method.)
- dxtb (Same as above.)
- ASE

References
----------

References are given in the source code.

Quick Start 
-----

After downloading the repository using `git clone` or similar commands, move to the generated directory and run the following:

.. code-block:: bash

    python optmain.py SN2.xyz -ma 150 1 6

.. code-block:: bash

    python optmain.py aldol_rxn.xyz -ma 95 1 5 50 3 11

For SADDLE calculation:

.. code-block:: bash

    python optmain.py aldol_rxn_PT.xyz -xtb GFN2-xTB -opt RFO3_Bofill -order 1 -fc 5

For NEB method:

.. code-block:: bash

    python nebmain.py aldol_rxn -xtb GFN2-xTB -ns 50 

For iEIP method:

.. code-block:: bash

    python ieipmain.py ieip_test -xtb GFN2-xTB 

For Molecular Dynamics (MD):

.. code-block:: bash

    python mdmain.py aldol_rxn_PT.xyz -xtb GFN2-xTB -temp 298 -traj 1 -time 100000

(Default deterministic algorithm for MD is Nosé–Hoover thermostat.)

For orientation search:

.. code-block:: bash

    python orientation_search.py aldol_rxn.xyz -part 1-4 -ma 95 1 5 50 3 11 -nsample 5 -xtb GFN2-xTB 


Options
-------

**`-opt`**

Specify the algorithm to be used for structural optimization.

Example:

- `-opt FIRE`: Perform structural optimization using the FIRE method.
- `-opt RFO_FSB`: Use RFO (Rational Function Optimization) combined with BFGS and SR1.

Available methods:

- FIRE (suitable for locally optimal solutions)
- RFO_FSB (quasi-Newton method)
- RFO3_Bofill (saddle point calculation)

**`-ma`**

Add potential by AFIR function. Specify energy (kJ/mol) and atom/fragments.

Example:

- `-ma 195 1 5`: Apply 195 kJ/mol potential between atoms 1 and 5.
- `-ma -195 1-3 5,6`: Apply -195 kJ/mol between fragment 1-3 and fragment 5,6.

**Other options**:

See the original file for further details.

Author
------

The author of this program is ss0832.

License
-------

GNU Affero General Public License v3.0

