# BiasPotPy
An optimizer for quantum chemical calculation including artificial force induced reaction method using Psi4

This program reproduces AFIR method in python for learning purposes.

About AFIR: https://afir.sci.hokudai.ac.jp/documents/manual/2

## Features

- It is intended to be used in a linux environment.
- It can be used not only with AFIR functions, but also with other bias potentials.



## Required Modules

 - psi4 v15 (Official page:https://psicode.org/   For local download:https://psicode.org/installs/v15/)
 - PySCF v2.4.0 (Not always necessary)
 - numpy
 - matplotlib
 - scipy
 - tblite (If you use extended tight binding (xTB) method, this module is required.)
 - pytorch (for calculating derivatives)
## References

References are given in the source code.

## Usage
```
python optmain.py SN2.xyz -ma 150 1 6
```
```
python optmain.py aldol_rxn.xyz -ma 95 1 5 50 3 11
```

For NEB method

```
python nebmain.py aldol_rxn -xtb GFN2-xTB -ns 50 
```

For iEIP method

```
python ieipmain.py ieip_test -xtb GFN2-xTB 
```
## Options
(optmain.py)

**`-opt`**

Specify the algorithm to be used for structural optimization.

example 1) `-opt FIRE`.

Perform structural optimization using the FIRE method.

example 2) `-opt FSB`: Perform structural optimization using the FIRE method.

Optimize by quasi-Newton method. A combination of the BFGS and SR1 methods is used to update the model Hessian.

example 3) `-opt RFO_FSB`

RFO (Rational Function Optimization method, one of the quasi-Newtonian methods) is used for optimization. A combination of the BFGS and SR1 methods is used to update the model Hessian.

example 4) `-opt RFO_mFSB`

Optimization is performed using the RFO method (Rational Function Optimization method, one of the quasi-Newton methods). A combination of the BFGS and SR1 methods is used to update the model Hessian. The displacement and gradient changes used to update the Hessian include information from two or more previous gradients and displacements.

Available optimization methods:

`RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE, mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, RFO_BFGS, FSB, RFO_FSB, BFGS`

Recommended optimization methods:

- AdaBelief (fast convergence)
- AdaDiff (Carefully descend the potential energy surface)
- FIRE (suitable for finding locally optimal solutions)
- RFO_FSB (can use quasi-Newton method)


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

## TODO

Second derivative of each bias potential is analytically formulated and implemented.
- (2023/09/01) Some of second derivative of bias potentials is implemented with exceptions of keep angle and keep dihedral angle.
- (2023/11/11) Autograd function (pytorch) Calculates all derivatives. 

Implementation of an algorithm to find more than a primary saddle point

## Author

Author of this program is ss0832.

## License

GNU Affero General Public License v3.0
