# **MultiOptPy Configuration Reference**

(ver. 1.19.4)

This document outlines all available settings for MultiOptPy, based on the definitions in interface.py. These settings can be configured either via a JSON file (like config.json used by run\_autots.py) or as command-line arguments (used by optmain.py, nebmain.py, etc.).

## **Basic Rules for JSON Configuration**

1. **Key-Value Mapping**: The JSON key (e.g., "NSTEP") directly corresponds to the command-line argument's destination (e.g., \--NSTEP).  
2. **Omitted Keys Use Defaults**: If a key is **omitted** from the JSON file, the program will automatically use the Default value listed in this guide.  
3. **Data Types**:  
   * Use standard JSON types: string, number (for integers and floats), and boolean (true/false).  
   * Arguments that accept multiple values (like nargs="\*") should be specified as a JSON array (e.g., "opt\_method": \["FIRELARS"\]).  
   * Bias/Constraint settings (Section 3\) are parsed from strings, so they must be provided as a list\[string\] (e.g., "manual\_AFIR": \["300", "4", "17"\]).  
4. **nargs='?' Arguments**: For arguments like use\_model\_hessian or conjugate\_gradient:  
   * To **enable** the feature with its specific value, provide the string (e.g., "use\_model\_hessian": "fischerd3").  
   * To **disable** it (and use the default), specify null (e.g., "use\_model\_hessian": null).

## **1\. Optimization Settings (optimizeparser)**

Used by: optmain.py, run\_autots.py (Steps 1, 3, & 4\)

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | \-bs, \--basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| functional | \-func, \--functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| sub\_basisset | \-sub\_bs, \--sub\_basisset | sub\_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub\_basisset": \["I", "LanL2DZ"\] |
| effective\_core\_potential | \-ecp, \--effective\_core\_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis\_set name)". | list\[string\] | \[\] | "effective\_core\_potential": \["I", "LanL2DZ"\] |
| excited\_state | \-es, \--excited\_state | calculate excited state (default: 0\) (e.g.) if you set spin\_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | int | 0 | "excited\_state": 1 |
| NSTEP | \-ns, \--NSTEP | number of iteration (default: 1000\) | int | 1000 | "NSTEP": 500 |
| N\_THREAD | \-core, \--N\_THREAD | threads | int | 8 | "N\_THREAD": 16 |
| SET\_MEMORY | \-mem, \--SET\_MEMORY | use mem(ex. 1GB) | string | "2GB" | "SET\_MEMORY": "8GB" |
| DELTA | \-d, \--DELTA | move step | string | "x" | "DELTA": "y" |
| max\_trust\_radius | \-tr, \--max\_trust\_radius | max trust radius to restrict step size (unit: ang.) (default: 0.1 for n-th order saddle point optimization, 0.5 for minimum point optimization) (notice: default minimum trust radius is 0.01) | float | None | "max\_trust\_radius": 0.2 |
| min\_trust\_radius | \-mintr, \--min\_trust\_radius | min trust radius to restrict step size (unit: ang.) (default: 0.01) | float | 0.01 | "min\_trust\_radius": 0.005 |
| unrestrict | \-u, \--unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| fix\_atoms | \-fix, \--fix\_atoms | fix atoms (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \[\] | "fix\_atoms": \["1,2,3-6"\] |
| geom\_info | \-gi, \--geom\_info | calculate atom distances, angles, and dihedral angles in every iteration (energy\_profile is also saved.) (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \["1"\] | "geom\_info": \["1,2", "1,2,3"\] |
| dissociate\_check | \-dc, \--dissociate\_check | Terminate calculation if distance between two fragments is exceed this value. (default) 100 \[ang.\] | string | "100" | "dissociate\_check": "50.0" |
| opt\_method | \-opt, \--opt\_method | optimization method for QM calculation (default: FIRE) (mehod\_list:(steepest descent method group) FIRE, CG etc. (quasi-Newton method group) rsirfo\_fsb rsirfo\_bofill etc.) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. \[\[steepest descent\] \[quasi-Newton method\]\]) (ex.) \[opt\_method\] | list\[string\] | \["FIRELARS"\] | "opt\_method": \["rsirfo\_block\_fsb"\] |
| calc\_exact\_hess | \-fc, \--calc\_exact\_hess | calculate exact hessian per geometry optimization steps and IRC steps (ex.) \[steps per one hess calculation\] | int | \-1 | "calc\_exact\_hess": 5 |
| calc\_model\_hess | \-mfc, \--calc\_model\_hess | calculate model hessian per steps (ex.) \[steps per one hess calculation\] | int | 50 | "calc\_model\_hess": 25 |
| usextb | \-xtb, \--usextb | use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| usedxtb | \-dxtb, \--usedxtb | use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usedxtb": "GFN1-xTB" |
| sqm1 | \-sqm1, \--sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | \-sqm2, \--sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| cpcm\_solv\_model | \-cpcm, \--cpcm\_solv\_model | use CPCM solvent model for xTB (Default setting is not using this model.) (ex.) water | string | None | "cpcm\_solv\_model": "water" |
| alpb\_solv\_model | \-alpb, \--alpb\_solv\_model | use ALPB solvent model for xTB (Default setting is not using this model.) (ex.) water | string | None | "alpb\_solv\_model": "toluene" |
| pyscf | \-pyscf, \--pyscf | use pyscf module. | boolean | false | "pyscf": true |
| electronic\_charge | \-elec, \--electronic\_charge | formal electronic charge (ex.) \[charge (0)\] | int | 0 | "electronic\_charge": \-1 |
| spin\_multiplicity | \-spin, \--spin\_multiplicity | spin multiplicity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha \- Nbeta)) (ex.) \[multiplicity (0)\] | int | 1 | "spin\_multiplicity": 2 |
| saddle\_order | \-order, \--saddle\_order | optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) \[order (0)\] | int | 0 | "saddle\_order": 1 |
| cmds | \-cmds, \--cmds | Apply classical multidimensional scaling to calculated approx. reaction path. | boolean | false | "cmds": true |
| pca | \-pca, \--pca | Apply principal component analysis to calculated approx. reaction path. | boolean | false | "pca": true |
| koopman | \-km, \--koopman | Apply Koopman model to analyze the convergence | boolean | false | "koopman": true |
| intrinsic\_reaction\_coordinates | \-irc, \--intrinsic\_reaction\_coordinates | Calculate intrinsic reaction coordinates. (ex.) \[\[step\_size\], \[max\_step\], \[IRC\_method\]\] (Recommended) \[0.5 300 lqa\] | list\[string\] | \[\] | "intrinsic\_reaction\_coordinates": \["0.5", "200", "lqa"\] |
| opt\_fragment | \-of, \--opt\_fragment | Several atoms are grouped together as fragments and optimized. (This method does not work if you use quasi-newton method for optimazation.) (ex.) \[\[atoms (ex.) 1-4\] ...\] | list\[string\] | \[\] | "opt\_fragment": \["1-4", "5-8"\] |
| dft\_grid | \-grid, \--dft\_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft\_grid": 4 |
| othersoft | \-os, \--othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace\_mp etc. | string | "None" | "othersoft": "orca" |
| software\_path\_file | \-osp, \--software\_path\_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software\_path.conf | string | "./software\_path.conf" | "software\_path\_file": "/path/to/my.conf" |
| tight\_convergence\_criteria | \-tcc, \--tight\_convergence\_criteria | apply tight opt criteria. | boolean | false | "tight\_convergence\_criteria": true |
| loose\_convergence\_criteria | \-lcc, \--loose\_convergence\_criteria | apply loose opt criteria. | boolean | false | "loose\_convergence\_criteria": true |
| use\_model\_hessian | \-modelhess, \--use\_model\_hessian | use model hessian. (Default: not using model hessian If you specify only option, Improved Lindh \+ Grimme's D3 dispersion model hessian is used.) (ex. lindh, gfnff, gfn0xtb, fischer, fischerd3, fischerd4, schlegel, swart, lindh2007, lindh2007d3, lindh2007d4) | string | null | "use\_model\_hessian": "fischerd3" |
| shape\_conditions | \-sc, \--shape\_conditions | Exit optimization if these conditions are not satisfied. (e.g.) \[\[(ang.) gt(lt) 2,3 (bond)\] \[(deg.) gt(lt) 2,3,4 (bend)\] ...\] \[\[(deg.) gt(lt) 2,3,4,5 (torsion)\] ...\] | list\[string\] | \[\] | "shape\_conditions": \["gt", "2,3", "lt", "2,3,4"\] |
| projection\_constrain | \-pc, \--projection\_constrain | apply constrain conditions with projection of gradient and hessian (ex.) \[\[(constraint condition name) (atoms(ex. 1,2))\] ...\] | list\[string\] | \[\] | "projection\_constrain": \["bond", "1,2"\] |
| oniom\_flag | \-oniom, \--oniom\_flag | apply ONIOM method (low layer: GFN1-xTB) (ex.) \[(atom\_number of high layer (ex. 1,2))\] (caution) \-pc option is not available. If there are not link atoms, please input "none" | list\[string\] | \[\] | "oniom\_flag": \["1,2,5-10", "none", "GFN1-xTB"\] |
| frequency\_analysis | \-freq, \--frequency\_analysis | Perform normal vibrational analysis after converging geometry optimization. (Caution: Unable to use this analysis with oniom method) | boolean | false | "frequency\_analysis": true |
| temperature | \-temp, \--temperature | temperatrue to calculate thermochemistry (Unit: K) (default: 298.15K) | float | 298.15 | "temperature": 300.0 |
| pressure | \-press, \--pressure | pressure to calculate thermochemistry (Unit: Pa) (default: 101325Pa) | float | 101325 | "pressure": 100000 |

## **2\. NEB Settings (nebparser)**

Used by: nebmain.py, run\_autots.py (Step 2\)

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | \-bs, \--basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| sub\_basisset | \-sub\_bs, \--sub\_basisset | sub\_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub\_basisset": \["I", "LanL2DZ"\] |
| effective\_core\_potential | \-ecp, \--effective\_core\_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis\_set name)". | list\[string\] | \[\] | "effective\_core\_potential": \["I", "LanL2DZ"\] |
| functional | \-func, \--functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| unrestrict | \-u, \--unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| excited\_state | \-es, \--excited\_state | calculate excited state (default: 0\) (e.g.) if you set spin\_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | int | 0 | "excited\_state": 1 |
| NSTEP | \-ns, \--NSTEP | iter. number | int | 10 | "NSTEP": 50 |
| OM | \-om, \--OM | J. Chem. Phys. 155, 074103 (2021) doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action. | boolean | false | "OM": true |
| LUP | \-lup, \--LUP | J. Chem. Phys. 92, 1510–1511 (1990) doi:https://doi.org/10.1063/1.458112 locally updated planes (LUP) method | boolean | false | "LUP": true |
| BNEB | \-bneb, \--BNEB | NEB using Wilson's B matrix for calculating the perpendicular force. | boolean | false | "BNEB": true |
| BNEB2 | \-bneb2, \--BNEB2 | NEB using Wilson's B matrix for calculating the perpendicular force with parallel spring force. | boolean | false | "BNEB2": true |
| DNEB | \-dneb, \--DNEB | J. Chem. Phys. 120, 2082–2094 (2004) doi:https://doi.org/10.1063/1.1636455 doubly NEB method (DNEB) method | boolean | false | "DNEB": true |
| NESB | \-nesb, \--NESB | J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169 Nudged elastic stiffness band (NESB) method | boolean | false | "NESB": true |
| DMF | \-dmf, \--DMF | Direct Max Flux (DMF) method | boolean | false | "DMF": true |
| EWBNEB | \-ewbneb, \--EWBNEB | Energy-weighted Nudged elastic band method | boolean | false | "EWBNEB": true |
| QSM | \-qsm, \--QSM | Quadratic String Method (J. Chem. Phys. 124, 054109 (2006)) | boolean | false | "QSM": true |
| ANEB | \-aneb, \--ANEB | Adaptic NEB method (ref.: J. Chem. Phys. 117, 4651 (2002)) (Usage: \-aneb \[interpolation\_num (ex. 2)\] \[frequency (ex. 5)\], Default setting is not applying adaptic NEB method.) | list\[string\] | None | "ANEB": \["3", "5"\] |
| use\_image\_dependent\_pair\_potential | \-idpp, \--use\_image\_dependent\_pair\_potential | use image dependent pair potential (IDPP) method (ref. arXiv:1406.1512v1) | boolean | false | "use\_image\_dependent\_pair\_potential": true |
| use\_correlated\_flat\_bottom\_elastic\_network\_model | \-cfbenm, \--use\_correlated\_flat\_bottom\_elastic\_network\_model | use correlated flat-bottom elastic network model (CFBENM) method (ref. s: J.Chem.TheoryComput.2025,21,3513−3522) | boolean | false | "use\_correlated\_flat\_bottom\_elastic\_network\_model": true |
| align\_distances | \-ad, \--align\_distances | distribute images at equal intervals on the reaction coordinate | int | 0 | "align\_distances": 9999 |
| align\_distances\_spline | \-ads, \--align\_distances\_spline | distribute images at equal intervals on the reaction coordinate using spline interpolation | int | 0 | "align\_distances\_spline": 5 |
| align\_distances\_spline\_ver2 | \-ads2, \--align\_distances\_spline\_ver2 | distribute images at equal intervals on the reaction coordinate using spline interpolation ver.2 | int | 0 | "align\_distances\_spline\_ver2": 5 |
| align\_distances\_geodesic | \-adg, \--align\_distances\_geodesic | distribute images at equal intervals on the reaction coordinate using geodesic interpolation | int | 0 | "align\_distances\_geodesic": 5 |
| align\_distances\_bernstein | \-adb, \--align\_distances\_bernstein | distribute images at equal intervals on the reaction coordinate using Bernstein interpolation | int | 0 | "align\_distances\_bernstein": 5 |
| align\_distances\_savgol | \-adsg, \--align\_distances\_savgol | distribute images at equal intervals on the reaction coordinate using Savitzky-Golay interpolation (ex.) \[\[iteration(int)\],\[window\_size(int, 5 is recommended)\],\[poly\_order(int) 3 is recommended\]\] (default: 0,0,0 (not using Savitzky-Golay interpolation)) | string | "0,0,0" | "align\_distances\_savgol": "5,5,3" |
| node\_distance | \-nd, \--node\_distance | distribute images at equal intervals linearly based ont specific distance (ex.) \[distance (ang.)\] (default: None) | float | None | "node\_distance": 0.8 |
| node\_distance\_spline | \-nds, \--node\_distance\_spline | distribute images at equal intervals using spline interpolation based ont specific distance (ex.) \[distance (ang.)\] (default: None) | float | None | "node\_distance\_spline": 0.8 |
| node\_distance\_bernstein | \-ndb, \--node\_distance\_bernstein | distribute images at equal intervals using Bernstein interpolation based ont specific distance (ex.) \[distance (ang.)\] (default: None) | float | None | "node\_distance\_bernstein": 0.8 |
| node\_distance\_savgol | \-ndsg, \--node\_distance\_savgol | distribute images at equal intervals using Savitzky-Golay interpolation based ont specific distance (ex.) \[\[distance (ang.)\],\[window\_size(int, 5 is recommended)\],\[poly\_order(int) 3 is recommended\]\] (default: None) | string | None | "node\_distance\_savgol": "0.8,5,3" |
| partition | \-p, \--partition | number of nodes | int | 0 | "partition": 10 |
| N\_THREAD | \-core, \--N\_THREAD | threads | int | 8 | "N\_THREAD": 16 |
| SET\_MEMORY | \-mem, \--SET\_MEMORY | use mem(ex. 1GB) | string | "1GB" | "SET\_MEMORY": "4GB" |
| apply\_CI\_NEB | \-cineb, \--apply\_CI\_NEB | apply CI\_NEB method | int | 99999 | "apply\_CI\_NEB": 10 |
| steepest\_descent | \-sd, \--steepest\_descent | apply steepest\_descent method | int | 99999 | "steepest\_descent": 5 |
| conjugate\_gradient | \-cg, \--conjugate\_gradient | apply conjugate\_gradient method for path optimization (Available update method of CG parameters :FR, PR, HS, DY, HZ), default update method is HS.) | string | false | "conjugate\_gradient": "HS" |
| memory\_limited\_BFGS | \-lbfgs, \--memory\_limited\_BFGS | apply L-BFGS method for path optimization | boolean | false | "memory\_limited\_BFGS": true |
| not\_ts\_optimization | \-notsopt, \--not\_ts\_optimization | not apply TS optimization during NEB calculation | boolean | false | "not\_ts\_optimization": true |
| calc\_exact\_hess | \-fc, \--calc\_exact\_hess | calculate exact hessian per steps (ex.) \[steps per one hess calculation\] | int | \-1 | "calc\_exact\_hess": 1 |
| global\_quasi\_newton | \-gqnt, \--global\_quasi\_newton | use global quasi-Newton method | boolean | false | "global\_quasi\_newton": true |
| usextb | \-xtb, \--usextb | use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| usedxtb | \-dxtb, \--usedxtb | use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usedxtb": "GFN1-xTB" |
| sqm1 | \-sqm1, \--sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | \-sqm2, \--sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| pyscf | \-pyscf, \--pyscf | use pyscf module. | boolean | false | "pyscf": true |
| fixedges | \-fe, \--fixedges | fix edges of nodes (1=initial\_node, 2=end\_node, 3=both\_nodes) | int | 0 | "fixedges": 3 |
| fix\_atoms | \-fix, \--fix\_atoms | fix atoms (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \[\] | "fix\_atoms": \["1,2,3-6"\] |
| projection\_constrain | \-pc, \--projection\_constrain | apply constrain conditions with projection of gradient and hessian (ex.) \[\[(constraint condition name) (atoms(ex. 1,2))\] ...\] | list\[string\] | \[\] | "projection\_constrain": \["bond", "1,2"\] |
| cpcm\_solv\_model | \-cpcm, \--cpcm\_solv\_model | use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "cpcm\_solv\_model": "water" |
| alpb\_solv\_model | \-alpb, \--alpb\_solv\_model | use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "alpb\_solv\_model": "toluene" |
| save\_pict | \-spng, \--save\_pict | Save picture for visualization. | boolean | false | "save\_pict": true |
| apply\_convergence\_criteria | \-aconv, \--apply\_convergence\_criteria | Apply convergence criteria for NEB calculation. | boolean | false | "apply\_convergence\_criteria": true |
| climbing\_image | \-ci, \--climbing\_image | climbing image for NEB calculation. (start of ITR., interval) The default setting is not using climbing image. | list\[int\] | \[999999, 999999\] | "climbing\_image": \[10, 1\] |
| electronic\_charge | \-elec, \--electronic\_charge | formal electronic charge (ex.) \[charge (0)\] | int | 0 | "electronic\_charge": \-1 |
| spin\_multiplicity | \-spin, \--spin\_multiplicity | spin multiplcity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha \- Nbeta)) (ex.) \[multiplcity (0)\] | int | 1 | "spin\_multiplicity": 2 |
| dft\_grid | \-grid, \--dft\_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft\_grid": 4 |
| use\_model\_hessian | \-modelhess, \--use\_model\_hessian | use model hessian. (Default: not using model hessian If you specify only option, Fischer \+ Grimme's D3 dispersion model hessian is used.) (ex. lindh, gfnff, gfn0xtb, fischer, fischerd3, fischerd4, schlegel, swart, lindh2007, lindh2007d3, lindh2007d4) | string | null | "use\_model\_hessian": "fischerd3" |
| calc\_model\_hess | \-mfc, \--calc\_model\_hess | calculate model hessian per steps (ex.) \[steps per one hess calculation\] | int | 50 | "calc\_model\_hess": 25 |
| othersoft | \-os, \--othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace\_mp etc. | string | "None" | "othersoft": "orca" |
| software\_path\_file | \-osp, \--software\_path\_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software\_path.conf | string | "./software\_path.conf" | "software\_path\_file": "/path/to/my.conf" |
| ratio\_of\_rfo\_step | \-rrs, \--ratio\_of\_rfo\_step | ratio of rfo step (default: 0.5). This option is for optimizer using Hessian (-fc or \-modelhess). | float | 0.5 | "ratio\_of\_rfo\_step": 0.25 |

## **3\. Common Bias/Constraint Settings (parser\_for\_biasforce)**

Used by all parsers (optimizeparser, nebparser, ieipparser, mdparser).  
Note: All arguments in this section are parsed from strings. In JSON, they must be provided as a list\[string\], where each set of N elements forms one definition.

| JSON Key | CMD Argument | Elements per Set | Description | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- |
| manual\_AFIR | \-ma, \--manual\_AFIR | 3 | manual-AFIR (ex.) \[\[Gamma(kJ/mol)\] \[Fragm.1(ex. 1,2,3-5)\] \[Fragm.2\] ...\] | "manual\_AFIR": \["300", "1,2", "3-5"\] |
| repulsive\_potential | \-rp, \--repulsive\_potential | 5 | Add LJ repulsive\_potential based on UFF (ex.) \[\[well\_scale\] \[dist\_scale\] \[Fragm.1(ex. 1,2,3-5)\] \[Fragm.2\] \[scale or value(kJ/mol ang.)\] ...\] | "repulsive\_potential": \["1.0", "1.0", "1,2", "3-5", "scale"\] |
| repulsive\_potential\_v2 | \-rpv2, \--repulsive\_potential\_v2 | 10 | Add LJ repulsive\_potential based on UFF (ver.2) (eq. V \= ε\[A \* (σ/r)^(rep) \- B \* (σ/r)^(attr)\]) (ex.) \[...\] | "repulsive\_potential\_v2": \["1.0", "1.0", "3.0", "1.0", "0.0", "12.0", "6.0", "1,2", "3-5", "scale"\] |
| repulsive\_potential\_gaussian | \-rpg, \--repulsive\_potential\_gaussian | 7 | Add LJ repulsive\_potential based on UFF (ver.2) (eq. V \= ε\_LJ\[(σ/r)^(12) \- 2 \* (σ/r)^(6)\] \- ε\_gau \* exp(-((r-σ\_gau)/b)^2)) (ex.) \[...\] | "repulsive\_potential\_gaussian": \["10.0", "3.0", "20.0", "3.0", "0.5", "1,2", "3-5"\] |
| cone\_potential | \-cp, \--cone\_potential | 6 | Add cone type LJ repulsive\_potential based on UFF (ex.) \[\[well\_value (epsilon) (kJ/mol)\] \[dist (sigma) (ang.)\] \[cone angle (deg.)\] \[LJ center atom (1)\] \[three atoms (2,3,4) \] \[target atoms (5-9)\] ...\] | "cone\_potential": \["10.0", "3.0", "90.0", "1", "2,3,4", "5-9"\] |
| flux\_potential | \-fp, \--flux\_potential | 4 | Add potential to make flow. ( k/p\*(x-x\_0)^p )(ex.) \[\[x,y,z (constant (a.u.))\] \[x,y,z (order)\] \[x,y,z coordinate (ang.)\] \[Fragm.(ex. 1,2,3-5)\] ...\] | "flux\_potential": \["1,0,0", "2,0,0", "0,0,0", "1,2"\] |
| keep\_pot | \-kp, \--keep\_pot | 3 | keep potential 0.5\*k\*(r \- r0)^2 (ex.) \[\[spring const.(a.u.)\] \[keep distance (ang.)\] \[atom1,atom2\] ...\] | "keep\_pot": \["0.5", "1.5", "1,2"\] |
| keep\_pot\_v2 | \-kpv2, \--keep\_pot\_v2 | 4 | keep potential\_v2 0.5\*k\*(r \- r0)^2 (ex.) \[\[spring const.(a.u.)\] \[keep distance (ang.)\] \[Fragm.1\] \[Fragm.2\] ...\] | "keep\_pot\_v2": \["0.5", "3.0", "1-3", "4-6"\] |
| anharmonic\_keep\_pot | \-akp, \--anharmonic\_keep\_pot | 4 | Morse potential De\*\[1-exp(-((k/2\*De)^0.5)\*(r \- r0))\]^2 (ex.) \[\[potential well depth (a.u.)\] \[spring const.(a.u.)\] \[keep distance (ang.)\] \[atom1,atom2\] ...\] | "anharmonic\_keep\_pot": \["0.1", "0.5", "1.5", "1,2"\] |
| keep\_angle | \-ka, \--keep\_angle | 3 | keep angle 0.5\*k\*(θ \- θ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep angle (degrees)\] \[atom1,atom2,atom3\] ...\] | "keep\_angle": \["0.1", "109.5", "1,2,3"\] |
| keep\_angle\_v2 | \-kav2, \--keep\_angle\_v2 | 5 | keep angle\_v2 0.5\*k\*(θ \- θ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] ...\] | "keep\_angle\_v2": \["0.1", "120.0", "1,2", "3,4", "5,6"\] |
| universal\_potential | \-up, \--universal\_potential | 2 | Potential to gather specified atoms to a single point (ex.) \[\[potential (kJ/mol)\] \[target atoms (1,2)\] ...\] | "universal\_potential": \["100.0", "1,3,5"\] |
| atom\_distance\_dependent\_keep\_angle | \-ddka, \--atom\_distance\_dependent\_keep\_angle | 7 | atom-distance-dependent keep angle (ex.) \[\[spring const.(a.u.)\] \[minimum keep angle (degrees)\] \[maximum keep angle (degrees)\] \[base distance (ang.)\] \[reference atom (1 atom)\] \[center atom (1 atom)\] \[atom1,atom2,atom3\] ...\] | "atom\_distance\_dependent\_keep\_angle": \["0.1", "90.0", "120.0", "3.0", "1", "2", "2,3,4"\] |
| keep\_dihedral\_angle | \-kda, \--keep\_dihedral\_angle | 3 | keep dihedral angle 0.5\*k\*(φ \- φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep dihedral angle (degrees)\] \[atom1,atom2,atom3,atom4\] ...\] | "keep\_dihedral\_angle": \["0.1", "180.0", "1,2,3,4"\] |
| keep\_out\_of\_plain\_angle | \-kopa, \--keep\_out\_of\_plain\_angle | 3 | keep\_out\_of\_plain\_angle 0.5\*k\*(φ \- φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep out of plain angle (degrees)\] \[atom1,atom2,atom3,atom4\] ...\] | "keep\_out\_of\_plain\_angle": \["0.1", "0.0", "1,2,3,4"\] |
| keep\_dihedral\_angle\_v2 | \-kdav2, \--keep\_dihedral\_angle\_v2 | 6 | keep dihedral angle\_v2 0.5\*k\*(φ \- φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep dihedral angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] \[Fragm.4\] ...\] | "keep\_dihedral\_angle\_v2": \["0.1", "180.0", "1,2", "3,4", "5,6", "7,8"\] |
| keep\_dihedral\_angle\_cos | \-kdac, \--keep\_dihedral\_angle\_cos | 7 | keep dihedral angle\_cos k\*\[1 \+ cos(n \* φ \- (φ0 \+ pi))\] (0 \~ 180 deg.) (ex.) \[\[potential const.(a.u.)\] \[angle const. (unitless)\] \[keep dihedral angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] \[Fragm.4\] ...\] | "keep\_dihedral\_angle\_cos": \["0.1", "2", "180.0", "1,2", "3,4", "5,6", "7,8"\] |
| keep\_out\_of\_plain\_angle\_v2 | \-kopav2, \--keep\_out\_of\_plain\_angle\_v2 | 6 | keep out\_of\_plain angle\_v2 0.5\*k\*(φ \- φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep out\_of\_plain angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] \[Fragm.4\] ...\] | "keep\_out\_of\_plain\_angle\_v2": \["0.1", "0.0", "1,2", "3,4", "5,6", "7,8"\] |
| void\_point\_pot | \-vpp, \--void\_point\_pot | 5 | void point keep potential (ex.) \[\[spring const.(a.u.)\] \[keep distance (ang.)\] \[void\_point (x,y,z) (ang.)\] \[atoms(ex. 1,2,3-5)\] \[order p "(1/p)*k*(r \- r0)^p"\] ...\] | "void\_point\_pot": \["0.5", "1.5", "0,0,0", "1,2", "2.0"\] |
| bond\_range\_potential | \-brp, \--bond\_range\_potential | 6 | Add potential to confine atom distance. (ex.) \[\[upper const.(a.u.)\] \[lower const.(a.u.)\] \[upper distance (ang.)\] \[lower distance (ang.)\] \[Fragm.1\] \[Fragm.2\] ...\] | "bond\_range\_potential": \["0.5", "0.5", "2.0", "1.5", "1,2", "3,4"\] |
| well\_pot | \-wp, \--well\_pot | 4 | Add potential to limit atom distance. (ex.) \[\[wall energy (kJ/mol)\] \[fragm.1\] \[fragm.2\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] ...\] | "well\_pot": \["100", "1,2", "3-5", "1.0,1.5,2.5,3.0"\] |
| wall\_well\_pot | \-wwp, \--wall\_well\_pot | 4 | Add potential to limit atoms movement. (like sandwich) (ex.) \[\[wall energy (kJ/mol)\] \[direction (x,y,z)\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] \[target atoms (1,2,3-5)\] ...\] | "wall\_well\_pot": \["100", "z", "0,1,5,6", "1-10"\] |
| void\_point\_well\_pot | \-vpwp, \--void\_point\_well\_pot | 4 | Add potential to limit atom movement. (like sphere) (ex.) \[\[wall energy (kJ/mol)\] \[coordinate (x,y,z) (ang.)\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] \[target atoms (1,2,3-5)\] ...\] | "void\_point\_well\_pot": \["100", "0,0,0", "0,2,8,10", "1-10"\] |
| around\_well\_pot | \-awp, \--around\_well\_pot | 4 | Add potential to limit atom movement. (like sphere around fragment) (ex.) \[\[wall energy (kJ/mol)\] \[center (1,2-4)\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] \[target atoms (2,3-5)\] ...\] | "around\_well\_pot": \["100", "1,2", "0,2,8,10", "3-5"\] |
| spacer\_model\_potential | \-smp, \--spacer\_model\_potential | 5 | Add potential based on Morse potential to reproduce solvent molecules around molecule. (ex.) \[\[solvent particle well depth (kJ/mol)\] \[solvent particle e.q. distance (ang.)\] \[scaling of cavity (2.0)\] \[number of particles\] \[target atoms (2,3-5)\] ...\] | "spacer\_model\_potential": \["5.0", "3.0", "2.0", "10", "1-5"\] |
| metadynamics | \-metad, \--metadynamics | 4 | apply meta-dynamics (use gaussian potential) (ex.) \[\[\[bond\] \[potential height (kJ/mol)\] \[potential width (ang.)\] \[(atom1),(atom2)\]\] \[\[angle\] \[potential height (kJ/mol)\] \[potential width (deg.)\] \[(atom1),(atom2),(atom3)\]\] ...\] | "metadynamics": \["bond", "10.0", "0.2", "1,2"\] |
| linear\_mechano\_force\_pot | \-lmefp, \--linear\_mechano\_force\_pot | 3 | add linear mechanochemical force (ex.) \[\[force(pN)\] \[atoms1(ex. 1,2)\] \[atoms2(ex. 3,4)\] ...\] | "linear\_mechano\_force\_pot": \["100", "1,2", "3,4"\] |
| linear\_mechano\_force\_pot\_v2 | \-lmefpv2, \--linear\_mechano\_force\_pot\_v2 | 2 | add linear mechanochemical force (ex.) \[\[force(pN)\] \[atom(ex. 1)\] \[direction (xyz)\] ...\] | "linear\_mechano\_force\_pot\_v2": \["100", "1,0,0"\] |
| asymmetric\_ellipsoidal\_repulsive\_potential | \-aerp, \--asymmetric\_ellipsoidal\_repulsive\_potential | 5 | add asymmetric ellipsoidal repulsive potential (use GNB parameters (JCTC, 2024)) (ex.) \[...\] | "asymmetric\_ellipsoidal\_repulsive\_potential": \["10.0", "3.0,3.0,3.0,3.0,3.0,3.0", "3.0", "1,2", "3-5"\] |
| asymmetric\_ellipsoidal\_repulsive\_potential\_v2 | \-aerpv2, \--asymmetric\_ellipsoidal\_repulsive\_potential\_v2 | 5 | add asymmetric ellipsoidal repulsive potential (ex.) \[...\] | "asymmetric\_ellipsoidal\_repulsive\_potential\_v2": \["10.0", "3.0,3.0,3.0,3.0,3.0,3.0", "3.0", "1,2", "3-5"\] |
| nano\_reactor\_potential | \-nrp, \--nano\_reactor\_potential | 6 | add nano reactor potential (ex.) \[\[inner wall (ang.)\] \[outer wall (ang.)\] \[contraction time (ps)\] \[expansion time (ps)\] \[contraction force const (kcal/mol/A^2)\] \[expansion force const (kcal/mol/A^2)\]\] (Recommendation: 8.0 14.0 1.5 0.5 1.0 0.5) | "nano\_reactor\_potential": \["8.0", "14.0", "1.5", "0.5", "1.0", "0.5"\] |

## **4\. i-EIP Settings (ieipparser)**

Used by: ieipmain.py

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | \-bs, \--basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| functional | \-func, \--functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| NSTEP | \-ns, \--NSTEP | iter. number | int | 999 | "NSTEP": 500 |
| opt\_method | \-opt, \--opt\_method | optimization method for QM calclation (default: FIRE) ... | list\[string\] | \["FIRELARS"\] | "opt\_method": \["RFO\_BFGS"\] |
| sub\_basisset | \-sub\_bs, \--sub\_basisset | sub\_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub\_basisset": \["I", "LanL2DZ"\] |
| effective\_core\_potential | \-ecp, \--effective\_core\_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis\_set name)". | list\[string\] | \[\] | "effective\_core\_potential": \["I", "LanL2DZ"\] |
| gradient\_fix\_atoms | \-gfix, \--gradient\_fix\_atoms | set the gradient of internal coordinates between atoms to zero (ex.) \[\[atoms (ex.) 1,2\] ...\] | list\[string\] | "" | "gradient\_fix\_atoms": \["1,2"\] |
| N\_THREAD | \-core, \--N\_THREAD | threads | int | 8 | "N\_THREAD": 16 |
| microiter | \-mi, \--microiter | microiteration for relaxing reaction pathways | int | 0 | "microiter": 5 |
| BETA | \-beta, \--BETA | force for optimization | float | 1.0 | "BETA": 0.5 |
| SET\_MEMORY | \-mem, \--SET\_MEMORY | use mem(ex. 1GB) | string | "2GB" | "SET\_MEMORY": "8GB" |
| excited\_state | \-es, \--excited\_state | calculate excited state (default: \[0(initial state), 0(final state)\]) (e.g.) if you set spin\_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | list\[int\] | \[0, 0\] | "excited\_state": \[0, 1\] |
| model\_function\_mode | \-mf, \--model\_function\_mode | use model function to optimization (seam, avoiding, conical, mesx, meci) | string | "None" | "model\_function\_mode": "seam" |
| calc\_exact\_hess | \-fc, \--calc\_exact\_hess | calculate exact hessian per steps (ex.) \[steps per one hess calculation\] | int | \-1 | "calc\_exact\_hess": 1 |
| usextb | \-xtb, \--usextb | use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| usedxtb | \-dxtb, \--usedxtb | use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usedxtb": "GFN1-xTB" |
| sqm1 | \-sqm1, \--sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | \-sqm2, \--sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| pyscf | \-pyscf, \--pyscf | use pyscf module. | boolean | false | "pyscf": true |
| unrestrict | \-u, \--unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| electronic\_charge | \-elec, \--electronic\_charge | formal electronic charge (ex.) \[charge (0)\] | list\[int\] | \[0, 0\] | "electronic\_charge": \[0, 0\] |
| spin\_multiplicity | \-spin, \--spin\_multiplicity | spin multiplcity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha \- Nbeta)) (ex.) \[multiplcity (0)\] | list\[int\] | \[1, 1\] | "spin\_multiplicity": \[1, 1\] |
| cpcm\_solv\_model | \-cpcm, \--cpcm\_solv\_model | use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "cpcm\_solv\_model": "water" |
| alpb\_solv\_model | \-alpb, \--alpb\_solv\_model | use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "alpb\_solv\_model": "toluene" |
| dft\_grid | \-grid, \--dft\_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft\_grid": 4 |
| othersoft | \-os, \--othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace\_mp etc. | string | "None" | "othersoft": "orca" |
| software\_path\_file | \-osp, \--software\_path\_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software\_path.conf | string | "./software\_path.conf" | "software\_path\_file": "/path/to/my.conf" |
| use\_gnt | \-gnt, \--use\_gnt | Use GNT (Growing Newton Trajectory) | boolean | false | "use\_gnt": true |
| gnt\_vec | \-gnt\_vec, \--gnt\_vec | set vector to calculate Newton trajectory (ex. 1,2,3 (default:calculate vector reactant to product) ) | string | None | "gnt\_vec": "1,2,3" |
| gnt\_step\_len | \-gnt\_step, \--gnt\_step\_len | set step length for Newton trajectory (default: 0.5) | float | 0.5 | "gnt\_step\_len": 0.25 |
| gnt\_microiter | \-gnt\_mi, \--gnt\_microiter | max number of micro-iteration for Newton trajectory (default: 25\) | int | 25 | "gnt\_microiter": 10 |
| use\_addf | \-addf, \--use\_addf | Use ADDF-like method (default: False) | boolean | false | "use\_addf": true |
| addf\_step\_size | \-addf\_step, \--addf\_step\_size | set step size for ADDF-like method (default: 0.1) | float | 0.1 | "addf\_step\_size": 0.2 |
| addf\_step\_num | \-addf\_num, \--addf\_step\_num | set number of steps for ADDF-like method (default: 300\) | int | 300 | "addf\_step\_num": 100 |
| number\_of\_add | \-addf\_nadd, \--number\_of\_add | set number of number of searching ADD (A larger ADD takes precedence.) for ADDF-like method (default: 5\) | int | 5 | "number\_of\_add": 3 |
| use\_2pshs | \-2pshs, \--use\_2pshs | Use 2PSHS-like method (default: False) | boolean | false | "use\_2pshs": true |
| twoPshs\_step\_size | \-2pshs\_step, \--twoPshs\_step\_size | set step size for 2PSHS-like method (default: 0.05) | float | 0.05 | "twoPshs\_step\_size": 0.1 |
| twoPshs\_step\_num | \-2pshs\_num, \--twoPshs\_step\_num | set number of steps for 2PSHS-like method (default: 300\) | int | 300 | "twoPshs\_step\_num": 100 |
| use\_dimer | \-use\_dimer, \--use\_dimer | Use Dimer method for searching direction of TS (default: False) | boolean | false | "use\_dimer": true |
| dimer\_separation | \-dimer\_sep, \--dimer\_separation | set dimer separation (default: 0.0001) | float | 0.0001 | "dimer\_separation": 0.0005 |
| dimer\_trial\_angle | \-dimer\_trial\_angle, \--dimer\_trial\_angle | set dimer trial angle (default: pi/32) | float | 0.09817... | "dimer\_trial\_angle": 0.1 |
| dimer\_max\_iterations | \-dimer\_maxiter, \--dimer\_max\_iterations | set max iterations for dimer method (default: 1000\) | int | 1000 | "dimer\_max\_iterations": 500 |

## **5\. Molecular Dynamics Settings (mdparser)**

Used by: mdmain.py

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | \-bs, \--basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| functional | \-func, \--functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| sub\_basisset | \-sub\_bs, \--sub\_basisset | sub\_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub\_basisset": \["I", "LanL2DZ"\] |
| effective\_core\_potential | \-ecp, \--effective\_core\_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis\_set name)". | list\[string\] | \[\] | "effective\_core\_potential": \["I", "LanL2DZ"\] |
| excited\_state | \-es, \--excited\_state | calculate excited state (default: 0\) (e.g.) if you set spin\_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | int | 0 | "excited\_state": 1 |
| additional\_inputs | \-addint, \--additional\_inputs | (ex.) \[(excited state) (fromal charge) (spin multiplicity) ...\] | list\[int\] | \[\] | "additional\_inputs": \[1, \-1, 2\] |
| NSTEP | \-time, \--NSTEP | time scale | int | 100000 | "NSTEP": 50000 |
| TRAJECTORY | \-traj, \--TRAJECTORY | number of trajectory to generate (default) 1 | int | 1 | "TRAJECTORY": 10 |
| temperature | \-temp, \--temperature | temperature \[unit. K\] (default) 298.15 K | float | 298.15 | "temperature": 300.0 |
| timestep | \-ts, \--timestep | time step \[unit. atom unit\] (default) 0.1 a.u. | float | 0.1 | "timestep": 0.5 |
| pressure | \-press, \--pressure | pressure \[unit. kPa\] (default) 1013 kPa | float | 101.3 | "pressure": 101.3 |
| N\_THREAD | \-core, \--N\_THREAD | threads | int | 8 | "N\_THREAD": 16 |
| SET\_MEMORY | \-mem, \--SET\_MEMORY | use mem(ex. 1GB) | string | "1GB" | "SET\_MEMORY": "4GB" |
| unrestrict | \-u, \--unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| mdtype | \-mt, \--mdtype | specify condition to do MD (ex.) velocityverlet (default) nosehoover | string | "nosehoover" | "mdtype": "velocityverlet" |
| fix\_atoms | \-fix, \--fix\_atoms | fix atoms (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \[\] | "fix\_atoms": \["1,2,3-6"\] |
| geom\_info | \-gi, \--geom\_info | calculate atom distances, angles, and dihedral angles in every iteration (energy\_profile is also saved.) (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \["1"\] | "geom\_info": \["1,2", "1,2,3"\] |
| pyscf | \-pyscf, \--pyscf | use pyscf module. | boolean | false | "pyscf": true |
| electronic\_charge | \-elec, \--electronic\_charge | formal electronic charge (ex.) \[charge (0)\] | int | 0 | "electronic\_charge": \-1 |
| spin\_multiplicity | \-spin, \--spin\_multiplicity | spin multiplcity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha \- Nbeta)) (ex.) \[multiplcity (0)\] | int | 1 | "spin\_multiplicity": 2 |
| saddle\_order | \-order, \--saddle\_order | optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) \[order (0)\] | int | 0 | "saddle\_order": 0 |
| cmds | \-cmds, \--cmds | apply classical multidimensional scaling to calculated approx. reaction path. | boolean | false | "cmds": true |
| usextb | \-xtb, \--usextb | use extended tight bonding method to calculate. default is GFN2-xTB (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| sqm1 | \-sqm1, \--sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | \-sqm2, \--sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| change\_temperature | \-ct, \--change\_temperature | change temperature of thermostat (defalut) No change (ex.) \[1000(time), 500(K) 5000(time), 1000(K)...\] | list\[string\] | \[\] | "change\_temperature": \["1000", "500", "5000", "1000"\] |
| constraint\_condition | \-cc, \--constraint\_condition | apply constraint conditions for optimazation (ex.) \[\[(dinstance (ang.)), (atom1),(atom2)\] \[(bond\_angle (deg.)), (atom1),(atom2),(atom3)\] \[(dihedral\_angle (deg.)), (atom1),(atom2),(atom3),(atom4)\] ...\] | list\[string\] | \[\] | "constraint\_condition": \["1.5,1,2", "109.5,1,2,3"\] |
| othersoft | \-os, \--othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace\_mp etc. | string | "None" | "othersoft": "orca" |
| software\_path\_file | \-osp, \--software\_path\_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software\_path.conf | string | "./software\_path.conf" | "software\_path\_file": "/path/to/my.conf" |
| periodic\_boundary\_condition | \-pbc, \--periodic\_boundary\_condition | apply periodic boundary condition (Default is not applying.) (ex.) \[periodic boundary (x,y,z) (ang.)\] | list\[string\] | \[\] | "periodic\_boundary\_condition": \["10.0,10.0,10.0"\] |
| projection\_constrain | \-pc, \--projection\_constrain | apply constrain conditions with projection of gradient and hessian (ex.) \[\[(constraint condition name) (atoms(ex. 1,2))\] ...\] | list\[string\] | \[\] | "projection\_constrain": \["bond", "1,2"\] |
| cpcm\_solv\_model | \-cpcm, \--cpcm\_solv\_model | use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "cpcm\_solv\_model": "water" |
| alpb\_solv\_model | \-alpb, \--alpb\_solv\_model | use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "alpb\_solv\_model": "toluene" |
| pca | \-pca, \--pca | Apply principal component analysis to calculated approx. reaction path. | boolean | false | "pca": true |
| dft\_grid | \-grid, \--dft\_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft\_grid": 4 |

