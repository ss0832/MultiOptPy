# **MultiOptPy Configuration Reference**

(ver. 1.19.6)

This document outlines all available settings for MultiOptPy, based on the definitions in interface.py. These settings can be configured either via a JSON file (like config.json used by run_autots.py) or as command-line arguments (used by optmain.py, nebmain.py, etc.).

## **Basic Rules for JSON Configuration**

1. **Key-Value Mapping**: The JSON key (e.g., "NSTEP") directly corresponds to the command-line argument's destination (e.g., --NSTEP).  
2. **Omitted Keys Use Defaults**: If a key is **omitted** from the JSON file, the program will automatically use the Default value listed in this guide.  
3. **Data Types**:  
   * Use standard JSON types: string, number (for integers and floats), and boolean (true/false).  
   * Arguments that accept multiple values (like nargs="\*") should be specified as a JSON array (e.g., "opt_method": \["FIRELARS"\]).  
   * Bias/Constraint settings (Section 3\) are parsed from strings, so they must be provided as a list\[string\] (e.g., "manual_AFIR": \["300", "4", "17"\]).  
4. **nargs='?' Arguments**: For arguments like use_model_hessian or conjugate_gradient:  
   * To **enable** the feature with its specific value, provide the string (e.g., "use_model_hessian": "fischerd3").  
   * To **disable** it (and use the default), specify null (e.g., "use_model_hessian": null).

## **1\. Optimization Settings (optimizeparser)**

Used by: optmain.py, run_autots.py (Steps 1, 3, & 4\)

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | -bs, --basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G*" |
| functional | -func, --functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| sub_basisset | -sub_bs, --sub_basisset | sub_basisset (ex. I LanL2DZ) | list[string] | [] | "sub_basisset": ["I", "LanL2DZ"] |
| effective_core_potential | -ecp, --effective_core_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". | list[string] | [] | "effective_core_potential": ["I", "LanL2DZ"] |
| excited_state | -es, --excited_state | calculate excited state (default: 0) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | int | 0 | "excited_state": 1 |
| NSTEP | -ns, --NSTEP | number of iteration (default: 1000) | int | 1000 | "NSTEP": 500 |
| N_THREAD | -core, --N_THREAD | threads | int | 8 | "N_THREAD": 16 |
| SET_MEMORY | -mem, --SET_MEMORY | use mem(ex. 1GB) | string | "2GB" | "SET_MEMORY": "8GB" |
| DELTA | -d, --DELTA | move step | string | "x" | "DELTA": "y" |
| max_trust_radius | -tr, --max_trust_radius | max trust radius to restrict step size (unit: ang.) (default: 0.1 for n-th order saddle point optimization, 0.5 for minimum point optimization) (notice: default minimum trust radius is 0.01) | float | None | "max_trust_radius": 0.2 |
| min_trust_radius | -mintr, --min_trust_radius | min trust radius to restrict step size (unit: ang.) (default: 0.01) | float | 0.01 | "min_trust_radius": 0.005 |
| unrestrict | -u, --unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| fix_atoms | -fix, --fix_atoms | fix atoms (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \[\] | "fix_atoms": \["1,2,3-6"\] |
| geom_info | -gi, --geom_info | calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \["1"\] | "geom_info": \["1,2", "1,2,3"\] |
| dissociate_check | -dc, --dissociate_check | Terminate calculation if distance between two fragments is exceed this value. (default) 100 \[ang.\] | string | "100" | "dissociate_check": "50.0" |
| opt_method | -opt, --opt_method | optimization method for QM calculation (default: FIRE) (mehod_list:(steepest descent method group) FIRE, CG etc. (quasi-Newton method group) rsirfo_fsb rsirfo_bofill etc.) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. \[\[steepest descent\] \[quasi-Newton method\]\]) (ex.) \[opt_method\] | list\[string\] | \["FIRELARS"\] | "opt_method": \["rsirfo_block_fsb"\] |
| calc_exact_hess | -fc, --calc_exact_hess | calculate exact hessian per geometry optimization steps and IRC steps (ex.) \[steps per one hess calculation\] | int | -1 | "calc_exact_hess": 5 |
| calc_model_hess | -mfc, --calc_model_hess | calculate model hessian per steps (ex.) \[steps per one hess calculation\] | int | 50 | "calc_model_hess": 25 |
| usextb | -xtb, --usextb | use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| usedxtb | -dxtb, --usedxtb | use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usedxtb": "GFN1-xTB" |
| sqm1 | -sqm1, --sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | -sqm2, --sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| cpcm_solv_model | -cpcm, --cpcm_solv_model | use CPCM solvent model for xTB (Default setting is not using this model.) (ex.) water | string | None | "cpcm_solv_model": "water" |
| alpb_solv_model | -alpb, --alpb_solv_model | use ALPB solvent model for xTB (Default setting is not using this model.) (ex.) water | string | None | "alpb_solv_model": "toluene" |
| pyscf | -pyscf, --pyscf | use pyscf module. | boolean | false | "pyscf": true |
| electronic_charge | -elec, --electronic_charge | formal electronic charge (ex.) \[charge (0)\] | int | 0 | "electronic_charge": -1 |
| spin_multiplicity | -spin, --spin_multiplicity | spin multiplicity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha - Nbeta)) (ex.) \[multiplicity (0)\] | int | 1 | "spin_multiplicity": 2 |
| saddle_order | -order, --saddle_order | optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) \[order (0)\] | int | 0 | "saddle_order": 1 |
| cmds | -cmds, --cmds | Apply classical multidimensional scaling to calculated approx. reaction path. | boolean | false | "cmds": true |
| pca | -pca, --pca | Apply principal component analysis to calculated approx. reaction path. | boolean | false | "pca": true |
| koopman | -km, --koopman | Apply Koopman model to analyze the convergence | boolean | false | "koopman": true |
| intrinsic_reaction_coordinates | -irc, --intrinsic_reaction_coordinates | Calculate intrinsic reaction coordinates. (ex.) \[\[step_size\], \[max_step\], \[IRC_method\]\] (Recommended) \[0.5 300 lqa\] | list\[string\] | \[\] | "intrinsic_reaction_coordinates": \["0.5", "200", "lqa"\] |
| opt_fragment | -of, --opt_fragment | Several atoms are grouped together as fragments and optimized. (This method does not work if you use quasi-newton method for optimazation.) (ex.) \[\[atoms (ex.) 1-4\] ...\] | list\[string\] | \[\] | "opt_fragment": \["1-4", "5-8"\] |
| dft_grid | -grid, --dft_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft_grid": 4 |
| othersoft | -os, --othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc. | string | "None" | "othersoft": "orca" |
| software_path_file | -osp, --software_path_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf | string | "./software_path.conf" | "software_path_file": "/path/to/my.conf" |
| tight_convergence_criteria | -tcc, --tight_convergence_criteria | apply tight opt criteria. | boolean | false | "tight_convergence_criteria": true |
| loose_convergence_criteria | -lcc, --loose_convergence_criteria | apply loose opt criteria. | boolean | false | "loose_convergence_criteria": true |
| use_model_hessian | -modelhess, --use_model_hessian | use model hessian. (Default: not using model hessian If you specify only option, Improved Lindh \+ Grimme's D3 dispersion model hessian is used.) (ex. lindh, gfnff, gfn0xtb, fischer, fischerd3, fischerd4, schlegel, swart, lindh2007, lindh2007d3, lindh2007d4) | string | null | "use_model_hessian": "fischerd3" |
| shape_conditions | -sc, --shape_conditions | Exit optimization if these conditions are not satisfied. (e.g.) \[\[(ang.) gt(lt) 2,3 (bond)\] \[(deg.) gt(lt) 2,3,4 (bend)\] ...\] \[\[(deg.) gt(lt) 2,3,4,5 (torsion)\] ...\] | list\[string\] | \[\] | "shape_conditions": \["gt", "2,3", "lt", "2,3,4"\] |
| projection_constrain | -pc, --projection_constrain | apply constrain conditions with projection of gradient and hessian (ex.) \[\[(constraint condition name) (atoms(ex. 1,2))\] ...\] | list\[string\] | \[\] | "projection_constrain": \["bond", "1,2"\] |
| oniom_flag | -oniom, --oniom_flag | apply ONIOM method (low layer: GFN1-xTB) (ex.) \[(atom_number of high layer (ex. 1,2))\] (caution) -pc option is not available. If there are not link atoms, please input "none" | list\[string\] | \[\] | "oniom_flag": \["1,2,5-10", "none", "GFN1-xTB"\] |
| frequency_analysis | -freq, --frequency_analysis | Perform normal vibrational analysis after converging geometry optimization. (Caution: Unable to use this analysis with oniom method) | boolean | false | "frequency_analysis": true |
| temperature | -temp, --temperature | temperatrue to calculate thermochemistry (Unit: K) (default: 298.15K) | float | 298.15 | "temperature": 300.0 |
| pressure | -press, --pressure | pressure to calculate thermochemistry (Unit: Pa) (default: 101325Pa) | float | 101325 | "pressure": 100000 |
| detect_negative_eigenvalues | -negeigval, --detect_negative_eigenvalues | Detect negative eigenvalues in the Hessian matrix at ITR. 0 if you caluculate exact hessian (-fc >0). If negative eigenvalues are not detected and saddle_order > 0, the optimization is stopped. | boolean | false | "detect_negative_eigenvalues": false |

## **2\. NEB Settings (nebparser)**

Used by: nebmain.py, run_autots.py (Step 2\)

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | -bs, --basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| sub_basisset | -sub_bs, --sub_basisset | sub_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub_basisset": \["I", "LanL2DZ"\] |
| effective_core_potential | -ecp, --effective_core_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". | list\[string\] | \[\] | "effective_core_potential": \["I", "LanL2DZ"\] |
| functional | -func, --functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| unrestrict | -u, --unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| excited_state | -es, --excited_state | calculate excited state (default: 0\) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | int | 0 | "excited_state": 1 |
| NSTEP | -ns, --NSTEP | iter. number | int | 10 | "NSTEP": 50 |
| OM | -om, --OM | J. Chem. Phys. 155, 074103 (2021) doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action. | boolean | false | "OM": true |
| LUP | -lup, --LUP | J. Chem. Phys. 92, 1510–1511 (1990) doi:https://doi.org/10.1063/1.458112 locally updated planes (LUP) method | boolean | false | "LUP": true |
| BNEB | -bneb, --BNEB | NEB using Wilson's B matrix for calculating the perpendicular force. | boolean | false | "BNEB": true |
| BNEB2 | -bneb2, --BNEB2 | NEB using Wilson's B matrix for calculating the perpendicular force with parallel spring force. | boolean | false | "BNEB2": true |
| DNEB | -dneb, --DNEB | J. Chem. Phys. 120, 2082–2094 (2004) doi:https://doi.org/10.1063/1.1636455 doubly NEB method (DNEB) method | boolean | false | "DNEB": true |
| NESB | -nesb, --NESB | J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169 Nudged elastic stiffness band (NESB) method | boolean | false | "NESB": true |
| DMF | -dmf, --DMF | Direct Max Flux (DMF) method | boolean | false | "DMF": true |
| EWBNEB | -ewbneb, --EWBNEB | Energy-weighted Nudged elastic band method | boolean | false | "EWBNEB": true |
| QSM | -qsm, --QSM | Quadratic String Method (J. Chem. Phys. 124, 054109 (2006)) | boolean | false | "QSM": true |
| ANEB | -aneb, --ANEB | Adaptic NEB method (ref.: J. Chem. Phys. 117, 4651 (2002)) (Usage: -aneb \[interpolation_num (ex. 2)\] \[frequency (ex. 5)\], Default setting is not applying adaptic NEB method.) | list\[string\] | None | "ANEB": \["3", "5"\] |
| use_image_dependent_pair_potential | -idpp, --use_image_dependent_pair_potential | use image dependent pair potential (IDPP) method (ref. arXiv:1406.1512v1) | boolean | false | "use_image_dependent_pair_potential": true |
| use_correlated_flat_bottom_elastic_network_model | -cfbenm, --use_correlated_flat_bottom_elastic_network_model | use correlated flat-bottom elastic network model (CFBENM) method (ref. s: J.Chem.TheoryComput.2025,21,3513−3522) | boolean | false | "use_correlated_flat_bottom_elastic_network_model": true |
| align_distances | -ad, --align_distances | distribute images at equal intervals on the reaction coordinate | int | 0 | "align_distances": 9999 |
| align_distances_spline | -ads, --align_distances_spline | distribute images at equal intervals on the reaction coordinate using spline interpolation | int | 0 | "align_distances_spline": 5 |
| align_distances_spline_ver2 | -ads2, --align_distances_spline_ver2 | distribute images at equal intervals on the reaction coordinate using spline interpolation ver.2 | int | 0 | "align_distances_spline_ver2": 5 |
| align_distances_geodesic | -adg, --align_distances_geodesic | distribute images at equal intervals on the reaction coordinate using geodesic interpolation | int | 0 | "align_distances_geodesic": 5 |
| align_distances_bernstein | -adb, --align_distances_bernstein | distribute images at equal intervals on the reaction coordinate using Bernstein interpolation | int | 0 | "align_distances_bernstein": 5 |
| align_distances_savgol | -adsg, --align_distances_savgol | distribute images at equal intervals on the reaction coordinate using Savitzky-Golay interpolation (ex.) \[\[iteration(int)\],\[window_size(int, 5 is recommended)\],\[poly_order(int) 3 is recommended\]\] (default: 0,0,0 (not using Savitzky-Golay interpolation)) | string | "0,0,0" | "align_distances_savgol": "5,5,3" |
| node_distance | -nd, --node_distance | distribute images at equal intervals linearly based ont specific distance (ex.) \[distance (ang.)\] (default: None) | float | None | "node_distance": 0.8 |
| node_distance_spline | -nds, --node_distance_spline | distribute images at equal intervals using spline interpolation based ont specific distance (ex.) \[distance (ang.)\] (default: None) | float | None | "node_distance_spline": 0.8 |
| node_distance_bernstein | -ndb, --node_distance_bernstein | distribute images at equal intervals using Bernstein interpolation based ont specific distance (ex.) \[distance (ang.)\] (default: None) | float | None | "node_distance_bernstein": 0.8 |
| node_distance_savgol | -ndsg, --node_distance_savgol | distribute images at equal intervals using Savitzky-Golay interpolation based ont specific distance (ex.) \[\[distance (ang.)\],\[window_size(int, 5 is recommended)\],\[poly_order(int) 3 is recommended\]\] (default: None) | string | None | "node_distance_savgol": "0.8,5,3" |
| partition | -p, --partition | number of nodes | int | 0 | "partition": 10 |
| N_THREAD | -core, --N_THREAD | threads | int | 8 | "N_THREAD": 16 |
| SET_MEMORY | -mem, --SET_MEMORY | use mem(ex. 1GB) | string | "1GB" | "SET_MEMORY": "4GB" |
| apply_CI_NEB | -cineb, --apply_CI_NEB | apply CI_NEB method | int | 99999 | "apply_CI_NEB": 10 |
| steepest_descent | -sd, --steepest_descent | apply steepest_descent method | int | 99999 | "steepest_descent": 5 |
| conjugate_gradient | -cg, --conjugate_gradient | apply conjugate_gradient method for path optimization (Available update method of CG parameters :FR, PR, HS, DY, HZ), default update method is HS.) | string | false | "conjugate_gradient": "HS" |
| memory_limited_BFGS | -lbfgs, --memory_limited_BFGS | apply L-BFGS method for path optimization | boolean | false | "memory_limited_BFGS": true |
| not_ts_optimization | -notsopt, --not_ts_optimization | not apply TS optimization during NEB calculation | boolean | false | "not_ts_optimization": true |
| calc_exact_hess | -fc, --calc_exact_hess | calculate exact hessian per steps (ex.) \[steps per one hess calculation\] | int | -1 | "calc_exact_hess": 1 |
| global_quasi_newton | -gqnt, --global_quasi_newton | use global quasi-Newton method | boolean | false | "global_quasi_newton": true |
| usextb | -xtb, --usextb | use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| usedxtb | -dxtb, --usedxtb | use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usedxtb": "GFN1-xTB" |
| sqm1 | -sqm1, --sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | -sqm2, --sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| pyscf | -pyscf, --pyscf | use pyscf module. | boolean | false | "pyscf": true |
| fixedges | -fe, --fixedges | fix edges of nodes (1=initial_node, 2=end_node, 3=both_nodes) | int | 0 | "fixedges": 3 |
| fix_atoms | -fix, --fix_atoms | fix atoms (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \[\] | "fix_atoms": \["1,2,3-6"\] |
| projection_constrain | -pc, --projection_constrain | apply constrain conditions with projection of gradient and hessian (ex.) \[\[(constraint condition name) (atoms(ex. 1,2))\] ...\] | list\[string\] | \[\] | "projection_constrain": \["bond", "1,2"\] |
| cpcm_solv_model | -cpcm, --cpcm_solv_model | use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "cpcm_solv_model": "water" |
| alpb_solv_model | -alpb, --alpb_solv_model | use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "alpb_solv_model": "toluene" |
| save_pict | -spng, --save_pict | Save picture for visualization. | boolean | false | "save_pict": true |
| apply_convergence_criteria | -aconv, --apply_convergence_criteria | Apply convergence criteria for NEB calculation. | boolean | false | "apply_convergence_criteria": true |
| climbing_image | -ci, --climbing_image | climbing image for NEB calculation. (start of ITR., interval) The default setting is not using climbing image. | list\[int\] | \[999999, 999999\] | "climbing_image": \[10, 1\] |
| electronic_charge | -elec, --electronic_charge | formal electronic charge (ex.) \[charge (0)\] | int | 0 | "electronic_charge": -1 |
| spin_multiplicity | -spin, --spin_multiplicity | spin multiplcity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha - Nbeta)) (ex.) \[multiplcity (0)\] | int | 1 | "spin_multiplicity": 2 |
| dft_grid | -grid, --dft_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft_grid": 4 |
| use_model_hessian | -modelhess, --use_model_hessian | use model hessian. (Default: not using model hessian If you specify only option, Fischer \+ Grimme's D3 dispersion model hessian is used.) (ex. lindh, gfnff, gfn0xtb, fischer, fischerd3, fischerd4, schlegel, swart, lindh2007, lindh2007d3, lindh2007d4) | string | null | "use_model_hessian": "fischerd3" |
| calc_model_hess | -mfc, --calc_model_hess | calculate model hessian per steps (ex.) \[steps per one hess calculation\] | int | 50 | "calc_model_hess": 25 |
| othersoft | -os, --othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc. | string | "None" | "othersoft": "orca" |
| software_path_file | -osp, --software_path_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf | string | "./software_path.conf" | "software_path_file": "/path/to/my.conf" |
| ratio_of_rfo_step | -rrs, --ratio_of_rfo_step | ratio of rfo step (default: 0.5). This option is for optimizer using Hessian (-fc or -modelhess). | float | 0.5 | "ratio_of_rfo_step": 0.25 |

## **3\. Common Bias/Constraint Settings (parser_for_biasforce)**

Used by all parsers (optimizeparser, nebparser, ieipparser, mdparser).  
Note: All arguments in this section are parsed from strings. In JSON, they must be provided as a list\[string\], where each set of N elements forms one definition.

| JSON Key | CMD Argument | Elements per Set | Description | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- |
| manual_AFIR | -ma, --manual_AFIR | 3 | manual-AFIR (ex.) \[\[Gamma(kJ/mol)\] \[Fragm.1(ex. 1,2,3-5)\] \[Fragm.2\] ...\] | "manual_AFIR": \["300", "1,2", "3-5"\] |
| repulsive_potential | -rp, --repulsive_potential | 5 | Add LJ repulsive_potential based on UFF (ex.) \[\[well_scale\] \[dist_scale\] \[Fragm.1(ex. 1,2,3-5)\] \[Fragm.2\] \[scale or value(kJ/mol ang.)\] ...\] | "repulsive_potential": \["1.0", "1.0", "1,2", "3-5", "scale"\] |
| repulsive_potential_v2 | -rpv2, --repulsive_potential_v2 | 10 | Add LJ repulsive_potential based on UFF (ver.2) (eq. V \= ε\[A \* (σ/r)^(rep) - B \* (σ/r)^(attr)\]) (ex.) \[...\] | "repulsive_potential_v2": \["1.0", "1.0", "3.0", "1.0", "0.0", "12.0", "6.0", "1,2", "3-5", "scale"\] |
| repulsive_potential_gaussian | -rpg, --repulsive_potential_gaussian | 7 | Add LJ repulsive_potential based on UFF (ver.2) (eq. V \= ε_LJ\[(σ/r)^(12) - 2 \* (σ/r)^(6)\] - ε_gau \* exp(-((r-σ_gau)/b)^2)) (ex.) \[...\] | "repulsive_potential_gaussian": \["10.0", "3.0", "20.0", "3.0", "0.5", "1,2", "3-5"\] |
| cone_potential | -cp, --cone_potential | 6 | Add cone type LJ repulsive_potential based on UFF (ex.) \[\[well_value (epsilon) (kJ/mol)\] \[dist (sigma) (ang.)\] \[cone angle (deg.)\] \[LJ center atom (1)\] \[three atoms (2,3,4) \] \[target atoms (5-9)\] ...\] | "cone_potential": \["10.0", "3.0", "90.0", "1", "2,3,4", "5-9"\] |
| flux_potential | -fp, --flux_potential | 4 | Add potential to make flow. ( k/p\*(x-x_0)^p )(ex.) \[\[x,y,z (constant (a.u.))\] \[x,y,z (order)\] \[x,y,z coordinate (ang.)\] \[Fragm.(ex. 1,2,3-5)\] ...\] | "flux_potential": \["1,0,0", "2,0,0", "0,0,0", "1,2"\] |
| keep_pot | -kp, --keep_pot | 3 | keep potential 0.5\*k\*(r - r0)^2 (ex.) \[\[spring const.(a.u.)\] \[keep distance (ang.)\] \[atom1,atom2\] ...\] | "keep_pot": \["0.5", "1.5", "1,2"\] |
| keep_pot_v2 | -kpv2, --keep_pot_v2 | 4 | keep potential_v2 0.5\*k\*(r - r0)^2 (ex.) \[\[spring const.(a.u.)\] \[keep distance (ang.)\] \[Fragm.1\] \[Fragm.2\] ...\] | "keep_pot_v2": \["0.5", "3.0", "1-3", "4-6"\] |
| anharmonic_keep_pot | -akp, --anharmonic_keep_pot | 4 | Morse potential De\*\[1-exp(-((k/2\*De)^0.5)\*(r - r0))\]^2 (ex.) \[\[potential well depth (a.u.)\] \[spring const.(a.u.)\] \[keep distance (ang.)\] \[atom1,atom2\] ...\] | "anharmonic_keep_pot": \["0.1", "0.5", "1.5", "1,2"\] |
| keep_angle | -ka, --keep_angle | 3 | keep angle 0.5\*k\*(θ - θ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep angle (degrees)\] \[atom1,atom2,atom3\] ...\] | "keep_angle": \["0.1", "109.5", "1,2,3"\] |
| keep_angle_v2 | -kav2, --keep_angle_v2 | 5 | keep angle_v2 0.5\*k\*(θ - θ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] ...\] | "keep_angle_v2": \["0.1", "120.0", "1,2", "3,4", "5,6"\] |
| universal_potential | -up, --universal_potential | 2 | Potential to gather specified atoms to a single point (ex.) \[\[potential (kJ/mol)\] \[target atoms (1,2)\] ...\] | "universal_potential": \["100.0", "1,3,5"\] |
| atom_distance_dependent_keep_angle | -ddka, --atom_distance_dependent_keep_angle | 7 | atom-distance-dependent keep angle (ex.) \[\[spring const.(a.u.)\] \[minimum keep angle (degrees)\] \[maximum keep angle (degrees)\] \[base distance (ang.)\] \[reference atom (1 atom)\] \[center atom (1 atom)\] \[atom1,atom2,atom3\] ...\] | "atom_distance_dependent_keep_angle": \["0.1", "90.0", "120.0", "3.0", "1", "2", "2,3,4"\] |
| keep_dihedral_angle | -kda, --keep_dihedral_angle | 3 | keep dihedral angle 0.5\*k\*(φ - φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep dihedral angle (degrees)\] \[atom1,atom2,atom3,atom4\] ...\] | "keep_dihedral_angle": \["0.1", "180.0", "1,2,3,4"\] |
| keep_out_of_plain_angle | -kopa, --keep_out_of_plain_angle | 3 | keep_out_of_plain_angle 0.5\*k\*(φ - φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep out of plain angle (degrees)\] \[atom1,atom2,atom3,atom4\] ...\] | "keep_out_of_plain_angle": \["0.1", "0.0", "1,2,3,4"\] |
| keep_dihedral_angle_v2 | -kdav2, --keep_dihedral_angle_v2 | 6 | keep dihedral angle_v2 0.5\*k\*(φ - φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep dihedral angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] \[Fragm.4\] ...\] | "keep_dihedral_angle_v2": \["0.1", "180.0", "1,2", "3,4", "5,6", "7,8"\] |
| keep_dihedral_angle_cos | -kdac, --keep_dihedral_angle_cos | 7 | keep dihedral angle_cos k\*\[1 \+ cos(n \* φ - (φ0 \+ pi))\] (0 \~ 180 deg.) (ex.) \[\[potential const.(a.u.)\] \[angle const. (unitless)\] \[keep dihedral angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] \[Fragm.4\] ...\] | "keep_dihedral_angle_cos": \["0.1", "2", "180.0", "1,2", "3,4", "5,6", "7,8"\] |
| keep_out_of_plain_angle_v2 | -kopav2, --keep_out_of_plain_angle_v2 | 6 | keep out_of_plain angle_v2 0.5\*k\*(φ - φ0)^2 (0 \~ 180 deg.) (ex.) \[\[spring const.(a.u.)\] \[keep out_of_plain angle (degrees)\] \[Fragm.1\] \[Fragm.2\] \[Fragm.3\] \[Fragm.4\] ...\] | "keep_out_of_plain_angle_v2": \["0.1", "0.0", "1,2", "3,4", "5,6", "7,8"\] |
| void_point_pot | -vpp, --void_point_pot | 5 | void point keep potential (ex.) \[\[spring const.(a.u.)\] \[keep distance (ang.)\] \[void_point (x,y,z) (ang.)\] \[atoms(ex. 1,2,3-5)\] \[order p "(1/p)*k*(r - r0)^p"\] ...\] | "void_point_pot": \["0.5", "1.5", "0,0,0", "1,2", "2.0"\] |
| bond_range_potential | -brp, --bond_range_potential | 6 | Add potential to confine atom distance. (ex.) \[\[upper const.(a.u.)\] \[lower const.(a.u.)\] \[upper distance (ang.)\] \[lower distance (ang.)\] \[Fragm.1\] \[Fragm.2\] ...\] | "bond_range_potential": \["0.5", "0.5", "2.0", "1.5", "1,2", "3,4"\] |
| well_pot | -wp, --well_pot | 4 | Add potential to limit atom distance. (ex.) \[\[wall energy (kJ/mol)\] \[fragm.1\] \[fragm.2\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] ...\] | "well_pot": \["100", "1,2", "3-5", "1.0,1.5,2.5,3.0"\] |
| wall_well_pot | -wwp, --wall_well_pot | 4 | Add potential to limit atoms movement. (like sandwich) (ex.) \[\[wall energy (kJ/mol)\] \[direction (x,y,z)\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] \[target atoms (1,2,3-5)\] ...\] | "wall_well_pot": \["100", "z", "0,1,5,6", "1-10"\] |
| void_point_well_pot | -vpwp, --void_point_well_pot | 4 | Add potential to limit atom movement. (like sphere) (ex.) \[\[wall energy (kJ/mol)\] \[coordinate (x,y,z) (ang.)\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] \[target atoms (1,2,3-5)\] ...\] | "void_point_well_pot": \["100", "0,0,0", "0,2,8,10", "1-10"\] |
| around_well_pot | -awp, --around_well_pot | 4 | Add potential to limit atom movement. (like sphere around fragment) (ex.) \[\[wall energy (kJ/mol)\] \[center (1,2-4)\] \[a,b,c,d (a\<b\<c\<d) (ang.)\] \[target atoms (2,3-5)\] ...\] | "around_well_pot": \["100", "1,2", "0,2,8,10", "3-5"\] |
| spacer_model_potential | -smp, --spacer_model_potential | 5 | Add potential based on Morse potential to reproduce solvent molecules around molecule. (ex.) \[\[solvent particle well depth (kJ/mol)\] \[solvent particle e.q. distance (ang.)\] \[scaling of cavity (2.0)\] \[number of particles\] \[target atoms (2,3-5)\] ...\] | "spacer_model_potential": \["5.0", "3.0", "2.0", "10", "1-5"\] |
| metadynamics | -metad, --metadynamics | 4 | apply meta-dynamics (use gaussian potential) (ex.) \[\[\[bond\] \[potential height (kJ/mol)\] \[potential width (ang.)\] \[(atom1),(atom2)\]\] \[\[angle\] \[potential height (kJ/mol)\] \[potential width (deg.)\] \[(atom1),(atom2),(atom3)\]\] ...\] | "metadynamics": \["bond", "10.0", "0.2", "1,2"\] |
| linear_mechano_force_pot | -lmefp, --linear_mechano_force_pot | 3 | add linear mechanochemical force (ex.) \[\[force(pN)\] \[atoms1(ex. 1,2)\] \[atoms2(ex. 3,4)\] ...\] | "linear_mechano_force_pot": \["100", "1,2", "3,4"\] |
| linear_mechano_force_pot_v2 | -lmefpv2, --linear_mechano_force_pot_v2 | 2 | add linear mechanochemical force (ex.) \[\[force(pN)\] \[atom(ex. 1)\] \[direction (xyz)\] ...\] | "linear_mechano_force_pot_v2": \["100", "1,0,0"\] |
| asymmetric_ellipsoidal_repulsive_potential | -aerp, --asymmetric_ellipsoidal_repulsive_potential | 5 | add asymmetric ellipsoidal repulsive potential (use GNB parameters (JCTC, 2024)) (ex.) \[...\] | "asymmetric_ellipsoidal_repulsive_potential": \["10.0", "3.0,3.0,3.0,3.0,3.0,3.0", "3.0", "1,2", "3-5"\] |
| asymmetric_ellipsoidal_repulsive_potential_v2 | -aerpv2, --asymmetric_ellipsoidal_repulsive_potential_v2 | 5 | add asymmetric ellipsoidal repulsive potential (ex.) \[...\] | "asymmetric_ellipsoidal_repulsive_potential_v2": \["10.0", "3.0,3.0,3.0,3.0,3.0,3.0", "3.0", "1,2", "3-5"\] |
| nano_reactor_potential | -nrp, --nano_reactor_potential | 6 | add nano reactor potential (ex.) \[\[inner wall (ang.)\] \[outer wall (ang.)\] \[contraction time (ps)\] \[expansion time (ps)\] \[contraction force const (kcal/mol/A^2)\] \[expansion force const (kcal/mol/A^2)\]\] (Recommendation: 8.0 14.0 1.5 0.5 1.0 0.5) | "nano_reactor_potential": \["8.0", "14.0", "1.5", "0.5", "1.0", "0.5"\] |

## **4\. i-EIP Settings (ieipparser)**

Used by: ieipmain.py

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | -bs, --basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| functional | -func, --functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| NSTEP | -ns, --NSTEP | iter. number | int | 999 | "NSTEP": 500 |
| opt_method | -opt, --opt_method | optimization method for QM calclation (default: FIRE) ... | list\[string\] | \["FIRELARS"\] | "opt_method": \["RFO_BFGS"\] |
| sub_basisset | -sub_bs, --sub_basisset | sub_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub_basisset": \["I", "LanL2DZ"\] |
| effective_core_potential | -ecp, --effective_core_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". | list\[string\] | \[\] | "effective_core_potential": \["I", "LanL2DZ"\] |
| gradient_fix_atoms | -gfix, --gradient_fix_atoms | set the gradient of internal coordinates between atoms to zero (ex.) \[\[atoms (ex.) 1,2\] ...\] | list\[string\] | "" | "gradient_fix_atoms": \["1,2"\] |
| N_THREAD | -core, --N_THREAD | threads | int | 8 | "N_THREAD": 16 |
| microiter | -mi, --microiter | microiteration for relaxing reaction pathways | int | 0 | "microiter": 5 |
| BETA | -beta, --BETA | force for optimization | float | 1.0 | "BETA": 0.5 |
| SET_MEMORY | -mem, --SET_MEMORY | use mem(ex. 1GB) | string | "2GB" | "SET_MEMORY": "8GB" |
| excited_state | -es, --excited_state | calculate excited state (default: \[0(initial state), 0(final state)\]) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | list\[int\] | \[0, 0\] | "excited_state": \[0, 1\] |
| model_function_mode | -mf, --model_function_mode | use model function to optimization (seam, avoiding, conical, mesx, meci) | string | "None" | "model_function_mode": "seam" |
| calc_exact_hess | -fc, --calc_exact_hess | calculate exact hessian per steps (ex.) \[steps per one hess calculation\] | int | -1 | "calc_exact_hess": 1 |
| usextb | -xtb, --usextb | use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| usedxtb | -dxtb, --usedxtb | use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usedxtb": "GFN1-xTB" |
| sqm1 | -sqm1, --sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | -sqm2, --sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| pyscf | -pyscf, --pyscf | use pyscf module. | boolean | false | "pyscf": true |
| unrestrict | -u, --unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| electronic_charge | -elec, --electronic_charge | formal electronic charge (ex.) \[charge (0)\] | list\[int\] | \[0, 0\] | "electronic_charge": \[0, 0\] |
| spin_multiplicity | -spin, --spin_multiplicity | spin multiplcity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha - Nbeta)) (ex.) \[multiplcity (0)\] | list\[int\] | \[1, 1\] | "spin_multiplicity": \[1, 1\] |
| cpcm_solv_model | -cpcm, --cpcm_solv_model | use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "cpcm_solv_model": "water" |
| alpb_solv_model | -alpb, --alpb_solv_model | use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "alpb_solv_model": "toluene" |
| dft_grid | -grid, --dft_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft_grid": 4 |
| othersoft | -os, --othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc. | string | "None" | "othersoft": "orca" |
| software_path_file | -osp, --software_path_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf | string | "./software_path.conf" | "software_path_file": "/path/to/my.conf" |
| use_gnt | -gnt, --use_gnt | Use GNT (Growing Newton Trajectory) | boolean | false | "use_gnt": true |
| gnt_vec | -gnt_vec, --gnt_vec | set vector to calculate Newton trajectory (ex. 1,2,3 (default:calculate vector reactant to product) ) | string | None | "gnt_vec": "1,2,3" |
| gnt_step_len | -gnt_step, --gnt_step_len | set step length for Newton trajectory (default: 0.5) | float | 0.5 | "gnt_step_len": 0.25 |
| gnt_microiter | -gnt_mi, --gnt_microiter | max number of micro-iteration for Newton trajectory (default: 25\) | int | 25 | "gnt_microiter": 10 |
| use_addf | -addf, --use_addf | Use ADDF-like method (default: False) | boolean | false | "use_addf": true |
| addf_step_size | -addf_step, --addf_step_size | set step size for ADDF-like method (default: 0.1) | float | 0.1 | "addf_step_size": 0.2 |
| addf_step_num | -addf_num, --addf_step_num | set number of steps for ADDF-like method (default: 300\) | int | 300 | "addf_step_num": 100 |
| number_of_add | -addf_nadd, --number_of_add | set number of number of searching ADD (A larger ADD takes precedence.) for ADDF-like method (default: 5\) | int | 5 | "number_of_add": 3 |
| use_2pshs | -2pshs, --use_2pshs | Use 2PSHS-like method (default: False) | boolean | false | "use_2pshs": true |
| twoPshs_step_size | -2pshs_step, --twoPshs_step_size | set step size for 2PSHS-like method (default: 0.05) | float | 0.05 | "twoPshs_step_size": 0.1 |
| twoPshs_step_num | -2pshs_num, --twoPshs_step_num | set number of steps for 2PSHS-like method (default: 300\) | int | 300 | "twoPshs_step_num": 100 |
| use_dimer | -use_dimer, --use_dimer | Use Dimer method for searching direction of TS (default: False) | boolean | false | "use_dimer": true |
| dimer_separation | -dimer_sep, --dimer_separation | set dimer separation (default: 0.0001) | float | 0.0001 | "dimer_separation": 0.0005 |
| dimer_trial_angle | -dimer_trial_angle, --dimer_trial_angle | set dimer trial angle (default: pi/32) | float | 0.09817... | "dimer_trial_angle": 0.1 |
| dimer_max_iterations | -dimer_maxiter, --dimer_max_iterations | set max iterations for dimer method (default: 1000\) | int | 1000 | "dimer_max_iterations": 500 |

## **5\. Molecular Dynamics Settings (mdparser)**

Used by: mdmain.py

| JSON Key | Command-Line Argument | Description | Type | Default | Example (JSON) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| basisset | -bs, --basisset | basisset (ex. 6-31G\*) | string | "6-31G(d)" | "basisset": "6-31G\*" |
| functional | -func, --functional | functional(ex. b3lyp) | string | "b3lyp" | "functional": "b3lyp" |
| sub_basisset | -sub_bs, --sub_basisset | sub_basisset (ex. I LanL2DZ) | list\[string\] | \[\] | "sub_basisset": \["I", "LanL2DZ"\] |
| effective_core_potential | -ecp, --effective_core_potential | ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". | list\[string\] | \[\] | "effective_core_potential": \["I", "LanL2DZ"\] |
| excited_state | -es, --excited_state | calculate excited state (default: 0\) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state. | int | 0 | "excited_state": 1 |
| additional_inputs | -addint, --additional_inputs | (ex.) \[(excited state) (fromal charge) (spin multiplicity) ...\] | list\[int\] | \[\] | "additional_inputs": \[1, -1, 2\] |
| NSTEP | -time, --NSTEP | time scale | int | 100000 | "NSTEP": 50000 |
| TRAJECTORY | -traj, --TRAJECTORY | number of trajectory to generate (default) 1 | int | 1 | "TRAJECTORY": 10 |
| temperature | -temp, --temperature | temperature \[unit. K\] (default) 298.15 K | float | 298.15 | "temperature": 300.0 |
| timestep | -ts, --timestep | time step \[unit. atom unit\] (default) 0.1 a.u. | float | 0.1 | "timestep": 0.5 |
| pressure | -press, --pressure | pressure \[unit. kPa\] (default) 1013 kPa | float | 101.3 | "pressure": 101.3 |
| N_THREAD | -core, --N_THREAD | threads | int | 8 | "N_THREAD": 16 |
| SET_MEMORY | -mem, --SET_MEMORY | use mem(ex. 1GB) | string | "1GB" | "SET_MEMORY": "4GB" |
| unrestrict | -u, --unrestrict | use unrestricted method (for radical reaction and excite state etc.) | boolean | false | "unrestrict": true |
| mdtype | -mt, --mdtype | specify condition to do MD (ex.) velocityverlet (default) nosehoover | string | "nosehoover" | "mdtype": "velocityverlet" |
| fix_atoms | -fix, --fix_atoms | fix atoms (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \[\] | "fix_atoms": \["1,2,3-6"\] |
| geom_info | -gi, --geom_info | calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) \[atoms (ex.) 1,2,3-6\] | list\[string\] | \["1"\] | "geom_info": \["1,2", "1,2,3"\] |
| pyscf | -pyscf, --pyscf | use pyscf module. | boolean | false | "pyscf": true |
| electronic_charge | -elec, --electronic_charge | formal electronic charge (ex.) \[charge (0)\] | int | 0 | "electronic_charge": -1 |
| spin_multiplicity | -spin, --spin_multiplicity | spin multiplcity (if you use pyscf, please input S value (mol.spin \= 2S \= Nalpha - Nbeta)) (ex.) \[multiplcity (0)\] | int | 1 | "spin_multiplicity": 2 |
| saddle_order | -order, --saddle_order | optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) \[order (0)\] | int | 0 | "saddle_order": 0 |
| cmds | -cmds, --cmds | apply classical multidimensional scaling to calculated approx. reaction path. | boolean | false | "cmds": true |
| usextb | -xtb, --usextb | use extended tight bonding method to calculate. default is GFN2-xTB (ex.) GFN1-xTB, GFN2-xTB | string | "None" | "usextb": "GFN2-xTB" |
| sqm1 | -sqm1, --sqm1 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm1": true |
| sqm2 | -sqm2, --sqm2 | use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method. | boolean | false | "sqm2": true |
| change_temperature | -ct, --change_temperature | change temperature of thermostat (defalut) No change (ex.) \[1000(time), 500(K) 5000(time), 1000(K)...\] | list\[string\] | \[\] | "change_temperature": \["1000", "500", "5000", "1000"\] |
| constraint_condition | -cc, --constraint_condition | apply constraint conditions for optimazation (ex.) \[\[(dinstance (ang.)), (atom1),(atom2)\] \[(bond_angle (deg.)), (atom1),(atom2),(atom3)\] \[(dihedral_angle (deg.)), (atom1),(atom2),(atom3),(atom4)\] ...\] | list\[string\] | \[\] | "constraint_condition": \["1.5,1,2", "109.5,1,2,3"\] |
| othersoft | -os, --othersoft | use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc. | string | "None" | "othersoft": "orca" |
| software_path_file | -osp, --software_path_file | read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf | string | "./software_path.conf" | "software_path_file": "/path/to/my.conf" |
| periodic_boundary_condition | -pbc, --periodic_boundary_condition | apply periodic boundary condition (Default is not applying.) (ex.) \[periodic boundary (x,y,z) (ang.)\] | list\[string\] | \[\] | "periodic_boundary_condition": \["10.0,10.0,10.0"\] |
| projection_constrain | -pc, --projection_constrain | apply constrain conditions with projection of gradient and hessian (ex.) \[\[(constraint condition name) (atoms(ex. 1,2))\] ...\] | list\[string\] | \[\] | "projection_constrain": \["bond", "1,2"\] |
| cpcm_solv_model | -cpcm, --cpcm_solv_model | use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "cpcm_solv_model": "water" |
| alpb_solv_model | -alpb, --alpb_solv_model | use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water | string | None | "alpb_solv_model": "toluene" |
| pca | -pca, --pca | Apply principal component analysis to calculated approx. reaction path. | boolean | false | "pca": true |
| dft_grid | -grid, --dft_grid | fineness of grid for DFT calculation (default: 3 (0\~9)) | int | 3 | "dft_grid": 4 |

