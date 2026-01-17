import argparse
import sys
import numpy as np


"""
    MultiOptPy
    Copyright (C) 2023-2026 ss0832

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


def ieipparser(parser, args_list=None):
    parser = call_ieipparser(parser)
    parser = parser_for_biasforce(parser)
    
    # Pass the args_list to parser2args
    args = parser2args(parser, args_list)
    
    args.fix_atoms = []
    args.gradient_fix_atoms = []
    args.geom_info = ["0"]
    args.projection_constrain = []
    args.opt_fragment = []
    args.oniom_flag = []
    return args

def optimizeparser(parser, args_list=None):
    parser = call_optimizeparser(parser)
    parser = parser_for_biasforce(parser)

    # Pass the args_list to parser2args
    args = parser2args(parser, args_list)

    # Handle INPUT logic safely
    if isinstance(args.INPUT, list) and len(args.INPUT) == 1:
        args.INPUT = args.INPUT[0]
    
    args.constraint_condition = []
    return args

def nebparser(parser, args_list=None):
    parser = call_nebparser(parser)
    parser = parser_for_biasforce(parser)

    # Pass the args_list to parser2args
    args = parser2args(parser, args_list)
    
    args.geom_info = ["0"]
    args.opt_method = ""
    args.opt_fragment = []
    args.oniom_flag = []
    return args

def mdparser(parser, args_list=None):
    parser = call_mdparser(parser)
    parser = parser_for_biasforce(parser)

    # Pass the args_list to parser2args
    args = parser2args(parser, args_list)
    
    args.geom_info = ["0"]
    args.opt_method = ""
    args.opt_fragment = []
    args.oniom_flag = []
    return args


def call_ieipparser(parser):
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-ns", "--NSTEP",  type=int, default='999', help='iter. number')
    
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["FIRELARS"], help='optimization method for QM calclation (default: FIRE) (mehod_list:(steepest descent method group) FIRE, CG etc. (quasi-Newton method group) RFO_FSB, RFO_BFGS, RFO3_Bifill  etc.) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-ecp", "--effective_core_potential", type=str, nargs="*", default='', help='ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". ')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default="", help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mi", "--microiter",  type=int, default=0, help='microiteration for relaxing reaction pathways')
    parser.add_argument("-beta", "--BETA",  type=float, default='1.0', help='force for optimization')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-es", "--excited_state", type=int, nargs="*", default=[0, 0],
                        help='calculate excited state (default: [0(initial state), 0(final state)]) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')
    parser.add_argument("-mf", "--model_function_mode", help="use model function to optimization (seam, avoiding, conical, mesx, meci)", type=str, default='None',)
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    
    
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-dxtb", "--usedxtb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd diffential method is available.)) (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-sqm1", "--sqm1", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument("-sqm2", "--sqm2", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, nargs="*", default=[0, 0], help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, nargs="*", default=[1, 1], help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-cpcm", "--cpcm_solv_model",  type=str, default=None, help='use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water')
    parser.add_argument("-alpb", "--alpb_solv_model",  type=str, default=None, help='use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water')#ref.: J. Chem. Theory Comput. 2021, 17, 7, 4250–4261 https://doi.org/10.1021/acs.jctc.1c00471
    parser.add_argument("-grid", "--dft_grid", type=int, default=3, help="fineness of grid for DFT calculation (default: 3 (0~9))")
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument("-osp", "--software_path_file",  type=str, default="./software_path.conf", help='read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf')
    
    # Other Chain of State methods
    parser.add_argument('-gnt','--use_gnt', help="Use GNT (Growing Newton Trajectory)", action='store_true')
    parser.add_argument('-gnt_vec','--gnt_vec', help="set vector to calculate Newton trajectory (ex. 1,2,3 (default:calculate vector reactant to product) )", type=str, default=None)
    parser.add_argument('-gnt_step','--gnt_step_len', help="set step length for Newton trajectory (default: 0.5)", type=float, default=0.5)
    parser.add_argument('-gnt_mi','--gnt_microiter', help="max number of micro-iteration for Newton trajectory (default: 25)", type=int, default=25)

    parser.add_argument('-addf','--use_addf', help="Use ADDF-like method (default: False)", action='store_true')
    parser.add_argument('-addf_step','--addf_step_size', help="set step size for ADDF-like method (default: 0.1)", type=float, default=0.1)
    parser.add_argument('-addf_num','--addf_step_num', help="set number of steps for ADDF-like method (default: 300)", type=int, default=300)
    parser.add_argument('-addf_nadd','--number_of_add', help="set number of number of searching ADD (A larger ADD takes precedence.) for ADDF-like method (default: 5)", type=int, default=5)
    
    parser.add_argument('-2pshs','--use_2pshs', help="Use 2PSHS-like method (default: False)", action='store_true')
    parser.add_argument('-2pshs_step','--twoPshs_step_size', help="set step size for 2PSHS-like method (default: 0.05)", type=float, default=0.05)
    parser.add_argument('-2pshs_num','--twoPshs_step_num', help="set number of steps for 2PSHS-like method (default: 300)", type=int, default=300)

    parser.add_argument('-use_dimer','--use_dimer', help="Use Dimer method for searching direction of TS (default: False)", action='store_true'
                        )
    parser.add_argument('-dimer_sep','--dimer_separation', help="set dimer separation (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument('-dimer_trial_angle','--dimer_trial_angle', help="set dimer trial angle (default: pi/32)", type=float, default=np.pi / 32.0)
    parser.add_argument('-dimer_maxiter','--dimer_max_iterations', help="set max iterations for dimer method (default: 1000)", type=int, default=1000)
    parser.add_argument('-use_spm','--use_spm', help="Use Spring Pair method for searching direction of TS (default: False)", action='store_true'
                        )    
    return parser

def call_optimizeparser(parser):
    parser.add_argument("INPUT", help='input xyz file name', nargs="*")
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-ecp", "--effective_core_potential", type=str, nargs="*", default='', help='ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". ')
    parser.add_argument("-es", "--excited_state", type=int, default=0, help='calculate excited state (default: 0) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')
    parser.add_argument("-ns", "--NSTEP",  type=int, default='1000', help='number of iteration (default: 1000)')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')
    parser.add_argument("-tr", "--max_trust_radius",  type=float, default=None, help='max trust radius to restrict step size (unit: ang.) (default: 0.1 for n-th order saddle point optimization, 0.5 for minimum point optimization) (notice: default minimum trust radius is 0.01)')
    parser.add_argument("-mintr", "--min_trust_radius",  type=float, default=0.01, help='min trust radius to restrict step size (unit: ang.) (default: 0.01) ')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-dc", "--dissociate_check", nargs="*",  type=str, default="10", help='Terminate calculation if distance between two fragments is exceed this value. (default) 10 [ang.]')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["FIRELARS"], help='optimization method for QM calculation (default: FIRE) (mehod_list:(steepest descent method group) FIRE, CG etc. (quasi-Newton method group) rsirfo_fsb rsirfo_bofill  etc.) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per geometry optimization steps and IRC steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-mfc", "--calc_model_hess",  type=int, default=50, help='calculate model hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-dxtb", "--usedxtb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd differential method is available.)) (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-sqm1", "--sqm1", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument("-sqm2", "--sqm2", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument("-cpcm", "--cpcm_solv_model",  type=str, default=None, help='use CPCM solvent model for xTB (Default setting is not using this model.) (ex.) water')
    parser.add_argument("-alpb", "--alpb_solv_model",  type=str, default=None, help='use ALPB solvent model for xTB (Default setting is not using this model.) (ex.) water')#ref.: J. Chem. Theory Comput. 2021, 17, 7, 4250–4261 https://doi.org/10.1021/acs.jctc.1c00471

    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplicity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplicity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    parser.add_argument('-cmds','--cmds', help="Apply classical multidimensional scaling to calculated approx. reaction path.", action='store_true')
    parser.add_argument('-pca','--pca', help="Apply principal component analysis to calculated approx. reaction path.", action='store_true')
    parser.add_argument('-km', '--koopman', help="Apply Koopman model to analyze the convergence", action='store_true')
    parser.add_argument('-irc','--intrinsic_reaction_coordinates', help="Calculate intrinsic reaction coordinates. (ex.) [[step_size], [max_step], [IRC_method]] (Recommended) [0.5 300 lqa]", nargs="*", type=str, default=[])    
    parser.add_argument("-of", "--opt_fragment", nargs="*", type=str, default=[], help="Several atoms are grouped together as fragments and optimized. (This method does not work if you use quasi-newton method for optimazation.) (ex.) [[atoms (ex.) 1-4] ...] ")#(2024/3/26) this option doesn't work if you use quasi-Newton method for optimization.
    parser.add_argument("-grid", "--dft_grid", type=int, default=3, help="fineness of grid for DFT calculation (default: 3 (0~9))")
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument("-osp", "--software_path_file",  type=str, default="./software_path.conf", help='read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf')    
    parser.add_argument('-tcc','--tight_convergence_criteria', help="apply tight opt criteria.", action='store_true')
    parser.add_argument('-lcc','--loose_convergence_criteria', help="apply loose opt criteria.", action='store_true')

    class ModelhessAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values is None:
                setattr(namespace, self.dest, 'fischerd3')
            else:
                setattr(namespace, self.dest, values)
    
    parser.add_argument('-modelhess','--use_model_hessian', nargs='?', help="use model hessian. (Default: not using model hessian If you specify only option, Improved Lindh + Grimme's D3 dispersion model hessian is used.) (ex. lindh, gfnff, gfn0xtb, fischer, fischerd3, fischerd4, schlegel, swart, lindh2007, lindh2007d3, lindh2007d4)", action=ModelhessAction, default=None)
    parser.add_argument("-sc", "--shape_conditions", nargs="*", type=str, default=[], help="Exit optimization if these conditions are not satisfied. (e.g.) [[(ang.) gt(lt) 2,3 (bond)] [(deg.) gt(lt) 2,3,4 (bend)] ...] [[(deg.) gt(lt) 2,3,4,5 (torsion)] ...]")
    parser.add_argument("-pc", "--projection_constrain", nargs="*",  type=str, default=[], help='apply constrain conditions with projection of gradient and hessian (ex.) [[(constraint condition name) (atoms(ex. 1,2))] ...] ')
    parser.add_argument("-oniom", "--oniom_flag", nargs="*",  type=str, default=[], help='apply ONIOM method (Warning: This option is unavailable.)')
    parser.add_argument("-freq", "--frequency_analysis",  help="Perform normal vibrational analysis after converging geometry optimization. (Caution: Unable to use this analysis with oniom method)", action='store_true')
    parser.add_argument("-temp", "--temperature",  type=float, default='298.15', help='temperatrue to calculate thermochemistry (Unit: K) (default: 298.15K)')
    parser.add_argument("-press", "--pressure",  type=float, default='101325', help='pressure to calculate thermochemistry (Unit: Pa) (default: 101325Pa)')
    parser.add_argument("-negeigval", "--detect_negative_eigenvalues", help="Detect negative eigenvalues in the Hessian matrix at ITR. 0 if you calculate exact hessian (-fc >0). If negative eigenvalues are not detected and saddle_order > 0, the optimization is stopped.", action='store_true')
    parser.add_argument("-mf", "--model_function", nargs="*",  type=str, default=[], help='minimize model function(ex.) [[model function type (seam, avoid, conical etc.)] [electronic charge] [spin multiplicity]] ')
    
    return parser

def parser_for_biasforce(parser):
    parser.add_argument("-ma", "--manual_AFIR", nargs="*",  type=str, default=[], help='manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]')
    parser.add_argument("-rp", "--repulsive_potential", nargs="*",  type=str, default=[], help='Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpv2", "--repulsive_potential_v2", nargs="*",  type=str, default=[], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpg", "--repulsive_potential_gaussian", nargs="*",  type=str, default=[], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε_LJ[(σ/r)^(12) - 2 * (σ/r)^(6)] - ε_gau * exp(-((r-σ_gau)/b)^2)) (ex.) [[LJ_well_depth (kJ/mol)] [LJ_dist (ang.)] [Gaussian_well_depth (kJ/mol)] [Gaussian_dist (ang.)] [Gaussian_range (ang.)] [Fragm.1 (1,2)] [Fragm.2 (3-5,8)] ...]')
    
    parser.add_argument("-cp", "--cone_potential", nargs="*",  type=str, default=[], help='Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
    
    parser.add_argument("-fp", "--flux_potential", nargs="*",  type=str, default=[], help='Add potential to make flow. ( k/p*(x-x_0)^p )(ex.) [[x,y,z (constant (a.u.))] [x,y,z (order)] [x,y,z coordinate (ang.)] [Fragm.(ex. 1,2,3-5)] ...]')
    parser.add_argument("-kp", "--keep_pot", nargs="*",  type=str, default=[], help='keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-kpv2", "--keep_pot_v2", nargs="*",  type=str, default=[], help='keep potential_v2 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [Fragm.1] [Fragm.2] ...] ')
    parser.add_argument("-akp", "--anharmonic_keep_pot", nargs="*",  type=str, default=[], help='Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-ka", "--keep_angle", nargs="*",  type=str, default=[], help='keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] ')
    parser.add_argument("-kav2", "--keep_angle_v2", nargs="*",  type=str, default=[], help='keep angle_v2 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] ...] ')
    parser.add_argument("-up", "--universal_potential", nargs="*",  type=str, default=[], help="Potential to gather specified atoms to a single point (ex.) [[potential (kJ/mol)] [target atoms (1,2)] ...]")
    
    parser.add_argument("-ddka", "--atom_distance_dependent_keep_angle", nargs="*",  type=str, default=[], help='atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] ')
    
    
    parser.add_argument("-kda", "--keep_dihedral_angle", nargs="*",  type=str, default=[], help='keep dihedral angle 0.5*k*(φ - φ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] ')
    parser.add_argument("-kopa", "--keep_out_of_plain_angle", nargs="*",  type=str, default=[], help='keep_out_of_plain_angle 0.5*k*(φ - φ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep out of plain angle (degrees)] [atom1,atom2,atom3,atom4] ...] ')
    parser.add_argument("-kdav2", "--keep_dihedral_angle_v2", nargs="*",  type=str, default=[], help='keep dihedral angle_v2 0.5*k*(φ - φ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] [Fragm.4] ...] ')
    parser.add_argument("-kdac", "--keep_dihedral_angle_cos", nargs="*",  type=str, default=[], help='keep dihedral angle_cos k*[1 + cos(n * φ - (φ0 + pi))] (0 ~ 180 deg.) (ex.) [[potential const.(a.u.)] [angle const. (unitless)] [keep dihedral angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] [Fragm.4] ...] ')
    parser.add_argument("-kopav2", "--keep_out_of_plain_angle_v2", nargs="*",  type=str, default=[], help='keep out_of_plain angle_v2 0.5*k*(φ - φ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep out_of_plain angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] [Fragm.4] ...] ')
    parser.add_argument("-vpp", "--void_point_pot", nargs="*",  type=str, default=[], help='void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] ')
    
    parser.add_argument("-brp", "--bond_range_potential", nargs="*",  type=str, default=[], help='Add potential to confine atom distance. (ex.) [[upper const.(a.u.)] [lower const.(a.u.)] [upper distance (ang.)] [lower distance (ang.)] [Fragm.1] [Fragm.2] ...] ')
    parser.add_argument("-wp", "--well_pot", nargs="*", type=str, default=[], help="Add potential to limit atom distance. (ex.) [[wall energy (kJ/mol)] [fragm.1] [fragm.2] [a,b,c,d (a<b<c<d) (ang.)] ...]")
    parser.add_argument("-wwp", "--wall_well_pot", nargs="*", type=str, default=[], help="Add potential to limit atoms movement. (like sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (1,2,3-5)] ...]")
    parser.add_argument("-vpwp", "--void_point_well_pot", nargs="*", type=str, default=[], help="Add potential to limit atom movement. (like sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (1,2,3-5)] ...]")
    parser.add_argument("-awp", "--around_well_pot", nargs="*", type=str, default=[], help="Add potential to limit atom movement. (like sphere around fragment) (ex.) [[wall energy (kJ/mol)] [center (1,2-4)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")
    parser.add_argument("-smp", "--spacer_model_potential", nargs="*", type=str, default=[], help="Add potential based on Morse potential to reproduce solvent molecules around molecule. (ex.) [[solvent particle well depth (kJ/mol)] [solvent particle e.q. distance (ang.)] [scaling of cavity (2.0)] [number of particles] [target atoms (2,3-5)] ...]")
    parser.add_argument("-metad", "--metadynamics", nargs="*", type=str, default=[], help="apply meta-dynamics (use gaussian potential) (ex.) [[[bond] [potential height (kJ/mol)] [potential width (ang.)] [(atom1),(atom2)]] [[angle] [potential height (kJ/mol)] [potential width (deg.)] [(atom1),(atom2),(atom3)]] [[dihedral] [potential height (kJ/mol)] [potential width (deg.)] [(atom1),(atom2),(atom3),(atom4)]] [[outofplain] [potential height (kJ/mol)] [potential width (deg.)] [(atom1),(atom2),(atom3),(atom4)]]...] ")
    parser.add_argument("-lmefp", "--linear_mechano_force_pot", nargs="*",  type=str, default=[], help='add linear mechanochemical force (ex.) [[force(pN)] [atoms1(ex. 1,2)] [atoms2(ex. 3,4)] ...]')
    parser.add_argument("-lmefpv2", "--linear_mechano_force_pot_v2", nargs="*",  type=str, default=[], help='add linear mechanochemical force (ex.) [[force(pN)] [atom(ex. 1)] [direction (xyz)] ...]')
    parser.add_argument("-aerp", "--asymmetric_ellipsoidal_repulsive_potential", nargs="*",  type=str, default=[], help='add asymmetric ellipsoidal repulsive potential (use GNB parameters (JCTC, 2024)) (ex.) [[well_value (epsilon) (kJ/mol)] [dist_value (sigma) (a1,a2,b1,b2,c1,c2) (ang.)] [dist_value (distance) (ang.)] [target atom (1,2)] [off target atoms (3-5)] ...]')
    parser.add_argument("-aerpv2", "--asymmetric_ellipsoidal_repulsive_potential_v2", nargs="*",  type=str, default=[], help='add asymmetric ellipsoidal repulsive potential (ex.) [[well_value (epsilon) (kJ/mol)] [dist_value (sigma) (a1,a2,b1,b2,c1,c2) (ang.)] [dist_value (distance) (ang.)] [target atom (1,2)] [off target atoms (3-5)] ...]')
    parser.add_argument("-nrp", "--nano_reactor_potential", nargs="*",  type=str, default=[], help='add nano reactor potential (ex.) [[inner wall (ang.)] [outer wall (ang.)] [contraction time (ps)] [expansion time (ps)] [contraction force const (kcal/mol/A^2)] [expansion force const (kcal/mol/A^2)]] (Recommendation: 8.0 14.0 1.5 0.5 1.0 0.5)')
    return parser

def call_nebparser(parser):
    parser.add_argument("INPUT", help='input folder', nargs="*")
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-ecp", "--effective_core_potential", type=str, nargs="*", default='', help='ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". ')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-es", "--excited_state", type=int, default=0, help='calculate excited state (default: 0) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='10', help='iter. number')
    parser.add_argument("-om", "--OM", action='store_true', help='J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action.')
    parser.add_argument("-lup", "--LUP", action='store_true', help='J. Chem. Phys. 92, 1510–1511 (1990) doi:https://doi.org/10.1063/1.458112 locally updated planes (LUP) method')
    parser.add_argument("-bneb", "--BNEB", action='store_true', help="NEB using Wilson's B matrix for calculating the perpendicular force.")
    parser.add_argument("-bneb2", "--BNEB2", action='store_true', help="NEB using Wilson's B matrix for calculating the perpendicular force with parallel spring force.")
    parser.add_argument("-dneb", "--DNEB", action='store_true', help='J. Chem. Phys. 120, 2082–2094 (2004) doi:https://doi.org/10.1063/1.1636455 doubly NEB method (DNEB) method')
    parser.add_argument("-nesb", "--NESB", action='store_true', help='J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169 Nudged elastic stiffness band (NESB) method')
    parser.add_argument("-dmf", "--DMF", action='store_true', help='Direct Max Flux (DMF) method')
    parser.add_argument("-ewbneb", "--EWBNEB", action='store_true', help='Energy-weighted Nudged elastic band method')
    parser.add_argument("-qsm", "--QSM", action='store_true', help='Quadratic String Method (J. Chem. Phys. 124, 054109 (2006))')
    parser.add_argument("-qsmv2", "--QSMv2", action='store_true', help='Quadratic String Method v2 (J. Chem. Phys. 124, 054109 (2006))')
    parser.add_argument("-aneb", "--ANEB", default=None, nargs="*", help='Adaptic NEB method (ref.: J. Chem. Phys. 117, 4651 (2002)) (Usage: -aneb [interpolation_num (ex. 2)] [frequency (ex. 5)], Default setting is not applying adaptic NEB method.)')

    parser.add_argument("-idpp", "--use_image_dependent_pair_potential", action='store_true', help='use image dependent pair potential (IDPP) method (ref. arXiv:1406.1512v1)')
    parser.add_argument("-cfbenm", "--use_correlated_flat_bottom_elastic_network_model", action='store_true', help='use correlated flat-bottom elastic network model (CFBENM) method (ref. s: J.Chem.TheoryComput.2025,21,3513−3522)')
    parser.add_argument("-ad", "--align_distances", type=int, default=0, help='distribute images at equal intervals on the reaction coordinate')
    parser.add_argument("-adene", "--align_distances_energy", type=int, default=0, help='distribute images at energy-weighted intervals on the reaction coordinate')
    parser.add_argument("-adpred", "--align_distances_energy_predicted", type=int, default=0, help='distribute images at intervals on the reaction coordinate using cubic predicted interpolation')
    parser.add_argument("-ads", "--align_distances_spline", type=int, default=0, help='distribute images at equal intervals on the reaction coordinate using spline interpolation')
    parser.add_argument("-ads2", "--align_distances_spline_ver2", type=int, default=0, help='distribute images at equal intervals on the reaction coordinate using spline interpolation ver.2')
    parser.add_argument("-adg", "--align_distances_geodesic", type=int, default=0, help='distribute images at equal intervals on the reaction coordinate using geodesic interpolation')
    parser.add_argument("-adb", "--align_distances_bernstein", type=int, default=0, help='distribute images at equal intervals on the reaction coordinate using Bernstein interpolation')
    parser.add_argument("-adbene", "--align_distances_bernstein_energy", type=int, default=0, help='distribute images at energy-weighted intervals on the reaction coordinate using Bernstein interpolation')
    
  
    parser.add_argument("-adadene", "--align_distances_adaptive_energy", type=int, default=0, help='distribute images at energy-weighted intervals on the reaction coordinate using Adaptive Geometry + Energy interpolation')
    

    
    parser.add_argument("-adsg", "--align_distances_savgol", type=str, default="0,0,0", help='distribute images at equal intervals on the reaction coordinate using Savitzky-Golay interpolation (ex.) [[iteration(int)],[window_size(int, 5 is recommended)],[poly_order(int) 3 is recommended]] (default: 0,0,0 (not using Savitzky-Golay interpolation))')
    
    parser.add_argument("-nd", "--node_distance", type=float, default=None, help='distribute images at equal intervals linearly based ont specific distance (ex.) [distance (ang.)] (default: None)')
    parser.add_argument("-nds", "--node_distance_spline", type=float, default=None, help='distribute images at equal intervals using spline interpolation based ont specific distance (ex.) [distance (ang.)] (default: None)')
    parser.add_argument("-ndb", "--node_distance_bernstein", type=float, default=None, help='distribute images at equal intervals using Bernstein interpolation based ont specific distance (ex.) [distance (ang.)] (default: None)')
    parser.add_argument("-ndsg", "--node_distance_savgol", type=str, default=None, help='distribute images at equal intervals using Savitzky-Golay interpolation based ont specific distance (ex.) [[distance (ang.)],[window_size(int, 5 is recommended)],[poly_order(int) 3 is recommended]] (default: None)')
    parser.add_argument("-p", "--partition",  type=int, default='0', help='number of nodes')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-cineb", "--apply_CI_NEB",  type=int, default='99999', help='apply CI_NEB method')
    parser.add_argument("-sd", "--steepest_descent",  type=int, default='99999', help='apply steepest_descent method')

    
    class CGAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values is None:
                setattr(namespace, self.dest, 'HS')
            else:
                setattr(namespace, self.dest, values)
    
    parser.add_argument("-cg", "--conjugate_gradient", nargs='?', help='apply conjugate_gradient method for path optimization (Available update method of CG parameters :FR, PR, HS, DY, HZ), default update method is HS.) ', action=CGAction, default=False)
    parser.add_argument("-lbfgs", "--memory_limited_BFGS", action='store_true', help='apply L-BFGS method for path optimization ')
    parser.add_argument("-notsopt", "--not_ts_optimization", action='store_true', help='not apply TS optimization during NEB calculation')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-gqnt", "--global_quasi_newton",  action='store_true', help='use global quasi-Newton method')
    
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-dxtb", "--usedxtb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd diffential method is available.)) (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-sqm1", "--sqm1", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument("-sqm2", "--sqm2", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-fe", "--fixedges",  type=int, default=0, help='fix edges of nodes (1=initial_node, 2=end_node, 3=both_nodes) ')
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default=[], help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-pc", "--projection_constrain", nargs="*",  type=str, default=[], help='apply constrain conditions with projection of gradient and hessian (ex.) [[(constraint condition name) (atoms(ex. 1,2))] ...] ')
    parser.add_argument("-cpcm", "--cpcm_solv_model",  type=str, default=None, help='use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water')
    parser.add_argument("-alpb", "--alpb_solv_model",  type=str, default=None, help='use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water')#ref.: J. Chem. Theory Comput. 2021, 17, 7, 4250–4261 https://doi.org/10.1021/acs.jctc.1c00471
    parser.add_argument("-spng", "--save_pict",  action='store_true', help='Save picture for visualization.')
    parser.add_argument("-aconv", "--apply_convergence_criteria",  action='store_true', help='Apply convergence criteria for NEB calculation.')
    parser.add_argument("-ci", "--climbing_image", type=int, default=[999999, 999999], nargs="*", help='climbing image for NEB calculation. (start of ITR., interval) The default setting is not using climbing image.')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-grid", "--dft_grid", type=int, default=3, help="fineness of grid for DFT calculation (default: 3 (0~9))")

    class ModelhessAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values is None:
                setattr(namespace, self.dest, 'fischerd3')
            else:
                setattr(namespace, self.dest, values)
                
    parser.add_argument('-modelhess','--use_model_hessian', nargs='?', help="use model hessian. (Default: not using model hessian If you specify only option, Fischer + Grimme's D3 dispersion model hessian is used.) (ex. lindh, gfnff, gfn0xtb, fischer, fischerd3, fischerd4, schlegel, swart, lindh2007, lindh2007d3, lindh2007d4)", action=ModelhessAction, default=None)
    parser.add_argument("-mfc", "--calc_model_hess",  type=int, default=50, help='calculate model hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument("-osp", "--software_path_file",  type=str, default="./software_path.conf", help='read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf')
    parser.add_argument("-rrs", "--ratio_of_rfo_step", type=float, default=0.5, help='ratio of rfo step (default: 0.5).  This option is for optimizer using Hessian (-fc or -modelhess).')

    
    return parser
    
def call_mdparser(parser):
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-ecp", "--effective_core_potential", type=str, nargs="*", default='', help='ECP (ex. I LanL2DZ) (notice) If you assign ECP to all atoms of inputs, type "default (basis_set name)". ')
    parser.add_argument("-es", "--excited_state", type=int, default=0,
                        help='calculate excited state (default: 0) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')
    parser.add_argument("-addint", "--additional_inputs", type=int, nargs="*", default=[], help=' (ex.) [(excited state) (fromal charge) (spin multiplicity) ...]')
    parser.add_argument("-time", "--NSTEP",  type=int, default='100000', help='time scale')
    parser.add_argument("-traj", "--TRAJECTORY",  type=int, default='1', help='number of trajectory to generate (default) 1')
   
    parser.add_argument("-temp", "--temperature",  type=float, default='298.15', help='temperature [unit. K] (default) 298.15 K')
    parser.add_argument("-ts", "--timestep",  type=float, default=0.1, help='time step [unit. atom unit] (default) 0.1 a.u.')
    parser.add_argument("-press", "--pressure",  type=float, default='101.3', help='pressure [unit. kPa] (default) 1013 kPa')
    
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-mt", "--mdtype",  type=str, default='nosehoover', help='specify condition to do MD (ex.) velocityverlet (default) nosehoover')
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    parser.add_argument('-cmds','--cmds', help="apply classical multidimensional scaling to calculated approx. reaction path.", action='store_true')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is GFN2-xTB (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-sqm1", "--sqm1", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument("-sqm2", "--sqm2", action='store_true', help='use experimental semiempirical method based on GFN0-xTB to calculate. default is not using semiempirical method.')
    parser.add_argument("-ct", "--change_temperature",  type=str, nargs="*", default=[], help='change temperature of thermostat (defalut) No change (ex.) [1000(time), 500(K) 5000(time), 1000(K)...]')
    parser.add_argument("-cc", "--constraint_condition", nargs="*", type=str, default=[], help="apply constraint conditions for optimazation (ex.) [[(dinstance (ang.)), (atom1),(atom2)] [(bond_angle (deg.)), (atom1),(atom2),(atom3)] [(dihedral_angle (deg.)), (atom1),(atom2),(atom3),(atom4)] ...] ")
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument("-osp", "--software_path_file",  type=str, default="./software_path.conf", help='read the list of file directory of other QM softwares to use them. default is current directory. (require python module, ASE (Atomic Simulation Environment)) (ex.) ./software_path.conf')
    parser.add_argument("-pbc", "--periodic_boundary_condition",  type=str, default=[], help='apply periodic boundary condition (Default is not applying.) (ex.) [periodic boundary (x,y,z) (ang.)] ')
    parser.add_argument("-pc", "--projection_constrain", nargs="*",  type=str, default=[], help='apply constrain conditions with projection of gradient and hessian (ex.) [[(constraint condition name) (atoms(ex. 1,2))] ...] ')
    parser.add_argument("-cpcm", "--cpcm_solv_model",  type=str, default=None, help='use CPCM solvent model for xTB (Defalut setting is not using this model.) (ex.) water')
    parser.add_argument("-alpb", "--alpb_solv_model",  type=str, default=None, help='use ALPB solvent model for xTB (Defalut setting is not using this model.) (ex.) water')#ref.: J. Chem. Theory Comput. 2021, 17, 7, 4250–4261 https://doi.org/10.1021/acs.jctc.1c00471
    parser.add_argument('-pca','--pca', help="Apply principal component analysis to calculated approx. reaction path.", action='store_true')
    parser.add_argument("-grid", "--dft_grid", type=int, default=3, help="fineness of grid for DFT calculation (default: 3 (0~9))")

    
    
    return parser


def init_parser():
    parser = argparse.ArgumentParser()
    return parser

def parser2args(parser, args_list=None):
    """
    Parses arguments and returns the args namespace.
    
    If args_list is None, it parses from sys.argv[1:] (command line).
    If args_list is provided (e.g., []), it parses from that list.
    """
    if args_list is None:
        # Original behavior: parse from command line
        args = parser.parse_args()
    else:
        # New behavior: parse from the provided list
        args = parser.parse_args(args_list)
    return args

def force_data_parser(args):
    def num_parse(numbers):
        sub_list = []
        
        sub_tmp_list = numbers.split(",")
        for sub in sub_tmp_list:                        
            if "-" in sub:
                for j in range(int(sub.split("-")[0]),int(sub.split("-")[1])+1):
                    sub_list.append(j)
            else:
                sub_list.append(int(sub))    
        return sub_list
    force_data = {}
    #---------------------
    force_data["oniom_flag"] = []
    
    if len(args.oniom_flag) == 3:
        high_layer = num_parse(args.oniom_flag[0])
        if str(args.oniom_flag[1]).lower() == "none":
            link_atoms = []
        else:
            link_atoms = num_parse(args.oniom_flag[1])
        low_layer_model = args.oniom_flag[2]
        force_data["oniom_flag"] = [high_layer, link_atoms, low_layer_model]
        
        
    elif len(args.oniom_flag) == 0:
        force_data["oniom_flag"] = []
    else:
        print("invaild input (-oniom) ")
        sys.exit(0)
        
    #---------------------
    force_data["nano_reactor_potential"] = []
    if len(args.nano_reactor_potential) % 6 != 0:
        print("invaild input (-nrp) ")
        sys.exit(0)
    
    ### add nano reactor potential (ex.) [[inner wall (ang.)] [outer wall (ang.)] [contraction time (ps)] [expansion time (ps)]]
    for i in range(int(len(args.nano_reactor_potential)/6)):
        force_data["nano_reactor_potential"].append([float(args.nano_reactor_potential[6*i]), float(args.nano_reactor_potential[6*i+1]), float(args.nano_reactor_potential[6*i+2]), float(args.nano_reactor_potential[6*i+3]), float(args.nano_reactor_potential[6*i+4]), float(args.nano_reactor_potential[6*i+5])])
    
    #---------------------
    force_data["projection_constraint_constant"] = []
    force_data["projection_constraint_condition_list"] = []
    force_data["projection_constraint_atoms"] = []
    if len(args.projection_constrain) > 0:
        if args.projection_constrain[0] == "manual":
            if len(args.projection_constrain) % 4 != 0:
                print("invaild input (-pc) ")
                sys.exit(0)
            
            tmp_val = args.projection_constrain
            for i in range(int(len(tmp_val)/4)):
                force_data["projection_constraint_condition_list"].append(str(tmp_val[4*i+1]))
                force_data["projection_constraint_atoms"].append(num_parse(tmp_val[4*i+2]))
                force_data["projection_constraint_constant"].append(float(tmp_val[4*i+3]))
            
        
        else:#auto
            if len(args.projection_constrain) % 2 != 0:
                print("invaild input (-pc) ")
                sys.exit(0)



            for i in range(int(len(args.projection_constrain)/2)):
                force_data["projection_constraint_condition_list"].append(str(args.projection_constrain[2*i]))
                force_data["projection_constraint_atoms"].append(num_parse(args.projection_constrain[2*i+1]))

    
    #---------------------
    force_data["asymmetric_ellipsoidal_repulsive_potential_v2_eps"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_v2_sig"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_v2_dist"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_v2_offtgt"] = []

    if len(args.asymmetric_ellipsoidal_repulsive_potential_v2) % 5 != 0:
        print("invaild input (-aerpv2) ")
        sys.exit(0)
    
    for i in range(int(len(args.asymmetric_ellipsoidal_repulsive_potential_v2)/5)):
        force_data["asymmetric_ellipsoidal_repulsive_potential_v2_eps"].append(float(args.asymmetric_ellipsoidal_repulsive_potential_v2[5*i]))
        force_data["asymmetric_ellipsoidal_repulsive_potential_v2_sig"].append([float(x) for x in args.asymmetric_ellipsoidal_repulsive_potential_v2[5*i+1].split(",")])
        force_data["asymmetric_ellipsoidal_repulsive_potential_v2_dist"].append(float(args.asymmetric_ellipsoidal_repulsive_potential_v2[5*i+2]))
        force_data["asymmetric_ellipsoidal_repulsive_potential_v2_atoms"].append(num_parse(args.asymmetric_ellipsoidal_repulsive_potential_v2[5*i+3]))
        force_data["asymmetric_ellipsoidal_repulsive_potential_v2_offtgt"].append(num_parse(args.asymmetric_ellipsoidal_repulsive_potential_v2[5*i+4]))    
    
    #---------------------
    force_data["asymmetric_ellipsoidal_repulsive_potential_eps"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_sig"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_dist"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_atoms"] = []
    force_data["asymmetric_ellipsoidal_repulsive_potential_offtgt"] = []

    if len(args.asymmetric_ellipsoidal_repulsive_potential) % 5 != 0:
        print("invaild input (-aerp) ")
        sys.exit(0)
    
    for i in range(int(len(args.asymmetric_ellipsoidal_repulsive_potential)/5)):
        force_data["asymmetric_ellipsoidal_repulsive_potential_eps"].append(float(args.asymmetric_ellipsoidal_repulsive_potential[5*i]))
        force_data["asymmetric_ellipsoidal_repulsive_potential_sig"].append([float(x) for x in args.asymmetric_ellipsoidal_repulsive_potential[5*i+1].split(",")])
        force_data["asymmetric_ellipsoidal_repulsive_potential_dist"].append(float(args.asymmetric_ellipsoidal_repulsive_potential[5*i+2]))
        force_data["asymmetric_ellipsoidal_repulsive_potential_atoms"].append(num_parse(args.asymmetric_ellipsoidal_repulsive_potential[5*i+3]))
        force_data["asymmetric_ellipsoidal_repulsive_potential_offtgt"].append(num_parse(args.asymmetric_ellipsoidal_repulsive_potential[5*i+4]))    
    
    #---------------------
    force_data["linear_mechano_force_v2"] = []
    force_data["linear_mechano_force_atom_v2"] = []
    

    if len(args.linear_mechano_force_pot_v2) % 2 != 0:
        print("invaild input (-lmefpv2)")
        sys.exit(0)
    
    for i in range(int(len(args.linear_mechano_force_pot_v2)/2)):
        force_data["linear_mechano_force_v2"].append(float(args.linear_mechano_force_pot_v2[2*i]))
        force_data["linear_mechano_force_atom_v2"].append(num_parse(args.linear_mechano_force_pot_v2[2*i+1]))
       

   
    #---------------------
    force_data["linear_mechano_force"] = []
    force_data["linear_mechano_force_atoms_1"] = []
    force_data["linear_mechano_force_atoms_2"] = []

    if len(args.linear_mechano_force_pot) % 3 != 0:
        print("invaild input (-lmefp)")
        sys.exit(0)
    
    for i in range(int(len(args.linear_mechano_force_pot)/3)):
        force_data["linear_mechano_force"].append(float(args.linear_mechano_force_pot[3*i]))
        force_data["linear_mechano_force_atoms_1"].append(num_parse(args.linear_mechano_force_pot[3*i+1]))
        force_data["linear_mechano_force_atoms_2"].append(num_parse(args.linear_mechano_force_pot[3*i+2]))


    #---------------------
    force_data["value_range_upper_const"] = []
    force_data["value_range_lower_const"] = []
    force_data["value_range_upper_distance"] = []
    force_data["value_range_lower_distance"] = []
    force_data["value_range_fragm_1"] = []
    force_data["value_range_fragm_2"] = []

    if len(args.bond_range_potential) % 6 != 0: 
        print("invaild input (-brp)")
        sys.exit(0)
    
    for i in range(int(len(args.bond_range_potential)/6)):
        force_data["value_range_upper_const"].append(float(args.bond_range_potential[6*i]))
        force_data["value_range_lower_const"].append(float(args.bond_range_potential[6*i+1]))
        force_data["value_range_upper_distance"].append(float(args.bond_range_potential[6*i+2]))
        force_data["value_range_lower_distance"].append(float(args.bond_range_potential[6*i+3]))
        force_data["value_range_fragm_1"].append(num_parse(args.bond_range_potential[6*i+4]))
        force_data["value_range_fragm_2"].append(num_parse(args.bond_range_potential[6*i+5]))
    
    
    
    #-----------------------
    if len(args.flux_potential) % 4 != 0:
        print("invaild input (-fp)")
        sys.exit(0)
    
    force_data["flux_pot_const"] = []
    force_data["flux_pot_order"] = []
    force_data["flux_pot_direction"] = []
    force_data["flux_pot_target"] = []
    for i in range(int(len(args.flux_potential)/4)):
        tmp = args.flux_potential[4*i].split(",")
        if len(tmp) != 3:
            print("invaild input (-fp)")
            sys.exit(0)
        sc_list = np.array([tmp[0], tmp[1], tmp[2]], dtype="float64")
        force_data["flux_pot_const"].append(sc_list)
        
        tmp = args.flux_potential[4*i+1].split(",")
        if len(tmp) != 3:
            print("invaild input (-fp)")
            sys.exit(0)
        order_list = np.array([tmp[0], tmp[1], tmp[2]], dtype="float64")
        force_data["flux_pot_order"].append(order_list)
        
        tmp = args.flux_potential[4*i+2].split(",")
        if len(tmp) != 3:
            print("invaild input (-fp)")
            sys.exit(0)
        direction = np.array([tmp[0], tmp[1], tmp[2]], dtype="float64")
        force_data["flux_pot_direction"].append(direction)
        
        force_data["flux_pot_target"].append(num_parse(args.flux_potential[4*i+3]))
        
    
    #---------------------
    if len(args.universal_potential) % 2 != 0:
        print("invaild input (-up)")
        sys.exit(0)
    force_data["universal_pot_const"] = []
    force_data["universal_pot_target"] = []
    for i in range(int(len(args.universal_potential)/2)):
        force_data["universal_pot_const"].append(float(args.universal_potential[2*i]))
        force_data["universal_pot_target"].append(num_parse(args.universal_potential[2*i+1]))
        if len(force_data["universal_pot_target"][i]) < 2:
            print("more than one atom for universal_pot_target! exit...")
            sys.exit(0)
    #---------------------
    if len(args.spacer_model_potential) % 5 != 0:
        print("invaild input (-smp)")
        sys.exit(0)
    force_data["spacer_model_potential_target"] = []
    force_data["spacer_model_potential_distance"] = []
    force_data["spacer_model_potential_well_depth"] = []
    force_data["spacer_model_potential_particle_number"] = [] #ang.
    force_data["spacer_model_potential_cavity_scaling"] = []
    
    for i in range(int(len(args.spacer_model_potential)/5)):
        force_data["spacer_model_potential_well_depth"].append(float(args.spacer_model_potential[5*i]))
        force_data["spacer_model_potential_distance"].append(float(args.spacer_model_potential[5*i+1]))
        force_data["spacer_model_potential_cavity_scaling"].append(float(args.spacer_model_potential[5*i+2]))
        force_data["spacer_model_potential_particle_number"].append(int(args.spacer_model_potential[5*i+3]))
        force_data["spacer_model_potential_target"].append(num_parse(args.spacer_model_potential[5*i+4]))

  
    #---------------------
    if len(args.repulsive_potential) % 5 != 0:
        print("invaild input (-rp)")
        sys.exit(0)
    
    force_data["repulsive_potential_well_scale"] = []
    force_data["repulsive_potential_dist_scale"] = []
    force_data["repulsive_potential_Fragm_1"] = []
    force_data["repulsive_potential_Fragm_2"] = []
    force_data["repulsive_potential_unit"] = []
    
    for i in range(int(len(args.repulsive_potential)/5)):
        force_data["repulsive_potential_well_scale"].append(float(args.repulsive_potential[5*i]))
        force_data["repulsive_potential_dist_scale"].append(float(args.repulsive_potential[5*i+1]))
        force_data["repulsive_potential_Fragm_1"].append(num_parse(args.repulsive_potential[5*i+2]))
        force_data["repulsive_potential_Fragm_2"].append(num_parse(args.repulsive_potential[5*i+3]))
        force_data["repulsive_potential_unit"].append(str(args.repulsive_potential[5*i+4]))
    
 
    #---------------------
    if len(args.repulsive_potential_v2) % 10 != 0:
        print("invaild input (-rpv2)")
        sys.exit(0)
    
    force_data["repulsive_potential_v2_well_scale"] = []
    force_data["repulsive_potential_v2_dist_scale"] = []
    force_data["repulsive_potential_v2_length"] = []
    force_data["repulsive_potential_v2_const_rep"] = []
    force_data["repulsive_potential_v2_const_attr"] = []
    force_data["repulsive_potential_v2_order_rep"] = []
    force_data["repulsive_potential_v2_order_attr"] = []
    force_data["repulsive_potential_v2_center"] = []
    force_data["repulsive_potential_v2_target"] = []
    force_data["repulsive_potential_v2_unit"] = []
    
    for i in range(int(len(args.repulsive_potential_v2)/10)):
        force_data["repulsive_potential_v2_well_scale"].append(float(args.repulsive_potential_v2[10*i+0]))
        force_data["repulsive_potential_v2_dist_scale"].append(float(args.repulsive_potential_v2[10*i+1]))
        force_data["repulsive_potential_v2_length"].append(float(args.repulsive_potential_v2[10*i+2]))
        force_data["repulsive_potential_v2_const_rep"].append(float(args.repulsive_potential_v2[10*i+3]))
        force_data["repulsive_potential_v2_const_attr"].append(float(args.repulsive_potential_v2[10*i+4]))
        force_data["repulsive_potential_v2_order_rep"].append(float(args.repulsive_potential_v2[10*i+5]))
        force_data["repulsive_potential_v2_order_attr"].append(float(args.repulsive_potential_v2[10*i+6]))
        force_data["repulsive_potential_v2_center"].append(num_parse(args.repulsive_potential_v2[10*i+7]))
        force_data["repulsive_potential_v2_target"].append(num_parse(args.repulsive_potential_v2[10*i+8]))
        force_data["repulsive_potential_v2_unit"].append(str(args.repulsive_potential_v2[10*i+9]))
        if len(force_data["repulsive_potential_v2_center"][i]) != 2:
            print("invaild input (-rpv2 center)")
            sys.exit(0)

    #---------------------
    if len(args.repulsive_potential_gaussian) % 7 != 0:
        print("invaild input (-rpg)")
        sys.exit(0)

    force_data["repulsive_potential_gaussian_LJ_well_depth"] = []
    force_data["repulsive_potential_gaussian_LJ_dist"] = []
    force_data["repulsive_potential_gaussian_gau_well_depth"] = []
    force_data["repulsive_potential_gaussian_gau_dist"] = []
    force_data["repulsive_potential_gaussian_gau_range"] = []
    force_data["repulsive_potential_gaussian_fragm_1"] = []
    force_data["repulsive_potential_gaussian_fragm_2"] = []

    
    for i in range(int(len(args.repulsive_potential_gaussian)/7)):
        force_data["repulsive_potential_gaussian_LJ_well_depth"].append(float(args.repulsive_potential_gaussian[7*i+0]))
        force_data["repulsive_potential_gaussian_LJ_dist"].append(float(args.repulsive_potential_gaussian[7*i+1]))
        force_data["repulsive_potential_gaussian_gau_well_depth"].append(float(args.repulsive_potential_gaussian[7*i+2]))
        force_data["repulsive_potential_gaussian_gau_dist"].append(float(args.repulsive_potential_gaussian[7*i+3]))
        force_data["repulsive_potential_gaussian_gau_range"].append(float(args.repulsive_potential_gaussian[7*i+4]))
        force_data["repulsive_potential_gaussian_fragm_1"].append(num_parse(args.repulsive_potential_gaussian[7*i+5]))
        force_data["repulsive_potential_gaussian_fragm_2"].append(num_parse(args.repulsive_potential_gaussian[7*i+6]))
       

    #---------------------    
    if len(args.cone_potential) % 6 != 0:
        print("invaild input (-cp)")
        sys.exit(0)
    
    force_data["cone_potential_well_value"] = []
    force_data["cone_potential_dist_value"] = []
    force_data["cone_potential_cone_angle"] = []
    force_data["cone_potential_center"] = []
    force_data["cone_potential_three_atoms"] = []
    force_data["cone_potential_target"] = []

    for i in range(int(len(args.cone_potential)/6)):
        force_data["cone_potential_well_value"].append(float(args.cone_potential[6*i+0]))
        force_data["cone_potential_dist_value"].append(float(args.cone_potential[6*i+1]))
        force_data["cone_potential_cone_angle"].append(float(args.cone_potential[6*i+2]))
        force_data["cone_potential_center"].append(int(args.cone_potential[6*i+3]))
        force_data["cone_potential_three_atoms"].append(num_parse(args.cone_potential[6*i+4]))
        force_data["cone_potential_target"].append(num_parse(args.cone_potential[6*i+5]))

        if len(force_data["cone_potential_three_atoms"][i]) != 3:
            print("invaild input (-cp three atoms)")
            sys.exit(0)               
                            
    
    #--------------------
    if len(args.manual_AFIR) % 3 != 0:
        print("invaild input (-ma)")
        sys.exit(0)
    
    force_data["AFIR_gamma"] = []
    force_data["AFIR_Fragm_1"] = []
    force_data["AFIR_Fragm_2"] = []
    

    for i in range(int(len(args.manual_AFIR)/3)):
        force_data["AFIR_gamma"].append(list(map(float, args.manual_AFIR[3*i].split(","))))#kj/mol
        force_data["AFIR_Fragm_1"].append(num_parse(args.manual_AFIR[3*i+1]))
        force_data["AFIR_Fragm_2"].append(num_parse(args.manual_AFIR[3*i+2]))
    
    
    #---------------------
    if len(args.anharmonic_keep_pot) % 4 != 0:
        print("invaild input (-akp)")
        sys.exit(0)
    
    force_data["anharmonic_keep_pot_potential_well_depth"] = []
    force_data["anharmonic_keep_pot_spring_const"] = []
    force_data["anharmonic_keep_pot_distance"] = []
    force_data["anharmonic_keep_pot_atom_pairs"] = []
    
    for i in range(int(len(args.anharmonic_keep_pot)/4)):
        force_data["anharmonic_keep_pot_potential_well_depth"].append(float(args.anharmonic_keep_pot[4*i]))#au
        force_data["anharmonic_keep_pot_spring_const"].append(float(args.anharmonic_keep_pot[4*i+1]))#au
        force_data["anharmonic_keep_pot_distance"].append(float(args.anharmonic_keep_pot[4*i+2]))#ang
        force_data["anharmonic_keep_pot_atom_pairs"].append(num_parse(args.anharmonic_keep_pot[4*i+3]))
        if len(force_data["anharmonic_keep_pot_atom_pairs"][i]) != 2:
            print("invaild input (-akp atom_pairs)")
            sys.exit(0)
        
    #---------------------
    if len(args.keep_pot) % 3 != 0:
        print("invaild input (-kp)")
        sys.exit(0)
    
    force_data["keep_pot_spring_const"] = []
    force_data["keep_pot_distance"] = []
    force_data["keep_pot_atom_pairs"] = []
    
    for i in range(int(len(args.keep_pot)/3)):
        force_data["keep_pot_spring_const"].append(float(args.keep_pot[3*i]))#au
        force_data["keep_pot_distance"].append(float(args.keep_pot[3*i+1]))#ang
        force_data["keep_pot_atom_pairs"].append(num_parse(args.keep_pot[3*i+2]))
        if len(force_data["keep_pot_atom_pairs"][i]) != 2:
            print("invaild input (-kp atom_pairs)")
            sys.exit(0)
        
    #---------------------
    if len(args.keep_pot_v2) % 4 != 0:
        print("invaild input (-kpv2)")
        sys.exit(0)
    
    force_data["keep_pot_v2_spring_const"] = []
    force_data["keep_pot_v2_distance"] = []
    force_data["keep_pot_v2_fragm1"] = []
    force_data["keep_pot_v2_fragm2"] = []
    
    for i in range(int(len(args.keep_pot_v2)/4)):
        force_data["keep_pot_v2_spring_const"].append(list(map(float, args.keep_pot_v2[4*i].split(","))))#au
        force_data["keep_pot_v2_distance"].append(list(map(float, args.keep_pot_v2[4*i+1].split(","))))#ang
        force_data["keep_pot_v2_fragm1"].append(num_parse(args.keep_pot_v2[4*i+2]))
        force_data["keep_pot_v2_fragm2"].append(num_parse(args.keep_pot_v2[4*i+3]))
        
        
    #---------------------
    if len(args.keep_angle) % 3 != 0:
        print("invaild input (-ka)")
        sys.exit(0)
    
    force_data["keep_angle_spring_const"] = []
    force_data["keep_angle_angle"] = []
    force_data["keep_angle_atom_pairs"] = []
    
    for i in range(int(len(args.keep_angle)/3)):
        force_data["keep_angle_spring_const"].append(float(args.keep_angle[3*i]))#au
        force_data["keep_angle_angle"].append(float(args.keep_angle[3*i+1]))#degrees
        force_data["keep_angle_atom_pairs"].append(num_parse(args.keep_angle[3*i+2]))
        if len(force_data["keep_angle_atom_pairs"][i]) != 3:
            print("invaild input (-ka atom_pairs)")
            sys.exit(0)
    

            
    #---------------------
    if len(args.keep_angle_v2) % 5 != 0:
        print("invaild input (-kav2)")
        sys.exit(0)
    
    force_data["keep_angle_v2_spring_const"] = []
    force_data["keep_angle_v2_angle"] = []
    force_data["keep_angle_v2_fragm1"] = []
    force_data["keep_angle_v2_fragm2"] = []
    force_data["keep_angle_v2_fragm3"] = []
    
    for i in range(int(len(args.keep_angle_v2)/5)):
        force_data["keep_angle_v2_spring_const"].append(list(map(float, args.keep_angle_v2[5*i].split(","))))#au
        force_data["keep_angle_v2_angle"].append(list(map(float, args.keep_angle_v2[5*i+1].split(","))))#degrees
        force_data["keep_angle_v2_fragm1"].append(num_parse(args.keep_angle_v2[5*i+2]))
        force_data["keep_angle_v2_fragm2"].append(num_parse(args.keep_angle_v2[5*i+3]))
        force_data["keep_angle_v2_fragm3"].append(num_parse(args.keep_angle_v2[5*i+4]))
       
    #---------------------
    
    if len(args.keep_dihedral_angle) % 3 != 0:
        print("invaild input (-kda)")
        sys.exit(0)
        
    force_data["keep_dihedral_angle_spring_const"] = []
    force_data["keep_dihedral_angle_angle"] = []
    force_data["keep_dihedral_angle_atom_pairs"] = []
    
    for i in range(int(len(args.keep_dihedral_angle)/3)):
        force_data["keep_dihedral_angle_spring_const"].append(float(args.keep_dihedral_angle[3*i]))#au
        force_data["keep_dihedral_angle_angle"].append(float(args.keep_dihedral_angle[3*i+1]))#degrees
        force_data["keep_dihedral_angle_atom_pairs"].append(num_parse(args.keep_dihedral_angle[3*i+2]))
        if len(force_data["keep_dihedral_angle_atom_pairs"][i]) != 4:
            print("invaild input (-kda atom_pairs)")
            sys.exit(0)
    
    #---------------------
    
    if len(args.keep_out_of_plain_angle) % 3 != 0:
        print("invaild input (-kopa)")
        sys.exit(0)
        
    force_data["keep_out_of_plain_angle_spring_const"] = []
    force_data["keep_out_of_plain_angle_angle"] = []
    force_data["keep_out_of_plain_angle_atom_pairs"] = []
    
    for i in range(int(len(args.keep_out_of_plain_angle)/3)):
        force_data["keep_out_of_plain_angle_spring_const"].append(float(args.keep_out_of_plain_angle[3*i]))#au
        force_data["keep_out_of_plain_angle_angle"].append(float(args.keep_out_of_plain_angle[3*i+1]))#degrees
        force_data["keep_out_of_plain_angle_atom_pairs"].append(num_parse(args.keep_out_of_plain_angle[3*i+2]))
        if len(force_data["keep_out_of_plain_angle_atom_pairs"][i]) != 4:
            print("invaild input (-kopa atom_pairs)")
            sys.exit(0)
    
    #---------------------
    
    if len(args.keep_out_of_plain_angle_v2) % 6 != 0:
        print("invaild input (-kopav2)")
        sys.exit(0)
        
    force_data["keep_out_of_plain_angle_v2_spring_const"] = []
    force_data["keep_out_of_plain_angle_v2_angle"] = []
    force_data["keep_out_of_plain_angle_v2_fragm1"] = []
    force_data["keep_out_of_plain_angle_v2_fragm2"] = []
    force_data["keep_out_of_plain_angle_v2_fragm3"] = []
    force_data["keep_out_of_plain_angle_v2_fragm4"] = []
    
    for i in range(int(len(args.keep_out_of_plain_angle_v2)/6)):
        force_data["keep_out_of_plain_angle_v2_spring_const"].append(list(map(float, args.keep_out_of_plain_angle_v2[6*i].split(","))))#au
        force_data["keep_out_of_plain_angle_v2_angle"].append(list(map(float, args.keep_out_of_plain_angle_v2[6*i+1].split(","))))#degrees
        force_data["keep_out_of_plain_angle_v2_fragm1"].append(num_parse(args.keep_out_of_plain_angle_v2[6*i+2]))
        force_data["keep_out_of_plain_angle_v2_fragm2"].append(num_parse(args.keep_out_of_plain_angle_v2[6*i+3]))
        force_data["keep_out_of_plain_angle_v2_fragm3"].append(num_parse(args.keep_out_of_plain_angle_v2[6*i+4]))
        force_data["keep_out_of_plain_angle_v2_fragm4"].append(num_parse(args.keep_out_of_plain_angle_v2[6*i+5]))

        
    #---------------------
    
    if len(args.keep_dihedral_angle_v2) % 6 != 0:
        print("invaild input (-kdav2)")
        sys.exit(0)
        
    force_data["keep_dihedral_angle_v2_spring_const"] = []
    force_data["keep_dihedral_angle_v2_angle"] = []
    force_data["keep_dihedral_angle_v2_fragm1"] = []
    force_data["keep_dihedral_angle_v2_fragm2"] = []
    force_data["keep_dihedral_angle_v2_fragm3"] = []
    force_data["keep_dihedral_angle_v2_fragm4"] = []
    
    for i in range(int(len(args.keep_dihedral_angle_v2)/6)):
        force_data["keep_dihedral_angle_v2_spring_const"].append(list(map(float, args.keep_dihedral_angle_v2[6*i].split(","))))#au
        force_data["keep_dihedral_angle_v2_angle"].append(list(map(float, args.keep_dihedral_angle_v2[6*i+1].split(","))))#degrees
        force_data["keep_dihedral_angle_v2_fragm1"].append(num_parse(args.keep_dihedral_angle_v2[6*i+2]))
        force_data["keep_dihedral_angle_v2_fragm2"].append(num_parse(args.keep_dihedral_angle_v2[6*i+3]))
        force_data["keep_dihedral_angle_v2_fragm3"].append(num_parse(args.keep_dihedral_angle_v2[6*i+4]))
        force_data["keep_dihedral_angle_v2_fragm4"].append(num_parse(args.keep_dihedral_angle_v2[6*i+5]))
        
    #---------------------
    
    if len(args.keep_dihedral_angle_cos) % 7 != 0:
        print("invaild input (-kdac)")
        sys.exit(0)
        
    force_data["keep_dihedral_angle_cos_potential_const"] = []
    force_data["keep_dihedral_angle_cos_angle_const"] = []
    force_data["keep_dihedral_angle_cos_angle"] = []
    force_data["keep_dihedral_angle_cos_fragm1"] = []
    force_data["keep_dihedral_angle_cos_fragm2"] = []
    force_data["keep_dihedral_angle_cos_fragm3"] = []
    force_data["keep_dihedral_angle_cos_fragm4"] = []
    
    for i in range(int(len(args.keep_dihedral_angle_cos)/7)):
        force_data["keep_dihedral_angle_cos_potential_const"].append(list(map(float, args.keep_dihedral_angle_cos[7*i].split(","))))#au
        force_data["keep_dihedral_angle_cos_angle_const"].append(list(map(float, args.keep_dihedral_angle_cos[7*i+1].split(","))))#degrees
        force_data["keep_dihedral_angle_cos_angle"].append(list(map(float, args.keep_dihedral_angle_cos[7*i+2].split(","))))#degrees
        force_data["keep_dihedral_angle_cos_fragm1"].append(num_parse(args.keep_dihedral_angle_cos[7*i+3]))
        force_data["keep_dihedral_angle_cos_fragm2"].append(num_parse(args.keep_dihedral_angle_cos[7*i+4]))
        force_data["keep_dihedral_angle_cos_fragm3"].append(num_parse(args.keep_dihedral_angle_cos[7*i+5]))
        force_data["keep_dihedral_angle_cos_fragm4"].append(num_parse(args.keep_dihedral_angle_cos[7*i+6]))
    
    
    #---------------------
    if len(args.well_pot) % 4 != 0:
        print("invaild input (-wp)")
        sys.exit(0)
        
    force_data["well_pot_wall_energy"] = []
    force_data["well_pot_fragm_1"] = []
    force_data["well_pot_fragm_2"] = []
    force_data["well_pot_limit_dist"] = []
    
    for i in range(int(len(args.well_pot)/4)):
        force_data["well_pot_wall_energy"].append(float(args.well_pot[4*i]))#kJ/mol
        force_data["well_pot_fragm_1"].append(num_parse(args.well_pot[4*i+1]))
        force_data["well_pot_fragm_2"].append(num_parse(args.well_pot[4*i+2]))
        force_data["well_pot_limit_dist"].append(args.well_pot[4*i+3].split(","))#ang
        if float(force_data["well_pot_limit_dist"][i][0]) < float(force_data["well_pot_limit_dist"][i][1]) and float(force_data["well_pot_limit_dist"][i][1]) < float(force_data["well_pot_limit_dist"][i][2]) and float(force_data["well_pot_limit_dist"][i][2]) < float(force_data["well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-wp a<b<c<d)")
            sys.exit(0)
            
    #---------------------
    if len(args.wall_well_pot) % 4 != 0:
        print("invaild input (-wwp)")
        sys.exit(0)
        
    force_data["wall_well_pot_wall_energy"] = []
    force_data["wall_well_pot_direction"] = []
    force_data["wall_well_pot_limit_dist"] = []
    force_data["wall_well_pot_target"] = []
    
    for i in range(int(len(args.wall_well_pot)/4)):
        force_data["wall_well_pot_wall_energy"].append(float(args.wall_well_pot[4*i]))#kJ/mol
        force_data["wall_well_pot_direction"].append(args.wall_well_pot[4*i+1])
        
        if force_data["wall_well_pot_direction"][i] == "x" or force_data["wall_well_pot_direction"][i] == "y" or force_data["wall_well_pot_direction"][i] == "z":
            pass
        else:
            print("invaild input (-wwp direction)")
            sys.exit(0)
        
        force_data["wall_well_pot_limit_dist"].append(args.wall_well_pot[4*i+2].split(","))#ang
        if float(force_data["wall_well_pot_limit_dist"][i][0]) < float(force_data["wall_well_pot_limit_dist"][i][1]) and float(force_data["wall_well_pot_limit_dist"][i][1]) < float(force_data["wall_well_pot_limit_dist"][i][2]) and float(force_data["wall_well_pot_limit_dist"][i][2]) < float(force_data["wall_well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-wwp a<b<c<d)")
            sys.exit(0)
        
        force_data["wall_well_pot_target"].append(num_parse(args.wall_well_pot[4*i+3]))
    #---------------------
    
    if len(args.void_point_well_pot) % 4 != 0:
        print("invaild input (-vpwp)")
        sys.exit(0)
        
    force_data["void_point_well_pot_wall_energy"] = []
    force_data["void_point_well_pot_coordinate"] = []
    force_data["void_point_well_pot_limit_dist"] = []
    force_data["void_point_well_pot_target"] = []
    
    for i in range(int(len(args.void_point_well_pot)/4)):
        force_data["void_point_well_pot_wall_energy"].append(float(args.void_point_well_pot[4*i]))#kJ/mol
        
        
        force_data["void_point_well_pot_coordinate"].append(list(map(float, args.void_point_well_pot[4*i+1].split(","))))
        
        if len(force_data["void_point_well_pot_coordinate"][i]) != 3:
            print("invaild input (-vpwp coordinate)")
            sys.exit(0)
        
        force_data["void_point_well_pot_limit_dist"].append(args.void_point_well_pot[4*i+2].split(","))#ang
        if float(force_data["void_point_well_pot_limit_dist"][i][0]) < float(force_data["void_point_well_pot_limit_dist"][i][1]) and float(force_data["void_point_well_pot_limit_dist"][i][1]) < float(force_data["void_point_well_pot_limit_dist"][i][2]) and float(force_data["void_point_well_pot_limit_dist"][i][2]) < float(force_data["void_point_well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-vpwp a<b<c<d)")
            sys.exit(0)
            
        force_data["void_point_well_pot_target"].append(num_parse(args.void_point_well_pot[4*i+3]))
        
    #---------------------
    
    if len(args.around_well_pot) % 4 != 0:
        print("invaild input (-awp)")
        sys.exit(0)
        
    force_data["around_well_pot_wall_energy"] = []
    force_data["around_well_pot_center"] = []
    force_data["around_well_pot_limit_dist"] = []
    force_data["around_well_pot_target"] = []
    
    for i in range(int(len(args.around_well_pot)/4)):
        force_data["around_well_pot_wall_energy"].append(float(args.around_well_pot[4*i]))#kJ/mol
        
        
        force_data["around_well_pot_center"].append(num_parse(args.around_well_pot[4*i+1]))
        
        
        force_data["around_well_pot_limit_dist"].append(args.around_well_pot[4*i+2].split(","))#ang
        if float(force_data["around_well_pot_limit_dist"][i][0]) < float(force_data["around_well_pot_limit_dist"][i][1]) and float(force_data["around_well_pot_limit_dist"][i][1]) < float(force_data["around_well_pot_limit_dist"][i][2]) and float(force_data["around_well_pot_limit_dist"][i][2]) < float(force_data["around_well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-vpwp a<b<c<d)")
            sys.exit(0)
            
        force_data["around_well_pot_target"].append(num_parse(args.around_well_pot[4*i+3]))
        
    #---------------------
    
    if len(args.void_point_pot) % 5 != 0:
        print("invaild input (-vpp)")
        sys.exit(0)
    
    force_data["void_point_pot_spring_const"] = []
    force_data["void_point_pot_distance"] = []
    force_data["void_point_pot_coord"] = []
    force_data["void_point_pot_atoms"] = []
    force_data["void_point_pot_order"] = []
    
    for i in range(int(len(args.void_point_pot)/5)):
        force_data["void_point_pot_spring_const"].append(float(args.void_point_pot[5*i]))#au
        force_data["void_point_pot_distance"].append(float(args.void_point_pot[5*i+1]))#ang
        coord = args.void_point_pot[5*i+2].split(",")
        force_data["void_point_pot_coord"].append(list(map(float, coord)))#ang
        force_data["void_point_pot_atoms"].append(num_parse(args.void_point_pot[5*i+3]))
        force_data["void_point_pot_order"].append(float(args.void_point_pot[5*i+4]))
    #---------------------
    force_data["gaussian_potential_target"] = []
    force_data["gaussian_potential_height"] = []
    force_data["gaussian_potential_width"] = []
    force_data["gaussian_potential_tgt_atom"] = []

    if len(args.metadynamics) > 0:
        for i in range(int(len(args.metadynamics)/4)):
            force_data["gaussian_potential_target"].append(str(args.metadynamics[4*i]))
            force_data["gaussian_potential_height"].append(float(args.metadynamics[4*i+1]))
            force_data["gaussian_potential_width"].append(float(args.metadynamics[4*i+2]))
            force_data["gaussian_potential_tgt_atom"].append(num_parse(args.metadynamics[4*i+3]))
        
    if len(args.fix_atoms) > 0:
        force_data["fix_atoms"] = num_parse(args.fix_atoms[0])
    else:
        force_data["fix_atoms"] = ""
    

    force_data["geom_info"] = num_parse(args.geom_info[0])
    
    force_data["opt_method"] = args.opt_method
    
    force_data["xtb"] = args.usextb
    force_data["opt_fragment"] = [num_parse(args.opt_fragment[i]) for i in range(len(args.opt_fragment))]

    
    return force_data
