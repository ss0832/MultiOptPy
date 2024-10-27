import argparse
import sys
import numpy as np


"""
    BiasPotPy
    Copyright (C) 2023-2024 ss0832

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

"""
#please input psi4 inputfile.
XMOL format (Enter the formal charge and spin multiplicity on the comment line, e.g., "0 1")
....
"""

"""
references:

Psi4
 D. G. A. Smith, L. A. Burns, A. C. Simmonett, R. M. Parrish, M. C. Schieber, R. Galvelis, P. Kraus, H. Kruse, R. Di Remigio, A. Alenaizan, A. M. James, S. Lehtola, J. P. Misiewicz, M. Scheurer, R. A. Shaw, J. B. Schriber, Y. Xie, Z. L. Glick, D. A. Sirianni, J. S. O'Brien, J. M. Waldrop, A. Kumar, E. G. Hohenstein, B. P. Pritchard, B. R. Brooks, H. F. Schaefer III, A. Yu. Sokolov, K. Patkowski, A. E. DePrince III, U. Bozkaya, R. A. King, F. A. Evangelista, J. M. Turney, T. D. Crawford, C. D. Sherrill, "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).
 
PySCF
Recent developments in the PySCF program package, Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan, J. Chem. Phys., 153, 024109 (2020). doi:10.1063/5.0006074

GFN2-xTB(tblite)
J. Chem. Theory Comput. 2019, 15, 3, 1652–1671 
GFN1-xTB(tblite, dxtb)
J. Chem. Theory Comput. 2017, 13, 5, 1989–2009
"""


def init_parser():
    parser = argparse.ArgumentParser()
    return parser


def ieipparser(parser):
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-ns", "--NSTEP",  type=int, default='999', help='iter. number')
    
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["FIRELARS"], help='optimization method for QM calclation (default: FIRE) (mehod_list:(steepest descent method group) FIRE, CG etc. (quasi-Newton method group) RFO_FSB, RFO_BFGS, RFO3_Bifill  etc.) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default="", help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mi", "--microiter",  type=int, default=0, help='microiteration for relaxing reaction pathways')
    parser.add_argument("-beta", "--BETA",  type=float, default='1.0', help='force for optimization')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-es", "--excited_state", type=int, nargs="*", default=[0, 0],
                        help='calculate excited state (default: [0(initial state), 0(final state)]) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')
    parser.add_argument("-mf", "--model_function_mode", help="use model function to optimization", type=str, default='None',)
    parser = parser_for_biasforce(parser)
    
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-dxtb", "--usedxtb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd diffential method is available.)) (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, nargs="*", default=[0, 0], help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, nargs="*", default=[1, 1], help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')


    
    args = parser.parse_args()#model_function_mode
    args.fix_atoms = []
    args.gradient_fix_atoms = []
    args.geom_info = ["0"]
     
    args.opt_fragment = []
    args.oniom_method = []
    return args



def optimizeparser(parser):
    parser.add_argument("INPUT", help='input xyz file name', nargs="*")
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-es", "--excited_state", type=int, default=0, help='calculate excited state (default: 0) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')
    parser.add_argument("-ns", "--NSTEP",  type=int, default='300', help='iter. number')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-oniom", "--oniom_method", type=str, nargs="*", default=[], help='Use ONIOM2 (our own 2-layered integrated molecular orbital and molecular machine) method (ex.) [[atom label numbers for high layer (1,2)] [link atoms (3,4)] [calculation level for low layer (GFN2-xTB, GFN1-xTB, mace_mp, etc.)]] (ref.) Int. J. Quantum Chem., 60, 1101-1109 (1996).')
    parser = parser_for_biasforce(parser)
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default="", help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')
    parser.add_argument("-md", "--md_like_perturbation",  type=str, default="0.0", help='add perturbation like molecule dynamics (ex.) [[temperature (unit. K)]]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["FIRELARS"], help='optimization method for QM calclation (default: FIRE) (mehod_list:(steepest descent method group) FIRE, CG etc. (quasi-Newton method group) RFO_FSB, RFO_BFGS, RFO3_Bifill  etc.) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-mfc", "--calc_model_hess",  type=int, default=50, help='calculate model hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-dxtb", "--usedxtb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd diffential method is available.)) (ex.) GFN1-xTB, GFN2-xTB ')
    #parser.add_argument("-cpcm", "--cpcm_solv_model",  type=str, default=None, help='use CPCM solvent model for xTB (ex.) water')

    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    parser.add_argument('-cmds','--cmds', help="Apply classical multidimensional scaling to calculated approx. reaction path.", action='store_true')
    parser.add_argument('-pca','--pca', help="Apply principal component analysis to calculated approx. reaction path.", action='store_true')
    parser.add_argument('-irc','--intrinsic_reaction_coordinates', help="Calculate intrinsic reaction coordinates. (ex.) [[step_size], [max_step], [IRC_method]] (Recommended) [0.5 300 lqa]", nargs="*", type=str, default=[])    
    parser.add_argument('-rc','--ricci_curvature', help="calculate Ricci scalar of calculated approx. reaction path.", action='store_true')
    parser.add_argument("-of", "--opt_fragment", nargs="*", type=str, default=[], help="Several atoms are grouped together as fragments and optimized. (This method doesnot work if you use quasi-newton method for optimazation.) (ex.) [[atoms (ex.) 1-4] ...] ")#(2024/3/26) this option doesn't work if you use quasi-Newton method for optimization.
    parser.add_argument("-cc", "--constraint_condition", nargs="*", type=str, default=[], help="apply constraint conditions for optimazation (ex.) [[(dinstance (ang.)),(atom1),(atom2)] [(bond_angle (deg.)),(atom1),(atom2),(atom3)] [(dihedral_angle (deg.)),(atom1),(atom2),(atom3),(atom4)] ...] ")
    parser.add_argument("-nro", "--NRO_analysis",  help="apply Natural Reaction Orbial analysis. (ref. Phys. Chem. Chem. Phys. 24, 3532 (2022))", action='store_true')
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument('-tcc','--tight_convergence_criteria', help="apply tight opt criteria.", action='store_true')
    parser.add_argument('-lcc','--loose_convergence_criteria', help="apply loose opt criteria.", action='store_true')

    parser.add_argument('-modelhess','--use_model_hessian', help="use model hessian.", action='store_true')
    parser.add_argument("-sc", "--shape_conditions", nargs="*", type=str, default=[], help="Exit optimization if these conditions are not satisfied. (e.g.) [[(ang.) gt(lt) 2,3 (bond)] [(deg.) gt(lt) 2,3,4 (bend)] ...] [[(deg.) gt(lt) 2,3,4,5 (torsion)] ...]")
    parser.add_argument("-lc", "--lagrange_constrain", nargs="*",  type=str, default=[], help='apply constrain conditions with lagrange multiplier (ex.) [[(constraint condition name) (atoms(ex. 1,2))] ...] ')
    
    args = parser.parse_args()
    if len(args.INPUT) < 2:
        args.INPUT = args.INPUT[0]

    return args

def parser_for_biasforce(parser):
    parser.add_argument("-ma", "--manual_AFIR", nargs="*",  type=str, default=[], help='manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]')
    parser.add_argument("-rp", "--repulsive_potential", nargs="*",  type=str, default=[], help='Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpv2", "--repulsive_potential_v2", nargs="*",  type=str, default=[], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpg", "--repulsive_potential_gaussian", nargs="*",  type=str, default=[], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε_LJ[(σ/r)^(12) - 2 * (σ/r)^(6)] - ε_gau * exp(-((r-σ_gau)/b)^2)) (ex.) [[LJ_well_depth (kJ/mol)] [LJ_dist (ang.)] [Gaussian_well_depth (kJ/mol)] [Gaussian_dist (ang.)] [Gaussian_range (ang.)] [Fragm.1 (1,2)] [Fragm.2 (3-5,8)] ...]')
    
    parser.add_argument("-cp", "--cone_potential", nargs="*",  type=str, default=[], help='Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
    
    parser.add_argument("-fp", "--flux_potential", nargs="*",  type=str, default=[], help='Add potential to make flow. ( k/p*(x-x_0)^p )(ex.) [[x,y,z (constant (a.u.))] [x,y,z (order)] [x,y,z coordinate (ang.)] [Fragm.(ex. 1,2,3-5)] ...]')
    parser.add_argument("-kp", "--keep_pot", nargs="*",  type=str, default=[], help='keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-kpv2", "--keep_pot_v2", nargs="*",  type=str, default=[], help='keep potential_v2 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [Fragm.1] [Fragm.2] ...] ')
    parser.add_argument("-anikpv2", "--aniso_keep_pot_v2", nargs="*",  type=str, default=[], help='aniso keep potential_v2 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)(xx xy xz yx yy yz zx zy zz)] [keep distance (ang.)] [Fragm.1] [Fragm.2] ...] ')
    parser.add_argument("-akp", "--anharmonic_keep_pot", nargs="*",  type=str, default=[], help='Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-ka", "--keep_angle", nargs="*",  type=str, default=[], help='keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] ')
    parser.add_argument("-kav2", "--keep_angle_v2", nargs="*",  type=str, default=[], help='keep angle_v2 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] ...] ')
    parser.add_argument("-lpka", "--lone_pair_keep_angle", nargs="*",  type=str, default=[], help='lone pair keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [lone_pair_1 (center,atom1,atom2,atom3)] [lone_pair_2 (center,atom1,atom2,atom3)] ...] ')
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
    parser.add_argument("-aerp", "--asymmetric_ellipsoidal_repulsive_potential", nargs="*",  type=str, default=[], help='add asymmetric ellipsoidal repulsive potential (ex.) [[well_value (epsilon) (kJ/mol)] [dist_value (sigma) (a1,a2,b1,b2,c1,c2) (ang.)] [dist_value (distance) (ang.)] [target atom (1,2)] [off target atoms (3-5)] ...]')
    parser.add_argument("-aerpv2", "--asymmetric_ellipsoidal_repulsive_potential_v2", nargs="*",  type=str, default=[], help='add asymmetric ellipsoidal repulsive potential (ex.) [[well_value (epsilon) (kJ/mol)] [dist_value (sigma) (a1,a2,b1,b2,c1,c2) (ang.)] [dist_value (distance) (ang.)] [target atom (1,2)] [off target atoms (3-5)] ...]')
    return parser


def nebparser(parser):
    parser.add_argument("INPUT", help='input folder')
    
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-es", "--excited_state", type=int, default=0, help='calculate excited state (default: 0) (e.g.) if you set spin_multiplicity as 1 and set this option as "n", this program calculate S"n" state.')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='10', help='iter. number')
    parser.add_argument("-om", "--OM", action='store_true', help='J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action.')
    parser.add_argument("-lup", "--LUP", action='store_true', help='J. Chem. Phys. 92, 1510–1511 (1990) doi:https://doi.org/10.1063/1.458112 locally updated planes (LUP) method')
    parser.add_argument("-dneb", "--DNEB", action='store_true', help='J. Chem. Phys. 120, 2082–2094 (2004) doi:https://doi.org/10.1063/1.1636455 doubly NEB method (DNEB) method')
    parser.add_argument("-nesb", "--NESB", action='store_true', help='J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169 Nudged elastic stiffness band (NESB) method')
    parser.add_argument("-p", "--partition",  type=int, default='0', help='number of nodes')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-cineb", "--apply_CI_NEB",  type=int, default='99999', help='apply CI_NEB method')
    parser.add_argument("-sd", "--steepest_descent",  type=int, default='99999', help='apply steepest_descent method')
    parser.add_argument("-qnt", "--QUASI_NEWTOM_METHOD", action='store_true', help='changing optimizer to quasi-Newton method')
    parser.add_argument("-gqnt", "--GLOBAL_QUASI_NEWTOM_METHOD", action='store_true', help='changing optimizer to global-quasi-Newton method')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-fe", "--fixedges",  type=int, default=0, help='fix edges of nodes (1=initial_node, 2=end_node, 3=both_nodes) ')
    parser.add_argument("-aneb", "--ANEB_num",  type=int, default=0, help='execute adaptic NEB (ANEB) method. (default setting is not executing ANEB.)')
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default=[], help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default=[], help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')


    parser = parser_for_biasforce(parser)
    args = parser.parse_args()

    args.geom_info = ["0"]
    args.opt_method = ""
    args.opt_fragment = []
    args.oniom_method = []
    return args


def mdparser(parser):
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
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
    parser.add_argument("-ct", "--change_temperature",  type=str, nargs="*", default=[], help='change temperature of thermostat (defalut) No change (ex.) [1000(time), 500(K) 5000(time), 1000(K)...]')
    parser.add_argument("-cc", "--constraint_condition", nargs="*", type=str, default=[], help="apply constraint conditions for optimazation (ex.) [[(dinstance (ang.)), (atom1),(atom2)] [(bond_angle (deg.)), (atom1),(atom2),(atom3)] [(dihedral_angle (deg.)), (atom1),(atom2),(atom3),(atom4)] ...] ")
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument("-pbc", "--periodic_boundary_condition",  type=str, default=[], help='apply periodic boundary condition (Default is not applying.) (ex.) [periodic boundary (x,y,z) (ang.)] ')


    parser = parser_for_biasforce(parser)
    args = parser.parse_args()
    args.geom_info = ["0"]
    args.gradient_fix_atoms = []
    args.opt_method = ""
    args.opt_fragment = []
    args.oniom_method = []
    return args



class BiasPotInterface:
    def __init__(self):
        self.linear_mechano_force_pot_v2 = []
        self.linear_mechano_force_pot = []#['0.0', '1.0', '1,2,3,4']#add linear mechanochemical force (ex.) [[force(pN)] [atoms1(ex. 1,2)] [atoms2(ex. 3,4s)] ...]
        self.manual_AFIR = []#['0.0', '1', '2'] #manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]
        self.repulsive_potential = []#['0.0','1.0', '1', '2', 'scale'] #Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value (ang. kJ/mol)] ...]
        self.repulsive_potential_v2 = []#['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2', 'scale']#Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value (ang. kJ/mol)] ...]
        self.cone_potential = []#['0.0','1.0','90','1', '2,3,4', '5-9']#'Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
        
        self.keep_pot = []#['0.0', '1.0', '1,2']#keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...]
        self.keep_pot_v2 = []#['0.0', '1.0', '1','2']#keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...]
        self.aniso_keep_pot_v2 = []#['0.0','0.0','0.0','0.0','0.0','0.0','0.0','0.0','0.0', '1.0', '1', '2']
        
        self.universal_potential = []
        self.anharmonic_keep_pot = []#['0.0', '1.0', '1.0', '1,2']#Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...]
        self.keep_angle = []#['0.0', '90', '1,2,3']#keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...]
        self.keep_angle_v2 = []#['0.0', '90', '1','2','3']#keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...]
        self.atom_distance_dependent_keep_angle = []#['0.0', '90', "120", "1.4", "5", "1", '2,3,4']#'atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] '
        self.lone_pair_keep_angle = []#['0.0', '90', '1,2,3,4', '5,6,7,8']

        self.flux_potential = []#['0.0', '1.0', '1,2,3,4']#flux potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2,atom3,atom4] ...]
        self.keep_dihedral_angle = []#['0.0', '90', '1,2,3,4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...]
        self.keep_out_of_plain_angle = []#['0.0', '90', '1,2,3,4']#keep out_of_plain angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep out_of_plain angle (degrees)] [atom1,atom2,atom3,atom4] ...]
        self.keep_dihedral_angle_v2 = []#['0.0', '90', '1','2','3','4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...]
        self.keep_dihedral_angle_cos = []
        self.keep_out_of_plain_angle_v2 = []#['0.0', '90', '1','2','3','4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...]
        self.void_point_pot = []#['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"]#void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...]

        self.bond_range_potential = []
        self.well_pot = []#['0.0','1','2','0.5,0.6,1.5,1.6']
        self.wall_well_pot = []#['0.0','x','0.5,0.6,1.5,1.6', '1']#Add potential to limit atoms movement. (sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.void_point_well_pot = []#['0.0','0.0,0.0,0.0','0.5,0.6,1.5,1.6', '1']#"Add potential to limit atom movement. (sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.around_well_pot = []#['0.0','1','0.5,0.6,1.5,1.6',"2"] #Add potential to limit atom movement. (like sphere around 1 atom) (ex.) [[wall energy (kJ/mol)] [1 atom (1)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")
        self.spacer_model_potential = []#['0.0',"1.0",'1.0','5',"1,2"]
        self.asymmetric_ellipsoidal_repulsive_potential = []#['0.0','1.0,1.0,1.0,1.0,1.0,1.0', '2.0', '1,2','3-5']#add ovoid repulsive potential (ex.) [[well_value (epsilon) (kJ/mol)] [dist_value (sigma) (a1,a2,b1,b2,c1,c2) (ang.)] [dist_value (distance) (ang.)] [target atom (1,2)] [off target atoms (3-5)] ...]
        self.asymmetric_ellipsoidal_repulsive_potential_v2 = []#['0.0','1.0,1.0,1.0,1.0,1.0,1.0', '2.0', '1,2','3-5']#add ovoid repulsive potential (ex.) [[well_value (epsilon) (kJ/mol)] [dist_value (sigma) (a1,a2,b1,b2,c1,c2) (ang.)] [dist_value (distance) (ang.)] [target atom (1,2)] [off target atoms (3-5)] ...]
        self.metadynamics = []

class iEIPInterface(BiasPotInterface):# inheritance is not good for readable code.
    def __init__(self, folder_name=""):
        super().__init__()
        self.INPUT = folder_name
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        self.functional = 'b3lyp'#functional(ex. b3lyp)
        self.sub_basisset = '' #sub_basisset (ex. I LanL2DZ)
        self.excited_state = [0, 0]
        self.N_THREAD = 8  # threads
        self.SET_MEMORY = '1GB'  # use mem(ex. 1GB)

        self.usextb = "None"  # use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB
        self.usedxtb = "None"
        self.model_function_mode = "None"
        self.pyscf = False
        self.electronic_charge = [0, 0]
        self.spin_multiplicity = [1, 1]  # 'spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]'

        self.fix_atoms = []  
        self.geom_info = []
        self.opt_method = ["AdaBelief"]
        self.opt_fragment = []
        self.NSTEP = "999"

        return

class NEBInterface(BiasPotInterface):# inheritance is not good for readable code.
    def __init__(self, folder_name=""):
        super().__init__()
        self.INPUT = folder_name
        self.basisset = '6-31G(d)'
        self.sub_basisset = ''
        self.functional = 'b3lyp'
        self.excited_state = 0
        self.NSTEP = "10"
        self.unrestrict = False
        self.OM = False
        self.LUP = False
        self.DNEB = False
        self.NSEB = False
        self.partition = "0"
        self.N_THREAD = "8"
        self.SET_MEMORY = "1GB"
        self.apply_CI_NEB = '99999'
        self.steepest_descent = '99999'
        self.QUASI_NEWTOM_METHOD = False
        self.GLOBAL_QUASI_NEWTOM_METHOD = False
        self.ANEB_num = "0"
        self.usextb = "None"
        self.usedxtb = "None"
        self.fix_atoms = []  
        self.geom_info = []
        self.opt_method = ""
        self.opt_fragment = []
        self.fixedges = 0
        self.gradient_fix_atoms = ""
        self.pyscf = False
        return
    
class OptimizeInterface(BiasPotInterface):# inheritance is not good for readable code.
    def __init__(self, input_file=""):
        super().__init__()
        self.INPUT = input_file
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        self.functional = 'b3lyp'#functional(ex. b3lyp)
        self.sub_basisset = '' #sub_basisset (ex. I LanL2DZ)

        self.NSTEP = 300 #iter. number
        self.N_THREAD = 8 #threads
        self.SET_MEMORY = '1GB' #use mem(ex. 1GB)
        self.DELTA = 'x'
        self.excited_state = 0
        self.fix_atoms = ""#fix atoms (ex.) [atoms (ex.) 1,2,3-6]
        self.md_like_perturbation = "0.0"
        self.geom_info = "1"#calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]
        self.opt_method = ["AdaBelief"]#optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE third_order_momentum_Adam (quasi-Newton method) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]
        self.calc_exact_hess = -1#calculate exact hessian per steps (ex.) [steps per one hess calculation]
        self.usextb = "None"#use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB 
        self.usedxtb = "None"
        self.DS_AFIR = False
        self.pyscf = False
        self.electronic_charge = 0
        self.spin_multiplicity = 1#'spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]'
        self.saddle_order = 0
        self.opt_fragment = []
        self.constraint_condition = []
        self.othersoft = "None"
        self.NRO_analysis = False
        self.intrinsic_reaction_coordinate = []
        self.oniom_method = []
        self.tight_convergence_criteria = False
        self.loose_convergence_criteria = False

        self.shape_conditions = []
        self.use_model_hessian = False
        self.cpcm_solv_model = None
        self.cmds = False
        self.pca = False
        self.lagrange_constrain = []
        return
 
 
class MDInterface(BiasPotInterface):
    def __init__(self, input_file=""):
        # UNDER CONSTRACTION
        super().__init__()
        self.INPUT = input_file
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        
  
        self.timestep = 0.1
        return


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
    force_data["lagrange_constraint_condition_list"] = []
    force_data["lagrange_constraint_atoms"] = []
    if len(args.lagrange_constrain) % 2 != 0:
        print("invaild input (-lc) ")
        sys.exit(0)

    for i in range(int(len(args.lagrange_constrain)/2)):
        force_data["lagrange_constraint_condition_list"].append(str(args.lagrange_constrain[2*i]))
        force_data["lagrange_constraint_atoms"].append(num_parse(args.lagrange_constrain[2*i+1]))

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
    
    #------------------------
    #lone_pair_keep_angle
    if len(args.lone_pair_keep_angle) % 4 != 0:
        print("invaild input (-lpka)")
        sys.exit(0)
    
    force_data["lone_pair_keep_angle_spring_const"] = []
    force_data["lone_pair_keep_angle_angle"] = []
    force_data["lone_pair_keep_angle_atom_pair_1"] = []
    force_data["lone_pair_keep_angle_atom_pair_2"] = []
    
    for i in range(int(len(args.lone_pair_keep_angle)/4)):
        force_data["lone_pair_keep_angle_spring_const"].append(float(args.lone_pair_keep_angle[4*i]))#au
        force_data["lone_pair_keep_angle_angle"].append(float(args.lone_pair_keep_angle[4*i+1]))#degrees
        force_data["lone_pair_keep_angle_atom_pair_1"].append(num_parse(args.lone_pair_keep_angle[4*i+2]))
        force_data["lone_pair_keep_angle_atom_pair_2"].append(num_parse(args.lone_pair_keep_angle[4*i+3]))
        if len(force_data["lone_pair_keep_angle_atom_pair_1"][i]) != 4:
            print("invaild input (-ka lone_pair_atom_pairs_1)")
            sys.exit(0)
        if len(force_data["lone_pair_keep_angle_atom_pair_2"][i]) != 4:
            print("invaild input (-ka lone_pair_atom_pairs_2)")
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
    
    if len(args.aniso_keep_pot_v2) % 12 != 0:
        print("invaild input (-anikpv2)")
        sys.exit(0)
    
    force_data["aniso_keep_pot_v2_spring_const_mat"] = []
    force_data["aniso_keep_pot_v2_dist"] = []
    force_data["aniso_keep_pot_v2_fragm1"] = []
    force_data["aniso_keep_pot_v2_fragm2"] = []
    
    for i in range(int(len(args.aniso_keep_pot_v2)/12)):
        tmp_mat = np.array([[float(args.aniso_keep_pot_v2[12*i]),float(args.aniso_keep_pot_v2[12*i+1]),float(args.aniso_keep_pot_v2[12*i+2])],
                   [float(args.aniso_keep_pot_v2[12*i+3]),float(args.aniso_keep_pot_v2[12*i+4]),float(args.aniso_keep_pot_v2[12*i+5])],
                   [float(args.aniso_keep_pot_v2[12*i+6]),float(args.aniso_keep_pot_v2[12*i+7]),float(args.aniso_keep_pot_v2[12*i+8])]], dtype="float64")
        force_data["aniso_keep_pot_v2_spring_const_mat"].append(tmp_mat)#au
        force_data["aniso_keep_pot_v2_dist"].append(float(args.aniso_keep_pot_v2[12*i+9]))#degrees
        force_data["aniso_keep_pot_v2_fragm1"].append(num_parse(args.aniso_keep_pot_v2[12*i+10]))
        force_data["aniso_keep_pot_v2_fragm2"].append(num_parse(args.aniso_keep_pot_v2[12*i+11]))
       
    #---------------------
    
    
    if len(args.atom_distance_dependent_keep_angle) % 7 != 0:#[[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...]
        print("invaild input (-ddka)")
        sys.exit(0)
    
    force_data["aDD_keep_angle_spring_const"] = []
    force_data["aDD_keep_angle_min_angle"] = []
    force_data["aDD_keep_angle_max_angle"] = []
    force_data["aDD_keep_angle_base_dist"] = []
    force_data["aDD_keep_angle_reference_atom"] = []
    force_data["aDD_keep_angle_center_atom"] = []
    force_data["aDD_keep_angle_atoms"] = []
    
    for i in range(int(len(args.atom_distance_dependent_keep_angle)/7)):
        force_data["aDD_keep_angle_spring_const"].append(float(args.atom_distance_dependent_keep_angle[7*i]))#au
        force_data["aDD_keep_angle_min_angle"].append(float(args.atom_distance_dependent_keep_angle[7*i+1]))#degrees
        force_data["aDD_keep_angle_max_angle"].append(float(args.atom_distance_dependent_keep_angle[7*i+2]))#degrees
        if float(args.atom_distance_dependent_keep_angle[7*i+1]) > float(args.atom_distance_dependent_keep_angle[7*i+2]):
            print("invaild input (-ddka min_angle > max_angle)")
            sys.exit(0)
        
        force_data["aDD_keep_angle_base_dist"].append(float(args.atom_distance_dependent_keep_angle[7*i+3]))#ang.
        force_data["aDD_keep_angle_reference_atom"].append(int(args.atom_distance_dependent_keep_angle[7*i+4]))#ang.
        force_data["aDD_keep_angle_center_atom"].append(int(args.atom_distance_dependent_keep_angle[7*i+5]))#ang.
        force_data["aDD_keep_angle_atoms"].append(num_parse(args.atom_distance_dependent_keep_angle[7*i+6]))
        if len(force_data["aDD_keep_angle_atoms"][i]) != 3:
            print("invaild input (-ddka atoms)")
            sys.exit(0)
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
    
    if len(args.gradient_fix_atoms) > 0:
        force_data["gradient_fix_atoms"] = []
        
        for j in range(len(args.gradient_fix_atoms)):
           
            force_data["gradient_fix_atoms"].append(num_parse(args.gradient_fix_atoms[j]))
    else:
        force_data["gradient_fix_atoms"] = ""
    
    
    
    force_data["geom_info"] = num_parse(args.geom_info[0])
    
    force_data["opt_method"] = args.opt_method
    
    force_data["xtb"] = args.usextb
    force_data["opt_fragment"] = [num_parse(args.opt_fragment[i]) for i in range(len(args.opt_fragment))]
    if len(args.oniom_method) > 0:
        force_data["oniom_method"] = [num_parse(args.oniom_method[0]), num_parse(args.oniom_method[1]), args.oniom_method[2]]
    
    
    return force_data
