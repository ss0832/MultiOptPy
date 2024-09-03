import argparse
import sys
import random
import inspect
import numpy as np


try:
    import psi4
    
except:
    pass

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
GFN-xTB(tblite)
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
    
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method group) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE, conjugate_gradient_descent (quasi-Newton method group) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
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
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, nargs="*", default=[0, 0], help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, nargs="*", default=[1, 1], help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-proj", "--project_out", nargs="*", type=str, default=[], help="project out optional vector (pair of group of atoms) from gradient and hessian (e.g.) [[1 2,3 (bond)] ...]")
    parser.add_argument("-bproj", "--bend_project_out", nargs="*", type=str, default=[], help="project out optional vector (bending of fragments) from gradient and hessian (e.g.) [[1 2,3 4 (bend)] ...]")
    parser.add_argument("-tproj", "--torsion_project_out", nargs="*", type=str, default=[], help="project out optional vector (torsion of fragments) from gradient and hessian (e.g.) [[1 2,3 4 5-7 (torsion)] ...]")
    parser.add_argument("-oproj", "--outofplain_project_out", nargs="*", type=str, default=[], help="project out optional vector (out-of-plain angle of fragments) from gradient and hessian (e.g.) [[1 2,3 4 7-9(out-of-plain angle)] ...]")

    
    args = parser.parse_args()#model_function_mode
    args.fix_atoms = []
    args.gradient_fix_atoms = []
    args.geom_info = ["0"]
     
    args.opt_fragment = []
    args.oniom_method = []
    return args



def optimizeparser(parser):
    parser.add_argument("INPUT", help='input xyz file name')
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
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method group) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE, conjugate_gradient_descent (quasi-Newton method group) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-dxtb", "--usedxtb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (This option is for dxtb module (hessian calculated by autograd diffential method is available.)) (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    parser.add_argument('-cmds','--cmds', help="Apply classical multidimensional scaling to calculated approx. reaction path.", action='store_true')
    parser.add_argument('-irc','--intrinsic_reaction_coordinates', help="Calculate intrinsic reaction coordinates. (ex.) [[step_size], [max_step], [IRC_method]] (Recommended) [0.5 300 lqa]", nargs="*", type=str, default=[])    
    parser.add_argument('-rc','--ricci_curvature', help="calculate Ricci scalar of calculated approx. reaction path.", action='store_true')
    parser.add_argument("-of", "--opt_fragment", nargs="*", type=str, default=[], help="Several atoms are grouped together as fragments and optimized. (This method doesnot work if you use quasi-newton method for optimazation.) (ex.) [[atoms (ex.) 1-4] ...] ")#(2024/3/26) this option doesn't work if you use quasi-Newton method for optimization.
    parser.add_argument("-cc", "--constraint_condition", nargs="*", type=str, default=[], help="apply constraint conditions for optimazation (ex.) [[(dinstance (ang.)),(atom1),(atom2)] [(bond_angle (deg.)),(atom1),(atom2),(atom3)] [(dihedral_angle (deg.)),(atom1),(atom2),(atom3),(atom4)] ...] ")
    parser.add_argument("-nro", "--NRO_analysis",  help="apply Natural Reaction Orbial analysis. (ref. Phys. Chem. Chem. Phys. 24, 3532 (2022))", action='store_true')
    parser.add_argument("-os", "--othersoft",  type=str, default="None", help='use other QM software. default is not using other QM software. (require python module, ASE (Atomic Simulation Environment)) (ex.) orca, gaussian, gamessus, mace_mp etc.')
    parser.add_argument('-tcc','--tight_convergence_criteria', help="apply tight opt criteria.", action='store_true')
    parser.add_argument("-proj", "--project_out", nargs="*", type=str, default=[], help="project out optional vector (pair of group of atoms) from gradient and hessian (e.g.) [[1 2,3 (bond)] ...]")
    parser.add_argument("-bproj", "--bend_project_out", nargs="*", type=str, default=[], help="project out optional vector (bending of fragments) from gradient and hessian (e.g.) [[1 2,3 4 (bend)] ...]")
    parser.add_argument("-tproj", "--torsion_project_out", nargs="*", type=str, default=[], help="project out optional vector (torsion of fragments) from gradient and hessian (e.g.) [[1 2,3 4 5-7 (torsion)] ...]")
    parser.add_argument("-oproj", "--outofplain_project_out", nargs="*", type=str, default=[], help="project out optional vector (out-of-plain angle of fragments) from gradient and hessian (e.g.) [[1 2,3 4 7-9(out-of-plain angle)] ...]")
    parser.add_argument('-modelhess','--use_model_hessian', help="use model hessian.", action='store_true')
    args = parser.parse_args()
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
    parser.add_argument("-proj", "--project_out", nargs="*", type=str, default=[], help="project out optional vector (pair of group of atoms) from gradient and hessian (e.g.) [[1 2,3 (bond)] ...]")
    parser.add_argument("-bproj", "--bend_project_out", nargs="*", type=str, default=[], help="project out optional vector (bending of fragments) from gradient and hessian (e.g.) [[1 2,3 4 (bend)] ...]")
    parser.add_argument("-tproj", "--torsion_project_out", nargs="*", type=str, default=[], help="project out optional vector (torsion of fragments) from gradient and hessian (e.g.) [[1 2,3 4 5-7 (torsion)] ...]")
    parser.add_argument("-oproj", "--outofplain_project_out", nargs="*", type=str, default=[], help="project out optional vector (out-of-plain angle of fragments) from gradient and hessian (e.g.) [[1 2,3 4 7-9(out-of-plain angle)] ...]")

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
    parser.add_argument("-proj", "--project_out", nargs="*", type=str, default=[], help="project out optional vector (pair of group of atoms) from gradient and hessian (e.g.) [[1 2,3 (bond)] ...]")
    parser.add_argument("-bproj", "--bend_project_out", nargs="*", type=str, default=[], help="project out optional vector (bending of fragments) from gradient and hessian (e.g.) [[1 2,3 4 (bend)] ...]")
    parser.add_argument("-tproj", "--torsion_project_out", nargs="*", type=str, default=[], help="project out optional vector (torsion of fragments) from gradient and hessian (e.g.) [[1 2,3 4 5-7 (torsion)] ...]")
    parser.add_argument("-oproj", "--outofplain_project_out", nargs="*", type=str, default=[], help="project out optional vector (out-of-plain angle of fragments) from gradient and hessian (e.g.) [[1 2,3 4 7-9(out-of-plain angle)] ...]")

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
        self.project_out = []
        self.bend_project_out = []
        self.torsion_project_out = []
        self.outofplain_project_out = []
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
        self.project_out = []
        self.bend_project_out = []
        self.torsion_project_out = []
        self.outofplain_project_out = []
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
        self.project_out = []
        self.bend_project_out = []
        self.torsion_project_out = []
        self.outofplain_project_out = [] 
        self.use_model_hessian = False
        return
 
 
class MDInterface(BiasPotInterface):
    def __init__(self, input_file=""):
        # UNDER CONSTRACTION
        super().__init__()
        self.INPUT = input_file
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        
        self.project_out = []
        self.bend_project_out = []
        self.torsion_project_out = []
        self.outofplain_project_out = []
        return



def force_data_parser(args):
    def num_parse(numbers):
        result = []
        for part in numbers.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
        return result

    def parse_values(data, structure, num_items):
        parsed_data = {key: [] for key in structure}
        if len(data) % num_items != 0:
            print(f"invalid input ({data})")
            sys.exit(0)
        
        for i in range(0, len(data), num_items):
            
            for j, key in enumerate(structure):
                
                
                if type(structure[key]) is list:
                    tmp_list = []
                    
                    for k in range(len(structure[key])):
                        if not inspect.isclass(structure[key][k]):
                            if structure[key][k].__code__ ==  num_parse.__code__:  # If we expect a num_parse operation
                                tmp_list.append(num_parse(data[i + j + k]))    
        
                            elif isinstance(structure[key][k], type(list(map(float, [0])))):  # If we expect a list of floats
                            
                                tmp_list.append(list(map(float, data[i + j + k].split(","))))
                        
                        else:  # Otherwise, just cast to the type expected (float, int, etc.)
                            tmp_list.append(structure[key][k](data[i + j + k]))
                    
                    parsed_data[key].append(tmp_list)
                
                else:
                    if not inspect.isclass(structure[key]):
                        if structure[key].__code__ ==  num_parse.__code__:  # If we expect a num_parse operation
                            parsed_data[key].append(num_parse(data[i + j]))    
                        elif isinstance(structure[key], type(list(map(float, [0])))):  # If we expect a list of floats
                            parsed_data[key].append(list(map(float, data[i + j].split(","))))
                    
                    else:  # Otherwise, just cast to the type expected (float, int, etc.)
                        parsed_data[key].append(structure[key](data[i + j]))
            
        return parsed_data

    force_data = {}

    force_data.update(parse_values(args.bond_range_potential, {
        "value_range_upper_const": float,
        "value_range_lower_const": float,
        "value_range_upper_distance": float,
        "value_range_lower_distance": float,
        "value_range_fragm_1": num_parse,
        "value_range_fragm_2": num_parse
    }, 6))

    force_data.update(parse_values(args.project_out, {
        "project_out_fragm_pair": [num_parse, num_parse]
    }, 2))

    force_data.update(parse_values(args.bend_project_out, {
        "project_out_bend_fragms": [num_parse, num_parse, num_parse]
    }, 3))

    force_data.update(parse_values(args.torsion_project_out, {
        "project_out_torsion_fragms": [num_parse, num_parse, num_parse, num_parse]
    }, 4))

    force_data.update(parse_values(args.outofplain_project_out, {
        "project_out_outofplain_fragms": [num_parse, num_parse, num_parse, num_parse]
    }, 4))

    force_data.update(parse_values(args.flux_potential, {
        "flux_pot_const": lambda x: np.array(x.split(","), dtype="float64"),
        "flux_pot_order": lambda x: np.array(x.split(","), dtype="float64"),
        "flux_pot_direction": lambda x: np.array(x.split(","), dtype="float64"),
        "flux_pot_target": num_parse
    }, 4))

    force_data.update(parse_values(args.universal_potential, {
        "universal_pot_const": float,
        "universal_pot_target": num_parse
    }, 2))

    force_data.update(parse_values(args.spacer_model_potential, {
        "spacer_model_potential_well_depth": float,
        "spacer_model_potential_distance": float,
        "spacer_model_potential_cavity_scaling": float,
        "spacer_model_potential_particle_number": int,
        "spacer_model_potential_target": num_parse
    }, 5))



    force_data.update(parse_values(args.repulsive_potential, {
        "repulsive_potential_well_scale": float,
        "repulsive_potential_dist_scale": float,
        "repulsive_potential_Fragm_1": num_parse,
        "repulsive_potential_Fragm_2": num_parse,
        "repulsive_potential_unit": str
    }, 5))


    force_data.update(parse_values(args.repulsive_potential_v2, {
        "repulsive_potential_v2_well_scale": float,
        "repulsive_potential_v2_dist_scale": float,
        "repulsive_potential_v2_length": float,
        "repulsive_potential_v2_const_rep": float,
        "repulsive_potential_v2_const_attr": float,
        "repulsive_potential_v2_order_rep": float,
        "repulsive_potential_v2_order_attr": float,
        "repulsive_potential_v2_center": num_parse,
        "repulsive_potential_v2_target": num_parse,
        "repulsive_potential_v2_unit": str
    }, 10))

    force_data.update(parse_values(args.repulsive_potential_gaussian, {
        "repulsive_potential_gaussian_LJ_well_depth": float,
        "repulsive_potential_gaussian_LJ_dist": float,
        "repulsive_potential_gaussian_gau_well_depth": float,
        "repulsive_potential_gaussian_gau_dist": float,
        "repulsive_potential_gaussian_gau_range": float,
        "repulsive_potential_gaussian_fragm_1": num_parse,
        "repulsive_potential_gaussian_fragm_2": num_parse
    }, 7))

    force_data.update(parse_values(args.cone_potential, {
        "cone_potential_well_value": float,
        "cone_potential_dist_value": float,
        "cone_potential_cone_angle": float,
        "cone_potential_center": int,
        "cone_potential_three_atoms": num_parse,
        "cone_potential_target": num_parse
    }, 6))

    force_data.update(parse_values(args.manual_AFIR, {
        "AFIR_gamma": lambda x: list(map(float, x.split(","))),
        "AFIR_Fragm_1": num_parse,
        "AFIR_Fragm_2": num_parse
    }, 3))
    
    
    force_data.update(parse_values(args.anharmonic_keep_pot, {
        "anharmonic_keep_pot_potential_well_depth": float,
        "anharmonic_keep_pot_spring_const": float,
        "anharmonic_keep_pot_distance": float,
        "anharmonic_keep_pot_atom_pairs": num_parse
    }, 4))
    
    
    force_data.update(parse_values(args.keep_pot, {
        "keep_pot_spring_const": float,
        "keep_pot_distance": float,
        "keep_pot_atom_pairs": num_parse
    }, 3))
    
    force_data.update(parse_values(args.keep_pot_v2, {
        "keep_pot_v2_spring_const": lambda x: list(map(float, x.split(","))),
        "keep_pot_v2_distance": lambda x: list(map(float, x.split(","))),
        "keep_pot_v2_fragm1": num_parse,
        "keep_pot_v2_fragm2": num_parse
    }, 4))    
    
    force_data.update(parse_values(args.keep_angle, {
        "keep_angle_spring_const": float,
        "keep_angle_angle": float,
        "keep_angle_atom_pairs": num_parse
    }, 3))
    
    force_data.update(parse_values(args.lone_pair_keep_angle, {
        "lone_pair_keep_angle_spring_const": float,
        "lone_pair_keep_angle_angle": float,
        "lone_pair_keep_angle_atom_pair_1": num_parse,
        "lone_pair_keep_angle_atom_pair_2": num_parse
    }, 4))
    
    
    force_data.update(parse_values(args.keep_angle_v2, {
        "keep_angle_v2_spring_const": float,
        "keep_angle_v2_angle": float,
        "keep_angle_v2_fragm1": num_parse,
        "keep_angle_v2_fragm2": num_parse,
        "keep_angle_v2_fragm3": num_parse
    }, 5))
    
    force_data.update(parse_values(args.aniso_keep_pot_v2, {
        "aniso_keep_pot_v2_spring_const_mat": lambda x: np.array([[float(x[0]),float(x[1]),float(x[2])],
            [float(x[3]),float(x[4]),float(x[5])],
            [float(x[6]),float(x[7]),float(x[8])]], dtype="float64"),
        "aniso_keep_pot_v2_dist": float,
        "aniso_keep_pot_v2_fragm1": num_parse,
        "aniso_keep_pot_v2_fragm2": num_parse
    }, 12))
    
    force_data.update(parse_values(args.atom_distance_dependent_keep_angle, {
        "aDD_keep_angle_spring_const": float,
        "aDD_keep_angle_min_angle": float,
        "aDD_keep_angle_max_angle": float,
        "aDD_keep_angle_base_dist": float,
        "aDD_keep_angle_reference_atom": int,
        "aDD_keep_angle_center_atom": int,
        "aDD_keep_angle_atoms": num_parse
    }, 7))
    
    force_data.update(parse_values(args.keep_dihedral_angle, {
        "keep_dihedral_angle_spring_const": float,
        "keep_dihedral_angle_angle": float,
        "keep_dihedral_angle_atom_pairs": num_parse
    }, 3))
    
    force_data.update(parse_values(args.keep_out_of_plain_angle, {
        "keep_out_of_plain_angle_spring_const": float,
        "keep_out_of_plain_angle_angle": float,
        "keep_out_of_plain_angle_atom_pairs": num_parse
    }, 3))
    
    force_data.update(parse_values(args.keep_out_of_plain_angle_v2, {
        "keep_out_of_plain_angle_v2_spring_const": lambda x: list(map(float, x.split(","))),
        "keep_out_of_plain_angle_v2_angle": lambda x: list(map(float, x.split(","))),
        "keep_out_of_plain_angle_v2_fragm1": num_parse,
        "keep_out_of_plain_angle_v2_fragm2": num_parse,
        "keep_out_of_plain_angle_v2_fragm3": num_parse,
        "keep_out_of_plain_angle_v2_fragm4": num_parse
    }, 6))
    
    force_data.update(parse_values(args.keep_dihedral_angle_v2, {
        "keep_dihedral_angle_v2_spring_const": lambda x: list(map(float, x.split(","))),
        "keep_dihedral_angle_v2_angle": lambda x: list(map(float, x.split(","))),
        "keep_dihedral_angle_v2_fragm1": num_parse,
        "keep_dihedral_angle_v2_fragm2": num_parse,
        "keep_dihedral_angle_v2_fragm3": num_parse,
        "keep_dihedral_angle_v2_fragm4": num_parse
    }, 6))
    
    force_data.update(parse_values(args.keep_dihedral_angle_cos, {
        "keep_dihedral_angle_cos_potential_const": lambda x: list(map(float, x.split(","))),
        "keep_dihedral_angle_cos_angle_const": lambda x: list(map(float, x.split(","))),
        "keep_dihedral_angle_cos_angle": lambda x: list(map(float, x.split(","))),
        "keep_dihedral_angle_cos_fragm1": num_parse,
        "keep_dihedral_angle_cos_fragm2": num_parse,
        "keep_dihedral_angle_cos_fragm3": num_parse,
        "keep_dihedral_angle_cos_fragm4": num_parse
    }, 7))
    
    force_data.update(parse_values(args.well_pot, {
        "well_pot_wall_energy": float,#kJ/mol
        "well_pot_fragm_1": num_parse,
        "well_pot_fragm_2": num_parse,
        "well_pot_limit_dist": lambda x: list(map(float, x.split(",")))
    }, 4))
    
    force_data.update(parse_values(args.wall_well_pot, {
        "wall_well_pot_wall_energy": float,#kJ/mol
        "wall_well_pot_direction": str,
        "wall_well_pot_limit_dist": lambda x: list(map(float, x.split(","))),
        "wall_well_pot_target": num_parse
    }, 4))
    
    force_data.update(parse_values(args.void_point_well_pot, {
        "void_point_well_pot_wall_energy": float,#kJ/mol
        "void_point_well_pot_coordinate": lambda x: list(map(float, x.split(","))),
        "void_point_well_pot_limit_dist": lambda x: list(map(float, x.split(","))),
        "void_point_well_pot_target": num_parse
    }, 4))
    
    force_data.update(parse_values(args.around_well_pot, {
        "around_well_pot_wall_energy": float,#kJ/mol
        "around_well_pot_center": num_parse,
        "around_well_pot_limit_dist": lambda x: list(map(float, x.split(","))),
        "around_well_pot_target": num_parse
    }, 4))
    
    force_data.update(parse_values(args.void_point_pot, {
        "void_point_pot_spring_const": float,#au
        "void_point_pot_distance": float,#ang
        "void_point_pot_coord": lambda x: list(map(float, x.split(","))),#ang
        "void_point_pot_atoms": num_parse,
        "void_point_pot_order": float
    }, 5))
    
    force_data.update(parse_values(args.metadynamics, {
        "gaussian_potential_target": str,
        "gaussian_potential_height": float,
        "gaussian_potential_width": float,
        "gaussian_potential_tgt_atom": num_parse
    }, 4))
    
        

        
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
