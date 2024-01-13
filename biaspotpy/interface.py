import argparse
import sys
import os
import copy
import glob
import itertools
import datetime
import time

import numpy as np

from optimizer import CalculateMoveVector, Opt_calc_tmps, Model_hess_tmp
from bias_pot import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from param import UnitValueLib, element_number

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




def optimizeparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='300', help='iter. number')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')

    parser = parser_for_biasforce(parser)
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-md", "--md_like_perturbation",  type=str, default="0.0", help='add perturbation like molecule dynamics (ex.) [[temperature (unit. K)]]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method group) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE, conjugate_gradient_descent (quasi-Newton method group) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for n-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    args = parser.parse_args()
    return args

def parser_for_biasforce(parser):
    parser.add_argument("-ma", "--manual_AFIR", nargs="*",  type=str, default=['0.0', '1', '2'], help='manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]')
    parser.add_argument("-rp", "--repulsive_potential", nargs="*",  type=str, default=['0.0','1.0', '1', '2', 'scale'], help='Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpv2", "--repulsive_potential_v2", nargs="*",  type=str, default=['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2', 'scale'], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value(kJ/mol ang.)] ...]')


    parser.add_argument("-cp", "--cone_potential", nargs="*",  type=str, default=['0.0','1.0','90','1', '2,3,4', '5-9'], help='Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
    
    
    parser.add_argument("-kp", "--keep_pot", nargs="*",  type=str, default=['0.0', '1.0', '1,2'], help='keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-akp", "--anharmonic_keep_pot", nargs="*",  type=str, default=['0.0', '1.0', '1.0', '1,2'], help='Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-ka", "--keep_angle", nargs="*",  type=str, default=['0.0', '90', '1,2,3'], help='keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] ')
    
    parser.add_argument("-ddka", "--atom_distance_dependent_keep_angle", nargs="*",  type=str, default=['0.0', '90', "120", "1.4", "5", "1", '2,3,4'], help='atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] ')
    
    
    parser.add_argument("-kda", "--keep_dihedral_angle", nargs="*",  type=str, default=['0.0', '90', '1,2,3,4'], help='keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] ')
    parser.add_argument("-vpp", "--void_point_pot", nargs="*",  type=str, default=['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"], help='void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] ')
   
    parser.add_argument("-wp", "--well_pot", nargs="*", type=str, default=['0.0','1','2','0.5,0.6,1.5,1.6'], help="Add potential to limit atom distance. (ex.) [[wall energy (kJ/mol)] [fragm.1] [fragm.2] [a,b,c,d (a<b<c<d) (ang.)] ...]")
    parser.add_argument("-wwp", "--wall_well_pot", nargs="*", type=str, default=['0.0','x','0.5,0.6,1.5,1.6',"1"], help="Add potential to limit atoms movement. (like sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (1,2,3-5)] ...]")
    parser.add_argument("-vpwp", "--void_point_well_pot", nargs="*", type=str, default=['0.0','0.0,0.0,0.0','0.5,0.6,1.5,1.6',"1"], help="Add potential to limit atom movement. (like sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (1,2,3-5)] ...]")
    parser.add_argument("-awp", "--around_well_pot", nargs="*", type=str, default=['0.0','1','0.5,0.6,1.5,1.6',"2"], help="Add potential to limit atom movement. (like sphere around fragment) (ex.) [[wall energy (kJ/mol)] [center (1,2-4)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")
    return parser


def nebparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='10', help='iter. number')
    parser.add_argument("-o", "--OM", action='store_true', help='J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action. (OM + DNEB)')
    parser.add_argument("-p", "--partition",  type=int, default='0', help='number of nodes')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-cineb", "--apply_CI_NEB",  type=int, default='99999', help='apply CI_NEB method')
    parser.add_argument("-sd", "--steepest_descent",  type=int, default='99999', help='apply steepest_descent method')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser = parser_for_biasforce(parser)
    args = parser.parse_args()
    args.fix_atoms = []
    
    args.geom_info = ["0"]
    args.opt_method = ""
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
        force_data["AFIR_gamma"].append(float(args.manual_AFIR[3*i]))#kj/mol
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

    
    
    if len(args.fix_atoms) > 0:
        force_data["fix_atoms"] = num_parse(args.fix_atoms[0])
    else:
        force_data["fix_atoms"] = ""
    
    force_data["geom_info"] = num_parse(args.geom_info[0])
    
    force_data["opt_method"] = args.opt_method
    
    force_data["xtb"] = args.usextb
    
    return force_data




class NEBInterface:
    def __init__(self, folder_name=""):
        self.INPUT = folder_name
        self.basisset = '6-31G(d)'
        self.functional = 'b3lyp'
        self.NSTEP = "10"
        self.OM = False
        self.partition = "0"
        self.N_THREAD = "8"
        self.SET_MEMORY = "1GB"
        self.apply_CI_NEB = '99999'
        self.steepest_descent = '99999'
        self.usextb = "None"
        self.manual_AFIR = ['0.0', '1', '2'] #manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]
        self.repulsive_potential = ['0.0','1.0', '1', '2', 'scale'] #Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value (ang. kJ/mol)] ...]
        self.repulsive_potential_v2 = ['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2', 'scale']#Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value (ang. kJ/mol)] ...]

        self.cone_potential = ['0.0','1.0','90','1', '2,3,4', '5-9']#'Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
        
        self.keep_pot = ['0.0', '1.0', '1,2']#keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.anharmonic_keep_pot = ['0.0', '1.0', '1.0', '1,2']#Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.keep_angle = ['0.0', '90', '1,2,3']#keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] 
        self.atom_distance_dependent_keep_angle = ['0.0', '90', "120", "1.4", "5", "1", '2,3,4']#'atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] '
        
        self.keep_dihedral_angle = ['0.0', '90', '1,2,3,4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] 
        self.void_point_pot = ['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"]#void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] 

        self.well_pot = ['0.0','1','2','0.5,0.6,1.5,1.6']
        self.wall_well_pot = ['0.0','x','0.5,0.6,1.5,1.6', '1']#Add potential to limit atoms movement. (sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.void_point_well_pot = ['0.0','0.0,0.0,0.0','0.5,0.6,1.5,1.6', '1']#"Add potential to limit atom movement. (sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.around_well_pot =['0.0','1','0.5,0.6,1.5,1.6',"2"] #Add potential to limit atom movement. (like sphere around 1 atom) (ex.) [[wall energy (kJ/mol)] [1 atom (1)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")
    
        self.fix_atoms = []
    
        self.geom_info = []
        self.opt_method = ""
    

        
        


class OptimizeInterface:
    def __init__(self, input_file=""):
        self.INPUT = input_file
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        self.functional = 'b3lyp'#functional(ex. b3lyp)
        self.sub_basisset = '' #sub_basisset (ex. I LanL2DZ)

        self.NSTEP = 300 #iter. number
        self.N_THREAD = 8 #threads
        self.SET_MEMORY = '1GB' #use mem(ex. 1GB)
        self.DELTA = 'x'

        self.manual_AFIR = ['0.0', '1', '2'] #manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]
        self.repulsive_potential = ['0.0','1.0', '1', '2', 'scale'] #Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value (ang. kJ/mol)] ...]
        self.repulsive_potential_v2 = ['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2', 'scale']#Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value (ang. kJ/mol)] ...]

        self.cone_potential = ['0.0','1.0','90','1', '2,3,4', '5-9']#'Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
        
        self.keep_pot = ['0.0', '1.0', '1,2']#keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.anharmonic_keep_pot = ['0.0', '1.0', '1.0', '1,2']#Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.keep_angle = ['0.0', '90', '1,2,3']#keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] 
        self.atom_distance_dependent_keep_angle = ['0.0', '90', "120", "1.4", "5", "1", '2,3,4']#'atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] '
        
        self.keep_dihedral_angle = ['0.0', '90', '1,2,3,4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] 
        self.void_point_pot = ['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"]#void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] 

        self.well_pot = ['0.0','1','2','0.5,0.6,1.5,1.6']
        self.wall_well_pot = ['0.0','x','0.5,0.6,1.5,1.6', '1']#Add potential to limit atoms movement. (sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.void_point_well_pot = ['0.0','0.0,0.0,0.0','0.5,0.6,1.5,1.6', '1']#"Add potential to limit atom movement. (sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.around_well_pot =['0.0','1','0.5,0.6,1.5,1.6',"2"] #Add potential to limit atom movement. (like sphere around 1 atom) (ex.) [[wall energy (kJ/mol)] [1 atom (1)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")
        
        self.fix_atoms = ""#fix atoms (ex.) [atoms (ex.) 1,2,3-6]
        self.md_like_perturbation = "0.0"
        self.geom_info = "1"#calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]
        self.opt_method = ["AdaBelief"]#optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE third_order_momentum_Adam (quasi-Newton method) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]
        self.calc_exact_hess = -1#calculate exact hessian per steps (ex.) [steps per one hess calculation]
        self.usextb = "None"#use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB 
        self.DS_AFIR = False
        self.pyscf = False
        self.electronic_charge = 0
        self.spin_multiplicity = 1#'spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]'
        self.saddle_order = 0
        return
 


class Optimize:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
 
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #

        self.MAX_FORCE_THRESHOLD = 0.0003 #
        self.RMS_FORCE_THRESHOLD = 0.0002 #
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0015 # 
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0010 #

        
        self.args = args #
        self.FC_COUNT = args.calc_exact_hess # 
        #---------------------------
        self.temperature = float(args.md_like_perturbation)
        
        #---------------------------
        if len(args.opt_method) > 2:
            print("invaild input (-opt)")
            sys.exit(0)
        
        if args.DELTA == "x":
            if args.opt_method[0] == "FSB":
                args.DELTA = 0.5
            elif args.opt_method[0] == "RFO_FSB":
                args.DELTA = 0.5
            elif args.opt_method[0] == "BFGS":
                args.DELTA = 0.5
            elif args.opt_method[0] == "RFO_BFGS":
                args.DELTA = 0.5
                
            elif args.opt_method[0] == "mBFGS":
                args.DELTA = 0.50
            elif args.opt_method[0] == "mFSB":
                args.DELTA = 0.50
            elif args.opt_method[0] == "RFO_mBFGS":
                args.DELTA = 0.30
            elif args.opt_method[0] == "RFO_mFSB":
                args.DELTA = 0.30

            elif args.opt_method[0] == "Adabound":
                args.DELTA = 0.01
            elif args.opt_method[0] == "AdaMax":
                args.DELTA = 0.01
            elif  args.opt_method[0] == "CG":
                args.DELTA = 1.0
                
            elif args.opt_method[0] == "TRM_FSB":
                args.DELTA = 0.60
            elif args.opt_method[0] == "TRM_BFGS":
                args.DELTA = 0.60
            else:
                args.DELTA = 0.06
        else:
            pass 
        self.DELTA = float(args.DELTA) # 

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT #
        self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        
        if args.pyscf:
            self.electronic_charge = args.electronic_charge
            self.spin_multiplicity = args.spin_multiplicity
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET) # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            else:
                self.SUB_BASIS_SET = { "default" : self.BASIS_SET}
        else:#psi4
            self.SUB_BASIS_SET = "" # 
            self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
            
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET +="\nassign "+str(self.BASIS_SET)+"\n" # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            
        #-----------------------------
        if args.usextb == "None":
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        else:
            self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+args.usextb+"_"+str(time.time())+"/"
        
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True) #
        
        self.Model_hess = None #
        self.Opt_params = None #
        self.DC_check_dist = 10.0#ang.
        
        return

    def optimize_using_tblite(self):
        from tblite_calculation_tools import SinglePoint
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = SinglePoint(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess)
        #-----------------------------------
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        #----------------------------------
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = glob.glob(self.BPA_FOLDER_DIRECTORY+"*.txt")
            for file in exit_file_detect:
                if "end.txt" in file:
                    exit_flag = True
                    break
            if exit_flag:
                if psi4:
                    psi4.core.clean()
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.tblite_calculation(file_directory, element_number_list,  electric_charge_and_multiplicity, iter, force_data["xtb"])
            self.Model_hess = SP.Model_hess
            #---------------------------------------
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            else:
                pass

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt(g**2).mean())+"\n")#abs(np.sqrt(B_g**2).mean()
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            CalcBiaspot.Model_hess = self.Model_hess
            
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            

            #----------------------------

            #----------------------------
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, self.args.saddle_order, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, Opt_params, Model_hess, trust_radii = CMV.calc_move_vector(iter, geom_num_list, B_g, force_data["opt_method"], pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            if len(force_data["geom_info"]) > 1:
                CSI = CalculationStructInfo()
               
                data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
                
                for num, i in enumerate(force_data["geom_info"]):
                    cos = CSI.calculate_cos(B_g[i-1] - g[i-1], g[i-1])
                    cos_list[num].append(cos)
                if iter == 0:
                    with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                        f.write(",".join(data_name_list)+"\n")
                
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                    f.write(",".join(list(map(str,data_list)))+"\n")
                    
            
            #----------------------------
            displacement_vector = geom_num_list - pre_geom
            print("caluculation results (unit a.u.):")
            print("OPT method            : {} ".format(force_data["opt_method"]))
            print("                         Value                         Threshold ")
            print("ENERGY                : {:>15.12f} ".format(e))
            print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
            print("Maxinum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
            print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(B_g**2).mean()), self.RMS_FORCE_THRESHOLD))
            print("Maxinum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), self.MAX_DISPLACEMENT_THRESHOLD))
            print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(displacement_vector**2).mean()), self.RMS_DISPLACEMENT_THRESHOLD))
            print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
            print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
            
            
            grad_list.append(np.linalg.norm(g))
            if abs(B_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt(B_g**2).mean()) < self.RMS_FORCE_THRESHOLD and  abs(displacement_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt(displacement_vector**2).mean()) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            
            #dissociation check
            atom_label_list = [i for i in range(len(new_geometry))]
            fragm_atom_num_list = []
            while len(atom_label_list) > 0:
                tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0])
                
                for j in tmp_fragm_list:
                    atom_label_list.remove(j)
                fragm_atom_num_list.append(tmp_fragm_list)
            
            print("\nfragm_list:", fragm_atom_num_list)
            
            if len(fragm_atom_num_list) > 1:
                fragm_dist_list = []
                for fragm_1_num, fragm_2_num in list(itertools.combinations(fragm_atom_num_list, 2)):
                    dist = Calculationtools().calc_fragm_distance(new_geometry, fragm_1_num, fragm_2_num)
                    fragm_dist_list.append(dist)
                
                
                if min(fragm_dist_list) > self.DC_check_dist:
                    print("mean fragm distance (ang.)", min(fragm_dist_list), ">", self.DC_check_dist)
                    
                    print("This molecules are dissociated.")
                    break
            
            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = FIO.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient [a.u.]", name="gradient")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return

    def optimize_using_psi4(self):
        from psi4_calculation_tools import SinglePoint
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = SinglePoint(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess)
        #----------------------------------
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = glob.glob(self.BPA_FOLDER_DIRECTORY+"*.txt")
            for file in exit_file_detect:
                if "end.txt" in file:
                    exit_flag = True
                    break
            if exit_flag:
                if psi4:
                    psi4.core.clean()
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.psi4_calculation(file_directory, element_list,  electric_charge_and_multiplicity, iter)
            self.Model_hess = SP.Model_hess
            
            #---------------------------------------
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            else:
                pass

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt(g**2).mean())+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            CalcBiaspot.Model_hess = self.Model_hess
            
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            

            #----------------------------

            #----------------------------
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, self.args.saddle_order, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, Opt_params, Model_hess, trust_radii = CMV.calc_move_vector(iter, geom_num_list, B_g, force_data["opt_method"], pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            if len(force_data["geom_info"]) > 1:
                CSI = CalculationStructInfo()
               
                data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
                
                for num, i in enumerate(force_data["geom_info"]):
                    cos = CSI.calculate_cos(B_g[i-1] - g[i-1], g[i-1])
                    cos_list[num].append(cos)
                if iter == 0:
                    with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                        f.write(",".join(data_name_list)+"\n")
                
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                    f.write(",".join(list(map(str,data_list)))+"\n")
                    
            
            #----------------------------
            displacement_vector = geom_num_list - pre_geom
            print("caluculation results (unit a.u.):")
            print("OPT method            : {} ".format(force_data["opt_method"]))
            print("                         Value                         Threshold ")
            print("ENERGY                : {:>15.12f} ".format(e))
            print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
            print("Maxinum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
            print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(B_g**2).mean()), self.RMS_FORCE_THRESHOLD))
            print("Maxinum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), self.MAX_DISPLACEMENT_THRESHOLD))
            print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(displacement_vector**2).mean()), self.RMS_DISPLACEMENT_THRESHOLD))
            print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
            print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
            
            
            grad_list.append(np.linalg.norm(g))
            if abs(B_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt(B_g**2).mean()) < self.RMS_FORCE_THRESHOLD and  abs(displacement_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt(displacement_vector**2).mean()) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            
            #dissociation check
            atom_label_list = [i for i in range(len(new_geometry))]
            fragm_atom_num_list = []
            while len(atom_label_list) > 0:
                tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0])
                
                for j in tmp_fragm_list:
                    atom_label_list.remove(j)
                fragm_atom_num_list.append(tmp_fragm_list)
            
            print("\nfragm_list:", fragm_atom_num_list)
            
            if len(fragm_atom_num_list) > 1:
                fragm_dist_list = []
                for fragm_1_num, fragm_2_num in list(itertools.combinations(fragm_atom_num_list, 2)):
                    dist = Calculationtools().calc_fragm_distance(new_geometry, fragm_1_num, fragm_2_num)
                    fragm_dist_list.append(dist)
                
                
                if min(fragm_dist_list) > self.DC_check_dist:
                    print("mean fragm distance (ang.)", min(fragm_dist_list), ">", self.DC_check_dist)
                    
                    print("This molecules are dissociated.")
                    break
            
            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = FIO.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient [a.u.]", name="gradient")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
    
    def optimize_using_pyscf(self):
        from pyscf_calculation_tools import SinglePoint
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = force_data_parser(self.args)
        finish_frag = False
        geometry_list, element_list = FIO.make_geometry_list_for_pyscf()
        file_directory = FIO.make_pyscf_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(self.args))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = SinglePoint(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess)
        #----------------------------------
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

        #----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = glob.glob(self.BPA_FOLDER_DIRECTORY+"*.txt")
            for file in exit_file_detect:
                if "end.txt" in file:
                    exit_flag = True
                    break
            if exit_flag:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.pyscf_calculation(file_directory, element_list, iter)
            self.Model_hess = SP.model_hess
            #---------------------------------------
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            else:
                pass

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt(g**2).mean())+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            CalcBiaspot.Model_hess = self.Model_hess
            
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            

            #----------------------------

            #----------------------------
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, self.args.saddle_order, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, Opt_params, Model_hess, trust_radii = CMV.calc_move_vector(iter, geom_num_list, B_g, force_data["opt_method"], pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            if len(force_data["geom_info"]) > 1:
                CSI = CalculationStructInfo()
               
                data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
                
                for num, i in enumerate(force_data["geom_info"]):
                    cos = CSI.calculate_cos(B_g[i-1] - g[i-1], g[i-1])
                    cos_list[num].append(cos)
                if iter == 0:
                    with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                        f.write(",".join(data_name_list)+"\n")
                
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                    f.write(",".join(list(map(str,data_list)))+"\n")
                    
            
            #----------------------------
            displacement_vector = geom_num_list - pre_geom
            print("caluculation results (unit a.u.):")
            print("OPT method            : {} ".format(force_data["opt_method"]))
            print("                         Value                         Threshold ")
            print("ENERGY                : {:>15.12f} ".format(e))
            print("BIAS  ENERGY          : {:>15.12f} ".format(B_e))
            print("Maxinum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(B_g.max()), self.MAX_FORCE_THRESHOLD))
            print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(B_g**2).mean()), self.RMS_FORCE_THRESHOLD))
            print("Maxinum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(displacement_vector.max()), self.MAX_DISPLACEMENT_THRESHOLD))
            print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(displacement_vector**2).mean()), self.RMS_DISPLACEMENT_THRESHOLD))
            print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
            print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(B_e - pre_B_e))
            
            
            grad_list.append(np.linalg.norm(g))
            if abs(B_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt(B_g**2).mean()) < self.RMS_FORCE_THRESHOLD and  abs(displacement_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt(displacement_vector**2).mean()) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            print("\ngeometry:")
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            #----------------------------
            #dissociation check
            atom_label_list = [i for i in range(len(new_geometry))]
            fragm_atom_num_list = []
            while len(atom_label_list) > 0:
                tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0])
                
                for j in tmp_fragm_list:
                    atom_label_list.remove(j)
                fragm_atom_num_list.append(tmp_fragm_list)
            
            print("\nfragm_list:", fragm_atom_num_list)
            
            if len(fragm_atom_num_list) > 1:
                fragm_dist_list = []
                for fragm_1_num, fragm_2_num in list(itertools.combinations(fragm_atom_num_list, 2)):
                    dist = Calculationtools().calc_fragm_distance(new_geometry, fragm_1_num, fragm_2_num)
                    fragm_dist_list.append(dist)
                
               
                if min(fragm_dist_list) > self.DC_check_dist:
                    print("mean fragm distance (ang.)", min(fragm_dist_list), ">", self.DC_check_dist)
                    
                    print("This molecules are dissociated.")
                    break
             
            #----------------------------
            
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = FIO.make_geometry_list_2_for_pyscf(new_geometry, element_list)
            file_directory = FIO.make_pyscf_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient [a.u.]", name="gradient")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make_for_pyscf()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
    
    def run(self):
        if self.args.pyscf:
            self.optimize_using_pyscf()
        elif self.args.usextb != "None":
            self.optimize_using_tblite()
        else:
            self.optimize_using_psi4()
    