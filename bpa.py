import os
import sys
import glob
import copy
import time
import datetime

import random
import math
import argparse
import itertools

from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import numpy as np


try:
    import pyscf
    from pyscf.hessian import thermo
except:
    print("You can't use pyscf.")
try:#Unless you import tblite before pytorch, tblite doesnt work well. 
    from tblite.interface import Calculator
except:
    print("You can't use extended tight binding method.")
try:
    import psi4
except:
    print("You can't use psi4.")

try:
    import torch
    
except:
    print("please install pytorch. (pip install torch)")
    sys.exit(1)




"""
    BiasPotentialAddition
    Copyright (C) 2023 ss0832

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
references(-opt):

RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
FSB
 J. Chem. Phys. 1999, 111, 10806
Psi4
 D. G. A. Smith, L. A. Burns, A. C. Simmonett, R. M. Parrish, M. C. Schieber, R. Galvelis, P. Kraus, H. Kruse, R. Di Remigio, A. Alenaizan, A. M. James, S. Lehtola, J. P. Misiewicz, M. Scheurer, R. A. Shaw, J. B. Schriber, Y. Xie, Z. L. Glick, D. A. Sirianni, J. S. O'Brien, J. M. Waldrop, A. Kumar, E. G. Hohenstein, B. P. Pritchard, B. R. Brooks, H. F. Schaefer III, A. Yu. Sokolov, K. Patkowski, A. E. DePrince III, U. Bozkaya, R. A. King, F. A. Evangelista, J. M. Turney, T. D. Crawford, C. D. Sherrill, "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).
 
PySCF
Recent developments in the PySCF program package, Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan, J. Chem. Phys., 153, 024109 (2020). doi:10.1063/5.0006074

GFN2-xTB(tblite)
J. Chem. Theory Comput. 2019, 15, 3, 1652–1671 
GFN-xTB(tblite)
J. Chem. Theory Comput. 2017, 13, 5, 1989–2009
"""

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='300', help='iter. number')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')

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
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-md", "--md_like_perturbation",  type=str, default="0.0", help='add perturbation like molecule dynamics (ex.) [[temperature (unit. K)]]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method group) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE, conjugate_gradient_descent (quasi-Newton method group) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-dsafir','--DS_AFIR', help="use DS-AFIR method.", action='store_true')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for n-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    args = parser.parse_args()
    return args

class Interface:
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
        
    def force_data_parser(self, args):
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

#------------------------------------------
#constant section
def UFF_VDW_distance_lib(element):
    if element is int:
        element = number_element(element)
    UFF_VDW_distance = {'H':2.886,'He':2.362 ,
                        'Li' : 2.451 ,'Be': 2.745, 'B':4.083 ,'C': 3.851, 'N':3.660,'O':3.500 , 'F':3.364,'Ne': 3.243, 
                        'Na':2.983,'Mg': 3.021 ,'Al':4.499 ,'Si': 4.295, 'P':4.147, 'S':4.035 ,'Cl':3.947,'Ar':3.868 ,
                        'K':3.812 ,'Ca':3.399 ,'Sc':3.295 ,'Ti':3.175 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 2.912,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':2.763 ,'Ga': 4.383,'Ge':4.280,'As':4.230 ,'Se':4.205,'Br':4.189,'Kr':4.141 ,
                        'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':2.963 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.463 ,'Sn':4.392 ,'Sb':4.420 ,'Te':4.470 , 'I':4.50, 'Xe':4.404 , 
                        'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,
                        'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr

def UFF_VDW_well_depth_lib(element):
    if element is int:
        element = number_element(element)         
    UFF_VDW_well_depth = {'H':0.044, 'He':0.056 ,
                          'Li':0.025 ,'Be':0.085 ,'B':0.180,'C': 0.105, 'N':0.069, 'O':0.060,'F':0.050,'Ne':0.042 , 
                          'Na':0.030, 'Mg':0.111 ,'Al':0.505 ,'Si': 0.402, 'P':0.305, 'S':0.274, 'Cl':0.227,  'Ar':0.185 ,
                          'K':0.035 ,'Ca':0.238 ,'Sc':0.019 ,'Ti':0.017 ,'V':0.016 , 'Cr':0.015, 'Mn':0.013 ,'Fe': 0.013,'Co':0.014 ,'Ni':0.015 ,'Cu':0.005 ,'Zn':0.124 ,'Ga':0.415 ,'Ge':0.379, 'As':0.309 ,'Se':0.291,'Br':0.251,'Kr':0.220 ,
                          'Rb':0.04 ,'Sr':0.235 ,'Y':0.072 ,'Zr':0.069 ,'Nb':0.059 ,'Mo':0.056 ,'Tc':0.048 ,'Ru':0.056 ,'Rh':0.053 ,'Pd':0.048 ,'Ag':0.036 ,'Cd':0.228 ,'In':0.599 ,'Sn':0.567 ,'Sb':0.449 ,'Te':0.398 , 'I':0.339,'Xe':0.332 , 
                          'Cs':0.045 ,'Ba':0.364 , 'La':0.017 , 'Ce':0.013 ,'Pr':0.010 ,'Nd':0.010 ,'Pm':0.009 ,'Sm':0.008 ,'Eu':0.008 ,'Gd':0.009 ,'Tb':0.007 ,'Dy':0.007 ,'Ho':0.007 ,'Er':0.007 ,'Tm':0.006 ,'Yb':0.228 ,'Lu':0.041 ,'Hf':0.072 ,
                          'Ta':0.081 ,'W':0.067 ,'Re':0.066 ,'Os':0.037 ,'Ir':0.073 ,'Pt':0.080 ,'Au':0.039 ,'Hg':0.385 ,'Tl':0.680 ,'Pb':0.663 ,'Bi':0.518 ,'Po':0.325 ,'At':0.284 ,'Rn':0.248, 'X':0.010}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 # kcal/mol
                
    return UFF_VDW_well_depth[element] / UnitValueLib().hartree2kcalmol
                
def covalent_radii_lib(element):
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, 
           "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, 
           "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, 
           "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,
           "Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42, 'X':1.000}#ang.
    # ref. Pekka Pyykkö; Michiko Atsumi (2009). “Molecular single-bond covalent radii for elements 1 - 118”. Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. (H...Rn)
            
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr

def element_number(elem):
    num = {"H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, 
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,"In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53,"Xe": 54,
        "Cs": 55 ,"Ba": 56, "La": 57,"Ce":58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,"Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,"Hf": 72,"Ta": 73,"W": 74,"Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,"Tl": 81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86}
        
    return num[elem]
def number_element(num):
    elem = {1: "H",  2:"He",
         3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
        11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
        19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr",
        37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",43:"Tc",44:"Ru",45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe",
        55:"Cs", 56:"Ba", 57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy" ,67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn"}
        
    return elem[num]

def atomic_mass(elem):
    if elem is int:
        elem_num = elem
    else:    
        elem_num = element_number(elem)
    mass = {1: 1.00782503223, 2: 4.00260325413,
    3: 7.0160034366, 4: 9.012183065, 5: 11.00930536, 6: 12.0, 7: 14.00307400443, 8: 15.99491461957, 9: 18.99840316273, 10: 19.9924401762,
    11: 22.989769282, 12: 23.985041697, 13: 26.98153853, 14: 27.97692653465, 15: 30.97376199842, 16: 31.9720711744, 17: 34.968852682, 18: 39.9623831237,
    19: 38.9637064864, 20: 39.962590863, 21: 44.95590828, 22: 47.94794198, 23: 50.94395704, 24: 51.94050623, 25: 54.93804391, 26: 55.93493633, 27: 58.93319429, 28: 57.93534241, 29: 62.92959772, 30: 63.92914201, 31: 68.9255735, 32: 73.921177761, 33: 74.92159457, 34: 79.9165218, 35: 78.9183376, 36: 83.9114977282,
    37: 84.9117897379, 38: 87.9056125, 39: 88.9058403, 40: 89.9046977, 41: 92.906373, 42: 97.90540482, 43: 96.9063667, 44: 101.9043441, 45: 102.905498, 46: 105.9034804, 47: 106.9050916, 48: 113.90336509, 49: 114.903878776, 50: 119.90220163, 51: 120.903812, 52: 129.906222748, 53: 126.9044719, 54: 131.9041550856,
    55: 132.905451961, 56: 137.905247, 57: 138.9063563, 58: 139.9054431, 59: 140.9076576, 60: 141.907729, 61: 144.9127559, 62: 151.9197397, 63: 152.921238, 64: 157.9241123, 65: 158.9253547, 66: 163.9291819, 67: 164.9303288,
    68: 165.9302995, 69: 168.9342179, 70: 173.9388664, 71: 174.9407752, 72: 179.946557, 73: 180.9479958, 74: 183.95093092, 75: 186.9557501, 76: 191.961477, 77: 192.9629216, 78: 194.9647917, 79: 196.96656879, 80: 201.9706434, 81: 204.9744278,
    82: 207.9766525, 83: 208.9803991, 84: 208.9824308, 85: 209.9871479, 86: 222.0175782}# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    return mass[elem_num]


class UnitValueLib: 
    def __init__(self):
        self.hartree2kcalmol = 627.509 #
        self.bohr2angstroms = 0.52917721067 #
        self.hartree2kjmol = 2625.500 #
        return
        
        
#end of constant section
#----------------------------------------

class CalculateMoveVector:
    def __init__(self, DELTA, Opt_params, Model_hess, BPA_hessian, trust_radii, saddle_order=0,  FC_COUNT=-1, temperature=0.0):
        self.Opt_params = Opt_params 
        self.DELTA = DELTA
        self.Model_hess = Model_hess
        self.temperature = temperature
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.FC_COUNT = FC_COUNT
        self.MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.RMS_FORCE_SWITCHING_THRESHOLD = 0.0008
        self.BPA_hessian = BPA_hessian
        self.trust_radii = trust_radii
        self.saddle_order = saddle_order
        
    def calc_move_vector(self, iter, geom_num_list, B_g, opt_method_list, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):#geom_num_list:Bohr
        def update_trust_radii(trust_radii, B_e, pre_B_e):
            if pre_B_e >= B_e:
                trust_radii *= 3.0
            else:
                trust_radii *= 0.1
                    
            return np.clip(trust_radii, 0.001, 1.2)

        def BFGS_hessian_update(hess, displacement, delta_grad):
            
            A = delta_grad - np.dot(hess, displacement)

            delta_hess = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))

            return delta_hess
        def FSB_hessian_update(hess, displacement, delta_grad):

            A = delta_grad - np.dot(hess, displacement)
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS

            return delta_hess

        

        def RFO_BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("RFO_BFGS_quasi_newton_method")
            print("saddle order:", self.saddle_order)
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            DELTA_for_QNM = self.DELTA
           

            delta_hess = BFGS_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
            
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
            matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            eigenvalue = np.sort(eigenvalue)
            lambda_for_calc = float(eigenvalue[self.saddle_order])
            

                
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            DELTA_for_QNM = self.DELTA
            
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM)
            
                

            self.Model_hess = Model_hess_tmp(new_hess)
            
            return move_vector
            
        def BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("BFGS_quasi_newton_method")
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
           
           
            
            delta_hess_BFGS = BFGS_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
                
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            

            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
        

        def RFO_FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("RFO_FSB_quasi_newton_method")
            print("saddle order:", self.saddle_order)
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            DELTA_for_QNM = self.DELTA
            
                

            delta_hess = FSB_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
            matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            eigenvalue = np.sort(eigenvalue)
            lambda_for_calc = float(eigenvalue[self.saddle_order])
            

                
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            DELTA_for_QNM = self.DELTA

                
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM)
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
            
        def FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("FSB_quasi_newton_method")
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            

            delta_hess = FSB_hessian_update(self.Model_hess.model_hess, displacement, delta_grad)
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            

            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
        
        # arXiv:2307.13744v1
        def momentum_based_BFGS(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("momentum_based_BFGS")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:

                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
                
            beta = 0.50
            
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            

            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
            

            delta_hess_BFGS = BFGS_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)


            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
  
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            
            return move_vector 
        
        def momentum_based_FSB(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("momentum_based_FSB")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.50
            
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            

            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
           
 
            delta_hess = FSB_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
         
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
                
          
        def RFO_momentum_based_BFGS(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("RFO_momentum_based_BFGS")
            print("saddle order:", self.saddle_order)
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.5
            
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
          

            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
          
        
           
            
            delta_hess_BFGS = BFGS_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian
             
            DELTA_for_QNM = self.DELTA

            matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            eigenvalue = np.sort(eigenvalue)
            lambda_for_calc = float(eigenvalue[self.saddle_order])
            

            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
        
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
        
        def RFO_momentum_based_FSB(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g):
            print("RFO_momentum_based_FSB")
            print("saddle order:", self.saddle_order)
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.5
            print("beta :", beta)
            delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            

            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
            
            delta_hess = FSB_hessian_update(self.Model_hess.model_hess, delta_momentum_disp, delta_momentum_grad)
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess + self.BPA_hessian
            else:
                new_hess = self.Model_hess.model_hess + self.BPA_hessian

            DELTA_for_QNM = self.DELTA
            
            matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            eigenvalue = np.sort(eigenvalue)
            lambda_for_calc = float(eigenvalue[self.saddle_order])
            

            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
        
        def conjugate_gradient_descent(geom_num_list, pre_move_vector, B_g, pre_B_g):
            #cg method
            
            alpha = np.dot(B_g.reshape(1, len(geom_num_list)*3), (self.Opt_params.adam_v).reshape(len(geom_num_list)*3, 1)) / np.dot(self.Opt_params.adam_v.reshape(1, len(geom_num_list)*3), self.Opt_params.adam_v.reshape(len(geom_num_list)*3, 1))
            
            move_vector = self.DELTA * alpha * self.Opt_params.adam_v
            
            beta = np.dot(B_g.reshape(1, len(geom_num_list)*3), (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)) / np.dot(pre_B_g.reshape(1, len(geom_num_list)*3), pre_B_g.reshape(len(geom_num_list)*3, 1)) ** 2 
            
            self.Opt_params.adam_v = copy.copy(-1 * B_g + abs(beta) * self.Opt_params.adam_v)
            
            
            
            return move_vector
        
        #arXiv:1412.6980v9
        def AdaMax(geom_num_list, B_g):#not worked well
            print("AdaMax")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count == 1:
                adamax_u = 1e-8
                self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, adamax_u)
                
            else:
                adamax_u = self.Opt_params.eve_d_tilde#eve_d_tilde = adamax_u 
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
            new_adamax_u = max(beta_v*adamax_u, np.linalg.norm(B_g))
               
            move_vector = []

            for i in range(len(geom_num_list)):
                move_vector.append((self.DELTA / (beta_m ** adam_count)) * (adam_m[i] / new_adamax_u))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, new_adamax_u)
            return move_vector
            
        #https://cs229.stanford.edu/proj2015/054_report.pdf
        def NAdam(geom_num_list, B_g):
            print("NAdam")
            mu = 0.975
            nu = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = []
            new_adam_v_hat = []
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(mu*adam_m[i] + (1.0 - mu)*(B_g[i]))
                new_adam_v[i] = copy.copy((nu*adam_v[i]) + (1.0 - nu)*(B_g[i]) ** 2)
                new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64") * ( mu / (1.0 - mu ** adam_count)) + np.array(B_g[i], dtype="float64") * ((1.0 - mu)/(1.0 - mu ** adam_count)))        
                new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64") * (nu / (1.0 - nu ** adam_count)))
            
            move_vector = []
            for i in range(len(geom_num_list)):
                move_vector.append( (self.DELTA*new_adam_m_hat[i]) / (np.sqrt(new_adam_v_hat[i] + Epsilon)))
                
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        
        
        #FIRE
        #Physical Review Letters, Vol. 97, 170201 (2006)
        def FIRE(geom_num_list, B_g):#MD-like optimization method. This method tends to converge local minima.
            print("FIRE")
            adam_count = self.Opt_params.adam_count
            N_acc = 5
            f_inc = 1.10
            f_acc = 0.99
            f_dec = 0.50
            dt_max = 0.8
            alpha_start = 0.1
            if adam_count == 1:
                self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, [0.1, alpha_start, 0])
                #valuable named 'eve_d_tilde' is parameters for FIRE.
                # [0]:dt [1]:alpha [2]:n_reset
            dt = self.Opt_params.eve_d_tilde[0]
            alpha = self.Opt_params.eve_d_tilde[1]
            n_reset = self.Opt_params.eve_d_tilde[2]
            
            pre_velocity = self.Opt_params.adam_v
            
            velocity = (1.0 - alpha) * pre_velocity + alpha * (np.linalg.norm(pre_velocity, ord=2)/np.linalg.norm(B_g, ord=2)) * B_g
            
            if adam_count > 1 and np.dot(pre_velocity.reshape(1, len(geom_num_list)*3), B_g.reshape(len(geom_num_list)*3, 1)) > 0:
                if n_reset > N_acc:
                    dt = min(dt * f_inc, dt_max)
                    alpha = alpha * f_acc
                n_reset += 1
            else:
                velocity *= 0.0
                alpha = alpha_start
                dt *= f_dec
                n_reset = 0
            
            velocity += dt*B_g
            
            move_vector = velocity * 0.0
            move_vector = copy.copy(dt * velocity)
            
            print("dt, alpha, n_reset :", dt, alpha, n_reset)
            self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, velocity, adam_count, [dt, alpha, n_reset])
            return move_vector
        #RAdam
        #arXiv:1908.03265v4
        def RADAM(geom_num_list, B_g):
            print("RADAM")
            beta_m = 0.9
            beta_v = 0.99
            rho_inf = 2.0 / (1.0- beta_v) - 1.0 
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = []
            new_adam_v_hat = []
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy((beta_v*adam_v[i]) + (1.0-beta_v)*(B_g[i] - new_adam_m[i])**2) + Epsilon
                new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64")/(1.0-beta_m**adam_count))        
                new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64")/(1.0-beta_v**adam_count))
            rho = rho_inf - (2.0*adam_count*beta_v**adam_count)/(1.0 -beta_v**adam_count)
                        
            move_vector = []
            if rho > 4.0:
                l_alpha = []
                for j in range(len(new_adam_v)):
                    l_alpha.append(np.sqrt((abs(1.0 - beta_v**adam_count))/new_adam_v[j]))
                l_alpha = np.array(l_alpha, dtype="float64")
                r = np.sqrt(((rho-4.0)*(rho-2.0)*rho_inf)/((rho_inf-4.0)*(rho_inf-2.0)*rho))
                for i in range(len(geom_num_list)):
                    move_vector.append(self.DELTA*r*new_adam_m_hat[i]*l_alpha[i])
            else:
                for i in range(len(geom_num_list)):
                    move_vector.append(self.DELTA*new_adam_m_hat[i])
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #AdaBelief
        #ref. arXiv:2010.07468v5
        def AdaBelief(geom_num_list, B_g):
            print("AdaBelief")
            beta_m = 0.9
            beta_v = 0.99
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i]-new_adam_m[i])**2)
               
            move_vector = []

            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #AdaDiff
        #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
        def AdaDiff(geom_num_list, B_g, pre_B_g):
            print("AdaDiff")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2 + (1.0-beta_v) * (B_g[i] - pre_B_g[i]) ** 2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        
        #EVE
        #ref.arXiv:1611.01505v3
        def EVE(geom_num_list, B_g, B_e, pre_B_e, pre_B_g):
            print("EVE")
            beta_m = 0.9
            beta_v = 0.999
            beta_d = 0.999
            c = 10
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            if adam_count > 1:
                eve_d_tilde = self.Opt_params.eve_d_tilde
            else:
                eve_d_tilde = 1.0
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i])/(1 - beta_v**adam_count))
                
            if adam_count > 1:
                eve_d = abs(B_e - pre_B_e)/ min(B_e, pre_B_e)
                eve_d_hat = np.clip(eve_d, 1/c , c)
                eve_d_tilde = beta_d*eve_d_tilde + (1.0 - beta_d)*eve_d_hat
                
            else:
                pass
            
            for i in range(len(geom_num_list)):
                 move_vector.append((self.DELTA/eve_d_tilde)*new_adam_m_hat[i]/(np.sqrt(new_adam_v_hat[i])+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, eve_d_tilde)
            return move_vector
        #AdamW
        #arXiv:2302.06675v4
        def AdamW(geom_num_list, B_g):
            print("AdamW")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            AdamW_lambda = 0.001
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon) + AdamW_lambda * geom_num_list[i])
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #Adam
        #arXiv:1412.6980
        def Adam(geom_num_list, B_g):
            print("Adam")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
            
        def third_order_momentum_Adam(geom_num_list, B_g):
            print("third_order_momentum_Adam")
            #self.Opt_params.eve_d_tilde is 3rd-order momentum
            beta_m = 0.9
            beta_v = 0.999
            beta_s = 0.9999999999
            
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count > 1:                
                adam_s = self.Opt_params.eve_d_tilde
            else:
                adam_s = adam_m*0.0
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            new_adam_s = adam_s*0.0
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            new_adam_s_hat = adam_s*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i])**2)
                new_adam_s[i] = copy.copy(beta_s*adam_s[i] + (1.0-beta_s)*(B_g[i])**3)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
                new_adam_s_hat[i] = copy.copy((new_adam_s[i] + Epsilon)/(1 - beta_s**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA * new_adam_m_hat[i] / np.abs(np.sqrt(new_adam_v_hat[i]+Epsilon) - ((new_adam_m_hat[i] * (new_adam_s_hat[i]) ** (1 / 3)) / (2.0 * np.sqrt(new_adam_v_hat[i]+Epsilon)) )) )
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, new_adam_s)
            return move_vector
                    
        #Adafactor
        #arXiv:1804.04235v1
        def Adafactor(geom_num_list, B_g):
            print("Adafactor")
            Epsilon_1 = 1e-08
            Epsilon_2 = self.DELTA

            adam_count = self.Opt_params.adam_count
            beta = 1 - adam_count ** (-0.8)
            rho = min(0.01, 1/np.sqrt(adam_count))
            alpha = max(np.sqrt(np.square(geom_num_list).mean()),  Epsilon_2) * rho
            adam_v = self.Opt_params.adam_m
            adam_u = self.Opt_params.adam_v
            new_adam_v = adam_v*0.0
            new_adam_u = adam_u*0.0
            new_adam_v = adam_v*0.0
            new_adam_u = adam_u*0.0
            new_adam_u_hat = adam_u*0.0
            for i in range(len(geom_num_list)):
                new_adam_v[i] = copy.copy(beta*adam_v[i] + (1.0-beta)*((B_g[i])**2 + np.array([1,1,1]) * Epsilon_1))
                new_adam_u[i] = copy.copy(B_g[i]/np.sqrt(new_adam_v[i]))
                
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_u_hat[i] = copy.copy(new_adam_u[i] / max(1, np.sqrt(np.square(new_adam_u).mean())))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(alpha*new_adam_u_hat[i])
                 
            self.Opt_params = Opt_calc_tmps(new_adam_v, new_adam_u, adam_count)
        
            return move_vector
        #Prodigy
        #arXiv:2306.06101v1
        def Prodigy(geom_num_list, B_g, initial_geom_num_list):
            print("Prodigy")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count

            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count == 1:
                adam_r = 0.0
                adam_s = adam_m*0.0
                d = 1e-1
                new_d = d
                self.Opt_params = Opt_calc_tmps(adam_m, adam_v, adam_count - 1, [d, adam_r, adam_s])
            else:
                d = self.Opt_params.eve_d_tilde[0]
                adam_r = self.Opt_params.eve_d_tilde[1]
                adam_s = self.Opt_params.eve_d_tilde[2]
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            new_adam_s = adam_s*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]*d))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(B_g[i]*d)**2)
                
                new_adam_s[i] = np.sqrt(beta_v)*adam_s[i] + (1.0 - np.sqrt(beta_v))*self.DELTA*B_g[i]*d**2  
            new_adam_r = np.sqrt(beta_v)*adam_r + (1.0 - np.sqrt(beta_v))*(np.dot(B_g.reshape(1,len(B_g)*3), (initial_geom_num_list - geom_num_list).reshape(len(B_g)*3,1)))*self.DELTA*d**2
            
            new_d = float(max((new_adam_r / np.linalg.norm(new_adam_s ,ord=1)), d))
            move_vector = []

            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_d*new_adam_m[i]/(np.sqrt(new_adam_v[i])+Epsilon*d))
            
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, [new_d, new_adam_r, new_adam_s])
            return move_vector
        
        #AdaBound
        #arXiv:1902.09843v1
        def Adabound(geom_num_list, B_g):
            print("AdaBound")
            adam_count = self.Opt_params.adam_count
            move_vector = []
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            if adam_count == 1:
                adam_m = self.Opt_params.adam_m
                adam_v = np.zeros((len(geom_num_list),3,3))
            else:
                adam_m = self.Opt_params.adam_m
                adam_v = self.Opt_params.adam_v
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            V = adam_m*0.0
            Eta = adam_m*0.0
            Eta_hat = adam_m*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(B_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(np.dot(np.array([B_g[i]]).T, np.array([B_g[i]]))))
                V[i] = copy.copy(np.diag(new_adam_v[i]))
                
                Eta_hat[i] = copy.copy(np.clip(self.DELTA/np.sqrt(V[i]), 0.1 - (0.1/(1.0 - beta_v) ** (adam_count + 1)) ,0.1 + (0.1/(1.0 - beta_v) ** adam_count) ))
                Eta[i] = copy.copy(Eta_hat[i]/np.sqrt(adam_count))
                    
            for i in range(len(geom_num_list)):
                move_vector.append(Eta[i] * new_adam_m[i])
            
            return move_vector    
        
        #Adadelta
        #arXiv:1212.5701v1
        def Adadelta(geom_num_list, B_g):#delta is not required. This method tends to converge local minima.
            print("Adadelta")
            rho = 0.9
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            Epsilon = 1e-06
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(rho * adam_m[i] + (1.0 - rho)*(B_g[i]) ** 2)
            move_vector = []
            
            for i in range(len(geom_num_list)):
                if adam_count > 1:
                    move_vector.append(B_g[i] * (np.sqrt(np.square(adam_v).mean()) + Epsilon)/(np.sqrt(np.square(new_adam_m).mean()) + Epsilon))
                else:
                    move_vector.append(B_g[i])
            if abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) > self.RMS_FORCE_THRESHOLD:
                move_vector = B_g

            for i in range(len(geom_num_list)):
                new_adam_v[i] = copy.copy(rho * adam_v[i] + (1.0 - rho) * (move_vector[i]) ** 2)
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        

        def Perturbation(move_vector):#This function is just for fun. Thus, it is no scientific basis.
            """Langevin equation"""
            Boltzmann_constant = 3.16681*10**(-6) # hartree/K
            damping_coefficient = 10.0
	
            temperature = self.temperature
            perturbation = self.DELTA * np.sqrt(2.0 * damping_coefficient * Boltzmann_constant * temperature) * np.random.normal(loc=0.0, scale=1.0, size=3*len(move_vector)).reshape(len(move_vector), 3)

            return perturbation

        move_vector_list = []
     
        #---------------------------------
        
        for opt_method in opt_method_list:
            # group of steepest descent
            if opt_method == "RADAM":
                tmp_move_vector = RADAM(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adam":
                tmp_move_vector = Adam(geom_num_list, B_g) 
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adadelta":
                tmp_move_vector = Adadelta(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdamW":
                tmp_move_vector = AdamW(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaDiff":
                tmp_move_vector = AdaDiff(geom_num_list, B_g, pre_B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adafactor":
                tmp_move_vector = Adafactor(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaBelief":
                tmp_move_vector = AdaBelief(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adabound":
                tmp_move_vector = Adabound(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "EVE":
                tmp_move_vector = EVE(geom_num_list, B_g, B_e, pre_B_e, pre_B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Prodigy":
                tmp_move_vector = Prodigy(geom_num_list, B_g, initial_geom_num_list)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaMax":
                tmp_move_vector = AdaMax(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "NAdam":    
                tmp_move_vector = NAdam(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "FIRE":
                tmp_move_vector = FIRE(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
            
            elif opt_method == "third_order_momentum_Adam":
                tmp_move_vector = third_order_momentum_Adam(geom_num_list, B_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "CG":
                if iter != 0:
                    tmp_move_vector = conjugate_gradient_descent(geom_num_list, pre_move_vector, B_g, pre_B_g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    self.Opt_params.adam_v = copy.copy(-1 * B_g)
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            
            # group of quasi-Newton method
            
            
            elif opt_method == "BFGS":
                if iter != 0:
                    tmp_move_vector = BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_BFGS":
                if iter != 0:
                    tmp_move_vector = RFO_BFGS_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "FSB":
                if iter != 0:
                    tmp_move_vector = FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "RFO_FSB":
                if iter != 0:
                    tmp_move_vector = RFO_FSB_quasi_newton_method(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mBFGS":
                if iter != 0:
                    tmp_move_vector = momentum_based_BFGS(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mFSB":
                if iter != 0:
                    tmp_move_vector = momentum_based_FSB(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mBFGS":
                if iter != 0:
                    tmp_move_vector = RFO_momentum_based_BFGS(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mFSB":
                if iter != 0:
                    tmp_move_vector = RFO_momentum_based_FSB(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, pre_g, g)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*B_g 
                    move_vector_list.append(tmp_move_vector)
            else:
                print("optimization method that this program is not sppourted is selected... thus, default method is selected.")
                tmp_move_vector = AdaBelief(geom_num_list, B_g)
        #---------------------------------
        
        if len(move_vector_list) > 1:
            if abs(B_g.max()) < self.MAX_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) < self.RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method: ", opt_method_list[1])
            else:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method: ", opt_method_list[0])
        else:
            move_vector = copy.copy(move_vector_list[0])
        
        perturbation = Perturbation(move_vector)
        
        move_vector += perturbation
        print("perturbation: ", np.linalg.norm(perturbation))
        
        if iter % self.FC_COUNT == 0 and self.FC_COUNT != -1 and self.saddle_order < 1:
            self.trust_radii = 0.01
        elif self.FC_COUNT == -1:
            self.trust_radii = 1.0
        else:
            self.trust_radii = update_trust_radii(self.trust_radii, B_e, pre_B_e)
            
            
        if np.linalg.norm(move_vector) > self.trust_radii:
            move_vector = self.trust_radii * move_vector/np.linalg.norm(move_vector)
        print("trust radii: ", self.trust_radii)
        print("step  radii: ", np.linalg.norm(move_vector))
        
        hess_eigenvalue, _ = np.linalg.eig(self.Model_hess.model_hess)
        
        print("NORMAL MODE EIGENVALUE:\n",np.sort(hess_eigenvalue),"\n")
        
        #---------------------------------
        new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms
        
        return new_geometry, np.array(move_vector, dtype="float64"), self.Opt_params, self.Model_hess, self.trust_radii
#---------------------

#---------------------
#bias potential calculation section
class LJRepulsivePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        return
    
    
    def calc_energy_scale(self, geom_num_list):#geom_num_list: torch.float64
        """
        # required variables: self.config["repulsive_potential_well_scale"], 
                             self.config["repulsive_potential_dist_scale"], 
                             self.config["repulsive_potential_Fragm_1"],
                             self.config["repulsive_potential_Fragm_2"]
                             self.config["element_list"]
        """
        energy = 0.0

        for i, j in itertools.product(self.config["repulsive_potential_Fragm_1"], self.config["repulsive_potential_Fragm_2"]):
            UFF_VDW_well_depth = math.sqrt(self.config["repulsive_potential_well_scale"]*UFF_VDW_well_depth_lib(self.config["element_list"][i-1]) + self.config["repulsive_potential_well_scale"]*UFF_VDW_well_depth_lib(self.config["element_list"][j-1]))
            UFF_VDW_distance = math.sqrt(UFF_VDW_distance_lib(self.config["element_list"][i-1])*self.config["repulsive_potential_dist_scale"] + UFF_VDW_distance_lib(self.config["element_list"][j-1])*self.config["repulsive_potential_dist_scale"])
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            energy += UFF_VDW_well_depth * ( -2 * ( UFF_VDW_distance / vector ) ** 6 + ( UFF_VDW_distance / vector ) ** 12)
            
        return energy
    
    def calc_energy_value(self, geom_num_list):#geom_num_list: torch.float64
        """
        # required variables: self.config["repulsive_potential_well_value"], 
                             self.config["repulsive_potential_dist_value"], 
                             self.config["repulsive_potential_Fragm_1"],
                             self.config["repulsive_potential_Fragm_2"]
                             self.config["element_list"]
        """
        energy = 0.0

        for i, j in itertools.product(self.config["repulsive_potential_Fragm_1"], self.config["repulsive_potential_Fragm_2"]):
            UFF_VDW_well_depth = self.config["repulsive_potential_well_value"]/self.hartree2kjmol
            UFF_VDW_distance = self.config["repulsive_potential_dist_value"]/self.bohr2angstroms
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            energy += UFF_VDW_well_depth * ( -2 * ( UFF_VDW_distance / vector ) ** 6 + ( UFF_VDW_distance / vector ) ** 12)
            
        return
    
    def calc_energy_scale_v2(self, geom_num_list):
        """
        # required variables: self.config["repulsive_potential_v2_well_scale"], 
                             self.config["repulsive_potential_v2_dist_scale"], 
                             self.config["repulsive_potential_v2_length"],
                             self.config["repulsive_potential_v2_const_rep"]
                             self.config["repulsive_potential_v2_const_attr"], 
                             self.config["repulsive_potential_v2_order_rep"], 
                             self.config["repulsive_potential_v2_order_attr"],
                             self.config["repulsive_potential_v2_center"]
                             self.config["repulsive_potential_v2_target"]
                             self.config["element_list"]
        """
        energy = 0.0
        
        LJ_pot_center = geom_num_list[self.config["repulsive_potential_v2_center"][1]-1] + (self.config["repulsive_potential_v2_length"]/self.bohr2angstroms) * (geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1] / torch.linalg.norm(geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1])) 
        for i in self.config["repulsive_potential_v2_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["repulsive_potential_v2_well_scale"]*UFF_VDW_well_depth_lib(self.config["element_list"][self.config["repulsive_potential_v2_center"][1]-1]) * UFF_VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(UFF_VDW_distance_lib(self.config["element_list"][self.config["repulsive_potential_v2_center"][1]-1])*self.config["repulsive_potential_v2_dist_scale"] * UFF_VDW_distance_lib(self.config["repulsive_potential_v2_center"][i-1]))
            
            vector = torch.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
            energy += UFF_VDW_well_depth * ( abs(self.config["repulsive_potential_v2_const_rep"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_rep"] -1 * abs(self.config["repulsive_potential_v2_const_attr"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_attr"])
            
        return energy
    def calc_energy_value_v2(self, geom_num_list):

        """
        # required variables: self.config["repulsive_potential_v2_well_value"], 
                             self.config["repulsive_potential_v2_dist_value"], 
                             self.config["repulsive_potential_v2_length"],
                             self.config["repulsive_potential_v2_const_rep"]
                             self.config["repulsive_potential_v2_const_attr"], 
                             self.config["repulsive_potential_v2_order_rep"], 
                             self.config["repulsive_potential_v2_order_attr"],
                             self.config["repulsive_potential_v2_center"]
                             self.config["repulsive_potential_v2_target"]
                             self.config["element_list"]
        """
        energy = 0.0
        
        LJ_pot_center = geom_num_list[self.config["repulsive_potential_v2_center"][1]-1] + (self.config["repulsive_potential_v2_length"]/self.bohr2angstroms) * (geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1] / torch.linalg.norm(geom_num_list[self.config["repulsive_potential_v2_length"][1]-1] - geom_num_list[self.config["repulsive_potential_v2_length"][0]-1])) 
        for i in self.config["repulsive_potential_v2_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["repulsive_potential_v2_well_value"]/self.hartree2kjmol * UFF_VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(self.config["repulsive_potential_v2_dist_value"]/self.bohr2angstroms * UFF_VDW_distance_lib(self.config["repulsive_potential_v2_center"][i-1]))
            
            vector = torch.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
            energy += UFF_VDW_well_depth * ( abs(self.config["repulsive_potential_v2_const_rep"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_rep"] -1 * abs(self.config["repulsive_potential_v2_const_attr"]) * ( UFF_VDW_distance / vector ) ** self.config["repulsive_potential_v2_order_attr"])
            
        return energy
    
 

    
    def calc_cone_potential_energy(self, geom_num_list):

        a_value = 1.0
        """
        # ref.  ACS Catal. 2022, 12, 7, 3752–3766
        # required variables: self.config["cone_potential_well_value"], 
                             self.config["cone_potential_dist_value"], 
                             self.config["cone_potential_cone_angle"],
                             self.config["cone_potential_center"], 
                             self.config["cone_potential_three_atoms"]
                             self.config["cone_potential_target"]   
                             self.config["element_list"]
        """
        apex_vector = geom_num_list[self.config["cone_potential_center"]-1] - (2.28/self.bohr2angstroms) * ((geom_num_list[self.config["cone_potential_three_atoms"][0]-1] + geom_num_list[self.config["cone_potential_three_atoms"][1]-1] + geom_num_list[self.config["cone_potential_three_atoms"][2]-1] -3.0 * geom_num_list[self.config["cone_potential_center"]-1]) / torch.linalg.norm(geom_num_list[self.config["cone_potential_three_atoms"][0]-1] + geom_num_list[self.config["cone_potential_three_atoms"][1]-1] + geom_num_list[self.config["cone_potential_three_atoms"][2]-1] -3.0 * geom_num_list[self.config["cone_potential_center"]-1]))
        cone_angle = torch.deg2rad(torch.tensor(self.config["cone_potential_cone_angle"], dtype=torch.float64))
        energy = 0.0
        for i in self.config["cone_potential_target"]:
            UFF_VDW_well_depth = math.sqrt(self.config["cone_potential_well_value"]/self.hartree2kjmol * UFF_VDW_well_depth_lib(self.config["element_list"][i-1]))
            UFF_VDW_distance = math.sqrt(self.config["cone_potential_dist_value"]/self.bohr2angstroms * UFF_VDW_distance_lib(self.config["element_list"][i-1]))
            s_a_length = (geom_num_list[i-1] - apex_vector).view(1,3)
            c_a_length = (geom_num_list[self.config["cone_potential_center"]-1] - apex_vector).view(1,3)
            sub_angle = torch.arccos((torch.matmul(c_a_length, s_a_length.T)) / (torch.linalg.norm(c_a_length) * torch.linalg.norm(s_a_length)))#rad
            dist = torch.linalg.norm(s_a_length)
            
            if sub_angle - cone_angle / 2 <= torch.pi / 2:
                length = (dist * torch.sin(sub_angle - cone_angle / 2)).view(1,1)
            
            else:
                length = dist.view(1,1)
            
            energy += 4 * UFF_VDW_well_depth * ((UFF_VDW_distance / (length + a_value * UFF_VDW_distance)) ** 12 - (UFF_VDW_distance / (length + a_value * UFF_VDW_distance)) ** 6)
            
        
        return energy
        
class AFIRPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["AFIR_gamma"], 
                             self.config["AFIR_Fragm_1"], 
                             self.config["AFIR_Fragm_2"],
                             self.config["element_list"]
        """
        """
        ###  Reference  ###
            Chem. Rec., 2016, 16, 2232
            J. Comput. Chem., 2018, 39, 233
            WIREs Comput. Mol. Sci., 2021, 11, e1538
        """
        R_0 = 3.8164/self.bohr2angstroms #ang.→bohr
        EPSIRON = 1.0061/self.hartree2kjmol #kj/mol→hartree
        if self.config["AFIR_gamma"] > 0.0 or self.config["AFIR_gamma"] < 0.0:
            alpha = (self.config["AFIR_gamma"]/self.hartree2kjmol) / ((2 ** (-1/6) - (1 + math.sqrt(1 + (abs(self.config["AFIR_gamma"]/self.hartree2kjmol) / EPSIRON))) ** (-1/6))*R_0) #hartree/Bohr
        else:
            alpha = 0.0
        A = 0.0
        B = 0.0
        
        p = 6.0

        for i, j in itertools.product(self.config["AFIR_Fragm_1"], self.config["AFIR_Fragm_2"]):
            R_i = covalent_radii_lib(self.config["element_list"][i-1])
            R_j = covalent_radii_lib(self.config["element_list"][j-1])
            vector = torch.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
            omega = ((R_i + R_j) / vector) ** p #no unit
            A += omega * vector
            B += omega
        
        energy = alpha*(A/B)#A/B:Bohr
        return energy #hartree
      
class StructKeepPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_pot_spring_const"], 
                             self.config["keep_pot_distance"], 
                             self.config["keep_pot_atom_pairs"],
                             
        """
        vector = torch.linalg.norm((geom_num_list[self.config["keep_pot_atom_pairs"][0]-1] - geom_num_list[self.config["keep_pot_atom_pairs"][1]-1]), ord=2)
        energy = 0.5 * self.config["keep_pot_spring_const"] * (vector - self.config["keep_pot_distance"]/self.bohr2angstroms) ** 2
        return energy #hartree

class StructAnharmonicKeepPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["anharmonic_keep_pot_spring_const"],
                              self.config["anharmonic_keep_pot_potential_well_depth"]
                              self.config["anharmonic_keep_pot_atom_pairs"]
                              self.config["anharmonic_keep_pot_distance"]

        """
        vector = torch.linalg.norm((geom_num_list[self.config["anharmonic_keep_pot_atom_pairs"][0]-1] - geom_num_list[self.config["anharmonic_keep_pot_atom_pairs"][1]-1]), ord=2)
        if self.config["anharmonic_keep_pot_potential_well_depth"] != 0.0:
            energy = self.config["anharmonic_keep_pot_potential_well_depth"] * ( 1.0 - torch.exp( - math.sqrt(self.config["anharmonic_keep_pot_spring_const"] / (2 * self.config["anharmonic_keep_pot_potential_well_depth"])) * (vector - self.config["anharmonic_keep_pot_distance"]/self.bohr2angstroms)) ) ** 2
        else:
            energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

        return energy

class StructKeepAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_angle_atom_pairs"],
                              self.config["keep_angle_spring_const"]
                              self.config["keep_angle_angle"]

        """
       
        vector1 = geom_num_list[self.config["keep_angle_atom_pairs"][0]-1] - geom_num_list[self.config["keep_angle_atom_pairs"][1]-1]
        vector2 = geom_num_list[self.config["keep_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_angle_atom_pairs"][1]-1]
        magnitude1 = torch.linalg.norm(vector1)
        magnitude2 = torch.linalg.norm(vector2)
        dot_product = torch.matmul(vector1, vector2)
        cos_theta = dot_product / (magnitude1 * magnitude2)
        theta = torch.arccos(cos_theta)
        energy = 0.5 * self.config["keep_angle_spring_const"] * (theta - torch.deg2rad(torch.tensor(self.config["keep_angle_angle"]))) ** 2
        return energy #hartree
        
    def calc_atom_dist_dependent_energy(self, geom_num_list):
        """
        # required variables: self.config["aDD_keep_angle_spring_const"] 
                              self.config["aDD_keep_angle_min_angle"] 
                              self.config["aDD_keep_angle_max_angle"]
                              self.config["aDD_keep_angle_base_dist"]
                              self.config["aDD_keep_angle_reference_atom"] 
                              self.config["aDD_keep_angle_center_atom"] 
                              self.config["aDD_keep_angle_atoms"]
        
        """
        energy = 0.0
        self.config["keep_angle_spring_const"] = self.config["aDD_keep_angle_spring_const"] 
        max_angle = torch.tensor(self.config["aDD_keep_angle_max_angle"])
        min_angle = torch.tensor(self.config["aDD_keep_angle_min_angle"])
        ref_dist = torch.linalg.norm(geom_num_list[self.config["aDD_keep_angle_center_atom"]-1] - geom_num_list[self.config["aDD_keep_angle_reference_atom"]-1]) / self.bohr2angstroms
        base_dist = self.config["aDD_keep_angle_base_dist"] / self.bohr2angstroms
        eq_angle = min_angle + ((max_angle - min_angle)/(1 + torch.exp(-(ref_dist - base_dist))))
        
        self.config["keep_angle_angle"] = eq_angle
        
        
        self.config["keep_angle_atom_pairs"] = [self.config["aDD_keep_angle_atoms"][0] , self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]]
        energy += self.calc_energy(geom_num_list)
        self.config["keep_angle_atom_pairs"] = [self.config["aDD_keep_angle_atoms"][2] , self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][1]]
        energy += self.calc_energy(geom_num_list)
        self.config["keep_angle_atom_pairs"] = [self.config["aDD_keep_angle_atoms"][0] , self.config["aDD_keep_angle_center_atom"], self.config["aDD_keep_angle_atoms"][2]]
        energy += self.calc_energy(geom_num_list)
    
        return energy

class StructKeepDihedralAnglePotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["keep_dihedral_angle_spring_const"],
                              self.config["keep_dihedral_angle_atom_pairs"]
                              self.config["keep_dihedral_angle_angle"]
                        
        """
        a1 = geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][1]-1] - geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][0]-1]
        a2 = geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][2]-1] - geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][1]-1]
        a3 = geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][3]-1] - geom_num_list[self.config["keep_dihedral_angle_atom_pairs"][2]-1]

        v1 = torch.cross(a1, a2)
        v1 = v1 / torch.linalg.norm(v1, ord=2)
        v2 = torch.cross(a2, a3)
        v2 = v2 / torch.linalg.norm(v2, ord=2)
        angle = torch.arccos((v1*v2).sum(-1) / ((v1**2).sum() * (v2**2).sum())**0.5)

        energy = 0.5 * self.config["keep_dihedral_angle_spring_const"] * (angle - torch.deg2rad(torch.tensor(self.config["keep_dihedral_angle_angle"]))) ** 2
        
        return energy #hartree    

class VoidPointPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["void_point_pot_spring_const"],
                              self.config["void_point_pot_atoms"]
                              self.config["void_point_pot_coord"]  #need to convert tensor type 
                              
                              self.config["void_point_pot_distance"]
                              self.config["void_point_pot_order"]
                        
        """
        vector = torch.linalg.norm((geom_num_list[self.config["void_point_pot_atoms"]-1] - self.config["void_point_pot_coord"]), ord=2)
        energy = (1 / self.config["void_point_pot_order"]) * self.config["void_point_pot_spring_const"] * (vector - self.config["void_point_pot_distance"]/self.bohr2angstroms) ** self.config["void_point_pot_order"]
        return energy #hartree

class WellPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        return
    
    def calc_energy(self, geom_num_list):
        """
        # required variables: self.config["well_pot_wall_energy"]
                              self.config["well_pot_fragm_1"]
                              self.config["well_pot_fragm_2"]
                              self.config["well_pot_limit_dist"]
                              
                              
        """
        fragm_1_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["well_pot_fragm_1"]:
            fragm_1_center = fragm_1_center + geom_num_list[i-1]
        
        fragm_1_center = fragm_1_center / len(self.config["well_pot_fragm_1"])
        
        fragm_2_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["well_pot_fragm_2"]:
            fragm_2_center = fragm_2_center + geom_num_list[i-1]
        
        fragm_2_center = fragm_2_center / len(self.config["well_pot_fragm_2"])        
        
        vec_norm = torch.linalg.norm(fragm_1_center - fragm_2_center) 
        a = float(self.config["well_pot_limit_dist"][0]) / self.bohr2angstroms
        b = float(self.config["well_pot_limit_dist"][1]) / self.bohr2angstroms
        c = float(self.config["well_pot_limit_dist"][2]) / self.bohr2angstroms
        d = float(self.config["well_pot_limit_dist"][3]) / self.bohr2angstroms
        short_dist_linear_func_slope = 0.5 / (b - a)
        short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
        long_dist_linear_func_slope = 0.5 / (c - d)
        long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

        x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
        x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

        if vec_norm <= a:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
            
        elif a < vec_norm and vec_norm <= b:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                
        elif b < vec_norm and vec_norm < c:
            energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
            
        elif c <= vec_norm and vec_norm < d:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
            
        elif d <= vec_norm:
            energy = (self.config["well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
            
        else:
            print("well pot error")
            raise "well pot error"
        #print(energy)
        return energy
        
    def calc_energy_wall(self, geom_num_list):
        """
        # required variables: self.config["wall_well_pot_wall_energy"]
                              self.config["wall_well_pot_direction"] 
                              self.config["wall_well_pot_limit_dist"]
                              self.config["wall_well_pot_target"]
                              
                              
        """
        
        if self.config["wall_well_pot_direction"] == "x":
            direction_num = 0
        elif self.config["wall_well_pot_direction"] == "y":
            direction_num = 1
        elif self.config["wall_well_pot_direction"] == "z":
            direction_num = 2
                     
        
        
        energy = 0.0
        for i in self.config["wall_well_pot_target"]:

            vec_norm = abs(torch.linalg.norm(geom_num_list[i-1][direction_num])) 
                     
            a = float(self.config["wall_well_pot_limit_dist"][0]) / self.bohr2angstroms
            b = float(self.config["wall_well_pot_limit_dist"][1]) / self.bohr2angstroms
            c = float(self.config["wall_well_pot_limit_dist"][2]) / self.bohr2angstroms
            d = float(self.config["wall_well_pot_limit_dist"][3]) / self.bohr2angstroms
            short_dist_linear_func_slope = 0.5 / (b - a)
            short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
            long_dist_linear_func_slope = 0.5 / (c - d)
            long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

            x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
            x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

            if vec_norm <= a:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
                
            elif a < vec_norm and vec_norm <= b:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                    
            elif b < vec_norm and vec_norm < c:
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["wall_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
        
    def calc_energy_vp(self, geom_num_list):
        """
        # required variables: self.config["void_point_well_pot_wall_energy"]
                              self.config["void_point_well_pot_coordinate"] 
                              self.config["void_point_well_pot_limit_dist"]
                              self.config["void_point_well_pot_target"]
                              
                              
        """
        self.config["void_point_well_pot_coordinate"]  = torch.tensor(self.config["void_point_well_pot_coordinate"], dtype=torch.float64)

        
        energy = 0.0
        for i in self.config["void_point_well_pot_target"]:

            vec_norm = torch.linalg.norm(geom_num_list[i-1] - self.config["void_point_well_pot_coordinate"]) 
                     
            a = float(self.config["void_point_well_pot_limit_dist"][0]) / self.bohr2angstroms
            b = float(self.config["void_point_well_pot_limit_dist"][1]) / self.bohr2angstroms
            c = float(self.config["void_point_well_pot_limit_dist"][2]) / self.bohr2angstroms
            d = float(self.config["void_point_well_pot_limit_dist"][3]) / self.bohr2angstroms
            short_dist_linear_func_slope = 0.5 / (b - a)
            short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
            long_dist_linear_func_slope = 0.5 / (c - d)
            long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

            x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
            x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

            if vec_norm <= a:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
                
            elif a < vec_norm and vec_norm <= b:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                    
            elif b < vec_norm and vec_norm < c:
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["void_point_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
        
    def calc_energy_around(self, geom_num_list):
        """
        # required variables: self.config["around_well_pot_wall_energy"]
                              self.config["around_well_pot_center"] 
                              self.config["around_well_pot_limit_dist"]
                              self.config["around_well_pot_target"]
                              
                              
        """
        geom_center_coord = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        for i in self.config["around_well_pot_center"]:
            geom_center_coord = geom_center_coord + geom_num_list[i-1]
        geom_center_coord = geom_center_coord/len(self.config["around_well_pot_center"])
        energy = 0.0
        for i in self.config["around_well_pot_target"]:

            vec_norm = torch.linalg.norm(geom_num_list[i-1] - geom_center_coord) 
                     
            a = float(self.config["around_well_pot_limit_dist"][0]) / self.bohr2angstroms
            b = float(self.config["around_well_pot_limit_dist"][1]) / self.bohr2angstroms
            c = float(self.config["around_well_pot_limit_dist"][2]) / self.bohr2angstroms
            d = float(self.config["around_well_pot_limit_dist"][3]) / self.bohr2angstroms
            short_dist_linear_func_slope = 0.5 / (b - a)
            short_dist_linear_func_intercept = 1.0 - 0.5 * b / (b - a) 
            long_dist_linear_func_slope = 0.5 / (c - d)
            long_dist_linear_func_intercept = 1.0 - 0.5 * c / (c - d) 

            x_short = short_dist_linear_func_slope * vec_norm + short_dist_linear_func_intercept
            x_long = long_dist_linear_func_slope * vec_norm + long_dist_linear_func_intercept

            if vec_norm <= a:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_short + 2.875)
                
            elif a < vec_norm and vec_norm <= b:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_short ** 3 + 30.0 * x_short ** 4 - 12.0 * x_short ** 5)
                    
            elif b < vec_norm and vec_norm < c:
                energy += torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
                
            elif c <= vec_norm and vec_norm < d:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (2.0 - 20.0 * x_long ** 3 + 30.0 * x_long ** 4 - 12.0 * x_long ** 5)
                
            elif d <= vec_norm:
                energy += (self.config["around_well_pot_wall_energy"] / self.hartree2kjmol) * (-3.75 * x_long + 2.875)
                
            else:
                print("well pot error")
                raise "well pot error"
                
 
        #print(energy)
        return energy
            
class DSAFIRPotential:
    def __init__(self, **kwarg):
        self.config = kwarg
        self.DS_BETA = 0.03 #hartree
        return
         
    def calc_energy(self, q_geometry): #tensor
        """
        # required variables
        self.config["p_geometry"]
        self.config["initial_geometry"]
        self.config["q_gradient"]
        # notice
        condition to work nearly correctly 
        ・optmethod = Adabelief
        ・if Z > 0.0:
            Y = Z / (1 + Z) + 0.5 -> + 0 is correct. However, it is not working well.
        else:
            Y = 0.5
        -> Y = 0 is correct. However, it is not working well.
        """
        delta_q_p_geo = q_geometry - self.config["p_geometry"]
        delta_q_ini_geo = q_geometry - self.config["initial_geometry"]
        
        
        Z = (torch.linalg.norm(delta_q_ini_geo) / torch.linalg.norm(delta_q_p_geo)) + ((torch.matmul(delta_q_p_geo.reshape(1, len(self.config["q_gradient"])*3), delta_q_ini_geo.reshape(len(self.config["q_gradient"])*3, 1))) / (torch.linalg.norm(delta_q_ini_geo) * torch.linalg.norm(delta_q_p_geo)))
        if Z > 0.0:
            Y = Z / (1 + Z) + 0.5
        else:
            Y = 0.5
                
        u = Y * (delta_q_p_geo.reshape(len(self.config["q_gradient"])*3, 1)) / torch.linalg.norm(delta_q_p_geo) - (1.0 - Y) * (delta_q_ini_geo.reshape(len(self.config["q_gradient"])*3, 1) / torch.linalg.norm(delta_q_ini_geo))

        X = (self.DS_BETA / torch.linalg.norm(u)) - (torch.matmul(self.config["q_gradient"].reshape(1, len(self.config["q_gradient"])*3), u) / (torch.linalg.norm(u) ** 2))
        ene = X * Y * torch.linalg.norm(delta_q_p_geo) -1 * X * (1.0 - Y) * torch.linalg.norm(delta_q_ini_geo) 
        
        return ene
        
class BiasPotentialCalculation:
    def __init__(self, Model_hess, FC_COUNT):
        torch.set_printoptions(precision=12)
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.Model_hess = Model_hess
        self.FC_COUNT = FC_COUNT
        self.JOBID = random.randint(0, 1000000)
    
    def ndarray2tensor(self, ndarray):
        tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64, requires_grad=True))

        return tensor

    def ndarray2nogradtensor(self, ndarray):
        tensor = copy.copy(torch.tensor(ndarray, dtype=torch.float64))

        return tensor

    def tensor2ndarray(self, tensor):
        ndarray = copy.copy(tensor.detach().numpy())
        return ndarray
    
    def main(self, e, g, geom_num_list, element_list,  force_data, pre_B_g, iter, initial_geom_num_list):
        numerical_derivative_delta = 0.0001 #unit:Bohr
        
        #g:hartree/Bohr
        #e:hartree
        #geom_num_list:Bohr

  
        #--------------------------------------------------
        B_e = e
        BPA_grad_list = g*0.0
        BPA_hessian = np.zeros((3*len(g), 3*len(g)))
        #debug_delta_BPA_grad_list = g*0.0
        geom_num_list = self.ndarray2tensor(geom_num_list)
        #print("rpv2")   torch.tensor(*** , dtype=torch.float64, requires_grad=True)




        for i in range(len(force_data["repulsive_potential_v2_well_scale"])):
            if force_data["repulsive_potential_v2_well_scale"][i] != 0.0:
                if force_data["repulsive_potential_v2_unit"][i] == "scale":
                    LJRP = LJRepulsivePotential(repulsive_potential_v2_well_scale=force_data["repulsive_potential_v2_well_scale"][i], 
                                                repulsive_potential_v2_dist_scale=force_data["repulsive_potential_v2_dist_scale"][i], 
                                                repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                                repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                                repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"], 
                                                repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"], 
                                                repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"],
                                                repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"],
                                                repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_scale_v2(geom_num_list)
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_scale_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_scale_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

                elif force_data["repulsive_potential_v2_unit"][i] == "value":
                    LJRP = LJRepulsivePotential(repulsive_potential_v2_well_value=force_data["repulsive_potential_v2_well_scale"][i], 
                                                repulsive_potential_v2_dist_value=force_data["repulsive_potential_v2_dist_scale"][i], 
                                                repulsive_potential_v2_length=force_data["repulsive_potential_v2_length"][i],
                                                repulsive_potential_v2_const_rep=force_data["repulsive_potential_v2_const_rep"][i],
                                                repulsive_potential_v2_const_attr=force_data["repulsive_potential_v2_const_attr"], 
                                                repulsive_potential_v2_order_rep=force_data["repulsive_potential_v2_order_rep"], 
                                                repulsive_potential_v2_order_attr=force_data["repulsive_potential_v2_order_attr"],
                                                repulsive_potential_v2_center=force_data["repulsive_potential_v2_center"],
                                                repulsive_potential_v2_target=force_data["repulsive_potential_v2_target"],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_value_v2(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_value_v2)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_value_v2)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    print("error -rpv2")
                    raise "error -rpv2"
            else:
                pass

        #print("rp")
        #------------------
        for i in range(len(force_data["repulsive_potential_dist_scale"])):
            if force_data["repulsive_potential_well_scale"][i] != 0.0:
                if force_data["repulsive_potential_unit"][i] == "scale":
                    LJRP = LJRepulsivePotential(repulsive_potential_well_scale=force_data["repulsive_potential_well_scale"][i], 
                                                repulsive_potential_dist_scale=force_data["repulsive_potential_dist_scale"][i], 
                                                repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                                repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_scale(geom_num_list)
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_scale)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_scale)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    
                    
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

                elif force_data["repulsive_potential_unit"][i] == "value":
                    LJRP = LJRepulsivePotential(repulsive_potential_well_value=force_data["repulsive_potential_well_scale"][i], 
                                                repulsive_potential_dist_value=force_data["repulsive_potential_dist_scale"][i], 
                                                repulsive_potential_Fragm_1=force_data["repulsive_potential_Fragm_1"][i],
                                                repulsive_potential_Fragm_2=force_data["repulsive_potential_Fragm_2"][i],
                                                element_list=element_list,
                                                jobid=self.JOBID)
                    
                    B_e += LJRP.calc_energy_value(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_energy_value)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(LJRP.calc_energy_value)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    print("error -rpv2")
                    raise "error -rpv2"
            else:
                pass
        
    
        
        #------------------
        for i in range(len(force_data["cone_potential_well_value"])):
            if force_data["cone_potential_well_value"][i] != 0.0:
              
                LJRP = LJRepulsivePotential(cone_potential_well_value=force_data["cone_potential_well_value"][i], 
                                            cone_potential_dist_value=force_data["cone_potential_dist_value"][i], 
                                            cone_potential_cone_angle=force_data["cone_potential_cone_angle"][i],
                                            cone_potential_center=force_data["cone_potential_center"][i],
                                            cone_potential_three_atoms=force_data["cone_potential_three_atoms"][i],
                                            cone_potential_target=force_data["cone_potential_target"][i],
                                            element_list=element_list
                                            )

                B_e += LJRP.calc_cone_potential_energy(geom_num_list)
                tensor_BPA_grad = torch.func.jacfwd(LJRP.calc_cone_potential_energy)(geom_num_list).view(len(geom_num_list), 3)
                
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(LJRP.calc_cone_potential_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))

                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
       
        #------------------
        
        for i in range(len(force_data["keep_pot_spring_const"])):
            if force_data["keep_pot_spring_const"][i] != 0.0:
                SKP = StructKeepPotential(keep_pot_spring_const=force_data["keep_pot_spring_const"][i], 
                                            keep_pot_distance=force_data["keep_pot_distance"][i], 
                                            keep_pot_atom_pairs=force_data["keep_pot_atom_pairs"][i])
                
                B_e += SKP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SKP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(SKP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
            else:
                pass
        #print("akp")
        #------------------        
        for i in range(len(force_data["anharmonic_keep_pot_spring_const"])):
            if force_data["anharmonic_keep_pot_spring_const"][i] != 0.0:
                SAKP = StructAnharmonicKeepPotential(anharmonic_keep_pot_spring_const=force_data["anharmonic_keep_pot_spring_const"][i], 
                                            anharmonic_keep_pot_potential_well_depth=force_data["anharmonic_keep_pot_potential_well_depth"][i], 
                                            anharmonic_keep_pot_atom_pairs=force_data["anharmonic_keep_pot_atom_pairs"][i],
                                            anharmonic_keep_pot_distance=force_data["anharmonic_keep_pot_distance"][i])
                
                B_e += SAKP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(SAKP.calc_energy)(geom_num_list)
            
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                
                tensor_BPA_hessian = torch.func.hessian(SAKP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
               
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
     
        #------------------
        

        #print("wp")
        for i in range(len(force_data["well_pot_wall_energy"])):
            if force_data["well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(well_pot_wall_energy=force_data["well_pot_wall_energy"][i], 
                                            well_pot_fragm_1=force_data["well_pot_fragm_1"][i], 
                                            well_pot_fragm_2=force_data["well_pot_fragm_2"][i], 
                                            well_pot_limit_dist=force_data["well_pot_limit_dist"][i])
                
                
                B_e += WP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------
        

        #print("wp")
        for i in range(len(force_data["wall_well_pot_wall_energy"])):
            if force_data["wall_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(wall_well_pot_wall_energy=force_data["wall_well_pot_wall_energy"][i],
                                            wall_well_pot_direction=force_data["wall_well_pot_direction"][i], 
                                            wall_well_pot_limit_dist=force_data["wall_well_pot_limit_dist"][i],
                                            wall_well_pot_target=force_data["wall_well_pot_target"][i])
                
                B_e += WP.calc_energy_wall(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_wall)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_wall)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------
        

        #print("wp")
        for i in range(len(force_data["void_point_well_pot_wall_energy"])):
            if force_data["void_point_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(void_point_well_pot_wall_energy=force_data["void_point_well_pot_wall_energy"][i], 
                                            void_point_well_pot_coordinate=force_data["void_point_well_pot_coordinate"][i], 
                                            void_point_well_pot_limit_dist=force_data["void_point_well_pot_limit_dist"][i],
                                            void_point_well_pot_target=force_data["void_point_well_pot_target"][i])
                
                B_e += WP.calc_energy_vp(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_vp)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_vp)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
                
            
        #------------------    

        for i in range(len(force_data["around_well_pot_wall_energy"])):
            if force_data["around_well_pot_wall_energy"][i] != 0.0:
                WP = WellPotential(around_well_pot_wall_energy=force_data["around_well_pot_wall_energy"][i], 
                                            around_well_pot_center=force_data["around_well_pot_center"][i], 
                                            around_well_pot_limit_dist=force_data["around_well_pot_limit_dist"][i],
                                            around_well_pot_target=force_data["around_well_pot_target"][i])
                
                B_e += WP.calc_energy_around(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(WP.calc_energy_around)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                tensor_BPA_hessian = torch.func.hessian(WP.calc_energy_around)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                
            else:
                pass
                
            
        #------------------   
        
        if len(geom_num_list) > 2:
            for i in range(len(force_data["keep_angle_spring_const"])):
                if force_data["keep_angle_spring_const"][i] != 0.0:
                    SKAngleP = StructKeepAnglePotential(keep_angle_atom_pairs=force_data["keep_angle_atom_pairs"][i], 
                                                keep_angle_spring_const=force_data["keep_angle_spring_const"][i], 
                                                keep_angle_angle=force_data["keep_angle_angle"][i])
                    
                    B_e += SKAngleP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKAngleP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKAngleP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        
        if len(geom_num_list) > 2:
            for i in range(len(force_data["aDD_keep_angle_spring_const"])):
                if force_data["aDD_keep_angle_spring_const"][i] != 0.0:
                    aDDKAngleP = StructKeepAnglePotential(aDD_keep_angle_spring_const=force_data["aDD_keep_angle_spring_const"][i], 
                                                aDD_keep_angle_min_angle=force_data["aDD_keep_angle_min_angle"][i], 
                                                aDD_keep_angle_max_angle=force_data["aDD_keep_angle_max_angle"][i],
                                                aDD_keep_angle_base_dist=force_data["aDD_keep_angle_base_dist"][i],
                                                aDD_keep_angle_reference_atom=force_data["aDD_keep_angle_reference_atom"][i],
                                                aDD_keep_angle_center_atom=force_data["aDD_keep_angle_center_atom"][i],
                                                aDD_keep_angle_atoms=force_data["aDD_keep_angle_atoms"][i])

                    B_e += aDDKAngleP.calc_atom_dist_dependent_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(aDDKAngleP.calc_atom_dist_dependent_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(aDDKAngleP.calc_atom_dist_dependent_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)

        else:
            pass
        
        #------------------
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_dihedral_angle_spring_const"])):
                if force_data["keep_dihedral_angle_spring_const"][i] != 0.0:
                    SKDAP = StructKeepDihedralAnglePotential(keep_dihedral_angle_spring_const=force_data["keep_dihedral_angle_spring_const"][i], 
                                                keep_dihedral_angle_atom_pairs=force_data["keep_dihedral_angle_atom_pairs"][i], 
                                                keep_dihedral_angle_angle=force_data["keep_dihedral_angle_angle"][i])
                    
                    B_e += SKDAP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(SKDAP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(SKDAP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
                else:
                    pass
        else:
            pass

        #------------------
        for i in range(len(force_data["void_point_pot_spring_const"])):
            if force_data["void_point_pot_spring_const"][i] != 0.0:
                for j in force_data["void_point_pot_atoms"][i]:
                    VPP = VoidPointPotential(void_point_pot_spring_const=force_data["void_point_pot_spring_const"][i], 
                                            void_point_pot_atoms=j, 
                                            void_point_pot_coord=self.ndarray2tensor(np.array(force_data["void_point_pot_coord"][i], dtype="float64")),
                                            void_point_pot_distance=force_data["void_point_pot_distance"][i],
                                            void_point_pot_order=force_data["void_point_pot_order"][i])
                                            
                    
                    B_e += VPP.calc_energy(geom_num_list)
                    
                    tensor_BPA_grad = torch.func.jacfwd(VPP.calc_energy)(geom_num_list)
                    BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)

                    tensor_BPA_hessian = torch.func.hessian(VPP.calc_energy)(geom_num_list)
                    tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                    BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
              
            else:
                pass
        
        #------------------
        for i in range(len(force_data["AFIR_gamma"])):
            if force_data["AFIR_gamma"][i] != 0.0:
                AP = AFIRPotential(AFIR_gamma=force_data["AFIR_gamma"][i], 
                                            AFIR_Fragm_1=force_data["AFIR_Fragm_1"][i], 
                                            AFIR_Fragm_2=force_data["AFIR_Fragm_2"][i],
                                            element_list=element_list)
                
                B_e += AP.calc_energy(geom_num_list)
                
                tensor_BPA_grad = torch.func.jacfwd(AP.calc_energy)(geom_num_list)
                BPA_grad_list += self.tensor2ndarray(tensor_BPA_grad)
                
                tensor_BPA_hessian = torch.func.hessian(AP.calc_energy)(geom_num_list)
                tensor_BPA_hessian = torch.reshape(tensor_BPA_hessian, (len(geom_num_list)*3, len(geom_num_list)*3))
                BPA_hessian += self.tensor2ndarray(tensor_BPA_hessian)
            else:
                pass
        #------------------        
        B_g = g + BPA_grad_list

        
        
        B_e = B_e.item()
        #new_geometry:ang. 
        #B_e:hartree

        return BPA_grad_list, B_e, B_g, BPA_hessian
        
    def calc_DSAFIRPotential(self, q_e, q_g, q_geom_num_list, p_geom_num_list, initial_q_geom_num_list):#calclate bais optential for DS_AFIR
        q_geometry = self.ndarray2tensor(q_geom_num_list)
        p_geometry = self.ndarray2nogradtensor(p_geom_num_list)
        initial_geometry = self.ndarray2nogradtensor(initial_q_geom_num_list)
        q_gradient = self.ndarray2nogradtensor(q_g)
        DSAP = DSAFIRPotential(p_geometry=p_geometry, initial_geometry=initial_geometry, q_gradient=q_gradient)
        DSAFIR_e = q_e + DSAP.calc_energy(q_geometry)
        DSAFIR_e = DSAFIR_e.item()
        
        tmp_grad = torch.func.jacfwd(DSAP.calc_energy)(q_geometry)
        DSAFIR_g = self.tensor2ndarray(tmp_grad)[0][0]

        tmp_hessian = torch.func.hessian(DSAP.calc_energy)(q_geometry)
        tmp_hessian = torch.reshape(tmp_hessian, (len(q_geometry)*3, len(q_geometry)*3))
        DSAFIR_hessian = self.tensor2ndarray(tmp_hessian)
        #print(DSAFIR_e, DSAFIR_g, DSAFIR_hessian)
        return DSAFIR_e, DSAFIR_g, DSAFIR_hessian 
#end of bias pot section 
#----------------

class FileIO:
    def __init__(self, folder_dir, file):
        self.BPA_FOLDER_DIRECTORY = folder_dir
        self.START_FILE = file
        return
        
    def make_geometry_list(self, args_electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
 
        with open(self.START_FILE, "r") as f:
            words = f.readlines()
            
        start_data = []
        for word in words:
            start_data.append(word.split())
        
        if len(start_data[1]) == 2:#(charge ex. 0) (spin ex. 1)
            electric_charge_and_multiplicity = start_data[1]
            
        else:
            electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
            
        element_list = []
            


        for i in range(2, len(start_data)):
            element_list.append(start_data[i][0])
                
        geometry_list.append(start_data)


        return geometry_list, element_list, electric_charge_and_multiplicity

    def make_geometry_list_for_pyscf(self):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
 
        with open(self.START_FILE,"r") as f:
            words = f.readlines()
            
        start_data = []
        for word in words[2:]:
            start_data.append(word.split())
            
        element_list = []
            


        for i in range(len(start_data)):
            element_list.append(start_data[i][0])
                
        geometry_list.append(start_data)


        return geometry_list, element_list

    def make_geometry_list_2(self, new_geometry, element_list, electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
        
        geometry_list = []
        print("\ngeometry:")
        new_data = [electric_charge_and_multiplicity]
        for num, geometry in enumerate(new_geometry):
           
            geometry = list(map(str, geometry))
            geometry = [element_list[num]] + geometry
            new_data.append(geometry)
            print(" ".join(geometry))
            
        geometry_list.append(new_data)
        print("")
        return geometry_list
        
    def make_geometry_list_2_for_pyscf(self, new_geometry, element_list):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
        print("\ngeometry:")
        geometry_list = []

        new_data = []
        for num, geometry in enumerate(new_geometry):
           
            geometry = list(map(str, geometry))
            geometry = [element_list[num]] + geometry
            new_data.append(geometry)
            print(" ".join(geometry))
            
        geometry_list.append(new_data)
        print("")
        return geometry_list
        
    def make_psi4_input_file(self, geometry_list, iter):
        """structure updated geometry is saved."""
        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory

    def make_pyscf_input_file(self, geometry_list, iter):
        """structure updated geometry is saved."""
        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                w.write(str(len(geometry))+"\n\n")
                for rows in geometry:   
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory
        
    def xyz_file_make_for_pyscf(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")  
        #print(file_list,"\n")
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
                with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w:
                    atom_num = len(sample)-2
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                
                for i in sample[2:]:
                    with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w2:
                        w2.write(i)
        print("\ngeometry collection is completed...\n")
        return
        
    def xyz_file_make(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")  
        step_num = len(file_list)
        for m, file in enumerate(file_list[1:], 1):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
            with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            
            
            for i in sample[1:]:
                with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w2:
                    w2.write(i)
                    
            if m == step_num - 1:
                with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_optimized.xyz","w") as w3:
                    w3.write(str(atom_num)+"\n")
                    w3.write("Optimized Structure\n")
                    for i in sample[1:]:
                        w3.write(i)
        print("\ngeometry collection is completed...\n")
        return
        
    def xyz_file_make_for_DSAFIR(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]_reactant/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]_reactant/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]_product/*.xyz")[::-1]   
        #print(file_list,"\n")
        
        
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
            with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            del sample[0]
            for i in sample:
                with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w2:
                    w2.write(i)
        print("\ngeometry collection is completed...\n")
        return
    
    def argrelextrema_txt_save(self, save_list, name, min_max):
        NUM_LIST = [i for i in range(len(save_list))]
        if min_max == "max":
            local_max_energy_list_index = argrelextrema(np.array(save_list), np.greater)
            with open(self.BPA_FOLDER_DIRECTORY+name+".txt","w") as f:
                for j in local_max_energy_list_index[0].tolist():
                    f.write(str(NUM_LIST[j])+"\n")
        elif min_max == "min":
            inverse_energy_list = (-1)*np.array(save_list, dtype="float64")
            local_min_energy_list_index = argrelextrema(np.array(inverse_energy_list), np.greater)
            with open(self.BPA_FOLDER_DIRECTORY+name+".txt","w") as f:
                for j in local_min_energy_list_index[0].tolist():
                    f.write(str(NUM_LIST[j])+"\n")
        else:
            print("error")
    
        return
        
        
    def make_psi4_input_file_for_DSAFIR(self, geometry_list, iter, mode):
        """structure updated geometry is saved."""

        if mode == "r":
            file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)+"_reactant"
            try:
                os.mkdir(file_directory)
            except:
                pass


            for y, geometry in enumerate(geometry_list):
                with open(file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                    for rows in geometry:
                        for row in rows:
                            w.write(str(row))
                            w.write(" ")
                        w.write("\n")
        elif mode == "p":
            file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)+"_product"
            try:
                os.mkdir(file_directory)
            except:
                pass


            for y, geometry in enumerate(geometry_list):
                with open(file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                    for rows in geometry:
                        for row in rows:
                            w.write(str(row))
                            w.write(" ")
                        w.write("\n")
        else:
            print("unknown mode error")
            raise


        return file_directory
        
        
    def make_geometry_list_for_DSAFIR(self, mode):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
        start_data = []
        element_list = []
        if mode == "r":
            with open(self.START_FILE[:-4]+"_reactant.xyz","r") as f:
                words = f.readlines()
            for word in words:
                start_data.append(word.split())
            electric_charge_and_multiplicity = start_data[0]

            for i in range(1, len(start_data)):
                element_list.append(start_data[i][0])
            geometry_list.append(start_data)

        elif mode == "p":
            with open(self.START_FILE[:-4]+"_product.xyz","r") as f:
                words = f.readlines()

            for word in words:
                start_data.append(word.split())

            electric_charge_and_multiplicity = start_data[0]
            
            for i in range(1, len(start_data)):
                element_list.append(start_data[i][0])
            geometry_list.append(start_data)
        else:
            print("unknown mode error")
            raise
        return geometry_list, element_list, electric_charge_and_multiplicity
        
class Graph:
    def __init__(self, folder_directory):
        self.BPA_FOLDER_DIRECTORY = folder_directory
        return
    def double_plot(self, num_list, energy_list, energy_list_2, add_file_name=""):
        
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.plot(num_list, energy_list, "g--.")
        ax2.plot(num_list, energy_list_2, "b--.")

        ax1.set_xlabel('ITR.')
        ax2.set_xlabel('ITR.')

        ax1.set_ylabel('Electronic Energy [kcal/mol]')
        ax2.set_ylabel('Electronic Energy [kcal/mol]')
        plt.title('normal_above AFIR_below')
        plt.tight_layout()
        plt.savefig(self.BPA_FOLDER_DIRECTORY+"Energy_plot_"+add_file_name+".png", format="png", dpi=300)
        plt.close()
        return
        
    def single_plot(self, num_list, energy_list, file_directory, atom_num, axis_name_1="ITR. ", axis_name_2="cosθ", name="orthogonality"):
        fig, ax = plt.subplots()
        ax.plot(num_list,energy_list, "r--o" , markersize=2)

        ax.set_title(str(atom_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(self.BPA_FOLDER_DIRECTORY+"Plot_"+name+"_"+str(atom_num)+".png", format="png", dpi=200)
        plt.close()
         
        return

class Calculationtools:
    def __init__(self):
        return
    
    def project_out_hess_tr_and_rot(self, hessian, element_list, geomerty):
        natoms = len(element_list)
        elem_mass = np.array([atomic_mass(elem) for elem in element_list], dtype="float64")
        
        M = np.diag(np.repeat(elem_mass, 3))
        #M_plus_sqrt = np.diag(np.repeat(elem_mass, 3) ** (0.5))
        M_minus_sqrt = np.diag(np.repeat(elem_mass, 3) ** (-0.5))

        m_plus_sqrt = np.repeat(elem_mass, 3) ** (0.5)
        #m_minus_sqrt = np.repeat(elem_mass, 3) ** (-0.5)

        mw_hessian = np.dot(np.dot(M_minus_sqrt, hessian), M_minus_sqrt)#mw = mass weighted
        
        tr_x = (np.tile(np.array([1, 0, 0]), natoms)).reshape(-1, 3)
        tr_y = (np.tile(np.array([0, 1, 0]), natoms)).reshape(-1, 3)
        tr_z = (np.tile(np.array([0, 0, 1]), natoms)).reshape(-1, 3)

        mw_rot_x = np.cross(geomerty, tr_x).flatten() * m_plus_sqrt
        mw_rot_y = np.cross(geomerty, tr_y).flatten() * m_plus_sqrt
        mw_rot_z = np.cross(geomerty, tr_z).flatten() * m_plus_sqrt

        mw_tr_x = tr_x.flatten() * m_plus_sqrt
        mw_tr_y = tr_y.flatten() * m_plus_sqrt
        mw_tr_z = tr_z.flatten() * m_plus_sqrt

        TR_vectors = np.vstack([mw_tr_x, mw_tr_y, mw_tr_z, mw_rot_x, mw_rot_y, mw_rot_z])
        
        Q, R = np.linalg.qr(TR_vectors.T)
        keep_indices = ~np.isclose(np.diag(R), 0, atol=1e-6, rtol=0)
        TR_vectors = Q.T[keep_indices]
        n_tr = len(TR_vectors)

        P = np.identity(natoms * 3)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)

        hess_proj = np.dot(np.dot(P.T, mw_hessian), P)

        eigenvalues, eigenvectors = np.linalg.eigh(hess_proj)
        eigenvalues = eigenvalues[n_tr:]
        eigenvectors = eigenvectors[:, n_tr:]
        print("=== hessian projected out transition and rotation (before add bias potential) ===")
        print("eigenvalues: ", eigenvalues)
        return hess_proj
    
  
    def check_atom_connectivity(self, mol_list, element_list, atom_num, covalent_radii_threshold_scale=1.2):
        connected_atoms = [atom_num]
        searched_atoms = []
        while True:
            for i in connected_atoms:
                if i in searched_atoms:
                    continue
                
                for j in range(len(mol_list)):
                    dist = np.linalg.norm(np.array(mol_list[i], dtype="float64") - np.array(mol_list[j], dtype="float64"))
                    
                    covalent_dist_threshold = covalent_radii_threshold_scale * (covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j]))
                    
                    if dist < covalent_dist_threshold:
                        if not j in connected_atoms:
                            connected_atoms.append(j)
                
                searched_atoms.append(i)
            
            if len(connected_atoms) == len(searched_atoms):
                break
     
        return sorted(connected_atoms)
    
    def calc_fragm_distance(self, geom_num_list, fragm_1_num, fragm_2_num):
        fragm_1_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        fragm_2_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        
        for num in fragm_1_num:
            fragm_1_coord += geom_num_list[num]
        
        fragm_1_coord /= len(fragm_1_num)
            
        for num in fragm_2_num:
            fragm_2_coord += geom_num_list[num]
        
        fragm_2_coord /= len(fragm_2_num)
        
        dist = np.linalg.norm(fragm_1_coord - fragm_2_coord)
        
        return dist
 


class Opt_calc_tmps:
    def __init__(self, adam_m, adam_v, adam_count, eve_d_tilde=0.0):
        self.adam_m = adam_m
        self.adam_v = adam_v
        self.adam_count = 1 + adam_count
        self.eve_d_tilde = eve_d_tilde
            
class Model_hess_tmp:
    def __init__(self, model_hess, momentum_disp=0, momentum_grad=0):
        self.model_hess = model_hess
        self.momentum_disp = momentum_disp
        self.momentum_grad = momentum_grad

class CalculationStructInfo:
    def __init__(self):
        return
    
    def calculate_cos(self, bg, g):
        if np.linalg.norm(bg) == 0.0 or np.linalg.norm(g) == 0.0:
            cos = 2.0
        else:
            cos = np.sum(bg * g) / (np.linalg.norm(g) * np.linalg.norm(bg))
        return cos
     
    
    def calculate_distance(self, atom1, atom2):
        atom1, atom2 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64")
        distance = np.linalg.norm(atom2 - atom1)
        return distance

    
    def calculate_bond_angle(self, atom1, atom2, atom3):
        atom1, atom2, atom3 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64"), np.array(atom3, dtype="float64")
        vector1 = atom1 - atom2
        vector2 = atom3 - atom2

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cos_angle)
        angle_deg = np.degrees(angle)

        return angle_deg
        
    def calculate_dihedral_angle(self, atom1, atom2, atom3, atom4):
        atom1, atom2, atom3, atom4 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64"), np.array(atom3, dtype="float64"), np.array(atom4, dtype="float64")
        
        a1 = atom2 - atom1
        a2 = atom3 - atom2
        a3 = atom4 - atom3

        v1 = np.cross(a1, a2)
        v1 = v1 / np.linalg.norm(v1, ord=2)
        v2 = np.cross(a2, a3)
        v2 = v2 / np.linalg.norm(v2, ord=2)
        porm = np.sign((v1 * a3).sum(-1))
        angle = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
        if not porm == 0:
            angle = angle * porm
            
        dihedral_angle_deg = np.degrees(angle)

        return dihedral_angle_deg
        

    def read_xyz_file(self, file_name):
        with open(file_name,"r") as f:
            words = f.readlines()
        mole_struct_list = []
            
        for word in words[1:]:
            mole_struct_list.append(word.split())
        return mole_struct_list

    def Data_extract(self, file, atom_numbers):
        data_list = []
        data_name_list = [] 
         
        
        
        mole_struct_list = self.read_xyz_file(file)
        DBD_list = []
        DBD_name_list = []
        print(file)
        if len(atom_numbers) > 1:
            for a1, a2 in list(itertools.combinations(atom_numbers,2)):
                try:
                    distance = self.calculate_distance(mole_struct_list[int(a1) - 1][1:4], mole_struct_list[int(a2) - 1][1:4])
                    DBD_name_list.append("Distance ("+str(a1)+"-"+str(a2)+")  [ang.]")
                    DBD_list.append(distance)
                        
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Distance ("+str(a1)+"-"+str(a2)+")  [ang.]")
                    DBD_list.append("nan")
                
        if len(atom_numbers) > 2:
            for a1, a2, a3 in list(itertools.permutations(atom_numbers,3)):
                try:
                    bond_angle = self.calculate_bond_angle(mole_struct_list[int(a1)-1][1:4], mole_struct_list[int(a2)-1][1:4], mole_struct_list[int(a3)-1][1:4])
                    DBD_name_list.append("Bond_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+") [deg.]")
                    DBD_list.append(bond_angle)
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Bond_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+") [deg.]")
                    DBD_list.append("nan")            
        
        if len(atom_numbers) > 3:
            for a1, a2, a3, a4 in list(itertools.permutations(atom_numbers,4)):
                try:
                    dihedral_angle = self.calculate_dihedral_angle(mole_struct_list[int(a1)-1][1:4], mole_struct_list[int(a2)-1][1:4],mole_struct_list[int(a3)-1][1:4], mole_struct_list[int(a4)-1][1:4])
                    DBD_name_list.append("Dihedral_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+"-"+str(a4)+") [deg.]")
                    DBD_list.append(dihedral_angle)
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Dihedral_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+"-"+str(a4)+") [deg.]")
                    DBD_list.append("nan")        

        data_list = DBD_list 
        
        data_name_list = DBD_name_list    
        return data_list, data_name_list
 
 
class BiasPotPy:
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
        

    def psi4_calculation(self, file_directory, element_list, electric_charge_and_multiplicity, iter):
        """execute QM calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")
                if int(electric_charge_and_multiplicity[1]) > 1:
                    psi4.set_options({'reference': 'uks'})
                logfile = file_directory+"/"+self.START_FILE[:-4]+'_'+str(num)+'.log'
                psi4.set_options({"MAXITER": 700})
                if len(self.SUB_BASIS_SET) > 0:
                    psi4.basis_helper(self.SUB_BASIS_SET, name='User_Basis_Set', set_option=False)
                    psi4.set_options({"basis":'User_Basis_Set'})
                else:
                    psi4.set_options({"basis":self.BASIS_SET})
                
                psi4.set_output_file(logfile)
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                
                psi4.set_options({"cubeprop_tasks": ["esp"],'cubeprop_filepath': file_directory})
                
                with open(input_file,"r") as f:
                    read_data = f.readlines()
                input_data = ""
                if iter == 0:
                    input_data += " ".join(list(map(str, electric_charge_and_multiplicity)))+"\n"
                    for data in read_data[2:]:
                        input_data += data
                else:
                    for data in read_data:
                        input_data += data
                
                input_data = psi4.geometry(input_data)#ang.
                input_data_for_display = np.array(input_data.geometry(), dtype = "float64")#Bohr
                            
                g, wfn = psi4.gradient(self.FUNCTIONAL, molecule=input_data, return_wfn=True)
                e = float(wfn.energy())
                g = np.array(g, dtype = "float64")
                psi4.oeprop(wfn, 'DIPOLE')
                psi4.oeprop(wfn, 'MULLIKEN_CHARGES')
                psi4.oeprop(wfn, 'LOWDIN_CHARGES')
                #psi4.oeprop(wfn, 'WIBERG_LOWDIN_INDICES')
                lumo_alpha = wfn.nalpha()
                lumo_beta = wfn.nbeta()

                MO_levels =np.array(wfn.epsilon_a_subset("AO","ALL")).tolist()#MO energy levels
                with open(self.BPA_FOLDER_DIRECTORY+"MO_levels.csv" ,"a") as f:
                    f.write(",".join(list(map(str,MO_levels))+[str(lumo_alpha),str(lumo_beta)])+"\n")
                with open(self.BPA_FOLDER_DIRECTORY+"dipole.csv" ,"a") as f:
                    f.write(",".join(list(map(str,(psi4.constants.dipmom_au2debye*wfn.variable('DIPOLE')).tolist()))+[str(np.linalg.norm(psi4.constants.dipmom_au2debye*wfn.variable('DIPOLE'),ord=2))])+"\n")
                with open(self.BPA_FOLDER_DIRECTORY+"MULLIKEN_CHARGES.csv" ,"a") as f:
                    f.write(",".join(list(map(str,wfn.variable('MULLIKEN CHARGES').tolist())))+"\n")           
                #with open(input_file[:-4]+"_WIBERG_LOWDIN_INDICES.csv" ,"a") as f:
                #    for i in range(len(np.array(wfn.variable('WIBERG LOWDIN INDICES')).tolist())):
                #        f.write(",".join(list(map(str,np.array(wfn.variable('WIBERG LOWDIN INDICES')).tolist()[i])))+"\n")           
                        

                print("\n")

                
                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    
                    """exact hessian"""
                    _, wfn = psi4.frequencies(self.FUNCTIONAL, return_wfn=True, ref_gradient=wfn.gradient())
                    exact_hess = np.array(wfn.hessian())
                    
                    freqs = np.array(wfn.frequencies())
                    
                    print("frequencies: \n",freqs)
                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot(exact_hess, element_list, input_data_for_display)

                    self.Model_hess = Model_hess_tmp(exact_hess, momentum_disp=self.Model_hess.momentum_disp, momentum_grad=self.Model_hess.momentum_grad)
                
                


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag 
                
            psi4.core.clean() 
        return e, g, input_data_for_display, finish_frag


    def pyscf_calculation(self, file_directory, element_list, iter):
        """execute QM calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                
                pyscf.lib.num_threads(self.N_THREAD)
                
                with open(input_file, "r") as f:
                    words = f.readlines()
                input_data_for_display = []
                for word in words[2:]:
                    input_data_for_display.append(np.array(word.split()[1:4], dtype="float64")/self.bohr2angstroms)
                input_data_for_display = np.array(input_data_for_display, dtype="float64")
                
                print("\n",input_file,"\n")
                mol = pyscf.gto.M(atom = input_file,
                                  charge = self.electronic_charge,
                                  spin = self.spin_multiplicity,
                                  basis = self.SUB_BASIS_SET,
                                  max_memory = float(self.SET_MEMORY.replace("GB","")) * 1024, #SET_MEMORY unit is GB
                                  verbose=3)
                if self.FUNCTIONAL == "hf" or self.FUNCTIONAL == "HF":
                    if int(self.spin_multiplicity) > 0:
                        mf = mol.UHF().x2c().density_fit()
                    else:
                        mf = mol.RHF().density_fit()
                else:
                    if int(self.spin_multiplicity) > 1:
                        mf = mol.UKS().x2c().density_fit()
                    else:
                        mf = mol.RKS().density_fit()
                    mf.xc = self.FUNCTIONAL
   
            
          
                g = mf.run().nuc_grad_method().kernel()
                e = float(vars(mf)["e_tot"])
                g = np.array(g, dtype = "float64")

                print("\n")


                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    
                    """exact hessian"""
                    exact_hess = mf.Hessian().kernel()
                    
                    freqs = thermo.harmonic_analysis(mf.mol, exact_hess)
                    exact_hess = exact_hess.transpose(0,2,1,3).reshape(len(input_data_for_display)*3, len(input_data_for_display)*3)
                    print("frequencies: \n",freqs["freq_wavenumber"])
                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot(exact_hess, element_list, input_data_for_display)

                    self.Model_hess = Model_hess_tmp(exact_hess, momentum_disp=self.Model_hess.momentum_disp, momentum_grad=self.Model_hess.momentum_grad)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag   
            
      
        return e, g, input_data_for_display, finish_frag
        

    def tblite_calculation(self, file_directory, element_number_list, electric_charge_and_multiplicity, iter, method):
        """execute extended tight binding method calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")

                with open(input_file,"r") as f:
                    input_data = f.readlines()
                
                positions = []
                if iter == 0:
                    for word in input_data[2:]:
                        positions.append(word.split()[1:4])
                else:
                    for word in input_data[1:]:
                        positions.append(word.split()[1:4])
                    
                
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                
                calc = Calculator(method, element_number_list, positions)
                calc.set("max-iter", 2500)
                calc.set("verbosity", 0)
                res = calc.singlepoint()
                e = float(res.get("energy"))  #hartree
                g = res.get("gradient") #hartree/Bohr
                        
                print("\n")

                
                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    print("error (cant calculate hessian)")
                    return 0, 0, 0, finish_frag 
                


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag 
        
        return e, g, positions, finish_frag


    def main_for_DSAFIR(self):#This implementation doesnt work well
        """
        DS-AFIR ref.:S. Maeda, et al., J. Comput. Chem., 2018, 39, 233–251.
        """

        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        finish_frag = False
        force_data = Interface().force_data_parser(args)
        rea_geometry_list, element_list, rea_electric_charge_and_multiplicity = FIO.make_geometry_list_for_DSAFIR(mode="r")
        reactant_file_directory = FIO.make_psi4_input_file_for_DSAFIR(rea_geometry_list, "input", mode="r")
        pro_geometry_list, element_list, pro_electric_charge_and_multiplicity = FIO.make_geometry_list_for_DSAFIR(mode="p")
        product_file_directory = FIO.make_psi4_input_file_for_DSAFIR(pro_geometry_list, "input", mode="p")
        rea_trust_radii = 0.10
        pro_trust_radii = 0.10
        reactant_energy_list = []
        product_energy_list = []
        reactant_bias_energy_list = []
        product_bias_energy_list = []
        self.DS_AFIR_ENERGY_LIST_FOR_PLOTTING = []
        #------------------------------------
        
        adam_m = []
        adam_v = []
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        
        self.reactant_Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.product_Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.reactant_Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
        self.product_Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
        
        rea_BPC = BiasPotentialCalculation(self.reactant_Model_hess, self.FC_COUNT)
        pro_BPC = BiasPotentialCalculation(self.product_Model_hess, self.FC_COUNT)

        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(args))
        pre_DSAFIR_e = 0.0
        
        pre_rea_e = 0.0
        pre_pro_e = 0.0
        pre_rea_B_e = 0.0
        pre_pro_B_e = 0.0
        
        pre_rea_g = []
        pre_pro_g = []
        for i in range(len(element_list)):
            pre_rea_g.append(np.array([0,0,0], dtype="float64"))
            pre_pro_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_rea_move_vector = copy.copy(pre_rea_g)
        pre_pro_move_vector = copy.copy(pre_pro_g)
        
        pre_rea_B_g = copy.copy(pre_rea_g)
        pre_pro_B_g = copy.copy(pre_pro_g)
        pre_rea_DSAFIR_g = copy.copy(pre_rea_g)
        pre_pro_DSAFIR_g = copy.copy(pre_pro_g)
        
        if force_data["xtb"] == "None":
            pass
        else:
            element_number_list = []
            for elem in element_list:
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list, dtype="int")
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------

        if force_data["xtb"] == "None":
        
            self.Model_hess = self.reactant_Model_hess
            rea_e, rea_g, rea_geom_num_list, finish_frag = self.psi4_calculation(reactant_file_directory, element_list, rea_electric_charge_and_multiplicity, -1)
            self.reactant_Model_hess = self.Model_hess

            self.Model_hess = self.product_Model_hess
            pro_e, pro_g, pro_geom_num_list, finish_frag = self.psi4_calculation(product_file_directory, element_list, rea_electric_charge_and_multiplicity, -1)
            self.product_Model_hess = self.Model_hess

            initial_rea_geom_num_list = rea_geom_num_list
            initial_pro_geom_num_list = pro_geom_num_list
            pre_rea_geom = initial_rea_geom_num_list
            pre_pro_geom = initial_pro_geom_num_list
            
            if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                sys.exit(1)
            _, rea_B_e, rea_B_g, rea_BPA_hessian = rea_BPC.main(rea_e, rea_g, rea_geom_num_list, element_list, force_data, pre_rea_B_g, -1, initial_rea_geom_num_list)#new_geometry:ang.
            _, pro_B_e, pro_B_g, pro_BPA_hessian = pro_BPC.main(pro_e, pro_g, pro_geom_num_list, element_list, force_data, pre_pro_B_g, -1, initial_pro_geom_num_list)#new_geometry:ang.
        else:
        
            self.Model_hess = self.reactant_Model_hess
            rea_e, rea_g, rea_geom_num_list, finish_frag = self.tblite_calculation(reactant_file_directory, element_number_list, rea_electric_charge_and_multiplicity, -1, force_data["xtb"])
            self.reactant_Model_hess = self.Model_hess

            self.Model_hess = self.product_Model_hess
            pro_e, pro_g, pro_geom_num_list, finish_frag = self.tblite_calculation(product_file_directory, element_number_list, pro_electric_charge_and_multiplicity, -1, force_data["xtb"])
            self.product_Model_hess = self.Model_hess

            initial_rea_geom_num_list = rea_geom_num_list
            initial_pro_geom_num_list = pro_geom_num_list
            pre_rea_geom = initial_rea_geom_num_list
            pre_pro_geom = initial_pro_geom_num_list
            
            if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                sys.exit(1)
            
            _, rea_B_e, rea_B_g, rea_BPA_hessian = rea_BPC.main(rea_e, rea_g, rea_geom_num_list, element_number_list, force_data, pre_rea_B_g, -1, initial_rea_geom_num_list)#new_geometry:ang.
            _, pro_B_e, pro_B_g, pro_BPA_hessian = pro_BPC.main(pro_e, pro_g, pro_geom_num_list, element_number_list, force_data, pre_pro_B_g, -1, initial_pro_geom_num_list)#new_geometry:ang.


        rea_CMV = CalculateMoveVector(self.DELTA, self.reactant_Opt_params, self.reactant_Model_hess, rea_BPA_hessian, rea_trust_radii, args.saddle_order, self.FC_COUNT, self.temperature)
        pro_CMV = CalculateMoveVector(self.DELTA, self.product_Opt_params, self.product_Model_hess, pro_BPA_hessian, pro_trust_radii, args.saddle_order, self.FC_COUNT, self.temperature)


        rea_new_geometry, rea_move_vector, self.reactant_Opt_params, self.reactant_Model_hess, rea_trust_radii = rea_CMV.calc_move_vector(0, rea_geom_num_list, rea_B_g, force_data["opt_method"], pre_rea_B_g, pre_rea_geom, rea_B_e, pre_rea_B_e, pre_rea_move_vector, initial_rea_geom_num_list, rea_g, pre_rea_g)
        self.reactant_Opt_params = rea_CMV.Opt_params
        self.reactant_Model_hess = rea_CMV.Model_hess

        pro_new_geometry, pro_move_vector, self.product_Opt_params, self.product_Model_hess, pro_trust_radii = pro_CMV.calc_move_vector(0, pro_geom_num_list, pro_B_g, force_data["opt_method"], pre_pro_g, pre_pro_geom, pro_B_e, pre_pro_B_e, pre_pro_move_vector, initial_pro_geom_num_list, pro_g, pre_pro_g)
        self.product_Opt_params = pro_CMV.Opt_params
        self.product_Model_hess = pro_CMV.Model_hess


        print("caluculation results (unit a.u.):")
        print("OPT method            : {} ".format(force_data["opt_method"]))
        print("                         Value                        ")
        print("ENERGY                      (reactant) : {:>15.12f} ".format(rea_e))
        print("Maxinum        Force        (reactant) : {:>15.12f}              ".format(abs(rea_g.max())))
        print("RMS            Force        (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_g).mean()))))
        print("ENERGY SHIFT                (reactant) : {:>15.12f} ".format(rea_e - pre_rea_e))
        print("ENERGY  (bias)              (reactant) : {:>15.12f} ".format(rea_B_e))
        print("Maxinum (bias) Force        (reactant) : {:>15.12f}              ".format(abs(rea_B_g.max())))
        print("RMS     (bias) Force        (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_B_g).mean()))))
        print("Maxinum        Displacement (reactant) : {:>15.12f}              ".format(abs(rea_move_vector.max())))
        print("RMS            Displacement (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_move_vector).mean()))))
        
        print("ENERGY SHIFT (bias)         (reactant) : {:>15.12f} ".format(rea_B_e - pre_rea_B_e))
        print("ENERGY                      (product ) : {:>15.12f} ".format(pro_e))
        print("Maxinum        Force        (product ) : {:>15.12f}              ".format(abs(pro_g.max())))
        print("RMS            Force        (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_g).mean()))))
        print("ENERGY SHIFT                (product ) : {:>15.12f} ".format(pro_e - pre_pro_e))
        print("ENERGY  (bias)              (product ) : {:>15.12f} ".format(pro_B_e))
        print("Maxinum (bias) Force        (product ) : {:>15.12f}              ".format(abs(pro_B_g.max())))
        print("RMS     (bias) Force        (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_B_g).mean()))))
        print("Maxinum        Displacement (product ) : {:>15.12f}              ".format(abs(pro_move_vector.max())))
        print("RMS            Displacement (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_move_vector).mean()))))
        print("ENERGY SHIFT (bias)         (product ) : {:>15.12f} ".format(pro_B_e - pre_pro_B_e))
        

        rea_geometry_list = FIO.make_geometry_list_2(rea_new_geometry, element_list, rea_electric_charge_and_multiplicity)
        pro_geometry_list = FIO.make_geometry_list_2(pro_new_geometry, element_list, pro_electric_charge_and_multiplicity)

        reactant_file_directory = FIO.make_psi4_input_file_for_DSAFIR(rea_geometry_list, "init", mode="r")
        product_file_directory = FIO.make_psi4_input_file_for_DSAFIR(pro_geometry_list, "init", mode="p")


        pre_rea_e = rea_e#Hartree
        pre_rea_B_e = rea_B_e#Hartree
        pre_rea_g = copy.copy(rea_g)#Hartree/Bohr
        pre_rea_B_g = copy.copy(rea_B_g)#Hartree/Bohr
        pre_rea_geom = rea_geom_num_list#Bohr
        pre_rea_move_vector = rea_move_vector
        pre_pro_e = pro_e#Hartree
        pre_pro_B_e = pro_B_e#Hartree
        pre_pro_g = copy.copy(pro_g)#Hartree/Bohr
        pre_pro_B_g = copy.copy(pro_B_g)#Hartree/Bohr
        pre_pro_geom = pro_geom_num_list#Bohr
        pre_pro_move_vector = pro_move_vector
        
        #------------------------------------
        
        adam_m = []
        adam_v = []
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        
        self.reactant_Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.product_Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.reactant_Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
        self.product_Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
        
        rea_BPC = BiasPotentialCalculation(self.reactant_Model_hess, self.FC_COUNT)
        pro_BPC = BiasPotentialCalculation(self.product_Model_hess, self.FC_COUNT)

        #-----------------------------------
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
            

            if rea_B_e < pro_B_e:
                if force_data["xtb"] == "None":
                    self.Model_hess = self.reactant_Model_hess
                    rea_e, rea_g, rea_geom_num_list, finish_frag = self.psi4_calculation(reactant_file_directory, element_list, rea_electric_charge_and_multiplicity, iter)
                    self.reactant_Model_hess = self.Model_hess

                    if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                        break
                    _, rea_B_e, rea_B_g, rea_BPA_hessian = rea_BPC.main(rea_e, rea_g, rea_geom_num_list, element_list, force_data, pre_rea_B_g, iter, initial_rea_geom_num_list)#new_geometry:ang.
                else:
                    self.Model_hess = self.reactant_Model_hess
                    rea_e, rea_g, rea_geom_num_list, finish_frag = self.tblite_calculation(reactant_file_directory, element_number_list, rea_electric_charge_and_multiplicity, iter, force_data["xtb"])
                    self.reactant_Model_hess = self.Model_hess
                    if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                        break 
                    
                    _, rea_B_e, rea_B_g, rea_BPA_hessian = rea_BPC.main(rea_e, rea_g, rea_geom_num_list, element_number_list, force_data, pre_rea_B_g, iter, initial_rea_geom_num_list)#new_geometry:ang.



                #-------------------energy profile 
                if iter == 0:
                    with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                        f.write("reactant energy [hartree], product energy [hartree] \n")
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write(str(rea_e)+","+str(pro_e)+"\n")

                #-------------------
                print("-----")

                DS_BPC = BiasPotentialCalculation(self.reactant_Model_hess, self.FC_COUNT)
                
                DSAFIR_e, DSAFIR_g, DS_AFIR_hessian = DS_BPC.calc_DSAFIRPotential(rea_B_e, rea_B_g, rea_geom_num_list, pro_geom_num_list, initial_rea_geom_num_list)#new_geometry:ang.
                DSAFIR_g += rea_B_g
                
                rea_CMV = CalculateMoveVector(self.DELTA, self.reactant_Opt_params, self.reactant_Model_hess, DS_AFIR_hessian+rea_BPA_hessian, rea_trust_radii, args.saddle_order, self.FC_COUNT, self.temperature)

                rea_new_geometry, rea_move_vector, self.reactant_Opt_params, self.reactant_Model_hess, rea_trust_radii = rea_CMV.calc_move_vector(iter, rea_geom_num_list, DSAFIR_g, force_data["opt_method"], pre_rea_DSAFIR_g, pre_rea_geom, DSAFIR_e, pre_DSAFIR_e, pre_rea_move_vector, initial_rea_geom_num_list, rea_g, pre_rea_g)
                self.reactant_Opt_params = rea_CMV.Opt_params
                self.reactant_Model_hess = rea_CMV.Model_hess

                print("caluculation results (unit a.u.):")
                print("OPT method            : {} ".format(force_data["opt_method"]))
                print("                         Value                        ")
                
                print("ENERGY                         (reactant) : {:>15.12f} ".format(rea_e))
                print("Maxinum        Force           (reactant) : {:>15.12f}              ".format(abs(rea_g.max())))
                print("RMS            Force           (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_g).mean()))))
                print("ENERGY SHIFT                   (reactant) : {:>15.12f} ".format(rea_e - pre_rea_e))
                print("ENERGY  (bias)                 (reactant) : {:>15.12f} ".format(rea_B_e))
                print("Maxinum (bias) Force           (reactant) : {:>15.12f}              ".format(abs(rea_B_g.max())))
                print("RMS     (bias) Force           (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_B_g).mean()))))
                
                print("ENERGY SHIFT (bias)            (reactant) : {:>15.12f} ".format(rea_B_e - pre_rea_B_e))
                print("Maxinum (DS-AFIR) Force        (reactant) : {:>15.12f}              ".format(abs(DSAFIR_g.max())))
                print("RMS     (DS-AFIR) Force        (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(DSAFIR_g).mean()))))
                print("Maxinum (DS-AFIR) Displacement (reactant) : {:>15.12f}              ".format(abs(rea_move_vector.max())))
                print("RMS     (DS-AFIR) Displacement (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_move_vector).mean()))))
                print("DSAFIR ENERGY                             : {:>15.12f} ".format(DSAFIR_e))
                print("DSAFIR ENERGY SHIFT                       : {:>15.12f} ".format(DSAFIR_e - pre_DSAFIR_e))
                print("-----")
                delta_geom = rea_new_geometry - pro_new_geometry
                print("distance:", np.linalg.norm(delta_geom))
                if np.linalg.norm(delta_geom) < 0.12:
                    print("convergence criteria are satisfied.")
                    break

                if iter != 0:
                    if pre_rea_e > rea_e:
                        initial_rea_geom_num_list = rea_geom_num_list



                pre_rea_DSAFIR_g = DSAFIR_g
                reactant_bias_energy_list.append(rea_B_e*self.hartree2kcalmol)
                reactant_energy_list.append(rea_e*self.hartree2kcalmol)
                rea_geometry_list = FIO.make_geometry_list_2(rea_new_geometry, element_list, rea_electric_charge_and_multiplicity)
                reactant_file_directory = FIO.make_psi4_input_file_for_DSAFIR(rea_geometry_list, iter, mode="r")

                pre_rea_e = rea_e#Hartree
                pre_rea_B_e = rea_B_e#Hartree
                pre_rea_g = copy.copy(rea_g)#Hartree/Bohr
                pre_rea_B_g = copy.copy(rea_B_g)#Hartree/Bohr
                pre_rea_geom = rea_geom_num_list#Bohr
                pre_rea_move_vector = rea_move_vector
            

            else:
                if force_data["xtb"] == "None":
                    self.Model_hess = self.product_Model_hess
                    pro_e, pro_g, pro_geom_num_list, finish_frag = self.psi4_calculation(product_file_directory, element_list, rea_electric_charge_and_multiplicity, iter)
                    self.product_Model_hess = self.Model_hess
                    
                    if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                        break
                    _, pro_B_e, pro_B_g, pro_BPA_hessian = pro_BPC.main(pro_e, pro_g, pro_geom_num_list, element_list, force_data, pre_pro_B_g, iter, initial_pro_geom_num_list)#new_geometry:ang.

                else:
                    self.Model_hess = self.product_Model_hess
                    pro_e, pro_g, pro_geom_num_list, finish_frag = self.tblite_calculation(product_file_directory, element_number_list, rea_electric_charge_and_multiplicity, iter, force_data["xtb"])
                    self.product_Model_hess = self.Model_hess
                    if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                        break
                    _, pro_B_e, pro_B_g, pro_BPA_hessian = pro_BPC.main(pro_e, pro_g, pro_geom_num_list, element_number_list, force_data, pre_pro_B_g, iter, initial_pro_geom_num_list)#new_geometry:ang.
                



                #-------------------energy profile 
                if iter == 0:
                    with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                        f.write("reactant energy [hartree], product energy [hartree] \n")
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write(str(rea_e)+","+str(pro_e)+"\n")



                #-------------------
                print("-----")
                DS_BPC = BiasPotentialCalculation(self.product_Model_hess, self.FC_COUNT)
                DSAFIR_e, DSAFIR_g, DS_AFIR_hessian = DS_BPC.calc_DSAFIRPotential(pro_B_e, pro_B_g, pro_geom_num_list, rea_geom_num_list, initial_pro_geom_num_list)#new_geometry:ang.
                DSAFIR_g += pro_B_g 
                
                pro_CMV = CalculateMoveVector(self.DELTA, self.product_Opt_params, self.product_Model_hess, DS_AFIR_hessian+pro_BPA_hessian, pro_trust_radii, args.saddle_order, self.FC_COUNT, self.temperature)
                pro_new_geometry, pro_move_vector, self.product_Opt_params, self.product_Model_hess, pro_trust_radii = pro_CMV.calc_move_vector(iter, pro_geom_num_list, DSAFIR_g, force_data["opt_method"], pre_pro_DSAFIR_g, pre_pro_geom, DSAFIR_e, pre_DSAFIR_e, pre_pro_move_vector, initial_pro_geom_num_list, pro_g, pre_pro_g)
                self.product_Opt_params = pro_CMV.Opt_params
                self.product_Model_hess = pro_CMV.Model_hess


                print("caluculation results (unit a.u.):")
                print("OPT method            : {} ".format(force_data["opt_method"]))
                print("                         Value                        ")
                print("ENERGY                         (product ) : {:>15.12f} ".format(pro_e))
                print("Maxinum        Force           (product ) : {:>15.12f}              ".format(abs(pro_g.max())))
                print("RMS            Force           (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_g).mean()))))
                print("ENERGY SHIFT                   (product ) : {:>15.12f} ".format(pro_e - pre_pro_e))
                
                print("ENERGY  (bias)                 (product ) : {:>15.12f} ".format(pro_B_e))
                print("Maxinum (bias) Force           (product ) : {:>15.12f}              ".format(abs(pro_B_g.max())))
                print("RMS     (bias) Force           (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_B_g).mean())))) 
                print("ENERGY SHIFT (bias)            (product ) : {:>15.12f} ".format(pro_B_e - pre_pro_B_e))
                
                print("Maxinum (DS-AFIR) Force        (product ) : {:>15.12f}              ".format(abs(DSAFIR_g.max())))
                print("RMS     (DS-AFIR) Force        (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(DSAFIR_g).mean()))))
                print("Maxinum (DS-AFIR) Displacement (product ) : {:>15.12f}              ".format(abs(pro_move_vector.max())))
                print("RMS     (DS-AFIR) Displacement (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_move_vector).mean()))))
                print("DSAFIR ENERGY                             : {:>15.12f} ".format(DSAFIR_e))
                print("DSAFIR ENERGY SHIFT                       : {:>15.12f} ".format(DSAFIR_e - pre_DSAFIR_e))
                print("-----")
                delta_geom = rea_new_geometry - pro_new_geometry
                print("distance:", np.linalg.norm(delta_geom))
                if np.linalg.norm(delta_geom) < 0.12:
                    print("convergence criteria are satisfied.")
                    break

            
                if iter != 0:
                    if pre_pro_e > pro_e:
                        initial_pro_geom_num_list = pro_geom_num_list
                        

                        
                pre_pro_DSAFIR_g = DSAFIR_g
                product_bias_energy_list.append(pro_B_e*self.hartree2kcalmol)
                product_energy_list.append(pro_e*self.hartree2kcalmol)
                pro_geometry_list = FIO.make_geometry_list_2(pro_new_geometry, element_list, pro_electric_charge_and_multiplicity)
                product_file_directory = FIO.make_psi4_input_file_for_DSAFIR(pro_geometry_list, iter, mode="p")


                pre_pro_e = pro_e#Hartree
                pre_pro_B_e = pro_B_e#Hartree
                pre_pro_g = copy.copy(pro_g)#Hartree/Bohr
                pre_pro_B_g = copy.copy(pro_B_g)#Hartree/Bohr
                pre_pro_geom = pro_geom_num_list#Bohr
                pre_pro_move_vector = pro_move_vector

            pre_DSAFIR_e = DSAFIR_e#Hartree
            self.DS_AFIR_ENERGY_LIST_FOR_PLOTTING.append(DSAFIR_e*self.hartree2kcalmol)
            self.NUM_LIST.append(iter)
            

        self.ENERGY_LIST_FOR_PLOTTING = np.array(reactant_energy_list + product_energy_list[::-1], dtype="float64")
        self.BIAS_ENERGY_LIST_FOR_PLOTTING = np.array(reactant_bias_energy_list + product_bias_energy_list[::-1], dtype="float64")
        self.DS_AFIR_ENERGY_LIST_FOR_PLOTTING = np.array(self.DS_AFIR_ENERGY_LIST_FOR_PLOTTING, dtype="float64")
        
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.DS_AFIR_ENERGY_LIST_FOR_PLOTTING, add_file_name="normal")
        G.double_plot(self.NUM_LIST, self.BIAS_ENERGY_LIST_FOR_PLOTTING, self.DS_AFIR_ENERGY_LIST_FOR_PLOTTING, add_file_name="bias")
        

        FIO.xyz_file_make_for_DSAFIR()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
        #----------------------
        print("Complete...")
    
        return

    def main_using_tblite(self):

        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = Interface().force_data_parser(self.args)
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

            e, g, geom_num_list, finish_frag = self.tblite_calculation(file_directory, element_number_list,  electric_charge_and_multiplicity, iter, force_data["xtb"])
            
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
                    print("mean fragm distance (ang.)", mean_dist, ">", self.DC_check_dist)
                    
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



    def main_using_psi4(self):

        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = Interface().force_data_parser(self.args)
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
            
            e, g, geom_num_list, finish_frag = self.psi4_calculation(file_directory, element_list,  electric_charge_and_multiplicity, iter)

            
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
                    print("mean fragm distance (ang.)", mean_dist, ">", self.DC_check_dist)
                    
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
    
    def main_using_pyscf(self):
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        trust_radii = 0.01
        force_data = Interface().force_data_parser(args)
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
            f.write(str(args))
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

            e, g, geom_num_list, finish_frag = self.pyscf_calculation(file_directory, element_list, iter)
            
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
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, BPA_hessian, trust_radii, args.saddle_order, self.FC_COUNT, self.temperature)
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
                    print("mean fragm distance (ang.)", mean_dist, ">", self.DC_check_dist)
                    
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
        if self.args.DS_AFIR:
            self.main_for_DSAFIR()
        elif self.args.pyscf:
            self.main_using_pyscf()
        elif self.args.usextb != "None":
            self.main_using_tblite()
        else:
            self.main_using_psi4()



if __name__ == "__main__":
    args = parser()
    bpa = BiasPotPy(args)
    bpa.run()
