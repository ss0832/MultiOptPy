import pyscf
import glob
import os

import numpy as np

from pyscf import gto, scf, dft, tddft, tdscf
from pyscf.hessian import thermo

from calc_tools import Calculationtools
from parameter import UnitValueLib
from fileio import xyz2list

"""
Ref.:PySCF
Recent developments in the PySCF program package, Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan, J. Chem. Phys., 153, 024109 (2020). doi:10.1063/5.0006074

"""

class Calculation:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2eV = UVL.hartree2eV
        self.START_FILE = kwarg["START_FILE"]
        self.SUB_BASIS_SET = kwarg["SUB_BASIS_SET"]
        self.BASIS_SET = kwarg["BASIS_SET"]
        self.N_THREAD = kwarg["N_THREAD"]
        self.SET_MEMORY = kwarg["SET_MEMORY"]
        self.FUNCTIONAL = kwarg["FUNCTIONAL"]
        self.FC_COUNT = kwarg["FC_COUNT"]
        self.BPA_FOLDER_DIRECTORY = kwarg["BPA_FOLDER_DIRECTORY"]
        self.Model_hess = kwarg["Model_hess"]
        self.electronic_charge = kwarg["electronic_charge"]
        self.spin_multiplicity = kwarg["spin_multiplicity"]
        self.unrestrict = kwarg["unrestrict"]
        self.hessian_flag = False
        if kwarg["excited_state"]:
            self.excited_state = kwarg["excited_state"] # Available up to third excited state
        else:
            self.excited_state = 0
            
    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity="", method="", geom_num_list=None):
        """execute QM calclation."""
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                pyscf.lib.num_threads(self.N_THREAD)
                
                if geom_num_list is not None:
                    geom_num_list = np.array(geom_num_list, dtype="float64")
                    input_data_for_display = geom_num_list / self.bohr2angstroms
                    input_data = [[element_list[i], geom_num_list[i][0], geom_num_list[i][1], geom_num_list[i][2]] for i in range(len(geom_num_list))]
                    print("position is not read from xyz file. The position is read from input variable.")
                    mol = pyscf.gto.M(atom = input_data,
                                    charge = self.electronic_charge,
                                    spin = self.spin_multiplicity,
                                    basis = self.SUB_BASIS_SET,
                                    max_memory = float(self.SET_MEMORY.replace("GB","")) * 1024, #SET_MEMORY unit is GB
                                    verbose=4)
                else:
                    positions, element_list, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                    input_data_for_display = np.array(positions, dtype="float64")/self.bohr2angstroms
                
                    
                    mol = pyscf.gto.M(atom = input_file,
                                    charge = self.electronic_charge,
                                    spin = self.spin_multiplicity,
                                    basis = self.SUB_BASIS_SET,
                                    max_memory = float(self.SET_MEMORY.replace("GB","")) * 1024, #SET_MEMORY unit is GB
                                    verbose=4)
                
                scf_max_cycle = 300 + 10 * len(element_list)
                
                if self.excited_state == 0:
                    if self.FUNCTIONAL == "hf" or self.FUNCTIONAL == "HF":
                        if int(self.spin_multiplicity) > 0 or self.unrestrict:
                            mf = mol.UHF().density_fit()
                        else:
                            mf = mol.RHF().density_fit()
                    else:
                        if int(self.spin_multiplicity) > 0 or self.unrestrict:
                            mf = mol.UKS().x2c().density_fit()
                        else:
                            mf = mol.RKS().density_fit()
                        mf.xc = self.FUNCTIONAL
                
                    mf.direct_scf = True
                    g = mf.run(max_cycle=scf_max_cycle).nuc_grad_method().kernel()
                    e = float(vars(mf)["e_tot"])
                else:
                    if self.FUNCTIONAL == "hf" or self.FUNCTIONAL == "HF":
                        if int(self.spin_multiplicity) > 0 or self.unrestrict:
                            mf = mol.UHF().density_fit().run(max_cycle=scf_max_cycle)

                        else:
                            mf = mol.RHF().density_fit().run(max_cycle=scf_max_cycle)

                    else:
                        if int(self.spin_multiplicity) > 0 or self.unrestrict:
                            mf = mol.UKS().x2c().density_fit().run(max_cycle=scf_max_cycle)
                            mf.xc = self.FUNCTIONAL

                        else:
                            mf = mol.RKS().density_fit().run(max_cycle=scf_max_cycle)
                            mf.xc = self.FUNCTIONAL
                    
                    mf.direct_scf = True
                    ground_e = float(vars(mf)["e_tot"])
                    mf = tdscf.TDA(mf)
                    g = mf.run(max_cycle=scf_max_cycle).nuc_grad_method().kernel(state=self.excited_state)
                    e = vars(mf)["e"][self.excited_state-1]
                    e += ground_e

                g = np.array(g, dtype = "float64")

                print("\n")

                
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.exact_hessian(element_list, input_data_for_display, mf)
        
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    
                    self.exact_hessian(element_list, input_data_for_display, mf)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                print("Input file: ",file_list,"\n")
                finish_frag = True
                return np.array([0]), np.array([0]), input_data_for_display, finish_frag
             
        self.energy = e
        self.gradient = g
        self.coordinate = input_data_for_display
        
        return e, g, input_data_for_display, finish_frag

    def exact_hessian(self, element_list, input_data_for_display, mf):
        """exact hessian"""
        exact_hess = mf.Hessian().kernel()
                    
        freqs = thermo.harmonic_analysis(mf.mol, exact_hess)
        exact_hess = exact_hess.transpose(0,2,1,3).reshape(len(input_data_for_display)*3, len(input_data_for_display)*3)
        print("frequencies: \n",freqs["freq_wavenumber"])
                    #eigenvalues, _ = np.linalg.eigh(exact_hess)
                    #print("=== hessian (before add bias potential) ===")
                    #print("eigenvalues: ", eigenvalues)
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, input_data_for_display, display_eigval=False)

        self.Model_hess = exact_hess
    
    
    