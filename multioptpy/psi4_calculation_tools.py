import psi4
import glob
import os

import numpy as np

from calc_tools import Calculationtools
from fileio import xyz2list
"""
Psi4
 D. G. A. Smith, L. A. Burns, A. C. Simmonett, R. M. Parrish, M. C. Schieber, R. Galvelis, P. Kraus, H. Kruse, R. Di Remigio, A. Alenaizan, A. M. James, S. Lehtola, J. P. Misiewicz, M. Scheurer, R. A. Shaw, J. B. Schriber, Y. Xie, Z. L. Glick, D. A. Sirianni, J. S. O'Brien, J. M. Waldrop, A. Kumar, E. G. Hohenstein, B. P. Pritchard, B. R. Brooks, H. F. Schaefer III, A. Yu. Sokolov, K. Patkowski, A. E. DePrince III, U. Bozkaya, R. A. King, F. A. Evangelista, J. M. Turney, T. D. Crawford, C. D. Sherrill, "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).

"""
class Calculation:
    def __init__(self, **kwarg):
        
        self.START_FILE = kwarg["START_FILE"]
        self.SUB_BASIS_SET = kwarg["SUB_BASIS_SET"]
        self.BASIS_SET = kwarg["BASIS_SET"]
        self.N_THREAD = kwarg["N_THREAD"]
        self.SET_MEMORY = kwarg["SET_MEMORY"]
        self.FUNCTIONAL = kwarg["FUNCTIONAL"]
        self.FC_COUNT = kwarg["FC_COUNT"]
        self.BPA_FOLDER_DIRECTORY = kwarg["BPA_FOLDER_DIRECTORY"]
        self.Model_hess = kwarg["Model_hess"]
        self.unrestrict = kwarg["unrestrict"]
        self.hessian_flag = False
        if kwarg["excited_state"]:
            self.excited_state = kwarg["excited_state"]
        else:
            self.excited_state = 0
        return
    
    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method="", geom_num_list=None):
        """execute QM calclation."""
        finish_frag = False
        input_data_for_display = None
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                
                if int(electric_charge_and_multiplicity[1]) > 1 or self.unrestrict:
                    psi4.set_options({'reference': 'uks'})
                logfile = file_directory+"/"+self.START_FILE[:-4]+'_'+str(num)+'.log'
                psi4.set_options({"MAXITER": 1000})
                if len(self.SUB_BASIS_SET) > 0:
                    psi4.basis_helper(self.SUB_BASIS_SET, name='User_Basis_Set', set_option=False)
                    psi4.set_options({"basis":'User_Basis_Set'})
                else:
                    psi4.set_options({"basis":self.BASIS_SET})
                    
                if self.excited_state > 0:
                    psi4.set_options({'TDSCF_STATES': self.excited_state})

                psi4.set_output_file(logfile)
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                #psi4.procrouting.response.scf_response.tdscf_excitations
                psi4.set_options({"cubeprop_tasks": ["esp"],'cubeprop_filepath': file_directory})
                
                if geom_num_list is None:
                    
                    input_data = ""
                    position, element_list, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                    input_data_for_display = np.array(position, dtype="float64")/psi4.constants.bohr2angstroms
                    input_data += " ".join(list(map(str, electric_charge_and_multiplicity)))+"\n"
                    for j in range(len(position)):
                        input_data += element_list[j]+" "+" ".join(position[j])+"\n"
                else:
                    print("Input data is given as a numpy array.")
                    input_data = ""
                  
                    input_data += " ".join(list(map(str, electric_charge_and_multiplicity)))+"\n"
                    for j in range(len(geom_num_list)):
                        input_data += element_list[j]+" "+" ".join(list(map(str, geom_num_list[j].tolist())))+"\n"
                    input_data_for_display = geom_num_list / psi4.constants.bohr2angstroms
                
                input_data = psi4.geometry(input_data)#ang.
                
            
                g, wfn = psi4.gradient(self.FUNCTIONAL, molecule=input_data, return_wfn=True)

                e = float(wfn.energy())
                g = np.array(g, dtype = "float64")
                psi4.oeprop(wfn, 'DIPOLE')
                psi4.oeprop(wfn, 'MULLIKEN_CHARGES')
                psi4.oeprop(wfn, 'LOWDIN_CHARGES')
                #psi4.oeprop(wfn, 'WIBERG_LOWDIN_INDICES')
                lumo_alpha = wfn.nalpha()
                lumo_beta = wfn.nbeta()

                MO_levels = np.array(wfn.epsilon_a_subset("AO","ALL")).tolist()#MO energy levels
                with open(self.BPA_FOLDER_DIRECTORY+"MO_levels.csv" ,"a") as f:
                    f.write(",".join(list(map(str,MO_levels))+[str(lumo_alpha),str(lumo_beta)])+"\n")
                with open(self.BPA_FOLDER_DIRECTORY+"dipole.csv" ,"a") as f:
                    f.write(",".join(list(map(str,(psi4.constants.dipmom_au2debye*wfn.variable('DIPOLE')).tolist()))+[str(np.linalg.norm(psi4.constants.dipmom_au2debye*wfn.variable('DIPOLE'),ord=2))])+"\n")
                with open(self.BPA_FOLDER_DIRECTORY+"MULLIKEN_CHARGES.csv" ,"a") as f:
                    f.write(",".join(list(map(str,wfn.variable('MULLIKEN CHARGES').tolist())))+"\n")
                    
                alpha_first_ionization_energy = -1 * MO_levels[lumo_alpha-1]
                alpha_electron_affinity = MO_levels[lumo_alpha]
                global_electrophilicity_index = (alpha_first_ionization_energy + alpha_electron_affinity) / (8 * (alpha_first_ionization_energy - alpha_electron_affinity + 1e-15))
                
                print("=== global electrophilicity index ===")
                print(global_electrophilicity_index, "hartree")
                
                #with open(input_file[:-4]+"_WIBERG_LOWDIN_INDICES.csv" ,"a") as f:
                #    for i in range(len(np.array(wfn.variable('WIBERG LOWDIN INDICES')).tolist())):
                #        f.write(",".join(list(map(str,np.array(wfn.variable('WIBERG LOWDIN INDICES')).tolist()[i])))+"\n")           
                        
                print("\n")
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.exact_hessian(element_list, input_data_for_display, wfn)
                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_list, input_data_for_display, wfn)
                
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                print("Input file: ",file_list,"\n")
                finish_frag = True
                return np.array([0]), np.array([0]), input_data_for_display, finish_frag 
                
            psi4.core.clean() 
        self.energy = e
        self.gradient = g
        if geom_num_list is None:
            self.coordinate = input_data_for_display
        else:
            self.coordinate = geom_num_list / psi4.constants.bohr2angstroms
            return e, g, self.coordinate, finish_frag
            
        
        return e, g, input_data_for_display, finish_frag

    def exact_hessian(self, element_list, input_data_for_display, wfn):
        """exact hessian"""
        _, wfn = psi4.frequencies(self.FUNCTIONAL, return_wfn=True, ref_gradient=wfn.gradient())
        exact_hess = np.array(wfn.hessian())
                    
        freqs = np.array(wfn.frequencies())
                    
        print("frequencies: \n",freqs)
        #eigenvalues, _ = np.linalg.eigh(exact_hess)
        #print("=== hessian (before add bias potential) ===")
        #print("eigenvalues: ", eigenvalues)
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, input_data_for_display, display_eigval=False)
        self.Model_hess = exact_hess