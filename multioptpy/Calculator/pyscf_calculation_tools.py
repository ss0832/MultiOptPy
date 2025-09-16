
import glob
import os
import numpy as np

from abc import ABC, abstractmethod


try:
    import pyscf
    from pyscf import tdscf
    from pyscf.hessian import thermo
except:
    pass

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer

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
        self.ECP = kwarg["ECP"]
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
        self.dft_grid = kwarg["dft_grid"]
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

        if file_directory is None:
            file_list = ["dummy"]
        else:
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
                                    ecp = self.ECP,
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
                                    ecp = self.ECP,
                                    max_memory = float(self.SET_MEMORY.replace("GB","")) * 1024, #SET_MEMORY unit is GB
                                    verbose=4)
                
                scf_max_cycle = 500 + 5 * len(element_list)
                
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
                        mf.grids.level = self.dft_grid
                    print("dft grid: ", self.dft_grid)
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
                        
                        else:
                            mf = mol.RKS().density_fit().run(max_cycle=scf_max_cycle)
                        mf.xc = self.FUNCTIONAL
                        mf.grids.level = self.dft_grid
                    mf.direct_scf = True
                    print("dft grid: ", self.dft_grid)
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
    
    

class CalculationEngine(ABC):
    """Base class for calculation engines"""
    
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        """Calculate energy and gradients"""
        pass
    
    def _get_file_list(self, file_directory):
        """Get list of input files"""
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) 
                   for i in range(1, 7)], [])
    
    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        """Process common visualization tasks"""
        try:
            if config.save_pict:
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                
                gradient_norm_list = [np.sqrt(np.linalg.norm(g)**2/(len(g)*3)) for g in gradient_list]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")




class PySCFEngine(CalculationEngine):
    """PySCF calculation engine"""
    
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                geometry_list, element_list, electric_charge_and_multiplicity = xyz2list(input_file, None)
                words = []
                for i in range(len(geometry_list)):
                    words.append([element_list[i], float(geometry_list[i][0]), float(geometry_list[i][1]), float(geometry_list[i][2])])
                
                input_data_for_display = np.array(geometry_list, dtype="float64") / config.bohr2angstroms
                
                mol = pyscf.gto.M(atom=words,
                                  charge=int(electric_charge_and_multiplicity[0]),
                                  spin=int(electric_charge_and_multiplicity[1]),
                                  basis=config.SUB_BASIS_SET,
                                  ecp=config.ECP,
                                  max_memory=float(config.SET_MEMORY.replace("GB","")) * 1024,
                                  verbose=4)
                
                if config.excited_state == 0:
                    if config.FUNCTIONAL == "hf" or config.FUNCTIONAL == "HF":
                        if int(electric_charge_and_multiplicity[1]) > 0 or config.unrestrict:
                            mf = mol.UHF().density_fit()
                        else:
                            mf = mol.RHF().density_fit()
                    else:
                        if int(electric_charge_and_multiplicity[1]) > 0 or config.unrestrict:
                            mf = mol.UKS().x2c().density_fit()
                        else:
                            mf = mol.RKS().density_fit()
                        mf.xc = config.FUNCTIONAL
                        mf.grids.level = config.dft_grid 
                    g = mf.run().nuc_grad_method().kernel()
                    e = float(vars(mf)["e_tot"])
                else:
                    if config.FUNCTIONAL == "hf" or config.FUNCTIONAL == "HF":
                        if int(electric_charge_and_multiplicity[1])-1 > 0 or config.unrestrict:
                            mf = mol.UHF().density_fit().run()
                        else:
                            mf = mol.RHF().density_fit().run()
                    else:
                        if int(electric_charge_and_multiplicity[1])-1 > 0 or config.unrestrict:
                            mf = mol.UKS().x2c().density_fit().run()
                        else:
                            mf = mol.RKS().density_fit().run()
                        mf.xc = config.FUNCTIONAL
                        mf.grids.level = config.dft_grid
                        
                    ground_e = float(vars(mf)["e_tot"])
                    
                    mf = tdscf.TDA(mf)
                    g = mf.run().nuc_grad_method().kernel(state=config.excited_state)
                    e = vars(mf)["e"][config.excited_state-1]
                    e += ground_e

                g = np.array(g, dtype="float64")
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))  # RMS
                geometry_num_list.append(input_data_for_display)
                num_list.append(num)
                
                if config.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    """exact hessian"""
                    exact_hess = mf.Hessian().kernel()
                    freqs = thermo.harmonic_analysis(mf.mol, exact_hess)
                    exact_hess = exact_hess.transpose(0,2,1,3).reshape(len(input_data_for_display)*3, len(input_data_for_display)*3)
                    print("frequencies: \n", freqs["freq_wavenumber"])
                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, input_data_for_display)
                 
                    np.save(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".npy", exact_hess)
                    with open(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".csv", "a") as f:
                        f.write("frequency," + ",".join(map(str, freqs["freq_wavenumber"])) + "\n")
                hess_count += 1
                
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)

        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return (np.array(energy_list, dtype="float64"), 
                np.array(gradient_list, dtype="float64"), 
                np.array(geometry_num_list, dtype="float64"), 
                pre_total_velocity)

