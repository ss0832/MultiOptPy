import glob
import os
import numpy as np
import datetime
from abc import ABC, abstractmethod

try:
    from ase import Atoms
    from ase.vibrations import Vibrations
except ImportError:
    print("ASE is not installed. Please install ASE to use this module.")

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, number_element
from multioptpy.fileio import read_software_path, xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
from multioptpy.Calculator.ase_tools.gamess import ASE_GAMESSUS
from multioptpy.Calculator.ase_tools.nwchem import ASE_NWCHEM
from multioptpy.Calculator.ase_tools.gaussian import ASE_GAUSSIAN
from multioptpy.Calculator.ase_tools.orca import ASE_ORCA
from multioptpy.Calculator.ase_tools.fairchem import ASE_FAIRCHEM
from multioptpy.Calculator.ase_tools.mace import ASE_MACE
from multioptpy.Calculator.ase_tools.mopac import ASE_MOPAC
from multioptpy.Calculator.ase_tools.pygfn0 import ASE_GFN0
from multioptpy.Calculator.ase_tools.gxtb_dev import ASE_gxTB_Dev


"""
referrence:

 @inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }

ASE
Ask Hjorth Larsen, Jens Jørgen Mortensen, Jakob Blomqvist,
Ivano E. Castelli, Rune Christensen, Marcin Dułak, Jesper Friis,
Michael N. Groves, Bjørk Hammer, Cory Hargus, Eric D. Hermes,
Paul C. Jennings, Peter Bjerre Jensen, James Kermode, John R. Kitchin,
Esben Leonhard Kolsbjerg, Joseph Kubal, Kristen Kaasbjerg,
Steen Lysgaard, Jón Bergmann Maronsson, Tristan Maxson, Thomas Olsen,
Lars Pastewka, Andrew Peterson, Carsten Rostgaard, Jakob Schiøtz,
Ole Schütt, Mikkel Strange, Kristian S. Thygesen, Tejs Vegge,
Lasse Vilhelmsen, Michael Walter, Zhenhua Zeng, Karsten Wedel Jacobsen
The Atomic Simulation Environment—A Python library for working with atoms
J. Phys.: Condens. Matter Vol. 29 273002, 2017
 
"""

class Calculation:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2eV = UVL.hartree2eV

        self.START_FILE =  kwarg.get("START_FILE", None)
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY", "2GB")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL", "PBE")
        self.BASIS_SET = kwarg.get("BASIS_SET", "6-31G(d)")
        self.FC_COUNT = kwarg.get("FC_COUNT", 1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", None)
        self.Model_hess = kwarg.get("Model_hess", None)
        self.unrestrict = kwarg.get("unrestrict", None)
        self.software_type = kwarg.get("software_type", None)
        self.hessian_flag = False
        self.software_path_dict = read_software_path(kwarg.get("software_path_file", "./software_path.conf"))

    def calc_exact_hess(self, calc_obj, positions, element_list):
        
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]
        
        if self.software_type == "gaussian":
            print("Calculating exact Hessian using Gaussian...")
            exact_hess = calc_obj.calc_analytic_hessian()  # in hartree/Bohr^2
        elif self.software_type == "orca":
            hess_path = calc_obj.run_frequency_analysis()
            exact_hess = calc_obj.get_hessian_matrix(hess_path)
        else:
            vib = Vibrations(atoms=calc_obj.atom_obj, delta=0.001, name="z_hess_"+timestamp)
            vib.run()
            result_vib = vib.get_vibrations()
            exact_hess = result_vib.get_hessian_2d() # eV/Å²
            vib.clean()  
            exact_hess = exact_hess / self.hartree2eV * (self.bohr2angstroms ** 2)
        
        if type(element_list[0]) is str:
            exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, positions)
        else:
            exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, [number_element(elem_num) for elem_num in element_list], positions)
        exact_hess = (exact_hess + exact_hess.T) / 2  # make sure it's symmetric
        
        self.Model_hess = exact_hess
       
        return exact_hess

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
        """execute extended tight binding method calculation."""

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
            if True:#try:
                if geom_num_list is None:
                    positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    positions = geom_num_list        
               
                positions = np.array(positions, dtype="float64")  # ang.
                atom_obj = Atoms(element_list, positions)
                calc_obj = setup_calculator(
                    atom_obj,
                    self.software_type,
                    electric_charge_and_multiplicity,
                    input_file=input_file,
                    software_path_dict=self.software_path_dict,
                    functional=self.FUNCTIONAL,
                    basis_set=self.BASIS_SET,
                    set_memory=self.SET_MEMORY,
                )
                calc_obj.run()
                g = -1*calc_obj.atom_obj.get_forces(apply_constraint=False) * self.bohr2angstroms / self.hartree2eV  # eV/ang. to a.u.
                e = calc_obj.atom_obj.get_potential_energy(apply_constraint=False) / self.hartree2eV  # eV to hartree
                
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        _ = self.calc_exact_hess(calc_obj, positions, element_list)
                    
                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    # exact numerical hessian
                    _ = self.calc_exact_hess(calc_obj, positions, element_list)             
            #except Exception as error:
            #    print(error)
            #    print("This molecule could not be optimized.")
            #    finish_frag = True
            #    return np.array([0]), np.array([0]), np.array([0]), finish_frag 

        positions /= self.bohr2angstroms
        self.energy = e
        self.gradient = g
        self.coordinate = positions
        
        return e, g, positions, finish_frag



class CalculationEngine(ABC):
    #Base class for calculation engines
    
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        #Calculate energy and gradients
        pass
    
    def _get_file_list(self, file_directory):
        #Get list of input files
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) 
                   for i in range(1, 7)], [])
    
    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        #Process common visualization tasks
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



class ASEEngine(CalculationEngine):
    """ASE-based calculation engine supporting multiple quantum chemistry software packages.
    
    This engine uses the Atomic Simulation Environment (ASE) to interface with various
    quantum chemistry software packages like GAMESSUS, NWChem, Gaussian, ORCA, MACE, and MOPAC.
    """
    def __init__(self, **kwargs):
        UVL = UnitValueLib()
        self.software_path_file = kwargs.get("software_path_file", "./software_path.conf")
        self.software_path_dict = read_software_path(self.software_path_file)
        
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2eV = UVL.hartree2eV

    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        """Calculate energy and gradients using ASE and the configured software.
        
        Args:
            file_directory (str): Directory containing input files
            optimize_num (int): Optimization iteration number
            pre_total_velocity (np.ndarray): Previous velocities for dynamics
            config (object): Configuration object with calculation parameters
            
        Returns:
            tuple: (energy_list, gradient_list, geometry_num_list, pre_total_velocity)
        """
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        if not file_list:
            print("No input files found in directory.")
            return (np.array([], dtype="float64"), 
                    np.array([], dtype="float64"), 
                    np.array([], dtype="float64"), 
                    pre_total_velocity)
        
        # Get element list from the first file
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        
        hess_count = 0
        software_type = config.othersoft
        
        for num, input_file in enumerate(file_list):
            try:
                print(f"Processing file: {input_file}")
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                
                positions = np.array(positions, dtype="float64")  # in angstroms
                atom_obj = Atoms(element_list, positions)
                calc_obj = setup_calculator(
                    atom_obj,
                    software_type,
                    electric_charge_and_multiplicity,
                    input_file=input_file,
                    software_path_dict=self.software_path_dict,
                    functional=config.FUNCTIONAL,
                    basis_set=config.basisset,
                    set_memory=config.SET_MEMORY,
                )
                calc_obj.run()
                # Calculate energy and forces
                g = -1 * calc_obj.atom_obj.get_forces(apply_constraint=False) * self.bohr2angstroms / self.hartree2eV  # eV/ang. to a.u.
                e = calc_obj.atom_obj.get_potential_energy(apply_constraint=False) / self.hartree2eV  # eV to hartree

                # Store results
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))  # RMS
                geometry_num_list.append(positions / self.bohr2angstroms)  # Convert to Bohr
                num_list.append(num)
                
                # Handle hessian calculation if needed
                if config.FC_COUNT == -1 or isinstance(optimize_num, str):
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    # Calculate exact numerical hessian
                    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]
                    if software_type == "gaussian":
                        exact_hess = calc_obj.calc_analytic_hessian()  # in hartree/Bohr^2
                   
                    elif software_type == "orca":
                        hess_path = calc_obj.run_frequency_analysis()
                        exact_hess = calc_obj.get_hessian_matrix(hess_path)
                    else:
                        vib = Vibrations(atoms=calc_obj.atom_obj, delta=0.001, name="z_hess_"+timestamp)
                        vib.run()
                        result_vib = vib.get_vibrations()
                        exact_hess = result_vib.get_hessian_2d()  # eV/Å²
                        vib.clean()  
                    
                        # Convert hessian units
                        exact_hess = exact_hess / self.hartree2eV * (self.bohr2angstroms ** 2)

                    # Project out translational and rotational modes
                    calc_tools = Calculationtools()
                    exact_hess = calc_tools.project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, positions)
                       
                    
                    # Save hessian
                    np.save(os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{hess_count}.npy"), exact_hess)
                    hess_count += 1
                
            except Exception as error:
                print(f"Error: {error}")
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
        
        # Process visualization
        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)
        
        # Update velocities if needed
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
    


def setup_calculator(atom_obj, software_type, electric_charge_and_multiplicity, input_file=None, software_path_dict=None, functional=None, basis_set=None, set_memory=None):
    """Module-level helper to attach the appropriate calculator to an ASE Atoms object.

    Parameters mirror the previous inline branches. Returns the atoms object with
    .calc set.
    """
    software_path_dict = software_path_dict or {}

    if software_type == "gamessus":
        calc_obj = ASE_GAMESSUS(atom_obj=atom_obj,
                                electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                                gamessus_path=software_path_dict.get("gamessus"),
                                functional=functional,
                                basis_set=basis_set)

        return calc_obj
    if software_type == "nwchem":
        calc_obj = ASE_NWCHEM(atom_obj=atom_obj,
                              electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                              input_file=input_file,
                              functional=functional,
                              basis_set=basis_set,
                              memory=set_memory)

        return calc_obj
    if software_type == "gaussian":
        calc_obj = ASE_GAUSSIAN(atom_obj=atom_obj,
                                electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                                functional=functional,
                                basis_set=basis_set,
                                memory=set_memory,
                                software_path_dict=software_path_dict)
        return calc_obj

    if software_type == "orca":
        calc_obj = ASE_ORCA(atom_obj=atom_obj,
                            electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                            input_file=input_file,
                            orca_path=software_path_dict.get("orca"),
                            functional=functional,
                            basis_set=basis_set)
        
        return calc_obj

    if software_type == "uma-s-1":
        calc_obj = ASE_FAIRCHEM(atom_obj=atom_obj,
                                electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                                software_path=software_path_dict.get("uma-s-1"),
                                software_type=software_type)
        return calc_obj
    
    if software_type == "uma-s-1p1-cuda":
        calc_obj = ASE_FAIRCHEM(atom_obj=atom_obj,
                                electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                                software_path=software_path_dict.get("uma-s-1p1"),
                                software_type=software_type,
                                device_mode="cuda")
        return calc_obj
    
    if software_type == "uma-s-1p1":
        calc_obj = ASE_FAIRCHEM(atom_obj=atom_obj,
                                electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                                software_path=software_path_dict.get("uma-s-1p1"),
                                software_type=software_type)
        return calc_obj
    

    
    
    if software_type == "uma-m-1p1":
        calc_obj = ASE_FAIRCHEM(atom_obj=atom_obj,
                                electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                                software_path=software_path_dict.get("uma-m-1p1"),
                                software_type=software_type)

        return calc_obj
    if software_type == "mace_mp":
        calc_obj = ASE_MACE(atom_obj=atom_obj,
                             electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                             software_path=software_path_dict.get("mace_mp"),
                             software_type=software_type)
        return calc_obj
    if software_type == "mace_off":
        calc_obj = ASE_MACE(atom_obj=atom_obj,
                             electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                             software_path=software_path_dict.get("mace_off"),
                             software_type=software_type)
        
        return calc_obj
    if software_type == "GFN0-xTB":
        calc_obj = ASE_GFN0(atom_obj=atom_obj,
                             electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                             software_type=software_type)
        
        return calc_obj
    if software_type == "mopac":
        calc_obj = ASE_MOPAC(atom_obj=atom_obj,
                             electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                             input_file=input_file)
        return calc_obj

    if software_type == "gxtb_dev":
        calc_obj = ASE_gxTB_Dev(atom_obj=atom_obj,
                             electric_charge_and_multiplicity=electric_charge_and_multiplicity,
                             software_type=software_type)
        return calc_obj
    
    # Unknown software type
    raise ValueError(f"Unsupported software type: {software_type}")


