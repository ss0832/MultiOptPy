import glob
import os
import re
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
        
        self.START_FILE = kwarg["START_FILE"]
        self.N_THREAD = kwarg["N_THREAD"]
        self.SET_MEMORY = kwarg["SET_MEMORY"]
        self.FUNCTIONAL = kwarg["FUNCTIONAL"]
        self.BASIS_SET = kwarg["BASIS_SET"]
        self.FC_COUNT = kwarg["FC_COUNT"]
        self.BPA_FOLDER_DIRECTORY = kwarg["BPA_FOLDER_DIRECTORY"]
        self.Model_hess = kwarg["Model_hess"]
        self.unrestrict = kwarg["unrestrict"]
        self.software_type = kwarg["software_type"]
        self.hessian_flag = False
        self.software_path_dict = read_software_path(kwarg.get("software_path_file", "./software_path.conf"))

    def calc_exact_hess(self, atom_obj, positions, element_list):
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]
        vib = Vibrations(atoms=atom_obj, delta=0.001, name="z_hess_"+timestamp)
        vib.run()
        result_vib = vib.get_vibrations()
        exact_hess = result_vib.get_hessian_2d() # eV/Å²
        vib.clean()  
        exact_hess = exact_hess / self.hartree2eV * (self.bohr2angstroms ** 2)
        if type(element_list[0]) is str:
            exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, element_list, positions)
        else:
            exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, [number_element(elem_num) for elem_num in element_list], positions)
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
            try:
                if geom_num_list is None:
                    positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    positions = geom_num_list        
               
                positions = np.array(positions, dtype="float64")  # ang.
                atom_obj = Atoms(element_list, positions)
                atom_obj = setup_calculator(
                    atom_obj,
                    self.software_type,
                    electric_charge_and_multiplicity,
                    input_file=input_file,
                    software_path_dict=self.software_path_dict,
                    functional=self.FUNCTIONAL,
                    basis_set=self.BASIS_SET,
                    set_memory=self.SET_MEMORY,
                )

                e = atom_obj.get_potential_energy(apply_constraint=False) / self.hartree2eV  # eV to hartree
                g = -1*atom_obj.get_forces(apply_constraint=False) * self.bohr2angstroms / self.hartree2eV  # eV/ang. to a.u.
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        _ = self.calc_exact_hess(atom_obj, positions, element_list)
                    
                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    # exact numerical hessian
                    _ = self.calc_exact_hess(atom_obj, positions, element_list)             
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), np.array([0]), finish_frag 

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
                print(f"\n{input_file}\n")
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                
                positions = np.array(positions, dtype="float64")  # in angstroms
                atom_obj = Atoms(element_list, positions)
                atom_obj = setup_calculator(
                    atom_obj,
                    software_type,
                    electric_charge_and_multiplicity,
                    input_file=input_file,
                    software_path_dict=self.software_path_dict,
                    functional=config.FUNCTIONAL,
                    basis_set=config.basisset,
                    set_memory=config.SET_MEMORY,
                )

                # Calculate energy and forces
                e = atom_obj.get_potential_energy(apply_constraint=False) / self.hartree2eV  # eV to hartree
                g = -1 * atom_obj.get_forces(apply_constraint=False) * self.bohr2angstroms / self.hartree2eV  # eV/ang. to a.u.
                
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
                    vib = Vibrations(atoms=atom_obj, delta=0.001, name="z_hess_"+timestamp)
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
        return use_GAMESSUS(atom_obj, electric_charge_and_multiplicity, software_path_dict.get("gamessus"), functional, basis_set)
    if software_type == "nwchem":
        return use_NWChem(atom_obj, electric_charge_and_multiplicity, input_file, functional, basis_set, set_memory)
    if software_type == "gaussian":
        return use_Gaussian(atom_obj, electric_charge_and_multiplicity, functional, basis_set, set_memory)
    if software_type == "orca":
        return use_ORCA(atom_obj, electric_charge_and_multiplicity, input_file, software_path_dict.get("orca"), functional, basis_set)
    if software_type == "uma-s-1":
        return use_FAIRCHEMNNP(atom_obj, electric_charge_and_multiplicity, software_path_dict.get("uma-s-1"))
    if software_type == "uma-s-1p1":
        return use_FAIRCHEMNNP(atom_obj, electric_charge_and_multiplicity, software_path_dict.get("uma-s-1p1"))
    if software_type == "uma-m-1p1":
        return use_FAIRCHEMNNP(atom_obj, electric_charge_and_multiplicity, software_path_dict.get("uma-m-1p1"))
    if software_type == "mace_mp":
        return use_MACE_MP(atom_obj, electric_charge_and_multiplicity)
    if software_type == "mace_off":
        return use_MACE_OFF(atom_obj, electric_charge_and_multiplicity)
    if software_type == "mopac":
        return use_MOPAC(atom_obj, electric_charge_and_multiplicity, input_file)

    # Unknown software type
    raise ValueError(f"Unsupported software type: {software_type}")


    
def use_GAMESSUS(atom_obj, electric_charge_and_multiplicity, gamessus_path, functional, basis_set):
    """Set up and return ASE atoms object with GAMESSUS calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        gamessus_path: Path to GAMESSUS executable/userscr
        functional: DFT functional to use
        basis_set: Basis set to use
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from ase.calculators.gamess_us import GAMESSUS
    atom_obj.calc = GAMESSUS(userscr=gamessus_path,
                            contrl=dict(dfttyp=functional),
                            charge=electric_charge_and_multiplicity[0],
                            mult=electric_charge_and_multiplicity[1],
                            basis=basis_set)
    return atom_obj
    
def use_NWChem(atom_obj, electric_charge_and_multiplicity, input_file, functional, basis_set, memory):
    """Set up and return ASE atoms object with NWChem calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        input_file: Path to input file
        functional: DFT functional to use
        basis_set: Basis set to use
        memory: Memory specification string
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from ase.calculators.nwchem import NWChem
    input_dir = os.path.dirname(input_file)
    pattern = r"(\d+)([A-Za-z]+)"
    match = re.match(pattern, memory.lower())
    if match:
        number = match.group(1)
        unit = match.group(2)
    else:
        raise ValueError("Invalid memory string format")

    calc = NWChem(label=input_dir,
                  xc=functional,
                  charge=electric_charge_and_multiplicity[0],
                  basis=basis_set,
                  memory=number+" "+unit)
    atom_obj.set_calculator(calc)
    return atom_obj

def use_Gaussian(atom_obj, electric_charge_and_multiplicity, functional, basis_set, memory):
    """Set up and return ASE atoms object with Gaussian calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        functional: DFT functional to use
        basis_set: Basis set to use
        memory: Memory specification string
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from ase.calculators.gaussian import Gaussian
    atom_obj.calc = Gaussian(xc=functional,
                           basis=basis_set,
                           scf='maxcycle=500',
                           mem=memory)
    return atom_obj
    
def use_ORCA(atom_obj, electric_charge_and_multiplicity, input_file, orca_path, functional, basis_set):
    """Set up and return ASE atoms object with ORCA calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        input_file: Path to input file
        orca_path: Path to ORCA executable
        functional: DFT functional to use
        basis_set: Basis set to use
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from ase.calculators.orca import ORCA
    input_dir = os.path.dirname(input_file)
    atom_obj.calc = ORCA(label=input_dir,
                        profile=orca_path,
                        charge=int(electric_charge_and_multiplicity[0]),
                        mult=int(electric_charge_and_multiplicity[1]),
                        orcasimpleinput=functional+' '+basis_set)
                        #orcablocks='%pal nprocs 16 end')
    return atom_obj

def use_MACE_MP(atom_obj, electric_charge_and_multiplicity):
    """Set up and return ASE atoms object with MACE_MP calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from mace.calculators import mace_mp
    macemp = mace_mp()
    atom_obj.calc = macemp
    return atom_obj

def use_FAIRCHEMNNP(atom_obj, electric_charge_and_multiplicity, fairchem_path): # fairchem.core: version 2.2.0
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError:
        raise ImportError("FAIRChem.core modules not found")
    # Load the prediction unit
    predict_unit = load_predict_unit(path=fairchem_path, device="cpu")
    
    # Set up the FAIRChem calculator
    fairchem_calc = FAIRChemCalculator(predict_unit=predict_unit, task_name="omol")
    atom_obj.info = {"charge": int(electric_charge_and_multiplicity[0]),
                     "spin": int(electric_charge_and_multiplicity[1])}
    atom_obj.calc = fairchem_calc
    return atom_obj

def use_MACE_OFF(atom_obj, electric_charge_and_multiplicity):
    """Set up and return ASE atoms object with MACE_OFF calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from mace.calculators import mace_off
    maceoff = mace_off()
    atom_obj.calc = maceoff
    return atom_obj

def use_MOPAC(atom_obj, electric_charge_and_multiplicity, input_file):
    """Set up and return ASE atoms object with MOPAC calculator.
    
    Args:
        atom_obj: ASE Atoms object
        electric_charge_and_multiplicity: List with [charge, multiplicity]
        input_file: Path to input file
        
    Returns:
        ASE Atoms object with calculator attached
    """
    from ase.calculators.mopac import MOPAC
    input_dir = os.path.dirname(input_file)
    atom_obj.calc = MOPAC(label=input_dir,
                        task="1SCF GRADIENTS DISP",
                        charge=int(electric_charge_and_multiplicity[0]),
                        mult=int(electric_charge_and_multiplicity[1]))
    return atom_obj



