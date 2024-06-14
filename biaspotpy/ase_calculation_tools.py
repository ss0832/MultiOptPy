import glob
import os
import copy
import re
import numpy as np

from ase import Atoms

from calc_tools import Calculationtools
from parameter import UnitValueLib, element_number
from fileio import read_software_path


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
        self.software_path_dict = read_software_path()

    
    
    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method, geom_num_list=None):
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
                if geom_num_list is None:
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
                else:
                    positions = geom_num_list        
               
                positions = np.array(positions, dtype="float64")#ang.
              

                atom_obj = Atoms(element_list, positions)

                if self.software_type == "gamessus":
                    atom_obj = self.use_GAMESSUS(atom_obj, electric_charge_and_multiplicity)

                elif self.software_type == "nwchem":
                    atom_obj = self.use_NWChem(atom_obj, electric_charge_and_multiplicity, input_file)
                    
                elif self.software_type == "gaussian":
                    atom_obj = self.use_Gaussian(atom_obj, electric_charge_and_multiplicity)
                    
                elif self.software_type == "orca":
                    atom_obj = self.use_ORCA(atom_obj, electric_charge_and_multiplicity, input_file)
                    
                elif self.software_type == "mace_mp":#Neural Network Potential
                    atom_obj = self.use_MACE_MP(atom_obj, electric_charge_and_multiplicity)
                    
                elif self.software_type == "mace_off":#Neural Network Potential
                    atom_obj = self.use_MACE_OFF(atom_obj, electric_charge_and_multiplicity)
        
                elif self.software_type == "mopac":
                    atom_obj = self.use_MOPAC(atom_obj, electric_charge_and_multiplicity, input_file)
                else:
                    print("This software isn't available...")
                    raise

                e = atom_obj.get_potential_energy(apply_constraint=False) / self.hartree2eV # eV to hartree
                g = -1*atom_obj.get_forces(apply_constraint=False) * self.bohr2angstroms / self.hartree2eV  # eV/ang. to a.u.
                
                
                """
                if self.FC_COUNT == -1 or type(iter) is str:
                    pass
                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    #exact numerical hessian
                    exact_hess = self.numerical_hessian(positions, element_number_list, method, electric_charge_and_multiplicity)

                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot(exact_hess, element_number_list.tolist(), positions)
                    self.Model_hess = exact_hess
                """
                                 
                
                


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
    
 
    def use_GAMESSUS(self, atom_obj, electric_charge_and_multiplicity):
        from ase.calculators.gamess_us import GAMESSUS
        atom_obj.calc = GAMESSUS(userscr=self.software_path_dict["gamessus"],
                                contrl = dict(dfttyp=self.FUNCTIONAL),
                                charge = electric_charge_and_multiplicity[0],
                                mult = electric_charge_and_multiplicity[1],
                                basis = self.BASIS_SET)
        return atom_obj
        
    def use_NWChem(self, atom_obj, electric_charge_and_multiplicity, input_file):                    
        from ase.calculators.nwchem import NWChem
        input_dir = os.path.dirname(input_file)
        pattern = r"(\d+)([A-Za-z]+)"
        match = re.match(pattern, self.SET_MEMORY.lower())
        if match:
            # 数字と文字を取得
            number = match.group(1)
            unit = match.group(2)
        
        else:
            raise ValueError("Invalid memory string format")

        calc = NWChem(label=input_dir,
                                xc=self.FUNCTIONAL,
                                charge = electric_charge_and_multiplicity[0],
                                basis = self.BASIS_SET,
                                memory=number+" "+unit)
        atom_obj.set_calculator(calc)
        return atom_obj
    
    def use_Gaussian(self, atom_obj, electric_charge_and_multiplicity):                               
        from ase.calculators.gaussian import Gaussian
        atom_obj.calc = Gaussian(xc=self.FUNCTIONAL,
                                basis = self.BASIS_SET,
                                scf='maxcycle=500',
                                mem=self.SET_MEMORY)
        return atom_obj
        
    def use_ORCA(self, atom_obj, electric_charge_and_multiplicity, input_file):    
        from ase.calculators.orca import ORCA
        input_dir = os.path.dirname(input_file)
        atom_obj.calc = ORCA(label=input_dir,
                            profile=self.software_path_dict["orca"],
                            charge = int(electric_charge_and_multiplicity[0]),
                            mult = int(electric_charge_and_multiplicity[1]),
                            orcasimpleinput=self.FUNCTIONAL+' '+self.BASIS_SET,)
                            #orcablocks='%pal nprocs 16 end')
        return atom_obj
    
    def use_MACE_MP(self, atom_obj, electric_charge_and_multiplicity):    
        from mace.calculators import mace_mp
        macemp = mace_mp()
        atom_obj.calc = macemp
        return atom_obj
    
    def use_MACE_OFF(self, atom_obj, electric_charge_and_multiplicity):
        from mace.calculators import mace_off
        maceoff = mace_off()
        atom_obj.calc = maceoff
        return atom_obj
    
    def use_MOPAC(self, atom_obj, electric_charge_and_multiplicity, input_file):
        from ase.calculators.mopac import MOPAC
        input_dir = os.path.dirname(input_file)
        atom_obj.calc = MOPAC(label=input_dir,
                            task="1SCF GRADIENTS DISP",
                            charge = int(electric_charge_and_multiplicity[0]),
                            mult = int(electric_charge_and_multiplicity[1]),)
        return atom_obj
