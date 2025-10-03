import os
import numpy as np


# If you want to use Gaussian with ASE, you need to have ASE installed and Gaussian 16.
# ASE can be installed via pip:
# pip install ase
# Gaussian 16 must be installed separately and is not free software.
# Make sure to set the GAUSS_EXEDIR environment variable to point to the directory containing the Gaussian executables.
# You must write the directory of the Gaussian executables in the software_path.conf file.
# (e.x., C:\g16\bin for Windows or /usr/local/g16/g16 for Linux)

class ASE_GAUSSIAN:
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.functional = kwargs.get('functional', None)
        self.basis_set = kwargs.get('basis_set', None)
        self.memory = kwargs.get('memory', None)
        self.software_path_dict = kwargs.get('software_path_dict', None)
        self.gau_prog = "g16"  # Default to Gaussian 16
        from ase.calculators.gaussian import Gaussian
        self.Gaussian = Gaussian

        
    def run(self):
        if os.name == "nt":  # windows (for Gaussian16W)
            os.environ['GAUSS_EXEDIR'] = self.software_path_dict.get("gaussian", "")
            input_file = self.atom_obj.info.get('input_file', 'unknown')
            abs_path = os.path.abspath(input_file)
            #input_file_name = os.path.basename(abs_path)
            input_dir = os.path.dirname(abs_path)
            self.atom_obj.calc = self.Gaussian(xc=self.functional,
                            basis=self.basis_set,
                            scf='xqc,maxcon=128,maxcyc=32,conver=8',
                            mem=self.memory,
                            command=f'{self.gau_prog} {input_dir}\\Gaussian.com {input_dir}\\Gaussian.log',
                                charge=int(self.electric_charge_and_multiplicity[0]),
                                mult=int(self.electric_charge_and_multiplicity[1])
                            )
            
        elif os.name == "posix":  # Linux (for Gaussian16)
            os.environ['GAUSS_EXEDIR'] = self.software_path_dict.get("gaussian", "")
            input_file = self.atom_obj.info.get('input_file', 'unknown')
            abs_path = os.path.abspath(input_file)
            #input_file_name = os.path.basename(abs_path)
            input_dir = os.path.dirname(abs_path)
            atom_obj.calc = self.Gaussian(xc=functional,
                            basis=basis_set,
                            scf='xqc,maxcon=128,maxcyc=32,conver=8',
                            mem=memory,
                            command=f'{self.gau_prog} < {input_dir}/Gaussian.com > {input_dir}/Gaussian.log',
                                charge=int(electric_charge_and_multiplicity[0]),
                                mult=int(electric_charge_and_multiplicity[1])
                            )
        else:
            raise EnvironmentError("Unsupported operating system")

        return self.atom_obj
    
    
    def calc_analytic_hessian(self):
        """Calculate and return the analytic Hessian matrix."""
        
        if self.atom_obj.calc is None:
            raise ValueError("Calculator not set. Please run the 'run' method first.")
        
        if os.name == "nt":  # windows (for Gaussian16W)
            os.environ['GAUSS_EXEDIR'] = self.software_path_dict.get("gaussian", "")
            input_file = self.atom_obj.info.get('input_file', 'unknown')
            abs_path = os.path.abspath(input_file)
            input_dir = os.path.dirname(abs_path)
            self.atom_obj.calc = self.Gaussian(xc=self.functional,
                            basis=self.basis_set,
                            scf='xqc,maxcon=128,maxcyc=32,conver=8',
                            extra='freq=noraman',
                            mem=self.memory,
                            command=f'{self.gau_prog} {input_dir}\\Gaussian.com {input_dir}\\Gaussian.log',
                                charge=int(self.electric_charge_and_multiplicity[0]),
                                mult=int(self.electric_charge_and_multiplicity[1]),
                               
                            )
            self.atom_obj.calc.calculate(self.atom_obj)
       
        elif os.name == "posix":  # Linux (for Gaussian16)
            os.environ['GAUSS_EXEDIR'] = self.software_path_dict.get("gaussian", "")
            input_file = self.atom_obj.info.get('input_file', 'unknown')
            abs_path = os.path.abspath(input_file)
            input_dir = os.path.dirname(abs_path)
            self.atom_obj.calc = self.Gaussian(xc=self.functional,
                            basis=self.basis_set,
                            scf='xqc,maxcon=128,maxcyc=32,conver=8',
                            extra='freq=noraman',
                            mem=self.memory,
                            command=f'{self.gau_prog} < {input_dir}/Gaussian.com > {input_dir}/Gaussian.log',
                                charge=int(self.electric_charge_and_multiplicity[0]),
                                mult=int(self.electric_charge_and_multiplicity[1]),
                                
                            )
            self.atom_obj.calc.calculate(self.atom_obj)
        else:   # Unsupported OS
            raise EnvironmentError("Unsupported operating system")
        
        hessian = self._extract_hessian_from_output(input_dir + "/Gaussian")

        return hessian # in hartree/Bohr^2

    def _extract_hessian_from_output(self, label):
        log_file = f"{label}.log"
        
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Gaussian output file {log_file} not found")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find Hessian in output
        reading_hessian = False
        
        counter_1 = 0
        counter_2 = 0
        
        for i, line in enumerate(lines):
            # Determine number of atoms
            if 'NAtoms=' in line:
                n_atoms = int(line.split('NAtoms=')[1].split()[0])
                n_coords = n_atoms * 3
                hessian = np.zeros((n_coords, n_coords))
            
            # Locate Hessian matrix section
            if 'Force constants in Cartesian coordinates:' in line:
                reading_hessian = True
                tmp_i = i
                continue
            
            if reading_hessian and tmp_i + 1 < i:
                # Parse Hessian data
                if line.strip() == '':
                    reading_hessian = False
                    continue
                
                parts = line.split()
                atom_counter_i = 0 + 5 * counter_1
                
                if len(parts) > 1:
                    
                    if "D" in parts[1]:
                        values = [float(x.replace('D', 'E')) for x in parts[1:]]
                        values = np.array(values, dtype=np.float64)
                        
                        values_len = len(values)
                    
                        hessian[(atom_counter_i + counter_2), atom_counter_i:(atom_counter_i + values_len)] = values
                        hessian[atom_counter_i:(atom_counter_i + values_len), (atom_counter_i + counter_2)] = values
                        counter_2 += 1
                    else:
                        counter_1 += 1
                        counter_2 = 0
                    
    
        hessian = (hessian + hessian.T) / 2  # Symmetrize Hessian
        
        return hessian #hartree/Bohr^2
    
    
