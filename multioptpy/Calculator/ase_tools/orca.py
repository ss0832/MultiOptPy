import os
import shutil
import subprocess
import numpy as np
from ase.calculators.orca import ORCA, OrcaProfile
from ase.data import atomic_numbers

"""
Please specify the absolute path to the ORCA executable in software_path.conf using the format orca::<path>. For Linux, provide the path to the binary (e.g., /absolute/path/to/orca), and for Windows, provide the path to the executable file (e.g., C:\absolute\path\to\orca.exe).
"""


class ASE_ORCA:
    TARGET_ORCA_VERSION = '6.1.0'

    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        
        # NOTE: self.input_file is updated in _setup_calculator to enforce 'orca.inp' in CWD.
        raw_input_file = kwargs.get('input_file', 'orca.inp')
        self.input_file = os.path.abspath(raw_input_file).replace('\\', '/')
        
        self.orca_path = kwargs.get('orca_path', None)
        self.functional = kwargs.get('functional', 'B3LYP')
        self.basis_set = kwargs.get('basis_set', 'def2-SVP')

        # Optional ORCA input blocks (raw string)
        self.orca_blocks = kwargs.get('orca_blocks', '')

        # Auto-fix for unsupported Pople basis sets on heavy elements
        self.auto_fix_basis = kwargs.get('auto_fix_basis', True)
        self.heavy_atom_basis = kwargs.get('heavy_atom_basis', 'def2-SVP')

    def _resolve_orca_exe(self, provided_path):
        """
        Resolve the absolute path to the ORCA executable.
        Handles directories, stripping whitespace from config files, and WSL/Windows paths.
        """
        if not provided_path:
            return None
        
        # CRITICAL FIX: Strip whitespace/newlines that might come from config parsing
        clean_path = provided_path.strip()
        # Expand ~ to home directory if present
        clean_path = os.path.expanduser(clean_path)
        clean_path = os.path.normpath(clean_path)

        candidates = []
        # If the path is a directory, look for the executable inside it
        if os.path.isdir(clean_path):
            candidates.append(os.path.join(clean_path, 'orca'))
            candidates.append(os.path.join(clean_path, 'orca.exe'))
        else:
            # If it's a file path (or doesn't exist yet), use it as is
            candidates.append(clean_path)

        for candidate in candidates:
            # Check if file exists
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return os.path.abspath(candidate)
            
            # Check system PATH
            resolved = shutil.which(candidate)
            if resolved and os.path.exists(resolved):
                return os.path.abspath(resolved)

        # Use repr() in error message to reveal hidden characters like \n
        candidate_reprs = [repr(c) for c in candidates]
        raise FileNotFoundError(f"Cannot locate ORCA executable. Checked: {', '.join(candidate_reprs)}")

    def _is_pople_basis(self, basis_name):
        if not basis_name: return False
        b = basis_name.strip().lower()
        return b.startswith("6-31") or b.startswith("6-311")

    def _get_heavy_elements_for_pople(self):
        if self.atom_obj is None: return []
        symbols = self.atom_obj.get_chemical_symbols()
        return sorted({s for s in symbols if atomic_numbers.get(s, 0) > 30})

    def _build_orca_blocks(self):
        blocks = (self.orca_blocks or "").strip()
        heavy_elements = []
        if self._is_pople_basis(self.basis_set):
            heavy_elements = self._get_heavy_elements_for_pople()

        if heavy_elements:
            if not self.auto_fix_basis:
                raise ValueError("Unsupported Pople basis for heavy elements.")

            basis_lines = ["%basis"]
            for elem in heavy_elements:
                # Per user instruction: No ECP, just GTO
                basis_lines.append(f'  NewGTO {elem} "{self.heavy_atom_basis}"')
            basis_lines.append("end")

            if blocks:
                blocks = blocks + "\n" + "\n".join(basis_lines)
            else:
                blocks = "\n".join(basis_lines)

        return blocks if blocks else ""

    def _print_orca_output_on_error(self):
        """Helper to print ORCA output file content if it exists."""
        if not hasattr(self, 'input_file'): return
        
        out_file = os.path.splitext(self.input_file)[0] + ".out"
        if os.path.exists(out_file):
            print(f"\n--- ORCA OUTPUT ({out_file}) ---")
            try:
                with open(out_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    print(content[-3000:] if len(content) > 3000 else content)
            except Exception as e:
                print(f"Failed to read output file: {e}")
            print("--- END ORCA OUTPUT ---\n")
        else:
            print(f"\n--- ORCA OUTPUT NOT FOUND ({out_file}) ---\n")

    def _setup_calculator(self, task_keyword):
        # Force usage of Current Working Directory + "orca.inp"
        cwd = os.getcwd()
        label_path = os.path.join(cwd, 'orca').replace('\\', '/')
        self.input_file = label_path + '.inp'
        
        print(f"DEBUG: ASE Label Path : {label_path}")

        simple_input = f"{self.functional} {self.basis_set} {task_keyword}"
        
        profile_obj = None
        if self.orca_path:
            real_exe_path = self._resolve_orca_exe(self.orca_path)
            orca_dir = os.path.dirname(real_exe_path)
            path_env = os.environ.get('PATH', '')
            if orca_dir not in path_env:
                os.environ['PATH'] = orca_dir + os.pathsep + path_env

            ase_safe_path = real_exe_path.replace('\\', '/')
            profile_obj = OrcaProfile(ase_safe_path)
            print(f"DEBUG: ORCA Executable: {ase_safe_path}")

        orca_blocks = self._build_orca_blocks()
        
        calc = ORCA(
            label=label_path,
            profile=profile_obj,
            charge=int(self.electric_charge_and_multiplicity[0]),
            mult=int(self.electric_charge_and_multiplicity[1]),
            orcasimpleinput=simple_input,
            orcablocks=orca_blocks
        )
        
        self.atom_obj.calc = calc
        return self.atom_obj

    def run(self):
        self._setup_calculator("EnGrad")
        print(f"--- Starting Gradient Calculation (ORCA {self.TARGET_ORCA_VERSION}) ---")
        try:
            forces = self.atom_obj.get_forces()
            potential_energy = self.atom_obj.get_potential_energy()
            print("Gradient calculation completed.")
            return forces, potential_energy
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL: ORCA execution failed with exit code {e.returncode}")
            self._print_orca_output_on_error()
            raise e
        except Exception as e:
            print(f"CRITICAL: An unexpected error occurred: {e}")
            self._print_orca_output_on_error()
            raise e

    def run_frequency_analysis(self):
        print(f"--- Starting Frequency Calculation (ORCA {self.TARGET_ORCA_VERSION}) ---")
        self._setup_calculator("Freq")
        try:
            self.atom_obj.get_potential_energy()
            print("Frequency calculation completed.")
            # Use self.input_file to construct hess_path instead of relying on calc.label
            hess_path = os.path.splitext(self.input_file)[0] + ".hess"
            return hess_path
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL: ORCA Frequency execution failed with exit code {e.returncode}")
            self._print_orca_output_on_error()
            raise e
        except Exception as e:
            print(f"CRITICAL: An unexpected error occurred during frequency analysis: {e}")
            self._print_orca_output_on_error()
            raise e

    def get_hessian_matrix(self, hess_file_path=None):
        if hess_file_path is None:
            # Default to orca.hess in the same dir as input_file
            input_dir = os.path.dirname(self.input_file)
            hess_file_path = os.path.join(input_dir, 'orca.hess')

        if not os.path.exists(hess_file_path):
            raise FileNotFoundError(f"Hessian file not found: {hess_file_path}")

        print(f"Reading Hessian from: {hess_file_path}")
        
        with open(hess_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        hessian_matrix = None
        iterator = iter(lines)
        for line in iterator:
            if "$hessian" in line:
                dim_line = next(iterator).strip().split()
                # Parse dimensions: square matrix usually provides just one dimension
                if len(dim_line) == 1:
                    n_rows = int(dim_line[0])
                    n_cols = n_rows
                else:
                    n_rows, n_cols = map(int, dim_line[:2])
                
                hessian_matrix = np.zeros((n_rows, n_cols))
                col_pointer = 0
                while col_pointer < n_cols:
                    header = next(iterator).strip()
                    if not header: break
                    col_indices = [int(c) for c in header.split()]
                    for r in range(n_rows):
                        row_data = next(iterator).strip().split()
                        row_idx = int(row_data[0])
                        values = [float(x) for x in row_data[1:]]
                        for i, val in enumerate(values):
                            hessian_matrix[row_idx, col_indices[i]] = val
                    col_pointer += len(col_indices)
                break

        if hessian_matrix is None:
            raise ValueError("Could not find $hessian block in the file.")
        return hessian_matrix