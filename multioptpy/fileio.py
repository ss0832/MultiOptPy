import glob
import os
import re
import numpy as np

from scipy.signal import argrelextrema
from multioptpy.Utils.calc_tools import calc_RMS

def save_bias_pot_info(file_path, energy, gradient, bias_pot_id):
    max_grad = np.max(np.abs(gradient))
    rms_grad = calc_RMS(gradient)
    save_path = file_path+"bias_pot_info_"+str(bias_pot_id)+".log"
    
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write("Energy, MaxGrad, RMSGrad\n")
            
    with open(save_path, "a") as f:
        f.write(str(energy)+","+str(max_grad)+","+str(rms_grad)+"\n")
    return

def save_bias_param_grad_info(file_path, gradient, bias_pot_id):
    save_path = file_path+"bias_param_grad_info_"+str(bias_pot_id)+".log"
    
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write("Gradient\n")
    with open(save_path, "a") as f:
        f.write(str(gradient)+"\n")
    return

def read_software_path(file_path="./software_path.conf"):
    print("Reading software path from", file_path)
    with open(file_path, "r") as f:
        words = f.read().splitlines()
    software_path_dict = {}
    for word in words:
        tmp_split = word.split("::")
        soft_name = tmp_split[0]
        soft_path = tmp_split[1]
        software_path_dict[soft_name] = soft_path
    return software_path_dict

def xyz2list(file_path, args_electric_charge_and_multiplicity):
    pattern_cs = get_pattern_cs()
    pattern_xyz = get_pattern_xyz()
    electric_charge_and_multiplicity = None
    element_list = []
    with open(file_path, "r") as f:
        words = f.read().splitlines()
    geometry_list = []
    for word in words:
        if re.match(pattern_cs, word):
            electric_charge_and_multiplicity = list(map(str, word.split()))
        if re.match(pattern_xyz, word):
            geometry_list.append(word.split()[1:4])
            element_list.append(word.split()[0])
    if electric_charge_and_multiplicity is None:
        electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
   
    return geometry_list, element_list, electric_charge_and_multiplicity
import re



def _parse_gamess(lines):
    """Internal function to parse GAMESS input."""
    pattern_atom = get_pattern_gamess_atom()
    element_list = []
    geometry_list = []
    
    is_data_section = False
    for line in lines:
        if "$DATA" in line.upper(): is_data_section = True; continue
        if "$END" in line.upper() and is_data_section: break
        if is_data_section:
            match = pattern_atom.match(line)
            if match:
                element_list.append(match.group(1))
                geometry_list.append([match.group(2), match.group(3), match.group(4)])
    return geometry_list, element_list

def _parse_orca(lines):
    """Internal function to parse ORCA input."""
    pattern_atom = get_pattern_orca_atom()
    element_list = []
    geometry_list = []
    electric_charge_and_multiplicity = ["0", "1"] # Default

    is_coord_section = False
    for line in lines:
        # Check for start of coordinate block, e.g., *xyz 0 1
        if line.strip().startswith("*xyz"):
            is_coord_section = True
            parts = line.strip().split()
            if len(parts) == 3:
                electric_charge_and_multiplicity = [parts[1], parts[2]]
            continue
        
        # Check for end of coordinate block
        if is_coord_section and line.strip() == "*":
            break
            
        if is_coord_section:
            match = pattern_atom.match(line)
            if match:
                element_list.append(match.group(1))
                geometry_list.append([match.group(2), match.group(3), match.group(4)])
    return geometry_list, element_list, electric_charge_and_multiplicity

def _parse_qchem(lines):
    """Internal function to parse Q-Chem input."""
    pattern_atom = get_pattern_qchem_atom()
    element_list = []
    geometry_list = []
    electric_charge_and_multiplicity = ["0", "1"] # Default

    is_molecule_section = False
    for line in lines:
        if "$molecule" in line.lower():
            is_molecule_section = True
            # Read charge and multiplicity from the next line
            charge_mult_line = next(iter(lines), "").strip()
            parts = charge_mult_line.split()
            if len(parts) == 2:
                electric_charge_and_multiplicity = parts
            continue
            
        if "$end" in line.lower() and is_molecule_section:
            break
            
        if is_molecule_section:
            # Skip the charge/multiplicity line itself
            if re.match(r"^\s*[+-]?\d+\s+[+-]?\d+\s*$", line.strip()):
                continue
            match = pattern_atom.match(line)
            if match:
                element_list.append(match.group(1))
                geometry_list.append([match.group(2), match.group(3), match.group(4)])
    return geometry_list, element_list, electric_charge_and_multiplicity


def inp2list(file_path, args_electric_charge_and_multiplicity=["0", "1"]):
    """
    Automatically detects the input file format (GAMESS, ORCA, Q-Chem)
    and parses the atomic coordinates.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.splitlines()
    
    # --- Format Detection Logic ---
    detected_format = None
    if "$CONTRL" in content and "$DATA" in content:
        detected_format = "gamess"
    elif re.search(r"^\s*!", content, re.MULTILINE) and "*xyz" in content:
        detected_format = "orca"
    elif "$molecule" in content:
        detected_format = "qchem"
    else:
        print("Error: Could not determine the file format.")
        return [], [], None, None

    # --- Parsing based on detected format ---
    if detected_format == "gamess":
        print("Detected format: GAMESS")
        geometry_list, element_list = _parse_gamess(lines)
        # GAMESS does not have a standard charge/multiplicity line in $DATA
        electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
    
    elif detected_format == "orca":
        print("Detected format: ORCA")
        geometry_list, element_list, electric_charge_and_multiplicity = _parse_orca(lines)

    elif detected_format == "qchem":
        print("Detected format: Q-Chem")
        geometry_list, element_list, electric_charge_and_multiplicity = _parse_qchem(lines)

    return geometry_list, element_list, electric_charge_and_multiplicity

def mol2list(file_path, args_electric_charge_and_multiplicity):
    """Parses a MOL file (.mol)."""
    pattern_atom = get_pattern_mol_atom()
    
    element_list = []
    geometry_list = []
    electric_charge_and_multiplicity = args_electric_charge_and_multiplicity

    with open(file_path, "r") as f:
        lines = f.readlines()
        
    # Get the number of atoms from the counts line (4th line)
    try:
        num_atoms = int(lines[3].strip().split()[0])
    except (IndexError, ValueError):
        # Return empty lists for an invalid format
        return [], [], electric_charge_and_multiplicity

    # Process the atom block
    atom_block_lines = lines[4 : 4 + num_atoms]
    for line in atom_block_lines:
        match = pattern_atom.match(line)
        if match:
            # MOL format order is: X, Y, Z, Symbol
            geometry_list.append([match.group(1), match.group(2), match.group(3)])
            element_list.append(match.group(4))
            
    return geometry_list, element_list, electric_charge_and_multiplicity

def mol22list(file_path, args_electric_charge_and_multiplicity):
    """Parses a MOL2 file (.mol2)."""
    pattern_atom = get_pattern_mol2_atom()
    
    element_list = []
    geometry_list = []
    electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
    
    is_atom_section = False
    with open(file_path, "r") as f:
        for line in f:
            # Detect the start of the ATOM section
            if "@<TRIPOS>ATOM" in line:
                is_atom_section = True
                continue
            
            # End processing if another section starts
            if is_atom_section and "@<TRIPOS>" in line:
                break
                
            # Process atom lines within the ATOM section
            if is_atom_section:
                match = pattern_atom.match(line)
                if match:
                    # Extract the element symbol from the atom name (e.g., "C1", "Oa")
                    atom_name = match.group(1)
                    element = "".join(filter(str.isalpha, atom_name))
                    element_list.append(element)
                    
                    # MOL2 format order is: Name, X, Y, Z
                    geometry_list.append([match.group(2), match.group(3), match.group(4)])
                    
    return geometry_list, element_list, electric_charge_and_multiplicity

def traj2list(file_path, args_electric_charge_and_multiplicity):
    pattern_cs = get_pattern_cs()
    pattern_xyz = get_pattern_xyz()
    
    electric_charge_and_multiplicity = None
    cs_flag = True
    
    with open(file_path, "r") as f:
        words = f.read().splitlines()
        
    geometry_list = []
    element_list = []
    geometries = []
    elements = []
    for word in words:
        if re.match(pattern_cs, word) and cs_flag:
            electric_charge_and_multiplicity = list(map(str, word.split()))
            cs_flag = False
        if re.match(pattern_xyz, word):
            geometry_list.append(word.split()[1:4])
            element_list.append(word.split()[0])
        else:
            if len(geometry_list) != 0:
                geometries.append(geometry_list)
            if len(element_list) != 0:
                elements.append(element_list)
           
            geometry_list = []
            element_list = []
        
    if electric_charge_and_multiplicity is None:
        electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
   
    return geometries, elements, electric_charge_and_multiplicity

def get_pattern_xyz():
    pattern_xyz = re.compile(r"\s*([A-Za-z]+)\s+([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)(?:\s+([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?))(?:\s+([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?))\s*")
    return pattern_xyz

def get_pattern_cs():
    pattern_cs = re.compile(r"-*[0-9]+\s+-*[0-9]+\s*")
    return pattern_cs


def get_pattern_qchem_atom():
    """Returns a regex pattern for Q-Chem atom lines."""
    coordinate_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"
    pattern = re.compile(
        r"^\s*([A-Za-z]+)\s+"      # Element
        + coordinate_pattern + r"\s+" # X
        + coordinate_pattern + r"\s+" # Y
        + coordinate_pattern + r"\s*"  # Z
    )
    return pattern

def get_pattern_orca_atom():
    """Returns a regex pattern for ORCA atom lines."""
    coordinate_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"
    pattern = re.compile(
        r"^\s*([A-Za-z]+)\s+"      # Element
        + coordinate_pattern + r"\s+" # X
        + coordinate_pattern + r"\s+" # Y
        + coordinate_pattern + r"\s*"  # Z
    )
    return pattern

def get_pattern_gamess_atom():
    """
    Returns a regex pattern to match atom lines in a GAMESS input file.
    Supports both decimal and scientific notation for coordinates.
    """
    # Example: "O 8.0 0.0 0.0 0.0" or "C 6.0 1.234e+01 -5.67E-02 8.9"
    coordinate_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"
    
    pattern = re.compile(
        r"^\s*([A-Za-z]+)\s+"      # Element
        r"[+-]?\d+\.\d+\s+"         # Atomic Number
        + coordinate_pattern + r"\s+" # X coordinate
        + coordinate_pattern + r"\s+" # Y coordinate
        + coordinate_pattern + r"\s*"  # Z coordinate
    )
    return pattern


def get_pattern_mol_atom():
    """
    Returns a regex pattern to match atom lines in a MOL/SDF file.
    Supports both decimal and scientific notation for coordinates.
    """
    # Example: " 0.0000 0.0000 0.0000 O ..." or " 1.23e-05 -4.56E+00 7.89 O ..."
    coordinate_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"

    pattern = re.compile(
        r"^\s*" + coordinate_pattern + r"\s+" # X coordinate
        + coordinate_pattern + r"\s+"         # Y coordinate
        + coordinate_pattern + r"\s+"         # Z coordinate
        r"([A-Za-z]+)\s+.*"                   # Element
    )
    return pattern

def get_pattern_mol2_atom():
    """
    Returns a regex pattern to match atom lines in a MOL2 file.
    Supports both decimal and scientific notation for coordinates.
    """
    # Example: " 1 O 0.0000 0.0000 0.0000 O.3 ..." or " 2 C 1.2e1 -3.4E-1 5.6 C.ar ..."
    coordinate_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"

    pattern = re.compile(
        r"^\s*\d+\s+"                 # Atom ID
        r"([A-Za-z]+)\w*\s+"          # Atom Name
        + coordinate_pattern + r"\s+" # X coordinate
        + coordinate_pattern + r"\s+" # Y coordinate
        + coordinate_pattern + r"\s+.*" # Z coordinate
    )
    return pattern


class FileIO:
    def __init__(self, folder_dir, file):
        self.work_directory = folder_dir
        self.START_FILE = file
        self.NOEXT_START_FILE = os.path.splitext(os.path.basename(self.START_FILE))[0]
        self.is_save_gjf_file = True
        return
    
    def make_geometry_list(self, args_electric_charge_and_multiplicity):
        """Load initial structure"""
        tmp_geometry_list, element_list, electric_charge_and_multiplicity = xyz2list(self.START_FILE, args_electric_charge_and_multiplicity)
        natoms = len(tmp_geometry_list)
        
        # Create start_data with list comprehension instead of for loop
        start_data = [
            str(natoms),
            electric_charge_and_multiplicity,
            *[[element_list[j]] + tmp_geometry_list[j] for j in range(len(tmp_geometry_list))]
        ]
        
        return [start_data], element_list, electric_charge_and_multiplicity


    def print_geometry_list(self, new_geometry, element_list, electric_charge_and_multiplicity, display_flag=True):
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
    
        
        # Process all geometries at once with list comprehension
        formatted_geometries = []
        for num, geometry in enumerate(new_geometry):
            element = element_list[num]
            formatted_geometry = [element] + list(map(str, geometry))
            formatted_geometries.append(formatted_geometry)
        
        if display_flag:
            for num, geometry in enumerate(new_geometry):    
                element = element_list[num]
                print(f"{element:2}   {float(geometry[0]):>17.12f}   {float(geometry[1]):>17.12f}   {float(geometry[2]):>17.12f}")
            print("\n")
            
        geometry_list = [[electric_charge_and_multiplicity, *formatted_geometries]]
        
        return geometry_list
        
    
    def make_psi4_input_file(self, geometry_list, iter, path=None):#geometry_list: ang.
        """structure updated geometry is saved."""
        if path is not None:
            file_directory = os.path.join(path, f"samples_{self.NOEXT_START_FILE}_{iter}")
        else:
            file_directory = self.work_directory+"samples_"+self.NOEXT_START_FILE+"_"+str(iter)
        tmp_cs = ["SAMPLE"+str(iter), ""]


        float_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"

        os.makedirs(file_directory, exist_ok=True)
        
        for y, geometry in enumerate(geometry_list):
            tmp_geometry = []
            for geom in geometry:
                if len(geom) == 4 \
                  and re.match(r"[A-Za-z]+", str(geom[0])) \
                  and all(re.match(float_pattern, str(x)) for x in geom[1:]):
                        tmp_geometry.append(geom)

                if len(geom) == 2 and re.match(r"-*\d+", str(geom[0])) and re.match(r"-*\d+", str(geom[1])):
                    tmp_cs = geom   
                    
            with open(file_directory+"/"+self.NOEXT_START_FILE+"_"+str(y)+".xyz","w") as w:
                w.write(str(len(tmp_geometry))+"\n")
                w.write(str(tmp_cs[0])+" "+str(tmp_cs[1])+"\n")
                for rows in tmp_geometry:
                    w.write(f"{rows[0]:2}   {float(rows[1]):>17.12f}   {float(rows[2]):>17.12f}   {float(rows[3]):>17.12f}\n")
        return file_directory

    def read_gjf_file(self, args_electric_charge_and_multiplicity=None):        
        geometry_list = []
        element_list = []
        with open(self.START_FILE, 'r') as f:
            lines = f.read().splitlines()
        read_flag = False
        pattern = r"-*[0-9]+\s+-*[0-9]+\s*"
        repatter = re.compile(pattern)
        for i in range(len(lines)):
            if bool(re.match(repatter, lines[i])) is True:
                electric_charge_and_multiplicity = lines[i].split()
                read_flag = True
                geometry_list.append("dummy")
                geometry_list.append(lines[i].split())
                continue
            if read_flag and len(lines[i]) == 0:
                break
            if read_flag:
                element_list.append(lines[i].split()[0])
                geometry_list.append(lines[i].split())   
        geometry_list = [geometry_list]
        return geometry_list, element_list, electric_charge_and_multiplicity



    def read_mol_file(self, args_electric_charge_and_multiplicity=None):
        """
        Reads a .mol file, formats output to match the gjf reader structure.
        """
        # Call the internal parser to get clean lists
        coords_list, elements, charge_multiplicity = mol2list(
            self.START_FILE, args_electric_charge_and_multiplicity
        )

        if not elements:
            return [[]], [], charge_multiplicity

        # Reconstruct the geometry_list to match the target format
        # [["dummy", [charge, mult], [element, x, y, z], ...]]
        output_geometry_list = ["dummy", charge_multiplicity]
        for i, element in enumerate(elements):
            full_atom_line = [element] + coords_list[i]
            output_geometry_list.append(full_atom_line)

        return [output_geometry_list], elements, charge_multiplicity

    def read_mol2_file(self, args_electric_charge_and_multiplicity=None):
        """
        Reads a .mol2 file, formats output to match the gjf reader structure.
        """
        # Call the internal parser to get clean lists
        coords_list, elements, charge_multiplicity = mol22list(
            self.START_FILE, args_electric_charge_and_multiplicity
        )

        if not elements:
            return [[]], [], charge_multiplicity

        # Reconstruct the geometry_list to match the target format
        output_geometry_list = ["dummy", charge_multiplicity]
        for i, element in enumerate(elements):
            full_atom_line = [element] + coords_list[i]
            output_geometry_list.append(full_atom_line)

        return [output_geometry_list], elements, charge_multiplicity

    def read_gamess_inp_file(self, args_electric_charge_and_multiplicity=None):
        """
        Reads a .inp file, formats output to match the gjf reader structure.
        """
        # Call the internal parser to get clean lists
        coords_list, elements, charge_multiplicity = inp2list(
            self.START_FILE, args_electric_charge_and_multiplicity
        )

        if not elements:
            return [[]], [], charge_multiplicity

        # Reconstruct the geometry_list to match the target format
        output_geometry_list = ["dummy", charge_multiplicity]
        for i, element in enumerate(elements):
            full_atom_line = [element] + coords_list[i]
            output_geometry_list.append(full_atom_line)

        return [output_geometry_list], elements, charge_multiplicity



    def save_gjf_file(self, geometry_list):
        with open(self.work_directory+self.NOEXT_START_FILE+".gjf","w") as f:
            f.write("%mem=4GB\n")
            f.write("%nprocshared=4\n")
            f.write("#p B3LYP/6-31G* opt freq\n")
            f.write("\n")
            f.write("Title Card Required\n")
            f.write("\n")
            f.write("0 1\n")# This line is required for fixing the charge and multiplicity.
            for geometry in geometry_list:
                f.write(geometry)
                f.write("\n")
            f.write("\n\n\n")
        return 
    

    def make_traj_file(self, name=""):
        """optimized path is saved."""
        print("\nprocessing geometry collection ...\n")
        if name == "":
            file_list = sum([sorted(glob.glob(os.path.join(self.work_directory, f"samples_*_" + "[0-9]" * i, "*.xyz")))
                 for i in range(1, 7)], [])
        else:    
            file_list = sum([sorted(glob.glob(os.path.join(self.work_directory, f"samples_*_{name}_" + "[0-9]" * i, "*.xyz")))
                 for i in range(1, 7)], [])
        step_num = len(file_list)

        for m, file in enumerate(file_list):
            sample = []
            tmp_geometry_list, element_list, _ = xyz2list(file, None)
            for j in range(len(element_list)):
                sample.append(f"{element_list[j]:2}  {float(tmp_geometry_list[j][0]):>17.12f}   {float(tmp_geometry_list[j][1]):>17.12f}   {float(tmp_geometry_list[j][2]):>17.12f}")
            with open(self.work_directory+self.NOEXT_START_FILE+"_traj.xyz","a") as w:
                atom_num = len(sample)
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
                for i in sample:    
                    if "\n" == i or "" == i:
                        continue
                    w.write(i+"\n")
                
            if m == step_num - 1:
                if self.is_save_gjf_file:
                    self.save_gjf_file(sample)
                with open(self.work_directory+self.NOEXT_START_FILE+"_optimized.xyz","a") as w2:
                    w2.write(str(atom_num)+"\n")
                    w2.write("OptimizedStructure\n")
                    for i in sample:
                        if "\n" == i or "" == i:
                            continue
                        w2.write(i+"\n")
        print("\ngeometry collection was completed...\n")
        return

    def xyz_file_save_for_IRC(self, element_list, geometry_list):
        count = 0
        for geometry in geometry_list:
            with open(self.work_directory+"IRC_path.xyz","a") as w:
                atom_num = len(geometry)
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(count)+"\n")
                for i in range(len(geometry)):
                    w.write(f"{element_list[i]:2}  {float(geometry[i][0]):>17.12f}   {float(geometry[i][1]):>17.12f}   {float(geometry[i][2]):>17.12f}\n")
            count += 1
        print("\ngeometry collection for IRC was completed...\n")
        return

    def make_traj_file_for_DM(self, img_1="reactant", img_2="product"):
        """optimized path is saved."""
        print("\nprocessing geometry collection ...\n")
        file_list = sum(
            [sorted(glob.glob(self.work_directory + f"samples_*_{str(img_1)}_{'[0-9]' * i}/*.xyz"))
            for i in range(1, 7)],
            []
        ) + sum(
            [sorted(glob.glob(self.work_directory + f"samples_*_{str(img_2)}_{'[0-9]' * i}/*.xyz"))[::-1]
            for i in range(6, 0, -1)],
            []
        )        
        for m, file in enumerate(file_list[1:-1]):
            sample = []
            tmp_geometry_list, element_list, _ = xyz2list(file, None)
            for j in range(len(element_list)):
                sample.append(f"{element_list[j]:2}  {float(tmp_geometry_list[j][0]):>17.12f}   {float(tmp_geometry_list[j][1]):>17.12f}   {float(tmp_geometry_list[j][2]):>17.12f}")
                
            with open(self.work_directory+self.NOEXT_START_FILE+"_traj.xyz","a") as w:
                atom_num = len(sample)
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
                for i in sample:
                    if "\n" == i or "" == i:
                        continue
                    w.write(i+"\n")
        print("\ngeometry collection was completed...\n")
        return
    
    def argrelextrema_txt_save(self, save_list, name, min_max):
        NUM_LIST = [i for i in range(len(save_list))]
        if min_max == "max":
            local_max_energy_list_index = argrelextrema(np.array(save_list), np.greater)
            with open(self.work_directory+name+".txt","w") as f:
                for j in local_max_energy_list_index[0].tolist():
                    f.write(str(NUM_LIST[j])+"\n")
        elif min_max == "min":
            inverse_energy_list = (-1)*np.array(save_list, dtype="float64")
            local_min_energy_list_index = argrelextrema(np.array(inverse_energy_list), np.greater)
            with open(self.work_directory+name+".txt","w") as f:
                for j in local_min_energy_list_index[0].tolist():
                    f.write(str(NUM_LIST[j])+"\n")
        else:
            print("error")
    
        return



def write_xyz_file(element_list, coords, file_path, comment="save"):# element_list: list of element symbols, coords: np.array of coordinates (ang.)
    with open(file_path, 'w') as f:
        f.write(str(len(element_list)) + '\n')
        f.write(comment+'\n')
        for i in range(len(element_list)):
            f.write(element_list[i] + ' ' + ' '.join([str(j) for j in coords[i]]) + '\n')
    return

def make_workspace(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def stack_path(directory):
    file_list = glob.glob(directory + '/*_[0-9].xyz') + glob.glob(directory + '/*_[0-9][0-9].xyz') + glob.glob(directory + '/*_[0-9][0-9][0-9].xyz') + glob.glob(directory + '/*_[0-9][0-9][0-9][0-9].xyz') + glob.glob(directory + '/*_[0-9][0-9][0-9][0-9][0-9].xyz') + glob.glob(directory + '/*_[0-9][0-9][0-9][0-9][0-9][0-9].xyz') + glob.glob(directory + '/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].xyz') 
 
    with open(directory + '/path.xyz', 'w') as f:
        for file in file_list:
            with open(file, 'r') as g:
                lines = g.read().splitlines()
            for line in lines:
                f.write(line + '\n')
    return