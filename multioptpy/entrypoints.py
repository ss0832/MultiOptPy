import sys
import os
import shutil
import datetime
import json
import argparse
import random
import numpy as np
import itertools

# --- 1. Core Imports (Relying on pip installation) ---
import multioptpy
from multioptpy.Parameters.unit_values import UnitValueLib
from multioptpy.Utils import calc_tools
from multioptpy.Utils.bond_connectivity import BondConnectivity
from multioptpy.Wrapper.autots import AutoTSWorkflow
from multioptpy.Wrapper.autots import AutoTSWorkflow_v2

# --- 2. Entry Point Functions (Matching pyproject.toml) ---

def run_optmain():
    """ Entry point for the main geometry optimization script (optmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.optimizeparser(parser)    
    bpa = multioptpy.optimization.Optimize(args)
    bpa.run()


def run_ieipmain():
    """ Entry point for the iEIP calculation script (ieipmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.ieipparser(parser)
    iEIP = multioptpy.ieip.iEIP(args)
    iEIP.run()



def run_mdmain():
    """ Entry point for the molecular dynamics script (mdmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.mdparser(parser)
    MD = multioptpy.moleculardynamics.MD(args)
    MD.run()



def run_nebmain():
    """ Entry point for the Nudged Elastic Band (NEB) calculation script (nebmain.py). """
    parser = multioptpy.interface.init_parser()
    args = multioptpy.interface.nebparser(parser)
    NEB = multioptpy.neb.NEB(args)
    NEB.run()



def run_relaxedscan():
    """ Entry point for the relaxed scan script (relaxed_scan.py). """
    # When you use psi4 with this script, segmentation fault may occur. Thus, I recommend to use pyscf, tblite and so on with the following script.

    def relaxed_scan_perser(parser):
        parser.add_argument("-nsample", "--number_of_samples", type=int, default=10, help='the number of sampling relaxed scan coordinates')
        parser.add_argument("-scan", "--scan_tgt", nargs="*", type=str, help='scan target (ex.) [[bond, angle, or dihedral etc.] [atom_num] [(value_1(ex. 1.0 ang.)),(value_2(ex. 1.5 ang.))] ...] ', default=None)
        parser.add_argument("-fo", "--first_only", action="store_true", help='use only input structure for relax scan ')
        return parser


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


    def save_xyz_file(coord_list, element_list, file_name, additional_name, directory):
        no_ext_file_name = os.path.splitext(file_name)[0]
        sample_file_name = no_ext_file_name+"_RelaxedScan_"+str(additional_name)+".xyz"
        with open(directory + sample_file_name, 'w') as f:
            f.write(str(len(coord_list))+"\n")
            f.write("RelaxedScan_"+str(additional_name)+"\n")
            for i in range(len(coord_list)):
                f.write(element_list[i]+" "+str(coord_list[i][0])+" "+str(coord_list[i][1])+" "+str(coord_list[i][2])+"\n")
        
        return sample_file_name

    def make_scan_tgt(scan_list):
        scan_tgt_list = []
        atom_num_list = []
        scan_range_list = []
        for i in range(int(len(scan_list)/3)):
            scan = scan_list[i*3:i*3+3]
            scan_type = str(scan[0])
            atom_num_list.append(scan[1])
            tmp_scan_tgt = scan[2].split(",")
            scan_tgt_list.append(scan_type)
            scan_range_list.append([float(tmp_scan_tgt[0]), float(tmp_scan_tgt[1])])
        return scan_tgt_list, atom_num_list, scan_range_list
       


    parser = multioptpy.interface.init_parser()
    parser = relaxed_scan_perser(parser)
    args = multioptpy.interface.optimizeparser(parser)
    first_only_flag = args.first_only
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    directory = os.path.splitext(args.INPUT)[0]+"_RScan_"+date+"/"
    os.makedirs(directory, exist_ok=True)   
    ene_profile_filepath = directory + os.path.splitext(args.INPUT)[0]+"_scan_energy_profile_"+date+".csv"
    
    scan_part = int(args.number_of_samples)
    scan_tgt_list, atom_num_list, scan_range_list = make_scan_tgt(args.scan_tgt)
    
    with open(args.INPUT, 'r') as f:
        words = f.read().splitlines()
    
    
    print("Start relaxed scan .... ")
    print("Scan target: ", scan_tgt_list, atom_num_list)
    print("Scan range: ", scan_range_list)
    scan_job_list = np.array([np.linspace(start, end, scan_part) for start, end in scan_range_list]).T
    
    
    original_input_file = args.INPUT
    original_args = args

    with open(ene_profile_filepath, "w") as f:
        f.write(str(",".join(scan_tgt_list))+",energy,bias_energy\n")
        tmp_atom_num_list = [str(i).replace(",", "_") for i in atom_num_list]
        f.write(str(",".join(tmp_atom_num_list))+",(hartree),(hartree)\n")


    shutil.copyfile(original_input_file, directory + os.path.splitext(original_input_file)[0]+"_RelaxedScan_0.xyz")
    shutil.copyfile(original_input_file, os.path.splitext(original_input_file)[0]+"_RelaxedScan_0.xyz")
    init_input_file = os.path.splitext(original_input_file)[0]+"_RelaxedScan_0.xyz"
    args.INPUT = init_input_file

    with open(directory + "input.txt", 'w') as f:
        f.write(str(vars(args)))

    for i in range(scan_part):
        print("scan type(s)  : ", scan_tgt_list)
        print("scan atom(s)  : ", atom_num_list)
        print("scan value(s) : ", scan_job_list[i])
        args.projection_constrain = []
        for scan_tgt, atom_num, scan_value in zip(scan_tgt_list, atom_num_list, scan_job_list[i]):
            args.projection_constrain.extend(["manual", scan_tgt, atom_num, str(scan_value)])
            
        bpa = multioptpy.optimization.Optimize(args)
        bpa.run()
        
        opted_geometry = bpa.final_geometry * UnitValueLib().bohr2angstroms # get optimized geometry
        element_list = bpa.element_list
        energy = bpa.final_energy
        bias_energy = bpa.final_bias_energy
        
        next_input_file_for_save = directory + save_xyz_file(opted_geometry, element_list, original_input_file, i+1, directory)
        next_input_file = save_xyz_file(opted_geometry, element_list, original_input_file, i+1, "")
        args = original_args
        if first_only_flag:
            args.INPUT = init_input_file
        else:
            args.INPUT = next_input_file
        print("Done.")
        
        with open(ene_profile_filepath, "a") as f:
            f.write(str(",".join(list(map(str, scan_job_list[i].tolist())))) + "," + str(energy) + "," + str(bias_energy) + "\n")
        del bpa
    
    print("Relaxed scan done ...")


def run_orientsearch():
    """ Entry point for the orientation search script (orientation_search.py). """
        
    def orientation_search(parser):
        parser.add_argument("-nsample", "--number_of_samples", type=int, default=5, help='the number of sampling orientations')
        parser.add_argument("-part", "--part", nargs="*", type=str, help='the part number (ex.) 1,2,3 or 1-3', default=None)
        parser.add_argument("-dist", "--distance",  type=float, default=5.0, help='the distance of parts [ang.]')
        return parser
    
    
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
    
    
    def save_xyz_file(coord_list, element_list, file_name, n_sample):
        no_ext_file_name = os.path.splitext(file_name)[0]
        sample_file_name = no_ext_file_name+"_Sample_"+str(n_sample)+".xyz"
        with open(sample_file_name, 'w') as f:
            f.write(str(len(coord_list))+"\n")
            f.write("Sample_"+str(n_sample)+"\n")
            for i in range(len(coord_list)):
                f.write(element_list[i]+" "+str(coord_list[i][0])+" "+str(coord_list[i][1])+" "+str(coord_list[i][2])+"\n")
        
        return sample_file_name
    
    
    def make_random_orientation_xyz_file(coord_list, part_list, part_dist, input_file_name, n_sample, element_list):#ang.
        part_coord_list = []
        part_element_list = []
        for part in part_list:
            tmp_part_coord_list = []
            tmp_part_element_list = []
            for p in part:
                tmp_part_coord_list.append(coord_list[p-1])
                tmp_part_element_list.append(element_list[p-1])
            part_coord_list.append(np.array(tmp_part_coord_list))
            part_element_list.append(tmp_part_element_list)
    
        # Random rotation
        rand_rotated_part_coord_list = []
        for i in range(len(part_coord_list)):
            part_center = calc_tools.Calculationtools().calc_center(part_coord_list[i], part_element_list[i])
            centered_part_coord_list = part_coord_list[i] - part_center 
            random_x_angle = random.uniform(0, 2*np.pi)
            random_y_angle = random.uniform(0, 2*np.pi)
            random_z_angle = random.uniform(0, 2*np.pi)
            print("Random angles (Radian): ", random_x_angle, random_y_angle, random_z_angle)
            rotated_part_coord = calc_tools.rotate_molecule(centered_part_coord_list, "x", random_x_angle)
            rotated_part_coord = calc_tools.rotate_molecule(rotated_part_coord, "y", random_y_angle)
            rotated_part_coord = calc_tools.rotate_molecule(rotated_part_coord, "z", random_z_angle)
            rand_rotated_part_coord_list.append(rotated_part_coord)
    
    
        # Random translation
        rand_tr_rot_part_coord_list = rand_rotated_part_coord_list
    
        if len(rand_tr_rot_part_coord_list) == 2:
            rand_dist = random.uniform(part_dist, 1.1*part_dist)
            rand_tr_rot_part_coord_list[0] = rand_tr_rot_part_coord_list[0] - np.array([part_dist/2, 0, 0])
            rand_tr_rot_part_coord_list[1] = rand_tr_rot_part_coord_list[1] + np.array([part_dist/2, 0, 0])
        else:
            fragm_num = len(rand_tr_rot_part_coord_list)
            rand_dist = random.uniform(part_dist, 1.1*part_dist)
            rand_sample_list = random.sample(range(fragm_num), fragm_num)
            distance_polygen_point_from_center = 0.5 * rand_dist * np.tan((fragm_num - 2) * np.pi / fragm_num)
            tmp_angle = 0.0
            for i in range(len(rand_sample_list)):
                rand_tr_rot_part_coord_list[rand_sample_list[i]] = rand_tr_rot_part_coord_list[rand_sample_list[i]] + np.array([distance_polygen_point_from_center * np.cos(tmp_angle), distance_polygen_point_from_center * np.sin(tmp_angle), 0])
                tmp_angle += 2 * np.pi / fragm_num
        
        combined_coord_list = np.zeros((len(coord_list), 3))       
    
        # Combine all parts
        for i in range(len(part_list)):
            part = part_list[i]
        
            for j in range(len(part)):
                
                combined_coord_list[part[j]-1] = rand_tr_rot_part_coord_list[i][j]
    
        input_file_name = save_xyz_file(combined_coord_list, element_list, input_file_name, n_sample)
        
        return input_file_name




    parser = multioptpy.interface.init_parser()
    parser = orientation_search(parser)
    args = multioptpy.interface.optimizeparser(parser)
    
    part_dist = args.distance
    
    with open(args.INPUT, 'r') as f:
        words = f.read().splitlines()
    
    element_list = []
    coord_list = []
    
    for word in words:
        splitted_word = word.split()

        if len(splitted_word) > 3:
            element_list.append(str(splitted_word[0]))
            coord_list.append([float(splitted_word[1]), float(splitted_word[2]), float(splitted_word[3])])
    
    coord_list = np.array(coord_list, dtype="float64")
    atom_num = len(coord_list)

    if args.part is None:
        print("No part is specified. exit....")
        sys.exit(0)
    
    part_list = []
    for i in range(len(args.part)):
        part_list.append(num_parse(args.part[i]))
    
    flattened_part_list = [item for sublist in part_list for item in sublist]
    missing_parts = [num for num in range(1, atom_num+1) if num not in flattened_part_list]
    if len(missing_parts) > 0:
        part_list.append(missing_parts)

    print("Number of Parts: ", len(part_list)) 
    n_sample = args.number_of_samples
    
    original_input_file = args.INPUT
    
    for i in range(n_sample):
        print("Sampling orientation: ", i)    
        args.INPUT = make_random_orientation_xyz_file(coord_list, part_list, part_dist, original_input_file, i, element_list)
        bpa = multioptpy.optimization.Optimize(args)
        bpa.run()
    
        
        print("Sampling orientation: ", i, "done")
    
    print("Orientation search done")


def run_autots():
    """ Entry point for the Automated Transition State (AutoTS) workflow (run_autots.py). """
    # ======================================================================
    # AUTO-TS WORKFLOW (V1/V2) CONFIGURATION GUIDE
    # ======================================================================
    # Config is loaded from 'config.json'.
    #
    # V1 (Legacy): Uses top-level keys like 'step1_settings', 'skip_step1'.
    # V2 (Dynamic): Uses the "workflow": [...] block to define execution.
    #
    # CRITICAL GUIDELINE:
    # To understand options (e.g., 'opt_method', 'NSTEP'),
    # refer to 'multioptpy/interface.py':
    # 
    # 1. Step 1, 3, & 4 Settings: call_optimizeparser()
    # 2. Step 2 Settings: call_nebparser()
    # ======================================================================
    
    
    def load_config_from_file(config_path):
        """Loads configuration settings from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"Successfully loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {config_path}. Check file format.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred loading config: {e}")
            sys.exit(1)
    
    
    def launch_workflow(config):
        """
        Launches the AutoTSWorkflow (v1 or v2) based on config.
        
        Args:
            config (dict): The complete configuration dictionary.
        """
        
        # --- 1. Apply settings (for v1 compatibility) ---
        # This is still useful for v2 if 'stepX_settings' are used as base keys.
        local_conf_name = os.path.basename(config.get("software_path_file_source", "software_path.conf"))
        for i in range(1, 5): # Steps 1, 2, 3, and 4
            step_key = f"step{i}_settings"
            if step_key in config:
                config[step_key]["software_path_file"] = local_conf_name
            elif i == 4 and "step4_settings" not in config: # Create step4 settings if not in config
                config["step4_settings"] = {"software_path_file": local_conf_name}
    
        # --- 2. Print Summary ---
        print("--- AutoTS Workflow Starting ---")
        print(f"Input File: {config.get('initial_mol_file')}")
        
        # --- 3. NEW: Dynamic v1/v2 Selection ---
        if "workflow" in config:
            print(">>> Detected 'workflow' key. Initializing AutoTSWorkflow_v2.")
            workflow = AutoTSWorkflow_v2(config=config)
        else:
            print(">>> No 'workflow' key found. Initializing AutoTSWorkflow (v1).")
            # Print v1-specific flags
            print(f"Skip Step 1: {config.get('skip_step1', False)}")
            print(f"Skip to Step 4: {config.get('skip_to_step4', False)}")
            print(f"Run Step 4 (IRC): {config.get('run_step4', False)}")
            
            if not config.get('skip_step1', False) and not config.get('skip_to_step4', False):
                print(f"AFIR Params: {config.get('step1_settings', {}).get('manual_AFIR')}")
            
            workflow = AutoTSWorkflow(config=config)
    
        # --- 4. Run the selected workflow ---
        # Both v1 and v2 classes have a .run_workflow() method
        workflow.run_workflow()
    
    def main():
        """
        Main function for command-line execution.
        Parses CMD arguments, loads config, merges them, and calls launch_workflow.
        (Modified for v1/v2 compatibility)
        """
        parser = argparse.ArgumentParser(
            description="Run the Automated Transition State (AutoTS) workflow (v1 or v2)."
        )
        
        # --- (Parser arguments remain the same) ---
        parser.add_argument(
            "input_file",
            type=str,
            help="Path to the initial structure file. If --skip_to_step4 is used (v1), this must be the TS file."
        )
        parser.add_argument(
            "-cfg", "--config_file",
            type=str,
            default="./config.json",
            help="Path to the configuration JSON file. Default is './config.json'."
        )
        parser.add_argument(
            "-ma", "--manual_AFIR",
            nargs="*",
            required=False, 
            help="Manual AFIR parameters for Step 1. Overrides config file's 'step1_settings'."
        )
        parser.add_argument(
            "-osp", "--software_path_file",
            type=str,
            default="./software_path.conf",
            help="Path to the 'software_path.conf' file. Defaults to './software_path.conf'"
        )
        parser.add_argument(
            "-n", "--top_n",
            type=int,
            default=None, # Default will be read from JSON
            help="Refine the top N highest energy candidates from NEB. Overrides config file."
        )
        
        # --- V1-specific flags ---
        parser.add_argument(
            "--skip_step1",
            action="store_true",
            help="Skip the AFIR scan (Step 1). The input_file must be the NEB trajectory file."
        )
        parser.add_argument(
            "--run_step4",
            action="store_true",
            help="Run Step 4 (IRC + Endpoint Optimization) after Step 3 completes."
        )
        parser.add_argument(
            "--skip_to_step4",
            action="store_true",
            help="Skip Steps 1-3 and run only Step 4. The 'input_file' must be the TS structure file."
        )
        
        args = parser.parse_args()
    
        # --- 1. Load Base Configuration from File ---
        workflow_config = load_config_from_file(args.config_file)
        
        # --- 2. Override Config with CMD Arguments ---
        # These apply to both v1 and v2
        workflow_config["initial_mol_file"] = args.input_file
        workflow_config["software_path_file_source"] = os.path.abspath(args.software_path_file)
        
        # Merge V1-specific CMD flags (v2 will ignore them)
        workflow_config["skip_step1"] = args.skip_step1
        workflow_config["run_step4"] = args.run_step4
        workflow_config["skip_to_step4"] = args.skip_to_step4
    
        if args.top_n is not None:
            workflow_config["top_n_candidates"] = args.top_n
    
        # --- 3. AFIR Validation (v1/v2 compatibility logic) ---
        
        # Ensure 'step1_settings' key exists for v1 compatibility
        workflow_config.setdefault("step1_settings", {})
        
        # Check if v1 is running step 1
        is_v1_running_step1 = (
            "workflow" not in workflow_config and 
            not args.skip_step1 and 
            not args.skip_to_step4
        )
        
        # Check if v2 is running step 1
        is_v2_running_step1 = False
        if "workflow" in workflow_config:
            for entry in workflow_config.get("workflow", []):
                if entry.get("step") == "step1" and entry.get("enabled", True):
                    is_v2_running_step1 = True
                    break
    
        # Check AFIR status in config (base 'step1_settings' only) and CMD
        config_has_afir = workflow_config["step1_settings"].get("manual_AFIR")
        cmd_has_afir = args.manual_AFIR is not None
    
        if cmd_has_afir:
            # Case 1: CMD argument is given. It *always* overrides the base 'step1_settings'.
            workflow_config["step1_settings"]["manual_AFIR"] = args.manual_AFIR
            print(f"Using 'manual_AFIR' from command line (overrides 'step1_settings'): {args.manual_AFIR}")
            # This will be used by v1, or by v2 if 'step1_settings' is its base key.
        
        elif not config_has_afir:
            # Case 2: No AFIR in CMD, and *also* no AFIR in the base 'step1_settings'.
            
            if is_v1_running_step1:
                # For v1, this is a fatal error.
                print("\nError (v1 mode): 'manual_AFIR' is not defined in 'step1_settings' and was not provided via -ma.")
                print("       Please add 'manual_AFIR' to your JSON or use the -ma argument.")
                sys.exit(1)
                
            elif is_v2_running_step1:
                # For v2, this is a warning, as it might be defined in 'param_override'.
                print(f"Warning: 'manual_AFIR' not found in 'step1_settings' or via -ma.")
                print("       (v2 mode): Ensure 'manual_AFIR' is defined in your 'workflow' entry")
                print("       (either in the base 'settings_key' block or 'param_override').")
                
        elif is_v1_running_step1 and config_has_afir:
            # Case 3: v1 is running, no CMD override, but config has it. Just print confirmation.
            print(f"Using 'manual_AFIR' from config file: {config_has_afir}")
    
        # --- 4. Call the launcher function ---
        launch_workflow(workflow_config)
    
    
    main()


def run_confsearch():
    """ Entry point for the conformation search script (conformation_search.py). """
    
    # Unit conversion constants
    bohr2ang = 0.529177210903
    ang2bohr = 1.0 / bohr2ang
    
    #Example: python conformation_search.py s8_for_confomation_search_test.xyz -xtb GFN2-xTB -ns 2000
    
    def calc_boltzmann_distribution(energy_list, temperature=298.15):
        """
        Calculate the Boltzmann distribution.
        """
        energy_list = np.array(energy_list)
        energy_list = energy_list - min(energy_list)
        energy_list = energy_list * 627.509
        boltzmann_distribution = np.exp(-energy_list / (0.0019872041 * temperature))
        boltzmann_distribution = boltzmann_distribution / np.sum(boltzmann_distribution)
        
        return boltzmann_distribution
    
    def calc_penalized_boltzmann_distribution(energy_list, visit_counts, temperature=298.15, alpha=0.5):
        """
        Calculate the Boltzmann distribution with frequency penalty (Tabu Search / Metadynamics approach).
        
        P_i^{select} ∝ P_i^{Boltzmann} × exp(-α × N_i)
        
        where:
        - P_i^{Boltzmann}: Standard Boltzmann probability
        - N_i: Number of times conformer i has been selected as starting point
        - α: Penalty coefficient (larger values = stronger penalty)
        
        This prevents the search from staying in a specific basin and encourages
        broader exploration of conformational space.
        """
        # Calculate standard Boltzmann distribution
        energy_list = np.array(energy_list)
        energy_list = energy_list - min(energy_list)
        energy_list = energy_list * 627.509  # Convert to kcal/mol
        boltzmann_weights = np.exp(-energy_list / (0.0019872041 * temperature))
        
        # Apply frequency penalty: exp(-α × N_i)
        visit_counts = np.array(visit_counts)
        frequency_penalty = np.exp(-alpha * visit_counts)
        
        # Combined probability
        penalized_weights = boltzmann_weights * frequency_penalty
        
        # Normalize
        penalized_distribution = penalized_weights / np.sum(penalized_weights)
        
        return penalized_distribution
    
    def get_index_from_distribution(probabilities):
        if not abs(sum(probabilities) - 1.0) < 1e-8:
            raise ValueError("the sum of probabilities is not 1.0")
        
        cumulative_distribution = []
        cumulative_sum = 0
        for p in probabilities:
            cumulative_sum += p
            cumulative_distribution.append(cumulative_sum)
        
        rand = random.random()
        
        for i, threshold in enumerate(cumulative_distribution):
            if rand < threshold:
                return i
    
    
    def calc_distance_matrix(geom_num_list):
        natoms = len(geom_num_list)
        combination_natoms = int(natoms * (natoms - 1) / 2)
        distance_matrix = np.zeros((combination_natoms))
        
        count = 0
        for i, j in itertools.combinations(range(natoms), 2):
            distance_matrix[count] = np.linalg.norm(geom_num_list[i] - geom_num_list[j])
            count += 1    
        
        return distance_matrix
    
    def sort_distance_matrix(distance_matrix):
        sorted_dist_matrix = np.sort(distance_matrix)
        return sorted_dist_matrix
    
    def check_identical(geom_num_list_1, geom_num_list_2, threshold=1e-3):
        distance_matrix_1 = calc_distance_matrix(geom_num_list_1)
        distance_matrix_2 = calc_distance_matrix(geom_num_list_2)
        sort_distance_matrix_1 = sort_distance_matrix(distance_matrix_1)
        sort_distance_matrix_2 = sort_distance_matrix(distance_matrix_2)
    
        # Check if the two geometries are identical
        if np.all(np.abs(sort_distance_matrix_1 - sort_distance_matrix_2) < threshold):
            print("The two geometries are identical.")
            return True
        else:
            print("The two geometries are not identical.")
            return False
        
    def read_xyz(file_name):
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
        
        geom_num_list = []
        element_list = []
        
        for i in range(2, len(data)):
            splitted_data = data[i].split()
            element_list.append(splitted_data[0])
            geom_num_list.append(splitted_data[1:4])
        
        geom_num_list = np.array(geom_num_list, dtype="float64")
        
        return geom_num_list, element_list
    
    def get_bond_connectivity_table(geom_num_list, element_list):
        """
        Calculate bond connectivity table from geometry and element list.
        The geometry should be in Angstrom units.
        Returns a sorted list of bond pairs for comparison.
        """
        BC = BondConnectivity()
        # Convert Angstrom to Bohr for BondConnectivity class
        geom_bohr = geom_num_list * ang2bohr
        bond_connect_mat = BC.bond_connect_matrix(element_list, geom_bohr)
        bond_table = BC.bond_connect_table(bond_connect_mat)
        # Sort the bond table for consistent comparison
        sorted_bond_table = sorted([tuple(sorted(bond)) for bond in bond_table])
        return sorted_bond_table
    
    def check_bond_connectivity_preserved(init_bond_table, conformer_bond_table):
        """
        Check if the bond connectivity is preserved between initial structure and conformer.
        Returns a tuple: (is_preserved, added_bonds, removed_bonds)
        """
        if init_bond_table == conformer_bond_table:
            print("Bond connectivity is preserved.")
            return True, [], []
        else:
            # Find added and removed bonds
            init_set = set(init_bond_table)
            conformer_set = set(conformer_bond_table)
            
            added_bonds = list(conformer_set - init_set)
            removed_bonds = list(init_set - conformer_set)
            
            if added_bonds:
                print(f"New bonds formed: {added_bonds}")
            if removed_bonds:
                print(f"Bonds broken: {removed_bonds}")
            
            print("Bond connectivity is NOT preserved. This conformer will be excluded.")
            return False, added_bonds, removed_bonds
    
    def save_rejected_conformer(conformer, element_list, energy, trial_num, 
                                rejected_folder, init_INPUT, added_bonds, removed_bonds,
                                rejected_count):
        """
        Save a conformer that was rejected due to bond connectivity change.
        """
        no_ext_init_INPUT = os.path.splitext(os.path.basename(init_INPUT))[0]
        
        # Save xyz file
        rejected_file_name = f"{rejected_folder}/{no_ext_init_INPUT}_REJECTED{rejected_count}.xyz"
        with open(rejected_file_name, 'w') as f:
            f.write(str(len(conformer))+"\n")
            f.write(f"Rejected_Trial{trial_num}_Energy{energy:.8f}\n")
            for i in range(len(conformer)):
                f.write(f"{element_list[i]} {conformer[i][0]} {conformer[i][1]} {conformer[i][2]}\n")
        
        # Save bond change information
        info_file_name = f"{rejected_folder}/{no_ext_init_INPUT}_REJECTED{rejected_count}_info.txt"
        with open(info_file_name, 'w') as f:
            f.write(f"Rejected Conformer Information\n")
            f.write(f"="*50 + "\n")
            f.write(f"Trial number: {trial_num}\n")
            f.write(f"Energy: {energy}\n")
            f.write(f"XYZ file: {no_ext_init_INPUT}_REJECTED{rejected_count}.xyz\n")
            f.write(f"\nBond Connectivity Changes:\n")
            if added_bonds:
                f.write(f"  New bonds formed:\n")
                for bond in added_bonds:
                    f.write(f"    {bond[0]+1} - {bond[1]+1}\n")
            if removed_bonds:
                f.write(f"  Bonds broken:\n")
                for bond in removed_bonds:
                    f.write(f"    {bond[0]+1} - {bond[1]+1}\n")
        
        return rejected_file_name
    
    def conformation_search(parser):
        parser.add_argument("-bf", "--base_force", type=float, default=100.0, help='bias force to search conformations (default: 100.0 kJ)')
        parser.add_argument("-ms", "--max_samples", type=int, default=50, help='the number of trial of calculation (default: 50)')
        parser.add_argument("-nl", "--number_of_lowest",  type=int, default=5, help='termination condition of calculation for updating list (default: 5)')
        parser.add_argument("-nr", "--number_of_rank",  type=int, default=10, help='termination condition of calculation for making list (default: 10)')
        parser.add_argument("-tgta", "--target_atoms", nargs="*", type=str, help='the atom to add bias force to perform conformational search (ex.) 1,2,3 or 1-3', default=None)
        parser.add_argument("-st", "--sampling_temperature", type=float, help='set temperature to select conformer using Boltzmann distribution (default) 298.15 (K)', default=298.15)
        parser.add_argument("-nost", "--no_stochastic", action="store_true", help='no switching EQ structure during conformation sampling')
        parser.add_argument("-pbc", "--preserve_bond_connectivity", action="store_true", help='exclude conformers with different bond connectivity from the initial optimized structure')
        parser.add_argument("-tabu", "--tabu_search", action="store_true", help='enable Tabu Search / Metadynamics-like frequency penalty to avoid revisiting the same conformers')
        parser.add_argument("-alpha", "--tabu_alpha", type=float, default=0.5, help='penalty coefficient for Tabu Search (default: 0.5). Larger values = stronger penalty for frequently visited conformers')
        return parser
    
    def return_pair_idx(i, j):
        ii = max(i, j) + 1
        jj = min(i, j) + 1
        pair_idx = int(ii * (ii - 1) / 2 - (ii - jj)) - 1
        return pair_idx
    
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
    
    
    def save_xyz_file(coord_list, element_list, file_name, add_name):
        no_ext_file_name = os.path.splitext(file_name)[0]
        sample_file_name = no_ext_file_name+"_"+str(add_name)+".xyz"
        with open(sample_file_name, 'w') as f:
            f.write(str(len(coord_list))+"\n")
            f.write("Sample_"+str(add_name)+"\n")
            for i in range(len(coord_list)):
                f.write(element_list[i]+" "+str(coord_list[i][0])+" "+str(coord_list[i][1])+" "+str(coord_list[i][2])+"\n")
        
        return sample_file_name
    
    
    def read_energy_file(file_name):
        """
        Read energy file and return energy list.
        Format:
        -3.000000
        -1.000000
        -2.000000
        ....
        """
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
        
        energy_list = []
        
        for i in range(len(data)):
            splitted_data = data[i].split()
            if len(splitted_data) == 0:
                continue
            energy_list.append(float(splitted_data[0]))
        
        return energy_list
    
    def read_visit_counts_file(file_name):
        """
        Read visit counts file and return visit counts list.
        Format:
        0
        2
        1
        ....
        """
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
        
        visit_counts = []
        
        for i in range(len(data)):
            splitted_data = data[i].split()
            if len(splitted_data) == 0:
                continue
            visit_counts.append(int(splitted_data[0]))
        
        return visit_counts
    
    def read_rejected_count_file(file_name):
        """
        Read rejected count file and return the count.
        """
        if not os.path.exists(file_name):
            return 0
        with open(file_name, 'r') as f:
            data = f.read().strip()
        if data:
            return int(data)
        return 0
    
    def read_bond_table_file(file_name):
        """
        Read bond table from file.
        Format:
        1 2
        1 3
        2 4
        ...
        """
        if not os.path.exists(file_name):
            return None
        
        bond_table = []
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        # Convert from 1-indexed (file) to 0-indexed (internal)
                        bond_table.append((int(parts[0])-1, int(parts[1])-1))
        
        return sorted(bond_table)
    
    def save_bond_table_file(bond_table, file_name):
        """
        Save bond table to file.
        """
        with open(file_name, 'w') as f:
            f.write("# Reference Bond Connectivity Table (1-indexed)\n")
            f.write("# Obtained from initial optimized structure (EQ0)\n")
            for bond in bond_table:
                # Convert from 0-indexed (internal) to 1-indexed (file)
                f.write(f"{bond[0]+1} {bond[1]+1}\n")
    
    def make_tgt_atom_pair(geom_num_list, element_list, target_atoms):
        norm_dist_min = 1.0
        norm_dist_max = 8.0
        norm_distance_list = calc_tools.calc_normalized_distance_list(geom_num_list, element_list)
        bool_tgt_atom_list = np.where((norm_dist_min < norm_distance_list) & (norm_distance_list < norm_dist_max), True, False)
        updated_target_atom_pairs = []
        for i, j in itertools.combinations(target_atoms, 2):
            
            pair_idx = return_pair_idx(i, j)
            if bool_tgt_atom_list[pair_idx]:
                updated_target_atom_pairs.append([[i, j], "p"])
                updated_target_atom_pairs.append([[i, j], "m"])
        
        return updated_target_atom_pairs
    
    
    def is_identical(conformer, energy, energy_list, folder_name, init_INPUT, ene_threshold=1e-4, dist_threshold=1e-1):
        no_ext_init_INPUT = os.path.splitext(init_INPUT)[0]
        ene_identical_list = []
        
        for i in range(len(energy_list)):
            if abs(energy_list[i] - energy) < ene_threshold:
                print("Energy is identical. Check distance matrix.")
                ene_identical_list.append(i)
            
        if len(ene_identical_list) == 0:
            print("Energy is not identical. Register this conformer.")
            return False
    
        for i in range(len(energy_list)):
            conformer_file_name = folder_name+"/"+no_ext_init_INPUT+"_EQ"+str(i)+".xyz"
            conformer_geom_num_list, conformer_element_list = read_xyz(conformer_file_name)
            
            bool_identical = check_identical(conformer, conformer_geom_num_list, threshold=dist_threshold)
            
            if bool_identical:
                print("This conformer is identical to the existing conformer. Skip this conformer.")
                return True
            
        
        print("This conformer is not identical to the existing conformer. Register this conformer.")
        return False
    
    def switch_conformer(energy_list, temperature=298.15):
        boltzmann_distribution = calc_boltzmann_distribution(energy_list, temperature)
        idx = get_index_from_distribution(boltzmann_distribution)
        return idx
    
    def switch_conformer_with_tabu(energy_list, visit_counts, temperature=298.15, alpha=0.5):
        """
        Select conformer using Boltzmann distribution with frequency penalty.
        This implements a Tabu Search / Metadynamics-like approach.
        """
        penalized_distribution = calc_penalized_boltzmann_distribution(
            energy_list, visit_counts, temperature, alpha
        )
        idx = get_index_from_distribution(penalized_distribution)
        return idx, penalized_distribution

    # ========== Main execution starts here ==========
    
    parser = multioptpy.interface.init_parser()
    parser = conformation_search(parser)
    
    args = multioptpy.interface.optimizeparser(parser)
    no_stochastic = args.no_stochastic
    preserve_bond_connectivity = args.preserve_bond_connectivity
    use_tabu_search = args.tabu_search
    tabu_alpha = args.tabu_alpha
    init_geom_num_list, init_element_list = read_xyz(args.INPUT)
    sampling_temperature = args.sampling_temperature
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.splitext(args.INPUT)[0]+"_"+str(int(args.base_force))+"KJ_CS_"+date_str
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Create folder for rejected conformers (bond connectivity changed)
    rejected_folder = folder_name + "/REJECTED_CONFORMERS"
    if preserve_bond_connectivity:
        if not os.path.exists(rejected_folder):
            os.makedirs(rejected_folder)
    
    # Log Tabu Search settings
    if use_tabu_search:
        print(f"Tabu Search / Metadynamics-like frequency penalty is enabled (alpha = {tabu_alpha}).")
        with open(folder_name+"/tabu_search_settings.log", "w") as f:
            f.write("Tabu Search / Metadynamics-like Frequency Penalty Settings\n")
            f.write("="*60 + "\n")
            f.write(f"Alpha (penalty coefficient): {tabu_alpha}\n")
            f.write(f"Formula: P_i^select ∝ P_i^Boltzmann × exp(-α × N_i)\n")
            f.write(f"  where N_i is the number of times conformer i has been selected.\n")
    
    energy_list_file_path = folder_name+"/EQ_energy.dat"
    visit_counts_file_path = folder_name+"/visit_counts.dat"
    rejected_count_file_path = rejected_folder+"/rejected_count.dat"
    rejected_energy_file_path = rejected_folder+"/rejected_energy.dat"
    reference_bond_table_file_path = folder_name+"/reference_bond_table.dat"
    
    if os.path.exists(energy_list_file_path):
        energy_list = read_energy_file(energy_list_file_path)
    else:
        energy_list = []
    
    # Initialize or load visit counts for Tabu Search
    if use_tabu_search:
        if os.path.exists(visit_counts_file_path):
            visit_counts = read_visit_counts_file(visit_counts_file_path)
            # Extend visit_counts if energy_list is longer (new conformers added)
            while len(visit_counts) < len(energy_list):
                visit_counts.append(0)
        else:
            visit_counts = [0] * len(energy_list)
    else:
        visit_counts = None
    
    # Initialize or load rejected conformer count
    if preserve_bond_connectivity:
        bond_reject_count = read_rejected_count_file(rejected_count_file_path)
        # Load rejected energies if exists
        if os.path.exists(rejected_energy_file_path):
            rejected_energy_list = read_energy_file(rejected_energy_file_path)
        else:
            rejected_energy_list = []
        # Try to load reference bond table (from previous run)
        init_bond_table = read_bond_table_file(reference_bond_table_file_path)
        if init_bond_table is not None:
            print("Loaded reference bond connectivity from previous run.")
            print(f"Reference bond table: {init_bond_table}")
    else:
        bond_reject_count = 0
        rejected_energy_list = []
        init_bond_table = None
    
    with open(folder_name+"/input.txt", "a") as f:
        f.write(str(vars(args))+"\n")
    
    
    if args.target_atoms is not None:
        target_atoms = [i-1 for i in num_parse(args.target_atoms[0])]
    else:
        target_atoms = [i for i in range(len(init_geom_num_list))]
   
    init_INPUT = args.INPUT
    init_AFIR_CONFIG = args.manual_AFIR
   
    atom_pair_list = make_tgt_atom_pair(init_geom_num_list, init_element_list, target_atoms)
    random.shuffle(atom_pair_list)
    
    # prepare for the first calculation
    prev_rank_list = None
    no_update_count = 0
    if len(energy_list) == 0:
        count = len(energy_list)
    else:
        count = len(energy_list) - 1
    reason = ""
    
    if len(energy_list) == 0:
        print("initial conformer.")
        bpa = multioptpy.optimization.Optimize(args)
        bpa.run()
        if not bpa.optimized_flag:
            print("Optimization is failed. Exit...")
            sys.exit(1)
        energy = bpa.final_energy
        init_conformer = bpa.final_geometry #Bohr
        init_conformer = init_conformer * bohr2ang #Angstrom
        
        # Store bond connectivity from the initial OPTIMIZED structure
        if preserve_bond_connectivity:
            init_bond_table = get_bond_connectivity_table(init_conformer, init_element_list)
            print("Bond connectivity preservation is enabled.")
            print("Reference bond connectivity is determined from the initial OPTIMIZED structure (EQ0).")
            print(f"Reference bond table: {init_bond_table}")
            
            # Save reference bond table to file for restart
            save_bond_table_file(init_bond_table, reference_bond_table_file_path)
            
            # Write human-readable log
            with open(folder_name+"/reference_bond_connectivity.log", "w") as f:
                f.write("Reference Bond Connectivity Table\n")
                f.write("="*50 + "\n")
                f.write("Obtained from: Initial OPTIMIZED structure (EQ0)\n")
                f.write("Note: This is the structure after optimization of the input XYZ file.\n")
                f.write("\nBond List (1-indexed):\n")
                for bond in init_bond_table:
                    f.write(f"  {bond[0]+1} - {bond[1]+1}\n")
                f.write(f"\nTotal number of bonds: {len(init_bond_table)}\n")
            
            # Create README for rejected folder
            with open(rejected_folder+"/README.txt", "w") as f:
                f.write("Rejected Conformers due to Bond Connectivity Change\n")
                f.write("="*55 + "\n\n")
                f.write("This folder contains conformers that were excluded from the\n")
                f.write("main conformer search results because their bond connectivity\n")
                f.write("differs from the initial OPTIMIZED structure (EQ0).\n\n")
                f.write("Reference structure: EQ0 (optimized from input XYZ)\n\n")
                f.write("Each rejected conformer has:\n")
                f.write("  - XYZ file: *_REJECTEDn.xyz\n")
                f.write("  - Info file: *_REJECTEDn_info.txt (contains bond change details)\n\n")
                f.write("These structures may represent:\n")
                f.write("  - Reaction products\n")
                f.write("  - Isomers with different connectivity\n")
                f.write("  - Structures with broken/formed bonds\n")
        
        energy_list.append(energy)
        with open(energy_list_file_path, 'a') as f:
            f.write(str(energy)+"\n")
        
        # Initialize visit count for the first conformer
        if use_tabu_search:
            visit_counts.append(0)
            with open(visit_counts_file_path, 'w') as f:
                for vc in visit_counts:
                    f.write(str(vc)+"\n")
        
        print("initial conformer.")
        print("Energy: ", energy)
        save_xyz_file(init_conformer, init_element_list, folder_name+"/"+init_INPUT, "EQ"+str(0))                
    
    if len(atom_pair_list) == 0:
        print("Cannot make atom_pair list. exit...")
        sys.exit(1)
    else:
        with open(folder_name+"/search_atom_pairs.log", 'a') as f:
            for atom_pair in atom_pair_list:
                f.write(str(atom_pair[0])+" "+str(atom_pair[1])+"\n")
            
    for i in range(args.max_samples):
        if os.path.exists(folder_name+"/end.txt"):
            print("The stop signal is detected. Exit....")
            reason = "The stop signal is detected. Exit...."
            break
        
        if len(atom_pair_list) <= i + 1:
            print("All possible atom pairs are searched. Exit....")
            reason = "All possible atom pairs are searched. Exit...."
            break
        
        print("Sampling conformation: ", i)
        
        
        atom_pair = atom_pair_list[i][0]
        if atom_pair_list[i][1] == "p":
            args.manual_AFIR = init_AFIR_CONFIG + [str(args.base_force), str(atom_pair[0]+1), str(atom_pair[1]+1)]
        else:
            args.manual_AFIR = init_AFIR_CONFIG + [str(-args.base_force), str(atom_pair[0]+1), str(atom_pair[1]+1)]
        
        bpa = multioptpy.optimization.Optimize(args)
        bpa.run()
        DC_check_flag = bpa.state.DC_check_flag
        if not DC_check_flag:
            bias_opted_geom_num_list = bpa.final_geometry #Bohr
            bias_opted_geom_num_list = bias_opted_geom_num_list * bohr2ang #Angstrom
            sample_file_name = save_xyz_file(bias_opted_geom_num_list, init_element_list, init_INPUT, "tmp")
            args.INPUT = sample_file_name
            args.manual_AFIR = init_AFIR_CONFIG
            bpa = multioptpy.optimization.Optimize(args)
            bpa.run()
            optimized_flag = bpa.optimized_flag
            energy = bpa.final_energy
            conformer = bpa.final_geometry #Bohr
            conformer = conformer * bohr2ang #Angstrom
            
            # Check bond connectivity if preservation is enabled
            bond_preserved = True
            added_bonds = []
            removed_bonds = []
            if preserve_bond_connectivity:
                conformer_bond_table = get_bond_connectivity_table(conformer, init_element_list)
                bond_preserved, added_bonds, removed_bonds = check_bond_connectivity_preserved(init_bond_table, conformer_bond_table)
                if not bond_preserved:
                    # Save rejected conformer to separate folder
                    rejected_file = save_rejected_conformer(
                        conformer, init_element_list, energy, i,
                        rejected_folder, init_INPUT, added_bonds, removed_bonds,
                        bond_reject_count
                    )
                    print(f"Rejected conformer saved to: {rejected_file}")
                    
                    # Update rejected energy list
                    rejected_energy_list.append(energy)
                    with open(rejected_energy_file_path, 'a') as f:
                        f.write(str(energy)+"\n")
                    
                    # Update rejected count
                    bond_reject_count += 1
                    with open(rejected_count_file_path, 'w') as f:
                        f.write(str(bond_reject_count))
                    
                    # Log to summary file
                    with open(rejected_folder+"/rejected_summary.log", "a") as f:
                        f.write(f"REJECTED{bond_reject_count-1}: Trial {i}, Energy {energy:.8f}\n")
                        if added_bonds:
                            f.write(f"  New bonds: {[(b[0]+1, b[1]+1) for b in added_bonds]}\n")
                        if removed_bonds:
                            f.write(f"  Broken bonds: {[(b[0]+1, b[1]+1) for b in removed_bonds]}\n")
            
            # Check identical
            bool_identical = is_identical(conformer, energy, energy_list, folder_name, init_INPUT)
        else:
            optimized_flag = False
            bool_identical = True
            bond_preserved = True  # Not applicable when DC_check_flag is True
          
        
        if bool_identical or not optimized_flag or DC_check_flag or not bond_preserved:
            if not optimized_flag:
                print("Optimization is failed...")
            if DC_check_flag:    
                print("DC is detected...")
            if not bond_preserved:
                print("Bond connectivity changed. Conformer saved to REJECTED_CONFORMERS folder.")
        
        else:
            count += 1
            energy_list.append(energy)
            
            with open(energy_list_file_path, 'w') as f:
                for ene in energy_list:
                    f.write(str(ene)+"\n")
            
            # Add visit count for new conformer
            if use_tabu_search:
                visit_counts.append(0)
                with open(visit_counts_file_path, 'w') as f:
                    for vc in visit_counts:
                        f.write(str(vc)+"\n")
            
            print("Find new conformer.")
            print("Energy: ", energy)
            save_xyz_file(conformer, init_element_list, folder_name+"/"+init_INPUT, "EQ"+str(count))
        
        
        # Check termination criteria
        if len(energy_list) > args.number_of_rank:
            sorted_energy_list = np.sort(np.array(energy_list))
            rank_list = sorted_energy_list[:args.number_of_rank]
            
            if np.all(rank_list == prev_rank_list):
                no_update_count += 1
            else:
                no_update_count = 0
                
            prev_rank_list = rank_list
            
            if no_update_count > args.number_of_lowest:
                print("The number of lowest energy conformers is not updated. Exit....")
                reason = "The number of lowest energy conformers is not updated. Exit...."
                break
            with open(folder_name+"/no_update_count.log", "a") as f:
                f.write(str(i)+" "+str(no_update_count)+"\n")
            
        else: 
            print("The number of conformers is less than the number of rank.")
        
        # Switch conformer
        if len(energy_list) > 1:
            if no_stochastic:
                idx = 0
            else:
                if use_tabu_search:
                    # Use Tabu Search / Metadynamics-like frequency penalty
                    if i % 5 == 0:
                        idx, prob_dist = switch_conformer_with_tabu(
                            energy_list, visit_counts, sampling_temperature*10, tabu_alpha
                        )
                    else:
                        idx, prob_dist = switch_conformer_with_tabu(
                            energy_list, visit_counts, sampling_temperature, tabu_alpha)
                    
                    # Update visit count for selected conformer
                    visit_counts[idx] += 1
                    with open(visit_counts_file_path, 'w') as f:
                        for vc in visit_counts:
                            f.write(str(vc)+"\n")
                    
                    # Log selection probabilities (periodically for debugging)
                    if i % 10 == 0:
                        with open(folder_name+"/tabu_selection_log.log", "a") as f:
                            f.write(f"\nTrial {i}: Selection probabilities (with frequency penalty)\n")
                            f.write(f"  Visit counts: {visit_counts}\n")
                            f.write(f"  Probabilities: {[f'{p:.4f}' for p in prob_dist]}\n")
                            f.write(f"  Selected: EQ{idx}\n")
                else:
                    # Standard Boltzmann selection
                    if i % 5 == 0:
                        idx = switch_conformer(energy_list, sampling_temperature*10)
                    else:
                        idx = switch_conformer(energy_list, sampling_temperature)
            
            no_ext_init_INPUT = os.path.splitext(init_INPUT)[0]
            args.INPUT = folder_name + "/" + no_ext_init_INPUT + "_EQ" + str(idx) + ".xyz"
            print("Switch conformer: EQ"+str(idx))
            with open(folder_name+"/switch_conformer.log", 'a') as f:
                if use_tabu_search:
                    f.write(f"Trial {i}: Switch conformer: EQ{idx} (visit_count={visit_counts[idx]})\n")
                else:
                    f.write("Trial "+str(i)+": Switch conformer: EQ"+str(idx)+"\n")
        else:
            args.INPUT = init_INPUT
        
        ##########
    else:
        print("Max samples are reached. Exit....")
        reason = "Max samples are reached. Exit...."
    
    energy_list_suumary_file_path = folder_name+"/EQ_summary.log"
    with open(energy_list_suumary_file_path, "w") as f:
        f.write("Summary\n"+"Reason of Termination: "+reason+"\n")
        print("conformer of lowest energy: ", min(energy_list))
        f.write("conformer of lowest energy: "+str(min(energy_list))+"\n")
        print("structure of lowest energy: ", "EQ"+str(energy_list.index(min(energy_list))))
        f.write("structure of lowest energy: "+"EQ"+str(energy_list.index(min(energy_list)))+"\n")
        print("conformer of highest energy: ", max(energy_list))
        f.write("conformer of highest energy: "+str(max(energy_list))+"\n")
        print("structure of highest energy: ", "EQ"+str(energy_list.index(max(energy_list))))
        f.write("structure of highest energy: "+"EQ"+str(energy_list.index(max(energy_list)))+"\n")
        if preserve_bond_connectivity:
            f.write(f"\nBond Connectivity Preservation Results:\n")
            f.write(f"  Reference structure: EQ0 (initial optimized structure)\n")
            f.write(f"  Conformers rejected: {bond_reject_count}\n")
            f.write(f"  Rejected conformers saved in: REJECTED_CONFORMERS/\n")
            if rejected_energy_list:
                f.write(f"  Rejected conformer energies:\n")
                for rej_idx, rej_ene in enumerate(rejected_energy_list):
                    f.write(f"    REJECTED{rej_idx}: {rej_ene}\n")
                f.write(f"  Lowest rejected energy: {min(rejected_energy_list)}\n")
                f.write(f"  Highest rejected energy: {max(rejected_energy_list)}\n")
        if use_tabu_search:
            f.write(f"\nTabu Search Statistics:\n")
            f.write(f"  Alpha (penalty coefficient): {tabu_alpha}\n")
            f.write(f"  Final visit counts: {visit_counts}\n")
            total_visits = sum(visit_counts) if visit_counts else 0
            f.write(f"  Total conformer switches: {total_visits}\n")
        

    print("Conformation search is finished.")
    if preserve_bond_connectivity:
        print(f"Total conformers rejected due to bond connectivity change: {bond_reject_count}")
        if bond_reject_count > 0:
            print(f"Rejected conformers saved in: {rejected_folder}/")
    if use_tabu_search:
        print(f"Tabu Search - Final visit counts: {visit_counts}")