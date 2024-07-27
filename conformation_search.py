import sys
import os
import random
sys.path.append('./biaspotpy')
import biaspotpy.calc_tools

import biaspotpy
import numpy as np
import itertools

bohr2ang = 0.529177210903

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
    sort_distance_matrix = np.sort(distance_matrix)
    return sort_distance_matrix

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

def conformation_search(parser):
    parser.add_argument("-bf", "--base_force", type=float, default=100.0, help='bias force to search conformations (default: 100.0 kJ)', required=True)
    parser.add_argument("-ms", "--max_samples", type=int, default=50, help='the number of trial of calculation (default: 50)')
    parser.add_argument("-nl", "--number_of_lowest",  type=int, default=5, help='termination condition of calculation for updating list (default: 5)')
    parser.add_argument("-nr", "--number_of_rank",  type=int, default=10, help='termination condition of calculation for making list (default: 10)')
    parser.add_argument("-tgta", "--target_atoms", nargs="*", type=str, help='the atom to add bias force to perform conformational search (ex.) 1,2,3 or 1-3', default=None)
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
    
    for i in range(1, len(data)):
        splitted_data = data[i].split()
        
        energy_list.append(float(splitted_data[0]))
    
    return energy_list

def make_tgt_atom_pair(geom_num_list, element_list, target_atoms):
    norm_dist_min = 1.5
    norm_dist_max = 5.0
    norm_distance_list = biaspotpy.calc_tools.calc_normalized_distance_list(geom_num_list, element_list)
    bool_tgt_atom_list = np.where((norm_dist_min < norm_distance_list) & (norm_distance_list < norm_dist_max), True, False)
    updated_target_atom_pairs = []
    for i, j in itertools.combinations(target_atoms, 2):
        pair_idx = return_pair_idx(i, j)
        if bool_tgt_atom_list[pair_idx]:
            updated_target_atom_pairs.append([i, j])
    
    return updated_target_atom_pairs


def is_identical(conformer, energy, energy_list, folder_name, init_INPUT,ene_threshold=1e-4, dist_threshold=1e-1):
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


if __name__ == '__main__':
    parser = biaspotpy.interface.init_parser()
    parser = conformation_search(parser)
    args = biaspotpy.interface.optimizeparser(parser)
    
    init_geom_num_list, init_element_list = read_xyz(args.INPUT)
    
    folder_name = os.path.splitext(args.INPUT)[0]+"_"+str(int(args.base_force))+"KJ_CS_REPORT"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    
    energy_list_file_path = folder_name+"/EQ_energy.dat"
    if os.path.exists(energy_list_file_path):
        energy_list = read_energy_file(energy_list_file_path)
        
    else:
        energy_list = []
    
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
    EQ_num = len(energy_list)
    count = 0
    reason = ""
    if len(energy_list) == 0:
        print("initial conformer.")
        bpa = biaspotpy.optimization.Optimize(args)
        bpa.run()
        
        energy = bpa.final_energy
        init_conformer = bpa.final_geometry #Bohr
        init_conformer = init_conformer * bohr2ang #Angstrom
        energy_list.append(energy)
        with open(energy_list_file_path, 'a') as f:
            f.write(str(energy)+"\n")
        print("initial conformer.")
        print("Energy: ", energy)
        save_xyz_file(init_conformer, init_element_list, folder_name+"/"+init_INPUT, "EQ"+str(0))                
    
    
    for i in range(args.max_samples):
        if os.path.exists(folder_name+"/end.txt"):
            print("The stop signal is detected. Exit....")
            reason = "The stop signal is detected. Exit...."
            break
        
        if len(atom_pair_list) < i:
            print("All possible atom pairs are searched. Exit....")
            reason = "All possible atom pairs are searched. Exit...."
            break
        
        print("Sampling conformation: ", i)
        args.INPUT = init_INPUT
        
        atom_pair = atom_pair_list[i]
        
        args.manual_AFIR = init_AFIR_CONFIG + [str(args.base_force), str(atom_pair[0]+1), str(atom_pair[1]+1)]
        
        bpa = biaspotpy.optimization.Optimize(args)
        bpa.run()
        
        bias_opted_geom_num_list = bpa.final_geometry #Bohr
        bias_opted_geom_num_list = bias_opted_geom_num_list * bohr2ang #Angstrom
        sample_file_name = save_xyz_file(bias_opted_geom_num_list, init_element_list, init_INPUT, "tmp")
        args.INPUT = sample_file_name
        args.manual_AFIR = init_AFIR_CONFIG
        bpa = biaspotpy.optimization.Optimize(args)
        bpa.run()
        
        energy = bpa.final_energy
        conformer = bpa.final_geometry #Bohr
        conformer = conformer * bohr2ang #Angstrom
        # Check identical
        bool_identical = is_identical(conformer, energy, energy_list, folder_name, init_INPUT)
        
        if bool_identical:
            pass
        
        else:
            count += 1
            energy_list.append(energy)
            
            with open(energy_list_file_path, 'w') as f:
                for energy in energy_list:
                    f.write(str(energy)+"\n")
            
            print("Find new conformer.")
            print("Energy: ", energy)
            save_xyz_file(conformer, init_element_list, folder_name+"/"+init_INPUT, "EQ"+str(count+EQ_num))
        
        
        # Check termination criteria
        if len(energy_list) > args.number_of_rank:
            sorted_energy_list = np.sort(np.array(energy_list))
            rank_list = sorted_energy_list[-args.number_of_rank:]
            
            if np.all(rank_list == prev_rank_list):
                no_update_count += 1
            else:
                no_update_count = 0
                
            prev_rank_list = rank_list
            
            if no_update_count > args.number_of_lowest:
                print("The number of lowest energy conformers is not updated. Exit....")
                reason = "The number of lowest energy conformers is not updated. Exit...."
                break
            
        else: 
            print("The number of conformers is less than the number of rank.")
        
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
        

    print("Conformation search is finished.")
    
    
        
        
        
        
        
    
    
       