import sys
import os
import random
sys.path.append('./biaspotpy')
import biaspotpy.calc_tools
import numpy as np

import biaspotpy

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


def make_random_orientation_xyz_file(coord_list, part_list, part_dist, input_file_name, n_sample):#ang.
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
        part_center = biaspotpy.calc_tools.Calculationtools().calc_center(part_coord_list[i], part_element_list[i])
        centered_part_coord_list = part_coord_list[i] - part_center 
        random_x_angle = random.uniform(0, 2*np.pi)
        random_y_angle = random.uniform(0, 2*np.pi)
        random_z_angle = random.uniform(0, 2*np.pi)
        print("Random angles (Radian): ", random_x_angle, random_y_angle, random_z_angle)
        rotated_part_coord = biaspotpy.calc_tools.rotate_molecule(centered_part_coord_list, "x", random_x_angle)
        rotated_part_coord = biaspotpy.calc_tools.rotate_molecule(rotated_part_coord, "y", random_y_angle)
        rotated_part_coord = biaspotpy.calc_tools.rotate_molecule(rotated_part_coord, "z", random_z_angle)
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



if __name__ == '__main__':
    parser = biaspotpy.interface.init_parser()
    parser = orientation_search(parser)
    args = biaspotpy.interface.optimizeparser(parser)
    
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
        args.INPUT = make_random_orientation_xyz_file(coord_list, part_list, part_dist, original_input_file, i)
        bpa = biaspotpy.optimization.Optimize(args)
        bpa.run()
    
        
        print("Sampling orientation: ", i, "done")
    
    print("Orientation search done")