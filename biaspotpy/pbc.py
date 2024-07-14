import numpy as np
import itertools
import copy

from calc_tools import Calculationtools


def fragment_check(new_geometry, element_list):
    atom_label_list = [i for i in range(len(new_geometry))]
    fragm_atom_num_list = []
    while len(atom_label_list) > 0:
        tmp_fragm_list = Calculationtools().check_atom_connectivity(new_geometry, element_list, atom_label_list[0], covalent_radii_threshold_scale=1.2)
        
        for j in tmp_fragm_list:
            atom_label_list.remove(j)
        fragm_atom_num_list.append(tmp_fragm_list)
    
    print("\nfragm_list:", fragm_atom_num_list)
    
    return fragm_atom_num_list    
    

def apply_periodic_boundary_condition(geom_num_list, element_list, box, fragm_check=True):
    #geom_num_list: ndarray 3 × N, Bohr.
    #box: ndarray, [x, y, z] 
    if fragm_check:
        fragm_list = fragment_check(geom_num_list, element_list)
        center_of_mass_list = []
        
        
        for fragm in fragm_list:
            tmp_elem_list = [element_list[i] for i in fragm]
            center_of_mass = Calculationtools().calc_center_of_mass(geom_num_list[fragm], tmp_elem_list)
            center_of_mass_list.append(center_of_mass)
        
        for i in range(len(center_of_mass_list)):

            for j in range(3):
                if center_of_mass_list[i][j] < 0:
                    tmp_center_of_mass_point = center_of_mass_list[i][j] % box[j]
                    dist = abs(center_of_mass_list[i][j] - tmp_center_of_mass_point)
                
                    
                    for k in range(len(fragm_list[i])):
    
                        geom_num_list[fragm_list[i][k]][j] = copy.copy(geom_num_list[fragm_list[i][k]][j] + dist)
                    
                    
                elif center_of_mass_list[i][j] > box[j]:
                    tmp_center_of_mass_point = center_of_mass_list[i][j] % box[j]
                    dist = abs(center_of_mass_list[i][j] - tmp_center_of_mass_point)
    
                    for k in range(len(fragm_list[i])):
            
                        geom_num_list[fragm_list[i][k]][j] = copy.copy(geom_num_list[fragm_list[i][k]][j] - dist)
                        
                    
                else:
                    pass
    else:
        geom_num_list[:, 0] = geom_num_list[:, 0] % box[0]
        geom_num_list[:, 1] = geom_num_list[:, 1] % box[1]
        geom_num_list[:, 2] = geom_num_list[:, 2] % box[2]
        
    return geom_num_list
