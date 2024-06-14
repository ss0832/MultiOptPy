import sys
import os

import numpy as np

from parameter import covalent_radii_lib, UnitValueLib



def link_number_high_layer_and_low_layer(high_layer_atom_num):
    total_2_highlayer_label_connect_dict = {}
    highlayer_2_total_label_connect_dict = {}
    for i in range(len(high_layer_atom_num)):
        total_2_highlayer_label_connect_dict[high_layer_atom_num[i]] = i + 1
        highlayer_2_total_label_connect_dict[i + 1] = high_layer_atom_num[i]
    
    return total_2_highlayer_label_connect_dict, highlayer_2_total_label_connect_dict


def separate_high_layer_and_low_layer(mol_list, linker_atom_pair_num, high_layer_atom_num, element_list):
    
    high_layer_geom_num_list = []
    high_layer_element_list = []
    
    for i in range(len(high_layer_atom_num)):
        high_layer_geom_num_list.append(mol_list[high_layer_atom_num[i]-1])
        high_layer_element_list.append(element_list[high_layer_atom_num[i]-1])

    high_layer_geom_num_list = np.array(high_layer_geom_num_list, dtype="float64")


    for base, link in linker_atom_pair_num:
        unit_vec = (mol_list[link-1] - mol_list[base-1]) / np.linalg.norm(mol_list[link-1] - mol_list[base-1])
        tmp_coord = np.array([mol_list[base-1] + unit_vec * (covalent_radii_lib(element_list[base-1]) + covalent_radii_lib("H"))], dtype="float64")
        
        high_layer_geom_num_list = np.append(high_layer_geom_num_list, tmp_coord, axis=0)   
        high_layer_element_list.append("H")
    
    return high_layer_geom_num_list, high_layer_element_list


def specify_link_atom_pairs(mol_list, element_list, high_layer_atom_num, link_atom_num, covalent_radii_threshold_scale=1.2):
    linker_atom_pair_num = []
    for link_atom in link_atom_num:
        min_dist = 1e+10
        min_dist_atom_num = 1000
        for high_layer_atom in high_layer_atom_num:
            dist = np.linalg.norm(mol_list[high_layer_atom-1] - mol_list[link_atom-1])
            if dist < min_dist:
                min_dist = dist
                min_dist_atom_num = high_layer_atom
        
        linker_atom_pair_num.append([min_dist_atom_num, link_atom])
    
    return linker_atom_pair_num#[[high_layer_atom_number, linker_atom_num] ...]
