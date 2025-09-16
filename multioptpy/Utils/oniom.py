import numpy as np
from multioptpy.Parameters.parameter import covalent_radii_lib

def link_number_high_layer_and_low_layer(high_layer_atom_num):
    """
    Create mapping dictionaries between high layer atom indices and full system indices.
    
    Args:
        high_layer_atom_num: List of atom indices in the high layer (1-indexed)
        
    Returns:
        Tuple of two dictionaries:
            real_2_highlayer_label_connect_dict: Maps full system indices to high layer indices
            highlayer_2_real_label_connect_dict: Maps high layer indices to full system indices
    """
    real_2_highlayer_label_connect_dict = {}
    highlayer_2_real_label_connect_dict = {}
    for i in range(len(high_layer_atom_num)):
        real_2_highlayer_label_connect_dict[high_layer_atom_num[i]] = i + 1
        highlayer_2_real_label_connect_dict[i + 1] = high_layer_atom_num[i]
    
    return real_2_highlayer_label_connect_dict, highlayer_2_real_label_connect_dict


def specify_link_atom_pairs(mol_list, element_list, high_layer_atom_num, link_atom_num, covalent_radii_threshold_scale=1.2):
    """
    Identify pairs of atoms that form the boundary between high and low layer.
    
    Args:
        mol_list: Coordinates of all atoms in the system (Bohr)
        element_list: List of element symbols for all atoms
        high_layer_atom_num: List of atom indices in the high layer (1-indexed)
        link_atom_num: List of atom indices that are linked to high layer (1-indexed)
        covalent_radii_threshold_scale: Scale factor for covalent radii threshold
        
    Returns:
        List of pairs [high_layer_atom_index, linker_atom_index]
    """
    # Handle case where no link atoms are specified
    if link_atom_num == "none":
        return []
    
    linker_atom_pair_num = []
    for link_atom in link_atom_num:
        min_dist = 1e+10
        min_dist_atom_num = None
        for high_layer_atom in high_layer_atom_num:
            dist = np.linalg.norm(mol_list[high_layer_atom-1] - mol_list[link_atom-1])
            if dist < min_dist:
                min_dist = dist
                min_dist_atom_num = high_layer_atom
        
        if min_dist_atom_num is not None:
            linker_atom_pair_num.append([min_dist_atom_num, link_atom])
    
    return linker_atom_pair_num  # [[high_layer_atom_number, linker_atom_num] ...]


def separate_high_layer_and_low_layer(mol_list, linker_atom_pair_num, high_layer_atom_num, element_list):
    """
    Create high-layer geometry and element lists, adding link atoms where needed.
    
    Args:
        mol_list: Coordinates of all atoms in the system (Bohr)
        linker_atom_pair_num: List of pairs [high_layer_atom_index, linker_atom_index]
        high_layer_atom_num: List of atom indices in the high layer (1-indexed)
        element_list: List of element symbols for all atoms
        
    Returns:
        Tuple of:
            high_layer_geom_num_list: Coordinates of high layer atoms (Bohr)
            high_layer_element_list: List of element symbols for high layer atoms
    """
    # Extract high layer atoms
    high_layer_geom_num_list = []
    high_layer_element_list = []
    
    for i in range(len(high_layer_atom_num)):
        high_layer_geom_num_list.append(mol_list[high_layer_atom_num[i]-1])
        high_layer_element_list.append(element_list[high_layer_atom_num[i]-1])

    high_layer_geom_num_list = np.array(high_layer_geom_num_list, dtype="float64")

    # Add link atoms (hydrogen atoms at appropriate positions)
    for base, link in linker_atom_pair_num:
        # Calculate unit vector from base atom to link atom
        vector = mol_list[link-1] - mol_list[base-1]
        distance = np.linalg.norm(vector)
        if distance > 0:  # Avoid division by zero
            unit_vec = vector / distance
            # Position link atom at appropriate distance from base atom
            link_atom_position = mol_list[base-1] + unit_vec * (covalent_radii_lib(element_list[base-1]) + covalent_radii_lib("H"))
            high_layer_geom_num_list = np.append(high_layer_geom_num_list, [link_atom_position], axis=0)   
            high_layer_element_list.append("H")
    
    return high_layer_geom_num_list, high_layer_element_list