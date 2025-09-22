import numpy as np

from multioptpy.Utils.calc_tools import calc_path_length_list

def distribute_geometry_by_length(geometry_list, angstrom_spacing):
    """Distribute geometries by specified distance spacing"""
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    new_geometry_list = []
    
    max_steps = int(total_length // angstrom_spacing)
    for i in range(1, max_steps):
        dist = i * angstrom_spacing
       
        for j in range(len(path_length_list) - 1):
            if path_length_list[j] <= dist <= path_length_list[j+1]:
                break
      
        delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
        new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
        new_geometry_list.append(new_geometry)

    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list

def distribute_geometry(geometry_list):
    """Distribute geometries evenly along the path"""
    nnode = len(geometry_list)
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    node_dist = total_length / (nnode-1)
    
    new_geometry_list = [geometry_list[0]]
    for i in range(1, nnode-1):
        dist = i * node_dist
        for j in range(nnode-1):
            if path_length_list[j] <= dist and dist <= path_length_list[j+1]:
                break
        delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
        new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
        new_geometry_list.append(new_geometry)
    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list

