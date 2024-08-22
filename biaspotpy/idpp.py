import numpy as np
import itertools
import copy

from calc_tools import return_pair_idx

ITER = 100
ALPHA = 5e-3

#Image Dependent Pair Potential
#ref.: J. Chem. Phys. 140(21), 214106 (2014). 

def calc_pair_distance(geometry):
    pair_distance_list = []
    
    for i, j in itertools.combinations(range(len(geometry)), 2):
        pair_distance_list.append(np.linalg.norm(geometry[i] - geometry[j]))
    pair_distance_list = np.array(pair_distance_list)
    return pair_distance_list


def make_pair_distance_list(geometry_list):
    #r_ij initial strucure
    init_pair_dist = calc_pair_distance(geometry_list[0])
    #r_ij final structure
    final_pair_dist = calc_pair_distance(geometry_list[-1])
    n_image = len(geometry_list)
    pair_distance_list = [init_pair_dist + k / n_image * (final_pair_dist - init_pair_dist) for k in range(n_image)]
    return pair_distance_list

def calc_idpp_gradient(geometry, distance):
    gradient = np.zeros_like(geometry)
    for i, j in itertools.combinations(range(len(geometry)), 2):
        idx = return_pair_idx(i, j)
        d_distance = distance[idx]
        distance_ij = np.linalg.norm(geometry[i] - geometry[j])
        gradient[i] += 2 * (1 / d_distance) ** 4 * (d_distance - distance_ij) * (geometry[i] - geometry[j])
        gradient[j] += 2 * (1 / d_distance) ** 4 * (d_distance - distance_ij) * (geometry[j] - geometry[i])
    gradient = np.array(gradient)
    return gradient

def calc_idpp(geometry, distance):
    idpp = 0.0
    for i, j in itertools.combinations(range(len(geometry)), 2):
        idx = return_pair_idx(i, j)
        d_distance = distance[idx]
        distance_ij = np.linalg.norm(geometry[i] - geometry[j])
        idpp += (1 / d_distance) ** 4 * (d_distance - distance_ij) ** 2
    return idpp

def optimize_initial_guess(geometry_list):
    print("IDPP optimization start")
    optimized_geometry_list = copy.copy(geometry_list)
    for i in range(ITER):
        pair_distance_list = make_pair_distance_list(geometry_list) 
        idpp_list = []
        idpp_grad_list = [np.zeros_like(geometry_list[0])]
        for j in range(1, len(geometry_list) - 1):
            idp_pot = calc_idpp(geometry_list[j], pair_distance_list[j])
            idpp_list.append(idp_pot)
            idpp_grad = calc_idpp_gradient(geometry_list[j], pair_distance_list[j])
            idpp_grad_list.append(idpp_grad)
        idpp_grad_list.append(np.zeros_like(geometry_list[-1]))
        
        trust_radii = min(np.linalg.norm(np.array(idpp_grad_list)), ALPHA)
        
        geometry_list -= trust_radii * np.array(idpp_grad_list) / np.linalg.norm(np.array(idpp_grad_list))
        
        if i % 10 == 0:
       
            print(f"Iteration: {i}")
            print(f"IDPP: {idpp_list}")
                    
    optimized_geometry_list = geometry_list
    print("Initial guess optimization done")
    
    return optimized_geometry_list