import numpy as np


def update_trust_radii(B_e, pre_B_e, pre_B_g, pre_move_vector, model_hess, geom_num_list, trust_radii):
    #ref.: https://geometric.readthedocs.io/en/latest/how-it-works.html#trust-radius-adjustment

    Sc = 2.0
    Ce = (np.dot(pre_B_g.reshape(1, len(geom_num_list)), pre_move_vector.reshape(len(geom_num_list), 1)) + 0.5 * np.dot(np.dot(pre_move_vector.reshape(1, len(geom_num_list)), model_hess), pre_move_vector.reshape(len(geom_num_list), 1)))
    r = (pre_B_e - B_e) / (Ce)
    print("reference_value_of_trust_radius: ", r)
    r_min = 0.25
    r_good = 0.75

    if r <= r_min or r >= (2.0 - r_min):
        trust_radii /= Sc
        print("decrease trust radii")
    
    elif r >= r_good and r <= (2.0 - r_good) and abs(np.linalg.norm(pre_move_vector) - trust_radii) < 1e-3:
        trust_radii *= Sc ** 0.5
        print("increase trust radii")
    
    else:
        print("keep trust radii")
            
    return np.clip(trust_radii, 0.01, 0.5)   
    