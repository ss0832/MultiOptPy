import numpy as np


def update_trust_radii(B_e, pre_B_e, pre_B_g, pre_move_vector, model_hess, geom_num_list, trust_radii, trust_radii_update):
    if trust_radii_update == "trust":
        Sc = 2.0
        Ce = (np.dot(pre_B_g.reshape(1, len(geom_num_list)), pre_move_vector.reshape(len(geom_num_list), 1)) + 0.5 * np.dot(np.dot(pre_move_vector.reshape(1, len(geom_num_list)), model_hess), pre_move_vector.reshape(len(geom_num_list), 1)))
        r = (pre_B_e - B_e) / (Ce)
        print("reference_value_of_trust_radius: ", r)
        r_min = 0.75
        r_good = 0.80
        if r <= r_min or r >= (2.0 - r_min):
            trust_radii /= Sc
            print("decrease trust radii")
        
        elif r >= r_good and r <= (2.0 - r_good) and abs(np.linalg.norm(pre_move_vector) - trust_radii) < 1e-3:
            trust_radii *= Sc ** 0.5
            print("increase trust radii")
        else:
            print("keep trust radii")
            
    elif trust_radii_update == "legacy":
        if pre_B_e >= B_e:
            trust_radii *= 3.0
        else:
            trust_radii *= 0.1
    else:
        pass
    
    return np.clip(trust_radii, 0.01, 1.0)                            
    
