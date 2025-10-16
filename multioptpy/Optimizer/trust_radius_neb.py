import numpy as np


class TR_NEB:
    def __init__(self, **config):
        self.NEB_FOLDER_DIRECTORY = config.get("NEB_FOLDER_DIRECTORY", None)
        self.fix_init_edge = config.get("fix_init_edge", False)
        self.fix_end_edge = config.get("fix_end_edge", False)
        self.apply_convergence_criteria = config.get("apply_convergence_criteria", False)
        self.threshold_max_force = 0.00045
        self.threshold_rms_force = 0.00030
        self.threshold_max_displacement = 0.0018
        self.threshold_rms_displacement = 0.0012
                
        return
    
    def TR_calc(self, geometry_num_list, total_force_list, total_delta, biased_energy_list, pre_biased_energy_list, pre_geom):
        if self.fix_init_edge:
            move_vector = [total_delta[0]*0.0]
        else:
            init_norm_move_vector = np.linalg.norm(total_delta[0])
            init_tr = min(0.5, init_norm_move_vector)
            if init_norm_move_vector < 1e-15:
                move_vector = [total_delta[0]*0.0]
            else:
                move_vector = [init_tr * total_delta[0] / init_norm_move_vector]
            
        trust_radii_1_list = []
        trust_radii_2_list = []
        
        for i in range(1, len(total_delta)-1):
            #if biased_energy_list[i] > pre_biased_energy_list[i] and pre_geom is not None:
            #    total_delta[i] = pre_geom[i] - geometry_num_list[i]
            #    print("Energy increased... ")
            
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            
            trust_radii_1_list.append(str(trust_radii_1*2))
            trust_radii_2_list.append(str(trust_radii_2*2))
            
            normalized_vec_1 = (geometry_num_list[i-1] - geometry_num_list[i])/(np.linalg.norm(geometry_num_list[i-1] - geometry_num_list[i]) + 1e-15)
            normalized_vec_2 = (geometry_num_list[i+1] - geometry_num_list[i])/(np.linalg.norm(geometry_num_list[i+1] - geometry_num_list[i]) + 1e-15)
            normalized_delta =  total_delta[i] / np.linalg.norm(total_delta[i])
            
            cos_1 = np.sum(normalized_vec_1 * normalized_delta) 
            cos_2 = np.sum(normalized_vec_2 * normalized_delta)
            
            force_move_vec_cos = np.sum(total_force_list[i] * total_delta[i]) / (np.linalg.norm(total_force_list[i]) * np.linalg.norm(total_delta[i])) 
            
            if force_move_vec_cos >= 0: #Projected velocity-verlet algorithm
                if (cos_1 > 0 and cos_2 < 0) or (cos_1 < 0 and cos_2 > 0):
                    if np.linalg.norm(total_delta[i]) > trust_radii_1 and cos_1 > 0:
                        move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
                        
                    elif np.linalg.norm(total_delta[i]) > trust_radii_2 and cos_2 > 0:
                        move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
                        
                    else:
                        move_vector.append(total_delta[i])
                        
                elif (cos_1 < 0 and cos_2 < 0):
                    move_vector.append(total_delta[i])
                    
                else:
                    if np.linalg.norm(total_delta[i]) > trust_radii_1:
                        move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
                        
                    elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                        move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
                        
                    else:
                        move_vector.append(total_delta[i])      
            else:
                print("no displacements (Projected velocity-verlet algorithm): # NODE "+str(i))
                move_vector.append(total_delta[i] * 0.0) 
            
            
        with open(self.NEB_FOLDER_DIRECTORY+"procrustes_distance_1.csv", "a") as f:
            f.write(",".join(trust_radii_1_list)+"\n")
        
        with open(self.NEB_FOLDER_DIRECTORY+"procrustes_distance_2.csv", "a") as f:
            f.write(",".join(trust_radii_2_list)+"\n")
        
        if self.fix_end_edge:
            move_vector.append(total_delta[-1]*0.0)
        else:
            end_norm_move_vector = np.linalg.norm(total_delta[-1])
            end_tr = min(0.5, end_norm_move_vector) 
            if end_norm_move_vector < 1e-15:
                move_vector.append(total_delta[-1]*0.0)
            else:
                move_vector.append(end_tr * total_delta[-1] / end_norm_move_vector)
        
        if self.apply_convergence_criteria:
            move_vector = self.check_convergence(total_force_list, move_vector)
        
        return move_vector        
    
    def check_convergence(self, total_force_list, move_vec_list):
        
        for i in range(1, len(total_force_list)-1):
            max_grad = np.max(total_force_list[i])
            rms_grad = np.sqrt(np.sum(total_force_list[i]**2)/(len(total_force_list[i])*3))
            max_move = np.max(move_vec_list[i])
            rms_move = np.sqrt(np.sum(move_vec_list[i]**2)/(len(move_vec_list[i])*3))
            print("--------------------")
            print("NODE #"+str(i))
            print(f"MAXIMUM NEB FORCE    :    {float(max_grad):12.8f}")
            print(f"RMS NEB FORCE        :    {float(rms_grad):12.8f}")
            print(f"MAXIMUM DISPLACEMENT :    {float(max_move):12.8f}")
            print(f"RMS DISPLACEMENT     :    {float(rms_move):12.8f}")
            
            if max_grad < self.threshold_max_force and rms_grad < self.threshold_rms_force and max_move < self.threshold_max_displacement and rms_move < self.threshold_rms_displacement:
                print("Converged?: YES")
                move_vec_list[i] = move_vec_list[i]*0.0
            else:
                print("Converged?: NO")
        print("--------------------")
        return move_vec_list
    