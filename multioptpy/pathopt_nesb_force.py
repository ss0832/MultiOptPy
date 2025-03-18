import numpy as np
from scipy.signal import argrelextrema

def extremum_list_index(energy_list):
    local_max_energy_list_index = argrelextrema(energy_list, np.greater)
    inverse_energy_list = (-1)*energy_list
    local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)

    local_max_energy_list_index = local_max_energy_list_index[0].tolist()
    local_min_energy_list_index = local_min_energy_list_index[0].tolist()
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    return local_max_energy_list_index, local_min_energy_list_index





class CaluculationNESB:
    def __init__(self, APPLY_CI_NEB=99999):
        self.spring_constant_k = 0.01
        self.APPLY_CI_NEB = APPLY_CI_NEB
        self.force_const_for_cineb = 0.01
        self.NESB_band_width = 0.1

    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):
        #Nudged elastic stiffness band method 
        #ref.: J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169
        print("NESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESB")
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)

        
        tau_list = [np.array(gradient_list[0], dtype = "float64") * 0.0]
        total_force_list = [((-1)*np.array(gradient_list[0], dtype = "float64")).tolist()]
        for i in range(1,len(energy_list)-1):
            tau_plus, tau_minus, tau = [], [], []
            
            delta_max_energy = np.array(max([(energy_list[i+1]-energy_list[i]),(energy_list[i-1]-energy_list[i])]), dtype = "float64")
            delta_min_energy = np.array(min([(energy_list[i+1]-energy_list[i]),(energy_list[i-1]-energy_list[i])]), dtype = "float64")
            
            if (energy_list[i-1] < energy_list[i]) and (energy_list[i] < energy_list[i+1]):
                for t in range(len(geometry_num_list[i])):
                    tau_vector = geometry_num_list[i+1][t]-geometry_num_list[i][t]
                    tau_norm = np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)
                    tau.append(np.divide(tau_vector, tau_norm, out=np.zeros_like(geometry_num_list[i][t]) ,where=np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)!=0).tolist())       
           
                 
            elif (energy_list[i-1] > energy_list[i]) and (energy_list[i] > energy_list[i+1]):
                for t in range(len(geometry_num_list[i])):      
                    tau_vector = geometry_num_list[i][t]-geometry_num_list[i-1][t]
                    tau_norm = np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)
                    tau.append(np.divide(tau_vector, tau_norm, out=np.zeros_like(geometry_num_list[i][t]), where=np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)!=0).tolist())       
          
            
            
            else: #((energy_list[i-1] >= energy_list[i]) and (energy_list[i] <= energy_list[i+1])) or ((energy_list[i-1] <= energy_list[i]) and (energy_list[i] >= energy_list[i+1])):
                for t in range(len(geometry_num_list[i])):         
                    tau_minus_vector = geometry_num_list[i][t]-geometry_num_list[i-1][t]
                    tau_minus_norm = np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)
                    tau_minus.append(np.divide(tau_minus_vector, tau_minus_norm
                                     ,out=np.zeros_like(geometry_num_list[i][t]),
                                     where=np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)!=0).tolist())       

                for t in range(len(geometry_num_list[i])):
                    tau_plus_vector = geometry_num_list[i+1][t]-geometry_num_list[i][t]
                    tau_plus_norm = np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)
                    tau_plus.append(np.divide(tau_plus_vector, tau_plus_norm, out=np.zeros_like(geometry_num_list[i][t]), where=np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)!=0).tolist())

                if energy_list[i-1] > energy_list[i+1]:
                    for t in range(len(geometry_num_list[i])):
                        tau_vector = (tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy)
                        tau_norm = np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy, ord=2)
                        tau.append(np.divide(tau_vector,tau_norm, out=np.zeros_like(tau_plus[0]) ,where=np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy!=0)).tolist())
                else:
                    for t in range(len(geometry_num_list[i])):
                        tau_vector = (tau_plus[t]*delta_max_energy+tau_minus[t]*delta_min_energy)
                        tau_norm = np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy, ord=2)
                        tau.append(np.divide(tau_vector, tau_norm,out=np.zeros_like(tau_minus[0]) ,where=np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy, ord=2)!=0 ).tolist())

            tau_plus, tau_minus, tau = np.array(tau_plus, dtype = "float64"), np.array(tau_minus, dtype = "float64"), np.array(tau, dtype = "float64")    
            #print("tau_minus:\n",tau_minus)
            #print("tau_plus:\n",tau_plus)
            #print("tau:\n",str(tau))
            tau_list.append(tau)
            #-----------------------------------
        tau_list.append(np.array(gradient_list[0], dtype = "float64") * 0.0)
        tangent_tau_list = [np.array(gradient_list[0], dtype = "float64") * 0.0]
        for i in range(1,len(energy_list)-1):
            #NESB

            tangent_tau = []
            
            v_1 = geometry_num_list[i-1] - geometry_num_list[i]
            v_2 = geometry_num_list[i+1] - geometry_num_list[i]
            
            
            for l in range(len(geometry_num_list[i])):
                
                v_1_tau = np.dot(v_1[l], tau_list[i][l].T)
                v_2_tau = np.dot(v_2[l], tau_list[i][l].T)
                if abs(v_1_tau) < 1e-8 and abs(v_2_tau) < 1e-8:
                    tangent_tau.append(v_1[l])
                
                elif abs(v_1_tau) < 1e-8:
                    tmp_a = -1 * (v_1_tau / v_2_tau)
                    tangent_tau.append(v_1[l]+v_2[l]*tmp_a)
                
                elif abs(v_1_tau) > 0.9 and abs(v_2_tau) > 0.9:
                   
                    tmp_a = -1 * (np.dot(tangent_tau_list[i-1][l], tau_list[i][l].T) / np.dot(tau_list[i][l], tau_list[i][l].T))
                    tangent_tau.append(tmp_a*tau_list[i][l]+tangent_tau_list[i-1][l])
              
                else:
                    tmp_a = -1 * (v_2_tau / v_1_tau)
                    tangent_tau.append(v_2[l]+v_1[l]*tmp_a) 
                
            
            tangent_tau = np.array(tangent_tau, dtype="float64")
            
            
            if i > 1:
                check_direction = np.sum(np.dot(tangent_tau, tangent_tau_list[i-1].T))
                if check_direction <= 0:
                    tangent_tau *= -1 
            tangent_tau = tangent_tau/(np.linalg.norm(tangent_tau)+1e-8)

            tangent_tau_list.append(tangent_tau)
            #force_stiff

        tangent_tau_list.append(np.array(gradient_list[0], dtype = "float64") * 0.0)    
            
            #-------------------------------------
        force_stiff_list = [np.array(gradient_list[0], dtype = "float64") * 0.0, np.array(gradient_list[0], dtype = "float64") * 0.0]
        for i in range(2,len(energy_list)-2):
            virtual_image_in_geometry_num_list = geometry_num_list[i] + 0.5 * self.NESB_band_width * tangent_tau_list[i]
            virtual_image_out_geometry_num_list = geometry_num_list[i] - 0.5 * self.NESB_band_width * tangent_tau_list[i]
            next_virtual_image_in_geometry_num_list = geometry_num_list[i+1] + 0.5 * self.NESB_band_width * tangent_tau_list[i+1]
            next_virtual_image_out_geometry_num_list = geometry_num_list[i+1] - 0.5 * self.NESB_band_width * tangent_tau_list[i+1]
            vi_in_geom_dist = np.linalg.norm(virtual_image_in_geometry_num_list) 
            vi_out_geom_dist = np.linalg.norm(virtual_image_out_geometry_num_list) 
            next_vi_in_geom_dist = np.linalg.norm(next_virtual_image_in_geometry_num_list) 
            next_vi_out_geom_dist = np.linalg.norm(next_virtual_image_out_geometry_num_list) 
            force_stiff_plus = 0.5 * (next_vi_out_geom_dist - next_vi_in_geom_dist) * tangent_tau_list[i+1] 
            force_stiff_minus = 0.5 * (vi_out_geom_dist - vi_in_geom_dist) * tangent_tau_list[i] 
            force_stiff = force_stiff_minus + force_stiff_plus
            force_stiff_list.append(force_stiff)
            
        force_stiff_list.append(np.array(gradient_list[0], dtype = "float64") * 0.0)
        force_stiff_list.append(np.array(gradient_list[0], dtype = "float64") * 0.0)
            
        for i in range(1,len(energy_list)-1):
            force_perpendicularity, force_parallelism = [], []
            
            if energy_list[i] == energy_list[local_max_energy_list_index[0]] and self.APPLY_CI_NEB < optimize_num: #CI-NEB
                for f in range(len(geometry_num_list[i])):
                    force_perpendicularity.append(np.array((-1)*self.force_const_for_cineb*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau_list[i][f]))*tau_list[i][f]), dtype = "float64"))
                    #print(str(force_perpendicularity))
                total_force = np.array(force_perpendicularity, dtype="float64")
                del local_max_energy_list_index[0]
                #print(str(total_force))
            #elif energy_list[i] == energy_list[local_min_energy_list_index[0]]: #for discovering intermidiate
            #    for f in range(len(geometry_num_list[i])):
            #        force_perpendicularity.append(np.array(((-1)*(gradient_list[i][f])), dtype = "float64"))
            #        #print(str(force_perpendicularity))
            #    total_force = np.array(force_perpendicularity, dtype="float64")
            #    del local_min_energy_list_index[0]
            else:    
                for f in range(len(geometry_num_list[i])):
                    grad = 0.0
                    
                    for gg in range(len(gradient_list[i])):
                        grad += np.linalg.norm(gradient_list[i][gg], ord=2)
                    
                    grad = grad/len(gradient_list[i])
                    
       
                    #print("spring_constant:",self.spring_constant_k)
                    
                    force_parallelism.append(np.array((self.spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*self.spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2)))*tau[f], dtype = "float64"))  
                
                    force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau_list[i][f]))*tau_list[i][f], dtype = "float64"))
                    #doubly nudged elastic band method :https://doi.org/10.1063/1.1636455
                
                   
                
                  
                force_perpendicularity, force_parallelism = np.array(force_perpendicularity, dtype = "float64"), np.array(force_parallelism, dtype = "float64")
                if np.sum(np.dot(force_parallelism, force_stiff_list[i].T)) > 0:
                    force_stiff_list[i] *= -1
                
                total_force = np.array((-1)*force_perpendicularity - force_parallelism + force_stiff_list[i], dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        
       
        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")