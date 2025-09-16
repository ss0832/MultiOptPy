import numpy as np
from scipy.signal import argrelextrema
from multioptpy.Parameters.parameter import atomic_mass

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

class CaluculationOM:
    def __init__(self, APPLY_CI_NEB=99999):
        self.spring_constant_k = 0.01
        self.APPLY_CI_NEB = APPLY_CI_NEB
        self.force_const_for_cineb = 0.01
        
    def calc_force(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):#J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593
        #This improved NEB method is inspired by the Onsager-Machlup (OM) action.
        print("OMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOM")
        local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)
        delta_t = 1
        damping_const = 1
        OM_spring_const = 0.001
       
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

            L_minus = []
            L_neutral = []
            for t in range(len(geometry_num_list[i])):
                atom_mass = atomic_mass(element_list[t])
                L_minus.append(-(delta_t/(atom_mass * damping_const)) * gradient_list[i-1][t])
                L_neutral.append(-(delta_t/(atom_mass * damping_const)) * gradient_list[i][t])
            L_minus = np.array(L_minus, dtype="float64")
            L_neutral = np.array(L_neutral, dtype="float64")
        
            OM_action_force = OM_spring_const * (geometry_num_list[i+1] + geometry_num_list[i-1] - 2 * geometry_num_list[i] + L_minus - L_neutral)
            
            cos_phi = np.sum((geometry_num_list[i+1]-geometry_num_list[i]) * (geometry_num_list[i]-geometry_num_list[i-1])) /(np.linalg.norm(geometry_num_list[i+1]-geometry_num_list[i]) * np.linalg.norm(geometry_num_list[i]-geometry_num_list[i-1]))
            phi = np.arccos(cos_phi)
            
            if 0 <= phi and phi <= np.pi/2:
                f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
            else:
                f_phi = 1.0
            OM_action_force_perpendicularity, OM_action_force_parallelism = [], []
            
            force_perpendicularity, force_parallelism = [], []
            
            if energy_list[i] == energy_list[local_max_energy_list_index[0]] and self.APPLY_CI_NEB < optimize_num: #CI-NEB
                for f in range(len(geometry_num_list[i])):
                    force_perpendicularity.append(np.array((-1)*self.force_const_for_cineb*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau[f]))*tau[f]), dtype = "float64"))
                    #print(str(force_perpendicularity))
                total_force = np.array(force_perpendicularity, dtype="float64")
                del local_max_energy_list_index[0]

            else:    
                for f in range(len(geometry_num_list[i])):
                    grad = 0.0
                    
                    for gg in range(len(gradient_list[i])):
                        grad += np.linalg.norm(gradient_list[i][gg], ord=2)
                    
                    grad = grad/len(gradient_list[i])
                    
       
                    #print("spring_constant:",self.spring_constant_k)
                    
                    force_parallelism.append(np.array((self.spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*self.spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2)))*tau[f], dtype = "float64"))
                    OM_action_force_parallelism.append(np.array(OM_action_force[f] * np.dot(tau[f], tau[f]), dtype = "float64"))
                
                    force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
                    OM_action_force_perpendicularity.append(f_phi * np.array(OM_action_force[f]-(np.dot(OM_action_force[f], tau[f]))*tau[f], dtype = "float64"))
                    #doubly nudged elastic band method :https://doi.org/10.1063/1.1636455
                
                    
                force_perpendicularity, force_parallelism = np.array(force_perpendicularity, dtype = "float64"), np.array(force_parallelism, dtype = "float64")
                
                OM_action_force_perpendicularity, OM_action_force_parallelism = np.array(OM_action_force_perpendicularity, dtype = "float64"), np.array(OM_action_force_parallelism, dtype = "float64")
                
                
                
                
                total_force = np.array((-1)*force_perpendicularity - force_parallelism + OM_action_force_parallelism + OM_action_force_perpendicularity, dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        
        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")


