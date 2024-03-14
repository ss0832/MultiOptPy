import os
import numpy as np
import csv
import glob
import time
import matplotlib.pyplot as plt
import random
import copy
from scipy.signal import argrelextrema

try:
    import psi4
except:
    print("You can't use psi4.")

try:
    from tblite.interface import Calculator
except:
    print("You can't use extended tight binding method.")

#reference about LUP method:J. Chem. Phys. 94, 751–760 (1991) https://doi.org/10.1063/1.460343

color_list = ["g"] #use for matplotlib

from interface import force_data_parser

from parameter import element_number, atomic_mass
from potential import BiasPotentialCalculation
from optimizer import Model_hess_tmp
from calc_tools import Calculationtools 

class NEB:
    def __init__(self, args):
    
        self.basic_set_and_function = args.functional+"/"+args.basisset
        self.QUASI_NEWTOM_METHOD = args.QUASI_NEWTOM_METHOD
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.NEB_NUM = args.NSTEP
        self.partition = args.partition
        #please input psi4 inputfile.

        self.spring_constant_k = 0.01
        self.bohr2angstroms = 0.52917721067
        self.hartree2kcalmol = 627.509
        #parameter_for_FIRE_method
        self.FIRE_dt = 0.1
        self.FIRE_N_accelerate = 5
        self.FIRE_f_inc = 1.10
        self.FIRE_f_accelerate = 0.99
        self.FIRE_f_decelerate = 0.5
        self.FIRE_a_start = 0.1
        self.FIRE_dt_max = 3.0
        self.APPLY_CI_NEB = args.apply_CI_NEB
        self.start_folder = args.INPUT
        self.om = args.OM
        self.lup = args.LUP
        self.dneb = args.DNEB
        self.nesb = args.NESB
        self.NESB_band_width = 0.1
        self.usextb = args.usextb
        self.sd = args.steepest_descent
        self.unrestrict = args.unrestrict
        if args.usextb == "None":
            self.NEB_FOLDER_DIRECTORY = args.INPUT+"_NEB_"+self.basic_set_and_function.replace("/","_")+"_"+str(time.time())+"/"
        else:
            self.NEB_FOLDER_DIRECTORY = args.INPUT+"_NEB_"+self.usextb+"_"+str(time.time())+"/"
        self.args = args
        os.mkdir(self.NEB_FOLDER_DIRECTORY)
        self.ANEB_num = args.ANEB_num
        if args.fixedges <= 0:
        
            self.fix_init_edge = False
            self.fix_end_edge = False
        elif args.fixedges == 1:
            self.fix_init_edge = True
            self.fix_end_edge = False
        elif args.fixedges == 2:
            self.fix_init_edge = False
            self.fix_end_edge = True
        else:
            self.fix_init_edge = True
            self.fix_end_edge = True
        
        self.force_const_for_cineb = 0.01
        return

    def force2velocity(self, gradient_list, element_list):
        velocity_list = gradient_list
        return np.array(velocity_list, dtype="float64")

    def make_geometry_list(self, start_folder, partition_function):
        #print(partition_function)
        start_file_list = glob.glob(start_folder+"/*_[0-9].xyz") + glob.glob(start_folder+"/*_[0-9][0-9].xyz") + glob.glob(start_folder+"/*_[0-9][0-9][0-9].xyz") + glob.glob(start_folder+"/*_[0-9][0-9][0-9][0-9].xyz")
        loaded_geometry_list = []
        geometry_list = []
        for start_file in start_file_list:
            with open(start_file,"r") as f:
                reader = csv.reader(f, delimiter=' ')
                pre_start_data = [row for row in reader]
                start_data = []
                for i in pre_start_data:
                    start_data.append([row.strip() for row in i if row != "" and row != "\t"])    
                loaded_geometry_list.append(start_data)
        #print(len(loaded_geometry_list))
        
        electric_charge_and_multiplicity = start_data[0]
        element_list = []
        loaded_geometry_num_list = []

        for i in range(1, len(start_data)):
            element_list.append(start_data[i][0])
            #start_num_list.append(start_data[i][1:4])
            #end_num_list.append(end_data[i][1:4])
        for geom_list in loaded_geometry_list:
            num_list = []
            for i in range(1, len(start_data)):
                num_list.append(geom_list[i][1:4])
            loaded_geometry_num_list.append(num_list)
        geometry_list.append(loaded_geometry_list[0])
        for k in range(len(loaded_geometry_list)-1):    
            delta_num_geom = (np.array(loaded_geometry_num_list[k+1], dtype = "float64") - np.array(loaded_geometry_num_list[k], dtype = "float64")) / (partition_function+1)
            frame_geom = np.array(loaded_geometry_num_list[k], dtype = "float64")
            for i in range(0, partition_function+1):
                
                for j in range(1, len(start_data)):
                    frame_geom = np.array(loaded_geometry_num_list[k], dtype = "float64") + delta_num_geom*i
                    frame_file = [start_data[0]]+[[]*n for n in range(len(start_data)-1)]
                    for x in range(0, len(start_data)-1):
                        frame_file[x+1].append(element_list[x])
                        frame_file[x+1].extend(frame_geom[x].tolist())
                geometry_list.append(frame_file)
        
        geometry_list.append(loaded_geometry_list[-1])
        
        #print(len(geometry_list))
        print("\n geometry datas are loaded. \n")
        
        return geometry_list, element_list, electric_charge_and_multiplicity

    def make_geometry_list_2(self, new_geometory, element_list, electric_charge_and_multiplicity):
        new_geometory = new_geometory.tolist()
        #print(new_geometory)
        geometry_list = []
        for geometries in new_geometory:
            new_data = [electric_charge_and_multiplicity]
            for num, geometry in enumerate(geometries):
                geometory = list(map(str, geometry))
                geometory = [element_list[num]] + geometory
                new_data.append(geometory)
            
            geometry_list.append(new_data)
        return geometry_list

    def make_psi4_input_file(self, geometry_list, optimize_num):
        file_directory = self.NEB_FOLDER_DIRECTORY+"samples_"+str(optimize_num)+"_"+str(self.start_folder)+"_"+str(time.time())
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.start_folder+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory
        
    def sinple_plot(self, num_list, energy_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Electronic Energy [kcal/mol]", name="energy"):
        fig, ax = plt.subplots()
        ax.plot(num_list,energy_list, color_list[random.randint(0,len(color_list)-1)]+"--o" )

        ax.set_title(str(optimize_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(self.NEB_FOLDER_DIRECTORY+"Plot_"+name+"_"+str(optimize_num)+"_"+str(time.time())+".png", format="png", dpi=200)
        plt.close()
        #del fig, ax

        return
        
    def psi4_calculation(self, file_directory, optimize_num, pre_total_velocity):
        psi4.core.clean()
        gradient_list = []
        gradient_norm_list = []
        energy_list = []
        geometry_num_list = []
        num_list = []
        delete_pre_total_velocity = []
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")
            
                logfile = file_directory+"/"+self.start_folder+'_'+str(num)+'.log'
                #psi4.set_options({'pcm': True})
                #psi4.pcm_helper(pcm)
                
                psi4.set_output_file(logfile)
                
                
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                if self.unrestrict:
                    psi4.set_options({'reference': 'uks'})
                with open(input_file,"r") as f:
                    input_data = f.read()
                    input_data = psi4.geometry(input_data)
                    input_data_for_display = np.array(input_data.geometry(), dtype = "float64")
                if np.nanmean(np.nanmean(input_data_for_display)) > 1e+5:
                    raise Exception("geometry is abnormal.")
                    #print('geometry:\n'+str(input_data_for_display))            
            
               
                g, wfn = psi4.gradient(self.basic_set_and_function, molecule=input_data, return_wfn=True)
                g = np.array(g, dtype = "float64")
                e = float(wfn.energy())
  
                #print("gradient:\n"+str(g))
                print('energy:'+str(e)+" a.u.")

                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))#RMS
                energy_list.append(e)
                num_list.append(num)
                geometry_num_list.append(input_data_for_display)
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
                
                
            psi4.core.clean()
        print("data sampling completed...")


        try:
            self.sinple_plot(num_list, np.array(energy_list, dtype="float64")*self.hartree2kcalmol, file_directory, optimize_num)
            print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            self.sinple_plot(num_list, gradient_norm_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="RMS Gradient [a.u.]", name="gradient")
          
            print("gradient graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot gradient graph.")
        
        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return np.array(energy_list, dtype = "float64"), np.array(gradient_list, dtype = "float64"), np.array(geometry_num_list, dtype = "float64"), pre_total_velocity
    
    def tblite_calculation(self, file_directory, optimize_num, pre_total_velocity, element_number_list, electric_charge_and_multiplicity):
        #execute extended tight binding method calclation.
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        finish_frag = False
        method = self.args.usextb
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")

                with open(input_file,"r") as f:
                    input_data = f.readlines()
                    
                positions = []
                for word in input_data[1:]:
                    positions.append(word.split()[1:4])
                        
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                if int(electric_charge_and_multiplicity[1]) > 1 or self.unrestrict:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
                else:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]))                
                calc.set("max-iter", 500)
                calc.set("verbosity", 1)
                
                
                res = calc.singlepoint()
                e = float(res.get("energy"))  #hartree
                g = res.get("gradient") #hartree/Bohr
                        
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))#RMS
                geometry_num_list.append(positions)
                num_list.append(num)
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        try:
            self.sinple_plot(num_list, np.array(energy_list, dtype="float64")*self.hartree2kcalmol, file_directory, optimize_num)
            print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            self.sinple_plot(num_list, gradient_norm_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Gradient (RMS) [a.u.]", name="gradient")
          
            print("gradient graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot gradient graph.")

        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return np.array(energy_list, dtype = "float64"), np.array(gradient_list, dtype = "float64"), np.array(geometry_num_list, dtype = "float64"), pre_total_velocity

    def xyz_file_make(self, file_directory):
        print("\ngeometry integration processing...\n")
        file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
        #print(file_list,"\n")
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
                with open(file_directory+"/"+self.start_folder+"_integration.xyz","a") as w:
                    atom_num = len(sample)-1
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                del sample[0]
                for i in sample:
                    with open(file_directory+"/"+self.start_folder+"_integration.xyz","a") as w2:
                        w2.write(i)
        print("\ngeometry integration complete...\n")
        return


    def extremum_list_index(self, energy_list):
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

    def OM_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):#J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593
        #This improved NEB method is inspired by the Onsager-Machlup (OM) action.
        print("OMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOMOM")
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)
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

    def NEB_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num):
        print("NEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEBNEB")
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)

      
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
            force_perpendicularity, force_parallelism = [], []
            
            if energy_list[i] == energy_list[local_max_energy_list_index[0]] and self.APPLY_CI_NEB < optimize_num: #CI-NEB
                for f in range(len(geometry_num_list[i])):
                    force_perpendicularity.append(np.array((-1)*self.force_const_for_cineb*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau[f]))*tau[f]), dtype = "float64"))
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
                
                    force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
                    #doubly nudged elastic band method :https://doi.org/10.1063/1.1636455
                
                   
                
                  
                force_perpendicularity, force_parallelism = np.array(force_perpendicularity, dtype = "float64"), np.array(force_parallelism, dtype = "float64")
                total_force = np.array((-1)*force_perpendicularity - force_parallelism, dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        
       
        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")



    def NESB_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num):
        #Nudged elastic stiffness band method 
        #ref.: J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169
        print("NESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESBNESB")
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)

        
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

    def DNEB_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num):
        print("DNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEBDNEB")
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)

      
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
            force_perpendicularity, force_parallelism, force_parallelism_perpendicularity , swiching_double_neb_force = [], [], [], []
            
            if energy_list[i] == energy_list[local_max_energy_list_index[0]] and self.APPLY_CI_NEB < optimize_num: #CI-NEB
                for f in range(len(geometry_num_list[i])):
                    force_perpendicularity.append(np.array((-1)*self.force_const_for_cineb*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau[f]))*tau[f]), dtype = "float64"))
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
                
                    force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
                    #doubly nudged elastic band method :https://doi.org/10.1063/1.1636455
                
                    force_parallelism_perpendicularity.append(np.array(np.array((self.spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*self.spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2))), dtype = "float64") - (np.dot(np.array((self.spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*self.spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2))), dtype = "float64"), tau[f]))*tau[f], dtype = "float64"))
                
                    swiching_double_neb_force.append((2.0/np.pi)*np.arctan(np.divide(np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)**2, np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)**2 ,out=np.zeros_like(force_parallelism_perpendicularity[f]) ,where=np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)!=0))*(force_parallelism_perpendicularity[f] - np.dot(force_parallelism_perpendicularity[f],force_perpendicularity[f])*force_perpendicularity[f]))
            
                force_perpendicularity, force_parallelism, force_parallelism_perpendicularity, swiching_double_neb_force = np.array(force_perpendicularity, dtype = "float64"), np.array(force_parallelism, dtype = "float64"),np.array(force_parallelism_perpendicularity, dtype = "float64"), np.array(swiching_double_neb_force, dtype = "float64")
                total_force = np.array((-1)*force_perpendicularity - force_parallelism - swiching_double_neb_force, dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        
       
        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")

    def LUP_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num):
        print("LUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUPLUP")
        #ref. Ulitsky, A., & Elber, R. (1990). A new technique to calculate steepest descent paths in flexible polyatomic systems. The Journal of Chemical Physics, 92(2), 1510.
        #https://doi.org/10.1063/1.458112
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)

           
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
            force_perpendicularity = []
            
        
            for f in range(len(geometry_num_list[i])):
                grad = 0.0
                    
                for gg in range(len(gradient_list[i])):
                    grad += np.linalg.norm(gradient_list[i][gg], ord=2)
                    
                grad = grad/len(gradient_list[i])
                    
             
                
                force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
               
                
            
            force_perpendicularity = np.array(force_perpendicularity, dtype = "float64")
            total_force = np.array((-1)*force_perpendicularity, dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        

        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")



    def FIRE_calc(self, geometry_num_list, total_force_list, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a, cos_list):
        velocity_neb = []
        
        for num, each_velocity in enumerate(total_velocity):
            part_velocity_neb = []
            for i in range(len(total_force_list[0])):
                 
                part_velocity_neb.append((1.0-a)*total_velocity[num][i]+a*np.sqrt(np.dot(total_velocity[num][i],total_velocity[num][i])/np.dot(total_force_list[num][i],total_force_list[num][i]))*total_force_list[num][i])
            velocity_neb.append(part_velocity_neb)
           
        
        velocity_neb = np.array(velocity_neb)
        
        np_dot_param = 0
        if optimize_num != 0 and len(pre_total_velocity) > 1:
            for num_1, total_force in enumerate(total_force_list):
                for num_2, total_force_num in enumerate(total_force):
                    np_dot_param += (np.dot(pre_total_velocity[num_1][num_2] ,total_force_num.T))
            print(np_dot_param)
        else:
            pass
        if optimize_num > 0 and np_dot_param > 0 and len(pre_total_velocity) > 1:
            if n_reset > self.FIRE_N_accelerate:
                dt = min(dt*self.FIRE_f_inc, self.FIRE_dt_max)
                a = a*self.FIRE_N_accelerate
            n_reset += 1
        else:
            velocity_neb = velocity_neb*0
            a = self.FIRE_a_start
            dt = dt*self.FIRE_f_decelerate
            n_reset = 0
        total_velocity = velocity_neb + dt*(total_force_list)
        if optimize_num != 0 and len(pre_total_velocity) > 1:
            total_delta = dt*(total_velocity+pre_total_velocity)
        else:
            total_delta = dt*(total_velocity)
       
        total_delta_average = np.nanmean(total_delta)
        print("total_delta_average:",str(total_delta_average))
        
        #---------------------
        if self.fix_init_edge:
            move_vector = [total_delta[0]*0.0]
        else:
            move_vector = [total_delta[0]]
        trust_radii_1_list = []
        trust_radii_2_list = []
        
        for i in range(1, len(total_delta)-1):
            #total_delta[i] *= (abs(cos_list[i]) ** 0.1 + 0.1)
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            
            trust_radii_1_list.append(str(trust_radii_1*2))
            trust_radii_2_list.append(str(trust_radii_2*2))
            
            normalized_vec_1 = (geometry_num_list[i-1] - geometry_num_list[i])/np.linalg.norm(geometry_num_list[i-1] - geometry_num_list[i])
            normalized_vec_2 = (geometry_num_list[i+1] - geometry_num_list[i])/np.linalg.norm(geometry_num_list[i+1] - geometry_num_list[i])
            normalized_delta =  total_delta[i] / np.linalg.norm(total_delta[i])
            
            cos_1 = np.sum(normalized_vec_1 * normalized_delta) 
            cos_2 = np.sum(normalized_vec_2 * normalized_delta)
            print("DEBUG:  vector (cos_1, cos_2)", cos_1, cos_2)
            if (cos_1 > 0 and cos_2 > 0) or (cos_1 < 0 and cos_2 < 0):
                if np.linalg.norm(total_delta[i]) > trust_radii_1 and cos_1 > 0:
                    move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
                    print("DEBUG: TR radii 1 (considered cos_1)")
                elif np.linalg.norm(total_delta[i]) > trust_radii_2 and cos_2 > 0:
                    move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
                    print("DEBUG: TR radii 2 (considered cos_2)")
                else:
                    move_vector.append(total_delta[i])
            else:
                if np.linalg.norm(total_delta[i]) > trust_radii_1:
                    move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
                    print("DEBUG: TR radii 1")
                elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                    move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
                    print("DEBUG: TR radii 2")
                else:
                    move_vector.append(total_delta[i])      
                    
            
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_1.csv", "a") as f:
            f.write(",".join(trust_radii_1_list)+"\n")
        
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_2.csv", "a") as f:
            f.write(",".join(trust_radii_2_list)+"\n")
        
        if self.fix_end_edge:
            move_vector.append(total_delta[-1]*0.0)
        else:
            move_vector.append(total_delta[-1])
        #--------------------
        
        
        new_geometory = (geometry_num_list + move_vector)*self.bohr2angstroms
         
        return new_geometory, dt, n_reset, a

    def SD_calc(self, geometry_num_list, total_force_list):
        total_delta = []
        delta = 0.5
        for i in range(len(total_force_list)):
            total_delta.append(delta*total_force_list[i])

        #---------------------
        if self.fix_init_edge:
            move_vector = [total_delta[0]*0.0]
        else:
            move_vector = [total_delta[0]]
        for i in range(1, len(total_delta)-1):
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            if np.linalg.norm(total_delta[i]) > trust_radii_1:
                move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
            elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
            else:
                move_vector.append(total_delta[i])
        if self.fix_end_edge:
            move_vector.append(total_delta[-1]*0.0)
        else:
            move_vector.append(total_delta[-1])
        #--------------------
        new_geometory = (geometry_num_list + move_vector)*self.bohr2angstroms

        return new_geometory
    
    def FSB_quasi_newton_calc(self, geom_num_list, pre_geom, g, pre_g, hessian, biased_energy_list, pre_biased_energy_list):
        print("Quasi-Newton method")
        total_delta = []
        for i in range(len(geom_num_list)):

            delta_grad = (g[i] - pre_g[i]).reshape(len(geom_num_list[i])*3, 1)
            displacement = (geom_num_list[i] - pre_geom[i]).reshape(len(geom_num_list[i])*3, 1)
            
            delta_hess = self.FSB_hessian_update(hessian[i], displacement, delta_grad, geom_num_list[i]) 
            hessian[i] += delta_hess
            DELTA_for_QNM = 0.03
            matrix_for_RFO = np.append(hessian[i], g[i].reshape(len(geom_num_list[i])*3, 1), axis=1)
            tmp = np.array([np.append(g[i].reshape(1, len(geom_num_list[i])*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
            RFO_eigenvalue = np.sort(RFO_eigenvalue)
            lambda_for_calc = float(RFO_eigenvalue[0])
            print("# NODE",i," LAMBDA: ", lambda_for_calc)
          
            if biased_energy_list[i] < pre_biased_energy_list[i] + np.dot(pre_g[i].reshape(1, len(geom_num_list[i])*3), displacement.reshape(len(geom_num_list[i])*3, 1)):
                
                delta = (DELTA_for_QNM*np.linalg.solve((hessian[i] -0.05*lambda_for_calc*(np.eye(len(geom_num_list[i])*3)) ), g[i].reshape(len(geom_num_list[i])*3, 1))).reshape(len(geom_num_list[i]), 3)
            
            else:
                
                print("#NODE", i," linesearching...")
                alpha = np.abs(np.dot(g[i].reshape(1, len(geom_num_list[i])*3), displacement) / (np.dot(displacement.T, displacement) + 1e-8))
                cos = np.sum(displacement.reshape(len(geom_num_list[i]), 3) * g[i]) / (np.linalg.norm(displacement) * np.linalg.norm(g[i]) + 1e-8)
                print("cos = ", cos)
                print("alpha =", alpha)
                delta = -1 * (abs(cos) * alpha) * displacement.reshape(len(geom_num_list[i]), 3)
            
            total_delta.append(delta)
        #---------------------
        move_vector = [total_delta[0]]
        trust_radii_1_list = []
        trust_radii_2_list = []
        
        for i in range(1, len(total_delta)-1):
            #total_delta[i] *= (abs(cos_list[i]) ** 0.1 + 0.1)
            trust_radii_1 = np.linalg.norm(geom_num_list[i] - geom_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geom_num_list[i] - geom_num_list[i+1]) / 2.0
            
            trust_radii_1_list.append(str(trust_radii_1*2))
            trust_radii_2_list.append(str(trust_radii_2*2))
            
            
            if np.linalg.norm(total_delta[i]) > trust_radii_1:
                move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
            elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
            else:
                move_vector_delta = min(0.05, np.linalg.norm(move_vector))
                move_vector.append(move_vector_delta*total_delta[i]/np.linalg.norm(total_delta[i]))
            
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_1.csv", "a") as f:
            f.write(",".join(trust_radii_1_list)+"\n")
        
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_2.csv", "a") as f:
            f.write(",".join(trust_radii_2_list)+"\n")
        
        move_vector.append(total_delta[-1])
        #--------------------
        
        new_geometory = (geom_num_list + move_vector)*self.bohr2angstroms
        return new_geometory
        
    def FSB_hessian_update(self, hess, displacement, delta_grad, geom_num_list):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / (np.dot(A.T, displacement) + 1e-8) 
        delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / (np.dot(displacement.T, delta_grad) + 1e-8))  - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ (np.dot(np.dot(displacement.T, hess), displacement) + 1e-8)) 
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / (np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement) + 1e-8)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, geom_num_list)
        return delta_hess

    def adaptic_method(self, energy_list, gradient_list, new_geometory, pre_total_velocity, file_directory, electric_charge_and_multiplicity, element_list):
        print("ANEB (Adaptic NEB)")#J. Chem. Phys. 117, 4651–4658 (2002) https://doi.org/10.1063/1.1495401
        image_num = 1
        part_num = 2
        for adaptic_num in range(self.ANEB_num):
            print("Adaptic NEB cycle:", adaptic_num)
            idx_max_ene = np.argmax(energy_list)
            
            if idx_max_ene == 0 or idx_max_ene == len(energy_list)-1:
                print("maximun energy is not found. exit...")
                break
            node_num = len(energy_list)
            print(max(idx_max_ene-image_num, 0), min(idx_max_ene+1+image_num, node_num-1))
            energy_list = copy.copy(energy_list[max(idx_max_ene-image_num, 0):min(idx_max_ene+1+image_num, node_num)])
            new_geometory = copy.copy(new_geometory[max(idx_max_ene-image_num, 0):min(idx_max_ene+1+image_num, node_num)])
           
            geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, self.NEB_NUM*(adaptic_num+1))
            geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list(file_directory, part_num)
            new_geometory = []
            for geom in geometry_list:
                tmp_list = []
                for g in geom[1:]:
                    tmp_list.append(g[1:4])
                new_geometory.append(tmp_list)
            new_geometory = np.array(new_geometory, dtype="float64")
           
            
            geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, self.NEB_NUM*(adaptic_num+1))
            pre_total_velocity = [[[]]]
            force_data = force_data_parser(self.args)
            #prepare for FIRE method 
            dt = 0.5
            n_reset = 0
            a = self.FIRE_a_start
            dummy_hess = np.eye(len(element_list*3))
            #prepare for quasi-Newton method
            if self.QUASI_NEWTOM_METHOD:
                hessian_list = np.array([np.eye(len(element_list*3)) for i in range(len(geometry_list))], dtype="float64")
            
            
            if self.args.usextb == "None":
                pass
            else:
                element_number_list = []
                for elem in element_list:
                    element_number_list.append(element_number(elem))
                element_number_list = np.array(element_number_list, dtype="int")
            exit_flag = False
            with open(self.NEB_FOLDER_DIRECTORY+"input.txt", "w") as f:
                f.write(str(vars(self.args)))
            
            init_num = self.NEB_NUM*(adaptic_num+1)+1
            for optimize_num in range(self.NEB_NUM*(adaptic_num+1)+1, self.NEB_NUM*(adaptic_num+2)+1):
                
                exit_file_detect = os.path.exists(self.NEB_FOLDER_DIRECTORY+"end.txt")

                if exit_file_detect:
                    if psi4:
                        psi4.core.clean()
                    break
                print("\n\n\nNEB:   "+str(optimize_num)+" ITR. \n\n\n")
                self.xyz_file_make(file_directory)
                #------------------
                if self.args.usextb == "None":
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory,optimize_num, pre_total_velocity)
                else:
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list, electric_charge_and_multiplicity)
                
                biased_energy_list = []
                biased_gradient_list = []
                for i in range(len(energy_list)):
                    _, B_e, B_g, _ = BiasPotentialCalculation(dummy_hess, -1).main(energy_list[i], gradient_list[i], geometry_num_list[i], element_list, force_data)
                    biased_energy_list.append(B_e)
                    biased_gradient_list.append(B_g)
                biased_energy_list = np.array(biased_energy_list ,dtype="float64")
                biased_gradient_list = np.array(biased_gradient_list ,dtype="float64")
                #------------------
                if self.om:
                    total_force = self.OM_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list)
                elif self.lup:
                    total_force = self.LUP_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
                elif self.dneb:
                    total_force = self.DNEB_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
                elif self.nesb:
                    total_force = self.NESB_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
                else:
                    total_force = self.NEB_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
                #------------------
                cos_list = []
                for i in range(len(total_force)):
                    cos = np.sum(total_force[i]*biased_gradient_list[i])/(np.linalg.norm(total_force[i])*np.linalg.norm(biased_gradient_list[i]))
                    cos_list.append(cos)
                
                self.sinple_plot([x for x in range(len(total_force))], cos_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="cosθ", name="orthogonality")
                
                #------------------
                if self.QUASI_NEWTOM_METHOD and optimize_num > init_num:
                    new_geometory = self.FSB_quasi_newton_calc(geometry_num_list, pre_geom, total_force, pre_total_force, hessian_list, biased_energy_list, pre_biased_energy_list)
                
                
                elif optimize_num < self.sd:
                    total_velocity = self.force2velocity(total_force, element_list)
                    new_geometory, dt, n_reset, a = self.FIRE_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a, cos_list)
                    
                else:
                    new_geometory = self.SD_calc(geometry_num_list, total_force)
                #------------------
                pre_geom = geometry_num_list
                geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
                file_directory = self.make_psi4_input_file(geometry_list, optimize_num+1)
                pre_total_force = total_force
                pre_total_velocity = total_velocity
                pre_biased_energy_list = biased_energy_list
                #------------------
                with open(self.NEB_FOLDER_DIRECTORY+"energy_plot.csv", "a") as f:
                    f.write(",".join(list(map(str,biased_energy_list.tolist())))+"\n")
                    
            if exit_file_detect:
                break
        return


    def run(self):
        
        geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list(self.start_folder, self.partition)
        self.element_list = element_list
        file_directory = self.make_psi4_input_file(geometry_list,0)
        pre_total_velocity = [[[]]]
        force_data = force_data_parser(self.args)
        #prepare for FIRE method 
        dt = 0.5
        n_reset = 0
        a = self.FIRE_a_start
        dummy_hess = np.eye(len(element_list*3))
        if self.args.usextb == "None":
            pass
        else:
            element_number_list = []
            for elem in element_list:
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list, dtype="int")
        exit_flag = False
        with open(self.NEB_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        #prepare for quasi-Newton method
        if self.QUASI_NEWTOM_METHOD:
            hessian_list = np.array([np.eye(len(element_list*3)) for i in range(len(geometry_list))], dtype="float64")
        
        min_biased_energy_list = []
        for optimize_num in range(self.NEB_NUM):
            
            exit_file_detect = os.path.exists(self.NEB_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                if psi4:
                    psi4.core.clean()
                break
            print("\n\n\nNEB:   "+str(optimize_num)+" ITR. \n\n\n")
            self.xyz_file_make(file_directory)
            #------------------
            if self.args.usextb == "None":
                energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory,optimize_num, pre_total_velocity)
            else:
                energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list, electric_charge_and_multiplicity)
            
            biased_energy_list = []
            biased_gradient_list = []
            for i in range(len(energy_list)):
                _, B_e, B_g, _ = BiasPotentialCalculation(dummy_hess, -1).main(energy_list[i], gradient_list[i], geometry_num_list[i], element_list, force_data)
                biased_energy_list.append(B_e)
                biased_gradient_list.append(B_g)
            biased_energy_list = np.array(biased_energy_list ,dtype="float64")
            biased_gradient_list = np.array(biased_gradient_list ,dtype="float64")
            
            #------------------
            if self.om:
                total_force = self.OM_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list)
            elif self.lup:
                total_force = self.LUP_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
            elif self.dneb:
                total_force = self.DNEB_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
            elif self.nesb:
                total_force = self.NESB_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
            else:
                total_force = self.NEB_calc(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num)
            #------------------
            cos_list = []
            for i in range(len(total_force)):
                cos = np.sum(total_force[i]*biased_gradient_list[i])/(np.linalg.norm(total_force[i])*np.linalg.norm(biased_gradient_list[i]))
                cos_list.append(cos)
            
            self.sinple_plot([x for x in range(len(total_force))], cos_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="cosθ", name="orthogonality")
            
            #------------------
            if self.QUASI_NEWTOM_METHOD and optimize_num > 0:
                new_geometory = self.FSB_quasi_newton_calc(geometry_num_list, pre_geom, total_force, pre_total_force, hessian_list, biased_energy_list, pre_biased_energy_list)
                
            elif optimize_num < self.sd:
                total_velocity = self.force2velocity(total_force, element_list)
                new_geometory, dt, n_reset, a = self.FIRE_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a, cos_list)
                
            else:
                new_geometory = self.SD_calc(geometry_num_list, total_force)
            
            #------------------
            pre_geom = geometry_num_list
            geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, optimize_num+1)
            pre_total_force = total_force
            pre_total_velocity = total_velocity
            pre_biased_energy_list = biased_energy_list
            #------------------
            with open(self.NEB_FOLDER_DIRECTORY+"energy_plot.csv", "a") as f:
                f.write(",".join(list(map(str,biased_energy_list.tolist())))+"\n")
            
            #------------------
    

        
        print("\n\n\nNEB: final\n\n\n")
        self.xyz_file_make(file_directory) 
        if self.args.usextb == "None":
            energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory, optimize_num, pre_total_velocity)
        else:
            energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list, electric_charge_and_multiplicity)
            
        pre_total_velocity = np.array(total_velocity, dtype="float64")
        geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
        energy_list = energy_list
        
        if self.ANEB_num > 0:
            #Adaptic NEB
            self.adaptic_method(energy_list, gradient_list, new_geometory, pre_total_velocity, file_directory, electric_charge_and_multiplicity, element_list)
        
        
        print("Complete...")
        return


