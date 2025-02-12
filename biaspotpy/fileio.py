import glob
import os
import re
import numpy as np

from scipy.signal import argrelextrema
from calc_tools import calc_RMS

def save_bias_pot_info(file_path, energy, gradient, bias_pot_id):
    max_grad = np.max(np.abs(gradient))
    rms_grad = calc_RMS(gradient)
    save_path = file_path+"bias_pot_info_"+str(bias_pot_id)+".log"
    
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write("Energy, MaxGrad, RMSGrad\n")
            
    with open(save_path, "a") as f:
        f.write(str(energy)+","+str(max_grad)+","+str(rms_grad)+"\n")
    return

def save_bias_param_grad_info(file_path, gradient, bias_pot_id):
    save_path = file_path+"bias_param_grad_info_"+str(bias_pot_id)+".log"
    
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write("Gradient\n")
    with open(save_path, "a") as f:
        f.write(str(gradient)+"\n")
    return

def read_software_path(file_path="./"):
    with open(file_path+"software_path.conf", "r") as f:
        words = f.read().splitlines()
    software_path_dict = {}
    for word in words:
        tmp_split = word.split("::")
        soft_name = tmp_split[0]
        soft_path = tmp_split[1]
        software_path_dict[soft_name] = soft_path
    return software_path_dict

def xyz2list(file_path, args_electric_charge_and_multiplicity):
    pattern_xyz = r"\s*([A-Za-z]+)\s+([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)(?:\s+([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?))(?:\s+([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?))\s*"    
    pattern_cs = r"[0-9]+\s+[0-9]+\s*"
    electric_charge_and_multiplicity = None
    element_list = []
    with open(file_path, "r") as f:
        words = f.read().splitlines()
    geometry_list = []
    for word in words:
        if re.match(pattern_cs, word):
            electric_charge_and_multiplicity = word.split()
        if re.match(pattern_xyz, word):
            geometry_list.append(word.split()[1:4])
            element_list.append(word.split()[0])
    if electric_charge_and_multiplicity is None:
        electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
   
    return geometry_list, element_list, electric_charge_and_multiplicity


class FileIO:
    def __init__(self, folder_dir, file):
        self.BPA_FOLDER_DIRECTORY = folder_dir
        self.START_FILE = file
        self.NOEXT_START_FILE = os.path.splitext(os.path.basename(self.START_FILE))[0]
        self.is_save_gjf_file = True
        return
    
    def make_geometry_list(self, args_electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
        element_list = []
        start_data = []
        with open(self.START_FILE, "r") as f:
            words = f.read().splitlines()
        for word in words:
            start_data.append(word.split())
        
        if len(start_data[1]) == 2:#(charge ex. 0) (spin ex. 1)
            electric_charge_and_multiplicity = start_data[1]
        else:
            electric_charge_and_multiplicity = args_electric_charge_and_multiplicity#list
            
        for i in range(1, len(start_data)):
            if len(start_data[i]) < 4:
                continue
            element_list.append(start_data[i][0])
        geometry_list.append(start_data)
        return geometry_list, element_list, electric_charge_and_multiplicity

    def make_geometry_list_for_pyscf(self):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
 
        with open(self.START_FILE,"r") as f:
            words = f.readlines()
            
        start_data = []
        for word in words[2:]:
            start_data.append(word.split())
            
        element_list = []
            
        for i in range(len(start_data)):
            if len(start_data[i]) < 4:
                continue
            element_list.append(start_data[i][0])
                
        geometry_list.append(start_data)
        return geometry_list, element_list

    def make_geometry_list_2(self, new_geometry, element_list, electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
        
        geometry_list = []
        print("\n")
        new_data = [electric_charge_and_multiplicity]
        for num, geometry in enumerate(new_geometry):
           
            geometry = list(map(str, geometry))
            geometry = [element_list[num]] + geometry
            new_data.append(geometry)
            print(" ".join(geometry))
            
        geometry_list.append(new_data)
        print("")
        return geometry_list
        
    def make_geometry_list_2_for_pyscf(self, new_geometry, element_list):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
        print("\n")
        geometry_list = []

        new_data = []
        for num, geometry in enumerate(new_geometry):
           
            geometry = list(map(str, geometry))
            geometry = [element_list[num]] + geometry
            new_data.append(geometry)
            print(" ".join(geometry))
            
        geometry_list.append(new_data)
        print("")
        return geometry_list
        
    def make_psi4_input_file(self, geometry_list, iter):#geometry_list: ang.
        """structure updated geometry is saved."""

        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+self.NOEXT_START_FILE+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.NOEXT_START_FILE+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
      
        return file_directory

    def make_pyscf_input_file(self, geometry_list, iter):#geometry_list: ang.
        """structure updated geometry is saved."""
     
        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+self.NOEXT_START_FILE+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.NOEXT_START_FILE+"_"+str(y)+".xyz","w") as w:
                w.write(str(len(geometry))+"\n\n")
                for rows in geometry:   
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory
    
    
    def read_gjf_file(self, args_electric_charge_and_multiplicity):        
        geometry_list = []
        element_list = []
        with open(self.START_FILE, 'r') as f:
            lines = f.read().splitlines()
        read_flag = False
        pattern = '[0-9]+ [0-9]+'
        repatter = re.compile(pattern)
        for i in range(len(lines)):
            if bool(re.match(repatter, lines[i])) is True:
                electric_charge_and_multiplicity = lines[i].split()
                read_flag = True
                geometry_list.append("dummy")
                geometry_list.append(lines[i].split())
                continue
            if read_flag and len(lines[i]) == 0:
                break
            if read_flag:
                element_list.append(lines[i].split()[0])
                geometry_list.append(lines[i].split())   
        geometry_list = [geometry_list]

        return geometry_list, element_list, electric_charge_and_multiplicity
    
    
    def save_gjf_file(self, geometry_list):
   
        with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+".gjf","w") as f:
            f.write("%mem=4GB\n")
            f.write("%nprocshared=4\n")
            f.write("#p B3LYP/6-31G* opt freq\n")
            f.write("\n")
            f.write("Title Card Required\n")
            f.write("\n")
            f.write("0 1\n")# This line is required for fixing the charge and multiplicity.
            for geometry in geometry_list:
                f.write(geometry)
                f.write("\n")
            f.write("\n\n\n")
        
        return 
    
    def xyz_file_make_for_pyscf(self, name=""):
        """optimized path is saved."""
        print("\nprocessing geometry collection ...\n")
        if name == "":
            file_list = sum([sorted(glob.glob(os.path.join(self.BPA_FOLDER_DIRECTORY, f"samples_*_" + "[0-9]" * i, "*.xyz"))) for i in range(1, 7)], [])
        else:    
            file_list = sum([sorted(glob.glob(os.path.join(self.BPA_FOLDER_DIRECTORY, f"samples_*_{name}_" + "[0-9]" * i, "*.xyz"))) for i in range(1, 7)], [])
        #print(file_list,"\n")
        step_num = len(file_list)
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.read().splitlines()
                with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_collection.xyz","a") as w:
                    atom_num = len(sample)-2
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                
                for i in sample[2:]:
                    with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_collection.xyz","a") as w2:
                        if "\n" == i or "" == i:
                            continue
                        w2.write(i+"\n")
                
                if m == step_num - 1:
                    if self.is_save_gjf_file:
                        self.save_gjf_file(sample[2:])
                    
                    
                    with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_optimized.xyz","a") as w2:
                        w2.write(str(atom_num)+"\n")
                        w2.write("OptimizedStructure\n")
                        for i in sample[2:]:
                            
                            if "\n" == i or "" == i:
                                continue
                            w2.write(i+"\n")
                
        print("\ngeometry collection was completed...\n")
        return
        
    def xyz_file_make(self, name=""):
        """optimized path is saved."""
        print("\nprocessing geometry collection ...\n")
        if name == "":
            file_list = sum([sorted(glob.glob(os.path.join(self.BPA_FOLDER_DIRECTORY, f"samples_*_" + "[0-9]" * i, "*.xyz")))
                 for i in range(1, 7)], [])
        else:    
            file_list = sum([sorted(glob.glob(os.path.join(self.BPA_FOLDER_DIRECTORY, f"samples_*_{name}_" + "[0-9]" * i, "*.xyz")))
                 for i in range(1, 7)], [])
        step_num = len(file_list)
        for m, file in enumerate(file_list[1:], 1):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.read().splitlines()
            with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_collection.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            
            
            for i in sample[1:]:
                with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_collection.xyz","a") as w2:
                    if "\n" == i or "" == i:
                        continue
                    w2.write(i+"\n")
                    
            if m == step_num - 1:
                if self.is_save_gjf_file:
                    self.save_gjf_file(sample[1:])
                
                with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_optimized.xyz","w") as w3:
                    w3.write(str(atom_num)+"\n")
                    w3.write("OptimizedStructure\n")
                    for i in sample[1:]:
                        if "\n" == i or "" == i:
                            continue
                        
                        w3.write(i+"\n")
        print("\ngeometry collection was completed...\n")
        return

    def xyz_file_save_for_IRC(self, element_list, geometry_list):
        count = 0
        for geometry in geometry_list:
            with open(self.BPA_FOLDER_DIRECTORY+"IRC_path_collection.xyz","a") as w:
                atom_num = len(geometry)
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(count)+"\n")
                for i in range(len(geometry)):
                    w.write(element_list[i]+" "+str(geometry[i][0])+" "+str(geometry[i][1])+" "+str(geometry[i][2])+"\n")
            count += 1
        print("\ngeometry collection for IRC was completed...\n")
        return
    
    
    def xyz_file_make_for_DM(self, img_1="reactant", img_2="product"):
        """optimized path is saved."""
        print("\nprocessing geometry collection ...\n")
        file_list = sum(
            [sorted(glob.glob(self.BPA_FOLDER_DIRECTORY + f"samples_*_{str(img_1)}_{'[0-9]' * i}/*.xyz"))
            for i in range(1, 7)],
            []
        ) + sum(
            [sorted(glob.glob(self.BPA_FOLDER_DIRECTORY + f"samples_*_{str(img_2)}_{'[0-9]' * i}/*.xyz"))[::-1]
            for i in range(6, 0, -1)],
            []
        )
        
        for m, file in enumerate(file_list[1:-1]):
        
            with open(file,"r") as f:
                sample = f.readlines()
            with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_collection.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            del sample[0]
            for i in sample:
                with open(self.BPA_FOLDER_DIRECTORY+self.NOEXT_START_FILE+"_collection.xyz","a") as w2:
                    w2.write(i)
        print("\ngeometry collection was completed...\n")
        return
    
    def argrelextrema_txt_save(self, save_list, name, min_max):
        NUM_LIST = [i for i in range(len(save_list))]
        if min_max == "max":
            local_max_energy_list_index = argrelextrema(np.array(save_list), np.greater)
            with open(self.BPA_FOLDER_DIRECTORY+name+".txt","w") as f:
                for j in local_max_energy_list_index[0].tolist():
                    f.write(str(NUM_LIST[j])+"\n")
        elif min_max == "min":
            inverse_energy_list = (-1)*np.array(save_list, dtype="float64")
            local_min_energy_list_index = argrelextrema(np.array(inverse_energy_list), np.greater)
            with open(self.BPA_FOLDER_DIRECTORY+name+".txt","w") as f:
                for j in local_min_energy_list_index[0].tolist():
                    f.write(str(NUM_LIST[j])+"\n")
        else:
            print("error")
    
        return
 