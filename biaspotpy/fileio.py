import glob
import os

import numpy as np

from scipy.signal import argrelextrema

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


class FileIO:
    def __init__(self, folder_dir, file):
        self.BPA_FOLDER_DIRECTORY = folder_dir
        self.START_FILE = file
        return
    
    
    
    def make_geometry_list(self, args_electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
 
        with open(self.START_FILE, "r") as f:
            words = f.read().splitlines()
            
        start_data = []
        for word in words:
            start_data.append(word.split())
        
        if len(start_data[1]) == 2:#(charge ex. 0) (spin ex. 1)
            electric_charge_and_multiplicity = start_data[1]
            
        else:
            electric_charge_and_multiplicity = args_electric_charge_and_multiplicity#list
            
        element_list = []
            


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
        print("\ngeometry:")
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
        print("\ngeometry:")
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
        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(os.path.basename(self.START_FILE)[:-4])+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+os.path.basename(self.START_FILE[:-4])+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory

    def make_pyscf_input_file(self, geometry_list, iter):#geometry_list: ang.
        """structure updated geometry is saved."""
        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(os.path.basename(self.START_FILE)[:-4])+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+os.path.basename(self.START_FILE)[:-4]+"_"+str(y)+".xyz","w") as w:
                w.write(str(len(geometry))+"\n\n")
                for rows in geometry:   
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory
        
    def xyz_file_make_for_pyscf(self, name=""):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        if name == "":
            file_list = sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz"))
        else:    
            file_list = sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")) 
        #print(file_list,"\n")
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
                with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_collection.xyz","a") as w:
                    atom_num = len(sample)-2
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                
                for i in sample[2:]:
                    with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_collection.xyz","a") as w2:
                        w2.write(i)
        print("\ngeometry collection is completed...\n")
        return
        
    def xyz_file_make(self, name=""):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        if name == "":
            file_list = sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz"))
        else:    
            file_list = sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+name+"_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")) 
        step_num = len(file_list)
        for m, file in enumerate(file_list[1:], 1):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
            with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_collection.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            
            
            for i in sample[1:]:
                with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_collection.xyz","a") as w2:
                    if "\n" == i:
                        continue
                    w2.write(i)
                    
            if m == step_num - 1:
                with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_optimized.xyz","w") as w3:
                    w3.write(str(atom_num)+"\n")
                    w3.write("OptimizedStructure\n")
                    for i in sample[1:]:
                        if "\n" == i:
                            continue
                        
                        w3.write(i)
        print("\ngeometry collection is completed...\n")
        return

    def xyz_file_save_for_IRC(self):
        return
    
    
    def xyz_file_make_for_DM(self, img_1="reactant", img_2="product"):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*"+str(img_1)+"_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")) + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz"))[::-1] + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9][0-9][0-9]/*.xyz"))[::-1] + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9][0-9]/*.xyz"))[::-1] + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9]/*.xyz"))[::-1] + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9]/*.xyz"))[::-1] + sorted(glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9]/*.xyz"))[::-1]   
        #print(file_list,"\n")
        
        
        for m, file in enumerate(file_list[1:-1]):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
            with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_collection.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            del sample[0]
            for i in sample:
                with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_collection.xyz","a") as w2:
                    w2.write(i)
        print("\ngeometry collection is completed...\n")
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
 