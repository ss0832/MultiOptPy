import glob
import os

import numpy as np

from scipy.signal import argrelextrema

class FileIO:
    def __init__(self, folder_dir, file):
        self.BPA_FOLDER_DIRECTORY = folder_dir
        self.START_FILE = file
        return
    
    
    
    def xyz2oniom2dict(self, file_path):
        oniom_dict = {}
        """ ### format ###
        37
        C11N2H16O8
        C       -0.110337444      0.412195157     -2.544391560 h
        H       -0.482924196      1.278536980     -2.018150001 h
        C        0.518604003     -0.573260445     -1.868806998 h
        C        0.993027108     -1.701809996     -2.579924744 h
        O        1.562929338     -2.603445423     -1.992388061 h
        N        0.807579022     -1.769245286     -3.913764794 h
        H        1.129397061     -2.539393866     -4.407936307 h
        C        0.177142125     -0.772430742     -4.562310901 h
        O        0.016992190     -0.847041893     -5.765279485 h
        N       -0.278406741      0.305271806     -3.898123587 h
        H       -0.730869883      1.012958538     -4.383480018 h
        C        0.713473449     -0.472982388     -0.377844064 hl
        O        0.149490619      0.751951638      0.094895876
        C        0.281272877      0.946667386      1.504417917
        H        1.331208445      0.865706901      1.785998389
        C       -0.242124802      2.335691466      1.879106903
        H       -1.280286139      2.431716171      1.561106974
        C       -0.153278120      2.513685307      3.397869843
        H        0.890018674      2.467161391      3.710091455
        C       -0.942630384      1.391782695      4.079388437
        H       -1.994948611      1.468689659      3.805943805
        C       -0.392179295      0.039199533      3.619736268
        H        0.648852151     -0.053721253      3.929212221
        O       -0.475608671     -0.048880644      2.195910600
        C       -1.214159689     -1.086987873      4.249891980
        O       -0.630267327     -2.347531768      3.915197061
        O       -0.808134448      1.506520854      5.497437554
        O       -0.705810518      3.779679634      3.763947328
        O        0.549491371      3.334562627      1.232922526
        H        0.219240142     -1.312879049      0.110500267
        H        1.778844628     -0.494927250     -0.148377583
        H       -1.224021716     -0.967351966      5.333191084
        H       -2.235380396     -1.047402217      3.870682258
        H       -1.100589037     -3.108731386      4.282049742
        H       -1.133880772      2.343008112      5.856938959
        H       -0.256707089      4.535867557      3.361916849
        H        0.542496644      3.278390857      0.267565666

        """
        with open(file_path, "r") as f:
            words = f.read().splitlines()
        highlow_layer_tag_list = []
        coord_list = []
        element_list = []

        for word in words[2:]:
            word_list = word.split()
            
            elem = word_list[0]
            coord = word_list[1:4]
            if len(word_list) > 4:
                if word_list[4] == "h" or word_list[4] == "H":
                    hl_layer_tag = "H"
                elif word_list[4] == "hl" or word_list[4] == "HL" or word_list[4] == "lh" or word_list[4] == "LH":
                    hl_layer_tag = "C"#Connection between High and Low layer
                else:
                    raise "error (invalid layer specification)" 
            else:
                hl_layer_tag = "L"

            highlow_layer_tag_list.append(hl_layer_tag)
            coord_list.append(coord)
            element_list.append(elem)
    
        oniom_dict["highlow_layer_tag_list"] = highlow_layer_tag_list
        oniom_dict["coord_list"] = coord_list
        oniom_dict["element_list"] = element_list
        
        return oniom_dict
    
    def make_geometry_list(self, args_electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
 
        with open(self.START_FILE, "r") as f:
            words = f.readlines()
            
        start_data = []
        for word in words:
            start_data.append(word.split())
        
        if len(start_data[1]) == 2:#(charge ex. 0) (spin ex. 1)
            electric_charge_and_multiplicity = start_data[1]
            
        else:
            electric_charge_and_multiplicity = args_electric_charge_and_multiplicity
            
        element_list = []
            


        for i in range(2, len(start_data)):
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
        
    def xyz_file_make_for_pyscf(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")  
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
        
    def xyz_file_make(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")  
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
                    w2.write(i)
                    
            if m == step_num - 1:
                with open(self.BPA_FOLDER_DIRECTORY+os.path.basename(self.START_FILE)[:-4]+"_optimized.xyz","w") as w3:
                    w3.write(str(atom_num)+"\n")
                    w3.write("Optimized Structure\n")
                    for i in sample[1:]:
                        w3.write(i)
        print("\ngeometry collection is completed...\n")
        return
        
    def xyz_file_make_for_DM(self, img_1="reactant", img_2="product"):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_1)+"_[0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*"+str(img_1)+"_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9][0-9][0-9]/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9][0-9]/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9][0-9]/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9][0-9]/*.xyz")[::-1] + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_"+str(img_2)+"_[0-9]/*.xyz")[::-1]   
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
 