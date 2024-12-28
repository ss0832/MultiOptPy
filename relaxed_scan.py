import sys
import os
import shutil
import datetime
sys.path.append('./biaspotpy')
import biaspotpy.calc_tools
from biaspotpy.parameter import UnitValueLib
import numpy as np
import biaspotpy

# When you use psi4 with this script, segmentation fault may occur. Thus, I recommend to use pyscf, tblite and so on with the following script.

def relaxed_scan_perser(parser):
    parser.add_argument("-nsample", "--number_of_samples", type=int, default=10, help='the number of sampling relaxed scan coordinates')
    parser.add_argument("-scan", "--scan_tgt", nargs="*", type=str, help='scan target (ex.) [[bond, angle, or dihedral etc.] [atom_num] [(value_1(ex. 1.0 ang.)),(value_2(ex. 1.5 ang.))] ...] ', default=None)
    return parser


def num_parse(numbers):
    sub_list = []
    
    sub_tmp_list = numbers.split(",")
    for sub in sub_tmp_list:                        
        if "-" in sub:
            for j in range(int(sub.split("-")[0]),int(sub.split("-")[1])+1):
                sub_list.append(j)
        else:
            sub_list.append(int(sub))    
    return sub_list


def save_xyz_file(coord_list, element_list, file_name, additional_name, directory):
    no_ext_file_name = os.path.splitext(file_name)[0]
    sample_file_name = no_ext_file_name+"_RelaxedScan_"+str(additional_name)+".xyz"
    with open(directory + sample_file_name, 'w') as f:
        f.write(str(len(coord_list))+"\n")
        f.write("RelaxedScan_"+str(additional_name)+"\n")
        for i in range(len(coord_list)):
            f.write(element_list[i]+" "+str(coord_list[i][0])+" "+str(coord_list[i][1])+" "+str(coord_list[i][2])+"\n")
    
    return sample_file_name

def make_scan_tgt(scan_list):
    scan_tgt_list = []
    atom_num_list = []
    scan_range_list = []
    for i in range(int(len(scan_list)/3)):
        scan = scan_list[i*3:i*3+3]
        scan_type = str(scan[0])
        atom_num_list.append(scan[1])
        tmp_scan_tgt = scan[2].split(",")
        scan_tgt_list.append(scan_type)
        scan_range_list.append([float(tmp_scan_tgt[0]), float(tmp_scan_tgt[1])])
    return scan_tgt_list, atom_num_list, scan_range_list
   

if __name__ == '__main__':
    parser = biaspotpy.interface.init_parser()
    parser = relaxed_scan_perser(parser)
    args = biaspotpy.interface.optimizeparser(parser)
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    directory = os.path.splitext(args.INPUT)[0]+"_RScan_"+date+"/"
    os.makedirs(directory, exist_ok=True)   
    ene_profile_filepath = directory + os.path.splitext(args.INPUT)[0]+"_scan_energy_profile_"+date+".csv"
    
    scan_part = int(args.number_of_samples)
    scan_tgt_list, atom_num_list, scan_range_list = make_scan_tgt(args.scan_tgt)
    
    with open(args.INPUT, 'r') as f:
        words = f.read().splitlines()
    
  
    print("Start relaxed scan .... ")
    print("Scan target: ", scan_tgt_list, atom_num_list)
    print("Scan range: ", scan_range_list)
    scan_job_list = np.array([np.linspace(start, end, scan_part) for start, end in scan_range_list]).T
    
  
    original_input_file = args.INPUT
    original_args = args

    with open(ene_profile_filepath, "w") as f:
        f.write(str(",".join(scan_tgt_list))+",energy,bias_energy\n")
        tmp_atom_num_list = [str(i).replace(",", "_") for i in atom_num_list]
        f.write(str(",".join(tmp_atom_num_list))+",(hartree),(hartree)\n")


    shutil.copyfile(original_input_file, directory + os.path.splitext(original_input_file)[0]+"_RelaxedScan_0.xyz")
    shutil.copyfile(original_input_file, os.path.splitext(original_input_file)[0]+"_RelaxedScan_0.xyz")
    args.INPUT = os.path.splitext(original_input_file)[0]+"_RelaxedScan_0.xyz"

    with open(directory + "input.txt", 'w') as f:
        f.write(str(vars(args)))

    for i in range(scan_part):
        print("scan type(s)  : ", scan_tgt_list)
        print("scan atom(s)  : ", atom_num_list)
        print("scan value(s) : ", scan_job_list[i])
        args.projection_constrain = []
        for scan_tgt, atom_num, scan_value in zip(scan_tgt_list, atom_num_list, scan_job_list[i]):
            args.projection_constrain.extend(["manual", scan_tgt, atom_num, str(scan_value)])
            
        bpa = biaspotpy.optimization.Optimize(args)
        bpa.run()
        
        opted_geometry = bpa.final_geometry * UnitValueLib().bohr2angstroms # get optimized geometry
        element_list = bpa.element_list
        energy = bpa.final_energy
        bias_energy = bpa.final_bias_energy
        
        next_input_file_for_save = directory + save_xyz_file(opted_geometry, element_list, original_input_file, i+1, directory)
        next_input_file = save_xyz_file(opted_geometry, element_list, original_input_file, i+1, "")
        args = original_args
        args.INPUT = next_input_file
        print("Done.")
        
        with open(ene_profile_filepath, "a") as f:
            f.write(str(",".join(list(map(str, scan_job_list[i].tolist())))) + "," + str(energy) + "," + str(bias_energy) + "\n")
        del bpa
    
    print("Relaxed scan done ...")