import os
import numpy as np
import sys
import glob
import time
import matplotlib.pyplot as plt
import random
import copy


from constraint_condition import ProjectOutConstrain
try:
    import psi4
except:
    psi4 = None
    print("You can't use psi4.")
try:
    from tblite.interface import Calculator
except:
    print("You can't use extended tight binding method.")
try:
    import pyscf
    from pyscf import gto, scf, dft, tddft, tdscf
    from pyscf.hessian import thermo
except:
    print("You can't use pyscf.")
#reference about LUP method:J. Chem. Phys. 94, 751–760 (1991) https://doi.org/10.1063/1.460343

from interface import force_data_parser
from parameter import element_number
from potential import BiasPotentialCalculation
from pathopt_bneb_force import CaluculationBNEB
from pathopt_dneb_force import CaluculationDNEB
from pathopt_nesb_force import CaluculationNESB
from pathopt_lup_force import CaluculationLUP
from pathopt_om_force import CaluculationOM
from calc_tools import Calculationtools
from idpp import IDPP
from Optimizer.rfo import RationalFunctionOptimization 

color_list = ["b"] #use for matplotlib


class NEB:
    def __init__(self, args):
    
        self.basic_set_and_function = args.functional+"/"+args.basisset
        self.FUNCTIONAL = args.functional
        
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        
        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(args.basisset) # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            else:
                self.SUB_BASIS_SET = { "default" : args.basisset}
            
        else:#psi4
            self.SUB_BASIS_SET = args.basisset # 
            
            
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET +="\nassign "+str(args.basisset)+"\n" # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
        
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model  
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.NEB_NUM = args.NSTEP
        self.partition = args.partition
        #please input psi4 inputfile.
        self.pyscf = args.pyscf
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
        self.bneb = args.BNEB
        self.IDPP_flag = args.use_image_dependent_pair_potential
        self.align_distances = args.align_distances
        self.excited_state = args.excited_state
        self.FC_COUNT = args.calc_exact_hess
        self.usextb = args.usextb
        self.sd = args.steepest_descent
        self.unrestrict = args.unrestrict
        if args.usextb == "None":
            self.NEB_FOLDER_DIRECTORY = args.INPUT+"_NEB_"+self.basic_set_and_function.replace("/","_")+"_"+str(time.time()).replace(".","_")+"/"
        else:
            self.NEB_FOLDER_DIRECTORY = args.INPUT+"_NEB_"+self.usextb+"_"+str(time.time()).replace(".","_")+"/"
        self.args = args
        os.mkdir(self.NEB_FOLDER_DIRECTORY)
       
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
        start_file_list = glob.glob(start_folder + "/*_[0-9].xyz") + glob.glob(start_folder + "/*_[0-9][0-9].xyz") + glob.glob(start_folder + "/*_[0-9][0-9][0-9].xyz") + glob.glob(start_folder + "/*_[0-9][0-9][0-9][0-9].xyz")
        loaded_geometry_list = []

        for start_file in start_file_list:
            with open(start_file, "r") as f:
                lines = f.read().splitlines()
            tmp_data = []
            for line in lines:
                tmp_data.append(line.split())
            loaded_geometry_list.append(tmp_data)
       
        electric_charge_and_multiplicity = loaded_geometry_list[0][0]
        element_list = [row[0] for row in loaded_geometry_list[0][1:]]
        
        loaded_geometry_num_list = [[list(map(float, row[1:4])) for row in geometry[1:]] for geometry in loaded_geometry_list]

        geometry_list = [loaded_geometry_list[0]] 

        tmp_data = [loaded_geometry_num_list[0]]
        
        
        for k in range(len(loaded_geometry_list) - 1):
            delta_num_geom = (np.array(loaded_geometry_num_list[k + 1], dtype="float64") - 
                            np.array(loaded_geometry_num_list[k], dtype="float64")) / (partition_function + 1)
            
            for i in range(partition_function + 1):
                frame_geom = np.array(loaded_geometry_num_list[k], dtype="float64") + delta_num_geom * i
                tmp_data.append(frame_geom)
                
        tmp_data.append(loaded_geometry_num_list[-1])
        tmp_data = np.array(tmp_data, dtype="float64")
        
        
        if self.IDPP_flag:
            IDPP_obj = IDPP()
            tmp_data = IDPP_obj.opt_path(tmp_data)
  
        if self.align_distances:
            tmp_data = distribute_geometry(tmp_data)
        
        for data in tmp_data:
            geometry_list.append([electric_charge_and_multiplicity] + [[element_list[num]] + list(map(str, geometry)) for num, geometry in enumerate(data)])        
        
        
        print("\n geometry data are loaded. \n")
        return geometry_list, element_list, electric_charge_and_multiplicity

    def make_geometry_list_2(self, new_geometry, element_list, electric_charge_and_multiplicity):
        new_geometry = new_geometry.tolist()
        #print(new_geometry)
        geometry_list = []
        for geometries in new_geometry:
            new_data = [electric_charge_and_multiplicity]
            for num, geometry in enumerate(geometries):
                geometory = list(map(str, geometry))
                geometory = [element_list[num]] + geometory
                new_data.append(geometory)
            
            geometry_list.append(new_data)
        return geometry_list

    def make_psi4_input_file(self, geometry_list, optimize_num):
        file_directory = self.NEB_FOLDER_DIRECTORY+"path_ITR_"+str(optimize_num)+"_"+str(self.start_folder)
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
        fig.savefig(self.NEB_FOLDER_DIRECTORY+"Plot_"+name+"_"+str(optimize_num)+".png", format="png", dpi=200)
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
        
        hess_count = 0
        
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
                
                
                if self.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                
                elif optimize_num % self.FC_COUNT == 0:
                    """exact hessian"""
                    _, wfn = psi4.frequencies(self.FUNCTIONAL, return_wfn=True, ref_gradient=wfn.gradient())
                    exact_hess = np.array(wfn.hessian())
                    freqs = np.array(wfn.frequencies())
                    print("frequencies: \n",freqs)
                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, self.element_list, input_data_for_display)
                    np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".npy", exact_hess)
                    with open(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".csv", "w") as f:
                        f.write("frequency,"+",".join(map(str, freqs))+"\n")
                
                hess_count += 1    


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
                
                
            psi4.core.clean()
        print("data sampling completed...")


        try:
            tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
            self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
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
    
    def pyscf_calculation(self, file_directory, optimize_num, pre_total_velocity, electric_charge_and_multiplicity):
        #execute extended tight binding method calclation.
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        finish_frag = False
        
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
        
        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
            
                print("\n",input_file,"\n")

                with open(input_file, "r") as f:
                    words = f.readlines()
                input_data_for_display = []
                for word in words[1:]:
                    input_data_for_display.append(np.array(word.split()[1:4], dtype="float64")/self.bohr2angstroms)
                input_data_for_display = np.array(input_data_for_display, dtype="float64")
                
                print("\n",input_file,"\n")
                mol = pyscf.gto.M(atom = words[1:],
                                  charge = int(electric_charge_and_multiplicity[0]),
                                  spin = int(electric_charge_and_multiplicity[1])-1,
                                  basis = self.SUB_BASIS_SET,
                                  max_memory = float(self.SET_MEMORY.replace("GB","")) * 1024, #SET_MEMORY unit is GB
                                  verbose=4)
                if self.excited_state  == 0:
                    if self.FUNCTIONAL == "hf" or self.FUNCTIONAL == "HF":
                        if int(electric_charge_and_multiplicity[1])-1 > 0 or self.unrestrict:
                            mf = mol.UHF().density_fit()
                        else:
                            mf = mol.RHF().density_fit()
                    else:
                        if int(electric_charge_and_multiplicity[1])-1 > 0 or self.unrestrict:
                            mf = mol.UKS().x2c().density_fit()
                        else:
                            mf = mol.RKS().density_fit()
                        mf.xc = self.FUNCTIONAL
                    g = mf.run().nuc_grad_method().kernel()
                    e = float(vars(mf)["e_tot"])
                else:
                    if self.FUNCTIONAL == "hf" or self.FUNCTIONAL == "HF":
                        if int(electric_charge_and_multiplicity[1])-1 > 0 or self.unrestrict:
                            mf = mol.UHF().density_fit().run()
                        else:
                            mf = mol.RHF().density_fit().run()
                    else:
                        if int(electric_charge_and_multiplicity[1])-1 > 0 or self.unrestrict:
                            mf = mol.UKS().x2c().density_fit().run()
                            mf.xc = self.FUNCTIONAL
                        else:
                            mf = mol.RKS().density_fit().run()
                            mf.xc = self.FUNCTIONAL
                    ground_e = float(vars(mf)["e_tot"])
                    mf = tdscf.TDA(mf)
                    g = mf.run().nuc_grad_method().kernel(state=self.excited_state)
                    e = vars(mf)["e"][self.excited_state-1]
                    e += ground_e

                g = np.array(g, dtype = "float64")
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))#RMS
                geometry_num_list.append(input_data_for_display)
                num_list.append(num)
                
                
                if self.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                
                elif optimize_num % self.FC_COUNT == 0:
                    """exact hessian"""
                    exact_hess = mf.Hessian().kernel()
                    freqs = thermo.harmonic_analysis(mf.mol, exact_hess)
                    exact_hess = exact_hess.transpose(0,2,1,3).reshape(len(input_data_for_display)*3, len(input_data_for_display)*3)
                    print("frequencies: \n",freqs["freq_wavenumber"])
                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(exact_hess, self.element_list, input_data_for_display)
                    exact_hess = np.eye(len(input_data_for_display)*3)
                    np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".npy", exact_hess)
                    with open(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".csv", "w") as f:
                        f.write("frequency,"+",".join(map(str, freqs["freq_wavenumber"]))+"\n")
                hess_count += 1
                
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        try:
            tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
            self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
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
                if not self.cpcm_solv_model is None:        
                    print("Apply CPCM solvation model")
                    calc.add("cpcm-solvation", self.cpcm_solv_model)
                if not self.alpb_solv_model is None:
                    print("Apply ALPB solvation model")
                    calc.add("alpb-solvation", self.alpb_solv_model)
                            
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
            tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
            self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
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
       
        for m, file in enumerate(file_list):
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
    
        #--------------------
        move_vector = self.TR_calc(geometry_num_list, total_force_list, total_delta)
        
        new_geometry = (geometry_num_list + move_vector)*self.bohr2angstroms
         
        return new_geometry, dt, n_reset, a

    def TR_calc(self, geometry_num_list, total_force_list, total_delta):
        if self.fix_init_edge:
            move_vector = [total_delta[0]*0.0]
        else:
            move_vector = [total_force_list[0]*0.1]
        trust_radii_1_list = []
        trust_radii_2_list = []
        
        for i in range(1, len(total_delta)-1):
            #total_delta[i] *= (abs(cos_list[i]) ** 0.1 + 0.1)
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            
            trust_radii_1_list.append(str(trust_radii_1*2))
            trust_radii_2_list.append(str(trust_radii_2*2))
            
            normalized_vec_1 = (geometry_num_list[i-1] - geometry_num_list[i])/(np.linalg.norm(geometry_num_list[i-1] - geometry_num_list[i]) + 1e-15)
            normalized_vec_2 = (geometry_num_list[i+1] - geometry_num_list[i])/(np.linalg.norm(geometry_num_list[i+1] - geometry_num_list[i]) + 1e-15)
            normalized_delta =  total_delta[i] / np.linalg.norm(total_delta[i])
            
            cos_1 = np.sum(normalized_vec_1 * normalized_delta) 
            cos_2 = np.sum(normalized_vec_2 * normalized_delta)
            print("DEBUG:  vector (cos_1, cos_2)", cos_1, cos_2)
            force_move_vec_cos = np.sum(total_force_list[i] * total_delta[i]) / (np.linalg.norm(total_force_list[i]) * np.linalg.norm(total_delta[i])) 
            
            if force_move_vec_cos >= 0: #Projected velocity-verlet algorithm
                if (cos_1 > 0 and cos_2 < 0) or (cos_1 < 0 and cos_2 > 0):
                    if np.linalg.norm(total_delta[i]) > trust_radii_1 and cos_1 > 0:
                        move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
                        print("DEBUG: TR radii 1 (considered cos_1)")
                    elif np.linalg.norm(total_delta[i]) > trust_radii_2 and cos_2 > 0:
                        move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
                        print("DEBUG: TR radii 2 (considered cos_2)")
                    else:
                        move_vector.append(total_delta[i])
                        print("DEBUG: no TR")
                elif (cos_1 < 0 and cos_2 < 0):
                    move_vector.append(total_delta[i])
                    print("DEBUG: no TR")
                else:
                    if np.linalg.norm(total_delta[i]) > trust_radii_1:
                        move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
                        print("DEBUG: TR radii 1")
                    elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                        move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
                        print("DEBUG: TR radii 2")
                    else:
                        move_vector.append(total_delta[i])      
            else:
                print("zero move vec (Projected velocity-verlet algorithm)")
                move_vector.append(total_delta[i] * 0.0) 
            
            print("---")
            
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_1.csv", "a") as f:
            f.write(",".join(trust_radii_1_list)+"\n")
        
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_2.csv", "a") as f:
            f.write(",".join(trust_radii_2_list)+"\n")
        
        if self.fix_end_edge:
            move_vector.append(total_force_list[-1]*0.0)
        else:
            move_vector.append(total_force_list[-1]*0.1)
        
        return move_vector        
        
    def RFO_calc(self, geometry_num_list, total_force_list, prev_geometry_num_list, prev_total_force_list, biased_gradient_list, optimize_num, STRING_FORCE_CALC, biased_energy_list):
        natoms = len(geometry_num_list[0])
        if optimize_num % self.FC_COUNT == 0:
            for num in range(1, len(total_force_list)-1):
                hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy")
                proj_hess = STRING_FORCE_CALC.projection_hessian(geometry_num_list[num-1], geometry_num_list[num], geometry_num_list[num+1], biased_gradient_list[num-1:num+2], hess, biased_energy_list[num-1:num+2])
                np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy", proj_hess)
        
        total_delta = []
        for num, total_force in enumerate(total_force_list):
            hessian = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy")
            if num == 0 or num == len(total_force_list) - 1:
                OPT = RationalFunctionOptimization(method="rfo_fsb", saddle_order=0)
            else:
                OPT = RationalFunctionOptimization(method="rfo3_bofill", saddle_order=1)

            OPT.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
            OPT.set_hessian(hessian)
            if optimize_num == 0:
                OPT.Initialization = True
                pre_B_g = None
                pre_geom = None
            else:
                OPT.Initialization = False
                OPT.lambda_clip_flag = True
                OPT.lambda_s_scale = 0.1 * 1.0 / (1.0 + np.exp(np.linalg.norm(total_force) - 5.0))
                
                pre_B_g = -1 * prev_total_force_list[num].reshape(-1, 1)
                pre_geom = prev_geometry_num_list[num].reshape(-1, 1)        
            geom_num_list = geometry_num_list[num].reshape(-1, 1)
           
            B_g = -1 * total_force.reshape(-1, 1)
            
            move_vec = -1 * OPT.run(geom_num_list, B_g, pre_B_g, pre_geom, 0.0, 0.0, [], [], B_g, pre_B_g)
          
            total_delta.append(move_vec.reshape(-1, 3))
            hessian = OPT.get_hessian()
            np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy", hessian)
        
        move_vector_list = self.TR_calc(geometry_num_list, total_force_list, total_delta)
       
        new_geometry_list = (geometry_num_list + move_vector_list) * self.bohr2angstroms
        
        return new_geometry_list


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
        new_geometry = (geometry_num_list + move_vector)*self.bohr2angstroms
        return new_geometry
    
    def run(self):
        geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list(self.start_folder, self.partition)
        self.element_list = element_list
        
        file_directory = self.make_psi4_input_file(geometry_list, 0)
        pre_total_velocity = [[[]]]
        force_data = force_data_parser(self.args)
        if len(force_data["projection_constraint_condition_list"]) > 0:
            projection_constraint_flag = True
        else:
            projection_constraint_flag = False        

        #prepare for FIRE method 
        dt = 0.5
        n_reset = 0
        a = self.FIRE_a_start
        if self.args.usextb == "None":
            pass
        else:
            element_number_list = []
            for elem in element_list:
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list, dtype="int")
        
        with open(self.NEB_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        
        if self.om:
            STRING_FORCE_CALC = CaluculationOM(self.APPLY_CI_NEB)
        elif self.lup:
            STRING_FORCE_CALC = CaluculationLUP(self.APPLY_CI_NEB)
        elif self.dneb:
            STRING_FORCE_CALC = CaluculationDNEB(self.APPLY_CI_NEB)
        elif self.nesb:
            STRING_FORCE_CALC = CaluculationNESB(self.APPLY_CI_NEB)
        elif self.bneb:
            STRING_FORCE_CALC = CaluculationBNEB(self.APPLY_CI_NEB)
        else:
            STRING_FORCE_CALC = CaluculationBNEB(self.APPLY_CI_NEB)
        
        if len(force_data["fix_atoms"]) > 0:
            fix_atom_flag = True
        else:
            fix_atom_flag = False
        
        pre_geom = None
        pre_total_force = None
        pre_total_velocity = []
        total_velocity = []
        #------------------
        for optimize_num in range(self.NEB_NUM):
            exit_file_detect = os.path.exists(self.NEB_FOLDER_DIRECTORY+"end.txt")
            if exit_file_detect:
                if psi4:
                    psi4.core.clean()
                break
            
            print("\n\n\n NEB: ITR.  "+str(optimize_num)+"  \n\n\n")
            self.xyz_file_make(file_directory)
            #------------------
            #get energy and gradient
            if self.args.usextb == "None":
                if self.pyscf:
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.pyscf_calculation(file_directory, optimize_num, pre_total_velocity, electric_charge_and_multiplicity)
                else:
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory,optimize_num, pre_total_velocity)
            else:
                energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list, electric_charge_and_multiplicity)
            
            
            if optimize_num == 0:
                init_geometry_num_list = geometry_num_list
            
            biased_energy_list = []
            biased_gradient_list = []
            for i in range(len(energy_list)):
                _, B_e, B_g, B_hess = BiasPotentialCalculation(self.NEB_FOLDER_DIRECTORY).main(energy_list[i], gradient_list[i], geometry_num_list[i], element_list, force_data)
                if self.FC_COUNT > 0:
                    hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(i)+".npy")
                    np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(i)+".npy", B_hess + hess)
                biased_energy_list.append(B_e)
                biased_gradient_list.append(B_g)
            biased_energy_list = np.array(biased_energy_list ,dtype="float64")
            biased_gradient_list = np.array(biased_gradient_list ,dtype="float64")
            

            if projection_constraint_flag and optimize_num == 0:
                PC_list = []
                for i in range(len(energy_list)):
                    PC_list.append(ProjectOutConstrain(force_data["projection_constraint_condition_list"], force_data["projection_constraint_atoms"], force_data["projection_constraint_constant"]))
                    PC_list[i].initialize(geometry_num_list[i])

            if projection_constraint_flag:
                for i in range(len(energy_list)):
                    biased_gradient_list[i] = copy.copy(PC_list[i].calc_project_out_grad(geometry_num_list[i], biased_gradient_list[i]))            

            #------------------
            #calculate force
            total_force = STRING_FORCE_CALC.calc_force(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list)

            #------------------
            cos_list = []
            tot_force_rms_list = []
            tot_force_max_list = []
            for i in range(len(total_force)):
                cos = np.sum(total_force[i]*biased_gradient_list[i])/(np.linalg.norm(total_force[i])*np.linalg.norm(biased_gradient_list[i]))
                cos_list.append(cos)
                tot_force_rms = np.sqrt(np.mean(total_force[i]**2))
                tot_force_rms_list.append(tot_force_rms)
                tot_force_max = np.max(total_force[i])
                tot_force_max_list.append(tot_force_max)
                
            
            self.sinple_plot([x for x in range(len(total_force))], cos_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="cosθ", name="orthogonality")
            self.sinple_plot([x for x in range(len(total_force))][1:-1], tot_force_rms_list[1:-1], file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Perpendicular Gradient (RMS) [a.u.]", name="perp_rms_gradient")
            self.sinple_plot([x for x in range(len(total_force))], tot_force_max_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Perpendicular Gradient (MAX) [a.u.]", name="perp_max_gradient")
          
            #------------------
            #relax path
            if self.FC_COUNT != -1:
                new_geometry = self.RFO_calc(geometry_num_list, total_force, pre_geom, pre_total_force, biased_gradient_list, optimize_num, STRING_FORCE_CALC, biased_energy_list)
            
            elif optimize_num < self.sd:
                total_velocity = self.force2velocity(total_force, element_list)
                new_geometry, dt, n_reset, a = self.FIRE_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a, cos_list)
             
            else:
                new_geometry = self.SD_calc(geometry_num_list, total_force)
            
            #------------------
            #fix atoms
            if fix_atom_flag:
                for k in range(len(new_geometry)):
                    for j in force_data["fix_atoms"]:
                        new_geometry[k][j-1] = copy.copy(init_geometry_num_list[k][j-1]*self.bohr2angstroms)
        
            if projection_constraint_flag:
                for x in range(len(new_geometry)):
                    tmp_new_geometry = new_geometry[x] / self.bohr2angstroms
                    tmp_new_geometry = PC_list[x].adjust_init_coord(tmp_new_geometry) * self.bohr2angstroms    
                    new_geometry[x] = copy.copy(tmp_new_geometry)
            
            if not fix_atom_flag:
                for k in range(len(new_geometry)-1):
                    tmp_new_geometry, _ = Calculationtools().kabsch_algorithm(new_geometry[k], new_geometry[k+1])
                    new_geometry[k] = copy.copy(tmp_new_geometry)
            
            if self.align_distances and optimize_num > 0:
                tmp_new_geometry = distribute_geometry(np.array(new_geometry))
                for k in range(len(new_geometry)):
                    new_geometry[k] = copy.copy(tmp_new_geometry[k])
            #------------------
            pre_geom = geometry_num_list
            
            geometry_list = self.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, optimize_num+1)
            pre_total_force = total_force
            pre_total_velocity = total_velocity
            pre_biased_energy_list = biased_energy_list
            #------------------
            with open(self.NEB_FOLDER_DIRECTORY+"energy_plot.csv", "a") as f:
                f.write(",".join(list(map(str,biased_energy_list.tolist())))+"\n")
            
        self.xyz_file_make(file_directory) 
        print("Complete...")
        return


def distribute_geometry(geometry_list):
    nnode = len(geometry_list)
 
    path_length_list = [0.0]
    for i in range(nnode-1):
        path_length = path_length_list[-1] + np.linalg.norm(geometry_list[i+1] - geometry_list[i])
        path_length_list.append(path_length)
    total_length = path_length_list[-1]
    node_dist = total_length / (nnode-1)
    
    new_geometry_list = [geometry_list[0]]
    for i in range(1, nnode-1):
        dist = i * node_dist
        for j in range(nnode-1):
            if path_length_list[j] <= dist and dist <= path_length_list[j+1]:
                break
        delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
        new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
        new_geometry_list.append(new_geometry)
    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list
