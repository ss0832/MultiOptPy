import os
import numpy as np
import sys
import glob
import time
import matplotlib.pyplot as plt
import random
import copy
import re


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
try:
    import dxtb
except:
    print("You can't use dxtb.")
    
try:
    import torch
except:
    print("You can't use pytorch.")

from interface import force_data_parser
from parameter import element_number
from potential import BiasPotentialCalculation
from pathopt_bneb_force import CaluculationBNEB, CaluculationBNEB2
from pathopt_dneb_force import CaluculationDNEB
from pathopt_nesb_force import CaluculationNESB
from pathopt_lup_force import CaluculationLUP
from pathopt_om_force import CaluculationOM
from pathopt_ewbneb_force import CaluculationEWBNEB
from calc_tools import Calculationtools
from idpp import IDPP
from Optimizer.rfo import RationalFunctionOptimization 
from interpolation import spline_interpolation
from constraint_condition import ProjectOutConstrain
from fileio import xyz2list, traj2list, FileIO
from multioptpy.Optimizer import lbfgs_neb 
#from multioptpy.Optimizer import afire_neb
#from multioptpy.Optimizer import quickmin_neb
from multioptpy.Optimizer import conjugate_gradient_neb
from multioptpy.Optimizer import trust_radius_neb 

color_list = ["b"] #use for matplotlib

class NEB:
    def __init__(self, args):
    
        self.basic_set_and_function = args.functional+"/"+args.basisset
        self.FUNCTIONAL = args.functional
        
        self.set_sub_basisset(args)
        
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
        self.dt = 0.5
        self.a = 0.10
        self.n_reset = 0
        self.FIRE_N_accelerate = 5
        self.FIRE_f_inc = 1.10
        self.FIRE_f_accelerate = 0.99
        self.FIRE_f_decelerate = 0.5
        self.FIRE_a_start = 0.1
        self.FIRE_dt_max = 1.0
        self.APPLY_CI_NEB = args.apply_CI_NEB
        self.init_input = args.INPUT
        self.om = args.OM
        self.lup = args.LUP
        self.dneb = args.DNEB
        self.nesb = args.NESB
        self.bneb = args.BNEB
        self.ewbneb = args.EWBNEB
        self.mix = args.MIX
        self.IDPP_flag = args.use_image_dependent_pair_potential
        self.align_distances = args.align_distances #integer
        self.excited_state = args.excited_state
        self.FC_COUNT = args.calc_exact_hess
        self.usextb = args.usextb
        self.usedxtb = args.usedxtb
        self.sd = args.steepest_descent
        self.unrestrict = args.unrestrict
        self.save_pict = args.save_pict
        self.climbing_image_start = args.climbing_image[0]
        self.climbing_image_interval = args.climbing_image[1]
        self.apply_convergence_criteria = args.apply_convergence_criteria
        self.node_distance = args.node_distance
        self.NEB_FOLDER_DIRECTORY = self.make_neb_work_directory(args)
        
        os.mkdir(self.NEB_FOLDER_DIRECTORY)
       
        self.set_fixed_edges(args)
            
        self.global_quasi_newton = args.global_quasi_newton
        self.force_const_for_cineb = 0.01
        if self.FC_COUNT > 0 and args.usextb != "None":
            print("Currently, you can't use exact hessian calculation with extended tight binding method.")
            self.FC_COUNT = -1
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity
        self.args = args
        
        self.cg_method = args.conjugate_gradient
        self.lbfgs_method = args.memory_limited_BFGS
        return

    def set_fixed_edges(self, args):
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

    def set_sub_basisset(self, args):
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        
        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(args.basisset)
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
            else:
                self.SUB_BASIS_SET = {"default": args.basisset}
        else:
            self.SUB_BASIS_SET = args.basisset
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET += "\nassign " + str(args.basisset) + "\n"
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign " + args.sub_basisset[2*j] + " " + args.sub_basisset[2*j+1] + "\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)


    def make_neb_work_directory(self, args):
        """Return NEB folder directory path based on user arguments."""
        
        if os.path.splitext(args.INPUT)[1] == ".xyz":
            tmp_name = os.path.splitext(args.INPUT)[0] 
        else:
            tmp_name = args.INPUT 
        
        if args.usextb == "None" and args.usedxtb == "None":
            return tmp_name + "_NEB_" + self.basic_set_and_function.replace("/", "_") + "_" + str(time.time()).replace(".", "_") + "/"
        else:
            if self.usextb != "None":
                return tmp_name + "_NEB_" + self.usextb + "_" + str(time.time()).replace(".", "_") + "/"
            else:
                return tmp_name + "_NEB_" + self.usedxtb + "_" + str(time.time()).replace(".", "_") + "/"


    def force2velocity(self, gradient_list, element_list):
        velocity_list = gradient_list
        return np.array(velocity_list, dtype="float64")

    def make_geometry_list(self, init_input, partition_function):
        if os.path.splitext(init_input)[1] == ".xyz":
            self.init_input = os.path.splitext(init_input)[0]
            xyz_flag = True
        else:
            xyz_flag = False 
            
        start_file_list = sum([sorted(glob.glob(os.path.join(init_input, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])

        loaded_geometry_list = []

        if xyz_flag:
            geometry_list, elements, electric_charge_and_multiplicity = traj2list(init_input, [self.electronic_charge, self.spin_multiplicity])
            
            element_list = elements[0]
            
            for i in range(len(geometry_list)):
                loaded_geometry_list.append([electric_charge_and_multiplicity] + [[element_list[num]] + list(map(str, geometry)) for num, geometry in enumerate(geometry_list[i])])
        
        
        else:
            for start_file in start_file_list:
                tmp_geometry_list, element_list, electric_charge_and_multiplicity = xyz2list(start_file, [self.electronic_charge, self.spin_multiplicity])
                tmp_data = [electric_charge_and_multiplicity]

                for i in range(len(tmp_geometry_list)):
                    tmp_data.append([element_list[i]] + list(map(str, tmp_geometry_list[i])))
                loaded_geometry_list.append(tmp_data)
            
        
        electric_charge_and_multiplicity = loaded_geometry_list[0][0]
        element_list = [row[0] for row in loaded_geometry_list[0][1:]]
        
        loaded_geometry_num_list = [[list(map(float, row[1:4])) for row in geometry[1:]] for geometry in loaded_geometry_list]

        geometry_list = [loaded_geometry_list[0]] 

        tmp_data = []
        
        
        for k in range(len(loaded_geometry_list) - 1):
            delta_num_geom = (np.array(loaded_geometry_num_list[k + 1], dtype="float64") - 
                            np.array(loaded_geometry_num_list[k], dtype="float64")) / (partition_function + 1)
            
            for i in range(partition_function + 1):
                frame_geom = np.array(loaded_geometry_num_list[k], dtype="float64") + delta_num_geom * i
                tmp_data.append(frame_geom)
                
        tmp_data = np.array(tmp_data, dtype="float64")
        
        
        if self.IDPP_flag:
            IDPP_obj = IDPP()
            tmp_data = IDPP_obj.opt_path(tmp_data)
  
        
        
        if self.align_distances > 0:
            tmp_data = distribute_geometry(tmp_data)
        
        if self.node_distance is not None:
            tmp_data = distribute_geometry_by_length(tmp_data, self.node_distance)
        
        for data in tmp_data:
            geometry_list.append([electric_charge_and_multiplicity] + [[element_list[num]] + list(map(str, geometry)) for num, geometry in enumerate(data)])        
        
        
        print("\n Geometries are loaded. \n")
        return geometry_list, element_list, electric_charge_and_multiplicity

    def print_geometry_list(self, new_geometry, element_list, electric_charge_and_multiplicity):
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
        file_directory = self.NEB_FOLDER_DIRECTORY+"path_ITR_"+str(optimize_num)+"_"+str(self.init_input)
        try:
            os.mkdir(file_directory)
        except:
            pass
        tmp_cs = [self.electronic_charge, self.spin_multiplicity]
        float_pattern = r"([+-]?(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)"
        for y, geometry in enumerate(geometry_list):
            tmp_geometry = []
            for geom in geometry:
                if len(geom) == 4 and re.match(r"[A-Za-z]+", str(geom[0])) \
                  and all(re.match(float_pattern, str(x)) for x in geom[1:]):
                        tmp_geometry.append(geom)

                if len(geom) == 2 and re.match(r"-*\d+", str(geom[0])) and re.match(r"-*\d+", str(geom[1])):
                    tmp_cs = geom   
                    
            with open(file_directory+"/"+self.init_input+"_"+str(y)+".xyz","w") as w:
                w.write(str(len(tmp_geometry))+"\n")
                w.write(str(tmp_cs[0])+" "+str(tmp_cs[1])+"\n")
                for rows in tmp_geometry:
                    w.write(f"{rows[0]:2}   {float(rows[1]):>17.12f}   {float(rows[2]):>17.12f}   {float(rows[3]):>17.12f}\n")
        return file_directory
        
    def sinple_plot(self, num_list, energy_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Electronic Energy [kcal/mol]", name="energy"):
        fig, ax = plt.subplots()
        ax.plot(num_list,energy_list, color_list[random.randint(0,len(color_list)-1)]+"--o" )

        ax.set_title(str(optimize_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(self.NEB_FOLDER_DIRECTORY+"plot_"+name+"_"+str(optimize_num)+".png", format="png", dpi=200)
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
        
        os.makedirs(file_directory, exist_ok=True)
        
        file_list = sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
       
        hess_count = 0
       
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
            
                logfile = file_directory+"/"+self.init_input+'_'+str(num)+'.log'
                #psi4.set_options({'pcm': True})
                #psi4.pcm_helper(pcm)
                
                psi4.set_output_file(logfile)
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                if self.unrestrict:
                    psi4.set_options({'reference': 'uks'})
                
                geometry_list, element_list, electric_charge_and_multiplicity = xyz2list(input_file, None)
                
                input_data = str(electric_charge_and_multiplicity[0])+" "+str(electric_charge_and_multiplicity[1])+"\n"
                for j in range(len(geometry_list)):
                    input_data += element_list[j]+"  "+geometry_list[j][0]+"  "+geometry_list[j][1]+"  "+geometry_list[j][2]+"\n"
                
                input_data = psi4.geometry(input_data)
                input_data_for_display = np.array(input_data.geometry(), dtype = "float64")
                   
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
                    with open(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".csv", "a") as f:
                        f.write("frequency,"+",".join(map(str, freqs))+"\n")
                
                hess_count += 1    


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
                
                
            psi4.core.clean()
        print("data sampling was completed...")


        try:
            if self.save_pict:
                tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
                self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
                print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            if self.save_pict:
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
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
        
        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
            
                print(input_file)
                geometry_list, element_list, electric_charge_and_multiplicity = xyz2list(input_file, None)
                words = []
                for i in range(len(geometry_list)):
                    words.append([element_list[i], float(geometry_list[i][0]), float(geometry_list[i][1]), float(geometry_list[i][2])])
                
                input_data_for_display = np.array(geometry_list, dtype="float64") / self.bohr2angstroms
                
                
                mol = pyscf.gto.M(atom = words,
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
                 
                    np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".npy", exact_hess)
                    with open(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".csv", "a") as f:
                        f.write("frequency,"+",".join(map(str, freqs["freq_wavenumber"]))+"\n")
                hess_count += 1
                
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        try:
            if self.save_pict:
                tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
                self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
                print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            if self.save_pict:
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
        os.makedirs(file_directory, exist_ok=True)
        file_list = sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
        
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)

                    
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                        
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                if int(electric_charge_and_multiplicity[1]) > 1 or self.unrestrict:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]), uhf=int(electric_charge_and_multiplicity[1]))
                else:
                    calc = Calculator(method, element_number_list, positions, charge=int(electric_charge_and_multiplicity[0]))                
                calc.set("max-iter", 500)
                calc.set("verbosity", 0)
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
            if self.save_pict:
                tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
                self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
                print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            if self.save_pict:
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

    
    def dxtb_calculation(self, file_directory, optimize_num, pre_total_velocity, element_number_list, electric_charge_and_multiplicity):
        #execute extended tight binding method calclation.
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        finish_frag = False
        method = self.usedxtb
        os.makedirs(file_directory, exist_ok=True)
        file_list = sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
        torch_element_number_list = torch.tensor(element_number_list)
        hess_count = 0
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                
                
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                torch_positions = torch.tensor(positions, requires_grad=True, dtype=torch.float32)
                
                max_scf_iteration = len(element_number_list) * 50 + 1000
                settings = {"maxiter": max_scf_iteration}
                
                
                if method == "GFN1-xTB":
                    calc = dxtb.calculators.GFN1Calculator(torch_element_number_list, opts=settings)
                elif method == "GFN2-xTB":
                    calc = dxtb.calculators.GFN2Calculator(torch_element_number_list, opts=settings)
                else:
                    print("method error")
                    raise

                if int(electric_charge_and_multiplicity[1]) > 1:

                    pos = torch_positions.clone().requires_grad_(True)
                    e = calc.get_energy(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1])) # hartree
                    calc.reset()
                    pos = torch_positions.clone().requires_grad_(True)
                    g = -1 * calc.get_forces(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1])) #hartree/Bohr
                    calc.reset()
                else:
                    pos = torch_positions.clone().requires_grad_(True)
                    e = calc.get_energy(pos, chrg=int(electric_charge_and_multiplicity[0])) # hartree
                    calc.reset()
                    pos = torch_positions.clone().requires_grad_(True)
                    g = -1 * calc.get_forces(pos, chrg=int(electric_charge_and_multiplicity[0])) #hartree/Bohr
                    calc.reset()
                    
                return_e = e.to('cpu').detach().numpy().copy()
                return_g = g.to('cpu').detach().numpy().copy()
                
                energy_list.append(return_e)
                gradient_list.append(return_g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(return_g)**2/(len(return_g)*3)))#RMS
                geometry_num_list.append(positions)
                num_list.append(num)
                
                if self.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                
                elif optimize_num % self.FC_COUNT == 0:
                    """exact autograd hessian"""
                    pos = torch_positions.clone().requires_grad_(True)
                    if int(electric_charge_and_multiplicity[1]) > 1:
                        exact_hess = calc.get_hessian(pos, chrg=int(electric_charge_and_multiplicity[0]), spin=int(electric_charge_and_multiplicity[1]))
                    else:
                        exact_hess = calc.get_hessian(pos, chrg=int(electric_charge_and_multiplicity[0]))
                    exact_hess = exact_hess.reshape(3*len(element_number_list), 3*len(element_number_list))
                    return_exact_hess = exact_hess.to('cpu').detach().numpy().copy()
                    
                    return_exact_hess = copy.copy(Calculationtools().project_out_hess_tr_and_rot_for_coord(return_exact_hess, element_number_list.tolist(), positions))
                    np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(hess_count)+".npy", return_exact_hess)
                   
                    calc.reset()
                hess_count += 1
                
            except Exception as error:
                print(error)
                try:
                    calc.reset()
                except:
                    pass
                
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        try:
            if self.save_pict:
                tmp_ene_list = np.array(energy_list, dtype="float64")*self.hartree2kcalmol
                self.sinple_plot(num_list, tmp_ene_list - tmp_ene_list[0], file_directory, optimize_num)
                print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            if self.save_pict:
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



    def make_traj_file(self, file_directory):
        print("\nprocessing geometry collecting ...\n")
        file_list = sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
        
        for m, file in enumerate(file_list):
            tmp_geometry_list, element_list, _ = xyz2list(file, None)
            atom_num = len(tmp_geometry_list)
            with open(file_directory+"/"+self.init_input+"_path.xyz","a") as w:
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
                for i in range(len(tmp_geometry_list)):
                    w.write(f"{element_list[i]:2}   {float(tmp_geometry_list[i][0]):17.12f}  {float(tmp_geometry_list[i][1]):17.12f}  {float(tmp_geometry_list[i][2]):17.12f}\n")
        print("\ncollecting geometries was complete...\n")
        return


    def FIRE_calc(self, geometry_num_list, total_force_list, pre_total_velocity, optimize_num, total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom):
        velocity_neb = []

        for num in range(len(total_velocity)):
            part_velocity_neb = []
            for i in range(len(total_force_list[0])):
                force_norm = np.linalg.norm(total_force_list[num][i])
                velocity_norm = np.linalg.norm(total_velocity[num][i])
                part_velocity_neb.append((1.0 - self.a) * total_velocity[num][i] + self.a * (velocity_norm / force_norm) * total_force_list[num][i])
            velocity_neb.append(part_velocity_neb)

        velocity_neb = np.array(velocity_neb)

        np_dot_param = 0.0
        if optimize_num != 0 and len(pre_total_velocity) > 1:
            np_dot_param = np.sum([np.dot(pre_total_velocity[num_1][num_2], total_force_num.T) for num_1, total_force in enumerate(total_force_list) for num_2, total_force_num in enumerate(total_force)])
            print(np_dot_param)

        if optimize_num > 0 and np_dot_param > 0 and len(pre_total_velocity) > 1:
            if self.n_reset > self.FIRE_N_accelerate:
                self.dt = min(self.dt * self.FIRE_f_inc, self.FIRE_dt_max)
                self.a *= self.FIRE_f_inc
            self.n_reset += 1
        else:
            velocity_neb *= 0
            self.a = self.FIRE_a_start
            self.dt *= self.FIRE_f_decelerate
            self.n_reset = 0

        total_velocity = velocity_neb + self.dt * total_force_list

        if optimize_num != 0 and len(pre_total_velocity) > 1:
            total_delta = self.dt * (total_velocity + pre_total_velocity)
        else:
            total_delta = self.dt * total_velocity

        # Calculate the movement vector using TR_calc
        move_vector = self.NEB_TR.TR_calc(geometry_num_list, total_force_list, total_delta, biased_energy_list, pre_biased_energy_list, pre_geom)

        # Update geometry using the move vector
        new_geometry = (geometry_num_list + move_vector) * self.bohr2angstroms

        return new_geometry


    def GRFO_calc(self, geometry_num_list, total_force_list, prev_geometry_num_list, prev_total_force_list, biased_gradient_list, optimize_num, STRING_FORCE_CALC, biased_energy_list, pre_biased_energy_list):
        natoms = len(geometry_num_list[0])
        nnode_minus2 = len(geometry_num_list) - 2
        nnode = len(geometry_num_list)
        total_delta = []
        if optimize_num % self.FC_COUNT == 0 and self.FC_COUNT > 0:
            for num in range(1, len(total_force_list)-1):
                hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy")
                proj_hess = STRING_FORCE_CALC.projection_hessian(geometry_num_list[num-1], geometry_num_list[num], geometry_num_list[num+1], biased_gradient_list[num-1:num+2], hess, biased_energy_list[num-1:num+2])
                np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy", proj_hess)
        
        
        if optimize_num == 0:
            global_hess = np.eye(3*natoms*nnode_minus2)
            if self.FC_COUNT < 1:
                for i in range(nnode):
                    indentity_hess = np.eye(3*natoms)
                    np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(i)+".npy", indentity_hess)
        else:
            global_hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_global_hessian.npy")
        
        if optimize_num % self.FC_COUNT == 0:
            for num in range(nnode_minus2):
                hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(num)+".npy")
                global_hess[3*natoms*num:3*natoms*(num+1), 3*natoms*num:3*natoms*(num+1)] = hess
        
        # Opt for node 0
        min_hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_0.npy")
        OPTmin = RationalFunctionOptimization(method="rfo_fsb", saddle_order=0, trust_radius=0.1)
        OPTmin.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
        OPTmin.set_hessian(min_hess)
        if optimize_num == 0:
            OPTmin.Initialization = True
            pre_B_g = None
            pre_geom = None
        else:
            OPTmin.Initialization = False
            OPTmin.lambda_clip_flag = True
            OPTmin.lambda_s_scale = 0.1 * 1.0 / (1.0 + np.exp(np.linalg.norm(total_force_list[0]) - 5.0))
            pre_B_g = -1 * prev_total_force_list[0].reshape(-1, 1)
            pre_geom = prev_geometry_num_list[0].reshape(-1, 1)
        geom_num_list = geometry_num_list[0].reshape(-1, 1)    
        B_g = -1 * total_force_list[0].reshape(-1, 1)    
        move_vec = -1 * OPTmin.run(geom_num_list, B_g, pre_B_g, pre_geom, 0.0, 0.0, [], [], B_g, pre_B_g)
        total_delta.append(move_vec.reshape(-1, 3))
        min_hess = OPTmin.get_hessian()
        np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_0.npy", min_hess)
        
        # Opt for node 1 to nnode-1
        globalOPT = RationalFunctionOptimization(method="rfo_neb_bofill", saddle_order=1, trust_radius=0.1)
        globalOPT.set_bias_hessian(np.zeros((3*natoms*nnode_minus2, 3*natoms*nnode_minus2)))
        globalOPT.set_hessian(global_hess)
       
        if optimize_num == 0:
            globalOPT.Initialization = True
            pre_B_g = None
            pre_geom = None
        else:
            globalOPT.Initialization = False
            globalOPT.lambda_clip_flag = True
            globalOPT.lambda_s_scale = 0.1 * 1.0 / (1.0 + np.exp(np.linalg.norm(total_force_list[0]) - 5.0))
            pre_B_g = -1 * prev_total_force_list[1:nnode-1].reshape(-1, 1)
            pre_geom = prev_geometry_num_list[1:nnode-1].reshape(-1, 1)
        geom_num_list = geometry_num_list[1:nnode-1].reshape(-1, 1)
        B_g = -1 * total_force_list[1:nnode-1].reshape(-1, 1)
       
        move_vec = -1 * globalOPT.run(geom_num_list, B_g, pre_B_g, pre_geom, 0.0, 0.0, [], [], B_g, pre_B_g)
        for i in range(nnode_minus2):
            total_delta.append(move_vec[3*natoms*(i):3*natoms*(i+1)].reshape(-1, 3))
        global_hess = globalOPT.get_hessian()
        
        np.save(self.NEB_FOLDER_DIRECTORY+"tmp_global_hessian.npy", global_hess)
        del globalOPT
        del global_hess
        # Opt for node nnode
        
        min_hess = np.load(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(nnode-1)+".npy")
        OPTmin = RationalFunctionOptimization(method="rfo_fsb", saddle_order=0, trust_radius=0.1)
        OPTmin.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
        OPTmin.set_hessian(min_hess)
        if optimize_num == 0:
            OPTmin.Initialization = True
            pre_B_g = None
            pre_geom = None
        else:
            OPTmin.Initialization = False
            OPTmin.lambda_clip_flag = True
            OPTmin.lambda_s_scale = 0.1 * 1.0 / (1.0 + np.exp(np.linalg.norm(total_force_list[-1]) - 5.0))
            pre_B_g = -1 * prev_total_force_list[-1].reshape(-1, 1)
            pre_geom = prev_geometry_num_list[-1].reshape(-1, 1)
        geom_num_list = geometry_num_list[-1].reshape(-1, 1)    
        B_g = -1 * total_force_list[-1].reshape(-1, 1)    
        move_vec = -1 * OPTmin.run(geom_num_list, B_g, pre_B_g, pre_geom, 0.0, 0.0, [], [], B_g, pre_B_g)
        total_delta.append(move_vec.reshape(-1, 3))
        min_hess = OPTmin.get_hessian()
        np.save(self.NEB_FOLDER_DIRECTORY+"tmp_hessian_"+str(nnode-1)+".npy", min_hess)
        
        # calculate move vector
        move_vector_list = self.NEB_TR.TR_calc(geometry_num_list, total_force_list, total_delta, biased_energy_list, pre_biased_energy_list, pre_geom)
        
        new_geometry_list = (geometry_num_list + move_vector_list) * self.bohr2angstroms
        
        return new_geometry_list
    
    
    def RFO_calc(self, geometry_num_list, total_force_list, prev_geometry_num_list, prev_total_force_list, biased_gradient_list, optimize_num, STRING_FORCE_CALC, biased_energy_list, pre_biased_energy_list):
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
                OPT = RationalFunctionOptimization(method="rsirfo_fsb", saddle_order=0, trust_radius=0.1)
            else:
                OPT = RationalFunctionOptimization(method="rfo_neb_bofill", saddle_order=1, trust_radius=0.1)
            OPT.set_bias_hessian(np.zeros((3*natoms, 3*natoms)))
            OPT.set_hessian(hessian)
            OPT.iter = optimize_num
            
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
        
        move_vector_list = self.NEB_TR.TR_calc(geometry_num_list, total_force_list, total_delta, biased_energy_list, pre_biased_energy_list, pre_geom)
        
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
        geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list(self.init_input, self.partition)
        self.element_list = element_list
        
        file_directory = self.make_psi4_input_file(geometry_list, 0)
        pre_total_velocity = [[[]]]
        force_data = force_data_parser(self.args)
        if len(force_data["projection_constraint_condition_list"]) > 0:
            projection_constraint_flag = True
        else:
            projection_constraint_flag = False        
        #prepare for FIRE method 
        self.NEB_TR = trust_radius_neb.TR_NEB(NEB_FOLDER_DIRECTORY=self.NEB_FOLDER_DIRECTORY, 
         fix_init_edge=self.fix_init_edge,
         fix_end_edge=self.fix_end_edge,
         apply_convergence_criteria=self.apply_convergence_criteria,) 
        
        
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
            STRING_FORCE_CALC = CaluculationBNEB2(self.APPLY_CI_NEB)
        elif self.mix:
            STRING_FORCE_CALC_1 = CaluculationBNEB(self.APPLY_CI_NEB)
            STRING_FORCE_CALC_2 = CaluculationOM(self.APPLY_CI_NEB)
        elif self.ewbneb:
            STRING_FORCE_CALC = CaluculationEWBNEB(self.APPLY_CI_NEB)
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
        
        if self.cg_method and self.lbfgs_method: 
            print("You can not use CG and LBFGS at the same time.")
            exit()
            
        if self.cg_method:
            CGOptimizer = conjugate_gradient_neb.ConjugateGradientNEB(TR_NEB=self.NEB_TR, cg_method=self.cg_method)
        
        if self.lbfgs_method:
            LBFGSOptimizer = lbfgs_neb.LBFGS_NEB(TR_NEB=self.NEB_TR)
        #AFIREOptimizer = afire_neb.AFIRE_NEB(TR_NEB=self.NEB_TR)
        #QuickMinOptimizer = quickmin_neb.QuickMin_NEB(TR_NEB=self.NEB_TR)
        #------------------
        for optimize_num in range(self.NEB_NUM):
            exit_file_detect = os.path.exists(self.NEB_FOLDER_DIRECTORY+"end.txt")
            if exit_file_detect:
                if psi4:
                    psi4.core.clean()
                break
            
            print("\n\n\n NEB: ITR.  "+str(optimize_num)+"  \n\n\n")
            self.make_traj_file(file_directory)
            #------------------
            #get energy and gradient
            if self.args.usextb == "None" and self.usedxtb == "None":
                if self.pyscf:
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.pyscf_calculation(file_directory, optimize_num, pre_total_velocity, electric_charge_and_multiplicity)
                else:
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory,optimize_num, pre_total_velocity)
            else:
                if self.usedxtb == "None":
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list, electric_charge_and_multiplicity)
                else:
                    energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.dxtb_calculation(file_directory, optimize_num, pre_total_velocity, element_number_list, electric_charge_and_multiplicity)
            
            
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
            if optimize_num == 0:
                pre_biased_energy_list = copy.copy(biased_energy_list)
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
            if self.mix:
                total_force_1 = STRING_FORCE_CALC_1.calc_force(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list) / 2
                total_force_2 = STRING_FORCE_CALC_2.calc_force(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list) / 2
                total_force =  total_force_1 + total_force_2
            else:
                total_force = STRING_FORCE_CALC.calc_force(geometry_num_list, biased_energy_list, biased_gradient_list, optimize_num, element_list)

            #------------------
            cos_list = []
            tot_force_rms_list = []
            tot_force_max_list = []
            bias_force_rms_list = []
            for i in range(len(total_force)):
                cos = np.sum(total_force[i]*biased_gradient_list[i])/(np.linalg.norm(total_force[i])*np.linalg.norm(biased_gradient_list[i]))
                cos_list.append(cos)
                tot_force_rms = np.sqrt(np.mean(total_force[i]**2))
                tot_force_rms_list.append(tot_force_rms)
                tot_force_max = np.max(total_force[i])
                tot_force_max_list.append(tot_force_max)
                bias_force_rms = np.sqrt(np.mean(biased_gradient_list[i]**2))
                bias_force_rms_list.append(bias_force_rms)
                
            with open(self.NEB_FOLDER_DIRECTORY+"bias_force_rms.csv", "a") as f:
                f.write(",".join(list(map(str,bias_force_rms_list)))+"\n")
            
            if self.save_pict:
                self.sinple_plot([x for x in range(len(total_force))], cos_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="cosθ", name="orthogonality")
            
            with open(self.NEB_FOLDER_DIRECTORY+"orthogonality.csv", "a") as f:
                f.write(",".join(list(map(str,cos_list)))+"\n")
            
            if self.save_pict:
                self.sinple_plot([x for x in range(len(total_force))][1:-1], tot_force_rms_list[1:-1], file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Perpendicular Gradient (RMS) [a.u.]", name="perp_rms_gradient")
            
            with open(self.NEB_FOLDER_DIRECTORY+"perp_rms_gradient.csv", "a") as f:
                f.write(",".join(list(map(str,tot_force_rms_list)))+"\n")
            
            if self.save_pict:
                self.sinple_plot([x for x in range(len(total_force))], tot_force_max_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Perpendicular Gradient (MAX) [a.u.]", name="perp_max_gradient")
            
            with open(self.NEB_FOLDER_DIRECTORY+"perp_max_gradient.csv", "a") as f:
                f.write(",".join(list(map(str,tot_force_max_list)))+"\n")
            

            #------------------
            #relax path
            if self.global_quasi_newton:
                new_geometry = self.GRFO_calc(geometry_num_list, total_force, pre_geom, pre_total_force, biased_gradient_list, optimize_num, STRING_FORCE_CALC, biased_energy_list, pre_biased_energy_list)
            
            elif self.FC_COUNT != -1:
                new_geometry = self.RFO_calc(geometry_num_list, total_force, pre_geom, pre_total_force, biased_gradient_list, optimize_num, STRING_FORCE_CALC, biased_energy_list, pre_biased_energy_list)
            
            elif optimize_num < self.sd:
                
                total_velocity = self.force2velocity(total_force, element_list)
                
                if self.lbfgs_method:
                    new_geometry  = LBFGSOptimizer.LBFGS_NEB_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
                if self.cg_method:
                    new_geometry  = CGOptimizer.CG_NEB_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
                else:
                    new_geometry = self.FIRE_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
                    #new_geometry = AFIREOptimizer.AFIRE_NEB_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
                    #new_geometry = QuickMinOptimizer.QuickMin_NEB_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, cos_list, biased_energy_list, pre_biased_energy_list, pre_geom)
            else:
                new_geometry = self.SD_calc(geometry_num_list, total_force)
                
            if optimize_num > self.climbing_image_start and (optimize_num - self.climbing_image_start) % self.climbing_image_interval == 0:
                new_geometry = apply_climbing_image(new_geometry, biased_energy_list)
            
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
            
            if self.align_distances < 1:
                pass
            elif optimize_num % self.align_distances == 0 and optimize_num > 0:
                print("Aligning geometries...")
                tmp_new_geometry = distribute_geometry(np.array(new_geometry))
                for k in range(len(new_geometry)):
                    new_geometry[k] = copy.copy(tmp_new_geometry[k])
            
            tmp_instance_fileio = FileIO(file_directory+"/", "dummy.txt")
            tmp_instance_fileio.argrelextrema_txt_save(biased_energy_list, "approx_TS_node", "max")
            tmp_instance_fileio.argrelextrema_txt_save(biased_energy_list, "approx_EQ_node", "min")
            tmp_instance_fileio.argrelextrema_txt_save(bias_force_rms_list, "local_min_bias_grad_node", "min")
            
            #------------------
            pre_geom = geometry_num_list
            
            geometry_list = self.print_geometry_list(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, optimize_num+1)
            pre_total_force = total_force
            pre_total_velocity = total_velocity
            pre_biased_energy_list = biased_energy_list
            #------------------
            with open(self.NEB_FOLDER_DIRECTORY+"energy_plot.csv", "a") as f:
                f.write(",".join(list(map(str,biased_energy_list.tolist())))+"\n")
            
        self.make_traj_file(file_directory) 
        print("Complete...")
        return

def calc_path_length_list(geometry_list):
    path_length_list = [0.0]
    for i in range(len(geometry_list)-1):
        tmp_geometry_list_j = geometry_list[i+1] - np.mean(geometry_list[i+1], axis=0)
        tmp_geometry_list_i = geometry_list[i] - np.mean(geometry_list[i], axis=0)
        
        
        path_length = path_length_list[-1] + np.linalg.norm(tmp_geometry_list_j - tmp_geometry_list_i)
        path_length_list.append(path_length)
    return path_length_list

def apply_climbing_image(geometry_list, energy_list):
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    local_maxima, local_minima = spline_interpolation(path_length_list, energy_list)
    print(local_maxima)
    for distance, energy in local_maxima:
        print("Local maximum at distance: ", distance)
        for i in range(2, len(path_length_list)-2):
            if path_length_list[i] >= distance or distance >= path_length_list[i+1]:
                continue
            delta_t = (distance - path_length_list[i]) / (path_length_list[i+1] - path_length_list[i])
            tmp_geometry = geometry_list[i] + (geometry_list[i+1] - geometry_list[i]) * delta_t
            tmp_geom_list = [geometry_list[i], tmp_geometry, geometry_list[i+1]]
            idpp_instance = IDPP()
            tmp_geom_list = idpp_instance.opt_path(tmp_geom_list)
            geometry_list[i] = tmp_geom_list[1]
    return geometry_list

def distribute_geometry(geometry_list):
    nnode = len(geometry_list)
    path_length_list = calc_path_length_list(geometry_list)
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

def distribute_geometry_by_length(geometry_list, angstrom_spacing):
    path_length_list = calc_path_length_list(geometry_list)
    total_length = path_length_list[-1]
    new_geometry_list = [geometry_list[0]]
    
    max_steps = int(total_length // angstrom_spacing)
    for i in range(1, max_steps):
        dist = i * angstrom_spacing
       
        for j in range(len(path_length_list) - 1):
            if path_length_list[j] <= dist <= path_length_list[j+1]:
                break
      
        delta_t = (dist - path_length_list[j]) / (path_length_list[j+1] - path_length_list[j])
        new_geometry = geometry_list[j] + (geometry_list[j+1] - geometry_list[j]) * delta_t
        new_geometry_list.append(new_geometry)

    new_geometry_list.append(geometry_list[-1])
    return new_geometry_list