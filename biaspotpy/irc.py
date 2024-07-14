import numpy as np
import os
import copy
import glob

from Optimizer.hessian_update import ModelHessianUpdate
from potential import BiasPotentialCalculation
from parameter import atomic_mass, UnitValueLib
from calc_tools import Calculationtools
from visualization import Graph
from fileio import FileIO

def convergence_check(grad, MAX_FORCE_THRESHOLD, RMS_FORCE_THRESHOLD):
    if abs(grad.max()) < MAX_FORCE_THRESHOLD and abs(np.sqrt((grad**2).mean())) < RMS_FORCE_THRESHOLD:#convergent criteria
        return True
    else:
        return False


class LQA:#local quadratic approximation method
    def __init__(self, element_list, electric_charge_and_multiplicity, FC_count, file_directory, final_directory, force_data, max_step=1000, step_size=0.1, init_coord=None, init_hess=None, calc_engine=None, xtb_method=None):
        self.max_step = max_step
        self.step_size = step_size
        self.N_euler = 20000
        self.ModelHessianUpdate = ModelHessianUpdate()
        self.CE = calc_engine
        self.FC_count = FC_count
        # initial condition
        self.coords = init_coord
        self.init_hess = init_hess
        self.xtb_method = xtb_method
        
        # convergence criteria
        self.MAX_FORCE_THRESHOLD = 0.0006 #0.0003
        self.RMS_FORCE_THRESHOLD = 0.0004 #0.0002

        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.directory = file_directory
        self.final_directory = final_directory
        self.force_data = force_data
        
        # IRC data
        self.irc_bias_energy_list = []
        self.irc_energy_list = []
        self.irc_coords = []
        self.irc_mw_coords = []
        self.irc_mw_gradients = []
        self.irc_mw_bias_gradients = []
        self.path_bending_angle_list = []
        
        
    def run(self):
        #ref1: J. Chem. Phys. 93, 5634–5642 (1990)
        #ref2: J. Chem. Phys. 120, 9918–9924 (2004)
        print("local quadratic approximation method")
        geom_num_list = self.coords
        mw_hessian = self.init_hess
        CalcBiaspot = BiasPotentialCalculation(mw_hessian, self.FC_count, self.directory)
        for iter in range(1, self.max_step):
            print("# STEP: ", iter)
            exit_file_detect = os.path.exists(self.directory+"end.txt")

            if exit_file_detect:
                break
                  
            e, g, geom_num_list, finish_frag = self.CE.single_point(self.final_directory, self.element_list, iter, self.electric_charge_and_multiplicity, self.xtb_method,  UnitValueLib().bohr2angstroms*geom_num_list)
            _, B_e, B_g, BPA_hessian = CalcBiaspot.main(e, g, geom_num_list, self.element_list, self.force_data, g, iter, geom_num_list)
            
            
            if finish_frag:
                break
            
            if iter % self.FC_count == 0:
                mw_hessian = self.CE.Model_hess
                mw_hessian = Calculationtools().project_out_hess_tr_and_rot(mw_hessian, self.element_list, geom_num_list)
            
            elem_mass_list = np.array([atomic_mass(elem) for elem in self.element_list], dtype="float64")
            
            three_elem_mass_list = np.stack([elem_mass_list, elem_mass_list, elem_mass_list]).T.reshape(len(elem_mass_list)*3)
            
            mw_BPA_hessian = np.dot(np.diag(1/three_elem_mass_list), np.dot(BPA_hessian, np.diag(1/three_elem_mass_list)))
            
            mw_geom_num_list = copy.copy(geom_num_list)
            for i in range(len(geom_num_list)):
                
                mw_geom_num_list[i] *= elem_mass_list[i]
            
            mw_B_g = copy.copy(B_g)
            mw_g = copy.copy(g)
            for i in range(len(g)):#mass-weighted gradients
                
                mw_g[i] /= elem_mass_list[i]
                mw_B_g[i] /= elem_mass_list[i]
            

            self.irc_energy_list.append(e)
            self.irc_bias_energy_list.append(B_e)
            self.irc_coords.append(geom_num_list)
            self.irc_mw_coords.append(mw_geom_num_list)
            self.irc_mw_gradients.append(mw_g)
            self.irc_mw_bias_gradients.append(mw_B_g)
                 

            if iter > 1:
                delta_g = (self.irc_mw_gradients[-1] - self.irc_mw_gradients[-2]).reshape(len(geom_num_list)*3, 1)
                delta_x = (self.irc_mw_coords[-1] - self.irc_mw_coords[-2]).reshape(len(geom_num_list)*3, 1)
               
                delta_hess = self.ModelHessianUpdate.FSB_hessian_update(mw_hessian, delta_x, delta_g)
                mw_hessian += delta_hess

                eigenvalues, eigenvectors = np.linalg.eigh(mw_hessian+mw_BPA_hessian)
                small_eigvals = np.abs(eigenvalues) < 1e-8
                eigenvalues = eigenvalues[~small_eigvals]
                eigenvectors = eigenvectors[:,~small_eigvals]

                #for j in range(len(eigenvectors.T)):
                #     eigenvectors[:, j] = eigenvectors[:, j] / np.linalg.norm(eigenvectors[:, j])
                
                dt = 1 / self.N_euler * self.step_size / np.linalg.norm(mw_B_g)

                mw_gradient_proj = np.dot(eigenvectors.T, mw_B_g.reshape(len(geom_num_list)*3, 1))

                #integration of the step size
                t = dt
                current_length = 0
                for j in range(self.N_euler):
                    dsdt = np.sqrt(np.sum(mw_gradient_proj**2 * np.exp(-2*eigenvalues*t)))
                    current_length += dsdt * dt
                    if current_length > self.step_size:
                        break
                    t += dt
                    
                    
                alphas = (np.exp(-eigenvalues*t) - 1) / eigenvalues
                A = np.dot(eigenvectors, np.dot(np.diag(alphas), eigenvectors.T))
                step = np.dot(A, mw_B_g.reshape(len(geom_num_list)*3, 1))
                trust_radius = max(min(np.linalg.norm(step), self.step_size), 0.0001)
                
                step = step / np.linalg.norm(step) * trust_radius
                geom_num_list = geom_num_list + step.reshape(len(geom_num_list), 3)
                geom_num_list -= Calculationtools().calc_center_of_mass(geom_num_list, self.element_list)

                if convergence_check(B_g, self.MAX_FORCE_THRESHOLD, self.RMS_FORCE_THRESHOLD):
                    print("convergence reached. (IRC)")
                    break
                
            else:
                geom_num_list = geom_num_list - mw_B_g * self.step_size * 0.1 / np.linalg.norm(mw_B_g)
                geom_num_list -= Calculationtools().calc_center_of_mass(geom_num_list, self.element_list)
            
            for i in range(len(geom_num_list)):
                print(self.element_list[i]+" "+str(geom_num_list[i][0]*UnitValueLib().bohr2angstroms)+" "+str(geom_num_list[i][1]*UnitValueLib().bohr2angstroms)+" "+str(geom_num_list[i][2]*UnitValueLib().bohr2angstroms))
            
            if iter > 2:
                bend_angle = Calculationtools().calc_multi_dim_vec_angle(self.irc_mw_coords[-3]-self.irc_mw_coords[-2], self.irc_mw_coords[-1]-self.irc_mw_coords[-2])
                self.path_bending_angle_list.append(np.degrees(bend_angle))
                print("Path bending angle: ", np.degrees(bend_angle))
                
            #display information
            print("Energy: ", e)
            print("Bias Energy: ", B_e)
        return
        



class IRC:
    def __init__(self, directory, final_directory, irc_method, QM_interface, element_list, electric_charge_and_multiplicity, force_data, xtb_method, FC_count=-1, hessian=None):
        if hessian is None:
            self.hessian_flag = False
            
        else:
            self.hessian_flag = True
            self.hessian = hessian
        
        self.step_size = float(irc_method[0])
        self.max_step = int(irc_method[1])
        self.method = str(irc_method[2])
            
        self.file_directory = directory
        self.final_directory = final_directory
        self.QM_interface = QM_interface
        
        self.element_list = element_list
        self.electric_charge_and_multiplicity = electric_charge_and_multiplicity
        self.xtb_method = xtb_method
        
        self.force_data = force_data
        self.FC_count = FC_count
        
    
    def saddle_check(self):
        if not self.hessian_flag:
            self.QM_interface.hessian_flag = True
            iter = 1
        else:
            self.QM_interface.FC_COUNT = -1
            iter = 1
        
        fin_xyz = glob.glob(self.final_directory+"/*.xyz")
        with open(fin_xyz[0], "r") as f:
            words = f.read().splitlines()
        geom_num_list = []
        for i in range(len(words)):
            if len(words[i].split()) > 3:
                geom_num_list.append([float(words[i].split()[1]), float(words[i].split()[2]), float(words[i].split()[3])])

        init_e, init_g, geom_num_list, finish_frag = self.QM_interface.single_point(self.final_directory, self.element_list, iter, self.electric_charge_and_multiplicity, self.xtb_method, geom_num_list)
        self.QM_interface.hessian_flag = False
        self.QM_interface.FC_COUNT = self.FC_count
        if finish_frag:
            return 0, 0, 0, finish_frag
        self.hessian = self.QM_interface.Model_hess
        self.QM_interface.hessian_flag = False
        CalcBiaspot = BiasPotentialCalculation(self.hessian, self.FC_count, self.final_directory)
        _, init_B_e, init_B_g, BPA_hessian = CalcBiaspot.main(init_e, init_g, geom_num_list, self.element_list, self.force_data, init_g, iter, geom_num_list)#new_geometry:ang.
        self.hessian += BPA_hessian
        
        self.hessian = Calculationtools().project_out_hess_tr_and_rot(self.hessian, self.element_list, geom_num_list)
       
            
        self.init_e = init_e
        self.init_g = init_g
        self.init_B_e = init_B_e
        self.init_B_g = init_B_g
        eigenvalue, eigenvector = np.linalg.eigh(self.hessian)
        tmp_list = np.where(eigenvalue < -1e-8, True, False)
        imaginary_count = np.count_nonzero(tmp_list == True)
        print("number of imaginary eigenvalue: ", imaginary_count)
        if imaginary_count == 1:
            print("execute IRC")
            imaginary_spring_const_idx = np.where(eigenvalue < -1e-8)[0]
            initial_step = eigenvector[imaginary_spring_const_idx] / np.linalg.norm(eigenvector[imaginary_spring_const_idx]) * self.step_size * 0.1
            initial_step = initial_step.reshape(len(geom_num_list), 3)
            IRC_flag = True
        else:
            print("execute meta-IRC")
            initial_step = self.QM_interface.gradient / np.linalg.norm(self.QM_interface.gradient) * self.step_size * 0.1
            initial_step = initial_step.reshape(len(geom_num_list), 3)
            for i in range(len(initial_step)):
                initial_step[i] /= atomic_mass(self.element_list[i])
            IRC_flag = False
            
        return initial_step, IRC_flag, geom_num_list, finish_frag
    
    
    def calc_IRCpath(self):
        print("IRC carry out...")
        if self.method.upper() == "LQA":
            if self.IRC_flag:
                #forward
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                IRCmethod = LQA(self.element_list, self.electric_charge_and_multiplicity, self.FC_count, self.file_directory, self.final_directory, self.force_data, max_step=self.max_step, step_size=self.step_size, init_coord=init_geom, init_hess=self.hessian, calc_engine=self.QM_interface, xtb_method=self.xtb_method)
                IRCmethod.run()
                
                irc_bias_energy_list = IRCmethod.irc_bias_energy_list[::-1] + [self.init_B_e]
                irc_energy_list = IRCmethod.irc_energy_list[::-1] + [self.init_e]
                irc_coords = IRCmethod.irc_coords[::-1] + [self.geom_num_list]
                irc_mw_coords = IRCmethod.irc_mw_coords[::-1] 
                irc_mw_gradients =  IRCmethod.irc_mw_gradients[::-1] 
                irc_mw_bias_gradients = IRCmethod.irc_mw_bias_gradients[::-1]
                path_bending_angle_list = IRCmethod.path_bending_angle_list[::-1]
                
                #backward
                print("Backward IRC")
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = LQA(self.element_list, self.electric_charge_and_multiplicity, self.FC_count, self.file_directory, self.final_directory, self.force_data, max_step=self.max_step, step_size=self.step_size, init_coord=init_geom, init_hess=self.hessian, calc_engine=self.QM_interface, xtb_method=self.xtb_method)
                IRCmethod.run()
                
                irc_bias_energy_list += IRCmethod.irc_bias_energy_list
                irc_energy_list += IRCmethod.irc_energy_list
                irc_coords += IRCmethod.irc_coords
                irc_mw_coords += IRCmethod.irc_mw_coords
                irc_mw_gradients +=  IRCmethod.irc_mw_gradients
                irc_mw_bias_gradients += IRCmethod.irc_mw_bias_gradients
                path_bending_angle_list += IRCmethod.path_bending_angle_list
                
            else:
                #meta-IRC
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = LQA(self.element_list, self.electric_charge_and_multiplicity, self.FC_count, self.file_directory, self.final_directory, self.force_data, max_step=self.max_step, step_size=self.step_size, init_coord=init_geom, init_hess=self.hessian, calc_engine=self.QM_interface, xtb_method=self.xtb_method)
                IRCmethod.run()
                
                irc_bias_energy_list = IRCmethod.irc_bias_energy_list
                irc_energy_list = IRCmethod.irc_energy_list
                irc_coords = IRCmethod.irc_coords
                irc_mw_coords = IRCmethod.irc_mw_coords
                irc_mw_gradients =  IRCmethod.irc_mw_gradients
                irc_mw_bias_gradients = IRCmethod.irc_mw_bias_gradients
                path_bending_angle_list = IRCmethod.path_bending_angle_list
                
                
        else:
            print("Unexpected method. (default method is LQA.)")
            if self.IRC_flag:
                #forward
                print("Forward IRC")
                init_geom = self.geom_num_list + self.initial_step
                IRCmethod = LQA(self.element_list, self.electric_charge_and_multiplicity, self.FC_count, self.file_directory, self.final_directory, self.force_data, max_step=self.max_step, step_size=self.step_size, init_coord=init_geom, init_hess=self.hessian, calc_engine=self.QM_interface, xtb_method=self.xtb_method)
                IRCmethod.run()
                
                irc_bias_energy_list = IRCmethod.irc_bias_energy_list[::-1] + [self.init_B_e]
                irc_energy_list = IRCmethod.irc_energy_list[::-1] + [self.init_e]
                irc_coords = IRCmethod.irc_coords[::-1] + [self.geom_num_list]
                irc_mw_coords = IRCmethod.irc_mw_coords[::-1]
                irc_mw_gradients =  IRCmethod.irc_mw_gradients[::-1]
                irc_mw_bias_gradients = IRCmethod.irc_mw_bias_gradients[::-1]
                path_bending_angle_list = IRCmethod.path_bending_angle_list[::-1]
                
                #backward
                print("Backward IRC")
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = LQA(self.element_list, self.electric_charge_and_multiplicity, self.FC_count, self.file_directory, self.final_directory, self.force_data, max_step=self.max_step, step_size=self.step_size, init_coord=init_geom, init_hess=self.hessian, calc_engine=self.QM_interface, xtb_method=self.xtb_method)
                IRCmethod.run()
                
                irc_bias_energy_list += IRCmethod.irc_bias_energy_list
                irc_energy_list += IRCmethod.irc_energy_list
                irc_coords += IRCmethod.irc_coords
                irc_mw_coords += IRCmethod.irc_mw_coords
                irc_mw_gradients +=  IRCmethod.irc_mw_gradients
                irc_mw_bias_gradients += IRCmethod.irc_mw_bias_gradients
                path_bending_angle_list += IRCmethod.path_bending_angle_list
                
            else:
                #meta-IRC
                init_geom = self.geom_num_list - self.initial_step
                IRCmethod = LQA(self.element_list, self.electric_charge_and_multiplicity, self.FC_count, self.file_directory, self.final_directory, self.force_data, max_step=self.max_step, step_size=self.step_size, init_coord=init_geom, init_hess=self.hessian, calc_engine=self.QM_interface, xtb_method=self.xtb_method)
                IRCmethod.run()
                
                irc_bias_energy_list = IRCmethod.irc_bias_energy_list
                irc_energy_list = IRCmethod.irc_energy_list
                irc_coords = IRCmethod.irc_coords
                irc_mw_coords = IRCmethod.irc_mw_coords
                irc_mw_gradients =  IRCmethod.irc_mw_gradients
                irc_mw_bias_gradients = IRCmethod.irc_mw_bias_gradients
                path_bending_angle_list = IRCmethod.path_bending_angle_list
        
        
        G = Graph(self.file_directory)
        G.double_plot(np.array(range(len(irc_energy_list))), UnitValueLib().hartree2kcalmol * np.array(irc_energy_list), UnitValueLib().hartree2kcalmol * np.array(irc_bias_energy_list), add_file_name="IRC_path_energy")
        G.single_plot(np.array(range(len(path_bending_angle_list))), np.array(path_bending_angle_list), self.file_directory, atom_num=0, axis_name_1="# STEP", axis_name_2="bending angle [degrees]", name="IRC_path_bending_angle")
        FIO = FileIO(self.file_directory, "")
        FIO.xyz_file_save_for_IRC(self.element_list, np.array(irc_coords) * UnitValueLib().bohr2angstroms)
        
        return
    
    
    def run(self):
        
        self.initial_step, self.IRC_flag, self.geom_num_list, finish_flag = self.saddle_check()
        if finish_flag:
            print("IRC calculation is failed.")
            return
        self.calc_IRCpath()
        print("IRC calculation is finished.")
        return

