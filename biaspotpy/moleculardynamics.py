import sys
import os
import shutil
import copy
import glob
import itertools
import datetime
import time

import numpy as np

from optimizer import CalculateMoveVector
from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib, element_number, atomic_mass
from interface import force_data_parser
from approx_hessian import ApproxHessian
from cmds_analysis import CMDSPathAnalysis
from constraint_condition import shake_parser, SHAKE


class Thermostat:
    def __init__(self, momentum_list, temperature, pressure, element_list=[]):

        self.momentum_list = momentum_list #list
        
        self.temperature = temperature #K
        self.initial_temperature = temperature #K
        self.pressure = pressure * (3.39893 * 10 ** (-11)) #kPa -> a.u.
        self.initial_pressure = pressure * (3.39893 * 10 ** (-11)) #kPa -> a.u.

        self.Langevin_zeta = 0.01
        self.zeta = 0.0
        self.eta = 0.0
        self.scaling = 1.0
        self.Ps_momentum = 0.0
        
        self.g_value = len(momentum_list) * 3
        self.Q_value = 1e-1
        
        
        self.M_value = 1e+12
        self.Boltzmann_constant = 3.16681 * 10 ** (-6) # hartree/K
        self.delta_timescale = 1e-1
        self.volume = 1e-23 * (1/UnitValueLib().bohr2m) ** 3#m^3 -> Bohr^3
        
        # Nose-Hoover-chain
        self.Q_value_chain = [1.0, 2.0, 3.0, 6.0, 10.0, 20, 40, 50, 100, 200]#mass of thermostat 
        self.zeta_chain = [0.0 for i in range(len(self.Q_value_chain))]
        
        
        self.Instantaneous_temperatures_list = []
        self.Instantaneous_momentum_list = []
        
        self.element_list = element_list
        return
    
    
    def calc_tot_kinetic_energy(self):
        tot_kinetic_ene = 0.0
        
        for i, elem in enumerate(self.element_list):
            tot_kinetic_ene += (np.sum(self.momentum_list[i] ** 2) /(2 * atomic_mass(elem)))
        self.tot_kinetic_ene = tot_kinetic_ene
        return tot_kinetic_ene
    
    def calc_inst_temperature(self):
        #temperature
        tot_kinetic_ene = self.calc_tot_kinetic_energy()
        Instantaneous_temperature = 2 * tot_kinetic_ene / (self.g_value * self.Boltzmann_constant)
        print("Instantaneous_temperature: ",Instantaneous_temperature ," K")

        self.Instantaneous_temperature = Instantaneous_temperature
        #-----------------
        return Instantaneous_temperature
    
    def add_inst_temperature_list(self):
        #self.add_inst_temperature_list()
        self.Instantaneous_temperatures_list.append(self.Instantaneous_temperature)
        
    
    def Nose_Hoover_thermostat(self, geom_num_list, new_g):#fixed volume #NVT ensemble
        new_g *= -1
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta * 0.5)

        self.momentum_list += new_g * self.delta_timescale * 0.5
        print("Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem))
        
        new_geometry = geom_num_list + tmp_list
        #------------
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #----------
        self.zeta += self.delta_timescale * (2 * self.tot_kinetic_ene - self.g_value * self.Boltzmann_constant * self.initial_temperature) / self.Q_value
        
        #print(tmp_value, self.g_value * self.Boltzmann_constant * self.temperature)
        
        
        self.momentum_list += new_g * self.delta_timescale * 0.5
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta * 0.5)
        
        
        return new_geometry

    def Nose_Hoover_chain_thermostat(self, geom_num_list, new_g):#fixed volume #NVT ensemble
        #ref. J. Chem. Phys. 97, 2635-2643 (1992)
        new_g *= -1
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta_chain[0] * 0.5)

        self.momentum_list += new_g * self.delta_timescale * 0.5
        print("Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
        
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem))
        
        new_geometry = geom_num_list + tmp_list
        #------------
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #----------
        self.zeta_chain[0] += self.delta_timescale * (2 * self.tot_kinetic_ene - self.g_value * self.Boltzmann_constant * self.initial_temperature) / self.Q_value_chain[0] -1* self.delta_timescale * (self.zeta_chain[0] * self.zeta_chain[1])
        
        for j in range(1, len(self.zeta_chain)-1):
            self.zeta_chain[j] += self.delta_timescale * (self.Q_value_chain[j-1]*self.zeta_chain[j-1]**2 - self.Boltzmann_constant * self.initial_temperature) / self.Q_value_chain[j] -1* self.delta_timescale * (self.zeta_chain[j] * self.zeta_chain[j+1])
        
        self.zeta_chain[-1] += self.delta_timescale * (self.Q_value_chain[-2]*self.zeta_chain[-2]**2 -1*self.Boltzmann_constant * self.initial_temperature) / self.Q_value_chain[-1]
        
        #print(tmp_value, self.g_value * self.Boltzmann_constant * self.temperature)
    
        
        self.momentum_list += new_g * self.delta_timescale * 0.5
        self.momentum_list = self.momentum_list * np.exp(-self.delta_timescale * self.zeta_chain[0] * 0.5)
        print("zeta_list (Coefficient of friction): ", self.zeta_chain)    
        return new_geometry

    def Velocity_Verlet(self, geom_num_list, new_g, iter):#NVE ensemble 
        new_g *= -1
        if iter != 0:
            self.momentum_list += new_g*(self.delta_timescale)
        else:
            self.momentum_list += new_g*(self.delta_timescale*0.5)
        #-----------
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem))
        new_geometry = geom_num_list + tmp_list
        #------------    
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #-------------
        
        #self.momentum_list += new_g*(self.delta_timescale*0.5)
        
        return new_geometry
    
    
    def Nose_Hoover_Andersen_method(self, geom_num_list, new_g, energy, iter):
        # NPT ensemble (isotropic pressure) ### This implementation is wrong.
        # ref. J. Phys. Chem. A, 2019, 123, 1689-1699 
        if iter == 0:#initialization
            self.factorization_list = [(1/(2 * (2-2**(1/3)))), (1/(2-2**(1/3))), ((1-2**(1/3))/(2 * (2-2**(1/3)))), ((-2**(1/3))/(2 * (2-2**(1/3)))), ((1-2**(1/3))/(2 * (2-2**(1/3)))), (1/(2-2**(1/3))), (1/(2 * (2-2**(1/3))))]
            self.tau_thermo = 1
            self.tau_baro = 10
            self.thermo_eta_Q = 3 * len(self.element_list) * self.tau_thermo ** 2 * self.Boltzmann_constant * self.initial_temperature
            self.thermo_zeta_Q = self.tau_thermo ** 2 * self.Boltzmann_constant * self.initial_temperature
            self.baro_W = len(self.element_list) * self.tau_baro ** 2 * self.Boltzmann_constant * self.initial_temperature / self.volume
            self.velocity_volume = 0.0
            self.velocity_eta = 0.0
            self.velocity_zeta = 0.0
        
        G_eta = self.calc_tot_kinetic_energy() * 2 -3 * len(self.element_list) * self.initial_temperature * self.Boltzmann_constant
        
        print("G_eta :", G_eta)
        #5 6
        for j in range(3):
            self.velocity_eta += (self.factorization_list[2*j+0] * self.delta_timescale/2) * G_eta / self.thermo_eta_Q
            self.eta += (self.factorization_list[2*j+1] * self.delta_timescale/2) * self.velocity_eta
            self.momentum_list *= np.exp(-1*(self.factorization_list[2*j+1] * self.delta_timescale/2) * self.velocity_eta)
        self.velocity_eta += (self.factorization_list[6] * self.delta_timescale/2) * G_eta / self.thermo_eta_Q 
        #3 4
        
        G_zeta = self.baro_W * self.velocity_volume ** 2 -1*self.initial_temperature * self.Boltzmann_constant
        print("G_zeta :", G_zeta)
        tmp_value = 0.0
        for i, elem in enumerate(self.element_list):
            tmp_value += np.sum(geom_num_list[i] * new_g[i])
            
        self.pressure = (1 / (3 * self.volume)) * (tmp_value + self.calc_tot_kinetic_energy() * 2)
        delta_pressure = self.pressure - self.initial_pressure 
        
        print("Pressure (kPa):", self.pressure * (1/ (3.39893 * 10 ** (-11))))
        
        
        for j in range(3):
            self.velocity_zeta += (self.factorization_list[2*j+0] * self.delta_timescale/2) * G_zeta / self.thermo_zeta_Q
            self.zeta += (self.factorization_list[2*j+1] * self.delta_timescale/2) * self.velocity_zeta
            self.velocity_volume = self.velocity_volume * np.exp(-1*self.factorization_list[2*j+1] * self.delta_timescale / 2 * self.velocity_zeta) + delta_pressure / self.baro_W / self.velocity_zeta * (1-np.exp(-1*self.factorization_list[2*j+1] * self.delta_timescale / 2 * self.velocity_zeta))
        self.velocity_zeta += (self.factorization_list[6] * self.delta_timescale/2) * G_zeta / self.thermo_zeta_Q 
         
        # 2
        velocity_eps = self.velocity_volume / (3 * self.volume)
        self.momentum_list = self.momentum_list*np.exp(velocity_eps/2 * self.delta_timescale) + new_g / velocity_eps * (1.0 - np.exp(-1*velocity_eps/2 * self.delta_timescale))
        #self.momentum_list = self.momentum_list + new_g * self.delta_timescale / 2
        # 1
        velocity = []
        for i in range(len(self.element_list)): 
            velocity.append(self.momentum_list[i]/atomic_mass(self.element_list[i]))
        velocity = np.array(velocity, dtype="float64")
        new_geometry = geom_num_list * np.exp(-1*velocity_eps * self.delta_timescale) + velocity / velocity_eps * (np.exp(self.delta_timescale * velocity_eps) - 1.0)
        #new_geometry = geom_num_list + velocity * self.delta_timescale
        
        self.volume += self.delta_timescale * self.velocity_volume
        # 2
        velocity_eps = self.velocity_volume / (3 * self.volume)
        self.momentum_list = self.momentum_list*np.exp(-1*velocity_eps/2 * self.delta_timescale) + new_g / velocity_eps * (1.0 - np.exp(-1*velocity_eps/2 * self.delta_timescale))
        #self.momentum_list = self.momentum_list + new_g * self.delta_timescale / 2
        
        #3 4
       
        G_zeta = self.baro_W * self.velocity_volume ** 2 -1*self.initial_temperature * self.Boltzmann_constant
        print("G_zeta :", G_zeta)
        tmp_value = 0.0
        for i, elem in enumerate(self.element_list):
            tmp_value += np.sum(geom_num_list[i] * new_g[i])
            
        self.pressure = (1 / (3 * self.volume)) * (tmp_value + self.calc_tot_kinetic_energy() * 2)
        delta_pressure = self.pressure - self.initial_pressure 
        print("Pressure (kPa) :", self.pressure * (1/ (3.39893 * 10 ** (-11))))
        
        for j in range(3):
            self.velocity_zeta += (self.factorization_list[2*j+0] * self.delta_timescale/2) * G_zeta / self.thermo_zeta_Q
            self.zeta += (self.factorization_list[2*j+1] * self.delta_timescale/2) * self.velocity_zeta
            self.velocity_volume = self.velocity_volume * np.exp(-1*self.factorization_list[2*j+1] * self.delta_timescale / 2 * self.velocity_zeta) + delta_pressure / self.baro_W / self.velocity_zeta * (1-np.exp(-1*self.factorization_list[2*j+1] * self.delta_timescale / 2 * self.velocity_zeta))
        self.velocity_zeta += (self.factorization_list[6] * self.delta_timescale/2) * G_zeta / self.thermo_zeta_Q 
         

        #5 6 
       
        G_eta = self.calc_tot_kinetic_energy() * 2 -3 * len(self.element_list) * self.initial_temperature * self.Boltzmann_constant
        print("G_eta :", G_eta)
        for j in range(3):
            self.velocity_eta += (self.factorization_list[2*j+0] * self.delta_timescale/2) * G_eta / self.thermo_eta_Q
            self.eta += (self.factorization_list[2*j+1] * self.delta_timescale/2) * self.velocity_eta
            self.momentum_list *= np.exp(-1*(self.factorization_list[2*j+1] * self.delta_timescale/2) * self.velocity_eta)
        self.velocity_eta += (self.factorization_list[6] * self.delta_timescale/2) * G_eta / self.thermo_eta_Q 
        print("velocity_eta, velocity_zeta :", self.velocity_eta, self.velocity_zeta)
        print("volume (m^3)", self.volume * UnitValueLib().bohr2m**3)
        print("velocity volume", self.velocity_volume)
        inst_temperature = self.calc_inst_temperature()
        self.add_inst_temperature_list()
        return new_geometry

    def Nose_Poincare_Andarsen_method(self, geom_num_list, new_g, energy, iter):
        # NPT ensemble (isotropic pressure) ### this implementation may not be correct. ###
        # (Comments) We may be able to get a trajectory that we can use to search for conformation. 
        new_g *= -1
        time_and_volume_scaling_momentum_list = self.scaling * abs(self.volume) ** (1/3) * self.momentum_list
        
        scaled_geom_num_list = abs(self.volume) ** (-1/3) * geom_num_list 
        
        if iter == 0:
            tmp_value = 0.0
            
            for i in range(len(time_and_volume_scaling_momentum_list)):
                tmp_value += np.sum(time_and_volume_scaling_momentum_list[i] ** 2) / (2 *  atomic_mass(self.element_list[i]) * self.scaling ** 2 + self.volume ** (2/3))
                
            self.init_hamiltonian = self.scaling * (tmp_value + (energy ) + self.Ps_momentum ** 2 / (2 * self.Q_value) + self.g_value * self.Boltzmann_constant * self.initial_temperature * np.log(self.scaling) + self.pressure ** 2 / (2 * self.M_value) + self.initial_pressure * self.volume) 
        
        #------------------
        self.scaling *= (1.0 + (self.Ps_momentum/(2 * self.Q_value))*(self.delta_timescale / 2))
        self.Ps_momentum = self.Ps_momentum / (1.0 + (self.Ps_momentum/(2 * self.Q_value))*(self.delta_timescale / 2))
        #------------------
        time_and_volume_scaling_momentum_list += self.scaling * abs(self.volume) ** (1/3) * new_g * 0.5 * self.delta_timescale
        self.Ps_momentum -= ((energy) + self.initial_pressure * abs(self.volume)) * 0.5 * self.delta_timescale  
        
        tmp_value = 0.0
        tmp_value_2 = 0.0
        for i, elem in enumerate(self.element_list):
            tmp_value += np.sum(geom_num_list[i] * -1*new_g[i])
            tmp_value_2 += np.sum(time_and_volume_scaling_momentum_list[i] / (atomic_mass(self.element_list[i]) * self.scaling ** 2 * self.volume ** (2/3)))
           
        self.pressure += self.scaling * ((1 / (3 * self.volume + 1e-10)) * (tmp_value + 0) - (self.initial_pressure)) * self.delta_timescale * 0.5 
        
        print((1 / (3 * self.volume + 1e-10)) * tmp_value,  self.initial_pressure)
        #raise
        #------------------
        self.Ps_momentum -= (self.pressure ** 2)/(2*self.M_value) * (self.delta_timescale/2)
        self.volume += self.scaling * (self.pressure/self.M_value) * (self.delta_timescale/2)
        
        #------------------
        tmp_list = []
        tmp_value_2 = 0.0
        for i in range(len(time_and_volume_scaling_momentum_list)):
            tmp_list.append(time_and_volume_scaling_momentum_list[i]/atomic_mass(self.element_list[i]))
            tmp_value_2 += np.sum(time_and_volume_scaling_momentum_list[i]**2/atomic_mass(self.element_list[i]))
        tmp_list = np.array(tmp_list, dtype="float64")
        scaled_geom_num_list += tmp_list / (self.scaling * abs(self.volume) ** (2/3)) * self.delta_timescale
        self.Ps_momentum += (tmp_value_2 / (2*self.scaling ** 2 *abs(self.volume)**(2/3)) -1 * self.g_value * self.Boltzmann_constant * self.initial_temperature * np.log(abs(self.scaling)) + self.init_hamiltonian -1*self.g_value * self.Boltzmann_constant * self.initial_temperature ) * self.delta_timescale
        self.pressure += tmp_value_2 / (3*self.scaling*abs(self.volume)**(5/3)) * self.delta_timescale
        
        #------------------
        self.Ps_momentum -= (self.pressure ** 2)/(2*self.M_value) * (self.delta_timescale/2)
        self.volume += self.scaling * (self.pressure/self.M_value) * (self.delta_timescale/2)
        #-----------------
        time_and_volume_scaling_momentum_list += self.scaling * abs(self.volume) ** (1/3) * new_g * 0.5 * self.delta_timescale
        self.Ps_momentum -= ((energy ) + self.initial_pressure * self.volume) * 0.5 * self.delta_timescale  
        
        tmp_value = 0.0
        
        for i, elem in enumerate(self.element_list):
            tmp_value += np.sum(geom_num_list[i] * -1*new_g[i])
        self.pressure += self.scaling * ((1 / (3 * self.volume + 1e-10)) * tmp_value - self.initial_pressure) * self.delta_timescale * 0.5 
        #-----------------
        self.scaling *= (1.0 + (self.Ps_momentum/(2 * self.Q_value))*(self.delta_timescale / 2))
        self.Ps_momentum = self.Ps_momentum / (1.0 + (self.Ps_momentum/(2 * self.Q_value))*(self.delta_timescale / 2))
        
        #------------------
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        print("Ps_momentum :", self.Ps_momentum)
        print("time scaling :", self.scaling)
        print("Volume (m^3): ", self.volume * (UnitValueLib().bohr2m ** 3))
        print("Pressure : ", self.pressure)#) * (1 / (3.39893 * 10 ** (-11))))

        #--------------
        
        new_geometry = scaled_geom_num_list * abs(self.volume) ** (1/3)
        self.momentum_list = time_and_volume_scaling_momentum_list * (1/self.scaling) * abs(self.volume) ** (-1/3)
       
        return new_geometry

    
    def Langevin_equation(self, geom_num_list, new_g, pre_g, iter):
        #self.Langevin_zeta
        #Allen argorithum
        C_0 = np.exp(-1 * self.Langevin_zeta * self.delta_timescale)
        C_1 = (1 - C_0) / (self.Langevin_zeta * self.delta_timescale)
        C_2 = (1 - C_1) / (self.Langevin_zeta * self.delta_timescale)
        for i in range(len(self.element_list)):
            self.momentum_list[i] += C_2 * (-1) * pre_g[i] * self.delta_timescale
        
        delta_coord_Gaussian = self.generate_normal_random_variables(len(self.element_list)*3).reshape(len(self.element_list), 3)
        delta_momentum_Gaussian = self.generate_normal_random_variables(len(self.element_list)*3).reshape(len(self.element_list), 3)
        
        for i in range(len(self.element_list)):
            
            delta_coord_Gaussian[i] *= np.sqrt(self.delta_timescale * ((self.Boltzmann_constant * self.temperature) / (self.Langevin_zeta * atomic_mass(self.element_list[i]))) * (2.0 -1 * (1/(self.Langevin_zeta * self.delta_timescale)) * (3.0 -4 * np.exp(-1 * self.Langevin_zeta * self.delta_timescale) + np.exp(-1 * self.Langevin_zeta * self.delta_timescale * 2))))
            delta_momentum_Gaussian *= np.sqrt((atomic_mass(self.element_list[i]) * self.Boltzmann_constant * self.temperature) * (1.0 -np.exp(-1 * self.Langevin_zeta * self.delta_timescale * 2)))
        
        new_geometry = copy.copy(geom_num_list)
        for i in range(len(self.element_list)):
            new_geometry[i] += C_1 * (self.momentum_list[i] / atomic_mass(self.element_list[i])) * self.delta_timescale + C_2 * ((-1) *new_g[i] / atomic_mass(self.element_list[i])) * self.delta_timescale ** 2 + delta_coord_Gaussian[i]
            
            self.momentum_list[i] *= C_0 
            self.momentum_list[i] += (C_1 - C_2) * (-1) * new_g[i] * self.delta_timescale + delta_momentum_Gaussian[i]
        
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        return new_geometry
    
    
    def generate_normal_random_variables(self, n_variables):
        random_variables = []
        for _ in range(n_variables // 2):
            u1, u2 = np.random.rand(2)
            #Box-Muller method
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            random_variables.extend([z1, z2])
        
        if n_variables % 2 == 1:
            u1, u2 = np.random.rand(2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            random_variables.append(z1)
        
        return np.array([random_variables], dtype="float64")

    def calc_rand_moment_based_on_boltzman_const(self, random_variables):
        rand_moment = random_variables
        
        for i in range(len(self.element_list)):
            random_variables[i] *= np.sqrt(self.Boltzmann_constant * self.temperature / atomic_mass(self.element_list[i]))
        
        
        return rand_moment
    
    def calc_DRC_init_momentum(self, coord, cart_gradient):
        #dynamic reaction coordinate (DRC)
        #ref. J. Chem. Phys. 93, 5902–5911 (1990) etc.
        AH = ApproxHessian()
        hess = AH.main(coord, self.element_list, cart_gradient)
 
        eigenvalue, eigenvector = np.linalg.eig(hess)
        
        drc_momentum = eigenvector.T[0].reshape(len(self.element_list), 3)
        for i in range(len(self.element_list)):
            drc_momentum[i] *= atomic_mass(self.element_list[i])

        return drc_momentum
    
    def init_purtubation(self, geometry, B_e, B_g):
        random_variables = self.generate_normal_random_variables(len(self.element_list)*3).reshape(len(self.element_list), 3)
        random_moment_flag = True
        
        if random_moment_flag:
            addtional_velocity = self.calc_rand_moment_based_on_boltzman_const(random_variables) # velocity
            init_momentum = addtional_velocity * 0.0
            
            for i in range(len(self.element_list)):
                init_momentum[i] += addtional_velocity[i] * atomic_mass(self.element_list[i])
        else:
            init_momentum = self.calc_DRC_init_momentum(geometry, B_g)
        
        
        self.momentum_list += init_momentum
        self.init_energy = B_e
        #self.init_hamiltonian = B_e
        #for i, elem in enumerate(element_list):
        #    self.init_hamiltonian += (np.sum(self.momentum_list[i]) ** 2 / (2 * atomic_mass(elem)))
        return


class MD:
    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.Boltzmann_constant = 3.16681 * 10 ** (-6) # hartree/K
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #
        self.time_atom_unit = 2.419 * 10 ** (-17)
        self.args = args #
        self.FC_COUNT = -1 # 
        self.excited_state = args.excited_state
        self.initial_temperature = args.temperature
        self.num_of_trajectory = args.TRAJECTORY
        self.change_temperature = args.change_temperature
        self.momentum_list = None
        self.initial_pressure = args.pressure

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT #
        self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        
        if args.pyscf:
            self.electronic_charge = args.electronic_charge
            self.spin_multiplicity = args.spin_multiplicity
            self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET) # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2*j]] = args.sub_basisset[2*j+1]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            else:
                self.SUB_BASIS_SET = { "default" : self.BASIS_SET}
        else:#psi4
            self.SUB_BASIS_SET = "" # 
            self.electric_charge_and_multiplicity = [int(args.electronic_charge), int(args.spin_multiplicity)]
            
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET +="\nassign "+str(self.BASIS_SET)+"\n" # 
                for j in range(int(len(args.sub_basisset)/2)):
                    self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET) #
            
        #-----------------------------

        self.mdtype = args.mdtype
        self.Model_hess = None #
        self.Opt_params = None #
        self.DC_check_dist = 10.0#ang.
        self.unrestrict = args.unrestrict
        if len(args.constraint_condition) > 0:
            self.constraint_condition_list = shake_parser(args.constraint_condition)
        else:
            self.constraint_condition_list = []
        self.args = args
        return
    


    def exec_md(self, TM, geom_num_list, prev_geom_num_list, B_g, B_e, pre_B_g, iter):
        if iter == 0 and len(self.constraint_condition_list) > 0:
            self.class_SHAKE = SHAKE(TM.delta_timescale, self.constraint_condition_list)
        if self.mdtype == "nosehoover": 
            new_geometry = TM.Nose_Hoover_thermostat(geom_num_list, B_g)
        elif self.mdtype == "nosehooverchain": 
            new_geometry = TM.Nose_Hoover_chain_thermostat(geom_num_list, B_g)
        elif self.mdtype == "velocityverlet":
            new_geometry = TM.Velocity_Verlet(geom_num_list, B_g, iter)
            
        elif self.mdtype == "nosepoincareandersen":
            new_geometry = TM.Nose_Poincare_Andarsen_method(geom_num_list, B_g, B_e, iter)    
        #elif self.mdtype == "nosehooverandersen":
        #    new_geometry = TM.Nose_Hoover_Andersen_method( geom_num_list, B_g, B_e, iter)
        elif self.mdtype == "langevin":
            new_geometry = TM.Langevin_equation(geom_num_list, B_g, pre_B_g, iter)
        else:
            print("Unexpected method.", self.mdtype)
            raise
        
        if iter > 0 and len(self.constraint_condition_list) > 0:
            
            new_geometry, tmp_momentum_list = self.class_SHAKE.run(new_geometry, prev_geom_num_list, TM.momentum_list, TM.element_list)
            TM.momentum_list = copy.copy(tmp_momentum_list)

        #tmp_value = 0.0
        
        #for i in range(len(geom_num_list)):
        #    tmp_value += np.sum(TM.momentum_list[i] ** 2) / (2 * atomic_mass(element_list[i]))
        
        #hamiltonian = B_e + tmp_value
        #print("hamiltonian :", hamiltonian)
        
        return new_geometry

    def md_tblite(self):
        from tblite_calculation_tools import Calculation
        
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.args.usextb+"_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        temperature_list = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        initial_geom_num_list = np.zeros((len(element_list), 3))
        
        self.Model_hess = np.eye(len(element_list*3))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT, self.BPA_FOLDER_DIRECTORY)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         unrestrict = self.unrestrict)
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure, element_list=element_list)
        #-----------------------------------
        element_number_list = []
        for elem in element_list:
            element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype="int")
        #----------------------------------
        
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []

        #----------------------------------
        ct_count = 0
        
        
        for iter in range(self.NSTEP):
            #-----------------------------
            if ct_count < len(self.change_temperature):
                if int(self.change_temperature[ct_count]) == iter:
                    TM.initial_temperature = float(self.change_temperature[ct_count+1])
                    ct_count += 2
                    
            
            #------------------------------
            
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# STEP. "+str(iter)+" ("+str(TM.delta_timescale * iter * self.time_atom_unit * 10 ** 15)+" fs)\n")
            #---------------------------------------
            
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_number_list, iter, electric_charge_and_multiplicity, force_data["xtb"])
            self.Model_hess = SP.Model_hess
            if finish_frag:
                break
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            

             
            #---------------------------------------
            if iter == 0:
                TM.init_purtubation(geom_num_list, B_e, B_g)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            #----------------------------
            #----------------------------
            
            new_geometry = self.exec_md(TM, geom_num_list, pre_geom, B_g, B_e, pre_B_g, iter)

            if iter % 10 != 0:
                shutil.rmtree(file_directory)
            

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
            #----------------------------

            #----------------------------
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
        
        
        
        #----------------------
        print("Complete...")
        return

    def md_psi4(self):
        from psi4_calculation_tools import Calculation
        
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        initial_geom_num_list = np.zeros((len(element_list), 3))
        
        self.Model_hess = np.eye(len(element_list*3))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT, self.BPA_FOLDER_DIRECTORY)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         unrestrict = self.unrestrict,
                         software_type = self.args.othersoft,
                         excited_state = self.excited_state)
        #----------------------------------
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure, element_list=element_list)
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        ct_count = 0
        
        #----------------------------------
        for iter in range(self.NSTEP):
            
            if ct_count < len(self.change_temperature):
                if int(self.change_temperature[ct_count]) == iter:
                    TM.initial_temperature = float(self.change_temperature[ct_count+1])
                    ct_count += 2
                    
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_list, iter, electric_charge_and_multiplicity)
            self.Model_hess = SP.Model_hess
            if finish_frag:
                break
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
                         
            #---------------------------------------
            if iter == 0:
                TM.init_purtubation(geom_num_list, B_e, B_g)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass
            

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            

            #----------------------------

            #----------------------------
            
            new_geometry = self.exec_md(TM, geom_num_list, pre_geom, B_g, B_e, pre_B_g, iter)
        
            if iter % 10 != 0:
                shutil.rmtree(file_directory)
            

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
    
    
    def md_ase(self):
        from ase_calculation_tools import Calculation

        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_ASE_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        print("Use", self.args.othersoft)
        with open(self.BPA_FOLDER_DIRECTORY+"use_"+self.args.othersoft+".txt", "w") as f:
            f.write(self.args.othersoft+"\n")
            f.write(self.BASIS_SET+"\n")
            f.write(self.FUNCTIONAL+"\n")
        

        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        initial_geom_num_list = np.zeros((len(element_list), 3))
        
        self.Model_hess = np.eye(len(element_list*3))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT, self.BPA_FOLDER_DIRECTORY)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         unrestrict = self.unrestrict,
                         software_type = self.args.othersoft)
        #----------------------------------
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure, element_list=element_list)
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        ct_count = 0
        
        #----------------------------------
        for iter in range(self.NSTEP):
            
            if ct_count < len(self.change_temperature):
                if int(self.change_temperature[ct_count]) == iter:
                    TM.initial_temperature = float(self.change_temperature[ct_count+1])
                    ct_count += 2
                    
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# STEP. "+str(iter)+" ("+str(TM.delta_timescale * iter * self.time_atom_unit * 10 ** 15)+" fs)\n")
            
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_list, iter, electric_charge_and_multiplicity, method="")
            self.Model_hess = SP.Model_hess
            if finish_frag:
                break
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
                         
            #---------------------------------------
            if iter == 0:
                TM.init_purtubation(geom_num_list, B_e, B_g)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass
            

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            

            #----------------------------

            #----------------------------
            
            new_geometry = self.exec_md(TM, geom_num_list, pre_geom, B_g, B_e, pre_B_g, iter)
        
            if iter % 10 != 0:
                shutil.rmtree(file_directory)
            

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2(new_geometry*self.bohr2angstroms, element_list, electric_charge_and_multiplicity)
            file_directory = FIO.make_psi4_input_file(geometry_list, iter+1)
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
    
    def md_pyscf(self):
        from pyscf_calculation_tools import Calculation
        
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_MD_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        FIO = FileIO(self.BPA_FOLDER_DIRECTORY, self.START_FILE)
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)
        self.NUM_LIST = []
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = []
        force_data = force_data_parser(self.args)
        finish_frag = False
        
        geometry_list, element_list, electric_charge_and_multiplicity = FIO.make_geometry_list(self.electric_charge_and_multiplicity)
        file_directory = FIO.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        self.momentum_list = np.zeros((len(element_list), 3))
        initial_geom_num_list = np.zeros((len(element_list), 3))
       
        self.Model_hess = np.eye(len(element_list*3))
         
        CalcBiaspot = BiasPotentialCalculation(self.Model_hess, self.FC_COUNT, self.BPA_FOLDER_DIRECTORY)
        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(vars(self.args)))
        pre_B_e = 0.0
        pre_e = 0.0
        pre_B_g = []
        pre_g = []
        for i in range(len(element_list)):
            pre_B_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_B_g
        pre_g = pre_B_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        SP = Calculation(START_FILE = self.START_FILE,
                         SUB_BASIS_SET = self.SUB_BASIS_SET,
                         BASIS_SET = self.BASIS_SET,
                         N_THREAD = self.N_THREAD,
                         SET_MEMORY = self.SET_MEMORY ,
                         FUNCTIONAL = self.FUNCTIONAL,
                         FC_COUNT = self.FC_COUNT,
                         BPA_FOLDER_DIRECTORY = self.BPA_FOLDER_DIRECTORY,
                         Model_hess = self.Model_hess,
                         spin_multiplicity = self.spin_multiplicity,
                         electronic_charge = self.electronic_charge,
                         unrestrict = self.unrestrict,
                         excited_state = self.excited_state)
        #----------------------------------
        TM = Thermostat(self.momentum_list, self.initial_temperature, self.initial_pressure, element_list=element_list)
        cos_list = [[] for i in range(len(force_data["geom_info"]))]
        grad_list = []
        ct_count = 0
        
        #----------------------------------
        for iter in range(self.NSTEP):
            if ct_count < len(self.change_temperature):
                if int(self.change_temperature[ct_count]) == iter:
                    TM.initial_temperature = float(self.change_temperature[ct_count+1])
                    ct_count += 2
                    
            exit_file_detect = os.path.exists(self.BPA_FOLDER_DIRECTORY+"end.txt")

            if exit_file_detect:
                break
            print("\n# ITR. "+str(iter)+"\n")
            #---------------------------------------
            SP.Model_hess = self.Model_hess
            e, g, geom_num_list, finish_frag = SP.single_point(file_directory, element_list, iter)

            self.Model_hess = SP.Model_hess

            if finish_frag:
                break
            _, B_e, B_g, _ = CalcBiaspot.main(e, g, geom_num_list, element_list, force_data, pre_B_g, iter, initial_geom_num_list)#new_geometry:ang.
                 
            #---------------------------------------
            if iter == 0:
                TM.init_purtubation(geom_num_list, B_e, B_g)
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list    
            else:
                pass

            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------gradient profile
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                    f.write("gradient (RMS) [hartree/Bohr] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"gradient_profile.csv","a") as f:
                f.write(str(np.sqrt((g**2).mean()))+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            

            #----------------------------

            #----------------------------
            
            new_geometry = self.exec_md(TM, geom_num_list, pre_geom, B_g, B_e, pre_B_g, iter)
          
            if iter % 10 != 0:
                shutil.rmtree(file_directory)
            

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(B_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            
            #--------------------geometry info
            self.geom_info_extract(force_data, file_directory, B_g, g)    
            #----------------------------

            
            grad_list.append(np.sqrt((g**2).mean()))
           
            #-------------------------
            
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.copy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            
            #------------------------            

            #----------------------------
            pre_B_e = B_e#Hartree
            pre_e = e
            pre_B_g = B_g#Hartree/Bohr
            pre_g = g
            pre_geom = geom_num_list#Bohr
            
            geometry_list = FIO.make_geometry_list_2_for_pyscf(new_geometry*self.bohr2angstroms, element_list)
            file_directory = FIO.make_pyscf_input_file(geometry_list, iter+1)
        #plot graph
        G = Graph(self.BPA_FOLDER_DIRECTORY)
        G.double_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING)
        G.single_plot(self.NUM_LIST, grad_list, file_directory, "", axis_name_2="gradient (RMS) [a.u.]", name="gradient")
        G.single_plot(self.NUM_LIST, TM.Instantaneous_temperatures_list, file_directory, "", axis_name_2="temperature [K]", name="temperature")
        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                self.single_plot(self.NUM_LIST, cos_list[num], file_directory, i)
        
        #
        FIO.xyz_file_make_for_pyscf()
        
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")
        
        
        
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile_kcalmol.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
    

    def geom_info_extract(self, force_data, file_directory, B_g, g):
        if len(force_data["geom_info"]) > 1:
            CSI = CalculationStructInfo()
            
            data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
            
            for num, i in enumerate(force_data["geom_info"]):
                cos = CSI.calculate_cos(B_g[i-1] - g[i-1], g[i-1])
                self.cos_list[num].append(cos)
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                    f.write(",".join(data_name_list)+"\n")
            
            with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                f.write(",".join(list(map(str,data_list)))+"\n")                 
        return
    
    def run(self):
        
        for i in range(self.num_of_trajectory):
            print("trajectory :", i)
            if self.args.pyscf:
                self.md_pyscf()
            elif self.args.usextb != "None":
                self.md_tblite()
            elif self.args.othersoft != "None":
                self.md_ase()
            else:
                self.md_psi4()
    
        print("All complete...")
        
        return