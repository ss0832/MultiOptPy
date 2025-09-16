import numpy as np
import copy

from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass


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
        self.timestep = None
        
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
        print("NVT ensemble (Nose_Hoover) : Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
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
        print("NVT ensemble (Nose_Hoover chain) : Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
        
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

    def Velocity_Verlet(self, geom_num_list, new_g, prev_g, iter):#NVE ensemble 
        tmp_new_g = copy.copy(-1*new_g)
        tmp_prev_g = copy.copy(-1*prev_g)
        #print("NVE ensemble (Velocity_Verlet)")
        self.momentum_list += ( tmp_new_g + tmp_prev_g ) * (self.delta_timescale) * 0.5 #+ third_term + fourth_term
        #-----------
        tmp_list = []
        for i, elem in enumerate(self.element_list):
            tmp_list.append(self.delta_timescale * self.momentum_list[i] / atomic_mass(elem) + self.delta_timescale ** 2 * tmp_new_g[i] / (2.0 * atomic_mass(elem)))
        new_geometry = geom_num_list + tmp_list
        #------------    
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        #-------------
 
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
    
    
    def init_purtubation(self, geometry, B_e, B_g):
        random_variables = self.generate_normal_random_variables(len(self.element_list)*3).reshape(len(self.element_list), 3)
        
        addtional_velocity = self.calc_rand_moment_based_on_boltzman_const(random_variables) # velocity
        init_momentum = addtional_velocity * 0.0
            
        for i in range(len(self.element_list)):
            init_momentum[i] += addtional_velocity[i] * atomic_mass(self.element_list[i])
        
        
        self.momentum_list += init_momentum
        self.init_energy = B_e
        #self.init_hamiltonian = B_e
        #for i, elem in enumerate(element_list):
        #    self.init_hamiltonian += (np.sum(self.momentum_list[i]) ** 2 / (2 * atomic_mass(elem)))
        return
