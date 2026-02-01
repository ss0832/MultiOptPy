import numpy as np
import copy
from multioptpy.Parameters.parameter import UnitValueLib, atomic_mass

class Thermostat:
    def __init__(self, momentum_list, temperature, pressure, element_list=None):
        # Mutable default argument fix
        if element_list is None:
            self.element_list = []
        else:
            self.element_list = element_list

        # ---------------------------------------------------------
        # [Optimization] Pre-compute masses for vectorization
        # shape: (N_atoms, 1) to allow broadcasting: momentum (N,3) / mass (N,1)
        # ---------------------------------------------------------
        self.masses = np.array([atomic_mass(e) for e in self.element_list], dtype=np.float64)[:, None]
        self.inverse_masses = 1.0 / self.masses
        
        # Keep momentum as numpy array internally for performance
        self.momentum_list = np.array(momentum_list, dtype=np.float64)

        self.temperature = temperature # K
        self.initial_temperature = temperature # K
        
        # Pressure conversion
        self.pressure = pressure * (3.39893 * 10 ** (-11)) # kPa -> a.u.
        self.initial_pressure = self.pressure

        # Thermostat Parameters
        self.Langevin_zeta = 0.01
        self.zeta = 0.0
        self.eta = 0.0
        self.scaling = 1.0
        self.Ps_momentum = 0.0
        
        # Degrees of freedom (3N)
        self.g_value = len(momentum_list) * 3
        
        # Constants
        self.Q_value = 1e-1
        self.M_value = 1e+12
        self.Boltzmann_constant = 3.16681 * 10 ** (-6) # hartree/K
        self.delta_timescale = 1e-1
        self.volume = 1e-23 * (1/UnitValueLib().bohr2m) ** 3 # m^3 -> Bohr^3
        
        # Nose-Hoover Chain Parameters
        self.Q_value_chain = np.array([1.0, 2.0, 3.0, 6.0, 10.0, 20, 40, 50, 100, 200], dtype=np.float64)
        self.zeta_chain = np.zeros(len(self.Q_value_chain), dtype=np.float64)
        self.timestep = None
        
        # History
        self.Instantaneous_temperatures_list = []
        self.Instantaneous_momentum_list = []
        self.tot_kinetic_ene = 0.0
        self.Instantaneous_temperature = 0.0

    # =========================================================================
    # Internal Helpers (Vectorized & Logical Separation)
    # =========================================================================

    def _update_momentum(self, forces, scaling=1.0):
        """Vectorized momentum update"""
        pass 

    def _update_position(self, current_geometry_arr, dt):
        """Vectorized position update: r(t+dt) = r(t) + v(t)*dt"""
        # v = p / m
        velocities = self.momentum_list * self.inverse_masses
        return current_geometry_arr + velocities * dt

    def _propagate_nhc_zeta(self, dt, kinetic_energy_2x):
        """
        Propagate Nose-Hoover Chain variables.
        """
        # 1. Update first chain link force
        driving_force = (kinetic_energy_2x - self.g_value * self.Boltzmann_constant * self.initial_temperature)
        
        self.zeta_chain[0] += dt * driving_force / self.Q_value_chain[0] 
        self.zeta_chain[0] -= dt * self.zeta_chain[0] * self.zeta_chain[1] # Coupling with next

        # 2. Update middle chain links
        for j in range(1, len(self.zeta_chain)-1):
            driving_force_j = (self.Q_value_chain[j-1] * self.zeta_chain[j-1]**2 - self.Boltzmann_constant * self.initial_temperature)
            self.zeta_chain[j] += dt * driving_force_j / self.Q_value_chain[j]
            self.zeta_chain[j] -= dt * self.zeta_chain[j] * self.zeta_chain[j+1]

        # 3. Update last chain link
        last = -1
        driving_force_last = (self.Q_value_chain[last-1] * self.zeta_chain[last-1]**2 - self.Boltzmann_constant * self.initial_temperature)
        self.zeta_chain[last] += dt * driving_force_last / self.Q_value_chain[last]

    # =========================================================================
    # Public API (Compatible with moleculardynamics.py)
    # =========================================================================

    def calc_tot_kinetic_energy(self):
        """
        Vectorized calculation of total kinetic energy.
        KE = sum(p^2 / 2m)
        """
        p_sq = self.momentum_list ** 2
        p_sq_atom = np.sum(p_sq, axis=1)
        self.tot_kinetic_ene = np.sum(p_sq_atom / (2.0 * self.masses.flatten()))
        return self.tot_kinetic_ene
    
    def calc_inst_temperature(self):
        """Calculates and stores instantaneous temperature."""
        self.calc_tot_kinetic_energy()
        self.Instantaneous_temperature = 2.0 * self.tot_kinetic_ene / (self.g_value * self.Boltzmann_constant)
        print(f"Instantaneous_temperature: {self.Instantaneous_temperature:.8f} K")
        return self.Instantaneous_temperature
    
    def add_inst_temperature_list(self):
        self.Instantaneous_temperatures_list.append(self.Instantaneous_temperature)

    def Nose_Hoover_thermostat(self, geom_num_list, new_g): # fixed volume #NVT ensemble
        """
        Single Nose-Hoover implementation.
        """
        geom_arr = np.array(geom_num_list, dtype=np.float64)
        force = -1.0 * np.array(new_g, dtype=np.float64) 

        # 1. First half-step thermostat scaling
        self.momentum_list *= np.exp(-self.delta_timescale * self.zeta * 0.5)

        # 2. First half-step momentum update (Force contribution)
        self.momentum_list += force * self.delta_timescale * 0.5
        
        print("NVT ensemble (Nose_Hoover) : Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))

        # 3. Position update (Full step)
        new_geometry = self._update_position(geom_arr, self.delta_timescale)

        # 4. Thermostat Propagation
        self.calc_inst_temperature()
        self.add_inst_temperature_list()

        driving_force = (2 * self.tot_kinetic_ene - self.g_value * self.Boltzmann_constant * self.initial_temperature)
        self.zeta += self.delta_timescale * driving_force / self.Q_value

        # 5. Second half-step momentum update
        self.momentum_list += force * self.delta_timescale * 0.5

        # 6. Second half-step thermostat scaling
        self.momentum_list *= np.exp(-self.delta_timescale * self.zeta * 0.5)

        return new_geometry # Corrected: Returns numpy array, not list

    def Nose_Hoover_chain_thermostat(self, geom_num_list, new_g): # fixed volume #NVT ensemble
        """
        Nose-Hoover Chain implementation.
        """
        geom_arr = np.array(geom_num_list, dtype=np.float64)
        force = -1.0 * np.array(new_g, dtype=np.float64)

        # 1. First half-step thermostat scaling
        self.momentum_list *= np.exp(-self.delta_timescale * self.zeta_chain[0] * 0.5)

        # 2. First half-step momentum update
        self.momentum_list += force * self.delta_timescale * 0.5
        
        print("NVT ensemble (Nose_Hoover chain) : Sum of momenta (absolute value):", np.sum(np.abs(self.momentum_list)))
        
        # 3. Position update
        new_geometry = self._update_position(geom_arr, self.delta_timescale)

        # 4. Thermostat Propagation
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
        
        self._propagate_nhc_zeta(self.delta_timescale, 2 * self.tot_kinetic_ene)
        
        print("zeta_list (Coefficient of friction): ", self.zeta_chain)    

        # 5. Second half-step momentum update
        self.momentum_list += force * self.delta_timescale * 0.5

        # 6. Second half-step thermostat scaling
        self.momentum_list *= np.exp(-self.delta_timescale * self.zeta_chain[0] * 0.5)
        
        return new_geometry # Corrected: Returns numpy array, not list

    def Velocity_Verlet(self, geom_num_list, new_g, prev_g, iter): # NVE ensemble 
        """
        Velocity Verlet integration.
        """
        geom_arr = np.array(geom_num_list, dtype=np.float64)
        
        force_new = -1.0 * np.array(new_g, dtype=np.float64)
        force_prev = -1.0 * np.array(prev_g, dtype=np.float64)
        
        # 1. Update Momentum
        self.momentum_list += (force_new + force_prev) * self.delta_timescale * 0.5

        # 2. Position Update
        term1 = (self.momentum_list * self.inverse_masses) * self.delta_timescale
        term2 = (force_new * self.inverse_masses) * (self.delta_timescale**2 / 2.0)
        
        new_geometry = geom_arr + term1 + term2
        
        # Stats
        self.calc_inst_temperature()
        self.add_inst_temperature_list()
 
        return new_geometry # Corrected: Returns numpy array, not list

    def generate_normal_random_variables(self, n_variables):
        """Vectorized Box-Muller transformation"""
        n_pairs = (n_variables + 1) // 2
        u1 = np.random.rand(n_pairs)
        u2 = np.random.rand(n_pairs)
        
        r = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2
        
        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)
        
        result = np.empty(n_pairs * 2)
        result[0::2] = z1
        result[1::2] = z2
        
        return result[:n_variables]

    def calc_rand_moment_based_on_boltzman_const(self, random_variables):
        """
        Scales random variables by sqrt(kB * T * m).
        """
        rand_moment = np.array(random_variables, dtype=np.float64)
        scale_factors = np.sqrt(self.Boltzmann_constant * self.temperature * self.masses)
        rand_moment *= scale_factors
        return rand_moment
    
    def init_purtubation(self, geometry, B_e, B_g):
        """Initializes momenta with random thermal noise."""
        N = len(self.element_list)
        random_vars = self.generate_normal_random_variables(N * 3).reshape(N, 3)
        v_thermal = random_vars * np.sqrt(self.Boltzmann_constant * self.temperature * self.inverse_masses)
        init_momentum = v_thermal * self.masses
        
        self.momentum_list += init_momentum
        self.init_energy = B_e
        return
    def Langevin_thermostat(self, geom_num_list, new_g):
        """
        Langevin Dynamics (BAOAB integrator)
        Reference: B. Leimkuhler and C. Matthews, J. Chem. Phys. 138, 174102 (2013).
        
        Structure:
          B: Momentum += 0.5 * dt * Force
          A: Position += 0.5 * dt * Velocity
          O: Momentum = c1 * Momentum + c2 * Noise (Ornstein-Uhlenbeck)
          A: Position += 0.5 * dt * Velocity
          B: Momentum += 0.5 * dt * Force
        """
        geom_arr = np.array(geom_num_list, dtype=np.float64)
        force = -1.0 * np.array(new_g, dtype=np.float64)
        
        
        dt = self.delta_timescale
        gamma = self.Langevin_zeta  # (1/time)
        target_temp = self.initial_temperature 

        # 
        c1 = np.exp(-gamma * dt)
        c2 = np.sqrt(1.0 - c1**2)
        
        # sigma = sqrt(m * kB * T)
        # self.masses shape: (N, 1) -> broadcasting works
        sigma = np.sqrt(self.masses * self.Boltzmann_constant * target_temp)

        # 1. B: Half-step Momentum Update (Force)
        self.momentum_list += 0.5 * dt * force

        # 2. A: Half-step Position Update
        # r(t+dt/2) = r(t) + 0.5 * dt * v(t+dt/2)
        new_geometry = self._update_position(geom_arr, 0.5 * dt)

        # 3. O: Fluctuation-Dissipation (Thermostat)
        # p' = c1 * p + c2 * sigma * noise
      
        noise = np.random.normal(0.0, 1.0, self.momentum_list.shape)
        self.momentum_list = c1 * self.momentum_list + c2 * sigma * noise

        # 4. A: Half-step Position Update
        # r(t+dt) = r(t+dt/2) + 0.5 * dt * v'
        new_geometry = self._update_position(new_geometry, 0.5 * dt)

        # 5. B: Half-step Momentum Update (Force)
        
        self.momentum_list += 0.5 * dt * force

       
        self.calc_inst_temperature()
        self.add_inst_temperature_list()

        return new_geometry