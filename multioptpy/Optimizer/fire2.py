import numpy as np
import copy


class FIRE2:
    """Fast Inertial Relaxation Engine 2.0 (FIRE2.0) Optimization Algorithm.
    
    Implementation of the FIRE2.0 algorithm as described in:
    J. Guénolé, W.G. Nöhring, A. Vaid, F. Houllé, Z. Xie, A. Prakash, E. Bitzek,
    Assessment and optimization of the fast inertial relaxation engine (fire)
    for energy minimization in atomistic simulations and its implementation in lammps,
    Comput. Mater. Sci. 175 (2020) 109584.
    https://doi.org/10.1016/j.commatsci.2020.109584.
    
    This implementation uses the Euler semi-implicit integrator.
    """
    
    def __init__(self, **config):
        """Initialize FIRE2.0 optimizer with configuration parameters.
        
        Parameters
        ----------
        N_min : int, optional
            Minimum steps before time step increase, default 5
        f_inc : float, optional
            Factor to increase time step, default 1.10
        f_alpha : float, optional
            Factor to decrease alpha, default 0.99
        f_dec : float, optional
            Factor to decrease time step, default 0.50
        dt_max : float, optional
            Maximum time step, default 1.0
        dt_min : float, optional
            Minimum time step, default 0.01
        alpha_start : float, optional
            Initial mixing parameter, default 0.25
        maxstep : float, optional
            Maximum step size per coordinate, default 0.2
        halfstepback : bool, optional
            Whether to perform half step back when power becomes negative, default True
        display_flag : bool, optional
            Print optimization information, default True
        max_vdotf_negatif : int, optional
            Maximum number of consecutive negative power steps before stopping, default 0 (disabled)
        """
        # Default parameters (based on the FIRE2.0 paper)
        self.N_min = 5
        self.f_inc = 1.10
        self.f_alpha = 0.99
        self.f_dec = 0.50
        self.dt_max = 1.0
        self.dt_min = 0.01
        self.alpha_start = 0.25
        self.maxstep = 0.2
        self.halfstepback = True
        self.display_flag = True
        self.max_vdotf_negatif = 0  # Default: disabled
        
        # Override defaults with user config
        for key, value in config.items():
            setattr(self, key, value)
        
        # Internal state variables
        self.iteration = 1
        self.Initialization = True
        self.hessian = None
        self.bias_hessian = None
        self.vdotf_negatif = 0
        self.last_positions = None
        self.last_energy = None
        
    def run(self, geom_num_list, B_g, pre_B_g=[], pre_geom=[], B_e=0.0, pre_B_e=0.0, 
            pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[]):
        """Run one step of the FIRE2.0 optimization algorithm.
        
        Parameters
        ----------
        geom_num_list : numpy.ndarray
            Current geometry/coordinates
        B_g : numpy.ndarray
            Current gradient/forces
        pre_B_g : numpy.ndarray, optional
            Previous gradient
        pre_geom : numpy.ndarray, optional
            Previous geometry
        B_e : float, optional
            Current energy
        pre_B_e : float, optional
            Previous energy
        pre_move_vector : numpy.ndarray, optional
            Previous move vector
        initial_geom_num_list : numpy.ndarray, optional
            Initial geometry
        g : numpy.ndarray, optional
            Alternative gradient representation
        pre_g : numpy.ndarray, optional
            Alternative previous gradient representation
            
        Returns
        -------
        numpy.ndarray
            Move vector to update positions
        """
        # Initialize on first call
        if self.Initialization:
            self.dt = 0.1
            self.alpha = self.alpha_start
            self.Nsteps = 0
            self.velocity = np.zeros_like(geom_num_list)
            self.last_positions = geom_num_list.copy()
            self.last_energy = B_e
            self.Initialization = False
        
        # Save the current positions and energy for potential rollback
        current_positions = geom_num_list.copy()
        current_energy = B_e
        
        # Calculate power (P = F·V)
        power = np.dot(self.velocity.flatten(), B_g.flatten())
        
        if power > 0.0:
            # Moving downhill
            # Increment Nsteps and check if we should increase dt and decrease alpha
            self.Nsteps += 1
            if self.Nsteps > self.N_min:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.alpha *= self.f_alpha
        else:
            # Moving uphill or starting
            # Reset Nsteps, decrease dt, and reset alpha
            self.Nsteps = 0
            self.dt = max(self.dt * self.f_dec, self.dt_min)
            self.alpha = self.alpha_start
            
            # Count consecutive negative power steps
            self.vdotf_negatif += 1
            if self.max_vdotf_negatif > 0 and self.vdotf_negatif > self.max_vdotf_negatif:
                if self.display_flag:
                    print("Maximum number of consecutive negative power steps reached")
                return np.zeros_like(geom_num_list)
            
            # Apply half step back correction if enabled
            if self.halfstepback and power < 0.0:
                # Back up half a step
                half_step_correction = -0.5 * self.dt * self.velocity
                geom_num_list += half_step_correction
                
                if self.display_flag:
                    print("Applying half step back correction")
            
            # Reset velocity
            self.velocity = np.zeros_like(geom_num_list)
        
        # Update velocity using Euler semi-implicit integration
        # First update velocity with current forces (v += dt * F)
        self.velocity += self.dt * B_g
        
        # Only apply FIRE mixing if power was positive
        if power > 0.0:
            # Calculate the velocity magnitude and force magnitude
            v_norm = np.linalg.norm(self.velocity)
            f_norm = np.linalg.norm(B_g)
            
            if v_norm > 1e-10 and f_norm > 1e-10:
                # Apply FIRE mixing: v = (1-α)v + α|v|F̂
                # Where F̂ is the normalized force vector
                scaled_velocity = (1.0 - self.alpha) * self.velocity
                scaled_force = self.alpha * (v_norm / f_norm) * B_g
                self.velocity = scaled_velocity + scaled_force
        
        # Calculate the move vector (dr = dt * v)
        move_vector = self.dt * self.velocity
        
        # Check if the maximum step size is exceeded
        move_norm = np.linalg.norm(move_vector)
        if move_norm > self.maxstep:
            move_vector *= (self.maxstep / move_norm)
        
        if self.display_flag:
            print("FIRE2.0")
            print(f"Iteration: {self.iteration}")
            print(f"dt: {self.dt:.6f}, alpha: {self.alpha:.6f}, Nsteps: {self.Nsteps}")
            print(f"Power (v·F): {power:.6e}")
            print(f"Max velocity: {np.max(np.abs(self.velocity)):.6e}")
            print(f"Max force: {np.max(np.abs(B_g)):.6e}")
            print(f"Max step: {np.max(np.abs(move_vector)):.6e}")
        
        # Update state for next iteration
        self.last_positions = current_positions
        self.last_energy = current_energy
        self.iteration += 1
        
        # Reset negative power counter if power is positive
        if power > 0.0:
            self.vdotf_negatif = 0
            
        return move_vector
    
    def set_hessian(self, hessian):
        """Set Hessian matrix for the optimizer.
        
        Parameters
        ----------
        hessian : numpy.ndarray
            Hessian matrix
        """
        self.hessian = hessian
        
    def set_bias_hessian(self, bias_hessian):
        """Set bias Hessian matrix for the optimizer.
        
        Parameters
        ----------
        bias_hessian : numpy.ndarray
            Bias Hessian matrix
        """
        self.bias_hessian = bias_hessian
    
    def get_hessian(self):
        """Get Hessian matrix.
        
        Returns
        -------
        numpy.ndarray
            Hessian matrix
        """
        return self.hessian
    
    def get_bias_hessian(self):
        """Get bias Hessian matrix.
        
        Returns
        -------
        numpy.ndarray
            Bias Hessian matrix
        """
        return self.bias_hessian
    
    def reset(self):
        """Reset the optimizer to initial state."""
        self.dt = 0.1
        self.alpha = self.alpha_start
        self.Nsteps = 0
        self.velocity = None
        self.Initialization = True
        self.iteration = 1
        self.vdotf_negatif = 0
        self.last_positions = None
        self.last_energy = None