import numpy as np

class TrustRadius:    
    def __init__(self, 
                 initial_trust_radius=0.3,
                 min_trust_radius=0.01, 
                 max_trust_radius=0.5,
                 history_size=5,
                 adaptive_factor_scale=0.8,
                 energy_precision_threshold=1e-8,
                 quality='Normal'):
        """
        Initialize the composite-step trust radius method
        
        Args:
            initial_trust_radius: Initial trust radius value
            min_trust_radius: Minimum allowed trust radius
            max_trust_radius: Maximum allowed trust radius
            history_size: Number of previous steps to consider in history
            adaptive_factor_scale: Scale factor for adaptive adjustment (0-1)
            energy_precision_threshold: Threshold for energy precision
            quality: Convergence quality setting ('VeryBasic', 'Basic', 'Normal', 'Good', 'VeryGood')
        """
        # Basic parameters
        self.trust_radius = initial_trust_radius
        self.min_trust_radius = min_trust_radius
        self.max_trust_radius = max_trust_radius
        
        # Adaptive parameters
        self.history_size = history_size
        self.adaptive_factor_scale = adaptive_factor_scale
        self.energy_precision_threshold = energy_precision_threshold
        
        # History data
        self.energy_ratios = []  # History of actual/predicted energy change ratios
        self.step_sizes = []     # History of step sizes
        self.energy_changes = [] # History of energy changes
        
        # Optimization state
        self.iteration_count = 0
        self.last_energy = None
        
      
        # AMS convergence criteria based on quality setting
        self.set_convergence_quality(quality)
    
    def set_min_trust_radius(self, min_trust_radius):
        """Set minimum trust radius"""
        self.min_trust_radius = min_trust_radius
    
    def set_max_trust_radius(self, max_trust_radius):
        """Set maximum trust radius"""
        self.max_trust_radius = max_trust_radius
    
    
    
    def set_convergence_quality(self, quality):
        """
        Set convergence criteria based on quality setting
        
        Args:
            quality: Quality setting ('VeryBasic', 'Basic', 'Normal', 'Good', 'VeryGood')
        """
        # Default to Normal if invalid quality is provided
        if quality not in ['VeryBasic', 'Basic', 'Normal', 'Good', 'VeryGood']:
            quality = 'Normal'
            
        quality_map = {
            'VeryBasic': {'energy': 1e-3, 'gradients': 1e-1, 'step': 1.0, 'stress': 5e-2},
            'Basic':     {'energy': 1e-4, 'gradients': 1e-2, 'step': 0.1, 'stress': 5e-3},
            'Normal':    {'energy': 1e-5, 'gradients': 1e-3, 'step': 0.01, 'stress': 5e-4},
            'Good':      {'energy': 1e-6, 'gradients': 1e-4, 'step': 0.001, 'stress': 5e-5},
            'VeryGood':  {'energy': 1e-7, 'gradients': 1e-5, 'step': 0.0001, 'stress': 5e-6}
        }
        
        self.convergence_criteria = quality_map[quality]
        self.quality = quality
    
    def calculate_adaptive_factor(self):
        """
        Calculate adaptive adjustment factor based on optimization history
        
        Returns:
            float: Adaptive factor determining adjustment strength
        """
        # Return default value if no history is available
        if len(self.energy_ratios) == 0:
            return 2.0
        
        # Calculate based on recent prediction accuracy
        recent_ratios = self.energy_ratios[-min(self.history_size, len(self.energy_ratios)):]
        ratio_variance = np.var(recent_ratios) if len(recent_ratios) > 1 else 0.0
        
        # More cautious adjustment when prediction variance is high
        base_factor = 2.0 * np.exp(-ratio_variance)
        
        # More cautious when approaching convergence
        if self.is_approaching_convergence():
            base_factor *= self.adaptive_factor_scale
        
      
        
        return max(1.1, min(base_factor, 3.0))  # Limit to range 1.1-3.0
    
    def is_approaching_convergence(self):
        """
        Determine if optimization is approaching convergence
        
        Returns:
            bool: True if approaching convergence
        """
        if len(self.energy_changes) < 2:
            return False
        
        # Check if recent energy changes are small and decreasing
        recent_changes = np.abs(self.energy_changes[-min(3, len(self.energy_changes)):])
        return np.all(recent_changes < 0.01) and np.mean(recent_changes) < 0.005
    
    
    def update_trust_radii(self, 
                          B_e, 
                          pre_B_e, 
                          pre_B_g, 
                          pre_move_vector, 
                          model_hess, 
                          geom_num_list, 
                          trust_radii,
                          atom_types=None,
                          constraints=None):
        """
        Update trust radius using composite-step approach
        
        Args:
            B_e: Current energy
            pre_B_e: Previous energy
            pre_B_g: Previous gradient
            pre_move_vector: Previous movement vector
            model_hess: Model Hessian
            geom_num_list: Geometry number list
            trust_radii: Current trust radius
            atom_types: Optional list of atom types
            constraints: Optional constraint information
            
        Returns:
            float: Updated trust radius
        """
        if self.iteration_count == 0:
            self.iteration_count += 1
            return trust_radii
      
        
        # Calculate predicted energy change
        Ce = (np.dot(pre_B_g.reshape(1, len(geom_num_list)), 
                    pre_move_vector.reshape(len(geom_num_list), 1)) + 
              0.5 * np.dot(np.dot(pre_move_vector.reshape(1, len(geom_num_list)), 
                                 model_hess), 
                          pre_move_vector.reshape(len(geom_num_list), 1)))
        
        # Handle numerical stability
        if abs(Ce) < self.energy_precision_threshold:
            Ce += np.sign(Ce) * self.energy_precision_threshold
            if abs(Ce) < self.energy_precision_threshold:  # If sign is unknown
                Ce = self.energy_precision_threshold
        
        # Ratio of actual to predicted energy change
        r = (pre_B_e - B_e) / Ce
        
        # Update history
        self.energy_ratios.append(float(r))
        self.step_sizes.append(float(np.linalg.norm(pre_move_vector)))
        self.energy_changes.append(float(pre_B_e - B_e))
        
        # Calculate adaptive factor
        adaptive_factor = self.calculate_adaptive_factor()
        
        # Debug information
        print(f"Iteration: {self.iteration_count}")
        print(f"Energy ratio (actual/predicted): {r}")
        print(f"Adaptive factor: {adaptive_factor}")
        print(f"Current trust radius: {trust_radii}")
        
        # Trust radius adjustment logic (improved version)
        r_min = 0.25
        r_good = 0.75
        
        if r <= r_min or r >= (2.0 - r_min):
            # Decrease trust radius for poor prediction
            trust_radii /= adaptive_factor
            print("Decrease trust radius")
        elif r >= r_good and r <= (2.0 - r_good):
            # For good prediction
            if abs(np.linalg.norm(pre_move_vector) - trust_radii) < self.energy_precision_threshold:
                # Increase trust radius if previous step was limited by trust radius
                trust_radii *= adaptive_factor ** 0.5
                print("Increase trust radius")
            else:
                # Otherwise maintain trust radius
                print("Keep trust radius (good prediction)")
        else:
            # Maintain trust radius for moderate prediction
            print("Keep trust radius (moderate prediction)")
        
        self.iteration_count += 1
                
        # Clip final trust radius between min and max values
        return np.clip(trust_radii, self.min_trust_radius, self.max_trust_radius)
    
