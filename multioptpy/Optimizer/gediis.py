import numpy as np
from .gdiis import GDIIS
from .ediis import EDIIS

class GEDIIS:
    """
    Combined GDIIS and EDIIS optimization method.
    
    This class implements a combined gradient- and energy-based DIIS approach
    that adaptively blends the two methods based on their performance.
    """
    
    def __init__(self):
        # Create individual optimizers
        self.gdiis = GDIIS()
        self.ediis = EDIIS()
        
        # Configuration parameters
        self.gediis_mode = 'adaptive'    # 'adaptive', 'sequential', or 'blend'
        self.gediis_ediis_weight = 0.6   # Initial EDIIS weight in blend mode
        self.gediis_auto_switch = True   # Automatically switch based on performance
        
        # Tracking variables
        self.energy_history = []
        self.grad_rms_history = []
        self.gdiis_success_count = 0
        self.ediis_success_count = 0
        self.iter = 0
        
        # For sequential mode
        self.gediis_phase = 'ediis'      # Start with EDIIS ('ediis' or 'gdiis')
        self.gediis_phase_steps = 0
        self.gediis_phase_switch = 3     # Switch phases every N steps
        return
    
    def _evaluate_performance(self):
        """
        Evaluate the relative performance of GDIIS and EDIIS
        
        Returns:
        --------
        tuple
            (ediis_weight, gdiis_weight) - Relative weights for the two methods
        """
        # With limited history, favor EDIIS early and GDIIS later
        if len(self.energy_history) < 3:
            if self.iter < 10:
                # Early iterations: favor EDIIS (better for large steps)
                return 0.7, 0.3
            else:
                # Later iterations: favor GDIIS (better for fine convergence)
                return 0.3, 0.7
        
        # Look at recent energy and gradient trends
        energy_decreasing = self.energy_history[-1] < self.energy_history[-2]
        grad_decreasing = self.grad_rms_history[-1] < self.grad_rms_history[-2]
        
        # Adjust success counters
        if energy_decreasing:
            self.ediis_success_count += 1
        else:
            self.ediis_success_count = max(0, self.ediis_success_count - 1)
            
        if grad_decreasing:
            self.gdiis_success_count += 1
        else:
            self.gdiis_success_count = max(0, self.gdiis_success_count - 1)
        
        # Calculate weights based on success counts and iteration phase
        total_success = self.ediis_success_count + self.gdiis_success_count + 1  # +1 to avoid division by zero
        ediis_raw_weight = self.ediis_success_count / total_success
        
        # Adjust based on optimization phase (EDIIS early, GDIIS late)
        phase_factor = max(0.0, min(1.0, (20 - self.iter) / 20))  # 1.0→0.0 over 20 iterations
        ediis_weight = 0.3 + ediis_raw_weight * 0.4 + phase_factor * 0.3
        
        # Ensure weights are in reasonable range
        ediis_weight = max(0.2, min(0.8, ediis_weight))
        gdiis_weight = 1.0 - ediis_weight
        
        return ediis_weight, gdiis_weight
        
    def run(self, geom_num_list, energy, B_g, pre_B_g, original_move_vector):
        """
        Run combined GEDIIS optimization
        
        Parameters:
        -----------
        geom_num_list : numpy.ndarray
            Current geometry
        energy : float
            Current energy value
        B_g : numpy.ndarray
            Current gradient
        pre_B_g : numpy.ndarray
            Previous gradient
        original_move_vector : numpy.ndarray
            Original optimization step
            
        Returns:
        --------
        numpy.ndarray
            Optimized step vector
        """
        print("GEDIIS method (combined GDIIS and EDIIS)")
        n_coords = len(geom_num_list)
        grad_rms = np.sqrt(np.mean(B_g ** 2))
        
        # Update history
        self.energy_history.append(energy)
        self.grad_rms_history.append(grad_rms)
        
        # Always run both methods to update their histories
        gdiis_step = self.gdiis.run(geom_num_list, B_g, pre_B_g, original_move_vector)
        ediis_step = self.ediis.run(geom_num_list, energy, B_g, original_move_vector)
        
        # Determine mode of operation
        if self.gediis_mode == 'sequential':
            # Sequential mode switches between methods periodically
            self.gediis_phase_steps += 1
            if self.gediis_phase_steps >= self.gediis_phase_switch:
                self.gediis_phase = 'ediis' if self.gediis_phase == 'gdiis' else 'gdiis'
                self.gediis_phase_steps = 0
                print(f"Switching to {self.gediis_phase.upper()} phase")
            
            if self.gediis_phase == 'ediis':
                print("Using EDIIS step (sequential mode)")
                step = ediis_step
            else:
                print("Using GDIIS step (sequential mode)")
                step = gdiis_step
                
        elif self.gediis_mode == 'adaptive':
            # Adaptive mode chooses the best method based on performance
            if self.gediis_auto_switch and len(self.energy_history) > 1:
                ediis_weight, gdiis_weight = self._evaluate_performance()
                
                # Blend with emphasis on better-performing method
                if ediis_weight > 0.7:
                    print(f"Using EDIIS step (adaptive mode, {ediis_weight:.2f})")
                    step = ediis_step
                elif gdiis_weight > 0.7:
                    print(f"Using GDIIS step (adaptive mode, {gdiis_weight:.2f})")
                    step = gdiis_step
                else:
                    # Blend the steps
                    step = ediis_weight * ediis_step + gdiis_weight * gdiis_step
                    print(f"Using blended step: {ediis_weight:.2f}*EDIIS + {gdiis_weight:.2f}*GDIIS")
            else:
                # Default behavior early in optimization
                if self.iter < 5:
                    print("Using EDIIS step (early iterations)")
                    step = ediis_step
                else:
                    print("Using GDIIS step (later iterations)")
                    step = gdiis_step
                
        else:  # blend mode
            # Blend mode always combines both methods with fixed weights
            ediis_weight = self.gediis_ediis_weight
            gdiis_weight = 1.0 - ediis_weight
            step = ediis_weight * ediis_step + gdiis_weight * gdiis_step
            print(f"Using blended step: {ediis_weight:.2f}*EDIIS + {gdiis_weight:.2f}*GDIIS")
        
        # Final safety checks
        step_norm = np.linalg.norm(step)
        orig_norm = np.linalg.norm(original_move_vector)
        
        # Check for problematic steps
        if np.any(np.isnan(step)) or np.any(np.isinf(step)) or step_norm < 1e-10:
            print("Warning: Invalid GEDIIS step, falling back to original step")
            step = original_move_vector
        elif step_norm > 3.0 * orig_norm and orig_norm > 1e-10:
            # Cap step size to avoid large jumps
            print(f"Warning: GEDIIS step too large ({step_norm:.4f} > {3.0 * orig_norm:.4f}), scaling down")
            scale_factor = 3.0 * orig_norm / step_norm
            step = scale_factor * step
        
        # Monitor convergence behavior
        if len(self.energy_history) > 3:
            # Check if optimization is oscillating
            oscillating = False
            energy_diffs = np.diff(self.energy_history[-4:])
            if np.all(np.abs(energy_diffs) > 0):  # All non-zero differences
                signs = np.sign(energy_diffs)
                if np.sum(np.abs(np.diff(signs))) >= 2:  # Sign changes indicate oscillation
                    oscillating = True
                    
            if oscillating:
                print("Warning: Possible oscillatory behavior detected")
                # Switch mode or adjust parameters to stabilize
                if self.gediis_mode == 'adaptive':
                    # Increase EDIIS weight for stability
                    ediis_weight = max(0.7, ediis_weight)
                    gdiis_weight = 1.0 - ediis_weight
                    # Recalculate step with higher EDIIS weight
                    step = ediis_weight * ediis_step + gdiis_weight * gdiis_step
                    print(f"Stabilizing oscillation: {ediis_weight:.2f}*EDIIS + {gdiis_weight:.2f}*GDIIS")
        
        # Update iteration counter
        self.iter += 1
        
        return step