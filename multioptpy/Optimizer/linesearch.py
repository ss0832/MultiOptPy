import numpy as np
import scipy.linalg # Keep for potential future use, though not needed now
from scipy.optimize import minimize # Keep for potential future use
from typing import List, Tuple, Optional


class LineSearch:
    """
    Stateful line search using simple energy decrease and extrapolation.
    Accepts the last step that showed energy decrease.
    Terminates if energy increases or if gradient becomes orthogonal (cosine < threshold).
    Returns incremental steps compatible with cumulative updates.
    """
    def __init__(self,
                 cos_threshold: float = 0.05, # Threshold for |cos(g_k, p)| to terminate
                 maxstep: float = 0.2,
                 damping: float = 0.8, # Start with a smaller initial step
                 stpmin: float = 1e-8, # Minimum allowed *incremental* alpha for extrapolation
                 stpmax: float = 5.0, # Max total alpha allowed
                 max_iterations: int = 10,
                 extrapolation_factor: float = 1.5 # Factor to increase alpha
                 ):
        """Initializes the Simple Extrapolation LineSearch configuration."""
        
        # --- Configuration ---
        self.condition_type = 'simple_decrease_and_cosine' # Indicate the strategy
        self.cos_threshold = cos_threshold # Cosine orthogonality condition
        self.maxstep = maxstep
        self.damping = damping
        self.stpmin = stpmin
        self.stpmax = stpmax
        self.max_iterations = max_iterations
        self.extrapolation_factor = extrapolation_factor

        # --- Internal State ---
        self.active_search = False
        self.p = None           # Fixed search direction vector
        self.p_norm = None      # Norm of the search direction vector
        self.g0 = None          # Gradient at geom0 (stored for info)
        self.e0 = None          # Energy at geom0
        self.derphi0 = None     # Directional derivative at geom0 (stored for info)
        self.current_total_alpha = 0.0 # Total alpha from geom0 for the point *being evaluated*
        self.current_iteration = 0
        self.ITR = 0
        self.geom0 = None       # Coordinates at the start of the line search (x_k)
        self.last_returned_total_alpha = 0.0 # Total alpha corresponding to the previous *returned* step

        # --- State for this strategy ---
        self.best_valid_total_alpha = 0.0 # Best total alpha found so far (energy decreased)
        self.best_valid_phi = np.inf    # Energy corresponding to best_valid_total_alpha

        # --- Observation Lists (for debugging/record-keeping) ---
        self.observed_alphas = []
        self.observed_phis = []
        self.observed_phi_primes = []


    def _add_observation(self, total_alpha: float, phi: float, phi_prime: float) -> bool:
        """Adds observation (primarily for record keeping/debugging now)."""
        
        if any(abs(total_alpha - a_obs) < np.finfo(float).eps * 10 for a_obs in self.observed_alphas):
            return False
            
        self.observed_alphas.append(total_alpha)
        self.observed_phis.append(phi)
        self.observed_phi_primes.append(phi_prime)
        
        data = sorted(zip(self.observed_alphas, self.observed_phis, self.observed_phi_primes))
        self.observed_alphas, self.observed_phis, self.observed_phi_primes = [list(t) for t in zip(*data)]
        
        return True


    def run(self, geom_num_list, gradient, prev_gradient, energy, prev_energy, move_vector):
        """
        Performs one stateful iteration of the simple extrapolation line search.
        Returns the INCREMENTAL step vector.
        """

        if not self.active_search:
            # --- Start a New Line Search ---
            print(f"\n===== Starting Line Search (Cosine Thresh={self.cos_threshold}, damp={self.damping}) =====")
            
            self.active_search = True
            self.current_iteration = 0
            self.p = np.asarray(move_vector).reshape(-1)
            self.g0 = np.asarray(prev_gradient).reshape(-1)
            self.e0 = float(prev_energy)
            self.geom0 = np.asarray(geom_num_list).reshape(-1)
            self.derphi0 = np.dot(self.g0, self.p) 

          
            # Store the norm of the (final) search direction
            self.p_norm = np.linalg.norm(self.p)

            # --- Initialize Search State ---
            self.best_valid_total_alpha = 0.0
            self.best_valid_phi = self.e0
            self.last_returned_total_alpha = 0.0

            self.observed_alphas = []
            self.observed_phis = []
            self.observed_phi_primes = []
            self._add_observation(0.0, self.e0, self.derphi0)

            # --- Calculate Initial Step ---
            p_norm_max = np.max(np.abs(self.p)) # Use max component for scaling
            if p_norm_max < 1e-10:
                print("Warning: Move vector max component is zero.")
                self.active_search = False
                self.geom0 = None
                return np.zeros_like(self.p.reshape(-1, 1))

            scale = abs(self.maxstep / p_norm_max)
            
            next_total_alpha = min(1.0, scale) * self.damping
            
            next_total_alpha = np.clip(next_total_alpha, self.stpmin, self.stpmax) 

            print(f"Start E: {self.e0:.6f}, Deriv (g0.p): {self.derphi0:.6f}")
            print(f"Initial trial total alpha: {next_total_alpha:.6f} (using damping={self.damping})")

            self.current_total_alpha = next_total_alpha
            incremental_alpha = self.current_total_alpha - self.last_returned_total_alpha
            
            return incremental_alpha * self.p.reshape(-1, 1)

        
        # --- Continue Existing Line Search Iteration ---
        print(f"\n--- Line Search Iteration {self.current_iteration} (eval total alpha={self.current_total_alpha:.6f}) ---")
        
        self.current_iteration += 1
        current_geom = np.asarray(geom_num_list).reshape(-1)
        e_curr = float(energy) 
        g_curr = np.asarray(gradient).reshape(-1)
        
        derphi_curr = np.dot(g_curr, self.p) 

        _ = self._add_observation(self.current_total_alpha, e_curr, derphi_curr)

        print(f"E_curr: {e_curr:.6f}") 
        if self.geom0 is not None:
             norm_diff = np.linalg.norm(current_geom - self.geom0)
             print(f"   Norm diff from start: {norm_diff:.6f}")

        terminate = False
        accepted_total_alpha = 0.0

        # --- Check Conditions ---
        
        if e_curr < self.best_valid_phi:
            # --- Energy Decreased: Update best and check cosine condition ---
            print(f"Energy decreased to {e_curr:.6f}.")
            
            self.best_valid_total_alpha = self.current_total_alpha
            self.best_valid_phi = e_curr
            
            # --- NEW: Check cosine condition (g_k orthogonal to p) ---
            g_norm = np.linalg.norm(g_curr)
            denominator = g_norm * self.p_norm

            cosine_theta = 0.0
            if denominator < 1e-15:
                # Gradient norm or p norm is zero.
                # If g_norm is zero, we are at a minimum, so accept.
                cosine_theta = 0.0
                print("   Note: Gradient norm is near zero.")
            else:
                # cosine_theta = (g_k . p) / (||g_k|| * ||p||)
                cosine_theta = derphi_curr / denominator
            
            print(f"   Checking cosine(g_curr, p): |cos(theta)|={abs(cosine_theta):.6f} vs threshold={self.cos_threshold:.6f}")
            
            if abs(cosine_theta) < self.cos_threshold:
                print("   Cosine condition met (gradient orthogonal to search). Accepting this step.")
                terminate = True
                accepted_total_alpha = self.current_total_alpha
            
            else:
                # --- Cosine NOT met: Extrapolate ---
                print("   Cosine not met. Extrapolating.")
                
                next_total_alpha = self.current_total_alpha * self.extrapolation_factor
                next_total_alpha = np.clip(next_total_alpha, self.stpmin, self.stpmax)

                next_incremental_alpha = next_total_alpha - self.current_total_alpha

                if abs(next_incremental_alpha) < self.stpmin:
                    print(f"Warning: Extrapolation step too small ({next_incremental_alpha:.2e}). Accepting current best step.")
                    terminate = True
                    accepted_total_alpha = self.best_valid_total_alpha
                elif self.current_iteration >= self.max_iterations:
                     print("Warning: Max iterations reached during extrapolation. Accepting last successful step.")
                     terminate = True
                     accepted_total_alpha = self.best_valid_total_alpha

                if not terminate:
                    self.last_returned_total_alpha = self.current_total_alpha 
                    self.current_total_alpha = next_total_alpha 
                    return next_incremental_alpha * self.p.reshape(-1, 1)

        else:
            # --- Energy Increased: Terminate ---
            print(f"Energy increased or stalled (E_curr={e_curr:.6f} >= E_best={self.best_valid_phi:.6f}). Accepting previous best step.")
            terminate = True
            accepted_total_alpha = self.best_valid_total_alpha
            
        # --- Handle Termination ---
        if terminate:
            if accepted_total_alpha <= 0: 
                 print("No energy decrease found compared to start. Returning zero incremental step.")
                 incremental_alpha = 0.0 - self.current_total_alpha
                 accepted_phi = self.e0
            else:
                 print(f"Accepting total alpha: {accepted_total_alpha:.6f} with E={self.best_valid_phi:.6f}")
                 incremental_alpha = accepted_total_alpha - self.current_total_alpha
                 accepted_phi = self.best_valid_phi


            print(f"===== Line Search Complete =====")
            print(f"Final total alpha: {accepted_total_alpha:.6f}")
            print(f"Final energy: {accepted_phi:.6f} (improvement: {self.e0 - accepted_phi:.6f})")
            
            self.ITR += 1
            print(f"Total line searches completed: {self.ITR}")
            
            self.active_search = False
            self.geom0 = None
            
            return incremental_alpha * self.p.reshape(-1, 1)

            
        print("Error: Line search reached unexpected state.")
        self.active_search = False
        self.geom0 = None
        return np.zeros_like(self.p.reshape(-1, 1))