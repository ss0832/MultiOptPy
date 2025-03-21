from .linesearch import LineSearch
import numpy as np

class LBFGS:
    """Limited-memory BFGS optimizer.
    
    A limited memory version of the BFGS algorithm that approximates the inverse Hessian
    matrix using a limited amount of memory. Unlike the standard BFGS algorithm, 
    LBFGS does not explicitly store the Hessian matrix, but instead builds it implicitly
    from previous gradients and positions.
    """
    
    def __init__(self, **config):
        # Configuration parameters
        self.config = config
        
        # Initialize flags
        self.Initialization = True
        self.linesearchflag = False
        self.optimal_step_flag = False
        
        # Set default parameters
        self.DELTA = config.get("delta", 1.0)  # Step size scaling factor
        self.FC_COUNT = config.get("fc_count", -1)  # Frequency of computing full Hessian
        self.saddle_order = 0
        self.iter = 0
        self.memory = config.get("memory", 30)  # Number of previous steps to remember
        self.damping = config.get("damping", 0.75)  # Damping factor for step size
        self.alpha = config.get("alpha", 10.0)  # Initial Hessian scaling
        self.beta = config.get("beta", 0.1)  # Momentum factor
        
        # Storage for L-BFGS vectors
        self.s = []  # Position differences
        self.y = []  # Gradient differences
        self.rho = []  # 1 / (y_k^T s_k)
        
        # Initialize Hessian related variables
        self.hessian = None  # Not explicitly stored in L-BFGS
        self.bias_hessian = None  # For additional Hessian terms
        self.H0 = 1.0 / self.alpha  # Initial approximation of inverse Hessian
        
        # For line search
        self.prev_move_vector = None
        
        return
    
    def project_out_hess_tr_and_rot_for_coord(self, hessian, geometry):
        """Project out translation and rotation from Hessian.
        
        This is only used for line search in this implementation.
        """
        natoms = len(geometry)
        
        geometry -= self.calc_center(geometry)
        
        tr_x = (np.tile(np.array([1, 0, 0]), natoms)).reshape(-1, 3)
        tr_y = (np.tile(np.array([0, 1, 0]), natoms)).reshape(-1, 3)
        tr_z = (np.tile(np.array([0, 0, 1]), natoms)).reshape(-1, 3)

        rot_x = np.cross(geometry, tr_x).flatten()
        rot_y = np.cross(geometry, tr_y).flatten() 
        rot_z = np.cross(geometry, tr_z).flatten()
        tr_x = tr_x.flatten()
        tr_y = tr_y.flatten()
        tr_z = tr_z.flatten()

        TR_vectors = np.vstack([tr_x, tr_y, tr_z, rot_x, rot_y, rot_z])
        
        Q, R = np.linalg.qr(TR_vectors.T)
        keep_indices = ~np.isclose(np.diag(R), 0, atol=1e-6, rtol=0)
        TR_vectors = Q.T[keep_indices]
        n_tr = len(TR_vectors)

        P = np.identity(natoms * 3)
        for vector in TR_vectors:
            P -= np.outer(vector, vector)

        hess_proj = np.dot(np.dot(P.T, hessian), P)

        return hess_proj
    
    def calc_center(self, geometry, element_list=[]):
        """Calculate center of geometry."""
        center = np.array([0.0, 0.0, 0.0], dtype="float64")
        for i in range(len(geometry)):
            center += geometry[i] 
        center /= float(len(geometry))
        
        return center
    
    def set_hessian(self, hessian):
        """Set explicit Hessian matrix (not used in LBFGS)."""
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        """Set bias Hessian matrix."""
        self.bias_hessian = bias_hessian
        return
    
    def get_hessian(self):
        """Get Hessian matrix.
        
        Note: In LBFGS, the Hessian is not explicitly stored, 
        but this method is provided for compatibility.
        """
        return self.hessian
    
    def get_bias_hessian(self):
        """Get bias Hessian matrix."""
        return self.bias_hessian
    
    def update_vectors(self, displacement, delta_grad):
        """Update the vectors used for the L-BFGS approximation."""
        # Flatten vectors
        s = displacement.flatten()
        y = delta_grad.flatten()
        
        # Calculate rho = 1 / (y^T * s)
        dot_product = np.dot(y, s)
        if abs(dot_product) < 1e-10:
            # Avoid division by very small numbers
            rho = 1000.0
        else:
            rho = 1.0 / dot_product
        
        # Add to history
        self.s.append(s)
        self.y.append(y)
        self.rho.append(rho)
        
        # Remove oldest vectors if exceeding memory limit
        if len(self.s) > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
    
    def compute_lbfgs_direction(self, gradient):
        """Compute the search direction using the L-BFGS algorithm."""
        # Flatten gradient
        q = gradient.flatten()
        
        # Number of vectors to use
        loopmax = min(self.memory, len(self.s))
        a = np.empty((loopmax,), dtype=np.float64)
        
        # First loop: compute alpha_i = rho_i * s_i^T * q
        for i in range(loopmax - 1, -1, -1):
            a[i] = self.rho[i] * np.dot(self.s[i], q)
            q = q - a[i] * self.y[i]
        
        # Apply initial Hessian approximation: z = H_0 * q
        z = self.H0 * q
        
        # Second loop: compute search direction
        for i in range(loopmax):
            b = self.rho[i] * np.dot(self.y[i], z)
            z = z + self.s[i] * (a[i] - b)
        
        # Reshape to original gradient shape
        z = z.reshape(gradient.shape)
        
        return z
    
    def determine_step(self, dr):
        """Determine step to take according to maxstep."""
        if self.config.get("maxstep") is None:
            return dr
        
        # Get maxstep from config
        maxstep = self.config.get("maxstep")
        
        # Calculate step lengths
        dr_reshaped = dr.reshape(-1, 3) if dr.size % 3 == 0 else dr.reshape(-1, dr.size)
        steplengths = np.sqrt((dr_reshaped**2).sum(axis=1))
        longest_step = np.max(steplengths)
        
        # Scale step if necessary
        if longest_step > maxstep:
            dr = dr * (maxstep / longest_step)
        
        return dr
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        """Normal L-BFGS optimization step."""
        
        if self.linesearchflag:
            print("linesearch mode")
        else:
            print("normal mode")
        
        # First iteration - just return scaled gradient
        if self.Initialization:
            self.Initialization = False
            return self.DELTA * B_g
        
        # Calculate displacement and gradient difference
        delta_grad = (g - pre_g).reshape(len(geom_num_list), 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list), 1)
        
        # Update L-BFGS vectors
        self.update_vectors(displacement, delta_grad)
        
        # Compute L-BFGS search direction
        move_vector = self.compute_lbfgs_direction(B_g)
        
        # Scale the step
        move_vector = self.DELTA * move_vector
        
        # Apply step size constraints
        move_vector = self.determine_step(move_vector)
        
        # Handle line search if enabled
        if self.iter > 1 and self.linesearchflag:
            # For line search, we need an explicit Hessian approximation
            if self.hessian is not None:
                tmp_hess = self.project_out_hess_tr_and_rot_for_coord(
                    self.hessian, geom_num_list.reshape(-1, 3)
                )
            else:
                tmp_hess = None
                
            if self.optimal_step_flag or self.iter == 2:
                self.LS = LineSearch(
                    self.prev_move_vector, move_vector, 
                    B_g, pre_B_g, B_e, pre_B_e, tmp_hess
                )
            
            new_move_vector, self.optimal_step_flag = self.LS.linesearch(
                self.prev_move_vector, move_vector, 
                B_g, pre_B_g, B_e, pre_B_e, tmp_hess
            )
            move_vector = new_move_vector
            
            if self.optimal_step_flag or self.iter == 1:
                self.prev_move_vector = move_vector
        else:
            self.optimal_step_flag = True
        
        print("step size: ", self.DELTA, "\n")
        
        self.iter += 1
        return move_vector
    
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        """Run the optimization step."""
        method = self.config.get("method", "lbfgs").lower()
        
        
        move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        
        return move_vector