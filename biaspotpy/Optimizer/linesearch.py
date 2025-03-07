import numpy as np


class LineSearch:
    def __init__(self, prev_move_vector, move_vector, gradient, prev_gradient, energy, prev_energy, hessian=None):
        """Initialize the LineSearch class with optimization parameters."""
        self.move_vector = move_vector
        self.prev_move_vector = prev_move_vector
        self.gradient = gradient
        self.prev_gradient = prev_gradient
        self.energy = energy
        self.prev_energy = prev_energy
        self.hessian = hessian
        
        # Line search control parameters
        self.maxstep = 0.2  # Maximum step size for line search
        self.stpmin = 1e-5  # Minimum step size threshold
        self.stpmax = 5.0  # Maximum step length
        self.damping = 1.0  # Initial step size damping
        self.alpha = 0.1    # Wolfe parameter: sufficient decrease
        self.min_step_dec = 0.5  # Minimum decrease in step size
        self.max_step_dec = 0.1  # Maximum decrease in step size
        
        # For convergence detection
        self.force_consistent = True  # Use force-based convergence test

    def linesearch(self, prev_move_vector, move_vector, gradient, prev_gradient, energy, prev_energy, hessian=None):
        """
        Args:
            prev_move_vector: Previous move vector
            move_vector: Current move vector (search direction)
            gradient: Current gradient
            prev_gradient: Previous gradient
            energy: Current energy at alpha=1.0
            prev_energy: Previous energy at alpha=0.0
            hessian: Hessian matrix (optional)
            
        Returns:
            new_move_vector: The move vector adjusted by the optimal step size
            optimal_step_flag: Boolean indicating if an optimal step was found
        """
        print("\n===== Starting Line Search =====")
        print(f"Previous energy: {prev_energy:.6f}")
        print(f"Initial energy: {energy:.6f}")
        
        # Convert inputs to proper shape
        search_direction = move_vector.reshape(-1)
        prev_gradient = prev_gradient.reshape(-1)
        gradient = gradient.reshape(-1)
        
        # Calculate step length
        steplength = np.linalg.norm(search_direction)
        if steplength > self.stpmax:
            search_direction = search_direction * self.stpmax / steplength
            steplength = self.stpmax
        
        # Calculate directional derivative
        derphi0 = np.dot(prev_gradient, search_direction)
        
        # Check if we have a descent direction
        if derphi0 >= 0:
            print("Not a descent direction. Resetting search direction.")
            # If not a descent direction, use negative gradient instead
            search_direction = -prev_gradient.copy()
            derphi0 = np.dot(prev_gradient, search_direction)
            
        
        initial_step_size = abs(self.maxstep / np.max(abs(search_direction)))
        
        # Adjust step size based on damping
        stp = initial_step_size * self.damping
        
        # Apply check based on energy change
        if energy > prev_energy:
           
            print("Energy increased. Reducing step size.")
            # Alpha is the factor in sufficient decrease condition
            # Calculate a, b, c coefficients for a quadratic approximation
            a = ((energy - prev_energy) - derphi0) / (stp * stp)
            if a > 0:
                # Quadratic has a minimum
                stp_min = -derphi0 / (2.0 * a)
                if stp_min < self.min_step_dec * stp:
                    stp = self.min_step_dec * stp
                elif stp_min > self.max_step_dec * stp:
                    stp = self.max_step_dec * stp
                else:
                    stp = stp_min
            else:
                # Quadratic doesn't have a minimum, use default decrease
                stp = self.min_step_dec * stp
                
            print(f"Adjusted step size: {stp:.6f}")
        else:
            
            print("Energy decreased. Accepting step.")
            
            # Ratio of actual to predicted decrease
            actual_reduction = prev_energy - energy
            predicted_reduction = -self.alpha * stp * derphi0
            
            if actual_reduction > 0.5 * predicted_reduction:
                # Good reduction, can try a slightly larger step next time
                self.damping = min(2.0 * self.damping, 1.0)
                print(f"Good step. Increasing damping to: {self.damping:.6f}")
            else:
                # Mediocre reduction, keep current or decrease slightly
                self.damping = max(0.5 * self.damping, 0.1)
                print(f"Mediocre step. Decreasing damping to: {self.damping:.6f}")
        
        # Create the new move vector with the adjusted step size
        new_move_vector = stp * search_direction.reshape(-1, 1)
        
        # Check for convergence based on step size
        if np.linalg.norm(new_move_vector) < self.stpmin:
            print(f"Step size {np.linalg.norm(new_move_vector):.6e} below threshold.")
            print("Line search has converged.")
            return -1*new_move_vector, True
        
        print(f"New step size: {stp:.6f}")
        print(f"New direction: {search_direction[:3]}...")
        print("===== Line Search Complete =====\n")
        
        return -1*new_move_vector, False