import numpy as np


class LineSearch:
    def __init__(self, prev_move_vector, move_vector, gradient, prev_gradient, energy, prev_energy,  hessian=None):
        
        self.move_vector = move_vector
        self.prev_move_vector = prev_move_vector
        self.gradient = gradient
        self.prev_gradient = prev_gradient
        self.energy = energy
        self.prev_energy = prev_energy
        self.hessian = hessian
        self.convergence_criterion = 0.2
        self.order = 0.5
        

    def linesearch(self):
        self.prev_gradient = self.prev_gradient.reshape(len(self.prev_gradient)*3, 1)
        self.gradient = self.prev_gradient.reshape(len(self.gradient)*3, 1)
        self.prev_move_vector = self.prev_move_vector.reshape(len(self.prev_move_vector)*3, 1)
        
        #self.gradient = self.gradient/np.linalg.norm(self.gradient)
        #self.prev_move_vector = self.prev_move_vector/np.linalg.norm(self.prev_move_vector)
        
        cos = np.sum(self.gradient*self.prev_move_vector)/(np.linalg.norm(self.gradient)*np.linalg.norm(self.prev_move_vector)+1e-8)
        print("orthogonality", cos)
        if abs(cos) < self.convergence_criterion:
            new_move_vector = self.move_vector
            print("optimal step is found.")
            optimal_step_flag = True
        else:
            if self.prev_energy > self.energy:
                new_move_vector = abs(cos) ** self.order * self.prev_move_vector# / np.linalg.norm(self.prev_move_vector)
            
            else:
                new_move_vector = -1 * abs(cos) ** self.order * self.prev_move_vector# / np.linalg.norm(self.prev_move_vector)
            
            print("linesearching...")
            optimal_step_flag = False
            
        return new_move_vector, optimal_step_flag