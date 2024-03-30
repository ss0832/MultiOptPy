import numpy as np
import copy

from parameter import UnitValueLib
from calc_tools import Calculationtools

"""
RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
 Theor chim Acta (1992) 82: 189-205
FSB, Bofill
 J. Chem. Phys. 1999, 111, 10806
MSP
 Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
"""


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

class Adabelief:
    def __init__(self, **config):
        #AdaBelief
        #ref. arXiv:2010.07468v5
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.99
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("AdaBelief")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i]-new_adam_m[i])**2)
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+self.Epsilon))
        
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class Adaderivative:
    def __init__(self, **config):
        #Engineering Applications of Artificial Intelligence 2023, 119, 105755. https://doi.org/10.1016/j.engappai.2022.105755
        self.adam_count = 1
        self.DELTA = 0.001
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("Adaderivative")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy((self.beta_m * adam_m[i] + (1.0-self.beta_m) * (B_g[i]) ))
            new_adam_v[i] = copy.copy((self.beta_v * adam_v[i] + (1.0-self.beta_v) * (B_g[i] - pre_B_g[i]) ** 2))
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+self.Epsilon))
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1

        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
         
class SAdam:
    def __init__(self, **config):
        #arXiv:1908.00700v2
        self.adam_count = 1
        self.DELTA = 1.0
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("SAdam")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
    
        move_vector = []
        
        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/(np.log(1.0 + np.exp(np.sqrt(new_adam_v[i])))+self.Epsilon))
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1

        return move_vector#Bohr.
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class SAMSGrad:
    def __init__(self, **config):
        #arXiv:1908.00700v2
        self.adam_count = 1
        self.DELTA = 1.0
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("SAMSGrad")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
            for j in range(len(new_adam_v[i])):
                
                new_adam_v[i][j] = max(new_adam_v[i][j], adam_v[i][j])
                    
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*new_adam_m[i]/(np.log(1.0 + np.exp(np.sqrt(new_adam_v[i])))+self.Epsilon))
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class QHAdam:
    def __init__(self, **config):
        #arXiv:1810.06801v4    
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.nu_m = 1.0
        self.nu_v = 1.0
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("QHAdam")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
    
                    
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*((1.0 - self.nu_m) * B_g[i] + new_adam_m[i] * self.nu_m)/np.sqrt(((1.0 - self.nu_v) * B_g[i] ** 2 + new_adam_v[i] * self.nu_v)+self.Epsilon))
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return 

class YOGI:
    def __init__(self, **config):
        #https://doi.org/10.1145/3441501.3441507
        self.adam_count = 1
        self.DELTA = 0.05
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("YOGI")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(adam_v[i] - (1.0 - self.beta_v) * np.sign(adam_v[i] - B_g[i] ** 2) * (B_g[i])**2)
    
                    
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append(self.DELTA*(new_adam_m[i])/np.sqrt((new_adam_v[i])+self.Epsilon))
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return

class AdaMax:
    def __init__(self, **config):
        #arXiv:1412.6980v9
        #not worked well
        self.adam_count = 1
        self.DELTA = 0.005
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.adamax_u = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("AdaMax")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
        new_adamax_u = max(self.beta_v*self.adamax_u, np.linalg.norm(B_g))
            
        move_vector = []

        for i in range(len(geom_num_list)):
            move_vector.append((self.DELTA / (self.beta_m ** adam_count + self.Epsilon)) * (adam_m[i] / (new_adamax_u + self.Epsilon)))
            
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return

class NAdam:
    def __init__(self, **config):
        #https://cs229.stanford.edu/proj2015/054_report.pdf
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.mu = 0.975
        self.nu = 0.999
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("NAdam")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = []
        new_adam_v_hat = []
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.mu*adam_m[i] + (1.0 - self.mu)*(B_g[i]))
            new_adam_v[i] = copy.copy((self.nu*adam_v[i]) + (1.0 - self.nu)*(B_g[i]) ** 2)
            new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64") * ( self.mu / (1.0 - self.mu ** adam_count)) + np.array(B_g[i], dtype="float64") * ((1.0 - self.mu)/(1.0 - self.mu ** adam_count)))        
            new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64") * (self.nu / (1.0 - self.nu ** adam_count)))
        
        move_vector = []
        for i in range(len(geom_num_list)):
            move_vector.append( (self.DELTA*new_adam_m_hat[i]) / (np.sqrt(new_adam_v_hat[i] + self.Epsilon)))
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class FIRE:
    def __init__(self, **config):#MD-like optimization method. 
        #FIRE
        #Physical Review Letters, Vol. 97, 170201 (2006)
        self.adam_count = 1
        self.N_acc = 5
        self.f_inc = 1.10
        self.f_acc = 0.99
        self.f_dec = 0.50
        self.dt_max = 0.8
        self.alpha_start = 0.1
        

        self.config = config
        self.Initialization = True
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("FIRE")
        adam_count = self.adam_count

        if self.Initialization:
            self.dt = 0.1
            self.alpha = self.alpha_start
            self.n_reset = 0
            self.pre_velocity = geom_num_list * 0.0
            self.Initialization = False
            
        
        
        velocity = (1.0 - self.alpha) * self.pre_velocity + self.alpha * (np.linalg.norm(self.pre_velocity, ord=2)/np.linalg.norm(B_g, ord=2)) * B_g
        
        if adam_count > 0 and np.dot(self.pre_velocity.reshape(1, len(geom_num_list)*3), B_g.reshape(len(geom_num_list)*3, 1)) > 0:
            if self.n_reset > self.N_acc:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.alpha = self.alpha * self.f_acc
            self.n_reset += 1
        else:
            velocity *= 0.0
            self.alpha = self.alpha_start
            self.dt *= self.f_dec
            self.n_reset = 0
        
        velocity += self.dt*B_g
        
        move_vector = copy.copy(self.dt * velocity)
        
        print("dt, alpha, n_reset :", self.dt, self.alpha, self.n_reset)
        
        self.pre_velocity = velocity
        adam_count += 1
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class RADAM:
    def __init__(self, **config):
         #arXiv:1908.03265v4
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("RADAM")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
        rho_inf = 2.0 / (1.0- self.beta_v) - 1.0 
         
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        
        new_adam_m_hat = []
        new_adam_v_hat = []
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy((self.beta_v*adam_v[i]) + (1.0-self.beta_v)*(B_g[i] - new_adam_m[i])**2) + self.Epsilon
            new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64")/(1.0-self.beta_m**adam_count))        
            new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64")/(1.0-self.beta_v**adam_count))
        rho = rho_inf - (2.0*adam_count*self.beta_v**adam_count)/(1.0 -self.beta_v**adam_count)
                    
        move_vector = []
        if rho > 4.0:
            l_alpha = []
            for j in range(len(new_adam_v)):
                l_alpha.append(np.sqrt((abs(1.0 - self.beta_v**adam_count))/new_adam_v[j]))
            l_alpha = np.array(l_alpha, dtype="float64")
            r = np.sqrt(((rho-4.0)*(rho-2.0)*rho_inf)/((rho_inf-4.0)*(rho_inf-2.0)*rho))
            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*r*new_adam_m_hat[i]*l_alpha[i])
        else:
            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i])
        
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class AdaDiff:
    def __init__(self, **config):
        #AdaDiff
        #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
        self.adam_count = 1
        self.DELTA = 0.06
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("AdaDiff")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.Initialization = False
         
        adam_count = self.adam_count
        adam_m = self.adam_m
        adam_v = self.adam_v
        new_adam_m = adam_m*0.0
        new_adam_v = adam_v*0.0
        new_adam_m_hat = adam_m*0.0
        new_adam_v_hat = adam_v*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2 + (1.0-self.beta_v) * (B_g[i] - pre_B_g[i]) ** 2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + self.Epsilon)/(1 - self.beta_v**adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+self.Epsilon))
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return     
        
class Adamod:
    def __init__(self, **config):
        #AdaDiff
        #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
        self.adam_count = 1
        self.DELTA = 10.0
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.beta_s = 0.999
        self.Epsilon = 1e-8
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("Adamod")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.adam_r = geom_num_list * 0.0
            self.adam_s = geom_num_list * 0.0
            self.Initialization = False
         
        adam_count = self.adam_count

        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        new_adam_r = self.adam_r*0.0
        new_adam_s = self.adam_s*0.0
        new_adam_m_hat = self.adam_m*0.0
        new_adam_v_hat = self.adam_v*0.0
        new_adam_r_hat = self.adam_r*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
            
            
            
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + self.Epsilon)/(1 - self.beta_v**adam_count))
            new_adam_s[i] =  self.DELTA / (np.sqrt(self.beta_v) + self.Epsilon)
            new_adam_r[i] = self.beta_s * self.adam_r[i] + (1 - self.beta_s) * new_adam_s[i]
        
        for i in range(len(geom_num_list)):
            for j in range(len(new_adam_r_hat[i])):
                new_adam_r_hat[i][j] = min(new_adam_r[i][j], new_adam_s[i][j])
        
        move_vector = []

        for i in range(len(geom_num_list)):
                move_vector.append(new_adam_r_hat[i]*new_adam_m_hat[i])
        
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_r = new_adam_r
        self.adam_s = new_adam_s
        self.adam_count += 1
        
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class EVE:
    def __init__(self, **config):
        #EVE
        #ref.arXiv:1611.01505v3
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.beta_d = 0.999
        self.DELTA = 0.06
        self.c = 10
        self.eve_d_tilde = 1.0
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
        
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("EVE")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            
            self.Initialization = False
         
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        
        new_adam_m_hat = self.adam_m*0.0
        new_adam_v_hat = self.adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**self.adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i])/(1 - self.beta_v**self.adam_count))
            
        if self.adam_count > 1:
            eve_d = abs(B_e - pre_B_e)/ min(B_e, pre_B_e)
            eve_d_hat = np.clip(eve_d, 1/self.c , self.c)
            self.eve_d_tilde = self.beta_d*self.eve_d_tilde + (1.0 - self.beta_d)*eve_d_hat
            
        else:
            pass
        
        for i in range(len(geom_num_list)):
            move_vector.append((self.DELTA/self.eve_d_tilde)*new_adam_m_hat[i]/(np.sqrt(new_adam_v_hat[i])+self.Epsilon))
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v

        self.adam_count += 1
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class AdamW:
    def __init__(self, **config):
        #AdamW
        #arXiv:2302.06675v4
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.06
        self.AdamW_lambda = 0.001
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("AdamW")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            
            self.Initialization = False
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        new_adam_m_hat = self.adam_m*0.0
        new_adam_v_hat = self.adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**self.adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + self.Epsilon)/(1 - self.beta_v**self.adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+self.Epsilon) + self.AdamW_lambda * geom_num_list[i])
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1

        return move_vector#Bohr
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class Adam:
    def __init__(self, **config):
        #Adam
        #arXiv:1412.6980
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.06
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
        
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("Adam")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            
            self.Initialization = False
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        
        new_adam_m_hat = self.adam_m*0.0
        new_adam_v_hat = self.adam_v*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**self.adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + self.Epsilon)/(1 - self.beta_v**self.adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+self.Epsilon))
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class third_order_momentum_Adam:#This is toy class. Thus, it is useless.
    def __init__(self, **config):
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.beta_s = 0.9999999999
        self.DELTA = 0.06
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("third_order_momentum_Adam")
        #self.Opt_params.eve_d_tilde is 3rd-order momentum
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.adam_s = geom_num_list * 0.0
            self.Initialization = False
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        new_adam_s = self.adam_s*0.0
        new_adam_m_hat = self.adam_m*0.0
        new_adam_v_hat = self.adam_v*0.0
        new_adam_s_hat = self.adam_s*0.0
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i])**2)
            new_adam_s[i] = copy.copy(self.beta_s*self.adam_s[i] + (1.0-self.beta_s)*(B_g[i])**3)
    
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - self.beta_m**self.adam_count))
            new_adam_v_hat[i] = copy.copy((new_adam_v[i] + self.Epsilon)/(1 - self.beta_v**self.adam_count))
            new_adam_s_hat[i] = copy.copy((new_adam_s[i] + self.Epsilon)/(1 - self.beta_s**self.adam_count))
        
        
        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA * new_adam_m_hat[i] / np.abs(np.sqrt(new_adam_v_hat[i]+self.Epsilon) - ((new_adam_m_hat[i] * (new_adam_s_hat[i]) ** (1 / 3)) / (2.0 * np.sqrt(new_adam_v_hat[i]+self.Epsilon)) )) )
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_s = new_adam_s
        self.adam_count += 1
        return move_vector#Bohr
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return            

class Adafactor:
    def __init__(self, **config):
        #Adafactor
        #arXiv:1804.04235v1
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.06
        self.Epsilon_1 = 1e-08
        self.Epsilon_2 = self.DELTA
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("Adafactor")
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            self.adam_u = geom_num_list * 0.0
            
            self.Initialization = False

        beta = 1 - self.adam_count ** (-0.8)
        rho = min(0.01, 1/np.sqrt(self.adam_count))
        alpha = max(np.sqrt(np.square(geom_num_list).mean()),  self.Epsilon_2) * rho
        new_adam_m = self.adam_m
        new_adam_v = self.adam_v*0.0
        new_adam_u = self.adam_u*0.0
        new_adam_u_hat = self.adam_u*0.0
        for i in range(len(geom_num_list)):
            new_adam_v[i] = copy.copy(beta*self.adam_v[i] + (1.0-beta)*((B_g[i])**2 + np.array([1,1,1]) * self.Epsilon_1))
            new_adam_u[i] = copy.copy(B_g[i]/np.sqrt(new_adam_v[i]))
            
                    
        move_vector = []
        for i in range(len(geom_num_list)):
            new_adam_u_hat[i] = copy.copy(new_adam_u[i] / max(1, np.sqrt(np.square(new_adam_u).mean())))
        
        
        for i in range(len(geom_num_list)):
            move_vector.append(alpha*new_adam_u_hat[i])
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_u = new_adam_u
        self.adam_count += 1
        
        return move_vector
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class Prodigy:    
    def __init__(self, **config):
        #Prodigy
        #arXiv:2306.06101v1
        self.adam_count = 1
        self.d = 0.1 
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.06
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("Prodigy")
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0        
            self.adam_s = geom_num_list * 0.0
            self.adam_r = 0.0
            new_d = self.d
            self.initial_geom_num_list = geom_num_list
            self.Initialization = False
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        new_adam_s = self.adam_s*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]*self.d))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(B_g[i]*self.d)**2)
            new_adam_s[i] = np.sqrt(self.beta_v)*self.adam_s[i] + (1.0 - np.sqrt(self.beta_v))*self.DELTA*B_g[i]*self.d**2  
        
        new_adam_r = np.sqrt(self.beta_v)*self.adam_r + (1.0 - np.sqrt(self.beta_v))*(np.dot(B_g.reshape(1,len(B_g)*3), (self.initial_geom_num_list - geom_num_list).reshape(len(B_g)*3,1)))*self.DELTA*self.d**2
        
        new_d = float(max((new_adam_r / np.linalg.norm(new_adam_s ,ord=1)), self.d))
        move_vector = []

        for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_d*new_adam_m[i]/(np.sqrt(new_adam_v[i])+self.Epsilon*self.d))
        
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_r = new_adam_r
        self.d = new_d
        self.adam_count += 1
        return move_vector
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class AdaBound:
    def __init__(self, **config):
        #AdaBound
        #arXiv:1902.09843v1
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.05
        self.Epsilon = 1e-08
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        print("AdaBound")
        
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = np.zeros((len(geom_num_list), 3, 3))
            
            self.Initialization = False
        
        move_vector = []
 
            
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0
        V = self.adam_m*0.0
        Eta = self.adam_m*0.0
        Eta_hat = self.adam_m*0.0
        
        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(self.beta_m*self.adam_m[i] + (1.0-self.beta_m)*(B_g[i]))
            new_adam_v[i] = copy.copy(self.beta_v*self.adam_v[i] + (1.0-self.beta_v)*(np.dot(np.array([B_g[i]]).T, np.array([B_g[i]]))))
            V[i] = copy.copy(np.diag(new_adam_v[i]))
            
            Eta_hat[i] = copy.copy(np.clip(self.DELTA/(np.sqrt(V[i]) + self.Epsilon), 0.1 - (0.1/((1.0 - self.beta_v) ** (self.adam_count + 1) + self.Epsilon)) ,0.1 + 0.1/((1.0 - self.beta_v) ** self.adam_count + self.Epsilon) ))
            Eta[i] = copy.copy(Eta_hat[i]/np.sqrt(self.adam_count))
                
        for i in range(len(geom_num_list)):
            move_vector.append(Eta[i] * new_adam_m[i])
            
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        return move_vector    
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return

class Adadelta:
    def __init__(self, **config):
        #Adadelta
        #arXiv:1212.5701v1
        self.adam_count = 1
        self.beta_m = 0.9
        self.beta_v = 0.999
        self.DELTA = 0.03
        self.Epsilon = 1e-06
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0
        self.RMS_FORCE_THRESHOLD = 1e+10
        self.Initialization = True
        self.config = config
        return
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):#delta is not required. This method tends to converge local minima. This class doesnt work well.
        print("Adadelta")
        if self.Initialization:
            self.adam_m = geom_num_list * 0.0
            self.adam_v = geom_num_list * 0.0
            
            self.Initialization = False
        rho = 0.9
        new_adam_m = self.adam_m*0.0
        new_adam_v = self.adam_v*0.0

        for i in range(len(geom_num_list)):
            new_adam_m[i] = copy.copy(rho * self.adam_m[i] + (1.0 - rho)*(B_g[i]) ** 2)
        move_vector = []
        
        for i in range(len(geom_num_list)):
            if self.adam_count > 1:
                move_vector.append(B_g[i] * (np.sqrt(np.square(self.adam_v).mean()) + self.Epsilon)/(np.sqrt(np.square(new_adam_m).mean()) + self.Epsilon))
            else:
                move_vector.append(B_g[i])
        if abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) > self.RMS_FORCE_THRESHOLD:
            move_vector = B_g

        for i in range(len(geom_num_list)):
            new_adam_v[i] = copy.copy(rho * self.adam_v[i] + (1.0 - rho) * (move_vector[i]) ** 2)
                
        self.adam_m = new_adam_m
        self.adam_v = new_adam_v
        self.adam_count += 1
        return move_vector
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
class Perturbation:
    def __init__(self, **config):
        self.config = config
        self.DELTA = 0.06
        self.Boltzmann_constant = 3.16681*10**(-6) # hartree/K
        self.damping_coefficient = 10.0
        self.temperature = self.config["temperature"]
        return
    def boltzmann_dist_perturb(self, move_vector):#This function is just for fun. Thus, it is no scientific basis.

        temperature = self.temperature
        perturbation = self.DELTA * np.sqrt(2.0 * self.damping_coefficient * self.Boltzmann_constant * temperature) * np.random.normal(loc=0.0, scale=1.0, size=3*len(move_vector)).reshape(len(move_vector), 3)

        return perturbation


class ConjgateGradient:
    def __init__(self, **config):
        #required variables in config(dict): method 
        self.config = config
        self.epsilon = 1e-8
        self.DELTA = 1.0
        self.Initialization = True
        return
    
    def calc_alpha(self):
        alpha = np.dot(self.gradient.reshape(1, len(self.geom_num_list)*3), (self.d_vector).reshape(len(self.geom_num_list)*3, 1)) / (np.dot(self.d_vector.reshape(1, len(self.geom_num_list)*3), self.d_vector.reshape(len(self.geom_num_list)*3, 1)) + self.epsilon)
        return alpha
    
    def calc_beta(self):#based on polak-ribiere
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)*3), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list)*3, 1)) / (np.dot(self.prev_gradient.reshape(1, len(self.geom_num_list)*3), self.prev_gradient.reshape(len(self.geom_num_list)*3, 1)) ** 2 + self.epsilon)#based on polak-ribiere
        return beta
    
    def calc_beta_PR(self):#polak-ribiere
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)*3), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list)*3, 1)) / (np.dot(self.prev_gradient.reshape(1, len(self.geom_num_list)*3), self.prev_gradient.reshape(len(self.geom_num_list)*3, 1)) + self.epsilon) #polak-ribiere
        return beta
    
    def calc_beta_FR(self):#fletcher-reeeves
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)*3), self.gradient.reshape(len(self.geom_num_list)*3, 1)) / (np.dot(self.prev_gradient.reshape(1, len(self.geom_num_list)*3), self.prev_gradient.reshape(len(self.geom_num_list)*3, 1)) + self.epsilon)
        return beta
    
    def calc_beta_HS(self):#Hestenes-stiefel
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)*3), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list)*3, 1)) / (np.dot(self.d_vector.reshape(1, len(self.geom_num_list)*3), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list)*3, 1)) + self.epsilon)
        return beta
    
    def calc_beta_DY(self):#Dai-Yuan
        beta = np.dot(self.gradient.reshape(1, len(self.geom_num_list)*3), self.gradient.reshape(len(self.geom_num_list)*3, 1)) / (np.dot(self.d_vector.reshape(1, len(self.geom_num_list)*3), (self.gradient - self.prev_gradient).reshape(len(self.geom_num_list)*3, 1)) + self.epsilon)
        return beta
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        #cg method
        self.geom_num_list = np.array(geom_num_list)
        self.gradient = np.array(B_g)
        self.prev_gradient = np.array(pre_B_g)
        
        if self.Initialization:
            self.d_vector = geom_num_list * 0.0
            self.Initialization = False
            return -1*self.DELTA*B_g
        
        alpha = self.calc_alpha()
        
        move_vector = self.DELTA * alpha * self.d_vector
        if self.config["method"] == "CG_PR":
            beta = self.calc_beta_PR()
        elif self.config["method"] == "CG_FR":
            beta = self.calc_beta_FR()
        elif self.config["method"] == "CG_HS":
            beta = self.calc_beta_HS()
        elif self.config["method"] == "CG_DY":
            beta = self.calc_beta_DY()
        else: 
            beta = self.calc_beta()
        
        self.d_vector = copy.copy(-1 * B_g + abs(beta) * self.d_vector)
        return move_vector#Bohr.
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
  

class ModelHessianUpdate:
    def __init__(self):
        self.Initialization = True
        return
    def BFGS_hessian_update(self, hess, displacement, delta_grad):
        
        A = delta_grad - np.dot(hess, displacement)

        delta_hess = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        return delta_hess
    
    def FSB_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
        delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        return delta_hess

    def Bofill_hessian_update(self, hess, displacement, delta_grad):
        #J. Chem. Phys. 1999, 111, 10806
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
        delta_hess_PSB = -1 * np.dot(A.T, displacement) * np.dot(displacement, displacement.T) / np.dot(displacement.T, displacement) ** 2 - (np.dot(A, displacement.T) + np.dot(displacement, A.T)) / np.dot(displacement.T, displacement)
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = Bofill_const*delta_hess_SR1 + (1 - Bofill_const)*delta_hess_PSB
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        return delta_hess  
    
    def MSP_hessian_update(self, hess, displacement, delta_grad):
        #Journal of Molecular Structure: THEOCHEM 2002, 591 (1–3), 35–57.
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_MS = np.dot(A, A.T) / np.dot(A.T, displacement) #SR1
        delta_hess_P = -1 * np.dot(A.T, displacement) * np.dot(displacement, displacement.T) / np.dot(displacement.T, displacement) ** 2 - (np.dot(A, displacement.T) + np.dot(displacement, A.T)) / np.dot(displacement.T, displacement) #PSB
        A_norm = np.linalg.norm(A) + 1e-8
        displacement_norm = np.linalg.norm(displacement) + 1e-8
        
        phi = np.sin(np.arccos(np.dot(displacement.T, A) / (A_norm * displacement_norm))) ** 2
        delta_hess = phi*delta_hess_P + (1 - phi)*delta_hess_MS
        #delta_hess = Calculationtools().project_out_hess_tr_and_rot(delta_hess, self.element_list, self.geom_num_list)
        
        return delta_hess

class RationalFunctionOptimization:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        self.Initialization = True
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = 0 #
        self.iter = 0 #
        self.beta = 0.5
        return
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def hessian_update(self, displacement, delta_grad):
        if "MSP" in self.config["method"]:
            print("RFO_MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(self.hessian, displacement, delta_grad)
        elif "BFGS" in self.config["method"]:
            print("RFO_BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(self.hessian, displacement, delta_grad)
        elif "FSB" in self.config["method"]:
            print("RFO_FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "Bofill" in self.config["method"]:
            print("RFO_Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(self.hessian, displacement, delta_grad)
        else:
            raise "method error"
        return delta_hess
    
    
    def normal_v2(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        
        delta_hess = self.hessian_update(displacement, delta_grad)

        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        RFO_eigenvalue, _ = np.linalg.eig(matrix_for_RFO)
        RFO_eigenvalue = np.sort(RFO_eigenvalue)
        lambda_for_calc = float(RFO_eigenvalue[self.saddle_order])

        hess_eigenvalue, hess_eigenvector = np.linalg.eig(new_hess)
        hess_eigenvector = hess_eigenvector.T
        move_vector = np.zeros((len(geom_num_list)*3, 1))
        DELTA_for_QNM = self.DELTA
        
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)
        hess_eigenvector = hess_eigenvector.astype(np.float64)
        
        for i in range(len(hess_eigenvalue)):
            
            tmp_vector = np.array([hess_eigenvector[i].T], dtype="float64")
            move_vector += DELTA_for_QNM * np.dot(tmp_vector, B_g.reshape(len(geom_num_list)*3, 1)) * tmp_vector.T / (hess_eigenvalue[i] - lambda_for_calc + 1e-8)
            
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        move_vector = move_vector.reshape(len(geom_num_list), 3)
        self.hessian += delta_hess 
        self.iter += 1
     
        return move_vector#Bohr.
        
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("normal mode")
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        print("saddle order:", self.saddle_order)
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        DELTA_for_QNM = self.DELTA
        

        delta_hess = self.hessian_update(displacement, delta_grad)
        
        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
            
        #move_vector = DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        DELTA_for_QNM = self.DELTA
        
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM)
        
            
        move_vector = move_vector.reshape(len(geom_num_list), 3)
        self.hessian += delta_hess 
        self.iter += 1
        return move_vector
        
    def moment(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("moment mode")
        print("saddle order:", self.saddle_order)

        if self.Initialization:
            self.momentum_disp = geom_num_list * 0.0
            self.momentum_grad = geom_num_list * 0.0
            self.Initialization = False
            return -1*self.DELTA*B_g
            
        
        if self.iter == 1:
            self.momentum_disp = geom_num_list - pre_geom
            self.momentum_grad = B_g - pre_B_g

        
        delta_grad = (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        

        new_momentum_disp = self.beta * self.momentum_disp + (1.0 - self.beta) * geom_num_list
        new_momentum_grad = self.beta * self.momentum_grad + (1.0 - self.beta) * B_g
        
        delta_momentum_disp = (new_momentum_disp - self.momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - self.momentum_grad).reshape(len(geom_num_list)*3, 1)
        
    
        
        delta_hess = self.hessian_update(delta_momentum_disp, delta_momentum_grad)
        

        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
            
        DELTA_for_QNM = self.DELTA

        matrix_for_RFO = np.append(new_hess, B_g.reshape(len(geom_num_list)*3, 1), axis=1)
        tmp = np.array([np.append(B_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
        
        matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
        eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
        eigenvalue = np.sort(eigenvalue)
        lambda_for_calc = float(eigenvalue[self.saddle_order])
        

        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3)), B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
    
        print("lambda   : ",lambda_for_calc)
        print("step size: ",DELTA_for_QNM,"\n")
        self.hessian += delta_hess
        self.momentum_disp = new_momentum_disp
        self.momentum_grad = new_momentum_grad
        self.iter += 1
        return move_vector#Bohr.   

    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        if "mRFO" in self.config["method"]:
            move_vector = self.moment(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        elif "RFO2" in self.config["method"]:
            move_vector = self.normal_v2(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        else:
            move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector
 

class Newton:
    def __init__(self, **config):
        self.config = config
        self.hess_update = ModelHessianUpdate()
        self.Initialization = True
        self.linesearchflag = False
        
        self.DELTA = 0.5
        self.FC_COUNT = -1 #
        self.saddle_order = 0 #
        self.iter = 0 #
        self.beta = 0.5
        return
    
    def set_hessian(self, hessian):
        self.hessian = hessian
        return

    def set_bias_hessian(self, bias_hessian):
        self.bias_hessian = bias_hessian
        return
    
    def hessian_update(self, displacement, delta_grad):
        if "MSP" in self.config["method"]:
            print("MSP_quasi_newton_method")
            delta_hess = self.hess_update.MSP_hessian_update(self.hessian, displacement, delta_grad)
        elif "BFGS" in self.config["method"]:
            print("BFGS_quasi_newton_method")
            delta_hess = self.hess_update.BFGS_hessian_update(self.hessian, displacement, delta_grad)
        elif "FSB" in self.config["method"]:
            print("FSB_quasi_newton_method")
            delta_hess = self.hess_update.FSB_hessian_update(self.hessian, displacement, delta_grad)
        elif "Bofill" in self.config["method"]:
            print("Bofill_quasi_newton_method")
            delta_hess = self.hess_update.Bofill_hessian_update(self.hessian, displacement, delta_grad)
        else:
            raise "method error"
        return delta_hess
    
    def normal(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("normal mode")
        if self.linesearchflag:
            print("linesearch mode")
        if self.Initialization:
            self.Initialization = False
            return -1*self.DELTA*B_g
        
        delta_grad = (g - pre_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        
        delta_hess = self.hessian_update(displacement, delta_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
            
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)
        
        if self.iter > 1 and self.linesearchflag:
            LS = LineSearch(self.prev_move_vector, move_vector, B_g, pre_B_g, B_e, pre_B_e)
            new_move_vector, optimal_step_flag = LS.linesearch()
        else:
            new_move_vector = move_vector
            optimal_step_flag = True
        
        print("step size: ",DELTA_for_QNM,"\n")


        if self.iter > 0 and self.linesearchflag:
            move_vector = new_move_vector.reshape(len(geom_num_list), 3)
            if optimal_step_flag or self.iter == 1:
                self.prev_move_vector = move_vector    
                
        if not self.linesearchflag or np.linalg.norm(move_vector) > 1e-4:
            self.hessian += delta_hess       
            
        self.iter += 1
        return move_vector#Bohr
  
    # arXiv:2307.13744v1
    def moment(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g):
        print("moment mode")
        if self.Initialization:
            self.momentum_disp = geom_num_list * 0.0
            self.momentum_grad = geom_num_list * 0.0
            self.Initialization = False
            return -1*self.DELTA*B_g
        
        if self.iter == 1:
            self.momentum_disp = geom_num_list - pre_geom
            self.momentum_grad = B_g - pre_B_g

        
        delta_grad = (B_g - pre_B_g).reshape(len(geom_num_list)*3, 1)
        displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
        
        

        new_momentum_disp = self.beta * self.momentum_disp + (1.0 - self.beta) * geom_num_list
        new_momentum_grad = self.beta * self.momentum_grad + (1.0 - self.beta) * B_g
        
        delta_momentum_disp = (new_momentum_disp - self.momentum_disp).reshape(len(geom_num_list)*3, 1)
        delta_momentum_grad = (new_momentum_grad - self.momentum_grad).reshape(len(geom_num_list)*3, 1)
        

        delta_hess = self.hessian_update(delta_momentum_disp, delta_momentum_grad)


        if self.iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
            new_hess = self.hessian + delta_hess + self.bias_hessian
        else:
            new_hess = self.hessian + self.bias_hessian
        
        DELTA_for_QNM = self.DELTA
        
        
        #move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), B_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
        move_vector = DELTA_for_QNM * np.linalg.solve(new_hess, B_g.reshape(len(geom_num_list)*3, 1)).reshape(len(geom_num_list), 3)

        
        print("step size: ",DELTA_for_QNM,"\n")
        self.hessian += delta_hess
        self.iter += 1
        self.momentum_disp = new_momentum_disp
        self.momentum_grad = new_momentum_grad
        return move_vector 
    
    def run(self, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g):
        if self.config["method"] == "mBFGS" or self.config["method"] == "mFSB" or self.config["method"] == "mMSP" or self.config["method"] == "mBofill":
            move_vector = self.moment(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        else:
            move_vector = self.normal(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_g, g)
        return move_vector


class CalculateMoveVector:
    def __init__(self, DELTA, trust_radii, element_list, saddle_order=0,  FC_COUNT=-1, temperature=0.0):
        self.DELTA = DELTA
        self.temperature = temperature
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.unitval = UnitValueLib()
        self.FC_COUNT = FC_COUNT
        self.MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.RMS_FORCE_SWITCHING_THRESHOLD = 0.0008
        self.trust_radii = trust_radii
        self.saddle_order = saddle_order
        self.iter = 0
        self.element_list = element_list
        self.trust_radii_update = "legacy"
    
    def initialization(self, method):
        optimizer_instances = []
        newton_tag = []
        for i, m in enumerate(method):
            # group of steepest descent
            if m == "AdaBelief":
                optimizer_instances.append(Adabelief())
                newton_tag.append(False)
            elif m == "RADAM":
                optimizer_instances.append(RADAM())
                newton_tag.append(False)
            elif m == "Adamod":
                optimizer_instances.append(Adamod())       
                newton_tag.append(False)                   
            elif m == "YOGI":
                optimizer_instances.append(YOGI())      
                newton_tag.append(False)            
            elif m == "SADAM":
                optimizer_instances.append(SAdam()) 
                newton_tag.append(False)
            elif m == "QHADAM":
                optimizer_instances.append(QHAdam())
                newton_tag.append(False)
            elif m == "SAMSGrad":
                optimizer_instances.append(SAMSGrad())
                newton_tag.append(False)
            elif m == "Adam":
                optimizer_instances.append(Adam()) 
                newton_tag.append(False)
            elif m == "Adadelta":
                optimizer_instances.append(Adadelta())
                newton_tag.append(False)
            elif m == "AdamW":
                optimizer_instances.append(AdamW())
                newton_tag.append(False)
            elif m == "AdaDiff":
                optimizer_instances.append(AdaDiff())
                newton_tag.append(False)
            elif m == "Adafactor":
                optimizer_instances.append(Adafactor())
                newton_tag.append(False)
            elif m == "Adabound":
                optimizer_instances.append(AdaBound())
                newton_tag.append(False)
            elif m == "EVE":
                optimizer_instances.append(EVE())
                newton_tag.append(False)
            elif m == "Prodigy":
                optimizer_instances.append(Prodigy())
                newton_tag.append(False)
            elif m == "AdaMax":
                optimizer_instances.append(AdaMax())
                newton_tag.append(False)
            elif m == "NAdam":    
                optimizer_instances.append(NAdam())
                newton_tag.append(False)
            elif m == "FIRE":
                optimizer_instances.append(FIRE())
                newton_tag.append(False)
            elif m == "third_order_momentum_Adam":
                optimizer_instances.append(third_order_momentum_Adam())
                newton_tag.append(False)
            elif m == "Adaderivative":
                optimizer_instances.append(Adaderivative())
                newton_tag.append(False)
            elif m == "CG" or m == "CG_PR" or m == "CG_FR" or m == "CG_HS" or m == "CG_DY":
                optimizer_instances.append(ConjgateGradient(method=m))
                newton_tag.append(False)
            # group of quasi-Newton method
               
            elif m == "BFGS" or m == "FSB" or m == "Bofill" or m == "MSP":
                optimizer_instances.append(Newton(method=m))
                optimizer_instances[i].DELTA = 0.10
                newton_tag.append(True)
            elif m == "mBFGS" or m == "mFSB" or m == "mBofill" or m == "mMSP":
                optimizer_instances.append(Newton(method=m))
                optimizer_instances[i].DELTA = 0.10
                newton_tag.append(True)
            elif m == "BFGS_LS" or m == "FSB_LS" or m == "Bofill_LS" or m == "MSP_LS":
                optimizer_instances.append(Newton(method=m))
                optimizer_instances[i].DELTA = 0.5
                optimizer_instances[i].linesearchflag = True
                newton_tag.append(True)
            elif m == "RFO_BFGS" or m == "RFO_FSB" or m == "RFO_Bofill" or m == "RFO_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m))
                optimizer_instances[i].DELTA = 0.50
                newton_tag.append(True)
            elif m == "RFO2_BFGS" or m == "RFO2_FSB" or m == "RFO2_Bofill" or m == "RFO2_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m))
                optimizer_instances[i].DELTA = 0.50
                newton_tag.append(True)
            elif m == "mRFO_BFGS" or m == "mRFO_FSB" or m == "mRFO_Bofill" or m == "mRFO_MSP":
                optimizer_instances.append(RationalFunctionOptimization(method=m))
                optimizer_instances[i].DELTA = 0.30
                newton_tag.append(True)
            else:
                
                print("This method is not implemented. :", m, " Thus, Default method is used.")
                optimizer_instances.append(Adabelief())
                newton_tag.append(False)
                
        self.method = method
        self.newton_tag = newton_tag
        return optimizer_instances
        
    def update_trust_radii(self, trust_radii, B_e, pre_B_e, pre_B_g, pre_move_vector):
    
        if self.trust_radii_update == "trust":
            Sc = 2.0
            Ce = (np.dot(pre_B_g.reshape(1, len(self.geom_num_list)*3), pre_move_vector.reshape(len(self.geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot(pre_move_vector.reshape(1, len(self.geom_num_list)*3), self.Model_hess.model_hess), pre_move_vector.reshape(len(self.geom_num_list)*3, 1)))
            r = (B_e - pre_B_e) / Ce
            
            if r < 0.25:
                trust_radii /= 2*Sc
            
            elif r > 0.25 and (trust_radii - np.linalg.norm(pre_move_vector)) < 1e-3:
                trust_radii *= Sc ** 0.5
            else:
                pass
        elif self.trust_radii_update == "legacy":
            if pre_B_e >= B_e:
                trust_radii *= 3.0
            else:
                trust_radii *= 0.1
        else:
            pass
                                   
        return np.clip(trust_radii, 0.01, 1.0)


    def diag_hess_and_display(self, optimizer_instance):
        #------------------------------------------------------------
        # diagonize hessian matrix and display eigenvalues
        #-----------------------------------------------------------
        hess_eigenvalue, _ = np.linalg.eig(optimizer_instance.hessian + optimizer_instance.bias_hessian)
        hess_eigenvalue = hess_eigenvalue.astype(np.float64)#not display imagnary values 
        print("NORMAL MODE EIGENVALUE:\n",np.sort(hess_eigenvalue),"\n")
        return

  
    def calc_move_vector(self, iter, geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g, optimizer_instances):#geom_num_list:Bohr
        self.iter = iter
        self.geom_num_list = geom_num_list
        move_vector_list = []
        #-------------------------------------------------------------
        #update trust radii
        #-------------------------------------------------------------
        if self.iter % self.FC_COUNT == 0 and self.FC_COUNT != -1:
            self.trust_radii = 0.01
        elif self.FC_COUNT == -1:
            self.trust_radii = 1.0
        else:
            self.trust_radii = self.update_trust_radii(self.trust_radii, B_e, pre_B_e, pre_B_g, pre_move_vector)
            if self.saddle_order > 0:
                self.trust_radii = min(self.trust_radii, 0.01)
            
        #---------------------------------
        #calculate move vector
        #---------------------------------
        for i in range(len(optimizer_instances)):
            move_vector_list.append(optimizer_instances[i].run(geom_num_list, B_g, pre_B_g, pre_geom, B_e, pre_B_e, pre_move_vector, initial_geom_num_list, g, pre_g))
            
            
        #---------------------------------
        # switich step update method
        #---------------------------------
        if len(move_vector_list) > 1:
            if abs(B_g.max()) < self.MAX_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(B_g).mean())) < self.RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method:", self.method[1])
                if self.newton_tag[1]:
                    self.diag_hess_and_display(optimizer_instances[1])
                        
            else:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method:", self.method[0])
                if self.newton_tag[0]:
                    self.diag_hess_and_display(optimizer_instances[0])
        else:
            move_vector = copy.copy(move_vector_list[0])
            if self.newton_tag[0]:
                self.diag_hess_and_display(optimizer_instances[0])
        # add perturbation (toy function)
        P = Perturbation(temperature=self.temperature)
        perturbation = P.boltzmann_dist_perturb(move_vector)
        
        move_vector += perturbation
        print("perturbation: ", np.linalg.norm(perturbation))

        #-------------------------------------------------------------
        #display trust radii
        #-------------------------------------------------------------
        if np.linalg.norm(move_vector) > self.trust_radii:
            move_vector = self.trust_radii * move_vector/np.linalg.norm(move_vector)
        print("trust radii: ", self.trust_radii)
        print("step  radii: ", np.linalg.norm(move_vector))

        new_geometry = (geom_num_list - move_vector) * self.unitval.bohr2angstroms 
        #---------------------------------
        return new_geometry, np.array(move_vector, dtype="float64"), optimizer_instances, self.trust_radii


