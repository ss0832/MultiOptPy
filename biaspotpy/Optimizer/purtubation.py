import numpy as np

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
        perturbation = self.DELTA * np.sqrt(2.0 * self.damping_coefficient * self.Boltzmann_constant * temperature) * np.random.normal(loc=0.0, scale=1.0, size=len(move_vector)).reshape(len(move_vector), 1)

        return perturbation
