
from multioptpy.Parameters.parameter import UnitValueLib, UFF_VDW_distance_lib, UFF_VDW_well_depth_lib
from multioptpy.Optimizer.fire import FIRE

import numpy as np
import torch
import copy


class SpacerModelPotential:
    def __init__(self, mm_pot_type="UFF", **kwarg):
        if mm_pot_type == "UFF":
            self.VDW_distance_lib = UFF_VDW_distance_lib #function
            self.VDW_well_depth_lib = UFF_VDW_well_depth_lib #function
        else:
            raise "Unexpected MM potential type"
        self.config = kwarg
        
        UVL = UnitValueLib()
        self.hartree2kcalmol = UVL.hartree2kcalmol 
        self.bohr2angstroms = UVL.bohr2angstroms 
        self.hartree2kjmol = UVL.hartree2kjmol 
        
        self.a = 1.0
        self.num2tgtatomlabel = {}
        for num, tgt_atom_num in enumerate(self.config["spacer_model_potential_target"]):
            self.num2tgtatomlabel[num] = tgt_atom_num - 1

        self.nparticle = self.config["spacer_model_potential_particle_number"]
        self.element_list = self.config["element_list"]
        self.natom = len(self.element_list)
        self.particle_num_list = None

        self.file_directory = self.config["directory"]
        self.lj_repulsive_order = 12.0
        self.lj_attractive_order = 6.0
        
        self.micro_iteration = 5000 * self.config["spacer_model_potential_particle_number"]
        self.rand_search_iteration = 250 * self.config["spacer_model_potential_particle_number"]
        self.threshold = 2e-6
        self.init = True
        return
    
    def save_state(self):      
        with open(self.file_directory + "/spacer.xyz", "a") as f:
            f.write(str(len(self.tmp_geom_num_list_for_save)) + "\n")
            f.write("spacer\n")
            for i in range(len(self.tmp_geom_num_list_for_save)):
                f.write(self.tmp_element_list_for_save[i] + " " + str(self.tmp_geom_num_list_for_save[i][0].item()) + " " + str(self.tmp_geom_num_list_for_save[i][1].item()) + " " + str(self.tmp_geom_num_list_for_save[i][2].item()) + "\n")
        
        return
    

    def lennard_johns_potential(self, distance, sigma, epsilon):
        ene = epsilon * ((sigma/distance) ** self.lj_repulsive_order -2 * (sigma/distance) ** self.lj_attractive_order)
        return ene


    def morse_potential(self, distance, sigma, epsilon):
        ene = epsilon * (torch.exp(-2 * self.a * (distance - sigma)) -2 * torch.exp(-1 * self.a * (distance - sigma)))
        return ene

    
    def barrier_switching_potential(self, distance, sigma):
        normalized_distance = distance / sigma
        min_norm_dist = 0.9
        max_norm_dist = 1.0
        const = 0.5
        
        in_range = (normalized_distance >= min_norm_dist) & (normalized_distance < max_norm_dist)
        out_of_range = normalized_distance >= max_norm_dist
        ene = torch.zeros_like(normalized_distance) 

        normalized_diff = (normalized_distance - min_norm_dist) / (max_norm_dist - min_norm_dist)
        ene[in_range] = -const * (
            1 - 10 * normalized_diff[in_range]**3
            + 15 * normalized_diff[in_range]**4
            - 6 * normalized_diff[in_range]**5
        ) + const

        ene[out_of_range] = const * normalized_distance[out_of_range]
        ene = torch.sum(ene)
        return ene

    
    def calc_potential(self, geom_num_list, particle_num_list, bias_pot_params):
        energy = 0.0
        particle_sigma = self.config["spacer_model_potential_distance"] / self.bohr2angstroms
        particle_epsilon = self.config["spacer_model_potential_well_depth"] / self.hartree2kjmol
        #atom-particle interactions
        spacer_indices = torch.tensor([i for i in range(len(self.config["element_list"]))])
        geom_particles = geom_num_list[spacer_indices]  # shape: (M, 3), M = len(spacer_model_potential_target)
        atom_sigmas = torch.tensor([self.VDW_distance_lib(self.config["element_list"][idx]) for idx in spacer_indices])
        atom_epsilons = torch.tensor([self.VDW_well_depth_lib(self.config["element_list"][idx]) for idx in spacer_indices])
        sigma = particle_sigma + atom_sigmas.unsqueeze(1)  # shape: (M, 1) + (1, N) -> (M, N)
        epsilon = torch.sqrt(particle_epsilon * atom_epsilons.unsqueeze(1))  # shape: (M, 1)
        diffs = geom_particles.unsqueeze(1) - particle_num_list.unsqueeze(0)  # shape: (M, N, 3)
        distances = torch.linalg.norm(diffs, dim=-1)  # shape: (M, N)
        pairwise_energies = self.lennard_johns_potential(distances, sigma, epsilon)
        energy = energy + pairwise_energies.sum()

        #particle-particle interactions
        diffs = particle_num_list.unsqueeze(1) - particle_num_list.unsqueeze(0)  # shape: (N, N, 3)
        distances = torch.linalg.norm(diffs, dim=-1)  # shape: (N, N), diagonal is 0 (self distance)
        i, j = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
        pairwise_distances = distances[i, j]  
        pairwise_energies = self.lennard_johns_potential(pairwise_distances, 2 * particle_sigma, particle_epsilon)
        energy = energy + pairwise_energies.sum()

        #avoid scattering particle to outside of cavity
        target_geom = geom_num_list[np.array(self.config["spacer_model_potential_target"]) - 1]

        norm_diff = torch.abs(torch.linalg.norm(target_geom, dim=1).unsqueeze(1) - torch.linalg.norm(particle_num_list, dim=1).unsqueeze(0))
        min_dist, min_idx = torch.min(norm_diff, dim=0)
     
        element_indices = [self.num2tgtatomlabel[idx.item()] for idx in min_idx]
        atom_sigmas = self.config["spacer_model_potential_cavity_scaling"] * torch.tensor(
            [self.VDW_distance_lib(self.config["element_list"][idx]) for idx in element_indices]
        )

        energy =  energy + self.barrier_switching_potential(min_dist, atom_sigmas)

        self.tmp_geom_num_list_for_save = torch.cat([geom_num_list, particle_num_list], dim=0) * self.bohr2angstroms
        self.tmp_element_list_for_save = self.config["element_list"] + ["He"] * len(particle_num_list)
        
        return energy
    
    def rand_search(self, geom_num_list, bias_pot_params):
        max_energy = 1e+10
        print("rand_search")
        for i in range(self.rand_search_iteration):
            center = torch.mean(geom_num_list[np.array(self.config["spacer_model_potential_target"])-1], dim=0)
            tmp_particle_num_list = torch.normal(mean=0, std=100, size=(self.config["spacer_model_potential_particle_number"], 3)) + center
            energy = self.calc_potential(geom_num_list, tmp_particle_num_list, bias_pot_params)
            if energy < max_energy:
                max_energy = energy
                self.particle_num_list = tmp_particle_num_list
        print("rand_search done")
        print("max_energy: ", max_energy.item())
        return
    
    
    def microiteration(self, geom_num_list, bias_pot_params):
        nparticle = self.config["spacer_model_potential_particle_number"]
        if self.init:
            self.rand_search(geom_num_list, bias_pot_params)
            self.init = False
        
        prev_particle_grad = torch.zeros_like(self.particle_num_list)
        Opt = FIRE()
        Opt.display_flag = False
        
        for j in range(self.micro_iteration):

            
            particle_grad = torch.func.jacrev(self.calc_potential, argnums=1)(geom_num_list, self.particle_num_list, bias_pot_params)
            if torch.linalg.norm(particle_grad) < self.threshold:
                print("Converged!")
                print("M. itr: ", j)
                break
            if j == self.micro_iteration - 1:
                print("Not converged!")
                break
            
            tmp_particle_list = copy.copy(self.particle_num_list.clone().detach().numpy()).reshape(3*nparticle, 1)
            tmp_particle_grad = copy.copy(particle_grad.clone().detach().numpy()).reshape(3*nparticle, 1)
            tmp_prev_particle_grad = copy.copy(prev_particle_grad.clone().detach().numpy()).reshape(3*nparticle, 1)

            move_vector = Opt.run(tmp_particle_list, tmp_particle_grad, tmp_prev_particle_grad, pre_geom=[], B_e=0.0, pre_B_e=0.0, pre_move_vector=[], initial_geom_num_list=[], g=[], pre_g=[])
            move_vector = torch.tensor(move_vector, dtype=torch.float64).reshape(nparticle, 3)
            self.particle_num_list = self.particle_num_list - 0.5 * move_vector
            # update rot_angle_list
            if j % 100 == 0:
                print("M. itr: ", j)
                print("energy: ", self.calc_potential(geom_num_list, self.particle_num_list, bias_pot_params).item())
                print("particle_grad: ", np.linalg.norm(particle_grad.detach().numpy()))
            

            prev_particle_grad = particle_grad
        
        
        energy = self.calc_potential(geom_num_list, self.particle_num_list, bias_pot_params)
        print("energy: ", self.calc_potential(geom_num_list, self.particle_num_list, bias_pot_params).item())
        return energy
    
    
    def calc_energy(self, geom_num_list, bias_pot_params=[]):
        """
        # required variables:self.config["spacer_model_potential_target"]
                             self.config["spacer_model_potential_distance"]
                             self.config["spacer_model_potential_well_depth"]
                             self.config["spacer_model_potential_cavity_scaling"]
                             self.config["spacer_model_potential_particle_number"]
                             self.config["element_list"]
                             self.config["directory"]
                             
        """
        energy = self.microiteration(geom_num_list, bias_pot_params)
        
        return energy
    
    def calc_pot_for_eff_hess(self, coord_and_ell_angle, bias_pot_params):
        geom_num_list = coord_and_ell_angle[:len(self.element_list)*3].reshape(-1, 3)
        particle_num_list = coord_and_ell_angle[len(self.element_list)*3:].reshape(self.nparticle, 3)
        energy = self.calc_potential(geom_num_list, particle_num_list, bias_pot_params)
        return energy


    def calc_eff_hessian(self, geom_num_list, bias_pot_params):
        transformed_geom_num_list = geom_num_list.reshape(-1, 1)
        transformed_particle_num_list = self.particle_num_list.reshape(-1, 1)
        coord_and_particle = torch.cat((transformed_geom_num_list, transformed_particle_num_list), dim=0)
        combined_hess = torch.func.hessian(self.calc_pot_for_eff_hess, argnums=0)(coord_and_particle, bias_pot_params).reshape(len(self.element_list)*3+self.nparticle*3, len(self.element_list)*3+self.nparticle*3)
        coupling_hess_1 = combined_hess[:len(self.element_list)*3, len(self.element_list)*3:]
        coupling_hess_2 = combined_hess[len(self.element_list)*3:, :len(self.element_list)*3]
        angle_hess = combined_hess[len(self.element_list)*3:, len(self.element_list)*3:]
        eff_hess = -1 * torch.matmul(torch.matmul(coupling_hess_1, torch.linalg.inv(angle_hess)), coupling_hess_2)
        return eff_hess

    
#[[solvent particle well depth (kJ/mol)] [solvent particle e.q. distance (ang.)] [scaling of cavity (2.0)] [number of particles] [target atoms (2,3-5)] ...]