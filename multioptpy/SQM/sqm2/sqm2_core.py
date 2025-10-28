import numpy as np
import torch
from multioptpy.SQM.sqm2.sqm2_data import SQM2Parameters
from multioptpy.SQM.sqm2.sqm2_rep import RepulsionCalculator
from multioptpy.SQM.sqm2.sqm2_srb import SRBCalculator
from multioptpy.SQM.sqm2.sqm2_disp import DispersionCalculator
from multioptpy.SQM.sqm2.sqm2_eeq import IESEnergyCalculator
from multioptpy.SQM.sqm2.sqm2_qm import EHTCalculator
from multioptpy.SQM.sqm2.sqm2_basis import BasisSet
from multioptpy.SQM.sqm2.sqm2_bond import BondCalculator

ANG2BOHR = 1.8897261246257704

class SQM2Calculator:
    def __init__(self, xyz, element_list, charge, spin):# xyz: (N,3) in Angstrom, element_list: atomic numbers (1-indexed)
        # ref.: https://doi.org/10.26434/chemrxiv.8326202.v1
        sqm_params = SQM2Parameters()
        self.element_list = np.array(element_list, dtype=np.int64) - 1 # 0-indexed
        self.xyz = np.array(xyz, dtype=np.float64)
        self.xyz = self.xyz * ANG2BOHR  # Angstrom -> Bohr
        self.charge = charge
        self.spin = spin
        
        self.repulsion_calculator = RepulsionCalculator(self.element_list, sqm_params)
        self.srb_calculator = SRBCalculator(self.element_list, sqm_params)
        self.dispersion_calculator = DispersionCalculator(self.element_list, sqm_params)        
        self.ies_calculator = IESEnergyCalculator(self.element_list, charge, sqm_params)
        self.basis_set = BasisSet(self.element_list, sqm_params)
        #print(self.basis_set.basis)
        self.eht_calculator = EHTCalculator(self.element_list, charge, spin, sqm_params, self.basis_set)
        self.bond_calculator = BondCalculator(self.element_list, sqm_params)    
        self.sqm_params = sqm_params
        
    def get_overlap_matrix(self): 
        if not self.eht_calculator:
            raise ValueError("EHT calculator is not initialized.")
        return self.eht_calculator.get_overlap_integral_matrix()

    def get_eht_mo_energy(self):
        if not self.eht_calculator:
            raise ValueError("EHT calculator is not initialized.")
        return self.eht_calculator.get_mo_energy()
    
    def get_eht_mo_coeff(self):
        if not self.eht_calculator:
            raise ValueError("EHT calculator is not initialized.")
        return self.eht_calculator.get_mo_coeff()

    def total_energy(self, xyz): # xyz: ndarray (N,3) in Angstrom
        xyz = xyz * ANG2BOHR  # Angstrom -> Bohr

        repulsion_energy = self.repulsion_calculator.energy(xyz)
        srb_energy = self.srb_calculator.energy(xyz)
        ies_energy = self.ies_calculator.energy(xyz)

        eeq_charge = self.ies_calculator.eeq_charge(xyz)
        eeq_charge = eeq_charge.detach().numpy()
        
        cn = self.ies_calculator.cn(xyz)
        cn = cn.detach().numpy()
        dispersion_energy = self.dispersion_calculator.energy(xyz)
        eht_energy = self.eht_calculator.energy(xyz, eeq_charge, cn)
        #bond_energy = self.bond_calculator.energy(xyz) 
        total_energy = repulsion_energy + srb_energy + ies_energy + dispersion_energy + eht_energy
        total_energy = float(total_energy.item())
        print("Total Energy:", total_energy)
        
        return total_energy # in Hartree: float

    def total_gradient(self, xyz): # xyz: ndarray (N,3) in Angstrom
        xyz = xyz * ANG2BOHR  # Angstrom -> Bohr
        eeq_charge = self.ies_calculator.eeq_charge(xyz)
        eeq_charge = eeq_charge.detach().numpy()
        _, d_eeq_charge = self.ies_calculator.d_eeq_charge_d_xyz(xyz)
        #print("d_eeq_charge:", d_eeq_charge)
        d_eeq_charge = d_eeq_charge.detach().numpy()
        cn = self.ies_calculator.cn(xyz)
        cn = cn.detach().numpy()
        _, d_cn = self.ies_calculator.d_cn_d_xyz(xyz)
        #print("d_cn:", d_cn)
        d_cn = d_cn.detach().numpy()
        #eht_energy = self.eht_calculator.energy(xyz, eeq_charge, cn)#, eeq_charge, cn, d_eeq_charge, d_cn)
        eht_energy, eht_gradient = self.eht_calculator.gradient(xyz, eeq_charge, cn, d_eeq_charge, d_cn)
        eht_gradient = torch.nan_to_num(eht_gradient, nan=0.0, posinf=0.0, neginf=0.0)
        repulsion_energy, repulsion_gradient = self.repulsion_calculator.gradient(xyz)
        repulsion_gradient = torch.nan_to_num(repulsion_gradient, nan=0.0, posinf=0.0, neginf=0.0)

        srb_energy, srb_gradient = self.srb_calculator.gradient(xyz)
        srb_gradient = torch.nan_to_num(srb_gradient, nan=0.0, posinf=0.0, neginf=0.0)
        ies_energy, ies_gradient = self.ies_calculator.gradient(xyz)
        ies_gradient = torch.nan_to_num(ies_gradient, nan=0.0, posinf=0.0, neginf=0.0)
        dispersion_energy, dispersion_gradient = self.dispersion_calculator.gradient(xyz)
        dispersion_gradient = torch.nan_to_num(dispersion_gradient, nan=0.0, posinf=0.0, neginf=0.0)
        
        total_energy = repulsion_energy + srb_energy + ies_energy + dispersion_energy + eht_energy
        total_gradient = repulsion_gradient + srb_gradient + ies_gradient + dispersion_gradient + eht_gradient
        print("Total gradient norm:", torch.norm(total_gradient).item())
        total_energy = float(total_energy.item())
        total_gradient = total_gradient.detach().numpy()
        return total_energy, total_gradient # gradient: ndarray (N,3) in Bohr

    def total_hessian(self, xyz): # xyz: ndarray (N,3) in Angstrom
        xyz = xyz * ANG2BOHR  # Angstrom -> Bohr
        eeq_charge = self.ies_calculator.eeq_charge(xyz)
        eeq_charge = eeq_charge.detach().numpy()

        _, d_eeq_charge = self.ies_calculator.d_eeq_charge_d_xyz(xyz)
        _, dd_eeq_charge = self.ies_calculator.d2_eeq_charge_d_xyz2(xyz)
        d_eeq_charge = d_eeq_charge.detach().numpy()
        dd_eeq_charge = dd_eeq_charge.detach().numpy()
        #print("dd_eeq_charge:", dd_eeq_charge)

        
        cn = self.ies_calculator.cn(xyz)
        cn = cn.detach().numpy()
        _, d_cn = self.ies_calculator.d_cn_d_xyz(xyz)
        d_cn = d_cn.detach().numpy()
        _, dd_cn = self.ies_calculator.d2_cn_d_xyz2(xyz)
        dd_cn = dd_cn.detach().numpy()
        #print("dd_cn:", dd_cn)        
        
        
        eht_hessian = self.eht_calculator.hessian(xyz, eeq_charge, cn, d_eeq_charge, dd_eeq_charge, d_cn, dd_cn)
        eht_hessian = torch.nan_to_num(eht_hessian, nan=0.0, posinf=0.0, neginf=0.0)
        
        repulsion_energy, repulsion_hessian = self.repulsion_calculator.hessian(xyz)
        repulsion_hessian = torch.nan_to_num(repulsion_hessian, nan=0.0, posinf=0.0, neginf=0.0)
        srb_energy, srb_hessian = self.srb_calculator.hessian(xyz)
        srb_hessian = torch.nan_to_num(srb_hessian, nan=0.0, posinf=0.0, neginf=0.0)
        ies_energy, ies_hessian = self.ies_calculator.hessian(xyz)
        ies_hessian = torch.nan_to_num(ies_hessian, nan=0.0, posinf=0.0, neginf=0.0)
        dispersion_energy, dispersion_hessian = self.dispersion_calculator.hessian(xyz)
        dispersion_hessian = torch.nan_to_num(dispersion_hessian, nan=0.0, posinf=0.0, neginf=0.0)
        #bond_energy, bond_hessian = self.bond_calculator.hessian(xyz)
        #bond_hessian = torch.nan_to_num(bond_hessian, nan=0.0, posinf=0.0, neginf=0.0)
        
        total_energy = repulsion_energy + srb_energy + ies_energy + dispersion_energy #+ bond_energy
        total_hessian = eht_hessian + repulsion_hessian + srb_hessian + ies_hessian + dispersion_hessian #+ bond_hessian

        total_energy = float(total_energy.item())
        total_hessian = total_hessian.detach().numpy()
        
        return total_hessian # hessian: ndarray (3N,3N) in Bohr


if __name__ == "__main__":
    
    print("\nStarting structural optimization for H2O using steepest descent...\n")    
    element_list = np.array([8, 1, 1])# O, H, H  
    xyz = np.array([[0.0000,  0.0000,  0.0173], 
                    [0.0000,  0.7572, -0.4692],
                    [0.0000, -0.7572, -0.4692]])#(Angstrom)
    xyz = np.array(xyz)
    charge = 0 
    spin = 0
    
    
    calculator = SQM2Calculator(xyz, element_list, charge, spin)
    
    E_total = calculator.total_energy(xyz)    
        

    
    # Parameters for optimization
    max_iter = 40
    lr = 0.1  # Learning rate (adjusted for Angstrom units; may need tuning)
    threshold = 1e-3  # Convergence threshold for gradient norm
    xyz_ang = xyz
    for iter in range(max_iter):
   
        # Get total energy and gradient (gradient is dE/dx where x internal is Bohr)
        total_energy, total_gradient_bohr = calculator.total_gradient(xyz_ang)
        
        total_gradient_ang = total_gradient_bohr * ANG2BOHR
        
        # Update xyz in Angstrom
        xyz_ang = xyz_ang - lr * total_gradient_ang
   
        # Check convergence
        grad_norm = np.linalg.norm(total_gradient_ang)
        print(f"Iteration {iter + 1}: Energy = {total_energy:.6f}, Gradient Norm = {grad_norm:.6f}")
        xyz_ang = xyz_ang
        if grad_norm < threshold:
            print("Optimization converged.")
            break
    
    # Display optimized structure
    optimized_xyz = xyz_ang
    elements = ['O', 'H', 'H']  # Corresponding to atomic numbers [8,1,1]
    
    print("\nOptimized Structure:")
    for i in range(len(elements)):
        print(f"{elements[i]} {optimized_xyz[i, 0]:.6f} {optimized_xyz[i, 1]:.6f} {optimized_xyz[i, 2]:.6f}")
    
    print("\nStarting structural optimization for H3N using steepest descent...\n")    
    element_list = np.array([5, 1, 1, 1])# B, H, H, H
    xyz = np.array([[  -3.51351354,    0.43156059,    0.00000000], 
                    [   -3.18019165,   -0.51125250,    0.00000000],
                    [  -3.18017444,    0.90296076,    0.81649673],
                    [ -3.18017444,    0.90296076,   -0.81649673]])#(Angstrom)
    xyz = np.array(xyz)
    charge = 0 
    spin = 0
    
    
    calculator = SQM2Calculator(xyz, element_list, charge, spin)
    
    E_total = calculator.total_energy(xyz)    
    
    #=============================================================================================
    
    # Parameters for optimization
    max_iter = 50
    lr = 0.1  # Learning rate (adjusted for Angstrom units; may need tuning)
    threshold = 5e-3  # Convergence threshold for gradient norm
    xyz_ang = xyz
    for iter in range(max_iter):
   
        # Get total energy and gradient (gradient is dE/dx where x internal is Bohr)
        total_energy, total_gradient_bohr = calculator.total_gradient(xyz_ang)
        
        total_gradient_ang = total_gradient_bohr * ANG2BOHR
        
        # Update xyz in Angstrom
        xyz_ang = xyz_ang - lr * total_gradient_ang
   
        # Check convergence
        grad_norm = np.linalg.norm(total_gradient_ang)
        print(f"Iteration {iter + 1}: Energy = {total_energy:.6f}, Gradient Norm = {grad_norm:.6f}")
        xyz_ang = xyz_ang
        if grad_norm < threshold:
            print("Optimization converged.")
            break
    
    # Display optimized structure
    optimized_xyz = xyz_ang
    elements = ['B', 'H', 'H', 'H']  # Corresponding to atomic numbers [5,1,1,1]
    
    print("\nOptimized Structure:")
    for i in range(len(elements)):
        print(f"{elements[i]} {optimized_xyz[i, 0]:.6f} {optimized_xyz[i, 1]:.6f} {optimized_xyz[i, 2]:.6f}")
    
    

    
    print("\nStarting structural optimization for Pd(PH3) using steepest descent...\n")

    element_list = np.array([28, 15, 1, 1, 1])


    xyz = np.array([[0.0000,  0.0000,  2.3000],  # Pd 
                    [0.0000,  0.0000,  0.0000],  # P
                    [1.3900,  0.0000, -0.3300],  # H1 (P-H 1.42 A, Pd-P-H 100 deg)
                    [-0.6950,  1.2038, -0.3300],  # H2
                    [-0.6950, -1.2038, -0.3300]]) # H3 (Angstrom)
  

    xyz = np.array(xyz)
    charge = 0 
    spin = 0


    calculator = SQM2Calculator(xyz, element_list, charge, spin)

    E_total = calculator.total_energy(xyz)      

    # Parameters for optimization
    max_iter = 40
    lr = 0.1  # Learning rate (adjusted for Angstrom units; may need tuning)
    threshold = 1e-2  # Convergence threshold for gradient norm
    xyz_ang = xyz
    for iter in range(max_iter):

        # Get total energy and gradient (gradient is dE/dx where x internal is Bohr)
        total_energy, total_gradient_bohr = calculator.total_gradient(xyz_ang)

       
        total_gradient_ang = total_gradient_bohr * ANG2BOHR

        # Update xyz in Angstrom
        xyz_ang = xyz_ang - lr * total_gradient_ang

        # Check convergence
        grad_norm = np.linalg.norm(total_gradient_ang)
        print(f"Iteration {iter + 1}: Energy = {total_energy:.6f}, Gradient Norm = {grad_norm:.6f}")
        xyz_ang = xyz_ang
        if grad_norm < threshold:
            print("Optimization converged.")
            break
        
    # Display optimized structure
    optimized_xyz = xyz_ang

  
    elements = ['Pd', 'P', 'H', 'H', 'H']  # Corresponding to atomic numbers [46,15,1,1,1]
  
    print("\nOptimized Structure:")
    for i in range(len(elements)):
        print(f"{elements[i]} {optimized_xyz[i, 0]:.6f} {optimized_xyz[i, 1]:.6f} {optimized_xyz[i, 2]:.6f}")
    
    
    
  