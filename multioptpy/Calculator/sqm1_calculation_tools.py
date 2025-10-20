import glob
import os
import copy
import numpy as np
import torch


from multioptpy.SQM.sqm1.sqm1_core import SQM1Calculator, SQM1Parameters, ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, is_covalently_bonded
from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer


"""
Experimental semiempirical electronic structure approach inspired by GFN0-xTB (SQM1)

This module provides calculator utility helpers wrapping the Python implementation
of an experimental semiempirical electronic structure approach inspired by GFN0-xTB.
It mirrors the interface style of tblite_calculation_tools for convenience.
"""

class Calculation:
    def __init__(self, **kwarg):
        if UnitValueLib is not None:
            UVL = UnitValueLib()
            self.bohr2angstroms = UVL.bohr2angstroms
        else:
            self.bohr2angstroms = BOHR_TO_ANGSTROM
        # Optional keys kept for interface parity; use .get to avoid KeyError
        self.START_FILE = kwarg.get("START_FILE")
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL")
        self.FC_COUNT = kwarg.get("FC_COUNT", -1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", "./")
        self.Model_hess = kwarg.get("Model_hess")
        self.unrestrict = kwarg.get("unrestrict", False)
        self.dft_grid = kwarg.get("dft_grid")
        self.hessian_flag = False
        # Load SQM1 parameters (now embedded, no file needed)
        self.params = SQM1Parameters()
        self.device = kwarg.get("device", "cpu")
        self.dtype = kwarg.get("dtype", torch.float64)
        # Calculator instance will be created per calculation with appropriate geometry
        self.calculator = None
        # Distance constraint parameters
        self.use_distance_constraints = kwarg.get("use_distance_constraints", True)
        self.max_distance_deviation = kwarg.get("max_distance_deviation", 0.10)  # 10% default
        self.constraint_penalty_strength = kwarg.get("constraint_penalty_strength", 1000.0)
        self.initial_distances = {}  # Will store bonded pair distances on first calculation
        self.bonded_pairs = set()
        self.constraints_initialized = False

    def _update_distance_constraints(self, positions, element_number_list):
        """
        Update distance constraints for covalently bonded atom pairs.
        This is called every time energy is calculated to track dynamic bond formation/breaking.
        
        Args:
            positions: Atomic positions in Angstrom (n_atoms, 3)
            element_number_list: Atomic numbers
        """
        if not self.use_distance_constraints:
            return
        
        positions = np.array(positions, dtype='float64').reshape(-1, 3)
        n_atoms = len(positions)
        
        # Update bonded pairs based on current geometry
        current_bonded_pairs = set()
        
        # Identify covalently bonded pairs at current geometry
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate distance in Angstrom
                d_angstrom = np.linalg.norm(positions[i] - positions[j])
                
                # Check if atoms are covalently bonded at current geometry
                if is_covalently_bonded(element_number_list[i], element_number_list[j], d_angstrom):
                    current_bonded_pairs.add((i, j))
                    
                    # If this is a new bond, record its initial distance
                    if (i, j) not in self.initial_distances:
                        d_bohr = d_angstrom * ANGSTROM_TO_BOHR
                        self.initial_distances[(i, j)] = d_bohr
        
        # Update the bonded pairs to reflect current state
        self.bonded_pairs = current_bonded_pairs
        self.constraints_initialized = True
        
    def _calculate_distance_constraint_penalty(self, positions):
        """
        Calculate penalty energy and gradient for distance constraint violations.
        
        Args:
            positions: Current positions in Angstrom (n_atoms, 3)
            
        Returns:
            Tuple of (penalty_energy, penalty_gradient)
            - penalty_energy: Penalty energy in Hartree
            - penalty_gradient: Gradient of penalty in Hartree/Bohr (n_atoms, 3)
        """
        if not self.use_distance_constraints or not self.constraints_initialized:
            return 0.0, None
        
        positions = np.array(positions, dtype='float64').reshape(-1, 3)
        n_atoms = len(positions)
        penalty = 0.0
        penalty_gradient = np.zeros((n_atoms, 3), dtype='float64')
        
        # Only apply penalty to CURRENTLY bonded pairs
        for (i, j) in self.bonded_pairs:
            if (i, j) not in self.initial_distances:
                continue  # Skip if we don't have a reference distance
            
            d_init_bohr = self.initial_distances[(i, j)]
            
            # Calculate distance vector in Angstrom
            r_ij_angstrom = positions[i] - positions[j]
            d_current_angstrom = np.linalg.norm(r_ij_angstrom)
            
            # Convert to Bohr
            r_ij_bohr = r_ij_angstrom * ANGSTROM_TO_BOHR
            d_current_bohr = d_current_angstrom * ANGSTROM_TO_BOHR
            
            # Calculate relative deviation (dimensionless)
            deviation = abs(d_current_bohr - d_init_bohr) / d_init_bohr
            
            # Apply penalty if deviation exceeds threshold
            if deviation > self.max_distance_deviation:
                # Energy penalty (in Hartree)
                penalty += self.constraint_penalty_strength * (deviation - self.max_distance_deviation)**2
                
                # Gradient penalty calculation:
                # E = k * (dev - thresh)²
                # dev = |d_curr - d_init| / d_init
                # 
                # Using chain rule:
                # ∂E/∂r_bohr = ∂E/∂dev * ∂dev/∂d_curr * ∂d_curr/∂r_bohr
                # 
                # ∂E/∂dev = 2*k*(dev - thresh)
                # ∂dev/∂d_curr = sign(d_curr - d_init) / d_init
                # ∂d_curr/∂r_bohr = r_ij_bohr / d_curr_bohr (unit vector in Bohr)
                #
                # Combined:
                # ∂E/∂r_bohr = 2*k*(dev - thresh) * sign(d_curr - d_init)/d_init * r_ij_bohr/d_curr_bohr
                
                # Sign of deviation
                sign = 1.0 if d_current_bohr > d_init_bohr else -1.0
                
                # Gradient prefactor (in Hartree/Bohr)
                grad_prefactor = 2.0 * self.constraint_penalty_strength * (deviation - self.max_distance_deviation) * sign / d_init_bohr
                
                # Direction vector (unit vector in Bohr space)
                direction_bohr = r_ij_bohr / d_current_bohr
                
                # Gradient in Hartree/Bohr
                grad_contribution = grad_prefactor * direction_bohr
                
                # Apply to both atoms (i and j)
                penalty_gradient[i] += grad_contribution
                penalty_gradient[j] -= grad_contribution
        
        return penalty, penalty_gradient

    def _calculate_distance_constraint_penalty_hessian(self, positions):
        """
        Calculate Hessian of distance constraint penalty.
        
        Args:
            positions: Current positions in Angstrom (n_atoms, 3)
            
        Returns:
            Hessian of penalty in Hartree/Bohr^2 (3*n_atoms, 3*n_atoms)
        """
        if not self.use_distance_constraints or not self.constraints_initialized:
            return None
        
        positions = np.array(positions, dtype='float64').reshape(-1, 3)
        n_atoms = len(positions)
        penalty_hessian = np.zeros((3*n_atoms, 3*n_atoms), dtype='float64')
        
        # Only apply penalty to CURRENTLY bonded pairs
        for (i, j) in self.bonded_pairs:
            if (i, j) not in self.initial_distances:
                continue  # Skip if we don't have a reference distance
            
            d_init_bohr = self.initial_distances[(i, j)]
            # Calculate distance vector in Angstrom
            r_ij_angstrom = positions[i] - positions[j]
            d_current_angstrom = np.linalg.norm(r_ij_angstrom)
            
            # Convert to Bohr
            r_ij_bohr = r_ij_angstrom * ANGSTROM_TO_BOHR
            d_current_bohr = d_current_angstrom * ANGSTROM_TO_BOHR
            
            # Calculate relative deviation (dimensionless)
            deviation = abs(d_current_bohr - d_init_bohr) / d_init_bohr
            
            # Only calculate if deviation exceeds threshold
            if deviation > self.max_distance_deviation:
                # Sign of deviation
                sign = 1.0 if d_current_bohr > d_init_bohr else -1.0
                
                # Unit vector along bond (in Bohr)
                u = r_ij_bohr / d_current_bohr
                
                # For harmonic penalty E = k * (dev - thresh)²
                # where dev = sign * (d - d0) / d0
                #
                # The Hessian has two terms:
                # H = ∂²E/∂r_i∂r_j
                #
                # For a pair of atoms (atom_a, atom_b):
                # H_aa = projection along bond + projection perpendicular
                # H_ab = -H_aa (Newton's 3rd law)
                # H_bb = H_aa
                #
                # The full formula for harmonic penalty Hessian:
                # H_αβ = (2k/d0²) * [u_α u_β + (dev-thresh)/sign * (δ_αβ - u_α u_β) / d]
                #
                # Simplified for the case where dev > thresh:
                # Second derivative along bond direction:
                k = self.constraint_penalty_strength
                d0 = d_init_bohr
                d = d_current_bohr
                
                # Prefactor for second derivative (in Hartree/Bohr²)
                # ∂²E/∂d² = 2k/d0²
                prefactor = 2.0 * k / (d0 * d0)
                
                # Additional term from directional derivative
                # When dev > thresh, we have an active constraint
                # The Hessian includes both the force constant and geometric terms
                if abs(d - d0) > 1e-10:  # Avoid division by zero
                    # Coefficient for the projection onto bond direction
                    coeff_parallel = prefactor
                    # Coefficient for the projection perpendicular to bond
                    # This comes from the d/dr term in the gradient
                    coeff_perp = prefactor * (deviation - self.max_distance_deviation) * sign / d
                else:
                    coeff_parallel = prefactor
                    coeff_perp = 0.0
                
                # Build 3x3 block for this atom pair
                # H_block = coeff_parallel * u⊗u + coeff_perp * (I - u⊗u)
                identity = np.eye(3, dtype='float64')
                u_outer = np.outer(u, u)
                H_block = coeff_parallel * u_outer + coeff_perp * (identity - u_outer)
                
                # Add to Hessian (symmetric contributions)
                # H_ii += H_block
                penalty_hessian[3*i:3*i+3, 3*i:3*i+3] += H_block
                # H_jj += H_block (symmetric)
                penalty_hessian[3*j:3*j+3, 3*j:3*j+3] += H_block
                # H_ij -= H_block (off-diagonal)
                penalty_hessian[3*i:3*i+3, 3*j:3*j+3] -= H_block
                # H_ji -= H_block (symmetric off-diagonal)
                penalty_hessian[3*j:3*j+3, 3*i:3*i+3] -= H_block
        
        return penalty_hessian

    def numerical_hessian(self, geom_num_list, element_list, total_charge):
        """
        Calculate numerical Hessian using finite differences of gradients.
        
        Args:
            geom_num_list: Atomic positions in Angstrom (n_atoms, 3)
            element_list: Atomic numbers (n_atoms,)
            total_charge: Total molecular charge
            
        Returns:
            Hessian matrix (3*n_atoms, 3*n_atoms) in Hartree/Bohr^2
        """
        numerical_delivative_delta = 1.0e-4  # in Angstrom
        geom_num_list = np.array(geom_num_list, dtype="float64")
        n_atoms = len(geom_num_list)
        hessian = np.zeros((3*n_atoms, 3*n_atoms))
        count = 0
        
        # Update distance constraints based on current geometry
        self._update_distance_constraints(geom_num_list, element_list)
        
        for a in range(n_atoms):
            for i in range(3):
                for b in range(n_atoms):
                    for j in range(3):
                        if count > 3*b + j:
                            continue
                        tmp_grad = []
                        for direction in [1, -1]:
                            shifted = geom_num_list.copy()
                            shifted[a, i] += direction * numerical_delivative_delta
                            # Create calculator for this geometry
                            calc = SQM1Calculator(
                                atomic_numbers=element_list,
                                positions=shifted,
                                charge=total_charge,
                                uhf=0,
                                params=self.params,
                                device=self.device,
                                dtype=self.dtype
                            )
                            # Get gradient in Hartree/Bohr
                            _, grad = calc.calculate_energy_and_gradient()
                            grad_np = grad.cpu().detach().numpy()
                            
                            # Add penalty gradient
                            _, penalty_grad = self._calculate_distance_constraint_penalty(shifted)
                            if penalty_grad is not None:
                                grad_np += penalty_grad
                            
                            tmp_grad.append(grad_np[b, j])
                        # Finite difference in Angstrom, convert to Bohr
                        val = (tmp_grad[0] - tmp_grad[1]) / (2*numerical_delivative_delta)
                        hessian[3*a+i, 3*b+j] = val
                        hessian[3*b+j, 3*a+i] = val
                count += 1
        return hessian

    def exact_hessian(self, element_number_list, total_charge, positions):
        """
        Calculate exact Hessian using automatic differentiation.
        
        Args:
            element_number_list: Atomic numbers (n_atoms,)
            total_charge: Total molecular charge
            positions: Atomic positions in Angstrom (n_atoms, 3)
            
        Returns:
            Projected Hessian matrix (3*n_atoms, 3*n_atoms) in Hartree/Bohr^2
        """
        # Update distance constraints based on current geometry
        self._update_distance_constraints(positions, element_number_list)
        
        # Create calculator for this geometry
        calc = SQM1Calculator(
            atomic_numbers=element_number_list,
            positions=positions,
            charge=total_charge,
            uhf=0,
            params=self.params,
            device=self.device,
            dtype=self.dtype
        )
        # Calculate Hessian using automatic differentiation
        exact_hess = calc.calculate_hessian(method='analytical')
        exact_hess_np = exact_hess.cpu().detach().numpy()
        
        # Add penalty Hessian
        penalty_hess = self._calculate_distance_constraint_penalty_hessian(positions)
        if penalty_hess is not None:
            exact_hess_np += penalty_hess
        
        # Project out translation and rotation if Calculationtools is available
        if Calculationtools is not None:
            exact_hess_np = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                exact_hess_np, element_number_list.tolist(), positions, display_eigval=False
            )
        self.Model_hess = exact_hess_np

    def single_point(self, file_directory, element_number_list, iter_index, electric_charge_and_multiplicity, method="", geom_num_list=None):
        """
        Calculate single point energy and gradient.
        
        Args:
            file_directory: Directory containing xyz files
            element_number_list: Atomic numbers
            iter_index: Iteration index
            electric_charge_and_multiplicity: [charge, multiplicity]
            geom_num_list: Optional geometry (n_atoms, 3) in Angstrom
            
        Returns:
            Tuple of (energy, gradient, positions, finish_flag)
        """
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        finish_frag = False

        if isinstance(element_number_list[0], str):
            tmp = copy.copy(element_number_list)
            element_number_list = []
            if element_number is not None:
                for elem in tmp:
                    element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)

        try:
            os.mkdir(file_directory)
        except Exception:
            pass

        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = glob.glob(file_directory+"/*_[0-9].xyz")

        total_charge = int(electric_charge_and_multiplicity[0])

        for num, input_file in enumerate(file_list):
            if True:
                if geom_num_list is None and xyz2list is not None:
                    tmp_positions, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                else:
                    tmp_positions = geom_num_list

                positions = np.array(tmp_positions, dtype="float64").reshape(-1, 3)  # Angstrom
                
                # Update distance constraints on every calculation to track dynamic bonding
                self._update_distance_constraints(positions, element_number_list)
                
                # Create calculator for this geometry
                calc = SQM1Calculator(
                    atomic_numbers=element_number_list,
                    positions=positions,
                    charge=total_charge,
                    uhf=0,
                    params=self.params,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # Calculate energy and gradient
                e, g = calc.calculate_energy_and_gradient()
                e = e.cpu().detach().numpy().item()  # Hartree
                g = g.cpu().detach().numpy()  # Hartree/Bohr
                
                # Apply distance constraint penalty
                penalty, penalty_grad = self._calculate_distance_constraint_penalty(positions)
                e += penalty
                if penalty_grad is not None:
                    g += penalty_grad
                positions /= BOHR_TO_ANGSTROM  # Convert back to Angstrom for output
                # Save results
                self.energy = e
                self.gradient = g
                self.coordinate = positions
             
                if self.FC_COUNT == -1 or isinstance(iter_index, str):
                    if self.hessian_flag:
                        self.exact_hessian(element_number_list, total_charge, positions)
                elif iter_index % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_number_list, total_charge, positions)

            #except Exception as error:
            #    print(error)
            #    print("This molecule could not be optimized.")
            #    print("Input file: ", file_list, "\n")
            #    finish_frag = True
            #    return np.array([0]), np.array([0]), positions, finish_frag
       
        return e, g, positions, finish_frag

    def single_point_no_directory(self, positions, element_number_list, electric_charge_and_multiplicity):
        """
        Calculate single point energy and gradient without file I/O.
        
        Args:
            positions: Atomic positions in Angstrom (n_atoms, 3)
            element_number_list: Atomic numbers
            electric_charge_and_multiplicity: [charge, multiplicity]
            
        Returns:
            Tuple of (energy, gradient, finish_flag)
        """
        finish_frag = False
        if isinstance(element_number_list[0], str):
            tmp = copy.copy(element_number_list)
            element_number_list = []
            if element_number is not None:
                for elem in tmp:
                    element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list)
        try:
            positions = np.array(positions, dtype='float64')
            total_charge = int(electric_charge_and_multiplicity[0])
            
            # Update distance constraints on every calculation to track dynamic bonding
            self._update_distance_constraints(positions, element_number_list)
            
            # Create calculator for this geometry
            calc = SQM1Calculator(
                atomic_numbers=element_number_list,
                positions=positions,
                charge=total_charge,
                uhf=0,
                params=self.params,
                device=self.device,
                dtype=self.dtype
            )
            
            # Calculate energy and gradient
            e, g = calc.calculate_energy_and_gradient()
            e = e.cpu().detach().numpy().item()  # Hartree
            g = g.cpu().detach().numpy()  # Hartree/Bohr
            
            # Apply distance constraint penalty
            penalty, penalty_grad = self._calculate_distance_constraint_penalty(positions)
            e += penalty
            if penalty_grad is not None:
                g += penalty_grad
            
            self.energy = e
            self.gradient = g
        except Exception as error:
            print(error)
            print("This molecule could not be optimized.")
            finish_frag = True
            return np.array([0]), np.array([0]), finish_frag
        return e, g, finish_frag


class CalculationEngine:
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        raise NotImplementedError

    def _get_file_list(self, file_directory):
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])

    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        try:
            if getattr(config, 'save_pict', False):
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype='float64') * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                gradient_norm_list = [np.sqrt(np.linalg.norm(g)**2/(len(g)*3)) for g in gradient_list]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")


class SQM1Engine(CalculationEngine):
    """SQM1 calculation engine wrapping SQM1Calculator"""
    def __init__(self, param_file=None, device="cpu", dtype=torch.float64):
        # Parameters are now embedded, param_file is ignored (kept for backward compatibility)
        self.params = SQM1Parameters()
        self.device = device
        self.dtype = dtype

    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []

        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        if xyz2list is None:
            raise ImportError("xyz2list is required for this method")
            
        geometry_list_tmp, element_list, _ = xyz2list(file_list[0], None)
        element_number_list = []
        if element_number is not None:
            for elem in element_list:
                element_number_list.append(element_number(elem))
        element_number_list = np.array(element_number_list, dtype='int')

        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                positions, _, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions = np.array(positions, dtype='float64').reshape(-1, 3)  # Angstrom
                total_charge = int(electric_charge_and_multiplicity[0])
                
                # Create calculator for this geometry
                calc = SQM1Calculator(
                    atomic_numbers=element_number_list,
                    positions=positions,
                    charge=total_charge,
                    uhf=0,
                    params=self.params,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # Calculate energy and gradient
                e, g = calc.calculate_energy_and_gradient()
                e = e.cpu().detach().numpy().item()  # Hartree
                g = g.cpu().detach().numpy()  # Hartree/Bohr
                
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                geometry_num_list.append(positions)
                num_list.append(num)
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)

        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)

        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype='float64').tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype='float64')

        return (np.array(energy_list, dtype='float64'),
                np.array(gradient_list, dtype='float64'),
                np.array(geometry_num_list, dtype='float64'),
                pre_total_velocity)