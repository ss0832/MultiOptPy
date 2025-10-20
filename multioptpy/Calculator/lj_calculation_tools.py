import glob
import os
import numpy as np
from abc import ABC, abstractmethod


from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, number_element
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer

class LennardJonesCore:
    """
    Core calculator for Lennard-Jones potential using UFF parameters.
    Handles both homo- and hetero-atomic clusters using combining rules.
    """
    # UFF parameters with well depth D_i in kcal/mol.
    # Source: Rappe, A. K., et al. J. Am. Chem. Soc. 1992, 114, 10024-10035.
    UFF_PARAMETERS = {
        'He': {'x_i': 2.868, 'D_i': 0.0216},
        'Ne': {'x_i': 3.087, 'D_i': 0.0731},
        'Ar': {'x_i': 3.817, 'D_i': 0.237},
        'Kr': {'x_i': 4.047, 'D_i': 0.357},
        'Xe': {'x_i': 4.363, 'D_i': 0.507},
        'Rn': {'x_i': 4.500, 'D_i': 0.635}, # Extrapolated/common value
    }
    SIGMA_CONV_FACTOR = 1 / (2**(1/6))

    def __init__(self):
        """Initializes a general Lennard-Jones calculator."""
        self.UVL = UnitValueLib()
        # Cache for memoizing parameters of atom types
        self._param_cache = {}

    def get_parameters(self, atom_symbols):
        """
        Retrieves and converts UFF parameters for a list of atom symbols.
        Returns arrays of sigma and epsilon values for each atom in the list.
        """
        sigmas = np.zeros(len(atom_symbols))
        epsilons = np.zeros(len(atom_symbols))

        for i, symbol in enumerate(atom_symbols):
            
            
            if symbol not in self._param_cache:
                if symbol not in self.UFF_PARAMETERS:
                    raise ValueError(f"Atom symbol '{symbol}' is not supported. "
                                     f"Supported: {list(self.UFF_PARAMETERS.keys())}")
                
                params = self.UFF_PARAMETERS[symbol]
                sigma_angstrom = params['x_i'] * self.SIGMA_CONV_FACTOR
                sigma = sigma_angstrom / self.UVL.bohr2angstroms
                # Corrected conversion from kcal/mol to hartree
                epsilon = params['D_i'] / self.UVL.hartree2kcalmol
                self._param_cache[symbol] = (sigma, epsilon)
            
            sigmas[i], epsilons[i] = self._param_cache[symbol]
            
        return sigmas, epsilons

    def calculate_energy_and_gradient(self, coords_bohr, atom_symbols):
        """Calculates the LJ energy and gradient for a list of atoms."""
        num_atoms = coords_bohr.shape[0]
        if num_atoms <= 1:
            return {"energy": 0.0, "gradient": np.zeros_like(coords_bohr)}
     
        base_sigmas, base_epsilons = self.get_parameters(atom_symbols)

        a, b = np.triu_indices(num_atoms, 1)
        diffs = coords_bohr[a] - coords_bohr[b]
        dists_sq = np.sum(diffs**2, axis=1)
        dists = np.sqrt(dists_sq)

        # Apply Lorentz-Berthelot combining rules for each pair
        sigmas_ab = (base_sigmas[a] + base_sigmas[b]) / 2.0
        epsilons_ab = np.sqrt(base_epsilons[a] * base_epsilons[b])

        sigma_over_r = sigmas_ab / dists
        sigma_over_r_6 = sigma_over_r**6
        sigma_over_r_12 = sigma_over_r_6**2

        energy = np.sum(4 * epsilons_ab * (sigma_over_r_12 - sigma_over_r_6))

        grad_mag_over_r = -24 * epsilons_ab / dists_sq * (2 * sigma_over_r_12 - sigma_over_r_6)
        grad_pairs = grad_mag_over_r[:, np.newaxis] * diffs
        gradient = np.zeros_like(coords_bohr)
        np.add.at(gradient, a, grad_pairs)
        np.add.at(gradient, b, -grad_pairs)

        return {"energy": energy, "gradient": gradient}

    def calculate_hessian(self, coords_bohr, atom_symbols):
        """Calculates the LJ Hessian for a list of atoms using a vectorized approach."""
        num_atoms = coords_bohr.shape[0]
        hessian = np.zeros((num_atoms * 3, num_atoms * 3))
        if num_atoms <= 1:
            return {"hessian": hessian}

        base_sigmas, base_epsilons = self.get_parameters(atom_symbols)
        
        a, b = np.triu_indices(num_atoms, 1)
        diffs = coords_bohr[a] - coords_bohr[b]
        dists_sq = np.sum(diffs**2, axis=1)
        dists = np.sqrt(dists_sq)

        sigmas_ab = (base_sigmas[a] + base_sigmas[b]) / 2.0
        epsilons_ab = np.sqrt(base_epsilons[a] * base_epsilons[b])

        sigma_over_r = sigmas_ab / dists
        sigma_over_r_6 = sigma_over_r**6
        sigma_over_r_12 = sigma_over_r_6**2

        grad_mag_over_r = -24 * epsilons_ab / dists_sq * (2 * sigma_over_r_12 - sigma_over_r_6)
        d2V_dr2 = 24 * epsilons_ab / dists_sq * (26 * sigma_over_r_12 - 7 * sigma_over_r_6)
        dV_dr_over_r = -grad_mag_over_r

        term1_mag = (d2V_dr2 - dV_dr_over_r) / dists_sq
        term2_mag = dV_dr_over_r
        term1 = np.einsum('p,pi,pj->pij', term1_mag, diffs, diffs)
        term2 = np.identity(3)[np.newaxis, :, :] * term2_mag[:, np.newaxis, np.newaxis]
        sub_hessians = term1 + term2

        # **Vectorized Hessian Assembly**
        # Create meshgrid of indices for all 3x3 sub-blocks
        p, q = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
        
        # Indices for off-diagonal blocks (a, b) and (b, a)
        row_indices_ab = (a[:, None, None] * 3 + p).flatten()
        col_indices_ab = (b[:, None, None] * 3 + q).flatten()
        
        # Indices for diagonal blocks (a, a) and (b, b)
        row_indices_aa = (a[:, None, None] * 3 + p).flatten()
        col_indices_aa = (a[:, None, None] * 3 + q).flatten()
        row_indices_bb = (b[:, None, None] * 3 + p).flatten()
        col_indices_bb = (b[:, None, None] * 3 + q).flatten()

        # Flatten the sub-hessian blocks to match the indices
        flat_sub_hessians = sub_hessians.flatten()
        
        # Atomically add/subtract the blocks into the Hessian matrix
        np.subtract.at(hessian, (row_indices_ab, col_indices_ab), flat_sub_hessians)
        np.subtract.at(hessian, (col_indices_ab, row_indices_ab), flat_sub_hessians)
        np.add.at(hessian, (row_indices_aa, col_indices_aa), flat_sub_hessians)
        np.add.at(hessian, (row_indices_bb, col_indices_bb), flat_sub_hessians)
            
        return {"hessian": hessian}

class Calculation:
    """
    High-level wrapper for Lennard-Jones calculations.
    """
    def __init__(self, **kwarg):
        UVL = UnitValueLib()
        self.bohr2angstroms = UVL.bohr2angstroms
        self.atom_symbol = kwarg.get("atom_symbol", None) # Can be None initially
        self.FC_COUNT = kwarg.get("FC_COUNT", -1)
        self.Model_hess = kwarg.get("Model_hess")
        self.hessian_flag = kwarg.get("hessian_flag", False)
        self.calculator = LennardJonesCore()
        self.energy = None
        self.gradient = None
        self.coordinate = None

    def exact_hessian(self, element_list, positions_bohr):
        """Calculates and projects the Hessian."""
        results = self.calculator.calculate_hessian(positions_bohr, self.atom_symbol)
        exact_hess = results['hessian']

        self.Model_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            exact_hess, element_list, positions_bohr, display_eigval=False
        )

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method="", geom_num_list=None):
        """
        Executes a Lennard-Jones single point calculation, reading from a file
        or using a provided geometry.
        """
        finish_frag = False
        e, g, positions_bohr = None, None, None

        try:
            os.makedirs(file_directory, exist_ok=True)
        except (OSError, TypeError): # TypeError if file_directory is None
            pass

        if file_directory is None:
            file_list = ["dummy"] # To run the loop once for geom_num_list
        else:
            file_list = sorted(glob.glob(os.path.join(file_directory, "*_[0-9].xyz")))
            if not file_list and geom_num_list is None:
                 raise FileNotFoundError(f"No XYZ files found in {file_directory}")

        for num, input_file in enumerate(file_list):
            try:
                positions_angstrom = None
                if geom_num_list is None:
                    positions_angstrom, read_elements, _ = xyz2list(input_file, electric_charge_and_multiplicity)
                  
                    element_list = read_elements
                else:
                    positions_angstrom = geom_num_list
           
                if self.atom_symbol is None:
                    if element_list is None or len(element_list) == 0:
                        raise ValueError("Element list is empty. Cannot determine atom symbol.")
                    first_element = element_list
                    if type(element_list[0]) is not str:
                        first_element = []
                        for i in range(len(element_list)):
                            first_element.append(number_element(element_list[i]))
                   
                    self.atom_symbol = first_element
                    print(f"Atom symbol set to '{self.atom_symbol}' based on the first structure.")
            
                positions_bohr = np.array(positions_angstrom, dtype="float64") / self.bohr2angstroms
             
                results = self.calculator.calculate_energy_and_gradient(positions_bohr, self.atom_symbol)
                e = results['energy']
                g = results['gradient']

                if self.FC_COUNT == -1 or isinstance(iter, str):
                    if self.hessian_flag:
                        self.exact_hessian(element_list, positions_bohr)
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_list, positions_bohr)
                
                break

            except Exception as error:
                print(f"Error during Lennard-Jones calculation for {input_file}: {error}")
                finish_frag = True
                return np.array([0]), np.array([0]), np.array([0]), finish_frag

        self.energy = e
        self.gradient = g
        self.coordinate = positions_bohr
        return e, g, positions_bohr, finish_frag

class CalculationEngine(ABC):
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        pass
    def _get_file_list(self, file_directory):
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) for i in range(1, 7)], [])
    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        try:
            if hasattr(config, 'save_pict') and config.save_pict:
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                gradient_norm_list = [np.sqrt(np.linalg.norm(g)**2 / (len(g) * 3)) for g in gradient_list if g.size > 0]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")

class LJEngine(CalculationEngine):
    def __init__(self, atom_symbol=None):
        super().__init__()
        self.atom_symbol = atom_symbol
        self.calculator = LennardJonesCore()
        self.bohr2angstroms = UnitValueLib().bohr2angstroms

    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list, energy_list, geometry_num_list, num_list = [], [], [], []
        delete_pre_total_velocity = []
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)

        if not file_list:
            print(f"No XYZ files found in directory: {file_directory}")
            return np.array([]), np.array([]), np.array([]), pre_total_velocity

        for num, input_file in enumerate(file_list):
            try:
        
                print(f"Processing file: {input_file}")
                positions_angstrom, element_list, _ = xyz2list(input_file, None)
               
                if self.atom_symbol is None:
                    if element_list is None or len(element_list) == 0:
                         raise ValueError("Element list from file is empty.")
                    first_element = element_list
                    if type(element_list[0]) is not str:
                        first_element = []
                        for i in range(len(element_list)):
                            first_element.append(number_element(element_list[i]))

                    self.atom_symbol = first_element
                    print(f"Engine atom symbol set to '{self.atom_symbol}' based on the first file.")
         
                positions_angstrom = np.array(positions_angstrom, dtype='float64').reshape(-1, 3)
                positions_bohr = positions_angstrom / self.bohr2angstroms
                
                results = self.calculator.calculate_energy_and_gradient(positions_bohr, self.atom_symbol)
                
                energy_list.append(results['energy'])
                gradient_list.append(results['gradient'])
                geometry_num_list.append(positions_bohr)
                num_list.append(num)
            except Exception as error:
                print(f"Error processing {input_file}: {error}")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)

        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)
        if optimize_num != 0 and len(pre_total_velocity) > 0 and delete_pre_total_velocity:
            pre_total_velocity = np.delete(np.array(pre_total_velocity), delete_pre_total_velocity, axis=0)
        return (np.array(energy_list, dtype='float64'),
                np.array(gradient_list, dtype='float64'),
                np.array(geometry_num_list, dtype='float64'),
                pre_total_velocity)