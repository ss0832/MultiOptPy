import glob
import os
from collections import defaultdict
from math import log, sqrt
import numpy as np
from abc import ABC, abstractmethod

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib, element_number, number_element
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer


class EMT:
    """
    A standalone Python implementation of the Effective Medium Theory (EMT)
    potential.
     
    This implementation is suitable for calculating energy, forces, and Hessians
    on atomic clusters. The Hessian is calculated via finite differences.
    
    Notes on units:
    - Input coordinates for calculation are expected in Angstroms.
    - Internal calculations are performed using eV and Angstroms.
    - Output energy is in Hartree.
    - Output forces are in Hartree/Bohr.
    - Output Hessian is in Hartree/Bohr^2.
    """
    
    # Physical constants for internal use and unit conversion
    BOHR = UnitValueLib().bohr2angstroms  # Angstroms
    EV_TO_HARTREE = 1 / UnitValueLib().hartree2eV
    EV_PER_ANG_TO_HARTREE_PER_BOHR = EV_TO_HARTREE / BOHR
    HARTREE_PER_BOHR_SQ_TO_EV_PER_ANG_SQ = 1 / (EV_TO_HARTREE / BOHR**2)

    # Atomic data
    CHEMICAL_SYMBOLS = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 13: 'Al', 29: 'Cu', 46: 'Pd',
        47: 'Ag', 78: 'Pt', 79: 'Au', 28: 'Ni'
    }
    ATOMIC_NUMBERS = {v: k for k, v in CHEMICAL_SYMBOLS.items()}

    PARAMETERS = {
        #      E0     s0    V0     eta2    kappa   lambda  n0
        #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
        'Al': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
        'Cu': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
        'Ag': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
        'Au': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
        'Ni': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
        'Pd': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
        'Pt': (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
        'H':  (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
        'C':  (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
        'N':  (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
        'O':  (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850)
    }
    BETA = 1.809  # (16 * pi / 3)**(1.0 / 3) / 2**0.5

    def __init__(self, symbols, asap_cutoff=False):
        self.symbols = symbols
        self.numbers = np.array([self.ATOMIC_NUMBERS[s] for s in symbols])
        self.asap_cutoff = asap_cutoff
        
        self.positions = None
        self.energy_ev = 0.0
        self.forces_ev_per_ang = None
        
        self._initialize_parameters()

    def _initialize_parameters(self):
        self.rc, self.rc_list, self.acut = self._calc_cutoff()

        unique_numbers, self.ia2iz = np.unique(self.numbers, return_inverse=True)
        self.par = defaultdict(lambda: np.empty(len(unique_numbers)))
        for i, Z in enumerate(unique_numbers):
            sym = self.CHEMICAL_SYMBOLS[Z]
            if sym not in self.PARAMETERS:
                raise NotImplementedError(f'No EMT-potential for {sym}')
            
            p = self.PARAMETERS[sym]
            s0 = p[1] * self.BOHR
            eta2 = p[3] / self.BOHR
            kappa = p[4] / self.BOHR
            gamma1, gamma2 = self._calc_gammas(s0, eta2, kappa)
            
            self.par['Z'][i] = Z
            self.par['E0'][i] = p[0]
            self.par['s0'][i] = s0
            self.par['V0'][i] = p[2]
            self.par['eta2'][i] = eta2
            self.par['kappa'][i] = kappa
            self.par['lambda'][i] = p[5] / self.BOHR
            self.par['n0'][i] = p[6] / self.BOHR**3
            self.par['inv12gamma1'][i] = 1.0 / (12.0 * gamma1)
            self.par['neghalfv0overgamma2'][i] = -0.5 * p[2] / gamma2

        self.chi = self.par['n0'][None, :] / self.par['n0'][:, None]

    def _calc_cutoff(self):
        if self.asap_cutoff:
            relevant_pars = {
                symb: p for symb, p in self.PARAMETERS.items()
                if self.ATOMIC_NUMBERS[symb] in self.numbers
            }
        else:
            relevant_pars = self.PARAMETERS
            
        max_s0_bohr = max(par[1] for par in relevant_pars.values())
        maxseq = max_s0_bohr * self.BOHR
        r1nn = self.BETA * maxseq
        rc = r1nn * 0.5 * (sqrt(3.0) + 2.0)
        r4nn = r1nn * 2.0
        eps = 1e-4
        acut = log(1.0 / eps - 1.0) / (r4nn - rc)
        rc_list = rc * 1.045 if self.asap_cutoff else rc + 0.5
        return rc, rc_list, acut

    def _calc_gammas(self, s0, eta2, kappa):
        n = np.array([12, 6, 24])
        r = self.BETA * s0 * np.sqrt([1.0, 2.0, 3.0])
        w = 1.0 / (1.0 + np.exp(self.acut * (r - self.rc)))
        x = n * w / 12.0
        gamma1 = np.dot(x, np.exp(-eta2 * (r - self.BETA * s0)))
        gamma2 = np.dot(x, np.exp(-kappa / self.BETA * (r - self.BETA * s0)))
        return gamma1, gamma2

    def _get_energy_and_forces_internal(self, positions_angstrom):
        """Calculates energy and forces in internal units (eV, eV/A)."""
        self.positions = positions_angstrom
        natoms = len(self.positions)
        
        self.energies = np.zeros(natoms)
        self.forces_ev_per_ang = np.zeros((natoms, 3))
        self.deds = np.zeros(natoms)
        
        ps = {}
        for a1 in range(natoms):
            diffs = self.positions - self.positions[a1]
            dists = np.linalg.norm(diffs, axis=1)
            neighbor_indices = np.where((dists > 1e-9) & (dists < self.rc_list))[0]
            if len(neighbor_indices) == 0: continue

            a2, d, r = neighbor_indices, diffs[neighbor_indices], dists[neighbor_indices]
            w, dwdroverw = self._calc_theta(r)
            dsigma1s, dsigma1o = self._calc_dsigma1(a1, a2, r, w)
            dsigma2s, dsigma2o = self._calc_dsigma2(a1, a2, r, w)
            ps[a1] = {
                'a2': a2, 'd': d, 'r': r, 'invr': 1.0 / r, 'w': w,
                'dwdroverw': dwdroverw, 'dsigma1s': dsigma1s, 'dsigma1o': dsigma1o,
                'dsigma2s': dsigma2s, 'dsigma2o': dsigma2o,
            }

        for a1, p in ps.items(): self._calc_e_c_a2(a1, p['dsigma1s'])
        for a1, p in ps.items(): self._calc_pairwise_forces(a1, **p)

        self.energies -= self.par['E0'][self.ia2iz]
        self.energy_ev = np.sum(self.energies)
        return self.energy_ev, self.forces_ev_per_ang

    def calculate_energy_and_forces(self, positions_angstrom):
        """Calculates energy and forces, returning them in atomic units."""
        energy_ev, forces_ev_ang = self._get_energy_and_forces_internal(positions_angstrom)
        energy_hartree = energy_ev * self.EV_TO_HARTREE
        forces_hartree_per_bohr = forces_ev_ang * self.EV_PER_ANG_TO_HARTREE_PER_BOHR
        return energy_hartree, forces_hartree_per_bohr

    def calculate_hessian(self, positions_angstrom, fd_step=1e-5):
        """
        Calculates the Hessian matrix by vectorized finite differences, returning
        it in atomic units (Hartree/Bohr^2).
        A smaller finite difference step is used for better numerical accuracy.
        """
        num_atoms = len(positions_angstrom)
        hessian_ev_per_ang_sq = np.zeros((num_atoms * 3, num_atoms * 3))
        
        for i in range(num_atoms):
            for j in range(3):
                # Central difference method for higher accuracy
                pos_plus = positions_angstrom.copy()
                pos_plus[i, j] += fd_step
                _, forces_plus = self._get_energy_and_forces_internal(pos_plus)

                pos_minus = positions_angstrom.copy()
                pos_minus[i, j] -= fd_step
                _, forces_minus = self._get_energy_and_forces_internal(pos_minus)

                # Gradient is negative force: g = -F
                # Hessian H_ij = d(g_i)/d(r_j) = d(-F_i)/d(r_j)
                # Using central differences: H_ij approx - (F_i(r+h) - F_i(r-h)) / (2h)
                # The column of the Hessian corresponding to displacement 'k' is d(g)/dr_k = -dF/dr_k
                hessian_col = -(forces_plus.flatten() - forces_minus.flatten()) / (2 * fd_step)
                hessian_ev_per_ang_sq[:, i * 3 + j] = hessian_col
        
        # Symmetrize the Hessian to reduce numerical noise
        hessian_ev_per_ang_sq = 0.5 * (hessian_ev_per_ang_sq + hessian_ev_per_ang_sq.T)
        
        # Convert to atomic units
        hessian_hartree_per_bohr_sq = hessian_ev_per_ang_sq / self.HARTREE_PER_BOHR_SQ_TO_EV_PER_ANG_SQ
        return hessian_hartree_per_bohr_sq

    def _calc_theta(self, r):
        w = 1.0 / (1.0 + np.exp(self.acut * (r - self.rc)))
        dwdroverw = self.acut * (w - 1.0)
        return w, dwdroverw

    def _calc_dsigma1(self, a1, a2, r, w):
        s0s, s0o = self.par['s0'][self.ia2iz[a1]], self.par['s0'][self.ia2iz[a2]]
        eta2s, eta2o = self.par['eta2'][self.ia2iz[a1]], self.par['eta2'][self.ia2iz[a2]]
        chi = self.chi[self.ia2iz[a1], self.ia2iz[a2]]
        dsigma1s = np.exp(-eta2o * (r - self.BETA * s0o)) * chi * w
        dsigma1o = np.exp(-eta2s * (r - self.BETA * s0s)) / chi * w
        return dsigma1s, dsigma1o

    def _calc_dsigma2(self, a1, a2, r, w):
        s0s, s0o = self.par['s0'][self.ia2iz[a1]], self.par['s0'][self.ia2iz[a2]]
        kappas, kappao = self.par['kappa'][self.ia2iz[a1]], self.par['kappa'][self.ia2iz[a2]]
        chi = self.chi[self.ia2iz[a1], self.ia2iz[a2]]
        dsigma2s = np.exp(-kappao * (r / self.BETA - s0o)) * chi * w
        dsigma2o = np.exp(-kappas * (r / self.BETA - s0s)) / chi * w
        return dsigma2s, dsigma2o

    def _calc_e_c_a2(self, a1, dsigma1s):
        sigma1 = np.sum(dsigma1s)
        if sigma1 < 1e-20: return

        iz1 = self.ia2iz[a1]
        e0s, v0s, eta2s = self.par['E0'][iz1], self.par['V0'][iz1], self.par['eta2'][iz1]
        lmds, kappas, inv12gamma1s = self.par['lambda'][iz1], self.par['kappa'][iz1], self.par['inv12gamma1'][iz1]

        ds = -np.log(sigma1 * inv12gamma1s) / (self.BETA * eta2s)
        lmdsds = lmds * ds
        expneglmdds = np.exp(-lmdsds)
        self.energies[a1] += e0s * (1.0 + lmdsds) * expneglmdds
        self.deds[a1] += -e0s * lmds * lmdsds * expneglmdds
        
        sixv0expnegkppds = 6.0 * v0s * np.exp(-kappas * ds)
        self.energies[a1] += sixv0expnegkppds
        self.deds[a1] += -kappas * sixv0expnegkppds
        self.deds[a1] /= -self.BETA * eta2s * sigma1

    def _calc_pairwise_forces(self, a1, a2, d, invr, dwdroverw, dsigma1s, dsigma1o, dsigma2s, dsigma2o, **kwargs):
        iz1, iz2 = self.ia2iz[a1], self.ia2iz[a2]
        eta2s, eta2o = self.par['eta2'][iz1], self.par['eta2'][iz2]
        ddsigma1sdr = dsigma1s * (dwdroverw - eta2o)
        ddsigma1odr = dsigma1o * (dwdroverw - eta2s)
        dedrs = self.deds[a1] * ddsigma1sdr
        dedro = self.deds[a2] * ddsigma1odr
        f_cohesive = (dedrs + dedro) * invr

        neghalfv0overgamma2s, neghalfv0overgamma2o = self.par['neghalfv0overgamma2'][iz1], self.par['neghalfv0overgamma2'][iz2]
        kappas, kappao = self.par['kappa'][iz1], self.par['kappa'][iz2]
        es, eo = neghalfv0overgamma2s * dsigma2s, neghalfv0overgamma2o * dsigma2o
        self.energies[a1] += 0.5 * np.sum(es)
        self.energies[a2] += 0.5 * np.sum(eo)
        dedrs_pair = es * (dwdroverw - kappao / self.BETA)
        dedro_pair = eo * (dwdroverw - kappas / self.BETA)
        f_pair = (dedrs_pair + dedro_pair) * invr
        
        f_total_mag = f_cohesive + f_pair
        f_pairs_vec = f_total_mag[:, None] * d
        
        self.forces_ev_per_ang[a1] += np.sum(f_pairs_vec, axis=0)
        np.add.at(self.forces_ev_per_ang, a2, -f_pairs_vec)

class EMTCore:
    """
    Core calculator for EMT potential, using the standalone implementation.
    This class acts as a wrapper and cache for the EMT calculator.
    """
    def __init__(self):
        self.UVL = UnitValueLib()
        self._calculator_cache = {}

    def _get_calculator(self, atom_symbols):
        symbols_tuple = tuple(sorted(set(atom_symbols)))
        if symbols_tuple not in self._calculator_cache:
            self._calculator_cache[symbols_tuple] = EMT(symbols_tuple)
        
        calc = self._calculator_cache[symbols_tuple]
        calc.symbols = atom_symbols
        calc.numbers = np.array([calc.ATOMIC_NUMBERS[s] for s in atom_symbols])
        calc._initialize_parameters()
        return calc

    def calculate_energy_and_gradient(self, coords_bohr, atom_symbols):
        """Calculates EMT energy (Hartree) and gradient (Hartree/Bohr)."""
        if coords_bohr.shape[0] == 0:
            return {"energy": 0.0, "gradient": np.zeros_like(coords_bohr)}

        coords_angstrom = coords_bohr * self.UVL.bohr2angstroms
        calculator = self._get_calculator(atom_symbols)
        
        try:
            energy_hartree, forces_hartree_per_bohr = calculator.calculate_energy_and_forces(coords_angstrom)
            gradient_hartree_bohr = -forces_hartree_per_bohr
            return {"energy": energy_hartree, "gradient": gradient_hartree_bohr}
        except NotImplementedError as e:
            print(f"Error during EMT calculation: {e}")
            return {"energy": 0.0, "gradient": np.zeros_like(coords_bohr)}

    def calculate_hessian(self, coords_bohr, atom_symbols):
        
        """Calculates EMT Hessian (Hartree/Bohr^2) via finite differences."""
        print("Warning: EMT Hessian calculation is not tested well. Use with caution.")
        if coords_bohr.shape[0] == 0:
            return {"hessian": np.zeros((0,0))}
            
        # FIX: Convert coordinates from Bohr to Angstrom for Hessian calculation
        coords_angstrom = coords_bohr * self.UVL.bohr2angstroms
        calculator = self._get_calculator(atom_symbols)
        
        hessian_hartree_per_bohr_sq = calculator.calculate_hessian(coords_angstrom)
        return {"hessian": hessian_hartree_per_bohr_sq}


class Calculation:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()
        self.bohr2angstroms = UVL.bohr2angstroms
        self.atom_symbol = kwarg.get("atom_symbol", None)
        self.FC_COUNT = kwarg.get("FC_COUNT", -1)
        self.Model_hess = kwarg.get("Model_hess")
        self.hessian_flag = kwarg.get("hessian_flag", False)
        self.calculator = EMTCore()
        self.energy = None
        self.gradient = None
        self.coordinate = None

    def exact_hessian(self, element_list, positions_bohr):
        """Calculates and projects the Hessian."""
        results = self.calculator.calculate_hessian(positions_bohr, element_list)
        exact_hess = results['hessian']
        element_number_list = [element_number(elem) for elem in element_list]
        self.Model_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            exact_hess, element_number_list, positions_bohr, display_eigval=False
        )

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, geom_num_list=None):
        """Executes an EMT single point calculation."""
        finish_frag = False
        e, g, positions_bohr = None, None, None

        try:
            os.makedirs(file_directory, exist_ok=True)
        except (OSError, TypeError):
            pass
       
        if element_list is not None and len(element_list) > 0 and (type(element_list[0]) is np.int64 or type(element_list[0]) is int or type(element_list[0]) is np.int32):
            element_list = [number_element(elem) for elem in element_list]
        
        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = sorted(glob.glob(os.path.join(file_directory, "*_[0-9].xyz")))
            if not file_list and geom_num_list is None:
                 raise FileNotFoundError(f"No XYZ files found in {file_directory}")

        for num, input_file in enumerate(file_list):
            try:
                if geom_num_list is None:
                    positions_angstrom, read_elements, _ = xyz2list(input_file, electric_charge_and_multiplicity)
                   
                    if element_list is None or len(element_list) == 0:
                        element_list = read_elements
                else:
                    positions_angstrom = geom_num_list

                if self.atom_symbol is None and (element_list is not None and len(element_list) > 0):
                    unique_elements = set(element_list)
                    if len(unique_elements) == 1:
                        self.atom_symbol = unique_elements.pop()
                        print(f"System type detected as homo-atomic. Atom symbol set to '{self.atom_symbol}'.")
                    else:
                        print(f"System type detected as hetero-atomic: {unique_elements}")
                
                positions_bohr = np.array(positions_angstrom, dtype="float64") / self.bohr2angstroms
                results = self.calculator.calculate_energy_and_gradient(positions_bohr, element_list)
                e, g = results['energy'], results['gradient']
                
                if self.FC_COUNT == -1 or isinstance(iter, str):
                    if self.hessian_flag: self.exact_hessian(element_list, positions_bohr)
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.exact_hessian(element_list, positions_bohr)
                break
            except Exception as error:
                print(f"Error during EMT calculation for {input_file}: {error}")
                finish_frag = True
                return np.array([0]), np.array([0]), np.array([0]), finish_frag

        self.energy, self.gradient, self.coordinate = e, g, positions_bohr
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
                self.UVL = UnitValueLib()
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * self.UVL.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                gradient_norm_list = [np.linalg.norm(g) for g in gradient_list if hasattr(g, 'size') and g.size > 0]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")

class EMTEngine(CalculationEngine):
    def __init__(self, **kwargs):
        super().__init__()
        self.calculator = EMTCore()
        self.UVL = UnitValueLib()
        self.bohr2angstroms = self.UVL.bohr2angstroms

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
                
                if element_list is None or len(element_list) == 0:
                    raise ValueError("Element list from file is empty.")

                positions_bohr = np.array(positions_angstrom, dtype='float64').reshape(-1, 3) / self.bohr2angstroms
                results = self.calculator.calculate_energy_and_gradient(positions_bohr, element_list)
                
                energy_list.append(results['energy'])
                gradient_list.append(results['gradient'])
                geometry_num_list.append(positions_angstrom)
                num_list.append(num)
            except Exception as error:
                print(f"Error processing {input_file}: {error}")
                if optimize_num != 0: delete_pre_total_velocity.append(num)

        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)
        if optimize_num != 0 and hasattr(pre_total_velocity, '__len__') and len(pre_total_velocity) > 0 and delete_pre_total_velocity:
            pre_total_velocity = np.delete(np.array(pre_total_velocity), delete_pre_total_velocity, axis=0)
        return (np.array(energy_list, dtype='float64'),
                np.array(gradient_list, dtype='float64'),
                np.array(geometry_num_list, dtype='float64'),
                pre_total_velocity)