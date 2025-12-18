import os
import sys
import glob
import copy
import itertools
import inspect
import datetime
import numpy as np

from multioptpy.optimizer import CalculateMoveVector
from multioptpy.Visualization.visualization import Graph
from multioptpy.fileio import FileIO
from multioptpy.Parameters.parameter import UnitValueLib, element_number
from multioptpy.interface import force_data_parser
from multioptpy.ModelHessian.approx_hessian import ApproxHessian
from multioptpy.PESAnalyzer.cmds_analysis import CMDSPathAnalysis
from multioptpy.PESAnalyzer.pca_analysis import PCAPathAnalysis
from multioptpy.PESAnalyzer.koopman_analysis import KoopmanAnalyzer
from multioptpy.Potential.potential import BiasPotentialCalculation
from multioptpy.Utils.calc_tools import CalculationStructInfo, Calculationtools
from multioptpy.Constraint.constraint_condition import ProjectOutConstrain
from multioptpy.irc import IRC
from multioptpy.Utils.bond_connectivity import judge_shape_condition
from multioptpy.Utils.oniom import (
    separate_high_layer_and_low_layer,
    specify_link_atom_pairs,
    link_number_high_layer_and_low_layer,
)
from multioptpy.Utils.symmetry_analyzer import analyze_symmetry
from multioptpy.Thermo.normal_mode_analyzer import MolecularVibrations
from multioptpy.ModelFunction.opt_meci import OptMECI
from multioptpy.ModelFunction.opt_mesx import OptMESX
from multioptpy.ModelFunction.opt_mesx_2 import OptMESX2
from multioptpy.ModelFunction.seam_model_function import SeamModelFunction
from multioptpy.ModelFunction.conical_model_function import ConicalModelFunction
from multioptpy.ModelFunction.avoiding_model_function import AvoidingModelFunction
from multioptpy.ModelFunction.binary_image_ts_search_model_function import BITSSModelFunction
# =====================================================================================
# 1. Configuration (Immutable Settings)
# =====================================================================================
class OptimizationConfig:
    """
    Immutable settings derived from CLI args.
    """

    def __init__(self, args):
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)

        # Physical constants
        self.hartree2kcalmol = UVL.hartree2kcalmol
        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2kjmol = UVL.hartree2kjmol

        # Base args
        self.args = args
        self._set_convergence_criteria(args)

        # Core parameters
        self.microiter_num = 100
        self.FC_COUNT = args.calc_exact_hess
        self.mFC_COUNT = args.calc_model_hess
        self.temperature = float(args.temperature) if args.temperature else 0.0
        self.CMDS = args.cmds
        self.PCA = args.pca
        self.DELTA = "x" if args.DELTA == "x" else float(args.DELTA)
        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.NSTEP = args.NSTEP
        self.BASIS_SET = args.basisset
        self.FUNCTIONAL = args.functional
        self.excited_state = args.excited_state

        # Sub-basis and ECP
        self._check_sub_basisset(args)

        # Advanced settings
        self.DC_check_dist = float(args.dissociate_check)
        self.unrestrict = args.unrestrict
        self.irc = args.intrinsic_reaction_coordinates
        self.othersoft = args.othersoft
        self.cpcm_solv_model = args.cpcm_solv_model
        self.alpb_solv_model = args.alpb_solv_model
        self.shape_conditions = args.shape_conditions
        self.oniom = args.oniom_flag
        self.use_model_hessian = args.use_model_hessian
        self.sqm1 = args.sqm1
        self.sqm2 = args.sqm2
        self.freq_analysis = args.frequency_analysis
        self.thermo_temperature = float(args.temperature)
        self.thermo_pressure = float(args.pressure)
        self.dft_grid = int(args.dft_grid)
        self.max_trust_radius = args.max_trust_radius
        self.min_trust_radius = args.min_trust_radius
        self.software_path_file = args.software_path_file
        self.koopman_analysis = args.koopman
        self.detect_negative_eigenvalues = args.detect_negative_eigenvalues
        self.excited_state = args.excited_state
        self.spin_multiplicity = args.spin_multiplicity
        self.electronic_charge = args.electronic_charge
        self.model_function = args.model_function
        
        

    def _set_convergence_criteria(self, args):
        if args.tight_convergence_criteria and not args.loose_convergence_criteria:
            self.MAX_FORCE_THRESHOLD = 0.000015
            self.RMS_FORCE_THRESHOLD = 0.000010
            self.MAX_DISPLACEMENT_THRESHOLD = 0.000060
            self.RMS_DISPLACEMENT_THRESHOLD = 0.000040
            
            if len(args.projection_constrain) > 0:
                self.MAX_DISPLACEMENT_THRESHOLD *= 4
                self.RMS_DISPLACEMENT_THRESHOLD *= 4
            
            
            
        elif not args.tight_convergence_criteria and args.loose_convergence_criteria:
            self.MAX_FORCE_THRESHOLD = 0.0030
            self.RMS_FORCE_THRESHOLD = 0.0020
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0100
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0070

             
        else:
            self.MAX_FORCE_THRESHOLD = 0.0003
            self.RMS_FORCE_THRESHOLD = 0.0002
            self.MAX_DISPLACEMENT_THRESHOLD = 0.0015
            self.RMS_DISPLACEMENT_THRESHOLD = 0.0010
            if len(args.projection_constrain) > 0: 
                self.MAX_DISPLACEMENT_THRESHOLD *= 4
                self.RMS_DISPLACEMENT_THRESHOLD *= 4   
        
    def _check_sub_basisset(self, args):
        if len(args.sub_basisset) % 2 != 0:
            print("invalid input (-sub_bs)")
            sys.exit(0)

        self.electric_charge_and_multiplicity = [
            int(args.electronic_charge),
            int(args.spin_multiplicity),
        ]
        self.electronic_charge = args.electronic_charge
        self.spin_multiplicity = args.spin_multiplicity

        if args.pyscf:
            self.SUB_BASIS_SET = {}
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET["default"] = str(self.BASIS_SET)
                for j in range(int(len(args.sub_basisset) / 2)):
                    self.SUB_BASIS_SET[args.sub_basisset[2 * j]] = args.sub_basisset[
                        2 * j + 1
                    ]
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)
            else:
                self.SUB_BASIS_SET = {"default": self.BASIS_SET}
        else:
            self.SUB_BASIS_SET = ""
            if len(args.sub_basisset) > 0:
                self.SUB_BASIS_SET += "\nassign " + str(self.BASIS_SET) + "\n"
                for j in range(int(len(args.sub_basisset) / 2)):
                    self.SUB_BASIS_SET += (
                        "assign "
                        + args.sub_basisset[2 * j]
                        + " "
                        + args.sub_basisset[2 * j + 1]
                        + "\n"
                    )
                print("Basis Sets defined by User are detected.")
                print(self.SUB_BASIS_SET)

        if len(args.effective_core_potential) % 2 != 0:
            print("invaild input (-ecp)")
            sys.exit(0)

        if args.pyscf:
            self.ECP = {}
            if len(args.effective_core_potential) > 0:
                for j in range(int(len(args.effective_core_potential) / 2)):
                    self.ECP[
                        args.effective_core_potential[2 * j]
                    ] = args.effective_core_potential[2 * j + 1]
        else:
            self.ECP = ""

# =====================================================================================
# 2. State (Mutable Data)
# =====================================================================================
class OptimizationState:
    """
    Dynamic state of the optimization.
    """

    def __init__(self, element_list):
        natom = len(element_list)
        self.iter = 0

        # Energies and gradients
        self.energies = {}
        self.gradients = {}
        self.raw_energy = 0.0
        self.raw_gradient = np.zeros((natom, 3), dtype="float64")
        self.bias_energy = 0.0
        self.bias_gradient = np.zeros((natom, 3), dtype="float64")
        self.effective_energy = 0.0
        self.effective_gradient = np.zeros((natom, 3), dtype="float64")

        # Geometry
        self.geometry = None  # Bohr
        self.initial_geometry = None  # Bohr

        # Previous step
        self.pre_geometry = np.zeros((natom, 3), dtype="float64")
        self.pre_effective_gradient = np.zeros((natom, 3), dtype="float64")
        self.pre_bias_gradient = np.zeros((natom, 3), dtype="float64")
        self.pre_effective_energy = 0.0
        self.pre_bias_energy = 0.0
        self.pre_raw_gradient = np.zeros((natom, 3), dtype="float64")
        self.pre_move_vector = np.zeros((natom, 3), dtype="float64")
        self.pre_raw_energy = 0.0

        # Hessian
        self.Model_hess = np.eye(natom * 3)

        # Logs
        self.history = {
            "iter": [],
            "energies": {},
            "grad_rms": [],
            "bias_grad_rms": [],
        }

        # For plotting and outputs
        self.ENERGY_LIST_FOR_PLOTTING = []
        self.BIAS_ENERGY_LIST_FOR_PLOTTING = []
        self.NUM_LIST = []
        self.grad_list = []
        self.bias_grad_list = []
        self.cos_list = []

        # Final results
        self.final_file_directory = None
        self.final_geometry = None
        self.final_energy = None
        self.final_bias_energy = None
        self.bias_pot_params_grad_list = None
        self.bias_pot_params_grad_name_list = None
        self.symmetry = None

        # Flags
        self.exit_flag = False
        self.converged_flag = False
        self.dissociation_flag = False
        self.optimized_flag = False
        self.DC_check_flag = False

# =====================================================================================
# 3. Potential Handlers (Strategy)
# =====================================================================================
class BasePotentialHandler:
    def __init__(self, config, file_io, base_dir, force_data):
        self.config = config
        self.file_io = file_io
        self.base_dir = base_dir
        self.force_data = force_data
        self.bias_pot_calc = BiasPotentialCalculation(base_dir)

    def compute(self, state: OptimizationState):
        raise NotImplementedError("Subclasses must implement compute()")

    def _add_bias_and_update_state(self, state, raw_energy, raw_gradient):
        # Store raw
        state.raw_energy = raw_energy
        state.raw_gradient = raw_gradient
        state.energies["raw"] = raw_energy
        state.gradients["raw"] = raw_gradient

        # Bias
        _, bias_e, bias_g, bpa_hess = self.bias_pot_calc.main(
            raw_energy,
            raw_gradient,
            state.geometry,
            state.element_list,
            self.force_data,
            state.pre_bias_gradient,
            state.iter,
            initial_geom_num_list=state.initial_geometry,
        )
        state.bias_energy = bias_e
        state.bias_gradient = bias_g
        state.energies["bias"] = bias_e
        state.gradients["bias"] = bias_g

        # Effective
        state.effective_energy = bias_e
        state.effective_gradient = bias_g
        state.energies["effective"] = state.effective_energy
        state.gradients["effective"] = state.effective_gradient

        # Attach bias Hessian for later use
        state.bias_hessian = bpa_hess
        return state


class StandardHandler(BasePotentialHandler):
    """
    Handles standard single-PES calculations.
    """

    def __init__(self, calculator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculator = calculator

    def compute(self, state: OptimizationState):
        file_path = self.file_io.make_psi4_input_file(
            self.file_io.print_geometry_list(
                state.geometry * self.config.bohr2angstroms,
                state.element_list,
                self.config.electric_charge_and_multiplicity,
            ),
            state.iter,
        )

        self.calculator.Model_hess = copy.deepcopy(state.Model_hess)
        e, g, geom_num_list, exit_flag = self.calculator.single_point(
            file_path,
            state.element_number_list,
            state.iter,
            self.config.electric_charge_and_multiplicity,
            self.calculator.xtb_method,
        )
        state.geometry = np.array(geom_num_list, dtype="float64")
        if exit_flag:
            state.exit_flag = True
            return state

        state = self._add_bias_and_update_state(state, e, g)
        state.Model_hess = copy.deepcopy(self.calculator.Model_hess)
        return state

class ModelFunctionHandler(BasePotentialHandler):
    def __init__(self, calc1, calc2, mf_args, config, file_io, base_dir, force_data):
        super().__init__(config, file_io, base_dir, force_data)
        self.calc1 = calc1
        self.calc2 = calc2
        
        self.method_name = mf_args[0].lower()
        self.params = mf_args[1:]
        
        self.is_bitss = "bitss" in self.method_name
        self.single_element_list = None 
        
        self.mf_instance = self._load_mf_class()
        self._apply_config_params()

        self.bitss_geom1_history = []
        self.bitss_geom2_history = []
        self.bitss_ref_geom = None
        
        self.bitss_initialized = False 

        if self.is_bitss:
            self._setup_bitss_initialization()

    def _load_mf_class(self):
        if self.method_name == "opt_meci": 
            return OptMECI()
        if self.method_name == "opt_mesx": 
            return OptMESX()
        if self.method_name == "opt_mesx_2":
            return OptMESX2()
        if self.method_name == "seam":
            return SeamModelFunction()
        if self.method_name == "conical": 
            return ConicalModelFunction()
        if self.method_name == "avoiding": 
            return AvoidingModelFunction()
        if self.method_name == "bitss": 
            return BITSSModelFunction(np.zeros(1), np.zeros(1))
        raise ValueError(f"Unknown Model Function: {self.method_name}")

    def _apply_config_params(self):
        if hasattr(self.config.args, 'alpha') and self.config.args.alpha is not None:
            if hasattr(self.mf_instance, 'alpha'):
                self.mf_instance.alpha = float(self.config.args.alpha)
        if hasattr(self.config.args, 'sigma') and self.config.args.sigma is not None:
            if hasattr(self.mf_instance, 'sigma'):
                self.mf_instance.sigma = float(self.config.args.sigma)

    def _setup_bitss_initialization(self):
        if len(self.params) < 1:
            raise ValueError("BITSS requires a reference geometry file path.")
        
        temp_io = FileIO(self.base_dir, self.params[0])
        g_list, _, _ = temp_io.make_geometry_list(self.config.electric_charge_and_multiplicity)
        
        coords_ang = np.array([atom[1:4] for atom in g_list[0][2:]], dtype=float)
        self.bitss_ref_geom = coords_ang / self.config.bohr2angstroms

    def compute(self, state: OptimizationState):
        iter_idx = state.iter
        
        if self.single_element_list is None:
            self.single_element_list = state.element_list[:len(state.element_list)//2] if self.is_bitss else state.element_list

        # --- 1. Prepare Geometries ---
        if self.is_bitss:
            n_atoms = len(self.single_element_list)
            geom_1, geom_2 = state.geometry[:n_atoms], state.geometry[n_atoms:]
            
            if not self.bitss_initialized:
                self.mf_instance = BITSSModelFunction(geom_1, geom_2)
                self._apply_config_params()
                self.bitss_initialized = True
        else:
            geom_1 = geom_2 = state.geometry

        # State 1
        e1, g1, ex1 = self._run_calc(self.calc1, geom_1, self.single_element_list, self.config.electric_charge_and_multiplicity, "State1", iter_idx)
        
        # State 2
        if self.is_bitss:
            chg_mult_2 = self.config.electric_charge_and_multiplicity
        else:
            if len(self.params) >= 2:
                chg_mult_2 = [int(self.params[0]), int(self.params[1])]
            else:
                chg_mult_2 = self.config.electric_charge_and_multiplicity

        e2, g2, ex2 = self._run_calc(self.calc2, geom_2, self.single_element_list, chg_mult_2, "State2", iter_idx)

        if ex1 or ex2:
            state.exit_flag = True
            return state

        h1 = self.calc1.Model_hess
        h2 = self.calc2.Model_hess

        # --- 3. Compute Model Function Energy, Gradient, Hessian ---
        if self.is_bitss:
            mf_E = self.mf_instance.calc_energy(e1, e2, geom_1, geom_2, g1, g2, iter_idx)
            mf_G1, mf_G2 = self.mf_instance.calc_grad(e1, e2, geom_1, geom_2, g1, g2)
            mf_G = np.vstack((np.array(mf_G1), np.array(mf_G2))).astype(np.float64)
            
            if hasattr(self.mf_instance, "calc_hess"):
                 try:
                     raw_H = self.mf_instance.calc_hess(e1, e2, g1, g2, h1, h2)
                     if raw_H is not None:
                         mf_H = raw_H
                     else:
                         mf_H = self._make_block_diag_hess(h1, h2)
                 except:
                     mf_H = self._make_block_diag_hess(h1, h2)
            else:
                 mf_H = self._make_block_diag_hess(h1, h2)

            self.bitss_geom1_history.append(geom_1 * self.config.bohr2angstroms)
            self.bitss_geom2_history.append(geom_2 * self.config.bohr2angstroms)

        else:
            # Standard Mode (3N)
            mf_E = self.mf_instance.calc_energy(e1, e2)
            
            raw_output = self.mf_instance.calc_grad(e1, e2, g1, g2)
            if isinstance(raw_output, (tuple, list)):
                raw_G = np.array(raw_output[0]).astype(np.float64)
            else:
                raw_G = np.array(raw_output).astype(np.float64)

            if raw_G.ndim != 2:
                 if raw_G.size == len(self.single_element_list) * 3:
                     mf_G = raw_G.reshape(len(self.single_element_list), 3)
                 else:
                     mf_G = raw_G
            else:
                 mf_G = raw_G

            mf_H = None
            if hasattr(self.mf_instance, "calc_hess"):
                try:
                    raw_H = self._call_calc_hess_safely(self.mf_instance, e1, e2, g1, g2, h1, h2)
                    if raw_H is not None:
                        if isinstance(raw_H, (tuple, list)):
                            mf_H = raw_H[0]
                        else:
                            mf_H = raw_H
                except Exception as e:
                    print(f"Note: calc_hess failed or not applicable ({e}), falling back to average.")
            
            if mf_H is None:
                mf_H = 0.5 * (h1 + h2)

        # --- 4. Apply Bias Potential ---
        if self.is_bitss:
            _, be1, bg1, bh1 = self.bias_pot_calc.main(
                0.0, g1 * 0.0, geom_1, self.single_element_list, self.force_data, 
                state.pre_bias_gradient[:len(geom_1)] if state.pre_bias_gradient is not None else None,
                iter_idx
            )
            _, be2, bg2, bh2 = self.bias_pot_calc.main(
                0.0, g2 * 0.0, geom_2, self.single_element_list, self.force_data,
                state.pre_bias_gradient[len(geom_1):] if state.pre_bias_gradient is not None else None,
                iter_idx
            )
            
            final_E = mf_E + be1 + be2
            final_G = mf_G + np.vstack((bg1, bg2))
            bias_H = self._make_block_diag_hess(bh1, bh2)
            
        else:
            _, final_E, final_G, bias_H = self.bias_pot_calc.main(
                mf_E, mf_G, state.geometry, self.single_element_list, self.force_data,
                state.pre_bias_gradient, iter_idx
            )

        # --- 5. Update State ---
        state.raw_energy = mf_E
        state.raw_gradient = mf_G
        state.bias_energy = final_E
        state.bias_gradient = final_G
        state.effective_gradient = final_G
        
        state.energies["raw"] = mf_E
        state.energies["effective"] = final_E
        state.gradients["raw"] = mf_G
        state.gradients["effective"] = final_G
        
        state.Model_hess = mf_H 
        state.bias_hessian = bias_H
        
        return state

    def _make_block_diag_hess(self, h1, h2):
        d1 = h1.shape[0]
        d2 = h2.shape[0]
        full_H = np.zeros((d1 + d2, d1 + d2))
        full_H[:d1, :d1] = h1
        full_H[d1:, d1:] = h2
        return full_H

    def _call_calc_hess_safely(self, instance, e1, e2, g1, g2, h1, h2):
        sig = inspect.signature(instance.calc_hess)
        params = sig.parameters
        if len(params) == 2:
            return instance.calc_hess(h1, h2)
        elif len(params) == 4:
            return instance.calc_hess(g1, g2, h1, h2)
        else:
            return instance.calc_hess(e1, e2, g1, g2, h1, h2)

    def _run_calc(self, calc_inst, geom, elems, chg_mult, label, iter_idx):
        run_dir = os.path.join(self.base_dir, label, f"iter{iter_idx}")
        os.makedirs(run_dir, exist_ok=True)
        old_dir = calc_inst.BPA_FOLDER_DIRECTORY
        calc_inst.BPA_FOLDER_DIRECTORY = run_dir
        
        # Charge/Multiplicity update for PySCF compatibility
        calc_inst.electronic_charge = chg_mult[0]
        calc_inst.spin_multiplicity = chg_mult[1]

        geom_str = self.file_io.print_geometry_list(geom * self.config.bohr2angstroms, elems, chg_mult, display_flag=True)
        inp_path = self.file_io.make_psi4_input_file(geom_str, iter_idx, path=run_dir)
        
        # Method string for xTB
        method_str = getattr(calc_inst, "xtb_method", "")
        if method_str is None:
            method_str = ""

        # [FIX] Convert list to numpy array (int) to avoid 'list has no attribute tolist' error in tblite tools
        atom_nums = np.array([element_number(el) for el in elems], dtype=int)

        e, g, _, ex = calc_inst.single_point(
            inp_path, 
            atom_nums,  # Passing numpy array instead of list
            iter_idx, 
            chg_mult, 
            method=method_str
        )
        
        calc_inst.BPA_FOLDER_DIRECTORY = old_dir
        return e, g, ex

    def finalize_bitss_trajectory(self):
        if not self.is_bitss or not self.bitss_geom1_history: return
        filename = os.path.join(self.base_dir, f"{self.file_io.NOEXT_START_FILE}_traj.xyz")
        full_seq = self.bitss_geom1_history + self.bitss_geom2_history[::-1]
        with open(filename, 'w') as f:
            for s, g in enumerate(full_seq):
                f.write(f"{len(g)}\nBITSS_Step {s}\n")
                for i, atom in enumerate(g):
                    f.write(f"{self.single_element_list[i]:2s} {atom[0]:12.8f} {atom[1]:12.8f} {atom[2]:12.8f}\n")
  
class ONIOMHandler(BasePotentialHandler):
    """
    Handles ONIOM calculations with microiterations.
    """

    def __init__(
        self,
        high_calc,
        low_calc,
        high_atoms,
        link_atoms,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.high_calc = high_calc
        self.low_calc = low_calc
        self.high_atoms = high_atoms
        self.link_atoms = link_atoms

    def compute(self, state: OptimizationState):
        raise "Not Implemented!"
        # This handler runs one ONIOM iteration including microiterations.
        # The logic mirrors optimize_oniom from the legacy code.
        force_data = self.force_data
        config = self.config
        bohr2angstroms = config.bohr2angstroms

        # Build mappings
        linker_atom_pair_num = specify_link_atom_pairs(
            state.geometry, state.element_list, self.high_atoms, self.link_atoms
        )
        real_2_highlayer_label_connect_dict, highlayer_2_real_label_connect_dict = (
            link_number_high_layer_and_low_layer(self.high_atoms)
        )

        # Separate layers
        high_layer_geom_num_list, high_layer_element_list = (
            separate_high_layer_and_low_layer(
                state.geometry, linker_atom_pair_num, self.high_atoms, state.element_list
            )
        )

        # Prepare Hessians
        LL_Model_hess = copy.deepcopy(state.Model_hess)
        HL_Model_hess = np.eye(len(high_layer_element_list) * 3)

        # Bias calculators
        LL_Calc_BiasPot = BiasPotentialCalculation(self.base_dir)
        CalcBiaspot_real = self.bias_pot_calc  # Use parent

        # Previous vars
        pre_model_HL_B_e = 0.0
        pre_model_HL_B_g = np.zeros((len(high_layer_element_list), 3))
        pre_model_HL_g = np.zeros((len(high_layer_element_list), 3))
        pre_real_LL_B_e = state.pre_bias_energy
        pre_real_LL_e = state.pre_raw_energy
        pre_real_LL_B_g = copy.deepcopy(state.pre_bias_gradient)
        pre_real_LL_g = copy.deepcopy(state.pre_raw_gradient)
        pre_real_LL_move_vector = copy.deepcopy(state.pre_move_vector)
        pre_model_HL_move_vector = np.zeros((len(high_layer_element_list), 3))

        # High-layer optimizer
        HL_CMV = CalculateMoveVector(
            config.DELTA,
            high_layer_element_list[: len(self.high_atoms)],
            config.args.saddle_order,
            config.FC_COUNT,
            config.temperature,
            max_trust_radius=config.max_trust_radius,
            min_trust_radius=config.min_trust_radius,
        )
        HL_optimizer_instances = HL_CMV.initialization(force_data["opt_method"])
        for inst in HL_optimizer_instances:
            inst.set_hessian(
                HL_Model_hess[: len(self.high_atoms) * 3, : len(self.high_atoms) * 3]
            )
            if config.DELTA != "x":
                inst.DELTA = config.DELTA

        # Model low-layer calc
        self.low_calc.Model_hess = LL_Model_hess
        model_LL_e, model_LL_g, high_layer_geom_num_list, finish_frag = (
            self.low_calc.single_point(
                self.file_io.make_psi4_input_file(
                    self.file_io.print_geometry_list(
                        state.geometry * bohr2angstroms,
                        state.element_list,
                        config.electric_charge_and_multiplicity,
                    ),
                    state.iter,
                ),
                high_layer_element_list,
                state.iter,
                config.electric_charge_and_multiplicity,
                force_data["oniom_flag"][2],
                geom_num_list=high_layer_geom_num_list * bohr2angstroms,
            )
        )
        if finish_frag:
            state.exit_flag = True
            return state

        # Microiterations on low layer
        LL_CMV = CalculateMoveVector(
            config.DELTA,
            state.element_list,
            config.args.saddle_order,
            config.FC_COUNT,
            config.temperature,
        )
        LL_optimizer_instances = LL_CMV.initialization(["fire"])
        LL_optimizer_instances[0].display_flag = False

        current_geom_num_list = copy.deepcopy(state.geometry)
        real_initial_geom_num_list = copy.deepcopy(state.geometry)
        real_pre_geom = copy.deepcopy(state.geometry)

        low_layer_converged = False
        for microiter in range(config.microiter_num):
            self.low_calc.Model_hess = LL_Model_hess
            real_LL_e, real_LL_g, current_geom_num_list, finish_frag = (
                self.low_calc.single_point(
                    self.file_io.make_psi4_input_file(
                        self.file_io.print_geometry_list(
                            current_geom_num_list * bohr2angstroms,
                            state.element_list,
                            config.electric_charge_and_multiplicity,
                            display_flag=False,
                        ),
                        microiter,
                    ),
                    state.element_list,
                    microiter,
                    config.electric_charge_and_multiplicity,
                    force_data["oniom_flag"][2],
                    geom_num_list=current_geom_num_list * bohr2angstroms,
                )
            )
            LL_Model_hess = copy.deepcopy(self.low_calc.Model_hess)
            LL_Calc_BiasPot.Model_hess = LL_Model_hess
            _, real_LL_B_e, real_LL_B_g, LL_BPA_hessian = LL_Calc_BiasPot.main(
                real_LL_e,
                real_LL_g,
                current_geom_num_list,
                state.element_list,
                force_data,
                pre_real_LL_B_g,
                microiter,
                real_initial_geom_num_list,
            )

            for inst in LL_optimizer_instances:
                inst.set_bias_hessian(LL_BPA_hessian)
                if microiter % config.FC_COUNT == 0:
                    inst.set_hessian(LL_Model_hess)

            if len(force_data["opt_fragment"]) > 0:
                real_LL_B_g = self._calc_fragment_grads(
                    real_LL_B_g, force_data["opt_fragment"]
                )
                real_LL_g = self._calc_fragment_grads(
                    real_LL_g, force_data["opt_fragment"]
                )

            prev_geom = current_geom_num_list.copy()
            current_geom_num_list_ang, LL_move_vector, LL_optimizer_instances = (
                LL_CMV.calc_move_vector(
                    microiter,
                    current_geom_num_list,
                    real_LL_B_g,
                    pre_real_LL_B_g,
                    real_pre_geom,
                    real_LL_B_e,
                    pre_real_LL_B_e,
                    pre_real_LL_move_vector,
                    real_initial_geom_num_list,
                    real_LL_g,
                    pre_real_LL_g,
                    LL_optimizer_instances,
                    print_flag=False,
                )
            )
            current_geom_num_list = current_geom_num_list_ang / bohr2angstroms

            # Fix high-layer atoms
            for key, value in highlayer_2_real_label_connect_dict.items():
                current_geom_num_list[value - 1] = copy.deepcopy(
                    high_layer_geom_num_list[key - 1]
                )

            # Fix user-specified atoms
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    current_geom_num_list[j - 1] = copy.deepcopy(
                        real_initial_geom_num_list[j - 1]
                    )

            displacement_vector = current_geom_num_list - prev_geom
            low_layer_grads = []
            low_layer_displacements = []
            for i in range(len(state.element_list)):
                if (i + 1) not in self.high_atoms:
                    low_layer_grads.append(real_LL_B_g[i])
                    low_layer_displacements.append(displacement_vector[i])
            low_layer_grads = np.array(low_layer_grads)
            low_layer_displacements = np.array(low_layer_displacements)

            low_layer_rms_grad = self._calculate_rms_safely(low_layer_grads)
            max_displacement = np.abs(displacement_vector).max()
            rms_displacement = self._calculate_rms_safely(displacement_vector)

            if (
                (low_layer_rms_grad < 0.0003)
                and (low_layer_grads.max() < 0.0006 if len(low_layer_grads) > 0 else True)
                and (max_displacement < 0.003)
                and (rms_displacement < 0.002)
            ):
                low_layer_converged = True
                break

            # Update previous for microiter
            pre_real_LL_B_e = real_LL_B_e
            pre_real_LL_g = real_LL_g
            pre_real_LL_B_g = real_LL_B_g
            pre_real_LL_move_vector = LL_move_vector
            real_pre_geom = current_geom_num_list

        # Update state geometry after microiterations
        state.geometry = current_geom_num_list
        geom_num_list = current_geom_num_list

        # Model high-layer calc
        self.high_calc.Model_hess = HL_Model_hess
        model_HL_e, model_HL_g, high_layer_geom_num_list, finish_frag = (
            self.high_calc.single_point(
                self.file_io.make_psi4_input_file(
                    self.file_io.print_geometry_list(
                        geom_num_list * bohr2angstroms,
                        state.element_list,
                        config.electric_charge_and_multiplicity,
                    ),
                    state.iter,
                ),
                high_layer_element_list,
                state.iter,
                config.electric_charge_and_multiplicity,
                method="",
                geom_num_list=high_layer_geom_num_list * bohr2angstroms,
            )
        )
        HL_Model_hess = copy.deepcopy(self.high_calc.Model_hess)
        if finish_frag:
            state.exit_flag = True
            return state

        # Combine gradients
        _, tmp_model_HL_B_e, tmp_model_HL_B_g, LL_BPA_hessian = LL_Calc_BiasPot.main(
            0.0,
            real_LL_g * 0.0,
            geom_num_list,
            state.element_list,
            force_data,
            pre_real_LL_B_g * 0.0,
            state.iter,
            real_initial_geom_num_list,
        )
        tmp_model_HL_g = tmp_model_HL_B_g * 0.0
        for key, value in real_2_highlayer_label_connect_dict.items():
            tmp_model_HL_B_g[key - 1] += model_HL_g[value - 1] - model_LL_g[value - 1]
            tmp_model_HL_g[key - 1] += model_HL_g[value - 1] - model_LL_g[value - 1]

        bool_list = []
        for i in range(len(state.element_list)):
            if i in self.high_atoms:
                bool_list.extend([True, True, True])
            else:
                bool_list.extend([False, False, False])

        HL_BPA_hessian = LL_BPA_hessian[np.ix_(bool_list, bool_list)]
        for inst in HL_optimizer_instances:
            inst.set_bias_hessian(HL_BPA_hessian)
            if state.iter % config.FC_COUNT == 0:
                inst.set_hessian(
                    HL_Model_hess[: len(self.high_atoms) * 3, : len(self.high_atoms) * 3]
                )

        if len(force_data["opt_fragment"]) > 0:
            tmp_model_HL_B_g = self._calc_fragment_grads(
                tmp_model_HL_B_g, force_data["opt_fragment"]
            )
            tmp_model_HL_g = self._calc_fragment_grads(
                tmp_model_HL_g, force_data["opt_fragment"]
            )

        model_HL_B_g = copy.deepcopy(model_HL_g)
        model_HL_B_e = model_HL_e + tmp_model_HL_B_e
        for key, value in real_2_highlayer_label_connect_dict.items():
            model_HL_B_g[value - 1] += tmp_model_HL_B_g[key - 1]

        pre_high_layer_geom_num_list = high_layer_geom_num_list.copy()
        high_layer_geom_num_list_ang, move_vector, HL_optimizer_instances = (
            HL_CMV.calc_move_vector(
                state.iter,
                high_layer_geom_num_list[: len(self.high_atoms)],
                model_HL_B_g[: len(self.high_atoms)],
                pre_model_HL_B_g[: len(self.high_atoms)],
                pre_high_layer_geom_num_list[: len(self.high_atoms)],
                model_HL_B_e,
                pre_model_HL_B_e,
                pre_model_HL_move_vector[: len(self.high_atoms)],
                high_layer_geom_num_list[: len(self.high_atoms)],
                model_HL_g[: len(self.high_atoms)],
                pre_model_HL_g[: len(self.high_atoms)],
                HL_optimizer_instances,
            )
        )
        high_layer_geom_num_list = high_layer_geom_num_list_ang / bohr2angstroms

        # Update full system geometry with high layer changes
        for l in range(len(high_layer_geom_num_list) - len(linker_atom_pair_num)):
            geom_num_list[
                highlayer_2_real_label_connect_dict[l + 1] - 1
            ] = copy.deepcopy(high_layer_geom_num_list[l])

        geom_num_list -= Calculationtools().calc_center_of_mass(
            geom_num_list, state.element_list
        )
        geom_num_list, _ = Calculationtools().kabsch_algorithm(
            geom_num_list, real_pre_geom
        )
        state.geometry = geom_num_list
        high_layer_geom_num_list, high_layer_element_list = (
            separate_high_layer_and_low_layer(
                geom_num_list, linker_atom_pair_num, self.high_atoms, state.element_list
            )
        )

        # Combine energies and gradients for real system
        real_e = real_LL_e + model_HL_e - model_LL_e
        real_B_e = real_LL_B_e + model_HL_B_e - model_LL_e
        real_g = real_LL_g + tmp_model_HL_g
        real_B_g = real_LL_B_g + tmp_model_HL_g

        state.raw_energy = real_e
        state.bias_energy = real_B_e
        state.raw_gradient = real_g
        state.bias_gradient = real_B_g
        state.effective_energy = real_B_e
        state.effective_gradient = real_B_g
        state.Model_hess = LL_Model_hess  # keep low layer hess for next step
        state.bias_hessian = HL_BPA_hessian
        state.energies["raw"] = real_e
        state.energies["bias"] = real_B_e
        state.energies["effective"] = real_B_e
        state.gradients["raw"] = real_g
        state.gradients["bias"] = real_B_g
        state.gradients["effective"] = real_B_g

        # Update previous caches for next iteration
        state.pre_raw_energy = real_e
        state.pre_bias_energy = real_B_e
        state.pre_raw_gradient = real_g
        state.pre_bias_gradient = real_B_g
        state.pre_move_vector = move_vector
        return state

    @staticmethod
    def _calc_fragment_grads(gradient, fragment_list):
        calced_gradient = gradient
        for fragment in fragment_list:
            tmp_grad = np.array([0.0, 0.0, 0.0], dtype="float64")
            for atom_num in fragment:
                tmp_grad += gradient[atom_num - 1]
            tmp_grad /= len(fragment)
            for atom_num in fragment:
                calced_gradient[atom_num - 1] = copy.deepcopy(tmp_grad)
        return calced_gradient

    @staticmethod
    def _calculate_rms_safely(vector, threshold=1e-10):
        filtered_vector = vector[np.abs(vector) > threshold]
        if filtered_vector.size > 0:
            return np.sqrt((filtered_vector**2).mean())
        else:
            return 0.0

class EDEELHandler(BasePotentialHandler):
    """
    Handles EDEEL calculations for Electron Transfer.
    Computes V11 (Reactant) and V22 (Product) diabatic potentials using
    energy decomposition of Complex, Donor, and Acceptor.
    # ref.: https://doi.org/10.1039/D3RA05784D
    # This is under constraction.
    Target function can be switched between:
    - 'reactant': Minimize V11
    - 'product':  Minimize V22
    - 'sx':       Minimize Seam of Crossing penalty function (Default)
    """
    def __init__(self, complex_calc, donor_atoms, acceptor_atoms, ede_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complex_calc = complex_calc
        self.donor_atoms = np.array(donor_atoms)
        self.acceptor_atoms = np.array(acceptor_atoms)
        
        # Charges and Multiplicities for all 5 states
        # Expected ede_params keys: 
        # 'complex': [chg, mult], 'd_ox': [chg, mult], 'd_red': [chg, mult], ...
        self.params = ede_params
        
        # SX penalty weight (sigma) and target mode
        self.sigma = kwargs.get('sigma', 2.0)
        self.target_mode = kwargs.get('target_mode', 'sx') 

        # Setup directories for components to avoid file collision
        self.dirs = {
            'complex': self.base_dir, # Complex runs in root
            'd_ox': os.path.join(self.base_dir, "Components", "Donor_Ox"),
            'd_red': os.path.join(self.base_dir, "Components", "Donor_Red"),
            'a_ox': os.path.join(self.base_dir, "Components", "Acceptor_Ox"),
            'a_red': os.path.join(self.base_dir, "Components", "Acceptor_Red"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def compute(self, state: OptimizationState):
        config = self.config
        full_geom = state.geometry
        
        # 1. Extract Fragment Geometries
        # Note: Atoms indices are 1-based, numpy is 0-based
        d_indices = self.donor_atoms - 1
        a_indices = self.acceptor_atoms - 1
        
        geom_d = full_geom[d_indices]
        geom_a = full_geom[a_indices]
        
        # Helper to run calculation for one component
        def run_component(label, calc_geom, atom_indices, chg_mult, target_dir):
            # Prepare file IO for this component
            # Note: We create a specific input file in the target directory
            input_content = self.file_io.print_geometry_list(
                calc_geom * config.bohr2angstroms,
                [state.element_list[i] for i in atom_indices], 
                chg_mult,
                display_flag=False
            )
            
            # Temporarily switch directory in calculator
            original_dir = self.complex_calc.BPA_FOLDER_DIRECTORY
            self.complex_calc.BPA_FOLDER_DIRECTORY = target_dir
            
            # Run Single Point
            # Assuming make_psi4_input_file can accept a 'path' argument or we construct the path manually
            # Here we assume the file_io or calculator handles the path based on BPA_FOLDER_DIRECTORY
            e, g, _, exit_flag = self.complex_calc.single_point(
                self.file_io.make_psi4_input_file(input_content, state.iter, path=target_dir),
                [state.element_list[i] for i in atom_indices], 
                state.iter,
                chg_mult,
                method="" 
            )
            
            # Restore directory
            self.complex_calc.BPA_FOLDER_DIRECTORY = original_dir
            
            return e, g, exit_flag

        # 2. Run 5 Calculations
        # A. Complex (n+m)
        e_c, g_c, exit_c = run_component(
            'complex', full_geom, range(len(full_geom)), 
            self.params['complex'], self.dirs['complex']
        )
        
        # B. Donor Ox (n)
        e_do, g_do, exit_do = run_component(
            'd_ox', geom_d, d_indices, 
            self.params['d_ox'], self.dirs['d_ox']
        )
        
        # C. Donor Red (n+1)
        e_dr, g_dr, exit_dr = run_component(
            'd_red', geom_d, d_indices, 
            self.params['d_red'], self.dirs['d_red']
        )
        
        # D. Acceptor Ox (m)
        e_ao, g_ao, exit_ao = run_component(
            'a_ox', geom_a, a_indices, 
            self.params['a_ox'], self.dirs['a_ox']
        )
        
        # E. Acceptor Red (m+1)
        e_ar, g_ar, exit_ar = run_component(
            'a_red', geom_a, a_indices, 
            self.params['a_red'], self.dirs['a_red']
        )

        if any([exit_c, exit_do, exit_dr, exit_ao, exit_ar]):
            state.exit_flag = True
            return state

        # 3. Construct V11 and V22
        # V11 (Initial State) = E_Complex - E_Donor_Ox + E_Donor_Red
        # V22 (Final State)   = E_Complex - E_Acceptor_Ox + E_Acceptor_Red
        
        V11 = e_c - e_do + e_dr
        V22 = e_c - e_ao + e_ar
        
        state.energies['V11'] = V11
        state.energies['V22'] = V22
        state.energies['gap'] = V11 - V22
        
        # 4. Construct Gradients
        # Helper to map fragment gradients back to full system dimensions
        def map_grad(g_frag, indices):
            g_full = np.zeros_like(g_c)
            for i, idx in enumerate(indices):
                g_full[idx] = g_frag[i]
            return g_full

        grad_V11 = g_c - map_grad(g_do, d_indices) + map_grad(g_dr, d_indices)
        grad_V22 = g_c - map_grad(g_ao, a_indices) + map_grad(g_ar, a_indices)
        
        state.gradients['V11'] = grad_V11
        state.gradients['V22'] = grad_V22

        # 5. Determine Effective Energy/Gradient for Optimizer
        if self.target_mode == 'reactant':
            eff_E = V11
            eff_G = grad_V11
        elif self.target_mode == 'product':
            eff_E = V22
            eff_G = grad_V22
        else: # 'sx' (Seam of Crossing Search)
            # Penalty function approach: L = mean(V) + sigma * (V1 - V2)^2
            mean_V = 0.5 * (V11 + V22)
            diff_V = V11 - V22
            penalty = self.sigma * (diff_V ** 2)
            
            eff_E = mean_V + penalty
            
            # Gradient of L: dL/dX = 0.5(g1 + g2) + 2*sigma*diff_V * (g1 - g2)
            eff_G = 0.5 * (grad_V11 + grad_V22) + 2 * self.sigma * diff_V * (grad_V11 - grad_V22)
            
            state.energies['penalty'] = penalty

        # Store results
        state.raw_energy = eff_E
        state.raw_gradient = eff_G
        
        # Bias processing (if any)
        self._add_bias_and_update_state(state, eff_E, eff_G)
        
        return state

# =====================================================================================
# 4. Managers for constraints, convergence, Hessian, logging, result paths
# =====================================================================================
class ConstraintManager:
    def __init__(self, config):
        self.config = config

    def constrain_flag_check(self, force_data):
        
        projection_constrain = len(force_data["projection_constraint_condition_list"]) > 0 and any(s.lower() == "crsirfo" for s in self.config.args.opt_method)
     
        allactive_flag = len(force_data["fix_atoms"]) == 0
        if (
            "x" in force_data["projection_constraint_condition_list"]
            or "y" in force_data["projection_constraint_condition_list"]
            or "z" in force_data["projection_constraint_condition_list"]
        ):
            allactive_flag = False
        return projection_constrain, allactive_flag

    def init_projection_constraint(self, PC, geom_num_list, iter_idx, projection_constrain, hessian=None):
        if iter_idx == 0:
            if projection_constrain:
                PC.initialize(geom_num_list, hessian=hessian)
            return PC
        else:
            return PC

    def apply_projection_constraints(self, projection_constrain, PC, geom_num_list, g, B_g):
        if projection_constrain:
            g = copy.deepcopy(PC.calc_project_out_grad(geom_num_list, g))
            proj_d_B_g = copy.deepcopy(PC.calc_project_out_grad(geom_num_list, B_g - g))
            B_g = copy.deepcopy(g + proj_d_B_g)
            print("Projection was applied to gradient.")
        return g, B_g, PC

    def apply_projection_constraints_to_geometry(self, projection_constrain, PC, new_geometry, hessian=None):
        if projection_constrain:
            tmp_new_geometry = new_geometry / self.config.bohr2angstroms
            adjusted_geometry = (
                PC.adjust_init_coord(tmp_new_geometry, hessian=hessian)
                * self.config.bohr2angstroms
            )
            return adjusted_geometry, PC
        return new_geometry, PC

    def zero_fixed_atom_gradients(self, allactive_flag, force_data, g, B_g):
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                g[j - 1] = copy.deepcopy(g[j - 1] * 0.0)
                B_g[j - 1] = copy.deepcopy(B_g[j - 1] * 0.0)
        return g, B_g

    def project_out_translation_rotation(self, new_geometry, geom_num_list, allactive_flag):
        if allactive_flag:
            aligned_geometry, _ = Calculationtools().kabsch_algorithm(
                new_geometry / self.config.bohr2angstroms, geom_num_list
            )
            aligned_geometry *= self.config.bohr2angstroms
            return aligned_geometry
        else:
            return new_geometry

    def reset_fixed_atom_positions(
        self, new_geometry, initial_geom_num_list, allactive_flag, force_data
    ):
        if not allactive_flag:
            for j in force_data["fix_atoms"]:
                new_geometry[j - 1] = copy.deepcopy(
                    initial_geom_num_list[j - 1] * self.config.bohr2angstroms
                )
        return new_geometry

    @staticmethod
    def calc_fragment_grads(gradient, fragment_list):
        calced_gradient = gradient
        for fragment in fragment_list:
            tmp_grad = np.array([0.0, 0.0, 0.0], dtype="float64")
            for atom_num in fragment:
                tmp_grad += gradient[atom_num - 1]
            tmp_grad /= len(fragment)
            for atom_num in fragment:
                calced_gradient[atom_num - 1] = copy.deepcopy(tmp_grad)
        return calced_gradient


class ConvergenceChecker:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def calculate_rms_safely(vector, threshold=1e-10):
        filtered_vector = vector[np.abs(vector) > threshold]
        if filtered_vector.size > 0:
            return np.sqrt((filtered_vector**2).mean())
        else:
            return 0.0

    def check_convergence(self, state, displacement_vector, optimizer_instances):
        max_force = np.abs(state.effective_gradient).max()
        rms_force = self.calculate_rms_safely(state.effective_gradient)

        delta_max_force_threshold = max(
            0.0, self.config.MAX_FORCE_THRESHOLD - 1 * max_force
        )
        delta_rms_force_threshold = max(
            0.0, self.config.RMS_FORCE_THRESHOLD - 1 * rms_force
        )

        max_displacement = np.abs(displacement_vector).max()
        rms_displacement = self.calculate_rms_safely(displacement_vector)

        max_displacement_threshold = max(
            self.config.MAX_DISPLACEMENT_THRESHOLD,
            self.config.MAX_DISPLACEMENT_THRESHOLD + delta_max_force_threshold,
        )
        rms_displacement_threshold = max(
            self.config.RMS_DISPLACEMENT_THRESHOLD,
            self.config.RMS_DISPLACEMENT_THRESHOLD + delta_rms_force_threshold,
        )

        if (
            max_force < self.config.MAX_FORCE_THRESHOLD
            and rms_force < self.config.RMS_FORCE_THRESHOLD
            and max_displacement < max_displacement_threshold
            and rms_displacement < rms_displacement_threshold
        ):
            return True, max_displacement_threshold, rms_displacement_threshold
        
        for opt in optimizer_instances:
            if getattr(opt, 'proj_grad_converged', False):
                return True, max_displacement_threshold, rms_displacement_threshold
                
        
        
        return False, max_displacement_threshold, rms_displacement_threshold

    def judge_early_stop_due_to_no_negative_eigenvalues(self, geom_num_list, hessian, saddle_order, FC_COUNT, detect_negative_eigenvalues, folder_dir):
        if detect_negative_eigenvalues and FC_COUNT > 0:
            proj_hessian = Calculationtools().project_out_hess_tr_and_rot_for_coord(
                hessian, geom_num_list, geom_num_list, display_eigval=False
            )
            if proj_hessian is not None:
                eigvals = np.linalg.eigvalsh(proj_hessian)
                if not np.any(eigvals < -1e-10) and saddle_order > 0:
                    print("No negative eigenvalues detected while saddle_order > 0. Stopping optimization.")
                    with open(
                        folder_dir + "no_negative_eigenvalues_detected.txt",
                        "w",
                    ) as f:
                        f.write("No negative eigenvalues detected while saddle_order > 0. Stopping optimization.")
                    return True
        return False


class HessianManager:
    def __init__(self, config):
        self.config = config

    def calc_eff_hess_for_fix_atoms_and_set_hess(
        self,
        state,
        allactive_flag,
        force_data,
        BPA_hessian,
        n_fix,
        optimizer_instances,
        geom_num_list,
        B_g,
        g,
        projection_constrain,
        PC,
    ):
        if not allactive_flag:
            fix_num = []
            for fnum in force_data["fix_atoms"]:
                fix_num.extend([3 * (fnum - 1) + 0, 3 * (fnum - 1) + 1, 3 * (fnum - 1) + 2])
            fix_num = np.array(fix_num, dtype="int64")
            tmp_fix_hess = state.Model_hess[np.ix_(fix_num, fix_num)] + np.eye(
                (3 * n_fix)
            ) * 1e-10
            inv_tmp_fix_hess = np.linalg.pinv(tmp_fix_hess)
            tmp_fix_bias_hess = BPA_hessian[np.ix_(fix_num, fix_num)] + np.eye(
                (3 * n_fix)
            ) * 1e-10
            inv_tmp_fix_bias_hess = np.linalg.pinv(tmp_fix_bias_hess)
            BPA_hessian -= np.dot(
                BPA_hessian[:, fix_num], np.dot(inv_tmp_fix_bias_hess, BPA_hessian[fix_num, :])
            )

        for inst in optimizer_instances:
            if projection_constrain:
                if np.all(np.abs(BPA_hessian) < 1e-20):
                    proj_bpa_hess = PC.calc_project_out_hess(geom_num_list, B_g - g, BPA_hessian)
                else:
                    proj_bpa_hess = BPA_hessian
                inst.set_bias_hessian(proj_bpa_hess)
            else:
                inst.set_bias_hessian(BPA_hessian)

            if state.iter % self.config.FC_COUNT == 0 or (
                self.config.use_model_hessian is not None
                and state.iter % self.config.mFC_COUNT == 0
            ):
                if not allactive_flag:
                    state.Model_hess -= np.dot(
                        state.Model_hess[:, fix_num],
                        np.dot(inv_tmp_fix_hess, state.Model_hess[fix_num, :]),
                    )
                if projection_constrain:
                    proj_model_hess = PC.calc_project_out_hess(
                        geom_num_list, g, state.Model_hess
                    )
                    inst.set_hessian(proj_model_hess)
                else:
                    inst.set_hessian(state.Model_hess)
        return optimizer_instances


class RunLogger:
    def __init__(self, config):
        self.config = config

    def log_dynamic_csv(self, state, folder_dir, convergence_checker):
        csv_path = os.path.join(folder_dir, "energy_profile.csv")
        keys = sorted(state.energies.keys())
        if state.iter == 0:
            with open(csv_path, "w") as f:
                f.write("iter," + ",".join(keys) + "\n")
        values = [str(state.energies[k]) for k in keys]
        with open(csv_path, "a") as f:
            f.write(f"{state.iter}," + ",".join(values) + "\n")

        grad_profile = os.path.join(folder_dir, "gradient_profile.csv")
        if state.iter == 0:
            with open(grad_profile, "w") as f:
                f.write("gradient (RMS) [hartree/Bohr] \n")
        with open(grad_profile, "a") as f:
            f.write(str(convergence_checker.calculate_rms_safely(state.raw_gradient)) + "\n")

        bias_grad_profile = os.path.join(folder_dir, "bias_gradient_profile.csv")
        if state.iter == 0:
            with open(bias_grad_profile, "w") as f:
                f.write("bias gradient (RMS) [hartree/Bohr] \n")
        with open(bias_grad_profile, "a") as f:
            f.write(str(convergence_checker.calculate_rms_safely(state.bias_gradient)) + "\n")

    def save_energy_profiles(self, state, folder_dir):
        with open(folder_dir + "energy_profile_kcalmol.csv", "w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(state.ENERGY_LIST_FOR_PLOTTING)):
                f.write(
                    str(i)
                    + ","
                    + str(
                        state.ENERGY_LIST_FOR_PLOTTING[i]
                        - state.ENERGY_LIST_FOR_PLOTTING[0]
                    )
                    + "\n"
                )

    def geom_info_extract(self, state, force_data, file_directory, B_g, g, folder_dir):
        if len(force_data["geom_info"]) > 1:
            CSI = CalculationStructInfo()
            data_list, data_name_list = CSI.Data_extract(
                glob.glob(file_directory + "/*.xyz")[0], force_data["geom_info"]
            )

            for num, i in enumerate(force_data["geom_info"]):
                cos = CSI.calculate_cos(B_g[i - 1] - g[i - 1], g[i - 1])
                state.cos_list[num].append(cos)

            if state.iter == 0:
                with open(folder_dir + "geometry_info.csv", "a") as f:
                    f.write(",".join(data_name_list) + "\n")

            with open(folder_dir + "geometry_info.csv", "a") as f:
                f.write(",".join(list(map(str, data_list))) + "\n")
        return


class ResultPaths:
    @staticmethod
    def get_result_file_path(folder_dir, start_file):
        try:
            if folder_dir and start_file:
                base_name = os.path.splitext(os.path.basename(start_file))[0]
                optimized_filename = f"{base_name}_optimized.xyz"
                traj_filename = f"{base_name}_traj.xyz"

                optimized_struct_file = os.path.abspath(
                    os.path.join(folder_dir, optimized_filename)
                )
                traj_file = os.path.abspath(
                    os.path.join(folder_dir, traj_filename)
                )

                print("Optimized structure file path:", optimized_struct_file)
                print("Trajectory file path:", traj_file)
                return optimized_struct_file, traj_file
            else:
                print(
                    "Error: BPA_FOLDER_DIRECTORY or START_FILE is not set. Please run optimize() first."
                )
                return None, None
        except Exception as e:
            print(f"Error setting result file paths: {e}")
            return None, None

# =====================================================================================
# 5. Optimize Runner
# =====================================================================================
class Optimize:
    """
    Main runner orchestrating config, state, handler, and loop.
    """

    def __init__(self, args):
        self.config = OptimizationConfig(args)
        self.state = None
        self.handler = None
        self.file_io = None
        self.BPA_FOLDER_DIRECTORY = None
        self.START_FILE = None
        self.element_list = None
        self.SP = None
        self.final_file_directory = None
        self.final_geometry = None
        self.final_energy = None
        self.final_bias_energy = None
        self.symmetry = None
        self.irc_terminal_struct_paths = []
        self.optimized_struct_file = None
        self.traj_file = None
        self.bias_pot_params_grad_list = None
        self.bias_pot_params_grad_name_list = None

        # Managers
        self.constraints = ConstraintManager(self.config)
        self.convergence = ConvergenceChecker(self.config)
        self.hessian_mgr = HessianManager(self.config)
        self.logger = RunLogger(self.config)
        self.result_paths = ResultPaths()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_directory(self, input_file):
        self.START_FILE = input_file
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-2]
        date = datetime.datetime.now().strftime("%Y_%m_%d")
        base_dir = f"{date}/{os.path.splitext(input_file)[0]}_OPT_"

        if self.config.othersoft != "None":
            suffix = "ASE"
        elif self.config.sqm2:
            suffix = "SQM2"
        elif self.config.sqm1:
            suffix = "SQM1"
        elif self.config.args.usextb == "None" and self.config.args.usedxtb == "None":
            suffix = f"{self.config.FUNCTIONAL}_{self.config.BASIS_SET}"
        else:
            method = (
                self.config.args.usedxtb
                if self.config.args.usedxtb != "None"
                else self.config.args.usextb
            )
            suffix = method
        self.BPA_FOLDER_DIRECTORY = f"{base_dir}{suffix}_{timestamp}/"
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True)

        with open(os.path.join(self.BPA_FOLDER_DIRECTORY, "input.txt"), "w") as f:
            f.write(str(vars(self.config.args)))

    def _init_calculation_module(self):
        xtb_method = None
        if self.config.args.pyscf:
            from multioptpy.Calculator.pyscf_calculation_tools import Calculation
        elif self.config.sqm2:
            from multioptpy.Calculator.sqm2_calculation_tools import Calculation
            print("Use SQM2 potential.")
        elif self.config.sqm1:
            from multioptpy.Calculator.sqm1_calculation_tools import Calculation
        elif self.config.othersoft and self.config.othersoft != "None":
            if self.config.othersoft.lower() == "lj":
                from multioptpy.Calculator.lj_calculation_tools import Calculation
                print("Use Lennard-Jones cluster potential.")
            elif self.config.othersoft.lower() == "emt":
                from multioptpy.Calculator.emt_calculation_tools import Calculation
                print("Use EMT potential.")
            elif self.config.othersoft.lower() == "tersoff":
                from multioptpy.Calculator.tersoff_calculation_tools import Calculation
                print("Use Tersoff potential.")
            else:
                from multioptpy.Calculator.ase_calculation_tools import Calculation
                print("Use", self.config.othersoft)
                with open(
                    self.BPA_FOLDER_DIRECTORY + "use_" + self.config.othersoft + ".txt",
                    "w",
                ) as f:
                    f.write(self.config.othersoft + "\n")
                    f.write(self.config.BASIS_SET + "\n")
                    f.write(self.config.FUNCTIONAL + "\n")
        else:
            if self.config.args.usedxtb and self.config.args.usedxtb != "None":
                from multioptpy.Calculator.dxtb_calculation_tools import Calculation
                xtb_method = self.config.args.usedxtb
            elif self.config.args.usextb and self.config.args.usextb != "None":
                from multioptpy.Calculator.tblite_calculation_tools import Calculation
                xtb_method = self.config.args.usextb
            else:
                from multioptpy.Calculator.psi4_calculation_tools import Calculation
        return Calculation, xtb_method

    def _create_calculation(self, Calculation, xtb_method, model_hess, override_dir=None):
        target_dir = override_dir if override_dir else self.BPA_FOLDER_DIRECTORY
        calc = Calculation(
            START_FILE=self.START_FILE,
            N_THREAD=self.config.N_THREAD,
            SET_MEMORY=self.config.SET_MEMORY,
            FUNCTIONAL=self.config.FUNCTIONAL,
            FC_COUNT=self.config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=target_dir, 
            Model_hess=model_hess,
            software_type=self.config.othersoft,
            unrestrict=self.config.unrestrict,
            SUB_BASIS_SET=self.config.SUB_BASIS_SET,
            BASIS_SET=self.config.BASIS_SET,
            spin_multiplicity=self.config.spin_multiplicity,
            electronic_charge=self.config.electronic_charge,
            excited_state=self.config.excited_state,
            dft_grid=self.config.dft_grid,
            ECP=self.config.ECP,
            software_path_file=self.config.software_path_file,
        )
        calc.cpcm_solv_model = self.config.cpcm_solv_model
        calc.alpb_solv_model = self.config.alpb_solv_model
        calc.xtb_method = xtb_method
        return calc

    def _initialize_handler(self, element_list, force_data):
        Calculation, xtb_method = self._init_calculation_module()
        
   
        self.SP = self._create_calculation(Calculation, xtb_method, self.state.Model_hess)
        
        self.state.element_list = element_list
        self.state.element_number_list = np.array(
            [element_number(elem) for elem in element_list], dtype="int"
        )
        
        # --- EDEEL Mode ---
        # Assuming the parser sets an 'edeel' flag in args or force_data
        if hasattr(self.config.args, 'edeel') and self.config.args.edeel:
            print("Mode: EDEEL (Energy Decomposition and Extrapolation-based Electron Localization)")
            
            # Note: The user handles the parser. 
            # We assume 'ede_params', 'donor_atoms', and 'acceptor_atoms' are available in force_data.
            # Example structure for ede_params: {'complex': [0,1], 'd_ox': [0,1], ...}
            
            ede_params = force_data.get('edeel_params')
            d_atoms = force_data.get('donor_atoms', [])
            a_atoms = force_data.get('acceptor_atoms', [])
            
            if not ede_params or not d_atoms or not a_atoms:
                raise ValueError("EDEEL mode requires 'edeel_params', 'donor_atoms', and 'acceptor_atoms' in input.")

            return EDEELHandler(
                self.SP, 
                d_atoms, 
                a_atoms, 
                ede_params,
                self.config, 
                self.file_io, 
                self.BPA_FOLDER_DIRECTORY, 
                force_data
            )
        # --- Model Function Mode (NEW) ---
        elif len(self.config.args.model_function) > 0:
            print("Mode: Model Function Optimization")
            Calculation, xtb_method = self._init_calculation_module()
            
            # Create independent base directories for State 1 and State 2
            dir1 = os.path.join(self.BPA_FOLDER_DIRECTORY, "State1_base")
            dir2 = os.path.join(self.BPA_FOLDER_DIRECTORY, "State2_base")
            os.makedirs(dir1, exist_ok=True)
            os.makedirs(dir2, exist_ok=True)

            # Initialize two independent calculators to prevent state contamination
            calc1 = self._create_calculation(Calculation, xtb_method, self.state.Model_hess, override_dir=dir1)
            calc2 = self._create_calculation(Calculation, xtb_method, self.state.Model_hess, override_dir=dir2)

            handler = ModelFunctionHandler(
                calc1, calc2, 
                self.config.args.model_function, 
                self.config, 
                self.file_io, 
                self.BPA_FOLDER_DIRECTORY, 
                force_data
            )
            
            # --- BITSS Mode ---
            if handler.is_bitss:
                print("BITSS Mode detected: Expanding state to 2N atoms.")
                geom1 = self.state.geometry
                geom2 = handler.bitss_ref_geom
                
                if geom1.shape != geom2.shape:
                    raise ValueError("BITSS: Input and Reference geometries must have the same dimensions.")
                
                # Expand geometry and gradients to 6N dimensions
                self.state.geometry = np.vstack((geom1, geom2))
                self.state.initial_geometry = copy.deepcopy(self.state.geometry)
                self.state.pre_geometry = copy.deepcopy(self.state.geometry)
                
                n_atoms = len(element_list) # Original N
                
                # Current gradients
                self.state.raw_gradient = np.zeros((2 * n_atoms, 3))
                self.state.bias_gradient = np.zeros((2 * n_atoms, 3))
                self.state.effective_gradient = np.zeros((2 * n_atoms, 3))
                
                
                self.state.pre_raw_gradient = np.zeros((2 * n_atoms, 3))
                self.state.pre_bias_gradient = np.zeros((2 * n_atoms, 3))
                self.state.pre_effective_gradient = np.zeros((2 * n_atoms, 3))
                self.state.pre_move_vector = np.zeros((2 * n_atoms, 3))
               

                self.state.Model_hess = np.eye(2 * n_atoms * 3)
                
                # Double the element lists
                self.state.element_list = element_list + element_list
                self.state.element_number_list = np.concatenate((self.state.element_number_list, self.state.element_number_list))
            else:
                print(f"Standard Model Function Mode ({handler.method_name}): Using {len(element_list)} atoms.")
                
            return handler
        
        # --- ONIOM Mode ---
        elif len(self.config.args.oniom_flag) > 0:
            # ONIOM
            high_atoms = force_data["oniom_flag"][0]
            link_atoms = force_data["oniom_flag"][1]
            
            
            hl_dir = os.path.join(self.BPA_FOLDER_DIRECTORY, "High_Layer")
            ll_dir = os.path.join(self.BPA_FOLDER_DIRECTORY, "Low_Layer")
            os.makedirs(hl_dir, exist_ok=True)
            os.makedirs(ll_dir, exist_ok=True)
            
          
            high_calc = self._create_calculation(Calculation, xtb_method, self.state.Model_hess, override_dir=hl_dir)
            low_calc = self._create_calculation(Calculation, xtb_method, self.state.Model_hess, override_dir=ll_dir)
           

            return ONIOMHandler(
                high_calc,
                low_calc,
                high_atoms,
                link_atoms,
                self.config,
                self.file_io,
                self.BPA_FOLDER_DIRECTORY,
                force_data,
            )
        else:
            # Standard
            return StandardHandler(
                self.SP, self.config, self.file_io, self.BPA_FOLDER_DIRECTORY, force_data
            )
            
            
    # ------------------------------------------------------------------
    # Geometry parsing helper
    # ------------------------------------------------------------------
    def _extract_geom_from_geometry_list(self, geometry_list):
        """
        Extract geometry (Bohr) from geometry_list returned by FileIO.make_geometry_list.
        Assumes geometry_list[0][0] and [0][1] hold charge/multiplicity,
        and atom records start from geometry_list[0][2], each formatted as
        [element, x, y, z] in Angstrom.
        """
        try:
            atom_entries = geometry_list[0][2:]
            coords_ang = np.array([atom[1:4] for atom in atom_entries], dtype=float)
            geom_bohr = coords_ang / self.config.bohr2angstroms
            return geom_bohr
        except Exception as e:
            raise ValueError(f"Failed to parse geometry_list: {geometry_list}") from e

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self):
        input_list = (
            [self.config.args.INPUT]
            if isinstance(self.config.args.INPUT, str)
            else self.config.args.INPUT
        )
        job_file_list = []
        for job_file in input_list:
            if "*" in job_file:
                job_file_list.extend(glob.glob(job_file))
            else:
                job_file_list.append(job_file)

        for file in job_file_list:
            self._run_single_job(file)

        print("All calculations were completed.")
        self.get_result_file_path()
        return

    # ------------------------------------------------------------------
    # Extracted single-job runner to simplify run()
    # ------------------------------------------------------------------
    def _run_single_job(self, file):
        print("********************************")
        print(file)
        print("********************************")
        if not os.path.exists(file):
            print(f"{file} does not exist.")
            return

        self._setup_directory(file)
        self.file_io = FileIO(self.BPA_FOLDER_DIRECTORY, file)

        # Read geometry
        geometry_list, element_list, chg_mult = self.file_io.make_geometry_list(
            self.config.electric_charge_and_multiplicity
        )
        geom = self._extract_geom_from_geometry_list(geometry_list)
        self.element_list = element_list

        # Initialize state
        # NOTE: element_list is passed here, but self.state.element_list might be modified later (e.g. for BITSS)
        self.state = OptimizationState(element_list)
        self.state.geometry = copy.deepcopy(geom)
        self.state.initial_geometry = copy.deepcopy(geom)
        self.state.element_list = element_list
        self.state.element_number_list = np.array(
            [element_number(elem) for elem in element_list], dtype="int"
        )
        self.state.cos_list = [
            [] for _ in range(len(force_data_parser(self.config.args)["geom_info"]))
        ]

        force_data = force_data_parser(self.config.args)
        
        # Initialize Handler (This may update self.state.element_list and self.state.geometry for BITSS)
        self.handler = self._initialize_handler(element_list, force_data)

        # Constraint setup
        PC = ProjectOutConstrain(
            force_data["projection_constraint_condition_list"],
            force_data["projection_constraint_atoms"],
            force_data["projection_constraint_constant"],
        )
        projection_constrain, allactive_flag = self.constraints.constrain_flag_check(force_data)
        n_fix = len(force_data["fix_atoms"])

        # Move vector and optimizer
        # FIX: Use self.state.element_list which handles the 2N size in BITSS mode
        CMV = CalculateMoveVector(
            self.config.DELTA,
            self.state.element_list,
            self.config.args.saddle_order,
            self.config.FC_COUNT,
            self.config.temperature,
            self.config.use_model_hessian,
            max_trust_radius=self.config.max_trust_radius,
            min_trust_radius=self.config.min_trust_radius,
            projection_constraint=PC
        )
        optimizer_instances = CMV.initialization(force_data["opt_method"])
        for opt in optimizer_instances:
            opt.set_hessian(self.state.Model_hess)
            if self.config.DELTA != "x":
                opt.DELTA = self.config.DELTA
            # Gracefully handle optimizers without newton_tag
            supports_exact_hess = getattr(opt, "newton_tag", True)
            if (not supports_exact_hess) and self.config.FC_COUNT > 0 and (
                "eigvec" not in force_data["projection_constraint_condition_list"]
            ):
                print(
                    "Error: This optimizer does not support exact Hessian calculations."
                )
                sys.exit(0)

        # Koopman
        if self.config.koopman_analysis:
            # FIX: Use self.state.element_list
            KA = KoopmanAnalyzer(len(self.state.element_list), file_directory=self.BPA_FOLDER_DIRECTORY)
        else:
            KA = None

        # Initial files
        # FIX: Use self.state.element_list
        self.file_io.print_geometry_list(
            self.state.geometry * self.config.bohr2angstroms,
            self.state.element_list,
            chg_mult,
        )

        # Main loop
        for iter_idx in range(self.config.NSTEP):
            self.state.iter = iter_idx
            self.state.exit_flag = os.path.exists(
                self.BPA_FOLDER_DIRECTORY + "end.txt"
            )
            if self.state.exit_flag:
                break

            # shape condition check
            self.state.exit_flag = judge_shape_condition(
                self.state.geometry, self.config.shape_conditions
            )
            if self.state.exit_flag:
                break

            print(f"# ITR. {iter_idx}")

            # Compute potentials
            self.state = self.handler.compute(self.state)
            if self.state.exit_flag:
                break

            # Exact model Hessian update
            if (
                iter_idx % self.config.mFC_COUNT == 0
                and self.config.use_model_hessian is not None
                and self.config.FC_COUNT < 1
            ):
                self.state.Model_hess = ApproxHessian().main(
                    self.state.geometry,
                    self.state.element_list, # FIX: Use self.state.element_list
                    self.state.raw_gradient,
                    self.config.use_model_hessian,
                )
                if isinstance(self.handler, StandardHandler):
                    self.handler.calculator.Model_hess = self.state.Model_hess

            # Initial geometry save
            if iter_idx == 0:
                # FIX: Use self.state.element_list
                initial_geom_num_list, pre_geom = self._save_init_geometry(
                    self.state.geometry, self.state.element_list, allactive_flag
                )
                self.state.pre_geometry = pre_geom

            # Build combined hessian
            Hess = (
                self.state.bias_hessian + self.state.Model_hess
                if hasattr(self.state, "bias_hessian")
                else self.state.Model_hess
            )
            if iter_idx == 0 and self.convergence.judge_early_stop_due_to_no_negative_eigenvalues(
                self.state.geometry,
                Hess,
                self.config.args.saddle_order,
                self.config.FC_COUNT,
                self.config.detect_negative_eigenvalues,
                self.BPA_FOLDER_DIRECTORY,
            ):
                break

            # Constraints / projection
            PC = self.constraints.init_projection_constraint(
                PC, self.state.geometry, iter_idx, projection_constrain, hessian=Hess
            )
            optimizer_instances = self.hessian_mgr.calc_eff_hess_for_fix_atoms_and_set_hess(
                self.state,
                allactive_flag,
                force_data,
                self.state.bias_hessian if hasattr(self.state, "bias_hessian") else np.zeros_like(Hess),
                n_fix,
                optimizer_instances,
                self.state.geometry,
                self.state.bias_gradient,
                self.state.raw_gradient,
                projection_constrain,
                PC,
            )

            if not allactive_flag and len(force_data["opt_fragment"]) > 0:
                self.state.bias_gradient = self.constraints.calc_fragment_grads(
                    self.state.bias_gradient, force_data["opt_fragment"]
                )
                self.state.raw_gradient = self.constraints.calc_fragment_grads(
                    self.state.raw_gradient, force_data["opt_fragment"]
                )

            # logging
            self.logger.log_dynamic_csv(self.state, self.BPA_FOLDER_DIRECTORY, self.convergence)
            self.state.ENERGY_LIST_FOR_PLOTTING.append(
                self.state.raw_energy * self.config.hartree2kcalmol
            )
            self.state.BIAS_ENERGY_LIST_FOR_PLOTTING.append(
                self.state.bias_energy * self.config.hartree2kcalmol
            )
            self.state.NUM_LIST.append(iter_idx)

            # Geometry info extract
            # FIX: Use self.state.element_list
            self.logger.geom_info_extract(
                self.state,
                force_data,
                self.file_io.make_psi4_input_file(
                    self.file_io.print_geometry_list(
                        self.state.geometry * self.config.bohr2angstroms,
                        self.state.element_list,
                        chg_mult, 
                        display_flag=False
                    ),
                    iter_idx,
                ),
                self.state.bias_gradient,
                self.state.raw_gradient,
                self.BPA_FOLDER_DIRECTORY,
            )

            # Apply constraints to gradients
            g = copy.deepcopy(self.state.raw_gradient)
            B_g = copy.deepcopy(self.state.bias_gradient)
            g, B_g, PC = self.constraints.apply_projection_constraints(
                projection_constrain, PC, self.state.geometry, g, B_g
            )
            g, B_g = self.constraints.zero_fixed_atom_gradients(allactive_flag, force_data, g, B_g)
            
            self.state.raw_gradient = g
            self.state.bias_gradient = B_g
            self.state.effective_gradient = g + (B_g - g)

            if self.config.koopman_analysis and KA is not None:
                # FIX: Use self.state.element_list
                _ = KA.run(iter_idx, self.state.geometry, B_g, self.state.element_list)

            # Move vector
            new_geometry, move_vector, optimizer_instances = CMV.calc_move_vector(
                iter_idx,
                self.state.geometry,
                B_g,
                self.state.pre_bias_gradient,
                self.state.pre_geometry,
                self.state.bias_energy,
                self.state.pre_bias_energy,
                self.state.pre_move_vector,
                self.state.initial_geometry,
                g,
                self.state.pre_raw_gradient,
                optimizer_instances,
                projection_constrain,
            )

            # Projection / alignment
            new_geometry = self.constraints.project_out_translation_rotation(
                new_geometry, self.state.geometry, allactive_flag
            )
            new_geometry, PC = self.constraints.apply_projection_constraints_to_geometry(
                projection_constrain, PC, new_geometry, hessian=Hess
            )

            displacement_vector = (
                move_vector
                if iter_idx == 0
                else new_geometry / self.config.bohr2angstroms - self.state.geometry
            )

            # convergence
            converge_flag, max_disp_th, rms_disp_th = self.convergence.check_convergence(
                self.state, displacement_vector, optimizer_instances
            )

            # track gradients
            self.state.grad_list.append(self.convergence.calculate_rms_safely(g))
            self.state.bias_grad_list.append(self.convergence.calculate_rms_safely(B_g))

            # reset fixed atoms
            new_geometry = self.constraints.reset_fixed_atom_positions(
                new_geometry, self.state.initial_geometry, allactive_flag, force_data
            )

            # dissociation
            # FIX: Use self.state.element_list
            DC_exit_flag = self.dissociation_check(new_geometry, self.state.element_list)

            # print info
            self._print_info(
                self.state.raw_energy,
                self.state.bias_energy,
                B_g,
                displacement_vector,
                self.state.pre_raw_energy,
                self.state.pre_bias_energy,
                max_disp_th,
                rms_disp_th,
            )

            if converge_flag:
                if projection_constrain and iter_idx == 0:
                    pass
                else:
                    self.state.optimized_flag = True
                    print("\n=====================================================")
                    print("converged!!!")
                    print("=====================================================")
                    break

            if DC_exit_flag:
                self.state.DC_check_flag = True
                break

            # update previous
            self.state.pre_bias_energy = self.state.bias_energy
            self.state.pre_raw_energy = self.state.raw_energy
            self.state.pre_bias_gradient = B_g
            self.state.pre_raw_gradient = g
            self.state.pre_geometry = self.state.geometry
            self.state.pre_move_vector = move_vector
            self.state.geometry = new_geometry / self.config.bohr2angstroms

            # write next input
            # FIX: Use self.state.element_list
            self.file_io.print_geometry_list(
                new_geometry, self.state.element_list, chg_mult, display_flag=False
            )

        else:
            self.state.optimized_flag = False
            print("Reached maximum number of iterations. This is not converged.")
            with open(
                self.BPA_FOLDER_DIRECTORY + "not_converged.txt", "w"
            ) as f:
                f.write("Reached maximum number of iterations. This is not converged.")

        # Post steps
        if self.state.DC_check_flag:
            print("Dissociation is detected. Optimization stopped.")
            with open(
                self.BPA_FOLDER_DIRECTORY + "dissociation_is_detected.txt", "w"
            ) as f:
                f.write("Dissociation is detected. Optimization stopped.")

        # Vibrational analysis
        if self.config.freq_analysis and not self.state.exit_flag and not self.state.DC_check_flag:
            self._perform_vibrational_analysis(
                self.SP,
                self.state.geometry,
                self.state.element_list, # FIX: Use self.state.element_list
                self.state.initial_geometry,
                force_data,
                self._is_exact_hessian(iter_idx),
                self.file_io.make_psi4_input_file(
                    self.file_io.print_geometry_list(
                        self.state.geometry * self.config.bohr2angstroms,
                        self.state.element_list, # FIX: Use self.state.element_list
                        chg_mult,
                    ),
                    iter_idx,
                ),
                iter_idx,
                chg_mult,
                self.SP.xtb_method if hasattr(self.SP, "xtb_method") else None,
                self.state.raw_energy,
            )

        # Finalize
        # FIX: Use self.state.element_list
        self._finalize_optimization(
            self.file_io,
            Graph(self.BPA_FOLDER_DIRECTORY),
            self.state.grad_list,
            self.state.bias_grad_list,
            self.file_io.make_psi4_input_file(
                self.file_io.print_geometry_list(
                    self.state.geometry * self.config.bohr2angstroms,
                    self.state.element_list,
                    chg_mult,
                ),
                iter_idx,
            ),
            force_data,
            self.state.geometry,
            self.state.raw_energy,
            self.state.bias_energy,
            self.SP,
            self.state.exit_flag,
        )
        self._copy_final_results_from_state()

        # Analyses
        if self.state and self.config.CMDS:
            CMDPA = CMDSPathAnalysis(
                self.BPA_FOLDER_DIRECTORY,
                self.state.ENERGY_LIST_FOR_PLOTTING,
                self.state.BIAS_ENERGY_LIST_FOR_PLOTTING,
            )
            CMDPA.main()
        if self.state and self.config.PCA:
            PCAPA = PCAPathAnalysis(
                self.BPA_FOLDER_DIRECTORY,
                self.state.ENERGY_LIST_FOR_PLOTTING,
                self.state.BIAS_ENERGY_LIST_FOR_PLOTTING,
            )
            PCAPA.main()

        if self.state and len(self.config.irc) > 0:
            if self.config.args.usextb != "None":
                xtb_method = self.config.args.usextb
            else:
                xtb_method = "None"
            if self.state.iter % self.config.FC_COUNT == 0:
                hessian = self.state.Model_hess
            else:
                hessian = None
            
            # NOTE: IRC logic for BITSS might need special handling, but keeping standard
            EXEC_IRC = IRC(
                self.BPA_FOLDER_DIRECTORY,
                self.state.final_file_directory,
                self.config.irc,
                self.SP,
                self.state.element_list, # FIX: Use self.state.element_list
                self.config.electric_charge_and_multiplicity,
                force_data_parser(self.config.args),
                xtb_method,
                FC_count=int(self.config.FC_COUNT),
                hessian=hessian,
            )
            EXEC_IRC.run()
            self.irc_terminal_struct_paths = EXEC_IRC.terminal_struct_paths
        else:
            self.irc_terminal_struct_paths = []

        print(f"Trial of geometry optimization ({file}) was completed.")
        
    # ------------------------------------------------------------------
    # Secondary helpers reused from legacy
    # ------------------------------------------------------------------
    def _save_init_geometry(self, geom_num_list, element_list, allactive_flag):
        if allactive_flag:
            initial_geom_num_list = geom_num_list - Calculationtools().calc_center(
                geom_num_list, element_list
            )
            pre_geom = initial_geom_num_list - Calculationtools().calc_center(
                geom_num_list, element_list
            )
        else:
            initial_geom_num_list = geom_num_list
            pre_geom = initial_geom_num_list
        return initial_geom_num_list, pre_geom

    def dissociation_check(self, new_geometry, element_list):
        atom_label_list = list(range(len(new_geometry)))
        fragm_atom_num_list = []

        while len(atom_label_list) > 0:
            tmp_fragm_list = Calculationtools().check_atom_connectivity(
                new_geometry, element_list, atom_label_list[0]
            )
            atom_label_list = list(set(atom_label_list) - set(tmp_fragm_list))
            fragm_atom_num_list.append(tmp_fragm_list)

        if len(fragm_atom_num_list) > 1:
            fragm_dist_list = []
            geom_np = np.asarray(new_geometry)
            for fragm_1_indices, fragm_2_indices in itertools.combinations(
                fragm_atom_num_list, 2
            ):
                coords1 = geom_np[fragm_1_indices]
                coords2 = geom_np[fragm_2_indices]
                diff_matrix = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
                sq_dist_matrix = np.sum(diff_matrix ** 2, axis=2)
                min_sq_dist = np.min(sq_dist_matrix)
                min_dist = np.sqrt(min_sq_dist)
                fragm_dist_list.append(min_dist)

            min_interfragment_dist = min(fragm_dist_list)
            if min_interfragment_dist > self.config.DC_check_dist:
                print(
                    f"Minimum fragment distance (ang.) {min_interfragment_dist:.4f} > {self.config.DC_check_dist}"
                )
                print("These molecules are dissociated.")
                return True
        return False

    def _perform_vibrational_analysis(
        self,
        SP,
        geom_num_list,
        element_list,
        initial_geom_num_list,
        force_data,
        exact_hess_flag,
        file_directory,
        iter_idx,
        electric_charge_and_multiplicity,
        xtb_method,
        e,
    ):
        print("\n====================================================")
        print("Performing vibrational analysis...")
        print("====================================================\n")
        print("Is Exact Hessian calculated? : ", exact_hess_flag)

        if exact_hess_flag:
            g = np.zeros_like(geom_num_list, dtype="float64")
            exit_flag = False
        else:
            print("Calculate exact Hessian...")
            SP.hessian_flag = True
            e, g, geom_num_list, exit_flag = SP.single_point(
                file_directory, element_list, iter_idx, electric_charge_and_multiplicity, xtb_method
            )
            SP.hessian_flag = False

        if exit_flag:
            print("Error: QM calculation failed.")
            return

        _, B_e, _, BPA_hessian = self.handler.bias_pot_calc.main(
            e,
            g,
            geom_num_list,
            element_list,
            force_data,
            pre_B_g="",
            iter=iter_idx,
            initial_geom_num_list="",
        )
        tmp_hess = copy.deepcopy(SP.Model_hess)
        tmp_hess += BPA_hessian

        MV = MolecularVibrations(
            atoms=element_list, coordinates=geom_num_list, hessian=tmp_hess
        )
        MV.calculate_thermochemistry(
            e_tot=B_e,
            temperature=self.config.thermo_temperature,
            pressure=self.config.thermo_pressure,
        )
        MV.print_thermochemistry(
            output_file=self.BPA_FOLDER_DIRECTORY + "/thermochemistry.txt"
        )
        MV.print_normal_modes(
            output_file=self.BPA_FOLDER_DIRECTORY + "/normal_modes.txt"
        )
        MV.create_vibration_animation(
            output_dir=self.BPA_FOLDER_DIRECTORY + "/vibration_animation"
        )

        if not self.state.optimized_flag:
            print(
                "Warning: Vibrational analysis was performed, but the optimization did not converge. The result of thermochemistry is unreliable."
            )

    def _is_exact_hessian(self, iter_idx):
        if self.config.FC_COUNT == -1:
            return False
        elif iter_idx % self.config.FC_COUNT == 0 and self.config.FC_COUNT > 0:
            return True
        return False

    def _finalize_optimization(
        self,
        FIO,
        G,
        grad_list,
        bias_grad_list,
        file_directory,
        force_data,
        geom_num_list,
        e,
        B_e,
        SP,
        exit_flag,
    ):
        
        G.double_plot(
            self.state.NUM_LIST,
            self.state.ENERGY_LIST_FOR_PLOTTING,
            self.state.BIAS_ENERGY_LIST_FOR_PLOTTING,
        )
        G.single_plot(
            self.state.NUM_LIST,
            grad_list,
            file_directory,
            "",
            axis_name_2="gradient (RMS) [a.u.]",
            name="gradient",
        )
        G.single_plot(
            self.state.NUM_LIST,
            bias_grad_list,
            file_directory,
            "",
            axis_name_2="bias gradient (RMS) [a.u.]",
            name="bias_gradient",
        )

        if len(force_data["geom_info"]) > 1:
            for num, i in enumerate(force_data["geom_info"]):
                G.single_plot(self.state.NUM_LIST, self.state.cos_list[num], file_directory, i)

        FIO.make_traj_file()
        FIO.argrelextrema_txt_save(self.state.ENERGY_LIST_FOR_PLOTTING, "approx_TS", "max")
        FIO.argrelextrema_txt_save(self.state.ENERGY_LIST_FOR_PLOTTING, "approx_EQ", "min")
        FIO.argrelextrema_txt_save(grad_list, "local_min_grad", "min")

        self.logger.save_energy_profiles(self.state, self.BPA_FOLDER_DIRECTORY)

        self.state.bias_pot_params_grad_list = self.handler.bias_pot_calc.bias_pot_params_grad_list
        self.state.bias_pot_params_grad_name_list = self.handler.bias_pot_calc.bias_pot_params_grad_name_list
        self.state.final_file_directory = file_directory
        self.state.final_geometry = geom_num_list
        self.state.final_energy = e
        self.state.final_bias_energy = B_e

        if not exit_flag:
            self.symmetry = analyze_symmetry(self.element_list, self.state.final_geometry)
            self.state.symmetry = self.symmetry
            with open(self.BPA_FOLDER_DIRECTORY + "symmetry.txt", "w") as f:
                f.write(f"Symmetry of final structure: {self.symmetry}")
            print(f"Symmetry: {self.symmetry}")

        if isinstance(self.handler, ModelFunctionHandler) and self.handler.is_bitss:
            # We need original single-N element list for writing single frames
            # But state.element_list is doubled.
            # Use cached or slice.
            # The handler.finalize_bitss_trajectory needs access to single element list
            single_elem_len = len(self.state.element_list) // 2
            real_elems = self.state.element_list[:single_elem_len]
            self.config.args.element_list_cache = real_elems # Ensure correct list is used
            
          
            self.handler.finalize_bitss_trajectory()

    def _copy_final_results_from_state(self):
        if self.state:
            self.final_file_directory = self.state.final_file_directory
            self.final_geometry = self.state.final_geometry
            self.final_energy = self.state.final_energy
            self.final_bias_energy = self.state.final_bias_energy
            self.symmetry = getattr(self.state, "symmetry", None)
            self.bias_pot_params_grad_list = self.state.bias_pot_params_grad_list
            self.bias_pot_params_grad_name_list = self.state.bias_pot_params_grad_name_list
            self.optimized_flag = self.state.optimized_flag

    def geom_info_extract(self, force_data, file_directory, B_g, g):
        # kept for backward compatibility; delegate to logger
        return self.logger.geom_info_extract(self.state, force_data, file_directory, B_g, g, self.BPA_FOLDER_DIRECTORY)

    def _print_info(
        self,
        e,
        B_e,
        B_g,
        displacement_vector,
        pre_e,
        pre_B_e,
        max_displacement_threshold,
        rms_displacement_threshold,
    ):
        rms_force = self.convergence.calculate_rms_safely(np.abs(B_g))
        rms_displacement = self.convergence.calculate_rms_safely(np.abs(displacement_vector))
        max_B_g = np.abs(B_g).max()
        max_displacement = np.abs(displacement_vector).max()
        print("caluculation results (unit a.u.):")
        print("                         Value                     Threshold ")
        print("ENERGY                 : {:>15.12f} ".format(e))
        print("BIAS  ENERGY           : {:>15.12f} ".format(B_e))
        print(
            "Maximum  Force         : {0:>15.12f}                 {1:>15.12f} ".format(
                max_B_g, self.config.MAX_FORCE_THRESHOLD
            )
        )
        print(
            "RMS      Force         : {0:>15.12f}                 {1:>15.12f} ".format(
                rms_force, self.config.RMS_FORCE_THRESHOLD
            )
        )
        print(
            "Maximum  Displacement  : {0:>15.12f}                 {1:>15.12f} ".format(
                max_displacement, max_displacement_threshold
            )
        )
        print(
            "RMS      Displacement  : {0:>15.12f}                 {1:>15.12f} ".format(
                rms_displacement, rms_displacement_threshold
            )
        )
        print("ENERGY SHIFT           : {:>15.12f} ".format(e - pre_e))
        print("BIAS ENERGY SHIFT      : {:>15.12f} ".format(B_e - pre_B_e))

    def get_result_file_path(self):
        self.optimized_struct_file, self.traj_file = self.result_paths.get_result_file_path(
            self.BPA_FOLDER_DIRECTORY, self.START_FILE
        )

if __name__ == "__main__":
    # The actual CLI interface is expected to supply args.
    pass