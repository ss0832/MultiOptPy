import glob
import os
import numpy as np
from abc import ABC, abstractmethod

try:
    import psi4
except ImportError:
    print("Psi4 is not available.")
    psi4 = None  # Set to None to handle gracefully in code

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
from multioptpy.ModelHessian.o1numhess import O1NumHessCalculator

"""
Psi4
 D. G. A. Smith, et al., "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).
"""

class Calculation:
    """
    Handles Psi4 calculation logic.
    Supports both legacy directory-based execution and direct in-memory execution.
    """
    def __init__(self, **kwarg):
        
        self.START_FILE = kwarg.get("START_FILE", None)
        self.SUB_BASIS_SET = kwarg.get("SUB_BASIS_SET", {}) # Default to empty dict if not provided
        self.BASIS_SET = kwarg.get("BASIS_SET", "sto-3g")
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY", "2GB")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL", "b3lyp")
        self.FC_COUNT = kwarg.get("FC_COUNT", 1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", "./")
        self.Model_hess = kwarg.get("Model_hess", None)
        self.unrestrict = kwarg.get("unrestrict", False)
        self.dft_grid = kwarg.get("dft_grid", 3)
        self.hessian_flag = False
        
        if kwarg.get("excited_state"):
            self.excited_state = kwarg["excited_state"]
        else:
            self.excited_state = 0

    def set_dft_grid(self):
        """set dft grid"""
        if self.dft_grid == 0 or self.dft_grid == 1:
            psi4.set_options({'DFT_RADIAL_POINTS': 50, 'DFT_SPHERICAL_POINTS': 194})
            # print("DFT Grid (50, 194): SG1") # Suppress print for cleaner loop output
        elif self.dft_grid == 2 or self.dft_grid == 3:
            psi4.set_options({'DFT_RADIAL_POINTS': 75, 'DFT_SPHERICAL_POINTS': 302})
            # print("DFT Grid (70, 302): Default")
        elif self.dft_grid == 4 or self.dft_grid == 5:
            psi4.set_options({'DFT_RADIAL_POINTS': 99, 'DFT_SPHERICAL_POINTS': 590})
            # print("DFT Grid (99, 590): Fine")
        elif self.dft_grid == 6 or self.dft_grid == 7:
            psi4.set_options({'DFT_RADIAL_POINTS': 150, 'DFT_SPHERICAL_POINTS': 770})
            # print("DFT Grid (150, 770): UltraFine")
        elif self.dft_grid == 8 or self.dft_grid == 9:
            psi4.set_options({'DFT_RADIAL_POINTS': 250, 'DFT_SPHERICAL_POINTS': 974})
            # print("DFT Grid (250, 974): SuperFine")
        else:
            # raise ValueError("Invalid dft grid setting.") # Or just warn and use default
            pass

    def _setup_psi4_options(self, charge_mult, file_directory=None):
        """Internal helper to set common Psi4 options"""
        psi4.core.clean()
        psi4.set_num_threads(nthread=self.N_THREAD)
        psi4.set_memory(self.SET_MEMORY)
        
        charge = int(charge_mult[0])
        mult = int(charge_mult[1])
        
        if mult > 1 or self.unrestrict:
            psi4.set_options({'reference': 'uks'})
        else:
            psi4.set_options({'reference': 'rks'}) # Explicitly set rks for closed shell if not unrestrict

        psi4.set_options({"MAXITER": 500})
        self.set_dft_grid()
        
        if len(self.SUB_BASIS_SET) > 0:
            psi4.basis_helper(self.SUB_BASIS_SET, name='User_Basis_Set', set_option=False)
            psi4.set_options({"basis":'User_Basis_Set'})
        else:
            psi4.set_options({"basis":self.BASIS_SET})
            
        if self.excited_state > 0:
            psi4.set_options({'TDSCF_STATES': self.excited_state})
            
        if file_directory:
             psi4.set_options({"cubeprop_tasks": ["esp"], 'cubeprop_filepath': file_directory})

    def run_calculation(self, positions_ang, element_list, charge_mult, logfile_path=None):
        """
        Execute Psi4 calculation for a single geometry.
        
        Args:
            positions_ang (np.ndarray): Coordinates in Angstroms.
            element_list (list): List of element symbols.
            charge_mult (list): [charge, multiplicity].
            logfile_path (str): Path to output log file.
            
        Returns:
            tuple: (energy, gradient, wfn)
        """
        if logfile_path:
             psi4.set_output_file(logfile_path)
        else:
             psi4.core.set_output_file("psi4_output.dat", True) # Redirect to file or suppress

        self._setup_psi4_options(charge_mult)
        
        # Build Geometry String
        geom_str = f"{charge_mult[0]} {charge_mult[1]}\n"
        for i, el in enumerate(element_list):
            geom_str += f"{el} {positions_ang[i][0]:.10f} {positions_ang[i][1]:.10f} {positions_ang[i][2]:.10f}\n"
        geom_str += "units angstrom\n"
        
        mol = psi4.geometry(geom_str)
        
        # Execute Gradient (and Energy)
        # Note: gradient() returns (ndarray, wfn)
        g, wfn = psi4.gradient(self.FUNCTIONAL, molecule=mol, return_wfn=True)
        e = float(wfn.energy())
        g = np.array(g, dtype="float64")
        
        return e, g, wfn

    def calc_exact_hess(self, wfn, positions_ang, element_list):
        """
        Calculate exact Hessian using the provided Wavefunction object.
        """
        # Frequencies calculation also computes Hessian
        # ref_gradient is needed for numerical hessian sometimes, or just efficiency
        _, wfn = psi4.frequencies(self.FUNCTIONAL, return_wfn=True, ref_gradient=wfn.gradient())
        exact_hess = np.array(wfn.hessian()) # Hessian in atomic units (Hartree/Bohr^2) usually
        
        freqs = np.array(wfn.frequencies())
        # print("frequencies: \n", freqs) # Optional logging
        
        # Input for projection tool needs to be Bohr usually if Hessian is AU?
        # Original code passed `input_data_for_display` which was `geom_ang / bohr2ang` (Bohr).
        # So we convert Ang -> Bohr here.
        positions_bohr = positions_ang / psi4.constants.bohr2angstroms
        np.save(self.BPA_FOLDER_DIRECTORY+"raw_hessian.npy", exact_hess)
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            exact_hess, element_list, positions_bohr, display_eigval=False
        )
        
        self.Model_hess = exact_hess
        return exact_hess, freqs

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity, method="", geom_num_list=None):
        """
        Legacy method for directory-based execution.
        """
        finish_frag = False
        input_data_for_display = None
        
        try:
            os.mkdir(file_directory)
        except:
            pass

        if file_directory is None:
            file_list = ["dummy"]
        else:
            file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        
        e = 0.0
        g = np.zeros(1)
        
        for num, input_file in enumerate(file_list):
            try:
                logfile = file_directory+"/"+self.START_FILE[:-4]+'_'+str(num)+'.log'
                
                if geom_num_list is None:
                    pos_ang, element_list, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                    positions_ang = np.array(pos_ang, dtype="float64")
                else:
                    # geom_num_list is assumed to be Angstrom based on original usage logic in Psi4Engine?
                    # Wait, in original `single_point`: `input_data_for_display = geom_num_list / bohr2ang`.
                    # This implies geom_num_list was Angstrom.
                    positions_ang = np.array(geom_num_list, dtype="float64")
                    # Element list and charge needs to be consistent
                
                # Execute using new method
                e, g, wfn = self.run_calculation(
                    positions_ang, 
                    element_list, 
                    electric_charge_and_multiplicity, 
                    logfile_path=logfile
                )
                
                # Coordinate for display (Bohr)
                input_data_for_display = positions_ang / psi4.constants.bohr2angstroms
                
                # Optional: Properties (Dipole, Charges etc.)
                # If these are critical for NEB, they should be moved to run_calculation or kept here if side-effect only.
                # Keeping minimal for now.
                
                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.calc_exact_hess(wfn, positions_ang, element_list)
                
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.calc_exact_hess(wfn, positions_ang, element_list)
                
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), input_data_for_display, finish_frag 
                
            psi4.core.clean() 
            
        self.energy = e
        self.gradient = g
        if input_data_for_display is not None:
             self.coordinate = input_data_for_display
        else:
             self.coordinate = positions_ang / psi4.constants.bohr2angstroms # Fallback

        return e, g, self.coordinate, finish_frag


class CalculationEngine(ABC):
    """Base class for calculation engines"""
    
    @abstractmethod
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        """Calculate energy and gradients"""
        pass
    
    def _get_file_list(self, file_directory):
        """Get list of input files"""
        return sum([sorted(glob.glob(os.path.join(file_directory, f"*_" + "[0-9]" * i + ".xyz"))) 
                   for i in range(1, 7)], [])
    
    def _process_visualization(self, energy_list, gradient_list, num_list, optimize_num, config):
        """Process common visualization tasks"""
        try:
            if config.save_pict:
                visualizer = NEBVisualizer(config)
                tmp_ene_list = np.array(energy_list, dtype="float64") * config.hartree2kcalmol
                visualizer.plot_energy(num_list, tmp_ene_list - tmp_ene_list[0], optimize_num)
                print("energy graph plotted.")
                
                gradient_norm_list = [np.sqrt(np.linalg.norm(g)**2/(len(g)*3)) for g in gradient_list]
                visualizer.plot_gradient(num_list, gradient_norm_list, optimize_num)
                print("gradient graph plotted.")
        except Exception as e:
            print(f"Visualization error: {e}")


class Psi4Engine(CalculationEngine):
    """Psi4 calculation engine"""
    
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        if 'psi4' not in globals() and 'psi4' not in locals(): # Check import
             try:
                 import psi4
             except ImportError:
                 raise ImportError("Psi4 is not available")
        
        psi4.core.clean()
        gradient_list = []
        gradient_norm_list = []
        energy_list = []
        geometry_num_list = []
        num_list = []
        delete_pre_total_velocity = []
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        # Instantiate Calculation Class
        calc_instance = Calculation(
            START_FILE=config.init_input,
            SUB_BASIS_SET=config.SUB_BASIS_SET if hasattr(config, 'SUB_BASIS_SET') else {},
            BASIS_SET=config.basisset,
            N_THREAD=config.N_THREAD,
            SET_MEMORY=config.SET_MEMORY,
            FUNCTIONAL=config.functional, # config.functional vs FUNCTIONAL in kwargs
            FC_COUNT=config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY,
            Model_hess=config.model_hessian,
            unrestrict=config.unrestrict,
            dft_grid=config.dft_grid,
            excited_state=config.excited_state
        )
        
        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
                print(input_file)
                logfile = file_directory + "/" + config.init_input + '_' + str(num) + '.log'
                
                # Load Geometry
                pos_ang, element_list, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions_ang = np.array(pos_ang, dtype="float64")
                
                # --- Execute Calculation ---
                e, g, wfn = calc_instance.run_calculation(
                    positions_ang, 
                    element_list, 
                    electric_charge_and_multiplicity,
                    logfile_path=logfile
                )
                # ---------------------------

                print('energy:' + str(e) + " a.u.")

                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                energy_list.append(e)
                num_list.append(num)
                
                # Convert to Bohr for storage
                input_data_for_display = positions_ang / psi4.constants.bohr2angstroms
                geometry_num_list.append(input_data_for_display)
                
                # Hessian Calculation
                if config.MFC_COUNT != -1 and optimize_num % config.MFC_COUNT == 0 and config.model_hessian.lower() == "o1numhess":
                    print(f" Calculating O1NumHess for image {num} using {config.model_hessian}...")
                    o1numhess = O1NumHessCalculator(calc_instance, 
                        element_list, 
                        electric_charge_and_multiplicity,
                        method="")
                    seminumericalhessian = o1numhess.compute_hessian(positions_ang)
                    np.save(os.path.join(config.NEB_FOLDER_DIRECTORY, f"tmp_hessian_{hess_count}.npy"), seminumericalhessian)
                    hess_count += 1
                
                elif config.FC_COUNT == -1 or type(optimize_num) is str:
                    pass
                elif optimize_num % config.FC_COUNT == 0:
                    print(f"  Calculating Hessian for image {num}...")
                    
                    exact_hess, freqs = calc_instance.calc_exact_hess(
                        wfn, positions_ang, element_list
                    )
                    
                    np.save(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".npy", exact_hess)
                    with open(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".csv", "a") as f:
                        f.write("frequency," + ",".join(map(str, freqs)) + "\n")
                
                hess_count += 1    

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
                
            psi4.core.clean()
        
        print("data sampling was completed...")
        
        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)
        
        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64").tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return (np.array(energy_list, dtype="float64"), 
                np.array(gradient_list, dtype="float64"), 
                np.array(geometry_num_list, dtype="float64"), 
                pre_total_velocity)

