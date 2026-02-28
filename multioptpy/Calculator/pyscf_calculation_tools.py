
import glob
import os
import numpy as np
from abc import ABC, abstractmethod

try:
    import pyscf
    from pyscf import tdscf
    from pyscf.hessian import thermo
except:
    pass

from multioptpy.Utils.calc_tools import Calculationtools
from multioptpy.Parameters.parameter import UnitValueLib
from multioptpy.fileio import xyz2list
from multioptpy.Visualization.visualization import NEBVisualizer
from multioptpy.ModelHessian.o1numhess import O1NumHessCalculator
"""
Ref.:PySCF
Recent developments in the PySCF program package, Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan, J. Chem. Phys., 153, 024109 (2020). doi:10.1063/5.0006074

"""

class Calculation:
    """
    Handles PySCF calculation logic.
    Supports both legacy directory-based execution and direct in-memory execution.
    """
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        self.hartree2eV = UVL.hartree2eV
        
        # Unpack kwargs
        self.START_FILE = kwarg.get("START_FILE", None)
        self.SUB_BASIS_SET = kwarg.get("SUB_BASIS_SET", None)
        self.ECP = kwarg.get("ECP", None)
        self.BASIS_SET = kwarg.get("BASIS_SET", "sto-3g")
        self.N_THREAD = kwarg.get("N_THREAD", 1)
        self.SET_MEMORY = kwarg.get("SET_MEMORY", "4GB")
        self.FUNCTIONAL = kwarg.get("FUNCTIONAL", "b3lyp")
        self.FC_COUNT = kwarg.get("FC_COUNT", 1)
        self.BPA_FOLDER_DIRECTORY = kwarg.get("BPA_FOLDER_DIRECTORY", "./")
        self.Model_hess = kwarg.get("Model_hess", None)
        self.electronic_charge = kwarg.get("electronic_charge", 0)
        self.spin_multiplicity = kwarg.get("spin_multiplicity", 1)
        self.unrestrict = kwarg.get("unrestrict", False)
        self.dft_grid = kwarg.get("dft_grid", 3)
        self.hessian_flag = False
        
        if kwarg.get("excited_state"):
            self.excited_state = kwarg["excited_state"] 
        else:
            self.excited_state = 0

    def run_calculation(self, positions_ang, element_list, charge_mult):
        """
        Execute PySCF calculation for a single geometry.
        
        Args:
            positions_ang (np.ndarray): Coordinates in Angstroms.
            element_list (list): List of element symbols.
            charge_mult (list): [charge, multiplicity].
            
        Returns:
            tuple: (energy, gradient, mf_object)
        """
        # Set threads
        pyscf.lib.num_threads(self.N_THREAD)
        
        # Format atoms for PySCF
        atom_data = []
        for i, el in enumerate(element_list):
            atom_data.append([el, positions_ang[i][0], positions_ang[i][1], positions_ang[i][2]])
            
        # Parse charge/mult
        charge = int(charge_mult[0])
        spin = max(int(charge_mult[1]) - 1, 0) # PySCF uses 2S, so spin=mult-1
        
        # Build Molecule
        mol = pyscf.gto.M(atom=atom_data,
                          unit='Angstrom',
                          charge=charge,
                          spin=spin,
                          basis=self.SUB_BASIS_SET,
                          ecp=self.ECP,
                          max_memory=float(self.SET_MEMORY.replace("GB","")) * 1024,
                          verbose=4)
        # Setup Mean Field (HF or DFT)
        scf_max_cycle = 500 + 5 * len(element_list)
        
        is_uhf = (spin > 0 or self.unrestrict)
        is_dft = not (self.FUNCTIONAL.lower() == "hf")

        if not is_dft: # HF
            if is_uhf:
                mf = mol.UHF().density_fit()
            else:
                mf = mol.RHF().density_fit()
        else: # DFT
            if is_uhf:
                mf = mol.UKS().x2c().density_fit()
            else:
                mf = mol.RKS().density_fit()
            mf.xc = self.FUNCTIONAL
            mf.grids.level = self.dft_grid

        # Run SCF/DFT
        mf.max_cycle = scf_max_cycle
        mf.direct_scf = True
        
        # Execute Ground State
        if self.excited_state == 0:
            mf.run()
            grad_calc = mf.nuc_grad_method()
            g = grad_calc.kernel()
            e = float(mf.e_tot)
            final_mf = mf
        else:
            # Excited State (TDA/TDDFT)
            mf.run() # Run ground state first
            ground_e = float(mf.e_tot)
            
            td_obj = tdscf.TDA(mf)
            td_obj.max_cycle = scf_max_cycle
            # Note: nuc_grad_method().kernel(state=...) returns gradient
            g = td_obj.run().nuc_grad_method().kernel(state=self.excited_state)
            
            # e_tot for excited state
            # vars(mf)["e"] is usually a list of excitation energies
            exc_energy = td_obj.e[self.excited_state-1]
            e = ground_e + exc_energy
            final_mf = td_obj # Return TD object for consistency if needed, though Hessian usually uses ground MF
            
            # Note: Analytical Hessian for TDDFT/TDA in PySCF might be limited.
            # Using ground state MF for Hessian if excited state Hessian is not supported in this script's scope
            # Recalibrating final_mf to ground state for Hessian calculation if needed, 
            # but usually one wants the Hessian of the surface being optimized.
            # The original code did 'mf.Hessian().kernel()' on the ground state MF even in excited block?
            # Re-checking original: It calls `mf.Hessian().kernel()` on `mf`. 
            # In the excited block, `mf` is reassigned to `tdscf.TDA(mf)`. 
            # PySCF TDA object *does* have a Hessian method in recent versions, or it might fall back.
            final_mf = td_obj

        g = np.array(g, dtype="float64")
        np.save(self.BPA_FOLDER_DIRECTORY+"raw_grad.npy", g)
        
        return e, g, final_mf

    def calc_exact_hess(self, mf_obj, positions_ang, element_list):
        """
        Calculate exact Hessian using the provided Mean-Field object.
        """
        # PySCF Hessian (returns Hartree/Bohr^2 usually)
        # Note: If mf_obj is TDA, ensure it supports Hessian, otherwise might need ground state
        try:
            exact_hess = mf_obj.Hessian().kernel()
        except AttributeError:
            print("Warning: Hessian not supported for this method/state. calculating ground state hessian.")
            exact_hess = mf_obj._scf.Hessian().kernel()

        # Input data for display is used in original code for reshape size. 
        # It corresponds to atom count.
        n_atoms = len(positions_ang)
        
        # Reshape: (N,N,3,3) -> (N,3,N,3) -> (3N, 3N)
        exact_hess = exact_hess.transpose(0,2,1,3).reshape(n_atoms*3, n_atoms*3)
        
        # Thermo analysis (optional, for logging)
        try:
            freqs = thermo.harmonic_analysis(mf_obj.mol, exact_hess)
            freq_wavenumber = freqs["freq_wavenumber"]
            print("frequencies: \n", freq_wavenumber)
        except:
            freq_wavenumber = []

        # Project out translation and rotation
        # Note: Calculationtools expects positions in Bohr usually? 
        # Re-checking original: `input_data_for_display` was passed. 
        # In original `single_point`, `input_data_for_display` was `geom / bohr2ang` (Bohr).
        # So we must pass Bohr coords to projection tool.
        positions_bohr = positions_ang / self.bohr2angstroms
        np.save(self.BPA_FOLDER_DIRECTORY+"raw_hessian.npy", exact_hess)
        exact_hess = Calculationtools().project_out_hess_tr_and_rot_for_coord(
            exact_hess, element_list, positions_bohr, display_eigval=True
        )
        
        self.Model_hess = exact_hess
        return exact_hess, freq_wavenumber

    def single_point(self, file_directory, element_list, iter, electric_charge_and_multiplicity="", method="", geom_num_list=None):
        """
        Legacy method for directory-based execution.
        """
        finish_frag = False
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
        input_data_for_display = np.zeros(1)

        for num, input_file in enumerate(file_list):
            try:
                if geom_num_list is not None:
                    geom_ang = np.array(geom_num_list, dtype="float64")
                    # charge/mult logic for manual input needs to be handled if not passed explicitly
                    # assuming electric_charge_and_multiplicity is passed or defaults
                    if not electric_charge_and_multiplicity:
                        electric_charge_and_multiplicity = [self.electronic_charge, self.spin_multiplicity]
                else:
                    geom_ang, _, electric_charge_and_multiplicity = xyz2list(input_file, electric_charge_and_multiplicity)
                    geom_ang = np.array(geom_ang, dtype="float64")

                # Parse charge/mult for run_calculation
                # xyz2list returns list [charge, mult] or similar
                # Ensure it's in the [charge, mult] format
                if isinstance(electric_charge_and_multiplicity, str):
                    # Handle string case if needed, though usually list
                    pass 

                # Execute using the new method
                e, g, mf = self.run_calculation(geom_ang, element_list, electric_charge_and_multiplicity)
                
                # Coordinate for display/output (Bohr)
                input_data_for_display = geom_ang / self.bohr2angstroms

                if self.FC_COUNT == -1 or type(iter) is str:
                    if self.hessian_flag:
                        self.calc_exact_hess(mf, geom_ang, element_list)
        
                elif iter % self.FC_COUNT == 0 or self.hessian_flag:
                    self.calc_exact_hess(mf, geom_ang, element_list)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return np.array([0]), np.array([0]), input_data_for_display, finish_frag
             
        self.energy = e
        self.gradient = g
        self.coordinate = input_data_for_display
        
        return e, g, input_data_for_display, finish_frag

    def exact_hessian(self, element_list, input_data_for_display, mf):
        """Legacy wrapper for calc_exact_hess to maintain exact signature match if needed internally."""
        # Convert input_data_for_display (Bohr) back to Angstrom for the new API if needed, 
        # but calc_exact_hess handles the projection conversion.
        # Actually, let's just delegate.
        positions_ang = input_data_for_display * self.bohr2angstroms
        self.calc_exact_hess(mf, positions_ang, element_list)


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


class PySCFEngine(CalculationEngine):
    """PySCF calculation engine"""
    
    def calculate(self, file_directory, optimize_num, pre_total_velocity, config):
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        
        os.makedirs(file_directory, exist_ok=True)
        file_list = self._get_file_list(file_directory)
        
        # Instantiate Calculation Class once
        calc_instance = Calculation(
            START_FILE=config.init_input, # Mapping config to kwargs
            SUB_BASIS_SET=config.SUB_BASIS_SET,
            ECP=config.ECP,
            BASIS_SET=config.BASIS_SET,
            N_THREAD=config.N_THREAD,
            SET_MEMORY=config.SET_MEMORY,
            FUNCTIONAL=config.FUNCTIONAL,
            FC_COUNT=config.FC_COUNT,
            BPA_FOLDER_DIRECTORY=config.NEB_FOLDER_DIRECTORY, # Assuming this maps
            Model_hess=config.model_hessian,
            electronic_charge=config.electronic_charge,
            spin_multiplicity=config.spin_multiplicity,
            unrestrict=config.unrestrict,
            dft_grid=config.dft_grid,
            excited_state=config.excited_state
        )

        hess_count = 0
        
        for num, input_file in enumerate(file_list):
            try:
                print(f"Processing: {input_file}")
                # Parse XYZ
                positions, element_list, electric_charge_and_multiplicity = xyz2list(input_file, None)
                positions_ang = np.array(positions, dtype="float64")
                
                # --- Execute Calculation via Instance ---
                e, g, mf = calc_instance.run_calculation(
                    positions_ang, 
                    element_list, 
                    electric_charge_and_multiplicity
                )
                # ----------------------------------------

                g = np.array(g, dtype="float64")
                
                # Store results
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))
                
                # Convert to Bohr for geometry_num_list output
                input_data_for_display = positions_ang / config.bohr2angstroms
                geometry_num_list.append(input_data_for_display)
                num_list.append(num)
                
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
                        mf, positions_ang, element_list
                    )
                 
                    # Save results
                    np.save(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".npy", exact_hess)
                    with open(config.NEB_FOLDER_DIRECTORY + "tmp_hessian_" + str(hess_count) + ".csv", "a") as f:
                        f.write("frequency," + ",".join(map(str, freqs)) + "\n")
                    hess_count += 1
                
            except Exception as error:
                print(f"Error in {input_file}: {error}")
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        self._process_visualization(energy_list, gradient_list, num_list, optimize_num, config)

        if optimize_num != 0 and len(pre_total_velocity) != 0:
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return (np.array(energy_list, dtype="float64"), 
                np.array(gradient_list, dtype="float64"), 
                np.array(geometry_num_list, dtype="float64"), 
                pre_total_velocity)

