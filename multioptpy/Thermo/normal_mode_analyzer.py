import os
import numpy as np
import sys

from multioptpy.Utils.symmetry_analyzer import SymmetryAnalyzer
from multioptpy.Parameters.parameter import atomic_mass, UnitValueLib
from multioptpy.Utils.calc_tools import Calculationtools

# Physical constants
HARTREE_TO_J = UnitValueLib().hartree2j
AVOGADRO = UnitValueLib().mol2au
KB = UnitValueLib().boltzmann_constant
PLANCK = UnitValueLib().planck_constant
BOHR = UnitValueLib().bohr2m
ATOMIC_MASS = UnitValueLib().amu2kg
LIGHT_SPEED = UnitValueLib().vacume_light_speed
LINDEP_THRESHOLD = 1e-7
BOHR2ANGSTROM = UnitValueLib().bohr2angstroms

def format_number_sequence(values, start_idx, end_idx):
    """Format a sequence of numbers for aligned output.

    Parameters
    ----------
    values : array_like
        Array of values to format
    start_idx : int
        Starting index
    end_idx : int
        Ending index

    Returns
    -------
    str
        Formatted string of values
    """
    return ''.join('%20.4f' % values[i] for i in range(start_idx, end_idx))


def format_mode_vectors(mode_array, atom_idx, start_idx, end_idx):
    """Format a row of normal mode vectors for output.

    Parameters
    ----------
    mode_array : array_like
        3D array of mode vectors
    atom_idx : int
        Index of the atom
    start_idx : int
        Starting mode index
    end_idx : int
        Ending mode index

    Returns
    -------
    str
        Formatted string of mode vectors
    """
    return '  '.join('%9.5f  %9.5f  %9.5f' % (mode_array[i, atom_idx, 0], 
                                          mode_array[i, atom_idx, 1], 
                                          mode_array[i, atom_idx, 2])
                     for i in range(start_idx, end_idx))


def output_to_console_and_file(text, console_output=sys.stdout, file_path=None):
    """Output text to both console and file if specified.

    Parameters
    ----------
    text : str
        Text to output
    console_output : file object
        Console output stream (defaults to sys.stdout)
    file_path : str or None
        Path to output file (if None, file output is skipped)
    """
    print(text, file=console_output)
    if file_path is not None:
        with open(file_path, 'a') as f:
            f.write(text + '\n')


def convert_energy_units(results, property_prefix, component_keys, unit):
    """Convert thermochemistry results for output.

    Parameters
    ----------
    results : dict
        Results dictionary
    property_prefix : str
        Property prefix (e.g., 'S', 'H', 'G')
    component_keys : tuple
        Component keys (e.g., 'elec', 'trans', 'rot', 'vib')
    unit : str
        Current unit

    Returns
    -------
    str
        Formatted string with converted values
    """
    # Only use atomic units (no conversion)
    return ' '.join('%20.10f' % (results.get(f"{property_prefix}_{key}", (0,))[0]) 
                   for key in component_keys)


def output_thermodynamic_property(results, title, property_prefix, component_keys, 
                                 console_output, file_path=None):
    """Write a full line of thermochemistry output.

    Parameters
    ----------
    results : dict
        Results dictionary
    title : str
        Property title
    property_prefix : str
        Property prefix (e.g., 'S', 'H', 'G')
    component_keys : tuple
        Component keys (e.g., 'elec', 'trans', 'rot', 'vib')
    console_output : file object
        Console output stream
    file_path : str or None
        Path to output file (if None, file output is skipped)
    """
    total_value, unit = results[f"{property_prefix}_tot"]
    formatted_values = convert_energy_units(results, property_prefix, component_keys, unit)
    # Always display in atomic units (Eh)
    output_line = '%-20s %s' % (f"{title} [{unit}]", formatted_values)
    output_to_console_and_file(output_line, console_output, file_path)


class MolecularVibrations:
    """
    Comprehensive analyzer for molecular vibrations, thermochemistry, and vibrational animations.
    
    This class integrates normal mode analysis, thermochemistry calculations, and
    visualization of vibrational modes for molecular systems.
    """

    def __init__(self, atoms, coordinates, hessian, symm_tolerance=1e-4, max_symm_fold=6):
        """
        Parameters
        ----------
        atoms : list of str
            List of atomic symbols.
        coordinates : np.ndarray
            Atomic coordinates (n_atoms, 3).
        hessian : np.ndarray
            Hessian matrix (3*n_atoms, 3*n_atoms) in atomic units.
        symm_tolerance : float
            Distance tolerance for symmetry operations.
        max_symm_fold : int
            Maximum n-fold rotation to check.
        """
        self.atoms = atoms
        self.coordinates = np.array(coordinates)
        self.hessian = np.array(hessian)
        self.n_atoms = len(atoms)
        self.symm_tolerance = symm_tolerance
        self.max_symm_fold = max_symm_fold

        # Initialize symmetry analyzer
        self.symmetry_analyzer = SymmetryAnalyzer(atoms, coordinates, tol=symm_tolerance, max_n_fold=max_symm_fold)
        self.point_group = self.symmetry_analyzer.analyze()

        # Atomic masses in atomic mass units
        self.mass = np.array([self._get_atomic_mass(atom) for atom in atoms])

        # Center the molecule at its center of mass
        self.com = np.einsum('z,zx->x', self.mass, self.coordinates) / self.mass.sum()
        self.coordinates -= self.com

        # Analysis results will be stored here
        self.results = {}

    def _get_atomic_mass(self, atom):
        """Returns atomic mass for a given element symbol."""
        atomic_weights = atomic_mass(atom)
        return atomic_weights


    def analyze_normal_modes(self, exclude_trans_and_rot=True, imaginary_freq=True):
        """
        Perform normal mode analysis.

        Parameters
        ----------
        exclude_trans_and_rot : bool
            Whether to exclude translational and rotational modes.
        imaginary_freq : bool
            Whether to represent imaginary frequencies as complex numbers.

        Returns
        -------
        dict
            Results of normal mode analysis.
        """
        results = {}

        if exclude_trans_and_rot:
            # Project out translation and rotation from Hessian
            h = Calculationtools().project_out_hess_tr_and_rot(
                self.hessian, 
                self.atoms, 
                self.coordinates, 
                display_eigval=False
            )
        else:
            # Just mass-weight the Hessian without projection
            mass_vec = np.repeat(self.mass, 3) ** -0.5
            h = self.hessian * np.outer(mass_vec, mass_vec)

        # Diagonalize the projected Hessian to get eigenvalues and eigenvectors
        force_const_au, mode = np.linalg.eigh(h)

        freq_au = np.lib.scimath.sqrt(force_const_au)
        results['freq_error'] = np.count_nonzero(freq_au.imag > 0)
        if not imaginary_freq and np.iscomplexobj(freq_au):
            freq_au = freq_au.real - np.abs(freq_au.imag)

        results['freq_au'] = freq_au
        au2hz = (HARTREE_TO_J / (ATOMIC_MASS * BOHR ** 2)) ** 0.5 / (2 * np.pi)
        results['freq_wavenumber'] = freq_au * au2hz / LIGHT_SPEED * 1e-2

        # Reshape mode vectors to (n_modes, n_atoms, 3)
        mode_reshape = mode.T.reshape(-1, self.n_atoms, 3)
        
        # Mass-weight the mode vectors - vectorized version
        mass_sqrt_inv = 1.0 / np.sqrt(self.mass).reshape(1, -1, 1)  # Shape (1, n_atoms, 1)
        norm_mode = mode_reshape * mass_sqrt_inv  # Broadcasting applies to all modes and coordinates
        
        results['norm_mode'] = norm_mode
        
        # Calculate reduced mass - vectorized version
        reduced_mass = 1.0 / np.sum(np.sum(norm_mode * norm_mode, axis=2), axis=1)
        results['reduced_mass'] = reduced_mass

        # Vibrational temperature
        results['vib_temperature'] = freq_au * au2hz * PLANCK / KB

        # Force constants
        dyne = 1e-2 * HARTREE_TO_J / BOHR ** 2
        results['force_const_au'] = force_const_au
        results['force_const_dyne'] = results['reduced_mass'] * force_const_au * dyne

        self.results.update(results)
        return results

    def calculate_thermochemistry(self, e_tot=0.0, temperature=298.15, pressure=101325):
        """
        Calculate thermochemical properties.

        Parameters
        ----------
        e_tot : float
            Total electronic energy in Hartree.
        temperature : float
            Temperature in Kelvin.
        pressure : float
            Pressure in Pascal.

        Returns
        -------
        dict
            Thermochemistry results.
        """
        if 'freq_au' not in self.results:
            self.analyze_normal_modes()

        results = {}
        R_Eh = KB * AVOGADRO / (HARTREE_TO_J * AVOGADRO)

        results['temperature'] = (temperature, 'K')
        results['pressure'] = (pressure, 'Pa')
        results['E0'] = (e_tot, 'Eh')

        multiplicity = 1  # This could be a parameter
        results['S_elec'] = (R_Eh * np.log(multiplicity), 'Eh/K')
        results['Cv_elec'] = results['Cp_elec'] = (0, 'Eh/K')
        results['E_elec'] = results['H_elec'] = (e_tot, 'Eh')

        total_mass = self.mass.sum() * ATOMIC_MASS
        q_trans = ((2.0 * np.pi * total_mass * KB * temperature / PLANCK ** 2) ** 1.5
                   * KB * temperature / pressure)
        results['S_trans'] = (R_Eh * (2.5 + np.log(q_trans)), 'Eh/K')
        results['Cv_trans'] = (1.5 * R_Eh, 'Eh/K')
        results['Cp_trans'] = (2.5 * R_Eh, 'Eh/K')
        results['E_trans'] = (1.5 * R_Eh * temperature, 'Eh')
        results['H_trans'] = (2.5 * R_Eh * temperature, 'Eh')

        rot_const = self.get_rotational_constants('GHz')
        results['rot_const'] = (rot_const, 'GHz')
        rotor_type = self._get_rotor_type(rot_const)

        sym_number = self.get_rotational_symmetry_number()
        results['sym_number'] = (sym_number, '')

        if rotor_type == 'ATOM':
            results['S_rot'] = (0, 'Eh/K')
            results['Cv_rot'] = results['Cp_rot'] = (0, 'Eh/K')
            results['E_rot'] = results['H_rot'] = (0, 'Eh')
        elif rotor_type == 'LINEAR':
            B = rot_const[1] * 1e9
            q_rot = KB * temperature / (sym_number * PLANCK * B)
            results['S_rot'] = (R_Eh * (1 + np.log(q_rot)), 'Eh/K')
            results['Cv_rot'] = results['Cp_rot'] = (R_Eh, 'Eh/K')
            results['E_rot'] = results['H_rot'] = (R_Eh * temperature, 'Eh')
        else:
            ABC = rot_const * 1e9
            q_rot = ((KB * temperature / PLANCK) ** 1.5 * np.pi ** .5
                     / (sym_number * np.prod(ABC) ** .5))
            results['S_rot'] = (R_Eh * (1.5 + np.log(q_rot)), 'Eh/K')
            results['Cv_rot'] = results['Cp_rot'] = (1.5 * R_Eh, 'Eh/K')
            results['E_rot'] = results['H_rot'] = (1.5 * R_Eh * temperature, 'Eh')

        freq_au = self.results['freq_au']
        au2hz = (HARTREE_TO_J / (ATOMIC_MASS * BOHR ** 2)) ** 0.5 / (2 * np.pi)
        # Use np.where to safely filter positive frequencies
        pos_idx = np.where(freq_au.real > 0)[0]
        vib_temperature = freq_au.real[pos_idx] * au2hz * PLANCK / KB
        rt = vib_temperature / max(1e-14, temperature)
        exp_neg_rt = np.exp(-rt)

        ZPE = R_Eh * 0.5 * vib_temperature.sum()
        
        results['ZPE'] = (ZPE, 'Eh')
        
        tmp_denom = 1 - exp_neg_rt
        mask = np.abs(tmp_denom) < 1e-10
        tmp_denom[mask] = 1e-10
        
      
        results['S_vib'] = (R_Eh * (rt * exp_neg_rt / tmp_denom - np.log(tmp_denom)).sum(), 'Eh/K')
        results['Cv_vib'] = results['Cp_vib'] = (R_Eh * (exp_neg_rt * rt ** 2 / tmp_denom ** 2).sum(), 'Eh/K')
        results['E_vib'] = results['H_vib'] = (
            ZPE + R_Eh * temperature * (rt * exp_neg_rt / tmp_denom).sum(), 'Eh')

        results['G_elec'] = (results['H_elec'][0] - temperature * results['S_elec'][0], 'Eh')
        results['G_trans'] = (results['H_trans'][0] - temperature * results['S_trans'][0], 'Eh')
        results['G_rot'] = (results['H_rot'][0] - temperature * results['S_rot'][0], 'Eh')
        results['G_vib'] = (results['H_vib'][0] - temperature * results['S_vib'][0], 'Eh')

        # Calculate total thermodynamic properties
        keys = ('elec', 'trans', 'rot', 'vib')
        for prop in ['S', 'Cv', 'Cp', 'E', 'H', 'G']:
            results[f'{prop}_tot'] = (
                sum(results.get(f"{prop}_{key}", (0,))[0] for key in keys),
                'Eh' if prop in ['E', 'H', 'G'] else 'Eh/K'
            )

        results['E_0K'] = (e_tot + ZPE, 'Eh')

        self.results.update(results)
        return results

    def get_rotational_constants(self, unit='GHz'):
        """
        Calculate rotational constants.

        Parameters
        ----------
        unit : str
            Unit for rotational constants ('GHz' or 'wavenumber').

        Returns
        -------
        np.ndarray
            Rotational constants.
        """
        r = self.coordinates - self.com
        inertia_tensor = np.einsum('z,zr,zs->rs', self.mass, r, r)
        inertia_tensor = np.eye(3) * inertia_tensor.trace() - inertia_tensor
        eigvals = np.sort(np.linalg.eigvalsh(inertia_tensor))

        unit_inertia = ATOMIC_MASS * BOHR ** 2
        unit_hz = PLANCK / (4 * np.pi * unit_inertia)

        with np.errstate(divide='ignore'):
            if unit.lower() == 'ghz':
                eigvals = unit_hz / eigvals * 1e-9
            elif unit.lower() == 'wavenumber':
                eigvals = unit_hz / eigvals / LIGHT_SPEED * 1e-2
            else:
                raise ValueError(f"Unsupported unit {unit}")
        return eigvals

    def _get_rotor_type(self, rot_const):
        """Determine the rotor type from rotational constants."""
        if np.all(rot_const > 1e8):
            rotor_type = 'ATOM'
        elif rot_const[0] > 1e8 and (rot_const[1] - rot_const[2] < 1e-3):
            rotor_type = 'LINEAR'
        else:
            rotor_type = 'REGULAR'
        return rotor_type

    def get_rotational_symmetry_number(self):
        """
        Determine the rotational symmetry number based on the point group.

        Returns
        -------
        int
            Symmetry number.
        """
        group = self.point_group

        if group == 'C∞v':
            sigma = 1
        elif group == 'D∞h':
            sigma = 2
        elif group in ['T', 'Td']:
            sigma = 12
        elif group == 'Oh':
            sigma = 24
        elif group == 'Ih':
            sigma = 60
        elif group.startswith('C'):
            n = ''.join(filter(str.isdigit, group))
            sigma = int(n) if n else 1
        elif group.startswith('D'):
            n = ''.join(filter(str.isdigit, group))
            sigma = 2 * int(n) if n else 2
        elif group.startswith('S'):
            n = ''.join(filter(str.isdigit, group))
            sigma = int(n) // 2 if n else 1
        elif group in ['C1', 'Ci', 'Cs']:
            sigma = 1
        else:
            sigma = 1
        return sigma

    def print_normal_modes(self, output_stream=sys.stdout, output_file=None, include_imag=True, cutoff_freq=0.1):
        """
        Print normal mode information to both console and file (if output_file is given).

        Parameters
        ----------
        output_stream : file object
            Console output stream (defaults to sys.stdout).
        output_file : str or None
            Path to the file to append the output (if None, file output is skipped).
        include_imag : bool
            Whether to include imaginary frequencies in the output.
        cutoff_freq : float
            Cutoff frequency in cm^-1. Modes with absolute frequency below this value are considered 
            translational/rotational and will be excluded.
        """
        # Check if key exists instead of evaluating the array
        if 'freq_wavenumber' not in self.results:
            self.analyze_normal_modes()

        freq_wn = self.results['freq_wavenumber']
        
        # Filter out modes based on cutoff frequency (exclude small values that are likely translational/rotational)
        if include_imag:
            # Include both real and imaginary frequencies, but filter out near-zero values
            idx = np.where((np.abs(freq_wn.real) > cutoff_freq) | (freq_wn.imag > cutoff_freq))[0]
        else:
            # Only positive real frequencies above cutoff
            idx = np.where(freq_wn.real > cutoff_freq)[0]
        
        # Sort modes: real frequencies first in ascending order, then imaginary in descending magnitude
        sort_idx = np.argsort(freq_wn[idx].real)
        idx = idx[sort_idx]
        
        freq_wn_filtered = freq_wn[idx]
        nfreq = len(idx)
        
        # Filter other arrays based on selected frequencies
        r_mass = self.results['reduced_mass'][idx]
        force = self.results['force_const_dyne'][idx]
        vib_t = self.results['vib_temperature'][idx]
        mode = self.results['norm_mode'][idx]
        
        # Check if any frequencies are imaginary
        is_imag = np.zeros(nfreq, dtype=bool)
        if include_imag:
            is_imag = freq_wn_filtered.imag > 0
        
        for col0, col1 in self._chunk_iterator(0, nfreq, 3):
            header = 'Mode              %s' % ''.join('%20d' % i for i in range(col0, col1))
            output_to_console_and_file(header, output_stream, output_file)
            
            freq_line = 'Freq [cm^-1]          '
            for i in range(col0, col1):
                if i < nfreq:
                    if is_imag[i]:
                        # Imaginary frequency
                        freq_value = -np.abs(freq_wn_filtered[i])
                        freq_line += f'{freq_value.real:20.4f}'
                    else:
                        # Real frequency
                        freq_line += f'{freq_wn_filtered[i].real:20.4f}'
                else:
                    freq_line += ' ' * 20
            
            output_to_console_and_file(freq_line, output_stream, output_file)
            
            # For imaginary frequencies, some values might not be physical
            line = 'Reduced mass [au]     %s' % format_number_sequence(r_mass.real, col0, col1)
            output_to_console_and_file(line, output_stream, output_file)
            
            line = 'Force const [Dyne/A]  %s' % format_number_sequence(force.real, col0, col1)
            output_to_console_and_file(line, output_stream, output_file)
            
            line = 'Char temp [K]         %s' % format_number_sequence(vib_t.real, col0, col1)
            output_to_console_and_file(line, output_stream, output_file)
            
            line = 'Normal mode            %s' % ('       x         y         z     ' * (col1 - col0))
            output_to_console_and_file(line, output_stream, output_file)
            
            for j, atom in enumerate(self.atoms):
                line = '    %4s               %s' % (atom, format_mode_vectors(mode.real, j, col0, col1))
                output_to_console_and_file(line, output_stream, output_file)
                
            output_to_console_and_file('', output_stream, output_file)
            
    def print_thermochemistry(self, output_stream=sys.stdout, output_file=None):
        """
        Print thermochemistry information to both console and file (if output_file is given).

        Parameters
        ----------
        output_stream : file object
            Console output stream (defaults to sys.stdout).
        output_file : str or None
            Path to the file to append the output (if None, file output is skipped).
        """
        # Changed: Check if key exists instead of evaluating the array
        if 'S_tot' not in self.results:
            self.calculate_thermochemistry()

        results = self.results
        keys = ('tot', 'elec', 'trans', 'rot', 'vib')

        output_to_console_and_file('Point group: %s' % self.point_group, output_stream, output_file)
        output_to_console_and_file('Temperature %.4f [%s]' % results['temperature'], output_stream, output_file)
        output_to_console_and_file('Pressure %.2f [%s]' % results['pressure'], output_stream, output_file)
        output_to_console_and_file('Rotational constants [%s] %.5f %.5f %.5f' %
                        ((results['rot_const'][1],) + tuple(results['rot_const'][0])), output_stream, output_file)
        output_to_console_and_file('Symmetry number %d' % results['sym_number'][0], output_stream, output_file)
        output_to_console_and_file('Zero-point energy (ZPE) %.10f [Eh]' % results['ZPE'][0], output_stream, output_file)
        output_to_console_and_file('                    %s' % ' '.join('%20s' % x for x in keys), 
                                output_stream, output_file)

        output_thermodynamic_property(results, 'Entropy', 'S', keys, output_stream, output_file)
        output_thermodynamic_property(results, 'Cv', 'Cv', keys, output_stream, output_file)
        output_thermodynamic_property(results, 'Cp', 'Cp', keys, output_stream, output_file)

        output_to_console_and_file('Internal energy [Eh]     %20.10f    %20.10f    %20.10f    %20.10f    %20.10f' %
                        (results['E_tot'][0], results['E_elec'][0], 
                        results['E_trans'][0], results['E_rot'][0], 
                        results['E_vib'][0]), 
                        output_stream, output_file)
        output_to_console_and_file('Total internal energy [Eh]  %.10f' % results['E_tot'][0], 
                        output_stream, output_file)
        output_to_console_and_file('Electronic energy [Eh]      %.10f' % results['E0'][0], 
                        output_stream, output_file)
        
        output_to_console_and_file('Enthalpy [Eh]           %20.10f    %20.10f    %20.10f    %20.10f    %20.10f' %
                        (results['H_tot'][0], results['H_elec'][0], 
                        results['H_trans'][0], results['H_rot'][0], 
                        results['H_vib'][0]),
                        output_stream, output_file)
        output_to_console_and_file('Total enthalpy [Eh]        %.10f' % results['H_tot'][0], 
                        output_stream, output_file)
        
        output_to_console_and_file('Gibbs free energy [Eh]  %20.10f    %20.10f    %20.10f    %20.10f    %20.10f' %
                        (results['G_tot'][0], results['G_elec'][0], 
                        results['G_trans'][0], results['G_rot'][0], 
                        results['G_vib'][0]),
                        output_stream, output_file)
        output_to_console_and_file('Total Gibbs free energy [Eh]  %.10f' % results['G_tot'][0], 
                        output_stream, output_file)
        
    def create_vibration_animation(self, mode_indices=None, n_frames=20, amplitude=3.0, output_dir=None, 
                                 include_imag=True, cutoff_freq=10.0):
        """
        Create animations of normal modes and output to xyz files.

        Parameters
        ----------
        mode_indices : list or int or None
            Indices of modes to animate.
            Animates all supported modes if None.
        n_frames : int
            Number of frames in each animation.
        amplitude : float
            Amplitude of vibration (in Angstroms).
        output_dir : str
            Output directory (current directory if None).
        include_imag : bool
            Whether to include imaginary frequencies.
        cutoff_freq : float
            Cutoff frequency in cm^-1. Modes with absolute frequency below this value 
            will be excluded.

        Returns
        -------
        list
            List of paths to the output files.
        """
        animator = _VibrationalModeAnimator(self, output_dir, include_imag, cutoff_freq)
        if mode_indices is None:
            # Animate all modes
            return animator.create_all_animations(n_frames, amplitude)
        elif isinstance(mode_indices, int):
            # Animate a single mode
            return [animator.create_animation(mode_indices, n_frames, amplitude)]
        else:
            # Animate specified modes
            results = []
            for idx in mode_indices:
                results.append(animator.create_animation(idx, n_frames, amplitude))
            return results

    def _chunk_iterator(self, start, end, step):
        """Helper function for iterating in chunks for printing."""
        for i in range(start, end, step):
            yield i, min(i + step, end)

class _VibrationalModeAnimator:
    """
    Internal helper class to create vibrational mode animations and output to xyz files.
    """

    def __init__(self, analyzer, output_dir=None, include_imag=True, cutoff_freq=0.0001):
        """
        Parameters
        ----------
        analyzer : MolecularVibrations
            Analyzer object that performed normal mode analysis.
        output_dir : str
            Output directory (current directory if None).
        include_imag : bool
            Whether to include imaginary frequencies.
        cutoff_freq : float
            Cutoff frequency in cm^-1. Modes with absolute frequency below this value 
            will be excluded.
        """
        self.analyzer = analyzer
        self.output_dir = output_dir or os.getcwd()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Check if key exists instead of evaluating the array
        if 'freq_wavenumber' not in analyzer.results:
            analyzer.analyze_normal_modes()

        # Prepare data for animation
        self._prepare_animation_data(include_imag, cutoff_freq)

    def _prepare_animation_data(self, include_imag=True, cutoff_freq=10.0):
        """
        Prepare data for animation.
        
        Parameters
        ----------
        include_imag : bool
            Whether to include imaginary frequencies.
        cutoff_freq : float
            Cutoff frequency in cm^-1. Modes with absolute frequency below this value 
            will be excluded.
        """
        self.atoms = self.analyzer.atoms
        self.coordinates = self.analyzer.coordinates
        freq_wn = self.analyzer.results['freq_wavenumber']

        # Filter out modes based on cutoff frequency
        if include_imag:
            # Include both real and imaginary frequencies, but filter out near-zero values
            idx = np.where((np.abs(freq_wn.real) > cutoff_freq) | (freq_wn.imag > cutoff_freq))[0]
        else:
            # Only positive real frequencies above cutoff
            idx = np.where(freq_wn.real > cutoff_freq)[0]
        
        # Sort modes: real frequencies first in ascending order, then imaginary in descending magnitude
        sort_idx = np.argsort(freq_wn[idx].real)
        idx = idx[sort_idx]
        
        self.freq_wn = freq_wn[idx]
        self.is_imag = self.freq_wn.imag > 0
        self.norm_mode = self.analyzer.results['norm_mode'][idx]
        self.n_modes = len(idx)

    def create_animation(self, mode_index, n_frames=20, amplitude=1.0, filename=None):
        """
        Create an animation for a specified mode and output to an xyz file.

        Parameters
        ----------
        mode_index : int
            Mode index (starting from 0).
        n_frames : int
            Number of frames in the animation.
        amplitude : float
            Amplitude of vibration (in Angstroms).
        filename : str
            Output filename (auto-generated if None).

        Returns
        -------
        str
            Path to the output file.
        """
        if mode_index >= self.n_modes or mode_index < 0:
            raise ValueError(f"Mode index must be between 0 and {self.n_modes-1}")

        is_imag = self.is_imag[mode_index] if hasattr(self, 'is_imag') else False
        freq_display = self.freq_wn[mode_index]
        if is_imag:
            freq_str = f"{abs(freq_display.imag):.0f}i"
        else:
            freq_str = f"{freq_display.real:.0f}"

        if filename is None:
            filename = f"mode_{mode_index+1}_{freq_str}_wave_number.xyz"

        filepath = os.path.join(self.output_dir, filename)

        mode_vector = self.norm_mode[mode_index].real  # Use real part of mode vector

        with open(filepath, 'w') as f:
            for frame in range(n_frames):
                phase = 2 * np.pi * frame / (n_frames - 1)
                displacement = amplitude * np.sin(phase)
                displaced_coords = self.coordinates.copy()
                for atom_idx in range(len(self.atoms)):
                    displaced_coords[atom_idx] += displacement * mode_vector[atom_idx]
                f.write(f"{len(self.atoms)}\n")
                f.write(f"Mode {mode_index+1}, Freq: {freq_str} cm-1, Frame: {frame+1}/{n_frames}\n")
                for atom_idx, atom in enumerate(self.atoms):
                    x, y, z = displaced_coords[atom_idx] * BOHR2ANGSTROM
                    f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")
        return filepath

    def create_all_animations(self, n_frames=20, amplitude=0.5, base_filename=None):
        """
        Create animations for all vibrational modes.

        Parameters
        ----------
        n_frames : int
            Number of frames for each animation.
        amplitude : float
            Amplitude of vibration (in Angstroms).
        base_filename : str
            Base name for output files (default is "mode_").

        Returns
        -------
        list
            List of paths to the output files.
        """
        base_filename = base_filename or "mode_"
        output_files = []
        for i in range(self.n_modes):
            self.is_imag[i] = self.freq_wn[i].imag > 0
            if self.is_imag[i]:
                freq_str = f"{abs(self.freq_wn[i].imag):.0f}i"
            else:
                freq_str = f"{self.freq_wn[i].real:.0f}"
            
            filename = f"{base_filename}{i+1}_{freq_str}_wave_number.xyz"
            output_path = self.create_animation(i, n_frames, amplitude, filename)
            output_files.append(output_path)
        return output_files

def analyze_molecular_vibrations(atoms, coordinates, hessian, symm_tolerance=0.25, max_symm_fold=6):
    """
    Analyze normal modes of a molecular system.

    Parameters
    ----------
    atoms : list of str
        List of atomic symbols.
    coordinates : np.ndarray
        Atomic coordinates (n_atoms, 3).
    hessian : np.ndarray
        Hessian matrix (3*n_atoms, 3*n_atoms) in atomic units.
    symm_tolerance : float
        Distance tolerance for symmetry operations.
    max_symm_fold : int
        Maximum n-fold rotation to check.

    Returns
    -------
    dict
        Results of normal mode analysis.
    """
    analyzer = MolecularVibrations(atoms, coordinates, hessian, symm_tolerance, max_symm_fold)
    return analyzer.analyze_normal_modes()


def calculate_molecular_thermochemistry(atoms, coordinates, hessian, e_tot=0.0, temperature=298.15,
                              pressure=101325, symm_tolerance=0.25, max_symm_fold=6):
    """
    Calculate thermochemical properties for a molecular system.

    Parameters
    ----------
    atoms : list of str
        List of atomic symbols.
    coordinates : np.ndarray
        Atomic coordinates (n_atoms, 3).
    hessian : np.ndarray
        Hessian matrix (3*n_atoms, 3*n_atoms) in atomic units.
    e_tot : float
        Total electronic energy in Hartree.
    temperature : float
        Temperature in Kelvin.
    pressure : float
        Pressure in Pascal.
    symm_tolerance : float
        Distance tolerance for symmetry operations.
    max_symm_fold : int
        Maximum n-fold rotation to check.

    Returns
    -------
    dict
        Thermochemistry results.
    """
    analyzer = MolecularVibrations(atoms, coordinates, hessian, symm_tolerance, max_symm_fold)
    analyzer.analyze_normal_modes()
    return analyzer.calculate_thermochemistry(e_tot, temperature, pressure)


def generate_vibration_animation(atoms, coordinates, hessian, mode_index=None, 
                               n_frames=20, amplitude=3.0, output_dir=None, 
                               symm_tolerance=0.25, max_symm_fold=6):
    """
    Generate animation files for molecular vibrations.
    
    Parameters
    ----------
    atoms : list of str
        List of atomic symbols.
    coordinates : np.ndarray
        Atomic coordinates (n_atoms, 3).
    hessian : np.ndarray
        Hessian matrix (3*n_atoms, 3*n_atoms) in atomic units.
    mode_index : int or list or None
        Index or indices of modes to animate (None for all modes)
    n_frames : int
        Number of frames in each animation.
    amplitude : float
        Amplitude of vibration (in Angstroms).
    output_dir : str
        Output directory (current directory if None).
    symm_tolerance : float
        Distance tolerance for symmetry operations.
    max_symm_fold : int
        Maximum n-fold rotation to check.
        
    Returns
    -------
    list
        List of paths to the animation files.
    """
    analyzer = MolecularVibrations(atoms, coordinates, hessian, symm_tolerance, max_symm_fold)
    analyzer.analyze_normal_modes()
    return analyzer.create_vibration_animation(mode_index, n_frames, amplitude, output_dir)