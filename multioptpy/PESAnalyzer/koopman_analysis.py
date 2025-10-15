import numpy as np
import csv
import os

from multioptpy.Parameters.unit_values import UnitValueLib


class KoopmanAnalyzer:
    def __init__(self, natom, window_size=10, num_frames=10, amplitude=1.5, rank_threshold=1e-5, poly_degree=2, file_directory=None):
        self.natom = natom
        self.window_size = window_size
        self.num_frames = num_frames
        self.amplitude = amplitude
        self.rank_threshold = rank_threshold
        self.poly_degree = poly_degree  # Degree of polynomial terms in observation function
        if file_directory is not None:
            tmp_directory = file_directory + '/koopman_analysis'
            os.makedirs(tmp_directory, exist_ok=True)
            self.coord_file = tmp_directory + '/coordinates.csv'
            self.eigs_file = tmp_directory + '/koopman_eigs.csv'
            self.modes_file = tmp_directory + '/koopman_modes.log'
            self.anim_file_prefix = tmp_directory + '/koopman_mode_anim_'
        else:
            self.coord_file = 'coordinates.csv'
            self.eigs_file = 'koopman_eigs.csv'
            self.modes_file = 'koopman_modes.log'
            self.anim_file_prefix = 'koopman_mode_anim_'
        
        self.coordinates = []  # List of (iteration, coords) tuples
        self.gradients = []  # List of (iteration, grads) tuples
        
        self.bohr2ang = UnitValueLib().bohr2angstroms

    def observation_function(self, coords, grads=None):
        """Extended observation function for EDMD: polynomial terms, interatomic distances, and gradients."""
        x = coords.flatten()
        # Polynomial terms from x^1 to x^poly_degree
        poly_terms = np.concatenate([x**k for k in range(1, self.poly_degree + 1)])
        
        # Add physical observables: interatomic distances
        distances = []
        for i in range(self.natom):
            for j in range(i+1, self.natom):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        distances = np.array(distances)
        
        
        # Concatenate all observables
        psi = np.concatenate([poly_terms, distances])
        
        # Add gradients if provided
        if grads is not None:
            g = grads.flatten()
            psi = np.concatenate([psi, g])
        
        return psi

    def append_coordinates(self, iteration, coords, grads=None):
        """Append coordinates and gradients to CSV log and internal lists."""
        if len(coords) != 3 * self.natom:
            raise ValueError("Coords must be of length 3*Natom")
        if grads is not None and len(grads) != 3 * self.natom:
            raise ValueError("Grads must be of length 3*Natom")
        
        # Append to internal lists
        self.coordinates.append((iteration, np.array(coords)))
        if grads is not None:
            self.gradients.append((iteration, np.array(grads)))
        
        # Write to CSV
        file_exists = os.path.isfile(self.coord_file)
        with open(self.coord_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['iteration'] + [f'{axis}{i+1}' for i in range(self.natom) for axis in ['x', 'y', 'z']]
                if grads is not None:
                    header += [f'grad_{axis}{i+1}' for i in range(self.natom) for axis in ['x', 'y', 'z']]
                writer.writerow(header)
            row = [iteration] + coords.tolist()
            if grads is not None:
                row += grads.tolist()
            writer.writerow(row)

    def perform_koopman_analysis(self):
        """Perform EDMD-style Koopman analysis with rank truncation."""
        if len(self.coordinates) < self.window_size + 1:
            print("Not enough data for analysis.")
            return None
        
        # Get the last window_size + 1 coords and grads for EDMD
        recent_coords = [coord[1] for coord in self.coordinates[-self.window_size-1:]]
        recent_grads = [grad[1] for grad in self.gradients[-self.window_size-1:]] if self.gradients else [None] * len(recent_coords)
        
        # Apply observation function to extend data
        Psi_X = np.column_stack([self.observation_function(c, g) for c, g in zip(recent_coords[:-1], recent_grads[:-1])])
        Psi_X_prime = np.column_stack([self.observation_function(c, g) for c, g in zip(recent_coords[1:], recent_grads[1:])])
        
        # EDMD with rank truncation
        U, Sigma, Vt = np.linalg.svd(Psi_X, full_matrices=False)
        
        rank = np.sum(Sigma > self.rank_threshold)
        if rank == 0:
            print("Warning: All singular values are below threshold. No rank found.")
            return None
            
        U_r = U[:, :rank]
        Sigma_r = np.diag(Sigma[:rank])
        Vt_r = Vt[:rank, :]
        
        # Corrected A_tilde calculation to match standard EDMD form: A_tilde = U_r^T Psi_X_prime Vt_r^T Sigma_r^{-1}
        A_tilde = np.dot(np.dot(U_r.T, Psi_X_prime), Vt_r.T) @ np.linalg.inv(Sigma_r)
        lambdas, W = np.linalg.eig(A_tilde)
        
        # Compute modes (eigenfunctions of Koopman operator in extended space)
        Phi = np.dot(np.dot(Psi_X_prime, Vt_r.T), np.linalg.solve(Sigma_r, W))
        
        # Sort by eigenvalue absolute value (descending)
        sort_indices = np.argsort(-np.abs(lambdas))
        lambdas = lambdas[sort_indices]
        Phi = Phi[:, sort_indices]
        
        # Log eigenvalues to CSV: one row per iteration, with columns for each mode's real, imag, norm
        current_iter = self.coordinates[-1][0]
        file_exists = os.path.isfile(self.eigs_file)
        with open(self.eigs_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['iteration', 'rank']
                for i in range(len(lambdas)):
                    header.extend([f'mode_{i}_real', f'mode_{i}_imag', f'mode_{i}_norm'])
                writer.writerow(header)
            row = [current_iter, rank]
            for lam in lambdas:
                row.extend([lam.real, lam.imag, np.abs(lam)])
            writer.writerow(row)

        # Save modes to file
        with open(self.modes_file, 'a') as f:
            f.write(f"Iteration {current_iter}:\n")
            for i in range(len(lambdas)):
                lam = lambdas[i]
                f.write(f"Eigenvalue {i}: Real={lam.real:.6f}, Imag={lam.imag:.6f}, Norm={np.abs(lam):.6f}\n")
                f.write("Mode vector (real part):\n")
                np.savetxt(f, Phi[:, i].real, fmt='%.6f')
                f.write("Mode vector (imag part):\n")
                np.savetxt(f, Phi[:, i].imag, fmt='%.6f')
                f.write("\n")
            f.write("\n")
        
        # For animation, project back to original coordinate space (take the identity part of the modes)
        # Modes are in extended space, so take the first 3*natom components (identity part)
        modes = Phi[:3*self.natom, :].real
        # Normalize each mode vector
        for i in range(modes.shape[1]):
            norm = np.linalg.norm(modes[:, i])
            if norm > 1e-9: # Avoid division by zero
                modes[:, i] /= norm
        return modes, lambdas

    def generate_animations(self, modes, element_list=None):
        """Generate XYZ animations for all modes using simple sinusoidal oscillation."""
        if len(self.coordinates) == 0:
            print("No coordinates available.")
            return
        
        # Remove previous animation files
        for filename in os.listdir('.'):
            if filename.startswith(self.anim_file_prefix) and filename.endswith('.xyz'):
                os.remove(filename)
        
        latest_coords = self.coordinates[-1][1].reshape((self.natom, 3))
        
        for mode_idx in range(modes.shape[1]):
            mode = modes[:, mode_idx]     # real mode vector in original space

            anim_file = f"{self.anim_file_prefix}{mode_idx}.xyz"
            with open(anim_file, 'w') as f:
                for frame in range(self.num_frames):
                    # Use simple sinusoidal oscillation of the real part of the mode
                    displacement = self.amplitude * np.sin(2 * np.pi * frame / (self.num_frames - 1)) * mode.reshape((self.natom, 3))
                    
                    displaced_coords = latest_coords + displacement
                    displaced_coords *= self.bohr2ang  # Convert to Angstroms
                    
                    f.write(f"{self.natom}\n")
                    f.write(f"Mode {mode_idx} Frame {frame}\n")
                    for i in range(self.natom):
                        if element_list is not None and i < len(element_list):
                            elem = element_list[i]
                        else:
                            elem = 'X'  # Placeholder element
                        f.write(f"{elem} {displaced_coords[i, 0]:.6f} {displaced_coords[i, 1]:.6f} {displaced_coords[i, 2]:.6f}\n")

    def run(self, iteration, coords, grad, element_list=None):#coords:Bohr. grad:Hartree/Bohr
        """Convenience method to append data, perform analysis, and generate animations."""
        if len(coords) != 3 * self.natom:
            coords = coords.flatten()
        if grad is not None and len(grad) != 3 * self.natom:
            grad = grad.flatten()
            
        self.append_coordinates(iteration, coords, grad)
        result = self.perform_koopman_analysis()
        if result is not None:
            modes, lambdas = result
            self.generate_animations(modes, element_list)
            return modes, lambdas

        return None

"""
| Number | |λ|   | Re(λ) | Im(λ) | Behavior                  | Result                           | Mathematical Basis and Example |
|--------|---------|--------|--------|---------------------------|----------------------------------|-------------------------------|
| 1      | =1     | >0    | =0    | Constant amplitude, monotonic progression | Does not converge, proceeds in the same direction | λ = positive real number (e.g., λ=1). Response: Increases like a ramp function. Equivalent to an integrator. |
| 2      | <1     | >0    | =0    | Damping, monotonic progression | Converges, straight toward the minimum value | λ = positive real number <1 (e.g., λ=0.5). Exponential decay: x(k) ~ (0.5)^k. Stable. |
| 3      | <1     | <0    | =0    | Damping, direction reversal | Converges with alternating sign changes (simple oscillation) | λ = negative real number >-1 (e.g., λ=-0.5). Damped oscillation: (-0.5)^k with sign alternation. |
| 4      | =1     | <0    | =0    | Constant amplitude, reversal | Oscillates without damping, does not converge | λ = -1. Sustained oscillation: (-1)^k with alternating signs. Oscillator. |
| 5      | >1     | >0    | =0    | Amplification, monotonic progression | Diverges, convergence impossible | λ >1 (e.g., λ=1.5). Exponential amplification: (1.5)^k diverges. |
| 6      | >1     | <0    | =0    | Amplification, reversal | Diverges with oscillation | λ < -1 (e.g., λ=-1.5). Amplified oscillation: (-1.5)^k with diverging vibration. |
| 7      | <1     | >0    | ≠0    | Damping, spiral/oscillation | Converges with decaying rotation and vibration | Complex λ, |λ|<1, Re>0. Spiral damping: ρ^k e^{jθk} (ρ<1). |
| 8      | <1     | <0    | ≠0    | Damping, reversal + spiral | Converges spirally with reversal | Complex λ, |λ|<1, Re<0. Reversal spiral damping. |
| 9      | =1     | >0    | ≠0    | Constant amplitude, spiral | Persists with rotation without convergence | Complex λ, |λ|=1, Re>0. Sustained rotation: e^{jθk}. |
| 10     | =1     | <0    | ≠0    | Constant amplitude, reversal + spiral | Persists with reversal and rotation, without convergence | Complex λ, |λ|=1, Re<0. Reversal sustained rotation. |
| 11     | >1     | >0    | ≠0    | Amplification, spiral | Diverges with rotation | Complex λ, |λ|>1, Re>0. Spiral divergence. |
| 12     | >1     | <0    | ≠0    | Amplification, reversal + spiral | Diverges with reversal + rotation | Complex λ, |λ|>1, Re<0. Reversal spiral divergence. |

"""



def main():
    natom = 5
   
    analyzer = KoopmanAnalyzer(natom)
    
    # Generate dummy data with stable oscillation and noise
    np.random.seed(42)
    base_coords = np.random.rand(3 * natom) * 5
    slow_mode_vec = np.random.randn(3 * natom)
    slow_mode_vec /= np.linalg.norm(slow_mode_vec) # Normalize

    last_modes = None
    for iteration in range(35):
        # Artificially add a very slowly decaying oscillation mode
        slow_oscillation = 0.5 * np.sin(iteration * 0.1) * slow_mode_vec
        # Derivative of the sine curve for gradients
        slow_gradient = 0.5 * 0.1 * np.cos(iteration * 0.1) * slow_mode_vec
        # A bit of fast noise
        noise = 0.05 * np.random.randn(3 * natom)
        coords = base_coords + slow_oscillation + noise
        grad = slow_gradient + noise  # Add noise to gradients as well for realism
        analyzer.append_coordinates(iteration, coords, grad)
        
        # Perform analysis every 'window_size' steps
        if (iteration + 1) >= analyzer.window_size:
            result = analyzer.perform_koopman_analysis()
            if result is not None:
                modes, lambdas = result
                print(f"\nAnalysis at iteration {iteration}:")
                print(f"Top 3 mode norms: {[f'{np.abs(l):.4f}' for l in lambdas[:3]]}")
                last_modes = modes  # Store the latest modes

    # Generate animations using the modes from the last iteration
    if last_modes is not None:
        analyzer.generate_animations(last_modes)

if __name__ == "__main__":
    main()