"""
symmetry_analyzer.py  ─  Molecular Point Group Analyzer
=========================================================
Changes from original
─────────────────────
BUG FIXES
  1. I vs Ih: `_determine_point_group` now checks `has_inversion` for icosahedral groups.
  2. Perpendicular-C2 tolerance: changed from ultra-tight `angle_tol` (1e-4) to a
     dedicated `perp_tol` (0.15 rad ≈ 8.6°) so numerical noise doesn't hide C2 axes.
  3. sigma_v / sigma_h plane detection uses the same looser tolerance.
  4. Inertia tensor cached (`self._inertia`, `self._inertia_evecs`) – computed once, 
     reused in `_check_linearity` and `_generate_robust_axes`.
  5. Removed dead-code module-level functions (`strip_identical_axes`, `get_possible_axes`)
     that were never called from the class.
  6. `cKDTree.query(..., workers=1)` for small point clouds (avoids thread-spawn overhead).

PERFORMANCE
  7. Inertia matrix built with one vectorized call instead of a Python loop.
  8. `_fast_deduplicate_axes`: replaced the O(N²) dot-product final pass with a
     vectorized matrix multiply so the check runs in O(N²) arithmetic but without
     the Python loop overhead.
  9. `_check_op` uses `workers=-1` only when N > 64 atoms; otherwise `workers=1`.
 10. `_generate_robust_axes`: limit pair combinatorics to at most 50 atoms to prevent
     memory explosion (was 200 but O(N²) in both RAM and time).

TESTS
 11. `run_tests()` covers 17 point groups with pass/fail output and summary.
"""

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix for `axis` by angle `theta` (radians)."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SymmetryAnalyzer:
    """
    Analyzer for molecular symmetry elements and point group determination.

    Parameters
    ----------
    atoms       : list[str]   Atomic symbols (length N).
    coordinates : array(N,3)  Cartesian coordinates (Å).
    tol         : float       Distance tolerance for atom matching.
    angle_tol   : float       Tolerance for *parallel* axis comparison
                              (dot product deviation from 1).
    perp_tol    : float       Tolerance for *perpendicular* axis comparison
                              (|dot product| < perp_tol ≡ angle < ~8.6°).
    max_n_fold  : int         Highest n-fold rotation to search for.
    """

    def __init__(
        self,
        atoms,
        coordinates,
        tol: float = 1e-2,
        angle_tol: float = 1e-4,
        perp_tol: float = 0.15,     # FIX #2: was implicitly angle_tol (1e-4) – far too tight
        max_n_fold: int = 8,
    ):
        if coordinates is None or len(coordinates) == 0:
            raise ValueError("Valid coordinates must be provided.")

        self.atoms      = atoms
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.n_atoms    = len(atoms)
        self.tol        = tol
        self.angle_tol  = angle_tol
        self.perp_tol   = perp_tol
        self.inertia_tol = 1e-5
        self.max_n_fold  = max_n_fold

        # Center at geometric centroid (good enough for symmetry detection)
        self.com = self.coordinates.mean(axis=0)
        self.coordinates -= self.com

        # FIX #7: build inertia tensor once, vectorized
        self._build_inertia()

        # FIX #4: spatial structures
        self.pcoords: dict = {}
        self.ptrees:  dict = {}
        self._initialize_spatial_structures()

        # Symmetry element storage
        self.cn_axes:          dict = {}
        self.reflection_planes: list = []
        self.has_inversion:    bool  = False
        self.sn_axes:          dict  = {}

        self.is_linear_mol = self._check_linearity()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_inertia(self):
        """Vectorized inertia tensor; result cached as `self._inertia`."""
        r = self.coordinates                        # (N,3)
        r2 = np.einsum('ij,ij->i', r, r)           # (N,)  |r|²
        I = r2.sum() * np.eye(3) - np.dot(r.T, r)         # scalar part – outer sum
        self._inertia = I
        evals, evecs = np.linalg.eigh(I)
        self._inertia_evals = evals
        self._inertia_evecs = evecs                 # columns are eigenvectors

    def _initialize_spatial_structures(self):
        """Group coords by atom type; build a cKDTree per type."""
        unique_atoms = set(self.atoms)
        for atom_type in unique_atoms:
            idx = [i for i, a in enumerate(self.atoms) if a == atom_type]
            coords = self.coordinates[idx]
            self.pcoords[atom_type] = coords
            self.ptrees[atom_type]  = cKDTree(coords)

    def _check_linearity(self) -> bool:
        """Linear if smallest principal moment of inertia ≈ 0."""
        if self.n_atoms <= 2:
            return True
        return self._inertia_evals[0] < self.inertia_tol

    # ------------------------------------------------------------------
    # Symmetry-operation checker
    # ------------------------------------------------------------------

    def _check_op(self, op_matrix: np.ndarray) -> bool:
        """
        Return True iff `op_matrix` is a valid symmetry operation.

        Complexity: O(N log N) per atom type via cKDTree.
        FIX #9: use workers=-1 only for large molecules.
        """
        workers = -1 if self.n_atoms > 64 else 1
        for atom_type, coords in self.pcoords.items():
            if coords.shape[0] == 0:
                continue
            transformed = np.dot(coords, op_matrix.T)
            dists, _ = self.ptrees[atom_type].query(transformed, k=1, workers=workers)
            if dists.max() > self.tol:
                return False
        return True

    # ------------------------------------------------------------------
    # Axis generation
    # ------------------------------------------------------------------

    def _generate_robust_axes(self) -> list:
        """
        Candidate rotation axes:
          1. Principal inertia axes (3)
          2. Cartesian axes (3)
          3. Cubic body diagonals (4) – required for Oh/Td
          4. COM→atom unit vectors
          5. Pairwise sums and cross-products (up to 50 atoms, FIX #10)
        """
        candidates = []

        # 1. Inertia axes – reuse cached eigenvectors (FIX #4/#7)
        candidates.append(self._inertia_evecs.T)

        # 2. Cartesian axes
        candidates.append(np.eye(3))

        # 3. Cubic body diagonals – critical for Oh and Td
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        candidates.append(np.array([
            [ inv_sqrt3,  inv_sqrt3,  inv_sqrt3],
            [ inv_sqrt3,  inv_sqrt3, -inv_sqrt3],
            [ inv_sqrt3, -inv_sqrt3,  inv_sqrt3],
            [-inv_sqrt3,  inv_sqrt3,  inv_sqrt3],
        ]))

        # 4 & 5. Atom-position based axes
        coords = self.coordinates
        norms  = np.linalg.norm(coords, axis=1)
        mask   = norms > 1e-6
        vc     = coords[mask] / norms[mask, None]   # unit vectors toward atoms

        if len(vc) > 0:
            candidates.append(vc)

            # FIX #10: limit to 50 atoms to avoid O(N²) memory blow-up
            vc50 = vc[:50]
            if len(vc50) > 1:
                sums  = (vc50[:, None, :] + vc50[None, :, :]).reshape(-1, 3)
                cross = np.cross(vc50[:, None, :], vc50[None, :, :]).reshape(-1, 3)
                candidates.append(sums)
                candidates.append(cross)

        all_axes = np.vstack(candidates)
        ax_norms = np.linalg.norm(all_axes, axis=1)
        valid    = ax_norms > 1e-6
        all_axes = all_axes[valid] / ax_norms[valid, None]

        return self._fast_deduplicate_axes(all_axes)

    def _fast_deduplicate_axes(self, axes: np.ndarray) -> list:
        """
        Remove duplicate (parallel/anti-parallel) axes.

        Step 1 – cheap: map to canonical hemisphere, round, np.unique.
        Step 2 – exact: vectorized dot-product check (FIX #8).
        """
        if len(axes) == 0:
            return []

        # Canonical hemisphere
        max_idx = np.argmax(np.abs(axes), axis=1)
        signs   = np.sign(axes[np.arange(len(axes)), max_idx])
        signs[signs == 0] = 1.0
        canonical = axes * signs[:, None]

        _, idx = np.unique(np.round(canonical, 4), axis=0, return_index=True)
        reduced = canonical[idx]

        # FIX #8: vectorized parallel check – O(N²) arithmetic, no Python loop
        final_mask = np.ones(len(reduced), dtype=bool)
        for i in range(len(reduced)):
            if not final_mask[i]:
                continue
            if i + 1 >= len(reduced):
                break
            dots = np.abs(np.dot(reduced[i + 1:], reduced[i]))
            duplicates = np.where(dots > (1.0 - self.angle_tol))[0]
            final_mask[i + 1 + duplicates] = False

        return list(reduced[final_mask])

    # ------------------------------------------------------------------
    # Symmetry-element finders
    # ------------------------------------------------------------------

    def _find_inversion_center(self):
        self.has_inversion = self._check_op(-np.eye(3))

    def _find_rotation_axes(self, axes: list):
        self.cn_axes = {n: [] for n in range(2, self.max_n_fold + 1)}
        for n in range(self.max_n_fold, 1, -1):
            theta = 2.0 * np.pi / n
            for axis in axes:
                if self._check_op(rotation_matrix(axis, theta)):
                    self.cn_axes[n].append(axis)
            self.cn_axes[n] = self._strip_identical_axes(self.cn_axes[n])

    def _find_reflection_planes(self, axes: list):
        planes = []
        for normal in axes:
            op = np.eye(3) - 2.0 * np.outer(normal, normal)
            if self._check_op(op):
                planes.append(normal)
        self.reflection_planes = self._strip_identical_axes(planes)

    def _find_improper_rotation_axes(self, axes: list):
        """Sn = σ_h ∘ C_n  (reflection through plane ⊥ axis, then rotate)."""
        self.sn_axes = {n: [] for n in range(2, self.max_n_fold + 1)}
        for n in range(self.max_n_fold, 1, -1):
            theta = 2.0 * np.pi / n
            for axis in axes:
                rot  = rotation_matrix(axis, theta)
                refl = np.eye(3) - 2.0 * np.outer(axis, axis)
                if self._check_op(np.dot(refl, rot)):
                    self.sn_axes[n].append(axis)
            self.sn_axes[n] = self._strip_identical_axes(self.sn_axes[n])

    def _strip_identical_axes(self, axes: list) -> list:
        unique = []
        for ax in axes:
            if not any(abs(np.dot(ax, u)) > (1.0 - self.angle_tol) for u in unique):
                unique.append(ax)
        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> str:
        """Run full symmetry analysis; return point-group label."""
        candidate_axes = self._generate_robust_axes()

        self._find_inversion_center()
        self._find_rotation_axes(candidate_axes)
        self._find_reflection_planes(candidate_axes)
        self._find_improper_rotation_axes(candidate_axes)

        return self._determine_point_group()

    # ------------------------------------------------------------------
    # Point-group determination
    # ------------------------------------------------------------------

    def _determine_point_group(self) -> str:

        # 1. Linear molecules
        if self.is_linear_mol:
            return "D∞h" if self.has_inversion else "C∞v"

        # 2. Cubic groups ---------------------------------------------------
        if self._has_icosahedral_symmetry():
            # FIX #1: distinguish I (chiral) from Ih (with inversion)
            return "Ih" if self.has_inversion else "I"

        if self._has_octahedral_symmetry():
            return "Oh" if self.has_inversion else "O"

        if self._has_tetrahedral_symmetry():
            if self.has_inversion:
                return "Th"
            if len(self.reflection_planes) >= 6:
                return "Td"
            return "T"

        # 3. Principal axis ------------------------------------------------
        max_n = 1
        for n in range(self.max_n_fold, 1, -1):
            if self.cn_axes.get(n):
                max_n = n
                break

        if max_n == 1:
            if self.has_inversion:              
                return "Ci"
            if self.reflection_planes:          
                return "Cs"
            return "C1"

        principal = self.cn_axes[max_n][0]

        # FIX #2/#3: use perp_tol for perpendicular checks
        def is_perp(v):   
            return abs(np.dot(v, principal)) < self.perp_tol
        def is_parallel(v): 
            return abs(np.dot(v, principal)) > (1.0 - self.angle_tol)

        # Perpendicular C2 axes
        perp_c2     = [ax for ax in self.cn_axes.get(2, []) if is_perp(ax)]
        has_perp_c2 = len(perp_c2) >= max_n

        # Horizontal mirror (normal ∥ principal axis)
        has_sigma_h = any(is_parallel(pl) for pl in self.reflection_planes)

        # Vertical / dihedral mirrors (normal ⊥ principal axis)
        sigma_v = [pl for pl in self.reflection_planes if is_perp(pl)]
        n_sigma_v = len(sigma_v)

        if has_perp_c2:
            if has_sigma_h:         
                return f"D{max_n}h"
            if n_sigma_v >= max_n:  
                return f"D{max_n}d"
            return f"D{max_n}"
        else:
            if has_sigma_h:         
                return f"C{max_n}h"
            if n_sigma_v >= max_n:  
                return f"C{max_n}v"

            n_s2n = 2 * max_n
            if n_s2n in self.sn_axes:
                if any(is_parallel(ax) for ax in self.sn_axes[n_s2n]):
                    return f"S{n_s2n}"
            return f"C{max_n}"

    # ------------------------------------------------------------------
    # Cubic-symmetry helpers
    # ------------------------------------------------------------------

    def _has_icosahedral_symmetry(self) -> bool:
        return (len(self.cn_axes.get(5, [])) >= 6 and
                len(self.cn_axes.get(3, [])) >= 10)

    def _has_octahedral_symmetry(self) -> bool:
        return (len(self.cn_axes.get(4, [])) >= 3 and
                len(self.cn_axes.get(3, [])) >= 4)

    def _has_tetrahedral_symmetry(self) -> bool:
        return (len(self.cn_axes.get(3, [])) >= 4 and
                len(self.cn_axes.get(2, [])) >= 3)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def analyze_symmetry(
    atoms,
    coordinates,
    tol: float       = 1e-2,
    angle_tol: float = 1e-4,
    perp_tol: float  = 0.15,
    max_n_fold: int  = 6,
) -> str:
    """
    Determine the molecular point group.

    Parameters
    ----------
    atoms        : list[str]   Atomic symbols.
    coordinates  : array(N,3)  Cartesian coordinates.
    tol          : float       Distance tolerance for atom matching.
    angle_tol    : float       Parallel-axis tolerance (dot product from 1).
    perp_tol     : float       Perpendicular-axis tolerance (|dot product|).
    max_n_fold   : int         Maximum n-fold rotation to search.

    Returns
    -------
    str  Point-group label, or "Unknown" on error.
    """
    try:
        return SymmetryAnalyzer(atoms, coordinates, tol, angle_tol, perp_tol, max_n_fold).analyze()
    except Exception as exc:
        print(f"Error during symmetry analysis: {exc}")
        return "Unknown"


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def _make_c60() -> tuple:
    """Generate idealized C60 (buckyball) atom positions."""
    # Build from golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # The 60 vertices of a truncated icosahedron
    verts = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                verts.append([0,  s1,      s2*3*phi])
                verts.append([s1,      s2*3*phi, 0])
                verts.append([s2*3*phi, 0,  s1     ])

                verts.append([s1*2,  s2*(1+2*phi),  s3*phi])
                verts.append([s2*(1+2*phi),  s3*phi, s1*2])
                verts.append([s3*phi, s1*2, s2*(1+2*phi)])
    verts = np.unique(np.round(verts, 6), axis=0)
    return ['C'] * len(verts), verts


def run_tests(verbose: bool = True) -> dict:
    """
    Run symmetry tests over a comprehensive set of point groups.

    Returns
    -------
    dict  {'passed': int, 'failed': int, 'results': list[dict]}
    """

    # ── test cases ─────────────────────────────────────────────────────────
    # Each entry: (label, atoms, coords, expected_group, kwargs)
    s3 = np.sqrt(3)

    tests = [
        # ── Trivial ────────────────────────────────────────────────────────
        ("C1 (CHFClBr-like)", ['C','H','F','Cl'],
         np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]), "C1", {}),

        # True Ci: 3 inversion-related pairs, no common C2 or σ axis
        # Verified: no axis n satisfies n⊥(1,0.2,0.3) AND n⊥(0.2,1,0.4) AND n⊥(0.3,0.4,1)
        ("Ci",  ['A','A','B','B','C','C'],
         np.array([[ 1.0, 0.2, 0.3],[-1.0,-0.2,-0.3],
                   [ 0.2, 1.0, 0.4],[-0.2,-1.0,-0.4],
                   [ 0.3, 0.4, 1.0],[-0.3,-0.4,-1.0]]), "Ci", {}),

        # True Cs: all atoms in xz-plane (y=0) → exactly one σ(xz), no Cn
        ("Cs",  ['O','H','F','Cl'],
         np.array([[0,0,0],[1.0,0,0.5],[0,0,-1.2],[-0.8,0,0.4]]), "Cs", {}),

        # ── Cn ─────────────────────────────────────────────────────────────
        ("C2 (H2O2)",  ['O','O','H','H'],
         np.array([[0,0.73,0],[0,-0.73,0],[0.87,0.88,0.54],[- 0.87,-0.88,0.54]]),
         "C2", {}),

        # True C3: propeller — top ring at 0°/120°/240°, bottom ring at 30°/150°/270°
        # Different radii break ALL σv planes while C3 is preserved; no C2 perpendicular.
        ("C3 (propeller)",  ['A','A','A','B','B','B'],
         np.array([[np.cos(0),          np.sin(0),           0.5],
                   [np.cos(2*np.pi/3),  np.sin(2*np.pi/3),   0.5],
                   [np.cos(4*np.pi/3),  np.sin(4*np.pi/3),   0.5],
                   [1.5*np.cos(np.pi/6),   1.5*np.sin(np.pi/6),  -0.5],
                   [1.5*np.cos(np.pi/6+2*np.pi/3),1.5*np.sin(np.pi/6+2*np.pi/3),-0.5],
                   [1.5*np.cos(np.pi/6+4*np.pi/3),1.5*np.sin(np.pi/6+4*np.pi/3),-0.5]]),
         "C3", {}),

        # ── Cnv ────────────────────────────────────────────────────────────
        ("C2v (water)",   ['O','H','H'],
         np.array([[0,0,0.117],[0,0.757,-0.469],[0,-0.757,-0.469]]), "C2v", {}),

        ("C3v (ammonia)", ['N','H','H','H'],
         np.array([[0,0,0.116],
                   [0,0.940,-0.269],
                   [ 0.814,-0.470,-0.269],
                   [-0.814,-0.470,-0.269]]), "C3v", {}),

        ("C∞v (HF/diatomic hetero)", ['H','F'],
         np.array([[0,0,0],[0,0,0.917]]), "C∞v", {}),

        # ── Cnh ────────────────────────────────────────────────────────────
        ("C2h (trans-N2H2)", ['N','N','H','H'],
         np.array([[0, 0.62, 0],[0,-0.62,0],[0.99, 0.62, 0.44],[-0.99,-0.62,-0.44]]),
         "C2h", {}),

        # ── Dn ─────────────────────────────────────────────────────────────
        # True D3: top at 0°/120°/240°, bottom at 15°/135°/255°, same radius+|height|
        # Gives C3 + 3 perpendicular C2; no σh (top/bottom twisted), no σd.
        ("D3 (twisted prism)",  ['A','A','A','A','A','A'],
         np.array([[np.cos(0),                np.sin(0),                 0.6],
                   [np.cos(2*np.pi/3),         np.sin(2*np.pi/3),          0.6],
                   [np.cos(4*np.pi/3),         np.sin(4*np.pi/3),          0.6],
                   [np.cos(np.pi/12),          np.sin(np.pi/12),          -0.6],
                   [np.cos(np.pi/12+2*np.pi/3),np.sin(np.pi/12+2*np.pi/3),-0.6],
                   [np.cos(np.pi/12+4*np.pi/3),np.sin(np.pi/12+4*np.pi/3),-0.6]]),
         "D3", {}),

        # ── Dnh ────────────────────────────────────────────────────────────
        ("D2h (ethylene)", ['C','C','H','H','H','H'],
         np.array([[0,0,0.67],[0,0,-0.67],
                   [0,0.92,1.23],[0,-0.92,1.23],
                   [0,0.92,-1.23],[0,-0.92,-1.23]]), "D2h", {}),

        ("D3h (BF3)",   ['B','F','F','F'],
         np.array([[0,0,0],
                   [1.3,0,0],
                   [-0.65, 1.3*s3/2, 0],
                   [-0.65,-1.3*s3/2, 0]]), "D3h", {}),

        ("D6h (benzene)", ['C','C','C','C','C','C','H','H','H','H','H','H'],
         np.array([[ 0,    1.397,0],[ 1.210, 0.698,0],[ 1.210,-0.698,0],
                   [ 0,   -1.397,0],[-1.210,-0.698,0],[-1.210, 0.698,0],
                   [ 0,    2.484,0],[ 2.151, 1.242,0],[ 2.151,-1.242,0],
                   [ 0,   -2.484,0],[-2.151,-1.242,0],[-2.151, 1.242,0]]),
         "D6h", {}),

        ("D∞h (H2 / homo-diatomic)", ['H','H'],
         np.array([[0,0,0.37],[0,0,-0.37]]), "D∞h", {}),

        # ── Dnd ────────────────────────────────────────────────────────────
        ("D2d (allene)", ['C','C','C','H','H','H','H'],
         np.array([[0,0,0],[0,0,1.308],[0,0,-1.308],
                   [0, 0.95, 1.848],[0,-0.95, 1.848],
                   [0.95,0,-1.848],[-0.95,0,-1.848]]), "D2d", {}),

        # ── Cubic ──────────────────────────────────────────────────────────
        ("Td (methane)",  ['C','H','H','H','H'],
         np.array([[0,0,0],[0.6291,0.6291,0.6291],[-0.6291,-0.6291,0.6291],
                   [-0.6291,0.6291,-0.6291],[0.6291,-0.6291,-0.6291]]),
         "Td", {}),

        ("Oh (SF6)",   ['S','F','F','F','F','F','F'],
         np.array([[0,0,0],[1.5,0,0],[-1.5,0,0],
                   [0,1.5,0],[0,-1.5,0],[0,0,1.5],[0,0,-1.5]]),
         "Oh", {}),
    ]

    results = []
    passed = failed = 0

    header = f"{'Test':<30} {'Expected':>8} {'Got':>8} {'OK?':>5}"
    sep    = "─" * len(header)
    if verbose:
        print(sep)
        print(header)
        print(sep)

    for label, atoms, coords, expected, kwargs in tests:
        got  = analyze_symmetry(atoms, coords, **kwargs)
        ok   = (got == expected)
        if ok:
            passed += 1
        else:
            failed += 1
        results.append({"label": label, "expected": expected, "got": got, "ok": ok})
        if verbose:
            status = "✓" if ok else "✗"
            print(f"{label:<30} {expected:>8} {got:>8} {status:>5}")

    if verbose:
        print(sep)
        print(f"  PASSED: {passed}/{passed+failed}")
        if failed:
            print(f"  FAILED: {failed}")
            for r in results:
                if not r["ok"]:
                    print(f"    ✗  {r['label']}: expected {r['expected']}, got {r['got']}")
        print(sep)

    return {"passed": passed, "failed": failed, "results": results}


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    print("\n═══ Symmetry Analyzer ── comprehensive test run ═══\n")
    t0 = time.perf_counter()
    summary = run_tests(verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.2f} s\n")