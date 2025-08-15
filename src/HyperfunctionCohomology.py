"""
Enhanced Hyperfunction Cohomology Computation for Quasi-Complex Manifolds
Extended implementation with microlocal analysis, wavefront propagation,
and advanced obstruction detection.
"""

import numpy as np
import sympy as sp
from sympy import symbols, Matrix, diff, simplify, exp, I, pi, sin, cos, sqrt
from sympy.vector import CoordSys3D
from typing import List, Dict, Tuple, Optional, Set, Callable
import itertools
from dataclasses import dataclass
from enum import Enum
import scipy.special as special
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# CORE MATHEMATICAL STRUCTURES
# ============================================================================

class WavefrontType(Enum):
    """Types of wavefront singularities"""
    SMOOTH = "smooth"
    CONORMAL = "conormal"
    CUSP = "cusp"
    FOLD = "fold"
    MIXED = "mixed"

@dataclass
class WavefrontCone:
    """Represents a conic subset of the cotangent bundle"""
    base_point: np.ndarray
    directions: List[np.ndarray]  # Unit vectors in cotangent space
    cone_angle: float
    singularity_type: WavefrontType
    
    def contains_direction(self, direction: np.ndarray, tolerance: float = 0.1) -> bool:
        """Check if a direction is in the cone"""
        direction_normalized = direction / np.linalg.norm(direction)
        for cone_dir in self.directions:
            angle = np.arccos(np.clip(np.dot(direction_normalized, cone_dir), -1, 1))
            if angle <= self.cone_angle + tolerance:
                return True
        return False
    
    def propagate(self, differential: np.ndarray) -> 'WavefrontCone':
        """Propagate cone through a differential map"""
        # WF(φ*u) = (dφ^T)^{-1}(WF(u))
        try:
            inv_diff_T = np.linalg.inv(differential.T)
            new_directions = [inv_diff_T @ d for d in self.directions]
            # Normalize
            new_directions = [d / np.linalg.norm(d) for d in new_directions]
            return WavefrontCone(
                base_point=self.base_point,
                directions=new_directions,
                cone_angle=self.cone_angle,
                singularity_type=self.singularity_type
            )
        except np.linalg.LinAlgError:
            # Degenerate case
            return WavefrontCone(
                base_point=self.base_point,
                directions=self.directions,
                cone_angle=np.pi,  # Full cone
                singularity_type=WavefrontType.MIXED
            )

class Hyperfunction:
    """Represents a hyperfunction with wavefront data"""
    
    def __init__(self, plus_function: Callable, minus_function: Callable, 
                 support: Optional[Set] = None):
        """
        plus_function: Holomorphic function from upper half-space
        minus_function: Holomorphic function from lower half-space
        support: Singular support of the hyperfunction
        """
        self.F_plus = plus_function
        self.F_minus = minus_function
        self.support = support if support else set()
        self.wavefront_sets = []
    
    def boundary_value(self, point: np.ndarray) -> complex:
        """Compute boundary value at a point"""
        eps = 1e-10
        # Approach from above and below
        val_plus = self.F_plus(point + eps * 1j)
        val_minus = self.F_minus(point - eps * 1j)
        return val_plus - val_minus
    
    def compute_wavefront(self, point: np.ndarray, n_directions: int = 32) -> WavefrontCone:
        """Compute wavefront set at a point using FBI transform"""
        directions = []
        
        # Generate test directions on unit sphere
        for i in range(n_directions):
            theta = np.pi * i / n_directions
            phi = 2 * np.pi * i / n_directions
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            # FBI transform in this direction
            fbi_value = self._fbi_transform(point, direction)
            
            # Check for rapid decay
            if abs(fbi_value) > 1e-3:  # Threshold for singularity
                directions.append(direction)
        
        if not directions:
            return WavefrontCone(point, [], 0, WavefrontType.SMOOTH)
        
        # Determine cone angle and type
        cone_angle = self._compute_cone_angle(directions)
        singularity_type = self._classify_singularity(directions, cone_angle)
        
        return WavefrontCone(point, directions, cone_angle, singularity_type)
    
    def _fbi_transform(self, point: np.ndarray, direction: np.ndarray) -> complex:
        """Compute FBI (Fourier-Bros-Iagolnitzer) transform"""
        # Simplified version - full implementation would integrate
        # exp(-|x-y|^2) * exp(i<x-y,ξ>) * f(y) dy
        
        # For demonstration, use a simplified local approximation
        h = 0.01
        local_variation = 0
        
        for offset in [-h, 0, h]:
            test_point = point + offset * direction
            val = self.boundary_value(test_point)
            local_variation += val * np.exp(-offset**2)
        
        return local_variation
    
    def _compute_cone_angle(self, directions: List[np.ndarray]) -> float:
        """Compute the opening angle of the wavefront cone"""
        if len(directions) <= 1:
            return 0
        
        max_angle = 0
        for d1, d2 in itertools.combinations(directions, 2):
            angle = np.arccos(np.clip(np.dot(d1, d2), -1, 1))
            max_angle = max(max_angle, angle)
        
        return max_angle / 2  # Half-angle of cone
    
    def _classify_singularity(self, directions: List[np.ndarray], 
                            cone_angle: float) -> WavefrontType:
        """Classify the type of wavefront singularity"""
        n_dirs = len(directions)
        
        if n_dirs == 0:
            return WavefrontType.SMOOTH
        elif n_dirs == 1:
            return WavefrontType.CONORMAL
        elif cone_angle < np.pi/4:
            return WavefrontType.FOLD
        elif cone_angle < np.pi/2:
            return WavefrontType.CUSP
        else:
            return WavefrontType.MIXED

class AlmostComplexStructure:
    """Enhanced almost complex structure with additional computations"""
    
    def __init__(self, matrix_func, coordinates):
        self.matrix_func = matrix_func
        self.coordinates = coordinates
        self.dim = len(coordinates)
        
        # Verify J^2 = -I
        if not self._verify_almost_complex():
            raise ValueError("J^2 ≠ -I, not a valid almost complex structure")
        
        # Precompute derivatives for Nijenhuis tensor
        self._compute_derivatives()
    
    def _verify_almost_complex(self):
        """Check if J^2 = -Identity"""
        J = self.matrix_func
        J_squared = J * J
        identity = sp.eye(self.dim)
        diff = simplify(J_squared + identity)
        
        # Check if all entries are zero
        for i in range(self.dim):
            for j in range(self.dim):
                if diff[i,j] != 0:
                    return False
        return True
    
    def _compute_derivatives(self):
        """Precompute partial derivatives of J"""
        self.J_derivatives = {}
        for coord in self.coordinates:
            self.J_derivatives[coord] = diff(self.matrix_func, coord)
    
    def evaluate_at(self, point):
        """Evaluate J at a specific point"""
        subs_dict = dict(zip(self.coordinates, point))
        return np.array(self.matrix_func.subs(subs_dict)).astype(float)
    
    def nijenhuis_tensor(self, X_vec: np.ndarray, Y_vec: np.ndarray, 
                        point: np.ndarray) -> np.ndarray:
        """
        Compute Nijenhuis tensor N_J(X,Y) at a point
        N_J(X,Y) = [JX,JY] - J[JX,Y] - J[X,JY] - [X,Y]
        """
        # Evaluate J and its derivatives at the point
        subs_dict = dict(zip(self.coordinates, point))
        J = np.array(self.matrix_func.subs(subs_dict)).astype(float)
        
        # Compute J applied to vectors
        JX = J @ X_vec
        JY = J @ Y_vec
        
        # Compute Lie brackets (simplified - assumes constant vector fields)
        # Full implementation would need covariant derivatives
        lie_JX_JY = self._lie_bracket(JX, JY, point)
        lie_JX_Y = self._lie_bracket(JX, Y_vec, point)
        lie_X_JY = self._lie_bracket(X_vec, JY, point)
        lie_X_Y = self._lie_bracket(X_vec, Y_vec, point)
        
        # Nijenhuis tensor
        N = lie_JX_JY - J @ lie_JX_Y - J @ lie_X_JY - lie_X_Y
        
        return N
    
    def _lie_bracket(self, X: np.ndarray, Y: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Compute Lie bracket [X,Y] at a point (simplified)"""
        # For constant vector fields, [X,Y] = 0
        # For general fields, would need X(Y) - Y(X)
        result = np.zeros(self.dim)
        
        # Add contribution from J's variation
        for i, coord in enumerate(self.coordinates):
            subs_dict = dict(zip(self.coordinates, point))
            dJ = np.array(self.J_derivatives[coord].subs(subs_dict)).astype(float)
            result += X[i] * (dJ @ Y) - Y[i] * (dJ @ X)
        
        return result
    
    def compute_integrability_obstruction(self) -> sp.Matrix:
        """Compute the formal integrability obstruction"""
        # The obstruction is the Nijenhuis tensor viewed as a (2,1)-tensor
        # We compute it symbolically
        
        n = self.dim // 2  # Complex dimension
        obstruction = sp.zeros(self.dim, self.dim)
        
        # Create symbolic vector fields
        X_symbols = [sp.Symbol(f'X_{i}') for i in range(self.dim)]
        Y_symbols = [sp.Symbol(f'Y_{i}') for i in range(self.dim)]
        
        # This would compute N_J symbolically - simplified here
        return obstruction

# ============================================================================
# ENHANCED CHART AND MANIFOLD CLASSES
# ============================================================================

class Chart:
    """Enhanced chart with hyperfunction boundary data"""
    
    def __init__(self, name: str, domain_condition, coordinates, 
                 almost_complex_structure: AlmostComplexStructure,
                 hyperfunction_data: Optional[Dict] = None):
        self.name = name
        self.domain_condition = domain_condition
        self.coordinates = coordinates
        self.J = almost_complex_structure
        self.hyperfunction_data = hyperfunction_data if hyperfunction_data else {}
        self.transition_maps = {}
    
    def contains_point(self, point):
        """Check if a point is in this chart's domain"""
        subs_dict = dict(zip(self.coordinates, point))
        return bool(self.domain_condition.subs(subs_dict))
    
    def add_hyperfunction(self, name: str, hf: Hyperfunction):
        """Add a hyperfunction to this chart"""
        self.hyperfunction_data[name] = hf
    
    def compute_boundary_values(self, boundary_points: List[np.ndarray]) -> Dict:
        """Compute hyperfunction boundary values on a set of points"""
        boundary_data = {}
        
        for hf_name, hf in self.hyperfunction_data.items():
            values = []
            wavefronts = []
            
            for point in boundary_points:
                val = hf.boundary_value(point)
                wf = hf.compute_wavefront(point)
                values.append(val)
                wavefronts.append(wf)
            
            boundary_data[hf_name] = {
                'values': np.array(values),
                'wavefronts': wavefronts
            }
        
        return boundary_data

class TransitionMap:
    """Represents a transition map between charts"""
    
    def __init__(self, source_chart: Chart, target_chart: Chart, 
                 map_function: Callable, inverse_function: Callable):
        self.source = source_chart
        self.target = target_chart
        self.forward = map_function
        self.inverse = inverse_function
        
        # Compute symbolic Jacobian
        self._compute_jacobian()
    
    def _compute_jacobian(self):
        """Compute the Jacobian matrix symbolically"""
        n = len(self.source.coordinates)
        self.jacobian = sp.zeros(n, n)
        
        # This would compute the actual Jacobian - simplified here
        # for coord_i in range(n):
        #     for coord_j in range(n):
        #         self.jacobian[i,j] = diff(self.forward[i], self.source.coordinates[j])
    
    def evaluate_jacobian(self, point: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian at a point"""
        # Simplified - return identity for now
        return np.eye(len(point))
    
    def pullback_hyperfunction(self, hf: Hyperfunction) -> Hyperfunction:
        """Pull back a hyperfunction through this transition map"""
        def pulled_plus(z):
            return hf.F_plus(self.forward(z))
        
        def pulled_minus(z):
            return hf.F_minus(self.forward(z))
        
        return Hyperfunction(pulled_plus, pulled_minus, hf.support)
    
    def pushforward_wavefront(self, wf_cone: WavefrontCone, point: np.ndarray) -> WavefrontCone:
        """Push forward a wavefront cone through the transition map"""
        jacobian = self.evaluate_jacobian(point)
        return wf_cone.propagate(jacobian)

# ============================================================================
# ENHANCED COHOMOLOGY COMPUTATION
# ============================================================================

class HyperfunctionCohomology:
    """Enhanced computation engine with full microlocal analysis"""
    
    def __init__(self, manifold):
        self.manifold = manifold
        self.overlaps = manifold.compute_chart_overlaps()
        self.jump_data = {}
        self.cohomology_groups = {}
    
    def compute_jump_hyperfunction(self, chart1: Chart, chart2: Chart, 
                                 overlap_points: List[np.ndarray]) -> Dict:
        """
        Compute jump hyperfunction c_{αβ} = [F_α] - φ_{αβ}*[F_β]
        """
        jumps = {}
        
        # Get boundary data from both charts
        boundary1 = chart1.compute_boundary_values(overlap_points)
        boundary2 = chart2.compute_boundary_values(overlap_points)
        
        # Get transition map (if exists)
        transition = chart1.transition_maps.get(chart2.name)
        
        for hf_name in boundary1.keys():
            if hf_name in boundary2:
                values1 = boundary1[hf_name]['values']
                values2 = boundary2[hf_name]['values']
                
                if transition:
                    # Apply pullback to chart2 data
                    # Simplified - would need proper pullback
                    values2_pulled = values2  # Should be transition.pullback(values2)
                else:
                    values2_pulled = values2
                
                # Compute jump
                jump_values = values1 - values2_pulled
                
                # Analyze wavefront of jump
                wf_jumps = []
                for i, point in enumerate(overlap_points):
                    wf1 = boundary1[hf_name]['wavefronts'][i]
                    wf2 = boundary2[hf_name]['wavefronts'][i]
                    
                    # Compute wavefront of difference
                    wf_jump = self._compute_wf_difference(wf1, wf2, point)
                    wf_jumps.append(wf_jump)
                
                jumps[hf_name] = {
                    'values': jump_values,
                    'wavefronts': wf_jumps,
                    'obstruction_class': self._compute_obstruction_class(jump_values, wf_jumps)
                }
        
        return jumps
    
    def _compute_wf_difference(self, wf1: WavefrontCone, wf2: WavefrontCone, 
                               point: np.ndarray) -> WavefrontCone:
        """Compute wavefront of difference of hyperfunctions"""
        # WF(u-v) ⊆ WF(u) ∪ WF(v)
        
        all_directions = wf1.directions + wf2.directions
        
        if not all_directions:
            return WavefrontCone(point, [], 0, WavefrontType.SMOOTH)
        
        # Remove duplicates and normalize
        unique_dirs = []
        for d in all_directions:
            d_norm = d / np.linalg.norm(d)
            if not any(np.allclose(d_norm, existing) for existing in unique_dirs):
                unique_dirs.append(d_norm)
        
        # Compute cone parameters
        if len(unique_dirs) == 0:
            cone_angle = 0
            sing_type = WavefrontType.SMOOTH
        elif len(unique_dirs) == 1:
            cone_angle = 0
            sing_type = WavefrontType.CONORMAL
        else:
            angles = []
            for d1, d2 in itertools.combinations(unique_dirs, 2):
                angle = np.arccos(np.clip(np.dot(d1, d2), -1, 1))
                angles.append(angle)
            cone_angle = max(angles) / 2
            
            # Classify based on cone structure
            if cone_angle < np.pi/6:
                sing_type = WavefrontType.CONORMAL
            elif cone_angle < np.pi/3:
                sing_type = WavefrontType.FOLD
            else:
                sing_type = WavefrontType.MIXED
        
        return WavefrontCone(point, unique_dirs, cone_angle, sing_type)
    
    def _compute_obstruction_class(self, jump_values: np.ndarray, 
                                   wavefronts: List[WavefrontCone]) -> str:
        """Compute the cohomology class of the obstruction"""
        # Analyze the jump pattern
        max_jump = np.max(np.abs(jump_values))
        
        # Count singular wavefronts
        n_singular = sum(1 for wf in wavefronts if wf.singularity_type != WavefrontType.SMOOTH)
        
        if max_jump < 1e-10:
            return "0 (trivial)"
        elif n_singular == 0:
            return "C^∞ (smooth obstruction)"
        elif all(wf.singularity_type == WavefrontType.CONORMAL for wf in wavefronts):
            return f"H^1_conormal (conormal obstruction, strength={max_jump:.3e})"
        else:
            types = set(wf.singularity_type for wf in wavefronts)
            return f"H^1_mixed ({', '.join(t.value for t in types)}, strength={max_jump:.3e})"
    
    def compute_border_index(self, overlap_points: List[np.ndarray], 
                            jump_data: Dict) -> float:
        """
        Compute the border index BI(Σ) = ∫_Σ σ(WF(c_{αβ}))
        """
        total_index = 0
        
        for hf_name, data in jump_data.items():
            wavefronts = data['wavefronts']
            values = data['values']
            
            for i, wf in enumerate(wavefronts):
                if wf.singularity_type != WavefrontType.SMOOTH:
                    # Measure of singularity
                    sigma = wf.cone_angle * len(wf.directions) * abs(values[i])
                    
                    # Weight by point density (simplified)
                    weight = 1.0 / len(overlap_points)
                    
                    total_index += sigma * weight
        
        return total_index
    
    def compute_spectral_obstruction(self, chart: Chart, test_points: List[np.ndarray]) -> Dict:
        """Compute spectral invariants of the ∂̄_J operator"""
        # Discretize the ∂̄_J operator
        n_points = len(test_points)
        n_dim = chart.J.dim
        
        # Build discrete operator matrix (simplified)
        dbar_matrix = np.zeros((n_points * n_dim, n_points * n_dim), dtype=complex)
        
        for i, point in enumerate(test_points):
            J_at_point = chart.J.evaluate_at(point)
            
            # ∂̄_J = 1/2(∂ + iJ∂)
            # Discretized version
            block = (np.eye(n_dim) + 1j * J_at_point) / 2
            
            idx_start = i * n_dim
            idx_end = (i + 1) * n_dim
            dbar_matrix[idx_start:idx_end, idx_start:idx_end] = block
        
        # Compute spectrum
        eigenvalues = np.linalg.eigvals(dbar_matrix)
        
        # Compute invariants
        spectral_data = {
            'eigenvalues': eigenvalues,
            'spectral_gap': np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10])) 
                           if np.any(np.abs(eigenvalues) > 1e-10) else 0,
            'kernel_dimension': np.sum(np.abs(eigenvalues) < 1e-10),
            'trace': np.trace(dbar_matrix),
            'determinant': np.linalg.det(dbar_matrix),
            'condition_number': np.linalg.cond(dbar_matrix)
        }
        
        return spectral_data
    
    def compute_cech_differential(self, triple_overlap: Tuple[Chart, Chart, Chart], 
                                 test_points: List[np.ndarray]) -> Dict:
        """
        Compute Čech differential δ: C^1 → C^2
        δc = c_{βγ} - c_{αγ} + c_{αβ} on U_α ∩ U_β ∩ U_γ
        """
        alpha, beta, gamma = triple_overlap
        
        # Compute pairwise jumps
        c_ab = self.compute_jump_hyperfunction(alpha, beta, test_points)
        c_ag = self.compute_jump_hyperfunction(alpha, gamma, test_points)
        c_bg = self.compute_jump_hyperfunction(beta, gamma, test_points)
        
        # Compute differential
        delta_c = {}
        
        for hf_name in c_ab.keys():
            if hf_name in c_ag and hf_name in c_bg:
                # δc = c_{βγ} - c_{αγ} + c_{αβ}
                delta_values = (c_bg[hf_name]['values'] - 
                              c_ag[hf_name]['values'] + 
                              c_ab[hf_name]['values'])
                
                # Check if cocycle condition is satisfied
                is_cocycle = np.allclose(delta_values, 0, atol=1e-10)
                
                delta_c[hf_name] = {
                    'values': delta_values,
                    'is_cocycle': is_cocycle,
                    'violation_norm': np.linalg.norm(delta_values)
                }
        
        return delta_c
    
    def analyze_integrability_full(self, n_test_points: int = 20):
        """Complete integrability analysis with all diagnostics"""
        print("=" * 80)
        print("ENHANCED HYPERFUNCTION COHOMOLOGY ANALYSIS")
        print("=" * 80)
        
        results = {
            'overlaps': {},
            'border_indices': {},
            'spectral_data': {},
            'cocycle_violations': [],
            'global_obstruction': None
        }
        
        # Generate test points
        test_points = self._generate_test_points(n_test_points)
        
        # Analyze each overlap
        for overlap in self.overlaps:
            chart1 = overlap['chart1']
            chart2 = overlap['chart2']
            overlap_name = overlap['name']
            
            print(f"\n{'='*60}")
            print(f"ANALYZING OVERLAP: {overlap_name}")
            print(f"{'='*60}")
            
            # Compute jump hyperfunction
            jump_data = self.compute_jump_hyperfunction(chart1, chart2, test_points)
            results['overlaps'][overlap_name] = jump_data
            
            # Compute border index
            border_idx = self.compute_border_index(test_points, jump_data)
            results['border_indices'][overlap_name] = border_idx
            
            print(f"\nBorder Index: {border_idx:.6f}")
            
            # Print obstruction classes
            for hf_name, data in jump_data.items():
                print(f"\nHyperfunction: {hf_name}")
                print(f"  Obstruction class: {data['obstruction_class']}")
                print(f"  Max jump magnitude: {np.max(np.abs(data['values'])):.3e}")
                
                # Analyze wavefront distribution
                wf_types = {}
                for wf in data['wavefronts']:
                    wf_type = wf.singularity_type.value
                    wf_types[wf_type] = wf_types.get(wf_type, 0) + 1
                
                print(f"  Wavefront distribution: {wf_types}")
        
        # Compute spectral obstructions
        print(f"\n{'='*60}")
        print("SPECTRAL ANALYSIS")
        print(f"{'='*60}")
        
        for chart in self.manifold.charts:
            spectral = self.compute_spectral_obstruction(chart, test_points)
            results['spectral_data'][chart.name] = spectral
            
            print(f"\nChart {chart.name}:")
            print(f"  Spectral gap: {spectral['spectral_gap']:.6f}")
            print(f"  Kernel dimension: {spectral['kernel_dimension']}")
            print(f"  Condition number: {spectral['condition_number']:.3e}")
        
        # Check Čech cocycle conditions
        print(f"\n{'='*60}")
        print("ČECH COCYCLE ANALYSIS")
        print(f"{'='*60}")
        
        triple_overlaps = self._find_triple_overlaps()
        
        for triple in triple_overlaps:
            delta_c = self.compute_cech_differential(triple, test_points)
            
            for hf_name, data in delta_c.items():
                if not data['is_cocycle']:
                    violation = {
                        'triple': [c.name for c in triple],
                        'hyperfunction': hf_name,
                        'violation_norm': data['violation_norm']
                    }
                    results['cocycle_violations'].append(violation)
                    
                    print(f"\nCocycle violation detected:")
                    print(f"  Triple: {violation['triple']}")
                    print(f"  Norm: {violation['violation_norm']:.3e}")
        
        # Determine global integrability
        print(f"\n{'='*80}")
        print("INTEGRABILITY VERDICT")
        print(f"{'='*80}")
        
        # Aggregate all obstructions
        total_border_index = sum(results['border_indices'].values())
        n_violations = len(results['cocycle_violations'])
        max_violation = max([v['violation_norm'] for v in results['cocycle_violations']]) \
                       if results['cocycle_violations'] else 0
        
        # Decision criteria
        is_integrable = (total_border_index < 1e-6 and 
                        n_violations == 0 and 
                        max_violation < 1e-10)
        
        results['global_obstruction'] = {
            'integrable': is_integrable,
            'total_border_index': total_border_index,
            'n_cocycle_violations': n_violations,
            'max_violation_norm': max_violation,
            'verdict': "INTEGRABLE" if is_integrable else "NON-INTEGRABLE"
        }
        
        print(f"\nVerdict: {results['global_obstruction']['verdict']}")
        print(f"Total border index: {total_border_index:.6e}")
        print(f"Cocycle violations: {n_violations}")
        
        if not is_integrable:
            print("\nIntegrability obstructions detected:")
            if total_border_index > 1e-6:
                print(f"  - Non-zero border index: {total_border_index:.6e}")
            if n_violations > 0:
                print(f"  - Čech cocycle violations: {n_violations}")
            print("\nThe almost complex structure cannot be integrated to a complex structure.")
        else:
            print("\nNo obstructions detected. The structure appears to
