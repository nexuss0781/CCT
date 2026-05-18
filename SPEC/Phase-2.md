# Phase 2: Spectral Solver Implementation and Validation

## Overview

**Duration**: 5 weeks  
**Goal**: Implement high-performance spectral solvers for causal wave equations with O(n log n) complexity.  
**Exit Criteria**: Solver passes all accuracy tests; achieves target performance on GPU/TPU hardware.

---

## Scope

### In Scope
1. FFT-based propagator implementation in JAX
2. Time-stepping schemes (explicit, implicit, semi-implicit)
3. Learnable spectral kernels with parameterization
4. Boundary condition handling (periodic, Dirichlet, Neumann, absorbing)
5. Automatic differentiation for gradient computation
6. Mixed-precision training support
7. Numerical stability analysis

### Out of Scope
- Manifold data structure modifications (Phase 1 complete)
- Semantic embedding learning (Phase 3)
- Distributed multi-GPU scaling (Phase 6)

---

## Mathematical Specifications

### 2.1 Causal Wave Equation

**Governing PDE**:
```
□φ(x,t) + V(x;θ)φ(x,t) = J(x,t)

where:
  □ = ∂_t² - c²Δ is the wave operator (d'Alembertian)
  V(x;θ) is learnable potential parameterized by θ
  J(x,t) is source term (input events)
  φ(x,t) is causal field (hidden representation)
```

**Initial Conditions**:
```
φ(x,0) = φ₀(x)
∂φ/∂t(x,0) = ψ₀(x)
```

**Boundary Conditions** (configurable):
- **Periodic**: φ(0,t) = φ(L,t)
- **Dirichlet**: φ(boundary,t) = 0
- **Neumann**: ∂φ/∂n(boundary,t) = 0
- **Absorbing (PML)**: Perfectly Matched Layer for open boundaries

### 2.2 Frequency Domain Solution

**Fourier Transform Convention**:
```
φ̂(k) = ∫ φ(x) e^{-ik·x} dx
φ(x) = (2π)^{-d} ∫ φ̂(k) e^{ik·x} dk
```

**Green's Function in Fourier Space**:
```
Ĝ(ω,k) = 1 / (ω² - c²|k|² - V̂(k) + iε)

where ε → 0⁺ ensures causality (retarded Green's function)
```

**Solution Formula**:
```
φ̂(ω,k) = Ĝ(ω,k) · Ĵ(ω,k)
φ(x,t) = ℱ^{-1}[φ̂(ω,k)]
```

### 2.3 Dispersion Relations

**Numerical Dispersion** (finite difference):
```
ω_num² = (4/Δt²) sin²(ωΔt/2) - (4c²/Δx²) Σ_i sin²(k_i Δx/2)

Error: |ω_num - ω_exact| / ω_exact < tolerance
```

**Stability Condition** (CFL):
```
CFL number: ν = cΔt/Δx
Explicit scheme stable iff: ν ≤ 1/√d

For d=4: Δt ≤ Δx / 2
```

### 2.4 Learnable Potentials

**Parameterization Options**:

**A. Spectral Filter** (frequency-dependent):
```
V̂(k;θ) = Σ_m θ_m · B_m(k)

where B_m(k) are basis functions:
  - Gaussians: exp(-|k-k_m|²/σ_m²)
  - Wavelets: ψ((k-k_m)/s_m)
  - Polynomials: |k|^{2m}
```

**B. Real-Space Potential** (local):
```
V(x;θ) = MLP_θ(ρ(x))

where ρ(x) is event density field
```

**C. Non-local Operator** (integral kernel):
```
(Vφ)(x) = ∫ K(x,y;θ) φ(y) dy

with K(x,y;θ) = K(|x-y|;θ) for translation invariance
```

---

## Technical Specifications

### 3.1 JAX Implementation Architecture

**Module Structure**:
```
spectral_solver/
├── __init__.py
├── fft_propagator.py      # Core FFT operations
├── time_stepper.py        # Time integration schemes
├── kernels.py             # Learnable spectral kernels
├── boundary_conditions.py # BC implementations
├── stability.py           # CFL checking, dispersion analysis
└── gradients.py           # Custom VJP rules
```

### 3.2 FFT Propagator

**Core Class**:
```python
@jax.jit
class FFTPropagator:
    def __init__(
        self,
        shape: Tuple[int, ...],
        spacing: float = 1.0,
        c: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.shape = shape
        self.spacing = spacing
        self.c = c
        self.freq_grid = self._build_freq_grid()
        
    def _build_freq_grid(self) -> Tuple[jnp.ndarray, ...]:
        """Construct frequency coordinate grids."""
        freq_axes = [jnp.fft.fftfreq(n, d=self.spacing) for n in self.shape]
        return jnp.meshgrid(*freq_axes, indexing='ij')
    
    def green_function(
        self,
        omega: jnp.ndarray,
        potential: Optional[jnp.ndarray] = None,
        epsilon: float = 1e-6,
    ) -> jnp.ndarray:
        """Compute retarded Green's function in frequency space."""
        k_sq = sum(k**2 for k in self.freq_grid)
        denominator = omega**2 - self.c**2 * k_sq
        if potential is not None:
            denominator -= potential
        return 1.0 / (denominator + 1j * epsilon)
    
    def propagate(
        self,
        source: jnp.ndarray,
        potential: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply Green's function to source field."""
        # Forward FFT
        source_hat = jnp.fft.fftn(source)
        
        # Construct frequency-domain source (add time dimension)
        # Assuming monochromatic or use full (ω,k) grid
        
        # Apply Green's function
        G_hat = self.green_function(omega, potential)
        field_hat = G_hat * source_hat
        
        # Inverse FFT
        return jnp.fft.ifftn(field_hat).real
```

**Complexity Analysis**:
- FFT: O(n log n) where n = ∏ shape[i]
- Pointwise multiplication: O(n)
- IFFT: O(n log n)
- **Total**: O(n log n) ✓

### 3.3 Time Stepping Schemes

**Scheme A: Leapfrog (Explicit, 2nd Order)**:
```python
@jax.jit
def leapfrog_step(phi, psi, source, dt, potential):
    """
    phi: field at time t
    psi: ∂φ/∂t at time t
    source: J(x,t)
    """
    # Compute Laplacian via FFT
    laplacian_phi = -sum(k**2 for k in freq_grid) * phi  # in Fourier space
    
    # Update velocity (half step)
    psi_new = psi + dt * (laplacian_phi - potential * phi + source)
    
    # Update field (full step)
    phi_new = phi + dt * psi_new
    
    return phi_new, psi_new
```

**Scheme B: Crank-Nicolson (Implicit, 2nd Order)**:
```python
@jax.jit
def crank_nicolson_step(phi, psi, source, dt, potential):
    """
    Unconditionally stable but requires linear solve.
    """
    # Formulate as (I - dt/2 * L) φ^{n+1} = (I + dt/2 * L) φ^n
    # Solve via conjugate gradient in Fourier space
    pass
```

**Scheme C: Runge-Kutta 4 (Explicit, 4th Order)**:
```python
@jax.jit
def rk4_step(phi, psi, source_fn, dt, potential):
    """
    Higher accuracy at cost of 4 function evaluations per step.
    """
    def derivatives(phi, psi, t):
        laplacian = compute_laplacian_fft(phi)
        dphi_dt = psi
        dpsi_dt = laplacian - potential * phi + source_fn(t)
        return dphi_dt, dpsi_dt
    
    # Standard RK4 stages
    k1_phi, k1_psi = derivatives(phi, psi, t)
    k2_phi, k2_psi = derivatives(phi + dt/2*k1_phi, psi + dt/2*k1_psi, t + dt/2)
    k3_phi, k3_psi = derivatives(phi + dt/2*k2_phi, psi + dt/2*k2_psi, t + dt/2)
    k4_phi, k4_psi = derivatives(phi + dt*k3_phi, psi + dt*k3_psi, t + dt)
    
    phi_new = phi + dt/6 * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    psi_new = psi + dt/6 * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi)
    
    return phi_new, psi_new
```

### 3.4 Boundary Conditions

**Implementation via Ghost Cells**:
```python
def apply_boundary_conditions(field, bc_type):
    match bc_type:
        case 'periodic':
            # Natural for FFT (inherent periodicity)
            return field
            
        case 'dirichlet':
            # Zero padding
            return field.at[..., 0, :].set(0).at[..., -1, :].set(0)
            
        case 'neumann':
            # Mirror reflection for zero normal derivative
            return field.at[..., 0, :].set(field.at[..., 1, :].get())
            
        case 'absorbing':
            # Perfectly Matched Layer (PML)
            return apply_pml(field)
```

**PML Implementation**:
```python
@jax.jit
def apply_pml(field, pml_width=10, sigma_max=1.0):
    """
    Absorbing boundary layer to simulate open domain.
    """
    # Construct absorption profile
    def sigma_profile(i, width, max_val):
        return jnp.where(i < width, 
                        max_val * (i / width)**2,
                        0.0)
    
    # Apply damping in frequency space
    damping = construct_pml_tensor(field.shape, pml_width, sigma_max)
    return field * damping
```

### 3.5 Automatic Differentiation

**Custom VJP for FFT Operations**:
```python
@jax.custom_vjp
def spectral_solve(source, potential_params):
    """Forward solve with custom gradient."""
    potential = build_potential(potential_params)
    return fft_propagator.propagate(source, potential)

def spectral_solve_fwd(source, potential_params):
    """Forward pass with cached intermediates."""
    potential = build_potential(potential_params)
    result = spectral_solve(source, potential_params)
    return result, (source, potential, result)

def spectral_solve_bwd(residuals, grad_output):
    """Backward pass via adjoint method."""
    source, potential, field = residuals
    
    # Adjoint equation: same operator, different source
    adjoint_source = grad_output  # or more complex depending on loss
    adjoint_field = spectral_solve(adjoint_source, potential_params)
    
    # Gradient w.r.t. potential parameters
    grad_params = compute_param_gradients(field, adjoint_field, potential_params)
    
    return None, grad_params

spectral_solve.defvjp(spectral_solve_fwd, spectral_solve_bwd)
```

---

## API Contract

### 4.1 Python Interface

```python
class SpectralSolver:
    def __init__(
        self,
        manifold_shape: Tuple[int, ...],
        spacing: float = 1.0,
        wave_speed: float = 1.0,
        time_step: Optional[float] = None,  # Auto-compute from CFL if None
        scheme: str = 'leapfrog',  # ['leapfrog', 'rk4', 'crank_nicolson']
        boundary_conditions: str = 'periodic',
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize spectral solver."""
        pass
    
    def initialize_field(
        self,
        initial_phi: jnp.ndarray,
        initial_psi: jnp.ndarray,
    ) -> FieldState:
        """Set initial conditions φ(x,0) and ∂φ/∂t(x,0)."""
        pass
    
    def step(
        self,
        state: FieldState,
        source: jnp.ndarray,
        potential_params: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> FieldState:
        """Advance solution by one time step."""
        pass
    
    def solve_trajectory(
        self,
        sources: Sequence[jnp.ndarray],
        potential_params: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Sequence[FieldState]:
        """Solve for entire time trajectory."""
        pass
    
    def check_stability(self) -> StabilityReport:
        """Verify CFL condition and numerical stability."""
        pass
    
    def dispersion_analysis(self) -> DispersionDiagram:
        """Compute numerical dispersion relation."""
        pass
```

### 4.2 Kernel Parameterization

```python
class SpectralKernel:
    def __init__(
        self,
        param_type: str,  # ['gaussian_basis', 'polynomial', 'neural']
        num_params: int,
        init_params: Optional[Dict] = None,
    ):
        self.params = self.initialize_parameters(param_type, num_params, init_params)
    
    def build_potential(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Construct potential V(x) or V̂(k) from parameters."""
        pass
    
    def param_to_array(self) -> jnp.ndarray:
        """Flatten parameters for optimization."""
        pass
    
    def array_to_param(self, flat: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Reconstruct parameters from flat array."""
        pass
```

---

## Testing Strategy

### 5.1 Accuracy Tests

**Test 1: Analytic Solution Comparison**
```python
def test_gaussian_wave_packet():
    """Compare against known analytic solution."""
    # Initial condition: Gaussian wave packet
    phi0 = gaussian(x, x0=0.5, sigma=0.1)
    psi0 = -c * d_dx(gaussian(x, x0=0.5, sigma=0.1))  # Right-moving
    
    # Evolve numerically
    solver = SpectralSolver(shape=(256,), spacing=0.01)
    trajectory = solver.solve_trajectory(sources=[zeros])
    
    # Compare to analytic solution at t=T
    phi_analytic = gaussian(x - c*T, x0=0.5, sigma=0.1)
    error = jnp.max(jnp.abs(trajectory[-1].phi - phi_analytic))
    
    assert error < 1e-3, f"Accuracy test failed: error={error}"
```

**Test 2: Energy Conservation**
```python
def test_energy_conservation():
    """Verify energy conserved in absence of sources/dissipation."""
    E_initial = compute_total_energy(phi0, psi0)
    
    trajectory = solver.solve_trajectory(sources=[zeros]*100)
    energies = [compute_total_energy(state.phi, state.psi) 
                for state in trajectory]
    
    relative_drift = max(energies) - min(energies) / E_initial
    assert relative_drift < 1e-4, f"Energy drift too large: {relative_drift}"
```

### 5.2 Convergence Tests

**Grid Convergence**:
```python
def test_spatial_convergence():
    """Verify 2nd-order spatial accuracy."""
    resolutions = [64, 128, 256, 512]
    errors = []
    
    for N in resolutions:
        solver = SpectralSolver(shape=(N,))
        # ... solve and compare to reference ...
        errors.append(error)
    
    # Check convergence rate
    rates = [np.log(errors[i]/errors[i+1]) / np.log(2) 
             for i in range(len(errors)-1)]
    assert np.mean(rates) > 1.9, f"Convergence rate {np.mean(rates)} < 2.0"
```

**Temporal Convergence**:
```python
def test_temporal_convergence():
    """Verify order of time-stepping scheme."""
    time_steps = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    
    for dt in time_steps:
        solver = SpectralSolver(shape=(256,), time_step=dt)
        # ... solve and compare to reference ...
        errors.append(error)
    
    # Expected rate depends on scheme (2 for leapfrog, 4 for RK4)
    expected_rate = 2.0 if solver.scheme == 'leapfrog' else 4.0
    # ... verify convergence rate ...
```

### 5.3 Performance Benchmarks

**Benchmark Suite**:
```python
@benchmark
def bench_fft_propagation(benchmark):
    """Measure FFT propagation throughput."""
    solver = SpectralSolver(shape=(512, 512))
    source = jnp.random.normal(key, shape=(512, 512))
    
    result = benchmark(solver.propagate, source)
    return result

@benchmark  
def bench_time_evolution(benchmark):
    """Measure long-time evolution performance."""
    solver = SpectralSolver(shape=(256, 256, 256))
    
    def evolve_100_steps():
        state = solver.initialize_field(phi0, psi0)
        for _ in range(100):
            state = solver.step(state, source=zeros)
        return state
    
    result = benchmark(evolve_100_steps)
    return result
```

**Performance Targets**:
- Single FFT propagation: < 1ms for 512³ grid on A100
- Time evolution: > 100 steps/sec for 256³ grid
- Gradient computation: < 2× forward pass time
- Memory bandwidth utilization: > 80% of theoretical peak

---

## Deliverables

### Week 1: Core FFT Infrastructure
- [ ] `FFTPropagator` class with Green's function
- [ ] Frequency grid construction utilities
- [ ] Basic propagation tests

### Week 2: Time-Stepping Schemes
- [ ] Leapfrog integrator
- [ ] RK4 integrator
- [ ] Crank-Nicolson (optional)
- [ ] CFL checking utilities

### Week 3: Boundary Conditions & Kernels
- [ ] Periodic, Dirichlet, Neumann BCs
- [ ] PML absorbing layers
- [ ] Learnable spectral kernel classes
- [ ] Parameter initialization strategies

### Week 4: Automatic Differentiation
- [ ] Custom VJP rules for spectral solve
- [ ] Gradient verification (finite differences)
- [ ] Integration with JAX optimizers

### Week 5: Testing & Optimization
- [ ] Complete accuracy test suite
- [ ] Convergence tests
- [ ] Performance benchmarks
- [ ] Mixed-precision support (FP16/BF16)
- [ ] Documentation

---

## Acceptance Criteria

### Functional Requirements
- ✅ All time-stepping schemes implemented and tested
- ✅ All boundary condition types functional
- ✅ Learnable kernels integrate with JAX optimizers
- ✅ Gradients verified to machine precision

### Numerical Requirements
- ✅ Spatial accuracy: 2nd-order minimum
- ✅ Temporal accuracy: matches scheme order
- ✅ Energy conservation: < 0.01% drift over 1000 steps
- ✅ CFL condition enforced with safety margin (0.9× limit)

### Performance Requirements
- ✅ Throughput: > 100M cells/sec on A100 GPU
- ✅ Scaling: Strong scaling efficiency > 80% up to 4 GPUs
- ✅ Memory: < 8 bytes per cell (single precision)
- ✅ No O(n²) operations in hot path

### Robustness Requirements
- ✅ Handles NaN/Inf gracefully (returns structured error)
- ✅ Stable for 10⁶+ time steps
- ✅ Works with random initialization
- ✅ Deterministic with fixed PRNG seed

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FFT aliasing artifacts | High | Medium | Anti-aliasing filters; oversampling |
| CFL violation crashes | High | Low | Runtime checks; adaptive time-stepping |
| Gradient explosion/vanishing | Medium | Medium | Gradient clipping; careful initialization |
| GPU memory exhaustion | Medium | High | Chunked processing; out-of-core options |
| PML reflection errors | Low | Medium | Tuned absorption profiles; validation tests |

---

## Dependencies

### Python Packages
- `jax` (v0.4+): Core numerical library
- `jaxlib` (v0.4+): XLA backend
- `numpy` (v1.24+): Array operations
- `scipy` (v1.10+): Special functions
- `pytest` (v7+): Testing framework
- `pytest-benchmark` (v4+): Performance testing

### Hardware Requirements
- NVIDIA A100/V100 GPU (or equivalent)
- CUDA 11.8+
- 32GB GPU memory minimum
- NVLink for multi-GPU configurations

---

## Success Metrics

1. **Accuracy**: Relative error < 10⁻³ vs analytic solutions
2. **Convergence**: Observed order matches theoretical order
3. **Performance**: Meets throughput targets on target hardware
4. **Stability**: No blow-ups in 10⁶-step stress test
5. **Gradient Quality**: Finite-difference check passes to 10⁻⁵

---

*Phase Owner: Numerical Methods Team*  
*Review Gate: End of Week 5*  
*Next Phase: Phase 3 - Semantic Geometry and Embedding Learning*
