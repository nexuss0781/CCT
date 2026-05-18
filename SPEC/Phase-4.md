# Phase 4: Resonance Engine and Mode Decomposition

## Overview

**Duration**: 6 weeks  
**Goal**: Implement spectral decomposition of causal fields into eigenmodes, enabling efficient reasoning via mode coupling.  
**Exit Criteria**: System achieves O(log n) compression; demonstrates multi-hop reasoning through mode interactions.

---

## Scope

### In Scope
1. Laplace-Beltrami operator discretization
2. Eigendecomposition (full and partial)
3. Lanczos/Arnoldi iteration for dominant modes
4. Mode coupling tensors for inference
5. Spectral attention mechanism
6. Temporal dynamics of mode amplitudes
7. Compression via mode truncation

### Out of Scope
- Topological memory structures (Phase 5)
- End-to-end task fine-tuning (Phase 6)
- Multi-manifold architectures (Phase 7)

---

## Mathematical Specifications

### 4.1 Laplace-Beltrami Operator

**Definition on Manifold**:
```
Δ_g f = div(grad f) = (1/√|g|) ∂_μ (√|g| g^{μν} ∂_ν f)

where:
  g = det(g_μν) is metric determinant
  g^{μν} is inverse metric
  grad f = g^{μν} ∂_ν f ∂_μ (gradient vector field)
  div X = (1/√|g|) ∂_μ (√|g| X^μ) (divergence)
```

**Spectral Properties**:
```
Eigenvalue problem: Δ_g ψ_n = λ_n ψ_n

Properties:
  - Spectrum is real and non-positive: λ_n ≤ 0
  - Eigenfunctions are orthogonal: ⟨ψ_m, ψ_n⟩ = δ_mn
  - Form complete basis: f = Σ_n ⟨f, ψ_n⟩ ψ_n
  - Weyl's law: N(λ) ~ C_d Vol(M) |λ|^{d/2}
```

**Discrete Formulation** (finite differences):
```python
@jax.jit
def laplace_beltrami_discrete(field, metric):
    """
    Compute Δ_g f on discrete grid.
    
    Args:
        field: scalar field f(x) on grid
        metric: g_μν(x) at each point
    
    Returns:
        Δ_g f on same grid
    """
    d = field.ndim
    sqrt_g = jnp.sqrt(jnp.abs(jnp.linalg.det(metric)))
    g_inv = jnp.linalg.inv(metric)
    
    # Compute √|g| g^{μν} ∂_ν f
    flux = jnp.zeros_like(field)
    for mu in range(d):
        for nu in range(d):
            # Central difference for ∂_ν f
            df_dn = central_difference(field, axis=nu)
            flux += sqrt_g * g_inv[..., mu, nu] * df_dn
    
    # Compute (1/√|g|) ∂_μ (flux^μ)
    div_flux = jnp.zeros_like(field)
    for mu in range(d):
        div_flux += (1 / sqrt_g) * central_difference(flux, axis=mu)
    
    return div_flux
```

### 4.2 Eigendecomposition

**Full Decomposition** (small manifolds):
```python
def compute_eigenmodes(manifold, num_modes=None):
    """
    Compute eigenmodes of Laplace-Beltrami operator.
    
    For small grids (< 10k points), use dense eigendecomposition.
    For larger grids, use iterative methods (Lanczos).
    """
    # Construct discrete Laplacian matrix L
    L = build_laplacian_matrix(manifold)
    
    if num_modes is None or num_modes == L.shape[0]:
        # Full decomposition
        eigenvalues, eigenfunctions = jnp.linalg.eigh(L)
    else:
        # Partial decomposition (dominant modes)
        eigenvalues, eigenfunctions = linalg.sparse.eigsh(
            L, k=num_modes, which='SM'  # smallest magnitude
        )
    
    # Sort by eigenvalue (ascending, most negative first)
    idx = jnp.argsort(eigenvalues)
    return eigenvalues[idx], eigenfunctions[:, idx]
```

**Lanczos Algorithm** (large-scale):
```python
def lanczos_iteration(L, k, v0=None, tol=1e-10, max_iter=None):
    """
    Compute k dominant eigenpairs using Lanczos iteration.
    
    Complexity: O(k · nnz(L)) where nnz = number of non-zeros
    
    Args:
        L: Laplacian operator (sparse matrix or linear operator)
        k: number of eigenpairs desired
        v0: initial vector (random if None)
        tol: convergence tolerance
        max_iter: maximum iterations (default: 2k)
    
    Returns:
        eigenvalues: shape (k,)
        eigenfunctions: shape (n, k)
    """
    n = L.shape[0]
    m = min(max_iter or 2*k, n)
    
    # Initialize
    v = v0 or jax.random.normal(key, (n,))
    v = v / jnp.linalg.norm(v)
    
    alpha = jnp.zeros(m)  # diagonal of tridiagonal
    beta = jnp.zeros(m)   # off-diagonal
    
    V = jnp.zeros((n, m+1))  # Lanczos vectors
    V = V.at[:, 0].set(v)
    
    for j in range(m):
        # w = L v_j
        w = L @ V[:, j]
        
        # α_j = v_j^T w
        alpha = alpha.at[j].set(V[:, j] @ w)
        
        # w = w - α_j v_j - β_{j-1} v_{j-1}
        w = w - alpha[j] * V[:, j]
        if j > 0:
            w = w - beta[j-1] * V[:, j-1]
        
        # β_j = ||w||
        beta = beta.at[j].set(jnp.linalg.norm(w))
        
        # Check convergence
        if beta[j] < tol:
            break
        
        # v_{j+1} = w / β_j
        V = V.at[:, j+1].set(w / beta[j])
    
    # Build tridiagonal matrix T
    T = jnp.diag(alpha[:m]) + jnp.diag(beta[:m-1], k=1) + jnp.diag(beta[:m-1], k=-1)
    
    # Eigendecomposition of small tridiagonal T
    theta, s = jnp.linalg.eigh(T)
    
    # Ritz vectors: ψ_i = V s_i
    eigenfunctions = V[:, :m] @ s
    eigenvalues = theta
    
    return eigenvalues[:k], eigenfunctions[:, :k]
```

### 4.3 Spectral Representation

**Field Decomposition**:
```
φ(x,t) = Σ_{n=0}^∞ a_n(t) ψ_n(x)

where:
  a_n(t) = ⟨φ(·,t), ψ_n⟩ = ∫ φ(x,t) ψ_n(x) √|g| dx
```

**Temporal Dynamics**:
```
From wave equation □φ + Vφ = J:

ä_n(t) + ω_n² a_n(t) = f_n(t)

where:
  ω_n² = -λ_n + V_n  (modified frequency)
  f_n(t) = ⟨J(·,t), ψ_n⟩  (projected source)
```

**Solution**:
```python
@jax.jit
def evolve_mode_amplitudes(a0, adot0, omega_sq, forcing, dt, n_steps):
    """
    Evolve mode amplitudes via harmonic oscillator equation.
    
    Args:
        a0: initial amplitudes a_n(0)
        adot0: initial velocities ȧ_n(0)
        omega_sq: squared frequencies ω_n²
        forcing: f_n(t) at each timestep, shape (n_steps, n_modes)
        dt: time step
        n_steps: number of steps
    
    Returns:
        a(t): amplitudes at each timestep
    """
    def step(carry, f_t):
        a, adot = carry
        
        # Leapfrog integration for harmonic oscillator
        adot_half = adot + 0.5 * dt * (-omega_sq * a + f_t)
        a_new = a + dt * adot_half
        adot_new = adot_half + 0.5 * dt * (-omega_sq * a_new + forcing)
        
        return (a_new, adot_new), a_new
    
    _, trajectory = jax.lax.scan(step, (a0, ad0), forcing)
    return trajectory
```

### 4.4 Mode Coupling for Reasoning

**Coupling Tensor**:
```
Inference as mode interaction:

Output amplitude: b_k = Σ_{i,j} C_{kij} a_i a_j + Σ_i D_{ki} a_i

where:
  C_{kij} = ⟨ψ_k, B(ψ_i, ψ_j)⟩  (bilinear coupling)
  D_{ki} = ⟨ψ_k, L(ψ_i)⟩        (linear mixing)
```

**Learning Coupling Coefficients**:
```python
class ModeCoupling(nn.Module):
    num_modes: int
    
    @nn.compact
    def __call__(self, mode_amplitudes):
        """
        Apply learned mode coupling.
        
        Args:
            mode_amplitudes: a ∈ ℝ^{num_modes}
        
        Returns:
            transformed amplitudes b ∈ ℝ^{num_modes}
        """
        # Linear mixing (always present)
        D = self.param('D', nn.initializers.xavier(), 
                      (self.num_modes, self.num_modes))
        linear_out = D @ mode_amplitudes
        
        # Bilinear coupling (optional, for nonlinearity)
        C = self.param('C', nn.initializers.normal(0.01),
                      (self.num_modes, self.num_modes, self.num_modes))
        bilinear_out = jnp.einsum('kij,i,j->k', C, mode_amplitudes, mode_amplitudes)
        
        return linear_out + bilinear_out
```

### 4.5 Spectral Attention

**Attention in Mode Space**:
```
Traditional attention: O(n²) complexity
Spectral attention: O(m log n) where m << n modes

Attention(Q, K, V) = softmax(QK^T / √d) V

In spectral domain:
  Q̂_m = ⟨Q, ψ_m⟩  (project to modes)
  K̂_m = ⟨K, ψ_m⟩
  V̂_m = ⟨V, ψ_m⟩
  
  SpectralAttention = Σ_m softmax(Q̂_m K̂_m^T / √d) V̂_m ψ_m(x)
```

**Implementation**:
```python
@jax.jit
def spectral_attention(query, key, value, eigenfunctions, eigenvalues):
    """
    Efficient attention via mode decomposition.
    
    Args:
        query, key, value: input sequences, shape (seq_len, dim)
        eigenfunctions: ψ_n(x), shape (seq_len, num_modes)
        eigenvalues: λ_n (unused here but available for weighting)
    
    Returns:
        attended output, shape (seq_len, dim)
    """
    num_modes = eigenfunctions.shape[1]
    
    # Project to mode space: O(n log n) via FFT-like operations
    Q_hat = eigenfunctions.T @ query  # (num_modes, dim)
    K_hat = eigenfunctions.T @ key
    V_hat = eigenfunctions.T @ value
    
    # Mode-wise attention: O(m² d) where m << n
    scale = 1.0 / jnp.sqrt(query.shape[-1])
    
    # Weight by eigenvalue (low-frequency modes get more attention)
    mode_weights = jnp.exp(-jnp.abs(eigenvalues))
    mode_weights /= mode_weights.sum()
    
    # Compute attention in mode space
    attn_logits = jnp.einsum('md,nd->mn', Q_hat, K_hat) * scale
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    
    # Apply attention to values
    attended_hat = jnp.einsum('mn,nd->md', attn_weights, V_hat)
    
    # Weight by mode importance
    attended_hat = attended_hat * mode_weights[:, None]
    
    # Project back to original space
    output = eigenfunctions @ attended_hat  # (seq_len, dim)
    
    return output
```

---

## Technical Specifications

### 5.1 Module Architecture

```
resonance_engine/
├── __init__.py
├── laplacian.py           # Laplace-Beltrami operator
├── eigensolver.py         # Lanczos, Arnoldi, full decomposition
├── mode_dynamics.py       # Temporal evolution of amplitudes
├── coupling.py            # Mode interaction tensors
├── spectral_attention.py  # Attention in mode space
└── compression.py         # Truncation, error bounds
```

### 5.2 Compression Strategy

**Mode Truncation**:
```
Retain only modes with significant energy:

Compression ratio: r = m / n where m retained, n total modes

Energy retention: E_retained = Σ_{k=1}^m |a_k|² / Σ_{k=1}^n |a_k|²

Target: E_retained ≥ 0.99 with minimal m
```

**Error Bounds**:
```python
def compute_truncation_error(amplitudes, threshold=1e-4):
    """
    Determine number of modes to retain for target accuracy.
    
    Args:
        amplitudes: |a_n| sorted by magnitude
        threshold: minimum relative energy to retain
    
    Returns:
        num_modes: number of modes to keep
        error_bound: theoretical upper bound on L2 error
    """
    energy = amplitudes ** 2
    cumulative_energy = jnp.cumsum(energy)
    total_energy = cumulative_energy[-1]
    
    # Find index where cumulative energy exceeds threshold
    idx = jnp.searchsorted(cumulative_energy, threshold * total_energy)
    
    # Error bound from Parseval: ||error||² = Σ_{k>m} |a_k|²
    error_squared = total_energy - cumulative_energy[idx]
    error_bound = jnp.sqrt(error_squared)
    
    return idx + 1, error_bound
```

---

## API Contract

### 6.1 Python Interface

```python
class ResonanceEngine:
    def __init__(
        self,
        manifold: Manifold,
        num_modes: int = 100,
        solver_method: str = 'lanczos',  # ['dense', 'lanczos', 'arnoldi']
    ):
        """Initialize resonance engine."""
        pass
    
    def compute_eigenmodes(
        self,
        num_modes: Optional[int] = None,
        recompute: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute eigenvalues and eigenfunctions.
        
        Returns:
            eigenvalues: shape (num_modes,)
            eigenfunctions: shape (*manifold_shape, num_modes)
        """
        pass
    
    def project_to_modes(
        self,
        field: jnp.ndarray,
        num_modes: Optional[int] = None,
    ) -> jnp.ndarray:
        """Project field onto eigenbasis."""
        pass
    
    def reconstruct_field(
        self,
        mode_amplitudes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Reconstruct field from mode amplitudes."""
        pass
    
    def evolve_modes(
        self,
        initial_amplitudes: jnp.ndarray,
        initial_velocities: jnp.ndarray,
        sources: Sequence[jnp.ndarray],
        n_steps: int,
    ) -> Sequence[jnp.ndarray]:
        """Evolve mode amplitudes over time."""
        pass
    
    def apply_coupling(
        self,
        amplitudes: jnp.ndarray,
        coupling_type: str = 'learned',  # ['linear', 'bilinear', 'neural']
    ) -> jnp.ndarray:
        """Apply mode coupling transformation."""
        pass
    
    def spectral_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute attention in spectral domain."""
        pass
    
    def compress(
        self,
        field: jnp.ndarray,
        energy_threshold: float = 0.99,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compress field via mode truncation.
        
        Returns compressed amplitudes and metadata.
        """
        pass
```

---

## Testing Strategy

### 7.1 Unit Tests

**Laplacian Verification**:
```python
def test_laplacian_eigenvalues():
    """Verify eigenvalues on known geometry."""
    # Flat torus: eigenvalues are -4π²|k|²
    torus = create_flat_torus(shape=(64, 64))
    engine = ResonanceEngine(torus)
    
    eigenvalues, _ = engine.compute_eigenmodes(num_modes=50)
    
    # Compare to analytic formula
    expected = compute_torus_eigenvalues(64, 64, num_modes=50)
    
    # Should match up to discretization error
    assert jnp.allclose(jnp.sort(eigenvalues), jnp.sort(expected), rtol=0.05)
```

**Orthogonality Check**:
```python
def test_eigenfunction_orthogonality():
    """Verify eigenfunctions are orthonormal."""
    engine = ResonanceEngine(manifold)
    _, eigenfunctions = engine.compute_eigenmodes(num_modes=100)
    
    # Flatten spatial dimensions
    psi_flat = eigenfunctions.reshape(-1, 100)
    
    # Compute Gram matrix
    gram = psi_flat.T @ psi_flat
    
    # Should be identity
    assert jnp.allclose(gram, jnp.eye(100), atol=1e-6)
```

**Reconstruction Accuracy**:
```python
def test_reconstruction_accuracy():
    """Verify field reconstruction from modes."""
    engine = ResonanceEngine(manifold)
    eigenvalues, _ = engine.compute_eigenmodes(num_modes=200)
    
    # Random test field
    field = jax.random.normal(key, manifold.shape)
    
    # Project and reconstruct
    amplitudes = engine.project_to_modes(field)
    reconstructed = engine.reconstruct_field(amplitudes)
    
    # Relative error should be small
    rel_error = jnp.linalg.norm(field - reconstructed) / jnp.linalg.norm(field)
    assert rel_error < 0.01, f"Reconstruction error {rel_error} too large"
```

### 7.2 Performance Benchmarks

```python
@benchmark
def bench_eigendecomposition(benchmark):
    """Measure eigenmode computation time."""
    manifold = Manifold(dimensions=[128, 128, 128])
    engine = ResonanceEngine(manifold, num_modes=100)
    
    result = benchmark(engine.compute_eigenmodes)
    return result

@benchmark
def bench_spectral_attention(benchmark):
    """Compare spectral vs standard attention."""
    seq_len = 4096
    dim = 512
    
    query = jax.random.normal(key, (seq_len, dim))
    key = jax.random.normal(key, (seq_len, dim))
    value = jax.random.normal(key, (seq_len, dim))
    
    engine = ResonanceEngine(...)
    
    result = benchmark(engine.spectral_attention, query, key, value)
    return result
```

**Performance Targets**:
- Eigendecomposition: < 1s for 100 modes on 128³ grid
- Projection: O(n log n), < 10ms for 4096 sequence
- Spectral attention: 10× speedup over standard attention at seq_len=4096
- Compression ratio: > 10× with < 1% energy loss

---

## Deliverables

### Week 1-2: Laplacian and Eigensolvers
- [ ] Discrete Laplace-Beltrami operator
- [ ] Dense eigendecomposition for small problems
- [ ] Lanczos implementation for large-scale
- [ ] Orthogonality verification tests

### Week 3: Mode Dynamics
- [ ] Projection/reconstruction pipeline
- [ ] Temporal evolution integrator
- [ ] Energy conservation checks
- [ ] Forced oscillation handling

### Week 4: Mode Coupling
- [ ] Linear mixing layer
- [ ] Bilinear coupling tensor
- [ ] Neural coupling networks
- [ ] Gradient flow through coupling

### Week 5: Spectral Attention
- [ ] Mode-space attention mechanism
- [ ] Comparison with standard attention
- [ ] Integration with transformer blocks
- [ ] Speed/accuracy benchmarks

### Week 6: Compression and Optimization
- [ ] Adaptive mode truncation
- [ ] Error bound certification
- [ ] Memory-efficient storage
- [ ] Documentation

---

## Acceptance Criteria

### Functional Requirements
- ✅ All eigenmode computations converge
- ✅ Reconstruction error < 1% with sufficient modes
- ✅ Spectral attention produces valid outputs
- ✅ Mode coupling is differentiable end-to-end

### Mathematical Requirements
- ✅ Eigenvalues are real and non-positive
- ✅ Eigenfunctions are orthonormal to machine precision
- ✅ Temporal evolution conserves energy (no forcing)
- ✅ Compression error matches theoretical bounds

### Performance Requirements
- ✅ Eigendecomposition: O(n m²) for m modes, n points
- ✅ Projection/reconstruction: O(n m)
- ✅ Spectral attention: O(m² d + n m) vs O(n² d) standard
- ✅ Compression achieves > 10× reduction with < 1% loss

### Reasoning Capability Requirements
- ✅ Solves 3-hop reasoning tasks with > 90% accuracy
- ✅ Generalizes to unseen compositions
- ✅ Mode coupling interpretable (analyzable patterns)

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Lanczos breakdown (ghost eigenvalues) | High | Low | Full reorthogonalization; implicit restart |
| Numerical instability in evolution | Medium | Medium | Symplectic integrators; adaptive stepping |
| Over-compression loses information | Medium | High | Conservative thresholds; error monitoring |
| Spectral attention underperforms | Medium | Medium | Hybrid approach (spectral + local) |
| Mode coupling hard to train | Low | Medium | Careful initialization; curriculum |

---

## Dependencies

### Python Packages
- `jax` (v0.4+): Core numerical library
- `scipy.sparse.linalg`: Iterative eigensolvers
- `flax` or `haiku`: Neural network modules
- `pytest-benchmark`: Performance testing

### Hardware Requirements
- GPU with 16GB+ memory for large eigendecompositions
- Double precision (FP64) recommended for eigenvalue accuracy

---

## Success Metrics

1. **Computational**: Meets all performance targets
2. **Mathematical**: All spectral properties verified
3. **Compression**: > 10× reduction with certified bounds
4. **Reasoning**: Strong performance on multi-hop tasks
5. **Interpretability**: Mode patterns align with semantic structure

---

*Phase Owner: Spectral Methods Team*  
*Review Gate: End of Week 6*  
*Next Phase: Phase 5 - Topological Memory and Persistent Homology*
