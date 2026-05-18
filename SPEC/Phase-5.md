# Phase 5: Topological Memory and Persistent Homology

## Overview

**Duration**: 5 weeks  
**Goal**: Implement topological memory structures using persistent homology for robust, content-addressable long-term memory.  
**Exit Criteria**: System demonstrates stable memory retrieval under noise; Betti numbers correlate with semantic structure.

---

## Scope

### In Scope
1. Simplicial complex construction from point clouds
2. Persistent homology computation
3. Persistence diagram manipulation
4. Morse complex for critical point analysis
5. Conley index for isolating blocks
6. Topological feature-based retrieval
7. Memory consolidation via topological simplification

### Out of Scope
- Neural memory augmentation (Phase 6)
- Cross-manifold memory sharing (Phase 7)

---

## Mathematical Specifications

### 5.1 Simplicial Complexes

**Definition**:
```
A simplicial complex K is a collection of simplices such that:
  1. Every face of a simplex in K is also in K
  2. Intersection of any two simplices is a common face

Types:
  - Vietoris-Rips complex: VR_ε(X) = {σ ⊂ X | diam(σ) ≤ ε}
  - Čech complex: Č_ε(X) = {σ ⊂ X | ∩_{x∈σ} B_ε(x) ≠ ∅}
  - Alpha complex: Delaunay-based, computationally efficient
```

**Construction from Events**:
```python
def build_vietoris_rips_complex(points, epsilon):
    """
    Build VR complex from point cloud at scale ε.
    
    Args:
        points: N×d array of event coordinates
        epsilon: maximum simplex diameter
    
    Returns:
        dict mapping dimension → list of simplices
    """
    from scipy.spatial.distance import pdist, squareform
    
    dist_matrix = squareform(pdist(points))
    
    # 0-simplices (vertices)
    complexes = {0: [[i] for i in range(len(points))]}
    
    # Higher-dimensional simplices
    max_dim = min(10, len(points) // 2)  # Practical limit
    
    for dim in range(1, max_dim + 1):
        complexes[dim] = []
        
        # Check all (dim+1)-tuples
        for simplex in combinations(range(len(points)), dim + 1):
            # Verify all faces exist
            faces_valid = all(
                tuple(sorted(simplex[:i] + simplex[i+1:])) in 
                [tuple(s) for s in complexes[dim-1]]
                for i in range(dim + 1)
            )
            
            if faces_valid:
                # Check diameter condition
                max_dist = max(dist_matrix[i, j] 
                              for i, j in combinations(simplex, 2))
                if max_dist <= epsilon:
                    complexes[dim].append(list(simplex))
    
    return complexes
```

### 5.2 Persistent Homology

**Filtration**:
```
Nested sequence of simplicial complexes:
  K_0 ⊆ K_1 ⊆ ... ⊆ K_N = K

For VR complex: filtration by increasing ε
```

**Persistence Diagram**:
```
Dgm_k(f) = {(b_i, d_i)} where:
  b_i = birth time (ε value) of k-dimensional hole
  d_i = death time of k-dimensional hole
  persistence = d_i - b_i (lifetime)

Points far from diagonal = significant topological features
```

**Computation** (matrix reduction):
```python
def compute_persistence(boundary_matrix):
    """
    Compute persistence via matrix reduction algorithm.
    
    Complexity: O(n³) worst case, often much faster in practice
    
    Args:
        boundary_matrix: ∂ in matrix form
    
    Returns:
        persistence_pairs: list of (birth, death) for each feature
    """
    m, n = boundary_matrix.shape
    pivot = {}
    
    for j in range(n):
        while boundary_matrix[:, j].any():
            # Find pivot
            i = np.argmax(boundary_matrix[:, j])
            
            if i in pivot:
                # Column operation: add pivot column
                boundary_matrix[:, j] ^= boundary_matrix[:, pivot[i]]
            else:
                # Found new pivot
                pivot[i] = j
                break
        
        # If column became zero, feature is born at j
        # If column reduced using pivot[i], feature dies at j
    
    # Extract persistence pairs from pivot dictionary
    persistence_pairs = extract_pairs(pivot, n)
    
    return persistence_pairs
```

### 5.3 Betti Numbers

**Definition**:
```
β_k = rank(H_k(K)) = number of k-dimensional holes

where:
  H_k(K) = ker(∂_k) / im(∂_{k+1}) is k-th homology group
  ∂_k: C_k → C_{k-1} is boundary operator
```

**Interpretation**:
- β₀: Number of connected components
- β₁: Number of loops/tunnels
- β₂: Number of voids/cavities
- β₃+: Higher-dimensional analogues

**Memory Encoding**:
```python
def encode_memory_topologically(events):
    """
    Encode memory as topological signature.
    
    Args:
        events: list of Event objects
    
    Returns:
        topological_signature: dict with Betti numbers and persistence
    """
    # Extract coordinates
    points = np.array([e.temporal_tensor for e in events])
    
    # Compute persistence across scales
    max_scale = estimate_optimal_scale(points)
    scales = np.linspace(0, max_scale, 50)
    
    betti_curves = {k: [] for k in range(4)}
    persistence_diagrams = []
    
    for eps in scales:
        complex = build_vietoris_rips_complex(points, eps)
        
        # Compute Betti numbers at this scale
        for k in range(4):
            beta_k = len(complex[k])  # Simplified; actual computation needs reduction
            betti_curves[k].append(beta_k)
        
        # Store persistence info
        if should_record_features(eps):
            diagrams = compute_persistence_at_scale(complex)
            persistence_diagrams.append(diagrams)
    
    return {
        'betti_curves': betti_curves,
        'persistence_diagrams': persistence_diagrams,
        'total_persistence': sum_persistence(persistence_diagrams),
    }
```

### 5.4 Morse Theory

**Morse Function**:
```
f: M → ℝ is Morse if all critical points are non-degenerate:
  det(Hess f)|_{critical points} ≠ 0

Critical point types:
  - Minimum (index 0): all eigenvalues positive
  - Saddle (index k): k negative eigenvalues
  - Maximum (index d): all eigenvalues negative
```

**Morse Complex**:
```
Chain complex generated by critical points:
  C_k = span{critical points of index k}
  ∂: C_k → C_{k-1} counts gradient flow lines

Homology of Morse complex ≅ singular homology of M
```

**Memory Navigation**:
```python
def navigate_morse_complex(query, morse_graph):
    """
    Navigate memory via gradient flow on Morse function.
    
    Args:
        query: query embedding
        morse_graph: graph of critical points with flow edges
    
    Returns:
        retrieved_memories: list of relevant memories
    """
    # Find nearest critical point
    start_node = find_nearest_critical_point(query, morse_graph.nodes)
    
    # Follow gradient flow (downhill for minima, uphill for maxima)
    trajectory = follow_gradient_flow(start_node, morse_graph)
    
    # Collect memories along trajectory
    retrieved = [node.memory for node in trajectory 
                 if node.has_memory]
    
    return retrieved
```

### 5.5 Conley Index

**Isolating Block**:
```
N ⊂ M is isolating block for flow φ_t if:
  inv(N) = {x ∈ N | φ_t(x) ∈ N ∀t} ⊂ int(N)

Conley index measures topology of isolated invariant set.
```

**Application to Memory**:
```
Robust retrieval despite perturbations:
  - Memory encoded as isolated invariant set
  - Query induces flow toward attractor
  - Conley index certifies existence of memory
```

---

## Technical Specifications

### 6.1 Module Architecture

```
topological_memory/
├── __init__.py
├── simplicial.py         # Complex construction
├── persistence.py        # Persistent homology computation
├── betti.py             # Betti number tracking
├── morse.py             # Morse complex, critical points
├── conley.py            # Conley index computation
├── retrieval.py         # Topological memory access
└── consolidation.py     # Memory stabilization
```

### 6.2 Efficient Computation

**Approximate Persistence** (for large datasets):
```python
def approximate_persistence(points, subsample_size=1000):
    """
    Use subsampling for scalable persistence computation.
    
    Complexity: O(m³) where m << n is subsample size
    """
    # Subsample points
    indices = random_subsample(len(points), subsample_size)
    subset = points[indices]
    
    # Compute persistence on subset
    persistence = compute_persistence_exact(subset)
    
    # Bootstrap confidence intervals
    bootstrap_results = []
    for _ in range(100):
        boot_indices = random_subsample(len(points), subsample_size)
        boot_subset = points[boot_indices]
        boot_persistence = compute_persistence_exact(boot_subset)
        bootstrap_results.append(boot_persistence)
    
    # Aggregate results
    confidence_intervals = compute_ci(bootstrap_results)
    
    return persistence, confidence_intervals
```

---

## API Contract

### 7.1 Python Interface

```python
class TopologicalMemory:
    def __init__(
        self,
        manifold: Manifold,
        max_dimension: int = 3,
        computation_method: str = 'exact',  # ['exact', 'approximate']
    ):
        """Initialize topological memory system."""
        pass
    
    def encode(
        self,
        events: Sequence[Event],
        memory_id: str,
    ) -> TopologicalSignature:
        """Encode memory as topological signature."""
        pass
    
    def retrieve(
        self,
        query: Event,
        k: int = 10,
        similarity_metric: str = 'bottleneck',  # ['bottleneck', 'wasserstein']
    ) -> List[Memory]:
        """Retrieve memories by topological similarity."""
        pass
    
    def compute_betti_numbers(
        self,
        events: Sequence[Event],
        scale: Optional[float] = None,
    ) -> Dict[int, int]:
        """Compute Betti numbers at given scale."""
        pass
    
    def build_morse_complex(
        self,
        events: Sequence[Event],
    ) -> MorseGraph:
        """Construct Morse complex from event distribution."""
        pass
    
    def consolidate(
        self,
        memory_ids: List[str],
        method: str = 'simplification',
    ) -> str:
        """Consolidate multiple memories into unified structure."""
        pass
    
    def visualize(
        self,
        memory_id: str,
        plot_type: str = 'persistence',
    ) -> Any:
        """Generate visualization (diagram, barcode, etc.)."""
        pass
```

---

## Testing Strategy

### 8.1 Unit Tests

**Betti Number Verification**:
```python
def test_sphere_homology():
    """Verify Betti numbers for 2-sphere."""
    # Sample points on S²
    points = sample_sphere(num_points=1000)
    
    mem = TopologicalMemory(...)
    betti = mem.compute_betti_numbers(points, scale=0.5)
    
    # Expected: β₀=1 (connected), β₁=0 (no loops), β₂=1 (one void)
    assert betti[0] == 1
    assert betti[1] == 0
    assert betti[2] == 1
```

**Persistence Stability**:
```python
def test_persistence_stability():
    """Verify stability under perturbation."""
    points = generate_test_cloud()
    
    mem = TopologicalMemory(...)
    sig1 = mem.encode(points, 'original')
    
    # Add small noise
    noisy_points = points + 0.01 * np.random.randn(*points.shape)
    sig2 = mem.encode(noisy_points, 'noisy')
    
    # Persistence diagrams should be close (stability theorem)
    distance = bottleneck_distance(sig1.persistence, sig2.persistence)
    assert distance < 0.1  # Proportional to noise level
```

---

## Deliverables

### Week 1: Simplicial Complex Infrastructure
- [ ] VR complex builder
- [ ] Boundary matrix construction
- [ ] Basic homology computation

### Week 2: Persistent Homology
- [ ] Matrix reduction algorithm
- [ ] Persistence diagram generation
- [ ] Bottleneck/Wasserstein distance

### Week 3: Betti Analysis
- [ ] Betti curve computation
- [ ] Scale optimization
- [ ] Feature significance testing

### Week 4: Morse Theory Integration
- [ ] Critical point detection
- [ ] Gradient flow computation
- [ ] Morse complex construction

### Week 5: Retrieval and Consolidation
- [ ] Topological similarity search
- [ ] Memory consolidation algorithms
- [ ] Visualization tools
- [ ] Documentation

---

## Acceptance Criteria

### Functional Requirements
- ✅ All topological computations correct
- ✅ Retrieval returns semantically relevant memories
- ✅ Consolidation reduces redundancy
- ✅ Visualizations are informative

### Mathematical Requirements
- ✅ Betti numbers match known examples
- ✅ Persistence satisfies stability theorem
- ✅ Morse complex homology matches manifold homology
- ✅ Conley index correctly identifies attractors

### Performance Requirements
- ✅ Exact persistence: O(n³) for n ≤ 1000 points
- ✅ Approximate persistence: O(m³) for m=100 subsample
- ✅ Retrieval: O(log N) for N stored memories
- ✅ Memory footprint: O(k) for k topological features

### Memory Quality Requirements
- ✅ Recall@10 > 80% on benchmark tasks
- ✅ Robust to 10% noise in queries
- ✅ Consolidation improves retrieval speed by 2×

---

*Phase Owner: Topological Methods Team*  
*Review Gate: End of Week 5*  
*Next Phase: Phase 6 - Integration and Production Hardening*
