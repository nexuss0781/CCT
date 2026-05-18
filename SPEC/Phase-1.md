# Phase 1: Mathematical Foundations and Core Data Structures

## Overview

**Duration**: 4 weeks  
**Goal**: Establish rigorous mathematical foundations and implement core data structures with formal verification.  
**Exit Criteria**: All core structures pass property-based tests; complexity bounds are formally verified.

---

## Scope

### In Scope
1. Formal specification of causal manifold mathematics
2. Implementation of sparse manifold data structure in Rust
3. Event representation with causal metadata
4. Basic operations: insertion, deletion, neighborhood queries
5. Complexity analysis and benchmarking
6. Property-based testing suite

### Out of Scope
- Spectral solvers (Phase 2)
- Semantic embeddings (Phase 3)
- Training infrastructure (Phase 6)

---

## Mathematical Specifications

### 1.1 Causal Manifold Definition

**Formal Definition**:
```
A causal manifold is a tuple M = (D, S, g, ≺) where:
  D ⊂ ℤ^d is a d-dimensional discrete grid (d ∈ {2,3,4})
  S: D → 𝒫(Event) is a sparse assignment function
  g: D × D → ℝ is a discrete metric tensor
  ≺ ⊂ D × D is a strict partial order (causal relation)
```

**Axioms**:
1. **Asymmetry**: ∀x,y: x ≺ y ⇒ ¬(y ≺ x)
2. **Transitivity**: ∀x,y,z: (x ≺ y ∧ y ≺ z) ⇒ x ≺ z
3. **Local finiteness**: |{y : x ≺ y}| < ∞ for all x
4. **Metric compatibility**: d(x,y)² = g_μν(x)(x-y)^μ(x-y)^ν

### 1.2 Event Structure

**Type Signature**:
```rust
struct Event {
    semantic_vector: Vec<f32>,      // ∈ ℝ^{d_sem}
    temporal_tensor: Vec<isize>,    // ∈ ℤ^d (coordinates in D)
    causal_potential_vector: Vec<f32>, // ∈ ℝ^{d_causal}
    timestamp: u64,                  // Logical clock
    causal_parents: Vec<EventId>,   // References to antecedent events
}
```

**Invariants**:
- `semantic_vector.len() == d_sem` (configurable, default 512)
- `temporal_tensor.len() == d` (manifold dimension)
- `∀t ∈ temporal_tensor: 0 ≤ t < manifold_dim[t]`
- `causal_potential_vector` satisfies energy conditions (see §2.3)

### 1.3 Energy Conditions

**Weak Energy Condition (WEC)**:
```
∀ timelike vectors v^μ: T_μν v^μ v^ν ≥ 0

Discrete form:
  Σ_i,j T_ij v^i v^j ≥ 0 for all test vectors v
```

**Implementation Constraint**:
```rust
fn verify_wec(event: &Event) -> bool {
    let energy_density = event.causal_potential_vector[0];
    energy_density >= 0.0
}
```

---

## Technical Specifications

### 2.1 Sparse Manifold Representation

**Data Structure Choice**: Adaptive Octree (3D) / Quadtree (2D)

**Rationale**:
- O(log n) insertion and lookup
- Memory efficiency for sparse distributions
- Natural support for multi-resolution queries

**Rust Implementation**:
```rust
pub struct Manifold {
    dimensions: Vec<usize>,
    root: OctreeNode,
    occupancy_rate: f64,
    max_depth: u8,
}

enum OctreeNode {
    Leaf {
        events: Vec<Event>,
        bbox: BoundingBox,
    },
    Internal {
        children: [Box<OctreeNode>; 8],
        bbox: BoundingBox,
    }
}
```

**Complexity Guarantees**:
| Operation | Time | Space |
|-----------|------|-------|
| Insert event | O(log n) | O(1) amortized |
| Delete event | O(log n) | O(1) amortized |
| Get at coords | O(log n) | O(1) |
| Range query | O(k + log n) | O(k) |
| k-nearest neighbors | O(k log n) | O(k) |

where n = total events, k = result size.

### 2.2 Coordinate Systems

**Supported Systems**:
1. **Cartesian**: Standard integer lattice ℤ^d
2. **Light-cone**: Null coordinates u = t - x, v = t + x
3. **Geodesic**: Distance-based from reference point

**Conversion Functions**:
```rust
trait CoordinateSystem {
    fn to_cartesian(&self) -> Vec<isize>;
    fn from_cartesian(coords: Vec<isize>) -> Self;
    fn causal_interval(&self, other: &Self) -> i64;
}
```

**Causal Interval**:
```
Δs² = -c²Δt² + Δx² + Δy² + Δz²

Sign convention:
  Δs² < 0: timelike separation (causally connected)
  Δs² = 0: lightlike (null separation)
  Δs² > 0: spacelike (causally disconnected)
```

### 2.3 Memory Management

**Arena Allocation**:
- Events allocated in contiguous memory pools
- Reference by `EventId = u64` instead of `Box<Event>`
- Cache-friendly iteration over active regions

**Garbage Collection**:
- Reference counting for shared events
- Epoch-based reclamation for bulk deletions
- Configurable retention policies (FIFO, LRU, priority-based)

---

## API Contract

### 3.1 Manifold Interface

```rust
pub trait ManifoldOps {
    /// Create new manifold with specified dimensions
    fn new(dimensions: Vec<usize>, config: ManifoldConfig) -> Self;
    
    /// Insert event at specified coordinates
    /// Returns Err if coordinates out of bounds or cell occupied
    fn insert(&mut self, event: Event) -> Result<(), ManifoldError>;
    
    /// Remove event at coordinates
    /// Returns the removed event or None if empty
    fn remove(&mut self, coords: &[isize]) -> Option<Event>;
    
    /// Query event at exact coordinates
    fn get(&self, coords: &[isize]) -> Option<&Event>;
    
    /// Find all events within causal future of point
    fn causal_future(&self, point: &[isize]) -> Vec<&Event>;
    
    /// Find all events within causal past of point
    fn causal_past(&self, point: &[isize]) -> Vec<&Event>;
    
    /// K-nearest neighbors by geodesic distance
    fn k_nearest(&self, point: &[isize], k: usize) -> Vec<&Event>;
    
    /// Iterate over all events in bounding box
    fn range_query(&self, bbox: &BoundingBox) -> impl Iterator<Item = &Event>;
    
    /// Return manifold statistics (occupancy, depth distribution, etc.)
    fn stats(&self) -> ManifoldStats;
}
```

### 3.2 Error Handling

```rust
#[derive(Debug, Error)]
pub enum ManifoldError {
    #[error("Coordinates {0:?} out of bounds")]
    OutOfBounds(Vec<isize>),
    
    #[error("Cell already occupied at {0:?}")]
    CellOccupied(Vec<isize>),
    
    #[error("Invalid causal structure: {0}")]
    CausalViolation(String),
    
    #[error("Energy condition violated at {0:?}")]
    EnergyConditionViolation(Vec<isize>),
    
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),
}
```

### 3.3 Python Bindings (PyO3)

```python
class Manifold:
    def __init__(self, dimensions: List[int], config: Optional[Dict] = None)
    def insert(self, event: Event) -> None
    def remove(self, coords: List[int]) -> Optional[Event]
    def get(self, coords: List[int]) -> Optional[Event]
    def causal_future(self, point: List[int]) -> List[Event]
    def causal_past(self, point: List[int]) -> List[Event]
    def k_nearest(self, point: List[int], k: int) -> List[Event]
    def stats(self) -> Dict[str, Any]
```

---

## Testing Strategy

### 4.1 Unit Tests

**Coverage Requirements**:
- 100% line coverage for core data structures
- 95% branch coverage for error handling paths
- All public APIs tested with edge cases

**Test Categories**:
1. **Construction**: Valid/invalid dimensions, boundary sizes
2. **Insertion**: Normal, duplicate, out-of-bounds, full capacity
3. **Query**: Exact match, near-miss, empty regions
4. **Causal queries**: Timelike, lightlike, spacelike separations
5. **Memory**: Allocation limits, fragmentation scenarios

### 4.2 Property-Based Testing (QuickCheck)

**Properties to Verify**:

```rust
// P1: Insertion followed by get returns same event
prop_insert_get(manifold: Manifold, event: Event, coords: Coords) -> bool

// P2: Causal ordering is transitive
prop_causal_transitivity(m: &Manifold, a: Coords, b: Coords, c: Coords) -> bool

// P3: Removal is inverse of insertion
prop_remove_inverse(m: Manifold, event: Event, coords: Coords) -> bool

// P4: k-nearest returns exactly k results (when available)
prop_knn_count(m: &Manifold, point: Coords, k: usize) -> bool

// P5: Range query contains only points in bbox
prop_range_bbox(m: &Manifold, bbox: BoundingBox) -> bool

// P6: Complexity bound holds empirically
prop_complexity_log_n(sizes: Vec<usize>) -> bool
```

### 4.3 Performance Benchmarks

**Benchmark Suite**:
```rust
#[bench]
fn bench_insert_sequential(b: &mut Bencher) {
    // Measure O(log n) scaling
}

#[bench]
fn bench_insert_random(b: &mut Bencher) {
    // Measure cache miss impact
}

#[bench]
fn bench_causal_query(b: &mut Bencher) {
    // Measure light cone traversal
}

#[bench]
fn bench_knn(b: &mut Bencher) {
    // Measure neighbor search scaling
}
```

**Performance Targets**:
- Insert: < 100ns for n=10⁶ events
- Query: < 50ns for single point lookup
- k-NN: < 1μs for k=10, n=10⁶
- Memory overhead: < 2× raw event size

---

## Deliverables

### Week 1-2: Core Implementation
- [ ] `Manifold` struct with octree backend
- [ ] `Event` struct with validation
- [ ] Basic CRUD operations
- [ ] Coordinate system conversions
- [ ] Initial PyO3 bindings

### Week 3: Advanced Operations
- [ ] Causal future/past queries
- [ ] k-nearest neighbors
- [ ] Range queries with iterators
- [ ] Memory management optimizations

### Week 4: Testing & Verification
- [ ] Complete unit test suite
- [ ] Property-based tests (QuickCheck)
- [ ] Benchmark suite
- [ ] Documentation (rustdoc + Sphinx)
- [ ] Complexity analysis report

---

## Acceptance Criteria

### Functional Requirements
- ✅ All API methods implemented per spec
- ✅ Zero undefined behavior (Miri validation)
- ✅ Thread-safe for concurrent reads (Send + Sync)
- ✅ Python bindings fully functional

### Non-Functional Requirements
- ✅ Worst-case O(log n) for all queries (empirically verified)
- ✅ Memory usage < 3× theoretical minimum
- ✅ No allocations in hot path (benchmark profiled)
- ✅ Passes all property-based tests (10k+ iterations)

### Documentation Requirements
- ✅ Inline rustdoc comments for all public items
- ✅ Architecture decision records (ADRs) for key choices
- ✅ Usage examples in both Rust and Python
- ✅ Mathematical proofs in supplementary document

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Octree imbalance degrades to O(n) | High | Medium | Implement rebalancing; fallback to k-d tree |
| Memory fragmentation | Medium | High | Arena allocator with compaction |
| PyO3 GIL contention | Medium | Medium | Batch operations; release GIL during compute |
| Numerical precision in causal tests | Low | High | Use interval arithmetic for robustness |

---

## Dependencies

### External Crates
- `ndarray` (v0.15+): Multi-dimensional arrays
- `pyo3` (v0.18+): Python bindings
- `thiserror` (v1.0+): Error handling
- `quickcheck` (v1.0+): Property-based testing
- `criterion` (v0.5+): Benchmarking framework

### System Requirements
- Rust 1.70+ (for generic associated types)
- Python 3.9+ (for typing improvements)
- 16GB RAM for large-scale benchmarks
- AVX2 instruction set (for vectorization)

---

## Success Metrics

1. **Correctness**: 100% test pass rate, zero sanitizer warnings
2. **Performance**: All benchmarks meet targets (±10% variance)
3. **Completeness**: All planned features implemented
4. **Documentation**: 100% public API documented with examples
5. **Maintainability**: Cyclomatic complexity < 15 per function

---

*Phase Owner: Core Systems Team*  
*Review Gate: End of Week 4*  
*Next Phase: Phase 2 - Spectral Solver Implementation*
