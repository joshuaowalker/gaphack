# Development Notes for Claude

This file contains development notes and future considerations for the gapHACk project.

## Quick Reference

### Architecture Overview
- **Distance Provider Architecture**: Two-tier system (MSA-cached, precomputed) - see [Performance Considerations](#performance-considerations)
- **Cluster Graph**: BLAST K-NN graph for O(K) proximity queries - see [Cluster Graph Infrastructure](#cluster-graph-infrastructure)
- **Iterative Refinement**: Neighborhood-based refinement with convergence tracking - see [Refinement Implementation](#refinement-implementation)
- **Result Tracking**: Comprehensive refinement tracking via ProcessingStageInfo - see [Result Tracking and Reproducibility](#result-tracking-and-reproducibility)

### Key Files by Functionality
- **Core clustering**: `core.py` (main algorithm + DistanceCache)
- **Target mode**: `target_clustering.py`
- **Refinement orchestration**: `cluster_refinement.py` (iterative neighborhood-based refinement)
- **Refinement types**: `refinement_types.py` (ClusterIDGenerator, ProcessingStageInfo, cluster ID utilities)
- **CLI entry points**: `cli.py` (main gaphack), `refine_cli.py` (gaphack-refine), `analyze_cli.py` (gaphack-analyze)
- **Distance computation**: `distance_providers.py` (MSACachedDistanceProvider, PrecomputedDistanceProvider)
- **Cluster proximity**: `cluster_graph.py` (ClusterGraph with BLAST K-NN)
- **BLAST integration**: `blast_neighborhood.py`
- **Neighborhood finding**: `neighborhood_finder.py`, `vsearch_neighborhood.py`

### Testing
- **Comprehensive test suite** with integration, performance, and quality tests
- **Real biological data**: 1,429 Russula sequences with ground truth
- **Performance baselines**: See [Performance Baselines](#performance-baselines)
- **Quality metrics**: ARI >0.85, Homogeneity >0.90, Completeness >0.85

## Algorithm Development History

### Gap-Optimized Clustering Evolution

The gap-optimized clustering algorithm has evolved through several iterations:

1. **Original Greedy (Distance-based)**: Used inter-cluster distance as heuristic, calculated gaps only for evaluation
2. **Beam Search Implementation**: Used gap as heuristic with configurable beam width to explore multiple search paths
3. **Current Gap-based Greedy**: Uses gap as heuristic in a greedy search, running to completion and tracking the best gap encountered (removed early termination and min_gap_size parameter)

### Future Algorithm Considerations

**Beam Search Re-implementation**:
The beam search approach was removed in favor of the simpler gap-based greedy algorithm. However, beam search could be valuable for future exploration if we need:
- Better solution quality (beam search can find gaps ~10% better than distance-based greedy)
- Exploration of multiple search paths simultaneously
- Analysis of search space characteristics

Key implementation details for future reference:
- Beam search maintained top-k partitions at each level based on gap scores
- Used partition caching to avoid recalculating the same configurations
- Explored 35-40% fewer partitions than exhaustive search while finding equivalent solutions
- Default beam width of 10 provided good balance of quality vs. performance

The beam search implementation can be found in git history if needed for future algorithm research.

## Technical Debt and Known Issues

(No current issues)

### Performance Considerations

#### Distance Provider Architecture
The codebase implements a two-tier distance computation system (`distance_providers.py`):

**1. MSA-Cached Distance Provider** (`MSACachedDistanceProvider`)
- Creates SPOA multiple sequence alignment once for all sequences in scope
- Computes pairwise distances on-demand from the shared MSA
- Maintains cache (`_distance_cache`) of computed distances
- Uses median sequence length for distance normalization
- Raises `MSAAlignmentError` if SPOA fails

**2. Precomputed Distance Provider** (`PrecomputedDistanceProvider`)
- Wraps full precomputed distance matrices for classic gaphack usage
- No-op for `ensure_distances_computed()` since all distances exist
- Used by core algorithm and when full matrix is available

This architecture provides:
- Consistent distances within refinement scopes (all pairs share same MSA context)
- Efficient on-demand computation with caching
- Clean abstraction via `DistanceProvider` base class

#### Distance Cache in Core Algorithm
The multiprocessing implementation uses a specialized `DistanceCache` class (`core.py:171`):
- Maintains persistent worker caches across parallel operations
- Uses offset-based coordinate mapping for efficient workload distribution
- Enables ~4x speedup with 8 cores (200→800 steps/second)
- Reduces initialization overhead by reusing worker state

## Refinement Implementation

### Iterative Refinement Architecture
**Purpose**: Optimize clustering quality through iterative neighborhood-based refinement

**Key Components**:
- `iterative_refinement()` (`cluster_refinement.py`): Main refinement loop using proximity graph
- `refine_scope_with_gaphack()`: Runs full gapHACk on seed + neighborhood
- `build_refinement_scope()`: Builds seed cluster + neighbors within max_lump + context

**Refinement Process**:
1. Build proximity graph (BLAST/vsearch K-NN)
2. For each seed cluster (deterministic priority order):
   - Find nearby clusters within max_lump
   - Expand context beyond max_lump (gradient area)
   - Run full gapHACk on combined scope
   - Update clusters if changed
3. Iterate until convergence (AMI change < threshold)

**Convergence Criteria**:
- AMI change < 0.001 between iterations (stable clustering)
- All scopes converged (no changes in last 2 iterations)
- Maximum iterations reached (default: 10)

**Refinement Configuration** (`cluster_refinement.py::RefinementConfig`):
- `max_full_gaphack_size` (300): Maximum sequences for full gapHACk refinement
- `close_threshold`: Distance threshold for finding nearby clusters
- `max_iterations` (10): Maximum refinement iterations
- `k_neighbors` (20): K-NN graph parameter
- `search_method` ("blast"): BLAST or vsearch for proximity graph
- `random_seed`: Seed for randomizing seed order (None = deterministic)

### Result Tracking and Reproducibility
The `ProcessingStageInfo` dataclass (`refinement_types.py`) provides comprehensive tracking:

**Stage Information**:
- `stage_name`: Name of the processing stage (e.g., "Iterative Refinement")
- `clusters_before`: Cluster state before stage
- `clusters_after`: Cluster state after stage
- `components_processed`: Details about each component refined
- `summary_stats`: Aggregate statistics (counts, changes, timing, iterations, convergence)

**Cluster ID Management** (`refinement_types.py::ClusterIDGenerator`):
- Generates globally unique cluster IDs with stage suffixes
- Format: `cluster_{NNNNN}{SUFFIX}` where SUFFIX is I (initial) or R1/R2/R3... (refinement iterations)
- Helper functions: `format_cluster_id()`, `parse_cluster_id()`, `get_next_cluster_number()`

### Cluster Graph Infrastructure
**Purpose**: Efficient proximity queries for finding nearby clusters during refinement (`cluster_graph.py`)

**ClusterGraph Implementation**:
- **BLAST-based K-NN graph**: Uses BLAST to find K nearest neighbors for each cluster medoid
- **Complexity**: O(K) rather than O(C) for proximity queries (C = total clusters)
- **Medoid caching**: Computes and caches cluster medoids (sequence with minimum total distance to others)
- **Dynamic updates**: Graph can be updated as clusters merge or split during refinement

**Key Parameters**:
- `k_neighbors` (default: 20): Number of nearest neighbors to maintain per cluster
- `blast_evalue` (default: 1e-5): BLAST e-value threshold
- `blast_identity` (default: 90.0): Minimum BLAST identity percentage

**Usage in Post-Processing**:
- Finds clusters within distance threshold for close cluster refinement
- Enables efficient "expand context" operations by quickly finding nearby clusters
- Scalable to 100K+ sequences with thousands of clusters

## Testing Infrastructure

### Comprehensive Test Suite
**Test Organization** (custom pytest markers):
- `@pytest.mark.integration`: End-to-end pipeline tests
- `@pytest.mark.performance`: Performance regression tests
- `@pytest.mark.memory`: Memory usage and scaling tests
- `@pytest.mark.scalability`: Algorithm complexity validation
- `@pytest.mark.quality`: Biological relevance and quality metrics

### Real Biological Validation
**Russula Dataset**: 1,429 fungal ITS sequences with 143 ground truth groups
- Power-law size distribution (largest group: 105 sequences)
- Taxonomic and geographic annotations for coherence analysis
- Located in: `examples/data/Russula_INxx.fasta`

**Test Subsets** (for faster iteration):
- `tests/test_data/russula_diverse_50.fasta` (50 sequences)
- `tests/test_data/russula_diverse_100.fasta` (100 sequences)
- `tests/test_data/russula_diverse_200.fasta` (200 sequences)
- `tests/test_data/russula_diverse_300.fasta` (300 sequences)
- `tests/test_data/russula_diverse_500.fasta` (500 sequences)
- `tests/test_data/russula_300.fasta` (300 sequences, older subset)

### Performance Baselines
**Established Thresholds** (for regression detection):
- 100 sequences: <60 seconds
- 300 sequences: <300 seconds (5 minutes)
- 500 sequences: <600 seconds (10 minutes)
- Full dataset (1,429): <600 seconds with optimizations
- Memory usage: <4GB for full dataset

### Quality Metrics
**Empirically Validated Thresholds**:
- **ARI (Adjusted Rand Index)**: >0.85 (empirical: 0.948)
- **Homogeneity**: >0.90 (empirical: 0.972)
- **Completeness**: >0.85 (empirical: 0.969)
- **Taxonomic coherence**: Mean purity >85%
- **Geographic signal**: 30-90% (balanced expectation)

**See**: `tests/PHASE4_SUMMARY.md` for comprehensive testing documentation

### Testing Best Practices
- Gap-based greedy with gap heuristic is functionally equivalent to beam search with beam_width=1
- Current approach finds optimal gap within the greedy search path (not globally optimal)
- Runs to completion rather than early termination, ensuring full exploration of valid merge space
- N+N pruning effectiveness can be tested by comparing gaps with different context ratios
- Context expansion effectiveness validated by tracking gap improvement across iterations

## Future Work and Experimental Features

### Incremental Clustering (Partial Implementation)
The refinement system includes infrastructure for incremental/streaming clustering:
- `incremental_search_distance` parameter in `RefinementConfig`
- `max_closest_clusters` limit for local refinement scope
- Designed for scenarios where new sequences are added to existing clustering

**Current Status**: Infrastructure present but not fully implemented
**Use Case**: Online clustering where sequences arrive over time
**Implementation Location**: `cluster_refinement.py` (configuration only)

**Design Considerations**:
- New sequences can be assigned to existing clusters via proximity search
- Only nearby clusters need reconsideration (O(K) via ClusterGraph)
- Maintains global clustering quality while processing incrementally
- Would require additional logic for determining when full refinement is needed

### Algorithm Research Opportunities

**Beam Search Re-exploration**:
- Current greedy approach finds optimal gap within its path (not globally optimal)
- Beam search implementation exists in git history
- Could provide ~10% better gaps with beam width 10
- Trade-off: 2-3x slower but explores multiple paths
- Consider if: dataset has complex taxonomy requiring better solution quality

**Alternative Heuristics**:
- Current: gap-based heuristic (maximize barcode gap at each step)
- Alternative 1: Mixed heuristic (distance + gap weighted combination)
- Alternative 2: Lookahead heuristic (consider next N merges before deciding)
- Alternative 3: Hierarchical heuristic (optimize different percentiles at different levels)

**Refinement Scope Optimization**:
- Current: Fixed thresholds for scope expansion
- Alternative: Adaptive thresholds based on cluster density
- Could reduce over-refinement in sparse regions
- Could increase refinement in dense regions (subspecies complexes)

### Known Limitations and Improvement Opportunities

**Current Constraints**:
1. **Scope size limits**: Max 300 sequences for full gapHACk refinement
   - Larger scopes could improve quality but increase runtime
   - Consider hierarchical refinement for larger scopes

2. **Fixed K for K-NN graph**: Currently k=20 for all datasets
   - Dense datasets might benefit from higher K
   - Sparse datasets might benefit from lower K
   - Could adapt K based on dataset characteristics

3. **Target percentile consistency**: Hardcoded 95th in `target_clustering.py:309`
   - Should respect `target_percentile` parameter
   - Simple fix but affects behavior with non-default percentiles

### Performance Optimization Opportunities

**Potential Improvements**:
1. **Parallel BLAST operations**: Currently sequential BLAST calls
   - Could parallelize neighborhood discovery across multiple targets
   - Would improve refinement iteration speed

2. **Incremental medoid updates**: Currently recomputes all medoids
   - After small refinements, most medoids unchanged
   - Could track which clusters changed and update selectively

3. **Approximate nearest neighbor search**: Currently exact BLAST K-NN
   - LSH or other ANN methods could speed up very large datasets
   - Trade-off: approximation vs. speed

4. **GPU acceleration**: Distance calculations are embarrassingly parallel
   - Could offload to GPU for 10-100x speedup on large matrices
   - Useful for datasets with 10K+ sequences in single refinement scope