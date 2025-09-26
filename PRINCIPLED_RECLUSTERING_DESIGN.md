# Principled Post-Processing for gapHACk-Decompose: Scope-Limited Reclustering Design

## Executive Summary

This document outlines a principled approach to address three critical limitations of gaphack-decompose:
1. **Non-MECE clustering** due to sequence conflicts across multiple clusters
2. **Close clusters** lacking clear barcode gaps between them
3. **Limited incremental refinement** capability for new sequences or cluster evolution

The proposed solution uses **scope-limited reclustering** with classic gapHACk applied to carefully selected cluster neighborhoods, ensuring MECE output while maintaining computational efficiency.

## Problem Analysis

### Current State Assessment

Based on codebase analysis, gaphack-decompose exhibits the following characteristics:

**Strengths:**
- Scales to 100K+ sequences via BLAST-based neighborhoods
- Handles large datasets through iterative target clustering
- Implements medoid caching and triangle inequality filtering
- Provides conflict detection and basic overlap merging

**Critical Limitations:**

1. **Overlapping Assignments**: Sequences can belong to multiple clusters, violating MECE principles
2. **Suboptimal Boundaries**: Local optimization in target clustering may miss global optima
3. **Close Cluster Proliferation**: Medoid-based detection identifies close clusters but current merging is too conservative
4. **No Incremental Refinement**: Adding new sequences requires full reprocessing

### Algorithmic Root Causes

**Target Clustering Limitations:**
- **Binary optimization**: Only considers target cluster vs. remaining sequences
- **Local scope**: Cannot optimize across multiple potential clusters simultaneously
- **Greedy growth**: May create boundaries that global optimization would reject

**Current Merging Limitations:**
- **Conservative validation**: Gap-based merging only succeeds if p95 ≤ max_lump
- **Pairwise focus**: Cannot handle complex multi-cluster overlaps optimally
- **Static scope**: Fixed cluster pairs rather than dynamic neighborhood selection

## Solution Architecture

### Core Principle: Scope-Limited Reclustering

Apply **classic gapHACk** to carefully selected sequence subsets to achieve:
- **MECE guarantee**: Classic gapHACk ensures mutually exclusive assignment
- **Global optimization**: Within scope, finds optimal clustering configuration
- **Bounded complexity**: Limit scope size to maintain performance (≤300 sequences)

### Three Reclustering Use Cases

#### 1. Conflict Resolution Reclustering
**Trigger**: Sequence assigned to multiple clusters
**Scope**: Union of all clusters containing conflicted sequence(s)
**Outcome**: MECE partition resolving all conflicts in scope

#### 2. Close Cluster Refinement Reclustering
**Trigger**: Clusters with medoids closer than expected gap
**Scope**: Neighborhood of close clusters (expandable)
**Outcome**: Optimal clustering within barcode gap constraints

#### 3. Incremental Update Reclustering
**Trigger**: New sequence addition or cluster evolution
**Scope**: Existing clusters closest to new sequence
**Outcome**: Updated clustering incorporating new information

### Scope Selection Strategy

**Dynamic Neighborhood Expansion:**
```
1. Start with minimal scope (conflicted clusters, close pairs, etc.)
2. Expand scope to include "nearby" clusters based on:
   - Medoid distances ≤ scope_expansion_threshold
   - Sequence overlap patterns
   - Gap analysis suggesting potential merges
3. Stop expansion when:
   - Scope size approaches performance limit (~300 sequences)
   - No additional clusters meet inclusion criteria
   - Expansion would create disconnected components
```

## Detailed Algorithm Specifications

### Algorithm 1: Conflict Resolution Reclustering

```python
def resolve_conflicts_via_reclustering(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]],
                                     sequences: List[str], headers: List[str]) -> Dict[str, List[str]]:
    """
    Resolves assignment conflicts using classic gapHACk reclustering.

    Args:
        conflicts: sequence_id -> list of cluster_ids containing sequence
        all_clusters: cluster_id -> list of sequence_headers
        sequences: full sequence list
        headers: full header list

    Returns:
        Updated cluster dictionary with conflicts resolved
    """

    # Step 1: Group conflicts by connected components
    conflict_components = find_connected_conflict_components(conflicts, all_clusters)

    updated_clusters = all_clusters.copy()

    for component_clusters in conflict_components:
        # Step 2: Extract scope sequences
        scope_sequences = set()
        for cluster_id in component_clusters:
            scope_sequences.update(all_clusters[cluster_id])

        # Step 3: Apply scope expansion if beneficial
        expanded_scope = expand_scope_for_conflicts(
            scope_sequences, component_clusters, all_clusters,
            expansion_threshold=1.5 * max_lump
        )

        # Step 4: Apply classic gapHACk to scope
        if len(expanded_scope.sequences) <= MAX_CLASSIC_GAPHACK_SIZE:
            classic_result = apply_classic_gaphack(expanded_scope.sequences)

            # Step 5: Replace original clusters with classic result
            for cluster_id in expanded_scope.cluster_ids:
                del updated_clusters[cluster_id]

            for i, new_cluster in enumerate(classic_result.clusters):
                new_cluster_id = f"resolved_{hash(tuple(sorted(new_cluster)))}"
                updated_clusters[new_cluster_id] = new_cluster

        else:
            # Fallback: use conservative pairwise merging for oversized scopes
            updated_clusters = apply_conservative_conflict_resolution(
                component_clusters, updated_clusters
            )

    return updated_clusters
```

### Algorithm 2: Close Cluster Refinement Reclustering

```python
def refine_close_clusters(all_clusters: Dict[str, List[str]],
                         sequences: List[str], headers: List[str],
                         close_threshold: float = None) -> Dict[str, List[str]]:
    """
    Refines clusters that are closer than expected barcode gaps.

    Args:
        all_clusters: current cluster dictionary
        sequences: full sequence list
        headers: full header list
        close_threshold: distance threshold for "close" clusters (default: max_lump)

    Returns:
        Updated cluster dictionary with close clusters refined
    """

    if close_threshold is None:
        close_threshold = max_lump

    # Step 1: Identify close cluster pairs via medoid analysis
    close_pairs = find_close_cluster_pairs(all_clusters, sequences, headers, close_threshold)

    # Step 2: Group close pairs into connected components
    close_components = find_connected_close_components(close_pairs)

    # Step 3: Track processed components to avoid infinite loops
    processed_components = set()
    updated_clusters = all_clusters.copy()

    for component_clusters in close_components:
        component_signature = frozenset(component_clusters)

        if component_signature in processed_components:
            continue  # Skip already processed components

        # Step 4: Extract and expand scope
        scope_sequences = set()
        for cluster_id in component_clusters:
            scope_sequences.update(updated_clusters[cluster_id])

        expanded_scope = expand_scope_for_close_clusters(
            scope_sequences, component_clusters, updated_clusters,
            expansion_threshold=close_threshold * 1.2
        )

        # Step 5: Apply classic gapHACk if scope is manageable
        if len(expanded_scope.sequences) <= MAX_CLASSIC_GAPHACK_SIZE:
            classic_result = apply_classic_gaphack(expanded_scope.sequences)

            # Step 6: Check if classic result differs from input
            if clusters_significantly_different(expanded_scope.clusters, classic_result.clusters):
                # Replace with classic result
                for cluster_id in expanded_scope.cluster_ids:
                    del updated_clusters[cluster_id]

                for i, new_cluster in enumerate(classic_result.clusters):
                    new_cluster_id = f"refined_{hash(tuple(sorted(new_cluster)))}"
                    updated_clusters[new_cluster_id] = new_cluster

            # Mark as processed regardless of outcome
            processed_components.add(component_signature)

        else:
            # Skip oversized components, mark as processed
            processed_components.add(component_signature)

    return updated_clusters
```

### Algorithm 3: Incremental Update Reclustering

```python
def incremental_update_reclustering(new_sequences: List[str],
                                  new_headers: List[str],
                                  existing_clusters: Dict[str, List[str]],
                                  all_sequences: List[str],
                                  all_headers: List[str]) -> Dict[str, List[str]]:
    """
    Incrementally updates clustering with new sequences.

    Args:
        new_sequences: sequences to be incorporated
        new_headers: headers for new sequences
        existing_clusters: current cluster dictionary
        all_sequences: combined old + new sequences
        all_headers: combined old + new headers

    Returns:
        Updated cluster dictionary incorporating new sequences
    """

    updated_clusters = existing_clusters.copy()

    for new_seq, new_header in zip(new_sequences, new_headers):
        # Step 1: Find closest existing clusters
        closest_clusters = find_closest_clusters_to_sequence(
            new_seq, new_header, existing_clusters, all_sequences, all_headers,
            max_distance=max_lump * 2.0,  # Broader search for incremental updates
            max_clusters=5  # Limit scope size
        )

        if not closest_clusters:
            # Step 2a: No close clusters - create singleton cluster
            singleton_id = f"singleton_{hash(new_header)}"
            updated_clusters[singleton_id] = [new_header]

        else:
            # Step 2b: Include new sequence in reclustering scope
            scope_sequences = {new_header}
            scope_cluster_ids = []

            for cluster_id, distance in closest_clusters:
                scope_sequences.update(updated_clusters[cluster_id])
                scope_cluster_ids.append(cluster_id)

            # Step 3: Apply classic gapHACk to scope + new sequence
            if len(scope_sequences) <= MAX_CLASSIC_GAPHACK_SIZE:
                classic_result = apply_classic_gaphack(list(scope_sequences))

                # Step 4: Replace original clusters with classic result
                for cluster_id in scope_cluster_ids:
                    del updated_clusters[cluster_id]

                for i, new_cluster in enumerate(classic_result.clusters):
                    new_cluster_id = f"incremental_{hash(tuple(sorted(new_cluster)))}"
                    updated_clusters[new_cluster_id] = new_cluster

            else:
                # Fallback: assign to closest cluster or create singleton
                closest_cluster_id, min_distance = closest_clusters[0]
                if min_distance <= max_lump:
                    updated_clusters[closest_cluster_id].append(new_header)
                else:
                    singleton_id = f"singleton_{hash(new_header)}"
                    updated_clusters[singleton_id] = [new_header]

    return updated_clusters
```

## Cluster Proximity Graph Architecture

### Scalability Challenge: From Brute Force to K-Nearest Neighbors

**Current Approach Limitations:**
The existing medoid-based overlap detection computes all pairwise distances between cluster medoids:
- **Current scale**: ~1,600 clusters → ~1.3M distance calculations (manageable)
- **Target scale**: ~10K-20K clusters → 50M-200M distance calculations (prohibitive)

**Proposed Solution: K-Nearest Neighbor Graph**
Replace exhaustive pairwise distance computation with a sparse K-NN graph of cluster medoids, enabling:
- **Scalable proximity queries**: O(K) rather than O(C) for finding close clusters
- **BLAST-based construction**: Leverage existing BLAST infrastructure for sequence similarity
- **Incremental updates**: Efficient graph maintenance as clusters evolve

### Abstract Cluster Proximity Interface

```python
from abc import ABC, abstractmethod

class ClusterProximityGraph(ABC):
    """Abstract interface for cluster proximity queries and close cluster detection."""

    @abstractmethod
    def get_neighbors_within_distance(self, cluster_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Find all clusters within specified distance of given cluster."""
        pass

    @abstractmethod
    def get_k_nearest_neighbors(self, cluster_id: str, k: int) -> List[Tuple[str, float]]:
        """Find K nearest neighbor clusters to given cluster."""
        pass

    @abstractmethod
    def find_close_pairs(self, max_distance: float) -> List[Tuple[str, str, float]]:
        """Find all cluster pairs closer than max_distance."""
        pass

    @abstractmethod
    def update_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Update graph when cluster medoid changes."""
        pass

    @abstractmethod
    def remove_cluster(self, cluster_id: str) -> None:
        """Remove cluster from proximity graph."""
        pass

    @abstractmethod
    def add_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Add new cluster to proximity graph."""
        pass
```

### Implementation Strategy: Brute Force First, Optimize Later

**Phase 1: Brute Force Implementation (Current)**
```python
class BruteForceProximityGraph(ClusterProximityGraph):
    """Brute force implementation using full pairwise distance computation."""

    def __init__(self, clusters: Dict[str, List[str]], sequences: List[str],
                 headers: List[str], distance_provider: DistanceProvider):
        self.clusters = clusters
        self.distance_provider = distance_provider
        self.medoid_cache = {}
        self._compute_all_medoids()

    def get_neighbors_within_distance(self, cluster_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Current O(C) implementation - compute distances to all other clusters."""
        target_medoid = self.medoid_cache[cluster_id]
        neighbors = []

        for other_id, other_medoid in self.medoid_cache.items():
            if other_id != cluster_id:
                distance = self.distance_provider.get_distance(target_medoid, other_medoid)
                if distance <= max_distance:
                    neighbors.append((other_id, distance))

        return sorted(neighbors, key=lambda x: x[1])

    def find_close_pairs(self, max_distance: float) -> List[Tuple[str, str, float]]:
        """Current implementation - all pairwise comparisons."""
        close_pairs = []
        cluster_ids = list(self.medoid_cache.keys())

        for i, cluster1_id in enumerate(cluster_ids):
            for cluster2_id in cluster_ids[i+1:]:
                medoid1 = self.medoid_cache[cluster1_id]
                medoid2 = self.medoid_cache[cluster2_id]
                distance = self.distance_provider.get_distance(medoid1, medoid2)

                if distance <= max_distance:
                    close_pairs.append((cluster1_id, cluster2_id, distance))

        return close_pairs
```

**Phase 2: BLAST K-NN Optimization (Future)**
```python
class BlastKNNProximityGraph(ClusterProximityGraph):
    """BLAST-based K-NN graph for scalable cluster proximity queries."""

    def __init__(self, clusters: Dict[str, List[str]], sequences: List[str],
                 headers: List[str], k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.clusters = clusters
        self.sequences = sequences
        self.headers = headers
        self.medoid_cache = {}
        self.knn_graph = {}  # cluster_id -> [(neighbor_id, distance), ...]
        self._build_blast_knn_graph()

    def _build_blast_knn_graph(self) -> None:
        """Build K-NN graph using BLAST on cluster medoid sequences."""
        # 1. Compute medoids for all clusters
        self._compute_all_medoids()

        # 2. Create BLAST database from medoid sequences
        medoid_sequences = []
        medoid_cluster_map = {}

        for cluster_id, medoid_idx in self.medoid_cache.items():
            medoid_sequence = self.sequences[medoid_idx]
            medoid_sequences.append(medoid_sequence)
            medoid_cluster_map[len(medoid_sequences) - 1] = cluster_id

        blast_db = create_blast_database(medoid_sequences)

        # 3. Query each medoid against database to find K nearest neighbors
        for query_idx, medoid_sequence in enumerate(medoid_sequences):
            query_cluster_id = medoid_cluster_map[query_idx]

            blast_results = run_blast_query(
                query_sequence=medoid_sequence,
                blast_db=blast_db,
                max_hits=self.k_neighbors + 1,  # +1 to exclude self-hit
                evalue_threshold=1e-5
            )

            # 4. Convert BLAST results to distance-based neighbors
            neighbors = []
            for hit in blast_results:
                if hit.subject_idx != query_idx:  # Exclude self
                    neighbor_cluster_id = medoid_cluster_map[hit.subject_idx]
                    # Convert BLAST identity to adjusted identity distance
                    distance = identity_to_adjusted_distance(hit.identity_percent)
                    neighbors.append((neighbor_cluster_id, distance))

            # 5. Sort by distance and keep K nearest
            neighbors.sort(key=lambda x: x[1])
            self.knn_graph[query_cluster_id] = neighbors[:self.k_neighbors]

    def get_neighbors_within_distance(self, cluster_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Efficient O(K) implementation using precomputed K-NN graph."""
        if cluster_id not in self.knn_graph:
            return []

        neighbors = []
        for neighbor_id, distance in self.knn_graph[cluster_id]:
            if distance <= max_distance:
                neighbors.append((neighbor_id, distance))
            else:
                break  # Neighbors are sorted by distance

        return neighbors

    def find_close_pairs(self, max_distance: float) -> List[Tuple[str, str, float]]:
        """Efficient close pair detection using K-NN graph."""
        close_pairs = []
        processed_pairs = set()

        for cluster_id, neighbors in self.knn_graph.items():
            for neighbor_id, distance in neighbors:
                if distance > max_distance:
                    break  # Neighbors sorted by distance

                # Ensure each pair is only reported once
                pair_key = tuple(sorted([cluster_id, neighbor_id]))
                if pair_key not in processed_pairs:
                    close_pairs.append((cluster_id, neighbor_id, distance))
                    processed_pairs.add(pair_key)

        return close_pairs
```

### Integration with Reclustering Algorithms

**Modified Algorithm Signatures:**
```python
def resolve_conflicts_via_reclustering(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]],
                                     sequences: List[str], headers: List[str],
                                     proximity_graph: ClusterProximityGraph) -> Dict[str, List[str]]:

def refine_close_clusters(all_clusters: Dict[str, List[str]],
                         sequences: List[str], headers: List[str],
                         proximity_graph: ClusterProximityGraph,
                         close_threshold: float = None) -> Dict[str, List[str]]:

def incremental_update_reclustering(new_sequences: List[str],
                                  new_headers: List[str],
                                  existing_clusters: Dict[str, List[str]],
                                  all_sequences: List[str], all_headers: List[str],
                                  proximity_graph: ClusterProximityGraph) -> Dict[str, List[str]]:
```

**Scope Expansion with K-NN Graph:**
```python
def expand_scope_with_proximity_graph(core_cluster_ids: List[str],
                                     proximity_graph: ClusterProximityGraph,
                                     expansion_threshold: float,
                                     max_scope_size: int) -> List[str]:
    """Expand reclustering scope using K-NN graph for efficiency."""

    expanded_cluster_ids = set(core_cluster_ids)
    candidates = []

    # Find all neighbors within expansion threshold
    for cluster_id in core_cluster_ids:
        neighbors = proximity_graph.get_neighbors_within_distance(cluster_id, expansion_threshold)
        for neighbor_id, distance in neighbors:
            if neighbor_id not in expanded_cluster_ids:
                candidates.append((neighbor_id, distance))

    # Sort candidates by distance and add until size limit
    candidates.sort(key=lambda x: x[1])

    for neighbor_id, distance in candidates:
        if len(expanded_cluster_ids) >= max_scope_size:
            break
        expanded_cluster_ids.add(neighbor_id)

    return list(expanded_cluster_ids)
```

### Performance Analysis: Brute Force vs. K-NN Graph

**Computational Complexity:**

| Operation | Brute Force | K-NN Graph | Improvement |
|-----------|-------------|------------|-------------|
| Graph Construction | O(C²) | O(C×K + BLAST) | Significant for C >> K |
| Find Close Pairs | O(C²) | O(C×K) | ~K/C improvement |
| Neighbor Queries | O(C) | O(K) | ~K/C improvement |
| Graph Updates | O(C) | O(K + BLAST) | Moderate |

**Memory Requirements:**

| Approach | Storage | Notes |
|----------|---------|-------|
| Brute Force | O(C²) | Full distance matrix |
| K-NN Graph | O(C×K) | Sparse neighbor lists |

**Scaling Projections:**
- **Current (1.6K clusters)**: Brute force acceptable, ~1.3M distance calculations
- **Target (20K clusters)**: K-NN graph essential, reduces 200M → 400K calculations (K=20)

### Implementation Phasing Strategy

**Phase 1: Foundation with Brute Force (Weeks 1-4)**
- Implement `ClusterProximityGraph` interface
- Create `BruteForceProximityGraph` implementation
- Integrate with conflict resolution reclustering
- Validate correctness on current test datasets

**Phase 2: K-NN Graph Optimization (Weeks 5-8)**
- Implement `BlastKNNProximityGraph` with BLAST-based construction
- Add graph update mechanisms for cluster changes
- Performance testing and parameter tuning (K, BLAST parameters)
- Comparative validation with brute force results

**Phase 3: Production Deployment (Weeks 9-10)**
- Automatic fallback mechanisms (K-NN → brute force for small datasets)
- Configuration management for scaling parameters
- Production testing on full 100K+ sequence datasets

This phased approach ensures immediate progress on conflict resolution while establishing the architecture needed for scalable close cluster refinement and incremental updates.

## Revised Implementation Strategy: Close Clusters First

### Strategic Rationale

The implementation order has been revised to prioritize **close cluster refinement** before K-NN graph optimization. This approach provides several key advantages:

**1. Quality Improvement at Current Scale**
- Address close cluster proliferation immediately with existing infrastructure
- Improve clustering quality for current 1.6K cluster workloads
- Validate proximity graph interface with different access patterns

**2. Direct Performance Comparison Framework**
- Implement close cluster refinement with brute force proximity graph first
- Then implement BLAST K-NN graph as drop-in replacement
- Measure identical workloads on both implementations for accurate comparison
- Validate that both approaches produce identical clustering results

**3. Concrete K-NN Graph Validation**
- Close cluster detection stresses proximity graph differently than conflict resolution
- More intensive proximity queries (finding all close pairs vs. conflict neighborhoods)
- Better validation of K-NN graph correctness before production deployment

**4. Implementation Risk Reduction**
- Validate proximity graph interface thoroughly before major K-NN graph investment
- Identify interface gaps or performance bottlenecks early
- Ensure architectural decisions support both current and future scale requirements

### Technical Implementation Benefits

**Close Cluster Workload Characteristics:**
- **Query Pattern**: `find_close_pairs()` across entire cluster set (O(C²) brute force vs O(C×K) K-NN)
- **Scope Size**: Typically larger than conflict resolution (multiple close clusters + neighbors)
- **Iteration Potential**: May require multiple refinement passes until convergence

**Performance Comparison Methodology:**
```python
# Phase 2A: Benchmark with brute force
brute_force_graph = BruteForceProximityGraph(clusters, sequences, headers, distance_provider)
start_time = time.time()
result_brute = refine_close_clusters(clusters, sequences, headers, brute_force_graph, config)
brute_force_time = time.time() - start_time

# Phase 2B: Benchmark with K-NN graph (identical input)
knn_graph = BlastKNNProximityGraph(clusters, sequences, headers, k_neighbors=20)
start_time = time.time()
result_knn = refine_close_clusters(clusters, sequences, headers, knn_graph, config)
knn_time = time.time() - start_time

# Validation: Results must be identical
assert result_brute == result_knn, "K-NN graph produces different clustering results"
speedup = brute_force_time / knn_time
```

**Interface Stress Testing:**
- `find_close_pairs(max_distance)` - Tests comprehensive proximity detection
- `get_neighbors_within_distance()` - Tests scope expansion for refinement
- Graph updates during cluster merging - Tests dynamic graph maintenance

This revised approach ensures robust validation of the K-NN graph optimization while delivering immediate quality improvements for close cluster handling.

### Expected Performance Characteristics

**Brute Force Proximity Graph (Phase 2A baseline):**
- **Time Complexity**: O(C²) for close pair detection across C clusters
- **Memory**: O(C²) medoid distance cache
- **Scaling**: Viable up to ~2000 clusters (4M medoid distance calculations)
- **Accuracy**: 100% accurate proximity queries (reference implementation)

**BLAST K-NN Graph (Phase 2B target):**
- **Time Complexity**: O(C×K) for K-NN proximity queries, where K << C
- **Memory**: O(C×K) for K-NN graph storage (~40x smaller than brute force at 2K clusters)
- **Scaling**: Target 10K+ clusters with K=20 neighbors
- **Accuracy**: Must be proven identical to brute force on identical inputs

**Performance Validation Benchmarks:**
```
Cluster Count | Brute Force Time | K-NN Time | Expected Speedup | Memory Reduction
500           | 1s              | 0.5s      | 2x               | 25x
1000          | 4s              | 1s        | 4x               | 50x
2000          | 16s             | 2s        | 8x               | 100x
5000          | 100s            | 5s        | 20x              | 250x
10000         | 400s            | 10s       | 40x              | 500x
```

**Quality Validation Requirements:**
- Identical clustering results on all test datasets
- Same convergence behavior (number of refinement iterations)
- Equivalent gap improvement trajectories

## Scope Selection and Expansion Strategies

### Conflict Component Detection

```python
def find_connected_conflict_components(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]]) -> List[List[str]]:
    """
    Groups conflicted clusters into connected components.

    Two clusters are connected if they share any conflicted sequence.
    """

    # Build cluster adjacency graph
    cluster_graph = defaultdict(set)

    for seq_id, cluster_ids in conflicts.items():
        for i, cluster1 in enumerate(cluster_ids):
            for cluster2 in cluster_ids[i+1:]:
                cluster_graph[cluster1].add(cluster2)
                cluster_graph[cluster2].add(cluster1)

    # Find connected components using DFS
    visited = set()
    components = []

    for cluster_id in cluster_graph:
        if cluster_id not in visited:
            component = []
            dfs_traverse(cluster_id, cluster_graph, visited, component)
            components.append(component)

    return components
```

### Scope Expansion Heuristics

```python
def expand_scope_for_conflicts(initial_sequences: Set[str],
                             core_cluster_ids: List[str],
                             all_clusters: Dict[str, List[str]],
                             expansion_threshold: float) -> ExpandedScope:
    """
    Expands conflict resolution scope to include nearby clusters.

    Expansion criteria:
    1. Clusters with medoids within expansion_threshold of core clusters
    2. Clusters with significant sequence overlap (>10% Jaccard similarity)
    3. Respect maximum scope size constraints
    """

    expanded_sequences = initial_sequences.copy()
    expanded_cluster_ids = core_cluster_ids.copy()

    # Calculate medoids for all clusters
    cluster_medoids = compute_all_cluster_medoids(all_clusters, sequences, headers)

    for candidate_cluster_id, candidate_sequences in all_clusters.items():
        if candidate_cluster_id in expanded_cluster_ids:
            continue

        # Check distance criterion
        should_include = False
        candidate_medoid = cluster_medoids[candidate_cluster_id]

        for core_cluster_id in core_cluster_ids:
            core_medoid = cluster_medoids[core_cluster_id]
            distance = compute_distance(candidate_medoid, core_medoid)

            if distance <= expansion_threshold:
                should_include = True
                break

        # Check overlap criterion
        if not should_include:
            overlap_ratio = len(expanded_sequences & set(candidate_sequences)) / len(set(candidate_sequences))
            if overlap_ratio > 0.1:  # 10% overlap threshold
                should_include = True

        # Include if criteria met and size constraints satisfied
        if should_include:
            potential_size = len(expanded_sequences | set(candidate_sequences))
            if potential_size <= MAX_CLASSIC_GAPHACK_SIZE:
                expanded_sequences.update(candidate_sequences)
                expanded_cluster_ids.append(candidate_cluster_id)

    return ExpandedScope(
        sequences=list(expanded_sequences),
        cluster_ids=expanded_cluster_ids
    )
```

## Performance Considerations

### Computational Complexity Analysis

**Classic gapHACk Scaling:**
- **Time Complexity**: O(n³) for gap calculations, O(n²) for distance computations
- **Practical Limits**: ~300 sequences for reasonable runtime (<30 seconds)
- **Memory**: O(n²) for distance matrix storage

**Scope Size Management:**
```python
MAX_CLASSIC_GAPHACK_SIZE = 300  # Conservative limit for performance
EXPANSION_SIZE_BUFFER = 50      # Reserve capacity for scope expansion
PREFERRED_SCOPE_SIZE = 250      # Target size for optimal performance
```

**Performance Optimization Strategies:**

1. **Precomputed Distance Reuse**: Cache distances from decompose process
2. **Incremental Gap Calculation**: Reuse gap computations where possible
3. **Parallel Processing**: Process independent scope components concurrently
4. **Early Termination**: Skip reclustering if scope expansion exceeds limits

### Memory Management

**Distance Matrix Caching:**
```python
class ScopedDistanceProvider:
    """Efficient distance provider for scope-limited reclustering."""

    def __init__(self, global_provider: DistanceProvider, scope_indices: List[int]):
        self.global_provider = global_provider
        self.scope_indices = scope_indices
        self.local_cache: Dict[Tuple[int, int], float] = {}

    def get_distance(self, local_i: int, local_j: int) -> float:
        global_i = self.scope_indices[local_i]
        global_j = self.scope_indices[local_j]
        return self.global_provider.get_distance(global_i, global_j)
```

## Implementation Roadmap

### Phase 1: Foundation with Brute Force Proximity Graph (4 weeks)

**Priority 1: Cluster Proximity Graph Interface (Week 1)**
- [ ] Implement abstract `ClusterProximityGraph` interface
- [ ] Create `BruteForceProximityGraph` implementation
- [ ] Add medoid computation and caching functionality
- [ ] Implement basic proximity query methods

**Priority 2: Classic gapHACk Integration (Week 1-2)**
- [ ] Create `ScopedDistanceProvider` for efficient scope-limited distance computation
- [ ] Integrate existing classic gapHACk algorithm with scoped providers
- [ ] Add triangle inequality filtering for scope-limited distance matrices
- [ ] Implement result integration back into decompose cluster structure

**Priority 3: Conflict Resolution Algorithm (Week 2-3)**
- [ ] Implement `resolve_conflicts_via_reclustering()` using proximity graph
- [ ] Add connected component grouping for conflicts
- [ ] Integrate with existing `AssignmentTracker` and conflicts detection
- [ ] Add comprehensive test coverage for conflict resolution scenarios

**Priority 4: Validation and Testing (Week 3-4)**
- [ ] Create test suite with current 9K sequence / 1.6K cluster dataset
- [ ] Validate MECE properties of conflict resolution output
- [ ] Performance benchmarking of brute force approach
- [ ] Comparison with direct classic gapHACk results for validation

### Phase 2: Close Cluster Refinement with Performance Comparison (6 weeks)

**Priority 5: Close Cluster Refinement with Brute Force (Week 5-6)**
- [ ] Implement `refine_close_clusters()` using existing BruteForceProximityGraph
- [ ] Add close cluster pair detection via medoid analysis
- [ ] Implement connected component grouping for close clusters
- [ ] Add infinite loop prevention with processed component tracking
- [ ] Add `--refine-close-clusters` CLI option and integration
- [ ] Test on datasets with known close cluster issues

**Priority 6: BLAST K-NN Graph Implementation (Week 7-8)**
- [ ] Implement `BlastKNNProximityGraph` as drop-in replacement for BruteForceProximityGraph
- [ ] Add medoid sequence BLAST database creation and querying
- [ ] Implement efficient K-NN graph storage and update mechanisms
- [ ] Add identity-to-distance conversion utilities
- [ ] Implement dynamic graph updates for cluster changes

**Priority 7: Performance Validation and Comparison (Week 9-10)**
- [ ] Direct performance comparison: BruteForce vs BLAST K-NN on identical close cluster workloads
- [ ] Validate that both implementations produce identical clustering results
- [ ] BLAST parameter tuning for medoid-level similarity detection
- [ ] Memory usage profiling and optimization
- [ ] Scalability testing with increasing cluster counts
- [ ] Document performance characteristics and scaling thresholds

### Phase 3: Production Reclustering Suite (2 weeks)

**Priority 8: Incremental Updates and Production Integration (Week 11-12)**
- [ ] Implement `incremental_update_reclustering()` with proximity graph interface
- [ ] Add efficient closest cluster detection for new sequences
- [ ] Design API for incremental sequence addition and batch processing
- [ ] Integrate all three reclustering modes into `gaphack-decompose` CLI
- [ ] Add automatic scaling (brute force ↔ K-NN graph) based on dataset size
- [ ] Add comprehensive CLI options: `--refine-close-clusters`, `--incremental-update`, `--reclustering-mode`
- [ ] Implement configuration management for complex reclustering scenarios

### Phase 4: Production Validation (2 weeks)

**Priority 11: Large-Scale Testing (Week 13-14)**
- [ ] Test on full 100K+ sequence datasets
- [ ] Validate scalability of K-NN graph approach
- [ ] Performance profiling and bottleneck identification
- [ ] Production deployment and monitoring setup

## Configuration Parameters

### Reclustering Thresholds
```python
class ReclusteringConfig:
    """Configuration for principled reclustering algorithms."""

    # Performance limits
    max_classic_gaphack_size: int = 300
    preferred_scope_size: int = 250
    expansion_size_buffer: int = 50

    # Scope expansion criteria
    conflict_expansion_threshold: float = 1.5 * max_lump
    close_cluster_expansion_threshold: float = 1.2 * max_lump
    incremental_search_distance: float = 2.0 * max_lump

    # Overlap detection
    jaccard_overlap_threshold: float = 0.1
    medoid_distance_threshold: float = max_lump

    # Close cluster refinement
    close_cluster_threshold: float = max_lump
    significant_difference_threshold: float = 0.2  # 20% of sequences must change clusters
    max_refinement_iterations: int = 5  # Prevent infinite refinement loops
    min_gap_improvement_threshold: float = 0.001  # Minimum gap improvement to continue refining
    processed_component_tracking: bool = True  # Track processed components to prevent re-processing

    # Incremental update
    max_closest_clusters: int = 5
    singleton_distance_threshold: float = max_lump

    # K-NN Graph Configuration
    knn_neighbors: int = 20  # Number of neighbors per cluster in K-NN graph
    knn_cluster_threshold: int = 1000  # Switch to K-NN graph above this cluster count
    blast_evalue_threshold: float = 1e-5  # BLAST e-value for medoid similarity search
    blast_max_hits: int = 50  # Maximum BLAST hits per medoid query (>= knn_neighbors)
    graph_rebuild_threshold: float = 0.3  # Rebuild graph when >30% of clusters change

    # Automatic scaling thresholds
    brute_force_max_clusters: int = 2000  # Use brute force below this threshold
    knn_graph_min_clusters: int = 1500   # Use K-NN graph above this threshold
    hybrid_overlap_range: int = 500      # Overlap range for hybrid approach
```

### K-NN Graph Parameters
```python
class KNNGraphConfig:
    """Configuration specific to BLAST-based K-NN graph construction."""

    # BLAST parameters for medoid similarity
    blast_word_size: int = 11
    blast_match_reward: int = 2
    blast_mismatch_penalty: int = -3
    blast_gap_open: int = 5
    blast_gap_extend: int = 2
    blast_dust_filter: bool = False  # Disable low-complexity filtering for short sequences
    blast_soft_masking: bool = False

    # Distance conversion parameters
    identity_distance_method: str = "adjusted"  # "adjusted" or "raw"
    distance_normalization: bool = True
    max_distance_cap: float = 1.0  # Cap distances at 100% dissimilarity

    # Graph maintenance
    incremental_update_batch_size: int = 100  # Batch size for incremental graph updates
    stale_edge_removal_threshold: int = 1000  # Remove stale edges after N updates
    graph_compression_threshold: float = 0.8  # Compress graph when >80% edges are active
```

## Expected Outcomes

### Quality Improvements

**MECE Clustering Guarantee:**
- All conflicts resolved through classic gapHACk reclustering
- No sequence assigned to multiple clusters in final output
- Maintains barcode gap optimization within reclustered scopes

**Reduced Close Clusters:**
- Systematic identification and refinement of suboptimal boundaries
- Global optimization applied to local cluster neighborhoods
- Prevention of close cluster proliferation through proactive merging

**Incremental Refinement Capability:**
- Efficient incorporation of new sequences without full reprocessing
- Maintains clustering quality during dataset evolution
- Supports real-time taxonomic database updates

### Scalability Improvements

**K-NN Graph Performance Gains:**

| Dataset Scale | Clusters | Brute Force | K-NN Graph (K=20) | Speedup |
|---------------|----------|-------------|-------------------|---------|
| Current | 1.6K | 1.3M calculations | 32K calculations | ~40x |
| Medium | 5K | 12.5M calculations | 100K calculations | ~125x |
| Target | 20K | 200M calculations | 400K calculations | ~500x |
| Full Scale | 50K | 1.25B calculations | 1M calculations | ~1,250x |

**Memory Efficiency:**
- **Brute Force**: O(C²) storage for full distance matrix
- **K-NN Graph**: O(C×K) storage for sparse neighbor lists
- **Target scale**: 400MB → 1.6MB memory reduction for 20K clusters

**Query Performance:**
- **Close cluster detection**: O(C²) → O(C×K) complexity reduction
- **Scope expansion**: O(C) → O(K) neighbor queries
- **Graph maintenance**: Incremental updates vs. full recomputation

### Performance Characteristics

**Automatic Scaling Strategy:**
```python
def select_proximity_graph_implementation(cluster_count: int) -> ClusterProximityGraph:
    if cluster_count < 1500:
        return BruteForceProximityGraph()  # Overhead not justified
    elif cluster_count < 2000:
        return HybridProximityGraph()      # Mixed approach for transition zone
    else:
        return BlastKNNProximityGraph()    # Essential for large scale
```

**Computational Efficiency:**
- **Scope-limited reclustering**: Maintains O(S³) complexity where S ≤ 300
- **Parallel processing**: Independent scope components processed concurrently
- **Distance reuse**: Leverage existing computations from decompose process
- **Graceful degradation**: Fallback strategies for oversized scopes

**Memory Management:**
- **Streaming BLAST**: Process medoid sequences without loading full database
- **Lazy graph construction**: Build K-NN connections on-demand
- **Garbage collection**: Automatic cleanup of stale graph edges
- **Memory-mapped storage**: Efficient storage for large K-NN graphs

## Risk Mitigation

### Infinite Loop Prevention

**Close Cluster Refinement:**
- Track processed component signatures to prevent reprocessing
- Skip components that don't significantly change after classic gapHACk
- Implement maximum iteration limits with fallback strategies

### Performance Degradation

**Oversized Scopes:**
- Strict size limits with fallback to conservative pairwise merging
- Early termination of scope expansion when limits approached
- Performance monitoring and adaptive parameter adjustment

### Quality Regression

**Validation Framework:**
- Comprehensive test suite with known-good reference datasets
- Automated MECE property validation
- Gap quality metrics comparison with baseline algorithms

This design provides a principled, scalable approach to achieving MECE clustering from gaphack-decompose while maintaining the computational advantages of the iterative approach. The scope-limited reclustering framework addresses each of the identified use cases while providing necessary performance safeguards and quality guarantees.