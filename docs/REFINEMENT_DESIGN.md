# Two-Pass Cluster Refinement Design

**Version**: 1.0
**Date**: 2025-01-07
**Status**: Design specification for implementation

---

## Overview

This document specifies a two-pass refinement architecture for gaphack-decompose that achieves:
1. **MECE property** (mutually exclusive, collectively exhaustive clustering)
2. **Comprehensive coverage** (every cluster refined at least once)
3. **Convergence** (iterates until cluster set stabilizes)
4. **Scalability** (bounded scope sizes, predictable performance)

### Key Design Changes from Current Implementation

1. **Retire connected component approach** → Use radius-based seeding
2. **Retire iterative context expansion** → Use distance-based context selection
3. **Density-independent scope construction** → Use distance thresholds, not sequence counts
4. **Add local convergence tracking** → Skip refined sets that have converged
5. **Guarantee comprehensive coverage** → Every cluster serves as seed

---

## Architecture: Two Passes

### Pass 1: Conflict Resolution + Individual Refinement
**Purpose**: Establish MECE property and split over-lumped clusters
**Tendency**: Increases cluster count (splits large clusters)
**Iterations**: Single pass (no convergence loop)

### Pass 2: Radius-Based Close Cluster Refinement
**Purpose**: Merge under-split clusters and optimize boundaries
**Tendency**: Decreases cluster count (merges close clusters)
**Iterations**: Multiple passes until convergence or limit reached

---

## Pass 1: Conflict Resolution + Individual Refinement

### Algorithm

```python
def pass1_resolve_and_split(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    conflicts: Dict[str, List[str]],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: RefinementConfig
) -> Tuple[Dict[str, List[str]], ProcessingStageInfo]:
    """
    Pass 1: Resolve conflicts and individually refine all clusters.

    Returns:
        Tuple of (refined_clusters, tracking_info)
    """
    tracking_info = ProcessingStageInfo(
        stage_name="Pass 1: Resolve and Split",
        clusters_before=all_clusters.copy()
    )

    # Step 1: Resolve conflicts using minimal scope (current approach)
    if conflicts:
        clusters_after_conflicts, conflict_info = resolve_conflicts(
            conflicts=conflicts,
            all_clusters=all_clusters,
            sequences=sequences,
            headers=headers,
            config=config,
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile
        )
        conflicted_cluster_ids = get_all_conflicted_cluster_ids(conflicts, all_clusters)
    else:
        clusters_after_conflicts = all_clusters.copy()
        conflicted_cluster_ids = set()

    # Step 2: Build proximity graph for context selection
    proximity_graph = ClusterGraph(
        clusters_after_conflicts, sequences, headers,
        k_neighbors=20, search_method="blast"
    )

    # Step 3: Individually refine every non-conflicted cluster
    final_clusters = clusters_after_conflicts.copy()

    for cluster_id in sorted(clusters_after_conflicts.keys()):
        if cluster_id in conflicted_cluster_ids:
            continue  # Already refined during conflict resolution

        # Refine this cluster individually with context
        scope_clusters, scope_sequences, scope_headers = build_refinement_scope(
            seed_clusters=[cluster_id],
            all_clusters=final_clusters,
            proximity_graph=proximity_graph,
            sequences=sequences,
            headers=headers,
            close_threshold=max_lump,  # Use max_lump as threshold for Pass 1
            config=config
        )

        # Apply full gapHACk to scope
        refined_clusters = apply_full_gaphack_to_scope(
            scope_sequences, scope_headers,
            min_split, max_lump, target_percentile
        )

        # Replace original cluster(s) with refined result
        for old_id in scope_clusters:
            if old_id in final_clusters:
                del final_clusters[old_id]

        for new_id, new_headers in refined_clusters.items():
            final_clusters[new_id] = new_headers

    tracking_info.clusters_after = final_clusters
    tracking_info.summary_stats = {
        'clusters_before': len(all_clusters),
        'clusters_after': len(final_clusters),
        'conflicts_resolved': len(conflicts),
        'individual_refinements': len(clusters_after_conflicts) - len(conflicted_cluster_ids)
    }

    return final_clusters, tracking_info
```

### Key Properties

- **Every cluster touched exactly once**: Either in conflict resolution or individual refinement
- **No iteration**: Single deterministic pass
- **Tends to split**: Individual refinement finds subclusters within large clusters
- **Context provided**: Even singleton clusters get context for gap calculation

---

## Pass 2: Radius-Based Close Cluster Refinement

### Algorithm

```python
def pass2_iterative_merge(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    close_threshold: float,
    max_iterations: int,
    config: RefinementConfig
) -> Tuple[Dict[str, List[str]], ProcessingStageInfo]:
    """
    Pass 2: Iteratively refine close clusters using radius-based seeding.
    Continues until convergence (no changes) or iteration limit reached.

    Args:
        close_threshold: Distance threshold for "close" clusters (typically max_lump)
        max_iterations: Maximum refinement iterations (default: 10)

    Returns:
        Tuple of (refined_clusters, tracking_info)
    """
    tracking_info = ProcessingStageInfo(
        stage_name="Pass 2: Iterative Merge",
        clusters_before=all_clusters.copy()
    )

    current_clusters = all_clusters.copy()
    global_iteration = 0

    # Track cluster signatures for equivalence checking
    cluster_signatures = compute_all_signatures(current_clusters)

    # Track converged scopes (sets of clusters that refined to themselves)
    converged_scopes = set()  # Set[frozenset[cluster_id]]

    while global_iteration < max_iterations:
        global_iteration += 1
        logger.info(f"Pass 2 iteration {global_iteration}: {len(current_clusters)} clusters")

        # Build proximity graph for current cluster state
        proximity_graph = ClusterGraph(
            current_clusters, sequences, headers,
            k_neighbors=20, search_method="blast"
        )

        # Track which clusters have been processed this iteration
        processed_this_iteration = set()

        # Collect all refinement operations for this iteration
        refinement_operations = []

        # Every cluster serves as seed (deterministic ID-based order)
        for seed_id in sorted(current_clusters.keys()):
            if seed_id in processed_this_iteration:
                continue  # Already processed as part of another seed's scope

            # Build refinement scope (seed + neighbors + context)
            scope_cluster_ids, scope_sequences, scope_headers = build_refinement_scope(
                seed_clusters=[seed_id],
                all_clusters=current_clusters,
                proximity_graph=proximity_graph,
                sequences=sequences,
                headers=headers,
                close_threshold=close_threshold,
                config=config
            )

            # Check if this scope has already converged
            scope_signature = frozenset(scope_cluster_ids)
            if scope_signature in converged_scopes:
                logger.debug(f"Skipping converged scope: {scope_signature}")
                processed_this_iteration.update(scope_cluster_ids)
                continue

            # Apply full gapHACk to scope
            refined_clusters = apply_full_gaphack_to_scope(
                scope_sequences, scope_headers,
                min_split, max_lump, target_percentile
            )

            # Store operation for batch execution
            refinement_operations.append({
                'seed_id': seed_id,
                'input_cluster_ids': set(scope_cluster_ids),
                'output_clusters': refined_clusters,
                'scope_signature': scope_signature
            })

            # Mark all input clusters as processed
            processed_this_iteration.update(scope_cluster_ids)

        # Execute all refinement operations and track changes
        next_clusters, changes_made, new_converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=refinement_operations,
            cluster_signatures=cluster_signatures
        )

        # Update converged scopes
        converged_scopes.update(new_converged)

        # Update cluster signatures for next iteration
        cluster_signatures = compute_all_signatures(next_clusters)

        # Check convergence: no changes made in this full pass
        if not changes_made:
            logger.info(f"Convergence achieved at iteration {global_iteration}")
            tracking_info.summary_stats['convergence_reason'] = 'no_changes'
            break

        # Alternative: Check full set equivalence (stricter)
        if check_full_set_equivalence(current_clusters, next_clusters):
            logger.info(f"Full set equivalence at iteration {global_iteration}")
            tracking_info.summary_stats['convergence_reason'] = 'set_equivalence'
            break

        current_clusters = next_clusters

    if global_iteration >= max_iterations:
        logger.warning(f"Reached iteration limit ({max_iterations}) without convergence")
        tracking_info.summary_stats['convergence_reason'] = 'iteration_limit'

    tracking_info.clusters_after = current_clusters
    tracking_info.summary_stats.update({
        'clusters_before': len(all_clusters),
        'clusters_after': len(current_clusters),
        'iterations': global_iteration,
        'converged_scopes_count': len(converged_scopes)
    })

    return current_clusters, tracking_info
```

### Iteration Counting

**Definition**: One iteration = one complete pass through all current cluster seeds

- Iteration 1: Process clusters {A, B, C, D, E} → produces {A', B', C', D', E', F'}
- Iteration 2: Process clusters {A', B', C', D', E', F'} → produces {A'', B'', C'', D''}
- And so on...

The number of seeds changes each iteration as clusters split/merge, but each full pass counts as one iteration.

---

## Core Algorithm: Build Refinement Scope

This replaces iterative context expansion with a simpler, single-pass approach.

```python
def build_refinement_scope(
    seed_clusters: List[str],
    all_clusters: Dict[str, List[str]],
    proximity_graph: ClusterGraph,
    sequences: List[str],
    headers: List[str],
    close_threshold: float,
    config: RefinementConfig
) -> Tuple[List[str], List[str], List[str]]:
    """
    Build refinement scope with three components:
    1. Seed clusters (core)
    2. Neighbor clusters within close_threshold (core neighbors)
    3. Context clusters between close_threshold and context_threshold (for gap calculation)

    Uses distance-based thresholds rather than sequence counts to ensure
    density-independent behavior that remains stable as datasets grow.

    Args:
        seed_clusters: Initial cluster(s) to refine (usually 1, could be multiple)
        close_threshold: Distance threshold for including core neighbors
        config: Contains max_full_gaphack_size, context_threshold_multiplier

    Returns:
        Tuple of (scope_cluster_ids, scope_sequences, scope_headers)
    """
    max_scope_size = config.max_full_gaphack_size  # Hard limit (default: 300)
    context_threshold = close_threshold * config.context_threshold_multiplier  # Default: 2.0× close_threshold

    # Step 1: Start with seed cluster(s)
    scope_cluster_ids = list(seed_clusters)
    current_size = sum(len(all_clusters[cid]) for cid in scope_cluster_ids)

    # Step 2: Collect all neighbors within close_threshold from all seeds
    core_neighbors = []
    for seed_id in seed_clusters:
        neighbors = proximity_graph.get_neighbors_within_distance(seed_id, close_threshold)
        for neighbor_id, distance in neighbors:
            if neighbor_id not in scope_cluster_ids:
                core_neighbors.append((neighbor_id, distance))

    # Deduplicate and sort by distance
    neighbor_dict = {}
    for neighbor_id, distance in core_neighbors:
        if neighbor_id not in neighbor_dict or distance < neighbor_dict[neighbor_id]:
            neighbor_dict[neighbor_id] = distance

    neighbors_sorted = sorted(neighbor_dict.items(), key=lambda x: x[1])

    # Step 3: Add core neighbors within close_threshold (closest first) up to max_scope_size
    for neighbor_id, distance in neighbors_sorted:
        neighbor_size = len(all_clusters[neighbor_id])
        if current_size + neighbor_size <= max_scope_size:
            scope_cluster_ids.append(neighbor_id)
            current_size += neighbor_size
        else:
            # Can't fit this neighbor - stop adding neighbors
            logger.warning(f"Scope size limit reached while adding core neighbors at distance {distance:.4f}")
            break

    # Step 4: Add context clusters beyond close_threshold up to context_threshold
    # This ensures inter-cluster distances for gap calculation
    context_candidates = []
    for seed_id in seed_clusters:
        # Get neighbors between close_threshold and context_threshold
        neighbors = proximity_graph.get_neighbors_within_distance(seed_id, context_threshold)
        for neighbor_id, distance in neighbors:
            if (distance > close_threshold and
                neighbor_id not in scope_cluster_ids):
                context_candidates.append((neighbor_id, distance))

    # Deduplicate and sort by distance
    context_dict = {}
    for neighbor_id, distance in context_candidates:
        if neighbor_id not in context_dict or distance < context_dict[neighbor_id]:
            context_dict[neighbor_id] = distance

    context_sorted = sorted(context_dict.items(), key=lambda x: x[1])

    # Add context clusters (closest first) up to max_scope_size
    context_added = 0
    for context_id, distance in context_sorted:
        context_size = len(all_clusters[context_id])
        if current_size + context_size <= max_scope_size:
            scope_cluster_ids.append(context_id)
            current_size += context_size
            context_added += 1
        else:
            # Would exceed max size - stop adding context
            break

    # Step 5: Ensure at least one context cluster for gap calculation
    # If we have core neighbors but no context, gap calculation may fail
    if context_added == 0 and len(neighbors_sorted) > 0:
        # We have core neighbors but no context - try to add at least one
        # Look for any neighbor beyond context_threshold (relaxed distance requirement)
        extended_candidates = []
        for seed_id in seed_clusters:
            all_neighbors = proximity_graph.get_k_nearest_neighbors(seed_id, k=30)
            for neighbor_id, distance in all_neighbors:
                if (distance > close_threshold and
                    neighbor_id not in scope_cluster_ids):
                    extended_candidates.append((neighbor_id, distance))

        if extended_candidates:
            # Sort by distance and try to add closest available context
            extended_sorted = sorted(extended_candidates, key=lambda x: x[1])
            for context_id, distance in extended_sorted:
                context_size = len(all_clusters[context_id])
                if current_size + context_size <= max_scope_size:
                    scope_cluster_ids.append(context_id)
                    current_size += context_size
                    context_added += 1
                    logger.debug(f"Added extended context at distance {distance:.4f} to ensure gap calculation")
                    break  # Just need one

    # Step 6: Extract sequences and headers for scope
    scope_headers_set = set()
    for cluster_id in scope_cluster_ids:
        scope_headers_set.update(all_clusters[cluster_id])

    scope_headers = sorted(scope_headers_set)  # Deterministic order
    header_to_idx = {h: i for i, h in enumerate(headers)}
    scope_sequences = [sequences[header_to_idx[h]] for h in scope_headers]

    logger.debug(f"Built scope: {len(scope_cluster_ids)} clusters, "
                f"{len(scope_headers)} sequences, "
                f"seeds={seed_clusters}, "
                f"core_neighbors={len(neighbors_sorted)}, "
                f"context_added={context_added}, "
                f"size={current_size}")

    return scope_cluster_ids, scope_sequences, scope_headers
```

### Context Selection Properties

1. **Deterministic**: Same inputs always produce same scope
2. **Bounded**: Never exceeds max_scope_size (hard limit)
3. **Density-independent**: Uses distance thresholds, not sequence counts
   - Behavior stable as datasets grow
   - Same threshold produces similar scopes across different dataset densities
4. **Prioritized**:
   - Seeds (always included)
   - Core neighbors within close_threshold (sorted by distance)
   - Context neighbors between close_threshold and context_threshold (sorted by distance)
   - Extended context beyond context_threshold (only if needed for gap calculation)

### Design Rationale: Distance Thresholds vs. Sequence Counts

**Problem with sequence count approach** (e.g., "add context to reach 250 sequences"):
- Dense regions (many nearby sequences) → few distant context clusters
- Sparse regions (few nearby sequences) → many distant context clusters
- Results depend on local sequence density
- Unstable as new sequences added to dataset

**Advantage of distance threshold approach** (e.g., "add context up to 2× close_threshold"):
- Same threshold in dense and sparse regions
- Behavior driven by biological distance, not sampling density
- Results stable as dataset grows
- More principled and interpretable

**Example**:
```
Region A (dense): 50 sequences within 0.02 distance
Region B (sparse): 5 sequences within 0.02 distance

Sequence count approach:
  - Region A: add context to reach 250 → add few distant clusters
  - Region B: add context to reach 250 → add many distant clusters
  - Different refinement behavior despite similar biology

Distance threshold approach:
  - Both regions: add context up to 0.04 distance (2× 0.02)
  - Same refinement behavior regardless of sampling density
  - Consistent results as new sequences discovered
```

---

## Core Algorithm: Execute Refinement Operations

Handles overlapping scopes and tracks convergence.

```python
def execute_refinement_operations(
    current_clusters: Dict[str, List[str]],
    operations: List[Dict],
    cluster_signatures: Dict[str, frozenset]
) -> Tuple[Dict[str, List[str]], bool, Set[frozenset]]:
    """
    Execute all refinement operations, handling overlaps and tracking changes.

    Args:
        current_clusters: Current cluster state
        operations: List of refinement operations to apply
        cluster_signatures: Mapping of cluster_id → frozenset(headers)

    Returns:
        Tuple of (next_clusters, changes_made, new_converged_scopes)
    """
    next_clusters = current_clusters.copy()
    changes_made = False
    new_converged_scopes = set()

    for op in operations:
        seed_id = op['seed_id']
        input_cluster_ids = op['input_cluster_ids']
        output_clusters = op['output_clusters']
        scope_signature = op['scope_signature']

        # Check if all input clusters still exist (not consumed by earlier op)
        inputs_still_exist = all(cid in next_clusters for cid in input_cluster_ids)

        if not inputs_still_exist:
            logger.debug(f"Skipping operation for seed {seed_id} - inputs already consumed")
            continue

        # Check equivalence: are outputs identical to inputs?
        is_unchanged = check_cluster_set_equivalence(
            input_cluster_ids=input_cluster_ids,
            output_clusters=output_clusters,
            current_clusters=next_clusters,
            cluster_signatures=cluster_signatures
        )

        if is_unchanged:
            # No changes - mark scope as converged
            new_converged_scopes.add(scope_signature)
            logger.debug(f"Scope converged: {scope_signature}")
            continue

        # Apply refinement: remove inputs, add outputs
        for input_id in input_cluster_ids:
            if input_id in next_clusters:
                del next_clusters[input_id]

        for output_id, output_headers in output_clusters.items():
            next_clusters[output_id] = output_headers

        changes_made = True

    return next_clusters, changes_made, new_converged_scopes


def check_cluster_set_equivalence(
    input_cluster_ids: Set[str],
    output_clusters: Dict[str, List[str]],
    current_clusters: Dict[str, List[str]],
    cluster_signatures: Dict[str, frozenset]
) -> bool:
    """
    Check if output clusters are equivalent to input clusters.

    Equivalence means: same set of sequence clusters (order-independent).

    Returns:
        True if refinement produced no changes (converged)
    """
    # Get input signatures
    input_signatures = {cluster_signatures[cid] for cid in input_cluster_ids}

    # Get output signatures
    output_signatures = {frozenset(headers) for headers in output_clusters.values()}

    # Check set equivalence
    return input_signatures == output_signatures


def compute_all_signatures(clusters: Dict[str, List[str]]) -> Dict[str, frozenset]:
    """Compute frozenset signatures for all clusters."""
    return {
        cluster_id: frozenset(headers)
        for cluster_id, headers in clusters.items()
    }


def check_full_set_equivalence(clusters1: Dict[str, List[str]],
                               clusters2: Dict[str, List[str]]) -> bool:
    """Check if two cluster dictionaries contain identical cluster sets."""
    sigs1 = {frozenset(headers) for headers in clusters1.values()}
    sigs2 = {frozenset(headers) for headers in clusters2.values()}
    return sigs1 == sigs2
```

---

## Convergence Properties

### Local Convergence
A **scope** (set of clusters) has locally converged when:
```
gapHACk(scope) produces identical cluster set (order-independent)
```

Once a scope converges, it is added to `converged_scopes` and skipped in future iterations.

### Global Convergence
The **entire cluster set** has converged when:
```
No changes made in a complete pass through all seeds
```

This is detected by `changes_made = False` after executing all operations.

### Convergence Guarantees

**Will converge if**:
- Gap-optimized clustering is deterministic (same inputs → same outputs) ✓
- Seed order is deterministic (sorted by ID) ✓
- Scope construction is deterministic ✓
- No cyclic refinements (A+B→C+D, then C+D→A+B)

**Protection against non-convergence**:
- Iteration limit (default: 10)
- Converged scope tracking (reduces wasted computation)
- Full iteration history logging (detect oscillation patterns)

### Expected Convergence Behavior

**Typical trajectory**:
- Iteration 1-2: Major merging (close clusters combined)
- Iteration 3-4: Boundary adjustments (small changes)
- Iteration 5+: Minimal changes → convergence

**Most datasets**: Converge in 3-5 iterations
**Complex datasets**: May reach iteration limit (10) with stable but non-converged state

---

## Implementation Phases

### Phase 1: Pass 1 Implementation
**Goal**: Conflict resolution + individual refinement

**Tasks**:
1. Implement `pass1_resolve_and_split()`
2. Implement `get_all_conflicted_cluster_ids()` helper
3. Update `resolve_conflicts()` to return conflicted cluster IDs
4. Add tracking/logging for Pass 1 operations
5. Add tests for Pass 1 (conflict resolution + individual refinement)

**Deliverable**: Pass 1 working independently, can be called from refine CLI

### Phase 2: Pass 2 Basic Implementation
**Goal**: Radius-based refinement without convergence tracking

**Tasks**:
1. Implement `build_refinement_scope()` (single-pass context selection)
2. Implement `pass2_iterative_merge()` basic version (no convergence tracking)
3. Implement `execute_refinement_operations()` with overlap handling
4. Add iteration counting and logging
5. Add tests for Pass 2 basic functionality

**Deliverable**: Pass 2 working with iteration limit, no convergence optimization

### Phase 3: Convergence Tracking
**Goal**: Add local convergence detection and scope skipping

**Tasks**:
1. Implement `check_cluster_set_equivalence()`
2. Implement `compute_all_signatures()`
3. Add `converged_scopes` tracking to Pass 2
4. Add convergence detection and early stopping
5. Add tests for convergence behavior

**Deliverable**: Pass 2 with full convergence tracking and optimization

### Phase 4: Integration and Testing
**Goal**: Wire both passes into refine CLI and validate on real data

**Tasks**:
1. Update `gaphack-refine` CLI to call both passes
2. Add command-line flags: `--pass1-only`, `--pass2-only`, `--max-iterations`
3. Add comprehensive logging and progress tracking
4. Test on Russula dataset (1,429 sequences)
5. Performance benchmarking and optimization
6. Documentation updates

**Deliverable**: Production-ready two-pass refinement

---

## Configuration Parameters

```python
@dataclass
class RefinementConfig:
    """Configuration for two-pass refinement."""

    # Scope sizing
    max_full_gaphack_size: int = 300      # Hard limit for gapHACk input
    context_threshold_multiplier: float = 2.0  # Context distance = close_threshold × this

    # Pass 2 parameters
    close_threshold: Optional[float] = None  # Distance for "close" (default: max_lump)
    max_iterations: int = 10              # Maximum Pass 2 iterations

    # Proximity graph
    k_neighbors: int = 20                 # K-NN graph parameter
    search_method: str = "blast"          # "blast" or "vsearch"

    # Legacy parameters (for backward compatibility)
    conflict_expansion_threshold: Optional[float] = None
    close_cluster_expansion_threshold: Optional[float] = None
```

### Default Values Rationale

- **max_full_gaphack_size = 300**: Empirically determined limit where gapHACk remains tractable
- **context_threshold_multiplier = 2.0**: Context extends to 2× close_threshold
  - Provides inter-cluster distances for gap calculation
  - Density-independent (stable as datasets grow)
  - Can be adjusted based on data characteristics
- **k_neighbors = 20**: Sufficient for local neighborhood without excessive computation
- **max_iterations = 10**: Conservative limit, most datasets converge in 3-5

---

## Testing Strategy

### Unit Tests

1. **Scope construction**:
   - Single seed with no neighbors
   - Single seed with neighbors within threshold
   - Seed with neighbors exceeding max_scope_size (pruning)
   - Context addition when below preferred_scope_size

2. **Convergence detection**:
   - Identical input/output clusters (converged)
   - Different input/output clusters (not converged)
   - Full set equivalence check

3. **Operation execution**:
   - Non-overlapping operations
   - Overlapping operations (input consumption)
   - Converged scope skipping

### Integration Tests

1. **Pass 1**:
   - Dataset with conflicts → all conflicts resolved
   - Dataset with large clusters → clusters split
   - Every cluster refined exactly once

2. **Pass 2**:
   - Dataset with close clusters → clusters merged
   - Convergence achieved within iteration limit
   - Converged scopes skipped in subsequent iterations

3. **Two-pass pipeline**:
   - Pass 1 → Pass 2 → final MECE result
   - Cluster count trajectory (increase in Pass 1, decrease in Pass 2)
   - Quality metrics (ARI, homogeneity, completeness)

### Performance Tests

1. **Scalability**: 100, 300, 500, 1000 sequences
2. **Convergence speed**: Iterations to convergence by dataset size
3. **Scope size distribution**: Verify scopes stay within limits

---

## Migration Path

### Current Code → New Design

**Keep**:
- `resolve_conflicts()` (minimal scope approach)
- `apply_full_gaphack_to_scope()`
- `ClusterGraph` infrastructure
- `ProcessingStageInfo` tracking

**Replace**:
- `find_connected_close_components()` → radius-based seeding
- `expand_context_for_gap_optimization()` → `build_refinement_scope()`
- `refine_close_clusters()` → `pass2_iterative_merge()`

**Add**:
- `pass1_resolve_and_split()`
- `execute_refinement_operations()`
- Convergence tracking data structures

### Backward Compatibility

To support existing workflows, provide CLI flags:
```bash
# Legacy mode (current behavior)
gaphack-refine --legacy

# New two-pass mode (default)
gaphack-refine --pass1-only  # Stop after Pass 1
gaphack-refine --pass2-only  # Skip Pass 1 (assume MECE input)
gaphack-refine               # Both passes with convergence
```

---

## Open Questions and Future Work

### Q1: Seed Selection Optimization
**Current**: ID-based (deterministic, arbitrary)
**Alternative**: Size-based (largest first) or connectivity-based (most neighbors first)

**Experiment**: Compare convergence speed with different seed orderings

### Q2: Iteration Limit Tuning
**Current**: Default 10 iterations
**Question**: Is this sufficient for large datasets? Too many for small datasets?

**Action**: Collect empirical data on convergence iterations across dataset sizes

### Q3: Adaptive Thresholds
**Current**: Fixed `close_threshold` (typically `max_lump`)
**Alternative**: Adaptive threshold based on distance distribution

**Future**: Could adjust threshold between iterations based on cluster density

### Q4: Parallelization
**Current**: Sequential processing of seeds
**Opportunity**: Parallel refinement of non-overlapping scopes

**Challenge**: Detecting non-overlapping scopes requires proximity graph queries

### Q5: Incremental Updates
**Current**: Full proximity graph rebuild each iteration
**Optimization**: Incremental graph updates after refinement

**Tradeoff**: Complexity vs. performance gain

---

## Success Criteria

The implementation will be considered successful when:

1. ✓ **MECE property guaranteed**: No conflicts after Pass 1
2. ✓ **Comprehensive coverage**: Every cluster refined at least once
3. ✓ **Convergence achievable**: 80%+ of test datasets converge within 10 iterations
4. ✓ **Bounded performance**: Scopes never exceed max_scope_size
5. ✓ **Quality improvement**: ARI ≥ 0.85, Homogeneity ≥ 0.90, Completeness ≥ 0.85 on Russula
6. ✓ **Scalability**: Handle 1,000+ sequences within reasonable time (<30 minutes)
7. ✓ **Reproducibility**: Deterministic results from same input

---

## Known Issues and Follow-Up Work

### 🔴 High Priority (Active Work)

#### Issue 1: Summary Reporting Doesn't Match New Architecture
**Location**: `refine_cli.py:224-427` (generate_refinement_summary)

**Problem**: The summary report still assumes the old two-stage model (conflict resolution + close cluster refinement) rather than the new two-pass architecture.

**Missing reporting**:
- Individual refinement statistics from Pass 1 (how many non-conflicted clusters were refined)
- Pass 2 iteration count and convergence reason (no_changes, set_equivalence, iteration_limit)
- Converged scope counts (how many scopes reached local convergence)
- Whether refinement stopped due to convergence vs iteration limit

**Fix**: Rewrite summary generation sections to properly reflect:
- Pass 1: Conflict Resolution + Individual Refinement
  - Conflicts resolved: X sequences
  - Individual refinements: Y clusters
  - Tendency: splitting (clusters increase)
- Pass 2: Iterative Close Cluster Refinement
  - Iterations: N (max: M)
  - Convergence reason: [no_changes|set_equivalence|iteration_limit]
  - Converged scopes: K
  - Tendency: merging (clusters decrease)

#### Issue 3: Comprehensive Integration Test Coverage
**Location**: Missing test files

**Problem**: Zero tests exist for the new implementation:
- `pass1_resolve_and_split()`
- `pass2_iterative_merge()`
- `build_refinement_scope()` with distance-based thresholds
- `execute_refinement_operations()`
- `two_pass_refinement()` entry point

**Required tests** (from design doc Section "Testing Strategy"):

**Unit Tests**:
- Scope construction with distance thresholds
  - Single seed with no neighbors → only seed returned
  - Seed with neighbors within threshold → neighbors added
  - Neighbors exceeding max_scope_size → pruned by distance
  - Context addition with distance-based thresholds
  - Ensuring at least one context cluster
- Convergence detection
  - Identical input/output clusters → converged
  - Different input/output clusters → not converged
  - Full set equivalence check
- Operation execution
  - Non-overlapping operations → all applied
  - Overlapping operations → input consumption handled
  - Converged scope skipping

**Integration Tests**:
- Pass 1: Dataset with conflicts → all conflicts resolved, every cluster refined
- Pass 1: Dataset with large clusters → clusters split
- Pass 2: Dataset with close clusters → clusters merged
- Pass 2: Convergence within iteration limit
- Pass 2: Converged scopes skipped in subsequent iterations
- Two-pass pipeline: Pass 1 → Pass 2 → final MECE result

**Test Files**:
- `tests/test_two_pass_refinement.py` - Unit tests for helper functions (exists)
- `tests/test_two_pass_russula.py` - Integration tests on Russula data (exists, needs completion)

#### Issue 11: Remove Legacy Implementations (NEW)
**Location**: `cluster_refinement.py`, `refine_cli.py`, `RefinementConfig`

**Problem**: Legacy code still present, creating maintenance burden and confusion. No backwards compatibility required.

**Legacy Functions to Remove**:
1. **`cluster_refinement.py`**:
   - `expand_context_for_gap_optimization()` (replaced by `build_refinement_scope()`)
   - `find_connected_close_components()` (replaced by radius-based seeding)
   - Old `refine_close_clusters()` (replaced by `pass2_iterative_merge()`)

2. **`refine_cli.py`**:
   - `--use-legacy` flag and all associated logic
   - Old two-stage execution path
   - Legacy-specific argument handling

3. **`RefinementConfig`**:
   - `preferred_scope_size` (deprecated - was for N+N pruning)
   - `expansion_size_buffer` (deprecated - was for iterative expansion)
   - `conflict_expansion_threshold` (deprecated - replaced by `close_threshold`)
   - `close_cluster_expansion_threshold` (deprecated - replaced by `close_threshold`)

**Benefits**:
- Cleaner codebase with single clear path
- No confusion between old and new approaches
- Easier to maintain and understand
- Better foundation for future work

---

### ⏸️ Deferred (Different Approach Later)

#### Issue 2: Timing Breakdown is Approximate
**Location**: `refine_cli.py:865-886` (timing extraction from tracking_stages)

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Current implementation uses crude approximations for timing display.

#### Issue 4: Quality Metrics Validation on Full Russula Dataset
**Location**: Missing validation test

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Need quality metrics validation on full Russula dataset (1,429 sequences).

#### Issue 5: CLI Validation for Invalid Flag Combinations
**Location**: `refine_cli.py:651-830` (argument parsing and validation)

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Missing validations for invalid flag combinations.

#### Issue 6: Progress Reporting During Pass 2 Iterations
**Location**: `cluster_refinement.py:655-818` (pass2_iterative_merge)

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Limited progress reporting during Pass 2 iterations.

#### Issue 7: Cluster Mapping Report Accuracy
**Location**: `refine_cli.py:151-221` (generate_cluster_mapping_report)

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Cluster mapping report may be inaccurate for two-pass mode.

#### Issue 8: Error Handling for Edge Cases

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Unclear behavior for edge cases (graph construction fails, scopes exceed max size, non-convergence, missing context).

#### Issue 9: Detailed Convergence Analytics
**Location**: `pass2_iterative_merge` convergence tracking

**Status**: DEFERRED - Will approach from different angle later

**Original Problem**: Missing detailed convergence analytics (per-iteration stats, oscillation detection, trajectory visualization).

---

### 📋 Stable Checkpoint (Do Later)

#### Issue 10: Documentation Updates

**Status**: Will be done in single documentation pass when codebase reaches stable checkpoint

**Scope**:
- README.md: Update usage examples
- CLAUDE.md: Add two-pass architecture section
- Docstrings: Verify completeness

---

## Implementation Status & Execution Plan

### ✅ Completed (2025-01-07)
- Core two-pass refinement architecture
- Distance-based scope construction
- Convergence tracking (local and global)
- Helper functions for cluster signatures
- Basic CLI integration
- Basic unit tests (`test_two_pass_refinement.py`)
- Integration test scaffolding (`test_two_pass_russula.py`)

### 🔴 Active Work (3 Phases)

**Phase 1: Remove Legacy Code** (Issue 11)
- Remove legacy functions from `cluster_refinement.py`
- Remove `--use-legacy` flag and logic from `refine_cli.py`
- Clean up deprecated `RefinementConfig` parameters
- Update affected tests
- **Goal**: Clean codebase with single clear architecture

**Phase 2: Fix Summary Reporting** (Issue 1)
- Rewrite `generate_refinement_summary()` to match two-pass architecture
- Pass 1 section: conflicts resolved, individual refinements, splitting tendency
- Pass 2 section: iterations, convergence reason, converged scopes, merging tendency
- Simplified timing display (detailed metrics deferred)
- **Goal**: Accurate reporting that reflects actual implementation

**Phase 3: Comprehensive Integration Tests** (Issue 3)
- Complete existing integration tests in `test_two_pass_russula.py`
- Add edge case scenarios (scope construction, Pass 1/Pass 2 specific behaviors)
- Run full test suite on Russula 100/300 subsets
- **Goal**: Confidence in correctness and robustness

### ⏸️ Deferred
- Issues 2, 4, 5, 6, 7, 8, 9 (different approach later)

### 📋 Future
- Issue 10 (documentation pass when stable)

---

## References

- `cluster_refinement.py`: Current refinement implementation
- `cluster_graph.py`: K-NN proximity graph infrastructure
- `core.py`: Gap-optimized clustering algorithm
- `tests/PHASE4_SUMMARY.md`: Quality metrics and testing documentation
- `refine_cli.py`: CLI implementation with two-pass integration
