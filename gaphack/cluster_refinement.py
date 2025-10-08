"""
Cluster refinement algorithms for achieving conflict-free clustering from gaphack-decompose.

This module implements scope-limited refinement using full gapHACk to resolve
conflicts, refine close clusters, and handle incremental updates.
"""

import logging
import copy
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union
import numpy as np

from .cluster_graph import ClusterGraph
from .distance_providers import DistanceProvider
from .core import GapOptimizedClustering
from .decompose import DecomposeResults
from .utils import calculate_distance_matrix

logger = logging.getLogger(__name__)


class RefinementConfig:
    """Configuration for two-pass cluster refinement.

    This configuration supports the new two-pass architecture:
    - Pass 1: Conflict resolution + individual refinement (tendency to split)
    - Pass 2: Radius-based close cluster refinement (tendency to merge)
    """

    def __init__(self,
                 max_full_gaphack_size: int = 300,
                 context_threshold_multiplier: float = 2.0,
                 close_threshold: Optional[float] = None,
                 max_iterations: int = 10,
                 k_neighbors: int = 20,
                 search_method: str = "blast"):
        """Initialize refinement configuration.

        Args:
            max_full_gaphack_size: Hard limit for gapHACk input size (default: 300)
            context_threshold_multiplier: Context distance = close_threshold × this (default: 2.0)
            close_threshold: Distance threshold for "close" clusters (default: max_lump)
            max_iterations: Maximum Pass 2 iterations (default: 10)
            k_neighbors: K-NN graph parameter (default: 20)
            search_method: "blast" or "vsearch" for proximity graph
        """
        self.max_full_gaphack_size = max_full_gaphack_size
        self.context_threshold_multiplier = context_threshold_multiplier
        self.close_threshold = close_threshold
        self.max_iterations = max_iterations
        self.k_neighbors = k_neighbors
        self.search_method = search_method


# ============================================================================
# Helper Functions for Two-Pass Refinement
# ============================================================================

def compute_all_signatures(clusters: Dict[str, List[str]]) -> Dict[str, frozenset]:
    """Compute frozenset signatures for all clusters.

    Signatures are order-independent representations used for equivalence checking.

    Args:
        clusters: Dict mapping cluster_id -> list of sequence headers

    Returns:
        Dict mapping cluster_id -> frozenset(headers)
    """
    return {
        cluster_id: frozenset(headers)
        for cluster_id, headers in clusters.items()
    }


def check_full_set_equivalence(clusters1: Dict[str, List[str]],
                               clusters2: Dict[str, List[str]]) -> bool:
    """Check if two cluster dictionaries contain identical cluster sets.

    Order-independent comparison using frozenset signatures.

    Args:
        clusters1: First cluster dictionary
        clusters2: Second cluster dictionary

    Returns:
        True if both contain the same set of clusters (order-independent)
    """
    sigs1 = {frozenset(headers) for headers in clusters1.values()}
    sigs2 = {frozenset(headers) for headers in clusters2.values()}
    return sigs1 == sigs2


def check_cluster_set_equivalence(
    input_cluster_ids: Set[str],
    output_clusters: Dict[str, List[str]],
    current_clusters: Dict[str, List[str]],
    cluster_signatures: Dict[str, frozenset]
) -> bool:
    """Check if output clusters are equivalent to input clusters.

    Equivalence means: same set of sequence clusters (order-independent).

    Args:
        input_cluster_ids: Set of input cluster IDs
        output_clusters: Dict of output cluster_id -> headers
        current_clusters: Current cluster state (for looking up inputs)
        cluster_signatures: Pre-computed signatures for current clusters

    Returns:
        True if refinement produced no changes (converged)
    """
    # Get input signatures
    input_signatures = {cluster_signatures[cid] for cid in input_cluster_ids
                       if cid in cluster_signatures}

    # Get output signatures
    output_signatures = {frozenset(headers) for headers in output_clusters.values()}

    # Check set equivalence
    return input_signatures == output_signatures


def get_all_conflicted_cluster_ids(conflicts: Dict[str, List[str]],
                                   all_clusters: Dict[str, List[str]]) -> Set[str]:
    """Extract all cluster IDs involved in conflicts.

    Args:
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        all_clusters: Dict mapping cluster_id -> list of sequence headers

    Returns:
        Set of all cluster IDs that contain conflicted sequences
    """
    conflicted_ids = set()
    for seq_id, cluster_ids in conflicts.items():
        conflicted_ids.update(cluster_ids)

    # Verify these clusters actually exist
    return {cid for cid in conflicted_ids if cid in all_clusters}


def build_refinement_scope(
    seed_clusters: List[str],
    all_clusters: Dict[str, List[str]],
    proximity_graph: ClusterGraph,
    sequences: List[str],
    headers: List[str],
    close_threshold: float,
    config: RefinementConfig
) -> Tuple[List[str], List[str], List[str]]:
    """Build refinement scope with distance-based thresholds (not sequence counts).

    Constructs a refinement scope with three components:
    1. Seed clusters (core)
    2. Neighbor clusters within close_threshold (core neighbors)
    3. Context clusters between close_threshold and context_threshold (for gap calculation)

    Uses distance-based thresholds rather than sequence counts to ensure
    density-independent behavior that remains stable as datasets grow.

    Args:
        seed_clusters: Initial cluster(s) to refine (usually 1, could be multiple)
        all_clusters: All available clusters
        proximity_graph: Graph for finding neighbors
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        close_threshold: Distance threshold for including core neighbors
        config: Contains max_full_gaphack_size, context_threshold_multiplier

    Returns:
        Tuple of (scope_cluster_ids, scope_sequences, scope_headers)
    """
    max_scope_size = config.max_full_gaphack_size
    context_threshold = close_threshold * config.context_threshold_multiplier  # Default: 2.0× close_threshold

    # Step 1: Start with seed cluster(s)
    scope_cluster_ids = list(seed_clusters)
    current_size = sum(len(all_clusters[cid]) for cid in scope_cluster_ids if cid in all_clusters)

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
        if neighbor_id not in all_clusters:
            continue
        neighbor_size = len(all_clusters[neighbor_id])
        if current_size + neighbor_size <= max_scope_size:
            scope_cluster_ids.append(neighbor_id)
            current_size += neighbor_size
        else:
            # Can't fit this neighbor - stop adding neighbors
            logger.debug(f"Scope size limit reached while adding core neighbors at distance {distance:.4f}")
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
        if context_id not in all_clusters:
            continue
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
                    neighbor_id not in scope_cluster_ids and
                    neighbor_id in all_clusters):
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
        if cluster_id in all_clusters:
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


def find_conflict_components(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]]) -> List[List[str]]:
    """Group conflicted clusters into connected components.

    Two clusters are connected if they share any conflicted sequence.

    Args:
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        all_clusters: Dict mapping cluster_id -> list of sequence headers

    Returns:
        List of connected components, each being a list of cluster IDs
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

    def dfs_traverse(cluster_id: str, component: List[str]) -> None:
        visited.add(cluster_id)
        component.append(cluster_id)
        for neighbor in cluster_graph[cluster_id]:
            if neighbor not in visited:
                dfs_traverse(neighbor, component)

    for cluster_id in cluster_graph:
        if cluster_id not in visited:
            component = []
            dfs_traverse(cluster_id, component)
            components.append(component)

    return components


def apply_full_gaphack_to_scope_with_metadata(scope_sequences: List[str], scope_headers: List[str],
                                              min_split: float = 0.005, max_lump: float = 0.02,
                                              target_percentile: int = 95, cluster_id_generator=None) -> Tuple[Dict[str, List[str]], Dict]:
    """Apply full gapHACk clustering to a scope of sequences, returning clusters and metadata.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Tuple of (cluster_dict, metadata_dict) where metadata includes gap_size
    """
    # Create MSA-based distance provider for scope
    # This provides consistent alignment across all sequences in the scope
    from .distance_providers import MSACachedDistanceProvider
    msa_provider = MSACachedDistanceProvider(scope_sequences, scope_headers)

    # Build distance matrix from MSA
    distance_matrix = msa_provider.build_distance_matrix()

    # Apply full gapHACk clustering
    clusterer = GapOptimizedClustering(
        min_split=min_split,
        max_lump=max_lump,
        target_percentile=target_percentile,
        show_progress=True,  # Disable progress for scope-limited clustering
        logger=logger
    )

    # Get clustering result with metadata
    final_clusters, singletons, metadata = clusterer.cluster(distance_matrix)

    # Convert result to cluster dictionary format
    clusters = {}

    # Import here to avoid circular imports
    from .decompose import ClusterIDGenerator

    # Use provided generator or create a temporary one
    if cluster_id_generator is None:
        # For internal refinement, use "refined" stage with no specific count
        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

    # Process multi-member clusters (list of lists of indices)
    for cluster_idx, cluster_indices_list in enumerate(final_clusters):
        cluster_headers = []
        for seq_idx in cluster_indices_list:
            cluster_headers.append(scope_headers[seq_idx])

        cluster_id = cluster_id_generator.next_id()
        clusters[cluster_id] = cluster_headers

    # Process singletons (list of indices)
    for singleton_idx in singletons:
        cluster_headers = [scope_headers[singleton_idx]]
        cluster_id = cluster_id_generator.next_id()
        clusters[cluster_id] = cluster_headers

    # Extract gap size from metadata
    gap_size = metadata.get('best_config', {}).get('gap_size', float('-inf'))

    logger.debug(f"Classic gapHACk on scope: {len(scope_sequences)} sequences → {len(clusters)} clusters, gap={gap_size:.4f}")
    return clusters, {'gap_size': gap_size, 'metadata': metadata}


def apply_full_gaphack_to_scope(scope_sequences: List[str], scope_headers: List[str], min_split: float = 0.005,
                                max_lump: float = 0.02, target_percentile: int = 95, cluster_id_generator=None) -> Dict[str, List[str]]:
    """Apply full gapHACk clustering to a scope of sequences.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Dict mapping cluster_id -> list of sequence headers
    """
    # Use the metadata version and just return the clusters
    clusters, _ = apply_full_gaphack_to_scope_with_metadata(scope_sequences, scope_headers, min_split, max_lump,
                                                            target_percentile, cluster_id_generator)
    return clusters


# ============================================================================
# Pass 1: Conflict Resolution + Individual Refinement
# ============================================================================

def pass1_resolve_and_split(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    conflicts: Dict[str, List[str]],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: RefinementConfig,
    cluster_id_generator=None
) -> Tuple[Dict[str, List[str]], 'ProcessingStageInfo']:
    """Pass 1: Resolve conflicts and individually refine all clusters.

    This pass ensures MECE property and has a tendency to split large clusters.
    Every cluster is touched exactly once: either in conflict resolution or
    individual refinement.

    Args:
        all_clusters: Current cluster dictionary
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters (also used as close_threshold for Pass 1)
        target_percentile: Percentile for gap optimization
        config: Configuration for refinement parameters
        cluster_id_generator: Optional cluster ID generator

    Returns:
        Tuple of (refined_clusters, tracking_info)
    """
    import time
    from .decompose import ProcessingStageInfo, ClusterIDGenerator

    # Initialize cluster ID generator if not provided
    if cluster_id_generator is None:
        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

    tracking_info = ProcessingStageInfo(
        stage_name="Pass 1: Resolve and Split",
        clusters_before=all_clusters.copy()
    )

    logger.info(f"=== Pass 1: Resolve and Split ===")
    logger.info(f"Starting with {len(all_clusters)} clusters")

    # Timing tracking
    pass1_start = time.time()
    timing = {}

    # Step 1: Resolve conflicts using minimal scope (current approach)
    conflict_start = time.time()
    if conflicts:
        logger.info(f"Resolving {len(conflicts)} conflicts...")
        clusters_after_conflicts, conflict_info = resolve_conflicts(
            conflicts=conflicts,
            all_clusters=all_clusters,
            sequences=sequences,
            headers=headers,
            config=config,
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile,
            cluster_id_generator=cluster_id_generator
        )
        conflicted_cluster_ids = get_all_conflicted_cluster_ids(conflicts, all_clusters)
        logger.info(f"Conflicts resolved: {len(all_clusters)} clusters → {len(clusters_after_conflicts)} clusters")
    else:
        logger.info("No conflicts to resolve")
        clusters_after_conflicts = all_clusters.copy()
        conflicted_cluster_ids = set()
    timing['conflict_resolution'] = time.time() - conflict_start

    # Step 2: Build proximity graph for context selection
    graph_start = time.time()
    logger.info("Building proximity graph for individual refinement...")
    proximity_graph = ClusterGraph(
        clusters_after_conflicts, sequences, headers,
        k_neighbors=config.k_neighbors,
        search_method=config.search_method
    )
    timing['proximity_graph'] = time.time() - graph_start

    # Step 3: Individually refine every non-conflicted cluster
    refinement_start = time.time()
    final_clusters = clusters_after_conflicts.copy()
    non_conflicted_count = len(clusters_after_conflicts) - len(conflicted_cluster_ids)

    logger.info(f"Individually refining {non_conflicted_count} non-conflicted clusters...")

    refined_count = 0
    for cluster_id in sorted(clusters_after_conflicts.keys()):
        if cluster_id in conflicted_cluster_ids:
            continue  # Already refined during conflict resolution

        # Skip if cluster was already processed as part of another scope
        if cluster_id not in final_clusters:
            logger.debug(f"Skipping {cluster_id} - already processed in another scope")
            continue

        # Skip if cluster is empty (shouldn't happen, but defensive check)
        if not final_clusters[cluster_id]:
            logger.warning(f"Skipping {cluster_id} - empty cluster")
            continue

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
            min_split, max_lump, target_percentile,
            cluster_id_generator=cluster_id_generator
        )

        # Replace original cluster(s) with refined result
        for old_id in scope_clusters:
            if old_id in final_clusters:
                del final_clusters[old_id]

        for new_id, new_headers in refined_clusters.items():
            final_clusters[new_id] = new_headers

        refined_count += 1
        if refined_count % 10 == 0:
            logger.debug(f"Progress: {refined_count}/{non_conflicted_count} clusters refined")

    timing['individual_refinement'] = time.time() - refinement_start
    timing['individual_refinement_avg'] = timing['individual_refinement'] / non_conflicted_count if non_conflicted_count > 0 else 0.0
    timing['total'] = time.time() - pass1_start

    logger.info(f"Pass 1 complete: {len(all_clusters)} clusters → {len(final_clusters)} clusters "
               f"({len(final_clusters) - len(all_clusters):+d})")
    logger.info(f"Pass 1 timing: conflict={timing['conflict_resolution']:.1f}s, "
               f"graph={timing['proximity_graph']:.1f}s, "
               f"refinement={timing['individual_refinement']:.1f}s, "
               f"total={timing['total']:.1f}s")

    tracking_info.clusters_after = final_clusters
    tracking_info.summary_stats = {
        'clusters_before': len(all_clusters),
        'clusters_after': len(final_clusters),
        'cluster_count_change': len(final_clusters) - len(all_clusters),
        'conflicts_resolved': len(conflicts),
        'individual_refinements': non_conflicted_count,
        'timing': timing
    }

    return final_clusters, tracking_info


# ============================================================================
# Pass 2: Radius-Based Close Cluster Refinement
# ============================================================================

def execute_refinement_operations(
    current_clusters: Dict[str, List[str]],
    operations: List[Dict],
    cluster_signatures: Dict[str, frozenset]
) -> Tuple[Dict[str, List[str]], bool, Set[frozenset]]:
    """Execute all refinement operations, handling overlaps and tracking changes.

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


def pass2_iterative_merge(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    close_threshold: float,
    max_iterations: int,
    config: RefinementConfig,
    cluster_id_generator=None,
    show_progress: bool = False
) -> Tuple[Dict[str, List[str]], 'ProcessingStageInfo']:
    """Pass 2: Iteratively refine close clusters using radius-based seeding.

    Continues until convergence (no changes) or iteration limit reached.
    Every cluster serves as a seed in each iteration.

    Args:
        all_clusters: Current cluster dictionary
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        close_threshold: Distance threshold for "close" clusters (typically max_lump)
        max_iterations: Maximum refinement iterations (default: 10)
        config: Configuration for refinement parameters
        cluster_id_generator: Optional cluster ID generator
        show_progress: Show progress bar for each iteration

    Returns:
        Tuple of (refined_clusters, tracking_info)
    """
    import time
    from tqdm import tqdm
    from .decompose import ProcessingStageInfo, ClusterIDGenerator

    # Initialize cluster ID generator if not provided
    if cluster_id_generator is None:
        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

    tracking_info = ProcessingStageInfo(
        stage_name="Pass 2: Iterative Merge",
        clusters_before=all_clusters.copy()
    )

    logger.info(f"=== Pass 2: Iterative Merge ===")
    logger.info(f"Starting with {len(all_clusters)} clusters")
    logger.info(f"Close threshold: {close_threshold:.4f}, Max iterations: {max_iterations}")

    pass2_start = time.time()
    timing = {
        'iterations': [],  # Per-iteration timing
        'proximity_graphs': [],  # Per-graph timing
        'refinements': []  # Per-iteration refinement timing
    }

    current_clusters = all_clusters.copy()
    global_iteration = 0

    # Track cluster signatures for equivalence checking
    cluster_signatures = compute_all_signatures(current_clusters)

    # Track converged scopes (sets of clusters that refined to themselves)
    converged_scopes = set()  # Set[frozenset[cluster_id]]

    while global_iteration < max_iterations:
        iteration_start = time.time()
        global_iteration += 1
        logger.info(f"Pass 2 iteration {global_iteration}: {len(current_clusters)} clusters")

        # Build proximity graph for current cluster state
        graph_start = time.time()
        proximity_graph = ClusterGraph(
            current_clusters, sequences, headers,
            k_neighbors=config.k_neighbors,
            search_method=config.search_method
        )
        graph_time = time.time() - graph_start
        timing['proximity_graphs'].append(graph_time)

        # Track which clusters have been processed this iteration
        processed_this_iteration = set()

        # Collect all refinement operations for this iteration
        refinement_operations = []

        # Every cluster serves as seed (deterministic ID-based order)
        refinement_start = time.time()
        seed_list = sorted(current_clusters.keys())

        if show_progress:
            pbar = tqdm(total=len(seed_list),
                       desc=f"Pass 2 Iteration {global_iteration}",
                       unit=" seeds")

        for seed_id in seed_list:
            if seed_id in processed_this_iteration:
                if show_progress:
                    pbar.update(1)
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
                min_split, max_lump, target_percentile,
                cluster_id_generator=cluster_id_generator
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

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        refinement_time = time.time() - refinement_start
        timing['refinements'].append(refinement_time)

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

        iteration_time = time.time() - iteration_start
        timing['iterations'].append(iteration_time)

        logger.info(f"Iteration {global_iteration}: {len(current_clusters)} → {len(next_clusters)} clusters "
                   f"({len(next_clusters) - len(current_clusters):+d}), "
                   f"{len(new_converged)} scopes converged, "
                   f"time={iteration_time:.1f}s (graph={graph_time:.1f}s, refinement={refinement_time:.1f}s)")

        # Check convergence: no changes made in this full pass
        if not changes_made:
            logger.info(f"Convergence achieved at iteration {global_iteration} (no changes)")
            tracking_info.summary_stats = {'convergence_reason': 'no_changes'}
            break

        # Alternative: Check full set equivalence (stricter)
        if check_full_set_equivalence(current_clusters, next_clusters):
            logger.info(f"Convergence achieved at iteration {global_iteration} (set equivalence)")
            tracking_info.summary_stats = {'convergence_reason': 'set_equivalence'}
            break

        current_clusters = next_clusters

    if global_iteration >= max_iterations:
        logger.warning(f"Reached iteration limit ({max_iterations}) without convergence")
        tracking_info.summary_stats = {'convergence_reason': 'iteration_limit'}

    timing['total'] = time.time() - pass2_start
    timing['avg_iteration'] = sum(timing['iterations']) / len(timing['iterations']) if timing['iterations'] else 0.0
    timing['avg_graph'] = sum(timing['proximity_graphs']) / len(timing['proximity_graphs']) if timing['proximity_graphs'] else 0.0
    timing['avg_refinement'] = sum(timing['refinements']) / len(timing['refinements']) if timing['refinements'] else 0.0

    logger.info(f"Pass 2 complete: {len(all_clusters)} clusters → {len(current_clusters)} clusters "
               f"({len(current_clusters) - len(all_clusters):+d})")
    logger.info(f"Pass 2 timing: total={timing['total']:.1f}s, "
               f"avg_iteration={timing['avg_iteration']:.1f}s, "
               f"avg_graph={timing['avg_graph']:.1f}s, "
               f"avg_refinement={timing['avg_refinement']:.1f}s")

    tracking_info.clusters_after = current_clusters
    if tracking_info.summary_stats is None:
        tracking_info.summary_stats = {}
    tracking_info.summary_stats.update({
        'clusters_before': len(all_clusters),
        'clusters_after': len(current_clusters),
        'cluster_count_change': len(current_clusters) - len(all_clusters),
        'iterations': global_iteration,
        'converged_scopes_count': len(converged_scopes),
        'timing': timing
    })

    return current_clusters, tracking_info


def two_pass_refinement(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    conflicts: Dict[str, List[str]],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: Optional[RefinementConfig] = None,
    run_pass1: bool = True,
    run_pass2: bool = True,
    cluster_id_generator=None,
    show_progress: bool = False
) -> Tuple[Dict[str, List[str]], List['ProcessingStageInfo']]:
    """Two-pass cluster refinement: resolve conflicts, split, then merge.

    This is the main entry point for the new refinement architecture.

    Pass 1: Conflict resolution + individual refinement (tendency to split)
    Pass 2: Radius-based close cluster refinement (tendency to merge)

    Args:
        all_clusters: Current cluster dictionary
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        config: Configuration for refinement parameters
        run_pass1: Whether to run Pass 1 (default: True)
        run_pass2: Whether to run Pass 2 (default: True)
        cluster_id_generator: Optional cluster ID generator
        show_progress: Show progress bars during Pass 2 iterations

    Returns:
        Tuple of (refined_clusters, tracking_info_list)
    """
    if config is None:
        config = RefinementConfig()

    from .decompose import ClusterIDGenerator
    if cluster_id_generator is None:
        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

    tracking_stages = []
    current_clusters = all_clusters.copy()

    logger.info("=" * 80)
    logger.info("TWO-PASS CLUSTER REFINEMENT")
    logger.info("=" * 80)
    logger.info(f"Initial clusters: {len(all_clusters)}")
    logger.info(f"Conflicts: {len(conflicts)}")
    logger.info(f"Config: max_scope={config.max_full_gaphack_size}, "
               f"context_multiplier={config.context_threshold_multiplier}, "
               f"max_iterations={config.max_iterations}")

    # Pass 1: Conflict resolution + individual refinement
    if run_pass1:
        clusters_after_pass1, pass1_tracking = pass1_resolve_and_split(
            all_clusters=current_clusters,
            sequences=sequences,
            headers=headers,
            conflicts=conflicts,
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile,
            config=config,
            cluster_id_generator=cluster_id_generator
        )
        tracking_stages.append(pass1_tracking)
        current_clusters = clusters_after_pass1
        logger.info(f"After Pass 1: {len(current_clusters)} clusters")
    else:
        logger.info("Skipping Pass 1 (run_pass1=False)")

    # Pass 2: Radius-based close cluster refinement
    if run_pass2:
        # Determine close threshold (default to max_lump if not specified)
        close_threshold = config.close_threshold if config.close_threshold is not None else max_lump

        clusters_after_pass2, pass2_tracking = pass2_iterative_merge(
            all_clusters=current_clusters,
            sequences=sequences,
            headers=headers,
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile,
            close_threshold=close_threshold,
            max_iterations=config.max_iterations,
            config=config,
            cluster_id_generator=cluster_id_generator,
            show_progress=show_progress
        )
        tracking_stages.append(pass2_tracking)
        current_clusters = clusters_after_pass2
        logger.info(f"After Pass 2: {len(current_clusters)} clusters")
    else:
        logger.info("Skipping Pass 2 (run_pass2=False)")

    logger.info("=" * 80)
    logger.info(f"TWO-PASS REFINEMENT COMPLETE")
    logger.info(f"Final result: {len(all_clusters)} clusters → {len(current_clusters)} clusters "
               f"({len(current_clusters) - len(all_clusters):+d})")
    logger.info("=" * 80)

    return current_clusters, tracking_stages


def resolve_conflicts(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]],
                                     sequences: List[str],
                                     headers: List[str],
                                     config: Optional[RefinementConfig] = None,
                                     min_split: float = 0.005,
                                     max_lump: float = 0.02,
                                     target_percentile: int = 95,
                                     cluster_id_generator=None) -> Tuple[Dict[str, List[str]], 'ProcessingStageInfo']:
    """Resolve assignment conflicts using full gapHACk refinement with minimal scope.

    Uses only conflicted clusters (no expansion) for fastest, most predictable conflict-free fixes.
    This is pure correctness operation - quality improvement belongs to close cluster refinement.

    Args:
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        all_clusters: Dict mapping cluster_id -> list of sequence headers
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        config: Configuration for refinement parameters
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Tuple of (updated_clusters, tracking_info): Updated cluster dictionary with conflicts resolved and tracking information
    """
    if config is None:
        config = RefinementConfig()

    # Initialize tracking
    from .decompose import ProcessingStageInfo
    tracking_info = ProcessingStageInfo(
        stage_name="Conflict Resolution",
        clusters_before=all_clusters.copy(),
        summary_stats={
            'conflicts_count': len(conflicts),
            'conflicted_sequences': list(conflicts.keys()),
            'clusters_before_count': len(all_clusters)
        }
    )

    if not conflicts:
        logger.info("No conflicts to resolve")
        result = all_clusters.copy()
        tracking_info.clusters_after = result
        tracking_info.summary_stats.update({
            'clusters_after_count': len(result),
            'components_processed_count': 0
        })
        return result, tracking_info

    logger.info(f"Resolving conflicts for {len(conflicts)} sequences across clusters")

    # Step 1: Group conflicts by connected components
    conflict_components = find_conflict_components(conflicts, all_clusters)
    logger.debug(f"Found {len(conflict_components)} connected conflict components")

    updated_clusters = all_clusters.copy()

    # Process each conflict component
    for component_idx, component_clusters in enumerate(conflict_components):
        logger.debug(f"Processing conflict component {component_idx+1}/{len(conflict_components)} "
                    f"with {len(component_clusters)} clusters")

        # Step 2: Extract scope sequences (minimal scope - only conflicted clusters)
        scope_headers_set = set()
        for cluster_id in component_clusters:
            scope_headers_set.update(all_clusters[cluster_id])

        scope_headers = list(scope_headers_set)  # Headers same as sequences for decompose

        # Track component before processing
        component_info = {
            'component_index': component_idx,
            'clusters_before': list(component_clusters),
            'clusters_before_count': len(component_clusters),
            'sequences_count': len(scope_headers_set),
            'processed': False
        }

        # Step 3: Apply full gapHACk to minimal conflict scope (no expansion)
        if len(scope_headers_set) == 0:
            logger.warning(f"Skipping conflict component {component_idx+1} - empty scope")
            component_info['processed'] = False
            component_info['clusters_after'] = []
            component_info['clusters_after_count'] = 0
            components_processed.append(component_info)
            continue

        if len(scope_headers_set) <= config.max_full_gaphack_size:
            logger.debug(f"Applying full gapHACk to minimal conflict scope of {len(scope_headers_set)} sequences")

            # Map headers to actual sequences
            header_to_idx = {h: i for i, h in enumerate(headers)}
            scope_sequence_list = [sequences[header_to_idx[h]] for h in scope_headers]

            full_result = apply_full_gaphack_to_scope(scope_sequence_list, scope_headers, min_split, max_lump,
                                                      target_percentile, cluster_id_generator)

            # Step 4: Replace original conflicted clusters with classic result
            for cluster_id in component_clusters:
                if cluster_id in updated_clusters:
                    del updated_clusters[cluster_id]

            # Add new clusters from classic result
            for cluster_id, cluster_headers in full_result.items():
                updated_clusters[cluster_id] = cluster_headers

            # Update component tracking with destination clusters
            component_info.update({
                'clusters_after': list(full_result.keys()),
                'clusters_after_count': len(full_result),
                'processed': True
            })

            logger.info(f"Resolved conflict component: {len(component_clusters)} clusters → "
                       f"{len(full_result)} clusters")

        else:
            # Fallback: skip oversized components with warning
            logger.warning(f"Skipping conflict component with {len(scope_headers_set)} sequences "
                          f"(exceeds limit of {config.max_full_gaphack_size})")
            component_info.update({
                'clusters_after': list(component_clusters),  # Unchanged
                'clusters_after_count': len(component_clusters),
                'processed': False,
                'skipped_reason': 'oversized'
            })

        # Add component info to tracking
        tracking_info.components_processed.append(component_info)

    # Verify conflicts are resolved
    remaining_conflicts = 0
    for seq_id, cluster_ids in conflicts.items():
        active_clusters = [cid for cid in cluster_ids if cid in updated_clusters]
        if len(active_clusters) > 1:
            remaining_conflicts += 1

    if remaining_conflicts > 0:
        logger.warning(f"{remaining_conflicts} conflicts remain unresolved")
    else:
        logger.info("All conflicts successfully resolved")

    # Finalize tracking information
    tracking_info.clusters_after = updated_clusters
    tracking_info.summary_stats.update({
        'clusters_after_count': len(updated_clusters),
        'components_processed_count': len(conflict_components),
        'remaining_conflicts_count': remaining_conflicts
    })
    return updated_clusters, tracking_info


# ============================================================================
# Legacy functions removed (replaced by two-pass refinement)
# ============================================================================
# - find_connected_close_components() → replaced by radius-based seeding
# - expand_context_for_gap_optimization() → replaced by build_refinement_scope()
# - refine_close_clusters() → replaced by pass2_iterative_merge()
# See docs/REFINEMENT_DESIGN.md for migration details


def verify_no_conflicts(clusters: Dict[str, List[str]],
                                   original_conflicts: Optional[Dict[str, List[str]]] = None,
                                   context: str = "final") -> Dict[str, any]:
    """Perform comprehensive verification of cluster assignments for conflict-free property.

    Scans all cluster assignments to detect:
    1. Remaining conflicts (sequences in multiple clusters)
    2. Assignment coverage and consistency
    3. Comparison with original conflict set if provided

    Args:
        clusters: Dictionary mapping cluster_id -> list of sequence headers
        original_conflicts: Optional original conflicts for comparison
        context: Context string for logging (e.g., "final", "after_resolution")

    Returns:
        Dictionary with verification results:
        - 'conflicts': Dict[str, List[str]] - remaining conflicts
        - 'conflict_count': int - number of conflicted sequences
        - 'total_sequences': int - total unique sequences across all clusters
        - 'total_assignments': int - total sequence assignments (with duplicates)
        - 'no_conflicts': bool - True if conflict-free (no conflicts)
        - 'new_conflicts': Dict[str, List[str]] - conflicts not in original set
        - 'unresolved_conflicts': Dict[str, List[str]] - original conflicts still present
        - 'resolved_conflicts': Dict[str, List[str]] - original conflicts that were resolved
    """
    logger.info(f"Performing comprehensive conflict-free verification ({context})")

    # Build sequence -> cluster mapping to detect conflicts
    sequence_assignments = defaultdict(list)
    total_assignments = 0

    for cluster_id, sequence_headers in clusters.items():
        for seq_header in sequence_headers:
            sequence_assignments[seq_header].append(cluster_id)
            total_assignments += 1

    # Identify current conflicts
    current_conflicts = {}
    for seq_header, cluster_list in sequence_assignments.items():
        if len(cluster_list) > 1:
            current_conflicts[seq_header] = sorted(cluster_list)  # Sort for consistency

    # Calculate basic metrics
    total_sequences = len(sequence_assignments)
    conflict_count = len(current_conflicts)
    no_conflicts = conflict_count == 0

    # Compare with original conflicts if provided
    new_conflicts = {}
    unresolved_conflicts = {}
    resolved_conflicts = {}

    if original_conflicts:
        # Find conflicts that are new (not in original set)
        for seq_id, cluster_ids in current_conflicts.items():
            if seq_id not in original_conflicts:
                new_conflicts[seq_id] = cluster_ids

        # Check resolution status of original conflicts
        for seq_id, original_cluster_ids in original_conflicts.items():
            if seq_id in current_conflicts:
                # Conflict still exists - check if it's the same or different
                current_cluster_ids = current_conflicts[seq_id]
                unresolved_conflicts[seq_id] = {
                    'original': sorted(original_cluster_ids),
                    'current': sorted(current_cluster_ids)
                }
            else:
                # Conflict was resolved
                resolved_conflicts[seq_id] = original_cluster_ids

    # Log verification results
    logger.info(f"conflict-free Verification Results ({context}):")
    logger.info(f"  Total sequences: {total_sequences}")
    logger.info(f"  Total assignments: {total_assignments}")
    logger.info(f"  Conflicted sequences: {conflict_count}")
    logger.info(f"  conflict-free property satisfied: {no_conflicts}")

    if original_conflicts:
        original_count = len(original_conflicts)
        resolved_count = len(resolved_conflicts)
        unresolved_count = len(unresolved_conflicts)
        new_count = len(new_conflicts)

        logger.info(f"  Original conflicts: {original_count}")
        logger.info(f"  Resolved conflicts: {resolved_count}")
        logger.info(f"  Unresolved conflicts: {unresolved_count}")
        logger.info(f"  New conflicts: {new_count}")

        if resolved_count > 0:
            resolution_rate = (resolved_count / original_count) * 100
            logger.info(f"  Resolution rate: {resolution_rate:.1f}%")

    # Log detailed conflict information for debugging
    if current_conflicts:
        logger.warning(f"conflict-free property violated: {conflict_count} sequences in multiple clusters")
        if conflict_count <= 10:  # Log details for small conflict sets
            for seq_id, cluster_ids in current_conflicts.items():
                logger.warning(f"  Conflict: {seq_id} → clusters {cluster_ids}")
        else:
            # Sample of conflicts for large sets
            sample_conflicts = list(current_conflicts.items())[:5]
            for seq_id, cluster_ids in sample_conflicts:
                logger.warning(f"  Conflict: {seq_id} → clusters {cluster_ids}")
            logger.warning(f"  ... and {conflict_count - 5} more conflicts")

    if new_conflicts:
        logger.error(f"NEW conflicts detected: {len(new_conflicts)} sequences have conflicts not in original set")
        for seq_id, cluster_ids in new_conflicts.items():
            logger.error(f"  New conflict: {seq_id} → clusters {cluster_ids}")

    if unresolved_conflicts:
        logger.warning(f"UNRESOLVED conflicts: {len(unresolved_conflicts)} original conflicts remain")
        for seq_id, conflict_info in list(unresolved_conflicts.items())[:5]:  # Log first 5
            orig_clusters = conflict_info['original']
            curr_clusters = conflict_info['current']
            logger.warning(f"  Unresolved: {seq_id} → was {orig_clusters}, now {curr_clusters}")

    # Final status determination - critical for catching missed conflicts
    if context.startswith("final"):
        if no_conflicts:
            logger.info(f"✓ FINAL VERIFICATION PASSED: conflict-free property satisfied - no conflicts detected in final cluster assignments")
        else:
            logger.error(f"✗ FINAL VERIFICATION FAILED: conflict-free property violated - {conflict_count} conflicts remain in final output")
            logger.error("This indicates conflicts were missed or introduced during processing!")

        # Log summary of what was processed
        logger.info(f"Final verification summary: {total_sequences} sequences across {len(clusters)} clusters")

        # If this is truly the final verification, any conflicts are critical
        if conflict_count > 0:
            logger.error("CRITICAL: Final output contains conflicts - clustering is not conflict-free!")
            logger.error("This may indicate:")
            logger.error("  1. Conflicts exceeded MAX_FULL_GAPHACK_SIZE and were skipped")
            logger.error("  2. New conflicts were introduced during full gapHACk refinement")
            logger.error("  3. Scope expansion was insufficient to capture all related conflicts")
            logger.error("  4. Multi-cluster conflicts were not properly handled")

    # Return comprehensive results
    return {
        'conflicts': current_conflicts,
        'conflict_count': conflict_count,
        'total_sequences': total_sequences,
        'total_assignments': total_assignments,
        'no_conflicts': no_conflicts,
        'new_conflicts': new_conflicts,
        'unresolved_conflicts': unresolved_conflicts,
        'resolved_conflicts': resolved_conflicts,
        'verification_context': context,
        'critical_failure': context.startswith("final") and not no_conflicts
    }