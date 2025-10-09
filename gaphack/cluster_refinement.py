"""
Cluster refinement algorithms for achieving conflict-free clustering from gaphack-decompose.

This module implements scope-limited refinement using full gapHACk to resolve
conflicts, refine close clusters, and handle incremental updates.
"""

import logging
import copy
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from .cluster_graph import ClusterGraph
from .distance_providers import DistanceProvider
from .core import GapOptimizedClustering
from .decompose import DecomposeResults
from .utils import calculate_distance_matrix

logger = logging.getLogger("gaphack.refine")


def compute_ami_for_refinement(input_clusters: Set[str], output_clusters: Dict[str, List[str]],
                                all_clusters: Dict[str, List[str]]) -> float:
    """Compute adjusted mutual information between input and output clusters.

    Args:
        input_clusters: Set of input cluster IDs
        output_clusters: Dict of output cluster_id -> headers
        all_clusters: Dict containing all clusters (for looking up input clusters)

    Returns:
        AMI score (0-1, where 1 is perfect agreement)
    """
    # Collect all headers involved
    all_headers = set()
    for cid in input_clusters:
        if cid in all_clusters:
            all_headers.update(all_clusters[cid])

    for headers in output_clusters.values():
        all_headers.update(headers)

    if len(all_headers) < 2:
        return 1.0  # Perfect agreement for trivial cases

    # Create sorted header list for consistent indexing
    header_list = sorted(all_headers)
    header_to_idx = {h: i for i, h in enumerate(header_list)}

    # Create label arrays
    input_labels = np.full(len(header_list), -1, dtype=int)
    output_labels = np.full(len(header_list), -1, dtype=int)

    # Assign input cluster labels
    for cluster_idx, cid in enumerate(input_clusters):
        if cid in all_clusters:
            for header in all_clusters[cid]:
                if header in header_to_idx:
                    input_labels[header_to_idx[header]] = cluster_idx

    # Assign output cluster labels
    for cluster_idx, (_, headers) in enumerate(output_clusters.items()):
        for header in headers:
            if header in header_to_idx:
                output_labels[header_to_idx[header]] = cluster_idx

    # Only compute AMI for sequences present in both
    valid_mask = (input_labels >= 0) & (output_labels >= 0)
    if valid_mask.sum() < 2:
        return 1.0  # Perfect agreement if too few sequences

    # Suppress sklearn warning about many clusters looking like regression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        return adjusted_mutual_info_score(input_labels[valid_mask], output_labels[valid_mask])


class RefinementConfig:
    """Configuration for two-pass cluster refinement.

    This configuration supports the new two-pass architecture:
    - Pass 1: Conflict resolution + individual refinement (tendency to split)
    - Pass 2: Radius-based close cluster refinement (tendency to merge)
    """

    def __init__(self,
                 max_full_gaphack_size: int = 300,
                 close_threshold: Optional[float] = None,
                 max_iterations: int = 10,
                 k_neighbors: int = 20,
                 search_method: str = "blast",
                 random_seed: Optional[int] = None):
        """Initialize refinement configuration.

        Args:
            max_full_gaphack_size: Hard limit for gapHACk input size (default: 300)
            close_threshold: Distance threshold for "close" clusters (default: max_lump)
            max_iterations: Maximum Pass 2 iterations (default: 10)
            k_neighbors: K-NN graph parameter (default: 20)
            search_method: "blast" or "vsearch" for proximity graph
            random_seed: Seed for randomizing seed order in Pass 2 (default: None for random seed)
        """
        self.max_full_gaphack_size = max_full_gaphack_size
        self.close_threshold = close_threshold
        self.max_iterations = max_iterations
        self.k_neighbors = k_neighbors
        self.search_method = search_method
        self.random_seed = random_seed


# ============================================================================
# Helper Functions for Two-Pass Refinement
# ============================================================================

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
    max_lump: float,
    close_threshold: float,
    config: RefinementConfig
) -> Tuple[List[str], List[str], List[str]]:
    """Build refinement scope with distance-based thresholds (not sequence counts).

    Constructs a refinement scope with three components:
    1. Seed clusters (core)
    2. Core scope clusters within max_lump (merge candidates)
    3. Context clusters between max_lump and close_threshold (for inter-cluster distances in gap calculation)

    Uses distance-based thresholds rather than sequence counts to ensure
    density-independent behavior that remains stable as datasets grow.

    Args:
        seed_clusters: Initial cluster(s) to refine (usually 1, could be multiple)
        all_clusters: All available clusters
        proximity_graph: Graph for finding neighbors
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        max_lump: Distance threshold for core scope (merge candidates)
        close_threshold: Outer distance threshold for context
        config: Contains max_full_gaphack_size

    Returns:
        Tuple of (scope_cluster_ids, scope_sequences, scope_headers)
    """
    max_scope_size = config.max_full_gaphack_size

    # Step 1: Start with seed cluster(s)
    scope_cluster_ids = list(seed_clusters)
    current_size = sum(len(all_clusters[cid]) for cid in scope_cluster_ids if cid in all_clusters)

    # Step 2: Collect all neighbors within max_lump from all seeds (core scope - merge candidates)
    core_neighbors = []
    for seed_id in seed_clusters:
        neighbors = proximity_graph.get_neighbors_within_distance(seed_id, max_lump)
        for neighbor_id, distance in neighbors:
            if neighbor_id not in scope_cluster_ids:
                core_neighbors.append((neighbor_id, distance))

    # Deduplicate and sort by distance
    neighbor_dict = {}
    for neighbor_id, distance in core_neighbors:
        if neighbor_id not in neighbor_dict or distance < neighbor_dict[neighbor_id]:
            neighbor_dict[neighbor_id] = distance

    neighbors_sorted = sorted(neighbor_dict.items(), key=lambda x: x[1])

    # Step 3: Add core neighbors within max_lump (closest first) up to max_scope_size
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

    # Step 4: Add context clusters between max_lump and close_threshold (for gap calculation)
    # Search all neighbors within close_threshold, filter to those beyond max_lump
    context_candidates = []
    for seed_id in seed_clusters:
        # Get all neighbors within close_threshold
        all_neighbors = proximity_graph.get_neighbors_within_distance(seed_id, close_threshold)
        for neighbor_id, distance in all_neighbors:
            # Keep only those beyond max_lump (context zone)
            if (distance > max_lump and
                neighbor_id not in scope_cluster_ids and
                neighbor_id in all_clusters):
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

    # Step 5: Extract sequences and headers for scope
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
                                              target_percentile: int = 95, cluster_id_generator=None,
                                              quiet: bool = True) -> Tuple[Dict[str, List[str]], Dict]:
    """Apply full gapHACk clustering to a scope of sequences, returning clusters and metadata.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        cluster_id_generator: Optional cluster ID generator
        quiet: If True, suppress verbose logging (default: True)

    Returns:
        Tuple of (cluster_dict, metadata_dict) where metadata includes gap_size
    """
    # Create MSA-based distance provider for scope
    # This provides consistent alignment across all sequences in the scope
    from .distance_providers import MSACachedDistanceProvider
    msa_provider = MSACachedDistanceProvider(scope_sequences, scope_headers)

    # Build distance matrix from MSA
    distance_matrix = msa_provider.build_distance_matrix()

    # Create a custom logger that suppresses INFO logs if quiet mode is enabled
    if quiet:
        quiet_logger = logging.getLogger(__name__ + ".quiet")
        quiet_logger.setLevel(logging.WARNING)
    else:
        quiet_logger = logger

    # Apply full gapHACk clustering
    clusterer = GapOptimizedClustering(
        min_split=min_split,
        max_lump=max_lump,
        target_percentile=target_percentile,
        show_progress=False,  # Disable progress for scope-limited clustering
        logger=quiet_logger
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
                                max_lump: float = 0.02, target_percentile: int = 95, cluster_id_generator=None,
                                quiet: bool = True) -> Dict[str, List[str]]:
    """Apply full gapHACk clustering to a scope of sequences.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        cluster_id_generator: Optional cluster ID generator
        quiet: If True, suppress verbose logging (default: True)

    Returns:
        Dict mapping cluster_id -> list of sequence headers
    """
    # Use the metadata version and just return the clusters
    clusters, _ = apply_full_gaphack_to_scope_with_metadata(scope_sequences, scope_headers, min_split, max_lump,
                                                            target_percentile, cluster_id_generator, quiet)
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

    # Step 2: Individually refine every non-conflicted cluster in isolation (no neighbors, no context)
    # This allows splitting of non-cohesive clusters
    refinement_start = time.time()
    final_clusters = clusters_after_conflicts.copy()
    non_conflicted_count = len(clusters_after_conflicts) - len(conflicted_cluster_ids)
    total_clusters = len(clusters_after_conflicts)  # Total for progress counter

    logger.info(f"Individually refining {non_conflicted_count} non-conflicted clusters in isolation...")

    # Map headers to indices for sequence extraction
    header_to_idx = {h: i for i, h in enumerate(headers)}

    clusters_processed = 0  # Track cumulative input clusters
    for cluster_id in sorted(clusters_after_conflicts.keys()):
        if cluster_id in conflicted_cluster_ids:
            continue  # Already refined during conflict resolution

        # Skip if cluster was already processed
        if cluster_id not in final_clusters:
            logger.debug(f"Skipping {cluster_id} - already processed")
            continue

        # Skip if cluster is empty (shouldn't happen, but defensive check)
        if not final_clusters[cluster_id]:
            logger.warning(f"Skipping {cluster_id} - empty cluster")
            continue

        # Extract sequences for this cluster only (isolated refinement)
        cluster_headers = final_clusters[cluster_id]
        cluster_sequences = [sequences[header_to_idx[h]] for h in cluster_headers]

        # Apply full gapHACk to this cluster in isolation
        refined_clusters, metadata = apply_full_gaphack_to_scope_with_metadata(
            cluster_sequences, cluster_headers,
            min_split, max_lump, target_percentile,
            cluster_id_generator=cluster_id_generator,
            quiet=True
        )

        # Extract gap info and compute AMI for logging
        num_input = 1  # Always 1 cluster input in isolated mode
        num_output = len(refined_clusters)
        gap_size = metadata.get('gap_size', float('-inf'))

        # Compute AMI between input and output
        ami = compute_ami_for_refinement({cluster_id}, refined_clusters, final_clusters)

        # Format gap info
        gap_info_str = ""
        best_config = metadata.get('metadata', {}).get('best_config', {})
        gap_metrics = best_config.get('gap_metrics')
        if gap_metrics:
            target_key = f'p{target_percentile}'
            if target_key in gap_metrics:
                intra = gap_metrics[target_key].get('intra_upper', 0.0)
                inter = gap_metrics[target_key].get('inter_lower', 0.0)
                gap_info_str = f"Gap {gap_size:.4f} (intra≤{intra:.4f}, inter≥{inter:.4f})"
        elif gap_size > float('-inf'):
            gap_info_str = f"Gap {gap_size:.4f}"

        # Replace original cluster with refined result
        del final_clusters[cluster_id]
        for new_id, new_headers in refined_clusters.items():
            final_clusters[new_id] = new_headers

        clusters_processed += 1
        logger.info(f"Pass 1 ({clusters_processed} of {total_clusters}): "
                   f"{num_input} -> {num_output} clusters.  {gap_info_str}.  AMI {ami:.3f}")

    timing['individual_refinement'] = time.time() - refinement_start
    timing['individual_refinement_avg'] = timing['individual_refinement'] / non_conflicted_count if non_conflicted_count > 0 else 0.0
    timing['total'] = time.time() - pass1_start

    # Compute AMI between input and output clusters
    pass1_ami = compute_ami_for_refinement(
        set(all_clusters.keys()),
        final_clusters,
        all_clusters
    )

    logger.info(f"Pass 1 complete: {len(all_clusters)} clusters → {len(final_clusters)} clusters "
               f"({len(final_clusters) - len(all_clusters):+d}).  AMI {pass1_ami:.3f}")
    logger.info(f"Pass 1 timing: conflict={timing['conflict_resolution']:.1f}s, "
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
    operations: List[Dict]
) -> Tuple[Dict[str, List[str]], bool, Set[frozenset]]:
    """Execute all refinement operations, handling overlaps and tracking changes.

    Args:
        current_clusters: Current cluster state
        operations: List of refinement operations to apply

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
        ami = op['ami']

        # Check if all input clusters still exist (not consumed by earlier op)
        inputs_still_exist = all(cid in next_clusters for cid in input_cluster_ids)

        if not inputs_still_exist:
            logger.info(f"Skipping operation for seed {seed_id} - inputs already consumed")
            continue

        # Check convergence: AMI == 1.0 means perfect agreement (no changes)
        if ami == 1.0:
            # No changes - mark scope as converged
            new_converged_scopes.add(scope_signature)
            logger.debug(f"Scope converged (AMI=1.0): {len(scope_signature)} sequences")
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
    logger.info(f"Seed prioritization: per-sequence reclustering counts (deterministic)")

    pass2_start = time.time()
    timing = {
        'iterations': [],  # Per-iteration timing
        'proximity_graphs': [],  # Per-graph timing
        'refinements': []  # Per-iteration refinement timing
    }

    current_clusters = all_clusters.copy()
    global_iteration = 0

    # Track converged scopes (sets of sequences that refined to themselves)
    converged_scopes = set()  # Set[frozenset[sequence_id]]

    # Track per-sequence reclustering counts for prioritization
    sequence_recluster_count = defaultdict(int)  # sequence_id -> count

    while global_iteration < max_iterations:
        iteration_start = time.time()
        global_iteration += 1
        logger.info(f"Pass 2 iteration {global_iteration}: {len(current_clusters)} clusters")

        # Build proximity graph for current cluster state
        graph_start = time.time()
        proximity_graph = ClusterGraph(
            current_clusters, sequences, headers,
            k_neighbors=config.k_neighbors,
            search_method=config.search_method,
            show_progress=True,
            close_threshold=close_threshold  # Include all neighbors within close_threshold
        )
        graph_time = time.time() - graph_start
        timing['proximity_graphs'].append(graph_time)

        # Track which clusters have been processed this iteration
        processed_this_iteration = set()

        # Collect all refinement operations for this iteration
        refinement_operations = []

        # Track statistics for iteration summary
        iteration_stats = {
            'best_gap': float('-inf'),
            'best_gap_info': None,
            'seeds_processed': 0,
            'seeds_skipped_dependency': 0,
            'seeds_skipped_convergence': 0,
            'seeds_skipped_other': 0
        }

        # Calculate cluster priorities based on minimum per-sequence reclustering count
        refinement_start = time.time()
        cluster_priorities = {}
        for cluster_id, cluster_headers in current_clusters.items():
            if cluster_headers:
                # Priority = minimum reclustering count across all sequences in cluster
                min_count = min(sequence_recluster_count[h] for h in cluster_headers)
                cluster_priorities[cluster_id] = min_count
            else:
                cluster_priorities[cluster_id] = 0

        # Sort seeds by priority (lowest count first), with cluster size as tiebreaker
        seed_list = sorted(
            current_clusters.keys(),
            key=lambda cid: (
                cluster_priorities[cid],
                -len(current_clusters[cid])  # Larger clusters first for ties
            )
        )

        # Log priority distribution for diagnostics
        priority_counts = defaultdict(int)
        for priority in cluster_priorities.values():
            priority_counts[priority] += 1
        priority_dist = ", ".join(f"{k}:{v}" for k, v in sorted(priority_counts.items())[:10])
        logger.debug(f"Priority distribution (count:clusters): {priority_dist}")

        total_clusters = len(seed_list)  # Total for progress counter
        seeds_processed = 0  # Track seeds processed (1 per seed, whether executed or skipped)

        for seed_id in seed_list:
            seeds_processed += 1  # Increment for every seed

            if seed_id in processed_this_iteration:
                iteration_stats['seeds_skipped_dependency'] += 1
                logger.info(f"Pass 2 Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                          f"Skipping seed {seed_id} - seed already processed")
                continue  # Already processed as part of another seed's scope

            # Build refinement scope (seed + neighbors + context) using COMPLETE neighborhood
            scope_cluster_ids, scope_sequences, scope_headers = build_refinement_scope(
                seed_clusters=[seed_id],
                all_clusters=current_clusters,
                proximity_graph=proximity_graph,
                sequences=sequences,
                headers=headers,
                max_lump=max_lump,
                close_threshold=close_threshold,
                config=config
            )

            # Check if ANY cluster in the scope has been processed this iteration
            # If so, skip because the proximity graph is stale for this neighborhood
            scope_has_processed_clusters = any(cid in processed_this_iteration for cid in scope_cluster_ids)
            if scope_has_processed_clusters:
                iteration_stats['seeds_skipped_dependency'] += 1
                logger.info(f"Pass 2 Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                          f"Skipping seed {seed_id} - neighborhood changed")
                continue

            # Check if this scope has already converged (use sequence set as signature)
            scope_signature = frozenset(scope_headers)
            if scope_signature in converged_scopes:
                iteration_stats['seeds_skipped_convergence'] += 1
                logger.info(f"Pass 2 Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                          f"Skipping seed {seed_id} - prior convergence detected ({len(scope_signature)} sequences)")
                # Don't mark as processed - we didn't actually process anything
                continue

            # Apply full gapHACk to scope and get metadata
            refined_clusters, metadata = apply_full_gaphack_to_scope_with_metadata(
                scope_sequences, scope_headers,
                min_split, max_lump, target_percentile,
                cluster_id_generator=cluster_id_generator,
                quiet=True
            )

            # Track statistics for this refinement
            num_input = len(scope_cluster_ids)
            num_output = len(refined_clusters)
            gap_size = metadata.get('gap_size', float('-inf'))

            # Compute AMI between input and output
            ami = compute_ami_for_refinement(set(scope_cluster_ids), refined_clusters, current_clusters)

            # Extract gap details from metadata
            gap_info_str = ""
            best_config = metadata.get('metadata', {}).get('best_config', {})
            gap_metrics = best_config.get('gap_metrics')
            if gap_metrics:
                target_key = f'p{target_percentile}'
                if target_key in gap_metrics:
                    intra = gap_metrics[target_key].get('intra_upper', 0.0)
                    inter = gap_metrics[target_key].get('inter_lower', 0.0)
                    gap_info_str = f"Gap {gap_size:.4f} (intra≤{intra:.4f}, inter≥{inter:.4f})"
            elif gap_size > float('-inf'):
                gap_info_str = f"Gap {gap_size:.4f}"

            # Track best gap for this iteration
            if gap_size > iteration_stats['best_gap']:
                iteration_stats['best_gap'] = gap_size
                if gap_metrics:
                    target_key = f'p{target_percentile}'
                    if target_key in gap_metrics:
                        iteration_stats['best_gap_info'] = {
                            'intra_upper': gap_metrics[target_key].get('intra_upper', 0.0),
                            'inter_lower': gap_metrics[target_key].get('inter_lower', 0.0)
                        }

            # Log individual refinement
            logger.info(f"Pass 2 Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                       f"{num_input} -> {num_output} clusters.  {gap_info_str}.  AMI {ami:.3f}")

            # Store operation for batch execution
            refinement_operations.append({
                'seed_id': seed_id,
                'input_cluster_ids': set(scope_cluster_ids),
                'output_clusters': refined_clusters,
                'scope_signature': scope_signature,
                'ami': ami
            })

            # Update reclustering counts for all sequences in this scope
            for header in scope_headers:
                sequence_recluster_count[header] += 1

            # Mark all input clusters as processed
            processed_this_iteration.update(scope_cluster_ids)

            # Increment processed counter
            iteration_stats['seeds_processed'] += 1

        refinement_time = time.time() - refinement_start
        timing['refinements'].append(refinement_time)

        # Log reclustering count statistics for diagnostics
        if sequence_recluster_count:
            counts = list(sequence_recluster_count.values())
            min_count = min(counts)
            max_count = max(counts)
            mean_count = sum(counts) / len(counts)
            logger.info(f"Pass 2 Iter {global_iteration} Reclustering stats: "
                       f"min={min_count}, max={max_count}, mean={mean_count:.1f}")

        # Execute all refinement operations and track changes
        next_clusters, changes_made, new_converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=refinement_operations
        )

        # Update converged scopes
        converged_scopes.update(new_converged)

        iteration_time = time.time() - iteration_start
        timing['iterations'].append(iteration_time)

        # Compute AMI between clusters before and after this iteration
        iteration_ami = compute_ami_for_refinement(
            set(current_clusters.keys()),
            next_clusters,
            current_clusters
        )

        # Validate tracking counts
        total_seeds = total_clusters
        accounted_seeds = (iteration_stats['seeds_processed'] +
                          iteration_stats['seeds_skipped_dependency'] +
                          iteration_stats['seeds_skipped_convergence'])
        iteration_stats['seeds_skipped_other'] = total_seeds - accounted_seeds

        # Log comprehensive iteration summary
        logger.info(f"Pass 2 Iter {global_iteration} Summary:")
        logger.info(f"  Clusters: {len(current_clusters)} -> {len(next_clusters)} "
                   f"({len(next_clusters) - len(current_clusters):+d})")
        logger.info(f"  AMI: {iteration_ami:.3f}")
        logger.info(f"  Seeds: {total_seeds} total")
        logger.info(f"    - Processed: {iteration_stats['seeds_processed']}")
        logger.info(f"    - Skipped (dependency): {iteration_stats['seeds_skipped_dependency']}")
        logger.info(f"    - Skipped (convergence): {iteration_stats['seeds_skipped_convergence']}")
        if iteration_stats['seeds_skipped_other'] != 0:
            logger.warning(f"    - Skipped (other/ERROR): {iteration_stats['seeds_skipped_other']}")

        # Check convergence: AMI = 1.0 means perfect agreement (identical clustering)
        if iteration_ami == 1.0:
            logger.info(f"Convergence achieved at iteration {global_iteration} (AMI = 1.0)")
            tracking_info.summary_stats = {'convergence_reason': 'ami_convergence'}
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
    logger.info(f"Found {len(conflict_components)} connected conflict components")

    updated_clusters = all_clusters.copy()
    total_components = len(conflict_components)

    # Calculate total clusters in conflicts for progress tracking
    total_conflict_clusters = sum(len(comp) for comp in conflict_components)
    clusters_processed = 0  # Track cumulative input clusters

    # Process each conflict component
    for component_idx, component_clusters in enumerate(conflict_components):
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
            logger.warning(f"Pass 1 Conflict Resolution ({component_idx+1} of {total_components}): "
                         f"Skipping empty scope")
            component_info['processed'] = False
            component_info['clusters_after'] = []
            component_info['clusters_after_count'] = 0
            tracking_info.components_processed.append(component_info)
            continue

        if len(scope_headers_set) <= config.max_full_gaphack_size:
            # Map headers to actual sequences
            header_to_idx = {h: i for i, h in enumerate(headers)}
            scope_sequence_list = [sequences[header_to_idx[h]] for h in scope_headers]

            # Apply full gapHACk and get metadata
            full_result, metadata = apply_full_gaphack_to_scope_with_metadata(
                scope_sequence_list, scope_headers, min_split, max_lump,
                target_percentile, cluster_id_generator, quiet=True
            )

            # Extract gap info and compute AMI for logging
            gap_size = metadata.get('gap_size', float('-inf'))

            # Compute AMI between input and output
            ami = compute_ami_for_refinement(set(component_clusters), full_result, all_clusters)

            # Format gap info
            gap_info_str = ""
            best_config = metadata.get('metadata', {}).get('best_config', {})
            gap_metrics = best_config.get('gap_metrics')
            if gap_metrics:
                target_key = f'p{target_percentile}'
                if target_key in gap_metrics:
                    intra = gap_metrics[target_key].get('intra_upper', 0.0)
                    inter = gap_metrics[target_key].get('inter_lower', 0.0)
                    gap_info_str = f"Gap {gap_size:.4f} (intra≤{intra:.4f}, inter≥{inter:.4f})"
            elif gap_size > float('-inf'):
                gap_info_str = f"Gap {gap_size:.4f}"

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

            clusters_processed += len(component_clusters)
            logger.info(f"Pass 1 Conflict ({clusters_processed} of {total_conflict_clusters}): "
                       f"{len(component_clusters)} -> {len(full_result)} clusters.  {gap_info_str}.  AMI {ami:.3f}")

        else:
            # Fallback: skip oversized components with warning
            logger.warning(f"Pass 1 Conflict Resolution ({component_idx+1} of {total_components}): "
                         f"Skipping oversized component with {len(scope_headers_set)} sequences "
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