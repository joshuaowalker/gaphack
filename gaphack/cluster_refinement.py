"""
Cluster refinement algorithms for iterative neighborhood-based refinement.

This module implements scope-limited refinement using full gapHACk to optimize
cluster boundaries through iterative refinement of local neighborhoods.
"""

import logging
import copy
import warnings
import json
import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from .cluster_graph import ClusterGraph
from .distance_providers import DistanceProvider
from .core import GapOptimizedClustering
from .refinement_types import ProcessingStageInfo, ClusterIDGenerator
from .utils import calculate_distance_matrix

logger = logging.getLogger("gaphack.refine")


# Convergence metrics configuration
# Number of nearest neighbor clusters to use for inter-cluster distance calculation
# Higher K provides more robust estimates but increases computation
CONVERGENCE_METRICS_K_NEIGHBORS = 3

# Cache for per-cluster gap computations used in convergence tracking
# Key: (frozenset(cluster_headers), tuple(sorted(frozenset(neighbor_headers))), target_percentile)
# Value: gap value (float)
_convergence_metrics_cache: Dict[Tuple[frozenset, Tuple[frozenset, ...], int], float] = {}


def compute_convergence_metrics(
    clusters: Dict[str, List[str]],
    proximity_graph: 'ClusterGraph',
    sequences: List[str],
    headers: List[str],
    target_percentile: int = 95,
    show_progress: bool = True
) -> Dict[str, float]:
    """Compute convergence metrics across all clusters for refinement tracking.

    Calculates per-cluster barcode gaps using K nearest neighbor clusters for
    inter-cluster distances, then aggregates into summary metrics. This provides
    a quality metric for tracking refinement convergence.

    Creates MSA-based distance providers for each cluster's minimal scope
    (cluster + K neighbors) to avoid cost-prohibitive global MSA.

    Note: This is different from --gap-method global vs local. This function
    computes per-cluster gaps (similar to local method) and aggregates them
    for convergence tracking across the entire clustering.

    Args:
        clusters: Dict mapping cluster_id -> list of sequence headers
        proximity_graph: K-NN graph for finding nearest neighbor clusters
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        target_percentile: Percentile for gap calculation (default: 95)
        show_progress: Show progress bar during computation (default: True)

    Returns:
        Dict with convergence metrics:
        - 'mean_gap': Mean gap across all clusters (unweighted)
        - 'weighted_gap': Mean gap weighted by cluster size
        - 'gap_coverage': Fraction of clusters with positive gap
        - 'gap_coverage_sequences': Fraction of sequences in positive-gap clusters
    """
    from .distance_providers import MSACachedDistanceProvider
    from tqdm import tqdm

    # Map headers to indices
    header_to_idx = {h: i for i, h in enumerate(headers)}

    # Track per-cluster gaps and sizes
    per_cluster_gaps = []
    per_cluster_sizes = []
    positive_gap_clusters = 0
    positive_gap_sequences = 0
    total_sequences = 0

    # Track clusters with issues for debugging
    nan_gap_count = 0
    skipped_no_neighbors = 0
    cache_hits = 0
    cache_misses = 0

    cluster_iter = tqdm(clusters.items(), desc="Computing convergence metrics", unit="cluster", disable=not show_progress)

    for cluster_id, cluster_headers in cluster_iter:
        cluster_size = len(cluster_headers)
        total_sequences += cluster_size

        # Get K nearest neighbor clusters
        k_nearest = proximity_graph.get_k_nearest_neighbors(cluster_id, CONVERGENCE_METRICS_K_NEIGHBORS)

        # Build minimal scope: cluster + K neighbors
        scope_headers = list(cluster_headers)  # Start with current cluster
        neighbor_cluster_ids = []

        for neighbor_id, _ in k_nearest:
            if neighbor_id in clusters:
                neighbor_cluster_ids.append(neighbor_id)
                scope_headers.extend(clusters[neighbor_id])

        # Skip if no neighbors found
        if not neighbor_cluster_ids:
            skipped_no_neighbors += 1
            logger.debug(f"Cluster {cluster_id} has no neighbors for gap calculation")
            continue

        # Check cache before expensive MSA computation
        cache_key = (
            frozenset(cluster_headers),
            tuple(sorted(frozenset(clusters[nid]) for nid in neighbor_cluster_ids)),
            target_percentile
        )

        if cache_key in _convergence_metrics_cache:
            # Cache hit - reuse computed gap
            gap = _convergence_metrics_cache[cache_key]
            cache_hits += 1

            # Track statistics
            per_cluster_gaps.append(gap)
            per_cluster_sizes.append(cluster_size)

            if gap > 0:
                positive_gap_clusters += 1
                positive_gap_sequences += cluster_size

            continue  # Skip expensive computation

        # Cache miss - need to compute gap
        cache_misses += 1

        # Create scoped MSA-based distance provider for this cluster + neighbors
        scope_sequences = [sequences[header_to_idx[h]] for h in scope_headers]
        scope_distance_provider = MSACachedDistanceProvider(scope_sequences, scope_headers)

        # Build local header-to-scope-index mapping
        scope_header_to_idx = {h: i for i, h in enumerate(scope_headers)}

        # 1. Compute intra-cluster distances
        intra_distances = []
        if cluster_size > 1:
            # All pairwise distances within cluster
            for i in range(len(cluster_headers)):
                for j in range(i + 1, len(cluster_headers)):
                    scope_i = scope_header_to_idx[cluster_headers[i]]
                    scope_j = scope_header_to_idx[cluster_headers[j]]
                    dist = scope_distance_provider.get_distance(scope_i, scope_j)
                    # Filter out NaN distances
                    if not np.isnan(dist):
                        intra_distances.append(dist)

        # Singleton case: intra = 0 (perfect cohesion)
        if not intra_distances:
            intra_upper = 0.0
        else:
            intra_upper = np.percentile(intra_distances, target_percentile)
            # Check for NaN in percentile result
            if np.isnan(intra_upper):
                logger.warning(f"Cluster {cluster_id}: NaN intra_upper from {len(intra_distances)} distances")
                intra_upper = 0.0

        # 2. Compute inter-cluster distances to K nearest neighbors
        inter_distances = []

        for neighbor_id in neighbor_cluster_ids:
            neighbor_headers = clusters[neighbor_id]

            # All pairwise distances between clusters
            for cluster_h in cluster_headers:
                for neighbor_h in neighbor_headers:
                    scope_i = scope_header_to_idx[cluster_h]
                    scope_j = scope_header_to_idx[neighbor_h]
                    dist = scope_distance_provider.get_distance(scope_i, scope_j)
                    # Filter out NaN distances
                    if not np.isnan(dist):
                        inter_distances.append(dist)

        # Compute inter-cluster lower bound
        if not inter_distances:
            logger.debug(f"Cluster {cluster_id} has no valid inter-cluster distances")
            continue

        inter_lower = np.percentile(inter_distances, 100 - target_percentile)

        # Check for NaN in percentile result
        if np.isnan(inter_lower):
            logger.warning(f"Cluster {cluster_id}: NaN inter_lower from {len(inter_distances)} distances")
            continue

        # 3. Compute gap for this cluster
        gap = inter_lower - intra_upper

        # Final NaN check
        if np.isnan(gap):
            nan_gap_count += 1
            logger.warning(f"Cluster {cluster_id}: NaN gap (intra={intra_upper:.4f}, inter={inter_lower:.4f})")
            continue

        # Store in cache for future iterations
        _convergence_metrics_cache[cache_key] = gap

        per_cluster_gaps.append(gap)
        per_cluster_sizes.append(cluster_size)

        if gap > 0:
            positive_gap_clusters += 1
            positive_gap_sequences += cluster_size

    # Log debug statistics if there were issues
    if nan_gap_count > 0 or skipped_no_neighbors > 0:
        logger.info(f"Convergence metrics computation: skipped {skipped_no_neighbors} clusters (no neighbors), "
                   f"{nan_gap_count} clusters (NaN gaps)")

    # Aggregate into global metrics
    if not per_cluster_gaps:
        # No gaps computed - return zeros
        logger.warning("No valid gaps computed - all clusters skipped")
        return {
            'mean_gap': 0.0,
            'weighted_gap': 0.0,
            'gap_coverage': 0.0,
            'gap_coverage_sequences': 0.0
        }

    mean_gap = np.mean(per_cluster_gaps)
    weighted_gap = np.average(per_cluster_gaps, weights=per_cluster_sizes)
    gap_coverage = positive_gap_clusters / len(per_cluster_gaps)
    gap_coverage_sequences = positive_gap_sequences / total_sequences if total_sequences > 0 else 0.0

    # Final sanity check for NaN in aggregated metrics
    if np.isnan(mean_gap) or np.isnan(weighted_gap):
        logger.error(f"NaN in final metrics! mean_gap={mean_gap}, weighted_gap={weighted_gap}, "
                    f"gaps computed: {len(per_cluster_gaps)}")

    return {
        'mean_gap': float(mean_gap),
        'weighted_gap': float(weighted_gap),
        'gap_coverage': float(gap_coverage),
        'gap_coverage_sequences': float(gap_coverage_sequences)
    }


def compute_scope_convergence_metrics(
    output_clusters: Dict[str, List[str]],
    scope_cluster_ids: List[str],
    distance_matrix: np.ndarray,
    scope_headers: List[str],
    target_percentile: int = 95,
    max_neighbors: int = 3,
    max_lump: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """Compute convergence metrics for output clusters using precomputed distance matrix.

    This function reuses the distance matrix already computed during refinement to
    calculate per-cluster gaps without requiring additional MSA computations.

    Finds K nearest neighbors within the scope using the distance matrix directly,
    since output clusters have new IDs that don't exist in the proximity graph yet.

    Args:
        output_clusters: Clusters produced by refinement (cluster_id -> headers)
        scope_cluster_ids: All cluster IDs in the refinement scope (not used, kept for compatibility)
        distance_matrix: Precomputed distance matrix for scope sequences
        scope_headers: Headers for sequences in scope (indices match distance_matrix)
        target_percentile: Percentile for gap calculation (default: 95)
        max_neighbors: Maximum number of neighbor clusters to use (default: 3)
        max_lump: Maximum lump threshold, used as inter-cluster estimate when no neighbors exist (default: None)

    Returns:
        Dict mapping cluster_id -> {gap, intra_upper, inter_lower, cluster_size, num_neighbors}
    """
    header_to_scope_idx = {h: i for i, h in enumerate(scope_headers)}
    per_cluster_metrics = {}

    # Build mapping of output clusters in scope
    output_cluster_headers_in_scope = {}
    for cluster_id, cluster_headers in output_clusters.items():
        headers_in_scope = [h for h in cluster_headers if h in header_to_scope_idx]
        if headers_in_scope:
            output_cluster_headers_in_scope[cluster_id] = headers_in_scope

    for cluster_id, cluster_headers in output_cluster_headers_in_scope.items():
        cluster_size = len(cluster_headers)

        # Get indices in distance matrix for this cluster
        cluster_indices = [header_to_scope_idx[h] for h in cluster_headers]

        # 1. Compute intra-cluster distances
        intra_distances = []
        if len(cluster_indices) > 1:
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    idx1, idx2 = cluster_indices[i], cluster_indices[j]
                    dist = distance_matrix[idx1, idx2]
                    if not np.isnan(dist):
                        intra_distances.append(dist)

        # Singleton case: intra = 0 (perfect cohesion)
        if not intra_distances:
            intra_upper = 0.0
        else:
            intra_upper = np.percentile(intra_distances, target_percentile)
            if np.isnan(intra_upper):
                intra_upper = 0.0

        # 2. Find neighbor clusters within scope
        # Since output clusters have new IDs that don't exist in proximity graph yet,
        # we use distance matrix to find K nearest neighbors within the scope

        # Compute distances from this cluster to all other clusters in scope
        cluster_to_other_distances = []
        for other_id in output_cluster_headers_in_scope.keys():
            if other_id == cluster_id:
                continue  # Skip self

            other_headers = output_cluster_headers_in_scope[other_id]
            other_indices = [header_to_scope_idx[h] for h in other_headers]

            # Compute minimum distance between this cluster and other cluster
            min_dist = float('inf')
            for cluster_idx in cluster_indices:
                for other_idx in other_indices:
                    dist = distance_matrix[cluster_idx, other_idx]
                    if not np.isnan(dist) and dist < min_dist:
                        min_dist = dist

            if min_dist != float('inf'):
                cluster_to_other_distances.append((other_id, min_dist))

        # Sort by distance and select K nearest
        cluster_to_other_distances.sort(key=lambda x: x[1])
        neighbor_cluster_ids = [
            other_id for other_id, _ in cluster_to_other_distances[:max_neighbors]
        ]

        if not neighbor_cluster_ids:
            # No neighbors available - use max_lump as conservative estimate if provided
            if max_lump is not None:
                inter_lower = max_lump
                logger.debug(f"Cluster {cluster_id}: no neighbors in scope, using max_lump={max_lump:.4f} as inter-cluster estimate")
            else:
                # No max_lump provided - skip this cluster
                logger.debug(f"Cluster {cluster_id}: no neighbors in scope and no max_lump provided, skipping")
                continue
        else:
            # 3. Compute inter-cluster distances to selected neighbors
            inter_distances = []
            for neighbor_id in neighbor_cluster_ids:
                neighbor_headers = output_cluster_headers_in_scope[neighbor_id]
                neighbor_indices = [header_to_scope_idx[h] for h in neighbor_headers]

                for cluster_idx in cluster_indices:
                    for neighbor_idx in neighbor_indices:
                        dist = distance_matrix[cluster_idx, neighbor_idx]
                        if not np.isnan(dist):
                            inter_distances.append(dist)

            if not inter_distances:
                logger.debug(f"Cluster {cluster_id}: no valid inter-cluster distances")
                continue

            inter_lower = np.percentile(inter_distances, 100 - target_percentile)
            if np.isnan(inter_lower):
                logger.debug(f"Cluster {cluster_id}: NaN inter_lower")
                continue

        # 4. Compute gap
        gap = inter_lower - intra_upper

        per_cluster_metrics[cluster_id] = {
            'gap': float(gap),
            'intra_upper': float(intra_upper),
            'inter_lower': float(inter_lower),
            'cluster_size': cluster_size,
            'num_neighbors': len(neighbor_cluster_ids)
        }

    return per_cluster_metrics


def aggregate_convergence_metrics(
    per_cluster_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Aggregate per-cluster metrics into summary statistics.

    Args:
        per_cluster_metrics: Dict mapping cluster_id -> metric dict with keys:
            - gap: barcode gap value
            - cluster_size: number of sequences in cluster
            - (other fields ignored for aggregation)

    Returns:
        Summary metrics:
        - mean_gap: Unweighted mean gap across clusters
        - weighted_gap: Mean gap weighted by cluster size
        - gap_coverage: Fraction of clusters with positive gap
        - gap_coverage_sequences: Fraction of sequences in positive-gap clusters
        - clusters_with_metrics: Number of clusters with computed metrics
    """
    if not per_cluster_metrics:
        logger.warning("No cluster metrics to aggregate")
        return {
            'mean_gap': 0.0,
            'weighted_gap': 0.0,
            'gap_coverage': 0.0,
            'gap_coverage_sequences': 0.0,
            'clusters_with_metrics': 0
        }

    gaps = []
    sizes = []
    positive_gap_clusters = 0
    positive_gap_sequences = 0
    total_sequences = 0

    for cluster_id, metrics in per_cluster_metrics.items():
        gap = metrics['gap']
        size = metrics['cluster_size']

        gaps.append(gap)
        sizes.append(size)
        total_sequences += size

        if gap > 0:
            positive_gap_clusters += 1
            positive_gap_sequences += size

    mean_gap = np.mean(gaps)
    weighted_gap = np.average(gaps, weights=sizes)
    gap_coverage = positive_gap_clusters / len(gaps)
    gap_coverage_sequences = positive_gap_sequences / total_sequences if total_sequences > 0 else 0.0

    return {
        'mean_gap': float(mean_gap),
        'weighted_gap': float(weighted_gap),
        'gap_coverage': float(gap_coverage),
        'gap_coverage_sequences': float(gap_coverage_sequences),
        'clusters_with_metrics': len(per_cluster_metrics)
    }


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


def checkpoint_iteration(
    iteration: int,
    current_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    header_mapping: Dict[str, str],
    sequence_recluster_count: Dict[str, int],
    converged_scopes: Set[frozenset],
    iteration_stats: Dict,
    timing: Dict,
    base_output_dir: Path
) -> Path:
    """Checkpoint current iteration state to timestamped directory.

    Creates a complete snapshot of the current iteration that can be:
    - Loaded for post-mortem analysis
    - Used as input for resuming refinement

    Args:
        iteration: Current iteration number
        current_clusters: Current cluster assignments
        sequences: Full sequence list
        headers: Full header list
        header_mapping: Dict mapping sequence ID to full header (ID + description)
        sequence_recluster_count: Per-sequence reclustering counts
        converged_scopes: Set of converged scope signatures
        iteration_stats: Summary statistics for this iteration
        timing: Timing information for this iteration
        base_output_dir: Base output directory (parent of timestamped directories)

    Returns:
        Path to checkpoint directory
    """
    # Import here to avoid circular dependency
    from .refine_cli import write_output_clusters

    # Create timestamped directory with iteration suffix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = base_output_dir / f"{timestamp}_iter{iteration:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Write cluster FASTA files (reuse existing function)
    write_output_clusters(
        clusters=current_clusters,
        sequences=sequences,
        headers=headers,
        unassigned_headers=[],  # Don't checkpoint unassigned
        output_dir=checkpoint_dir,
        header_mapping=header_mapping,
        renumber=False  # Keep IDs stable across iterations
    )

    # Write iteration state
    state = {
        "iteration": iteration,
        "timestamp": timestamp,
        "summary": iteration_stats,
        "reclustering_counts": dict(sequence_recluster_count),
        # Note: converged_scopes omitted - they're a performance optimization
        # that gets rebuilt on resume. Can't reconstruct frozensets from hashes.
        "timing": timing
    }

    state_file = checkpoint_dir / "state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    logger.info(f"Checkpointed iteration {iteration} to {checkpoint_dir.name}")
    return checkpoint_dir


class RefinementConfig:
    """Configuration for iterative cluster refinement.

    Controls parameters for neighborhood-based iterative refinement that
    continues until convergence or iteration limit is reached.
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
            close_threshold: Distance threshold for finding nearby clusters (default: max_lump)
            max_iterations: Maximum refinement iterations (default: 10)
            k_neighbors: K-NN graph parameter (default: 20)
            search_method: "blast" or "vsearch" for proximity graph
            random_seed: Seed for randomizing seed order (default: None for random seed)
        """
        self.max_full_gaphack_size = max_full_gaphack_size
        self.close_threshold = close_threshold
        self.max_iterations = max_iterations
        self.k_neighbors = k_neighbors
        self.search_method = search_method
        self.random_seed = random_seed


# ============================================================================
# Helper Functions for Iterative Refinement
# ============================================================================


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


def apply_full_gaphack_to_scope_with_metadata(scope_sequences: List[str], scope_headers: List[str],
                                              min_split: float = 0.005, max_lump: float = 0.02,
                                              target_percentile: int = 95, cluster_id_generator=None,
                                              quiet: bool = True, gap_method: str = 'global',
                                              alpha: float = 0.0,
                                              return_distance_data: bool = False) -> Tuple[Dict[str, List[str]], Dict]:
    """Apply full gapHACk clustering to a scope of sequences, returning clusters and metadata.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        cluster_id_generator: Optional cluster ID generator
        quiet: If True, suppress verbose logging (default: True)
        gap_method: Gap calculation method ('global' or 'local')
        alpha: Parsimony parameter for local gap (default: 0.0)
        return_distance_data: If True, include distance matrix in metadata for reuse (default: False)

    Returns:
        Tuple of (cluster_dict, metadata_dict) where metadata includes gap_size and
        optionally distance_matrix and scope_headers if return_distance_data=True
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
        logger=quiet_logger,
        gap_method=gap_method,
        alpha=alpha
    )

    # Get clustering result with metadata
    final_clusters, singletons, metadata = clusterer.cluster(distance_matrix)

    # Convert result to cluster dictionary format
    clusters = {}

    # Import here to avoid circular imports
    from .refinement_types import ClusterIDGenerator

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

    # Build metadata dict
    result_metadata = {'gap_size': gap_size, 'metadata': metadata}

    # Include distance data if requested
    if return_distance_data:
        result_metadata['distance_matrix'] = distance_matrix
        result_metadata['scope_headers'] = scope_headers

    return clusters, result_metadata


def apply_full_gaphack_to_scope(scope_sequences: List[str], scope_headers: List[str], min_split: float = 0.005,
                                max_lump: float = 0.02, target_percentile: int = 95, cluster_id_generator=None,
                                quiet: bool = True, gap_method: str = 'global',
                                alpha: float = 0.0) -> Dict[str, List[str]]:
    """Apply full gapHACk clustering to a scope of sequences.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        cluster_id_generator: Optional cluster ID generator
        quiet: If True, suppress verbose logging (default: True)
        gap_method: Gap calculation method ('global' or 'local')
        alpha: Parsimony parameter for local gap (default: 0.0)

    Returns:
        Dict mapping cluster_id -> list of sequence headers
    """
    # Use the metadata version and just return the clusters
    clusters, _ = apply_full_gaphack_to_scope_with_metadata(scope_sequences, scope_headers, min_split, max_lump,
                                                            target_percentile, cluster_id_generator, quiet, gap_method,
                                                            alpha)
    return clusters


# ============================================================================
# Iterative Refinement with Neighborhood-Based Seeding
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
        scope_signature = op['scope_signature']  # POST-refinement signature
        ami = op['ami']

        # Check if all input clusters still exist (not consumed by earlier op)
        inputs_still_exist = all(cid in next_clusters for cid in input_cluster_ids)

        if not inputs_still_exist:
            logger.info(f"Skipping operation for seed {seed_id} - inputs already consumed")
            continue

        # Always cache the POST-refinement signature
        # This optimization is valid because gapHACk is deterministic on sequences
        # Cache semantics: "Refining these sequences produces this clustering"
        # Confirmed empirically through extensive testing
        new_converged_scopes.add(scope_signature)

        # Check if any changes needed: AMI == 1.0 means perfect agreement (no changes)
        if ami == 1.0:
            # No changes - clustering is already in stable state
            seq_set, cluster_pattern = scope_signature
            logger.debug(f"Scope stable (AMI=1.0): {len(seq_set)} sequences, {len(cluster_pattern)} clusters")
            continue

        # Apply refinement: remove inputs, add outputs
        for input_id in input_cluster_ids:
            if input_id in next_clusters:
                del next_clusters[input_id]

        for output_id, output_headers in output_clusters.items():
            next_clusters[output_id] = output_headers

        changes_made = True

    return next_clusters, changes_made, new_converged_scopes


def iterative_refinement(
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
    show_progress: bool = False,
    checkpoint_frequency: int = 0,
    checkpoint_output_dir: Optional[Path] = None,
    header_mapping: Optional[Dict[str, str]] = None,
    resume_state: Optional[Dict] = None,
    gap_method: str = 'global',
    alpha: float = 0.0
) -> Tuple[Dict[str, List[str]], 'ProcessingStageInfo']:
    """Iteratively refine clusters using neighborhood-based seeding.

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
        checkpoint_frequency: Checkpoint every N iterations (0=disabled, default: 0)
        checkpoint_output_dir: Base output directory for checkpoints
        header_mapping: Mapping of sequence IDs to full headers (for checkpointing)
        resume_state: Optional state to resume from (loaded from state.json)
        gap_method: Gap calculation method ('global' or 'local')
        alpha: Parsimony parameter for local gap method (default: 0.0)
              Score = local_gap / (num_clusters^alpha)

    Returns:
        Tuple of (refined_clusters, tracking_info)
    """
    import time
    from tqdm import tqdm
    from .refinement_types import ProcessingStageInfo, ClusterIDGenerator

    # Initialize cluster ID generator if not provided
    if cluster_id_generator is None:
        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

    tracking_info = ProcessingStageInfo(
        stage_name="Iterative Refinement",
        clusters_before=all_clusters.copy()
    )

    logger.info(f"=== Iterative Refinement ===")
    logger.info(f"Starting with {len(all_clusters)} clusters")
    logger.info(f"Close threshold: {close_threshold:.4f}, Max iterations: {max_iterations}")
    logger.info(f"Seed prioritization: per-sequence reclustering counts (deterministic)")

    refinement_start_time = time.time()
    timing = {
        'iterations': [],  # Per-iteration timing
        'proximity_graphs': [],  # Per-graph timing
        'refinements': []  # Per-iteration refinement timing
    }

    current_clusters = all_clusters.copy()

    # Initialize or resume iteration state
    if resume_state:
        global_iteration = resume_state.get('iteration', 0)
        logger.info(f"Resuming from iteration {global_iteration}")
        # Restore reclustering counts
        sequence_recluster_count = defaultdict(int, resume_state.get('reclustering_counts', {}))
        # Restore converged scopes (convert hashes back to set, though we don't have full signatures)
        # For now, start fresh with converged scopes on resume
        converged_scopes = set()
        logger.info(f"Restored {len(sequence_recluster_count)} reclustering counts")
    else:
        global_iteration = 0
        sequence_recluster_count = defaultdict(int)  # sequence_id -> count
        converged_scopes = set()  # Set[frozenset[sequence_id]]

    # Initialize convergence metrics cache (persists across iterations)
    # Maps cluster_id -> metrics dict for reuse in skipped seeds
    convergence_metrics_cache = {}

    while global_iteration < max_iterations:
        iteration_start = time.time()
        global_iteration += 1
        logger.info(f"Refinement iteration {global_iteration}: {len(current_clusters)} clusters")

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

        # Initialize incremental convergence metrics accumulator for this iteration
        convergence_metrics_accumulator = {}

        # Track which clusters have been processed this iteration
        processed_this_iteration = set()

        # Collect all refinement operations for this iteration
        refinement_operations = []

        # Track statistics for iteration summary
        iteration_stats = {
            'best_gap': float('-inf'),
            'best_gap_info': None,
            'seeds_processed': 0,
            'seeds_skipped_already_processed': 0,  # Seed was in a processed scope
            'seeds_skipped_neighborhood_changed': 0,  # Neighborhood modified this iteration
            'seeds_skipped_convergence': 0,
            'seeds_skipped_other': 0
        }

        # Track unique clusters and sequences in each category
        # These are sets to avoid double-counting
        unique_clusters_processed = set()
        unique_sequences_processed = set()
        unique_sequences_converged = set()
        unique_sequences_neighborhood_changed = set()

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
                iteration_stats['seeds_skipped_already_processed'] += 1
                # Note: sequences from this seed are already in unique_sequences_processed
                # from when the seed was processed as part of another scope
                logger.info(f"Refinement Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
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
                iteration_stats['seeds_skipped_neighborhood_changed'] += 1
                # Track sequences in this scope (will be deduplicated against processed later)
                unique_sequences_neighborhood_changed.update(scope_headers)
                # Reuse cached convergence metrics for the seed only (neighbors will be seeds later)
                if seed_id in convergence_metrics_cache:
                    convergence_metrics_accumulator[seed_id] = convergence_metrics_cache[seed_id]
                logger.info(f"Refinement Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                          f"Skipping seed {seed_id} - neighborhood changed")
                continue

            # Check if this scope has already converged
            # Pre-refinement signature is used for lookup to determine if current state
            # matches a previously observed stable output (post-refinement clustering)
            # Signature includes both sequence set AND current clustering pattern to handle
            # cases where overlapping scopes change the clustering between iterations
            pre_refinement_clustering_pattern = set()
            for cluster_id in scope_cluster_ids:
                if cluster_id in current_clusters:
                    # Add frozenset of headers for this cluster to capture the grouping
                    cluster_headers_in_scope = current_clusters[cluster_id]
                    if cluster_headers_in_scope:  # Only add non-empty clusters
                        pre_refinement_clustering_pattern.add(frozenset(cluster_headers_in_scope))

            # Combined signature: (sequences, clustering_pattern)
            pre_refinement_signature = (
                frozenset(scope_headers),
                frozenset(pre_refinement_clustering_pattern)
            )

            if pre_refinement_signature in converged_scopes:
                iteration_stats['seeds_skipped_convergence'] += 1
                # Track sequences in this converged scope
                unique_sequences_converged.update(scope_headers)
                # Reuse cached convergence metrics for the seed only (neighbors will be seeds later)
                if seed_id in convergence_metrics_cache:
                    convergence_metrics_accumulator[seed_id] = convergence_metrics_cache[seed_id]
                logger.info(f"Refinement Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                          f"Skipping seed {seed_id} - scope converged ({len(scope_headers)} sequences, "
                          f"{len(pre_refinement_clustering_pattern)} clusters)")
                # Don't mark as processed - we didn't actually process anything
                continue

            # Apply full gapHACk to scope and get metadata (including distance matrix for metrics)
            refined_clusters, metadata = apply_full_gaphack_to_scope_with_metadata(
                scope_sequences, scope_headers,
                min_split, max_lump, target_percentile,
                cluster_id_generator=cluster_id_generator,
                quiet=True,
                gap_method=gap_method,
                alpha=alpha,
                return_distance_data=True  # Request distance data for incremental metrics
            )

            # Compute convergence metrics for output clusters using scope distance matrix
            if 'distance_matrix' in metadata and 'scope_headers' in metadata:
                scope_convergence_metrics = compute_scope_convergence_metrics(
                    output_clusters=refined_clusters,
                    scope_cluster_ids=scope_cluster_ids,
                    distance_matrix=metadata['distance_matrix'],
                    scope_headers=metadata['scope_headers'],
                    target_percentile=target_percentile,
                    max_neighbors=CONVERGENCE_METRICS_K_NEIGHBORS,  # Use K=3 limit
                    max_lump=max_lump  # For estimating inter-cluster distance when no neighbors
                )
                # Accumulate metrics (overwrites if cluster was seen before)
                convergence_metrics_accumulator.update(scope_convergence_metrics)
                # Update cache with fresh metrics for future iterations
                convergence_metrics_cache.update(scope_convergence_metrics)

            # Create POST-refinement signature for caching
            # This represents the stable output produced by gapHACk for these sequences
            # Cache semantics: "Refining these sequences produces this clustering"
            post_refinement_clustering_pattern = set()
            for cluster_headers_list in refined_clusters.values():
                if cluster_headers_list:  # Only add non-empty clusters
                    post_refinement_clustering_pattern.add(frozenset(cluster_headers_list))

            post_refinement_signature = (
                frozenset(scope_headers),
                frozenset(post_refinement_clustering_pattern)
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
            # gap_metrics can be a dict (global method) or float (total method)
            if gap_metrics and isinstance(gap_metrics, dict):
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
                if gap_metrics and isinstance(gap_metrics, dict):
                    target_key = f'p{target_percentile}'
                    if target_key in gap_metrics:
                        iteration_stats['best_gap_info'] = {
                            'intra_upper': gap_metrics[target_key].get('intra_upper', 0.0),
                            'inter_lower': gap_metrics[target_key].get('inter_lower', 0.0)
                        }

            # Log individual refinement
            logger.info(f"Refinement Iter {global_iteration} ({seeds_processed} of {total_clusters}): "
                       f"{num_input} -> {num_output} clusters.  {gap_info_str}.  AMI {ami:.3f}")

            # Store operation for batch execution
            # Note: scope_signature is the POST-refinement signature (deterministic output)
            refinement_operations.append({
                'seed_id': seed_id,
                'input_cluster_ids': set(scope_cluster_ids),
                'output_clusters': refined_clusters,
                'scope_signature': post_refinement_signature,
                'ami': ami
            })

            # Update reclustering counts for all sequences in this scope
            for header in scope_headers:
                sequence_recluster_count[header] += 1

            # Mark all input clusters as processed
            processed_this_iteration.update(scope_cluster_ids)

            # Track unique clusters and sequences in processed scopes
            unique_clusters_processed.update(scope_cluster_ids)
            unique_sequences_processed.update(scope_headers)

            # Increment seed counter
            iteration_stats['seeds_processed'] += 1

        refinement_time = time.time() - refinement_start
        timing['refinements'].append(refinement_time)

        # Aggregate convergence metrics from all processed scopes
        convergence_metrics = aggregate_convergence_metrics(convergence_metrics_accumulator)

        # Store convergence metrics in iteration stats for checkpointing
        iteration_stats['convergence_metrics'] = convergence_metrics

        # Log reclustering count statistics for diagnostics
        if sequence_recluster_count:
            counts = list(sequence_recluster_count.values())
            min_count = min(counts)
            max_count = max(counts)
            mean_count = sum(counts) / len(counts)
            logger.info(f"Refinement Iter {global_iteration} Reclustering stats: "
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
                          iteration_stats['seeds_skipped_already_processed'] +
                          iteration_stats['seeds_skipped_neighborhood_changed'] +
                          iteration_stats['seeds_skipped_convergence'])
        iteration_stats['seeds_skipped_other'] = total_seeds - accounted_seeds

        # Compute unique counts with precedence: processed > converged > neighborhood_changed
        # Remove sequences that appear in higher-priority categories
        unique_sequences_converged_only = unique_sequences_converged - unique_sequences_processed
        unique_sequences_neighborhood_changed_only = (unique_sequences_neighborhood_changed -
                                                      unique_sequences_processed -
                                                      unique_sequences_converged)

        # Store final counts in iteration_stats
        iteration_stats['clusters_processed'] = len(unique_clusters_processed)
        iteration_stats['clusters_skipped_already_processed'] = iteration_stats['seeds_skipped_already_processed']
        iteration_stats['clusters_skipped_neighborhood_changed'] = iteration_stats['seeds_skipped_neighborhood_changed']
        iteration_stats['clusters_skipped_convergence'] = iteration_stats['seeds_skipped_convergence']
        iteration_stats['sequences_processed'] = len(unique_sequences_processed)
        iteration_stats['sequences_skipped_convergence'] = len(unique_sequences_converged_only)
        iteration_stats['sequences_skipped_neighborhood_changed'] = len(unique_sequences_neighborhood_changed_only)
        # Note: sequences_skipped_already_processed is always 0 (those sequences are in processed)
        iteration_stats['sequences_skipped_already_processed'] = 0

        # Verify all sequences are accounted for
        total_sequences_in_clusters = sum(len(headers) for headers in current_clusters.values())
        accounted_sequences = (iteration_stats['sequences_processed'] +
                              iteration_stats['sequences_skipped_convergence'] +
                              iteration_stats['sequences_skipped_neighborhood_changed'] +
                              iteration_stats['sequences_skipped_already_processed'])
        if accounted_sequences != total_sequences_in_clusters:
            logger.warning(f"Sequence accounting mismatch: {accounted_sequences} accounted, "
                          f"{total_sequences_in_clusters} total in clusters")

        # Log comprehensive iteration summary
        logger.info(f"Refinement Iter {global_iteration} Summary:")
        logger.info(f"  Clusters: {len(current_clusters)} -> {len(next_clusters)} "
                   f"({len(next_clusters) - len(current_clusters):+d})")
        logger.info(f"  AMI: {iteration_ami:.3f}")
        logger.info(f"  Seeds: {total_seeds} total")
        logger.info(f"    - Processed: {iteration_stats['seeds_processed']} seeds, "
                   f"{iteration_stats['clusters_processed']} clusters, "
                   f"{iteration_stats['sequences_processed']} sequences")
        logger.info(f"    - Skipped (already processed): {iteration_stats['seeds_skipped_already_processed']} seeds, "
                   f"{iteration_stats['clusters_skipped_already_processed']} clusters, "
                   f"{iteration_stats['sequences_skipped_already_processed']} sequences")
        logger.info(f"    - Skipped (neighborhood changed): {iteration_stats['seeds_skipped_neighborhood_changed']} seeds, "
                   f"{iteration_stats['clusters_skipped_neighborhood_changed']} clusters, "
                   f"{iteration_stats['sequences_skipped_neighborhood_changed']} sequences")
        logger.info(f"    - Skipped (convergence): {iteration_stats['seeds_skipped_convergence']} seeds, "
                   f"{iteration_stats['clusters_skipped_convergence']} clusters, "
                   f"{iteration_stats['sequences_skipped_convergence']} sequences")
        if iteration_stats['seeds_skipped_other'] != 0:
            logger.warning(f"    - Skipped (other/ERROR): {iteration_stats['seeds_skipped_other']}")

        # Log convergence metrics (incremental, post-refinement)
        mean_gap = convergence_metrics['mean_gap']
        weighted_gap = convergence_metrics['weighted_gap']
        gap_coverage = convergence_metrics['gap_coverage']
        gap_coverage_sequences = convergence_metrics['gap_coverage_sequences']
        clusters_with_metrics = convergence_metrics['clusters_with_metrics']
        # Use next_clusters (post-refinement) as denominator since metrics are for output clusters
        coverage_pct = (clusters_with_metrics / len(next_clusters) * 100) if next_clusters else 0

        logger.info(f"  Convergence metrics (incremental):")
        logger.info(f"    - Mean gap: {mean_gap:.4f}, Weighted gap: {weighted_gap:.4f}")
        logger.info(f"    - Cluster coverage: {gap_coverage*100:.1f}%, Sequence coverage: {gap_coverage_sequences*100:.1f}%")
        logger.info(f"    - Computed for: {clusters_with_metrics}/{len(next_clusters)} clusters ({coverage_pct:.1f}%)")

        # Check if metrics are stabilized (all sequences processed at least once)
        if sequence_recluster_count:
            counts = list(sequence_recluster_count.values())
            min_recluster_count = min(counts)
            if min_recluster_count == 0:
                # Some sequences haven't been processed yet
                sequences_never_processed = sum(1 for c in counts if c == 0)
                logger.info(f"    - NOTE: Metrics not yet stabilized - {sequences_never_processed} sequences not yet processed in any refinement scope")

        # Checkpoint if enabled and at checkpoint interval
        if checkpoint_frequency > 0 and global_iteration % checkpoint_frequency == 0:
            if checkpoint_output_dir and header_mapping:
                # Add summary stats for checkpointing
                iteration_stats['clusters_before'] = len(current_clusters)
                iteration_stats['clusters_after'] = len(next_clusters)
                iteration_stats['ami'] = iteration_ami

                checkpoint_timing = {
                    'graph_build': graph_time,
                    'refinement': refinement_time,
                    'total': iteration_time
                }

                checkpoint_iteration(
                    iteration=global_iteration,
                    current_clusters=next_clusters,  # Checkpoint the NEW state
                    sequences=sequences,
                    headers=headers,
                    header_mapping=header_mapping,
                    sequence_recluster_count=sequence_recluster_count,
                    converged_scopes=converged_scopes,
                    iteration_stats=iteration_stats,
                    timing=checkpoint_timing,
                    base_output_dir=checkpoint_output_dir
                )

        # Check convergence: AMI = 1.0 means perfect agreement (identical clustering)
        if iteration_ami == 1.0:
            logger.info(f"Convergence achieved at iteration {global_iteration} (AMI = 1.0)")
            tracking_info.summary_stats = {'convergence_reason': 'ami_convergence'}
            break

        current_clusters = next_clusters

    if global_iteration >= max_iterations:
        logger.warning(f"Reached iteration limit ({max_iterations}) without convergence")
        tracking_info.summary_stats = {'convergence_reason': 'iteration_limit'}

    timing['total'] = time.time() - refinement_start_time
    timing['avg_iteration'] = sum(timing['iterations']) / len(timing['iterations']) if timing['iterations'] else 0.0
    timing['avg_graph'] = sum(timing['proximity_graphs']) / len(timing['proximity_graphs']) if timing['proximity_graphs'] else 0.0
    timing['avg_refinement'] = sum(timing['refinements']) / len(timing['refinements']) if timing['refinements'] else 0.0

    logger.info(f"Iterative refinement complete: {len(all_clusters)} clusters → {len(current_clusters)} clusters "
               f"({len(current_clusters) - len(all_clusters):+d})")
    logger.info(f"Refinement timing: total={timing['total']:.1f}s, "
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


def refine_clusters(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: Optional[RefinementConfig] = None,
    cluster_id_generator=None,
    show_progress: bool = False,
    checkpoint_frequency: int = 0,
    checkpoint_output_dir: Optional[Path] = None,
    header_mapping: Optional[Dict[str, str]] = None,
    resume_state: Optional[Dict] = None,
    gap_method: str = 'global',
    alpha: float = 0.0
) -> Tuple[Dict[str, List[str]], 'ProcessingStageInfo']:
    """Iteratively refine clusters using neighborhood-based approach.

    Main entry point for cluster refinement.

    Args:
        all_clusters: Current cluster dictionary
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        config: Configuration for refinement parameters
        cluster_id_generator: Optional cluster ID generator
        show_progress: Show progress bars during iterations
        checkpoint_frequency: Checkpoint every N iterations (0=disabled)
        checkpoint_output_dir: Base output directory for checkpoints
        header_mapping: Mapping of sequence IDs to full headers (for checkpointing)
        resume_state: Optional state to resume from (loaded from state.json)
        gap_method: Gap calculation method ('global' or 'local')
        alpha: Parsimony parameter for local gap method (default: 0.0)
              Score = local_gap / (num_clusters^alpha)

    Returns:
        Tuple of (refined_clusters, tracking_info)
    """
    if config is None:
        config = RefinementConfig()

    from .refinement_types import ClusterIDGenerator
    if cluster_id_generator is None:
        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

    logger.info("=" * 80)
    logger.info("ITERATIVE CLUSTER REFINEMENT")
    logger.info("=" * 80)
    logger.info(f"Initial clusters: {len(all_clusters)}")
    logger.info(f"Config: max_scope={config.max_full_gaphack_size}, "
               f"max_iterations={config.max_iterations}")

    # Determine close threshold (default to max_lump if not specified)
    close_threshold = config.close_threshold if config.close_threshold is not None else max_lump

    # Run iterative refinement
    final_clusters, tracking_info = iterative_refinement(
        all_clusters=all_clusters,
        sequences=sequences,
        headers=headers,
        min_split=min_split,
        max_lump=max_lump,
        target_percentile=target_percentile,
        close_threshold=close_threshold,
        max_iterations=config.max_iterations,
        config=config,
        cluster_id_generator=cluster_id_generator,
        show_progress=show_progress,
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_output_dir=checkpoint_output_dir,
        header_mapping=header_mapping,
        resume_state=resume_state,
        gap_method=gap_method,
        alpha=alpha
    )

    logger.info("=" * 80)
    logger.info(f"ITERATIVE REFINEMENT COMPLETE")
    logger.info(f"Final result: {len(all_clusters)} clusters → {len(final_clusters)} clusters "
               f"({len(final_clusters) - len(all_clusters):+d})")
    logger.info("=" * 80)

    return final_clusters, tracking_info


# ============================================================================
# Legacy functions removed (replaced by iterative refinement)
# ============================================================================
# - find_connected_close_components() → replaced by radius-based seeding
# - expand_context_for_gap_optimization() → replaced by build_refinement_scope()
# - refine_close_clusters() → replaced by iterative_refinement()
# - resolve_conflicts() → conflicts now handled by iterative refinement
# - find_conflict_components() → no longer needed
# - pass1_resolve_and_split() → removed (redundant with iterative refinement)
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