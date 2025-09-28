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

from .cluster_proximity import ClusterProximityGraph, BruteForceProximityGraph
from .scoped_distances import ScopedDistanceProvider, create_scoped_distance_provider
from .lazy_distances import DistanceProvider
from .core import GapOptimizedClustering
from .decompose import DecomposeResults


logger = logging.getLogger(__name__)

# Configuration constants
MAX_FULL_GAPHACK_SIZE = 300
PREFERRED_SCOPE_SIZE = 250
EXPANSION_SIZE_BUFFER = 50


class RefinementConfig:
    """Configuration for cluster refinement algorithms."""

    def __init__(self,
                 max_full_gaphack_size: int = 300,
                 preferred_scope_size: int = 250,
                 expansion_size_buffer: int = 50,
                 conflict_expansion_threshold: Optional[float] = None,
                 close_cluster_expansion_threshold: Optional[float] = None,
                 incremental_search_distance: Optional[float] = None,
                 jaccard_overlap_threshold: float = 0.1,
                 significant_difference_threshold: float = 0.2,
                 max_closest_clusters: int = 5):
        """Initialize refinement configuration.

        Args:
            max_full_gaphack_size: Maximum sequences for full gapHACk
            preferred_scope_size: Target scope size for optimal performance
            expansion_size_buffer: Reserve capacity for scope expansion
            conflict_expansion_threshold: Distance threshold for conflict scope expansion
            close_cluster_expansion_threshold: Distance threshold for close cluster expansion
            incremental_search_distance: Search distance for incremental updates
            jaccard_overlap_threshold: Overlap threshold for scope expansion
            significant_difference_threshold: Threshold for detecting significant clustering changes
            max_closest_clusters: Maximum clusters to consider for incremental updates
        """
        self.max_full_gaphack_size = max_full_gaphack_size
        self.preferred_scope_size = preferred_scope_size
        self.expansion_size_buffer = expansion_size_buffer
        self.conflict_expansion_threshold = conflict_expansion_threshold
        self.close_cluster_expansion_threshold = close_cluster_expansion_threshold
        self.incremental_search_distance = incremental_search_distance
        self.jaccard_overlap_threshold = jaccard_overlap_threshold
        self.significant_difference_threshold = significant_difference_threshold
        self.max_closest_clusters = max_closest_clusters


class ExpandedScope:
    """Container for expanded refinement scope information."""

    def __init__(self, sequences: List[str], headers: List[str], cluster_ids: List[str]):
        self.sequences = sequences
        self.headers = headers
        self.cluster_ids = cluster_ids


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


def expand_scope_for_conflicts(initial_sequences: Set[str],
                             core_cluster_ids: List[str],
                             all_clusters: Dict[str, List[str]],
                             proximity_graph: ClusterProximityGraph,
                             expansion_threshold: float,
                             max_scope_size: int = MAX_FULL_GAPHACK_SIZE) -> ExpandedScope:
    """Expand conflict resolution scope to include nearby clusters.

    Args:
        initial_sequences: Initial set of sequence headers in scope
        core_cluster_ids: Core cluster IDs that must be included
        all_clusters: All cluster data
        proximity_graph: Graph for finding nearby clusters
        expansion_threshold: Distance threshold for expansion
        max_scope_size: Maximum number of sequences in expanded scope

    Returns:
        ExpandedScope containing expanded sequence and cluster sets
    """
    expanded_sequences = initial_sequences.copy()
    expanded_cluster_ids = set(core_cluster_ids)

    # Find candidate clusters for expansion
    candidates = []
    for cluster_id in core_cluster_ids:
        neighbors = proximity_graph.get_neighbors_within_distance(cluster_id, expansion_threshold)
        for neighbor_id, distance in neighbors:
            if neighbor_id not in expanded_cluster_ids and neighbor_id in all_clusters:
                candidates.append((neighbor_id, distance))

    # Sort candidates by distance and add until size limit
    candidates.sort(key=lambda x: x[1])

    for neighbor_id, distance in candidates:
        neighbor_sequences = set(all_clusters[neighbor_id])
        potential_size = len(expanded_sequences | neighbor_sequences)

        if potential_size <= max_scope_size:
            expanded_sequences.update(neighbor_sequences)
            expanded_cluster_ids.add(neighbor_id)
        else:
            break  # Would exceed size limit

    return ExpandedScope(
        sequences=list(expanded_sequences),
        headers=list(expanded_sequences),  # Headers same as sequences for decompose
        cluster_ids=list(expanded_cluster_ids)
    )


def apply_full_gaphack_to_scope_with_metadata(scope_sequences: List[str],
                                               scope_headers: List[str],
                                               global_sequences: List[str],
                                               global_headers: List[str],
                                               global_distance_provider: DistanceProvider,
                                               min_split: float = 0.005,
                                               max_lump: float = 0.02,
                                               target_percentile: int = 95,
                                               cluster_id_generator=None) -> Tuple[Dict[str, List[str]], Dict]:
    """Apply full gapHACk clustering to a scope of sequences, returning clusters and metadata.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        global_sequences: Full sequence list
        global_headers: Full header list
        global_distance_provider: Distance provider for full dataset
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Tuple of (cluster_dict, metadata_dict) where metadata includes gap_size
    """
    # Create scoped distance provider
    scoped_provider = create_scoped_distance_provider(
        global_distance_provider, scope_headers, global_headers
    )

    # Build distance matrix for full gapHACk
    distance_matrix = scoped_provider.build_distance_matrix()

    # Apply full gapHACk clustering
    clusterer = GapOptimizedClustering(
        min_split=min_split,
        max_lump=max_lump,
        target_percentile=target_percentile,
        show_progress=False,  # Disable progress for scope-limited clustering
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
        cluster_id_generator = ClusterIDGenerator(prefix="classic")

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


def apply_full_gaphack_to_scope(scope_sequences: List[str],
                                 scope_headers: List[str],
                                 global_sequences: List[str],
                                 global_headers: List[str],
                                 global_distance_provider: DistanceProvider,
                                 min_split: float = 0.005,
                                 max_lump: float = 0.02,
                                 target_percentile: int = 95,
                                 cluster_id_generator=None) -> Dict[str, List[str]]:
    """Apply full gapHACk clustering to a scope of sequences.

    Args:
        scope_sequences: Sequences to cluster (in scope order)
        scope_headers: Headers for scope sequences
        global_sequences: Full sequence list
        global_headers: Full header list
        global_distance_provider: Distance provider for full dataset
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Dict mapping cluster_id -> list of sequence headers
    """
    # Use the metadata version and just return the clusters
    clusters, _ = apply_full_gaphack_to_scope_with_metadata(
        scope_sequences, scope_headers, global_sequences, global_headers,
        global_distance_provider, min_split, max_lump, target_percentile,
        cluster_id_generator
    )
    return clusters


def resolve_conflicts(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]],
                                     sequences: List[str],
                                     headers: List[str],
                                     distance_provider: DistanceProvider,
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
        distance_provider: Provider for distance calculations
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
        scope_sequences = set()
        for cluster_id in component_clusters:
            scope_sequences.update(all_clusters[cluster_id])

        scope_headers = list(scope_sequences)  # Headers same as sequences for decompose

        # Track component before processing
        component_info = {
            'component_index': component_idx,
            'clusters_before': list(component_clusters),
            'clusters_before_count': len(component_clusters),
            'sequences_count': len(scope_sequences),
            'processed': False
        }

        # Step 3: Apply full gapHACk to minimal conflict scope (no expansion)
        if len(scope_sequences) <= config.max_full_gaphack_size:
            logger.debug(f"Applying full gapHACk to minimal conflict scope of {len(scope_sequences)} sequences")

            full_result = apply_full_gaphack_to_scope(
                scope_headers, scope_headers,  # Use minimal scope directly
                sequences, headers, distance_provider,
                min_split, max_lump, target_percentile,
                cluster_id_generator
            )

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
            logger.warning(f"Skipping conflict component with {len(scope_sequences)} sequences "
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


def find_connected_close_components(close_pairs: List[Tuple[str, str, float]]) -> List[List[str]]:
    """Group close cluster pairs into connected components.

    Two clusters are connected if they form a close pair.

    Args:
        close_pairs: List of (cluster1_id, cluster2_id, distance) tuples

    Returns:
        List of connected components, each being a list of cluster IDs
    """
    # Build cluster adjacency graph
    cluster_graph = defaultdict(set)
    all_clusters = set()

    for cluster1_id, cluster2_id, distance in close_pairs:
        cluster_graph[cluster1_id].add(cluster2_id)
        cluster_graph[cluster2_id].add(cluster1_id)
        all_clusters.add(cluster1_id)
        all_clusters.add(cluster2_id)

    # Find connected components using DFS
    visited = set()
    components = []

    def dfs_traverse(cluster_id: str, component: List[str]) -> None:
        visited.add(cluster_id)
        component.append(cluster_id)
        for neighbor in cluster_graph[cluster_id]:
            if neighbor not in visited:
                dfs_traverse(neighbor, component)

    for cluster_id in all_clusters:
        if cluster_id not in visited:
            component = []
            dfs_traverse(cluster_id, component)
            components.append(component)

    return components


def needs_minimal_context_for_gap_calculation(core_cluster_ids: List[str],
                                             all_clusters: Dict[str, List[str]],
                                             proximity_graph: ClusterProximityGraph,
                                             max_lump: float) -> bool:
    """Check if we need to add context to prevent gap calculation issues.

    Returns True if all pairwise cluster distances are within max_lump,
    meaning full gapHACk would merge everything and have no inter-cluster distances.

    Args:
        core_cluster_ids: Core cluster IDs being refined
        all_clusters: All cluster data
        proximity_graph: Graph for distance queries
        max_lump: Maximum distance for full gapHACk merging

    Returns:
        True if minimal context is needed for proper gap calculation
    """
    if len(core_cluster_ids) <= 1:
        return True  # Single cluster needs context by definition

    # Check if all pairwise cluster distances are within max_lump
    for i, cluster1_id in enumerate(core_cluster_ids):
        for cluster2_id in core_cluster_ids[i+1:]:
            # Find distance between these clusters using proximity graph
            neighbors = proximity_graph.get_neighbors_within_distance(cluster1_id, max_lump * 2)
            cluster_distance = None

            for neighbor_id, distance in neighbors:
                if neighbor_id == cluster2_id:
                    cluster_distance = distance
                    break

            if cluster_distance is None:
                # Distance not found in neighbors, likely > max_lump
                return False
            elif cluster_distance > max_lump:
                return False  # Found a pair beyond max_lump

    return True  # All distances ≤ max_lump → everything would merge → need context


def add_context_at_distance_threshold(current_cluster_ids: Set[str],
                                     all_clusters: Dict[str, List[str]],
                                     proximity_graph: ClusterProximityGraph,
                                     distance_threshold: float,
                                     max_scope_size: int) -> Tuple[bool, str, float]:
    """Add one context cluster beyond the given distance threshold.

    Args:
        current_cluster_ids: Current clusters in scope
        all_clusters: All available clusters
        proximity_graph: Graph for finding neighbors
        distance_threshold: Minimum distance for context clusters
        max_scope_size: Maximum total sequences allowed

    Returns:
        Tuple of (success, cluster_id_added, distance_added)
    """
    current_sequences = sum(len(all_clusters[cid]) for cid in current_cluster_ids if cid in all_clusters)

    for cluster_id in current_cluster_ids:
        # Find clusters beyond distance threshold
        all_neighbors = proximity_graph.get_k_nearest_neighbors(cluster_id, k=30)
        context_candidates = [
            (neighbor_id, distance) for neighbor_id, distance in all_neighbors
            if (distance > distance_threshold and
                neighbor_id not in current_cluster_ids and
                neighbor_id in all_clusters)
        ]

        if context_candidates:
            # Try to add the closest context cluster
            context_id, context_distance = context_candidates[0]
            context_size = len(all_clusters[context_id])

            if current_sequences + context_size <= max_scope_size:
                return True, context_id, context_distance

    return False, "", 0.0


def expand_context_for_gap_optimization(core_cluster_ids: List[str],
                                       all_clusters: Dict[str, List[str]],
                                       sequences: List[str],
                                       headers: List[str],
                                       distance_provider: DistanceProvider,
                                       proximity_graph: ClusterProximityGraph,
                                       expansion_threshold: float,
                                       max_scope_size: int,
                                       max_lump: float,
                                       min_split: float,
                                       target_percentile: int,
                                       target_gap: float = 0.001,
                                       max_iterations: int = 5,
                                       cluster_id_generator=None) -> Tuple[ExpandedScope, Dict]:
    """Iteratively expand context until positive gap is achieved.

    Args:
        core_cluster_ids: Core clusters to refine
        all_clusters: All available clusters
        sequences: Full sequence list
        headers: Full header list
        distance_provider: Distance calculation provider
        proximity_graph: Graph for finding neighbors
        expansion_threshold: Regular expansion threshold
        max_scope_size: Maximum sequences in scope
        max_lump: Classic gapHACk max lump threshold
        min_split: Classic gapHACk min split threshold
        target_percentile: Gap calculation percentile
        target_gap: Minimum gap to achieve
        max_iterations: Maximum context expansion iterations

    Returns:
        Tuple of (final_scope, full_gaphack_result)
    """
    # Start with core clusters
    current_cluster_ids = set(core_cluster_ids)
    iteration = 0
    best_result = None
    best_gap = float('-inf')

    logger.debug(f"Starting iterative context expansion for {len(core_cluster_ids)} core clusters")

    while iteration < max_iterations:
        # Build current scope
        scope_sequences = set()
        for cluster_id in current_cluster_ids:
            if cluster_id in all_clusters:
                scope_sequences.update(all_clusters[cluster_id])

        current_scope = ExpandedScope(
            sequences=list(scope_sequences),
            headers=list(scope_sequences),
            cluster_ids=list(current_cluster_ids)
        )

        # Apply full gapHACk to current scope
        if len(scope_sequences) <= max_scope_size:
            logger.debug(f"Iteration {iteration}: testing scope with {len(current_cluster_ids)} clusters, "
                        f"{len(scope_sequences)} sequences")

            full_clusters, full_metadata = apply_full_gaphack_to_scope_with_metadata(
                current_scope.sequences, current_scope.headers,
                sequences, headers, distance_provider,
                min_split, max_lump, target_percentile,
                cluster_id_generator=cluster_id_generator  # Use shared generator
            )

            # Extract gap from the metadata
            current_gap = full_metadata.get('gap_size', float('-inf'))

            logger.debug(f"Iteration {iteration}: achieved gap = {current_gap:.4f}")

            # Track best result
            if current_gap > best_gap:
                best_gap = current_gap
                best_result = full_clusters

            # Check if we've achieved target gap
            if current_gap >= target_gap:
                logger.info(f"Achieved positive gap {current_gap:.4f} after {iteration} context expansions")
                return current_scope, full_clusters

        # Gap still insufficient - add more distant context
        context_distance = max_lump * (1.5 + iteration * 0.5)  # 1.5x, 2.0x, 2.5x, etc.
        logger.debug(f"Iteration {iteration}: gap {best_gap:.4f} < target {target_gap:.4f}, "
                    f"adding context at distance {context_distance:.3f}")

        # Try to add context cluster
        success, context_id, actual_distance = add_context_at_distance_threshold(
            current_cluster_ids, all_clusters, proximity_graph,
            context_distance, max_scope_size
        )

        if success:
            current_cluster_ids.add(context_id)
            logger.debug(f"Added context cluster {context_id} at distance {actual_distance:.3f}")
        else:
            logger.debug(f"No context available at distance {context_distance:.3f} or would exceed size limit")
            break

        iteration += 1

    # Return best result found
    if best_result is None:
        logger.warning("No valid full gapHACk results obtained during iterative expansion")
        # Fallback to minimal scope
        scope_sequences = set()
        for cluster_id in core_cluster_ids:
            if cluster_id in all_clusters:
                scope_sequences.update(all_clusters[cluster_id])

        final_scope = ExpandedScope(
            sequences=list(scope_sequences),
            headers=list(scope_sequences),
            cluster_ids=core_cluster_ids
        )
        return final_scope, {}

    logger.info(f"Iterative expansion complete: best gap {best_gap:.4f} after {iteration} iterations")

    # Build final scope corresponding to best result
    final_scope_sequences = set()
    for cluster_id in current_cluster_ids:
        if cluster_id in all_clusters:
            final_scope_sequences.update(all_clusters[cluster_id])

    final_scope = ExpandedScope(
        sequences=list(final_scope_sequences),
        headers=list(final_scope_sequences),
        cluster_ids=list(current_cluster_ids)
    )

    return final_scope, best_result


def expand_scope_for_close_clusters(initial_sequences: Set[str],
                                   core_cluster_ids: List[str],
                                   all_clusters: Dict[str, List[str]],
                                   proximity_graph: ClusterProximityGraph,
                                   expansion_threshold: float,
                                   max_scope_size: int = MAX_FULL_GAPHACK_SIZE,
                                   max_lump: float = 0.02) -> ExpandedScope:
    """Expand close cluster refinement scope to include nearby clusters.

    Args:
        initial_sequences: Initial set of sequence headers in scope
        core_cluster_ids: Core cluster IDs that must be included
        all_clusters: All cluster data
        proximity_graph: Graph for finding nearby clusters
        expansion_threshold: Distance threshold for expansion
        max_scope_size: Maximum number of sequences in expanded scope
        max_lump: Maximum distance for full gapHACk merging

    Returns:
        ExpandedScope containing expanded sequence and cluster sets
    """
    expanded_sequences = initial_sequences.copy()
    expanded_cluster_ids = set(core_cluster_ids)

    # Step 1: Check if we need minimal context for proper gap calculation
    needs_context = needs_minimal_context_for_gap_calculation(
        core_cluster_ids, all_clusters, proximity_graph, max_lump
    )

    if needs_context:
        logger.debug(f"Adding minimal context: all core clusters within max_lump ({max_lump:.3f})")

        # Add the closest cluster that's beyond max_lump distance
        context_threshold = max_lump * 1.01  # Just beyond max_lump
        context_added = False

        for cluster_id in core_cluster_ids:
            # Find clusters beyond max_lump distance
            all_neighbors = proximity_graph.get_k_nearest_neighbors(cluster_id, k=20)
            context_candidates = [
                (neighbor_id, distance) for neighbor_id, distance in all_neighbors
                if (distance > context_threshold and
                    neighbor_id not in expanded_cluster_ids and
                    neighbor_id in all_clusters)
            ]

            if context_candidates:
                # Add the closest context cluster
                context_id, context_distance = context_candidates[0]
                context_sequences = set(all_clusters[context_id])
                potential_size = len(expanded_sequences | context_sequences)

                if potential_size <= max_scope_size:
                    expanded_sequences.update(context_sequences)
                    expanded_cluster_ids.add(context_id)
                    logger.debug(f"Added minimal context cluster {context_id} at distance {context_distance:.3f}")
                    context_added = True
                    break

        if not context_added:
            logger.debug("Could not add minimal context within size constraints")

    # Step 2: Regular expansion within threshold (if there's still room)
    candidates = []
    for cluster_id in core_cluster_ids:
        neighbors = proximity_graph.get_neighbors_within_distance(cluster_id, expansion_threshold)
        for neighbor_id, distance in neighbors:
            if neighbor_id not in expanded_cluster_ids and neighbor_id in all_clusters:
                candidates.append((neighbor_id, distance))

    # Sort candidates by distance and add until size limit
    candidates.sort(key=lambda x: x[1])

    for neighbor_id, distance in candidates:
        neighbor_sequences = set(all_clusters[neighbor_id])
        potential_size = len(expanded_sequences | neighbor_sequences)

        if potential_size <= max_scope_size:
            expanded_sequences.update(neighbor_sequences)
            expanded_cluster_ids.add(neighbor_id)
        else:
            break  # Would exceed size limit

    return ExpandedScope(
        sequences=list(expanded_sequences),
        headers=list(expanded_sequences),  # Headers same as sequences for decompose
        cluster_ids=list(expanded_cluster_ids)
    )


def refine_close_clusters(all_clusters: Dict[str, List[str]],
                         sequences: List[str],
                         headers: List[str],
                         distance_provider: DistanceProvider,
                         proximity_graph: ClusterProximityGraph,
                         config: Optional[RefinementConfig] = None,
                         min_split: float = 0.005,
                         max_lump: float = 0.02,
                         target_percentile: int = 95,
                         close_threshold: Optional[float] = None,
                         cluster_id_generator=None) -> Tuple[Dict[str, List[str]], 'ProcessingStageInfo']:
    """Refine clusters that are closer than expected barcode gaps.

    Args:
        all_clusters: Current cluster dictionary
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        distance_provider: Provider for distance calculations
        proximity_graph: Graph for finding close cluster pairs
        config: Configuration for refinement parameters
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        close_threshold: Distance threshold for "close" clusters (default: max_lump)

    Returns:
        Tuple of (updated_clusters, tracking_info): Updated cluster dictionary with close clusters refined and tracking information
    """
    if config is None:
        config = RefinementConfig()

    if close_threshold is None:
        close_threshold = max_lump

    # Initialize tracking
    from .decompose import ProcessingStageInfo
    tracking_info = ProcessingStageInfo(
        stage_name="Close Cluster Refinement",
        clusters_before=all_clusters.copy(),
        summary_stats={
            'close_threshold': close_threshold,
            'clusters_before_count': len(all_clusters)
        }
    )

    if len(all_clusters) < 2:
        logger.info("Insufficient clusters for close cluster refinement")
        result = all_clusters.copy()
        tracking_info.clusters_after = result
        tracking_info.summary_stats.update({
            'clusters_after_count': len(result),
            'components_processed_count': 0,
            'close_pairs_found': 0
        })
        return result, tracking_info

    logger.info(f"Refining close clusters with threshold {close_threshold:.4f}")

    # Step 1: Identify close cluster pairs via medoid analysis
    close_pairs = proximity_graph.find_close_pairs(close_threshold)

    if not close_pairs:
        logger.info("No close cluster pairs found")
        result = all_clusters.copy()
        tracking_info.clusters_after = result
        tracking_info.summary_stats.update({
            'clusters_after_count': len(result),
            'components_processed_count': 0,
            'close_pairs_found': 0
        })
        return result, tracking_info

    logger.debug(f"Found {len(close_pairs)} close cluster pairs")

    # Update tracking with close pairs info
    tracking_info.summary_stats['close_pairs_found'] = len(close_pairs)
    tracking_info.summary_stats['close_pairs_list'] = [(p[0], p[1], p[2]) for p in close_pairs]

    # Step 2: Group close pairs into connected components
    close_components = find_connected_close_components(close_pairs)
    logger.debug(f"Found {len(close_components)} close cluster components")

    # Step 3: Track processed components to avoid infinite loops
    processed_components = set()
    updated_clusters = all_clusters.copy()

    # Track all cluster changes for accurate net change calculation
    clusters_deleted_during_processing = set()
    clusters_created_during_processing = set()

    # Create shared cluster ID generator for all components
    if cluster_id_generator is None:
        from .decompose import ClusterIDGenerator
        cluster_id_generator = ClusterIDGenerator(prefix="classic")

    for component_idx, component_clusters in enumerate(close_components):
        component_signature = frozenset(component_clusters)

        if component_signature in processed_components:
            logger.debug(f"Skipping already processed component {component_idx+1}")
            continue  # Skip already processed components

        logger.debug(f"Processing close cluster component {component_idx+1}/{len(close_components)} "
                    f"with {len(component_clusters)} clusters")

        # Track component before processing - only count clusters that actually exist
        existing_component_clusters = [cid for cid in component_clusters if cid in updated_clusters]
        component_info = {
            'component_index': component_idx,
            'clusters_before': existing_component_clusters,
            'clusters_before_count': len(existing_component_clusters),
            'processed': False
        }

        # Step 4: Extract and expand scope
        scope_sequences = set()
        for cluster_id in existing_component_clusters:
            if cluster_id in updated_clusters:
                scope_sequences.update(updated_clusters[cluster_id])

        component_info['sequences_count'] = len(scope_sequences)

        if not scope_sequences:
            logger.warning(f"No sequences found for component {component_idx+1} (all existing clusters are empty)")
            component_info.update({
                'processed': False,
                'skipped_reason': 'no_sequences',
                'clusters_after': existing_component_clusters,  # They remain unchanged
                'clusters_after_count': len(existing_component_clusters)
            })
            tracking_info.components_processed.append(component_info)
            processed_components.add(component_signature)
            continue

        # Step 5: Apply iterative context expansion for positive gap
        expansion_threshold = config.close_cluster_expansion_threshold
        if expansion_threshold is None:
            expansion_threshold = close_threshold * 1.2  # Slightly broader than close threshold

        # Use iterative expansion to achieve positive gap
        logger.debug(f"Applying iterative context expansion to component with {len(existing_component_clusters)} clusters")

        expanded_scope, full_result = expand_context_for_gap_optimization(
            existing_component_clusters,
            updated_clusters,
            sequences,
            headers,
            distance_provider,
            proximity_graph,
            expansion_threshold,
            config.max_full_gaphack_size,
            max_lump,
            min_split,
            target_percentile,
            target_gap=0.001,  # Target positive gap
            max_iterations=5,
            cluster_id_generator=cluster_id_generator
        )

        # Step 6: Check if we got a valid result
        if full_result and len(full_result) > 0:
            # Step 7: Check if classic result differs significantly from input
            original_scope_clusters = {cid: updated_clusters[cid] for cid in expanded_scope.cluster_ids
                                     if cid in updated_clusters}

            if clusters_significantly_different(original_scope_clusters, full_result,
                                              config.significant_difference_threshold):
                # Step 8: Replace original clusters with classic result
                for cluster_id in expanded_scope.cluster_ids:
                    if cluster_id in updated_clusters:
                        clusters_deleted_during_processing.add(cluster_id)
                        del updated_clusters[cluster_id]

                # Add new clusters from classic result
                for cluster_id, cluster_headers in full_result.items():
                    clusters_created_during_processing.add(cluster_id)
                    updated_clusters[cluster_id] = cluster_headers

                # Update component tracking for successful refinement
                component_info.update({
                    'clusters_before': list(expanded_scope.cluster_ids),  # Use expanded scope clusters
                    'clusters_before_count': len(expanded_scope.cluster_ids),  # Use expanded scope as "before"
                    'clusters_after': list(full_result.keys()),
                    'clusters_after_count': len(full_result),
                    'processed': True,
                    'significantly_different': True,
                    'expanded_scope_size': len(expanded_scope.cluster_ids)
                })

                logger.info(f"Refined close cluster component: {len(expanded_scope.cluster_ids)} clusters → "
                           f"{len(full_result)} clusters")
            else:
                # Update component tracking for no significant change
                component_info.update({
                    'clusters_before': list(expanded_scope.cluster_ids),  # Use expanded scope clusters
                    'clusters_before_count': len(expanded_scope.cluster_ids),  # Use expanded scope as "before"
                    'clusters_after': list(expanded_scope.cluster_ids),  # Keep expanded scope clusters
                    'clusters_after_count': len(expanded_scope.cluster_ids),  # No net change
                    'processed': True,
                    'significantly_different': False,
                    'expanded_scope_size': len(expanded_scope.cluster_ids)
                })
                logger.debug(f"Classic gapHACk result not significantly different, keeping original clusters")

        else:
            # Fallback: iterative expansion failed
            component_info.update({
                'clusters_after': list(component_clusters),
                'clusters_after_count': len(component_clusters),
                'processed': False,
                'skipped_reason': 'expansion_failed'
            })
            logger.warning(f"Iterative context expansion failed for component with {len(component_clusters)} clusters")

        # Add component info to tracking
            tracking_info.components_processed.append(component_info)

        # Mark as processed regardless of outcome
        processed_components.add(component_signature)

    # Summary statistics
    original_count = len(all_clusters)
    final_count = len(updated_clusters)
    logger.info(f"Close cluster refinement: {original_count} clusters → {final_count} clusters "
               f"({final_count - original_count:+d})")

    # Finalize tracking information
    tracking_info.clusters_after = updated_clusters
    tracking_info.summary_stats.update({
        'clusters_after_count': final_count,
        'components_processed_count': len(close_components),
        'cluster_count_change': final_count - original_count,
        'total_clusters_deleted': len(clusters_deleted_during_processing),
        'total_clusters_created': len(clusters_created_during_processing),
        'net_cluster_change_from_tracking': len(clusters_created_during_processing) - len(clusters_deleted_during_processing)
    })
    return updated_clusters, tracking_info


def clusters_significantly_different(original_clusters: Dict[str, List[str]],
                                   new_clusters: Dict[str, List[str]],
                                   threshold: float = 0.2) -> bool:
    """Check if clustering results are significantly different.

    Args:
        original_clusters: Original cluster assignments
        new_clusters: New cluster assignments
        threshold: Fraction of sequences that must change clusters

    Returns:
        True if clusters are significantly different
    """
    # Create sequence -> cluster mappings
    original_assignments = {}
    for cluster_id, sequences in original_clusters.items():
        for seq in sequences:
            original_assignments[seq] = cluster_id

    new_assignments = {}
    for cluster_id, sequences in new_clusters.items():
        for seq in sequences:
            new_assignments[seq] = cluster_id

    # Count sequences with different assignments
    all_sequences = set(original_assignments.keys()) | set(new_assignments.keys())
    changed_sequences = 0

    for seq in all_sequences:
        original_cluster = original_assignments.get(seq)
        new_cluster = new_assignments.get(seq)
        if original_cluster != new_cluster:
            changed_sequences += 1

    change_fraction = changed_sequences / len(all_sequences) if all_sequences else 0
    return change_fraction >= threshold


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