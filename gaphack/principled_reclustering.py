"""
Principled reclustering algorithms for achieving MECE clustering from gaphack-decompose.

This module implements scope-limited reclustering using classic gapHACk to resolve
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
MAX_CLASSIC_GAPHACK_SIZE = 300
PREFERRED_SCOPE_SIZE = 250
EXPANSION_SIZE_BUFFER = 50


class ReclusteringConfig:
    """Configuration for principled reclustering algorithms."""

    def __init__(self,
                 max_classic_gaphack_size: int = 300,
                 preferred_scope_size: int = 250,
                 expansion_size_buffer: int = 50,
                 conflict_expansion_threshold: Optional[float] = None,
                 close_cluster_expansion_threshold: Optional[float] = None,
                 incremental_search_distance: Optional[float] = None,
                 jaccard_overlap_threshold: float = 0.1,
                 significant_difference_threshold: float = 0.2,
                 max_closest_clusters: int = 5):
        """Initialize reclustering configuration.

        Args:
            max_classic_gaphack_size: Maximum sequences for classic gapHACk
            preferred_scope_size: Target scope size for optimal performance
            expansion_size_buffer: Reserve capacity for scope expansion
            conflict_expansion_threshold: Distance threshold for conflict scope expansion
            close_cluster_expansion_threshold: Distance threshold for close cluster expansion
            incremental_search_distance: Search distance for incremental updates
            jaccard_overlap_threshold: Overlap threshold for scope expansion
            significant_difference_threshold: Threshold for detecting significant clustering changes
            max_closest_clusters: Maximum clusters to consider for incremental updates
        """
        self.max_classic_gaphack_size = max_classic_gaphack_size
        self.preferred_scope_size = preferred_scope_size
        self.expansion_size_buffer = expansion_size_buffer
        self.conflict_expansion_threshold = conflict_expansion_threshold
        self.close_cluster_expansion_threshold = close_cluster_expansion_threshold
        self.incremental_search_distance = incremental_search_distance
        self.jaccard_overlap_threshold = jaccard_overlap_threshold
        self.significant_difference_threshold = significant_difference_threshold
        self.max_closest_clusters = max_closest_clusters


class ExpandedScope:
    """Container for expanded reclustering scope information."""

    def __init__(self, sequences: List[str], headers: List[str], cluster_ids: List[str]):
        self.sequences = sequences
        self.headers = headers
        self.cluster_ids = cluster_ids


def find_connected_conflict_components(conflicts: Dict[str, List[str]],
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
                             max_scope_size: int = MAX_CLASSIC_GAPHACK_SIZE) -> ExpandedScope:
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


def apply_classic_gaphack_to_scope(scope_sequences: List[str],
                                 scope_headers: List[str],
                                 global_sequences: List[str],
                                 global_headers: List[str],
                                 global_distance_provider: DistanceProvider,
                                 min_split: float = 0.005,
                                 max_lump: float = 0.02,
                                 target_percentile: int = 95) -> Dict[str, List[str]]:
    """Apply classic gapHACk clustering to a scope of sequences.

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
    # Create scoped distance provider
    scoped_provider = create_scoped_distance_provider(
        global_distance_provider, scope_headers, global_headers
    )

    # Build distance matrix for classic gapHACk
    distance_matrix = scoped_provider.build_distance_matrix()

    # Apply classic gapHACk clustering
    clusterer = GapOptimizedClustering(
        min_split=min_split,
        max_lump=max_lump,
        target_percentile=target_percentile,
        show_progress=False,  # Disable progress for scope-limited clustering
        logger=logger
    )

    # Get clustering result
    final_clusters, singletons, metadata = clusterer.cluster(distance_matrix)

    # Convert result to cluster dictionary format
    clusters = {}

    # Process multi-member clusters (list of lists of indices)
    for cluster_idx, cluster_indices_list in enumerate(final_clusters):
        cluster_headers = []
        for seq_idx in cluster_indices_list:
            cluster_headers.append(scope_headers[seq_idx])

        cluster_id = f"classic_{hash(tuple(sorted(cluster_headers)))}"
        clusters[cluster_id] = cluster_headers

    # Process singletons (list of indices)
    for singleton_idx in singletons:
        cluster_headers = [scope_headers[singleton_idx]]
        cluster_id = f"classic_{hash(tuple(cluster_headers))}"
        clusters[cluster_id] = cluster_headers

    logger.debug(f"Classic gapHACk on scope: {len(scope_sequences)} sequences → {len(clusters)} clusters")
    return clusters


def resolve_conflicts_via_reclustering(conflicts: Dict[str, List[str]],
                                     all_clusters: Dict[str, List[str]],
                                     sequences: List[str],
                                     headers: List[str],
                                     distance_provider: DistanceProvider,
                                     proximity_graph: ClusterProximityGraph,
                                     config: Optional[ReclusteringConfig] = None,
                                     min_split: float = 0.005,
                                     max_lump: float = 0.02,
                                     target_percentile: int = 95) -> Dict[str, List[str]]:
    """Resolve assignment conflicts using classic gapHACk reclustering.

    Args:
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        all_clusters: Dict mapping cluster_id -> list of sequence headers
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        distance_provider: Provider for distance calculations
        proximity_graph: Graph for finding nearby clusters
        config: Configuration for reclustering parameters
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Updated cluster dictionary with conflicts resolved
    """
    if config is None:
        config = ReclusteringConfig()

    if not conflicts:
        logger.info("No conflicts to resolve")
        return all_clusters.copy()

    logger.info(f"Resolving conflicts for {len(conflicts)} sequences across clusters")

    # Step 1: Group conflicts by connected components
    conflict_components = find_connected_conflict_components(conflicts, all_clusters)
    logger.debug(f"Found {len(conflict_components)} connected conflict components")

    updated_clusters = all_clusters.copy()

    # Process each conflict component
    for component_idx, component_clusters in enumerate(conflict_components):
        logger.debug(f"Processing conflict component {component_idx+1}/{len(conflict_components)} "
                    f"with {len(component_clusters)} clusters")

        # Step 2: Extract scope sequences
        scope_sequences = set()
        for cluster_id in component_clusters:
            scope_sequences.update(all_clusters[cluster_id])

        # Step 3: Apply scope expansion if beneficial and within size limits
        expansion_threshold = config.conflict_expansion_threshold
        if expansion_threshold is None:
            expansion_threshold = 1.5 * 0.02  # 1.5 * default max_lump

        expanded_scope = expand_scope_for_conflicts(
            scope_sequences, component_clusters, all_clusters,
            proximity_graph, expansion_threshold, config.max_classic_gaphack_size
        )

        # Step 4: Apply classic gapHACk to scope if size is manageable
        if len(expanded_scope.sequences) <= config.max_classic_gaphack_size:
            logger.debug(f"Applying classic gapHACk to scope of {len(expanded_scope.sequences)} sequences")

            classic_result = apply_classic_gaphack_to_scope(
                expanded_scope.sequences, expanded_scope.headers,
                sequences, headers, distance_provider,
                min_split, max_lump, target_percentile
            )

            # Step 5: Replace original clusters with classic result
            for cluster_id in expanded_scope.cluster_ids:
                if cluster_id in updated_clusters:
                    del updated_clusters[cluster_id]

            # Add new clusters from classic result
            for cluster_id, cluster_headers in classic_result.items():
                updated_clusters[cluster_id] = cluster_headers

            logger.info(f"Resolved conflict component: {len(expanded_scope.cluster_ids)} clusters → "
                       f"{len(classic_result)} clusters")

        else:
            # Fallback: skip oversized components with warning
            logger.warning(f"Skipping conflict component with {len(expanded_scope.sequences)} sequences "
                          f"(exceeds limit of {config.max_classic_gaphack_size})")

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

    return updated_clusters


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