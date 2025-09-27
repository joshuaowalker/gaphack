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
                                     config: Optional[ReclusteringConfig] = None,
                                     min_split: float = 0.005,
                                     max_lump: float = 0.02,
                                     target_percentile: int = 95) -> Dict[str, List[str]]:
    """Resolve assignment conflicts using classic gapHACk reclustering with minimal scope.

    Uses only conflicted clusters (no expansion) for fastest, most predictable MECE fixes.
    This is pure correctness operation - quality improvement belongs to close cluster refinement.

    Args:
        conflicts: Dict mapping sequence_id -> list of cluster_ids containing sequence
        all_clusters: Dict mapping cluster_id -> list of sequence headers
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        distance_provider: Provider for distance calculations
        config: Configuration for reclustering parameters
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization

    Returns:
        Updated cluster dictionary with conflicts resolved using minimal scope
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

        # Step 2: Extract scope sequences (minimal scope - only conflicted clusters)
        scope_sequences = set()
        for cluster_id in component_clusters:
            scope_sequences.update(all_clusters[cluster_id])

        scope_headers = list(scope_sequences)  # Headers same as sequences for decompose

        # Step 3: Apply classic gapHACk to minimal conflict scope (no expansion)
        if len(scope_sequences) <= config.max_classic_gaphack_size:
            logger.debug(f"Applying classic gapHACk to minimal conflict scope of {len(scope_sequences)} sequences")

            classic_result = apply_classic_gaphack_to_scope(
                scope_headers, scope_headers,  # Use minimal scope directly
                sequences, headers, distance_provider,
                min_split, max_lump, target_percentile
            )

            # Step 4: Replace original conflicted clusters with classic result
            for cluster_id in component_clusters:
                if cluster_id in updated_clusters:
                    del updated_clusters[cluster_id]

            # Add new clusters from classic result
            for cluster_id, cluster_headers in classic_result.items():
                updated_clusters[cluster_id] = cluster_headers

            logger.info(f"Resolved conflict component: {len(component_clusters)} clusters → "
                       f"{len(classic_result)} clusters")

        else:
            # Fallback: skip oversized components with warning
            logger.warning(f"Skipping conflict component with {len(scope_sequences)} sequences "
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


def expand_scope_for_close_clusters(initial_sequences: Set[str],
                                   core_cluster_ids: List[str],
                                   all_clusters: Dict[str, List[str]],
                                   proximity_graph: ClusterProximityGraph,
                                   expansion_threshold: float,
                                   max_scope_size: int = MAX_CLASSIC_GAPHACK_SIZE) -> ExpandedScope:
    """Expand close cluster refinement scope to include nearby clusters.

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


def refine_close_clusters(all_clusters: Dict[str, List[str]],
                         sequences: List[str],
                         headers: List[str],
                         distance_provider: DistanceProvider,
                         proximity_graph: ClusterProximityGraph,
                         config: Optional[ReclusteringConfig] = None,
                         min_split: float = 0.005,
                         max_lump: float = 0.02,
                         target_percentile: int = 95,
                         close_threshold: Optional[float] = None) -> Dict[str, List[str]]:
    """Refine clusters that are closer than expected barcode gaps.

    Args:
        all_clusters: Current cluster dictionary
        sequences: Full sequence list
        headers: Full header list (indices must match sequences)
        distance_provider: Provider for distance calculations
        proximity_graph: Graph for finding close cluster pairs
        config: Configuration for reclustering parameters
        min_split: Minimum distance to split clusters
        max_lump: Maximum distance to lump clusters
        target_percentile: Percentile for gap optimization
        close_threshold: Distance threshold for "close" clusters (default: max_lump)

    Returns:
        Updated cluster dictionary with close clusters refined
    """
    if config is None:
        config = ReclusteringConfig()

    if close_threshold is None:
        close_threshold = max_lump

    if len(all_clusters) < 2:
        logger.info("Insufficient clusters for close cluster refinement")
        return all_clusters.copy()

    logger.info(f"Refining close clusters with threshold {close_threshold:.4f}")

    # Step 1: Identify close cluster pairs via medoid analysis
    close_pairs = proximity_graph.find_close_pairs(close_threshold)

    if not close_pairs:
        logger.info("No close cluster pairs found")
        return all_clusters.copy()

    logger.debug(f"Found {len(close_pairs)} close cluster pairs")

    # Step 2: Group close pairs into connected components
    close_components = find_connected_close_components(close_pairs)
    logger.debug(f"Found {len(close_components)} close cluster components")

    # Step 3: Track processed components to avoid infinite loops
    processed_components = set()
    updated_clusters = all_clusters.copy()

    for component_idx, component_clusters in enumerate(close_components):
        component_signature = frozenset(component_clusters)

        if component_signature in processed_components:
            logger.debug(f"Skipping already processed component {component_idx+1}")
            continue  # Skip already processed components

        logger.debug(f"Processing close cluster component {component_idx+1}/{len(close_components)} "
                    f"with {len(component_clusters)} clusters")

        # Step 4: Extract and expand scope
        scope_sequences = set()
        for cluster_id in component_clusters:
            if cluster_id in updated_clusters:
                scope_sequences.update(updated_clusters[cluster_id])

        if not scope_sequences:
            logger.warning(f"No sequences found for component {component_idx+1}")
            processed_components.add(component_signature)
            continue

        # Step 5: Apply scope expansion if beneficial and within size limits
        expansion_threshold = config.close_cluster_expansion_threshold
        if expansion_threshold is None:
            expansion_threshold = close_threshold * 1.2  # Slightly broader than close threshold

        expanded_scope = expand_scope_for_close_clusters(
            scope_sequences, component_clusters, updated_clusters,
            proximity_graph, expansion_threshold, config.max_classic_gaphack_size
        )

        # Step 6: Apply classic gapHACk if scope is manageable
        if len(expanded_scope.sequences) <= config.max_classic_gaphack_size:
            logger.debug(f"Applying classic gapHACk to close cluster scope of {len(expanded_scope.sequences)} sequences")

            classic_result = apply_classic_gaphack_to_scope(
                expanded_scope.sequences, expanded_scope.headers,
                sequences, headers, distance_provider,
                min_split, max_lump, target_percentile
            )

            # Step 7: Check if classic result differs significantly from input
            original_scope_clusters = {cid: updated_clusters[cid] for cid in expanded_scope.cluster_ids
                                     if cid in updated_clusters}

            if clusters_significantly_different(original_scope_clusters, classic_result,
                                              config.significant_difference_threshold):
                # Step 8: Replace original clusters with classic result
                for cluster_id in expanded_scope.cluster_ids:
                    if cluster_id in updated_clusters:
                        del updated_clusters[cluster_id]

                # Add new clusters from classic result
                for cluster_id, cluster_headers in classic_result.items():
                    updated_clusters[cluster_id] = cluster_headers

                logger.info(f"Refined close cluster component: {len(expanded_scope.cluster_ids)} clusters → "
                           f"{len(classic_result)} clusters")
            else:
                logger.debug(f"Classic gapHACk result not significantly different, keeping original clusters")

        else:
            # Fallback: skip oversized components with warning
            logger.warning(f"Skipping close cluster component with {len(expanded_scope.sequences)} sequences "
                          f"(exceeds limit of {config.max_classic_gaphack_size})")

        # Mark as processed regardless of outcome
        processed_components.add(component_signature)

    # Summary statistics
    original_count = len(all_clusters)
    final_count = len(updated_clusters)
    logger.info(f"Close cluster refinement: {original_count} clusters → {final_count} clusters "
               f"({final_count - original_count:+d})")

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


def verify_cluster_assignments_mece(clusters: Dict[str, List[str]],
                                   original_conflicts: Optional[Dict[str, List[str]]] = None,
                                   context: str = "final") -> Dict[str, any]:
    """Perform comprehensive verification of cluster assignments for MECE property.

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
        - 'mece_property': bool - True if MECE (no conflicts)
        - 'new_conflicts': Dict[str, List[str]] - conflicts not in original set
        - 'unresolved_conflicts': Dict[str, List[str]] - original conflicts still present
        - 'resolved_conflicts': Dict[str, List[str]] - original conflicts that were resolved
    """
    logger.info(f"Performing comprehensive MECE verification ({context})")

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
    mece_property = conflict_count == 0

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
    logger.info(f"MECE Verification Results ({context}):")
    logger.info(f"  Total sequences: {total_sequences}")
    logger.info(f"  Total assignments: {total_assignments}")
    logger.info(f"  Conflicted sequences: {conflict_count}")
    logger.info(f"  MECE property satisfied: {mece_property}")

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
        logger.warning(f"MECE property violated: {conflict_count} sequences in multiple clusters")
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
        if mece_property:
            logger.info(f"✓ FINAL VERIFICATION PASSED: MECE property satisfied - no conflicts detected in final cluster assignments")
        else:
            logger.error(f"✗ FINAL VERIFICATION FAILED: MECE property violated - {conflict_count} conflicts remain in final output")
            logger.error("This indicates conflicts were missed or introduced during processing!")

        # Log summary of what was processed
        logger.info(f"Final verification summary: {total_sequences} sequences across {len(clusters)} clusters")

        # If this is truly the final verification, any conflicts are critical
        if conflict_count > 0:
            logger.error("CRITICAL: Final output contains conflicts - clustering is not MECE!")
            logger.error("This may indicate:")
            logger.error("  1. Conflicts exceeded MAX_CLASSIC_GAPHACK_SIZE and were skipped")
            logger.error("  2. New conflicts were introduced during classic gapHACk reclustering")
            logger.error("  3. Scope expansion was insufficient to capture all related conflicts")
            logger.error("  4. Multi-cluster conflicts were not properly handled")

    # Return comprehensive results
    return {
        'conflicts': current_conflicts,
        'conflict_count': conflict_count,
        'total_sequences': total_sequences,
        'total_assignments': total_assignments,
        'mece_property': mece_property,
        'new_conflicts': new_conflicts,
        'unresolved_conflicts': unresolved_conflicts,
        'resolved_conflicts': resolved_conflicts,
        'verification_context': context,
        'critical_failure': context.startswith("final") and not mece_property
    }