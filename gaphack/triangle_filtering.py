"""
Triangle inequality-based outlier detection for alignment failure filtering.

This module provides shared triangle inequality filtering logic used across
gapHACk modes to detect and filter spurious distances caused by poor sequence
alignments, particularly for short sequences with minimal overlap.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union
from .lazy_distances import DistanceProvider


# Shared constants for triangle inequality filtering
DEFAULT_VIOLATION_TOLERANCE = 0.05  # 5% tolerance for adjusted identity distance errors
DEFAULT_MIN_VALIDATIONS = 3  # Minimum successful triangle checks before marking distance as good
DEFAULT_ENABLE_FILTERING = True  # Enable by default for consistency across modes

logger = logging.getLogger(__name__)


def filter_distance_dict_triangles(distances: Dict[int, float],
                                 distance_provider: DistanceProvider,
                                 violation_tolerance: float = DEFAULT_VIOLATION_TOLERANCE,
                                 min_validations: int = DEFAULT_MIN_VALIDATIONS,
                                 context: str = "") -> Dict[int, float]:
    """
    Filter alignment failures using triangle inequality with maximum distance heuristic.

    Uses single-violation detection: any triangle inequality violation triggers filtering
    of the largest distance in that triangle. This replaces arbitrary percentage thresholds
    with a more principled approach.

    Args:
        distances: Dict mapping sequence indices to distances (candidate->cluster or intra-cluster)
        distance_provider: Provider for calculating additional distances
        violation_tolerance: Expected error margin for adjusted identity distances
        min_validations: Minimum successful validations before marking distance as good
        context: Logging context string

    Returns:
        Filtered dict with violating distances removed
    """
    if len(distances) < 3:
        return distances

    sequence_indices = list(distances.keys())
    violations = set()
    validated_good = set()  # Distances that have passed enough triangle checks
    min_validations = min(min_validations, len(sequence_indices) - 1)

    # Test each distance for triangle inequality violations
    for seq_i in sequence_indices:
        if seq_i in violations or seq_i in validated_good:
            continue  # Already determined status

        d_i = distances[seq_i]
        validation_count = 0

        # Test against other sequences until we find a violation or enough validations
        for seq_j in sequence_indices:
            if seq_i == seq_j:
                continue

            d_j = distances[seq_j]
            d_ij = distance_provider.get_distance(seq_i, seq_j)

            # Check triangle inequality: d_i <= d_j + d_ij
            if d_i > d_j + d_ij + violation_tolerance:
                # Violation found - mark as bad and stop testing
                violations.add(seq_i)
                break
            else:
                # No violation - count as validation
                validation_count += 1
                if validation_count >= min_validations:
                    # Enough validations - mark as good and stop testing
                    validated_good.add(seq_i)
                    break

    # Remove violating distances
    filtered_distances = {k: v for k, v in distances.items() if k not in violations}

    if violations:
        violating_distances = [distances[idx] for idx in violations]
        logger.debug(f"{context}: removed {len(violations)} triangle inequality violations: "
                    f"{[f'{d:.4f}' for d in violating_distances]} using max-distance heuristic")

    return filtered_distances


def filter_intra_cluster_triangles(cluster_indices: List[int],
                                  distance_provider: DistanceProvider,
                                  violation_tolerance: float = DEFAULT_VIOLATION_TOLERANCE,
                                  min_validations: int = DEFAULT_MIN_VALIDATIONS) -> List[Tuple[int, int]]:
    """
    Filter intra-cluster distance outliers using triangle inequality with max-distance heuristic.

    Args:
        cluster_indices: List of sequence indices in the cluster
        distance_provider: Provider for distance calculations
        violation_tolerance: Expected error margin for adjusted identity distances
        min_validations: Minimum successful validations before marking pair as good

    Returns:
        List of (seq1, seq2) pairs to exclude from intra-cluster distance calculations
    """
    if len(cluster_indices) < 4:
        return []

    violations = set()
    validated_good = set()  # Pairs that have passed enough triangle checks
    min_validations = min(min_validations, len(cluster_indices) - 2)

    # Test each pairwise distance for triangle inequality violations
    for i, seq_i in enumerate(cluster_indices):
        for j, seq_j in enumerate(cluster_indices[i+1:], i+1):
            pair = (seq_i, seq_j)
            if pair in violations or pair in validated_good:
                continue  # Already determined status

            d_ij = distance_provider.get_distance(seq_i, seq_j)
            validation_count = 0

            # Test this pair against other sequences in the cluster
            for k, seq_k in enumerate(cluster_indices):
                if k == i or k == j:
                    continue  # Skip the sequences in the pair we're testing

                d_ik = distance_provider.get_distance(seq_i, seq_k)
                d_jk = distance_provider.get_distance(seq_j, seq_k)

                # Check triangle inequality: d_ij <= d_ik + d_jk
                if d_ij > d_ik + d_jk + violation_tolerance:
                    # Violation found - mark as bad and stop testing this pair
                    violations.add(pair)
                    break
                else:
                    # No violation - count as validation
                    validation_count += 1
                    if validation_count >= min_validations:
                        # Enough validations - mark as good and stop testing
                        validated_good.add(pair)
                        break

    violation_list = list(violations)
    if violation_list:
        logger.debug(f"Intra-cluster: removed {len(violation_list)} triangle inequality violations "
                    f"using max-distance heuristic")

    return violation_list


def filter_distance_matrix_triangles(distance_matrix: np.ndarray,
                                    violation_tolerance: float = DEFAULT_VIOLATION_TOLERANCE,
                                    show_progress: bool = True) -> np.ndarray:
    """
    Apply triangle inequality filtering to entire distance matrix.
    Sets violating distances to NaN using maximum distance heuristic.

    This function performs a single pass over the distance matrix to identify
    and filter alignment failures before clustering begins.

    Args:
        distance_matrix: Pairwise distance matrix (n x n numpy array)
        violation_tolerance: Expected error margin for adjusted identity distances
        show_progress: Show progress bar for large matrices

    Returns:
        Filtered distance matrix with violating distances set to NaN
    """
    n = distance_matrix.shape[0]
    filtered_matrix = distance_matrix.copy()
    violation_count = 0

    # Progress tracking for large matrices
    total_triangles = n * (n - 1) * (n - 2) // 6
    progress_interval = max(1000, total_triangles // 100)  # Update every 1% or 1000 triangles

    if show_progress and total_triangles > 10000:
        from tqdm import tqdm
        pbar = tqdm(total=total_triangles, desc="Triangle filtering", unit=" triangles")
    else:
        pbar = None

    triangle_count = 0

    # Check all triangles in the matrix
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangle_count += 1

                # Get the three distances in this triangle
                d_ij = distance_matrix[i, j]
                d_ik = distance_matrix[i, k]
                d_jk = distance_matrix[j, k]

                # Skip if any distance is already NaN
                if np.isnan(d_ij) or np.isnan(d_ik) or np.isnan(d_jk):
                    continue

                # Check triangle inequalities
                violations = []
                if d_ij > d_ik + d_jk + violation_tolerance:
                    violations.append((i, j, d_ij))
                if d_ik > d_ij + d_jk + violation_tolerance:
                    violations.append((i, k, d_ik))
                if d_jk > d_ij + d_ik + violation_tolerance:
                    violations.append((j, k, d_jk))

                # Apply max-distance heuristic: blame largest violating distance
                if violations:
                    max_violation = max(violations, key=lambda x: x[2])
                    r, c = max_violation[0], max_violation[1]
                    filtered_matrix[r, c] = np.nan
                    filtered_matrix[c, r] = np.nan  # Symmetric matrix
                    violation_count += 1

                # Update progress
                if pbar and triangle_count % progress_interval == 0:
                    pbar.update(progress_interval)

    if pbar:
        # Final update for remaining triangles
        remaining = triangle_count % progress_interval
        if remaining > 0:
            pbar.update(remaining)
        pbar.close()

    if violation_count > 0:
        total_distances = n * (n - 1) // 2
        filter_percentage = 100.0 * violation_count / total_distances
        logger.info(f"Triangle inequality filtering: removed {violation_count} distances "
                   f"({filter_percentage:.1f}%) from {total_distances} total distances")
    else:
        logger.debug("Triangle inequality filtering: no violations found")

    return filtered_matrix


def add_nan_filtering_to_distance_list(distances: List[float]) -> List[float]:
    """
    Filter NaN values from a list of distances.

    Args:
        distances: List of distances that may contain NaN values

    Returns:
        List with NaN values removed
    """
    return [d for d in distances if not np.isnan(d)]