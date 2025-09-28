"""
Scoped distance providers for scope-limited cluster refinement.

This module provides efficient distance computation for subsets of sequences
used in scope-limited cluster refinement algorithms.
"""

import logging
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

from .lazy_distances import DistanceProvider


logger = logging.getLogger(__name__)


class ScopedDistanceProvider:
    """Efficient distance provider for scope-limited cluster refinement.

    Maps local indices within a scope to global indices in the full dataset,
    enabling classic gapHACk to work on sequence subsets while reusing
    existing distance computations from the global provider.
    """

    def __init__(self, global_provider: DistanceProvider, scope_headers: List[str],
                 all_headers: List[str]):
        """Initialize scoped distance provider.

        Args:
            global_provider: Provider for full dataset distances
            scope_headers: Headers of sequences in the refinement scope
            all_headers: Full header list (indices must match global provider)
        """
        self.global_provider = global_provider
        self.scope_headers = scope_headers
        self.all_headers = all_headers

        # Create mapping from scope indices to global indices
        self.scope_to_global = []
        self.global_to_scope = {}

        for local_idx, header in enumerate(scope_headers):
            try:
                global_idx = all_headers.index(header)
                self.scope_to_global.append(global_idx)
                self.global_to_scope[global_idx] = local_idx
            except ValueError:
                raise ValueError(f"Header '{header}' not found in global header list")

        # Cache for frequently accessed distance computations
        self.local_cache: Dict[Tuple[int, int], float] = {}

        logger.debug(f"Initialized ScopedDistanceProvider with {len(scope_headers)} sequences")

    @property
    def n(self) -> int:
        """Number of sequences in scope."""
        return len(self.scope_headers)

    def get_distance(self, local_i: int, local_j: int) -> float:
        """Get distance between two sequences using local scope indices.

        Args:
            local_i: Local index of first sequence
            local_j: Local index of second sequence

        Returns:
            Distance between sequences
        """
        if local_i < 0 or local_i >= len(self.scope_to_global):
            raise IndexError(f"Local index {local_i} out of range")
        if local_j < 0 or local_j >= len(self.scope_to_global):
            raise IndexError(f"Local index {local_j} out of range")

        # Check cache first
        cache_key = (min(local_i, local_j), max(local_i, local_j))
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # Map to global indices and compute distance
        global_i = self.scope_to_global[local_i]
        global_j = self.scope_to_global[local_j]
        distance = self.global_provider.get_distance(global_i, global_j)

        # Cache result
        self.local_cache[cache_key] = distance
        return distance

    def get_distances_from_sequence(self, local_idx: int, target_local_indices: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to a set of target sequences.

        Args:
            local_idx: Local index of source sequence
            target_local_indices: Set of local indices for target sequences

        Returns:
            Dict mapping target local indices to distances
        """
        distances = {}
        for target_idx in target_local_indices:
            distances[target_idx] = self.get_distance(local_idx, target_idx)
        return distances

    def build_distance_matrix(self) -> np.ndarray:
        """Build full distance matrix for scope (for classic gapHACk integration).

        Returns:
            n x n distance matrix where n is the number of sequences in scope
        """
        n = len(self.scope_to_global)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                distance = self.get_distance(i, j)
                matrix[i, j] = distance
                matrix[j, i] = distance  # Symmetric matrix

        return matrix

    def get_global_index(self, local_idx: int) -> int:
        """Convert local scope index to global index.

        Args:
            local_idx: Local index within scope

        Returns:
            Global index in full dataset
        """
        if local_idx < 0 or local_idx >= len(self.scope_to_global):
            raise IndexError(f"Local index {local_idx} out of range")
        return self.scope_to_global[local_idx]

    def get_local_index(self, global_idx: int) -> Optional[int]:
        """Convert global index to local scope index.

        Args:
            global_idx: Global index in full dataset

        Returns:
            Local index within scope, or None if not in scope
        """
        return self.global_to_scope.get(global_idx)

    def get_scope_headers(self) -> List[str]:
        """Get headers of sequences in scope.

        Returns:
            List of sequence headers in scope order
        """
        return self.scope_headers.copy()

    def clear_cache(self) -> None:
        """Clear local distance cache to free memory."""
        self.local_cache.clear()
        logger.debug("Cleared ScopedDistanceProvider cache")


class PrecomputedScopedDistanceProvider:
    """Alternative scoped provider using precomputed distance matrix.

    For cases where the scope is large enough that precomputing all distances
    is more efficient than on-demand computation.
    """

    def __init__(self, scope_headers: List[str], distance_matrix: np.ndarray):
        """Initialize with precomputed distance matrix.

        Args:
            scope_headers: Headers of sequences in scope
            distance_matrix: Precomputed n x n distance matrix
        """
        if len(scope_headers) != distance_matrix.shape[0]:
            raise ValueError("Header count must match distance matrix size")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")

        self.scope_headers = scope_headers
        self.distance_matrix = distance_matrix
        self.n = len(scope_headers)

        logger.debug(f"Initialized PrecomputedScopedDistanceProvider with {self.n} sequences")

    def get_distance(self, local_i: int, local_j: int) -> float:
        """Get distance between two sequences using local scope indices."""
        if local_i < 0 or local_i >= self.n:
            raise IndexError(f"Local index {local_i} out of range")
        if local_j < 0 or local_j >= self.n:
            raise IndexError(f"Local index {local_j} out of range")

        return self.distance_matrix[local_i, local_j]

    def get_distances_from_sequence(self, local_idx: int, target_local_indices: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to a set of target sequences."""
        distances = {}
        for target_idx in target_local_indices:
            distances[target_idx] = self.get_distance(local_idx, target_idx)
        return distances

    def build_distance_matrix(self) -> np.ndarray:
        """Return the precomputed distance matrix."""
        return self.distance_matrix.copy()

    def get_scope_headers(self) -> List[str]:
        """Get headers of sequences in scope."""
        return self.scope_headers.copy()


def create_scoped_distance_provider(global_provider: DistanceProvider,
                                  scope_headers: List[str],
                                  all_headers: List[str],
                                  precompute_threshold: int = 100) -> ScopedDistanceProvider:
    """Factory function to create appropriate scoped distance provider.

    Args:
        global_provider: Provider for full dataset distances
        scope_headers: Headers of sequences in refinement scope
        all_headers: Full header list
        precompute_threshold: Scope size above which to precompute matrix

    Returns:
        Appropriate ScopedDistanceProvider implementation
    """
    if len(scope_headers) >= precompute_threshold:
        # For larger scopes, precompute the full matrix
        scoped_provider = ScopedDistanceProvider(global_provider, scope_headers, all_headers)
        distance_matrix = scoped_provider.build_distance_matrix()
        scoped_provider.clear_cache()  # Free cache since we have the matrix
        return PrecomputedScopedDistanceProvider(scope_headers, distance_matrix)
    else:
        # For smaller scopes, use on-demand computation
        return ScopedDistanceProvider(global_provider, scope_headers, all_headers)