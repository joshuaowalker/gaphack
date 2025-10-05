"""MSA-based distance calculation for gapHACk clustering."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MSAAlignmentError(Exception):
    """Raised when SPOA multiple sequence alignment fails."""
    pass


class DistanceProvider(ABC):
    """Abstract base class for distance providers."""

    @abstractmethod
    def get_distance(self, seq_idx1: int, seq_idx2: int) -> float:
        """Get distance between two sequences by index."""
        pass

    @abstractmethod
    def get_distances_from_sequence(self, seq_idx: int, target_indices: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to a set of target sequences."""
        pass

    @abstractmethod
    def ensure_distances_computed(self, seq_indices: Set[int]) -> None:
        """Ensure all pairwise distances within a set are computed."""
        pass


class MSACachedDistanceProvider(DistanceProvider):
    """Distance provider that caches MSA and computes distances on-demand.

    This provider runs SPOA once to create a multiple sequence alignment,
    then computes pairwise distances on-demand from the shared alignment space.
    Raises MSAAlignmentError if SPOA fails.
    """

    def __init__(self, sequences: List[str], headers: Optional[List[str]] = None):
        """Initialize with sequences and create MSA.

        Args:
            sequences: List of DNA sequences
            headers: List of sequence headers (for debugging)

        Raises:
            MSAAlignmentError: If SPOA fails to create alignment
        """
        from .utils import run_spoa_msa, replace_terminal_gaps

        self.sequences = sequences
        self.headers = headers if headers is not None else [f"seq_{i}" for i in range(len(sequences))]
        self.n = len(sequences)
        self._distance_cache: Dict[Tuple[int, int], float] = {}

        # Run SPOA once and cache aligned sequences
        logger.debug(f"Creating MSA for {self.n} sequences using SPOA")
        aligned = run_spoa_msa(sequences)

        if aligned is None:
            # SPOA failed - raise exception
            raise MSAAlignmentError(
                f"SPOA failed to create multiple sequence alignment for {self.n} sequences. "
                "This could be due to: empty sequence list, extremely divergent sequences, "
                "SPOA subprocess error, or incomplete alignment output."
            )

        # SPOA succeeded - use MSA-based scoring
        self.aligned_sequences = replace_terminal_gaps(aligned)
        logger.debug(f"MSA created successfully, alignment length: {len(aligned[0])}")

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Get distance between two sequences using cached MSA."""
        from .utils import compute_msa_distance

        if idx1 == idx2:
            return 0.0

        # Check cache
        cache_key = (min(idx1, idx2), max(idx1, idx2))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # Compute distance from MSA
        distance = compute_msa_distance(
            self.aligned_sequences[idx1],
            self.aligned_sequences[idx2]
        )

        # Cache and return
        self._distance_cache[cache_key] = distance
        return distance

    def get_distances_from_sequence(self, idx: int, targets: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to multiple targets."""
        return {target_idx: self.get_distance(idx, target_idx)
                for target_idx in targets}

    def ensure_distances_computed(self, indices: Set[int]) -> None:
        """No-op for MSA provider - all distances available from MSA."""
        pass

    def build_distance_matrix(self) -> np.ndarray:
        """Build full distance matrix from cached MSA."""
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = self.get_distance(i, j)
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix


class PrecomputedDistanceProvider(DistanceProvider):
    """Distance provider that wraps a precomputed distance matrix.

    This provider is used when all pairwise distances have been computed
    in advance (e.g., for testing or when a full distance matrix is available).
    """

    def __init__(self, distance_matrix: np.ndarray):
        """Initialize with precomputed distance matrix.

        Args:
            distance_matrix: Symmetric matrix of pairwise distances
        """
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Get precomputed distance between two sequences."""
        return self.distance_matrix[idx1, idx2]

    def get_distances_from_sequence(self, idx: int, targets: Set[int]) -> Dict[int, float]:
        """Get precomputed distances from one sequence to multiple targets."""
        return {target_idx: self.distance_matrix[idx, target_idx]
                for target_idx in targets}

    def ensure_distances_computed(self, indices: Set[int]) -> None:
        """No-op for precomputed provider - all distances already available."""
        pass

    def build_distance_matrix(self) -> np.ndarray:
        """Return the precomputed distance matrix."""
        return self.distance_matrix


