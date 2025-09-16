"""Lazy distance calculation for efficient target mode clustering."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from .utils import calculate_distance_matrix

logger = logging.getLogger(__name__)


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


class PrecomputedDistanceProvider(DistanceProvider):
    """Distance provider using a precomputed full distance matrix."""
    
    def __init__(self, distance_matrix: np.ndarray):
        """Initialize with precomputed distance matrix."""
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
    
    def get_distance(self, seq_idx1: int, seq_idx2: int) -> float:
        """Get distance between two sequences."""
        return float(self.distance_matrix[seq_idx1, seq_idx2])
    
    def get_distances_from_sequence(self, seq_idx: int, target_indices: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to target sequences."""
        return {target_idx: self.get_distance(seq_idx, target_idx) 
                for target_idx in target_indices}
    
    def ensure_distances_computed(self, seq_indices: Set[int]) -> None:
        """All distances already computed - no-op."""
        pass


class LazyDistanceProvider(DistanceProvider):
    """Distance provider that computes distances on-demand."""
    
    def __init__(self, sequences: List[str], 
                 alignment_method: str = "adjusted",
                 end_skip_distance: int = 20,
                 normalize_homopolymers: bool = True,
                 handle_iupac_overlap: bool = True,
                 normalize_indels: bool = True,
                 max_repeat_motif_length: int = 2):
        """Initialize with sequences and distance calculation parameters."""
        self.sequences = sequences
        self.n = len(sequences)
        self.alignment_method = alignment_method
        self.end_skip_distance = end_skip_distance
        self.normalize_homopolymers = normalize_homopolymers
        self.handle_iupac_overlap = handle_iupac_overlap
        self.normalize_indels = normalize_indels
        self.max_repeat_motif_length = max_repeat_motif_length
        
        # Cache for computed distances
        self._distance_cache: Dict[Tuple[int, int], float] = {}
        self._unique_computations = 0
    
    def _compute_pairwise_distances(self, seq_indices: Set[int]) -> np.ndarray:
        """Compute distance matrix for a subset of sequences."""
        subset_sequences = [self.sequences[i] for i in seq_indices]
        
        logger.debug(f"Computing distance matrix for {len(subset_sequences)} sequences")
        
        # Suppress progress bar for small calculations (target mode creates many small ones)
        show_progress = len(subset_sequences) > 50
        
        distance_matrix = calculate_distance_matrix(
            subset_sequences,
            alignment_method=self.alignment_method,
            end_skip_distance=self.end_skip_distance,
            normalize_homopolymers=self.normalize_homopolymers,
            handle_iupac_overlap=self.handle_iupac_overlap,
            normalize_indels=self.normalize_indels,
            max_repeat_motif_length=self.max_repeat_motif_length,
            show_progress=show_progress
        )
        
        # Cache all computed distances
        seq_list = list(seq_indices)
        for i, idx1 in enumerate(seq_list):
            for j, idx2 in enumerate(seq_list):
                if i <= j:  # Cache both directions
                    distance = float(distance_matrix[i, j])
                    self._distance_cache[(idx1, idx2)] = distance
                    self._distance_cache[(idx2, idx1)] = distance
                    if i < j:  # Only count unique pairs
                        self._unique_computations += 1
        
        return distance_matrix
    
    def _compute_single_distance(self, seq_idx1: int, seq_idx2: int) -> float:
        """Compute distance between two specific sequences."""
        try:
            from adjusted_identity import align_and_score, AdjustmentParams
        except ImportError:
            raise ImportError(
                "adjusted-identity package is required. "
                "Install it with: pip install git+https://github.com/joshuaowalker/adjusted-identity.git"
            )
        
        # Create alignment parameters
        if self.alignment_method == "adjusted":
            params = AdjustmentParams(
                end_skip_distance=self.end_skip_distance,
                normalize_homopolymers=self.normalize_homopolymers,
                handle_iupac_overlap=self.handle_iupac_overlap,
                normalize_indels=self.normalize_indels,
                max_repeat_motif_length=self.max_repeat_motif_length
            )
        else:  # traditional
            from adjusted_identity import RAW_ADJUSTMENT_PARAMS
            params = RAW_ADJUSTMENT_PARAMS
        
        # Compute distance between the two sequences
        try:
            # Pass shortest sequence first for consistent infix alignment
            seq1, seq2 = self.sequences[seq_idx1], self.sequences[seq_idx2]
            if len(seq1) <= len(seq2):
                result = align_and_score(seq1, seq2, params)
            else:
                result = align_and_score(seq2, seq1, params)
            distance = 1.0 - result.identity
        except Exception as e:
            logger.warning(f"Alignment failed for sequences {seq_idx1} and {seq_idx2}: {e}")
            distance = 1.0  # Maximum distance for failed alignments
        
        return distance
    
    def _get_cache_key(self, seq_idx1: int, seq_idx2: int) -> Tuple[int, int]:
        """Get canonical cache key for two sequence indices."""
        return (min(seq_idx1, seq_idx2), max(seq_idx1, seq_idx2))
    
    def get_distance(self, seq_idx1: int, seq_idx2: int) -> float:
        """Get distance between two sequences, computing if necessary."""
        if seq_idx1 == seq_idx2:
            return 0.0
        
        cache_key = self._get_cache_key(seq_idx1, seq_idx2)
        
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Need to compute this distance
        logger.debug(f"Computing distance between sequences {seq_idx1} and {seq_idx2}")
        distance = self._compute_single_distance(seq_idx1, seq_idx2)
        self._distance_cache[cache_key] = distance
        self._unique_computations += 1

        return distance
    
    def get_distances_from_sequence(self, seq_idx: int, target_indices: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to target sequences."""
        # Check which distances we need to compute
        missing_indices = set()
        results = {}
        
        for target_idx in target_indices:
            if seq_idx == target_idx:
                results[target_idx] = 0.0
                continue
                
            cache_key = self._get_cache_key(seq_idx, target_idx)
            if cache_key in self._distance_cache:
                results[target_idx] = self._distance_cache[cache_key]
            else:
                missing_indices.add(target_idx)
        
        # Compute missing distances if any - only compute what we need
        if missing_indices:
            logger.debug(f"Computing distances from sequence {seq_idx} to {len(missing_indices)} target sequences")
            
            # Compute only the distances we need, not a full matrix
            for target_idx in missing_indices:
                distance = self._compute_single_distance(seq_idx, target_idx)
                cache_key = self._get_cache_key(seq_idx, target_idx)
                self._distance_cache[cache_key] = distance
                results[target_idx] = distance
                self._unique_computations += 1
        
        return results
    
    def ensure_distances_computed(self, seq_indices: Set[int]) -> None:
        """Ensure all pairwise distances within a set are computed."""
        # Check which distances are missing
        seq_list = list(seq_indices)
        missing_pairs = []
        
        for i, idx1 in enumerate(seq_list):
            for j, idx2 in enumerate(seq_list):
                if i <= j:  # Only check upper triangle
                    cache_key = self._get_cache_key(idx1, idx2)
                    if cache_key not in self._distance_cache:
                        missing_pairs.append((idx1, idx2))
        
        if missing_pairs:
            logger.debug(f"Computing {len(missing_pairs)} missing distances within sequence set of size {len(seq_indices)}")
            self._compute_pairwise_distances(seq_indices)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        return {
            'cached_distances': self._unique_computations,
            'total_computations': self._unique_computations,
            'theoretical_max': (self.n * (self.n - 1)) // 2
        }


class DistanceProviderFactory:
    """Factory for creating distance providers."""
    
    @staticmethod
    def create_lazy_provider(sequences: List[str], **kwargs) -> LazyDistanceProvider:
        """Create a lazy distance provider."""
        return LazyDistanceProvider(sequences, **kwargs)
    
    @staticmethod
    def create_precomputed_provider(distance_matrix: np.ndarray) -> PrecomputedDistanceProvider:
        """Create a precomputed distance provider."""
        return PrecomputedDistanceProvider(distance_matrix)
    
    @staticmethod
    def create_provider(sequences: List[str] = None, 
                       distance_matrix: np.ndarray = None,
                       use_lazy: bool = True,
                       **kwargs) -> DistanceProvider:
        """Create appropriate distance provider based on inputs."""
        if distance_matrix is not None:
            return DistanceProviderFactory.create_precomputed_provider(distance_matrix)
        elif sequences is not None and use_lazy:
            return DistanceProviderFactory.create_lazy_provider(sequences, **kwargs)
        elif sequences is not None and not use_lazy:
            # Fall back to precomputed approach
            logger.warning("Creating full distance matrix - this may be slow for large datasets")
            distance_matrix = calculate_distance_matrix(sequences, **kwargs)
            return DistanceProviderFactory.create_precomputed_provider(distance_matrix)
        else:
            raise ValueError("Either sequences or distance_matrix must be provided")