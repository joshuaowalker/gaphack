"""
Target mode clustering for gapHACk.

This module implements single-target clustering that focuses on growing one cluster
from a seed set of target sequences. This is algorithmically simpler than full
gap-optimized clustering since it reduces to complete linkage with gap evaluation.
"""

import logging
import copy
from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np
from tqdm import tqdm

from .core import DistanceCache, GapCalculator
from .lazy_distances import DistanceProvider, DistanceProviderFactory


class TargetModeClustering:
    """
    Target-focused hierarchical agglomerative clustering.
    
    This algorithm grows a single cluster from a seed set of target sequences
    by iteratively merging the closest remaining sequences. The gap calculation
    focuses only on the target cluster vs. all remaining sequences.
    
    This simplifies the optimization problem compared to full gapHACk since
    we only need to track one cluster's growth, making it equivalent to
    complete linkage with gap-based stopping criteria.
    """
    
    def __init__(self, 
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 95,
                 show_progress: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the target mode clustering algorithm.
        
        Args:
            min_split: Minimum distance to split clusters - sequences closer are lumped together
            max_lump: Maximum distance to lump clusters - sequences farther are kept split
            target_percentile: Which percentile to use for gap optimization and linkage decisions
            show_progress: If True, show progress bars during clustering
            logger: Optional logger instance for output; uses default logging if None
        """
        self.min_split = min_split
        self.max_lump = max_lump
        self.target_percentile = target_percentile
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)
        
    def cluster(self, 
               distance_provider: Union[np.ndarray, DistanceProvider], 
               target_indices: List[int],
               sequences: Optional[List[str]] = None) -> Tuple[List[int], List[int], Dict]:
        """
        Perform target mode clustering.
        
        Args:
            distance_provider: Either a distance matrix (np.ndarray) or DistanceProvider instance
            target_indices: List of sequence indices that form the initial seed cluster
            sequences: List of sequences (required if distance_provider is not a DistanceProvider)
        
        Returns:
            Tuple of (target_cluster, remaining_sequences, clustering_history) where:
            - target_cluster is List[int] of indices in the final target cluster
            - remaining_sequences is List[int] of indices not merged into target cluster
            - clustering_history is Dict containing optimization metrics and history
        """
        # Handle backward compatibility and create distance provider
        if isinstance(distance_provider, np.ndarray):
            # Legacy mode: precomputed distance matrix
            distance_matrix = distance_provider
            n = len(distance_matrix)
            provider = DistanceProviderFactory.create_precomputed_provider(distance_matrix)
            self.logger.debug("Using precomputed distance matrix")
        elif isinstance(distance_provider, DistanceProvider):
            # Modern mode: distance provider
            provider = distance_provider
            if hasattr(provider, 'n'):
                n = provider.n
            elif sequences is not None:
                n = len(sequences)
            else:
                # Determine n from target indices (assumes they represent valid range)
                n = max(target_indices) + 1
            self.logger.debug("Using distance provider for on-demand computation")
        else:
            raise ValueError("distance_provider must be either np.ndarray or DistanceProvider instance")
        
        # Validate target indices
        for idx in target_indices:
            if idx < 0 or idx >= n:
                raise ValueError(f"Target index {idx} is out of range for distance matrix of size {n}")
        
        # Initialize target cluster and remaining sequences
        target_cluster = set(target_indices)
        remaining = set(range(n)) - target_cluster
        
        if not remaining:
            self.logger.warning("All sequences are in target set - no clustering needed")
            return list(target_cluster), [], {'best_config': {}, 'gap_history': []}
        
        # Note: we don't need to ensure all pairwise distances in target cluster are precomputed
        # since target clustering uses lazy distance calculation and gets distances on-demand
        
        # Track best configuration
        best_config = {
            'target_cluster': copy.deepcopy(target_cluster),
            'gap_size': -1,
            'merge_distance': 0,
            'gap_percentile': self.target_percentile,
            'gap_metrics': None
        }
        
        # Initialize gap calculation (we'll create DistanceCache on-demand when needed)
        gap_calculator = GapCalculator(self.target_percentile)
        
        # Calculate initial gap
        initial_gap_metrics = self._calculate_target_gap_metrics(
            target_cluster, remaining, provider, gap_calculator
        )
        current_gap = initial_gap_metrics[f'p{self.target_percentile}']['gap_size']
        
        # Update best config with initial state
        if current_gap > best_config['gap_size']:
            best_config = {
                'target_cluster': copy.deepcopy(target_cluster),
                'gap_size': float(current_gap),
                'merge_distance': 0.0,
                'gap_percentile': self.target_percentile,
                'gap_metrics': initial_gap_metrics
            }
        
        self.logger.debug(f"Target mode clustering: initial cluster size {len(target_cluster)}, "
                        f"remaining {len(remaining)} sequences")
        
        gap_history = []
        step = 0
        
        # Create progress bar for merging phase
        pbar = None
        if self.show_progress:
            pbar = tqdm(total=len(remaining), 
                       desc="Target clustering", 
                       unit=" merges")
            pbar.set_postfix({
                "target_size": len(target_cluster),
                "remaining": len(remaining),
                "gap": f"{current_gap:.4f}",
                "best": f"{best_config['gap_size']:.4f}"
            })
        
        # Main clustering loop: grow target cluster by adding closest sequences
        try:
            while remaining:
                # Find closest sequence to target cluster
                closest_seq, closest_distance = self._find_closest_to_target(
                    target_cluster, remaining, provider
                )
                
                # Stop if closest sequence exceeds max threshold
                if closest_distance > self.max_lump:
                    self.logger.debug(f"Stopping: closest distance {closest_distance:.4f} exceeds max_lump {self.max_lump}")
                    break
                
                # Add sequence to target cluster
                target_cluster.add(closest_seq)
                remaining.remove(closest_seq)
                
                # Calculate gap for this configuration
                if remaining:  # Only calculate gap if there are still remaining sequences
                    gap_metrics = self._calculate_target_gap_metrics(
                        target_cluster, remaining, provider, gap_calculator
                    )
                    current_gap = gap_metrics[f'p{self.target_percentile}']['gap_size']
                else:
                    # No remaining sequences - gap is undefined
                    gap_metrics = None
                    current_gap = 0.0
                
                # Record history
                gap_history.append({
                    'step': step,
                    'target_cluster_size': len(target_cluster),
                    'remaining_count': len(remaining),
                    'merge_distance': float(closest_distance),
                    'gap_size': float(current_gap),
                    'gap_exists': bool(current_gap > 0) if gap_metrics else False,
                    'merged_sequence': int(closest_seq)
                })
                
                # Track best configuration
                if current_gap > best_config['gap_size']:
                    best_config = {
                        'target_cluster': copy.deepcopy(target_cluster),
                        'gap_size': float(current_gap),
                        'merge_distance': float(closest_distance),
                        'gap_percentile': self.target_percentile,
                        'gap_metrics': gap_metrics
                    }
                    self.logger.debug(f"New best gap found: {current_gap:.4f} with target cluster size {len(target_cluster)}")
                
                step += 1
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        "target_size": len(target_cluster),
                        "remaining": len(remaining),
                        "gap": f"{current_gap:.4f}",
                        "best": f"{best_config['gap_size']:.4f}"
                    })
        
        finally:
            if pbar:
                pbar.close()
        
        # Report results
        final_target_size = len(best_config['target_cluster'])
        final_remaining = n - final_target_size
        
        self.logger.debug(f"Target clustering complete. Best gap: {best_config['gap_size']:.4f}")
        self.logger.debug(f"Final target cluster: {final_target_size} sequences, "
                        f"remaining: {final_remaining} sequences")
        
        # Convert to lists for return
        final_target_cluster = list(best_config['target_cluster'])
        final_remaining_sequences = [i for i in range(n) if i not in best_config['target_cluster']]
        
        return final_target_cluster, final_remaining_sequences, {
            'best_config': {
                'target_cluster': final_target_cluster,
                'gap_size': float(best_config['gap_size']),
                'merge_distance': float(best_config['merge_distance']),
                'gap_percentile': int(best_config['gap_percentile']),
                'gap_metrics': best_config['gap_metrics']
            },
            'gap_history': gap_history
        }
    
    def _find_closest_to_target(self, target_cluster: Set[int], remaining: Set[int], 
                               distance_provider: DistanceProvider) -> Tuple[int, float]:
        """
        Find the sequence in remaining that is closest to the target cluster.
        Uses complete linkage distance (maximum distance to any member of target cluster).
        
        Args:
            target_cluster: Set of indices in current target cluster
            remaining: Set of indices of remaining sequences
            distance_provider: Provider for distance calculations
            
        Returns:
            Tuple of (closest_sequence_index, distance_to_cluster)
        """
        closest_seq = -1
        min_distance = float('inf')
        
        for seq_idx in remaining:
            # Get distances from this sequence to all target cluster members
            distances_to_cluster = distance_provider.get_distances_from_sequence(seq_idx, target_cluster)
            
            # Calculate complete linkage distance to target cluster
            max_distance_to_cluster = max(distances_to_cluster.values())
            
            # Track minimum of the maximum distances (complete linkage)
            if max_distance_to_cluster < min_distance:
                min_distance = max_distance_to_cluster
                closest_seq = seq_idx
        
        return closest_seq, min_distance
    
    def _calculate_target_gap_metrics(self, target_cluster: Set[int], remaining: Set[int],
                                     distance_provider: DistanceProvider, gap_calculator: GapCalculator) -> Dict:
        """
        Calculate gap metrics for target cluster vs remaining sequences.
        
        Args:
            target_cluster: Set of indices in target cluster
            remaining: Set of indices of remaining sequences
            distance_provider: Provider for distance calculations
            gap_calculator: Gap calculation utility
            
        Returns:
            Dict with gap metrics at different percentiles
        """
        if not remaining:
            # No remaining sequences - gap is undefined
            return {
                f'p{self.target_percentile}': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p100': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p95': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p90': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0}
            }
        
        # Note: we don't need to precompute all distances - get_distance will compute on-demand
        
        # Get intra-cluster distances (within target cluster)
        intra_distances = []
        target_list = list(target_cluster)
        for i, seq1 in enumerate(target_list):
            for j, seq2 in enumerate(target_list):
                if i < j:  # Avoid duplicates and self-distances
                    distance = distance_provider.get_distance(seq1, seq2)
                    intra_distances.append(distance)
        
        # Get inter-cluster distances (target cluster to remaining sequences)  
        inter_distances = []
        for target_seq in target_cluster:
            distances_to_remaining = distance_provider.get_distances_from_sequence(target_seq, remaining)
            inter_distances.extend(distances_to_remaining.values())
        
        # Convert to sorted lists for gap calculation
        sorted_intra = sorted(intra_distances) if intra_distances else []
        sorted_inter = sorted(inter_distances) if inter_distances else []
        
        # Calculate gap metrics at standard percentiles
        if not sorted_intra or not sorted_inter:
            return {
                f'p{self.target_percentile}': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p100': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p95': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p90': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0}
            }
        
        result = {
            'p100': gap_calculator._calculate_single_percentile_gap(sorted_intra, sorted_inter, 100),
            'p95': gap_calculator._calculate_single_percentile_gap(sorted_intra, sorted_inter, 95),
            'p90': gap_calculator._calculate_single_percentile_gap(sorted_intra, sorted_inter, 90)
        }
        
        # Add specific target percentile if not standard
        if self.target_percentile not in [100, 95, 90]:
            result[f'p{self.target_percentile}'] = gap_calculator._calculate_single_percentile_gap(
                sorted_intra, sorted_inter, self.target_percentile
            )
        else:
            result[f'p{self.target_percentile}'] = result[f'p{self.target_percentile}']
        
        return result