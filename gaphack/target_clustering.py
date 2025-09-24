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
        
        # Update best config with initial state (prefer larger clusters when gaps are equal)
        if current_gap > best_config['gap_size'] or (current_gap == best_config['gap_size'] and len(target_cluster) > len(best_config['target_cluster'])):
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
                sequence = sequences[closest_seq]
                self.logger.debug(f"Target sequence: {sequence[:20]}..{len(sequence)-40}..{sequence[-20:]}")

                # Stop if closest sequence exceeds max_lump threshold (beyond clustering range)
                if closest_distance > self.max_lump:
                    self.logger.debug(f"Stopping: closest distance {closest_distance:.4f} exceeds max_lump {self.max_lump}")
                    break

                # Determine merge decision based on distance thresholds
                should_merge = False
                merge_reason = ""

                if closest_distance <= self.min_split:
                    # Phase 1: Mandatory merging - always merge sequences within min_split
                    should_merge = True
                    merge_reason = f"mandatory (distance {closest_distance:.4f} <= min_split {self.min_split})"
                else:
                    # Phase 2: Gap-optimized merging - explore all merges up to max_lump, then choose best
                    should_merge = True
                    merge_reason = f"gap-optimized exploration (distance {closest_distance:.4f} between min_split {self.min_split} and max_lump {self.max_lump})"

                self.logger.debug(f"Merge decision for distance {closest_distance:.4f}: {merge_reason}")

                if not should_merge:
                    self.logger.debug(f"Stopping gap-optimized merging - no beneficial merges remaining")
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
                    gap_data = gap_metrics[f'p{self.target_percentile}']
                    self.logger.debug(f"Gap: {current_gap:.4f} (intra_p{self.target_percentile}={gap_data['intra_upper']:.4f}, "
                                    f"inter_p{100-self.target_percentile}={gap_data['inter_lower']:.4f})")
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
                    'merged_sequence': int(closest_seq),
                    'merge_reason': merge_reason
                })
                
                # Track best configuration (prefer larger clusters when gaps are equal)
                update_best = False
                if current_gap > best_config['gap_size']:
                    # Strictly better gap
                    update_best = True
                elif current_gap == best_config['gap_size'] and len(target_cluster) > len(best_config['target_cluster']):
                    # Same gap but larger cluster (tie-breaking)
                    update_best = True

                if update_best:
                    best_config = {
                        'target_cluster': copy.deepcopy(target_cluster),
                        'gap_size': float(current_gap),
                        'merge_distance': float(closest_distance),
                        'gap_percentile': self.target_percentile,
                        'gap_metrics': gap_metrics
                    }
                    self.logger.debug(f"New best configuration: gap={current_gap:.4f}, cluster_size={len(target_cluster)}")
                
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
        Uses 95th percentile linkage distance (consistent with gap calculation percentiles).
        
        Args:
            target_cluster: Set of indices in current target cluster
            remaining: Set of indices of remaining sequences
            distance_provider: Provider for distance calculations
            
        Returns:
            Tuple of (closest_sequence_index, p95_distance_to_cluster)
        """
        closest_seq = -1
        min_distance = float('inf')
        
        for seq_idx in remaining:
            # Get distances from this sequence to all target cluster members
            distances_to_cluster = distance_provider.get_distances_from_sequence(seq_idx, target_cluster)
            
            # Calculate p95 linkage distance to target cluster with outlier detection
            import numpy as np

            # Filter outliers using triangle inequality violation detection
            filtered_distances_dict = self._filter_distance_outliers_triangle_inequality(seq_idx, distances_to_cluster, distance_provider)

            # Use p95 of filtered distances (consistent with gap calculation)
            if filtered_distances_dict:
                filtered_distances = list(filtered_distances_dict.values())
                p95_distance_to_cluster = np.percentile(filtered_distances, 95)
            else:
                # Fallback to original if all distances were filtered (shouldn't happen normally)
                cluster_distances = list(distances_to_cluster.values())
                p95_distance_to_cluster = np.percentile(cluster_distances, 95)

            # Track minimum of the p95 distances (p95 linkage)
            if p95_distance_to_cluster < min_distance:
                min_distance = p95_distance_to_cluster
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

        # Get intra-cluster distances (within target cluster) with outlier filtering
        intra_distances = []
        target_list = list(target_cluster)

        # Filter intra-cluster outliers using triangle inequality
        outlier_pairs = self._filter_intra_cluster_outliers(target_list, distance_provider)
        outlier_pairs_set = set(outlier_pairs) | set((j, i) for i, j in outlier_pairs)  # Both directions

        for i in range(len(target_list)):
            for j in range(i + 1, len(target_list)):
                # Skip pairs identified as triangle inequality violations
                if (target_list[i], target_list[j]) not in outlier_pairs_set:
                    distance = distance_provider.get_distance(target_list[i], target_list[j])
                    intra_distances.append(distance)

        # Get inter-cluster distances (from target cluster to remaining sequences)
        inter_distances = []
        remaining_list = list(remaining)
        for target_idx in target_list:
            for remaining_idx in remaining_list:
                distance = distance_provider.get_distance(target_idx, remaining_idx)
                inter_distances.append(distance)

        # Sort distances for gap calculation
        sorted_intra = sorted(intra_distances) if intra_distances else []
        sorted_inter = sorted(inter_distances) if inter_distances else []

        # Debug: Log distance distributions when inter_p5 might be 0
        if sorted_inter and len(target_cluster) >= 2:
            zero_inter_count = sum(1 for d in sorted_inter if d == 0.0)
            if zero_inter_count > 0:
                self.logger.debug(f"Distance debug: {zero_inter_count}/{len(sorted_inter)} inter-cluster distances = 0.0")
                self.logger.debug(f"Inter distances range: min={min(sorted_inter):.4f}, max={max(sorted_inter):.4f}")
                if len(sorted_intra) > 0:
                    self.logger.debug(f"Intra distances range: min={min(sorted_intra):.4f}, max={max(sorted_intra):.4f}")

# Detailed distance debugging removed - issue identified as complete vs average linkage

        # Handle edge cases where we don't have enough distances
        if not sorted_intra or not sorted_inter:
            # For single sequence clusters or no remaining sequences, return zero gaps
            default_metrics = {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0}
            percentiles = [100, 95, 90, self.target_percentile]
            result = {}
            for p in percentiles:
                result[f'p{p}'] = default_metrics.copy()
        else:
            # Calculate gap metrics at different percentiles
            percentiles = [100, 95, 90, self.target_percentile]
            result = {}

            for p in percentiles:
                gap_metrics = gap_calculator._calculate_single_percentile_gap(sorted_intra, sorted_inter, p)
                result[f'p{p}'] = gap_metrics

        # Ensure target percentile is included if different from standard ones
        if self.target_percentile not in [100, 95, 90]:
            result[f'p{self.target_percentile}'] = result[f'p{self.target_percentile}']
        else:
            result[f'p{self.target_percentile}'] = result[f'p{self.target_percentile}']

        return result

    def _filter_distance_outliers_triangle_inequality(self, candidate_seq: int,
                                                     cluster_distances: Dict[int, float],
                                                     distance_provider: DistanceProvider,
                                                     violation_tolerance: float = 0.05) -> Dict[int, float]:
        """
        Filter alignment failures using triangle inequality violation detection.

        This addresses infix alignment issues where sequences may have spurious
        distances due to poor overlap patterns. Uses triangle inequality to detect
        inconsistent distance measurements rather than statistical outlier detection.

        Args:
            candidate_seq: Index of sequence being evaluated for cluster membership
            cluster_distances: Dict mapping cluster member indices to distances from candidate
            distance_provider: Provider for calculating additional distances as needed
            violation_tolerance: Expected error margin for adjusted identity distances (default 5%)

        Returns:
            Filtered dict of cluster member indices to distances with violations removed
        """
        if len(cluster_distances) < 3:
            # Need at least 3 distances for triangle inequality validation
            return cluster_distances

        cluster_members = list(cluster_distances.keys())
        violations = []

        # Check each distance against triangle inequality constraints
        for i, member_i in enumerate(cluster_members):
            suspect_distance = cluster_distances[member_i]
            violation_count = 0
            total_checks = 0

            # Validate against other cluster members using triangle inequality
            for j, member_j in enumerate(cluster_members):
                if i == j:
                    continue

                # Get required distances for triangle inequality check
                intra_distance = distance_provider.get_distance(member_i, member_j)
                candidate_to_j = cluster_distances[member_j]

                # Triangle inequality: d(candidate, member_i) ≤ d(candidate, member_j) + d(member_j, member_i)
                expected_upper_bound = candidate_to_j + intra_distance

                # Check for violation with tolerance for adjusted identity errors
                if suspect_distance > expected_upper_bound + violation_tolerance:
                    violation_count += 1

                total_checks += 1

            # If majority of triangle inequality checks fail, mark as violation
            if total_checks > 0 and violation_count / total_checks > 0.5:
                violations.append(member_i)

        # Remove violating distances
        filtered_distances = {k: v for k, v in cluster_distances.items()
                             if k not in violations}

        # Log violations for debugging
        if violations:
            violating_distances = [cluster_distances[idx] for idx in violations]
            self.logger.debug(f"Seq {candidate_seq}: removed {len(violations)} triangle inequality violations: "
                            f"{[f'{d:.4f}' for d in violating_distances]} "
                            f"(tolerance={violation_tolerance:.3f})")

        # Safety: keep at least half of distances (if too many violations, likely systematic issue)
        if len(filtered_distances) < len(cluster_distances) * 0.5:
            self.logger.debug(f"Seq {candidate_seq}: triangle inequality filtering too aggressive, keeping original distances")
            return cluster_distances

        return filtered_distances

    def _filter_intra_cluster_outliers(self, cluster_indices: List[int],
                                      distance_provider: DistanceProvider,
                                      violation_tolerance: float = 0.05) -> List[Tuple[int, int]]:
        """
        Filter intra-cluster distance outliers using triangle inequality.

        Args:
            cluster_indices: List of sequence indices in the cluster
            distance_provider: Provider for distance calculations
            violation_tolerance: Expected error margin for adjusted identity distances

        Returns:
            List of (seq1, seq2) pairs to exclude from intra-cluster distance calculations
        """
        if len(cluster_indices) < 4:
            # Need at least 4 sequences for meaningful triangle inequality validation
            return []

        violations = []

        # Check all pairwise distances for triangle inequality violations
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                seq_i, seq_j = cluster_indices[i], cluster_indices[j]
                suspect_distance = distance_provider.get_distance(seq_i, seq_j)

                violation_count = 0
                total_checks = 0

                # Validate against other sequences in cluster
                for k in range(len(cluster_indices)):
                    if k == i or k == j:
                        continue

                    seq_k = cluster_indices[k]
                    dist_i_k = distance_provider.get_distance(seq_i, seq_k)
                    dist_j_k = distance_provider.get_distance(seq_j, seq_k)

                    # Triangle inequality: d(i,j) ≤ d(i,k) + d(k,j)
                    expected_upper_bound = dist_i_k + dist_j_k

                    if suspect_distance > expected_upper_bound + violation_tolerance:
                        violation_count += 1

                    total_checks += 1

                # If majority of checks fail, mark pair as violation
                if total_checks > 0 and violation_count / total_checks > 0.5:
                    violations.append((seq_i, seq_j))

        if violations:
            self.logger.debug(f"Intra-cluster: removed {len(violations)} triangle inequality violations")

        return violations