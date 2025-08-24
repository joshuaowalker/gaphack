"""
Core clustering algorithm for gapHACk.

This module implements the gap-optimized hierarchical agglomerative clustering
algorithm that maximizes the barcode gap between intra-species and inter-species
genetic distances.
"""

import logging
import copy
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


class GapOptimizedClustering:
    """
    Gap-optimized hierarchical agglomerative clustering for DNA barcoding.
    
    This algorithm implements a two-phase clustering approach:
    1. Fast greedy merging below a minimum threshold
    2. Gap-optimized merging with real-time threshold optimization
    """
    
    def __init__(self, 
                 min_threshold: float = 0.005,
                 max_threshold: float = 0.02,
                 target_percentile: int = 95,
                 merge_percentile: int = 95,
                 min_gap_size: float = 0.005):
        """
        Initialize the gap-optimized clustering algorithm.
        
        Args:
            min_threshold: Minimum distance for gap optimization (default 0.5%)
            max_threshold: Maximum distance for cluster merging (default 2%)
            target_percentile: Which percentile gap to optimize (default 95)
            merge_percentile: Which percentile to use for merge decisions (default 95)
            min_gap_size: Minimum gap size to consider "sufficient" (default 0.5%)
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_percentile = target_percentile
        self.merge_percentile = merge_percentile
        self.min_gap_size = min_gap_size
        
    def cluster(self, distance_matrix: np.ndarray) -> Tuple[List[List[int]], List[int], Dict]:
        """
        Perform gap-optimized hierarchical clustering.
        
        Args:
            distance_matrix: Pairwise distance matrix (n x n numpy array)
        
        Returns:
            Tuple of (clusters, singletons, gap_history) where:
            - clusters is List[List[int]] of cluster indices
            - singletons is List[int] of singleton indices
            - gap_history is Dict containing optimization history
        """
        n = len(distance_matrix)
        
        # Initialize each sequence as its own cluster
        clusters = [{i} for i in range(n)]
        
        # Track best configuration
        best_config = {
            'clusters': copy.deepcopy(clusters),
            'gap_size': -1,
            'merge_distance': 0,
            'gap_percentile': self.target_percentile,
            'gap_metrics': None
        }
        
        # Phase 1: Fast merging below minimum threshold
        step = 0
        while len(clusters) > 1:
            min_distance, merge_i, merge_j = self._find_next_merge_candidate(
                clusters, distance_matrix, self.merge_percentile
            )
            
            if min_distance >= self.min_threshold:
                logging.debug(f"Gap-optimized clustering: Reached minimum threshold at step {step}")
                break
            
            # Merge without gap checking (assumed intraspecific)
            clusters = self._perform_cluster_merge(clusters, merge_i, merge_j)
            step += 1
        
        # Phase 2: Gap-aware merging between thresholds
        gap_history = []
        
        while len(clusters) > 1:
            min_distance, merge_i, merge_j = self._find_next_merge_candidate(
                clusters, distance_matrix, self.merge_percentile
            )
            
            # Stop if exceeding maximum threshold
            if min_distance > self.max_threshold:
                logging.info(f"Gap optimization stopped at max threshold: {min_distance:.4f} > {self.max_threshold}")
                break
            
            # Tentatively merge
            new_clusters = self._perform_cluster_merge(copy.deepcopy(clusters), merge_i, merge_j)
            
            # Calculate gap metrics for new configuration
            gap_metrics = self._calculate_gap_for_clustering(
                new_clusters, distance_matrix, self.target_percentile
            )
            
            gap_history.append({
                'num_clusters': len(new_clusters),
                'merge_distance': float(min_distance),
                'gap_size': float(gap_metrics[f'p{self.target_percentile}']['gap_size']),
                'gap_exists': bool(gap_metrics[f'p{self.target_percentile}']['gap_exists'])
            })
            
            # Update best configuration if gap improved
            current_gap = gap_metrics[f'p{self.target_percentile}']['gap_size']
            if current_gap > best_config['gap_size']:
                best_config = {
                    'clusters': copy.deepcopy(new_clusters),
                    'gap_size': float(current_gap),
                    'merge_distance': float(min_distance),
                    'gap_percentile': self.target_percentile,
                    'gap_metrics': gap_metrics
                }
                
                # Early termination if sufficient gap achieved
                if current_gap >= self.min_gap_size:
                    logging.info(f"Sufficient gap found: {current_gap:.4f} at distance {min_distance:.4f}")
                    # Look ahead to see if gap improves further
                    lookahead_improved = False
                    lookahead_clusters = copy.deepcopy(new_clusters)
                    
                    for lookahead_step in range(2):  # Look ahead 2 merges
                        next_dist, next_i, next_j = self._find_next_merge_candidate(
                            lookahead_clusters, distance_matrix, self.merge_percentile
                        )
                        if next_dist > self.max_threshold:
                            break
                        lookahead_clusters = self._perform_cluster_merge(lookahead_clusters, next_i, next_j)
                        lookahead_gap = self._calculate_gap_for_clustering(
                            lookahead_clusters, distance_matrix, self.target_percentile
                        )
                        if lookahead_gap[f'p{self.target_percentile}']['gap_size'] > current_gap * 1.1:  # 10% improvement
                            lookahead_improved = True
                            break
                    
                    if not lookahead_improved:
                        # Gap won't improve significantly, stop here
                        logging.info(f"Gap optimization complete: No significant improvement expected")
                        break
            
            # Check for gap degradation
            if len(gap_history) > 1:
                prev_gap = gap_history[-2]['gap_size']
                if current_gap < prev_gap * 0.8:  # 20% degradation
                    logging.warning(f"Gap degraded from {prev_gap:.4f} to {current_gap:.4f}")
                    # Consider stopping if we had a good gap before
                    if prev_gap >= self.min_gap_size * 0.7:
                        logging.info("Reverting to previous configuration with better gap")
                        break  # Use best_config which has the better gap
            
            clusters = new_clusters
            step += 1
        
        # If no gap was ever calculated, use current clustering state
        if best_config['gap_size'] == -1:
            logging.warning(f"Gap optimization found no gaps within thresholds. Using current clustering state with {len(clusters)} clusters.")
            best_config['clusters'] = clusters
            best_config['merge_distance'] = self.max_threshold
            # Try to calculate a gap for the current configuration
            if len(clusters) > 1:
                try:
                    gap_metrics = self._calculate_gap_for_clustering(
                        clusters, distance_matrix, self.target_percentile
                    )
                    best_config['gap_size'] = gap_metrics[f'p{self.target_percentile}']['gap_size']
                    best_config['gap_metrics'] = gap_metrics
                except Exception as e:
                    logging.warning(f"Could not calculate gap metrics for final configuration: {e}")
                    best_config['gap_size'] = 0.0
        
        # Convert best configuration to required format
        final_clusters = []
        singletons_set = set()
        
        for cluster_set in best_config['clusters']:
            if len(cluster_set) >= 2:
                final_clusters.append(list(cluster_set))
            else:
                singletons_set.update(cluster_set)
        
        singletons = list(singletons_set)
        
        logging.info(f"Gap optimization complete. Best gap: {best_config['gap_size']:.4f} "
                    f"at distance {best_config['merge_distance']:.4f} with {len(final_clusters)} clusters")
        
        # Convert sets to lists for JSON serialization
        json_safe_config = {
            'clusters': [list(cluster) for cluster in best_config['clusters']],
            'gap_size': float(best_config['gap_size']),
            'merge_distance': float(best_config['merge_distance']),
            'gap_percentile': int(best_config['gap_percentile']),
            'gap_metrics': best_config['gap_metrics']
        }
        
        # Return clusters, singletons, and optimization history
        return final_clusters, singletons, {
            'best_config': json_safe_config,
            'gap_history': gap_history,
            'final_gap_metrics': best_config['gap_metrics']
        }
    
    def _find_next_merge_candidate(self, clusters: List[Set[int]], 
                                  distance_matrix: np.ndarray,
                                  percentile: float = 95.0) -> Tuple[float, int, int]:
        """
        Find the next pair of clusters to merge based on percentile complete linkage.
        
        Args:
            clusters: List of cluster sets
            distance_matrix: Pairwise distance matrix
            percentile: Percentile for complete linkage (default 95.0)
            
        Returns:
            Tuple of (min_distance, cluster_i, cluster_j)
        """
        min_distance = float('inf')
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self._percentile_complete_linkage_distance(
                    clusters[i], clusters[j], distance_matrix, percentile
                )
                if dist < min_distance:
                    min_distance = dist
                    merge_i, merge_j = i, j
        
        return float(min_distance), merge_i, merge_j
    
    def _percentile_complete_linkage_distance(self, cluster1: Set[int], cluster2: Set[int], 
                                            distance_matrix: np.ndarray, 
                                            percentile: float = 95.0) -> float:
        """
        Calculate percentile-based complete linkage distance between two clusters.
        Uses 95th percentile instead of maximum to handle alignment failures.
        
        Args:
            cluster1, cluster2: Sets of sequence indices
            distance_matrix: Pairwise distance matrix
            percentile: Percentile to use (95.0 = 95th percentile)
            
        Returns:
            Percentile distance between clusters
        """
        distances = []
        for i in cluster1:
            for j in cluster2:
                distances.append(distance_matrix[i, j])
        
        if not distances:
            return float('inf')
        
        # Calculate percentile
        sorted_distances = sorted(distances)
        percentile_idx = (len(sorted_distances) - 1) * (percentile / 100.0)
        
        if percentile_idx == int(percentile_idx):
            return float(sorted_distances[int(percentile_idx)])
        else:
            lower_idx = int(percentile_idx)
            upper_idx = min(lower_idx + 1, len(sorted_distances) - 1)
            fraction = percentile_idx - lower_idx
            return float(sorted_distances[lower_idx] + fraction * (sorted_distances[upper_idx] - sorted_distances[lower_idx]))
    
    def _perform_cluster_merge(self, clusters: List[Set[int]], merge_i: int, merge_j: int) -> List[Set[int]]:
        """
        Merge two clusters and return the updated cluster list.
        
        Args:
            clusters: List of cluster sets
            merge_i, merge_j: Indices of clusters to merge
            
        Returns:
            Updated cluster list with merged clusters
        """
        # Merge clusters
        new_cluster = clusters[merge_i].union(clusters[merge_j])
        
        # Remove old clusters (remove higher index first to maintain indices)
        if merge_i < merge_j:
            clusters.pop(merge_j)
            clusters.pop(merge_i)
        else:
            clusters.pop(merge_i)
            clusters.pop(merge_j)
        
        # Add merged cluster
        clusters.append(new_cluster)
        
        return clusters
    
    def _calculate_gap_for_clustering(self, clusters: List[Set[int]], 
                                     distance_matrix: np.ndarray,
                                     target_percentile: int = 90) -> Dict:
        """
        Calculate barcode gap metrics for a given clustering configuration.
        
        Args:
            clusters: List of cluster sets containing sequence indices
            distance_matrix: Pairwise distance matrix
            target_percentile: Which percentile to calculate (default 90)
            
        Returns:
            Dict with gap metrics at different percentiles
        """
        # Collect intra-cluster and inter-cluster distances
        intra_distances = []
        inter_distances = []
        
        # Calculate intra-cluster distances
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_list = list(cluster)
                for i in range(len(cluster_list)):
                    for j in range(i + 1, len(cluster_list)):
                        idx1, idx2 = cluster_list[i], cluster_list[j]
                        intra_distances.append(distance_matrix[idx1, idx2])
        
        # Calculate inter-cluster distances
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1, cluster2 = clusters[i], clusters[j]
                for idx1 in cluster1:
                    for idx2 in cluster2:
                        inter_distances.append(distance_matrix[idx1, idx2])
        
        # If no intra or inter distances, return empty metrics
        if not intra_distances or not inter_distances:
            return {
                f'p{target_percentile}': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p100': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p95': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p90': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0}
            }
        
        # Calculate gap metrics at standard percentiles
        result = {
            'p100': self._calculate_single_percentile_gap(intra_distances, inter_distances, 100),
            'p95': self._calculate_single_percentile_gap(intra_distances, inter_distances, 95),
            'p90': self._calculate_single_percentile_gap(intra_distances, inter_distances, 90)
        }
        
        # Add specific target percentile if not standard
        if target_percentile not in [100, 95, 90]:
            result[f'p{target_percentile}'] = self._calculate_single_percentile_gap(
                intra_distances, inter_distances, target_percentile
            )
        else:
            result[f'p{target_percentile}'] = result[f'p{target_percentile}']
        
        return result
    
    def _calculate_single_percentile_gap(self, intra_distances: List[float], 
                                        inter_distances: List[float],
                                        percentile: int) -> Dict:
        """
        Calculate gap at a specific percentile.
        
        Args:
            intra_distances: List of intra-cluster distances
            inter_distances: List of inter-cluster distances
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Dict with gap metrics at the specified percentile
        """
        sorted_intra = sorted(intra_distances)
        sorted_inter = sorted(inter_distances)
        
        # Calculate percentile boundaries
        # For intra: use upper percentile (e.g., 95th percentile of intra distances)
        # For inter: use lower percentile (e.g., 5th percentile of inter distances for P95 gap)
        intra_percentile = percentile / 100.0
        inter_percentile = 1.0 - (percentile / 100.0)
        
        # Calculate percentile values
        intra_index = (len(sorted_intra) - 1) * intra_percentile
        if intra_index == int(intra_index):
            intra_upper = sorted_intra[int(intra_index)]
        else:
            lower_idx = int(intra_index)
            upper_idx = min(lower_idx + 1, len(sorted_intra) - 1)
            fraction = intra_index - lower_idx
            intra_upper = sorted_intra[lower_idx] + fraction * (sorted_intra[upper_idx] - sorted_intra[lower_idx])
        
        inter_index = (len(sorted_inter) - 1) * inter_percentile
        if inter_index == int(inter_index):
            inter_lower = sorted_inter[int(inter_index)]
        else:
            lower_idx = int(inter_index)
            upper_idx = min(lower_idx + 1, len(sorted_inter) - 1)
            fraction = inter_index - lower_idx
            inter_lower = sorted_inter[lower_idx] + fraction * (sorted_inter[upper_idx] - sorted_inter[lower_idx])
        
        # Calculate gap
        gap_size = inter_lower - intra_upper
        
        return {
            'gap_exists': bool(gap_size > 0),
            'gap_size': float(max(0, gap_size)),
            'intra_upper': float(intra_upper),
            'inter_lower': float(inter_lower)
        }