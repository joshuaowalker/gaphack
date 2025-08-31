"""
Core clustering algorithm for gapHACk.

This module implements the gap-optimized hierarchical agglomerative clustering
algorithm that maximizes the barcode gap between intra-species and inter-species
genetic distances.
"""

import logging
import copy
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations
from functools import lru_cache
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os


class PersistentWorker:
    """
    Persistent worker class that maintains caches across multiple work assignments.
    This reduces initialization overhead compared to recreating caches for each task.
    """
    
    def __init__(self, distance_matrix, min_split, max_lump, target_percentile):
        """Initialize worker with persistent cache and gap calculator."""
        self.distance_matrix = distance_matrix
        self.min_split = min_split
        self.max_lump = max_lump
        self.target_percentile = target_percentile
        
        # Create persistent instances that will be reused
        self.cache = DistanceCache(distance_matrix)
        self.gap_calculator = GapCalculator(target_percentile)
        
        # Track current cluster state
        self.current_clusters = None
    
    def update_clusters(self, clusters_list):
        """Update worker's cluster state and refresh global caches."""
        # Convert clusters from lists to sets
        self.current_clusters = [set(cluster) for cluster in clusters_list]
        
        # Refresh global caches for new clustering state
        self.cache.refresh_global_intra(self.current_clusters)
        self.cache.refresh_global_inter(self.current_clusters)
    
    def evaluate_pairs_range(self, start_i, end_i):
        """Evaluate merge pairs for a specific range of i values."""
        if self.current_clusters is None:
            raise ValueError("Worker clusters not initialized. Call update_clusters first.")
        
        clusters = self.current_clusters
        n_clusters = len(clusters)
        
        best_gap = float('-inf')
        best_merge_i, best_merge_j = -1, -1
        best_merge_distance = -1.0
        processed_count = 0
        
        # Generate pairs on-the-fly for the assigned range
        for i in range(start_i, min(end_i, n_clusters)):
            for j in range(i + 1, n_clusters):
                # Check if merge would exceed max threshold
                merge_distance = self.gap_calculator.calculate_percentile_cluster_distance(
                    clusters[i], clusters[j], self.cache
                )
                
                if merge_distance <= self.max_lump:
                    # Calculate gap incrementally without materializing hypothetical cluster
                    try:
                        gap = self.gap_calculator.calculate_incremental_gap(
                            clusters, i, j, self.cache
                        )
                        
                        # Update best if this gap is better
                        if gap > best_gap:
                            best_gap = gap
                            best_merge_i, best_merge_j = i, j
                            best_merge_distance = merge_distance
                            
                    except Exception:
                        # If gap calculation fails, skip this merge
                        continue
                
                processed_count += 1
        
        return best_gap, best_merge_i, best_merge_j, best_merge_distance, processed_count
    
    def evaluate_pairs_list(self, pair_indices):
        """Evaluate merge pairs for a specific list of (i,j) pair indices."""
        if self.current_clusters is None:
            raise ValueError("Worker clusters not initialized. Call update_clusters first.")
        
        clusters = self.current_clusters
        
        best_gap = float('-inf')
        best_merge_i, best_merge_j = -1, -1
        best_merge_distance = -1.0
        processed_count = 0
        
        # Evaluate assigned pairs
        for i, j in pair_indices:
            # Check if merge would exceed max threshold
            merge_distance = self.gap_calculator.calculate_percentile_cluster_distance(
                clusters[i], clusters[j], self.cache
            )
            
            if merge_distance <= self.max_lump:
                # Calculate gap incrementally without materializing hypothetical cluster
                try:
                    gap = self.gap_calculator.calculate_incremental_gap(
                        clusters, i, j, self.cache
                    )
                    
                    # Update best if this gap is better
                    if gap > best_gap:
                        best_gap = gap
                        best_merge_i, best_merge_j = i, j
                        best_merge_distance = merge_distance
                        
                except Exception:
                    # If gap calculation fails, skip this merge
                    continue
            
            processed_count += 1
        
        return best_gap, best_merge_i, best_merge_j, best_merge_distance, processed_count


# Global worker instance for multiprocessing
_worker_instance = None


def _init_worker(distance_matrix, min_split, max_lump, target_percentile):
    """Initialize persistent worker instance for this process."""
    global _worker_instance
    _worker_instance = PersistentWorker(distance_matrix, min_split, max_lump, target_percentile)


def _evaluate_merge_pairs_multiprocess_worker(args):
    """
    Multiprocessing worker function that uses persistent worker instance.
    Much more efficient than recreating caches for each task.
    Supports both range-based and list-based pair distribution.
    """
    global _worker_instance
    
    if len(args) == 3:
        # Range-based: (start_i, end_i, clusters_list)
        start_i, end_i, clusters_list = args
        _worker_instance.update_clusters(clusters_list)
        return _worker_instance.evaluate_pairs_range(start_i, end_i)
    else:
        # List-based: (pair_indices, clusters_list)
        pair_indices, clusters_list = args
        _worker_instance.update_clusters(clusters_list)
        return _worker_instance.evaluate_pairs_list(pair_indices)


class DistanceCache:
    """Cache for both intra-cluster and inter-cluster distances to avoid recomputation."""
    
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.intra_cache = {}  # frozenset(cluster) -> sorted list of distances
        self.inter_cache = {}  # frozenset([cluster1_key, cluster2_key]) -> sorted list of distances
        
        # Global distance caches for current clustering state
        self.global_intra_distances = None
        self.global_inter_distances = None
        self.global_inter_sorted = None
        
    
    def get_intra_distances(self, cluster: Set[int]) -> tuple:
        """Get sorted intra-cluster distances as immutable tuple."""
        cluster_key = frozenset(cluster)
        
        # Check cache first
        if cluster_key in self.intra_cache:
            return self.intra_cache[cluster_key]
        
        # Calculate distances
        if len(cluster) <= 1:
            calculated_distances = []
        else:
            cluster_list = list(cluster)
            calculated_distances = []
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    calculated_distances.append(self.distance_matrix[cluster_list[i], cluster_list[j]])
            calculated_distances.sort()
        
        # Convert to immutable tuple and cache
        calculated_tuple = tuple(calculated_distances)
        self.intra_cache[cluster_key] = calculated_tuple
        return calculated_tuple
    
    def get_inter_distances(self, cluster1: Set[int], cluster2: Set[int]) -> tuple:
        """Get sorted inter-cluster distances as immutable tuple."""
        # Create a canonical key that's independent of parameter order
        cluster1_key = frozenset(cluster1)
        cluster2_key = frozenset(cluster2)
        pair_key = frozenset([cluster1_key, cluster2_key])
        
        # Check cache first
        if pair_key in self.inter_cache:
            return self.inter_cache[pair_key]
        
        # Calculate distances
        calculated_distances = []
        for i in cluster1:
            for j in cluster2:
                calculated_distances.append(self.distance_matrix[i, j])
        calculated_distances.sort()
        
        # Convert to immutable tuple and cache
        calculated_tuple = tuple(calculated_distances)
        self.inter_cache[pair_key] = calculated_tuple
        return calculated_tuple
    
    def refresh_global_intra(self, clusters: List[Set[int]]):
        """Refresh global intra-cluster distances for current clustering."""
        self.global_intra_distances = []
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_distances = self.get_intra_distances(cluster)
                if cluster_distances:
                    self.global_intra_distances.extend(cluster_distances)
    
    def refresh_global_inter(self, clusters: List[Set[int]]):
        """Refresh global inter-cluster distances for current clustering."""
        inter_distances = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_distances = self.get_inter_distances(clusters[i], clusters[j])
                if pair_distances:
                    inter_distances.extend(pair_distances)
        self.global_inter_sorted = sorted(inter_distances)


class GapCalculator:
    """Handles all gap-related calculations for clustering optimization."""
    
    def __init__(self, target_percentile: int = 95):
        self.target_percentile = target_percentile
    
    def calculate_percentile_cluster_distance(self, cluster1: Set[int], cluster2: Set[int], 
                                             cache: DistanceCache) -> float:
        """Calculate percentile distance for merged cluster without materializing the merged cluster."""
        # Get sorted distances from cache
        distances1 = cache.get_intra_distances(cluster1)
        distances2 = cache.get_intra_distances(cluster2)
        inter_distances = cache.get_inter_distances(cluster1, cluster2)
        
        # Calculate target percentile index for the merged list
        total_distances = len(distances1) + len(distances2) + len(inter_distances)
        if total_distances == 0:
            return 0.0
            
        percentile_idx = int(total_distances * (self.target_percentile / 100.0))
        if percentile_idx >= total_distances:
            percentile_idx = total_distances - 1
        
        # Walk the three sorted lists in parallel until we reach the percentile index
        i1 = i2 = i_inter = 0
        current_idx = 0
        
        while current_idx <= percentile_idx:
            candidates = []
            if i1 < len(distances1):
                candidates.append((distances1[i1], 1))
            if i2 < len(distances2):
                candidates.append((distances2[i2], 2))
            if i_inter < len(inter_distances):
                candidates.append((inter_distances[i_inter], 3))
            
            # Find minimum
            min_val, source = min(candidates)
            
            # If we've reached the target index, return this value
            if current_idx == percentile_idx:
                return min_val
            
            # Advance the appropriate pointer
            if source == 1:
                i1 += 1
            elif source == 2:
                i2 += 1
            else:
                i_inter += 1
            
            current_idx += 1
        
        # Should never reach here, but fallback
        return 0.0
    
    def calculate_incremental_gap(self, clusters: List[Set[int]], merge_i: int, merge_j: int, 
                                  cache: DistanceCache) -> float:
        """
        Calculate gap size for a hypothetical merge without materializing the merged cluster.
        
        Args:
            clusters: Current cluster configuration
            merge_i, merge_j: Indices of clusters to hypothetically merge
            cache: Distance cache (with refreshed global caches)
            
        Returns:
            Gap size if the merge were performed
        """
        # Start with cached global intra-cluster distances
        intra_distances = cache.global_intra_distances[:]
        
        # Add inter-cluster distances between i and j (now become intra-cluster)
        cluster_i = clusters[merge_i]
        cluster_j = clusters[merge_j]
        ij_distances = cache.get_inter_distances(cluster_i, cluster_j)
        if ij_distances:
            intra_distances.extend(ij_distances)
        
        # Get inter-cluster distances by subtracting inter(i,j) from base
        if ij_distances:
            # Sort the distances to subtract
            sorted_ij_distances = sorted(ij_distances)
            
            # Create inter_distances by walking base and skipping ij_distances
            inter_distances = []
            base_idx = 0
            ij_idx = 0
            
            while base_idx < len(cache.global_inter_sorted):
                base_val = cache.global_inter_sorted[base_idx]
                
                # Check if this value should be skipped (it's in ij_distances)
                if ij_idx < len(sorted_ij_distances) and base_val == sorted_ij_distances[ij_idx]:
                    # Skip this value and advance the ij pointer
                    ij_idx += 1
                else:
                    # Keep this value
                    inter_distances.append(base_val)
                
                base_idx += 1
        else:
            # No distances to subtract, use cached global as-is
            inter_distances = cache.global_inter_sorted[:]
        
        # Calculate gap (inter_distances already sorted from subtraction)
        if not intra_distances or not inter_distances:
            return 0.0
        
        sorted_intra = sorted(intra_distances)
        sorted_inter = inter_distances  # Already sorted from the subtraction operation
        
        return self._calculate_percentile_gap(sorted_intra, sorted_inter)
    
    def calculate_gap_for_clustering(self, clusters: List[Set[int]], cache: DistanceCache) -> Dict:
        """
        Calculate barcode gap metrics for a given clustering configuration.
        
        Args:
            clusters: List of cluster sets containing sequence indices
            cache: Distance cache
            
        Returns:
            Dict with gap metrics at different percentiles
        """
        # Collect all distances and sort once
        intra_distances = []
        inter_distances = []

        for cluster in clusters:
            if len(cluster) > 1:
                cluster_distances = cache.get_intra_distances(cluster)
                if cluster_distances:  # Only add non-empty lists
                    intra_distances.extend(cluster_distances)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1, cluster2 = clusters[i], clusters[j]
                pair_distances = cache.get_inter_distances(cluster1, cluster2)
                if pair_distances:  # Only add non-empty lists
                    inter_distances.extend(pair_distances)

        # Sort once outside of _calculate_single_percentile_gap
        sorted_intra_distances = sorted(intra_distances) if intra_distances else []
        sorted_inter_distances = sorted(inter_distances) if inter_distances else []
        
        # If no intra or inter distances, return empty metrics
        if not sorted_intra_distances or not sorted_inter_distances:
            return {
                f'p{self.target_percentile}': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p100': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p95': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0},
                'p90': {'gap_size': 0.0, 'gap_exists': False, 'intra_upper': 0.0, 'inter_lower': 0.0}
            }
        
        # Calculate gap metrics at standard percentiles (pass already-sorted lists)
        result = {
            'p100': self._calculate_single_percentile_gap(sorted_intra_distances, sorted_inter_distances, 100),
            'p95': self._calculate_single_percentile_gap(sorted_intra_distances, sorted_inter_distances, 95),
            'p90': self._calculate_single_percentile_gap(sorted_intra_distances, sorted_inter_distances, 90)
        }
        
        # Add specific target percentile if not standard
        if self.target_percentile not in [100, 95, 90]:
            result[f'p{self.target_percentile}'] = self._calculate_single_percentile_gap(
                sorted_intra_distances, sorted_inter_distances, self.target_percentile
            )
        else:
            result[f'p{self.target_percentile}'] = result[f'p{self.target_percentile}']
        
        return result
    
    def _calculate_percentile_gap(self, sorted_intra: List[float], sorted_inter: List[float]) -> float:
        """Calculate gap at target percentile."""
        gap_data = self._calculate_single_percentile_gap(sorted_intra, sorted_inter, self.target_percentile)
        return gap_data['gap_size']
    
    def _calculate_single_percentile_gap(self, sorted_intra_distances: List[float], 
                                        sorted_inter_distances: List[float],
                                        percentile: int) -> Dict:
        """
        Calculate gap at a specific percentile.
        
        Args:
            sorted_intra_distances: Already-sorted list of intra-cluster distances
            sorted_inter_distances: Already-sorted list of inter-cluster distances
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Dict with gap metrics at the specified percentile
        """
        
        # Calculate percentile boundaries
        # For intra: use upper percentile (e.g., 95th percentile of intra distances)
        # For inter: use lower percentile (e.g., 5th percentile of inter distances for P95 gap)
        intra_percentile = percentile / 100.0
        inter_percentile = 1.0 - (percentile / 100.0)
        
        # Calculate percentile values
        intra_index = (len(sorted_intra_distances) - 1) * intra_percentile
        if intra_index == int(intra_index):
            intra_upper = sorted_intra_distances[int(intra_index)]
        else:
            lower_idx = int(intra_index)
            upper_idx = min(lower_idx + 1, len(sorted_intra_distances) - 1)
            fraction = intra_index - lower_idx
            intra_upper = sorted_intra_distances[lower_idx] + fraction * (sorted_intra_distances[upper_idx] - sorted_intra_distances[lower_idx])
        
        inter_index = (len(sorted_inter_distances) - 1) * inter_percentile
        if inter_index == int(inter_index):
            inter_lower = sorted_inter_distances[int(inter_index)]
        else:
            lower_idx = int(inter_index)
            upper_idx = min(lower_idx + 1, len(sorted_inter_distances) - 1)
            fraction = inter_index - lower_idx
            inter_lower = sorted_inter_distances[lower_idx] + fraction * (sorted_inter_distances[upper_idx] - sorted_inter_distances[lower_idx])
        
        # Calculate gap
        gap_size = inter_lower - intra_upper
        
        return {
            'gap_exists': bool(gap_size > 0),
            'gap_size': float(gap_size),
            'intra_upper': float(intra_upper),
            'inter_lower': float(inter_lower)
        }


class GapOptimizedClustering:
    """
    Gap-optimized hierarchical agglomerative clustering for DNA barcoding.
    
    This algorithm implements a two-phase clustering approach:
    1. Fast greedy merging below a minimum threshold
    2. Gap-optimized merging with real-time threshold optimization
    
    For library usage:
    - Set show_progress=False to disable progress bars in headless environments
    - Pass a custom logger to integrate with your application's logging system
    - Default behavior (show_progress=True, logger=None) is appropriate for CLI usage
    """
    
    def __init__(self, 
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 95,
                 show_progress: bool = True,
                 logger: Optional[logging.Logger] = None,
                 num_threads: Optional[int] = None):
        """
        Initialize the gap-optimized clustering algorithm.
        
        Args:
            min_split: Minimum distance to split clusters - sequences closer are lumped together (default 0.5%)
            max_lump: Maximum distance to lump clusters - sequences farther are kept split (default 2%)
            target_percentile: Which percentile to use for gap optimization and linkage decisions (default 95)
            show_progress: If True, show progress bars during clustering (default True)
            logger: Optional logger instance for output; uses default logging if None
            num_threads: Number of threads for parallel processing (default: auto-detect)
        """
        self.min_split = min_split
        self.max_lump = max_lump
        self.target_percentile = target_percentile
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)
        self.num_threads = num_threads
        
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
        
        # Combined clustering with single progress bar
        step = 0
        initial_clusters = len(clusters)
        gap_history = []

        self.logger.info(f"Running fast clustering up to {self.min_split} min split.")

        # Create single progress bar for entire clustering process (if enabled)
        pbar = None
        if self.show_progress:
            # Total is max possible merges (down to 1 cluster)
            pbar = tqdm(total=initial_clusters - 1, 
                       desc="Clustering", 
                       unit=" merges")

        # Phase 1: Fast merging below minimum threshold
        while len(clusters) > 1:
            min_distance, merge_i, merge_j = self._find_next_merge_candidate(
                clusters, distance_matrix, self.target_percentile
            )
            
            if min_distance >= self.min_split:
                self.logger.debug(f"Gap-optimized clustering: Reached minimum threshold at step {step}")
                break
            
            # Merge without gap checking (assumed intraspecific)
            clusters = self._perform_cluster_merge(clusters, merge_i, merge_j)
            step += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"phase": "fast", "clusters": len(clusters), "dist": f"{min_distance:.4f}"})

        if pbar:
            pbar.close()

        # Initialize distance cache and gap calculator
        cache = DistanceCache(distance_matrix)
        gap_calculator = GapCalculator(self.target_percentile)

        gap_metrics = gap_calculator.calculate_gap_for_clustering(clusters, cache)

        current_gap = gap_metrics[f'p{self.target_percentile}']['gap_size']

        # Phase 2: Gap-aware merging between thresholds
        self.logger.info(f"Running gap-optimized clustering.")

        n = len(clusters)
        expected_pairs = int(n*(n-1)*(n+1)/6)
        if self.show_progress:
            pbar = tqdm(total=expected_pairs,
                       desc="Clustering",
                       unit="steps")
            pbar.set_postfix({"phase": "gap-aware",
                              "clusters": len(clusters),
                              "gap": f"{current_gap:.4f}",
                              "best": f"{best_config['gap_size']:.4f}"})

        # Determine number of threads for the entire gap-aware phase
        max_clusters = len(clusters)
        max_pairs = max_clusters * (max_clusters - 1) // 2
        num_threads = self.num_threads or min(os.cpu_count() or 4, max_pairs, 4)
        
        # Create ProcessPoolExecutor with persistent worker initialization
        with ProcessPoolExecutor(
            max_workers=num_threads,
            initializer=_init_worker,
            initargs=(cache.distance_matrix, self.min_split, self.max_lump, self.target_percentile)
        ) as executor:
            while len(clusters) > 1:
                if pbar:
                    pbar.set_postfix({"phase": "gap-aware",
                                    "clusters": len(clusters),
                                    "gap": f"{current_gap:.4f}",
                                    "best": f"{best_config['gap_size']:.4f}"})

                # Find best merge candidate using gap heuristic
                best_gap, best_merge_i, best_merge_j, best_merge_distance = self._find_best_gap_merge(
                    clusters, pbar, cache, gap_calculator, executor, num_threads
                )
                
                # Stop if no valid merges (all exceed max threshold)
                if best_merge_i == -1:
                    break
                
                # Perform the merge that produces the best gap
                clusters = self._perform_cluster_merge(clusters, best_merge_i, best_merge_j)
                
                # Calculate gap metrics for this configuration
                gap_metrics = gap_calculator.calculate_gap_for_clustering(clusters, cache)
                
                current_gap = gap_metrics[f'p{self.target_percentile}']['gap_size']
                
                gap_history.append({
                    'num_clusters': len(clusters),
                    'merge_distance': float(best_merge_distance),
                    'gap_size': float(current_gap),
                    'gap_exists': bool(gap_metrics[f'p{self.target_percentile}']['gap_exists'])
                })
                
                # Track best configuration encountered so far
                if current_gap > best_config['gap_size']:
                    best_config = {
                        'clusters': copy.deepcopy(clusters),
                        'gap_size': float(current_gap),
                        'merge_distance': float(best_merge_distance),
                        'gap_percentile': self.target_percentile,
                        'gap_metrics': gap_metrics
                    }
                    self.logger.debug(f"New best gap found: {current_gap:.4f} with {len(clusters)} clusters")

        # Close progress bar and report results
        if pbar:
            pbar.close()
            
        # Report why clustering stopped
        if len(clusters) > 1:
            self.logger.info(f"Gap optimization stopped: no more valid merges within max lump threshold")
        else:
            self.logger.debug(f"Gap optimization complete: merged down to single cluster")
        
        # If no gap was ever calculated, use current clustering state
        if best_config['gap_size'] == -1:
            self.logger.warning(f"Gap optimization found no gaps within thresholds. Using current clustering state with {len(clusters)} clusters.")
            best_config['clusters'] = clusters
            best_config['merge_distance'] = self.max_lump
            # Try to calculate a gap for the current configuration
            if len(clusters) > 1:
                try:
                    gap_metrics = gap_calculator.calculate_gap_for_clustering(clusters, cache)
                    best_config['gap_size'] = gap_metrics[f'p{self.target_percentile}']['gap_size']
                    best_config['gap_metrics'] = gap_metrics
                except Exception as e:
                    self.logger.warning(f"Could not calculate gap metrics for final configuration: {e}")
                    best_config['gap_size'] = 0.0
        
        # Convert best configuration to required format, sorted by cluster size
        final_clusters = []
        singletons_set = set()
        
        for cluster_set in best_config['clusters']:
            if len(cluster_set) >= 2:
                final_clusters.append(list(cluster_set))
            else:
                singletons_set.update(cluster_set)
        
        # Sort clusters by size (largest first) for consistent output ordering
        final_clusters.sort(key=len, reverse=True)
        singletons = list(singletons_set)
        
        # Report gap components from the best configuration
        if best_config['gap_metrics']:
            target_metrics = best_config['gap_metrics'][f'p{self.target_percentile}']
            self.logger.info(f"Gap optimization complete. Best gap: {best_config['gap_size']:.4f} "
                        f"(intra≤{target_metrics['intra_upper']:.4f}, inter≥{target_metrics['inter_lower']:.4f})")
        else:
            self.logger.info(f"Gap optimization complete. Best gap: {best_config['gap_size']:.4f}")
        
        # Report cluster size summary
        if final_clusters:
            cluster_sizes = [len(cluster) for cluster in final_clusters]
            self.logger.info(f"{len(final_clusters)} clusters {cluster_sizes} and {len(singletons)} singletons")
        elif singletons:
            self.logger.info(f"No clusters formed, {len(singletons)} singletons")
        
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
    
    def _find_best_gap_merge(self, clusters: List[Set[int]], pbar, cache: DistanceCache,
                             gap_calculator: GapCalculator, executor: ProcessPoolExecutor, 
                             num_threads: int) -> Tuple[float, int, int, float]:
        """
        Find the merge that produces the best gap using gap-based heuristic with parallel evaluation.
        
        Args:
            clusters: List of cluster sets
            pbar: Progress bar instance
            cache: Distance cache (not used in multiprocessing, each process gets its own)
            gap_calculator: Gap calculation instance
            executor: ProcessPoolExecutor for parallel processing
            num_threads: Number of processes to use
            
        Returns:
            Tuple of (best_gap, merge_i, merge_j, merge_distance)
            Returns (-1, -1, -1, -1) if no valid merges exist
        """
        n_clusters = len(clusters)
        
        if n_clusters <= 1:
            return float('-inf'), -1, -1, -1.0
        
        # Calculate total number of pairs for progress tracking
        total_pairs = n_clusters * (n_clusters - 1) // 2
        
        # Convert clusters to lists for serialization to processes
        clusters_list = [list(cluster) for cluster in clusters]
        
        # Generate all pairs and distribute evenly among processes for better load balancing
        all_pairs = [(i, j) for i in range(n_clusters) for j in range(i + 1, n_clusters)]
        
        # Distribute pairs evenly among processes
        pairs_per_process = len(all_pairs) // num_threads
        remainder_pairs = len(all_pairs) % num_threads
        
        process_pair_chunks = []
        start_idx = 0
        for p in range(num_threads):
            # Give remainder pairs to first few processes
            chunk_size = pairs_per_process + (1 if p < remainder_pairs else 0)
            end_idx = start_idx + chunk_size
            if start_idx < len(all_pairs):  # Only add non-empty chunks
                chunk_pairs = all_pairs[start_idx:end_idx]
                process_pair_chunks.append(chunk_pairs)
            start_idx = end_idx
        
        # Execute in parallel using multiprocessing
        best_gap = float('-inf')
        best_merge_i, best_merge_j = -1, -1
        best_merge_distance = -1.0
        total_processed = 0
        
        # Submit all worker tasks with pair-based distribution
        futures = []
        for pair_chunk in process_pair_chunks:
            if pair_chunk:  # Only submit non-empty chunks
                worker_args = (pair_chunk, clusters_list)
                future = executor.submit(_evaluate_merge_pairs_multiprocess_worker, worker_args)
                futures.append(future)
        
        # Collect results from all workers
        for future in futures:
            try:
                worker_gap, worker_i, worker_j, worker_distance, processed_count = future.result()
                total_processed += processed_count
                if worker_gap > best_gap:
                    best_gap = worker_gap
                    best_merge_i, best_merge_j = worker_i, worker_j
                    best_merge_distance = worker_distance
            except Exception as e:
                self.logger.warning(f"Worker process failed: {e}")
        
        # Update progress bar with total processed pairs
        if pbar:
            pbar.update(total_processed)
        
        return best_gap, best_merge_i, best_merge_j, best_merge_distance
    
