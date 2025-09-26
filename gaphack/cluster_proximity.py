"""
Cluster proximity graph infrastructure for principled reclustering.

This module provides abstract interfaces and implementations for efficient
cluster proximity queries needed for scope-limited reclustering algorithms.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

from .lazy_distances import DistanceProvider


logger = logging.getLogger(__name__)


class ClusterProximityGraph(ABC):
    """Abstract interface for cluster proximity queries and close cluster detection."""

    @abstractmethod
    def get_neighbors_within_distance(self, cluster_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Find all clusters within specified distance of given cluster.

        Args:
            cluster_id: ID of query cluster
            max_distance: Maximum distance threshold

        Returns:
            List of (neighbor_cluster_id, distance) tuples sorted by distance
        """
        pass

    @abstractmethod
    def get_k_nearest_neighbors(self, cluster_id: str, k: int) -> List[Tuple[str, float]]:
        """Find K nearest neighbor clusters to given cluster.

        Args:
            cluster_id: ID of query cluster
            k: Number of neighbors to return

        Returns:
            List of (neighbor_cluster_id, distance) tuples sorted by distance
        """
        pass

    @abstractmethod
    def find_close_pairs(self, max_distance: float) -> List[Tuple[str, str, float]]:
        """Find all cluster pairs closer than max_distance.

        Args:
            max_distance: Maximum distance threshold

        Returns:
            List of (cluster1_id, cluster2_id, distance) tuples
        """
        pass

    @abstractmethod
    def update_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Update graph when cluster medoid changes.

        Args:
            cluster_id: ID of cluster to update
            medoid_sequence: New medoid sequence content
            medoid_index: Global index of new medoid sequence
        """
        pass

    @abstractmethod
    def remove_cluster(self, cluster_id: str) -> None:
        """Remove cluster from proximity graph.

        Args:
            cluster_id: ID of cluster to remove
        """
        pass

    @abstractmethod
    def add_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Add new cluster to proximity graph.

        Args:
            cluster_id: ID of new cluster
            medoid_sequence: Medoid sequence content
            medoid_index: Global index of medoid sequence
        """
        pass


class BruteForceProximityGraph(ClusterProximityGraph):
    """Brute force implementation using full pairwise distance computation.

    This implementation computes distances between all cluster medoids on-demand.
    Suitable for smaller datasets where the O(C²) complexity is acceptable.
    """

    def __init__(self, clusters: Dict[str, List[str]], sequences: List[str],
                 headers: List[str], distance_provider: DistanceProvider):
        """Initialize brute force proximity graph.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            sequences: Full sequence list
            headers: Full header list (indices must match sequences)
            distance_provider: Provider for distance calculations
        """
        self.clusters = clusters
        self.sequences = sequences
        self.headers = headers
        self.distance_provider = distance_provider
        self.medoid_cache: Dict[str, int] = {}

        # Compute all medoids upfront
        self._compute_all_medoids()

        logger.info(f"Initialized BruteForceProximityGraph with {len(clusters)} clusters")

    def _compute_all_medoids(self) -> None:
        """Compute medoids for all clusters and cache them."""
        logger.debug(f"Computing medoids for {len(self.clusters)} clusters")

        for cluster_id, cluster_headers in self.clusters.items():
            medoid_idx = self._find_cluster_medoid(cluster_headers)
            self.medoid_cache[cluster_id] = medoid_idx

        logger.debug(f"Computed medoids for all clusters")

    def _find_cluster_medoid(self, cluster_headers: List[str]) -> int:
        """Find medoid (sequence with minimum total distance to all others in cluster).

        Args:
            cluster_headers: List of sequence headers in cluster

        Returns:
            Global index of medoid sequence
        """
        if len(cluster_headers) == 1:
            return self.headers.index(cluster_headers[0])

        cluster_indices = [self.headers.index(h) for h in cluster_headers]

        min_total_distance = float('inf')
        medoid_idx = cluster_indices[0]

        for candidate_idx in cluster_indices:
            total_distance = 0.0
            for other_idx in cluster_indices:
                if candidate_idx != other_idx:
                    distance = self.distance_provider.get_distance(candidate_idx, other_idx)
                    total_distance += distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid_idx = candidate_idx

        return medoid_idx

    def get_neighbors_within_distance(self, cluster_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Find all clusters within specified distance of given cluster."""
        if cluster_id not in self.medoid_cache:
            logger.warning(f"Cluster {cluster_id} not found in medoid cache")
            return []

        target_medoid = self.medoid_cache[cluster_id]
        neighbors = []

        for other_id, other_medoid in self.medoid_cache.items():
            if other_id != cluster_id:
                distance = self.distance_provider.get_distance(target_medoid, other_medoid)
                if distance <= max_distance:
                    neighbors.append((other_id, distance))

        return sorted(neighbors, key=lambda x: x[1])

    def get_k_nearest_neighbors(self, cluster_id: str, k: int) -> List[Tuple[str, float]]:
        """Find K nearest neighbor clusters to given cluster."""
        if cluster_id not in self.medoid_cache:
            logger.warning(f"Cluster {cluster_id} not found in medoid cache")
            return []

        target_medoid = self.medoid_cache[cluster_id]
        distances = []

        for other_id, other_medoid in self.medoid_cache.items():
            if other_id != cluster_id:
                distance = self.distance_provider.get_distance(target_medoid, other_medoid)
                distances.append((other_id, distance))

        # Sort by distance and return top K
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def find_close_pairs(self, max_distance: float) -> List[Tuple[str, str, float]]:
        """Find all cluster pairs closer than max_distance."""
        close_pairs = []
        cluster_ids = list(self.medoid_cache.keys())

        for i, cluster1_id in enumerate(cluster_ids):
            for cluster2_id in cluster_ids[i+1:]:
                medoid1 = self.medoid_cache[cluster1_id]
                medoid2 = self.medoid_cache[cluster2_id]
                distance = self.distance_provider.get_distance(medoid1, medoid2)

                if distance <= max_distance:
                    close_pairs.append((cluster1_id, cluster2_id, distance))

        return close_pairs

    def update_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Update graph when cluster medoid changes."""
        # For brute force, we just update the medoid cache
        # The sequence/medoid_sequence parameter is not used since we recompute from cluster
        if cluster_id in self.clusters:
            new_medoid_idx = self._find_cluster_medoid(self.clusters[cluster_id])
            self.medoid_cache[cluster_id] = new_medoid_idx
            logger.debug(f"Updated medoid for cluster {cluster_id}")
        else:
            logger.warning(f"Cannot update non-existent cluster {cluster_id}")

    def remove_cluster(self, cluster_id: str) -> None:
        """Remove cluster from proximity graph."""
        if cluster_id in self.clusters:
            del self.clusters[cluster_id]
        if cluster_id in self.medoid_cache:
            del self.medoid_cache[cluster_id]
        logger.debug(f"Removed cluster {cluster_id}")

    def add_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Add new cluster to proximity graph."""
        # For brute force, we need to know the cluster composition to compute medoid
        # This is a limitation - we'll need the full cluster data
        if cluster_id in self.clusters:
            medoid_idx = self._find_cluster_medoid(self.clusters[cluster_id])
            self.medoid_cache[cluster_id] = medoid_idx
            logger.debug(f"Added cluster {cluster_id}")
        else:
            logger.warning(f"Cannot add cluster {cluster_id} - cluster data not available")