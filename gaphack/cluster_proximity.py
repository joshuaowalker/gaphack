"""
Cluster proximity graph infrastructure for cluster refinement.

This module provides abstract interfaces and implementations for efficient
cluster proximity queries needed for scope-limited cluster refinement algorithms.
"""

import logging
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
import numpy as np

from .lazy_distances import DistanceProvider
from .blast_neighborhood import BlastNeighborhoodFinder


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


class BlastKNNProximityGraph(ClusterProximityGraph):
    """BLAST-based K-NN graph for scalable cluster proximity queries.

    This implementation uses BLAST to find K nearest neighbors for each cluster medoid,
    providing O(K) rather than O(C) complexity for proximity queries.
    """

    def __init__(self, clusters: Dict[str, List[str]], sequences: List[str],
                 headers: List[str], distance_provider: DistanceProvider,
                 k_neighbors: int = 20, blast_evalue: float = 1e-5,
                 blast_identity: float = 90.0, cache_dir: Optional[Path] = None):
        """Initialize BLAST K-NN proximity graph.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            sequences: Full sequence list
            headers: Full header list (indices must match sequences)
            distance_provider: Provider for accurate distance calculations
            k_neighbors: Number of nearest neighbors to maintain per cluster
            blast_evalue: BLAST e-value threshold for similarity detection
            blast_identity: Minimum BLAST identity percentage (0-100)
            cache_dir: Directory for BLAST database caching
        """
        self.clusters = clusters
        self.sequences = sequences
        self.headers = headers
        self.distance_provider = distance_provider
        self.k_neighbors = k_neighbors
        self.blast_evalue = blast_evalue
        self.blast_identity = blast_identity
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "gaphack_knn_cache"

        # Initialize data structures
        self.medoid_cache: Dict[str, int] = {}
        self.knn_graph: Dict[str, List[Tuple[str, float]]] = {}
        self.blast_finder: Optional[BlastNeighborhoodFinder] = None

        # Build the K-NN graph
        self._build_knn_graph()

        logger.info(f"Initialized BlastKNNProximityGraph with {len(clusters)} clusters, K={k_neighbors}")

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
                    # Use the same distance calculation as the rest of gapHACk
                    distance = self.distance_provider.get_distance(candidate_idx, other_idx)
                    total_distance += distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid_idx = candidate_idx

        return medoid_idx


    def _build_knn_graph(self) -> None:
        """Build K-NN graph using BLAST on cluster medoid sequences."""
        logger.debug("Building BLAST K-NN graph")

        # Step 1: Compute medoids for all clusters
        self._compute_all_medoids()

        if not self.medoid_cache:
            logger.warning("No clusters found for K-NN graph construction")
            return

        # Step 2: Deduplicate medoid sequences for BLAST database
        # Create mapping from unique sequences to clusters that use them
        unique_sequences = []
        unique_headers = []
        sequence_to_clusters = {}  # Maps sequence content -> list of cluster_ids
        sequence_to_unique_idx = {}  # Maps sequence content -> index in unique arrays

        for cluster_id, medoid_global_idx in self.medoid_cache.items():
            medoid_sequence = self.sequences[medoid_global_idx]

            logger.debug(f"Cluster {cluster_id}: medoid sequence {len(medoid_sequence)} chars, "
                        f"hash={hash(medoid_sequence)}, starts with: {medoid_sequence[:100]}")

            if medoid_sequence not in sequence_to_clusters:
                # First time seeing this sequence - add to unique arrays
                unique_idx = len(unique_sequences)
                unique_sequences.append(medoid_sequence)
                unique_headers.append(f"medoid_{unique_idx}")  # Simple sequential header
                sequence_to_clusters[medoid_sequence] = []
                sequence_to_unique_idx[medoid_sequence] = unique_idx
                logger.debug(f"Added new unique medoid sequence {unique_idx} for cluster {cluster_id}")
            else:
                logger.debug(f"Cluster {cluster_id} uses existing medoid sequence (identical to previous cluster)")

            # Track which clusters use this sequence
            sequence_to_clusters[medoid_sequence].append(cluster_id)

        logger.debug(f"Created medoid sequence set with {len(unique_sequences)} unique sequences for {len(self.medoid_cache)} clusters")

        # Debug: Print the actual medoid sequences and their mappings
        for i, (header, sequence) in enumerate(zip(unique_headers, unique_sequences)):
            clusters = sequence_to_clusters[sequence]
            logger.debug(f"Unique medoid {i}: {header} -> clusters {clusters}, "
                        f"sequence: {sequence[:50]}...{sequence[-20:] if len(sequence) > 70 else ''}")

        # Step 3: Create BLAST database from unique medoid sequences
        self.blast_finder = BlastNeighborhoodFinder(
            sequences=unique_sequences,
            headers=unique_headers,
            cache_dir=self.cache_dir
        )

        # Step 4: Batch query all unique medoid sequences against the database
        # Each unique sequence represents one or more clusters
        batch_queries = [(header, sequence) for header, sequence in zip(unique_headers, unique_sequences)]

        logger.debug(f"Running batch BLAST queries for {len(batch_queries)} unique medoid sequences")

        # Run batched BLAST search
        blast_results = self.blast_finder._get_candidates_for_sequences(
            query_sequences=batch_queries,
            max_targets=self.k_neighbors + 10,  # Get extra hits to account for filtering
            e_value_threshold=self.blast_evalue,
            min_identity=self.blast_identity
        )

        # Step 5: Convert BLAST results to K-NN graph by expanding results to all clusters
        # Initialize empty neighbor lists for all clusters
        for cluster_id in self.medoid_cache.keys():
            self.knn_graph[cluster_id] = []

        # Process BLAST results for each unique medoid sequence
        for query_header in blast_results:
            # Extract unique sequence index from header (format: "medoid_N")
            query_idx = int(query_header.split("_")[1])
            query_sequence = unique_sequences[query_idx]
            query_clusters = sequence_to_clusters[query_sequence]

            logger.debug(f"Processing BLAST results for {query_header}: {len(blast_results[query_header])} hits, "
                        f"represents clusters {query_clusters}")

            # Process all hits for this query
            subject_distances = []
            for hit in blast_results[query_header]:
                # hit.sequence_hash contains the BLAST subject sequence hash
                subject_sequence_hash = hit.sequence_hash

                # Look up the subject header from the BLAST finder's sequence_lookup
                if subject_sequence_hash in self.blast_finder.sequence_lookup:
                    # Get the first match (there could be multiple due to hash collisions)
                    subject_sequence, subject_original_header, subject_blast_idx = self.blast_finder.sequence_lookup[subject_sequence_hash][0]

                    # The subject_original_header is the header we used in our BLAST database (e.g., "medoid_0")
                    # Extract the index to map back to our unique sequences
                    try:
                        subject_idx = int(subject_original_header.split("_")[1])
                        subject_sequence = unique_sequences[subject_idx]
                        subject_clusters = sequence_to_clusters[subject_sequence]
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse subject header: {subject_original_header}")
                        continue
                else:
                    logger.warning(f"Subject sequence hash not found in lookup: {subject_sequence_hash}")
                    continue

                # Skip self-hits (same sequence)
                if subject_sequence == query_sequence:
                    logger.debug(f"Skipping self-hit: {query_header} -> {subject_original_header}")
                    continue

                # Calculate proper distance using DistanceProvider between cluster medoids
                # Get the global indices of the medoids for distance calculation
                query_medoid_global_idx = None
                for query_cluster_id in query_clusters:
                    if query_cluster_id in self.medoid_cache:
                        query_medoid_global_idx = self.medoid_cache[query_cluster_id]
                        break

                if query_medoid_global_idx is None:
                    logger.warning(f"Could not find medoid for query clusters {query_clusters}")
                    continue

                # Add distance to all subject clusters using proper distance calculation
                for subject_cluster_id in subject_clusters:
                    if subject_cluster_id in self.medoid_cache:
                        subject_medoid_global_idx = self.medoid_cache[subject_cluster_id]

                        # Use the same distance calculation as the main clustering algorithm
                        distance = self.distance_provider.get_distance(query_medoid_global_idx, subject_medoid_global_idx)

                        logger.debug(f"BLAST hit: {query_header} -> {subject_original_header}, "
                                   f"blast_identity={hit.blast_identity:.1f}%, proper_distance={distance:.4f}, "
                                   f"subject_cluster={subject_cluster_id}")

                        subject_distances.append((subject_cluster_id, distance))
                    else:
                        logger.warning(f"Could not find medoid for subject cluster {subject_cluster_id}")

            # Handle identical medoids (clusters with same sequence as query)
            for other_cluster_id in query_clusters:
                # For identical medoids, add all other clusters with same medoid at distance 0.0
                for same_medoid_cluster in query_clusters:
                    if same_medoid_cluster != other_cluster_id:
                        subject_distances.append((same_medoid_cluster, 0.0))

            # Assign neighbors to all clusters that use this query sequence
            for query_cluster_id in query_clusters:
                # Get distances to all other clusters (excluding self)
                neighbors = [(subject_id, dist) for subject_id, dist in subject_distances
                           if subject_id != query_cluster_id]

                # Sort by distance and keep K nearest neighbors
                neighbors.sort(key=lambda x: x[1])
                self.knn_graph[query_cluster_id] = neighbors[:self.k_neighbors]

        logger.debug(f"Built K-NN graph with average {np.mean([len(neighbors) for neighbors in self.knn_graph.values()]):.1f} neighbors per cluster")


    def get_neighbors_within_distance(self, cluster_id: str, max_distance: float) -> List[Tuple[str, float]]:
        """Find all clusters within specified distance of given cluster."""
        if cluster_id not in self.knn_graph:
            logger.warning(f"Cluster {cluster_id} not found in K-NN graph")
            return []

        neighbors = []
        for neighbor_id, distance in self.knn_graph[cluster_id]:
            if distance <= max_distance:
                neighbors.append((neighbor_id, distance))
            else:
                # Since neighbors are sorted by distance, we can break early
                break

        return neighbors

    def get_k_nearest_neighbors(self, cluster_id: str, k: int) -> List[Tuple[str, float]]:
        """Find K nearest neighbor clusters to given cluster."""
        if cluster_id not in self.knn_graph:
            logger.warning(f"Cluster {cluster_id} not found in K-NN graph")
            return []

        # Return up to k neighbors (may be fewer if cluster has fewer neighbors)
        return self.knn_graph[cluster_id][:k]

    def find_close_pairs(self, max_distance: float) -> List[Tuple[str, str, float]]:
        """Find all cluster pairs closer than max_distance."""
        close_pairs = []
        processed_pairs = set()

        for cluster_id, neighbors in self.knn_graph.items():
            for neighbor_id, distance in neighbors:
                if distance > max_distance:
                    # Since neighbors are sorted by distance, we can break early
                    break

                # Ensure each pair is only reported once (avoid duplicates)
                pair_key = tuple(sorted([cluster_id, neighbor_id]))
                if pair_key not in processed_pairs:
                    close_pairs.append((cluster_id, neighbor_id, distance))
                    processed_pairs.add(pair_key)

        return close_pairs

    def update_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Update graph when cluster medoid changes."""
        # For now, we rebuild the entire graph when any cluster changes
        # This is simple but not optimal - could be improved with incremental updates
        logger.debug(f"Updating cluster {cluster_id} - rebuilding K-NN graph")

        # Update the cluster in our data structures
        if cluster_id in self.clusters:
            # Recompute medoid
            new_medoid_idx = self._find_cluster_medoid(self.clusters[cluster_id])
            self.medoid_cache[cluster_id] = new_medoid_idx

            # Rebuild the graph
            self._build_knn_graph()
        else:
            logger.warning(f"Cannot update non-existent cluster {cluster_id}")

    def remove_cluster(self, cluster_id: str) -> None:
        """Remove cluster from proximity graph."""
        # Remove from all data structures
        if cluster_id in self.clusters:
            del self.clusters[cluster_id]
        if cluster_id in self.medoid_cache:
            del self.medoid_cache[cluster_id]
        if cluster_id in self.knn_graph:
            del self.knn_graph[cluster_id]

        # Remove this cluster from other clusters' neighbor lists
        for other_cluster_id in self.knn_graph:
            self.knn_graph[other_cluster_id] = [
                (neighbor_id, distance) for neighbor_id, distance in self.knn_graph[other_cluster_id]
                if neighbor_id != cluster_id
            ]

        logger.debug(f"Removed cluster {cluster_id}")

    def add_cluster(self, cluster_id: str, medoid_sequence: str, medoid_index: int) -> None:
        """Add new cluster to proximity graph."""
        # For now, we rebuild the entire graph when any cluster is added
        # This is simple but not optimal - could be improved with incremental updates
        logger.debug(f"Adding cluster {cluster_id} - rebuilding K-NN graph")

        # Add to clusters and compute medoid
        if cluster_id in self.clusters:
            new_medoid_idx = self._find_cluster_medoid(self.clusters[cluster_id])
            self.medoid_cache[cluster_id] = new_medoid_idx

            # Rebuild the graph
            self._build_knn_graph()
        else:
            logger.warning(f"Cannot add cluster {cluster_id} - cluster data not available")