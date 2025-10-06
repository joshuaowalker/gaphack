"""
Cluster proximity graph infrastructure for cluster refinement.

This module provides abstract interfaces and implementations for efficient
cluster proximity queries needed for scope-limited cluster refinement algorithms.
"""

import logging
import tempfile
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .blast_neighborhood import BlastNeighborhoodFinder
from .vsearch_neighborhood import VsearchNeighborhoodFinder
from .neighborhood_finder import NeighborhoodFinder


logger = logging.getLogger(__name__)


class ClusterGraph:
    """BLAST-based K-NN graph for scalable cluster proximity queries.

    This implementation uses BLAST to find K nearest neighbors for each cluster medoid,
    providing O(K) rather than O(C) complexity for proximity queries.
    """

    def __init__(self, clusters: Dict[str, List[str]], sequences: List[str],
                 headers: List[str], k_neighbors: int = 20,
                 blast_evalue: float = 1e-5, blast_identity: float = 90.0,
                 cache_dir: Optional[Path] = None, search_method: str = "blast",
                 show_progress: bool = False):
        """Initialize K-NN proximity graph.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            sequences: Full sequence list
            headers: Full header list (indices must match sequences)
            k_neighbors: Number of nearest neighbors to maintain per cluster
            blast_evalue: BLAST e-value threshold for similarity detection
            blast_identity: Minimum BLAST/vsearch identity percentage (0-100)
            cache_dir: Directory for database caching
            search_method: Search method for K-NN: 'blast' or 'vsearch' (default: 'blast')
            show_progress: Whether to show progress bars (default: False)
        """
        self.clusters = clusters
        self.sequences = sequences
        self.headers = headers
        self.k_neighbors = k_neighbors
        self.blast_evalue = blast_evalue
        self.blast_identity = blast_identity
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "gaphack_knn_cache"
        self.search_method = search_method
        self.show_progress = show_progress

        # Initialize data structures
        self.medoid_cache: Dict[str, int] = {}
        self.knn_graph: Dict[str, List[Tuple[str, float]]] = {}
        self.neighborhood_finder: Optional[NeighborhoodFinder] = None

        # Build the K-NN graph
        self._build_knn_graph()

        logger.info(f"Initialized {search_method.upper()} K-NN ProximityGraph with {len(clusters)} clusters, K={k_neighbors}")

    def _compute_all_medoids(self) -> None:
        """Compute medoids for all clusters and cache them."""
        logger.debug(f"Computing medoids for {len(self.clusters)} clusters")

        cluster_items = list(self.clusters.items())
        iterator = tqdm(cluster_items, desc="Computing cluster medoids", unit="cluster") if self.show_progress else cluster_items

        for cluster_id, cluster_headers in iterator:
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

        # For clusters with multiple sequences, use MSA-based distance calculation
        # This provides more consistent distances by aligning all sequences together
        from .distance_providers import MSACachedDistanceProvider

        cluster_sequences = [self.sequences[idx] for idx in cluster_indices]
        msa_provider = MSACachedDistanceProvider(cluster_sequences, cluster_headers)

        min_total_distance = float('inf')
        medoid_idx = cluster_indices[0]

        # Use local indices (0, 1, 2, ...) for MSA provider
        for local_candidate_idx, global_candidate_idx in enumerate(cluster_indices):
            total_distance = 0.0

            for local_other_idx, global_other_idx in enumerate(cluster_indices):
                if local_candidate_idx != local_other_idx:
                    # Use MSA-based distance calculation
                    distance = msa_provider.get_distance(local_candidate_idx, local_other_idx)
                    total_distance += distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid_idx = global_candidate_idx

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

        # Step 3: Create neighborhood finder database from unique medoid sequences
        if self.search_method == "blast":
            self.neighborhood_finder = BlastNeighborhoodFinder(
                sequences=unique_sequences,
                headers=unique_headers,
                output_dir=self.cache_dir
            )
        elif self.search_method == "vsearch":
            self.neighborhood_finder = VsearchNeighborhoodFinder(
                sequences=unique_sequences,
                headers=unique_headers,
                output_dir=self.cache_dir
            )
        else:
            raise ValueError(f"Unknown search method: {self.search_method}. Choose 'blast' or 'vsearch'.")

        # Step 4: Batch query all unique medoid sequences against the database
        # Each unique sequence represents one or more clusters
        batch_queries = [(header, sequence) for header, sequence in zip(unique_headers, unique_sequences)]

        logger.debug(f"Running batch {self.search_method.upper()} queries for {len(batch_queries)} unique medoid sequences")

        # Run batched search with progress bar
        if self.show_progress:
            with tqdm(total=1, desc=f"Running {self.search_method.upper()} K-NN search", unit="batch") as pbar:
                search_results = self.neighborhood_finder._get_candidates_for_sequences(
                    query_sequences=batch_queries,
                    max_targets=self.k_neighbors + 10,  # Get extra hits to account for filtering
                    e_value_threshold=self.blast_evalue,
                    min_identity=self.blast_identity
                )
                pbar.update(1)
        else:
            search_results = self.neighborhood_finder._get_candidates_for_sequences(
                query_sequences=batch_queries,
                max_targets=self.k_neighbors + 10,  # Get extra hits to account for filtering
                e_value_threshold=self.blast_evalue,
                min_identity=self.blast_identity
            )

        # Step 5: Convert search results to K-NN graph by expanding results to all clusters
        # Initialize empty neighbor lists for all clusters
        for cluster_id in self.medoid_cache.keys():
            self.knn_graph[cluster_id] = []

        # Process search results for each unique medoid sequence
        search_result_items = list(search_results.items())
        result_iterator = tqdm(search_result_items, desc="Building K-NN graph from results", unit="query") if self.show_progress else search_result_items

        for query_header, query_hits in result_iterator:
            # Extract unique sequence index from header (format: "medoid_N")
            query_idx = int(query_header.split("_")[1])
            query_sequence = unique_sequences[query_idx]
            query_clusters = sequence_to_clusters[query_sequence]

            logger.debug(f"Processing search results for {query_header}: {len(query_hits)} hits, "
                        f"represents clusters {query_clusters}")

            # Collect all unique sequences in this query's neighborhood for MSA
            neighborhood_sequences = [query_sequence]
            neighborhood_headers = [query_header]
            neighborhood_unique_indices = [query_idx]

            for hit in query_hits:
                # hit.sequence_hash contains the subject sequence hash
                subject_sequence_hash = hit.sequence_hash

                # Look up the subject header from the finder's sequence_lookup
                if subject_sequence_hash in self.neighborhood_finder.sequence_lookup:
                    # Get the first match (there could be multiple due to hash collisions)
                    subject_sequence, subject_original_header, subject_idx = self.neighborhood_finder.sequence_lookup[subject_sequence_hash][0]

                    # The subject_original_header is the header we used in our search database (e.g., "medoid_0")
                    # Extract the index to map back to our unique sequences
                    try:
                        subject_idx = int(subject_original_header.split("_")[1])
                        subject_sequence = unique_sequences[subject_idx]
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

                # Add to neighborhood for MSA
                neighborhood_sequences.append(subject_sequence)
                neighborhood_headers.append(subject_original_header)
                neighborhood_unique_indices.append(subject_idx)

            # Create MSA-based distance provider for this query's neighborhood
            # This provides consistent alignment across all medoids in the BLAST/vsearch neighborhood
            from .distance_providers import MSACachedDistanceProvider
            logger.debug(f"Creating MSA for {len(neighborhood_sequences)} medoids in {query_header}'s neighborhood")

            msa_provider = MSACachedDistanceProvider(
                neighborhood_sequences,
                neighborhood_headers
            )

            # Calculate distances using MSA-based provider
            subject_distances = []
            for local_subject_idx in range(1, len(neighborhood_sequences)):  # Start at 1 to skip query itself
                subject_unique_idx = neighborhood_unique_indices[local_subject_idx]
                subject_sequence = unique_sequences[subject_unique_idx]
                subject_clusters = sequence_to_clusters[subject_sequence]

                # Calculate distance from query (local index 0) to this subject using MSA
                distance = msa_provider.get_distance(0, local_subject_idx)

                # Map distance to all clusters that use this subject medoid
                for subject_cluster_id in subject_clusters:
                    if subject_cluster_id in self.medoid_cache:
                        logger.debug(f"MSA distance: {query_header} -> medoid_{subject_unique_idx}, "
                                   f"distance={distance:.4f}, subject_cluster={subject_cluster_id}")
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