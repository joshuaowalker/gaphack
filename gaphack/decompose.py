"""Decomposition clustering for gaphack using BLAST neighborhoods and target clustering."""

import logging
import copy
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .blast_neighborhood import BlastNeighborhoodFinder
from .target_clustering import TargetModeClustering
from .utils import load_sequences_from_fasta, load_sequences_with_deduplication, calculate_distance_matrix
from .lazy_distances import DistanceProviderFactory, SubsetDistanceProvider, DistanceProvider
from .core import GapCalculator

logger = logging.getLogger(__name__)


@dataclass
class DecomposeResults:
    """Results from decomposition clustering."""
    clusters: Dict[str, List[str]] = field(default_factory=dict)  # cluster_id -> sequence headers (non-conflicted only)
    all_clusters: Dict[str, List[str]] = field(default_factory=dict)  # cluster_id -> sequence headers (including conflicts)
    unassigned: List[str] = field(default_factory=list)  # sequence headers never assigned
    conflicts: Dict[str, List[str]] = field(default_factory=dict)  # seq_id -> cluster_ids (multi-assigned)
    iteration_summaries: List[Dict] = field(default_factory=list)  # per-iteration statistics
    total_iterations: int = 0
    total_sequences_processed: int = 0
    coverage_percentage: float = 0.0


class AssignmentTracker:
    """Tracks sequence assignments and detects conflicts."""
    
    def __init__(self):
        self.assignments: Dict[str, List[Tuple[str, int]]] = {}  # seq_id -> [(cluster_id, iteration), ...]
        self.assigned_sequences: Set[str] = set()
    
    def assign_sequence(self, seq_id: str, cluster_id: str, iteration: int) -> None:
        """Assign a sequence to a cluster."""
        if seq_id not in self.assignments:
            self.assignments[seq_id] = []
        
        self.assignments[seq_id].append((cluster_id, iteration))
        self.assigned_sequences.add(seq_id)
    
    def assign_sequences(self, seq_ids: List[str], cluster_id: str, iteration: int) -> None:
        """Assign multiple sequences to a cluster."""
        for seq_id in seq_ids:
            self.assign_sequence(seq_id, cluster_id, iteration)
    
    def is_assigned(self, seq_id: str) -> bool:
        """Check if sequence has been assigned to any cluster."""
        return seq_id in self.assigned_sequences
    
    def get_conflicts(self) -> Dict[str, List[str]]:
        """Get sequences assigned to multiple clusters."""
        conflicts = {}
        for seq_id, assignments in self.assignments.items():
            if len(assignments) > 1:
                cluster_ids = [cluster_id for cluster_id, _ in assignments]
                conflicts[seq_id] = cluster_ids
        return conflicts
    
    def get_single_assignments(self) -> Dict[str, str]:
        """Get sequences assigned to exactly one cluster."""
        single_assignments = {}
        for seq_id, assignments in self.assignments.items():
            if len(assignments) == 1:
                cluster_id, _ = assignments[0]
                single_assignments[seq_id] = cluster_id
        return single_assignments
    
    def get_all_assignments(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get all assignments including conflicts."""
        return self.assignments.copy()
    
    def get_unassigned(self, all_sequence_ids: List[str]) -> List[str]:
        """Get sequences that were never assigned."""
        return [seq_id for seq_id in all_sequence_ids if not self.is_assigned(seq_id)]


class SupervisedTargetSelector:
    """Target selection strategy for supervised mode using provided target sequences."""
    
    def __init__(self, target_headers: List[str]):
        self.target_headers = target_headers
        self.used_targets: Set[str] = set()
    
    def get_next_target(self, assignment_tracker: AssignmentTracker) -> Optional[List[str]]:
        """Get next target sequence(s) for clustering."""
        # Find first unused target that hasn't been assigned yet
        for target_header in self.target_headers:
            if (target_header not in self.used_targets and 
                not assignment_tracker.is_assigned(target_header)):
                self.used_targets.add(target_header)
                return [target_header]
        
        return None  # No more unassigned targets
    
    def has_more_targets(self, assignment_tracker: AssignmentTracker) -> bool:
        """Check if there are more targets to process."""
        for target_header in self.target_headers:
            if (target_header not in self.used_targets and
                not assignment_tracker.is_assigned(target_header)):
                return True
        return False
    
    def add_blast_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """No-op for supervised mode - doesn't use BLAST memory."""
        pass
    
    def mark_sequences_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """No-op for supervised mode - doesn't need memory management."""
        pass


class BlastResultMemory:
    """Memory pool for storing BLAST neighborhoods for spiral target selection."""
    
    def __init__(self):
        self.unprocessed_neighborhoods: Dict[str, Set[str]] = {}  # target_header -> neighborhood_headers
        self.candidate_pool: Set[str] = set()  # All unassigned sequences from previous neighborhoods
        self.fully_processed_targets: Set[str] = set()
    
    def add_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """Add a BLAST neighborhood to memory."""
        neighborhood_set = set(neighborhood_headers)
        self.unprocessed_neighborhoods[target_header] = neighborhood_set
        self.candidate_pool.update(neighborhood_set)
        logger.debug(f"Added neighborhood for {target_header}: {len(neighborhood_headers)} sequences, "
                    f"total pool: {len(self.candidate_pool)}")
    
    def get_spiral_candidates(self, assignment_tracker: AssignmentTracker) -> List[str]:
        """Get unassigned sequences from BLAST neighborhoods for spiral selection."""
        candidates = []
        for seq_header in self.candidate_pool:
            if not assignment_tracker.is_assigned(seq_header):
                candidates.append(seq_header)
        return candidates
    
    def mark_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """Mark sequences as processed and clean up empty neighborhoods."""
        processed_set = set(processed_headers)
        
        if not allow_overlaps:
            # No overlaps mode: remove processed sequences from candidate pool
            self.candidate_pool -= processed_set
            
            # Clean up neighborhoods that are now fully processed
            targets_to_remove = []
            for target_header, neighborhood in self.unprocessed_neighborhoods.items():
                neighborhood -= processed_set
                if not neighborhood:  # Neighborhood is empty
                    targets_to_remove.append(target_header)
            
            # Remove empty neighborhoods
            for target_header in targets_to_remove:
                del self.unprocessed_neighborhoods[target_header]
                self.fully_processed_targets.add(target_header)
            
            logger.debug(f"Processed {len(processed_headers)} sequences (no overlaps), "
                        f"pool now: {len(self.candidate_pool)}, "
                        f"active neighborhoods: {len(self.unprocessed_neighborhoods)}")
        else:
            # Overlap mode: keep sequences in candidate pool for future clusters
            logger.debug(f"Processed {len(processed_headers)} sequences (overlaps allowed), "
                        f"pool unchanged: {len(self.candidate_pool)}, "
                        f"active neighborhoods: {len(self.unprocessed_neighborhoods)}")


class SpiralTargetSelector:
    """Target selection strategy using spiral exploration with random fallback."""
    
    def __init__(self, all_headers: List[str], max_clusters: Optional[int] = None, 
                 max_sequences: Optional[int] = None):
        self.all_headers = all_headers
        self.max_clusters = max_clusters
        self.max_sequences = max_sequences
        self.iteration_count = 0
        self.blast_memory = BlastResultMemory()
        self.used_targets: Set[str] = set()
        
        # Initialize with random seed if no BLAST history available
        import random
        self.random_state = random.Random(42)  # Deterministic for reproducibility
    
    def get_next_target(self, assignment_tracker: AssignmentTracker) -> Optional[List[str]]:
        """Get next target using spiral logic with random fallback."""
        self.iteration_count += 1
        
        # Try spiral selection first: pick from previous BLAST neighborhoods
        spiral_candidates = self.blast_memory.get_spiral_candidates(assignment_tracker)
        spiral_candidates = [h for h in spiral_candidates if h not in self.used_targets]
        
        target_header = None
        selection_method = ""
        
        if spiral_candidates:
            # Spiral selection: choose from BLAST neighborhood candidates
            target_header = self.random_state.choice(spiral_candidates)
            selection_method = "spiral"
        else:
            # Random fallback: choose any unassigned sequence
            unassigned_candidates = [h for h in self.all_headers 
                                   if (not assignment_tracker.is_assigned(h) and 
                                       h not in self.used_targets)]
            if unassigned_candidates:
                target_header = self.random_state.choice(unassigned_candidates)
                selection_method = "random"
        
        if target_header:
            self.used_targets.add(target_header)
            logger.debug(f"Iteration {self.iteration_count}: selected '{target_header}' via {selection_method} "
                        f"(spiral_pool: {len(spiral_candidates)}, total_unassigned: {len([h for h in self.all_headers if not assignment_tracker.is_assigned(h)])})")
            return [target_header]
        
        return None  # No more targets available
    
    def has_more_targets(self, assignment_tracker: AssignmentTracker) -> bool:
        """Check if there are more targets to process based on stopping criteria."""
        # Check cluster count limit
        if self.max_clusters and self.iteration_count >= self.max_clusters:
            return False
        
        # Check sequence assignment limit
        if self.max_sequences:
            assigned_count = len(assignment_tracker.assigned_sequences)
            if assigned_count >= self.max_sequences:
                return False
        
        # Check if any unassigned sequences remain
        unassigned_candidates = [h for h in self.all_headers 
                               if (not assignment_tracker.is_assigned(h) and 
                                   h not in self.used_targets)]
        return len(unassigned_candidates) > 0
    
    def add_blast_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """Store BLAST neighborhood before pruning for future spiral selection."""
        self.blast_memory.add_neighborhood(target_header, neighborhood_headers)
    
    def mark_sequences_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """Update memory after clustering iteration."""
        self.blast_memory.mark_processed(processed_headers, allow_overlaps)


class DecomposeClustering:
    """Main orchestrator for decomposition clustering."""
    
    def __init__(self,
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 95,
                 blast_max_hits: int = 1000,
                 blast_threads: Optional[int] = None,
                 blast_evalue: float = 1e-5,
                 min_identity: Optional[float] = None,
                 allow_overlaps: bool = True,
                 merge_overlaps: bool = False,
                 containment_threshold: float = 0.8,
                 resolve_conflicts: bool = False,
                 refine_close_clusters: bool = False,
                 proximity_graph: str = 'brute-force',
                 knn_neighbors: int = 20,
                 show_progress: bool = True,
                 logger: Optional[logging.Logger] = None):
        """Initialize decomposition clustering.

        Args:
            min_split: Minimum distance to split clusters in target clustering
            max_lump: Maximum distance to lump clusters in target clustering
            target_percentile: Percentile for gap optimization
            blast_max_hits: Maximum BLAST hits per query (default: 1000)
            blast_threads: BLAST thread count (auto if None)
            blast_evalue: BLAST e-value threshold
            min_identity: BLAST identity threshold (auto if None)
            allow_overlaps: Allow sequences to appear in multiple clusters (default: True)
            merge_overlaps: Enable post-processing to merge overlapping clusters (default: False)
            containment_threshold: Containment coefficient threshold for merging (default: 0.8)
            resolve_conflicts: Enable principled reclustering for conflict resolution (default: False)
            refine_close_clusters: Enable principled reclustering for close cluster refinement (default: False)
            proximity_graph: Proximity graph implementation ('brute-force' or 'blast-knn', default: 'brute-force')
            knn_neighbors: Number of K-nearest neighbors for BLAST K-NN graph (default: 20)
            show_progress: Show progress bars
            logger: Logger instance
        """
        self.min_split = min_split
        self.max_lump = max_lump
        self.target_percentile = target_percentile
        self.blast_max_hits = blast_max_hits
        self.blast_threads = blast_threads
        self.blast_evalue = blast_evalue
        self.min_identity = min_identity
        self.allow_overlaps = allow_overlaps
        self.merge_overlaps = merge_overlaps
        self.containment_threshold = containment_threshold
        self.resolve_conflicts = resolve_conflicts
        self.refine_close_clusters = refine_close_clusters
        self.proximity_graph = proximity_graph
        self.knn_neighbors = knn_neighbors
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)

        # Initialize persistent distance provider (will be created on first use)
        self._global_distance_provider = None

        # Initialize target clustering (disable individual progress bars)
        self.target_clustering = TargetModeClustering(
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile,
            show_progress=False,  # Disable individual progress bars
            logger=self.logger
        )
    
    def decompose(self, input_fasta: str, 
                 targets_fasta: Optional[str] = None,
                 strategy: str = "supervised",
                 max_clusters: Optional[int] = None,
                 max_sequences: Optional[int] = None) -> DecomposeResults:
        """Perform decomposition clustering.
        
        Args:
            input_fasta: Path to input FASTA file with all sequences
            targets_fasta: Path to FASTA file with target sequences (supervised mode)
            strategy: Target selection strategy ("supervised", "random", "spiral")
            max_clusters: Maximum clusters to create (unsupervised modes)
            max_sequences: Maximum sequences to assign (unsupervised modes)
            
        Returns:
            DecomposeResults with clustering results
        """
        self.logger.info(f"Starting decomposition clustering with strategy '{strategy}'")
        
        # Validate inputs early
        if strategy == "supervised":
            if not targets_fasta:
                raise ValueError("targets_fasta is required for supervised mode")
        elif strategy == "unsupervised":
            # No validation needed - will run until input exhausted if no limits specified
            pass
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Valid strategies: 'supervised', 'unsupervised'")
        
        # Load input sequences with deduplication
        sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)
        self.logger.info(f"Loaded {len(sequences)} unique sequences from {input_fasta}")

        # For backward compatibility, create headers list using hash_ids
        headers = hash_ids
        
        # Log overlap policy
        overlap_mode = "allowed" if self.allow_overlaps else "disabled"
        self.logger.info(f"Sequence overlaps: {overlap_mode}")
        
        # Initialize BLAST neighborhood finder
        blast_finder = BlastNeighborhoodFinder(sequences, headers)
        
        # Initialize assignment tracker
        assignment_tracker = AssignmentTracker()
        
        # Initialize target selector based on strategy
        if strategy == "supervised":
            target_sequences, target_headers, _ = load_sequences_from_fasta(targets_fasta)
            self.logger.info(f"Loaded {len(target_headers)} target sequences from targets file")

            # Find matching sequences in input based on sequence content
            self.logger.info(f"Attempting to match {len(target_sequences)} target sequences against {len(sequences)} input sequences")
            matched_hash_ids = self._find_matching_targets_by_content(target_sequences, target_headers, sequences, hash_ids)

            if not matched_hash_ids:
                self.logger.error("No target sequences found matching sequences in input file")
                self.logger.error("Target headers preview:")
                for i, header in enumerate(target_headers[:3]):
                    self.logger.error(f"  Target {i+1}: {header}")
                self.logger.error("Input hash_ids preview:")
                for i, hash_id in enumerate(hash_ids[:3]):
                    self.logger.error(f"  Input {i+1}: {hash_id}")
                raise ValueError("No target sequences found matching sequences in input file")

            self.logger.info(f"Found {len(matched_hash_ids)} target sequences matching input sequences")

            # Target selector now works with hash_ids directly
            target_selector = SupervisedTargetSelector(matched_hash_ids)
        elif strategy == "unsupervised":
            # Create spiral target selector with all sequence headers
            target_selector = SpiralTargetSelector(
                all_headers=headers,
                max_clusters=max_clusters, 
                max_sequences=max_sequences
            )
            self.logger.info(f"Initialized unsupervised mode with spiral target selection: "
                           f"max_clusters={max_clusters}, max_sequences={max_sequences}")
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
        
        # Initialize progress tracking based on strategy and stopping criteria
        if strategy == "supervised":
            # Supervised mode: track target completion
            progress_mode = "targets"
            progress_total = len(matched_hash_ids)
            progress_unit = " targets"
            progress_desc = "Processing targets"
        elif strategy == "unsupervised":
            if max_clusters:
                # Cluster count mode: track cluster creation
                progress_mode = "clusters"
                progress_total = max_clusters
                progress_unit = " clusters"
                progress_desc = "Creating clusters"
            elif max_sequences:
                # Sequence count mode: track sequence assignment
                progress_mode = "sequences"
                progress_total = max_sequences
                progress_unit = " sequences"
                progress_desc = "Assigning sequences"
            else:
                # Exhaustive mode: track sequences until all assigned
                progress_mode = "sequences_exhaustive"
                progress_total = len(headers)
                progress_unit = " sequences"
                progress_desc = "Assigning sequences"
        else:
            # Fallback
            progress_mode = "targets"
            progress_total = float('inf')
            progress_unit = " targets"
            progress_desc = "Processing"
        
        # Statistics tracking for progress bar
        cluster_sizes = []
        gap_sizes = []
        
        # Create overall progress bar
        from tqdm import tqdm
        pbar = tqdm(total=progress_total, desc=progress_desc, unit=progress_unit,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        
        # Main decomposition loop
        iteration = 0
        results = DecomposeResults()
        
        try:
            while target_selector.has_more_targets(assignment_tracker):
                iteration += 1
                # All iteration start messages now at debug level - progress bar shows the info
                self.logger.debug(f"Starting iteration {iteration}")
                # Get next target(s)
                target_headers_for_iteration = target_selector.get_next_target(assignment_tracker)
                if not target_headers_for_iteration:
                    break
                
                self.logger.debug(f"Iteration {iteration}: targeting sequences {target_headers_for_iteration}")
                
                # Find BLAST neighborhood
                neighborhood_headers = blast_finder.find_neighborhood(target_headers_for_iteration,
                                                                      max_hits=self.blast_max_hits,
                                                                      e_value_threshold=self.blast_evalue,
                                                                      min_identity=self.min_identity)

                
                self.logger.debug(f"Found neighborhood of {len(neighborhood_headers)} sequences")
                
                # Store BLAST neighborhood in memory for spiral target selection (before pruning)
                if target_headers_for_iteration:
                    target_selector.add_blast_neighborhood(target_headers_for_iteration[0], neighborhood_headers)
                
                # Filter neighborhood based on overlap policy
                if self.allow_overlaps:
                    # Allow overlaps: use all sequences in neighborhood
                    sequences_for_clustering = neighborhood_headers
                    self.logger.debug(f"Using all {len(sequences_for_clustering)} sequences in neighborhood (overlaps allowed)")
                else:
                    # No overlaps: filter to unassigned sequences only  
                    sequences_for_clustering = [h for h in neighborhood_headers 
                                              if not assignment_tracker.is_assigned(h)]
                    
                    if len(sequences_for_clustering) < len(target_headers_for_iteration):
                        self.logger.debug(f"Some targets already assigned - using {len(sequences_for_clustering)} unassigned sequences")
                    
                    if not sequences_for_clustering:
                        self.logger.debug(f"No unassigned sequences in neighborhood - skipping iteration")
                        continue
                
                # Get target indices in the neighborhood subset
                neighborhood_to_full_idx = {h: headers.index(h) for h in sequences_for_clustering}
                target_indices_in_neighborhood = []
                
                for target_header in target_headers_for_iteration:
                    if target_header in sequences_for_clustering:
                        neighborhood_idx = sequences_for_clustering.index(target_header)
                        target_indices_in_neighborhood.append(neighborhood_idx)
                
                if not target_indices_in_neighborhood:
                    self.logger.debug("No targets found in neighborhood subset - skipping iteration")
                    continue
                
                # Extract sequences for neighborhood
                neighborhood_sequences = [sequences[headers.index(h)] for h in sequences_for_clustering]

                short_count = 0
                long_count = 0
                for sequence in neighborhood_sequences:
                    if len(sequence) < 400:
                        short_count += 1
                    else:
                        long_count += 1

                self.logger.debug(f"Found {short_count} short and {long_count} long sequences")

                # OPTIMIZATION: Prune neighborhood based on distance to targets
                # This reduces computational cost of target clustering by focusing on closest candidates
                pruned_sequences, pruned_headers, pruned_target_indices = self._prune_neighborhood_by_distance(
                    neighborhood_sequences, sequences_for_clustering, target_indices_in_neighborhood, sequences, headers
                )
                
                # Use global distance provider with subset mapping for pruned neighborhood
                self.logger.debug(f"Using global distance provider for {len(pruned_sequences)} pruned neighborhood sequences")
                global_distance_provider = self._get_or_create_distance_provider(sequences)

                # Map pruned headers back to global indices
                pruned_global_indices = [headers.index(h) for h in pruned_headers]

                # Create subset provider that maps pruned indices to global indices
                subset_distance_provider = SubsetDistanceProvider(global_distance_provider, pruned_global_indices)

                # Perform target clustering on pruned neighborhood
                target_cluster_indices, remaining_indices, clustering_metrics = self.target_clustering.cluster(
                    subset_distance_provider, pruned_target_indices, pruned_sequences
                )
                
                # Log distance computation statistics
                if hasattr(global_distance_provider, 'get_cache_stats'):
                    stats = global_distance_provider.get_cache_stats()
                    if stats['theoretical_max'] > 0:
                        coverage_pct = 100.0 * stats['cached_distances'] / stats['theoretical_max']
                        self.logger.debug(f"Distance computation stats: {stats['cached_distances']} computed "
                                       f"out of {stats['theoretical_max']} possible "
                                       f"({coverage_pct:.1f}% coverage)")
                    else:
                        self.logger.debug(f"Distance computation stats: {stats['cached_distances']} computed "
                                       f"(no pairwise distances needed for single sequence)")
                
                # Convert indices back to sequence headers
                target_cluster_headers = [pruned_headers[i] for i in target_cluster_indices]
                remaining_headers = [pruned_headers[i] for i in remaining_indices]
                
                # Generate unique cluster ID
                cluster_id = f"cluster_{iteration:03d}"
                
                # Assign sequences to cluster
                assignment_tracker.assign_sequences(target_cluster_headers, cluster_id, iteration)
                
                # Update BLAST memory for spiral target selection
                target_selector.mark_sequences_processed(target_cluster_headers, self.allow_overlaps)
                
                # Record iteration summary
                iteration_summary = {
                    'iteration': iteration,
                    'target_headers': target_headers_for_iteration,
                    'neighborhood_size': len(neighborhood_headers),
                    'sequences_for_clustering': len(sequences_for_clustering),
                    'pruned_size': len(pruned_sequences),
                    'cluster_size': len(target_cluster_headers),
                    'remaining_size': len(remaining_headers),
                    'cluster_id': cluster_id,
                    'gap_size': clustering_metrics['best_config'].get('gap_size', 0.0)
                }
                
                results.iteration_summaries.append(iteration_summary)
                
                # Update statistics for progress bar
                cluster_sizes.append(len(target_cluster_headers))
                gap_sizes.append(clustering_metrics['best_config'].get('gap_size', 0.0))
                
                # Calculate statistics for progress bar
                median_cluster_size = sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0
                median_gap_size = sorted(gap_sizes)[len(gap_sizes)//2] if gap_sizes else 0
                assigned_total = len(assignment_tracker.assigned_sequences)
                clusters_created = iteration
                
                # Update progress based on mode
                if progress_mode == "targets":
                    # Supervised mode: increment by 1 target processed
                    progress_increment = 1
                    postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                elif progress_mode == "clusters":
                    # Cluster count mode: increment by 1 cluster created
                    progress_increment = 1
                    postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                elif progress_mode in ["sequences", "sequences_exhaustive"]:
                    # Sequence count mode: track unique sequences assigned
                    if self.allow_overlaps:
                        # In overlap mode, count unique sequences assigned (deduplicate)
                        unique_assigned_count = len(assignment_tracker.assigned_sequences)
                        progress_increment = unique_assigned_count - pbar.n  # Only increment by new unique assignments
                    else:
                        # In no-overlap mode, all assignments are unique
                        progress_increment = len(target_cluster_headers)
                    
                    postfix = f"clusters={clusters_created}, med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}"
                else:
                    # Fallback
                    progress_increment = 1
                    postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                
                # Update progress bar
                pbar.set_postfix_str(postfix)
                pbar.update(progress_increment)
                
                # All iteration completion messages at debug level - progress bar shows the main info
                gap_size = clustering_metrics['best_config'].get('gap_size', 0.0)
                self.logger.debug(f"Iteration {iteration} complete: "
                                f"clustered {len(target_cluster_headers)} sequences, "
                                f"gap size: {gap_size:.4f}")
                
                # Check stopping criteria for unsupervised modes
                if max_clusters and iteration >= max_clusters:
                    self.logger.info(f"Reached maximum clusters ({max_clusters}) - stopping")
                    break
                
                if max_sequences:
                    assigned_count = len(assignment_tracker.assigned_sequences)
                    if assigned_count >= max_sequences:
                        self.logger.info(f"Reached maximum sequences ({max_sequences}) - stopping")
                        break
        
        finally:
            # Close progress bar
            pbar.close()
            # Clean up BLAST database
            blast_finder.cleanup()
        
        # Process final results
        results.total_iterations = iteration
        results.total_sequences_processed = len(assignment_tracker.assigned_sequences)
        results.coverage_percentage = (len(assignment_tracker.assigned_sequences) / len(headers)) * 100.0
        
        # Get single assignments (no conflicts) for summary reporting
        single_assignments = assignment_tracker.get_single_assignments()
        
        # Group non-conflicted sequences by cluster
        results.clusters = {}
        for seq_id, cluster_id in single_assignments.items():
            if cluster_id not in results.clusters:
                results.clusters[cluster_id] = []
            results.clusters[cluster_id].append(seq_id)
        
        # Group ALL sequences by cluster (including conflicts) for FASTA generation
        results.all_clusters = {}
        all_assignments = assignment_tracker.get_all_assignments()
        for seq_id, assignments in all_assignments.items():
            for cluster_id, iteration in assignments:
                if cluster_id not in results.all_clusters:
                    results.all_clusters[cluster_id] = []
                results.all_clusters[cluster_id].append(seq_id)
        
        # Get unassigned sequences
        results.unassigned = assignment_tracker.get_unassigned(headers)
        
        # Get conflicts
        results.conflicts = assignment_tracker.get_conflicts()
        
        # Enhanced final summary with aggregate statistics
        total_clustered_seqs = sum(len(cluster) for cluster in results.clusters.values())
        avg_cluster_size = total_clustered_seqs / len(results.clusters) if results.clusters else 0
        large_clusters = sum(1 for cluster in results.clusters.values() if len(cluster) > 10)
        
        self.logger.info(f"Decomposition complete: {len(results.clusters)} clusters created "
                        f"({large_clusters} with >10 sequences), "
                        f"{results.total_sequences_processed} sequences assigned ({results.coverage_percentage:.1f}% coverage), "
                        f"average cluster size: {avg_cluster_size:.1f}")
        
        if results.conflicts:
            self.logger.info(f"Conflicts detected: {len(results.conflicts)}")

        # Apply cluster merging if enabled
        if self.merge_overlaps:
            self.logger.info("Starting cluster overlap detection and merging")
            results = self._merge_overlapping_clusters(results, sequences, headers)

        # Apply principled reclustering for conflict resolution if enabled
        if getattr(self, 'resolve_conflicts', False) and results.conflicts:
            self.logger.info(f"Starting principled reclustering for {len(results.conflicts)} conflicts")
            results = self._resolve_conflicts_via_reclustering(results, sequences, headers)

        # Apply principled reclustering for close cluster refinement if enabled
        if getattr(self, 'refine_close_clusters', False):
            self.logger.info("Starting principled reclustering for close cluster refinement")
            results = self._refine_close_clusters_via_reclustering(results, sequences, headers)

        # Expand hash IDs back to original headers
        results = self._expand_hash_ids_to_headers(results, hash_to_headers)

        return results
    
    def _prune_neighborhood_by_distance(self, neighborhood_sequences: List[str],
                                       neighborhood_headers: List[str],
                                       target_indices: List[int],
                                       global_sequences: List[str],
                                       global_headers: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """Prune neighborhood using principled N+N approach for balanced cluster/context representation.

        Strategy:
        - Take all N sequences within max_lump distance (potential cluster members)
        - Take N additional closest sequences beyond max_lump (context for gap estimation)
        - This ensures complete cluster coverage while providing balanced intra/inter-cluster
          context for robust barcode gap evaluation at target percentiles

        Args:
            neighborhood_sequences: All sequences in neighborhood
            neighborhood_headers: Headers for neighborhood sequences
            target_indices: Indices of target sequences within neighborhood
            global_sequences: Full global sequence list
            global_headers: Full global header list

        Returns:
            Tuple of (pruned_sequences, pruned_headers, new_target_indices)
        """
        self.logger.debug(f"Pruning neighborhood from {len(neighborhood_sequences)} sequences")

        # Calculate distances from all sequences to target sequences
        target_sequences = [neighborhood_sequences[i] for i in target_indices]

        # Use global distance provider with subset mapping for pruning calculation
        global_distance_provider = self._get_or_create_distance_provider(global_sequences)

        # Map neighborhood headers to global indices
        neighborhood_global_indices = [global_headers.index(h) for h in neighborhood_headers]

        # Create subset provider for neighborhood
        neighborhood_distance_provider = SubsetDistanceProvider(global_distance_provider, neighborhood_global_indices)

        # Calculate minimum distance from each sequence to any target
        sequence_min_distances = []
        for seq_idx in range(len(neighborhood_sequences)):
            if seq_idx in target_indices:
                # Target sequences have distance 0 to themselves
                min_dist = 0.0
            else:
                # Find minimum distance to any target
                distances_to_targets = neighborhood_distance_provider.get_distances_from_sequence(
                    seq_idx, set(target_indices)
                )
                min_dist = min(distances_to_targets.values())

            sequence_min_distances.append((seq_idx, min_dist))

        # Sort by distance for principled selection
        sequence_min_distances.sort(key=lambda x: x[1])

        # N+N selection: N sequences within max_lump + N closest sequences beyond max_lump
        within_max_lump = [(idx, dist) for idx, dist in sequence_min_distances if dist <= self.max_lump]
        beyond_max_lump = [(idx, dist) for idx, dist in sequence_min_distances if dist > self.max_lump]

        N = len(within_max_lump)  # Number of potential cluster members

        # Take all sequences within max_lump (complete cluster coverage)
        selected_indices = [idx for idx, _ in within_max_lump]

        # Take N additional closest sequences beyond max_lump for gap estimation context
        # (or all available if fewer than N sequences beyond max_lump)
        context_count = min(N, len(beyond_max_lump))
        context_sequences = beyond_max_lump[:context_count]
        selected_indices.extend([idx for idx, _ in context_sequences])

        self.logger.debug(f"N+N pruning: {N} sequences within max_lump distance {self.max_lump:.4f}")
        self.logger.debug(f"Added {context_count} context sequences beyond max_lump for gap estimation")
        self.logger.debug(f"Total selected: {len(selected_indices)} sequences (N+N = {N}+{context_count})")

        # Build pruned data structures
        pruned_sequences = [neighborhood_sequences[i] for i in selected_indices]
        pruned_headers = [neighborhood_headers[i] for i in selected_indices]

        # Map old target indices to new indices in pruned set
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        pruned_target_indices = [old_to_new_idx[old_idx] for old_idx in target_indices
                               if old_idx in old_to_new_idx]

        if not pruned_target_indices:
            self.logger.error("All target sequences were pruned - this should not happen!")
            # Fallback to original data
            return neighborhood_sequences, neighborhood_headers, target_indices

        self.logger.debug(f"Pruned neighborhood: {len(pruned_sequences)} sequences retained")

        return pruned_sequences, pruned_headers, pruned_target_indices
    
    def _find_matching_targets(self, target_sequences: List[str], target_headers: List[str], 
                              input_sequences: List[str], input_headers: List[str]) -> List[Dict[str, str]]:
        """Find target sequences that match input sequences by sequence content.
        
        Args:
            target_sequences: List of target sequences
            target_headers: List of target headers
            input_sequences: List of input sequences
            input_headers: List of input headers
            
        Returns:
            List of dicts with keys: 'target_header', 'input_header', 'target_sequence'
        """
        matched_targets = []
        
        # Create a mapping of normalized input sequences to headers for fast lookup
        input_seq_to_header = {}
        for seq, header in zip(input_sequences, input_headers):
            normalized_seq = seq.upper().strip()
            if normalized_seq in input_seq_to_header:
                # Handle duplicate sequences - keep first occurrence
                self.logger.debug(f"Duplicate sequence found in input, keeping first occurrence: {input_seq_to_header[normalized_seq]}")
            else:
                input_seq_to_header[normalized_seq] = header
        
        # Find matches
        for i, (target_seq, target_header) in enumerate(zip(target_sequences, target_headers)):
            normalized_target = target_seq.upper().strip()

            if normalized_target in input_seq_to_header:
                matched_targets.append({
                    'target_header': target_header,
                    'input_header': input_seq_to_header[normalized_target],
                    'target_sequence': target_seq
                })
                self.logger.info(f"MATCH {i+1}: Target '{target_header}' -> Input '{input_seq_to_header[normalized_target]}'")
            else:
                self.logger.warning(f"NO MATCH {i+1}: Target '{target_header}' (length: {len(target_seq)}) not found in input sequences")
        
        return matched_targets

    def _find_matching_targets_by_content(self, target_sequences: List[str], target_headers: List[str],
                                        input_sequences: List[str], input_hash_ids: List[str]) -> List[str]:
        """Find target sequences that match input sequences by content, return matching hash_ids.

        Args:
            target_sequences: List of target sequences
            target_headers: List of target headers
            input_sequences: List of unique input sequences
            input_hash_ids: List of hash_ids corresponding to input_sequences

        Returns:
            List of matching hash_ids from input
        """
        matched_hash_ids = []

        # Create a mapping of normalized input sequences to hash_ids for fast lookup
        input_seq_to_hash_id = {}
        for seq, hash_id in zip(input_sequences, input_hash_ids):
            normalized_seq = seq.upper().strip()
            input_seq_to_hash_id[normalized_seq] = hash_id

        # Find matches
        for i, (target_seq, target_header) in enumerate(zip(target_sequences, target_headers)):
            normalized_target = target_seq.upper().strip()

            if normalized_target in input_seq_to_hash_id:
                hash_id = input_seq_to_hash_id[normalized_target]
                if hash_id not in matched_hash_ids:  # Avoid duplicates
                    matched_hash_ids.append(hash_id)
                self.logger.info(f"MATCH {i+1}: Target '{target_header}' -> Hash '{hash_id}'")
            else:
                self.logger.warning(f"NO MATCH {i+1}: Target '{target_header}' (length: {len(target_seq)}) not found in input sequences")

        return matched_hash_ids

    def _expand_hash_ids_to_headers(self, results: DecomposeResults, hash_to_headers: Dict[str, List[str]]) -> DecomposeResults:
        """Expand hash IDs back to original headers in the results.

        Args:
            results: Results with hash IDs
            hash_to_headers: Mapping from hash_id to list of original headers

        Returns:
            Results with original headers expanded
        """
        expanded_results = DecomposeResults()
        expanded_results.total_iterations = results.total_iterations
        expanded_results.total_sequences_processed = results.total_sequences_processed
        expanded_results.coverage_percentage = results.coverage_percentage
        expanded_results.iteration_summaries = results.iteration_summaries

        # Expand clusters
        for cluster_id, hash_ids in results.clusters.items():
            expanded_headers = []
            for hash_id in hash_ids:
                if hash_id in hash_to_headers:
                    expanded_headers.extend(hash_to_headers[hash_id])
                else:
                    # Fallback - keep hash_id if no mapping found
                    expanded_headers.append(hash_id)
            expanded_results.clusters[cluster_id] = expanded_headers

        # Expand all_clusters
        for cluster_id, hash_ids in results.all_clusters.items():
            expanded_headers = []
            for hash_id in hash_ids:
                if hash_id in hash_to_headers:
                    expanded_headers.extend(hash_to_headers[hash_id])
                else:
                    expanded_headers.append(hash_id)
            expanded_results.all_clusters[cluster_id] = expanded_headers

        # Expand unassigned
        expanded_unassigned = []
        for hash_id in results.unassigned:
            if hash_id in hash_to_headers:
                expanded_unassigned.extend(hash_to_headers[hash_id])
            else:
                expanded_unassigned.append(hash_id)
        expanded_results.unassigned = expanded_unassigned

        # Expand conflicts
        for hash_id, cluster_ids in results.conflicts.items():
            if hash_id in hash_to_headers:
                # Create conflict entries for each original header
                for original_header in hash_to_headers[hash_id]:
                    expanded_results.conflicts[original_header] = cluster_ids
            else:
                expanded_results.conflicts[hash_id] = cluster_ids

        return expanded_results

    def _merge_overlapping_clusters(self, results: DecomposeResults, sequences: List[str], headers: List[str]) -> DecomposeResults:
        """Iteratively detect and merge overlapping clusters with medoid caching.

        Uses iterative merging to allow newly merged clusters to participate in
        subsequent merges. Caches expensive medoid calculations to avoid recomputation
        for unchanged clusters across iterations.

        Args:
            results: Initial decomposition results
            sequences: Full sequence list
            headers: Full header list

        Returns:
            Updated results with merged clusters
        """
        if not results.all_clusters or len(results.all_clusters) < 2:
            self.logger.info("No overlapping clusters to merge (less than 2 clusters)")
            return results

        self.logger.info(f"Starting iterative cluster merging with {len(results.all_clusters)} clusters")

        # Convert cluster data to sets for easier containment calculation
        cluster_sets = {}
        for cluster_id, cluster_headers in results.all_clusters.items():
            cluster_sets[cluster_id] = set(cluster_headers)

        # Initialize medoid cache - expensive to compute, so cache across iterations
        medoid_cache = {}

        # Iterative merging until no more valid merges
        iteration = 1
        total_merges = 0
        max_iterations = 10  # Safety limit to prevent runaway

        # Track conflicts across iterations - needs to be updated when clusters merge
        current_conflicts = results.conflicts.copy()

        while iteration <= max_iterations:
            self.logger.info(f"Merge iteration {iteration}: analyzing {len(cluster_sets)} clusters")

            # Find overlapping cluster pairs (conflict-based is cheap, always recompute)
            conflict_pairs = self._find_overlapping_cluster_pairs_from_conflicts(cluster_sets, current_conflicts)

            # Find medoid-based pairs with caching (expensive, cache across iterations)
            medoid_pairs = self._find_overlapping_cluster_pairs_from_medoids_cached(
                cluster_sets, sequences, headers, medoid_cache
            )

            # Combine and deduplicate candidate pairs
            all_candidate_pairs = conflict_pairs + medoid_pairs
            overlapping_pairs = self._deduplicate_candidate_pairs(all_candidate_pairs)

            if not overlapping_pairs:
                self.logger.info(f"Iteration {iteration}: No overlapping pairs found - converged")
                break

            self.logger.info(f"Iteration {iteration}: Found {len(conflict_pairs)} conflict-based, "
                           f"{len(medoid_pairs)} medoid-based candidates")
            self.logger.info(f"Iteration {iteration}: {len(overlapping_pairs)} unique pairs to evaluate")

            # Attempt one merge per iteration (allows fresh overlap detection)
            merged_info = self._perform_single_best_merge(overlapping_pairs, cluster_sets, sequences, headers, medoid_cache)

            if not merged_info:
                self.logger.info(f"Iteration {iteration}: No valid merges found - converged")
                break

            # Update conflicts structure to reflect the merge
            cluster1_id, cluster2_id = merged_info
            self._update_conflicts_after_merge(current_conflicts, cluster1_id, cluster2_id)

            total_merges += 1
            iteration += 1

        if iteration > max_iterations:
            self.logger.warning(f"Reached maximum iterations ({max_iterations}) - stopping merge process")

        self.logger.info(f"Completed iterative merging: {total_merges} total merges across {iteration-1} iterations")
        self.logger.info(f"Final result: {len(cluster_sets)} clusters after merging")

        # Rebuild results with final merged clusters
        return self._rebuild_results_with_merged_clusters(results, cluster_sets)

    def _update_conflicts_after_merge(self, conflicts: Dict[str, List[str]],
                                    surviving_cluster: str, deleted_cluster: str) -> None:
        """Update conflicts dictionary after merging clusters.

        Args:
            conflicts: Conflicts dictionary to update in-place
            surviving_cluster: ID of cluster that absorbed the other
            deleted_cluster: ID of cluster that was deleted
        """
        # Update any conflicts that referenced the deleted cluster
        for seq_id, cluster_ids in conflicts.items():
            if deleted_cluster in cluster_ids:
                # Replace deleted cluster with surviving cluster
                updated_ids = [surviving_cluster if cid == deleted_cluster else cid for cid in cluster_ids]
                # Remove duplicates while preserving order
                seen = set()
                conflicts[seq_id] = [x for x in updated_ids if not (x in seen or seen.add(x))]

        # Remove any conflicts that now only reference one cluster (no longer conflicts)
        conflicts_to_remove = []
        for seq_id, cluster_ids in conflicts.items():
            if len(cluster_ids) <= 1:
                conflicts_to_remove.append(seq_id)

        for seq_id in conflicts_to_remove:
            del conflicts[seq_id]

    def _find_overlapping_cluster_pairs_from_conflicts(self, cluster_sets: Dict[str, Set[str]],
                                                      conflicts: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
        """Find pairs of clusters that have conflicts and exceed the containment threshold.

        Args:
            cluster_sets: Dictionary mapping cluster_id to set of sequence headers
            conflicts: Dictionary mapping sequence_id to list of cluster_ids it belongs to

        Returns:
            List of tuples (cluster1_id, cluster2_id, containment_coefficient)
        """
        overlapping_pairs = []
        cluster_pair_candidates = set()

        # Find all cluster pairs that share sequences (have conflicts)
        for seq_id, cluster_ids in conflicts.items():
            if len(cluster_ids) >= 2:
                # Create all pairs from clusters this sequence belongs to
                for i, cluster1_id in enumerate(cluster_ids):
                    for cluster2_id in cluster_ids[i+1:]:
                        cluster_pair_candidates.add((min(cluster1_id, cluster2_id),
                                                   max(cluster1_id, cluster2_id)))

        self.logger.info(f"Found {len(cluster_pair_candidates)} unique cluster pairs with shared sequences")

        # Check containment threshold for each candidate pair
        for cluster1_id, cluster2_id in cluster_pair_candidates:
            cluster1_seqs = cluster_sets[cluster1_id]
            cluster2_seqs = cluster_sets[cluster2_id]

            # Calculate containment coefficient
            containment = self._calculate_containment_coefficient(cluster1_seqs, cluster2_seqs)

            if containment >= self.containment_threshold:
                overlapping_pairs.append((cluster1_id, cluster2_id, containment))
                self.logger.debug(f"Overlap detected: {cluster1_id} <-> {cluster2_id} "
                                f"(containment: {containment:.3f})")

        return overlapping_pairs

    def _calculate_containment_coefficient(self, cluster1_seqs: Set[str], cluster2_seqs: Set[str]) -> float:
        """Calculate containment coefficient between two clusters.

        Args:
            cluster1_seqs: Set of sequence headers in cluster 1
            cluster2_seqs: Set of sequence headers in cluster 2

        Returns:
            Containment coefficient (fraction of smaller cluster contained in larger)
        """
        if not cluster1_seqs or not cluster2_seqs:
            return 0.0

        # Calculate intersection first
        intersection = len(cluster1_seqs & cluster2_seqs)

        if intersection == 0:
            return 0.0  # No overlap

        # Find size of smaller cluster
        smaller_size = min(len(cluster1_seqs), len(cluster2_seqs))

        # Return containment coefficient
        return intersection / smaller_size if smaller_size > 0 else 0.0

    def _perform_single_best_merge(self, overlapping_pairs: List[Tuple[str, str, float]],
                                  cluster_sets: Dict[str, Set[str]], sequences: List[str], headers: List[str],
                                  medoid_cache: Dict[str, int]) -> Optional[Tuple[str, str]]:
        """Perform single best merge from overlapping pairs and update cache.

        Args:
            overlapping_pairs: List of overlapping cluster pairs sorted by containment
            cluster_sets: Current cluster sets (modified in-place)
            sequences: Full sequence list
            headers: Full header list
            medoid_cache: Medoid cache (modified in-place to remove invalidated entries)

        Returns:
            Tuple of (cluster1_id, cluster2_id) if merge performed, None if no valid merges found
        """
        # Create header to index mapping
        header_to_idx = {header: i for i, header in enumerate(headers)}

        # Use global distance provider for gap calculations
        distance_provider = self._get_or_create_distance_provider(sequences)

        # Initialize gap calculator
        gap_calculator = GapCalculator(self.target_percentile)

        # Try merging pairs in order of containment coefficient (highest first)
        sorted_pairs = sorted(overlapping_pairs, key=lambda x: x[2], reverse=True)

        for cluster1_id, cluster2_id, containment in sorted_pairs:
            # Skip if either cluster was already merged in previous iteration
            if cluster1_id not in cluster_sets or cluster2_id not in cluster_sets:
                continue

            # Get current cluster sets
            cluster1_headers = cluster_sets[cluster1_id]
            cluster2_headers = cluster_sets[cluster2_id]

            # Convert to indices for gap calculation
            cluster1_indices = [header_to_idx[h] for h in cluster1_headers if h in header_to_idx]
            cluster2_indices = [header_to_idx[h] for h in cluster2_headers if h in header_to_idx]

            if not cluster1_indices or not cluster2_indices:
                self.logger.warning(f"Could not find indices for clusters {cluster1_id}, {cluster2_id}")
                continue

            # Test if merge improves gap
            should_merge = self._should_merge_clusters_by_gap(
                cluster1_indices, cluster2_indices, distance_provider, gap_calculator
            )

            if should_merge:
                # Perform merge: combine into cluster1, remove cluster2
                merged_cluster = cluster1_headers | cluster2_headers
                cluster_sets[cluster1_id] = merged_cluster
                del cluster_sets[cluster2_id]

                # Invalidate medoid cache entries for merged clusters
                # cluster1 changed composition, cluster2 was deleted
                medoid_cache.pop(cluster1_id, None)  # Will be recomputed with new composition
                medoid_cache.pop(cluster2_id, None)  # No longer exists

                self.logger.info(f"Merged {cluster2_id} into {cluster1_id} "
                               f"(containment: {containment:.3f}, final size: {len(merged_cluster)})")
                return (cluster1_id, cluster2_id)  # Success - return merged cluster IDs

        # No valid merges found
        return None

    def _should_merge_clusters_by_gap(self, cluster1_indices: List[int], cluster2_indices: List[int],
                                     distance_provider, gap_calculator: GapCalculator) -> bool:
        """Determine if two clusters should be merged based on gap optimization.

        Args:
            cluster1_indices: Sequence indices for cluster 1
            cluster2_indices: Sequence indices for cluster 2
            distance_provider: Distance provider for calculations
            gap_calculator: Gap calculator instance

        Returns:
            True if clusters should be merged, False otherwise
        """
        try:
            # Calculate current individual gaps (each cluster vs all others)
            # For simplicity, we'll use a heuristic: if merged cluster has reasonable intra-cluster distances
            # and the merge doesn't create a cluster that's too dispersed, allow the merge

            # Calculate merged cluster
            merged_indices = cluster1_indices + cluster2_indices

            # Get intra-cluster distances for merged cluster
            intra_distances = []
            for i in range(len(merged_indices)):
                for j in range(i + 1, len(merged_indices)):
                    distance = distance_provider.get_distance(merged_indices[i], merged_indices[j])
                    intra_distances.append(distance)

            if not intra_distances:
                return True  # Single sequence - always allow

            # Check if merged cluster's p95 intra-cluster distance is reasonable
            import numpy as np
            p95_intra = np.percentile(intra_distances, 95)

            # Allow merge if p95 intra-cluster distance is within max_lump threshold
            should_merge = p95_intra <= self.max_lump

            self.logger.debug(f"Merge evaluation: p95_intra={p95_intra:.4f}, "
                            f"max_lump={self.max_lump}, should_merge={should_merge}")

            return should_merge

        except Exception as e:
            self.logger.warning(f"Error in gap calculation for merge decision: {e}")
            return False  # Conservative: don't merge if gap calculation fails

    def _rebuild_results_with_merged_clusters(self, original_results: DecomposeResults,
                                            merged_clusters: Dict[str, Set[str]]) -> DecomposeResults:
        """Rebuild DecomposeResults with merged cluster data.

        Args:
            original_results: Original decomposition results
            merged_clusters: Dictionary of merged cluster sets

        Returns:
            Updated DecomposeResults
        """
        # Create new results object
        new_results = DecomposeResults()

        # Copy non-cluster data
        new_results.iteration_summaries = original_results.iteration_summaries
        new_results.total_iterations = original_results.total_iterations
        new_results.total_sequences_processed = original_results.total_sequences_processed
        new_results.coverage_percentage = original_results.coverage_percentage

        # Convert merged clusters back to lists and create cluster mappings
        new_results.all_clusters = {}
        new_results.clusters = {}

        for cluster_id, cluster_set in merged_clusters.items():
            cluster_headers = list(cluster_set)
            new_results.all_clusters[cluster_id] = cluster_headers
            new_results.clusters[cluster_id] = cluster_headers  # No conflicts after merging

        # Clear conflicts since merging resolves them
        new_results.conflicts = {}

        # Update unassigned sequences
        all_assigned = set()
        for cluster_headers in new_results.all_clusters.values():
            all_assigned.update(cluster_headers)

        # Keep original unassigned sequences
        new_results.unassigned = original_results.unassigned

        self.logger.info(f"Cluster merging complete: {len(original_results.all_clusters)} -> "
                        f"{len(new_results.all_clusters)} clusters")

        return new_results

    def _get_or_create_distance_provider(self, sequences: List[str]) -> DistanceProvider:
        """Get or create the global distance provider for program lifetime."""
        if self._global_distance_provider is None:
            self.logger.debug(f"Creating global distance provider for {len(sequences)} sequences")
            self._global_distance_provider = DistanceProviderFactory.create_lazy_provider(
                sequences,
                alignment_method="adjusted",
                end_skip_distance=20,
                normalize_homopolymers=True,
                handle_iupac_overlap=True,
                normalize_indels=True,
                max_repeat_motif_length=2
            )
        return self._global_distance_provider

    def _find_cluster_medoid(self, cluster_headers: Set[str], sequences: List[str], headers: List[str]) -> int:
        """Find medoid (sequence with minimum total distance to all others in cluster)."""

        if len(cluster_headers) == 1:
            return headers.index(list(cluster_headers)[0])

        cluster_indices = [headers.index(h) for h in cluster_headers]

        # Use the global distance provider for cached computation
        distance_provider = self._get_or_create_distance_provider(sequences)

        min_total_distance = float('inf')
        medoid_idx = cluster_indices[0]

        for candidate_idx in cluster_indices:
            total_distance = 0.0
            for other_idx in cluster_indices:
                if candidate_idx != other_idx:
                    distance = distance_provider.get_distance(candidate_idx, other_idx)
                    total_distance += distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid_idx = candidate_idx

        return medoid_idx

    def _find_overlapping_cluster_pairs_from_medoids_cached(self, cluster_sets: Dict[str, Set[str]],
                                                          sequences: List[str], headers: List[str],
                                                          medoid_cache: Dict[str, int]) -> List[Tuple[str, str, float]]:
        """Find cluster pairs where medoids are within max_lump distance with caching.

        Args:
            cluster_sets: Cluster ID -> Set of sequence headers
            sequences: Full sequence list
            headers: Full header list
            medoid_cache: Cache mapping cluster_id -> medoid_index (modified in-place)

        Returns:
            List of (cluster1_id, cluster2_id, containment_coefficient) tuples
        """

        if len(cluster_sets) < 2:
            return []

        self.logger.debug(f"Finding medoid-based overlap candidates among {len(cluster_sets)} clusters")

        # Calculate medoids with caching - only compute for clusters not in cache
        new_clusters = []
        cached_count = 0

        for cluster_id, cluster_headers in cluster_sets.items():
            if cluster_id not in medoid_cache:
                new_clusters.append((cluster_id, cluster_headers))
            else:
                cached_count += 1

        self.logger.debug(f"Medoid cache: {cached_count} cached, {len(new_clusters)} new clusters to compute")

        # Compute medoids for new clusters with progress bar
        if new_clusters:
            from tqdm import tqdm

            if self.show_progress and len(new_clusters) > 1:
                medoid_pbar = tqdm(new_clusters, desc="Computing new cluster medoids", unit=" clusters")
            else:
                medoid_pbar = new_clusters

            try:
                for cluster_id, cluster_headers in medoid_pbar:
                    medoid_idx = self._find_cluster_medoid(cluster_headers, sequences, headers)
                    medoid_cache[cluster_id] = medoid_idx
                    self.logger.debug(f"Cluster {cluster_id}: computed medoid is sequence {medoid_idx} ({headers[medoid_idx]})")
            finally:
                if self.show_progress and hasattr(medoid_pbar, 'close'):
                    medoid_pbar.close()

        # Check all pairs of medoids for proximity
        candidate_pairs = []
        cluster_ids = list(cluster_sets.keys())
        total_pairs = len(cluster_ids) * (len(cluster_ids) - 1) // 2  # n choose 2

        # Use the global distance provider for cached computation
        distance_provider = self._get_or_create_distance_provider(sequences)

        if self.show_progress and total_pairs > 10:
            distance_pbar = tqdm(total=total_pairs, desc="Computing medoid distances", unit=" pairs")
        else:
            distance_pbar = None

        try:
            for i, cluster1_id in enumerate(cluster_ids):
                for cluster2_id in cluster_ids[i+1:]:
                    medoid1_idx = medoid_cache[cluster1_id]
                    medoid2_idx = medoid_cache[cluster2_id]

                    # Calculate distance between medoids using cached provider
                    distance = distance_provider.get_distance(medoid1_idx, medoid2_idx)

                    if distance <= self.max_lump:
                        # Calculate containment for consistency with existing approach
                        containment = self._calculate_containment_coefficient(
                            cluster_sets[cluster1_id], cluster_sets[cluster2_id]
                        )
                        candidate_pairs.append((cluster1_id, cluster2_id, containment))
                        self.logger.debug(f"Medoid-based overlap candidate: {cluster1_id} <-> {cluster2_id} "
                                        f"(medoid distance: {distance:.4f}, containment: {containment:.3f})")

                    if distance_pbar:
                        distance_pbar.update(1)
        finally:
            if distance_pbar:
                distance_pbar.close()

        self.logger.debug(f"Found {len(candidate_pairs)} medoid-based overlap candidate pairs")
        return candidate_pairs

    def _deduplicate_candidate_pairs(self, all_pairs: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Deduplicate candidate pairs, keeping the highest containment coefficient for duplicates."""

        # Track unique pairs by cluster ID tuple
        unique_pairs = {}

        for cluster1_id, cluster2_id, containment in all_pairs:
            pair_key = (min(cluster1_id, cluster2_id), max(cluster1_id, cluster2_id))

            if pair_key in unique_pairs:
                # Keep the pair with higher containment coefficient
                if containment > unique_pairs[pair_key][2]:
                    unique_pairs[pair_key] = (cluster1_id, cluster2_id, containment)
            else:
                unique_pairs[pair_key] = (cluster1_id, cluster2_id, containment)

        return list(unique_pairs.values())

    def _resolve_conflicts_via_reclustering(self, results: DecomposeResults,
                                          sequences: List[str], headers: List[str]) -> DecomposeResults:
        """Resolve conflicts using principled reclustering with classic gapHACk.

        Args:
            results: Current decomposition results with conflicts
            sequences: Full sequence list
            headers: Full header list

        Returns:
            Updated DecomposeResults with conflicts resolved
        """
        from .principled_reclustering import resolve_conflicts_via_reclustering, ReclusteringConfig
        from .cluster_proximity import BruteForceProximityGraph, BlastKNNProximityGraph

        # Get global distance provider for the full dataset
        distance_provider = self._get_or_create_distance_provider(sequences)

        # Create proximity graph for cluster proximity queries
        proximity_graph = self._create_proximity_graph(results.all_clusters, sequences, headers, distance_provider)

        # Create reclustering configuration with appropriate thresholds
        config = ReclusteringConfig(
            max_classic_gaphack_size=300,  # Conservative limit for performance
            conflict_expansion_threshold=1.5 * self.max_lump,  # Expand scope near conflicted clusters
            jaccard_overlap_threshold=0.1,  # Include clusters with 10%+ overlap
            significant_difference_threshold=0.2  # 20% sequences must change for significant difference
        )

        # Apply conflict resolution
        resolved_clusters = resolve_conflicts_via_reclustering(
            conflicts=results.conflicts,
            all_clusters=results.all_clusters,
            sequences=sequences,
            headers=headers,
            distance_provider=distance_provider,
            proximity_graph=proximity_graph,
            config=config,
            min_split=self.min_split,
            max_lump=self.max_lump,
            target_percentile=self.target_percentile
        )

        # Rebuild results with resolved clusters
        new_results = DecomposeResults()
        new_results.iteration_summaries = results.iteration_summaries
        new_results.total_iterations = results.total_iterations
        new_results.total_sequences_processed = results.total_sequences_processed
        new_results.coverage_percentage = results.coverage_percentage

        # Renumber clusters sequentially for consistent naming
        renumbered_clusters = self._renumber_clusters_sequentially(resolved_clusters)

        # Set resolved clusters (should be MECE now)
        new_results.all_clusters = renumbered_clusters
        new_results.clusters = renumbered_clusters  # No conflicts after resolution

        # Clear conflicts since they've been resolved
        new_results.conflicts = {}

        # Keep original unassigned sequences
        new_results.unassigned = results.unassigned

        self.logger.info(f"Conflict resolution complete: {len(results.conflicts)} conflicts resolved, "
                        f"{len(results.all_clusters)} -> {len(new_results.all_clusters)} clusters")

        return new_results

    def _refine_close_clusters_via_reclustering(self, results: DecomposeResults,
                                              sequences: List[str], headers: List[str]) -> DecomposeResults:
        """Refine close clusters using principled reclustering with classic gapHACk.

        Args:
            results: Current decomposition results
            sequences: Full sequence list
            headers: Full header list

        Returns:
            Updated DecomposeResults with close clusters refined
        """
        from .principled_reclustering import refine_close_clusters, ReclusteringConfig
        from .cluster_proximity import BruteForceProximityGraph, BlastKNNProximityGraph

        # Get global distance provider for the full dataset
        distance_provider = self._get_or_create_distance_provider(sequences)

        # Create proximity graph for cluster proximity queries
        proximity_graph = self._create_proximity_graph(results.all_clusters, sequences, headers, distance_provider)

        # Create reclustering configuration with appropriate thresholds
        config = ReclusteringConfig(
            max_classic_gaphack_size=300,  # Conservative limit for performance
            close_cluster_expansion_threshold=1.2 * self.max_lump,  # Expand scope near close clusters
            jaccard_overlap_threshold=0.1,  # Include clusters with 10%+ overlap
            significant_difference_threshold=0.2  # 20% sequences must change for significant difference
        )

        # Apply close cluster refinement
        refined_clusters = refine_close_clusters(
            all_clusters=results.all_clusters,
            sequences=sequences,
            headers=headers,
            distance_provider=distance_provider,
            proximity_graph=proximity_graph,
            config=config,
            min_split=self.min_split,
            max_lump=self.max_lump,
            target_percentile=self.target_percentile,
            close_threshold=self.max_lump  # Use max_lump as close threshold
        )

        # Rebuild results with refined clusters
        new_results = DecomposeResults()
        new_results.iteration_summaries = results.iteration_summaries
        new_results.total_iterations = results.total_iterations
        new_results.total_sequences_processed = results.total_sequences_processed
        new_results.coverage_percentage = results.coverage_percentage

        # Apply sequential renumbering for consistency
        refined_clusters = self._renumber_clusters_sequentially(refined_clusters)

        # Update cluster assignments
        new_results.all_clusters = refined_clusters
        new_results.clusters = refined_clusters  # No conflicts expected after refinement
        new_results.conflicts = {}  # Close cluster refinement doesn't create conflicts
        new_results.unassigned = results.unassigned

        self.logger.info(f"Close cluster refinement complete: "
                        f"{len(results.all_clusters)} -> {len(new_results.all_clusters)} clusters")

        return new_results

    def _create_proximity_graph(self, clusters: Dict[str, List[str]], sequences: List[str],
                               headers: List[str], distance_provider) -> 'ClusterProximityGraph':
        """Create proximity graph based on configuration.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            sequences: Full sequence list
            headers: Full header list
            distance_provider: Provider for distance calculations

        Returns:
            ClusterProximityGraph instance
        """
        from .cluster_proximity import BruteForceProximityGraph, BlastKNNProximityGraph

        if self.proximity_graph == 'blast-knn':
            self.logger.info(f"Creating BLAST K-NN proximity graph with K={self.knn_neighbors}")
            return BlastKNNProximityGraph(
                clusters=clusters,
                sequences=sequences,
                headers=headers,
                distance_provider=distance_provider,
                k_neighbors=self.knn_neighbors,
                blast_evalue=self.blast_evalue,  # Use user-specified e-value
                blast_identity=self.min_identity or 90.0  # Use user-specified identity or default
            )
        else:
            self.logger.info("Creating brute force proximity graph")
            return BruteForceProximityGraph(
                clusters=clusters,
                sequences=sequences,
                headers=headers,
                distance_provider=distance_provider
            )

    def _renumber_clusters_sequentially(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Renumber clusters with sequential cluster_XXX naming for consistency.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers

        Returns:
            Dictionary with clusters renumbered as cluster_001, cluster_002, etc.
        """
        if not clusters:
            return {}

        # Sort clusters by size (largest first) for consistent ordering
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Create new sequential mapping
        renumbered = {}
        for i, (old_cluster_id, cluster_headers) in enumerate(sorted_clusters, 1):
            new_cluster_id = f"cluster_{i:03d}"
            renumbered[new_cluster_id] = cluster_headers

        self.logger.debug(f"Renumbered {len(clusters)} clusters with sequential IDs: "
                         f"{list(clusters.keys())[:3]}... → {list(renumbered.keys())[:3]}...")

        return renumbered