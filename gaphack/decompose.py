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

logger = logging.getLogger(__name__)


class ClusterIDGenerator:
    """Generates sequential cluster IDs for internal processing."""

    def __init__(self, prefix: str = "active"):
        self.prefix = prefix
        self.counter = 1

    def next_id(self) -> str:
        """Generate next sequential active cluster ID."""
        cluster_id = f"{self.prefix}_{self.counter:04d}"
        self.counter += 1
        return cluster_id

    def get_current_count(self) -> int:
        """Get current counter value."""
        return self.counter - 1


@dataclass
class ProcessingStageInfo:
    """Information about a processing stage (conflict resolution or close cluster refinement)."""
    stage_name: str
    clusters_before: Dict[str, List[str]] = field(default_factory=dict)  # active_id -> sequence headers
    clusters_after: Dict[str, List[str]] = field(default_factory=dict)   # active_id -> sequence headers
    components_processed: List[Dict] = field(default_factory=list)       # details about each component processed
    summary_stats: Dict = field(default_factory=dict)                    # before/after counts, etc.


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
    verification_results: Dict[str, Dict] = field(default_factory=dict)  # comprehensive conflict verification results

    # Enhanced tracking for debugging
    processing_stages: List[ProcessingStageInfo] = field(default_factory=list)  # conflict resolution, refinement stages
    active_to_final_mapping: Dict[str, str] = field(default_factory=dict)       # active_id -> final_id
    command_line: str = ""                                                        # command used to run decompose
    start_time: str = ""                                                         # ISO timestamp of run start


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


class TargetSelector:
    """Target selection strategy using provided target sequences."""
    
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
        """No-op for directed mode - doesn't use BLAST memory."""
        pass
    
    def mark_sequences_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """No-op for directed mode - doesn't need memory management."""
        pass


class BlastResultMemory:
    """Memory pool for storing BLAST neighborhoods for nearby target selection."""
    
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
    
    def get_nearby_candidates(self, assignment_tracker: AssignmentTracker) -> List[str]:
        """Get unassigned sequences from BLAST neighborhoods for nearby selection."""
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


class NearbyTargetSelector:
    """Target selection strategy using nearby sequence exploration with random fallback."""
    
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
        """Get next target using nearby sequence logic with random fallback."""
        self.iteration_count += 1
        
        # Try nearby selection first: pick from previous BLAST neighborhoods
        nearby_candidates = self.blast_memory.get_nearby_candidates(assignment_tracker)
        nearby_candidates = [h for h in nearby_candidates if h not in self.used_targets]
        
        target_header = None
        selection_method = ""
        
        if nearby_candidates:
            # Nearby selection: choose from BLAST neighborhood candidates
            target_header = self.random_state.choice(nearby_candidates)
            selection_method = "nearby"
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
                        f"(nearby_pool: {len(nearby_candidates)}, total_unassigned: {len([h for h in self.all_headers if not assignment_tracker.is_assigned(h)])})")
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
        """Store BLAST neighborhood before pruning for future nearby selection."""
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
                 resolve_conflicts: bool = False,
                 refine_close_clusters: bool = False,
                 close_cluster_threshold: float = 0.0,
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
            resolve_conflicts: Enable cluster refinement for conflict resolution with minimal scope (default: False)
            refine_close_clusters: Enable cluster refinement for close cluster refinement (default: False)
            close_cluster_threshold: Distance threshold for close cluster refinement and scope expansion (default: 0.0)
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
        self.resolve_conflicts = resolve_conflicts
        self.refine_close_clusters = refine_close_clusters
        self.close_cluster_threshold = close_cluster_threshold
        self.knn_neighbors = 20  # Hardcoded for BLAST K-NN cluster graph
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
                 max_clusters: Optional[int] = None,
                 max_sequences: Optional[int] = None) -> DecomposeResults:
        """Perform decomposition clustering.

        Args:
            input_fasta: Path to input FASTA file with all sequences
            targets_fasta: Path to FASTA file with target sequences (directed mode)
            max_clusters: Maximum clusters to create (undirected mode)
            max_sequences: Maximum sequences to assign (undirected mode)
            
        Returns:
            DecomposeResults with clustering results
        """
        # Auto-detect mode based on targets
        if targets_fasta:
            mode = "directed"
            self.logger.info("Starting decomposition clustering in directed mode (targets provided)")
        else:
            mode = "undirected"
            self.logger.info("Starting decomposition clustering in undirected mode")
        
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
        
        # Initialize target selector based on mode
        if mode == "directed":
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
            target_selector = TargetSelector(matched_hash_ids)
        elif mode == "undirected":
            # Create nearby target selector with all sequence headers
            target_selector = NearbyTargetSelector(
                all_headers=headers,
                max_clusters=max_clusters,
                max_sequences=max_sequences
            )
            self.logger.info(f"Initialized undirected mode with nearby target selection: "
                           f"max_clusters={max_clusters}, max_sequences={max_sequences}")
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        
        # Initialize progress tracking based on mode and stopping criteria
        if mode == "directed":
            # Directed mode: track target completion
            progress_mode = "targets"
            progress_total = len(matched_hash_ids)
            progress_unit = " targets"
            progress_desc = "Processing targets"
        elif mode == "undirected":
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
                
                # Store BLAST neighborhood in memory for nearby target selection (before pruning)
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
                cluster_id = f"initial_{iteration:03d}"
                
                # Assign sequences to cluster
                assignment_tracker.assign_sequences(target_cluster_headers, cluster_id, iteration)
                
                # Update BLAST memory for nearby target selection
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
                    # Directed mode: increment by 1 target processed
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
                
                # Check stopping criteria for undirected mode
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

        # Perform initial conflict verification after decomposition
        initial_verification = self._verify_no_conflicts(results.all_clusters, results.conflicts, "after_decomposition")

        # Store original conflicts for tracking resolution progress
        original_conflicts = results.conflicts.copy() if results.conflicts else {}

        # Apply cluster refinement for conflict resolution if enabled
        if getattr(self, 'resolve_conflicts', False) and results.conflicts:
            self.logger.info(f"Starting cluster refinement for {len(results.conflicts)} conflicts")
            results = self._resolve_conflicts(results, sequences, headers)

            # Verify conflict resolution effectiveness
            if original_conflicts:
                post_resolution_verification = self._verify_no_conflicts(
                    results.all_clusters, original_conflicts, "after_conflict_resolution"
                )

        # Apply cluster refinement for close cluster refinement if enabled
        if getattr(self, 'refine_close_clusters', False):
            self.logger.info("Starting cluster refinement for close cluster refinement")
            results = self._refine_close_clusters_via_refinement(results, sequences, headers)

        # Expand hash IDs back to original headers
        results = self._expand_hash_ids_to_headers(results, hash_to_headers)

        # CRITICAL: Always perform final comprehensive conflict verification
        # This runs regardless of whether initial conflicts were detected or resolved
        # to catch any conflicts that may have been missed or introduced during processing
        self.logger.info("Performing final comprehensive conflict verification (always runs regardless of initial conflict status)")
        final_verification = self._verify_no_conflicts(results.all_clusters, original_conflicts, "final_comprehensive")

        # Store verification results in the results object for external access
        results.verification_results = {
            'initial': initial_verification,
            'final': final_verification
        }

        # Add post-resolution verification if conflict resolution was performed
        if getattr(self, 'resolve_conflicts', False) and original_conflicts and 'post_resolution_verification' in locals():
            results.verification_results['post_resolution'] = post_resolution_verification

        # Update final results based on comprehensive verification findings
        # The final verification is the authoritative source of truth for conflicts
        if final_verification.get('critical_failure', False):
            # Override the conflicts field with what final verification actually found
            results.conflicts = final_verification['conflicts']
            self.logger.warning(f"Final verification detected {len(final_verification['conflicts'])} conflicts that were missed by initial detection!")

        # Always update the conflicts field with the final verification results for accuracy
        # This ensures the output reflects the true state regardless of intermediate processing
        results.conflicts = final_verification['conflicts']

        # Log final status
        if final_verification['no_conflicts']:
            self.logger.info("🎉 Final verification confirms conflict-free clustering achieved")
        else:
            self.logger.error(f"❌ Final verification reveals {len(final_verification['conflicts'])} conflicts in output")

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

        # Copy enhanced tracking fields
        expanded_results.processing_stages = results.processing_stages
        expanded_results.active_to_final_mapping = results.active_to_final_mapping
        expanded_results.command_line = results.command_line
        expanded_results.start_time = results.start_time
        expanded_results.verification_results = results.verification_results

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

    def _resolve_conflicts(self, results: DecomposeResults,
                                          sequences: List[str], headers: List[str]) -> DecomposeResults:
        """Resolve conflicts using cluster refinement with full gapHACk.

        Args:
            results: Current decomposition results with conflicts
            sequences: Full sequence list
            headers: Full header list

        Returns:
            Updated DecomposeResults with conflicts resolved
        """
        from .cluster_refinement import resolve_conflicts, RefinementConfig

        # Get global distance provider for the full dataset
        distance_provider = self._get_or_create_distance_provider(sequences)

        # Create refinement configuration for minimal conflict resolution
        config = RefinementConfig(
            max_full_gaphack_size=300  # Conservative limit for performance
        )

        # Apply conflict resolution (no proximity graph needed - uses minimal scope only)
        conflict_id_generator = ClusterIDGenerator(prefix="deconflicted")
        resolved_clusters, conflict_tracking = resolve_conflicts(
            conflicts=results.conflicts,
            all_clusters=results.all_clusters,
            sequences=sequences,
            headers=headers,
            distance_provider=distance_provider,
            config=config,
            min_split=self.min_split,
            max_lump=self.max_lump,
            target_percentile=self.target_percentile,
            cluster_id_generator=conflict_id_generator
        )

        # Rebuild results with resolved clusters
        new_results = DecomposeResults()
        new_results.iteration_summaries = results.iteration_summaries
        new_results.total_iterations = results.total_iterations
        new_results.total_sequences_processed = results.total_sequences_processed
        new_results.coverage_percentage = results.coverage_percentage

        # Renumber clusters sequentially for consistent naming
        renumbered_clusters, mapping = self._renumber_clusters_sequentially(resolved_clusters)

        # Set resolved clusters (should be conflict-free now)
        new_results.all_clusters = renumbered_clusters
        new_results.clusters = renumbered_clusters  # No conflicts after resolution

        # Initialize active to final mapping (will be set properly at final renumbering)
        new_results.active_to_final_mapping = {}

        # Add conflict resolution tracking info
        new_results.processing_stages = results.processing_stages.copy()
        new_results.processing_stages.append(conflict_tracking)

        # Preserve command line and start time
        new_results.command_line = results.command_line
        new_results.start_time = results.start_time

        # Clear conflicts since they've been resolved
        new_results.conflicts = {}

        # Keep original unassigned sequences
        new_results.unassigned = results.unassigned

        self.logger.info(f"Conflict resolution complete: {len(results.conflicts)} conflicts resolved, "
                        f"{len(results.all_clusters)} -> {len(new_results.all_clusters)} clusters")

        return new_results

    def _refine_close_clusters_via_refinement(self, results: DecomposeResults,
                                              sequences: List[str], headers: List[str]) -> DecomposeResults:
        """Refine close clusters using cluster refinement with full gapHACk.

        Args:
            results: Current decomposition results
            sequences: Full sequence list
            headers: Full header list

        Returns:
            Updated DecomposeResults with close clusters refined
        """
        from .cluster_refinement import refine_close_clusters, RefinementConfig
        from .cluster_graph import ClusterGraph

        # Get global distance provider for the full dataset
        distance_provider = self._get_or_create_distance_provider(sequences)

        # Create proximity graph for cluster proximity queries
        proximity_graph = self._create_proximity_graph(results.all_clusters, sequences, headers, distance_provider)

        # Create refinement configuration with user-provided threshold
        config = RefinementConfig(
            max_full_gaphack_size=300,  # Conservative limit for performance
            close_cluster_expansion_threshold=self.close_cluster_threshold  # User-controlled expansion threshold
        )

        # Apply close cluster refinement
        refinement_id_generator = ClusterIDGenerator(prefix="refined")
        refined_clusters, refinement_tracking = refine_close_clusters(
            all_clusters=results.all_clusters,
            sequences=sequences,
            headers=headers,
            distance_provider=distance_provider,
            proximity_graph=proximity_graph,
            config=config,
            min_split=self.min_split,
            max_lump=self.max_lump,
            target_percentile=self.target_percentile,
            close_threshold=self.max_lump,  # Use max_lump as close threshold
            cluster_id_generator=refinement_id_generator
        )

        # Rebuild results with refined clusters
        new_results = DecomposeResults()
        new_results.iteration_summaries = results.iteration_summaries
        new_results.total_iterations = results.total_iterations
        new_results.total_sequences_processed = results.total_sequences_processed
        new_results.coverage_percentage = results.coverage_percentage

        # Apply sequential renumbering for consistency
        refined_clusters, mapping = self._renumber_clusters_sequentially(refined_clusters)

        # Update cluster assignments
        new_results.all_clusters = refined_clusters
        new_results.clusters = refined_clusters  # No conflicts expected after refinement

        # Set final active to final mapping (replace any previous mappings)
        new_results.active_to_final_mapping = mapping

        # Add close cluster refinement tracking info
        new_results.processing_stages = results.processing_stages.copy()
        new_results.processing_stages.append(refinement_tracking)

        # Preserve command line and start time
        new_results.command_line = results.command_line
        new_results.start_time = results.start_time

        new_results.conflicts = {}  # Close cluster refinement doesn't create conflicts
        new_results.unassigned = results.unassigned

        self.logger.info(f"Close cluster refinement complete: "
                        f"{len(results.all_clusters)} -> {len(new_results.all_clusters)} clusters")

        return new_results

    def _create_proximity_graph(self, clusters: Dict[str, List[str]], sequences: List[str],
                               headers: List[str], distance_provider) -> 'ClusterGraph':
        """Create proximity graph based on configuration.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            sequences: Full sequence list
            headers: Full header list
            distance_provider: Provider for distance calculations

        Returns:
            ClusterGraph instance
        """
        from .cluster_graph import ClusterGraph

        self.logger.info(f"Creating BLAST K-NN cluster graph with K={self.knn_neighbors}")
        return ClusterGraph(
            clusters=clusters,
            sequences=sequences,
            headers=headers,
            distance_provider=distance_provider,
            k_neighbors=self.knn_neighbors,
            blast_evalue=self.blast_evalue,  # Use user-specified e-value
            blast_identity=self.min_identity or 90.0  # Use user-specified identity or default
        )

    def _renumber_clusters_sequentially(self, clusters: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Renumber clusters with sequential cluster_XXX naming for consistency.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers

        Returns:
            Tuple of (renumbered_clusters, active_to_final_mapping)
        """
        if not clusters:
            return {}, {}

        # Sort clusters by size (largest first) for consistent ordering
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Create new sequential mapping and track active→final mapping
        renumbered = {}
        active_to_final_mapping = {}
        for i, (old_cluster_id, cluster_headers) in enumerate(sorted_clusters, 1):
            new_cluster_id = f"cluster_{i:03d}"
            renumbered[new_cluster_id] = cluster_headers
            active_to_final_mapping[old_cluster_id] = new_cluster_id

        self.logger.debug(f"Renumbered {len(clusters)} clusters with sequential IDs: "
                         f"{list(clusters.keys())[:3]}... → {list(renumbered.keys())[:3]}...")

        return renumbered, active_to_final_mapping

    def _verify_no_conflicts(self, clusters: Dict[str, List[str]],
                             original_conflicts: Optional[Dict[str, List[str]]] = None,
                             context: str = "verification") -> Dict[str, any]:
        """Verify no conflicts of cluster assignments using comprehensive verification.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            original_conflicts: Optional original conflicts for comparison
            context: Context string for logging

        Returns:
            Dictionary with verification results from comprehensive scan
        """
        from .cluster_refinement import verify_no_conflicts

        return verify_no_conflicts(
            clusters=clusters,
            original_conflicts=original_conflicts,
            context=context
        )