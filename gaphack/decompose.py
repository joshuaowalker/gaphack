"""Decomposition clustering for gaphack using BLAST neighborhoods and target clustering."""

import logging
import copy
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .blast_neighborhood import BlastNeighborhoodFinder
from .target_clustering import TargetModeClustering
from .utils import load_sequences_from_fasta, load_sequences_with_deduplication, calculate_distance_matrix
from .lazy_distances import DistanceProviderFactory
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
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)
        
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
                    neighborhood_sequences, sequences_for_clustering, target_indices_in_neighborhood
                )
                
                # Create lazy distance provider for pruned neighborhood
                self.logger.debug(f"Creating distance provider for {len(pruned_sequences)} pruned neighborhood sequences")
                distance_provider = DistanceProviderFactory.create_lazy_provider(
                    pruned_sequences,
                    alignment_method="adjusted",  # Use consistent parameters
                    end_skip_distance=20,
                    normalize_homopolymers=True,
                    handle_iupac_overlap=True,
                    normalize_indels=True,
                    max_repeat_motif_length=2
                )
                
                # Perform target clustering on pruned neighborhood
                target_cluster_indices, remaining_indices, clustering_metrics = self.target_clustering.cluster(
                    distance_provider, pruned_target_indices, pruned_sequences
                )
                
                # Log distance computation statistics
                if hasattr(distance_provider, 'get_cache_stats'):
                    stats = distance_provider.get_cache_stats()
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

        # Expand hash IDs back to original headers
        results = self._expand_hash_ids_to_headers(results, hash_to_headers)

        return results
    
    def _prune_neighborhood_by_distance(self, neighborhood_sequences: List[str], 
                                       neighborhood_headers: List[str], 
                                       target_indices: List[int]) -> Tuple[List[str], List[str], List[int]]:
        """Prune neighborhood by distance to targets to reduce computational cost.
        
        Args:
            neighborhood_sequences: All sequences in neighborhood
            neighborhood_headers: Headers for neighborhood sequences
            target_indices: Indices of target sequences within neighborhood
            
        Returns:
            Tuple of (pruned_sequences, pruned_headers, new_target_indices)
        """
        if len(neighborhood_sequences) <= 100:  # Skip pruning for small neighborhoods
            return neighborhood_sequences, neighborhood_headers, target_indices
        
        self.logger.debug(f"Pruning neighborhood from {len(neighborhood_sequences)} sequences")
        
        # Calculate distances from all sequences to target sequences
        target_sequences = [neighborhood_sequences[i] for i in target_indices]
        
        # Create a temporary distance provider for pruning calculation
        temp_distance_provider = DistanceProviderFactory.create_lazy_provider(
            neighborhood_sequences,
            alignment_method="adjusted",
            end_skip_distance=20,
            normalize_homopolymers=True,
            handle_iupac_overlap=True,
            normalize_indels=True,
            max_repeat_motif_length=2
        )
        
        # Calculate minimum distance from each sequence to any target
        sequence_min_distances = []
        for seq_idx in range(len(neighborhood_sequences)):
            if seq_idx in target_indices:
                # Target sequences have distance 0 to themselves
                min_dist = 0.0
            else:
                # Find minimum distance to any target
                distances_to_targets = temp_distance_provider.get_distances_from_sequence(
                    seq_idx, set(target_indices)
                )
                min_dist = min(distances_to_targets.values())
            
            sequence_min_distances.append((seq_idx, min_dist))
        
        # Count sequences within max_lump distance
        within_max_lump = sum(1 for _, dist in sequence_min_distances if dist <= self.max_lump)
        
        # Calculate target size for pruning (2x sequences within max_lump, minimum 50)
        target_pruned_size = max(50, min(len(neighborhood_sequences), 2 * within_max_lump))
        
        self.logger.debug(f"Found {within_max_lump} sequences within max_lump distance {self.max_lump:.4f}")
        self.logger.debug(f"Pruning to {target_pruned_size} closest sequences (2x estimate)")
        
        # Sort by distance and take the closest sequences
        sequence_min_distances.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in sequence_min_distances[:target_pruned_size]]
        
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
        """Detect and merge overlapping clusters based on containment coefficient.

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

        self.logger.info(f"Analyzing {len(results.all_clusters)} clusters for overlaps")

        # Convert cluster data to sets for easier containment calculation
        cluster_sets = {}
        for cluster_id, cluster_headers in results.all_clusters.items():
            cluster_sets[cluster_id] = set(cluster_headers)

        # Find overlapping cluster pairs using conflict information
        overlapping_pairs = self._find_overlapping_cluster_pairs_from_conflicts(cluster_sets, results.conflicts)

        if not overlapping_pairs:
            self.logger.info("No overlapping clusters found")
            return results

        self.logger.info(f"Found {len(overlapping_pairs)} overlapping cluster pairs")

        # Perform merges with gap optimization
        merged_clusters = self._perform_cluster_merges(overlapping_pairs, cluster_sets, sequences, headers)

        # Rebuild results with merged clusters
        return self._rebuild_results_with_merged_clusters(results, merged_clusters)

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

    def _perform_cluster_merges(self, overlapping_pairs: List[Tuple[str, str, float]],
                               cluster_sets: Dict[str, Set[str]],
                               sequences: List[str], headers: List[str]) -> Dict[str, Set[str]]:
        """Perform cluster merges using gap optimization to validate merges.

        Args:
            overlapping_pairs: List of overlapping cluster pairs
            cluster_sets: Original cluster sets
            sequences: Full sequence list
            headers: Full header list

        Returns:
            Dictionary of final merged cluster sets
        """
        # Create header to index mapping
        header_to_idx = {header: i for i, header in enumerate(headers)}

        # Initialize merged clusters as copy of originals
        merged_clusters = cluster_sets.copy()

        # Create distance provider for gap calculations
        distance_provider = DistanceProviderFactory.create_lazy_provider(
            sequences,
            alignment_method="adjusted",
            end_skip_distance=20,
            normalize_homopolymers=True,
            handle_iupac_overlap=True,
            normalize_indels=True,
            max_repeat_motif_length=2
        )

        # Initialize gap calculator
        gap_calculator = GapCalculator(self.target_percentile)

        # Sort pairs by containment coefficient (highest first for greedy merging)
        sorted_pairs = sorted(overlapping_pairs, key=lambda x: x[2], reverse=True)

        merged_count = 0
        for cluster1_id, cluster2_id, containment in sorted_pairs:
            # Skip if either cluster was already merged
            if cluster1_id not in merged_clusters or cluster2_id not in merged_clusters:
                continue

            # Get current cluster sets (may have been updated by previous merges)
            cluster1_headers = merged_clusters[cluster1_id]
            cluster2_headers = merged_clusters[cluster2_id]

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
                merged_clusters[cluster1_id] = merged_cluster
                del merged_clusters[cluster2_id]
                merged_count += 1

                self.logger.info(f"Merged {cluster2_id} into {cluster1_id} "
                               f"(containment: {containment:.3f}, "
                               f"final size: {len(merged_cluster)})")
            else:
                self.logger.debug(f"Gap optimization rejected merge of {cluster1_id} and {cluster2_id}")

        self.logger.info(f"Completed cluster merging: {merged_count} merges performed, "
                        f"{len(merged_clusters)} final clusters")

        return merged_clusters

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