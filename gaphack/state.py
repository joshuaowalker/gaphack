"""State management for incremental decompose clustering.

This module implements persistent state for gaphack-decompose, enabling:
- Checkpoint and resume of clustering runs
- Incremental cluster addition
- Staged refinement application
- Manual FASTA editing between stages

Design Principle: FASTA files are the source of truth for cluster membership.
The state JSON contains only metadata, parameters, and file paths - never cluster data.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .cluster_id_utils import (
    get_stage_suffix, format_cluster_id, parse_cluster_id,
    get_next_cluster_number, get_stage_directory, format_cluster_filename,
    load_all_stage_clusters, get_latest_stage_directory
)

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass
class InputInfo:
    """Information about input FASTA file."""
    fasta_path: str
    fasta_hash: str
    total_sequences: int
    deduplicated_sequences: int


@dataclass
class StageInfo:
    """Information about a processing stage."""
    completed: bool = False
    stage_directory: str = ""  # Relative path to stage directory (e.g., "work/initial")
    unassigned_file: str = ""  # Relative path to unassigned file

    # Statistics (for reporting only, not used for logic)
    total_clusters: int = 0
    total_sequences: int = 0


@dataclass
class InitialClusteringStage(StageInfo):
    """Initial clustering stage information."""
    total_iterations: int = 0
    coverage_percentage: float = 0.0
    max_clusters_limit: Optional[int] = None
    max_sequences_limit: Optional[int] = None


@dataclass
class FinalizedStage(StageInfo):
    """Finalization stage information."""
    total_clusters: int = 0  # Number of final numbered clusters
    total_sequences: int = 0  # Total sequences in final clusters
    unassigned_sequences: int = 0  # Sequences left unassigned
    source_stage: str = ""  # Stage that was finalized (initial/deconflicted/refined)


@dataclass
class DecomposeState:
    """Persistent state for incremental decompose runs.

    Contains only metadata, parameters, and file paths.
    Never contains cluster membership data (FASTA files are source of truth).
    """
    version: str
    status: str  # "in_progress", "completed"
    stage: str   # "initial_clustering" or "finalized"

    input: InputInfo
    parameters: Dict[str, Any]

    # Stage information
    initial_clustering: InitialClusteringStage
    finalized: FinalizedStage

    # Metadata
    command_history: List[str] = field(default_factory=list)
    start_time: str = ""
    last_checkpoint: str = ""
    gaphack_version: str = ""

    def save(self, output_dir: Path) -> None:
        """Save state to output_dir/state.json.

        Uses atomic write (temp file + rename) to prevent corruption.

        Args:
            output_dir: Output directory containing state file
        """
        state_file = output_dir / "state.json"
        temp_file = output_dir / ".state.json.tmp"

        # Update last checkpoint timestamp
        self.last_checkpoint = datetime.now().isoformat()

        # Write to temporary file
        with open(temp_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Atomic rename
        temp_file.rename(state_file)
        logger.debug(f"State saved to {state_file}")

    @classmethod
    def load(cls, output_dir: Path) -> 'DecomposeState':
        """Load state from output_dir/state.json.

        Args:
            output_dir: Output directory containing state file

        Returns:
            Loaded DecomposeState object

        Raises:
            FileNotFoundError: If state.json doesn't exist
            ValueError: If state.json is invalid
        """
        state_file = output_dir / "state.json"

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        try:
            with open(state_file, 'r') as f:
                data = json.load(f)

            return cls.from_dict(data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid state file (corrupted JSON): {e}")
        except Exception as e:
            raise ValueError(f"Failed to load state: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'status': self.status,
            'stage': self.stage,
            'input': asdict(self.input),
            'parameters': self.parameters,
            'stages': {
                'initial_clustering': asdict(self.initial_clustering),
                'finalized': asdict(self.finalized)
            },
            'metadata': {
                'command_history': self.command_history,
                'start_time': self.start_time,
                'last_checkpoint': self.last_checkpoint,
                'gaphack_version': self.gaphack_version
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecomposeState':
        """Create DecomposeState from dictionary loaded from JSON."""
        input_info = InputInfo(**data['input'])

        stages = data.get('stages', {})
        initial = InitialClusteringStage(**stages.get('initial_clustering', {}))
        finalized_stage = FinalizedStage(**stages.get('finalized', {}))

        metadata = data.get('metadata', {})

        return cls(
            version=data['version'],
            status=data['status'],
            stage=data['stage'],
            input=input_info,
            parameters=data['parameters'],
            initial_clustering=initial,
            finalized=finalized_stage,
            command_history=metadata.get('command_history', []),
            start_time=metadata.get('start_time', ''),
            last_checkpoint=metadata.get('last_checkpoint', ''),
            gaphack_version=metadata.get('gaphack_version', '')
        )

    def validate_input_hash(self, input_fasta: str, force: bool = False) -> bool:
        """Verify input FASTA matches recorded hash.

        Args:
            input_fasta: Path to input FASTA file
            force: If True, allow hash mismatch with warning

        Returns:
            True if hash matches or force=True

        Raises:
            ValueError: If hash doesn't match and force=False
        """
        current_hash = compute_file_hash(Path(input_fasta))

        if current_hash == self.input.fasta_hash:
            return True

        if force:
            logger.warning(f"Input FASTA hash mismatch (forced continuation)")
            logger.warning(f"  Expected: {self.input.fasta_hash}")
            logger.warning(f"  Current:  {current_hash}")
            return True

        raise ValueError(
            f"Input FASTA has changed since initial run.\n"
            f"  Expected hash: {self.input.fasta_hash}\n"
            f"  Current hash:  {current_hash}\n"
            f"Use --force-input-change to override (may cause inconsistencies)."
        )

    def get_current_stage_directory(self, output_dir: Path) -> Path:
        """Get stage directory for current stage.

        Args:
            output_dir: Output directory for decompose run

        Returns:
            Path to current stage directory
        """
        if self.stage == "initial_clustering":
            return output_dir / self.initial_clustering.stage_directory
        elif self.stage == "finalized":
            return output_dir / "clusters" / "latest"
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def can_continue_clustering(self) -> bool:
        """Check if more initial clustering can be added.

        Returns:
            True if initial clustering is incomplete and can be continued
        """
        return (self.stage == "initial_clustering" and
                not self.initial_clustering.completed)

    def update_stage_completion(self, stage: str, **stats) -> None:
        """Mark stage complete and record statistics.

        Args:
            stage: Stage name to update
            **stats: Statistics to record
        """
        if stage == "initial_clustering":
            self.initial_clustering.completed = True
            for key, value in stats.items():
                if hasattr(self.initial_clustering, key):
                    setattr(self.initial_clustering, key, value)
            self.stage = "initial_clustering"

        elif stage == "finalized":
            self.finalized.completed = True
            self.stage = "finalized"
            self.status = "completed"

    def add_command(self, command: str) -> None:
        """Add command to history.

        Args:
            command: Command line string to record
        """
        self.command_history.append(command)


class StateManager:
    """Manages state persistence and cluster reconstruction from FASTA files."""

    def __init__(self, output_dir: Path):
        """Initialize state manager.

        Args:
            output_dir: Output directory for decompose run
        """
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "state.json"

    def checkpoint(self, state: DecomposeState) -> None:
        """Save checkpoint (called periodically during clustering).

        Args:
            state: Current state to save
        """
        state.save(self.output_dir)

    def load_clusters_from_stage_directory(self, stage_dir: Path) -> Dict[str, List[str]]:
        """Rebuild cluster dict by reading FASTA files from stage directory.

        Args:
            stage_dir: Path to stage directory (e.g., output_dir/work/initial)

        Returns:
            Dict mapping cluster_id -> list of sequence headers

        Note:
            This reconstructs cluster membership from FASTA files (source of truth).
            Cluster IDs are extracted from filenames (e.g., cluster_00001I.fasta).
        """
        from Bio import SeqIO

        clusters = {}

        if not stage_dir.exists():
            logger.warning(f"Stage directory does not exist: {stage_dir}")
            return clusters

        # Find all cluster FASTA files in stage directory
        cluster_files = sorted(stage_dir.glob("cluster_*.fasta"))

        for cluster_file in cluster_files:
            # Extract cluster ID from filename (e.g., "cluster_00001I.fasta" -> "cluster_00001I")
            cluster_id = cluster_file.stem

            # Read sequence headers from FASTA
            headers = []
            for record in SeqIO.parse(cluster_file, "fasta"):
                headers.append(record.id)

            if headers:
                clusters[cluster_id] = headers
                logger.debug(f"Loaded cluster {cluster_id}: {len(headers)} sequences")

        logger.info(f"Loaded {len(clusters)} clusters from {stage_dir}")
        return clusters

    def load_unassigned_sequences(self, unassigned_file: str) -> List[str]:
        """Load unassigned sequence headers from file.

        Args:
            unassigned_file: Filename of unassigned sequences

        Returns:
            List of sequence headers
        """
        from Bio import SeqIO

        unassigned_path = self.output_dir / unassigned_file

        if not unassigned_path.exists():
            logger.debug(f"No unassigned file found: {unassigned_file}")
            return []

        headers = []
        for record in SeqIO.parse(unassigned_path, "fasta"):
            headers.append(record.id)

        logger.info(f"Loaded {len(headers)} unassigned sequences")
        return headers

    def rebuild_assignment_tracker(self,
                                   clusters: Dict[str, List[str]],
                                   all_headers: List[str]) -> 'AssignmentTracker':
        """Reconstruct AssignmentTracker from FASTA files.

        Args:
            clusters: Cluster dict loaded from FASTA files
            all_headers: All sequence headers from input

        Returns:
            Reconstructed AssignmentTracker

        Note:
            This rebuilds the assignment tracking state from FASTA files.
            Conflicts are detected if a sequence appears in multiple clusters.
        """
        from .decompose import AssignmentTracker

        tracker = AssignmentTracker()

        # Assign sequences to clusters (iteration number unknown, use 0)
        for cluster_id, seq_headers in clusters.items():
            tracker.assign_sequences(seq_headers, cluster_id, iteration=0)

        logger.info(f"Rebuilt assignment tracker: {len(tracker.assigned_sequences)} assigned sequences")

        # Check for conflicts
        conflicts = tracker.get_conflicts()
        if conflicts:
            logger.warning(f"Detected {len(conflicts)} conflicts in loaded clusters")

        return tracker

    def detect_conflicts(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Find sequences appearing in multiple cluster files.

        Args:
            clusters: Cluster dict loaded from FASTA files

        Returns:
            Dict mapping sequence_id -> list of cluster_ids
        """
        sequence_to_clusters = {}

        for cluster_id, headers in clusters.items():
            for header in headers:
                if header not in sequence_to_clusters:
                    sequence_to_clusters[header] = []
                sequence_to_clusters[header].append(cluster_id)

        # Filter to only conflicted sequences (appears in > 1 cluster)
        conflicts = {
            seq_id: cluster_ids
            for seq_id, cluster_ids in sequence_to_clusters.items()
            if len(cluster_ids) > 1
        }

        return conflicts

    def save_stage_fasta(self,
                        clusters: Dict[str, List[str]],
                        sequences: List[str],
                        headers: List[str],
                        stage_dir: Path) -> None:
        """Save cluster FASTA files for a stage.

        Args:
            clusters: Dict mapping cluster_id -> list of sequence headers
            sequences: Full sequence list
            headers: Full header list (sequence IDs)
            stage_dir: Path to stage directory where clusters will be saved
        """
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO

        # Create stage directory if it doesn't exist
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Create header to index mapping
        header_to_idx = {header: i for i, header in enumerate(headers)}

        # Save each cluster to a separate FASTA file
        for cluster_id, cluster_headers in clusters.items():
            # Filename is just cluster_id.fasta (e.g., cluster_00001I.fasta)
            filename = f"{cluster_id}.fasta"
            cluster_file = stage_dir / filename

            records = []
            for header in cluster_headers:
                if header in header_to_idx:
                    seq_idx = header_to_idx[header]
                    record = SeqRecord(
                        Seq(sequences[seq_idx]),
                        id=header,
                        description=""
                    )
                    records.append(record)

            if records:
                with open(cluster_file, 'w') as f:
                    SeqIO.write(records, f, "fasta-2line")
                logger.debug(f"Wrote {len(records)} sequences to {cluster_file}")

        logger.debug(f"Saved {len(clusters)} cluster files to {stage_dir}")

    def save_stage_results(self,
                          results: 'DecomposeResults',
                          stage_dir: Path,
                          sequences: List[str],
                          headers: List[str],
                          hash_to_headers: Dict[str, List[str]]) -> None:
        """Save results from a processing stage to FASTA files.

        Args:
            results: DecomposeResults object with clusters
            stage_dir: Path to stage directory where clusters will be saved
            sequences: Full sequence list
            headers: Hash IDs corresponding to sequences
            hash_to_headers: Mapping from hash IDs to original headers
        """
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO

        # Create stage directory if it doesn't exist
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Create header to index mapping (using hash IDs)
        header_to_idx = {header: i for i, header in enumerate(headers)}

        # Save each cluster to a separate FASTA file
        for cluster_id, cluster_headers in results.clusters.items():
            # Filename is just cluster_id.fasta (e.g., cluster_00001C.fasta)
            filename = f"{cluster_id}.fasta"
            cluster_file = stage_dir / filename

            records = []
            for header in cluster_headers:
                if header in header_to_idx:
                    seq_idx = header_to_idx[header]
                    # Expand hash ID to all original headers
                    if header in hash_to_headers:
                        original_headers = hash_to_headers[header]
                    else:
                        original_headers = [header]

                    # Write a record for each original header (handles duplicates)
                    for orig_header in original_headers:
                        record = SeqRecord(
                            Seq(sequences[seq_idx]),
                            id=orig_header,
                            description=""
                        )
                        records.append(record)

            if records:
                with open(cluster_file, 'w') as f:
                    SeqIO.write(records, f, "fasta-2line")
                logger.debug(f"Wrote {len(records)} sequences to {cluster_file}")

        logger.debug(f"Saved {len(results.clusters)} cluster files to {stage_dir}")

        # Save unassigned sequences if any
        if results.unassigned:
            unassigned_file = stage_dir / "unassigned.fasta"
            records = []
            for header in results.unassigned:
                if header in header_to_idx:
                    seq_idx = header_to_idx[header]
                    # Expand hash ID to all original headers
                    if header in hash_to_headers:
                        original_headers = hash_to_headers[header]
                    else:
                        original_headers = [header]

                    for orig_header in original_headers:
                        record = SeqRecord(
                            Seq(sequences[seq_idx]),
                            id=orig_header,
                            description=""
                        )
                        records.append(record)

            if records:
                with open(unassigned_file, 'w') as f:
                    SeqIO.write(records, f, "fasta-2line")
                logger.info(f"Wrote {len(records)} unassigned sequences to {unassigned_file}")


def create_initial_state(input_fasta: str,
                        parameters: Dict[str, Any],
                        command: str,
                        version: str = "0.5.0") -> DecomposeState:
    """Create initial state for new decompose run.

    Args:
        input_fasta: Path to input FASTA file
        parameters: Clustering parameters
        command: Command line string
        version: gaphack version

    Returns:
        New DecomposeState object
    """
    from .utils import load_sequences_with_deduplication

    # Compute input hash
    fasta_path = Path(input_fasta).resolve()
    fasta_hash = compute_file_hash(fasta_path)

    # Count sequences
    sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(str(fasta_path))
    total_headers = sum(len(headers) for headers in hash_to_headers.values())

    input_info = InputInfo(
        fasta_path=str(fasta_path),
        fasta_hash=fasta_hash,
        total_sequences=total_headers,
        deduplicated_sequences=len(sequences)
    )

    # Create empty stage info with new directory structure
    initial = InitialClusteringStage(
        stage_directory="work/initial",
        unassigned_file="work/initial/unassigned.fasta"
    )
    finalized_stage = FinalizedStage()

    state = DecomposeState(
        version=version,
        status="in_progress",
        stage="initial_clustering",
        input=input_info,
        parameters=parameters,
        initial_clustering=initial,
        finalized=finalized_stage,
        start_time=datetime.now().isoformat(),
        gaphack_version=version
    )

    state.add_command(command)

    return state