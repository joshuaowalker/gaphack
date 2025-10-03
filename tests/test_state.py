"""Tests for state management and incremental restart functionality."""

import json
import pytest
import tempfile
from pathlib import Path
from gaphack.state import (
    DecomposeState,
    StateManager,
    InputInfo,
    InitialClusteringStage,
    ConflictResolutionStage,
    CloseClusterRefinementStage,
    FinalizedStage,
    compute_file_hash,
    create_initial_state
)


class TestComputeFileHash:
    """Tests for file hashing."""

    def test_hash_consistency(self, tmp_path):
        """Test that hashing the same file twice produces same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        hash1 = compute_file_hash(str(test_file))
        hash2 = compute_file_hash(str(test_file))

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_different_content_different_hash(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))

        assert hash1 != hash2


class TestDecomposeState:
    """Tests for DecomposeState serialization."""

    def test_state_serialization_deserialization(self, tmp_path):
        """Test that state can be saved and loaded."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a state
        state = DecomposeState(
            version="0.4.0",
            status="in_progress",
            stage="initial_clustering",
            input=InputInfo(
                fasta_path="input.fasta",
                fasta_hash="abc123",
                total_sequences=100,
                deduplicated_sequences=95
            ),
            parameters={
                "min_split": 0.005,
                "max_lump": 0.02,
                "target_percentile": 95
            },
            initial_clustering=InitialClusteringStage(),
            conflict_resolution=ConflictResolutionStage(),
            close_cluster_refinement=CloseClusterRefinementStage(),
            finalized=FinalizedStage(),
            command_history=["gaphack-decompose input.fasta -o output"],
            start_time="2025-01-01T00:00:00",
            gaphack_version="0.4.0"
        )

        # Save state
        state.save(output_dir)

        # Verify state file exists
        state_file = output_dir / "state.json"
        assert state_file.exists()

        # Load state
        loaded_state = DecomposeState.load(output_dir)

        # Verify loaded state matches original
        assert loaded_state.version == state.version
        assert loaded_state.status == state.status
        assert loaded_state.stage == state.stage
        assert loaded_state.input.fasta_path == state.input.fasta_path
        assert loaded_state.input.fasta_hash == state.input.fasta_hash
        assert loaded_state.parameters == state.parameters
        assert loaded_state.command_history == state.command_history

    def test_state_json_structure(self, tmp_path):
        """Test that saved JSON has expected structure."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        state = DecomposeState(
            version="0.4.0",
            status="in_progress",
            stage="initial_clustering",
            input=InputInfo(
                fasta_path="input.fasta",
                fasta_hash="abc123",
                total_sequences=100,
                deduplicated_sequences=95
            ),
            parameters={"min_split": 0.005},
            initial_clustering=InitialClusteringStage(),
            conflict_resolution=ConflictResolutionStage(),
            close_cluster_refinement=CloseClusterRefinementStage(),
            finalized=FinalizedStage(),
            command_history=[],
            start_time="2025-01-01T00:00:00",
            gaphack_version="0.4.0"
        )

        state.save(output_dir)

        # Load JSON directly
        state_file = output_dir / "state.json"
        with open(state_file) as f:
            data = json.load(f)

        # Check structure
        assert "version" in data
        assert "status" in data
        assert "stage" in data
        assert "input" in data
        assert "parameters" in data
        assert "stages" in data
        assert "initial_clustering" in data["stages"]
        assert "conflict_resolution" in data["stages"]
        assert "close_cluster_refinement" in data["stages"]
        assert "finalized" in data["stages"]
        assert "metadata" in data
        assert "command_history" in data["metadata"]

    def test_atomic_save(self, tmp_path):
        """Test that state saving is atomic (temp + rename)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        state = DecomposeState(
            version="0.4.0",
            status="in_progress",
            stage="initial_clustering",
            input=InputInfo(
                fasta_path="input.fasta",
                fasta_hash="abc123",
                total_sequences=100,
                deduplicated_sequences=95
            ),
            parameters={},
            initial_clustering=InitialClusteringStage(),
            conflict_resolution=ConflictResolutionStage(),
            close_cluster_refinement=CloseClusterRefinementStage(),
            finalized=FinalizedStage(),
            command_history=[],
            start_time="2025-01-01T00:00:00",
            gaphack_version="0.4.0"
        )

        # Save state
        state.save(output_dir)

        # Verify no temp files left behind
        temp_files = list(output_dir.glob("state.json.tmp*"))
        assert len(temp_files) == 0

        # Verify state file exists
        state_file = output_dir / "state.json"
        assert state_file.exists()

    def test_validate_input_hash(self, tmp_path):
        """Test input file hash validation."""
        # Create input file
        input_file = tmp_path / "input.fasta"
        input_file.write_text(">seq1\nATCG\n>seq2\nGGCC\n")

        # Create state with correct hash
        file_hash = compute_file_hash(input_file)
        state = DecomposeState(
            version="0.4.0",
            status="in_progress",
            stage="initial_clustering",
            input=InputInfo(
                fasta_path=str(input_file),
                fasta_hash=file_hash,
                total_sequences=2,
                deduplicated_sequences=2
            ),
            parameters={},
            initial_clustering=InitialClusteringStage(),
            conflict_resolution=ConflictResolutionStage(),
            close_cluster_refinement=CloseClusterRefinementStage(),
            finalized=FinalizedStage(),
            command_history=[],
            start_time="2025-01-01T00:00:00",
            gaphack_version="0.4.0"
        )

        # Validation should pass
        assert state.validate_input_hash(str(input_file), force=False)

        # Modify file
        input_file.write_text(">seq1\nATCG\n>seq2\nGGCC\n>seq3\nTTAA\n")

        # Validation should fail
        with pytest.raises(ValueError, match="Input FASTA has changed"):
            state.validate_input_hash(str(input_file), force=False)

        # Force should allow it (returns True with warning)
        assert state.validate_input_hash(str(input_file), force=True)


class TestCreateInitialState:
    """Tests for create_initial_state helper."""

    def test_create_initial_state(self, tmp_path):
        """Test creating initial state from parameters."""
        # Create input file
        input_file = tmp_path / "input.fasta"
        input_file.write_text(">seq1\nATCG\n>seq2\nGGCC\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        parameters = {
            "min_split": 0.005,
            "max_lump": 0.02,
            "target_percentile": 95,
            "blast_max_hits": 1000,
            "blast_evalue": 1e-5,
            "min_identity": 90.0,
            "max_clusters": 10,
            "max_sequences": 100,
            "resolve_conflicts": True,
            "refine_close_clusters": True,
            "close_cluster_threshold": 0.015
        }

        state = create_initial_state(
            input_fasta=str(input_file),
            parameters=parameters,
            command="gaphack-decompose input.fasta -o output",
            version="0.4.0"
        )

        # Check basic properties
        assert state.version == "0.4.0"
        assert state.status == "in_progress"
        assert state.stage == "initial_clustering"
        assert state.input.fasta_path == str(input_file)
        assert state.input.total_sequences == 2

        # Check parameters
        assert state.parameters["min_split"] == 0.005
        assert state.parameters["max_lump"] == 0.02
        assert state.parameters["target_percentile"] == 95
        assert state.parameters["max_clusters"] == 10
        assert state.parameters["max_sequences"] == 100

        # Check command history
        assert len(state.command_history) == 1
        assert "gaphack-decompose" in state.command_history[0]


class TestStateManager:
    """Tests for StateManager."""

    def test_save_and_load_clusters(self, tmp_path):
        """Test saving and loading cluster FASTA files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test sequences
        sequences = ["ATCG", "GGCC", "TTAA", "CCGG"]
        headers = ["seq1", "seq2", "seq3", "seq4"]

        state_manager = StateManager(output_dir)

        # Create clusters
        clusters = {
            "cluster_001": ["seq1", "seq2"],
            "cluster_002": ["seq3", "seq4"]
        }

        # Create stage directory
        stage_dir = output_dir / "work/initial"

        # Save clusters
        state_manager.save_stage_fasta(clusters, sequences, headers, stage_dir)

        # Verify files exist
        assert (stage_dir / "cluster_001.fasta").exists()
        assert (stage_dir / "cluster_002.fasta").exists()

        # Load clusters back
        loaded_clusters = state_manager.load_clusters_from_stage_directory(stage_dir)

        # Verify loaded clusters match original
        # Note: cluster IDs are loaded as-is from filenames
        assert len(loaded_clusters) == 2
        assert set(loaded_clusters["cluster_001"]) == set(clusters["cluster_001"])
        assert set(loaded_clusters["cluster_002"]) == set(clusters["cluster_002"])

    def test_rebuild_assignment_tracker(self, tmp_path):
        """Test rebuilding AssignmentTracker from clusters."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sequences = ["ATCG", "GGCC", "TTAA"]
        headers = ["seq1", "seq2", "seq3"]

        state_manager = StateManager(output_dir)

        clusters = {
            "cluster_001": ["seq1", "seq2"],
            "cluster_002": ["seq3"]
        }

        # Rebuild assignment tracker
        tracker = state_manager.rebuild_assignment_tracker(clusters, headers)

        # Check assignments
        assert tracker.is_assigned("seq1")
        assert tracker.is_assigned("seq2")
        assert tracker.is_assigned("seq3")

        # Check that seq1 and seq2 are in cluster_001
        single_assignments = tracker.get_single_assignments()
        assert single_assignments["seq1"] == "cluster_001"
        assert single_assignments["seq2"] == "cluster_001"
        assert single_assignments["seq3"] == "cluster_002"

    def test_detect_conflicts(self, tmp_path):
        """Test conflict detection."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sequences = ["ATCG", "GGCC"]
        headers = ["seq1", "seq2"]

        state_manager = StateManager(output_dir)

        # Create clusters with conflict (seq1 in both)
        clusters = {
            "cluster_001": ["seq1", "seq2"],
            "cluster_002": ["seq1"]
        }

        conflicts = state_manager.detect_conflicts(clusters)

        # Check that seq1 is detected as conflict
        assert "seq1" in conflicts
        assert set(conflicts["seq1"]) == {"cluster_001", "cluster_002"}

        # seq2 should not be a conflict
        assert "seq2" not in conflicts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])