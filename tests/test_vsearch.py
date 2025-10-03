"""Tests for vsearch integration."""

import pytest
import tempfile
import subprocess
from pathlib import Path

from gaphack.vsearch_neighborhood import VsearchNeighborhoodFinder


def check_vsearch_available():
    """Check if vsearch is installed and available."""
    try:
        result = subprocess.run(['vsearch', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not check_vsearch_available(), reason="vsearch not installed")
class TestVsearchNeighborhoodFinder:
    """Tests for VsearchNeighborhoodFinder class."""

    def test_init(self):
        """Test initialization of VsearchNeighborhoodFinder."""
        sequences = ["ATCGATCG", "GCTAGCTA"]
        headers = ["seq1", "seq2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            finder = VsearchNeighborhoodFinder(sequences, headers, output_dir=Path(tmpdir))

            assert finder.sequences == sequences
            assert finder.headers == headers
            assert len(finder.sequence_lookup) == 2
            assert finder.vsearch_db_path.exists()

            finder.cleanup()

    def test_find_neighborhood(self):
        """Test finding neighborhoods with vsearch."""
        sequences = [
            "ATCGATCGATCG",  # seq1
            "ATCGATCGATCT",  # seq2 (1 mismatch - different)
            "GCTAGCTAGCTA",  # seq3 (very different)
        ]
        headers = ["seq1", "seq2", "seq3"]

        with tempfile.TemporaryDirectory() as tmpdir:
            finder = VsearchNeighborhoodFinder(sequences, headers, output_dir=Path(tmpdir))

            # Find neighborhood for seq1
            neighborhood = finder.find_neighborhood(["seq1"], max_hits=10, min_identity=90.0)

            # Should include at least seq1
            assert "seq1" in neighborhood
            # seq2 and seq3 depend on alignment quality

            # Verify it's a list of headers
            assert isinstance(neighborhood, list)
            assert all(isinstance(h, str) for h in neighborhood)

            finder.cleanup()

    def test_cleanup(self):
        """Test cleanup of vsearch database."""
        sequences = ["ATCGATCG"]
        headers = ["seq1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            finder = VsearchNeighborhoodFinder(sequences, headers, output_dir=Path(tmpdir))
            db_path = finder.vsearch_db_path

            assert db_path.exists()

            finder.cleanup()

            assert not db_path.exists()

    def test_hash_collision_handling(self):
        """Test handling of hash collisions."""
        sequences = ["A" * 100, "A" * 100]  # Identical sequences
        headers = ["seq1", "seq2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            finder = VsearchNeighborhoodFinder(sequences, headers, output_dir=Path(tmpdir))

            # Both sequences should map to same hash
            assert len(finder.sequence_lookup) >= 1

            finder.cleanup()


@pytest.mark.skipif(not check_vsearch_available(), reason="vsearch not installed")
class TestVsearchVsBlast:
    """Comparative tests between vsearch and BLAST."""

    def test_vsearch_finds_similar_neighborhoods_to_blast(self):
        """Test that vsearch finds similar neighborhoods to BLAST."""
        from gaphack.blast_neighborhood import BlastNeighborhoodFinder

        # Use longer, more diverse sequences for better testing
        sequences = [
            "ATCGATCGATCGATCGATCGATCG",  # seq1
            "ATCGATCGATCGATCGATCGATCT",  # seq2 (1 mismatch at end)
            "ATCGATCGTTCGATCGATCGATCG",  # seq3 (1 mismatch in middle)
            "GCTAGCTAGCTAGCTAGCTAGCTA",  # seq4 (completely different)
        ]
        headers = ["seq1", "seq2", "seq3", "seq4"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with vsearch
            vsearch_finder = VsearchNeighborhoodFinder(sequences, headers, output_dir=Path(tmpdir) / "vsearch")
            vsearch_neighborhood = vsearch_finder.find_neighborhood(["seq1"], max_hits=10, min_identity=90.0)
            vsearch_finder.cleanup()

            # Test with BLAST
            blast_finder = BlastNeighborhoodFinder(sequences, headers, output_dir=Path(tmpdir) / "blast")
            blast_neighborhood = blast_finder.find_neighborhood(["seq1"], max_hits=10, min_identity=90.0)
            blast_finder.cleanup()

            # Both should find at least seq1
            assert "seq1" in vsearch_neighborhood
            assert "seq1" in blast_neighborhood

            # Neighborhoods should both be non-empty
            assert len(vsearch_neighborhood) > 0
            assert len(blast_neighborhood) > 0

            # Neighborhoods should be similar in size (within factor of 3)
            # Allow more variance since algorithms differ
            size_ratio = len(vsearch_neighborhood) / len(blast_neighborhood)
            assert 0.33 <= size_ratio <= 3.0, f"Neighborhood sizes differ too much: vsearch={len(vsearch_neighborhood)}, blast={len(blast_neighborhood)}"
