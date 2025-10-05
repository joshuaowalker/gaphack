"""
Tests for distance provider architecture (MSA-cached).
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from gaphack.distance_providers import (
    DistanceProvider,
    MSACachedDistanceProvider,
    MSAAlignmentError
)


class TestDistanceProviderABC:
    """Test suite for DistanceProvider abstract base class."""

    def test_abstract_methods(self):
        """Test that DistanceProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DistanceProvider()


class TestMSACachedDistanceProvider:
    """Test suite for MSACachedDistanceProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATCGATCGATCG",
            "ATCGATCGATCC",
            "TTTTGGGGCCCC",
            "TTTTGGGGCCCT"
        ]

    @patch('gaphack.utils.run_spoa_msa')
    @patch('gaphack.utils.replace_terminal_gaps')
    def test_initialization_with_msa(self, mock_replace_gaps, mock_spoa):
        """Test MSACachedDistanceProvider initialization with successful MSA."""
        # Mock SPOA success
        aligned = [
            "ATCGATCGATCG",
            "ATCGATCGATCC",
            "TTTTGGGGCCCC",
            "TTTTGGGGCCCT"
        ]
        mock_spoa.return_value = aligned
        mock_replace_gaps.return_value = aligned

        provider = MSACachedDistanceProvider(self.test_sequences)

        assert provider.n == 4
        assert provider.sequences == self.test_sequences
        assert hasattr(provider, 'aligned_sequences')
        mock_spoa.assert_called_once()

    @patch('gaphack.utils.run_spoa_msa')
    def test_initialization_raises_on_spoa_failure(self, mock_spoa):
        """Test MSACachedDistanceProvider raises exception when SPOA fails."""
        # Mock SPOA failure
        mock_spoa.return_value = None

        with pytest.raises(MSAAlignmentError) as exc_info:
            MSACachedDistanceProvider(self.test_sequences)

        assert "SPOA failed" in str(exc_info.value)
        mock_spoa.assert_called_once()

    @patch('gaphack.utils.run_spoa_msa')
    @patch('gaphack.utils.compute_msa_distance')
    @patch('gaphack.utils.replace_terminal_gaps')
    def test_get_distance_with_msa(self, mock_replace_gaps, mock_msa_distance, mock_spoa):
        """Test distance calculation with MSA."""
        # Mock SPOA success
        aligned = ["ATCG", "ATCC", "TTGG", "TTGT"]
        mock_spoa.return_value = aligned
        mock_replace_gaps.return_value = aligned
        mock_msa_distance.return_value = 0.15

        provider = MSACachedDistanceProvider(self.test_sequences)

        # Get distance
        distance = provider.get_distance(0, 1)
        assert distance == 0.15
        mock_msa_distance.assert_called_once()

    @patch('gaphack.utils.run_spoa_msa')
    @patch('gaphack.utils.replace_terminal_gaps')
    def test_get_distance_caching(self, mock_replace_gaps, mock_spoa):
        """Test that distances are cached."""
        # Mock SPOA success
        aligned = ["ATCG", "ATCC", "TTGG", "TTGT"]
        mock_spoa.return_value = aligned
        mock_replace_gaps.return_value = aligned

        with patch('gaphack.utils.compute_msa_distance') as mock_msa_distance:
            mock_msa_distance.return_value = 0.15

            provider = MSACachedDistanceProvider(self.test_sequences)

            # First call should compute
            distance1 = provider.get_distance(0, 1)
            assert distance1 == 0.15
            assert mock_msa_distance.call_count == 1

            # Second call should use cache
            distance2 = provider.get_distance(0, 1)
            assert distance2 == 0.15
            assert mock_msa_distance.call_count == 1  # No additional calls

    @patch('gaphack.utils.run_spoa_msa')
    @patch('gaphack.utils.replace_terminal_gaps')
    def test_get_distance_self(self, mock_replace_gaps, mock_spoa):
        """Test distance to self."""
        aligned = ["ATCG", "ATCC"]
        mock_spoa.return_value = aligned
        mock_replace_gaps.return_value = aligned

        provider = MSACachedDistanceProvider(self.test_sequences[:2])
        assert provider.get_distance(0, 0) == 0.0
        assert provider.get_distance(1, 1) == 0.0


    @patch('gaphack.utils.run_spoa_msa')
    @patch('gaphack.utils.replace_terminal_gaps')
    def test_build_distance_matrix(self, mock_replace_gaps, mock_spoa):
        """Test building full distance matrix from MSA."""
        aligned = ["ATCG", "ATCC", "TTGG"]
        mock_spoa.return_value = aligned
        mock_replace_gaps.return_value = aligned

        with patch('gaphack.utils.compute_msa_distance') as mock_msa_distance:
            mock_msa_distance.return_value = 0.15

            provider = MSACachedDistanceProvider(self.test_sequences[:3])
            matrix = provider.build_distance_matrix()

            assert matrix.shape == (3, 3)
            assert np.all(np.diag(matrix) == 0.0)  # Diagonal should be zero
            assert matrix[0, 1] == matrix[1, 0]  # Should be symmetric
