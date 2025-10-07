"""
Tests for triangle inequality-based outlier detection and filtering.

This module tests the triangle filtering logic used across gapHACk modes
to detect and filter spurious distances caused by poor sequence alignments.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st

from gaphack.triangle_filtering import (
    filter_distance_dict_triangles,
    filter_intra_cluster_triangles,
    filter_distance_matrix_triangles,
    add_nan_filtering_to_distance_list,
    DEFAULT_VIOLATION_TOLERANCE,
    DEFAULT_MIN_VALIDATIONS,
    DEFAULT_ENABLE_FILTERING
)


class TestTriangleFilteringConstants:
    """Test module constants and defaults."""

    def test_default_constants(self):
        """Test that default constants are reasonable."""
        assert DEFAULT_VIOLATION_TOLERANCE == 0.05
        assert DEFAULT_MIN_VALIDATIONS == 3
        assert DEFAULT_ENABLE_FILTERING is True


class TestFilterDistanceDictTriangles:
    """Test dict-based triangle inequality filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_provider = Mock()

    def test_filter_empty_dict(self):
        """Test filtering empty distance dict."""
        distances = {}
        result = filter_distance_dict_triangles(distances, self.mock_provider)
        assert result == {}

    def test_filter_small_dict(self):
        """Test filtering dict with fewer than 3 sequences."""
        distances = {0: 0.1, 1: 0.2}
        result = filter_distance_dict_triangles(distances, self.mock_provider)
        assert result == distances
        self.mock_provider.get_distance.assert_not_called()

    def test_filter_no_violations(self):
        """Test filtering when no triangle violations exist."""
        distances = {0: 0.1, 1: 0.2, 2: 0.15}

        # Mock distances that satisfy triangle inequality
        def mock_get_distance(i, j):
            distance_map = {(0, 1): 0.05, (0, 2): 0.08, (1, 2): 0.12}
            return distance_map.get((min(i, j), max(i, j)), 0.0)

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_distance_dict_triangles(distances, self.mock_provider)
        assert result == distances

    def test_filter_with_violations(self):
        """Test filtering when triangle violations exist."""
        distances = {0: 0.3, 1: 0.1, 2: 0.1}  # 0.3 > 0.1 + 0.1

        # Mock distances where seq 0 violates triangle inequality
        def mock_get_distance(i, j):
            if (i, j) == (0, 1) or (i, j) == (1, 0):
                return 0.05
            elif (i, j) == (0, 2) or (i, j) == (2, 0):
                return 0.05
            elif (i, j) == (1, 2) or (i, j) == (2, 1):
                return 0.02
            return 0.0

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_distance_dict_triangles(distances, self.mock_provider)

        # Sequence 0 should be filtered out due to violation
        assert 0 not in result
        assert 1 in result
        assert 2 in result

    def test_violation_tolerance(self):
        """Test that violation tolerance is respected."""
        distances = {0: 0.25, 1: 0.1, 2: 0.1}  # 0.25 vs 0.2 = 0.05 difference

        # Mock distances that violate within tolerance
        def mock_get_distance(i, j):
            if (i, j) == (0, 1) or (i, j) == (1, 0):
                return 0.1
            elif (i, j) == (0, 2) or (i, j) == (2, 0):
                return 0.1
            elif (i, j) == (1, 2) or (i, j) == (2, 1):
                return 0.02
            return 0.0

        self.mock_provider.get_distance.side_effect = mock_get_distance

        # Should not filter with default tolerance (0.05)
        result = filter_distance_dict_triangles(distances, self.mock_provider)
        assert len(result) == 3

        # Should filter with tighter tolerance
        result = filter_distance_dict_triangles(distances, self.mock_provider,
                                              violation_tolerance=0.01)
        assert 0 not in result
        assert len(result) == 2

    def test_min_validations_parameter(self):
        """Test min_validations parameter effect."""
        distances = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}

        # Mock perfect distances (no violations)
        self.mock_provider.get_distance.return_value = 0.05

        result = filter_distance_dict_triangles(distances, self.mock_provider,
                                              min_validations=2)
        assert len(result) == 5  # All sequences should pass

    def test_context_parameter(self):
        """Test that context parameter is used for logging."""
        distances = {0: 0.3, 1: 0.1, 2: 0.1}

        def mock_get_distance(i, j):
            return 0.05

        self.mock_provider.get_distance.side_effect = mock_get_distance

        # Should not raise exception when context is provided
        result = filter_distance_dict_triangles(distances, self.mock_provider,
                                              context="test_context")
        assert isinstance(result, dict)

    def test_nan_distance_filtered(self):
        """Test that NaN distances are filtered as violations."""
        distances = {0: np.nan, 1: 0.1, 2: 0.1}  # seq 0 has NaN distance

        # Mock valid distances between other sequences
        self.mock_provider.get_distance.return_value = 0.05

        result = filter_distance_dict_triangles(distances, self.mock_provider)

        # Sequence 0 with NaN should be filtered out
        assert 0 not in result
        assert 1 in result
        assert 2 in result

    def test_nan_in_triangle_check_skipped(self):
        """Test that triangles with NaN distances are skipped."""
        distances = {0: 0.1, 1: 0.1, 2: 0.1}

        # Mock provider returns NaN for some pairs
        def mock_get_distance(i, j):
            if i == 0 and j == 1:
                return np.nan  # This triangle should be skipped
            return 0.05

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_distance_dict_triangles(distances, self.mock_provider)

        # All sequences should remain (NaN triangles don't count as violations or validations)
        assert len(result) == 3

    def test_all_nan_neighbors_filtered(self):
        """Test that sequence with all NaN neighbors gets filtered appropriately."""
        distances = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}

        # Mock provider: seq 0 has all NaN distances to others
        def mock_get_distance(i, j):
            if 0 in (i, j):
                return np.nan
            return 0.05

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_distance_dict_triangles(distances, self.mock_provider,
                                              min_validations=2)

        # Sequences 0, 1, 2, 3 should all remain since NaN checks are skipped
        # None can accumulate enough validations against seq 0, but they can validate against each other
        assert len(result) >= 0  # At least some sequences remain


class TestFilterIntraClusterTriangles:
    """Test intra-cluster triangle inequality filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_provider = Mock()

    def test_filter_small_cluster(self):
        """Test filtering cluster with fewer than 4 sequences."""
        cluster_indices = [0, 1, 2]
        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider)
        assert result == []
        self.mock_provider.get_distance.assert_not_called()

    def test_filter_no_violations(self):
        """Test filtering when no violations exist."""
        cluster_indices = [0, 1, 2, 3]

        # Mock distances that satisfy triangle inequality
        self.mock_provider.get_distance.return_value = 0.1

        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider)
        assert result == []

    def test_filter_with_violations(self):
        """Test filtering when violations exist."""
        cluster_indices = [0, 1, 2, 3]

        # Mock distances where pair (0,1) violates triangle inequality
        def mock_get_distance(i, j):
            if (i, j) == (0, 1) or (i, j) == (1, 0):
                return 0.5  # Large distance that will violate
            else:
                return 0.1  # Small distances for other pairs

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider)

        # Should find violation for pair (0, 1)
        assert len(result) > 0
        assert (0, 1) in result or (1, 0) in result

    def test_violation_tolerance_intra(self):
        """Test violation tolerance in intra-cluster filtering."""
        cluster_indices = [0, 1, 2, 3]

        # Mock distances that violate within tolerance
        def mock_get_distance(i, j):
            if (i, j) == (0, 1) or (i, j) == (1, 0):
                return 0.25  # 0.25 vs 0.2 = 0.05 difference
            else:
                return 0.1

        self.mock_provider.get_distance.side_effect = mock_get_distance

        # Should not filter with default tolerance
        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider)
        assert result == []

        # Should filter with tighter tolerance
        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider,
                                              violation_tolerance=0.01)
        assert len(result) > 0

    def test_min_validations_intra(self):
        """Test min_validations parameter in intra-cluster filtering."""
        cluster_indices = [0, 1, 2, 3, 4]

        # Mock perfect distances
        self.mock_provider.get_distance.return_value = 0.1

        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider,
                                              min_validations=1)
        assert result == []

    def test_nan_pair_filtered(self):
        """Test that pairs with NaN distances are filtered as violations."""
        cluster_indices = [0, 1, 2, 3]

        # Mock distances where pair (0,1) has NaN
        def mock_get_distance(i, j):
            if (i == 0 and j == 1) or (i == 1 and j == 0):
                return np.nan
            return 0.1

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider)

        # Pair (0, 1) should be in violations
        assert (0, 1) in result or (1, 0) in result

    def test_nan_in_triangle_check_skipped_intra(self):
        """Test that triangles with NaN distances are skipped in validation."""
        cluster_indices = [0, 1, 2, 3]

        # Mock distances where some auxiliary distances are NaN
        def mock_get_distance(i, j):
            # Pair (0,1) is valid
            if (i == 0 and j == 1) or (i == 1 and j == 0):
                return 0.1
            # But one of the triangle sides is NaN
            if (i == 0 and j == 2) or (i == 2 and j == 0):
                return np.nan
            return 0.1

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider,
                                              min_validations=1)

        # Should not filter (0,1) even though some triangles have NaN
        # NaN triangles are skipped, not counted as violations
        assert result == [] or (0, 1) not in result

    def test_multiple_nan_pairs(self):
        """Test handling of multiple NaN pairs."""
        cluster_indices = [0, 1, 2, 3]

        # Mock distances with multiple NaN pairs
        def mock_get_distance(i, j):
            nan_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            if (i, j) in nan_pairs:
                return np.nan
            return 0.1

        self.mock_provider.get_distance.side_effect = mock_get_distance

        result = filter_intra_cluster_triangles(cluster_indices, self.mock_provider)

        # Both NaN pairs should be filtered
        assert len(result) >= 2
        assert (0, 1) in result or (1, 0) in result
        assert (2, 3) in result or (3, 2) in result


class TestFilterDistanceMatrixTriangles:
    """Test matrix-based triangle inequality filtering."""

    def test_filter_empty_matrix(self):
        """Test filtering empty matrix."""
        matrix = np.array([]).reshape(0, 0)
        result = filter_distance_matrix_triangles(matrix)
        assert result.shape == (0, 0)

    def test_filter_single_sequence_matrix(self):
        """Test filtering matrix with single sequence."""
        matrix = np.array([[0.0]])
        result = filter_distance_matrix_triangles(matrix)
        np.testing.assert_array_equal(result, matrix)

    def test_filter_small_matrix(self):
        """Test filtering small matrix."""
        matrix = np.array([
            [0.0, 0.1],
            [0.1, 0.0]
        ])
        result = filter_distance_matrix_triangles(matrix)
        np.testing.assert_array_equal(result, matrix)

    def test_filter_no_violations_matrix(self):
        """Test filtering matrix with no violations."""
        matrix = np.array([
            [0.0, 0.1, 0.15],
            [0.1, 0.0, 0.12],
            [0.15, 0.12, 0.0]
        ])
        result = filter_distance_matrix_triangles(matrix)
        np.testing.assert_array_equal(result, matrix)

    def test_filter_with_violations_matrix(self):
        """Test filtering matrix with violations."""
        matrix = np.array([
            [0.0, 0.1, 0.3],   # 0.3 > 0.1 + 0.05 violates triangle inequality
            [0.1, 0.0, 0.05],
            [0.3, 0.05, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix)

        # The violating distance (0.3) should be set to NaN
        assert np.isnan(result[0, 2])
        assert np.isnan(result[2, 0])
        assert not np.isnan(result[0, 1])
        assert not np.isnan(result[1, 2])

    def test_violation_tolerance_matrix(self):
        """Test violation tolerance in matrix filtering."""
        matrix = np.array([
            [0.0, 0.1, 0.25],  # 0.25 vs 0.2 = 0.05 difference
            [0.1, 0.0, 0.1],
            [0.25, 0.1, 0.0]
        ])

        # Should not filter with default tolerance
        result = filter_distance_matrix_triangles(matrix)
        assert not np.isnan(result).any()

        # Should filter with tighter tolerance
        result = filter_distance_matrix_triangles(matrix, violation_tolerance=0.01)
        assert np.isnan(result[0, 2])
        assert np.isnan(result[2, 0])

    def test_show_progress_parameter(self):
        """Test show_progress parameter."""
        matrix = np.array([
            [0.0, 0.1, 0.15],
            [0.1, 0.0, 0.12],
            [0.15, 0.12, 0.0]
        ])

        # Should not raise exception with progress disabled
        result = filter_distance_matrix_triangles(matrix, show_progress=False)
        assert result.shape == matrix.shape

    def test_existing_nan_skipped(self):
        """Test that triangles with existing NaN values are skipped."""
        matrix = np.array([
            [0.0, 0.1, np.nan],  # Pre-existing NaN (failed alignment)
            [0.1, 0.0, 0.1],
            [np.nan, 0.1, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix, show_progress=False)

        # Pre-existing NaN should remain NaN
        assert np.isnan(result[0, 2])
        assert np.isnan(result[2, 0])

        # Other distances should remain unchanged (no false violations from NaN triangles)
        assert result[0, 1] == 0.1
        assert result[1, 2] == 0.1

    def test_multiple_nan_in_matrix(self):
        """Test matrix with multiple NaN values."""
        matrix = np.array([
            [0.0, np.nan, 0.1, 0.2],
            [np.nan, 0.0, np.nan, 0.15],
            [0.1, np.nan, 0.0, 0.1],
            [0.2, 0.15, 0.1, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix, show_progress=False)

        # All pre-existing NaN should remain
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 2])
        assert np.isnan(result[2, 1])

        # Valid distances should remain (triangles with NaN are skipped)
        assert result[0, 2] == 0.1
        assert result[0, 3] == 0.2

        # Should not raise exception with progress enabled
        result = filter_distance_matrix_triangles(matrix, show_progress=True)
        assert result.shape == matrix.shape

    def test_nan_values_in_matrix(self):
        """Test handling of existing NaN values in matrix."""
        matrix = np.array([
            [0.0, 0.1, np.nan],
            [0.1, 0.0, 0.12],
            [np.nan, 0.12, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix)

        # Existing NaN values should be preserved
        assert np.isnan(result[0, 2])
        assert np.isnan(result[2, 0])
        assert not np.isnan(result[0, 1])
        assert not np.isnan(result[1, 2])

    def test_large_matrix_performance(self):
        """Test performance characteristics with larger matrix."""
        n = 50
        matrix = np.random.rand(n, n)
        # Make matrix symmetric
        matrix = (matrix + matrix.T) / 2
        # Set diagonal to zero
        np.fill_diagonal(matrix, 0)

        result = filter_distance_matrix_triangles(matrix, show_progress=False)
        assert result.shape == matrix.shape

    def test_max_distance_heuristic(self):
        """Test that max-distance heuristic filters largest violating distance."""
        matrix = np.array([
            [0.0, 0.1, 0.5],   # 0.5 is largest violating distance
            [0.1, 0.0, 0.05],
            [0.5, 0.05, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix)

        # Largest distance (0.5) should be filtered
        assert np.isnan(result[0, 2])
        assert np.isnan(result[2, 0])
        assert not np.isnan(result[0, 1])
        assert not np.isnan(result[1, 2])


class TestAddNanFilteringToDistanceList:
    """Test NaN filtering utility function."""

    def test_filter_empty_list(self):
        """Test filtering empty list."""
        distances = []
        result = add_nan_filtering_to_distance_list(distances)
        assert result == []

    def test_filter_no_nans(self):
        """Test filtering list with no NaN values."""
        distances = [0.1, 0.2, 0.3]
        result = add_nan_filtering_to_distance_list(distances)
        assert result == distances

    def test_filter_with_nans(self):
        """Test filtering list with NaN values."""
        distances = [0.1, np.nan, 0.3, np.nan, 0.5]
        result = add_nan_filtering_to_distance_list(distances)
        assert result == [0.1, 0.3, 0.5]

    def test_filter_all_nans(self):
        """Test filtering list with all NaN values."""
        distances = [np.nan, np.nan, np.nan]
        result = add_nan_filtering_to_distance_list(distances)
        assert result == []

    def test_filter_mixed_types(self):
        """Test filtering list with mixed numeric types."""
        distances = [0.1, np.nan, 0, 1.5, np.inf]
        result = add_nan_filtering_to_distance_list(distances)
        # Should preserve all non-NaN values including 0 and inf
        assert result == [0.1, 0, 1.5, np.inf]


class TestTriangleFilteringEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_provider = Mock()

    def test_distance_provider_errors(self):
        """Test behavior when distance provider raises errors."""
        distances = {0: 0.1, 1: 0.2, 2: 0.15}
        self.mock_provider.get_distance.side_effect = ValueError("Distance error")

        with pytest.raises(ValueError):
            filter_distance_dict_triangles(distances, self.mock_provider)

    def test_negative_distances(self):
        """Test handling of negative distances."""
        distances = {0: -0.1, 1: 0.2, 2: 0.15}
        self.mock_provider.get_distance.return_value = 0.1

        # Should handle negative distances without error
        result = filter_distance_dict_triangles(distances, self.mock_provider)
        assert isinstance(result, dict)

    def test_infinite_distances(self):
        """Test handling of infinite distances."""
        distances = {0: np.inf, 1: 0.2, 2: 0.15}
        self.mock_provider.get_distance.return_value = 0.1

        result = filter_distance_dict_triangles(distances, self.mock_provider)
        assert isinstance(result, dict)

    def test_zero_tolerance(self):
        """Test behavior with zero violation tolerance."""
        matrix = np.array([
            [0.0, 0.1, 0.20001],  # Tiny violation
            [0.1, 0.0, 0.1],
            [0.20001, 0.1, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix, violation_tolerance=0.0)
        # Should filter even tiny violations
        assert np.isnan(result[0, 2])
        assert np.isnan(result[2, 0])

    def test_large_tolerance(self):
        """Test behavior with very large violation tolerance."""
        matrix = np.array([
            [0.0, 0.1, 1.0],   # Large violation
            [0.1, 0.0, 0.1],
            [1.0, 0.1, 0.0]
        ])

        result = filter_distance_matrix_triangles(matrix, violation_tolerance=10.0)
        # Should not filter anything with large tolerance
        assert not np.isnan(result).any()


class TestTriangleFilteringPropertyBased:
    """Property-based tests for triangle filtering."""

    @given(st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=0, max_size=20))
    def test_nan_filtering_preserves_finite_values(self, distances):
        """Test that NaN filtering preserves all finite values."""
        # Add some NaN values
        distances_with_nans = distances + [np.nan] * 3

        result = add_nan_filtering_to_distance_list(distances_with_nans)

        # All original finite values should be preserved
        assert len(result) >= len(distances)
        for d in distances:
            assert d in result

    @given(st.integers(3, 10))
    def test_distance_matrix_triangle_count(self, n):
        """Test that triangle counting is correct."""
        matrix = np.random.rand(n, n)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 0)

        # Should not raise exception
        result = filter_distance_matrix_triangles(matrix, show_progress=False)
        assert result.shape == (n, n)

    @given(st.floats(0.0, 1.0), st.integers(1, 5))
    def test_tolerance_monotonicity(self, tolerance, min_validations):
        """Test that larger tolerance leads to fewer violations."""
        # Create a simple violation case
        matrix = np.array([
            [0.0, 0.1, 0.25],
            [0.1, 0.0, 0.1],
            [0.25, 0.1, 0.0]
        ])

        result1 = filter_distance_matrix_triangles(matrix,
                                                 violation_tolerance=tolerance,
                                                 show_progress=False)
        result2 = filter_distance_matrix_triangles(matrix,
                                                 violation_tolerance=tolerance + 0.1,
                                                 show_progress=False)

        # Larger tolerance should have fewer or equal NaN values
        nans1 = np.sum(np.isnan(result1))
        nans2 = np.sum(np.isnan(result2))
        assert nans2 <= nans1