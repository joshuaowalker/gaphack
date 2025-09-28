"""
Tests for distance provider architecture (lazy and scoped).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from hypothesis import given, strategies as st, settings

from gaphack.lazy_distances import (
    DistanceProvider,
    PrecomputedDistanceProvider,
    LazyDistanceProvider,
    SubsetDistanceProvider,
    DistanceProviderFactory
)
from gaphack.scoped_distances import (
    ScopedDistanceProvider,
    PrecomputedScopedDistanceProvider,
    create_scoped_distance_provider
)


class TestDistanceProviderABC:
    """Test suite for DistanceProvider abstract base class."""

    def test_abstract_methods(self):
        """Test that DistanceProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DistanceProvider()


class TestPrecomputedDistanceProvider:
    """Test suite for PrecomputedDistanceProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple 4x4 symmetric distance matrix
        self.distance_matrix = np.array([
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.15, 0.25],
            [0.2, 0.15, 0.0, 0.1],
            [0.3, 0.25, 0.1, 0.0]
        ])
        self.provider = PrecomputedDistanceProvider(self.distance_matrix)

    def test_initialization(self):
        """Test PrecomputedDistanceProvider initialization."""
        assert self.provider.n == 4
        assert np.array_equal(self.provider.distance_matrix, self.distance_matrix)

    def test_get_distance(self):
        """Test distance retrieval."""
        assert self.provider.get_distance(0, 1) == 0.1
        assert self.provider.get_distance(1, 0) == 0.1  # Symmetric
        assert self.provider.get_distance(2, 3) == 0.1
        assert self.provider.get_distance(0, 0) == 0.0  # Self-distance

    def test_get_distances_from_sequence(self):
        """Test batch distance retrieval from one sequence."""
        target_indices = {1, 2, 3}
        distances = self.provider.get_distances_from_sequence(0, target_indices)

        expected = {1: 0.1, 2: 0.2, 3: 0.3}
        assert distances == expected

    def test_ensure_distances_computed(self):
        """Test ensure_distances_computed (no-op for precomputed)."""
        # Should not raise any errors
        self.provider.ensure_distances_computed({0, 1, 2})

    def test_get_distance_out_of_bounds(self):
        """Test error handling for out-of-bounds indices."""
        with pytest.raises(IndexError):
            self.provider.get_distance(0, 5)


class TestLazyDistanceProvider:
    """Test suite for LazyDistanceProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATCGATCGATCG",
            "ATCGATCGATCC",
            "TTTTGGGGCCCC",
            "TTTTGGGGCCCT"
        ]

    @patch('gaphack.lazy_distances.calculate_distance_matrix')
    def test_initialization(self, mock_calc_matrix):
        """Test LazyDistanceProvider initialization."""
        provider = LazyDistanceProvider(
            self.test_sequences,
            alignment_method="adjusted",
            end_skip_distance=20
        )

        assert provider.n == 4
        assert provider.sequences == self.test_sequences
        assert provider.alignment_method == "adjusted"
        assert provider.end_skip_distance == 20
        assert len(provider._distance_cache) == 0

    @patch('gaphack.lazy_distances.LazyDistanceProvider._compute_single_distance')
    def test_get_distance_caching(self, mock_compute):
        """Test distance retrieval with caching."""
        mock_compute.return_value = 0.15

        provider = LazyDistanceProvider(self.test_sequences)

        # First call should compute
        distance1 = provider.get_distance(0, 1)
        assert distance1 == 0.15
        mock_compute.assert_called_once_with(0, 1)

        # Second call should use cache
        mock_compute.reset_mock()
        distance2 = provider.get_distance(0, 1)
        assert distance2 == 0.15
        mock_compute.assert_not_called()

        # Reverse order should also use cache
        distance3 = provider.get_distance(1, 0)
        assert distance3 == 0.15
        mock_compute.assert_not_called()

    def test_get_distance_self(self):
        """Test distance to self."""
        provider = LazyDistanceProvider(self.test_sequences)
        assert provider.get_distance(0, 0) == 0.0
        assert provider.get_distance(2, 2) == 0.0

    @patch('gaphack.lazy_distances.LazyDistanceProvider._compute_single_distance')
    def test_get_distances_from_sequence(self, mock_compute):
        """Test batch distance retrieval."""
        mock_compute.side_effect = [0.1, 0.2, 0.3]  # Return different distances

        provider = LazyDistanceProvider(self.test_sequences)
        target_indices = {1, 2, 3}
        distances = provider.get_distances_from_sequence(0, target_indices)

        expected = {1: 0.1, 2: 0.2, 3: 0.3}
        assert distances == expected
        assert mock_compute.call_count == 3

    @patch('gaphack.lazy_distances.LazyDistanceProvider._compute_pairwise_distances')
    def test_ensure_distances_computed(self, mock_compute_matrix):
        """Test ensuring distances are computed."""
        # Mock distance matrix computation
        mock_matrix = np.array([[0.0, 0.1], [0.1, 0.0]])
        mock_compute_matrix.return_value = mock_matrix

        provider = LazyDistanceProvider(self.test_sequences)

        # Should call matrix computation for uncached distances
        provider.ensure_distances_computed({0, 1})
        mock_compute_matrix.assert_called_once_with({0, 1})

    def test_get_cache_stats(self):
        """Test cache statistics."""
        provider = LazyDistanceProvider(self.test_sequences)

        stats = provider.get_cache_stats()
        assert stats['cached_distances'] == 0
        assert stats['total_computations'] == 0
        assert stats['theoretical_max'] == 6  # 4 choose 2

    @patch('adjusted_identity.align_and_score')
    @patch('adjusted_identity.AdjustmentParams')
    def test_compute_single_distance_adjusted(self, mock_params, mock_align):
        """Test single distance computation with adjusted alignment."""
        # Mock alignment result
        mock_result = Mock()
        mock_result.identity = 0.85
        mock_align.return_value = mock_result

        provider = LazyDistanceProvider(self.test_sequences, alignment_method="adjusted")
        distance = provider._compute_single_distance(0, 1)

        assert abs(distance - 0.15) < 1e-10  # 1.0 - 0.85 (accounting for floating point precision)
        mock_align.assert_called_once()

    @patch('adjusted_identity.align_and_score')
    def test_compute_single_distance_failure(self, mock_align):
        """Test single distance computation with alignment failure."""
        # Mock alignment failure
        mock_align.side_effect = Exception("Alignment failed")

        provider = LazyDistanceProvider(self.test_sequences)
        distance = provider._compute_single_distance(0, 1)

        assert distance == 1.0  # Maximum distance for failed alignments

    def test_cache_key_generation(self):
        """Test cache key generation is canonical."""
        provider = LazyDistanceProvider(self.test_sequences)

        key1 = provider._get_cache_key(0, 1)
        key2 = provider._get_cache_key(1, 0)

        assert key1 == key2
        assert key1 == (0, 1)  # Canonical form: smaller index first


class TestSubsetDistanceProvider:
    """Test suite for SubsetDistanceProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock global provider
        self.global_provider = Mock(spec=DistanceProvider)
        self.global_provider.get_distance.side_effect = lambda i, j: abs(i - j) * 0.1

        # Subset maps local indices [0, 1, 2] to global indices [1, 3, 5]
        self.subset_indices = [1, 3, 5]
        self.subset_provider = SubsetDistanceProvider(self.global_provider, self.subset_indices)

    def test_initialization(self):
        """Test SubsetDistanceProvider initialization."""
        assert self.subset_provider.n == 3
        assert self.subset_provider.subset_indices == [1, 3, 5]

    def test_get_distance(self):
        """Test distance retrieval with index mapping."""
        # Local indices 0, 1 map to global indices 1, 3
        distance = self.subset_provider.get_distance(0, 1)

        # Should call global provider with indices 1, 3
        self.global_provider.get_distance.assert_called_with(1, 3)
        assert distance == 0.2  # abs(1 - 3) * 0.1

    def test_get_distances_from_sequence(self):
        """Test batch distance retrieval with index mapping."""
        # Mock the return value from global provider
        self.global_provider.get_distances_from_sequence.return_value = {3: 0.2, 5: 0.4}

        target_indices = {1, 2}  # Local indices
        distances = self.subset_provider.get_distances_from_sequence(0, target_indices)

        # Should map to global indices: source=1, targets={3, 5}
        expected_global_call = {3, 5}
        self.global_provider.get_distances_from_sequence.assert_called_with(1, expected_global_call)

        # Should return distances mapped back to local indices
        expected_local_distances = {1: 0.2, 2: 0.4}
        assert distances == expected_local_distances

    def test_ensure_distances_computed(self):
        """Test ensuring distances computed with index mapping."""
        seq_indices = {0, 2}  # Local indices
        self.subset_provider.ensure_distances_computed(seq_indices)

        # Should map to global indices {1, 5}
        expected_global_indices = {1, 5}
        self.global_provider.ensure_distances_computed.assert_called_with(expected_global_indices)


class TestDistanceProviderFactory:
    """Test suite for DistanceProviderFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = ["ATCG", "GCTA", "CGAT"]
        self.test_matrix = np.array([
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.15],
            [0.2, 0.15, 0.0]
        ])

    def test_create_lazy_provider(self):
        """Test lazy provider creation."""
        provider = DistanceProviderFactory.create_lazy_provider(
            self.test_sequences, alignment_method="adjusted"
        )

        assert isinstance(provider, LazyDistanceProvider)
        assert provider.sequences == self.test_sequences
        assert provider.alignment_method == "adjusted"

    def test_create_precomputed_provider(self):
        """Test precomputed provider creation."""
        provider = DistanceProviderFactory.create_precomputed_provider(self.test_matrix)

        assert isinstance(provider, PrecomputedDistanceProvider)
        assert np.array_equal(provider.distance_matrix, self.test_matrix)

    def test_create_provider_with_matrix(self):
        """Test factory method with distance matrix."""
        provider = DistanceProviderFactory.create_provider(distance_matrix=self.test_matrix)

        assert isinstance(provider, PrecomputedDistanceProvider)

    def test_create_provider_with_sequences_lazy(self):
        """Test factory method with sequences (lazy)."""
        provider = DistanceProviderFactory.create_provider(
            sequences=self.test_sequences, use_lazy=True
        )

        assert isinstance(provider, LazyDistanceProvider)

    @patch('gaphack.lazy_distances.calculate_distance_matrix')
    def test_create_provider_with_sequences_precomputed(self, mock_calc_matrix):
        """Test factory method with sequences (precomputed)."""
        mock_calc_matrix.return_value = self.test_matrix

        provider = DistanceProviderFactory.create_provider(
            sequences=self.test_sequences, use_lazy=False
        )

        assert isinstance(provider, PrecomputedDistanceProvider)
        mock_calc_matrix.assert_called_once()

    def test_create_provider_invalid_input(self):
        """Test factory method with invalid input."""
        with pytest.raises(ValueError, match="Either sequences or distance_matrix must be provided"):
            DistanceProviderFactory.create_provider()


class TestScopedDistanceProvider:
    """Test suite for ScopedDistanceProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.global_provider = Mock(spec=DistanceProvider)
        self.global_provider.get_distance.side_effect = lambda i, j: abs(i - j) * 0.1

        self.all_headers = ["seq_0", "seq_1", "seq_2", "seq_3", "seq_4"]
        self.scope_headers = ["seq_1", "seq_3", "seq_4"]  # Indices 1, 3, 4 in global

        self.scoped_provider = ScopedDistanceProvider(
            self.global_provider, self.scope_headers, self.all_headers
        )

    def test_initialization(self):
        """Test ScopedDistanceProvider initialization."""
        assert self.scoped_provider.n == 3
        assert self.scoped_provider.scope_headers == self.scope_headers
        assert self.scoped_provider.scope_to_global == [1, 3, 4]
        assert self.scoped_provider.global_to_scope == {1: 0, 3: 1, 4: 2}

    def test_initialization_missing_header(self):
        """Test initialization with missing header."""
        bad_scope_headers = ["seq_1", "seq_missing", "seq_4"]

        with pytest.raises(ValueError, match="Header 'seq_missing' not found"):
            ScopedDistanceProvider(self.global_provider, bad_scope_headers, self.all_headers)

    def test_get_distance(self):
        """Test distance retrieval with local-to-global mapping."""
        # Local indices 0, 1 map to global indices 1, 3
        distance = self.scoped_provider.get_distance(0, 1)

        self.global_provider.get_distance.assert_called_with(1, 3)
        assert distance == 0.2  # abs(1 - 3) * 0.1

    def test_get_distance_caching(self):
        """Test local caching in scoped provider."""
        # First call
        distance1 = self.scoped_provider.get_distance(0, 1)
        assert self.global_provider.get_distance.call_count == 1

        # Second call should use cache
        distance2 = self.scoped_provider.get_distance(0, 1)
        assert distance1 == distance2
        assert self.global_provider.get_distance.call_count == 1  # No additional calls

        # Reverse order should also use cache
        distance3 = self.scoped_provider.get_distance(1, 0)
        assert distance3 == distance1
        assert self.global_provider.get_distance.call_count == 1

    def test_get_distance_out_of_bounds(self):
        """Test error handling for out-of-bounds local indices."""
        with pytest.raises(IndexError, match="Local index 5 out of range"):
            self.scoped_provider.get_distance(0, 5)

        with pytest.raises(IndexError, match="Local index -1 out of range"):
            self.scoped_provider.get_distance(-1, 0)

    def test_get_distances_from_sequence(self):
        """Test batch distance retrieval."""
        target_indices = {1, 2}
        distances = self.scoped_provider.get_distances_from_sequence(0, target_indices)

        assert len(distances) == 2
        assert 1 in distances
        assert 2 in distances

    def test_build_distance_matrix(self):
        """Test building full distance matrix for scope."""
        matrix = self.scoped_provider.build_distance_matrix()

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0.0  # Distance to self
        assert matrix[0, 1] == matrix[1, 0]  # Symmetric

    def test_get_global_index(self):
        """Test local-to-global index conversion."""
        assert self.scoped_provider.get_global_index(0) == 1
        assert self.scoped_provider.get_global_index(1) == 3
        assert self.scoped_provider.get_global_index(2) == 4

        with pytest.raises(IndexError):
            self.scoped_provider.get_global_index(5)

    def test_get_local_index(self):
        """Test global-to-local index conversion."""
        assert self.scoped_provider.get_local_index(1) == 0
        assert self.scoped_provider.get_local_index(3) == 1
        assert self.scoped_provider.get_local_index(4) == 2
        assert self.scoped_provider.get_local_index(0) is None  # Not in scope

    def test_get_scope_headers(self):
        """Test scope headers retrieval."""
        headers = self.scoped_provider.get_scope_headers()
        assert headers == self.scope_headers
        assert headers is not self.scope_headers  # Should be a copy

    def test_clear_cache(self):
        """Test cache clearing."""
        # Populate cache
        self.scoped_provider.get_distance(0, 1)
        assert len(self.scoped_provider.local_cache) > 0

        # Clear cache
        self.scoped_provider.clear_cache()
        assert len(self.scoped_provider.local_cache) == 0


class TestPrecomputedScopedDistanceProvider:
    """Test suite for PrecomputedScopedDistanceProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scope_headers = ["seq_1", "seq_3", "seq_4"]
        self.distance_matrix = np.array([
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.15],
            [0.2, 0.15, 0.0]
        ])
        self.provider = PrecomputedScopedDistanceProvider(self.scope_headers, self.distance_matrix)

    def test_initialization(self):
        """Test PrecomputedScopedDistanceProvider initialization."""
        assert self.provider.n == 3
        assert self.provider.scope_headers == self.scope_headers
        assert np.array_equal(self.provider.distance_matrix, self.distance_matrix)

    def test_initialization_size_mismatch(self):
        """Test initialization with mismatched sizes."""
        bad_matrix = np.array([[0.0, 0.1], [0.1, 0.0]])  # 2x2 matrix for 3 headers

        with pytest.raises(ValueError, match="Header count must match distance matrix size"):
            PrecomputedScopedDistanceProvider(self.scope_headers, bad_matrix)

    def test_initialization_non_square_matrix(self):
        """Test initialization with non-square matrix."""
        bad_matrix = np.array([[0.0, 0.1, 0.2]])  # 1x3 matrix

        with pytest.raises(ValueError, match="Distance matrix must be square"):
            PrecomputedScopedDistanceProvider(["seq_1"], bad_matrix)

    def test_get_distance(self):
        """Test distance retrieval from precomputed matrix."""
        assert self.provider.get_distance(0, 1) == 0.1
        assert self.provider.get_distance(1, 2) == 0.15
        assert self.provider.get_distance(0, 0) == 0.0

    def test_get_distance_out_of_bounds(self):
        """Test error handling for out-of-bounds indices."""
        with pytest.raises(IndexError):
            self.provider.get_distance(0, 5)

    def test_get_distances_from_sequence(self):
        """Test batch distance retrieval."""
        target_indices = {1, 2}
        distances = self.provider.get_distances_from_sequence(0, target_indices)

        expected = {1: 0.1, 2: 0.2}
        assert distances == expected

    def test_build_distance_matrix(self):
        """Test distance matrix return."""
        matrix = self.provider.build_distance_matrix()

        assert np.array_equal(matrix, self.distance_matrix)
        assert matrix is not self.distance_matrix  # Should be a copy

    def test_get_scope_headers(self):
        """Test scope headers retrieval."""
        headers = self.provider.get_scope_headers()
        assert headers == self.scope_headers
        assert headers is not self.scope_headers  # Should be a copy


class TestCreateScopedDistanceProvider:
    """Test suite for scoped distance provider factory function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.global_provider = Mock(spec=DistanceProvider)
        self.all_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]

    def test_create_small_scope(self):
        """Test factory function with small scope (below threshold)."""
        scope_headers = ["seq_1", "seq_2"]  # Small scope

        provider = create_scoped_distance_provider(
            self.global_provider, scope_headers, self.all_headers, precompute_threshold=5
        )

        assert isinstance(provider, ScopedDistanceProvider)
        assert provider.scope_headers == scope_headers

    @patch('gaphack.scoped_distances.ScopedDistanceProvider.build_distance_matrix')
    def test_create_large_scope(self, mock_build_matrix):
        """Test factory function with large scope (above threshold)."""
        scope_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]  # Large scope
        mock_matrix = np.eye(4) * 0.1
        mock_build_matrix.return_value = mock_matrix

        provider = create_scoped_distance_provider(
            self.global_provider, scope_headers, self.all_headers, precompute_threshold=2
        )

        assert isinstance(provider, PrecomputedScopedDistanceProvider)
        assert provider.scope_headers == scope_headers
        mock_build_matrix.assert_called_once()


class TestDistanceProviderPropertyBased:
    """Property-based tests for distance providers."""

    @given(
        matrix_size=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(deadline=500)
    def test_precomputed_provider_symmetry(self, matrix_size, seed):
        """Property-based test for distance matrix symmetry."""
        np.random.seed(seed)

        # Generate symmetric distance matrix
        matrix = np.random.rand(matrix_size, matrix_size)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 0.0)  # Zero diagonal

        provider = PrecomputedDistanceProvider(matrix)

        # Test symmetry property
        for i in range(matrix_size):
            for j in range(matrix_size):
                assert provider.get_distance(i, j) == provider.get_distance(j, i)

        # Test self-distance property
        for i in range(matrix_size):
            assert provider.get_distance(i, i) == 0.0

    @given(
        scope_size=st.integers(min_value=1, max_value=8),
        global_size=st.integers(min_value=1, max_value=12)
    )
    @settings(deadline=500)
    def test_scoped_provider_index_mapping(self, scope_size, global_size):
        """Property-based test for scoped provider index mapping."""
        if scope_size > global_size:
            return  # Skip invalid combinations

        # Create headers
        all_headers = [f"seq_{i}" for i in range(global_size)]
        scope_indices = sorted(np.random.choice(global_size, scope_size, replace=False))
        scope_headers = [all_headers[i] for i in scope_indices]

        # Mock global provider
        global_provider = Mock(spec=DistanceProvider)
        global_provider.get_distance.side_effect = lambda i, j: abs(i - j) * 0.1

        provider = ScopedDistanceProvider(global_provider, scope_headers, all_headers)

        # Test that local-to-global mapping is correct
        for local_idx, expected_global_idx in enumerate(scope_indices):
            assert provider.get_global_index(local_idx) == expected_global_idx

        # Test reverse mapping
        for global_idx in scope_indices:
            local_idx = provider.get_local_index(global_idx)
            assert local_idx is not None
            assert provider.get_global_index(local_idx) == global_idx


class TestDistanceProviderIntegration:
    """Integration tests for distance provider interactions."""

    def test_lazy_provider_with_subset(self):
        """Test LazyDistanceProvider wrapped with SubsetDistanceProvider."""
        sequences = ["ATCG", "GCTA", "CGAT", "TACG"]

        # Create lazy provider
        lazy_provider = LazyDistanceProvider(sequences)

        # Wrap with subset provider (indices 1, 3)
        subset_indices = [1, 3]
        subset_provider = SubsetDistanceProvider(lazy_provider, subset_indices)

        # Mock the single distance computation
        with patch.object(lazy_provider, '_compute_single_distance', return_value=0.25):
            distance = subset_provider.get_distance(0, 1)  # Local indices 0, 1 -> global 1, 3

            # Should have called global provider with correct indices
            assert distance == 0.25

    def test_scoped_provider_with_precomputed_global(self):
        """Test ScopedDistanceProvider with PrecomputedDistanceProvider."""
        # Create global distance matrix
        global_matrix = np.array([
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.0, 0.15, 0.25, 0.35],
            [0.2, 0.15, 0.0, 0.1, 0.3],
            [0.3, 0.25, 0.1, 0.0, 0.2],
            [0.4, 0.35, 0.3, 0.2, 0.0]
        ])
        global_provider = PrecomputedDistanceProvider(global_matrix)

        # Create scoped provider for subset
        all_headers = [f"seq_{i}" for i in range(5)]
        scope_headers = ["seq_1", "seq_3", "seq_4"]
        scoped_provider = ScopedDistanceProvider(global_provider, scope_headers, all_headers)

        # Test that scoped distances match global distances
        # Local (0, 1) maps to global (1, 3)
        scoped_distance = scoped_provider.get_distance(0, 1)
        global_distance = global_provider.get_distance(1, 3)
        assert scoped_distance == global_distance

        # Test distance matrix building
        scoped_matrix = scoped_provider.build_distance_matrix()
        assert scoped_matrix.shape == (3, 3)

        # Verify matrix values match global matrix subset
        assert scoped_matrix[0, 1] == global_matrix[1, 3]  # seq_1 to seq_3
        assert scoped_matrix[1, 2] == global_matrix[3, 4]  # seq_3 to seq_4


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def test_empty_distance_matrix(self):
        """Test behavior with empty distance matrix."""
        empty_matrix = np.array([]).reshape(0, 0)
        provider = PrecomputedDistanceProvider(empty_matrix)
        assert provider.n == 0

    def test_single_sequence_distance_matrix(self):
        """Test behavior with single sequence."""
        single_matrix = np.array([[0.0]])
        provider = PrecomputedDistanceProvider(single_matrix)
        assert provider.n == 1
        assert provider.get_distance(0, 0) == 0.0

    def test_lazy_provider_empty_sequences(self):
        """Test LazyDistanceProvider with empty sequence list."""
        provider = LazyDistanceProvider([])
        assert provider.n == 0

        stats = provider.get_cache_stats()
        assert stats['theoretical_max'] == 0

    def test_scoped_provider_empty_scope(self):
        """Test ScopedDistanceProvider with empty scope."""
        global_provider = Mock(spec=DistanceProvider)
        all_headers = ["seq_0", "seq_1"]
        scope_headers = []

        provider = ScopedDistanceProvider(global_provider, scope_headers, all_headers)
        assert provider.n == 0

        matrix = provider.build_distance_matrix()
        assert matrix.shape == (0, 0)

    def test_subset_provider_empty_subset(self):
        """Test SubsetDistanceProvider with empty subset."""
        global_provider = Mock(spec=DistanceProvider)
        subset_indices = []

        provider = SubsetDistanceProvider(global_provider, subset_indices)
        assert provider.n == 0