"""
Tests that the fast pairwise scoring path gives exactly the results of the
straightforward one: per-character column filtering followed by
adjusted_identity.score_alignment().
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import adjusted_identity

from gaphack.utils import (
    _filter_msa_positions_python,
    compute_msa_distance_detailed,
    filter_msa_positions,
    load_sequences_from_fasta,
    replace_terminal_gaps,
    run_spoa_msa,
    trim_alignment_by_coverage,
)

TEST_DATA = Path(__file__).parent / "test_data"

MYCOBLAST_PARAMS = adjusted_identity.AdjustmentParams(
    end_skip_distance=0,
    normalize_homopolymers=True,
    handle_iupac_overlap=True,
    normalize_indels=True,
    max_repeat_motif_length=1,
)


def reference_distance(seq1_aligned, seq2_aligned, original_len1, original_len2, normalization_length):
    """The distance calculation written out the slow way, via score_alignment().

    Returns (is_valid, mismatches, pairwise_overlap, distance_normalized, distance_pairwise).
    """
    invalid = (False, 0, 0, None, None)
    if original_len1 == 0 or original_len2 == 0:
        return invalid
    seq1_clean, seq2_clean = _filter_msa_positions_python(seq1_aligned, seq2_aligned)
    overlap = len(seq1_clean)
    if overlap < min(original_len1, original_len2) * 0.5 or overlap == 0:
        return invalid
    mismatches = adjusted_identity.score_alignment(
        seq1_clean, seq2_clean, adjustment_params=MYCOBLAST_PARAMS
    ).mismatches
    return (True, mismatches, overlap, mismatches / normalization_length, mismatches / overlap)


def as_tuple(result):
    if not result.is_valid:
        return (False, 0, 0, None, None)
    return (True, result.mismatches, result.pairwise_overlap,
            result.distance_normalized, result.distance_pairwise)


@pytest.fixture(scope="module")
def russula_msa():
    """50 real ITS sequences aligned the way MSACachedDistanceProvider aligns them."""
    sequences, _, _ = load_sequences_from_fasta(str(TEST_DATA / "russula_diverse_50.fasta"))
    aligned = run_spoa_msa(sequences)
    if aligned is None:
        pytest.skip("SPOA not available")
    aligned = trim_alignment_by_coverage(replace_terminal_gaps(aligned))
    return sequences, aligned


class TestFilterMsaPositions:
    """filter_msa_positions must match the per-character reference on any input."""

    @given(st.integers(min_value=0, max_value=200).flatmap(
        lambda n: st.tuples(
            st.text(alphabet="ACGTNRY-.", min_size=n, max_size=n),
            st.text(alphabet="ACGTNRY-.", min_size=n, max_size=n),
        )
    ))
    @settings(max_examples=300, deadline=None)
    def test_matches_reference_on_random_alignments(self, pair):
        seq1, seq2 = pair
        assert filter_msa_positions(seq1, seq2) == _filter_msa_positions_python(seq1, seq2)

    def test_drops_terminal_gaps_and_shared_indels_only(self):
        #          keep  .x    x.    --    -A    A-    keep
        seq1 = "A" + "." + "C" + "-" + "-" + "G" + "T"
        seq2 = "A" + "C" + "." + "-" + "T" + "-" + "T"
        assert filter_msa_positions(seq1, seq2) == ("A-GT", "AT-T")

    def test_empty_alignment(self):
        assert filter_msa_positions("", "") == ("", "")

    def test_all_positions_removed(self):
        assert filter_msa_positions("..--", "AC--") == ("", "")

    def test_non_ascii_input_uses_reference(self):
        seq1, seq2 = "ACéT.", "ACGT."
        assert filter_msa_positions(seq1, seq2) == _filter_msa_positions_python(seq1, seq2)
        assert filter_msa_positions(seq1, seq2) == ("ACéT", "ACGT")

    def test_longer_second_sequence_is_truncated_like_reference(self):
        # The reference walks the first sequence's positions only.
        assert filter_msa_positions("AC-", "AC-GGG") == _filter_msa_positions_python("AC-", "AC-GGG")

    def test_shorter_second_sequence_raises_like_reference(self):
        with pytest.raises(IndexError):
            filter_msa_positions("ACGT", "AC")

    def test_real_alignment_matches_reference(self, russula_msa):
        _, aligned = russula_msa
        for i in range(len(aligned)):
            for j in range(i + 1, len(aligned)):
                assert (filter_msa_positions(aligned[i], aligned[j]) ==
                        _filter_msa_positions_python(aligned[i], aligned[j]))


class TestDistanceMatchesScoreAlignment:
    """Mismatch counts and distances must equal those from score_alignment()."""

    def test_every_pair_of_a_real_alignment(self, russula_msa):
        sequences, aligned = russula_msa
        normalization_length = int(np.median([len(s) for s in sequences]))

        mismatch_counts = set()
        for i in range(len(aligned)):
            for j in range(i + 1, len(aligned)):
                args = (aligned[i], aligned[j], len(sequences[i]), len(sequences[j]), normalization_length)
                expected = reference_distance(*args)
                assert as_tuple(compute_msa_distance_detailed(*args)) == expected, f"pair {i},{j}"
                mismatch_counts.add(expected[1])

        # Guard against a vacuous pass: the data must exercise a range of scores.
        assert len(mismatch_counts) > 10

    @pytest.mark.parametrize("seq1,seq2", [
        ("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT"),          # identical
        ("ACGTACGTACGTACGTACGT", "ACGTACGTTCGTACGTACGT"),          # one substitution
        ("ACGTAAAACGTACGTACGTT", "ACGTAAA-CGTACGTACGTT"),          # homopolymer length difference
        ("ACGTACGT----ACGTACGT", "ACGTACGTTTGCACGTACGT"),          # multi-base indel
        ("ACGTACGTRCGTACGTACGT", "ACGTACGTACGTACGTACGT"),          # IUPAC ambiguity code
        ("ACGTACGTNCGTACGTACGT", "ACGTACGTYCGTACGTACGT"),          # two ambiguity codes
        ("....ACGTACGTACGTACGT", "ACGTACGTACGTACGT...."),          # terminal gaps on both sides
        ("ACGT--ACGTACGTACGTAC", "ACGT--ACGTACGAACGTAC"),          # shared indel column
    ])
    def test_hand_written_alignments(self, seq1, seq2):
        length1 = sum(1 for c in seq1 if c not in ".-")
        length2 = sum(1 for c in seq2 if c not in ".-")
        args = (seq1, seq2, length1, length2, 20)
        assert as_tuple(compute_msa_distance_detailed(*args)) == reference_distance(*args)

    def test_insufficient_overlap_is_invalid(self):
        seq1 = "A" * 20 + "." * 80
        seq2 = "." * 15 + "A" * 85
        result = compute_msa_distance_detailed(seq1, seq2, 20, 85, 50)
        assert not result.is_valid
        assert np.isnan(result.distance_normalized)
