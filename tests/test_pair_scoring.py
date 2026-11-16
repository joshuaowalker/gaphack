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

import gaphack.utils as utils
from gaphack.utils import (
    PairScoreCache,
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


@pytest.fixture(autouse=True)
def fresh_pair_score_cache(monkeypatch):
    """Every test starts with an empty pair score cache of the default size."""
    cache = PairScoreCache(200_000)
    monkeypatch.setattr(utils, "pair_score_cache", cache)
    return cache


@pytest.fixture
def counted_scorer(monkeypatch):
    """Count how many times the underlying scorer actually runs."""
    real = utils._get_mismatch_counter()
    calls = []

    def counting(seq1_clean, seq2_clean):
        calls.append((seq1_clean, seq2_clean))
        return real(seq1_clean, seq2_clean)

    monkeypatch.setattr(utils, "_mismatch_counter", counting)
    return calls


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
                # and again, now answered from the pair score cache
                assert as_tuple(compute_msa_distance_detailed(*args)) == expected, f"pair {i},{j} (cached)"
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


class TestPairScoreCache:
    """A pair is scored once; repeats are answered from the cache with the same result."""

    SEQ1 = "ACGTACGTACGTACGTACGT"
    SEQ2 = "ACGTACGTTCGTACGTACGT"

    def test_repeat_is_not_rescored_and_matches(self, counted_scorer, fresh_pair_score_cache):
        first = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)
        second = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)

        assert len(counted_scorer) == 1
        assert as_tuple(first) == as_tuple(second)
        assert first.mismatches == 1
        assert fresh_pair_score_cache.hits == 1

    def test_equal_strings_from_another_alignment_reuse_the_score(self, counted_scorer):
        compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)
        # New string objects with the same content, as a rebuilt MSA would produce
        again1, again2 = "".join(list(self.SEQ1)), "".join(list(self.SEQ2))
        result = compute_msa_distance_detailed(again1, again2, 20, 20, 20)

        assert len(counted_scorer) == 1
        assert result.mismatches == 1

    def test_normalization_length_is_applied_per_call(self, counted_scorer):
        short = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 10)
        long = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 40)

        assert len(counted_scorer) == 1
        assert short.distance_normalized == 1 / 10
        assert long.distance_normalized == 1 / 40

    def test_overlap_rule_is_applied_per_call(self, counted_scorer):
        """The overlap rule depends on the caller's original lengths, so it is never cached."""
        seq1 = "ACGTACGTAC" + "." * 30
        seq2 = "ACGTACGTAC" + "A" * 30
        # 10bp overlap is under half of the shorter original length (30): invalid, not scored
        rejected = compute_msa_distance_detailed(seq1, seq2, 30, 40, 30)
        assert not rejected.is_valid
        assert len(counted_scorer) == 0
        # Same aligned strings, shorter original length (10): now valid, and scored
        accepted = compute_msa_distance_detailed(seq1, seq2, 10, 40, 30)
        assert accepted.is_valid and accepted.pairwise_overlap == 10
        assert len(counted_scorer) == 1
        # ...and the first caller is still rejected afterwards
        assert not compute_msa_distance_detailed(seq1, seq2, 30, 40, 30).is_valid

    def test_argument_order_is_part_of_the_key(self, counted_scorer):
        compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)
        compute_msa_distance_detailed(self.SEQ2, self.SEQ1, 20, 20, 20)
        assert len(counted_scorer) == 2

    def test_failed_scoring_is_retried_not_cached(self, monkeypatch):
        outcomes = [RuntimeError("boom"), 3]

        def flaky(seq1_clean, seq2_clean):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(utils, "_mismatch_counter", flaky)
        assert not compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20).is_valid
        retry = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)
        assert retry.is_valid and retry.mismatches == 3

    def test_size_zero_disables_caching(self, monkeypatch, counted_scorer):
        monkeypatch.setattr(utils, "pair_score_cache", PairScoreCache(0))
        first = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)
        second = compute_msa_distance_detailed(self.SEQ1, self.SEQ2, 20, 20, 20)

        assert len(counted_scorer) == 2
        assert as_tuple(first) == as_tuple(second)
        assert len(utils.pair_score_cache) == 0

    def test_cache_never_exceeds_its_limit_and_drops_oldest_first(self):
        cache = PairScoreCache(8)
        for i in range(20):
            cache.put((f"a{i}", f"b{i}"), i, i)
            assert len(cache) <= 8

        assert ("a19", "b19") in cache          # newest kept
        assert ("a0", "b0") not in cache        # oldest dropped
        assert cache.get(("a19", "b19")) == (19, 19)

    def test_updating_an_entry_at_capacity_does_not_evict(self):
        cache = PairScoreCache(4)
        for i in range(4):
            cache.put((f"a{i}", "b"), i, None)
        cache.put(("a0", "b"), 0, 7)            # filtered earlier, scored now

        assert len(cache) == 4
        assert cache.get(("a0", "b")) == (0, 7)
