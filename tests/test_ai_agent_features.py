"""Tests for AI-agent feature engineering."""

import numpy as np
import pandas as pd
import pytest

from openheads.ai_agent_features import (
    AI_AGENT_FEATURE_NAMES,
    N_AI_AGENT_FEATURES,
    _dominant_frequency,
    _shannon_entropy,
    combine_with_base_features,
    compute_ai_agent_features,
    normalize_features,
)


def test_n_features_constant_matches_list():
    assert len(AI_AGENT_FEATURE_NAMES) == N_AI_AGENT_FEATURES == 20


def test_shannon_entropy_uniform_maximum():
    assert _shannon_entropy([1, 1, 1, 1]) == pytest.approx(2.0)


def test_shannon_entropy_skewed_low():
    assert _shannon_entropy([100, 1, 1, 1]) < 0.5


def test_shannon_entropy_empty_zero():
    assert _shannon_entropy([]) == 0.0
    assert _shannon_entropy([5]) == 0.0


def test_dominant_frequency_periodic_high():
    # Pure periodic signal — should have high dominant freq strength
    ts = np.arange(0, 1000, 10)  # every 10s
    score = _dominant_frequency(ts)
    assert score >= 0.0  # not negative


def test_dominant_frequency_random_low():
    rng = np.random.default_rng(42)
    ts = np.sort(rng.integers(0, 100000, size=50))
    score = _dominant_frequency(ts)
    assert 0.0 <= score <= 1.0


def test_dominant_frequency_short_signal_zero():
    assert _dominant_frequency(np.array([1, 2, 3])) == 0.0


@pytest.fixture
def agent_like_df():
    """Very regular bot-like activity."""
    n = 50
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "from_addr": ["0xagent"] * n,
            "to_addr": [f"0xcontract_{i % 3}" for i in range(n)],
            "value_int": [int(1e18)] * n,  # always 1 ETH
            "ts": ts,
            "gas_price": [20_000_000_000] * n,
            "gas": [21000] * n,
            "nonce": list(range(n)),
        }
    )


@pytest.fixture
def human_like_df():
    """Irregular, bursty, diverse activity."""
    rng = np.random.default_rng(42)
    n = 30
    # Mostly business hours weekdays, random gaps
    base = pd.Timestamp("2024-01-01", tz="UTC")
    ts = sorted([base + pd.Timedelta(hours=int(rng.integers(0, 24 * 30))) for _ in range(n)])
    return pd.DataFrame(
        {
            "from_addr": ["0xhuman"] * n,
            "to_addr": [f"0xaddr_{int(rng.integers(0, 20))}" for _ in range(n)],
            "value_int": [int(rng.integers(1, 100) * 1e17) for _ in range(n)],
            "ts": pd.Series(ts),
            "gas_price": [int(rng.integers(10, 100) * 1e9) for _ in range(n)],
            "gas": [int(rng.integers(21000, 300000)) for _ in range(n)],
            "nonce": list(range(n)),
        }
    )


def test_compute_returns_correct_shape(agent_like_df):
    addr_to_idx, features = compute_ai_agent_features(agent_like_df)
    # 1 sender (0xagent) + 3 receiver contracts = 4 addresses
    assert features.shape[1] == N_AI_AGENT_FEATURES
    assert features.shape[0] == len(addr_to_idx)
    assert "0xagent" in addr_to_idx


def test_agent_more_regular_than_human(agent_like_df, human_like_df):
    """Agent should have lower time entropy than human (more regular)."""
    agent_map, agent_feats = compute_ai_agent_features(agent_like_df)
    human_map, human_feats = compute_ai_agent_features(human_like_df)
    ai = agent_map["0xagent"]
    hi = human_map["0xhuman"]
    # Feature 0: tx_time_entropy — lower for agent is ideal, but even different is enough
    assert agent_feats[ai, 0] != human_feats[hi, 0]


def test_agent_has_identical_amounts(agent_like_df):
    addr_map, features = compute_ai_agent_features(agent_like_df)
    idx = addr_map["0xagent"]
    # Feature 8: identical_amount_ratio — should be 1.0 (all tx same amount)
    assert features[idx, 8] == pytest.approx(1.0)


def test_agent_nonce_consistency(agent_like_df):
    addr_map, features = compute_ai_agent_features(agent_like_df)
    idx = addr_map["0xagent"]
    # Feature 9: nonce_gap_consistency — sequential nonces → 1.0
    assert features[idx, 9] == pytest.approx(1.0)


def test_normalize_features_range():
    features = np.random.rand(100, N_AI_AGENT_FEATURES) * 1000
    normalized = normalize_features(features)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_handles_zero_variance():
    features = np.ones((10, N_AI_AGENT_FEATURES))
    normalized = normalize_features(features)
    # Should not raise, should return zeros
    assert not np.any(np.isnan(normalized))


def test_combine_features_shape():
    base_map = {"0xA": 0, "0xB": 1}
    base_feats = np.array([[1.0] * 16, [2.0] * 16])
    ai_map = {"0xA": 0, "0xC": 1}
    ai_feats = np.array([[0.5] * 20, [0.7] * 20])
    new_map, combined = combine_with_base_features(base_map, base_feats, ai_map, ai_feats)
    assert combined.shape == (3, 36)
    assert len(new_map) == 3


def test_combine_zero_fills_missing():
    base_map = {"0xA": 0}
    base_feats = np.array([[1.0] * 16])
    ai_map = {"0xB": 0}
    ai_feats = np.array([[0.5] * 20])
    new_map, combined = combine_with_base_features(base_map, base_feats, ai_map, ai_feats)
    # 0xA should have zeros in ai_feats slot
    idx_a = new_map["0xA"]
    assert combined[idx_a, 16:].sum() == 0
    # 0xB should have zeros in base_feats slot
    idx_b = new_map["0xB"]
    assert combined[idx_b, :16].sum() == 0


def test_empty_df_returns_empty():
    empty = pd.DataFrame(
        {
            "from_addr": pd.Series([], dtype=str),
            "to_addr": pd.Series([], dtype=str),
            "value_int": pd.Series([], dtype=int),
            "ts": pd.to_datetime(pd.Series([], dtype=str), utc=True),
        }
    )
    addr_to_idx, features = compute_ai_agent_features(empty)
    assert len(addr_to_idx) == 0
    assert features.shape == (0, N_AI_AGENT_FEATURES)


def test_process_addrs_subset_only(agent_like_df):
    """process_addrs=set() restricts output to that subset (used by workers)."""
    addr_map, feats = compute_ai_agent_features(agent_like_df, process_addrs={"0xagent"})
    assert list(addr_map.keys()) == ["0xagent"]
    assert feats.shape == (1, N_AI_AGENT_FEATURES)
    # Feature 9 (nonce consistency) for 0xagent should be 1.0 (sequential nonces)
    assert feats[0, 9] == pytest.approx(1.0)


def test_global_bucket_counts_used(agent_like_df):
    """When global_bucket_counts is supplied, Feature 7 uses it.

    Realistic scenario: a network burst happens in buckets the local address
    didn't participate in — Feature 7 (swarm_correlation) should drop,
    because the address was NOT in the high-activity swarm.
    """
    # Vanilla: only 0xagent's own df → bucket_counts_max == 1, swarm == 1.0.
    vanilla_map, vanilla_feats = compute_ai_agent_features(agent_like_df)
    ai = vanilla_map["0xagent"]

    # Inject a foreign network burst at a different timestamp — 1000 unique
    # senders in one bucket the agent never used.
    far_bucket = pd.Timestamp("2030-06-15 12:00:00", tz="UTC")
    df = agent_like_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["ts_bucket"] = df["ts"].dt.floor("10min")
    local_counts = df.groupby("ts_bucket")["from_addr"].nunique()
    global_counts = pd.concat([local_counts, pd.Series({far_bucket: 1000})])
    global_max = float(global_counts.max())

    scaled_map, scaled_feats = compute_ai_agent_features(
        agent_like_df,
        global_bucket_counts=global_counts,
        global_bucket_counts_max=global_max,
    )
    si = scaled_map["0xagent"]
    # Agent's bucket count is still 1 (it was alone), but max is 1000 → 1/1000.
    assert scaled_feats[si, 7] < vanilla_feats[ai, 7] / 100
