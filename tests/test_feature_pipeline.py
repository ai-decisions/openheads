"""Tests for GNN feature pipeline."""

import pandas as pd
import pytest

from openheads.feature_pipeline import (
    N_FEATURES,
    build_edge_index,
    compute_node_features,
    load_parquet_transactions,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "from_address": ["0xAAA", "0xAAA", "0xBBB", "0xCCC", "0xAAA"],
            "to_address": ["0xBBB", "0xCCC", "0xCCC", "0xAAA", "0xDDD"],
            "value": [
                1000000000000000000,
                2000000000000000000,
                500000000000000000,
                100000000000000000,
                10000000000000000000,
            ],
            "block_timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        }
    )


def test_load_parquet_creates_correct_columns(tmp_path):
    df = pd.DataFrame(
        {
            "from_address": ["0xA", "0xB"],
            "to_address": ["0xB", "0xC"],
            "value": [1000000000000000000, 2000000000000000000],
            "block_timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        }
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    result = load_parquet_transactions([str(path)])
    assert "from_addr" in result.columns
    assert "to_addr" in result.columns
    assert "value_eth" in result.columns
    assert len(result) == 2


def test_compute_node_features_returns_correct_shape(tmp_path):
    df = pd.DataFrame(
        {
            "from_address": ["0xAAA", "0xAAA", "0xBBB"],
            "to_address": ["0xBBB", "0xCCC", "0xCCC"],
            "value": [1000000000000000000, 2000000000000000000, 500000000000000000],
            "block_timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        }
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    loaded = load_parquet_transactions([str(path)])
    addr_to_idx, features = compute_node_features(loaded)
    assert len(addr_to_idx) == 3
    assert len(features) == 3
    assert len(features[0]) == N_FEATURES


def test_build_edge_index(tmp_path):
    df = pd.DataFrame(
        {
            "from_address": ["0xaaa", "0xaaa", "0xbbb"],
            "to_address": ["0xbbb", "0xccc", "0xccc"],
            "value": [1000000000000000000, 2000000000000000000, 500000000000000000],
            "block_timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        }
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    loaded = load_parquet_transactions([str(path)])
    addr_to_idx, _ = compute_node_features(loaded)
    src, dst = build_edge_index(loaded, addr_to_idx)
    assert len(src) == len(dst)
    assert len(src) > 0
