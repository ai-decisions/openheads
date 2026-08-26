"""Unit tests for GNN data preparation (uses synthetic data, no Elliptic download)."""

import numpy as np
import pytest

from openheads.synthetic import SyntheticGraphGenerator

# Skip all tests if torch not installed
torch = pytest.importorskip("torch")
from openheads.data_prep import (  # noqa: E402
    TRAIN_END,
    VAL_END,
    ShadowNetworkDataset,
)


class TestShadowNetworkDataset:
    def test_from_dict_creates_data(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_shadow_network(n_nodes=100)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        assert data.num_nodes == 100
        assert data.x.shape == (100, 169)
        assert data.y.shape == (100,)

    def test_edge_index_tensor(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_shadow_network(n_nodes=50)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        assert data.edge_index.dtype == torch.long
        assert data.edge_index.shape[0] == 2

    def test_temporal_masks_non_overlapping(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_shadow_network(n_nodes=200)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        # Masks should not overlap
        overlap_train_val = (data.train_mask & data.val_mask).sum().item()
        overlap_train_test = (data.train_mask & data.test_mask).sum().item()
        overlap_val_test = (data.val_mask & data.test_mask).sum().item()
        assert overlap_train_val == 0
        assert overlap_train_test == 0
        assert overlap_val_test == 0

    def test_temporal_masks_cover_all(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_shadow_network(n_nodes=200)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        total = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
        assert total.item() == 200

    def test_temporal_split_boundaries(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_shadow_network(n_nodes=500)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        train_ts = data.timesteps[data.train_mask]
        val_ts = data.timesteps[data.val_mask]
        test_ts = data.timesteps[data.test_mask]

        if train_ts.numel() > 0:
            assert train_ts.max().item() <= TRAIN_END
        if val_ts.numel() > 0:
            assert val_ts.min().item() > TRAIN_END
            assert val_ts.max().item() <= VAL_END
        if test_ts.numel() > 0:
            assert test_ts.min().item() > VAL_END

    def test_labels_preserved(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_shadow_network(n_nodes=100)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        np.testing.assert_array_equal(data.y.numpy(), raw["labels"])

    def test_agent_data_converts(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        raw = gen.generate_agent_patterns(n_agents=5, fan_size=3)
        dataset = ShadowNetworkDataset()
        data = dataset.from_dict(raw)

        assert data.num_nodes == 5 * 7  # 5 agents × (1 + 3 + 3) nodes
        assert data.edge_index.shape[1] == 5 * 6  # 5 agents × 2×3 edges
