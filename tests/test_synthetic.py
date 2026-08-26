"""Unit tests for SyntheticGraphGenerator."""

import numpy as np

from openheads.synthetic import SyntheticGraphGenerator


class TestShadowNetwork:
    def test_generates_correct_node_count(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_shadow_network(n_nodes=100, n_communities=3)
        assert data["node_features"].shape[0] == 100

    def test_feature_dimensions(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_shadow_network(n_nodes=50)
        assert data["node_features"].shape[1] == 169  # 166 + 3 structural

    def test_edge_index_shape(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_shadow_network(n_nodes=100)
        assert data["edge_index"].shape[0] == 2
        assert data["edge_index"].shape[1] > 0

    def test_labels_distribution(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_shadow_network(n_nodes=200, illicit_fraction=0.1)
        labels = data["labels"]
        assert set(np.unique(labels)).issubset({0, 1, 2})
        assert np.sum(labels == 0) > 0  # licit
        assert np.sum(labels == 1) > 0  # illicit

    def test_timesteps_range(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_shadow_network(n_nodes=100)
        ts = data["timesteps"]
        assert ts.min() >= 1
        assert ts.max() <= 49

    def test_hub_nodes_present(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_shadow_network(n_nodes=100, hub_fraction=0.1)
        assert len(data["hub_nodes"]) >= 1

    def test_deterministic_with_seed(self) -> None:
        gen1 = SyntheticGraphGenerator(seed=123)
        gen2 = SyntheticGraphGenerator(seed=123)
        d1 = gen1.generate_shadow_network(n_nodes=50)
        d2 = gen2.generate_shadow_network(n_nodes=50)
        np.testing.assert_array_equal(d1["labels"], d2["labels"])


class TestAgentPatterns:
    def test_generates_correct_structure(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_agent_patterns(n_agents=5, fan_size=3)
        expected_nodes = 5 * (1 + 3 * 2)  # 5 agents × 7 nodes each
        assert data["node_features"].shape[0] == expected_nodes

    def test_edge_count(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_agent_patterns(n_agents=3, fan_size=4)
        # Each agent: fan_size fan-out + fan_size fan-in = 2 * fan_size edges
        expected_edges = 3 * 2 * 4
        assert data["edge_index"].shape[1] == expected_edges

    def test_transit_nodes_illicit(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_agent_patterns(n_agents=5, fan_size=3)
        nodes_per_agent = 1 + 3 * 2
        for i in range(5):
            transit = i * nodes_per_agent
            assert data["labels"][transit] == 1  # transit = illicit

    def test_agent_timesteps_late(self) -> None:
        gen = SyntheticGraphGenerator(seed=42)
        data = gen.generate_agent_patterns(n_agents=3)
        assert data["timesteps"].min() >= 40
