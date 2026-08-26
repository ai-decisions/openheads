"""Tests for GNN bridge (nx_to_pyg converter)."""

import networkx as nx
import pytest

from openheads.bridge import HAS_TORCH, nx_to_pyg


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestNxToPyg:
    def test_basic_conversion(self):
        g = nx.Graph()
        g.add_node("a", anomaly_score=0.5)
        g.add_node("b", anomaly_score=0.1)
        g.add_edge("a", "b")

        data = nx_to_pyg(g)
        assert data is not None
        assert data.x.shape[0] == 2  # 2 nodes
        assert data.x.shape[1] == 5  # 5 structural features
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 2  # undirected = 2 directed edges
        assert data.node_names == ["a", "b"]

    def test_empty_graph(self):
        g = nx.Graph()
        data = nx_to_pyg(g)
        assert data is not None
        assert data.x.shape == (0, 5)
        assert data.edge_index.shape == (2, 0)

    def test_no_edges(self):
        g = nx.Graph()
        g.add_node("a")
        g.add_node("b")
        data = nx_to_pyg(g)
        assert data.x.shape[0] == 2
        assert data.edge_index.shape[1] == 0

    def test_structural_features_normalized(self):
        g = nx.Graph()
        for i in range(5):
            g.add_node(f"n{i}")
        g.add_edge("n0", "n1")
        g.add_edge("n1", "n2")
        g.add_edge("n2", "n3")
        g.add_edge("n3", "n4")
        data = nx_to_pyg(g)
        # 5 structural features, min-max normalized to [0, 1]
        assert data.x.shape == (5, 5)
        assert data.x.min().item() >= 0.0
        assert data.x.max().item() <= 1.0

    def test_node_to_idx_mapping(self):
        g = nx.Graph()
        for i in range(5):
            g.add_node(f"n{i}")
        data = nx_to_pyg(g)
        assert len(data.node_to_idx) == 5
        assert data.node_to_idx["n0"] == 0
