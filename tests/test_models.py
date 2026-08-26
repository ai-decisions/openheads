"""Unit tests for GNN model architectures."""

import pytest

torch = pytest.importorskip("torch")

from openheads.models import (  # noqa: E402
    GATClassifier,
    GraphSAGEClassifier,
    HGTClassifier,
    create_model,
)


def _make_small_graph():
    """Create a tiny graph for forward pass tests."""
    n_nodes = 20
    n_features = 169
    x = torch.randn(n_nodes, n_features)
    # Simple chain + some random edges
    edges = [[i, i + 1] for i in range(n_nodes - 1)]
    edges += [[0, 5], [3, 10], [7, 15]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return x, edge_index


class TestGraphSAGE:
    def test_forward_shape(self) -> None:
        x, edge_index = _make_small_graph()
        model = GraphSAGEClassifier(in_channels=169, hidden_channels=64, out_channels=3)
        logits = model(x, edge_index)
        assert logits.shape == (20, 3)

    def test_embeddings_shape(self) -> None:
        x, edge_index = _make_small_graph()
        model = GraphSAGEClassifier(in_channels=169, hidden_channels=128)
        emb = model.get_embeddings(x, edge_index)
        assert emb.shape == (20, 128)

    def test_eval_mode(self) -> None:
        x, edge_index = _make_small_graph()
        model = GraphSAGEClassifier(in_channels=169, dropout=0.5)
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)


class TestGAT:
    def test_forward_shape(self) -> None:
        x, edge_index = _make_small_graph()
        model = GATClassifier(in_channels=169, hidden_channels=64, out_channels=3, heads=4)
        logits = model(x, edge_index)
        assert logits.shape == (20, 3)

    def test_embeddings_shape(self) -> None:
        x, edge_index = _make_small_graph()
        model = GATClassifier(in_channels=169, hidden_channels=128, heads=8)
        emb = model.get_embeddings(x, edge_index)
        assert emb.shape == (20, 128)


class TestHGT:
    def test_forward_shape(self) -> None:
        x, edge_index = _make_small_graph()
        model = HGTClassifier(in_channels=169, hidden_channels=64, out_channels=3)
        logits = model(x, edge_index)
        assert logits.shape == (20, 3)

    def test_embeddings_shape(self) -> None:
        x, edge_index = _make_small_graph()
        model = HGTClassifier(in_channels=169, hidden_channels=128)
        emb = model.get_embeddings(x, edge_index)
        assert emb.shape == (20, 128)


class TestModelRegistry:
    def test_create_graphsage(self) -> None:
        model = create_model("graphsage", in_channels=10)
        assert isinstance(model, GraphSAGEClassifier)

    def test_create_gat(self) -> None:
        model = create_model("gat", in_channels=10)
        assert isinstance(model, GATClassifier)

    def test_create_hgt(self) -> None:
        model = create_model("hgt", in_channels=10)
        assert isinstance(model, HGTClassifier)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent", in_channels=10)
