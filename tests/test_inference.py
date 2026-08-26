"""Unit tests for GNN inference (scorer + entity update)."""

import pytest

torch = pytest.importorskip("torch")

from openheads.data_prep import ShadowNetworkDataset  # noqa: E402
from openheads.inference import GNNScorer  # noqa: E402
from openheads.synthetic import SyntheticGraphGenerator  # noqa: E402


def _train_and_save(tmp_path):
    """Train a tiny model and save checkpoint."""
    from openheads.trainer import TrainingOrchestrator

    gen = SyntheticGraphGenerator(seed=42)
    raw = gen.generate_shadow_network(n_nodes=100, n_communities=2)
    ds = ShadowNetworkDataset()
    data = ds.from_dict(raw)

    orchestrator = TrainingOrchestrator(output_dir=str(tmp_path))
    metrics = orchestrator.train_local(
        data,
        model_name="graphsage",
        max_epochs=2,
        hidden_channels=32,
        accelerator="cpu",
    )
    return data, metrics["model_path"]


class TestGNNScorer:
    def test_loads_checkpoint(self, tmp_path) -> None:
        """Scorer loads a saved model checkpoint."""
        _data, model_path = _train_and_save(tmp_path)
        scorer = GNNScorer(
            model_path=model_path,
            model_name="graphsage",
            in_channels=169,
            hidden_channels=32,
            out_channels=3,
        )
        assert scorer.model is not None

    def test_score_graph_returns_all_nodes(self, tmp_path) -> None:
        """score_graph returns scores for every node."""
        data, model_path = _train_and_save(tmp_path)
        scorer = GNNScorer(
            model_path=model_path,
            model_name="graphsage",
            in_channels=169,
            hidden_channels=32,
            out_channels=3,
        )
        scores = scorer.score_graph(data)
        assert len(scores) == 100

    def test_score_structure(self, tmp_path) -> None:
        """Each node score has expected keys."""
        data, model_path = _train_and_save(tmp_path)
        scorer = GNNScorer(
            model_path=model_path,
            model_name="graphsage",
            in_channels=169,
            hidden_channels=32,
            out_channels=3,
        )
        scores = scorer.score_graph(data)
        node_score = scores[0]
        assert "anomaly_score" in node_score
        assert "embedding" in node_score
        assert "predicted_class" in node_score
        assert "probabilities" in node_score
        assert 0.0 <= node_score["anomaly_score"] <= 1.0
        assert len(node_score["embedding"]) == 32  # hidden_channels
        assert node_score["predicted_class"] in {0, 1, 2}

    def test_probabilities_sum_to_one(self, tmp_path) -> None:
        """Probabilities for each node sum to ~1."""
        data, model_path = _train_and_save(tmp_path)
        scorer = GNNScorer(
            model_path=model_path,
            model_name="graphsage",
            in_channels=169,
            hidden_channels=32,
            out_channels=3,
        )
        scores = scorer.score_graph(data)
        for node_score in scores.values():
            total = sum(node_score["probabilities"].values())
            assert abs(total - 1.0) < 0.01


class TestUpdateEntities:
    def test_updates_entity_store(self, tmp_path) -> None:
        """update_entities writes scores to entity store."""
        data, model_path = _train_and_save(tmp_path)
        scorer = GNNScorer(
            model_path=model_path,
            model_name="graphsage",
            in_channels=169,
            hidden_channels=32,
            out_channels=3,
        )
        scores = scorer.score_graph(data)

        entity_store = {"entity_a": {}, "entity_b": {}}
        node_to_entity = {0: "entity_a", 5: "entity_b"}

        updated = scorer.update_entities(scores, entity_store, node_to_entity)
        assert updated == 2
        assert "anomaly_score" in entity_store["entity_a"]
        assert "gnn_embedding" in entity_store["entity_b"]

    def test_skips_missing_entities(self, tmp_path) -> None:
        """update_entities skips nodes without matching entity."""
        data, model_path = _train_and_save(tmp_path)
        scorer = GNNScorer(
            model_path=model_path,
            model_name="graphsage",
            in_channels=169,
            hidden_channels=32,
            out_channels=3,
        )
        scores = scorer.score_graph(data)

        entity_store = {"entity_a": {}}
        node_to_entity = {0: "entity_a", 5: "entity_missing"}

        updated = scorer.update_entities(scores, entity_store, node_to_entity)
        assert updated == 1
