"""Unit tests for GNN training pipeline."""

import pytest

torch = pytest.importorskip("torch")

from openheads.data_prep import ShadowNetworkDataset  # noqa: E402
from openheads.models import GraphSAGEClassifier  # noqa: E402
from openheads.synthetic import SyntheticGraphGenerator  # noqa: E402
from openheads.trainer import GNNLightningModule, TrainingOrchestrator  # noqa: E402


def _make_small_data():
    """Create a small PyG Data for training tests."""
    gen = SyntheticGraphGenerator(seed=42)
    raw = gen.generate_shadow_network(n_nodes=100, n_communities=2)
    ds = ShadowNetworkDataset()
    return ds.from_dict(raw)


class TestGNNLightningModule:
    def test_forward(self) -> None:
        model = GraphSAGEClassifier(in_channels=169, hidden_channels=32, out_channels=3)
        module = GNNLightningModule(model=model, num_classes=3)
        data = _make_small_data()
        logits = module(data.x, data.edge_index)
        assert logits.shape == (100, 3)

    def test_training_step_returns_loss(self) -> None:
        model = GraphSAGEClassifier(in_channels=169, hidden_channels=32, out_channels=3)
        module = GNNLightningModule(model=model, num_classes=3, lr=0.01)
        data = _make_small_data()
        loss = module.training_step(data, 0)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0


class TestTrainingOrchestrator:
    def test_train_local_runs(self, tmp_path: object) -> None:
        """Training on tiny data completes and returns metrics."""
        data = _make_small_data()
        orchestrator = TrainingOrchestrator(output_dir=str(tmp_path))
        metrics = orchestrator.train_local(
            data,
            model_name="graphsage",
            max_epochs=3,
            hidden_channels=32,
            lr=0.01,
            accelerator="cpu",
        )
        assert "val_auroc" in metrics
        assert "model_path" in metrics
        assert metrics["model_name"] == "graphsage"

    def test_train_gat(self, tmp_path: object) -> None:
        """GAT model trains without errors."""
        data = _make_small_data()
        orchestrator = TrainingOrchestrator(output_dir=str(tmp_path))
        metrics = orchestrator.train_local(
            data,
            model_name="gat",
            max_epochs=2,
            hidden_channels=32,
            lr=0.01,
            accelerator="cpu",
        )
        assert metrics["model_name"] == "gat"

    def test_model_saved(self, tmp_path: object) -> None:
        """Model checkpoint is saved after training."""
        from pathlib import Path

        data = _make_small_data()
        orchestrator = TrainingOrchestrator(output_dir=str(tmp_path))
        metrics = orchestrator.train_local(
            data,
            model_name="graphsage",
            max_epochs=2,
            hidden_channels=32,
            accelerator="cpu",
        )
        assert Path(metrics["model_path"]).exists()
