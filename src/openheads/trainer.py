"""GNN Training Pipeline — PyTorch Lightning module + orchestrator."""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import structlog
import torch
from torch import Tensor
from torch.nn import functional as f
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)

from openheads.data_prep import ELLIPTIC_ILLICIT_INDEX
from openheads.models import create_model

logger = structlog.get_logger(__name__)


class GNNLightningModule(pl.LightningModule):
    """Wraps any GNN model for training with pytorch-lightning."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 3,
        class_weights: Tensor | None = None,
        lr: float = 1e-3,
        illicit_class_index: int = ELLIPTIC_ILLICIT_INDEX,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        # Which class index `val_f1_illicit` reports. Defaults to the Elliptic
        # convention (ELLIPTIC_ILLICIT_INDEX = 0); datasets that put illicit at
        # another index must pass it explicitly.
        self.illicit_class_index = illicit_class_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Metrics
        self.train_auroc = MulticlassAUROC(num_classes=num_classes)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        self.val_ap = MulticlassAveragePrecision(num_classes=num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.model(x, edge_index)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        logits = self.model(batch.x, batch.edge_index)
        mask = (
            batch.train_mask
            if hasattr(batch, "train_mask")
            else torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.y.device)
        )
        # y == -1 marks unlabelled nodes (feature_pipeline.build_pyg_data
        # ships them that way); cross_entropy and bincount both blow up on a
        # negative target, so the loss only ever sees labelled rows.
        mask = mask & (batch.y >= 0)
        loss = f.cross_entropy(logits[mask], batch.y[mask], weight=self.class_weights)
        probs = f.softmax(logits[mask], dim=1)
        self.train_auroc.update(probs, batch.y[mask])
        self.log("train_loss", loss, prog_bar=True, batch_size=mask.sum().item())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        logits = self.model(batch.x, batch.edge_index)
        mask = (
            batch.val_mask
            if hasattr(batch, "val_mask")
            else torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.y.device)
        )
        mask = mask & (batch.y >= 0)  # never score unlabelled (-1) rows
        loss = f.cross_entropy(logits[mask], batch.y[mask])
        probs = f.softmax(logits[mask], dim=1)
        self.val_auroc.update(probs, batch.y[mask])
        self.val_f1.update(probs, batch.y[mask])
        self.val_ap.update(probs, batch.y[mask])
        self.log("val_loss", loss, prog_bar=True, batch_size=mask.sum().item())

    def on_validation_epoch_end(self) -> None:
        auroc = self.val_auroc.compute()
        f1_per_class = self.val_f1.compute()
        ap = self.val_ap.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_f1_illicit", f1_per_class[self.illicit_class_index], prog_bar=True)
        self.log("val_ap", ap, prog_bar=True)
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_ap.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TrainingOrchestrator:
    """Runs GNN training locally on a single machine."""

    def __init__(self, output_dir: str = "data/processed/models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_local(
        self,
        data: Data,
        model_name: str = "graphsage",
        max_epochs: int = 50,
        hidden_channels: int = 256,
        lr: float = 1e-3,
        dropout: float = 0.3,
        accelerator: str = "auto",
    ) -> dict[str, Any]:
        """Train a GNN model locally. Returns metrics dict.

        Accepts `feature_pipeline.build_pyg_data` output directly: unlabelled
        nodes arrive as y == -1 and are excluded everywhere, and when the Data
        carries no train/val masks a seeded 80/20 split over the LABELLED
        nodes is derived here — the two shipped stages compose without an
        undocumented manual step in between.
        """
        in_channels = data.x.shape[1]
        labelled = data.y >= 0
        if not bool(labelled.any()):
            raise ValueError("data.y carries no labelled nodes (every y is -1)")
        num_classes = int(data.y[labelled].max().item()) + 1

        model = create_model(
            model_name,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
        )

        has_train = getattr(data, "train_mask", None) is not None
        has_val = getattr(data, "val_mask", None) is not None
        if has_train != has_val:
            # Deriving over a half-supplied pair would silently overwrite the
            # caller's mask; the mismatch is theirs to resolve.
            raise ValueError(
                "data carries only one of train_mask/val_mask — supply both "
                "or neither (masks are derived only when both are absent)"
            )
        if not has_train:
            labelled_idx = labelled.nonzero(as_tuple=True)[0]
            if len(labelled_idx) < 2:
                raise ValueError(
                    f"{len(labelled_idx)} labelled node(s) cannot support a "
                    "train/val split — at least 2 are required"
                )
            gen = torch.Generator().manual_seed(42)
            perm = labelled_idx[torch.randperm(len(labelled_idx), generator=gen)]
            n_val = max(1, int(len(perm) * 0.2))
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask[perm[:n_val]] = True
            train_mask[perm[n_val:]] = True
            data.train_mask, data.val_mask = train_mask, val_mask

        # Compute class weights from LABELLED training rows only — bincount
        # raises on the -1 sentinel, and a user-supplied mask may include it.
        train_labels = data.y[data.train_mask & labelled]
        class_counts = torch.bincount(train_labels, minlength=num_classes).float()
        class_counts = class_counts.clamp(min=1)
        class_weights = (1.0 / class_counts) / (1.0 / class_counts).sum() * num_classes

        lightning_module = GNNLightningModule(
            model=model,
            num_classes=num_classes,
            class_weights=class_weights,
            lr=lr,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            # Everything this run writes goes under output_dir. Without
            # default_root_dir, Lightning writes its checkpoints into the
            # CURRENT WORKING DIRECTORY — a library dropping files into the
            # caller's repository, ignoring the output path it was given.
            default_root_dir=str(self.output_dir),
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_auroc", patience=10, mode="max"),
                pl.callbacks.ModelCheckpoint(dirpath=str(self.output_dir / "checkpoints")),
            ],
        )

        # For small graphs, use full-batch; for large, use NeighborLoader
        if data.num_nodes <= 5000:
            train_loader = DataLoader([data], batch_size=1)
            val_loader = DataLoader([data], batch_size=1)
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10, 10],
                batch_size=256,
                input_nodes=data.train_mask,
            )
            val_loader = NeighborLoader(
                data,
                num_neighbors=[10, 10, 10],
                batch_size=256,
                input_nodes=data.val_mask,
            )

        trainer.fit(lightning_module, train_loader, val_loader)

        # Save model
        model_path = self.output_dir / f"{model_name}_best.pt"
        torch.save(model.state_dict(), model_path)

        # Extract final metrics
        metrics = {
            "model_name": model_name,
            "epochs_trained": trainer.current_epoch,
            "val_auroc": trainer.callback_metrics.get("val_auroc", 0),
            "val_f1_illicit": trainer.callback_metrics.get("val_f1_illicit", 0),
            "val_ap": trainer.callback_metrics.get("val_ap", 0),
            "model_path": str(model_path),
        }

        # Convert tensors to float for JSON serialization
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                metrics[k] = round(float(v.item()), 4)

        logger.info("training_complete", **metrics)
        return metrics

    def hyperparameter_search(
        self,
        data: Data,
        n_trials: int = 20,
        max_epochs: int = 30,
    ) -> dict[str, Any]:
        """Optuna HPO search. Returns best params + metrics."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            hidden = trial.suggest_categorical("hidden_channels", [128, 256, 512])
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            model_name = trial.suggest_categorical("model_name", ["graphsage", "gat"])

            metrics = self.train_local(
                data,
                model_name=model_name,
                max_epochs=max_epochs,
                hidden_channels=hidden,
                lr=lr,
                dropout=dropout,
            )
            val_auroc = metrics.get("val_auroc", 0)
            return float(val_auroc) if isinstance(val_auroc, int | float) else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        logger.info(
            "hpo_complete",
            best_auroc=best.value,
            best_params=best.params,
            n_trials=len(study.trials),
        )

        return {
            "best_auroc": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials),
        }
