"""GNN Inference — load trained model, score graphs, update entities."""

from pathlib import Path
from typing import Any

import structlog

from openheads.data_prep import ELLIPTIC_ILLICIT_INDEX

logger = structlog.get_logger(__name__)

try:
    import torch
    from torch_geometric.data import Data

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Data = None  # type: ignore[assignment,misc]


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and torch-geometric required. "
            "Install with: pip install torch torch-geometric"
        )


class GNNScorer:
    """Load a trained GNN checkpoint and score graph nodes."""

    def __init__(
        self,
        model_path: str | Path,
        model_name: str = "graphsage",
        in_channels: int = 169,
        hidden_channels: int = 256,
        out_channels: int | None = None,
        illicit_class_index: int = ELLIPTIC_ILLICIT_INDEX,
        **model_kwargs: int | float,
    ) -> None:
        _require_torch()
        from openheads.models import create_model

        # Which class index carries P(illicit). The label space is a property
        # of the checkpoint: Elliptic-convention checkpoints put illicit at 0;
        # a checkpoint trained with another convention must pass its index
        # explicitly.
        self.illicit_class_index = illicit_class_index

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Auto-detect out_channels from checkpoint classifier layer
        if out_channels is None:
            if "classifier.weight" not in state_dict:
                raise ValueError(
                    f"Cannot auto-detect out_channels: 'classifier.weight' not in checkpoint. "
                    f"Keys: {list(state_dict.keys())[:5]}"
                )
            out_channels = state_dict["classifier.weight"].shape[0]

        # Auto-detect hidden_channels from first conv layer
        if "convs.0.lin_l.weight" in state_dict:
            hidden_channels = state_dict["convs.0.lin_l.weight"].shape[0]

        self.model = create_model(
            model_name,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **model_kwargs,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.out_channels = out_channels
        logger.info(
            "gnn_scorer_loaded",
            model_name=model_name,
            model_path=str(model_path),
            out_channels=out_channels,
        )

    def score_graph(self, data: "Data") -> dict[int, dict[str, Any]]:
        """Score all nodes in a graph.

        Returns dict: node_idx -> {anomaly_score, embedding, predicted_class, probabilities}
        """
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            probs = torch.softmax(logits, dim=1)
            embeddings = self.model.get_embeddings(data.x, data.edge_index)

        results: dict[int, dict[str, Any]] = {}
        n_classes = probs.shape[1]
        for i in range(data.num_nodes):
            prob_vec = probs[i].cpu().numpy()
            # P(illicit) sits at self.illicit_class_index (Elliptic convention:
            # 0). Under that convention prob_vec[1] is the LICIT class, so
            # reading index 1 here would invert anomaly_score.
            illicit_ix = self.illicit_class_index
            licit_ix = 1 - illicit_ix  # licit/illicit occupy {0, 1}; unknown (if any) is 2
            anomaly = float(prob_vec[illicit_ix]) if n_classes >= 2 else 0.0
            probabilities = {
                "licit": round(float(prob_vec[licit_ix]), 4),
                "illicit": round(anomaly, 4),
            }
            if n_classes >= 3:
                probabilities["unknown"] = round(float(prob_vec[2]), 4)
            results[i] = {
                "anomaly_score": round(anomaly, 4),
                "embedding": embeddings[i].cpu().numpy().tolist(),
                "predicted_class": int(probs[i].argmax().item()),
                "probabilities": probabilities,
            }

        logger.info(
            "graph_scored",
            nodes=len(results),
            high_risk=sum(1 for r in results.values() if r["anomaly_score"] > 0.5),
        )
        return results

    def update_entities(
        self,
        scores: dict[int, dict[str, Any]],
        entity_store: dict[str, Any],
        node_to_entity: dict[int, str],
    ) -> int:
        """Write anomaly scores and embeddings to entity store.

        Args:
            scores: output from score_graph()
            entity_store: dict-like storage {entity_id: entity_dict}
            node_to_entity: mapping from node index to entity_id

        Returns:
            number of entities updated
        """
        updated = 0
        for node_idx, entity_id in node_to_entity.items():
            if node_idx not in scores:
                continue
            if entity_id not in entity_store:
                continue

            node_scores = scores[node_idx]
            entity = entity_store[entity_id]
            entity["anomaly_score"] = node_scores["anomaly_score"]
            entity["gnn_embedding"] = node_scores["embedding"]
            entity["predicted_class"] = node_scores["predicted_class"]
            updated += 1

        logger.info("entities_updated", count=updated)
        return updated
