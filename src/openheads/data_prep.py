"""GNN Data Preparation — PyG Dataset classes for Elliptic and synthetic data.

EllipticDataset: loads Elliptic Bitcoin Dataset (203K tx, 166 features).
ShadowNetworkDataset: wraps any data dict into PyG Data with temporal split.
"""

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Data = None  # type: ignore[assignment,misc]
    InMemoryDataset = object  # type: ignore[assignment,misc]


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and torch-geometric required. "
            "Install with: pip install torch torch-geometric"
        )


# Elliptic class mapping: "1"=illicit→0, "2"=licit→1, "unknown"→2.
# Verified against elliptic_txs_classes.csv from the public dataset: "1" has
# 4,545 rows (the ~2% minority = illicit), "2" has 42,019, "unknown" 157,205.
ELLIPTIC_CLASS_MAP = {"1": 0, "2": 1, "unknown": 2}

# Index of the illicit class after the mapping above. Metrics that report an
# illicit-class figure must use this index, not 1.
ELLIPTIC_ILLICIT_INDEX = 0

# Temporal split boundaries
TRAIN_END = 34
VAL_END = 39
# Test: 40-49


class EllipticDataset:
    """Load and prepare Elliptic Bitcoin Dataset for GNN training.

    Expects CSV files in data_dir:
        - elliptic_txs_features.csv (203769 rows × 167 cols: txId + 166 features)
        - elliptic_txs_edgelist.csv (234355 rows × 2 cols: txId1, txId2)
        - elliptic_txs_classes.csv (203769 rows × 2 cols: txId, class)
    """

    def __init__(self, data_dir: str | Path = "data/raw/elliptic") -> None:
        _require_torch()
        self.data_dir = Path(data_dir)

    def load(self) -> "Data":
        """Load Elliptic dataset and return a PyG Data object."""
        import pandas as pd

        features_path = self.data_dir / "elliptic_txs_features.csv"
        edges_path = self.data_dir / "elliptic_txs_edgelist.csv"
        classes_path = self.data_dir / "elliptic_txs_classes.csv"

        for p in [features_path, edges_path, classes_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}. Run: python -m openheads.download_elliptic")

        # Load features (no header in Elliptic CSV)
        df_features = pd.read_csv(features_path, header=None)
        tx_ids = df_features.iloc[:, 0].values  # first col is txId
        # Col 1 = timestep, cols 2-167 = 166 features
        timesteps = df_features.iloc[:, 1].values.astype(np.int64)
        features = df_features.iloc[:, 2:].values.astype(np.float32)

        # Build txId → index mapping
        tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

        # Load classes
        df_classes = pd.read_csv(classes_path)
        labels = np.full(len(tx_ids), 2, dtype=np.int64)  # default = unknown
        for _, row in df_classes.iterrows():
            tx_id = row.iloc[0]
            cls_str = str(row.iloc[1]).strip()
            if tx_id in tx_to_idx:
                labels[tx_to_idx[tx_id]] = ELLIPTIC_CLASS_MAP.get(cls_str, 2)

        # Load edges
        df_edges = pd.read_csv(edges_path)
        edge_list = []
        for _, row in df_edges.iterrows():
            src = row.iloc[0]
            dst = row.iloc[1]
            if src in tx_to_idx and dst in tx_to_idx:
                edge_list.append([tx_to_idx[src], tx_to_idx[dst]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Add structural features (3 extra)
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(len(tx_ids)))
        for e in edge_list:
            g.add_edge(e[0], e[1])

        degree_c = nx.degree_centrality(g)
        # Approximate betweenness for large graphs (exact is O(n*m))
        k = min(100, g.number_of_nodes())
        betweenness_c = nx.betweenness_centrality(g, k=k)
        clustering_c = nx.clustering(g)

        structural = np.zeros((len(tx_ids), 3), dtype=np.float32)
        for i in range(len(tx_ids)):
            structural[i, 0] = degree_c.get(i, 0)
            structural[i, 1] = betweenness_c.get(i, 0)
            structural[i, 2] = clustering_c.get(i, 0)

        all_features = np.concatenate([features, structural], axis=1)

        data = Data(
            x=torch.tensor(all_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
        )
        data.timesteps = torch.tensor(timesteps, dtype=torch.long)

        # Temporal masks
        data.train_mask = data.timesteps <= TRAIN_END
        data.val_mask = (data.timesteps > TRAIN_END) & (data.timesteps <= VAL_END)
        data.test_mask = data.timesteps > VAL_END

        self._log_stats(data, labels)
        return data

    @staticmethod
    def _log_stats(data: "Data", labels: np.ndarray) -> None:
        logger.info(
            "elliptic_loaded",
            nodes=data.num_nodes,
            edges=data.num_edges,
            features=data.x.shape[1],
            licit=int(np.sum(labels == 0)),
            illicit=int(np.sum(labels == 1)),
            unknown=int(np.sum(labels == 2)),
            train=int(data.train_mask.sum()),
            val=int(data.val_mask.sum()),
            test=int(data.test_mask.sum()),
        )


class ShadowNetworkDataset:
    """Convert a synthetic data dict into a PyG Data object with temporal split."""

    def __init__(self) -> None:
        _require_torch()

    def from_dict(self, data_dict: dict[str, Any]) -> "Data":
        """Convert synthetic generator output to PyG Data."""
        node_features = data_dict["node_features"]
        edge_index = data_dict["edge_index"]
        labels = data_dict["labels"]
        timesteps = data_dict["timesteps"]

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
        )
        data.timesteps = torch.tensor(timesteps, dtype=torch.long)

        # Temporal masks
        data.train_mask = data.timesteps <= TRAIN_END
        data.val_mask = (data.timesteps > TRAIN_END) & (data.timesteps <= VAL_END)
        data.test_mask = data.timesteps > VAL_END

        logger.info(
            "shadow_dataset_created",
            nodes=data.num_nodes,
            edges=data.num_edges,
            features=data.x.shape[1],
            train=int(data.train_mask.sum()),
            val=int(data.val_mask.sum()),
            test=int(data.test_mask.sum()),
        )
        return data
