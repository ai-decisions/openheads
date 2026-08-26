"""GNN Model Architectures — GraphSAGE, GAT, HGT for node classification."""

import structlog
import torch
from torch import Tensor, nn
from torch.nn import functional as f
from torch_geometric.nn import GATConv, SAGEConv

logger = structlog.get_logger(__name__)


class GraphSAGEClassifier(nn.Module):
    """3-layer GraphSAGE with BatchNorm + dropout for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass returning logits [batch_size, out_channels]."""
        x = self._encode(x, edge_index)
        return self.classifier(x)

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return penultimate layer embeddings [batch_size, hidden_channels]."""
        return self._encode(x, edge_index)

    def _encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = f.relu(x)
        return x


class GATClassifier(nn.Module):
    """3-layer GAT with multi-head attention for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        self.conv3 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self._encode(x, edge_index)
        return self.classifier(x)

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self._encode(x, edge_index)

    def _encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = f.elu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = f.elu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = f.elu(x)
        return x


class HGTClassifier(nn.Module):
    """Simplified HGT placeholder — single-type graph, same interface.

    Full heterogeneous support (multi-type nodes/edges) planned for a future
    release. Currently uses SAGEConv layers as backbone with the same API.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self._encode(x, edge_index)
        return self.classifier(x)

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self._encode(x, edge_index)

    def _encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = f.relu(x)
        return x


class PerChainNormEncoder(nn.Module):
    """GraphSAGE encoder with per-chain LayerNorm on the output embedding.

    When several chains are merged into one node space, a shared encoder can
    place each chain's embeddings in a separate cluster with systematically
    different norms, and post-hoc linear normalisation cannot repair that
    misalignment. Here each chain gets its own learnable LayerNorm INSIDE the
    architecture, so the alignment is trained end-to-end, not glued on
    afterwards.

    ``chain_starts`` — sorted node-index offsets of each chain's partition
    in the merged graph, e.g. ``[0, n_nodes_of_chain_0]`` for a two-chain
    merge. ``chain_id(node) = bucketize(node, chain_starts, right=True) - 1``.
    Generalises to any number of chains by passing more offsets — no code
    change.

    ``tail_start``/``tail_chain`` — tail routing: when extra rows belonging
    to one chain (e.g. a token-transfer tail) are appended AFTER the last
    chain partition, plain bucketize would route those rows to the LAST
    chain's LayerNorm. With ``tail_start`` set, every id >= tail_start is
    routed to ``chain_norms[tail_chain]`` instead. Plain attributes, NOT
    buffers: state_dict stays key-identical with checkpoints trained without
    a tail in both directions (strict load works verbatim); the routing is
    recorded by the caller's metadata, not the checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        chain_starts: list[int] | None = None,
        dropout: float = 0.3,
        tail_start: int | None = None,
        tail_chain: int = 0,
    ) -> None:
        super().__init__()
        # Sortedness is checked, not assumed: chain ids come from
        # torch.bucketize, which requires ascending boundaries. Given
        # [0, 500, 100] it returns silently wrong ids — nodes 100..499 route to
        # chain 0's normalisation and the middle chain gets no rows at all.
        # The offsets usually arrive from configuration, so one transposed
        # digit would otherwise produce a trained model with the wrong
        # per-chain statistics rather than an error.
        if not chain_starts or chain_starts[0] != 0:
            raise ValueError(
                f"chain_starts must be a sorted list starting at 0, got {chain_starts!r}"
            )
        if any(b <= a for a, b in zip(chain_starts, chain_starts[1:], strict=False)):
            raise ValueError(
                f"chain_starts must be strictly ascending, got {chain_starts!r}"
            )
        if tail_start is not None:
            if tail_start <= chain_starts[-1]:
                raise ValueError(
                    f"tail_start {tail_start} must lie beyond the last chain "
                    f"start {chain_starts[-1]}"
                )
            if not 0 <= tail_chain < len(chain_starts):
                raise ValueError(f"tail_chain {tail_chain} out of range")
        self.tail_start = tail_start
        self.tail_chain = tail_chain
        self.backbone = GraphSAGEClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=3,  # classifier head unused; embeddings only
            dropout=dropout,
        )
        self.n_chains = len(chain_starts)
        self.chain_norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(self.n_chains)
        )
        # Buffer (not Parameter): saved in state_dict so a checkpoint is
        # self-describing about the partition it was trained on.
        self.register_buffer(
            "chain_starts",
            torch.tensor(chain_starts, dtype=torch.int64),
            persistent=True,
        )

    def chain_ids_of(self, global_node_ids: Tensor) -> Tensor:
        """Map global merged-graph node ids to chain ids [0..n_chains)."""
        cids = (torch.bucketize(global_node_ids, self.chain_starts, right=True) - 1).clamp_(min=0)
        if self.tail_start is not None:
            cids = torch.where(
                global_node_ids >= self.tail_start,
                cids.new_full((), self.tail_chain),
                cids,
            )
        return cids

    def get_embeddings(self, x: Tensor, edge_index: Tensor, global_node_ids: Tensor) -> Tensor:
        """Per-chain-normalised embeddings [batch, hidden].

        ``global_node_ids`` must align rows of ``x`` (NeighborLoader's
        ``batch.n_id``) — the chain of every subgraph row is derived from
        its MERGED-graph id, never from features.
        """
        emb = self.backbone.get_embeddings(x, edge_index)
        cids = self.chain_ids_of(global_node_ids.to(emb.device))
        out = emb.new_empty(emb.shape)
        for c, ln in enumerate(self.chain_norms):
            mask = cids == c
            if mask.any():
                out[mask] = ln(emb[mask])
        return out


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "graphsage": GraphSAGEClassifier,
    "gat": GATClassifier,
    "hgt": HGTClassifier,
}


def create_model(
    name: str,
    in_channels: int,
    hidden_channels: int = 256,
    out_channels: int = 3,
    **kwargs: int | float,
) -> nn.Module:
    """Create a GNN model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        **kwargs,
    )
