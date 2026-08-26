"""Adjacency extraction for a neighbour-sampled batch."""

from __future__ import annotations

from typing import Any


def batch_adj(batch: Any) -> Any:
    """Return the adjacency a message-passing layer should consume.

    A sampled batch carries its adjacency either as a sparse ``adj_t`` (when
    the graph was built with one) or as ``edge_index``. SAGEConv accepts
    both, so the choice is invisible at the call site — but reading
    ``edge_index`` from a batch built with ``adj_t`` yields an empty tensor
    rather than an error, and the layer then trains on a graph with no edges.
    That failure is silent: the loss still falls, the model just learns
    nothing from structure. Hence one accessor, used everywhere.
    """
    adj_t = getattr(batch, "adj_t", None)
    if adj_t is not None:
        return adj_t
    return batch.edge_index
