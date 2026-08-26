"""The backend-free loader must not scramble the graph.

This file exists because of a defect that passed every other gate: the
fallback loader put the seed rows first (permuting the feature matrix) and
handed on the ORIGINAL edge list. Message passing then aggregated over the
wrong neighbours. Nothing failed — the loss stayed finite, the checkpoint was
written and reloaded, the smoke printed both of its marker lines — and the
model had trained on a scrambled graph.

So the checks below are about correspondence, not about shapes:
row order, edge renumbering and the seeds-first contract are each asserted
against the original graph.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the loader yields torch tensors")
pytest.importorskip("torch_geometric", reason="graph container comes from torch_geometric")

from torch_geometric.data import Data  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "train_backbone", Path(__file__).resolve().parents[1] / "scripts" / "train_backbone.py"
)
train_backbone = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(train_backbone)

N_NODES = 10
N_FEATURES = 3


def _graph() -> Data:
    """A graph whose features identify their own row: x[i] == [i, i, i].

    That makes a permutation visible: if a row lands in the wrong place, its
    contents say so.
    """
    x = torch.arange(N_NODES, dtype=torch.float32).unsqueeze(1).repeat(1, N_FEATURES)
    ring = [(i, (i + 1) % N_NODES) for i in range(N_NODES)]
    graph = Data(x=x, edge_index=torch.tensor(ring, dtype=torch.long).t().contiguous())
    graph.num_nodes = N_NODES
    return graph


def _batches(seeds: list[int], batch_size: int) -> list:
    loader = train_backbone._FullGraphLoader(
        _graph(), torch.tensor(seeds, dtype=torch.int64), batch_size
    )
    return list(loader)


def test_feature_rows_follow_n_id() -> None:
    """Row k of x must be the features of node n_id[k], not of node k."""
    for batch in _batches([7, 3], batch_size=2):
        for position, node_id in enumerate(batch.n_id.tolist()):
            assert batch.x[position, 0].item() == pytest.approx(float(node_id))


def test_edges_are_renumbered_into_row_positions() -> None:
    """Every yielded edge, read back through n_id, must be an original edge."""
    original = {(int(a), int(b)) for a, b in _graph().edge_index.t().tolist()}
    for batch in _batches([7, 3], batch_size=2):
        n_id = batch.n_id.tolist()
        assert batch.edge_index.max().item() < len(n_id), "edge index out of row range"
        recovered = {(n_id[int(a)], n_id[int(b)]) for a, b in batch.edge_index.t().tolist()}
        assert recovered == original, "renumbering lost or invented edges"


def test_seeds_come_first_and_every_node_appears_once() -> None:
    seeds = [7, 3]
    for batch in _batches(seeds, batch_size=len(seeds)):
        assert batch.batch_size == len(seeds)
        assert batch.n_id[: batch.batch_size].tolist() == seeds
        assert sorted(batch.n_id.tolist()) == list(range(N_NODES))


def test_duplicate_seeds_are_refused_not_silently_merged() -> None:
    with pytest.raises(RuntimeError, match="duplicate seed"):
        _batches([4, 4], batch_size=2)


def test_adjacency_accessor_prefers_the_sparse_form() -> None:
    """`batch_adj` must not read edge_index off a batch that carries adj_t."""
    from openheads.graph_batching import batch_adj

    batch = next(iter(_batches([1], batch_size=1)))
    assert batch_adj(batch) is batch.edge_index  # adj_t is None here
    batch.adj_t = "sparse-placeholder"
    assert batch_adj(batch) == "sparse-placeholder"
