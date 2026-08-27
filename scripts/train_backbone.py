#!/usr/bin/env python3
"""Supervised backbone retrain over a neighbour-sampled merged graph.

Warm-starts the encoder from a previous backbone checkpoint and trains it
jointly with fresh heads. Two details here are the whole point of the file:

WARM START, STRICTLY. The checkpoint is loaded with ``strict=True``. A
non-strict load silently accepts a checkpoint whose keys or shapes have
drifted, leaves the mismatched tensors randomly initialised, and still
trains — producing a run that looks like a warm start and is not. The heads
are deliberately NOT warm-started: they are cheap to relearn and carry the
label distribution of the previous run.

TAIL ROUTING. When rows of one chain are appended AFTER the last chain
partition of a merged graph, a plain partition lookup routes them to the
LAST chain's normalisation. ``tail_start`` sends them to the chain they
actually belong to. This is a plain attribute of the encoder, not a buffer,
so the state_dict stays key-identical with checkpoints trained without a
tail — loading works in both directions.

Smoke modes (run the cheap one before committing to a full pass):
  --synthetic       tiny random graph in memory, no inputs, no GPU, no object
                    storage; exercises the strict self-load, the tail-routing
                    gate, and a checkpoint write/read round-trip.
  OPENHEADS_SMOKE=1 subset of the real inputs (caps below).

Env for a real run (all paths, no defaults that point anywhere private):
  OPENHEADS_FEATURES     node-feature tensor (.pt)
  OPENHEADS_CSR_DIR      directory with colptr.pt / row.pt
  OPENHEADS_POOL_DIR     directory with nodes.npy / y_*.npy
  OPENHEADS_WARM_CKPT    encoder checkpoint to warm-start from
  OPENHEADS_OUT_DIR      output directory (default ./runs/backbone)
  OPENHEADS_CHAIN_STARTS comma-separated node-id offsets, one per chain
  OPENHEADS_TAIL_START   optional: first id of the appended tail
  OPENHEADS_TAIL_CHAIN   chain index the tail rows are routed to (default 0)
  OPENHEADS_N_NODES      node count of the merged graph
  OPENHEADS_INPUT_PINS   optional json {name: sha256} to verify inputs against
  OPENHEADS_HEAD_TAG     name tag of the per-head export files (default "all")
  OPENHEADS_MARKER_DIR   where SUCCESS/FAIL marker files go (default tempdir)
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from openheads.heads import EMBED_DIM

# ---------------------------------------------------------------------------
# RECIPE CONSTANTS — carried verbatim between runs; do not tune casually.
# Changing one of these makes a run incomparable with earlier numbers.
# ---------------------------------------------------------------------------
LR_BACKBONE = 5e-4
LR_HEADS = 1e-3
BATCH_SIZE = 2048
FANOUT = [25, 10]
MAX_EPOCHS = 3
INFER_BATCH = 4096
PLATEAU_DEGRADE_MARGIN = 0.01
HEARTBEAT_EVERY_N_BATCHES = 50
# The encoder's hidden width IS the embedding width the heads consume, so it
# reads from the single source next to the heads instead of a repeated 128.
HIDDEN_CHANNELS = EMBED_DIM

SMOKE = os.environ.get("OPENHEADS_SMOKE") == "1"
SMOKE_POS_CAP = 2_000
SMOKE_NEG_CAP = 20_000

FEATURES = os.environ.get("OPENHEADS_FEATURES", "")
CSR_DIR = os.environ.get("OPENHEADS_CSR_DIR", "")
POOL_DIR = os.environ.get("OPENHEADS_POOL_DIR", "")
WARM_CKPT = os.environ.get("OPENHEADS_WARM_CKPT", "")
OUT_DIR = Path(os.environ.get("OPENHEADS_OUT_DIR", "./runs/backbone"))
_STARTS_RAW = os.environ.get("OPENHEADS_CHAIN_STARTS", "")
CHAIN_STARTS = [int(v) for v in _STARTS_RAW.split(",")] if _STARTS_RAW else [0]
_TAIL = os.environ.get("OPENHEADS_TAIL_START", "")
TAIL_START = int(_TAIL) if _TAIL else None
TAIL_CHAIN = int(os.environ.get("OPENHEADS_TAIL_CHAIN", "0"))
N_NODES = int(os.environ.get("OPENHEADS_N_NODES", "0"))

# Optional integrity pins for the inputs: {"features": "<sha256>", ...}.
# Empty = not checked. Supply them and a swapped input fails the run instead
# of producing numbers that cannot be traced to the data they came from.
INPUT_PINS: dict[str, str] = json.loads(os.environ.get("OPENHEADS_INPUT_PINS", "{}"))

_MARKER_DIR = Path(os.environ.get("OPENHEADS_MARKER_DIR", tempfile.gettempdir()))
SUCCESS_MARKER = _MARKER_DIR / "openheads_backbone.SUCCESS"
FAIL_MARKER = _MARKER_DIR / "openheads_backbone.FAIL"


def _sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def verify_inputs(log) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Fail-closed sha gate on every input byte the model touches."""
    files = {
        "colptr": f"{CSR_DIR}/colptr.pt",
        "row": f"{CSR_DIR}/row.pt",
        "features": FEATURES,
        "nodes": f"{POOL_DIR}/nodes.npy",
        "y_fincrime": f"{POOL_DIR}/y_fincrime.npy",
        "y_aiagent": f"{POOL_DIR}/y_aiagent.npy",
        "warm_ckpt": WARM_CKPT,
    }
    observed, bad = {}, []
    for key, path in files.items():
        got = _sha256(path)
        observed[key] = got
        want = INPUT_PINS.get(key)
        if want is None:
            log.info("input sha %s=%s (unpinned)", key, got)
        elif got != want:
            bad.append(f"{key}: got {got}, want {want}")
            log.error("INPUT PIN MISMATCH %s", key)
        else:
            log.info("input pin OK %s", key)
    if bad:
        raise RuntimeError("INPUT INTEGRITY FAILED:\n" + "\n".join(bad))
    return observed


# ---------------------------------------------------------------------------
# MODEL — per-chain-LN encoder + the verbatim 2-head detector
# ---------------------------------------------------------------------------


def full_warm_load(encoder, ckpt_path, log):  # type: ignore[no-untyped-def]
    """Load a previous backbone in FULL, strictly.

    Every key and shape must match, and the `chain_starts` buffer recorded in
    the checkpoint must equal the one this encoder was constructed with: a
    checkpoint trained on a different chain partition would load cleanly and
    then normalise every chain with the wrong statistics. Tail routing lives
    in plain attributes, not in the state_dict, so it does not participate.
    """
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    own = encoder.state_dict()
    if set(state.keys()) != set(own.keys()):
        missing = sorted(set(own) - set(state))
        extra = sorted(set(state) - set(own))
        raise RuntimeError(
            f"warm load: key sets differ (missing {missing}, extra {extra})")
    for key, tensor in state.items():
        if tuple(tensor.shape) != tuple(own[key].shape):
            raise RuntimeError(
                f"warm load: {key} shape {tuple(tensor.shape)} != "
                f"{tuple(own[key].shape)}")
    ctor_starts = own["chain_starts"]
    if not torch.equal(state["chain_starts"], ctor_starts):
        raise RuntimeError(
            f"warm load: chain_starts buffer {state['chain_starts'].tolist()} "
            f"!= ctor {ctor_starts.tolist()} — wrong checkpoint or wrong merge")
    encoder.load_state_dict(state, strict=True)
    log.info("FULL warm load: %d tensors strict=True, chain_starts %s, "
             "tail routing tail_start=%s tail_chain=%s",
             len(state), ctor_starts.tolist(),
             encoder.tail_start, encoder.tail_chain)
    return encoder


def build_full_model(encoder, heads):  # type: ignore[no-untyped-def]
    """Encoder + heads as one module.

    The forward takes `n_id` (the batch's MERGED-graph node ids) because the
    encoder derives each row's chain from its global id, never from features.
    """
    import torch.nn as nn

    class FullModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = encoder       # PerChainNormEncoder (trainable)
            self.heads = heads            # TwoHeadDetector (.aiagent / .fincrime)

        def forward(self, x_feat, adj, batch_size, n_id):  # type: ignore[no-untyped-def]
            emb = self.backbone.get_embeddings(x_feat, adj, n_id)[:batch_size]
            return (self.heads.aiagent(emb).squeeze(-1),
                    self.heads.fincrime(emb).squeeze(-1))

    return FullModel()


def make_loader(graph, input_nodes_np, batch_size: int, fanout, shuffle: bool,
                *, drop_last: bool = False):  # type: ignore[no-untyped-def]
    """Neighbour-sampled batches over the graph.

    ``num_workers=0`` is mandatory, not a default: each worker fork
    materialises its own copy of the mmap'd CSR, so any positive number OOMs
    a graph of this size. Single-threaded sampling is the accepted bottleneck.

    Neighbour sampling needs a compiled backend (``pyg-lib`` or
    ``torch-sparse``). When neither is installed, this falls back to full-graph
    batches — enough to exercise wiring and shapes, NOT a substitute for
    sampling: without it a real graph will not fit in memory. The fallback
    says so out loud rather than looking like a successful sampled run.
    """
    import torch
    from torch_geometric.loader import NeighborLoader

    seeds = torch.from_numpy(input_nodes_np).to(torch.int64)
    if not _sampling_backend_available():
        # Loud, once per loader: a silent fallback is indistinguishable from a
        # successful sampled run in the log a reader trusts.
        print(
            "WARNING: neither pyg-lib nor torch-sparse is installed — falling "
            "back to FULL-GRAPH batches. Correct, but it holds the whole graph "
            "in memory, so it is for smoke runs only. Install "
            "'openheads[sampling]' for neighbour sampling.",
            flush=True,
        )
        return _FullGraphLoader(graph, seeds, batch_size, drop_last=drop_last)
    return NeighborLoader(
        graph,
        num_neighbors=fanout,
        input_nodes=seeds,
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def _sampling_backend_available() -> bool:
    """Is a compiled neighbour-sampling backend importable?

    Checked up front, not by catching an exception around construction:
    NeighborLoader builds fine without a backend and raises only on the first
    iteration, deep inside the sampler.
    """
    import importlib.util

    return any(importlib.util.find_spec(name) is not None
               for name in ("pyg_lib", "torch_sparse"))


class _FullGraphLoader:
    """Backend-free stand-in for NeighborLoader: whole graph, seeds first.

    Yields objects carrying the attributes the training loop reads (``x``,
    ``n_id``, ``batch_size``, ``edge_index``), with the seed rows placed first
    so ``n_id[:batch_size]`` are the seed ids.

    THE ADJACENCY IS RENUMBERED, and that is the whole difficulty here.
    Putting the seeds first permutes the feature rows, so a row's position in
    ``x`` no longer equals its graph id. Handing the original ``edge_index``
    to a layer that indexes into those permuted rows makes it aggregate over
    the WRONG neighbours: the loss stays finite, the checkpoint still writes,
    every gate still passes, and the model has learned from a scrambled
    graph. So each batch maps its edges through the inverse permutation.
    """

    def __init__(self, graph, seeds, batch_size: int, *, drop_last: bool = False) -> None:
        self.graph = graph
        self.seeds = seeds
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):  # type: ignore[no-untyped-def]
        import torch

        n_total = int(self.graph.num_nodes)
        all_ids = torch.arange(n_total, dtype=torch.int64)
        for start in range(0, len(self.seeds), self.batch_size):
            seed_ids = self.seeds[start:start + self.batch_size]
            if self.drop_last and len(seed_ids) < self.batch_size:
                return
            if len(torch.unique(seed_ids)) != len(seed_ids):
                raise RuntimeError(
                    "duplicate seed ids in one batch: the permutation below "
                    "would not be a bijection and rows would be mislabelled")
            is_seed = torch.zeros(n_total, dtype=torch.bool)
            is_seed[seed_ids] = True
            n_id = torch.cat([seed_ids, all_ids[~is_seed]])
            # position_of[graph_id] = row of that node in x / n_id
            position_of = torch.empty(n_total, dtype=torch.int64)
            position_of[n_id] = torch.arange(len(n_id), dtype=torch.int64)
            yield _FullGraphBatch(
                x=self.graph.x[n_id],
                n_id=n_id,
                batch_size=len(seed_ids),
                edge_index=position_of[self.graph.edge_index],
            )


class _FullGraphBatch:
    """Attribute bag mirroring the fields the loop reads off a sampled batch."""

    def __init__(self, x, n_id, batch_size: int, edge_index) -> None:  # type: ignore[no-untyped-def]
        self.x = x
        self.n_id = n_id
        self.batch_size = batch_size
        self.edge_index = edge_index
        self.adj_t = None


def _seed_global_ids(batch):  # type: ignore[no-untyped-def]
    return batch.n_id[: batch.batch_size].cpu().numpy()


def infer_embeddings(model, graph, node_idx_np, device, log, *,
                     fanout=FANOUT):  # type: ignore[no-untyped-def]
    """Embedding pass over seed nodes. `n_id` goes to the encoder because
    chain routing is by MERGED-graph id — local batch rows carry no chain."""
    import numpy as np
    import torch

    from openheads.graph_batching import batch_adj

    out = np.empty((len(node_idx_np), EMBED_DIM), dtype=np.float16)
    loader = make_loader(graph, node_idx_np, INFER_BATCH, fanout, shuffle=False)
    model.eval()
    cursor = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            bs = batch.batch_size
            adj = batch_adj(batch)
            adj = adj.to(device) if hasattr(adj, "to") else adj
            emb = model.backbone.get_embeddings(
                batch.x.to(device), adj, batch.n_id.to(device))[:bs]
            out[cursor: cursor + bs] = emb.half().cpu().numpy()
            cursor += bs
            if bi % 50 == 0:
                log.info("infer %d/%d seeds (%.1fs)", cursor, len(node_idx_np),
                         time.time() - t0)
    if cursor != len(node_idx_np):
        raise RuntimeError(
            f"infer_embeddings emitted {cursor} != {len(node_idx_np)}")
    return out


def train_full(model, graph, needed, y_fincrime, y_aiagent,
               fit_node_idx, val_fpos_node, val_fneg_node,
               device, epochs, log, *, label="backbone"):  # type: ignore[no-untyped-def]
    """Train encoder and heads jointly over a neighbour-sampled loader.

    The forward receives `batch.n_id`, the batch's MERGED-graph node ids: the
    encoder routes each row to its chain by global id, so passing only local
    rows would normalise every row as if it belonged to the first chain.

    Returns the val-recall trajectory as well as the model, because a single
    end-of-run number hides an early-stopped degradation.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from openheads.graph_batching import batch_adj
    from openheads.heads import ALPHA_AIAGENT, BETA_FINCRIME, log_json, recall_at_fpr

    torch.manual_seed(42)

    rows_fit = np.searchsorted(needed, fit_node_idx)
    yf_fit = y_fincrime[rows_fit]
    ya_fit = y_aiagent[rows_fit]

    def _pw(y):  # type: ignore[no-untyped-def]
        pos = float(y.sum())
        neg = float(len(y) - pos)
        return torch.tensor(max(neg / max(pos, 1.0), 1.0), device=device)

    bce_a = nn.BCEWithLogitsLoss(pos_weight=_pw(ya_fit))
    bce_f = nn.BCEWithLogitsLoss(pos_weight=_pw(yf_fit))
    log.info("[%s] plain pos-weighted BCE loss (no focal reweighting)", label)

    opt = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": list(model.heads.aiagent.parameters())
                   + list(model.heads.fincrime.parameters()), "lr": LR_HEADS},
    ])

    model = model.to(device)
    prev_val_recall = -1.0
    epochs_run = 0
    val_trajectory: list[float] = []

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(42 + epoch)
        epoch_seeds = fit_node_idx[rng.permutation(len(fit_node_idx))]
        loader = make_loader(graph, epoch_seeds, BATCH_SIZE,
                             FANOUT, shuffle=False, drop_last=True)
        est_batches = len(epoch_seeds) // BATCH_SIZE
        first_losses: list[float] = []
        recent_losses: list[float] = []
        backbone_grad_seen = False
        t0 = time.time()
        n_batches = 0
        for batch in loader:
            bs = batch.batch_size
            seed_global = _seed_global_ids(batch)
            rows = np.searchsorted(needed, seed_global)
            if not np.array_equal(needed[rows], seed_global):
                raise RuntimeError(
                    f"[{label}] epoch {epochs_run + 1}: a NeighborLoader seed "
                    f"is absent from `needed` — searchsorted would mislabel. HALT.")
            yf_b = torch.from_numpy(y_fincrime[rows].astype("float32")).to(device)
            ya_b = torch.from_numpy(y_aiagent[rows].astype("float32")).to(device)
            adj = batch_adj(batch)
            adj = adj.to(device) if hasattr(adj, "to") else adj
            la, lf = model(batch.x.to(device), adj, bs, batch.n_id.to(device))
            loss = ALPHA_AIAGENT * bce_a(la, ya_b) + BETA_FINCRIME * bce_f(lf, yf_b)
            opt.zero_grad()
            loss.backward()
            if not backbone_grad_seen:
                g = sum(float(p.grad.abs().sum())
                        for p in model.backbone.parameters() if p.grad is not None)
                backbone_grad_seen = g > 0.0
            opt.step()
            n_batches += 1
            lv = float(loss.item())
            if len(first_losses) < 5:
                first_losses.append(lv)
            recent_losses.append(lv)
            if n_batches % HEARTBEAT_EVERY_N_BATCHES == 0:
                log.info("[%s] epoch %d batch %d/%d loss=%.4f avg50=%.4f (%.0fs)",
                         label, epoch + 1, n_batches, est_batches, lv,
                         sum(recent_losses[-50:]) / len(recent_losses[-50:]),
                         time.time() - t0)
        epochs_run = epoch + 1
        if not backbone_grad_seen:
            raise RuntimeError(
                f"[{label}] epoch {epochs_run}: NO gradient reached the backbone")

        emb_vp = infer_embeddings(model, graph, val_fpos_node, device, log)
        emb_vn = infer_embeddings(model, graph, val_fneg_node, device, log)
        with torch.no_grad():
            sp = torch.sigmoid(model.heads.fincrime(
                torch.from_numpy(emb_vp.astype("float32")).to(device)
            ).squeeze(-1)).cpu().numpy()
            sn = torch.sigmoid(model.heads.fincrime(
                torch.from_numpy(emb_vn.astype("float32")).to(device)
            ).squeeze(-1)).cpu().numpy()
            y_val = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))]).astype("int8")
            s_val = np.concatenate([sp, sn])
        val_recall = recall_at_fpr(y_val, s_val, 0.001)
        val_trajectory.append(round(float(val_recall), 4))

        last_mean_loss = (sum(recent_losses) / len(recent_losses)
                          if recent_losses else float("nan"))
        log_json(log, "epoch_done", label=label, epoch=epochs_run,
                 lr_backbone=LR_BACKBONE, lr_heads=LR_HEADS,
                 first_losses=first_losses, n_batches=n_batches,
                 epoch_s=round(time.time() - t0, 1),
                 val_recall_fpr001=round(float(val_recall), 4))
        if prev_val_recall >= 0 and val_recall < prev_val_recall - PLATEAU_DEGRADE_MARGIN:
            log.info("[%s] val recall degraded %.4f→%.4f at epoch %d — early-stop",
                     label, prev_val_recall, val_recall, epochs_run)
            break
        prev_val_recall = val_recall

    return model, epochs_run, val_trajectory, last_mean_loss


# ---------------------------------------------------------------------------
# DATA — the labelled node set and its stratified split
# ---------------------------------------------------------------------------


def load_pool_labels(log):  # type: ignore[no-untyped-def]
    """nodes.npy + y-vectors, re-sorted to the sorted `needed` order that
    searchsorted requires. Dedup asserted (the pool is unique by build)."""
    import numpy as np

    nodes = np.load(f"{POOL_DIR}/nodes.npy", allow_pickle=False)
    y_f = np.load(f"{POOL_DIR}/y_fincrime.npy", allow_pickle=False)
    y_a = np.load(f"{POOL_DIR}/y_aiagent.npy", allow_pickle=False)
    if not (len(nodes) == len(y_f) == len(y_a)):
        raise RuntimeError("pool file length mismatch")
    if len(np.unique(nodes)) != len(nodes):
        raise RuntimeError("pool nodes are not unique")
    order = np.argsort(nodes)
    needed = nodes[order].astype(np.int64)
    y_f = y_f[order].astype(np.int8)
    y_a = y_a[order].astype(np.int8)
    log.info("pool labels: %d nodes, %d fincrime pos, %d aiagent pos",
             len(needed), int(y_f.sum()), int(y_a.sum()))
    return needed, y_f, y_a


def stratified_split(needed, y_fincrime, y_aiagent, log):  # type: ignore[no-untyped-def]
    """Seed-42 stratified test/val/fit. The seed and the draw order are part
    of the recipe: change either and the split — and every number reported
    against it — stops being comparable with earlier runs."""
    import numpy as np

    from openheads.heads import RANDOM_SEED, TEST_FRACTION

    n_rows = len(needed)
    grp = np.zeros(n_rows, dtype=np.int8)
    grp[y_fincrime == 1] = 1
    grp[(y_aiagent == 1) & (y_fincrime == 0)] = 2
    rng = np.random.default_rng(RANDOM_SEED)
    is_test = np.zeros(n_rows, dtype=bool)
    for g in (0, 1, 2):
        rows_g = np.where(grp == g)[0]
        if len(rows_g) == 0:
            continue
        perm = rng.permutation(len(rows_g))
        n_test = int(len(rows_g) * TEST_FRACTION)
        is_test[rows_g[perm[:n_test]]] = True
    test_rows = np.where(is_test)[0]
    train_pool_rows = np.where(~is_test)[0]
    rng_val = np.random.default_rng(RANDOM_SEED + 7)
    is_val = np.zeros(n_rows, dtype=bool)
    for g in (0, 1, 2):
        rows_g = train_pool_rows[grp[train_pool_rows] == g]
        if len(rows_g) == 0:
            continue
        perm = rng_val.permutation(len(rows_g))
        n_v = int(len(rows_g) * TEST_FRACTION)
        is_val[rows_g[perm[:n_v]]] = True
    val_rows = np.where(is_val)[0]
    fit_rows = train_pool_rows[~is_val[train_pool_rows]]
    log.info("split: fit %d / val %d / test %d rows",
             len(fit_rows), len(val_rows), len(test_rows))
    return grp, fit_rows, val_rows, test_rows


def _smoke_subset(rows, y_fincrime, y_aiagent, rng, log, tag):  # type: ignore[no-untyped-def]
    """Cap a row set for the subset smoke, so every code path still runs."""
    import numpy as np

    pos = rows[(y_fincrime[rows] == 1) | (y_aiagent[rows] == 1)]
    neg = rows[(y_fincrime[rows] == 0) & (y_aiagent[rows] == 0)]
    pos = pos[rng.permutation(len(pos))[:SMOKE_POS_CAP]]
    neg = neg[rng.permutation(len(neg))[:SMOKE_NEG_CAP]]
    out = np.sort(np.concatenate([pos, neg]))
    log.info("SMOKE %s subset: %d rows (%d pos-ish, %d neg)",
             tag, len(out), len(pos), len(neg))
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def run(log) -> int:  # type: ignore[no-untyped-def]
    import numpy as np
    import torch
    from torch_geometric.data import Data

    from openheads.heads import build_model, recall_at_fpr
    from openheads.models import PerChainNormEncoder

    # torch-sparse is an OPTIONAL extra (`pip install openheads[sampling]`):
    # it needs a compiled wheel matched to your torch build, which a bare
    # `pip install openheads` cannot guarantee. Imported here, inside the
    # real-data path, so the synthetic smoke and every test run without it —
    # an unconditional import at module scope made the documented install
    # fail on the flagship entry point.
    try:
        from torch_sparse import SparseTensor
    except ImportError as exc:
        raise SystemExit(
            "the real-data path needs torch-sparse for the CSR graph: "
            "pip install 'openheads[sampling]' (a wheel matching your torch "
            "build is required). The synthetic smoke does not need it: "
            "python scripts/train_backbone.py --synthetic"
        ) from exc

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not SMOKE:
        raise RuntimeError(
            "CUDA not available — a full pass on CPU takes days and its "
            "numbers would not match a GPU recipe run; use --synthetic or "
            "OPENHEADS_SMOKE=1 for CPU-sized checks")
    log.info("=== backbone retrain begin (device=%s smoke=%s tail_start=%s) ===",
             device, SMOKE, TAIL_START)

    pins = verify_inputs(log)

    needed, y_f, y_a = load_pool_labels(log)
    grp, fit_rows, val_rows, test_rows = stratified_split(needed, y_f, y_a, log)

    rng = np.random.default_rng(7)
    if SMOKE:
        fit_rows = _smoke_subset(fit_rows, y_f, y_a, rng, log, "fit")
        val_rows = _smoke_subset(val_rows, y_f, y_a, rng, log, "val")
        test_rows = _smoke_subset(test_rows, y_f, y_a, rng, log, "test")

    fit_node_idx = needed[fit_rows]
    val_fpos_node = needed[val_rows[y_f[val_rows] == 1]]
    val_fneg_node = needed[val_rows[grp[val_rows] == 0]]

    # mmap the features and the CSR: a graph of this size does not fit in RAM
    # as a dense read, and a copy per worker would OOM the box.
    x = torch.load(FEATURES, map_location="cpu", weights_only=True, mmap=True)
    colptr = torch.load(f"{CSR_DIR}/colptr.pt", map_location="cpu",
                        weights_only=True, mmap=True)
    row = torch.load(f"{CSR_DIR}/row.pt", map_location="cpu",
                     weights_only=True, mmap=True)
    n = int(x.shape[0])
    if n < int(needed.max()) + 1:
        raise RuntimeError(f"graph nodes {n} < max needed idx {int(needed.max())}")
    if N_NODES and n != N_NODES:
        raise RuntimeError(
            f"graph nodes {n} != declared OPENHEADS_N_NODES {N_NODES} — wrong inputs")
    graph = Data(x=x, adj_t=SparseTensor(rowptr=colptr, col=row,
                                         sparse_sizes=(n, n), is_sorted=True))
    graph.num_nodes = n
    log.info("graph %d nodes, %d edges, %d channels", n, row.numel(), x.shape[1])

    # Encoder: full strict warm-start, appended tail routed to its own chain.
    encoder = PerChainNormEncoder(in_channels=int(x.shape[1]),
                                  hidden_channels=HIDDEN_CHANNELS,
                                  chain_starts=CHAIN_STARTS,
                                  tail_start=TAIL_START, tail_chain=TAIL_CHAIN)
    encoder = full_warm_load(encoder, WARM_CKPT, log)
    log.info("encoder FULL warm-start from %s; heads fresh", WARM_CKPT)
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    heads = build_model()

    model = build_full_model(encoder, heads)
    epochs = 1 if SMOKE else MAX_EPOCHS
    model, epochs_run, val_traj, last_loss = train_full(
        model, graph, needed, y_f, y_a,
        fit_node_idx, val_fpos_node, val_fneg_node, device, epochs, log,
        label="backbone-warm")

    # Leak-free check: recall@FPR=1e-3 on the frozen TEST split (backbone never
    # trained on these rows) with the in-loop heads.
    test_fpos_node = needed[test_rows[y_f[test_rows] == 1]]
    test_fneg_node = needed[test_rows[grp[test_rows] == 0]]
    emb_tp = infer_embeddings(model, graph, test_fpos_node, device, log)
    emb_tn = infer_embeddings(model, graph, test_fneg_node, device, log)
    with torch.no_grad():
        sp = torch.sigmoid(model.heads.fincrime(
            torch.from_numpy(emb_tp.astype("float32")).to(device)).squeeze(-1)).cpu().numpy()
        sn = torch.sigmoid(model.heads.fincrime(
            torch.from_numpy(emb_tn.astype("float32")).to(device)).squeeze(-1)).cpu().numpy()
    y_test = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))]).astype(np.int8)
    s_test = np.concatenate([sp, sn])
    test_recall = float(recall_at_fpr(y_test, s_test, 0.001))
    log.info("LEAK-FREE test-split fincrime recall@FPR=1e-3 = %.4f "
             "(%d pos / %d neg)", test_recall, len(sp), len(sn))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc_path = OUT_DIR / "backbone.pt"
    heads_path = OUT_DIR / "heads.pt"
    torch.save(model.backbone.state_dict(), enc_path)
    torch.save(model.heads.state_dict(), heads_path)

    # Each head is ALSO written on its own, with bare keys. heads.pt holds the
    # two-head module, so its keys are prefixed ("fincrime.0.weight") and the
    # calibration step — which loads a plain nn.Sequential with strict=True —
    # cannot read it. Without these two files the next stage in this repository
    # could not consume this stage's output without an undocumented rename and
    # a manual key rewrite.
    chain_tag = os.environ.get("OPENHEADS_HEAD_TAG", "all")
    for name, module in (("fincrime", model.heads.fincrime),
                        ("ai_agent", model.heads.aiagent)):
        path = OUT_DIR / f"{name}_{chain_tag}.pt"
        torch.save(module.state_dict(), path)
        log.info("head written for calibration: %s", path)

    # Reload gate: the saved encoder must load strict into a fresh instance
    # built with the SAME tail routing.
    probe = PerChainNormEncoder(in_channels=int(x.shape[1]),
                                hidden_channels=HIDDEN_CHANNELS,
                                chain_starts=CHAIN_STARTS,
                                tail_start=TAIL_START, tail_chain=TAIL_CHAIN)
    probe.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=True), strict=True)
    log.info("checkpoint reload gate PASS (strict=True)")

    meta = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": "supervised, per-chain normalisation, tail routing, warm start",
        "smoke": SMOKE,
        "recipe": {"lr_backbone": LR_BACKBONE, "lr_heads": LR_HEADS,
                   "batch_size": BATCH_SIZE, "fanout": FANOUT,
                   "max_epochs": MAX_EPOCHS, "epochs_run": epochs_run},
        "warm_start": {"mode": "full-strict", "ckpt": str(WARM_CKPT),
                       "sha256": pins["warm_ckpt"]},
        "input_pins": pins,
        "chain_starts": CHAIN_STARTS,
        "tail_routing": {"tail_start": encoder.tail_start,
                         "tail_chain": encoder.tail_chain,
                         "meaning": "appended tail rows -> that chain's norm"},
        "val_recall_trajectory_fpr001": val_traj,
        "test_split_leakfree_recall_fpr001": round(test_recall, 4),
        "note": ("per-chain evaluation numbers are produced by the emit and "
                 "eval stages; the backbone here saw the FIT split only."),
        "artefacts": {"encoder": str(enc_path), "heads": str(heads_path),
                      "encoder_sha256": _sha256(enc_path),
                      "heads_sha256": _sha256(heads_path)},
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("=== backbone retrain done in %.0fs; epochs=%d val_traj=%s test=%.4f ===",
             time.time() - started, epochs_run, val_traj, test_recall)
    SUCCESS_MARKER.write_text("backbone_trained\n")
    return 0


def run_synthetic(log) -> int:  # type: ignore[no-untyped-def]
    """Laptop wiring smoke: tiny random graph, no staged inputs, no pins.
    Exercises: encoder warm-ish init, n_id forward, train_full loop, infer,
    save/reload gate. Asserts backbone receives gradient and loss is finite."""
    import numpy as np
    import torch
    from torch_geometric.data import Data

    from openheads.heads import build_model
    from openheads.models import PerChainNormEncoder

    global BATCH_SIZE, INFER_BATCH

    n, ch = 5_000, 22
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    x = torch.randn(n, ch)
    # edge_index graph: batch_adj falls back to it when adj_t is absent, and
    # NeighborLoader samples it without the pyg-lib/torch-sparse backends —
    # exactly what a laptop has. The box path uses the CSR SparseTensor.
    graph = Data(x=x, edge_index=torch.randint(0, n, (2, n * 8)))
    graph.num_nodes = n

    starts = [0, n // 2]
    encoder = PerChainNormEncoder(in_channels=ch, hidden_channels=HIDDEN_CHANNELS,
                                  chain_starts=starts)
    tmp = Path(tempfile.mkdtemp(prefix="openheads_syn_"))
    warm = tmp / "warm.pt"
    torch.save(encoder.state_dict(), warm)   # self-warm: exercises strict load

    # --- tail-routing gate, run before any paid GPU time: (1) chain_ids_of
    # must route tail ids to tail_chain and leave every pre-tail id alone;
    # (2) an encoder warm-loaded FULL from a no-tail checkpoint must be
    # BIT-IDENTICAL on pre-tail rows (the state_dict carries no routing).
    enc_nt = PerChainNormEncoder(in_channels=ch, hidden_channels=HIDDEN_CHANNELS,
                                 chain_starts=[0, n // 2])
    w_nt = tmp / "w_nt.pt"
    torch.save(enc_nt.state_dict(), w_nt)
    enc_t = PerChainNormEncoder(in_channels=ch, hidden_channels=HIDDEN_CHANNELS,
                                chain_starts=[0, n // 2],
                                tail_start=n, tail_chain=0)
    enc_t = full_warm_load(enc_t, w_nt, log)
    ids = torch.tensor([0, n // 2 - 1, n // 2, n - 1, n, n + 7, 10 * n])
    want_nt = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    want_t = torch.tensor([0, 0, 1, 1, 0, 0, 0])
    got_nt = enc_nt.chain_ids_of(ids)
    got_t = enc_t.chain_ids_of(ids)
    if not (torch.equal(got_nt, want_nt) and torch.equal(got_t, want_t)):
        raise RuntimeError(
            f"tail-routing gate FAILED: no-tail {got_nt.tolist()} want "
            f"{want_nt.tolist()}; tail {got_t.tolist()} want {want_t.tolist()}")
    enc_nt.eval()
    enc_t.eval()
    with torch.no_grad():
        xg = torch.randn(64, ch)
        ei = torch.randint(0, 64, (2, 256))
        nid_pre = torch.randint(0, n, (64,))     # every row pre-tail
        e_nt = enc_nt.get_embeddings(xg, ei, nid_pre)
        e_t = enc_t.get_embeddings(xg, ei, nid_pre)
    if not torch.equal(e_nt, e_t):
        raise RuntimeError("tail routing changed PRE-TAIL embeddings — "
                           "construction broken")
    log.info("tail-routing gate PASS: ids routed %s, pre-tail rows "
             "bit-identical after full warm load", got_t.tolist())

    needed = np.arange(n, dtype=np.int64)
    y_f = (rng.random(n) < 0.02).astype(np.int8)
    y_a = (rng.random(n) < 0.01).astype(np.int8)

    grp, fit_rows, val_rows, test_rows = stratified_split(needed, y_f, y_a, log)

    enc2 = PerChainNormEncoder(in_channels=ch, hidden_channels=HIDDEN_CHANNELS,
                               chain_starts=starts)
    enc2.load_state_dict(torch.load(warm, map_location="cpu", weights_only=True), strict=True)
    model = build_full_model(enc2, build_model())

    BATCH_SIZE = 256
    INFER_BATCH = 512
    model, epochs_run, val_traj, last_loss = train_full(
        model, graph, needed, y_f, y_a,
        needed[fit_rows][:2048],
        needed[val_rows[y_f[val_rows] == 1]],
        needed[val_rows[grp[val_rows] == 0]][:2000],
        "cpu", 1, log, label="synthetic")

    # Gate 1: training actually ran and did not diverge. A NaN loss trains
    # happily to the end of the loop and leaves a checkpoint behind, so the
    # smoke asserts finiteness rather than assuming it.
    if not math.isfinite(last_loss):
        raise RuntimeError(f"loss is not finite: {last_loss}")
    log.info("loss finite (last epoch mean=%.6f)", last_loss)

    # Gate 2: the checkpoint round-trips. Writing a file proves nothing —
    # a checkpoint that cannot be read back strictly, with tensors equal to
    # what was saved, is not a checkpoint.
    enc_path = tmp / "backbone.pt"
    torch.save(model.backbone.state_dict(), enc_path)
    if not enc_path.is_file() or enc_path.stat().st_size == 0:
        raise RuntimeError(f"checkpoint was not written: {enc_path}")
    probe = PerChainNormEncoder(in_channels=ch, hidden_channels=HIDDEN_CHANNELS,
                                chain_starts=starts)
    reloaded = torch.load(enc_path, map_location="cpu", weights_only=True)
    probe.load_state_dict(reloaded, strict=True)
    saved = model.backbone.state_dict()
    for key, tensor in saved.items():
        if not torch.equal(tensor.cpu(), probe.state_dict()[key].cpu()):
            raise RuntimeError(f"checkpoint round-trip changed tensor {key}")
    log.info("checkpoint written/reloaded (%d tensors, %d bytes, strict=True)",
             len(saved), enc_path.stat().st_size)

    log.info("SYNTHETIC SMOKE PASS: epochs=%d val_traj=%s", epochs_run, val_traj)
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="laptop wiring smoke on a tiny random graph")
    args = ap.parse_args()

    try:
        from openheads.heads import setup_logging
        log = setup_logging()
    except (PermissionError, OSError):
        # Read-only or unwritable log dir: fall back to stdout only.
        import logging
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)sZ openheads %(levelname)s %(message)s",
                            force=True)
        log = logging.getLogger("openheads")
    SUCCESS_MARKER.unlink(missing_ok=True)
    FAIL_MARKER.unlink(missing_ok=True)
    try:
        if args.synthetic:
            return run_synthetic(log)
        return run(log)
    except Exception as e:  # noqa: BLE001
        log.exception("RUN FAILED: %s", e)
        FAIL_MARKER.write_text(f"{type(e).__name__}: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
