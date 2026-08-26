#!/usr/bin/env python3
"""Per-chain scoring head trained on POOLED embeddings.

Same protocol as the full-population variant (`export_perchain_head.py`), fed
a sampled pool instead of a whole chain. What is carried over verbatim:
  - head architecture (128-256-128-1), focal loss (gamma=2) over
    pos_weight-balanced BCE, Adam lr 1e-3, 8 epochs, batch 8192;
  - seed 42 with the positive 80/20 permutation as the FIRST rng draw, so the
    same entities land in the held-out set as in the full-population run;
  - negatives split into sorted halves: train half, eval half;
  - held-out recall reported against BOTH tau conventions, strict `>`.

The one real adaptation, and why it is not a shortcut: tau cannot be a
quantile over all chain scores, because the full chain was not scored. A
quantile over the negative sample alone would UNDERSTATE tau, because the
true population also contains the positive mass — and understated tau reads
back as optimistic recall. The estimator here is a weighted mixture: all
positives at weight 1, plus the UNTRAINED eval half of the negatives at
weight (N_chain - P)/len(neg_eval); tau is the smallest score whose weighted
count above it stays within fpr * N_chain. The train half is excluded on
purpose: the model saw those rows as 0-labels, their scores are biased low,
and including them would drag tau down.

Env:
  OPENHEADS_CHAIN           chain name (required)
  OPENHEADS_POOL_EMB        pooled embeddings .npy (required)
  OPENHEADS_POOL_NODES      pooled node ids .npy (required)
  OPENHEADS_POOL_LABELS     pooled labels .npy (required)
  OPENHEADS_CHAIN_STARTS    comma-separated node-id offsets, one per chain
  OPENHEADS_N_TOTAL         total node count of the merged graph
  OPENHEADS_CHAINS          comma-separated chain names, same order as starts
  OPENHEADS_OUT_DIR         output directory (default ./runs/heads_<chain>)
  OPENHEADS_SMOKE=1         caps negatives to 20K and epochs to 1, so every
                            code path runs before committing to a full pass
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

CHAIN = os.environ.get("OPENHEADS_CHAIN", "")
EMB_POOL = os.environ.get("OPENHEADS_POOL_EMB", "")
POOL_NODES = os.environ.get("OPENHEADS_POOL_NODES", "")
POOL_LABELS = os.environ.get("OPENHEADS_POOL_LABELS", "")
_STARTS_RAW = os.environ.get("OPENHEADS_CHAIN_STARTS", "")
STARTS = [int(v) for v in _STARTS_RAW.split(",")] if _STARTS_RAW else []
N_TOTAL = int(os.environ.get("OPENHEADS_N_TOTAL", "0"))
OUT_DIR = os.environ.get("OPENHEADS_OUT_DIR", f"./runs/heads_{CHAIN or 'chain'}")
SMOKE = os.environ.get("OPENHEADS_SMOKE") == "1"

# Chain name -> its index in CHAIN_STARTS. Derived from the declared chain
# order, so adding a chain needs no code change.
CHAINS = os.environ.get("OPENHEADS_CHAINS", "").split(",") if os.environ.get(
    "OPENHEADS_CHAINS") else []
CHAIN_RANGE = {name: i for i, name in enumerate(CHAINS)}
REQUIRED_ENV = (
    "OPENHEADS_CHAIN",
    "OPENHEADS_POOL_EMB",
    "OPENHEADS_POOL_NODES",
    "OPENHEADS_POOL_LABELS",
    "OPENHEADS_CHAIN_STARTS",
    "OPENHEADS_N_TOTAL",
    "OPENHEADS_CHAINS",
)
SEED = 42
FPRS = (0.01, 0.001, 0.0001)
EPOCHS = 1 if SMOKE else 8
BATCH = 8192
SMOKE_NEG = 20_000


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {m}", flush=True)


def build_head() -> nn.Module:
    return nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 1),
    )


def tau_population_est(s_pos: np.ndarray, s_neg_ev: np.ndarray,
                       n_chain: int, fpr: float) -> tuple[float, float]:
    """Weighted-mixture estimate of the population (1-fpr) quantile.

    Population = P positives (all present, weight 1) + (N-P) non-positives
    represented by the untrained uniform eval half (weight (N-P)/len(neg_ev)).
    Returns (tau, weighted_flagged_estimate)."""
    p = len(s_pos)
    w_neg = float(n_chain - p) / float(len(s_neg_ev))
    vals = np.concatenate([s_pos, s_neg_ev])
    wts = np.concatenate([np.ones(p), np.full(len(s_neg_ev), w_neg)])
    order = np.argsort(-vals, kind="stable")
    vals, wts = vals[order], wts[order]
    csum = np.cumsum(wts)
    budget = fpr * n_chain
    k = int(np.searchsorted(csum, budget, side="right"))
    if k >= len(vals):
        k = len(vals) - 1
    tau = float(vals[k])
    flagged = float(csum[k - 1]) if k > 0 else 0.0
    return tau, flagged


def main() -> int:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit("missing required environment: " + ", ".join(missing))
    if CHAIN not in CHAIN_RANGE:
        raise SystemExit(f"{CHAIN!r} is not in OPENHEADS_CHAINS={CHAINS}")
    t0 = time.time()
    ci = CHAIN_RANGE[CHAIN]
    lo = STARTS[ci]
    hi = STARTS[ci + 1] if ci + 1 < len(STARTS) else N_TOTAL
    n_chain = hi - lo

    nodes = np.load(POOL_NODES, allow_pickle=False)
    y_f = np.load(POOL_LABELS, allow_pickle=False)
    emb = np.load(EMB_POOL, mmap_mode="r", allow_pickle=False)
    if emb.shape[0] != len(nodes):
        raise SystemExit(f"emb rows {emb.shape[0]} != nodes {len(nodes)}")

    in_chain = np.where((nodes >= lo) & (nodes < hi))[0]
    # Sorted by node id so the seeded 80/20 permutation picks the same held-out
    # ENTITIES as the full-population run — otherwise the two numbers are not
    # comparable even with the same seed.
    in_chain = in_chain[np.argsort(nodes[in_chain], kind="stable")]
    pos_rows = in_chain[y_f[in_chain] == 1]
    neg_rows = in_chain[y_f[in_chain] == 0]
    log(f"{CHAIN}: chain nodes {n_chain:,}; pool rows {len(in_chain):,} "
        f"(pos {len(pos_rows):,} neg {len(neg_rows):,})")
    assert len(pos_rows) >= 100

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pos_rows))          # FIRST rng draw, as upstream
    k = max(1, int(len(pos_rows) * 0.2))
    held, train_pos = pos_rows[perm[:k]], pos_rows[perm[k:]]

    if SMOKE:
        neg_rows = neg_rows[:SMOKE_NEG]
        log(f"SMOKE: negatives capped to {len(neg_rows):,}, 1 epoch")
    neg_tr = neg_rows[: len(neg_rows) // 2]
    neg_ev = neg_rows[len(neg_rows) // 2:]
    log(f"negatives: train {len(neg_tr):,} eval {len(neg_ev):,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    head = build_head().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    rows = np.concatenate([train_pos, neg_tr])
    y = np.concatenate([np.ones(len(train_pos), dtype=np.float32),
                        np.zeros(len(neg_tr), dtype=np.float32)])
    pos_w = torch.tensor(len(neg_tr) / max(len(train_pos), 1),
                         device=dev, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_w)
    for ep in range(EPOCHS):
        idx = rng.permutation(len(rows))
        tot = 0.0
        head.train()
        for b0 in range(0, len(idx), BATCH):
            sel = idx[b0:b0 + BATCH]
            xb = torch.from_numpy(
                np.ascontiguousarray(emb[rows[sel]])).float().to(dev)
            yb = torch.from_numpy(y[sel]).to(dev)
            logit = head(xb).squeeze(-1)
            p = torch.sigmoid(logit)
            pt = torch.where(yb > 0.5, p, 1 - p)
            loss = ((1 - pt).clamp_min(1e-6).pow(2.0) *
                    bce(logit, yb)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        log(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(idx):.5f}")

    head.eval()

    def score_rows(rr: np.ndarray) -> np.ndarray:
        out = np.empty(len(rr), dtype=np.float32)
        with torch.no_grad():
            for b0 in range(0, len(rr), 262_144):
                xb = torch.from_numpy(np.ascontiguousarray(
                    emb[rr[b0:b0 + 262_144]])).float().to(dev)
                out[b0:b0 + 262_144] = torch.sigmoid(
                    head(xb).squeeze(-1)).cpu().numpy()
        return out

    s_pos_all = score_rows(pos_rows)     # in pos_rows (sorted) order
    s_held = score_rows(held)
    s_neg_ev = np.sort(score_rows(neg_ev))
    log(f"scored pos {len(s_pos_all):,} / held {len(s_held):,} / "
        f"neg_ev {len(s_neg_ev):,} ({time.time()-t0:.0f}s)")

    res = {"chain": CHAIN, "n_nodes": int(n_chain), "labels": int(len(pos_rows)),
           "train_pos": int(len(train_pos)), "held": int(len(held)),
           "head": "MLP 128-256-128-1, focal g=2, pos_weight balanced",
           "input": "pooled embeddings (sampled rows only, not the full chain)",
           "tau_population_estimator":
               "weighted mixture: all positives (w=1) + untrained eval-half "
               "negatives (w=(N-P)/n_ev); train-half excluded (scores biased "
               "low by training)",
           "smoke": SMOKE,
           "fpr_points": {}}
    for fpr in FPRS:
        tau_pop, flagged = tau_population_est(s_pos_all, s_neg_ev, n_chain, fpr)
        kk = max(0, min(len(s_neg_ev) - 1, int(len(s_neg_ev) * (1 - fpr))))
        tau_neg = float(s_neg_ev[kk])
        res["fpr_points"][str(fpr)] = {
            "tau_population": tau_pop,
            "tau_negative_sample": tau_neg,
            "held_recall_vs_population_tau": float((s_held > tau_pop).mean()),
            "held_recall_vs_negative_tau": float((s_held > tau_neg).mean()),
            "flagged_at_population_tau_est": flagged,
        }
        log(f"  fpr={fpr}: held_recall(pop tau est)="
            f"{(s_held > tau_pop).mean():.4f} "
            f"(neg tau)={(s_held > tau_neg).mean():.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/perchain_head_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    # Name and key layout are the contract tau_scan.py loads with
    # strict=True: bare nn.Sequential keys in fincrime_<chain>.pt.
    torch.save(head.state_dict(), f"{OUT_DIR}/fincrime_{CHAIN}.pt")
    log(f"PERCHAIN_POOL_HEAD_DONE {CHAIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
