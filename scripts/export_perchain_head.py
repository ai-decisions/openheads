#!/usr/bin/env python3
"""Per-chain scoring head: train, score the full population, export top-K.

One chain at a time (env OPENHEADS_CHAIN):
  1. reuse transfer embeddings already computed for that chain
     (<work dir>/<chain>/<chain>_embeddings_fp16.pt);
  2. train the fincrime head architecture used elsewhere in this repo
     (128->256->128->1, alpha-balanced focal loss, gamma=2) on the label-set
     rows of this chain, 80/20 seeded split;
  3. score the FULL chain population and take tau as the population quantile
     at the target FPR (0.1% of THIS chain's nodes above tau);
  4. export the top-K flagged nodes with their addresses.

Held-out recall is reported against BOTH tau conventions (population
quantile and negative-sample quantile) so the reported number is never
silently the easier of the two.

Env:
  OPENHEADS_CHAIN     chain name (required)
  OPENHEADS_WORK_DIR  directory holding the chain's inputs (default ./runs)
  OPENHEADS_OUT_URI   optional object-storage prefix to upload results to;
                      unset = local output only, nothing is uploaded
  OPENHEADS_REGION    region for the upload, required only with OUT_URI
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

CHAIN = os.environ.get("OPENHEADS_CHAIN", "")
D = os.path.join(os.environ.get("OPENHEADS_WORK_DIR", "./runs"), CHAIN)
OUT_URI = os.environ.get("OPENHEADS_OUT_URI", "")
REGION = os.environ.get("OPENHEADS_REGION", "")
SEED = 42
TOPK = 10_000
FPRS = (0.01, 0.001, 0.0001)
EPOCHS = 8
BATCH = 8192


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {m}", flush=True)


def run(argv: list[str]) -> None:
    """Run a command as an argv list — never through a shell.

    The upload path interpolates a chain name and file names; built as a
    string for a shell, that is a command-injection hole in a published
    script.
    """
    subprocess.run(argv, check=True)


def build_head() -> nn.Module:
    return nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 1),
    )


def main() -> int:
    if not CHAIN:
        raise SystemExit("OPENHEADS_CHAIN is required (see the module docstring)")
    t0 = time.time()
    emb = torch.load(f"{D}/{CHAIN}_embeddings_fp16.pt", map_location="cpu",
                     weights_only=True)
    n = emb.shape[0]
    log(f"{CHAIN}: embeddings {tuple(emb.shape)}")

    import pyarrow.parquet as pq
    label_set = os.environ.get("OPENHEADS_LABEL_SET", os.path.join(D, "label_set.parquet"))
    tl = pq.read_table(label_set, columns=["chain", "address"]).to_pydict()
    with open(f"{D}/addr_to_idx.json") as fh:
        a2i = json.load(fh)
    pos = []
    for ch, ad in zip(tl["chain"], tl["address"], strict=True):
        if ch != CHAIN:
            continue
        i = a2i.get(ad, a2i.get(f"{CHAIN}:{ad}"))
        if i is not None:
            pos.append(int(i))
    pos = np.unique(np.array(pos, dtype=np.int64))
    log(f"labels resolved: {len(pos):,}")
    assert len(pos) >= 100

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pos))
    k = max(1, int(len(pos) * 0.2))
    held, train_pos = pos[perm[:k]], pos[perm[k:]]

    excl = set(pos.tolist())
    neg, seen = [], set()
    n_neg = min(500_000, n // 4)
    while len(neg) < n_neg:
        for v in rng.integers(0, n, size=n_neg).tolist():
            if v in excl or v in seen:
                continue
            seen.add(v)
            neg.append(v)
            if len(neg) >= n_neg:
                break
    neg = np.array(sorted(neg), dtype=np.int64)
    neg_tr, neg_ev = neg[: len(neg) // 2], neg[len(neg) // 2:]
    log(f"negatives: train {len(neg_tr):,} eval {len(neg_ev):,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {dev}")
    head = build_head().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    rows = np.concatenate([train_pos, neg_tr])
    y = np.concatenate([np.ones(len(train_pos), dtype=np.float32),
                        np.zeros(len(neg_tr), dtype=np.float32)])
    pos_w = torch.tensor(len(neg_tr) / max(len(train_pos), 1), device=dev)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_w)
    for ep in range(EPOCHS):
        idx = rng.permutation(len(rows))
        tot = 0.0
        head.train()
        for b0 in range(0, len(idx), BATCH):
            sel = idx[b0:b0 + BATCH]
            xb = emb[rows[sel]].float().to(dev)
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
    scores = torch.empty(n, dtype=torch.float32)
    with torch.no_grad():
        for b0 in range(0, n, 262_144):
            xb = emb[b0:b0 + 262_144].float().to(dev)
            scores[b0:b0 + 262_144] = torch.sigmoid(
                head(xb).squeeze(-1)).cpu()
    log(f"full population scored ({n:,} nodes, {time.time()-t0:.0f}s)")

    s_np = scores.numpy()
    s_held = s_np[held]
    s_neg_ev = np.sort(s_np[neg_ev])
    res = {"chain": CHAIN, "n_nodes": int(n), "labels": int(len(pos)),
           "train_pos": int(len(train_pos)), "held": int(len(held)),
           "head": "MLP 128-256-128-1, focal g=2, pos_weight balanced",
           "fpr_points": {}}
    for fpr in FPRS:
        tau_pop = float(np.quantile(s_np, 1.0 - fpr))
        kk = max(0, min(len(s_neg_ev) - 1, int(len(s_neg_ev) * (1 - fpr))))
        tau_neg = float(s_neg_ev[kk])
        res["fpr_points"][str(fpr)] = {
            "tau_population": tau_pop,
            "tau_negative_sample": tau_neg,
            "held_recall_vs_population_tau": float((s_held > tau_pop).mean()),
            "held_recall_vs_negative_tau": float((s_held > tau_neg).mean()),
            "flagged_at_population_tau": int((s_np > tau_pop).sum()),
        }
        log(f"  fpr={fpr}: held_recall(pop τ)="
            f"{(s_held > tau_pop).mean():.4f} "
            f"(neg τ)={(s_held > tau_neg).mean():.4f}")

    top_idx = np.argsort(-s_np)[:TOPK]
    i2a: dict[int, str] = {}
    want = set(int(v) for v in top_idx.tolist())
    for a, i in a2i.items():
        if int(i) in want:
            i2a[int(i)] = a
    top = [{"rank": r + 1, "node_idx": int(i),
            "address": i2a.get(int(i), ""), "score": float(s_np[i])}
           for r, i in enumerate(top_idx.tolist())]
    known = set(pos.tolist())
    n_known = sum(1 for t in top if t["node_idx"] in known)
    res["topk"] = {"k": TOPK, "known_label_hits": n_known,
                   "addresses_resolved": sum(1 for t in top if t["address"])}
    log(f"top-{TOPK}: {n_known} known labels inside, "
        f"{res['topk']['addresses_resolved']} addresses resolved")

    with open(f"{D}/top{TOPK}.json", "w") as fh:
        json.dump(top, fh, indent=1)
    with open(f"{D}/perchain_head_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    # Name and key layout are the contract tau_scan.py loads with
    # strict=True: bare nn.Sequential keys in fincrime_<chain>.pt.
    torch.save(head.state_dict(), f"{D}/fincrime_{CHAIN}.pt")
    if OUT_URI:
        for name in (f"top{TOPK}.json", "perchain_head_results.json",
                     f"fincrime_{CHAIN}.pt"):
            argv = ["aws", "s3", "cp", os.path.join(D, name),
                    f"{OUT_URI.rstrip('/')}/{name}", "--only-show-errors"]
            if REGION:
                argv += ["--region", REGION]
            run(argv)
        log(f"uploaded 3 artefacts to {OUT_URI}")
    log(f"PERCHAIN_HEAD_EXPORT_DONE {CHAIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
