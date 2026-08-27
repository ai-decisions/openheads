#!/usr/bin/env python3
"""Population threshold (tau) scan over every scored node.

Scores EVERY node with the head its chain is judged by, then records the
population quantiles at the target false-positive rates. This is what makes
alert volume knowable in advance: a threshold set as a population quantile
answers "how many alerts per day", which a threshold fitted on a labelled
negative sample cannot.

Two conventions are recorded side by side, because they do NOT agree: a
quantile over the whole population and a quantile over labelled negatives
land on different values, and a threshold carried from one convention to the
other silently shifts the alert count.

Chain routing matters: a head trained on one chain does not transfer, so
serving must route per chain to honour the measured numbers.

Env — required (fail-closed in _check_env, no defaults):
  OPENHEADS_CHUNKS_DIR    directory of scored embedding chunks
  OPENHEADS_HEADS_DIR     directory of head state_dicts
  OPENHEADS_CHAIN_STARTS  comma-separated node-id offsets, one per chain
  OPENHEADS_N_TOTAL       total node count
  OPENHEADS_CHAINS        comma-separated chain names, same order and length
                          as the offsets
Env — optional:
  OPENHEADS_TAU_OUT        output json path (default ./runs/tau.json)
  OPENHEADS_TAIL_START     ids >= this belong to the tail chain
  OPENHEADS_TAIL_CHAIN     chain index the tail rows are routed to (default 0)
  OPENHEADS_AI_AGENT_CHAIN chain index scored by the optional second head
  OPENHEADS_SHARED_HEADS   "borrower=owner[,..]" pairs, e.g. "1=0": chain 1
                           is scored by chain 0's head (chains trained
                           together share one file)
Heads dir layout (bare nn.Sequential state_dicts):
  fincrime_<chain>.pt, and optionally ai_agent_<chain>.pt
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

from openheads.heads import build_aiagent_head, build_fincrime_head

# Env is read at import time so every knob sits in one visible block; nothing
# is validated here — the fail-closed check lives in _check_env(), one call
# before any work starts, and names every missing variable at once instead of
# dying on the first attribute some later line happens to touch.
REQUIRED_ENV = (
    "OPENHEADS_CHUNKS_DIR",
    "OPENHEADS_HEADS_DIR",
    "OPENHEADS_CHAIN_STARTS",
    "OPENHEADS_N_TOTAL",
    "OPENHEADS_CHAINS",
)
CHUNKS = os.environ.get("OPENHEADS_CHUNKS_DIR", "")
HEADS = os.environ.get("OPENHEADS_HEADS_DIR", "")
OUT = os.environ.get("OPENHEADS_TAU_OUT", "./runs/tau.json")
_STARTS_RAW = os.environ.get("OPENHEADS_CHAIN_STARTS", "")
STARTS = [int(v) for v in _STARTS_RAW.split(",")] if _STARTS_RAW else []
N_TOTAL = int(os.environ.get("OPENHEADS_N_TOTAL", "0"))
# When extra rows of one chain are appended AFTER the last chain partition
# (a token-transfer tail), every id >= this offset is routed to the tail
# chain's head. Unset = plain partition lookup.
_TAIL = os.environ.get("OPENHEADS_TAIL_START", "")
TAIL_START = int(_TAIL) if _TAIL else None
TAIL_CHAIN = int(os.environ.get("OPENHEADS_TAIL_CHAIN", "0"))
CHAINS = os.environ.get("OPENHEADS_CHAINS", "").split(",") if os.environ.get(
    "OPENHEADS_CHAINS") else []
FPRS = (0.01, 0.001, 0.0001)
BATCH = 262_144


def _check_env() -> None:
    """Fail closed: no private path or graph geometry is guessed for you."""
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "missing required environment: "
            + ", ".join(missing)
            + " (see this module's docstring)"
        )
    if len(CHAINS) != len(STARTS):
        # A silent mismatch here surfaces hours later as a KeyError in the
        # middle of the population pass, after the heads are already loaded.
        raise SystemExit(
            f"OPENHEADS_CHAINS has {len(CHAINS)} names but "
            f"OPENHEADS_CHAIN_STARTS has {len(STARTS)} offsets"
        )
    if TAIL_START is not None and not 0 <= TAIL_CHAIN < len(STARTS):
        raise SystemExit(
            f"OPENHEADS_TAIL_CHAIN={TAIL_CHAIN} is out of range for "
            f"{len(STARTS)} chains"
        )


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {m}", flush=True)


# Architectures live in openheads.heads — one source for the scripts that
# train the checkpoints and this scan that loads them strict=True.
fincrime_arch = build_fincrime_head
aiagent_arch = build_aiagent_head


def load_head(build, path: str, dev: str) -> nn.Module:
    head = build()
    state = torch.load(path, map_location="cpu", weights_only=True)
    head.load_state_dict(state, strict=True)
    return head.to(dev).eval()


def chain_of(idx: int) -> int:
    if TAIL_START is not None and idx >= TAIL_START:
        return TAIL_CHAIN             # appended tail rows -> tail chain's head
    for ci in range(len(STARTS) - 1, -1, -1):
        if idx >= STARTS[ci]:
            return ci
    raise ValueError(idx)


def seg_end_of(idx: int) -> int:
    """End (exclusive) of the CONTIGUOUS segment containing idx — the run
    the chunk loop may score with one head before re-deriving the chain."""
    if TAIL_START is not None:
        if idx >= TAIL_START:
            return N_TOTAL            # tail runs to the end of the graph
        for ci in range(len(STARTS) - 1, -1, -1):
            if idx >= STARTS[ci]:
                return STARTS[ci + 1] if ci + 1 < len(STARTS) else TAIL_START
        raise ValueError(idx)
    ci = chain_of(idx)
    return STARTS[ci + 1] if ci + 1 < len(STARTS) else N_TOTAL


def main() -> int:
    _check_env()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # expected per-chain row counts (the tail chain owns the appended rows)
    seg_ends = STARTS[1:] + [TAIL_START if TAIL_START is not None else N_TOTAL]
    expected = [seg_ends[i] - STARTS[i] for i in range(len(STARTS))]
    if TAIL_START is not None:
        expected[TAIL_CHAIN] += N_TOTAL - TAIL_START
    # One fincrime head per chain: <HEADS>/fincrime_<chain>.pt. Chains that
    # were trained together share one file — declare that with
    # OPENHEADS_SHARED_HEADS="1=0" (chain 1 is scored by chain 0's head).
    shared: dict[int, int] = {}
    for pair in os.environ.get("OPENHEADS_SHARED_HEADS", "").split(","):
        if pair.strip():
            borrower, owner = pair.split("=")
            shared[int(borrower)] = int(owner)
    fin_heads: dict[int, nn.Module] = {}
    for ci, name in enumerate(CHAINS):
        if ci not in shared:
            fin_heads[ci] = load_head(fincrime_arch, f"{HEADS}/fincrime_{name}.pt", dev)
    for borrower, owner in shared.items():
        fin_heads[borrower] = fin_heads[owner]
    # The second head is optional: it stays off on chains without class
    # labels, so it is scored for one declared chain index only.
    _ai_chain = os.environ.get("OPENHEADS_AI_AGENT_CHAIN", "")
    ai_chain = int(_ai_chain) if _ai_chain else None
    ai_head = (
        load_head(aiagent_arch, f"{HEADS}/ai_agent_{CHAINS[ai_chain]}.pt", dev)
        if ai_chain is not None
        else None
    )
    log(f"heads loaded strict=True on {dev}")

    with open(os.path.join(CHUNKS, "index.json")) as fh:
        index = json.load(fh)
    scores: dict[int, list[np.ndarray]] = {c: [] for c in range(len(CHAINS))}
    ai_scores: list[np.ndarray] = []
    rows_seen = 0
    for rec in index["chunks"]:
        buf = torch.load(os.path.join(CHUNKS, rec["file"]),
                         map_location="cpu", weights_only=True)
        start = rec["start"]
        n = rec["n"]
        pos = 0
        while pos < n:
            ci = chain_of(start + pos)
            seg_end = min(n, seg_end_of(start + pos) - start)
            xb = buf[pos:seg_end]
            with torch.no_grad():
                for b0 in range(0, len(xb), BATCH):
                    blk = xb[b0:b0 + BATCH].float().to(dev)
                    s = torch.sigmoid(
                        fin_heads[ci](blk).squeeze(-1)).cpu().numpy()
                    scores[ci].append(s.astype(np.float32))
                    if ai_head is not None and ci == ai_chain:
                        sa = torch.sigmoid(
                            ai_head(blk).squeeze(-1)).cpu().numpy()
                        ai_scores.append(sa.astype(np.float32))
            pos = seg_end
        rows_seen += n
        if rec["chunk_id"] % 100 == 0:
            log(f"chunk {rec['chunk_id']}/{len(index['chunks'])} "
                f"({time.time()-t0:.0f}s)")
    if rows_seen != N_TOTAL:
        raise SystemExit(f"rows {rows_seen} != {N_TOTAL}")

    out: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "n_total": N_TOTAL, "routing": "per-chain heads",
                 "chains": {}}
    all_scores = []
    for ci, name in enumerate(CHAINS):
        s = np.concatenate(scores[ci])
        if len(s) != expected[ci]:
            raise SystemExit(f"{name}: {len(s)} != {expected[ci]}")
        all_scores.append(s)
        entry = {"n": int(len(s)), "fpr_points": {}}
        for fpr in FPRS:
            tau = float(np.quantile(s, 1.0 - fpr))
            entry["fpr_points"][str(fpr)] = {
                "tau_population": tau,
                "flagged": int((s > tau).sum())}
        out["chains"][name] = entry
        log(f"{name}: n={len(s):,} tau@1e-3={entry['fpr_points']['0.001']['tau_population']:.6f}")
    glob = np.concatenate(all_scores)
    out["global_population"] = {str(f): float(np.quantile(glob, 1.0 - f))
                               for f in FPRS}
    out["note_global"] = ("a global threshold mixes per-chain score scales; "
                          "recorded for visibility, routing stays per-chain")
    if ai_scores:
        sa = np.concatenate(ai_scores)
        out[f"ai_agent_{CHAINS[ai_chain]}"] = {
            str(f): float(np.quantile(sa, 1.0 - f)) for f in FPRS}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"TAU_SCAN_DONE {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
