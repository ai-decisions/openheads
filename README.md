# openheads

Training code for a graph-based financial-crime detector, built and used in
production by [AI DECISIONS](https://aidecisions.ai/): a three-layer
GraphSAGE encoder with per-chain normalisation over a multi-chain
transaction graph, small MLP scoring heads on the shared embedding, a
warm-start training recipe, and decision thresholds calibrated as exact
population quantiles of the score distribution. In production the method
runs on a five-EVM-chain graph of 835,330,427 addresses and 15,826,261,934
edges. This repository is the **method**, not the data or the result.

## Boundary — what is open here, and what is not

**Open (this repo):**

- **Model architectures** (`openheads.models`): the GraphSAGE-3 encoder,
  the per-chain normalisation encoder (`PerChainNormEncoder` — one
  learnable normalisation per chain segment, routed by node index), and
  the MLP scoring heads.
- **Backbone training recipe, including warm-start**
  (`openheads.heads`, `openheads.graph_batching`,
  `scripts/train_backbone.py`): the optimiser and sampling settings, the
  strict full warm-start from a previous backbone checkpoint with freshly
  initialised heads, and the neighbour-sampled training loop.
- **Head training** (`scripts/train_heads.py`,
  `scripts/export_perchain_head.py`): scoring heads on the shared
  embedding — one trained on a sampled pool, one on a chain's full
  population with a top-K export. Both report held-out recall against
  both threshold conventions, so the reported number is never silently
  the easier one.
- **Threshold calibration mechanism** (`scripts/tau_scan.py`): thresholds
  as population quantiles measured by an exact scan of every score, per
  chain segment — the mechanism that makes the alert volume known in
  advance. The mechanism is open; the production threshold values are not.
- **Label-set builder** (`scripts/build_label_set.py`,
  `openheads.label_mapping`): assembles a training label set from public
  label sources you configure. The builder is open; no labels ship with it.
- **A Lightning-based trainer for the Elliptic path** (`openheads.trainer`,
  extra `[legacy-trainer]`): the older single-graph training loop. It is not
  on the backbone path above and needs its extra installed to import.
- **A reproducible legacy path on the public Elliptic benchmark**
  (`openheads.download_elliptic`, `openheads.data_prep`,
  `openheads.feature_pipeline`): downloads the public dataset at run time
  and trains on it. We publish **no benchmark figure** for this path: the
  accompanying paper explicitly reports no Elliptic result, and this
  repository makes no such claim either.
- **A synthetic training smoke** (`openheads.synthetic`,
  `scripts/train_backbone.py --synthetic`): trains on an in-memory
  synthetic graph — no data, no object storage, no GPU.

**Not open (deliberately):**

- Trained weights, the assembled production label set (301,304 rows, as
  reported in the accompanying paper), the graph substrate and emitted
  embeddings, the production threshold values, and the serving code.

**The open path is the backbone path.** The assembly of the production
heads (`head_v8`) is **not** open: its steps — extending the labelled pool
and measuring the candidate heads — are not part of this repository. What
you can reproduce here is the backbone recipe, the architecture, the head
training mechanics and the calibration mechanism, not the shipped
production heads.

## No data, no weights

The repository ships **no datasets at all**: no warehouse data, no label
set, no graph, no embeddings, no model weights, no threshold values.
Training inputs are parameters you supply; warehouse configuration is
fail-closed (there is no default bucket or prefix anywhere in the tree).
The one dataset the code can fetch is the **public** Elliptic benchmark,
downloaded at run time from its public distribution — it is not
redistributed here.

## Install

```bash
pip install -e .                 # core: everything the architecture and the synthetic smoke need
pip install -e .[warehouse]      # + pyarrow, boto3: read training inputs from object storage
pip install -e .[sampling]       # + torch-sparse: the real-data CSR path (needs a wheel matching your torch)
pip install -e .[legacy-trainer] # + pytorch-lightning, torchmetrics, optuna: openheads.trainer
pip install -e .[cluster]        # + hdbscan, scikit-learn extras: openheads.community_analysis
pip install -e .[elliptic]       # + kagglehub: fetch the public Elliptic benchmark
pip install -e .[dev]            # everything the test suite touches
```

Python 3.11+. The reproducibility digests are pinned for 3.11 and 3.12 (the
versions CI runs); on any other version the digest is printed and its
assertion is skipped rather than failing a healthy install.

`torch-sparse` is deliberately NOT a core dependency: it ships as a compiled
wheel that must match your exact torch build, so a plain `pip install` cannot
guarantee it. Only the real-data path needs it, and that path says so with a
precise message if it is missing.

## Quickstart

```bash
pip install -e .[dev]
python scripts/train_backbone.py --synthetic
```

This trains the real architecture on an in-memory synthetic graph: no
data, no credentials, no GPU. The run prints `loss finite` and
`checkpoint written/reloaded`; CI gates on exactly those lines.

Runnable entry points. Two take command-line flags; the other three are
configured entirely through `OPENHEADS_*` environment variables and print
the list of variables they need when one is missing (see each module's
docstring):

```text
scripts/train_backbone.py       flags   backbone training (--synthetic runs the smoke)
scripts/build_label_set.py      flags   label-set assembly (--batch, --batch-id, --execute)
scripts/train_heads.py          env     scoring head on pooled embeddings
scripts/export_perchain_head.py env     scoring head on a chain's full population + top-K export
scripts/tau_scan.py             env     population-quantile threshold scan
```

## How the stages connect

Each stage reads what the previous one writes. The file names and the
state_dict key layout ARE the interface — the calibration step loads heads
with `strict=True`, so a renamed file or a prefixed key is a hard failure,
not a warning:

```text
train_backbone.py   -> backbone.pt          encoder state_dict
                       heads.pt             two-head module (PREFIXED keys)
                       fincrime_<tag>.pt    one head, bare keys  <- calibration reads this
                       ai_agent_<tag>.pt    one head, bare keys  <- calibration reads this
train_heads.py      -> fincrime_<chain>.pt  one head, bare keys
export_perchain_head.py -> fincrime_<chain>.pt + top-K json + metrics json
tau_scan.py         <- fincrime_<chain>.pt for every chain in OPENHEADS_CHAINS
                       (plus ai_agent_<chain>.pt when OPENHEADS_AI_AGENT_CHAIN is set)
                    -> tau.json            population quantiles per chain
build_label_set.py  -> label_set.parquet + manifest.json (append-only ledger)
```

`<tag>` for the backbone heads defaults to `all` and is set with
`OPENHEADS_HEAD_TAG`; name it after the chain when you calibrate per chain.

## Reproducibility

Two classes of checks run in CI on clean clones, under Python 3.11 and
3.12 independently:

- **Deterministic path:** a fixed-seed forward pass of the architecture on
  a synthetic input; the sha256 of the output tensor must be identical
  across both runners (`pytest -s tests/test_repro_determinism.py` prints
  it). Numbers are pinned to fixed precision before hashing so
  interpreter-level repr drift cannot slip through.
- **Stochastic path (training):** `scripts/train_backbone.py --synthetic`
  must run end to end with a finite loss and a checkpoint that is written
  and read back. Weight shas are deliberately **not** gated: torch
  training is legitimately non-deterministic across machines.

## Contributing

Contributions are accepted under the Developer Certificate of Origin
(sign-off line in commits, `git commit -s`). License: Apache-2.0, see
`LICENSE` and `NOTICE`.
