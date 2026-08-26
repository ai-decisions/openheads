# Security Policy

## Reporting a vulnerability

Please report vulnerabilities **privately** to **mail@aidecisions.ai**.
Do not open a public issue for security problems. We will acknowledge
your report and coordinate disclosure with you; a fix or a mitigation
plan is published before details are.

This policy covers the code in this repository. Issues in the hosted
AI DECISIONS platform (aidecisions.ai) go to the same address.

## What this code does — and does not do

- The training tools read inputs **you supply** as parameters. The
  optional warehouse extra (`pyarrow`, with `boto3` for credentials) reads training inputs
  from object storage **you configure**; configuration is fail-closed —
  no default bucket, prefix or region ships anywhere in this tree.
- The only outbound network call this code makes on its own is the
  user-invoked download of the **public** Elliptic benchmark dataset over
  HTTPS (`openheads.download_elliptic`). No telemetry, no callbacks.
- No credentials ship with the repo. AWS access, if you use the warehouse
  extra, goes through pyarrow's S3FileSystem and the standard AWS
  credential chain — this repository never reads a credential itself.
- The repository ships **no data, no label set, no model weights, and no
  threshold values** (see `README.md` → Boundary).
- Checkpoints are loaded with `torch.load(weights_only=True)`, which
  refuses arbitrary-object deserialisation. Still, load only checkpoints
  you trust.

## Supply chain

- CI (GitHub Actions, standard runners) gates every change: module
  compile, lint (`ruff`), a **sanitize gate** (`tools/sanitize_gate.py` —
  a generic scan that blocks secrets, storage URIs, machine-local paths,
  restricted-source data, and model weights), unit tests, a determinism
  job that recomputes the deterministic-forward sha, and a CPU training
  smoke on synthetic data.
- Actions are pinned to major versions; runtime dependencies are minimal
  (`numpy`, `torch`, `torch-geometric`, `pandas`, `networkx`,
  `scikit-learn`, `structlog`; extras add `pyarrow`/`boto3`, `torch-sparse`,
  `pytorch-lightning`, `hdbscan`, `kagglehub`).
- Contributions require a DCO sign-off (`git commit -s`).
