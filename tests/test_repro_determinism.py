"""Determinism of the published forward path.

What this pins, and why each piece is here:

- ``test_architecture_fingerprint_is_pinned`` hashes the *shape* of the model
  (state_dict keys and tensor shapes). A silent architecture edit — one layer
  wider, one norm dropped — changes this hash even when every numeric test
  still passes.
- ``test_forward_sha_is_reproducible_in_process`` proves the forward pass is
  deterministic at all: same seed, same input, two independent constructions,
  identical digest.
- ``test_forward_sha_matches_pinned_reference`` is the criterion itself: the
  digest must equal a value pinned from a CI run. Without this assertion the
  step would only *print* a number and could never fail.

Values are hashed at fixed precision (``f"{v:.6f}"``), never via ``repr``:
float32 carries ~7 significant decimal digits, so 6 decimals are stable
across interpreter and numpy versions, while ``repr`` prints 17 and turns a
last-bit difference into a different sha.

Run ``pytest -s -q tests/test_repro_determinism.py`` to see the digests.
"""

from __future__ import annotations

import hashlib
import sys

import pytest

torch = pytest.importorskip("torch", reason="determinism pins need torch")
pytest.importorskip("torch_geometric", reason="the encoder is built on torch_geometric")

from openheads.models import PerChainNormEncoder  # noqa: E402

# Fixed geometry: two chain partitions in one merged node space, a tail that
# belongs to chain 0. Small enough to run on any CPU runner in milliseconds,
# wide enough to exercise per-chain routing (the paper's central claim).
SEED = 42
IN_CHANNELS = 16
HIDDEN_CHANNELS = 32
CHAIN_STARTS = [0, 6]
TAIL_START = 10
N_NODES = 12

# Both interpreters in the matrix must produce these; a mismatch means the
# forward path changed and the change has to be explained, not re-pinned
# silently.
#
# The architecture digest hashes state_dict keys and shapes — plain strings,
# identical on any interpreter — so it is pinned directly.
PINNED_ARCHITECTURE_SHA = "d6bf8f2443eedb371225f19d3c053fe2992f59e458124d44e3d8b2456e130196"
# The forward digest hashes float output and is pinned ONLY from a CI run:
# pinning a value measured on a local interpreter would encode whatever that
# machine's torch build happens to produce. Measured evidence that this
# matters: a local Python 3.13 / torch 2.13 build produced a different digest
# than the CI runners, which is why the value below comes from a CI run and
# not from a developer machine. Re-taken whenever the fixture changes: making
# the per-chain norms distinguishable changed it, as it should.
PINNED_FORWARD_SHA = "d3bc48b73e66b02960419f379b592449aecd0c2132aad58939a6be5bf40cd612"
# Interpreter versions the digest above was measured on. The CI matrix pins
# these two; on any other version the digest is reported and the assertion is
# skipped rather than failing a healthy install.
PINNED_ON_PYTHON = {"3.11", "3.12"}


def _build_encoder() -> PerChainNormEncoder:
    torch.manual_seed(SEED)
    encoder = PerChainNormEncoder(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        chain_starts=CHAIN_STARTS,
        dropout=0.3,
        tail_start=TAIL_START,
        tail_chain=0,
    )
    # Give each chain's normalisation DISTINCT parameters. Freshly built, every
    # LayerNorm holds weight=1 / bias=0, so which one a row is routed to makes
    # no numeric difference — and the digest below would then be blind to
    # routing entirely: deleting tail routing, or collapsing all norms into
    # one, would leave it unchanged. Measured, not assumed: that was the case
    # before these two lines.
    with torch.no_grad():
        for index, norm in enumerate(encoder.chain_norms):
            norm.weight.fill_(1.0 + index)
            norm.bias.fill_(float(index))
    encoder.eval()  # dropout off, norms on running statistics
    return encoder


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A deterministic graph: features from a seeded generator, a fixed ring."""
    generator = torch.Generator().manual_seed(SEED)
    x = torch.rand((N_NODES, IN_CHANNELS), generator=generator)
    ring = [(i, (i + 1) % N_NODES) for i in range(N_NODES)]
    edge_index = torch.tensor(ring, dtype=torch.long).t().contiguous()
    global_node_ids = torch.arange(N_NODES, dtype=torch.long)
    return x, edge_index, global_node_ids


def _digest(values: list[float]) -> str:
    payload = "|".join(f"{v:.6f}" for v in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def architecture_sha(encoder: PerChainNormEncoder) -> str:
    parts = [f"{k}:{tuple(v.shape)}" for k, v in sorted(encoder.state_dict().items())]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def forward_sha(encoder: PerChainNormEncoder) -> str:
    x, edge_index, global_node_ids = _fixed_batch()
    with torch.no_grad():
        embeddings = encoder.get_embeddings(x, edge_index, global_node_ids)
    return _digest(embeddings.flatten().tolist())


def test_architecture_fingerprint_is_pinned() -> None:
    sha = architecture_sha(_build_encoder())
    print(f"REPRO_SHA architecture {sha}")
    if PINNED_ARCHITECTURE_SHA != "PIN_FROM_CI":
        assert sha == PINNED_ARCHITECTURE_SHA, (
            "architecture fingerprint changed: state_dict keys or shapes differ "
            "from the pinned reference"
        )


def test_forward_sha_is_reproducible_in_process() -> None:
    first = forward_sha(_build_encoder())
    second = forward_sha(_build_encoder())
    print(f"REPRO_SHA forward {first}")
    assert first == second, "forward pass is not deterministic under a fixed seed"


def test_forward_sha_matches_pinned_reference() -> None:
    sha = forward_sha(_build_encoder())
    if PINNED_FORWARD_SHA == "PIN_FROM_CI":
        pytest.fail(
            "PINNED_FORWARD_SHA is unset: take the digest printed by the CI job "
            f"'Repro (deterministic forward sha)' and pin it here. Measured now: {sha}"
        )
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if version not in PINNED_ON_PYTHON:
        # A float digest is a property of the interpreter and torch build that
        # produced it. Asserting the pin on an unpinned version would report a
        # perfectly healthy install as a regression, so this states the fact
        # and skips instead of lying in either direction.
        pytest.skip(
            f"forward digest is pinned for Python {sorted(PINNED_ON_PYTHON)}, "
            f"running {version}; measured here: {sha}"
        )
    assert sha == PINNED_FORWARD_SHA, (
        f"deterministic forward digest changed on Python {version}: "
        f"{sha} != {PINNED_FORWARD_SHA}"
    )


def test_tail_routing_is_part_of_the_pinned_path() -> None:
    """A tail id must reach chain 0's norm, not the last chain's.

    Guards the pin above from becoming a hash of the wrong behaviour: without
    tail routing the digest would still be stable, just wrong.
    """
    encoder = _build_encoder()
    ids = torch.tensor([0, 6, TAIL_START, N_NODES - 1], dtype=torch.long)
    chain_ids = encoder.chain_ids_of(ids).tolist()
    assert chain_ids == [0, 1, 0, 0], f"tail routing broken: {chain_ids}"
