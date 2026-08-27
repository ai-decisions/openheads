"""Label mapping for blockchain addresses.

Loads labels from sources YOU supply:
  - a sanctions list in OFAC SDN CSV form (designated addresses -> illicit)
  - your own entity CSV (address,label) for anything else
  - the public Elliptic dataset (Bitcoin transaction labels)

No addresses are hardcoded in this module, and that is deliberate. A label
asserting that an address belongs to a named exchange, or that it is
controlled by a named attacker, is a factual claim about a real party. Such a
claim needs a source and a date next to it, which is exactly what the
label-set builder in this repository enforces for every row it ingests
(`scripts/build_label_set.py` refuses a row with no `source_url` /
`source_date`). A dict baked into library code carries neither, and it rots:
a sanctions designation can be lifted — Tornado Cash was delisted in March
2025 — and a shipped constant keeps asserting the old status forever.

Label encoding:
  0 = licit (known clean: exchanges, verified entities)
  1 = illicit (sanctioned, flagged, confirmed criminal)
  -1 = unknown (default, no label)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import structlog

from openheads.address_case import canonical_case

logger = structlog.get_logger(__name__)

LABEL_LICIT = 0
LABEL_ILLICIT = 1
LABEL_UNKNOWN = -1

ETH_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]{40}")


def load_ofac_labels(sdn_csv_path: str | Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    path = Path(sdn_csv_path)
    if not path.exists():
        logger.warning("ofac_file_not_found", path=str(path))
        return labels

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            text = " ".join(row)
            addresses = ETH_ADDRESS_PATTERN.findall(text.lower())
            for addr in addresses:
                labels[addr] = LABEL_ILLICIT

    logger.info("loaded_ofac_labels", n_addresses=len(labels))
    return labels


def load_known_labels(entity_csv_path: str | Path) -> dict[str, int]:
    """Load `address,label` rows from a CSV you supply.

    `label` is either the integer encoding above or one of the words
    `licit` / `illicit`. Rows with anything else are skipped and counted,
    never silently coerced: a typo that became `unknown` would quietly
    remove an entity from the training signal.
    """
    labels: dict[str, int] = {}
    path = Path(entity_csv_path)
    if not path.exists():
        logger.warning("entity_file_not_found", path=str(path))
        return labels

    word_to_label = {"licit": LABEL_LICIT, "illicit": LABEL_ILLICIT}
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                skipped += 1
                continue
            addr, raw = row[0].strip(), row[1].strip().lower()
            if not ETH_ADDRESS_PATTERN.fullmatch(addr):
                skipped += 1
                continue
            if raw in word_to_label:
                labels[addr.lower()] = word_to_label[raw]
            elif raw in {str(LABEL_LICIT), str(LABEL_ILLICIT)}:
                labels[addr.lower()] = int(raw)
            else:
                skipped += 1
    logger.info("loaded_known_labels", total=len(labels), skipped=skipped)
    return labels


def merge_labels(*label_dicts: dict[str, int]) -> dict[str, int]:
    """Merge label dicts; on conflict, illicit wins.

    Keys are normalised through `canonical_case`, NOT a blanket `.lower()`:
    base58check addresses (BTC legacy, Tron, ...) carry their case as payload
    and lowercasing destroys them — `address_case` states that as a MUST for
    every ingest path, and this is an ingest path.
    """
    merged: dict[str, int] = {}
    for d in label_dicts:
        for addr, label in d.items():
            key = canonical_case(addr)
            if key in merged and merged[key] != label:
                # illicit takes priority over licit
                merged[key] = LABEL_ILLICIT
            else:
                merged[key] = label
    logger.info("merged_labels", total=len(merged))
    return merged


def propagate_labels_1hop(
    labels: dict[str, int],
    edges: list[tuple[str, str]],
    suspicious_score: float = 0.5,
) -> dict[str, float]:
    """Semi-supervised: 1-hop neighbors of illicit nodes get a suspicion score.

    Returns dict of address → suspicion score (0.0 to 1.0).
    Does NOT override existing labels — only adds scores for unlabeled neighbors.
    """
    illicit_addrs = {addr for addr, label in labels.items() if label == LABEL_ILLICIT}
    scores: dict[str, float] = {}

    for src, dst in edges:
        # canonical_case, not .lower(): edges may carry base58check
        # addresses, whose case is payload (see address_case).
        src_l = canonical_case(src)
        dst_l = canonical_case(dst)
        if src_l in illicit_addrs and dst_l not in labels:
            scores[dst_l] = max(scores.get(dst_l, 0.0), suspicious_score)
        if dst_l in illicit_addrs and src_l not in labels:
            scores[src_l] = max(scores.get(src_l, 0.0), suspicious_score)

    logger.info("propagated_labels", n_suspicious=len(scores))
    return scores
