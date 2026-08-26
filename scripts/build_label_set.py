#!/usr/bin/env python3
"""Fold an attributed label batch into a label set, APPEND-ONLY, with an
explicit `provenance` column.

The point of this script is that a label set is a *ledger*, not a table you
overwrite. A new generation is the previous one plus attributions you derived
yourself from a first-party anchor (a sanctions designation, a court filing,
a licensed-registry entry, an on-chain fact), each carrying source_url and
source_date. No third-party label strings are copied in — the licences of
label aggregators do not survive redistribution, and an attribution you
cannot point at a document is not evidence.

Guarantees (asserted at build time — a violation HALTs, nothing is written):
  * append-only: every (chain, address) key of the base survives; rows never
    decrease. A label set that can lose rows silently invalidates every
    model trained against an earlier generation.
  * case-intact: an address whose case has been destroyed (lower-cased
    checksum address) or a synthetic '::' key with no on-chain address is
    REFUSED, never ingested. Case-destroyed EVM addresses screen as
    false negatives, so accepting one poisons the set quietly.
  * provenance: every row carries a `provenance` array; base rows get [];
    each attributed add/merge appends one entry
    {batch_id, anchor_type, method, source_url, source_date, attributed_utc}.

Batch input: JSONL, one object per line:
  {"chain":"eth","address":"0x...","classes":["exchange","vasp:alias::name"],
   "anchor_type":"ofac|court|registry|onchain","method":"...",
   "source_url":"https://...","source_date":"2026-08-01","note":"..."}

`merge_batch` is a pure function and is unit-tested offline; `main` does the
object-storage I/O.

Env:
  OPENHEADS_LABELS_URI      base object-storage prefix (required with --execute)
  OPENHEADS_BASE_PARQUET    key of the base parquet under that prefix
  OPENHEADS_BASE_MANIFEST   key of the base manifest under that prefix
  OPENHEADS_OUT_PREFIX      key prefix for the new generation
  OPENHEADS_REGION          region for object storage
  OPENHEADS_BASE_ROWS       expected base row count (integrity check)
  OPENHEADS_BASE_SHA        expected base sha prefix (integrity check)

Usage:
    python3 build_label_set.py --batch attribution_batch.jsonl \\
        --batch-id batch-01 [--execute]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openheads.address_case import canonical_case, is_case_destroyed
from openheads.tron_address import InvalidTronAddress, base58check_to_hex

LABELS_URI = os.environ.get("OPENHEADS_LABELS_URI", "")
BASE_PARQUET = os.environ.get("OPENHEADS_BASE_PARQUET", "label_set_base.parquet")
BASE_MANIFEST = os.environ.get("OPENHEADS_BASE_MANIFEST", "manifest.json")
OUT_PREFIX = os.environ.get("OPENHEADS_OUT_PREFIX", "label_set_next")
# Integrity expectations for the base generation. Unset = not checked; set
# them and the build refuses a base that is not the one you meant to extend.
_BASE_ROWS = os.environ.get("OPENHEADS_BASE_ROWS", "")
BASE_EXPECT_ROWS = int(_BASE_ROWS) if _BASE_ROWS else None
BASE_EXPECT_SHA = os.environ.get("OPENHEADS_BASE_SHA", "")

CHAIN_NORM = {
    "ethereum": "eth", "eth": "eth",
    "xdai": "gnosis", "gnosis": "gnosis",
    "arbitrum": "arbitrum", "arb": "arbitrum",
    "base": "base", "tron": "tron", "trx": "tron",
    "btc": "btc", "bitcoin": "btc",
}
INCLUDED = {"eth", "tron", "base", "arbitrum", "gnosis"}


def norm_chain(c: str) -> str:
    c = (c or "").strip().lower()
    return CHAIN_NORM.get(c, c)


def make_provenance(rec: dict, batch_id: str, now_iso: str) -> str:
    """One JSON provenance entry for an attributed row."""
    return json.dumps(
        {
            "batch_id": batch_id,
            "anchor_type": rec.get("anchor_type"),
            "method": rec.get("method"),
            "source_url": rec.get("source_url"),
            "source_date": rec.get("source_date"),
            "attributed_utc": now_iso,
            "note": rec.get("note"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


class BatchRefused(Exception):  # noqa: N818 — refusal signal, not an error condition
    pass


class AppendOnlyViolation(RuntimeError):  # noqa: N818 — invariant name, matches repo precedent
    """Raised (never assert — `python -O` must not disable it) when a merge
    would drop a base key or shrink the set."""


def validate_batch_row(rec: dict) -> tuple[str, str]:
    """Return (chain, canonical_address) or raise BatchRefused with a reason."""
    addr = (rec.get("address") or "").strip()
    if not addr:
        raise BatchRefused("no address")
    if "::" in addr:
        raise BatchRefused("synthetic '::' key (no on-chain address)")
    chain = norm_chain(rec.get("chain"))
    if not chain:
        raise BatchRefused("no chain")
    a = canonical_case(addr)
    # Tron: the product/graph key form is 0x-hex (0x41 prefix stripped). A
    # A base58 T-form address is normalised to hex here. Left as-is it would
    # create a ('tron', 'T...') key that lookups keyed on hex never reach --
    # the row exists, screening misses it.
    if chain == "tron" and len(a) == 34 and a[0] == "T":
        try:
            a = base58check_to_hex(a)
        except InvalidTronAddress:
            raise BatchRefused("invalid tron base58check address") from None
    if is_case_destroyed(a, chain):
        raise BatchRefused("case-destroyed address")
    if not rec.get("source_url") or not rec.get("source_date"):
        raise BatchRefused("missing source_url/source_date provenance")
    cls = rec.get("classes") or []
    if not isinstance(cls, list) or not cls:
        raise BatchRefused("no classes")
    return chain, a


def merge_batch(
    base: dict[tuple[str, str], dict],
    batch: list[dict],
    batch_id: str,
    now_iso: str,
) -> tuple[dict[tuple[str, str], dict], dict]:
    """Pure append-only merge of an attributed *batch* onto *base* rows.

    *base* maps (chain, address) -> row dict with at least keys
    classes/sources/origins/graph_key/provenance (provenance defaulted to []).
    Returns (rows, stats). Never removes a base key. Refused rows are counted,
    not merged.
    """
    rows = {k: dict(v) for k, v in base.items()}
    for r in rows.values():
        r.setdefault("provenance", [])
        r["classes"] = set(r.get("classes") or [])
        r["sources"] = set(r.get("sources") or [])
        r["origins"] = set(r.get("origins") or [])

    base_keys = set(rows)
    st = collections.Counter()
    refused: list[dict] = []
    origin = f"attribution_batch:{batch_id}"

    for rec in batch:
        try:
            chain, a = validate_batch_row(rec)
        except BatchRefused as e:
            st[f"refused:{e}"] += 1
            refused.append({"row": rec, "reason": str(e)})
            continue
        key = (chain, a)
        prov = make_provenance(rec, batch_id, now_iso)
        classes = set(rec.get("classes") or [])
        if key in rows:
            row = rows[key]
            row["classes"] |= classes
            row["sources"].add(origin)
            row["origins"].add(origin)
            row["provenance"] = list(row.get("provenance") or []) + [prov]
            st["merged_into_existing"] += 1
        else:
            rows[key] = {
                "chain": chain,
                "address": a,
                "classes": classes,
                "sources": {origin},
                "origins": {origin},
                "graph_key": f"{chain}:{a}" if chain in INCLUDED else None,
                "case_broken": False,
                "case_provenance": f"attributed:{batch_id}",
                "label_addr_original": None,
                "provenance": [prov],
            }
            st["added_new"] += 1

    # append-only invariant — explicit exception, not assert (-O safe)
    if not base_keys <= set(rows):
        raise AppendOnlyViolation("base key dropped")
    if len(rows) < len(base):
        raise AppendOnlyViolation("row count shrank")

    stats = {
        "batch_rows": len(batch),
        "added_new": st["added_new"],
        "merged_into_existing": st["merged_into_existing"],
        "refused_total": len(refused),
        "refused_breakdown": {k: v for k, v in st.items() if k.startswith("refused:")},
        "refused_rows": refused,
    }
    return rows, stats


def rows_to_table(recs: list[dict]):
    """Materialise merged rows as an arrow table.

    The schema is the base schema plus the `provenance` column: a downstream
    gate reading a column that a new generation dropped fails at load time,
    so columns are carried forward even when this build does not use them.
    """
    import pyarrow as pa

    return pa.table({
        "chain": [r["chain"] for r in recs],
        "address": [r["address"] for r in recs],
        "classes": [sorted(r["classes"]) for r in recs],
        "sources": [sorted(r["sources"]) for r in recs],
        "origins": [sorted(r["origins"]) for r in recs],
        "graph_key": [r["graph_key"] for r in recs],
        "case_broken": [r["case_broken"] for r in recs],
        "case_provenance": [r["case_provenance"] for r in recs],
        "label_addr_original": [r["label_addr_original"] for r in recs],
        "in_included_5": [r["chain"] in INCLUDED for r in recs],
        "provenance": [list(r.get("provenance") or []) for r in recs],
    })


# --------------------------------------------------------------------------- #
# Object-storage build                                                        #
# --------------------------------------------------------------------------- #
def _load_base(fs) -> dict[tuple[str, str], dict]:
    import pyarrow.parquet as pq

    with fs.open_input_stream(f"{LABELS_URI}/{BASE_MANIFEST}") as f:
        man = json.loads(f.read())
    if BASE_EXPECT_ROWS is not None and man.get("rows_total") != BASE_EXPECT_ROWS:
        sys.exit(f"HALT: base rows_total {man.get('rows_total')} != {BASE_EXPECT_ROWS}")
    if BASE_EXPECT_SHA and not man.get("parquet_sha256", "").startswith(BASE_EXPECT_SHA):
        sys.exit(f"HALT: base sha {man.get('parquet_sha256','')[:20]} != {BASE_EXPECT_SHA}")

    raw = fs.open_input_file(f"{LABELS_URI}/{BASE_PARQUET}").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != man["parquet_sha256"]:
        sys.exit(f"HALT: base parquet sha mismatch {got[:20]} != manifest")
    t = pq.read_table(io.BytesIO(raw))
    base: dict[tuple[str, str], dict] = {}
    cols = {c: t.column(c).to_pylist() for c in t.column_names}
    for i in range(t.num_rows):
        ch, ad = cols["chain"][i], cols["address"][i]
        base[(ch, ad)] = {
            "chain": ch, "address": ad,
            "classes": cols["classes"][i] or [],
            "sources": cols["sources"][i] or [],
            "origins": cols["origins"][i] or [],
            "graph_key": cols["graph_key"][i],
            "case_broken": cols["case_broken"][i],
            "case_provenance": cols["case_provenance"][i],
            "label_addr_original": cols["label_addr_original"][i],
            "provenance": [],
        }
    if BASE_EXPECT_ROWS is not None and len(base) != BASE_EXPECT_ROWS:
        sys.exit(f"HALT: loaded {len(base)} rows != {BASE_EXPECT_ROWS} (duplicate keys?)")
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True, help="attribution batch JSONL")
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--min-batch-rows", type=int, default=0,
                    help="warn when the batch carries fewer rows than this")
    ap.add_argument("--execute", action="store_true",
                    help="write to object storage; default is a dry run")
    args = ap.parse_args()
    if args.execute and not LABELS_URI:
        raise SystemExit("OPENHEADS_LABELS_URI is required with --execute")

    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    # timezone.utc, not datetime.UTC: managed training images still ship
    # Python 3.10, where datetime.UTC (3.11+) does not exist.
    now_iso = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    fs = pafs.S3FileSystem(region=os.environ.get("OPENHEADS_REGION") or None)

    base = _load_base(fs)
    batch_path = Path(args.batch)
    batch = [json.loads(line)
             for line in batch_path.read_text().splitlines() if line.strip()]
    if args.min_batch_rows and len(batch) < args.min_batch_rows:
        print(f"WARNING: batch has {len(batch)} rows (< {args.min_batch_rows})",
              file=sys.stderr)

    rows, stats = merge_batch(base, batch, args.batch_id, now_iso)

    recs = sorted(rows.values(), key=lambda r: (r["chain"], r["address"]))
    table = rows_to_table(recs)

    per_chain = collections.Counter(r["chain"] for r in recs)
    per_chain_base = collections.Counter(k[0] for k in base)
    rows_with_prov = sum(1 for r in recs if r.get("provenance"))

    manifest = {
        "name": OUT_PREFIX,
        "definition": "base label set + append-only attributed batch with provenance column",
        "built_utc": now_iso,
        "base": {"rows": len(base), "sha256_prefix": BASE_EXPECT_SHA or None},
        "batch_id": args.batch_id,
        "batch": {k: v for k, v in stats.items() if k != "refused_rows"},
        "rows_total": len(recs),
        "rows_added_vs_base": len(recs) - len(base),
        "rows_with_provenance": rows_with_prov,
        "per_chain": dict(sorted(per_chain.items())),
        "diff_vs_base": {
            "rows_total": {"base": len(base), "next": len(recs),
                           "delta": len(recs) - len(base)},
            "per_chain_delta": {ch: per_chain.get(ch, 0) - per_chain_base.get(ch, 0)
                                for ch in sorted(set(per_chain) | set(per_chain_base))
                                if per_chain.get(ch, 0) != per_chain_base.get(ch, 0)},
        },
        "append_only_verified": True,
        "notes": [
            "APPEND-ONLY: every base (chain,address) key retained; rows never decrease.",
            "case-intact: case-destroyed / synthetic '::' batch rows refused, not ingested.",
            "provenance[] holds JSON entries; base rows carry [] (added only on attribution).",
        ],
    }

    print(json.dumps({k: manifest[k] for k in
                      ("rows_total", "rows_added_vs_base", "rows_with_provenance",
                       "batch", "diff_vs_base")}, indent=2, ensure_ascii=False))
    if stats["refused_rows"]:
        print(f"-- {len(stats['refused_rows'])} refused (first 5): "
              f"{json.dumps(stats['refused_rows'][:5], ensure_ascii=False)}", file=sys.stderr)

    if not args.execute:
        print("-- DRY RUN. Nothing written. Pass --execute to write.", file=sys.stderr)
        return 0

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    data = buf.getvalue()
    manifest["parquet_sha256"] = hashlib.sha256(data).hexdigest()
    manifest["parquet_bytes"] = len(data)
    out_parquet = f"{LABELS_URI}/{OUT_PREFIX}/label_set.parquet"
    with fs.open_output_stream(out_parquet) as f:
        f.write(data)
    with fs.open_output_stream(f"{LABELS_URI}/{OUT_PREFIX}/manifest.json") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=False).encode())
    print(f"WROTE {out_parquet} "
          f"({len(data)} B, sha {manifest['parquet_sha256'][:16]}) + manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
