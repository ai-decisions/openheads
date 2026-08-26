"""Label-set ingestion pipeline: append-only merge with a provenance column.

Tests the pure `merge_batch` in scripts/build_label_set.py: every base key
survives, attributed rows carry provenance, and case-destroyed / synthetic /
provenance-less batch rows are refused (never ingested)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from openheads.tron_address import base58check_to_hex

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_label_set", ROOT / "scripts" / "build_label_set.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
merge_batch = _mod.merge_batch
validate_batch_row = _mod.validate_batch_row
rows_to_table = _mod.rows_to_table
BatchRefused = _mod.BatchRefused

# SYNTHETIC tron address: base58check of the mainnet prefix byte (0x41)
# followed by an all-zero 20-byte payload. It decodes to
# 0x0000000000000000000000000000000000000000 — not a real wallet, safe to
# tag with any fixture class below.
TRON_B58 = "T9yD14Nj9j7xAB4dbGeiX9h8unkKHxuWwb"
TRON_HEX = base58check_to_hex(TRON_B58)

NOW = "2026-08-17T17:00:00+00:00"
BID = "test-batch-01"

# genesis P2PKH (case intact) and its case-destroyed lowercase form
BTC_OK = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
BTC_DESTROYED = "1a1zp1ep5qgefi2dmptftl5slmv7divfna"
ETH_EXIST = "0x1111111111111111111111111111111111111111"
ETH_NEW = "0x2222222222222222222222222222222222222222"


def base_rows() -> dict:
    return {
        ("eth", ETH_EXIST): {
            "chain": "eth",
            "address": ETH_EXIST,
            "classes": ["vasp"],
            "sources": ["seed_source"],
            "origins": ["seed_source"],
            "graph_key": f"eth:{ETH_EXIST}",
            "case_broken": False,
            "case_provenance": None,
            "label_addr_original": None,
            "provenance": [],
        },
        ("btc", BTC_OK): {
            "chain": "btc",
            "address": BTC_OK,
            "classes": ["exchange"],
            "sources": ["licit_source"],
            "origins": ["licit_source"],
            "graph_key": None,
            "case_broken": False,
            "case_provenance": None,
            "label_addr_original": None,
            "provenance": [],
        },
    }


def _prov(**kw):
    d = {
        "anchor_type": "ofac",
        "method": "sdn_xml",
        "source_url": "https://sanctionslistservice.ofac.treas.gov",
        "source_date": "2026-08-01",
    }
    d.update(kw)
    return d


def test_append_only_and_provenance():
    base = base_rows()
    batch = [
        # new address, attributed
        {"chain": "eth", "address": ETH_NEW, "classes": ["mixer"], **_prov(anchor_type="onchain")},
        # merge into existing base row (union classes + append provenance)
        {
            "chain": "eth",
            "address": ETH_EXIST.upper().replace("0X", "0x"),
            "classes": ["sanctioned"],
            **_prov(anchor_type="ofac"),
        },
    ]
    rows, stats = merge_batch(base, batch, BID, NOW)

    # append-only: both base keys survive
    assert ("eth", ETH_EXIST) in rows
    assert ("btc", BTC_OK) in rows
    assert len(rows) >= len(base)

    # new row added with provenance
    assert stats["added_new"] == 1
    new = rows[("eth", ETH_NEW)]
    assert new["classes"] == {"mixer"}
    assert len(new["provenance"]) == 1
    p = json.loads(new["provenance"][0])
    assert p["batch_id"] == BID and p["anchor_type"] == "onchain"
    assert new["origins"] == {f"attribution_batch:{BID}"}

    # merged row: classes unioned, provenance appended, base row untouched otherwise
    assert stats["merged_into_existing"] == 1
    merged = rows[("eth", ETH_EXIST)]
    assert merged["classes"] == {"vasp", "sanctioned"}
    assert "seed_source" in merged["sources"]
    assert f"attribution_batch:{BID}" in merged["origins"]
    assert len(merged["provenance"]) == 1

    # untouched base row keeps empty provenance
    assert rows[("btc", BTC_OK)]["provenance"] == []


def test_case_destroyed_refused():
    with pytest.raises(BatchRefused):
        validate_batch_row(
            {"chain": "btc", "address": BTC_DESTROYED, "classes": ["exchange"], **_prov()}
        )


def test_synthetic_key_refused():
    with pytest.raises(BatchRefused):
        validate_batch_row(
            {"chain": "multi", "address": "registry::license::123", "classes": ["vasp"], **_prov()}
        )


def test_missing_provenance_refused():
    with pytest.raises(BatchRefused):
        validate_batch_row(
            {"chain": "eth", "address": ETH_NEW, "classes": ["mixer"], "anchor_type": "onchain"}
        )  # no source_url/source_date


def test_tron_t_form_normalized_to_hex():
    """A base58 T-form tron batch address must land on the hex key — the
    key-symmetry class: a ('tron','T…') key would be unreachable by lookups
    that use the hex key form."""
    chain, a = validate_batch_row(
        {"chain": "tron", "address": TRON_B58, "classes": ["sanctioned"], **_prov()}
    )
    assert (chain, a) == ("tron", TRON_HEX)

    # and it MERGES with an existing hex-form row instead of duplicating
    base = base_rows()
    base[("tron", TRON_HEX)] = {
        "chain": "tron",
        "address": TRON_HEX,
        "classes": ["sanctioned"],
        "sources": ["sanctions_source"],
        "origins": ["sanctions_source"],
        "graph_key": f"tron:{TRON_HEX}",
        "case_broken": False,
        "case_provenance": None,
        "label_addr_original": None,
        "provenance": [],
    }
    batch = [{"chain": "tron", "address": TRON_B58, "classes": ["illicit"], **_prov()}]
    rows, stats = merge_batch(base, batch, BID, NOW)
    assert stats["added_new"] == 0 and stats["merged_into_existing"] == 1
    assert ("tron", TRON_B58) not in rows
    assert rows[("tron", TRON_HEX)]["classes"] == {"sanctioned", "illicit"}
    assert len(rows[("tron", TRON_HEX)]["provenance"]) == 1


def test_invalid_tron_base58_refused():
    bad = "T" + "1" * 33  # right shape, invalid base58check checksum
    with pytest.raises(BatchRefused):
        validate_batch_row({"chain": "tron", "address": bad, "classes": ["x"], **_prov()})


def test_rows_to_table_keeps_base_schema_plus_provenance():
    """The output must not silently drop the base `in_included_5` column
    (downstream gates read it)."""
    base = base_rows()
    batch = [
        {"chain": "eth", "address": ETH_NEW, "classes": ["mixer"], **_prov(anchor_type="onchain")}
    ]
    rows, _ = merge_batch(base, batch, BID, NOW)
    table = rows_to_table(sorted(rows.values(), key=lambda r: (r["chain"], r["address"])))
    assert table.column_names == [
        "chain",
        "address",
        "classes",
        "sources",
        "origins",
        "graph_key",
        "case_broken",
        "case_provenance",
        "label_addr_original",
        "in_included_5",
        "provenance",
    ]
    by_key = {
        (c, a): i
        for i, (c, a) in enumerate(
            zip(
                table.column("chain").to_pylist(),
                table.column("address").to_pylist(),
                strict=True,
            )
        )
    }
    inc = table.column("in_included_5").to_pylist()
    assert inc[by_key[("eth", ETH_NEW)]] is True
    assert inc[by_key[("btc", BTC_OK)]] is False


def test_refused_rows_not_ingested():
    base = base_rows()
    batch = [
        {"chain": "btc", "address": BTC_DESTROYED, "classes": ["exchange"], **_prov()},
        {"chain": "multi", "address": "x::y", "classes": ["vasp"], **_prov()},
        {"chain": "eth", "address": ETH_NEW, "classes": ["mixer"], "anchor_type": "onchain"},
    ]
    rows, stats = merge_batch(base, batch, BID, NOW)
    assert stats["added_new"] == 0
    assert stats["merged_into_existing"] == 0
    assert stats["refused_total"] == 3
    assert len(rows) == len(base)  # nothing ingested
    # the case-destroyed string never entered the set
    assert ("btc", BTC_DESTROYED) not in rows
