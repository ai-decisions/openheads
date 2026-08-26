"""Tests for label mapping."""

import csv
import tempfile

from openheads.label_mapping import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    load_known_labels,
    load_ofac_labels,
    merge_labels,
    propagate_labels_1hop,
)

# Synthetic addresses (repeating hex patterns), never real parties: this
# module refuses to ship attributions about real entities without a source.
SYNTHETIC_LICIT = "0x1111111111111111111111111111111111111111"
SYNTHETIC_ILLICIT = "0x2222222222222222222222222222222222222222"


def _write_csv(rows: list[list[str]]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        csv.writer(f).writerows(rows)
        return f.name


def test_load_known_labels_reads_words_and_integers():
    path = _write_csv([
        [SYNTHETIC_LICIT, "licit"],
        [SYNTHETIC_ILLICIT, str(LABEL_ILLICIT)],
    ])
    labels = load_known_labels(path)
    assert labels == {SYNTHETIC_LICIT: LABEL_LICIT, SYNTHETIC_ILLICIT: LABEL_ILLICIT}


def test_load_known_labels_skips_unparseable_rows():
    """A bad row is dropped, not coerced: silent coercion loses training signal."""
    path = _write_csv([
        [SYNTHETIC_LICIT, "licti"],           # typo, not a label
        ["not-an-address", "licit"],          # not an address
        [SYNTHETIC_ILLICIT],                  # missing the label column
        [SYNTHETIC_ILLICIT, "illicit"],       # the only valid row
    ])
    labels = load_known_labels(path)
    assert labels == {SYNTHETIC_ILLICIT: LABEL_ILLICIT}


def test_load_known_labels_missing_file_is_empty_not_fatal():
    assert load_known_labels("/nonexistent/entities.csv") == {}


def test_load_ofac_labels():
    # The address below is synthetic fixture data (repeating hex pattern),
    # not a real sanctioned address.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "info"])
        writer.writerow(
            [
                "1",
                "Bad Actor",
                "Digital Currency Address 0xabcdef1234567890abcdef1234567890abcdef12",
            ]
        )
        writer.writerow(["2", "Clean Person", "No crypto here"])
        path = f.name

    labels = load_ofac_labels(path)
    assert "0xabcdef1234567890abcdef1234567890abcdef12" in labels
    assert labels["0xabcdef1234567890abcdef1234567890abcdef12"] == LABEL_ILLICIT
    assert len(labels) == 1


def test_merge_labels_illicit_priority():
    a = {"0xaaa": LABEL_LICIT}
    b = {"0xaaa": LABEL_ILLICIT}
    merged = merge_labels(a, b)
    assert merged["0xaaa"] == LABEL_ILLICIT


def test_propagate_labels_1hop():
    labels = {"0xbad": LABEL_ILLICIT}
    edges = [("0xbad", "0xneighbor"), ("0xclean", "0xother")]
    scores = propagate_labels_1hop(labels, edges)
    assert "0xneighbor" in scores
    assert scores["0xneighbor"] == 0.5
    assert "0xother" not in scores
