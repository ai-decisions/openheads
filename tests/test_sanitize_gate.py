"""Tests for tools/sanitize_gate.py — the gate needs its own gate.

The gate had no test at all, and two of its failure modes were only found by
review: it silently ran the generic ruleset when no ruleset was selected
(so CI reported "clean" while checking nothing installation-specific), and it
excluded itself from the scan while holding private literals.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GATE = _ROOT / "tools" / "sanitize_gate.py"


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    full_env = dict(os.environ)
    full_env.pop("SANITIZE_PRIVATE_PATTERNS", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(_GATE), *args],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=full_env,
    )


class TestModeSelectionFailsClosed:
    def test_no_ruleset_is_an_error_not_a_clean_run(self) -> None:
        r = _run()
        assert r.returncode == 2, r.stdout
        assert "no ruleset selected" in r.stdout

    def test_public_mode_runs_the_generic_ruleset(self) -> None:
        r = _run("--public-mode")
        assert r.returncode == 0, r.stdout
        assert "[generic]" in r.stdout

    def test_empty_env_value_is_an_error(self) -> None:
        r = _run(env={"SANITIZE_PRIVATE_PATTERNS": ""})
        assert r.returncode == 2, r.stdout

    def test_missing_pattern_file_is_an_error(self) -> None:
        r = _run("--private-patterns", "/nonexistent/patterns.txt")
        assert r.returncode == 2, r.stdout
        assert "not found" in r.stdout

    def test_empty_pattern_file_is_an_error(self, tmp_path: Path) -> None:
        f = tmp_path / "p.txt"
        f.write_text("# only a comment\n")
        r = _run("--private-patterns", str(f))
        assert r.returncode == 2, r.stdout


class TestDetection:
    @pytest.fixture
    def patterns(self, tmp_path: Path) -> Path:
        f = tmp_path / "patterns.txt"
        f.write_text("test-bucket: totally-private-bucket-name\n")
        return f

    # Probe payloads are built from CHARACTER CODES. The gate scans this file
    # too, and it collapses `"a" + "b"` seams before matching, so neither a
    # literal nor a concatenated literal can live here.
    @staticmethod
    def _chars(*codes: int) -> str:
        return "".join(map(chr, codes))

    def _with_probe(self, body: str, name: str = "_probe_gate.py"):
        probe = _ROOT / "src" / "openheads" / name
        probe.write_text(body, encoding="utf-8")
        return probe

    def test_private_literal_in_a_normal_file_is_caught(self, patterns: Path) -> None:
        probe = self._with_probe('x = "totally-private-bucket-name"\n')
        try:
            r = _run("--private-patterns", str(patterns))
            assert r.returncode == 1, r.stdout
            assert "private:test-bucket" in r.stdout
        finally:
            probe.unlink()

    def test_generic_patterns_are_case_insensitive(self) -> None:
        gcs = self._chars(71, 83, 58, 47, 47) + "bucket/x"
        home = self._chars(47, 104, 111, 109, 101, 47) + "someone/creds"
        probe = self._with_probe(f'a = "{gcs}"\nb = "{home}"\n')
        try:
            r = _run("--public-mode")
            assert r.returncode == 1, r.stdout
            assert "gcs-uri" in r.stdout and "home-path" in r.stdout
        finally:
            probe.unlink()

    def test_internal_qa_note_is_caught(self) -> None:
        note = self._chars(91) + self._chars(97, 100, 100, 114, 101, 115, 115) + self._chars(
            32, 98, 108, 97, 110, 107, 101, 100
        ) + " 2026-04-23: not found at cited URL]"
        probe = self._with_probe(f'd = "{note}"\n')
        try:
            r = _run("--public-mode")
            assert r.returncode == 1, r.stdout
            assert "internal-qa-note" in r.stdout
        finally:
            probe.unlink()

    def test_iso_dates_are_not_false_positives(self) -> None:
        probe = self._with_probe('TRAIN_END = "2024-07-01"\ndates = ["2020-01-01"]\n')
        try:
            r = _run("--public-mode")
            assert r.returncode == 0, r.stdout
        finally:
            probe.unlink()

    def test_undecodable_byte_does_not_hide_a_literal(self, patterns: Path) -> None:
        probe = _ROOT / "src" / "openheads" / "_probe_bytes.py"
        probe.write_bytes(b'x = "totally-private-bucket-name"  # caf\xe9\n')
        try:
            r = _run("--private-patterns", str(patterns))
            assert r.returncode == 1, r.stdout
        finally:
            probe.unlink()

    def test_dangling_symlink_with_restricted_name_is_caught(self) -> None:
        link = _ROOT / "src" / "openheads" / "policy.pt"
        link.symlink_to("/nonexistent/target")
        try:
            r = _run("--public-mode")
            assert r.returncode == 1, r.stdout
            assert "restricted-data-file" in r.stdout
        finally:
            link.unlink()

    def test_the_gate_itself_is_scanned_for_private_literals(self, tmp_path: Path) -> None:
        """The gate excludes itself from the generic scan — it must NOT be
        exempt from the private ruleset, which is exactly how a private
        inventory ended up living inside it."""
        f = tmp_path / "patterns.txt"
        # A literal that genuinely appears in the gate's own source.
        f.write_text("gate-self: RESTRICTED_FILENAME\n")
        r = _run("--private-patterns", str(f))
        assert r.returncode == 1, r.stdout
        assert "tools/sanitize_gate.py" in r.stdout
