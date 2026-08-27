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
    """Probes live in a THROWAWAY tree scanned via --root — never inside
    src/. A probe planted in the package and cleaned in `finally` survives an
    aborted run (Ctrl-C, OOM) and then fails every later gate run; writing
    into src/ also breaks read-only checkouts and parallel test runs."""

    @pytest.fixture
    def patterns(self, tmp_path: Path) -> Path:
        f = tmp_path / "patterns.txt"
        f.write_text("test-bucket: totally-private-bucket-name\n")
        return f

    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        t = tmp_path / "tree"
        t.mkdir()
        (t / "clean.py").write_text('GREETING = "hello"\n')
        return t

    # Probe payloads are built from CHARACTER CODES. The gate scans this file
    # too, and it collapses `"a" + "b"` seams before matching, so neither a
    # literal nor a concatenated literal can live here.
    @staticmethod
    def _chars(*codes: int) -> str:
        return "".join(map(chr, codes))

    def test_private_literal_in_a_normal_file_is_caught(
        self, patterns: Path, tree: Path
    ) -> None:
        (tree / "probe.py").write_text('x = "totally-private-bucket-name"\n')
        r = _run("--private-patterns", str(patterns), "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "private:test-bucket" in r.stdout

    def test_generic_patterns_are_case_insensitive(self, tree: Path) -> None:
        gcs = self._chars(71, 83, 58, 47, 47) + "bucket/x"
        home = self._chars(47, 104, 111, 109, 101, 47) + "someone/creds"
        (tree / "probe.py").write_text(f'a = "{gcs}"\nb = "{home}"\n')
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "gcs-uri" in r.stdout and "home-path" in r.stdout

    def test_internal_qa_note_is_caught(self, tree: Path) -> None:
        note = self._chars(91) + self._chars(97, 100, 100, 114, 101, 115, 115) + self._chars(
            32, 98, 108, 97, 110, 107, 101, 100
        ) + " 2026-04-23: not found at cited URL]"
        (tree / "probe.py").write_text(f'd = "{note}"\n')
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "internal-qa-note" in r.stdout

    def test_iso_dates_are_not_false_positives(self, tree: Path) -> None:
        (tree / "probe.py").write_text('TRAIN_END = "2024-07-01"\ndates = ["2020-01-01"]\n')
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 0, r.stdout

    def test_undecodable_byte_does_not_hide_a_literal(
        self, patterns: Path, tree: Path
    ) -> None:
        (tree / "probe.py").write_bytes(b'x = "totally-private-bucket-name"  # caf\xe9\n')
        r = _run("--private-patterns", str(patterns), "--root", str(tree))
        assert r.returncode == 1, r.stdout

    def test_dangling_symlink_with_restricted_name_is_caught(self, tree: Path) -> None:
        (tree / "policy.pt").symlink_to("/nonexistent/target")
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "restricted-data-file" in r.stdout

    def test_weight_file_flagged_by_name_not_decoded(self, tree: Path) -> None:
        """A weight file is a violation by NAME; its bytes must not also be
        decoded as text — that buried the verdict under home-path noise from
        pickled payloads."""
        home = self._chars(47, 85, 115, 101, 114, 115, 47)  # /Users/ inside bytes
        (tree / "model.pt").write_bytes(f"garbage {home}someone".encode())
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "restricted-data-file" in r.stdout
        assert "home-path" not in r.stdout

    def test_service_dir_with_files_fails_empty_passes(self, tree: Path) -> None:
        """A file-bearing agent-config dir must fail a published-tree scan (a
        flip done by COPYING a working dir would ship it); an empty
        bookkeeping dir the harness drops everywhere is not a violation."""
        svc = tree / ".claude"
        svc.mkdir()
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 0, r.stdout
        (svc / "settings.json").write_text("{}\n")
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "service-directory" in r.stdout

    def test_service_dir_nested_and_file_form_are_caught(self, tree: Path) -> None:
        """Two bypasses a top-level-only check allowed: a service dir nested
        below the root, and a plain FILE named like the directory."""
        nested = tree / "src" / "x" / ".claude"
        nested.mkdir(parents=True)
        (nested / "settings.json").write_text("{}\n")
        r = _run("--public-mode", "--root", str(tree))
        assert r.returncode == 1, r.stdout
        assert "service-directory" in r.stdout

        flat = tree / "flat"
        flat.mkdir()
        (flat / "ok.py").write_text("x = 1\n")
        (flat / ".claude").write_text("config-as-a-file\n")
        r = _run("--public-mode", "--root", str(flat))
        assert r.returncode == 1, r.stdout
        assert "service-directory" in r.stdout

    def test_empty_root_is_an_error_not_clean(self, tmp_path: Path) -> None:
        """`clean (0 files scanned)` is how a mistyped --root reads as a
        pass; an empty scan must refuse instead."""
        empty = tmp_path / "empty"
        empty.mkdir()
        r = _run("--public-mode", "--root", str(empty))
        assert r.returncode == 2, r.stdout
        assert "nothing to scan" in r.stdout

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
