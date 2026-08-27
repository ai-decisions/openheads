#!/usr/bin/env python3
"""Sanitize gate — fails the build if internal identifiers, secrets, or
restricted data ever land in this repository.

Checks every text file in the tree (excluding .git and this file) for:
  1. secret-looking material (AWS access key IDs, private key blocks,
     tokens, non-empty string literals assigned to *_KEY / *_TOKEN /
     *_SECRET);
  2. storage URIs and machine-local paths that should never appear in a
     public tree (s3:// / gs:// URIs, /Users/... home paths, local://
     pseudo-URIs);
  3. internal project markers (Cyrillic text, sprint/task/contract
     markers, internal module or warehouse names);
  4. restricted data files (model weights, proprietary label caches,
     OpenSanctions- or Spellbook-derived datasets) anywhere in the tree.

Installation-specific private literals (bucket names, warehouse
identifiers, private repo URLs) are deliberately NOT stored in this
file: a public checker must not double as an inventory of the private
values it guards against. Maintainers run the full gate before every
push by supplying those literals from a file OUTSIDE the repository:

    SANITIZE_PRIVATE_PATTERNS=/path/to/private_patterns.txt \
        python3 tools/sanitize_gate.py
    # or: python3 tools/sanitize_gate.py --private-patterns FILE

Pattern-file format: one `name: literal` per line; blank lines and
lines starting with '#' are ignored. Literals are matched verbatim
(regex-escaped). If the variable/flag points to a missing or empty
file the gate fails closed (exit 2).

MODE SELECTION IS FAIL-CLOSED. Running with no ruleset is an error
(exit 2), because a run that silently skips the private literals looks
identical to a clean run. A public clone has no pattern file and must
say so explicitly:

    python3 tools/sanitize_gate.py --public-mode

THIS FILE IS CANONICAL AND SHARED across the published repositories;
do not edit a repository's copy in place — edit the canon and roll it
out. Repository-specific differences live in an OPTIONAL sibling file
`sanitize_overrides.py` — literal data only, ENFORCED by an ast-based
loader (no exec; ASCII-only; secret-class patterns not overridable):

    DISABLED_PATTERNS: set[str]        # generic pattern names to skip
    PATTERN_EXEMPT: dict[str, set[str]]  # pattern name -> posix rel paths
    EXTRA_RESTRICTED_FILENAME: str     # extra regex OR-ed into the name check

Exit 0 = clean; exit 1 = violations printed, one per line; exit 2 =
the gate could not run with a defined ruleset.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
OVERRIDES_PATH = SELF.parent / "sanitize_overrides.py"

# Suffixes whose CONTENT is never scanned, because they are binary by
# definition. Everything else is read and scanned — an allow-list of "text"
# extensions meant a literal inside .log, .ipynb, .tex or .env was invisible
# while the file still counted as scanned. The weight/pickle family is here
# too: those files are ALREADY violations by name (RESTRICTED_FILENAME), and
# decoding megabytes of pickle as UTF-8 only buries that verdict in noise.
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp4",
    ".parquet",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npz",
    ".npy",
    ".pkl",
    ".pickle",
    ".joblib",
    ".h5",
    ".onnx",
}

# Agent/tool configuration directories. They live legitimately in a private
# working tree (and are .gitignore'd), but a tree that CLAIMS to be the
# publishable artefact (--public-mode) must not contain them: a flip done by
# copying a working directory instead of `git archive` would ship them.
SERVICE_DIRS = {".claude", ".codex", ".hoff_tmp"}

CONTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gcs-uri", re.compile(r"gs://", re.IGNORECASE)),
    ("s3-bucket-uri", re.compile(r"s3://[a-z0-9][a-z0-9.-]{2,}", re.IGNORECASE)),
    ("local-uri", re.compile(r"local://", re.IGNORECASE)),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key-block", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
    (
        "secret-literal",
        re.compile(r"""(?:_KEY|_TOKEN|_SECRET|_PASSWORD)\s*=\s*["'][^"']{8,}["']"""),
    ),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("cyrillic", re.compile(r"[\u0400-\u04ff]")),
    ("sprint-marker", re.compile(r"\bSprint[\s-]*\d", re.IGNORECASE)),
    ("founder-marker", re.compile(r"\bfounder\b", re.IGNORECASE)),
    ("contract-marker", re.compile(r"\b(addendum-\d+|earn-it|non-goal \d)\b", re.IGNORECASE)),
    ("memory-file-marker", re.compile(r"\bfeedback_[a-z_]{6,}\b")),
    ("home-path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]")),
    (
        # TD-\d+ alone missed the LIVE naming convention (TD-Q, TD-AB,
        # TD-GATE-SYNC): the debt registry moved to letter IDs long ago.
        "internal-task-marker",
        re.compile(
            r"\b(?:TD-[A-Z0-9][A-Z0-9-]{0,24}\b|T2\.[ABC]\b|[Tt]ask\s+\d+\.\d+[a-z]?)",
            re.IGNORECASE,
        ),
    ),
    ("internal-tasks-path", re.compile(r"tasks/(?:contracts|qa|handoff|lessons)")),
    # Internal QA annotations of the form "[address blanked 2026-04-23: ...]"
    # shipped inside data descriptions and, worse, inside module literals.
    ("internal-qa-note", re.compile(r"\[(?:[^\]\n]*\b20\d\d-\d\d-\d\d\s*:|address blanked)")),
    ("private-module-marker", re.compile(r"\babm_v2\b")),
    ("private-stack-marker", re.compile(r"google\.cloud|\bBig ?Query\b", re.IGNORECASE)),
]

# Weights and pickled state, by every extension they actually ship under.
# Covering only .pt/.pth left .ckpt (what Lightning writes by default),
# .safetensors, .bin and the pickle family passing as clean. The vasps_
# dated-dump pattern is canonical (raised from the openlabels copy): no
# published tree legitimately ships one.
RESTRICTED_FILENAME_SRC = (
    r"(opensanctions|spellbook|labels_cache|probe|vasps_\d{4}-\d{2}-\d{2})"
    r".*\.(json|csv|parquet|ndjson)$"
    r"|\.(pt|pth|ckpt|safetensors|bin|npz|npy|pkl|pickle|joblib|h5|onnx)$"
)


# Patterns that guard secret-class material can never be disabled or
# exempted by a repo override: an overrides commit is the one file the
# generic scan does not read, so it must not be able to switch these off.
NON_OVERRIDABLE = {
    "aws-access-key-id",
    "private-key-block",
    "secret-literal",
    "github-token",
}
_EXEMPT_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def _overrides_fail(msg: str) -> None:
    print(f"SANITIZE GATE: overrides rejected — {msg}")
    raise SystemExit(2)


def load_overrides() -> tuple[set[str], dict[str, set[str]], re.Pattern[str], str]:
    """Load the optional per-repo overrides file.

    The file is DATA, and that is enforced, not assumed: it is parsed with
    `ast` and evaluated with `ast.literal_eval` — module-level assignments of
    the three known names plus a docstring, nothing else. Executable content
    of any kind fails closed (exit 2), because this is the one file the
    generic content scan does not read. Non-ASCII anywhere in the file fails
    closed for the same reason.
    """
    disabled: set[str] = set()
    exempt: dict[str, set[str]] = {}
    restricted_src = RESTRICTED_FILENAME_SRC
    extra = ""
    if OVERRIDES_PATH.is_file():
        raw = OVERRIDES_PATH.read_text(encoding="utf-8")
        bad = [c for c in raw if ord(c) > 126 and c not in "\n\t"]
        if bad:
            _overrides_fail(f"non-ASCII character {bad[0]!r} (unscanned file must stay ASCII)")
        try:
            tree = ast.parse(raw)
        except SyntaxError as e:
            _overrides_fail(f"not parseable: {e}")
        known_names = {"DISABLED_PATTERNS", "PATTERN_EXEMPT", "EXTRA_RESTRICTED_FILENAME"}
        values: dict[str, object] = {}
        for i, node in enumerate(tree.body):
            if (
                i == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue  # module docstring
            if (
                not isinstance(node, ast.Assign)
                or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
            ):
                _overrides_fail(
                    f"statement {type(node).__name__} at line {node.lineno} "
                    "is not a plain assignment"
                )
            name = node.targets[0].id
            if name not in known_names:
                _overrides_fail(f"unknown name {name!r} at line {node.lineno}")
            try:
                values[name] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                _overrides_fail(f"{name} at line {node.lineno} is not a literal")

        disabled = {str(x) for x in values.get("DISABLED_PATTERNS", set())}
        exempt = {
            str(name): {str(p) for p in paths}
            for name, paths in dict(values.get("PATTERN_EXEMPT", {})).items()
        }
        extra = str(values.get("EXTRA_RESTRICTED_FILENAME", ""))
        if extra:
            try:
                re.compile(extra)
            except re.error as e:
                _overrides_fail(f"EXTRA_RESTRICTED_FILENAME does not compile: {e}")
            restricted_src = f"{restricted_src}|{extra}"
        known = {n for n, _ in CONTENT_PATTERNS}
        unknown = (disabled | set(exempt)) - known
        if unknown:
            _overrides_fail(f"unknown pattern names: {sorted(unknown)}")
        untouchable = (disabled | set(exempt)) & NON_OVERRIDABLE
        if untouchable:
            _overrides_fail(f"secret-class patterns cannot be overridden: {sorted(untouchable)}")
        for name, paths in exempt.items():
            for p in paths:
                if not _EXEMPT_PATH_RE.fullmatch(p):
                    _overrides_fail(f"exempt path {p!r} for {name} is not a plain rel path")
    return disabled, exempt, re.compile(restricted_src, re.IGNORECASE), extra


def load_private_patterns(path: str) -> list[tuple[str, re.Pattern[str]]]:
    """Load `name: literal` lines; fail closed on a missing/empty file."""
    p = Path(path)
    if not p.is_file():
        print(f"SANITIZE GATE: private-patterns file not found: {path}")
        raise SystemExit(2)
    out: list[tuple[str, re.Pattern[str]]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, literal = line.partition(":")
        name, literal = name.strip(), literal.strip()
        if not name or not literal:
            print(f"SANITIZE GATE: malformed private-pattern line: {raw!r}")
            raise SystemExit(2)
        out.append((f"private:{name}", re.compile(re.escape(literal), re.IGNORECASE)))
    if not out:
        print(f"SANITIZE GATE: private-patterns file is empty: {path}")
        raise SystemExit(2)
    return out


# Two seams, both of which produce ONE string at runtime:
#   "my-priv" + "ate-bucket"      explicit concatenation
#   "my-priv" "ate-bucket"        adjacent literals, joined by the parser
# The second needs no operator at all and may straddle lines inside
# parentheses, which is why the whitespace class includes newlines.
_CONCAT_JOIN = re.compile(r"""["'](?:\s*\+\s*|\s+)["']""", re.DOTALL)


def deobfuscate(text: str) -> str:
    """Collapse Python string-concatenation seams.

    A literal split as ``"my-priv" + "ate-bucket"`` — or, with no operator at
    all, ``"my-priv" "ate-bucket"`` — reads as the joined value to any human
    and to `git show`, but not to a verbatim scan. Both forms are collapsed
    before matching, and the second is the more dangerous one precisely
    because it looks like two harmless fragments.

    Scope: seams only. A value assembled by ``"".join(...)``, ``chr()`` or
    base64 still passes, so this lowers the odds of an ACCIDENT, not of
    intent.
    """
    return _CONCAT_JOIN.sub("", text)


def iter_files(root: Path = ROOT) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        parts = rel.parts
        # build/, dist/ and *.egg-info hold COPIES of the tree: skipping them
        # meant a literal could sit in a staged artefact and pass the gate.
        # They are scanned; only VCS and tool caches are skipped. The skip
        # applies BEFORE the symlink branch so cache-dir symlinks stay out.
        if parts[0] in {".git", ".venv", ".ruff_cache", ".pytest_cache"} or "__pycache__" in parts:
            continue
        if p.is_symlink():
            # A dangling symlink is not is_file(), so it would skip even the
            # restricted-filename check. Keep it in the list; the content
            # read below turns its OSError into a violation.
            out.append(p)
            continue
        if not p.is_file():
            continue
        if p.resolve() in (SELF, OVERRIDES_PATH):
            # Excluded from the GENERIC content scan only: the pattern
            # definitions (and the override names/paths that quote them)
            # would match their own regex sources. Private literals are
            # still checked against both files in main(); the overrides file
            # is additionally forced to be ASCII-only literal data by
            # load_overrides().
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--private-patterns", default=os.environ.get("SANITIZE_PRIVATE_PATTERNS"))
    ap.add_argument(
        "--public-mode",
        action="store_true",
        help="run the generic ruleset only; required to acknowledge that no "
        "installation-specific literals are being checked",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="tree to scan (default: this repository). Lets tests gate a "
        "fixture tree instead of planting probe files inside src/",
    )
    args = ap.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT

    env_set = "SANITIZE_PRIVATE_PATTERNS" in os.environ
    if args.private_patterns is None and env_set:
        # Set-but-empty must not silently downgrade to the generic ruleset.
        print("SANITIZE GATE: SANITIZE_PRIVATE_PATTERNS is set but empty")
        return 2
    if args.private_patterns is not None and not str(args.private_patterns).strip():
        print("SANITIZE GATE: --private-patterns given an empty value")
        return 2
    if not args.private_patterns and not args.public_mode:
        print(
            "SANITIZE GATE: no ruleset selected. Pass --private-patterns FILE "
            "(or SANITIZE_PRIVATE_PATTERNS) to check installation-specific "
            "literals, or --public-mode to acknowledge the generic-only run."
        )
        return 2

    disabled, exempt, restricted_filename, extra_restricted = load_overrides()
    patterns = [(n, p) for n, p in CONTENT_PATTERNS if n not in disabled]
    private_patterns: list[tuple[str, re.Pattern[str]]] = []
    mode = "generic"
    if disabled or exempt or extra_restricted:
        # Overrides must be visible in every run: a silently narrowed ruleset
        # reads identical to the full one.
        print(
            "SANITIZE GATE: repo overrides active — "
            f"disabled={sorted(disabled)} exempt={sorted(exempt)} "
            f"extra_restricted={extra_restricted!r}"
        )
    if args.private_patterns:
        private_patterns = load_private_patterns(args.private_patterns)
        patterns += private_patterns
        mode = "generic+private"

    violations: list[str] = []
    # A published artefact must not carry agent/tool config FILES — at ANY
    # depth, and a plain file named like the directory counts too. Only
    # file-bearing paths are flagged: the agent harness drops empty
    # bookkeeping dirs into any working tree, and an empty directory ships
    # nothing (git cannot even represent one).
    scan_list = iter_files(root)
    service_hits = sorted(
        {
            str(p.relative_to(root))
            for p in scan_list
            if p.name in SERVICE_DIRS or SERVICE_DIRS & set(p.relative_to(root).parts[:-1])
        }
    )
    for hit in service_hits:
        violations.append(
            f"{hit}: service-directory (agent/tool config must not " "ship in a published tree)"
        )
    for p in scan_list:
        rel = p.relative_to(root)
        if restricted_filename.search(p.name):
            violations.append(f"{rel}: restricted-data-file (name pattern)")
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            # errors="replace": a single stray byte must not drop the whole
            # file from the scan while still counting as "scanned".
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            # A file the gate cannot read is a file the gate cannot clear —
            # silently skipping it counted as "scanned" (fail-open).
            violations.append(f"{rel}: unreadable-file ({e.__class__.__name__})")
            continue
        haystacks = [text]
        collapsed = deobfuscate(text)
        if collapsed != text:
            haystacks.append(collapsed)
        for name, pat in patterns:
            if rel.as_posix() in exempt.get(name, ()):
                continue
            for i, hay in enumerate(haystacks):
                hits = list(pat.finditer(hay))
                if not hits:
                    continue
                suffix = "" if i == 0 else " (concat-obfuscated)"
                for m in hits:
                    line = hay.count("\n", 0, m.start()) + 1
                    violations.append(f"{rel}:{line}: {name}{suffix}: {m.group(0)[:60]!r}")
                break

    if private_patterns:
        # The gate excludes itself (and the overrides file) from the generic
        # scan; both must still be checked for the private literals this gate
        # exists to keep out (the exact defect that put a private inventory
        # in this file).
        for cfg in (SELF, OVERRIDES_PATH):
            if not cfg.is_file():
                continue
            cfg_text = deobfuscate(cfg.read_text(encoding="utf-8", errors="replace"))
            rel_cfg = cfg.relative_to(ROOT) if cfg.is_relative_to(ROOT) else cfg
            for name, pat in private_patterns:
                for m in pat.finditer(cfg_text):
                    line = cfg_text.count("\n", 0, m.start()) + 1
                    violations.append(f"{rel_cfg}:{line}: {name}: {m.group(0)[:60]!r}")

    if violations:
        print(f"SANITIZE GATE: {len(violations)} violation(s) [{mode}]")
        for v in violations:
            print(" ", v)
        return 1
    if not scan_list:
        # "clean (0 files scanned)" is how a mistyped --root reads as a pass.
        print(f"SANITIZE GATE: nothing to scan under {root} — wrong --root?")
        return 2
    print(f"SANITIZE GATE: clean ({len(scan_list)} files scanned) [{mode}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
