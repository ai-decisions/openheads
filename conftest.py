"""Repo-root pytest configuration.

- No sys.path hacks: `tests` is a package under the rootdir that pytest
  inserts, `openheads` itself is installed (`pip install -e .`), and the
  scripts are loaded by path where a test needs one — so nothing here has
  to reach into the filesystem to make imports work.
- No environment defaults are exported here, on purpose: warehouse
  configuration is fail-closed (no default bucket or prefix anywhere in
  the tree), and the test suite runs entirely on synthetic or
  downloaded-public inputs. If a test ever needs an `OPENHEADS_*`
  variable, it must set it locally in that test — a repo-wide default in
  this file would mask exactly the fail-closed behaviour the code is
  required to have.
"""
