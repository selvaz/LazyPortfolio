"""Thin pytest wrapper around the behavioral Tree Studio GUI contract tests.

The real assertions live in `tests/js/tree_studio_v2_contract.test.mjs`
(Node's built-in test runner + jsdom, driving the actual `renderNode()`/
`applyNodeForm()` functions from `project/tree_studio.html` - not a
reimplementation of the app's logic, and not mere string-presence checks
like the pre-existing `tests/test_tree_studio_financing_ui.py`).

Skips gracefully (matching this repo's existing importorskip-guarded
ecosystem-test convention) when the Node/npm toolchain for this optional
front-end test suite isn't set up, e.g. `npm install` was never run. CI runs
it for real in a dedicated job with Node.js set up (see `.github/workflows/ci.yml`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _npm_test_runnable() -> bool:
    return shutil.which("npm") is not None and (REPO_ROOT / "node_modules").is_dir()


@pytest.mark.skipif(
    not _npm_test_runnable(),
    reason="npm/node_modules not set up for the Tree Studio JS test suite; run `npm install` first",
)
def test_tree_studio_gui_contract_suite() -> None:
    result = subprocess.run(
        "npm test --silent",
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        shell=True,
    )
    assert result.returncode == 0, (
        f"Tree Studio JS contract tests failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
