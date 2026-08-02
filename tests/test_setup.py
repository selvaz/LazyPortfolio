"""Regression tests for lazyportfolio-setup's market-data-hub install logic.

Covers the exact gap flagged in review: the no-sibling-checkout fallback
must install the datacore extra's pinned revision (never an unpinned URL),
and must fail loudly rather than silently degrading if that pin is ever
unavailable.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyportfolio._setup import (
    MARKET_DATA_HUB_GIT_URL,
    _ask_optional_path,
    _extra_requirements,
    _install_market_data_hub,
    _installed_via_main_branch,
    _remote_main_commit,
)


def _fake_direct_url_dist(vcs_info: dict | None):
    """A stand-in for importlib.metadata.distribution()'s return value,
    with just enough surface (.read_text) for _installed_via_main_branch."""
    import json as _json

    dist = MagicMock()
    dist.read_text.return_value = None if vcs_info is None else _json.dumps({"vcs_info": vcs_info})
    return dist


def test_extra_requirements_datacore_returns_the_pinned_direct_reference() -> None:
    specs = _extra_requirements("datacore")
    assert len(specs) == 1
    assert specs[0].startswith("market-data-hub @ git+")
    assert MARKET_DATA_HUB_GIT_URL in specs[0]
    assert "@" in specs[0].split("git+", 1)[1]  # a commit/tag is present, not a bare URL


def test_install_with_sibling_checkout_uses_editable_install() -> None:
    sibling = Path("/fake/sibling/market-data-hub")
    with patch("lazyportfolio._setup._pip") as mock_pip:
        _install_market_data_hub(sibling_hub=sibling)
    mock_pip.assert_called_once_with("install", "-e", str(sibling))


def test_install_without_sibling_checkout_pins_to_the_datacore_extra_spec() -> None:
    with patch("lazyportfolio._setup._installed_via_main_branch", return_value=False):
        with patch("lazyportfolio._setup._pip") as mock_pip:
            _install_market_data_hub(sibling_hub=None)
    pinned = _extra_requirements("datacore")
    mock_pip.assert_called_once_with("install", *pinned)
    # The exact spec _pip received must carry a pinned ref, never the bare URL alone.
    installed_spec = mock_pip.call_args.args[-1]
    assert installed_spec != f"market-data-hub @ git+{MARKET_DATA_HUB_GIT_URL}"


def test_install_without_sibling_or_datacore_extra_raises_instead_of_falling_back() -> None:
    """The exact gap flagged in review: a missing datacore extra must fail
    loudly, not silently install an unpinned URL."""
    with patch("lazyportfolio._setup._installed_via_main_branch", return_value=False):
        with patch("lazyportfolio._setup._extra_requirements", return_value=[]):
            with patch("lazyportfolio._setup._pip") as mock_pip:
                with pytest.raises(RuntimeError, match="datacore"):
                    _install_market_data_hub(sibling_hub=None)
    mock_pip.assert_not_called()


def test_install_without_sibling_checkout_skips_reinstall_if_installed_from_main() -> None:
    """A fresh `pip install market-data-hub @ ...main` moments earlier in the
    same setup session must not be clobbered by the datacore extra's older
    pinned revision."""
    with patch("lazyportfolio._setup._installed_via_main_branch", return_value=True):
        with patch("lazyportfolio._setup._pip") as mock_pip:
            _install_market_data_hub(sibling_hub=None)
    mock_pip.assert_not_called()


def test_install_without_sibling_checkout_upgrades_stale_leftover_install() -> None:
    """An installed market-data-hub that is NOT from this repo's main branch
    (e.g. a much older pin left over from a previous lazyportfolio-setup run)
    must still be upgraded to the current pin -- "already installed" alone
    must not be treated as "current"."""
    with patch("lazyportfolio._setup._installed_via_main_branch", return_value=False):
        with patch("lazyportfolio._setup._pip") as mock_pip:
            _install_market_data_hub(sibling_hub=None)
    pinned = _extra_requirements("datacore")
    mock_pip.assert_called_once_with("install", *pinned)


def test_ask_optional_path_persists_typed_value(monkeypatch) -> None:
    monkeypatch.delenv("LAZYPORTFOLIO_ARTIFACTS_DB", raising=False)
    with patch("builtins.input", return_value="C:\\data\\artifacts.db"):
        with patch("lazyportfolio._setup._set_persistent_env_var") as mock_set:
            _ask_optional_path("Artifact DB", "LAZYPORTFOLIO_ARTIFACTS_DB")
    mock_set.assert_called_once_with("LAZYPORTFOLIO_ARTIFACTS_DB", "C:\\data\\artifacts.db")


def test_ask_optional_path_keeps_existing_on_empty_answer(monkeypatch) -> None:
    monkeypatch.setenv("LAZYPORTFOLIO_ARTIFACTS_DB", "C:\\existing.db")
    with patch("builtins.input", return_value=""):
        with patch("lazyportfolio._setup._set_persistent_env_var") as mock_set:
            _ask_optional_path("Artifact DB", "LAZYPORTFOLIO_ARTIFACTS_DB")
    mock_set.assert_called_once_with("LAZYPORTFOLIO_ARTIFACTS_DB", "C:\\existing.db")


def test_ask_optional_path_skips_when_unset_and_no_answer(monkeypatch) -> None:
    monkeypatch.delenv("LAZYPORTFOLIO_ARTIFACTS_DB", raising=False)
    with patch("builtins.input", return_value=""):
        with patch("lazyportfolio._setup._set_persistent_env_var") as mock_set:
            _ask_optional_path("Artifact DB", "LAZYPORTFOLIO_ARTIFACTS_DB")
    mock_set.assert_not_called()


# --------------------------------------------------------------------------- #
# _remote_main_commit / _installed_via_main_branch -- the real implementation.
# Covers the exact gap flagged in review (on LazyFin's identical logic):
# `requested_revision == "main"` alone can't tell "installed from main a
# minute ago" apart from "installed from main months ago, now stale relative
# to a newer pin" -- only comparing against the CURRENT remote tip can.
# --------------------------------------------------------------------------- #
def test_remote_main_commit_parses_git_ls_remote_output() -> None:
    fake_result = type("R", (), {"stdout": "abc123def456\trefs/heads/main\n"})()
    with patch("lazyportfolio._setup.subprocess.run", return_value=fake_result) as mock_run:
        assert _remote_main_commit("https://example.com/repo.git") == "abc123def456"
    expected_args = ["git", "ls-remote", "https://example.com/repo.git", "refs/heads/main"]
    assert mock_run.call_args.args[0] == expected_args


def test_remote_main_commit_returns_none_on_failure() -> None:
    import subprocess

    timeout = subprocess.TimeoutExpired(cmd="git", timeout=15)
    with patch("lazyportfolio._setup.subprocess.run", side_effect=timeout):
        assert _remote_main_commit("https://example.com/repo.git") is None


def test_installed_via_main_branch_false_when_not_installed() -> None:
    not_found = metadata.PackageNotFoundError()
    with patch("lazyportfolio._setup.metadata.distribution", side_effect=not_found):
        assert _installed_via_main_branch("market-data-hub", MARKET_DATA_HUB_GIT_URL) is False


def test_installed_via_main_branch_false_when_pinned_to_a_tag_not_main() -> None:
    vcs_info = {"vcs": "git", "requested_revision": "v0.1.0", "commit_id": "aaa"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        assert _installed_via_main_branch("market-data-hub", MARKET_DATA_HUB_GIT_URL) is False


def test_installed_via_main_branch_true_when_installed_commit_matches_current_remote_tip() -> None:
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "abc123"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        with patch("lazyportfolio._setup._remote_main_commit", return_value="abc123"):
            assert _installed_via_main_branch("market-data-hub", MARKET_DATA_HUB_GIT_URL) is True


def test_installed_via_main_branch_false_when_stale_relative_to_current_main() -> None:
    """installed from `@main` a long time ago (before the pin was last
    bumped) still reports requested_revision == "main" forever, but its
    commit no longer matches main's live tip -- must fall through to
    installing the pin, not skip."""
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "old-stale-commit"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        with patch("lazyportfolio._setup._remote_main_commit", return_value="new-current-commit"):
            assert _installed_via_main_branch("market-data-hub", MARKET_DATA_HUB_GIT_URL) is False


def test_installed_via_main_branch_false_when_remote_check_fails() -> None:
    """Network/git failure must be treated as "unknown" (fall through to
    installing the pin), never as "matches"."""
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "abc123"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        with patch("lazyportfolio._setup._remote_main_commit", return_value=None):
            assert _installed_via_main_branch("market-data-hub", MARKET_DATA_HUB_GIT_URL) is False
