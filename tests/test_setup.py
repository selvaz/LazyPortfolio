"""Regression tests for lazyportfolio-setup's market-data-hub install logic.

Covers the exact gap flagged in review: the no-sibling-checkout fallback
must install the datacore extra's pinned revision (never an unpinned URL),
and must fail loudly rather than silently degrading if that pin is ever
unavailable.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyportfolio._setup import (
    MARKET_DATA_HUB_GIT_URL,
    _ask_optional_path,
    _extra_requirements,
    _github_compare_status,
    _install_market_data_hub,
    _installed_satisfies_pin,
    _spec_ref,
)


def _fake_direct_url_dist(vcs_info: dict | None):
    """A stand-in for importlib.metadata.distribution()'s return value,
    with just enough surface (.read_text) for _installed_satisfies_pin."""
    dist = MagicMock()
    dist.read_text.return_value = None if vcs_info is None else json.dumps({"vcs_info": vcs_info})
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
    with patch("lazyportfolio._setup._installed_satisfies_pin", return_value=False):
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
    with patch("lazyportfolio._setup._extra_requirements", return_value=[]):
        with patch("lazyportfolio._setup._pip") as mock_pip:
            with pytest.raises(RuntimeError, match="datacore"):
                _install_market_data_hub(sibling_hub=None)
    mock_pip.assert_not_called()


def test_install_without_sibling_checkout_skips_reinstall_if_pin_is_satisfied() -> None:
    """A fresh `pip install market-data-hub @ ...main` moments earlier in the
    same setup session must not be clobbered by the datacore extra's older
    pinned revision."""
    with patch("lazyportfolio._setup._installed_satisfies_pin", return_value=True):
        with patch("lazyportfolio._setup._pip") as mock_pip:
            _install_market_data_hub(sibling_hub=None)
    mock_pip.assert_not_called()


def test_install_without_sibling_checkout_upgrades_stale_leftover_install() -> None:
    """An installed market-data-hub that does NOT satisfy the current pin
    (e.g. a much older pin left over from a previous lazyportfolio-setup
    run) must still be upgraded -- "already installed" alone must not be
    treated as "current"."""
    with patch("lazyportfolio._setup._installed_satisfies_pin", return_value=False):
        with patch("lazyportfolio._setup._pip") as mock_pip:
            _install_market_data_hub(sibling_hub=None)
    pinned = _extra_requirements("datacore")
    mock_pip.assert_called_once_with("install", *pinned)


def test_spec_ref_extracts_ref_after_the_url_not_the_name_separator() -> None:
    assert _spec_ref("market-data-hub @ git+https://x.git@abc123") == "abc123"
    assert _spec_ref("market-data-hub @ git+https://x.git") is None  # no trailing ref
    assert _spec_ref("pytest>=7") is None  # no " @ " at all


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
# _github_compare_status / _installed_satisfies_pin -- the real
# implementation. Covers the exact gap flagged in review across three rounds
# (on LazyFin's identical logic): (1) presence alone can't tell fresh-from-
# main apart from a stale leftover, (2) requested_revision=="main" alone
# can't tell "installed a minute ago" apart from "installed months ago, now
# stale", and (3) exact-tip equality over-corrects -- an install newer than
# the pin but a few commits behind today's live tip would be wrongly
# downgraded back to the pin. Only true ancestry (is the installed commit at
# or after the pin?) answers all three correctly.
# --------------------------------------------------------------------------- #
def test_github_compare_status_parses_the_api_response() -> None:
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps({"status": "ahead"}).encode()
    fake_response.__enter__.return_value = fake_response
    with patch(
        "lazyportfolio._setup.urllib.request.urlopen", return_value=fake_response
    ) as mock_urlopen:
        status = _github_compare_status(
            "https://github.com/selvaz/market-data-hub.git", "aaa", "bbb"
        )
    assert status == "ahead"
    called_request = mock_urlopen.call_args.args[0]
    assert (
        called_request.full_url
        == "https://api.github.com/repos/selvaz/market-data-hub/compare/aaa...bbb"
    )


def test_github_compare_status_none_for_non_github_url() -> None:
    assert (
        _github_compare_status("https://gitlab.com/selvaz/market-data-hub.git", "aaa", "bbb")
        is None
    )


def test_github_compare_status_none_on_network_failure() -> None:
    import urllib.error

    with patch(
        "lazyportfolio._setup.urllib.request.urlopen", side_effect=urllib.error.URLError("boom")
    ):
        assert (
            _github_compare_status("https://github.com/selvaz/market-data-hub.git", "aaa", "bbb")
            is None
        )


def test_installed_satisfies_pin_false_when_not_installed() -> None:
    not_found = metadata.PackageNotFoundError()
    with patch("lazyportfolio._setup.metadata.distribution", side_effect=not_found):
        assert (
            _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "abc123") is False
        )


def test_installed_satisfies_pin_true_when_commit_exactly_matches_pin() -> None:
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "abc123"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        assert (
            _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "abc123") is True
        )


def test_installed_satisfies_pin_true_when_installed_commit_is_ahead_of_pin() -> None:
    """The exact gap flagged in the 4th review round (on LazyFin's identical
    logic): installed from @main a week ago (newer than the pin, but behind
    today's live tip) must still satisfy the pin -- not be wrongly
    downgraded back to it."""
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "newer-than-pin"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        with patch(
            "lazyportfolio._setup._github_compare_status", return_value="ahead"
        ) as mock_compare:
            assert (
                _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "old-pin")
                is True
            )
    mock_compare.assert_called_once_with(MARKET_DATA_HUB_GIT_URL, "old-pin", "newer-than-pin")


def test_installed_satisfies_pin_false_when_installed_commit_is_behind_pin() -> None:
    """A stale leftover install (older than even the pin) must not be
    treated as satisfying it."""
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "very-old-commit"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        with patch("lazyportfolio._setup._github_compare_status", return_value="behind"):
            assert (
                _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "newer-pin")
                is False
            )


def test_installed_satisfies_pin_false_when_compare_api_unavailable() -> None:
    """Network/API failure must be treated as "unknown" (fall through to
    installing the pin), never as "satisfies"."""
    vcs_info = {"vcs": "git", "requested_revision": "main", "commit_id": "some-commit"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        with patch("lazyportfolio._setup._github_compare_status", return_value=None):
            assert (
                _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "some-pin")
                is False
            )


def test_installed_satisfies_pin_true_for_editable_install_regardless_of_revision() -> None:
    """An editable install (`pip install -e /local/checkout`) records
    `dir_info`, not `vcs_info` -- must be trusted unconditionally (matching
    how the sibling_hub branch above already trusts a local checkout
    without ever checking its revision either), never silently replaced
    by reinstalling the git pin."""
    dist = MagicMock()
    dist.read_text.return_value = json.dumps(
        {"dir_info": {"editable": True}, "url": "file:///home/dev/market-data-hub"}
    )
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        assert (
            _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "abc123") is True
        )


def test_installed_satisfies_pin_false_when_not_a_git_install() -> None:
    vcs_info = {"vcs": "hg", "requested_revision": "main", "commit_id": "abc123"}
    dist = _fake_direct_url_dist(vcs_info)
    with patch("lazyportfolio._setup.metadata.distribution", return_value=dist):
        assert (
            _installed_satisfies_pin("market-data-hub", MARKET_DATA_HUB_GIT_URL, "abc123") is False
        )
