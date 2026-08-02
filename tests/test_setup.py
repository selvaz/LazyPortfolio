"""Regression tests for lazyportfolio-setup's market-data-hub install logic.

Covers the exact gap flagged in review: the no-sibling-checkout fallback
must install the datacore extra's pinned revision (never an unpinned URL),
and must fail loudly rather than silently degrading if that pin is ever
unavailable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lazyportfolio._setup import (
    MARKET_DATA_HUB_GIT_URL,
    _ask_optional_path,
    _extra_requirements,
    _install_market_data_hub,
)


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
