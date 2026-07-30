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
