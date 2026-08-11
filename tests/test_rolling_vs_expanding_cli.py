"""Tests for the explicit job boundary of the rolling/expanding script."""

from __future__ import annotations

import pytest
from scripts import rolling_vs_expanding_backtest as job


def test_cli_preserves_caller_supplied_scope_and_runtime_policy() -> None:
    args = job.parse_args(
        [
            "--tree",
            "Tree A",
            "--tree",
            "Tree B",
            "--pruning-tree",
            "Tree A",
            "--max-workers",
            "4",
            "--telegram",
        ]
    )
    assert args.trees == ["Tree A", "Tree B"]
    assert args.pruning_trees == ["Tree A"]
    assert args.max_workers == 4
    assert args.telegram is True


def test_notification_policy_resolves_to_one_sender() -> None:
    assert job.select_sender(False) is job._no_send
    assert job.select_sender(True) is job._send_telegram_document


@pytest.mark.parametrize(
    "argv",
    [
        ["--tree", "Tree A", "--max-workers", "4"],
        ["--tree", "Tree A", "--max-workers", "0", "--no-telegram"],
        [
            "--tree",
            "Tree A",
            "--pruning-tree",
            "Tree B",
            "--max-workers",
            "4",
            "--no-telegram",
        ],
    ],
)
def test_cli_rejects_implicit_or_inconsistent_policy(argv) -> None:
    with pytest.raises(SystemExit):
        job.parse_args(argv)
