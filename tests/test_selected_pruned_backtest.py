"""Pruning's own fitting parameters, made settable in the selection arm too.

`selected_pruned_backtest.py`'s module docstring says it is "same idea as
adaptive_pruning_backtest.py, extended to a multi-way choice". That script
already exposes pruning's two thresholds on the CLI
(--min-sharpe-improvement, --max-drawdown-per-vol-ratio); this one hardcoded
`PruningRule(required_protocols=("accumulated",))` and left both at the
dataclass default regardless of what a caller asked for. Pruning is one
mechanism with one set of knobs -- a second arm that runs it should not get
a second, silently different way to configure it.

The rest of this 315-line script -- the walk-forward loop, the causal
profile-selection comparison, the report assembly -- has no test here. It
predates this pass, its own commit says "neither tests nor a review", and
giving it real coverage needs market-data fixtures on the scale of
test_adaptive_pruning.py's, which is a separate undertaking from making its
pruning parameters match the standard the rest of the codebase already set.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "selected_pruned_backtest.py"


class _StopAfterRun(Exception):
    """Raised by a stubbed `run()` once its arguments are captured, so a test
    never reaches the real walk-forward (`store.read_model`, market data,
    `build_client_report`, ...) that `main()` calls after `run()` returns."""


def _load():
    """Load the script the way Python itself would run it.

    Run directly (`python scripts/selected_pruned_backtest.py`), the
    interpreter puts the script's own directory on `sys.path[0]`
    automatically, which is what lets `import pruning_runner` -- a sibling
    file in `scripts/`, not a package -- resolve. `importlib` does not do
    that for a module loaded by file location, so it has to be added here or
    every import in the module under test fails before any test body runs.
    """
    import importlib.util
    import sys

    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("selected_pruned_backtest_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loads_the_script_from_the_worktree_under_review() -> None:
    mod = _load()
    assert Path(mod.__file__).resolve() == SCRIPT.resolve()


def test_run_accepts_the_same_two_pruning_thresholds_as_the_reference_script() -> None:
    mod = _load()
    params = inspect.signature(mod.run).parameters
    assert "min_sharpe_improvement" in params
    assert "max_drawdown_per_vol_ratio" in params
    # same defaults as PruningRule's own, not a second copy of the numbers
    assert params["min_sharpe_improvement"].default == mod.PruningRule.min_sharpe_improvement
    assert params["max_drawdown_per_vol_ratio"].default == mod.PruningRule.max_drawdown_per_vol_ratio


def test_main_forwards_the_parsed_thresholds_into_run_not_the_defaults(monkeypatch) -> None:
    """The gap this whole file exists to close: the flags could be declared
    and never actually reach `run()`, which would parse but silently keep
    using PruningRule's hardcoded defaults regardless of what was passed.

    Runs the real CLI parse -> main() -> run() call, not a text search: a
    dest typo or the two threshold kwargs swapped in the call to `run()`
    would parse fine and only show up here, in what `run()` actually
    receives.
    """
    mod = _load()
    captured = {}

    def fake_run(base, profiles, **kwargs):
        captured.update(base=base, profiles=profiles, **kwargs)
        raise _StopAfterRun

    monkeypatch.setattr(mod, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["selected_pruned_backtest.py", "mybase", "profile_a", "profile_b",
         "--min-sharpe-improvement", "0.07", "--max-drawdown-per-vol-ratio", "2.5"],
    )
    with pytest.raises(_StopAfterRun):
        mod.main()

    assert captured["base"] == "mybase"
    assert captured["profiles"] == ["profile_a", "profile_b"]
    # exact values, not just presence -- catches a swap between the two
    assert captured["min_sharpe_improvement"] == 0.07
    assert captured["max_drawdown_per_vol_ratio"] == 2.5


def test_burn_in_fold_targets_reads_the_given_fold_not_some_other_ones_position() -> None:
    """Regression for a real bug caught by review: a previous version fetched
    this fold's target as `base_reference.folds[i]`, where `i` was the local,
    post-`max_folds`-slice loop index but `base_reference.folds` was the
    full, untruncated list -- silently returning an *earlier* fold's target
    once `max_folds` truncated the run. `_burn_in_fold_targets` takes the
    fold object directly, so there is no second, differently-truncated list
    for an index to drift out of sync with; this pins that down using two
    folds with distinguishable targets, the way the truncated/untruncated
    mismatch would actually surface.
    """
    mod = _load()

    class _FakeFold:
        def __init__(self, targets):
            self.targets = targets

    early_fold = _FakeFold({"FINAL": {"AAA": 1.0}, "FORWARD_FINAL": {"AAA": 0.9}})
    late_fold = _FakeFold({"FINAL": {"BBB": 1.0}, "FORWARD_FINAL": {"BBB": 0.9}})

    base_target, forward_target = mod._burn_in_fold_targets(late_fold)
    assert base_target == {"BBB": 1.0}
    assert forward_target == {"BBB": 0.9}
    # never silently returns another fold's data
    assert base_target != dict(early_fold.targets["FINAL"])


def test_burn_in_fold_targets_falls_back_to_final_when_forward_final_is_absent() -> None:
    mod = _load()

    class _FakeFold:
        def __init__(self, targets):
            self.targets = targets

    fold = _FakeFold({"FINAL": {"AAA": 1.0}})
    base_target, forward_target = mod._burn_in_fold_targets(fold)
    assert base_target == {"AAA": 1.0}
    assert forward_target == {"AAA": 1.0}


def test_run_builds_pruning_rule_from_its_own_parameters_not_hardcoded_numbers() -> None:
    """The regression this closes: `PruningRule(required_protocols=("accumulated",))`
    with no threshold arguments at all, so every run silently used 0.03/1.10
    no matter what a caller wanted."""
    mod = _load()
    source = inspect.getsource(mod.run)
    # the construction must reference the parameters, not literal numbers
    rule_call = source[source.index("PruningRule(") : source.index("PruningRule(") + 300]
    assert "min_sharpe_improvement" in rule_call
    assert "max_drawdown_per_vol_ratio" in rule_call
    assert "0.03" not in rule_call
    assert "1.10" not in rule_call and "1.1" not in rule_call
