"""Replicate github.com/I-am-Uchenna/regime-allocation-strategy exactly
(leaky: 2-state HMM on daily %VIX change, fit once on the whole sample,
hard switch 100% SPY (low-vol state) / 100% TLT (high-vol state), 1-day
execution lag), then rerun the identical rule causally (HMM refit
quarterly on data <= date only, daily filtered inference in between,
same infer_with_params/window-ends-at-day pattern already validated
today) to see whether the published Sharpe ~1.22 survives without the
full-sample fit the author's own README admits is leaky.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(ROOT / "src"), str(ROOT / "project"), str(ROOT.parent / "LazyStats" / "src"),
    str(ROOT.parent / "market-data-hub"),
]
from lazystats.regimes.core import HMMParams, MSRegimeEngine, infer_with_params  # noqa: E402
from market_data_hub import extract  # noqa: E402

START, END = "2004-11-19", "2026-01-16"
REFIT_EVERY = pd.DateOffset(months=6)


def load_data():
    prices, _ = extract.extract_series(["SPY", "TLT", "^VIX"], transform="level", frequency="D", start=START, end=END)
    prices = prices.dropna(how="any")
    returns = prices[["SPY", "TLT"]].pct_change().dropna()
    vix_pct_change = prices["^VIX"].pct_change().dropna()
    common = returns.index.intersection(vix_pct_change.index)
    return returns.loc[common], vix_pct_change.loc[common]


def metrics(curve: pd.Series) -> dict:
    wealth = (1 + curve).cumprod()
    years = len(curve) / 252.0
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1)
    vol = float(curve.std(ddof=1) * np.sqrt(252))
    sharpe = float(curve.mean() / curve.std(ddof=1) * np.sqrt(252)) if curve.std(ddof=1) else 0.0
    mdd = float((wealth / wealth.cummax() - 1).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": mdd, "n_obs": len(curve)}


def high_vol_state_label(means) -> int:
    return int(np.argmax(np.asarray(means).ravel()))


def run_leaky(returns: pd.DataFrame, vix_pct_change: pd.Series) -> pd.Series:
    engine = MSRegimeEngine(S_max=2, S_min=2, criterion="bic", n_starts=30, random_state=7, standardize=True)
    run = engine.fit(vix_pct_change.to_frame("V"), model="panel")
    high_state = high_vol_state_label(run.meta["V"]["means_"])
    state = run.panel["V_state"].reindex(returns.index).ffill()
    lagged_state = state.shift(1)  # 1-day execution lag
    weight_spy = (lagged_state != high_state).astype(float)
    curve = weight_spy * returns["SPY"] + (1 - weight_spy) * returns["TLT"]
    return curve.dropna()


def run_causal(returns: pd.DataFrame, vix_pct_change: pd.Series) -> pd.Series:
    checkpoints = pd.date_range(vix_pct_change.index[252], vix_pct_change.index[-1], freq=REFIT_EVERY)
    state_by_day: dict = {}
    for checkpoint in checkpoints:
        fit_window = vix_pct_change.loc[:checkpoint].to_frame("V")
        engine = MSRegimeEngine(S_max=2, S_min=2, criterion="bic", n_starts=20, random_state=7, standardize=True)
        run = engine.fit(fit_window, model="panel")
        params = HMMParams.from_dict(run.meta["V"])
        high_state = high_vol_state_label(params.means_)
        next_checkpoint = checkpoint + REFIT_EVERY
        score_days = vix_pct_change.loc[checkpoint:min(next_checkpoint, vix_pct_change.index[-1])].index
        for day in score_days:
            window = vix_pct_change.loc[checkpoint:day].to_numpy().reshape(-1, 1)
            res = infer_with_params(window, params)
            state_by_day[day] = int(res.viterbi_path_[-1] == high_state)
        print(f"checkpoint {checkpoint.date()}: fit on {len(fit_window)} rows, scored {len(score_days)} days", flush=True)
    state = pd.Series(state_by_day).reindex(returns.index).ffill()
    lagged_state = state.shift(1)
    weight_spy = (lagged_state != 1).astype(float)
    eval_start = checkpoints[0]
    curve = (weight_spy * returns["SPY"] + (1 - weight_spy) * returns["TLT"]).dropna()
    return curve.loc[eval_start:]


def main() -> None:
    returns, vix_pct_change = load_data()
    print(f"{len(returns)} common daily observations, {returns.index.min().date()} to {returns.index.max().date()}")

    leaky_curve = run_leaky(returns, vix_pct_change)
    print("LEAKY (full-sample fit, matching the published methodology):")
    print(json.dumps(metrics(leaky_curve), indent=2))
    print("Published: cagr=0.1941 sharpe=1.22 max_drawdown=-0.1954")

    print("\nBuilding causal version (quarterly refit, daily filtered inference)...")
    causal_curve = run_causal(returns, vix_pct_change)
    print("\nCAUSAL (same rule, no full-sample leak):")
    print(json.dumps(metrics(causal_curve), indent=2))

    spy_bh = metrics(returns["SPY"].loc[causal_curve.index])
    print("\nSPY buy & hold (same evaluation window, for reference):")
    print(json.dumps(spy_bh, indent=2))


if __name__ == "__main__":
    main()
