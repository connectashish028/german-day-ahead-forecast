"""Rolling-origin backtest harness.

Walks delivery dates, calls the predictor at each issue time, compares to the
ground-truth `actual_cons__grid_load`, and computes per-day errors plus the
overall skill score vs the TSO baseline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .baselines import ACTUAL_LOAD_COL, tso_baseline_predict
from .loader import issue_time_for, load_smard_15min, target_index_for
from .metrics import mae, mape, rmse, skill_score
from .types import PredictFn


@dataclass
class BacktestResult:
    per_step: pd.DataFrame  # one row per (issue_date, target_ts)
    per_day: pd.DataFrame  # one row per issue_date with mae_model / mae_tso / skill
    overall: dict[str, float]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.per_step.to_csv(path)


def _daterange(start: date, end: date, step_days: int = 1) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step_days)


def run_backtest(
    predict_fn: PredictFn,
    df: pd.DataFrame,
    start: date,
    end: date,
    step_days: int = 1,
    *,
    progress: bool = True,
    label: str = "model",
) -> BacktestResult:
    """Walk delivery dates [start, end], evaluate `predict_fn` and the TSO baseline."""

    rows: list[pd.DataFrame] = []
    iterator: Iterable[date] = list(_daterange(start, end, step_days))
    if progress:
        iterator = tqdm(iterator, desc=f"backtest:{label}", unit="day")

    for delivery in iterator:
        issue = issue_time_for(delivery)
        target_idx = target_index_for(delivery)

        # Truth for this day
        y_true = df[ACTUAL_LOAD_COL].reindex(target_idx)
        if y_true.isna().any():
            # Skip days where ground truth is incomplete (e.g. very recent edge of dataset).
            continue

        try:
            y_model = predict_fn(df, issue)
        except Exception as e:  # noqa: BLE001 — surface errors per-day, keep going
            tqdm.write(f"[backtest] {delivery}: predictor raised {e!r}; skipping")
            continue

        if not isinstance(y_model, pd.Series) or len(y_model) != 96:
            raise ValueError(
                f"Predictor for {delivery} returned {type(y_model).__name__} of "
                f"length {len(y_model) if hasattr(y_model, '__len__') else 'n/a'}, "
                "expected pd.Series of length 96."
            )
        y_model = y_model.reindex(target_idx)

        y_tso = tso_baseline_predict(df, issue).reindex(target_idx)

        rows.append(
            pd.DataFrame(
                {
                    "issue_date": delivery,
                    "target_ts": target_idx,
                    "horizon_qh": range(96),
                    "y_true": y_true.to_numpy(),
                    "y_model": y_model.to_numpy(),
                    "y_tso": y_tso.to_numpy(),
                }
            )
        )

    if not rows:
        raise RuntimeError("No backtest days produced — check date range / data coverage.")

    per_step = pd.concat(rows, ignore_index=True)

    per_day = (
        per_step.groupby("issue_date")
        .apply(
            lambda g: pd.Series(
                {
                    "mae_model": mae(g["y_true"], g["y_model"]),
                    "mae_tso": mae(g["y_true"], g["y_tso"]),
                    "rmse_model": rmse(g["y_true"], g["y_model"]),
                    "rmse_tso": rmse(g["y_true"], g["y_tso"]),
                    "mape_model": mape(g["y_true"], g["y_model"]),
                    "mape_tso": mape(g["y_true"], g["y_tso"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_day["skill"] = 1 - per_day["mae_model"] / per_day["mae_tso"]

    overall = {
        "mae_model": mae(per_step["y_true"], per_step["y_model"]),
        "mae_tso": mae(per_step["y_true"], per_step["y_tso"]),
        "rmse_model": rmse(per_step["y_true"], per_step["y_model"]),
        "rmse_tso": rmse(per_step["y_true"], per_step["y_tso"]),
        "mape_model": mape(per_step["y_true"], per_step["y_model"]),
        "mape_tso": mape(per_step["y_true"], per_step["y_tso"]),
    }
    overall["skill_score"] = skill_score(overall["mae_model"], overall["mae_tso"])
    overall["n_days"] = int(per_day.shape[0])

    return BacktestResult(per_step=per_step, per_day=per_day, overall=overall)


def quick_backtest(
    predict_fn: PredictFn,
    start: date,
    end: date,
    parquet_path: str | Path = "smard_merged_15min.parquet",
    **kwargs,
) -> BacktestResult:
    """Convenience wrapper that loads the default parquet."""
    df = load_smard_15min(parquet_path)
    return run_backtest(predict_fn, df, start, end, **kwargs)


__all__ = ["BacktestResult", "run_backtest", "quick_backtest"]
