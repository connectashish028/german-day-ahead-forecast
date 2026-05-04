"""CLI: `python -m loadforecast.backtest --start 2025-01-01 --end 2026-03-01`.

Runs the backtest using only the TSO baseline and the seasonal-naive baseline.
Once a model is trained (Milestone 4+), pass `--predictor module.path:fn_name`.
"""

from __future__ import annotations

import argparse
import importlib
from datetime import date

from .baselines import seasonal_naive_predict, tso_baseline_predict
from .harness import quick_backtest
from .sarimax_baseline import sarimax_residual_predict


def _lazy_lstm_plain(df, issue_time):
    """Defer the heavy TF import until the user actually picks the LSTM."""
    from ..models.predict import lstm_residual_predict
    return lstm_residual_predict(df, issue_time)


def _lazy_lstm_attention(df, issue_time):
    from ..models.predict import lstm_attention_predict
    return lstm_attention_predict(df, issue_time)


def _lazy_lstm_weather(df, issue_time):
    from ..models.predict import lstm_weather_predict
    return lstm_weather_predict(df, issue_time)


def _resolve_predictor(spec: str):
    if ":" not in spec:
        raise ValueError(f"--predictor must be 'module.path:fn_name', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Day-ahead load-forecast backtest.")
    p.add_argument("--start", type=_parse_date, required=True, help="First delivery date (YYYY-MM-DD)")
    p.add_argument("--end", type=_parse_date, required=True, help="Last delivery date (YYYY-MM-DD)")
    p.add_argument("--data", default="smard_merged_15min.parquet")
    p.add_argument(
        "--predictor",
        default="loadforecast.backtest.baselines:tso_baseline_predict",
        help="Dotted path to the predictor: 'module.path:fn_name'.",
    )
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="Optional path for per-step CSV.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--step-days", type=int, default=1, help="Stride between delivery dates.")
    args = p.parse_args()

    predictors = {
        "tso": tso_baseline_predict,
        "seasonal_naive": seasonal_naive_predict,
        "sarimax_residual": sarimax_residual_predict,
        "lstm_plain": _lazy_lstm_plain,
        "lstm_attention": _lazy_lstm_attention,
        "lstm_weather": _lazy_lstm_weather,
    }
    if args.predictor in predictors:
        fn = predictors[args.predictor]
        label = args.label or args.predictor
    else:
        fn = _resolve_predictor(args.predictor)
        label = args.label or args.predictor.split(":")[-1]

    result = quick_backtest(
        fn,
        args.start,
        args.end,
        parquet_path=args.data,
        progress=not args.no_progress,
        label=label,
        step_days=args.step_days,
    )

    print()
    print(f"=== Backtest [{label}] {args.start} -> {args.end} ({result.overall['n_days']} days) ===")
    for k in ("mae_model", "mae_tso", "rmse_model", "rmse_tso", "mape_model", "mape_tso"):
        print(f"  {k:>12s}: {result.overall[k]:.3f}")
    print(f"  {'skill':>12s}: {result.overall['skill_score']:+.4f}  (>0 = beats TSO)")

    if args.out:
        result.save(args.out)
        print(f"\nWrote per-step CSV -> {args.out}")


if __name__ == "__main__":
    main()
