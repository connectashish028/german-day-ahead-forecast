"""One-off: compute headline metrics for the deployed quantile model so we can
add a row to the README scoreboard."""
from __future__ import annotations

import numpy as np
import pandas as pd

df = pd.read_csv("backtest_results/lstm_quantile_step7.csv")
mae_model = float(np.abs(df["y_true"] - df["y_model"]).mean())
mae_tso = float(np.abs(df["y_true"] - df["y_tso"]).mean())
mape_model = float((np.abs(df["y_true"] - df["y_model"]) / np.abs(df["y_true"]) * 100).mean())
mape_tso = float((np.abs(df["y_true"] - df["y_tso"]) / np.abs(df["y_true"]) * 100).mean())
skill = 1 - mae_model / mae_tso

print(f"TSO MAE:    {mae_tso:.1f}")
print(f"Model MAE:  {mae_model:.1f}")
print(f"TSO MAPE:   {mape_tso:.2f}")
print(f"Model MAPE: {mape_model:.2f}")
print(f"Skill:      {skill:+.4f}  ({skill*100:+.1f} %)")
print(f"Days:       {df['issue_date'].nunique()}")
