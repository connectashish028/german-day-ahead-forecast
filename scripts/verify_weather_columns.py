"""Quick sanity check on weather columns in the parquet."""
import pandas as pd

df = pd.read_parquet("smard_merged_15min.parquet")
weather_cols = [c for c in df.columns if c.startswith("weather__")]
print(f"Weather columns: {weather_cols}")
print()
for c in weather_cols:
    s = df[c]
    print(f"{c:<35s}  non-null {s.notna().sum():>7d}/{len(s)}  "
          f"min={s.min():>8.2f}  max={s.max():>8.2f}  mean={s.mean():>8.2f}")

# Verify the 1 May 2026 high-PV event lines up with shortwave radiation
ts = pd.Timestamp("2026-05-01 12:00", tz="UTC")
print(f"\nWeather at {ts}:")
for c in weather_cols:
    print(f"  {c:<35s}  {df.loc[ts, c]:.2f}")
