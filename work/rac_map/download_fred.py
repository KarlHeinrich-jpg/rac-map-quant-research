"""Download public FRED macro series and merge them into the causal panel.

FRED observations are aligned to the next available ETF trading date and then
shifted by one trading day.  This is a conservative publication-lag rule; it
is not a claim that the vintage database is available.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SERIES = {"VIXCLS": "vix", "DFF": "fedfunds", "DGS10": "dgs10", "T10Y2Y": "t10y2y", "BAA10Y": "baa10y"}


def download_fred(output_dir: str | Path, feature_path: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for series, name in SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        frame = pd.read_csv(url, parse_dates=["observation_date"])
        frame = frame.rename(columns={"observation_date": "date", series: name})
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        pieces.append(frame.set_index("date"))
    macro = pd.concat(pieces, axis=1).sort_index()
    macro = macro.replace(".", pd.NA).apply(pd.to_numeric, errors="coerce")
    macro.index = pd.to_datetime(macro.index).normalize()
    raw_path = output_dir / "fred_macro_daily.csv"
    macro.reset_index().to_csv(raw_path, index=False, date_format="%Y-%m-%d")

    features = pd.read_parquet(feature_path)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(features["date"]).dt.normalize().unique()))
    # Merge by observation date, forward-fill, then lag one market trading day.
    aligned = macro.reindex(dates).ffill().shift(1)
    aligned.index.name = "date"
    for col in aligned.columns:
        # Scales are fixed ex ante to keep the state bounded and interpretable.
        scale = {"vix": 100.0, "fedfunds": 10.0, "dgs10": 10.0, "t10y2y": 10.0, "baa10y": 10.0}[col]
        aligned[f"macro_{col}"] = aligned[col] / scale
    macro_cols = [f"macro_{name}" for name in SERIES.values()]
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.merge(aligned[macro_cols].reset_index(), on="date", how="left")
    merged[macro_cols] = merged[macro_cols].ffill().bfill()
    out_path = output_dir / "features_macro.parquet"
    merged.to_parquet(out_path, index=False)
    metadata = {
        "source": "FRED fredgraph CSV",
        "series": SERIES,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_lag_rule": "reindex to ETF dates, forward-fill, then shift one trading day",
        "point_in_time_note": "FRED revised observations are not vintage-controlled; this is a conservative public-release lag proxy.",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "feature_macro_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "actual_macro_start": str(macro.index.min().date()),
        "actual_macro_end": str(macro.index.max().date()),
    }
    (output_dir / "fred_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="work/rac_map/data")
    parser.add_argument("--features", default="work/rac_map/data/features_daily.parquet")
    args = parser.parse_args()
    frame = download_fred(args.output_dir, args.features)
    print(f"saved {len(frame):,} rows to features_macro.parquet")
