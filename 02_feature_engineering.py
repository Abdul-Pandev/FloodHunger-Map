"""
╔══════════════════════════════════════════════════════════════╗
║   FLOODHUNGER GHANA — SCRIPT 02: FEATURE ENGINEERING        ║
║   Builds all ML-ready features from the master table        ║
╚══════════════════════════════════════════════════════════════╝

Input:  data/processed/master_monthly.csv
Output: data/processed/features.csv
        data/processed/feature_report.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path("data/processed")

def log(msg):
    print(f"  ▸ {msg}")

def separator(title):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def build_features(df):

    df = df.sort_values(["region", "year", "month"]).reset_index(drop=True)

    separator("FEATURE GROUP 1: RAINFALL LAG FEATURES")

    # Lag 1 and lag 2 month rainfall anomaly
    for lag in [1, 2, 3]:
        df[f"rainfall_anomaly_lag{lag}"] = (
            df.groupby("district")["rainfall_anomaly_pct"].shift(lag)
        )
        df[f"flood_flag_lag{lag}"] = (
            df.groupby("district")["flood_flag"].shift(lag).fillna(0)
        )
        df[f"drought_flag_lag{lag}"] = (
            df.groupby("district")["drought_flag"].shift(lag).fillna(0)
        )

    # 3-month and 6-month rolling mean rainfall
    df["rainfall_rolling_3m"] = (
        df.groupby("district")["rainfall_mm"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(3)
    )
    df["rainfall_rolling_6m"] = (
        df.groupby("district")["rainfall_mm"]
        .transform(lambda x: x.rolling(6, min_periods=1).mean())
        .round(3)
    )

    # Consecutive flood months (streak)
    def flood_streak(series):
        streak = []
        count = 0
        for v in series:
            if v == 1:
                count += 1
            else:
                count = 0
            streak.append(count)
        return streak

    df["flood_streak_months"] = df.groupby("district")["flood_flag"].transform(flood_streak)

    log("Rainfall lag 1/2/3 ✓")
    log("Flood flag lag 1/2/3 ✓")
    log("Rolling 3m and 6m rainfall mean ✓")
    log("Flood streak counter ✓")


    separator("FEATURE GROUP 2: PRICE LAG FEATURES")

    price_cols = ["price_maize_change_pct", "price_maize_volatility_3m"]

    for col in price_cols:
        if col in df.columns:
            for lag in [1, 2]:
                df[f"{col}_lag{lag}"] = (
                    df.groupby("district")[col].shift(lag)
                )

    # Flood × Price interaction (flood event followed by price spike)
    df["flood_price_lag1_interact"] = (
        df["flood_flag_lag1"] * df.get("price_maize_change_pct", pd.Series(0, index=df.index)).fillna(0)
    ).round(3)

    # Drought × Price interaction
    df["drought_price_interact"] = (
        df["drought_flag"] * df.get("price_maize_change_pct", pd.Series(0, index=df.index)).fillna(0)
    ).round(3)

    log("Price change lag 1/2 ✓")
    log("Flood × Price lag interaction ✓")
    log("Drought × Price interaction ✓")


    separator("FEATURE GROUP 3: CONFLICT LAG FEATURES")

    if "conflict_events" in df.columns:
        for lag in [1, 2]:
            df[f"conflict_lag{lag}"] = (
                df.groupby("district")["conflict_events"].shift(lag).fillna(0)
            )

        # 3-month rolling conflict intensity
        df["conflict_rolling_3m"] = (
            df.groupby("district")["conflict_events"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum())
        )
        log("Conflict lag 1/2 ✓")
        log("Conflict 3-month rolling sum ✓")


    separator("FEATURE GROUP 4: SEASON & CALENDAR FEATURES")

    # One-hot encode season phase
    season_dummies = pd.get_dummies(df["season_phase"], prefix="season")
    df = pd.concat([df, season_dummies], axis=1)

    # Cyclical month encoding (captures seasonal continuity)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)

    # Time trend (months since start, for capturing long-run inflation trend)
    df["time_trend"] = (df["year"] - df["year"].min()) * 12 + df["month"]

    log("Season one-hot encoded ✓")
    log("Cyclical month sin/cos ✓")
    log("Time trend ✓")


    separator("FEATURE GROUP 5: REGION RISK PROFILE")

    # Historical flood frequency per region (static feature)
    flood_freq = df.groupby("district")["flood_flag"].mean().rename("region_flood_freq_hist")
    df = df.merge(flood_freq, on="district", how="left")

    # Historical conflict intensity per region
    if "conflict_events" in df.columns:
        conf_intensity = df.groupby("district")["conflict_events"].mean().rename("region_conflict_intensity_hist")
        df = df.merge(conf_intensity, on="district", how="left")

    # Region type (agro-ecological zone proxy)
    north_regions = {"Northern", "Upper East", "Upper West", "Savannah", "North East"}
    coastal_regions = {"Greater Accra", "Central", "Western"}
    df["region_type"] = df["region"].apply(
        lambda r: 0 if r in north_regions else (2 if r in coastal_regions else 1)
    )
    # 0 = Savannah (unimodal, higher vulnerability)
    # 1 = Forest/Transition (bimodal)
    # 2 = Coastal (bimodal, urban market access)

    log("Region historical flood frequency ✓")
    log("Region historical conflict intensity ✓")
    log("Region agro-ecological type ✓")


    separator("FEATURE GROUP 6: COMPOUND RISK SCORE")

    # Weighted compound risk score (continuous version of IPC logic)
    df["compound_risk_score"] = (
        df["rainfall_anomaly_pct"].clip(-100, 100).abs() * 0.01 +
        df["flood_flag"] * 2.0 +
        df.get("price_maize_change_pct", pd.Series(0, index=df.index)).fillna(0).clip(0, 100) * 0.05 +
        df.get("conflict_events", pd.Series(0, index=df.index)).fillna(0) * 0.1 +
        df["is_lean_season"] * 1.0
    ).round(3)

    log("Compound risk score ✓")


    separator("FINAL FEATURE SUMMARY")

    # Drop intermediate/non-feature columns
    drop_cols = ["season_phase"]  # keep dummies, drop raw string
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill remaining NaN lag values with 0 (first month in region has no lag)
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    df[lag_cols] = df[lag_cols].fillna(0)

    feature_cols = [c for c in df.columns if c not in ["region", "year", "month", "ipc_phase"]]
    log(f"Total features built  : {len(feature_cols)}")
    log(f"Total rows            : {len(df):,}")
    log(f"IPC Phase distribution:")
    for phase, count in df["ipc_phase"].value_counts().sort_index().items():
        log(f"  Phase {phase}: {count:,} rows ({count/len(df)*100:.1f}%)")

    return df, feature_cols


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  FLOODHUNGER GHANA — FEATURE ENGINEERING")
    print("=" * 60)

    df = pd.read_csv(PROC / "master_monthly.csv")
    log(f"Loaded master: {len(df):,} rows, {len(df.columns)} columns")

    df_features, feature_cols = build_features(df)

    # Save features
    df_features.to_csv(PROC / "features.csv", index=False)
    log(f"\n  Saved → data/processed/features.csv")

    # Save feature list
    report_path = PROC / "feature_report.txt"
    with open(report_path, "w") as f:
        f.write("FLOODHUNGER GHANA — FEATURE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features: {len(feature_cols)}\n\n")
        for i, col in enumerate(sorted(feature_cols), 1):
            dtype = df_features[col].dtype
            missing = df_features[col].isna().sum()
            f.write(f"{i:3d}. {col:<45} [{dtype}] missing={missing}\n")
    log(f"  Saved → data/processed/feature_report.txt")

    print()
    print("=" * 60)
    print("  ✓ Done. Next step:")
    print("    python 03_train_models.py")
    print("=" * 60)
