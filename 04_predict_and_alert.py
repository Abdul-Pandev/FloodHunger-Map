"""
╔══════════════════════════════════════════════════════════════╗
║   FLOODHUNGER GHANA — SCRIPT 04: PREDICT & ALERT            ║
║   Runs trained models on latest data → generates alerts     ║
╚══════════════════════════════════════════════════════════════╝

Input:  data/processed/features.csv  (or any new monthly data)
        data/models/  (trained .pkl files)

Output: data/outputs/predictions_latest.csv   ← all predictions
        data/outputs/alerts_wfp.csv           ← WFP-ready alert feed
        data/outputs/farmer_alerts.txt        ← plain language farmer alerts
        data/outputs/dashboard_summary.txt    ← dashboard summary
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

PROC    = Path("data/processed")
MODELS  = Path("data/models")
OUTPUTS = Path("data/outputs")

IPC_LABELS = {
    1: "Minimal",
    2: "Stressed",
    3: "Crisis",
    4: "Emergency"
}

IPC_ACTIONS = {
    1: "No action needed — continue monitoring",
    2: "Alert field teams — pre-position supplies",
    3: "Deploy food stocks and vouchers immediately",
    4: "EMERGENCY — deploy cash transfers NOW"
}

IPC_COLORS = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}

def log(msg):
    print(f"  ▸ {msg}")

def separator(title):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


# ══════════════════════════════════════════════════════════════
#  LOAD MODELS & DATA
# ══════════════════════════════════════════════════════════════

def load_everything():
    separator("LOADING MODELS & DATA")

    df = pd.read_csv(PROC / "features.csv")
    log(f"Feature data: {len(df):,} rows")

    ipc_model      = joblib.load(MODELS / "ipc_classifier_xgb.pkl")
    feature_cols   = joblib.load(MODELS / "feature_cols.pkl")
    phase_to_idx   = joblib.load(MODELS / "phase_to_idx.pkl")
    idx_to_phase   = {v: k for k, v in phase_to_idx.items()}

    price_model    = joblib.load(MODELS / "price_regressor_rf.pkl")
    reg_features   = joblib.load(MODELS / "reg_feature_cols.pkl")

    anomaly_model  = joblib.load(MODELS / "anomaly_detector_if.pkl")
    anomaly_scaler = joblib.load(MODELS / "anomaly_scaler.pkl")
    anomaly_feats  = joblib.load(MODELS / "anomaly_feature_cols.pkl")

    log("All 3 models loaded ✓")
    return df, ipc_model, feature_cols, idx_to_phase, price_model, reg_features, anomaly_model, anomaly_scaler, anomaly_feats


# ══════════════════════════════════════════════════════════════
#  RUN PREDICTIONS
# ══════════════════════════════════════════════════════════════

def run_predictions(df, ipc_model, feature_cols, idx_to_phase,
                    price_model, reg_features,
                    anomaly_model, anomaly_scaler, anomaly_feats):
    separator("RUNNING PREDICTIONS ON ALL DATA")

    # ── IPC Phase prediction ──────────────────────────────
    X = df[feature_cols].fillna(0)
    ipc_pred_idx = ipc_model.predict(X)
    ipc_pred_prob = ipc_model.predict_proba(X)

    df["pred_ipc_phase"]  = [idx_to_phase[i] for i in ipc_pred_idx]
    df["pred_ipc_label"]  = df["pred_ipc_phase"].map(IPC_LABELS)
    df["pred_ipc_action"] = df["pred_ipc_phase"].map(IPC_ACTIONS)

    # Confidence = max probability across classes
    df["pred_ipc_confidence"] = ipc_pred_prob.max(axis=1).round(3)

    log(f"IPC predictions complete")
    log(f"Phase distribution: {df['pred_ipc_phase'].value_counts().sort_index().to_dict()}")

    # ── Price change prediction ───────────────────────────
    X_price = df[reg_features].fillna(0)
    df["pred_price_change_pct"] = price_model.predict(X_price).round(2)
    df["pred_price_direction"]  = df["pred_price_change_pct"].apply(
        lambda x: "↑ Rising" if x > 5 else ("↓ Falling" if x < -5 else "→ Stable")
    )
    log(f"Price predictions complete")

    # ── Anomaly detection ─────────────────────────────────
    X_anom_raw = df[anomaly_feats].fillna(0)
    X_anom = anomaly_scaler.transform(X_anom_raw)
    df["pred_anomaly_flag"]  = anomaly_model.predict(X_anom)   # -1 = anomaly
    df["pred_anomaly_score"] = (-anomaly_model.score_samples(X_anom)).round(4)
    df["is_anomaly"] = (df["pred_anomaly_flag"] == -1).astype(int)
    log(f"Anomaly detection complete: {df['is_anomaly'].sum()} flagged")

    return df


# ══════════════════════════════════════════════════════════════
#  GENERATE ALERTS
# ══════════════════════════════════════════════════════════════

def generate_alerts(df):
    separator("GENERATING WFP ALERT FEED")

    # Focus on latest 12 months of data
    max_year  = df["year"].max()
    max_month = df[df["year"] == max_year]["month"].max()
    recent = df[
        ((df["year"] == max_year) & (df["month"] <= max_month)) |
        ((df["year"] == max_year - 1) & (df["month"] > max_month))
    ].copy()

    log(f"Alert window: last 12 months ({max_year-1}/{max_month+1} → {max_year}/{max_month})")

    # WFP-ready alert table
    alerts = recent[[
        "region", "year", "month",
        "pred_ipc_phase", "pred_ipc_label", "pred_ipc_confidence",
        "pred_ipc_action", "pred_price_change_pct", "pred_price_direction",
        "is_anomaly", "pred_anomaly_score",
        "rainfall_anomaly_pct", "flood_flag", "conflict_events",
        "compound_risk_score"
    ]].copy()

    alerts = alerts.sort_values(["year", "month", "pred_ipc_phase"], ascending=[True, True, False])
    alerts.to_csv(OUTPUTS / "alerts_wfp.csv", index=False)
    log(f"WFP alert feed saved → data/outputs/alerts_wfp.csv ({len(alerts)} rows)")

    return alerts, recent


def generate_farmer_alerts(recent):
    separator("GENERATING FARMER VOICE ALERTS")

    # High-risk and anomalous months
    flagged = recent[
        (recent["pred_ipc_phase"] >= 2) | (recent["is_anomaly"] == 1)
    ].sort_values("pred_ipc_phase", ascending=False)

    alert_lines = []
    alert_lines.append("=" * 60)
    alert_lines.append("  FLOODHUNGER GHANA — FARMER ALERT BULLETIN")
    alert_lines.append(f"  Generated: {datetime.now().strftime('%B %Y')}")
    alert_lines.append("=" * 60)
    alert_lines.append("")

    if len(flagged) == 0:
        alert_lines.append("  ✅ No high-risk periods detected in recent 12 months.")
    else:
        alert_lines.append(f"  ⚠️  {len(flagged)} REGION-MONTHS FLAGGED FOR ATTENTION")
        alert_lines.append("")

        for _, row in flagged.iterrows():
            phase = int(row["pred_ipc_phase"])
            icon  = IPC_COLORS.get(phase, "⚪")
            month_name = datetime(int(row["year"]), int(row["month"]), 1).strftime("%B %Y")
            price_dir  = row["pred_price_direction"]
            rain_anom  = row["rainfall_anomaly_pct"]

            alert_lines.append(f"  {icon} {row['region'].upper()}  —  {month_name}")
            alert_lines.append(f"     Food Security  : Phase {phase} — {IPC_LABELS[phase]}")
            alert_lines.append(f"     WFP Action     : {IPC_ACTIONS[phase]}")
            alert_lines.append(f"     Food Prices    : {price_dir} ({row['pred_price_change_pct']:+.1f}%)")
            alert_lines.append(f"     Rainfall       : {rain_anom:+.1f}% vs historical average")

            if row["is_anomaly"] == 1:
                alert_lines.append(f"     ⚠️  ANOMALY FLAG: Unusual compound risk pattern detected")

            # Plain language farmer message (for voice SMS)
            if phase >= 2 or row["is_anomaly"] == 1:
                if rain_anom < -20:
                    weather_msg = "Rainfall is much lower than normal this period."
                elif rain_anom > 50:
                    weather_msg = "Heavy rainfall and flood risk detected."
                else:
                    weather_msg = "Weather conditions are unstable."

                if "Rising" in price_dir:
                    price_msg = "Food prices are expected to rise. Consider selling now or storing until the lean season."
                elif "Falling" in price_dir:
                    price_msg = "Food prices may fall. This may be a good time to buy and stock food."
                else:
                    price_msg = "Food prices are relatively stable."

                alert_lines.append(f"")
                alert_lines.append(f"     📢 FARMER VOICE ALERT (Twi/Dagbani/Ewe/Hausa):")
                alert_lines.append(f"     \"{weather_msg} {price_msg}\"")
                alert_lines.append(f"     [Translate & broadcast via Africa's Talking SMS gateway]")

            alert_lines.append("")

    alert_text = "\n".join(alert_lines)
    out_path = OUTPUTS / "farmer_alerts.txt"
    with open(out_path, "w") as f:
        f.write(alert_text)

    print(alert_text)
    log(f"\n  Saved → data/outputs/farmer_alerts.txt")


def generate_dashboard_summary(df, recent):
    separator("GENERATING DASHBOARD SUMMARY")

    lines = []
    lines.append("=" * 60)
    lines.append("  FLOODHUNGER GHANA — DASHBOARD SUMMARY")
    lines.append(f"  Data: {int(df['year'].min())}–{int(df['year'].max())}  |  {df['region'].nunique()} Regions")
    lines.append("=" * 60)
    lines.append("")

    # Overall IPC distribution
    lines.append("  IPC PHASE DISTRIBUTION (ALL HISTORY)")
    for phase, count in df["pred_ipc_phase"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        lines.append(f"    {IPC_COLORS[phase]} Phase {phase} — {IPC_LABELS[phase]:<12}: {count:>5,} months ({pct:4.1f}%)  {bar}")
    lines.append("")

    # Most anomalous regions
    lines.append("  TOP ANOMALOUS REGIONS (HISTORICAL)")
    top_anom = (
        df[df["is_anomaly"] == 1]
        .groupby("region")["is_anomaly"]
        .count()
        .sort_values(ascending=False)
        .head(5)
    )
    for region, count in top_anom.items():
        lines.append(f"    • {region}: {count} anomalous months")
    lines.append("")

    # Latest month snapshot
    max_year  = int(df["year"].max())
    max_month = int(df[df["year"] == max_year]["month"].max())
    latest    = df[(df["year"] == max_year) & (df["month"] == max_month)]
    month_name = datetime(max_year, max_month, 1).strftime("%B %Y")

    lines.append(f"  LATEST SNAPSHOT — {month_name}")
    for _, row in latest.sort_values("pred_ipc_phase", ascending=False).iterrows():
        icon = IPC_COLORS.get(int(row["pred_ipc_phase"]), "⚪")
        anom = "⚠️ ANOMALY" if row["is_anomaly"] == 1 else ""
        lines.append(
            f"    {icon} {row['region']:<20} Phase {int(row['pred_ipc_phase'])} — {row['pred_ipc_label']:<12} "
            f"Price: {row['pred_price_direction']:<12} {anom}"
        )

    lines.append("")
    lines.append("=" * 60)
    lines.append("  FILES GENERATED:")
    lines.append("    data/outputs/predictions_latest.csv   ← all predictions")
    lines.append("    data/outputs/alerts_wfp.csv           ← WFP alert feed")
    lines.append("    data/outputs/farmer_alerts.txt        ← farmer voice alerts")
    lines.append("    data/outputs/dashboard_summary.txt    ← this file")
    lines.append("    data/outputs/model_evaluation_report.txt")
    lines.append("    data/outputs/model1_confusion_matrix.png")
    lines.append("    data/outputs/model1_shap_importance.png")
    lines.append("    data/outputs/model2_feature_importance.png")
    lines.append("    data/outputs/model2_actual_vs_predicted.png")
    lines.append("    data/outputs/model3_anomaly_analysis.png")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)

    out_path = OUTPUTS / "dashboard_summary.txt"
    with open(out_path, "w") as f:
        f.write(summary)
    log(f"Saved → data/outputs/dashboard_summary.txt")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  FLOODHUNGER GHANA — PREDICT & ALERT")
    print("=" * 60)

    (df, ipc_model, feature_cols, idx_to_phase,
     price_model, reg_features,
     anomaly_model, anomaly_scaler, anomaly_feats) = load_everything()

    df = run_predictions(
        df, ipc_model, feature_cols, idx_to_phase,
        price_model, reg_features,
        anomaly_model, anomaly_scaler, anomaly_feats
    )

    # Save full predictions
    df.to_csv(OUTPUTS / "predictions_latest.csv", index=False)
    log(f"Full predictions saved → data/outputs/predictions_latest.csv")

    alerts, recent = generate_alerts(df)
    generate_farmer_alerts(recent)
    generate_dashboard_summary(df, recent)

    print()
    print("=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print()
    print("  When you get real data from HDX / ACLED:")
    print("  1. Drop new CSVs into data/raw/")
    print("  2. Re-run: python 01_clean_and_merge.py")
    print("             python 02_feature_engineering.py")
    print("             python 03_train_models.py")
    print("             python 04_predict_and_alert.py")
    print("=" * 60)
