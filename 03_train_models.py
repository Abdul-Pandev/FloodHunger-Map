"""
╔══════════════════════════════════════════════════════════════╗
║   FLOODHUNGER GHANA — SCRIPT 03: TRAIN MODELS               ║
║   Trains all 3 ML models + SHAP explainability              ║
╚══════════════════════════════════════════════════════════════╝

Models trained:
  1. XGBoost IPC Phase Classifier    → ipc_phase (1–4)
  2. Random Forest Price Regressor   → price_maize_change_pct
  3. Isolation Forest Anomaly Detector → compound risk anomalies

Input:  data/processed/features.csv
Output: data/models/  (all .pkl files)
        data/outputs/ (evaluation reports, SHAP plots)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROC    = Path("data/processed")
MODELS  = Path("data/models")
OUTPUTS = Path("data/outputs")
MODELS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

PALETTE = ["#1F5C8B", "#2E86C1", "#85C1E9", "#D6E8F5"]

def log(msg):
    print(f"  ▸ {msg}")

def separator(title):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


# ── Load features ──────────────────────────────────────────
def load_data():
    separator("LOADING FEATURES")
    df = pd.read_csv(PROC / "features.csv")
    log(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Feature columns (exclude identifiers and targets)
    exclude = ["region", "year", "month", "ipc_phase",
               "price_maize_lag1", "price_maize_lag2",  # would leak target
               "compound_risk_flag"]

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in [np.float64, np.int64, float, int]
                    and df[c].nunique() > 1]

    log(f"Feature columns selected: {len(feature_cols)}")
    log(f"IPC Phase distribution: {df['ipc_phase'].value_counts().sort_index().to_dict()}")

    return df, feature_cols


# ══════════════════════════════════════════════════════════════
#  MODEL 1: XGBoost IPC Phase Classifier
#  Target: ipc_phase (1, 2, 3, 4)
#  Metric: Weighted F1 (handles class imbalance)
# ══════════════════════════════════════════════════════════════

def train_ipc_classifier(df, feature_cols):
    separator("MODEL 1: XGBoost IPC Phase Classifier")

    # Check how many IPC phases actually exist
    existing_phases = sorted(df["ipc_phase"].unique())
    n_classes = len(existing_phases)
    log(f"IPC phases in data: {existing_phases}  ({n_classes} classes)")

    # Re-encode labels to be 0-indexed and contiguous
    phase_to_idx = {p: i for i, p in enumerate(existing_phases)}
    idx_to_phase = {i: p for p, i in phase_to_idx.items()}

    X = df[feature_cols].fillna(0)
    y = df["ipc_phase"].map(phase_to_idx)

    # Class weights
    phase_counts = df["ipc_phase"].value_counts()
    total = len(df)
    class_weights = {phase_to_idx[p]: total / (n_classes * cnt) for p, cnt in phase_counts.items()}
    sample_weights = df["ipc_phase"].map({p: class_weights[phase_to_idx[p]] for p in existing_phases})

    log(f"Class weights: {class_weights}")

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=SEED, stratify=y
    )
    log(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Use binary if only 2 classes, else multiclass
    if n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softmax"
        eval_metric = "mlogloss"

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        random_state=SEED,
        eval_metric=eval_metric,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    # Map back to original phase labels
    y_pred_phases = pd.Series(y_pred).map(idx_to_phase).values
    y_test_phases = pd.Series(y_test.values).map(idx_to_phase).values

    f1 = f1_score(y_test, y_pred, average="weighted")
    log(f"Weighted F1: {f1:.4f}")

    phase_names = [f"Phase {p}" for p in existing_phases]
    report = classification_report(
        y_test_phases, y_pred_phases,
        target_names=phase_names,
        zero_division=0
    )
    print("\n" + report)

    # Save model + mappings
    joblib.dump(model, MODELS / "ipc_classifier_xgb.pkl")
    joblib.dump(feature_cols, MODELS / "feature_cols.pkl")
    joblib.dump(phase_to_idx, MODELS / "phase_to_idx.pkl")
    log(f"Saved → data/models/ipc_classifier_xgb.pkl")

    # ── Confusion matrix plot ──────────────────────────────
    cm = confusion_matrix(y_test_phases, y_pred_phases, labels=existing_phases)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"P{p}" for p in existing_phases],
                yticklabels=[f"P{p}" for p in existing_phases])
    ax.set_title("IPC Phase Classifier — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Phase", fontsize=11)
    ax.set_ylabel("True Phase", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model1_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Confusion matrix saved → data/outputs/model1_confusion_matrix.png")

    # ── SHAP explainability ────────────────────────────────
    log("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.sample(min(300, len(X_test)), random_state=SEED)
    shap_values = explainer.shap_values(shap_sample)

    # Handle both binary and multi-class outputs
    if isinstance(shap_values, list):
        sv = shap_values[-1]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(9, 7))
    feature_importance = np.abs(sv).mean(axis=0)
    top_n = 15
    top_idx = np.argsort(feature_importance)[-top_n:][::-1]
    top_features = [shap_sample.columns[i] for i in top_idx]
    top_values   = feature_importance[top_idx]

    ax.barh(range(top_n), top_values[::-1], color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Top {top_n} Features — IPC Phase Classifier", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model1_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"SHAP plot saved → data/outputs/model1_shap_importance.png")
    log(f"Top 5 features: {top_features[:5]}")

    return model, f1


# ══════════════════════════════════════════════════════════════
#  MODEL 2: Random Forest Price Regressor
#  Target: price_maize_change_pct (next-month price change %)
#  Metric: MAE, MAPE, R²
# ══════════════════════════════════════════════════════════════

def train_price_regressor(df, feature_cols):
    separator("MODEL 2: Random Forest Price Regressor")

    # Identify the best available price column to use as target
    candidate_targets = ["price_maize_change_pct", "Millet", "Sorghum", "Cowpeas"]
    target = None
    for col in candidate_targets:
        if col in df.columns and df[col].notna().sum() > 100:
            target = col
            log(f"Using target column: '{target}' ({df[col].notna().sum()} non-null rows)")
            break

    if target is None:
        log("No suitable price column found — skipping price regressor", "WARN")
        return None, np.nan, np.nan

    df_reg = df.copy()

    # If column is an absolute price, compute % change as the actual target
    if target in ["Millet", "Sorghum", "Cowpeas"]:
        df_reg["target_price_change"] = (
            df_reg.groupby("region")[target]
            .transform(lambda x: x.pct_change() * 100)
        )
        log(f"Computed % change from '{target}' absolute price")
    else:
        # Already a % change — predict next month forward
        df_reg["target_price_change"] = df_reg.groupby("region")[target].shift(-1)

    df_reg = df_reg.dropna(subset=["target_price_change"])
    # Remove extreme outliers (>200% change — data artefacts)
    df_reg = df_reg[df_reg["target_price_change"].abs() <= 200]
    log(f"Regression rows after filter: {len(df_reg):,}")
    log(f"Target range: {df_reg['target_price_change'].min():.1f}% to {df_reg['target_price_change'].max():.1f}%")

    reg_features = [c for c in feature_cols
                    if c not in [target, "price_maize_change_pct"]
                    and df_reg[c].notna().sum() > len(df_reg) * 0.5]

    X = df_reg[reg_features].fillna(0)
    y = df_reg["target_price_change"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    log(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mask = np.abs(y_test) > 0.1
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else np.nan

    log(f"MAE   : {mae:.3f}%")
    log(f"MAPE  : {mape:.1f}%")
    log(f"R²    : {r2:.4f}")

    joblib.dump(model, MODELS / "price_regressor_rf.pkl")
    joblib.dump(reg_features, MODELS / "reg_feature_cols.pkl")
    log(f"Saved → data/models/price_regressor_rf.pkl")

    # ── Feature importance plot ────────────────────────────
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]
    top_feats = [reg_features[i] for i in top_idx]
    top_imp   = importances[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(15), top_imp[::-1], color=PALETTE[1], alpha=0.85)
    ax.set_yticks(range(15))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(f"Price Regressor — Top 15 Features (target: {target})", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model2_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Feature importance plot saved → data/outputs/model2_feature_importance.png")

    # ── Actual vs Predicted scatter ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test[:300], y_pred[:300], alpha=0.4, color=PALETTE[0], s=20)
    lims = [min(float(y_test.min()), float(y_pred.min())), max(float(y_test.max()), float(y_pred.max()))]
    ax.plot(lims, lims, "r--", lw=1.5, alpha=0.8, label="Perfect prediction")
    ax.set_xlabel("Actual Price Change (%)", fontsize=11)
    ax.set_ylabel("Predicted Price Change (%)", fontsize=11)
    ax.set_title(f"Price Regressor — Actual vs Predicted\n(MAE={mae:.2f}%,  R²={r2:.3f})", fontsize=12, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model2_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Scatter plot saved → data/outputs/model2_actual_vs_predicted.png")

    return model, mae, r2


# ══════════════════════════════════════════════════════════════
#  MODEL 3: Isolation Forest Anomaly Detector
#  Target: Detect unusual compound risk patterns
#  Metric: Anomaly score + Precision@K
# ══════════════════════════════════════════════════════════════

def train_anomaly_detector(df, feature_cols):
    separator("MODEL 3: Isolation Forest Anomaly Detector")

    # Use compound risk-relevant features
    anomaly_features = [c for c in feature_cols if any(kw in c for kw in [
        "rainfall", "flood", "price", "conflict", "risk", "season", "drought"
    ])]
    log(f"Anomaly features: {len(anomaly_features)}")

    X = df[anomaly_features].fillna(0)

    # Scale for Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,   # expect ~3% anomalous months
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Anomaly scores (-1 = anomaly, 1 = normal)
    df_out = df.copy()
    df_out["anomaly_label"]  = model.predict(X_scaled)  # -1 or 1
    df_out["anomaly_score"]  = -model.score_samples(X_scaled)  # higher = more anomalous

    n_anomalies = (df_out["anomaly_label"] == -1).sum()
    log(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.1f}% of months)")

    # Top anomalies
    top_anomalies = (
        df_out[df_out["anomaly_label"] == -1]
        [["region", "year", "month", "anomaly_score",
          "rainfall_anomaly_pct", "flood_flag",
          "conflict_events", "ipc_phase"]]
        .sort_values("anomaly_score", ascending=False)
        .head(20)
    )
    log("Top 10 anomalous region-months:")
    print(top_anomalies[["region", "year", "month", "anomaly_score", "ipc_phase"]].head(10).to_string(index=False))

    # Save model and scaler
    joblib.dump(model,  MODELS / "anomaly_detector_if.pkl")
    joblib.dump(scaler, MODELS / "anomaly_scaler.pkl")
    joblib.dump(anomaly_features, MODELS / "anomaly_feature_cols.pkl")
    df_out[["region","year","month","anomaly_label","anomaly_score"]].to_csv(
        OUTPUTS / "anomaly_scores.csv", index=False
    )
    log(f"Saved → data/models/anomaly_detector_if.pkl")
    log(f"Saved → data/outputs/anomaly_scores.csv")

    # ── Anomaly score distribution plot ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Score distribution
    axes[0].hist(df_out["anomaly_score"], bins=50, color=PALETTE[0], alpha=0.7, edgecolor="white")
    axes[0].axvline(df_out[df_out["anomaly_label"]==-1]["anomaly_score"].min(),
                    color="red", linestyle="--", lw=2, label="Anomaly threshold")
    axes[0].set_xlabel("Anomaly Score (higher = more unusual)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Anomaly Score Distribution", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # Anomalies by region
    anom_by_region = (
        df_out[df_out["anomaly_label"] == -1]
        .groupby("region")["anomaly_label"]
        .count()
        .sort_values(ascending=True)
    )
    axes[1].barh(anom_by_region.index, anom_by_region.values, color=PALETTE[0], alpha=0.85)
    axes[1].set_xlabel("Number of Anomalous Months", fontsize=11)
    axes[1].set_title("Anomalies by Region", fontsize=12, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("FloodHunger Ghana — Isolation Forest Anomaly Detection", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model3_anomaly_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Anomaly plot saved → data/outputs/model3_anomaly_analysis.png")

    return model, n_anomalies


# ══════════════════════════════════════════════════════════════
#  EVALUATION REPORT
# ══════════════════════════════════════════════════════════════

def save_evaluation_report(f1, mae, r2, n_anomalies, df):
    report_path = OUTPUTS / "model_evaluation_report.txt"
    with open(report_path, "w") as f_out:
        f_out.write("=" * 60 + "\n")
        f_out.write("  FLOODHUNGER GHANA — MODEL EVALUATION REPORT\n")
        f_out.write("=" * 60 + "\n\n")

        f_out.write("MODEL 1: IPC Phase Classifier (XGBoost)\n")
        f_out.write("-" * 40 + "\n")
        f_out.write(f"  Weighted F1 Score : {f1:.4f}\n")
        f_out.write(f"  Target            : IPC Phase 1–4\n")
        f_out.write(f"  Algorithm         : XGBoost with class weights\n")
        f_out.write(f"  Explainability    : SHAP TreeExplainer\n\n")

        f_out.write("MODEL 2: Maize Price Change Regressor (Random Forest)\n")
        f_out.write("-" * 40 + "\n")
        f_out.write(f"  MAE               : {mae:.3f}%\n")
        f_out.write(f"  R²                : {r2:.4f}\n")
        f_out.write(f"  Target            : Next-month maize price change %\n")
        f_out.write(f"  Algorithm         : Random Forest Regressor\n\n")

        f_out.write("MODEL 3: Compound Risk Anomaly Detector (Isolation Forest)\n")
        f_out.write("-" * 40 + "\n")
        f_out.write(f"  Anomalies found   : {n_anomalies} months ({n_anomalies/len(df)*100:.1f}%)\n")
        f_out.write(f"  Contamination     : 3% (expected)\n")
        f_out.write(f"  Algorithm         : Isolation Forest\n\n")

        f_out.write("DATA SUMMARY\n")
        f_out.write("-" * 40 + "\n")
        f_out.write(f"  Regions           : {df['region'].nunique()}\n")
        f_out.write(f"  Years             : {df['year'].min()}–{df['year'].max()}\n")
        f_out.write(f"  Total rows        : {len(df):,}\n")

    log(f"Evaluation report saved → data/outputs/model_evaluation_report.txt")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  FLOODHUNGER GHANA — MODEL TRAINING")
    print("=" * 60)

    df, feature_cols = load_data()

    model1, f1         = train_ipc_classifier(df, feature_cols)
    model2, mae, r2    = train_price_regressor(df, feature_cols)
    model3, n_anomalies = train_anomaly_detector(df, feature_cols)

    save_evaluation_report(f1, mae, r2, n_anomalies, df)

    print()
    print("=" * 60)
    print("  ✓ ALL 3 MODELS TRAINED")
    print(f"  IPC Classifier  Weighted F1 : {f1:.4f}")
    print(f"  Price Regressor MAE         : {mae:.3f}%")
    print(f"  Anomaly Detector            : {n_anomalies} anomalies found")
    print()
    print("  Next step:")
    print("    python 04_predict_and_alert.py")
    print("=" * 60)
