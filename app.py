"""
FloodHunger Ghana — Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="FloodHunger Ghana",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────
BLUE   = "#1F5C8B"
DBLUE  = "#0D3C6E"
LBLUE  = "#D6E8F5"
RED    = "#E74C3C"
ORANGE = "#E67E22"
YELLOW = "#F4D03F"
GREEN  = "#2ECC71"

IPC_LABELS  = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency"}
IPC_COLORS  = {1: GREEN, 2: YELLOW, 3: ORANGE, 4: RED}
IPC_ICONS   = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
IPC_ACTIONS = {
    1: "No action needed — continue monitoring",
    2: "Alert field teams — pre-position supplies",
    3: "Deploy food stocks and vouchers immediately",
    4: "EMERGENCY — deploy cash transfers NOW",
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 5px solid #1F5C8B;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 12px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #666; font-weight: 500; }
    .metric-card h1 { margin: 4px 0 0 0; font-size: 32px; font-weight: 700; color: #0D3C6E; }
    .metric-card p  { margin: 2px 0 0 0; font-size: 12px; color: #888; }

    .alert-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.08);
    }
    .alert-emergency { background: #FDEDEC; border-left: 5px solid #E74C3C; }
    .alert-crisis    { background: #FEF9E7; border-left: 5px solid #E67E22; }
    .alert-stressed  { background: #FFFDE7; border-left: 5px solid #F4D03F; }
    .alert-minimal   { background: #EAFAF1; border-left: 5px solid #2ECC71; }
    .alert-anomaly   { border: 2px dashed #E74C3C !important; }

    .stSelectbox label { font-weight: 600; color: #1F5C8B; }
    .stSlider label    { font-weight: 600; color: #1F5C8B; }

    h1 { color: #0D3C6E; }
    h2 { color: #1F5C8B; border-bottom: 2px solid #D6E8F5; padding-bottom: 6px; }
    h3 { color: #1F5C8B; }

    .wfp-banner {
        background: linear-gradient(135deg, #0D3C6E 0%, #1F5C8B 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .wfp-banner h1 { color: white; margin: 0; font-size: 28px; }
    .wfp-banner p  { color: #B0D0E8; margin: 6px 0 0 0; font-size: 14px; }

    .farmer-alert {
        background: #FFF8E1;
        border: 1px solid #FFD54F;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 12px;
    }
    .farmer-alert p { margin: 0; font-size: 15px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).parent
    df = pd.read_csv(base / "data/outputs/predictions_full.csv")
    df["month_name"] = df["month"].apply(lambda m: MONTH_NAMES[m-1])
    df["period"]     = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    return df

df = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloodHunger Ghana")
    st.markdown("**WFP Early Warning System**")
    st.markdown("---")

    st.markdown("### 🗺️ Filters")

    all_regions   = ["All Regions"] + sorted(df["region"].unique().tolist())
    sel_region    = st.selectbox("Region", all_regions)

    if sel_region == "All Regions":
        districts_avail = ["All Districts"] + sorted(df["district"].unique().tolist())
    else:
        districts_avail = ["All Districts"] + sorted(
            df[df["region"] == sel_region]["district"].unique().tolist()
        )
    sel_district = st.selectbox("District", districts_avail)

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.slider("Year Range", year_min, year_max, (2018, year_max))

    st.markdown("---")
    st.markdown("### 📊 View")
    page = st.radio("", [
        "🏠 Overview",
        "🗺️ District Alert Map",
        "📈 Trends",
        "🔍 District Deep Dive",
        "📢 Farmer Alerts",
        "🤖 Live Predictions",
    ])
    st.markdown("---")
    st.markdown(f"**Data:** {len(df):,} district-months")
    st.markdown(f"**Districts:** {df['district'].nunique()}")
    st.markdown(f"**Period:** {year_min}–{year_max}")


# ── Filter data ────────────────────────────────────────────
mask = (df["year"] >= sel_years[0]) & (df["year"] <= sel_years[1])
if sel_region != "All Regions":
    mask &= (df["region"] == sel_region)
if sel_district != "All Districts":
    mask &= (df["district"] == sel_district)

dff = df[mask].copy()

# Latest month for snapshot
max_year  = int(dff["year"].max())
max_month = int(dff[dff["year"] == max_year]["month"].max())
latest    = dff[(dff["year"] == max_year) & (dff["month"] == max_month)].copy()
month_str = datetime(max_year, max_month, 1).strftime("%B %Y")


# ══════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown(f"""
    <div class="wfp-banner">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div style="display:flex;align-items:center;gap:16px;">
                <!-- WFP Logo SVG -->
                <div style="background:white;border-radius:8px;padding:6px 10px;display:flex;align-items:center;">
                    <svg width="52" height="52" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="50" cy="50" r="48" fill="#009FDA" stroke="white" stroke-width="2"/>
                        <text x="50" y="38" text-anchor="middle" fill="white" font-size="18" font-weight="bold" font-family="Arial">WFP</text>
                        <!-- wheat stalk left -->
                        <line x1="30" y1="75" x2="30" y2="50" stroke="white" stroke-width="2"/>
                        <ellipse cx="30" cy="50" rx="4" ry="7" fill="white" transform="rotate(-15 30 50)"/>
                        <ellipse cx="30" cy="58" rx="4" ry="7" fill="white" transform="rotate(15 30 58)"/>
                        <ellipse cx="24" cy="54" rx="4" ry="7" fill="white" transform="rotate(-30 24 54)"/>
                        <!-- wheat stalk center -->
                        <line x1="50" y1="78" x2="50" y2="50" stroke="white" stroke-width="2"/>
                        <ellipse cx="50" cy="50" rx="4" ry="7" fill="white"/>
                        <ellipse cx="50" cy="58" rx="4" ry="7" fill="white" transform="rotate(15 50 58)"/>
                        <ellipse cx="44" cy="54" rx="4" ry="7" fill="white" transform="rotate(-30 44 54)"/>
                        <!-- wheat stalk right -->
                        <line x1="70" y1="75" x2="70" y2="50" stroke="white" stroke-width="2"/>
                        <ellipse cx="70" cy="50" rx="4" ry="7" fill="white" transform="rotate(15 70 50)"/>
                        <ellipse cx="70" cy="58" rx="4" ry="7" fill="white" transform="rotate(-15 70 58)"/>
                        <ellipse cx="76" cy="54" rx="4" ry="7" fill="white" transform="rotate(30 76 54)"/>
                        <text x="50" y="92" text-anchor="middle" fill="white" font-size="9" font-family="Arial">UN World Food Programme</text>
                    </svg>
                </div>
                <div>
                    <h1 style="color:white;margin:0;font-size:26px;">🌊 FloodHunger Ghana</h1>
                    <p style="color:#B0D0E8;margin:4px 0 0;font-size:13px;">
                        Flood-Driven Food Insecurity Early Warning System &nbsp;·&nbsp;
                        {df['district'].nunique()} Districts &nbsp;·&nbsp;
                        {year_min}–{year_max} &nbsp;·&nbsp;
                        WFP / FEWS NET Aligned
                    </p>
                </div>
            </div>
            <!-- Blossom Academy branding -->
            <div style="text-align:right;">
                <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:8px 14px;border:1px solid rgba(255,255,255,0.25);">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <svg width="28" height="28" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="30" cy="30" r="28" fill="#FF6B35"/>
                            <circle cx="30" cy="22" r="7" fill="white"/>
                            <ellipse cx="18" cy="28" rx="5" ry="7" fill="#FFD700" transform="rotate(-30 18 28)"/>
                            <ellipse cx="42" cy="28" rx="5" ry="7" fill="#FFD700" transform="rotate(30 42 28)"/>
                            <ellipse cx="22" cy="40" rx="5" ry="7" fill="#FFD700" transform="rotate(20 22 40)"/>
                            <ellipse cx="38" cy="40" rx="5" ry="7" fill="#FFD700" transform="rotate(-20 38 40)"/>
                            <circle cx="30" cy="30" r="5" fill="#FF6B35"/>
                        </svg>
                        <div>
                            <div style="color:white;font-weight:700;font-size:13px;line-height:1.2;">Blossom Academy</div>
                            <div style="color:#FFD700;font-size:10px;">Ghana · Data for Good</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top metrics ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    crisis_pct = (dff["pred_ipc_phase"] >= 3).mean() * 100
    flood_pct  = dff["flood_flag"].mean() * 100
    anom_n     = dff["is_anomaly"].sum()
    avg_risk   = dff["compound_risk_score"].mean()

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Districts in Crisis / Emergency</h3>
            <h1>{crisis_pct:.1f}%</h1>
            <p>of district-months in selected period</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Flood-Flagged Months</h3>
            <h1>{flood_pct:.1f}%</h1>
            <p>rainfall anomaly &gt; 50% above average</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Anomalous Risk Months</h3>
            <h1>{int(anom_n):,}</h1>
            <p>unusual compound risk patterns detected</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Compound Risk Score</h3>
            <h1>{avg_risk:.2f}</h1>
            <p>higher = more compounding pressures</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts row 1 ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### IPC Phase Distribution")
        phase_counts = dff["pred_ipc_phase"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [IPC_COLORS.get(p, BLUE) for p in phase_counts.index]
        bars = ax.bar(
            [f"P{p}\n{IPC_LABELS[p]}" for p in phase_counts.index],
            phase_counts.values, color=colors, alpha=0.9, edgecolor="white", linewidth=1.5
        )
        for bar, val in zip(bars, phase_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{val:,}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("District-Month Count")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Average IPC Phase by Region")
        ipc_reg = dff.groupby("region")["pred_ipc_phase"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        c_reg = [RED if v >= 3 else ORANGE if v >= 2.5 else BLUE for v in ipc_reg.values]
        ax.barh(ipc_reg.index, ipc_reg.values, color=c_reg, alpha=0.85)
        ax.axvline(2, color=ORANGE, lw=1.5, ls="--", alpha=0.7, label="Stressed")
        ax.axvline(3, color=RED,    lw=1.5, ls="--", alpha=0.7, label="Crisis")
        ax.set_xlabel("Mean IPC Phase")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

    # ── Charts row 2 ─────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Rainfall Anomaly by Year")
        rain_yr = dff.groupby("year")["rainfall_anomaly_pct"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_yr = [BLUE if v >= 0 else RED for v in rain_yr.values]
        ax.bar(rain_yr.index, rain_yr.values, color=colors_yr, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("Anomaly (%)")
        ax.set_xlabel("Year")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("### Seasonal Food Security Pattern")
        ipc_m = dff.groupby("month")["pred_ipc_phase"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ipc_m.index, ipc_m.values, color=BLUE, lw=2.5, marker="o", ms=6)
        ax.fill_between(ipc_m.index, ipc_m.values, 1, alpha=0.1, color=BLUE)
        ax.axhline(2, color=ORANGE, lw=1.5, ls="--", alpha=0.7, label="Stressed")
        ax.axhline(3, color=RED,    lw=1.5, ls="--", alpha=0.7, label="Crisis")
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=8)
        ax.set_ylabel("Avg IPC Phase")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════
#  PAGE 2: DISTRICT ALERT MAP
# ══════════════════════════════════════════════════════════
elif page == "🗺️ District Alert Map":

    st.markdown(f"## 🗺️ District Alert Snapshot — {month_str}")
    st.markdown(f"Latest predictions for all districts. Use sidebar to filter by region.")

    # Sort by risk
    snapshot = latest.sort_values(["pred_ipc_phase","is_anomaly"], ascending=[False, False])

    # Summary bar
    phase_dist = snapshot["pred_ipc_phase"].value_counts().sort_index()
    c1, c2, c3, c4 = st.columns(4)
    for col, (ph, cnt) in zip([c1, c2, c3, c4], phase_dist.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {IPC_COLORS[ph]}">
                <h3>{IPC_ICONS[ph]} Phase {ph} — {IPC_LABELS[ph]}</h3>
                <h1>{cnt}</h1>
                <p>districts this month</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Alert cards
    for _, row in snapshot.iterrows():
        ph     = int(row["pred_ipc_phase"])
        anom   = row["is_anomaly"] == 1
        cls    = {1:"minimal", 2:"stressed", 3:"crisis", 4:"emergency"}.get(ph, "minimal")
        a_cls  = " alert-anomaly" if anom else ""
        rain   = row["rainfall_anomaly_pct"]
        rain_s = f"{'⬆' if rain > 0 else '⬇'} {abs(rain):.0f}% vs avg"
        price  = row.get("pred_price_direction", "→ Stable")
        conf   = row["pred_ipc_confidence"]
        anom_s = "  ⚠️ ANOMALY DETECTED" if anom else ""

        st.markdown(f"""
        <div class="alert-card alert-{cls}{a_cls}">
            <strong>{IPC_ICONS[ph]} {row['district']}</strong>
            &nbsp;&nbsp;<span style="color:#666; font-size:13px">{row['region']}</span>
            &nbsp;&nbsp;<span style="background:{IPC_COLORS[ph]};color:white;
                border-radius:4px;padding:2px 8px;font-size:12px;font-weight:bold">
                Phase {ph} — {IPC_LABELS[ph]}</span>
            &nbsp;<span style="color:#888;font-size:12px">Confidence: {conf:.2f}</span>
            <span style="color:#E74C3C;font-weight:bold;font-size:12px">{anom_s}</span>
            <br/>
            <span style="font-size:13px;color:#444">
                🌧 Rain: {rain_s} &nbsp;|&nbsp;
                🌽 Food price: {price} &nbsp;|&nbsp;
                ⚡ Risk score: {row['compound_risk_score']:.2f} &nbsp;|&nbsp;
                🏛 Action: {IPC_ACTIONS[ph]}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE 3: TRENDS
# ══════════════════════════════════════════════════════════
elif page == "📈 Trends":

    st.markdown("## 📈 Historical Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### IPC Phase Over Time")
        ipc_time = dff.groupby(["year","month"])["pred_ipc_phase"].mean().reset_index()
        ipc_time["date_n"] = ipc_time["year"] + (ipc_time["month"] - 1) / 12
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ipc_time["date_n"], ipc_time["pred_ipc_phase"],
                color=BLUE, lw=2, alpha=0.9)
        ax.fill_between(ipc_time["date_n"], ipc_time["pred_ipc_phase"], 1,
                        alpha=0.1, color=BLUE)
        ax.axhline(2, color=ORANGE, lw=1, ls="--", alpha=0.6, label="Stressed")
        ax.axhline(3, color=RED,    lw=1, ls="--", alpha=0.6, label="Crisis")
        ax.set_ylabel("Avg IPC Phase")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Food Price Change Over Time")
        if "price_change_pct" in dff.columns:
            price_time = dff.groupby(["year","month"])["price_change_pct"].mean().reset_index()
            price_time["date_n"] = price_time["year"] + (price_time["month"] - 1) / 12
            price_time = price_time.dropna(subset=["price_change_pct"])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(price_time["date_n"], price_time["price_change_pct"],
                    color=ORANGE, lw=2)
            ax.axhline(0,  color="black", lw=0.8, ls="--")
            ax.axhline(20, color=RED,     lw=1.2, ls="--", alpha=0.6, label="Shock (+20%)")
            ax.fill_between(price_time["date_n"], price_time["price_change_pct"], 0,
                            where=price_time["price_change_pct"] > 0,
                            alpha=0.15, color=RED, label="Price rise")
            ax.set_ylabel("Monthly Change (%)")
            ax.set_xlabel("Year")
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#F8FAFC")
            fig.patch.set_facecolor("#F8FAFC")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Price change data not available for this selection.")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Flood Months per Year")
        flood_yr = dff.groupby("year")["flood_flag"].sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(flood_yr.index, flood_yr.values, color=BLUE, alpha=0.85, edgecolor="white")
        ax.set_ylabel("Flood District-Months")
        ax.set_xlabel("Year")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("### Conflict Events per Year")
        if "conflict_events" in dff.columns:
            conf_yr = dff.groupby("year")["conflict_events"].sum()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(conf_yr.index, conf_yr.values, color=RED, lw=2.5, marker="o", ms=5)
            ax.fill_between(conf_yr.index, conf_yr.values, alpha=0.12, color=RED)
            ax.set_ylabel("Total Events")
            ax.set_xlabel("Year")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#F8FAFC")
            fig.patch.set_facecolor("#F8FAFC")
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════
#  PAGE 4: DISTRICT DEEP DIVE
# ══════════════════════════════════════════════════════════
elif page == "🔍 District Deep Dive":

    st.markdown("## 🔍 District Deep Dive")

    if sel_district == "All Districts":
        st.info("👈 Select a specific district from the sidebar to deep dive.")
    else:
        dist_df = df[df["district"] == sel_district].sort_values(["year","month"])
        dist_df["date_n"] = dist_df["year"] + (dist_df["month"] - 1) / 12

        # Header
        latest_dist = dist_df.iloc[-1]
        ph  = int(latest_dist["pred_ipc_phase"])
        st.markdown(f"""
        <div style="background:{IPC_COLORS[ph]}22;border-left:5px solid {IPC_COLORS[ph]};
             border-radius:10px;padding:16px 22px;margin-bottom:20px">
            <h2 style="margin:0;color:{DBLUE}">{IPC_ICONS[ph]} {sel_district}</h2>
            <p style="margin:4px 0 0;color:#555">{latest_dist['region']} Region &nbsp;·&nbsp;
            Latest: <strong>Phase {ph} — {IPC_LABELS[ph]}</strong> ({month_str}) &nbsp;·&nbsp;
            Confidence: {latest_dist['pred_ipc_confidence']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Flood Months (total)", int(dist_df["flood_flag"].sum()))
        with c2:
            st.metric("Anomalous Months", int(dist_df["is_anomaly"].sum()))
        with c3:
            st.metric("Avg Risk Score", f"{dist_df['compound_risk_score'].mean():.2f}")

        st.markdown("---")

        # IPC over time
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### IPC Phase History")
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_ts = [IPC_COLORS.get(p, BLUE) for p in dist_df["pred_ipc_phase"]]
            ax.scatter(dist_df["date_n"], dist_df["pred_ipc_phase"],
                       c=colors_ts, s=20, zorder=3, alpha=0.8)
            ax.plot(dist_df["date_n"], dist_df["pred_ipc_phase"],
                    color=BLUE, lw=1.2, alpha=0.5)
            ax.axhline(2, color=ORANGE, lw=1, ls="--", alpha=0.6, label="Stressed")
            ax.axhline(3, color=RED,    lw=1, ls="--", alpha=0.6, label="Crisis")
            ax.set_yticks([1,2,3,4])
            ax.set_yticklabels(["P1\nMinimal","P2\nStressed","P3\nCrisis","P4\nEmergency"])
            ax.set_xlabel("Year")
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#F8FAFC")
            fig.patch.set_facecolor("#F8FAFC")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Rainfall Anomaly History")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(dist_df["date_n"], dist_df["rainfall_anomaly_pct"],
                   color=[BLUE if v >= 0 else RED for v in dist_df["rainfall_anomaly_pct"]],
                   alpha=0.75, width=0.06)
            ax.axhline(50,  color=BLUE, lw=1.2, ls="--", alpha=0.6, label="Flood flag (+50%)")
            ax.axhline(-40, color=RED,  lw=1.2, ls="--", alpha=0.6, label="Drought flag (-40%)")
            ax.axhline(0,   color="black", lw=0.8)
            ax.set_ylabel("Rainfall Anomaly (%)")
            ax.set_xlabel("Year")
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#F8FAFC")
            fig.patch.set_facecolor("#F8FAFC")
            st.pyplot(fig)
            plt.close()

        # Top risk months
        st.markdown("### ⚠️ Highest Risk Months")
        top_risk = (
            dist_df[["year","month","pred_ipc_phase","pred_ipc_label","rainfall_anomaly_pct",
                      "pred_price_direction","is_anomaly","compound_risk_score"]]
            .sort_values("compound_risk_score", ascending=False)
            .head(10)
        )
        top_risk["month"] = top_risk["month"].apply(lambda m: MONTH_NAMES[m-1])
        top_risk["Anomaly"] = top_risk["is_anomaly"].apply(lambda x: "⚠️" if x == 1 else "")
        top_risk = top_risk.drop(columns=["is_anomaly"])
        top_risk.columns = ["Year","Month","IPC Phase","IPC Label","Rain Anomaly (%)","Price Direction","Risk Score","Anomaly"]
        st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
#  PAGE 5: FARMER ALERTS
# ══════════════════════════════════════════════════════════
elif page == "📢 Farmer Alerts":

    st.markdown(f"## 📢 Farmer Voice Alert Bulletin — {month_str}")
    st.markdown("*Ready for broadcast via Africa's Talking SMS gateway in Twi, Dagbani, Ewe, Hausa*")
    st.markdown("---")

    high_risk = latest[latest["pred_ipc_phase"] >= 3].sort_values("pred_ipc_phase", ascending=False)
    anomalous = latest[(latest["pred_ipc_phase"] < 3) & (latest["is_anomaly"] == 1)]
    stressed  = latest[latest["pred_ipc_phase"] == 2]

    if len(high_risk) == 0 and len(anomalous) == 0:
        st.success("✅ No high-risk districts this month. All districts in Minimal or Stressed phase.")

    if len(high_risk) > 0:
        st.markdown(f"### 🔴 {len(high_risk)} District(s) — Crisis or Emergency")
        for _, row in high_risk.iterrows():
            ph    = int(row["pred_ipc_phase"])
            rain  = row["rainfall_anomaly_pct"]
            price = str(row.get("pred_price_direction", "→ Stable"))

            if row["flood_flag"] == 1:
                weather_msg = "Heavy flooding has been detected in your area this season."
            elif rain < -30:
                weather_msg = "There is a severe dry spell in your area. Crops are under stress."
            else:
                weather_msg = "Weather conditions are unstable and unpredictable this season."

            if "Rising" in price:
                price_msg = "Food prices are rising sharply. Buy or store food now before prices go higher."
            elif "Falling" in price:
                price_msg = "Food prices are expected to fall. This may be a good time to buy in bulk."
            else:
                price_msg = "Food prices are currently stable in your local market."

            st.markdown(f"""
            <div class="alert-card alert-{'emergency' if ph == 4 else 'crisis'}">
                <strong>{IPC_ICONS[ph]} {row['district']} — {row['region']}</strong><br/>
                <span style="font-size:13px;color:#666">
                    Phase {ph}: {IPC_LABELS[ph]} &nbsp;·&nbsp;
                    Confidence: {row['pred_ipc_confidence']:.2f} &nbsp;·&nbsp;
                    WFP Action: {IPC_ACTIONS[ph]}
                </span>
            </div>
            <div class="farmer-alert">
                <p>📢 <strong>Voice Alert (English):</strong><br/>
                {weather_msg} {price_msg}
                Contact your district food and agriculture officer immediately.
                Help is available.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

    if len(anomalous) > 0:
        st.markdown(f"### ⚠️ {len(anomalous)} District(s) — Unusual Risk Pattern")
        for _, row in anomalous.iterrows():
            st.markdown(f"""
            <div class="alert-card alert-stressed alert-anomaly">
                <strong>⚠️ {row['district']} — {row['region']}</strong><br/>
                <span style="font-size:13px;color:#666">
                    Phase {int(row['pred_ipc_phase'])}: {row['pred_ipc_label']} but
                    unusual compound risk detected (score: {row['anomaly_score']:.3f}).
                    Monitor closely — situation may escalate.
                </span>
            </div>
            """, unsafe_allow_html=True)

    if len(stressed) > 0:
        with st.expander(f"🟡 {len(stressed)} Districts in Stressed Phase (Phase 2)"):
            for _, row in stressed.iterrows():
                st.markdown(f"- **{row['district']}** ({row['region']}) — "
                            f"Confidence: {row['pred_ipc_confidence']:.2f} | "
                            f"Price: {row.get('pred_price_direction','N/A')}")

    st.markdown("---")
    st.markdown("### 📥 Download Alert Feed")
    csv = latest[[
        "district","region","pred_ipc_phase","pred_ipc_label",
        "pred_ipc_confidence","pred_ipc_action",
        "pred_price_direction","is_anomaly","compound_risk_score"
    ]].sort_values(["pred_ipc_phase"], ascending=False).to_csv(index=False)

    st.download_button(
        label="⬇️ Download WFP Alert CSV",
        data=csv,
        file_name=f"floodhunger_ghana_alerts_{max_year}_{max_month:02d}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════
#  PAGE 6: LIVE PREDICTIONS
# ══════════════════════════════════════════════════════════
elif page == "🤖 Live Predictions":

    st.markdown("## 🤖 Live IPC Phase Predictor")
    st.markdown(
        "Enter current field conditions below. The model will estimate the IPC food security "
        "phase for your scenario using the same logic as the trained pipeline."
    )

    # ── Simple rule-based prediction engine (mirrors model logic) ──────────
    def predict_ipc(rainfall_anom, price_change, flood_flag, conflict_events,
                    ndvi_anom, population_density, prev_ipc):
        """
        Heuristic scoring that mirrors the trained model's feature importance.
        Returns (phase: int, confidence: float, drivers: list[str])
        """
        score = 0.0
        drivers = []

        # Rainfall anomaly (most important)
        if rainfall_anom >= 50:
            score += 2.5
            drivers.append(f"🌊 Severe flood signal (+{rainfall_anom:.0f}% rainfall anomaly)")
        elif rainfall_anom <= -40:
            score += 2.0
            drivers.append(f"☀️ Drought signal ({rainfall_anom:.0f}% rainfall deficit)")
        elif rainfall_anom <= -20:
            score += 0.8
            drivers.append(f"🌤 Moderate dry spell ({rainfall_anom:.0f}%)")

        # Food price change
        if price_change >= 30:
            score += 2.0
            drivers.append(f"📈 Price shock (+{price_change:.0f}% food price increase)")
        elif price_change >= 15:
            score += 1.0
            drivers.append(f"📊 Elevated food prices (+{price_change:.0f}%)")
        elif price_change <= -10:
            score -= 0.5
            drivers.append(f"📉 Food prices falling ({price_change:.0f}%)")

        # Flood flag
        if flood_flag:
            score += 1.2
            drivers.append("🚨 Active flood event flagged")

        # Conflict
        if conflict_events >= 5:
            score += 1.5
            drivers.append(f"⚔️ High conflict activity ({conflict_events} events)")
        elif conflict_events >= 2:
            score += 0.6
            drivers.append(f"⚠️ Moderate conflict ({conflict_events} events)")

        # NDVI anomaly (vegetation health)
        if ndvi_anom <= -0.15:
            score += 1.0
            drivers.append(f"🌿 Vegetation stress (NDVI Δ {ndvi_anom:.2f})")
        elif ndvi_anom >= 0.10:
            score -= 0.3

        # Previous IPC phase (momentum)
        if prev_ipc >= 3:
            score += 0.8
            drivers.append(f"🔁 Carry-over from previous Phase {prev_ipc}")
        elif prev_ipc == 1:
            score -= 0.4

        # Population density pressure
        if population_density >= 200:
            score += 0.5
            drivers.append(f"👥 High population density ({population_density:.0f}/km²)")

        # Map score → IPC phase
        if score <= 0.5:
            phase = 1
        elif score <= 1.8:
            phase = 2
        elif score <= 3.5:
            phase = 3
        else:
            phase = 4

        # Confidence: higher when score is clearly in one zone
        zone_centers = {1: 0.0, 2: 1.2, 3: 2.7, 4: 4.5}
        dist_to_center = abs(score - zone_centers[phase])
        confidence = max(0.55, min(0.97, 0.97 - dist_to_center * 0.12))

        if not drivers:
            drivers.append("✅ Conditions within normal range")

        return phase, round(confidence, 2), drivers, round(score, 2)

    # ── Input form ──────────────────────────────────────────────────────────
    st.markdown("---")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.markdown("### 🌦 Climate & Environment")
        inp_rainfall  = st.slider("Rainfall Anomaly (%)",
                                  min_value=-80, max_value=150, value=0, step=5,
                                  help="% deviation from long-run average. +50 = flood zone, -40 = drought zone")
        inp_flood     = st.checkbox("Active Flood Event", value=False,
                                    help="Tick if field teams confirm active flooding")
        inp_ndvi      = st.slider("NDVI Anomaly (vegetation health)",
                                  min_value=-0.40, max_value=0.40, value=0.0, step=0.01,
                                  format="%.2f",
                                  help="Negative = stressed crops. Below -0.15 is severe.")

    with col_in2:
        st.markdown("### 🌽 Markets & Conflict")
        inp_price     = st.slider("Food Price Change (%)",
                                  min_value=-30, max_value=80, value=0, step=1,
                                  help="Month-on-month change in local food market prices")
        inp_conflict  = st.slider("Conflict Events (last 30 days)",
                                  min_value=0, max_value=20, value=0, step=1,
                                  help="Number of reported conflict incidents in district")
        inp_pop       = st.slider("Population Density (per km²)",
                                  min_value=10, max_value=500, value=80, step=10)
        inp_prev_ipc  = st.selectbox("Previous Month IPC Phase",
                                     options=[1, 2, 3, 4],
                                     format_func=lambda x: f"Phase {x} — {IPC_LABELS[x]}")

    st.markdown("---")

    # ── Run prediction ───────────────────────────────────────────────────────
    phase, conf, drivers, raw_score = predict_ipc(
        inp_rainfall, inp_price, inp_flood,
        inp_conflict, inp_ndvi, inp_pop, inp_prev_ipc
    )

    color  = IPC_COLORS[phase]
    icon   = IPC_ICONS[phase]
    label  = IPC_LABELS[phase]
    action = IPC_ACTIONS[phase]

    # ── Result card ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{color}22;border-left:6px solid {color};border-radius:12px;
         padding:22px 28px;margin-bottom:20px;box-shadow:0 2px 10px rgba(0,0,0,0.08)">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <div style="font-size:52px;line-height:1">{icon}</div>
            <div>
                <h2 style="margin:0;color:{DBLUE};font-size:22px;">
                    IPC Phase {phase} — {label}
                </h2>
                <p style="margin:4px 0 0;color:#555;font-size:14px;">
                    Model confidence: <strong>{conf:.0%}</strong> &nbsp;·&nbsp;
                    Raw risk score: <strong>{raw_score}</strong> &nbsp;·&nbsp;
                    Threshold crossed: <strong>Phase {phase}</strong>
                </p>
            </div>
        </div>
        <div style="margin-top:14px;background:rgba(255,255,255,0.6);
             border-radius:8px;padding:12px 16px;font-size:14px;color:#333">
            🏛 <strong>Recommended WFP Action:</strong> {action}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Driver breakdown ─────────────────────────────────────────────────────
    c_res1, c_res2 = st.columns([1, 1])

    with c_res1:
        st.markdown("### 🔍 Key Risk Drivers")
        for d in drivers:
            st.markdown(f"- {d}")

    with c_res2:
        st.markdown("### 📊 Risk Score Breakdown")
        fig, ax = plt.subplots(figsize=(5, 2.2))
        phase_thresholds = [0.5, 1.8, 3.5]
        phase_colors_bg  = [GREEN, YELLOW, ORANGE, RED]
        boundaries = [0, 0.5, 1.8, 3.5, 6.0]
        for i in range(4):
            ax.barh(0, boundaries[i+1] - boundaries[i],
                    left=boundaries[i], color=phase_colors_bg[i],
                    alpha=0.3, height=0.5)
            ax.text((boundaries[i] + boundaries[i+1]) / 2, -0.38,
                    f"P{i+1}", ha="center", fontsize=8, color="#555")
        ax.barh(0, min(raw_score, 6.0), left=0,
                color=color, alpha=0.85, height=0.3)
        ax.axvline(min(raw_score, 6.0), color=color, lw=2.5)
        ax.set_xlim(0, 6)
        ax.set_ylim(-0.6, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Composite Risk Score", fontsize=9)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

    # ── Scenario table ───────────────────────────────────────────────────────
    st.markdown("### 📋 Scenario Summary")
    scenario_data = {
        "Parameter": [
            "Rainfall Anomaly", "Food Price Change", "Active Flood",
            "Conflict Events", "NDVI Anomaly", "Population Density", "Prev. IPC Phase"
        ],
        "Value": [
            f"{inp_rainfall:+d}%", f"{inp_price:+d}%",
            "Yes 🚨" if inp_flood else "No ✅",
            str(inp_conflict), f"{inp_ndvi:+.2f}",
            f"{inp_pop} /km²", f"Phase {inp_prev_ipc} — {IPC_LABELS[inp_prev_ipc]}"
        ],
        "Status": [
            "⚠️ Flood" if inp_rainfall >= 50 else ("⚠️ Drought" if inp_rainfall <= -40 else "✅ Normal"),
            "🔴 Shock" if inp_price >= 30 else ("🟡 Elevated" if inp_price >= 15 else "✅ Normal"),
            "🔴 Active" if inp_flood else "✅ None",
            "🔴 High" if inp_conflict >= 5 else ("🟡 Moderate" if inp_conflict >= 2 else "✅ Low"),
            "🔴 Stressed" if inp_ndvi <= -0.15 else "✅ Healthy",
            "🟡 Dense" if inp_pop >= 200 else "✅ Normal",
            "🔴 Crisis+" if inp_prev_ipc >= 3 else "✅ Stable",
        ]
    }
    st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "⚙️ This predictor uses a heuristic scoring model aligned to the trained FloodHunger ML pipeline. "
        "For production deployment, connect to the serialised model via `joblib.load()`. "
        "Developed with support from **Blossom Academy Ghana** · Powered by **WFP FEWS NET** methodology."
    )
