"""
FloodHunger Ghana — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="FloodHunger Ghana",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme constants ───────────────────────────────────────
PRIMARY  = "#1F5C8B"
DARK     = "#0D3C6E"
LIGHT    = "#D6E8F5"
ACCENT   = "#2E86C1"
DANGER   = "#C0392B"
WARNING  = "#E67E22"
SUCCESS  = "#27AE60"
NEUTRAL  = "#7F8C8D"

IPC_LABELS  = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency"}
IPC_COLORS  = {1: SUCCESS, 2: "#F1C40F", 3: WARNING, 4: DANGER}
IPC_ICONS   = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
IPC_ACTIONS = {
    1: "Continue monitoring",
    2: "Alert field teams — pre-position supplies",
    3: "Deploy food stocks and vouchers immediately",
    4: "EMERGENCY — deploy cash transfers NOW",
}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
    background: #F7F9FC;
    font-family: 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{ background: {DARK}; }}
  [data-testid="stSidebar"] * {{ color: #ECF0F1 !important; }}
  [data-testid="stSidebar"] label {{
    color: #BDC3C7 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  .banner {{
    background: linear-gradient(135deg, {DARK} 0%, {PRIMARY} 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 22px;
    color: white;
  }}
  .banner h1 {{ margin:0; font-size:1.85rem; font-weight:800; color:white !important; }}
  .banner p  {{ margin:6px 0 0 0; opacity:0.78; font-size:0.92rem; }}

  .kcard {{
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 5px solid {PRIMARY};
    margin-bottom: 10px;
  }}
  .kcard.danger  {{ border-left-color:{DANGER}; }}
  .kcard.warning {{ border-left-color:{WARNING}; }}
  .kcard.success {{ border-left-color:{SUCCESS}; }}
  .kcard.neutral {{ border-left-color:{NEUTRAL}; }}
  .kcard .klabel {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:{NEUTRAL}; }}
  .kcard .kval   {{ font-size:1.9rem; font-weight:800; color:{DARK}; line-height:1.1; }}
  .kcard .ksub   {{ font-size:0.75rem; color:{NEUTRAL}; margin-top:2px; }}

  .sec {{ font-size:1rem; font-weight:700; color:{DARK};
    border-bottom:2px solid {LIGHT}; padding-bottom:5px; margin:18px 0 12px 0; }}

  .arow {{
    display:flex; align-items:flex-start; gap:12px;
    background:white; border-radius:10px; padding:13px 16px;
    margin-bottom:7px; box-shadow:0 1px 5px rgba(0,0,0,0.06);
    border-left:4px solid {NEUTRAL};
  }}
  .arow.p2 {{ border-left-color:#F1C40F; }}
  .arow.p3 {{ border-left-color:{WARNING}; }}
  .arow.p4 {{ border-left-color:{DANGER}; }}
  .adistrict {{ font-weight:700; color:{DARK}; font-size:0.93rem; }}
  .ameta {{ font-size:0.76rem; color:{NEUTRAL}; margin-top:3px; }}
  .aaction {{ font-size:0.8rem; color:{WARNING}; margin-top:2px; font-weight:600; }}

  .fbox {{
    background: linear-gradient(135deg, {DARK} 0%, {ACCENT} 100%);
    border-radius: 12px; padding:18px 22px; color:white; margin-bottom:9px;
  }}
  .fbox .ftitle {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.65; margin-bottom:5px; }}
  .fbox .ftext  {{ font-size:0.95rem; line-height:1.5; }}
  .fbox .flang  {{ font-size:0.72rem; opacity:0.55; margin-top:7px; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:2px solid {LIGHT}; }}
  .stTabs [data-baseweb="tab"] {{
    background:transparent; border-radius:8px 8px 0 0;
    color:{NEUTRAL}; font-weight:600; padding:7px 16px;
  }}
  .stTabs [aria-selected="true"] {{
    background:white !important; color:{PRIMARY} !important;
    border-bottom:2px solid {PRIMARY};
  }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  .block-container {{ padding-top:1.4rem; padding-bottom:2rem; }}
</style>
""", unsafe_allow_html=True)


# ── Load data & models ────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(Path("data/outputs/predictions_full.csv"))
    df["pred_ipc_phase"] = df["pred_ipc_phase"].astype(int)
    df["ipc_phase"]      = df["ipc_phase"].astype(int)
    return df

@st.cache_resource
def load_models():
    m = Path("data/models")
    return {
        "ipc":       joblib.load(m / "ipc_classifier_xgb.pkl"),
        "price":     joblib.load(m / "price_regressor_rf.pkl"),
        "anomaly":   joblib.load(m / "anomaly_detector_if.pkl"),
        "scaler":    joblib.load(m / "anomaly_scaler.pkl"),
        "feat":      joblib.load(m / "feature_cols.pkl"),
        "reg_feat":  joblib.load(m / "reg_feature_cols.pkl"),
        "anom_feat": joblib.load(m / "anomaly_feature_cols.pkl"),
        "p2i":       joblib.load(m / "phase_to_idx.pkl"),
    }

df     = load_data()
models = load_models()
i2p    = {v: k for k, v in models["p2i"].items()}

ALL_REGIONS   = sorted(df["region"].unique())
ALL_DISTRICTS = sorted(df["district"].unique())
ALL_YEARS     = sorted(df["year"].unique())
LATEST_YEAR   = int(df["year"].max())
LATEST_MONTH  = int(df[df["year"] == LATEST_YEAR]["month"].max())
MONTH_STR     = datetime(LATEST_YEAR, LATEST_MONTH, 1).strftime("%B %Y")

PRICE_COLS = [c for c in ["price_maize","price_millet","price_sorghum",
                            "price_rice","price_cowpeas","price_groundnuts"]
              if c in df.columns and df[c].notna().sum() > 10]


# ── Helpers ───────────────────────────────────────────────
def kpi(col, label, val, sub, kind=""):
    col.markdown(f"""
    <div class="kcard {kind}">
      <div class="klabel">{label}</div>
      <div class="kval">{val}</div>
      <div class="ksub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def plotly_defaults(fig, h=380):
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Segoe UI", height=h,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
    )
    return fig


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0 22px'>
      <div style='font-size:2.2rem;'>🌊</div>
      <div style='font-size:1.1rem;font-weight:800;color:white;margin-top:5px;'>FloodHunger Ghana</div>
      <div style='font-size:0.72rem;color:#95A5A6;margin-top:2px;'>Early Warning System</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("##### 🔎 Filters")
    sel_regions = st.multiselect("Regions", ALL_REGIONS, default=ALL_REGIONS)
    year_range  = st.slider("Year Range",
                            int(df["year"].min()), LATEST_YEAR,
                            (2015, LATEST_YEAR))
    sel_phases  = st.multiselect("IPC Phases", [1,2,3,4], default=[1,2,3,4],
                                  format_func=lambda x: f"Phase {x} — {IPC_LABELS[x]}")
    only_anom   = st.toggle("Anomalies only", False)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem;color:#95A5A6;line-height:1.7;'>
      <b style='color:#BDC3C7;'>Dataset</b><br>
      {df['district'].nunique()} districts · {df['region'].nunique()} regions<br>
      {df['year'].min()}–{LATEST_YEAR} · {len(df):,} rows<br><br>
      <b style='color:#BDC3C7;'>Models</b><br>
      XGBoost · Random Forest · Isolation Forest<br><br>
      <b style='color:#BDC3C7;'>Sources</b><br>
      CHIRPS v3 · WFP VAM · ACLED
    </div>""", unsafe_allow_html=True)

# Apply filters
fdf = df[
    df["region"].isin(sel_regions) &
    df["year"].between(*year_range) &
    df["pred_ipc_phase"].isin(sel_phases)
]
if only_anom:
    fdf = fdf[fdf["is_anomaly"] == 1]


# ══════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="banner">
  <h1>🌊 FloodHunger Ghana</h1>
  <p>Flood-Driven Food Insecurity Early Warning System &nbsp;·&nbsp;
  {df['district'].nunique()} Districts &nbsp;·&nbsp;
  {df['region'].nunique()} Regions &nbsp;·&nbsp;
  Latest: <strong>{MONTH_STR}</strong></p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  KPI ROW
# ══════════════════════════════════════════════════════════
latest   = df[(df["year"]==LATEST_YEAR) & (df["month"]==LATEST_MONTH)]
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpi(k1, "Crisis Districts",    int((latest["pred_ipc_phase"]>=3).sum()),    f"Phase 3–4  ·  {MONTH_STR}", "danger")
kpi(k2, "Stressed Districts",  int((latest["pred_ipc_phase"]==2).sum()),    f"Phase 2  ·  {MONTH_STR}",   "warning")
kpi(k3, "Anomalies",           int(latest["is_anomaly"].sum()),             "Unusual patterns this month","warning")
kpi(k4, "Avg Rain Anomaly",    f"{fdf['rainfall_anomaly_pct'].mean():+.1f}%", f"{year_range[0]}–{year_range[1]}", "neutral")
kpi(k5, "Flood Months",        f"{int(fdf['flood_flag'].sum()):,}",         "In filtered period",          "neutral")
kpi(k6, "Price Shocks",        f"{int(fdf['price_shock_flag'].sum()) if 'price_shock_flag' in fdf.columns else '—'}",
        "Months >20% price rise", "neutral")


# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
t1,t2,t3,t4,t5,t6 = st.tabs([
    "🗺️  District Overview",
    "📊  Food Security Trends",
    "🌧️  Rainfall",
    "💰  Food Prices",
    "⚠️  Alert Feed",
    "🔮  Live Prediction",
])


# ─── TAB 1: DISTRICT OVERVIEW ─────────────────────────────
with t1:
    sec("IPC Phase by Region")

    c1, c2 = st.columns([3,1])
    with c2:
        ov_year  = st.selectbox("Year",  sorted(ALL_YEARS, reverse=True), key="ov_yr")
        ov_month = st.selectbox("Month", range(1,13),
                                format_func=lambda x: MONTH_NAMES[x-1],
                                index=LATEST_MONTH-1, key="ov_mn")
        ov_col   = st.selectbox("Colour by", {
            "pred_ipc_phase":      "IPC Phase",
            "rainfall_anomaly_pct":"Rain Anomaly %",
            "compound_risk_score": "Risk Score",
            "anomaly_score":       "Anomaly Score",
            "conflict_events":     "Conflict Events",
        }.keys(), format_func=lambda x: {
            "pred_ipc_phase":      "IPC Phase",
            "rainfall_anomaly_pct":"Rain Anomaly %",
            "compound_risk_score": "Risk Score",
            "anomaly_score":       "Anomaly Score",
            "conflict_events":     "Conflict Events",
        }[x], key="ov_col")

    snap = df[(df["year"]==ov_year) & (df["month"]==ov_month)].copy()
    snap["phase_label"] = snap["pred_ipc_phase"].map(IPC_LABELS)

    reg_agg = snap.groupby("region").agg(
        metric   = (ov_col, "mean"),
        phase    = ("pred_ipc_phase", lambda x: int(x.mode()[0])),
        districts= ("district","count"),
    ).reset_index()
    reg_agg["phase_label"] = reg_agg["phase"].map(IPC_LABELS)

    with c1:
        fig = px.bar(
            reg_agg.sort_values("metric"),
            x="metric", y="region", orientation="h",
            color="metric",
            color_continuous_scale=[SUCCESS, "#F1C40F", WARNING, DANGER],
            labels={"metric": ov_col.replace("_"," ").title(), "region":""},
            title=f"{ov_col.replace('_',' ').title()} — {MONTH_NAMES[ov_month-1]} {ov_year}",
            hover_data={"districts":True,"phase_label":True},
            height=480,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Segoe UI", coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    sec("All Districts This Month")
    tbl = snap[[
        "district","region","pred_ipc_phase","pred_ipc_label",
        "pred_ipc_confidence","rainfall_anomaly_pct","flood_flag",
        "conflict_events","is_anomaly","compound_risk_score",
    ]].rename(columns={
        "pred_ipc_phase":"IPC Phase","pred_ipc_label":"Status",
        "pred_ipc_confidence":"Confidence","rainfall_anomaly_pct":"Rain Anomaly %",
        "flood_flag":"Flood","conflict_events":"Conflicts",
        "is_anomaly":"Anomaly","compound_risk_score":"Risk Score",
    }).sort_values("IPC Phase", ascending=False)

    def _bg(s):
        return [
            {1:"background-color:#d5f5e3",2:"background-color:#fef9e7",
             3:"background-color:#fdebd0",4:"background-color:#fadbd8"}.get(v,"")
            for v in s
        ]

    st.dataframe(
        tbl.style.apply(_bg, subset=["IPC Phase"])
                 .format({"Confidence":"{:.1%}","Rain Anomaly %":"{:+.1f}","Risk Score":"{:.2f}"}),
        use_container_width=True, height=360,
    )


# ─── TAB 2: FOOD SECURITY TRENDS ──────────────────────────
with t2:
    sec("IPC Phase Distribution Over Time")

    c1,c2 = st.columns([2,1])
    with c2:
        tr_region = st.selectbox("Region", ["All Ghana"]+ALL_REGIONS, key="tr_reg")
        tr_view   = st.radio("Group by", ["Year","Month"], horizontal=True, key="tr_view")

    tdf = fdf if tr_region=="All Ghana" else fdf[fdf["region"]==tr_region]

    with c1:
        grp = "year" if tr_view=="Year" else "month"
        ph_grp = tdf.groupby([grp,"pred_ipc_phase"]).size().reset_index(name="count")
        ph_grp["label"] = ph_grp["pred_ipc_phase"].map(IPC_LABELS)
        if grp == "month":
            ph_grp["month_name"] = ph_grp["month"].apply(lambda x: MONTH_NAMES[x-1])
            x_col, cat_ord = "month_name", {"month_name": MONTH_NAMES}
        else:
            x_col, cat_ord = "year", {}

        fig = px.bar(ph_grp, x=x_col, y="count", color="label",
                     color_discrete_map={"Minimal":SUCCESS,"Stressed":"#F1C40F",
                                         "Crisis":WARNING,"Emergency":DANGER},
                     labels={"count":"District-Months","label":"IPC Phase"},
                     title=f"IPC Phase Distribution by {tr_view} — {tr_region}",
                     category_orders=cat_ord, height=380)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Segoe UI", barmode="stack",
                          legend=dict(orientation="h",y=1.05),
                          margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig, use_container_width=True)

    sec("Average IPC Phase Timeline")
    avg_ipc = tdf.groupby(["year","month"])["pred_ipc_phase"].mean().reset_index()
    avg_ipc["date"] = pd.to_datetime(avg_ipc[["year","month"]].assign(day=1))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=avg_ipc["date"], y=avg_ipc["pred_ipc_phase"],
        mode="lines", fill="tozeroy",
        line=dict(color=PRIMARY, width=2.2),
        fillcolor="rgba(31,92,139,0.11)", name="Avg IPC Phase",
    ))
    for ph, lbl, col in [(2,"Stressed","#F1C40F"),(3,"Crisis",WARNING),(4,"Emergency",DANGER)]:
        fig2.add_hline(y=ph, line_dash="dot", line_color=col,
                       annotation_text=lbl, annotation_position="right")
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=280,
                       font_family="Segoe UI", yaxis_title="Avg IPC Phase",
                       yaxis=dict(range=[1,4.5],showgrid=True,gridcolor="#F0F0F0"),
                       xaxis=dict(showgrid=False),
                       margin=dict(l=20,r=80,t=20,b=20))
    st.plotly_chart(fig2, use_container_width=True)

    sec("Region × Month Heatmap")
    heat = tdf.groupby(["region","month"])["pred_ipc_phase"].mean().reset_index()
    hpiv = heat.pivot(index="region", columns="month", values="pred_ipc_phase")
    hpiv.columns = [MONTH_NAMES[c-1] for c in hpiv.columns]
    fig3 = px.imshow(hpiv,
                     color_continuous_scale=["#2ECC71","#F1C40F","#E67E22","#C0392B"],
                     zmin=1, zmax=4, aspect="auto",
                     labels=dict(color="IPC Phase"),
                     title="Avg IPC Phase — Region × Month",
                     height=370)
    fig3.update_layout(font_family="Segoe UI", paper_bgcolor="white",
                       margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig3, use_container_width=True)


# ─── TAB 3: RAINFALL ──────────────────────────────────────
with t3:
    sec("Rainfall Anomaly Over Time")

    c1,c2 = st.columns([2,1])
    with c2:
        rd = st.selectbox("District", ALL_DISTRICTS, key="rd")
        rc = st.multiselect("Compare with", ALL_DISTRICTS, default=[], max_selections=3, key="rc")

    rain_df = fdf[fdf["district"].isin([rd]+rc)].copy()
    rain_df["date"] = pd.to_datetime(rain_df[["year","month"]].assign(day=1))

    with c1:
        fig = px.line(rain_df, x="date", y="rainfall_anomaly_pct", color="district",
                      labels={"rainfall_anomaly_pct":"Anomaly (%)","date":""},
                      title="Rainfall Anomaly % — District Comparison", height=370)
        fig.add_hline(y=50,  line_dash="dash", line_color=DANGER,
                      annotation_text="Flood risk +50%")
        fig.add_hline(y=-40, line_dash="dash", line_color=WARNING,
                      annotation_text="Drought risk −40%")
        fig.add_hrect(y0=50, y1=250, fillcolor="rgba(192,57,43,0.06)", line_width=0)
        fig.add_hrect(y0=-250, y1=-40, fillcolor="rgba(230,126,34,0.06)", line_width=0)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Segoe UI",
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True,gridcolor="#F0F0F0"),
                          margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    flood_reg   = fdf.groupby("region")["flood_flag"].sum().sort_values().reset_index()
    drought_reg = fdf.groupby("region")["drought_flag"].sum().sort_values().reset_index()

    with c1:
        sec("Flood Months by Region")
        fig = px.bar(flood_reg, x="flood_flag", y="region", orientation="h",
                     color="flood_flag",
                     color_continuous_scale=[LIGHT, PRIMARY, DARK],
                     labels={"flood_flag":"Months","region":""}, height=340)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Segoe UI", coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Drought Months by Region")
        fig = px.bar(drought_reg, x="drought_flag", y="region", orientation="h",
                     color="drought_flag",
                     color_continuous_scale=[LIGHT, WARNING, DANGER],
                     labels={"drought_flag":"Months","region":""}, height=340)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Segoe UI", coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

    sec("Rainfall Anomaly Distribution")
    fig = px.histogram(
        fdf["rainfall_anomaly_pct"].clip(-100,150), nbins=80,
        color_discrete_sequence=[PRIMARY],
        labels={"value":"Rainfall Anomaly (%)"},
        title="All Districts — Rainfall Anomaly Distribution", height=280,
    )
    fig.add_vline(x=50,  line_dash="dash", line_color=DANGER,  annotation_text="Flood risk")
    fig.add_vline(x=-40, line_dash="dash", line_color=WARNING, annotation_text="Drought risk")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      font_family="Segoe UI", showlegend=False,
                      margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)


# ─── TAB 4: FOOD PRICES ───────────────────────────────────
with t4:
    sec("Commodity Price Trend")

    c1,c2 = st.columns([2,1])
    with c2:
        sel_comm    = st.selectbox("Commodity", PRICE_COLS,
                                   format_func=lambda x: x.replace("price_","").title(),
                                   key="comm")
        price_reg   = st.selectbox("Region", ["All Ghana"]+ALL_REGIONS, key="preg")
        show_pred   = st.toggle("Show predicted price change", True)

    pdf = fdf if price_reg=="All Ghana" else fdf[fdf["region"]==price_reg]
    pts = pdf.groupby(["year","month"])[[sel_comm,"pred_price_change"]].mean().reset_index()
    pts["date"] = pd.to_datetime(pts[["year","month"]].assign(day=1))
    pts = pts.dropna(subset=[sel_comm])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pts["date"], y=pts[sel_comm],
            mode="lines", name=sel_comm.replace("price_","").title()+" price",
            line=dict(color=PRIMARY, width=2.5),
            fill="tozeroy", fillcolor="rgba(31,92,139,0.09)",
        ))
        if show_pred and "pred_price_change" in pts.columns:
            fig.add_trace(go.Bar(
                x=pts["date"], y=pts["pred_price_change"],
                name="Pred. Change %",
                marker_color=[DANGER if v>0 else SUCCESS for v in pts["pred_price_change"].fillna(0)],
                opacity=0.45, yaxis="y2",
            ))
        fig.update_layout(
            title=f"{sel_comm.replace('price_','').title()} Price — {price_reg}",
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="Segoe UI", height=390,
            yaxis=dict(title="GHS/kg",showgrid=True,gridcolor="#F0F0F0"),
            yaxis2=dict(title="Pred. Change %", overlaying="y", side="right",
                        showgrid=False, zeroline=True, zerolinecolor="#DDD"),
            legend=dict(orientation="h",y=1.05),
            xaxis=dict(showgrid=False),
            margin=dict(l=20,r=60,t=55,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        sec("Latest Commodity Prices")
        lp   = df[(df["year"]==LATEST_YEAR) & (df["month"]==LATEST_MONTH)]
        pavg = {c: lp[c].mean() for c in PRICE_COLS}
        pdf2 = pd.DataFrame({"commodity":list(pavg.keys()),"price":list(pavg.values())})
        pdf2["commodity"] = pdf2["commodity"].str.replace("price_","").str.title()
        pdf2 = pdf2.dropna().sort_values("price")
        fig  = px.bar(pdf2, x="price", y="commodity", orientation="h",
                      color="price",
                      color_continuous_scale=[LIGHT, PRIMARY, DARK],
                      labels={"price":"Avg GHS/kg","commodity":""},
                      title=f"Avg Prices — {MONTH_STR}", height=300)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Segoe UI", coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "price_shock_flag" in fdf.columns:
            sec("Price Shock History")
            sh = fdf.groupby("year")["price_shock_flag"].sum().reset_index()
            fig= px.bar(sh, x="year", y="price_shock_flag",
                        color="price_shock_flag",
                        color_continuous_scale=[LIGHT, DANGER],
                        labels={"price_shock_flag":"Shock Months","year":""},
                        title="Price Shock Months per Year (>20% rise)", height=300)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_family="Segoe UI", coloraxis_showscale=False,
                              margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)


# ─── TAB 5: ALERT FEED ────────────────────────────────────
with t5:
    sec("WFP District Alert Feed")

    c1,c2 = st.columns([2,1])
    with c2:
        al_yr  = st.selectbox("Year",  sorted(ALL_YEARS,reverse=True), key="al_yr")
        al_mn  = st.selectbox("Month", range(1,13),
                               format_func=lambda x: MONTH_NAMES[x-1],
                               index=LATEST_MONTH-1, key="al_mn")
        min_ph = st.selectbox("Min Phase", [1,2,3,4], index=1,
                               format_func=lambda x: f"Phase {x} — {IPC_LABELS[x]}",
                               key="min_ph")

    al_df = df[
        (df["year"]==al_yr) & (df["month"]==al_mn) &
        (df["pred_ipc_phase"]>=min_ph)
    ].sort_values("pred_ipc_phase", ascending=False)

    with c1:
        if len(al_df)==0:
            st.success(f"✅ No districts at Phase {min_ph}+ in {MONTH_NAMES[al_mn-1]} {al_yr}")
        else:
            st.caption(f"**{len(al_df)} districts** at Phase {min_ph}+ — {MONTH_NAMES[al_mn-1]} {al_yr}")
            for _, r in al_df.iterrows():
                ph   = int(r["pred_ipc_phase"])
                anom = " &nbsp;⚠️ <b>ANOMALY</b>" if r["is_anomaly"]==1 else ""
                st.markdown(f"""
                <div class="arow p{ph}">
                  <div style="font-size:1.5rem;margin-top:2px;">{IPC_ICONS[ph]}</div>
                  <div style="flex:1;">
                    <div class="adistrict">{r['district']}
                      <span style="font-weight:400;color:{NEUTRAL};font-size:0.8rem;">
                        · {r['region']}</span>{anom}
                    </div>
                    <div class="aaction">{IPC_ACTIONS[ph]}</div>
                    <div class="ameta">
                      Rain anomaly <b>{r['rainfall_anomaly_pct']:+.1f}%</b> &nbsp;·&nbsp;
                      Flood <b>{'Yes' if r['flood_flag']==1 else 'No'}</b> &nbsp;·&nbsp;
                      Conflicts <b>{int(r['conflict_events'])}</b> &nbsp;·&nbsp;
                      Risk score <b>{r['compound_risk_score']:.2f}</b> &nbsp;·&nbsp;
                      Confidence <b>{r['pred_ipc_confidence']:.1%}</b>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    sec("📢 Farmer Voice Alert Bulletin")
    st.caption("Plain-language alerts for SMS broadcast — Twi · Dagbani · Ewe · Hausa")

    crisis = al_df[al_df["pred_ipc_phase"]>=3]
    if len(crisis)==0:
        st.info("No Phase 3+ districts this period.")
    else:
        for _, r in crisis.iterrows():
            rain    = r["rainfall_anomaly_pct"]
            weather = ("⛈️ Heavy flooding detected." if r["flood_flag"]==1
                       else "☀️ Severe dry spell — low rainfall." if rain < -30
                       else "🌤️ Unstable weather conditions.")
            pmsg    = ("Food prices rising — buy or store food now."
                       if "Rising" in str(r.get("pred_price_direction",""))
                       else "Food prices stable.")
            full    = f"{weather} {pmsg} Contact your district food officer."
            st.markdown(f"""
            <div class="fbox">
              <div class="ftitle">🔴 {r['district']} · {r['region']} · Phase {int(r['pred_ipc_phase'])}</div>
              <div class="ftext">{full}</div>
              <div class="flang">Translate before broadcast → Twi · Dagbani · Ewe · Hausa</div>
            </div>""", unsafe_allow_html=True)

    sec("🔍 Top Anomalous District-Months")
    anom_tbl = df[df["is_anomaly"]==1][[
        "district","region","year","month","anomaly_score",
        "ipc_phase","rainfall_anomaly_pct","conflict_events"
    ]].sort_values("anomaly_score", ascending=False).head(50).copy()
    anom_tbl["month"] = anom_tbl["month"].apply(lambda x: MONTH_NAMES[x-1])
    st.dataframe(anom_tbl.reset_index(drop=True), use_container_width=True, height=280)

    sec("📥 Download Alert Data")
    dl_df = al_df[[
        "district","region","pred_ipc_phase","pred_ipc_label",
        "pred_ipc_confidence","pred_ipc_action",
        "rainfall_anomaly_pct","flood_flag","conflict_events",
        "is_anomaly","compound_risk_score"
    ]]
    st.download_button(
        "⬇️ Download CSV", dl_df.to_csv(index=False),
        file_name=f"floodhunger_alerts_{al_yr}_{al_mn:02d}.csv",
        mime="text/csv",
    )


# ─── TAB 6: LIVE PREDICTION ───────────────────────────────
with t6:
    sec("🔮 Real-Time IPC Phase Prediction")
    st.caption("Enter current field conditions — models return an IPC phase, price forecast, and anomaly flag instantly.")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("**📍 Location & Time**")
        p_district = st.selectbox("District", ALL_DISTRICTS, key="pd")
        p_region   = st.selectbox("Region", ALL_REGIONS, key="pr")
        p_month    = st.selectbox("Month", range(1,13),
                                  format_func=lambda x: MONTH_NAMES[x-1], key="pm")
        p_year     = st.number_input("Year", 2003, 2030, LATEST_YEAR, key="py")

    with c2:
        st.markdown("**🌧️ Rainfall Conditions**")
        p_rain_mm   = st.slider("Rainfall this month (mm)",     0.0,  300.0,  55.0, 0.5)
        p_rain_anom = st.slider("Rainfall anomaly (%)",        -100.0,200.0,   0.0, 1.0)
        p_rain_3m   = st.slider("3-month cumulative (mm)",       0.0,  600.0, 160.0, 5.0)
        p_flood_l1  = st.toggle("Flood occurred last month?")
        p_drought   = st.toggle("Drought conditions present?")

    with c3:
        st.markdown("**💰 Prices & Conflict**")
        p_price_chg = st.slider("Food price change (%)",      -50.0, 100.0,  0.0, 0.5)
        p_price_vol = st.slider("Price volatility — 3m std",   0.0,  50.0,   5.0, 0.5)
        p_conflict  = st.slider("Conflict events this month",  0, 30, 0)
        p_conf_l1   = st.slider("Conflict events last month",  0, 30, 0)

    predict = st.button("🔮 Predict Now", type="primary", use_container_width=True)

    if predict:
        VULN = {"Northern","Upper East","Upper West","Savannah","North East","Oti"}

        fv = {f: 0.0 for f in models["feat"]}
        fv.update({
            "rainfall_mm":              p_rain_mm,
            "rainfall_anomaly_pct":     p_rain_anom,
            "rainfall_3m_mm":           p_rain_3m,
            "rainfall_rolling_3m":      p_rain_3m / 3,
            "rainfall_rolling_6m":      p_rain_3m / 2,
            "flood_flag":               1 if p_rain_anom > 50 else 0,
            "drought_flag":             1 if p_drought else 0,
            "flood_flag_lag1":          1 if p_flood_l1 else 0,
            "price_change_pct":         p_price_chg,
            "price_volatility_3m":      p_price_vol,
            "price_shock_flag":         1 if p_price_chg > 20 else 0,
            "conflict_events":          float(p_conflict),
            "conflict_events_lag1":     float(p_conf_l1),
            "conflict_lag1":            float(p_conf_l1),
            "is_lean_season":           1 if p_month in [12,1,2,3] else 0,
            "month_sin":                float(np.sin(2*np.pi*p_month/12)),
            "month_cos":                float(np.cos(2*np.pi*p_month/12)),
            "time_trend":               (p_year - 2003)*12 + p_month,
            "district_vulnerability":   2 if p_region in VULN else 1,
            "flood_price_interaction":  (1 if p_rain_anom>50 else 0) * p_price_chg,
            "flood_price_lag1_interact":(1 if p_flood_l1 else 0) * p_price_chg,
            "compound_risk_score":      (
                abs(p_rain_anom)*0.01 +
                (2.0 if p_rain_anom>50 else 0) +
                max(0, p_price_chg)*0.05 +
                p_conflict*0.1 +
                (1 if p_month in [12,1,2,3] else 0)
            ),
            "region_type":              0 if p_region in VULN else 2 if p_region in {"Greater Accra","Central","Western"} else 1,
        })

        Xi = pd.DataFrame([fv])[models["feat"]]

        pred_idx   = int(models["ipc"].predict(Xi)[0])
        pred_phase = i2p[pred_idx]
        pred_prob  = float(models["ipc"].predict_proba(Xi)[0].max())
        pred_price = float(models["price"].predict(Xi[models["reg_feat"]])[0])

        Xa_raw = Xi[[f for f in models["anom_feat"] if f in Xi.columns]].reindex(
            columns=models["anom_feat"], fill_value=0)
        Xa_sc  = models["scaler"].transform(Xa_raw)
        is_anom  = models["anomaly"].predict(Xa_sc)[0] == -1
        anom_sc  = float(-models["anomaly"].score_samples(Xa_sc)[0])

        st.markdown("---")
        r1,r2,r3,r4 = st.columns(4)
        kind = "danger" if pred_phase>=3 else "warning" if pred_phase==2 else "success"
        kpi(r1, "IPC Phase",       f"{IPC_ICONS[pred_phase]} Phase {pred_phase}", IPC_LABELS[pred_phase], kind)
        kpi(r2, "Confidence",      f"{pred_prob:.1%}",    "Model certainty",      "neutral")
        kpi(r3, "Price Forecast",  f"{pred_price:+.1f}%", "Next month change",
            "danger" if pred_price>10 else "success" if pred_price<-5 else "neutral")
        kpi(r4, "Anomaly",         "⚠️ YES" if is_anom else "✅ NO",
            f"Score: {anom_sc:.3f}", "warning" if is_anom else "success")

        weather = ("⛈️ Heavy flooding detected." if p_rain_anom > 50
                   else "☀️ Severe dry conditions." if p_drought
                   else "🌤️ Conditions appear normal.")
        pmsg    = ("Food prices rising — store food now." if pred_price>10
                   else "Prices stable." if pred_price>-5
                   else "Prices falling — good time to buy.")
        full    = f"{weather} {pmsg} Contact your district food officer."

        st.markdown(f"""
        <div class="fbox" style="margin-top:16px;">
          <div class="ftitle">WFP RECOMMENDED ACTION — {p_district}, {p_region}</div>
          <div class="ftext" style="font-size:1.05rem;font-weight:600;">{IPC_ACTIONS[pred_phase]}</div>
          <div class="ftext" style="margin-top:8px;font-size:0.9rem;opacity:0.85;">📢 {full}</div>
          <div class="flang">Translate before broadcast → Twi · Dagbani · Ewe · Hausa</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("🔬 Raw prediction inputs"):
            st.json({k: round(v,4) if isinstance(v,float) else v
                     for k,v in fv.items() if v != 0.0})
