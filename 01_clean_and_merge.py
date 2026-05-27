"""
╔══════════════════════════════════════════════════════════════╗
║   FLOODHUNGER GHANA — SCRIPT 01: CLEAN & MERGE (DISTRICT)   ║
║   District-level pipeline — 53 districts × 264 months      ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"  ▸ {msg}")
def separator(title):
    print(); print("─" * 60); print(f"  {title}"); print("─" * 60)

REGION_MAP = {
    "Greater Accra":"Greater Accra","Ashanti":"Ashanti","Northern":"Northern",
    "Upper East":"Upper East","Upper West":"Upper West","Volta":"Volta",
    "Eastern":"Eastern","Central":"Central","Western":"Western",
    "Brong-Ahafo":"Bono","Brong Ahafo":"Bono","Bono":"Bono",
    "Bono East":"Bono East","Ahafo":"Ahafo","Savannah":"Savannah",
    "Oti":"Oti","North East":"North East","Western North":"Western North",
    "Accra":"Greater Accra","Kumasi":"Ashanti","Tamale":"Northern",
    "Bolgatanga":"Upper East","Wa":"Upper West","Ho":"Volta",
    "Koforidua":"Eastern","Cape Coast":"Central","Sekondi-Takoradi":"Western","Sunyani":"Bono",
}
def std_region(name):
    if pd.isna(name): return "Unknown"
    return REGION_MAP.get(str(name).strip(), str(name).strip())


def clean_chirps():
    separator("STEP 1: CHIRPS RAINFALL — DISTRICT LEVEL")
    df = pd.read_csv(RAW / "chirps_ghana_rainfall.csv")
    log(f"Raw rows: {len(df):,}")
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")
    df             = df.dropna(subset=["date"])
    df["year"]     = df["date"].dt.year
    df["month"]    = df["date"].dt.month
    df["region"]   = df["adm1_name"].apply(std_region)
    df["district"] = df["adm2_name"].str.strip()
    rain_cols = ["rfh","rfh_avg","rfq","r1h","r1h_avg","r1q","r3h","r3h_avg","r3q"]
    for c in rain_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    monthly = (df.groupby(["region","district","year","month"])[rain_cols]
               .mean().round(3).reset_index())
    monthly = monthly.rename(columns={
        "rfh":"rainfall_mm","rfh_avg":"rainfall_avg_mm","rfq":"rainfall_anomaly_pct",
        "r1h":"rainfall_1m_mm","r1h_avg":"rainfall_1m_avg_mm","r1q":"rainfall_1m_anomaly_pct",
        "r3h":"rainfall_3m_mm","r3h_avg":"rainfall_3m_avg_mm","r3q":"rainfall_3m_anomaly_pct",
    })
    monthly["flood_flag"]   = (monthly["rainfall_anomaly_pct"] > 50).astype(int)
    monthly["drought_flag"] = (monthly["rainfall_anomaly_pct"] < -40).astype(int)
    log(f"District-month rows : {len(monthly):,}")
    log(f"Districts           : {monthly['district'].nunique()}")
    log(f"Regions             : {monthly['region'].nunique()}")
    log(f"Years               : {monthly['year'].min()} – {monthly['year'].max()}")
    log(f"Flood months        : {monthly['flood_flag'].sum()}")
    log(f"Drought months      : {monthly['drought_flag'].sum()}")
    monthly.to_csv(PROC / "clean_chirps.csv", index=False)
    log("Saved → data/processed/clean_chirps.csv")
    return monthly


def clean_prices():
    separator("STEP 2: WFP FOOD PRICES → BROADCAST TO DISTRICTS")
    df = pd.read_csv(RAW / "wfp_food_prices_ghana.csv")
    log(f"Raw rows: {len(df):,}")
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df           = df.dropna(subset=["date"])
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["region"] = df["admin1"].apply(std_region)
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")
    df           = df[df["price"].notna() & (df["price"] > 0)]
    key = ["Maize","Millet","Sorghum","Rice","Cowpeas","Groundnuts"]
    df["commodity_clean"] = df["commodity"].str.strip()
    df = df[df["commodity_clean"].apply(lambda x: any(k.lower() in x.lower() for k in key))]
    log(f"After commodity filter: {len(df):,} rows")
    monthly = (df.groupby(["region","year","month","commodity_clean"])["price"]
               .mean().round(4).reset_index())
    pivot = monthly.pivot_table(index=["region","year","month"],
                                columns="commodity_clean", values="price",
                                aggfunc="mean").reset_index()
    pivot.columns.name = None
    rename = {}
    for c in pivot.columns:
        cl = str(c).lower()
        if "maize" in cl:       rename[c] = "price_maize"
        elif "millet" in cl:    rename[c] = "price_millet"
        elif "sorghum" in cl:   rename[c] = "price_sorghum"
        elif "rice" in cl:      rename[c] = "price_rice"
        elif "cowpea" in cl:    rename[c] = "price_cowpeas"
        elif "groundnut" in cl: rename[c] = "price_groundnuts"
    pivot = pivot.rename(columns=rename)
    pivot = pivot.sort_values(["region","year","month"]).reset_index(drop=True)
    price_col = next((c for c in ["price_maize","price_millet","price_sorghum"]
                      if c in pivot.columns and pivot[c].notna().sum() > 50), None)
    if price_col:
        pivot[f"{price_col}_lag1"] = pivot.groupby("region")[price_col].shift(1)
        pivot["price_change_pct"]  = (
            (pivot[price_col] - pivot[f"{price_col}_lag1"])
            / pivot[f"{price_col}_lag1"] * 100).round(2)
        pivot["price_volatility_3m"] = (
            pivot.groupby("region")["price_change_pct"]
            .transform(lambda x: x.rolling(3, min_periods=1).std()).round(3))
        pivot["price_shock_flag"] = (pivot["price_change_pct"] > 20).astype(int)
        log(f"Price change computed from: {price_col}")
    else:
        pivot["price_change_pct"]    = np.nan
        pivot["price_volatility_3m"] = np.nan
        pivot["price_shock_flag"]    = 0
    log(f"Region-level price rows: {len(pivot):,}")
    pivot.to_csv(PROC / "clean_prices.csv", index=False)
    log("Saved → data/processed/clean_prices.csv")
    return pivot


def clean_acled():
    separator("STEP 3: ACLED CONFLICT → BROADCAST TO DISTRICTS")
    df = pd.read_csv(RAW / "acled_ghana_events.csv")
    log(f"Raw rows: {len(df):,}")
    df["date"]       = pd.to_datetime(df["event_date"], errors="coerce")
    df               = df.dropna(subset=["date"])
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["region"]     = df["admin1"].apply(std_region)
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    df["is_flood_event"] = df["event_type"].str.lower().str.contains(
        "flood|displacement|displaced", na=False).astype(int)
    monthly = (df.groupby(["region","year","month"])
               .agg(conflict_events=("event_id_cnty","count"),
                    fatalities_total=("fatalities","sum"),
                    flood_displace_events=("is_flood_event","sum"))
               .reset_index())
    monthly = monthly.sort_values(["region","year","month"]).reset_index(drop=True)
    monthly["conflict_events_lag1"]  = monthly.groupby("region")["conflict_events"].shift(1).fillna(0)
    monthly["flood_displace_lag1"]   = monthly.groupby("region")["flood_displace_events"].shift(1).fillna(0)
    q75 = monthly["conflict_events"].quantile(0.75)
    monthly["high_conflict_flag"] = (monthly["conflict_events"] >= q75).astype(int)
    log(f"Region-level conflict rows: {len(monthly):,}")
    log(f"Total events: {monthly['conflict_events'].sum():,}")
    monthly.to_csv(PROC / "clean_acled.csv", index=False)
    log("Saved → data/processed/clean_acled.csv")
    return monthly


def build_master(chirps, prices, acled):
    separator("STEP 4: DISTRICT MASTER TABLE")
    master = chirps.copy()
    log(f"Spine: {len(master):,} rows")
    master = master.merge(prices, on=["region","year","month"], how="left")
    master = master.merge(acled,  on=["region","year","month"], how="left")
    for c in ["conflict_events","fatalities_total","flood_displace_events",
              "conflict_events_lag1","flood_displace_lag1","high_conflict_flag"]:
        if c in master.columns: master[c] = master[c].fillna(0)

    north = {"Northern","Upper East","Upper West","Savannah","North East"}
    def get_season(row):
        m, r = row["month"], row["region"]
        if r in north:
            if m in [3,4,5]:  return "pre_planting"
            elif m in [6,7]:  return "planting"
            elif m in [8,9]:  return "growing"
            elif m in [10,11]:return "harvest"
            else:             return "lean"
        else:
            if m in [3,4]:    return "planting_s1"
            elif m in [5,6]:  return "growing_s1"
            elif m == 7:      return "harvest_s1"
            elif m == 8:      return "inter_season"
            elif m in [9,10]: return "growing_s2"
            elif m == 11:     return "harvest_s2"
            else:             return "lean"

    master["season_phase"]   = master.apply(get_season, axis=1)
    master["is_lean_season"] = master["season_phase"].isin(["lean","pre_planting"]).astype(int)

    vulnerable = {"Northern","Upper East","Upper West","Savannah","North East","Oti"}
    master["district_vulnerability"] = master["region"].apply(lambda r: 2 if r in vulnerable else 1)

    def compute_ipc(row):
        score = 0
        rain = row.get("rainfall_anomaly_pct", 0) or 0
        if rain < -40:   score += 2
        elif rain < -20: score += 1
        elif rain > 60:  score += 1
        score += row.get("flood_flag", 0) * 2
        score += row.get("drought_flag", 0)
        pc = row.get("price_change_pct", 0) or 0
        if pc > 30:      score += 3
        elif pc > 15:    score += 2
        elif pc > 5:     score += 1
        score += min(row.get("conflict_events", 0) or 0, 10) * 0.3
        score += row.get("is_lean_season", 0)
        score += row.get("district_vulnerability", 1) - 1
        if score <= 1:   return 1
        elif score <= 3: return 2
        elif score <= 6: return 3
        else:            return 4

    master["ipc_phase"] = master.apply(compute_ipc, axis=1)
    master["flood_price_interaction"] = (
        master["flood_flag"] * master["price_change_pct"].fillna(0)).round(3)
    master["compound_risk_flag"] = (
        (master["flood_flag"] == 1) &
        (master.get("price_shock_flag", pd.Series(0, index=master.index)).fillna(0) == 1) &
        (master["high_conflict_flag"].fillna(0) == 1)
    ).astype(int)

    master = master.sort_values(["district","year","month"]).reset_index(drop=True)
    master = master.dropna(subset=["rainfall_mm","region","district","year","month"])

    log(f"\n  DISTRICT MASTER TABLE SUMMARY")
    log(f"  Total rows   : {len(master):,}")
    log(f"  Districts    : {master['district'].nunique()}")
    log(f"  Regions      : {master['region'].nunique()}")
    log(f"  Columns      : {len(master.columns)}")
    log(f"  Years        : {master['year'].min()} – {master['year'].max()}")
    log(f"  IPC distribution:")
    for phase, count in master["ipc_phase"].value_counts().sort_index().items():
        pct = count / len(master) * 100
        bar = "█" * int(pct / 2)
        log(f"    Phase {phase}: {count:>6,} rows ({pct:4.1f}%)  {bar}")

    master.to_csv(PROC / "master_monthly.csv", index=False)
    log("\n  Saved → data/processed/master_monthly.csv")
    return master


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  FLOODHUNGER GHANA — CLEAN & MERGE (DISTRICT LEVEL)")
    print("=" * 60)
    chirps = clean_chirps()
    prices = clean_prices()
    acled  = clean_acled()
    master = build_master(chirps, prices, acled)
    print()
    print("=" * 60)
    print(f"  ✓ Done.  {len(master):,} rows  |  {master['district'].nunique()} districts")
    print("  Next: python 02_feature_engineering.py")
    print("=" * 60)
