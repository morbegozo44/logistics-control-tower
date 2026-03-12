
import math
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Logistics Control Tower Prototype", layout="wide")

TERMINAL_COORDS = {
    "Boston, MA": (42.3601, -71.0589),
    "Worcester, MA": (42.2626, -71.8023),
    "Providence, RI": (41.8240, -71.4128),
    "Manchester, NH": (42.9956, -71.4548),
    "Springfield, MA": (42.1015, -72.5898),
    "Hartford, CT": (41.7658, -72.6734),
    "Portland, ME": (43.6591, -70.2568),
    "New Haven, CT": (41.3083, -72.9279),
    "Albany, NY": (42.6526, -73.7562),
    "New York City / Newark, NY/NJ": (40.7306, -74.0060),
    "Montreal, QC": (45.5017, -73.5673),
    "Allentown, PA": (40.6023, -75.4714),
    "Syracuse, NY": (43.0481, -76.1474),
    "Philadelphia, PA": (39.9526, -75.1652),
    "Quebec City, QC": (46.8139, -71.2080),
    "Harrisburg, PA": (40.2732, -76.8867),
    "Rochester, NY": (43.1566, -77.6088),
    "Baltimore, MD": (39.2904, -76.6122),
    "Washington, DC": (38.9072, -77.0369),
    "Buffalo, NY": (42.8864, -78.8784),
    "Toronto, ON": (43.6532, -79.3832),
    "Norfolk, VA": (36.8508, -76.2859),
    "Richmond, VA": (37.5407, -77.4360),
    "Pittsburgh, PA": (40.4406, -79.9959),
    "Cleveland, OH": (41.4993, -81.6944),
    "Detroit, MI": (42.3314, -83.0458),
    "Columbus, OH": (39.9612, -82.9988),
    "Cincinnati, OH": (39.1031, -84.5120),
    "Indianapolis, IN": (39.7684, -86.1581),
    "Louisville, KY": (38.2527, -85.7585),
}

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"origin", "origin_city", "from", "from_city"}:
            col_map[c] = "Origin"
        elif lc in {"destination", "dest", "destination_city", "to", "to_city"}:
            col_map[c] = "Destination"
        elif "truck miles" in lc or lc in {"miles", "total miles", "truck_miles"}:
            col_map[c] = "Truck_Miles"
        elif "toll" in lc and "cost" in lc:
            col_map[c] = "Toll_Cost"
        elif "arrival status" in lc or "arrival performance" in lc:
            col_map[c] = "Arrival_Status"
    if col_map:
        df = df.rename(columns=col_map)
    return df

def add_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def lat_of(k): return TERMINAL_COORDS.get(k, (np.nan, np.nan))[0] if k else np.nan
    def lon_of(k): return TERMINAL_COORDS.get(k, (np.nan, np.nan))[1] if k else np.nan
    df["Origin_Lat"] = df["Origin"].map(lat_of) if "Origin" in df.columns else np.nan
    df["Origin_Lon"] = df["Origin"].map(lon_of) if "Origin" in df.columns else np.nan
    df["Dest_Lat"] = df["Destination"].map(lat_of) if "Destination" in df.columns else np.nan
    df["Dest_Lon"] = df["Destination"].map(lon_of) if "Destination" in df.columns else np.nan
    return df

def compute_costs(df: pd.DataFrame, mpg, diesel_price, driver_cpm, lease_cpm, overtime_per_late, penalty_per_late):
    df = df.copy()
    if "Truck_Miles" in df.columns:
        df["Truck_Miles"] = pd.to_numeric(df["Truck_Miles"], errors="coerce").fillna(0)
    else:
        df["Truck_Miles"] = 0
    if "Toll_Cost" in df.columns:
        df["Toll_Cost"] = pd.to_numeric(df["Toll_Cost"], errors="coerce").fillna(0)
    else:
        df["Toll_Cost"] = 0

    df["Fuel_Gallons"] = np.where(df["Truck_Miles"] > 0, df["Truck_Miles"]/mpg, 0.0)
    df["Fuel_Cost"] = df["Fuel_Gallons"] * diesel_price
    df["Driver_Cost"] = df["Truck_Miles"] * driver_cpm
    df["Lease_Cost"] = df["Truck_Miles"] * lease_cpm

    status = df["Arrival_Status"] if "Arrival_Status" in df.columns else pd.Series(["Unknown"]*len(df))
    is_late = status.astype(str).str.lower().str.contains("late")
    df["Late_Flag"] = is_late.astype(int)
    df["Overtime_Cost"] = df["Late_Flag"] * overtime_per_late
    df["Penalty_Cost"] = df["Late_Flag"] * penalty_per_late

    df["Total_Cost"] = df["Fuel_Cost"] + df["Toll_Cost"] + df["Driver_Cost"] + df["Lease_Cost"] + df["Overtime_Cost"] + df["Penalty_Cost"]
    df["Cost_per_Mile"] = np.where(df["Truck_Miles"] > 0, df["Total_Cost"]/df["Truck_Miles"], np.nan)
    return df

def lane_metrics(df):
    if "Origin" not in df.columns or "Destination" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["Origin","Destination"]).agg(
        Trips=("Origin","count"),
        LatePct=("Late_Flag","mean"),
        AvgCostPerMile=("Cost_per_Mile","mean"),
        AvgMiles=("Truck_Miles","mean"),
        TotalCost=("Total_Cost","sum"),
    ).reset_index()
    g["LatePct"] = (g["LatePct"]*100).round(1)
    g["AvgCostPerMile"] = g["AvgCostPerMile"].round(2)
    g["AvgMiles"] = g["AvgMiles"].round(0)
    return g.sort_values(["LatePct","AvgCostPerMile","Trips"], ascending=False)

st.sidebar.title("Upload + Assumptions")
uploaded = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx","csv"])

def load_df():
    if uploaded is None:
        st.info("Upload your Excel/CSV to begin.")
        return pd.DataFrame()
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)

raw = load_df()
if raw.empty:
    st.stop()

df = normalize_columns(raw)

mpg = st.sidebar.number_input("Truck MPG", 3.0, 12.0, 6.5, 0.1)
diesel = st.sidebar.number_input("Diesel price ($/gal)", 1.0, 8.0, 4.10, 0.05)
driver_cpm = st.sidebar.number_input("Driver pay ($/mile)", 0.30, 1.50, 0.70, 0.01)
lease_cpm = st.sidebar.number_input("Lease ($/mile)", 0.05, 1.00, 0.20, 0.01)
overtime = st.sidebar.number_input("Overtime per late trip ($)", 0.0, 1000.0, 75.0, 5.0)
penalty = st.sidebar.number_input("Penalty per late trip ($)", 0.0, 2000.0, 120.0, 10.0)

base = compute_costs(df, mpg, diesel, driver_cpm, lease_cpm, overtime, penalty)
lanes = lane_metrics(base)

tab_map, tab_reco, tab_fin = st.tabs(["🗺️ Map", "✅ Recommendations", "💰 Financial Impact"])

with tab_map:
    st.subheader("Terminals + Problem Lanes")

    terminals = pd.DataFrame([{"Terminal":k, "lat":v[0], "lon":v[1]} for k,v in TERMINAL_COORDS.items()])
    layer = pdk.Layer("ScatterplotLayer", data=terminals, get_position='[lon, lat]', get_radius=8000, pickable=True)
    view = pdk.ViewState(latitude=42.5, longitude=-73.0, zoom=4.0)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{Terminal}"}), use_container_width=True)

    st.caption("Problem lanes (sorted by late % then cost per mile)")
    if lanes.empty:
        st.write("Need Origin + Destination columns to compute lane metrics.")
    else:
        st.dataframe(lanes, use_container_width=True, height=520)

with tab_reco:
    st.subheader("Scenario Simulator (prototype logic)")
    scenario = st.radio("Scenario", ["Add Drivers", "New Facility", "Rebalance Terminals"], horizontal=True)

    scn = base.copy()
    invest = 0.0

    if scenario == "Add Drivers":
        add_drivers = st.slider("Drivers to add", 1, 50, 8, 1)
        invest = st.number_input("Monthly cost per added driver ($)", 3000.0, 15000.0, 8500.0, 100.0) * add_drivers
        late_reduction = st.slider("Late-trip reduction (%)", 0, 60, 18, 1) / 100.0

        late_idx = scn.index[scn["Late_Flag"]==1].tolist()
        rng = np.random.default_rng(7)
        n_convert = int(round(len(late_idx)*late_reduction))
        if n_convert>0:
            convert = set(rng.choice(late_idx, size=n_convert, replace=False).tolist())
            scn.loc[list(convert), "Late_Flag"] = 0
            scn.loc[list(convert), "Overtime_Cost"] = 0.0
            scn.loc[list(convert), "Penalty_Cost"] = 0.0

        scn["Total_Cost"] = scn["Fuel_Cost"] + scn["Toll_Cost"] + scn["Driver_Cost"] + scn["Lease_Cost"] + scn["Overtime_Cost"] + scn["Penalty_Cost"]
        scn["Cost_per_Mile"] = np.where(scn["Truck_Miles"]>0, scn["Total_Cost"]/scn["Truck_Miles"], np.nan)

    elif scenario == "New Facility":
        cand = st.selectbox("Facility city", options=sorted(TERMINAL_COORDS.keys()))
        lat, lon = TERMINAL_COORDS[cand]
        radius = st.slider("Capture radius (miles)", 25, 400, 150, 25)
        divert = st.slider("Divert share (%)", 0, 80, 25, 1) / 100.0
        miles_factor = st.slider("Miles reduction factor", 0.60, 0.98, 0.85, 0.01)
        invest = st.number_input("Monthly facility fixed cost ($)", 10000.0, 500000.0, 85000.0, 5000.0)

        geo = add_coords(scn)
        d = np.where(~np.isnan(geo["Dest_Lat"]),
                     [haversine_miles(lat, lon, a, b) for a,b in zip(geo["Dest_Lat"], geo["Dest_Lon"])],
                     np.nan)
        geo["Dist_to_Facility"] = d
        eligible = geo.index[geo["Dist_to_Facility"]<=radius].tolist()
        rng = np.random.default_rng(9)
        n_div = int(round(len(eligible)*divert))
        if n_div>0:
            divert_idx = set(rng.choice(eligible, size=n_div, replace=False).tolist())
            scn.loc[list(divert_idx), "Truck_Miles"] *= miles_factor
            scn.loc[list(divert_idx), "Toll_Cost"] *= miles_factor

            scn = compute_costs(scn, mpg, diesel, driver_cpm, lease_cpm, overtime, penalty)

    else:
        top_n = st.slider("Target top N origins (highest late rate)", 1, 10, 3, 1)
        shift_share = st.slider("Shift share (%)", 0, 70, 20, 1) / 100.0
        miles_factor = st.slider("Miles reduction factor", 0.70, 0.99, 0.90, 0.01)
        invest = st.number_input("Monthly program cost ($)", 0.0, 250000.0, 15000.0, 1000.0)

        if "Origin" in scn.columns:
            stats = scn.groupby("Origin").agg(late_rate=("Late_Flag","mean"), trips=("Origin","count")).sort_values(["late_rate","trips"], ascending=False)
            targets = stats.head(top_n).index.tolist()
            idx = scn.index[scn["Origin"].isin(targets)].tolist()
            rng = np.random.default_rng(11)
            n_shift = int(round(len(idx)*shift_share))
            if n_shift>0:
                shift_idx = set(rng.choice(idx, size=n_shift, replace=False).tolist())
                scn.loc[list(shift_idx), "Truck_Miles"] *= miles_factor
                scn.loc[list(shift_idx), "Toll_Cost"] *= miles_factor
                scn = compute_costs(scn, mpg, diesel, driver_cpm, lease_cpm, overtime, penalty)

    def summary(x):
        return {
            "Trips": len(x),
            "Late%": (x["Late_Flag"].mean()*100) if len(x) else 0,
            "TotalCost": float(x["Total_Cost"].sum()),
            "TotalMiles": float(x["Truck_Miles"].sum()),
            "AvgCPM": float(x["Cost_per_Mile"].mean())
        }

    b = summary(base)
    s = summary(scn)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Late %", f"{s['Late%']:.1f}%", f"{(s['Late%']-b['Late%']):+.1f} pts")
    c2.metric("Total Cost", f"${s['TotalCost']:,.0f}", f"${(s['TotalCost']-b['TotalCost']):,.0f}")
    c3.metric("Avg Cost/Mile", f"${s['AvgCPM']:.2f}", f"${(s['AvgCPM']-b['AvgCPM']):+.2f}")
    c4.metric("Monthly Investment", f"${invest:,.0f}")

    st.session_state["scenario_df"] = scn
    st.session_state["invest"] = invest

    st.subheader("Scenario preview (first 50 rows)")
    show_cols = [c for c in ["Origin","Destination","Truck_Miles","Toll_Cost","Late_Flag","Total_Cost","Cost_per_Mile"] if c in scn.columns]
    st.dataframe(scn[show_cols].head(50), use_container_width=True, height=320)

    st.download_button("Download scenario CSV", data=scn.to_csv(index=False).encode("utf-8"), file_name="scenario_output.csv", mime="text/csv")

with tab_fin:
    st.subheader("Baseline vs Scenario Financials")

    scn = st.session_state.get("scenario_df", base)
    invest = float(st.session_state.get("invest", 0.0))

    base_total = float(base["Total_Cost"].sum())
    scn_total = float(scn["Total_Cost"].sum())
    savings = base_total - scn_total

    roi = (savings/invest) if invest>0 else np.nan
    payback = (invest/savings) if (invest>0 and savings>0) else np.nan

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Baseline Cost", f"${base_total:,.0f}")
    c2.metric("Scenario Cost", f"${scn_total:,.0f}")
    c3.metric("Monthly Savings", f"${savings:,.0f}")
    c4.metric("Payback (months)", "—" if np.isnan(payback) else f"{payback:.1f}")

    st.markdown("### Cost Breakdown")
    def breakdown(x):
        return pd.DataFrame({
            "Fuel":[x["Fuel_Cost"].sum()],
            "Tolls":[x["Toll_Cost"].sum()],
            "Driver":[x["Driver_Cost"].sum()],
            "Lease":[x["Lease_Cost"].sum()],
            "Overtime":[x["Overtime_Cost"].sum()],
            "Penalties":[x["Penalty_Cost"].sum()],
            "Total":[x["Total_Cost"].sum()]
        }).T

    b = breakdown(base).rename(columns={0:"Baseline"})
    s = breakdown(scn).rename(columns={0:"Scenario"})
    out = b.join(s)
    out["Delta"] = out["Scenario"] - out["Baseline"]
    st.dataframe(out.style.format("{:,.0f}"), use_container_width=True)
