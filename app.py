"""
ParkSense — Bengaluru Parking Enforcement Intelligence
Working Prototype | Gridlock Hackathon
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, AntPath
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import h3
from datetime import datetime, timedelta

# ════════════════════════════════════════════════════════════════════════
#  🚀 CONFIGURATION — RAILWAY API
# ════════════════════════════════════════════════════════════════════════
BASE_URL = "https://web-production-25afb.up.railway.app"
PREDICT_ENDPOINT = f"{BASE_URL}/predict"
FEATURE_IMPORTANCE_ENDPOINT = f"{BASE_URL}/feature_importance"
REPEAT_OFFENDERS_ENDPOINT = f"{BASE_URL}/repeat_offenders"

# ── HARDCODED PREDICTION DATE ──
PREDICTION_DATE = "2024-04-09"

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkSense | Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CLEAN CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { 
    background: #0a0c14; 
}
[data-testid="stSidebar"] { 
    background: #0f1117; 
}
section[data-testid="stSidebar"] { 
    display:none; 
}

/* ── HERO SECTION ── */
.hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f35 50%, #0f1117 100%);
    border: 1px solid rgba(231,76,60,0.2);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(231,76,60,0.08);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #e74c3c);
    background-size: 200% 100%;
    animation: gradientMove 3s ease infinite;
}
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #f1c40f 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s ease-in-out infinite;
    margin: 0;
    line-height: 1.2;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.hero-sub {
    color: #8892a4;
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 500;
}
.hero-badge {
    display: inline-block;
    background: rgba(231,76,60,0.12);
    border: 1px solid rgba(231,76,60,0.25);
    color: #e74c3c;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 8px;
    margin-top: 10px;
    letter-spacing: 0.3px;
}
.status-badge {
    display: inline-block;
    background: rgba(39,174,96,0.15);
    border: 1px solid rgba(39,174,96,0.3);
    color: #27ae60;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.75rem;
    font-weight: 700;
    box-shadow: 0 0 20px rgba(39,174,96,0.15);
}

/* ── KPI CARDS ── */
.kpi {
    background: #0f1117;
    border: 1px solid #1e2640;
    border-radius: 14px;
    padding: 18px 20px;
    transition: all 0.3s ease;
    cursor: default;
}
.kpi:hover {
    border-color: #e74c3c;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(231,76,60,0.06);
}
.kpi-val {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.kpi-lab {
    font-size: 0.7rem;
    color: #8892a4;
    margin-top: 4px;
    letter-spacing: 0.3px;
    font-weight: 600;
    text-transform: uppercase;
}
.kpi-delta {
    font-size: 0.72rem;
    margin-top: 2px;
    opacity: 0.8;
}

/* ── SECTIONS & CARDS ── */
.section-head {
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    border-left: 3px solid #e74c3c;
    padding-left: 12px;
    margin: 18px 0 14px 0;
    letter-spacing: -0.2px;
}
.card {
    background: #0f1117;
    border: 1px solid #1e2640;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.card:hover {
    border-color: #2a3450;
}
.card-red    { border-left: 3px solid #e74c3c; }
.card-amber  { border-left: 3px solid #f39c12; }
.card-green  { border-left: 3px solid #27ae60; }
.card-blue   { border-left: 3px solid #3498db; }
.card-purple { border-left: 3px solid #9b59b6; }

.risk-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.pill-red    { background: rgba(231,76,60,0.15); color: #e74c3c; border:1px solid rgba(231,76,60,0.25); }
.pill-amber  { background: rgba(243,156,18,0.15); color: #f39c12; border:1px solid rgba(243,156,18,0.25); }
.pill-green  { background: rgba(39,174,96,0.15);  color: #27ae60; border:1px solid rgba(39,174,96,0.25); }
.pill-blue   { background: rgba(52,152,219,0.15); color: #3498db; border:1px solid rgba(52,152,219,0.25); }

.route-stop {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #0f1117;
    border: 1px solid #1e2640;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 6px;
    flex-wrap: wrap;
    transition: border-color 0.2s;
}
.route-stop:hover {
    border-color: #f39c12;
}
.stop-num {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg,#e74c3c,#f39c12);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 800;
    color: #fff;
    flex-shrink: 0;
}
.stop-info { flex: 1; min-width: 120px; }
.stop-name { font-size: 0.78rem; font-weight: 600; color: #fff; }
.stop-meta { font-size: 0.7rem; color: #8892a4; margin-top: 1px; }

.insight-box {
    background: linear-gradient(135deg,#1a1220,#1a1a2e);
    border: 1px solid #2d2060;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0;
    color: #ccc;
    font-size: 0.82rem;
}

div[data-testid="stTabs"] button {
    font-size: 0.9rem;
    font-weight: 600;
    color: #8892a4;
    padding: 8px 16px;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #fff;
    border-bottom: 2px solid #e74c3c;
}

.info-box {
    background: rgba(52,152,219,0.06);
    border: 1px solid rgba(52,152,219,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 20px;
    color: #8892a4;
}

.stButton button[data-testid="baseButton-primary"] {
    box-shadow: 0 0 20px rgba(231,76,60,0.15);
    transition: all 0.3s ease;
}
.stButton button[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 0 40px rgba(231,76,60,0.25);
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# ── API Helper Functions ──────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_predictions(timestamp: str):
    try:
        payload = {"timestamp": timestamp}
        response = requests.post(
            PREDICT_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch predictions: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_feature_importance():
    try:
        response = requests.get(FEATURE_IMPORTANCE_ENDPOINT, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_repeat_offenders():
    try:
        response = requests.get(REPEAT_OFFENDERS_ENDPOINT, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def patrol_route(zones_df):
    z = zones_df.reset_index(drop=True).copy()
    if len(z) <= 1:
        return z
    visited, remaining = [0], list(range(1, len(z)))
    while remaining:
        last = visited[-1]
        dists = [
            ((z.loc[i, "latitude"] - z.loc[last, "latitude"]) ** 2 +
             (z.loc[i, "longitude"] - z.loc[last, "longitude"]) ** 2) ** 0.5
            for i in remaining
        ]
        nearest = remaining[int(np.argmin(dists))]
        visited.append(nearest)
        remaining.remove(nearest)
    return z.loc[visited].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════
#  🏆 CLEAN HERO HEADER
# ════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span style="font-size:2.8rem; color:#fff; line-height:1;">🚦</span>
        <div>
            <div class="hero-title">ParkSense</div>
            <div class="hero-sub">Predictive Parking Enforcement Intelligence · Bengaluru Traffic Police</div>
            <div style="margin-top:12px">
            <span class="hero-badge">📍 40+ H3 Clusters</span>
            <span class="hero-badge">⚡ Real-time API</span>
            <span class="status-badge">● Live</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────
st.info(f"""
🧠 **Model Insight:** Trained on historical data up to **April 8, 2024**.  
Using lag features (previous hour/day patterns), the model accurately predicts the next day **{PREDICTION_DATE}**.  
🚀 *In a real system, we would retrain daily to enable continuous rolling predictions.*
""")

col1, col2, col3 = st.columns([1, 1, 0.8])
with col1:
    st.markdown(f"**📅 Prediction Date:** `{PREDICTION_DATE}`")
with col2:
    selected_hour = st.slider("🕐 Hour (IST)", 0, 23, 5, format="%d:00")
with col3:
    fetch_clicked = st.button("🚀 Fetch Predictions", use_container_width=True, type="primary")

timestamp_str = f"{PREDICTION_DATE} {selected_hour:02d}:00:00"

# ── Fetch Data ─────────────────────────────────────────────────────────
if 'api_data' not in st.session_state:
    st.session_state['api_data'] = None
if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = None
if 'repeat_offenders' not in st.session_state:
    st.session_state['repeat_offenders'] = None

if fetch_clicked or st.session_state['api_data'] is None:
    with st.spinner(f"Fetching predictions for {timestamp_str}..."):
        data = fetch_predictions(timestamp_str)
        if data:
            st.session_state['api_data'] = data
    
    if st.session_state['feature_importance'] is None:
        st.session_state['feature_importance'] = fetch_feature_importance()
    if st.session_state['repeat_offenders'] is None:
        st.session_state['repeat_offenders'] = fetch_repeat_offenders()

if st.session_state['api_data'] is None:
    st.info("👈 Select an hour, then click 'Fetch Predictions' to load live data.")
    st.stop()

# ── Parse Data ─────────────────────────────────────────────────────────
api_data = st.session_state['api_data']
feature_importance_data = st.session_state['feature_importance']
repeat_offenders_data = st.session_state['repeat_offenders']

clusters = api_data.get('clusters', [])
city_stats = api_data.get('city_stats', {})

if not clusters:
    st.warning("⚠️ No clusters returned from the API.")
    st.stop()

df_clusters = pd.DataFrame(clusters)

# ── KPI ROW ────────────────────────────────────────────────────────────
total_clusters = len(df_clusters)
total_violations = int(df_clusters['total_predicted_violations'].sum())
avg_violations = total_violations / total_clusters if total_clusters > 0 else 0

if city_stats:
    total_violations = city_stats.get('total_violations', total_violations)
    total_clusters = city_stats.get('total_clusters', total_clusters)
    avg_violations = city_stats.get('avg_violations', avg_violations)

k = st.columns(5)
kpis = [
    (f"{total_violations:,}", "Predicted Violations", "#e74c3c", f"At {selected_hour}:00"),
    (f"{total_clusters}", "Active Clusters", "#f39c12", "City-wide"),
    (f"{avg_violations:.1f}", "Avg per Cluster", "#27ae60", "Risk score avg"),
    (f"{len(df_clusters[df_clusters['risk_score'] > 0.4])}", "High Risk Zones", "#9b59b6", "Risk > 0.4"),
    (f"{df_clusters['h3_cell'].nunique()}", "H3 Cells Mapped", "#3498db", "Resolution 7"),
]
for col, (val, lab, color, delta) in zip(k, kpis):
    with col:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-val" style="color:{color}">{val}</div>
            <div class="kpi-lab">{lab}</div>
            <div class="kpi-delta" style="color:{color}88">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TABS 
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🗺️  Hotspot Intelligence",
    "👮  Patrol Scheduler",
    "🚚  Flipkart Corridor"
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — HOTSPOT INTELLIGENCE 
# ══════════════════════════════════════════════════════════════════════
with tab1:
    col_ctrl, col_map = st.columns([1, 2], gap="medium")

    with col_ctrl:
        st.markdown('<div class="section-head">🎛️ Map Controls</div>', unsafe_allow_html=True)
        show_dark = st.toggle("Dark zone hexagons", True)
        show_heat = st.toggle("Heatmap overlay", False)
        st.caption("🟡 Yellow = Named Junction  |  🟣 Purple = Dark Zone")
        st.markdown("---")

        st.markdown('<div class="section-head">⚡ Risk Snapshot</div>', unsafe_allow_html=True)
        n_high = len(df_clusters[df_clusters['risk_score'] > 0.4])
        avg_risk = df_clusters['risk_score'].mean()
        st.markdown(f"""
        <div class="card card-red">
            <div style="font-size:1.6rem;font-weight:800;color:#e74c3c">{n_high}</div>
            <div style="font-size:0.75rem;color:#8892a4">High-risk zones (risk > 0.4) at {selected_hour}:00</div>
        </div>
        <div class="card card-amber">
            <div style="font-size:1.6rem;font-weight:800;color:#f39c12">{avg_risk*100:.0f}%</div>
            <div style="font-size:0.75rem;color:#8892a4">Avg risk score city-wide</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head">🔥 Top Zones Now</div>', unsafe_allow_html=True)
        top3 = df_clusters.sort_values('risk_score', ascending=False).head(3)
        for i, (_, r) in enumerate(top3.iterrows()):
            pct = r['risk_score'] * 100
            pill = "pill-red" if pct > 70 else "pill-amber"
            st.markdown(f"""<div class="card" style="padding:10px 14px;margin-bottom:6px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div style="font-size:0.78rem;font-weight:600;color:#fff">Zone #{i+1}</div>
                    <span class="risk-pill {pill}">{pct:.0f}% risk</span>
                </div>
                <div style="font-size:0.7rem;color:#8892a4;margin-top:2px">
                    {r['area_name'][:30]} · {r['latitude']:.4f}°N
                </div>
            </div>""", unsafe_allow_html=True)

    with col_map:
        st.markdown('<div class="section-head">🗺️ Live Violation Map</div>', unsafe_allow_html=True)

        m = folium.Map(location=[12.97, 77.59], zoom_start=12,
                       tiles="CartoDB dark_matter")

        if show_heat:
            pts = [[r['latitude'], r['longitude']] for _, r in df_clusters.iterrows() if r['total_predicted_violations'] > 0]
            wts = [r['total_predicted_violations'] for _, r in df_clusters.iterrows() if r['total_predicted_violations'] > 0]
            if pts:
                HeatMap(list(zip([p[0] for p in pts], [p[1] for p in pts], wts)),
                        radius=16, blur=12, min_opacity=0.25,
                        gradient={"0.3": "#3498db", "0.6": "#f39c12",
                                  "0.8": "#e74c3c", "1.0": "#fff"}).add_to(m)

        for _, r in df_clusters.sort_values('total_predicted_violations', ascending=False).iterrows():
            cnt = r['total_predicted_violations']
            if cnt == 0:
                continue
            
            is_named = "BTP" in str(r['area_name'])
            area_name = r.get('area_name', 'Unknown')
            preds = r.get('predictions', {})
            top_viol = max(preds, key=preds.get) if preds else 'N/A'

            breakdown_html = ""
            if isinstance(preds, dict):
                for vtype, vcount in preds.items():
                    if vcount > 0:
                        breakdown_html += f"<li>{vtype}: <b>{vcount}</b></li>"

            popup_html = f"""
            <div style='font-family:sans-serif; min-width:200px;'>
                <b style='font-size:1.1rem;'>{area_name[:50]}</b><br>
                <hr style='margin:4px 0; border-color:#333;'>
                <b>Total Violations:</b> {cnt}<br>
                <b>Risk Score:</b> {r['risk_score']:.3f}<br>
                <b>Top Violation:</b> {top_viol}<br>
                <b>Peak Hour:</b> {r['historical_peak_hour']}:00<br>
                <b>Severe Ratio:</b> {r['historical_severe_ratio']*100:.0f}%<br>
                <b>Breakdown:</b>
                <ul style='margin:4px 0; padding-left:16px;'>
                    {breakdown_html}
                </ul>
                <i>{r['latitude']:.4f}°N, {r['longitude']:.4f}°E</i>
            </div>
            """

            if is_named:
                rad = min(max(int(cnt * 2), 5), 25)
                color = "#f39c12"
                folium.CircleMarker(
                    [r['latitude'], r['longitude']], radius=rad,
                    color=color, fill=True, fill_color=color, fill_opacity=0.75,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{area_name[:25]} ({cnt} viol.)"
                ).add_to(m)
            else:
                if show_dark:
                    try:
                        h3_cell = r['h3_cell']
                        bnd = h3.cell_to_boundary(h3_cell)
                        op = 0.15 + min(cnt / 20, 0.5)
                        folium.Polygon(
                            [[lat, lng] for lat, lng in bnd],
                            color="#9b59b6", fill=True, fill_color="#9b59b6",
                            fill_opacity=op, weight=2,
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=f"Dark Zone: {area_name[:20]} ({cnt} viol.)"
                        ).add_to(m)
                    except:
                        folium.CircleMarker(
                            [r['latitude'], r['longitude']], radius=10,
                            color="#9b59b6", fill=True, fill_color="#9b59b6", fill_opacity=0.5,
                            popup=folium.Popup(popup_html, max_width=300)
                        ).add_to(m)

        st_folium(m, width=None, height=520, returned_objects=[])

    # ── Bottom Section ──
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1.2, 1, 1])

    with b1:
        st.markdown('<div class="section-head">🎯 Impact Score — Top Zones</div>', unsafe_allow_html=True)
        df_clusters['impact_score'] = (
            (df_clusters['total_predicted_violations'] / df_clusters['total_predicted_violations'].max()) * 0.45 +
            df_clusters['historical_severe_ratio'] * 0.25 +
            df_clusters['historical_repeat_ratio'] * 0.20 +
            df_clusters['risk_score'] * 0.10
        )
        top_impact = df_clusters.sort_values('impact_score', ascending=False).head(8)

        fig_imp = go.Figure(go.Bar(
            x=top_impact['impact_score'].values[::-1],
            y=[n[:28] for n in top_impact['area_name'].values[::-1]],
            orientation="h",
            marker=dict(
                color=top_impact['impact_score'].values[::-1],
                colorscale=[[0, "#3498db"], [0.5, "#f39c12"], [1, "#e74c3c"]],
                showscale=False
            ),
            hovertemplate="%{y}<br>Impact: %{x:.3f}<extra></extra>"
        ))
        fig_imp.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#8892a4", height=280,
            margin=dict(l=0, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1e2640", title="Impact Score"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with b2:
        st.markdown('<div class="section-head">🧠 Model Explainability</div>', unsafe_allow_html=True)
        if feature_importance_data:
            fi_df = pd.DataFrame(list(feature_importance_data.items()), columns=["Feature", "Importance"])
            fi_df = fi_df.sort_values("Importance", ascending=False).head(12)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "#1e2640"], [0.5, "#3498db"], [1, "#e74c3c"]],
                    showscale=False
                ),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>"
            ))
            fig_fi.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font_color="#8892a4", height=280,
                margin=dict(l=0, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#1e2640", title="Importance"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown("""<div class="insight-box">
                💡 <b style="color:#f39c12">Key insight:</b> Historical time patterns and
                location clusters are the strongest predictors.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Feature importance data not available.")

    with b3:
        st.markdown('<div class="section-head">🔁 Repeat Offender Watchlist</div>', unsafe_allow_html=True)
        if repeat_offenders_data:
            repeat_offenders = repeat_offenders_data.get('repeat_offenders', [])
            if repeat_offenders:
                top_rt = repeat_offenders[:8]
                for r in top_rt:
                    vt = r.get('vehicle_type', 'Unknown').title()
                    cnt = r.get('total_violations', 0)
                    pill = "pill-red" if cnt >= 15 else "pill-amber" if cnt >= 10 else "pill-blue"
                    junc = r.get('top_junction', 'Unknown')[:20]
                    vehicle = r.get('vehicle_number', 'Unknown')[-8:]
                    st.markdown(f"""<div class="card" style="padding:10px 14px;margin-bottom:5px">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <div>
                                <div style="font-size:0.78rem;font-weight:700;color:#fff">
                                    {vehicle}
                                </div>
                                <div style="font-size:0.68rem;color:#8892a4">{vt} · {junc}</div>
                            </div>
                            <span class="risk-pill {pill}">{cnt}×</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                total_offenders = repeat_offenders_data.get('count', len(repeat_offenders))
                st.markdown(f"""<div class="insight-box" style="margin-top:8px">
                    ⚠️ <b style="color:#9b59b6">{total_offenders:,} serial violators</b> —
                    targeting the top 50 covers significant violations.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No repeat offenders data.")
        else:
            st.info("Repeat offenders not available.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — PATROL SCHEDULER 
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">👮 AI-Generated Patrol Plan</div>', unsafe_allow_html=True)
    st.markdown(f"<span style='color:#8892a4;font-size:0.85rem'>Route optimized for <b>{PREDICTION_DATE}</b> at <b>{selected_hour}:00</b> IST</span>", unsafe_allow_html=True)

    pc1, pc2, pc3, pc4 = st.columns([1, 1, 1, 1])
    with pc1:
        sched_hour = st.slider("Starting hour", 0, 23, 4, key="sched_hr")
    with pc2:
        n_cars = st.slider("Patrol cars available", 1, 10, 3)
    with pc3:
        n_zones = st.slider("Zones to cover", 4, 12, 8)

    day_risk = df_clusters.copy()
    day_risk['combined_score'] = day_risk['risk_score']

    top_zones = day_risk.sort_values('combined_score', ascending=False).head(n_zones).reset_index(drop=True)
    route_df = patrol_route(top_zones)

    total_risk = day_risk['risk_score'].sum()
    covered_risk = route_df['risk_score'].sum()
    coverage_pct = (covered_risk / total_risk * 100) if total_risk > 0 else 0
    avg_risk = route_df['risk_score'].mean() * 100
    estimated_prevented = int(route_df['total_predicted_violations'].sum() * 0.7)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""<div class="kpi" style="margin-top:12px">
            <div class="kpi-val" style="color:#e74c3c">{coverage_pct:.1f}%</div>
            <div class="kpi-lab">Risk coverage</div>
            <div class="kpi-delta" style="color:#e74c3c88">Top {n_zones} zones</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""<div class="kpi" style="margin-top:12px">
            <div class="kpi-val" style="color:#f39c12">{n_zones}</div>
            <div class="kpi-lab">Zones assigned</div>
            <div class="kpi-delta" style="color:#f39c1288">{int(np.ceil(n_zones / n_cars))} zones per car</div>
        </div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""<div class="kpi" style="margin-top:12px">
            <div class="kpi-val" style="color:#27ae60">{avg_risk:.0f}%</div>
            <div class="kpi-lab">Avg zone risk</div>
            <div class="kpi-delta" style="color:#27ae6088">At {sched_hour}:00</div>
        </div>""", unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""<div class="kpi" style="margin-top:12px">
            <div class="kpi-val" style="color:#9b59b6">{estimated_prevented:,}</div>
            <div class="kpi-lab">Est. violations prevented</div>
            <div class="kpi-delta" style="color:#9b59b688">With active enforcement</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    map_col, roster_col = st.columns([1.5, 1], gap="medium")

    with map_col:
        st.markdown('<div class="section-head">🗺️ Optimal Patrol Route</div>', unsafe_allow_html=True)
        pm = folium.Map(location=[12.97, 77.59], zoom_start=12, tiles="CartoDB dark_matter")

        route_coords = [[r['latitude'], r['longitude']] for _, r in route_df.iterrows()]
        if len(route_coords) > 1:
            AntPath(route_coords, color="#f39c12", weight=3, opacity=0.8,
                    dash_array=[10, 20], delay=800).add_to(pm)

        car_colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c"]

        for i, (_, r) in enumerate(route_df.iterrows()):
            risk = r['combined_score']
            color = "#e74c3c" if risk > 0.6 else "#f39c12" if risk > 0.35 else "#3498db"
            car_id = i % n_cars
            
            preds = r.get('predictions', {})
            breakdown_html = ""
            if isinstance(preds, dict):
                for vtype, vcount in preds.items():
                    if vcount > 0:
                        breakdown_html += f"<li>{vtype}: <b>{vcount}</b></li>"

            icon_html = f"""<div style="
                width:32px;height:32px;border-radius:50%;
                background:linear-gradient(135deg,{color},{color}88);
                border:2px solid {color};
                display:flex;align-items:center;justify-content:center;
                font-weight:800;font-size:13px;color:#fff;
                box-shadow:0 0 12px {color}66">{i+1}</div>"""

            maps_url = f"https://www.google.com/maps?q={r['latitude']},{r['longitude']}"
            popup_html = f"""
            <div style='font-family:sans-serif; min-width:200px;'>
                <b style='color:{color};'>Stop #{i+1}</b><br>
                <b>{r['area_name'][:30]}</b><br>
                Risk: <b>{risk*100:.0f}%</b><br>
                Violations: {r['total_predicted_violations']}<br>
                Peak Hour: {r['historical_peak_hour']}:00<br>
                Car: {car_id+1}<br>
                <b>Breakdown:</b>
                <ul style='margin:4px 0; padding-left:16px;'>
                    {breakdown_html}
                </ul>
                <a href="{maps_url}" target="_blank" style="color:#3498db;">🗺️ Navigate</a>
            </div>
            """

            folium.Marker(
                [r['latitude'], r['longitude']],
                icon=folium.DivIcon(html=icon_html, icon_size=(32, 32), icon_anchor=(16, 16)),
                tooltip=f"Stop #{i+1}: {r['area_name'][:20]} — {risk*100:.0f}% risk",
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(pm)

        st_folium(pm, width=None, height=450, returned_objects=[])

        st.markdown('<div class="section-head">📊 Before vs After Enforcement</div>', unsafe_allow_html=True)
        before = route_df['total_predicted_violations'].sum()
        after = int(before * 0.35)
        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(name="Without enforcement", x=["Violations"], y=[before], marker_color="#e74c3c", width=0.3))
        fig_ba.add_trace(go.Bar(name="With enforcement (est.)", x=["Violations"], y=[after], marker_color="#27ae60", width=0.3))
        fig_ba.update_layout(
            barmode="group", height=180,
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#8892a4", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(font=dict(color="#8892a4", size=11)),
            xaxis=dict(gridcolor="#1e2640"),
            yaxis=dict(gridcolor="#1e2640", title="Violations")
        )
        st.plotly_chart(fig_ba, use_container_width=True)

    with roster_col:
        st.markdown('<div class="section-head">📋 Patrol Roster</div>', unsafe_allow_html=True)

        for i, (_, r) in enumerate(route_df.iterrows()):
            car_id = i % n_cars
            risk_pct = r['combined_score'] * 100
            pill = "pill-red" if risk_pct > 60 else "pill-amber" if risk_pct > 35 else "pill-blue"
            car_color = car_colors[car_id % len(car_colors)]
            maps_url = f"https://www.google.com/maps?q={r['latitude']},{r['longitude']}"

            st.markdown(f"""
            <div class="route-stop" style="flex-wrap:wrap; gap:6px; padding:12px 14px;">
                <div class="stop-num">{i+1}</div>
                <div class="stop-info" style="flex:1; min-width:100px;">
                    <div class="stop-name" style="font-size:0.7rem;">📍 {r['area_name'][:20]}</div>
                    <div class="stop-meta">
                        Peak: {r['historical_peak_hour']}:00 &nbsp;·&nbsp;
                        <span style="color:{car_color}; font-weight:600;">🚔 Car {car_id+1}</span>
                        &nbsp;·&nbsp; ⚠️ {r['total_predicted_violations']}
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                    <a href="{maps_url}" target="_blank" style="
                        background:#0f1117; color:#3498db; padding:4px 12px; 
                        border-radius:16px; text-decoration:none; font-size:0.65rem; 
                        font-weight:700; border:1px solid #3498db; white-space:nowrap;
                    ">🗺️ Navigate</a>
                    <span class="risk-pill {pill}" style="font-size:0.65rem;">{risk_pct:.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        export_df = route_df[["cluster_id", "area_name", "latitude", "longitude",
                              "total_predicted_violations", "risk_score", "historical_peak_hour"]].copy()
        export_df.columns = ["Cluster ID", "Area", "Lat", "Lng", "Violations", "Risk", "Peak Hour"]
        export_df.index = range(1, len(export_df) + 1)
        st.download_button("📥 Export Patrol Plan (CSV)", export_df.to_csv().encode(),
                          "patrol_plan.csv", "text/csv", use_container_width=True)


        st.markdown('<div class="section-head">📈 Coverage Projection: +1 Patrol Car</div>', unsafe_allow_html=True)
        
        # Calculate current coverage vs. potential with +1 car
        uncovered_zones = day_risk[~day_risk['cluster_id'].isin(route_df['cluster_id'])]
        extra_violations = int(uncovered_zones['total_predicted_violations'].head(4).sum())
        current_violations = int(route_df['total_predicted_violations'].sum())
        new_violations = current_violations + extra_violations
        
        if current_violations > 0 and extra_violations > 0:
            pct_increase = (extra_violations / current_violations) * 100
            new_coverage_pct = ((new_violations) / total_violations) * 100 if total_violations > 0 else 0
            current_coverage_pct = (current_violations / total_violations) * 100 if total_violations > 0 else 0
            
            col_w1, col_w2, col_w3 = st.columns(3)
            
            with col_w1:
                st.markdown(f"""
                <div class="kpi" style="padding:12px 14px; border-left: 3px solid #27ae60;">
                    <div class="kpi-val" style="font-size:1.1rem; color:#27ae60;">{extra_violations:,}</div>
                    <div class="kpi-lab">Additional Violations Covered</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_w2:
                st.markdown(f"""
                <div class="kpi" style="padding:12px 14px; border-left: 3px solid #f39c12;">
                    <div class="kpi-val" style="font-size:1.1rem; color:#f39c12;">+{pct_increase:.1f}%</div>
                    <div class="kpi-lab">Increase in Coverage</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_w3:
                st.markdown(f"""
                <div class="kpi" style="padding:12px 14px; border-left: 3px solid #3498db;">
                    <div class="kpi-val" style="font-size:1.1rem; color:#3498db;">{new_coverage_pct:.1f}%</div>
                    <div class="kpi-lab">Total Coverage with +1 Car</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"🚔 One additional patrol car covers the next 4 highest-risk zones, boosting coverage from **{current_coverage_pct:.1f}%** to **{new_coverage_pct:.1f}%** of all predicted violations.")
            
        elif extra_violations == 0:
            st.success("✅ All high-risk zones are already covered by the current deployment.")
        else:
            st.info("ℹ️ Add more patrol cars to cover additional zones.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — FLIPKART CORRIDOR
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">🚚 Delivery Corridor Intelligence</div>', unsafe_allow_html=True)
    st.markdown(f"<span style='color:#8892a4;font-size:0.85rem'>Protecting Flipkart's delivery window using real-time predictions for <b>{PREDICTION_DATE}</b></span>", unsafe_allow_html=True)

    HUBS = {
        "HSR Layout Hub": {"lat": 12.9116, "lng": 77.6389, "vans": 120},
        "Koramangala Hub": {"lat": 12.9352, "lng": 77.6245, "vans": 95},
        "Whitefield Hub": {"lat": 12.9698, "lng": 77.7500, "vans": 80},
        "Yeshwanthpur Hub": {"lat": 13.0289, "lng": 77.5538, "vans": 65},
    }

    delivery_v = df_clusters['total_predicted_violations'].sum()
    high_risk = df_clusters[df_clusters['risk_score'] > 0.4]
    night_v = int(df_clusters[df_clusters['historical_peak_hour'].isin([0,1,2,3,4,5,22,23])]['total_predicted_violations'].sum())

    fk1, fk2, fk3, fk4 = st.columns(4)
    with fk1:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-val" style="color:#27ae60">{delivery_v:,}</div>
            <div class="kpi-lab">Total Predicted Violations</div>
            <div class="kpi-delta" style="color:#27ae6088">Window: {selected_hour}:00</div>
        </div>""", unsafe_allow_html=True)
    with fk2:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-val" style="color:#f39c12">{len(high_risk)}</div>
            <div class="kpi-lab">High-Risk Zones</div>
            <div class="kpi-delta" style="color:#f39c1288">Risk > 0.4</div>
        </div>""", unsafe_allow_html=True)
    with fk3:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-val" style="color:#e74c3c">{night_v:,}</div>
            <div class="kpi-lab">Night Violations</div>
            <div class="kpi-delta" style="color:#e74c3c88">Peak offence period</div>
        </div>""", unsafe_allow_html=True)
    with fk4:
        total_vans = sum(h["vans"] for h in HUBS.values())
        st.markdown(f"""<div class="kpi">
            <div class="kpi-val" style="color:#3498db">{total_vans}</div>
            <div class="kpi-lab">Delivery vans monitored</div>
            <div class="kpi-delta" style="color:#3498db88">Across 4 Bengaluru hubs</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fm1, fm2 = st.columns([1.5, 1], gap="medium")

    with fm1:
        st.markdown('<div class="section-head">🗺️ Corridor Risk Map</div>', unsafe_allow_html=True)

        cm_map = folium.Map(location=[12.97, 77.60], zoom_start=12, tiles="CartoDB dark_matter")

        for _, r in high_risk.sort_values('risk_score', ascending=False).head(20).iterrows():
            try:
                bnd = h3.cell_to_boundary(r['h3_cell'])
                op = 0.2 + r['risk_score'] * 0.5
                folium.Polygon(
                    [[lat, lng] for lat, lng in bnd],
                    color="#e74c3c", fill=True, fill_color="#e74c3c",
                    fill_opacity=min(op, 0.75), weight=1,
                    popup=folium.Popup(
                        f"<b style='color:#e74c3c'>⚠️ Risk zone</b><br>"
                        f"Risk: <b>{r['risk_score']*100:.0f}%</b><br>"
                        f"Violations: {r['total_predicted_violations']}<br>"
                        f"Pre-clear before peak", max_width=200)
                ).add_to(cm_map)
            except:
                pass

        for name, hub in HUBS.items():
            folium.Marker(
                [hub["lat"], hub["lng"]],
                icon=folium.DivIcon(html=f"""
                    <div style="background:#f39c12;border:2px solid #fff;
                    border-radius:8px;padding:4px 8px;font-size:11px;
                    font-weight:700;color:#000;white-space:nowrap;
                    box-shadow:0 2px 8px rgba(0,0,0,0.5)">
                    🏭 {name.replace(' Hub', '')}
                    </div>""", icon_size=(140, 30), icon_anchor=(70, 15)),
                popup=folium.Popup(f"<b>🏭 {name}</b><br>{hub['vans']} delivery vans", max_width=160)
            ).add_to(cm_map)

        st_folium(cm_map, width=None, height=430, returned_objects=[])

        st.markdown('<div class="section-head">📈 Top Violation Clusters</div>', unsafe_allow_html=True)
        top_clusters = df_clusters.nlargest(10, 'total_predicted_violations')
        fig_h = go.Figure(go.Bar(
            x=top_clusters['total_predicted_violations'],
            y=top_clusters['area_name'].str[:30],
            orientation='h',
            marker=dict(color=top_clusters['risk_score'], colorscale='Reds', showscale=True),
            hovertemplate="%{y}: %{x} violations<extra></extra>"
        ))
        fig_h.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#8892a4", height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="#1e2640", title="Violations"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with fm2:
        st.markdown('<div class="section-head">⚠️ Nearby Risk Zones — Pre-clear Alerts</div>', unsafe_allow_html=True)
        threat = high_risk.sort_values('risk_score', ascending=False).head(10)
        for _, r in threat.iterrows():
            rp = r['risk_score'] * 100
            pill = "pill-red" if rp > 60 else "pill-amber" if rp > 40 else "pill-blue"
            action = "🔴 Pre-clear" if rp > 60 else "🟡 Monitor" if rp > 40 else "🟢 Watch"
            st.markdown(f"""<div class="card card-{'red' if rp > 60 else 'amber'}" style="padding:10px 14px;margin-bottom:5px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-size:0.76rem;font-weight:700;color:#fff">
                            {r['area_name'][:25]}
                        </div>
                        <div style="font-size:0.68rem;color:#8892a4">
                            {r['latitude']:.3f}°N · {r['longitude']:.3f}°E
                        </div>
                    </div>
                    <span class="risk-pill {pill}">{rp:.0f}%</span>
                </div>
                <div style="font-size:0.7rem;color:#8892a4;margin-top:4px">{action} before peak</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-head">💰 Delay Savings Calculator</div>', unsafe_allow_html=True)
        n_vans = st.slider("Vans in congestion zone", 50, 400, 200)
        delay_m = st.slider("Delay per van (min)", 5, 45, 18)
        cost_pm = st.slider("Cost per van/min (₹)", 100, 800, 300)
        risk_factor = min(len(high_risk) / 10, 1.0)
        savings = n_vans * delay_m * cost_pm * risk_factor * 0.72

        st.markdown(f"""
        <div class="card card-green" style="margin-top:8px">
            <div style="font-size:0.78rem;color:#27ae60;font-weight:700;margin-bottom:4px">
                Est. daily savings — pre-clearance
            </div>
            <div style="font-size:2rem;font-weight:800;color:#fff">₹{savings:,.0f}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
            <div class="card" style="padding:10px 14px">
                <div style="font-size:0.85rem;font-weight:700;color:#fff">₹{savings*22:,.0f}</div>
                <div style="font-size:0.68rem;color:#8892a4">Monthly savings</div>
            </div>
            <div class="card" style="padding:10px 14px">
                <div style="font-size:0.85rem;font-weight:700;color:#fff">₹{savings*265/100000:,.1f}L</div>
                <div style="font-size:0.68rem;color:#8892a4">Annual savings</div>
            </div>
        </div>
        <div class="insight-box" style="margin-top:10px">
            💡 Clearing <b style="color:#f39c12;">{len(high_risk)} high-risk</b> zones protects the delivery window.
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.caption("🚦 ParkSense AI · Enforcement Prototype")