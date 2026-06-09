"""
Au/Ag Regime Intelligence Dashboard
Regime-aware direction classification for Gold & Silver futures
"""

import os, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Au/Ag Regime Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODEL_LABELS = {
    "arima":  "ARIMA",
    "arimax": "ARIMAX",
    "xgb":    "XGBoost",
    "rf":     "Random Forest",
    "lstm":   "LSTM",
}
MODEL_COLORS = {
    "ARIMA":         "#4A90D9",
    "ARIMAX":        "#2C5F8A",
    "XGBoost":       "#E84545",
    "Random Forest": "#2ECC8F",
    "LSTM":          "#A855F7",
}
GOLD_C   = "#D4AF37"
SILVER_C = "#9CA3AF"
POS_C    = "#10B981"
NEG_C    = "#EF4444"
BG       = "#08080E"
CARD_BG  = "#10101A"
BORDER   = "#1E1E2E"

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
    color: #E8E8F4;
}}
[data-testid="stSidebar"] {{
    background: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: #E8E8F4 !important; }}

.main-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, {GOLD_C} 0%, #F5E14A 50%, {SILVER_C} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}}
.sub-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6B7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}}
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {GOLD_C}, transparent);
}}
.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #6B7280;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.metric-value {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    color: {GOLD_C};
    line-height: 1;
}}
.metric-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6B7280;
    margin-top: 0.3rem;
}}
.section-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {GOLD_C};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}}
.insight-box {{
    background: {CARD_BG};
    border-left: 3px solid {GOLD_C};
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #A0A0B8;
    margin: 0.8rem 0;
    line-height: 1.6;
}}
.regime-badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
}}
[data-testid="stTab"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
}}
.stDataFrame {{ background: {CARD_BG}; }}
div[data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
    color: {GOLD_C} !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_preds(model_key: str, asset_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"preds_{model_key}_{asset_key}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else "Unnamed: 0"
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Normalise returns column
    ret_map = {"au": "au_returns", "ag": "ag_returns"}
    for c in ["returns", ret_map[asset_key]]:
        if c in df.columns:
            df["ret"] = df[c]
            break
    if "prob" not in df.columns:
        df["prob"] = np.nan
    return df

@st.cache_data
def load_price_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "gold_silver_garch.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.set_index("Date").sort_index()

@st.cache_data
def load_all_preds():
    data = {}
    for mk in MODEL_LABELS:
        data[mk] = {}
        for ak in ["au", "ag"]:
            data[mk][ak] = load_preds(mk, ak)
    return data

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, regime_col=None, regime_val=None) -> dict:
    if df.empty:
        return {}
    sub = df.copy()
    if regime_col and regime_val and regime_val != "All":
        sub = sub[sub[regime_col] == regime_val]
    if len(sub) < 5:
        return {}
    signal   = np.where(sub["pred"] == 1, 1, -1)
    strat_r  = (1 + signal * sub["ret"]).cumprod().iloc[-1] - 1
    bh_r     = (1 + sub["ret"]).cumprod().iloc[-1] - 1
    return {
        "accuracy":  accuracy_score(sub["actual"], sub["pred"]),
        "f1":        f1_score(sub["actual"], sub["pred"], average="weighted", zero_division=0),
        "strat_ret": strat_r,
        "bh_ret":    bh_r,
        "alpha":     strat_r - bh_r,
        "n":         len(sub),
        "hit_rate":  (signal == np.sign(sub["ret"].values)).mean(),
    }

def build_leaderboard(all_preds, asset_key):
    rows = []
    for mk, label in MODEL_LABELS.items():
        df = all_preds[mk][asset_key]
        m  = compute_metrics(df)
        if not m:
            continue
        rows.append({
            "Model":      label,
            "Accuracy":   round(m["accuracy"], 4),
            "F1":         round(m["f1"], 4),
            "Hit Rate":   f"{m['hit_rate']:.1%}",
            "Strategy":   f"{m['strat_ret']:+.1%}",
            "B&H":        f"{m['bh_ret']:+.1%}",
            "Alpha":      f"{m['alpha']:+.1%}",
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

def regime_accuracy_table(all_preds, asset_key, regime_col):
    rows = []
    for mk, label in MODEL_LABELS.items():
        df = all_preds[mk][asset_key]
        if df.empty or regime_col not in df.columns:
            continue
        for regime in sorted(df[regime_col].unique()):
            sub = df[df[regime_col] == regime]
            if len(sub) < 5:
                continue
            acc = accuracy_score(sub["actual"], sub["pred"])
            f1  = f1_score(sub["actual"], sub["pred"], average="weighted", zero_division=0)
            rows.append({"Model": label, "Regime": regime,
                         "Accuracy": acc, "F1": f1, "n": len(sub)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────
LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="IBM Plex Mono, monospace", color="#A0A0B8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1),
    xaxis=dict(gridcolor="#1A1A2E", zerolinecolor="#2A2A3E"),
    yaxis=dict(gridcolor="#1A1A2E", zerolinecolor="#2A2A3E"),
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;
    background:linear-gradient(135deg,#D4AF37,#9CA3AF);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:0.2rem'>⚡ AU/AG INTEL</div>
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
    color:#6B7280;letter-spacing:0.15em;margin-bottom:1.5rem'>
    REGIME SIGNAL DASHBOARD</div>
    """, unsafe_allow_html=True)

    asset_choice = st.radio(
        "ASSET",
        ["🥇 Gold", "🥈 Silver"],
        horizontal=True,
    )
    asset_key   = "au" if "Gold" in asset_choice else "ag"
    asset_label = "Gold" if asset_key == "au" else "Silver"
    asset_color = GOLD_C if asset_key == "au" else SILVER_C

    st.divider()

    selected_models = st.multiselect(
        "MODELS",
        options=list(MODEL_LABELS.values()),
        default=list(MODEL_LABELS.values()),
    )

    st.divider()

    st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#6B7280;letter-spacing:0.12em;text-transform:uppercase'>Regime Filter</div>", unsafe_allow_html=True)
    regime_type = st.selectbox(
        "Type",
        ["Volatility (GARCH)", "Period (COVID)"],
        label_visibility="collapsed"
    )
    if regime_type == "Volatility (GARCH)":
        regime_col  = "vol_regime"
        regime_opts = ["All", "low_vol", "high_vol"]
    else:
        regime_col  = "period_regime"
        regime_opts = ["All", "pre_covid", "covid", "post_covid"]

    regime_filter = st.selectbox("Regime", regime_opts, label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#6B7280;line-height:1.8'>
    <b style='color:#D4AF37'>Data:</b> Feb 2010 – Mar 2026<br>
    <b style='color:#D4AF37'>Test:</b> May 2021 – Mar 2026<br>
    <b style='color:#D4AF37'>Target:</b> Next-day direction<br>
    <b style='color:#D4AF37'>Regimes:</b> GARCH vol + COVID
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>Au/Ag Regime Intelligence</div>
<div class='sub-title'>Regime-Aware Direction Classification · Gold & Silver Futures · 6 Models · Walk-Forward Backtest</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
all_preds  = load_all_preds()
price_data = load_price_data()

# Map selected labels back to keys
selected_keys = [k for k, v in MODEL_LABELS.items() if v in selected_models]

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "🏆  Leaderboard",
    "🌡️  Regime Heatmap",
    "📈  Backtest",
    "🔬  Model Deep-Dive",
    "📊  Price & Signals",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ═════════════════════════════════════════════════════════════
with t1:
    lb = build_leaderboard(all_preds, asset_key)

    # ── Metric cards
    if not lb.empty:
        best   = lb.iloc[0]
        m_all  = {mk: compute_metrics(all_preds[mk][asset_key], regime_col,
                                       regime_filter if regime_filter != "All" else None)
                  for mk in selected_keys}
        m_all  = {k: v for k, v in m_all.items() if v}

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Best Accuracy</div>
            <div class='metric-value'>{best['Accuracy']:.1%}</div>
            <div class='metric-sub'>{best['Model']}</div></div>""", unsafe_allow_html=True)
        with c2:
            best_f1 = lb.loc[lb["F1"].idxmax()]
            st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Best F1 Score</div>
            <div class='metric-value'>{best_f1['F1']:.3f}</div>
            <div class='metric-sub'>{best_f1['Model']}</div></div>""", unsafe_allow_html=True)
        with c3:
            # Best alpha
            alpha_vals = [(mk, m["alpha"]) for mk, m in m_all.items() if "alpha" in m]
            if alpha_vals:
                best_alpha_key, best_alpha_val = max(alpha_vals, key=lambda x: x[1])
                st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Best Alpha</div>
                <div class='metric-value' style='color:{"#10B981" if best_alpha_val>0 else "#EF4444"}'>{best_alpha_val:+.1%}</div>
                <div class='metric-sub'>{MODEL_LABELS[best_alpha_key]} vs Buy & Hold</div></div>""", unsafe_allow_html=True)
        with c4:
            n_models = len(lb)
            above_random = (lb["Accuracy"] > 0.50).sum()
            st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Beat Random</div>
            <div class='metric-value'>{above_random}/{n_models}</div>
            <div class='metric-sub'>models above 50% accuracy</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Full Leaderboard</div>", unsafe_allow_html=True)

    # ── Leaderboard table
    lb_filtered = lb[lb["Model"].isin(selected_models)] if selected_models else lb
    st.dataframe(
        lb_filtered.style
            .background_gradient(subset=["Accuracy","F1"], cmap="YlOrRd", vmin=0.44, vmax=0.60)
            .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "12px"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='section-header'>Accuracy Comparison</div>", unsafe_allow_html=True)

    # ── Bar chart
    fig = go.Figure()
    lb_sel = lb_filtered.sort_values("Accuracy")
    bar_colors = [MODEL_COLORS.get(m, "#888") for m in lb_sel["Model"]]
    fig.add_trace(go.Bar(
        x=lb_sel["Accuracy"], y=lb_sel["Model"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color=BORDER, width=1)),
        text=[f"{v:.3f}" for v in lb_sel["Accuracy"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#E8E8F4"),
    ))
    fig.add_vline(x=0.50, line_dash="dot", line_color="#EF4444", line_width=1.5,
                  annotation_text="Random (50%)", annotation_font_size=10,
                  annotation_font_color="#EF4444")
    fig.update_layout(**{**LAYOUT,
        "title": dict(text=f"{asset_label} — Directional Accuracy", font=dict(size=13, color=asset_color)),
        "xaxis": {**LAYOUT["xaxis"], "range": [0.40, 0.65], "tickformat": ".0%"},
        "height": 320,
    })
    st.plotly_chart(fig, use_container_width=True)

    # ── Insight
    if not lb.empty:
        best_name = lb.iloc[0]["Model"]
        best_acc  = lb.iloc[0]["Accuracy"]
        st.markdown(f"""<div class='insight-box'>
        📌 <b>{best_name}</b> leads on {asset_label} with <b>{best_acc:.1%}</b> directional accuracy.
        All accuracies sit in the 50–57% range — expected for financial time series.
        A consistent 53% hit rate is commercially meaningful when combined with a long/short strategy.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 2 — REGIME HEATMAP
# ═════════════════════════════════════════════════════════════
with t2:
    st.markdown("<div class='section-header'>Accuracy by Volatility Regime (GARCH)</div>", unsafe_allow_html=True)

    # Vol regime heatmap
    vol_lb = regime_accuracy_table(all_preds, asset_key, "vol_regime")
    if not vol_lb.empty:
        vol_sel = vol_lb[vol_lb["Model"].isin(selected_models)]
        pivot_acc = vol_sel.pivot(index="Model", columns="Regime", values="Accuracy")
        pivot_n   = vol_sel.pivot(index="Model", columns="Regime", values="n")

        annot = [[f"{pivot_acc.loc[m,r]:.3f}<br><span style='font-size:9px'>n={int(pivot_n.loc[m,r])}</span>"
                  for r in pivot_acc.columns]
                 for m in pivot_acc.index]

        fig = go.Figure(go.Heatmap(
            z=pivot_acc.values,
            x=[c.replace("_", " ").title() for c in pivot_acc.columns],
            y=pivot_acc.index.tolist(),
            colorscale=[[0,"#EF4444"],[0.5,"#F59E0B"],[1,"#10B981"]],
            zmin=0.44, zmax=0.62,
            text=pivot_acc.values,
            texttemplate="%{text:.3f}",
            textfont=dict(family="IBM Plex Mono", size=13, color="white"),
            showscale=True,
            colorbar=dict(title="Acc", tickformat=".2f",
                          tickfont=dict(family="IBM Plex Mono", size=10)),
        ))
        fig.update_layout(**{**LAYOUT,
            "title": dict(text=f"{asset_label} — Accuracy by GARCH Volatility Regime",
                         font=dict(size=13, color=asset_color)),
            "height": 340,
            "xaxis": dict(side="bottom"),
        })
        st.plotly_chart(fig, use_container_width=True)

    # F1 heatmap
    if not vol_lb.empty:
        pivot_f1 = vol_sel.pivot(index="Model", columns="Regime", values="F1")
        fig2 = go.Figure(go.Heatmap(
            z=pivot_f1.values,
            x=[c.replace("_"," ").title() for c in pivot_f1.columns],
            y=pivot_f1.index.tolist(),
            colorscale=[[0,"#4B0082"],[0.5,"#7C3AED"],[1,"#A855F7"]],
            zmin=0.35, zmax=0.60,
            texttemplate="%{text:.3f}",
            text=pivot_f1.values,
            textfont=dict(family="IBM Plex Mono", size=13, color="white"),
            showscale=True,
        ))
        fig2.update_layout(**{**LAYOUT,
            "title": dict(text=f"{asset_label} — Weighted F1 by GARCH Volatility Regime",
                         font=dict(size=13, color=asset_color)),
            "height": 300,
        })
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Accuracy by Period Regime (COVID Phases)</div>", unsafe_allow_html=True)

    period_lb = regime_accuracy_table(all_preds, asset_key, "period_regime")
    if not period_lb.empty:
        per_sel   = period_lb[period_lb["Model"].isin(selected_models)]
        pivot_per = per_sel.pivot(index="Model", columns="Regime", values="Accuracy")
        fig3 = go.Figure(go.Heatmap(
            z=pivot_per.values,
            x=[c.replace("_"," ").title() for c in pivot_per.columns],
            y=pivot_per.index.tolist(),
            colorscale=[[0,"#EF4444"],[0.5,"#F59E0B"],[1,"#10B981"]],
            zmin=0.44, zmax=0.62,
            texttemplate="%{text:.3f}",
            text=pivot_per.values,
            textfont=dict(family="IBM Plex Mono", size=13, color="white"),
            showscale=True,
            colorbar=dict(title="Acc", tickformat=".2f",
                          tickfont=dict(family="IBM Plex Mono", size=10)),
        ))
        fig3.update_layout(**{**LAYOUT,
            "title": dict(text=f"{asset_label} — Accuracy by Period Regime",
                         font=dict(size=13, color=asset_color)),
            "height": 340,
        })
        st.plotly_chart(fig3, use_container_width=True)

    # Rolling accuracy chart
    st.markdown("<div class='section-header'>Rolling 63-Day Accuracy (All Models)</div>", unsafe_allow_html=True)
    fig4 = go.Figure()
    for mk in selected_keys:
        df = all_preds[mk][asset_key]
        if df.empty:
            continue
        roll_acc = (df["actual"] == df["pred"]).rolling(63).mean()
        fig4.add_trace(go.Scatter(
            x=roll_acc.index, y=roll_acc,
            name=MODEL_LABELS[mk],
            line=dict(color=MODEL_COLORS[MODEL_LABELS[mk]], width=1.5),
            opacity=0.9,
        ))
    fig4.add_hline(y=0.50, line_dash="dot", line_color="#EF4444", line_width=1.2)
    fig4.update_layout(**{**LAYOUT,
        "title": dict(text=f"{asset_label} — Rolling Directional Accuracy (63-day window)",
                     font=dict(size=13, color=asset_color)),
        "yaxis": {**LAYOUT["yaxis"], "tickformat": ".0%", "range": [0.3, 0.75]},
        "height": 360,
    })
    st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═════════════════════════════════════════════════════════════
with t3:
    st.markdown("<div class='section-header'>Cumulative Long/Short Strategy vs Buy & Hold</div>", unsafe_allow_html=True)

    fig_bt = go.Figure()
    bh_done = False
    for mk in selected_keys:
        df = all_preds[mk][asset_key]
        if df.empty:
            continue
        sub = df.copy()
        if regime_filter != "All" and regime_col in sub.columns:
            sub = sub[sub[regime_col] == regime_filter]
        if len(sub) < 10:
            continue
        signal    = np.where(sub["pred"] == 1, 1, -1)
        cum_strat = (1 + signal * sub["ret"]).cumprod()
        cum_bh    = (1 + sub["ret"]).cumprod()
        label     = MODEL_LABELS[mk]
        final_ret = cum_strat.iloc[-1] - 1

        fig_bt.add_trace(go.Scatter(
            x=sub.index, y=cum_strat,
            name=f"{label} ({final_ret:+.1%})",
            line=dict(color=MODEL_COLORS[label], width=1.8),
        ))
        if not bh_done:
            fig_bt.add_trace(go.Scatter(
                x=sub.index, y=cum_bh,
                name=f"Buy & Hold ({cum_bh.iloc[-1]-1:+.1%})",
                line=dict(color="#888", width=2, dash="dot"),
            ))
            bh_done = True

    fig_bt.add_hline(y=1.0, line_color="#333", line_width=0.8)
    regime_note = f" [{regime_filter.replace('_',' ').title()}]" if regime_filter != "All" else ""
    fig_bt.update_layout(**{**LAYOUT,
        "title": dict(text=f"{asset_label}{regime_note} — Cumulative Strategy Returns",
                     font=dict(size=13, color=asset_color)),
        "yaxis": {**LAYOUT["yaxis"], "title": "Growth of ₹1"},
        "height": 400,
    })
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Regime-split backtest
    col1, col2 = st.columns(2)
    for col, regime_val in zip([col1, col2], ["low_vol", "high_vol"]):
        with col:
            label_r = regime_val.replace("_"," ").title()
            st.markdown(f"<div class='section-header'>{label_r} Regime</div>", unsafe_allow_html=True)
            fig_r = go.Figure()
            bh_done_r = False
            for mk in selected_keys:
                df = all_preds[mk][asset_key]
                if df.empty or "vol_regime" not in df.columns:
                    continue
                sub = df[df["vol_regime"] == regime_val]
                if len(sub) < 5:
                    continue
                signal = np.where(sub["pred"] == 1, 1, -1)
                cum_s  = (1 + signal * sub["ret"]).cumprod()
                cum_b  = (1 + sub["ret"]).cumprod()
                lbl    = MODEL_LABELS[mk]
                fig_r.add_trace(go.Scatter(
                    x=sub.index, y=cum_s,
                    name=f"{lbl} ({cum_s.iloc[-1]-1:+.1%})",
                    line=dict(color=MODEL_COLORS[lbl], width=1.5),
                ))
                if not bh_done_r:
                    fig_r.add_trace(go.Scatter(
                        x=sub.index, y=cum_b,
                        name=f"B&H ({cum_b.iloc[-1]-1:+.1%})",
                        line=dict(color="#888", width=1.5, dash="dot"),
                    ))
                    bh_done_r = True
            fig_r.add_hline(y=1.0, line_color="#333", line_width=0.8)
            fig_r.update_layout(**{**LAYOUT,
                "height": 300,
                "showlegend": True,
                "legend": dict(font=dict(size=9)),
                "margin": dict(l=30, r=10, t=30, b=30),
            })
            st.plotly_chart(fig_r, use_container_width=True)

    # ── Monthly returns heatmap for best model
    st.markdown("<div class='section-header'>Monthly Strategy Returns — Best Model</div>", unsafe_allow_html=True)
    best_mk = None
    best_acc_val = 0
    for mk in selected_keys:
        df = all_preds[mk][asset_key]
        if df.empty:
            continue
        m = compute_metrics(df)
        if m and m["accuracy"] > best_acc_val:
            best_acc_val = m["accuracy"]
            best_mk = mk

    if best_mk:
        df_best = all_preds[best_mk][asset_key].copy()
        signal  = np.where(df_best["pred"] == 1, 1, -1)
        df_best["strat_ret"] = signal * df_best["ret"]
        df_best["year"]  = df_best.index.year
        df_best["month"] = df_best.index.month
        monthly = df_best.groupby(["year","month"])["strat_ret"].sum().unstack()
        monthly.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(monthly.columns)]

        fig_m = go.Figure(go.Heatmap(
            z=monthly.values,
            x=monthly.columns.tolist(),
            y=monthly.index.tolist(),
            colorscale=[[0,"#EF4444"],[0.5,"#1E1E2E"],[1,"#10B981"]],
            zmid=0,
            texttemplate="%{text:.1%}",
            text=monthly.values,
            textfont=dict(family="IBM Plex Mono", size=9, color="white"),
            showscale=True,
            colorbar=dict(tickformat=".0%"),
        ))
        fig_m.update_layout(**{**LAYOUT,
            "title": dict(text=f"{asset_label} Monthly Returns — {MODEL_LABELS[best_mk]}",
                         font=dict(size=12, color=asset_color)),
            "height": 380,
        })
        st.plotly_chart(fig_m, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 4 — MODEL DEEP-DIVE
# ═════════════════════════════════════════════════════════════
with t4:
    sel_model_label = st.selectbox(
        "SELECT MODEL",
        options=[MODEL_LABELS[k] for k in selected_keys],
        key="deep_dive_select"
    )
    sel_mk = [k for k, v in MODEL_LABELS.items() if v == sel_model_label][0]
    df_dd  = all_preds[sel_mk][asset_key]

    if not df_dd.empty:
        m = compute_metrics(df_dd)

        # ── Top metrics
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("Accuracy",  f"{m['accuracy']:.1%}", ""),
            ("F1 Score",  f"{m['f1']:.3f}", "weighted"),
            ("Hit Rate",  f"{m['hit_rate']:.1%}", "correct direction calls"),
            ("Alpha",     f"{m['alpha']:+.1%}", "vs buy & hold"),
        ]
        for col, (lbl, val, sub) in zip([c1,c2,c3,c4], stats):
            color = GOLD_C if "Alpha" not in lbl else ("#10B981" if m['alpha']>0 else "#EF4444")
            with col:
                st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-sub'>{sub}</div></div>""", unsafe_allow_html=True)

        st.markdown("")
        col_left, col_right = st.columns(2)

        # ── Confusion matrix
        with col_left:
            st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
            cm = confusion_matrix(df_dd["actual"], df_dd["pred"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Pred DOWN","Pred UP"],
                y=["Actual DOWN","Actual UP"],
                colorscale=[[0,CARD_BG],[1,MODEL_COLORS[sel_model_label]]],
                texttemplate="%{text}",
                text=cm,
                textfont=dict(family="IBM Plex Mono", size=16, color="white"),
                showscale=False,
            ))
            fig_cm.update_layout(**{**LAYOUT, "height": 280,
                "margin": dict(l=80, r=20, t=30, b=60)})
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Prob distribution (if available)
        with col_right:
            if not df_dd["prob"].isna().all():
                st.markdown("<div class='section-header'>Prediction Probability Distribution</div>", unsafe_allow_html=True)
                fig_prob = go.Figure()
                for cls, color, name in [(0, NEG_C, "DOWN"), (1, POS_C, "UP")]:
                    sub_prob = df_dd[df_dd["actual"] == cls]["prob"]
                    fig_prob.add_trace(go.Histogram(
                        x=sub_prob, name=f"Actual {name}",
                        marker_color=color, opacity=0.7,
                        nbinsx=40, histnorm="probability density",
                    ))
                fig_prob.add_vline(x=0.5, line_dash="dot", line_color="#888")
                fig_prob.update_layout(**{**LAYOUT, "height": 280,
                    "barmode": "overlay",
                    "xaxis": {**LAYOUT["xaxis"], "title": "P(UP)", "range": [0,1]},
                    "margin": dict(l=30, r=20, t=30, b=40)})
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.markdown("<div class='section-header'>Forecast Distribution</div>", unsafe_allow_html=True)
                if "forecast" in df_dd.columns:
                    fig_fc = go.Figure(go.Histogram(
                        x=df_dd["forecast"], nbinsx=80,
                        marker_color=MODEL_COLORS[sel_model_label],
                        opacity=0.8,
                    ))
                    fig_fc.add_vline(x=0, line_dash="dot", line_color="#EF4444",
                                     annotation_text="Decision boundary",
                                     annotation_font_size=10)
                    fig_fc.update_layout(**{**LAYOUT, "height": 280,
                        "xaxis": {**LAYOUT["xaxis"], "title": "Forecast Value"},
                        "margin": dict(l=30, r=20, t=30, b=40)})
                    st.plotly_chart(fig_fc, use_container_width=True)

        # ── Regime breakdown
        st.markdown("<div class='section-header'>Accuracy by Regime</div>", unsafe_allow_html=True)
        col_v, col_p = st.columns(2)
        for col, reg_col, reg_title in [
            (col_v, "vol_regime",    "Volatility Regime"),
            (col_p, "period_regime", "Period Regime"),
        ]:
            with col:
                if reg_col in df_dd.columns:
                    reg_data = []
                    for reg in sorted(df_dd[reg_col].unique()):
                        sub = df_dd[df_dd[reg_col] == reg]
                        if len(sub) < 5:
                            continue
                        reg_data.append({
                            "Regime": reg.replace("_"," ").title(),
                            "Accuracy": accuracy_score(sub["actual"], sub["pred"]),
                            "n": len(sub)
                        })
                    if reg_data:
                        reg_df = pd.DataFrame(reg_data).sort_values("Accuracy")
                        bar_cols = [POS_C if v > 0.5 else NEG_C for v in reg_df["Accuracy"]]
                        fig_r = go.Figure(go.Bar(
                            x=reg_df["Accuracy"], y=reg_df["Regime"],
                            orientation="h",
                            marker_color=bar_cols,
                            text=[f"{v:.3f} (n={n})" for v,n in zip(reg_df["Accuracy"], reg_df["n"])],
                            textposition="outside",
                            textfont=dict(family="IBM Plex Mono", size=10),
                        ))
                        fig_r.add_vline(x=0.50, line_dash="dot", line_color="#EF4444")
                        fig_r.update_layout(**{**LAYOUT, "height": 240,
                            "title": dict(text=reg_title, font=dict(size=11, color="#A0A0B8")),
                            "xaxis": {**LAYOUT["xaxis"], "range": [0.4, 0.68], "tickformat": ".0%"},
                            "margin": dict(l=100, r=60, t=40, b=30)})
                        st.plotly_chart(fig_r, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 5 — PRICE & SIGNALS
# ═════════════════════════════════════════════════════════════
with t5:
    signal_model = st.selectbox(
        "Signal source model",
        options=[MODEL_LABELS[k] for k in selected_keys],
        key="signal_model_select",
    )
    sig_mk = [k for k, v in MODEL_LABELS.items() if v == signal_model][0]
    df_sig = all_preds[sig_mk][asset_key]

    price_col = "gold" if asset_key == "au" else "silver"
    vol_col   = "garch_vol_au" if asset_key == "au" else "garch_vol_ag"

    if not price_data.empty and not df_sig.empty:
        # ── Price chart with signal overlay
        st.markdown("<div class='section-header'>Price Chart with Buy/Sell Signals</div>", unsafe_allow_html=True)

        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )

        # Full price history
        fig_price.add_trace(go.Scatter(
            x=price_data.index, y=price_data[price_col],
            name=f"{asset_label} Price",
            line=dict(color=asset_color, width=1.5),
            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(asset_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.05)",
        ), row=1, col=1)

        # Buy signals (pred=1) in test period
        buys  = df_sig[df_sig["pred"] == 1]
        sells = df_sig[df_sig["pred"] == 0]
        buy_prices  = price_data.loc[price_data.index.isin(buys.index), price_col]
        sell_prices = price_data.loc[price_data.index.isin(sells.index), price_col]

        if not buy_prices.empty:
            fig_price.add_trace(go.Scatter(
                x=buy_prices.index, y=buy_prices,
                name="UP Signal", mode="markers",
                marker=dict(color=POS_C, size=3, opacity=0.5),
            ), row=1, col=1)
        if not sell_prices.empty:
            fig_price.add_trace(go.Scatter(
                x=sell_prices.index, y=sell_prices,
                name="DOWN Signal", mode="markers",
                marker=dict(color=NEG_C, size=3, opacity=0.5),
            ), row=1, col=1)

        # GARCH vol
        if vol_col in price_data.columns:
            fig_price.add_trace(go.Scatter(
                x=price_data.index, y=price_data[vol_col],
                name="GARCH Cond. Vol",
                line=dict(color="#7C3AED", width=1.2),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
            ), row=2, col=1)

        # Shade test period
        test_start = df_sig.index[0]
        for row in [1, 2]:
            fig_price.add_vrect(
                x0=test_start, x1=price_data.index[-1],
                fillcolor="rgba(212,175,55,0.04)",
                line_width=0.5, line_color=GOLD_C,
                annotation_text="← Test Period",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=GOLD_C,
                row=row, col=1,
            )

        fig_price.update_layout(**{**LAYOUT,
            "title": dict(text=f"{asset_label} — Price History + {signal_model} Signals + GARCH Volatility",
                         font=dict(size=13, color=asset_color)),
            "height": 550,
            "showlegend": True,
            "legend": dict(x=0.01, y=0.99, font=dict(size=9)),
        })
        fig_price.update_yaxes(title_text="Price", row=1, col=1)
        fig_price.update_yaxes(title_text="GARCH Vol", row=2, col=1)
        st.plotly_chart(fig_price, use_container_width=True)

        # ── Regime overlay
        st.markdown("<div class='section-header'>GARCH Volatility Regimes on Price</div>", unsafe_allow_html=True)
        vol_reg_col = "vol_regime_au" if asset_key == "au" else "vol_regime_ag"

        if vol_reg_col in price_data.columns:
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=price_data.index, y=price_data[price_col],
                name=f"{asset_label} Price",
                line=dict(color=asset_color, width=1.5),
            ))
            # Shade low_vol / high_vol bands
            reg_series = price_data[vol_reg_col]
            prev_reg = None
            band_start = None
            for date, reg in reg_series.items():
                if reg != prev_reg:
                    if band_start is not None and prev_reg is not None:
                        fc = "rgba(16,185,129,0.07)" if prev_reg == "low_vol" else "rgba(239,68,68,0.07)"
                        fig_reg.add_vrect(x0=band_start, x1=date, fillcolor=fc, line_width=0)
                    band_start = date
                    prev_reg = reg

            fig_reg.update_layout(**{**LAYOUT,
                "title": dict(text=f"{asset_label} — GARCH Regime Overlay (Green=Low Vol, Red=High Vol)",
                             font=dict(size=12, color=asset_color)),
                "height": 340,
                "yaxis": {**LAYOUT["yaxis"], "title": "Price"},
            })
            st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    📌 <b>How to read signals:</b> Green dots = model predicts next-day UP (go long).
    Red dots = model predicts next-day DOWN (go short or exit).
    The shaded area marks the test period (May 2021 → Mar 2026) — out-of-sample predictions only.
    GARCH conditional volatility (bottom panel) defines the Low/High vol regimes used in all analysis.
    </div>""", unsafe_allow_html=True)
