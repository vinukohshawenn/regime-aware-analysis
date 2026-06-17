"""
Au/Ag Regime Intelligence Dashboard
Regime-aware direction classification · Gold & Silver Futures · 5 models
"""

import os, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Au/Ag Regime Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = BASE_DIR

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "arima":  "ARIMA",
    "arimax": "ARIMAX",
    "xgb":    "XGBoost",
    "rf":     "Random Forest",
    "lstm":   "LSTM",
}
MODEL_COLORS = {
    "ARIMA":         "#4A90D9",
    "ARIMAX":        "#2C5282",
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

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="monospace", color="#A0A0B8", size=11),
    margin=dict(l=48, r=20, t=44, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1),
    xaxis=dict(gridcolor="#1A1A2E", zerolinecolor="#2A2A3E"),
    yaxis=dict(gridcolor="#1A1A2E", zerolinecolor="#2A2A3E"),
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: {BG}; color: #E8E8F4;
}}
[data-testid="stSidebar"] {{
    background: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: #E8E8F4 !important; }}
.main-title {{
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 2.5rem; letter-spacing: -0.03em;
    background: linear-gradient(135deg, {GOLD_C} 0%, #F5E14A 50%, {SILVER_C} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 0.15rem;
}}
.sub-title {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: #6B7280; letter-spacing: 0.13em; text-transform: uppercase;
    margin-bottom: 1.5rem;
}}
.kpi {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
}}
.kpi::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, {GOLD_C}, transparent);
}}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
    color: #6B7280; letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.35rem;
}}
.kpi-value {{
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 1.85rem; line-height: 1; color: {GOLD_C};
}}
.kpi-sub {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.67rem;
    color: #6B7280; margin-top: 0.25rem;
}}
.sec {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: {GOLD_C};
    border-bottom: 1px solid {BORDER}; padding-bottom: 0.45rem;
    margin: 1.4rem 0 0.9rem;
}}
.insight {{
    background: {CARD_BG}; border-left: 3px solid {GOLD_C};
    border-radius: 0 6px 6px 0; padding: 0.75rem 1rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
    color: #A0A0B8; margin: 0.7rem 0; line-height: 1.65;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_pred(model_key: str, asset_key: str):

    candidates = [
        os.path.join(DATA_DIR, f"preds_{model_key}_{asset_key}.csv"),
        os.path.join(os.path.dirname(DATA_DIR), f"preds_{model_key}_{asset_key}.csv"),
    ]

    path = next((p for p in candidates if os.path.exists(p)), None)

    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path)

    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    for c in ["returns", "au_returns", "ag_returns"]:
        if c in df.columns:
            df["ret"] = df[c]
            break

    if "prob" not in df.columns:
        df["prob"] = np.nan

    return df

@st.cache_data
def load_price():

    candidates = [
        "gold_silver_garch.csv",
        "gold_silver_garch(2).csv",
        "gold_silver_garch(3).csv",
    ]

    path = None

    for f in candidates:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            path = p
            break

    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    if "ID_Column" in df.columns:
        df.drop(columns=["ID_Column"], inplace=True)

    return df


@st.cache_data
def load_all() -> dict:
    return {mk: {"au": load_pred(mk, "au"),
                 "ag": load_pred(mk, "ag")}
            for mk in MODEL_LABELS}

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def metrics(df: pd.DataFrame, rcol: str = None,
            rval: str = None) -> dict:
    if df.empty or "ret" not in df.columns:
        return {}
    sub = df.copy()
    if rcol and rval and rval != "All" and rcol in sub.columns:
        sub = sub[sub[rcol] == rval]
    if len(sub) < 10:
        return {}
    sig     = np.where(sub["pred"] == 1, 1, -1)
    strat   = (1 + sig * sub["ret"]).cumprod().iloc[-1] - 1
    bh      = (1 + sub["ret"]).cumprod().iloc[-1] - 1
    hit     = (sig == np.sign(sub["ret"].values)).mean()
    return dict(
        accuracy = accuracy_score(sub["actual"], sub["pred"]),
        f1       = f1_score(sub["actual"], sub["pred"],
                            average="weighted", zero_division=0),
        strat    = strat,
        bh       = bh,
        alpha    = strat - bh,
        hit      = hit,
        n        = len(sub),
    )


def leaderboard(preds, ak, sel_keys):

    rows = []

    for mk in sel_keys:

        df = preds[mk][ak]

        if df.empty:
            continue

        m = metrics(df)

        if not m:
            continue

        rows.append({
            "Model": MODEL_LABELS[mk],
            "Accuracy": round(m["accuracy"],4),
            "F1": round(m["f1"],4),
            "Hit Rate": f"{m['hit']:.1%}",
            "Strategy": f"{m['strat']:+.1%}",
            "B&H": f"{m['bh']:+.1%}",
            "Alpha": f"{m['alpha']:+.1%}",
        })

    if len(rows)==0:
        return pd.DataFrame(columns=[
            "Model","Accuracy","F1",
            "Hit Rate","Strategy","B&H","Alpha"
        ])

    return (
        pd.DataFrame(rows)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )

def regime_table(preds: dict, ak: str, sel_keys: list,
                 rcol: str) -> pd.DataFrame:
    rows = []
    for mk in sel_keys:
        df = preds[mk][ak]
        if df.empty or rcol not in df.columns:
            continue
        for rv in sorted(df[rcol].unique()):
            sub = df[df[rcol] == rv]
            if len(sub) < 10:
                continue
            acc = accuracy_score(sub["actual"], sub["pred"])
            f1  = f1_score(sub["actual"], sub["pred"],
                           average="weighted", zero_division=0)
            rows.append({"Model": MODEL_LABELS[mk],
                         "Regime": rv, "Accuracy": acc,
                         "F1": f1, "n": len(sub)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-weight:800;"
        f"font-size:1.25rem;background:linear-gradient(135deg,{GOLD_C},{SILVER_C});"
        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        f"margin-bottom:0.15rem'>⚡ AU/AG INTEL</div>"
        f"<div style='font-family:monospace;font-size:0.58rem;"
        f"color:#6B7280;letter-spacing:0.14em;margin-bottom:1.3rem'>"
        f"REGIME SIGNAL DASHBOARD</div>",
        unsafe_allow_html=True,
    )

    asset_choice = st.radio("ASSET", ["🥇 Gold", "🥈 Silver"], horizontal=True)
    ak           = "au" if "Gold" in asset_choice else "ag"
    asset_label  = "Gold"   if ak == "au" else "Silver"
    asset_color  = GOLD_C   if ak == "au" else SILVER_C

    st.divider()

    sel_labels = st.multiselect(
        "MODELS",
        options=list(MODEL_LABELS.values()),
        default=list(MODEL_LABELS.values()),
    )
    sel_keys = [k for k, v in MODEL_LABELS.items() if v in sel_labels]

    st.divider()

    regime_type = st.selectbox(
        "REGIME TYPE",
        ["Volatility (GARCH)", "Period (COVID)"],
    )
    rcol    = "vol_regime"    if "Volatility" in regime_type else "period_regime"
    r_opts  = ["All", "low_vol", "high_vol"] \
              if "Volatility" in regime_type \
              else ["All", "pre_covid", "covid", "post_covid"]
    rfilter = st.selectbox("REGIME FILTER", r_opts)

    st.divider()
    st.markdown(
        "<div style='font-family:monospace;font-size:0.62rem;"
        "color:#6B7280;line-height:1.9'>"
        f"<b style='color:{GOLD_C}'>Data</b>: Mar 2010 – Jun 2026<br>"
        f"<b style='color:{GOLD_C}'>Test</b>: Jul 2021 – Jun 2026<br>"
        f"<b style='color:{GOLD_C}'>Target</b>: Next-day direction<br>"
        f"<b style='color:{GOLD_C}'>Regimes</b>: GARCH vol + COVID</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-title'>Au/Ag Regime Intelligence</div>"
    "<div class='sub-title'>Regime-Aware Direction Classification · "
    "Gold &amp; Silver Futures · 5 Models · Walk-Forward Backtest</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
all_preds  = load_all()
price_data = load_price()

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

# ════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ════════════════════════════════════════════════════════════
with t1:
    if not sel_keys:
        st.warning("Select at least one model from the sidebar.")
        st.stop()

    lb = leaderboard(all_preds, ak, sel_keys)

    if not lb.empty:
        best      = lb.iloc[0]
        best_f1r  = lb.loc[lb["F1"].idxmax()]

        # alpha with regime filter applied
        alpha_map = {}
        for mk in sel_keys:
            m = metrics(all_preds[mk][ak], rcol,
                        rfilter if rfilter != "All" else None)
            if m:
                alpha_map[mk] = m["alpha"]

        c1, c2, c3, c4 = st.columns(4)
        kpi_data = [
            (c1, "BEST ACCURACY",  f"{best['Accuracy']:.1%}",   best["Model"]),
            (c2, "BEST F1 SCORE",  f"{best_f1r['F1']:.3f}",     best_f1r["Model"]),
        ]
        for col, lbl, val, sub in kpi_data:
            with col:
                st.markdown(
                    f"<div class='kpi'><div class='kpi-label'>{lbl}</div>"
                    f"<div class='kpi-value'>{val}</div>"
                    f"<div class='kpi-sub'>{sub}</div></div>",
                    unsafe_allow_html=True,
                )

        if alpha_map:
            best_a_key = max(alpha_map, key=alpha_map.get)
            best_a_val = alpha_map[best_a_key]
            a_color = POS_C if best_a_val > 0 else NEG_C
            with c3:
                st.markdown(
                    f"<div class='kpi'><div class='kpi-label'>BEST ALPHA</div>"
                    f"<div class='kpi-value' style='color:{a_color}'>"
                    f"{best_a_val:+.1%}</div>"
                    f"<div class='kpi-sub'>{MODEL_LABELS[best_a_key]} vs Buy & Hold"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        above50 = (lb["Accuracy"] > 0.50).sum()
        with c4:
            st.markdown(
                f"<div class='kpi'><div class='kpi-label'>BEAT RANDOM</div>"
                f"<div class='kpi-value'>{above50}/{len(lb)}</div>"
                f"<div class='kpi-sub'>models above 50% accuracy</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='sec'>Full Leaderboard</div>", unsafe_allow_html=True)

    # colour accuracy/F1 cells without matplotlib
    def _hl(val):
        try:
            v = float(val)
            if v >= 0.55:  return "background-color:#10B98122;color:#10B981;font-weight:600"
            elif v >= 0.52: return "background-color:#F59E0B18;color:#F59E0B;font-weight:600"
            return "color:#9CA3AF"
        except Exception:
            return ""

if lb.empty:
    st.error("No prediction files could be loaded.")
    st.stop()

st.dataframe(
    lb.style
      .map(_hl, subset=["Accuracy", "F1"])
      .set_properties(**{
          "font-family": "monospace",
          "font-size": "12px"
      }),
    use_container_width=True,
    hide_index=True,
)

    st.markdown("<div class='sec'>Accuracy Comparison</div>", unsafe_allow_html=True)
    lb_s = lb.sort_values("Accuracy")
    fig = go.Figure(go.Bar(
        x=lb_s["Accuracy"],
        y=lb_s["Model"],
        orientation="h",
        marker=dict(color=[MODEL_COLORS.get(m, "#888") for m in lb_s["Model"]],
                    line=dict(color=BORDER, width=1)),
        text=[f"{v:.3f}" for v in lb_s["Accuracy"]],
        textposition="outside",
        textfont=dict(family="monospace", size=11, color="#E8E8F4"),
    ))
    fig.add_vline(x=0.50, line_dash="dot", line_color=NEG_C,
                  line_width=1.5,
                  annotation_text="Random (50%)",
                  annotation_font=dict(size=10, color=NEG_C))
    fig.update_layout(**{**PLOT_LAYOUT,
        "title": dict(text=f"{asset_label} — Directional Accuracy",
                      font=dict(size=13, color=asset_color)),
        "xaxis": {**PLOT_LAYOUT["xaxis"],
                  "range": [0.40, 0.68], "tickformat": ".0%"},
        "height": 300,
    })
    st.plotly_chart(fig, use_container_width=True)

    best_name = lb.iloc[0]["Model"] if not lb.empty else "—"
    best_acc  = lb.iloc[0]["Accuracy"] if not lb.empty else 0
    st.markdown(
        f"<div class='insight'>📌 <b>{best_name}</b> leads on {asset_label} "
        f"with <b>{best_acc:.1%}</b> directional accuracy. "
        f"All models cluster 50–56% — expected for daily financial returns. "
        f"A sustained 53% hit rate is commercially meaningful when "
        f"combined with a disciplined long/short strategy.</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════
# TAB 2 — REGIME HEATMAP
# ════════════════════════════════════════════════════════════
with t2:
    def _heatmap(lb_df: pd.DataFrame, val_col: str,
                 title: str, cscale, vmin: float, vmax: float):
        if lb_df.empty:
            st.info("No data for selected models / regime.")
            return
        sub = lb_df[lb_df["Model"].isin(sel_labels)]
        if sub.empty:
            st.info("No data for selected models.")
            return
        try:
            pivot = sub.pivot(index="Model", columns="Regime",
                              values=val_col)
        except Exception:
            st.info("Not enough data to build heatmap.")
            return
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[c.replace("_", " ").title() for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=cscale,
            zmin=vmin, zmax=vmax,
            text=pivot.values,
            texttemplate="%{text:.3f}",
            textfont=dict(family="monospace", size=13, color="white"),
            showscale=True,
            colorbar=dict(tickformat=".2f",
                          tickfont=dict(family="monospace", size=10)),
        ))
        fig.update_layout(**{**PLOT_LAYOUT,
            "title": dict(text=title,
                          font=dict(size=12, color=asset_color)),
            "height": 320,
            "xaxis": dict(side="bottom"),
        })
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec'>Accuracy by Volatility Regime (GARCH)</div>",
                unsafe_allow_html=True)
    vol_lb = regime_table(all_preds, ak, sel_keys, "vol_regime")
    _heatmap(vol_lb, "Accuracy",
             f"{asset_label} — Accuracy by GARCH Volatility Regime",
             [[0, NEG_C], [0.5, "#F59E0B"], [1, POS_C]], 0.44, 0.65)
    _heatmap(vol_lb, "F1",
             f"{asset_label} — Weighted F1 by GARCH Volatility Regime",
             [[0, "#4B0082"], [0.5, "#7C3AED"], [1, "#A855F7"]], 0.35, 0.65)

    st.markdown("<div class='sec'>Accuracy by Period Regime</div>",
                unsafe_allow_html=True)
    per_lb = regime_table(all_preds, ak, sel_keys, "period_regime")
    _heatmap(per_lb, "Accuracy",
             f"{asset_label} — Accuracy by Period Regime",
             [[0, NEG_C], [0.5, "#F59E0B"], [1, POS_C]], 0.44, 0.65)

    st.markdown("<div class='sec'>Rolling 63-Day Accuracy — All Models</div>",
                unsafe_allow_html=True)
    fig4 = go.Figure()
    for mk in sel_keys:
        df = all_preds[mk][ak]
        if df.empty:
            continue
        ra = (df["actual"] == df["pred"]).rolling(63).mean()
        fig4.add_trace(go.Scatter(
            x=ra.index, y=ra,
            name=MODEL_LABELS[mk],
            line=dict(color=MODEL_COLORS[MODEL_LABELS[mk]], width=1.5),
        ))
    fig4.add_hline(y=0.50, line_dash="dot",
                   line_color=NEG_C, line_width=1.2)
    fig4.update_layout(**{**PLOT_LAYOUT,
        "title": dict(text=f"{asset_label} — Rolling Directional Accuracy (63-day)",
                      font=dict(size=12, color=asset_color)),
        "yaxis": {**PLOT_LAYOUT["yaxis"],
                  "tickformat": ".0%", "range": [0.28, 0.78]},
        "height": 360,
    })
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ════════════════════════════════════════════════════════════
with t3:
    st.markdown("<div class='sec'>Cumulative L/S Strategy vs Buy & Hold</div>",
                unsafe_allow_html=True)

    fig_bt = go.Figure()
    bh_done = False
    for mk in sel_keys:
        df = all_preds[mk][ak]
        if df.empty or "ret" not in df.columns:
            continue
        sub = df.copy()
        if rfilter != "All" and rcol in sub.columns:
            sub = sub[sub[rcol] == rfilter]
        if len(sub) < 10:
            continue
        sig       = np.where(sub["pred"] == 1, 1, -1)
        cum_strat = (1 + sig * sub["ret"]).cumprod()
        cum_bh    = (1 + sub["ret"]).cumprod()
        lbl       = MODEL_LABELS[mk]
        fig_bt.add_trace(go.Scatter(
            x=sub.index, y=cum_strat,
            name=f"{lbl} ({cum_strat.iloc[-1]-1:+.1%})",
            line=dict(color=MODEL_COLORS[lbl], width=1.8),
        ))
        if not bh_done:
            fig_bt.add_trace(go.Scatter(
                x=sub.index, y=cum_bh,
                name=f"Buy & Hold ({cum_bh.iloc[-1]-1:+.1%})",
                line=dict(color="#888888", width=2, dash="dot"),
            ))
            bh_done = True

    fig_bt.add_hline(y=1, line_color="#333344", line_width=0.8)
    rnote = f" [{rfilter.replace('_',' ').title()}]" if rfilter != "All" else ""
    fig_bt.update_layout(**{**PLOT_LAYOUT,
        "title": dict(
            text=f"{asset_label}{rnote} — Cumulative Strategy Returns",
            font=dict(size=13, color=asset_color)),
        "yaxis": {**PLOT_LAYOUT["yaxis"], "title": "Growth of ₹1"},
        "height": 420,
    })
    st.plotly_chart(fig_bt, use_container_width=True)

    # Regime-split backtests
    col1, col2 = st.columns(2)
    for col, rv in zip([col1, col2], ["low_vol", "high_vol"]):
        with col:
            rl = rv.replace("_", " ").title()
            st.markdown(f"<div class='sec'>{rl} Regime</div>",
                        unsafe_allow_html=True)
            fig_r = go.Figure()
            bh_r  = False
            for mk in sel_keys:
                df = all_preds[mk][ak]
                if df.empty or "vol_regime" not in df.columns:
                    continue
                sub = df[df["vol_regime"] == rv]
                if len(sub) < 5:
                    continue
                sig  = np.where(sub["pred"] == 1, 1, -1)
                cs   = (1 + sig * sub["ret"]).cumprod()
                cb   = (1 + sub["ret"]).cumprod()
                lbl  = MODEL_LABELS[mk]
                fig_r.add_trace(go.Scatter(
                    x=sub.index, y=cs,
                    name=f"{lbl} ({cs.iloc[-1]-1:+.1%})",
                    line=dict(color=MODEL_COLORS[lbl], width=1.4),
                ))
                if not bh_r:
                    fig_r.add_trace(go.Scatter(
                        x=sub.index, y=cb,
                        name=f"B&H ({cb.iloc[-1]-1:+.1%})",
                        line=dict(color="#888888", width=1.5, dash="dot"),
                    ))
                    bh_r = True
            fig_r.add_hline(y=1, line_color="#333344", line_width=0.7)
            fig_r.update_layout(**{**PLOT_LAYOUT,
                "height": 300,
                "legend": dict(font=dict(size=9)),
                "margin": dict(l=30, r=10, t=30, b=30),
            })
            st.plotly_chart(fig_r, use_container_width=True)

    # Monthly returns heatmap — best accuracy model
    best_mk = max(
        (mk for mk in sel_keys if not all_preds[mk][ak].empty),
        key=lambda mk: metrics(all_preds[mk][ak]).get("accuracy", 0),
        default=None,
    )
    if best_mk:
        st.markdown(
            f"<div class='sec'>Monthly P&L — {MODEL_LABELS[best_mk]}</div>",
            unsafe_allow_html=True,
        )
        df_b = all_preds[best_mk][ak].copy()
        sig  = np.where(df_b["pred"] == 1, 1, -1)
        df_b["sr"]    = sig * df_b["ret"]
        df_b["year"]  = df_b.index.year
        df_b["month"] = df_b.index.month
        monthly = df_b.groupby(["year", "month"])["sr"].sum().unstack()
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly.columns = [month_map.get(c, str(c)) for c in monthly.columns]
        fig_m = go.Figure(go.Heatmap(
            z=monthly.values,
            x=monthly.columns.tolist(),
            y=monthly.index.tolist(),
            colorscale=[[0, NEG_C], [0.5, CARD_BG], [1, POS_C]],
            zmid=0,
            text=monthly.values,
            texttemplate="%{text:.1%}",
            textfont=dict(family="monospace", size=9, color="white"),
            showscale=True,
            colorbar=dict(tickformat=".0%"),
        ))
        fig_m.update_layout(**{**PLOT_LAYOUT,
            "title": dict(
                text=f"{asset_label} Monthly P&L — {MODEL_LABELS[best_mk]}",
                font=dict(size=12, color=asset_color)),
            "height": 380,
        })
        st.plotly_chart(fig_m, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — MODEL DEEP-DIVE
# ════════════════════════════════════════════════════════════
with t4:
    if not sel_keys:
        st.info("Select at least one model from the sidebar.")
        st.stop()

    sel_label = st.selectbox(
        "SELECT MODEL",
        options=[MODEL_LABELS[k] for k in sel_keys],
        key="deep_dive_sel",
    )
    sel_mk = next(k for k, v in MODEL_LABELS.items() if v == sel_label)
    df_dd  = all_preds[sel_mk][ak]

    if df_dd.empty:
        st.warning("No data for this model.")
    else:
        m = metrics(df_dd)
        if m:
            c1, c2, c3, c4 = st.columns(4)
            for col, lbl, val, sub, colour in [
                (c1, "ACCURACY",  f"{m['accuracy']:.1%}", "", GOLD_C),
                (c2, "F1 SCORE",  f"{m['f1']:.3f}",     "weighted", GOLD_C),
                (c3, "HIT RATE",  f"{m['hit']:.1%}",    "correct direction calls", GOLD_C),
                (c4, "ALPHA",     f"{m['alpha']:+.1%}",  "vs buy & hold",
                 POS_C if m["alpha"] > 0 else NEG_C),
            ]:
                with col:
                    st.markdown(
                        f"<div class='kpi'><div class='kpi-label'>{lbl}</div>"
                        f"<div class='kpi-value' style='color:{colour}'>{val}</div>"
                        f"<div class='kpi-sub'>{sub}</div></div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("")

        col_l, col_r = st.columns(2)

        # Confusion matrix
        with col_l:
            st.markdown("<div class='sec'>Confusion Matrix</div>",
                        unsafe_allow_html=True)
            cm = confusion_matrix(df_dd["actual"], df_dd["pred"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Pred DOWN", "Pred UP"],
                y=["Actual DOWN", "Actual UP"],
                colorscale=[[0, CARD_BG],
                            [1, MODEL_COLORS.get(sel_label, GOLD_C)]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(family="monospace", size=18, color="white"),
                showscale=False,
            ))
            fig_cm.update_layout(**{**PLOT_LAYOUT,
                "height": 280,
                "margin": dict(l=80, r=20, t=30, b=60),
            })
            st.plotly_chart(fig_cm, use_container_width=True)

        # Prob / forecast distribution
        with col_r:
            if not df_dd["prob"].isna().all():
                st.markdown("<div class='sec'>Prediction Probability Distribution</div>",
                            unsafe_allow_html=True)
                fig_p = go.Figure()
                for cls, colour, name in [
                    (0, NEG_C, "Actual DOWN"),
                    (1, POS_C, "Actual UP"),
                ]:
                    sub_p = df_dd[df_dd["actual"] == cls]["prob"]
                    fig_p.add_trace(go.Histogram(
                        x=sub_p, name=name,
                        marker_color=colour, opacity=0.7,
                        nbinsx=40, histnorm="probability density",
                    ))
                fig_p.add_vline(x=0.5, line_dash="dot",
                                line_color="#888888", line_width=1.2)
                fig_p.update_layout(**{**PLOT_LAYOUT,
                    "barmode": "overlay", "height": 280,
                    "xaxis": {**PLOT_LAYOUT["xaxis"],
                              "title": "P(UP)", "range": [0, 1]},
                    "margin": dict(l=30, r=20, t=30, b=40),
                })
                st.plotly_chart(fig_p, use_container_width=True)

            elif "forecast" in df_dd.columns:
                st.markdown("<div class='sec'>Forecast Distribution</div>",
                            unsafe_allow_html=True)
                fig_fc = go.Figure(go.Histogram(
                    x=df_dd["forecast"], nbinsx=80,
                    marker_color=MODEL_COLORS.get(sel_label, GOLD_C),
                    opacity=0.85,
                ))
                fig_fc.add_vline(x=0, line_dash="dot",
                                 line_color=NEG_C, line_width=1.5,
                                 annotation_text="Decision boundary (0)",
                                 annotation_font=dict(size=10, color=NEG_C))
                fig_fc.update_layout(**{**PLOT_LAYOUT,
                    "height": 280,
                    "xaxis": {**PLOT_LAYOUT["xaxis"], "title": "Forecast Value"},
                    "margin": dict(l=30, r=20, t=30, b=40),
                })
                st.plotly_chart(fig_fc, use_container_width=True)

        # Per-regime accuracy bars
        st.markdown("<div class='sec'>Accuracy by Regime</div>",
                    unsafe_allow_html=True)
        cv1, cv2 = st.columns(2)
        for col, rc, rt in [
            (cv1, "vol_regime",    "Volatility Regime"),
            (cv2, "period_regime", "Period Regime"),
        ]:
            with col:
                if rc not in df_dd.columns:
                    continue
                rdata = []
                for rv in sorted(df_dd[rc].unique()):
                    sub = df_dd[df_dd[rc] == rv]
                    if len(sub) < 5:
                        continue
                    acc = accuracy_score(sub["actual"], sub["pred"])
                    rdata.append({
                        "Regime": rv.replace("_", " ").title(),
                        "Accuracy": acc,
                        "n": len(sub),
                    })
                if not rdata:
                    continue
                rdf = pd.DataFrame(rdata).sort_values("Accuracy")
                fig_rb = go.Figure(go.Bar(
                    x=rdf["Accuracy"],
                    y=rdf["Regime"],
                    orientation="h",
                    marker_color=[POS_C if v > 0.5 else NEG_C
                                  for v in rdf["Accuracy"]],
                    text=[f"{v:.3f}  (n={n})"
                          for v, n in zip(rdf["Accuracy"], rdf["n"])],
                    textposition="outside",
                    textfont=dict(family="monospace", size=10),
                ))
                fig_rb.add_vline(x=0.50, line_dash="dot",
                                 line_color=NEG_C, line_width=1.1)
                fig_rb.update_layout(**{**PLOT_LAYOUT,
                    "height": 240,
                    "title": dict(text=rt,
                                  font=dict(size=11, color="#A0A0B8")),
                    "xaxis": {**PLOT_LAYOUT["xaxis"],
                              "range": [0.40, 0.70], "tickformat": ".0%"},
                    "margin": dict(l=110, r=70, t=40, b=30),
                })
                st.plotly_chart(fig_rb, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — PRICE & SIGNALS
# ════════════════════════════════════════════════════════════
with t5:
    if not sel_keys:
        st.info("Select at least one model from the sidebar.")
        st.stop()

    sig_label = st.selectbox(
        "Signal source model",
        options=[MODEL_LABELS[k] for k in sel_keys],
        key="sig_sel",
    )
    sig_mk = next(k for k, v in MODEL_LABELS.items() if v == sig_label)
    df_sig = all_preds[sig_mk][ak]

    price_col   = "gold"          if ak == "au" else "silver"
    vol_col     = "garch_vol_au"  if ak == "au" else "garch_vol_ag"
    vreg_col    = "vol_regime_au" if ak == "au" else "vol_regime_ag"

    if price_data.empty or price_col not in price_data.columns:
        st.warning("Price data not found.")
    elif df_sig.empty:
        st.warning("No predictions for this model.")
    else:
        st.markdown("<div class='sec'>Price + Signals + GARCH Volatility</div>",
                    unsafe_allow_html=True)

        fig_ps = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )

        # Price line
        fig_ps.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data[price_col],
            name=f"{asset_label} Price",
            line=dict(color=asset_color, width=1.5),
        ), row=1, col=1)

        # Signals in test period
        test_idx = df_sig.index
        price_test = price_data.loc[
            price_data.index.isin(test_idx), price_col
        ]
        up_idx   = df_sig[df_sig["pred"] == 1].index
        down_idx = df_sig[df_sig["pred"] == 0].index
        up_p     = price_data.loc[price_data.index.isin(up_idx),   price_col]
        down_p   = price_data.loc[price_data.index.isin(down_idx), price_col]

        if not up_p.empty:
            fig_ps.add_trace(go.Scatter(
                x=up_p.index, y=up_p,
                name="UP Signal", mode="markers",
                marker=dict(color=POS_C, size=3.5, opacity=0.55),
            ), row=1, col=1)
        if not down_p.empty:
            fig_ps.add_trace(go.Scatter(
                x=down_p.index, y=down_p,
                name="DOWN Signal", mode="markers",
                marker=dict(color=NEG_C, size=3.5, opacity=0.55),
            ), row=1, col=1)

        # GARCH vol
        if vol_col in price_data.columns:
            fig_ps.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data[vol_col],
                name="GARCH Cond. Vol",
                line=dict(color="#7C3AED", width=1.2),
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.08)",
            ), row=2, col=1)

        # Test period shading (both panels)
        if len(test_idx) > 0:
            for r in [1, 2]:
                fig_ps.add_vrect(
                    x0=test_idx[0], x1=test_idx[-1],
                    fillcolor=f"rgba(212,175,55,0.04)",
                    line_width=0.8, line_color=GOLD_C,
                    annotation_text="Test Period →",
                    annotation_position="top left",
                    annotation_font=dict(size=9, color=GOLD_C),
                    row=r, col=1,
                )

        fig_ps.update_layout(**{**PLOT_LAYOUT,
            "title": dict(
                text=f"{asset_label} — Price + {sig_label} Signals + GARCH Vol",
                font=dict(size=13, color=asset_color)),
            "height": 560,
            "showlegend": True,
            "legend": dict(x=0.01, y=0.99, font=dict(size=9)),
        })
        fig_ps.update_yaxes(title_text="Price",     row=1, col=1)
        fig_ps.update_yaxes(title_text="GARCH Vol", row=2, col=1)
        st.plotly_chart(fig_ps, use_container_width=True)

        # Regime overlay
        if vreg_col in price_data.columns:
            st.markdown("<div class='sec'>GARCH Volatility Regimes on Price</div>",
                        unsafe_allow_html=True)
            fig_ro = go.Figure()
            fig_ro.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data[price_col],
                name=f"{asset_label} Price",
                line=dict(color=asset_color, width=1.5),
                zorder=5,
            ))

            # Efficient contiguous-block vrects
            rs     = price_data[vreg_col].dropna()
            blk_s  = rs.index[0]
            blk_r  = rs.iloc[0]
            for dt, reg in rs.items():
                if reg != blk_r:
                    fc = ("rgba(16,185,129,0.08)"
                          if blk_r == "low_vol"
                          else "rgba(239,68,68,0.08)")
                    fig_ro.add_vrect(x0=blk_s, x1=dt,
                                     fillcolor=fc, line_width=0)
                    blk_s = dt
                    blk_r = reg
            # last block
            fc = ("rgba(16,185,129,0.08)"
                  if blk_r == "low_vol"
                  else "rgba(239,68,68,0.08)")
            fig_ro.add_vrect(x0=blk_s, x1=rs.index[-1],
                             fillcolor=fc, line_width=0)

            fig_ro.update_layout(**{**PLOT_LAYOUT,
                "title": dict(
                    text=f"{asset_label} — Regime Overlay "
                         f"(Green=Low Vol · Red=High Vol)",
                    font=dict(size=12, color=asset_color)),
                "height": 340,
                "yaxis": {**PLOT_LAYOUT["yaxis"], "title": "Price"},
            })
            st.plotly_chart(fig_ro, use_container_width=True)

        st.markdown(
            "<div class='insight'>📌 <b>Reading signals:</b> "
            "Green dots = model predicts UP (go long). "
            "Red dots = model predicts DOWN (go short/exit). "
            "Shaded area = out-of-sample test period (Jul 2021 → Jun 2026). "
            "GARCH vol panel below shows the volatility environment — "
            "spikes correspond to High Vol regime periods.</div>",
            unsafe_allow_html=True,
        )
