import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# PAGE CONFIG & MOBILE-FIRST TERMINAL STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Rebound Pulse // Multimeter",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 760px;
        }
        .pulse-header {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }
        .pulse-title {
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 0;
            margin-bottom: 0.4rem;
            line-height: 1.2;
        }
        .regime-pill {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
        }
        .blocker-card {
            border-radius: 10px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
            border-left: 4px solid #ef4444;
            background-color: rgba(148, 163, 184, 0.08);
        }
        .blocker-title {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }
        .blocker-desc {
            font-size: 0.84rem;
            opacity: 0.88;
            margin: 0;
        }
        .small-muted {
            font-size: 0.78rem;
            color: #94a3b8;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HELPERS & TECHNICAL CALCULATIONS
# -----------------------------------------------------------------------------
def clamp(val: float, lo: float, hi: float) -> float:
    """Keep slider defaults safely within min/max bounds during extreme market moves."""
    return max(lo, min(hi, float(val)))


def get_sydney_time_str() -> str:
    """Return current Sydney time with seconds so refreshes are visibly confirmed."""
    try:
        tz = ZoneInfo("Australia/Sydney")
    except Exception:
        tz = datetime.timezone(datetime.timedelta(hours=11), name="AEDT")
    return datetime.datetime.now(tz).strftime("%d %b %Y, %I:%M:%S %p %Z")


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder-style 14-day RSI."""
    clean = series.dropna()
    if len(clean) < period + 2:
        return 50.0
    delta = clean.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Safely extract a single ticker's Close series across yfinance MultiIndex formats."""
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0) and ticker in df["Close"].columns:
            return df["Close"][ticker].dropna()
        if ticker in df.columns.get_level_values(0) and "Close" in df[ticker].columns:
            return df[ticker]["Close"].dropna()
    elif "Close" in df.columns:
        return df["Close"].dropna()
    return pd.Series(dtype=float)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_market_data() -> dict:
    """Fetch live macro & technical indicators from Yahoo Finance and FRED."""
    tickers = ["^TNX", "^VIX", "QQQ", "SMH", "GC=F", "DX-Y.NYB", "BTC-USD"]
    raw = yf.download(
        tickers,
        period="3mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    tnx = extract_close_series(raw, "^TNX")
    vix = extract_close_series(raw, "^VIX")
    qqq = extract_close_series(raw, "QQQ")
    smh = extract_close_series(raw, "SMH")
    gold = extract_close_series(raw, "GC=F")
    dxy = extract_close_series(raw, "DX-Y.NYB")
    btc = extract_close_series(raw, "BTC-USD")

    # Last Yahoo Finance equity candle date
    yf_last_date = qqq.index[-1].strftime("%Y-%m-%d") if len(qqq) else "N/A"

    # 1. 10-Year Treasury Yield (%) — normalize entire series before 5D delta
    if len(tnx):
        if float(tnx.iloc[-1]) > 20:
            tnx = tnx / 10.0
        yield_10y = float(tnx.iloc[-1])
        yield_5d_chg = float(yield_10y - tnx.iloc[-6]) if len(tnx) >= 6 else 0.0
    else:
        yield_10y, yield_5d_chg = 4.50, 0.0

    # 2. VIX level & trajectory
    vix_val = float(vix.iloc[-1]) if len(vix) else 20.0
    if len(vix) >= 5:
        vix_5d_max = float(vix.iloc[-5:].max())
        vix_3d_chg = float(vix.iloc[-1] - vix.iloc[-3])
        if vix_val >= vix_5d_max * 0.97 and vix_3d_chg > 0.5:
            vix_state = "Spiking"
        elif vix_val < vix_5d_max * 0.92 and vix_3d_chg < -0.5:
            vix_state = "Rolling Over"
        else:
            vix_state = "Peaking / Stalling"
    else:
        vix_state = "Peaking / Stalling"

    # 3. Tech & Semis RSI (average of QQQ and SMH)
    qqq_rsi = compute_rsi(qqq)
    smh_rsi = compute_rsi(smh)
    tech_rsi = round((qqq_rsi + smh_rsi) / 2.0, 1)

    # 4. Gold vs. Tech 10-day rolling return correlation
    gold_ret = gold.pct_change()
    qqq_ret = qqq.pct_change()
    aligned = pd.concat([gold_ret, qqq_ret], axis=1).dropna()
    if len(aligned) >= 10:
        gold_corr = float(aligned.iloc[-10:, 0].corr(aligned.iloc[-10:, 1]))
        if np.isnan(gold_corr):
            gold_corr = 0.0
    else:
        gold_corr = 0.0

    # 5. US Dollar Index (DXY)
    dxy_val = float(dxy.iloc[-1]) if len(dxy) else 103.0
    dxy_5d_chg = float(dxy_val - dxy.iloc[-6]) if len(dxy) >= 6 else 0.0

    # 6. High-Yield Credit Spread (FRED: BAMLH0A0HYM2, in %)
    hy_spread = 3.50
    hy_5d_chg = 0.0
    fred_ok = False
    fred_last_date = "N/A"
    try:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
        fred_df = pd.read_csv(fred_url)
        fred_df.columns = ["date", "val"]
        fred_df["val"] = pd.to_numeric(fred_df["val"], errors="coerce")
        fred_df = fred_df.dropna(subset=["val"])
        if len(fred_df) >= 6:
            hy_spread = float(fred_df["val"].iloc[-1])
            hy_5d_chg = float(hy_spread - fred_df["val"].iloc[-6])
            fred_last_date = str(fred_df["date"].iloc[-1])
            fred_ok = True
    except Exception:
        fred_ok = False

    # 7. Crypto Deleveraging Stress Proxy (BTC 5-day drawdown %)
    if len(btc) >= 6:
        btc_5d_high = float(btc.iloc[-6:].max())
        btc_dd_pct = max(0.0, (btc_5d_high - float(btc.iloc[-1])) / btc_5d_high * 100.0)
    else:
        btc_dd_pct = 0.0

    return {
        "timestamp": get_sydney_time_str(),
        "yf_last_date": yf_last_date,
        "fred_last_date": fred_last_date,
        "yield_10y": round(yield_10y, 2),
        "yield_5d_chg": round(yield_5d_chg, 2),
        "vix": round(vix_val, 2),
        "vix_state": vix_state,
        "qqq_rsi": round(qqq_rsi, 1),
        "smh_rsi": round(smh_rsi, 1),
        "tech_rsi": tech_rsi,
        "gold_corr": round(gold_corr, 2),
        "dxy": round(dxy_val, 2),
        "dxy_5d_chg": round(dxy_5d_chg, 2),
        "hy_spread": round(hy_spread, 2),
        "hy_5d_chg": round(hy_5d_chg, 2),
        "fred_ok": fred_ok,
        "btc_dd_pct": round(btc_dd_pct, 1),
    }


# -----------------------------------------------------------------------------
# SCORING & BOTTLENECK DIAGNOSTIC ENGINE
# -----------------------------------------------------------------------------
def evaluate_rebound(
    yield_10y: float,
    vix: float,
    vix_state: str,
    tech_rsi: float,
    gold_corr: float,
    hy_spread: float,
    dxy: float,
    btc_dd_pct: float,
) -> dict:
    components = []

    # 1. 10-Year Treasury Yield (Max 25 pts)
    max_y = 25.0
    if yield_10y <= 4.00:
        pts_y, status_y = 25.0, "GREEN LIGHT"
        msg_y = f"10Y Yield at {yield_10y:.2f}% is in the Goldilocks zone (3.5%–4.0%), supporting tech multiple expansion."
    elif yield_10y <= 4.50:
        pts_y = 18.0 + (4.50 - yield_10y) / 0.50 * 7.0
        status_y = "GREEN LIGHT"
        msg_y = f"10Y Yield at {yield_10y:.2f}% gives equities breathing room below the 4.50% pressure line."
    elif yield_10y <= 4.90:
        pts_y = 6.0 + (4.90 - yield_10y) / 0.40 * 12.0
        status_y = "CAUTION"
        msg_y = f"10Y Yield at {yield_10y:.2f}% is compressing high-growth tech valuations. Needs < 4.50%."
    else:
        pts_y = max(0.0, 6.0 - (yield_10y - 4.90) * 15.0)
        status_y = "CRITICAL BLOCKER"
        msg_y = f"10Y Yield at {yield_10y:.2f}% is in the 5% danger zone, choking off AI/tech recovery."
    components.append(("10Y Treasury Yield", pts_y, max_y, status_y, msg_y, "LT"))

    # 2. VIX Level & Trajectory (Max 18 pts)
    max_v = 18.0
    if vix_state == "Rolling Over":
        pts_v = 18.0 if vix >= 22 else 14.0
        status_v = "GREEN LIGHT"
        msg_v = f"VIX ({vix:.1f}) is rolling over — classic signal that panic selling is exhausting."
    elif vix_state == "Peaking / Stalling":
        pts_v = 11.0 if vix >= 25 else 9.0
        status_v = "CAUTION"
        msg_v = f"VIX ({vix:.1f}) is stalling. Watch for a lower high to confirm fear has peaked."
    else:
        pts_v = 3.0 if vix >= 30 else 5.0
        status_v = "CRITICAL BLOCKER"
        msg_v = f"VIX ({vix:.1f}) is actively spiking — catching a falling knife until volatility stops making new highs."
    components.append(("VIX Fear Trajectory", pts_v, max_v, status_v, msg_v, "ST"))

    # 3. Gold / Tech Correlation (Max 15 pts)
    max_g = 15.0
    if gold_corr <= 0.15:
        pts_g, status_g = 15.0, "GREEN LIGHT"
        msg_g = f"Gold/Tech correlation ({gold_corr:+.2f}) has decoupled — forced margin-call selling appears finished."
    elif gold_corr <= 0.45:
        pts_g, status_g = 9.0, "CAUTION"
        msg_g = f"Gold/Tech correlation ({gold_corr:+.2f}) is moderately positive — lingering cross-asset deleveraging."
    else:
        pts_g = max(0.0, 4.0 - (gold_corr - 0.45) * 8.0)
        status_g = "CRITICAL BLOCKER"
        msg_g = f"Gold/Tech correlation ({gold_corr:+.2f}) is elevated — safe havens are still being sold for cash."
    components.append(("Gold / Tech Decoupling", pts_g, max_g, status_g, msg_g, "BOTH"))

    # 4. High-Yield Credit Spreads (Max 15 pts)
    max_hy = 15.0
    if hy_spread <= 3.75:
        pts_hy, status_hy = 15.0, "GREEN LIGHT"
        msg_hy = f"HY Credit Spreads ({hy_spread:.2f}%) remain calm — bond market is not pricing a systemic freeze."
    elif hy_spread <= 4.75:
        pts_hy, status_hy = 8.0, "CAUTION"
        msg_hy = f"HY Credit Spreads ({hy_spread:.2f}%) are widening. Monitor for corporate stress."
    else:
        pts_hy = max(0.0, 4.0 - (hy_spread - 4.75) * 2.0)
        status_hy = "CRITICAL BLOCKER"
        msg_hy = f"HY Credit Spreads ({hy_spread:.2f}%) signal recession/credit stress — rallies risk failing."
    components.append(("HY Credit Spreads (OAS)", pts_hy, max_hy, status_hy, msg_hy, "LT"))

    # 5. Tech & Semis 14-Day RSI (Max 12 pts)
    max_r = 12.0
    if 28 <= tech_rsi <= 42:
        pts_r, status_r = 12.0, "GREEN LIGHT"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is in the coiled oversold-to-rebound sweet spot."
    elif tech_rsi < 28:
        pts_r, status_r = 9.0, "CAUTION"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is deeply oversold — primed for a sharp snapback, though still washed out."
    elif tech_rsi <= 55:
        pts_r, status_r = 8.0, "GREEN LIGHT"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is neutral and stabilizing."
    else:
        pts_r, status_r = 5.0, "CAUTION"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is not oversold — less pent-up short-covering fuel."
    components.append(("Tech & Semis RSI (QQQ/SMH)", pts_r, max_r, status_r, msg_r, "ST"))

    # 6. US Dollar Index — DXY (Max 8 pts)
    max_d = 8.0
    if dxy <= 101.5:
        pts_d, status_d = 8.0, "GREEN LIGHT"
        msg_d = f"DXY ({dxy:.2f}) is relaxed — global dollar liquidity pressure is off."
    elif dxy <= 104.5:
        pts_d, status_d = 4.5, "CAUTION"
        msg_d = f"DXY ({dxy:.2f}) is firm — mild 'dash for cash' headwind for risk assets & commodities."
    else:
        pts_d, status_d = 1.0, "CRITICAL BLOCKER"
        msg_d = f"DXY ({dxy:.2f}) is surging — strong dollar squeeze is crushing equities, gold, and crypto."
    components.append(("US Dollar Liquidity (DXY)", pts_d, max_d, status_d, msg_d, "LT"))

    # 7. Crypto Leverage Flush Proxy (Max 7 pts)
    max_c = 7.0
    if btc_dd_pct <= 4.0:
        pts_c, status_c = 7.0, "GREEN LIGHT"
        msg_c = f"BTC 5D drawdown ({btc_dd_pct:.1f}%) is muted — forced crypto liquidations have cooled."
    elif btc_dd_pct <= 10.0:
        pts_c, status_c = 4.0, "CAUTION"
        msg_c = f"BTC 5D drawdown ({btc_dd_pct:.1f}%) shows active deleveraging in high-beta risk."
    else:
        pts_c, status_c = 1.0, "CRITICAL BLOCKER"
        msg_c = f"BTC 5D drawdown ({btc_dd_pct:.1f}%) signals heavy forced liquidations still flushing out."
    components.append(("Crypto Leverage Flush", pts_c, max_c, status_c, msg_c, "ST"))

    total_score = round(sum(c[1] for c in components), 1)
    st_items = [c for c in components if c[5] in ("ST", "BOTH")]
    lt_items = [c for c in components if c[5] in ("LT", "BOTH")]
    st_score = round(sum(c[1] for c in st_items) / sum(c[2] for c in st_items) * 100, 1)
    lt_score = round(sum(c[1] for c in lt_items) / sum(c[2] for c in lt_items) * 100, 1)

    if total_score < 35:
        regime, color, bg = "FALLING KNIFE", "#ef4444", "rgba(239, 68, 68, 0.16)"
    elif total_score < 55:
        regime, color, bg = "DEAD-CAT BOUNCE ZONE", "#f59e0b", "rgba(245, 158, 11, 0.16)"
    elif total_score < 75:
        regime, color, bg = "EARLY ACCUMULATION", "#38bdf8", "rgba(56, 189, 248, 0.16)"
    else:
        regime, color, bg = "HIGH-CONVICTION REBOUND", "#10b981", "rgba(16, 185, 129, 0.16)"

    diagnostics = [
        {
            "name": name,
            "earned": round(earned, 1),
            "max": max_p,
            "drag": round(max_p - earned, 1),
            "status": status,
            "msg": msg,
        }
        for name, earned, max_p, status, msg, _ in components
    ]
    diagnostics.sort(key=lambda x: x["drag"], reverse=True)

    return {
        "score": total_score,
        "st_score": st_score,
        "lt_score": lt_score,
        "regime": regime,
        "color": color,
        "bg": bg,
        "diagnostics": diagnostics,
    }


def render_multimeter(score: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 46, "color": color, "family": "monospace"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#64748b",
                    "tickvals": [0, 35, 55, 75, 100],
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(15, 23, 42, 0.4)",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.25)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(239, 68, 68, 0.22)"},
                    {"range": [35, 55], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [55, 75], "color": "rgba(56, 189, 248, 0.22)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


# -----------------------------------------------------------------------------
# MAIN UI (LIVE REFRESH + TOAST + PRESETS)
# -----------------------------------------------------------------------------
st.markdown('<div class="pulse-header">REBOUND PULSE // LIVE MACRO TERMINAL</div>', unsafe_allow_html=True)
st.markdown('<div class="pulse-title">Market Rebound Multimeter</div>', unsafe_allow_html=True)

SIM_KEYS = [
    "sim_mode",
    "s_yield",
    "s_vix",
    "s_vix_state",
    "s_rsi",
    "s_corr",
    "s_hy",
    "s_dxy",
    "s_btc_dd",
]

PRESETS = {
    "Goldilocks Bottom": {
        "s_yield": 3.95,
        "s_vix": 24.0,
        "s_vix_state": "Rolling Over",
        "s_rsi": 34.0,
        "s_corr": -0.10,
        "s_hy": 3.40,
        "s_dxy": 100.5,
        "s_btc_dd": 3.0,
    },
    "Dead-Cat Bounce": {
        "s_yield": 4.90,
        "s_vix": 28.5,
        "s_vix_state": "Peaking / Stalling",
        "s_rsi": 36.0,
        "s_corr": 0.60,
        "s_hy": 4.10,
        "s_dxy": 104.25,
        "s_btc_dd": 8.5,
    },
    "2020-Style Freeze": {
        "s_yield": 5.20,
        "s_vix": 48.0,
        "s_vix_state": "Spiking",
        "s_rsi": 22.0,
        "s_corr": 0.85,
        "s_hy": 6.50,
        "s_dxy": 108.5,
        "s_btc_dd": 22.0,
    },
}


def reset_and_refresh_live():
    fetch_live_market_data.clear()
    for k in SIM_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["sim_mode"] = False
    st.session_state["just_refreshed"] = True


def apply_preset(preset_name: str):
    st.session_state["sim_mode"] = True
    for k, v in PRESETS[preset_name].items():
        st.session_state[k] = v


top_c1, top_c2 = st.columns([1.8, 1.2])
with top_c1:
    sim_mode = st.toggle("🎛️ What-If Simulator Mode", key="sim_mode")
with top_c2:
    st.button(
        "🔄 Reset & Refresh Live",
        on_click=reset_and_refresh_live,
        use_container_width=True,
    )

with st.spinner("Pulling live macro feeds (Yahoo Finance + FRED)..."):
    live = fetch_live_market_data()

if st.session_state.pop("just_refreshed", False):
    st.toast(f"Live feeds synced at {live['timestamp']}", icon="⚡")

if sim_mode:
    with st.expander("Adjust Indicators & Scenario Presets", expanded=True):
        p1, p2, p3 = st.columns(3)
        p1.button("🟢 Goldilocks", on_click=apply_preset, args=("Goldilocks Bottom",), use_container_width=True)
        p2.button("🟡 Dead-Cat", on_click=apply_preset, args=("Dead-Cat Bounce",), use_container_width=True)
        p3.button("🔴 2020 Freeze", on_click=apply_preset, args=("2020-Style Freeze",), use_container_width=True)

        s_yield = st.slider(
            "10-Year Treasury Yield (%)", 2.50, 6.00, clamp(live["yield_10y"], 2.50, 6.00), 0.05, key="s_yield"
        )
        s_vix = st.slider(
            "VIX Index", 10.0, 65.0, clamp(live["vix"], 10.0, 65.0), 0.5, key="s_vix"
        )
        vix_opts = ["Spiking", "Peaking / Stalling", "Rolling Over"]
        s_vix_state = (
            st.segmented_control(
                "VIX Trajectory",
                options=vix_opts,
                default=live["vix_state"],
                key="s_vix_state",
            )
            or live["vix_state"]
        )
        s_rsi = st.slider(
            "Tech & Semis 14D RSI (Avg QQQ/SMH)", 10.0, 85.0, clamp(live["tech_rsi"], 10.0, 85.0), 0.5, key="s_rsi"
        )
        s_corr = st.slider(
            "Gold / Tech 10D Correlation", -1.0, 1.0, clamp(live["gold_corr"], -1.0, 1.0), 0.05, key="s_corr"
        )
        s_hy = st.slider(
            "HY Credit Spreads — FRED OAS (%)", 2.00, 9.00, clamp(live["hy_spread"], 2.00, 9.00), 0.05, key="s_hy"
        )
        s_dxy = st.slider(
            "US Dollar Index (DXY)", 90.0, 115.0, clamp(live["dxy"], 90.0, 115.0), 0.25, key="s_dxy"
        )
        s_btc_dd = st.slider(
            "Crypto Flush Proxy (BTC 5D Drawdown %)", 0.0, 35.0, clamp(live["btc_dd_pct"], 0.0, 35.0), 0.5, key="s_btc_dd"
        )
else:
    s_yield = live["yield_10y"]
    s_vix = live["vix"]
    s_vix_state = live["vix_state"]
    s_rsi = live["tech_rsi"]
    s_corr = live["gold_corr"]
    s_hy = live["hy_spread"]
    s_dxy = live["dxy"]
    s_btc_dd = live["btc_dd_pct"]

result = evaluate_rebound(s_yield, s_vix, s_vix_state, s_rsi, s_corr, s_hy, s_dxy, s_btc_dd)

# Regime Pill & Gauge
st.markdown(
    f'<span class="regime-pill" style="color:{result["color"]}; background:{result["bg"]}; border:1px solid {result["color"]};">'
    f'● {result["regime"]}</span>',
    unsafe_allow_html=True,
)
st.plotly_chart(render_multimeter(result["score"], result["color"]), use_container_width=True, config={"displayModeBar": False})

# Dual Recovery Telemetry Bars
t1, t2 = st.columns(2)
with t1:
    st.metric("Short-Term Snapback (1–10D)", f"{result['st_score']:.0f}%")
    st.progress(min(100, int(result["st_score"])))
with t2:
    st.metric("Sustained Recovery (3–6M)", f"{result['lt_score']:.0f}%")
    st.progress(min(100, int(result["lt_score"])))

st.divider()

# Bottleneck & Blocker Diagnostics
st.subheader("🚨 Bottlenecks Holding Up the Rebound")
blockers = [d for d in result["diagnostics"] if d["status"] != "GREEN LIGHT"]
greens = [d for d in result["diagnostics"] if d["status"] == "GREEN LIGHT"]

if not blockers:
    st.success("All major macro and technical indicators are aligned for a sustained rebound.")
else:
    for b in blockers:
        border_col = "#ef4444" if b["status"] == "CRITICAL BLOCKER" else "#f59e0b"
        st.markdown(
            f"""
            <div class="blocker-card" style="border-left-color: {border_col};">
                <div class="blocker-title">
                    {b['name']} — <span style="color:{border_col};">{b['status']}</span>
                    <span style="float:right; font-family:monospace; color:{border_col};">-{b['drag']} pts</span>
                </div>
                <p class="blocker-desc">{b['msg']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if greens:
    with st.expander(f"✅ Supporting Indicators ({len(greens)} Green Lights)", expanded=False):
        for g in greens:
            st.markdown(f"**{g['name']}** ({g['earned']}/{g['max']} pts) — {g['msg']}")

st.divider()

# Live Telemetry Grid
st.subheader("📡 Live Market Telemetry")
m1, m2, m3 = st.columns(3)
m1.metric("10Y Yield (^TNX)", f"{s_yield:.2f}%", f"{live['yield_5d_chg']:+.2f}% (5D)", delta_color="inverse")
m2.metric("VIX", f"{s_vix:.1f}", s_vix_state)
m3.metric("Tech/Semi RSI", f"{s_rsi:.1f}", f"QQQ {live['qqq_rsi']} | SMH {live['smh_rsi']}")

m4, m5, m6 = st.columns(3)
m4.metric("Gold/Tech Corr", f"{s_corr:+.2f}", "Decoupled" if s_corr <= 0.15 else "Correlated")
m5.metric("HY Spread (OAS)", f"{s_hy:.2f}%", f"{live['hy_5d_chg']:+.2f}% (5D)", delta_color="inverse")
m6.metric("Dollar (DXY)", f"{s_dxy:.2f}", f"{live['dxy_5d_chg']:+.2f} (5D)", delta_color="inverse")

fred_note = f"FRED ({live['fred_last_date']})" if live["fred_ok"] else "FRED fallback"
st.markdown(
    f'<div class="small-muted">'
    f'Last sync: <b>{live["timestamp"]}</b><br>'
    f'Latest market candles: Yahoo Finance ({live["yf_last_date"]}) • {fred_note} • Auto-cache: 5 min'
    f'</div>',
    unsafe_allow_html=True,
)
