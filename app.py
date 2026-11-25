import datetime
from email.utils import parsedate_to_datetime
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
    page_title="Rebound Pulse // Live Timing Terminal",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 840px;
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
        .tape-box {
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(56, 189, 248, 0.25);
            background-color: rgba(15, 23, 42, 0.55);
        }
        .regime-pill {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }
        .playbook-card {
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
            margin: 0.5rem 0 0.9rem 0;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background-color: rgba(15, 23, 42, 0.45);
        }
        .playbook-label {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.75rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 0.2rem;
        }
        .playbook-action {
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .playbook-note {
            font-size: 0.85rem;
            opacity: 0.9;
            margin: 0;
            line-height: 1.45;
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
    return max(lo, min(hi, float(val)))


def get_sydney_time_str() -> str:
    try:
        tz = ZoneInfo("Australia/Sydney")
    except Exception:
        tz = datetime.timezone(datetime.timedelta(hours=11), name="AEDT")
    return datetime.datetime.now(tz).strftime("%d %b %Y, %I:%M:%S %p %Z")


def extract_field_series(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if field in df.columns.get_level_values(0) and ticker in df[field].columns:
            return df[field][ticker].dropna()
        if ticker in df.columns.get_level_values(0) and field in df[ticker].columns:
            return df[ticker][field].dropna()
    elif field in df.columns:
        return df[field].dropna()
    return pd.Series(dtype=float)


def compute_rsi_metrics(series: pd.Series, period: int = 14) -> tuple[float, float]:
    clean = series.dropna()
    if len(clean) < period + 5:
        return 50.0, 0.0
    delta = clean.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).dropna()
    if len(rsi) < 4:
        return 50.0, 0.0
    return float(rsi.iloc[-1]), float(rsi.iloc[-1] - rsi.iloc[-4])


def compute_atr(high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, period: int = 14) -> tuple[float, float]:
    """Return (14D ATR in $, 14D ATR as % of last close)."""
    df = pd.concat([high_s.rename("h"), low_s.rename("l"), close_s.rename("c")], axis=1).dropna()
    if len(df) < period + 2:
        last = float(close_s.iloc[-1]) if len(close_s) else 0.0
        return 0.0, 0.0
    prev_c = df["c"].shift(1)
    tr = pd.concat(
        [(df["h"] - df["l"]), (df["h"] - prev_c).abs(), (df["l"] - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr_val = float(tr.rolling(period).mean().iloc[-1])
    last_c = float(df["c"].iloc[-1])
    atr_pct = (atr_val / last_c * 100.0) if last_c > 0 else 0.0
    return round(atr_val, 2), round(atr_pct, 2)


def compute_red_vol_ratio(close_s: pd.Series, vol_s: pd.Series) -> float:
    aligned = pd.concat([close_s.rename("c"), vol_s.rename("v")], axis=1).dropna()
    if len(aligned) < 22:
        return 1.0
    avg_vol_20 = float(aligned["v"].rolling(20).mean().iloc[-1])
    if avg_vol_20 <= 0:
        return 1.0
    recent = aligned.iloc[-5:].copy()
    recent["chg"] = aligned["c"].diff().iloc[-5:]
    red_days = recent[recent["chg"] < 0]
    if red_days.empty:
        return 0.75
    return float(red_days["v"].iloc[-1] / avg_vol_20)


def ma200_dist_and_dd(s: pd.Series) -> tuple[float, float]:
    if len(s) < 30:
        return 0.0, 0.0
    window = min(200, len(s))
    ma200 = float(s.rolling(window).mean().iloc[-1])
    last = float(s.iloc[-1])
    high_52w = float(s.max())
    return (last - ma200) / ma200 * 100.0, (last - high_52w) / high_52w * 100.0


def ema_dist(s: pd.Series, span: int = 50) -> float:
    if len(s) < 10:
        return 0.0
    ema = float(s.ewm(span=min(span, len(s)), adjust=False).mean().iloc[-1])
    last = float(s.iloc[-1])
    return (last - ema) / ema * 100.0 if ema > 0 else 0.0


def compute_session_vwap_and_wick(df_15m: pd.DataFrame, ticker: str) -> tuple[float, float, float]:
    c = extract_field_series(df_15m, "Close", ticker)
    h = extract_field_series(df_15m, "High", ticker)
    l = extract_field_series(df_15m, "Low", ticker)
    v = extract_field_series(df_15m, "Volume", ticker)
    aligned = pd.concat([c.rename("c"), h.rename("h"), l.rename("l"), v.rename("v")], axis=1).dropna()
    if aligned.empty:
        return 0.0, 0.0, 50.0

    last_date = aligned.index[-1].date()
    sess = aligned[aligned.index.date == last_date]
    if sess.empty:
        sess = aligned.iloc[-26:]

    last_price = float(sess["c"].iloc[-1])
    tp = (sess["h"] + sess["l"] + sess["c"]) / 3.0
    vol_sum = float(sess["v"].sum())
    vwap = float((tp * sess["v"]).sum() / vol_sum) if vol_sum > 0 else float(tp.mean())
    vwap_dist_pct = (last_price - vwap) / vwap * 100.0 if vwap > 0 else 0.0

    sess_high = float(sess["h"].max())
    sess_low = float(sess["l"].min())
    wick_pos = (last_price - sess_low) / (sess_high - sess_low) * 100.0 if sess_high > sess_low else 50.0
    return last_price, round(vwap_dist_pct, 2), round(wick_pos, 0)


# -----------------------------------------------------------------------------
# 9-INDICATOR SCORING & BUYING PLAYBOOK ENGINE (TOTAL = 100 PTS)
# -----------------------------------------------------------------------------
def evaluate_rebound(
    yield_10y: float,
    vix: float,
    vix_ratio: float,
    vix_state: str,
    gold_corr: float,
    hy_spread: float,
    ma200_dist: float,
    breadth_spread: float,
    tech_rsi: float,
    mom_state: str,
    dxy: float,
    btc_dd_pct: float,
) -> dict:
    components = []

    # 1. 10-Year Treasury Yield (20 pts)
    max_y = 20.0
    if yield_10y <= 4.00:
        pts_y, status_y = 20.0, "GREEN LIGHT"
        msg_y = f"10Y Yield ({yield_10y:.2f}%) is in the Goldilocks zone — strong tailwind for tech multiples."
    elif yield_10y <= 4.50:
        pts_y = 14.0 + (4.50 - yield_10y) / 0.50 * 6.0
        status_y = "GREEN LIGHT"
        msg_y = f"10Y Yield ({yield_10y:.2f}%) is below the 4.50% pressure line, giving equities breathing room."
    elif yield_10y <= 4.90:
        pts_y = 5.0 + (4.90 - yield_10y) / 0.40 * 9.0
        status_y = "CAUTION"
        msg_y = f"10Y Yield ({yield_10y:.2f}%) is compressing growth valuations. Needs to cool under 4.50%."
    else:
        pts_y = max(0.0, 5.0 - (yield_10y - 4.90) * 12.0)
        status_y = "CRITICAL BLOCKER"
        msg_y = f"10Y Yield ({yield_10y:.2f}%) is in the 5% danger zone — directly capping AI & semi rallies."
    components.append(("10Y Treasury Yield", pts_y, max_y, status_y, msg_y, "LT"))

    # 2. VIX Level, Trajectory & Term Structure (15 pts)
    max_v = 15.0
    if vix_state == "Rolling Over" and vix_ratio < 1.00:
        pts_v = 15.0 if vix >= 22 else 12.5
        status_v = "GREEN LIGHT"
        msg_v = f"VIX ({vix:.1f}) is rolling over with healthy term structure (VIX/VIX3M = {vix_ratio:.2f})."
    elif vix_state == "Peaking / Stalling" or (vix_state == "Rolling Over" and vix_ratio >= 1.00):
        pts_v = 8.5 if vix_ratio < 1.05 else 6.0
        status_v = "CAUTION"
        msg_v = f"VIX ({vix:.1f}, ratio {vix_ratio:.2f}) is stalling. Watch for VIX/VIX3M to snap cleanly below 1.00."
    else:
        pts_v = 2.0 if vix_ratio >= 1.05 else 3.5
        status_v = "CRITICAL BLOCKER"
        msg_v = f"VIX ({vix:.1f}) is spiking in backwardation (VIX/VIX3M = {vix_ratio:.2f}) — active panic."
    components.append(("VIX & Term Structure", pts_v, max_v, status_v, msg_v, "ST"))

    # 3. Gold / Tech Correlation (13 pts)
    max_g = 13.0
    if gold_corr <= 0.15:
        pts_g, status_g = 13.0, "GREEN LIGHT"
        msg_g = f"Gold/Tech correlation ({gold_corr:+.2f}) has decoupled — forced margin-call selling is fading."
    elif gold_corr <= 0.45:
        pts_g, status_g = 7.5, "CAUTION"
        msg_g = f"Gold/Tech correlation ({gold_corr:+.2f}) is moderately positive — lingering deleveraging."
    else:
        pts_g = max(0.0, 3.5 - (gold_corr - 0.45) * 6.0)
        status_g = "CRITICAL BLOCKER"
        msg_g = f"Gold/Tech correlation ({gold_corr:+.2f}) is elevated — safe havens are still being sold for cash."
    components.append(("Gold / Tech Decoupling", pts_g, max_g, status_g, msg_g, "BOTH"))

    # 4. High-Yield Credit Spreads (12 pts)
    max_hy = 12.0
    if hy_spread <= 3.75:
        pts_hy, status_hy = 12.0, "GREEN LIGHT"
        msg_hy = f"HY Credit Spreads ({hy_spread:.2f}%) are calm — no systemic credit freeze."
    elif hy_spread <= 4.75:
        pts_hy, status_hy = 6.5, "CAUTION"
        msg_hy = f"HY Credit Spreads ({hy_spread:.2f}%) are widening — watch live HYG/IEF ratio above."
    else:
        pts_hy = max(0.0, 3.0 - (hy_spread - 4.75) * 1.5)
        status_hy = "CRITICAL BLOCKER"
        msg_hy = f"HY Credit Spreads ({hy_spread:.2f}%) signal recession/credit stress — bounces risk failing."
    components.append(("HY Credit Spreads (OAS)", pts_hy, max_hy, status_hy, msg_hy, "LT"))

    # 5. Price Structure: 200-Day Moving Average Support (12 pts)
    max_m = 12.0
    if -3.0 <= ma200_dist <= 5.0:
        pts_m, status_m = 12.0, "GREEN LIGHT"
        msg_m = f"QQQ/SMH ({ma200_dist:+.1f}% vs 200D MA) is sitting right at major institutional support."
    elif ma200_dist > 5.0:
        pts_m, status_m = 8.5, "GREEN LIGHT"
        msg_m = f"QQQ/SMH ({ma200_dist:+.1f}% above 200D MA) remains in a healthy primary uptrend."
    elif ma200_dist >= -8.0:
        pts_m, status_m = 5.5, "CAUTION"
        msg_m = f"QQQ/SMH ({ma200_dist:+.1f}% below 200D MA) broke trend support — scale in slowly until reclaimed."
    else:
        pts_m, status_m = 2.0, "CRITICAL BLOCKER"
        msg_m = f"QQQ/SMH ({ma200_dist:+.1f}% below 200D MA) is in deep bear-market territory — wait for base building."
    components.append(("200-Day MA Support (QQQ/SMH)", pts_m, max_m, status_m, msg_m, "LT"))

    # 6. Market Breadth: RSP vs QQQ 5D Spread (8 pts)
    max_b = 8.0
    if breadth_spread >= 0.5:
        pts_b, status_b = 8.0, "GREEN LIGHT"
        msg_b = f"Equal-Weight S&P (RSP) is beating QQQ by {breadth_spread:+.1f}% (5D) — broad participation."
    elif breadth_spread >= -1.0:
        pts_b, status_b = 5.5, "GREEN LIGHT"
        msg_b = f"Breadth spread ({breadth_spread:+.1f}%) is balanced — no severe narrow-leadership divergence."
    else:
        pts_b, status_b = 2.5, "CAUTION"
        msg_b = f"Breadth lagging ({breadth_spread:+.1f}%) — narrow mega-cap moves often signal a fragile bounce."
    components.append(("Market Breadth (RSP vs QQQ)", pts_b, max_b, status_b, msg_b, "BOTH"))

    # 7. Tech & Semis RSI + Selling Exhaustion (8 pts)
    max_r = 8.0
    if mom_state == "Diving (Heavy Vol)":
        pts_r, status_r = 2.0, "CRITICAL BLOCKER"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is still diving on heavy sell volume — no selling exhaustion yet."
    elif tech_rsi <= 45 and mom_state == "Curling Up":
        pts_r, status_r = 8.0, "GREEN LIGHT"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is curling up from oversold levels — selling exhaustion confirmed."
    elif tech_rsi <= 45 and mom_state == "Stalling / Drying Up":
        pts_r, status_r = 6.0, "GREEN LIGHT"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is stabilizing with fading red-day volume."
    elif tech_rsi <= 58:
        pts_r, status_r = 5.5, "GREEN LIGHT"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is neutral ({mom_state})."
    else:
        pts_r, status_r = 3.5, "CAUTION"
        msg_r = f"Tech/Semi RSI ({tech_rsi:.1f}) is elevated — limited oversold snapback fuel."
    components.append(("RSI & Selling Exhaustion", pts_r, max_r, status_r, msg_r, "ST"))

    # 8. US Dollar Index — DXY (6 pts)
    max_d = 6.0
    if dxy <= 101.5:
        pts_d, status_d = 6.0, "GREEN LIGHT"
        msg_d = f"DXY ({dxy:.2f}) is relaxed — global dollar liquidity pressure is off."
    elif dxy <= 104.5:
        pts_d, status_d = 3.5, "CAUTION"
        msg_d = f"DXY ({dxy:.2f}) is firm — mild headwind for risk assets & commodities."
    else:
        pts_d, status_d = 1.0, "CRITICAL BLOCKER"
        msg_d = f"DXY ({dxy:.2f}) is surging — strong dollar squeeze is pressuring global assets."
    components.append(("US Dollar Liquidity (DXY)", pts_d, max_d, status_d, msg_d, "LT"))

    # 9. Crypto Leverage Flush Proxy (6 pts)
    max_c = 6.0
    if btc_dd_pct <= 4.0:
        pts_c, status_c = 6.0, "GREEN LIGHT"
        msg_c = f"BTC 5D drawdown ({btc_dd_pct:.1f}%) is muted — forced crypto liquidations have cooled."
    elif btc_dd_pct <= 10.0:
        pts_c, status_c = 3.5, "CAUTION"
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
        regime, color, bg, alloc_pct = "FALLING KNIFE", "#ef4444", "rgba(239, 68, 68, 0.16)", 0
        playbook_action = "🛑 HOLD CASH (0% Deployed)"
        playbook_note = "Forced selling and macro stress are still active. Keep powder dry until VIX/VIX3M snaps under 1.00 and Gold decouples."
    elif total_score < 55:
        regime, color, bg, alloc_pct = "DEAD-CAT BOUNCE ZONE", "#f59e0b", "rgba(245, 158, 11, 0.16)", 15
        playbook_action = "🟡 SCOUT ONLY (10%–15% Starter Tranche)"
        playbook_note = "Short-term snapbacks are likely, but macro blockers remain. Only take small starter nibbles on red days when QQQ reclaims VWAP."
    elif total_score < 75:
        regime, color, bg, alloc_pct = "EARLY ACCUMULATION", "#38bdf8", "rgba(56, 189, 248, 0.16)", 40
        playbook_action = "🔵 SCALE IN (30%–50% Tranche)"
        playbook_note = "Macro pressure is easing and price structure is stabilizing. Deploy capital in staggered tranches on pullbacks rather than chasing green candles."
    else:
        regime, color, bg, alloc_pct = "HIGH-CONVICTION REBOUND", "#10b981", "rgba(16, 185, 129, 0.16)", 100
        playbook_action = "🟢 AGGRESSIVE DEPLOY (Remaining Cash)"
        playbook_note = "Yields, volatility term structure, breadth, and institutional support are aligned for a sustained multi-month recovery."

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
        "alloc_pct": alloc_pct,
        "playbook_action": playbook_action,
        "playbook_note": playbook_note,
        "diagnostics": diagnostics,
    }


# -----------------------------------------------------------------------------
# LIVE 24H FUTURES & INTRADAY TAPE (AUTO-REFRESHED EVERY 60s)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=55, show_spinner=False)
def fetch_live_intraday_tape() -> dict:
    tape_tickers = ["NQ=F", "ES=F", "GC=F", "BTC-USD", "QQQ", "SMH", "HYG", "IEF"]
    raw_15m = yf.download(
        tape_tickers,
        period="5d",
        interval="15m",
        prepost=True,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    def tape_quote(t: str) -> tuple[float, float]:
        s = extract_field_series(raw_15m, "Close", t)
        if len(s) < 2:
            return 0.0, 0.0
        last = float(s.iloc[-1])
        prev = float(s.iloc[max(0, len(s) - 96)])
        chg_pct = (last / prev - 1.0) * 100.0 if prev > 0 else 0.0
        return round(last, 2), round(chg_pct, 2)

    nq_p, nq_chg = tape_quote("NQ=F")
    es_p, es_chg = tape_quote("ES=F")
    gc_p, gc_chg = tape_quote("GC=F")
    btc_p, btc_chg = tape_quote("BTC-USD")

    _, qqq_vwap_dist, qqq_wick = compute_session_vwap_and_wick(raw_15m, "QQQ")
    _, smh_vwap_dist, smh_wick = compute_session_vwap_and_wick(raw_15m, "SMH")

    hyg = extract_field_series(raw_15m, "Close", "HYG")
    ief = extract_field_series(raw_15m, "Close", "IEF")
    cred_df = pd.concat([hyg.rename("hyg"), ief.rename("ief")], axis=1).dropna()
    if len(cred_df) >= 20:
        ratio = cred_df["hyg"] / cred_df["ief"]
        hyg_ief_chg = round((float(ratio.iloc[-1]) / float(ratio.iloc[max(0, len(ratio) - 32)]) - 1.0) * 100.0, 2)
    else:
        hyg_ief_chg = 0.0

    return {
        "synced_at": get_sydney_time_str(),
        "nq_p": nq_p,
        "nq_chg": nq_chg,
        "es_p": es_p,
        "es_chg": es_chg,
        "gc_p": gc_p,
        "gc_chg": gc_chg,
        "btc_p": btc_p,
        "btc_chg": btc_chg,
        "qqq_vwap_dist": qqq_vwap_dist,
        "qqq_wick": qqq_wick,
        "smh_vwap_dist": smh_vwap_dist,
        "smh_wick": smh_wick,
        "hyg_ief_chg": hyg_ief_chg,
    }


@st.fragment(run_every=60)
def render_live_24h_tape():
    tape = fetch_live_intraday_tape()
    st.markdown('<div class="tape-box">', unsafe_allow_html=True)
    st.caption(f"🟢 LIVE 24H FUTURES & INTRADAY PULSE (Auto-updates every 60s • {tape['synced_at']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nasdaq Fut (NQ)", f"{tape['nq_p']:,.0f}", f"{tape['nq_chg']:+.2f}%")
    c2.metric("S&P Fut (ES)", f"{tape['es_p']:,.0f}", f"{tape['es_chg']:+.2f}%")
    c3.metric("Gold Fut (GC)", f"${tape['gc_p']:,.0f}", f"{tape['gc_chg']:+.2f}%")
    c4.metric("Bitcoin (24/7)", f"${tape['btc_p']:,.0f}", f"{tape['btc_chg']:+.2f}%")

    i1, i2, i3 = st.columns(3)
    i1.metric("QQQ Intraday VWAP", f"{tape['qqq_vwap_dist']:+.2f}%", f"{'Above' if tape['qqq_vwap_dist'] >= 0 else 'Below'} • Wick {tape['qqq_wick']:.0f}%")
    i2.metric("SMH Intraday VWAP", f"{tape['smh_vwap_dist']:+.2f}%", f"{'Above' if tape['smh_vwap_dist'] >= 0 else 'Below'} • Wick {tape['smh_wick']:.0f}%")
    i3.metric("Live Credit (HYG/IEF)", f"{tape['hyg_ief_chg']:+.2f}%", "Tightening" if tape["hyg_ief_chg"] >= 0 else "Widening")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# LIVE MACRO DATA FETCH + 30-DAY MULTIMETER HISTORY
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_market_data() -> dict:
    tickers = ["^TNX", "^VIX", "^VIX3M", "QQQ", "SMH", "RSP", "GC=F", "DX-Y.NYB", "BTC-USD"]
    raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)

    tnx = extract_field_series(raw, "Close", "^TNX")
    if len(tnx) and float(tnx.iloc[-1]) > 20:
        tnx = tnx / 10.0
    vix = extract_field_series(raw, "Close", "^VIX")
    vix3m = extract_field_series(raw, "Close", "^VIX3M")
    qqq = extract_field_series(raw, "Close", "QQQ")
    smh = extract_field_series(raw, "Close", "SMH")
    qqq_vol = extract_field_series(raw, "Volume", "QQQ")
    smh_vol = extract_field_series(raw, "Volume", "SMH")
    rsp = extract_field_series(raw, "Close", "RSP")
    gold = extract_field_series(raw, "Close", "GC=F")
    dxy = extract_field_series(raw, "Close", "DX-Y.NYB")
    btc = extract_field_series(raw, "Close", "BTC-USD")

    yf_last_date = qqq.index[-1].strftime("%Y-%m-%d") if len(qqq) else "N/A"

    # FRED HY Credit Spreads
    hy_series = pd.Series(dtype=float)
    hy_spread, hy_5d_chg, fred_ok, fred_last_date = 3.50, 0.0, False, "N/A"
    try:
        fred_df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2")
        fred_df.columns = ["date", "val"]
        fred_df["date"] = pd.to_datetime(fred_df["date"], errors="coerce")
        fred_df["val"] = pd.to_numeric(fred_df["val"], errors="coerce")
        fred_df = fred_df.dropna(subset=["date", "val"]).set_index("date")
        hy_series = fred_df["val"]
        if len(hy_series) >= 6:
            hy_spread = float(hy_series.iloc[-1])
            hy_5d_chg = float(hy_spread - hy_series.iloc[-6])
            fred_last_date = hy_series.index[-1].strftime("%Y-%m-%d")
            fred_ok = True
    except Exception:
        fred_ok = False

    def snapshot_as_of(dt):
        t_s = tnx.loc[:dt]
        v_s = vix.loc[:dt]
        v3_s = vix3m.loc[:dt]
        q_s = qqq.loc[:dt]
        sm_s = smh.loc[:dt]
        qv_s = qqq_vol.loc[:dt]
        smv_s = smh_vol.loc[:dt]
        r_s = rsp.loc[:dt]
        g_s = gold.loc[:dt]
        d_s = dxy.loc[:dt]
        b_s = btc.loc[:dt]
        h_s = hy_series.loc[:dt] if len(hy_series) else pd.Series(dtype=float)

        y_val = float(t_s.iloc[-1]) if len(t_s) else 4.50
        v_val = float(v_s.iloc[-1]) if len(v_s) else 20.0
        v3_val = float(v3_s.iloc[-1]) if len(v3_s) else v_val * 1.05
        vr = round(v_val / v3_val, 2) if v3_val > 0 else 0.95
        if len(v_s) >= 5:
            v5max = float(v_s.iloc[-5:].max())
            v3chg = float(v_s.iloc[-1] - v_s.iloc[-3])
            if (v_val >= v5max * 0.97 and v3chg > 0.5) or vr >= 1.08:
                vs = "Spiking"
            elif v_val < v5max * 0.92 and v3chg < -0.5 and vr < 1.02:
                vs = "Rolling Over"
            else:
                vs = "Peaking / Stalling"
        else:
            vs = "Peaking / Stalling"

        qr, qr3 = compute_rsi_metrics(q_s)
        sr, sr3 = compute_rsi_metrics(sm_s)
        trsi = round((qr + sr) / 2.0, 1)
        r3 = round((qr3 + sr3) / 2.0, 1)
        rvr = round((compute_red_vol_ratio(q_s, qv_s) + compute_red_vol_ratio(sm_s, smv_s)) / 2.0, 2)
        ms = "Curling Up" if r3 >= 1.5 else ("Diving (Heavy Vol)" if (r3 <= -2.0 and rvr >= 1.10) else "Stalling / Drying Up")

        qm, qd = ma200_dist_and_dd(q_s)
        sm, sd = ma200_dist_and_dd(sm_s)
        am = round((qm + sm) / 2.0, 1)

        if len(r_s) >= 6 and len(q_s) >= 6:
            r5 = (float(r_s.iloc[-1]) / float(r_s.iloc[-6]) - 1.0) * 100.0
            q5 = (float(q_s.iloc[-1]) / float(q_s.iloc[-6]) - 1.0) * 100.0
            bs = round(r5 - q5, 2)
        else:
            r5, q5, bs = 0.0, 0.0, 0.0

        al = pd.concat([g_s.pct_change(), q_s.pct_change()], axis=1).dropna()
        gc = float(al.iloc[-10:, 0].corr(al.iloc[-10:, 1])) if len(al) >= 10 else 0.0
        if np.isnan(gc):
            gc = 0.0

        dv = float(d_s.iloc[-1]) if len(d_s) else 103.0
        hv = float(h_s.iloc[-1]) if len(h_s) else 3.50
        if len(b_s) >= 6:
            b5h = float(b_s.iloc[-6:].max())
            bdd = max(0.0, (b5h - float(b_s.iloc[-1])) / b5h * 100.0)
        else:
            bdd = 0.0

        return {
            "y": y_val, "v": v_val, "vr": vr, "vs": vs, "qr": qr, "sr": sr, "trsi": trsi,
            "r3": r3, "rvr": rvr, "ms": ms, "am": am, "qd": qd, "sd": sd, "r5": r5,
            "q5": q5, "bs": bs, "gc": gc, "dv": dv, "hv": hv, "bdd": bdd,
        }

    # Build 30-day historical multimeter trend
    history_dates, history_scores = [], []
    if len(qqq) >= 30:
        for dt in qqq.index[-30:]:
            snap = snapshot_as_of(dt)
            sc = evaluate_rebound(
                snap["y"], snap["v"], snap["vr"], snap["vs"], snap["gc"], snap["hv"],
                snap["am"], snap["bs"], snap["trsi"], snap["ms"], snap["dv"], snap["bdd"]
            )["score"]
            history_dates.append(dt.strftime("%b %d"))
            history_scores.append(sc)

    cur = snapshot_as_of(qqq.index[-1]) if len(qqq) else snapshot_as_of(pd.Timestamp.today())
    yield_5d_chg = float(cur["y"] - tnx.iloc[-6]) if len(tnx) >= 6 else 0.0
    dxy_5d_chg = float(cur["dv"] - dxy.iloc[-6]) if len(dxy) >= 6 else 0.0

    return {
        "timestamp": get_sydney_time_str(),
        "yf_last_date": yf_last_date,
        "fred_last_date": fred_last_date,
        "yield_10y": round(cur["y"], 2),
        "yield_5d_chg": round(yield_5d_chg, 2),
        "vix": round(cur["v"], 2),
        "vix_ratio": cur["vr"],
        "vix_state": cur["vs"],
        "qqq_rsi": round(cur["qr"], 1),
        "smh_rsi": round(cur["sr"], 1),
        "tech_rsi": cur["trsi"],
        "rsi_3d_chg": cur["r3"],
        "red_vol_ratio": cur["rvr"],
        "mom_state": cur["ms"],
        "ma200_dist": cur["am"],
        "qqq_dd": round(cur["qd"], 1),
        "smh_dd": round(cur["sd"], 1),
        "avg_dd_52w": round((cur["qd"] + cur["sd"]) / 2.0, 1),
        "qqq_5d": round(cur["q5"], 2),
        "breadth_spread": cur["bs"],
        "rsp_5d": round(cur["r5"], 1),
        "gold_corr": round(cur["gc"], 2),
        "dxy": round(cur["dv"], 2),
        "dxy_5d_chg": round(dxy_5d_chg, 2),
        "hy_spread": round(hy_spread, 2),
        "hy_5d_chg": round(hy_5d_chg, 2),
        "fred_ok": fred_ok,
        "btc_dd_pct": round(cur["bdd"], 1),
        "history_dates": history_dates,
        "history_scores": history_scores,
    }


# -----------------------------------------------------------------------------
# EARNINGS CALENDAR (CACHED 12H) + WATCHLIST SCANNER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=43200, show_spinner=False)
def fetch_earnings_dates(tickers_tuple: tuple) -> dict:
    """Return {ticker: 'YYYY-MM-DD'} for upcoming earnings (cached 12 hours)."""
    out = {}
    etfs = {"QQQ", "SMH", "SPY", "VOO", "SOXX", "IVV", "RSP"}
    for t in tickers_tuple:
        if t in etfs:
            continue
        try:
            cal = yf.Ticker(t).calendar
            edate = None
            if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
                edate = cal["Earnings Date"][0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
                edate = cal.loc["Earnings Date"].iloc[0]
            if edate is not None:
                out[t] = pd.to_datetime(edate).strftime("%Y-%m-%d")
        except Exception:
            pass
    return out


@st.cache_data(ttl=120, show_spinner=False)
def scan_watchlist(tickers_tuple: tuple, qqq_5d_return: float, earnings_map: dict) -> pd.DataFrame:
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()

    raw_daily = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)
    raw_ext = yf.download(tickers, period="2d", interval="15m", prepost=True, auto_adjust=True, progress=False, threads=True)

    rows = []
    index_etfs = {"QQQ", "SMH", "SPY", "VOO", "SOXX", "IVV", "RSP"}
    today = datetime.date.today()

    for t in tickers:
        c = extract_field_series(raw_daily, "Close", t)
        h = extract_field_series(raw_daily, "High", t)
        l = extract_field_series(raw_daily, "Low", t)
        if len(c) < 20:
            continue

        reg_close = float(c.iloc[-1])
        ext_s = extract_field_series(raw_ext, "Close", t)
        ext_price = float(ext_s.iloc[-1]) if len(ext_s) else reg_close
        ext_gap_pct = (ext_price / reg_close - 1.0) * 100.0 if reg_close > 0 else 0.0

        ret_5d = (reg_close / float(c.iloc[-6]) - 1.0) * 100.0 if len(c) >= 6 else 0.0
        rs_vs_qqq = ret_5d - qqq_5d_return
        rsi, rsi_3d = compute_rsi_metrics(c)
        ma200_d, _ = ma200_dist_and_dd(c)
        ema50_d = ema_dist(c, 50)
        atr_val, atr_pct = compute_atr(h, l, c, 14)

        # Earnings risk badge if within 7 days
        e_str = earnings_map.get(t, "")
        e_flag = ""
        if e_str:
            try:
                days_to_e = (datetime.date.fromisoformat(e_str) - today).days
                if 0 <= days_to_e <= 7:
                    e_flag = f"⚠️ {days_to_e}d"
            except Exception:
                pass

        mom_tag = "↗ Curling" if rsi_3d >= 1.5 else ("↘ Diving" if rsi_3d <= -2.0 else "→ Flat")

        if rs_vs_qqq >= 1.5 and ma200_d >= -2.0:
            setup = "RS Leader"
        elif rsi <= 36.0 and ma200_d >= -6.0:
            setup = "Coiled Oversold"
        elif -3.5 <= ma200_d <= 3.5:
            setup = "At 200D Support"
        elif ma200_d < -8.0:
            setup = "Below 200D (Weak)"
        else:
            setup = "Neutral"

        rank_score = 50.0 + clamp(rs_vs_qqq * 3.0, -15.0, 20.0)
        rank_score += 12.0 if 28 <= rsi <= 42 else (6.0 if rsi < 28 else (-10.0 if rsi > 65 else 0.0))
        rank_score += 10.0 if (rsi <= 48 and rsi_3d >= 1.5) else (-10.0 if rsi_3d <= -2.0 else 0.0)
        rank_score += 12.0 if -3.0 <= ma200_d <= 6.0 else (-15.0 if ma200_d < -8.0 else 0.0)

        rows.append(
            {
                "Ticker": t,
                "Close": round(reg_close, 2),
                "Ext. Price": round(ext_price, 2),
                "Ext. Gap": round(ext_gap_pct, 2),
                "RSI": round(rsi, 1),
                "Momentum": mom_tag,
                "vs 50D EMA": round(ema50_d, 1),
                "vs 200D MA": round(ma200_d, 1),
                "ATR $": atr_val,
                "ATR %": atr_pct,
                "Earnings": e_flag,
                "Setup": setup,
                "IsETF": t in index_etfs,
                "RankScore": round(clamp(rank_score, 0, 100), 1),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("RankScore", ascending=False).reset_index(drop=True)


def evaluate_ext_reading(row: pd.Series, macro_score: float, nq_chg: float) -> str:
    gap = float(row["Ext. Gap"])
    rsi = float(row["RSI"])
    rel_gap = gap - nq_chg
    if abs(gap) < 0.4:
        return "— Quiet"
    if abs(rel_gap) >= 2.5:
        return "⚡ Unusual Surge (News)" if gap > 0 else "🚨 Unusual Drop (News)"
    if gap >= 2.0 and macro_score < 55:
        return "⚠️ Don't Chase Gap"
    if gap <= -2.0 and macro_score >= 55:
        return "🎯 Overnight Flush"
    if gap >= 1.0 and rsi < 35:
        return "↗ Relief Bounce"
    if gap <= -1.0 and rsi > 65:
        return "↘ Cooling Off"
    return "≈ Tracking Macro"


def assign_action_signal(row: pd.Series, macro_score: float) -> str:
    if row["Earnings"]:
        return "⚠️ Earnings Risk"
    setup, is_etf, diving = row["Setup"], row["IsETF"], row["Momentum"] == "↘ Diving"
    if macro_score < 35:
        return "🛑 Wait (Macro Red)"
    elif macro_score < 55:
        return "🟡 Scout Starter" if (is_etf or setup == "RS Leader") and not diving else "⏳ Watch Only"
    elif macro_score < 75:
        if diving:
            return "⏳ Wait for Curl"
        return "🔵 Scale-In" if (setup in ("RS Leader", "Coiled Oversold", "At 200D Support") or is_etf) else "⏳ Watch Only"
    else:
        if diving:
            return "🟡 Wait for Curl"
        return "🟢 Buy / Deploy" if (setup in ("RS Leader", "Coiled Oversold", "At 200D Support") or is_etf) else "🔵 Accumulate"


# -----------------------------------------------------------------------------
# REAL-TIME NEWS ENGINE (MULTI-SOURCE + STRICT NEWEST-FIRST SORTING)
# -----------------------------------------------------------------------------
def tag_headline(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ("tariff", "trump", "china", "trade war", "war", "oil", "geopolitic", "sanction", "export")):
        return "🌍 Tariffs/Geo"
    if any(k in t for k in ("earnings", "revenue", "eps", "guidance", "quarter", "beats", "misses")):
        return "📊 Earnings"
    if any(k in t for k in ("fed", "powell", "cpi", "ppi", "inflation", "yield", "treasury", "rate cut", "fomc", "payroll")):
        return "🏛️ Fed/Rates"
    if any(k in t for k in ("upgrade", "downgrade", "price target", "analyst", "rating")):
        return "📈 Analyst"
    if any(k in t for k in ("ai", "chip", "semiconductor", "gpu", "nvidia", "data center", "hbm", "foundry")):
        return "🤖 AI/Semis"
    return "📰 Market"


def format_age(dt: datetime.datetime) -> str:
    """Convert UTC timestamp into a clean relative age like '18m ago' or '3h ago'."""
    now = datetime.datetime.now(datetime.timezone.utc)
    diff = max(0, int((now - dt).total_seconds()))
    if diff < 3600:
        return f"{max(1, diff // 60)}m ago"
    if diff < 86400:
        return f"{diff // 3600}h ago"
    return f"{diff // 86400}d ago"


def parse_rss_feed(url: str, default_source: str, limit: int = 15) -> list[dict]:
    items = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or default_source).strip()
            try:
                dt = parsedate_to_datetime(pub_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                else:
                    dt = dt.astimezone(datetime.timezone.utc)
            except Exception:
                dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
            if title:
                items.append({
                    "tag": tag_headline(title),
                    "title": title,
                    "source": source,
                    "dt": dt,
                    "time": format_age(dt),
                    "url": link,
                })
    except Exception:
        pass
    return items


@st.cache_data(ttl=120, show_spinner=False)
def fetch_breaking_macro_news(max_items: int = 10) -> list[dict]:
    """Merge breaking 12h Google News (tariffs, Fed, tech/Nasdaq) + CNBC Markets, sorted newest-first."""
    g_query = "(Trump OR tariff OR Nasdaq OR \"S&P 500\" OR \"Treasury yield\" OR Fed OR Nvidia OR semiconductors) (plunge OR drop OR surge OR rally OR selloff OR stocks OR market) when:12h"
    g_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(g_query)}&hl=en-US&gl=US&ceid=US:en"
    cnbc_url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"

    combined = parse_rss_feed(g_url, "Google News", limit=18) + parse_rss_feed(cnbc_url, "CNBC", limit=12)

    # Deduplicate by normalized title and sort strictly newest-first
    seen, unique = set(), []
    for it in sorted(combined, key=lambda x: x["dt"], reverse=True):
        key = it["title"].lower()[:55]
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique[:max_items]


@st.cache_data(ttl=120, show_spinner=False)
def fetch_ticker_news(ticker: str, max_items: int = 6) -> list[dict]:
    """Combine yfinance ticker news + 24h Google News for the ticker, sorted newest-first."""
    items = []
    try:
        for n in (yf.Ticker(ticker).news or [])[:max_items]:
            c = n.get("content", n) if isinstance(n, dict) else {}
            title = (c.get("title") or "").strip()
            url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link") or ""
            pub_raw = str(c.get("pubDate") or "")
            try:
                dt = datetime.datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)
            if title:
                items.append({
                    "tag": tag_headline(title),
                    "title": title,
                    "source": (c.get("provider") or {}).get("displayName") or c.get("publisher") or "Yahoo Finance",
                    "dt": dt,
                    "time": format_age(dt),
                    "url": url,
                })
    except Exception:
        pass

    g_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(f'{ticker} stock when:1d')}&hl=en-US&gl=US&ceid=US:en"
    items += parse_rss_feed(g_url, "Google News", limit=6)

    seen, unique = set(), []
    for it in sorted(items, key=lambda x: x["dt"], reverse=True):
        key = it["title"].lower()[:55]
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique[:max_items]


def render_news_list(items: list[dict], empty_msg: str = "No recent headlines found."):
    if not items:
        st.caption(empty_msg)
        return
    for it in items:
        link_md = f"[{it['title']}]({it['url']})" if it["url"] else it["title"]
        st.markdown(
            f"**{it['tag']}** — {link_md}  \n"
            f"<span class='small-muted'>{it['source']} • <b>{it['time']}</b></span>",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# CHARTS (GAUGE + 30-DAY TREND SPARKLINE)
# -----------------------------------------------------------------------------
def render_multimeter(score: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 44, "color": color, "family": "monospace"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b", "tickvals": [0, 35, 55, 75, 100]},
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
                "threshold": {"line": {"color": "#f8fafc", "width": 3}, "thickness": 0.8, "value": score},
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=15, b=0), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
    return fig


def render_30d_sparkline(dates: list, scores: list, line_color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)",
            hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
        )
    )
    for lvl, col in [(35, "#ef4444"), (55, "#f59e0b"), (75, "#10b981")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=col, opacity=0.45, line_width=1)
    fig.update_layout(
        height=135,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#94a3b8"), nticks=6),
        yaxis=dict(range=[0, 100], showgrid=False, tickvals=[0, 35, 55, 75, 100], tickfont=dict(size=10, color="#94a3b8")),
        showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# MAIN UI — ALWAYS-VISIBLE 3-SECOND TOP VIEW
# -----------------------------------------------------------------------------
st.markdown('<div class="pulse-header">REBOUND PULSE // LIVE 24H TIMING TERMINAL</div>', unsafe_allow_html=True)
st.markdown('<div class="pulse-title">Market Rebound Multimeter</div>', unsafe_allow_html=True)

render_live_24h_tape()

SIM_KEYS = ["sim_mode", "s_yield", "s_vix", "s_vix_ratio", "s_vix_state", "s_rsi", "s_mom_state", "s_ma200", "s_breadth", "s_corr", "s_hy", "s_dxy", "s_btc_dd"]
PRESETS = {
    "Goldilocks Bottom": {"s_yield": 3.95, "s_vix": 24.0, "s_vix_ratio": 0.94, "s_vix_state": "Rolling Over", "s_rsi": 34.0, "s_mom_state": "Curling Up", "s_ma200": 0.5, "s_breadth": 1.2, "s_corr": -0.10, "s_hy": 3.40, "s_dxy": 100.5, "s_btc_dd": 3.0},
    "Dead-Cat Bounce": {"s_yield": 4.90, "s_vix": 28.5, "s_vix_ratio": 1.03, "s_vix_state": "Peaking / Stalling", "s_rsi": 36.0, "s_mom_state": "Stalling / Drying Up", "s_ma200": -4.5, "s_breadth": -1.5, "s_corr": 0.60, "s_hy": 4.10, "s_dxy": 104.25, "s_btc_dd": 8.5},
    "2020-Style Freeze": {"s_yield": 5.20, "s_vix": 48.0, "s_vix_ratio": 1.18, "s_vix_state": "Spiking", "s_rsi": 22.0, "s_mom_state": "Diving (Heavy Vol)", "s_ma200": -12.0, "s_breadth": -3.0, "s_corr": 0.85, "s_hy": 6.50, "s_dxy": 108.5, "s_btc_dd": 22.0},
}


def reset_and_refresh_live():
    fetch_live_intraday_tape.clear()
    fetch_live_market_data.clear()
    scan_watchlist.clear()
    fetch_breaking_macro_news.clear()
    fetch_ticker_news.clear()
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
    st.button("🔄 Reset & Refresh All", on_click=reset_and_refresh_live, use_container_width=True)

with st.spinner("Syncing macro, term structure & 30-day trend..."):
    live = fetch_live_market_data()
    tape_snapshot = fetch_live_intraday_tape()

if st.session_state.pop("just_refreshed", False):
    st.toast(f"All feeds synced at {live['timestamp']}", icon="⚡")

if sim_mode:
    with st.expander("Adjust Indicators & Scenario Presets", expanded=True):
        p1, p2, p3 = st.columns(3)
        p1.button("🟢 Goldilocks", on_click=apply_preset, args=("Goldilocks Bottom",), use_container_width=True)
        p2.button("🟡 Dead-Cat", on_click=apply_preset, args=("Dead-Cat Bounce",), use_container_width=True)
        p3.button("🔴 2020 Freeze", on_click=apply_preset, args=("2020-Style Freeze",), use_container_width=True)

        s_yield = st.slider("10-Year Treasury Yield (%)", 2.50, 6.00, clamp(live["yield_10y"], 2.50, 6.00), 0.05, key="s_yield")
        s_vix = st.slider("VIX Index", 10.0, 65.0, clamp(live["vix"], 10.0, 65.0), 0.5, key="s_vix")
        s_vix_ratio = st.slider("VIX / VIX3M Ratio (>1.0 = Panic)", 0.70, 1.40, clamp(live["vix_ratio"], 0.70, 1.40), 0.01, key="s_vix_ratio")
        s_vix_state = st.segmented_control("VIX Trajectory", ["Spiking", "Peaking / Stalling", "Rolling Over"], default=live["vix_state"], key="s_vix_state") or live["vix_state"]
        s_rsi = st.slider("Tech & Semis 14D RSI", 10.0, 85.0, clamp(live["tech_rsi"], 10.0, 85.0), 0.5, key="s_rsi")
        s_mom_state = st.segmented_control("RSI Momentum & Volume", ["Diving (Heavy Vol)", "Stalling / Drying Up", "Curling Up"], default=live["mom_state"], key="s_mom_state") or live["mom_state"]
        s_ma200 = st.slider("QQQ/SMH vs 200-Day MA (%)", -20.0, 25.0, clamp(live["ma200_dist"], -20.0, 25.0), 0.5, key="s_ma200")
        s_breadth = st.slider("Breadth: RSP vs QQQ 5D (%)", -6.0, 6.0, clamp(live["breadth_spread"], -6.0, 6.0), 0.2, key="s_breadth")
        s_corr = st.slider("Gold / Tech 10D Correlation", -1.0, 1.0, clamp(live["gold_corr"], -1.0, 1.0), 0.05, key="s_corr")
        s_hy = st.slider("HY Credit Spreads — FRED OAS (%)", 2.00, 9.00, clamp(live["hy_spread"], 2.00, 9.00), 0.05, key="s_hy")
        s_dxy = st.slider("US Dollar Index (DXY)", 90.0, 115.0, clamp(live["dxy"], 90.0, 115.0), 0.25, key="s_dxy")
        s_btc_dd = st.slider("BTC 5D Drawdown (%)", 0.0, 35.0, clamp(live["btc_dd_pct"], 0.0, 35.0), 0.5, key="s_btc_dd")
else:
    s_yield, s_vix, s_vix_ratio, s_vix_state = live["yield_10y"], live["vix"], live["vix_ratio"], live["vix_state"]
    s_rsi, s_mom_state, s_ma200, s_breadth = live["tech_rsi"], live["mom_state"], live["ma200_dist"], live["breadth_spread"]
    s_corr, s_hy, s_dxy, s_btc_dd = live["gold_corr"], live["hy_spread"], live["dxy"], live["btc_dd_pct"]

result = evaluate_rebound(s_yield, s_vix, s_vix_ratio, s_vix_state, s_corr, s_hy, s_ma200, s_breadth, s_rsi, s_mom_state, s_dxy, s_btc_dd)

# Gauge + 30-Day Trend Sparkline
st.markdown(
    f'<span class="regime-pill" style="color:{result["color"]}; background:{result["bg"]}; border:1px solid {result["color"]};">'
    f'● {result["regime"]}</span>',
    unsafe_allow_html=True,
)
st.plotly_chart(render_multimeter(result["score"], result["color"]), use_container_width=True, config={"displayModeBar": False})

if live["history_dates"]:
    st.caption("📈 30-Day Multimeter Trend (Dotted lines: 35% Knife • 55% Scale-In • 75% High Conviction)")
    st.plotly_chart(render_30d_sparkline(live["history_dates"], live["history_scores"], result["color"]), use_container_width=True, config={"displayModeBar": False})

# Actionable Playbook Card
st.markdown(
    f"""
    <div class="playbook-card" style="border-left: 4px solid {result['color']};">
        <div class="playbook-label">Tranche Deployment Playbook</div>
        <div class="playbook-action" style="color:{result['color']};">{result['playbook_action']}</div>
        <p class="playbook-note">{result['playbook_note']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pre-load Watchlist Data so it's shared cleanly across Tabs 1, 2, and 4
default_wl = "QQQ, SMH, NVDA, AMD, AVGO, MU, META, PLTR, ORCL, MRVL"
if "wl_text" not in st.session_state:
    st.session_state["wl_text"] = default_wl
tickers_list = tuple(dict.fromkeys([t.strip().upper() for t in st.session_state["wl_text"].split(",") if t.strip()]))
earnings_map = fetch_earnings_dates(tickers_list)
wl_df = scan_watchlist(tickers_list, live["qqq_5d"], earnings_map)
if not wl_df.empty:
    wl_df["Ext. Reading"] = wl_df.apply(lambda r: evaluate_ext_reading(r, result["score"], tape_snapshot["nq_chg"]), axis=1)
    wl_df["Action"] = wl_df.apply(lambda r: assign_action_signal(r, result["score"]), axis=1)

# -----------------------------------------------------------------------------
# CLEAN 4-TAB WORKSPACE
# -----------------------------------------------------------------------------
tab_wl, tab_calc, tab_macro, tab_news = st.tabs(
    ["🎯 Watchlist & Levels", "🧮 Sizing & Limit Calc", "🚨 Macro & Diagnostics", "📰 News & Events"]
)

# TAB 1: WATCHLIST & LEVELS
with tab_wl:
    st.text_input("Watchlist Tickers (comma-separated)", key="wl_text")
    if wl_df.empty:
        st.warning("No valid ticker data returned. Check your ticker symbols.")
    else:
        display_cols = [
            "Ticker", "Close", "Ext. Price", "Ext. Gap", "Ext. Reading",
            "RSI", "Momentum", "vs 50D EMA", "vs 200D MA", "ATR %", "Earnings", "Action"
        ]
        st.dataframe(
            wl_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Close": st.column_config.NumberColumn(format="$%.2f"),
                "Ext. Price": st.column_config.NumberColumn(format="$%.2f"),
                "Ext. Gap": st.column_config.NumberColumn(format="%+.2f%%"),
                "RSI": st.column_config.NumberColumn(format="%.1f"),
                "vs 50D EMA": st.column_config.NumberColumn(format="%+.1f%%"),
                "vs 200D MA": st.column_config.NumberColumn(format="%+.1f%%"),
                "ATR %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption("ATR % = average daily swing over 14 days • ⚠️ Earnings flags reports within 7 days.")

# TAB 2: TRANCHE & LIMIT-ORDER CALCULATOR
with tab_calc:
    st.caption("Automatically sizes your tranche based on the live Multimeter Score and calculates ATR limit-order zones.")
    c_cash, c_tick = st.columns(2)
    with c_cash:
        avail_cash = st.number_input("Available Cash ($ USD)", min_value=500.0, max_value=10_000_000.0, value=10000.0, step=500.0)
    with c_tick:
        calc_options = wl_df["Ticker"].tolist() if not wl_df.empty else ["QQQ"]
        sel_calc = st.selectbox("Target Ticker", calc_options)

    alloc_pct = result["alloc_pct"]
    tranche_usd = avail_cash * (alloc_pct / 100.0)

    if not wl_df.empty and sel_calc in wl_df["Ticker"].values:
        row = wl_df[wl_df["Ticker"] == sel_calc].iloc[0]
        ref_price = float(row["Ext. Price"]) if float(row["Ext. Price"]) > 0 else float(row["Close"])
        atr_d = float(row["ATR $"])
        shares_now = int(tranche_usd // ref_price) if ref_price > 0 else 0

        k1, k2, k3 = st.columns(3)
        k1.metric("Suggested Tranche", f"{alloc_pct}%", f"${tranche_usd:,.0f} of ${avail_cash:,.0f}")
        k2.metric(f"{sel_calc} Ref Price", f"${ref_price:,.2f}", f"14D ATR: ${atr_d:.2f} ({row['ATR %']:.1f}%)")
        k3.metric("Est. Shares (Tranche)", f"{shares_now} sh", "Hold Cash" if alloc_pct == 0 else "Limit Order")

        if row["Earnings"]:
            st.warning(f"⚠️ {sel_calc} reports earnings in {row['Earnings'].replace('⚠️ ', '')}. Consider waiting or reducing size.")

        if alloc_pct == 0:
            st.error("Multimeter is in FALLING KNIFE territory (< 35%). Playbook recommends 0% deployment until macro pressure eases.")
        else:
            st.markdown("#### 🎯 Suggested Staggered Limit-Order Zones (ATR-Based)")
            z1, z2, z3 = st.columns(3)
            z1.metric("Bid 1 (Starter: -0.5 ATR)", f"${max(0.01, ref_price - 0.5 * atr_d):,.2f}", "40% of tranche")
            z2.metric("Bid 2 (Core: -1.0 ATR)", f"${max(0.01, ref_price - 1.0 * atr_d):,.2f}", "40% of tranche")
            z3.metric("Bid 3 (Flush: -1.5 ATR)", f"${max(0.01, ref_price - 1.5 * atr_d):,.2f}", "20% of tranche")

# TAB 3: MACRO & DIAGNOSTICS
with tab_macro:
    t1, t2 = st.columns(2)
    with t1:
        st.metric("Short-Term Snapback (1–10D)", f"{result['st_score']:.0f}%")
        st.progress(min(100, int(result["st_score"])))
    with t2:
        st.metric("Sustained Recovery (3–6M)", f"{result['lt_score']:.0f}%")
        st.progress(min(100, int(result["lt_score"])))

    st.subheader("🚨 Bottlenecks Holding Up the Rebound")
    blockers = [d for d in result["diagnostics"] if d["status"] != "GREEN LIGHT"]
    greens = [d for d in result["diagnostics"] if d["status"] == "GREEN LIGHT"]

    if not blockers:
        st.success("All 9 macro, term-structure, breadth, and exhaustion indicators are aligned.")
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
    st.subheader("📡 Live Market Telemetry")
    m1, m2, m3 = st.columns(3)
    m1.metric("10Y Yield (^TNX)", f"{s_yield:.2f}%", f"{live['yield_5d_chg']:+.2f}% (5D)", delta_color="inverse")
    m2.metric("VIX / VIX3M", f"{s_vix:.1f}", f"Ratio {s_vix_ratio:.2f} • {s_vix_state}")
    m3.metric("200D MA Dist", f"{s_ma200:+.1f}%", f"52W DD: {live['avg_dd_52w']:.1f}%")

    m4, m5, m6 = st.columns(3)
    m4.metric("Breadth (RSP-QQQ)", f"{s_breadth:+.1f}%", f"RSP 5D: {live['rsp_5d']:+.1f}%")
    m5.metric("RSI & Exhaustion", f"{s_rsi:.1f}", f"{s_mom_state} ({live['rsi_3d_chg']:+.1f} 3D)")
    m6.metric("Gold/Tech Corr", f"{s_corr:+.2f}", "Decoupled" if s_corr <= 0.15 else "Correlated")

    m7, m8, m9 = st.columns(3)
    m7.metric("HY Spread (OAS)", f"{s_hy:.2f}%", f"{live['hy_5d_chg']:+.2f}% (5D)", delta_color="inverse")
    m8.metric("Dollar (DXY)", f"{s_dxy:.2f}", f"{live['dxy_5d_chg']:+.2f} (5D)", delta_color="inverse")
    m9.metric("BTC 5D Drawdown", f"-{s_btc_dd:.1f}%", "Cooled" if s_btc_dd <= 4.0 else "Flushing")

# TAB 4: NEWS & EVENTS
with tab_news:
    n_macro, n_movers, n_tick, n_cal = st.tabs(["🏛️ Macro", "⚡ Unusual Movers", "🎯 Ticker", "📅 Earnings Radar"])

    with n_macro:
        render_news_list(
            fetch_breaking_macro_news(max_items=10),
            "Could not load breaking macro headlines right now.",
        )

    with n_movers:
        if not wl_df.empty and "Ext. Reading" in wl_df.columns:
            unusual = wl_df[wl_df["Ext. Reading"].str.contains("Unusual|Flush|Chase", na=False)]
            if unusual.empty:
                st.info("No unusual extended-hours movers detected in your watchlist right now.")
            else:
                for _, row in unusual.head(3).iterrows():
                    st.markdown(f"#### {row['Ticker']} ({row['Ext. Gap']:+.2f}% Ext.) — {row['Ext. Reading']}")
                    render_news_list(fetch_ticker_news(row["Ticker"], max_items=3))
        else:
            st.caption("Watchlist data unavailable.")

    with n_tick:
        sel_news_ticker = st.selectbox("Select Ticker", list(tickers_list) if tickers_list else ["QQQ", "SMH", "NVDA"], label_visibility="collapsed")
        render_news_list(fetch_ticker_news(sel_news_ticker, max_items=6), f"No recent headlines for {sel_news_ticker}.")

    with n_cal:
        st.caption("Upcoming earnings dates for your watchlist (cached 12h).")
        if earnings_map:
            cal_rows = [{"Ticker": k, "Next Earnings": v} for k, v in sorted(earnings_map.items(), key=lambda x: x[1])]
            st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming earnings dates returned for your current watchlist.")

# FOOTER
st.divider()
fred_note = f"FRED ({live['fred_last_date']})" if live["fred_ok"] else "FRED fallback"
st.markdown(
    f'<div class="small-muted">'
    f'Last full sync: <b>{live["timestamp"]}</b> • Red-day Vol vs 20D Avg: {live["red_vol_ratio"]:.2f}x<br>'
    f'Latest daily candles: Yahoo Finance ({live["yf_last_date"]}) • {fred_note} • Top tape auto-refreshes every 60s'
    f'</div>',
    unsafe_allow_html=True,
)
