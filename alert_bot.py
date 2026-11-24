import argparse
import datetime
import json
import os
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


def get_sydney_time_str() -> str:
    try:
        tz = ZoneInfo("Australia/Sydney")
    except Exception:
        tz = datetime.timezone(datetime.timedelta(hours=11), name="AEDT")
    return datetime.datetime.now(tz).strftime("%d %b %Y, %I:%M %p %Z")


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


def ma200_dist(s: pd.Series) -> float:
    if len(s) < 30:
        return 0.0
    window = min(200, len(s))
    ma200 = float(s.rolling(window).mean().iloc[-1])
    return (float(s.iloc[-1]) - ma200) / ma200 * 100.0


def fetch_and_score() -> dict:
    tickers = ["^TNX", "^VIX", "^VIX3M", "QQQ", "SMH", "RSP", "GC=F", "DX-Y.NYB", "BTC-USD"]
    raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)

    tnx = extract_field_series(raw, "Close", "^TNX")
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

    if len(tnx) and float(tnx.iloc[-1]) > 20:
        tnx = tnx / 10.0
    yield_10y = round(float(tnx.iloc[-1]), 2) if len(tnx) else 4.50

    vix_val = round(float(vix.iloc[-1]), 2) if len(vix) else 20.0
    vix3m_val = float(vix3m.iloc[-1]) if len(vix3m) else vix_val * 1.05
    vix_ratio = round(vix_val / vix3m_val, 2) if vix3m_val > 0 else 0.95
    if len(vix) >= 5:
        vix_5d_max = float(vix.iloc[-5:].max())
        vix_3d_chg = float(vix.iloc[-1] - vix.iloc[-3])
        if (vix_val >= vix_5d_max * 0.97 and vix_3d_chg > 0.5) or vix_ratio >= 1.08:
            vix_state = "Spiking"
        elif vix_val < vix_5d_max * 0.92 and vix_3d_chg < -0.5 and vix_ratio < 1.02:
            vix_state = "Rolling Over"
        else:
            vix_state = "Peaking / Stalling"
    else:
        vix_state = "Peaking / Stalling"

    qqq_rsi, qqq_rsi_3d = compute_rsi_metrics(qqq)
    smh_rsi, smh_rsi_3d = compute_rsi_metrics(smh)
    tech_rsi = round((qqq_rsi + smh_rsi) / 2.0, 1)
    rsi_3d_chg = round((qqq_rsi_3d + smh_rsi_3d) / 2.0, 1)
    red_vol_ratio = (compute_red_vol_ratio(qqq, qqq_vol) + compute_red_vol_ratio(smh, smh_vol)) / 2.0

    if rsi_3d_chg >= 1.5:
        mom_state = "Curling Up"
    elif rsi_3d_chg <= -2.0 and red_vol_ratio >= 1.10:
        mom_state = "Diving (Heavy Vol)"
    else:
        mom_state = "Stalling / Drying Up"

    avg_ma200 = round((ma200_dist(qqq) + ma200_dist(smh)) / 2.0, 1)

    if len(rsp) >= 6 and len(qqq) >= 6:
        breadth = round(((float(rsp.iloc[-1]) / float(rsp.iloc[-6])) - (float(qqq.iloc[-1]) / float(qqq.iloc[-6]))) * 100.0, 2)
    else:
        breadth = 0.0

    aligned = pd.concat([gold.pct_change(), qqq.pct_change()], axis=1).dropna()
    gold_corr = round(float(aligned.iloc[-10:, 0].corr(aligned.iloc[-10:, 1])), 2) if len(aligned) >= 10 else 0.0
    if np.isnan(gold_corr):
        gold_corr = 0.0

    dxy_val = round(float(dxy.iloc[-1]), 2) if len(dxy) else 103.0

    hy_spread = 3.50
    try:
        fred_df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2")
        hy_s = pd.to_numeric(fred_df.iloc[:, 1], errors="coerce").dropna()
        if len(hy_s):
            hy_spread = round(float(hy_s.iloc[-1]), 2)
    except Exception:
        pass

    if len(btc) >= 6:
        btc_5d_high = float(btc.iloc[-6:].max())
        btc_dd = round(max(0.0, (btc_5d_high - float(btc.iloc[-1])) / btc_5d_high * 100.0), 1)
    else:
        btc_dd = 0.0

    # 9-Indicator Scoring (matches app.py)
    pts_y = 20.0 if yield_10y <= 4.00 else (14.0 + (4.50 - yield_10y) / 0.50 * 6.0 if yield_10y <= 4.50 else (5.0 + (4.90 - yield_10y) / 0.40 * 9.0 if yield_10y <= 4.90 else max(0.0, 5.0 - (yield_10y - 4.90) * 12.0)))
    pts_v = (15.0 if vix_val >= 22 else 12.5) if (vix_state == "Rolling Over" and vix_ratio < 1.00) else ((8.5 if vix_ratio < 1.05 else 6.0) if (vix_state == "Peaking / Stalling" or vix_state == "Rolling Over") else (2.0 if vix_ratio >= 1.05 else 3.5))
    pts_g = 13.0 if gold_corr <= 0.15 else (7.5 if gold_corr <= 0.45 else max(0.0, 3.5 - (gold_corr - 0.45) * 6.0))
    pts_hy = 12.0 if hy_spread <= 3.75 else (6.5 if hy_spread <= 4.75 else max(0.0, 3.0 - (hy_spread - 4.75) * 1.5))
    pts_m = 12.0 if -3.0 <= avg_ma200 <= 5.0 else (8.5 if avg_ma200 > 5.0 else (5.5 if avg_ma200 >= -8.0 else 2.0))
    pts_b = 8.0 if breadth >= 0.5 else (5.5 if breadth >= -1.0 else 2.5)
    pts_r = 2.0 if mom_state == "Diving (Heavy Vol)" else (8.0 if (tech_rsi <= 45 and mom_state == "Curling Up") else (6.0 if tech_rsi <= 45 else (5.5 if tech_rsi <= 58 else 3.5)))
    pts_d = 6.0 if dxy_val <= 101.5 else (3.5 if dxy_val <= 104.5 else 1.0)
    pts_c = 6.0 if btc_dd <= 4.0 else (3.5 if btc_dd <= 10.0 else 1.0)

    score = round(pts_y + pts_v + pts_g + pts_hy + pts_m + pts_b + pts_r + pts_d + pts_c, 1)

    if score < 35:
        regime, color, action = "FALLING KNIFE", 0xEF4444, "🛑 HOLD CASH (0% Deployed)"
    elif score < 55:
        regime, color, action = "DEAD-CAT BOUNCE ZONE", 0xF59E0B, "🟡 SCOUT ONLY (10%–15% Starter)"
    elif score < 75:
        regime, color, action = "EARLY ACCUMULATION", 0x38BDF8, "🔵 SCALE IN (30%–50% Tranche)"
    else:
        regime, color, action = "HIGH-CONVICTION REBOUND", 0x10B981, "🟢 AGGRESSIVE DEPLOY"

    # Major Triggers Check
    triggers = []
    if score >= 55.0:
        triggers.append(f"🚀 **Multimeter crossed {score}%** ({regime})")
    if yield_10y < 4.50:
        triggers.append(f"📉 **10Y Yield cooled to {yield_10y:.2f}%** (< 4.50% pressure line)")
    if vix_state == "Rolling Over" and vix_ratio < 1.00:
        triggers.append(f"🧊 **VIX Rolling Over** ({vix_val:.1f}, VIX/VIX3M = {vix_ratio:.2f})")
    if gold_corr <= 0.15:
        triggers.append(f"🔓 **Gold/Tech Decoupled** (Corr: {gold_corr:+.2f})")
    if tech_rsi <= 45.0 and mom_state == "Curling Up":
        triggers.append(f"↗ **Selling Exhaustion Confirmed** (RSI {tech_rsi:.1f}, Curling Up)")

    return {
        "score": score,
        "regime": regime,
        "color": color,
        "action": action,
        "yield_10y": yield_10y,
        "vix": vix_val,
        "vix_ratio": vix_ratio,
        "vix_state": vix_state,
        "tech_rsi": tech_rsi,
        "mom_state": mom_state,
        "ma200": avg_ma200,
        "gold_corr": gold_corr,
        "hy_spread": hy_spread,
        "dxy": dxy_val,
        "triggers": triggers,
    }


def send_discord(webhook_url: str, data: dict, mode: str) -> None:
    title = (
        f"☀️ Morning Market Brief — {data['score']}% ({data['regime']})"
        if mode == "brief"
        else f"⚡ MAJOR TRIGGER ALERT — {data['score']}% ({data['regime']})"
    )

    trigger_text = "\n".join(data["triggers"]) if data["triggers"] else "No major green-light triggers active yet."

    embed = {
        "title": title,
        "description": f"**Playbook:** {data['action']}\n\n**Active Triggers:**\n{trigger_text}",
        "color": data["color"],
        "fields": [
            {"name": "10Y Yield", "value": f"{data['yield_10y']:.2f}%", "inline": True},
            {"name": "VIX / VIX3M", "value": f"{data['vix']:.1f} ({data['vix_ratio']:.2f})", "inline": True},
            {"name": "Tech RSI", "value": f"{data['tech_rsi']:.1f} ({data['mom_state']})", "inline": True},
            {"name": "200D MA Dist", "value": f"{data['ma200']:+.1f}%", "inline": True},
            {"name": "Gold/Tech Corr", "value": f"{data['gold_corr']:+.2f}", "inline": True},
            {"name": "HY Spread / DXY", "value": f"{data['hy_spread']:.2f}% / {data['dxy']:.1f}", "inline": True},
        ],
        "footer": {"text": f"Rebound Pulse • {get_sydney_time_str()}"},
    }

    payload = json.dumps({"username": "Rebound Pulse", "embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "ReboundPulseBot/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"Discord sent! Status: {resp.status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["brief", "triggers"], default="brief")
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL secret is not set.")

    data = fetch_and_score()
    print(f"Mode={args.mode} | Score={data['score']}% | Triggers={len(data['triggers'])}")

    if args.mode == "brief" or (args.mode == "triggers" and data["triggers"]):
        send_discord(webhook_url, data, args.mode)
    else:
        print("No major triggers fired. Skipping Discord ping.")


if __name__ == "__main__":
    main()
