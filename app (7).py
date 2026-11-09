"""Kishore RSI Scanner - RSI Long + RSI Short only (Binance Spot, Streamlit).
Supertrend is used ONLY for the RSI-Long exit (RSI >= 58 accepted when a bearish
Supertrend line rejects price). No ST Long / ST Short trades."""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- RULES (edit here) ----------------
LONG_IN, LONG_SL, LONG_TP, LONG_TP_ST = 20, 9, 65, 58   # RSI Long
SHORT_IN, SHORT_SL, SHORT_TP = 77, 88, 47               # RSI Short
RSI_LEN, ST_LEN, ST_MULT = 14, 10, 3.0
TFS = ["5m", "15m", "30m", "1h", "4h"]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com",
         "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
SKIP = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "AEUR", "USD1", "USDE", "XUSD", "BFUSD"}
COLS = ["ot", "open", "high", "low", "close", "vol", "ct"]


# ---------------- DATA ----------------
def api(path, params=None):
    last = None
    for h in HOSTS:
        try:
            r = requests.get(h + path, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            last = f"{h} -> {r.status_code}"
        except Exception as e:
            last = f"{h} -> {e}"
    raise RuntimeError(last)


@st.cache_data(ttl=600, show_spinner=False)
def top_symbols(n):
    df = pd.DataFrame(api("/api/v3/ticker/24hr"))
    df = df[df.symbol.str.endswith("USDT")]
    df = df[~df.symbol.str[:-4].isin(SKIP)].copy()
    df["q"] = df.quoteVolume.astype(float)
    return df.sort_values("q", ascending=False).symbol.head(n).tolist()


@st.cache_resource
def _store():
    return {}


def rsi_calc(c, n=RSI_LEN):
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def supertrend(df, n=ST_LEN, m=ST_MULT):
    h, l, c = df.high.values, df.low.values, df.close.values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    hl2 = (h + l) / 2
    ub, lb = hl2 + m * atr, hl2 - m * atr
    fub, flb = ub.copy(), lb.copy()
    d, line = np.ones(len(c)), lb.copy()
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
        line[i] = flb[i] if d[i] == 1 else fub[i]
    return line, d


def prep(df):
    df["rsi"] = rsi_calc(df.close)
    line, d = supertrend(df)
    # bearish Supertrend line rejects price: wick touches line, close stays below it
    df["rej"] = (d == -1) & (df.high.values >= line) & (df.close.values < line)
    df["t"] = pd.to_datetime(df.ct, unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    return df


def fetch_kl(sym, tf, limit, store, ttl):
    key = (sym, tf, limit)
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        raw = api("/api/v3/klines", {"symbol": sym, "interval": tf, "limit": limit})
        df = pd.DataFrame([r[:7] for r in raw], columns=COLS).astype(float)
        df = df[df.ct < time.time() * 1000].reset_index(drop=True)  # closed candles only
        if len(df) < 50:
            return None
        df = prep(df)
    except Exception:
        return None
    store[key] = (time.time(), df)
    return df


def load_many(pairs, limit, ttl):
    store = _store()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda p: fetch_kl(p[0], p[1], limit, store, ttl), pairs))
    return dict(zip(pairs, res))


# ---------------- BACKTEST ----------------
def _trade(sym, tf, side, j, i, res, df):
    cl, r, ts = df.close.values, df.rsi.values, df.t.dt.strftime("%d-%b %H:%M").values
    e, x = cl[j], cl[i]
    pct = (x / e - 1) * 100 if side == "RSI Long" else (e - x) / e * 100
    return {"Coin": sym.replace("USDT", ""), "TF": tf, "Logic": side,
            "Entry Time": ts[j], "Entry": e, "Exit Time": ts[i], "Exit": x,
            "RSI In": r[j], "RSI Out": r[i], "Candles": i - j, "Result": res,
            "PnL %": pct, "_ts": df.ct.values[j]}


def run_bt(df, sym, tf):
    r, rej = df.rsi.values, df.rej.values
    out, pos = [], None
    for i in range(len(df)):
        if np.isnan(r[i]):
            continue
        if pos is None:                       # entry at candle close
            if r[i] <= LONG_IN:
                pos = ("RSI Long", i)
            elif r[i] >= SHORT_IN:
                pos = ("RSI Short", i)
            continue
        side, j = pos
        res = None
        if side == "RSI Long":
            if r[i] <= LONG_SL:
                res = "FAIL"
            elif r[i] >= LONG_TP or (r[i] >= LONG_TP_ST and rej[i]):
                res = "SUCCESS"
        else:
            if r[i] >= SHORT_SL:
                res = "FAIL"
            elif r[i] <= SHORT_TP:
                res = "SUCCESS"
        if res:
            out.append(_trade(sym, tf, side, j, i, res, df))
            pos = None
    if pos:
        out.append(_trade(sym, tf, pos[0], pos[1], len(df) - 1, "OPEN", df))
    return out


def make_trades(data, notional):
    rows = []
    for (s, t), df in data.items():
        if df is not None:
            rows += run_bt(df, s, t)
    if not rows:
        return pd.DataFrame()
    tr = pd.DataFrame(rows)
    tr["PnL ₹"] = notional * tr["PnL %"] / 100
    return tr


def table(v, by):
    rows = []
    for key, g in v.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        w = int((g.Result == "SUCCESS").sum())
        rows.append({**dict(zip(by, key)), "Trades": len(g), "Win": w, "Fail": len(g) - w,
                     "Win %": round(w / len(g) * 100, 1),
                     "Net PnL ₹": round(g["PnL ₹"].sum()),
                     "Total Loss ₹": round(g.loc[g["PnL ₹"] < 0, "PnL ₹"].sum()),
                     "Worst Loss %": round(g["PnL %"].min(), 2)})
    return pd.DataFrame(rows)


def add_loss_rank(v):
    v = v.copy()
    v["Loss #"] = np.nan
    neg = v["PnL ₹"] < 0
    v.loc[neg, "Loss #"] = v.loc[neg, "PnL ₹"].rank(method="min")   # 1 = biggest loss
    return v


def hl(row):
    if row["Loss #"] == 1:
        s = "background-color:rgba(255,0,0,0.6);font-weight:bold"
    elif row["Result"] == "FAIL":
        s = "background-color:rgba(255,0,0,0.22)"
    elif row["Result"] == "SUCCESS":
        s = "background-color:rgba(0,200,0,0.15)"
    else:
        s = ""
    return [s] * len(row)


# ---------------- LIVE SCAN ----------------
def scan_live(data, fresh):
    rows = []
    for (s, t), df in data.items():
        if df is None:
            continue
        r = df.rsi.values
        for age in range(fresh):
            i = len(df) - 1 - age
            side = "🟢 RSI LONG" if r[i] <= LONG_IN else "🔴 RSI SHORT" if r[i] >= SHORT_IN else None
            if side:
                sl, tp = (LONG_SL, LONG_TP) if "LONG" in side else (SHORT_SL, SHORT_TP)
                rows.append({"Age": age, "Signal Time (IST)": df.t.dt.strftime("%d-%b %H:%M").values[i],
                             "Coin": s.replace("USDT", ""), "TF": t, "Signal": side,
                             "Price": df.close.values[i], "RSI": round(r[i], 1),
                             "SL @ RSI": sl, "TP @ RSI": tp,
                             "Status": "FRESH" if age == 0 else f"{age} candle ago"})
                break
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Age", "TF"]).drop(columns="Age").reset_index(drop=True)


# ---------------- UI ----------------
def live_tab():
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Coins (top by volume)", 10, 100, 40, 5, key="ln")
    tfs = c2.multiselect("Timeframes", TFS, default=["5m", "15m", "1h"], key="ltf")
    fresh = c3.slider("Signal age (candles)", 1, 5, 2, key="lf")

    @st.fragment(run_every=60)
    def body():
        try:
            pairs = [(s, t) for s in top_symbols(n) for t in tfs]
        except Exception as e:
            st.error(f"Binance symbols fetch failed: {e}")
            return
        data = load_many(pairs, 300, 50)
        bad = sum(v is None for v in data.values())
        res = scan_live(data, fresh)
        st.caption(f"Auto-refresh 60s • updated {pd.Timestamp.now(tz='Asia/Kolkata'):%H:%M:%S} IST "
                   f"• {len(pairs) - bad}/{len(pairs)} loaded")
        if res.empty:
            st.info("Ippudu RSI Long (≤20) / RSI Short (≥77) signal ledu.")
        else:
            st.dataframe(res, hide_index=True)

    if tfs:
        body()


def backtest_tab():
    c1, c2 = st.columns(2)
    n = c1.slider("Coins (top by volume)", 10, 100, 30, 5, key="bn")
    tfs = c2.multiselect("Timeframes", TFS, default=TFS, key="btf")
    c3, c4 = st.columns(2)
    cap = c3.number_input("Capital ₹", 10000, 10_000_000, 100000, 10000)
    per = c4.slider("Capital per trade %", 10, 100, 50, 10)

    if st.button("▶ Run backtest", type="primary"):
        with st.spinner("Binance data techukuntunnanu..."):
            try:
                pairs = [(s, t) for s in top_symbols(n) for t in tfs]
                data = load_many(pairs, 1000, 300)
            except Exception as e:
                st.error(f"Binance fetch failed: {e}")
                return
        bad = sum(v is None for v in data.values())
        st.session_state["bt"] = (make_trades(data, cap * per / 100), bad, len(pairs), cap)

    if "bt" not in st.session_state:
        st.info("Run backtest nokku. (Each TF lo last 1000 candles use avutayi.)")
        return
    tr, bad, total, cap = st.session_state["bt"]
    if bad:
        st.warning(f"{bad}/{total} coin-TF data load avvaledu (skip chesanu).")
    if tr.empty:
        st.warning("Ee data lo trades levu.")
        return

    f1, f2, f3 = st.columns(3)
    lg = f1.selectbox("Logic", ["All", "RSI Long", "RSI Short"])
    tf = f2.selectbox("Timeframe", ["All"] + TFS)
    mt = f3.slider("Min trades per coin", 1, 10, 3)
    v = tr
    if lg != "All":
        v = v[v.Logic == lg]
    if tf != "All":
        v = v[v.TF == tf]
    n_open = int((v.Result == "OPEN").sum())
    v = v[v.Result != "OPEN"]
    if v.empty:
        st.warning("Ee filter ki closed trades levu.")
        return
    v = add_loss_rank(v)

    w = int((v.Result == "SUCCESS").sum())
    net = v["PnL ₹"].sum()
    m = st.columns(4)
    m[0].metric("Trades", len(v))
    m[1].metric("Win rate", f"{w / len(v) * 100:.1f}%")
    m[2].metric("Net PnL", f"₹{net:,.0f}", f"{net / cap * 100:+.2f}% of capital")
    m[3].metric("Total loss", f"₹{v.loc[v['PnL ₹'] < 0, 'PnL ₹'].sum():,.0f}")
    if n_open:
        st.caption(f"{n_open} trade(s) inka open (stats lo lekkaledu).")

    wt = v.loc[v["PnL ₹"].idxmin()]
    if wt["PnL ₹"] < 0:
        st.error(f"🔻 BIGGEST LOSS: {wt.Coin} • {wt.TF} • {wt.Logic} • {wt['PnL %']:+.2f}% "
                 f"(₹{wt['PnL ₹']:,.0f}) • {wt['Entry Time']} → {wt['Exit Time']} • "
                 f"RSI {wt['RSI In']:.1f} → {wt['RSI Out']:.1f}")

    st.subheader("Timeframe-wise (Long + Short together)")
    tt = table(v, ["TF"])
    tt["o"] = tt.TF.map({t: k for k, t in enumerate(TFS)})
    st.dataframe(tt.sort_values("o").drop(columns="o"), hide_index=True)

    st.subheader("Logic × Timeframe")
    st.dataframe(table(v, ["Logic", "TF"]), hide_index=True)

    st.subheader("Coin-wise")
    ct = table(v, ["Coin"]).sort_values(["Win %", "Trades"], ascending=False)
    q = ct[ct.Trades >= mt]
    if q.empty:
        st.info(f"{mt}+ trades unna coin ledu - 'Min trades' taggu.")
    else:
        b = q.sort_values(["Win %", "Trades"], ascending=False).iloc[0]
        lo = q.sort_values("Total Loss ₹", ascending=False).iloc[0]
        hi = q.sort_values("Total Loss ₹").iloc[0]
        st.success(f"🏆 Highest win rate: {b.Coin} — {b['Win %']}% ({b.Trades} trades)")
        st.info(f"🛡️ Lowest loss: {lo.Coin} — ₹{lo['Total Loss ₹']:,.0f} total loss ({lo.Trades} trades)")
        st.error(f"⚠️ Highest loss: {hi.Coin} — ₹{hi['Total Loss ₹']:,.0f} total loss, "
                 f"worst trade {hi['Worst Loss %']}%")
    st.dataframe(ct, hide_index=True)
    st.bar_chart(ct.set_index("Coin")["Net PnL ₹"])

    st.subheader("Every trade (Loss #1 = biggest loss)")
    sort = st.radio("Sort", ["Biggest loss first", "Latest first"], horizontal=True)
    cols = ["Loss #", "Coin", "TF", "Logic", "Entry Time", "Entry", "Exit Time", "Exit",
            "RSI In", "RSI Out", "Candles", "Result", "PnL %", "PnL ₹"]
    d = v.sort_values("PnL ₹") if sort == "Biggest loss first" else v.sort_values("_ts", ascending=False)
    sty = d[cols].style.apply(hl, axis=1).format(
        {"Entry": "{:.6g}", "Exit": "{:.6g}", "RSI In": "{:.1f}", "RSI Out": "{:.1f}",
         "PnL %": "{:+.2f}", "PnL ₹": "{:+,.0f}", "Loss #": "{:.0f}"}, na_rep="")
    st.dataframe(sty, hide_index=True, height=520)


def main():
    st.set_page_config(page_title="Kishore RSI Scanner", page_icon="📉", layout="wide")
    st.title("📉 Kishore RSI Scanner")
    with st.expander("📖 Logic (RSI Long + RSI Short only)"):
        st.markdown(
            f"- **RSI Long** — RSI ≤ {LONG_IN} → buy at candle close. FAIL: RSI ≤ {LONG_SL}. "
            f"SUCCESS: RSI ≥ {LONG_TP} (or RSI ≥ {LONG_TP_ST} when a bearish Supertrend line rejects price).\n"
            f"- **RSI Short** — RSI ≥ {SHORT_IN} → sell. FAIL: RSI ≥ {SHORT_SL}. SUCCESS: RSI ≤ {SHORT_TP}.\n"
            "- RSI trades have **no price stop** — the RSI level is the stop, so a FAIL can be a big % loss.")
    t1, t2 = st.tabs(["🔴 Live scanner (auto)", "📊 Backtest"])
    with t1:
        live_tab()
    with t2:
        backtest_tab()


if __name__ == "__main__":
    main()
