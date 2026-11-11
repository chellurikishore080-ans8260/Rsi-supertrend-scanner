"""Kishore RSI Scanner - RSI Long + RSI Short only (Binance Spot, Streamlit).
Fixed coin list (Kishore's selected coins). Backtest = 10000 candles per coin per timeframe,
separate backtest block for every coin.
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
BT_CANDLES = 10000          # candles per coin per timeframe in the backtest
LIVE_AGE = 4                # live scanner: show signals up to this many candles old
# ONLY these coins (U, RLUSD, XRP, BTC removed)
COINS = ["GRAM", "ARB", "TAO", "BNB", "ERA", "UNI", "AVAX", "NEAR", "LINK", "SOL",
         "SUI", "CRCLB", "ENA", "INJ", "INIT", "DOGE", "NVDAB", "SPCXB", "ONE"]
SYMS = [c + "USDT" for c in COINS]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com",
         "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
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


def klines(sym, tf, total):
    """Fetch `total` candles, paging backwards 1000 at a time (Binance max per request)."""
    rows, end = [], None
    while len(rows) < total:
        lim = min(1000, total - len(rows))
        p = {"symbol": sym, "interval": tf, "limit": lim}
        if end is not None:
            p["endTime"] = end
        raw = api("/api/v3/klines", p)
        if not raw:
            break
        rows = raw + rows                 # older batch goes in front
        end = raw[0][0] - 1
        if len(raw) < lim:                # coin has no more history
            break
    return rows


def fetch_kl(sym, tf, limit, store, ttl):
    key = (sym, tf, limit)
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        raw = klines(sym, tf, limit)
        df = pd.DataFrame([r[:7] for r in raw], columns=COLS).astype(float)
        df = df.drop_duplicates("ot")
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
    cl, r, ts = df.close.values, df.rsi.values, df.t.dt.strftime("%d-%b-%y %H:%M").values
    e, x = cl[j], cl[i]
    pct = (x / e - 1) * 100 if side == "RSI Long" else (e - x) / e * 100
    return {"Coin": sym.replace("USDT", ""), "TF": tf, "Logic": side,
            "Entry Time": ts[j], "Entry": e, "Exit Time": ts[i], "Exit": x,
            "RSI In": r[j], "RSI Out": r[i], "Candles": i - j, "Result": res,
            "PnL %": pct, "_ts": df.ct.values[j]}


def run_bt(df, sym, tf, price_ok=False):
    r, rej, cl = df.rsi.values, df.rej.values, df.close.values
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
        if res == "SUCCESS" and price_ok:      # strict: price must also be in profit
            if not (cl[i] > cl[j] if side == "RSI Long" else cl[i] < cl[j]):
                res = None
        if res:
            out.append(_trade(sym, tf, side, j, i, res, df))
            pos = None
    if pos:
        out.append(_trade(sym, tf, pos[0], pos[1], len(df) - 1, "OPEN", df))
    return out


def make_trades(data, notional, price_ok=False):
    rows = []
    for (s, t), df in data.items():
        if df is not None:
            rows += run_bt(df, s, t, price_ok)      # separate backtest per coin + TF
    if not rows:
        return pd.DataFrame()
    tr = pd.DataFrame(rows)
    tr["PnL ₹"] = notional * tr["PnL %"] / 100
    tr["Real"] = np.where(tr["PnL ₹"] > 0, "WIN", "LOSS")
    return tr


def table(v, by):
    rows = []
    for key, g in v.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        w = int((g.Result == "SUCCESS").sum())
        rows.append({**dict(zip(by, key)), "Trades": len(g), "Win": w, "Fail": len(g) - w,
                     "Win %": round(w / len(g) * 100, 1),
                     "Real Win %": round((g["PnL ₹"] > 0).sum() / len(g) * 100, 1),
                     "RSI ✓ but Loss": int(((g.Result == "SUCCESS") & (g["PnL ₹"] <= 0)).sum()),
                     "Net PnL ₹": round(g["PnL ₹"].sum()),
                     "Total Loss ₹": round(g.loc[g["PnL ₹"] < 0, "PnL ₹"].sum()),
                     "Worst Loss %": round(g["PnL %"].min(), 2)})
    return pd.DataFrame(rows)


def coin_table(v):
    """One row for EVERY coin in COINS (0 trades if none)."""
    cols = ["Trades", "Win", "Fail", "Win %", "Real Win %", "RSI ✓ but Loss",
            "Net PnL ₹", "Total Loss ₹", "Worst Loss %"]
    t = table(v, ["Coin"]) if not v.empty else pd.DataFrame(columns=["Coin"] + cols)
    t = t.set_index("Coin").reindex(COINS)
    t[cols] = t[cols].fillna(0)
    for c in ["Trades", "Win", "Fail", "RSI ✓ but Loss"]:
        t[c] = t[c].astype(int)
    return t.reset_index()


def add_loss_rank(v):
    v = v.copy()
    v["Loss #"] = np.nan
    neg = v["PnL ₹"] < 0
    v.loc[neg, "Loss #"] = v.loc[neg, "PnL ₹"].rank(method="min")   # 1 = biggest loss
    return v


def hl(row):
    if row.get("Loss #") == 1:
        s = "background-color:rgba(255,0,0,0.6);font-weight:bold"
    elif row["Result"] == "FAIL":
        s = "background-color:rgba(255,0,0,0.22)"
    elif row["Result"] == "SUCCESS" and row["Real"] == "LOSS":
        s = "background-color:rgba(255,165,0,0.3)"      # RSI target hit but price in loss
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
TRADE_COLS = ["Loss #", "TF", "Logic", "Entry Time", "Entry", "Exit Time", "Exit",
              "RSI In", "RSI Out", "Candles", "Result", "Real", "PnL %", "PnL ₹"]
FMT = {"Entry": "{:.6g}", "Exit": "{:.6g}", "RSI In": "{:.1f}", "RSI Out": "{:.1f}",
       "PnL %": "{:+.2f}", "PnL ₹": "{:+,.0f}", "Loss #": "{:.0f}"}


def live_tab():
    tfs = st.multiselect("Timeframes", TFS, default=["5m", "15m", "1h"], key="ltf")
    st.caption("Coins: " + ", ".join(COINS))

    @st.fragment(run_every=60)
    def body():
        pairs = [(s, t) for s in SYMS for t in tfs]
        data = load_many(pairs, 300, 50)
        bad = sum(v is None for v in data.values())
        res = scan_live(data, LIVE_AGE)
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
    candles = c1.number_input("Candles per coin per TF", 1000, 20000, BT_CANDLES, 1000)
    tfs = c2.multiselect("Timeframes", TFS, default=TFS, key="btf")
    c3, c4 = st.columns(2)
    cap = c3.number_input("Capital ₹", 10000, 10_000_000, 100000, 10000)
    per = c4.slider("Capital per trade %", 10, 100, 50, 10)
    strict = st.checkbox("Strict: SUCCESS ki RSI target + price kuda profit lo undali", value=False)

    if st.button("▶ Run backtest", type="primary"):
        pairs = [(s, t) for s in SYMS for t in tfs]
        with st.spinner(f"Binance nundi {len(SYMS)} coins x {len(tfs)} TF x {candles} candles techukuntunnanu "
                        "(konchem time padutundi)..."):
            data = load_many(pairs, int(candles), 600)
        failed = [(s.replace("USDT", ""), t) for (s, t), d in data.items() if d is None]
        counts = {(s.replace("USDT", ""), t): len(d) for (s, t), d in data.items() if d is not None}
        st.session_state["bt"] = (make_trades(data, cap * per / 100, strict), failed, counts, cap)

    if "bt" not in st.session_state:
        st.info(f"Run backtest nokku. Prathi coin ki, prathi TF ki {BT_CANDLES} candles tho separate backtest avutundi.")
        return
    tr, failed, counts, cap = st.session_state["bt"]
    if failed:
        st.warning("Data load avvaledu (skip chesanu): " + ", ".join(f"{c} {t}" for c, t in failed))
    if tr.empty:
        st.warning("Ee data lo trades levu.")
        return

    f1, f2, f3 = st.columns(3)
    lg = f1.selectbox("Logic", ["All", "RSI Long", "RSI Short"])
    tf = f2.selectbox("Timeframe", ["All"] + TFS)
    mt = f3.slider("Min trades per coin", 1, 10, 3)
    vo = tr
    if lg != "All":
        vo = vo[vo.Logic == lg]
    if tf != "All":
        vo = vo[vo.TF == tf]
    n_open = int((vo.Result == "OPEN").sum())
    v = vo[vo.Result != "OPEN"]
    if v.empty:
        st.warning("Ee filter ki closed trades levu.")
        return
    v = add_loss_rank(v)

    w = int((v.Result == "SUCCESS").sum())
    net = v["PnL ₹"].sum()
    real_w = int((v["PnL ₹"] > 0).sum())
    sl = int(((v.Result == "SUCCESS") & (v["PnL ₹"] <= 0)).sum())
    m = st.columns(3)
    m[0].metric("Trades", len(v))
    m[1].metric("Win % (RSI rule)", f"{w / len(v) * 100:.1f}%")
    m[2].metric("Real Win % (PnL>0)", f"{real_w / len(v) * 100:.1f}%")
    m = st.columns(3)
    m[0].metric("Net PnL", f"₹{net:,.0f}", f"{net / cap * 100:+.2f}% of capital")
    m[1].metric("Total loss", f"₹{v.loc[v['PnL ₹'] < 0, 'PnL ₹'].sum():,.0f}")
    m[2].metric("RSI ✓ but loss", sl)
    if sl:
        st.caption("RSI ✓ but loss = RSI target hit ayindi kani price entry kante worse (orange rows).")
    if n_open:
        st.caption(f"{n_open} trade(s) inka open (stats lo lekkaledu).")

    wt = v.loc[v["PnL ₹"].idxmin()]
    if wt["PnL ₹"] < 0:
        st.error(f"🔻 BIGGEST LOSS: {wt.Coin} • {wt.TF} • {wt.Logic} • {wt['PnL %']:+.2f}% "
                 f"(₹{wt['PnL ₹']:,.0f}) • {wt['Entry Time']} → {wt['Exit Time']} • "
                 f"RSI {wt['RSI In']:.1f} → {wt['RSI Out']:.1f}")

    st.subheader("Coin-wise summary (anni coins)")
    ct = coin_table(v)
    q = ct[ct.Trades >= mt]
    if q.empty:
        st.info(f"{mt}+ trades unna coin ledu - 'Min trades' taggu.")
    else:
        b = q.sort_values(["Real Win %", "Trades"], ascending=False).iloc[0]
        lo = q.sort_values("Total Loss ₹", ascending=False).iloc[0]
        hi = q.sort_values("Total Loss ₹").iloc[0]
        st.success(f"🏆 Highest REAL win rate: {b.Coin} — {b['Real Win %']}% "
                   f"(RSI rule {b['Win %']}%, {b.Trades} trades)")
        st.info(f"🛡️ Lowest loss: {lo.Coin} — ₹{lo['Total Loss ₹']:,.0f} total loss ({lo.Trades} trades)")
        st.error(f"⚠️ Highest loss: {hi.Coin} — ₹{hi['Total Loss ₹']:,.0f} total loss, "
                 f"worst trade {hi['Worst Loss %']}%")
    st.dataframe(ct.sort_values(["Real Win %", "Trades"], ascending=False), hide_index=True)
    st.bar_chart(ct.set_index("Coin")["Net PnL ₹"])

    st.subheader("Timeframe-wise (Long + Short together)")
    tt = table(v, ["TF"])
    tt["o"] = tt.TF.map({t: k for k, t in enumerate(TFS)})
    st.dataframe(tt.sort_values("o").drop(columns="o"), hide_index=True)

    st.subheader("Prathi coin separate backtest")
    st.caption("Coin nokku → aa coin ki anni TF lo trades: Entry/Exit time + price. Biggest loss = Loss #1 (red).")
    for coin in COINS:
        g = v[v.Coin == coin]
        go = vo[(vo.Coin == coin) & (vo.Result == "OPEN")]
        cn = ", ".join(f"{t}: {counts[(coin, t)]}" for t in TFS if (coin, t) in counts)
        if g.empty:
            head = f"{coin} — 0 closed trades"
        else:
            rw = (g["PnL ₹"] > 0).sum() / len(g) * 100
            head = f"{coin} — {len(g)} trades • Real win {rw:.0f}% • Net ₹{g['PnL ₹'].sum():,.0f}"
        with st.expander(head):
            st.caption(f"Candles used → {cn if cn else 'data load avvaledu'}")
            if g.empty and go.empty:
                st.info("Ee coin ki ee filter lo trades levu.")
                continue
            if not g.empty:
                k = st.columns(4)
                k[0].metric("Trades", len(g))
                k[1].metric("Win / Fail", f"{int((g.Result == 'SUCCESS').sum())} / {int((g.Result == 'FAIL').sum())}")
                k[2].metric("Real Win %", f"{(g['PnL ₹'] > 0).sum() / len(g) * 100:.1f}%")
                k[3].metric("Net PnL", f"₹{g['PnL ₹'].sum():,.0f}")
                tbl = table(g, ["TF"])
                tbl["o"] = tbl.TF.map({t: k2 for k2, t in enumerate(TFS)})
                st.dataframe(tbl.sort_values("o").drop(columns="o"), hide_index=True)
            d = pd.concat([g, go]).sort_values("_ts", ascending=False)
            st.dataframe(d[TRADE_COLS].style.apply(hl, axis=1).format(FMT, na_rep=""),
                         hide_index=True, height=min(520, 60 + 36 * len(d)))
            if len(go):
                st.caption(f"{len(go)} trade inka open (Exit = latest close, stats lo lekkaledu).")


def main():
    st.set_page_config(page_title="Kishore RSI Scanner", page_icon="📉", layout="wide")
    st.title("📉 Kishore RSI Scanner")
    with st.expander("📖 Logic (RSI Long + RSI Short only)"):
        st.markdown(
            f"- **RSI Long** — RSI ≤ {LONG_IN} → buy at candle close. FAIL: RSI ≤ {LONG_SL}. "
            f"SUCCESS: RSI ≥ {LONG_TP} (or RSI ≥ {LONG_TP_ST} when a bearish Supertrend line rejects price).\n"
            f"- **RSI Short** — RSI ≥ {SHORT_IN} → sell. FAIL: RSI ≥ {SHORT_SL}. SUCCESS: RSI ≤ {SHORT_TP}.\n"
            "- **Win %** = RSI rule SUCCESS. **Real Win %** = price PnL > 0 (RSI target hit ayina price loss lo undochu).\n"
            "- RSI trades have **no price stop** — the RSI level is the stop, so a FAIL can be a big % loss.\n"
            f"- Coins: {', '.join(COINS)}")
    t1, t2 = st.tabs(["🔴 Live scanner (auto)", "📊 Backtest"])
    with t1:
        live_tab()
    with t2:
        backtest_tab()


if __name__ == "__main__":
    main()
