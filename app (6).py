"""
ALGOX RSI + Supertrend wick-rejection engine  --  Kishore
Binance Spot (USDT) data -> indicators -> signals -> backtest + live scan.

Strategies
  RSI_LONG         RSI <= 20 entry, SL when RSI touches 9, TP RSI 65
                   (early TP at RSI 58 if price is rejected by a bearish Supertrend line)
  RSI_SHORT        RSI >= 77 entry, SL RSI 88, TP RSI 47
  ST_SHORT         bearish Supertrend, wick touches line, body closes below, RSI >= 58
                   SL = candle-based, TP = 1:3
  ST_SHORT_NO_RSI same as ST_SHORT but without the RSI filter (for comparison)
  ST_LONG          bullish Supertrend, wick touches line, body closes above,
                   price above EMA99 or EMA200, SL = candle-based, TP = 1:3
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
HOSTS = [
    "https://data-api.binance.vision",   # works from cloud hosts (no geo-block)
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api-gcp.binance.com",
]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]
IST = "Asia/Kolkata"
WARMUP = 210  # candles skipped at the start (EMA200 / RSI / ATR warm-up)

STRATS = ["RSI_LONG", "RSI_SHORT", "ST_SHORT", "ST_SHORT_NO_RSI", "ST_LONG"]
LABEL = {
    "RSI_LONG": "RSI Long",
    "RSI_SHORT": "RSI Short",
    "ST_SHORT": "ST Short (RSI filter)",
    "ST_SHORT_NO_RSI": "ST Short (no RSI)",
    "ST_LONG": "ST Long (EMA filter)",
}
SIDE = {
    "RSI_LONG": "LONG",
    "RSI_SHORT": "SHORT",
    "ST_SHORT": "SHORT",
    "ST_SHORT_NO_RSI": "SHORT",
    "ST_LONG": "LONG",
}
NOT_COINS = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USD1", "PYUSD",
    "XUSD", "UST", "EUR", "EURI", "AEUR", "GBP", "TRY", "BRL", "RLUSD",
}


@dataclass
class Config:
    rsi_len: int = 14
    st_period: int = 10
    st_mult: float = 3.0
    # RSI long
    rl_entry: float = 20.0
    rl_sl: float = 9.0
    rl_tp: float = 65.0
    rl_st_exit: float = 58.0      # early TP when bearish ST rejects price (0 = off)
    # RSI short
    rs_entry: float = 77.0
    rs_sl: float = 88.0
    rs_tp: float = 47.0
    # Supertrend wick-rejection
    st_short_rsi_min: float = 58.0  # ST_SHORT needs RSI >= this
    ema_mode: str = "either"        # either | both | 99 | 200
    rr: float = 3.0                 # 1:3
    small_atr_mult: float = 0.5     # candle range < 0.5*ATR  => "small candle"
    sl_buffer: float = 0.10         # extra 10% of candle range beyond the wick
    body_fully_beyond: bool = True  # whole body must be on the far side of the line
    # money
    fee_pct: float = 0.05           # per side, %
    capital: float = 100000.0
    alloc_pct: float = 50.0         # % of capital per trade


# --------------------------------------------------------------------------- #
# Binance data
# --------------------------------------------------------------------------- #
_state = {"host": 0}


def _get(path: str, params: dict | None = None, timeout: int = 12):
    params = params or {}
    order = HOSTS[_state["host"]:] + HOSTS[:_state["host"]]
    last = "no host tried"
    for host in order:
        try:
            r = requests.get(host + path, params=params, timeout=timeout)
            if r.status_code == 200:
                _state["host"] = HOSTS.index(host)
                return r.json()
            last = f"{host} -> HTTP {r.status_code}"
        except requests.RequestException as e:
            last = f"{host} -> {type(e).__name__}"
    raise RuntimeError("Binance data unreachable: " + last)


def top_symbols(n: int = 50) -> list[str]:
    """Top-n USDT spot pairs by 24h quote volume (stable/fiat bases removed)."""
    data = _get("/api/v3/ticker/24hr")
    rows = []
    for d in data:
        s = d["symbol"]
        if not s.endswith("USDT"):
            continue
        if s[:-4] in NOT_COINS:
            continue
        rows.append((s, float(d.get("quoteVolume", 0) or 0)))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def klines(symbol: str, tf: str, n: int = 1000) -> pd.DataFrame:
    """Last n candles (pages of 1000 going backwards)."""
    rows: list = []
    end = None
    while len(rows) < n:
        lim = min(1000, n - len(rows))
        p = {"symbol": symbol, "interval": tf, "limit": lim}
        if end is not None:
            p["endTime"] = end
        d = _get("/api/v3/klines", p)
        if not d:
            break
        rows = d + rows
        end = int(d[0][0]) - 1
        if len(d) < lim:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def drop_unclosed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the still-forming candle so signals never repaint."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    return df[df["close_time"] < now].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# indicators
# --------------------------------------------------------------------------- #
def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / ad
        r = 100 - 100 / (1 + rs)
    r = r.where(~((ad == 0) & (au > 0)), 100.0)
    return r


def supertrend(d: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """TradingView-style Supertrend. dir: +1 bullish (line below), -1 bearish (line above)."""
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    n = len(d)
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()
    hl2 = (h + l) / 2
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    fub = np.full(n, np.nan)
    flb = np.full(n, np.nan)
    st = np.full(n, np.nan)
    dr = np.zeros(n, dtype=int)
    valid = np.flatnonzero(np.isfinite(atr))
    if len(valid) == 0:
        return st, dr, atr
    s = valid[0]
    fub[s], flb[s], dr[s], st[s] = ub[s], lb[s], 1, lb[s]
    for i in range(s + 1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if dr[i - 1] == -1:
            dr[i] = 1 if c[i] > fub[i] else -1
        else:
            dr[i] = -1 if c[i] < flb[i] else 1
        st[i] = flb[i] if dr[i] == 1 else fub[i]
    return st, dr, atr


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    d["rsi"] = rsi_wilder(c, cfg.rsi_len)
    d["ema99"] = c.ewm(span=99, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    st, dr, atr = supertrend(d, cfg.st_period, cfg.st_mult)
    d["st"], d["st_dir"], d["atr"] = st, dr, atr
    return d


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def add_signals(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = d.copy()
    n = len(d)
    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    st = d["st"].to_numpy()
    dr = d["st_dir"].to_numpy()
    rsi = d["rsi"].to_numpy()
    atr = d["atr"].to_numpy()
    valid = np.isfinite(st) & np.isfinite(rsi) & np.isfinite(atr) & (np.arange(n) >= WARMUP)

    with np.errstate(invalid="ignore"):
        bhi = np.maximum(o, c)
        blo = np.minimum(o, c)
        if cfg.body_fully_beyond:
            bear = (dr == -1) & (h >= st) & (bhi < st)
            bull = (dr == 1) & (l <= st) & (blo > st)
        else:
            bear = (dr == -1) & (h >= st) & (c < st)
            bull = (dr == 1) & (l <= st) & (c > st)
        bear_any = (dr == -1) & (h >= st) & (c < st)   # used for RSI-long early exit

        e99 = c > d["ema99"].to_numpy()
        e200 = c > d["ema200"].to_numpy()
        ema_ok = {"either": e99 | e200, "both": e99 & e200, "99": e99, "200": e200}[cfg.ema_mode]

        # candle-based stop distance: whole candle (+buffer), tiny candle => 2x candle
        rng = h - l
        small = rng < cfg.small_atr_mult * atr
        r_short = np.maximum(h - c + cfg.sl_buffer * rng, rng)
        r_long = np.maximum(c - l + cfg.sl_buffer * rng, rng)
        r_short = np.where(small, np.maximum(r_short, 2 * rng), r_short)
        r_long = np.where(small, np.maximum(r_long, 2 * rng), r_long)
        ok_rng = rng > 0

        d["risk_short"] = np.where(ok_rng, r_short, np.nan)
        d["risk_long"] = np.where(ok_rng, r_long, np.nan)
        d["rej_bear_any"] = bear_any & valid
        d["sig_RSI_LONG"] = valid & (rsi <= cfg.rl_entry) & (rsi > cfg.rl_sl)
        d["sig_RSI_SHORT"] = valid & (rsi >= cfg.rs_entry) & (rsi < cfg.rs_sl)
        d["sig_ST_SHORT"] = valid & bear & ok_rng & (rsi >= cfg.st_short_rsi_min)
        d["sig_ST_SHORT_NO_RSI"] = valid & bear & ok_rng
        d["sig_ST_LONG"] = valid & bull & ok_rng & ema_ok
    return d


# --------------------------------------------------------------------------- #
# backtest simulation
# --------------------------------------------------------------------------- #
def _walk_rsi(strat, i, n, c, rsi, rej, cfg):
    """Return (result, exit_idx, exit_price, reason) for an RSI-based trade."""
    for j in range(i + 1, n):
        r = rsi[j]
        if strat == "RSI_LONG":
            if r <= cfg.rl_sl:
                return "FAIL", j, c[j], f"RSI<={cfg.rl_sl:g} (SL)"
            if r >= cfg.rl_tp:
                return "SUCCESS", j, c[j], f"RSI>={cfg.rl_tp:g} (TP)"
            if cfg.rl_st_exit > 0 and rej[j] and r >= cfg.rl_st_exit:
                return "SUCCESS", j, c[j], f"Bear ST rejection @RSI>={cfg.rl_st_exit:g}"
        else:
            if r >= cfg.rs_sl:
                return "FAIL", j, c[j], f"RSI>={cfg.rs_sl:g} (SL)"
            if r <= cfg.rs_tp:
                return "SUCCESS", j, c[j], f"RSI<={cfg.rs_tp:g} (TP)"
    return "OPEN", n - 1, c[n - 1], "still open"


def _walk_price(side, i, n, h, l, c, sl, tp):
    for j in range(i + 1, n):
        if side == "SHORT":
            if h[j] >= sl:          # SL first when both hit in one candle (conservative)
                return "FAIL", j, sl, "SL hit"
            if l[j] <= tp:
                return "SUCCESS", j, tp, "TP 1:R hit"
        else:
            if l[j] <= sl:
                return "FAIL", j, sl, "SL hit"
            if h[j] >= tp:
                return "SUCCESS", j, tp, "TP 1:R hit"
    return "OPEN", n - 1, c[n - 1], "still open"


def simulate(d: pd.DataFrame, sym: str, tf: str, cfg: Config, strats=STRATS) -> list[dict]:
    n = len(d)
    c = d["close"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    rsi = d["rsi"].to_numpy()
    rej = d["rej_bear_any"].to_numpy()
    ent_t = (d["close_time"] + pd.Timedelta(milliseconds=1)).dt.tz_convert(IST).dt.strftime("%Y-%m-%d %H:%M").to_numpy()
    notional = cfg.capital * cfg.alloc_pct / 100.0
    fee2 = 2 * cfg.fee_pct
    out: list[dict] = []

    for strat in strats:
        side = SIDE[strat]
        sig = d["sig_" + strat].to_numpy()
        last_exit = -1
        for i in np.flatnonzero(sig):
            if i <= last_exit:
                continue
            entry = float(c[i])
            sl = tp = np.nan
            if strat.startswith("RSI"):
                res, xi, xp, why = _walk_rsi(strat, i, n, c, rsi, rej, cfg)
            else:
                risk = float(d["risk_short"].iat[i] if side == "SHORT" else d["risk_long"].iat[i])
                if not np.isfinite(risk) or risk <= 0:
                    continue
                if side == "SHORT":
                    sl, tp = entry + risk, entry - cfg.rr * risk
                else:
                    sl, tp = entry - risk, entry + cfg.rr * risk
                    if sl <= 0 or tp <= 0:
                        continue
                if side == "SHORT" and tp <= 0:
                    continue
                res, xi, xp, why = _walk_price(side, i, n, h, l, c, sl, tp)
            last_exit = xi
            gross = (xp - entry) / entry * 100 if side == "LONG" else (entry - xp) / entry * 100
            pnl_pct = gross - fee2
            out.append({
                "strategy": strat,
                "symbol": sym,
                "tf": tf,
                "side": side,
                "entry_time": ent_t[i],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rsi_entry": round(float(rsi[i]), 1),
                "exit_time": ent_t[xi] if res != "OPEN" else "",
                "exit": float(xp),
                "result": res,
                "reason": why,
                "bars": int(xi - i),
                "pnl_pct": round(pnl_pct, 3),
                "pnl_amt": round(notional * pnl_pct / 100.0, 2),
            })
    return out


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def summarize(trades: pd.DataFrame, by=("strategy", "tf")) -> pd.DataFrame:
    cols = list(by) + ["Trades", "Success", "Fail", "Open", "Win %", "PnL", "Avg %/trade", "Profit factor"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, grp in trades.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        cl = grp[grp["result"] != "OPEN"]
        w = int((cl["result"] == "SUCCESS").sum())
        f = int((cl["result"] == "FAIL").sum())
        gp = cl.loc[cl["pnl_amt"] > 0, "pnl_amt"].sum()
        gl = -cl.loc[cl["pnl_amt"] < 0, "pnl_amt"].sum()
        row = dict(zip(by, key))
        row.update({
            "Trades": len(cl),
            "Success": w,
            "Fail": f,
            "Open": len(grp) - len(cl),
            "Win %": round(100 * w / len(cl), 1) if len(cl) else np.nan,
            "PnL": round(float(cl["pnl_amt"].sum()), 0),
            "Avg %/trade": round(float(cl["pnl_pct"].mean()), 3) if len(cl) else np.nan,
            "Profit factor": round(gp / gl, 2) if gl > 0 else (np.inf if gp > 0 else np.nan),
        })
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    if "tf" in by:
        out["_o"] = out["tf"].map({t: i for i, t in enumerate(TIMEFRAMES)})
        out = out.sort_values([c for c in by if c != "tf"] + ["_o"]).drop(columns="_o")
    return out.reset_index(drop=True)


def best_picks(summary: pd.DataFrame, min_trades: int = 10):
    """Best strategy/timeframe by win rate and by PnL (needs a minimum sample)."""
    s = summary[summary["Trades"] >= min_trades]
    if s.empty:
        return None, None
    return s.loc[s["Win %"].idxmax()], s.loc[s["PnL"].idxmax()]


# --------------------------------------------------------------------------- #
# runners (threaded)
# --------------------------------------------------------------------------- #
def _prep(sym: str, tf: str, n: int, cfg: Config) -> pd.DataFrame:
    df = drop_unclosed(klines(sym, tf, n))
    if len(df) < WARMUP + 20:
        return pd.DataFrame()
    return add_signals(add_indicators(df, cfg), cfg)


def run_backtest(symbols, tfs, cfg: Config, candles: int = 2000, workers: int = 8, progress=None):
    jobs = [(s, tf) for s in symbols for tf in tfs]
    trades: list[dict] = []
    errors: list[str] = []

    def work(job):
        s, tf = job
        d = _prep(s, tf, candles, cfg)
        return simulate(d, s, tf, cfg) if len(d) else []

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                trades += f.result()
            except Exception as e:  # keep going, report at the end
                errors.append(f"{futs[f][0]} {futs[f][1]}: {e}")
            if progress:
                progress(done / len(jobs))
    return pd.DataFrame(trades), errors


def _scan_one(sym, tf, cfg, lookback, candles):
    d = _prep(sym, tf, candles, cfg)
    if d.empty:
        return []
    n = len(d)
    ent_t = (d["close_time"] + pd.Timedelta(milliseconds=1)).dt.tz_convert(IST).dt.strftime("%d-%b %H:%M").to_numpy()
    c = d["close"].to_numpy()
    rows = []
    for strat in STRATS:
        sig = d["sig_" + strat].to_numpy()
        side = SIDE[strat]
        for i in range(max(WARMUP, n - lookback), n):
            if not sig[i]:
                continue
            if strat.startswith("RSI"):
                sl_txt = f"RSI {cfg.rl_sl:g}" if strat == "RSI_LONG" else f"RSI {cfg.rs_sl:g}"
                tp_txt = f"RSI {cfg.rl_tp:g}" if strat == "RSI_LONG" else f"RSI {cfg.rs_tp:g}"
            else:
                risk = float(d["risk_short"].iat[i] if side == "SHORT" else d["risk_long"].iat[i])
                e = float(c[i])
                sl = e + risk if side == "SHORT" else e - risk
                tp = e - cfg.rr * risk if side == "SHORT" else e + cfg.rr * risk
                sl_txt, tp_txt = f"{sl:.6g}", f"{tp:.6g}"
            rows.append({
                "Signal time (IST)": ent_t[i],
                "Symbol": sym,
                "TF": tf,
                "Strategy": LABEL[strat],
                "Side": side,
                "Price": f"{c[i]:.6g}",
                "SL": sl_txt,
                "TP": tp_txt,
                "RSI": round(float(d["rsi"].iat[i]), 1),
                "Age (candles)": int(n - 1 - i),
                "_sort": (n - 1 - i),
            })
    return rows


def scan_all(symbols, tfs, cfg: Config, lookback: int = 3, workers: int = 10, candles: int = 500):
    rows: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_one, s, tf, cfg, lookback, candles): (s, tf) for s in symbols for tf in tfs}
        for f in as_completed(futs):
            try:
                rows += f.result()
            except Exception as e:
                errors.append(f"{futs[f][0]} {futs[f][1]}: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["_sort", "Symbol"]).reset_index(drop=True)
    return df, errors


# ==========================================================================
# STREAMLIT APP
# ==========================================================================
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


st.set_page_config(page_title="ALGOX RSI + Supertrend Scanner", page_icon="📡", layout="wide")
st.title("📡 ALGOX — RSI + Supertrend Wick-Reject Scanner")
st.caption("Kishore  •  Binance Spot USDT  •  closed candles only (no repaint)")

# ----------------------------------------------------------------------------- #
# settings (all rules editable, defaults = Kishore's rules)
# ----------------------------------------------------------------------------- #
with st.expander("⚙️ Rules / settings"):
    st.markdown("**RSI Long**")
    a, b, c, d = st.columns(4)
    rl_entry = a.number_input("Entry RSI ≤", 1.0, 50.0, 20.0, 1.0)
    rl_sl = b.number_input("SL when RSI ≤", 1.0, 30.0, 9.0, 1.0)
    rl_tp = c.number_input("TP RSI ≥", 40.0, 90.0, 65.0, 1.0)
    rl_st = d.number_input("Early TP RSI ≥ (bear ST rejection, 0=off)", 0.0, 90.0, 58.0, 1.0)
    st.markdown("**RSI Short**")
    a, b, c = st.columns(3)
    rs_entry = a.number_input("Entry RSI ≥", 50.0, 99.0, 77.0, 1.0)
    rs_sl = b.number_input("SL when RSI ≥", 51.0, 99.0, 88.0, 1.0)
    rs_tp = c.number_input("TP RSI ≤", 10.0, 70.0, 47.0, 1.0)
    st.markdown("**Supertrend wick-reject (Supertrend 10, 3)**")
    a, b, c, d = st.columns(4)
    st_rsi = a.number_input("ST Short needs RSI ≥", 0.0, 90.0, 58.0, 1.0)
    ema_mode = b.selectbox("ST Long EMA filter", ["either", "both", "99", "200"],
                           help="either = price above EMA99 OR EMA200")
    rr = c.number_input("Target (R multiple)", 1.0, 10.0, 3.0, 0.5)
    small_mult = d.number_input("Small candle < ATR ×", 0.1, 2.0, 0.5, 0.1,
                                help="Small candle => stop-loss = 2 × candle size")
    body_beyond = st.checkbox("Whole candle BODY must be beyond the line (only the wick touches)", True)
    st.markdown("**Backtest money**")
    a, b, c = st.columns(3)
    capital = a.number_input("Capital", 1000.0, 100000000.0, 100000.0, 10000.0)
    alloc = b.number_input("% capital per trade", 1.0, 100.0, 50.0, 5.0)
    fee = c.number_input("Fee per side %", 0.0, 0.5, 0.05, 0.01)

cfg = Config(
    rl_entry=rl_entry, rl_sl=rl_sl, rl_tp=rl_tp, rl_st_exit=rl_st,
    rs_entry=rs_entry, rs_sl=rs_sl, rs_tp=rs_tp,
    st_short_rsi_min=st_rsi, ema_mode=ema_mode, rr=rr, small_atr_mult=small_mult,
    body_fully_beyond=body_beyond, capital=capital, alloc_pct=alloc, fee_pct=fee,
)

with st.expander("📖 Logic (how each signal works)"):
    st.markdown(
        f"""
- **RSI Long** — RSI ≤ {rl_entry:g} → buy at candle close. Exit: RSI ≤ {rl_sl:g} = FAIL, RSI ≥ {rl_tp:g} = SUCCESS.
  If a bearish Supertrend line rejects price on the way up, RSI ≥ {rl_st:g} is accepted as SUCCESS.
- **RSI Short** — RSI ≥ {rs_entry:g} → sell. RSI ≥ {rs_sl:g} = FAIL, RSI ≤ {rs_tp:g} = SUCCESS.
- **ST Short** — Supertrend bearish (line above price). Candle **wick touches** the line and the **body closes below** it.
  RSI ≥ {st_rsi:g}. No EMA condition. *(ST Short (no RSI) = same without the RSI filter, for comparison.)*
- **ST Long** — Supertrend bullish (line below price). Wick touches the line, body closes above it,
  and price is above EMA99 {'or' if ema_mode == 'either' else 'and' if ema_mode == 'both' else '/'} EMA200 (mode: {ema_mode}).
- **Stop-loss (ST trades)** — beyond the wick of the signal candle, at least the whole candle length (+10%).
  If the candle is small (< {small_mult:g} × ATR) the stop is 2 × candle size. **Target = 1:{rr:g}**.
- If SL and TP are both inside one candle, SL is counted first (conservative).
- RSI trades have **no price stop** — the RSI level is the stop, so a FAIL can be a big % loss.
"""
    )

tab_live, tab_bt = st.tabs(["🔴 Live scanner (auto)", "📊 Backtest"])


# ----------------------------------------------------------------------------- #
# cached helpers
# ----------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner=False)
def symbols_cached(n: int):
    return top_symbols(n)


@st.cache_data(ttl=45, show_spinner=False)
def scan_cached(symbols: tuple, tfs: tuple, cfgd: dict, lookback: int):
    return scan_all(list(symbols), list(tfs), Config(**cfgd), lookback=lookback)


def now_ist() -> str:
    return datetime.now(ZoneInfo(IST)).strftime("%d-%b %H:%M:%S")


# ----------------------------------------------------------------------------- #
# LIVE
# ----------------------------------------------------------------------------- #
@st.fragment(run_every=60)
def live_view(n_coins, tfs, cfgd, lookback, picks):
    try:
        syms = symbols_cached(n_coins)
        df, errs = scan_cached(tuple(syms), tuple(tfs), cfgd, lookback)
    except Exception as e:
        st.error(f"Binance data fetch failed: {e}")
        return
    labels = {LABEL[k] for k in picks}
    if not df.empty:
        df = df[df["Strategy"].isin(labels)]
    st.caption(f"Auto-refresh every 60s  •  last scan {now_ist()} IST  •  {len(syms)} coins × {len(tfs)} TF"
               + (f"  •  ⚠️ {len(errs)} fetch errors" if errs else ""))
    if df.empty:
        st.info("No signals on the last candles right now. Scanner keeps running automatically.")
        return
    show = df.drop(columns=["_sort"])
    fresh = show[show["Age (candles)"] == 0]
    older = show[show["Age (candles)"] > 0]
    st.subheader(f"🆕 Fresh signals — just closed candle ({len(fresh)})")
    if fresh.empty:
        st.write("None on the latest candle.")
    else:
        for side, icon in (("SHORT", "🔴"), ("LONG", "🟢")):
            part = fresh[fresh["Side"] == side]
            if len(part):
                st.markdown(f"**{icon} {side}** ({len(part)})")
                st.dataframe(part, hide_index=True)
    if len(older):
        with st.expander(f"Earlier signals within last {lookback} candles ({len(older)})"):
            st.dataframe(older, hide_index=True)
    if errs:
        with st.expander("Fetch errors"):
            st.write(errs[:20])


with tab_live:
    c1, c2, c3 = st.columns(3)
    n_coins = c1.slider("Coins (top by volume)", 10, 150, 50, 10)
    lookback = c2.slider("Look back (closed candles)", 1, 10, 3)
    tfs = c3.multiselect("Timeframes", TIMEFRAMES, default=TIMEFRAMES)
    picks = st.multiselect("Strategies", STRATS, default=STRATS, format_func=lambda k: LABEL[k])
    if not tfs or not picks:
        st.warning("Select at least one timeframe and one strategy.")
    else:
        live_view(n_coins, tfs, asdict(cfg), lookback, picks)

# ----------------------------------------------------------------------------- #
# BACKTEST
# ----------------------------------------------------------------------------- #
with tab_bt:
    c1, c2, c3 = st.columns(3)
    bt_coins = c1.slider("Coins (top by volume)", 5, 100, 30, 5, key="bt_coins")
    bt_candles = c2.select_slider("Candles per timeframe", [1000, 2000, 3000, 5000, 8000], 3000)
    bt_tfs = c3.multiselect("Timeframes", TIMEFRAMES, default=TIMEFRAMES, key="bt_tfs")
    st.caption("5m×3000 ≈ 10 days, 15m ≈ 31 days, 30m ≈ 62 days, 1h ≈ 125 days, 4h ≈ 500 days. "
               "More candles = more trades = more reliable win rate.")

    if st.button("▶️ Run backtest", type="primary"):
        if not bt_tfs:
            st.warning("Select at least one timeframe.")
        else:
            bar = st.progress(0.0, text="Fetching Binance candles & simulating…")
            try:
                syms = symbols_cached(bt_coins)
                trades, errs = run_backtest(syms, bt_tfs, cfg, candles=bt_candles,
                                               progress=lambda p: bar.progress(min(p, 1.0)))
                st.session_state["bt"] = {"trades": trades, "errs": errs, "n": len(syms),
                                          "at": now_ist(), "cfg": asdict(cfg)}
            except Exception as e:
                st.error(f"Backtest failed: {e}")
            bar.empty()

    res = st.session_state.get("bt")
    if res is not None:
        trades = res["trades"]
        st.caption(f"Backtest run {res['at']} IST  •  {res['n']} coins  •  "
                   f"capital {res['cfg']['capital']:,.0f}, {res['cfg']['alloc_pct']:g}% per trade, "
                   f"fee {res['cfg']['fee_pct']:g}%/side")
        if res["errs"]:
            with st.expander(f"⚠️ {len(res['errs'])} fetch errors"):
                st.write(res["errs"][:30])
        if trades.empty:
            st.info("No trades generated. Try more candles or more coins.")
        else:
            summ = summarize(trades)
            overall = summarize(trades, by=("strategy",))
            min_tr = st.number_input("Minimum trades to rank (avoid tiny samples)", 1, 200, 20)
            best_wr, best_pnl = best_picks(summ, int(min_tr))
            m1, m2 = st.columns(2)
            if best_wr is not None:
                m1.metric("🏆 Highest win rate",
                          f"{LABEL[best_wr['strategy']]} · {best_wr['tf']}",
                          f"{best_wr['Win %']}% ({int(best_wr['Trades'])} trades)")
                m2.metric("💰 Highest PnL",
                          f"{LABEL[best_pnl['strategy']]} · {best_pnl['tf']}",
                          f"{best_pnl['PnL']:,.0f} ({int(best_pnl['Trades'])} trades)")
            else:
                st.info(f"No strategy/timeframe has {int(min_tr)}+ trades yet.")

            def nice(x):
                x = x.copy()
                if "strategy" in x:
                    x["strategy"] = x["strategy"].map(LABEL)
                return x

            st.subheader("Strategy × Timeframe")
            st.dataframe(nice(summ), hide_index=True)
            st.subheader("Win rate by timeframe (%)")
            pv = summ.pivot(index="strategy", columns="tf", values="Win %")
            pv = pv[[t for t in TIMEFRAMES if t in pv.columns]]
            pv.index = pv.index.map(LABEL)
            st.dataframe(pv)
            st.subheader("PnL by timeframe")
            pv2 = summ.pivot(index="strategy", columns="tf", values="PnL")
            pv2 = pv2[[t for t in TIMEFRAMES if t in pv2.columns]]
            pv2.index = pv2.index.map(LABEL)
            st.dataframe(pv2)
            st.subheader("Per strategy (all timeframes)")
            st.dataframe(nice(overall), hide_index=True)

            st.subheader("Every trade — SUCCESS / FAIL")
            f1, f2, f3, f4 = st.columns(4)
            fs = f1.multiselect("Strategy", STRATS, default=STRATS, format_func=lambda k: LABEL[k], key="f_s")
            ft = f2.multiselect("TF", TIMEFRAMES, default=[t for t in TIMEFRAMES if t in set(trades.tf)], key="f_t")
            fr = f3.multiselect("Result", ["SUCCESS", "FAIL", "OPEN"], default=["SUCCESS", "FAIL", "OPEN"], key="f_r")
            fc = f4.text_input("Coin contains", "", key="f_c").upper()
            view = trades[trades.strategy.isin(fs) & trades.tf.isin(ft) & trades.result.isin(fr)]
            if fc:
                view = view[view.symbol.str.contains(fc)]
            view = view.sort_values("entry_time", ascending=False)
            view = nice(view).assign(result=lambda x: x["result"].map({"SUCCESS": "✅ SUCCESS", "FAIL": "❌ FAIL", "OPEN": "⏳ OPEN"}))
            st.caption(f"{len(view)} trades shown")
            st.dataframe(view, hide_index=True)
            st.download_button("⬇️ Download all trades (CSV)", trades.to_csv(index=False).encode(),
                               file_name="algox_backtest_trades.csv", mime="text/csv")
