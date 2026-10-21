"""
RSI Alert Scanner
Scans a watchlist of live market data and flags instruments that are
at or approaching RSI 80 (overbought) or RSI 20 (oversold).

Run:
    pip install streamlit yfinance pandas
    streamlit run rsi_alerts.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="RSI Alerts", layout="wide")

# ---------- Defaults ----------
FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

FOREX_CROSSES = [
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "CHFJPY",
    "NZDJPY", "NZDCAD", "NZDCHF",
]

FOREX_EXOTICS = [
    "USDSGD", "USDHKD", "USDCNH", "USDINR", "USDTHB", "USDKRW", "USDIDR", "USDPHP",
    "USDZAR", "USDTRY", "USDMXN", "USDBRL",
    "USDNOK", "USDSEK", "USDDKK", "USDPLN", "USDHUF", "USDCZK",
    "EURNOK", "EURSEK", "EURPLN", "EURHUF", "EURCZK", "EURTRY", "EURZAR", "EURSGD",
    "GBPSGD", "GBPZAR", "GBPNOK", "GBPSEK",
    "AUDSGD", "SGDJPY",
]

INDICES = [
    # US
    "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX",
    # Europe
    "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^IBEX", "^AEX", "^SSMI", "^BFX", "^OMX",
    # Asia-Pacific
    "^N225", "^HSI", "^AXJO", "^AORD", "^NZ50", "000001.SS", "399001.SZ",
    "^KS11", "^TWII", "^STI", "^JKSE", "^KLSE", "^BSESN", "^NSEI",
    # Americas
    "^GSPTSE", "^BVSP", "^MXX", "^MERV",
]


def _fx(pairs):
    return [p + "=X" for p in pairs]


DEFAULT_WATCHLIST = "\n".join(
    _fx(FOREX_MAJORS) + _fx(FOREX_CROSSES) + _fx(FOREX_EXOTICS) + INDICES
) + "\n"

# interval -> how much history to pull (enough for RSI to settle)
PERIOD_FOR_INTERVAL = {
    "1m": "5d",
    "5m": "5d",
    "15m": "5d",
    "30m": "1mo",
    "1h": "1mo",
    "1d": "6mo",
}


# ---------- RSI ----------
def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)  # no losses -> RSI 100
    return rsi


def classify(rsi: float, upper: float, lower: float, buffer: float) -> str:
    if rsi >= upper:
        return "OVERBOUGHT"
    if rsi <= lower:
        return "OVERSOLD"
    if rsi >= upper - buffer:
        return "NEAR 80"
    if rsi <= lower + buffer:
        return "NEAR 20"
    return "neutral"


# ---------- Data ----------
def extract_close(raw: pd.DataFrame, ticker: str, single: bool):
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                s = raw[ticker]["Close"]
            elif single:
                s = raw["Close"].iloc[:, 0]
            else:
                return None
        else:
            s = raw["Close"]
        s = s.dropna()
        return s if len(s) else None
    except Exception:
        return None


@st.cache_data(ttl=20, show_spinner=False)
def fetch(tickers: tuple, interval: str) -> pd.DataFrame:
    return yf.download(
        list(tickers),
        period=PERIOD_FOR_INTERVAL[interval],
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def scan(tickers, interval, period, upper, lower, buffer) -> pd.DataFrame:
    raw = fetch(tuple(tickers), interval)
    rows = []
    single = len(tickers) == 1
    for t in tickers:
        close = extract_close(raw, t, single)
        if close is None or len(close) < period + 2:
            rows.append({"Ticker": t, "Price": None, "RSI": None, "Status": "no data", "Last bar": ""})
            continue
        rsi = wilder_rsi(close, period).iloc[-1]
        if pd.isna(rsi):
            rows.append({"Ticker": t, "Price": float(close.iloc[-1]), "RSI": None, "Status": "no data", "Last bar": ""})
            continue
        rows.append(
            {
                "Ticker": t,
                "Price": round(float(close.iloc[-1]), 4),
                "RSI": round(float(rsi), 1),
                "Status": classify(float(rsi), upper, lower, buffer),
                "Last bar": str(close.index[-1])[:16],
            }
        )
    df = pd.DataFrame(rows)
    # Alerts first, then closest to a threshold
    order = {"OVERBOUGHT": 0, "OVERSOLD": 0, "NEAR 80": 1, "NEAR 20": 1, "neutral": 2, "no data": 3}
    df["_o"] = df["Status"].map(order)
    df["_d"] = df["RSI"].apply(lambda r: 999 if pd.isna(r) else min(abs(upper - r), abs(r - lower)))
    return df.sort_values(["_o", "_d"]).drop(columns=["_o", "_d"]).reset_index(drop=True)


def beep():
    components.html(
        """
        <script>
        try {
          const c = new (window.AudioContext || window.webkitAudioContext)();
          const o = c.createOscillator(); const g = c.createGain();
          o.connect(g); g.connect(c.destination);
          o.frequency.value = 880; g.gain.value = 0.15;
          o.start(); setTimeout(() => { o.stop(); c.close(); }, 350);
        } catch (e) {}
        </script>
        """,
        height=0,
    )


def colour_row(row):
    s = row["Status"]
    if s == "OVERBOUGHT":
        c = "background-color:#7f1d1d;color:white"
    elif s == "OVERSOLD":
        c = "background-color:#14532d;color:white"
    elif s in ("NEAR 80", "NEAR 20"):
        c = "background-color:#78350f;color:white"
    else:
        c = ""
    return [c] * len(row)


# ---------- Backtest ----------
# Rules: SELL when RSI crosses up through the upper level, take profit when RSI
# reaches the lower level. BUY when RSI crosses down through the lower level,
# take profit when RSI reaches the upper level. Signals are read at bar close and
# filled at the NEXT bar's open (no look-ahead).
BT_PERIODS = {
    "1d": ["1y", "2y", "5y", "10y", "max"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "30m": ["5d", "1mo", "60d"],
    "15m": ["5d", "1mo", "60d"],
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hist(tickers: tuple, interval: str, period: str) -> pd.DataFrame:
    return yf.download(
        list(tickers),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def extract_ohlc(raw: pd.DataFrame, ticker: str, single: bool):
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                d = raw[ticker][["Open", "Close"]]
            elif single:
                d = pd.DataFrame(
                    {"Open": raw["Open"].iloc[:, 0], "Close": raw["Close"].iloc[:, 0]}
                )
            else:
                return None
        else:
            d = raw[["Open", "Close"]]
        d = d.dropna()
        return d if len(d) else None
    except Exception:
        return None


def _trade(ticker, side, idx, ei, epx, xi, xpx, reason, cost_pct):
    if side == 1:
        ret = (xpx / epx - 1) * 100
    else:
        ret = (epx - xpx) / epx * 100
    return {
        "Ticker": ticker,
        "Side": "Long" if side == 1 else "Short",
        "Entry time": idx[ei],
        "Entry": float(epx),
        "Exit time": idx[xi],
        "Exit": float(xpx),
        "Return %": float(ret - cost_pct),
        "Bars": int(xi - ei),
        "Result": reason,
    }


def backtest_ticker(ticker, df, period, upper, lower, stop_pct, cost_pct, reverse):
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(None) if idx.tz is None else idx.tz_convert("UTC").tz_localize(None)
    o = df["Open"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    r = wilder_rsi(df["Close"], period).to_numpy(dtype=float)

    trades = []
    pos = 0  # 1 long, -1 short, 0 flat
    entry_px, entry_i = 0.0, 0
    pend_exit = None  # reason string when an exit is queued for next open
    pend_entry = 0  # +1 / -1 when an entry is queued for next open

    for i in range(len(df)):
        # 1) execute orders queued from the previous bar at this bar's open
        if pend_exit is not None:
            if pos != 0:
                trades.append(_trade(ticker, pos, idx, entry_i, entry_px, i, o[i], pend_exit, cost_pct))
                pos = 0
            pend_exit = None
        if pend_entry != 0 and pos == 0:
            pos, entry_px, entry_i = pend_entry, o[i], i
        pend_entry = 0

        # 2) read signals at this bar's close
        ri = r[i]
        if np.isnan(ri):
            continue
        if pos != 0:
            stopped = stop_pct > 0 and (
                (pos == 1 and c[i] <= entry_px * (1 - stop_pct / 100))
                or (pos == -1 and c[i] >= entry_px * (1 + stop_pct / 100))
            )
            target = (pos == 1 and ri >= upper) or (pos == -1 and ri <= lower)
            if stopped:
                pend_exit = "stop"
            elif target:
                pend_exit = "target"
                if reverse:
                    pend_entry = -pos
        elif i > 0 and not np.isnan(r[i - 1]):
            if ri >= upper and r[i - 1] < upper:
                pend_entry = -1  # sell the overbought cross
            elif ri <= lower and r[i - 1] > lower:
                pend_entry = 1  # buy the oversold cross

    if pos != 0:  # still open: mark to market at the last close
        trades.append(_trade(ticker, pos, idx, entry_i, entry_px, len(df) - 1, c[-1], "open", cost_pct))
    return trades


def _stats(name: str, g: pd.DataFrame) -> dict:
    closed = g[g["Result"] != "open"]
    open_pl = g.loc[g["Result"] == "open", "Return %"].sum()
    wins = closed[closed["Return %"] > 0]["Return %"]
    losses = closed[closed["Return %"] <= 0]["Return %"]
    longs = closed[closed["Side"] == "Long"]
    shorts = closed[closed["Side"] == "Short"]

    def win_pct(x):
        return round((x["Return %"] > 0).mean() * 100, 1) if len(x) else None

    gross_loss = abs(losses.sum())
    return {
        "Ticker": name,
        "Trades": len(closed),
        "Win %": win_pct(closed),
        "Avg %": round(closed["Return %"].mean(), 3) if len(closed) else None,
        "Total %": round(closed["Return %"].sum(), 2),
        "Profit factor": round(wins.sum() / gross_loss, 2) if gross_loss > 0 else None,
        "Worst %": round(closed["Return %"].min(), 2) if len(closed) else None,
        "Avg bars": round(closed["Bars"].mean(), 1) if len(closed) else None,
        "Long win %": win_pct(longs),
        "Short win %": win_pct(shorts),
        "Open P/L %": round(open_pl, 2),
    }


def summarise(trades: pd.DataFrame) -> pd.DataFrame:
    rows = [_stats(t, g) for t, g in trades.groupby("Ticker")]
    body = pd.DataFrame(rows).sort_values("Total %", ascending=False)
    total = pd.DataFrame([_stats("ALL", trades)])
    return pd.concat([total, body], ignore_index=True)


def backtest_ui():
    st.subheader("Backtest: fade the RSI extremes")
    st.caption(
        f"Sell when RSI crosses above {upper}, take profit at {lower}. "
        f"Buy when RSI crosses below {lower}, take profit at {upper}. "
        "Uses the RSI period, levels and watchlist from the sidebar. "
        "Entries and exits fill at the next bar's open."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    tf = c1.selectbox("Timeframe", list(BT_PERIODS), index=0, key="bt_tf")
    hist = c2.selectbox("History", BT_PERIODS[tf], index=2, key=f"bt_hist_{tf}")
    stop = c3.number_input("Stop loss % (0 = none)", 0.0, 50.0, 0.0, 0.5, key="bt_stop")
    cost = c4.number_input("Cost per trade %", 0.0, 5.0, 0.0, 0.01, format="%.3f", key="bt_cost")
    reverse = c5.checkbox("Reverse at target", value=True, key="bt_rev",
                          help="Take profit and open the opposite trade on the same signal.")

    if st.button("Run backtest", type="primary"):
        if not tickers:
            st.warning("Add at least one ticker.")
        else:
            with st.spinner(f"Downloading {len(tickers)} instruments and running the backtest..."):
                try:
                    raw = fetch_hist(tuple(tickers), tf, hist)
                except Exception as e:
                    st.error(f"Data fetch failed: {e}")
                    raw = None
                if raw is not None:
                    single = len(tickers) == 1
                    all_trades, nodata = [], []
                    for t in tickers:
                        d = extract_ohlc(raw, t, single)
                        if d is None or len(d) < int(rsi_period) + 5:
                            nodata.append(t)
                            continue
                        all_trades.extend(
                            backtest_ticker(t, d, int(rsi_period), upper, lower, stop, cost, reverse)
                        )
                    st.session_state.bt = {
                        "trades": pd.DataFrame(all_trades),
                        "nodata": nodata,
                        "tested": len(tickers) - len(nodata),
                        "label": f"{tf} / {hist}, RSI({int(rsi_period)}) {upper}/{lower}, "
                                 f"stop {stop}%, cost {cost}%, reverse {'on' if reverse else 'off'}",
                    }

    res = st.session_state.get("bt")
    if not res:
        st.info("Set your options and press Run backtest.")
        return

    st.caption(f"Last run: {res['label']}. {res['tested']} instruments tested.")
    if res["nodata"]:
        st.caption("No usable data: " + ", ".join(res["nodata"]))
    trades = res["trades"]
    if trades.empty:
        st.info("No trades were triggered with these settings.")
        return

    summary = summarise(trades)
    top = summary.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closed trades", int(top["Trades"]))
    m2.metric("Win rate", f"{top['Win %']}%" if pd.notna(top["Win %"]) else "n/a")
    m3.metric("Total return (sum of trades)", f"{top['Total %']}%")
    m4.metric("Worst single trade", f"{top['Worst %']}%" if pd.notna(top["Worst %"]) else "n/a")

    st.markdown("**Results by instrument**")
    st.dataframe(summary, width="stretch", hide_index=True, height=420)

    view = st.selectbox("Equity curve and trade log for", ["ALL"] + sorted(trades["Ticker"].unique()), key="bt_view")
    sub = trades if view == "ALL" else trades[trades["Ticker"] == view]
    closed = sub[sub["Result"] != "open"].sort_values("Exit time").reset_index(drop=True)
    if len(closed):
        curve = pd.DataFrame(
            {"Cumulative return %": closed["Return %"].cumsum().to_numpy()},
            index=pd.RangeIndex(1, len(closed) + 1, name="Trade #"),
        )
        st.line_chart(curve)
    st.dataframe(
        sub.sort_values("Entry time", ascending=False),
        width="stretch",
        hide_index=True,
        height=350,
    )
    st.download_button(
        "Download all trades (CSV)",
        data=trades.to_csv(index=False),
        file_name="rsi_backtest_trades.csv",
        mime="text/csv",
    )
    st.caption(
        "Total return is the sum of per-trade % results with equal size, no leverage or compounding. "
        "Trades without a stop can sit far underwater before the target is hit; check Worst % and Open P/L %."
    )


# ---------- Sidebar ----------
st.sidebar.header("Settings")
watch_text = st.sidebar.text_area("Watchlist (one per line)", DEFAULT_WATCHLIST, height=420)
interval = st.sidebar.selectbox("Timeframe", list(PERIOD_FOR_INTERVAL), index=2)
rsi_period = st.sidebar.number_input("RSI period", 2, 50, 14)
upper = st.sidebar.number_input("Upper level", 50, 99, 80)
lower = st.sidebar.number_input("Lower level", 1, 50, 20)
buffer = st.sidebar.number_input("Alert when within (RSI points)", 0, 20, 5)
refresh = st.sidebar.number_input("Refresh every (seconds)", 10, 600, 30)
sound = st.sidebar.checkbox("Sound on new alert", value=True)
only_alerts = st.sidebar.checkbox("Show alerts only", value=False)

tickers = [t.strip().upper() for t in watch_text.splitlines() if t.strip()]
tickers = list(dict.fromkeys(tickers))

st.title("RSI Alert Scanner")
st.caption(f"Flags anything at or within {buffer} points of RSI {upper} / {lower}. Yahoo data can lag up to ~15 min on some exchanges.")

if "alerted" not in st.session_state:
    st.session_state.alerted = {}
if "log" not in st.session_state:
    st.session_state.log = []


@st.fragment(run_every=int(refresh))
def live_panel():
    if not tickers:
        st.warning("Add at least one ticker.")
        return
    try:
        df = scan(tickers, interval, int(rsi_period), upper, lower, buffer)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return

    # New-alert detection (only fire when a ticker's status changes)
    fired = False
    for _, r in df.iterrows():
        t, s = r["Ticker"], r["Status"]
        prev = st.session_state.alerted.get(t, "neutral")
        if s in ("OVERBOUGHT", "OVERSOLD", "NEAR 80", "NEAR 20") and s != prev:
            msg = f"{t}: {s} (RSI {r['RSI']}, price {r['Price']})"
            st.toast(msg, icon="🚨" if s in ("OVERBOUGHT", "OVERSOLD") else "⚠️")
            st.session_state.log.insert(0, f"{datetime.now():%H:%M:%S}  {msg}")
            fired = True
        st.session_state.alerted[t] = s
    st.session_state.log = st.session_state.log[:100]
    if fired and sound:
        beep()

    hot = df[df["Status"].isin(["OVERBOUGHT", "OVERSOLD", "NEAR 80", "NEAR 20"])]
    c1, c2, c3 = st.columns(3)
    c1.metric("Scanned", len(df))
    c2.metric("Alerts", len(hot))
    c3.metric("Updated", datetime.now().strftime("%H:%M:%S"))

    shown = hot if only_alerts else df
    if shown.empty:
        st.info("Nothing near the levels right now.")
    else:
        st.dataframe(
            shown.style.apply(colour_row, axis=1),
            width="stretch",
            hide_index=True,
            height=min(40 + 35 * len(shown), 700),
        )

    with st.expander("Alert log", expanded=False):
        st.text("\n".join(st.session_state.log) or "No alerts yet.")


tab_live, tab_bt = st.tabs(["Live scanner", "Backtest"])
with tab_live:
    live_panel()
with tab_bt:
    backtest_ui()
