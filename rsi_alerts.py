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
            use_container_width=True,
            hide_index=True,
            height=min(40 + 35 * len(shown), 700),
        )

    with st.expander("Alert log", expanded=False):
        st.text("\n".join(st.session_state.log) or "No alerts yet.")


live_panel()
