"""
================================================================================
RAMKAR-US v1.2 WR MODE — STREAMLIT WEB TARAMA
================================================================================
Scanner (ramkar_us_v12_wr_scanner.py) ile BİREBİR AYNI hesaplama.
Wilder RMA, aynı ADX, aynı StochRSI, aynı SAR, aynı SB.

Deploy:
  1. GitHub repo → bu dosya + requirements.txt
  2. streamlit.io/cloud → "New app" → Deploy
================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')

# ==============================================================================
# SAYFA AYARLARI
# ==============================================================================
st.set_page_config(
    page_title="RAMKAR-US WR",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# KOYU TEMA + MOBİL CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp { background: #0a0c10; }
section[data-testid="stSidebar"] { background: #12151c; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #e4e8f0; }

.main-header {
    background: linear-gradient(135deg, #12151c, #0f1a15);
    border: 1px solid #252a36; border-radius: 12px;
    padding: 20px; margin-bottom: 20px;
}
.main-header h1 { font-family: 'DM Sans'; font-size: 1.5rem; font-weight: 700; color: #06b6d4; margin: 0; }
.main-header .sub { font-size: 0.8rem; color: #7a8299; margin-top: 4px; }
.badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 700; padding: 3px 8px; border-radius: 4px;
    letter-spacing: 1px; text-transform: uppercase;
    background: #15803d; color: #22c55e; margin-bottom: 8px;
}

.signal-card {
    background: #12151c; border: 1px solid #252a36;
    border-left: 3px solid #22c55e; border-radius: 8px;
    padding: 16px; margin-bottom: 12px;
}
.signal-card.rejected { border-left-color: #f59e0b; opacity: 0.7; }
.signal-card.watching { border-left-color: #3b82f6; opacity: 0.6; }
.signal-card .ticker { font-family: 'JetBrains Mono'; font-size: 1.1rem; font-weight: 700; color: #e4e8f0; }
.signal-card .price { font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #7a8299; }
.signal-card .meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.signal-card .tag {
    font-family: 'JetBrains Mono'; font-size: 0.7rem; padding: 2px 8px;
    border-radius: 4px; background: #1a1e28; color: #7a8299;
}
.tag.green { background: #15803d; color: #22c55e; }
.tag.red { background: #991b1b; color: #ef4444; }
.tag.amber { background: #92400e; color: #f59e0b; }
.tag.blue { background: #1e3a5f; color: #3b82f6; }
.reject-reason { font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #f59e0b; margin-top: 6px; }

.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }
@media (max-width: 768px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
.stat-box { background: #12151c; border: 1px solid #252a36; border-radius: 8px; padding: 12px; text-align: center; }
.stat-num { font-family: 'JetBrains Mono'; font-size: 1.4rem; font-weight: 700; }
.stat-label { font-size: 0.7rem; color: #7a8299; margin-top: 4px; }
.green { color: #22c55e; } .amber { color: #f59e0b; } .red { color: #ef4444; }
.blue { color: #3b82f6; } .cyan { color: #06b6d4; }

#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stButton > button {
    background: linear-gradient(135deg, #15803d, #22c55e) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    padding: 12px 24px !important; width: 100% !important; font-size: 1rem !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #22c55e, #15803d) !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🔒 KİLİTLİ AYARLAR (scanner ile birebir)
# ==============================================================================
WR_ADX_MIN = 25.0
WR_ADX_MAX = 40.0
WR_MFI_MIN = 60.0
ATR_STOP_P7 = 2.5
P7_TP_PCT = 0.15
WR_MODE = True

ADX_ESIK = 20
HACIM_CARPAN = 1.2
MESAFE_MIN = -2.0
MESAFE_MAX = 30.0
MFI_ESIK = 50

SB_FLAT_BARS = 5
SB_FLAT_PCT = 10.0
SB_ATR_MULT = 0.9

MAX_RETRY = 3
RETRY_DELAY = 2

SPUS_SHARIAH_STOCKS = [
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AVGO', 'TSLA', 'LLY', 'XOM', 'JNJ', 'MU',
    'HD', 'ABBV', 'PG', 'AMD', 'CSCO', 'MRK', 'LRCX', 'ORCL', 'AMAT', 'PEP',
    'LIN', 'TMO', 'TXN', 'GEV', 'CRM', 'ABT', 'TJX', 'KLAC', 'ISRG',
    'QCOM', 'ADI', 'UBER', 'LOW', 'ACN', 'UNP', 'BKNG', 'DHR', 'COP', 'ANET',
    'MDT', 'SYK', 'BSX', 'MCK', 'REGN', 'EW', 'IDXX', 'RMD', 'BDX', 'DXCM',
    'HOLX', 'COO', 'STE', 'BIIB', 'INCY', 'PODD', 'ALGN', 'CRL', 'TECH', 'RVTY',
    'TT', 'EMR', 'MMM', 'ITW', 'JCI', 'ROK', 'CMI', 'DOV', 'IR', 'OTIS',
    'XYL', 'PNR', 'HUBB', 'FIX', 'WAB', 'NDSN', 'IEX', 'AOS', 'ALLE',
    'NKE', 'ORLY', 'TGT', 'ROST', 'AZO', 'MNST', 'TSCO', 'ULTA', 'BBY', 'WSM',
    'DECK', 'LULU', 'RL', 'TPR', 'POOL', 'GPC',
    'SLB', 'EOG', 'DVN', 'HAL', 'BKR', 'CTRA', 'FCX', 'NEM', 'NUE', 'STLD',
    'VMC', 'MLM', 'APD', 'ECL', 'SHW', 'PPG', 'ALB', 'CF', 'DD',
    'CAT', 'DE', 'FTV', 'GE', 'HON', 'ETN', 'PH', 'AME', 'ROP',
    'ADBE', 'NOW', 'PANW', 'CRWD', 'SNPS', 'CDNS', 'ADSK', 'FTNT', 'WDAY',
    'FICO', 'PTC', 'TYL', 'IT', 'AKAM', 'GDDY', 'EPAM', 'ZBRA', 'TRMB', 'TTD',
    'NXPI', 'MCHP', 'ON', 'SWKS', 'TER', 'MPWR', 'FSLR', 'SMCI',
    'PLD', 'WELL', 'EQIX', 'AVB', 'MAA', 'CPT',
    'CSX', 'NSC', 'ODFL', 'EXPD', 'CHRW', 'JBHT', 'UPS',
    'CL', 'MDLZ', 'KMB', 'HSY', 'MKC', 'CHD', 'CLX', 'ADM',
    'CEG', 'DASH', 'PWR', 'CTAS', 'RSG', 'WM', 'FAST', 'GWW', 'CTVA', 'CARR',
    'DHI', 'PHM', 'BLDR', 'EXPE', 'EBAY', 'CPRT', 'GRMN', 'NTAP', 'PKG', 'WY',
    'TPL', 'VRSN', 'CDW', 'ROL', 'AVY', 'FFIV', 'WST', 'LII', 'JBL', 'MTD',
    'GLW', 'STX', 'A', 'KVUE', 'VLTO', 'LH', 'WAT', 'CTSH', 'ROP', 'GNRC',
]

# ==============================================================================
# İNDİKATÖR FONKSİYONLARI — scanner'dan BİREBİR KOPYA
# ==============================================================================

def wilder_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_stoch_rsi(data, period=14, smooth_k=3, smooth_d=3):
    close = data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / (avg_loss + 0.0001)
    rsi = 100 - (100 / (1 + rs))
    stoch = ((rsi - rsi.rolling(period).min()) /
             (rsi.rolling(period).max() - rsi.rolling(period).min() + 0.0001)) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = wilder_rma(tr, period)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    plus_di = 100 * wilder_rma(plus_dm, period) / (atr + 0.0001)
    minus_di = 100 * wilder_rma(minus_dm, period) / (atr + 0.0001)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 0.0001)
    adx = wilder_rma(dx, period)
    return adx, plus_di, minus_di, atr

def calculate_mfi(data, period=14):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    mf = tp * data['Volume']
    pos_flow = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_flow = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfi = 100 - (100 / (1 + pos_flow / (neg_flow + 0.0001)))
    return mfi

def calculate_cmf(data, period=20):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    hl_diff = (high - low).replace(0, 1)
    mfm = ((close - low) - (high - close)) / hl_diff
    mfv = mfm * volume
    vol_sum = volume.rolling(period).sum().replace(0, 1)
    return mfv.rolling(period).sum() / vol_sum

def calculate_obv(data):
    close = data['Close']
    volume = data['Volume']
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def calculate_sar(data, af_start=0.02, af_step=0.02, af_max=0.2):
    high = data['High'].values
    low = data['Low'].values
    length = len(data)
    sar = np.zeros(length)
    af = af_start
    uptrend = True
    ep = low[0]
    sar[0] = high[0]

    for i in range(1, length):
        if uptrend:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = min(sar[i], low[i-1])
            if i > 1:
                sar[i] = min(sar[i], low[i-2])
            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = max(sar[i], high[i-1])
            if i > 1:
                sar[i] = max(sar[i], high[i-2])
            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=data.index)

# ==============================================================================
# BATCH İNDİRME — scanner'dan birebir
# ==============================================================================

def download_batch_data(symbols, period="90d", max_retry=MAX_RETRY, chunk_size=40):
    """Chunked download + exponential backoff — 170+ hisseyi 40'lık gruplarda indir."""
    all_data = {}
    errors = []

    # Chunk'lara böl
    chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]

    for chunk_idx, chunk in enumerate(chunks):
        remaining = list(chunk)

        for attempt in range(1, max_retry + 1):
            if not remaining:
                break
            try:
                raw = yf.download(
                    remaining, period=period, interval="1d",
                    group_by='ticker', threads=True, progress=False
                )
                success_this_round = []
                for symbol in remaining:
                    try:
                        if len(remaining) == 1:
                            df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                        else:
                            df = raw[symbol][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                        if len(df) >= 50:
                            all_data[symbol] = df
                            success_this_round.append(symbol)
                    except Exception:
                        pass
                remaining = [s for s in remaining if s not in success_this_round]
            except Exception as e:
                logging.warning("Chunk %d attempt %d failed: %s", chunk_idx, attempt, e)

            if remaining and attempt < max_retry:
                backoff = RETRY_DELAY * (2 ** (attempt - 1))  # exponential: 2, 4, 8...
                time.sleep(backoff)

        errors.extend(remaining)

    return all_data, errors

# ==============================================================================
# ANALİZ — scanner'dan BİREBİR KOPYA
# ==============================================================================

def analyze_stock(symbol, data):
    if len(data) < 50:
        return None
    try:
        ema20 = calculate_ema(data, 20)
        stoch_k, stoch_d = calculate_stoch_rsi(data)
        adx, di_plus, di_minus, atr = calculate_adx(data)
        mfi = calculate_mfi(data)
        sar = calculate_sar(data)
        vol_avg = data['Volume'].rolling(20).mean()
        cmf = calculate_cmf(data)
        obv = calculate_obv(data)

        last = data.iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_stoch_k = stoch_k.iloc[-1]
        last_stoch_d = stoch_d.iloc[-1]
        last_adx = adx.iloc[-1]
        last_di_plus = di_plus.iloc[-1]
        last_di_minus = di_minus.iloc[-1]
        last_mfi = mfi.iloc[-1]
        last_sar = sar.iloc[-1]
        last_vol_avg = vol_avg.iloc[-1]
        last_cmf = cmf.iloc[-1]
        last_atr = atr.iloc[-1]

        if pd.isna(last_ema20) or pd.isna(last_adx) or pd.isna(last_mfi):
            return None

        ema_dist = ((last['Close'] - last_ema20) / last_ema20) * 100
        atr_pct = (last_atr / last['Close']) * 100
        vol_ratio = last['Volume'] / last_vol_avg if last_vol_avg > 0 else 0

        k1 = bool((last['Close'] > last_ema20) and (last_stoch_k > last_stoch_d))
        k2 = bool((last_adx >= ADX_ESIK) and (last_di_plus > last_di_minus))
        k3 = bool(last['Volume'] >= (last_vol_avg * HACIM_CARPAN))
        k4 = bool(last['Close'] > last_sar)
        k5 = bool((ema_dist >= MESAFE_MIN) and (ema_dist <= MESAFE_MAX))
        k6 = bool(last_mfi > MFI_ESIK)

        score = sum([k1, k2, k3, k4, k5, k6])

        # Sessiz Birikim
        atr_avg_val = atr.rolling(20).mean()
        sb_atr = bool(atr.iloc[-2] < (atr_avg_val.iloc[-1] * SB_ATR_MULT)) if not pd.isna(atr_avg_val.iloc[-1]) else False
        price_high = data['Close'].rolling(SB_FLAT_BARS).max()
        price_low = data['Close'].rolling(SB_FLAT_BARS).min()
        price_range_pct = ((price_high - price_low) / price_low) * 100
        sb_flat = bool(price_range_pct.iloc[-1] < SB_FLAT_PCT) if not pd.isna(price_range_pct.iloc[-1]) else False
        sb_cmf = bool(last_cmf > 0) if not pd.isna(last_cmf) else False
        obv_change = obv.iloc[-1] - obv.iloc[-SB_FLAT_BARS] if len(obv) > SB_FLAT_BARS else 0
        sb_obv = bool(obv_change > 0)

        sb_count = sum([sb_atr, sb_flat, sb_cmf, sb_obv])
        if sb_atr and sb_flat and sb_cmf and sb_obv:
            sb_points, sb_label = 4, "PREM"
        elif sb_flat and (sb_cmf or sb_obv):
            sb_points, sb_label = 3, "STRONG"
        elif sb_count >= 2:
            sb_points, sb_label = 2, "NORM"
        else:
            sb_points, sb_label = 1, "WEAK"

        priority = 3 + sb_points

        # WR Gate
        wr_adx_ok = WR_ADX_MIN <= last_adx <= WR_ADX_MAX
        wr_mfi_ok = last_mfi > WR_MFI_MIN
        wr_pass = (score == 6) and (priority >= 7) and wr_adx_ok and wr_mfi_ok

        wr_reasons = []
        if score == 6 and priority >= 6:
            if priority < 7:
                wr_reasons.append("P6 → WR'da YOK")
            if not wr_adx_ok:
                wr_reasons.append(f"ADX {last_adx:.1f} ({'<25' if last_adx < WR_ADX_MIN else '>40'})")
            if not wr_mfi_ok:
                wr_reasons.append(f"MFI {last_mfi:.0f} ≤ 60")

        # Katman (scanner ile birebir)
        if wr_pass:
            katman = "WR_RADAR"
        elif score == 6 and priority >= 6:
            katman = "WR_ELENDI"
        elif score >= 4:
            katman = "IZLEME"
        else:
            katman = "DIGER"

        stop_price = last['Close'] - (last_atr * ATR_STOP_P7)
        stop_pct = ((last['Close'] - stop_price) / last['Close']) * 100
        target_price = last['Close'] * (1 + P7_TP_PCT)

        price_1d = ((last['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
        price_5d = ((last['Close'] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100 if len(data) > 5 else 0

        return {
            'symbol': symbol,
            'price': round(float(last['Close']), 2),
            'score': int(score),
            'priority': int(priority),
            'sb_label': sb_label,
            'adx': round(float(last_adx), 1),
            'mfi': round(float(last_mfi), 1),
            'cmf': round(float(last_cmf * 100), 1),
            'ema_dist': round(float(ema_dist), 1),
            'vol_ratio': round(float(vol_ratio), 2),
            'atr_pct': round(float(atr_pct), 2),
            'stop': round(float(stop_price), 2),
            'stop_pct': round(float(stop_pct), 1),
            'target': round(float(target_price), 2),
            'sar': round(float(last_sar), 2),
            'price_1d': round(float(price_1d), 1),
            'price_5d': round(float(price_5d), 1),
            'wr_pass': wr_pass,
            'wr_reasons': wr_reasons,
            'katman': katman,
            'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6,
        }
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)  # 5 dakika cache
def cached_download(period="90d"):
    """Cache'li indirme — aynı 5dk içinde tekrar basarsan yeniden indirmez."""
    return download_batch_data(SPUS_SHARIAH_STOCKS, period=period)

# ==============================================================================
# ==============================================================================

def sort_key(r):
    adx = r['adx']
    if 25 <= adx <= 35:
        adx_quality = 3
    elif 20 <= adx < 25:
        adx_quality = 2
    elif 35 < adx <= 45:
        adx_quality = 1
    else:
        adx_quality = 0
    return (-r['priority'], -r['score'], -adx_quality, -r['mfi'])

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="badge">WR MODE v1.2</div>
        <h1>RAMKAR-US</h1>
        <div class="sub">P7 + ADX 25-40 + MFI&gt;60 | BT: 21T WR %76 Sharpe 0.36</div>
    </div>
    """, unsafe_allow_html=True)

def render_stats(wr_count, rejected_count, watching_count, total):
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box"><div class="stat-num green">{wr_count}</div><div class="stat-label">WR RADAR</div></div>
        <div class="stat-box"><div class="stat-num amber">{rejected_count}</div><div class="stat-label">ELENEN</div></div>
        <div class="stat-box"><div class="stat-num blue">{watching_count}</div><div class="stat-label">İZLEME</div></div>
        <div class="stat-box"><div class="stat-num cyan">{total}</div><div class="stat-label">TARANAN</div></div>
    </div>
    """, unsafe_allow_html=True)

def render_signal_card(r, card_type="signal"):
    css_class = "" if card_type == "signal" else ("rejected" if card_type == "rejected" else "watching")
    criteria = "".join(["✓" if c else "✗" for c in [r['k1'], r['k2'], r['k3'], r['k4'], r['k5'], r['k6']]])
    adx_class = "green" if WR_ADX_MIN <= r['adx'] <= WR_ADX_MAX else "red"
    mfi_class = "green" if r['mfi'] > WR_MFI_MIN else ("amber" if r['mfi'] > MFI_ESIK else "red")
    stop_class = "green" if r['stop_pct'] <= 5 else ("amber" if r['stop_pct'] <= 7 else "red")

    reject_html = ""
    if r['wr_reasons']:
        reject_html = f'<div class="reject-reason">⚠ {" | ".join(r["wr_reasons"])}</div>'

    target_html = f'<span class="tag green">TP ${r["target"]}</span>' if card_type == "signal" else ""

    st.markdown(f"""
    <div class="signal-card {css_class}">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="ticker">{r['symbol']}</span>
            <span class="price">${r['price']}</span>
        </div>
        <div class="meta">
            <span class="tag">P{r['priority']}</span>
            <span class="tag {'green' if r['sb_label']=='PREM' else 'blue'}">{r['sb_label']}</span>
            <span class="tag {adx_class}">ADX {r['adx']}</span>
            <span class="tag {mfi_class}">MFI {r['mfi']}</span>
            <span class="tag {stop_class}">Stop {r['stop_pct']}%</span>
            <span class="tag">ATR {r['atr_pct']}%</span>
            {target_html}
        </div>
        <div class="meta" style="margin-top:4px">
            <span class="tag">{criteria}</span>
            <span class="tag">1D: {'+' if r['price_1d']>=0 else ''}{r['price_1d']}%</span>
            <span class="tag">5D: {'+' if r['price_5d']>=0 else ''}{r['price_5d']}%</span>
            <span class="tag">EMA {r['ema_dist']}%</span>
            <span class="tag">Vol {r['vol_ratio']}x</span>
        </div>
        {reject_html}
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# ANA UYGULAMA
# ==============================================================================

render_header()

if st.button("🔍 TARAMAYI BAŞLAT", use_container_width=True):

    progress = st.progress(0, text="Veriler indiriliyor...")
    status = st.empty()

    status.markdown('<div style="text-align:center;color:#7a8299;padding:20px;font-family:JetBrains Mono;font-size:0.8rem">📡 170+ hisse indiriliyor (retry destekli)...</div>', unsafe_allow_html=True)

    all_data, download_errors = cached_download()
    progress.progress(40, text=f"✅ {len(all_data)} hisse indirildi | Analiz ediliyor...")

    results = []
    analysis_errors = []
    total = len(all_data)

    for i, (symbol, data) in enumerate(all_data.items()):
        try:
            result = analyze_stock(symbol, data)
            if result:
                results.append(result)
        except Exception as e:
            logging.exception("analyze_stock failed for %s", symbol)
            analysis_errors.append(symbol)
        if i % 20 == 0:
            pct = 40 + int((i / max(total, 1)) * 55)
            progress.progress(min(pct, 95), text=f"Analiz: {i}/{total}")

    progress.progress(98, text="Sınıflandırılıyor...")

    results.sort(key=sort_key)

    wr_radar = [r for r in results if r.get('katman') == 'WR_RADAR']
    wr_rejected = [r for r in results if r.get('katman') == 'WR_ELENDI']
    watching = [r for r in results if r['score'] == 5 and r['priority'] >= 6]

    progress.progress(100, text="Tamamlandı!")
    time.sleep(0.3)
    progress.empty()
    status.empty()

    st.session_state['results'] = results
    st.session_state['wr_radar'] = wr_radar
    st.session_state['wr_rejected'] = wr_rejected
    st.session_state['watching'] = watching
    st.session_state['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    st.session_state['download_errors'] = download_errors
    st.session_state['analysis_errors'] = analysis_errors

# Sonuçları göster
if 'results' in st.session_state:
    wr_radar = st.session_state['wr_radar']
    wr_rejected = st.session_state['wr_rejected']
    watching = st.session_state['watching']
    results = st.session_state['results']
    dl_errors = st.session_state.get('download_errors', [])
    an_errors = st.session_state.get('analysis_errors', [])

    scan_info = f"Son tarama: {st.session_state['scan_time']} | {len(results)} hisse analiz edildi"
    total_errors = len(dl_errors) + len(an_errors)
    if total_errors:
        scan_info += f" | ⚠ {total_errors} hata ({len(dl_errors)} indirme, {len(an_errors)} analiz)"
    st.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#4a5168;margin-bottom:12px;font-family:JetBrains Mono'>{scan_info}</div>", unsafe_allow_html=True)

    render_stats(len(wr_radar), len(wr_rejected), len(watching), len(results))

    if wr_radar:
        st.markdown("### 🏆 WR RADAR — İşlem Al")
        for r in wr_radar:
            render_signal_card(r, "signal")
    else:
        st.markdown("""
        <div style="text-align:center;padding:30px;background:#12151c;border:1px solid #252a36;border-radius:8px;margin:12px 0">
            <div style="font-size:2rem">🔇</div>
            <div style="color:#7a8299;font-size:0.9rem;margin-top:8px">Bugün WR sinyali yok — sabır!</div>
            <div style="color:#4a5168;font-size:0.75rem;margin-top:4px">Ayda ~1.8 sinyal bekleniyor</div>
        </div>
        """, unsafe_allow_html=True)

    if wr_rejected:
        with st.expander(f"⚠ Elenenler ({len(wr_rejected)}) — 6/6 ama WR filtresi dışı"):
            for r in wr_rejected:
                render_signal_card(r, "rejected")

    if watching:
        with st.expander(f"👁 İzleme ({len(watching)}) — 5/6 skorlu yakın adaylar"):
            for r in watching[:20]:
                render_signal_card(r, "watching")

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:3rem;margin-bottom:16px">🎯</div>
        <div style="color:#7a8299;font-size:1rem">Taramayı başlatmak için butona bas</div>
        <div style="color:#4a5168;font-size:0.8rem;margin-top:8px">170+ Shariah hisse • WR Mode • Wilder RMA • ~60 saniye</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #252a36;font-size:0.65rem;color:#4a5168;font-family:'JetBrains Mono',monospace">
    RAMKAR-US WR v1.2 | Scanner ile birebir senkron | ⚠ Yatırım tavsiyesi değildir
</div>
""", unsafe_allow_html=True)
