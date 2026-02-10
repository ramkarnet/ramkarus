"""
================================================================================
RAMKAR-US v1.2 WR MODE — STREAMLIT WEB TARAMA
================================================================================
Streamlit Cloud'da deploy et, telefondan eriş.

Deploy:
  1. GitHub repo oluştur
  2. Bu dosyayı + requirements.txt yükle
  3. streamlit.io/cloud → "New app" → repo seç → Deploy

================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

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

/* Ana tema */
.stApp { background: #0a0c10; }
section[data-testid="stSidebar"] { background: #12151c; }

/* Font */
html, body, [class*="css"] { 
    font-family: 'DM Sans', sans-serif; 
    color: #e4e8f0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #12151c, #0f1a15);
    border: 1px solid #252a36;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.main-header h1 { 
    font-family: 'DM Sans', sans-serif;
    font-size: 1.5rem; font-weight: 700; color: #06b6d4; margin: 0; 
}
.main-header .sub { font-size: 0.8rem; color: #7a8299; margin-top: 4px; }
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 700;
    padding: 3px 8px; border-radius: 4px;
    letter-spacing: 1px; text-transform: uppercase;
    background: #15803d; color: #22c55e;
    margin-bottom: 8px;
}

/* Sinyal kartları */
.signal-card {
    background: #12151c;
    border: 1px solid #252a36;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.signal-card.rejected {
    border-left-color: #f59e0b;
    opacity: 0.7;
}
.signal-card.watching {
    border-left-color: #3b82f6;
    opacity: 0.6;
}
.signal-card .ticker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem; font-weight: 700; color: #e4e8f0;
}
.signal-card .price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; color: #7a8299;
}
.signal-card .meta {
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;
}
.signal-card .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; padding: 2px 8px;
    border-radius: 4px; background: #1a1e28; color: #7a8299;
}
.tag.green { background: #15803d; color: #22c55e; }
.tag.red { background: #991b1b; color: #ef4444; }
.tag.amber { background: #92400e; color: #f59e0b; }
.tag.blue { background: #1e3a5f; color: #3b82f6; }

/* Reject reason */
.reject-reason {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #f59e0b;
    margin-top: 6px;
}

/* Stat grid */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
    margin: 12px 0;
}
@media (max-width: 768px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
}
.stat-box {
    background: #12151c; border: 1px solid #252a36; border-radius: 8px;
    padding: 12px; text-align: center;
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 700;
}
.stat-label { font-size: 0.7rem; color: #7a8299; margin-top: 4px; }
.green { color: #22c55e; }
.amber { color: #f59e0b; }
.red { color: #ef4444; }
.blue { color: #3b82f6; }
.cyan { color: #06b6d4; }

/* Progress */
.scan-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: #7a8299;
    text-align: center; padding: 40px 20px;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #15803d, #22c55e) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    padding: 12px 24px !important; width: 100% !important;
    font-size: 1rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #15803d) !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🔒 KİLİTLİ AYARLAR
# ==============================================================================
WR_ADX_MIN = 25.0
WR_ADX_MAX = 40.0
WR_MFI_MIN = 60.0
ATR_STOP_P7 = 2.5
P7_TP_PCT = 0.15

ADX_ESIK = 20
HACIM_CARPAN = 1.2
MESAFE_MIN = -2.0
MESAFE_MAX = 30.0
MFI_ESIK = 50

SB_FLAT_BARS = 5
SB_FLAT_PCT = 10.0
SB_ATR_MULT = 0.9

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
# ANALİZ FONKSİYONLARI (scanner'dan birebir)
# ==============================================================================

def compute_stoch_rsi(close_series, period=14, smooth_k=3, smooth_d=3):
    rsi = pd.Series(close_series).diff()
    gain = rsi.clip(lower=0)
    loss = (-rsi.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    
    stoch = ((rsi_val - rsi_val.rolling(period).min()) / 
             (rsi_val.rolling(period).max() - rsi_val.rolling(period).min()).replace(0, np.nan)) * 100
    k = stoch.rolling(smooth_k).mean()
    d_val = k.rolling(smooth_d).mean()
    return k, d_val

def compute_adx(high, low, close_s, period=14):
    tr1 = high - low
    tr2 = abs(high - close_s.shift(1))
    tr3 = abs(low - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.rolling(period).mean()
    return adx_val, plus_di, minus_di

def compute_mfi(high, low, close_s, volume, period=14):
    tp = (high + low + close_s) / 3
    mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, mf, 0), index=close_s.index).rolling(period).sum()
    neg_mf = pd.Series(np.where(tp_diff <= 0, mf, 0), index=close_s.index).rolling(period).sum()
    mr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mr))

def compute_psar(high, low, close_s, af_start=0.02, af_step=0.02, af_max=0.2):
    length = len(close_s)
    psar = np.zeros(length)
    af = af_start
    bull = True
    ep = low.iloc[0]
    psar[0] = high.iloc[0]
    
    for i in range(1, length):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low.iloc[i-1], low.iloc[i-2] if i >= 2 else low.iloc[i-1])
            if low.iloc[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high.iloc[i-1], high.iloc[i-2] if i >= 2 else high.iloc[i-1])
            if high.iloc[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
    return pd.Series(psar, index=close_s.index)

def analyze_stock(symbol, data):
    """Tek hisse analizi — scanner ile birebir aynı mantık"""
    if data is None or len(data) < 50:
        return None
    
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        vol = data['Volume']
        last = data.iloc[-1]
        
        # İndikatörler
        ema20 = close.rolling(20).mean()
        ema50 = close.rolling(50).mean()
        k_val, d_val = compute_stoch_rsi(close)
        adx_val, di_plus, di_minus = compute_adx(high, low, close)
        mfi_val = compute_mfi(high, low, close, vol)
        vol_avg = vol.rolling(20).mean()
        sar = compute_psar(high, low, close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Son değerler
        last_ema20 = ema20.iloc[-1]
        last_k = k_val.iloc[-1]
        last_d = d_val.iloc[-1]
        last_adx = adx_val.iloc[-1]
        last_di_plus = di_plus.iloc[-1]
        last_di_minus = di_minus.iloc[-1]
        last_mfi = mfi_val.iloc[-1]
        last_vol_avg = vol_avg.iloc[-1]
        last_sar = sar.iloc[-1]
        last_atr = atr.iloc[-1]
        
        if pd.isna(last_ema20) or pd.isna(last_adx) or pd.isna(last_mfi):
            return None
        
        ema_dist = (last['Close'] - last_ema20) / last_ema20 * 100
        vol_ratio = last['Volume'] / last_vol_avg if last_vol_avg > 0 else 0
        atr_pct = (last_atr / last['Close']) * 100
        
        # CMF
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 0.0001)
        mfv = mfm * vol
        cmf = mfv.rolling(20).mean() / vol.rolling(20).mean()
        last_cmf = cmf.iloc[-1]
        
        # OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        
        # 6 Kriter
        k1 = (last['Close'] > last_ema20) and (last_k > last_d)
        k2 = (last_adx >= ADX_ESIK) and (last_di_plus > last_di_minus)
        k3 = (last['Volume'] >= last_vol_avg * HACIM_CARPAN)
        k4 = (last['Close'] > last_sar)
        k5 = (ema_dist >= MESAFE_MIN) and (ema_dist <= MESAFE_MAX)
        k6 = (last_mfi > MFI_ESIK)
        
        score = sum([k1, k2, k3, k4, k5, k6])
        
        # Sessiz Birikim
        atr_avg_val = atr.rolling(20).mean()
        sb_atr = last_atr < (atr_avg_val.iloc[-1] * SB_ATR_MULT) if not pd.isna(atr_avg_val.iloc[-1]) else False
        
        p_high = close.tail(SB_FLAT_BARS).max()
        p_low = close.tail(SB_FLAT_BARS).min()
        p_range = ((p_high - p_low) / p_low) * 100 if p_low > 0 else 100
        sb_flat = p_range < SB_FLAT_PCT
        
        sb_cmf = last_cmf > 0 if not pd.isna(last_cmf) else False
        obv_change = obv.iloc[-1] - obv.iloc[-5] if len(obv) > 5 else 0
        sb_obv = obv_change > 0
        
        sb_count = sum([sb_atr, sb_flat, sb_cmf, sb_obv])
        if sb_atr and sb_flat and sb_cmf and sb_obv:
            sb_points = 4
        elif sb_flat and (sb_cmf or sb_obv):
            sb_points = 3
        elif sb_count >= 2:
            sb_points = 2
        else:
            sb_points = 1
        
        sb_label = {4: "PREM", 3: "STRONG", 2: "NORM", 1: "WEAK"}[sb_points]
        
        # Priority
        priority = 3 + sb_points
        
        # WR Gate
        wr_adx_ok = WR_ADX_MIN <= last_adx <= WR_ADX_MAX
        wr_mfi_ok = last_mfi > WR_MFI_MIN
        wr_pass = (score == 6) and (priority >= 7) and wr_adx_ok and wr_mfi_ok
        
        # Reject reason
        wr_reasons = []
        if score == 6 and priority >= 6:
            if priority < 7:
                wr_reasons.append("P6 → WR'da YOK")
            if not wr_adx_ok:
                wr_reasons.append(f"ADX {last_adx:.1f} ({'<25' if last_adx < WR_ADX_MIN else '>40'})")
            if not wr_mfi_ok:
                wr_reasons.append(f"MFI {last_mfi:.0f} ≤60")
        
        # Stop/Target
        stop_price = last['Close'] - (last_atr * ATR_STOP_P7)
        stop_pct = ((last['Close'] - stop_price) / last['Close']) * 100
        target_price = last['Close'] * (1 + P7_TP_PCT)
        
        # Fiyat değişimleri
        price_1d = ((last['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
        price_5d = ((last['Close'] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100 if len(data) > 5 else 0
        
        return {
            'symbol': symbol,
            'price': round(last['Close'], 2),
            'score': score,
            'priority': priority,
            'sb_label': sb_label,
            'adx': round(last_adx, 1),
            'mfi': round(last_mfi, 1),
            'cmf': round(last_cmf * 100, 1),
            'ema_dist': round(ema_dist, 1),
            'vol_ratio': round(vol_ratio, 2),
            'atr_pct': round(atr_pct, 2),
            'stop': round(stop_price, 2),
            'stop_pct': round(stop_pct, 1),
            'target': round(target_price, 2),
            'sar': round(last_sar, 2),
            'price_1d': round(price_1d, 1),
            'price_5d': round(price_5d, 1),
            'wr_pass': wr_pass,
            'wr_reasons': wr_reasons,
            'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6,
        }
    except Exception as e:
        return None

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="badge">WR MODE v1.2</div>
        <h1>RAMKAR-US</h1>
        <div class="sub">P7 + ADX 25-40 + MFI>60 | BT: 21T WR %76 Sharpe 0.36</div>
    </div>
    """, unsafe_allow_html=True)

def render_stats(wr_count, rejected_count, watching_count, total):
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-num green">{wr_count}</div>
            <div class="stat-label">WR RADAR</div>
        </div>
        <div class="stat-box">
            <div class="stat-num amber">{rejected_count}</div>
            <div class="stat-label">ELENEN</div>
        </div>
        <div class="stat-box">
            <div class="stat-num blue">{watching_count}</div>
            <div class="stat-label">İZLEME</div>
        </div>
        <div class="stat-box">
            <div class="stat-num cyan">{total}</div>
            <div class="stat-label">TARANAN</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_signal_card(r, card_type="signal"):
    css_class = "" if card_type == "signal" else ("rejected" if card_type == "rejected" else "watching")
    
    # Kriterler
    criteria = "".join(["✓" if c else "✗" for c in [r['k1'], r['k2'], r['k3'], r['k4'], r['k5'], r['k6']]])
    
    # ADX renk
    adx_class = "green" if WR_ADX_MIN <= r['adx'] <= WR_ADX_MAX else "red"
    # MFI renk
    mfi_class = "green" if r['mfi'] > WR_MFI_MIN else ("amber" if r['mfi'] > MFI_ESIK else "red")
    # Stop renk
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
        </div>
        {reject_html}
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN APP
# ==============================================================================

render_header()

# Tarama butonu
if st.button("🔍 TARAMAYI BAŞLAT", use_container_width=True):
    
    progress = st.progress(0, text="Veriler indiriliyor...")
    status = st.empty()
    
    # Batch download
    try:
        status.markdown('<div class="scan-status">📡 170+ hisse indiriliyor...</div>', unsafe_allow_html=True)
        all_data = yf.download(SPUS_SHARIAH_STOCKS, period="3mo", group_by='ticker', progress=False, threads=True)
        progress.progress(40, text="Analiz ediliyor...")
    except Exception as e:
        st.error(f"İndirme hatası: {e}")
        st.stop()
    
    # Analiz
    results = []
    total = len(SPUS_SHARIAH_STOCKS)
    
    for i, symbol in enumerate(SPUS_SHARIAH_STOCKS):
        try:
            if symbol in all_data.columns.get_level_values(0):
                sym_data = all_data[symbol].dropna()
                result = analyze_stock(symbol, sym_data)
                if result:
                    results.append(result)
        except:
            pass
        
        if i % 20 == 0:
            pct = 40 + int((i / total) * 50)
            progress.progress(pct, text=f"Analiz: {i}/{total}")
    
    progress.progress(95, text="Sınıflandırılıyor...")
    
    # Sınıflandır
    results.sort(key=lambda r: (-r['priority'], -r['score'], -r['adx'], -r['mfi']))
    
    wr_radar = [r for r in results if r['wr_pass']]
    wr_rejected = [r for r in results if r['score'] == 6 and r['priority'] >= 6 and not r['wr_pass']]
    watching = [r for r in results if r['score'] == 5 and r['priority'] >= 6]
    
    progress.progress(100, text="Tamamlandı!")
    time.sleep(0.5)
    progress.empty()
    status.empty()
    
    # Sonuçları session'a kaydet
    st.session_state['results'] = results
    st.session_state['wr_radar'] = wr_radar
    st.session_state['wr_rejected'] = wr_rejected
    st.session_state['watching'] = watching
    st.session_state['scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')

# Sonuçları göster
if 'results' in st.session_state:
    wr_radar = st.session_state['wr_radar']
    wr_rejected = st.session_state['wr_rejected']
    watching = st.session_state['watching']
    results = st.session_state['results']
    
    st.markdown(f"<div style='text-align:center;font-size:0.75rem;color:#4a5168;margin-bottom:12px'>Son tarama: {st.session_state['scan_time']}</div>", unsafe_allow_html=True)
    
    render_stats(len(wr_radar), len(wr_rejected), len(watching), len(results))
    
    # WR RADAR
    if wr_radar:
        st.markdown("### 🏆 WR RADAR — İşlem Al")
        for r in wr_radar:
            render_signal_card(r, "signal")
    else:
        st.markdown("""
        <div style="text-align:center;padding:30px;background:#12151c;border:1px solid #252a36;border-radius:8px;margin:12px 0">
            <div style="font-size:2rem">🔇</div>
            <div style="color:#7a8299;font-size:0.9rem;margin-top:8px">Bugün WR sinyali yok — sabır!</div>
            <div style="color:#4a5168;font-size:0.75rem;margin-top:4px">Ayda 1-2 sinyal normal</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ELENENLER
    if wr_rejected:
        with st.expander(f"⚠ Elenenler ({len(wr_rejected)}) — 6/6 ama WR filtresi dışı"):
            for r in wr_rejected:
                render_signal_card(r, "rejected")
    
    # İZLEME
    if watching:
        with st.expander(f"👁 İzleme ({len(watching)}) — 5/6 skorlu yakın adaylar"):
            for r in watching[:15]:  # İlk 15
                render_signal_card(r, "watching")

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:3rem;margin-bottom:16px">🎯</div>
        <div style="color:#7a8299;font-size:1rem">Taramayı başlatmak için butona bas</div>
        <div style="color:#4a5168;font-size:0.8rem;margin-top:8px">170+ Shariah hisse • WR Mode filtre • ~60 saniye</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #252a36;font-size:0.65rem;color:#4a5168;font-family:'JetBrains Mono',monospace">
    RAMKAR-US WR v1.2 | ⚠ Yatırım tavsiyesi değildir
</div>
""", unsafe_allow_html=True)
