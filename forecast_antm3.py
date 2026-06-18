import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# ─────────────────────────────────────────────
# UTIL — define FIRST so available everywhere
# ─────────────────────────────────────────────
def hex_to_rgba(hex_str, alpha=1.0):
    h = hex_str.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ANTM Forecast",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS — Light Minimalist Elegant
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@300;400;600;700;800&family=Sora:wght@400;600;700&display=swap');

:root {
  --bg:        #F5F6FA;
  --surface:   #FFFFFF;
  --surface2:  #F0F2F8;
  --border:    #E2E5EF;
  --accent:    #3B6FE8;
  --accent2:   #5B8AFF;
  --accent-lt: #EEF2FF;
  --gold:      #E8A838;
  --gold-lt:   #FEF6E7;
  --green:     #18B57A;
  --green-lt:  #E6FAF3;
  --red:       #E85555;
  --red-lt:    #FEF0F0;
  --purple:    #8B5CF6;
  --purple-lt: #F3EFFE;
  --teal:      #0EA5A0;
  --teal-lt:   #E6FAF9;
  --text:      #1A1D2E;
  --subtext:   #6B7280;
  --caption:   #9CA3AF;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--text);
  font-family: 'Nunito Sans', sans-serif;
}
[data-testid="stSidebar"] {
  background-color: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
section.main > div { padding-top: 1.4rem; }

/* ── Header ── */
.app-header {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.6rem;
  display: flex;
  align-items: center;
  gap: 1.4rem;
}
.header-icon {
  width: 56px; height: 56px;
  background: var(--accent-lt);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.7rem;
  flex-shrink: 0;
}
.header-text h1 {
  font-family: 'Sora', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text);
  margin: 0;
  letter-spacing: -0.3px;
}
.header-text p {
  color: var(--subtext);
  font-size: 0.88rem;
  margin: 3px 0 0;
}
.header-badge {
  margin-left: auto;
  background: var(--green-lt);
  color: var(--green);
  font-size: 0.75rem;
  font-weight: 700;
  padding: 5px 12px;
  border-radius: 20px;
  border: 1px solid rgba(24,181,122,0.25);
}

/* ── Info Cards ── */
.info-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 18px;
  display: flex;
  align-items: flex-start;
  gap: 14px;
  transition: box-shadow 0.2s, transform 0.2s;
  height: 100%;
}
.info-card:hover {
  box-shadow: 0 6px 24px rgba(59,111,232,0.09);
  transform: translateY(-2px);
}
.info-card .ic-icon {
  width: 40px; height: 40px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem; flex-shrink: 0;
}
.ic-blue   { background: var(--accent-lt); }
.ic-gold   { background: var(--gold-lt); }
.ic-green  { background: var(--green-lt); }
.info-card h4 {
  font-family: 'Sora', sans-serif;
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 4px;
}
.info-card p { color: var(--subtext); font-size: 0.8rem; margin: 0; line-height: 1.5; }

/* ── Stat Mini Cards ── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 1.2rem;
}
.scard {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  position: relative;
  overflow: hidden;
}
.scard::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: 12px 12px 0 0;
}
.scard-blue::before  { background: var(--accent); }
.scard-gold::before  { background: var(--gold); }
.scard-green::before { background: var(--green); }
.scard-red::before   { background: var(--red); }
.scard-purple::before{ background: var(--purple); }
.scard-teal::before  { background: var(--teal); }
.scard .s-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--caption);
  margin-bottom: 4px;
}
.scard .s-value {
  font-family: 'Sora', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text);
}
.scard .s-sub {
  font-size: 0.73rem;
  color: var(--subtext);
  margin-top: 2px;
}

/* ── Desc Stat Cards Grid ── */
.desc-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-bottom: 1.2rem;
}
.dcard {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 13px 15px;
}
.dcard .d-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  color: var(--caption);
  margin-bottom: 5px;
}
.dcard .d-val {
  font-family: 'Sora', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
}
.dcard .d-sub {
  font-size: 0.7rem;
  color: var(--subtext);
  margin-top: 2px;
}
.dcard-blue   { border-top: 3px solid var(--accent);  }
.dcard-gold   { border-top: 3px solid var(--gold);    }
.dcard-green  { border-top: 3px solid var(--green);   }
.dcard-red    { border-top: 3px solid var(--red);     }
.dcard-purple { border-top: 3px solid var(--purple);  }
.dcard-teal   { border-top: 3px solid var(--teal);    }
.dcard-gray   { border-top: 3px solid var(--caption); }
.col-blue   { color: var(--accent);  }
.col-gold   { color: var(--gold);    }
.col-green  { color: var(--green);   }
.col-red    { color: var(--red);     }
.col-purple { color: var(--purple);  }
.col-teal   { color: var(--teal);    }
.col-gray   { color: var(--subtext); }

/* ── Section Title ── */
.sec-title {
  font-family: 'Sora', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
  margin: 1.8rem 0 0.9rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sec-title .st-dot {
  width: 8px; height: 8px;
  background: var(--accent);
  border-radius: 50%;
}
.sec-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
  margin-left: 6px;
}

/* ── Forecast Control Box ── */
.control-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 22px;
  margin-bottom: 1.2rem;
}

/* ── Period Radio Pills ── */
div[data-testid="stRadio"] > div {
  gap: 8px !important;
}
div[data-testid="stRadio"] label {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 6px 14px !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  color: var(--subtext) !important;
  cursor: pointer;
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: var(--accent-lt) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--surface) !important;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--subtext) !important;
  border-radius: 8px;
  font-family: 'Nunito Sans', sans-serif;
  font-size: 0.87rem;
  font-weight: 700;
  padding: 8px 20px;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #fff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Nunito Sans', sans-serif !important;
  font-size: 0.9rem !important;
  padding: 0.55rem 1.6rem !important;
  transition: opacity 0.2s, transform 0.1s !important;
  box-shadow: 0 2px 12px rgba(59,111,232,0.2) !important;
}
.stButton > button:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  background: var(--accent-lt) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(59,111,232,0.3) !important;
  border-radius: 10px !important;
  font-family: 'Nunito Sans', sans-serif !important;
  font-weight: 700 !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
div[data-testid="metric-container"] label {
  color: var(--caption) !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 700 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* ── Streamlit misc ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
.stSelectbox label, .stSlider label, div[data-testid="stNumberInput"] label {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}
footer { visibility: hidden; }

/* ── Sidebar ── */
.guide-step {
  background: var(--surface2);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
  padding: 10px 14px;
  margin-bottom: 10px;
}
.guide-step .step-num {
  font-size: 0.65rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--accent);
  margin-bottom: 2px;
}
.guide-step p {
  color: var(--subtext) !important;
  font-size: 0.8rem !important;
  margin: 0 !important;
  line-height: 1.45 !important;
}

/* ── Footer ── */
.app-footer {
  text-align: center;
  color: var(--caption);
  font-size: 0.74rem;
  margin-top: 3rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        from tensorflow.keras.models import load_model
        model  = load_model("model_lstm_antm.h5", compile=False)
        scaler = joblib.load("scaler_antm.pkl")
        return model, scaler, True
    except Exception:
        return None, None, False

model, scaler, model_loaded = load_artifacts()

# ─────────────────────────────────────────────
# 4. DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    df = yf.download("ANTM.JK", start="2020-01-01", end="2025-12-30", progress=False)
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────────
# 5. PREDICT
# ─────────────────────────────────────────────
def predict_recursive(model, scaled_input, days):
    inputs = scaled_input.copy().reshape(-1)
    predictions = []
    for _ in range(days):
        window = inputs[-60:].reshape(1, 60, 1)
        pred   = model.predict(window, verbose=0)
        val    = float(pred[0, 0])
        predictions.append(val)
        inputs = np.append(inputs, val)
    return np.array(predictions).reshape(-1, 1)

# ─────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.8rem; text-align:center;'>
      <div style='display:inline-flex; align-items:center; justify-content:center;
                  width:48px; height:48px; background:#EEF2FF; border-radius:12px;
                  font-size:1.5rem; margin-bottom:8px;'>🪙</div>
      <div style='font-family:"Sora",sans-serif; font-size:1.1rem; font-weight:700; color:#1A1D2E;'>ANTM Stock Forecast</div>
      <div style='font-size:0.75rem; color:#9CA3AF; margin-top:2px;'>PT Aneka Tambang Tbk</div>
    </div>
    <hr style='border-color:#E2E5EF; margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:1px; color:#9CA3AF; margin-bottom:0.7rem;'>📖 Panduan Penggunaan</div>", unsafe_allow_html=True)
    steps = [
        ("Langkah 1", "Buka tab <b>Beranda</b> untuk melihat data historis & statistik saham ANTM."),
        ("Langkah 2", "Pindah ke tab <b>Forecast</b> untuk prediksi harga ke depan."),
        ("Langkah 3", "Pilih periode: <b>7 hari</b>, <b>30 hari</b>, atau <b>Custom</b>."),
        ("Langkah 4", "Klik <b>Generate Forecast</b> dan tunggu proses prediksi."),
        ("Langkah 5", "Lihat grafik historis vs prediksi, lalu <b>download CSV</b>."),
    ]
    for title, desc in steps:
        st.markdown(f"""<div class='guide-step'>
          <div class='step-num'>{title}</div><p>{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E2E5EF; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:1px; color:#9CA3AF; margin-bottom:0.7rem;'>⚙️ Tentang Model</div>", unsafe_allow_html=True)
    model_info = [
        ("Arsitektur", "LSTM"),
        ("Look-back", "60 hari"),
        ("Target", "Close Price"),
        ("Sumber Data", "Yahoo Finance"),
        ("Periode", "2020 – 2025"),
        ("MAPE", "~4.7%"),
    ]
    for k, v in model_info:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; padding:5px 0;
                    border-bottom:1px solid #E2E5EF; font-size:0.8rem;'>
          <span style='color:#9CA3AF;'>{k}</span>
          <span style='color:#1A1D2E; font-weight:600;'>{v}</span>
        </div>""", unsafe_allow_html=True)

    sc = ("#18B57A", "E6FAF3", "Model Loaded ✓") if model_loaded else ("#E85555", "FEF0F0", "Model Not Found ✗")
    st.markdown(f"""<div style='margin-top:1rem; text-align:center;
        background:#{sc[1]}; border:1px solid {sc[0]}40;
        border-radius:8px; padding:7px; color:{sc[0]};
        font-size:0.8rem; font-weight:700;'>{sc[2]}</div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E2E5EF; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9CA3AF; font-size:0.72rem; text-align:center;'>© 2026 ANTM Forecast<br></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
  <div class='header-icon'>🪙</div>
  <div class='header-text'>
    <h1>ANTM Stock Forecast</h1>
    <p>Prediksi Harga Saham PT Aneka Tambang Tbk · Model Long Short-Term Memory· Data Yahoo Finance 2020–2025</p>
  </div>
  <div class='header-badge'>Live Data</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. FETCH DATA
# ─────────────────────────────────────────────
with st.spinner("Mengambil data dari Yahoo Finance…"):
    df = get_data()

close = df['Close'].squeeze()

# ─────────────────────────────────────────────
# 9. TABS
# ─────────────────────────────────────────────
tab_beranda, tab_forecast = st.tabs(["🧭  Beranda", "⚡  Forecast"])

# ═══════════════════════════════════════════════════
# TAB BERANDA
# ═══════════════════════════════════════════════════
with tab_beranda:

    # ── Info Cards ──────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='info-card'>
          <div class='ic-icon ic-blue'>📊</div>
          <div><h4>Data-Driven</h4>
          <p>Data historis pergerakan harga saham PT Aneka Tambang Tbk (ANTM) ditarik secara otomatis melalui API Yahoo Finance periode 2020 - 2025.</p></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='info-card'>
          <div class='ic-icon ic-gold'>⚡</div>
          <div><h4>Arsitektur LSTM</h4>
          <p>Menerapkan metode Long Short-Term Memory (Deep Learning) yang optimal dalam menangkap pola pergerakan data time-series jangka panjang..</p></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='info-card'>
          <div class='ic-icon ic-green'>🎯</div>
          <div><h4>Evaluasi</h4>
          <p>Model telah melalui proses hyperparameter tuning dan menghasilkan  (MAPE) sebesar 4.7%.</p></div>
        </div>""", unsafe_allow_html=True)

    # ── Quick Metrics ────────────────────────────────
    st.markdown("<div class='sec-title'><span class='st-dot'></span>Ringkasan Harga</div>", unsafe_allow_html=True)
    latest = float(close.iloc[-1])
    prev   = float(close.iloc[-2])
    d1     = latest - prev
    d1pct  = d1 / prev * 100
    wk52h  = float(close[-252:].max()) if len(close)>=252 else float(close.max())
    wk52l  = float(close[-252:].min()) if len(close)>=252 else float(close.min())
    ma30   = float(close[-30:].mean())
    vol30  = float(close[-30:].std())

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Harga Terakhir",  f"Rp {latest:,.0f}", f"{d1:+.0f} ({d1pct:+.2f}%)")
    m2.metric("52W High",        f"Rp {wk52h:,.0f}")
    m3.metric("52W Low",         f"Rp {wk52l:,.0f}")
    m4.metric("MA-30",           f"Rp {ma30:,.0f}")
    m5.metric("Volatilitas-30",  f"Rp {vol30:,.0f}")

    # ── Statistik Deskriptif CARDS ───────────────────
    st.markdown("<div class='sec-title'><span class='st-dot'></span>Statistik Deskriptif</div>", unsafe_allow_html=True)

    desc = close.describe()
    stat_items = [
        ("Jumlah Data",   f"{int(desc['count']):,}",        "observasi",          "dcard-blue",   "col-blue"),
        ("Rata-rata",     f"Rp {desc['mean']:,.0f}",         "mean harga",         "dcard-gold",   "col-gold"),
        ("Std. Deviasi",  f"Rp {desc['std']:,.0f}",          "dispersi harga",     "dcard-purple", "col-purple"),
        ("Minimum",       f"Rp {desc['min']:,.0f}",           "harga terendah",     "dcard-red",    "col-red"),
        ("Q1 (25%)",      f"Rp {desc['25%']:,.0f}",           "kuartil bawah",      "dcard-teal",   "col-teal"),
        ("Median (50%)",  f"Rp {desc['50%']:,.0f}",           "nilai tengah",       "dcard-green",  "col-green"),
        ("Q3 (75%)",      f"Rp {desc['75%']:,.0f}",           "kuartil atas",       "dcard-blue",   "col-blue"),
        ("Maksimum",      f"Rp {desc['max']:,.0f}",           "harga tertinggi",    "dcard-gold",   "col-gold"),
    ]

    # Render 4 per row × 2 rows
    for row in range(2):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row * 4 + i
            label, val, sub, card_cls, val_cls = stat_items[idx]
            col.markdown(f"""
            <div class='dcard {card_cls}'>
              <div class='d-label'>{label}</div>
              <div class='d-val {val_cls}'>{val}</div>
              <div class='d-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribusi per Tahun ─────────────────────────
    sd1, sd2 = st.columns([1, 1.6])
    with sd1:
        st.markdown("<div class='sec-title' style='margin-top:0.5rem;'><span class='st-dot'></span>Distribusi per Tahun</div>", unsafe_allow_html=True)
        palette = ["#3B6FE8","#E8A838","#18B57A","#E85555","#8B5CF6","#0EA5A0"]
        fig_box = go.Figure()
        for i, yr in enumerate(sorted(df.index.year.unique())):
            yr_data = close[close.index.year == yr]
            c = palette[i % len(palette)]
            fig_box.add_trace(go.Box(
                y=yr_data.values, name=str(yr),
                marker_color=c, line_color=c,
                fillcolor=hex_to_rgba(c, 0.12),
                boxmean=True,
            ))
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", family="Nunito Sans"),
            xaxis=dict(gridcolor="#F0F2F8"),
            yaxis=dict(gridcolor="#F0F2F8", tickformat=",.0f", title="IDR"),
            showlegend=False,
            margin=dict(l=0,r=0,t=10,b=0), height=280,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with sd2:
        st.markdown("<div class='sec-title' style='margin-top:0.5rem;'><span class='st-dot'></span>Return Harian (%)</div>", unsafe_allow_html=True)
        daily_ret = close.pct_change().dropna() * 100
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=daily_ret.values, nbinsx=60,
            marker_color="#3B6FE8", opacity=0.7,
            name="Return",
        ))
        fig_ret.add_vline(x=float(daily_ret.mean()), line=dict(color="#E8A838", width=2, dash="dash"),
                          annotation_text=f"Mean: {daily_ret.mean():.2f}%",
                          annotation=dict(font=dict(color="#E8A838", size=11)))
        fig_ret.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", family="Nunito Sans"),
            xaxis=dict(gridcolor="#F0F2F8", title="Return (%)"),
            yaxis=dict(gridcolor="#F0F2F8", title="Frekuensi"),
            showlegend=False,
            margin=dict(l=0,r=0,t=10,b=0), height=280,
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # ── Grafik Harga Historis ────────────────────────
    st.markdown("<div class='sec-title'><span class='st-dot'></span>Grafik Harga Historis (ANTM.JK)</div>", unsafe_allow_html=True)

    ma20   = close.rolling(20).mean()
    ma50   = close.rolling(50).mean()
    ma200  = close.rolling(200).mean()
    bb_std = close.rolling(20).std()
    bb_up  = ma20 + 2 * bb_std
    bb_lo  = ma20 - 2 * bb_std

    fig_h = go.Figure()
    # BB fill
    fig_h.add_trace(go.Scatter(
        x=list(close.index)+list(close.index[::-1]),
        y=list(bb_up.values)+list(bb_lo.values[::-1]),
        fill='toself', fillcolor='rgba(59,111,232,0.05)',
        line=dict(color='rgba(0,0,0,0)'), name='Bollinger Band', hoverinfo='skip',
    ))
    fig_h.add_trace(go.Scatter(x=close.index, y=bb_up,  line=dict(color='rgba(59,111,232,0.35)', width=1, dash='dot'), showlegend=False))
    fig_h.add_trace(go.Scatter(x=close.index, y=bb_lo,  line=dict(color='rgba(59,111,232,0.35)', width=1, dash='dot'), showlegend=False))
    fig_h.add_trace(go.Scatter(x=close.index, y=ma50,   line=dict(color='#E8A838', width=1.5), name='MA-50'))
    fig_h.add_trace(go.Scatter(x=close.index, y=ma200,  line=dict(color='#8B5CF6', width=1.5), name='MA-200'))
    fig_h.add_trace(go.Scatter(
        x=close.index, y=close.values,
        line=dict(color='#3B6FE8', width=2), name='Harga Penutupan',
        fill='tozeroy', fillcolor='rgba(59,111,232,0.05)',
    ))
    fig_h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B7280", family="Nunito Sans"),
        xaxis=dict(gridcolor="#F0F2F8", showline=True, linecolor="#E2E5EF",
                   rangeslider=dict(visible=True, bgcolor="#F5F6FA")),
        yaxis=dict(gridcolor="#F0F2F8", showline=True, linecolor="#E2E5EF",
                   title="Harga (IDR)", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E5EF", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0,r=0,t=10,b=0), height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Volume ───────────────────────────────────────
    if 'Volume' in df.columns:
        st.markdown("<div class='sec-title'><span class='st-dot'></span>Volume Transaksi Harian</div>", unsafe_allow_html=True)
        vol = df['Volume'].squeeze()
        vol_avg = float(vol.mean())
        vol_colors = ['#18B57A' if v >= vol_avg else '#E85555' for v in vol.values]
        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(x=vol.index, y=vol.values, marker_color=vol_colors, opacity=0.6, name='Volume'))
        fig_v.add_trace(go.Scatter(x=vol.index, y=vol.rolling(30).mean().values,
                                   line=dict(color='#3B6FE8', width=2), name='MA-30'))
        fig_v.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", family="Nunito Sans"),
            xaxis=dict(gridcolor="#F0F2F8"),
            yaxis=dict(gridcolor="#F0F2F8", title="Volume"),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E5EF", borderwidth=1),
            margin=dict(l=0,r=0,t=10,b=0), height=190, hovermode="x unified",
        )
        st.plotly_chart(fig_v, use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB FORECAST
# ═══════════════════════════════════════════════════
with tab_forecast:

    st.markdown("<div class='sec-title'><span class='st-dot'></span>Pengaturan Forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='control-box'>", unsafe_allow_html=True)

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        period = st.radio(
            "Pilih Periode Forecast",
            ["📅 7 Hari", "📆 30 Hari", "✏️ Custom"],
            horizontal=True,
            key="fc_period",
        )
    with fc2:
        if "Custom" in period:
            days = int(st.number_input("Jumlah Hari (maks. 90)", min_value=1, max_value=90, value=14, step=1))
        elif "7" in period:
            days = 7
            st.markdown("<div style='background:#EEF2FF; border:1px solid rgba(59,111,232,0.3); border-radius:8px; padding:8px 12px; color:#3B6FE8; font-size:0.82rem; font-weight:700; margin-top:1.8rem;'>📅 7 hari ke depan dipilih</div>", unsafe_allow_html=True)
        else:
            days = 30
            st.markdown("<div style='background:#EEF2FF; border:1px solid rgba(59,111,232,0.3); border-radius:8px; padding:8px 12px; color:#3B6FE8; font-size:0.82rem; font-weight:700; margin-top:1.8rem;'>📆 30 hari ke depan dipilih</div>", unsafe_allow_html=True)

    show_hist = st.slider("Hari historis yang ditampilkan pada grafik", 30, 365, 120, 10)
    st.markdown("</div>", unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ **Model tidak ditemukan.** Pastikan `model_lstm_antm.h5` dan `scaler_antm.pkl` berada satu folder dengan `app.py`.")

    gen = st.button("🔮  Generate Forecast")

    if gen:
        if not model_loaded:
            st.error("Model LSTM tidak dapat dimuat. Forecast tidak bisa dijalankan.")
        else:
            with st.spinner(f"Menjalankan prediksi {days} hari ke depan…"):
                last_60 = close.values[-60:].reshape(-1,1)
                scaled  = scaler.transform(last_60)
                raw     = predict_recursive(model, scaled, days)
                preds   = scaler.inverse_transform(raw).flatten()

                last_date = close.index[-1]
                # Convert to python datetime to avoid pandas Timestamp arithmetic issues
                last_date_dt = last_date.to_pydatetime()
                future_dates = []
                cur = last_date_dt
                while len(future_dates) < days:
                    cur += timedelta(days=1)
                    if cur.weekday() < 5:
                        future_dates.append(cur)
                # ISO strings for Plotly compatibility
                last_date_str   = last_date_dt.strftime("%Y-%m-%d")
                future_dates_str = [d.strftime("%Y-%m-%d") for d in future_dates]

                fc_df = pd.DataFrame({
                    "Tanggal": future_dates_str,
                    "Prediksi Harga (IDR)": [round(p, 2) for p in preds],
                })
                fc_df["Selisih (IDR)"]  = fc_df["Prediksi Harga (IDR)"].diff().fillna(0).round(2)
                fc_df["Perubahan (%)"]  = (fc_df["Prediksi Harga (IDR)"].pct_change()*100).fillna(0).round(2)

            # ── Ringkasan ─────────────────────────────────
            st.markdown("<div class='sec-title'><span class='st-dot'></span>Ringkasan Prediksi</div>", unsafe_allow_html=True)
            last_act = float(close.iloc[-1])
            p_first  = float(preds[0])
            p_last   = float(preds[-1])
            p_max    = float(preds.max())
            p_min    = float(preds.min())
            total_pct= (p_last - last_act) / last_act * 100

            sm1,sm2,sm3,sm4 = st.columns(4)
            sm1.metric("Prediksi Hari ke-1",    f"Rp {p_first:,.0f}", f"{p_first-last_act:+.0f} ({(p_first-last_act)/last_act*100:+.2f}%)")
            sm2.metric(f"Prediksi Hari ke-{days}", f"Rp {p_last:,.0f}", f"{total_pct:+.2f}% vs Terakhir")
            sm3.metric("Tertinggi Prediksi",    f"Rp {p_max:,.0f}")
            sm4.metric("Terendah Prediksi",     f"Rp {p_min:,.0f}")

            # ── Grafik Historis + Prediksi ─────────────────
            st.markdown("<div class='sec-title'><span class='st-dot'></span>Grafik Historis & Prediksi</div>", unsafe_allow_html=True)

            h_data = close[-show_hist:]
            h_idx  = h_data.index
            h_vals = h_data.values
            # Convert index to ISO strings for Plotly
            h_idx_str    = [d.strftime("%Y-%m-%d") for d in h_idx]
            h_last_str   = h_idx_str[-1]

            fig_fc = go.Figure()

            # Zona prediksi shaded
            all_x = [h_last_str] + future_dates_str
            all_y = np.concatenate([[h_vals[-1]], preds])
            fig_fc.add_trace(go.Scatter(
                x=all_x + all_x[::-1],
                y=list(all_y*1.025) + list(all_y[::-1]*0.975),
                fill='toself', fillcolor='rgba(59,111,232,0.06)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Zona Prediksi', hoverinfo='skip',
            ))

            # Historis
            fig_fc.add_trace(go.Scatter(
                x=h_idx_str, y=h_vals,
                line=dict(color='#3B6FE8', width=2.5),
                name='Historis', mode='lines',
            ))

            # Garis batas — pakai add_shape + add_annotation (hindari bug add_vline di Plotly terbaru)
            fig_fc.add_shape(
                type="line",
                x0=last_date_str, x1=last_date_str,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#E8A838", width=2, dash="dash"),
            )
            fig_fc.add_annotation(
                x=last_date_str,
                y=1,
                xref="x", yref="paper",
                text="◀ Historis  |  Prediksi ▶",
                showarrow=False,
                yanchor="bottom",
                font=dict(color="#E8A838", size=11, family="Nunito Sans"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#E8A838",
                borderwidth=1,
                borderpad=4,
            )

            # Garis prediksi
            fc_x = [h_last_str] + future_dates_str
            fc_y = [float(h_vals[-1])] + list(preds)
            fig_fc.add_trace(go.Scatter(
                x=fc_x, y=fc_y,
                line=dict(color='#18B57A', width=2.5),
                mode='lines+markers',
                marker=dict(size=5, color='#18B57A'),
                name='Prediksi LSTM',
            ))

            # Titik transisi
            fig_fc.add_trace(go.Scatter(
                x=[h_last_str], y=[float(h_vals[-1])],
                mode='markers',
                marker=dict(size=10, color='#E8A838', symbol='circle-open', line=dict(width=2.5)),
                name='Titik Transisi',
            ))

            fig_fc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6B7280", family="Nunito Sans"),
                xaxis=dict(gridcolor="#F0F2F8", showline=True, linecolor="#E2E5EF"),
                yaxis=dict(gridcolor="#F0F2F8", showline=True, linecolor="#E2E5EF",
                           title="Harga (IDR)", tickformat=",.0f"),
                legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#E2E5EF",
                            borderwidth=1, font=dict(size=11)),
                margin=dict(l=0,r=0,t=30,b=0), height=430,
                hovermode="x unified",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # ── Tabel ─────────────────────────────────────
            st.markdown("<div class='sec-title'><span class='st-dot'></span>Tabel Hasil Prediksi</div>", unsafe_allow_html=True)
            t1, t2 = st.columns([3, 1])
            with t1:
                st.dataframe(
                    fc_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Tanggal": st.column_config.TextColumn("Tanggal"),
                        "Prediksi Harga (IDR)": st.column_config.NumberColumn("Prediksi Harga (IDR)", format="Rp %.2f"),
                        "Selisih (IDR)":        st.column_config.NumberColumn("Selisih (IDR)",        format="%.2f"),
                        "Perubahan (%)":        st.column_config.NumberColumn("Perubahan (%)",        format="%.2f %%"),
                    },
                    height=min(40 + 35*len(fc_df), 460),
                )
            with t2:
                bar_colors = ['#18B57A' if v >= 0 else '#E85555' for v in fc_df["Selisih (IDR)"].values]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=fc_df["Tanggal"], y=fc_df["Prediksi Harga (IDR)"],
                    marker_color=bar_colors, opacity=0.85,
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6B7280", family="Nunito Sans", size=10),
                    xaxis=dict(gridcolor="#F0F2F8", showticklabels=False),
                    yaxis=dict(gridcolor="#F0F2F8", tickformat=",.0f"),
                    margin=dict(l=0,r=0,t=10,b=0),
                    height=min(40+35*len(fc_df), 460),
                    showlegend=False,
                    title=dict(text="Tren Prediksi", font=dict(color="#1A1D2E", size=12)),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Download ──────────────────────────────────
            buf = io.StringIO()
            fc_df.to_csv(buf, index=False)
            st.download_button(
                label="📥 Download Hasil Prediksi (.csv)",
                data=buf.getvalue(),
                file_name=f"forecast_ANTM_{last_date_str.replace('-','')}_{days}hari.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='app-footer'>
  © 2026 ANTM Forecast By Vira Aster &nbsp;·&nbsp;
  Sumber Data: Yahoo Finance &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)