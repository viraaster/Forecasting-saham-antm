import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# 1. CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="ANTM Forecast", page_icon="🪙", layout="wide")

st.markdown("""
<style>
/* CSS tetap sama seperti kode Anda sebelumnya agar tampilan konsisten */
.app-header { background: #FFFFFF; border: 1px solid #E2E5EF; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.6rem; display: flex; align-items: center; gap: 1.4rem; }
.sec-title { font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 700; margin: 1.8rem 0 0.9rem; display: flex; align-items: center; gap: 8px; }
.st-dot { width: 8px; height: 8px; background: #3B6FE8; border-radius: 50%; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. MODEL LOADER (DIPERBAIKI)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        # Pastikan file ada di folder yang sama dengan app.py
        model = load_model("model_lstm_antm.h5", compile=False)
        scaler = joblib.load("scaler_antm.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None, None

# Memuat model ke variabel global yang stabil
model_obj, scaler_obj = load_artifacts()

# ─────────────────────────────────────────────
# 3. DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    df = yf.download("ANTM.JK", start="2020-01-01", end="2025-12-30", progress=False)
    return df.dropna()

df = get_data()
close = df['Close'].squeeze()

# ─────────────────────────────────────────────
# 4. PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_recursive(model, scaled_input, days):
    inputs = scaled_input.copy().reshape(-1)
    predictions = []
    for _ in range(days):
        window = inputs[-60:].reshape(1, 60, 1)
        pred = model.predict(window, verbose=0)
        val = float(pred[0, 0])
        predictions.append(val)
        inputs = np.append(inputs, val)
    return np.array(predictions).reshape(-1, 1)

# ─────────────────────────────────────────────
# 5. MAIN APP
# ─────────────────────────────────────────────
st.markdown("<div class='app-header'><h1>ANTM Stock Forecast</h1></div>", unsafe_allow_html=True)

tab_beranda, tab_forecast = st.tabs(["🧭 Beranda", "⚡ Forecast"])

with tab_forecast:
    st.markdown("<div class='sec-title'><span class='st-dot'></span>Pengaturan Forecast</div>", unsafe_allow_html=True)
    
    period = st.radio("Pilih Periode", ["7 Hari", "30 Hari", "Custom"], horizontal=True)
    days = 7 if "7" in period else (30 if "30" in period else st.number_input("Hari", 1, 90, 14))
    
    gen = st.button("🔮 Generate Forecast")

    if gen:
        # Pengecekan aman untuk model
        if model_obj is None or scaler_obj is None:
            st.error("Model tidak dapat dimuat. Pastikan file .h5 dan .pkl ada.")
        else:
            with st.spinner("Menjalankan prediksi..."):
                last_60 = close.values[-60:].reshape(-1, 1)
                scaled = scaler_obj.transform(last_60)
                raw = predict_recursive(model_obj, scaled, days)
                preds = scaler_obj.inverse_transform(raw).flatten()

                # Visualisasi (Parameter width='stretch' sebagai pengganti use_container_width)
                st.success("Prediksi Berhasil!")
                
                # Contoh penggunaan parameter baru
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=preds, mode='lines+markers', name='Prediksi'))
                st.plotly_chart(fig, width='stretch')

                # Tabel
                st.dataframe(pd.DataFrame(preds, columns=["Harga"]), width='stretch')

# Footer
st.markdown("<div class='app-footer'>© 2026 ANTM Forecast</div>", unsafe_allow_html=True)
