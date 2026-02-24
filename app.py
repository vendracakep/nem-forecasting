import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import json
from datetime import timedelta

# ==========================================
# 1. SETUP HALAMAN & GAYA (AEMO STYLE)
# ==========================================
st.set_page_config(page_title="NEM Market Dashboard", page_icon="⚡", layout="wide")

# CSS Custom untuk tampilan profesional ala market dashboard
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa; border-left: 5px solid #C00000;
        padding: 15px; margin-bottom: 10px; border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1 { font-family: 'Arial', sans-serif; font-weight: 700; color: #212529; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_resources():
    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("xgb_electricity_model.json")
    
    # Load Feature Names (Wajib sama dengan notebook)
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    # Load History Buffer (Data terakhir dari notebook)
    buffer_df = pd.read_csv("last_data_buffer.csv")
    buffer_df['ds'] = pd.to_datetime(buffer_df['ds'])
    
    return model, feature_names, buffer_df

try:
    model, feature_names, buffer_df = load_resources()
except FileNotFoundError as e:
    st.error("⚠️ File deployment hilang! Pastikan `xgb_electricity_model.json`, `feature_names.json`, dan `last_data_buffer.csv` ada di folder ini.")
    st.stop()

# ==========================================
# 3. FEATURE ENGINEERING (REPLIKASI NOTEBOOK)
# ==========================================
def create_features(df_input):
    """
    LOGIC INI SAMA PERSIS DENGAN NOTEBOOK ANDA.
    Jangan diubah agar prediksi valid.
    """
    df = df_input.copy()
    
    # --- 1. LAG FEATURES ---
    # Shift target 'y' untuk mendapatkan nilai masa lalu
    # Di environment produksi, nilai ini diambil dari database/history real
    df['lag_1h'] = df['y'].shift(1)
    df['lag_2h'] = df['y'].shift(2)
    df['lag_3h'] = df['y'].shift(3)
    df['lag_6h'] = df['y'].shift(6)
    df['lag_12h'] = df['y'].shift(12)
    df['lag_24h'] = df['y'].shift(24)
    df['lag_48h'] = df['y'].shift(48)
    df['lag_168h'] = df['y'].shift(168)
    
    # --- 2. ROLLING STATISTICS ---
    # Perhatikan: shift(1) dulu baru rolling, mencegah data leakage
    df['rolling_mean_3h'] = df['y'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6h'] = df['y'].shift(1).rolling(window=6).mean()
    df['rolling_mean_12h'] = df['y'].shift(1).rolling(window=12).mean()
    df['rolling_mean_24h'] = df['y'].shift(1).rolling(window=24).mean()
    
    df['rolling_std_24h'] = df['y'].shift(1).rolling(window=24).std()
    df['rolling_min_24h'] = df['y'].shift(1).rolling(window=24).min()
    df['rolling_max_24h'] = df['y'].shift(1).rolling(window=24).max()
    
    # --- 3. TIME FEATURES ---
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_month'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # --- 4. BINARY FEATURES ---
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # --- 5. CYCLICAL ENCODING ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # --- 6. INTERACTION ---
    df['hour_dow'] = df['hour'] * df['day_of_week']
    
    return df

# ==========================================
# 4. LOGIKA PREDIKSI (ROLLING FORECAST)
# ==========================================
st.sidebar.header("⚙️ Forecast Settings")
horizon_hours = st.sidebar.slider("Forecast Horizon (Hours)", 12, 72, 24)

# Kita perlu melakukan prediksi 'Recursive': Prediksi jam ke-1 dipakai untuk jam ke-2, dst.
# Karena fitur Lag (misal lag_1h) butuh data masa depan yang belum ada.

# 1. Siapkan DataFrame Masa Depan
last_timestamp = buffer_df['ds'].iloc[-1]
future_dates = [last_timestamp + timedelta(hours=i+1) for i in range(horizon_hours)]

# Gabungkan data history + template masa depan
# Kita butuh buffer panjang (168 jam) karena ada fitur lag_168h
extended_df = buffer_df.copy()
future_template = pd.DataFrame({'ds': future_dates, 'y': np.nan}) # y kosong
extended_df = pd.concat([extended_df, future_template], ignore_index=True)

# 2. Iterative Prediction (Jam demi Jam)
# Ini teknik standar untuk model autoregressive seperti XGBoost Time Series
for i in range(len(buffer_df), len(extended_df)):
    
    # A. Generate Features untuk posisi saat ini (menggunakan data yang sudah ada sebelumnya)
    temp_df = create_features(extended_df.iloc[:i+1])
    
    # B. Ambil baris terakhir (yang mau diprediksi) & filter kolom sesuai training
    current_row = temp_df.iloc[[-1]][feature_names]
    
    # C. Prediksi
    pred_val = model.predict(current_row)[0]
    
    # D. Simpan hasil prediksi ke kolom 'y' agar bisa dipakai untuk lag jam berikutnya
    extended_df.loc[i, 'y'] = pred_val

# Ambil hanya bagian forecast
forecast_result = extended_df.iloc[-horizon_hours:].copy()

# ==========================================
# 5. DASHBOARD VISUALIZATION
# ==========================================
st.title("⚡ Electricity Market Dashboard")
st.markdown(f"**Region:** SA1 (South Australia) | **Last Update:** {last_timestamp.strftime('%d %b %Y %H:%M')}")

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
current_price = buffer_df['y'].iloc[-1]
avg_fc = forecast_result['y'].mean()
max_fc = forecast_result['y'].max()
min_fc = forecast_result['y'].min()

with col1: st.metric("Current Price", f"${current_price:.2f}")
with col2: st.metric("Avg Forecast", f"${avg_fc:.2f}", f"{avg_fc-current_price:.1f}")
with col3: st.metric("Peak Forecast", f"${max_fc:.2f}", delta_color="inverse")
with col4: st.metric("Low Forecast", f"${min_fc:.2f}", delta_color="normal")

st.markdown("---")

# --- CHART (AEMO STYLE) ---
fig = go.Figure()

# 1. Historical Data (48 jam terakhir)
plot_hist = buffer_df.iloc[-48:]
fig.add_trace(go.Scatter(
    x=plot_hist['ds'], y=plot_hist['y'],
    mode='lines', name='Actual History',
    line=dict(color='black', width=2)
))

# 2. Forecast Data
fig.add_trace(go.Scatter(
    x=forecast_result['ds'], y=forecast_result['y'],
    mode='lines+markers', name='Forecast',
    line=dict(color='#C00000', width=2, dash='dot'),
    marker=dict(size=5, color='#C00000')
))

fig.update_layout(
    title="Price Forecast Trend ($/MWh)",
    height=500,
    hovermode="x unified",
    plot_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="Price ($/MWh)"),
    legend=dict(orientation="h", y=1.02, x=0)
)

st.plotly_chart(fig, use_container_width=True)

# --- DATA TABLE ---
with st.expander("📊 View Detailed Forecast Data"):
    st.dataframe(forecast_result[['ds', 'y']].rename(columns={'ds': 'Time', 'y': 'Forecast ($/MWh)'}))