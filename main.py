import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NEM Pre-Dispatch Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Modern Dashboard CSS
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container - Clean White Background */
    .main {
        background: #f8fafc;
        padding: 1.5rem 2rem;
    }
    
    /* Header Section - Modern Gradient */
    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.025em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Region Tabs - Clean Look */
    .region-tabs {
        display: flex;
        gap: 0.75rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .region-tab {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1.75rem;
        border-radius: 10px;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
    }
    
    .region-tab.active {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(59,130,246,0.4);
    }
    
    /* Metric Cards - Clean & Modern */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.5rem 0;
        letter-spacing: -0.025em;
    }
    
    .metric-change {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .metric-change.positive { color: #10b981; }
    .metric-change.negative { color: #ef4444; }
    .metric-change.neutral { color: #64748b; }
    
    /* Alert Banner - High Contrast */
    .alert-banner {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(239,68,68,0.4);
        display: flex;
        align-items: center;
        gap: 1rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .alert-icon {
        font-size: 1.75rem;
    }
    
    /* Chart Container - Clean */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Info Boxes - Better Contrast */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1e40af;
        font-size: 0.95rem;
        border: 1px solid #bfdbfe;
    }
    
    .info-box strong {
        color: #1e3a8a;
        font-weight: 700;
    }
    
    /* Warning Box */
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #78350f;
        font-size: 0.95rem;
        border: 1px solid #fde68a;
    }
    
    .warning-box strong {
        color: #78350f;
        font-weight: 700;
    }
    
    /* Success Box */
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #065f46;
        font-size: 0.95rem;
        border: 1px solid #a7f3d0;
    }
    
    .success-box strong {
        color: #065f46;
        font-weight: 700;
    }
    
    /* Confidence Band Legend */
    .confidence-legend {
        display: flex;
        gap: 2rem;
        margin-top: 1.25rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        font-size: 0.875rem;
        border: 1px solid #e5e7eb;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #334155;
        font-weight: 500;
    }
    
    .legend-color {
        width: 32px;
        height: 4px;
        border-radius: 2px;
    }
    
    /* Tabs Styling - High Contrast */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8fafc;
        border-radius: 8px 8px 0 0;
        padding: 1rem 1.75rem;
        font-weight: 600;
        border: 1px solid #e5e7eb;
        border-bottom: none;
        color: #475569;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #3b82f6;
        border-color: #e5e7eb;
        border-bottom: 2px solid white;
        margin-bottom: -2px;
    }
    
    /* Data Table Styling */
    .dataframe {
        font-size: 0.9rem;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: #f1f5f9;
        color: #1e293b;
        font-weight: 700;
        padding: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background: #f8fafc;
    }
    
    .dataframe tbody tr:hover {
        background: #f1f5f9;
    }
    
    .dataframe tbody td {
        color: #334155;
        padding: 0.75rem 1rem;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div > div {
        background: #3b82f6;
    }
    
    /* Button Styling */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(59,130,246,0.4);
        transform: translateY(-1px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
        color: #1e293b;
        border: 1px solid #e5e7eb;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 24px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    .status-active {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #10b981;
    }
    
    .status-inactive {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #ef4444;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #78350f;
        border: 1px solid #f59e0b;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #0f172a;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #475569;
    }
    
    /* Select box styling */
    .stSelectbox label {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Metric styling override */
    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 1.875rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def format_price(value):
    """Format price dengan handling untuk nilai negatif dan extreme"""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"

def format_demand(value):
    """Format demand dalam MW"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"

def get_price_status(price, threshold_high=300, threshold_low=50):
    """Determine price status berdasarkan threshold"""
    if price > threshold_high:
        return "HIGH", "negative"
    elif price < threshold_low:
        return "LOW", "positive"
    else:
        return "NORMAL", "neutral"

def validate_data(df, required_cols=['ds', 'y']):
    """Validasi data integrity"""
    issues = []
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    if df[required_cols].isnull().any().any():
        issues.append("Contains missing values")
    
    # Check data length
    if len(df) < 168:  # Minimum 1 week for lag_168h
        issues.append(f"Insufficient data: {len(df)} rows (need at least 168)")
    
    # Check for duplicates
    if df['ds'].duplicated().any():
        issues.append("Contains duplicate timestamps")
    
    # Check for negative prices (valid in NEM but worth flagging)
    if (df['y'] < -1000).any():
        issues.append("Contains extreme negative prices (< -$1000)")
    
    return issues

# ==========================================
# 3. DATA LOADING WITH ROBUST ERROR HANDLING
# ==========================================

@st.cache_resource(show_spinner=False)
def load_resources():
    """Load model, features, dan buffer data dengan error handling"""
    try:
        # Load Model
        model = xgb.XGBRegressor()
        model.load_model("xgb_electricity_model.json")
        
        # Load Feature Names
        with open("feature_names.json", "r") as f:
            feature_names = json.load(f)
        
        # Load History Buffer
        buffer_df = pd.read_csv("last_data_buffer.csv")
        buffer_df['ds'] = pd.to_datetime(buffer_df['ds'])
        
        # Sort by timestamp
        buffer_df = buffer_df.sort_values('ds').reset_index(drop=True)
        
        # Validate data
        validation_issues = validate_data(buffer_df)
        
        return {
            'model': model,
            'feature_names': feature_names,
            'buffer_df': buffer_df,
            'validation_issues': validation_issues,
            'status': 'success'
        }
        
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error_type': 'file_not_found',
            'message': f"Missing file: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'error_type': 'unknown',
            'message': str(e)
        }

# Load resources
with st.spinner("🔄 Loading dashboard components..."):
    resources = load_resources()

# Error handling
if resources['status'] == 'error':
    st.error(f"❌ **Dashboard Load Failed**")
    st.markdown(f"""
    <div class='warning-box'>
        <strong>Error Type:</strong> {resources['error_type']}<br>
        <strong>Message:</strong> {resources['message']}<br><br>
        <strong>Required Files:</strong>
        <ul>
            <li>xgb_electricity_model.json</li>
            <li>feature_names.json</li>
            <li>last_data_buffer.csv</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Extract resources
model = resources['model']
feature_names = resources['feature_names']
buffer_df = resources['buffer_df']
validation_issues = resources['validation_issues']

# Display validation warnings if any
if validation_issues:
    st.warning(f"⚠️ Data Validation Issues: {', '.join(validation_issues)}")

# ==========================================
# 4. FEATURE ENGINEERING (ENHANCED VERSION)
# ==========================================

def create_features(df_input, fill_method='forward'):
    """
    Enhanced feature engineering dengan better NaN handling
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        Input dataframe dengan kolom 'ds' dan 'y'
    fill_method : str
        Method untuk handle NaN: 'forward', 'mean', atau 'zero'
    """
    df = df_input.copy()
    
    # === LAG FEATURES ===
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}h'] = df['y'].shift(lag)
    
    # === ROLLING STATISTICS ===
    # Mean windows
    for window in [3, 6, 12, 24]:
        df[f'rolling_mean_{window}h'] = df['y'].shift(1).rolling(window=window).mean()
    
    # Other statistics for 24h window
    df['rolling_std_24h'] = df['y'].shift(1).rolling(window=24).std()
    df['rolling_min_24h'] = df['y'].shift(1).rolling(window=24).min()
    df['rolling_max_24h'] = df['y'].shift(1).rolling(window=24).max()
    
    # === TIME FEATURES ===
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_month'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # === BINARY FEATURES ===
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # === CYCLICAL ENCODING ===
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # === INTERACTION FEATURES ===
    df['hour_dow'] = df['hour'] * df['day_of_week']
    
    # === HANDLE NaN VALUES ===
    if fill_method == 'forward':
        df = df.fillna(method='ffill')
    elif fill_method == 'mean':
        df = df.fillna(df.mean())
    elif fill_method == 'zero':
        df = df.fillna(0)
    
    return df

# ==========================================
# 5. PREDICTION WITH CONFIDENCE INTERVALS
# ==========================================

def predict_with_confidence(model, buffer_df, feature_names, horizon_hours, confidence_level=0.9):
    """
    Recursive forecasting dengan confidence intervals menggunakan expanding window
    
    Returns:
    --------
    forecast_df : pd.DataFrame
        Forecast results dengan lower/upper bounds
    """
    # Prepare future dates
    last_timestamp = buffer_df['ds'].iloc[-1]
    future_dates = [last_timestamp + timedelta(hours=i+1) for i in range(horizon_hours)]
    
    # Extended dataframe
    extended_df = buffer_df.copy()
    future_template = pd.DataFrame({'ds': future_dates, 'y': np.nan})
    extended_df = pd.concat([extended_df, future_template], ignore_index=True)
    
    # Storage for predictions and std
    predictions = []
    uncertainties = []
    
    # Iterative prediction
    for i in range(len(buffer_df), len(extended_df)):
        # Generate features
        temp_df = create_features(extended_df.iloc[:i+1], fill_method='forward')
        current_row = temp_df.iloc[[-1]][feature_names]
        
        # Predict
        pred_val = model.predict(current_row)[0]
        predictions.append(pred_val)
        
        # Estimate uncertainty (increases with horizon)
        hours_ahead = i - len(buffer_df) + 1
        base_uncertainty = buffer_df['y'].std() * 0.1  # 10% of historical std
        uncertainty = base_uncertainty * np.sqrt(hours_ahead)  # Increases with sqrt(horizon)
        uncertainties.append(uncertainty)
        
        # Update for next iteration
        extended_df.loc[i, 'y'] = pred_val
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'forecast': predictions,
        'uncertainty': uncertainties
    })
    
    # Calculate confidence intervals
    z_score = 1.645 if confidence_level == 0.9 else 1.96  # 90% or 95%
    forecast_df['lower_bound'] = forecast_df['forecast'] - z_score * forecast_df['uncertainty']
    forecast_df['upper_bound'] = forecast_df['forecast'] + z_score * forecast_df['uncertainty']
    
    return forecast_df

# ==========================================
# 6. DASHBOARD HEADER
# ==========================================

# Header
st.markdown("""
<div class='dashboard-header'>
    <div class='dashboard-title'>⚡ NEM Pre-Dispatch Dashboard</div>
    <div class='dashboard-subtitle'>National Electricity Market | Real-time Forecasting & Analytics</div>
</div>
""", unsafe_allow_html=True)

# Region Selector (mock - bisa diperluas untuk multi-region)
regions = ['NSW', 'QLD', 'VIC', 'SA', 'TAS']
selected_region = st.selectbox("Select Region", regions, index=3, label_visibility="collapsed")

st.markdown(f"""
<div class='region-tabs'>
    <div class='region-tab {"active" if selected_region == "NSW" else ""}'>NSW</div>
    <div class='region-tab {"active" if selected_region == "QLD" else ""}'>QLD</div>
    <div class='region-tab {"active" if selected_region == "VIC" else ""}'>VIC</div>
    <div class='region-tab {"active" if selected_region == "SA" else ""}'>SA</div>
    <div class='region-tab {"active" if selected_region == "TAS" else ""}'>TAS</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. SIDEBAR CONTROLS
# ==========================================

with st.sidebar:
    st.header("⚙️ Forecast Configuration")
    
    horizon_hours = st.slider(
        "Forecast Horizon (Hours)",
        min_value=12,
        max_value=72,
        value=24,
        step=6,
        help="Number of hours to forecast ahead"
    )
    
    confidence_level = st.select_slider(
        "Confidence Level",
        options=[0.80, 0.85, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    st.markdown("---")
    
    # Alert Thresholds
    st.subheader("🚨 Price Alert Thresholds")
    high_threshold = st.number_input("High Price Alert ($/MWh)", value=300, step=50)
    low_threshold = st.number_input("Low Price Alert ($/MWh)", value=50, step=10)
    
    st.markdown("---")
    
    # Display Settings
    st.subheader("📊 Display Settings")
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    show_historical = st.checkbox("Show Historical Data", value=True)
    historical_hours = st.slider("Historical Window (Hours)", 12, 168, 48) if show_historical else 48
    
    st.markdown("---")
    
    # Data Info
    st.subheader("ℹ️ Data Information")
    st.caption(f"**Last Update:** {buffer_df['ds'].iloc[-1].strftime('%d %b %Y %H:%M')}")
    st.caption(f"**Data Points:** {len(buffer_df):,}")
    st.caption(f"**Features:** {len(feature_names)}")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ==========================================
# 8. GENERATE FORECAST
# ==========================================

with st.spinner(f"🔮 Generating {horizon_hours}-hour forecast..."):
    forecast_df = predict_with_confidence(
        model, buffer_df, feature_names, horizon_hours, confidence_level
    )

# ==========================================
# 9. KEY METRICS CARDS
# ==========================================

current_price = buffer_df['y'].iloc[-1]
forecast_30min = forecast_df['forecast'].iloc[0] if len(forecast_df) > 0 else current_price
avg_forecast = forecast_df['forecast'].mean()
max_forecast = forecast_df['forecast'].max()
min_forecast = forecast_df['forecast'].min()
peak_time = forecast_df.loc[forecast_df['forecast'].idxmax(), 'ds']

# Price status
status_text, status_color = get_price_status(current_price, high_threshold, low_threshold)

# Alert Banner
if current_price > high_threshold:
    st.markdown(f"""
    <div class='alert-banner'>
        <div class='alert-icon'>⚠️</div>
        <div>
            <strong>HIGH PRICE ALERT</strong><br>
            Current spot price ({format_price(current_price)}) exceeds threshold ({format_price(high_threshold)})
        </div>
    </div>
    """, unsafe_allow_html=True)
elif max_forecast > high_threshold:
    st.markdown(f"""
    <div class='alert-banner' style='background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);'>
        <div class='alert-icon'>⚡</div>
        <div>
            <strong>FORECAST HIGH PRICE WARNING</strong><br>
            Peak forecast price ({format_price(max_forecast)}) expected at {peak_time.strftime('%H:%M on %d %b')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Metrics Grid
st.markdown("""
<div class='metric-grid'>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    change_pct = ((forecast_30min - current_price) / current_price * 100) if current_price != 0 else 0
    change_class = "positive" if change_pct < 0 else "negative" if change_pct > 0 else "neutral"
    arrow = "↓" if change_pct < 0 else "↑" if change_pct > 0 else "→"
    
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #667eea;'>
        <div class='metric-label'>Current Spot Price</div>
        <div class='metric-value'>{format_price(current_price)}</div>
        <div class='metric-change {change_class}'>{arrow} {abs(change_pct):.1f}% /MWh</div>
        <span class='status-badge status-{status_color}'>{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #8b5cf6;'>
        <div class='metric-label'>Forecast (Next 30min)</div>
        <div class='metric-value'>{format_price(forecast_30min)}</div>
        <div class='metric-change neutral'>Next interval</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_change = ((avg_forecast - current_price) / current_price * 100) if current_price != 0 else 0
    avg_class = "positive" if avg_change < 0 else "negative" if avg_change > 0 else "neutral"
    avg_arrow = "↓" if avg_change < 0 else "↑" if avg_change > 0 else "→"
    
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #10b981;'>
        <div class='metric-label'>Average Forecast</div>
        <div class='metric-value'>{format_price(avg_forecast)}</div>
        <div class='metric-change {avg_class}'>{avg_arrow} {abs(avg_change):.1f}% vs current</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #ef4444;'>
        <div class='metric-label'>Peak Forecast</div>
        <div class='metric-value'>{format_price(max_forecast)}</div>
        <div class='metric-change neutral'>{peak_time.strftime('%H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    low_time = forecast_df.loc[forecast_df['forecast'].idxmin(), 'ds']
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #06b6d4;'>
        <div class='metric-label'>Low Forecast</div>
        <div class='metric-value'>{format_price(min_forecast)}</div>
        <div class='metric-change neutral'>{low_time.strftime('%H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 10. MAIN CHART - PRICE FORECAST
# ==========================================

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>📈 Price Forecast Trend</div>", unsafe_allow_html=True)

fig = go.Figure()

# Historical data
if show_historical:
    plot_hist = buffer_df.iloc[-historical_hours:]
    fig.add_trace(go.Scatter(
        x=plot_hist['ds'],
        y=plot_hist['y'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#1a1a2e', width=2.5),
        hovertemplate='<b>Actual</b><br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
    ))

# Forecast line
fig.add_trace(go.Scatter(
    x=forecast_df['ds'],
    y=forecast_df['forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#3b82f6', width=3, dash='dot'),
    marker=dict(size=6, color='#3b82f6', line=dict(width=1, color='white')),
    hovertemplate='<b>Forecast</b><br>Time: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
))

# Confidence intervals
if show_confidence:
    # Upper bound
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name=f'Upper {int(confidence_level*100)}% CI',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['lower_bound'],
        mode='lines',
        name=f'Confidence Interval ({int(confidence_level*100)}%)',
        line=dict(width=0),
        fillcolor='rgba(59, 130, 246, 0.15)',
        fill='tonexty',
        hovertemplate='<b>Confidence Interval</b><br>Upper: $%{customdata[0]:.2f}<br>Lower: $%{y:.2f}<extra></extra>',
        customdata=forecast_df[['upper_bound']].values
    ))

# Threshold lines
fig.add_hline(
    y=high_threshold,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text=f"High Alert: ${high_threshold}",
    annotation_position="right"
)

fig.add_hline(
    y=low_threshold,
    line_dash="dash",
    line_color="#10b981",
    annotation_text=f"Low Alert: ${low_threshold}",
    annotation_position="right"
)

# Layout
fig.update_layout(
    height=500,
    hovermode="x unified",
    plot_bgcolor="#fafafa",
    paper_bgcolor="white",
    xaxis=dict(
        showgrid=True,
        gridcolor='#e5e7eb',
        title="Time",
        title_font=dict(size=13, color='#475569', weight=600),
        tickfont=dict(size=11, color='#64748b'),
        showline=True,
        linewidth=1,
        linecolor='#cbd5e1'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='#e5e7eb',
        title="Price ($/MWh)",
        title_font=dict(size=13, color='#475569', weight=600),
        tickfont=dict(size=11, color='#64748b'),
        showline=True,
        linewidth=1,
        linecolor='#cbd5e1',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='#cbd5e1'
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="white",
        bordercolor="#e5e7eb",
        borderwidth=1,
        font=dict(size=11, color='#334155')
    ),
    margin=dict(l=60, r=40, t=40, b=60),
    font=dict(family="Inter, sans-serif", color='#334155')
)

st.plotly_chart(fig, use_container_width=True)

# Legend untuk confidence intervals
if show_confidence:
    st.markdown(f"""
    <div class='confidence-legend'>
        <div class='legend-item'>
            <div class='legend-color' style='background: #0f172a;'></div>
            <span>Actual Price</span>
        </div>
        <div class='legend-item'>
            <div class='legend-color' style='background: #3b82f6;'></div>
            <span>Forecast</span>
        </div>
        <div class='legend-item'>
            <div class='legend-color' style='background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6;'></div>
            <span>{int(confidence_level*100)}% Confidence Interval</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 11. SECONDARY CHARTS & ANALYTICS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Hourly Analysis", "📅 Daily Pattern", "📈 Statistics", "📋 Data Table"])

with tab1:
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Hourly Price Distribution</div>", unsafe_allow_html=True)
    
    # Hourly box plot
    forecast_df['hour'] = forecast_df['ds'].dt.hour
    
    fig_hourly = go.Figure()
    
    for hour in sorted(forecast_df['hour'].unique()):
        hour_data = forecast_df[forecast_df['hour'] == hour]
        fig_hourly.add_trace(go.Box(
            y=hour_data['forecast'],
            name=f"{hour:02d}:00",
            marker=dict(color='#3b82f6'),
            boxmean='sd'
        ))
    
    fig_hourly.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Hour of Day",
        yaxis_title="Forecast Price ($/MWh)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color='#334155'),
        xaxis=dict(tickfont=dict(color='#64748b')),
        yaxis=dict(tickfont=dict(color='#64748b'))
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Daily Price Pattern</div>", unsafe_allow_html=True)
    
    # Daily aggregation
    forecast_df['date'] = forecast_df['ds'].dt.date
    daily_stats = forecast_df.groupby('date')['forecast'].agg(['mean', 'min', 'max']).reset_index()
    
    fig_daily = go.Figure()
    
    fig_daily.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['max'],
        name='Daily Max',
        line=dict(color='#ef4444', width=2),
        mode='lines+markers'
    ))
    
    fig_daily.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['mean'],
        name='Daily Average',
        line=dict(color='#667eea', width=2),
        mode='lines+markers'
    ))
    
    fig_daily.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['min'],
        name='Daily Min',
        line=dict(color='#10b981', width=2),
        mode='lines+markers'
    ))
    
    fig_daily.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Price ($/MWh)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", color='#334155'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(tickfont=dict(color='#64748b')),
        yaxis=dict(tickfont=dict(color='#64748b'))
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Forecast Statistics</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Statistical summary
        stats_dict = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', '25th Percentile', '75th Percentile'],
            'Value': [
                f"{forecast_df['forecast'].mean():.2f}",
                f"{forecast_df['forecast'].median():.2f}",
                f"{forecast_df['forecast'].std():.2f}",
                f"{forecast_df['forecast'].min():.2f}",
                f"{forecast_df['forecast'].max():.2f}",
                f"{forecast_df['forecast'].max() - forecast_df['forecast'].min():.2f}",
                f"{forecast_df['forecast'].quantile(0.25):.2f}",
                f"{forecast_df['forecast'].quantile(0.75):.2f}"
            ]
        }
        stats_df = pd.DataFrame(stats_dict)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        # Price distribution histogram
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=forecast_df['forecast'],
            nbinsx=30,
            marker=dict(
                color='#3b82f6',
                line=dict(color='white', width=1)
            ),
            name='Forecast Distribution'
        ))
        
        fig_dist.update_layout(
            height=300,
            xaxis_title="Price ($/MWh)",
            yaxis_title="Frequency",
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(family="Inter, sans-serif", color='#334155'),
            xaxis=dict(tickfont=dict(color='#64748b')),
            yaxis=dict(tickfont=dict(color='#64748b'))
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Volatility analysis
    st.markdown("### 📉 Volatility Analysis")
    
    # Calculate hour-to-hour changes
    forecast_df['price_change'] = forecast_df['forecast'].diff()
    forecast_df['price_change_pct'] = forecast_df['forecast'].pct_change() * 100
    
    vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
    
    with vol_col1:
        st.metric("Avg Absolute Change", f"${forecast_df['price_change'].abs().mean():.2f}")
    with vol_col2:
        st.metric("Max Price Jump", f"${forecast_df['price_change'].max():.2f}")
    with vol_col3:
        st.metric("Max Price Drop", f"${forecast_df['price_change'].min():.2f}")
    with vol_col4:
        st.metric("Volatility (Std)", f"{forecast_df['price_change'].std():.2f}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Detailed Forecast Data</div>", unsafe_allow_html=True)
    
    # Prepare display dataframe
    display_df = forecast_df[['ds', 'forecast', 'lower_bound', 'upper_bound', 'uncertainty']].copy()
    display_df.columns = ['Timestamp', 'Forecast ($/MWh)', 'Lower Bound', 'Upper Bound', 'Uncertainty']
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Format numeric columns
    for col in ['Forecast ($/MWh)', 'Lower Bound', 'Upper Bound', 'Uncertainty']:
        display_df[col] = display_df[col].round(2)
    
    # Add color coding based on price levels
    def highlight_prices(row):
        if row['Forecast ($/MWh)'] > high_threshold:
            return ['background-color: #fee2e2'] * len(row)
        elif row['Forecast ($/MWh)'] < low_threshold:
            return ['background-color: #d1fae5'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_prices, axis=1)
    
    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=400)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"nem_forecast_{selected_region}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 12. INSIGHTS & RECOMMENDATIONS
# ==========================================

st.markdown("---")
st.markdown("### 💡 Key Insights & Recommendations")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    # Price trend analysis
    price_trend = "increasing" if avg_forecast > current_price else "decreasing" if avg_forecast < current_price else "stable"
    trend_emoji = "📈" if price_trend == "increasing" else "📉" if price_trend == "decreasing" else "➡️"
    trend_color = "warning" if price_trend == "increasing" else "success" if price_trend == "decreasing" else "info"
    
    st.markdown(f"""
    <div class='{trend_color}-box'>
        <strong>{trend_emoji} Price Trend:</strong> The forecast indicates {price_trend} prices over the next {horizon_hours} hours.
        Average forecast is {format_price(avg_forecast)} ({abs(((avg_forecast - current_price) / current_price * 100)):.1f}% {'higher' if avg_forecast > current_price else 'lower'} than current).
    </div>
    """, unsafe_allow_html=True)
    
    # Volatility insight
    volatility_level = "high" if forecast_df['price_change'].std() > 50 else "moderate" if forecast_df['price_change'].std() > 20 else "low"
    vol_emoji = "⚠️" if volatility_level == "high" else "📊" if volatility_level == "moderate" else "✅"
    
    st.markdown(f"""
    <div class='info-box'>
        <strong>{vol_emoji} Volatility:</strong> Expected volatility is {volatility_level} with standard deviation of ${forecast_df['price_change'].std():.2f}/hour.
        {'Exercise caution with trading decisions.' if volatility_level == 'high' else 'Market conditions appear relatively stable.'}
    </div>
    """, unsafe_allow_html=True)

with insights_col2:
    # Peak/Off-peak analysis
    peak_hours = forecast_df[forecast_df['ds'].dt.hour.isin(range(6, 22))]
    offpeak_hours = forecast_df[~forecast_df['ds'].dt.hour.isin(range(6, 22))]
    
    if len(peak_hours) > 0 and len(offpeak_hours) > 0:
        peak_avg = peak_hours['forecast'].mean()
        offpeak_avg = offpeak_hours['forecast'].mean()
        
        st.markdown(f"""
        <div class='info-box'>
            <strong>⏰ Peak vs Off-Peak:</strong><br>
            • Peak hours (6AM-10PM): {format_price(peak_avg)}<br>
            • Off-peak hours: {format_price(offpeak_avg)}<br>
            • Difference: {format_price(abs(peak_avg - offpeak_avg))} ({abs((peak_avg - offpeak_avg) / offpeak_avg * 100):.1f}%)
        </div>
        """, unsafe_allow_html=True)
    
    # Alert recommendations
    high_price_count = len(forecast_df[forecast_df['forecast'] > high_threshold])
    if high_price_count > 0:
        st.markdown(f"""
        <div class='warning-box'>
            <strong>⚡ High Price Alert:</strong> {high_price_count} intervals forecast to exceed ${high_threshold}/MWh threshold.
            Consider demand response or hedging strategies during these periods.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='success-box'>
            <strong>✅ Price Stability:</strong> No intervals forecast to exceed high price threshold.
            Market conditions appear favorable for the forecast period.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 13. FOOTER & MODEL INFO
# ==========================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **📊 Model Information**
    - Model: XGBoost Regressor
    - Features: """ + str(len(feature_names)) + """
    - Training Data: Historical NEM prices
    """)

with footer_col2:
    st.markdown(f"""
    **⚙️ Forecast Configuration**
    - Horizon: {horizon_hours} hours
    - Confidence: {int(confidence_level * 100)}%
    - Method: Recursive Auto-regressive
    """)

with footer_col3:
    st.markdown(f"""
    **ℹ️ Disclaimer**
    - Forecasts are estimates only
    - Not financial advice
    - Use at your own risk
    """)

st.caption("Dashboard powered by Streamlit | Data source: AEMO NEM | © 2026")