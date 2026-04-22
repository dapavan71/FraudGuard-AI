import streamlit as st
import requests
import json
import random
import time

st.set_page_config(page_title="Fraudguard | Real-Time Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: scale(1.05);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .metric-title {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .fraud-alert {
        padding: 20px;
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        color: #ef4444;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    .normal-alert {
        padding: 20px;
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 5px solid #22c55e;
        border-radius: 8px;
        color: #22c55e;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; transform: scale(1.02); }
        100% { opacity: 0.8; }
    }
</style>
""", unsafe_allow_html=True)

# --- Define Helper Functions ---
API_URL = "http://localhost:8000/predict"

def generate_random_transaction(forced_fraud=False):
    t = {
        'Time': float(random.randint(0, 172800)),
        'Amount': round(random.uniform(1.0, 5000.0), 2)
    }
    # Using specific ranges to simulate regular V variables
    for i in range(1, 29):
        # Normal data
        val = random.gauss(0, 1)
        if forced_fraud and i in [3, 4, 10, 11, 12, 14, 16, 17]:
            # Push specifically correlated V features away from mean to simulate fraud
            val += random.choice([-3.0, 3.0])
        t[f'V{i}'] = float(val)
    return t

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #f8fafc; margin-bottom: 0;'>🛡️ FraudGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 40px;'>Real-time Credit Card Fraud Detection Powered by XGBoost & Isolation Forest</p>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Simulation Engine")
    st.markdown("Generate transaction scenarios to test the engine's capabilities.")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_normal = st.button("Generate Normal")
    with col2:
        sim_fraud = st.button("Simulate Fraud")
        
    st.divider()
    
    st.subheader("Manual Input Override")
    manual_amount = st.slider("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)
    manual_v1 = st.slider("V1 Feature Value", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    
    run_manual = st.button("Run Custom Transaction", use_container_width=True)

# --- Main Logic State ---
if 'current_transaction' not in st.session_state:
    st.session_state.current_transaction = generate_random_transaction()

if sim_normal:
    st.session_state.current_transaction = generate_random_transaction(forced_fraud=False)
elif sim_fraud:
    st.session_state.current_transaction = generate_random_transaction(forced_fraud=True)
elif run_manual:
    tx = generate_random_transaction()
    tx['Amount'] = manual_amount
    tx['V1'] = manual_v1
    st.session_state.current_transaction = tx

# --- Dashboard Layout ---
col_feat, col_results = st.columns([1, 1], gap="large")

with col_feat:
    st.markdown("### 📡 Live Transaction Stream")
    with st.container():
        # Make a visually appealing terminal-like output for the stream
        st.code(json.dumps(st.session_state.current_transaction, indent=4)[1:-1], language='json')
        
with col_results:
    st.markdown("### 🧠 AI Analysis Engine")
    
    # Process the transaction through API
    with st.spinner("Analyzing transaction patterns..."):
        time.sleep(0.5) # Fake slight delay for visual effect
        try:
            response = requests.post(API_URL, json=st.session_state.current_transaction, timeout=5)
            if response.status_code == 200:
                result = response.json()
                
                # Top Level Alert
                if result['prediction'] == 'Fraud':
                    st.markdown("<div class='fraud-alert'>🚨 CRITICAL ALERT: FRAUDULENT PATTERN DETECTED 🚨</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='normal-alert'>✅ TRANSACTION APPROVED: NO ANOMALIES DETECTED</div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics Grid
                m1, m2 = st.columns(2)
                
                with m1:
                    conf_pct = f"{result['confidence']*100:.1f}%"
                    color = "#ef4444" if result['prediction'] == 'Fraud' else "#22c55e"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Confidence Score</div>
                        <div class='metric-value' style='color: {color}'>{conf_pct}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m2:
                    score = result['anomaly_score']
                    card_color = "#eab308" if result['is_anomaly'] else "#f8fafc"
                    status = "ANOMALY" if result['is_anomaly'] else "Standard"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Isolation Forest Profile</div>
                        <div class='metric-value' style='font-size: 1.5rem; color: {card_color}'>{status} <br><span style='font-size: 1rem'>Score: {score:.3f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API Backend. Please ensure FastAPI (uvicorn api:app --port 8000) is running.")
            
