import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IDS Demo System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL - Mặc định backend chạy ở localhost:8000
API_URL = "http://localhost:8000/api"
BASE_URL = "http://localhost:8000"

# --- API HELPERS ---
def check_health():
    try:
        r = requests.get(f"{BASE_URL}/health")
        return r.status_code == 200
    except:
        return False

def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats")
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def simulate_attack_data(attack_type):
    """Gọi API để lấy dữ liệu giả lập cho loại tấn công cụ thể"""
    try:
        r = requests.get(f"{API_URL}/simulate/{attack_type}")
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Error simulating attack: {e}")
    return None

def predict(features):
    """Gửi features lên server để dự đoán"""
    try:
        r = requests.post(f"{API_URL}/predict", json=features)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Prediction failed: {r.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

# --- UI COMPONENTS ---
def show_header():
    st.title("🛡️ Intrusion Detection System")
    st.markdown("### Machine Learning based Network Security")
    
    # Status check
    if check_health():
        st.success("🟢 Backend System: Online")
    else:
        st.error("🔴 Backend System: Offline (Please start backend/main.py)")

def render_dashboard():
    st.header("📊 System Dashboard")
    
    stats = get_stats()
    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Status", stats.get('status', 'Unknown'))
        with col2:
            st.metric("Features Count", stats.get('features_count', 0))
        with col3:
            st.metric("Supported Classes", stats.get('classes_count', 0))
        
        st.json(stats)
    else:
        st.warning("Cannot fetch system stats. Is the backend running?")

def render_simulator():
    st.header("🎯 Attack Simulator & Detection")
    st.markdown("Simulate network traffic patterns and test the IDS model response.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Select Traffic Type")
        attack_type = st.selectbox(
            "Choose scenario to simulate:",
            ["normal", "dos", "probe", "r2l", "u2r"],
            format_func=lambda x: x.upper() + " Traffic"
        )
        
        if st.button("🎲 Generate Traffic Data", use_container_width=True):
            with st.spinner("Simulating network traffic..."):
                # 1. Get simulated data from backend
                sim_data = simulate_attack_data(attack_type)
                
                if sim_data:
                    st.session_state['current_features'] = sim_data['features']
                    st.session_state['current_type'] = sim_data['type']
                    st.session_state['description'] = sim_data['description']
                    st.session_state['explanation'] = sim_data['explanation']
                    st.success(f"Generated {sim_data['type']} traffic pattern!")

    with col2:
        st.subheader("2. Analysis & Prediction")
        
        if 'current_features' in st.session_state:
            # Display features nicely
            with st.expander("View Generated Network Features", expanded=True):
                features = st.session_state['current_features']
                # Show key features
                key_metrics = {
                    "Protocol": features.get('protocol_type'),
                    "Service": features.get('service'),
                    "Duration": f"{features.get('duration')}s",
                    "Bytes (Src/Dst)": f"{features.get('src_bytes')} / {features.get('dst_bytes')}"
                }
                st.write(key_metrics)
                st.json(features, expanded=False)
                
                st.info(f"**Scenario Context:** {st.session_state.get('description', '')}")

            # Predict Action
            if st.button("🚀 Analyze Traffic with IDS Model", type="primary", use_container_width=True):
                with st.spinner("Analyzing packet signatures..."):
                    time.sleep(0.5) # UX delay
                    result = predict(st.session_state['current_features'])
                    
                    if result:
                        prediction = result['prediction']
                        confidence = result['confidence']
                        
                        # Result Display
                        st.divider()
                        r_col1, r_col2 = st.columns(2)
                        
                        with r_col1:
                            if result['is_attack']:
                                st.error(f"🚨 ALERT: {prediction} DETECTED")
                            else:
                                st.success(f"✅ STATUS: {prediction}")
                        
                        with r_col2:
                            st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                        
                        # Probability Chart
                        probs = result['probabilities']
                        fig = px.bar(
                            x=list(probs.keys()), 
                            y=list(probs.values()),
                            labels={'x': 'Class', 'y': 'Probability'},
                            title="Model Classification Probabilities"
                        )
                        st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP FLOW ---
def main():
    show_header()
    
    # Sidebar Navigation
    page = st.sidebar.radio("Navigation", ["Simulator", "Dashboard"])
    
    st.sidebar.divider()
    st.sidebar.info(
        "**Guide:**\n"
        "1. Go to **Simulator**\n"
        "2. Select a traffic type (e.g., DoS)\n"
        "3. Click 'Generate'\n"
        "4. Click 'Analyze' to see IDS result"
    )

    if page == "Dashboard":
        render_dashboard()
    elif page == "Simulator":
        render_simulator()

if __name__ == "__main__":
    main()
