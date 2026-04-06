import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_manager import load_data, add_entry
from ml_engine import analyze_vitals

# PAGE CONFIG
st.set_page_config(
    page_title="VitalSync AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HELPER
def calculate_trend(df, column):
    if len(df) < 2:
        return 0, 0
    current = df.iloc[-1][column]
    previous = df.iloc[-2][column]
    diff = current - previous
    return current, diff

# --- LOAD DATA ---
df = load_data()
df_analyzed = analyze_vitals(df)

# --- SIDEBAR (DATA INPUT) ---
with st.sidebar:
    st.title("🧬 VitalSync AI")
    st.markdown("Enter your daily vitals to monitor your health trends and detect anomalies.")
    
    st.header("New Entry")
    with st.form("entry_form"):
        entry_date = st.date_input("Date", datetime.now())
        entry_hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75)
        
        col1, col2 = st.columns(2)
        with col1:
            entry_sys_bp = st.number_input("Systolic BP", min_value=70, max_value=250, value=120)
        with col2:
            entry_dia_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
            
        entry_sleep = st.number_input("Sleep (hours)", min_value=0.0, max_value=24.0, value=7.5, step=0.5)
        entry_steps = st.number_input("Steps", min_value=0, max_value=50000, value=8000, step=500)
        
        submitted = st.form_submit_button("Save Vitals")
        if submitted:
            new_data = {
                "date": entry_date.strftime("%Y-%m-%d"),
                "heart_rate": entry_hr,
                "systolic_bp": entry_sys_bp,
                "diastolic_bp": entry_dia_bp,
                "sleep_hours": entry_sleep,
                "steps": entry_steps
            }
            add_entry(new_data)
            st.success("Vitals Saved successfully!")
            st.rerun()

# --- MAIN DASHBOARD ---
st.title("User Health Overview")

# 1. ALERTS SECTION
latest_entry = df_analyzed.iloc[-1]
if latest_entry.get('anomaly', 1) == -1:
    st.error("⚠️ **AI Alert:** Anomalous vitals detected in your latest entry. Please monitor your health closely or consult a physician if symptoms occur.")
else:
    st.success("✅ **AI Status:** All vitals are tracking within expected baseline ranges.")

st.markdown("---")

# 2. KEY METRICS CARDS
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    cur_hr, diff_hr = calculate_trend(df_analyzed, "heart_rate")
    st.metric(label="Latest Heart Rate", value=f"{cur_hr:.0f} bpm", delta=f"{diff_hr:.0f} bpm", delta_color="inverse")

with kpi2:
    cur_sys, diff_sys = calculate_trend(df_analyzed, "systolic_bp")
    cur_dia, diff_dia = calculate_trend(df_analyzed, "diastolic_bp")
    st.metric(label="Latest Blood Pressure", value=f"{cur_sys:.0f}/{cur_dia:.0f}", delta=f"{diff_sys:.0f} sys", delta_color="inverse")
    
with kpi3:
    cur_sleep, diff_sleep = calculate_trend(df_analyzed, "sleep_hours")
    st.metric(label="Sleep Duration", value=f"{cur_sleep:.1f} hrs", delta=f"{diff_sleep:.1f} hrs")

with kpi4:
    cur_steps, diff_steps = calculate_trend(df_analyzed, "steps")
    st.metric(label="Daily Steps", value=f"{cur_steps:.0f}", delta=f"{diff_steps:.0f}")

st.markdown("---")

# 3. INTERACTIVE CHARTS
st.header("Trends & Analysis")

tab1, tab2, tab3 = st.tabs(["Heart Rate & BP", "Activity & Sleep", "Anomaly Timeline"])

# Ensure date is interpreted correctly
plot_df = df_analyzed.copy()
plot_df['date'] = pd.to_datetime(plot_df['date'])

with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_hr = px.line(plot_df, x="date", y="heart_rate", markers=True, title="Heart Rate Over Time")
        fig_hr.update_traces(line_color="#ff4b4b")
        st.plotly_chart(fig_hr, use_container_width=True)
        
    with col_chart2:
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['systolic_bp'], name="Systolic", mode='lines+markers', line=dict(color='#00d2ff')))
        fig_bp.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['diastolic_bp'], name="Diastolic", mode='lines+markers', line=dict(color='#0068c9')))
        fig_bp.update_layout(title="Blood Pressure Over Time")
        st.plotly_chart(fig_bp, use_container_width=True)

with tab2:
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        fig_steps = px.bar(plot_df, x="date", y="steps", title="Daily Step Count")
        fig_steps.update_traces(marker_color="#29b5e8")
        st.plotly_chart(fig_steps, use_container_width=True)
        
    with col_chart4:
        fig_sleep = px.area(plot_df, x="date", y="sleep_hours", title="Sleep Duration")
        fig_sleep.update_traces(line_color="#ba55d3", fillcolor="rgba(186,85,211,0.3)")
        st.plotly_chart(fig_sleep, use_container_width=True)

with tab3:
    st.subheader("Detected Anomalies")
    st.markdown("The chart below marks data points flagged by the **Isolation Forest** ML model as abnormal based on your historical baseline.")
    
    fig_anom = go.Figure()
    
    # Normal points
    normal_df = plot_df[plot_df['anomaly'] == 1]
    fig_anom.add_trace(go.Scatter(x=normal_df['date'], y=normal_df['heart_rate'], 
                                  mode='markers', name='Normal', marker=dict(color='#00d2ff', size=8)))
    
    # Anomaly points
    anom_df = plot_df[plot_df['anomaly'] == -1]
    fig_anom.add_trace(go.Scatter(x=anom_df['date'], y=anom_df['heart_rate'], 
                                  mode='markers', name='Anomaly', marker=dict(color='#ff4b4b', size=12, symbol='x')))
    
    fig_anom.update_layout(title="Anomalies in Heart Rate Context", xaxis_title="Date", yaxis_title="Heart Rate")
    st.plotly_chart(fig_anom, use_container_width=True)
