import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS styling with medical theme and dark mode
st.markdown("""
<style>
    /* Color Variables */
    :root {
        --primary-blue: #3a86ff;
        --secondary-blue: #4cc9f0;
        --success-green: #06d6a0;
        --warning-red: #ef476f;
        --accent-purple: #8338ec;
        --dark-bg: #121212;
        --card-bg: #1e1e1e;
        --text-light: #f8f9fa;
        --text-secondary: #b0b0b0;
    }
    
    /* Global Styling for Dark Theme */
    .stApp {
        font-family: 'Montserrat', 'Helvetica Neue', sans-serif;
        color: var(--text-light);
    }
    
    /* Dark mode overrides */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div {
        background-color: var(--card-bg) !important;
        color: var(--text-light) !important;
        border-color: #333 !important;
    }
    
    /* Header Banner with modern gradient */
    .header-banner {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--primary-blue) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(131, 56, 236, 0.3);
        position: relative;
        overflow: hidden;
        max-height: 180px;
        width: 90%;
        margin-left: auto;
        margin-right: auto;
    }
    
    .header-banner::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        opacity: 0.3;
    }
    
    .header-banner h1 {
        font-weight: 700;
        letter-spacing: 0.7px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        margin-bottom: 0.3rem;
        font-size: 2rem;
    }
    
    .header-banner p {
        font-weight: 300;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 5px rgba(0, 0, 0, 0.1);
        font-size: 0.9rem;
    }
    
    /* Section Headers with glowing effect */
    .section-header {
        background-color: var(--card-bg);
        padding: 0.8rem 1.2rem;
        border-left: 5px solid var(--primary-blue);
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(58, 134, 255, 0.15);
    }
    
    .section-header h3 {
        color: var(--text-light);
        position: relative;
        z-index: 2;
        margin: 0;
        font-weight: 600;
    }
    
    .section-header::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(to bottom, var(--primary-blue), var(--accent-purple));
        box-shadow: 0 0 20px 3px rgba(58, 134, 255, 0.5);
        opacity: 0.8;
    }
    
    /* Styled Card for Results with medical theme */
    .card {
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        margin: 1.5rem 0;
        background-color: var(--card-bg);
        border: 1px solid #333;
        position: relative;
        overflow: hidden;
    }
    
    .healthy-card {
        border-left: 5px solid var(--success-green);
    }
    
    .healthy-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at top left, rgba(6, 214, 160, 0.1) 0%, transparent 60%);
    }
    
    .risk-card {
        border-left: 5px solid var(--warning-red);
    }
    
    .risk-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at top left, rgba(239, 71, 111, 0.1) 0%, transparent 60%);
    }
    
    /* Styled Sidebar */
    .css-1d391kg, .css-163ttbj {
        background-color: var(--card-bg);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
    }
    
    /* Button Styling with modern gradient */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-blue), var(--accent-purple));
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(58, 134, 255, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 25px rgba(58, 134, 255, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .stButton>button::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s ease;
    }
    
    .stButton>button:hover::after {
        transform: translateX(100%);
    }
    
    /* Slider Customization with glowing effect */
    .stSlider > div > div {
        background-color: rgba(58, 134, 255, 0.2);
        border-radius: 10px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-blue), var(--secondary-blue));
        box-shadow: 0 0 10px rgba(58, 134, 255, 0.5);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-blue), var(--secondary-blue));
        box-shadow: 0 0 10px rgba(58, 134, 255, 0.3);
    }
    
    /* Improved spacing between sections */
    .spacer {
        height: 2rem;
    }
    
    /* Footer styling with modern look */
    .footer {
        text-align: center;
        padding: 1.5rem;
        background-color: var(--card-bg);
        border-radius: 12px;
        margin-top: 3rem;
        font-size: 0.9rem;
        border: 1px solid #333;
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-purple), var(--secondary-blue));
    }
    
    .footer p {
        margin: 0.4rem 0;
        color: var(--text-secondary);
    }
    
    .footer p strong {
        color: var(--text-light);
    }
    
    /* Enhanced recommendation boxes */
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        background-color: var(--card-bg);
        border: 1px solid #333;
        position: relative;
    }
    
    .recommendation-box h4 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .recommendation-box ul {
        padding-left: 1.2rem;
        margin-bottom: 0;
    }
    
    .recommendation-box li {
        margin-bottom: 0.5rem;
    }
    
    /* Input Summary Box styling */
    .summary-box {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #333;
    }
    
    .summary-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #333;
    }
    
    .summary-item:last-child {
        border-bottom: none;
    }
    
    /* Glowing effects on metrics */
    .metric-container {
        position: relative;
        padding: 1rem;
        border-radius: 10px;
        background-color: rgba(30, 30, 30, 0.6);
        border: 1px solid #333;
        overflow: hidden;
    }
    
    .metric-container.healthy::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--success-green), transparent);
        box-shadow: 0 0 15px 2px rgba(6, 214, 160, 0.5);
    }
    
    .metric-container.risk::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--warning-red), transparent);
        box-shadow: 0 0 15px 2px rgba(239, 71, 111, 0.5);
    }
    
    /* Sidebar info styling */
    .info-box {
        background-color: rgba(30, 30, 30, 0.6);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #333;
    }
    
    .info-box h4 {
        margin-top: 0;
        color: var(--secondary-blue);
        font-weight: 500;
    }
    
    /* Medical icon pulse animation */
    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.05);
            opacity: 0.8;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .medical-icon {
        animation: pulse 2s infinite ease-in-out;
        display: inline-block;
    }
    
    /* Custom widget label styling */
    div.stSlider > div:first-child > label {
        color: var(--secondary-blue) !important;
        font-weight: 500 !important;
    }
    
    /* Tooltip style */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: var(--card-bg);
        color: var(--text-light);
        text-align: center;
        border-radius: 6px;
        padding: 0.5rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        width: 200px;
        border: 1px solid #333;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Parameter warning levels */
    .normal-value {
        color: var(--success-green);
    }
    
    .warning-value {
        color: #ff9f1c;
    }
    
    .critical-value {
        color: var(--warning-red);
    }
    
    /* Animated heartbeat monitor line */
    .heartbeat-line {
        height: 2px;
        width: 100%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            var(--primary-blue) 50%, 
            transparent 100%);
        position: relative;
        margin: 20px 0;
        overflow: hidden;
    }
    
    .heartbeat-line::before {
        content: "";
        position: absolute;
        height: 100%;
        width: 100%;
        background: url("data:image/svg+xml,%3Csvg width='200' height='20' viewBox='0 0 200 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0,10 L40,10 L50,2 L60,18 L70,0 L80,10 L120,10 L130,2 L140,18 L150,0 L160,10 L200,10' stroke='%234cc9f0' stroke-width='2' fill='none'/%3E%3C/svg%3E") repeat-x;
        animation: heartbeat 5s linear infinite;
    }
    
    @keyframes heartbeat {
        0% {
            transform: translateX(0);
        }
        100% {
            transform: translateX(-100%);
        }
    }
    
    /* Parameter cards in result section */
    .parameter-card {
        padding: 10px;
        border-radius: 8px;
        background-color: rgba(30, 30, 30, 0.6);
        border: 1px solid #333;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .parameter-card h4 {
        margin: 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    .parameter-card p {
        margin: 5px 0 0 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler with error handling
@st.cache_resource
def load_model():
    try:
        model_path = 'models/trained_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("❌ Model files not found! Please train the model first.")
            st.info("""
            **To fix this:**
            1. Run: `python train_model.py`
            2. Commit and push models to GitHub
            3. Redeploy the app
            """)
            st.stop()
        
        # Load with error handling
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load metadata if exists
        metadata_path = 'models/metadata.pkl'
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            st.sidebar.success(f"Model loaded (sklearn {metadata.get('sklearn_version', 'unknown')})")
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Common fixes:**
        1. Retrain model with: `python train_model.py`
        2. Ensure scikit-learn versions match in requirements.txt
        3. Check model file compatibility
        """)
        st.stop()

model, scaler = load_model()

# Custom header with modern gradient banner and pulsing icon
st.markdown("""
<div class="header-banner">
    <span class="medical-icon" style="font-size: 1.3rem;">🩺</span>
    <h1>Diabetes Prediction System</h1>
    <p>Advanced AI-Powered Health Risk Assessment Tool</p>
    <div style="position: absolute; bottom: 3px; right: 8px; font-size: 0.6rem; opacity: 0.7;">v2.0</div>
</div>
""", unsafe_allow_html=True)

# Sidebar with modern styling and enhanced information
st.sidebar.markdown("""
<div class="section-header">
    <h3>📋 About This Tool</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="info-box">
    <p><strong>Developed By:</strong> Mayank Ninawe and Mahimna Bhuse</p>
    <p><strong>Accuracy:</strong> <span style="color: #4cc9f0;">~77%</span> based on clinical data</p>
    <p><strong>Purpose:</strong> Early diabetes risk screening</p>
    <div style="height: 1px; background: linear-gradient(to right, #3a86ff, transparent); margin: 10px 0;"></div>
    <p style="font-size: 0.9rem;">This tool analyzes key health indicators to estimate diabetes risk using advanced machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="spacer"></div>
<div class="section-header">
    <h3>🩺 Health Parameters</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="info-box" style="font-size: 0.9rem;">
    <p><span style="color: #4cc9f0;">•</span> <strong>Pregnancies:</strong> Number of pregnancies</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>Glucose:</strong> Blood sugar level</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>Blood Pressure:</strong> Resting blood pressure</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>Skin Thickness:</strong> Skin fold measurement</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>Insulin:</strong> Blood insulin level</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>BMI:</strong> Body Mass Index</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>Family History:</strong> Diabetes genetic factor</p>
    <p><span style="color: #4cc9f0;">•</span> <strong>Age:</strong> Patient's age in years</p>
</div>
""", unsafe_allow_html=True)

# Add model stats visualizer to sidebar with glow effect
st.sidebar.markdown("""
<div class="spacer"></div>
<div class="section-header">
    <h3>📊 Model Statistics</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="info-box">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span>Accuracy:</span>
        <div style="width: 60%; background: rgba(58, 134, 255, 0.2); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: 77%; background: linear-gradient(90deg, #3a86ff, #4cc9f0); height: 100%; border-radius: 4px;"></div>
        </div>
        <span>77%</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span>Precision:</span>
        <div style="width: 60%; background: rgba(58, 134, 255, 0.2); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: 73%; background: linear-gradient(90deg, #3a86ff, #4cc9f0); height: 100%; border-radius: 4px;"></div>
        </div>
        <span>73%</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>Recall:</span>
        <div style="width: 60%; background: rgba(58, 134, 255, 0.2); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: 69%; background: linear-gradient(90deg, #3a86ff, #4cc9f0); height: 100%; border-radius: 4px;"></div>
        </div>
        <span>69%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content - Two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="section-header">
        <h3>👤 Enter Patient Details</h3>
    </div>
    <p style="margin-bottom: 20px; color: #b0b0b0; font-size: 0.9rem;">
        Adjust the sliders below to input patient health parameters for analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Create input fields in 2 columns with enhanced styling
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown("""
        <div style="background-color: rgba(58, 134, 255, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #333;">
            <h4 style="margin-top: 0; color: #4cc9f0; font-size: 1rem;">Patient Background</h4>
        </div>
        """, unsafe_allow_html=True)
        
        pregnancies = st.slider("Pregnancies", 0, 20, 1, help="Number of times pregnant")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        glucose = st.slider("Glucose Level", 0, 200, 120, 
                           help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        blood_pressure = st.slider("Blood Pressure", 0, 150, 70,
                                 help="Diastolic blood pressure (mm Hg)")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20,
                                 help="Triceps skin fold thickness (mm)")
    
    with input_col2:
        st.markdown("""
        <div style="background-color: rgba(58, 134, 255, 0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #333;">
            <h4 style="margin-top: 0; color: #4cc9f0; font-size: 1rem;">Clinical Measurements</h4>
        </div>
        """, unsafe_allow_html=True)
        
        insulin = st.slider("Insulin Level", 0, 900, 80,
                          help="2-Hour serum insulin (mu U/ml)")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1,
                      help="Body mass index (weight in kg/(height in m)²)")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        dpf = st.slider("Family History Factor", 0.0, 2.5, 0.5, 0.01,
                      help="Diabetes pedigree function (genetic influence score)")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        
        age = st.slider("Age", 21, 100, 30,
                      help="Age of the patient in years")

with col2:
    st.markdown("""
    <div class="section-header">
        <h3>📋 Patient Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create more visually appealing summary with value indicators
    input_data = {
        'Pregnancies': [pregnancies, 20],
        'Glucose': [glucose, 200],
        'Blood Pressure': [blood_pressure, 150],
        'Skin Thickness': [skin_thickness, 100],
        'Insulin': [insulin, 900],
        'BMI': [bmi, 70],
        'Family History': [dpf, 2.5],
        'Age': [age, 100]
    }
    
    st.markdown("""
    <div class="summary-box">
    """, unsafe_allow_html=True)
    
    for key, values in input_data.items():
        value = values[0]
        max_val = values[1]
        # Calculate percentage for the progress indicator
        percentage = (value / max_val) * 100
        indicator_color = "#3a86ff"
        
        # Special coloring for certain values
        if key == "Glucose" and value > 140:
            indicator_color = "#ef476f"
        elif key == "BMI" and value > 30:
            indicator_color = "#ef476f"
        elif key == "Blood Pressure" and value > 90:
            indicator_color = "#ef476f"
            
        st.markdown(f"""
        <div class="summary-item">
            <span><b>{key}:</b></span>
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 10px;">{value}</span>
                <div style="width: 50px; height: 5px; background-color: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                    <div style="width: {percentage}%; height: 100%; background: {indicator_color};"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)
    
    # Add risk factors legend
    st.markdown("""
    <div style="margin-top: 20px; font-size: 0.8rem; color: #b0b0b0;">
        <p style="margin-bottom: 5px;"><strong>Reference Ranges:</strong></p>
        <p style="margin: 2px 0;"><span style="color: #ef476f;">■</span> Glucose > 140: Elevated</p>
        <p style="margin: 2px 0;"><span style="color: #ef476f;">■</span> BMI > 30: Obesity</p>
        <p style="margin: 2px 0;"><span style="color: #ef476f;">■</span> Blood Pressure > 90: Elevated</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add animated heart rate monitor line
    st.markdown("""
    <div style="margin-top: 30px;">
        <div class="heartbeat-line"></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# Prediction button with improved styling
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <p style="font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;">Click below to analyze the health parameters</p>
</div>
""", unsafe_allow_html=True)

if st.button("🔍 Analyze Diabetes Risk", type="primary", use_container_width=True):
    
    try:
        # Prepare input - ensure correct data types
        input_array = np.array([[
            float(pregnancies), 
            float(glucose), 
            float(blood_pressure), 
            float(skin_thickness),
            float(insulin), 
            float(bmi), 
            float(dpf), 
            float(age)
        ]], dtype=np.float64)
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display results with improved styling
        st.markdown("""
        <div class="section-header">
            <h2>🎯 Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced results card with color coding and animation
        if prediction == 0:
            st.markdown("""
            <div class="card healthy-card">
                <h3>✅ No Diabetes Detected</h3>
                <p>Based on the provided health parameters, the patient shows low risk indicators for diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card risk-card">
                <h3>⚠️ Diabetes Risk Detected</h3>
                <p>Based on the provided health parameters, the patient shows indicators that suggest diabetes risk.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics in a cleaner layout with containers
        result_col1, result_col2, result_col3 = st.columns(3)
        
        confidence = max(prediction_proba) * 100
        risk_score = prediction_proba[1] * 100
        health_status = "Healthy" if prediction == 0 else "At Risk"
        status_color = "var(--success-green)" if prediction == 0 else "var(--warning-red)"
        
        with result_col1:
            st.markdown(f"""
            <div class="metric-container {"healthy" if prediction == 0 else "risk"}">
                <h4 style="margin-top: 0; color: #b0b0b0; font-weight: 400; font-size: 0.9rem;">Confidence Level</h4>
                <p style="font-size: 1.5rem; margin: 5px 0 0 0; font-weight: 700;">{confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div class="metric-container {"healthy" if prediction == 0 else "risk"}">
                <h4 style="margin-top: 0; color: #b0b0b0; font-weight: 400; font-size: 0.9rem;">Risk Score</h4>
                <p style="font-size: 1.5rem; margin: 5px 0 0 0; font-weight: 700;">{risk_score:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with result_col3:
            st.markdown(f"""
            <div class="metric-container {"healthy" if prediction == 0 else "risk"}">
                <h4 style="margin-top: 0; color: #b0b0b0; font-weight: 400; font-size: 0.9rem;">Status</h4>
                <p style="font-size: 1.5rem; margin: 5px 0 0 0; font-weight: 700; color: {status_color};">{health_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Parameters Analysis
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h3>🔍 Key Parameters Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display key parameters that influenced the prediction
        param_cols = st.columns(4)
        
        # Define key parameters and their risk thresholds
        key_params = [
            {"name": "Glucose", "value": glucose, "unit": "mg/dL", "risk": 140, "high_risk": 180},
            {"name": "BMI", "value": bmi, "unit": "kg/m²", "risk": 30, "high_risk": 35},
            {"name": "Blood Pressure", "value": blood_pressure, "unit": "mmHg", "risk": 90, "high_risk": 110},
            {"name": "Family History", "value": dpf, "unit": "score", "risk": 1.0, "high_risk": 1.5}
        ]
        
        for i, param in enumerate(key_params):
            # Determine status color
            if param["value"] >= param["high_risk"]:
                status_class = "critical-value"
                status_text = "High Risk"
            elif param["value"] >= param["risk"]:
                status_class = "warning-value"
                status_text = "Elevated"
            else:
                status_class = "normal-value"
                status_text = "Normal"
                
            with param_cols[i]:
                st.markdown(f"""
                <div class="parameter-card">
                    <h4>{param["name"]}</h4>
                    <p>{param["value"]} {param["unit"]}</p>
                    <p class="{status_class}" style="font-size: 0.8rem;">{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Probability visualization with better styling
        st.markdown("""
        <div class="section-header" style="margin-top: 1.5rem;">
            <h3>📊 Risk Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.markdown("<p style='text-align: center; font-weight: 600; color: #06d6a0; margin-bottom: 8px;'>Healthy Probability</p>", unsafe_allow_html=True)
            st.progress(prediction_proba[0])
            st.markdown(f"<p style='text-align: center; font-weight: 600; font-size: 1.2rem;'>{prediction_proba[0]*100:.2f}%</p>", unsafe_allow_html=True)
        
        with prob_col2:
            st.markdown("<p style='text-align: center; font-weight: 600; color: #ef476f; margin-bottom: 8px;'>Diabetes Probability</p>", unsafe_allow_html=True)
            st.progress(prediction_proba[1])
            st.markdown(f"<p style='text-align: center; font-weight: 600; font-size: 1.2rem;'>{prediction_proba[1]*100:.2f}%</p>", unsafe_allow_html=True)
        
        # Recommendations with improved styling
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h3>💡 Health Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <div class="recommendation-box" style="background-color: rgba(231, 76, 60, 0.1); border-left: 4px solid #ef476f; padding: 1.5rem;">
                <h4 style="color: #ef476f; font-size: 1.2rem;">⚠️ Important Next Steps</h4>
                <ul>
                    <li><strong>Medical Consultation:</strong> Schedule an appointment with a healthcare provider</li>
                    <li><strong>Blood Glucose Monitoring:</strong> Begin regular tracking of blood sugar levels</li>
                    <li><strong>Dietary Changes:</strong> Consider a balanced diet low in refined carbohydrates</li>
                    <li><strong>Physical Activity:</strong> Implement a regular exercise routine (minimum 150 min/week)</li>
                    <li><strong>Regular Screening:</strong> Follow up with HbA1c testing as recommended</li>
                </ul>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; font-size: 0.9rem;">
                    <p style="margin: 0;"><em>Early intervention can significantly reduce diabetes progression and complications.</em></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-box" style="background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #06d6a0; padding: 1.5rem;">
                <h4 style="color: #06d6a0; font-size: 1.2rem;">✅ Healthy Habits to Maintain</h4>
                <ul>
                    <li><strong>Balanced Diet:</strong> Continue a varied diet rich in whole foods and vegetables</li>
                    <li><strong>Regular Exercise:</strong> Maintain physical activity (at least 30 min/day)</li>
                    <li><strong>Regular Checkups:</strong> Schedule annual health assessments</li>
                    <li><strong>Weight Management:</strong> Monitor and maintain healthy BMI</li>
                    <li><strong>Limited Sugar Intake:</strong> Minimize consumption of processed sugars</li>
                </ul>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; font-size: 0.9rem;">
                    <p style="margin: 0;"><em>Preventive health measures are key to maintaining long-term well-being.</em></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Analysis Error: {str(e)}")
        st.info("Please verify your input values and try again.")

# Enhanced footer with better styling
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p><strong>🩺 Diabetes Prediction System</strong></p>
    <p>Developed by Mayank Ninawe and Mahimna Bhuse| ML Capstone Project 2025</p>
    <div style="width: 150px; height: 1px; background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent); margin: 10px auto;"></div>
    <p style="color: #b0b0b0; font-size: 0.8rem; margin-top: 0.5rem;">
        <em>⚠️ This tool is for screening purposes only and not a substitute for professional medical diagnosis.</em>
    </p>
    <p style="color: #4cc9f0; font-size: 0.8rem; margin-top: 0.5rem;">
        Always consult with healthcare professionals for medical advice.
    </p>
</div>
""", unsafe_allow_html=True)