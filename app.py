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

# Title and description
st.title("🩺 Diabetes Prediction System")
st.markdown("### ML-Based Health Risk Assessment Tool")
st.markdown("---")

# Sidebar - Project Info
st.sidebar.header("📋 About Project")
st.sidebar.info("""
**Developed By:** Mayank Ninawe  
**Algorithm:** Random Forest Classifier  
**Accuracy:** ~77-78%  
**Dataset:** Pima Indians Diabetes Database  

This system predicts diabetes risk based on 8 health parameters.
""")

st.sidebar.markdown("---")
st.sidebar.header("📊 Feature Information")
st.sidebar.markdown("""
- **Pregnancies:** Number of times pregnant
- **Glucose:** Plasma glucose concentration
- **Blood Pressure:** Diastolic BP (mm Hg)
- **Skin Thickness:** Triceps skin fold (mm)
- **Insulin:** 2-Hour serum insulin (mu U/ml)
- **BMI:** Body mass index (weight/height²)
- **DPF:** Diabetes pedigree function
- **Age:** Age in years
""")

# Main content - Two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔬 Enter Patient Details")
    
    # Create input fields in 2 columns
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose Level", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    
    with input_col2:
        insulin = st.slider("Insulin Level", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
        age = st.slider("Age", 21, 100, 30)

with col2:
    st.header("📊 Input Summary")
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DPF': dpf,
        'Age': age
    }
    st.json(input_data)

st.markdown("---")

# Prediction button
if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
    
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
        
        # Display results
        st.markdown("## 🎯 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 0:
                st.success("### ✅ No Diabetes")
                st.markdown("**Status:** Healthy")
            else:
                st.error("### ❌ Diabetes Detected")
                st.markdown("**Status:** High Risk")
        
        with result_col2:
            confidence = max(prediction_proba) * 100
            st.metric("Confidence Level", f"{confidence:.2f}%")
        
        with result_col3:
            risk_score = prediction_proba[1] * 100
            st.metric("Diabetes Risk", f"{risk_score:.2f}%")
        
        # Probability bar
        st.markdown("### 📊 Prediction Probability")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.progress(prediction_proba[0])
            st.write(f"**No Diabetes:** {prediction_proba[0]*100:.2f}%")
        
        with prob_col2:
            st.progress(prediction_proba[1])
            st.write(f"**Diabetes:** {prediction_proba[1]*100:.2f}%")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Health Recommendations")
        
        if prediction == 1:
            st.warning("""
            **⚠️ Important Recommendations:**
            - Consult a healthcare professional immediately
            - Monitor blood glucose levels regularly
            - Maintain a healthy diet and exercise routine
            - Consider lifestyle modifications
            - Get regular health checkups
            """)
        else:
            st.info("""
            **✅ Preventive Measures:**
            - Maintain current healthy lifestyle
            - Regular physical activity (30 min/day)
            - Balanced diet with low sugar intake
            - Annual health checkups recommended
            - Monitor weight and BMI regularly
            """)
    
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🩺 Diabetes Prediction System</strong></p>
    <p>Developed by Mayank Ninawe | ML Capstone Project 2025</p>
    <p><em>⚠️ This is a prediction tool. Always consult healthcare professionals for medical advice.</em></p>
</div>
""", unsafe_allow_html=True)
