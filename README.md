"# 🩺 Diabetes Prediction System

**Machine Learning Capstone Project**  
Course: ECSP5004 - Machine Learning Lab  
Group ID: 2 | Section: V_A  
Institution: RCOEM, Nagpur | Semester: Odd 2024-25

---

## 👥 Team Members

| Name            | Roll Number | Email               | Contribution |
|-----------------|-------------|---------------------|--------------|
| Mayank Ninawe   | A3-39       | ninawemh@rknec.edu  | 50%          |
| Mahimna Bhuse   | A3-38       | bhusems@rknec.edu   | 50%          |

---

## 📌 Project Overview

A machine learning-based diabetes prediction system using **Random Forest Classifier** to predict diabetes risk from patient health parameters with **77% accuracy**. The system features an interactive **Streamlit web application** for real-time predictions and comprehensive data analysis through Jupyter notebooks.

**Keywords:** Diabetes Prediction, Random Forest, Healthcare ML, Classification, Streamlit Deployment

---

## 🎯 Objectives

- Develop accurate diabetes prediction model using ML techniques
- Handle missing data and perform feature engineering
- Create user-friendly GUI for healthcare professionals
- Deploy cloud-based system for real-world accessibility
- Demonstrate complete end-to-end ML pipeline

---

## 📊 Dataset Information

- **Source:** Pima Indians Diabetes Database (Kaggle)
- **Samples:** 768 female patients of Pima Indian heritage
- **Features:** 8 health parameters (Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Target:** Binary classification (0 = No Diabetes, 1 = Diabetes)
- **Class Distribution:** 500 non-diabetic (65%), 268 diabetic (35%)
- **Train-Test Split:** 80-20 stratified split

---

## 🔧 Tech Stack

| Category        | Technologies                  |
|-----------------|-------------------------------|
| Language        | Python 3.11                   |
| ML Framework    | Scikit-learn 1.3.2            |
| Web Framework   | Streamlit 1.39.0              |
| Data Processing | Pandas, NumPy                 |
| Visualization   | Matplotlib, Seaborn           |
| Development     | Jupyter Notebook, Git, GitHub |
| Deployment      | Streamlit Cloud               |

---

## 📁 Repository Structure

```
diabetes-prediction-ml/
│
├── data/
│   ├── diabetes.csv              # Pima Indians Diabetes Dataset
│   └── data_source.txt           # Dataset source and citation
│
├── models/
│   ├── trained_model.pkl         # Random Forest model (sklearn 1.3.2)
│   ├── scaler.pkl                # StandardScaler object
│   └── metadata.pkl              # Model metadata (version, accuracy)
│
├── notebooks/
│   └── diabetes_model_training.ipynb  # Complete ML pipeline notebook
│
├── src/
│   ├── app.py                    # Streamlit web application
│   └── train_model.py            # Model training script
│
├── screenshots/                  # Application screenshots for report
│   ├── 01_home_interface.png
│   ├── 02_healthy_prediction.png
│   ├── 03_diabetic_prediction.png
│   └── 04_notebook_results.png
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                   # Git ignore rules
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (for cloning)

### Quick Start

```bash
# 1. Clone Repository
git clone https://github.com/Mayank-Ninawe/diabetes-prediction-ml.git
cd diabetes-prediction-ml

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run Jupyter Notebook (Optional - for training)
cd notebooks
jupyter notebook diabetes_model_training.ipynb
# Run all cells to train model

# 4. Run Streamlit Application
cd ..
streamlit run src/app.py
```

App will open at: http://localhost:8501

---

## 📈 Model Performance

| Metric                | Value  | Description                      |
|-----------------------|--------|----------------------------------|
| Accuracy              | 77.27% | Overall correct predictions      |
| Precision (Diabetic)  | 0.73   | True positives / (TP + FP)      |
| Recall (Diabetic)     | 0.69   | True positives / (TP + FN)      |
| F1-Score (Diabetic)   | 0.71   | Harmonic mean of precision & recall |
| AUC-ROC               | 0.83   | Area under ROC curve            |

**Algorithm Details:**
- **Model:** Random Forest Classifier
- **Parameters:** n_estimators=100, max_depth=10, min_samples_split=10
- **Training Time:** ~2-3 seconds
- **Inference Time:** <2 seconds per prediction

**Feature Importance:**
- Glucose (42%) - Most predictive feature
- BMI (18%) - Second most important
- Age (12%) - Third most important
- Other features (28%)

---

## 🧪 Testing & Validation

**Test Cases Passed:**
- ✅ Data preprocessing and missing value handling
- ✅ Model training with fixed random seed (reproducible)
- ✅ Prediction accuracy on test set
- ✅ GUI functionality and user interactions
- ✅ Edge cases (boundary values, invalid inputs)
- ✅ Cross-validation results consistent

**Sample Test Cases:**

```python
# Healthy Patient (Expected: No Diabetes)
Input: [Pregnancies=2, Glucose=100, BP=70, Skin=20, 
        Insulin=80, BMI=23.0, DPF=0.4, Age=25]
Output: ✅ No Diabetes (Confidence: 85%)

# High-Risk Patient (Expected: Diabetes)
Input: [Pregnancies=8, Glucose=180, BP=90, Skin=35, 
        Insulin=150, BMI=35.0, DPF=1.2, Age=55]
Output: ❌ Diabetes Detected (Confidence: 78%)
```

---

## 🌐 Deployment

**Live Demo:** https://diabetes-prediction-ml-proj.streamlit.app/

**Deployment Features:**
- Cloud-hosted on Streamlit Cloud (free tier)
- 24/7 accessibility from any device
- Automatic HTTPS encryption
- Real-time predictions with <2 second response time
- Mobile-responsive design
- No client-side installation required

**Why Cloud Deployment vs Hardware (Raspberry Pi)?**

While the project guidelines suggested hardware deployment, we opted for cloud-based deployment because:
- **Accessibility:** Available globally, not limited to local network
- **Scalability:** Handles multiple concurrent users
- **Cost-Effective:** No hardware purchase/maintenance needed
- **Real-World Applicability:** Suitable for clinic/hospital web portals
- **Zero Setup:** Healthcare professionals can access instantly

---

## ✅ Reproducibility Checklist

- ✅ Random seeds fixed (random_state=42 throughout)
- ✅ requirements.txt with exact versions included
- ✅ All paths are relative (no absolute paths)
- ✅ Dataset included in repository
- ✅ Model weights saved and versioned
- ✅ Notebook runs end-to-end without errors
- ✅ Clear documentation and comments
- ✅ Git repository with version history
- ✅ Results reproducible on test set

**To Reproduce Results:**
```bash
# From project root
python src/train_model.py
# Expected: Accuracy ~77%, saved models in models/
```

---

## 📸 Screenshots

Complete application screenshots available in `screenshots/` folder:
- Home interface with input controls
- Healthy patient prediction result
- Diabetic patient prediction result
- Jupyter notebook analysis outputs

---

## 🔮 Future Enhancements

**Phase 1 (Short-term):**
1. Add explainability using SHAP/LIME
2. Implement patient history tracking
3. Add PDF report generation

**Phase 2 (Medium-term):**
4. Mobile application development (Flutter/React Native)
5. Integration with hospital management systems
6. Multi-language support

**Phase 3 (Long-term):**
7. Real-time patient monitoring dashboard
8. Multi-disease prediction (heart disease, hypertension)
9. Integration with wearable devices

---

## 📚 References

1. Smith, J.W., et al. (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus." Proceedings of the Symposium on Computer Applications and Medical Care, pp. 261-265.
2. Pima Indians Diabetes Database - UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/diabetes
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
4. Streamlit Documentation: https://docs.streamlit.io/
5. Random Forest Algorithm - Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.

---

## 🤝 Acknowledgments

- **Instructor:** Prof. Vikas Gupta
- **Department:** Electronics and Computer Science Engineering
- **Institution:** Shri Ramdeobaba College of Engineering and Management, Nagpur

Special thanks to:
- Kaggle community for the dataset
- Streamlit team for the deployment platform
- Open-source ML community

---

## 📄 License

MIT License - Academic Project

This project is developed as part of the Machine Learning Lab course at RCOEM, Nagpur. The code is open-source and available for educational purposes.

---

## 📧 Contact & Support

For queries or collaboration:
- **Mayank Ninawe:** ninawemh@rknec.edu
- **Mahimna Bhuse:** bhusems@rknec.edu
- **GitHub Repository:** https://github.com/Mayank-Ninawe/diabetes-prediction-ml

---

## 📝 Submission Information

- **Submission Date:** October 27, 2025
- **Report Format:** Google Docs + Spiral-bound hard copy
- **Code Submission:** ZIP file with complete repository
- **Evaluation:** As per ECSP5004 Capstone Project guidelines

---

## ⚠️ Disclaimer

This is a prediction tool developed for educational purposes. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and treatment decisions.

---

**Project Status:** ✅ Complete & Deployed  
**Version:** 1.0  
**Last Updated:** October 25, 2025" 
