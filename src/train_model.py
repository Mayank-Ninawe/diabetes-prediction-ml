import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sklearn

print("🚀 Starting model training...\n")
print(f"📦 scikit-learn version: {sklearn.__version__}\n")

# Create models folder if not exists
if not os.path.exists('../models'):
    os.makedirs('../models')
    print("✅ Created 'models' folder\n")

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv('../data/diabetes.csv')
print(f"✅ Dataset loaded: {df.shape}\n")

# Handle missing values (zeros)
print("🔧 Preprocessing data...")
df_clean = df.copy()
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    median_val = df_clean[df_clean[col] != 0][col].median()
    df_clean[col] = df_clean[col].replace(0, median_val)

print("✅ Missing values handled\n")

# Separate features and target
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🔀 Data split complete")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}\n")

# Feature scaling
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling complete\n")

# Train model with explicit parameters
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete\n")

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy*100:.2f}%\n")

# Save metadata
metadata = {
    'sklearn_version': sklearn.__version__,
    'accuracy': accuracy,
    'n_features': X.shape[1],
    'feature_names': list(X.columns)
}

# Save model and scaler with protocol 4 (better compatibility)
print("💾 Saving model and scaler...")
joblib.dump(model, '../models/trained_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(metadata, '../models/metadata.pkl')
print("✅ Model saved: models/trained_model.pkl")
print("✅ Scaler saved: models/scaler.pkl")
print("✅ Metadata saved: models/metadata.pkl\n")

# Test saved model
print("🔍 Testing saved model...")
loaded_model = joblib.load('../models/trained_model.pkl')
loaded_scaler = joblib.load('../models/scaler.pkl')

test_sample = [[2, 100, 70, 20, 80, 23.5, 0.4, 25]]
test_scaled = loaded_scaler.transform(test_sample)
test_pred = loaded_model.predict(test_scaled)

print(f"✅ Model test successful!")
print(f"   Test prediction: {'Diabetic' if test_pred[0] == 1 else 'Non-Diabetic'}\n")

print("🎉 ALL DONE! Model ready for Streamlit app!")
print(f"\n📌 Model trained with scikit-learn {sklearn.__version__}")
print("Run: streamlit run app.py")
