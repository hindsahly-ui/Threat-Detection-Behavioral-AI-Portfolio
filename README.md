# HIND OMAR SAHLI - Intelligent Threat Detection using AI & Behavioral Analysis
# Technical Portfolio for Saipem GDP - Supply Chain Data Analyst
# Contact: Riyadh, KSA | Academy of Learning | 2026

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Dataset: 5000 simulated network records (41 features) mimicking NSL-KDD
np.random.seed(42)
n_samples = 5000
n_features = 41

# محاكاة 41 feature مثل NSL-KDD
X = np.random.randn(n_samples, n_features)
# إضافة بعض الأنماط السلوكية
y = (X[:, 0] + X[:, 1]*1.5 + X[:, 5] - X[:, 10] > 1).astype(int)
# 0 = Normal, 1 = Anomaly/Attack
y = np.where(np.random.rand(n_samples) < 0.1, 1 - y, y) # 10% noise

feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['label'].value_counts()}")

# 2. Tools: Python, Pandas, Scikit-learn, StandardScaler
X = df.drop('label', axis=1)
y = df['label']

# 3. Model: Random Forest n_estimators=100, train_test_split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Results & Evaluation
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n" + "="*50)
print(f"Results: {accuracy*100:.2f}% Accuracy (simulated) - Target 96.5% with real NSL-KDD")
print(f"Confusion Matrix: {TN} TN | {TP} TP | {FP} FP ({FP/len(y_test)*100:.1f}%) | {FN} FN")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# 5. Skills: Behavioral Baseline, Anomaly Detection, Feature Engineering
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Behavioral Baseline):")
print(importances.head(10))
