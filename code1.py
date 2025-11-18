# Power System Fault Detection and Classification
# Author: Priya Pandey (Electrical Engineering ML Project)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 2: Load Dataset
data = pd.read_csv("power_system_faults_dataset.csv")

# Step 3: Explore Dataset
print("Data Shape:", data.shape)
print("Columns:", data.columns)
print("\nSample Data:\n", data.head())

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# Step 4: Preprocessing
# Encode target column (if categorical)
label_col = 'Fault Type'  
if data[label_col].dtype == 'object':
    le = LabelEncoder()
    data[label_col] = le.fit_transform(data[label_col])

# Separate features and target
X = data.drop(columns=[label_col])
y = data[label_col]

# Keep only numeric columns
X = X.select_dtypes(include=['float64', 'int64'])

# Normalize feature values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Train Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Power System Fault Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Feature Importance (to understand which electrical features matter most)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Important Features")
plt.show()

# Step 10: Predict new data sample (example)
# Replace the example array with real electrical measurement values
sample_input = np.array([0.95, 1.03, -0.12, 0.8, 1.1, 0.9, 0.75]).reshape(1, -1)
sample_input_scaled = scaler.transform(sample_input)
predicted_fault = model.predict(sample_input_scaled)
print("\nPredicted Fault Type:", le.inverse_transform(predicted_fault)[0])
