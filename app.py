# Power System Fault Detection and Classification Dashboard
# Author: Priya Pandey

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Power Fault Detection", layout="wide")
st.title("⚡ Power System Fault Detection and Classification")
st.markdown("### Electrical Engineering | Machine Learning Project")

# --- File Upload Section ---
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your power system dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # --- Load Data ---
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.write(data.head())

    # --- Data Information ---
    st.markdown("**Dataset Shape:** " + str(data.shape))
    st.markdown("**Columns:** " + ", ".join(data.columns))

    # Select target column
    target_col = st.sidebar.selectbox("Select Target Column (Fault Type)", data.columns)

    if st.sidebar.button("Train Model"):
        # --- Encode target if categorical ---
        le = None
        if data[target_col].dtype == 'object':
            le = LabelEncoder()
            data[target_col] = le.fit_transform(data[target_col])

        # Prepare features X and target y
        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()

        # Detect object (string) columns
        obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # We'll try to auto-handle them:
        # - If column name suggests an ID or if every value is unique -> drop it
        # - If high cardinality (>50 unique values) -> drop it (likely non-informative)
        # - If low cardinality -> label-encode and keep (store encoder for mapping)
        encoders = {}
        dropped_cols = []

        for col in obj_cols:
            nunique = X[col].nunique(dropna=False)
            lname = col.lower()
            # treat as ID if column name contains 'id' or starts with 'f' followed by digits (e.g., F001),
            # or if every value is unique, or if very high cardinality
            if ('id' in lname) or nunique == len(X) or nunique > 50 or (len(lname) >= 2 and lname[0] == 'f' and X[col].astype(str).str.match(r'^f\d+', case=False).all()):
                dropped_cols.append(col)
                X.drop(columns=[col], inplace=True)
            else:
                # safe to label-encode small-cardinality categorical features
                enc = LabelEncoder()
                X[col] = enc.fit_transform(X[col].astype(str))
                encoders[col] = enc

        # Inform user what happened
        if dropped_cols:
            st.warning(f"Dropped columns likely non-numeric/ID/high-cardinality: {', '.join(dropped_cols)}")
        if encoders:
            st.info("Encoded categorical columns: " + ", ".join(encoders.keys()))
            # show mapping for each encoded column (optional, helpful)
            for col, enc in encoders.items():
                mapping = {int(i): label for i, label in enumerate(enc.classes_)}
                st.write(f"**{col}** mapping:", mapping)

        # Now keep only numeric columns (extra safety)
        X = X.select_dtypes(include=[np.number])

        # Check there are features left
        if X.shape[1] == 0:
            st.error("No numeric features available after preprocessing. Please upload a dataset with numeric measurement columns or preprocess the data to include numeric features.")
        else:
            # --- Scaling ---
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- Train-test split ---
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # --- Train Random Forest ---
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- Evaluation ---
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained successfully with Accuracy: **{acc*100:.2f}%**")

            # (rest of your existing plotting / prediction code follows)
            # IMPORTANT: When creating the manual prediction inputs below, use X.columns (the processed numeric columns)

        with st.spinner("Training model... please wait ⏳"):

            # --- Encode target if categorical ---
            le = None
            if data[target_col].dtype == 'object':
                le = LabelEncoder()
                data[target_col] = le.fit_transform(data[target_col])

            X = data.drop(columns=[target_col])
            y = data[target_col]

            # --- Scaling ---
            # --- Ensure only numeric features are used ---
            X = X.select_dtypes(include=[np.number])

            # (Optional) Warn user if anything got dropped
            non_numeric = [col for col in data.drop(columns=[target_col]).columns if col not in X.columns]
            if non_numeric:
                st.warning(f"⚠️ Dropped non-numeric columns: {', '.join(non_numeric)} (e.g., IDs or text labels)")

            # --- Scaling ---
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)


            # --- Train-test split ---
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # --- Train Random Forest ---
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- Evaluation ---
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained successfully with Accuracy: **{acc*100:.2f}%**")

            # --- Confusion Matrix ---
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
            st.pyplot(fig)

            # --- Classification Report ---
            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # --- Feature Importance ---
            st.subheader("💡 Top Important Features")
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
            st.bar_chart(feature_importance.sort_values(ascending=False).head(10))

            # --- Prediction Section ---
            st.subheader("🔮 Predict Fault Type (Enter New Data)")
            input_values = []
            col1, col2 = st.columns(2)
            with col1:
                for feature in X.columns[:len(X.columns)//2]:
                    val = st.number_input(f"{feature}", value=0.0)
                    input_values.append(val)
            with col2:
                for feature in X.columns[len(X.columns)//2:]:
                    val = st.number_input(f"{feature}", value=0.0)
                    input_values.append(val)

            if st.button("Predict Fault"):
                input_array = np.array(input_values).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)

                if le:
                    predicted_label = le.inverse_transform(prediction)[0]
                else:
                    predicted_label = prediction[0]

                st.success(f"🔎 Predicted Fault Type: **{predicted_label}**")

else:
    st.info("👈 Upload a dataset from the sidebar to begin.")
