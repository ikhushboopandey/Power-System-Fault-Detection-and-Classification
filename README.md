# ⚡ Power System Fault Detection and Classification

### 🧠 Machine Learning Project

This project focuses on building a **machine learning model** to detect and classify different types of **faults in a power distribution system** using electrical measurement data such as **voltage and current phasors**.

The system distinguishes between **normal operating conditions** and various **fault conditions** (like line-to-ground, line-to-line, or three-phase faults).
It enables **rapid and accurate fault identification**, crucial for maintaining **power grid stability and reliability**.

---

## 🚀 Problem Statement

Design a machine learning model that can automatically **identify and classify faults** in an electrical power system. The goal is to improve fault detection time and ensure system reliability using real-time sensor data.

---

## 💡 Proposed Solution

* Collect and preprocess data from the [Power System Faults Dataset](https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset).
* Extract and scale electrical parameters (voltages, currents, angles).
* Train a classification model (Random Forest, SVM, or others) to predict fault type.
* Evaluate model accuracy and visualize performance metrics.
* Deploy a simple **Python Streamlit dashboard** for interactive prediction and visualization.

---

## 🧩 System Development Approach

1. **Data Preprocessing:** Handle missing data, encode categorical values, and normalize numeric features.
2. **Feature Scaling:** Use `StandardScaler` for consistent feature scaling.
3. **Model Training:** Apply `RandomForestClassifier` for fault type prediction.
4. **Evaluation:** Measure accuracy, confusion matrix, and classification report.
5. **Deployment:** Build an interactive dashboard using **Streamlit**.

---

## ⚙️ Algorithm & Tools

* **Algorithm:** Random Forest Classifier
* **Libraries Used:**

  * `pandas`, `numpy` – Data handling
  * `matplotlib`, `seaborn` – Visualization
  * `scikit-learn` – Machine learning and preprocessing
  * `streamlit` – Web app dashboard

---

## 📊 Results

* The model successfully classified multiple fault types with high accuracy.
* Output includes visualizations for:

  * Confusion matrix
  * Feature importance
  * Fault prediction results for sample input

*(Insert output graph or confusion matrix image here)*

---

## 🧾 Conclusion

The developed model provides a reliable and efficient method for detecting and classifying faults in power systems.
It reduces manual monitoring effort and enhances the **stability and safety** of electrical grids.

---

## 🔮 Future Scope

* Integrate the model with **real-time IoT-based monitoring systems**.
* Use **deep learning** for improved classification accuracy.
* Extend fault classification to include **transient and complex faults**.
* Deploy the model on **cloud platforms** for large-scale industrial use.

---

## 📚 References

* Kaggle Dataset: [Power System Faults Dataset](https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset)
* Scikit-learn Documentation
* IEEE Research Papers on Power System Fault Detection# Power-System-Fault-Detection-and-Classification
This project uses machine learning to detect and classify power system faults based on voltage and current data. A Random Forest model analyzes patterns to identify fault types quickly, improving grid reliability. The system is deployed with an interactive Streamlit dashboard for visualization.
