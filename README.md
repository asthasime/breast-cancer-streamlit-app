# 🧬 Breast Cancer Prediction App

A **machine learning web application** built using **Python, Streamlit, and Scikit-learn** to predict whether a tumor is **Benign or Malignant**, along with probability and explainability insights.

---

## 🚀 Problem Statement

Early detection of breast cancer is critical for effective treatment.
This project aims to:

* Predict tumor type using clinical features
* Provide **probability-based risk assessment**
* Offer **interpretable insights** for better understanding

---

## ⚙️ Solution Overview

The application provides two main functionalities:

### 📁 1. Upload CSV (Batch Prediction)

* Upload dataset containing multiple patient records
* Get predictions (Benign / Malignant)
* View probability scores
* Download results as CSV
* Visualize overall trends and feature importance

---

### ✍️ 2. Manual Input (Single Prediction)

* Enter feature values manually
* Get prediction with confidence score
* View risk level:

  * 🟢 Low Risk
  * 🟠 Moderate Risk
  * 🔴 High Risk
* Understand key factors influencing the prediction

---

## 🧠 Model Details

* **Algorithm:** Logistic Regression
* **Preprocessing:** StandardScaler
* **Input:** Clinical features (cell measurements)

**Output:**

* Prediction (Benign / Malignant)
* Probability score

---

## 📊 Explainability (SHAP)

To improve model transparency:

* Implemented **SHAP (LinearExplainer)**
* Identified **top contributing features**
* Converted technical feature names into **user-friendly terms**
* Provided **simple explanations** for non-technical users

**Example insights:**

* “Higher cell size increases cancer risk”
* “Lower symmetry reduces risk”

---

## 🎨 Key Features

* 📁 CSV Upload & Batch Prediction
* ✍️ Manual Input Interface
* 📊 Data Visualization (Matplotlib)
* 🧠 Explainable AI (SHAP)
* 🎯 Risk Categorization
* 🧾 Downloadable Results
* 💡 User-friendly interpretations

---

## 🐳 Docker Support

The application is containerized using Docker for easy deployment.

### 🔧 Build Image

```bash
docker build -t breast-cancer-app .
```

### ▶️ Run Container

```bash
docker run -p 8501:8501 breast-cancer-app
```

Then open in browser:
http://localhost:8501

---

## 💻 Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* SHAP
* Docker

---

## ☁️ Future Improvements

* Deploy on cloud platforms (AWS / Azure / Streamlit Cloud)
* Convert model into API using FastAPI
* Add database integration for real-time data
* Enhance UI/UX for clinical usability
* Advanced explainability (SHAP dashboards, summaries)

---

## 📌 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👩‍💻 Author

**Astha Pandey**
Machine Learning & Python Developer

🔗 GitHub: https://github.com/asthasime

---

## 💡 Notes

This project demonstrates:

* End-to-end machine learning workflow
* Model deployment using Streamlit
* Docker-based containerization
* Explainable AI (SHAP)
* User-focused application design
