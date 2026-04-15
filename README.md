# 🧬 Breast Cancer Prediction App

This is a Machine Learning web application built using **Streamlit** that predicts whether a tumor is **Benign or Malignant** based on clinical features.

---

## 🚀 Features

- Interactive UI built with Streamlit
- Logistic Regression model for prediction
- Feature grouping (Mean, Standard Error, Worst)
- Real-time prediction with probability score
- Visualization of prediction confidence
- Bulk prediction via CSV upload
- Dockerized for easy deployment

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Problem Type: Binary Classification
- Input: 30 numerical features from breast cancer dataset
- Output:
  - 0 → Benign
  - 1 → Malignant

---

## 📁 Project Structure
STREAMLIT/
│
├── app.py # Streamlit application
├── model.pkl # Trained ML model
├── requirements.txt # Dependencies
├── Dockerfile # Docker configuration
├── .gitignore # Ignored files
└── README.md # Project documentation

---

## ▶️ Run Locally

### 1. Install dependencies
pip install -r requirements.txt


### 2. Run the app
streamlit run app.py

### 3. Open in browser
http://localhost:8501


---

## 🐳 Run with Docker

### 1. Build Docker image
docker build -t breast-cancer-app .


### 2. Run container
docker run -p 8501:8501 breast-cancer-app


---

## 📊 Sample Features Used

- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Smoothness Mean
- (and more...)

---

## 💡 Key Highlights

- Ensured correct feature alignment using column reindexing
- Built user-friendly UI using tabs and sliders
- Added prediction probability visualization
- Designed for real-world usability and scalability

---

## 🔗 Author

**Astha Pandey**

GitHub: https://github.com/asthasime

---

## ⭐ Future Improvements

- Add model explainability (SHAP/LIME)
- Deploy on cloud (Streamlit Cloud / AWS)
- Improve UI with advanced visualizations

---