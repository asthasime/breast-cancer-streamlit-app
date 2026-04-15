import streamlit as st
import pickle
import pandas as pd

# -------------------- LOAD MODEL --------------------
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
columns = data['columns']

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🧬 Breast Cancer Prediction App")
st.write("Enter the values below to predict whether the tumor is **Benign or Malignant**.")

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Controls")
st.sidebar.info("Adjust values and click Predict")

# -------------------- FEATURE GROUPING --------------------
mean_cols = [col for col in columns if "_mean" in col]
se_cols = [col for col in columns if "_se" in col]
worst_cols = [col for col in columns if "_worst" in col]

input_data = {}

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["Mean Features", "SE Features", "Worst Features"])

with tab1:
    st.header("Mean Features")
    for col in mean_cols:
        input_data[col] = st.slider(col, 0.0, 100.0, 10.0)

with tab2:
    st.header("Standard Error Features")
    for col in se_cols:
        input_data[col] = st.slider(col, 0.0, 50.0, 5.0)

with tab3:
    st.header("Worst Features")
    for col in worst_cols:
        input_data[col] = st.slider(col, 0.0, 150.0, 20.0)

# -------------------- DATAFRAME --------------------
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=columns, fill_value=0)

# -------------------- SHOW INPUT --------------------
st.subheader("📋 Input Summary")
st.dataframe(input_df)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict"):

    with st.spinner("Analyzing..."):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

    st.subheader("🧾 Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction", "Malignant" if prediction == 1 else "Benign")

    with col2:
        st.metric("Confidence", f"{prob*100:.2f}%")

    # -------------------- RESULT MESSAGE --------------------
    if prediction == 1:
        st.error("⚠️ Malignant (Cancer Detected)")
    else:
        st.success("✅ Benign (No Cancer)")

    # -------------------- PROBABILITY CHART --------------------
    prob_df = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability": [1 - prob, prob]
    })

    st.subheader("📊 Prediction Confidence")
    st.bar_chart(prob_df.set_index("Class"))

    # -------------------- RISK BAR --------------------
    st.subheader("🔬 Risk Level")
    st.progress(int(prob * 100))
    st.write(f"Risk Score: {prob*100:.1f}%")

    # -------------------- INTERPRETATION --------------------
    st.subheader("🧠 Interpretation")

    if prediction == 1:
        st.write("The model predicts a high likelihood of malignant tumor. Medical consultation is recommended.")
    else:
        st.write("The model predicts a benign tumor. Risk appears low.")

# -------------------- FILE UPLOAD --------------------
st.subheader("📁 Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV or Excel file")

if uploaded_file:
    try:
        # Detect file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📋 Uploaded Data")
        st.dataframe(df)

        # Align columns
        df = df.reindex(columns=columns, fill_value=0)

        # Prediction
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        # Create result dataframe
        result_df = df.copy()
        result_df["Prediction"] = preds
        result_df["Prediction Label"] = result_df["Prediction"].map({
            0: "Benign",
            1: "Malignant"
        })
        result_df["Probability"] = probs

        st.success("✅ Prediction completed")
        st.write("📊 Results")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")