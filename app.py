import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
columns = data['columns']
scaler = data.get('scaler', None)

# SHAP Explainer
explainer = shap.Explainer(model, feature_names=columns)

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🧬 Breast Cancer Prediction App")
st.write("Predict whether a tumor is **Benign or Malignant**")

# ---------------- MODE ----------------
mode = st.sidebar.radio("Select Mode", ["Upload CSV", "Manual Input"])

# =========================================================
# 🔵 MODE 1: CSV UPLOAD
# =========================================================
if mode == "Upload CSV":

    st.header("📁 Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.subheader("📊 Uploaded Data")
        st.dataframe(df)

        df = df.reindex(columns=columns, fill_value=0)

        if scaler:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df

        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        df['Prediction'] = ["Malignant" if p == 1 else "Benign" for p in preds]
        df['Probability'] = probs

        st.subheader("✅ Prediction Results")
        st.dataframe(df)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

        # Visualization
        st.subheader("📊 Prediction Distribution")
        st.bar_chart(df['Prediction'].value_counts())

        st.subheader("📊 Probability Distribution")
        st.bar_chart(df['Probability'])

        st.subheader("📈 Feature Summary")
        st.write(df.describe())

# =========================================================
# 🟢 MODE 2: MANUAL INPUT
# =========================================================
else:

    st.header("✍️ Manual Input")

    mean_cols = [col for col in columns if "_mean" in col]
    se_cols = [col for col in columns if "_se" in col]
    worst_cols = [col for col in columns if "_worst" in col]

    input_data = {}

    tab1, tab2, tab3 = st.tabs(["Mean Features", "SE Features", "Worst Features"])

    with tab1:
        st.subheader("Mean Features")
        for col in mean_cols:
            input_data[col] = st.number_input(col, value=10.0)

    with tab2:
        st.subheader("Standard Error Features")
        for col in se_cols:
            input_data[col] = st.number_input(col, value=1.0)

    with tab3:
        st.subheader("Worst Features")
        for col in worst_cols:
            input_data[col] = st.number_input(col, value=20.0)

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    st.subheader("📋 Input Data")
    st.dataframe(input_df)

    if st.button("🔍 Predict"):

        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("🧾 Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", "Malignant" if prediction == 1 else "Benign")

        with col2:
            st.metric("Confidence", f"{prob*100:.2f}%")

        if prediction == 1:
            st.error("⚠️ Malignant (Cancer Detected)")
        else:
            st.success("✅ Benign (No Cancer)")

        # Probability visualization
        prob_df = pd.DataFrame({
            "Class": ["Benign", "Malignant"],
            "Probability": [1 - prob, prob]
        })

        st.subheader("📊 Prediction Probability")
        st.bar_chart(prob_df.set_index("Class"))

        # ================= SHAP =================
        st.subheader("🔍 Model Explainability (SHAP)")

        try:
            shap_values = explainer(input_scaled)

            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

            # ================= INTERPRETATION =================
            st.subheader("🧠 Interpretation")

            shap_vals = shap_values.values[0]

            feature_importance = sorted(
                zip(columns, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]

            for feature, value in feature_importance:
                if value > 0:
                    st.write(f"🔺 {feature} increases risk")
                else:
                    st.write(f"🔻 {feature} decreases risk")

        except Exception as e:
            st.warning("SHAP explanation not available for this model.")