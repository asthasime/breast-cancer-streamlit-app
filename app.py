import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
columns = data['columns']
scaler = data.get('scaler', None)

# ---------------- SHAP ----------------
explainer = shap.LinearExplainer(model, np.zeros((1, len(columns))))

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🧬 Breast Cancer Prediction App")
st.write("Predict whether a tumor is **Benign or Malignant**")

# ---------------- FRIENDLY NAMES ----------------
feature_explanations = {
    "radius_mean": "Average cell size",
    "radius_worst": "Largest cell size",
    "texture_mean": "Cell texture variation",
    "perimeter_worst": "Cell boundary size",
    "area_mean": "Cell area",
    "smoothness_mean": "Cell surface smoothness",
    "compactness_mean": "Cell density",
    "concave_points_worst": "Irregular cell shape",
    "concavity_mean": "Cell shape distortion",
    "symmetry_mean": "Cell symmetry",
    "fractal_dimension_mean": "Cell complexity"
}

# ---------------- MODE ----------------
mode = st.sidebar.radio("Select Mode", ["Upload CSV", "Manual Input"])

# =========================================================
# 🔵 CSV MODE
# =========================================================
if mode == "Upload CSV":

    st.header("📁 Upload CSV for Batch Prediction")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)

        df = df.reindex(columns=columns, fill_value=0)

        if scaler:
            df_scaled = pd.DataFrame(
                scaler.transform(df),
                columns=columns
            )
        else:
            df_scaled = df

        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        df['Prediction'] = ["Malignant" if p == 1 else "Benign" for p in preds]
        df['Probability'] = probs

        st.subheader("✅ Results")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "predictions.csv"
        )

        # -------- Distribution --------
        st.subheader("📊 Prediction Distribution")
        st.bar_chart(df['Prediction'].value_counts())

        st.subheader("📊 Probability Distribution")
        st.bar_chart(df['Probability'])

        # -------- USER FRIENDLY IMPORTANCE --------
        st.subheader("🧠 What factors are important?")

        try:
            shap_values = explainer.shap_values(df_scaled)
            mean_shap = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                "Feature": columns,
                "Impact Level": mean_shap
            }).sort_values(by="Impact Level", ascending=False).head(5)

            importance_df["Feature"] = importance_df["Feature"].apply(
                lambda x: feature_explanations.get(x, x)
            )

            # Reverse for better visualization
            importance_df = importance_df[::-1]

            fig, ax = plt.subplots()

            ax.barh(
                importance_df["Feature"],
                importance_df["Impact Level"]
            )

            ax.set_xlabel("Impact Level")
            ax.set_ylabel("Health Factor")
            ax.set_title("Top Factors Affecting Cancer Risk")

            plt.tight_layout()
            st.pyplot(fig)

            st.info("🔴 Higher impact means stronger influence on prediction")

            # Simple explanation
            st.subheader("📌 Simple Interpretation")
            for feature in importance_df["Feature"]:
                st.write(f"👉 {feature} plays an important role in cancer prediction")

        except:
            st.warning("Could not generate explanation.")

# =========================================================
# 🟢 MANUAL MODE
# =========================================================
else:

    st.header("✍️ Manual Input")

    input_data = {}
    for col in columns:
        input_data[col] = st.number_input(col, value=10.0)

    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df)

    if st.button("🔍 Predict"):

        if scaler:
            input_scaled = pd.DataFrame(
                scaler.transform(input_df),
                columns=columns
            )
        else:
            input_scaled = input_df

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        # -------- RESULT --------
        st.subheader("🧾 Result")

        if prediction == 1:
            st.error("⚠️ Malignant (High Risk Tumor)")
        else:
            st.success("✅ Benign (Low Risk Tumor)")

        st.metric("Confidence", f"{prob*100:.2f}%")

        # -------- RISK LEVEL --------
        if prob > 0.7:
            st.error("🔴 High Risk")
        elif prob > 0.4:
            st.warning("🟠 Moderate Risk")
        else:
            st.success("🟢 Low Risk")

        st.progress(int(prob * 100))

        # -------- SHAP --------
        st.subheader("🔍 Why this prediction?")

        try:
            shap_values = explainer.shap_values(input_scaled)
            shap_vals = shap_values[0]

            feature_importance = sorted(
                zip(columns, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            shap_df = pd.DataFrame(feature_importance, columns=["Feature", "Impact"])
            shap_df["Feature"] = shap_df["Feature"].apply(
                lambda x: feature_explanations.get(x, x)
            )

            shap_df = shap_df[::-1]

            fig, ax = plt.subplots()

            ax.barh(shap_df["Feature"], shap_df["Impact"])

            ax.set_xlabel("Impact on Prediction")
            ax.set_ylabel("Health Factor")
            ax.set_title("Why this result?")

            plt.tight_layout()
            st.pyplot(fig)

            # -------- SIMPLE EXPLANATION --------
            st.subheader("🧠 What does this mean?")

            for feature, value in feature_importance[:3]:
                explanation = feature_explanations.get(feature, feature)

                if value > 0:
                    st.write(f"👉 Higher {explanation} increases cancer risk")
                else:
                    st.write(f"👉 Lower {explanation} reduces cancer risk")

        except:
            st.warning("SHAP explanation not available.")