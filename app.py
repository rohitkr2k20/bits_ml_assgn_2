import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Premium ML Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================
# CUSTOM CSS (Premium UI Styling)
# =========================================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.metric-card {
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}
.metric-title {
    font-size: 18px;
    color: #9FA2B4;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #4CAF50;
}
.section-title {
    font-size: 26px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Premium Machine Learning Classification Dashboard")
st.write("### Breast Cancer Classification Models Comparison")

# =========================================
# LOAD MODELS
# =========================================
MODEL_PATH = "model/"

models_dict = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])
selected_model = st.sidebar.selectbox("Select Model", list(models_dict.keys()))
run_button = st.sidebar.button("Run Evaluation")

# =========================================
# MAIN LOGIC
# =========================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("CSV must contain a 'target' column.")
    else:

        X = df.drop("target", axis=1)
        y_true = df["target"]

        X_scaled = scaler.transform(X)

        if run_button:

            model_file = models_dict[selected_model]
            with open(os.path.join(MODEL_PATH, model_file), "rb") as f:
                model = pickle.load(f)

            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]

            # Metrics
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            # =========================================
            # METRICS DISPLAY (Premium KPI Cards)
            # =========================================
            st.markdown("<div class='section-title'>📈 Evaluation Metrics</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            metrics = [
                ("Accuracy", acc),
                ("AUC Score", auc),
                ("Precision", precision),
                ("Recall", recall),
                ("F1 Score", f1),
                ("MCC", mcc),
            ]

            for col, (name, value) in zip([col1, col2, col3, col4, col5, col6], metrics):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{name}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

            # =========================================
            # CONFUSION MATRIX
            # =========================================
            st.markdown("<div class='section-title'>🧮 Confusion Matrix</div>", unsafe_allow_html=True)

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # =========================================
            # CLASSIFICATION REPORT
            # =========================================
            st.markdown("<div class='section-title'>📄 Classification Report</div>", unsafe_allow_html=True)

            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap="Blues"))

            # =========================================
            # MODEL COMPARISON DASHBOARD
            # =========================================
            st.markdown("<div class='section-title'>📊 Model Comparison Dashboard</div>", unsafe_allow_html=True)

            comparison_results = []

            for name, file in models_dict.items():
                with open(os.path.join(MODEL_PATH, file), "rb") as f:
                    m = pickle.load(f)

                y_p = m.predict(X_scaled)
                y_pr = m.predict_proba(X_scaled)[:, 1]

                comparison_results.append([
                    name,
                    accuracy_score(y_true, y_p),
                    roc_auc_score(y_true, y_pr),
                    f1_score(y_true, y_p),
                    matthews_corrcoef(y_true, y_p)
                ])

            comp_df = pd.DataFrame(
                comparison_results,
                columns=["Model", "Accuracy", "AUC", "F1", "MCC"]
            )

            st.dataframe(
                comp_df.sort_values("Accuracy", ascending=False)
                .style.background_gradient(cmap="viridis")
            )

else:
    st.info("👈 Please upload a test CSV file from the sidebar to begin.")
