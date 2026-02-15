import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Premium ML Dashboard", layout="wide")
st.title("🚀 Premium ML Classification Dashboard")

# =====================================
# LOAD FIXED DATASET
# =====================================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙ Controls")

# Download dataset
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name='breast_cancer_dataset.csv',
    mime='text/csv'
)

# Model selection
model_name = st.sidebar.selectbox("Select Model", [
    "Overall Comparison",
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])


st.sidebar.markdown("### 🔧 Hyperparameter Tuning")

# Dynamic hyperparameters
if model_name == "Logistic Regression":
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)

elif model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_name == "KNN":
    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)

elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_name == "XGBoost":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)

run = st.sidebar.button("🚀 Run Model")

# =====================================
# MAIN DISPLAY
# =====================================
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

# =====================================
# MODEL EXECUTION
# =====================================
if run:

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================================
    # OVERALL COMPARISON MODE
    # =========================================
    if model_name == "Overall Comparison":

        st.header("📊 Overall Model Comparison Dashboard")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            results.append([
                name,
                accuracy_score(y_test, y_pred),
                roc_auc_score(y_test, y_prob),
                f1_score(y_test, y_pred),
                matthews_corrcoef(y_test, y_pred)
            ])

        comp_df = pd.DataFrame(
            results,
            columns=["Model", "Accuracy", "AUC", "F1 Score", "MCC"]
        )

        comp_df = comp_df.sort_values("Accuracy", ascending=False)

        st.subheader("📈 Performance Table")
        st.dataframe(
            comp_df.style.background_gradient(cmap="viridis")
        )

        # Highlight Best Model
        best_model = comp_df.iloc[0]["Model"]
        st.success(f"🏆 Best Performing Model: {best_model}")

        # Bar Chart Comparison
        st.subheader("📊 Accuracy Comparison")

        fig, ax = plt.subplots()
        sns.barplot(data=comp_df, x="Accuracy", y="Model", ax=ax)
        st.pyplot(fig)

    # =========================================
    # SINGLE MODEL MODE
    # =========================================
    else:

        st.header(f"📊 {model_name} Detailed Evaluation")

        # Model initialization
        if model_name == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=5000)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth)

        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        elif model_name == "Naive Bayes":
            model = GaussianNB()

        elif model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth
            )

        elif model_name == "XGBoost":
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                eval_metric='logloss'
            )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve
        st.subheader("📈 ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr)
        ax2.plot([0,1], [0,1], linestyle='--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        st.pyplot(fig2)

        # Classification Report
        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    # =====================================
    # FEATURE IMPORTANCE (if applicable)
    # =====================================
    if model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
        st.subheader("🌟 Feature Importance")

        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(10)

        fig3, ax3 = plt.subplots()
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax3)
        st.pyplot(fig3)
