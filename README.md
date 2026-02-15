# Machine Learning Classification Assignment

---

## a) Problem Statement

The objective of this project is to build and compare multiple Machine Learning classification models on a structured dataset and evaluate their performance using standard evaluation metrics.

The task is to classify whether a tumor is **Malignant (0)** or **Benign (1)** using various classification algorithms and compare their predictive performance.

---

## b) Dataset Description

Dataset Used: **Breast Cancer Wisconsin Dataset**  
Source: `sklearn.datasets`

### Dataset Details:

- Number of Instances: 569
- Number of Features: 30
- Target Variable: Binary Classification
  - 0 → Malignant
  - 1 → Benign
- Feature Type: Numerical

The dataset contains features computed from digitized images of breast mass, including:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension
- and other derived statistical measures

No missing values were found in the dataset.

---

## c) Models Used

The following Machine Learning classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble Boosting)

---

## Model Evaluation Metrics

The following evaluation metrics were calculated for each model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|------|
| Logistic Regression | 0.982456 | 0.995370 | 0.986111 | 0.986111 | 0.986111 | 0.962 |
| KNN | 0.956140 | 0.978836 | 0.958904 | 0.972222 | 0.965517 | 0.905 |
| XGBoost (Ensemble) | 0.956140 | 0.990079 | 0.946667 | 0.986111 | 0.965986 | 0.905 |
| Random Forest (Ensemble) | 0.956140 | 0.990079 | 0.958904 | 0.972222 | 0.965517 | 0.905 |
| Naive Bayes | 0.929825 | 0.986772 | 0.944444 | 0.944444 | 0.944444 | 0.849 |
| Decision Tree | 0.903509 | 0.902116 | 0.955224 | 0.888889 | 0.920863 | 0.801 |


---

## Observations About Model Performance

| ML Model Name | Observation About Model Performance |
|---------------|-------------------------------------|
| Logistic Regression |  |
| Decision Tree |  |
| KNN |  |
| Naive Bayes |  |
| Random Forest (Ensemble) |  |
| XGBoost (Ensemble) |  |

---

## Conclusion

The performance comparison of all six classification models demonstrates the strengths and weaknesses of different algorithms.

Ensemble models such as Random Forest and XGBoost generally provide better generalization performance due to reduced variance and improved bias-variance tradeoff.

The final model selection can be made based on:

- Highest Accuracy
- Highest AUC
- Highest MCC (best balanced metric)
- Business requirement (precision vs recall trade-off)

---

## Deployment

The models have been deployed using **Streamlit** with the following features:

- Dataset download option
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix visualization
- Classification report
- Model comparison dashboard

---

## Author

BITS Machine Learning Assignment  
