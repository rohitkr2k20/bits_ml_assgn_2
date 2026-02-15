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

1. **Logistic Regression** achieved the highest overall performance across almost all metrics, including Accuracy, AUC, F1 Score, and MCC. This suggests that the dataset is highly linearly separable.

2. **KNN, Random Forest, and XGBoost** showed similar performance levels with strong AUC and F1 scores, demonstrating that ensemble and distance-based methods also handle this dataset effectively.

3. **XGBoost** achieved a very high AUC score (0.990079), indicating strong class discrimination capability.

4. **Random Forest** performed consistently across all metrics, confirming the stability advantage of ensemble methods.

5. **Naive Bayes** performed reasonably well despite its strong independence assumption, but slightly underperformed compared to ensemble and linear models.

6. **Decision Tree** showed the lowest performance among all models, likely due to overfitting and high variance when used as a single tree.

7. The **MCC scores** confirm the same ranking pattern as Accuracy and F1 Score, indicating balanced predictive performance across classes.

---

## Conclusion

Based on the evaluation metrics:

- Logistic Regression is the best-performing model for this dataset.
- Ensemble methods (Random Forest and XGBoost) also provide strong and stable performance.
- The dataset appears to be highly linearly separable, which explains the excellent performance of Logistic Regression.

Therefore, Logistic Regression can be selected as the final recommended model for this classification task.


## Author

Rohit Kumar 
2025aa05867
Sem -1 ( ML )
M.Tech ( AI & ML )- BITS Pilani 
