# Customer Retention Intelligence Platform

An AI-powered customer churn prediction platform that helps businesses identify customers who are likely to discontinue their services. The system analyzes customer demographics, subscription details, and usage patterns to estimate churn risk, enabling organizations to take proactive retention measures.

---

## Features

- Customer churn prediction
- Churn probability estimation
- Interactive Streamlit dashboard
- Automated data preprocessing
- Class imbalance handling using SMOTE
- Multiple machine learning models compared
- Performance evaluation using cross-validation
- Customer risk analysis

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

## Project Structure

```
Customer_Retention_Intelligence_Platform/
│
├── app.py
├── churn_pred.ipynb
├── gradientboosting.pkl
├── requirements.txt
├── README.md
├── assets/
└── dataset/
```

---

## Machine Learning Pipeline

1. Data loading and exploration
2. Missing value handling
3. Feature encoding
4. Feature transformation
5. Feature scaling
6. Handling class imbalance using SMOTE
7. Model training
8. Cross-validation
9. Performance evaluation
10. Deployment with Streamlit

---

## Data Preprocessing

The preprocessing pipeline includes:

- Missing value imputation
- Ordinal encoding
- One-hot encoding
- Log transformation for skewed numerical features
- Standardization for distance-based and linear models
- Separate preprocessing pipelines for tree-based and non-tree-based algorithms

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes
- Gradient Boosting
- XGBoost

---

## Model Selection

Several machine learning models were trained and evaluated using stratified train-test splitting and cross-validation.

The final deployed model was selected based on its balance between predictive performance, precision, recall, F1-score, and overall generalization ability.

Gradient Boosting provided the most balanced performance for customer churn prediction and was selected for deployment.

---

## Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Cross-Validation Score

---

## Application Workflow

1. Enter customer information.
2. Input data is preprocessed automatically.
3. The trained machine learning model predicts churn risk.
4. The application displays:
   - Churn prediction
   - Churn probability
   - Customer retention probability


## Future Enhancements

- SHAP-based model explainability
- Customer segmentation
- Personalized retention recommendations
- Cloud deployment
- Model monitoring
- Automated retraining pipeline

---

## License

This project is intended for educational and research purposes.
