# 💳 Credit Default Prediction

## 🧠 Project Overview

This project builds a machine learning model to predict whether a credit card client will default on their payment next month. It uses a real-world dataset from a bank and covers an end-to-end ML pipeline.

The goal is to help financial institutions **reduce credit risk**, **optimize lending decisions**, and **detect high-risk clients** before they default.

---

## 📦 Dataset

- **Name**: Default of Credit Card Clients Dataset  
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- **Size**: 30,000 clients, 25 columns  
- **Target variable**: `default.payment.next.month`  
- **Features**: Demographics (age, gender, education), past payment behavior, credit and billing history

---

## 🧰 Tools & Libraries

- Python
- pandas, numpy
- scikit-learn, XGBoost
- imbalanced-learn (SMOTE)
- seaborn, matplotlib
- Jupyter Notebook

---

## 🚀 Workflow

1. **Data Loading & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Class Imbalance Handling** (SMOTE, class weights)
5. **Model Training**: Logistic Regression (baseline), XGBoost (advanced)
6. **Evaluation**: Confusion matrix, F1-score, ROC AUC
7. **Insights & Recommendations**

---

## 📊 Key Results

| Model                    | F1-score (class 1) | AUC   |
|--------------------------|--------------------|-------|
| Logistic Regression      | 0.26               | 0.62  |
| Logistic + SMOTE         | 0.41               | 0.72  |
| XGBoost (weighted)       | **0.47**           | **0.77** |

✅ XGBoost performed best in detecting defaults, especially with class balancing using `scale_pos_weight`.

---

## 📈 Visuals

- Distribution of target variable (imbalanced)
- Age vs default status
- Credit limit vs default rate
- ROC curves for each model

---

## 📌 Business Takeaways

- Clients with delayed past payments are far more likely to default again.
- Credit limit and age are strong predictors of risk.
- Proper handling of class imbalance is **critical** for reliable predictions.

---

## 💡 Possible Improvements

- Hyperparameter tuning (GridSearchCV, Optuna)
- Model explainability (e.g. SHAP values)
- Deployment (e.g. Streamlit app or Flask API)





