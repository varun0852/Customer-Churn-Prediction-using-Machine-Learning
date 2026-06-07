# 🏦 Customer Churn Prediction using Machine Learning

An end-to-end machine learning project that predicts customer churn in a banking dataset using EDA, statistical feature selection, multi-model comparison, SMOTE oversampling, and GridSearchCV hyperparameter tuning.

---

## 📊 Model Results

### Before SMOTE
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 83.71%   | 64.68%    | 42.23% | 51.10%   |
| XGBoost             | 82.56%   | 59.42%    | 42.49% | 49.55%   |
| Decision Tree       | 77.34%   | 44.12%    | 46.63% | 45.34%   |
| Logistic Regression | 82.40%   | 65.03%    | 27.46% | 38.62%   |

### After SMOTE
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 79.22%   | 48.66%    | 56.48% | 52.28%   |
| XGBoost             | 80.10%   | 50.64%    | 51.30% | 50.94%   |
| Logistic Regression | 70.50%   | 37.16%    | 67.10% | 47.83%   |
| Decision Tree       | 74.20%   | 37.89%    | 43.78% | 40.62%   |

### After GridSearchCV — Tuned Random Forest (Best Model)
| Class    | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Retained | 0.89      | 0.85   | 0.87     |
| Churned  | 0.49      | 0.57   | 0.53     |

🏆 **Best Cross-Validation F1: 86.61%**
⚙️ **Best Parameters:** `max_depth: 20, min_samples_leaf: 1, min_samples_split: 2, n_estimators: 200`

---

## 🎯 Objective

Predict whether a bank customer will churn (leave the bank) based on demographic and account features — enabling proactive retention strategies that save revenue.

---

## 📁 Dataset

- **Size:** ~7,000 records
- **Target:** Churn (1 = churned, 0 = retained)
- **Features:** Age, Balance, Credit Score, Geography, Gender, Tenure, Number of Products, Active Member, Estimated Salary

---

## ⚙️ Project Workflow

```
Data Loading & Exploration
        ↓
Exploratory Data Analysis (EDA)
(distributions, correlations, class imbalance check)
        ↓
Statistical Feature Testing
(Chi-Square for categorical, ANOVA for numerical)
        ↓
Data Preprocessing
(Label encoding, MinMaxScaler, IQR outlier removal)
        ↓
Feature Selection
(Sequential Feature Selector — top 7 features)
        ↓
Model Training & Comparison
(Logistic Regression, Decision Tree, Random Forest, XGBoost)
        ↓
SMOTE Oversampling
(Fix class imbalance — boosted Recall from 42% → 56%)
        ↓
GridSearchCV Hyperparameter Tuning
(Tuned Random Forest — Best CV F1: 86.61%)
```

---

## 📈 Visualizations

### Model Comparison Chart
![Model Comparison](model_comparision.png)

### Confusion Matrix — Tuned Random Forest
![Confusion Matrix](confusion_matrix.png)

### Top 10 Feature Importances
![Feature Importance](feature_importance.png)

---

## 🔑 Key Findings

- **SMOTE significantly improved Recall** — Random Forest Recall jumped from 42.23% → 56.48%
- **Logistic Regression** saw the biggest Recall boost: 27.46% → 67.10% after SMOTE
- **Tuned Random Forest** achieved best cross-validation F1 of **86.61%** via GridSearchCV
- **Top churn predictors:** Age, Balance, Active Member status, Country_Germany
- Without SMOTE, all models were heavily biased toward predicting "Retained"

---

## 🛠️ Tech Stack

| Layer             | Technology                         |
|-------------------|------------------------------------|
| Language          | Python 3.10+                       |
| Data Processing   | Pandas, NumPy                      |
| Visualization     | Matplotlib, Seaborn                |
| ML Models         | Scikit-learn, XGBoost              |
| Oversampling      | imbalanced-learn (SMOTE)           |
| Hyperparameter    | GridSearchCV (Scikit-learn)        |
| Feature Selection | Sequential Feature Selector        |
| Statistical Tests | Chi-Square, ANOVA (SciPy)         |
| Environment       | Google Colab / Jupyter Notebook    |

---

## 📁 Project Structure

```
Customer-Churn-Prediction-using-Machine-Learning/
├── customer_churn_prediction.ipynb   # Full notebook: EDA + modelling + SMOTE + GridSearchCV
├── model_comparision.png             # Bar chart comparing all 4 models
├── confusion_matrix.png              # Confusion matrix for Tuned Random Forest
├── feature_importance.png            # Top 10 feature importances
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/varun0852/Customer-Churn-Prediction-using-Machine-Learning.git
cd Customer-Churn-Prediction-using-Machine-Learning
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy imbalanced-learn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook customer_churn_prediction.ipynb
```

---

## 🔮 Next Steps

- Deploy Tuned Random Forest via Streamlit dashboard
- Try ensemble stacking of RF + XGBoost
- Experiment with threshold tuning to further improve Churned class F1

---

## 👤 Author

**Varun Diwakar** — AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/varun-diwakar-a87781274)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/varun0852)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:diwakarvarun752@gmail.com)
