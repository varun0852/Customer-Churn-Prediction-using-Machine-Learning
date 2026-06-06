# Customer-Churn-Prediction-using-Machine-Learning
# Bank Customer Churn Prediction using Machine Learning

## 📌 Overview

Customer churn is a major challenge in the banking industry, as retaining existing customers is often more cost-effective than acquiring new ones. This project leverages Machine Learning techniques to predict whether a customer is likely to leave a bank based on demographic, financial, and account-related attributes.

The project focuses on data preprocessing, exploratory data analysis (EDA), statistical feature analysis, feature engineering, and predictive modeling using Logistic Regression.

---

## 🎯 Objective

The primary objective of this project is to develop a classification model capable of identifying customers who are likely to churn, enabling banks to take proactive retention measures.

Target Variable:

* **0** → Customer Retained
* **1** → Customer Churned

---

## 📊 Dataset Description

The dataset contains customer information such as:

| Feature          | Description                     |
| ---------------- | ------------------------------- |
| Credit Score     | Customer credit score           |
| Country          | Customer's country              |
| Gender           | Male/Female                     |
| Age              | Customer age                    |
| Tenure           | Years with the bank             |
| Balance          | Account balance                 |
| Products Number  | Number of banking products used |
| Credit Card      | Credit card ownership status    |
| Active Member    | Customer activity status        |
| Estimated Salary | Annual estimated salary         |
| Churn            | Target variable                 |

---

## ⚙️ Project Workflow

### 1. Data Understanding

* Loaded and inspected the dataset.
* Examined data types and missing values.
* Analyzed target variable distribution.

### 2. Exploratory Data Analysis (EDA)

Performed extensive visualization and analysis using:

* Count Plots
* Histograms
* Distribution Plots
* Box Plots
* Correlation Analysis

Key insights were extracted regarding customer demographics, account balance, age distribution, and churn behavior.

### 3. Data Preprocessing

Implemented the following preprocessing techniques:

* Removal of unnecessary identifiers (`customer_id`)
* Handling of categorical variables
* One-Hot Encoding
* Feature Scaling using MinMaxScaler
* Outlier Detection and Removal using the IQR Method

### 4. Statistical Feature Analysis

Applied statistical methods to evaluate feature significance:

* Chi-Square Test (`chi2_contingency`)
* ANOVA Test (`f_oneway`)

These techniques helped identify features with strong relationships to customer churn.

### 5. Feature Selection

Implemented Sequential Feature Selection (SFS) to identify the most relevant features for model training.

### 6. Model Development

Built a Machine Learning classification model using:

* Logistic Regression

The model was trained on preprocessed data and optimized using hyperparameter tuning techniques.

### 7. Model Evaluation

Model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* Precision
* Recall
* F1 Score

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy
* Mlxtend

### Development Environment

* Jupyter Notebook

---

## 🔍 Machine Learning Techniques Applied

* Exploratory Data Analysis (EDA)
* Outlier Detection & Removal
* Feature Scaling
* One-Hot Encoding
* Statistical Hypothesis Testing
* Sequential Feature Selection
* Logistic Regression
* Hyperparameter Optimization
* Model Evaluation

---

## 📁 Project Structure

```text
Bank-Customer-Churn-Prediction/
│
├── Customer_Churn.ipynb
├── README.md
├── dataset.csv
└── requirements.txt
```

---

## 📈 Results

The developed Logistic Regression model successfully identifies patterns associated with customer churn by analyzing customer demographics and financial behavior.

Key outcomes:

* Identified important factors influencing churn.
* Improved data quality through preprocessing and outlier removal.
* Applied statistical analysis to validate feature importance.
* Built an interpretable classification model for churn prediction.

---

## 🚀 Future Improvements

* Implement Random Forest and XGBoost models
* Compare multiple classification algorithms
* Deploy the model using Streamlit
* Add SHAP-based explainability
* Build a real-time prediction API using FastAPI
* Integrate model monitoring and performance tracking

---

## 💡 Business Impact

Accurate churn prediction enables banks to:

* Improve customer retention
* Reduce revenue loss
* Design targeted retention campaigns
* Enhance customer satisfaction
* Make data-driven business decisions

---

## 👨‍💻 Author

**Varun**

GitHub: https://github.com/varun0852

If you found this project useful, consider giving it a ⭐ on GitHub.
