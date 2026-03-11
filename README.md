# 🏦 Customer Churn Prediction in Retail Banking

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Random Forest](https://img.shields.io/badge/Random%20Forest-ML-green) ![SMOTE](https://img.shields.io/badge/SMOTE--ENN-Resampling-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen) ![Grade](https://img.shields.io/badge/Academic%20Grade-Top%20of%20Class-gold)

> **Predicting which retail banking customers are likely to churn using Machine Learning — comparing Random Forest vs Logistic Regression on 10,000 customer records.**

---

## 📌 Project Overview

This project builds and evaluates machine learning models to predict customer churn in retail banking. Using a dataset of 10,000 anonymized bank customers, the study compares a linear model (Logistic Regression) against a non-linear tree-based model (Random Forest), after handling severe class imbalance using SMOTE-ENN resampling.

**Result: Random Forest significantly outperformed Logistic Regression, achieving a cross-validated ROC-AUC of 0.977.**

---

## 🏆 Academic Achievement
> ⭐ **Top grade in class** — FOM University of Applied Sciences, Essen (Jan 2026)
> Module: Area of Application — Business Analytics | M.Sc. Big Data & Business Analytics
> Group Project: Nandini Babu Rajendra Prasad, Vaishnavi Sai Satya Priya Ankem, Manasa Uppunuru

---

## 📊 Key Results

| Model | Test Accuracy | CV ROC-AUC | Churners Detected |
|-------|--------------|------------|-------------------|
| Logistic Regression | 19% | 0.575 | 0% ❌ |
| **Random Forest** | **59%** | **0.977** | **37%** ✅ |

✅ Random Forest outperformed Logistic Regression across **every single metric**

---

## 🗂️ Project Structure

```
customer-churn-prediction-banking/
│
├── 📓 R_Analysis_Code_.ipynb     # Full ML pipeline — EDA, models, evaluation
├── 📄 Churn_Modelling.csv        # Dataset (10,000 banking customers)
└── 📄 README.md
```

---

## 🔧 Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest, Logistic Regression) |
| Class Imbalance | imbalanced-learn (SMOTE-ENN) |
| Statistical Tests | SciPy (t-tests, ANOVA) |
| Visualisation | Matplotlib, Seaborn |
| Validation | VIF Analysis, Stratified K-Fold Cross Validation |
| Environment | Jupyter Notebook, Conda |

---

## 📥 Dataset

- **Source:** Kaggle — Retail Banking Customer Churn Dataset
- **Records:** 10,000 anonymized customer records
- **Target:** `Exited` (binary: 0=retained, 1=churned)
- **Churn Rate:** 19.57% (severely imbalanced)
- **Features:** Demographics (age, gender, geography), Financial (credit score, balance, salary), Behavioural (tenure, products, active membership)

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Identified **severe class imbalance** (19.57% churn rate)
- Histograms showed significant overlap between churned/retained distributions
- Pearson correlation matrix revealed weak linear relationships (highest |r| = 0.03)
- Statistical tests (t-tests, ANOVA) confirmed no significant group separators
- **Conclusion:** Complex non-linear patterns require tree-based models

### 2. Data Preprocessing
Full CRISP-DM compliant pipeline:
- Removed non-predictive identifiers (RowNumber, CustomerId, Surname)
- One-Hot Encoding for categorical variables (Geography, Gender)
- StandardScaler for numerical features
- VIF analysis confirmed no multicollinearity (all scores < 2)

### 3. Class Imbalance Handling
- Applied **SMOTE-ENN** exclusively to training data
- Training set: 8,000 → 8,301 samples (churn ratio: 19.55% → 67.61%)
- Test set kept at original distribution to prevent data leakage

### 4. Models Compared
- **Logistic Regression** — linear baseline
- **Random Forest** — non-linear tree-based model (n_estimators=100)

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 5-Fold stratified cross-validation on resampled training data
- Confusion matrices and ROC curve comparison

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy jupyter
```

### Steps
1. Clone the repository:
```bash
git clone https://github.com/Manasa-Uppunuru/customer-churn-prediction-banking.git
cd customer-churn-prediction-banking
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open and run `R_Analysis_Code_.ipynb`

---

## 💡 Key Findings

1. **Financial stress drives churn** — CreditScore (13.7%) and Balance (13.2%) are the top predictors via Random Forest feature importance
2. **Non-linear models are essential** — Logistic Regression detected 0 churners vs Random Forest detecting 37%
3. **SMOTE-ENN is effective** — resolved 80/20 imbalance enabling proper minority class learning
4. **Churn emerges from interactions** — no single feature separates churners; combinations matter
5. **Practical implication** — banks should target high-balance/low-credit-score customers for retention campaigns

---

## 🔮 Future Work
- Add behavioural features (transaction frequency, login patterns, support tickets)
- Compare with XGBoost, LightGBM, and neural network models
- Temporal analysis of churn patterns (monthly cohorts)
- A/B testing to validate retention intervention ROI
- Deploy as real-time churn scoring API

---

## 👩‍💻 Author

**Manasa Uppunuru**
M.Sc. Big Data & Business Analytics — FOM University, Essen
📧 uppunurumanasareddy@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/manasa-reddy-uppunuru-282447224/)
🔗 [GitHub](https://github.com/Manasa-Uppunuru)

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
