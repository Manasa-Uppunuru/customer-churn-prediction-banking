# 🏦 Customer Churn Prediction in Retail Banking

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Random Forest](https://img.shields.io/badge/Random%20Forest-ML-green) ![SMOTE](https://img.shields.io/badge/SMOTE--ENN-Resampling-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen) ![Grade](https://img.shields.io/badge/Academic%20Grade-Top%20of%20Class-gold)

> **Predicting which retail banking customers are likely to churn using Machine Learning — comparing Random Forest vs Logistic Regression on 10,000 customer records.**

---

## 📌 Project Overview

This project builds and evaluates machine learning models to predict customer churn in retail banking. Using a dataset of 10,000 anonymized bank customers, the study compares a linear model (Logistic Regression) against a non-linear tree-based model (Random Forest), after handling severe class imbalance using SMOTE-ENN resampling.

**Reported comparison:** Random Forest has a higher cross-validation ROC-AUC than Logistic Regression in the existing analysis. The reported test results are substantially weaker; see the evaluation limitations below before interpreting this as generalization performance.

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

These are the existing project-reported figures, not newly reproduced benchmarks. Cross-validation and test metrics describe different evaluations and should not be compared as equivalent evidence.

### Evaluation limitations

- With a reported churn rate of 19.57%, always predicting retention would achieve roughly 80.43% accuracy on that overall distribution. The reported 59% Random Forest test accuracy is below that reference, so this result does not establish a production-ready churn model. The exact test-set baseline should be computed from its labels.
- The documented cross-validation is performed on already resampled training data. If resampling occurs before fold splitting, information can cross validation boundaries and inflate scores. A stronger follow-up is an imbalanced-learn pipeline that fits preprocessing and resampling inside each training fold, with a separate untouched test set.
- Verify class labels, target alignment, decision thresholds, and the interpretation of "Churners Detected" against the notebook outputs. Report held-out ROC-AUC, precision-recall AUC, and class-specific recall alongside baselines.
- This is an academic group project. Results should be re-evaluated before using the model for customer decisions.

---

## Quick review

Open the [analysis notebook](R_Analysis_Code%20.ipynb) to inspect the workflow and stored outputs. The filename contains a space before `.ipynb`; the link above matches the repository file exactly. See [Churn_Modelling.csv](Churn_Modelling.csv) for the input dataset. Group-project contributors are credited above; individual task ownership is not specified here.

## 🗂️ Project Structure

```
customer-churn-prediction-banking/
│
├── 📓 R_Analysis_Code .ipynb     # Full ML pipeline — EDA, models, evaluation
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

3. Open and run `R_Analysis_Code .ipynb`

---

## 💡 Key Findings

1. **Reported feature importance** — CreditScore (13.7%) and Balance (13.2%) rank highly in the existing Random Forest analysis. Feature importance does not establish that financial stress causes churn.
2. **Model comparison requires further validation** — the reported churn-detection figures differ across models, but the weak test results and validation setup require investigation.
3. **Resampling changes the learning distribution** — SMOTE-ENN was explored to address imbalance; effectiveness must be evaluated on untouched data.
4. **Further analysis** — investigate feature relationships and model errors before drawing conclusions about churn drivers.
5. **Potential use case** — retention analysis is a possible application, but targeting recommendations need validated predictions, business-cost analysis, and appropriate review.

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
