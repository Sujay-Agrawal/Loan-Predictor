# 🏦 Loan Default Prediction with Naive Bayes

A machine learning project that predicts loan outcomes using a **Gaussian Naive Bayes** classifier. The notebook walks through the full workflow: exploratory data analysis, statistical significance testing, preprocessing, outlier removal, model training, evaluation, and model export.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Findings](#-key-findings)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Using the Saved Model](#-using-the-saved-model)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

Lenders need to estimate the risk attached to each loan application. This project builds a baseline classifier on applicant and loan attributes to predict `loan_status`, and shows how data cleaning (specifically outlier removal) improves model performance.

**Highlights**

- Full EDA with visualizations against the target variable
- Chi-square and Welch's t-tests to identify statistically significant features
- One-hot and label encoding for categorical variables
- IQR-based outlier removal
- Before/after model comparison
- Trained model exported as a `.pkl` file

---

## 📂 Dataset

**File:** `loan_data.csv`: 45,000 records, 14 columns, no duplicate rows.

| Feature | Type | Description |
|---|---|---|
| `person_age` | Numeric | Applicant age |
| `person_gender` | Categorical | Applicant gender |
| `person_education` | Categorical | Highest education level (5 categories) |
| `person_income` | Numeric | Annual income |
| `person_emp_exp` | Numeric | Years of employment experience |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `loan_amnt` | Numeric | Loan amount requested |
| `loan_intent` | Categorical | Purpose of the loan (6 categories) |
| `loan_int_rate` | Numeric | Loan interest rate (%) |
| `loan_percent_income` | Numeric | Loan amount as a fraction of income |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |
| `credit_score` | Numeric | Applicant credit score |
| `previous_loan_defaults_on_file` | Categorical | Whether the applicant has previously defaulted (Yes/No) |
| `loan_status` | **Target** | Loan outcome (0 / 1) |

---

## ⚙️ Project Workflow

1. **Data loading & inspection**: shape, dtypes, missing values, duplicates, summary statistics.
2. **Exploratory data analysis**: target distribution, boxplots for numeric features, and count plots for categorical features, all split by `loan_status`.
3. **Statistical testing**
   - Chi-square test for categorical features
   - Welch's t-test for numerical features
4. **Preprocessing**
   - One-hot encoding (`drop_first=True`) for `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`
   - Label encoding for `previous_loan_defaults_on_file`
   - Boolean-to-integer conversion and dropping of null rows
5. **Baseline model**: `GaussianNB` on an 80/20 train/test split (`random_state=42`).
6. **Outlier removal**: IQR method (1.5 × IQR) applied to age, income, employment experience, loan amount, interest rate, credit history length, and credit score. Dataset reduced from 45,000 to **37,267** rows.
7. **Retraining & evaluation**: same model and split settings on the cleaned data.
8. **Model export**: final model saved to `model_naive.pkl` with `pickle`.

---

## 💡 Key Findings

**Statistically significant predictors** (p < 0.05):
`person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`, `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`

**Not significant:**
`person_gender` (p ≈ 0.91) and `person_education` (p ≈ 0.73)

**Data quality:** the raw data contained clearly unrealistic values, such as ages of 123 and 144 and employment experience above 100 years, which hurt model performance until removed.

---

## 📊 Results

| Metric | Baseline (raw data) | After outlier removal |
|---|---|---|
| **Accuracy** | 80.26% | **84.14%** |
| Precision (class 1) | 0.64 | **0.77** |
| Recall (class 1) | 0.26 | **0.42** |
| F1-score (class 1) | 0.37 | **0.54** |
| Test set size | 9,000 | 7,454 |

**Confusion matrix (after outlier removal)**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 5,580 | 210 |
| **Actual 1** | 972 | 692 |

> **Note:** The model is strong on the majority class (recall 0.96) but still misses a majority of class 1 cases (recall 0.42). The classes are imbalanced (~78% / 22%), so accuracy alone overstates performance.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

### Run the notebook

```bash
jupyter notebook Naive_bias.ipynb
```

Make sure `loan_data.csv` is in the same directory as the notebook.

---

## 🧪 Using the Saved Model

```python
import pickle
import pandas as pd

with open("model_naive.pkl", "rb") as f:
    model = pickle.load(f)

# X_new must have the same 22 feature columns, in the same order,
# and the same encoding as the training data
predictions = model.predict(X_new)
```

---

## 🗂 Project Structure

```
├── Naive_bias.ipynb     # Main analysis and modeling notebook
├── loan_data.csv        # Dataset
├── model_naive.pkl      # Trained Gaussian Naive Bayes model
└── README.md
```

---

## 🔮 Future Improvements

- Handle class imbalance (SMOTE, class weights, threshold tuning)
- Compare against Logistic Regression, Random Forest, and XGBoost
- Use cross-validation and hyperparameter tuning
- Feature scaling and feature selection based on the significance tests
- Evaluate with ROC-AUC and precision-recall curves
- Wrap the model in a simple Streamlit or Flask app

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. Update this section if you choose a different license.
