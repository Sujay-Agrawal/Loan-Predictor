import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model_naive.pkl"

# Exact column order used when the model was trained
# (X = df.drop('loan_status', axis=1) after get_dummies(drop_first=True))
FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "person_gender_male",
    "person_education_Bachelor",
    "person_education_Doctorate",
    "person_education_High School",
    "person_education_Master",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
]

EDUCATION = ["Associate", "Bachelor", "Doctorate", "High School", "Master"]  # Associate = baseline
HOME = ["MORTGAGE", "OTHER", "OWN", "RENT"]  # MORTGAGE = baseline
INTENT = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]  # DEBTCONSOLIDATION = baseline


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_features(inputs: dict, columns=None) -> pd.DataFrame:
    """Turn raw form inputs into the encoded feature row the model expects."""
    row = {col: 0 for col in (columns or FEATURES)}

    for key in [
        "person_age", "person_income", "person_emp_exp", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score",
    ]:
        row[key] = inputs[key]

    row["previous_loan_defaults_on_file"] = 1 if inputs["previous_loan_defaults_on_file"] == "Yes" else 0
    row["person_gender_male"] = 1 if inputs["person_gender"] == "male" else 0

    # One-hot columns (baseline category has no column, so all zeros)
    for prefix, value in [
        ("person_education", inputs["person_education"]),
        ("person_home_ownership", inputs["person_home_ownership"]),
        ("loan_intent", inputs["loan_intent"]),
    ]:
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[list(row.keys())]


def main():
    st.set_page_config(page_title="Loan Risk Predictor", page_icon="🏦", layout="wide")

    st.title("🏦 Loan Risk Predictor")
    st.caption("Gaussian Naive Bayes model · enter the applicant and loan details below")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH.name}`. Place it in the same folder as `app.py`.")
        st.stop()

    model = load_model()
    columns = list(getattr(model, "feature_names_in_", FEATURES))

    with st.form("loan_form"):
        st.subheader("👤 Applicant")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 20, 40, 27)
            gender = st.selectbox("Gender", ["female", "male"])
        with c2:
            education = st.selectbox("Education", EDUCATION, index=1)
            emp_exp = st.number_input("Years of employment experience", 0, 16, 3)
        with c3:
            income = st.number_input("Annual income", 8000, 170000, 60000, step=1000)
            home = st.selectbox("Home ownership", HOME, index=3)

        st.subheader("💳 Credit profile")
        c4, c5, c6 = st.columns(3)
        with c4:
            credit_score = st.slider("Credit score", 490, 770, 650)
        with c5:
            cred_hist = st.number_input("Credit history length (years)", 2, 13, 4)
        with c6:
            prev_default = st.radio("Previous loan default on file?", ["No", "Yes"], horizontal=True)

        st.subheader("💰 Loan")
        c7, c8, c9 = st.columns(3)
        with c7:
            loan_amnt = st.number_input("Loan amount", 500, 22500, 8000, step=500)
        with c8:
            int_rate = st.slider("Interest rate (%)", 5.0, 20.0, 11.0, step=0.1)
        with c9:
            intent = st.selectbox("Loan purpose", INTENT, index=4)

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted:
        return

    pct_income = round(min(loan_amnt / income, 0.66), 2)
    st.info(f"Loan as share of income (calculated): **{pct_income:.0%}**")

    X_new = build_features(
        {
            "person_age": float(age),
            "person_gender": gender,
            "person_education": education,
            "person_income": float(income),
            "person_emp_exp": emp_exp,
            "person_home_ownership": home,
            "loan_amnt": float(loan_amnt),
            "loan_intent": intent,
            "loan_int_rate": float(int_rate),
            "loan_percent_income": pct_income,
            "cb_person_cred_hist_length": float(cred_hist),
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": prev_default,
        },
        columns,
    )

    pred = int(model.predict(X_new)[0])
    proba = model.predict_proba(X_new)[0]
    p_class1 = float(proba[list(model.classes_).index(1)])

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        # Model target: loan_status 1 = default, 0 = non-default
        if pred == 1:
            st.error("### ⚠️ High risk: likely to default")
        else:
            st.success("### ✅ Low risk: likely to repay")
        st.progress(p_class1, text=f"Probability of default: {p_class1:.1%}")
    with right:
        st.metric("Default probability", f"{p_class1:.1%}")
        st.metric("Repayment probability", f"{1 - p_class1:.1%}")

    with st.expander("See the encoded input sent to the model"):
        st.dataframe(X_new.T.rename(columns={0: "value"}), use_container_width=True)

    st.caption(
        "This is a baseline model (84% accuracy, ~42% recall on the default class). "
        "It is a learning project, not a substitute for real credit decisions."
    )


if __name__ == "__main__":
    main()
