import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="🏦", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "..", "models", "credit_risk_pipeline.joblib")
INFO_PATH = os.path.join(BASE, "..", "models", "model_info.json")

RAW_COLUMNS = ["person_age", "person_income", "person_home_ownership", "person_emp_length",
               "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
               "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]
GRADES = ["A", "B", "C", "D", "E", "F", "G"]
HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            return json.load(f)
    return {}


def add_engineered_features(data):
    """Same feature engineering as in the notebook (Section 1.7)."""
    data = data.copy()
    data["emp_to_age_ratio"] = (data["person_emp_length"] / data["person_age"]).round(4)
    data["interest_cost"] = (data["loan_amnt"] * data["loan_int_rate"] / 100).round(2)
    data["interest_to_income"] = (data["interest_cost"] / data["person_income"]).round(4)
    data["age_group"] = pd.cut(data["person_age"], bins=[0, 25, 35, 45, 100],
                               labels=["up to 25", "26-35", "36-45", "46+"]).astype(str)
    return data


def predict_proba(model, raw_df):
    return model.predict_proba(add_engineered_features(raw_df))[:, 1]


try:
    model = load_model()
except Exception as e:
    st.error("The saved model could not be loaded. Check that the models folder contains "
             "credit_risk_pipeline.joblib and that the same scikit-learn version is installed.")
    st.exception(e)
    st.stop()

info = load_info()

# ---------------- Header ----------------
st.title("🏦 Loan Default Risk Predictor")
st.caption("Devixo Solutions | AI/ML Internship Program | Task 03 | End-to-End Machine Learning Project")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
default_thr = float(info.get("threshold", 0.5))
threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, default_thr, 0.05,
                              help="A loan is flagged as high risk when its default probability is at least "
                                   "this value. A lower threshold finds more defaults but also flags more good loans.")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {info.get('model', 'Gradient Boosting')}")
metrics = info.get("test_metrics", {})
if metrics:
    st.sidebar.markdown(f"**Test accuracy:** {metrics.get('accuracy', 0) * 100:.2f}%  \n"
                        f"**Precision:** {metrics.get('precision', 0) * 100:.2f}%  \n"
                        f"**Recall:** {metrics.get('recall', 0) * 100:.2f}%  \n"
                        f"**F1 score:** {metrics.get('f1', 0):.4f}  \n"
                        f"**ROC-AUC:** {metrics.get('roc_auc', 0):.4f}")
st.sidebar.markdown("---")
st.sidebar.caption("Demo model trained on a public dataset. It is not financial advice.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Information", "📁 Batch Predictions"])

# ---------------- Tab 1: single prediction ----------------
with tab1:
    st.subheader("Enter the applicant and loan details")
    left, right = st.columns([3, 2])
    with left:
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 18, 100, 28)
        income = c2.number_input("Annual income", 1000, 6000000, 55000, step=1000)
        home = c3.selectbox("Home ownership", HOME)
        c4, c5, c6 = st.columns(3)
        emp = c4.number_input("Employment length (years)", 0.0, 60.0, 4.0, step=1.0)
        hist = c5.number_input("Credit history length (years)", 2, 40, 4)
        prev_default = c6.selectbox("Previous default on file", ["N", "Y"],
                                    format_func=lambda x: "Yes" if x == "Y" else "No")
        c7, c8, c9 = st.columns(3)
        intent = c7.selectbox("Loan purpose", INTENT)
        grade = c8.selectbox("Loan grade (A = lowest risk)", GRADES, index=2)
        amount = c9.number_input("Loan amount", 500, 35000, 8000, step=500)
        rate = st.slider("Interest rate (%)", 5.0, 25.0, 11.0, 0.1)

    pct_income = round(amount / income, 2)
    row = pd.DataFrame([{
        "person_age": age, "person_income": income, "person_home_ownership": home,
        "person_emp_length": emp, "loan_intent": intent, "loan_grade": grade,
        "loan_amnt": amount, "loan_int_rate": rate, "loan_percent_income": pct_income,
        "cb_person_default_on_file": prev_default, "cb_person_cred_hist_length": hist}])
    prob = float(predict_proba(model, row)[0])
    high_risk = prob >= threshold

    with right:
        st.subheader("Result")
        st.metric("Default probability", f"{prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))
        if high_risk:
            st.error(f"HIGH RISK: default is likely (probability {prob * 100:.1f}% is at or above the "
                     f"threshold of {threshold * 100:.0f}%).")
        else:
            st.success(f"LOW RISK: default is unlikely (probability {prob * 100:.1f}% is below the "
                       f"threshold of {threshold * 100:.0f}%).")
        st.markdown("**Calculated values**")
        st.write(f"- Loan as a share of income: **{pct_income * 100:.0f}%**")
        st.write(f"- Yearly interest cost: **{amount * rate / 100:,.0f}** "
                 f"({amount * rate / 100 / income * 100:.1f}% of income)")
        if pct_income > 0.3:
            st.warning("The loan is more than 30% of the yearly income, which is a risk factor.")
        if grade in ["E", "F", "G"]:
            st.warning("A low loan grade (E, F or G) is the strongest known risk factor in this data.")

# ---------------- Tab 2: model information ----------------
with tab2:
    st.subheader("About the model")
    st.write("The prediction comes from a complete scikit-learn pipeline (median imputation, encoding, scaling, "
             "feature selection and the model) that was trained on about 26,000 loans and saved with Joblib. "
             "Four features are engineered automatically: employment length to age ratio, yearly interest cost, "
             "interest to income ratio and age group.")
    if metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
        m2.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
        m3.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")
        m4.metric("F1 score", f"{metrics.get('f1', 0):.4f}")
        m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    if info.get("best_params"):
        st.markdown("**Best hyperparameters (from tuning)**")
        st.json(info["best_params"])
    try:
        names = model[:-1].get_feature_names_out()
        imp = pd.Series(model.named_steps["model"].feature_importances_, index=names)
        imp = imp.sort_values(ascending=False).head(10)
        imp.index = [n.split("__", 1)[-1] for n in imp.index]
        st.markdown("**Top 10 most important model features**")
        st.bar_chart(imp)
    except Exception:
        st.info("Feature importance is not available for this model.")
    st.markdown("**Known limitations**")
    st.write("- About a quarter of real defaults are not detected at the default threshold (recall about 75%).\n"
             "- Only 11 input columns are used (no credit score or payment history).\n"
             "- The loan grade is assigned by the lender, so it must be known before the model can be used.")

# ---------------- Tab 3: batch predictions ----------------
with tab3:
    st.subheader("Score many applicants from a CSV file")
    st.write("Upload a CSV file with these columns: " + ", ".join(f"`{c}`" for c in RAW_COLUMNS if c != "loan_percent_income")
             + ". The column `loan_percent_income` is optional and is calculated if it is missing.")
    sample = pd.DataFrame([
        {"person_age": 35, "person_income": 90000, "person_home_ownership": "MORTGAGE", "person_emp_length": 10,
         "loan_intent": "EDUCATION", "loan_grade": "A", "loan_amnt": 5000, "loan_int_rate": 7.5,
         "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 10},
        {"person_age": 22, "person_income": 25000, "person_home_ownership": "RENT", "person_emp_length": 1,
         "loan_intent": "MEDICAL", "loan_grade": "E", "loan_amnt": 12000, "loan_int_rate": 17.5,
         "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 2}])
    st.download_button("Download a sample CSV", sample.to_csv(index=False).encode("utf-8"),
                       "sample_applicants.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            if "loan_percent_income" not in data.columns and {"loan_amnt", "person_income"} <= set(data.columns):
                data["loan_percent_income"] = (data["loan_amnt"] / data["person_income"]).round(2)
            missing_cols = [c for c in RAW_COLUMNS if c not in data.columns]
            if missing_cols:
                st.error("Missing columns: " + ", ".join(missing_cols))
            elif not data["loan_grade"].isin(GRADES).all():
                st.error("The column loan_grade may only contain the values A, B, C, D, E, F and G.")
            else:
                probs = predict_proba(model, data[RAW_COLUMNS])
                out = data.copy()
                out["default_probability"] = np.round(probs, 4)
                out["decision"] = np.where(probs >= threshold, "High risk", "Low risk")
                s1, s2, s3 = st.columns(3)
                s1.metric("Applicants", f"{len(out):,}")
                s2.metric("High risk", f"{(probs >= threshold).sum():,}")
                s3.metric("Average probability", f"{probs.mean() * 100:.1f}%")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results", out.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"The file could not be processed: {e}")
