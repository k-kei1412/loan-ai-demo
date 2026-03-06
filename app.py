import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

# =========================
# タイトル
# =========================
st.title("Loan Default Prediction AI")

st.write("ローン返済リスクを予測するAIデモ")

# =========================
# モデル読み込み
# =========================
model = joblib.load("catboost_model.pkl")

# =========================
# 入力UI
# =========================

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)

person_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

person_emp_length = st.number_input(
    "Employment Length (years)",
    min_value=0.0,
    value=5.0
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

loan_int_rate = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.2
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=0.0,
    value=5.0
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

cb_person_default_on_file = st.selectbox(
    "Previous Default",
    ["Y", "N"]
)

# =========================
# 特徴量順序
# =========================

feature_order = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

# =========================
# 予測ボタン
# =========================

if st.button("Predict"):

    input_dict = {
        "person_age": float(person_age),
        "person_income": float(person_income),
        "person_home_ownership": person_home_ownership,
        "person_emp_length": float(person_emp_length),
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": float(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": float(cb_person_cred_hist_length)
    }

    input_df = pd.DataFrame([input_dict])

    input_df = input_df[feature_order]

    # カテゴリ特徴量
    cat_features = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    # CatBoost用Pool
    pool = Pool(input_df, cat_features=cat_features)

    # 予測
    prediction = model.predict(pool)[0]
    probability = model.predict_proba(pool)[0][1]

    # =========================
    # 結果表示
    # =========================

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Default Risk")
    else:
        st.success("Low Default Risk")

    st.write("Default Probability")
    st.write(f"{probability:.2%}")
