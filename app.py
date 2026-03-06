import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

st.set_page_config(page_title="銀行ローン審査AI")

st.title("🏦 銀行ローン審査AI")

# =====================
# モデル読み込み
# =====================

model = joblib.load("catboost_model.pkl")

# =====================
# モデル特徴量取得
# =====================

feature_names = model.feature_names_

# =====================
# 入力UI
# =====================

with st.form("loan_form"):

    gross = st.number_input("融資額",1000,5000000,50000)
    term = st.slider("返済期間(月)",12,360,120)
    interest = st.slider("金利",1.0,20.0,8.0)
    jobs = st.number_input("雇用人数",0,500,3)

    sector = st.selectbox(
        "業種",
        [
            "Accommodation_food services",
            "Retail trade",
            "Construction",
            "Information",
            "Manufacturing"
        ]
    )

    btype = st.selectbox(
        "企業形態",
        ["CORPORATION","SOLE PROPRIETORSHIP","PARTNERSHIP"]
    )

    bage = st.selectbox(
        "企業年齢",
        [
            "Startup, Loan Funds will Open Business",
            "Existing or more than 2 years old"
        ]
    )

    submit = st.form_submit_button("AI審査")

# =====================
# 予測
# =====================

if submit:

    input_dict = {
        "GrossApproval":gross,
        "SBAGuaranteedApproval":gross*0.75,
        "InitialInterestRate":interest,
        "TermInMonths":term,
        "JobsSupported":jobs,
        "NaicsSector":sector,
        "BusinessType":btype,
        "BusinessAge":bage,
        "RevolverStatus":"N"
    }

    input_df = pd.DataFrame([input_dict])

    # モデルの列順に合わせる
    input_df = input_df.reindex(columns=feature_names)

    # カテゴリ列
    cat_features = input_df.select_dtypes(include=["object"]).columns.tolist()

    pool = Pool(input_df, cat_features=cat_features)

    proba = model.predict_proba(pool)[0][1]

    st.subheader("審査結果")

    st.metric("デフォルト確率",f"{proba*100:.1f}%")

    if proba < 0.3:
        st.success("融資承認")
    elif proba < 0.6:
        st.warning("追加審査")
    else:
        st.error("融資拒否")
