import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool

st.title("ローンデフォルト予測AI")

# ======================
# モデル読み込み
# ======================

model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# ======================
# 入力
# ======================

gross = st.number_input("融資額", 1000, 1000000, 50000)

sba = st.number_input("保証額", 0, 1000000, 30000)

year = st.number_input("承認年度", 1990, 2025, 2010)

subprogram = st.selectbox(
    "ローンプログラム",
    ["7(a)", "FA$TRK (Small Loan Express)", "Express"]
)

rate = st.number_input("金利", 0.0, 20.0, 5.0)

rate_type = st.selectbox(
    "金利タイプ",
    ["Fixed", "Variable"]
)

term = st.number_input("返済期間(月)", 1, 360, 120)

sector = st.number_input("産業セクター", 1, 99, 44)

district = st.number_input("地区", 1, 60, 10)

business_type = st.selectbox(
    "企業形態",
    ["Corporation", "Partnership", "Sole Proprietorship", "LLC"]
)

business_age = st.selectbox(
    "企業年齢",
    ["Startup", "Existing"]
)

revolver = st.selectbox(
    "リボルビングローン",
    ["Y", "N"]
)

jobs = st.number_input("雇用人数", 0, 1000, 5)

collateral = st.selectbox(
    "担保",
    ["Y", "N"]
)

# ======================
# 予測
# ======================

if st.button("予測"):

    data = pd.DataFrame([{
        "GrossApproval": gross,
        "SBAGuaranteedApproval": sba,
        "ApprovalFiscalYear": year,
        "Subprogram": subprogram,
        "InitialInterestRate": rate,
        "FixedOrVariableInterestInd": rate_type,
        "TermInMonths": term,
        "NaicsSector": sector,
        "CongressionalDistrict": district,
        "BusinessType": business_type,
        "BusinessAge": business_age,
        "RevolverStatus": revolver,
        "JobsSupported": jobs,
        "CollateralInd": collateral
    }])

    cat_features = [
        "Subprogram",
        "FixedOrVariableInterestInd",
        "BusinessType",
        "BusinessAge",
        "RevolverStatus",
        "CollateralInd"
    ]

    pool = Pool(data, cat_features=cat_features)

    proba = model.predict_proba(pool)[0][1]

    st.write("デフォルト確率:", round(proba * 100, 2), "%")

    if proba > 0.5:
        st.error("デフォルトリスク高")
    else:
        st.success("デフォルトリスク低")
