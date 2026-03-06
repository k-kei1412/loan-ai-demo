import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool
from preprocess import preprocess

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

subprogram = st.text_input("ローンプログラム")

rate = st.number_input("金利", 0.0, 20.0, 5.0)

rate_type = st.selectbox(
    "金利タイプ",
    ["Fixed", "Variable"]
)

term = st.number_input("返済期間(月)", 1, 360, 120)

sector = st.number_input("産業セクター", 1, 99, 44)

district = st.number_input("地区", 1, 60, 10)

business_type = st.text_input("企業形態")

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

if st.button("AI審査"):

    input_df = pd.DataFrame([{
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

    input_df, cat_features = preprocess(input_df)

    pool = Pool(input_df, cat_features=cat_features)

    proba = model.predict_proba(pool)[0][1]

    st.write("デフォルト確率:", round(proba * 100, 2), "%")

    if proba > 0.5:
        st.error("融資危険")
    else:
        st.success("融資可能")
