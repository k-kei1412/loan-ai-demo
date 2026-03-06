import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

st.title("ローン審査AI")

# ======================
# モデル読み込み
# ======================

model = joblib.load("catboost_model.pkl")

# ======================
# 入力UI
# ======================

loan_id = st.number_input("Loan ID",0,999999,1)

gross = st.number_input("融資額",1000,5000000,50000)

fiscal = st.number_input("年度",1990,2030,2020)

interest = st.slider("金利",1.0,20.0,8.0)

term = st.slider("返済期間(月)",12,360,120)

jobs = st.number_input("雇用人数",0,500,3)

district = st.number_input("選挙区",0,60,1)

subprogram = st.selectbox(
    "ローンプログラム",
    [
        "7(a)",
        "FA$TRK (Small Loan Express)",
        "Community Express",
        "Express"
    ]
)

interest_type = st.selectbox(
    "金利タイプ",
    [
        "F",
        "V"
    ]
)

sector = st.selectbox(
    "業種",
    [
        "Accommodation_food services",
        "Retail trade",
        "Construction",
        "Manufacturing",
        "Information"
    ]
)

btype = st.selectbox(
    "企業形態",
    [
        "CORPORATION",
        "SOLE PROPRIETORSHIP",
        "PARTNERSHIP"
    ]
)

bage = st.selectbox(
    "企業年齢",
    [
        "Startup, Loan Funds will Open Business",
        "Existing or more than 2 years old"
    ]
)

revolver = st.selectbox(
    "リボルビング",
    ["Y","N"]
)

collateral = st.selectbox(
    "担保",
    ["Y","N"]
)

predict = st.button("AI審査")

# ======================
# 予測
# ======================

if predict:

    input_df = pd.DataFrame({

        "id":[loan_id],
        "GrossApproval":[gross],
        "SBAGuaranteedApproval":[gross*0.75],
        "ApprovalFiscalYear":[fiscal],
        "Subprogram":[subprogram],
        "InitialInterestRate":[interest],
        "FixedOrVariableInterestInd":[interest_type],
        "TermInMonths":[term],
        "NaicsSector":[sector],
        "CongressionalDistrict":[district],
        "BusinessType":[btype],
        "BusinessAge":[bage],
        "RevolverStatus":[revolver],
        "JobsSupported":[jobs],
        "CollateralInd":[collateral]

    })

    cat_cols = [
        "Subprogram",
        "FixedOrVariableInterestInd",
        "NaicsSector",
        "BusinessType",
        "BusinessAge",
        "RevolverStatus",
        "CollateralInd"
    ]

    for c in cat_cols:
        input_df[c] = input_df[c].astype(str)

    pool = Pool(input_df, cat_features=cat_cols)

    proba = model.predict_proba(pool)[0][1]

    st.subheader("審査結果")

    st.write("デフォルト確率:",round(proba*100,1),"%")

    if proba < 0.3:
        st.success("融資承認")

    elif proba < 0.6:
        st.warning("追加審査")

    else:
        st.error("融資拒否")
