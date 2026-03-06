import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

st.title("ローン審査AI")

# =========================
# モデル読み込み
# =========================

model = joblib.load("catboost_model.pkl")

# =========================
# 入力UI
# =========================

gross = st.number_input("融資額",1000,5000000,50000)

term = st.slider(
    "返済期間（月）",
    12,
    360,
    120
)

interest = st.slider(
    "金利",
    1.0,
    20.0,
    8.0
)

jobs = st.number_input(
    "雇用人数",
    0,
    500,
    3
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

predict = st.button("AI審査")

# =========================
# 予測
# =========================

if predict:

    input_df = pd.DataFrame({

        "GrossApproval":[gross],
        "SBAGuaranteedApproval":[gross*0.75],
        "InitialInterestRate":[interest],
        "TermInMonths":[term],
        "JobsSupported":[jobs],
        "NaicsSector":[sector],
        "BusinessType":[btype],
        "BusinessAge":[bage],
        "RevolverStatus":["N"]

    })

    # 文字型に変換（重要）
    cat_cols = [
        "NaicsSector",
        "BusinessType",
        "BusinessAge",
        "RevolverStatus"
    ]

    for col in cat_cols:
        input_df[col] = input_df[col].astype(str)

    # Pool作成
    pool = Pool(
        input_df,
        cat_features=cat_cols
    )

    # 予測
    proba = model.predict_proba(pool)[0][1]

    st.subheader("審査結果")

    st.write("デフォルト確率:",round(proba*100,1),"%")

    if proba < 0.3:
        st.success("融資承認")

    elif proba < 0.6:
        st.warning("追加審査")

    else:
        st.error("融資拒否")
