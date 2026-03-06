import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ==============================
# モデル読み込み
# ==============================

model = joblib.load("loan_model.pkl")

st.set_page_config(page_title="銀行ローン審査AI", layout="centered")

st.title("🏦 銀行ローン審査AI デモ")

# ==============================
# データ読み込み
# ==============================

DATA_PATH = "train.csv"

@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

df = get_data()

# ==============================
# データ概要
# ==============================

if not df.empty:

    st.subheader("📊 過去ローンデータ概要")

    col1,col2,col3 = st.columns(3)

    col1.metric("データ件数",len(df))
    col2.metric("平均融資額",f"${int(df['GrossApproval'].mean()):,}")
    col3.metric("平均返済期間",f"{int(df['TermInMonths'].mean())}ヶ月")

# ==============================
# 入力フォーム
# ==============================

with st.form("input_form"):

    st.markdown("### 📋 審査案件入力")

    gross_val = st.number_input(
        "融資申込総額 (USD)",
        min_value=1000,
        max_value=5000000,
        value=50000,
        step=1000
    )

    term_val = st.slider(
        "返済期間 (ヶ月)",
        min_value=12,
        max_value=360,
        value=120
    )

    interest_val = st.slider(
        "金利 (%)",
        min_value=1.0,
        max_value=20.0,
        value=8.0
    )

    jobs_val = st.number_input(
        "雇用人数",
        min_value=0,
        max_value=500,
        value=3
    )

    sector_val = st.selectbox(
        "業種",
        [
            "Accommodation_food services",
            "Retail trade",
            "Construction",
            "Information",
            "Manufacturing"
        ]
    )

    business_type = st.selectbox(
        "企業形態",
        [
            "CORPORATION",
            "SOLE PROPRIETORSHIP",
            "PARTNERSHIP"
        ]
    )

    business_age = st.selectbox(
        "企業年齢",
        [
            "Startup, Loan Funds will Open Business",
            "Existing or more than 2 years old"
        ]
    )

    submitted = st.form_submit_button("🚀 AI審査実行")

# ==============================
# 審査実行
# ==============================

if submitted:

    if df.empty:
        st.error("参照データが存在しません")
        st.stop()

    st.success("AI解析を実行しました")

    # ==============================
    # AI入力データ
    # ==============================

    input_df = pd.DataFrame({

        "id":[0],

        "GrossApproval":[gross_val],

        "SBAGuaranteedApproval":[gross_val*0.75],

        "ApprovalFiscalYear":[2020],

        "Subprogram":["7(a)"],

        "InitialInterestRate":[interest_val],

        "FixedOrVariableInterestInd":["F"],

        "TermInMonths":[term_val],

        "NaicsSector":[sector_val],

        "CongressionalDistrict":[1],

        "BusinessType":[business_type],

        "BusinessAge":[business_age],

        "RevolverStatus":["N"],

        "JobsSupported":[jobs_val],

        "CollateralInd":["N"]

    })

    # ==============================
    # 特徴量順を学習データと一致
    # ==============================

    feature_order = [

        "id",
        "GrossApproval",
        "SBAGuaranteedApproval",
        "ApprovalFiscalYear",
        "Subprogram",
        "InitialInterestRate",
        "FixedOrVariableInterestInd",
        "TermInMonths",
        "NaicsSector",
        "CongressionalDistrict",
        "BusinessType",
        "BusinessAge",
        "RevolverStatus",
        "JobsSupported",
        "CollateralInd"

    ]

    input_df = input_df[feature_order]

    # ==============================
    # AI予測
    # ==============================

    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🤖 AI審査結果")

    st.metric(
        label="デフォルト確率",
        value=f"{proba*100:.1f}%"
    )

    if proba < 0.3:
        st.success("✅ 融資承認")
    elif proba < 0.6:
        st.warning("⚠️ 要追加審査")
    else:
        st.error("❌ 融資拒否")

    # ==============================
    # リスク可視化
    # ==============================

    st.subheader("📉 リスクレベル")

    st.progress(float(proba))

    # ==============================
    # 類似案件検索
    # ==============================

    st.subheader("🔍 過去の類似案件")

    df["score"] = (

        abs(df["GrossApproval"]-gross_val)/gross_val +

        abs(df["TermInMonths"]-term_val)/term_val +

        abs(df["InitialInterestRate"]-interest_val)/interest_val

    )

    similar = df.sort_values("score").head(5).copy()

    similar["結果"] = similar["LoanStatus"].apply(
        lambda x:"✅ 完済" if x==1 else "⚠️ 不履行"
    )

    display = similar[[

        "GrossApproval",
        "TermInMonths",
        "InitialInterestRate",
        "結果"

    ]].rename(columns={

        "GrossApproval":"融資額",
        "TermInMonths":"期間",
        "InitialInterestRate":"金利"

    })

    st.table(display)

else:

    st.info("👆 融資条件を入力して審査を実行してください")
