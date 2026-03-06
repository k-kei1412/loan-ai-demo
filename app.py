import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(page_title="銀行ローン審査AI", layout="centered")

st.title("🏦 銀行ローン審査AI デモ")

# ==============================
# モデル読み込み
# ==============================

MODEL_PATH = "loan_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

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
        12,
        360,
        120
    )

    interest_val = st.slider(
        "金利 (%)",
        1.0,
        20.0,
        8.0
    )

    jobs_val = st.number_input(
        "雇用人数",
        0,
        500,
        3
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
        st.error("参照データがありません")
        st.stop()

    st.success("AI解析を実行しました")

    # ==============================
    # AI入力データ
    # ==============================

    input_df = df.drop(columns=["LoanStatus"]).iloc[[0]].copy()

    input_df["GrossApproval"] = gross_val
    input_df["SBAGuaranteedApproval"] = gross_val * 0.75
    input_df["InitialInterestRate"] = interest_val
    input_df["TermInMonths"] = term_val
    input_df["JobsSupported"] = jobs_val
    input_df["NaicsSector"] = sector_val
    input_df["BusinessType"] = business_type
    input_df["BusinessAge"] = business_age
    input_df["RevolverStatus"] = "N"

    # ==============================
    # AI予測
    # ==============================

    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🤖 AI審査結果")

    st.metric(
        "デフォルト確率",
        f"{proba*100:.1f}%"
    )

    # ==============================
    # 銀行審査基準
    # ==============================

    if proba < 0.15:
        st.success("✅ 融資承認")

    elif proba < 0.35:
        st.warning("⚠️ 要追加審査")

    else:
        st.error("❌ 融資拒否")

    # ==============================
    # リスクゲージ
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

    # ==============================
    # リスク分析
    # ==============================

    st.subheader("📊 リスク分析")

    col1,col2,col3 = st.columns(3)

    loan_ratio = gross_val / df["GrossApproval"].mean()
    term_ratio = term_val / df["TermInMonths"].mean()
    interest_ratio = interest_val / df["InitialInterestRate"].mean()

    col1.metric("融資額リスク",f"{loan_ratio:.2f}")
    col2.metric("期間リスク",f"{term_ratio:.2f}")
    col3.metric("金利リスク",f"{interest_ratio:.2f}")

else:

    st.info("👆 融資条件を入力して審査を実行してください")
