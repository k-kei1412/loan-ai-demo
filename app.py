import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="ローン審査AI", layout="centered")

st.title("🏦 ローン審査AI デモ")

# ==============================
# データ読み込み
# ==============================

DATA_PATH = "train.csv"

@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=['GrossApproval','TermInMonths','LoanStatus'])

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
        max_value=1000000,
        value=50000,
        step=1000
    )

    term_val = st.slider(
        "返済期間 (ヶ月)",
        min_value=12,
        max_value=360,
        value=120
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
    # 簡易AIスコア
    # ==============================

    loan_ratio = gross_val / df["GrossApproval"].mean()
    term_ratio = term_val / df["TermInMonths"].mean()

    risk_score = (loan_ratio*0.6 + term_ratio*0.4)

    default_prob = min(max((risk_score-0.8)/2,0),1)

    st.subheader("🤖 AI審査結果")

    st.metric(
        label="デフォルト確率",
        value=f"{default_prob*100:.1f}%"
    )

    if default_prob < 0.4:
        st.success("✅ 承認可能")
    elif default_prob < 0.7:
        st.warning("⚠️ 要注意")
    else:
        st.error("❌ リスク高")

    # ==============================
    # リスクゲージ
    # ==============================

    st.subheader("📉 リスクレベル")

    st.progress(float(default_prob))

    # ==============================
    # 類似案件検索
    # ==============================

    st.subheader("🔍 過去の類似案件")

    df["score"] = (
        abs(df["GrossApproval"] - gross_val)/gross_val +
        abs(df["TermInMonths"] - term_val)/term_val
    )

    similar = df.sort_values("score").head(5).copy()

    similar["結果"] = similar["LoanStatus"].apply(
        lambda x: "✅ 完済" if x==1 else "⚠️ 不履行"
    )

    display = similar[[
        "GrossApproval",
        "TermInMonths",
        "結果"
    ]].rename(columns={
        "GrossApproval":"融資額",
        "TermInMonths":"期間"
    })

    st.table(display)

else:

    st.info("👆 融資条件を入力して審査を実行してください")
