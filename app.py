import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI", layout="centered")
st.title("ローンデフォルト予測AI")

# 2. モデル読み込み
@st.cache_resource
def load_my_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

try:
    model = load_my_model()
    expected_features = model.feature_names_
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# 3. 入力フォーム
st.subheader("申請者情報の入力")

# ここで st.form を定義
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
        sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
        year = st.number_input("承認年度", 1990, 2026, 2010)
        rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
        term = st.number_input("返済期間 (月)", 1, 360, 120)
        jobs = st.number_input("雇用人数", 0, 1000, 5)

    with col2:
        subprogram = st.text_input("ローンプログラム", "7(a)")
        rate_type = st.selectbox("金利タイプ", ["Fixed", "Variable"])
        sector = st.number_input("産業セクター (NAICS)", 1, 99, 44)
        district = st.number_input("地区コード", 1, 60, 10)
        business_type = st.text_input("企業形態", "CORPORATION")
        business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
        revolver = st.selectbox("リボルビングローン", ["Y", "N"])
        collateral = st.selectbox("担保の有無", ["Y", "N"])

    # ここで submit 変数を定義
    submit = st.form_submit_button("AI審査を開始")

# ======================
# 3. 予測実行
# ======================
if submit:
    # 1. データの作成
    # 前回の修正: RevolverStatus は Float
    revolver_numeric = 1.0 if revolver == "Y" else 0.0

    input_data = {
        "GrossApproval": float(gross),
        "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year),
        "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate),
        "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term),
        "NaicsSector": str(int(sector)),  # 【修正】数値から文字列(カテゴリ)へ変換
        "CongressionalDistrict": float(district),
        "BusinessType": str(business_type),
        "BusinessAge": str(business_age),
        "RevolverStatus": float(revolver_numeric),
        "JobsSupported": float(jobs),
        "CollateralInd": str(collateral)
    }

    input_df = pd.DataFrame([input_data])

    # 2. 列の順序をモデルに合わせる
    input_df = input_df.reindex(columns=expected_features)

    # 3. 型の最終調整
    # 今回のエラーに基づき、NaicsSector を数値リストから除外し、文字列リストに入れます
    numeric_cols = [
        "GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear",
        "InitialInterestRate", "TermInMonths", 
        "CongressionalDistrict", "JobsSupported", "RevolverStatus"
    ]

    for col in input_df.columns:
        if col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
        else:
            # NaicsSector を含むカテゴリ列は確実に文字列にする
            input_df[col] = input_df[col].astype(str)

    # 4. カテゴリ変数のインデックス取得
    cat_features_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']

    try:
        pool = Pool(input_df, cat_features=cat_features_idx)
        proba = model.predict_proba(pool)[0][1]

        # 結果表示
        st.markdown("---")
        st.subheader("審査結果")
        st.metric("デフォルト確率", f"{round(proba * 100, 2)} %")

        if proba > 0.5:
            st.error("【判定】融資危険")
        else:
            st.success("【判定】融資可能")

    except Exception as e:
        st.error(f"予測中にエラーが発生しました。")
        st.write(f"詳細: {e}")
