import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

st.set_page_config(page_title="ローン審査AI", layout="centered")
st.title("ローンデフォルト予測AI")

# ======================
# 1. モデル読み込み
# ======================
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

# ======================
# 2. 入力フォーム
# ======================
st.subheader("申請者情報の入力")

with st.form("input_form"):
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

    submit = st.form_submit_button("AI審査を開始")

# ======================
# 3. 予測実行
# ======================
if submit:
    # 1. 辞書形式でデータを作成（エラーログの日本語名と英語名の混在を考慮）
    # モデルが期待する名前が「GrossApproval」ならそれを使います
    input_data = {
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
    }

    # 特徴量名が日本語で学習されている場合は、ここでキーを日本語に書き換えます
    # ※エラーログの expected_features を見て、もし日本語ならそれを使ってください
    # 今回はエラーログに基づいて、英語名で作成します
    input_df = pd.DataFrame([input_data])

    # 2. 列の順序をモデルに合わせる
    input_df = input_df.reindex(columns=expected_features)

    # 3. 【最重要】型を強制的に変換する
    # CatBoostがFloatと期待している列を確実に float にする
    numeric_cols = [
        "GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear",
        "InitialInterestRate", "TermInMonths", "NaicsSector", 
        "CongressionalDistrict", "JobsSupported"
    ]

    for col in input_df.columns:
        if col in numeric_cols:
            # 数値型に変換（エラーは NaN になるが、CatBoost は NaN を許容する）
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
        else:
            # カテゴリ列は確実に文字列型にする
            input_df[col] = input_df[col].astype(str)

    # 4. カテゴリ変数のインデックスを取得
    cat_features_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']

    try:
        # Poolの作成
        pool = Pool(input_df, cat_features=cat_features_idx)
        
        # 予測
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
        st.write(f"エラー詳細: {e}")
        st.write("現在のデータ型:", input_df.dtypes)
