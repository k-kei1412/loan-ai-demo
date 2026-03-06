import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI", layout="wide") # 広く使うためにwideに設定
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

    submit = st.form_submit_button("AI審査を開始")

# 4. 予測実行
if submit:
    # データ変換
    revolver_numeric = 1.0 if revolver == "Y" else 0.0
    input_data = {
        "GrossApproval": float(gross),
        "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year),
        "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate),
        "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term),
        "NaicsSector": str(int(sector)),
        "CongressionalDistrict": float(district),
        "BusinessType": str(business_type),
        "BusinessAge": str(business_age),
        "RevolverStatus": float(revolver_numeric),
        "JobsSupported": float(jobs),
        "CollateralInd": str(collateral)
    }

    df_final = pd.DataFrame([input_data]).reindex(columns=expected_features)
    
    # 型の強制
    numeric_cols = ["GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear", "InitialInterestRate", "TermInMonths", "CongressionalDistrict", "JobsSupported", "RevolverStatus"]
    for col in df_final.columns:
        if col in numeric_cols:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype(float)
        else:
            df_final[col] = df_final[col].astype(str)

    cat_features_idx = [i for i, col in enumerate(df_final.columns) if df_final[col].dtype == 'object']

    try:
        # 本番予測
        pool = Pool(df_final, cat_features=cat_features_idx)
        proba = model.predict_proba(pool)[0][1]

        # 🚀 結果表示
        st.markdown("---")
        st.subheader("審査結果")
        col_res1, col_res2 = st.columns(2)
        
        # 0.0%の対策として小さな値を表示
        display_proba = round(proba * 100, 4)
        col_res1.metric("デフォルト確率", f"{display_proba} %")

        if proba > 0.5:
            col_res2.error("【判定】融資危険")
        else:
            col_res2.success("【判定】融資可能")

        # 📊 💡 追加機能：今回の判断で重要だった項目
        st.write("### 💡 AIが注目した項目")
        importances = model.get_feature_importance()
        feat_imp_df = pd.DataFrame({'項目': expected_features, '影響度': importances}).sort_values(by='影響度', ascending=False).head(5)
        st.table(feat_imp_df)

        # 🔍 💡 追加機能：条件シミュレーション（似たようなケース）
        st.write("### 🔍 条件を変えた場合のシミュレーション")
        st.write("「もし金利が今より3%高かったら？」という似たケースと比較します。")
        
        sim_df = df_final.copy()
        sim_df["InitialInterestRate"] += 3.0 # 金利を3%アップ
        sim_pool = Pool(sim_df, cat_features=cat_features_idx)
        sim_proba = model.predict_proba(sim_pool)[0][1]
        
        col_sim1, col_sim2 = st.columns(2)
        col_sim1.metric("現在の条件", f"{display_proba} %")
        col_sim2.metric("金利 +3% の場合", f"{round(sim_proba * 100, 4)} %", delta=f"{round((sim_proba - proba)*100, 4)}%", delta_color="inverse")

    except Exception as e:
        st.error(f"エラー: {e}")
