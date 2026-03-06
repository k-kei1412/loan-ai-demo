import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
# 自作の前処理ファイルをインポート
from preprocess import preprocess

# ページ設定
st.set_page_config(page_title="ローン審査AI", layout="centered")
st.title("ローンデフォルト予測AI")

# ======================
# 1. モデル読み込み
# ======================
@st.cache_resource
def load_my_model():
    model = CatBoostClassifier()
    # モデルファイル名が catboost_model.cbm であることを確認してください
    model.load_model("catboost_model.cbm")
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。ファイル名を確認してください: {e}")
    st.stop()

# ======================
# 2. ユーザー入力フォーム
# ======================
st.subheader("申請者情報の入力")

col1, col2 = st.columns(2)

with col1:
    gross = st.number_input("融資額 ($)", 1000, 1000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 1000000, 30000)
    year = st.number_input("承認年度", 1990, 2026, 2010)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    jobs = st.number_input("雇用人数", 0, 1000, 5)

with col2:
    subprogram = st.text_input("ローンプログラム (例: 7(a))", "7(a)")
    rate_type = st.selectbox("金利タイプ", ["Fixed", "Variable"])
    sector = st.number_input("産業セクター (NAICS)", 1, 99, 44)
    district = st.number_input("地区コード", 1, 60, 10)
    business_type = st.text_input("企業形態 (例: CORPORATION)", "CORPORATION")
    business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
    revolver = st.selectbox("リボルビングローン", ["Y", "N"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])

# ======================
# 3. 予測実行
# ======================
if st.button("AI審査を開始"):
    # 入力データをデータフレーム化
    raw_df = pd.DataFrame([{
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

    # --- 前処理の適用 ---
    # preprocess.py の関数を使用
    df_processed, _ = preprocess(raw_df)

    # --- CatBoostエラー対策：列の順序を学習時と一致させる ---
    # モデルが期待する特徴量名を取得
    expected_features = model.feature_names_
    
    # モデルが期待する列だけを、期待する順番で抽出（足りない列はNaNで補完）
    df_final = df_processed.reindex(columns=expected_features)

    # カテゴリ変数のインデックスを現在の列順から再取得（安全策）
    cat_features_idx = [i for i, col in enumerate(df_final.columns) if df_final[col].dtype == 'object']

    try:
        # Poolの作成
        pool = Pool(df_final, cat_features=cat_features_idx)
        
        # 予測
        proba = model.predict_proba(pool)[0][1]

        # 結果表示
        st.markdown("---")
        st.subheader("審査結果")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("デフォルト確率", f"{round(proba * 100, 2)} %")

        if proba > 0.5:
            col_res2.error("【判定】融資危険")
            st.warning("この案件はデフォルト（債務不履行）のリスクが高いと判断されました。")
        else:
            col_res2.success("【判定】融資可能")
            st.info("この案件は正常に返済される可能性が高いと判断されました。")

    except Exception as e:
        # エラーが出た場合、詳細を表示してデバッグしやすくする
        st.error(f"予測中にエラーが発生しました。")
        with st.expander("エラー詳細を確認"):
            st.write(e)
            st.write("モデルが期待する特徴量:", expected_features)
            st.write("送信されたデータの列:", df_final.columns.tolist())
