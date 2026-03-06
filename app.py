import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

# ページ設定
st.set_page_config(page_title="ローン審査AI", layout="centered")
st.title("ローンデフォルト予測AI")

# ======================
# 1. モデル読み込み
# ======================
@st.cache_resource
def load_my_model():
    model = CatBoostClassifier()
    # モデルファイル名が catboost_model.cbm であることを確認
    model.load_model("catboost_model.cbm")
    return model

try:
    model = load_my_model()
    # モデルが期待する列名リストを取得
    expected_features = model.feature_names_
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# ======================
# 2. ユーザー入力フォーム
# ======================
st.subheader("申請者情報の入力")

with st.form("input_form"):
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

    submit = st.form_submit_button("AI審査を開始")

# ======================
# 3. 予測実行
# ======================
if submit:
    # 1. モデルの期待する「日本語列名」でデータフレームを作成
    # ※エラーログの「モデルが期待する特徴量」に完全に一致させています
    input_data = {
        "グロス・プラクソングレーション": float(gross),
        "SBAGuaranteedApproval": float(sba),
        "承認会計年度": float(year),
        "サブプログラム": str(subprogram),
        "初期金利": float(rate),
        "固定または可変興味インド": str(rate_type),
        "タームインマンス": float(term),
        "ナイクス・セクター": float(sector),
        "議会選挙区": float(district),
        "ビジネスタイプ": str(business_type),
        "ビジネスエイジ": str(business_age),
        "リボルバーステータス": str(revolver),
        "ジョブズサポートド": float(jobs),
        "コラテラルインド": str(collateral)
    }

    input_df = pd.DataFrame([input_data])

    # 2. 列の順序をモデルの学習時と完全に一致させる
    input_df = input_df.reindex(columns=expected_features)

    # 3. 数値列とカテゴリ列の型を明示的に指定
    # 数値として扱うべき列名（モデル側の名称）
    numeric_cols = [
        "グロス・プラクソングレーション", "SBAGuaranteedApproval", "承認会計年度",
        "初期金利", "タームインマンス", "ナイクス・セクター", 
        "議会選挙区", "ジョブズサポートド"
    ]
    
    for col in input_df.columns:
        if col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
        else:
            input_df[col] = input_df[col].astype(str)

    # 4. カテゴリ変数のインデックス（位置）を取得
    cat_features_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object' or col not in numeric_cols]

    try:
        # Poolの作成
        pool = Pool(input_df, cat_features=cat_features_idx)
        
        # 予測
        proba = model.predict_proba(pool)[0][1]

        # 結果表示
        st.markdown("---")
        st.subheader("審査結果")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("デフォルト確率", f"{round(proba * 100, 2)} %")

        if proba > 0.5:
            col_res2.error("【判定】融資危険")
            st.warning("リスクが高いと判断されました。条件の再検討を推奨します。")
        else:
            col_res2.success("【判定】融資可能")
            st.info("健全な案件である可能性が高いです。")

    except Exception as e:
        st.error("予測中にエラーが発生しました。")
        with st.expander("詳細なデバッグ情報"):
            st.write(f"エラー内容: {e}")
            st.write("入力データの型:", input_df.dtypes)
            st.write("作成されたデータ:", input_df)
