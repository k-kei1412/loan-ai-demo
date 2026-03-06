import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors # 近似事例の検索用

st.set_page_config(page_title="ローン審査AI", layout="wide")
st.title("ローンデフォルト予測AI")

# ======================
# 1. モデルと過去データの読み込み
# ======================
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    
    # 過去の事例を表示するために学習用データを読み込む
    # train_data.csv はご自身のファイル名に合わせてください
    try:
        train_df = pd.read_csv("train_data.csv")
    except:
        train_df = pd.DataFrame() # ファイルがない場合は空
    
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# ======================
# 2. 入力フォーム（中略）
# ======================
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

# ======================
# 3. 予測と事例検索
# ======================
if submit:
    # データ整形ロジック（前回同様）
    revolver_numeric = 1.0 if revolver == "Y" else 0.0
    input_data = {
        "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year), "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term), "NaicsSector": str(int(sector)),
        "CongressionalDistrict": float(district), "BusinessType": str(business_type),
        "BusinessAge": str(business_age), "RevolverStatus": float(revolver_numeric),
        "JobsSupported": float(jobs), "CollateralInd": str(collateral)
    }
    df_final = pd.DataFrame([input_data]).reindex(columns=expected_features)
    
    # 型変換
    numeric_cols = ["GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear", "InitialInterestRate", "TermInMonths", "CongressionalDistrict", "JobsSupported", "RevolverStatus"]
    for col in df_final.columns:
        if col in numeric_cols:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype(float)
        else:
            df_final[col] = df_final[col].astype(str)

    try:
        # 予測
        pool = Pool(df_final, cat_features=[i for i, col in enumerate(df_final.columns) if df_final[col].dtype == 'object'])
        proba = model.predict_proba(pool)[0][1]

        # 結果表示
        st.subheader("審査結果")
        st.metric("デフォルト確率", f"{round(proba * 100, 4)} %")

        # 💡 影響度（重要度）の表示
        st.write("### 💡 AIが注目した項目")
        importances = model.get_feature_importance()
        feat_imp_df = pd.DataFrame({'項目': expected_features, '影響度': importances}).sort_values(by='影響度', ascending=False).head(8)
        st.table(feat_imp_df.T) # 横向きに表示

        # 🔍 似た事例を10個抽出
        if not train_df.empty:
            st.write("### 📂 過去の類似事例（10件）")
            # 数値列だけで距離を計算して近い事例を探す
            train_numeric = train_df[numeric_cols].fillna(0)
            input_numeric = df_final[numeric_cols].fillna(0)
            
            nn = NearestNeighbors(n_neighbors=10)
            nn.fit(train_numeric)
            distances, indices = nn.kneighbors(input_numeric)
            
            # 抽出した10件を表示
            similar_cases = train_df.iloc[indices[0]]
            st.dataframe(similar_cases)
        else:
            st.warning("過去事例を表示するには train_data.csv を配置してください。")

    except Exception as e:
        st.error(f"エラー: {e}")
