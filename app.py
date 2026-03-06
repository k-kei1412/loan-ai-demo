import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：完全版", layout="wide")
st.title("🏦 ローンデフォルト予測 AIシステム")
st.markdown("AIの予測確率と、過去の類似事例10件の実績値を比較して審査を行います。")

# 2. リソースの読み込み
@st.cache_resource
def load_resources():
    # モデルの読み込み
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    
    # 過去データの読み込み（事例検索用）
    try:
        # train_data.csv が存在することを前提としています
        train_df = pd.read_csv("train.csv")
    except Exception as e:
        st.error(f"警告: train_data.csv が見つかりません。事例表示機能は無効です。 ({e})")
        train_df = pd.DataFrame()
    
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# 3. 入力フォーム
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    year = st.number_input("承認年度", 1990, 2026, 2010)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    jobs = st.number_input("雇用人数", 0, 1000, 5)
    subprogram = st.text_input("ローンプログラム", "7(a)")
    rate_type = st.selectbox("金利タイプ", ["Fixed", "Variable"])
    sector = st.number_input("産業セクター (NAICS)", 1, 99, 44)
    district = st.number_input("地区コード", 1, 60, 10)
    business_type = st.text_input("企業形態", "CORPORATION")
    business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
    revolver = st.selectbox("リボルビングローン", ["Y", "N"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    
    submit = st.button("AI審査を開始")

# 4. 審査ロジック
if submit:
    # データ整形
    revolver_val = 1.0 if revolver == "Y" else 0.0
    raw_input = {
        "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year), "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term), "NaicsSector": str(int(sector)), 
        "CongressionalDistrict": float(district), "BusinessType": str(business_type), 
        "BusinessAge": str(business_age), "RevolverStatus": float(revolver_val),
        "JobsSupported": float(jobs), "CollateralInd": str(collateral)
    }
    input_df = pd.DataFrame([raw_input]).reindex(columns=expected_features)

    # 型の強制
    numeric_cols = ["GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear", 
                    "InitialInterestRate", "TermInMonths", "CongressionalDistrict", 
                    "JobsSupported", "RevolverStatus"]
    
    for col in input_df.columns:
        if col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
        else:
            input_df[col] = input_df[col].astype(str)

    try:
        # AI予測の実行
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        pool = Pool(input_df, cat_features=cat_idx)
        proba = model.predict_proba(pool)[0][1]

        # --- 表示セクション1: メイン指標 ---
        st.subheader("🏁 総合審査結果")
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.metric("AI予測デフォルト確率", f"{proba * 100:.4f} %")
            if proba > 0.01: # 閾値を厳しめの1%に設定
                st.error("AI判定: ⚠️ 高リスク可能性あり")
            else:
                st.success("AI判定: ✅ 低リスク（モデル予測値）")

        # --- 表示セクション2: 類似事例の統計 ---
        if not train_df.empty:
            # 距離計算（数値ベース）
            num_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "JobsSupported"]
            train_num = train_df[num_features].fillna(0)
            input_num = input_df[num_features].fillna(0)
            
            nn = NearestNeighbors(n_neighbors=10)
            nn.fit(train_num)
            distances, indices = nn.kneighbors(input_num)
            
            similar_cases = train_df.iloc[indices[0]].copy()
            
            # 統計値の算出（'Default'列がある想定）
            if 'Default' in similar_cases.columns:
                actual_risk_count = similar_cases['Default'].sum()
                actual_risk_pct = (actual_risk_count / 10) * 100
                
                with m_col2:
                    st.metric("類似事例の実績デフォルト率", f"{actual_risk_pct:.1f} %")
                    if actual_risk_pct >= 20.0:
                        st.error(f"過去実績判定: 🚨 危険 (10件中{int(actual_risk_count)}件が失敗)")
                    elif actual_risk_pct > 0:
                        st.warning(f"過去実績判定: ⚠️ 要注意 (10件中{int(actual_risk_count)}件が失敗)")
                    else:
                        st.success("過去実績判定: ✨ 非常に安全 (10件中0件が失敗)")

        # --- 表示セクション3: 影響度 ---
        st.divider()
        st.subheader("💡 AIが注目した判断材料")
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({'項目': expected_features, '影響度': importances}).sort_values('影響度', ascending=False).head(10)
        st.table(imp_df.T)

        # --- 表示セクション4: 類似事例の詳細リスト ---
        if not train_df.empty:
            st.subheader("📂 比較対象とした類似事例の詳細")
            if 'Default' in similar_cases.columns:
                # 見やすくするためにラベル付け
                similar_cases['結果'] = similar_cases['Default'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
            
            st.dataframe(similar_cases, use_container_width=True)

    except Exception as e:
        st.error(f"予測エラー: {e}")
