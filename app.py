import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：究極完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 2. リソースの読み込み (train.csvを直接使用)
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        # ユーザー提供の train.csv を読み込み
        df = pd.read_csv("train.csv")
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        return model, df
    except Exception as e:
        st.error(f"学習データの読み込みに失敗しました: {e}")
        return model, pd.DataFrame()

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    
    # セクター一覧 (train.csv内のユニーク値)
    sector_list = sorted(train_df['NaicsSector'].unique()) if not train_df.empty else []
    sector_en = st.selectbox("産業セクター", options=sector_list)
    
    business_age = st.selectbox("企業年齢", ["Existing or more than 2 years old", "New Business or 2 years or less", "Startup, Loan Funds will Open Business", "Unanswered"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    if train_df.empty:
        st.error("学習データが読み込めていません。")
    else:
        try:
            # --- A. AI予測 (2000件超の全体傾向) ---
            input_data = {
                "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
                "InitialInterestRate": float(rate), "TermInMonths": float(term),
                "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "Guaranty",
                "FixedOrVariableInterestInd": "V", "CongressionalDistrict": 10.0,
                "BusinessType": "CORPORATION", "BusinessAge": business_age, 
                "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": collateral
            }
            input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
            cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
            proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

            # --- B. 類似事例検索 (一番精度が良かった「近傍検索」) ---
            # 1. 業界で絞り込み
            sector_df = train_df[train_df['NaicsSector'] == sector_en].copy()
            # もし同じ業界のデータが少なければ全データから探す
            search_pool = sector_df if len(sector_df) >= 50 else train_df.copy()

            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].fillna(0)
            input_num = input_df[search_features].fillna(0)
            
            # 検索重み (実務重視)
            weights = np.array([1.5, 1.0, 1.0]) 
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num) * weights
            input_scaled = scaler.transform(input_num) * weights

            nn = NearestNeighbors(n_neighbors=min(50, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

            # --- C. 実効リスク指数の計算 (実績重視 70%) ---
            # どんなに安全でも 0.5% のリスクを含ませる厳格設定
            strict_proba = np.clip(proba, 0.005, 0.995)
            strict_risk_pct = (def_count + 0.2) / (50 + 0.4) 
            risk_index = (strict_proba * 0.3) + (strict_risk_pct * 0.7)

            # --- D. 表示 ---
            st.subheader("🏁 総合審査報告書")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
                if risk_index < 0.05: st.success("総合判定: ✅ 安全")
                elif risk_index < 0.15: st.warning("総合判定: ⚠️ 注意")
                else: st.error("総合判定: 🚨 危険")
            with c2:
                st.metric("実績事故率 (類似50件)", f"{risk_pct:.1f} %")
                st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
            with c3:
                st.metric("AI完済確信度", f"{min((1-strict_proba)*100, 99.5):.1f} %")

            st.divider()

            # --- E. 影響度テーブル ---
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, '値': importances})
            # 日本語化して表示
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "CollateralInd": "担保"}
            imp_df['項目'] = imp_df['項目'].map(lambda x: name_map.get(x, x))
            imp_df = imp_df.groupby('項目').sum().sort_values('値', ascending=False).head(5)
            imp_df['影響度(%)'] = (imp_df['値'] / imp_df['値'].sum() * 100).round(1)
            st.table(imp_df[['影響度(%)']])

            # --- F. 事例詳細 ---
            st.write("### 📂 属性が近い類似事例 (赤色はデフォルト事故)")
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
            st.dataframe(similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector']].style.apply(
                lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
            ), use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
