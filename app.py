import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：真・完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 2. リソース読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        target = "train.csv" if os.path.exists("train.csv") else "train (4).csv"
        df = pd.read_csv(target)
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        df['SBA_Ratio'] = df['SBAGuaranteedApproval'] / df['GrossApproval']
        return model, df, target
    except:
        return model, pd.DataFrame(), "None"

model, train_df, file_name = load_resources()
expected_features = model.feature_names_

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    
    sector_list = sorted(train_df['NaicsSector'].unique()) if not train_df.empty else []
    sector_en = st.selectbox("産業セクター", options=sector_list)
    
    business_age = st.selectbox("企業年齢", ["Existing or more than 2 years old", "New Business or 2 years or less", "Startup, Loan Funds will Open Business", "Unanswered"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            # --- 0. 自身の設定値を固定表示 ---
            current_sba_ratio = sba / gross if gross > 0 else 0
            st.info(f"📍 **現在の申請条件:** 融資 ${gross:,} / 保証 ${sba:,} ({current_sba_ratio*100:.1f}%) / 金利 {rate}% / 期間 {term}ヶ月 / セクター: {sector_en}")

            # --- A. AI予測 ---
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
            raw_proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

            # --- B. 類似事例検索 (保証率重視) ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 100:
                search_pool = train_df.copy()

            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
            train_num = search_pool[search_features].fillna(0)
            input_num = input_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]].copy()
            input_num["SBA_Ratio"] = current_sba_ratio
            
            scaler = StandardScaler()
            weights = np.array([1.2, 1.0, 1.0, 2.0]) 
            train_scaled = scaler.fit_transform(train_num) * weights
            input_scaled = scaler.transform(input_num) * weights

            nn = NearestNeighbors(n_neighbors=min(100, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

            # --- C. リスク指数・期待値計算 ---
            strict_proba = np.clip(raw_proba, 0.03, 0.97) 
            risk_index = (strict_proba * 0.3) + (risk_pct / 100 * 0.7)
            penalty = 1.0 + (risk_index * 7.0)
            final_expected_success = max(0.0, (1 - (risk_index * penalty)) * 100)

            # --- D. 画面表示：メインメトリクス ---
            st.subheader("🏁 総合審査報告書")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
                status = "安全" if risk_index < 0.07 else "注意" if risk_index < 0.17 else "危険"
                if status == "安全": st.success("総合判定: ✅ 安全")
                elif status == "注意": st.warning("総合判定: ⚠️ 注意")
                else: st.error("総合判定: 🚨 危険")
            with c2:
                st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
            with c3:
                st.metric("完済期待値 (保守的評価)", f"{final_expected_success:.1f} %")
                st.caption("※デフォルトの重みを反映済み")

            st.divider()

            # --- E. 【順序変更】判断の主要構成要素 (%) ---
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "SBAGuaranteedApproval": "保証率", "CollateralInd": "担保"}
            imp_df['項目'] = imp_df['項目'].map(lambda x: name_map.get(x, x))
            imp_df['adj'] = imp_df['raw']
            imp_df.loc[imp_df['項目'] == '返済期間', 'adj'] *= 0.7
            imp_df.loc[imp_df['項目'] == '融資額', 'adj'] *= 3.9
            imp_df.loc[imp_df['項目'] == '金利', 'adj'] *= 2.4
            imp_df.loc[imp_df['項目'] == '保証率', 'adj'] *= 2.4
            total_adj = imp_df['adj'].sum()
            imp_df['影響度(%)'] = (imp_df['adj'] / total_adj * 100).round(1)
            st.table(imp_df.groupby('項目')['影響度(%)'].sum().sort_values(ascending=False).head(5))

            # --- F. 【順序変更】比較テーブル (自身をトップに) ---
            st.write("### 📂 申請データと類似事例の比較 (上位100件)")
            my_data = pd.DataFrame([{
                "結果": "📢 今回の申請", "GrossApproval": gross, "SBAGuaranteedApproval": sba,
                "SBA_Ratio": current_sba_ratio, "InitialInterestRate": rate, "TermInMonths": term, "CollateralInd": collateral
            }])
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
            display_cols = ['結果', 'GrossApproval', 'SBAGuaranteedApproval', 'SBA_Ratio', 'InitialInterestRate', 'TermInMonths', 'CollateralInd']
            comparison_df = pd.concat([my_data, similar_cases[display_cols]], ignore_index=True)
            comparison_df['保証率'] = (comparison_df['SBA_Ratio'] * 100).map('{:.1f}%'.format)
            
            def highlight_rows(row):
                if row['結果'] == "📢 今回の申請": return ['background-color: #e1f5fe; font-weight: bold'] * len(row)
                elif row['結果'] == "❌ 不履行": return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)

            st.dataframe(comparison_df.drop(columns=['SBA_Ratio']).style.apply(highlight_rows, axis=1), use_container_width=True, height=400)

        except Exception as e:
            st.error(f"エラー: {e}")
