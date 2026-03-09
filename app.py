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
        df['SBA_Ratio'] = (df['SBAGuaranteedApproval'] / df['GrossApproval']).fillna(0)
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
    rate = st.number_input("金利 (%)", 0.0, 35.0, 5.0)
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
            current_sba_ratio = sba / gross if gross > 0 else 0
            
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

            # --- B. 類似事例検索 ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 100: search_pool = train_df.copy()

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

            # --- C. リスク指数計算 (3段階ハイブリッド・ロジック) ---
            strict_proba = np.clip(raw_proba, 0.03, 0.97) 

            if gross >= 1000000:
                # 【大口】AI予測を最重視 (60%) + ペナルティ大
                risk_index = (strict_proba * 0.6) + (risk_pct / 100 * 0.4)
                penalty_factor = 12.0 
            elif gross >= 500000:
                # 【中口】AI予測と実績をバランス良く (40%) + ペナルティ中
                risk_index = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
                penalty_factor = 8.0
            else:
                # 【小口】実績統計を重視 (20%) + ペナルティ低
                risk_index = (strict_proba * 0.2) + (risk_pct / 100 * 0.8)
                penalty_factor = 5.0

            # 最終的な完済期待値の算出
            penalty = 1.0 + (risk_index * penalty_factor)
            final_expected_success = max(0.0, (1 - (risk_index * penalty)) * 100)
            
            # --- D. メイン表示 ---
            st.subheader("🏁 総合審査報告書")
            
            # --- 【新】重点確認アラート (日本語詳細版) ---
            st.write("### 🔍 実務者への重点確認事項")
            if gross >= 1000000:
                st.warning("💰 **【要確認：高額案件】** 融資申請額が $1,000,000 を超えており、学習データ内の希少事例に該当します。キャッシュフローの継続性と、万が一の際の回収シナリオを役員級で再精査してください。")
            if rate >= 20.0:
                st.error("🚨 **【要確認：高利得リスク】** 金利が 20% を超えています。AIリスク指数が低くても、逆選択（他で借りられない深刻な事情）の可能性を重点的に調査してください。")
            if current_sba_ratio >= 0.7:
                st.info("⚖️ **【要確認：保証依存】** 保証率が非常に高いです。AIはこれを『リスクが高いため保証で補完している兆候』と見ています。企業の定性面での強みを再確認してください。")

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

            # --- E. 影響度テーブル (割合固定) ---
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "SBAGuaranteedApproval": "保証率"}
            imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))

            imp_df['adj'] = imp_df['raw']
            imp_df.loc[imp_df['項目'] == 'TermInMonths', 'adj'] *= 0.23
            imp_df.loc[imp_df['項目'] == 'GrossApproval', 'adj'] *= 1.7
            imp_df.loc[imp_df['項目'] == 'SBAGuaranteedApproval', 'adj'] *= 0.8
            imp_df.loc[imp_df['項目'] == 'NaicsSector', 'adj'] *= 0.5
            imp_df.loc[imp_df['項目'] == 'InitialInterestRate', 'adj'] *= 0.9
            
            main_items = ["返済期間", "融資額", "金利", "業界", "保証率"]
            display_imp = imp_df[imp_df['項目名'].isin(main_items)].groupby('項目名')['adj'].sum().reset_index()
            total_main_adj = display_imp['adj'].sum()
            display_imp['影響度(%)'] = (display_imp['adj'] / total_main_adj * 100).round(1)
            st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

            st.divider()

            # --- F. 比較テーブル ---
            st.write("### 📂 申請データと類似事例の比較 (上位100件)")
            my_data = pd.DataFrame([{
                "結果": "📢 今回の申請", "GrossApproval": gross, 
                "保証率": current_sba_ratio, "InitialInterestRate": rate, 
                "TermInMonths": term, "CollateralInd": collateral
            }])
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
            similar_cases['保証率'] = similar_cases['SBA_Ratio']
            
            display_cols = ['結果', 'GrossApproval', '保証率', 'InitialInterestRate', 'TermInMonths', 'CollateralInd']
            comparison_df = pd.concat([my_data, similar_cases[display_cols]], ignore_index=True)
            
            comparison_df['保証率'] = (comparison_df['保証率'] * 100).map('{:.1f}%'.format)
            comparison_df['InitialInterestRate'] = comparison_df['InitialInterestRate'].map('{:.2f}'.format)
            comparison_df['GrossApproval'] = comparison_df['GrossApproval'].map('{:,.0f}'.format)
            
            comparison_df = comparison_df.rename(columns={'GrossApproval': '融資額($)', 'InitialInterestRate': '金利(%)', 'TermInMonths': '期間(月)', 'CollateralInd': '担保'})

            def highlight_rows(row):
                if row['結果'] == "📢 今回の申請": return ['background-color: #e1f5fe; font-weight: bold'] * len(row)
                elif row['結果'] == "❌ 不履行": return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)

            st.dataframe(comparison_df.style.apply(highlight_rows, axis=1), use_container_width=True, height=400)

        except Exception as e:
            st.error(f"エラー: {e}")
