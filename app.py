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
    rate = st.number_input("金利 (%)", 0.0, 35.0, 5.0) # 上限を少し拡張
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

            # --- C. リスク指数計算 ---
            strict_proba = np.clip(raw_proba, 0.03, 0.97) 
            risk_index = (strict_proba * 0.2) + (risk_pct / 100 * 0.8)
            penalty = 1.0 + (risk_index * 7.0)
            final_expected_success = max(0.0, (1 - (risk_index * penalty)) * 100)

            # --- D. メイン表示 ---
            st.subheader("🏁 総合審査報告書")
            
            # 特例警告アラート（実務的な要確認事項）
            if rate >= 20.0:
                st.error(f"🚨 **【高利得警告】** 金利が {rate}% と極めて高く設定されています。統計上は『安全』であっても、債務者の支払い能力を超えているリスク、または逆選択（他で借りられない事情）を重点的に調査してください。")
            if gross >= 5000000:
                st.warning(f"💰 **【巨額融資アラート】** 融資額が ${gross:,} に達しています。当行のポートフォリオに与える影響が大きいため、経営陣による二次審査を推奨します。")

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

            # --- E. 専門的アドバイス ---
            st.write("### 📝 AI専門アドバイス")
            if rate >= 20.0 and status == "安全":
                st.info("この案件は『ハイリスク・ハイリターン』の典型です。AIは高い収益性がリスクをカバーすると見ていますが、担保の流動性を再確認してください。")
            elif status == "安全":
                st.success("低リスクかつ標準的な案件です。過去の類似事例でも高い完済率を誇っており、迅速な承認手続きが推奨されます。")
            elif status == "注意":
                st.warning("デフォルトの兆候が一部の類似事例で見られます。返済期間の短縮、または保証率の引き上げを条件とした承認を検討してください。")
            else:
                st.error("不履行実績が非常に高いゾーンです。現条件での融資は極めて危険であり、事業計画の抜本的な見直しが必要です。")

            st.divider()

            # --- F. 影響度テーブル ---
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

            main_items = ["返済期間", "融資額", "金利", "業界", "保証率"]
            display_imp = imp_df[imp_df['項目名'].isin(main_items)].groupby('項目名')['adj'].sum().reset_index()
            total_main_adj = display_imp['adj'].sum()
            display_imp['影響度(%)'] = (display_imp['adj'] / total_main_adj * 100).round(1)
            st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

            # --- G. 比較テーブル ---
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
            
            # フォーマット調整
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
