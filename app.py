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
    gross = st.number_input("融資額 ($)", 0, 10000000, 700000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 400000)
    rate = st.number_input("金利 (%)", 0.0, 35.0, 15.0)
    term = st.number_input("返済期間 (月)", 1, 360, 95)
    
    sector_list = sorted(train_df['NaicsSector'].unique()) if not train_df.empty else []
    sector_en = st.selectbox("産業セクター", options=sector_list)
    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    
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
                "BusinessType": "CORPORATION", "BusinessAge": "Existing or more than 2 years old",
                "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": collateral_val
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

            # --- C. 数値正常化ロジック (論理安定性とアラートの統合) ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            
            term_gap = max(0.0, (term - dynamic_ceil) / 120.0) if term > dynamic_ceil else 0.0
            high_rate_risk = 0.15 if (gross >= 500000 and rate >= 20.0) else 0.0
            
            # AI予測(40%)と統計実績(60%)をブレンド
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)

            # 🌟 論理的安定性によるリスク圧縮 (kumagai式最適化)
            stability_bonus = 1.0
            if term <= dynamic_ceil: stability_bonus *= 0.8
            if current_sba_ratio >= 0.5: stability_bonus *= 0.9
            if rate <= 15.0: stability_bonus *= 0.9
            
            combined_risk = base_risk_idx * stability_bonus + term_gap + high_rate_risk
            combined_risk = np.clip(combined_risk, 0.02, 0.98)

            # 完済期待値の算出
            final_expected_success = (1.0 - combined_risk) * 100
            if combined_risk < 0.25:
                final_expected_success += (0.25 - combined_risk) * 40 # 低リスク帯への加点
            
            final_expected_success = max(5.0, min(98.5, final_expected_success))

            # --- D. メイン表示 ---
            st.subheader("🏁 総合審査報告書")
            
            # 🚨 アラートセクション
            st.write("### 🔍 実務者への重点確認事項")
            if gross >= 1000000:
                st.warning(f"💰 **【要確認：高額案件】** 融資額 $1M 超。キャッシュフローの継続性を役員級で再精査してください。")
            elif gross >= 500000:
                st.info(f"📂 **【中規模案件】** 融資額 $500k 超。事業計画の妥当性を重点的に確認してください。")

            if rate >= 20.0:
                st.error(f"🚨 **【要確認：高利得リスク】** 金利 20% 超。逆選択の可能性を重点調査してください。")

            if term > dynamic_ceil:
                st.warning(f"⏳ **【期間超過】** 本規模の適正上限（{int(dynamic_ceil)}ヶ月）を超過。回収シナリオの再考を推奨。")

            if term <= dynamic_ceil and rate <= 18.0 and current_sba_ratio >= 0.5:
                st.success("✨ **【論理的安定】** 構成要素が論理的に安定しており、リスクが抑制されています。")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                status = "安全" if final_expected_success > 70 else "注意" if final_expected_success > 40 else "危険"
                if status == "安全": st.success("総合判定: ✅ 安全")
                elif status == "注意": st.warning("総合判定: ⚠️ 注意")
                else: st.error("総合判定: 🚨 危険")
            with c2:
                st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
            with c3:
                st.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

            # --- E. 改善アクション ---
            st.write("### 💡 審査改善へのアクション案")
            with st.expander("アドバイスの詳細を確認する", expanded=True):
                advice = []
                if term > dynamic_ceil:
                    advice.append(f"✅ **期間の最適化**: {int(dynamic_ceil)}ヶ月以下への短縮で、回収確実性が大幅に向上します。")
                if current_sba_ratio < 0.75:
                    advice.append("✅ **保証枠の拡大**: 保証率を 75% 以上に引き上げることで、銀行の実効リスクを劇的に抑えられます。")
                if not advice:
                    st.write("✨ 現在の条件は論理的に安定しています。")
                else:
                    for a in advice: st.write(a)

            # --- F. 判断に影響した主要要素 ---
            st.write("### ⚖️ 判断に影響した主要要素")
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
            st.write("### 📂 類似事例との比較 (上位100件)")
            my_data = pd.DataFrame([{"結果": "📢 今回の申請", "融資額": gross, "保証率": f"{current_sba_ratio*100:.1f}%", "金利": rate, "期間": term}])
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
            similar_cases['融資額'] = similar_cases['GrossApproval']
            similar_cases['保証率'] = (similar_cases['SBA_Ratio'] * 100).map('{:.1f}%'.format)
            similar_cases['金利'] = similar_cases['InitialInterestRate']
            similar_cases['期間'] = similar_cases['TermInMonths']
            
            comparison_df = pd.concat([my_data, similar_cases[['結果', '融資額', '保証率', '金利', '期間']]], ignore_index=True)
            st.dataframe(comparison_df.style.apply(lambda r: ['background-color: #e1f5fe' if r['結果']=="📢 今回の申請" else 'background-color: #ffcccc' if r['結果']=="❌ 不履行" else '' for _ in r], axis=1), use_container_width=True)

        except Exception as e:
            st.error(f"分析エラー: {e}")
