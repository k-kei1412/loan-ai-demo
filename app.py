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

            # --- C. リスク指数計算 (動的適正化ロジック) ---
            strict_proba = np.clip(raw_proba, 0.02, 0.98) 
            
            # 1. 融資額に応じた動的な適正期間の設定
            # 額が大きいほど、長期間の返済を「適正」と認める
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36 # $2Mで最大120ヶ月まで許容
            dynamic_floor = 36 + (min(gross, 1000000) / 1000000) * 24 # $1Mで60ヶ月が下限
            
            if term < dynamic_floor:
                term_penalty = 1.0 + ((dynamic_floor - term) / dynamic_floor) * 0.4
            elif term > dynamic_ceil:
                term_penalty = 1.0 + ((term - dynamic_ceil) / 120.0) * 0.6
            else:
                term_penalty = 1.0

            # 2. 金利緩和と逆選択リスクの判定
            # 通常は高金利＝緩和だが、大口($500k~)で20%超なら逆にペナルティ
            base_relief = min(1.0, max(0.0, (rate - 8.0) / 12.0))
            if gross >= 500000 and rate >= 20.0:
                rate_factor = 1.2 # 逆選択ペナルティ
            else:
                rate_factor = 1.0 - (base_relief * 0.2) # 最大20%の緩和

            # 3. 総合リスクインデックスの統合
            if gross >= 1000000:
                risk_index = (strict_proba * 0.6) + (risk_pct / 100 * 0.4)
                penalty_factor = 10.0 * term_penalty * rate_factor
            else:
                risk_index = (strict_proba * 0.3) + (risk_pct / 100 * 0.7)
                penalty_factor = 7.5 * term_penalty * rate_factor

            # 完済期待値の算出 (S字カーブによる滑らかな減衰)
            # 強制引き算を廃止し、すべてを論理構成に統合
            final_risk = risk_index * penalty_factor
            final_expected_success = max(1.0, (1.0 / (1.0 + np.exp(5.0 * (final_risk - 0.5)))) * 100)

            # --- D. メイン表示 ---
            st.subheader("🏁 総合審査報告書")
            
            # --- 実務者への重点アラート ---
            st.write("### 🔍 審査のポイント")
            if term > dynamic_ceil:
                st.info(f"⏳ **【期間超過】** 本案件の規模に対して期間が長すぎます（適正上限: {int(dynamic_ceil)}ヶ月）。将来のリスクがAI予測以上に高い可能性があります。")
            if gross >= 500000 and rate >= 20.0:
                st.error("🚨 **【逆選択の疑い】** 大口案件かつ極端な高金利です。他行での否決理由など、定性面での徹底調査を推奨します。")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
                status = "安全" if final_expected_success > 75 else "注意" if final_expected_success > 45 else "危険"
                if status == "安全": st.success("総合判定: ✅ 安全")
                elif status == "注意": st.warning("総合判定: ⚠️ 注意")
                else: st.error("総合判定: 🚨 危険")
            with c2:
                st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
            with c3:
                st.metric("完済期待値 (論理評価)", f"{final_expected_success:.1f} %")
                if final_expected_success <= 45.0:
                    st.caption(":red[⚠️ 構造的リスクが高い案件です]")

            # --- E. 改善アドバイス (動的ロジック) ---
            st.write("### 💡 審査改善へのアクション案")
            with st.expander("アドバイスの詳細を確認する", expanded=True):
                advice = []
                if final_expected_success < 70:
                    if term > dynamic_ceil:
                        advice.append(f"✅ **期間の短縮**: 返済期間を **{int(dynamic_ceil)}ヶ月以下** に設定し直すことで、期待値の大幅な向上が見込めます。")
                    if gross >= 500000 and rate >= 20.0:
                        advice.append("✅ **金利・スキームの再考**: 金利設定がリスクを増幅させています。担保強化による金利の引き下げ、または分割実行を検討してください。")
                    if current_sba_ratio < 0.7:
                        advice.append("✅ **保証枠の拡大**: 保証率を75%以上に引き上げることで、銀行の実効リスクを劇的に抑えられます。")
                
                if not advice:
                    st.write("✨ 現在の構成は、金額・金利・期間のバランスが論理的に整っています。")
                else:
                    for a in advice: st.write(a)

            # --- F. 構成要素とデータ比較 (日本語化・最適化) ---
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
