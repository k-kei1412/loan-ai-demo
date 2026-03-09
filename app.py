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
    
    # 【修正】企業年齢を削除し、担保のみに
    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    
    submit = st.button("精密クロス審査を開始")

if submit:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            current_sba_ratio = sba / gross if gross > 0 else 0
            
            # --- A. AI予測 (企業年齢を削除して実行) ---
            input_data = {
                "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
                "InitialInterestRate": float(rate), "TermInMonths": float(term),
                "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "Guaranty",
                "FixedOrVariableInterestInd": "V", "CongressionalDistrict": 10.0,
                "BusinessType": "CORPORATION", 
                "BusinessAge": "Existing or more than 2 years old", # 内部的にはデフォルト値を設定
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

            # --- C. リスク指数計算 (適正ゾーン・期間逆転補正) ---
            strict_proba = np.clip(raw_proba, 0.03, 0.97) 
            rate_relief = min(1.0, max(0.0, (rate - 10.0) / 10.0)) 

            # 【追加】適正ゾーン(5-7年)ロジック
            ZONE_FLOOR = 60
            ZONE_CEIL = 84
            if term < ZONE_FLOOR:
                term_penalty_weight = 1.0 + ((ZONE_FLOOR - term) / ZONE_FLOOR) * 0.5
            elif term > ZONE_CEIL:
                term_penalty_weight = max(1.0, (term / ZONE_CEIL) ** 1.3) # 長期ほど厳格に
            else:
                term_penalty_weight = 1.0

            if gross >= 1000000:
                risk_index = (strict_proba * 0.6) + (risk_pct / 100 * 0.4)
                penalty_factor = max(9.0, (11.0 - rate_relief) * term_penalty_weight)
            elif gross >= 500000:
                risk_index = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
                penalty_factor = max(8.0, (9.5 - rate_relief) * term_penalty_weight)
            else:
                risk_index = (strict_proba * 0.2) + (risk_pct / 100 * 0.8)
                penalty_factor = 6.5 * term_penalty_weight

            penalty = 1.0 + (risk_index * penalty_factor)
            raw_success = (1 - (risk_index * penalty)) * 100
            
            warning_count = 0
            if gross >= 1000000: warning_count += 1
            if rate >= 20.0: warning_count += 1
            if current_sba_ratio >= 0.7: warning_count += 1
            
            final_expected_success = max(1.0, raw_success - (warning_count * 10.0))

            # --- D. メイン表示 ---
            st.subheader("🏁 総合審査報告書")
            
            # --- 重点確認アラート ---
            st.write("### 🔍 実務者への重点確認事項")
            col_alerts = st.columns(1)
            with col_alerts[0]:
                if gross >= 1000000:
                    st.warning("💰 **【高額案件】** 融資額が $1M 超。キャッシュフローと回収シナリオの再精査が必要です。")
                if rate >= 20.0:
                    st.error("🚨 **【高利得リスク】** 金利 20% 超。逆選択の可能性を重点的に調査してください。")
                if term > ZONE_CEIL:
                    st.info(f"⏳ **【長期融資リスク】** 返済期間が適正ゾーン(7年)を超えています。将来の不確実性を考慮してください。")
                if term < ZONE_FLOOR:
                    st.info(f"🕒 **【短期返済リスク】** 期間が5年未満です。月々の返済負荷によるキャッシュフロー圧迫を要確認。")

            # --- メトリクス表示 ---
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
                if final_expected_success <= 30.0:
                    st.metric("完済期待値", "30% 未満")
                    st.caption(":red[⚠️ 要・個別審査案件]")
                else:
                    st.metric("完済期待値", f"{final_expected_success:.1f} %")

            # --- 【新】AI実務アドバイス ---
            st.write("### 💡 審査改善へのアドバイス")
            with st.expander("アドバイスを表示する", expanded=True):
                advice_list = []
                if final_expected_success < 50.0:
                    if term > ZONE_CEIL:
                        advice_list.append(f"🚩 **期間の短縮**: 現在の{term}ヶ月から**84ヶ月(7年)以下**に短縮することで、長期不確実性リスクを低減し、期待値を向上させることが可能です。")
                    if term < ZONE_FLOOR:
                        advice_list.append(f"🚩 **期間の延長**: 現在の{term}ヶ月から**60ヶ月(5年)**程度まで延ばすことで、月々の返済負担を緩和し、デフォルト率を下げられる可能性があります。")
                    if current_sba_ratio < 0.5:
                        advice_list.append("🚩 **保証の強化**: 保証額を増やし保証率を50%以上に引き上げることで、銀行側の実効リスク指数を抑制できます。")
                
                if not advice_list:
                    st.write("✅ 現在の条件は論理的に安定しています。")
                else:
                    for a in advice_list:
                        st.write(a)

            # --- E. 影響度テーブル ---
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "SBAGuaranteedApproval": "保証率"}
            imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
            
            # 影響度の調整（表示用）
            imp_df['adj'] = imp_df['raw']
            imp_df.loc[imp_df['項目'] == 'TermInMonths', 'adj'] *= 0.23
            imp_df.loc[imp_df['項目'] == 'GrossApproval', 'adj'] *= 1.7
            
            display_imp = imp_df[imp_df['項目名'] != "その他"].groupby('項目名')['adj'].sum().reset_index()
            total = display_imp['adj'].sum()
            display_imp['影響度(%)'] = (display_imp['adj'] / total * 100).round(1)
            st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

            st.divider()

            # --- F. 比較テーブル (日本語化) ---
            st.write("### 📂 申請データと類似事例の比較 (上位100件)")
            my_data = pd.DataFrame([{
                "結果": "📢 今回の申請", "GrossApproval": gross, 
                "保証率": current_sba_ratio, "InitialInterestRate": rate, 
                "TermInMonths": term, "CollateralInd": collateral_val
            }])
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
            similar_cases['保証率'] = similar_cases['SBA_Ratio']
            
            display_cols = ['結果', 'GrossApproval', '保証率', 'InitialInterestRate', 'TermInMonths', 'CollateralInd']
            comparison_df = pd.concat([my_data, similar_cases[display_cols]], ignore_index=True)
            
            comparison_df['保証率'] = (comparison_df['保証率'] * 100).map('{:.1f}%'.format)
            comparison_df['InitialInterestRate'] = comparison_df['InitialInterestRate'].map('{:.2f}'.format)
            comparison_df['GrossApproval'] = comparison_df['GrossApproval'].map('{:,.0f}'.format)
            
            comparison_df = comparison_df.rename(columns={
                'GrossApproval': '融資額($)', 'InitialInterestRate': '金利(%)', 
                'TermInMonths': '期間(月)', 'CollateralInd': '担保'
            })

            def highlight_rows(row):
                if row['結果'] == "📢 今回の申請": return ['background-color: #e1f5fe; font-weight: bold'] * len(row)
                elif row['結果'] == "❌ 不履行": return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)

            st.dataframe(comparison_df.style.apply(highlight_rows, axis=1), use_container_width=True, height=400)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
