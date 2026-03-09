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

            # --- C. 数値正常化ロジック (改訂版) ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            
            # 1. 動的な適正期間の算出 (融資額に連動)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            
            # 2. リスク因子の計算 (掛け算の連鎖を廃止し、最大値を抑制)
            term_risk = max(0.0, (term - dynamic_ceil) / 120.0) * 0.5 if term > dynamic_ceil else 0.0
            # 大口かつ高金利の場合のみペナルティを加算
            selection_risk = 0.2 if (gross >= 500000 and rate >= 20.0) else 0.0
            
            # 3. リスク指数の統合 (配分をマイルドに調整)
            if gross >= 1000000:
                combined_risk = (strict_proba * 0.6) + (risk_pct / 100 * 0.4)
            else:
                # 中・小口はAIの個別判断と統計を半々に
                combined_risk = (strict_proba * 0.5) + (risk_pct / 100 * 0.5)

            # 最終的な期待値計算 (急激な減衰を抑えた解析的モデル)
            # base_success は 0~1 のスコア。これにリスク因子を引き算していく
            base_success = 1.0 - combined_risk
            
            # 期待値の算出：線形減衰と最小保証（ボトム）の組み合わせ
            # リスクが 40% なら期待値は 60% 付近、そこから期間や金利のペナルティを引く
            final_expected_success = (base_success - term_risk - selection_risk) * 100
            # 下限値を 5.0% に設定し、完全にゼロにはならないように調整
            final_expected_success = max(5.0, min(98.0, final_expected_success))

            # --- D. メイン表示 ---
            st.subheader("🏁 総合審査報告書")
            
            st.write("### 🔍 審査のポイント")
            if term > dynamic_ceil:
                st.info(f"⏳ **【期間の検討】** 本案件規模での理想的な期間（{int(dynamic_ceil)}ヶ月）を超過しています。")
            if final_expected_success > 60 and (gross >= 500000 and rate >= 12.0):
                st.write("✅ 中規模案件として、金利とリスクのバランスは実務的な範囲内です。")

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
                if final_expected_success <= 40.0:
                    st.caption(":red[⚠️ 慎重な稟議が必要です]")

            # --- E. 改善アドバイス ---
            st.write("### 💡 審査改善へのアクション案")
            with st.expander("アドバイスの詳細を確認する", expanded=True):
                advice = []
                if final_expected_success < 75:
                    if term > dynamic_ceil:
                        advice.append(f"✅ **期間の最適化**: 期間を {int(dynamic_ceil)}ヶ月に近づけると、さらにスコアが改善します。")
                    if current_sba_ratio < 0.6:
                        advice.append("✅ **保証利用の強化**: 保証比率を高めることで、銀行側の実質リスクを低減可能です。")
                
                if not advice:
                    st.write("✨ 構成は論理的に安定しています。")
                else:
                    for a in advice: st.write(a)

            # --- F. 構成要素と比較 ---
            st.write("### ⚖️ 判断に影響した主要要素")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "SBAGuaranteedApproval": "保証率"}
            imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
            
            # 影響度のスケーリング（表示用）
            display_imp = imp_df[imp_df['項目名'] != "その他"].groupby('項目名')['raw'].sum().reset_index()
            display_imp['影響度(%)'] = (display_imp['raw'] / display_imp['raw'].sum() * 100).round(1)
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
