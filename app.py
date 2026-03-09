import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
import scipy.stats as stats
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

# 3. 業界セクター翻訳ロジック
def get_japanese_sector(en_text):
    text = str(en_text).lower()
    if "other" in text: return "その他サービス業"
    if "accommodation" in text: return "宿泊・飲食サービス業"
    if "administrative" in text: return "運営支援・廃棄物処理"
    if "agriculture" in text: return "農業・林業・漁業"
    if "arts" in text: return "芸術・娯楽・レクリエーション"
    if "construction" in text: return "建設業"
    if "educational" in text: return "教育サービス業"
    if "finance" in text: return "金融業・保険業"
    if "health" in text: return "医療・福祉"
    if "information" in text: return "情報通信業"
    if "management" in text: return "企業管理・持株会社"
    if "manufacturing" in text: return "製造業"
    if "mining" in text: return "採鉱・石油ガス採掘"
    if "professional" in text: return "専門・科学・技術サービス"
    if "public" in text: return "公務"
    if "real estate" in text or "real_estate" in text: return "不動産・賃貸業"
    if "retail" in text: return "小売業"
    if "transportation" in text: return "運輸業・倉庫業"
    if "utilities" in text: return "公益事業（電気・ガス・水道）"
    if "wholesale" in text: return "卸売業"
    return en_text

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 500000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 300000)
    rate = st.number_input("金利 (%)", 0.0, 35.0, 15.0)
    term = st.number_input("返済期間 (月)", 1, 360, 84)
    
    if not train_df.empty:
        unique_en_sectors = sorted(train_df['NaicsSector'].unique())
        display_options = [get_japanese_sector(s) for s in unique_en_sectors]
        selected_jp = st.selectbox("産業セクター", options=display_options)
        sector_en = unique_en_sectors[display_options.index(selected_jp)]
    else:
        sector_en = "Finance_insurance"
        st.selectbox("産業セクター", options=["データ未読み込み"])

    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    submit = st.button("精密クロス審査を開始")

# 4. 分析ロジック
if submit:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            current_sba_ratio = sba / gross if gross > 0 else 0
            
            # --- A. AI予測用データ作成 ---
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

            # --- B. 類似事例検索 (k-NN) ---
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

            # --- C. 実務評価ロジック ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (term - dynamic_ceil) / 100.0) * 0.7 if term > dynamic_ceil else 0.0
            gross_risk = 0.0
            sba_bonus_flag = False
            if gross >= 1000000: gross_risk = 0.40 + (gross - 1000000) / 1000000
            elif gross > 500000: gross_risk = ((gross - 500000) // 100000) * 0.04
            if current_sba_ratio >= 0.80:
                gross_risk *= 0.5
                sba_bonus_flag = True
            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3
            if rate > 20.0: rate_risk += 0.1
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            stability_bonus = 1.0
            if term <= dynamic_ceil: stability_bonus *= 0.8
            if rate <= 15.0: stability_bonus *= 0.9
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            combined_risk = (base_risk_idx * stability_bonus * sba_offset) + term_gap + gross_risk + rate_risk
            combined_risk = np.clip(combined_risk, 0.02, 0.99)
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            # ==========================================
            # 5. タブによる画面分割 (表面/裏面)
            # ==========================================
            tab1, tab2 = st.tabs(["📄 総合審査報告書 (表面)", "🔬 高度数理解析 (裏面)"])

            with tab1:
                # --- D. 表示セクション (警告・判定) ---
                st.subheader("🏁 総合審査報告書")
                st.write("### 🔍 実務者への重点確認事項")
                
                if gross >= 1000000:
                    status = "危険"
                    st.error("🚨 **【最重要精査案件】** 融資額が $1M を超過。役員承認が必須。")
                elif gross >= 500000 and rate >= 20.0 and not sba_bonus_flag:
                    status = "注意"
                    st.error("💀 **【複合リスク】** 高額かつ高金利。80%以上の保証がないため警戒。")
                else:
                    status = "安全" if final_expected_success > 92 else "注意" if final_expected_success > 75 else "危険"

                if sba_bonus_flag:
                    st.success(f"🛡️ **【保全インセンティブ適用】** 保証率80%超により高額融資リスクを50%軽減。")
                if 500000 <= gross < 1000000:
                    st.info(f"📂 **【中規模案件】** 50万ドル超。リスク加重適用中。")
                if term > dynamic_ceil:
                    st.warning(f"⏳ **【期間超過】** 適正上限（{int(dynamic_ceil)}ヶ月）を超過。")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                    if status == "安全": st.success("総合判定: ✅ 安全")
                    elif status == "注意": st.warning("総合判定: ⚠️ 注意")
                    else: st.error("総合判定: 🚨 危険 (要精査)")

                with c2:
                    st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                    st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
                with c3:
                    st.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

                # --- E. 改善アクション ---
                st.write("### 💡 審査改善へのアクション案")
                with st.expander("アドバイスの詳細を確認する", expanded=True):
                    advice = []
                    if gross >= 1000000: advice.append("⚠️ **金額の再検討**: 可能であれば分割融資または担保の積み増しを。")
                    if term > dynamic_ceil: advice.append(f"✅ **期間の最適化**: {int(dynamic_ceil)}ヶ月以下への短縮を推奨。")
                    if current_sba_ratio < 0.80: advice.append("✅ **保証枠の拡大**: 80%以上に引き上げるとリスク加重が半減します。")
                    if not advice: st.write("✨ 現在の条件は論理的に非常に安定しています。")
                    else:
                        for a in advice: st.write(a)

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

            with tab2:
                st.header("🔬 高度数理エビデンス解析")
                
                # --- 1. SHAP解析 (解釈可能なAI) ---
                st.subheader("⚖️ 判定の数学的根拠 (SHAP)")
                st.write("AIが今回「なぜこの判定を出したか」を、項目ごとにプラス・マイナスで分解します。")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                st_shap(shap.plots.waterfall(shap_values[0]), height=400)
                

                st.divider()

                # --- 2. マートン・モデル (理論的デフォルトリスク) ---
                st.subheader("📉 理論的倒産距離 (Merton Model)")
                st.write("ゼミの研究に基づくオプション理論的なアプローチで、企業の財務的余裕を算出します。")
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    vol = st.slider("想定資産ボラティリティ (%)", 10, 100, 30) / 100
                    asset_val = float(gross) * st.slider("推定資産価値（融資額比）", 1.0, 5.0, 1.5)
                
                # DD計算ロジック
                r_m = float(rate) / 100
                k_m = float(gross)
                t_m = float(term) / 12
                dd = (np.log(asset_val / k_m) + (r_m - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
                theoretical_pd = stats.norm.cdf(-dd)

                with m_col2:
                    st.metric("倒産距離 (Distance to Default)", f"{dd:.2f}")
                    st.metric("理論的デフォルト確率 (EDF)", f"{theoretical_pd*100:.2f} %")
                

                st.divider()

                # --- 3. What-if 分析 (感度分析) ---
                st.subheader("🧪 条件変更シミュレーション (What-if)")
                st.write("金利が変動した際に、AIの予測する完済期待値がどう変化するかをシミュレーションします。")
                
                sim_rates = np.linspace(1.0, 35.0, 20)
                sim_success = []
                for sr in sim_rates:
                    sim_df = input_df.copy()
                    sim_df['InitialInterestRate'] = float(sr)
                    # CatBoostの生確率を取得
                    p = model.predict_proba(Pool(sim_df, cat_features=cat_idx))[0][1]
                    sim_success.append((1 - p) * 100)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(sim_rates, sim_success, marker='o', color='#0078D4', label="完済期待値")
                ax.axvline(x=float(rate), color='red', linestyle='--', label='現在の設定')
                ax.set_xlabel("金利 (%)")
                ax.set_ylabel("期待値 (%)")
                ax.legend()
                st.pyplot(fig)
                

                st.success("✅ 統計的推論(k-NN)、機械学習(CatBoost)、数理ファイナンス(Merton)の全工程が完了しました。")

        except Exception as e:
            st.error(f"分析エラー: {e}")
