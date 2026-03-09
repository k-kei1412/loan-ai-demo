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

# 3. 業界セクター翻訳
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

# 4. メイン分析ロジック
if submit:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            # 準備計算
            current_sba_ratio = sba / gross if gross > 0 else 0
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
            
            # AI予測
            raw_proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

            # 類似事例検索 (k-NN)
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

            # リスク指数計算
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (term - dynamic_ceil) / 100.0) * 0.7 if term > dynamic_ceil else 0.0
            gross_risk = 0.0
            sba_bonus_flag = (current_sba_ratio >= 0.80)
            if gross >= 1000000: gross_risk = 0.40 + (gross - 1000000) / 1000000
            elif gross > 500000: gross_risk = ((gross - 500000) // 100000) * 0.04
            if sba_bonus_flag: gross_risk *= 0.5
            
            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3
            if rate > 20.0: rate_risk += 0.1
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            # --- ここからタブ表示 ---
            st.success("✅ 解析が完了しました。以下のタブで詳細を確認してください。")
            tab1, tab2 = st.tabs(["📄 総合審査報告書 (表面)", "🔬 高度数理解析 (裏面)"])

            with tab1:
                st.subheader("🏁 総合判定結果")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                    if final_expected_success > 90: st.success("判定: ✅ 安全")
                    elif final_expected_success > 70: st.warning("判定: ⚠️ 注意")
                    else: st.error("判定: 🚨 危険")
                with c2:
                    st.metric("類似事例事故率", f"{risk_pct:.1f} %")
                with c3:
                    st.metric("完済期待値", f"{final_expected_success:.1f} %")

                # --- 5. 要素インパクトの表示（再統合） ---
                st.write("### ⚖️ 判断に影響した主要要素")
                importances = model.get_feature_importance()
                imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
                name_map = {
                    "TermInMonths": "返済期間", 
                    "GrossApproval": "融資額", 
                    "InitialInterestRate": "金利", 
                    "NaicsSector": "業界", 
                    "SBAGuaranteedApproval": "保証率"
                }
                imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
                
                # ご提示の重み付けロジック
                imp_df['adj'] = imp_df['raw']
                imp_df.loc[imp_df['項目'] == 'TermInMonths', 'adj'] *= 0.23
                imp_df.loc[imp_df['項目'] == 'GrossApproval', 'adj'] *= 1.7
                imp_df.loc[imp_df['項目'] == 'SBAGuaranteedApproval', 'adj'] *= 0.8
                imp_df.loc[imp_df['項目'] == 'NaicsSector', 'adj'] *= 0.5
                imp_df.loc[imp_df['項目'] == 'InitialInterestRate', 'adj'] *= 0.9
                
                display_imp = imp_df[imp_df['項目名'] != "その他"].groupby('項目名')['adj'].sum().reset_index()
                total_adj = display_imp['adj'].sum()
                display_imp['影響度(%)'] = (display_imp['adj'] / total_adj * 100).round(1)
                
                st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

                st.divider()
                st.write("### 📂 類似100件の明細")
                # (以下、類似事例の表示コード...)

                st.write("### 📂 類似100件の明細")
                similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
                st.dataframe(similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths']].head(10), use_container_width=True)

            with tab2:
                st.subheader("🔬 数理エビデンス・ダッシュボード")
                
                # 1. SHAP解析
                st.write("#### ⚖️ AIの判断根拠 (SHAP Waterfall)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                st_shap(shap.plots.waterfall(shap_values[0]), height=350)
                

                st.divider()

                # 2. マートン・モデル
                st.write("#### 📉 理論的倒産距離 (Merton Model)")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    vol = st.slider("想定ボラティリティ", 0.1, 1.0, 0.3)
                    asset = float(gross) * 1.5
                    t_m = float(term) / 12
                    dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
                    st.metric("倒産距離 (DD)", f"{dd:.2f}")
                with col_m2:
                    st.write("企業の資産価値が負債（融資額）を下回るまでの距離を標準偏差単位で測定しています。")
                

                st.divider()

                # 3. What-if 分析
                st.write("#### 🧪 金利感度シミュレーション")
                sim_rates = np.linspace(5.0, 30.0, 15)
                sim_probs = [100 * (1 - model.predict_proba(Pool(input_df.assign(InitialInterestRate=r), cat_features=cat_idx))[0][1]) for r in sim_rates]
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(sim_rates, sim_probs, '-o')
                ax.set_title("Interest Rate vs Success Probability")
                st.pyplot(fig)
                

        except Exception as e:
            st.error(f"実行エラー: {e}")
            st.info("モデルファイルやCSVファイルが正しいパスにあるか確認してください。")
