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
    sectors = {
        "other": "その他サービス業", "accommodation": "宿泊・飲食サービス業",
        "administrative": "運営支援・廃棄物処理", "agriculture": "農業・林業・漁業",
        "arts": "芸術・娯楽・レクリエーション", "construction": "建設業",
        "educational": "教育サービス業", "finance": "金融業・保険業",
        "health": "医療・福祉", "information": "情報通信業",
        "management": "企業管理・持株会社", "manufacturing": "製造業",
        "mining": "採鉱・石油ガス採掘", "professional": "専門・科学・技術サービス",
        "public": "公務", "real estate": "不動産・賃貸業", "retail": "小売業",
        "transportation": "運輸業・倉庫業", "utilities": "公益事業", "wholesale": "卸売業"
    }
    for k, v in sectors.items():
        if k in text: return v
    return en_text

# --- サイドバー管理 ---
st.sidebar.header("📋 審査・解析設定")
app_mode = st.sidebar.radio("📊 表示モード切替", ["総合報告 (表面)", "高度解析 (裏面)"])

with st.sidebar:
    st.divider()
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

# 4. 共通分析ロジック
if train_df.empty:
    st.error("学習データが見つかりません。")
else:
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

    # 実務リスク指数・期待値計算
    strict_proba = np.clip(raw_proba, 0.01, 0.99)
    dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
    term_gap = max(0.0, (term - dynamic_ceil) / 100.0) * 0.7 if term > dynamic_ceil else 0.0
    sba_bonus_flag = (current_sba_ratio >= 0.80)
    
    gross_risk = 0.40 + (gross - 1000000) / 1000000 if gross >= 1000000 else ((gross - 500000) // 100000) * 0.04 if gross > 500000 else 0.0
    if sba_bonus_flag: gross_risk *= 0.5
    
    rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3 + (0.1 if rate > 20.0 else 0)
    base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
    sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
    combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
    final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

    # --- 画面描画 ---
    if app_mode == "総合報告 (表面)":
        # --- D. 表示セクション (警告・判定) ---
        st.subheader("🏁 総合審査報告書")
        st.write("### 🔍 実務者への重点確認事項")
        
        # 判定ステータスの決定
        if gross >= 1000000:
            status = "危険"
            st.error("🚨 **【最重要精査案件】** 融資額が $1M を超過。役員承認が必須。")
        elif gross >= 500000 and rate >= 20.0 and not sba_bonus_flag:
            status = "注意"
            st.error("💀 **【複合リスク】** 高額かつ高金利。80%以上の保証がないため警戒。")
        else:
            status = "安全" if final_expected_success > 92 else "注意" if final_expected_success > 75 else "危険"

        # 個別フラグの表示
        if sba_bonus_flag:
            st.success(f"🛡️ **【保全インセンティブ適用】** 保証率80%超により高額融資リスクを50%軽減。")
        if 500000 <= gross < 1000000:
            st.info(f"📂 **【中規模案件】** 50万ドル超。リスク加重適用中。")
        if term > dynamic_ceil:
            st.warning(f"⏳ **【期間超過】** 適正上限（{int(dynamic_ceil)}ヶ月）を超過。")

        # 指標の3列表示
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
            reasons = []
            if gross >= 1000000: reasons.append("・100万ドル超の高額融資")
            elif gross >= 500000: reasons.append("・50万ドル超の中規模案件")
            if rate >= 20.0: reasons.append("・20%超の高金利")
            if term > dynamic_ceil: reasons.append("・返済期間の超過")
            if sba_bonus_flag: reasons.append("・（緩和）80%超の強固な保証")

            if status == "安全": st.success("総合判定: ✅ 安全")
            elif status == "注意":
                st.warning("総合判定: ⚠️ 注意")
                for r in reasons: st.caption(f":orange[{r}]")
            else:
                st.error("総合判定: 🚨 危険 (要精査)")
                for r in reasons: st.caption(f":red[{r}]")

        with c2:
            st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
        with c3:
            st.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

        st.divider()

        # 5. 要素インパクトの表示
        st.write("### ⚖️ 判断に影響した主要要素")
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
        name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "SBAGuaranteedApproval": "保証率"}
        imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
        
        # 重み付け調整
        imp_df['adj'] = imp_df['raw']
        imp_df.loc[imp_df['項目'] == 'TermInMonths', 'adj'] *= 0.23
        imp_df.loc[imp_df['項目'] == 'GrossApproval', 'adj'] *= 1.7
        imp_df.loc[imp_df['項目'] == 'SBAGuaranteedApproval', 'adj'] *= 0.8
        imp_df.loc[imp_df['項目'] == 'NaicsSector', 'adj'] *= 0.5
        imp_df.loc[imp_df['項目'] == 'InitialInterestRate', 'adj'] *= 0.9
        
        display_imp = imp_df[imp_df['項目名'] != "その他"].groupby('項目名')['adj'].sum().reset_index()
        display_imp['影響度(%)'] = (display_imp['adj'] / display_imp['adj'].sum() * 100).round(1)
        st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

        st.divider()
        st.write("### 📂 類似事例の傾向")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
        st.dataframe(similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths']].head(10), use_container_width=True)

    else:
        # --- 高度解析 (裏面) ---
        st.header("🔬 高度数理エビデンス解析")
        
        # 1. SHAP解析 (日本語化 & 反転)
        # 1. SHAP解析 (横棒グラフ & 日本語化)
        st.write("#### ⚖️ 項目別の完済寄与度 (SHAP Bar)")
        st.caption("※ 右に伸びるほど完済にポジティブ、左に伸びるほどリスク（不履行）要因であることを示します。")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        
        # 数値を反転（完済を正にする）
        shap_values.values = -shap_values.values
        
        # 項目名の日本語マップ（既存のname_mapを利用）
        jp_feature_names = [name_map.get(n, n) for n in expected_features]
        
        # matplotlibで描画して、軸ラベルを日本語に差し替える
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 横棒グラフの描画
        # shap.plots.bar は標準で絶対値順になることが多いため、
        # 符号を維持したまま表示するために shap.abs.mean(0) 等ではなく直接指定
        shap.plots.bar(shap_values[0], show=False)
        
        # Streamlitで表示
        plt.tight_layout()
        st.pyplot(plt.gcf())
        # 2. マートン・モデル (スライダー連動)
        st.write("#### 📉 理論的倒産距離 (Merton Model)")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            vol = st.slider("想定資産ボラティリティ (%)", 10, 100, 30) / 100
            asset = float(gross) * 1.5
            t_m = float(term) / 12
            dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
            st.metric("倒産距離 (DD)", f"{dd:.2f}")
            st.metric("デフォルト確率 (EDF)", f"{stats.norm.cdf(-dd)*100:.2f} %")
        with col_m2:
            x = np.linspace(-4, 4, 100)
            y = stats.norm.pdf(x, 0, 1)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(x, y, color="gray")
            ax.fill_between(x, y, where=(x < -dd), color='red', alpha=0.5, label='Default Risk')
            ax.axvline(-dd, color='red', linestyle='--')
            ax.set_title("Asset Value Distribution vs Default Point")
            st.pyplot(fig)

        st.divider()

        # 3. What-if 分析
        st.write("#### 🧪 金利感度シミュレーション")
        sim_rates = np.linspace(5.0, 30.0, 15)
        sim_probs = [100 * (1 - model.predict_proba(Pool(input_df.assign(InitialInterestRate=r), cat_features=cat_idx))[0][1]) for r in sim_rates]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sim_rates, sim_probs, '-o', color="#0078D4")
        ax.axvline(x=rate, color='red', linestyle='--', label='Current Rate')
        ax.set_ylabel("Success Probability (%)")
        ax.set_xlabel("Interest Rate (%)")
        st.pyplot(fig)
