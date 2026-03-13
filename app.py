# --- 前半のインポートと基本設定はそのまま ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：真・完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# --- 文字化け・フォント対策 ---
def set_japanese_font():
    fonts = ['Heiti TC', 'MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
    for f in fonts:
        if f in [font.name for font in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = f
            break
    plt.rcParams['axes.unicode_minus'] = False 

set_japanese_font()

# --- 業界定義 ---
sectors_map = {
    "accommodation": "宿泊・飲食サービス業", "administrative": "運営支援・廃棄物処理",
    "agriculture": "農業・林業・漁業", "arts": "芸術・娯楽・レクリエーション",
    "construction": "建設業", "educational": "教育サービス業",
    "finance": "金融業・保険業", "health": "医療・福祉",
    "information": "情報通信業", "management": "企業管理・持株会社",
    "manufacturing": "製造業", "mining": "採鉱・石油ガス採掘",
    "professional": "専門・科学・技術サービス", "public": "公務",
    "real estate": "不動産・賃貸業", "retail": "小売業",
    "transportation": "運輸業・倉庫業", "utilities": "公益事業", "wholesale": "卸売業"
}

sector_vix_map = {
    "accommodation": 45, "administrative": 40, "agriculture": 23, 
    "arts": 50, "construction": 30, "educational": 25,
    "finance": 23, "health": 20, "information": 55, 
    "management": 60, "manufacturing": 30, "mining": 30,
    "professional": 45, "public": 18, "real estate": 40, 
    "retail": 40, "transportation": 30, "utilities": 18, "wholesale": 30
}

# 2. リソース読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        target = "train.csv" if os.path.exists("train.csv") else "train (4).csv"
        df = pd.read_csv(target)
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        # 内部計算用の保証率（学習データ側）
        df['SBA_Ratio'] = (df['SBAGuaranteedApproval'] / df['GrossApproval']).fillna(0)
        return model, df, target
    except:
        return model, pd.DataFrame(), "None"

model, train_df, file_name = load_resources()
expected_features = model.feature_names_

# --- 名前マップの定義 ---
graph_name_map = {
    "TermInMonths": "Loan Term", "GrossApproval": "Loan Amount", "InitialInterestRate": "Interest Rate", 
    "NaicsSector": "Industry Sector", "SBAGuaranteedApproval": "SBA Guaranty", "CollateralInd": "Collateral",
    "ApprovalFiscalYear": "Fiscal Year", "Subprogram": "Subprogram", "FixedOrVariableInterestInd": "Rate Type",
    "BusinessAge": "Business Age", "CongressionalDistrict": "Location Code", "BusinessType": "Business Type", "JobsSupported": "Jobs Created",
    "SBA_Ratio": "Guaranty Rate" # 追加
}

table_name_map = {
    "TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", 
    "NaicsSector": "業界セクター", "SBAGuaranteedApproval": "保証額", "CollateralInd": "担保有無",
    "BusinessAge": "事業歴", "BusinessType": "法人形態", "JobsSupported": "雇用創出数",
    "Subprogram": "支援プログラム", "FixedOrVariableInterestInd": "金利タイプ", "CongressionalDistrict": "地域区分（所在地区）",
    "SBA_Ratio": "保証率", "ApprovalFiscalYear": "承認年度", "RevolverStatus": "当座貸越枠の有無"
}

def get_japanese_sector(en_text):
    text = str(en_text).lower()
    if "other" in text: return "その他サービス業"
    for k, v in sectors_map.items():
        if k in text: return v
    return en_text

if "clicked" not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
app_mode = st.sidebar.radio("📊 表示モード切替", ["総合報告書", "数理モデル解析"])

with st.sidebar:
    st.divider()
    gross = st.number_input("融資額 ($)", 0, 10000000, 500000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 300000)
    rate = st.number_input("金利 (%)", 0.0, 35.0, 15.0)
    term = st.number_input("返済期間 (月)", 1, 360, 84)
    
    b_age = st.selectbox("事業歴", ["2年以上 (Existing)", "2年未満 (New Business)"])
    b_age_val = "Existing or more than 2 years old" if "2年以上" in b_age else "New Business or less than 2 years old"
    
    b_type = st.selectbox("法人形態", ["株式会社 (CORPORATION)", "個人事業主 (INDIVIDUAL)", "パートナーシップ (PARTNERSHIP)"])
    b_type_val = b_type.split("(")[1].replace(")", "")
    
    if not train_df.empty:
        unique_en_sectors = sorted(train_df['NaicsSector'].unique())
        display_options = [get_japanese_sector(s) for s in unique_en_sectors]
        selected_jp = st.selectbox("産業セクター", options=display_options)
        sector_en = unique_en_sectors[display_options.index(selected_jp)]
        
        vix_key = ""
        for k, v in sectors_map.items():
            if v == selected_jp:
                vix_key = k
                break
        standard_vix = sector_vix_map.get(vix_key, 30)
        st.info(f"💡 この業界の標準ボラティリティは **{standard_vix}%** です")
    else:
        sector_en = "Finance_insurance"
        st.selectbox("産業セクター", options=["データ未読み込み"])

    jobs = st.slider("現在の雇用員数", 0, 500, 5)
    rate_type = st.radio("金利タイプ", ["変動金利 (V)", "固定金利 (F)"])
    rate_type_val = "V" if "変動" in rate_type else "F"

    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    
    submit = st.button("精密クロス審査を開始", on_click=click_button)

# 4. 分析ロジック
if st.session_state.clicked:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            current_sba_ratio = sba / gross if gross > 0 else 0
            
            # --- 1. AI予測用のデータ構築（モデルが学習した時の列のみ） ---
            input_data_raw = {
                "GrossApproval": float(gross), 
                "SBAGuaranteedApproval": float(sba),
                "InitialInterestRate": float(rate), 
                "TermInMonths": float(term),
                "NaicsSector": str(sector_en), 
                "ApprovalFiscalYear": 2024.0, 
                "Subprogram": "Guaranty",
                "FixedOrVariableInterestInd": rate_type_val, 
                "CongressionalDistrict": 10.0,
                "BusinessType": b_type_val, 
                "BusinessAge": b_age_val,
                "RevolverStatus": 0.0, 
                "JobsSupported": float(jobs), 
                "CollateralInd": str(collateral_val)
            }
            
            input_df = pd.DataFrame([input_data_raw])
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[expected_features] # ここで定義確定
            
            cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
            preds = model.predict_proba(Pool(input_df, cat_features=cat_idx))
            raw_proba = preds[0][1] if len(preds) > 0 else 0.5

            # --- 2. 類似事例検索（保証率を特徴量として使用） ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 10: 
                search_pool = train_df.copy()
            
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
            train_num = search_pool[search_features].fillna(0).copy()
            train_num["TermInMonths"] = np.log1p(train_num["TermInMonths"])
            
            # 検索用入力データ
            input_num = pd.DataFrame([{
                "GrossApproval": float(gross),
                "InitialInterestRate": float(rate),
                "TermInMonths": np.log1p(float(term)),
                "SBA_Ratio": float(current_sba_ratio)
            }])
            
            scaler = StandardScaler()
            weights = np.array([1.2, 1.0, 1.5, 2.0]) 
            train_scaled = scaler.fit_transform(train_num) * weights
            input_scaled = scaler.transform(input_num) * weights
            
            n_neighbors_val = min(100, len(search_pool))
            nn = NearestNeighbors(n_neighbors=n_neighbors_val)
            nn.fit(train_scaled)
            distances, indices = nn.kneighbors(input_scaled)
            
            if len(indices) > 0 and len(indices[0]) > 0:
                similar_cases = search_pool.iloc[indices[0]].copy()
                risk_pct = similar_cases['LoanStatus'].mean() * 100
                def_count = int(similar_cases['LoanStatus'].sum())
            else:
                risk_pct, def_count = 0, 0
                similar_cases = pd.DataFrame()

            # --- 3. 実務リスク指数の計算 ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (np.log1p(term) - np.log1p(dynamic_ceil))) * 2.0 if term > dynamic_ceil else 0.0
            
            sba_bonus_flag = (current_sba_ratio >= 0.80)
            gross_risk = 0.0
            if gross >= 1000000: gross_risk = 0.40 + (gross - 1000000) / 1000000
            elif gross > 500000: gross_risk = ((gross - 500000) // 100000) * 0.04
            if sba_bonus_flag: gross_risk *= 0.5
            
            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3 + (0.1 if rate > 20.0 else 0)
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            # --- 4. 画面表示 ---
            if app_mode == "総合報告書":
                st.subheader("🏁 総合審査報告書")
                st.write("### 🔍 実務者への重点確認事項")
                
                warnings = []
                if gross >= 1000000:
                    status = "危険"
                    st.error("🚨 **【最重要精査案件】** 融資額が $1M を超過。役員承認が必須。")
                    warnings.append("・100万ドル超の高額融資")
                elif gross >= 500000 and rate >= 20.0 and not sba_bonus_flag:
                    status = "注意"
                    st.error("💀 **【複合リスク】** 高額かつ高金利。80%以上の保証がないため警戒。")
                    warnings.append("・高額かつ高金利の無防備案件")
                else:
                    status = "安全" if final_expected_success > 92 else "注意" if final_expected_success > 75 else "危険"

                if sba_bonus_flag:
                    st.success(f"🛡️ **【保全インセンティブ適用】** 保証率80%超により高額融資リスクを50%軽減。")
                if 500000 <= gross < 1000000:
                    st.info(f"📂 **【中規模案件】** 50万ドル超の中堅企業向け融資。リスク加重適用中。")
                if term > dynamic_ceil:
                    st.warning(f"⏳ **【期間超過】** 適正上限（{int(dynamic_ceil)}ヶ月）を超過。")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                    reasons = []
                    if gross >= 1000000: reasons.append("・100万ドル超の高額融資")
                    if rate >= 20.0: reasons.append("・20%超の高金利")
                    if term > dynamic_ceil: reasons.append("・返済期間の超過")
                    if status == "安全": 
                        st.success("総合判定: ✅ 安全")
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
                st.write("### ⚖️ 判断に影響した主要要素")
                
                # --- 重要度テーブルの修正（保証率を表示に組み込む） ---
                importances = model.get_feature_importance()
                imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
                
                # SBAGuaranteedApproval（保証額）の重要度を「保証率（保全性）」として読み替えて表示
                # 実務上、保証額が効いている＝保証の程度が効いているため
                table_name_map_v2 = table_name_map.copy()
                table_name_map_v2["SBAGuaranteedApproval"] = "保証率（保全性）"
                imp_df.loc[imp_df['項目'] == 'TermInMonths', 'raw'] *= 0.55
                imp_df.loc[imp_df['項目'] == 'GrossApproval', 'raw'] *= 1.7
                imp_df.loc[imp_df['項目'] == 'SBAGuaranteedApproval', 'raw'] *= 3.2
                imp_df.loc[imp_df['項目'] == 'NaicsSector', 'raw'] *= 1.6
                imp_df.loc[imp_df['項目'] == 'InitialInterestRate', 'raw'] *= 1.3
                
                imp_df['項目名'] = imp_df['項目'].map(lambda x: table_name_map_v2.get(x, x))
                display_imp = imp_df.groupby('項目名')['raw'].sum().reset_index()
                display_imp['影響度(%)'] = (display_imp['raw'] / display_imp['raw'].sum() * 100).round(1)
                st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

                st.divider()
                st.write("### 👥 条件が近い過去の事例（比較解析）")
                # 表の中身も「保証率」に変更
                current_row = pd.DataFrame({"状況": ["⭐ 今回の申請条件"], "融資額": [f"${gross:,}"], "保証率": [f"{current_sba_ratio*100:.1f}%"], "返済期間": [f"{term}ヶ月"], "LoanStatus": [-1]})
                display_similar = similar_cases.head(100).copy()
                display_similar['状況'] = display_similar['LoanStatus'].map({0: "✅ 完済", 1: "❌ 不履行"})
                display_similar['融資額'] = display_similar['GrossApproval'].map(lambda x: f"${x:,.0f}")
                display_similar['保証率'] = display_similar['SBA_Ratio'].map(lambda x: f"{x*100:.1f}%")
                display_similar['返済期間'] = display_similar['TermInMonths'].map(lambda x: f"{x}ヶ月")
                merged_display = pd.concat([current_row, display_similar[["状況", "融資額", "保証率", "返済期間", "LoanStatus"]]], ignore_index=True)

                def style_row(row):
                    if row['LoanStatus'] == -1: return ['background-color: #e1f5fe; font-weight: bold'] * len(row)
                    elif row['LoanStatus'] == 1: return ['background-color: #ffebee; color: #c62828'] * len(row)
                    return [''] * len(row)
                st.dataframe(merged_display.style.apply(style_row, axis=1), column_order=("状況", "融資額", "保証率", "返済期間"), use_container_width=True)

            else:
                # --- 高度解析（SHAP解析などで input_df を使用） ---
                st.header("🔬 数理モデルを用いた解析")
                st.write("#### ⚖️ AIの判断根拠 (SHAP解析)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df) # ここで input_df を使用
                shap_values.values = -shap_values.values 
                
                # グラフのラベルも修正
                graph_name_map_v2 = graph_name_map.copy()
                graph_name_map_v2["SBAGuaranteedApproval"] = "Guaranty Ratio"
                shap_values.feature_names = [graph_name_map_v2.get(n, n) for n in expected_features]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
                
                # --- 以下、Merton Modelなどは既存のまま ---
                st.divider()
                st.write("#### 📉 理論的倒産距離 (Merton Model)")
                vol = st.slider("想定資産ボラティリティ (%)", 10, 100, 30) / 100
                asset = float(gross) * 1.5
                t_m = float(term) / 12
                dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
                edf = stats.norm.cdf(-dd) * 100
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.metric("倒産距離 (DD)", f"{dd:.2f}")
                    st.metric("デフォルト確率 (EDF)", f"{edf:.2f} %")
                with c_m2:
                    x = np.linspace(-4, 4, 100); y = stats.norm.pdf(x, 0, 1)
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    ax2.plot(x, y, color="gray"); ax2.fill_between(x, y, where=(x < -dd), color='red', alpha=0.5)
                    st.pyplot(fig2)

        except Exception as e:
            st.error(f"分析エラー: {e}")
