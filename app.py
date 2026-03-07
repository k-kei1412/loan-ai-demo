import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：完全日本語版", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# --- 各種翻訳辞書 ---
SECTOR_TRANSLATE = {
    "Agriculture_forestry_fishing_hunting": "農業・林業・漁業", "Mining_quarrying_oil_gas extraction": "鉱業・採掘",
    "Utilities": "電気・ガス・水道", "Construction": "建設業", "Manufacturing": "製造業",
    "Wholesale trade": "卸売業", "Retail trade": "小売業", "Transportation_warehousing": "運輸・倉庫業",
    "Information": "情報通信業", "Finance_insurance": "金融・保険業", "Real estate_rental_leasing": "不動産・賃貸業",
    "Professional_scientific_technical services": "専門・技術サービス", "Management of companies_enterprises": "企業管理",
    "Administrative_support_waste management_remediation services": "支援・廃棄物処理", "Educational services": "教育サービス",
    "Health care_social assistance": "医療・社会福祉", "Arts_entertainment_recreation": "娯楽・レクリエーション",
    "Accommodation_food services": "宿泊・飲食サービス", "Other services": "その他サービス", "Public administration": "公務"
}

NAICS_MAP = {
    "11": "Agriculture_forestry_fishing_hunting", "21": "Mining_quarrying_oil_gas extraction",
    "22": "Utilities", "23": "Construction", "31": "Manufacturing", "42": "Wholesale trade",
    "44": "Retail trade", "48": "Transportation_warehousing", "51": "Information",
    "52": "Finance_insurance", "53": "Real estate_rental_leasing",
    "54": "Professional_scientific_technical services",
    "56": "Administrative_support_waste management_remediation services",
    "61": "Educational services", "62": "Health care_social assistance",
    "71": "Arts_entertainment_recreation", "72": "Accommodation_food services",
    "81": "Other services", "92": "Public administration"
}

@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        train_df = pd.read_csv("train.csv")
        train_df['G_Ratio'] = train_df['SBAGuaranteedApproval'] / train_df['GrossApproval']
    except:
        train_df = pd.DataFrame()
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー入力（完全日本語化） ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    year = st.number_input("承認年度", 1990, 2026, 2010)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 8.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    jobs = st.number_input("雇用人数", 0, 1000, 5)
    
    # 選択肢の日本語化
    business_type_jp = st.selectbox("企業形態", ["株式会社/法人", "個人事業主", "パートナーシップ"])
    type_map = {"株式会社/法人": "CORPORATION", "個人事業主": "INDIVIDUAL", "パートナーシップ": "PARTNERSHIP"}
    
    business_age_jp = st.selectbox("企業年齢", ["新規創業 (Startup)", "既存企業 (Existing)"])
    age_map = {"新規創業 (Startup)": "Startup", "既存企業 (Existing)": "Existing"}
    
    sector_key = st.selectbox("産業セクター", options=list(NAICS_MAP.keys()), 
                              format_func=lambda x: f"{x}: {SECTOR_TRANSLATE[NAICS_MAP[x]]}", index=9)
    sector_en = NAICS_MAP[sector_key]
    
    revolver_jp = st.selectbox("リボルビング枠の利用", ["なし", "あり"])
    revolver_val = 1.0 if revolver_jp == "あり" else 0.0
    
    collateral_jp = st.selectbox("担保の提供", ["あり", "なし"])
    collateral_val = "Y" if collateral_jp == "あり" else "N"
    
    submit = st.button("精密クロス審査を開始")

if submit:
    try:
        # A. AI予測 (地区コードを固定値10.0として内部処理)
        input_data = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "ApprovalFiscalYear": float(year), "Subprogram": "7(a)",
            "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": "Fixed",
            "TermInMonths": float(term), "NaicsSector": str(sector_en), 
            "CongressionalDistrict": 10.0, # 地区コードは固定
            "BusinessType": type_map[business_type_jp], 
            "BusinessAge": age_map[business_age_jp], 
            "RevolverStatus": revolver_val,
            "JobsSupported": float(jobs), "CollateralInd": collateral_val
        }
        input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
        proba = model.predict_proba(input_df)[0][1]

        # B. 類似事例検索（高精度版）
        risk_pct, def_count = 0.0, 0
        similar_cases = pd.DataFrame()
        if not train_df.empty:
            filtered_df = train_df[train_df['NaicsSector'] == sector_en].copy()
            search_pool = filtered_df if len(filtered_df) >= 50 else train_df.copy()
            search_cols = ["GrossApproval", "TermInMonths", "G_Ratio"]
            input_ratio = float(sba) / float(gross) if gross > 0 else 0
            train_num = search_pool[search_cols].fillna(0)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            weights_nn = np.array([3.0, 1.5, 5.0])
            train_weighted = train_scaled * weights_nn
            input_weighted = scaler.transform(pd.DataFrame([[gross, term, input_ratio]], columns=search_cols)) * weights_nn
            nn = NearestNeighbors(n_neighbors=min(50, len(search_pool)))
            nn.fit(train_weighted)
            _, indices = nn.kneighbors(input_weighted)
            similar_cases = search_pool.iloc[indices[0]].copy()
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # C. 判定メトリクス
        st.subheader("🏁 総合審査報告書")
        risk_index = (proba * 0.3) + (risk_pct / 100 * 0.7)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            if risk_index >= 0.20: st.error("総合判定: 🚨 危険 (否決推奨)")
            elif risk_index >= 0.08: st.warning("総合判定: ⚠️ 注意")
            else: st.success("総合判定: ✅ 安全")
        with c2:
            st.metric("実績事故率", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 類似50件中、デフォルトは **{def_count}件**")
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")

        st.divider()
        
        # D. 判断構成要素（比重補正維持）
        col_imp, col_tips = st.columns(2)
        with col_imp:
            st.write("### ⚖️ 判断の主要構成要素 (実務補正済)")
            importances = model.get_feature_importance()
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資総額", "InitialInterestRate": "初期金利", 
                        "NaicsSector": "産業セクター", "SBAGuaranteedApproval": "SBA保証額", "CollateralInd": "担保有無"}
            imp_df = pd.DataFrame({'項目': [name_map.get(f, f) for f in expected_features], 'raw_imp': importances})
            imp_df['adj_imp'] = imp_df['raw_imp']
            imp_df.loc[imp_df['項目'] == '返済期間', 'adj_imp'] *= 0.3
            imp_df.loc[imp_df['項目'] == '融資総額', 'adj_imp'] *= 2.5
            total = imp_df['adj_imp'].sum()
            imp_df['影響度(%)'] = (imp_df['adj_imp'] / total * 100).round(2)
            st.table(imp_df.sort_values('影響度(%)', ascending=False).head(5)[['項目', '影響度(%)']])
            
        with col_tips:
            st.write("### 📝 審査のアドバイス")
            if def_count >= 15:
                st.error("🚨 **【警告】非常にリスクが高い案件です**")
                st.write(f"類似事例の {risk_pct:.1f}% がデフォルトしています。融資額に対して保証額や担保が十分か、極めて慎重に判断してください。")
            elif def_count >= 5:
                st.warning("⚠️ **【注意】不履行の兆候があります**")
                st.write("過去の類似事例で一部事故が発生しています。金利設定や返済計画の再精査を推奨します。")
            else:
                st.success("✅ **【良好】実績・AI共に安定しています**")
                st.write("類似事例の事故が少なく、AIの評価も高いため、健全な案件と判断されます。")

        # E. 類似事例詳細 (担保・業種名あり)
        st.write(f"### 📂 近しい属性の類似事例 (業種: {SECTOR_TRANSLATE.get(sector_en)})")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 事故" if x == 1 else "✅ 完済")
        similar_cases['業種名'] = similar_cases['NaicsSector'].map(SECTOR_TRANSLATE).fillna("不明")
        similar_cases['保証比率'] = (similar_cases['G_Ratio'] * 100).round(1).astype(str) + "%"
        similar_cases['担保'] = similar_cases['CollateralInd'].apply(lambda x: "あり" if x == "Y" else "なし")
        
        display_cols = ['結果', 'GrossApproval', 'SBAGuaranteedApproval', '保証比率', 'TermInMonths', '担保', '業種名']
        st.dataframe(similar_cases[display_cols].style.apply(lambda s: ['background-color: #ffcccc' if s.結果 == "❌ 事故" else '' for _ in s], axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"システムエラー: {e}")
