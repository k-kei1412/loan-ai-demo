import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：究極完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 翻訳辞書
SECTOR_TRANSLATE = {
    "Agriculture_forestry_fishing_hunting": "農業・林業・漁業",
    "Mining_quarrying_oil_gas extraction": "鉱業・採掘",
    "Utilities": "電気・ガス・水道",
    "Construction": "建設業",
    "Manufacturing": "製造業",
    "Wholesale trade": "卸売業",
    "Retail trade": "小売業",
    "Transportation_warehousing": "運輸・倉庫業",
    "Information": "情報通信業",
    "Finance_insurance": "金融・保険業",
    "Real estate_rental_leasing": "不動産・賃貸業",
    "Professional_scientific_technical services": "専門・技術サービス",
    "Management of companies_enterprises": "企業管理",
    "Administrative_support_waste management_remediation services": "支援・廃棄物処理",
    "Educational services": "教育サービス",
    "Health care_social assistance": "医療・社会福祉",
    "Arts_entertainment_recreation": "娯楽・レクリエーション",
    "Accommodation_food services": "宿泊・飲食サービス",
    "Other services": "その他サービス",
    "Public administration": "公務"
}

# 逆引き用
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
    except:
        train_df = pd.DataFrame()
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    year = st.number_input("承認年度", 1990, 2026, 2010)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    jobs = st.number_input("雇用人数", 0, 1000, 5)
    subprogram = st.text_input("ローンプログラム", "7(a)")
    rate_type = st.selectbox("金利タイプ", ["Fixed", "Variable"])
    
    sector_key = st.selectbox(
        "産業セクター", 
        options=list(NAICS_MAP.keys()), 
        format_func=lambda x: f"{x}: {SECTOR_TRANSLATE[NAICS_MAP[x]]}",
        index=9
    )
    sector_en = NAICS_MAP[sector_key]
    
    district = st.number_input("地区コード", 1, 60, 10)
    business_type = st.selectbox("企業形態", ["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"])
    business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
    revolver = st.selectbox("リボルビングローン", ["Y", "N"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    try:
        # AI予測用データ作成
        raw_input = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "ApprovalFiscalYear": float(year), "Subprogram": str(subprogram),
            "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": str(rate_type),
            "TermInMonths": float(term), "NaicsSector": str(sector_en), 
            "CongressionalDistrict": float(district), "BusinessType": str(business_type), 
            "BusinessAge": str(business_age), "RevolverStatus": 1.0 if revolver == "Y" else 0.0,
            "JobsSupported": float(jobs), "CollateralInd": str(collateral)
        }
        input_df = pd.DataFrame([raw_input]).reindex(columns=expected_features)
        
        # A. AI予測
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        pool = Pool(input_df, cat_features=cat_idx)
        proba = model.predict_proba(pool)[0][1]

        # B. 類似事例検索（精度の改善）
        risk_pct, def_count = 0.0, 0
        similar_cases = pd.DataFrame()
        if not train_df.empty:
            # 同じセクターで絞り込み
            filtered_df = train_df[train_df['NaicsSector'] == sector_en].copy()
            search_pool = filtered_df if len(filtered_df) >= 50 else train_df.copy()
            
            # 特徴量選びを「資金額」と「金利」に集中
            search_cols = ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate"]
            train_num = search_pool[search_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_num = input_df[search_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            nn = NearestNeighbors(n_neighbors=min(50, len(search_pool)), metric='euclidean')
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(scaler.transform(input_num))
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # C. 判定と表示
        st.subheader("🏁 総合審査報告書")
        risk_index = (proba * 0.4) + (risk_pct / 100 * 0.6) # AIと実績を4:6でブレンド
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            # --- ここで判定ラベルを表示 ---
            if risk_index >= 0.20:
                st.error("総合判定: 🚨 危険 (否決推奨)")
            elif risk_index >= 0.08:
                st.warning("総合判定: ⚠️ 注意")
            else:
                st.success("総合判定: ✅ 安全")
        
        with c2:
            st.metric("近傍の実績事故率", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 類似50件中、デフォルトは **{def_count}件**")
        
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")

        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### ⚖️ 判断の主要要素")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, '重要度': importances}).sort_values('重要度', ascending=False).head(5)
            st.table(imp_df)
            
        with col2:
            st.write("### 📝 審査のアドバイス")
            if def_count >= 10:
                st.error(f"🚨 **警戒が必要な業種・規模です**")
                st.write(f"類似案件で {def_count} 件の不履行が出ています。AI予測よりも「現場の実績不履行率」を重く見て判断してください。")
            else:
                st.success("✅ **実績ベースでは安定しています**")
                st.write("近しい条件の案件で事故は少なく、データ上は貸付可能な圏内です。")

        st.write("### 📂 属性が近い類似事例詳細")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        similar_cases['業種名'] = similar_cases['NaicsSector'].map(SECTOR_TRANSLATE).fillna("不明")
        
        display_cols = ['結果', 'GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate', 'TermInMonths', '業種名']
        st.dataframe(
            similar_cases[display_cols].style.apply(
                lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
            ), use_container_width=True
        )

    except Exception as e:
        st.error(f"エラー: {e}")
