import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：究極完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 【修正】CSVの中身（英語名）を日本語に変換する辞書
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

# サイドバー表示用の逆引き（コード -> 英語名）
# ※modelが期待する入力形式に合わせて調整してください
NAICS_MAP = {
    "11": "Agriculture_forestry_fishing_hunting",
    "21": "Mining_quarrying_oil_gas extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "42": "Wholesale trade",
    "44": "Retail trade",
    "48": "Transportation_warehousing",
    "51": "Information",
    "52": "Finance_insurance",
    "53": "Real estate_rental_leasing",
    "54": "Professional_scientific_technical services",
    "56": "Administrative_support_waste management_remediation services",
    "61": "Educational services",
    "62": "Health care_social assistance",
    "71": "Arts_entertainment_recreation",
    "72": "Accommodation_food services",
    "81": "Other services",
    "92": "Public administration"
}

@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        train_df = pd.read_csv("train.csv")
        train_df['NaicsSector'] = train_df['NaicsSector'].astype(str)
    except:
        train_df = pd.DataFrame()
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# 3. 入力フォーム
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
    
    # 選択肢は「コード：日本語」で見せ、値は「英語名」を送る
    sector_key = st.selectbox(
        "産業セクター", 
        options=list(NAICS_MAP.keys()), 
        format_func=lambda x: f"{x}: {SECTOR_TRANSLATE[NAICS_MAP[x]]}",
        index=9 # Finance_insurance
    )
    sector_name_en = NAICS_MAP[sector_key]
    
    district = st.number_input("地区コード", 1, 60, 10)
    business_type = st.selectbox("企業形態", ["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"])
    business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
    revolver = st.selectbox("リボルビングローン", ["Y", "N"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    revolver_val = 1.0 if revolver == "Y" else 0.0
    raw_input = {
        "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year), "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term), "NaicsSector": str(sector_name_en), # 英語名で渡す
        "CongressionalDistrict": float(district), "BusinessType": str(business_type), 
        "BusinessAge": str(business_age), "RevolverStatus": float(revolver_val),
        "JobsSupported": float(jobs), "CollateralInd": str(collateral)
    }
    input_df = pd.DataFrame([raw_input]).reindex(columns=expected_features)

    try:
        # AI予測
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        pool = Pool(input_df, cat_features=cat_idx)
        proba = model.predict_proba(pool)[0][1]

        # 類似検索
        if not train_df.empty:
            filtered_df = train_df[train_df['NaicsSector'] == sector_name_en].copy()
            search_pool = filtered_df if len(filtered_df) >= 50 else train_df.copy()
            
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_num = input_df[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            input_scaled = scaler.transform(input_num)
            nn = NearestNeighbors(n_neighbors=min(50, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # 表示
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1:
            risk_index = (proba * 10) + (risk_pct / 100 * 0.9)
            st.metric("実効リスク指数", f"{min(risk_index, 1.0) * 100:.2f} %")
        with c2:
            st.metric("近傍の実績事故率", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 類似50件（**{SECTOR_TRANSLATE.get(sector_name_en)}**）中、デフォルト **{def_count}件**")
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")

        st.divider()
        col_imp, col_tips = st.columns([1, 1])
        
        with col_imp:
            st.write("### ⚖️ 判断の主要構成要素")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, '影響度(%)': importances}).sort_values('影響度(%)', ascending=False).head(5)
            st.table(imp_df)

        with col_tips:
            st.write("### 📝 審査のアドバイス")
            if def_count >= 15:
                st.error(f"🚨 **【警告】地雷原エリア**")
                st.write(f"この業種（{SECTOR_TRANSLATE.get(sector_name_en)}）の類似案件はデフォルトが多発しています。")
            else:
                st.success("✅ **【良好】実績に大きな問題なし**")

        st.write("### 📂 属性が近い類似事例")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        
        # 英語名を日本語に変換
        similar_cases['業種名'] = similar_cases['NaicsSector'].map(SECTOR_TRANSLATE).fillna(similar_cases['NaicsSector'])
        
        display_cols = ['結果', 'GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate', 'TermInMonths', '業種名']
        st.dataframe(similar_cases[display_cols].style.apply(lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"エラー: {e}")
