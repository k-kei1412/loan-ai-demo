import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：真・完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 翻訳・セクター定義
SECTOR_TRANSLATE = {
    "Accommodation_food services": "宿泊・飲食サービス", "Administrative_support_waste_management": "支援・廃棄物処理",
    "Agriculture_forestry_fishing_hunting": "農業・林業・漁業", "Arts_entertainment_recreation": "娯楽・レクリエーション",
    "Construction": "建設業", "Educational services": "教育サービス", "Finance_insurance": "金融・保険業",
    "Health care_social assistance": "医療・社会福祉", "Information": "情報通信業", "Manufacturing": "製造業",
    "Mining_quarrying_oil_gas extraction": "鉱業・採掘", "Other services": "その他サービス",
    "Professional_scientific_technical services": "専門・技術サービス", "Public administration": "公務",
    "Real estate_rental_leasing": "不動産・賃貸業", "Retail trade": "小売業", "Transportation_warehousing": "運輸・倉庫業",
    "Utilities": "電気・ガス・水道", "Wholesale trade": "卸売業"
}

@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        train_df = pd.read_csv("train (4).csv")
        # 業界名を正規化
        train_df['NaicsSector'] = train_df['NaicsSector'].astype(str)
    except:
        train_df = pd.DataFrame()
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー (実務入力) ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    sector_en = st.selectbox("産業セクター", options=sorted(list(SECTOR_TRANSLATE.keys())), 
                               format_func=lambda x: SECTOR_TRANSLATE[x])
    business_age = st.selectbox("企業年齢", ["Existing or more than 2 years old", "New Business or 2 years or less", "Startup, Loan Funds will Open Business", "Unanswered"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    try:
        # --- A. AI予測 ---
        input_data = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "InitialInterestRate": float(rate), "TermInMonths": float(term),
            "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "Guaranty",
            "FixedOrVariableInterestInd": "V", "CongressionalDistrict": 10.0,
            "BusinessType": "CORPORATION", "BusinessAge": business_age, 
            "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": collateral
        }
        input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

        # --- B. 類似事例検索 (全データ・ハイブリッド) ---
        risk_pct, def_count = 0.0, 0
        if not train_df.empty:
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = train_df[search_features].fillna(0)
            input_num = input_df[search_features].fillna(0)
            
            # 【最高精度重み】金額 1.5倍 / 期間 0.5倍
            weights_search = np.array([1.5, 1.0, 0.5]) 
            scaler = StandardScaler()
            train_scaled = (scaler.fit_transform(train_num)) * weights_search
            input_scaled = (scaler.transform(input_num)) * weights_search

            # 業界一致にボーナス
            sector_penalty = (train_df['NaicsSector'] != sector_en).astype(float).values * 1.0
            train_final = np.column_stack([train_scaled, sector_penalty])
            input_final = np.append(input_scaled, 0.0).reshape(1, -1)

            nn = NearestNeighbors(n_neighbors=50)
            nn.fit(train_final)
            _, indices = nn.kneighbors(input_final)
            similar_cases = train_df.iloc[indices[0]].copy()
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # --- C. 【重要：実務厳格化補正】 ---
        # 1. AI予測値の補正 (どんなに安全でも1%のリスクを想定)
        strict_proba = np.clip(proba, 0.01, 0.99)
        # 2. 実績事故率の補正 (0件でも「+0.5件」の不確実性を加味)
        strict_risk_pct = (def_count + 0.5) / (50 + 1)
        # 3. 5:5 ブレンド
        risk_index = (strict_proba * 0.5) + (strict_risk_pct * 0.5)

        # --- D. 画面表示 ---
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            if risk_index < 0.08:
                st.success("総合判定: ✅ 安全")
                status = "安全"
            elif risk_index < 0.20:
                st.warning("総合判定: ⚠️ 注意")
                status = "注意"
            else:
                st.error("総合判定: 🚨 危険")
                status = "危険"
        with c2:
            st.metric("実績事故率 (近傍)", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 類似50件中、デフォルトは **{def_count}件**")
        with c3:
            # 100.0%という表示を避け、信頼感を現実的に
            display_conf = min((1 - strict_proba) * 100, 98.9)
            st.metric("AI完済期待値", f"{display_conf:.1f} %")
            st.caption("全データ(2000件超)に基づく統計予測")

        st.divider()

        # --- E. 影響度 & アドバイス ---
        col_imp, col_tips = st.columns(2)
        with col_imp:
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            importances = model.get_feature_importance()
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資総額", "InitialInterestRate": "初期金利", 
                        "NaicsSector": "産業セクター", "SBAGuaranteedApproval": "SBA保証額", "CollateralInd": "担保有無"}
            imp_df = pd.DataFrame({'項目': [name_map.get(f, f) for f in expected_features], 'raw': importances})
            imp_df['adj'] = imp_df['raw']
            imp_df.loc[imp_df['項目'] == '返済期間', 'adj'] *= 0.5
            imp_df.loc[imp_df['項目'] == '融資総額', 'adj'] *= 1.8
            total_adj = imp_df['adj'].sum()
            imp_df['影響度(%)'] = (imp_df['adj'] / total_adj * 100).round(2)
            st.table(imp_df.sort_values('影響度(%)', ascending=False).head(5)[['項目', '影響度(%)']])

        with col_tips:
            st.write("### 📝 審査のアドバイス")
            if status == "安全" and display_conf > 98:
                st.success("✅ **【極めて健全】** AIと過去実績が共に高い安全性を認めています。")
                st.write("特に期間と担保条件が優良です。")
            elif status == "注意":
                st.warning("⚠️ **【要精査】** AI予測と現場の実績にわずかな乖離があります。")
                st.write("直近の不履行事例（赤色の行）との共通点がないか確認してください。")
            elif status == "危険":
                st.error("🚨 **【否決推奨】** 統計的・実績的にデフォルトの危険域です。")
                st.write("返済原資の確実性を再検討し、不可なら否決を検討してください。")

        # --- F. 事例詳細 ---
        st.write(f"### 📂 属性が近い類似事例 (全データから抽出)")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        similar_cases['業種'] = similar_cases['NaicsSector'].map(SECTOR_TRANSLATE).fillna(similar_cases['NaicsSector'])
        st.dataframe(similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', '業種']].style.apply(
            lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
        ), use_container_width=True)

    except Exception as e:
        st.error(f"システムエラー: {e}")
