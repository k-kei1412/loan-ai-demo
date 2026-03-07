import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：ハイブリッド精度版", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 翻訳辞書
SECTOR_TRANSLATE = {
    "Agriculture_forestry_fishing_hunting": "農業・林業・漁業", "Construction": "建設業",
    "Finance_insurance": "金融・保険業", "Retail trade": "小売業", "Manufacturing": "製造業",
    "Wholesale trade": "卸売業", "Transportation_warehousing": "運輸・倉庫業",
    "Information": "情報通信業", "Real estate_rental_leasing": "不動産・賃貸業",
    "Professional_scientific_technical services": "専門・技術サービス"
}
NAICS_MAP = {"11": "Agriculture_forestry_fishing_hunting", "23": "Construction", "31": "Manufacturing", 
             "42": "Wholesale trade", "44": "Retail trade", "52": "Finance_insurance", "54": "Professional_scientific_technical services"}

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

# --- サイドバー (日本語UI) ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 8.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    
    sector_key = st.selectbox("産業セクター", options=list(NAICS_MAP.keys()), 
                              format_func=lambda x: f"{x}: {SECTOR_TRANSLATE[NAICS_MAP[x]]}", index=5)
    sector_en = NAICS_MAP[sector_key]
    
    business_age_jp = st.selectbox("企業年齢", ["新規創業 (Startup)", "既存企業 (Existing)"])
    age_map = {"新規創業 (Startup)": "Startup", "既存企業 (Existing)": "Existing"}
    
    collateral_jp = st.selectbox("担保の提供", ["あり", "なし"])
    collateral_val = "Y" if collateral_jp == "あり" else "N"
    
    submit = st.button("精密クロス審査を開始")

if submit:
    try:
        # A. AI予測
        input_data = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "InitialInterestRate": float(rate), "TermInMonths": float(term),
            "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "7(a)",
            "FixedOrVariableInterestInd": "Fixed", "CongressionalDistrict": 10.0,
            "BusinessType": "CORPORATION", "BusinessAge": age_map[business_age_jp], 
            "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": collateral_val
        }
        input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
        proba = model.predict_proba(input_df)[0][1]

        # B. 【ハイブリッド】類似事例検索
        if not train_df.empty:
            search_cols = ["GrossApproval", "TermInMonths", "G_Ratio"]
            input_ratio = float(sba) / float(gross) if gross > 0 else 0
            
            # 全データから検索
            train_num = train_df[search_cols].fillna(0)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            
            # 重み付け (保証比率 5.0, 金額 3.0, 期間 0.5)
            weights_nn = np.array([3.0, 0.5, 5.0])
            train_weighted = train_scaled * weights_nn
            input_weighted = scaler.transform(pd.DataFrame([[gross, term, input_ratio]], columns=search_cols)) * weights_nn
            
            # --- ここで「業界一致ボーナス」を付与 ---
            # 業界が違うデータに「距離ペナルティ(1.0)」を加え、同じ業界を優先する
            sector_penalty = (train_df['NaicsSector'] != sector_en).astype(float).values * 1.0
            train_final = np.column_stack([train_weighted, sector_penalty])
            input_final = np.append(input_weighted, 0.0).reshape(1, -1) # 入力側はペナルティ0

            nn = NearestNeighbors(n_neighbors=50)
            nn.fit(train_final)
            _, indices = nn.kneighbors(input_final)
            similar_cases = train_df.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # C. 表示
        risk_index = (proba * 0.3) + (risk_pct / 100 * 0.7)
        st.subheader("🏁 総合審査報告書")
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
        
        # D. 判断構成要素 (比重補正)
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
            st.table(imp_df.sort_values('adj_imp', ascending=False).head(5)[['項目']])
            
        with col_tips:
            st.write("### 📝 審査のアドバイス")
            if def_count >= 15:
                st.error(f"🚨 **実績不履行率 {risk_pct:.1f}% の警戒領域**")
                st.write("この金額規模・保証条件での過去事例は非常に不安定です。")
            else:
                st.success("✅ **実績ベースの懸念は低めです**")

        st.write(f"### 📂 属性が近い類似事例 (優先業界: {SECTOR_TRANSLATE.get(sector_en)})")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 事故" if x == 1 else "✅ 完済")
        similar_cases['業種名'] = similar_cases['NaicsSector'].map(SECTOR_TRANSLATE).fillna(similar_cases['NaicsSector'])
        similar_cases['保証比率'] = (similar_cases['G_Ratio'] * 100).round(1).astype(str) + "%"
        similar_cases['担保'] = similar_cases['CollateralInd'].apply(lambda x: "あり" if x == "Y" else "なし")
        
        st.dataframe(similar_cases[['結果', 'GrossApproval', 'SBAGuaranteedApproval', '保証比率', 'TermInMonths', '担保', '業種名']].style.apply(
            lambda s: ['background-color: #ffcccc' if s.結果 == "❌ 事故" else '' for _ in s], axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"システムエラー: {e}")
