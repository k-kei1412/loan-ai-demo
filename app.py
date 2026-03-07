import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：最高精度継承版", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 翻訳・マップ用辞書
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
        train_df['NaicsSector'] = train_df['NaicsSector'].astype(str)
        # 類似検索用の保証比率を追加
        train_df['G_Ratio'] = train_df['SBAGuaranteedApproval'] / train_df['GrossApproval']
    except:
        train_df = pd.DataFrame()
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー（漢字選択肢） ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
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
        # --- A. AI予測 ---
        input_data = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "InitialInterestRate": float(rate), "TermInMonths": float(term),
            "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "7(a)",
            "FixedOrVariableInterestInd": "Fixed", "CongressionalDistrict": 10.0,
            "BusinessType": "CORPORATION", "BusinessAge": age_map[business_age_jp], 
            "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": collateral_val
        }
        input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

        # --- B. 類似事例検索 (精度が良かった重みを継承) ---
        risk_pct, def_count = 0.0, 0
        if not train_df.empty:
            search_pool = train_df.copy()
            # 検索用の数値特徴
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].fillna(0)
            input_num = input_df[search_features].fillna(0)
            
            # 【継承】検索重み付け補正 (融資額重視 / 期間軽視)
            weights_search = np.array([1.5, 1.0, 0.5]) 
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num) * weights_search
            input_scaled = scaler.transform(input_num) * weights_search

            # ハイブリッド要素（業界一致を優先）
            sector_penalty = (search_pool['NaicsSector'] != sector_en).astype(float).values * 1.0
            train_final = np.column_stack([train_scaled, sector_penalty])
            input_final = np.append(input_scaled, 0.0).reshape(1, -1)

            nn = NearestNeighbors(n_neighbors=min(50, len(search_pool)))
            nn.fit(train_final)
            _, indices = nn.kneighbors(input_final)
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # --- C. 影響度計算 (割合%表示) ---
        importances = model.get_feature_importance()
        name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資総額", "InitialInterestRate": "初期金利", 
                    "NaicsSector": "産業セクター", "SBAGuaranteedApproval": "SBA保証額", "CollateralInd": "担保有無"}
        imp_df = pd.DataFrame({'項目': [name_map.get(f, f) for f in expected_features], 'raw': importances})
        # 【継承】実務バランス補正
        imp_df['adj'] = imp_df['raw']
        imp_df.loc[imp_df['項目'] == '返済期間', 'adj'] *= 0.5
        imp_df.loc[imp_df['項目'] == '融資総額', 'adj'] *= 1.8
        
        total_adj = imp_df['adj'].sum()
        imp_df['影響度(%)'] = (imp_df['adj'] / total_adj * 100).round(2)
        display_imp = imp_df.sort_values('影響度(%)', ascending=False).head(5)[['項目', '影響度(%)']]

        # --- D. 【継承】実効リスク指数 (実績重視 90%) ---
        risk_index = (proba * 0.1) + (risk_pct / 100 * 0.9)

        # --- E. 画面表示 ---
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            if risk_index < 0.05: status = "安全"; st.success(f"総合判定: ✅ {status}")
            elif risk_index < 0.15: status = "注意"; st.warning(f"総合判定: ⚠️ {status}")
            else: status = "危険"; st.error(f"総合判定: 🚨 慎重検討（否決推奨）")
        with c2:
            st.metric("近傍の実績事故率", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 類似50件中、デフォルトは **{def_count}件**")
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")

        st.divider()

        # --- F. 主要因 & 動的アドバイス ---
        col_imp, col_tips = st.columns(2)
        with col_imp:
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            st.table(display_imp)
            st.bar_chart(display_imp.set_index('項目'))

        with col_tips:
            st.write("### 📝 審査のアドバイス")
            if status == "危険" or def_count >= 15:
                st.error("🚨 **【否決推奨】致命的なリスクを確認**")
                st.write(f"実績事故率が {risk_pct:.1f}% と極めて高く、統計的に極めて危険な領域です。")
                st.write("融資額の減額か、担保の追加を必須条件としてください。")
            elif status == "注意":
                st.warning("⚠️ **【要確認】条件付き承認の検討**")
                st.write(f"実績で {def_count} 件の不履行があります。AI予測と実績の乖離に注意してください。")
                st.write("返済期間の短縮や、月々の返済余力を再精査することをお勧めします。")
            else:
                st.success("✅ **【承認推奨】極めて堅実な案件**")
                st.write("実績事故率が極めて低く、AIの完済確信度も十分です。現行条件での承認を推奨します。")

        # --- G. 事例詳細 ---
        st.write(f"### 📂 属性が近い類似事例 (優先業界: {SECTOR_TRANSLATE.get(sector_en)})")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        similar_cases['業種名'] = similar_cases['NaicsSector'].map(SECTOR_TRANSLATE).fillna(similar_cases['NaicsSector'])
        
        st.dataframe(similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', '業種名']].style.apply(
            lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
        ), use_container_width=True)

    except Exception as e:
        st.error(f"システムエラー: {e}")
