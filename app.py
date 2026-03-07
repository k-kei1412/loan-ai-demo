import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：精度特化型", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 翻訳・マップ用辞書
SECTOR_TRANSLATE = {
    "Agriculture_forestry_fishing_hunting": "農業・林業・漁業", "Construction": "建設業",
    "Finance_insurance": "金融・保険業", "Retail trade": "小売業", "Manufacturing": "製造業" # 主要どころを抜粋
}
NAICS_MAP = {"52": "Finance_insurance", "23": "Construction", "44": "Retail trade"} # 必要に応じて追加

@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        train_df = pd.read_csv("train.csv")
        # 保証比率を事前に計算して精度を上げる
        train_df['G_Ratio'] = train_df['SBAGuaranteedApproval'] / train_df['GrossApproval']
    except:
        train_df = pd.DataFrame()
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー (省略せず重要項目のみ) ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 8.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    sector_en = "Finance_insurance" # 今回は金融固定
    submit = st.button("精密クロス審査を開始")

if submit:
    try:
        # A. AI予測
        input_data = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "InitialInterestRate": float(rate), "TermInMonths": float(term),
            "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "7(a)",
            "FixedOrVariableInterestInd": "Fixed", "CongressionalDistrict": 10.0,
            "BusinessType": "CORPORATION", "BusinessAge": "Startup",
            "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": "Y"
        }
        input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
        proba = model.predict_proba(input_df)[0][1]

        # B. 【超精度】類似事例検索
        if not train_df.empty:
            # 1. 業界で絞り込み
            filtered_df = train_df[train_df['NaicsSector'] == sector_en].copy()
            
            # 2. 検索用数値の作成 (保証比率を追加)
            input_ratio = float(sba) / float(gross) if gross > 0 else 0
            search_cols = ["GrossApproval", "TermInMonths", "G_Ratio"]
            
            train_num = filtered_df[search_cols].copy()
            input_num = pd.DataFrame([[gross, term, input_ratio]], columns=search_cols)

            # 3. 重み付けスケーリング (金額と比率を重視)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            input_scaled = scaler.transform(input_num)
            
            # 重みの適用 (融資額:x3, 期間:x2, 保証比率:x5)
            weights = np.array([3.0, 2.0, 5.0])
            train_weighted = train_scaled * weights
            input_weighted = input_scaled * weights

            # 4. 検索実行
            nn = NearestNeighbors(n_neighbors=min(50, len(filtered_df)))
            nn.fit(train_weighted)
            _, indices = nn.kneighbors(input_weighted)
            similar_cases = filtered_df.iloc[indices[0]].copy()

            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # C. 判定表示
        risk_index = (proba * 0.3) + (risk_pct / 100 * 0.7) # 実績を7割重視
        
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            # 判定ラベルの復活
            if risk_index >= 0.20: st.error("総合判定: 🚨 危険 (否決推奨)")
            elif risk_index >= 0.08: st.warning("総合判定: ⚠️ 注意")
            else: st.success("総合判定: ✅ 安全")
        
        with c2:
            st.metric("実績事故率", f"{risk_pct:.1f} %")
            st.write(f"🔍 金融業の類似50件中、不履行は **{def_count}件**")
        
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")

        st.divider()

        # D. 類似事例テーブル
        st.write("### 📂 条件が酷似している事例 (保証比率・金額重視)")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 事故" if x == 1 else "✅ 完済")
        similar_cases['保証比率'] = (similar_cases['G_Ratio'] * 100).round(1).astype(str) + "%"
        
        display_cols = ['結果', 'GrossApproval', 'SBAGuaranteedApproval', '保証比率', 'TermInMonths']
        st.dataframe(similar_cases[display_cols].style.apply(
            lambda s: ['background-color: #ffcccc' if s.結果 == "❌ 事故" else '' for _ in s], axis=1
        ), use_container_width=True)

    except Exception as e:
        st.error(f"エラー: {e}")
