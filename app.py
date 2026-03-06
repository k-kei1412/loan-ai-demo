import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="次世代ローン審査AI", layout="wide")
st.title("🏦 ローンデフォルト予測 AIシステム [究極版]")
st.markdown("数値の標準化に基づいた高精度な類似事例検索と、AI予測のクロスチェックを行います。")

# 2. リソースの読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    
    try:
        # train.csv を読み込み
        train_df = pd.read_csv("train.csv")
    except Exception as e:
        st.error(f"警告: train.csv が見つかりません。 ({e})")
        train_df = pd.DataFrame()
    
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# 3. 入力フォーム (サイドバー)
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
    sector = st.number_input("産業セクター (NAICS)", 1, 99, 44)
    district = st.number_input("地区コード", 1, 60, 10)
    business_type = st.text_input("企業形態", "CORPORATION")
    business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
    revolver = st.selectbox("リボルビングローン", ["Y", "N"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    
    submit = st.button("精密審査を実行")

# 4. 審査ロジック
if submit:
    # データ整形
    revolver_val = 1.0 if revolver == "Y" else 0.0
    raw_input = {
        "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year), "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term), "NaicsSector": str(int(sector)), 
        "CongressionalDistrict": float(district), "BusinessType": str(business_type), 
        "BusinessAge": str(business_age), "RevolverStatus": float(revolver_val),
        "JobsSupported": float(jobs), "CollateralInd": str(collateral)
    }
    input_df = pd.DataFrame([raw_input]).reindex(columns=expected_features)

    # 型の強制
    numeric_cols = ["GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear", 
                    "InitialInterestRate", "TermInMonths", "CongressionalDistrict", 
                    "JobsSupported", "RevolverStatus"]
    
    for col in input_df.columns:
        if col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
        else:
            input_df[col] = input_df[col].astype(str)

    try:
        # --- AI予測 ---
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        pool = Pool(input_df, cat_features=cat_idx)
        proba = model.predict_proba(pool)[0][1]

        st.subheader("🏁 総合審査結果報告書")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("AI算出デフォルト確率", f"{proba * 100:.4f} %")
            if proba > 0.05:
                st.error("AI判定: 🚨 危険")
            elif proba > 0.01:
                st.warning("AI判定: ⚠️ 注意")
            else:
                st.success("AI判定: ✅ 良好")

        # --- 精密類似事例検索 (KNN + Scaler) ---
        if not train_df.empty:
            # 検索精度の要: 距離計算に使う数値列の選定
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "JobsSupported", "NaicsSector"]
            
            # 過去データから必要な列を抽出 (欠損値は平均で埋める)
            train_sub = train_df[search_features].fillna(train_df[search_features].mean())
            input_sub = input_df[search_features].copy()
            input_sub["NaicsSector"] = input_sub["NaicsSector"].astype(float) # 距離計算用に一時的に数値化
            
            # 標準化 (単位の差をなくす)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_sub)
            input_scaled = scaler.transform(input_sub)
            
            # 検索実行
            nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
            nn.fit(train_scaled)
            distances, indices = nn.kneighbors(input_scaled)
            
            similar_cases = train_df.iloc[indices[0]].copy()
            
            if 'Default' in similar_cases.columns:
                actual_risk_pct = similar_cases['Default'].mean() * 100
                with m_col2:
                    st.metric("類似事例の実績リスク", f"{actual_risk_pct:.1f} %")
                    if actual_risk_pct >= 20.0:
                        st.error("実績判定: 🚨 危険")
                    elif actual_risk_pct > 0:
                        st.warning("実績判定: ⚠️ 要注意")
                    else:
                        st.success("実績判定: ✨ 非常に安全")

                with m_col3:
                    # AIと実績の乖離をチェック
                    diff = actual_risk_pct - (proba * 100)
                    if diff > 10:
                        st.error("信頼性警告: AIが楽観的すぎます")
                        st.caption("過去の生データはより高いリスクを示しています。")
                    else:
                        st.success("信頼性: 正常")
                        st.caption("AIと過去の実績が一致しています。")

        # --- 影響度と詳細表示 ---
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.write("### 💡 AIの重要視ポイント")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, '影響度': importances}).sort_values('影響度', ascending=False).head(8)
            st.dataframe(imp_df, use_container_width=True)

        with c2:
            st.write("### 📂 高精度検索された類似事例 (10件)")
            if 'Default' in similar_cases.columns:
                similar_cases['結果'] = similar_cases['Default'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
            
            # 入力と見比べるために重要な列を左に寄せる
            display_cols = ['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector'] + \
                           [c for c in similar_cases.columns if c not in ['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector', 'Default']]
            st.dataframe(similar_cases[display_cols], use_container_width=True)

    except Exception as e:
        st.error(f"システムエラー: {e}")
