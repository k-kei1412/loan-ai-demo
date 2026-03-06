import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：究極完全体", layout="wide")
st.title("🏦 ローンデフォルト予測 AIシステム [究極完全体]")

# 2. リソースの読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        train_df = pd.read_csv("train.csv")
        # 型のクリーンアップ
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
    sector = st.selectbox("産業セクター (NAICS)", ["11", "21", "22", "23", "31", "42", "44", "48", "51", "52", "53", "54", "56", "61", "62", "71", "72", "81", "92"], index=6)
    district = st.number_input("地区コード", 1, 60, 10)
    business_type = st.selectbox("企業形態", ["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"])
    business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
    revolver = st.selectbox("リボルビングローン", ["Y", "N"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    # データ整形
    revolver_val = 1.0 if revolver == "Y" else 0.0
    raw_input = {
        "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year), "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate), "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term), "NaicsSector": str(sector), 
        "CongressionalDistrict": float(district), "BusinessType": str(business_type), 
        "BusinessAge": str(business_age), "RevolverStatus": float(revolver_val),
        "JobsSupported": float(jobs), "CollateralInd": str(collateral)
    }
    input_df = pd.DataFrame([raw_input]).reindex(columns=expected_features)

    try:
        # --- AI予測 ---
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        pool = Pool(input_df, cat_features=cat_idx)
        proba = model.predict_proba(pool)[0][1]

        # --- 精密類似事例検索ロジック (改善版) ---
        if not train_df.empty:
            # A. フィルタリング：まず「業種」と「企業年齢」が同じものを優先
            filtered_df = train_df[
                (train_df['NaicsSector'] == str(sector)) & 
                (train_df['BusinessAge'].str.contains(business_age, na=False))
            ].copy()
            
            # もし候補が少なすぎれば、業種だけの一致に広げる
            if len(filtered_df) < 10:
                filtered_df = train_df[train_df['NaicsSector'] == str(sector)].copy()
            
            # それでも少なければ全データ
            search_pool = filtered_df if len(filtered_df) >= 10 else train_df.copy()

            # B. 数値距離の計算
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_num = input_df[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            input_scaled = scaler.transform(input_num)

            nn = NearestNeighbors(n_neighbors=min(10, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()

            # --- 表示 ---
            st.subheader("🏁 総合審査報告書")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("AI予測デフォルト確率", f"{proba * 100:.4f} %")
                st.caption("モデルは非常に楽観的な判定をしています")

            with c2:
                risk_pct = similar_cases['LoanStatus'].mean() * 100
                st.metric("同業種・同条件の実績リスク", f"{risk_pct:.1f} %")
                if risk_pct > 10: st.error("🚨 類似事例で高い事故率")
                elif risk_pct > 0: st.warning("⚠️ 類似事例に失敗あり")

            with c3:
                # 乖離度を判定
                if risk_pct > 5 and proba < 0.001:
                    st.error("❌ 判定不一致")
                    st.caption("AIは安全としていますが、過去の同業種実績は危険です")
                else:
                    st.success("✅ 判定概ね一致")

            st.divider()
            
            # ハイライト表示
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
            display_cols = ['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector', 'BusinessAge', 'Subprogram']
            
            st.write("### 📂 属性が近い類似事例 (同業種優先)")
            st.dataframe(
                similar_cases[display_cols].style.apply(
                    lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
                ), use_container_width=True
            )

    except Exception as e:
        st.error(f"システムエラー: {e}")
