import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="次世代ローン審査AI：完全体", layout="wide")
st.title("🏦 ローンデフォルト予測 AIシステム [究極完全体]")

# 2. リソースの読み込み
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

# 3. サイドバー：入力フォーム
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
    submit = st.button("精密審査を開始")

# 4. 審査・検索ロジック
if submit:
    # データ変換
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
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce') if col in numeric_cols else input_df[col].astype(str)

    try:
        # --- AI予測 ---
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        pool = Pool(input_df, cat_features=cat_idx)
        proba = model.predict_proba(pool)[0][1]

        # --- 表示1: AI判定スコア ---
        st.subheader("🏁 総合審査結果")
        c1, c2, c3 = st.columns(3)
        with c1:
            # AI予測は保守的に表示
            st.metric("AI予測デフォルト確率", f"{proba * 100:.4f} %")
            if proba < 0.005: st.success("AI判定: 極めて安全")
            else: st.warning("AI判定: 慎重に検討")

        # --- 表示2: 高精度・多角的類似事例検索 ---
        if not train_df.empty:
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "JobsSupported"]
            train_num = train_df[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_num = input_df[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num)
            input_scaled = scaler.transform(input_num)

            # 検索1: 単純に一番近い10件
            nn = NearestNeighbors(n_neighbors=100)
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = train_df.iloc[indices[0]].copy()

            # 検索2: 過去の「失敗事例(Default=1)」の中から似ている3件を強制抽出
            if 'LoanStatus' in train_df.columns:
                fail_data = train_df[train_df['LoanStatus'] == 1]
                if not fail_data.empty:
                    fail_num = fail_data[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
                    fail_scaled = scaler.transform(fail_num)
                    nn_fail = NearestNeighbors(n_neighbors=min(3, len(fail_data)))
                    nn_fail.fit(fail_scaled)
                    _, fail_indices = nn_fail.kneighbors(input_scaled)
                    fail_cases = fail_data.iloc[fail_indices[0]]
                    # 統合（重複は削除）
                    similar_cases = pd.concat([similar_cases, fail_cases]).drop_duplicates().head(10)

            # 実績リスクの表示
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            with c2:
                st.metric("類似事例の実績リスク率", f"{risk_pct:.1f} %")
                if risk_pct > 0: st.error(f"警告: 類似事例に失敗あり")
                else: st.success("実績判定: 過去に失敗なし")

        # --- 表示3: 判断の決め手 ---
        st.divider()
        st.write("### 💡 判断に影響を与えた項目 (Top 5)")
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({'項目': expected_features, '影響度': importances}).sort_values('影響度', ascending=False).head(5)
        st.table(imp_df.T)

        # --- 表示4: 事例詳細 (ハイライト機能付き) ---
        st.write("### 📂 類似事例の比較リスト")
        st.caption("※赤色の行は過去にデフォルトが発生した事例です。")
        
        # 表示用整形
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        display_cols = ['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector', 'Subprogram']
        
        def highlight_risk(s):
            return ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s]

        st.dataframe(similar_cases[display_cols].style.apply(highlight_risk, axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
