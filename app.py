import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：究極完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 2. リソースの読み込み
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

# 3. 入力フォーム（サイドバー）
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

        # --- 類似事例検索 (50件・重み付け) ---
        if not train_df.empty:
            filtered_df = train_df[train_df['NaicsSector'] == str(sector)].copy()
            search_pool = filtered_df if len(filtered_df) >= 50 else train_df.copy()

            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_num = input_df[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # 【補正】検索時の「期間」への依存を下げ、「金額」の近さを重視
            train_num_w = train_num.copy(); train_num_w['GrossApproval'] *= 1.5; train_num_w['TermInMonths'] *= 0.5
            input_num_w = input_num.copy(); input_num_w['GrossApproval'] *= 1.5; input_num_w['TermInMonths'] *= 0.5

            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_num_w)
            input_scaled = scaler.transform(input_num_w)

            nn = NearestNeighbors(n_neighbors=min(50, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

        # --- 実効リスク指数の計算 (実績重視90%) ---
        risk_index = (proba * 10) + (risk_pct / 100 * 0.9)
        risk_index = min(risk_index, 1.0)

        # --- 表示：上部メトリクス ---
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            if risk_index < 0.05: st.success("総合判定: ✅ 安全")
            elif risk_index < 0.15: st.warning("総合判定: ⚠️ 注意")
            else: st.error("総合判定: 🚨 慎重検討")
        with c2:
            st.metric("近傍の実績事故率", f"{risk_pct:.1f} %")
            st.write(f"🔍 類似50件中、デフォルトは **{def_count}件**")
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")
            st.caption("統計モデル上の完済パターン一致率")

        st.divider()
        
        # --- 【究極補正】主要因の表示調整 ---
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({'項目': expected_features, 'raw_imp': importances})
        
        # 数学的な重み調整：独走するTermの重みを削り、Gross等を底上げする
        imp_df['adj_imp'] = imp_df['raw_imp']
        imp_df.loc[imp_df['項目'] == 'TermInMonths', 'adj_imp'] *= 0.5  # 期間の影響を半分に
        imp_df.loc[imp_df['項目'] == 'GrossApproval', 'adj_imp'] *= 1.8 # 金額の影響を1.8倍に
        imp_df.loc[imp_df['項目'] == 'NaicsSector', 'adj_imp'] *= 1.3  # 業種を1.3倍に
        
        # 合計を100%に再計算
        total = imp_df['adj_imp'].sum()
        imp_df['影響度(%)'] = (imp_df['adj_imp'] / total * 100).round(2)
        display_imp = imp_df.sort_values('影響度(%)', ascending=False).head(5)[['項目', '影響度(%)']]

        col_imp, col_tips = st.columns([1, 1])
        with col_imp:
            st.write("### ⚖️ 判断の主要構成要素 (実務補正済)")
            st.table(display_imp)
            

       with col_tips:
            st.write("### 📝 審査のアドバイス")
            
            # 【重要】AIの確信度より「目の前の実績(risk_pct)」を優先して喋らせる
            if risk_pct > 50:
                st.error(f"🚨 **【極めて危険】** 類似事例の {risk_pct:.1f}% がデフォルトしています。AIは完済パターンに近いと判断していますが、この業種・規模の直近の実績は最悪です。否決を強く推奨します。")
            elif risk_pct > 20:
                st.error(f"⚠️ **【高リスク】** 類似事例の約4～5件に1件で事故が発生。AIの予測以上に現場の状況はシビアです。赤色の行を精査してください。")
            elif (1-proba) > 0.4 and risk_pct < 5:
                st.warning(f"🤔 **【AI慎重・実績良好】** AIは警戒していますが、類似事例の実績は極めて安全です。担保が十分なら承認の余地があります。")
            else:
                st.success("✅ **【良好】** AIの予測と現場の実績が一致しています。自信を持って進められる案件です。")
        # 事例詳細
        st.write("### 📂 属性が近い類似事例 (赤色はデフォルト)")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        st.dataframe(
            similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector', 'BusinessAge']].style.apply(
                lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
            ), use_container_width=True
        )

    except Exception as e:
        st.error(f"エラー: {e}")
