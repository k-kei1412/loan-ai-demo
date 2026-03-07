import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：究極完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# NAICSコード辞書
NAICS_MAP = {
    "11": "農業・林業・漁業", "21": "鉱業・採掘", "22": "電気・ガス・水道",
    "23": "建設業", "31": "製造業", "42": "卸売業", "44": "小売業",
    "48": "運輸・倉庫業", "51": "情報通信業", "52": "金融・保険業",
    "53": "不動産・賃貸業", "54": "専門・技術サービス", "56": "支援・廃棄物処理",
    "61": "教育サービス", "62": "医療・社会福祉", "71": "娯楽・レクリエーション",
    "72": "宿泊・飲食サービス", "81": "その他サービス", "92": "公務"
}

# 2. リソースの読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        train_df = pd.read_csv("train.csv")
        # 読み込み時に型を安定させる
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
    sector_code = st.selectbox("産業セクター", options=list(NAICS_MAP.keys()), format_func=lambda x: f"{x}: {NAICS_MAP[x]}", index=6)
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
        "TermInMonths": float(term), "NaicsSector": str(sector_code), 
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

        # --- 類似事例検索 ---
        if not train_df.empty:
            filtered_df = train_df[train_df['NaicsSector'].str.startswith(str(sector_code))].copy()
            search_pool = filtered_df if len(filtered_df) >= 50 else train_df.copy()
            
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_num = input_df[search_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # 重み付け
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

        # --- 影響度計算 (日本語化) ---
        importances = model.get_feature_importance()
        name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資総額", "InitialInterestRate": "初期金利", "NaicsSector": "産業セクター", "SBAGuaranteedApproval": "SBA保証額"}
        imp_df = pd.DataFrame({'項目': [name_map.get(f, f) for f in expected_features], 'raw_imp': importances})
        
        imp_df['adj_imp'] = imp_df['raw_imp']
        imp_df.loc[imp_df['項目'] == '返済期間', 'adj_imp'] *= 0.5
        imp_df.loc[imp_df['項目'] == '融資総額', 'adj_imp'] *= 1.8
        imp_df.loc[imp_df['項目'] == '産業セクター', 'adj_imp'] *= 1.3
        
        total = imp_df['adj_imp'].sum()
        imp_df['影響度(%)'] = (imp_df['adj_imp'] / total * 100).round(2)
        display_imp = imp_df.sort_values('影響度(%)', ascending=False).head(5)[['項目', '影響度(%)']]

        # --- リスク指数 ---
        risk_index = (proba * 10) + (risk_pct / 100 * 0.9)
        risk_index = min(risk_index, 1.0)

        # --- 表示 ---
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
            if risk_index < 0.15: 
                st.warning("総合判定: ⚠️ 注意") if risk_index >= 0.05 else st.success("総合判定: ✅ 安全")
            else:
                st.error("総合判定: 🚨 慎重検討（否決推奨）")
        with c2:
            st.metric("近傍の実績事故率", f"{risk_pct:.1f} %")
            st.markdown(f"🔍 類似50件（**{NAICS_MAP[sector_code]}**）中、デフォルトは **{def_count}件**")
        with c3:
            st.metric("AI完済確信度", f"{(1-proba)*100:.1f} %")

        st.divider()
        col_imp, col_tips = st.columns([1, 1])
        with col_imp:
            st.write("### ⚖️ 判断の主要構成要素 (実務補正済)")
            st.table(display_imp)

        with col_tips:
            st.write("### 📝 審査のアドバイス")
            if def_count >= 15:
                st.error(f"🚨 **【最優先警告】地雷原エリア**")
                st.write(f"この業種（{NAICS_MAP[sector_code]}）の類似案件は3割以上が不履行です。AIがどう判断しようと、この領域での貸付は極めて高いリスクを伴います。")
            elif def_count >= 5:
                st.error(f"⚠️ **【警戒】実績リスク高騰**")
                st.write(f"50件中 {def_count} 件の事故を確認。現場の状況はAIの予測以上にシビアです。")
            else:
                st.success("✅ **【良好】データ上の懸念なし**")
                st.write("類似事例に事故はほとんど見られません。AIの確信度も踏まえ、前向きに検討可能な案件です。")

        # --- 事例詳細 (None対策) ---
        st.write(f"### 📂 属性が近い類似事例 (業種: {NAICS_MAP[sector_code]})")
        similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        
        # 【修正】Noneにならないよう、確実に整数に直してからマップする
        similar_cases['業種名'] = similar_cases['NaicsSector'].apply(lambda x: NAICS_MAP.get(str(int(float(x))), "不明"))
        
        display_cols = ['結果', 'GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate', 'TermInMonths', '業種名']
        st.dataframe(
            similar_cases[display_cols].style.apply(
                lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
            ), use_container_width=True
        )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
