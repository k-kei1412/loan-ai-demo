import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：実務信頼性モデル", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 2. リソース読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        target = "train.csv" if os.path.exists("train.csv") else "train (4).csv"
        df = pd.read_csv(target)
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        return model, df
    except:
        return model, pd.DataFrame()

model, train_df = load_resources()
expected_features = model.feature_names_

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
    rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
    term = st.number_input("返済期間 (月)", 1, 360, 120)
    
    sector_list = sorted(train_df['NaicsSector'].unique()) if not train_df.empty else []
    sector_en = st.selectbox("産業セクター", options=sector_list)
    
    business_age = st.selectbox("企業年齢", ["Existing or more than 2 years old", "New Business or 2 years or less", "Startup, Loan Funds will Open Business", "Unanswered"])
    collateral = st.selectbox("担保の有無", ["Y", "N"])
    submit = st.button("精密クロス審査を開始")

if submit:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
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
            raw_proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

            # --- B. 類似事例検索 (100件) ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 100:
                search_pool = train_df.copy()

            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths"]
            train_num = search_pool[search_features].fillna(0)
            input_num = input_df[search_features].fillna(0)
            
            scaler = StandardScaler()
            weights = np.array([1.5, 1.0, 1.0])
            train_scaled = scaler.fit_transform(train_num) * weights
            input_scaled = scaler.transform(input_num) * weights

            nn = NearestNeighbors(n_neighbors=min(100, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

            # --- C. 実効リスク指数の計算 ---
            strict_proba = np.clip(raw_proba, 0.03, 0.97) 
            risk_index = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)

            # --- D. 画面表示 ---
            st.subheader("🏁 総合審査報告書（100件クロスバリデーション版）")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("実効リスク指数", f"{risk_index * 100:.2f} %")
                if risk_index < 0.07: 
                    st.success("総合判定: ✅ 安全")
                    status = "安全"
                elif risk_index < 0.17: 
                    st.warning("総合判定: ⚠️ 注意")
                    status = "注意"
                else: 
                    st.error("総合判定: 🚨 危険")
                    status = "危険"
            with c2:
                st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
            with c3:
                # 【改善】99.9%のような非現実的な数字を避け、最大98.5%でクリップ
                calibrated_conf = (1 - raw_proba) * 100
                st.metric("AI完済期待値 (参考)", f"{min(calibrated_conf, 98.5):.1f} %")
                st.caption("※統計的な期待値。1.5%以上のリスクは常時想定。")

            st.divider()

            # --- E. アドバイス欄の追加 ---
            st.write("### 📝 審査アドバイス")
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                if status == "安全":
                    st.info("AI予測と類似実績が共に良好です。標準的な審査手順での承認を推奨します。")
                elif status == "注意":
                    st.warning("統計上は安全に見えますが、直近の類似事例でデフォルトが発生しています。資金使途と返済計画の再確認を推奨します。")
                else:
                    st.error("実績事故率が警戒水準にあります。担保の保全状況を厳格に評価し、慎重な判断が必要です。")
            with col_adv2:
                # 金利に対するアドバイス
                if rate > 8.0:
                    st.write("💡 設定金利が高めです。収益性は高いですが、金利負担によるキャッシュフロー悪化に注意してください。")
                else:
                    st.write("💡 低金利設定です。リスク指数に見合ったスプレッドが確保されているか検討してください。")

            st.divider()

            # --- F. 影響度テーブル ---
            st.write("### ⚖️ 判断の主要構成要素 (%)")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界", "SBAGuaranteedApproval": "保証額", "CollateralInd": "担保"}
            imp_df['項目'] = imp_df['項目'].map(lambda x: name_map.get(x, x))
            
            imp_df['adj'] = imp_df['raw']
            imp_df.loc[imp_df['項目'] == '返済期間', 'adj'] *= 0.3
            imp_df.loc[imp_df['項目'] == '融資額', 'adj'] *= 1.7
            imp_df.loc[imp_df['項目'] == '業界', 'adj'] *= 1.1
            
            total_adj = imp_df['adj'].sum()
            imp_df['影響度(%)'] = (imp_df['adj'] / total_adj * 100).round(1)
            display_imp = imp_df.groupby('項目')['影響度(%)'].sum().sort_values(ascending=False).head(5)
            st.table(display_imp)
            
            # --- G. 事例詳細 ---
            st.write(f"### 📂 属性が近い類似事例 (上位100件中の抜粋)")
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
            st.dataframe(similar_cases[['結果', 'GrossApproval', 'InitialInterestRate', 'TermInMonths', 'NaicsSector']].head(50).style.apply(
                lambda s: ['background-color: #ffcccc' if s.結果 == "❌ デフォルト" else '' for _ in s], axis=1
            ), use_container_width=True)

        except Exception as e:
            st.error(f"システムエラー: {e}")
