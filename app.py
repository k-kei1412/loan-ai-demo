import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：真・完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 2. リソース読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        target = "train.csv" if os.path.exists("train.csv") else "train (4).csv"
        df = pd.read_csv(target)
        # 確実に文字列型に変換
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        df['SBA_Ratio'] = (df['SBAGuaranteedApproval'] / df['GrossApproval']).fillna(0)
        return model, df, target
    except:
        return model, pd.DataFrame(), "None"

model, train_df, file_name = load_resources()
expected_features = model.feature_names_

# 3. 業界セクター：正規化対応マッピング
# キーを「小文字・スペースなし」で統一して定義します
RAW_MAPPING = {
    "accommodationfoodservices": "宿泊・飲食サービス業",
    "administrativesupportwastemanagementremediationservices": "運営支援・廃棄物処理",
    "agricultureforestryfishinghunting": "農業・林業・漁業・狩猟業",
    "artsentertainmentrecreation": "芸術・娯楽・レクリエーション",
    "construction": "建設業",
    "educationalservices": "教育サービス業",
    "financeinsurance": "金融業・保険業",
    "healthcaresocialassistance": "医療・福祉",
    "information": "情報通信業",
    "managementofcompaniesenterprises": "企業管理・持株会社",
    "manufacturing": "製造業",
    "miningquarryingoilgasextraction": "採鉱・石油ガス採掘",
    "professionalscientifictechnicalservices": "専門・科学・技術サービス",
    "publicadministration": "公務",
    "realestaterentalleasing": "不動産・賃貸業",
    "retailtrade": "小売業",
    "transportationwarehousing": "運輸業・倉庫業",
    "utilities": "公益事業（電気・ガス・水道）",
    "wholesaletrade": "卸売業"
    "otherservicesexceptpublicadministration": "その他サービス業",
}

def get_japanese_sector(en_text):
    """
    判定の優先順位を調整した最強版。
    Other services が Public administration に横取りされるのを防ぎます。
    """
    text = str(en_text).lower()
    
    # 1. まず「Other」を最優先でチェック（長い名前に Public が含まれるため）
    if "other" in text: return "その他サービス業"
    
    # 2. その他のキーワードチェック
    if "accommodation" in text: return "宿泊・飲食サービス業"
    if "administrative" in text: return "運営支援・廃棄物処理"
    if "agriculture" in text: return "農業・林業・漁業"
    if "arts" in text: return "芸術・娯楽・レクリエーション"
    if "construction" in text: return "建設業"
    if "educational" in text: return "教育サービス業"
    if "finance" in text: return "金融業・保険業"
    if "health" in text: return "医療・福祉"
    if "information" in text: return "情報通信業"
    if "management" in text: return "企業管理・持株会社"
    if "manufacturing" in text: return "製造業"
    if "mining" in text: return "採鉱・石油ガス採掘"
    if "professional" in text: return "専門・科学・技術サービス"
    if "public" in text: return "公務"
    if "real estate" in text or "real_estate" in text: return "不動産・賃貸業"
    if "retail" in text: return "小売業"
    if "transportation" in text: return "運輸業・倉庫業"
    if "utilities" in text: return "公益事業（電気・ガス・水道）"
    if "wholesale" in text: return "卸売業"
    
    return en_text
# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
with st.sidebar:
    gross = st.number_input("融資額 ($)", 0, 10000000, 700000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 400000)
    rate = st.number_input("金利 (%)", 0.0, 35.0, 15.0)
    term = st.number_input("返済期間 (月)", 1, 360, 95)
    
    if not train_df.empty:
        # データにある生の英語名一覧
        unique_en_sectors = sorted(train_df['NaicsSector'].unique())
        
        # 表示用に日本語変換（辞書を直接引かず、正規化関数を通す）
        display_options = [get_japanese_sector(s) for s in unique_en_sectors]
        
        selected_jp = st.selectbox("産業セクター", options=display_options)
        
        # 逆引き用：選ばれた日本語から「元の生英語名」を特定する
        # （表示リストと元リストのインデックスを合わせる）
        sector_idx = display_options.index(selected_jp)
        sector_en = unique_en_sectors[sector_idx]
    else:
        sector_en = "Finance_insurance"
        st.selectbox("産業セクター", options=["データ未読み込み"])

    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    submit = st.button("精密クロス審査を開始")

# 4. 分析ロジック
if submit:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            current_sba_ratio = sba / gross if gross > 0 else 0
            
            # --- A. AI予測 ---
            input_data = {
                "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
                "InitialInterestRate": float(rate), "TermInMonths": float(term),
                "NaicsSector": sector_en, "ApprovalFiscalYear": 2024.0, "Subprogram": "Guaranty",
                "FixedOrVariableInterestInd": "V", "CongressionalDistrict": 10.0,
                "BusinessType": "CORPORATION", "BusinessAge": "Existing or more than 2 years old",
                "RevolverStatus": 0.0, "JobsSupported": 5.0, "CollateralInd": collateral_val
            }
            input_df = pd.DataFrame([input_data]).reindex(columns=expected_features)
            cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
            raw_proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

            # --- B. 類似事例検索 ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 100: search_pool = train_df.copy()
            
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
            train_num = search_pool[search_features].fillna(0)
            input_num = input_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]].copy()
            input_num["SBA_Ratio"] = current_sba_ratio
            
            scaler = StandardScaler()
            weights = np.array([1.2, 1.0, 1.0, 2.0]) 
            train_scaled = scaler.fit_transform(train_num) * weights
            input_scaled = scaler.transform(input_num) * weights
            nn = NearestNeighbors(n_neighbors=min(100, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

            # --- C. リスク正規化ロジック ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (term - dynamic_ceil) / 100.0) * 0.7 if term > dynamic_ceil else 0.0
            
            gross_risk = 0.0
            sba_bonus_flag = False
            if gross >= 1000000: gross_risk = 0.40 + (gross - 1000000) / 1000000
            elif gross > 500000: gross_risk = ((gross - 500000) // 100000) * 0.04
            if current_sba_ratio >= 0.80:
                gross_risk *= 0.5
                sba_bonus_flag = True
                
            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
            combined_risk = np.clip(combined_risk, 0.02, 0.99)
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            # --- D. 表示セクション ---
            st.subheader("🏁 総合審査報告書")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                if final_expected_success > 92: st.success("総合判定: ✅ 安全")
                elif final_expected_success > 75: st.warning("総合判定: ⚠️ 注意")
                else: st.error("総合判定: 🚨 危険")
            with c2:
                st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                st.write(f"🔍 不履行事例: **{def_count}件**")
            with c3:
                st.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

            # --- E. 判断要素の割合表示 ---
            st.divider()
            st.write("### ⚖️ 判断に影響した主要要素")
            importances = model.get_feature_importance()
            imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
            name_map = {"TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", "NaicsSector": "業界セクター", "SBAGuaranteedApproval": "保証額"}
            imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
            
            # 重要度調整
            imp_df['adj'] = imp_df['raw']
            imp_df.loc[imp_df['項目'] == 'TermInMonths', 'adj'] *= 0.23
            imp_df.loc[imp_df['項目'] == 'GrossApproval', 'adj'] *= 1.7
            
            display_imp = imp_df[imp_df['項目名'] != "その他"].copy()
            total_adj = display_imp['adj'].sum()
            display_imp['影響度(%)'] = (display_imp['adj'] / total_adj * 100).round(1)
            
            st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

            st.divider()
            # 類似事例比較
            my_data = pd.DataFrame([{"結果": "📢 今回の申請", "融資額": gross, "保証率": f"{current_sba_ratio*100:.1f}%", "金利": rate, "期間": term}])
            similar_cases['結果'] = similar_cases['LoanStatus'].apply(lambda x: "❌ 不履行" if x == 1 else "✅ 完済")
            similar_cases['融資額'] = similar_cases['GrossApproval']
            similar_cases['保証率'] = (similar_cases['SBA_Ratio'] * 100).map('{:.1f}%'.format)
            similar_cases['金利'] = similar_cases['InitialInterestRate']
            similar_cases['期間'] = similar_cases['TermInMonths']
            st.dataframe(pd.concat([my_data, similar_cases[['結果', '融資額', '保証率', '金利', '期間']].head(10)]), use_container_width=True)

        except Exception as e:
            st.error(f"分析エラー: {e}")
