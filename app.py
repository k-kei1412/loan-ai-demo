# --- 前半のインポートと基本設定はそのまま ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：真・完全体", layout="wide")
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# --- 文字化け・フォント対策 ---
def set_japanese_font():
    fonts = ['Heiti TC', 'MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
    for f in fonts:
        if f in [font.name for font in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = f
            break
    plt.rcParams['axes.unicode_minus'] = False 

set_japanese_font()

# --- 業界定義（解析でも使うため共通変数として定義） ---
sectors_map = {
    "accommodation": "宿泊・飲食サービス業", "administrative": "運営支援・廃棄物処理",
    "agriculture": "農業・林業・漁業", "arts": "芸術・娯楽・レクリエーション",
    "construction": "建設業", "educational": "教育サービス業",
    "finance": "金融業・保険業", "health": "医療・福祉",
    "information": "情報通信業", "management": "企業管理・持株会社",
    "manufacturing": "製造業", "mining": "採鉱・石油ガス採掘",
    "professional": "専門・科学・技術サービス", "public": "公務",
    "real estate": "不動産・賃貸業", "retail": "小売業",
    "transportation": "運輸業・倉庫業", "utilities": "公益事業", "wholesale": "卸売業"
}

sector_vix_map = {
    "accommodation": 45, "administrative": 40, "agriculture": 23, 
    "arts": 50, "construction": 30, "educational": 25,
    "finance": 23, "health": 20, "information": 55, 
    "management": 60, "manufacturing": 30, "mining": 30,
    "professional": 45, "public": 18, "real estate": 40, 
    "retail": 40, "transportation": 30, "utilities": 18, "wholesale": 30
}

# 2. リソース読み込み
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        target = "train.csv" if os.path.exists("train.csv") else "train (4).csv"
        df = pd.read_csv(target)
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        df['SBA_Ratio'] = (df['SBAGuaranteedApproval'] / df['GrossApproval']).fillna(0)
        return model, df, target
    except:
        return model, pd.DataFrame(), "None"

model, train_df, file_name = load_resources()
expected_features = model.feature_names_

# --- 名前マップの定義 ---
graph_name_map = {
    "TermInMonths": "Loan Term", "GrossApproval": "Loan Amount", "InitialInterestRate": "Interest Rate", 
    "NaicsSector": "Industry Sector", "SBAGuaranteedApproval": "SBA Guaranty", "CollateralInd": "Collateral",
    "ApprovalFiscalYear": "Fiscal Year", "Subprogram": "Subprogram", "FixedOrVariableInterestInd": "Rate Type",
    "BusinessAge": "Business Age", "CongressionalDistrict": "District", "BusinessType": "Business Type", "JobsSupported": "Jobs Created"
}

table_name_map = {
    "TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", 
    "NaicsSector": "業界セクター", "SBAGuaranteedApproval": "保証額", "CollateralInd": "担保有無",
    "BusinessAge": "事業歴", "BusinessType": "法人形態", "JobsSupported": "雇用創出数",
    "Subprogram": "支援プログラム", "FixedOrVariableInterestInd": "金利タイプ", "CongressionalDistrict": "選挙区(地域)"
}

def get_japanese_sector(en_text):
    text = str(en_text).lower()
    if "other" in text: return "その他サービス業"
    for k, v in sectors_map.items():
        if k in text: return v
    return en_text

if "clicked" not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

# --- サイドバー入力 ---
st.sidebar.header("📋 申請者情報入力")
app_mode = st.sidebar.radio("📊 表示モード切替", ["総合報告 (表面)", "高度解析 (裏面)"])

with st.sidebar:
    st.divider()
    gross = st.number_input("融資額 ($)", 0, 10000000, 500000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 300000)
    rate = st.number_input("金利 (%)", 0.0, 35.0, 15.0)
    term = st.number_input("返済期間 (月)", 1, 360, 84)
    
    b_age = st.selectbox("事業歴", ["2年以上 (Existing)", "2年未満 (New Business)"])
    b_age_val = "Existing or more than 2 years old" if "2年以上" in b_age else "New Business or less than 2 years old"
    
    b_type = st.selectbox("法人形態", ["株式会社 (CORPORATION)", "個人事業主 (INDIVIDUAL)", "パートナーシップ (PARTNERSHIP)"])
    b_type_val = b_type.split("(")[1].replace(")", "")
    
    # 産業セクターの選択（修正ポイント：一箇所に統合）
    if not train_df.empty:
        unique_en_sectors = sorted(train_df['NaicsSector'].unique())
        display_options = [get_japanese_sector(s) for s in unique_en_sectors]
        selected_jp = st.selectbox("産業セクター", options=display_options)
        sector_en = unique_en_sectors[display_options.index(selected_jp)]
        
        # 選択された業種の英語キーを逆引きして標準値を表示
        vix_key = ""
        for k, v in sectors_map.items():
            if v == selected_jp:
                vix_key = k
                break
        
        standard_vix = sector_vix_map.get(vix_key, 30)
        st.info(f"💡 この業界の標準ボラティリティは **{standard_vix}%** です")
    else:
        sector_en = "Finance_insurance"
        st.selectbox("産業セクター", options=["データ未読み込み"])

    jobs = st.slider("現在の雇用員数", 0, 500, 5)
    rate_type = st.radio("金利タイプ", ["変動金利 (V)", "固定金利 (F)"])
    rate_type_val = "V" if "変動" in rate_type else "F"

    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    
    submit = st.button("精密クロス審査を開始", on_click=click_button)

# 4. 分析ロジック
if st.session_state.clicked:
    if train_df.empty:
        st.error("学習データが見つかりません。")
    else:
        try:
            current_sba_ratio = sba / gross if gross > 0 else 0
            
            # --- 入力データの構築 ---
            input_data = {
                "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
                "InitialInterestRate": float(rate), "TermInMonths": float(term),
                "NaicsSector": str(sector_en), "ApprovalFiscalYear": 2024.0, "Subprogram": "Guaranty",
                "FixedOrVariableInterestInd": rate_type_val, "CongressionalDistrict": 10.0,
                "BusinessType": b_type_val, "BusinessAge": b_age_val,
                "RevolverStatus": 0.0, "JobsSupported": float(jobs), "CollateralInd": str(collateral_val)
            }
            
            input_df = pd.DataFrame([input_data])
            for col in expected_features:
                if col not in input_df.columns: input_df[col] = 0.0
            input_df = input_df[expected_features]
            
            cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
            preds = model.predict_proba(Pool(input_df, cat_features=cat_idx))
            raw_proba = preds[0][1] if len(preds) > 0 else 0.5

            # --- 類似事例検索 ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 10: 
                search_pool = train_df.copy()
            
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
            train_num = search_pool[search_features].fillna(0).copy()
            train_num["TermInMonths"] = np.log1p(train_num["TermInMonths"])
            
            input_num = input_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]].copy()
            input_num["TermInMonths"] = np.log1p(input_num["TermInMonths"])
            input_num["SBA_Ratio"] = current_sba_ratio
            
            scaler = StandardScaler()
            weights = np.array([1.2, 1.0, 1.5, 2.0]) 
            train_scaled = scaler.fit_transform(train_num) * weights
            input_scaled = scaler.transform(input_num) * weights
            
            n_neighbors_val = min(100, len(search_pool))
            nn = NearestNeighbors(n_neighbors=n_neighbors_val)
            nn.fit(train_scaled)
            distances, indices = nn.kneighbors(input_scaled)
            
            if len(indices) > 0 and len(indices[0]) > 0:
                similar_cases = search_pool.iloc[indices[0]].copy()
                risk_pct = similar_cases['LoanStatus'].mean() * 100
                def_count = int(similar_cases['LoanStatus'].sum())
            else:
                risk_pct, def_count = 0, 0
                similar_cases = pd.DataFrame()

            # --- リスク指標・ロジック計算 ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (np.log1p(term) - np.log1p(dynamic_ceil))) * 2.0 if term > dynamic_ceil else 0.0
            
            sba_bonus_flag = (current_sba_ratio >= 0.80)
            gross_risk = 0.0
            if gross >= 1000000: gross_risk = 0.40 + (gross - 1000000) / 1000000
            elif gross > 500000: gross_risk = ((gross - 500000) // 100000) * 0.04
            if sba_bonus_flag: gross_risk *= 0.5
            
            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3 + (0.1 if rate > 20.0 else 0)
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            if app_mode == "総合報告":
                # --- 表面の表示内容はそのまま ---
                st.subheader("🏁 総合審査報告書")
                st.write("### 🔍 実務者への重点確認事項")
                
                warnings = []
                if gross >= 1000000:
                    status = "危険"
                    st.error("🚨 **【最重要精査案件】** 融資額が $1M を超過。役員承認が必須。")
                    warnings.append("・100万ドル超の高額融資")
                elif gross >= 500000 and rate >= 20.0 and not sba_bonus_flag:
                    status = "注意"
                    st.error("💀 **【複合リスク】** 高額かつ高金利。80%以上の保証がないため警戒。")
                    warnings.append("・高額かつ高金利の無防備案件")
                else:
                    status = "安全" if final_expected_success > 92 else "注意" if final_expected_success > 75 else "危険"

                if sba_bonus_flag:
                    st.success(f"🛡️ **【保全インセンティブ適用】** 保証率80%超により高額融資リスクを50%軽減。")
                if 500000 <= gross < 1000000:
                    st.info(f"📂 **【中規模案件】** 50万ドル超の中堅企業向け融資。リスク加重適用中。")
                if term > dynamic_ceil:
                    st.warning(f"⏳ **【期間超過】** 適正上限（{int(dynamic_ceil)}ヶ月）を超過。デフォルト確率が上昇傾向。")
                    warnings.append("・返済期間の超過")
                if b_age_val == "New Business or less than 2 years old":
                    st.warning(f"🌱 **【新規事業リスク】** 創業2年未満。キャッシュフローの安定性を要確認。")
                    warnings.append("・新規事業（業歴2年未満）")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                    if status == "安全": st.success("総合判定: ✅ 安全")
                    elif status == "注意": st.warning("総合判定: ⚠️ 注意")
                    else: st.error("総合判定: 🚨 危険 (要精査)")
                    for w in warnings: st.caption(f":orange[{w}]" if status == "注意" else f":red[{w}]")
                with c2:
                    st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                    st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
                with c3:
                    st.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

                st.write("### 💡 審査改善へのアクション案")
                with st.expander("アドバイスの詳細を確認する", expanded=True):
                    advice = []
                    if gross >= 1000000: advice.append("⚠️ **金額の再検討**: 可能であれば分割融資または担保の積み増しを検討してください。")
                    if term > dynamic_ceil: advice.append(f"✅ **期間の最適化**: {int(dynamic_ceil)}ヶ月以下への短縮により、リスク指数が大幅に改善します。")
                    if current_sba_ratio < 0.80: advice.append("✅ **保証枠の拡大**: SBA保証を80%以上に引き上げると、内部リスク加重が緩和されます。")
                    if b_age_val == "New Business or less than 2 years old": advice.append("📊 **補足資料**: 創業計画書および今後3年間の収支予測の厳密な精査を推奨。")
                    
                    if not advice: st.write("✨ 現在の条件は論理的に非常に安定しています。")
                    else:
                        for a in advice: st.write(a)

                st.divider()
                st.write("### ⚖️ 判断に影響した主要要素")
                importances = model.get_feature_importance()
                imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
                imp_df['項目名'] = imp_df['項目'].map(lambda x: table_name_map.get(x, x))
                display_imp = imp_df.groupby('項目名')['raw'].sum().reset_index()
                display_imp['影響度(%)'] = (display_imp['raw'] / display_imp['raw'].sum() * 100).round(1)
                st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

                st.divider()
                st.write("### 👥 条件が近い過去の事例（比較解析）")
                current_row = pd.DataFrame({"状況": ["⭐ 今回の申請条件"], "融資額": [f"${gross:,}"], "金利": [f"{rate}%"], "返済期間": [f"{term}ヶ月"], "LoanStatus": [-1]})
                display_similar = similar_cases.head(100).copy()
                display_similar['状況'] = display_similar['LoanStatus'].map({0: "✅ 完済", 1: "❌ 不履行"})
                display_similar['融資額'] = display_similar['GrossApproval'].map(lambda x: f"${x:,.0f}")
                display_similar['金利'] = display_similar['InitialInterestRate'].map(lambda x: f"{x}%")
                display_similar['返済期間'] = display_similar['TermInMonths'].map(lambda x: f"{x}ヶ月")
                merged_display = pd.concat([current_row, display_similar[["状況", "融資額", "金利", "返済期間", "LoanStatus"]]], ignore_index=True)

                def style_row(row):
                    if row['LoanStatus'] == -1: return ['background-color: #e1f5fe; font-weight: bold'] * len(row)
                    elif row['LoanStatus'] == 1: return ['background-color: #ffebee; color: #c62828'] * len(row)
                    return [''] * len(row)
                st.dataframe(merged_display.style.apply(style_row, axis=1), column_order=("状況", "融資額", "金利", "返済期間"), use_container_width=True)

            else:
                # --- 高度解析 ---
                st.header("🔬 数理解析")
                st.write("#### ⚖️ AIの判断根拠 (SHAP解析)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                shap_values.values = -shap_values.values 
                shap_values.feature_names = [graph_name_map.get(n, n) for n in expected_features]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                labels = [t.get_text() for t in ax.get_yticklabels()]
                new_labels = [l.split(" = ")[1] if " = " in l else l for l in labels]
                ax.set_yticklabels(new_labels)
                plt.xlabel("Contribution to Full Repayment")
                st.pyplot(plt.gcf(), clear_figure=True)
                
                st.divider()
                st.write("#### 📉 理論的倒産距離 (Merton Model)")
                vol = st.slider("想定資産ボラティリティ (%)", 10, 100, 30) / 100
                asset = float(gross) * 1.5
                t_m = float(term) / 12
                dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
                edf = stats.norm.cdf(-dd) * 100
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.metric("倒産距離 (DD)", f"{dd:.2f}")
                    st.metric("デフォルト確率 (EDF)", f"{edf:.2f} %")
                with c_m2:
                    x = np.linspace(-4, 4, 100); y = stats.norm.pdf(x, 0, 1)
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    ax2.plot(x, y, color="gray"); ax2.fill_between(x, y, where=(x < -dd), color='red', alpha=0.5)
                    st.pyplot(fig2)

                with st.expander("📚 専門用語の解説：デフォルト確率と倒産距離", expanded=True):
                    st.write("""
                    **1. 倒産距離 (Distance to Default: DD)**
                    企業の資産価値が、負債（融資額）の支払境界線からどれだけ離れているかを「標準偏差」の単位で表したもの。
                    - 数値が高いほど安全。一般的に **2.0以上** が優良基準。

                    **2. デフォルト確率 (Expected Default Frequency: EDF)**
                    マートン・モデルに基づき、将来的に企業の資産価値が負債額を下回る確率。
                    - 上図の赤色の領域がEDF。
                    """)

                st.divider()
                st.write("#### 🧪 金利感度シミュレーション")
                sim_rates = np.linspace(5.0, 30.0, 15)
                sim_probs = [100 * (1 - model.predict_proba(Pool(input_df.assign(InitialInterestRate=r), cat_features=cat_idx))[0][1]) for r in sim_rates]
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(sim_rates, sim_probs, '-o', color="#0078D4", linewidth=2)
                ax3.axvline(x=rate, color='red', linestyle='--')
                ax3.set_xlabel("Interest Rate (%)")
                ax3.set_ylabel("Expected Success (%)")
                st.pyplot(fig3)

        except Exception as e:
            st.error(f"分析エラー: {e}")
