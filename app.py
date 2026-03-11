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

# --- 文字化け対策 ---
def set_japanese_font():
    fonts = ['MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
    for f in fonts:
        if f in [font.name for font in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = f
            break
    plt.rcParams['axes.unicode_minus'] = False 

set_japanese_font()

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

name_map = {
    "TermInMonths": "返済期間", 
    "GrossApproval": "融資額", 
    "InitialInterestRate": "金利", 
    "NaicsSector": "業界セクター", 
    "SBAGuaranteedApproval": "保証額",
    "CollateralInd": "担保有無"
}

# 3. 業界セクター翻訳
def get_japanese_sector(en_text):
    text = str(en_text).lower()
    if "other" in text: return "その他サービス業"
    sectors = {
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
    for k, v in sectors.items():
        if k in text: return v
    return en_text

# --- 状態維持のためのセッション初期化 ---
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
    
    if not train_df.empty:
        unique_en_sectors = sorted(train_df['NaicsSector'].unique())
        display_options = [get_japanese_sector(s) for s in unique_en_sectors]
        selected_jp = st.selectbox("産業セクター", options=display_options)
        sector_en = unique_en_sectors[display_options.index(selected_jp)]
    else:
        sector_en = "Finance_insurance"
        st.selectbox("産業セクター", options=["データ未読み込み"])

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
            
            # --- A. AI予測 (エラー回避ロジックを強化) ---
            input_data = {
                "GrossApproval": float(gross), 
                "SBAGuaranteedApproval": float(sba),
                "InitialInterestRate": float(rate), 
                "TermInMonths": float(term),
                "NaicsSector": str(sector_en), 
                "ApprovalFiscalYear": 2024.0, 
                "Subprogram": "Guaranty",
                "FixedOrVariableInterestInd": "V", 
                "CongressionalDistrict": 10.0,
                "BusinessType": "CORPORATION", 
                "BusinessAge": "Existing or more than 2 years old",
                "RevolverStatus": 0.0, 
                "JobsSupported": 5.0, 
                "CollateralInd": str(collateral_val)
            }
            
            # DataFrameを作成し、モデルが期待する全列を確保（不足分は0埋め）
            input_df = pd.DataFrame([input_data])
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            
            # 列の並び順を学習時と完全に一致させる (index out of range 対策)
            input_df = input_df[expected_features]
            
            # カテゴリカル変数のインデックスを再取得
            cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
            
            # 予測の実行
            raw_proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

            # --- B. 類似事例検索 (対数距離補正) ---
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 100: search_pool = train_df.copy()
            
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
            
            nn = NearestNeighbors(n_neighbors=min(100, len(search_pool)))
            nn.fit(train_scaled)
            _, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy()
            risk_pct = similar_cases['LoanStatus'].mean() * 100
            def_count = int(similar_cases['LoanStatus'].sum())

            # --- C. 数値正常化ロジック (金融実務最適化) ---
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (np.log1p(term) - np.log1p(dynamic_ceil))) * 2.0 if term > dynamic_ceil else 0.0

            gross_risk = 0.0
            sba_bonus_flag = (current_sba_ratio >= 0.80)
            if gross >= 1000000: gross_risk = 0.40 + (gross - 1000000) / 1000000
            elif gross > 500000: gross_risk = ((gross - 500000) // 100000) * 0.04
            if sba_bonus_flag: gross_risk *= 0.5

            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3 + (0.1 if rate > 20.0 else 0)
            
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            
            combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            # 5. 表示ロジック
            if app_mode == "総合報告 (表面)":
                st.subheader("🏁 総合審査報告書")
                st.write("### 🔍 実務者への重点確認事項")
                
                if gross >= 1000000:
                    status = "危険"
                    st.error("🚨 **【最重要精査案件】** 融資額が $1M を超過。役員承認が必須。")
                elif gross >= 500000 and rate >= 20.0 and not sba_bonus_flag:
                    status = "注意"
                    st.error("💀 **【複合リスク】** 高額かつ高金利。80%以上の保証がないため警戒。")
                else:
                    status = "安全" if final_expected_success > 92 else "注意" if final_expected_success > 75 else "危険"

                if sba_bonus_flag:
                    st.success(f"🛡️ **【保全インセンティブ適用】** 保証率80%超により高額融資リスクを50%軽減。")
                if 500000 <= gross < 1000000:
                    st.info(f"📂 **【中規模案件】** 50万ドル超。リスク加重適用中。")
                if term > dynamic_ceil:
                    st.warning(f"⏳ **【期間超過】** 適正上限（{int(dynamic_ceil)}ヶ月）を超過。")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                    reasons = []
                    if gross >= 1000000: reasons.append("・100万ドル超の高額融資")
                    if rate >= 20.0: reasons.append("・20%超の高金利")
                    if term > dynamic_ceil: reasons.append("・返済期間の超過")

                    if status == "安全": st.success("総合判定: ✅ 安全")
                    elif status == "注意":
                        st.warning("総合判定: ⚠️ 注意")
                        for r in reasons: st.caption(f":orange[{r}]")
                    else:
                        st.error("総合判定: 🚨 危険 (要精査)")
                        for r in reasons: st.caption(f":red[{r}]")

                with c2:
                    st.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                    st.markdown(f"🔍 うち不履行事例: **{def_count}件**")
                with c3:
                    st.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

                st.write("### 💡 審査改善へのアクション案")
                with st.expander("アドバイスの詳細を確認する", expanded=True):
                    advice = []
                    if gross >= 1000000: advice.append("⚠️ **金額の再検討**: 可能であれば分割融資または担保の積み増しを。")
                    if term > dynamic_ceil: advice.append(f"✅ **期間の最適化**: {int(dynamic_ceil)}ヶ月以下への短縮を推奨。")
                    if current_sba_ratio < 0.80: advice.append("✅ **保証枠の拡大**: 80%以上に引き上げるとリスク加重が半減します。")
                    if not advice: st.write("✨ 現在の条件は論理的に非常に安定しています。")
                    else:
                        for a in advice: st.write(a)

                st.divider()
                st.write("### ⚖️ 判断に影響した主要要素")
                importances = model.get_feature_importance()
                imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
               # 重要度のカスタム調整（期間の影響を抑え、融資額を高める）
                imp_df.loc[imp_df['項目'] == 'TermInMonths', 'raw'] *= 0.23
                imp_df.loc[imp_df['項目'] == 'GrossApproval', 'raw'] *= 1.7
                imp_df.loc[imp_df['項目'] == 'SBAGuaranteedApproval', 'raw'] *= 0.8
                imp_df.loc[imp_df['項目'] == 'NaicsSector', 'raw'] *= 0.5
                imp_df.loc[imp_df['項目'] == 'InitialInterestRate', 'raw'] *= 0.9
                imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
                display_imp = imp_df[imp_df['項目名'] != "その他"].groupby('項目名')['raw'].sum().reset_index()
                display_imp['影響度(%)'] = (display_imp['raw'] / display_imp['raw'].sum() * 100).round(1)
                st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

            # --- 類似事例の復活 ---
                st.divider()
                st.write("### 👥 条件が近い過去の事例（上位5件）")
                display_similar = similar_cases.head(5).copy()
                display_similar['状況'] = display_similar['LoanStatus'].map({0: "✅ 完済", 1: "❌ 不履行"})
                display_similar = display_similar.rename(columns=name_map)
                st.dataframe(display_similar[[name_map[c] for c in ["GrossApproval", "InitialInterestRate", "TermInMonths"]] + ["状況"]], use_container_width=True)

                st.divider()

            else:
                # 高度解析 (裏面)
                st.header("🔬 高度数理エビデンス解析")
                
                st.write("#### ⚖️ AIの判断根拠 (SHAP Waterfall)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                shap_values.values = -shap_values.values
                shap_values.feature_names = [name_map.get(n, n) for n in expected_features]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                set_japanese_font()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(plt.gcf())
                
                st.divider()

                st.write("#### 📉 理論的倒産距離 (Merton Model)")
                vol = st.slider("想定資産ボラティリティ (%)", 10, 100, 30) / 100
                asset = float(gross) * 1.5
                t_m = float(term) / 12
                dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("倒産距離 (DD)", f"{dd:.2f}")
                    st.metric("デフォルト確率 (EDF)", f"{stats.norm.cdf(-dd)*100:.2f} %")
                with col_m2:
                    x = np.linspace(-4, 4, 100)
                    y = stats.norm.pdf(x, 0, 1)
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    ax2.plot(x, y, color="gray")
                    ax2.fill_between(x, y, where=(x < -dd), color='red', alpha=0.5)
                    ax2.axvline(-dd, color='red', linestyle='--')
                    st.pyplot(fig2)
                    # --- デフォルト確率と倒産距離の説明 ---
                with st.expander("📚 専門用語の解説：デフォルト確率と倒産距離", expanded=True):
                    st.write("""
                    **1. 倒産距離 (Distance to Default: DD)**
                    企業の資産価値が、負債（融資額）の支払境界線からどれだけ離れているかを「標準偏差」の単位で表したものです。
                    - 数値が高いほど安全です。一般的に **2.0以上** が優良な基準とされます。
                    - 資産のボラティリティ（変動幅）が大きいほど、DDは短くなり（危険）、倒産リスクが高まります。

                    **2. デフォルト確率 (Expected Default Frequency: EDF)**
                    マートン・モデルに基づき、将来的に企業の資産価値が負債額を下回る確率を算出したものです。
                    - 上図の赤色の領域がEDFに該当します。
                    - AI予測値とこのEDFを比較することで、統計的な観点とAI的な観点の両方からリスクを評価できます。
                    """)

                st.divider()

                st.write("#### 🧪 金利感度シミュレーション")
                sim_rates = np.linspace(5.0, 30.0, 15)
                sim_probs = [100 * (1 - model.predict_proba(Pool(input_df.assign(InitialInterestRate=r), cat_features=cat_idx))[0][1]) for r in sim_rates]
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(sim_rates, sim_probs, '-o', color="#0078D4", linewidth=2)
                ax3.axvline(x=rate, color='red', linestyle='--')
                st.pyplot(fig3)

        except Exception as e:
            st.error(f"分析エラー: {e}")
