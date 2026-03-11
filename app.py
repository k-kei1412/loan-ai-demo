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

# --- 文字化け・日本語フォント対策 ---
def set_japanese_font():
    # 利用可能な日本語フォントを探す
    font_list = [f.name for f in fm.fontManager.ttflist]
    target_fonts = ['MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
    for f in target_fonts:
        if f in font_list:
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

# 表示用名称マップ
name_map = {
    "TermInMonths": "返済期間", 
    "GrossApproval": "融資額", 
    "InitialInterestRate": "金利", 
    "NaicsSector": "業界セクター", 
    "SBAGuaranteedApproval": "保証額",
    "CollateralInd": "担保有無"
}

# 3. 業界セクター翻訳ロジック
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

# 4. 共通分析ロジック
if not train_df.empty:
    current_sba_ratio = sba / gross if gross > 0 else 0
    
    # 予測用データフレーム作成
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
    
    # AI予測
    raw_proba = model.predict_proba(Pool(input_df, cat_features=cat_idx))[0][1]

    # --- 数値正常化 (対数変換により「期間」の影響をマイルド化) ---
    strict_proba = np.clip(raw_proba, 0.01, 0.99)
    # 期間ペナルティの計算に対数を導入
    dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
    term_gap = max(0.0, (np.log1p(term) - np.log1p(dynamic_ceil))) * 2.0 if term > dynamic_ceil else 0.0

    # リスク計算
    sba_bonus_flag = (current_sba_ratio >= 0.80)
    gross_risk = 0.40 + (gross - 1000000) / 1000000 if gross >= 1000000 else ((gross - 500000) // 100000) * 0.04 if gross > 500000 else 0.0
    if sba_bonus_flag: gross_risk *= 0.5
    rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3

    # 類似事例検索 (対数距離を考慮)
    search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
    if len(search_pool) < 100: search_pool = train_df.copy()
    search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
    train_num = search_pool[search_features].fillna(0).copy()
    train_num["TermInMonths"] = np.log1p(train_num["TermInMonths"]) # 検索も対数で

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
    
    # 統合評価
    base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
    sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
    combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
    final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

    # --- 表面：総合報告 ---
    if app_mode == "総合報告 (表面)":
        st.subheader("🏁 総合審査報告書")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
        with c2: st.metric("実績事故率", f"{risk_pct:.1f} %")
        with c3: st.metric("完済期待値", f"{final_expected_success:.1f} %")

        st.divider()
        st.write("### ⚖️ 判断に影響した主要要素")
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
        # ここで独自の重み付け調整を適用
        imp_df.loc[imp_df['項目'] == 'TermInMonths', 'raw'] *= 0.23 # 期間の影響を圧縮
        imp_df.loc[imp_df['項目'] == 'GrossApproval', 'raw'] *= 1.7
        
        imp_df['項目名'] = imp_df['項目'].map(lambda x: name_map.get(x, "その他"))
        display_imp = imp_df[imp_df['項目名'] != "その他"].groupby('項目名')['raw'].sum().reset_index()
        display_imp['影響度(%)'] = (display_imp['raw'] / display_imp['raw'].sum() * 100).round(1)
        st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

    # --- 裏面：高度解析 (SHAP waterfall) ---
    else:
        st.header("🔬 高度数理エビデンス解析")
        st.write("#### ⚖️ AIの判断根拠 (SHAP Waterfall)")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        
        # 完済可能性の方向に変換 (デフォルトリスクの逆)
        shap_values.values = -shap_values.values 
        # 特徴量名を日本語に置換
        shap_values.feature_names = [name_map.get(n, n) for n in expected_features]
        
        # グラフ描画
        set_japanese_font()
        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title("完済可能性への寄与分析", fontsize=12)
        st.pyplot(plt.gcf(), clear_figure=True)

        # Merton Model (ブラックショールズの知見)
        st.divider()
        st.write("#### 📉 理論的倒産距離 (Merton Model DD)")
        # ... (既存のMerton Modelロジック) ...
        vol = 0.30
        asset_val = float(gross) * 1.5
        t_m = float(term) / 12
        dd = (np.log(asset_val / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
        st.metric("倒産距離 (DD)", f"{dd:.2f}")

else:
    st.error("学習データが見つかりません。")
